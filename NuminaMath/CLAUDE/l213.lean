import Mathlib

namespace NUMINAMATH_CALUDE_function_minimum_and_inequality_l213_21354

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |5 - x|

-- State the theorem
theorem function_minimum_and_inequality :
  ∃ (m : ℝ), 
    (∀ x, f x ≥ m) ∧ 
    (∃ x, f x = m) ∧
    m = 9/2 ∧
    ∀ (a b : ℝ), a ≥ 0 → b ≥ 0 → a + b = (2/3) * m → 
      1 / (a + 1) + 1 / (b + 2) ≥ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_and_inequality_l213_21354


namespace NUMINAMATH_CALUDE_log_inequality_l213_21372

theorem log_inequality : ∃ (a b c : ℝ), 
  a = Real.log 2 / Real.log 5 ∧ 
  b = Real.log 3 / Real.log 8 ∧ 
  c = (1 : ℝ) / 2 ∧ 
  a < c ∧ c < b :=
sorry

end NUMINAMATH_CALUDE_log_inequality_l213_21372


namespace NUMINAMATH_CALUDE_square_distance_sum_l213_21390

theorem square_distance_sum (s : Real) (h : s = 4) : 
  let midpoint_distance := 2 * s / 2
  let diagonal_distance := s * Real.sqrt 2
  let side_distance := s
  2 * midpoint_distance + 2 * Real.sqrt (midpoint_distance^2 + (s/2)^2) + diagonal_distance + side_distance = 10 + 4 * Real.sqrt 5 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_distance_sum_l213_21390


namespace NUMINAMATH_CALUDE_encoded_xyz_value_l213_21314

/-- Represents a digit in the base-6 encoding system -/
inductive Digit : Type
| U | V | W | X | Y | Z

/-- Converts a Digit to its corresponding natural number value -/
def digit_to_nat (d : Digit) : ℕ :=
  match d with
  | Digit.U => 0
  | Digit.V => 1
  | Digit.W => 2
  | Digit.X => 3
  | Digit.Y => 4
  | Digit.Z => 5

/-- Represents a three-digit number in the base-6 encoding system -/
structure EncodedNumber :=
  (hundreds : Digit)
  (tens : Digit)
  (ones : Digit)

/-- Converts an EncodedNumber to its base-10 value -/
def to_base_10 (n : EncodedNumber) : ℕ :=
  36 * (digit_to_nat n.hundreds) + 6 * (digit_to_nat n.tens) + (digit_to_nat n.ones)

/-- The theorem to be proved -/
theorem encoded_xyz_value :
  ∀ (v x y z : Digit),
    v ≠ x → v ≠ y → v ≠ z → x ≠ y → x ≠ z → y ≠ z →
    to_base_10 (EncodedNumber.mk v x z) + 1 = to_base_10 (EncodedNumber.mk v x y) →
    to_base_10 (EncodedNumber.mk v x y) + 1 = to_base_10 (EncodedNumber.mk v v y) →
    to_base_10 (EncodedNumber.mk x y z) = 184 :=
sorry

end NUMINAMATH_CALUDE_encoded_xyz_value_l213_21314


namespace NUMINAMATH_CALUDE_arithmetic_expressions_l213_21392

theorem arithmetic_expressions :
  let expr1 := (3.6 - 0.8) * (1.8 + 2.05)
  let expr2 := (34.28 / 2) - (16.2 / 4)
  (expr1 = (3.6 - 0.8) * (1.8 + 2.05)) ∧
  (expr2 = (34.28 / 2) - (16.2 / 4)) := by sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_l213_21392


namespace NUMINAMATH_CALUDE_intersection_line_slope_l213_21304

/-- Given two circles in the plane, this theorem states that the slope of the line
passing through their intersection points is -1/3. -/
theorem intersection_line_slope (x y : ℝ) :
  (x^2 + y^2 - 6*x + 4*y - 8 = 0) ∧ 
  (x^2 + y^2 - 8*x - 2*y + 10 = 0) →
  (∃ m b : ℝ, y = m*x + b ∧ m = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l213_21304


namespace NUMINAMATH_CALUDE_negative_irrational_less_than_neg_three_l213_21355

theorem negative_irrational_less_than_neg_three :
  ∃ x : ℝ, x < -3 ∧ Irrational x ∧ x < 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_negative_irrational_less_than_neg_three_l213_21355


namespace NUMINAMATH_CALUDE_root_reciprocal_sum_l213_21393

theorem root_reciprocal_sum (m : ℝ) : 
  (∃ α β : ℝ, α^2 - 2*(m+1)*α + m + 4 = 0 ∧ 
              β^2 - 2*(m+1)*β + m + 4 = 0 ∧ 
              α ≠ β ∧
              1/α + 1/β = 1) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_root_reciprocal_sum_l213_21393


namespace NUMINAMATH_CALUDE_gemma_pizza_payment_l213_21382

/-- The amount of money Gemma gave for her pizza order -/
def amount_given (num_pizzas : ℕ) (price_per_pizza : ℕ) (tip : ℕ) (change : ℕ) : ℕ :=
  num_pizzas * price_per_pizza + tip + change

/-- Proof that Gemma gave $50 for her pizza order -/
theorem gemma_pizza_payment :
  amount_given 4 10 5 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_gemma_pizza_payment_l213_21382


namespace NUMINAMATH_CALUDE_smallest_integer_solution_smallest_integer_solution_exists_l213_21362

theorem smallest_integer_solution (x : ℤ) : 
  (7 - 3 * x > 22) ∧ (x < 5) → x ≥ -6 :=
by
  sorry

theorem smallest_integer_solution_exists : 
  ∃ x : ℤ, (7 - 3 * x > 22) ∧ (x < 5) ∧ (x = -6) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_smallest_integer_solution_exists_l213_21362


namespace NUMINAMATH_CALUDE_horizontal_figure_area_l213_21350

/-- Represents a horizontally placed figure with specific properties -/
structure HorizontalFigure where
  /-- The oblique section diagram is an isosceles trapezoid -/
  is_isosceles_trapezoid : Bool
  /-- The base angle of the trapezoid is 45° -/
  base_angle : ℝ
  /-- The length of the legs of the trapezoid -/
  leg_length : ℝ
  /-- The length of the upper base of the trapezoid -/
  upper_base_length : ℝ

/-- Calculates the area of the original plane figure -/
def area (fig : HorizontalFigure) : ℝ :=
  sorry

/-- Theorem stating the area of the original plane figure -/
theorem horizontal_figure_area (fig : HorizontalFigure) 
  (h1 : fig.is_isosceles_trapezoid = true)
  (h2 : fig.base_angle = π / 4)
  (h3 : fig.leg_length = 1)
  (h4 : fig.upper_base_length = 1) :
  area fig = 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_horizontal_figure_area_l213_21350


namespace NUMINAMATH_CALUDE_haley_jason_difference_l213_21336

/-- The number of necklaces Haley has -/
def haley_necklaces : ℕ := 25

/-- The difference between Haley's and Josh's necklaces -/
def haley_josh_diff : ℕ := 15

/-- Represents the relationship between Josh's and Jason's necklaces -/
def josh_jason_ratio : ℚ := 1/2

/-- The number of necklaces Josh has -/
def josh_necklaces : ℕ := haley_necklaces - haley_josh_diff

/-- The number of necklaces Jason has -/
def jason_necklaces : ℕ := (2 * josh_necklaces)

theorem haley_jason_difference : haley_necklaces - jason_necklaces = 5 := by
  sorry

end NUMINAMATH_CALUDE_haley_jason_difference_l213_21336


namespace NUMINAMATH_CALUDE_race_time_A_l213_21370

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  runnerA : Runner
  runnerB : Runner
  timeDiff : ℝ
  distanceDiff : ℝ

/-- The main theorem that proves the race time for runner A -/
theorem race_time_A (race : Race) (h1 : race.distance = 1000) 
    (h2 : race.timeDiff = 10) (h3 : race.distanceDiff = 25) : 
    race.distance / race.runnerA.speed = 390 := by
  sorry

#check race_time_A

end NUMINAMATH_CALUDE_race_time_A_l213_21370


namespace NUMINAMATH_CALUDE_find_m_l213_21307

theorem find_m (x₁ x₂ m : ℝ) 
  (h1 : x₁^2 - 3*x₁ + m = 0) 
  (h2 : x₂^2 - 3*x₂ + m = 0)
  (h3 : x₁ + x₂ - x₁*x₂ = 1) : 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_find_m_l213_21307


namespace NUMINAMATH_CALUDE_concert_songs_theorem_l213_21328

/-- Represents the number of songs sung by each girl -/
structure SongCount where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ
  lucy : ℕ

/-- The total number of songs sung by the trios -/
def total_songs (s : SongCount) : ℕ :=
  (s.mary + s.alina + s.tina + s.hanna + s.lucy) / 3

/-- The conditions given in the problem -/
def satisfies_conditions (s : SongCount) : Prop :=
  s.hanna = 9 ∧
  s.lucy = 5 ∧
  s.mary > s.lucy ∧ s.mary < s.hanna ∧
  s.alina > s.lucy ∧ s.alina < s.hanna ∧
  s.tina > s.lucy ∧ s.tina < s.hanna

theorem concert_songs_theorem (s : SongCount) :
  satisfies_conditions s → total_songs s = 11 := by
  sorry

end NUMINAMATH_CALUDE_concert_songs_theorem_l213_21328


namespace NUMINAMATH_CALUDE_base4_to_base10_3201_l213_21398

/-- Converts a base 4 number to base 10 -/
def base4_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base 4 representation of the number -/
def base4_num : List Nat := [1, 0, 2, 3]

theorem base4_to_base10_3201 :
  base4_to_base10 base4_num = 225 := by
  sorry

end NUMINAMATH_CALUDE_base4_to_base10_3201_l213_21398


namespace NUMINAMATH_CALUDE_arithmetic_expressions_l213_21351

theorem arithmetic_expressions : 
  ((-8) - (-7) - |(-3)| = -4) ∧ 
  (-2^2 + 3 * (-1)^2019 - 9 / (-3) = 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expressions_l213_21351


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l213_21340

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l213_21340


namespace NUMINAMATH_CALUDE_bike_rides_total_l213_21375

/-- The number of times Billy rode his bike -/
def billy_rides : ℕ := 17

/-- The number of times John rode his bike -/
def john_rides : ℕ := 2 * billy_rides

/-- The number of times their mother rode her bike -/
def mother_rides : ℕ := john_rides + 10

/-- The total number of times they rode their bikes -/
def total_rides : ℕ := billy_rides + john_rides + mother_rides

theorem bike_rides_total : total_rides = 95 := by
  sorry

end NUMINAMATH_CALUDE_bike_rides_total_l213_21375


namespace NUMINAMATH_CALUDE_success_arrangements_l213_21312

/-- The number of permutations of a multiset -/
def multiset_permutations (n : ℕ) (repeats : List ℕ) : ℕ :=
  Nat.factorial n / (repeats.map Nat.factorial).prod

/-- The number of ways to arrange the letters of SUCCESS -/
theorem success_arrangements : multiset_permutations 7 [3, 2] = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_arrangements_l213_21312


namespace NUMINAMATH_CALUDE_solve_linear_equation_l213_21397

theorem solve_linear_equation (x : ℝ) : 3*x - 5*x + 8*x = 240 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l213_21397


namespace NUMINAMATH_CALUDE_square_area_adjacent_vertices_l213_21308

/-- The area of a square with adjacent vertices at (-2,3) and (4,3) is 36. -/
theorem square_area_adjacent_vertices : 
  let p1 : ℝ × ℝ := (-2, 3)
  let p2 : ℝ × ℝ := (4, 3)
  let distance := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := distance^2
  area = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_adjacent_vertices_l213_21308


namespace NUMINAMATH_CALUDE_units_digit_product_l213_21331

theorem units_digit_product (a b c : ℕ) : 
  a^2010 * b^1004 * c^1002 ≡ 0 [MOD 10] :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l213_21331


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l213_21360

theorem quadratic_minimum_value (c : ℝ) : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, -x^2 - 2*x + c ≥ -5) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, -x^2 - 2*x + c = -5) → 
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l213_21360


namespace NUMINAMATH_CALUDE_ce_length_l213_21387

/-- Given a triangle ABC, this function returns true if the triangle is right-angled -/
def is_right_triangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Given three points A, B, C, this function returns the measure of angle ABC in degrees -/
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given two points A and B, this function returns the distance between them -/
def distance (A B : ℝ × ℝ) : ℝ := sorry

theorem ce_length (A B C D E : ℝ × ℝ) 
  (h1 : is_right_triangle A B E)
  (h2 : is_right_triangle B C E)
  (h3 : is_right_triangle C D E)
  (h4 : angle_measure A E B = 60)
  (h5 : angle_measure B E C = 60)
  (h6 : angle_measure C E D = 60)
  (h7 : distance A E = 36) :
  distance C E = 9 := by sorry

end NUMINAMATH_CALUDE_ce_length_l213_21387


namespace NUMINAMATH_CALUDE_total_length_of_remaining_segments_l213_21373

/-- A figure with perpendicular adjacent sides -/
structure PerpendicularFigure where
  top_segments : List ℝ
  bottom_segment : ℝ
  left_segment : ℝ
  right_segment : ℝ

/-- The remaining figure after removing six sides -/
def RemainingFigure (f : PerpendicularFigure) : PerpendicularFigure :=
  { top_segments := [1],
    bottom_segment := f.bottom_segment,
    left_segment := f.left_segment,
    right_segment := 9 }

theorem total_length_of_remaining_segments (f : PerpendicularFigure)
  (h1 : f.top_segments = [3, 1, 1])
  (h2 : f.left_segment = 10)
  (h3 : f.bottom_segment = f.top_segments.sum)
  : (RemainingFigure f).top_segments.sum + 
    (RemainingFigure f).bottom_segment + 
    (RemainingFigure f).left_segment + 
    (RemainingFigure f).right_segment = 25 := by
  sorry


end NUMINAMATH_CALUDE_total_length_of_remaining_segments_l213_21373


namespace NUMINAMATH_CALUDE_set_equality_l213_21300

def M : Set ℝ := {x | x^2 - 2012*x - 2013 > 0}
def N (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem set_equality (a b : ℝ) : 
  M ∪ N a b = Set.univ ∧ 
  M ∩ N a b = Set.Ioo 2013 2014 →
  a = -2013 ∧ b = -2014 := by
sorry

end NUMINAMATH_CALUDE_set_equality_l213_21300


namespace NUMINAMATH_CALUDE_justin_bought_two_striped_jerseys_l213_21313

/-- The number of striped jerseys Justin bought -/
def num_striped_jerseys : ℕ := 2

/-- The cost of each long-sleeved jersey -/
def long_sleeve_cost : ℕ := 15

/-- The number of long-sleeved jerseys Justin bought -/
def num_long_sleeve : ℕ := 4

/-- The cost of each striped jersey before discount -/
def striped_cost : ℕ := 10

/-- The discount applied to each striped jersey after the first one -/
def striped_discount : ℕ := 2

/-- The total amount Justin spent -/
def total_spent : ℕ := 80

/-- Theorem stating that Justin bought 2 striped jerseys given the conditions -/
theorem justin_bought_two_striped_jerseys :
  num_long_sleeve * long_sleeve_cost +
  striped_cost +
  (num_striped_jerseys - 1) * (striped_cost - striped_discount) =
  total_spent :=
sorry

end NUMINAMATH_CALUDE_justin_bought_two_striped_jerseys_l213_21313


namespace NUMINAMATH_CALUDE_median_salary_proof_l213_21399

/-- Represents a position in the company with its count and salary -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions -/
def medianSalary (positions : List Position) : Nat :=
  sorry

theorem median_salary_proof (positions : List Position) :
  positions = [
    ⟨"CEO", 1, 140000⟩,
    ⟨"Senior Vice-President", 4, 95000⟩,
    ⟨"Manager", 12, 80000⟩,
    ⟨"Team Leader", 8, 55000⟩,
    ⟨"Office Assistant", 38, 25000⟩
  ] →
  (positions.map (λ p => p.count)).sum = 63 →
  medianSalary positions = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_proof_l213_21399


namespace NUMINAMATH_CALUDE_sector_max_area_l213_21317

/-- Given a sector with perimeter 60 cm, its maximum area is 225 cm² -/
theorem sector_max_area (r : ℝ) (l : ℝ) (S : ℝ → ℝ) :
  (0 < r) → (r < 30) →
  (l + 2 * r = 60) →
  (S = λ r => (1 / 2) * l * r) →
  (∀ r', S r' ≤ 225) ∧ (∃ r', S r' = 225) :=
sorry

end NUMINAMATH_CALUDE_sector_max_area_l213_21317


namespace NUMINAMATH_CALUDE_maria_sheets_problem_l213_21319

/-- The number of sheets in Maria's desk -/
def sheets_in_desk : ℕ := sorry

/-- The number of sheets in Maria's backpack -/
def sheets_in_backpack : ℕ := sorry

/-- The total number of sheets Maria has -/
def total_sheets : ℕ := 91

theorem maria_sheets_problem :
  (sheets_in_backpack = sheets_in_desk + 41) →
  (total_sheets = sheets_in_desk + sheets_in_backpack) →
  sheets_in_desk = 25 := by sorry

end NUMINAMATH_CALUDE_maria_sheets_problem_l213_21319


namespace NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_infinitely_many_primes_5_mod_6_l213_21353

-- Part 1: Infinitely many primes congruent to 3 modulo 4
theorem infinitely_many_primes_3_mod_4 : 
  ∀ S : Finset Nat, ∃ p : Nat, p ∉ S ∧ Prime p ∧ p % 4 = 3 := by sorry

-- Part 2: Infinitely many primes congruent to 5 modulo 6
theorem infinitely_many_primes_5_mod_6 : 
  ∀ S : Finset Nat, ∃ p : Nat, p ∉ S ∧ Prime p ∧ p % 6 = 5 := by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_3_mod_4_infinitely_many_primes_5_mod_6_l213_21353


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l213_21322

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units is 48π square units. -/
theorem circumscribed_circle_area (s : ℝ) (h : s = 12) : 
  let R := s / Real.sqrt 3
  π * R^2 = 48 * π := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_l213_21322


namespace NUMINAMATH_CALUDE_train_length_calculation_l213_21367

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 180

/-- Represents the time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- Theorem stating that under given conditions, the train length is 1500 meters -/
theorem train_length_calculation (train_length platform_length : ℝ) 
  (h1 : train_length = platform_length) 
  (h2 : train_speed * (1000 / 60) * crossing_time = 2 * train_length) : 
  train_length = 1500 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l213_21367


namespace NUMINAMATH_CALUDE_remaining_area_calculation_l213_21388

theorem remaining_area_calculation : 
  let large_square_side : ℝ := 3
  let small_square_side : ℝ := 1
  let triangle_base : ℝ := 1
  let triangle_height : ℝ := 3
  large_square_side ^ 2 - (small_square_side ^ 2 + (triangle_base * triangle_height / 2)) = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_calculation_l213_21388


namespace NUMINAMATH_CALUDE_speed_ratio_l213_21377

-- Define the speeds of A and B
def speed_A : ℝ := sorry
def speed_B : ℝ := sorry

-- Define the initial position of B
def initial_B : ℝ := -800

-- Define the equidistant condition after 1 minute
def equidistant_1 : Prop :=
  speed_A = |initial_B + speed_B|

-- Define the equidistant condition after 7 minutes
def equidistant_7 : Prop :=
  7 * speed_A = |initial_B + 7 * speed_B|

-- Theorem stating the ratio of speeds
theorem speed_ratio :
  equidistant_1 → equidistant_7 → speed_A / speed_B = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_speed_ratio_l213_21377


namespace NUMINAMATH_CALUDE_system_solution_l213_21352

theorem system_solution : 
  ∃! (x y : ℚ), (4 * x - 3 * y = 2) ∧ (5 * x + 4 * y = 3) ∧ x = 17/31 ∧ y = 2/31 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l213_21352


namespace NUMINAMATH_CALUDE_ball_problem_l213_21380

/-- Given the conditions of the ball problem, prove that the number of red, yellow, and white balls is (45, 40, 75). -/
theorem ball_problem (red yellow white : ℕ) : 
  (red + yellow + white = 160) →
  (2 * red / 3 + 3 * yellow / 4 + 4 * white / 5 = 120) →
  (4 * red / 5 + 3 * yellow / 4 + 2 * white / 3 = 116) →
  (red = 45 ∧ yellow = 40 ∧ white = 75) := by
sorry

end NUMINAMATH_CALUDE_ball_problem_l213_21380


namespace NUMINAMATH_CALUDE_divisibility_puzzle_l213_21361

theorem divisibility_puzzle :
  ∃ N : ℕ, (N % 2 = 0) ∧ (N % 4 = 0) ∧ (N % 12 = 0) ∧ (N % 24 ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_divisibility_puzzle_l213_21361


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l213_21323

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∃ d : ℝ, 3 * a 1 + d = (1/2) * a 3 ∧ (1/2) * a 3 + d = 2 * a 2) →  -- Arithmetic sequence condition
  (∃ q : ℝ, ∀ n, a (n+1) = q * a n) →  -- Geometric sequence definition
  (a 8 + a 9) / (a 6 + a 7) = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l213_21323


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l213_21332

def U : Finset Nat := {1,2,3,4,5,6}
def A : Finset Nat := {1,3,4,6}
def B : Finset Nat := {2,4,5,6}

theorem intersection_complement_equal : A ∩ (U \ B) = {1,3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l213_21332


namespace NUMINAMATH_CALUDE_payment_difference_l213_21302

-- Define the pizza parameters
def total_slices : ℕ := 10
def plain_pizza_cost : ℚ := 12
def double_cheese_cost : ℚ := 4

-- Define the number of slices each person ate
def bob_double_cheese_slices : ℕ := 5
def bob_plain_slices : ℕ := 2
def cindy_plain_slices : ℕ := 3

-- Calculate the total cost of the pizza
def total_pizza_cost : ℚ := plain_pizza_cost + double_cheese_cost

-- Calculate the cost per slice
def cost_per_slice : ℚ := total_pizza_cost / total_slices

-- Calculate Bob's payment
def bob_payment : ℚ := cost_per_slice * (bob_double_cheese_slices + bob_plain_slices)

-- Calculate Cindy's payment
def cindy_payment : ℚ := cost_per_slice * cindy_plain_slices

-- State the theorem
theorem payment_difference : bob_payment - cindy_payment = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_payment_difference_l213_21302


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l213_21342

/-- Converts a list of bits (represented as Bools) to a natural number. -/
def bitsToNat (bits : List Bool) : Nat :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- The binary number 10101₂ -/
def num1 : List Bool := [true, false, true, false, true]

/-- The binary number 1011₂ -/
def num2 : List Bool := [true, false, true, true]

/-- The binary number 1110₂ -/
def num3 : List Bool := [true, true, true, false]

/-- The binary number 110001₂ -/
def num4 : List Bool := [true, true, false, false, false, true]

/-- The binary number 1101₂ -/
def num5 : List Bool := [true, true, false, true]

/-- The binary number 101100₂ (the expected result) -/
def result : List Bool := [true, false, true, true, false, false]

theorem binary_addition_subtraction :
  bitsToNat num1 + bitsToNat num2 + bitsToNat num3 + bitsToNat num4 - bitsToNat num5 = bitsToNat result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l213_21342


namespace NUMINAMATH_CALUDE_units_digit_base8_product_l213_21324

/-- The units digit of a number in base 8 -/
def unitsDigitBase8 (n : ℕ) : ℕ := n % 8

/-- The product of 348 and 76 -/
def product : ℕ := 348 * 76

theorem units_digit_base8_product : unitsDigitBase8 product = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_base8_product_l213_21324


namespace NUMINAMATH_CALUDE_four_digit_decimal_problem_l213_21371

theorem four_digit_decimal_problem :
  ∃ (x : ℕ), 
    (1000 ≤ x ∧ x < 10000) ∧
    ((x : ℝ) - (x : ℝ) / 10 = 2059.2 ∨ (x : ℝ) - (x : ℝ) / 100 = 2059.2) ∧
    (x = 2288 ∨ x = 2080) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_decimal_problem_l213_21371


namespace NUMINAMATH_CALUDE_tan_alpha_value_l213_21386

theorem tan_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.tan (2 * α) = Real.sin α / (2 + Real.cos α)) : 
  Real.tan α = -Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l213_21386


namespace NUMINAMATH_CALUDE_product_sum_fractions_l213_21327

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l213_21327


namespace NUMINAMATH_CALUDE_parabola_translation_l213_21305

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 1 0 0  -- y = x^2
  let translated := translate original 3 2
  y = translated.a * (x - 3)^2 + translated.b * (x - 3) + translated.c ↔
  y = (x - 3)^2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l213_21305


namespace NUMINAMATH_CALUDE_gold_per_hour_l213_21364

/-- Calculates the amount of gold coins found per hour during a scuba diving expedition. -/
theorem gold_per_hour (hours : ℕ) (chest_coins : ℕ) (num_bags : ℕ) : 
  hours > 0 → 
  chest_coins > 0 → 
  num_bags > 0 → 
  (chest_coins + num_bags * (chest_coins / 2)) / hours = 25 :=
by
  sorry

#check gold_per_hour 8 100 2

end NUMINAMATH_CALUDE_gold_per_hour_l213_21364


namespace NUMINAMATH_CALUDE_extremum_and_max_min_of_f_l213_21374

def f (x : ℝ) := x^3 + 4*x^2 - 11*x + 16

theorem extremum_and_max_min_of_f :
  (∃ (x : ℝ), f x = 10 ∧ ∀ y, |y - 1| < |x - 1| → f y ≠ 10) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ 18) ∧
  (∃ x ∈ Set.Icc 0 2, f x = 18) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≥ 10) ∧
  (∃ x ∈ Set.Icc 0 2, f x = 10) :=
sorry

end NUMINAMATH_CALUDE_extremum_and_max_min_of_f_l213_21374


namespace NUMINAMATH_CALUDE_f_2007_equals_negative_two_l213_21391

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2007_equals_negative_two (f : ℝ → ℝ) 
  (h1 : isEven f) 
  (h2 : ∀ x, f (2 + x) = f (2 - x)) 
  (h3 : f (-3) = -2) : 
  f 2007 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2007_equals_negative_two_l213_21391


namespace NUMINAMATH_CALUDE_work_completion_rate_l213_21348

/-- Given that A can finish a work in 18 days and B can do the same work in half the time taken by A,
    prove that A and B working together can finish 1/6 of the work in a day. -/
theorem work_completion_rate (days_A : ℕ) (days_B : ℕ) : 
  days_A = 18 →
  days_B = days_A / 2 →
  (1 : ℚ) / days_A + (1 : ℚ) / days_B = (1 : ℚ) / 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_rate_l213_21348


namespace NUMINAMATH_CALUDE_harvard_attendance_l213_21356

theorem harvard_attendance 
  (total_applicants : ℕ) 
  (acceptance_rate : ℚ) 
  (attendance_rate : ℚ) :
  total_applicants = 20000 →
  acceptance_rate = 5 / 100 →
  attendance_rate = 90 / 100 →
  (total_applicants : ℚ) * acceptance_rate * attendance_rate = 900 :=
by sorry

end NUMINAMATH_CALUDE_harvard_attendance_l213_21356


namespace NUMINAMATH_CALUDE_max_sum_squares_l213_21316

theorem max_sum_squares : ∃ (m n : ℕ), 
  1 ≤ m ∧ m ≤ 2005 ∧ 
  1 ≤ n ∧ n ≤ 2005 ∧ 
  (n^2 + 2*m*n - 2*m^2)^2 = 1 ∧ 
  m^2 + n^2 = 702036 ∧ 
  ∀ (m' n' : ℕ), 
    1 ≤ m' ∧ m' ≤ 2005 → 
    1 ≤ n' ∧ n' ≤ 2005 → 
    (n'^2 + 2*m'*n' - 2*m'^2)^2 = 1 → 
    m'^2 + n'^2 ≤ 702036 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squares_l213_21316


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l213_21379

/-- Given a polynomial division, prove that the remainder is 1 -/
theorem polynomial_division_remainder : 
  let P (z : ℝ) := 4 * z^3 - 5 * z^2 - 17 * z + 4
  let D (z : ℝ) := 4 * z + 6
  let Q (z : ℝ) := z^2 - 4 * z + 1/2
  ∃ (R : ℝ → ℝ), (∀ z, P z = D z * Q z + R z) ∧ (∀ z, R z = 1) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l213_21379


namespace NUMINAMATH_CALUDE_cars_in_parking_lot_l213_21396

theorem cars_in_parking_lot (total_wheels : ℕ) (wheels_per_car : ℕ) (h1 : total_wheels = 48) (h2 : wheels_per_car = 4) :
  total_wheels / wheels_per_car = 12 := by
  sorry

end NUMINAMATH_CALUDE_cars_in_parking_lot_l213_21396


namespace NUMINAMATH_CALUDE_lindas_wallet_l213_21346

theorem lindas_wallet (total_amount : ℕ) (total_bills : ℕ) (five_dollar_bills : ℕ) (ten_dollar_bills : ℕ) :
  total_amount = 100 →
  total_bills = 15 →
  five_dollar_bills + ten_dollar_bills = total_bills →
  5 * five_dollar_bills + 10 * ten_dollar_bills = total_amount →
  five_dollar_bills = 10 :=
by sorry

end NUMINAMATH_CALUDE_lindas_wallet_l213_21346


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l213_21335

def is_root (x : ℝ) : Prop := x^2 - 5*x + 6 = 0

theorem isosceles_triangle_perimeter : 
  ∀ (leg : ℝ), 
  is_root leg → 
  leg > 0 → 
  leg + leg > 4 → 
  leg + leg + 4 = 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l213_21335


namespace NUMINAMATH_CALUDE_function_inequality_implies_k_range_l213_21345

theorem function_inequality_implies_k_range (k : ℝ) :
  (∀ x : ℝ, (k * x + 1 > 0) ∨ (x^2 - 1 > 0)) →
  k ∈ Set.Ioo (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_k_range_l213_21345


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l213_21334

theorem sum_of_coefficients_zero (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x, (1 - 4*x)^10 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                       a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁/2 + a₂/2^2 + a₃/2^3 + a₄/2^4 + a₅/2^5 + a₆/2^6 + a₇/2^7 + a₈/2^8 + a₉/2^9 + a₁₀/2^10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l213_21334


namespace NUMINAMATH_CALUDE_unique_modular_equivalence_l213_21326

theorem unique_modular_equivalence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_equivalence_l213_21326


namespace NUMINAMATH_CALUDE_score_difference_l213_21369

theorem score_difference (score : ℕ) (h : score = 15) : 3 * score - 2 * score = 15 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l213_21369


namespace NUMINAMATH_CALUDE_rapid_advance_min_cost_l213_21301

/-- Represents a ride model with its capacity and price -/
structure RideModel where
  capacity : ℕ
  price : ℕ

/-- The minimum amount needed to spend on tickets for a group -/
def minTicketCost (model1 model2 : RideModel) (groupSize : ℕ) : ℕ :=
  sorry

theorem rapid_advance_min_cost :
  let model1 : RideModel := { capacity := 7, price := 65 }
  let model2 : RideModel := { capacity := 5, price := 50 }
  let groupSize : ℕ := 73
  minTicketCost model1 model2 groupSize = 685 := by sorry

end NUMINAMATH_CALUDE_rapid_advance_min_cost_l213_21301


namespace NUMINAMATH_CALUDE_car_speed_problem_l213_21368

theorem car_speed_problem (v : ℝ) : 
  (∀ (t : ℝ), t = 3 → (70 - v) * t = 60) → v = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l213_21368


namespace NUMINAMATH_CALUDE_grains_in_gray_areas_l213_21389

/-- Given two circles with equal total grains, prove that the sum of their non-overlapping parts is 61 grains -/
theorem grains_in_gray_areas (total_circle1 total_circle2 overlap : ℕ) 
  (h1 : total_circle1 = 110)
  (h2 : total_circle2 = 87)
  (h3 : overlap = 68)
  (h4 : total_circle1 = total_circle2) : 
  (total_circle1 - overlap) + (total_circle2 - overlap) = 61 := by
  sorry

#check grains_in_gray_areas

end NUMINAMATH_CALUDE_grains_in_gray_areas_l213_21389


namespace NUMINAMATH_CALUDE_perpendicular_slope_l213_21311

/-- The slope of a line perpendicular to a line passing through (2, 3) and (7, 8) is -1 -/
theorem perpendicular_slope : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (7, 8)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  (-1 : ℝ) * m = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l213_21311


namespace NUMINAMATH_CALUDE_inequality_solution_l213_21378

theorem inequality_solution (x : ℝ) :
  3 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 9 * x + 1 →
  x > (5 + Real.sqrt 29) / 2 ∧ x < 11 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l213_21378


namespace NUMINAMATH_CALUDE_license_plate_combinations_l213_21303

/-- The number of possible characters for each position in the license plate -/
def numCharOptions : ℕ := 26 + 10

/-- The length of the license plate -/
def plateLength : ℕ := 4

/-- The number of ways to position two identical characters in non-adjacent positions in a 4-character plate -/
def numPairPositions : ℕ := 3

/-- The number of ways to choose characters for the non-duplicate positions -/
def numNonDuplicateChoices : ℕ := numCharOptions * (numCharOptions - 1)

theorem license_plate_combinations :
  numPairPositions * numCharOptions * numNonDuplicateChoices = 136080 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l213_21303


namespace NUMINAMATH_CALUDE_middle_number_proof_l213_21306

theorem middle_number_proof (a b c : ℕ) 
  (h_order : a < b ∧ b < c)
  (h_sum1 : a + b = 15)
  (h_sum2 : a + c = 20)
  (h_sum3 : b + c = 25) :
  b = 10 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l213_21306


namespace NUMINAMATH_CALUDE_range_of_m_l213_21381

/-- Proposition p: The equation x^2+mx+1=0 has exactly two distinct negative roots -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  ∀ x : ℝ, x^2 + m*x + 1 = 0 ↔ (x = x₁ ∨ x = x₂)

/-- Proposition q: The inequality 3^x-m+1≤0 has a real solution -/
def q (m : ℝ) : Prop :=
  ∃ x : ℝ, 3^x - m + 1 ≤ 0

/-- The range of m given the conditions -/
theorem range_of_m :
  ∀ m : ℝ, (∃ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ (1 < m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l213_21381


namespace NUMINAMATH_CALUDE_cube_split_73_l213_21395

/-- The first "split number" of m^3 -/
def firstSplitNumber (m : ℕ) : ℕ := m^2 - m + 1

/-- Predicate to check if a number is one of the "split numbers" of m^3 -/
def isSplitNumber (m : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < m ∧ n = firstSplitNumber m + 2 * k

theorem cube_split_73 (m : ℕ) (h1 : m > 1) (h2 : isSplitNumber m 73) : m = 9 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_73_l213_21395


namespace NUMINAMATH_CALUDE_inequality_proof_l213_21315

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l213_21315


namespace NUMINAMATH_CALUDE_marshmallow_ratio_l213_21321

theorem marshmallow_ratio (joe_marshmallows : ℕ) (dad_marshmallows : ℕ) : 
  dad_marshmallows = 21 →
  (joe_marshmallows / 2 + dad_marshmallows / 3 = 49) →
  (joe_marshmallows : ℚ) / dad_marshmallows = 4 := by
sorry

end NUMINAMATH_CALUDE_marshmallow_ratio_l213_21321


namespace NUMINAMATH_CALUDE_equality_or_power_relation_l213_21344

theorem equality_or_power_relation (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : x^y = y^x) :
  x = y ∨ ∃ m : ℝ, m > 0 ∧ m ≠ 1 ∧ x = m^(1/(m-1)) ∧ y = m^(m/(m-1)) := by
  sorry

end NUMINAMATH_CALUDE_equality_or_power_relation_l213_21344


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l213_21333

theorem inequality_and_minimum_value 
  (a b m n : ℝ) (x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hm : m > 0) (hn : n > 0)
  (hx : 0 < x ∧ x < 1/2) : 
  (m^2 / a + n^2 / b ≥ (m + n)^2 / (a + b)) ∧
  (2 / x + 9 / (1 - 2*x) ≥ 25) ∧
  (∀ y, 0 < y ∧ y < 1/2 → 2 / y + 9 / (1 - 2*y) ≥ 2 / x + 9 / (1 - 2*x)) ∧
  (x = 1/5) := by
sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l213_21333


namespace NUMINAMATH_CALUDE_first_studio_students_l213_21384

theorem first_studio_students (total : ℕ) (second : ℕ) (third : ℕ) 
  (h1 : total = 376)
  (h2 : second = 135)
  (h3 : third = 131) :
  total - (second + third) = 110 := by
  sorry

end NUMINAMATH_CALUDE_first_studio_students_l213_21384


namespace NUMINAMATH_CALUDE_xy_sum_theorem_l213_21337

theorem xy_sum_theorem (x y : ℕ) (hx : x > 0) (hy : y > 0) (hx_lt_20 : x < 20) (hy_lt_20 : y < 20) 
  (h_eq : x + y + x * y = 99) : x + y = 23 ∨ x + y = 18 :=
sorry

end NUMINAMATH_CALUDE_xy_sum_theorem_l213_21337


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l213_21320

theorem necessary_not_sufficient_condition :
  ∃ (x : ℝ), x ≠ 0 ∧ ¬(|2*x + 5| ≥ 7) ∧
  ∀ (y : ℝ), |2*y + 5| ≥ 7 → y ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l213_21320


namespace NUMINAMATH_CALUDE_zeros_of_f_l213_21309

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -x - 4 else x^2 - 4

-- Theorem statement
theorem zeros_of_f :
  (∃ x : ℝ, f x = 0) ↔ (x = -4 ∨ x = 2) :=
sorry

end NUMINAMATH_CALUDE_zeros_of_f_l213_21309


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l213_21376

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 7 = 16)
  (h_a3 : a 3 = 1) :
  a 9 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l213_21376


namespace NUMINAMATH_CALUDE_quadratic_properties_l213_21357

def f (x : ℝ) := -x^2 + 2*x + 1

theorem quadratic_properties :
  (∀ x y : ℝ, f x ≤ f y → x = y ∨ (x < y ∧ f ((x + y) / 2) > f x) ∨ (y < x ∧ f ((x + y) / 2) > f x)) ∧
  (∃ x : ℝ, ∀ y : ℝ, f y ≤ f x) ∧
  (∃! x : ℝ, ∀ y : ℝ, f y ≤ f x) ∧
  (∀ x : ℝ, f x ≤ f 1) ∧
  f 1 = 2 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → -2 ≤ f x ∧ f x ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l213_21357


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l213_21358

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x, (1 + a) * x > 1 + a ↔ x < 1) → a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l213_21358


namespace NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l213_21363

theorem absolute_value_and_quadratic_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x - 3| = 4 ↔ x^2 + b*x + c = 0) →
  b = -6 ∧ c = -7 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_and_quadratic_equivalence_l213_21363


namespace NUMINAMATH_CALUDE_total_boys_count_l213_21394

theorem total_boys_count (average_all : ℝ) (average_passed : ℝ) (average_failed : ℝ) (passed_count : ℕ) :
  average_all = 37 →
  average_passed = 39 →
  average_failed = 15 →
  passed_count = 110 →
  ∃ (total_count : ℕ), 
    total_count = passed_count + (total_count - passed_count) ∧
    (average_all * total_count : ℝ) = average_passed * passed_count + average_failed * (total_count - passed_count) ∧
    total_count = 120 :=
by
  sorry

end NUMINAMATH_CALUDE_total_boys_count_l213_21394


namespace NUMINAMATH_CALUDE_perfect_squares_between_75_and_400_l213_21318

def count_perfect_squares (a b : ℕ) : ℕ :=
  (Nat.sqrt b - Nat.sqrt a + 1).max 0

theorem perfect_squares_between_75_and_400 :
  count_perfect_squares 75 400 = 12 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_between_75_and_400_l213_21318


namespace NUMINAMATH_CALUDE_range_of_m_l213_21310

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 3}

-- Theorem statement
theorem range_of_m (m : ℝ) : B m ⊆ A → m < -4 ∨ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l213_21310


namespace NUMINAMATH_CALUDE_problem_odometer_miles_l213_21325

/-- Represents a faulty odometer that skips certain digits --/
structure FaultyOdometer where
  skipped_digits : List Nat
  display : Nat

/-- Converts a faulty odometer reading to actual miles traveled --/
def actualMiles (o : FaultyOdometer) : Nat :=
  sorry

/-- The specific faulty odometer in the problem --/
def problemOdometer : FaultyOdometer :=
  { skipped_digits := [4, 7], display := 5006 }

/-- Theorem stating that the problemOdometer has traveled 1721 miles --/
theorem problem_odometer_miles :
  actualMiles problemOdometer = 1721 := by
  sorry

end NUMINAMATH_CALUDE_problem_odometer_miles_l213_21325


namespace NUMINAMATH_CALUDE_container_capacity_container_capacity_proof_l213_21347

theorem container_capacity : ℝ → Prop :=
  fun capacity =>
    capacity > 0 ∧
    0.4 * capacity + 28 = 0.75 * capacity →
    capacity = 80

-- The proof is omitted
theorem container_capacity_proof : ∃ (capacity : ℝ), container_capacity capacity :=
  sorry

end NUMINAMATH_CALUDE_container_capacity_container_capacity_proof_l213_21347


namespace NUMINAMATH_CALUDE_book_cost_price_l213_21339

/-- Proves that given a book sold for Rs 70 with a 40% profit rate, the cost price of the book is Rs 50. -/
theorem book_cost_price (selling_price : ℝ) (profit_rate : ℝ) 
  (h1 : selling_price = 70)
  (h2 : profit_rate = 0.4) :
  selling_price / (1 + profit_rate) = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l213_21339


namespace NUMINAMATH_CALUDE_solution_set_implies_a_and_b_solution_set_when_a_negative_l213_21338

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - (a - 2) * x - 2

-- Part 1
theorem solution_set_implies_a_and_b :
  ∀ a b : ℝ, (∀ x : ℝ, f a x ≤ b ↔ -2 ≤ x ∧ x ≤ 1) → a = 1 ∧ b = 0 := by sorry

-- Part 2
theorem solution_set_when_a_negative :
  ∀ a : ℝ, a < 0 →
    (∀ x : ℝ, f a x ≥ 0 ↔
      ((-2 < a ∧ a < 0 ∧ 1 ≤ x ∧ x ≤ -2/a) ∨
       (a = -2 ∧ x = 1) ∨
       (a < -2 ∧ -2/a ≤ x ∧ x ≤ 1))) := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_and_b_solution_set_when_a_negative_l213_21338


namespace NUMINAMATH_CALUDE_percentage_of_men_l213_21385

/-- The percentage of employees who are men, given picnic attendance data. -/
theorem percentage_of_men (men_attendance : Real) (women_attendance : Real) (total_attendance : Real)
  (h1 : men_attendance = 0.2)
  (h2 : women_attendance = 0.4)
  (h3 : total_attendance = 0.29000000000000004) :
  ∃ (men_percentage : Real),
    men_percentage * men_attendance + (1 - men_percentage) * women_attendance = total_attendance ∧
    men_percentage = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_men_l213_21385


namespace NUMINAMATH_CALUDE_weight_comparison_l213_21349

/-- Given the weights of Mildred, Carol, and Tom, prove statements about their combined weights -/
theorem weight_comparison (mildred_weight carol_weight tom_weight : ℕ) 
  (h1 : mildred_weight = 59)
  (h2 : carol_weight = 9)
  (h3 : tom_weight = 20) :
  let combined_weight := carol_weight + tom_weight
  (combined_weight = 29) ∧ 
  (mildred_weight = combined_weight + 30) := by
  sorry

end NUMINAMATH_CALUDE_weight_comparison_l213_21349


namespace NUMINAMATH_CALUDE_f_of_3_eq_19_l213_21329

/-- The function f(x) = 2x^2 + 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- Theorem: f(3) = 19 -/
theorem f_of_3_eq_19 : f 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_f_of_3_eq_19_l213_21329


namespace NUMINAMATH_CALUDE_dealer_profit_theorem_l213_21343

/-- Represents the pricing and discount strategy of a dealer -/
structure DealerStrategy where
  markup_percentage : ℝ
  discount_percentage : ℝ
  bulk_deal_articles_sold : ℕ
  bulk_deal_articles_cost : ℕ

/-- Calculates the profit percentage for a dealer given their strategy -/
def calculate_profit_percentage (strategy : DealerStrategy) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that the dealer's profit percentage is 80% under the given conditions -/
theorem dealer_profit_theorem (strategy : DealerStrategy) 
  (h1 : strategy.markup_percentage = 100)
  (h2 : strategy.discount_percentage = 10)
  (h3 : strategy.bulk_deal_articles_sold = 20)
  (h4 : strategy.bulk_deal_articles_cost = 15) :
  calculate_profit_percentage strategy = 80 := by
  sorry

end NUMINAMATH_CALUDE_dealer_profit_theorem_l213_21343


namespace NUMINAMATH_CALUDE_last_two_digits_of_floor_fraction_l213_21359

theorem last_two_digits_of_floor_fraction : ∃ n : ℕ, 
  n ≥ 10^62 - 3 * 10^31 + 8 ∧ 
  n < 10^62 - 3 * 10^31 + 9 ∧ 
  n % 100 = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_floor_fraction_l213_21359


namespace NUMINAMATH_CALUDE_binomial_product_equals_6720_l213_21365

theorem binomial_product_equals_6720 : Nat.choose 10 3 * Nat.choose 8 3 = 6720 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_equals_6720_l213_21365


namespace NUMINAMATH_CALUDE_only_B_on_line_l213_21330

-- Define the points
def A : ℝ × ℝ := (0, -2)
def B : ℝ × ℝ := (-2, 1)
def C : ℝ × ℝ := (0, 0)
def D : ℝ × ℝ := (2, -9)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - 3 * y + 7 = 0

-- Theorem statement
theorem only_B_on_line :
  line_equation B.1 B.2 ∧
  ¬line_equation A.1 A.2 ∧
  ¬line_equation C.1 C.2 ∧
  ¬line_equation D.1 D.2 := by
  sorry

end NUMINAMATH_CALUDE_only_B_on_line_l213_21330


namespace NUMINAMATH_CALUDE_inverse_one_implies_one_l213_21341

theorem inverse_one_implies_one (a : ℝ) (h : a ≠ 0) : a⁻¹ = (-1)^0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_one_implies_one_l213_21341


namespace NUMINAMATH_CALUDE_exists_non_prime_combination_l213_21366

-- Define a function to check if a number is prime
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number has distinct digits (excluding 7)
def hasDistinctDigitsNo7 (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = 9 ∧ 
  7 ∉ digits ∧
  digits.toFinset.card = 9

-- Define a function to check if any three-digit combination is prime
def anyThreeDigitsPrime (n : Nat) : Prop :=
  ∀ i j k, 0 ≤ i ∧ i < j ∧ j < k ∧ k < 9 →
    isPrime (100 * (n.digits 10).get ⟨i, by sorry⟩ + 
             10 * (n.digits 10).get ⟨j, by sorry⟩ + 
             (n.digits 10).get ⟨k, by sorry⟩)

-- The main theorem
theorem exists_non_prime_combination :
  ∃ n : Nat, hasDistinctDigitsNo7 n ∧ ¬(anyThreeDigitsPrime n) :=
sorry

end NUMINAMATH_CALUDE_exists_non_prime_combination_l213_21366


namespace NUMINAMATH_CALUDE_lcm_of_180_and_504_l213_21383

theorem lcm_of_180_and_504 : Nat.lcm 180 504 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_180_and_504_l213_21383
