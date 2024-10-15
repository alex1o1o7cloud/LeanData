import Mathlib

namespace NUMINAMATH_CALUDE_parabola_line_intersection_right_angle_l1239_123978

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in 2D space of the form y^2 = kx -/
structure Parabola where
  k : ℝ

def on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

def on_parabola (p : Point) (para : Parabola) : Prop :=
  p.y^2 = para.k * p.x

def right_angle (a b c : Point) : Prop :=
  (b.x - a.x) * (c.x - a.x) + (b.y - a.y) * (c.y - a.y) = 0

theorem parabola_line_intersection_right_angle :
  ∀ (l : Line) (para : Parabola) (a b c : Point),
    l.a = 1 ∧ l.b = -2 ∧ l.c = -1 →
    para.k = 4 →
    on_line a l ∧ on_parabola a para →
    on_line b l ∧ on_parabola b para →
    on_parabola c para →
    right_angle a c b →
    (c.x = 1 ∧ c.y = -2) ∨ (c.x = 9 ∧ c.y = -6) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_right_angle_l1239_123978


namespace NUMINAMATH_CALUDE_circle_area_equality_l1239_123942

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 12) (h₂ : r₂ = 25) (h₃ : r₃ = Real.sqrt 481) :
  π * r₃^2 = π * r₂^2 - π * r₁^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equality_l1239_123942


namespace NUMINAMATH_CALUDE_rectangle_length_l1239_123941

/-- Represents a rectangle with length, width, diagonal, and perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  diagonal : ℝ
  perimeter : ℝ

/-- Theorem: A rectangle with diagonal 17 cm and perimeter 46 cm has a length of 15 cm -/
theorem rectangle_length (r : Rectangle) 
  (h_diagonal : r.diagonal = 17)
  (h_perimeter : r.perimeter = 46)
  (h_perimeter_def : r.perimeter = 2 * (r.length + r.width))
  (h_diagonal_def : r.diagonal ^ 2 = r.length ^ 2 + r.width ^ 2) :
  r.length = 15 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_length_l1239_123941


namespace NUMINAMATH_CALUDE_division_and_power_equality_l1239_123979

theorem division_and_power_equality : ((-125) / (-25)) ^ 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_division_and_power_equality_l1239_123979


namespace NUMINAMATH_CALUDE_acute_triangle_side_range_l1239_123996

-- Define an acute triangle with sides 3, 4, and a
def is_acute_triangle (a : ℝ) : Prop :=
  a > 0 ∧ 3 > 0 ∧ 4 > 0 ∧
  a + 3 > 4 ∧ a + 4 > 3 ∧ 3 + 4 > a ∧
  a^2 < 3^2 + 4^2 ∧ 3^2 < a^2 + 4^2 ∧ 4^2 < a^2 + 3^2

-- Theorem statement
theorem acute_triangle_side_range :
  ∀ a : ℝ, is_acute_triangle a → Real.sqrt 7 < a ∧ a < 5 :=
by sorry

end NUMINAMATH_CALUDE_acute_triangle_side_range_l1239_123996


namespace NUMINAMATH_CALUDE_pastry_combinations_linda_pastry_purchase_l1239_123910

theorem pastry_combinations : ℕ → ℕ → ℕ
  | n, k => Nat.choose (n + k - 1) (k - 1)

theorem linda_pastry_purchase : pastry_combinations 4 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_pastry_combinations_linda_pastry_purchase_l1239_123910


namespace NUMINAMATH_CALUDE_expected_red_balls_l1239_123938

/-- The expected number of red balls selected when choosing 2 balls from a box containing 4 black, 3 red, and 2 white balls -/
theorem expected_red_balls (total_balls : ℕ) (red_balls : ℕ) (selected_balls : ℕ) 
  (h_total : total_balls = 9)
  (h_red : red_balls = 3)
  (h_selected : selected_balls = 2) :
  (red_balls : ℚ) * selected_balls / total_balls = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_balls_l1239_123938


namespace NUMINAMATH_CALUDE_unique_solution_abs_equation_l1239_123997

theorem unique_solution_abs_equation :
  ∃! y : ℝ, y * |y| = -3 * y + 5 :=
by
  -- The unique solution is (-3 + √29) / 2
  use (-3 + Real.sqrt 29) / 2
  sorry

end NUMINAMATH_CALUDE_unique_solution_abs_equation_l1239_123997


namespace NUMINAMATH_CALUDE_jacqueline_erasers_l1239_123901

-- Define the given quantities
def cases : ℕ := 7
def boxes_per_case : ℕ := 12
def erasers_per_box : ℕ := 25

-- Define the total number of erasers
def total_erasers : ℕ := cases * boxes_per_case * erasers_per_box

-- Theorem to prove
theorem jacqueline_erasers : total_erasers = 2100 := by
  sorry

end NUMINAMATH_CALUDE_jacqueline_erasers_l1239_123901


namespace NUMINAMATH_CALUDE_acute_angles_sum_l1239_123960

theorem acute_angles_sum (x y : Real) : 
  0 < x ∧ x < π/2 →
  0 < y ∧ y < π/2 →
  4 * (Real.cos x)^2 + 3 * (Real.cos y)^2 = 1 →
  4 * Real.cos (2*x) - 3 * Real.cos (2*y) = 0 →
  x + 3*y = π/2 := by
sorry

end NUMINAMATH_CALUDE_acute_angles_sum_l1239_123960


namespace NUMINAMATH_CALUDE_horse_purchase_problem_l1239_123965

theorem horse_purchase_problem (cuirassier_total : ℝ) (dragoon_total : ℝ) (dragoon_extra : ℕ) (price_diff : ℝ) :
  cuirassier_total = 11250 ∧ 
  dragoon_total = 16000 ∧ 
  dragoon_extra = 15 ∧ 
  price_diff = 50 →
  ∃ (cuirassier_count dragoon_count : ℕ) (cuirassier_price dragoon_price : ℝ),
    cuirassier_count = 25 ∧
    dragoon_count = 40 ∧
    cuirassier_price = 450 ∧
    dragoon_price = 400 ∧
    cuirassier_count * cuirassier_price = cuirassier_total ∧
    dragoon_count * dragoon_price = dragoon_total ∧
    dragoon_count = cuirassier_count + dragoon_extra ∧
    cuirassier_price = dragoon_price + price_diff :=
by sorry

end NUMINAMATH_CALUDE_horse_purchase_problem_l1239_123965


namespace NUMINAMATH_CALUDE_triangle_side_range_l1239_123936

theorem triangle_side_range (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  a * Real.cos C = c * Real.sin A →
  (∃ (a₁ b₁ : ℝ) (A₁ B₁ : ℝ), a₁ ≠ a ∨ b₁ ≠ b ∨ A₁ ≠ A ∨ B₁ ≠ B) →
  Real.sqrt 2 < b ∧ b < 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l1239_123936


namespace NUMINAMATH_CALUDE_johns_candy_store_spending_l1239_123912

theorem johns_candy_store_spending (allowance : ℝ) (arcade_fraction : ℝ) (toy_store_fraction : ℝ)
  (h1 : allowance = 3.375)
  (h2 : arcade_fraction = 3/5)
  (h3 : toy_store_fraction = 1/3) :
  allowance * (1 - arcade_fraction) * (1 - toy_store_fraction) = 0.90 := by
  sorry

end NUMINAMATH_CALUDE_johns_candy_store_spending_l1239_123912


namespace NUMINAMATH_CALUDE_game_ends_in_finite_steps_l1239_123966

/-- Represents the state of the game at any point -/
structure GameState where
  m : ℕ+  -- Player A's number
  n : ℕ+  -- Player B's number
  t : ℤ   -- The other number written by the umpire
  k : ℕ   -- The current question number

/-- Represents whether a player knows the other's number -/
def knows (state : GameState) (player : Bool) : Prop :=
  if player 
  then state.t ≤ state.m + state.n - state.n / state.k
  else state.t ≥ state.m + state.n + state.n / state.k

/-- The main theorem stating that the game will end after a finite number of questions -/
theorem game_ends_in_finite_steps : 
  ∀ (initial_state : GameState), 
  ∃ (final_state : GameState), 
  (knows final_state true ∨ knows final_state false) ∧ 
  final_state.k ≥ initial_state.k :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_finite_steps_l1239_123966


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1239_123906

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    x = Real.sqrt 2 ∧ y = Real.sqrt 3) →
  (Real.sqrt (1 + b^2 / a^2) = 2) →
  (∀ x y : ℝ, x^2 - y^2 / 3 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1239_123906


namespace NUMINAMATH_CALUDE_fraction_calculation_l1239_123930

theorem fraction_calculation (x y : ℚ) 
  (hx : x = 7/8) 
  (hy : y = 5/6) 
  (hx_nonzero : x ≠ 0) 
  (hy_nonzero : y ≠ 0) : 
  (4*x - 6*y) / (60*x*y) = -6/175 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1239_123930


namespace NUMINAMATH_CALUDE_partition_iff_even_l1239_123984

def is_valid_partition (n : ℕ) (partition : List (List ℕ)) : Prop :=
  partition.length = n ∧
  partition.all (λ l => l.length = 4) ∧
  (partition.join.toFinset : Finset ℕ) = Finset.range (4 * n + 1) \ {0} ∧
  ∀ l ∈ partition, ∃ x ∈ l, 3 * x = (l.sum - x)

theorem partition_iff_even (n : ℕ) :
  (∃ partition : List (List ℕ), is_valid_partition n partition) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_partition_iff_even_l1239_123984


namespace NUMINAMATH_CALUDE_lawn_length_is_70_l1239_123935

/-- Represents a rectangular lawn with roads -/
structure LawnWithRoads where
  length : ℝ
  width : ℝ
  roadWidth : ℝ
  gravelCostPerSquareMeter : ℝ
  totalGravelCost : ℝ

/-- Calculates the total area of the roads -/
def roadArea (l : LawnWithRoads) : ℝ :=
  l.length * l.roadWidth + l.width * l.roadWidth - l.roadWidth * l.roadWidth

/-- Theorem stating that given the conditions, the length of the lawn must be 70 m -/
theorem lawn_length_is_70 (l : LawnWithRoads)
    (h1 : l.width = 30)
    (h2 : l.roadWidth = 5)
    (h3 : l.gravelCostPerSquareMeter = 4)
    (h4 : l.totalGravelCost = 1900)
    (h5 : l.totalGravelCost = l.gravelCostPerSquareMeter * roadArea l) :
    l.length = 70 := by
  sorry

end NUMINAMATH_CALUDE_lawn_length_is_70_l1239_123935


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1239_123982

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^8 + 5*x^7 + 10*x^6 + 2023*x^5 - 2021*x^4

-- Theorem statement
theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1239_123982


namespace NUMINAMATH_CALUDE_total_swimming_time_l1239_123993

/-- Represents the swimming times for various events -/
structure SwimmingTimes where
  freestyle : ℕ
  backstroke : ℕ
  butterfly : ℕ
  breaststroke : ℕ
  sidestroke : ℕ
  individual_medley : ℕ

/-- Calculates the total time for all events -/
def total_time (times : SwimmingTimes) : ℕ :=
  times.freestyle + times.backstroke + times.butterfly + 
  times.breaststroke + times.sidestroke + times.individual_medley

/-- Theorem stating the total time for all events -/
theorem total_swimming_time :
  ∀ (times : SwimmingTimes),
    times.freestyle = 48 →
    times.backstroke = times.freestyle + 4 + 2 →
    times.butterfly = times.backstroke + 3 + 3 →
    times.breaststroke = times.butterfly + 2 - 1 →
    times.sidestroke = times.butterfly + 5 + 4 →
    times.individual_medley = times.breaststroke + 6 + 3 →
    total_time times = 362 := by
  sorry

#eval total_time { freestyle := 48, backstroke := 54, butterfly := 60, 
                   breaststroke := 61, sidestroke := 69, individual_medley := 70 }

end NUMINAMATH_CALUDE_total_swimming_time_l1239_123993


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1239_123943

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₃ = 3 and a₅ = -3, prove a₇ = -9 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h3 : a 3 = 3) 
  (h5 : a 5 = -3) : 
  a 7 = -9 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1239_123943


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l1239_123973

theorem fraction_sum_theorem (a b c : ℝ) 
  (h : a / (35 - a) + b / (55 - b) + c / (70 - c) = 8) :
  7 / (35 - a) + 11 / (55 - b) + 14 / (70 - c) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l1239_123973


namespace NUMINAMATH_CALUDE_equation_solution_l1239_123962

theorem equation_solution (x : ℝ) :
  x > 6 →
  (Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x + 6 * Real.sqrt (x - 6)) - 3) ↔
  x ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1239_123962


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l1239_123987

theorem pure_imaginary_product (b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (∃ (y : ℝ), (1 + b * Complex.I) * (2 + Complex.I) = y * Complex.I) →
  b = 2 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l1239_123987


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1239_123944

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 6*p - 3 = 0 →
  q^3 - 8*q^2 + 6*q - 3 = 0 →
  r^3 - 8*r^2 + 6*r - 3 = 0 →
  p/(q*r-1) + q/(p*r-1) + r/(p*q-1) = -14 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1239_123944


namespace NUMINAMATH_CALUDE_total_painting_cost_l1239_123995

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def count_digits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

def house_cost (address : ℕ) : ℚ :=
  (1.5 : ℚ) * (count_digits address)

def side_cost (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℚ :=
  List.sum (List.map house_cost (List.map (arithmetic_sequence a₁ d) (List.range n)))

theorem total_painting_cost :
  side_cost 5 6 25 + side_cost 2 6 25 = 171 := by
  sorry

end NUMINAMATH_CALUDE_total_painting_cost_l1239_123995


namespace NUMINAMATH_CALUDE_second_half_duration_percentage_l1239_123902

/-- Proves that the second half of a trip takes 200% longer than the first half
    given specific conditions about distance and speed. -/
theorem second_half_duration_percentage (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_second_half_duration_percentage_l1239_123902


namespace NUMINAMATH_CALUDE_binomial_properties_l1239_123915

variable (X : Nat → ℝ)

def binomial_distribution (n : Nat) (p : ℝ) (X : Nat → ℝ) : Prop :=
  ∀ k, 0 ≤ k ∧ k ≤ n → X k = (n.choose k : ℝ) * p^k * (1-p)^(n-k)

def expectation (X : Nat → ℝ) : ℝ := sorry
def variance (X : Nat → ℝ) : ℝ := sorry

theorem binomial_properties :
  binomial_distribution 8 (1/2) X →
  expectation X = 4 ∧
  variance X = 2 ∧
  X 3 = X 5 := by sorry

end NUMINAMATH_CALUDE_binomial_properties_l1239_123915


namespace NUMINAMATH_CALUDE_A_work_days_l1239_123946

/-- The number of days B takes to finish the work alone -/
def B_days : ℕ := 15

/-- The total wages when A and B work together -/
def total_wages : ℕ := 3100

/-- A's share of the wages when working together with B -/
def A_wages : ℕ := 1860

/-- The number of days A takes to finish the work alone -/
def A_days : ℕ := 10

theorem A_work_days :
  B_days = 15 ∧
  total_wages = 3100 ∧
  A_wages = 1860 →
  A_days = 10 :=
by sorry

end NUMINAMATH_CALUDE_A_work_days_l1239_123946


namespace NUMINAMATH_CALUDE_special_table_sum_l1239_123981

/-- Represents a 2 × 7 table where each column after the first is the sum and difference of the previous column --/
def SpecialTable := Fin 7 → Fin 2 → ℤ

/-- The rule for generating subsequent columns --/
def nextColumn (col : Fin 2 → ℤ) : Fin 2 → ℤ :=
  fun i => if i = 0 then col 0 + col 1 else col 0 - col 1

/-- Checks if the table follows the special rule --/
def isValidTable (t : SpecialTable) : Prop :=
  ∀ j : Fin 6, t (j.succ) = nextColumn (t j)

/-- The theorem to be proved --/
theorem special_table_sum (t : SpecialTable) : 
  isValidTable t → t 6 0 = 96 → t 6 1 = 64 → t 0 0 + t 0 1 = 20 := by
  sorry

#check special_table_sum

end NUMINAMATH_CALUDE_special_table_sum_l1239_123981


namespace NUMINAMATH_CALUDE_student_count_l1239_123963

/-- In a class, given a student who is both the 30th best and 30th worst, 
    the total number of students in the class is 59. -/
theorem student_count (n : ℕ) (rob : ℕ) 
  (h1 : rob = 30)  -- Rob's position from the top
  (h2 : rob = n - 29) : -- Rob's position from the bottom
  n = 59 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1239_123963


namespace NUMINAMATH_CALUDE_distance_circle_to_line_l1239_123986

/-- The distance from the center of the circle ρ=2cos θ to the line 2ρsin(θ + π/3)=1 is (√3 - 1) / 2 -/
theorem distance_circle_to_line :
  let circle : ℝ → ℝ → Prop := λ ρ θ ↦ ρ = 2 * Real.cos θ
  let line : ℝ → ℝ → Prop := λ ρ θ ↦ 2 * ρ * Real.sin (θ + π/3) = 1
  let circle_center : ℝ × ℝ := (1, 0)
  let distance := (Real.sqrt 3 - 1) / 2
  ∃ (d : ℝ), d = distance ∧ 
    d = (|Real.sqrt 3 * circle_center.1 + circle_center.2 - 1|) / Real.sqrt (3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_to_line_l1239_123986


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_property_l1239_123961

/-- A hyperbola in a 2D plane -/
structure Hyperbola where
  -- Add necessary fields to define a hyperbola
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Foci of the hyperbola -/
def foci (h : Hyperbola) : (Point × Point) := sorry

/-- Check if a point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop := sorry

/-- Diameter of the director circle -/
def director_circle_diameter (h : Hyperbola) : ℝ := sorry

theorem hyperbola_focal_distance_property (h : Hyperbola) (p : Point) :
  is_on_hyperbola h p →
  let (f1, f2) := foci h
  |distance p f1 - distance p f2| = director_circle_diameter h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_property_l1239_123961


namespace NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_11_l1239_123985

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem largest_odd_digit_multiple_of_11 :
  ∀ n : ℕ,
    n < 10000 →
    has_only_odd_digits n →
    is_divisible_by_11 n →
    n ≤ 9559 :=
sorry

end NUMINAMATH_CALUDE_largest_odd_digit_multiple_of_11_l1239_123985


namespace NUMINAMATH_CALUDE_factorization_equality_l1239_123958

theorem factorization_equality (m n : ℝ) : 
  2*m^2 - m*n + 2*m + n - n^2 = (2*m + n)*(m - n + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1239_123958


namespace NUMINAMATH_CALUDE_max_value_theorem_l1239_123955

theorem max_value_theorem (a b c d : ℝ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_sum : a / b + b / c + c / d + d / a = 4) (h_prod : a * c = b * d) :
  ∃ (max : ℝ), max = -12 ∧ ∀ (x : ℝ), x ≤ max ∧ (∃ (a' b' c' d' : ℝ), 
    a' / b' + b' / c' + c' / d' + d' / a' = 4 ∧ 
    a' * c' = b' * d' ∧
    x = a' / c' + b' / d' + c' / a' + d' / b') :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1239_123955


namespace NUMINAMATH_CALUDE_smallest_positive_and_largest_negative_integer_l1239_123926

theorem smallest_positive_and_largest_negative_integer :
  (∀ n : ℤ, n > 0 → n ≥ 1) ∧ (∀ m : ℤ, m < 0 → m ≤ -1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_and_largest_negative_integer_l1239_123926


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l1239_123903

theorem sqrt_expression_simplification :
  (Real.sqrt 7 - 1)^2 - (Real.sqrt 14 - Real.sqrt 2) * (Real.sqrt 14 + Real.sqrt 2) = -4 - 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l1239_123903


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1239_123929

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x, 3 * x^2 + m * x = 5) → 
  (3 * 5^2 + m * 5 = 5) → 
  (3 * (-1/3)^2 + m * (-1/3) = 5) :=
by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1239_123929


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1239_123945

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Valid angle measures
  A + B + C = π →  -- Angle sum in a triangle
  2 * b * Real.cos B = a * Real.cos C + c * Real.cos A →  -- Given condition
  b^2 = 3 * a * c →  -- Given condition
  A = π/12 ∨ A = 7*π/12 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1239_123945


namespace NUMINAMATH_CALUDE_trains_combined_length_l1239_123989

/-- The combined length of two trains crossing a platform -/
theorem trains_combined_length (speed_A speed_B : ℝ) (platform_length time : ℝ) : 
  speed_A = 72 * (5/18) → 
  speed_B = 54 * (5/18) → 
  platform_length = 210 → 
  time = 26 → 
  (speed_A + speed_B) * time - platform_length = 700 := by
  sorry


end NUMINAMATH_CALUDE_trains_combined_length_l1239_123989


namespace NUMINAMATH_CALUDE_triangle_not_right_angle_l1239_123975

theorem triangle_not_right_angle (a b c : ℝ) (h_sum : a + b + c = 180) (h_ratio : ∃ k : ℝ, a = 3*k ∧ b = 4*k ∧ c = 5*k) : ¬(a = 90 ∨ b = 90 ∨ c = 90) := by
  sorry

end NUMINAMATH_CALUDE_triangle_not_right_angle_l1239_123975


namespace NUMINAMATH_CALUDE_company_choices_eq_24_l1239_123988

/-- The number of ways two students can choose companies with exactly one overlap -/
def company_choices : ℕ :=
  -- Number of ways to choose the shared company
  4 *
  -- Ways for student A to choose the second company
  3 *
  -- Ways for student B to choose the second company
  2

/-- Theorem stating that the number of ways to choose companies with one overlap is 24 -/
theorem company_choices_eq_24 : company_choices = 24 := by
  sorry

end NUMINAMATH_CALUDE_company_choices_eq_24_l1239_123988


namespace NUMINAMATH_CALUDE_correct_statements_count_l1239_123954

/-- A circle in a plane. -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in a plane. -/
structure Line where
  point1 : Point
  point2 : Point

/-- A statement about circle geometry. -/
inductive CircleStatement
  | perpRadiusTangent : CircleStatement
  | centerPerpTangentThruPoint : CircleStatement
  | tangentPerpThruCenterPoint : CircleStatement
  | radiusEndPerpTangent : CircleStatement
  | chordTangentMidpoint : CircleStatement

/-- Determines if a circle statement is correct. -/
def isCorrectStatement (s : CircleStatement) : Bool :=
  match s with
  | CircleStatement.perpRadiusTangent => false
  | CircleStatement.centerPerpTangentThruPoint => true
  | CircleStatement.tangentPerpThruCenterPoint => true
  | CircleStatement.radiusEndPerpTangent => false
  | CircleStatement.chordTangentMidpoint => true

/-- The list of all circle statements. -/
def allStatements : List CircleStatement :=
  [CircleStatement.perpRadiusTangent,
   CircleStatement.centerPerpTangentThruPoint,
   CircleStatement.tangentPerpThruCenterPoint,
   CircleStatement.radiusEndPerpTangent,
   CircleStatement.chordTangentMidpoint]

/-- Counts the number of correct statements. -/
def countCorrectStatements (statements : List CircleStatement) : Nat :=
  statements.filter isCorrectStatement |>.length

theorem correct_statements_count :
  countCorrectStatements allStatements = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_count_l1239_123954


namespace NUMINAMATH_CALUDE_product_difference_equals_two_tenths_l1239_123956

theorem product_difference_equals_two_tenths : 0.5 * 0.8 - 0.2 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_equals_two_tenths_l1239_123956


namespace NUMINAMATH_CALUDE_parallel_line_not_through_point_l1239_123911

-- Define a line in 2D space
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

def is_point_on_line (p : Point) (l : Line) : Prop :=
  l.A * p.x + l.B * p.y + l.C = 0

def are_lines_parallel (l1 l2 : Line) : Prop :=
  l1.A * l2.B = l1.B * l2.A

theorem parallel_line_not_through_point 
  (L : Line) (P : Point) (h : ¬ is_point_on_line P L) :
  ∃ (L' : Line), 
    are_lines_parallel L' L ∧ 
    ¬ is_point_on_line P L' ∧
    L'.A = L.A ∧ 
    L'.B = L.B ∧ 
    L'.C = L.C + (L.A * P.x + L.B * P.y + L.C) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_not_through_point_l1239_123911


namespace NUMINAMATH_CALUDE_smallest_integer_l1239_123983

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 50) :
  ∀ c : ℕ, c > 0 ∧ Nat.lcm a c / Nat.gcd a c = 50 → b ≤ c := by
  sorry

#check smallest_integer

end NUMINAMATH_CALUDE_smallest_integer_l1239_123983


namespace NUMINAMATH_CALUDE_second_exam_sleep_for_average_85_l1239_123937

/-- Represents the relationship between sleep hours and test score -/
structure ExamData where
  sleep : ℝ
  score : ℝ

/-- The constant product of sleep hours and test score -/
def sleepScoreProduct (data : ExamData) : ℝ := data.sleep * data.score

theorem second_exam_sleep_for_average_85 
  (first_exam : ExamData)
  (h_first_exam : first_exam.sleep = 6 ∧ first_exam.score = 60)
  (h_inverse_relation : ∀ exam : ExamData, sleepScoreProduct exam = sleepScoreProduct first_exam)
  (second_exam : ExamData)
  (h_second_exam : second_exam.sleep = 3.3) :
  (first_exam.score + second_exam.score) / 2 = 85 := by
sorry

end NUMINAMATH_CALUDE_second_exam_sleep_for_average_85_l1239_123937


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1239_123972

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  Real.sqrt ((3 - (-5))^2 + (y - 4)^2) = 12 → 
  y = 4 + 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1239_123972


namespace NUMINAMATH_CALUDE_range_of_m_for_quadratic_equation_l1239_123916

theorem range_of_m_for_quadratic_equation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0) ↔ 
  (m < -2 ∨ m > 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_quadratic_equation_l1239_123916


namespace NUMINAMATH_CALUDE_total_cargo_after_loading_l1239_123970

def initial_cargo : ℕ := 5973
def loaded_cargo : ℕ := 8723

theorem total_cargo_after_loading : initial_cargo + loaded_cargo = 14696 := by
  sorry

end NUMINAMATH_CALUDE_total_cargo_after_loading_l1239_123970


namespace NUMINAMATH_CALUDE_taxi_distance_is_ten_miles_l1239_123968

/-- Calculates the taxi fare distance given the total fare, initial fare, initial distance, and additional fare per unit distance -/
def taxi_fare_distance (total_fare : ℚ) (initial_fare : ℚ) (initial_distance : ℚ) (additional_fare_per_unit : ℚ) : ℚ :=
  initial_distance + (total_fare - initial_fare) / additional_fare_per_unit

/-- Theorem: Given the specified fare structure and total fare, the distance traveled is 10 miles -/
theorem taxi_distance_is_ten_miles :
  let total_fare : ℚ := 59
  let initial_fare : ℚ := 10
  let initial_distance : ℚ := 1/5
  let additional_fare_per_unit : ℚ := 1/(1/5)
  taxi_fare_distance total_fare initial_fare initial_distance additional_fare_per_unit = 10 := by
  sorry

#eval taxi_fare_distance 59 10 (1/5) 5

end NUMINAMATH_CALUDE_taxi_distance_is_ten_miles_l1239_123968


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1239_123908

theorem complex_equation_solution :
  ∀ z : ℂ, -Complex.I * z = (3 + 2 * Complex.I) * (1 - Complex.I) → z = 1 + 5 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1239_123908


namespace NUMINAMATH_CALUDE_unique_score_determination_l1239_123994

/-- Scoring system function -/
def score (c w : ℕ) : ℕ := 30 + 4 * c - w

/-- Proposition: There exists a unique combination of c and w such that the score is 92,
    and this is the only score above 90 that allows for a unique determination of c and w -/
theorem unique_score_determination :
  ∃! (c w : ℕ), score c w = 92 ∧
  (∀ (c' w' : ℕ), score c' w' > 90 ∧ score c' w' ≠ 92 → 
    ∃ (c'' w'' : ℕ), c'' ≠ c' ∧ w'' ≠ w' ∧ score c'' w'' = score c' w') :=
sorry

end NUMINAMATH_CALUDE_unique_score_determination_l1239_123994


namespace NUMINAMATH_CALUDE_unique_divisible_by_396_l1239_123951

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧
    n = x * 100000 + y * 10000 + 2 * 1000 + 4 * 100 + 3 * 10 + z

theorem unique_divisible_by_396 :
  ∃! n : ℕ, is_valid_number n ∧ n % 396 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_396_l1239_123951


namespace NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l1239_123967

theorem x_positive_necessary_not_sufficient :
  (∃ x : ℝ, x > 0 ∧ ¬(|x - 1| < 1)) ∧
  (∀ x : ℝ, |x - 1| < 1 → x > 0) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l1239_123967


namespace NUMINAMATH_CALUDE_inequality_proof_l1239_123909

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b + b * c + a * c)^2 ≥ 3 * a * b * c * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1239_123909


namespace NUMINAMATH_CALUDE_rikkis_earnings_l1239_123932

/-- Represents Rikki's poetry writing and selling scenario -/
structure PoetryScenario where
  price_per_word : ℚ
  words_per_unit : ℕ
  minutes_per_unit : ℕ
  total_hours : ℕ

/-- Calculates the expected earnings for a given poetry scenario -/
def expected_earnings (scenario : PoetryScenario) : ℚ :=
  let total_minutes : ℕ := scenario.total_hours * 60
  let total_units : ℕ := total_minutes / scenario.minutes_per_unit
  let total_words : ℕ := total_units * scenario.words_per_unit
  (total_words : ℚ) * scenario.price_per_word

/-- Rikki's specific poetry scenario -/
def rikkis_scenario : PoetryScenario :=
  { price_per_word := 1 / 100
  , words_per_unit := 25
  , minutes_per_unit := 5
  , total_hours := 2 }

theorem rikkis_earnings :
  expected_earnings rikkis_scenario = 6 := by
  sorry

end NUMINAMATH_CALUDE_rikkis_earnings_l1239_123932


namespace NUMINAMATH_CALUDE_natural_growth_determined_by_birth_and_death_rates_l1239_123900

/-- Represents the rate of change in a population -/
structure PopulationRate :=
  (value : ℝ)

/-- The natural growth rate of a population -/
def naturalGrowthRate (birthRate deathRate : PopulationRate) : PopulationRate :=
  ⟨birthRate.value - deathRate.value⟩

/-- Theorem stating that the natural growth rate is determined by both birth and death rates -/
theorem natural_growth_determined_by_birth_and_death_rates 
  (birthRate deathRate : PopulationRate) :
  ∃ (f : PopulationRate → PopulationRate → PopulationRate), 
    naturalGrowthRate birthRate deathRate = f birthRate deathRate :=
by
  sorry


end NUMINAMATH_CALUDE_natural_growth_determined_by_birth_and_death_rates_l1239_123900


namespace NUMINAMATH_CALUDE_younger_person_age_l1239_123924

theorem younger_person_age (y e : ℕ) : 
  e = y + 20 → 
  e - 4 = 5 * (y - 4) → 
  y = 9 := by
sorry

end NUMINAMATH_CALUDE_younger_person_age_l1239_123924


namespace NUMINAMATH_CALUDE_value_of_7a_plus_3b_l1239_123948

-- Define the function g
def g (x : ℝ) : ℝ := 7 * x - 4

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem value_of_7a_plus_3b 
  (a b : ℝ) 
  (h1 : ∀ x, g x = (Function.invFun (f a b)) x - 5) 
  (h2 : Function.Injective (f a b)) :
  7 * a + 3 * b = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_value_of_7a_plus_3b_l1239_123948


namespace NUMINAMATH_CALUDE_white_balls_count_l1239_123939

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) 
  (h_total : total = 60)
  (h_green : green = 18)
  (h_yellow : yellow = 8)
  (h_red : red = 5)
  (h_purple : purple = 7)
  (h_prob : prob_not_red_purple = 4/5) :
  total - (green + yellow + red + purple) = 22 := by
sorry

end NUMINAMATH_CALUDE_white_balls_count_l1239_123939


namespace NUMINAMATH_CALUDE_circle_triangle_area_constraint_l1239_123904

/-- The range of r for which there are exactly two points on the circle
    (x-2)^2 + y^2 = r^2 that form triangles with area 4 with given points A and B -/
theorem circle_triangle_area_constraint (r : ℝ) : 
  r > 0 →
  (∃! M N : ℝ × ℝ, 
    (M.1 - 2)^2 + M.2^2 = r^2 ∧
    (N.1 - 2)^2 + N.2^2 = r^2 ∧
    abs ((M.1 + 3) * (-2) - (M.2 - 0) * (-2)) / 2 = 4 ∧
    abs ((N.1 + 3) * (-2) - (N.2 - 0) * (-2)) / 2 = 4) →
  r ∈ Set.Ioo (Real.sqrt 2 / 2) (9 * Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_triangle_area_constraint_l1239_123904


namespace NUMINAMATH_CALUDE_fourth_power_sqrt_equals_256_l1239_123977

theorem fourth_power_sqrt_equals_256 (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sqrt_equals_256_l1239_123977


namespace NUMINAMATH_CALUDE_min_friend_pairs_2000_users_min_friend_pairs_within_bounds_l1239_123922

/-- Represents a social network with a fixed number of users and invitations per user. -/
structure SocialNetwork where
  numUsers : ℕ
  invitationsPerUser : ℕ

/-- Calculates the minimum number of friend pairs in a social network where friendships
    are formed only when invitations are mutual. -/
def minFriendPairs (network : SocialNetwork) : ℕ :=
  (network.numUsers * network.invitationsPerUser) / 2

/-- Theorem stating that in a social network with 2000 users, each inviting 1000 others,
    the minimum number of friend pairs is 1000. -/
theorem min_friend_pairs_2000_users :
  let network : SocialNetwork := { numUsers := 2000, invitationsPerUser := 1000 }
  minFriendPairs network = 1000 := by
  sorry

/-- Verifies that the calculated minimum number of friend pairs does not exceed
    the maximum possible number of pairs given the number of users. -/
theorem min_friend_pairs_within_bounds (network : SocialNetwork) :
  minFriendPairs network ≤ (network.numUsers.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_min_friend_pairs_2000_users_min_friend_pairs_within_bounds_l1239_123922


namespace NUMINAMATH_CALUDE_house_painting_cost_is_1900_l1239_123953

/-- The cost of painting a house given the contributions of three individuals. -/
def housePaintingCost (judsonContribution : ℕ) : ℕ :=
  let kennyContribution := judsonContribution + judsonContribution / 5
  let camiloContribution := kennyContribution + 200
  judsonContribution + kennyContribution + camiloContribution

/-- Theorem stating that the total cost of painting the house is $1900. -/
theorem house_painting_cost_is_1900 : housePaintingCost 500 = 1900 := by
  sorry

end NUMINAMATH_CALUDE_house_painting_cost_is_1900_l1239_123953


namespace NUMINAMATH_CALUDE_tenth_term_is_44_l1239_123934

/-- Arithmetic sequence with first term 8 and common difference 4 -/
def arithmetic_sequence (n : ℕ) : ℕ := 8 + 4 * (n - 1)

/-- The 10th term of the arithmetic sequence is 44 -/
theorem tenth_term_is_44 : arithmetic_sequence 10 = 44 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_44_l1239_123934


namespace NUMINAMATH_CALUDE_initial_speed_proof_l1239_123976

/-- Proves that given the conditions of the journey, the initial speed must be 60 mph -/
theorem initial_speed_proof (v : ℝ) : 
  (v * 3 + 85 * 2) / 5 = 70 → v = 60 := by
  sorry

end NUMINAMATH_CALUDE_initial_speed_proof_l1239_123976


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1239_123914

/-- A geometric sequence with a_1 = 1/9 and a_5 = 9 has a_3 = 1 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 / 9 →
  a 5 = 9 →
  a 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1239_123914


namespace NUMINAMATH_CALUDE_cos_120_degrees_l1239_123940

theorem cos_120_degrees : Real.cos (120 * π / 180) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l1239_123940


namespace NUMINAMATH_CALUDE_log_equation_solution_l1239_123957

theorem log_equation_solution : 
  ∃! x : ℝ, x > 0 ∧ 2 * Real.log x = Real.log 192 + Real.log 3 - Real.log 4 :=
by
  -- The unique solution is x = 12
  use 12
  constructor
  · -- Prove that x = 12 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check log_equation_solution

end NUMINAMATH_CALUDE_log_equation_solution_l1239_123957


namespace NUMINAMATH_CALUDE_quadratic_equation_with_root_one_l1239_123907

theorem quadratic_equation_with_root_one (a : ℝ) (h : a ≠ 0) :
  ∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 - a) ∧ f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_root_one_l1239_123907


namespace NUMINAMATH_CALUDE_valid_numbers_l1239_123971

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 730000 + 10000 * a + 1000 * b + 100 * c + 6 ∧
    b < 4 ∧
    n % 56 = 0 ∧
    (a % 40 = a % 61) ∧ (a % 61 = a % 810)

theorem valid_numbers :
  ∀ n : ℕ, is_valid_number n ↔ (n = 731136 ∨ n = 737016 ∨ n = 737296) :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1239_123971


namespace NUMINAMATH_CALUDE_triangle_cosine_C_l1239_123969

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ A < π / 2)
  (h2 : 0 < B ∧ B < π / 2)
  (h3 : 0 < C ∧ C < π / 2)
  (h4 : A + B + C = π)

-- State the theorem
theorem triangle_cosine_C (t : Triangle) 
  (h5 : f t.A = 3 / 2)
  (h6 : ∃ (D : ℝ), D = Real.sqrt 2 ∧ D * Real.sin (t.B / 2) = t.c * Real.sin (t.A / 2))
  (h7 : ∃ (D : ℝ), D = 2 ∧ D * Real.sin (t.A / 2) = t.b * Real.sin (t.C / 2)) :
  Real.cos t.C = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_cosine_C_l1239_123969


namespace NUMINAMATH_CALUDE_binomial_sum_l1239_123947

theorem binomial_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₁ + a₃ + a₅ = 123 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l1239_123947


namespace NUMINAMATH_CALUDE_john_annual_profit_l1239_123920

def annual_profit (num_subletters : ℕ) (subletter_rent : ℕ) (monthly_expense : ℕ) (months_per_year : ℕ) : ℕ :=
  (num_subletters * subletter_rent - monthly_expense) * months_per_year

theorem john_annual_profit :
  annual_profit 3 400 900 12 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_john_annual_profit_l1239_123920


namespace NUMINAMATH_CALUDE_smallest_integer_power_l1239_123927

theorem smallest_integer_power (x : ℕ) : (∀ y : ℕ, y < x → 27^y ≤ 3^24) ∧ 27^x > 3^24 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_power_l1239_123927


namespace NUMINAMATH_CALUDE_angle_SPQ_is_20_degrees_l1239_123999

/-- A geometric configuration with specific angle measures -/
structure GeometricConfiguration where
  -- Point Q lies on PR and Point S lies on QT (implied by the existence of these angles)
  angle_QST : ℝ  -- Measure of angle QST
  angle_TSP : ℝ  -- Measure of angle TSP
  angle_RQS : ℝ  -- Measure of angle RQS

/-- Theorem stating that under the given conditions, angle SPQ measures 20 degrees -/
theorem angle_SPQ_is_20_degrees (config : GeometricConfiguration)
  (h1 : config.angle_QST = 180)
  (h2 : config.angle_TSP = 50)
  (h3 : config.angle_RQS = 150) :
  config.angle_TSP + 20 = config.angle_RQS :=
sorry

end NUMINAMATH_CALUDE_angle_SPQ_is_20_degrees_l1239_123999


namespace NUMINAMATH_CALUDE_percentage_difference_l1239_123992

theorem percentage_difference (x y z : ℝ) : 
  x = 1.25 * y →
  x + y + z = 1110 →
  z = 300 →
  (y - z) / z = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1239_123992


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l1239_123933

def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum : 
  interior_sum 6 = 30 → interior_sum 8 + interior_sum 9 = 380 := by
sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l1239_123933


namespace NUMINAMATH_CALUDE_simplify_cubic_root_sum_exponents_l1239_123921

-- Define the expression
def radicand : ℕ → ℕ → ℕ → ℕ → ℕ
  | a, b, c, d => 60 * a^5 * b^7 * c^8 * d^2

-- Define the function to calculate the sum of exponents outside the radical
def sum_exponents_outside_radical : ℕ → ℕ → ℕ → ℕ → ℕ
  | a, b, c, d => 5

-- Theorem statement
theorem simplify_cubic_root_sum_exponents
  (a b c d : ℕ) :
  sum_exponents_outside_radical a b c d = 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_cubic_root_sum_exponents_l1239_123921


namespace NUMINAMATH_CALUDE_first_day_pages_l1239_123949

/-- Proves that given the specific writing pattern and remaining pages, the writer wrote 25 pages on the first day -/
theorem first_day_pages (total_pages remaining_pages day_4_pages : ℕ) 
  (h1 : total_pages = 500)
  (h2 : remaining_pages = 315)
  (h3 : day_4_pages = 10) : 
  ∃ x : ℕ, x + 2*x + 4*x + day_4_pages = total_pages - remaining_pages ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_first_day_pages_l1239_123949


namespace NUMINAMATH_CALUDE_first_group_size_first_group_size_proof_l1239_123974

/-- Given two groups of workers building walls, this theorem proves that the number of workers in the first group is 20, based on the given conditions. -/
theorem first_group_size : ℕ :=
  let wall_length_1 : ℝ := 66
  let days_1 : ℕ := 4
  let wall_length_2 : ℝ := 567.6
  let days_2 : ℕ := 8
  let workers_2 : ℕ := 86
  let workers_1 := (wall_length_1 * days_2 * workers_2) / (wall_length_2 * days_1)
  20

/-- Proof of the theorem -/
theorem first_group_size_proof : first_group_size = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_first_group_size_proof_l1239_123974


namespace NUMINAMATH_CALUDE_jessica_journey_length_l1239_123964

/-- Represents Jessica's journey in miles -/
def journey_distance : ℝ → Prop :=
  λ total_distance =>
    ∃ (rough_trail tunnel bridge : ℝ),
      -- The journey consists of three parts
      total_distance = rough_trail + tunnel + bridge ∧
      -- The rough trail is one-quarter of the total distance
      rough_trail = (1/4) * total_distance ∧
      -- The tunnel is 25 miles long
      tunnel = 25 ∧
      -- The bridge is one-fourth of the total distance
      bridge = (1/4) * total_distance

/-- Theorem stating that Jessica's journey is 50 miles long -/
theorem jessica_journey_length :
  journey_distance 50 := by
  sorry

end NUMINAMATH_CALUDE_jessica_journey_length_l1239_123964


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_example_l1239_123913

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: An arithmetic sequence starting with 2, ending with 1007, 
    and having a common difference of 5, contains 202 terms. -/
theorem arithmetic_sequence_length_example : 
  arithmetic_sequence_length 2 1007 5 = 202 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_example_l1239_123913


namespace NUMINAMATH_CALUDE_class_election_combinations_l1239_123959

/-- The number of candidates for class president -/
def president_candidates : ℕ := 3

/-- The number of candidates for vice president -/
def vice_president_candidates : ℕ := 5

/-- The total number of ways to choose one class president and one vice president -/
def total_ways : ℕ := president_candidates * vice_president_candidates

theorem class_election_combinations :
  total_ways = 15 :=
by sorry

end NUMINAMATH_CALUDE_class_election_combinations_l1239_123959


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l1239_123905

/-- Given an ellipse mx^2 + ny^2 = 1 intersecting with a line x + y - 1 = 0,
    if the slope of the line passing through the origin and the midpoint of
    the intersection points is √2/2, then n/m = √2 -/
theorem ellipse_line_intersection (m n : ℝ) :
  (∃ A B : ℝ × ℝ,
    m * A.1^2 + n * A.2^2 = 1 ∧
    m * B.1^2 + n * B.2^2 = 1 ∧
    A.1 + A.2 = 1 ∧
    B.1 + B.2 = 1 ∧
    (A ≠ B) ∧
    ((A.2 + B.2)/2) / ((A.1 + B.1)/2) = Real.sqrt 2 / 2) →
  n / m = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l1239_123905


namespace NUMINAMATH_CALUDE_rectangular_prism_ratios_l1239_123918

/-- A rectangular prism with edges a, b, c, and free surface ratios p, q, r -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ
  q : ℝ
  r : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  p_pos : 0 < p
  q_pos : 0 < q
  r_pos : 0 < r

/-- The theorem stating the edge ratios and conditions for p, q, r -/
theorem rectangular_prism_ratios (prism : RectangularPrism) :
  (prism.a : ℝ) / (prism.b : ℝ) = (2 * prism.p - 3 * prism.q + 2 * prism.r) / (-3 * prism.p + 2 * prism.q + 2 * prism.r) ∧
  (prism.b : ℝ) / (prism.c : ℝ) = (2 * prism.p + 2 * prism.q - 3 * prism.r) / (2 * prism.p - 3 * prism.q + 2 * prism.r) ∧
  (prism.c : ℝ) / (prism.a : ℝ) = (-3 * prism.p + 2 * prism.q + 2 * prism.r) / (2 * prism.p + 2 * prism.q - 3 * prism.r) ∧
  2 * prism.p + 2 * prism.r > 3 * prism.q ∧
  2 * prism.p + 2 * prism.q > 3 * prism.r ∧
  2 * prism.q + 2 * prism.r > 3 * prism.p := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_ratios_l1239_123918


namespace NUMINAMATH_CALUDE_no_solution_for_specific_a_l1239_123928

/-- The equation 7|x-4a|+|x-a²|+6x-2a=0 has no solution when a ∈ (-∞, -18) ∪ (0, +∞) -/
theorem no_solution_for_specific_a (a : ℝ) : 
  (a < -18 ∨ a > 0) → ¬∃ x : ℝ, 7*|x - 4*a| + |x - a^2| + 6*x - 2*a = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_specific_a_l1239_123928


namespace NUMINAMATH_CALUDE_problem_statement_l1239_123925

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (a / (1 + a)) + (b / (1 + b)) = 1) : 
  (a / (1 + b^2)) - (b / (1 + a^2)) = a - b := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1239_123925


namespace NUMINAMATH_CALUDE_restaurant_bill_total_l1239_123923

theorem restaurant_bill_total (num_people : ℕ) (individual_payment : ℚ) (total_bill : ℚ) : 
  num_people = 9 → 
  individual_payment = 514.19 → 
  total_bill = num_people * individual_payment → 
  total_bill = 4627.71 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_total_l1239_123923


namespace NUMINAMATH_CALUDE_quiz_probabilities_l1239_123950

/-- Represents a quiz with multiple-choice and true/false questions -/
structure Quiz where
  total_questions : ℕ
  multiple_choice : ℕ
  true_false : ℕ
  h_total : total_questions = multiple_choice + true_false

/-- Calculates the probability of an event in a quiz draw -/
def probability (q : Quiz) (favorable_outcomes : ℕ) : ℚ :=
  favorable_outcomes / (q.total_questions * (q.total_questions - 1))

theorem quiz_probabilities (q : Quiz) 
    (h_total : q.total_questions = 5)
    (h_mc : q.multiple_choice = 3)
    (h_tf : q.true_false = 2) :
  let p1 := probability q (q.true_false * q.multiple_choice)
  let p2 := 1 - probability q (q.true_false * (q.true_false - 1))
  p1 = 3/10 ∧ p2 = 9/10 := by
  sorry


end NUMINAMATH_CALUDE_quiz_probabilities_l1239_123950


namespace NUMINAMATH_CALUDE_boys_usual_time_to_school_l1239_123919

theorem boys_usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) : 
  usual_rate > 0 → 
  usual_time > 0 → 
  usual_rate * usual_time = (6/5 * usual_rate) * (usual_time - 4) → 
  usual_time = 24 := by
sorry

end NUMINAMATH_CALUDE_boys_usual_time_to_school_l1239_123919


namespace NUMINAMATH_CALUDE_inequality_proof_l1239_123980

theorem inequality_proof (a b : ℝ) (h : a * b ≥ 0) :
  a^4 + 2*a^3*b + 2*a*b^3 + b^4 ≥ 6*a^2*b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1239_123980


namespace NUMINAMATH_CALUDE_otimes_four_two_l1239_123917

-- Define the new operation ⊗
def otimes (a b : ℝ) : ℝ := 4 * a + 5 * b

-- Theorem to prove
theorem otimes_four_two : otimes 4 2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_otimes_four_two_l1239_123917


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l1239_123952

theorem quadratic_factorization_sum (p q r : ℤ) : 
  (∀ x, x^2 + 16*x + 63 = (x + p) * (x + q)) →
  (∀ x, x^2 - 15*x + 56 = (x - q) * (x - r)) →
  p + q + r = 22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l1239_123952


namespace NUMINAMATH_CALUDE_smallest_abs_value_not_one_l1239_123998

theorem smallest_abs_value_not_one : ¬(∀ q : ℚ, q ≠ 0 → |q| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_abs_value_not_one_l1239_123998


namespace NUMINAMATH_CALUDE_museum_revenue_l1239_123990

def minutes_between (start_hour start_min end_hour end_min : ℕ) : ℕ :=
  (end_hour - start_hour) * 60 + end_min - start_min

def total_intervals (interval_length : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / interval_length

def total_people (people_per_interval intervals : ℕ) : ℕ :=
  people_per_interval * intervals

def student_tickets (total_people : ℕ) : ℕ :=
  total_people / 4

def regular_tickets (student_tickets : ℕ) : ℕ :=
  3 * student_tickets

def total_revenue (student_tickets regular_tickets : ℕ) (student_price regular_price : ℕ) : ℕ :=
  student_tickets * student_price + regular_tickets * regular_price

theorem museum_revenue : 
  let total_mins := minutes_between 9 0 17 55
  let intervals := total_intervals 5 total_mins
  let total_ppl := total_people 30 intervals
  let students := student_tickets total_ppl
  let regulars := regular_tickets students
  total_revenue students regulars 4 8 = 22456 := by
  sorry

end NUMINAMATH_CALUDE_museum_revenue_l1239_123990


namespace NUMINAMATH_CALUDE_final_price_calculation_l1239_123991

/-- The markup percentage applied to the cost price -/
def markup : ℝ := 0.15

/-- The cost price of the computer table -/
def costPrice : ℝ := 5565.217391304348

/-- The final price paid by the customer -/
def finalPrice : ℝ := 6400

/-- Theorem stating that the final price is equal to the cost price plus the markup -/
theorem final_price_calculation :
  finalPrice = costPrice * (1 + markup) := by sorry

end NUMINAMATH_CALUDE_final_price_calculation_l1239_123991


namespace NUMINAMATH_CALUDE_remainder_233_divided_by_d_l1239_123931

theorem remainder_233_divided_by_d (a b c d : ℕ) : 
  1 < a → a < b → b < c → a + c = 13 → d = a * b * c → 
  233 % d = 53 := by sorry

end NUMINAMATH_CALUDE_remainder_233_divided_by_d_l1239_123931
