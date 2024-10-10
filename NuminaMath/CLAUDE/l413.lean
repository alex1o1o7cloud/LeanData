import Mathlib

namespace abcd_not_2012_l413_41326

theorem abcd_not_2012 (a b c d : ℤ) 
  (h : (a - b) * (c + d) = (a + b) * (c - d)) : 
  a * b * c * d ≠ 2012 := by
sorry

end abcd_not_2012_l413_41326


namespace infinite_solutions_l413_41300

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x - 4 * y = 10
def equation2 (x y : ℝ) : Prop := 6 * x - 8 * y = 20

-- Theorem stating that the system has infinitely many solutions
theorem infinite_solutions :
  ∃ (f : ℝ → ℝ × ℝ), ∀ t : ℝ,
    let (x, y) := f t
    equation1 x y ∧ equation2 x y ∧
    (∀ s : ℝ, s ≠ t → f s ≠ f t) :=
sorry

end infinite_solutions_l413_41300


namespace reciprocal_ratio_sum_inequality_l413_41357

theorem reciprocal_ratio_sum_inequality (a b : ℝ) (h : a * b < 0) :
  b / a + a / b ≤ -2 := by
  sorry

end reciprocal_ratio_sum_inequality_l413_41357


namespace escalator_standing_time_l413_41303

/-- Represents the time it takes to travel an escalator under different conditions -/
def EscalatorTime (normal_time twice_normal_time : ℝ) : Prop :=
  ∃ (x u : ℝ),
    x > 0 ∧ u > 0 ∧
    (u + x) * normal_time = (u + 2*x) * twice_normal_time ∧
    u * (normal_time * 1.5) = (u + x) * normal_time

theorem escalator_standing_time 
  (h : EscalatorTime 40 30) : 
  ∃ (standing_time : ℝ), standing_time = 60 :=
by sorry

end escalator_standing_time_l413_41303


namespace combinatorics_identities_l413_41334

theorem combinatorics_identities :
  (∀ n k : ℕ, Nat.choose n k = Nat.choose n (n - k)) ∧
  (Nat.choose 5 3 = Nat.choose 4 2 + Nat.choose 4 3) ∧
  (5 * Nat.factorial 5 = Nat.factorial 6 - Nat.factorial 5) :=
by sorry

end combinatorics_identities_l413_41334


namespace product_equals_533_l413_41308

/-- Converts a list of digits in a given base to its decimal representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

/-- The binary representation of the first number -/
def binary_num : List Nat := [1, 0, 1, 1]

/-- The ternary representation of the second number -/
def ternary_num : List Nat := [2, 1, 1, 1]

theorem product_equals_533 :
  (to_decimal binary_num 2) * (to_decimal ternary_num 3) = 533 := by
  sorry

end product_equals_533_l413_41308


namespace absolute_value_equation_solution_difference_l413_41312

theorem absolute_value_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  x₁ > x₂ ∧ 
  (|x₁ - 3| = 15) ∧ 
  (|x₂ - 3| = 15) ∧ 
  (x₁ - x₂ = 30) := by
  sorry

end absolute_value_equation_solution_difference_l413_41312


namespace ac_in_open_interval_sum_of_endpoints_l413_41370

/-- Represents a triangle ABC with an angle bisector from A to D on BC -/
structure AngleBisectorTriangle where
  -- The length of side AB
  ab : ℝ
  -- The length of CD (part of BC)
  cd : ℝ
  -- The length of AC
  ac : ℝ
  -- Assumption that AB = 15
  ab_eq : ab = 15
  -- Assumption that CD = 5
  cd_eq : cd = 5
  -- Assumption that AC is positive
  ac_pos : ac > 0
  -- Assumption that ABC forms a valid triangle
  triangle_inequality : ac + cd + (75 / ac) > ab ∧ ab + cd + (75 / ac) > ac ∧ ab + ac > cd + (75 / ac)
  -- Assumption that AD is the angle bisector
  angle_bisector : ab / ac = (75 / ac) / cd

/-- The main theorem stating that AC must be in the open interval (5, 25) -/
theorem ac_in_open_interval (t : AngleBisectorTriangle) : 5 < t.ac ∧ t.ac < 25 := by
  sorry

/-- The sum of the endpoints of the interval is 30 -/
theorem sum_of_endpoints : 5 + 25 = 30 := by
  sorry

end ac_in_open_interval_sum_of_endpoints_l413_41370


namespace initial_average_marks_l413_41302

/-- 
Given a class of students with an incorrect average mark, prove that the initial average 
before correcting an error in one student's mark is equal to a specific value.
-/
theorem initial_average_marks 
  (n : ℕ) -- number of students
  (wrong_mark correct_mark : ℕ) -- the wrong and correct marks for one student
  (final_average : ℚ) -- the correct average after fixing the error
  (h1 : n = 25) -- there are 25 students
  (h2 : wrong_mark = 60) -- the wrong mark was 60
  (h3 : correct_mark = 10) -- the correct mark is 10
  (h4 : final_average = 98) -- the final correct average is 98
  : ∃ (initial_average : ℚ), initial_average = 100 ∧ 
    n * initial_average - (wrong_mark - correct_mark) = n * final_average :=
by sorry

end initial_average_marks_l413_41302


namespace m_range_l413_41384

def A : Set ℝ := {x | (x + 1) / (x - 3) < 0}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

theorem m_range (m : ℝ) : 
  (∀ x, x ∈ B m → x ∈ A) ∧ 
  (∃ x, x ∈ A ∧ x ∉ B m) → 
  m > 2 :=
sorry

end m_range_l413_41384


namespace company_workers_l413_41393

theorem company_workers (total : ℕ) (men : ℕ) : 
  (total / 3 : ℚ) = total / 3 →  -- One-third of workers don't have a retirement plan
  (1 / 5 : ℚ) * (total / 3 : ℚ) = total / 15 →  -- 20% of workers without a retirement plan are women
  (2 / 5 : ℚ) * ((2 * total) / 3 : ℚ) = (4 * total) / 15 →  -- 40% of workers with a retirement plan are men
  men = 144 →  -- There are 144 men
  total - men = 126  -- The number of women workers is 126
  := by sorry

end company_workers_l413_41393


namespace prime_roots_integer_l413_41342

theorem prime_roots_integer (p : ℕ) : 
  Prime p ∧ 
  (∃ x y : ℤ, x ≠ y ∧ 
    x^2 + 2*p*x - 240*p = 0 ∧ 
    y^2 + 2*p*y - 240*p = 0) ↔ 
  p = 5 := by
sorry

end prime_roots_integer_l413_41342


namespace x_value_l413_41327

theorem x_value (y : ℝ) (x : ℝ) : 
  y = 125 * (1 + 0.1) → 
  x = y * (1 - 0.1) → 
  x = 123.75 := by
sorry

end x_value_l413_41327


namespace barrel_capacity_l413_41363

/-- Represents a barrel with two taps -/
structure Barrel :=
  (capacity : ℝ)
  (midwayTapRate : ℝ) -- Liters per minute
  (bottomTapRate : ℝ) -- Liters per minute

/-- Represents the scenario of drawing beer from the barrel -/
def drawBeer (barrel : Barrel) (earlyUseTime : ℝ) (assistantUseTime : ℝ) : Prop :=
  -- The capacity is twice the amount drawn early plus the amount drawn by the assistant
  barrel.capacity = 2 * (earlyUseTime * barrel.midwayTapRate + assistantUseTime * barrel.bottomTapRate)

/-- The main theorem stating the capacity of the barrel -/
theorem barrel_capacity : ∃ (b : Barrel), 
  b.midwayTapRate = 1 / 6 ∧ 
  b.bottomTapRate = 1 / 4 ∧ 
  drawBeer b 24 16 ∧ 
  b.capacity = 16 := by
  sorry

end barrel_capacity_l413_41363


namespace problem_statement_l413_41333

theorem problem_statement (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2014 + b^2013 = 1 := by
sorry

end problem_statement_l413_41333


namespace sally_pokemon_cards_l413_41340

theorem sally_pokemon_cards 
  (initial_cards : ℕ) 
  (dan_cards : ℕ) 
  (total_cards : ℕ) 
  (h1 : initial_cards = 27) 
  (h2 : dan_cards = 41) 
  (h3 : total_cards = 88) : 
  total_cards - (initial_cards + dan_cards) = 20 := by
sorry

end sally_pokemon_cards_l413_41340


namespace last_digit_of_one_over_two_to_fifteen_l413_41392

theorem last_digit_of_one_over_two_to_fifteen (n : ℕ) :
  n = 15 →
  (1 : ℚ) / (2^n : ℚ) * 10^n % 10 = 5 :=
by sorry

end last_digit_of_one_over_two_to_fifteen_l413_41392


namespace tangent_line_to_circle_l413_41362

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∀ x y : ℝ, x + 2*y = r → (x^2 + y^2 = 2*r → (∀ ε > 0, ∃ x' y', x' + 2*y' = r ∧ (x'-x)^2 + (y'-y)^2 < ε^2 ∧ x'^2 + y'^2 ≠ 2*r))) → 
  r = 10 := by
sorry

end tangent_line_to_circle_l413_41362


namespace unbroken_seashells_l413_41330

theorem unbroken_seashells (total : ℕ) (broken : ℕ) (unbroken : ℕ) 
  (h1 : total = 7)
  (h2 : broken = 4)
  (h3 : unbroken = total - broken) :
  unbroken = 3 := by
  sorry

end unbroken_seashells_l413_41330


namespace sphere_volume_from_surface_area_l413_41328

/-- Given a sphere with surface area 256π cm², its volume is (2048/3)π cm³ -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    4 * Real.pi * r^2 = 256 * Real.pi → 
    (4 / 3) * Real.pi * r^3 = (2048 / 3) * Real.pi := by
  sorry

end sphere_volume_from_surface_area_l413_41328


namespace sequence_remainder_l413_41324

def arithmetic_sequence_sum (a₁ : ℤ) (aₙ : ℤ) (n : ℕ) : ℤ :=
  n * (a₁ + aₙ) / 2

theorem sequence_remainder (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 3 →
  aₙ = 315 →
  d = 8 →
  aₙ = a₁ + (n - 1) * d →
  (arithmetic_sequence_sum a₁ aₙ n) % 8 = 4 := by
  sorry

end sequence_remainder_l413_41324


namespace subset_condition_1_subset_condition_2_l413_41344

-- Define the sets M, N1, and N2
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N1 (m : ℝ) : Set ℝ := {x | m - 6 ≤ x ∧ x ≤ 2*m - 1}
def N2 (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for part 1
theorem subset_condition_1 :
  ∀ m : ℝ, (M ⊆ N1 m) ↔ (2 ≤ m ∧ m ≤ 3) :=
by sorry

-- Theorem for part 2
theorem subset_condition_2 :
  ∀ m : ℝ, (N2 m ⊆ M) ↔ (m ≤ 3) :=
by sorry

end subset_condition_1_subset_condition_2_l413_41344


namespace vertical_strips_count_l413_41395

/-- Represents a rectangular grid with a hole -/
structure GridWithHole where
  outer_perimeter : ℕ
  hole_perimeter : ℕ
  horizontal_strips : ℕ

/-- The number of vertical strips in a GridWithHole -/
def vertical_strips (g : GridWithHole) : ℕ :=
  g.outer_perimeter / 2 + g.hole_perimeter / 2 - g.horizontal_strips

theorem vertical_strips_count (g : GridWithHole) 
  (h1 : g.outer_perimeter = 50)
  (h2 : g.hole_perimeter = 32)
  (h3 : g.horizontal_strips = 20) :
  vertical_strips g = 21 := by
  sorry

#eval vertical_strips { outer_perimeter := 50, hole_perimeter := 32, horizontal_strips := 20 }

end vertical_strips_count_l413_41395


namespace complete_square_factorization_l413_41391

theorem complete_square_factorization :
  ∀ x : ℝ, x^2 + 4 + 4*x = (x + 2)^2 := by
  sorry

end complete_square_factorization_l413_41391


namespace smallest_fourth_lucky_number_l413_41360

theorem smallest_fourth_lucky_number : 
  ∃ (n : ℕ), 
    n ≥ 10 ∧ n < 100 ∧
    (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ m < n →
      ¬((57 + 13 + 72 + m) * 5 = 
        (5 + 7 + 1 + 3 + 7 + 2 + (m / 10) + (m % 10)) * 25)) ∧
    (57 + 13 + 72 + n) * 5 = 
      (5 + 7 + 1 + 3 + 7 + 2 + (n / 10) + (n % 10)) * 25 ∧
    n = 38 := by
  sorry

end smallest_fourth_lucky_number_l413_41360


namespace reading_time_difference_l413_41387

/-- Proves the difference in reading time between two people for a given book -/
theorem reading_time_difference
  (xanthia_speed : ℕ)  -- Xanthia's reading speed in pages per hour
  (molly_speed : ℕ)    -- Molly's reading speed in pages per hour
  (book_pages : ℕ)     -- Number of pages in the book
  (h1 : xanthia_speed = 120)
  (h2 : molly_speed = 60)
  (h3 : book_pages = 360) :
  (book_pages / molly_speed - book_pages / xanthia_speed) * 60 = 180 :=
by sorry

end reading_time_difference_l413_41387


namespace linear_function_not_in_third_quadrant_l413_41349

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Given linear function passes through a point -/
def passesThroughPoint (f : LinearFunction) (p : Point) : Prop :=
  p.y = f.m * p.x + f.b

/-- The main theorem to be proved -/
theorem linear_function_not_in_third_quadrant :
  ∀ (p : Point), isInThirdQuadrant p → ¬passesThroughPoint ⟨-5, 2023⟩ p := by
  sorry

end linear_function_not_in_third_quadrant_l413_41349


namespace bags_at_end_of_week_l413_41336

/-- Calculates the total number of bags of cans at the end of the week given daily changes --/
def total_bags_at_end_of_week (
  monday : Real
  ) (tuesday : Real) (wednesday : Real) (thursday : Real) 
    (friday : Real) (saturday : Real) (sunday : Real) : Real :=
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

/-- Theorem stating the total number of bags at the end of the week --/
theorem bags_at_end_of_week : 
  total_bags_at_end_of_week 4 2.5 (-1.25) 0 3.75 (-1.5) 0 = 7.5 := by
  sorry

end bags_at_end_of_week_l413_41336


namespace lcm_problem_l413_41390

theorem lcm_problem (a b : ℕ+) : 
  a = 1491 → Nat.lcm a b = 5964 → b = 4 := by sorry

end lcm_problem_l413_41390


namespace winter_sales_l413_41358

/-- The number of pizzas sold in millions for each season -/
structure PizzaSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The total number of pizzas sold in millions -/
def total_sales (sales : PizzaSales) : ℝ :=
  sales.spring + sales.summer + sales.fall + sales.winter

/-- The given conditions of the problem -/
def pizza_problem (sales : PizzaSales) : Prop :=
  sales.summer = 6 ∧
  sales.spring = 2.5 ∧
  sales.fall = 3.5 ∧
  sales.summer = 0.4 * (total_sales sales)

/-- The theorem to be proved -/
theorem winter_sales (sales : PizzaSales) :
  pizza_problem sales → sales.winter = 3 :=
by
  sorry


end winter_sales_l413_41358


namespace arithmetic_sequence_contains_powers_of_four_l413_41343

theorem arithmetic_sequence_contains_powers_of_four (k : ℕ) :
  ∃ n : ℕ, 3 + 9 * (n - 1) = 3 * 4^k := by
  sorry

end arithmetic_sequence_contains_powers_of_four_l413_41343


namespace quadrant_function_m_range_l413_41361

/-- A proportional function passing through the second and fourth quadrants -/
structure QuadrantFunction where
  m : ℝ
  passes_through_second_fourth : (∀ x y, y = (1 - m) * x → 
    ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0)))

/-- The range of m for a QuadrantFunction -/
theorem quadrant_function_m_range (f : QuadrantFunction) : f.m > 1 := by
  sorry

end quadrant_function_m_range_l413_41361


namespace pizza_slices_l413_41318

/-- The number of slices in a whole pizza -/
def total_slices : ℕ := sorry

/-- The number of slices each person ate -/
def slices_per_person : ℚ := 3/2

/-- The number of people who ate pizza -/
def num_people : ℕ := 2

/-- The number of slices left -/
def slices_left : ℕ := 5

/-- Theorem: The original number of slices in the pizza is 8 -/
theorem pizza_slices : total_slices = 8 := by
  sorry

end pizza_slices_l413_41318


namespace linear_congruence_solution_l413_41335

theorem linear_congruence_solution (x : Int) : 
  (7 * x + 3) % 17 = 2 % 17 ↔ x % 17 = 12 % 17 := by
  sorry

end linear_congruence_solution_l413_41335


namespace quadratic_inequality_range_l413_41353

def quadratic_inequality (a : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 - 2 * a * x + 3 ≤ 0

theorem quadratic_inequality_range :
  {a : ℝ | quadratic_inequality a} = Set.Ici 3 ∪ Set.Iio 0 :=
sorry

end quadratic_inequality_range_l413_41353


namespace excess_of_repeating_over_terminating_l413_41310

/-- The value of the repeating decimal 0.727272... -/
def repeating_72 : ℚ := 72 / 99

/-- The value of the terminating decimal 0.72 -/
def terminating_72 : ℚ := 72 / 100

/-- The fraction by which 0.727272... exceeds 0.72 -/
def excess_fraction : ℚ := 800 / 1099989

theorem excess_of_repeating_over_terminating :
  repeating_72 - terminating_72 = excess_fraction := by
  sorry

end excess_of_repeating_over_terminating_l413_41310


namespace marble_distribution_l413_41366

def valid_divisors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (fun m => n % m = 0 ∧ m > 1 ∧ m < n ∧ n / m > 1)

theorem marble_distribution :
  (valid_divisors 420).card = 22 := by
  sorry

end marble_distribution_l413_41366


namespace subtraction_and_simplification_l413_41369

theorem subtraction_and_simplification :
  (9 : ℚ) / 23 - 5 / 69 = 22 / 69 ∧ 
  ∀ (a b : ℤ), (a : ℚ) / b = 22 / 69 → (a.gcd b = 1 → a = 22 ∧ b = 69) :=
by sorry

end subtraction_and_simplification_l413_41369


namespace basketball_prices_l413_41311

theorem basketball_prices (price_A price_B : ℝ) : 
  price_A = 2 * price_B - 48 →
  9600 / price_A = 7200 / price_B →
  price_A = 96 ∧ price_B = 72 := by
sorry

end basketball_prices_l413_41311


namespace quadratic_equations_common_root_condition_l413_41368

/-- Given three quadratic equations, this theorem states the necessary and sufficient condition
for each equation to have a common root with one another but not all share a single common root. -/
theorem quadratic_equations_common_root_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) : 
  let A := (a₁ + a₂ + a₃) / 2
  ∀ x₁ x₂ x₃ : ℝ,
  (x₁^2 - a₁*x₁ + b₁ = 0 ∧ 
   x₂^2 - a₂*x₂ + b₂ = 0 ∧ 
   x₃^2 - a₃*x₃ + b₃ = 0) →
  ((x₁ = x₂ ∨ x₂ = x₃ ∨ x₃ = x₁) ∧ 
   ¬(x₁ = x₂ ∧ x₂ = x₃)) ↔
  (b₁ = (A - a₂)*(A - a₃) ∧
   b₂ = (A - a₃)*(A - a₁) ∧
   b₃ = (A - a₁)*(A - a₂)) :=
by sorry

end quadratic_equations_common_root_condition_l413_41368


namespace chess_players_count_l413_41365

theorem chess_players_count : ℕ :=
  let total_players : ℕ := 40
  let never_lost_fraction : ℚ := 1/4
  let lost_at_least_once : ℕ := 30
  have h1 : (1 - never_lost_fraction) * total_players = lost_at_least_once := by sorry
  have h2 : never_lost_fraction * total_players + lost_at_least_once = total_players := by sorry
  total_players

end chess_players_count_l413_41365


namespace certain_instrument_count_l413_41345

/-- The number of the certain instrument Charlie owns -/
def x : ℕ := sorry

/-- Charlie's flutes -/
def charlie_flutes : ℕ := 1

/-- Charlie's horns -/
def charlie_horns : ℕ := 2

/-- Carli's flutes -/
def carli_flutes : ℕ := 2 * charlie_flutes

/-- Carli's horns -/
def carli_horns : ℕ := charlie_horns / 2

/-- Total number of instruments owned by Charlie and Carli -/
def total_instruments : ℕ := 7

theorem certain_instrument_count : 
  charlie_flutes + charlie_horns + x + carli_flutes + carli_horns = total_instruments ∧ x = 1 := by
  sorry

end certain_instrument_count_l413_41345


namespace share_distribution_l413_41317

theorem share_distribution (total : ℚ) (a b c : ℚ) : 
  total = 392 →
  a = (1 / 2) * b →
  b = (1 / 2) * c →
  total = a + b + c →
  c = 224 := by
sorry

end share_distribution_l413_41317


namespace michelle_crayon_count_l413_41306

/-- The number of crayons in a box of the first type -/
def crayons_in_first_type : ℕ := 5

/-- The number of crayons in a box of the second type -/
def crayons_in_second_type : ℕ := 12

/-- The number of boxes of the first type -/
def boxes_of_first_type : ℕ := 4

/-- The number of boxes of the second type -/
def boxes_of_second_type : ℕ := 3

/-- The number of crayons missing from one box of the first type -/
def missing_crayons : ℕ := 2

/-- The total number of boxes -/
def total_boxes : ℕ := boxes_of_first_type + boxes_of_second_type

theorem michelle_crayon_count : 
  (boxes_of_first_type * crayons_in_first_type - missing_crayons) + 
  (boxes_of_second_type * crayons_in_second_type) = 54 := by
  sorry

#check michelle_crayon_count

end michelle_crayon_count_l413_41306


namespace polynomial_expansion_sum_l413_41383

theorem polynomial_expansion_sum (m : ℝ) (a a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (1 + m * x)^6 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6) →
  (a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 64) →
  (m = 1 ∨ m = -3) :=
by sorry

end polynomial_expansion_sum_l413_41383


namespace problem_statement_l413_41329

theorem problem_statement (x y z k : ℝ) 
  (h1 : x + 1/y = k)
  (h2 : 2*y + 2/z = k)
  (h3 : 3*z + 3/x = k)
  (h4 : x*y*z = 3) :
  k = 4 := by
  sorry

end problem_statement_l413_41329


namespace translation_theorem_l413_41379

def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

def g (x : ℝ) : ℝ := -2 * (x + 2)^2 + 4 * (x + 2) + 4

theorem translation_theorem :
  ∀ x : ℝ, g x = f (x + 2) + 3 :=
by
  sorry

end translation_theorem_l413_41379


namespace f_value_at_8pi_over_3_l413_41356

def f (x : ℝ) : ℝ := sorry

theorem f_value_at_8pi_over_3 
  (h_even : ∀ x, f (-x) = f x)
  (h_periodic : ∀ x, f (x + π) = f x)
  (h_def : ∀ x, 0 ≤ x → x < π/2 → f x = Real.sqrt 3 * Real.tan x - 1) :
  f (8*π/3) = 2 := by sorry

end f_value_at_8pi_over_3_l413_41356


namespace final_S_value_l413_41396

/-- Calculates the final value of S after executing the loop three times -/
def final_S : ℕ → ℕ → ℕ → ℕ
| 0, s, i => s
| (n + 1), s, i => final_S n (s + i) (i + 2)

theorem final_S_value :
  final_S 3 0 1 = 9 := by
sorry

end final_S_value_l413_41396


namespace train_length_l413_41322

/-- The length of a train given its speed, the speed of a man running in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 50) 
  (h2 : man_speed = 4) 
  (h3 : passing_time = 8) : 
  (train_speed + man_speed) * passing_time * (1000 / 3600) = 120 := by
  sorry

end train_length_l413_41322


namespace total_peaches_l413_41301

theorem total_peaches (initial_baskets : Nat) (initial_peaches_per_basket : Nat)
                      (additional_baskets : Nat) (additional_peaches_per_basket : Nat) :
  initial_baskets = 5 →
  initial_peaches_per_basket = 20 →
  additional_baskets = 4 →
  additional_peaches_per_basket = 25 →
  initial_baskets * initial_peaches_per_basket +
  additional_baskets * additional_peaches_per_basket = 200 := by
  sorry

end total_peaches_l413_41301


namespace find_n_l413_41348

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) + 3 = 15 ∧ n = 3 := by sorry

end find_n_l413_41348


namespace unique_solution_for_k_l413_41377

/-- The equation (2x + 3)/(kx - 2) = x has exactly one solution when k = -4/3 -/
theorem unique_solution_for_k (k : ℚ) : 
  (∃! x, (2 * x + 3) / (k * x - 2) = x) ↔ k = -4/3 := by
  sorry

end unique_solution_for_k_l413_41377


namespace kindergarten_ratio_l413_41351

theorem kindergarten_ratio (boys girls : ℕ) (h1 : boys = 12) (h2 : 2 * girls = 3 * boys) : girls = 18 := by
  sorry

end kindergarten_ratio_l413_41351


namespace negative_square_power_2014_l413_41339

theorem negative_square_power_2014 : -(-(-1)^2)^2014 = -1 := by
  sorry

end negative_square_power_2014_l413_41339


namespace cookies_percentage_increase_l413_41364

def cookies_problem (monday tuesday wednesday : ℕ) : Prop :=
  monday = 5 ∧
  tuesday = 2 * monday ∧
  wednesday > tuesday ∧
  monday + tuesday + wednesday = 29

theorem cookies_percentage_increase :
  ∀ monday tuesday wednesday : ℕ,
  cookies_problem monday tuesday wednesday →
  (wednesday - tuesday : ℚ) / tuesday * 100 = 40 :=
by sorry

end cookies_percentage_increase_l413_41364


namespace quadratic_equality_l413_41398

/-- A quadratic function f(x) = ax^2 + bx + 6 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 6

/-- Theorem: If f(-1) = f(3) for a quadratic function f(x) = ax^2 + bx + 6, then f(2) = 6 -/
theorem quadratic_equality (a b : ℝ) : f a b (-1) = f a b 3 → f a b 2 = 6 := by
  sorry

end quadratic_equality_l413_41398


namespace bran_leftover_amount_l413_41307

/-- Represents Bran's financial situation for a semester --/
structure BranFinances where
  tuitionFee : ℝ
  additionalExpenses : ℝ
  hourlyWage : ℝ
  weeklyHours : ℝ
  scholarshipPercentage : ℝ
  semesterMonths : ℕ

/-- Calculates the amount left after paying expenses --/
def calculateLeftoverAmount (finances : BranFinances) : ℝ :=
  let scholarshipAmount := finances.tuitionFee * finances.scholarshipPercentage
  let tuitionAfterScholarship := finances.tuitionFee - scholarshipAmount
  let totalExpenses := tuitionAfterScholarship + finances.additionalExpenses
  let weeklyEarnings := finances.hourlyWage * finances.weeklyHours
  let totalEarnings := weeklyEarnings * (finances.semesterMonths * 4 : ℝ)
  totalEarnings - totalExpenses

/-- Theorem stating that Bran will have $1,481 left after expenses --/
theorem bran_leftover_amount :
  let finances : BranFinances := {
    tuitionFee := 2500,
    additionalExpenses := 600,
    hourlyWage := 18,
    weeklyHours := 12,
    scholarshipPercentage := 0.45,
    semesterMonths := 4
  }
  calculateLeftoverAmount finances = 1481 := by
  sorry

end bran_leftover_amount_l413_41307


namespace positive_distinct_solution_condition_l413_41320

theorem positive_distinct_solution_condition 
  (a b x y z : ℝ) 
  (eq1 : x + y + z = a) 
  (eq2 : x^2 + y^2 + z^2 = b^2) 
  (eq3 : x * y = z^2) 
  (pos_x : x > 0) 
  (pos_y : y > 0) 
  (pos_z : z > 0) 
  (distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) : 
  b^2 ≥ a^2 / 2 := by
sorry

end positive_distinct_solution_condition_l413_41320


namespace no_integer_solution_l413_41359

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
sorry

end no_integer_solution_l413_41359


namespace s_point_implies_a_value_l413_41338

/-- Definition of an S point for two functions -/
def is_S_point (f g : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = g x₀ ∧ deriv f x₀ = deriv g x₀

/-- The main theorem -/
theorem s_point_implies_a_value (a : ℝ) :
  (∃ x₀ : ℝ, is_S_point (λ x => a * x^2 - 1) (λ x => Real.log (a * x)) x₀) →
  a = 2 / Real.exp 1 :=
sorry

end s_point_implies_a_value_l413_41338


namespace condition_neither_sufficient_nor_necessary_l413_41376

def a (x : ℝ) : Fin 2 → ℝ := ![1, 2 - x]
def b (x : ℝ) : Fin 2 → ℝ := ![2 + x, 3]

def vectors_collinear (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * v i

def norm_squared (v : Fin 2 → ℝ) : ℝ :=
  (v 0) ^ 2 + (v 1) ^ 2

theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ x : ℝ, norm_squared (a x) = 2 → vectors_collinear (a x) (b x)) ∧
  ¬(∀ x : ℝ, vectors_collinear (a x) (b x) → norm_squared (a x) = 2) := by
  sorry

end condition_neither_sufficient_nor_necessary_l413_41376


namespace original_profit_percentage_l413_41372

theorem original_profit_percentage 
  (cost_price : ℝ) 
  (original_selling_price : ℝ) 
  (h1 : original_selling_price > 0) 
  (h2 : cost_price > 0) 
  (h3 : (2 * original_selling_price - cost_price) / cost_price = 2.6) : 
  (original_selling_price - cost_price) / cost_price = 0.8 := by
sorry

end original_profit_percentage_l413_41372


namespace group_purchase_equations_l413_41331

theorem group_purchase_equations (x y : ℤ) : 
  (∀ (z : ℤ), z * x - y = 5 → z = 9) ∧ 
  (∀ (w : ℤ), y - w * x = 4 → w = 6) → 
  (9 * x - 5 = y ∧ 6 * x + 4 = y) := by
  sorry

end group_purchase_equations_l413_41331


namespace min_visible_pairs_l413_41347

/-- Represents the number of birds on the circle -/
def num_birds : ℕ := 155

/-- Represents the maximum arc length for mutual visibility in degrees -/
def visibility_arc : ℝ := 10

/-- Calculates the number of pairs in a group of n birds -/
def pairs_in_group (n : ℕ) : ℕ := n.choose 2

/-- Represents the optimal grouping of birds -/
def optimal_grouping : List ℕ := [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

theorem min_visible_pairs :
  (List.sum (List.map pairs_in_group optimal_grouping) = 270) ∧
  (List.sum optimal_grouping = num_birds) ∧
  (List.length optimal_grouping * visibility_arc ≥ 360) ∧
  (∀ (grouping : List ℕ), 
    (List.sum grouping = num_birds) →
    (List.length grouping * visibility_arc ≥ 360) →
    (List.sum (List.map pairs_in_group grouping) ≥ 270)) := by
  sorry

end min_visible_pairs_l413_41347


namespace tank_problem_solution_l413_41325

def tank_problem (capacity : ℝ) (initial_fill : ℝ) (empty_percent : ℝ) (refill_percent : ℝ) : ℝ :=
  let initial_volume := capacity * initial_fill
  let emptied_volume := initial_volume * empty_percent
  let remaining_volume := initial_volume - emptied_volume
  let refilled_volume := remaining_volume * refill_percent
  remaining_volume + refilled_volume

theorem tank_problem_solution :
  tank_problem 8000 (3/4) 0.4 0.3 = 4680 := by
  sorry

end tank_problem_solution_l413_41325


namespace solution_set_f_range_of_a_l413_41314

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 1|

-- Theorem for the first part of the problem
theorem solution_set_f (x : ℝ) :
  f x ≤ 2 ↔ x ≤ -1 ∨ x ≥ -1/3 :=
sorry

-- Theorem for the second part of the problem
theorem range_of_a (a : ℝ) :
  (∃ b : ℝ, ∀ x : ℝ, |a + b| - |a - b| ≥ f x) ↔ a ≥ 5/4 ∨ a ≤ -5/4 :=
sorry

end solution_set_f_range_of_a_l413_41314


namespace sum_of_coefficients_l413_41315

theorem sum_of_coefficients (x : ℝ) : 
  let p : ℝ → ℝ := λ x => 2 * (4 * x^8 - 5 * x^3 + 6) + 8 * (x^6 + 3 * x^4 - 4)
  p 1 = 10 :=
by sorry

end sum_of_coefficients_l413_41315


namespace polynomial_simplification_l413_41316

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 4 * y^11 + 6 * y^9 + 3 * y^8) =
  15 * y^13 + 2 * y^12 - 8 * y^11 + 18 * y^10 - 3 * y^9 - 6 * y^8 := by
  sorry

end polynomial_simplification_l413_41316


namespace sin_alpha_value_l413_41371

theorem sin_alpha_value (α β : Real) 
  (eq1 : 1 - Real.cos α - Real.cos β + Real.sin α * Real.cos β = 0)
  (eq2 : 1 + Real.cos α - Real.sin β + Real.sin α * Real.cos β = 0) :
  Real.sin α = (1 - Real.sqrt 10) / 3 := by
sorry

end sin_alpha_value_l413_41371


namespace scientific_notation_378300_l413_41367

/-- Proves that 378300 is equal to 3.783 × 10^5 in scientific notation -/
theorem scientific_notation_378300 :
  ∃ (a : ℝ) (n : ℤ), 378300 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.783 ∧ n = 5 :=
by sorry

end scientific_notation_378300_l413_41367


namespace simplify_trigonometric_expression_l413_41389

theorem simplify_trigonometric_expression (α : ℝ) :
  (1 - Real.cos (2 * α)) * Real.cos (π / 4 + 2 * α) / (2 * Real.sin (2 * α) ^ 2 - Real.sin (4 * α)) =
  -Real.sqrt 2 / 4 * Real.tan α := by
  sorry

end simplify_trigonometric_expression_l413_41389


namespace trigonometric_identity_l413_41341

theorem trigonometric_identity (α β : ℝ) 
  (h : (Real.cos α)^4 / (Real.cos β)^2 + (Real.sin α)^4 / (Real.sin β)^2 = 1) :
  (Real.sin β)^4 / (Real.sin α)^2 + (Real.cos β)^4 / (Real.cos α)^2 = 1 := by
sorry

end trigonometric_identity_l413_41341


namespace rectangular_paper_area_l413_41346

theorem rectangular_paper_area (L W : ℝ) 
  (h1 : L + 2*W = 34) 
  (h2 : 2*L + W = 38) : 
  L * W = 140 := by
sorry

end rectangular_paper_area_l413_41346


namespace car_cost_sharing_l413_41337

theorem car_cost_sharing
  (total_cost : ℕ)
  (car_wash_funds : ℕ)
  (initial_friends : ℕ)
  (dropouts : ℕ)
  (h1 : total_cost = 1700)
  (h2 : car_wash_funds = 500)
  (h3 : initial_friends = 6)
  (h4 : dropouts = 1) :
  (total_cost - car_wash_funds) / (initial_friends - dropouts) -
  (total_cost - car_wash_funds) / initial_friends = 40 :=
by sorry

end car_cost_sharing_l413_41337


namespace second_attempt_score_l413_41373

/-- Represents the score of a dart throw attempt -/
structure DartScore where
  score : ℕ
  darts : ℕ
  min_per_dart : ℕ
  max_per_dart : ℕ

/-- The relationship between three dart throw attempts -/
structure ThreeAttempts where
  first : DartScore
  second : DartScore
  third : DartScore
  second_twice_first : first.score * 2 = second.score
  third_1_5_second : second.score * 3 = third.score * 2

/-- The theorem stating the score of the second attempt -/
theorem second_attempt_score (attempts : ThreeAttempts) 
  (h1 : attempts.first.darts = 8)
  (h2 : attempts.second.darts = 8)
  (h3 : attempts.third.darts = 8)
  (h4 : attempts.first.min_per_dart = 3)
  (h5 : attempts.first.max_per_dart = 9)
  (h6 : attempts.second.min_per_dart = 3)
  (h7 : attempts.second.max_per_dart = 9)
  (h8 : attempts.third.min_per_dart = 3)
  (h9 : attempts.third.max_per_dart = 9)
  : attempts.second.score = 48 := by
  sorry

end second_attempt_score_l413_41373


namespace smallest_square_sides_l413_41354

/-- Represents the configuration of three squares arranged as described in the problem -/
structure SquareArrangement where
  small_side : ℝ
  mid_side : ℝ
  large_side : ℝ
  mid_is_larger : mid_side = small_side + 8
  large_is_50 : large_side = 50

/-- The theorem stating the possible side lengths of the smallest square -/
theorem smallest_square_sides (arr : SquareArrangement) : 
  (arr.small_side = 2 ∨ arr.small_side = 32) ↔ 
  (∃ (x : ℝ), x * (x + 8) * 8 = x * (42 - x) * (x + 8)) :=
by sorry

end smallest_square_sides_l413_41354


namespace money_left_after_purchase_l413_41394

def initial_amount : ℕ := 85

def book_prices : List ℕ := [4, 6, 3, 7, 5, 8, 2, 6, 3, 5, 7, 4, 5, 6, 3]

theorem money_left_after_purchase : 
  initial_amount - (book_prices.sum) = 11 := by
  sorry

end money_left_after_purchase_l413_41394


namespace special_number_unique_l413_41375

/-- The unique three-digit positive integer that is one more than a multiple of 3, 4, 5, 6, and 7 -/
def special_number : ℕ := 421

/-- Predicate to check if a number is a three-digit positive integer -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Predicate to check if a number is one more than a multiple of 3, 4, 5, 6, and 7 -/
def is_special (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 3 * k + 1 ∧ n = 4 * k + 1 ∧ n = 5 * k + 1 ∧ n = 6 * k + 1 ∧ n = 7 * k + 1

theorem special_number_unique :
  is_three_digit special_number ∧
  is_special special_number ∧
  ∀ (n : ℕ), is_three_digit n → is_special n → n = special_number :=
sorry

end special_number_unique_l413_41375


namespace junsu_is_winner_l413_41305

-- Define the participants
inductive Participant
| Younghee
| Jimin
| Junsu

-- Define the amount of water drunk by each participant
def water_drunk : Participant → Float
  | Participant.Younghee => 1.4
  | Participant.Jimin => 1.8
  | Participant.Junsu => 2.1

-- Define the winner as the participant who drank the most water
def is_winner (p : Participant) : Prop :=
  ∀ q : Participant, water_drunk p ≥ water_drunk q

-- Theorem stating that Junsu is the winner
theorem junsu_is_winner : is_winner Participant.Junsu := by
  sorry

end junsu_is_winner_l413_41305


namespace perfect_square_binomial_l413_41350

/-- Proves that 25x^2 + 40x + 16 is a perfect square binomial -/
theorem perfect_square_binomial : 
  ∃ (p q : ℝ), ∀ x : ℝ, 25*x^2 + 40*x + 16 = (p*x + q)^2 := by
  sorry

end perfect_square_binomial_l413_41350


namespace running_days_calculation_l413_41397

/-- 
Given:
- Peter runs 3 miles more than Andrew per day
- Andrew runs 2 miles per day
- Their total combined distance is 35 miles

Prove that they have been running for 5 days.
-/
theorem running_days_calculation (andrew_miles : ℕ) (peter_miles : ℕ) (total_miles : ℕ) (days : ℕ) :
  andrew_miles = 2 →
  peter_miles = andrew_miles + 3 →
  total_miles = 35 →
  days * (andrew_miles + peter_miles) = total_miles →
  days = 5 := by
  sorry

#check running_days_calculation

end running_days_calculation_l413_41397


namespace distinct_arrangements_eq_twelve_l413_41380

/-- The number of distinct arrangements of a 4-letter word with one letter repeated twice -/
def distinct_arrangements : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

/-- Theorem stating that the number of distinct arrangements is 12 -/
theorem distinct_arrangements_eq_twelve : distinct_arrangements = 12 := by
  sorry

end distinct_arrangements_eq_twelve_l413_41380


namespace pascal_triangle_interior_sum_l413_41321

def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_triangle_interior_sum :
  (∀ k < 7, interior_sum k ≤ 50) ∧
  interior_sum 7 > 50 ∧
  interior_sum 7 = 62 := by
  sorry

end pascal_triangle_interior_sum_l413_41321


namespace soap_brand_usage_l413_41319

/-- Given a survey of households and their soap usage, prove the number using both brands --/
theorem soap_brand_usage (total : ℕ) (neither : ℕ) (only_A : ℕ) (both : ℕ) :
  total = 300 →
  neither = 80 →
  only_A = 60 →
  total = neither + only_A + both + 3 * both →
  both = 40 := by
sorry

end soap_brand_usage_l413_41319


namespace yellow_candy_bounds_l413_41309

/-- Represents the state of the candy game -/
structure CandyGame where
  total : ℕ
  yellow : ℕ
  colors : ℕ
  yi_turn : Bool

/-- Defines the rules of the candy game -/
def valid_game (game : CandyGame) : Prop :=
  game.total = 22 ∧
  game.colors = 4 ∧
  game.yellow ≤ game.total ∧
  ∀ other_color, other_color ≠ game.yellow → other_color < game.yellow

/-- Defines a valid move in the game -/
def valid_move (before after : CandyGame) : Prop :=
  (before.yi_turn ∧ 
    ((before.total ≥ 2 ∧ after.total = before.total - 2) ∨ 
     (before.total = 1 ∧ after.total = 0))) ∨
  (¬before.yi_turn ∧ 
    (after.total = before.total - before.colors + 1 ∨ after.total = 0))

/-- Defines the end state of the game -/
def game_end (game : CandyGame) : Prop :=
  game.total = 0

/-- Theorem stating the bounds on the number of yellow candies -/
theorem yellow_candy_bounds (initial : CandyGame) :
  valid_game initial →
  (∃ final : CandyGame, 
    game_end final ∧ 
    (∀ intermediate : CandyGame, valid_move initial intermediate → valid_move intermediate final)) →
  8 ≤ initial.yellow ∧ initial.yellow ≤ 9 :=
sorry

end yellow_candy_bounds_l413_41309


namespace three_zeros_implies_a_range_l413_41352

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

/-- The statement that f has three distinct zeros -/
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0

/-- The main theorem: if f has three distinct zeros, then -2 < a < 2 -/
theorem three_zeros_implies_a_range (a : ℝ) :
  has_three_distinct_zeros a → -2 < a ∧ a < 2 :=
by sorry

end three_zeros_implies_a_range_l413_41352


namespace triangle_area_l413_41304

theorem triangle_area (a b c A B C : Real) : 
  a + b = 3 →
  c = Real.sqrt 3 →
  Real.sin (2 * C - Real.pi / 6) = 1 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  1 / 2 * a * b * Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end triangle_area_l413_41304


namespace correct_calculation_l413_41355

theorem correct_calculation (y : ℝ) : -8 * y + 3 * y = -5 * y := by
  sorry

end correct_calculation_l413_41355


namespace prize_cost_l413_41388

theorem prize_cost (total_cost : ℕ) (num_prizes : ℕ) (cost_per_prize : ℕ) 
  (h1 : total_cost = 120)
  (h2 : num_prizes = 6)
  (h3 : total_cost = num_prizes * cost_per_prize) :
  cost_per_prize = 20 := by
  sorry

end prize_cost_l413_41388


namespace reciprocal_operations_l413_41381

theorem reciprocal_operations : ∃! n : ℕ, n = 2 ∧ 
  (¬ (1 / 4 + 1 / 8 = 1 / 12)) ∧
  (¬ (1 / 8 - 1 / 3 = 1 / 5)) ∧
  ((1 / 3) * (1 / 9) = 1 / 27) ∧
  ((1 / 15) / (1 / 3) = 1 / 5) :=
by
  sorry

end reciprocal_operations_l413_41381


namespace triangle_properties_l413_41378

theorem triangle_properties (A B C : Real) (a b c : Real) :
  -- Given conditions
  (b + b * Real.cos A = Real.sqrt 3 * Real.sin B) →
  (a = Real.sqrt 21) →
  (b = 4) →
  -- Conclusions to prove
  (A = π / 3) ∧
  (1/2 * b * c * Real.sin A = 5 * Real.sqrt 3) :=
by sorry

end triangle_properties_l413_41378


namespace inequality_proof_l413_41382

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a + b + 2 * c) + (b * c) / (b + c + 2 * a) + (c * a) / (c + a + 2 * b) ≤ (1 / 4) * (a + b + c) := by
  sorry

end inequality_proof_l413_41382


namespace parabola_translation_l413_41313

/-- Given a parabola y = 2(x+1)^2 - 3, prove that translating it right by 1 unit and up by 3 units results in y = 2x^2 -/
theorem parabola_translation (x y : ℝ) :
  (y = 2 * (x + 1)^2 - 3) →
  (y + 3 = 2 * x^2) := by
sorry

end parabola_translation_l413_41313


namespace min_sum_reciprocal_constraint_min_sum_reciprocal_constraint_equality_l413_41323

theorem min_sum_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 4/y = 1) : x + y ≥ 9 := by
  sorry

theorem min_sum_reciprocal_constraint_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1/x + 4/y = 1) : 
  (x + y = 9) ↔ (x = 3 ∧ y = 6) := by
  sorry

end min_sum_reciprocal_constraint_min_sum_reciprocal_constraint_equality_l413_41323


namespace nancy_picked_three_apples_l413_41385

/-- The number of apples Mike picked -/
def mike_apples : ℝ := 7.0

/-- The number of apples Keith ate -/
def keith_ate : ℝ := 6.0

/-- The number of apples left -/
def apples_left : ℝ := 4.0

/-- The number of apples Nancy picked -/
def nancy_apples : ℝ := 3.0

/-- Theorem stating that Nancy picked 3.0 apples -/
theorem nancy_picked_three_apples : 
  mike_apples + nancy_apples - keith_ate = apples_left :=
by sorry

end nancy_picked_three_apples_l413_41385


namespace comprehensive_score_calculation_l413_41374

theorem comprehensive_score_calculation (initial_score retest_score : ℝ) 
  (initial_weight retest_weight : ℝ) (h1 : initial_score = 400) 
  (h2 : retest_score = 85) (h3 : initial_weight = 0.4) (h4 : retest_weight = 0.6) :
  initial_score * initial_weight + retest_score * retest_weight = 211 := by
  sorry

end comprehensive_score_calculation_l413_41374


namespace smallest_number_with_given_remainders_l413_41386

theorem smallest_number_with_given_remainders :
  ∃ n : ℕ, (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 4 = 3) ∧
  (∀ m : ℕ, m < n → ¬((m % 2 = 1) ∧ (m % 3 = 2) ∧ (m % 4 = 3))) ∧
  n = 11 := by
  sorry

end smallest_number_with_given_remainders_l413_41386


namespace parallel_vectors_x_values_l413_41332

/-- Given two vectors a and b in ℝ², prove that if they are parallel and have the specified components, then x must be 2 or -1. -/
theorem parallel_vectors_x_values (x : ℝ) :
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (x - 1, 2)
  (∃ (k : ℝ), a = k • b) → x = 2 ∨ x = -1 :=
by sorry

end parallel_vectors_x_values_l413_41332


namespace smallest_with_twelve_divisors_l413_41399

-- Define a function to count the number of divisors of a positive integer
def countDivisors (n : ℕ+) : ℕ := sorry

-- Define a function to check if a number has exactly 12 divisors
def hasTwelveDivisors (n : ℕ+) : Prop :=
  countDivisors n = 12

-- Theorem statement
theorem smallest_with_twelve_divisors :
  ∀ n : ℕ+, hasTwelveDivisors n → n ≥ 72 :=
by sorry

end smallest_with_twelve_divisors_l413_41399
