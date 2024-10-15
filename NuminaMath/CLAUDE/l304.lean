import Mathlib

namespace NUMINAMATH_CALUDE_max_expected_games_max_at_half_l304_30418

/-- The expected number of games in a best-of-five series -/
def f (p : ℝ) : ℝ := 6 * p^4 - 12 * p^3 + 3 * p^2 + 3 * p + 3

/-- The theorem stating the maximum value of f(p) -/
theorem max_expected_games :
  ∀ p : ℝ, 0 ≤ p ∧ p ≤ 1 → f p ≤ 33/8 :=
by
  sorry

/-- The theorem stating that the maximum is achieved at p = 1/2 -/
theorem max_at_half :
  f (1/2) = 33/8 :=
by
  sorry

end NUMINAMATH_CALUDE_max_expected_games_max_at_half_l304_30418


namespace NUMINAMATH_CALUDE_ab_plus_cd_equals_zero_l304_30460

theorem ab_plus_cd_equals_zero 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*d - b*c = -1) : 
  a*b + c*d = 0 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_equals_zero_l304_30460


namespace NUMINAMATH_CALUDE_arithmetic_progression_condition_l304_30468

def list : List ℤ := [3, 7, 2, 7, 5, 2]

def mean (x : ℚ) : ℚ := (list.sum + x) / 7

def mode : ℤ := 7

noncomputable def median (x : ℚ) : ℚ :=
  if x ≤ 2 then 3
  else if x < 5 then x
  else 5

theorem arithmetic_progression_condition (x : ℚ) :
  (mode : ℚ) < median x ∧ median x < mean x ∧
  median x - mode = mean x - median x →
  x = 75 / 13 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_condition_l304_30468


namespace NUMINAMATH_CALUDE_equation_solution_l304_30499

theorem equation_solution : 
  ∃ x : ℚ, (1 / 6 + 7 / x = 15 / x + 1 / 15 + 2) ∧ (x = -80 / 19) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l304_30499


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l304_30419

/-- Geometric sequence with common ratio 4 and sum of first three terms equal to 21 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 4 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 4^(n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  geometric_sequence a → ∀ n : ℕ, a n = general_term n := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l304_30419


namespace NUMINAMATH_CALUDE_inequality_proof_l304_30413

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b) + Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2))) ≥ 5/2 ∧
  ((a / (b + c) + b / (c + a) + c / (a + b) + Real.sqrt ((a * b + b * c + c * a) / (a^2 + b^2 + c^2))) = 5/2 ↔ a = b ∧ b = c) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l304_30413


namespace NUMINAMATH_CALUDE_suzanne_reading_difference_l304_30425

/-- Represents the number of pages Suzanne read on Tuesday -/
def pages_tuesday (total_pages monday_pages remaining_pages : ℕ) : ℕ :=
  total_pages - monday_pages - remaining_pages

/-- The difference in pages read between Tuesday and Monday -/
def pages_difference (total_pages monday_pages remaining_pages : ℕ) : ℕ :=
  pages_tuesday total_pages monday_pages remaining_pages - monday_pages

theorem suzanne_reading_difference :
  pages_difference 64 15 18 = 16 := by sorry

end NUMINAMATH_CALUDE_suzanne_reading_difference_l304_30425


namespace NUMINAMATH_CALUDE_aarti_work_completion_time_l304_30487

/-- If Aarti can complete three times a piece of work in 24 days, 
    then she can complete one piece of work in 8 days. -/
theorem aarti_work_completion_time : 
  ∀ (work_time : ℝ), work_time > 0 → 3 * work_time = 24 → work_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_aarti_work_completion_time_l304_30487


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l304_30410

theorem polynomial_division_theorem (a b c : ℚ) : 
  (∀ x, (17 * x^2 - 3 * x + 4) - (a * x^2 + b * x + c) = (5 * x + 6) * (2 * x + 1)) →
  a - b - c = 29 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l304_30410


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l304_30427

theorem sum_of_roots_quadratic (a b : ℝ) : 
  (a^2 - 8*a + 5 = 0) → (b^2 - 8*b + 5 = 0) → (a + b = 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l304_30427


namespace NUMINAMATH_CALUDE_quadratic_rational_solutions_product_l304_30458

theorem quadratic_rational_solutions_product : ∃ (c₁ c₂ : ℕ+), 
  (∀ (c : ℕ+), (∃ (x : ℚ), 5 * x^2 + 11 * x + c.val = 0) ↔ (c = c₁ ∨ c = c₂)) ∧
  c₁.val * c₂.val = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_solutions_product_l304_30458


namespace NUMINAMATH_CALUDE_competition_finish_orders_l304_30467

theorem competition_finish_orders (n : ℕ) (h : n = 5) : 
  Nat.factorial n = 120 := by
  sorry

end NUMINAMATH_CALUDE_competition_finish_orders_l304_30467


namespace NUMINAMATH_CALUDE_work_completion_time_l304_30456

theorem work_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (1 / a = 1 / 6) → (1 / a + 1 / b = 1 / 4) → b = 12 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l304_30456


namespace NUMINAMATH_CALUDE_cube_split_and_stack_l304_30466

/-- The number of millimeters in a meter -/
def mm_per_m : ℕ := 1000

/-- The number of meters in a kilometer -/
def m_per_km : ℕ := 1000

/-- The edge length of the original cube in meters -/
def cube_edge_m : ℕ := 1

/-- The edge length of small cubes in millimeters -/
def small_cube_edge_mm : ℕ := 1

/-- The height of the column in kilometers -/
def column_height_km : ℕ := 1000

theorem cube_split_and_stack :
  (cube_edge_m * mm_per_m)^3 / small_cube_edge_mm = column_height_km * m_per_km * mm_per_m :=
sorry

end NUMINAMATH_CALUDE_cube_split_and_stack_l304_30466


namespace NUMINAMATH_CALUDE_min_unhappiness_theorem_l304_30446

/-- Represents the unhappiness levels of students -/
def unhappiness_levels : List ℝ := List.range 2017

/-- The number of groups to split the students into -/
def num_groups : ℕ := 15

/-- Calculates the minimum possible sum of average unhappiness levels -/
def min_unhappiness (levels : List ℝ) (groups : ℕ) : ℝ :=
  sorry

/-- The theorem stating the minimum unhappiness of the class -/
theorem min_unhappiness_theorem :
  min_unhappiness unhappiness_levels num_groups = 1120.5 := by
  sorry

end NUMINAMATH_CALUDE_min_unhappiness_theorem_l304_30446


namespace NUMINAMATH_CALUDE_least_x_for_divisibility_l304_30481

theorem least_x_for_divisibility (x : ℕ) : 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(3 ∣ 1894 * y)) ∧ (3 ∣ 1894 * x) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_x_for_divisibility_l304_30481


namespace NUMINAMATH_CALUDE_fuel_cost_savings_l304_30436

theorem fuel_cost_savings (old_efficiency : ℝ) (old_fuel_cost : ℝ) 
  (efficiency_improvement : ℝ) (fuel_cost_increase : ℝ) (journey_distance : ℝ)
  (h1 : efficiency_improvement = 0.6)
  (h2 : fuel_cost_increase = 0.25)
  (h3 : journey_distance = 1000) : 
  let new_efficiency := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost := old_fuel_cost * (1 + fuel_cost_increase)
  let old_journey_cost := journey_distance / old_efficiency * old_fuel_cost
  let new_journey_cost := journey_distance / new_efficiency * new_fuel_cost
  let percent_savings := (1 - new_journey_cost / old_journey_cost) * 100
  percent_savings = 21.875 := by
sorry

#eval (1 - (1000 / (1.6 * 1) * 1.25) / (1000 / 1 * 1)) * 100

end NUMINAMATH_CALUDE_fuel_cost_savings_l304_30436


namespace NUMINAMATH_CALUDE_cross_shaded_area_equality_l304_30479

-- Define the rectangle and shaded area properties
def rectangle_length : ℝ := 9
def rectangle_width : ℝ := 8
def shaded_rect1_width : ℝ := 3

-- Define the shaded area as a function of X
def shaded_area (x : ℝ) : ℝ :=
  shaded_rect1_width * rectangle_width + rectangle_length * x - shaded_rect1_width * x

-- Define the total area of the rectangle
def total_area : ℝ := rectangle_length * rectangle_width

-- State the theorem
theorem cross_shaded_area_equality (x : ℝ) :
  shaded_area x = (1 / 2) * total_area → x = 2 := by sorry

end NUMINAMATH_CALUDE_cross_shaded_area_equality_l304_30479


namespace NUMINAMATH_CALUDE_mark_change_factor_l304_30472

theorem mark_change_factor (n : ℕ) (initial_avg final_avg : ℚ) (h1 : n = 25) (h2 : initial_avg = 70) (h3 : final_avg = 140) :
  (n * final_avg) / (n * initial_avg) = 2 :=
sorry

end NUMINAMATH_CALUDE_mark_change_factor_l304_30472


namespace NUMINAMATH_CALUDE_square_difference_of_integers_l304_30483

theorem square_difference_of_integers (a b : ℕ+) 
  (sum_eq : a + b = 70)
  (diff_eq : a - b = 14) :
  a ^ 2 - b ^ 2 = 980 := by
sorry

end NUMINAMATH_CALUDE_square_difference_of_integers_l304_30483


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_52_l304_30492

/-- The maximum area of a rectangle with a perimeter of 52 centimeters is 169 square centimeters. -/
theorem max_area_rectangle_with_perimeter_52 :
  ∀ (length width : ℝ),
  length > 0 → width > 0 →
  2 * (length + width) = 52 →
  length * width ≤ 169 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_52_l304_30492


namespace NUMINAMATH_CALUDE_segment_length_ratio_l304_30465

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PS and PQ = 8QR,
    the length of segment RS is 5/8 of the length of PQ. -/
theorem segment_length_ratio (P Q R S : Real) 
  (h1 : P ≤ R) (h2 : R ≤ S) (h3 : S ≤ Q)  -- Points order on the line
  (h4 : Q - P = 4 * (S - P))  -- PQ = 4PS
  (h5 : Q - P = 8 * (Q - R))  -- PQ = 8QR
  : S - R = 5/8 * (Q - P) := by
  sorry

end NUMINAMATH_CALUDE_segment_length_ratio_l304_30465


namespace NUMINAMATH_CALUDE_square_root_problem_l304_30455

theorem square_root_problem (n : ℝ) (h : Real.sqrt (9 + n) = 8) : n + 2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l304_30455


namespace NUMINAMATH_CALUDE_system_of_inequalities_l304_30470

theorem system_of_inequalities (x : ℝ) : 2*x + 1 > x ∧ x < -3*x + 8 → -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l304_30470


namespace NUMINAMATH_CALUDE_solution_to_equation_l304_30473

theorem solution_to_equation : ∃ x : ℝ, ((18 + x) / 3 + 10) / 5 = 4 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l304_30473


namespace NUMINAMATH_CALUDE_gcd_repeated_digit_numbers_l304_30440

def repeated_digit_number (n : ℕ) : ℕ := n * 1001001001

theorem gcd_repeated_digit_numbers :
  ∃ (d : ℕ), d > 0 ∧ 
  (∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → d ∣ repeated_digit_number n) ∧
  (∀ (m : ℕ), (∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → m ∣ repeated_digit_number n) → m ∣ d) :=
by sorry

end NUMINAMATH_CALUDE_gcd_repeated_digit_numbers_l304_30440


namespace NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l304_30477

/-- Given a point with rectangular coordinates (8, 6) and polar coordinates (r, θ),
    prove that the point with polar coordinates (r³, 3π/2 * θ) has rectangular
    coordinates (-600, -800). -/
theorem polar_to_rectangular_transformation (r θ : ℝ) :
  r * Real.cos θ = 8 ∧ r * Real.sin θ = 6 →
  (r^3 * Real.cos ((3 * Real.pi / 2) * θ) = -600) ∧
  (r^3 * Real.sin ((3 * Real.pi / 2) * θ) = -800) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_transformation_l304_30477


namespace NUMINAMATH_CALUDE_min_queries_theorem_l304_30454

/-- Represents a card with either +1 or -1 written on it -/
inductive Card : Type
| plus_one : Card
| minus_one : Card

/-- Represents a deck of cards -/
def Deck := List Card

/-- Represents a query function that returns the product of three cards -/
def Query := Card → Card → Card → Int

/-- The minimum number of queries needed to determine the product of all cards in a deck -/
def min_queries (n : Nat) (circular : Bool) : Nat :=
  match n with
  | 30 => 10
  | 31 => 11
  | 32 => 12
  | 50 => if circular then 50 else 17  -- 17 is a placeholder for the non-circular case
  | _ => 0  -- placeholder for other cases

/-- Theorem stating the minimum number of queries needed for specific deck sizes -/
theorem min_queries_theorem (d : Deck) (q : Query) :
  (d.length = 30 → min_queries 30 false = 10) ∧
  (d.length = 31 → min_queries 31 false = 11) ∧
  (d.length = 32 → min_queries 32 false = 12) ∧
  (d.length = 50 → min_queries 50 true = 50) :=
sorry

end NUMINAMATH_CALUDE_min_queries_theorem_l304_30454


namespace NUMINAMATH_CALUDE_sin_2theta_value_l304_30406

theorem sin_2theta_value (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1/3) : 
  Real.sin (2 * θ) = -8/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l304_30406


namespace NUMINAMATH_CALUDE_sqrt_expression_eq_zero_quadratic_equation_solutions_l304_30491

-- Problem 1
theorem sqrt_expression_eq_zero (a : ℝ) (h : a > 0) :
  Real.sqrt (8 * a^3) - 4 * a^2 * Real.sqrt (1 / (8 * a)) - 2 * a * Real.sqrt (a / 2) = 0 := by
  sorry

-- Problem 2
theorem quadratic_equation_solutions :
  ∀ x : ℝ, x * (x - 2) + x - 2 = 0 ↔ x = -1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_eq_zero_quadratic_equation_solutions_l304_30491


namespace NUMINAMATH_CALUDE_function_properties_l304_30423

/-- Given a function f(x) = ax - bx^2 where a and b are positive real numbers,
    this theorem states two properties:
    1. If f(x) ≤ 1 for all real x, then a ≤ 2√b.
    2. When b > 1, for x in [0, 1], |f(x)| ≤ 1 if and only if b - 1 ≤ a ≤ 2√b. -/
theorem function_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun x : ℝ => a * x - b * x^2
  (∀ x, f x ≤ 1) → a ≤ 2 * Real.sqrt b ∧
  (b > 1 → (∀ x ∈ Set.Icc 0 1, |f x| ≤ 1) ↔ b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l304_30423


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l304_30412

theorem tangent_equation_solution (x : Real) :
  (Real.tan x * Real.tan (20 * π / 180) + 
   Real.tan (20 * π / 180) * Real.tan (40 * π / 180) + 
   Real.tan (40 * π / 180) * Real.tan x = 1) ↔
  (∃ k : ℤ, x = (30 + 180 * k) * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l304_30412


namespace NUMINAMATH_CALUDE_same_color_probability_problem_die_l304_30478

/-- Represents a 30-sided die with colored sides. -/
structure ColoredDie :=
  (maroon : ℕ)
  (teal : ℕ)
  (cyan : ℕ)
  (sparkly : ℕ)
  (total_sides : ℕ)
  (sum_equals_total : maroon + teal + cyan + sparkly = total_sides)

/-- Calculates the probability of rolling the same color on two identical dice. -/
def same_color_probability (die : ColoredDie) : ℚ :=
  let maroon_prob := (die.maroon : ℚ) / die.total_sides
  let teal_prob := (die.teal : ℚ) / die.total_sides
  let cyan_prob := (die.cyan : ℚ) / die.total_sides
  let sparkly_prob := (die.sparkly : ℚ) / die.total_sides
  maroon_prob ^ 2 + teal_prob ^ 2 + cyan_prob ^ 2 + sparkly_prob ^ 2

/-- The specific 30-sided die described in the problem. -/
def problem_die : ColoredDie :=
  { maroon := 5
    teal := 10
    cyan := 12
    sparkly := 3
    total_sides := 30
    sum_equals_total := by simp }

/-- Theorem stating that the probability of rolling the same color
    on two problem_die is 139/450. -/
theorem same_color_probability_problem_die :
  same_color_probability problem_die = 139 / 450 := by
  sorry

#eval same_color_probability problem_die

end NUMINAMATH_CALUDE_same_color_probability_problem_die_l304_30478


namespace NUMINAMATH_CALUDE_product_of_numbers_l304_30484

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 + y^2 = 200) : x * y = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l304_30484


namespace NUMINAMATH_CALUDE_nanguo_pear_profit_l304_30451

/-- Represents the weight difference from standard and the number of boxes for each difference -/
structure WeightDifference :=
  (difference : ℚ)
  (numBoxes : ℕ)

/-- Calculates the total profit from selling Nanguo pears -/
def calculateProfit (
  numBoxes : ℕ)
  (standardWeight : ℚ)
  (weightDifferences : List WeightDifference)
  (purchasePrice : ℚ)
  (highSellPrice : ℚ)
  (lowSellPrice : ℚ)
  (highSellProportion : ℚ) : ℚ :=
  sorry

theorem nanguo_pear_profit :
  let numBoxes : ℕ := 50
  let standardWeight : ℚ := 10
  let weightDifferences : List WeightDifference := [
    ⟨-2/10, 12⟩, ⟨-1/10, 3⟩, ⟨0, 3⟩, ⟨1/10, 7⟩, ⟨2/10, 15⟩, ⟨3/10, 10⟩
  ]
  let purchasePrice : ℚ := 4
  let highSellPrice : ℚ := 10
  let lowSellPrice : ℚ := 3/2
  let highSellProportion : ℚ := 3/5
  calculateProfit numBoxes standardWeight weightDifferences purchasePrice highSellPrice lowSellPrice highSellProportion = 27216/10
  := by sorry

end NUMINAMATH_CALUDE_nanguo_pear_profit_l304_30451


namespace NUMINAMATH_CALUDE_smallest_number_l304_30421

theorem smallest_number (a b c : ℝ) (ha : a = -0.5) (hb : b = 3) (hc : c = -2) :
  min a (min b c) = c := by sorry

end NUMINAMATH_CALUDE_smallest_number_l304_30421


namespace NUMINAMATH_CALUDE_movie_theater_screens_l304_30400

theorem movie_theater_screens (open_hours : ℕ) (movie_duration : ℕ) (total_movies : ℕ) : 
  open_hours = 8 → movie_duration = 2 → total_movies = 24 → 
  (total_movies * movie_duration) / open_hours = 6 :=
by
  sorry

#check movie_theater_screens

end NUMINAMATH_CALUDE_movie_theater_screens_l304_30400


namespace NUMINAMATH_CALUDE_smaller_number_problem_l304_30444

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 40) : 
  min x y = -5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l304_30444


namespace NUMINAMATH_CALUDE_first_month_sale_proof_l304_30441

/-- Calculates the sale in the first month given the sales for months 2 to 6 and the average sale for 6 months. -/
def first_month_sale (sale2 sale3 sale4 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale2 + sale3 + sale4 + sale5 + sale6)

/-- Proves that the sale in the first month is 8435 given the specified conditions. -/
theorem first_month_sale_proof :
  first_month_sale 8927 8855 9230 8562 6991 8500 = 8435 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_proof_l304_30441


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l304_30480

/-- Calculates the cost of paving a rectangular floor -/
def calculate_paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a 5.5m x 4m room at 850 Rs/m² is 18700 Rs -/
theorem paving_cost_calculation :
  calculate_paving_cost 5.5 4 850 = 18700 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l304_30480


namespace NUMINAMATH_CALUDE_ivan_fate_l304_30409

structure Animal where
  name : String
  always_truth : Bool
  alternating : Bool
  deriving Repr

def Statement := (Bool × Bool)

theorem ivan_fate (bear fox wolf : Animal)
  (h_bear : bear.always_truth = true ∧ bear.alternating = false)
  (h_fox : fox.always_truth = false ∧ fox.alternating = false)
  (h_wolf : wolf.always_truth = false ∧ wolf.alternating = true)
  (statement1 statement2 statement3 : Statement)
  (h_distinct : bear ≠ fox ∧ bear ≠ wolf ∧ fox ≠ wolf)
  : ∃ (animal1 animal2 animal3 : Animal),
    animal1 = fox ∧ animal2 = wolf ∧ animal3 = bear ∧
    (¬statement1.1 ∧ ¬statement1.2) ∧
    (statement2.1 ∧ ¬statement2.2) ∧
    (statement3.1 ∧ statement3.2) :=
by sorry

#check ivan_fate

end NUMINAMATH_CALUDE_ivan_fate_l304_30409


namespace NUMINAMATH_CALUDE_conditionA_not_necessary_nor_sufficient_l304_30462

/-- Condition A: The square root of 1 plus sine of theta equals a -/
def conditionA (θ : Real) (a : Real) : Prop :=
  Real.sqrt (1 + Real.sin θ) = a

/-- Condition B: The sine of half theta plus the cosine of half theta equals a -/
def conditionB (θ : Real) (a : Real) : Prop :=
  Real.sin (θ / 2) + Real.cos (θ / 2) = a

/-- Theorem stating that Condition A is neither necessary nor sufficient for Condition B -/
theorem conditionA_not_necessary_nor_sufficient :
  ¬(∀ θ a, conditionB θ a → conditionA θ a) ∧
  ¬(∀ θ a, conditionA θ a → conditionB θ a) :=
sorry

end NUMINAMATH_CALUDE_conditionA_not_necessary_nor_sufficient_l304_30462


namespace NUMINAMATH_CALUDE_series_sum_l304_30497

noncomputable def series_term (n : ℕ) : ℝ :=
  (2^n : ℝ) / (3^(2^n) + 1)

theorem series_sum : ∑' (n : ℕ), series_term n = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l304_30497


namespace NUMINAMATH_CALUDE_bowling_team_weight_l304_30411

theorem bowling_team_weight (x : ℝ) : 
  let initial_players : ℕ := 7
  let initial_avg_weight : ℝ := 94
  let new_players : ℕ := 2
  let known_new_player_weight : ℝ := 60
  let new_avg_weight : ℝ := 92
  (initial_players * initial_avg_weight + x + known_new_player_weight) / 
    (initial_players + new_players) = new_avg_weight → x = 110 :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_weight_l304_30411


namespace NUMINAMATH_CALUDE_inequality_condition_l304_30452

def f (x : ℝ) := x^2 + 3*x + 2

theorem inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, |x + 2| < b → |f x + 4| < a) ↔ b ≤ a/7 := by sorry

end NUMINAMATH_CALUDE_inequality_condition_l304_30452


namespace NUMINAMATH_CALUDE_evaluate_expression_l304_30403

theorem evaluate_expression : (64 / 0.08) - 2.5 = 797.5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l304_30403


namespace NUMINAMATH_CALUDE_repeating_decimal_is_rational_l304_30443

def repeating_decimal (a b c : ℕ) : ℚ :=
  a + b / (10^c.succ * 99)

theorem repeating_decimal_is_rational (a b c : ℕ) :
  ∃ (p q : ℤ), repeating_decimal a b c = p / q ∧ q ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_is_rational_l304_30443


namespace NUMINAMATH_CALUDE_cone_symmetry_properties_l304_30482

-- Define the types of cones
inductive ConeType
  | Bounded
  | UnboundedSingleNapped
  | UnboundedDoubleNapped

-- Define symmetry properties
structure SymmetryProperties where
  hasAxis : Bool
  hasPlaneBundleThroughAxis : Bool
  hasCentralSymmetry : Bool
  hasPerpendicularPlane : Bool

-- Function to determine symmetry properties based on cone type
def symmetryPropertiesForCone (coneType : ConeType) : SymmetryProperties :=
  match coneType with
  | ConeType.Bounded => {
      hasAxis := true,
      hasPlaneBundleThroughAxis := true,
      hasCentralSymmetry := false,
      hasPerpendicularPlane := false
    }
  | ConeType.UnboundedSingleNapped => {
      hasAxis := true,
      hasPlaneBundleThroughAxis := true,
      hasCentralSymmetry := false,
      hasPerpendicularPlane := false
    }
  | ConeType.UnboundedDoubleNapped => {
      hasAxis := true,
      hasPlaneBundleThroughAxis := true,
      hasCentralSymmetry := true,
      hasPerpendicularPlane := true
    }

theorem cone_symmetry_properties (coneType : ConeType) :
  (coneType = ConeType.Bounded ∨ coneType = ConeType.UnboundedSingleNapped) →
    (symmetryPropertiesForCone coneType).hasCentralSymmetry = false ∧
    (symmetryPropertiesForCone coneType).hasPerpendicularPlane = false
  ∧
  (coneType = ConeType.UnboundedDoubleNapped) →
    (symmetryPropertiesForCone coneType).hasCentralSymmetry = true ∧
    (symmetryPropertiesForCone coneType).hasPerpendicularPlane = true :=
by sorry

end NUMINAMATH_CALUDE_cone_symmetry_properties_l304_30482


namespace NUMINAMATH_CALUDE_at_least_ten_mutual_reports_l304_30469

-- Define the type for spies
def Spy : Type := ℕ

-- Define the total number of spies
def total_spies : ℕ := 20

-- Define the number of colleagues each spy reports on
def reports_per_spy : ℕ := 10

-- Define the reporting relation
def reports_on (s₁ s₂ : Spy) : Prop := sorry

-- State the theorem
theorem at_least_ten_mutual_reports :
  ∃ (mutual_reports : Finset (Spy × Spy)),
    (∀ (pair : Spy × Spy), pair ∈ mutual_reports →
      reports_on pair.1 pair.2 ∧ reports_on pair.2 pair.1) ∧
    mutual_reports.card ≥ 10 := by
  sorry

end NUMINAMATH_CALUDE_at_least_ten_mutual_reports_l304_30469


namespace NUMINAMATH_CALUDE_f_at_2_l304_30486

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem f_at_2 : f 2 = 243 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l304_30486


namespace NUMINAMATH_CALUDE_cost_per_bag_l304_30495

def num_friends : ℕ := 3
def num_bags : ℕ := 5
def payment_per_friend : ℚ := 5

theorem cost_per_bag : 
  (num_friends * payment_per_friend) / num_bags = 3 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_bag_l304_30495


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_corner_cut_pentagon_area_is_804_l304_30449

/-- 
  Represents a pentagon formed by cutting a triangular corner from a rectangle.
  The sides of the pentagon have lengths 12, 15, 18, 30, and 34 in some order.
-/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {12, 15, 18, 30, 34}

/-- The area of the CornerCutPentagon is 804. -/
theorem corner_cut_pentagon_area (p : CornerCutPentagon) : ℕ :=
  804

/-- Proves that the area of the CornerCutPentagon is indeed 804. -/
theorem corner_cut_pentagon_area_is_804 (p : CornerCutPentagon) : 
  corner_cut_pentagon_area p = 804 := by
  sorry

#check corner_cut_pentagon_area_is_804

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_corner_cut_pentagon_area_is_804_l304_30449


namespace NUMINAMATH_CALUDE_candidates_per_state_l304_30485

theorem candidates_per_state : 
  ∀ (x : ℕ), 
    (x * 6 / 100 : ℚ) + 80 = (x * 7 / 100 : ℚ) → 
    x = 8000 := by
  sorry

end NUMINAMATH_CALUDE_candidates_per_state_l304_30485


namespace NUMINAMATH_CALUDE_smallest_constant_two_l304_30433

/-- A function satisfying the given conditions on the interval [0,1] -/
structure SpecialFunction where
  f : Real → Real
  domain : ∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x
  f_one : f 1 = 1
  subadditive : ∀ x y, 0 ≤ x ∧ x ≤ 1 → 0 ≤ y ∧ y ≤ 1 → 0 ≤ x + y ∧ x + y ≤ 1 → 
    f x + f y ≤ f (x + y)

/-- The theorem stating that 2 is the smallest constant c such that f(x) ≤ cx for all x ∈ [0,1] -/
theorem smallest_constant_two (sf : SpecialFunction) : 
  (∀ x, 0 ≤ x ∧ x ≤ 1 → sf.f x ≤ 2 * x) ∧ 
  (∀ c, (∀ x, 0 ≤ x ∧ x ≤ 1 → sf.f x ≤ c * x) → 2 ≤ c) :=
sorry

end NUMINAMATH_CALUDE_smallest_constant_two_l304_30433


namespace NUMINAMATH_CALUDE_quadratic_trinomial_negative_l304_30476

theorem quadratic_trinomial_negative (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 7 * x + 4 * a < 0) ↔ a < -7/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_negative_l304_30476


namespace NUMINAMATH_CALUDE_square_sum_difference_equals_243_l304_30437

theorem square_sum_difference_equals_243 : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 243 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_difference_equals_243_l304_30437


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l304_30428

theorem least_addition_for_divisibility (n : ℕ) : 
  (∃ k : ℕ, k > 0 ∧ (625573 + k) % 3 = 0) → 
  (625573 + 2) % 3 = 0 ∧ ∀ m : ℕ, m < 2 → (625573 + m) % 3 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l304_30428


namespace NUMINAMATH_CALUDE_not_all_points_satisfy_equation_tan_eq_one_not_same_as_pi_over_four_rho_3_same_as_neg_3_l304_30453

-- Define a polar coordinate system
structure PolarCoordinate where
  r : ℝ
  θ : ℝ

-- Define a curve in polar coordinates
def PolarCurve := PolarCoordinate → Prop

-- Statement 1
theorem not_all_points_satisfy_equation (C : PolarCurve) :
  ¬ ∀ (P : PolarCoordinate), C P → (∀ (eq : PolarCoordinate → Prop), (∀ Q, C Q → eq Q) → eq P) :=
sorry

-- Statement 2
theorem tan_eq_one_not_same_as_pi_over_four :
  ∃ (P : PolarCoordinate), (Real.tan P.θ = 1) ≠ (P.θ = π / 4) :=
sorry

-- Statement 3
theorem rho_3_same_as_neg_3 :
  ∀ (P : PolarCoordinate), P.r = 3 ↔ P.r = -3 :=
sorry

end NUMINAMATH_CALUDE_not_all_points_satisfy_equation_tan_eq_one_not_same_as_pi_over_four_rho_3_same_as_neg_3_l304_30453


namespace NUMINAMATH_CALUDE_final_racers_count_l304_30463

def race_elimination (initial_racers : ℕ) : ℕ :=
  let after_first := initial_racers - 10
  let after_second := after_first - (after_first / 3)
  let after_third := after_second - (after_second / 4)
  let after_fourth := after_third - (after_third / 3)
  let after_fifth := after_fourth - (after_fourth / 2)
  after_fifth - (after_fifth * 3 / 4)

theorem final_racers_count :
  race_elimination 200 = 8 := by
  sorry

end NUMINAMATH_CALUDE_final_racers_count_l304_30463


namespace NUMINAMATH_CALUDE_simplified_inverse_sum_l304_30459

theorem simplified_inverse_sum (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ((1 / x^3) + (1 / y^3) + (1 / z^3))⁻¹ = (x^3 * y^3 * z^3) / (y^3 * z^3 + x^3 * z^3 + x^3 * y^3) := by
  sorry

end NUMINAMATH_CALUDE_simplified_inverse_sum_l304_30459


namespace NUMINAMATH_CALUDE_parallel_line_through_point_desired_line_equation_l304_30439

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a point (x, y) lies on a line -/
def on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

theorem parallel_line_through_point (P : ℝ × ℝ) (l : Line) :
  ∃ (l' : Line), parallel l' l ∧ on_line P.1 P.2 l' :=
by sorry

theorem desired_line_equation (P : ℝ × ℝ) (l l' : Line) :
  P = (-1, 3) →
  l = Line.mk 1 (-2) 3 →
  parallel l' l →
  on_line P.1 P.2 l' →
  l' = Line.mk 1 (-2) 7 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_desired_line_equation_l304_30439


namespace NUMINAMATH_CALUDE_online_price_theorem_l304_30422

/-- The price that the buyer observes online for a product sold by a distributor through an online store -/
theorem online_price_theorem (cost : ℝ) (commission_rate : ℝ) (profit_rate : ℝ) 
  (h_cost : cost = 19)
  (h_commission : commission_rate = 0.2)
  (h_profit : profit_rate = 0.2) :
  let distributor_price := cost * (1 + profit_rate)
  let online_price := distributor_price / (1 - commission_rate)
  online_price = 28.5 := by
sorry

end NUMINAMATH_CALUDE_online_price_theorem_l304_30422


namespace NUMINAMATH_CALUDE_number_solution_l304_30407

theorem number_solution : ∃ (x : ℝ), 50 + (x * 12) / (180 / 3) = 51 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l304_30407


namespace NUMINAMATH_CALUDE_movie_theater_tickets_l304_30434

theorem movie_theater_tickets (adult_price child_price total_revenue adult_tickets : ℕ) 
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_revenue = 5100)
  (h4 : adult_tickets = 500) :
  ∃ child_tickets : ℕ, 
    adult_price * adult_tickets + child_price * child_tickets = total_revenue ∧
    adult_tickets + child_tickets = 900 :=
by sorry

end NUMINAMATH_CALUDE_movie_theater_tickets_l304_30434


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l304_30401

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The terms a_1 + 1, a_3 + 2, and a_5 + 3 form a geometric sequence with ratio q -/
def GeometricSubsequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 3 + 2) = (a 1 + 1) * q ∧ (a 5 + 3) = (a 3 + 2) * q

theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : ArithmeticSequence a) (h2 : GeometricSubsequence a q) : q = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l304_30401


namespace NUMINAMATH_CALUDE_convex_ngon_division_constant_l304_30488

/-- A convex n-gon can be divided into triangles using non-intersecting diagonals -/
structure ConvexNGonDivision (n : ℕ) where
  (n_ge_3 : n ≥ 3)
  (triangles : ℕ)
  (diagonals : ℕ)

/-- The number of triangles and diagonals in any division of a convex n-gon is constant -/
theorem convex_ngon_division_constant (n : ℕ) (d : ConvexNGonDivision n) :
  d.triangles = n - 2 ∧ d.diagonals = n - 3 :=
sorry

end NUMINAMATH_CALUDE_convex_ngon_division_constant_l304_30488


namespace NUMINAMATH_CALUDE_sum_in_base5_l304_30474

/-- Converts a base 5 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 5 * acc + d) 0

/-- Converts a decimal number to its base 5 representation as a list of digits -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The main theorem to prove -/
theorem sum_in_base5 :
  toBase5 (toDecimal [2, 1, 3] + toDecimal [3, 2, 4] + toDecimal [1, 4, 1]) = [1, 3, 3, 3] :=
sorry

end NUMINAMATH_CALUDE_sum_in_base5_l304_30474


namespace NUMINAMATH_CALUDE_sets_intersection_and_union_l304_30408

def A (x : ℝ) : Set ℝ := {x^2, 2*x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem sets_intersection_and_union :
  ∃ x : ℝ, (B x ∩ A x = {9}) ∧ 
           (x = -3) ∧ 
           (A x ∪ B x = {-8, -7, -4, 4, 9}) := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_and_union_l304_30408


namespace NUMINAMATH_CALUDE_yoongi_has_smaller_number_l304_30461

theorem yoongi_has_smaller_number : 
  let jungkook_number := 6 + 3
  let yoongi_number := 4
  yoongi_number < jungkook_number := by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_smaller_number_l304_30461


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l304_30493

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 1

-- Define the interval
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (a b : ℝ), a ∈ interval ∧ b ∈ interval ∧
  (∀ x ∈ interval, f x ≤ f a) ∧
  (∀ x ∈ interval, f b ≤ f x) ∧
  f a = 1 ∧ f b = -2 :=
sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l304_30493


namespace NUMINAMATH_CALUDE_expression_evaluation_l304_30435

theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := 1
  ((2*x + y)^2 - y*(y + 4*x) - 8*x) / (-2*x) = 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l304_30435


namespace NUMINAMATH_CALUDE_shirt_price_reduction_l304_30450

theorem shirt_price_reduction (original_price : ℝ) (h : original_price > 0) :
  let first_reduction := 0.9 * original_price
  let second_reduction := 0.9 * first_reduction
  second_reduction = 0.81 * original_price :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_reduction_l304_30450


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l304_30447

theorem polynomial_multiplication (x : ℝ) :
  (x^4 + 20*x^2 + 400) * (x^2 - 20) = x^6 - 8000 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l304_30447


namespace NUMINAMATH_CALUDE_set_operations_l304_30429

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem set_operations :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 3}) ∧
  (A ∪ B = {x | x > 1}) ∧
  (A ∩ (Set.univ \ B) = {x | 1 < x ∧ x < 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l304_30429


namespace NUMINAMATH_CALUDE_quiz_variance_is_64_l304_30404

/-- Represents a multiple-choice quiz -/
structure Quiz where
  num_questions : ℕ
  options_per_question : ℕ
  points_per_correct : ℕ
  total_points : ℕ
  correct_probability : ℝ

/-- Calculates the variance of a student's score in the quiz -/
def quiz_score_variance (q : Quiz) : ℝ :=
  q.num_questions * q.correct_probability * (1 - q.correct_probability) * q.points_per_correct^2

/-- Theorem stating that the variance of the student's score in the given quiz is 64 -/
theorem quiz_variance_is_64 : 
  let q : Quiz := {
    num_questions := 25,
    options_per_question := 4,
    points_per_correct := 4,
    total_points := 100,
    correct_probability := 0.8
  }
  quiz_score_variance q = 64 := by
  sorry

end NUMINAMATH_CALUDE_quiz_variance_is_64_l304_30404


namespace NUMINAMATH_CALUDE_expression_value_l304_30490

theorem expression_value (x : ℝ) (h : x = 4) : 3 * (3 * x - 2)^2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l304_30490


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_product_l304_30489

def x : ℕ := 2 * 4 * 6 * 8 * 10 * 12 * 14 * 16 * 18 * 20

theorem greatest_prime_factor_of_product (x : ℕ) : 
  x = 2 * 4 * 6 * 8 * 10 * 12 * 14 * 16 * 18 * 20 →
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (18 * x * 14 * x) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (18 * x * 14 * x) → q ≤ p ∧ p = 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_product_l304_30489


namespace NUMINAMATH_CALUDE_vertical_asymptotes_sum_l304_30432

theorem vertical_asymptotes_sum (p q : ℚ) : 
  (∀ x, 4 * x^2 + 7 * x + 3 = 0 ↔ x = p ∨ x = q) →
  p + q = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_vertical_asymptotes_sum_l304_30432


namespace NUMINAMATH_CALUDE_point_above_x_axis_l304_30448

theorem point_above_x_axis (a : ℝ) : 
  (a > 0) → (a = Real.sqrt 3) → ∃ (x y : ℝ), x = -2 ∧ y = a ∧ y > 0 :=
by sorry

end NUMINAMATH_CALUDE_point_above_x_axis_l304_30448


namespace NUMINAMATH_CALUDE_correct_purchase_combinations_l304_30438

/-- The number of oreo flavors -/
def oreo_flavors : ℕ := 7

/-- The number of milk flavors -/
def milk_flavors : ℕ := 4

/-- The total number of product flavors -/
def total_flavors : ℕ := oreo_flavors + milk_flavors

/-- The total number of products they purchase -/
def total_products : ℕ := 4

/-- The number of ways Alpha and Beta could have left the store with 4 products collectively -/
def purchase_combinations : ℕ := sorry

theorem correct_purchase_combinations :
  purchase_combinations = 4054 := by sorry

end NUMINAMATH_CALUDE_correct_purchase_combinations_l304_30438


namespace NUMINAMATH_CALUDE_aluminium_hydroxide_weight_l304_30415

/-- The atomic weight of Aluminium in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The number of moles of Aluminium hydroxide -/
def num_moles : ℝ := 4

/-- The molecular weight of Aluminium hydroxide (Al(OH)₃) in g/mol -/
def molecular_weight_AlOH3 : ℝ := atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H

/-- The total weight of the given number of moles of Aluminium hydroxide in grams -/
def total_weight : ℝ := num_moles * molecular_weight_AlOH3

theorem aluminium_hydroxide_weight :
  total_weight = 312.04 := by sorry

end NUMINAMATH_CALUDE_aluminium_hydroxide_weight_l304_30415


namespace NUMINAMATH_CALUDE_grape_juice_mixture_l304_30414

theorem grape_juice_mixture (initial_volume : ℝ) (initial_percentage : ℝ) (added_volume : ℝ) :
  initial_volume = 50 →
  initial_percentage = 0.1 →
  added_volume = 10 →
  let initial_grape_juice := initial_volume * initial_percentage
  let total_grape_juice := initial_grape_juice + added_volume
  let final_volume := initial_volume + added_volume
  let final_percentage := total_grape_juice / final_volume
  final_percentage = 0.25 := by sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_l304_30414


namespace NUMINAMATH_CALUDE_min_m_for_inequality_l304_30494

theorem min_m_for_inequality : 
  (∃ (m : ℝ), ∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → x^2 - m ≤ 1) ∧ 
  (∀ (m' : ℝ), (∀ (x : ℝ), x ∈ Set.Icc (-1) 2 → x^2 - m' ≤ 1) → m' ≥ 3) :=
by sorry


end NUMINAMATH_CALUDE_min_m_for_inequality_l304_30494


namespace NUMINAMATH_CALUDE_calculate_principal_l304_30431

/-- Given a simple interest, interest rate, and time period, calculate the principal amount. -/
theorem calculate_principal (simple_interest rate time : ℝ) :
  simple_interest = 4020.75 →
  rate = 0.0875 →
  time = 5.5 →
  simple_interest = (8355.00 * rate * time) := by
  sorry

end NUMINAMATH_CALUDE_calculate_principal_l304_30431


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l304_30416

theorem two_digit_numbers_problem (A B : ℕ) : 
  A ≥ 10 ∧ A ≤ 99 ∧ B ≥ 10 ∧ B ≤ 99 →
  (100 * A + B) / B = 121 →
  (100 * B + A) / A = 84 ∧ (100 * B + A) % A = 14 →
  A = 42 ∧ B = 35 := by
sorry

end NUMINAMATH_CALUDE_two_digit_numbers_problem_l304_30416


namespace NUMINAMATH_CALUDE_smallest_angle_in_right_triangle_l304_30430

theorem smallest_angle_in_right_triangle (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 90 → a / b = 5 / 4 → min a b = 40 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_in_right_triangle_l304_30430


namespace NUMINAMATH_CALUDE_rectangle_area_l304_30496

/-- The area of a rectangle with width 5.4 meters and height 2.5 meters is 13.5 square meters. -/
theorem rectangle_area : 
  let width : Real := 5.4
  let height : Real := 2.5
  width * height = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l304_30496


namespace NUMINAMATH_CALUDE_polynomial_square_l304_30402

theorem polynomial_square (a b : ℚ) : 
  (∃ p q : ℚ, ∀ x : ℚ, x^4 + 3*x^3 + x^2 + a*x + b = (x^2 + p*x + q)^2) → 
  b = 25/64 := by
sorry

end NUMINAMATH_CALUDE_polynomial_square_l304_30402


namespace NUMINAMATH_CALUDE_sum_of_roots_l304_30420

theorem sum_of_roots (p : ℝ) : 
  let q : ℝ := p^2 - 1
  let f : ℝ → ℝ := λ x ↦ x^2 - p*x + q
  ∃ r s : ℝ, f r = 0 ∧ f s = 0 ∧ r + s = p :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l304_30420


namespace NUMINAMATH_CALUDE_license_plate_count_l304_30442

/-- The number of possible digits (0 to 9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A to Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 6

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_count + 1

/-- Calculates the total number of possible distinct license plates -/
def total_license_plates : ℕ :=
  block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count :
  total_license_plates = 122504000 :=
sorry

end NUMINAMATH_CALUDE_license_plate_count_l304_30442


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_40_l304_30498

theorem gcd_lcm_product_24_40 : Nat.gcd 24 40 * Nat.lcm 24 40 = 960 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_40_l304_30498


namespace NUMINAMATH_CALUDE_sequence_2014_term_l304_30445

/-- A positive sequence satisfying the given recurrence relation -/
def PositiveSequence (a : ℕ+ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, n * a (n + 1) = (n + 1) * a n ∧ 0 < a n

/-- The 2014th term of the sequence is equal to 2014 -/
theorem sequence_2014_term (a : ℕ+ → ℝ) (h : PositiveSequence a) : a 2014 = 2014 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2014_term_l304_30445


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l304_30471

-- Define the quadratic equation
def quadratic_equation (x m : ℚ) : Prop :=
  3 * x^2 - 7 * x + m = 0

-- Define the condition for exactly one solution
def has_exactly_one_solution (m : ℚ) : Prop :=
  ∃! x, quadratic_equation x m

-- Theorem statement
theorem unique_solution_quadratic :
  ∀ m : ℚ, has_exactly_one_solution m → m = 49 / 12 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l304_30471


namespace NUMINAMATH_CALUDE_ben_lighter_than_carl_l304_30426

/-- Given the weights of several people and their relationships, prove that Ben is 16 pounds lighter than Carl. -/
theorem ben_lighter_than_carl (al ben carl ed : ℕ) : 
  al = ben + 25 →  -- Al is 25 pounds heavier than Ben
  ed = 146 →       -- Ed weighs 146 pounds
  al = ed + 38 →   -- Ed is 38 pounds lighter than Al
  carl = 175 →     -- Carl weighs 175 pounds
  carl - ben = 16  -- Ben is 16 pounds lighter than Carl
:= by sorry

end NUMINAMATH_CALUDE_ben_lighter_than_carl_l304_30426


namespace NUMINAMATH_CALUDE_odds_to_probability_losing_l304_30464

-- Define the odds of winning
def odds_winning : ℚ := 5 / 6

-- Define the probability of losing
def prob_losing : ℚ := 6 / 11

-- Theorem statement
theorem odds_to_probability_losing : 
  odds_winning = 5 / 6 → prob_losing = 6 / 11 := by
  sorry

end NUMINAMATH_CALUDE_odds_to_probability_losing_l304_30464


namespace NUMINAMATH_CALUDE_sqrt_square_eq_x_for_nonnegative_l304_30457

theorem sqrt_square_eq_x_for_nonnegative (x : ℝ) (h : x ≥ 0) : (Real.sqrt x)^2 = x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_x_for_nonnegative_l304_30457


namespace NUMINAMATH_CALUDE_correct_result_l304_30417

def add_subtract_round (a b c : ℕ) : ℕ :=
  let sum := a + b - c
  (sum + 5) / 10 * 10

theorem correct_result : add_subtract_round 53 28 5 = 80 := by
  sorry

end NUMINAMATH_CALUDE_correct_result_l304_30417


namespace NUMINAMATH_CALUDE_local_tax_deduction_l304_30405

/-- Proves that given an hourly wage of 25 dollars and a 2% local tax rate, 
    the amount deducted for local taxes is 50 cents per hour. -/
theorem local_tax_deduction (hourly_wage : ℝ) (tax_rate : ℝ) :
  hourly_wage = 25 ∧ tax_rate = 0.02 →
  (hourly_wage * tax_rate * 100 : ℝ) = 50 := by
  sorry

#check local_tax_deduction

end NUMINAMATH_CALUDE_local_tax_deduction_l304_30405


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l304_30424

theorem imaginary_part_of_z : Complex.im ((1 + Complex.I) / Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l304_30424


namespace NUMINAMATH_CALUDE_movie_sale_price_is_10000_l304_30475

/-- The sale price of a movie given costs and profit -/
def movie_sale_price (actor_cost food_cost_per_person equipment_cost_multiplier num_people profit : ℕ) : ℕ :=
  let food_cost := food_cost_per_person * num_people
  let equipment_cost := equipment_cost_multiplier * (actor_cost + food_cost)
  let total_cost := actor_cost + food_cost + equipment_cost
  total_cost + profit

/-- Theorem stating the sale price of the movie is $10000 -/
theorem movie_sale_price_is_10000 :
  movie_sale_price 1200 3 2 50 5950 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_movie_sale_price_is_10000_l304_30475
