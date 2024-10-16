import Mathlib

namespace NUMINAMATH_CALUDE_pencil_cost_l1729_172948

/-- Given that 120 pencils cost $36, prove that 3000 pencils cost $900 -/
theorem pencil_cost (cost_120 : ℕ) (quantity : ℕ) (h1 : cost_120 = 36) (h2 : quantity = 3000) :
  (cost_120 * quantity) / 120 = 900 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l1729_172948


namespace NUMINAMATH_CALUDE_work_duration_problem_l1729_172904

/-- The problem of determining how long a worker worked on a task before another worker finished it. -/
theorem work_duration_problem 
  (W : ℝ) -- Total work
  (x_rate : ℝ) -- x's work rate per day
  (y_rate : ℝ) -- y's work rate per day
  (y_finish_time : ℝ) -- Time y took to finish the remaining work
  (hx : x_rate = W / 40) -- x's work rate condition
  (hy : y_rate = W / 20) -- y's work rate condition
  (h_finish : y_finish_time = 16) -- y's finish time condition
  : ∃ (d : ℝ), d * x_rate + y_finish_time * y_rate = W ∧ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_duration_problem_l1729_172904


namespace NUMINAMATH_CALUDE_function_inequality_implies_bound_l1729_172984

theorem function_inequality_implies_bound (a : ℝ) : 
  (∃ x : ℝ, 4 - x^2 ≥ |x - a| + a) → a ≤ 17/8 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_bound_l1729_172984


namespace NUMINAMATH_CALUDE_line_equation_fourth_quadrant_triangle_l1729_172931

/-- Given a line passing through (-b, 0) and cutting a triangle with area T in the fourth quadrant,
    prove that its equation is 2Tx - b²y - 2bT = 0 --/
theorem line_equation_fourth_quadrant_triangle (b T : ℝ) (h₁ : b > 0) (h₂ : T > 0) : 
  ∃ (m c : ℝ), ∀ (x y : ℝ),
    (x = -b ∧ y = 0) ∨ (x ≥ 0 ∧ y ≤ 0 ∧ y = m * x + c) →
    (1/2 * b * (-y)) = T →
    2 * T * x - b^2 * y - 2 * b * T = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_fourth_quadrant_triangle_l1729_172931


namespace NUMINAMATH_CALUDE_negation_equivalence_l1729_172949

theorem negation_equivalence (m : ℤ) : 
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1729_172949


namespace NUMINAMATH_CALUDE_cabbage_area_l1729_172960

theorem cabbage_area (garden_area_this_year garden_area_last_year : ℝ) 
  (cabbages_this_year cabbages_last_year : ℕ) :
  (garden_area_this_year = cabbages_this_year) →
  (garden_area_this_year = garden_area_last_year + 199) →
  (cabbages_this_year = 10000) →
  (∃ x y : ℝ, garden_area_last_year = x^2 ∧ garden_area_this_year = y^2) →
  (garden_area_this_year / cabbages_this_year = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_cabbage_area_l1729_172960


namespace NUMINAMATH_CALUDE_log_xy_value_l1729_172977

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^2) = 1) (h2 : Real.log (x^2 * y) = 1) :
  Real.log (x * y) = 2/3 := by sorry

end NUMINAMATH_CALUDE_log_xy_value_l1729_172977


namespace NUMINAMATH_CALUDE_intersection_M_N_l1729_172996

-- Define the sets M and N
def M : Set ℝ := {x | x / (x - 1) ≥ 0}
def N : Set ℝ := {y | ∃ x, y = 3 * x^2 + 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1729_172996


namespace NUMINAMATH_CALUDE_verify_conditions_max_boxes_A_l1729_172991

/-- Represents the price of a box of paint model A in yuan -/
def price_A : ℕ := 24

/-- Represents the price of a box of paint model B in yuan -/
def price_B : ℕ := 16

/-- Represents the total number of boxes to be purchased -/
def total_boxes : ℕ := 200

/-- Represents the maximum total cost in yuan -/
def max_cost : ℕ := 3920

/-- Verification of the given conditions -/
theorem verify_conditions : 
  price_A + 2 * price_B = 56 ∧ 
  2 * price_A + price_B = 64 := by sorry

/-- Theorem stating the maximum number of boxes of paint A that can be purchased -/
theorem max_boxes_A : 
  (∀ m : ℕ, m ≤ total_boxes → 
    m * price_A + (total_boxes - m) * price_B ≤ max_cost → 
    m ≤ 90) ∧ 
  90 * price_A + (total_boxes - 90) * price_B ≤ max_cost := by sorry

end NUMINAMATH_CALUDE_verify_conditions_max_boxes_A_l1729_172991


namespace NUMINAMATH_CALUDE_gcd_4557_1953_5115_l1729_172982

theorem gcd_4557_1953_5115 : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_4557_1953_5115_l1729_172982


namespace NUMINAMATH_CALUDE_parabola_coefficients_l1729_172926

/-- A parabola with equation y = ax^2 + bx + c, vertex at (5, -1), 
    vertical axis of symmetry, and passing through (2, 8) -/
def Parabola (a b c : ℝ) : Prop :=
  (∀ x y : ℝ, y = a * x^2 + b * x + c) ∧
  (a * 5^2 + b * 5 + c = -1) ∧
  (∀ x : ℝ, a * (x - 5)^2 + (a * 5^2 + b * 5 + c) = a * x^2 + b * x + c) ∧
  (a * 2^2 + b * 2 + c = 8)

/-- The values of a, b, and c for the given parabola are 1, -10, and 24 respectively -/
theorem parabola_coefficients : 
  ∃ a b c : ℝ, Parabola a b c ∧ a = 1 ∧ b = -10 ∧ c = 24 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficients_l1729_172926


namespace NUMINAMATH_CALUDE_scallops_per_pound_is_eight_l1729_172986

/-- The number of jumbo scallops that weigh one pound -/
def scallops_per_pound : ℕ := by sorry

/-- The cost of one pound of jumbo scallops in dollars -/
def cost_per_pound : ℕ := 24

/-- The number of scallops paired per person -/
def scallops_per_person : ℕ := 2

/-- The number of people Nate is cooking for -/
def number_of_people : ℕ := 8

/-- The total cost of scallops for Nate in dollars -/
def total_cost : ℕ := 48

theorem scallops_per_pound_is_eight :
  scallops_per_pound = 8 := by sorry

end NUMINAMATH_CALUDE_scallops_per_pound_is_eight_l1729_172986


namespace NUMINAMATH_CALUDE_no_real_solution_log_equation_l1729_172918

theorem no_real_solution_log_equation :
  ¬ ∃ (x : ℝ), (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 8*x + 15)) ∧
               (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 8*x + 15 > 0) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_log_equation_l1729_172918


namespace NUMINAMATH_CALUDE_john_weight_is_250_l1729_172951

/-- The weight bench capacity in pounds -/
def bench_capacity : ℝ := 1000

/-- The safety margin percentage -/
def safety_margin : ℝ := 0.20

/-- The weight John puts on the bar in pounds -/
def bar_weight : ℝ := 550

/-- John's weight in pounds -/
def john_weight : ℝ := bench_capacity * (1 - safety_margin) - bar_weight

theorem john_weight_is_250 : john_weight = 250 := by
  sorry

end NUMINAMATH_CALUDE_john_weight_is_250_l1729_172951


namespace NUMINAMATH_CALUDE_original_recipe_eggs_l1729_172980

/-- The number of eggs needed for an eight-person cake -/
def eggs_for_eight : ℕ := 3 + 1

/-- The number of people the original recipe serves -/
def original_servings : ℕ := 4

/-- The number of people Tyler wants to serve -/
def target_servings : ℕ := 8

/-- The number of eggs required for the original recipe -/
def eggs_for_original : ℕ := eggs_for_eight / 2

theorem original_recipe_eggs :
  eggs_for_original = 2 :=
sorry

end NUMINAMATH_CALUDE_original_recipe_eggs_l1729_172980


namespace NUMINAMATH_CALUDE_system_unique_solution_l1729_172956

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  4 * Real.sqrt y = x - a ∧ y^2 - x^2 + 2*y - 4*x - 3 = 0

-- Define the set of a values for which the system has a unique solution
def unique_solution_set : Set ℝ := {a | a < -5 ∨ a > -1}

-- Theorem statement
theorem system_unique_solution :
  ∀ a : ℝ, (∃! (x y : ℝ), system x y a) ↔ a ∈ unique_solution_set :=
sorry

end NUMINAMATH_CALUDE_system_unique_solution_l1729_172956


namespace NUMINAMATH_CALUDE_harvest_rent_proof_l1729_172964

/-- The total rent paid during the harvest season. -/
def total_rent (weekly_rent : ℕ) (weeks : ℕ) : ℕ :=
  weekly_rent * weeks

/-- Proof that the total rent paid during the harvest season is $527,292. -/
theorem harvest_rent_proof :
  total_rent 388 1359 = 527292 := by
  sorry

end NUMINAMATH_CALUDE_harvest_rent_proof_l1729_172964


namespace NUMINAMATH_CALUDE_prime_pair_divisibility_l1729_172993

theorem prime_pair_divisibility (n p : ℕ+) : 
  Nat.Prime p.val ∧ 
  n.val ≤ 2 * p.val ∧ 
  (n.val^(p.val - 1) ∣ (p.val - 1)^n.val + 1) → 
  ((n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
sorry

end NUMINAMATH_CALUDE_prime_pair_divisibility_l1729_172993


namespace NUMINAMATH_CALUDE_sum_of_digits_power_product_l1729_172970

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem sum_of_digits_power_product :
  sumOfDigits (2^2010 * 5^2012 * 7) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_product_l1729_172970


namespace NUMINAMATH_CALUDE_polynomial_factors_l1729_172905

/-- The polynomial 3x^4 - hx^2 + kx - 7 has x+1 and x-3 as factors if and only if h = 124/3 and k = 136/3 -/
theorem polynomial_factors (h k : ℚ) : 
  (∀ x : ℚ, (x + 1) * (x - 3) ∣ (3 * x^4 - h * x^2 + k * x - 7)) ↔ 
  (h = 124/3 ∧ k = 136/3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factors_l1729_172905


namespace NUMINAMATH_CALUDE_same_duration_trips_l1729_172913

/-- Proves that two trips with given distances and speed ratio have the same duration -/
theorem same_duration_trips (distance1 : ℝ) (distance2 : ℝ) (speed_ratio : ℝ) 
  (h1 : distance1 = 90) 
  (h2 : distance2 = 360) 
  (h3 : speed_ratio = 4) : 
  (distance1 / 1) = (distance2 / speed_ratio) := by
  sorry

#check same_duration_trips

end NUMINAMATH_CALUDE_same_duration_trips_l1729_172913


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1729_172914

/-- Two vectors in R² are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a and b in R², if a is perpendicular to b, 
    and a = (1, 2) and b = (x, 1), then x = -2 -/
theorem perpendicular_vectors_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  perpendicular a b → x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1729_172914


namespace NUMINAMATH_CALUDE_minimum_excellence_rate_l1729_172939

theorem minimum_excellence_rate (total : ℕ) (math_rate : ℚ) (chinese_rate : ℚ) 
  (h_math : math_rate = 70 / 100)
  (h_chinese : chinese_rate = 75 / 100)
  (h_total : total > 0) :
  ∃ (both_rate : ℚ), 
    both_rate ≥ 45 / 100 ∧ 
    both_rate * total ≤ math_rate * total ∧ 
    both_rate * total ≤ chinese_rate * total :=
sorry

end NUMINAMATH_CALUDE_minimum_excellence_rate_l1729_172939


namespace NUMINAMATH_CALUDE_max_dominoes_8x8_10removed_l1729_172906

/-- Represents a chessboard with some squares removed -/
structure Chessboard :=
  (size : Nat)
  (removed : Nat)

/-- Calculates the maximum number of dominoes that can be placed on a chessboard -/
def max_dominoes (board : Chessboard) : Nat :=
  let remaining := board.size * board.size - board.removed
  let worst_case_color := min (board.size * board.size / 2) (remaining - (board.size * board.size / 2 - board.removed))
  worst_case_color

/-- Theorem stating the maximum number of dominoes on an 8x8 chessboard with 10 squares removed -/
theorem max_dominoes_8x8_10removed :
  max_dominoes { size := 8, removed := 10 } = 23 := by
  sorry

end NUMINAMATH_CALUDE_max_dominoes_8x8_10removed_l1729_172906


namespace NUMINAMATH_CALUDE_five_player_tournament_l1729_172915

/-- The number of games in a tournament where each player plays every other player once -/
def tournament_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a tournament with 5 players where each player plays against every other player
    exactly once, the total number of games played is 10. -/
theorem five_player_tournament : tournament_games 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_five_player_tournament_l1729_172915


namespace NUMINAMATH_CALUDE_solve_system_l1729_172981

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1729_172981


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1729_172950

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the transformation
def transform (f : ℝ → ℝ) (horizontal_shift : ℝ) (vertical_shift : ℝ) : ℝ → ℝ :=
  λ x => f (x - horizontal_shift) + vertical_shift

-- Define the new function after transformation
def new_function : ℝ → ℝ := transform original_function 3 3

-- Theorem stating the equivalence
theorem quadratic_transformation :
  ∀ x : ℝ, new_function x = (x - 3)^2 + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1729_172950


namespace NUMINAMATH_CALUDE_second_person_work_days_l1729_172957

/-- Represents the number of days two people take to complete a task together -/
def two_people_time : ℝ := 10

/-- Represents the number of days one person takes to complete the task alone -/
def one_person_time : ℝ := 70

/-- Represents the number of days the first person took to complete the remaining work after the second person left -/
def remaining_work_time : ℝ := 42

/-- Represents the number of days the second person worked before leaving -/
def second_person_work_time : ℝ := 4

/-- Theorem stating that given the conditions, the second person worked for 4 days before leaving -/
theorem second_person_work_days :
  two_people_time = 10 ∧
  one_person_time = 70 ∧
  remaining_work_time = 42 →
  second_person_work_time = 4 :=
by sorry

end NUMINAMATH_CALUDE_second_person_work_days_l1729_172957


namespace NUMINAMATH_CALUDE_system_of_equations_l1729_172900

theorem system_of_equations (x y z k : ℝ) : 
  (2 * x - y + 3 * z = 9) → 
  (x + 2 * y - z = k) → 
  (-x + y + 4 * z = 6) → 
  (y = -1) → 
  (k = -3) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l1729_172900


namespace NUMINAMATH_CALUDE_product_equals_sum_and_difference_l1729_172937

theorem product_equals_sum_and_difference :
  ∀ a b : ℤ, (a * b = a + b ∧ a * b = a - b) → (a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_product_equals_sum_and_difference_l1729_172937


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_is_square_l1729_172953

theorem sum_of_fourth_powers_is_square (a b c : ℤ) (h : a + b + c = 0) :
  2 * (a^4 + b^4 + c^4) = (a^2 + b^2 + c^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_is_square_l1729_172953


namespace NUMINAMATH_CALUDE_rachel_homework_l1729_172942

/-- Rachel's homework problem -/
theorem rachel_homework (math_pages reading_pages : ℕ) : 
  math_pages = 5 → reading_pages = 2 → math_pages + reading_pages = 7 := by
  sorry

end NUMINAMATH_CALUDE_rachel_homework_l1729_172942


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1729_172983

theorem abs_neg_three_eq_three : abs (-3 : ℤ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1729_172983


namespace NUMINAMATH_CALUDE_team_selection_proof_l1729_172911

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 5 players from a team of 9 players, 
    where 2 seeded players must be included -/
def teamSelection : ℕ := sorry

theorem team_selection_proof :
  let totalPlayers : ℕ := 9
  let seededPlayers : ℕ := 2
  let selectCount : ℕ := 5
  teamSelection = choose (totalPlayers - seededPlayers) (selectCount - seededPlayers) := by
  sorry

end NUMINAMATH_CALUDE_team_selection_proof_l1729_172911


namespace NUMINAMATH_CALUDE_product_bounds_l1729_172934

theorem product_bounds (x₁ x₂ x₃ : ℝ) 
  (h_nonneg₁ : x₁ ≥ 0) (h_nonneg₂ : x₂ ≥ 0) (h_nonneg₃ : x₃ ≥ 0)
  (h_sum : x₁ + x₂ + x₃ = 1) :
  1 ≤ (x₁ + 3*x₂ + 5*x₃) * (x₁ + x₂/3 + x₃/5) ∧
  (x₁ + 3*x₂ + 5*x₃) * (x₁ + x₂/3 + x₃/5) ≤ 9/5 :=
by sorry

end NUMINAMATH_CALUDE_product_bounds_l1729_172934


namespace NUMINAMATH_CALUDE_ellipse_properties_l1729_172997

-- Define the ellipse C
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the condition for equilateral triangle formed by foci and minor axis endpoint
def equilateralCondition (a b c : ℝ) : Prop :=
  a = 2*c ∧ b = Real.sqrt 3 * c

-- Define the tangency condition for the circle
def tangencyCondition (a b c : ℝ) : Prop :=
  |c + 2| / Real.sqrt 2 = (Real.sqrt 6 / 2) * b

-- Define the vector addition condition
def vectorAdditionCondition (A B M : ℝ × ℝ) (t : ℝ) : Prop :=
  A.1 + B.1 = t * M.1 ∧ A.2 + B.2 = t * M.2

-- Main theorem
theorem ellipse_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  ∀ (c : ℝ), equilateralCondition a b c →
  tangencyCondition a b c →
  (∀ (A B M : ℝ × ℝ) (t : ℝ),
    A ∈ ellipse a b h →
    B ∈ ellipse a b h →
    M ∈ ellipse a b h →
    (∃ (k : ℝ), A.2 = k*(A.1 - 3) ∧ B.2 = k*(B.1 - 3)) →
    vectorAdditionCondition A B M t →
    (a = 2 ∧ b = Real.sqrt 3 ∧ c = 1) ∧
    ((a^2 - b^2) / a^2 = 1/4) ∧
    (ellipse a b h = {p : ℝ × ℝ | p.1^2/4 + p.2^2/3 = 1}) ∧
    (-2 < t ∧ t < 2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1729_172997


namespace NUMINAMATH_CALUDE_product_expansion_l1729_172943

theorem product_expansion (y : ℝ) (h : y ≠ 0) :
  (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1729_172943


namespace NUMINAMATH_CALUDE_wire_length_difference_l1729_172940

theorem wire_length_difference (total_length piece1 piece2 : ℝ) : 
  total_length = 30 →
  piece1 = 14 →
  piece2 = 16 →
  |piece2 - piece1| = 2 := by sorry

end NUMINAMATH_CALUDE_wire_length_difference_l1729_172940


namespace NUMINAMATH_CALUDE_spiral_stripe_length_l1729_172933

theorem spiral_stripe_length (c h : ℝ) (hc : c = 18) (hh : h = 8) :
  let stripe_length := Real.sqrt ((2 * c)^2 + h^2)
  stripe_length = Real.sqrt 1360 := by
  sorry

end NUMINAMATH_CALUDE_spiral_stripe_length_l1729_172933


namespace NUMINAMATH_CALUDE_number_puzzle_l1729_172972

theorem number_puzzle : ∃ x : ℚ, (x / 5 + 4 = x / 4 - 10) ∧ x = 280 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1729_172972


namespace NUMINAMATH_CALUDE_function_range_function_range_with_condition_l1729_172903

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin x - 1/2 * Real.cos (2*x) + a - 3/a + 1/2

theorem function_range (a : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 0) → a ∈ Set.Ioo 0 1 :=
by sorry

theorem function_range_with_condition (a : ℝ) (h : a ≠ 0) :
  a ≥ 2 → (∃ x : ℝ, f a x ≤ 0) → a ∈ Set.Icc 2 3 :=
by sorry

end NUMINAMATH_CALUDE_function_range_function_range_with_condition_l1729_172903


namespace NUMINAMATH_CALUDE_set_A_enumeration_l1729_172952

def A : Set ℚ := {z | ∃ p q : ℕ+, z = p / q ∧ p + q = 5}

theorem set_A_enumeration : A = {1/4, 2/3, 3/2, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_A_enumeration_l1729_172952


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l1729_172958

theorem division_multiplication_problem : (0.45 / 0.005) * 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l1729_172958


namespace NUMINAMATH_CALUDE_subset_star_inclusion_l1729_172955

/-- Given non-empty sets of real numbers M and P, where M ⊆ P, prove that P* ⊆ M* -/
theorem subset_star_inclusion {M P : Set ℝ} (hM : M.Nonempty) (hP : P.Nonempty) (h_subset : M ⊆ P) :
  {y : ℝ | ∀ x ∈ P, y ≥ x} ⊆ {y : ℝ | ∀ x ∈ M, y ≥ x} := by
  sorry

end NUMINAMATH_CALUDE_subset_star_inclusion_l1729_172955


namespace NUMINAMATH_CALUDE_complex_fraction_difference_l1729_172967

theorem complex_fraction_difference : 
  (Complex.mk 3 2) / (Complex.mk 2 (-3)) - (Complex.mk 3 (-2)) / (Complex.mk 2 3) = Complex.I * 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_difference_l1729_172967


namespace NUMINAMATH_CALUDE_gcf_36_60_l1729_172929

theorem gcf_36_60 : Nat.gcd 36 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcf_36_60_l1729_172929


namespace NUMINAMATH_CALUDE_min_b_for_real_roots_F_monotonic_iff_l1729_172985

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - Real.log x

-- Define the function F
def F (a : ℝ) (x : ℝ) : ℝ := f a x * Real.exp (-x)

-- Theorem for part 1
theorem min_b_for_real_roots (x : ℝ) :
  ∃ (b : ℝ), b ≥ 0 ∧ ∃ (x : ℝ), x > 0 ∧ f (-1) x = b / x ∧
  ∀ (b' : ℝ), b' < b → ¬∃ (x : ℝ), x > 0 ∧ f (-1) x = b' / x :=
sorry

-- Theorem for part 2
theorem F_monotonic_iff (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → F a x₁ < F a x₂) ∨
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → F a x₁ > F a x₂) ↔
  a ≤ 2 :=
sorry

end

end NUMINAMATH_CALUDE_min_b_for_real_roots_F_monotonic_iff_l1729_172985


namespace NUMINAMATH_CALUDE_max_value_a4a7_l1729_172961

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The maximum value of a_4 * a_7 in an arithmetic sequence where a_6 = 4 -/
theorem max_value_a4a7 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h6 : a 6 = 4) :
  (∀ d : ℝ, a 4 * a 7 ≤ 18) ∧ (∃ d : ℝ, a 4 * a 7 = 18) :=
sorry

end NUMINAMATH_CALUDE_max_value_a4a7_l1729_172961


namespace NUMINAMATH_CALUDE_slopes_intersect_ellipse_l1729_172928

/-- The set of possible slopes for a line with y-intercept (0,3) that intersects the ellipse 4x^2 + 25y^2 = 100 -/
def possible_slopes : Set ℝ :=
  {m : ℝ | m ≤ -2/5 ∨ m ≥ 2/5}

/-- The equation of the line with slope m and y-intercept (0,3) -/
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x + 3

/-- The equation of the ellipse 4x^2 + 25y^2 = 100 -/
def ellipse_equation (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

/-- Theorem stating that the set of possible slopes is correct -/
theorem slopes_intersect_ellipse :
  ∀ m : ℝ, m ∈ possible_slopes ↔
    ∃ x : ℝ, ellipse_equation x (line_equation m x) := by
  sorry

end NUMINAMATH_CALUDE_slopes_intersect_ellipse_l1729_172928


namespace NUMINAMATH_CALUDE_average_price_is_18_l1729_172935

/-- The average price per book given two book purchases -/
def average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℚ) : ℚ :=
  (price1 + price2) / (books1 + books2)

/-- Theorem stating that the average price per book is 18 for the given purchases -/
theorem average_price_is_18 :
  average_price_per_book 65 50 1150 920 = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_price_is_18_l1729_172935


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_gt_zero_l1729_172927

theorem a_gt_one_sufficient_not_necessary_for_a_gt_zero :
  (∃ a : ℝ, a > 0 ∧ ¬(a > 1)) ∧
  (∀ a : ℝ, a > 1 → a > 0) :=
sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_for_a_gt_zero_l1729_172927


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l1729_172907

/-- Given a rhombus with diagonals of 10 inches and 24 inches, its perimeter is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

#check rhombus_perimeter

end NUMINAMATH_CALUDE_rhombus_perimeter_l1729_172907


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1729_172954

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ+, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ+ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n : ℕ+, a n > 0)
  (h_a4 : a 4 = 4)
  (h_a6 : a 6 = 16) :
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ+, a (n + 1) = a n * q := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1729_172954


namespace NUMINAMATH_CALUDE_largest_perimeter_l1729_172944

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  h1 : side1 = 7
  h2 : side2 = 9
  h3 : side3 % 3 = 0

/-- Checks if the given sides form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of the triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem stating the largest possible perimeter --/
theorem largest_perimeter :
  ∀ t : Triangle, is_valid_triangle t →
  ∃ max_t : Triangle, is_valid_triangle max_t ∧
  perimeter max_t = 31 ∧
  ∀ other_t : Triangle, is_valid_triangle other_t →
  perimeter other_t ≤ perimeter max_t :=
sorry

end NUMINAMATH_CALUDE_largest_perimeter_l1729_172944


namespace NUMINAMATH_CALUDE_max_value_expression_l1729_172930

theorem max_value_expression (a b c d : ℝ) 
  (ha : -7 ≤ a ∧ a ≤ 7) 
  (hb : -7 ≤ b ∧ b ≤ 7) 
  (hc : -7 ≤ c ∧ c ≤ 7) 
  (hd : -7 ≤ d ∧ d ≤ 7) : 
  (∀ a' b' c' d' : ℝ, 
    -7 ≤ a' ∧ a' ≤ 7 → 
    -7 ≤ b' ∧ b' ≤ 7 → 
    -7 ≤ c' ∧ c' ≤ 7 → 
    -7 ≤ d' ∧ d' ≤ 7 → 
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' ≤ 210) ∧
  (∃ a' b' c' d' : ℝ, 
    -7 ≤ a' ∧ a' ≤ 7 ∧
    -7 ≤ b' ∧ b' ≤ 7 ∧
    -7 ≤ c' ∧ c' ≤ 7 ∧
    -7 ≤ d' ∧ d' ≤ 7 ∧
    a' + 2*b' + c' + 2*d' - a'*b' - b'*c' - c'*d' - d'*a' = 210) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1729_172930


namespace NUMINAMATH_CALUDE_inequalities_solution_l1729_172924

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x - 3 * (x - 2) > 4
def inequality2 (x : ℝ) : Prop := (2 * x - 1) / 3 ≤ (x + 1) / 2

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 1

-- Theorem statement
theorem inequalities_solution :
  ∀ x : ℝ, (inequality1 x ∧ inequality2 x) ↔ solution_set x :=
by sorry

end NUMINAMATH_CALUDE_inequalities_solution_l1729_172924


namespace NUMINAMATH_CALUDE_max_non_managers_l1729_172946

theorem max_non_managers (num_managers : ℕ) (ratio_managers : ℚ) (ratio_non_managers : ℚ) :
  num_managers = 8 →
  ratio_managers / ratio_non_managers > 7 / 24 →
  ∃ (max_non_managers : ℕ),
    (↑num_managers : ℚ) / (↑max_non_managers : ℚ) > ratio_managers / ratio_non_managers ∧
    ∀ (n : ℕ), n > max_non_managers →
      (↑num_managers : ℚ) / (↑n : ℚ) ≤ ratio_managers / ratio_non_managers →
      max_non_managers = 27 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l1729_172946


namespace NUMINAMATH_CALUDE_right_triangle_area_l1729_172987

/-- The area of a right triangle with hypotenuse 5 and shortest side 3 is 6 -/
theorem right_triangle_area : ∀ (a b c : ℝ),
  a = 3 →
  c = 5 →
  a ≤ b →
  b ≤ c →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1729_172987


namespace NUMINAMATH_CALUDE_class_field_trip_budget_l1729_172979

/-- The class's budget for a field trip to the zoo --/
theorem class_field_trip_budget
  (bus_rental_cost : ℕ)
  (admission_cost_per_student : ℕ)
  (number_of_students : ℕ)
  (h1 : bus_rental_cost = 100)
  (h2 : admission_cost_per_student = 10)
  (h3 : number_of_students = 25) :
  bus_rental_cost + admission_cost_per_student * number_of_students = 350 :=
by sorry

end NUMINAMATH_CALUDE_class_field_trip_budget_l1729_172979


namespace NUMINAMATH_CALUDE_opposite_sides_inequality_l1729_172959

/-- Given that point P(x₀, y₀) and point A(1, 2) are on opposite sides of the line 3x + 2y - 8 = 0,
    then 3x₀ + 2y₀ > 8 -/
theorem opposite_sides_inequality (x₀ y₀ : ℝ) : 
  (∃ (ε : ℝ), (3*x₀ + 2*y₀ - 8) * (3*1 + 2*2 - 8) = -ε ∧ ε > 0) →
  3*x₀ + 2*y₀ > 8 :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_inequality_l1729_172959


namespace NUMINAMATH_CALUDE_factory_growth_rate_l1729_172998

theorem factory_growth_rate (x : ℝ) : 
  (1 + x)^2 = 1.2 → x < 0.1 := by sorry

end NUMINAMATH_CALUDE_factory_growth_rate_l1729_172998


namespace NUMINAMATH_CALUDE_unique_cube_labeling_l1729_172923

/-- A cube labeling is a function from vertices to integers -/
def CubeLabeling := Fin 8 → Fin 8

/-- A face of the cube is a set of four vertices -/
def CubeFace := Finset (Fin 8)

/-- The set of all faces of a cube -/
def allFaces : Finset CubeFace := sorry

/-- A labeling is valid if it's a bijection (each number used once) -/
def isValidLabeling (l : CubeLabeling) : Prop :=
  Function.Bijective l

/-- The sum of labels on a face equals 22 -/
def faceSum22 (l : CubeLabeling) (face : CubeFace) : Prop :=
  (face.sum (λ v => (l v).val + 1) : ℕ) = 22

/-- All faces of a labeling sum to 22 -/
def allFacesSum22 (l : CubeLabeling) : Prop :=
  ∀ face ∈ allFaces, faceSum22 l face

/-- Two labelings are equivalent if they can be obtained by flipping the cube -/
def equivalentLabelings (l₁ l₂ : CubeLabeling) : Prop := sorry

/-- The main theorem: there is only one unique labeling up to equivalence -/
theorem unique_cube_labeling :
  ∃! l : CubeLabeling, isValidLabeling l ∧ allFacesSum22 l := by sorry

end NUMINAMATH_CALUDE_unique_cube_labeling_l1729_172923


namespace NUMINAMATH_CALUDE_min_cube_sum_l1729_172994

theorem min_cube_sum (w z : ℂ) (h1 : Complex.abs (w + z) = 2) (h2 : Complex.abs (w^2 + z^2) = 16) :
  Complex.abs (w^3 + z^3) ≥ 22 := by
  sorry

end NUMINAMATH_CALUDE_min_cube_sum_l1729_172994


namespace NUMINAMATH_CALUDE_min_sum_squares_l1729_172908

theorem min_sum_squares (x y z : ℝ) (h : x + y + z = 1) : x^2 + y^2 + z^2 ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1729_172908


namespace NUMINAMATH_CALUDE_circle_equation_and_tangent_lines_l1729_172901

/-- Circle C with center (a, b) and radius 5 -/
structure CircleC where
  a : ℝ
  b : ℝ
  center_on_line : a + b + 1 = 0
  passes_through_p : ((-2) - a)^2 + (0 - b)^2 = 25
  passes_through_q : (5 - a)^2 + (1 - b)^2 = 25

/-- Tangent line to circle C passing through point A(-3, 0) -/
structure TangentLine where
  k : ℝ

theorem circle_equation_and_tangent_lines (c : CircleC) :
  ((c.a = 2 ∧ c.b = -3) ∧
   (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 25 ↔ (x - c.a)^2 + (y - c.b)^2 = 25)) ∧
  (∃ t : TangentLine,
    (t.k = 0 ∧ ∀ x y : ℝ, y = t.k * (x + 3) ↔ x = -3) ∨
    (t.k = 8/15 ∧ ∀ x y : ℝ, y = t.k * (x + 3) ↔ y = (8/15) * (x + 3))) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_and_tangent_lines_l1729_172901


namespace NUMINAMATH_CALUDE_solution_set_characterization_l1729_172963

-- Define the properties of the function f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def increasing_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- State the theorem
theorem solution_set_characterization
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_incr : increasing_on_positive f)
  (h_zero : f (-3) = 0) :
  {x : ℝ | x * f x < 0} = Set.Ioo (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_characterization_l1729_172963


namespace NUMINAMATH_CALUDE_divisibility_conditions_l1729_172925

theorem divisibility_conditions (a b : ℕ) : 
  (∃ k : ℤ, (a^3 * b - 1) = k * (a + 1)) ∧ 
  (∃ m : ℤ, (a * b^3 + 1) = m * (b - 1)) ↔ 
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l1729_172925


namespace NUMINAMATH_CALUDE_harmonic_set_odd_cardinality_min_harmonic_set_cardinality_l1729_172909

/-- A set of positive integers is a "harmonic set" if removing any element
    results in the remaining elements being divisible into two disjoint sets
    with equal sum of elements. -/
def is_harmonic_set (A : Finset ℕ) : Prop :=
  A.card ≥ 3 ∧ ∀ a ∈ A, ∃ B C : Finset ℕ,
    B ⊆ A \ {a} ∧ C ⊆ A \ {a} ∧ B ∩ C = ∅ ∧ B ∪ C = A \ {a} ∧
    (B.sum id = C.sum id)

theorem harmonic_set_odd_cardinality (A : Finset ℕ) (h : is_harmonic_set A) :
  Odd A.card :=
sorry

theorem min_harmonic_set_cardinality :
  ∃ A : Finset ℕ, is_harmonic_set A ∧ A.card = 7 ∧
    ∀ B : Finset ℕ, is_harmonic_set B → B.card ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_harmonic_set_odd_cardinality_min_harmonic_set_cardinality_l1729_172909


namespace NUMINAMATH_CALUDE_pascals_triangle_15_numbers_4th_entry_l1729_172919

theorem pascals_triangle_15_numbers_4th_entry : 
  let n : ℕ := 14  -- The row number (15 numbers, so it's the 14th row)
  let k : ℕ := 4   -- The position of the number we're looking for
  Nat.choose (n - 1) (k - 1) = 286 := by
sorry

end NUMINAMATH_CALUDE_pascals_triangle_15_numbers_4th_entry_l1729_172919


namespace NUMINAMATH_CALUDE_shekars_average_marks_l1729_172975

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 47
def biology_score : ℕ := 85

def total_subjects : ℕ := 5

theorem shekars_average_marks :
  (mathematics_score + science_score + social_studies_score + english_score + biology_score) / total_subjects = 71 := by
  sorry

end NUMINAMATH_CALUDE_shekars_average_marks_l1729_172975


namespace NUMINAMATH_CALUDE_problem_zeros_count_l1729_172936

/-- The number of zeros in the binary representation of a natural number -/
def countZeros (n : ℕ) : ℕ := sorry

/-- The expression given in the problem -/
def problemExpression : ℕ := 
  ((18 * 8192 + 8 * 128 - 12 * 16) / 6 + 4 * 64 + 3^5 - (25 * 2))

/-- Theorem stating that the number of zeros in the binary representation of the problem expression is 6 -/
theorem problem_zeros_count : countZeros problemExpression = 6 := by sorry

end NUMINAMATH_CALUDE_problem_zeros_count_l1729_172936


namespace NUMINAMATH_CALUDE_size_and_precision_difference_l1729_172941

/-- Represents the precision of a number -/
inductive Precision
  | Ones
  | Tenths

/-- Represents a number with its value and precision -/
structure NumberWithPrecision where
  value : ℝ
  precision : Precision

/-- The statement that the size and precision of 3.0 and 3 are the same is false -/
theorem size_and_precision_difference : ∃ (a b : NumberWithPrecision), 
  a.value = b.value ∧ a.precision ≠ b.precision := by
  sorry

/-- The numerical value of 3.0 equals 3 -/
axiom value_equality : ∃ (a b : NumberWithPrecision), 
  a.value = 3 ∧ b.value = 3 ∧ a.value = b.value

/-- The precision of 3.0 is to the tenth -/
axiom precision_three_point_zero : ∃ (a : NumberWithPrecision), 
  a.value = 3 ∧ a.precision = Precision.Tenths

/-- The precision of 3 is to 1 -/
axiom precision_three : ∃ (b : NumberWithPrecision), 
  b.value = 3 ∧ b.precision = Precision.Ones

end NUMINAMATH_CALUDE_size_and_precision_difference_l1729_172941


namespace NUMINAMATH_CALUDE_custom_mult_solution_l1729_172922

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := 2 * a - b^2

/-- Theorem stating that given the custom multiplication and the equation a * 7 = 16, a equals 32.5 -/
theorem custom_mult_solution :
  ∃ a : ℝ, custom_mult a 7 = 16 ∧ a = 32.5 := by sorry

end NUMINAMATH_CALUDE_custom_mult_solution_l1729_172922


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1729_172978

/-- Given a line L1 defined by 2x + 3y = 9, and another line L2 that is perpendicular to L1
    with a y-intercept of -4, the x-intercept of L2 is 8/3. -/
theorem perpendicular_line_x_intercept :
  ∀ (L1 L2 : Set (ℝ × ℝ)),
  (∀ x y, (x, y) ∈ L1 ↔ 2 * x + 3 * y = 9) →
  (∃ m : ℝ, ∀ x y, (x, y) ∈ L2 ↔ y = m * x - 4) →
  (∀ x y₁ y₂, (x, y₁) ∈ L1 ∧ (x, y₂) ∈ L2 → (y₁ - y₂) * (x - 0) = -(1 : ℝ)) →
  (∃ x : ℝ, (x, 0) ∈ L2 ∧ x = 8 / 3) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l1729_172978


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l1729_172938

theorem degree_to_radian_conversion (angle_deg : ℝ) : 
  angle_deg * (π / 180) = -5 * π / 3 ↔ angle_deg = -300 :=
sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l1729_172938


namespace NUMINAMATH_CALUDE_triangle_side_length_l1729_172990

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  b = 7 →
  c = 5 →
  Real.cos (B - C) = 47 / 50 →
  a = Real.sqrt 54.4 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1729_172990


namespace NUMINAMATH_CALUDE_reciprocal_problems_l1729_172992

theorem reciprocal_problems :
  (1 / 1.5 = 2/3) ∧ (1 / 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problems_l1729_172992


namespace NUMINAMATH_CALUDE_power_of_three_expression_equals_zero_l1729_172969

theorem power_of_three_expression_equals_zero :
  3^2003 - 5 * 3^2002 + 6 * 3^2001 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_expression_equals_zero_l1729_172969


namespace NUMINAMATH_CALUDE_trip_time_difference_l1729_172966

/-- Proves that the difference in time between two trips is 60 minutes, 
    given the conditions of the problem. -/
theorem trip_time_difference 
  (speed : ℝ) 
  (distance1 : ℝ) 
  (distance2 : ℝ) 
  (h1 : speed = 60) 
  (h2 : distance1 = 540) 
  (h3 : distance2 = 600) : 
  (distance2 / speed - distance1 / speed) * 60 = 60 := by
  sorry

#check trip_time_difference

end NUMINAMATH_CALUDE_trip_time_difference_l1729_172966


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1729_172965

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, 280 * r = a ∧ a * r = 35 / 8) : a = 35 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1729_172965


namespace NUMINAMATH_CALUDE_unique_integer_pair_satisfying_equation_l1729_172932

theorem unique_integer_pair_satisfying_equation : 
  ∃! (m n : ℤ), m + 2*n = m*n + 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_pair_satisfying_equation_l1729_172932


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l1729_172962

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (n_balls : ℕ) (n_boxes : ℕ) : ℕ :=
  n_boxes ^ n_balls

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 3 distinguishable boxes is 3^6 -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 3^6 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l1729_172962


namespace NUMINAMATH_CALUDE_david_money_left_l1729_172920

/-- Represents the money situation of a person on a trip -/
def MoneyOnTrip (initial_amount spent_amount remaining_amount : ℕ) : Prop :=
  (initial_amount = spent_amount + remaining_amount) ∧
  (remaining_amount = spent_amount - 800)

theorem david_money_left :
  ∃ (spent_amount : ℕ), MoneyOnTrip 1800 spent_amount 500 :=
sorry

end NUMINAMATH_CALUDE_david_money_left_l1729_172920


namespace NUMINAMATH_CALUDE_don_profit_l1729_172912

/-- Represents a person's rose bundle transaction -/
structure Transaction where
  bought : ℕ
  sold : ℕ
  profit : ℚ

/-- Represents the prices of rose bundles -/
structure Prices where
  buy : ℚ
  sell : ℚ

/-- The main theorem -/
theorem don_profit 
  (jamie : Transaction)
  (linda : Transaction)
  (don : Transaction)
  (prices : Prices)
  (h1 : jamie.bought = 20)
  (h2 : jamie.sold = 15)
  (h3 : jamie.profit = 60)
  (h4 : linda.bought = 34)
  (h5 : linda.sold = 24)
  (h6 : linda.profit = 69)
  (h7 : don.bought = 40)
  (h8 : don.sold = 36)
  (h9 : prices.sell > prices.buy)
  (h10 : jamie.profit = jamie.sold * prices.sell - jamie.bought * prices.buy)
  (h11 : linda.profit = linda.sold * prices.sell - linda.bought * prices.buy)
  (h12 : don.profit = don.sold * prices.sell - don.bought * prices.buy) :
  don.profit = 252 := by sorry

end NUMINAMATH_CALUDE_don_profit_l1729_172912


namespace NUMINAMATH_CALUDE_sqrt_inequality_l1729_172916

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt (a - 2) - Real.sqrt (a - 3) > Real.sqrt a - Real.sqrt (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l1729_172916


namespace NUMINAMATH_CALUDE_man_half_father_age_l1729_172917

theorem man_half_father_age (father_age : ℝ) (man_age : ℝ) (years_later : ℝ) : 
  father_age = 30.000000000000007 →
  man_age = (2/5) * father_age →
  man_age + years_later = (1/2) * (father_age + years_later) →
  years_later = 6 := by
sorry

end NUMINAMATH_CALUDE_man_half_father_age_l1729_172917


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l1729_172921

def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

def min_value (b : ℕ → ℝ) : ℝ :=
  5 * b 1 + 6 * b 2

theorem geometric_sequence_min_value :
  ∀ b : ℕ → ℝ, geometric_sequence b → b 0 = 2 →
  ∃ m : ℝ, m = min_value b ∧ m = -25/12 ∧ ∀ b' : ℕ → ℝ, geometric_sequence b' → b' 0 = 2 → min_value b' ≥ m :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l1729_172921


namespace NUMINAMATH_CALUDE_dvd_money_calculation_l1729_172945

/-- Given the cost of one pack of DVDs and the number of packs that can be bought,
    calculate the total amount of money available. -/
theorem dvd_money_calculation (cost_per_pack : ℕ) (num_packs : ℕ) :
  cost_per_pack = 12 → num_packs = 11 → cost_per_pack * num_packs = 132 := by
  sorry

end NUMINAMATH_CALUDE_dvd_money_calculation_l1729_172945


namespace NUMINAMATH_CALUDE_library_book_count_l1729_172974

/-- The number of books in a library after two years of purchases -/
def total_books (initial : ℕ) (last_year : ℕ) (multiplier : ℕ) : ℕ :=
  initial + last_year + multiplier * last_year

/-- Theorem: The library now has 300 books -/
theorem library_book_count : total_books 100 50 3 = 300 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l1729_172974


namespace NUMINAMATH_CALUDE_four_row_arrangement_has_fourteen_triangles_l1729_172947

/-- Represents a triangular arrangement of smaller triangles. -/
structure TriangularArrangement where
  rows : Nat
  bottom_row_triangles : Nat

/-- Calculates the total number of triangles in the arrangement. -/
def total_triangles (arr : TriangularArrangement) : Nat :=
  sorry

/-- Theorem stating that a triangular arrangement with 4 rows and 4 triangles
    in the bottom row has a total of 14 triangles. -/
theorem four_row_arrangement_has_fourteen_triangles :
  ∀ (arr : TriangularArrangement),
    arr.rows = 4 →
    arr.bottom_row_triangles = 4 →
    total_triangles arr = 14 :=
  sorry

end NUMINAMATH_CALUDE_four_row_arrangement_has_fourteen_triangles_l1729_172947


namespace NUMINAMATH_CALUDE_tip_fraction_is_55_93_l1729_172902

/-- Represents the waiter's salary structure over four weeks -/
structure WaiterSalary where
  base : ℚ  -- Base salary
  tips1 : ℚ := 5/3 * base  -- Tips in week 1
  tips2 : ℚ := 3/2 * base  -- Tips in week 2
  tips3 : ℚ := base        -- Tips in week 3
  tips4 : ℚ := 4/3 * base  -- Tips in week 4
  expenses : ℚ := 2/5 * base  -- Total expenses over 4 weeks (10% per week)

/-- Calculates the fraction of total income after expenses that came from tips -/
def tipFraction (s : WaiterSalary) : ℚ :=
  let totalTips := s.tips1 + s.tips2 + s.tips3 + s.tips4
  let totalIncome := 4 * s.base + totalTips
  let incomeAfterExpenses := totalIncome - s.expenses
  totalTips / incomeAfterExpenses

/-- Theorem stating that the fraction of total income after expenses that came from tips is 55/93 -/
theorem tip_fraction_is_55_93 (s : WaiterSalary) : tipFraction s = 55/93 := by
  sorry


end NUMINAMATH_CALUDE_tip_fraction_is_55_93_l1729_172902


namespace NUMINAMATH_CALUDE_sue_movie_borrowing_l1729_172989

/-- The number of movies Sue initially borrowed -/
def initial_movies : ℕ := 6

/-- The number of books Sue initially borrowed -/
def initial_books : ℕ := 15

/-- The number of books Sue returned -/
def returned_books : ℕ := 8

/-- The number of additional books Sue checked out -/
def additional_books : ℕ := 9

/-- The total number of items Sue has at the end -/
def total_items : ℕ := 20

theorem sue_movie_borrowing :
  initial_movies = 6 ∧
  initial_books + initial_movies - returned_books - (initial_movies / 3) + additional_books = total_items :=
by sorry

end NUMINAMATH_CALUDE_sue_movie_borrowing_l1729_172989


namespace NUMINAMATH_CALUDE_binomial_expansion_terms_l1729_172971

theorem binomial_expansion_terms (x a : ℝ) (n : ℕ) : 
  (Nat.choose n 1 : ℝ) * x^(n-1) * a = 56 ∧
  (Nat.choose n 2 : ℝ) * x^(n-2) * a^2 = 168 ∧
  (Nat.choose n 3 : ℝ) * x^(n-3) * a^3 = 336 →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_terms_l1729_172971


namespace NUMINAMATH_CALUDE_time_difference_to_halfway_point_l1729_172988

/-- Given that Danny can reach Steve's house in 35 minutes and it takes Steve twice as long to reach Danny's house,
    prove that Steve will take 17.5 minutes longer than Danny to reach the halfway point between their houses. -/
theorem time_difference_to_halfway_point (danny_time : ℝ) (steve_time : ℝ) : 
  danny_time = 35 →
  steve_time = 2 * danny_time →
  steve_time / 2 - danny_time / 2 = 17.5 := by
sorry

end NUMINAMATH_CALUDE_time_difference_to_halfway_point_l1729_172988


namespace NUMINAMATH_CALUDE_pipe_C_empty_time_l1729_172910

/-- Represents the time (in minutes) it takes for pipe C to empty the cistern. -/
def empty_time (fill_time_A fill_time_B fill_time_all : ℚ) : ℚ :=
  let rate_A := 1 / fill_time_A
  let rate_B := 1 / fill_time_B
  let rate_all := 1 / fill_time_all
  let rate_C := rate_A + rate_B - rate_all
  1 / rate_C

/-- Theorem stating that given the fill times for pipes A and B, and the fill time when all pipes are open,
    the time it takes for pipe C to empty the cistern is 72 minutes. -/
theorem pipe_C_empty_time :
  empty_time 45 60 40 = 72 := by
  sorry

end NUMINAMATH_CALUDE_pipe_C_empty_time_l1729_172910


namespace NUMINAMATH_CALUDE_intersection_implies_a_values_union_implies_a_range_l1729_172973

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + (a-1)*x + a^2 - 5 = 0}

-- Part 1: A ∩ B = {2} implies a = -3 or a = 1
theorem intersection_implies_a_values (a : ℝ) : 
  A ∩ B a = {2} → a = -3 ∨ a = 1 := by sorry

-- Part 2: A ∪ B = A implies a ≤ -3 or a > 7/3
theorem union_implies_a_range (a : ℝ) :
  A ∪ B a = A → a ≤ -3 ∨ a > 7/3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_values_union_implies_a_range_l1729_172973


namespace NUMINAMATH_CALUDE_square_area_error_l1729_172976

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * 1.1
  let actual_area := s ^ 2
  let calculated_area := measured_side ^ 2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l1729_172976


namespace NUMINAMATH_CALUDE_f_2_equals_100_l1729_172999

-- Define the function f
def f (x y : ℝ) : ℝ := 2 * x^2 + y

-- State the theorem
theorem f_2_equals_100 :
  ∃ y : ℝ, f 5 y = 142 ∧ f 2 y = 100 :=
by sorry

end NUMINAMATH_CALUDE_f_2_equals_100_l1729_172999


namespace NUMINAMATH_CALUDE_copper_alloy_percentages_l1729_172968

theorem copper_alloy_percentages
  (x y : ℝ)  -- Percentages of copper in first and second alloys
  (m₁ m₂ : ℝ)  -- Masses of first and second alloys
  (h₁ : y = x + 40)  -- First alloy's copper percentage is 40% less than the second
  (h₂ : x * m₁ / 100 = 6)  -- First alloy contains 6 kg of copper
  (h₃ : y * m₂ / 100 = 12)  -- Second alloy contains 12 kg of copper
  (h₄ : 36 * (m₁ + m₂) / 100 = 18)  -- Mixture contains 36% copper
  : x = 20 ∧ y = 60 := by
  sorry

end NUMINAMATH_CALUDE_copper_alloy_percentages_l1729_172968


namespace NUMINAMATH_CALUDE_seal_releases_three_songs_per_month_l1729_172995

/-- Represents the earnings per song in dollars -/
def earnings_per_song : ℕ := 2000

/-- Represents the total earnings in the first 3 years in dollars -/
def total_earnings : ℕ := 216000

/-- Represents the number of months in 3 years -/
def months_in_three_years : ℕ := 3 * 12

/-- Represents the number of songs released per month -/
def songs_per_month : ℕ := total_earnings / earnings_per_song / months_in_three_years

theorem seal_releases_three_songs_per_month :
  songs_per_month = 3 :=
by sorry

end NUMINAMATH_CALUDE_seal_releases_three_songs_per_month_l1729_172995
