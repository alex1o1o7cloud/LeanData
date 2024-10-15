import Mathlib

namespace NUMINAMATH_CALUDE_club_members_problem_l988_98830

theorem club_members_problem (current_members : ℕ) : 
  (2 * current_members + 5 = current_members + 15) → 
  current_members = 10 := by
sorry

end NUMINAMATH_CALUDE_club_members_problem_l988_98830


namespace NUMINAMATH_CALUDE_rationalize_denominator_l988_98831

theorem rationalize_denominator :
  let x := (Real.sqrt 12 + Real.sqrt 2) / (Real.sqrt 3 + Real.sqrt 2)
  ∃ y, y = 4 - Real.sqrt 6 ∧ x = y :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l988_98831


namespace NUMINAMATH_CALUDE_system_solution_l988_98876

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁ - y₁ = 2 ∧ x₁^2 - 2*x₁*y₁ - 3*y₁^2 = 0 ∧ x₁ = 3 ∧ y₁ = 1) ∧
    (x₂ - y₂ = 2 ∧ x₂^2 - 2*x₂*y₂ - 3*y₂^2 = 0 ∧ x₂ = 1 ∧ y₂ = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l988_98876


namespace NUMINAMATH_CALUDE_list_price_correct_l988_98834

/-- Given a book's cost price, calculates the list price that results in a 40% profit
    after an 18% deduction from the list price -/
def listPrice (costPrice : ℝ) : ℝ :=
  costPrice * 1.7073

theorem list_price_correct (costPrice : ℝ) :
  let listPrice := listPrice costPrice
  let sellingPrice := listPrice * (1 - 0.18)
  sellingPrice = costPrice * 1.4 := by
  sorry

end NUMINAMATH_CALUDE_list_price_correct_l988_98834


namespace NUMINAMATH_CALUDE_triangle_area_proof_l988_98839

theorem triangle_area_proof (square_side : ℝ) (overlap_ratio_square : ℝ) (overlap_ratio_triangle : ℝ) : 
  square_side = 8 →
  overlap_ratio_square = 3/4 →
  overlap_ratio_triangle = 1/2 →
  let square_area := square_side * square_side
  let overlap_area := square_area * overlap_ratio_square
  let triangle_area := overlap_area / overlap_ratio_triangle
  triangle_area = 96 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l988_98839


namespace NUMINAMATH_CALUDE_functional_inequality_solution_l988_98816

theorem functional_inequality_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x * y) ≤ (1/2) * (f x + f y)) : 
  ∃ a c : ℝ, (∀ x : ℝ, x ≠ 0 → f x = c) ∧ (f 0 = a) ∧ (a ≤ c) := by
  sorry

end NUMINAMATH_CALUDE_functional_inequality_solution_l988_98816


namespace NUMINAMATH_CALUDE_catch_up_distance_l988_98802

/-- Represents a car traveling between two cities -/
structure Car where
  speed : ℝ
  startTime : ℝ

/-- Represents the problem scenario -/
structure TwoCarsScenario where
  carA : Car
  carB : Car
  totalDistance : ℝ

/-- The conditions of the problem -/
def problemConditions (scenario : TwoCarsScenario) : Prop :=
  scenario.totalDistance = 300 ∧
  scenario.carA.startTime = scenario.carB.startTime + 1 ∧
  (scenario.totalDistance / scenario.carA.speed) + scenario.carA.startTime =
    (scenario.totalDistance / scenario.carB.speed) + scenario.carB.startTime - 1

/-- The point where carA catches up with carB -/
def catchUpPoint (scenario : TwoCarsScenario) : ℝ :=
  scenario.totalDistance - (scenario.carA.speed * (scenario.carB.startTime - scenario.carA.startTime))

/-- The theorem to be proved -/
theorem catch_up_distance (scenario : TwoCarsScenario) :
  problemConditions scenario → catchUpPoint scenario = 150 := by
  sorry

end NUMINAMATH_CALUDE_catch_up_distance_l988_98802


namespace NUMINAMATH_CALUDE_solve_equations_l988_98828

theorem solve_equations :
  (∀ x : ℚ, (16 / 5) / x = (12 / 7) / (5 / 8) → x = 7 / 6) ∧
  (∀ x : ℚ, 2 * x + 3 * 0.9 = 24.7 → x = 11) :=
by sorry

end NUMINAMATH_CALUDE_solve_equations_l988_98828


namespace NUMINAMATH_CALUDE_min_fraction_sum_l988_98835

def Digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem min_fraction_sum (W X Y Z : ℕ) 
  (hw : W ∈ Digits) (hx : X ∈ Digits) (hy : Y ∈ Digits) (hz : Z ∈ Digits)
  (hdiff : W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) :
  (∀ W' X' Y' Z' : ℕ, 
    W' ∈ Digits → X' ∈ Digits → Y' ∈ Digits → Z' ∈ Digits →
    W' ≠ X' ∧ W' ≠ Y' ∧ W' ≠ Z' ∧ X' ≠ Y' ∧ X' ≠ Z' ∧ Y' ≠ Z' →
    (W : ℚ) / X + (Y : ℚ) / Z ≤ (W' : ℚ) / X' + (Y' : ℚ) / Z') →
  (W : ℚ) / X + (Y : ℚ) / Z = 23 / 21 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l988_98835


namespace NUMINAMATH_CALUDE_fred_initial_money_l988_98867

/-- Calculates the initial amount of money Fred had given the number of books bought,
    the average cost per book, and the amount left after buying. -/
def initial_money (num_books : ℕ) (avg_cost : ℕ) (money_left : ℕ) : ℕ :=
  num_books * avg_cost + money_left

/-- Proves that Fred initially had 236 dollars given the problem conditions. -/
theorem fred_initial_money :
  let num_books : ℕ := 6
  let avg_cost : ℕ := 37
  let money_left : ℕ := 14
  initial_money num_books avg_cost money_left = 236 := by
  sorry

end NUMINAMATH_CALUDE_fred_initial_money_l988_98867


namespace NUMINAMATH_CALUDE_smaller_fraction_l988_98850

theorem smaller_fraction (x y : ℚ) (sum_eq : x + y = 13/14) (prod_eq : x * y = 1/8) :
  min x y = 1/6 := by sorry

end NUMINAMATH_CALUDE_smaller_fraction_l988_98850


namespace NUMINAMATH_CALUDE_unique_solution_l988_98832

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  2 * x^3 = 2 * y * (x^2 + 1) - (z^2 + 1) ∧
  2 * y^4 = 3 * z * (y^2 + 1) - 2 * (x^2 + 1) ∧
  2 * z^5 = 4 * x * (z^2 + 1) - 3 * (y^2 + 1)

/-- The theorem stating that (1, 1, 1) is the unique positive real solution -/
theorem unique_solution :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ system x y z ∧ x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l988_98832


namespace NUMINAMATH_CALUDE_complex_number_real_part_l988_98817

theorem complex_number_real_part : 
  ∀ (z : ℂ) (a : ℝ), 
  (z / (2 + a * Complex.I) = 2 / (1 + Complex.I)) → 
  (z.im = -3) → 
  (z.re = 1) := by
sorry

end NUMINAMATH_CALUDE_complex_number_real_part_l988_98817


namespace NUMINAMATH_CALUDE_triangle_property_l988_98898

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
theorem triangle_property (A B C : Real) (a b c : Real) :
  -- Triangle existence conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B →
  -- Given condition
  Real.cos A / (1 + Real.sin A) = Real.sin B / (1 + Real.cos B) →
  -- Conclusions
  C = π / 2 ∧ 
  1 < (a * b + b * c + c * a) / (c^2) ∧ 
  (a * b + b * c + c * a) / (c^2) ≤ (1 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l988_98898


namespace NUMINAMATH_CALUDE_work_rate_problem_l988_98863

theorem work_rate_problem (a b c : ℝ) 
  (hab : a + b = 1/18)
  (hbc : b + c = 1/24)
  (hac : a + c = 1/36) :
  a + b + c = 1/16 := by
sorry

end NUMINAMATH_CALUDE_work_rate_problem_l988_98863


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l988_98875

theorem roots_of_quadratic (a b : ℝ) : 
  (a * b ≠ 0) →
  (a^2 + 2*b*a + a = 0) →
  (b^2 + 2*b*b + a = 0) →
  (a = -3 ∧ b = 1) := by
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l988_98875


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l988_98895

/-- The probability of drawing two white balls without replacement from a box containing 
    8 white balls and 10 black balls is 28/153. -/
theorem two_white_balls_probability (white_balls black_balls : ℕ) 
    (h1 : white_balls = 8) (h2 : black_balls = 10) :
  let total_balls := white_balls + black_balls
  let prob_first_white := white_balls / total_balls
  let prob_second_white := (white_balls - 1) / (total_balls - 1)
  prob_first_white * prob_second_white = 28 / 153 := by
  sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l988_98895


namespace NUMINAMATH_CALUDE_smallest_n_cookie_boxes_l988_98858

theorem smallest_n_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ 12 ∣ (17 * n - 1) ∧ ∀ (m : ℕ), m > 0 ∧ 12 ∣ (17 * m - 1) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_cookie_boxes_l988_98858


namespace NUMINAMATH_CALUDE_system_solution_l988_98820

theorem system_solution :
  ∃ x y : ℚ, (4 * x - 3 * y = -7) ∧ (5 * x + 6 * y = 4) ∧ (x = -10/13) ∧ (y = 17/13) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l988_98820


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l988_98803

theorem least_addition_for_divisibility : ∃! n : ℕ, 
  (∀ m : ℕ, m < n → ¬((1077 + m) % 23 = 0 ∧ (1077 + m) % 17 = 0)) ∧ 
  ((1077 + n) % 23 = 0 ∧ (1077 + n) % 17 = 0) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l988_98803


namespace NUMINAMATH_CALUDE_intersection_range_l988_98823

-- Define the curve C
def C (x y : ℝ) : Prop :=
  (x ≥ 0 ∧ y^2 = 4*x) ∨ (x ≤ 0 ∧ y = 0)

-- Define the line segment AB
def lineAB (a x y : ℝ) : Prop :=
  y = x + 1 ∧ x ≥ a - 1 ∧ x ≤ a

-- Theorem statement
theorem intersection_range (a : ℝ) :
  (∃! p : ℝ × ℝ, C p.1 p.2 ∧ lineAB a p.1 p.2) →
  a ∈ Set.Icc (-1) 0 ∪ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l988_98823


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l988_98800

-- Define the parabola and hyperbola
def parabola (x y : ℝ) : Prop := y^2 = 4*x
def hyperbola (x y a b : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem hyperbola_real_axis_length 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hf : ∃ (x y : ℝ), parabola x y ∧ hyperbola x y a b ∧ (x, y) ≠ parabola_focus) 
  (hperp : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    parabola x₁ y₁ ∧ hyperbola x₁ y₁ a b ∧
    parabola x₂ y₂ ∧ hyperbola x₂ y₂ a b ∧
    (x₁ + x₂) * (1 - x₁) + (y₁ + y₂) * (-y₁) = 0) :
  2 * a = 2 * Real.sqrt 2 - 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l988_98800


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l988_98841

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l988_98841


namespace NUMINAMATH_CALUDE_bridge_length_is_4km_l988_98810

/-- The length of a bridge crossed by a man -/
def bridge_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

/-- Theorem: The length of a bridge is 4 km when crossed by a man walking at 10 km/hr in 24 minutes -/
theorem bridge_length_is_4km (speed : ℝ) (time : ℝ) 
    (h1 : speed = 10) 
    (h2 : time = 24 / 60) : 
  bridge_length speed time = 4 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_is_4km_l988_98810


namespace NUMINAMATH_CALUDE_initial_puppies_count_l988_98892

/-- The number of puppies Sandy's dog initially had -/
def initial_puppies : ℕ := sorry

/-- The number of puppies Sandy gave away -/
def puppies_given_away : ℕ := 4

/-- The number of puppies Sandy has left -/
def puppies_left : ℕ := 4

/-- Theorem stating that the initial number of puppies is 8 -/
theorem initial_puppies_count : initial_puppies = 8 := by sorry

end NUMINAMATH_CALUDE_initial_puppies_count_l988_98892


namespace NUMINAMATH_CALUDE_inequality_proof_l988_98884

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^3 + y^4 ≤ x^2 + y^3) : x^3 + y^3 ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l988_98884


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l988_98888

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l988_98888


namespace NUMINAMATH_CALUDE_parsley_sprigs_left_l988_98869

/-- Calculates the number of parsley sprigs left after decorating plates --/
theorem parsley_sprigs_left 
  (initial_sprigs : ℕ) 
  (whole_sprig_plates : ℕ) 
  (half_sprig_plates : ℕ) : 
  initial_sprigs = 25 → 
  whole_sprig_plates = 8 → 
  half_sprig_plates = 12 → 
  initial_sprigs - (whole_sprig_plates + half_sprig_plates / 2) = 11 :=
by sorry

end NUMINAMATH_CALUDE_parsley_sprigs_left_l988_98869


namespace NUMINAMATH_CALUDE_hay_delivery_ratio_l988_98826

theorem hay_delivery_ratio : 
  let initial_bales : ℕ := 10
  let initial_cost_per_bale : ℕ := 15
  let new_cost_per_bale : ℕ := 18
  let additional_cost : ℕ := 210
  let new_bales : ℕ := (initial_bales * initial_cost_per_bale + additional_cost) / new_cost_per_bale
  (new_bales : ℚ) / initial_bales = 2 := by
  sorry

end NUMINAMATH_CALUDE_hay_delivery_ratio_l988_98826


namespace NUMINAMATH_CALUDE_max_k_value_l988_98837

/-- Given positive real numbers x, y, and k satisfying the equation
    5 = k³(x²/y² + y²/x²) + k(x/y + y/x) + 2k²,
    the maximum possible value of k is approximately 0.8. -/
theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 5 = k^3 * ((x^2 / y^2) + (y^2 / x^2)) + k * (x / y + y / x) + 2 * k^2) :
  k ≤ 0.8 := by sorry

end NUMINAMATH_CALUDE_max_k_value_l988_98837


namespace NUMINAMATH_CALUDE_purple_sequin_rows_purple_sequin_rows_proof_l988_98868

theorem purple_sequin_rows (blue_rows : Nat) (blue_per_row : Nat) 
  (purple_per_row : Nat) (green_rows : Nat) (green_per_row : Nat) 
  (total_sequins : Nat) : Nat :=
  let blue_sequins := blue_rows * blue_per_row
  let green_sequins := green_rows * green_per_row
  let non_purple_sequins := blue_sequins + green_sequins
  let purple_sequins := total_sequins - non_purple_sequins
  purple_sequins / purple_per_row

#check purple_sequin_rows 6 8 12 9 6 162 = 5

theorem purple_sequin_rows_proof :
  purple_sequin_rows 6 8 12 9 6 162 = 5 := by
  sorry

end NUMINAMATH_CALUDE_purple_sequin_rows_purple_sequin_rows_proof_l988_98868


namespace NUMINAMATH_CALUDE_tournament_games_count_l988_98838

/-- Calculates the number of games in a single-elimination tournament. -/
def singleEliminationGames (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- Calculates the total number of games in the tournament. -/
def totalGames (initialTeams : ℕ) : ℕ :=
  let preliminaryGames := initialTeams / 2
  let remainingTeams := initialTeams - preliminaryGames
  preliminaryGames + singleEliminationGames remainingTeams

/-- Theorem stating that the total number of games in the described tournament is 23. -/
theorem tournament_games_count :
  totalGames 24 = 23 := by sorry

end NUMINAMATH_CALUDE_tournament_games_count_l988_98838


namespace NUMINAMATH_CALUDE_cats_total_is_seven_l988_98889

/-- Calculates the total number of cats given the initial number of cats and the number of kittens -/
def total_cats (initial_cats female_kittens male_kittens : ℕ) : ℕ :=
  initial_cats + female_kittens + male_kittens

/-- Proves that the total number of cats is 7 given the initial conditions -/
theorem cats_total_is_seven :
  total_cats 2 3 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cats_total_is_seven_l988_98889


namespace NUMINAMATH_CALUDE_min_stamps_for_60_cents_l988_98870

/-- Represents the number of ways to make a certain amount using given denominations -/
def numWays (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

/-- Represents the minimum number of coins needed to make a certain amount using given denominations -/
def minCoins (amount : ℕ) (denominations : List ℕ) : ℕ :=
  sorry

theorem min_stamps_for_60_cents :
  minCoins 60 [5, 6] = 10 :=
sorry

end NUMINAMATH_CALUDE_min_stamps_for_60_cents_l988_98870


namespace NUMINAMATH_CALUDE_increasing_function_range_l988_98809

/-- Given a ∈ (0,1) and f(x) = a^x + (1+a)^x is increasing on (0,+∞), 
    prove that a ∈ [((5^(1/2)) - 1)/2, 1) -/
theorem increasing_function_range (a : ℝ) 
  (h1 : 0 < a ∧ a < 1) 
  (h2 : ∀ x > 0, Monotone (fun x => a^x + (1+a)^x)) : 
  a ∈ Set.Icc ((Real.sqrt 5 - 1) / 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_range_l988_98809


namespace NUMINAMATH_CALUDE_line_l1_equation_l988_98847

-- Define the lines and circle
def line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y + 1 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*y - 3 = 0

-- Define the property of being perpendicular
def perpendicular (l1 l2 : ℝ → ℝ → Prop) : Prop := sorry

-- Define the property of being tangent
def tangent (l : ℝ → ℝ → Prop) (c : ℝ → ℝ → Prop) : Prop := sorry

-- Theorem statement
theorem line_l1_equation :
  ∀ (l1 : ℝ → ℝ → Prop),
  perpendicular l1 line_l2 →
  tangent l1 circle_C →
  (∀ x y, l1 x y ↔ (3*x + 4*y + 14 = 0 ∨ 3*x + 4*y - 6 = 0)) :=
sorry

end NUMINAMATH_CALUDE_line_l1_equation_l988_98847


namespace NUMINAMATH_CALUDE_arithmetic_sum_1000_l988_98880

theorem arithmetic_sum_1000 : 
  ∀ m n : ℕ+, 
    (Finset.sum (Finset.range (m + 1)) (λ i => n + i) = 1000) ↔ 
    ((m = 4 ∧ n = 198) ∨ (m = 24 ∧ n = 28)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_1000_l988_98880


namespace NUMINAMATH_CALUDE_not_necessarily_divisible_by_44_l988_98896

theorem not_necessarily_divisible_by_44 (k : ℤ) (n : ℤ) : 
  n = k * (k + 1) * (k + 2) → 
  11 ∣ n → 
  ¬ (∀ m : ℤ, n = k * (k + 1) * (k + 2) ∧ 11 ∣ n → 44 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_divisible_by_44_l988_98896


namespace NUMINAMATH_CALUDE_max_value_of_expression_l988_98886

theorem max_value_of_expression (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 3) : 
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 729/108 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l988_98886


namespace NUMINAMATH_CALUDE_cube_volume_percentage_l988_98843

theorem cube_volume_percentage (box_length box_width box_height cube_side : ℕ) 
  (h1 : box_length = 8)
  (h2 : box_width = 6)
  (h3 : box_height = 12)
  (h4 : cube_side = 4) :
  (((box_length / cube_side) * (box_width / cube_side) * (box_height / cube_side) * cube_side^3) : ℚ) /
  (box_length * box_width * box_height) = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_percentage_l988_98843


namespace NUMINAMATH_CALUDE_largest_even_odd_two_digit_l988_98851

-- Define the set of two-digit numbers
def TwoDigitNumbers : Set Nat := {n : Nat | 10 ≤ n ∧ n ≤ 99}

-- Define even numbers
def IsEven (n : Nat) : Prop := ∃ k : Nat, n = 2 * k

-- Define odd numbers
def IsOdd (n : Nat) : Prop := ∃ k : Nat, n = 2 * k + 1

-- Theorem statement
theorem largest_even_odd_two_digit :
  (∀ n ∈ TwoDigitNumbers, IsEven n → n ≤ 98) ∧
  (∃ n ∈ TwoDigitNumbers, IsEven n ∧ n = 98) ∧
  (∀ n ∈ TwoDigitNumbers, IsOdd n → n ≤ 99) ∧
  (∃ n ∈ TwoDigitNumbers, IsOdd n ∧ n = 99) :=
sorry

end NUMINAMATH_CALUDE_largest_even_odd_two_digit_l988_98851


namespace NUMINAMATH_CALUDE_remainder_3_pow_2000_mod_17_l988_98856

theorem remainder_3_pow_2000_mod_17 : 3^2000 % 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2000_mod_17_l988_98856


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l988_98855

theorem roots_sum_and_product (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 4 = 0 → 
  x₂^2 - 2*x₂ - 4 = 0 → 
  x₁ + x₂ + x₁*x₂ = -2 :=
by sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l988_98855


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l988_98871

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 7 / 5)
  (hdb : d / b = 1 / 9) :
  a / c = 112.5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l988_98871


namespace NUMINAMATH_CALUDE_selection_theorem_l988_98874

/-- The number of ways to choose a president, vice-president, and 2-person committee from 10 people -/
def selection_ways (n : ℕ) : ℕ :=
  n * (n - 1) * (Nat.choose (n - 2) 2)

/-- Theorem stating the number of ways to make the selection -/
theorem selection_theorem :
  selection_ways 10 = 2520 :=
by sorry

end NUMINAMATH_CALUDE_selection_theorem_l988_98874


namespace NUMINAMATH_CALUDE_profit_percent_l988_98814

theorem profit_percent (P : ℝ) (C : ℝ) (h1 : C > 0) (h2 : (2/3) * P = 0.88 * C) :
  (P - C) / C = 0.32 := by
sorry

end NUMINAMATH_CALUDE_profit_percent_l988_98814


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l988_98833

theorem largest_prime_factor_of_expression : 
  (Nat.factors (16^4 + 2 * 16^2 + 1 - 13^4)).maximum = some 71 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l988_98833


namespace NUMINAMATH_CALUDE_bowl_glass_pairing_l988_98845

theorem bowl_glass_pairing (n : ℕ) (h : n = 5) : 
  (n : ℕ) * (n - 1) = 20 := by
  sorry

end NUMINAMATH_CALUDE_bowl_glass_pairing_l988_98845


namespace NUMINAMATH_CALUDE_abs_two_over_z_minus_z_equals_two_l988_98879

/-- Given a complex number z = 1 + i, prove that |2/z - z| = 2 -/
theorem abs_two_over_z_minus_z_equals_two :
  let z : ℂ := 1 + Complex.I
  Complex.abs (2 / z - z) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_two_over_z_minus_z_equals_two_l988_98879


namespace NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l988_98842

theorem discriminant_of_specific_quadratic (a b c : ℝ) : 
  a = 1 → b = -2 → c = 1 → b^2 - 4*a*c = 0 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_of_specific_quadratic_l988_98842


namespace NUMINAMATH_CALUDE_infinite_power_tower_four_equals_sqrt_two_l988_98859

/-- The infinite power tower function -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := 
  Real.log x / Real.log (Real.log x)

/-- Theorem: If the infinite power tower of x equals 4, then x equals √2 -/
theorem infinite_power_tower_four_equals_sqrt_two :
  ∀ x : ℝ, x > 0 → infinitePowerTower x = 4 → x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_power_tower_four_equals_sqrt_two_l988_98859


namespace NUMINAMATH_CALUDE_f_5_equals_357_l988_98854

def f (n : ℕ) : ℕ := 2 * n^3 + 3 * n^2 + 5 * n + 7

theorem f_5_equals_357 : f 5 = 357 := by sorry

end NUMINAMATH_CALUDE_f_5_equals_357_l988_98854


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l988_98836

theorem absolute_value_equation_product (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, (|5 * x₁| - 7 = 28) ∧ (|5 * x₂| - 7 = 28) ∧ (x₁ ≠ x₂) ∧ (x₁ * x₂ = -49)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l988_98836


namespace NUMINAMATH_CALUDE_circle_radius_values_l988_98899

/-- Given a circle and its tangent line, prove the possible values of its radius -/
theorem circle_radius_values (r : ℝ) (k : ℝ) : 
  r > 0 → 
  (∀ x y, (x - 1)^2 + (y - 3 * Real.sqrt 3)^2 = r^2) →
  (∃ x y, y = k * x + Real.sqrt 3) →
  (k = Real.sqrt 3 ∨ k = -Real.sqrt 3) →
  (r = Real.sqrt 3 / 2 ∨ r = 3 * Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_values_l988_98899


namespace NUMINAMATH_CALUDE_car_rental_rates_l988_98805

/-- The daily rate of the first car rental company -/
def daily_rate : ℝ := 21.95

/-- The per-mile rate of the first car rental company -/
def first_company_per_mile : ℝ := 0.19

/-- The fixed rate of City Rentals -/
def city_rentals_fixed : ℝ := 18.95

/-- The per-mile rate of City Rentals -/
def city_rentals_per_mile : ℝ := 0.21

/-- The number of miles at which the costs are equal -/
def equal_cost_miles : ℝ := 150.0

theorem car_rental_rates :
  daily_rate + first_company_per_mile * equal_cost_miles =
  city_rentals_fixed + city_rentals_per_mile * equal_cost_miles :=
by sorry

end NUMINAMATH_CALUDE_car_rental_rates_l988_98805


namespace NUMINAMATH_CALUDE_vector_computation_l988_98862

theorem vector_computation :
  let v1 : Fin 2 → ℝ := ![3, -5]
  let v2 : Fin 2 → ℝ := ![-1, 6]
  let v3 : Fin 2 → ℝ := ![2, -4]
  2 • v1 + 4 • v2 - 3 • v3 = ![(-4 : ℝ), 26] :=
by sorry

end NUMINAMATH_CALUDE_vector_computation_l988_98862


namespace NUMINAMATH_CALUDE_vector_addition_proof_l988_98824

def a : Fin 2 → ℝ := ![1, -2]
def b : Fin 2 → ℝ := ![3, 5]

theorem vector_addition_proof : 
  (2 • a + b) = ![5, 1] := by sorry

end NUMINAMATH_CALUDE_vector_addition_proof_l988_98824


namespace NUMINAMATH_CALUDE_bobs_family_children_l988_98891

/-- Given the following conditions about Bob's family and apple consumption:
  * Bob picked 450 apples in total
  * There are 40 adults in the family
  * Each adult ate 3 apples
  * Each child ate 10 apples
  This theorem proves that there are 33 children in Bob's family. -/
theorem bobs_family_children (total_apples : ℕ) (num_adults : ℕ) (apples_per_adult : ℕ) (apples_per_child : ℕ) :
  total_apples = 450 →
  num_adults = 40 →
  apples_per_adult = 3 →
  apples_per_child = 10 →
  (total_apples - num_adults * apples_per_adult) / apples_per_child = 33 :=
by sorry

end NUMINAMATH_CALUDE_bobs_family_children_l988_98891


namespace NUMINAMATH_CALUDE_tangent_line_equation_l988_98821

noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.log x

theorem tangent_line_equation :
  let A : ℝ × ℝ := (1, f 1)
  let m : ℝ := deriv f 1
  (λ (x y : ℝ) => x + y - 2 = 0) = (λ (x y : ℝ) => y - A.2 = m * (x - A.1)) := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l988_98821


namespace NUMINAMATH_CALUDE_inequality_implies_k_bound_l988_98846

theorem inequality_implies_k_bound :
  (∃ x : ℝ, |x + 1| - |x - 2| < k) → k > -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_k_bound_l988_98846


namespace NUMINAMATH_CALUDE_positive_reals_inequality_l988_98819

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) : x^2 + y^2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_reals_inequality_l988_98819


namespace NUMINAMATH_CALUDE_complex_equation_solution_l988_98881

theorem complex_equation_solution (z : ℂ) (b : ℝ) :
  z * (1 + Complex.I) = 1 - b * Complex.I →
  Complex.abs z = Real.sqrt 2 →
  b = Real.sqrt 3 ∨ b = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l988_98881


namespace NUMINAMATH_CALUDE_fraction_decimal_conversions_l988_98825

-- Define a function to round a rational number to n decimal places
def round_to_decimal_places (q : ℚ) (n : ℕ) : ℚ :=
  (↑(round (q * 10^n)) / 10^n)

theorem fraction_decimal_conversions :
  -- 1. 60/4 = 15 in both fraction and decimal form
  (60 : ℚ) / 4 = 15 ∧ 
  -- 2. 19/6 ≈ 3.167 when rounded to three decimal places
  round_to_decimal_places ((19 : ℚ) / 6) 3 = (3167 : ℚ) / 1000 ∧
  -- 3. 0.25 = 1/4
  (1 : ℚ) / 4 = (25 : ℚ) / 100 ∧
  -- 4. 0.08 = 2/25
  (2 : ℚ) / 25 = (8 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_decimal_conversions_l988_98825


namespace NUMINAMATH_CALUDE_henri_reads_1800_words_l988_98890

/-- Calculates the number of words read given total free time, movie durations, and reading rate. -/
def words_read (total_time : ℝ) (movie1_duration : ℝ) (movie2_duration : ℝ) (reading_rate : ℝ) : ℝ :=
  (total_time - movie1_duration - movie2_duration) * reading_rate * 60

/-- Proves that Henri reads 1800 words given the specified conditions. -/
theorem henri_reads_1800_words :
  words_read 8 3.5 1.5 10 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_henri_reads_1800_words_l988_98890


namespace NUMINAMATH_CALUDE_coefficient_b_is_zero_l988_98815

/-- Given an equation px + qy + bz = 1 with three solutions, prove that b = 0 -/
theorem coefficient_b_is_zero
  (p q b a : ℝ)
  (h1 : q * (3 * a) + b * 1 = 1)
  (h2 : p * (9 * a) + q * (-1) + b * 2 = 1)
  (h3 : q * (3 * a) = 1) :
  b = 0 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_b_is_zero_l988_98815


namespace NUMINAMATH_CALUDE_jake_and_sister_weight_l988_98804

/-- Jake's current weight in pounds -/
def jakes_weight : ℕ := 196

/-- Jake's sister's weight in pounds -/
def sisters_weight : ℕ := (jakes_weight - 8) / 2

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℕ := jakes_weight + sisters_weight

/-- Theorem stating that the combined weight of Jake and his sister is 290 pounds -/
theorem jake_and_sister_weight : combined_weight = 290 := by
  sorry

/-- Lemma stating that if Jake loses 8 pounds, he will weigh twice as much as his sister -/
lemma jake_twice_sister_weight : jakes_weight - 8 = 2 * sisters_weight := by
  sorry

end NUMINAMATH_CALUDE_jake_and_sister_weight_l988_98804


namespace NUMINAMATH_CALUDE_steven_arrangement_count_l988_98897

/-- The number of letters in "STEVEN" excluding one "E" -/
def n : ℕ := 5

/-- The number of permutations of "STEVEN" with one "E" fixed at the end -/
def steven_permutations : ℕ := n.factorial

theorem steven_arrangement_count : steven_permutations = 120 := by
  sorry

end NUMINAMATH_CALUDE_steven_arrangement_count_l988_98897


namespace NUMINAMATH_CALUDE_angle_D_measure_l988_98866

theorem angle_D_measure (A B C D : ℝ) :
  -- ABCD is a convex quadrilateral (implied by the angle sum condition)
  A + B + C + D = 360 →
  -- ∠C = 57°
  C = 57 →
  -- sin ∠A + sin ∠B = √2
  Real.sin A + Real.sin B = Real.sqrt 2 →
  -- cos ∠A + cos ∠B = 2 - √2
  Real.cos A + Real.cos B = 2 - Real.sqrt 2 →
  -- Then ∠D = 168°
  D = 168 := by
sorry

end NUMINAMATH_CALUDE_angle_D_measure_l988_98866


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l988_98887

/-- Given a quadratic inequality with specific properties, prove certain statements about its coefficients and solutions. -/
theorem quadratic_inequality_properties
  (a b : ℝ) (d : ℝ)
  (h_a_pos : a > 0)
  (h_solution_set : ∀ x : ℝ, x^2 + a*x + b > 0 ↔ x ≠ d) :
  (a^2 = 4*b) ∧
  (a^2 + 1/b ≥ 4) ∧
  (∀ c x₁ x₂ : ℝ, (∀ x : ℝ, x^2 + a*x + b < c ↔ x₁ < x ∧ x < x₂) →
    |x₁ - x₂| = 4 → c = 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l988_98887


namespace NUMINAMATH_CALUDE_mans_rate_in_still_water_l988_98860

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h1 : speed_with_stream = 26)
  (h2 : speed_against_stream = 12) :
  (speed_with_stream + speed_against_stream) / 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_in_still_water_l988_98860


namespace NUMINAMATH_CALUDE_ellipse_sum_l988_98893

-- Define the ellipse
def Ellipse (F₁ F₂ : ℝ × ℝ) (d : ℝ) :=
  {P : ℝ × ℝ | dist P F₁ + dist P F₂ = d}

-- Define the foci
def F₁ : ℝ × ℝ := (0, 0)
def F₂ : ℝ × ℝ := (6, 0)

-- Define the distance sum
def d : ℝ := 10

-- Theorem statement
theorem ellipse_sum (h k a b : ℝ) :
  Ellipse F₁ F₂ d →
  (∀ (x y : ℝ), (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1 ↔ (x, y) ∈ Ellipse F₁ F₂ d) →
  h + k + a + b = 12 := by sorry

end NUMINAMATH_CALUDE_ellipse_sum_l988_98893


namespace NUMINAMATH_CALUDE_second_person_share_correct_l988_98811

/-- Represents the rent sharing scenario -/
structure RentSharing where
  total_rent : ℕ
  base_share : ℕ
  first_multiplier : ℕ
  second_multiplier : ℕ
  third_multiplier : ℕ

/-- Calculates the share of the second person -/
def second_person_share (rs : RentSharing) : ℕ :=
  rs.base_share * rs.second_multiplier

/-- Theorem stating the correct share for the second person -/
theorem second_person_share_correct (rs : RentSharing) 
  (h1 : rs.total_rent = 5400)
  (h2 : rs.first_multiplier = 5)
  (h3 : rs.second_multiplier = 3)
  (h4 : rs.third_multiplier = 1)
  (h5 : rs.total_rent = rs.base_share * (rs.first_multiplier + rs.second_multiplier + rs.third_multiplier)) :
  second_person_share rs = 1800 := by
  sorry

#eval second_person_share { total_rent := 5400, base_share := 600, first_multiplier := 5, second_multiplier := 3, third_multiplier := 1 }

end NUMINAMATH_CALUDE_second_person_share_correct_l988_98811


namespace NUMINAMATH_CALUDE_partnership_investment_l988_98852

/-- Given the investments of partners A and B, the total profit, and A's share of the profit,
    calculate the investment of partner C in a partnership business. -/
theorem partnership_investment (a b total_profit a_profit : ℕ) (ha : a = 6300) (hb : b = 4200) 
    (h_total_profit : total_profit = 12200) (h_a_profit : a_profit = 3660) : 
    ∃ c : ℕ, c = 10490 ∧ a * total_profit = a_profit * (a + b + c) :=
by sorry

end NUMINAMATH_CALUDE_partnership_investment_l988_98852


namespace NUMINAMATH_CALUDE_quadratic_form_h_l988_98883

theorem quadratic_form_h (a k h : ℝ) : 
  (∀ x, 8 * x^2 + 12 * x + 7 = a * (x - h)^2 + k) → h = -3/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_h_l988_98883


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l988_98857

theorem hexagon_angle_measure (F I U R G E : ℝ) : 
  -- Hexagon angle sum is 720°
  F + I + U + R + G + E = 720 →
  -- Four angles are congruent
  F = I ∧ F = U ∧ F = R →
  -- G and E are supplementary
  G + E = 180 →
  -- Prove that E is 45°
  E = 45 := by
sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l988_98857


namespace NUMINAMATH_CALUDE_equation_solution_l988_98801

theorem equation_solution : 
  ∃ x : ℚ, (3 / 4 - 2 / 5 : ℚ) = 1 / x ∧ x = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l988_98801


namespace NUMINAMATH_CALUDE_cycling_competition_problem_l988_98864

/-- Represents the distance Natalia rode on each day of the week -/
structure CyclingDistance where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ

/-- The cycling competition problem -/
theorem cycling_competition_problem (d : CyclingDistance) : 
  d.tuesday = 50 ∧ 
  d.wednesday = 0.5 * d.tuesday ∧ 
  d.thursday = d.monday + d.wednesday ∧ 
  d.monday + d.tuesday + d.wednesday + d.thursday = 180 →
  d.monday = 40 := by
sorry


end NUMINAMATH_CALUDE_cycling_competition_problem_l988_98864


namespace NUMINAMATH_CALUDE_train_travel_time_l988_98829

/-- Given a train that travels 270 miles in 3 hours, prove that it takes 2 hours to travel an additional 180 miles at the same rate. -/
theorem train_travel_time (initial_distance : ℝ) (initial_time : ℝ) (additional_distance : ℝ) :
  initial_distance = 270 →
  initial_time = 3 →
  additional_distance = 180 →
  (additional_distance / (initial_distance / initial_time)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_l988_98829


namespace NUMINAMATH_CALUDE_apollo_wheel_replacement_ratio_l988_98885

/-- Represents the chariot wheel replacement scenario -/
structure WheelReplacement where
  initial_rate : ℕ  -- Initial rate in golden apples
  months : ℕ        -- Total number of months
  half_year : ℕ     -- Number of months before rate change
  total_payment : ℕ -- Total payment for the year

/-- Calculates the ratio of new rate to old rate -/
def rate_ratio (w : WheelReplacement) : ℚ :=
  let first_half_payment := w.initial_rate * w.half_year
  let second_half_payment := w.total_payment - first_half_payment
  (second_half_payment : ℚ) / (w.initial_rate * (w.months - w.half_year))

/-- Theorem stating that the rate ratio is 2 for the given scenario -/
theorem apollo_wheel_replacement_ratio :
  let w : WheelReplacement := ⟨3, 12, 6, 54⟩
  rate_ratio w = 2 := by
  sorry

end NUMINAMATH_CALUDE_apollo_wheel_replacement_ratio_l988_98885


namespace NUMINAMATH_CALUDE_cyclist_journey_l988_98873

theorem cyclist_journey (v t : ℝ) 
  (h1 : (v + 1) * (t - 0.5) = v * t)
  (h2 : (v - 1) * (t + 1) = v * t)
  : v * t = 6 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_journey_l988_98873


namespace NUMINAMATH_CALUDE_phyllis_gardens_tomato_percentage_l988_98849

/-- Represents a garden with a total number of plants and a fraction of tomato plants -/
structure Garden where
  total_plants : ℕ
  tomato_fraction : ℚ

/-- Calculates the percentage of tomato plants in two gardens combined -/
def combined_tomato_percentage (g1 g2 : Garden) : ℚ :=
  let total_plants := g1.total_plants + g2.total_plants
  let total_tomatoes := g1.total_plants * g1.tomato_fraction + g2.total_plants * g2.tomato_fraction
  (total_tomatoes / total_plants) * 100

/-- Theorem stating that the percentage of tomato plants in Phyllis's two gardens is 20% -/
theorem phyllis_gardens_tomato_percentage :
  let garden1 : Garden := { total_plants := 20, tomato_fraction := 1/10 }
  let garden2 : Garden := { total_plants := 15, tomato_fraction := 1/3 }
  combined_tomato_percentage garden1 garden2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_phyllis_gardens_tomato_percentage_l988_98849


namespace NUMINAMATH_CALUDE_tuesday_extra_minutes_l988_98861

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of minutes Ayen jogs on a regular weekday -/
def regular_jog : ℕ := 30

/-- The number of extra minutes Ayen jogged on Friday -/
def friday_extra : ℕ := 25

/-- The total number of minutes Ayen jogged this week -/
def total_jog : ℕ := 3 * 60

/-- The number of extra minutes Ayen jogged on Tuesday -/
def tuesday_extra : ℕ := total_jog - (weekdays * regular_jog) - friday_extra

theorem tuesday_extra_minutes : tuesday_extra = 5 := by sorry

end NUMINAMATH_CALUDE_tuesday_extra_minutes_l988_98861


namespace NUMINAMATH_CALUDE_range_of_m_l988_98848

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 4

def p (m : ℝ) : Prop := ∀ x ≥ 2, Monotone (f m)

def q (m : ℝ) : Prop := ∀ x, m*x^2 + 2*(m-2)*x + 1 > 0

theorem range_of_m :
  (∀ m : ℝ, (p m ∨ q m)) ∧ (¬∀ m : ℝ, (p m ∧ q m)) →
  ∀ m : ℝ, (m ∈ Set.Iic 1 ∪ Set.Ioo 2 4) ↔ (p m ∨ q m) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l988_98848


namespace NUMINAMATH_CALUDE_given_equation_is_quadratic_l988_98822

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation -/
def f (x : ℝ) : ℝ := 3 * x^2 - 5 * x

/-- Theorem: The given equation is a quadratic equation -/
theorem given_equation_is_quadratic : is_quadratic_equation f := by
  sorry

end NUMINAMATH_CALUDE_given_equation_is_quadratic_l988_98822


namespace NUMINAMATH_CALUDE_find_g_l988_98806

-- Define the functions f and g
def f : ℝ → ℝ := λ x ↦ 2 * x + 3

-- Define the property of g
def g_property (g : ℝ → ℝ) : Prop := ∀ x, g (x + 2) = f x

-- Theorem statement
theorem find_g : ∃ g : ℝ → ℝ, g_property g ∧ (∀ x, g x = 2 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_find_g_l988_98806


namespace NUMINAMATH_CALUDE_hourly_runoff_is_1000_l988_98872

/-- The total capacity of the sewers in gallons -/
def sewer_capacity : ℕ := 240000

/-- The number of days the sewers can handle rain before overflowing -/
def days_before_overflow : ℕ := 10

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the hourly runoff rate -/
def hourly_runoff_rate : ℕ := sewer_capacity / (days_before_overflow * hours_per_day)

/-- Theorem stating that the hourly runoff rate is 1000 gallons per hour -/
theorem hourly_runoff_is_1000 : hourly_runoff_rate = 1000 := by
  sorry

end NUMINAMATH_CALUDE_hourly_runoff_is_1000_l988_98872


namespace NUMINAMATH_CALUDE_hex_725_equals_octal_3445_l988_98865

-- Define a function to convert a base-16 number to base-10
def hexToDecimal (hex : String) : ℕ := sorry

-- Define a function to convert a base-10 number to base-8
def decimalToOctal (decimal : ℕ) : String := sorry

-- Theorem statement
theorem hex_725_equals_octal_3445 :
  decimalToOctal (hexToDecimal "725") = "3445" := by sorry

end NUMINAMATH_CALUDE_hex_725_equals_octal_3445_l988_98865


namespace NUMINAMATH_CALUDE_circle_symmetry_l988_98812

/-- Definition of the first circle C₁ -/
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

/-- Definition of the second circle C₂ -/
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

/-- Definition of the line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Function to check if two points are symmetric with respect to the line -/
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  symmetry_line ((x1 + x2) / 2) ((y1 + y2) / 2) ∧
  x2 - x1 = y2 - y1

/-- Theorem stating that C₂ is symmetric to C₁ with respect to the given line -/
theorem circle_symmetry :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle_C1 x1 y1 →
    circle_C2 x2 y2 →
    symmetric_points x1 y1 x2 y2 :=
by
  sorry


end NUMINAMATH_CALUDE_circle_symmetry_l988_98812


namespace NUMINAMATH_CALUDE_carter_baseball_cards_l988_98827

/-- Given that Marcus has 210 baseball cards and 58 more than Carter,
    prove that Carter has 152 baseball cards. -/
theorem carter_baseball_cards :
  let marcus_cards : ℕ := 210
  let difference : ℕ := 58
  let carter_cards : ℕ := marcus_cards - difference
  carter_cards = 152 :=
by sorry

end NUMINAMATH_CALUDE_carter_baseball_cards_l988_98827


namespace NUMINAMATH_CALUDE_complex_i_plus_i_squared_l988_98813

theorem complex_i_plus_i_squared : ∃ (i : ℂ), i * i = -1 ∧ i + i * i = -1 + i := by sorry

end NUMINAMATH_CALUDE_complex_i_plus_i_squared_l988_98813


namespace NUMINAMATH_CALUDE_exists_u_floor_power_minus_n_even_l988_98844

theorem exists_u_floor_power_minus_n_even :
  ∃ u : ℝ, u > 0 ∧ ∀ n : ℕ, n > 0 → ∃ k : ℤ, 
    (Int.floor (u ^ n) : ℤ) - n = 2 * k :=
by sorry

end NUMINAMATH_CALUDE_exists_u_floor_power_minus_n_even_l988_98844


namespace NUMINAMATH_CALUDE_sum_of_digits_power_minus_hundred_l988_98877

/-- Calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Calculates 10^n - 100 for n ≥ 2 -/
def power_minus_hundred (n : ℕ) : ℕ := 
  if n ≥ 2 then 10^n - 100 else 0

theorem sum_of_digits_power_minus_hundred : 
  sum_of_digits (power_minus_hundred 100) = 882 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_minus_hundred_l988_98877


namespace NUMINAMATH_CALUDE_matrix_P_satisfies_conditions_l988_98853

theorem matrix_P_satisfies_conditions : 
  let P : Matrix (Fin 2) (Fin 2) ℚ := !![2, -2/3; 3, -4]
  (P.mulVec ![4, 0] = ![8, 12]) ∧ 
  (P.mulVec ![2, -3] = ![2, -6]) := by
  sorry

end NUMINAMATH_CALUDE_matrix_P_satisfies_conditions_l988_98853


namespace NUMINAMATH_CALUDE_special_quad_integer_area_iff_conditions_l988_98840

/-- A quadrilateral ABCD with special properties -/
structure SpecialQuad where
  AB : ℝ
  CD : ℝ
  -- AB ⊥ BC and BC ⊥ CD
  perpendicular : True
  -- BC is tangent to a circle centered at O
  tangent : True
  -- AD is the diameter of the circle
  diameter : True

/-- The area of the special quadrilateral is an integer -/
def has_integer_area (q : SpecialQuad) : Prop :=
  ∃ n : ℕ, (q.AB + q.CD) * Real.sqrt (q.AB * q.CD) = n

/-- The product of AB and CD is a perfect square -/
def is_perfect_square_product (q : SpecialQuad) : Prop :=
  ∃ m : ℕ, q.AB * q.CD = m^2

theorem special_quad_integer_area_iff_conditions (q : SpecialQuad) :
  has_integer_area q ↔ is_perfect_square_product q ∧ has_integer_area q :=
sorry

end NUMINAMATH_CALUDE_special_quad_integer_area_iff_conditions_l988_98840


namespace NUMINAMATH_CALUDE_first_car_right_turn_distance_l988_98894

/-- The distance between two cars on a road --/
def distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - car1_distance - car2_distance

/-- The total distance traveled by the first car --/
def car1_total_distance (x : ℝ) : ℝ := 25 + x + 25

theorem first_car_right_turn_distance (initial_distance : ℝ) (car2_distance : ℝ) (final_distance : ℝ) :
  initial_distance = 113 ∧ 
  car2_distance = 35 ∧ 
  final_distance = 28 →
  ∃ x : ℝ, 
    car1_total_distance x + car2_distance = 
    distance_between_cars initial_distance 25 car2_distance + final_distance ∧
    x = 21 := by
  sorry

end NUMINAMATH_CALUDE_first_car_right_turn_distance_l988_98894


namespace NUMINAMATH_CALUDE_gcd_840_1764_l988_98818

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l988_98818


namespace NUMINAMATH_CALUDE_james_bag_weight_l988_98808

/-- The weight of James's bag given Oliver's bags' weights -/
theorem james_bag_weight (oliver_bag1 oliver_bag2 james_bag : ℝ) : 
  oliver_bag1 = (1 / 6) * james_bag →
  oliver_bag2 = (1 / 6) * james_bag →
  oliver_bag1 + oliver_bag2 = 6 →
  james_bag = 18 := by
  sorry

end NUMINAMATH_CALUDE_james_bag_weight_l988_98808


namespace NUMINAMATH_CALUDE_lineups_count_is_sixty_l988_98882

/-- The number of ways to arrange r items out of n items -/
def permutations (n : ℕ) (r : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - r)

/-- The number of possible lineups for 3 games with 3 athletes selected from 5 -/
def lineups_count : ℕ := permutations 5 3

theorem lineups_count_is_sixty : lineups_count = 60 := by sorry

end NUMINAMATH_CALUDE_lineups_count_is_sixty_l988_98882


namespace NUMINAMATH_CALUDE_complex_equation_problem_l988_98807

theorem complex_equation_problem (a b : ℝ) : 
  Complex.mk 1 (-2) = Complex.mk a b → a - b = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_problem_l988_98807


namespace NUMINAMATH_CALUDE_partition_rational_points_l988_98878

/-- Rational points in the plane -/
def RationalPoints : Set (ℚ × ℚ) :=
  {p : ℚ × ℚ | true}

/-- The theorem statement -/
theorem partition_rational_points :
  ∃ (A B : Set (ℚ × ℚ)),
    A ∩ B = ∅ ∧
    A ∪ B = RationalPoints ∧
    (∀ t : ℚ, Set.Finite {y : ℚ | (t, y) ∈ A}) ∧
    (∀ t : ℚ, Set.Finite {x : ℚ | (x, t) ∈ B}) :=
sorry

end NUMINAMATH_CALUDE_partition_rational_points_l988_98878
