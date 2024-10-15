import Mathlib

namespace NUMINAMATH_CALUDE_point_on_linear_function_l2978_297856

/-- Given that point P(a, b) is on the graph of y = -2x + 3, prove that 2a + b - 2 = 1 -/
theorem point_on_linear_function (a b : ℝ) (h : b = -2 * a + 3) : 2 * a + b - 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_function_l2978_297856


namespace NUMINAMATH_CALUDE_number_thought_of_l2978_297804

theorem number_thought_of (x : ℝ) : (x / 6 + 5 = 17) → x = 72 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l2978_297804


namespace NUMINAMATH_CALUDE_tangent_line_touches_both_curves_l2978_297896

noncomputable def curve1 (x : ℝ) : ℝ := x^2 - Real.log x

noncomputable def curve2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

noncomputable def tangent_line (x : ℝ) : ℝ := x

theorem tangent_line_touches_both_curves (a : ℝ) :
  (∀ x, x > 0 → curve1 x ≥ tangent_line x) ∧
  (curve1 1 = tangent_line 1) ∧
  (∀ x, curve2 a x ≥ tangent_line x) ∧
  (∃ x, curve2 a x = tangent_line x) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_touches_both_curves_l2978_297896


namespace NUMINAMATH_CALUDE_logarithmic_equation_solution_l2978_297850

/-- Given that 5(log_a x)^2 + 9(log_b x)^2 = (20(log x)^2) / (log a log b) and a, b, x > 1,
    prove that b = a^((20+√220)/10) or b = a^((20-√220)/10) -/
theorem logarithmic_equation_solution (a b x : ℝ) (ha : a > 1) (hb : b > 1) (hx : x > 1)
  (h : 5 * (Real.log x / Real.log a)^2 + 9 * (Real.log x / Real.log b)^2 = 20 * (Real.log x)^2 / (Real.log a * Real.log b)) :
  b = a^((20 + Real.sqrt 220) / 10) ∨ b = a^((20 - Real.sqrt 220) / 10) := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_solution_l2978_297850


namespace NUMINAMATH_CALUDE_dot_product_theorem_l2978_297842

def a : ℝ × ℝ := (1, 2)

theorem dot_product_theorem (b : ℝ × ℝ) 
  (h : (2 • a - b) = (3, 1)) : a • b = 5 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l2978_297842


namespace NUMINAMATH_CALUDE_triangle_with_sine_sides_l2978_297862

theorem triangle_with_sine_sides 
  (α β γ : Real) 
  (h_triangle : α + β + γ = π) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_less_than_pi : α < π ∧ β < π ∧ γ < π) : 
  ∃ (a b c : Real), 
    a = Real.sin α ∧ 
    b = Real.sin β ∧ 
    c = Real.sin γ ∧ 
    a + b > c ∧ 
    b + c > a ∧ 
    c + a > b := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_sine_sides_l2978_297862


namespace NUMINAMATH_CALUDE_min_rubles_to_win_l2978_297846

/-- Represents the state of the game --/
structure GameState :=
  (points : ℕ)
  (rubles : ℕ)

/-- Applies a move to the game state --/
def applyMove (state : GameState) (move : Bool) : GameState :=
  if move
  then { points := state.points * 2, rubles := state.rubles + 2 }
  else { points := state.points + 1, rubles := state.rubles + 1 }

/-- Checks if the game state is valid (not exceeding 50 points) --/
def isValidState (state : GameState) : Bool :=
  state.points <= 50

/-- Checks if the game is won (exactly 50 points) --/
def isWinningState (state : GameState) : Bool :=
  state.points = 50

/-- Theorem: The minimum number of rubles to win the game is 11 --/
theorem min_rubles_to_win :
  ∃ (moves : List Bool),
    let finalState := moves.foldl applyMove { points := 0, rubles := 0 }
    isWinningState finalState ∧
    finalState.rubles = 11 ∧
    (∀ (otherMoves : List Bool),
      let otherFinalState := otherMoves.foldl applyMove { points := 0, rubles := 0 }
      isWinningState otherFinalState →
      otherFinalState.rubles ≥ 11) :=
by
  sorry

end NUMINAMATH_CALUDE_min_rubles_to_win_l2978_297846


namespace NUMINAMATH_CALUDE_initial_number_proof_l2978_297897

theorem initial_number_proof : ∃ x : ℕ, 
  (↑x + 5.000000000000043 : ℝ) % 23 = 0 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l2978_297897


namespace NUMINAMATH_CALUDE_polygon_sides_sum_l2978_297874

/-- The sum of interior angles of a convex polygon with n sides --/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- The number of sides in a triangle --/
def triangle_sides : ℕ := 3

/-- The number of sides in a hexagon --/
def hexagon_sides : ℕ := 6

/-- The sum of interior angles of the given polygons --/
def total_angle_sum : ℝ := 1260

theorem polygon_sides_sum :
  ∃ (n : ℕ), n = triangle_sides + hexagon_sides ∧
  total_angle_sum = interior_angle_sum triangle_sides + interior_angle_sum hexagon_sides + interior_angle_sum (n - triangle_sides - hexagon_sides) :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_sum_l2978_297874


namespace NUMINAMATH_CALUDE_ellipse_constraint_l2978_297802

/-- An ellipse passing through (2,1) with |y| > 1 -/
def EllipseWithConstraint (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (4 / a^2 + 1 / b^2 = 1)

theorem ellipse_constraint (a b : ℝ) (h : EllipseWithConstraint a b) :
  {p : ℝ × ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 ∧ |p.2| > 1} =
  {p : ℝ × ℝ | p.1^2 + p.2^2 < 5 ∧ |p.2| > 1} := by
  sorry

end NUMINAMATH_CALUDE_ellipse_constraint_l2978_297802


namespace NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l2978_297800

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of possible digits (0 to 9) -/
def base : ℕ := 10

/-- The number of non-zero digits (1 to 9) -/
def non_zero_digits : ℕ := 9

/-- The total number of 6-digit numbers -/
def total_numbers : ℕ := non_zero_digits * base ^ (num_digits - 1)

/-- The number of 6-digit numbers with no zeros -/
def numbers_without_zero : ℕ := non_zero_digits ^ num_digits

/-- The number of 6-digit numbers with at least one zero -/
def numbers_with_zero : ℕ := total_numbers - numbers_without_zero

theorem six_digit_numbers_with_zero : numbers_with_zero = 368559 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_numbers_with_zero_l2978_297800


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2978_297826

theorem inequality_equivalence (x : ℝ) : (x - 5) / 2 + 1 > x - 3 ↔ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2978_297826


namespace NUMINAMATH_CALUDE_water_usage_for_car_cleaning_l2978_297853

/-- Represents the problem of calculating water usage for car cleaning --/
theorem water_usage_for_car_cleaning
  (total_water : ℝ)
  (plant_water_difference : ℝ)
  (plate_clothes_water : ℝ)
  (h1 : total_water = 65)
  (h2 : plant_water_difference = 11)
  (h3 : plate_clothes_water = 24)
  (h4 : plate_clothes_water * 2 = total_water - (2 * car_water + (2 * car_water - plant_water_difference))) :
  ∃ (car_water : ℝ), car_water = 7 :=
by sorry

end NUMINAMATH_CALUDE_water_usage_for_car_cleaning_l2978_297853


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2978_297849

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (10, 6)

/-- The line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 - Q.2 = m * (p.1 - Q.1)}

/-- The condition for non-intersection -/
def no_intersection (m : ℝ) : Prop :=
  line_through_Q m ∩ P = ∅

theorem parabola_line_intersection :
  ∃ (r s : ℝ), (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) → r + s = 40 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2978_297849


namespace NUMINAMATH_CALUDE_g_of_2_equals_6_l2978_297888

/-- The function g defined as g(x) = x³ - 2 for all real x -/
def g (x : ℝ) : ℝ := x^3 - 2

/-- Theorem stating that g(2) = 6 -/
theorem g_of_2_equals_6 : g 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_equals_6_l2978_297888


namespace NUMINAMATH_CALUDE_triangle_side_sum_l2978_297817

theorem triangle_side_sum (side_length : ℚ) (h : side_length = 14/8) : 
  3 * side_length = 21/4 := by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l2978_297817


namespace NUMINAMATH_CALUDE_john_earnings_proof_l2978_297820

def hours_per_workday : ℕ := 12
def days_in_month : ℕ := 30
def former_hourly_wage : ℚ := 20
def raise_percentage : ℚ := 30 / 100

def john_monthly_earnings : ℚ :=
  (days_in_month / 2) * hours_per_workday * (former_hourly_wage * (1 + raise_percentage))

theorem john_earnings_proof :
  john_monthly_earnings = 4680 := by
  sorry

end NUMINAMATH_CALUDE_john_earnings_proof_l2978_297820


namespace NUMINAMATH_CALUDE_percent_of_l_equal_to_75_percent_of_m_l2978_297869

-- Define variables
variable (j k l m : ℝ)

-- Define the conditions
def condition1 : Prop := 1.25 * j = 0.25 * k
def condition2 : Prop := 1.5 * k = 0.5 * l
def condition3 : Prop := 0.2 * m = 7 * j

-- Define the theorem
theorem percent_of_l_equal_to_75_percent_of_m 
  (h1 : condition1 j k)
  (h2 : condition2 k l)
  (h3 : condition3 j m) :
  ∃ x : ℝ, x / 100 * l = 0.75 * m ∧ x = 175 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_l_equal_to_75_percent_of_m_l2978_297869


namespace NUMINAMATH_CALUDE_square_area_error_l2978_297812

theorem square_area_error (x : ℝ) (h : x > 0) :
  let actual_edge := x
  let calculated_edge := x * (1 + 0.02)
  let actual_area := x^2
  let calculated_area := calculated_edge^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0404 := by sorry

end NUMINAMATH_CALUDE_square_area_error_l2978_297812


namespace NUMINAMATH_CALUDE_m_range_l2978_297811

def f (x : ℝ) : ℝ := -x^3 - 2*x^2 + 4*x

theorem m_range (m : ℝ) : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≥ m^2 - 14*m) → 
  m ∈ Set.Icc 3 11 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2978_297811


namespace NUMINAMATH_CALUDE_circle_intersection_range_l2978_297895

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 4}
def N (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}

-- State the theorem
theorem circle_intersection_range (r : ℝ) :
  r > 0 ∧ M ∩ N r = N r ↔ r ∈ Set.Ioo 0 (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l2978_297895


namespace NUMINAMATH_CALUDE_relationship_abc_l2978_297823

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.6 0.6
  let b : ℝ := Real.rpow 0.6 1.5
  let c : ℝ := Real.rpow 1.5 0.6
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l2978_297823


namespace NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_B_l2978_297851

def A : Set ℝ := {x | x + 1 ≤ 0 ∨ x - 4 ≥ 0}
def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

theorem intersection_nonempty (a : ℝ) :
  (A ∩ B a).Nonempty ↔ a ≤ -1/2 ∨ a = 2 := by sorry

theorem intersection_equals_B (a : ℝ) :
  A ∩ B a = B a ↔ a ≤ -1/2 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_intersection_equals_B_l2978_297851


namespace NUMINAMATH_CALUDE_kabadi_players_count_l2978_297898

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 30

/-- The number of people who play both kabadi and kho kho -/
def both_games : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 40

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := total_players - kho_kho_only + both_games

theorem kabadi_players_count : kabadi_players = 10 := by
  sorry

end NUMINAMATH_CALUDE_kabadi_players_count_l2978_297898


namespace NUMINAMATH_CALUDE_online_store_commission_percentage_l2978_297879

theorem online_store_commission_percentage 
  (cost : ℝ) 
  (online_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : cost = 17) 
  (h2 : online_price = 25.5) 
  (h3 : profit_percentage = 0.2) : 
  (online_price - (cost * (1 + profit_percentage))) / online_price = 0.2 := by
sorry

end NUMINAMATH_CALUDE_online_store_commission_percentage_l2978_297879


namespace NUMINAMATH_CALUDE_cosine_angle_in_ellipse_l2978_297880

/-- The cosine of the angle F₁PF₂ in an ellipse with specific properties -/
theorem cosine_angle_in_ellipse (P : ℝ × ℝ) :
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  let on_ellipse (p : ℝ × ℝ) := p.1^2 / 25 + p.2^2 / 9 = 1
  let triangle_area (p : ℝ × ℝ) := abs ((p.1 - F₁.1) * (p.2 - F₂.2) - (p.2 - F₁.2) * (p.1 - F₂.1)) / 2
  on_ellipse P ∧ triangle_area P = 3 * Real.sqrt 3 →
  let PF₁ := Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2)
  let PF₂ := Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)
  let cos_angle := (PF₁^2 + PF₂^2 - 64) / (2 * PF₁ * PF₂)
  cos_angle = 1/2 := by
sorry

end NUMINAMATH_CALUDE_cosine_angle_in_ellipse_l2978_297880


namespace NUMINAMATH_CALUDE_certain_number_proof_l2978_297845

theorem certain_number_proof : ∃ x : ℝ, (0.7 * x = 0.4 * 1050) ∧ (x = 600) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2978_297845


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_l2978_297860

/-- A parallelogram with adjacent sides of length 3 and 5 has a perimeter of 16. -/
theorem parallelogram_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 5) :
  2 * (a + b) = 16 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_l2978_297860


namespace NUMINAMATH_CALUDE_children_on_tricycles_l2978_297832

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of adults riding bicycles -/
def adults_on_bicycles : ℕ := 6

/-- The total number of wheels observed -/
def total_wheels : ℕ := 57

/-- Theorem stating that the number of children riding tricycles is 15 -/
theorem children_on_tricycles : 
  ∃ (c : ℕ), c * tricycle_wheels + adults_on_bicycles * bicycle_wheels = total_wheels ∧ c = 15 := by
  sorry

end NUMINAMATH_CALUDE_children_on_tricycles_l2978_297832


namespace NUMINAMATH_CALUDE_club_officer_selection_l2978_297865

/-- Represents the number of ways to choose officers in a club --/
def choose_officers (total_members : ℕ) (founding_members : ℕ) (positions : ℕ) : ℕ :=
  founding_members * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4)

/-- Theorem stating the number of ways to choose officers in the given scenario --/
theorem club_officer_selection :
  choose_officers 12 4 5 = 25920 := by
  sorry

end NUMINAMATH_CALUDE_club_officer_selection_l2978_297865


namespace NUMINAMATH_CALUDE_sqrt_81_division_l2978_297810

theorem sqrt_81_division :
  ∃ x : ℝ, x > 0 ∧ (Real.sqrt 81) / x = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_81_division_l2978_297810


namespace NUMINAMATH_CALUDE_game_result_l2978_297808

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 4
  else 0

def allie_rolls : List ℕ := [2, 6, 3, 1, 6]
def betty_rolls : List ℕ := [4, 6, 3, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result : 
  total_points allie_rolls * total_points betty_rolls = 1120 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2978_297808


namespace NUMINAMATH_CALUDE_cookie_count_bounds_l2978_297882

/-- Represents the number of cookies in a package -/
inductive PackageSize
| small : PackageSize  -- 6 cookies
| large : PackageSize  -- 12 cookies

/-- Profit from selling a package -/
def profit : PackageSize → ℕ
| PackageSize.small => 4
| PackageSize.large => 9

/-- Number of cookies in a package -/
def cookiesInPackage : PackageSize → ℕ
| PackageSize.small => 6
| PackageSize.large => 12

/-- Total profit from selling packages -/
def totalProfit : ℕ → ℕ → ℕ := λ x y => x * profit PackageSize.large + y * profit PackageSize.small

/-- Total number of cookies in packages -/
def totalCookies : ℕ → ℕ → ℕ := λ x y => x * cookiesInPackage PackageSize.large + y * cookiesInPackage PackageSize.small

theorem cookie_count_bounds :
  ∃ (x_min y_min x_max y_max : ℕ),
    totalProfit x_min y_min = 219 ∧
    totalProfit x_max y_max = 219 ∧
    totalCookies x_min y_min = 294 ∧
    totalCookies x_max y_max = 324 ∧
    (∀ x y, totalProfit x y = 219 → totalCookies x y ≥ 294 ∧ totalCookies x y ≤ 324) :=
by sorry

end NUMINAMATH_CALUDE_cookie_count_bounds_l2978_297882


namespace NUMINAMATH_CALUDE_similarity_transformation_result_l2978_297843

/-- A similarity transformation in 2D space -/
structure Similarity2D where
  center : ℝ × ℝ
  ratio : ℝ

/-- Apply a similarity transformation to a point -/
def apply_similarity (s : Similarity2D) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {(s.ratio * (p.1 - s.center.1) + s.center.1, s.ratio * (p.2 - s.center.2) + s.center.2),
   (-s.ratio * (p.1 - s.center.1) + s.center.1, -s.ratio * (p.2 - s.center.2) + s.center.2)}

theorem similarity_transformation_result :
  let s : Similarity2D := ⟨(0, 0), 2⟩
  let A : ℝ × ℝ := (2, 2)
  apply_similarity s A = {(4, 4), (-4, -4)} := by
  sorry

end NUMINAMATH_CALUDE_similarity_transformation_result_l2978_297843


namespace NUMINAMATH_CALUDE_max_x_minus_y_l2978_297858

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l2978_297858


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l2978_297872

/-- The time it takes for a train to pass a pole given its speed and the time it takes to cross a stationary train of known length -/
theorem train_passing_pole_time (v : ℝ) (t_cross : ℝ) (l_stationary : ℝ) :
  v = 64.8 →
  t_cross = 25 →
  l_stationary = 360 →
  ∃ t_pole : ℝ, abs (t_pole - 19.44) < 0.01 ∧ 
  t_pole = (v * t_cross - l_stationary) / v :=
by sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l2978_297872


namespace NUMINAMATH_CALUDE_largest_c_for_quadratic_range_l2978_297864

theorem largest_c_for_quadratic_range (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 2) ↔ c ≤ 11 :=
sorry

end NUMINAMATH_CALUDE_largest_c_for_quadratic_range_l2978_297864


namespace NUMINAMATH_CALUDE_range_of_a_l2978_297876

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then 3^x + 4*a else 2*x + a^2

theorem range_of_a (a : ℝ) (h₁ : a > 0) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2978_297876


namespace NUMINAMATH_CALUDE_range_of_a_in_second_quadrant_l2978_297818

/-- A complex number z = x + yi is in the second quadrant if x < 0 and y > 0 -/
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem range_of_a_in_second_quadrant (a : ℝ) :
  is_in_second_quadrant ((a - 2) + (a + 1) * I) ↔ -1 < a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_in_second_quadrant_l2978_297818


namespace NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l2978_297835

theorem root_in_interval_implies_m_range (m : ℝ) :
  (∃ x ∈ Set.Icc (-2) 1, 2 * m * x + 4 = 0) →
  m ∈ Set.Iic (-2) ∪ Set.Ici 1 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l2978_297835


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l2978_297854

theorem solution_to_system_of_equations :
  ∃! (x y : ℝ), (2*x + 3*y = (6-x) + (6-3*y)) ∧ (x - 2*y = (x-2) - (y+2)) ∧ x = -4 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l2978_297854


namespace NUMINAMATH_CALUDE_least_months_to_triple_l2978_297813

/-- The initial borrowed amount in dollars -/
def initial_amount : ℝ := 2000

/-- The monthly interest rate as a decimal -/
def monthly_rate : ℝ := 0.04

/-- The function that calculates the owed amount after t months -/
def owed_amount (t : ℕ) : ℝ := initial_amount * (1 + monthly_rate) ^ t

/-- Theorem stating that 30 is the least integer number of months 
    after which the owed amount exceeds three times the initial amount -/
theorem least_months_to_triple : 
  (∀ k : ℕ, k < 30 → owed_amount k ≤ 3 * initial_amount) ∧ 
  (owed_amount 30 > 3 * initial_amount) := by
  sorry

#check least_months_to_triple

end NUMINAMATH_CALUDE_least_months_to_triple_l2978_297813


namespace NUMINAMATH_CALUDE_john_toy_store_spending_l2978_297894

def weekly_allowance : ℚ := 240/100

theorem john_toy_store_spending (arcade_fraction : ℚ) (candy_store_amount : ℚ) 
  (h1 : arcade_fraction = 3/5)
  (h2 : candy_store_amount = 64/100) :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction)
  let toy_store_amount := remaining_after_arcade - candy_store_amount
  toy_store_amount / remaining_after_arcade = 1/3 := by sorry

end NUMINAMATH_CALUDE_john_toy_store_spending_l2978_297894


namespace NUMINAMATH_CALUDE_spencer_walk_distance_l2978_297819

/-- The total distance Spencer walked on his errands -/
def total_distance (house_to_library : ℝ) (library_to_post : ℝ) (post_to_grocery : ℝ) (grocery_to_coffee : ℝ) (coffee_to_house : ℝ) : ℝ :=
  house_to_library + library_to_post + post_to_grocery + grocery_to_coffee + coffee_to_house

/-- Theorem stating that Spencer walked 6.1 miles in total -/
theorem spencer_walk_distance :
  total_distance 1.2 0.8 1.5 0.6 2 = 6.1 := by
  sorry


end NUMINAMATH_CALUDE_spencer_walk_distance_l2978_297819


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2978_297801

theorem trigonometric_identity : 
  Real.sin (37 * π / 180) * Real.cos (34 * π / 180)^2 + 
  2 * Real.sin (34 * π / 180) * Real.cos (37 * π / 180) * Real.cos (34 * π / 180) - 
  Real.sin (37 * π / 180) * Real.sin (34 * π / 180)^2 = 
  (Real.sqrt 6 + Real.sqrt 2) / 4 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2978_297801


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2978_297861

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 3*x + 2 = 0 ↔ (x = 1 ∨ x = 2) := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2978_297861


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l2978_297828

theorem opposite_of_negative_two : 
  (∃ x : ℝ, -2 + x = 0) → (∃ x : ℝ, -2 + x = 0 ∧ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l2978_297828


namespace NUMINAMATH_CALUDE_student_competition_theorem_l2978_297873

/-- The number of ways students can sign up for competitions -/
def signup_ways (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_competitions ^ num_students

/-- The number of possible outcomes for championship winners -/
def championship_outcomes (num_students : ℕ) (num_competitions : ℕ) : ℕ :=
  num_students ^ num_competitions

/-- Theorem stating the correct number of ways for signup and championship outcomes -/
theorem student_competition_theorem :
  let num_students : ℕ := 5
  let num_competitions : ℕ := 4
  signup_ways num_students num_competitions = 4^5 ∧
  championship_outcomes num_students num_competitions = 5^4 := by
  sorry

end NUMINAMATH_CALUDE_student_competition_theorem_l2978_297873


namespace NUMINAMATH_CALUDE_kelly_nintendo_games_l2978_297824

/-- Proves that Kelly's initial number of Nintendo games is 121.0 --/
theorem kelly_nintendo_games :
  ∀ x : ℝ, (x - 99 = 22.0) → x = 121.0 := by
  sorry

end NUMINAMATH_CALUDE_kelly_nintendo_games_l2978_297824


namespace NUMINAMATH_CALUDE_valid_numbers_l2978_297827

def is_valid_number (n : ℕ) : Prop :=
  (1000 ≤ n) ∧ (n < 10000) ∧
  (∃ (a d : ℕ), 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    n = 120 * (10 * a + d))

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {1200, 2400, 3600, 4800} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l2978_297827


namespace NUMINAMATH_CALUDE_midpoint_condition_l2978_297857

-- Define the triangle ABC
structure RightTriangle where
  a : ℝ
  b : ℝ
  hypotenuse : ℝ
  hyp_eq : hypotenuse = Real.sqrt (a^2 + b^2)

-- Define point P on the hypotenuse
def PointOnHypotenuse (triangle : RightTriangle) := 
  { x : ℝ // 0 ≤ x ∧ x ≤ triangle.hypotenuse }

-- Define s as AP² + PB²
def s (triangle : RightTriangle) (p : PointOnHypotenuse triangle) : ℝ :=
  p.val^2 + (triangle.hypotenuse - p.val)^2

-- Define CP²
def CP_squared (triangle : RightTriangle) : ℝ := triangle.a^2

-- Theorem statement
theorem midpoint_condition (triangle : RightTriangle) :
  ∀ p : PointOnHypotenuse triangle, 
    s triangle p = 2 * CP_squared triangle ↔ 
    p.val = triangle.hypotenuse / 2 := by sorry

end NUMINAMATH_CALUDE_midpoint_condition_l2978_297857


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_l2978_297844

/-- Given a rectangular field with one side of 15 meters and an area of 120 square meters,
    the length of its diagonal is 17 meters. -/
theorem rectangular_field_diagonal (l w d : ℝ) : 
  l = 15 → 
  l * w = 120 → 
  d^2 = l^2 + w^2 → 
  d = 17 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_l2978_297844


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l2978_297833

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 4| + |3 - x| < a)) → a ∈ Set.Iic 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l2978_297833


namespace NUMINAMATH_CALUDE_liters_conversion_hours_conversion_cubic_meters_conversion_l2978_297814

-- Define conversion factors
def liters_to_milliliters : ℝ := 1000
def hours_per_day : ℝ := 24
def cubic_meters_to_cubic_centimeters : ℝ := 1000000

-- Theorem for 9.12 liters conversion
theorem liters_conversion (x : ℝ) (h : x = 9.12) :
  ∃ (l m : ℝ), x * liters_to_milliliters = l * liters_to_milliliters + m ∧ l = 9 ∧ m = 120 :=
sorry

-- Theorem for 4 hours conversion
theorem hours_conversion (x : ℝ) (h : x = 4) :
  x / hours_per_day = 1 / 6 :=
sorry

-- Theorem for 0.25 cubic meters conversion
theorem cubic_meters_conversion (x : ℝ) (h : x = 0.25) :
  x * cubic_meters_to_cubic_centimeters = 250000 :=
sorry

end NUMINAMATH_CALUDE_liters_conversion_hours_conversion_cubic_meters_conversion_l2978_297814


namespace NUMINAMATH_CALUDE_polynomial_roots_l2978_297805

def p (x : ℝ) : ℝ := x^3 - 3*x^2 - 4*x + 12

theorem polynomial_roots :
  (∀ x : ℝ, p x = 0 ↔ x = 2 ∨ x = -2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2978_297805


namespace NUMINAMATH_CALUDE_correct_equation_transformation_l2978_297807

theorem correct_equation_transformation (x : ℝ) : 
  3 * x - (2 - 4 * x) = 5 ↔ 3 * x + 4 * x - 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_transformation_l2978_297807


namespace NUMINAMATH_CALUDE_min_ratio_folded_strings_l2978_297863

theorem min_ratio_folded_strings (m n : ℕ) : 
  (∃ a : ℕ+, (2^m + 1 : ℕ) = a * (2^n + 1) ∧ a > 1) → 
  (∃ a : ℕ+, (2^m + 1 : ℕ) = a * (2^n + 1) ∧ a > 1 ∧ 
    ∀ b : ℕ+, (2^m + 1 : ℕ) = b * (2^n + 1) ∧ b > 1 → a ≤ b) → 
  (∃ m n : ℕ, (2^m + 1 : ℕ) = 3 * (2^n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_min_ratio_folded_strings_l2978_297863


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2978_297877

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℚ) := ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 7 + a 9 = 16) 
  (h_fourth : a 4 = 1) : 
  a 12 = 15 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2978_297877


namespace NUMINAMATH_CALUDE_sin_negative_225_degrees_l2978_297834

theorem sin_negative_225_degrees :
  Real.sin (-(225 * π / 180)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_225_degrees_l2978_297834


namespace NUMINAMATH_CALUDE_total_water_volume_is_10750_l2978_297838

def tank1_capacity : ℚ := 7000
def tank2_capacity : ℚ := 5000
def tank3_capacity : ℚ := 3000

def tank1_fill_ratio : ℚ := 3/4
def tank2_fill_ratio : ℚ := 4/5
def tank3_fill_ratio : ℚ := 1/2

def total_water_volume : ℚ := 
  tank1_capacity * tank1_fill_ratio + 
  tank2_capacity * tank2_fill_ratio + 
  tank3_capacity * tank3_fill_ratio

theorem total_water_volume_is_10750 : total_water_volume = 10750 := by
  sorry

end NUMINAMATH_CALUDE_total_water_volume_is_10750_l2978_297838


namespace NUMINAMATH_CALUDE_function_inequality_condition_l2978_297839

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = 4 * x + 3) →
  a > 0 →
  b > 0 →
  (∀ x : ℝ, |x + 3| < b → |f x + 5| < a) ↔
  b ≤ a / 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l2978_297839


namespace NUMINAMATH_CALUDE_faucet_filling_time_l2978_297852

/-- Given that four faucets can fill a 150-gallon tub in 8 minutes,
    prove that eight faucets will fill a 50-gallon tub in 4/3 minutes. -/
theorem faucet_filling_time 
  (volume_large : ℝ) 
  (volume_small : ℝ)
  (time_large : ℝ)
  (faucets_large : ℕ)
  (faucets_small : ℕ)
  (h1 : volume_large = 150)
  (h2 : volume_small = 50)
  (h3 : time_large = 8)
  (h4 : faucets_large = 4)
  (h5 : faucets_small = 8) :
  (volume_small * time_large * faucets_large) / (volume_large * faucets_small) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_faucet_filling_time_l2978_297852


namespace NUMINAMATH_CALUDE_math_club_composition_l2978_297830

theorem math_club_composition (boys girls : ℕ) : 
  boys = girls →
  (girls : ℚ) = 3/4 * (boys + girls - 1 : ℚ) →
  boys = 2 ∧ girls = 3 := by
sorry

end NUMINAMATH_CALUDE_math_club_composition_l2978_297830


namespace NUMINAMATH_CALUDE_sum_product_theorem_l2978_297815

def number_list : List ℕ := [2, 3, 4, 6]

theorem sum_product_theorem :
  ∃! (subset : Finset ℕ),
    subset.card = 3 ∧ 
    (∀ x ∈ subset, x ∈ number_list) ∧
    (subset.sum id = 11) ∧
    (subset.prod id = 36) :=
sorry

end NUMINAMATH_CALUDE_sum_product_theorem_l2978_297815


namespace NUMINAMATH_CALUDE_six_by_six_grid_squares_l2978_297841

/-- The number of squares of size n×n in a 6×6 grid -/
def squaresOfSize (n : ℕ) : ℕ := (6 - n) * (6 - n)

/-- The total number of squares in a 6×6 grid -/
def totalSquares : ℕ :=
  squaresOfSize 1 + squaresOfSize 2 + squaresOfSize 3 + squaresOfSize 4

theorem six_by_six_grid_squares :
  totalSquares = 54 := by
  sorry

end NUMINAMATH_CALUDE_six_by_six_grid_squares_l2978_297841


namespace NUMINAMATH_CALUDE_circle_C_properties_l2978_297847

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 2}

-- Define points A and B
def point_A : ℝ × ℝ := (4, 1)
def point_B : ℝ × ℝ := (2, 1)

-- Define the line x - y - 1 = 0
def tangent_line (p : ℝ × ℝ) : Prop := p.1 - p.2 - 1 = 0

-- Theorem stating the properties of circle C
theorem circle_C_properties :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  tangent_line point_B ∧
  (∀ p ∈ circle_C, (p.1 - 3)^2 + p.2^2 = 2) ∧
  (3, 0) ∈ circle_C ∧
  (∀ p ∈ circle_C, (p.1 - 3)^2 + p.2^2 = 2) :=
by
  sorry

#check circle_C_properties

end NUMINAMATH_CALUDE_circle_C_properties_l2978_297847


namespace NUMINAMATH_CALUDE_rectangular_box_volume_l2978_297875

/-- The volume of a rectangular box with face areas 24, 16, and 6 square inches is 48 cubic inches -/
theorem rectangular_box_volume (l w h : ℝ) 
  (area1 : l * w = 24)
  (area2 : w * h = 16)
  (area3 : l * h = 6) :
  l * w * h = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_volume_l2978_297875


namespace NUMINAMATH_CALUDE_exists_unformable_figure_l2978_297829

/-- Represents a geometric shape --/
inductive Shape
  | Square : Shape
  | Rectangle1x3 : Shape
  | Rectangle2x1 : Shape
  | LShape : Shape

/-- Represents a geometric figure --/
structure Figure where
  area : ℕ
  canBeFormed : Bool

/-- The set of available shapes --/
def availableShapes : List Shape :=
  [Shape.Square, Shape.Square, Shape.Rectangle1x3, Shape.Rectangle2x1, Shape.LShape]

/-- The total area of all available shapes --/
def totalArea : ℕ := 13

/-- There are eight different geometric figures --/
def figures : List Figure := sorry

/-- Theorem: There exists a figure that cannot be formed from the available shapes --/
theorem exists_unformable_figure :
  ∃ (f : Figure), f ∈ figures ∧ f.canBeFormed = false :=
sorry

end NUMINAMATH_CALUDE_exists_unformable_figure_l2978_297829


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2978_297881

theorem rectangle_perimeter (l w : ℝ) :
  l > 0 ∧ w > 0 ∧                             -- Positive dimensions
  2 * (w + l / 6) = 40 ∧                       -- Perimeter of smaller rectangle
  6 * w = l →                                  -- Relationship between l and w
  2 * (l + w) = 280 :=                         -- Perimeter of original rectangle
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l2978_297881


namespace NUMINAMATH_CALUDE_abs_a_minus_b_equals_eight_l2978_297887

theorem abs_a_minus_b_equals_eight (a b : ℝ) (h1 : a * b = 9) (h2 : a + b = 10) : 
  |a - b| = 8 := by
sorry

end NUMINAMATH_CALUDE_abs_a_minus_b_equals_eight_l2978_297887


namespace NUMINAMATH_CALUDE_product_equals_sqrt_ratio_l2978_297890

theorem product_equals_sqrt_ratio (a b c : ℝ) :
  a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1) →
  6 * 15 * 7 = (3/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_product_equals_sqrt_ratio_l2978_297890


namespace NUMINAMATH_CALUDE_cooking_and_weaving_count_l2978_297891

theorem cooking_and_weaving_count (total : ℕ) (yoga cooking weaving cooking_only cooking_and_yoga all : ℕ) 
  (h1 : yoga = 25)
  (h2 : cooking = 15)
  (h3 : weaving = 8)
  (h4 : cooking_only = 2)
  (h5 : cooking_and_yoga = 7)
  (h6 : all = 3) :
  cooking - (cooking_and_yoga + cooking_only) = 6 :=
by sorry

end NUMINAMATH_CALUDE_cooking_and_weaving_count_l2978_297891


namespace NUMINAMATH_CALUDE_triangle_inequality_l2978_297822

/-- Given three line segments of lengths a, 2, and 6, they can form a triangle if and only if 4 < a < 8 -/
theorem triangle_inequality (a : ℝ) : 
  (∃ (x y z : ℝ), x = a ∧ y = 2 ∧ z = 6 ∧ x + y > z ∧ y + z > x ∧ z + x > y) ↔ 4 < a ∧ a < 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2978_297822


namespace NUMINAMATH_CALUDE_journey_equations_correct_l2978_297866

/-- Represents a journey between two locations with uphill and flat sections. -/
structure Journey where
  uphill_speed : ℝ
  flat_speed : ℝ
  downhill_speed : ℝ
  time_ab : ℝ
  time_ba : ℝ

/-- The correct system of equations for the journey. -/
def correct_equations (j : Journey) (x y : ℝ) : Prop :=
  x / j.uphill_speed + y / j.flat_speed = j.time_ab / 60 ∧
  y / j.flat_speed + x / j.downhill_speed = j.time_ba / 60

/-- Theorem stating that the given system of equations is correct for the journey. -/
theorem journey_equations_correct (j : Journey) (x y : ℝ) 
    (h1 : j.uphill_speed = 3)
    (h2 : j.flat_speed = 4)
    (h3 : j.downhill_speed = 5)
    (h4 : j.time_ab = 70)
    (h5 : j.time_ba = 54) :
  correct_equations j x y :=
sorry

end NUMINAMATH_CALUDE_journey_equations_correct_l2978_297866


namespace NUMINAMATH_CALUDE_limit_nonexistent_l2978_297889

/-- The limit of (x^2 - y^2) / (x^2 + y^2) as x and y approach 0 does not exist. -/
theorem limit_nonexistent :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    x ≠ 0 ∧ y ≠ 0 ∧ x^2 + y^2 < δ^2 →
    |((x^2 - y^2) / (x^2 + y^2)) - L| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_nonexistent_l2978_297889


namespace NUMINAMATH_CALUDE_no_rational_solutions_l2978_297886

theorem no_rational_solutions (n : ℕ) (x y : ℚ) : (x + Real.sqrt 3 * y) ^ n ≠ Real.sqrt (1 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solutions_l2978_297886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2978_297803

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 9)
  (h_sum : a 7 + a 8 = 28) :
  a 4 = 7 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2978_297803


namespace NUMINAMATH_CALUDE_three_zeros_condition_l2978_297821

/-- The cubic function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- Theorem stating the condition for f to have exactly 3 real zeros -/
theorem three_zeros_condition (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 ∧
    ∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ↔ 
  a < -3 :=
sorry

end NUMINAMATH_CALUDE_three_zeros_condition_l2978_297821


namespace NUMINAMATH_CALUDE_door_replacement_cost_l2978_297899

/-- The total cost of replacing doors given the number of bedroom and outside doors,
    the cost of outside doors, and that bedroom doors cost half as much as outside doors. -/
def total_door_cost (num_bedroom_doors num_outside_doors outside_door_cost : ℕ) : ℕ :=
  num_outside_doors * outside_door_cost +
  num_bedroom_doors * (outside_door_cost / 2)

/-- Theorem stating that the total cost for replacing 3 bedroom doors and 2 outside doors
    is $70, given that outside doors cost $20 each and bedroom doors cost half as much. -/
theorem door_replacement_cost :
  total_door_cost 3 2 20 = 70 := by
  sorry


end NUMINAMATH_CALUDE_door_replacement_cost_l2978_297899


namespace NUMINAMATH_CALUDE_binomial_divisibility_l2978_297859

theorem binomial_divisibility (k n : ℕ) (hk : k > 1) (hn : n > 1) :
  let p := 2 * k - 1
  Prime p →
  p ∣ (n.choose 2 - k.choose 2) →
  p^2 ∣ (n.choose 2 - k.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_l2978_297859


namespace NUMINAMATH_CALUDE_cookout_buns_l2978_297867

/-- The number of packs of burger buns Alex needs to buy for his cookout. -/
def bun_packs_needed (guests : ℕ) (burgers_per_guest : ℕ) (no_meat_guests : ℕ) (no_bread_guests : ℕ) (buns_per_pack : ℕ) : ℕ :=
  let total_guests := guests - no_meat_guests
  let total_burgers := total_guests * burgers_per_guest
  let buns_needed := total_burgers - (no_bread_guests * burgers_per_guest)
  (buns_needed + buns_per_pack - 1) / buns_per_pack

theorem cookout_buns (guests : ℕ) (burgers_per_guest : ℕ) (no_meat_guests : ℕ) (no_bread_guests : ℕ) (buns_per_pack : ℕ)
    (h1 : guests = 10)
    (h2 : burgers_per_guest = 3)
    (h3 : no_meat_guests = 1)
    (h4 : no_bread_guests = 1)
    (h5 : buns_per_pack = 8) :
  bun_packs_needed guests burgers_per_guest no_meat_guests no_bread_guests buns_per_pack = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookout_buns_l2978_297867


namespace NUMINAMATH_CALUDE_roses_per_flat_l2978_297825

/-- Represents the number of flats of petunias -/
def petunia_flats : ℕ := 4

/-- Represents the number of petunias per flat -/
def petunias_per_flat : ℕ := 8

/-- Represents the number of flats of roses -/
def rose_flats : ℕ := 3

/-- Represents the number of Venus flytraps -/
def venus_flytraps : ℕ := 2

/-- Represents the amount of fertilizer needed for each petunia (in ounces) -/
def fertilizer_per_petunia : ℕ := 8

/-- Represents the amount of fertilizer needed for each rose (in ounces) -/
def fertilizer_per_rose : ℕ := 3

/-- Represents the amount of fertilizer needed for each Venus flytrap (in ounces) -/
def fertilizer_per_venus_flytrap : ℕ := 2

/-- Represents the total amount of fertilizer needed (in ounces) -/
def total_fertilizer : ℕ := 314

/-- Proves that the number of roses in each flat is 6 -/
theorem roses_per_flat : ℕ := by
  sorry

end NUMINAMATH_CALUDE_roses_per_flat_l2978_297825


namespace NUMINAMATH_CALUDE_index_card_problem_l2978_297831

theorem index_card_problem (n : ℕ+) : 
  ((n : ℝ) * (n + 1) * (2 * n + 1) / 6) / ((n : ℝ) * (n + 1) / 2) = 2023 → n = 3034 := by
  sorry

end NUMINAMATH_CALUDE_index_card_problem_l2978_297831


namespace NUMINAMATH_CALUDE_fourth_root_power_eight_l2978_297848

theorem fourth_root_power_eight : (((5 ^ (1/2)) ^ 5) ^ (1/4)) ^ 8 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_power_eight_l2978_297848


namespace NUMINAMATH_CALUDE_cupboard_books_count_l2978_297809

theorem cupboard_books_count :
  ∃! x : ℕ, x ≤ 400 ∧
    x % 4 = 1 ∧
    x % 5 = 1 ∧
    x % 6 = 1 ∧
    x % 7 = 0 ∧
    x = 301 := by
  sorry

end NUMINAMATH_CALUDE_cupboard_books_count_l2978_297809


namespace NUMINAMATH_CALUDE_brianna_marbles_l2978_297806

/-- Calculates the number of marbles Brianna has remaining after a series of events. -/
def remaining_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost - 2 * lost - lost / 2

/-- Theorem stating that Brianna has 10 marbles remaining given the initial conditions. -/
theorem brianna_marbles : remaining_marbles 24 4 = 10 := by
  sorry

#eval remaining_marbles 24 4

end NUMINAMATH_CALUDE_brianna_marbles_l2978_297806


namespace NUMINAMATH_CALUDE_reciprocal_sum_equals_one_l2978_297837

theorem reciprocal_sum_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1 / x + 1 / y = 1 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_equals_one_l2978_297837


namespace NUMINAMATH_CALUDE_divisible_by_101_l2978_297868

def repeat_two_digit (n : ℕ) : ℕ :=
  100000 * n + 1000 * n + n

theorem divisible_by_101 (n : ℕ) (h : n < 100) :
  101 ∣ repeat_two_digit n :=
sorry

end NUMINAMATH_CALUDE_divisible_by_101_l2978_297868


namespace NUMINAMATH_CALUDE_perimeter_of_isosceles_triangle_l2978_297892

-- Define the condition for x and y
def satisfies_equation (x y : ℝ) : Prop :=
  |x - 4| + Real.sqrt (y - 10) = 0

-- Define an isosceles triangle with side lengths x, y, and y
def isosceles_triangle (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + y > y ∧ y + y > x

-- Define the perimeter of the triangle
def triangle_perimeter (x y : ℝ) : ℝ :=
  x + y + y

-- Theorem statement
theorem perimeter_of_isosceles_triangle (x y : ℝ) :
  satisfies_equation x y → isosceles_triangle x y → triangle_perimeter x y = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_perimeter_of_isosceles_triangle_l2978_297892


namespace NUMINAMATH_CALUDE_complement_union_theorem_l2978_297871

def U : Finset Nat := {0, 1, 2, 3, 4, 5}
def A : Finset Nat := {1, 2, 3, 5}
def B : Finset Nat := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l2978_297871


namespace NUMINAMATH_CALUDE_line_l_equation_l2978_297870

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 1 = 0
def l₂ (x y : ℝ) : Prop := x + y + 2 = 0

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the property of l passing through P
def passes_through_P (l : ℝ → ℝ → Prop) : Prop := l P.1 P.2

-- Define the intersection points A and B
def A (l : ℝ → ℝ → Prop) : ℝ × ℝ := sorry
def B (l : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

-- Define the property of P being the midpoint of AB
def P_is_midpoint (l : ℝ → ℝ → Prop) : Prop :=
  P.1 = (A l).1 / 2 + (B l).1 / 2 ∧ P.2 = (A l).2 / 2 + (B l).2 / 2

-- Define the property of A and B being on l₁ and l₂ respectively
def A_on_l₁ (l : ℝ → ℝ → Prop) : Prop := l₁ (A l).1 (A l).2
def B_on_l₂ (l : ℝ → ℝ → Prop) : Prop := l₂ (B l).1 (B l).2

-- Define the equation of line l
def line_l (x y : ℝ) : Prop := 4 * x - y - 7 = 0

theorem line_l_equation : 
  ∀ l : ℝ → ℝ → Prop, 
    passes_through_P l → 
    P_is_midpoint l → 
    A_on_l₁ l → 
    B_on_l₂ l → 
    ∀ x y : ℝ, l x y ↔ line_l x y :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_l2978_297870


namespace NUMINAMATH_CALUDE_movie_ticket_distribution_l2978_297840

/-- The number of ways to distribute distinct objects to distinct recipients --/
def distribute_distinct (n_objects : ℕ) (n_recipients : ℕ) : ℕ :=
  (n_recipients - n_objects + 1).factorial / (n_recipients - n_objects).factorial

/-- The number of ways to distribute 3 different movie tickets among 10 people --/
theorem movie_ticket_distribution :
  distribute_distinct 3 10 = 720 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_distribution_l2978_297840


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l2978_297878

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  (b / a ≥ Real.sqrt 3) → e ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_theorem_l2978_297878


namespace NUMINAMATH_CALUDE_line_parameterization_l2978_297885

/-- Given a line y = 5x - 7 parameterized as (x, y) = (s, -3) + t(3, m),
    prove that s = 4/5 and m = 8 -/
theorem line_parameterization (s m : ℝ) : 
  (∀ t x y : ℝ, x = s + 3*t ∧ y = -3 + m*t → y = 5*x - 7) →
  s = 4/5 ∧ m = 8 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2978_297885


namespace NUMINAMATH_CALUDE_line_equation_from_triangle_l2978_297883

/-- Given a line passing through (-a, b) and intersecting the y-axis in the second quadrant,
    forming a triangle with area T and base ka along the x-axis, prove that
    the equation of the line is 2Tx - ka²y + ka²b + 2aT = 0 -/
theorem line_equation_from_triangle (a T k : ℝ) (b : ℝ) (hb : b ≠ 0) :
  ∃ (m c : ℝ), 
    (∀ x y, y = m * x + c ↔ 2 * T * x - k * a^2 * y + k * a^2 * b + 2 * a * T = 0) ∧
    m * (-a) + c = b ∧
    m > 0 ∧
    c > 0 ∧
    k > 0 ∧
    T = (1/2) * k * a * (c - b) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_triangle_l2978_297883


namespace NUMINAMATH_CALUDE_kittens_left_tim_kittens_left_l2978_297855

/-- Given an initial number of kittens and the number of kittens given to two people,
    calculate the number of kittens left. -/
theorem kittens_left (initial : ℕ) (given_to_jessica : ℕ) (given_to_sara : ℕ) :
  initial - (given_to_jessica + given_to_sara) = initial - given_to_jessica - given_to_sara :=
by sorry

/-- Prove that Tim has 9 kittens left after giving away some kittens. -/
theorem tim_kittens_left :
  let initial := 18
  let given_to_jessica := 3
  let given_to_sara := 6
  initial - (given_to_jessica + given_to_sara) = 9 :=
by sorry

end NUMINAMATH_CALUDE_kittens_left_tim_kittens_left_l2978_297855


namespace NUMINAMATH_CALUDE_solution_value_l2978_297893

theorem solution_value (a : ℝ) : (3 * 5 - 2 * a = 7) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2978_297893


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2978_297836

theorem max_value_of_expression (x y : Real) 
  (hx : 0 < x ∧ x < π/2) (hy : 0 < y ∧ y < π/2) : 
  (Real.sqrt (Real.sqrt (Real.sin x * Real.sin y))) / 
  (Real.sqrt (Real.sqrt (Real.tan x)) + Real.sqrt (Real.sqrt (Real.tan y))) 
  ≤ Real.sqrt (Real.sqrt 8) / 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2978_297836


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_120_over_8_l2978_297884

theorem largest_whole_number_less_than_120_over_8 : 
  (∀ n : ℕ, n > 14 → 8 * n ≥ 120) ∧ (8 * 14 < 120) := by
  sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_120_over_8_l2978_297884


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l2978_297816

theorem consecutive_squares_sum (n : ℕ) (h : n = 26) :
  (n - 1)^2 + n^2 + (n + 1)^2 = 2030 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l2978_297816
