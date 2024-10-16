import Mathlib

namespace NUMINAMATH_CALUDE_smallest_multiple_l385_38563

theorem smallest_multiple (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k) ∧ 
  (∃ m : ℕ, n - 6 = 73 * m) ∧
  (∀ x : ℕ, x < n → ¬((∃ k : ℕ, x = 17 * k) ∧ (∃ m : ℕ, x - 6 = 73 * m))) →
  n = 663 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l385_38563


namespace NUMINAMATH_CALUDE_diorama_building_time_l385_38553

/-- Given the total time spent on a diorama and the relationship between building and planning time,
    prove the time spent building the diorama. -/
theorem diorama_building_time (total_time planning_time building_time : ℕ) 
    (h1 : total_time = 67)
    (h2 : building_time = 3 * planning_time - 5)
    (h3 : total_time = planning_time + building_time) : 
    building_time = 49 := by
  sorry

end NUMINAMATH_CALUDE_diorama_building_time_l385_38553


namespace NUMINAMATH_CALUDE_four_positions_l385_38583

/-- Represents a cell in the 4x4 grid -/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the value in a cell -/
inductive CellValue
  | One
  | Two
  | Three
  | Four

/-- Represents the 4x4 grid -/
def Grid := Cell → Option CellValue

/-- Check if a 2x2 square is valid (contains 1, 2, 3, 4 exactly once) -/
def isValidSquare (g : Grid) (topLeft : Cell) : Prop := sorry

/-- Check if the entire grid is valid -/
def isValidGrid (g : Grid) : Prop := sorry

/-- The given partial grid -/
def partialGrid : Grid := sorry

/-- Theorem stating the positions of fours in the grid -/
theorem four_positions (g : Grid) 
  (h1 : isValidGrid g) 
  (h2 : g = partialGrid) : 
  g ⟨0, 2⟩ = some CellValue.Four ∧ 
  g ⟨1, 0⟩ = some CellValue.Four ∧ 
  g ⟨2, 1⟩ = some CellValue.Four ∧ 
  g ⟨3, 3⟩ = some CellValue.Four := by
  sorry

end NUMINAMATH_CALUDE_four_positions_l385_38583


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l385_38586

/-- An isosceles triangle with side lengths 3, 6, and 6 has a perimeter of 15. -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ c = 6 →  -- Two sides are 6, one side is 3
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 15  -- Perimeter is 15
:= by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l385_38586


namespace NUMINAMATH_CALUDE_ball_motion_problem_l385_38585

/-- Ball motion problem -/
theorem ball_motion_problem 
  (dist_A_to_wall : ℝ) 
  (dist_wall_to_B : ℝ) 
  (dist_AB : ℝ) 
  (initial_velocity : ℝ) 
  (acceleration : ℝ) 
  (h1 : dist_A_to_wall = 5)
  (h2 : dist_wall_to_B = 2)
  (h3 : dist_AB = 9)
  (h4 : initial_velocity = 5)
  (h5 : acceleration = -0.4) :
  ∃ (return_speed : ℝ) (required_initial_speed : ℝ),
    return_speed = 3 ∧ required_initial_speed = 4 := by
  sorry


end NUMINAMATH_CALUDE_ball_motion_problem_l385_38585


namespace NUMINAMATH_CALUDE_point_translation_second_quadrant_l385_38582

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point by a given vector -/
def translate (p : Point) (dx dy : ℝ) : Point :=
  { x := p.x + dx, y := p.y + dy }

/-- Check if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The main theorem -/
theorem point_translation_second_quadrant (m n : ℝ) :
  let A : Point := { x := m, y := n }
  let A' : Point := translate A 2 3
  isInSecondQuadrant A' → m < -2 ∧ n > -3 := by
  sorry

end NUMINAMATH_CALUDE_point_translation_second_quadrant_l385_38582


namespace NUMINAMATH_CALUDE_machine_output_for_26_l385_38576

def machine_operation (input : ℕ) : ℕ :=
  (input + 15) - 6

theorem machine_output_for_26 :
  machine_operation 26 = 35 := by
  sorry

end NUMINAMATH_CALUDE_machine_output_for_26_l385_38576


namespace NUMINAMATH_CALUDE_trigonometric_inequality_l385_38573

theorem trigonometric_inequality : 
  let a := Real.sin (2 * Real.pi / 7)
  let b := Real.cos (12 * Real.pi / 7)
  let c := Real.tan (9 * Real.pi / 7)
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_l385_38573


namespace NUMINAMATH_CALUDE_share_A_is_240_l385_38551

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_share_A (initial_A initial_B : ℕ) (withdraw_A advance_B : ℕ) (months : ℕ) (total_profit : ℕ) : ℕ :=
  let investment_months_A := initial_A * months + (initial_A - withdraw_A) * (12 - months)
  let investment_months_B := initial_B * months + (initial_B + advance_B) * (12 - months)
  let total_investment_months := investment_months_A + investment_months_B
  (investment_months_A * total_profit) / total_investment_months

theorem share_A_is_240 :
  calculate_share_A 3000 4000 1000 1000 8 630 = 240 := by
  sorry

#eval calculate_share_A 3000 4000 1000 1000 8 630

end NUMINAMATH_CALUDE_share_A_is_240_l385_38551


namespace NUMINAMATH_CALUDE_intersection_points_count_l385_38505

/-- Represents a line in the form ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Checks if a point (x, y) lies on a line --/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y = l.c

/-- The three lines given in the problem --/
def line1 : Line := { a := -3, b := 4, c := 2 }
def line2 : Line := { a := 2, b := 4, c := 4 }
def line3 : Line := { a := 6, b := -8, c := 3 }

theorem intersection_points_count :
  ∃ (p1 p2 : ℝ × ℝ),
    p1 ≠ p2 ∧
    (point_on_line p1.1 p1.2 line1 ∨ point_on_line p1.1 p1.2 line2 ∨ point_on_line p1.1 p1.2 line3) ∧
    (point_on_line p1.1 p1.2 line1 ∨ point_on_line p1.1 p1.2 line2 ∨ point_on_line p1.1 p1.2 line3) ∧
    (point_on_line p2.1 p2.2 line1 ∨ point_on_line p2.1 p2.2 line2 ∨ point_on_line p2.1 p2.2 line3) ∧
    (point_on_line p2.1 p2.2 line1 ∨ point_on_line p2.1 p2.2 line2 ∨ point_on_line p2.1 p2.2 line3) ∧
    (∀ (p : ℝ × ℝ),
      p ≠ p1 → p ≠ p2 →
      ¬((point_on_line p.1 p.2 line1 ∧ point_on_line p.1 p.2 line2) ∨
        (point_on_line p.1 p.2 line1 ∧ point_on_line p.1 p.2 line3) ∨
        (point_on_line p.1 p.2 line2 ∧ point_on_line p.1 p.2 line3))) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l385_38505


namespace NUMINAMATH_CALUDE_u_v_cube_sum_l385_38539

theorem u_v_cube_sum (u v : ℝ) (hu : u > 1) (hv : v > 1)
  (h : Real.log u / Real.log 4 ^ 3 + Real.log v / Real.log 5 ^ 3 + 9 = 
       9 * (Real.log u / Real.log 4) * (Real.log v / Real.log 5)) :
  u^3 + v^3 = 4^(9/2) + 5^(9/2) := by
sorry

end NUMINAMATH_CALUDE_u_v_cube_sum_l385_38539


namespace NUMINAMATH_CALUDE_sin_cos_sum_equality_l385_38508

theorem sin_cos_sum_equality : 
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (140 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equality_l385_38508


namespace NUMINAMATH_CALUDE_seventh_power_sum_l385_38541

theorem seventh_power_sum (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 2)
  (h2 : α₁^2 + α₂^2 + α₃^2 = 6)
  (h3 : α₁^3 + α₂^3 + α₃^3 = 14) :
  α₁^7 + α₂^7 + α₃^7 = 478 := by
  sorry

end NUMINAMATH_CALUDE_seventh_power_sum_l385_38541


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l385_38533

/-- Given a circular arrangement of students, if the 10th and 45th positions
    are opposite each other, then the total number of students is 70. -/
theorem circular_arrangement_students (n : ℕ) : 
  (10 + n / 2 ≡ 45 [MOD n]) → n = 70 := by sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l385_38533


namespace NUMINAMATH_CALUDE_power_of_i_third_quadrant_l385_38517

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Statement 1: i^2023 = -i
theorem power_of_i : i^2023 = -i := by sorry

-- Statement 2: -2-i is in the third quadrant
theorem third_quadrant : 
  let z : ℂ := -2 - i
  z.re < 0 ∧ z.im < 0 := by sorry

end NUMINAMATH_CALUDE_power_of_i_third_quadrant_l385_38517


namespace NUMINAMATH_CALUDE_cubic_meter_to_cubic_cm_l385_38506

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Theorem statement
theorem cubic_meter_to_cubic_cm :
  (1 : ℝ) * meters_to_cm^3 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_meter_to_cubic_cm_l385_38506


namespace NUMINAMATH_CALUDE_five_plumbers_three_areas_l385_38510

/-- The number of ways to assign plumbers to residential areas. -/
def assignment_plans (n_plumbers : ℕ) (n_areas : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem stating the number of assignment plans for 5 plumbers and 3 areas. -/
theorem five_plumbers_three_areas : 
  assignment_plans 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_five_plumbers_three_areas_l385_38510


namespace NUMINAMATH_CALUDE_target_hit_probability_l385_38554

theorem target_hit_probability 
  (prob_A prob_B prob_C : ℚ)
  (h_A : prob_A = 1/2)
  (h_B : prob_B = 1/3)
  (h_C : prob_C = 1/4) :
  1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_target_hit_probability_l385_38554


namespace NUMINAMATH_CALUDE_stock_price_uniqueness_l385_38580

theorem stock_price_uniqueness : ¬∃ (k l : ℕ), (117/100)^k * (83/100)^l = 1 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_uniqueness_l385_38580


namespace NUMINAMATH_CALUDE_derivative_exponential_sine_derivative_rational_function_derivative_logarithm_derivative_polynomial_product_derivative_cosine_l385_38535

-- Function 1: y = e^(sin x)
theorem derivative_exponential_sine (x : ℝ) :
  deriv (fun x => Real.exp (Real.sin x)) x = Real.exp (Real.sin x) * Real.cos x :=
sorry

-- Function 2: y = (x + 3) / (x + 2)
theorem derivative_rational_function (x : ℝ) :
  deriv (fun x => (x + 3) / (x + 2)) x = -1 / (x + 2)^2 :=
sorry

-- Function 3: y = ln(2x + 3)
theorem derivative_logarithm (x : ℝ) :
  deriv (fun x => Real.log (2 * x + 3)) x = 2 / (2 * x + 3) :=
sorry

-- Function 4: y = (x^2 + 2)(2x - 1)
theorem derivative_polynomial_product (x : ℝ) :
  deriv (fun x => (x^2 + 2) * (2 * x - 1)) x = 6 * x^2 - 2 * x + 4 :=
sorry

-- Function 5: y = cos(2x + π/3)
theorem derivative_cosine (x : ℝ) :
  deriv (fun x => Real.cos (2 * x + Real.pi / 3)) x = -2 * Real.sin (2 * x + Real.pi / 3) :=
sorry

end NUMINAMATH_CALUDE_derivative_exponential_sine_derivative_rational_function_derivative_logarithm_derivative_polynomial_product_derivative_cosine_l385_38535


namespace NUMINAMATH_CALUDE_volume_ratio_l385_38565

/-- A right rectangular prism with edge lengths 2, 3, and 5 -/
structure Prism where
  length : ℝ := 2
  width : ℝ := 3
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : Prism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
def volume_S (B : Prism) (r : ℝ) (a b c d : ℝ) : ℝ :=
  a * r^3 + b * r^2 + c * r + d

theorem volume_ratio (B : Prism) (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  b * c / (a * d) = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_l385_38565


namespace NUMINAMATH_CALUDE_mass_of_man_is_180_l385_38536

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sink_height water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * water_density

/-- Theorem stating that the mass of the man is 180 kg under the given conditions. -/
theorem mass_of_man_is_180 :
  let boat_length : ℝ := 6
  let boat_breadth : ℝ := 3
  let boat_sink_height : ℝ := 0.01
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth boat_sink_height water_density = 180 := by
  sorry

#eval mass_of_man 6 3 0.01 1000

end NUMINAMATH_CALUDE_mass_of_man_is_180_l385_38536


namespace NUMINAMATH_CALUDE_vector_equation_solution_l385_38581

theorem vector_equation_solution (α β : Real) :
  let A : Real × Real := (Real.cos α, Real.sin α)
  let B : Real × Real := (Real.cos β, Real.sin β)
  let C : Real × Real := (1/2, Real.sqrt 3/2)
  (C.1 = B.1 - A.1 ∧ C.2 = B.2 - A.2) → β = 2*Real.pi/3 ∨ β = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l385_38581


namespace NUMINAMATH_CALUDE_n_div_30_n_squared_cube_n_cubed_square_n_smallest_n_has_three_digits_l385_38587

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
theorem n_div_30 : 30 ∣ n := sorry

/-- n^2 is a perfect cube -/
theorem n_squared_cube : ∃ k : ℕ, n^2 = k^3 := sorry

/-- n^3 is a perfect square -/
theorem n_cubed_square : ∃ k : ℕ, n^3 = k^2 := sorry

/-- n is the smallest positive integer satisfying the conditions -/
theorem n_smallest : ∀ m : ℕ, m < n → ¬(30 ∣ m ∧ (∃ k : ℕ, m^2 = k^3) ∧ (∃ k : ℕ, m^3 = k^2)) := sorry

/-- The number of digits in n -/
def digits_of_n : ℕ := sorry

/-- Theorem stating that n has 3 digits -/
theorem n_has_three_digits : digits_of_n = 3 := sorry

end NUMINAMATH_CALUDE_n_div_30_n_squared_cube_n_cubed_square_n_smallest_n_has_three_digits_l385_38587


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l385_38516

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → a^2016 + b^2017 = 1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l385_38516


namespace NUMINAMATH_CALUDE_intersection_of_sets_l385_38572

theorem intersection_of_sets : 
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {1, 3, 4}
  M ∩ N = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l385_38572


namespace NUMINAMATH_CALUDE_sunday_only_papers_l385_38527

/-- The number of papers Kyle delivers in a week -/
def total_papers : ℕ := 720

/-- The number of houses Kyle delivers to from Monday to Saturday -/
def regular_houses : ℕ := 100

/-- The number of regular customers who don't receive the Sunday paper -/
def sunday_opt_out : ℕ := 10

/-- The number of days Kyle delivers from Monday to Saturday -/
def weekdays : ℕ := 6

theorem sunday_only_papers : 
  total_papers - (regular_houses * weekdays) - (regular_houses - sunday_opt_out) = 30 := by
  sorry

end NUMINAMATH_CALUDE_sunday_only_papers_l385_38527


namespace NUMINAMATH_CALUDE_three_sum_exists_l385_38564

theorem three_sum_exists (n : ℕ) (a : Fin (n + 1) → ℕ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_increasing : ∀ i j, i < j → a i < a j) 
  (h_bound : a (Fin.last n) < 2 * n) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i + a j = a k :=
sorry

end NUMINAMATH_CALUDE_three_sum_exists_l385_38564


namespace NUMINAMATH_CALUDE_smallest_alpha_inequality_half_satisfies_inequality_smallest_alpha_is_half_l385_38574

theorem smallest_alpha_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∀ α : ℝ, α > 0 → α < 1/2 →
    ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + y) / 2 < α * Real.sqrt (x * y) + (1 - α) * Real.sqrt ((x^2 + y^2) / 2) :=
by sorry

theorem half_satisfies_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / 2 ≥ (1/2) * Real.sqrt (x * y) + (1/2) * Real.sqrt ((x^2 + y^2) / 2) :=
by sorry

theorem smallest_alpha_is_half :
  ∀ α : ℝ, α > 0 →
    (∀ x y : ℝ, x > 0 → y > 0 →
      (x + y) / 2 ≥ α * Real.sqrt (x * y) + (1 - α) * Real.sqrt ((x^2 + y^2) / 2)) →
    α ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_alpha_inequality_half_satisfies_inequality_smallest_alpha_is_half_l385_38574


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_specific_proposition_l385_38548

theorem negation_of_universal_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 3 ≥ 0) ↔ (∃ x : ℝ, x^2 + 2*x + 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_negation_of_specific_proposition_l385_38548


namespace NUMINAMATH_CALUDE_train_passengers_l385_38599

theorem train_passengers (initial_passengers : ℕ) : 
  initial_passengers = 288 →
  let after_first := initial_passengers * 2 / 3 + 280
  let after_second := after_first / 2 + 12
  after_second = 248 := by
sorry

end NUMINAMATH_CALUDE_train_passengers_l385_38599


namespace NUMINAMATH_CALUDE_remainder_problem_l385_38552

theorem remainder_problem (N : ℕ) (D : ℕ) : 
  (N % 158 = 50) → (N % D = 13) → (D > 13) → (D < 158) → D = 37 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l385_38552


namespace NUMINAMATH_CALUDE_cube_root_1600_l385_38559

theorem cube_root_1600 (c d : ℕ+) (h1 : (1600 : ℝ)^(1/3) = c * d^(1/3)) 
  (h2 : ∀ (c' d' : ℕ+), (1600 : ℝ)^(1/3) = c' * d'^(1/3) → d ≤ d') : 
  c + d = 29 := by
sorry

end NUMINAMATH_CALUDE_cube_root_1600_l385_38559


namespace NUMINAMATH_CALUDE_specific_log_stack_count_l385_38555

/-- Represents a stack of logs -/
structure LogStack where
  bottomCount : ℕ
  topCount : ℕ
  decreaseRate : ℕ

/-- Calculates the total number of logs in the stack -/
def totalLogs (stack : LogStack) : ℕ :=
  let rowCount := stack.bottomCount - stack.topCount + 1
  let avgRowCount := (stack.bottomCount + stack.topCount) / 2
  rowCount * avgRowCount

/-- Theorem stating that the specific log stack has 110 logs -/
theorem specific_log_stack_count :
  ∃ (stack : LogStack),
    stack.bottomCount = 15 ∧
    stack.topCount = 5 ∧
    stack.decreaseRate = 1 ∧
    totalLogs stack = 110 := by
  sorry

end NUMINAMATH_CALUDE_specific_log_stack_count_l385_38555


namespace NUMINAMATH_CALUDE_inverse_g_150_l385_38503

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^4 + 6

-- State the theorem
theorem inverse_g_150 : 
  g ((2 : ℝ) * (3 : ℝ)^(1/4)) = 150 :=
by sorry

end NUMINAMATH_CALUDE_inverse_g_150_l385_38503


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l385_38502

def M : Set ℝ := {x | ∃ y, y = Real.log (1 - 2/x)}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_complement_theorem : N ∩ (Set.univ \ M) = Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l385_38502


namespace NUMINAMATH_CALUDE_soccer_team_goals_l385_38524

theorem soccer_team_goals (total_players : ℕ) (games_played : ℕ) (goals_other_players : ℕ) : 
  total_players = 24 →
  games_played = 15 →
  goals_other_players = 30 →
  (total_players / 3 * games_played + goals_other_players : ℕ) = 150 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_goals_l385_38524


namespace NUMINAMATH_CALUDE_wrapping_paper_fraction_l385_38507

theorem wrapping_paper_fraction (total_used : ℚ) (num_presents : ℕ) (h1 : total_used = 1/2) (h2 : num_presents = 5) :
  total_used / num_presents = 1/10 :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_fraction_l385_38507


namespace NUMINAMATH_CALUDE_min_value_on_circle_l385_38540

theorem min_value_on_circle (x y : ℝ) :
  x^2 + y^2 - 4*x - 6*y + 12 = 0 →
  ∃ (min_val : ℝ), min_val = 14 - 2 * Real.sqrt 13 ∧
    ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 6*y' + 12 = 0 →
      x'^2 + y'^2 ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l385_38540


namespace NUMINAMATH_CALUDE_quadratic_touches_x_axis_at_one_point_l385_38568

/-- A quadratic function g(x) = x^2 - 6x + k -/
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + k

/-- The discriminant of the quadratic function g -/
def discriminant (k : ℝ) : ℝ := (-6)^2 - 4*1*k

/-- Theorem: The value of k that makes g(x) touch the x-axis at exactly one point is 9 -/
theorem quadratic_touches_x_axis_at_one_point :
  ∃ (k : ℝ), (discriminant k = 0) ∧ (k = 9) := by sorry

end NUMINAMATH_CALUDE_quadratic_touches_x_axis_at_one_point_l385_38568


namespace NUMINAMATH_CALUDE_triangle_exists_iff_altitude_inequality_l385_38513

/-- A triangle with altitudes m_a, m_b, and m_c exists if and only if
    the sum of the reciprocals of any two altitudes is greater than
    the reciprocal of the third altitude. -/
theorem triangle_exists_iff_altitude_inequality 
  (m_a m_b m_c : ℝ) (h_pos_a : 0 < m_a) (h_pos_b : 0 < m_b) (h_pos_c : 0 < m_c) :
  (∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 
    m_a = b * c / (2 * (a * b + b * c + c * a).sqrt) ∧
    m_b = a * c / (2 * (a * b + b * c + c * a).sqrt) ∧
    m_c = a * b / (2 * (a * b + b * c + c * a).sqrt)) ↔
  (1 / m_a + 1 / m_b > 1 / m_c ∧
   1 / m_b + 1 / m_c > 1 / m_a ∧
   1 / m_c + 1 / m_a > 1 / m_b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_exists_iff_altitude_inequality_l385_38513


namespace NUMINAMATH_CALUDE_man_upstream_speed_l385_38567

/-- Calculates the upstream speed of a man given his downstream speed and the stream speed -/
def upstream_speed (downstream_speed stream_speed : ℝ) : ℝ :=
  downstream_speed - 2 * stream_speed

/-- Theorem stating that given a downstream speed of 13 kmph and a stream speed of 2.5 kmph, 
    the upstream speed is 8 kmph -/
theorem man_upstream_speed :
  upstream_speed 13 2.5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_man_upstream_speed_l385_38567


namespace NUMINAMATH_CALUDE_class_representatives_count_l385_38591

/-- The number of ways to select and arrange class representatives -/
def class_representatives (num_male num_female : ℕ) (male_to_select female_to_select : ℕ) : ℕ :=
  Nat.choose num_male male_to_select *
  Nat.choose num_female female_to_select *
  Nat.factorial (male_to_select + female_to_select)

/-- Theorem stating that the number of ways to select and arrange class representatives
    from 3 male and 3 female students, selecting 1 male and 2 females, is 54 -/
theorem class_representatives_count :
  class_representatives 3 3 1 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_class_representatives_count_l385_38591


namespace NUMINAMATH_CALUDE_rayden_has_more_birds_l385_38537

/-- The number of ducks Lily bought -/
def lily_ducks : ℕ := 20

/-- The number of geese Lily bought -/
def lily_geese : ℕ := 10

/-- The number of ducks Rayden bought -/
def rayden_ducks : ℕ := 3 * lily_ducks

/-- The number of geese Rayden bought -/
def rayden_geese : ℕ := 4 * lily_geese

/-- The difference in the total number of birds between Rayden and Lily -/
def bird_difference : ℕ := (rayden_ducks - lily_ducks) + (rayden_geese - lily_geese)

theorem rayden_has_more_birds :
  bird_difference = 70 := by sorry

end NUMINAMATH_CALUDE_rayden_has_more_birds_l385_38537


namespace NUMINAMATH_CALUDE_odd_numbers_mean_contradiction_l385_38579

theorem odd_numbers_mean_contradiction (a b c d e f g : ℤ) :
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧  -- Ordered
  Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f ∧ Odd g ∧  -- All odd
  (a + b + c + d + e + f + g) / 7 - d = 3 / 7  -- Mean minus middle equals 3/7
  → False := by sorry

end NUMINAMATH_CALUDE_odd_numbers_mean_contradiction_l385_38579


namespace NUMINAMATH_CALUDE_cab_delay_l385_38588

theorem cab_delay (usual_time : ℝ) (speed_ratio : ℝ) (h1 : usual_time = 25) (h2 : speed_ratio = 5/6) :
  (usual_time / speed_ratio) - usual_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_cab_delay_l385_38588


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_increase_l385_38598

theorem rectangular_prism_surface_area_increase 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  let original_surface_area := 2 * (a * b + b * c + a * c)
  let new_surface_area := 2 * ((1.8 * a) * (1.8 * b) + (1.8 * b) * (1.8 * c) + (1.8 * c) * (1.8 * a))
  (new_surface_area - original_surface_area) / original_surface_area = 2.24 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_increase_l385_38598


namespace NUMINAMATH_CALUDE_square_sum_power_of_two_l385_38589

theorem square_sum_power_of_two (n : ℕ) : 
  (∃ m : ℕ, 2^6 + 2^9 + 2^n = m^2) → n = 10 := by
sorry

end NUMINAMATH_CALUDE_square_sum_power_of_two_l385_38589


namespace NUMINAMATH_CALUDE_binomial_equation_sum_l385_38514

theorem binomial_equation_sum (A B C D : ℚ) : 
  (∀ n : ℕ, n ≥ 4 → n^4 = A * Nat.choose n 4 + B * Nat.choose n 3 + C * Nat.choose n 2 + D * Nat.choose n 1) →
  A + B + C + D = 75 := by
sorry

end NUMINAMATH_CALUDE_binomial_equation_sum_l385_38514


namespace NUMINAMATH_CALUDE_square_pentagon_exterior_angle_l385_38528

/-- The exterior angle formed by a square and a regular pentagon sharing a common side --/
def exteriorAngle (n : ℕ) : ℚ :=
  360 - (180 * (n - 2) / n) - 90

/-- Theorem: The exterior angle BAC formed by a square and a regular pentagon sharing a common side AD is 162° --/
theorem square_pentagon_exterior_angle :
  exteriorAngle 5 = 162 := by
  sorry

end NUMINAMATH_CALUDE_square_pentagon_exterior_angle_l385_38528


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l385_38569

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (5 * a^3 + 2014 * a + 4027 = 0) →
  (5 * b^3 + 2014 * b + 4027 = 0) →
  (5 * c^3 + 2014 * c + 4027 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 2416.2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l385_38569


namespace NUMINAMATH_CALUDE_expression_evaluation_l385_38522

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b^2 - 14*b + 45)
  (h2 : b = a^2 + 2*a + 5)
  (h3 : a = 3)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 9) / (c + 7) = 4923 / 2924 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l385_38522


namespace NUMINAMATH_CALUDE_curve_crosses_at_2_3_l385_38529

/-- A curve defined by x = t^2 - 4 and y = t^3 - 6t + 3 for all real t -/
def curve (t : ℝ) : ℝ × ℝ :=
  (t^2 - 4, t^3 - 6*t + 3)

/-- The point where the curve crosses itself -/
def crossing_point : ℝ × ℝ := (2, 3)

/-- Theorem stating that the curve crosses itself at (2, 3) -/
theorem curve_crosses_at_2_3 :
  ∃ (a b : ℝ), a ≠ b ∧ curve a = curve b ∧ curve a = crossing_point :=
sorry

end NUMINAMATH_CALUDE_curve_crosses_at_2_3_l385_38529


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l385_38578

theorem smallest_integer_with_remainders : ∃! x : ℕ+, 
  (x : ℕ) % 4 = 3 ∧ 
  (x : ℕ) % 3 = 2 ∧ 
  ∀ y : ℕ+, y < x → (y : ℕ) % 4 ≠ 3 ∨ (y : ℕ) % 3 ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l385_38578


namespace NUMINAMATH_CALUDE_total_spent_is_684_l385_38515

/-- Calculates the total amount spent by Christy and Tanya on face moisturizers and body lotions with discounts applied. -/
def total_spent (face_moisturizer_price : ℝ) (body_lotion_price : ℝ)
                (tanya_face : ℕ) (tanya_body : ℕ)
                (christy_face : ℕ) (christy_body : ℕ)
                (face_discount : ℝ) (body_discount : ℝ) : ℝ :=
  let tanya_total := (1 - face_discount) * (face_moisturizer_price * tanya_face) +
                     (1 - body_discount) * (body_lotion_price * tanya_body)
  let christy_total := (1 - face_discount) * (face_moisturizer_price * christy_face) +
                       (1 - body_discount) * (body_lotion_price * christy_body)
  tanya_total + christy_total

/-- Theorem stating that the total amount spent by Christy and Tanya is $684 under the given conditions. -/
theorem total_spent_is_684 :
  total_spent 50 60 2 4 3 5 0.1 0.15 = 684 ∧
  total_spent 50 60 2 4 3 5 0.1 0.15 = 2 * total_spent 50 60 2 4 2 4 0.1 0.15 :=
by sorry


end NUMINAMATH_CALUDE_total_spent_is_684_l385_38515


namespace NUMINAMATH_CALUDE_four_Z_one_equals_five_l385_38560

-- Define the Z operation
def Z (a b : ℝ) : ℝ := a^2 - 3*a*b + b^2

-- Theorem statement
theorem four_Z_one_equals_five : Z 4 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_four_Z_one_equals_five_l385_38560


namespace NUMINAMATH_CALUDE_inequality_preservation_l385_38511

theorem inequality_preservation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) : 
  a - c < b - c := by
sorry

end NUMINAMATH_CALUDE_inequality_preservation_l385_38511


namespace NUMINAMATH_CALUDE_max_value_theorem_l385_38557

theorem max_value_theorem (a b : ℝ) 
  (h : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a * x + b| ≤ 1) :
  ∃ M : ℝ, M = 80 ∧ 
    (∀ a' b' : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a' * x + b'| ≤ 1) → 
      |20 * a' + 14 * b'| + |20 * a' - 14 * b'| ≤ M) ∧
    (∃ a' b' : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |a' * x + b'| ≤ 1) ∧ 
      |20 * a' + 14 * b'| + |20 * a' - 14 * b'| = M) :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l385_38557


namespace NUMINAMATH_CALUDE_final_price_is_correct_l385_38596

def electronic_discount_rate : ℚ := 0.20
def clothing_discount_rate : ℚ := 0.15
def voucher_threshold : ℚ := 200
def voucher_value : ℚ := 20
def electronic_item_price : ℚ := 150
def clothing_item_price : ℚ := 80
def clothing_item_count : ℕ := 2

def calculate_final_price : ℚ := by
  -- Define the calculation here
  sorry

theorem final_price_is_correct :
  calculate_final_price = 236 := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_correct_l385_38596


namespace NUMINAMATH_CALUDE_cryptarithmetic_problem_l385_38592

theorem cryptarithmetic_problem (A B C D : ℕ) : 
  (A + B + C = 11) →
  (B + A + D = 10) →
  (A + D = 4) →
  (A ≠ B) → (A ≠ C) → (A ≠ D) → (B ≠ C) → (B ≠ D) → (C ≠ D) →
  (A < 10) → (B < 10) → (C < 10) → (D < 10) →
  C = 4 := by
sorry

end NUMINAMATH_CALUDE_cryptarithmetic_problem_l385_38592


namespace NUMINAMATH_CALUDE_extremum_implies_zero_derivative_zero_derivative_not_implies_extremum_l385_38544

/-- A function f : ℝ → ℝ attains an extremum at x₀ -/
def AttainsExtremum (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  (∀ x, f x ≤ f x₀) ∨ (∀ x, f x ≥ f x₀)

/-- The derivative of f at x₀ is 0 -/
def DerivativeZero (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  deriv f x₀ = 0

theorem extremum_implies_zero_derivative
  (f : ℝ → ℝ) (x₀ : ℝ) (h : Differentiable ℝ f) :
  AttainsExtremum f x₀ → DerivativeZero f x₀ :=
sorry

theorem zero_derivative_not_implies_extremum :
  ∃ (f : ℝ → ℝ) (x₀ : ℝ), Differentiable ℝ f ∧ DerivativeZero f x₀ ∧ ¬AttainsExtremum f x₀ :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_zero_derivative_zero_derivative_not_implies_extremum_l385_38544


namespace NUMINAMATH_CALUDE_jellybean_bags_l385_38521

theorem jellybean_bags (initial_average : ℕ) (new_bag_jellybeans : ℕ) (new_average : ℕ) :
  initial_average = 117 →
  new_bag_jellybeans = 362 →
  new_average = 124 →
  ∃ n : ℕ, n * initial_average + new_bag_jellybeans = (n + 1) * new_average ∧ n = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_jellybean_bags_l385_38521


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l385_38584

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Swaps the last two digits of a three-digit number -/
def ThreeDigitNumber.swap_last_two (n : ThreeDigitNumber) : ThreeDigitNumber :=
  { hundreds := n.hundreds
  , tens := n.ones
  , ones := n.tens
  , is_valid := by sorry }

theorem last_digit_of_sum (n : ThreeDigitNumber) :
  (n.value + (n.swap_last_two).value ≥ 1000) →
  (n.value + (n.swap_last_two).value < 2000) →
  (n.value + (n.swap_last_two).value) / 10 = 195 →
  (n.value + (n.swap_last_two).value) % 10 = 4 := by
  sorry


end NUMINAMATH_CALUDE_last_digit_of_sum_l385_38584


namespace NUMINAMATH_CALUDE_jose_investment_is_45000_l385_38577

/-- Represents the investment problem with Tom and Jose --/
structure InvestmentProblem where
  tom_investment : ℕ
  jose_join_delay : ℕ
  total_profit : ℕ
  jose_profit_share : ℕ

/-- Calculates Jose's investment amount based on the given parameters --/
def calculate_jose_investment (problem : InvestmentProblem) : ℕ :=
  let tom_investment_months : ℕ := problem.tom_investment * 12
  let jose_investment_months : ℕ := (12 - problem.jose_join_delay) * (problem.jose_profit_share * tom_investment_months) / (problem.total_profit - problem.jose_profit_share)
  jose_investment_months / (12 - problem.jose_join_delay)

/-- Theorem stating that Jose's investment is 45000 given the problem conditions --/
theorem jose_investment_is_45000 (problem : InvestmentProblem) 
  (h1 : problem.tom_investment = 30000)
  (h2 : problem.jose_join_delay = 2)
  (h3 : problem.total_profit = 54000)
  (h4 : problem.jose_profit_share = 30000) :
  calculate_jose_investment problem = 45000 := by
  sorry

end NUMINAMATH_CALUDE_jose_investment_is_45000_l385_38577


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l385_38518

theorem system_of_equations_sum (x y z : ℝ) 
  (eq1 : y + z = 20 - 4*x)
  (eq2 : x + z = -18 - 4*y)
  (eq3 : x + y = 10 - 4*z) :
  2*x + 2*y + 2*z = 4 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l385_38518


namespace NUMINAMATH_CALUDE_parabola_transformation_l385_38509

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

-- Define the transformation
def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + 3) - 2

-- Define the resulting parabola
def result_parabola (x : ℝ) : ℝ := 2 * x^2

-- Theorem statement
theorem parabola_transformation :
  ∀ x : ℝ, transform original_parabola x = result_parabola x :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l385_38509


namespace NUMINAMATH_CALUDE_office_staff_composition_l385_38590

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := sorry

/-- Represents the number of officers in an office -/
def num_officers : ℕ := 15

/-- Represents the average salary of all employees in rupees -/
def avg_salary_all : ℚ := 120

/-- Represents the average salary of officers in rupees -/
def avg_salary_officers : ℚ := 440

/-- Represents the average salary of non-officers in rupees -/
def avg_salary_non_officers : ℚ := 110

theorem office_staff_composition :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / (num_officers + num_non_officers) = avg_salary_all ∧
  num_non_officers = 480 := by sorry

end NUMINAMATH_CALUDE_office_staff_composition_l385_38590


namespace NUMINAMATH_CALUDE_population_sum_theorem_l385_38538

/-- The population of Springfield -/
def springfield_population : ℕ := 482653

/-- The difference in population between Springfield and Greenville -/
def population_difference : ℕ := 119666

/-- The total population of Springfield and Greenville -/
def total_population : ℕ := 845640

/-- Theorem stating that the sum of Springfield's population and a city with 119,666 fewer people equals the total population -/
theorem population_sum_theorem : 
  springfield_population + (springfield_population - population_difference) = total_population := by
  sorry

end NUMINAMATH_CALUDE_population_sum_theorem_l385_38538


namespace NUMINAMATH_CALUDE_vector_properties_l385_38549

def e₁ : ℝ × ℝ := (1, 0)
def e₂ : ℝ × ℝ := (0, 1)
def a : ℝ × ℝ := (3 * e₁.1 - 2 * e₂.1, 3 * e₁.2 - 2 * e₂.2)
def b : ℝ × ℝ := (4 * e₁.1 + e₂.1, 4 * e₁.2 + e₂.2)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 10) ∧
  (Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 * Real.sqrt 2) ∧
  (((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 10 * Real.sqrt 221 / 221) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l385_38549


namespace NUMINAMATH_CALUDE_practice_multiple_days_l385_38558

/-- Given a person who practices a constant amount each day, and 20 days ago had half as much
    practice as they have currently, prove that it takes 40(M - 1) days to reach M times
    their current practice. -/
theorem practice_multiple_days (d : ℝ) (P : ℝ) (M : ℝ) :
  (P / 2 + 20 * d = P) →  -- 20 days ago, had half as much practice
  (P = 40 * d) →          -- Current practice
  (∃ D : ℝ, D * d = M * P - P ∧ D = 40 * (M - 1)) :=
by sorry

end NUMINAMATH_CALUDE_practice_multiple_days_l385_38558


namespace NUMINAMATH_CALUDE_third_question_points_l385_38500

/-- Represents a quiz with a sequence of questions and their point values. -/
structure Quiz where
  num_questions : Nat
  first_question_points : Nat
  point_increase : Nat
  total_points : Nat

/-- Calculates the points for a specific question in the quiz. -/
def question_points (q : Quiz) (n : Nat) : Nat :=
  q.first_question_points + (n - 1) * q.point_increase

/-- Calculates the sum of points for all questions in the quiz. -/
def sum_points (q : Quiz) : Nat :=
  Finset.sum (Finset.range q.num_questions) (λ i => question_points q (i + 1))

/-- The main theorem stating that the third question is worth 39 points. -/
theorem third_question_points (q : Quiz) 
  (h1 : q.num_questions = 8)
  (h2 : q.point_increase = 4)
  (h3 : q.total_points = 360)
  (h4 : sum_points q = q.total_points) :
  question_points q 3 = 39 := by
  sorry

#eval question_points { num_questions := 8, first_question_points := 31, point_increase := 4, total_points := 360 } 3

end NUMINAMATH_CALUDE_third_question_points_l385_38500


namespace NUMINAMATH_CALUDE_probability_not_exceeding_60W_l385_38520

def total_bulbs : ℕ := 250
def bulbs_100W : ℕ := 100
def bulbs_60W : ℕ := 50
def bulbs_25W : ℕ := 50
def bulbs_15W : ℕ := 50

theorem probability_not_exceeding_60W :
  let p := (bulbs_60W + bulbs_25W + bulbs_15W) / total_bulbs
  p = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_not_exceeding_60W_l385_38520


namespace NUMINAMATH_CALUDE_tangent_line_problem_l385_38542

theorem tangent_line_problem (a : ℝ) :
  (∃ (m : ℝ), 
    (∀ x y : ℝ, y = x^3 → (y - 0 = m * (x - 1) → x = 1 ∨ (y - x^3) = 3 * x^2 * (x - x))) ∧
    (∀ x y : ℝ, y = a * x^2 + (15/4) * x - 9 → (y - 0 = m * (x - 1) → x = 1 ∨ (y - (a * x^2 + (15/4) * x - 9)) = (2 * a * x + 15/4) * (x - x))))
  → a = -1 ∨ a = -25/64 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_problem_l385_38542


namespace NUMINAMATH_CALUDE_raffle_ticket_cost_l385_38556

theorem raffle_ticket_cost (total_amount : ℕ) (num_tickets : ℕ) (cost_per_ticket : ℚ) : 
  total_amount = 620 → num_tickets = 155 → cost_per_ticket = 4 → 
  (total_amount : ℚ) / num_tickets = cost_per_ticket :=
by sorry

end NUMINAMATH_CALUDE_raffle_ticket_cost_l385_38556


namespace NUMINAMATH_CALUDE_rent_is_5000_l385_38501

/-- Calculates the monthly rent for John's computer business -/
def calculate_rent (component_cost : ℝ) (markup : ℝ) (computers_sold : ℕ) 
                   (extra_expenses : ℝ) (profit : ℝ) : ℝ :=
  let selling_price := component_cost * markup
  let total_revenue := selling_price * computers_sold
  let total_component_cost := component_cost * computers_sold
  total_revenue - total_component_cost - extra_expenses - profit

/-- Proves that the monthly rent is $5000 given the specified conditions -/
theorem rent_is_5000 : 
  calculate_rent 800 1.4 60 3000 11200 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_rent_is_5000_l385_38501


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l385_38597

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x : ℝ, (abs x > a → x^2 - x - 2 > 0) ∧ 
  (∃ y : ℝ, y^2 - y - 2 > 0 ∧ abs y ≤ a)) ↔ 
  a ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l385_38597


namespace NUMINAMATH_CALUDE_largest_common_term_l385_38545

def sequence1 (n : ℕ) : ℤ := 2 + 4 * (n - 1)
def sequence2 (n : ℕ) : ℤ := 5 + 6 * (n - 1)

def is_common_term (x : ℤ) : Prop :=
  ∃ (n m : ℕ), sequence1 n = x ∧ sequence2 m = x

def is_in_range (x : ℤ) : Prop := 1 ≤ x ∧ x ≤ 200

theorem largest_common_term :
  ∃ (x : ℤ), is_common_term x ∧ is_in_range x ∧
  ∀ (y : ℤ), is_common_term y ∧ is_in_range y → y ≤ x ∧
  x = 190 :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l385_38545


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l385_38526

theorem simplify_trig_expression (x : ℝ) :
  Real.sqrt 2 * Real.cos x - Real.sqrt 6 * Real.sin x = 2 * Real.sqrt 2 * Real.cos (π / 3 + x) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l385_38526


namespace NUMINAMATH_CALUDE_tangent_sum_l385_38561

-- Define the tangent and cotangent functions
noncomputable def tg (x : ℝ) : ℝ := Real.tan x
noncomputable def ctg (x : ℝ) : ℝ := 1 / Real.tan x

-- State the theorem
theorem tangent_sum (A B : ℝ) 
  (h1 : tg A + tg B = 2) 
  (h2 : ctg A + ctg B = 3) : 
  tg (A + B) = 6 := by sorry

end NUMINAMATH_CALUDE_tangent_sum_l385_38561


namespace NUMINAMATH_CALUDE_e_pi_plus_pi_e_approx_l385_38595

/-- Approximate value of e -/
def e_approx : ℝ := 2.718

/-- Approximate value of π -/
def π_approx : ℝ := 3.14159

/-- Theorem stating that e^π + π^e is approximately equal to 45.5999 -/
theorem e_pi_plus_pi_e_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |Real.exp π_approx + Real.exp e_approx - 45.5999| < ε :=
sorry

end NUMINAMATH_CALUDE_e_pi_plus_pi_e_approx_l385_38595


namespace NUMINAMATH_CALUDE_minerals_found_today_l385_38593

def minerals_yesterday (gemstones_yesterday : ℕ) : ℕ := 2 * gemstones_yesterday

theorem minerals_found_today 
  (gemstones_today minerals_today : ℕ) 
  (h1 : minerals_today = 48) 
  (h2 : gemstones_today = 21) : 
  minerals_today - minerals_yesterday gemstones_today = 6 := by
  sorry

end NUMINAMATH_CALUDE_minerals_found_today_l385_38593


namespace NUMINAMATH_CALUDE_card_game_combinations_l385_38519

/-- The number of cards in the deck -/
def deck_size : ℕ := 60

/-- The number of cards in a hand -/
def hand_size : ℕ := 12

/-- The number of distinct unordered hands -/
def num_hands : ℕ := 75287520

theorem card_game_combinations :
  Nat.choose deck_size hand_size = num_hands := by
  sorry

end NUMINAMATH_CALUDE_card_game_combinations_l385_38519


namespace NUMINAMATH_CALUDE_collinear_points_imply_a_values_l385_38504

-- Define the points A, B, and C in the plane
def A (a : ℝ) : ℝ × ℝ := (1, -a)
def B (a : ℝ) : ℝ × ℝ := (2, a^2)
def C (a : ℝ) : ℝ × ℝ := (3, a^3)

-- Define collinearity of three points
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Theorem statement
theorem collinear_points_imply_a_values (a : ℝ) :
  collinear (A a) (B a) (C a) → a = 0 ∨ a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_imply_a_values_l385_38504


namespace NUMINAMATH_CALUDE_stating_men_meet_at_calculated_point_l385_38566

/-- Two men walk towards each other from points A and B, which are 90 miles apart. -/
def total_distance : ℝ := 90

/-- The speed of the man starting from point A in miles per hour. -/
def speed_a : ℝ := 5

/-- The initial speed of the man starting from point B in miles per hour. -/
def initial_speed_b : ℝ := 2

/-- The hourly increase in speed for the man starting from point B. -/
def speed_increase_b : ℝ := 1

/-- The number of hours the man from A waits before starting. -/
def wait_time : ℕ := 1

/-- The total time in hours until the men meet. -/
def total_time : ℕ := 10

/-- The distance from point B where the men meet. -/
def meeting_point : ℝ := 52.5

/-- 
Theorem stating that the men meet at the specified distance from B after the given time,
given their walking patterns.
-/
theorem men_meet_at_calculated_point :
  let distance_a := speed_a * (total_time - wait_time)
  let distance_b := (total_time / 2 : ℝ) * (initial_speed_b + initial_speed_b + speed_increase_b * (total_time - 1))
  distance_a + distance_b = total_distance ∧ distance_b = meeting_point := by sorry

end NUMINAMATH_CALUDE_stating_men_meet_at_calculated_point_l385_38566


namespace NUMINAMATH_CALUDE_balloon_arrangement_count_l385_38547

def balloon_permutations : ℕ := 1260

theorem balloon_arrangement_count :
  let total_letters : ℕ := 7
  let repeated_l : ℕ := 2
  let repeated_o : ℕ := 2
  balloon_permutations = Nat.factorial total_letters / (Nat.factorial repeated_l * Nat.factorial repeated_o) := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangement_count_l385_38547


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l385_38562

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem complement_intersection_equals_set : 
  (M ∩ N)ᶜ = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l385_38562


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l385_38570

theorem least_number_with_remainder (n : ℕ) : n ≥ 261 ∧ n % 37 = 2 ∧ n % 7 = 2 → n = 261 :=
sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l385_38570


namespace NUMINAMATH_CALUDE_proportion_solution_l385_38530

-- Define the conversion factor from minutes to seconds
def minutes_to_seconds (minutes : ℚ) : ℚ := 60 * minutes

-- Define the proportion
def proportion (x : ℚ) : Prop :=
  x / 4 = 8 / (minutes_to_seconds 4)

-- Theorem statement
theorem proportion_solution :
  ∃ (x : ℚ), proportion x ∧ x = 1 / 7.5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l385_38530


namespace NUMINAMATH_CALUDE_evaluate_expression_l385_38512

theorem evaluate_expression : 6 - 5 * (7 - (Real.sqrt 16 + 2)^2) * 3 = -429 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l385_38512


namespace NUMINAMATH_CALUDE_largest_angle_is_120_degrees_l385_38571

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  2 * t.a + 3 * t.b + t.c = t.b^2 ∧ 2 * t.a + 3 * t.b - t.c = 1

-- Theorem statement
theorem largest_angle_is_120_degrees (t : Triangle) 
  (h : satisfies_conditions t) : 
  ∃ (angle : ℝ), angle = Real.arccos (-1/2) ∧ 
  angle = max (Real.arccos ((t.b^2 + t.c^2 - t.a^2) / (2*t.b*t.c))) 
              (max (Real.arccos ((t.a^2 + t.c^2 - t.b^2) / (2*t.a*t.c)))
                   (Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2*t.a*t.b)))) :=
sorry

end NUMINAMATH_CALUDE_largest_angle_is_120_degrees_l385_38571


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_2_sqrt_6_l385_38525

/-- Represents a hexagon with side length s -/
structure Hexagon where
  s : ℝ
  h_positive : s > 0

/-- Represents a rectangle with sides x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ
  h_positive_x : x > 0
  h_positive_y : y > 0

/-- The arrangement of rectangles around hexagons -/
structure HexagonArrangement where
  inner : Hexagon
  outer : Hexagon
  rectangle : Rectangle
  h_area_ratio : outer.s^2 = 6 * inner.s^2
  h_outer_perimeter : 6 * rectangle.x = 6 * outer.s
  h_inner_side : rectangle.y = inner.s / 2

theorem rectangle_ratio_is_2_sqrt_6 (arr : HexagonArrangement) :
  arr.rectangle.x / arr.rectangle.y = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_2_sqrt_6_l385_38525


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l385_38531

theorem geometric_sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 3 = 5/2 →
  S 3 = 15/2 →
  (∀ n, a n = 5/2) ∨ (∀ n, a n = 10 * (-1/2)^(n-1)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l385_38531


namespace NUMINAMATH_CALUDE_unbiased_scale_impossible_biased_scale_possible_l385_38550

/-- Represents the result of a weighing -/
inductive WeighResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing strategy -/
def WeighStrategy := List WeighResult → WeighResult

/-- Represents a set of weights -/
def Weights := List Nat

/-- Represents a balance scale -/
structure Balance where
  bias : Int  -- Positive means left pan is lighter

/-- Function to perform a weighing -/
def weigh (b : Balance) (left right : Weights) : WeighResult :=
  sorry

/-- Function to determine if a set of weights can be uniquely identified -/
def canIdentifyWeights (w : Weights) (b : Balance) (n : Nat) : Prop :=
  sorry

/-- The main theorem for the unbiased scale -/
theorem unbiased_scale_impossible 
  (w : Weights) 
  (h1 : w = [1000, 1002, 1004, 1005]) 
  (b : Balance) 
  (h2 : b.bias = 0) : 
  ¬ (canIdentifyWeights w b 4) :=
sorry

/-- The main theorem for the biased scale -/
theorem biased_scale_possible 
  (w : Weights) 
  (h1 : w = [1000, 1002, 1004, 1005]) 
  (b : Balance) 
  (h2 : b.bias = 1) : 
  canIdentifyWeights w b 4 :=
sorry

end NUMINAMATH_CALUDE_unbiased_scale_impossible_biased_scale_possible_l385_38550


namespace NUMINAMATH_CALUDE_gina_netflix_minutes_l385_38543

/-- Represents the number of times Gina chooses what to watch compared to her sister -/
def gina_choice_ratio : ℕ := 3

/-- Represents the number of times Gina's sister chooses what to watch -/
def sister_choice_ratio : ℕ := 1

/-- The number of shows Gina's sister watches per week -/
def sister_shows_per_week : ℕ := 24

/-- The length of each show in minutes -/
def show_length : ℕ := 50

/-- Theorem stating that Gina chooses 3600 minutes of Netflix per week -/
theorem gina_netflix_minutes :
  (sister_shows_per_week * gina_choice_ratio * show_length) / (gina_choice_ratio + sister_choice_ratio) = 3600 :=
sorry

end NUMINAMATH_CALUDE_gina_netflix_minutes_l385_38543


namespace NUMINAMATH_CALUDE_solve_equation_l385_38523

theorem solve_equation : 
  ∃ x : ℚ, (27 / 4) * x - 18 = 3 * x + 27 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l385_38523


namespace NUMINAMATH_CALUDE_dvd_book_capacity_l385_38546

/-- Represents the capacity of a DVD book -/
structure DVDBook where
  current : ℕ  -- Number of DVDs currently in the book
  remaining : ℕ  -- Number of additional DVDs that can be added

/-- Calculates the total capacity of a DVD book -/
def totalCapacity (book : DVDBook) : ℕ :=
  book.current + book.remaining

/-- Theorem: The total capacity of the given DVD book is 126 -/
theorem dvd_book_capacity : 
  ∀ (book : DVDBook), book.current = 81 → book.remaining = 45 → totalCapacity book = 126 :=
by
  sorry


end NUMINAMATH_CALUDE_dvd_book_capacity_l385_38546


namespace NUMINAMATH_CALUDE_six_eight_ten_pythagorean_l385_38575

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- Theorem: (6, 8, 10) is a Pythagorean triple -/
theorem six_eight_ten_pythagorean : is_pythagorean_triple 6 8 10 := by
  sorry

end NUMINAMATH_CALUDE_six_eight_ten_pythagorean_l385_38575


namespace NUMINAMATH_CALUDE_point_coordinates_given_distance_to_x_axis_l385_38534

def distance_to_x_axis (y : ℝ) : ℝ := |y|

theorem point_coordinates_given_distance_to_x_axis (m : ℝ) :
  distance_to_x_axis m = 4 → m = 4 ∨ m = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_given_distance_to_x_axis_l385_38534


namespace NUMINAMATH_CALUDE_circular_arrangement_equality_l385_38594

/-- Given a circular arrangement of n people numbered 1 to n,
    if the distance from person 31 to person 7 is equal to
    the distance from person 31 to person 14, then n = 41. -/
theorem circular_arrangement_equality (n : ℕ) : n > 30 →
  (n - 31 + 7) % n = (14 - 31 + n) % n →
  n = 41 := by
  sorry


end NUMINAMATH_CALUDE_circular_arrangement_equality_l385_38594


namespace NUMINAMATH_CALUDE_nail_salon_fingers_l385_38532

theorem nail_salon_fingers (total_earnings : ℚ) (cost_per_manicure : ℚ) (total_fingers : ℕ) (non_clients : ℕ) :
  total_earnings = 200 →
  cost_per_manicure = 20 →
  total_fingers = 210 →
  non_clients = 11 →
  ∃ (fingers_per_person : ℕ), 
    fingers_per_person = 10 ∧
    (total_earnings / cost_per_manicure + non_clients : ℚ) * fingers_per_person = total_fingers := by
  sorry

end NUMINAMATH_CALUDE_nail_salon_fingers_l385_38532
