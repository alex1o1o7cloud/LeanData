import Mathlib

namespace NUMINAMATH_CALUDE_power_multiplication_problem_solution_l3587_358705

theorem power_multiplication (n : ℕ) : n * (n ^ n) = n ^ (n + 1) := by sorry

theorem problem_solution : 3000 * (3000 ^ 3000) = 3000 ^ 3001 := by
  apply power_multiplication

end NUMINAMATH_CALUDE_power_multiplication_problem_solution_l3587_358705


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3587_358738

theorem tangent_line_to_circle (x y : ℝ) :
  (x^2 + y^2 = 4) →  -- Circle equation
  (1^2 + (Real.sqrt 3)^2 = 4) →  -- Point (1, √3) is on the circle
  (x + Real.sqrt 3 * y = 4) →  -- Proposed tangent line equation
  ∃ (k : ℝ), k * (x - 1) + Real.sqrt 3 * k * (y - Real.sqrt 3) = 0  -- Tangent line property
  :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3587_358738


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3587_358798

/-- Given a rectangle A composed of 3 equal squares with a perimeter of 112 cm,
    prove that a rectangle B composed of 4 of the same squares will have a perimeter of 140 cm. -/
theorem rectangle_perimeter (side_length : ℝ) : 
  (3 * side_length * 2 + 2 * side_length) = 112 → 
  (4 * side_length * 2 + 2 * side_length) = 140 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3587_358798


namespace NUMINAMATH_CALUDE_corn_plants_per_row_l3587_358739

/-- Calculates the number of corn plants in each row given the water pumping conditions. -/
theorem corn_plants_per_row (
  pump_rate : ℝ)
  (pump_time : ℝ)
  (num_rows : ℕ)
  (water_per_plant : ℝ)
  (num_pigs : ℕ)
  (water_per_pig : ℝ)
  (num_ducks : ℕ)
  (water_per_duck : ℝ)
  (h_pump_rate : pump_rate = 3)
  (h_pump_time : pump_time = 25)
  (h_num_rows : num_rows = 4)
  (h_water_per_plant : water_per_plant = 0.5)
  (h_num_pigs : num_pigs = 10)
  (h_water_per_pig : water_per_pig = 4)
  (h_num_ducks : num_ducks = 20)
  (h_water_per_duck : water_per_duck = 0.25) :
  (pump_rate * pump_time - (num_pigs * water_per_pig + num_ducks * water_per_duck)) / (num_rows * water_per_plant) = 15 := by
sorry

end NUMINAMATH_CALUDE_corn_plants_per_row_l3587_358739


namespace NUMINAMATH_CALUDE_triangle_area_product_l3587_358788

theorem triangle_area_product (p q : ℝ) : 
  p > 0 → q > 0 → 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ p * x + q * y = 12) →
  (1/2 * (12/p) * (12/q) = 12) →
  p * q = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_product_l3587_358788


namespace NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l3587_358728

theorem sqrt_inequality_solution_set (x : ℝ) :
  (x + 3 ≥ 0) → (∃ y, y > 1 ∧ x = y) ↔ (Real.sqrt (x + 3) > 3 - x) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_solution_set_l3587_358728


namespace NUMINAMATH_CALUDE_f_properties_l3587_358707

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

-- State the theorem
theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f a (-x) = -(f a x)) ∧
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 1) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l3587_358707


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3587_358779

theorem quadratic_expression_value (x : ℝ) : 
  x = -2 → x^2 + 6*x - 8 = -16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3587_358779


namespace NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3587_358730

/-- A line in a 2D plane. -/
structure Line where
  slope : ℝ
  x_intercept : ℝ × ℝ

/-- The y-intercept of a line. -/
def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.slope * (-l.x_intercept.1) + l.x_intercept.2)

/-- Theorem: For a line with slope -2 and x-intercept (5,0), the y-intercept is (0,10). -/
theorem y_intercept_for_specific_line :
  let l : Line := { slope := -2, x_intercept := (5, 0) }
  y_intercept l = (0, 10) := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_for_specific_line_l3587_358730


namespace NUMINAMATH_CALUDE_logarithm_difference_equals_three_l3587_358714

theorem logarithm_difference_equals_three :
  (Real.log 320 / Real.log 4) / (Real.log 80 / Real.log 4) -
  (Real.log 640 / Real.log 4) / (Real.log 40 / Real.log 4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_difference_equals_three_l3587_358714


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l3587_358723

/-- Represents the number of valid arrangements of frogs -/
def validFrogArrangements (n g r b : ℕ) : ℕ :=
  if n = g + r + b ∧ g ≥ 1 ∧ r ≥ 1 ∧ b = 1 then
    2 * (Nat.factorial r * Nat.factorial g)
  else
    0

/-- Theorem stating the number of valid frog arrangements for the given problem -/
theorem frog_arrangement_count :
  validFrogArrangements 8 3 4 1 = 288 := by
  sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l3587_358723


namespace NUMINAMATH_CALUDE_lowest_cost_plan_l3587_358787

/-- Represents a gardening style arrangement plan -/
structure ArrangementPlan where
  style_a : ℕ
  style_b : ℕ

/-- Represents the gardening problem setup -/
structure GardeningProblem where
  total_sets : ℕ
  type_a_flowers : ℕ
  type_b_flowers : ℕ
  style_a_type_a : ℕ
  style_a_type_b : ℕ
  style_b_type_a : ℕ
  style_b_type_b : ℕ
  style_a_cost : ℕ
  style_b_cost : ℕ

/-- Checks if an arrangement plan is feasible -/
def is_feasible (problem : GardeningProblem) (plan : ArrangementPlan) : Prop :=
  plan.style_a + plan.style_b = problem.total_sets ∧
  plan.style_a * problem.style_a_type_a + plan.style_b * problem.style_b_type_a ≤ problem.type_a_flowers ∧
  plan.style_a * problem.style_a_type_b + plan.style_b * problem.style_b_type_b ≤ problem.type_b_flowers

/-- Calculates the cost of an arrangement plan -/
def cost (problem : GardeningProblem) (plan : ArrangementPlan) : ℕ :=
  plan.style_a * problem.style_a_cost + plan.style_b * problem.style_b_cost

/-- The main theorem to be proved -/
theorem lowest_cost_plan (problem : GardeningProblem) 
  (h_problem : problem = { 
    total_sets := 50,
    type_a_flowers := 2660,
    type_b_flowers := 3000,
    style_a_type_a := 70,
    style_a_type_b := 30,
    style_b_type_a := 40,
    style_b_type_b := 80,
    style_a_cost := 800,
    style_b_cost := 960
  }) :
  ∃ (optimal_plan : ArrangementPlan),
    is_feasible problem optimal_plan ∧
    cost problem optimal_plan = 44480 ∧
    ∀ (other_plan : ArrangementPlan), 
      is_feasible problem other_plan → 
      cost problem other_plan ≥ cost problem optimal_plan :=
sorry

end NUMINAMATH_CALUDE_lowest_cost_plan_l3587_358787


namespace NUMINAMATH_CALUDE_angle_pairs_same_terminal_side_l3587_358713

def same_terminal_side (a b : Int) : Prop :=
  ∃ k : Int, b - a = k * 360

theorem angle_pairs_same_terminal_side :
  ¬ same_terminal_side 390 690 ∧
  same_terminal_side (-330) 750 ∧
  ¬ same_terminal_side 480 (-420) ∧
  ¬ same_terminal_side 3000 (-840) :=
by sorry

end NUMINAMATH_CALUDE_angle_pairs_same_terminal_side_l3587_358713


namespace NUMINAMATH_CALUDE_largest_solution_and_ratio_l3587_358711

theorem largest_solution_and_ratio (x : ℝ) (a b c d : ℤ) : 
  (7 * x / 4 - 2 = 5 / x) → 
  (x = (a + b * Real.sqrt c) / d) → 
  (∀ y : ℝ, (7 * y / 4 - 2 = 5 / y) → y ≤ x) →
  (x = (4 + 2 * Real.sqrt 39) / 7 ∧ a * c * d / b = 546) := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_and_ratio_l3587_358711


namespace NUMINAMATH_CALUDE_tensor_range_theorem_l3587_358741

/-- Custom operation ⊗ -/
def tensor (a b : ℝ) : ℝ := a * b + a + b^2

/-- Theorem stating the range of k given the condition -/
theorem tensor_range_theorem (k : ℝ) :
  (∀ x : ℝ, tensor k x > 0) → k ∈ Set.Ioo 0 4 := by
  sorry

end NUMINAMATH_CALUDE_tensor_range_theorem_l3587_358741


namespace NUMINAMATH_CALUDE_square_sum_value_l3587_358789

theorem square_sum_value (x y : ℝ) (h : (x^2 + y^2)^4 - 6*(x^2 + y^2)^2 + 9 = 0) : 
  x^2 + y^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_square_sum_value_l3587_358789


namespace NUMINAMATH_CALUDE_parabola_sum_coefficients_l3587_358751

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum_coefficients (p : Parabola) :
  p.x_coord 2 = -3 →  -- vertex at (-3, 2)
  p.x_coord (-1) = 1 →  -- passes through (1, -1)
  p.a < 0 →  -- opens to the left
  p.a + p.b + p.c = -23/9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_coefficients_l3587_358751


namespace NUMINAMATH_CALUDE_georges_expenses_l3587_358795

theorem georges_expenses (B : ℝ) (B_pos : B > 0) : ∃ (m s : ℝ),
  m = 0.25 * (B - s) ∧
  s = 0.05 * (B - m) ∧
  m + s = B :=
by sorry

end NUMINAMATH_CALUDE_georges_expenses_l3587_358795


namespace NUMINAMATH_CALUDE_toys_needed_l3587_358715

theorem toys_needed (available : ℕ) (people : ℕ) (per_person : ℕ) : 
  available = 68 → people = 14 → per_person = 5 → 
  (people * per_person - available : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_toys_needed_l3587_358715


namespace NUMINAMATH_CALUDE_largest_quotient_l3587_358797

def S : Set ℤ := {-36, -6, -4, 3, 7, 9}

def quotient (a b : ℤ) : ℚ := (a : ℚ) / (b : ℚ)

def valid_quotient (q : ℚ) : Prop :=
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ b ≠ 0 ∧ q = quotient a b

theorem largest_quotient :
  ∃ (max_q : ℚ), valid_quotient max_q ∧ 
  (∀ (q : ℚ), valid_quotient q → q ≤ max_q) ∧
  max_q = 9 := by sorry

end NUMINAMATH_CALUDE_largest_quotient_l3587_358797


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_3_4_5_l3587_358749

theorem least_three_digit_multiple_of_3_4_5 : 
  (∀ n : ℕ, n ≥ 100 ∧ n < 120 → ¬(3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n)) ∧ 
  (120 ≥ 100 ∧ 3 ∣ 120 ∧ 4 ∣ 120 ∧ 5 ∣ 120) := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_3_4_5_l3587_358749


namespace NUMINAMATH_CALUDE_identity_function_unique_l3587_358759

/-- A function satisfying the given conditions -/
def satisfying_function (f : ℝ → ℝ) : Prop :=
  (f 1 = 1) ∧ 
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ 
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = f x / x^2)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem identity_function_unique (f : ℝ → ℝ) (h : satisfying_function f) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_identity_function_unique_l3587_358759


namespace NUMINAMATH_CALUDE_corrected_mean_l3587_358717

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 30 ∧ incorrect_value = 23 ∧ correct_value = 48 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) = n * (30.5 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_corrected_mean_l3587_358717


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3587_358735

/-- Represents a repeating decimal with a given numerator and denominator. -/
def repeating_decimal (numerator denominator : ℕ) : ℚ :=
  numerator / denominator

/-- The sum of the given repeating decimals is equal to 2224/9999. -/
theorem sum_of_repeating_decimals :
  repeating_decimal 2 9 + repeating_decimal 2 99 + repeating_decimal 2 9999 = 2224 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3587_358735


namespace NUMINAMATH_CALUDE_standard_normal_probability_l3587_358760

/-- Standard normal distribution function -/
noncomputable def Φ : ℝ → ℝ := sorry

/-- Probability density function of the standard normal distribution -/
noncomputable def φ : ℝ → ℝ := sorry

theorem standard_normal_probability (X : ℝ → ℝ) : 
  (∀ (a b : ℝ), a < b → ∫ x in a..b, φ x = Φ b - Φ a) →
  Φ 2 - Φ (-1) = 0.8185 := by
  sorry

end NUMINAMATH_CALUDE_standard_normal_probability_l3587_358760


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3587_358706

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes 
  (l m : Line) (α β : Plane) 
  (h1 : parallel_plane α β) 
  (h2 : perpendicular_line_plane l α) 
  (h3 : parallel_line_plane m β) : 
  perpendicular_line l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3587_358706


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l3587_358799

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 64) : a / b = 4 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l3587_358799


namespace NUMINAMATH_CALUDE_time_is_point_eight_hours_l3587_358782

/-- The number of unique letters in the name --/
def name_length : ℕ := 6

/-- The number of rearrangements that can be written per minute --/
def rearrangements_per_minute : ℕ := 15

/-- Calculates the time in hours required to write all possible rearrangements of a name --/
def time_to_write_all_rearrangements : ℚ :=
  (Nat.factorial name_length : ℚ) / (rearrangements_per_minute : ℚ) / 60

/-- Theorem stating that the time to write all rearrangements is 0.8 hours --/
theorem time_is_point_eight_hours :
  time_to_write_all_rearrangements = 4/5 := by sorry

end NUMINAMATH_CALUDE_time_is_point_eight_hours_l3587_358782


namespace NUMINAMATH_CALUDE_arcsin_equation_solution_l3587_358772

theorem arcsin_equation_solution : 
  Real.arcsin (Real.sqrt (2/51)) + Real.arcsin (3 * Real.sqrt (2/51)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_equation_solution_l3587_358772


namespace NUMINAMATH_CALUDE_edwards_remaining_money_l3587_358725

/-- Calculates the remaining money after a purchase -/
def remainingMoney (initialAmount spentAmount : ℕ) : ℕ :=
  initialAmount - spentAmount

/-- Theorem: Edward's remaining money is $6 -/
theorem edwards_remaining_money :
  remainingMoney 22 16 = 6 := by
  sorry

end NUMINAMATH_CALUDE_edwards_remaining_money_l3587_358725


namespace NUMINAMATH_CALUDE_problem_1_l3587_358732

theorem problem_1 (a b : ℝ) (h : a ≠ 0) (h' : b ≠ 0) :
  (-3*b/(2*a)) * (6*a/b^3) = -9/b^2 :=
sorry

end NUMINAMATH_CALUDE_problem_1_l3587_358732


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3587_358775

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3587_358775


namespace NUMINAMATH_CALUDE_distance_between_sasha_and_kolya_l3587_358745

-- Define the race distance
def race_distance : ℝ := 100

-- Define the runners' speeds
variable (v_S v_L v_K : ℝ)

-- Define the conditions
axiom positive_speeds : 0 < v_S ∧ 0 < v_L ∧ 0 < v_K
axiom lyosha_behind_sasha : v_L / v_S = 0.9
axiom kolya_behind_lyosha : v_K / v_L = 0.9

-- Define the theorem
theorem distance_between_sasha_and_kolya :
  let t_S := race_distance / v_S
  let d_K := v_K * t_S
  race_distance - d_K = 19 := by sorry

end NUMINAMATH_CALUDE_distance_between_sasha_and_kolya_l3587_358745


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3587_358727

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3587_358727


namespace NUMINAMATH_CALUDE_du_chin_pies_l3587_358767

/-- The number of meat pies Du Chin bakes in a day -/
def num_pies : ℕ := 200

/-- The price of each meat pie in dollars -/
def price_per_pie : ℕ := 20

/-- The fraction of sales used to buy ingredients for the next day -/
def ingredient_fraction : ℚ := 3/5

/-- The amount remaining after setting aside money for ingredients -/
def remaining_amount : ℕ := 1600

/-- Theorem stating that the number of pies baked satisfies the given conditions -/
theorem du_chin_pies :
  (num_pies * price_per_pie : ℚ) * (1 - ingredient_fraction) = remaining_amount := by
  sorry

end NUMINAMATH_CALUDE_du_chin_pies_l3587_358767


namespace NUMINAMATH_CALUDE_first_number_value_l3587_358774

theorem first_number_value : ∃ x y : ℤ, 
  (x + 2 * y = 124) ∧ (y = 43) → x = 38 := by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l3587_358774


namespace NUMINAMATH_CALUDE_die_roll_probability_l3587_358763

/-- The probability of getting a different outcome on a six-sided die -/
def prob_different : ℚ := 5 / 6

/-- The probability of getting the same outcome on a six-sided die -/
def prob_same : ℚ := 1 / 6

/-- The number of rolls before the consecutive identical rolls -/
def num_rolls : ℕ := 10

theorem die_roll_probability : 
  prob_different ^ num_rolls * prob_same = 5^10 / 6^11 := by
  sorry

end NUMINAMATH_CALUDE_die_roll_probability_l3587_358763


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3587_358757

/-- A square with side length 8 -/
structure Square :=
  (side : ℝ)
  (is_eight : side = 8)

/-- A circle passing through two opposite vertices of a square and tangent to the opposite side -/
structure TangentCircle (s : Square) :=
  (radius : ℝ)
  (passes_through_vertices : True)  -- This is a simplification, as we can't directly represent geometric relations
  (tangent_to_side : True)  -- This is a simplification, as we can't directly represent geometric relations

/-- The radius of the tangent circle is 5 -/
theorem tangent_circle_radius (s : Square) (c : TangentCircle s) : c.radius = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3587_358757


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3587_358746

theorem smallest_integer_solution (x : ℤ) : 
  (x^4 - 40*x^2 + 324 = 0) → x ≥ -4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3587_358746


namespace NUMINAMATH_CALUDE_largest_integer_square_four_digits_base7_l3587_358748

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := 66

/-- Convert a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Count the number of digits in a number's base 7 representation -/
def digitCountBase7 (n : ℕ) : ℕ :=
  (toBase7 n).length

theorem largest_integer_square_four_digits_base7 :
  (M * M ≥ 7^3) ∧
  (M * M < 7^4) ∧
  (digitCountBase7 (M * M) = 4) ∧
  (∀ n : ℕ, n > M → digitCountBase7 (n * n) ≠ 4) := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_square_four_digits_base7_l3587_358748


namespace NUMINAMATH_CALUDE_preimages_of_one_l3587_358764

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem preimages_of_one (x : ℝ) : 
  f x = 1 ↔ x = -1 ∨ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_preimages_of_one_l3587_358764


namespace NUMINAMATH_CALUDE_dalton_needs_sixteen_more_l3587_358720

theorem dalton_needs_sixteen_more : ∀ (jump_rope board_game ball puzzle saved uncle_gift : ℕ),
  jump_rope = 9 →
  board_game = 15 →
  ball = 5 →
  puzzle = 8 →
  saved = 7 →
  uncle_gift = 14 →
  jump_rope + board_game + ball + puzzle - (saved + uncle_gift) = 16 := by
  sorry

end NUMINAMATH_CALUDE_dalton_needs_sixteen_more_l3587_358720


namespace NUMINAMATH_CALUDE_final_box_weight_l3587_358792

/-- The weight of the box after each step of adding ingredients --/
def box_weight (initial : ℝ) (triple : ℝ → ℝ) (add_two : ℝ → ℝ) (double : ℝ → ℝ) : ℝ :=
  double (add_two (triple initial))

/-- The theorem stating the final weight of the box --/
theorem final_box_weight :
  box_weight 2 (fun x => 3 * x) (fun x => x + 2) (fun x => 2 * x) = 16 := by
  sorry

#check final_box_weight

end NUMINAMATH_CALUDE_final_box_weight_l3587_358792


namespace NUMINAMATH_CALUDE_complex_product_squared_l3587_358776

theorem complex_product_squared (P R S : ℂ) : 
  P = 3 + 4*I ∧ R = 2*I ∧ S = 3 - 4*I → (P * R * S)^2 = -2500 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_squared_l3587_358776


namespace NUMINAMATH_CALUDE_second_polygon_sides_l3587_358785

theorem second_polygon_sides (n₁ n₂ : ℕ) (s₁ s₂ : ℝ) :
  n₁ = 50 →
  s₁ = 3 * s₂ →
  n₁ * s₁ = n₂ * s₂ →
  n₂ = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l3587_358785


namespace NUMINAMATH_CALUDE_isabel_paper_problem_l3587_358708

theorem isabel_paper_problem (total : ℕ) (used : ℕ) (remaining : ℕ) : 
  total = 900 → used = 156 → remaining = total - used → remaining = 744 := by
  sorry

end NUMINAMATH_CALUDE_isabel_paper_problem_l3587_358708


namespace NUMINAMATH_CALUDE_cubic_foot_to_cubic_inch_l3587_358722

theorem cubic_foot_to_cubic_inch :
  (1 : ℝ) * (foot ^ 3) = 1728 * (inch ^ 3) :=
by
  -- Define the relationship between foot and inch
  have foot_to_inch : (1 : ℝ) * foot = 12 * inch := sorry
  
  -- Cube both sides of the equation
  have cubed_equality : ((1 : ℝ) * foot) ^ 3 = (12 * inch) ^ 3 := sorry
  
  -- Simplify the left side
  have left_side : ((1 : ℝ) * foot) ^ 3 = (1 : ℝ) * (foot ^ 3) := sorry
  
  -- Simplify the right side
  have right_side : (12 * inch) ^ 3 = 1728 * (inch ^ 3) := sorry
  
  -- Combine the steps to prove the theorem
  sorry

end NUMINAMATH_CALUDE_cubic_foot_to_cubic_inch_l3587_358722


namespace NUMINAMATH_CALUDE_system_solution_l3587_358754

theorem system_solution :
  ∃ (x₁ x₂ x₃ : ℝ),
    (3 * x₁ - 2 * x₂ + x₃ = -10) ∧
    (2 * x₁ + 3 * x₂ - 4 * x₃ = 16) ∧
    (x₁ - 4 * x₂ + 3 * x₃ = -18) ∧
    (x₁ = -1) ∧ (x₂ = 2) ∧ (x₃ = -3) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3587_358754


namespace NUMINAMATH_CALUDE_intersection_of_three_lines_l3587_358701

/-- Given three lines that intersect at the same point, prove the value of k -/
theorem intersection_of_three_lines (x y : ℝ) :
  (y = 4 * x + 3) ∧ 
  (y = -2 * x - 25) ∧ 
  (y = 3 * x + k) →
  k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_three_lines_l3587_358701


namespace NUMINAMATH_CALUDE_ab_length_is_twelve_l3587_358790

-- Define the triangles and their properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Define the theorem
theorem ab_length_is_twelve
  (ABC : Triangle) (CBD : Triangle)
  (h1 : isIsosceles ABC)
  (h2 : isIsosceles CBD)
  (h3 : perimeter CBD = 24)
  (h4 : perimeter ABC = 26)
  (h5 : CBD.c = 10) :
  ABC.c = 12 := by
  sorry

end NUMINAMATH_CALUDE_ab_length_is_twelve_l3587_358790


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3587_358740

/-- Proves that a hyperbola with given conditions has the equation x²/3 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b / a = Real.sqrt 3 / 3) →
  (Real.sqrt 3 * a / 3 = 1) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 3 - y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3587_358740


namespace NUMINAMATH_CALUDE_repeating_digit_divisible_by_101_l3587_358780

/-- A 9-digit integer where the first three digits are the same as the middle three and last three digits -/
def RepeatingDigitInteger (x y z : ℕ) : ℕ :=
  100100100 * x + 10010010 * y + 1001001 * z

/-- Theorem stating that 101 is a factor of any RepeatingDigitInteger -/
theorem repeating_digit_divisible_by_101 (x y z : ℕ) (h : 0 < x ∧ x < 10 ∧ y < 10 ∧ z < 10) :
  101 ∣ RepeatingDigitInteger x y z := by
  sorry

#check repeating_digit_divisible_by_101

end NUMINAMATH_CALUDE_repeating_digit_divisible_by_101_l3587_358780


namespace NUMINAMATH_CALUDE_valentines_remaining_l3587_358731

/-- Given that Mrs. Wong initially had 30 Valentines and gave 8 away,
    prove that she has 22 Valentines left. -/
theorem valentines_remaining (initial : Nat) (given_away : Nat) :
  initial = 30 → given_away = 8 → initial - given_away = 22 := by
  sorry

end NUMINAMATH_CALUDE_valentines_remaining_l3587_358731


namespace NUMINAMATH_CALUDE_function_translation_l3587_358743

-- Define a function type for f
def FunctionType := ℝ → ℝ

-- Define the translation vector
def TranslationVector := ℝ × ℝ

-- State the theorem
theorem function_translation
  (f : FunctionType)
  (a : TranslationVector) :
  (∀ x y : ℝ, y = f (2*x - 1) + 1 ↔ 
              y + a.2 = f (2*(x + a.1) - 1) + 1) →
  (∀ x y : ℝ, y = f (2*x + 1) - 1 ↔ 
              y = f (2*(x + a.1) + 1) - 1) →
  a = (1, -2) :=
sorry

end NUMINAMATH_CALUDE_function_translation_l3587_358743


namespace NUMINAMATH_CALUDE_stacked_squares_area_l3587_358726

/-- Represents a square sheet of paper -/
structure Square where
  side_length : ℝ

/-- Represents the configuration of four stacked squares -/
structure StackedSquares where
  base : Square
  rotated45 : Square
  middle : Square
  rotated90 : Square

/-- The area of the polygon formed by the stacked squares -/
def polygon_area (s : StackedSquares) : ℝ := sorry

theorem stacked_squares_area :
  ∀ (s : StackedSquares),
    s.base.side_length = 8 ∧
    s.rotated45.side_length = 8 ∧
    s.middle.side_length = 8 ∧
    s.rotated90.side_length = 8 →
    polygon_area s = 192 - 128 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_stacked_squares_area_l3587_358726


namespace NUMINAMATH_CALUDE_scientific_notation_of_1230000_l3587_358783

theorem scientific_notation_of_1230000 :
  (1230000 : ℝ) = 1.23 * (10 : ℝ) ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1230000_l3587_358783


namespace NUMINAMATH_CALUDE_two_digit_property_three_digit_property_l3587_358747

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Three-digit positive integer -/
def ThreeDigitInt (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- Converts a two-digit number to its digits -/
def toDigits2 (n : ℕ) : ℕ × ℕ := (n / 10, n % 10)

/-- Converts a three-digit number to its digits -/
def toDigits3 (n : ℕ) : ℕ × ℕ × ℕ := (n / 100, (n / 10) % 10, n % 10)

theorem two_digit_property (n : ℕ) (h : TwoDigitInt n) :
  let (a, b) := toDigits2 n
  (a + 1) * (b + 1) = n + 1 ↔ b = 9 := by sorry

theorem three_digit_property (n : ℕ) (h : ThreeDigitInt n) :
  let (a, b, c) := toDigits3 n
  (a + 1) * (b + 1) * (c + 1) = n + 1 ↔ b = 9 ∧ c = 9 := by sorry

end NUMINAMATH_CALUDE_two_digit_property_three_digit_property_l3587_358747


namespace NUMINAMATH_CALUDE_vacuum_cost_proof_l3587_358702

/-- The cost of the vacuum cleaner Daria is saving for -/
def vacuum_cost : ℕ := 120

/-- The initial amount Daria has collected -/
def initial_amount : ℕ := 20

/-- The amount Daria adds to her savings each week -/
def weekly_savings : ℕ := 10

/-- The number of weeks Daria needs to save -/
def weeks_to_save : ℕ := 10

/-- Theorem stating that the vacuum cost is correct given the initial amount,
    weekly savings, and number of weeks to save -/
theorem vacuum_cost_proof :
  vacuum_cost = initial_amount + weekly_savings * weeks_to_save := by
  sorry

end NUMINAMATH_CALUDE_vacuum_cost_proof_l3587_358702


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l3587_358721

theorem square_area_error_percentage (s : ℝ) (h : s > 0) :
  let measured_side := s * 1.01
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let error_percentage := (area_error / actual_area) * 100
  error_percentage = 2.01 := by
sorry


end NUMINAMATH_CALUDE_square_area_error_percentage_l3587_358721


namespace NUMINAMATH_CALUDE_billys_dime_piles_l3587_358758

/-- Given Billy's coin arrangement, prove the number of dime piles -/
theorem billys_dime_piles 
  (quarter_piles : ℕ) 
  (coins_per_pile : ℕ) 
  (total_coins : ℕ) 
  (h1 : quarter_piles = 2)
  (h2 : coins_per_pile = 4)
  (h3 : total_coins = 20) :
  (total_coins - quarter_piles * coins_per_pile) / coins_per_pile = 3 := by
sorry

end NUMINAMATH_CALUDE_billys_dime_piles_l3587_358758


namespace NUMINAMATH_CALUDE_division_problem_l3587_358734

theorem division_problem (a b : ℕ+) (q r : ℤ) 
  (h1 : (a : ℤ) * (b : ℤ) = q * ((a : ℤ) + (b : ℤ)) + r)
  (h2 : 0 ≤ r ∧ r < (a : ℤ) + (b : ℤ))
  (h3 : q^2 + r = 2011) :
  ∃ t : ℕ, 1 ≤ t ∧ t ≤ 45 ∧ 
  ((a : ℤ) = t ∧ (b : ℤ) = t + 2012 ∨ (a : ℤ) = t + 2012 ∧ (b : ℤ) = t) :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l3587_358734


namespace NUMINAMATH_CALUDE_range_of_a_l3587_358729

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ici (-1 : ℝ), f a x ≥ a) →
  -2 ≤ a ∧ a ≤ (-1 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3587_358729


namespace NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l3587_358768

/-- The number of different rectangles in a rectangular grid --/
def num_rectangles (rows : ℕ) (cols : ℕ) : ℕ :=
  (rows.choose 2) * (cols.choose 2)

/-- Theorem: In a 5x4 grid, the number of different rectangles is 60 --/
theorem rectangles_in_5x4_grid :
  num_rectangles 5 4 = 60 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l3587_358768


namespace NUMINAMATH_CALUDE_euro_problem_l3587_358744

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- State the theorem
theorem euro_problem (n : ℝ) :
  euro 8 (euro 4 n) = 640 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_euro_problem_l3587_358744


namespace NUMINAMATH_CALUDE_bank_document_error_l3587_358778

def ends_with (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem bank_document_error (S D N R : ℕ) : 
  ends_with S 7 →
  ends_with N 3 →
  ends_with D 5 →
  ends_with R 1 →
  S = D * N + R →
  False :=
by sorry

end NUMINAMATH_CALUDE_bank_document_error_l3587_358778


namespace NUMINAMATH_CALUDE_calculation_proof_l3587_358718

theorem calculation_proof : 56.8 * 35.7 + 56.8 * 28.5 + 64.2 * 43.2 = 6420 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3587_358718


namespace NUMINAMATH_CALUDE_shortest_path_across_river_l3587_358773

/-- Given two points A and B on opposite sides of a straight line (river),
    with A being 5 km north and 1 km west of B,
    prove that the shortest path from A to B crossing the line perpendicularly
    is 6 km long. -/
theorem shortest_path_across_river (A B : ℝ × ℝ) : 
  A.1 = B.1 - 1 →  -- A is 1 km west of B
  A.2 = B.2 + 5 →  -- A is 5 km north of B
  ∃ (C : ℝ × ℝ), 
    (C.1 - A.1) * (B.2 - A.2) = (C.2 - A.2) * (B.1 - A.1) ∧  -- C is on the line AB
    (C.2 = A.2 ∨ C.2 = B.2) ∧  -- C is on the same level as A or B (representing the river)
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) + 
    Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 6 :=
by sorry

end NUMINAMATH_CALUDE_shortest_path_across_river_l3587_358773


namespace NUMINAMATH_CALUDE_moon_arrangements_count_l3587_358712

/-- The number of letters in the word "MOON" -/
def word_length : ℕ := 4

/-- The number of repeated letters (O's) in the word "MOON" -/
def repeated_letters : ℕ := 2

/-- The number of unique arrangements of the letters in "MOON" -/
def moon_arrangements : ℕ := Nat.factorial word_length / Nat.factorial repeated_letters

theorem moon_arrangements_count : moon_arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_arrangements_count_l3587_358712


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l3587_358724

/-- Represents a repeating decimal with a repeating part of two digits -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b) / 99

theorem repeating_decimal_ratio : 
  (RepeatingDecimal 5 4) / (RepeatingDecimal 1 8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l3587_358724


namespace NUMINAMATH_CALUDE_rectangle_area_l3587_358796

theorem rectangle_area (a b : ℝ) (h : a^2 + b^2 - 8*a - 6*b + 25 = 0) : a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3587_358796


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3587_358742

theorem geometric_sequence_sum (a r : ℚ) (n : ℕ) (h1 : a = 1/3) (h2 : r = 1/3) :
  (a * (1 - r^n) / (1 - r) = 26/81) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3587_358742


namespace NUMINAMATH_CALUDE_triangle_angle_inequalities_l3587_358784

theorem triangle_angle_inequalities (α β γ : ℝ) 
  (h_triangle : α + β + γ = Real.pi) : 
  ((1 - Real.cos α) * (1 - Real.cos β) * (1 - Real.cos γ) ≥ Real.cos α * Real.cos β * Real.cos γ) ∧
  (12 * Real.cos α * Real.cos β * Real.cos γ ≤ 
   2 * Real.cos α * Real.cos β + 2 * Real.cos α * Real.cos γ + 2 * Real.cos β * Real.cos γ) ∧
  (2 * Real.cos α * Real.cos β + 2 * Real.cos α * Real.cos γ + 2 * Real.cos β * Real.cos γ ≤ 
   Real.cos α + Real.cos β + Real.cos γ) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_inequalities_l3587_358784


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_root_l3587_358752

theorem quadratic_equation_integer_root (k : ℕ) : 
  (∃ x : ℕ, x^2 - 34*x + 34*k - 1 = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_root_l3587_358752


namespace NUMINAMATH_CALUDE_geometric_mean_max_l3587_358766

theorem geometric_mean_max (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_arithmetic_mean : (a + b) / 2 = 4) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) / 2 = 4 ∧ 
  Real.sqrt (x * y) = 4 ∧ 
  ∀ (c d : ℝ), c > 0 → d > 0 → (c + d) / 2 = 4 → Real.sqrt (c * d) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_max_l3587_358766


namespace NUMINAMATH_CALUDE_wendy_furniture_assembly_time_l3587_358719

/-- Calculates the total time spent assembling furniture --/
def total_assembly_time (chair_count : ℕ) (table_count : ℕ) (bookshelf_count : ℕ)
                        (chair_time : ℕ) (table_time : ℕ) (bookshelf_time : ℕ) : ℕ :=
  chair_count * chair_time + table_count * table_time + bookshelf_count * bookshelf_time

/-- Theorem stating that the total assembly time for Wendy's furniture is 84 minutes --/
theorem wendy_furniture_assembly_time :
  total_assembly_time 4 3 2 6 10 15 = 84 := by
  sorry

#eval total_assembly_time 4 3 2 6 10 15

end NUMINAMATH_CALUDE_wendy_furniture_assembly_time_l3587_358719


namespace NUMINAMATH_CALUDE_park_fencing_cost_l3587_358736

/-- Proves that for a rectangular park with sides in the ratio 3:2, area of 4704 sq m,
    and a total fencing cost of 140, the cost of fencing per meter is 50 paise. -/
theorem park_fencing_cost (length width : ℝ) (area perimeter total_cost : ℝ) : 
  length / width = 3 / 2 →
  area = 4704 →
  length * width = area →
  perimeter = 2 * (length + width) →
  total_cost = 140 →
  (total_cost / perimeter) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_park_fencing_cost_l3587_358736


namespace NUMINAMATH_CALUDE_total_price_calculation_l3587_358781

theorem total_price_calculation (refrigerator_price washing_machine_price total_price : ℕ) : 
  refrigerator_price = 4275 →
  washing_machine_price = refrigerator_price - 1490 →
  total_price = refrigerator_price + washing_machine_price →
  total_price = 7060 := by
sorry

end NUMINAMATH_CALUDE_total_price_calculation_l3587_358781


namespace NUMINAMATH_CALUDE_pet_store_cages_l3587_358755

theorem pet_store_cages (total_puppies : ℕ) (puppies_per_cage : ℕ) (last_cage_puppies : ℕ) :
  total_puppies = 38 →
  puppies_per_cage = 6 →
  last_cage_puppies = 4 →
  (total_puppies / puppies_per_cage + 1 : ℕ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3587_358755


namespace NUMINAMATH_CALUDE_power_function_monotone_iff_m_eq_three_l3587_358791

/-- A power function f(x) = (m^2 - 2m - 2) * x^(m-2) is monotonically increasing on (0, +∞) if and only if m = 3 -/
theorem power_function_monotone_iff_m_eq_three (m : ℝ) :
  (∀ x > 0, Monotone (fun x => (m^2 - 2*m - 2) * x^(m-2))) ↔ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_monotone_iff_m_eq_three_l3587_358791


namespace NUMINAMATH_CALUDE_special_natural_numbers_l3587_358700

theorem special_natural_numbers : 
  {x : ℕ | ∃ (y z : ℤ), x = 2 * y^2 - 1 ∧ x^2 = 2 * z^2 - 1} = {1, 7} := by
  sorry

end NUMINAMATH_CALUDE_special_natural_numbers_l3587_358700


namespace NUMINAMATH_CALUDE_garden_area_l3587_358762

/-- The total area of two triangles with given bases and a shared altitude -/
theorem garden_area (base1 base2 : ℝ) (area1 : ℝ) (h : ℝ) : 
  base1 = 50 →
  base2 = 40 →
  area1 = 800 →
  area1 = (1/2) * base1 * h →
  (1/2) * base1 * h + (1/2) * base2 * h = 1440 := by
  sorry

#check garden_area

end NUMINAMATH_CALUDE_garden_area_l3587_358762


namespace NUMINAMATH_CALUDE_claire_photos_l3587_358703

/-- Given that:
    - Lisa and Robert have taken the same number of photos
    - Lisa has taken 3 times as many photos as Claire
    - Robert has taken 28 more photos than Claire
    Prove that Claire has taken 14 photos. -/
theorem claire_photos (claire lisa robert : ℕ) 
  (h1 : lisa = robert)
  (h2 : lisa = 3 * claire)
  (h3 : robert = claire + 28) :
  claire = 14 := by
  sorry

end NUMINAMATH_CALUDE_claire_photos_l3587_358703


namespace NUMINAMATH_CALUDE_integral_inequality_l3587_358777

theorem integral_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_incr : Monotone f) (h_f0 : f 0 = 0) : 
  ∫ x in (0)..(1), f x * (deriv f x) ≥ (1/2) * (∫ x in (0)..(1), f x)^2 := by
  sorry

end NUMINAMATH_CALUDE_integral_inequality_l3587_358777


namespace NUMINAMATH_CALUDE_paul_filled_three_bags_sunday_l3587_358771

/-- Calculates the number of bags filled on Sunday given the total cans collected,
    bags filled on Saturday, and cans per bag. -/
def bags_filled_sunday (total_cans : ℕ) (saturday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (total_cans - saturday_bags * cans_per_bag) / cans_per_bag

/-- Proves that for the given problem, Paul filled 3 bags on Sunday. -/
theorem paul_filled_three_bags_sunday :
  bags_filled_sunday 72 6 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_paul_filled_three_bags_sunday_l3587_358771


namespace NUMINAMATH_CALUDE_oliver_monster_club_cards_l3587_358769

/-- Represents Oliver's card collection --/
structure CardCollection where
  alien_baseball : ℕ
  monster_club : ℕ
  battle_gremlins : ℕ

/-- The conditions of Oliver's card collection --/
def oliver_collection : CardCollection :=
  { alien_baseball := 18,
    monster_club := 27,
    battle_gremlins := 72 }

/-- Theorem stating the number of Monster Club cards Oliver has --/
theorem oliver_monster_club_cards :
  oliver_collection.monster_club = 27 ∧
  oliver_collection.monster_club = (3 / 2 : ℚ) * oliver_collection.alien_baseball ∧
  oliver_collection.battle_gremlins = 72 ∧
  oliver_collection.battle_gremlins = 4 * oliver_collection.alien_baseball :=
by
  sorry

end NUMINAMATH_CALUDE_oliver_monster_club_cards_l3587_358769


namespace NUMINAMATH_CALUDE_fourth_student_added_25_l3587_358793

/-- The number of jellybeans added by the fourth student to the average of the first three guesses -/
def jellybeans_added (first_guess : ℕ) (fourth_guess : ℕ) : ℕ :=
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let average := (first_guess + second_guess + third_guess) / 3
  fourth_guess - average

/-- Theorem stating that given the conditions in the problem, the fourth student added 25 jellybeans -/
theorem fourth_student_added_25 :
  jellybeans_added 100 525 = 25 := by
  sorry

#eval jellybeans_added 100 525

end NUMINAMATH_CALUDE_fourth_student_added_25_l3587_358793


namespace NUMINAMATH_CALUDE_ashas_borrowed_amount_l3587_358786

theorem ashas_borrowed_amount (brother mother granny savings spent_fraction remaining : ℚ)
  (h1 : brother = 20)
  (h2 : mother = 30)
  (h3 : granny = 70)
  (h4 : savings = 100)
  (h5 : spent_fraction = 3/4)
  (h6 : remaining = 65)
  (h7 : (1 - spent_fraction) * (brother + mother + granny + savings + father) = remaining) :
  father = 40 := by
  sorry

end NUMINAMATH_CALUDE_ashas_borrowed_amount_l3587_358786


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3587_358733

/-- Given a large square with side length z and an inscribed smaller square with side length w,
    prove that the perimeter of one of the four identical rectangles formed between the two squares is w + z. -/
theorem rectangle_perimeter (z w : ℝ) (hz : z > 0) (hw : w > 0) (hzw : z > w) : 
  let long_side := w
  let short_side := (z - w) / 2
  2 * long_side + 2 * short_side = w + z := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3587_358733


namespace NUMINAMATH_CALUDE_exist_good_numbers_without_digit_sum_property_l3587_358770

/-- A natural number is "good" if its decimal representation contains only zeros and ones. -/
def isGood (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The sum of digits of a natural number in base 10. -/
def digitSum (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Theorem stating that there exist good numbers whose product is good,
    but the sum of digits property doesn't hold. -/
theorem exist_good_numbers_without_digit_sum_property :
  ∃ (A B : ℕ), isGood A ∧ isGood B ∧ isGood (A * B) ∧
    digitSum (A * B) ≠ digitSum A * digitSum B := by
  sorry


end NUMINAMATH_CALUDE_exist_good_numbers_without_digit_sum_property_l3587_358770


namespace NUMINAMATH_CALUDE_initial_alcohol_percentage_l3587_358794

/-- Proves that the initial alcohol percentage in a mixture is 20% given the conditions --/
theorem initial_alcohol_percentage 
  (initial_volume : ℝ) 
  (added_water : ℝ) 
  (final_percentage : ℝ) : ℝ :=
  by
  have h1 : initial_volume = 18 := by sorry
  have h2 : added_water = 3 := by sorry
  have h3 : final_percentage = 17.14285714285715 := by sorry
  
  let final_volume : ℝ := initial_volume + added_water
  let initial_percentage : ℝ := (final_percentage * final_volume) / initial_volume
  
  have h4 : initial_percentage = 20 := by sorry
  
  exact initial_percentage

end NUMINAMATH_CALUDE_initial_alcohol_percentage_l3587_358794


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l3587_358756

/-- Represents the categories of teachers -/
inductive TeacherCategory
  | Senior
  | Intermediate
  | Junior

/-- Represents the school's teacher population -/
structure SchoolPopulation where
  total : Nat
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Represents the selected sample of teachers -/
structure SelectedSample where
  total : Nat
  senior : Nat
  intermediate : Nat
  junior : Nat

/-- Checks if the sample maintains the same proportion as the population -/
def isProportionalSample (pop : SchoolPopulation) (sample : SelectedSample) : Prop :=
  pop.senior * sample.total = sample.senior * pop.total ∧
  pop.intermediate * sample.total = sample.intermediate * pop.total ∧
  pop.junior * sample.total = sample.junior * pop.total

/-- The main theorem stating that the given sample is proportional -/
theorem stratified_sampling_correct 
  (pop : SchoolPopulation)
  (sample : SelectedSample)
  (h1 : pop.total = 150)
  (h2 : pop.senior = 15)
  (h3 : pop.intermediate = 45)
  (h4 : pop.junior = 90)
  (h5 : sample.total = 30)
  (h6 : sample.senior = 3)
  (h7 : sample.intermediate = 9)
  (h8 : sample.junior = 18) :
  isProportionalSample pop sample :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l3587_358756


namespace NUMINAMATH_CALUDE_problem_statement_l3587_358716

theorem problem_statement (t : ℝ) : 
  let x := 3 - 2*t
  let y := 5*t + 3
  x = 1 → y = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3587_358716


namespace NUMINAMATH_CALUDE_solution_values_l3587_358765

theorem solution_values (a : ℝ) : (a - 2) ^ (a + 1) = 1 → a = -1 ∨ a = 3 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_values_l3587_358765


namespace NUMINAMATH_CALUDE_equality_of_squared_terms_l3587_358709

theorem equality_of_squared_terms (a b : ℝ) : 7 * a^2 * b - 7 * b * a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_squared_terms_l3587_358709


namespace NUMINAMATH_CALUDE_cannot_make_65_cents_l3587_358750

def coin_value (coin : Nat) : Nat :=
  match coin with
  | 0 => 5  -- nickel
  | 1 => 10 -- dime
  | 2 => 25 -- quarter
  | 3 => 50 -- half-dollar
  | _ => 0  -- invalid coin

def is_valid_coin (c : Nat) : Prop := c ≤ 3

theorem cannot_make_65_cents :
  ¬ ∃ (a b c d e : Nat),
    is_valid_coin a ∧ is_valid_coin b ∧ is_valid_coin c ∧ is_valid_coin d ∧ is_valid_coin e ∧
    coin_value a + coin_value b + coin_value c + coin_value d + coin_value e = 65 :=
by sorry

end NUMINAMATH_CALUDE_cannot_make_65_cents_l3587_358750


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_other_numbers_rational_sqrt_three_unique_irrational_l3587_358753

theorem sqrt_three_irrational :
  ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ)) :=
by sorry

theorem other_numbers_rational :
  ∃ (a b c d e f : ℤ), 
    b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0 ∧
    (-32 : ℚ) / 7 = (a : ℚ) / (b : ℚ) ∧
    (0 : ℚ) = (c : ℚ) / (d : ℚ) ∧
    (3.5 : ℚ) = (e : ℚ) / (f : ℚ) :=
by sorry

theorem sqrt_three_unique_irrational 
  (h1 : ¬ (∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = (p : ℚ) / (q : ℚ)))
  (h2 : ∃ (a b c d e f : ℤ), 
    b ≠ 0 ∧ d ≠ 0 ∧ f ≠ 0 ∧
    (-32 : ℚ) / 7 = (a : ℚ) / (b : ℚ) ∧
    (0 : ℚ) = (c : ℚ) / (d : ℚ) ∧
    (3.5 : ℚ) = (e : ℚ) / (f : ℚ)) :
  Real.sqrt 3 = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_other_numbers_rational_sqrt_three_unique_irrational_l3587_358753


namespace NUMINAMATH_CALUDE_positive_solution_x_l3587_358737

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 10 - 3 * x - 2 * y)
  (eq2 : y * z = 10 - 5 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 2 * z)
  (x_pos : x > 0) :
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_x_l3587_358737


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_l3587_358704

theorem smallest_number_with_remainder (n : ℕ) : 
  300 % 25 = 0 →
  324 > 300 ∧
  324 % 25 = 24 ∧
  ∀ m : ℕ, m > 300 ∧ m % 25 = 24 → m ≥ 324 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_l3587_358704


namespace NUMINAMATH_CALUDE_total_students_l3587_358710

theorem total_students (absent_percentage : ℝ) (present_students : ℕ) 
  (h1 : absent_percentage = 14) 
  (h2 : present_students = 86) : 
  ↑present_students / (1 - absent_percentage / 100) = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l3587_358710


namespace NUMINAMATH_CALUDE_diamonds_in_F20_l3587_358761

/-- Definition of the number of diamonds in figure F_n -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 9
  else n^2 + (n-1)^2

/-- Theorem: The number of diamonds in F_20 is 761 -/
theorem diamonds_in_F20 :
  num_diamonds 20 = 761 := by sorry

end NUMINAMATH_CALUDE_diamonds_in_F20_l3587_358761
