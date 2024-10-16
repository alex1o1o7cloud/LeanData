import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_equation_l2381_238118

/-- The equation of a hyperbola C with focal length 2√5, whose asymptotes are tangent to the parabola y = 1/16x² + 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^2 + b^2 = 20) -- Focal length condition
  (h4 : ∃ x : ℝ, (1/16 * x^2 + 1 = (a/b) * x)) -- Tangency condition
  : a^2 = 4 ∧ b^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2381_238118


namespace NUMINAMATH_CALUDE_equation_solutions_l2381_238154

-- Define the equations
def equation1 (x : ℚ) : Prop := (1 - x) / 3 - 2 = x / 6

def equation2 (x : ℚ) : Prop := (x + 1) / (1/4) - (x - 2) / (1/2) = 5

-- State the theorem
theorem equation_solutions :
  (∃ x : ℚ, equation1 x ∧ x = -10/3) ∧
  (∃ x : ℚ, equation2 x ∧ x = -3/2) :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2381_238154


namespace NUMINAMATH_CALUDE_angle_EFG_value_l2381_238184

/-- A configuration of a square inside a regular octagon sharing one side -/
structure SquareInOctagon where
  /-- The measure of an internal angle of the regular octagon -/
  octagon_angle : ℝ
  /-- The measure of an internal angle of the square -/
  square_angle : ℝ
  /-- The measure of angle EFH -/
  angle_EFH : ℝ
  /-- The measure of angle EFG -/
  angle_EFG : ℝ

/-- Properties of the SquareInOctagon configuration -/
axiom octagon_angle_value (config : SquareInOctagon) : config.octagon_angle = 135
axiom square_angle_value (config : SquareInOctagon) : config.square_angle = 90
axiom angle_EFH_value (config : SquareInOctagon) : config.angle_EFH = config.octagon_angle - config.square_angle
axiom isosceles_triangle (config : SquareInOctagon) : config.angle_EFG = (180 - config.angle_EFH) / 2

/-- The main theorem: angle EFG measures 67.5° -/
theorem angle_EFG_value (config : SquareInOctagon) : config.angle_EFG = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_EFG_value_l2381_238184


namespace NUMINAMATH_CALUDE_derivative_at_negative_two_l2381_238155

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem derivative_at_negative_two (h : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
  |((f (-2 + Δx) - f (-2 - Δx)) / Δx) - (-2)| < ε) : 
  deriv f (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_negative_two_l2381_238155


namespace NUMINAMATH_CALUDE_problem_solution_l2381_238129

theorem problem_solution (x y : ℤ) : 
  y = 3 * x^2 ∧ 
  (2 * x : ℚ) / 5 = 1 / (1 - 2 / (3 + 1 / (4 - 5 / (6 - x)))) → 
  y = 147 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2381_238129


namespace NUMINAMATH_CALUDE_medicine_tablets_l2381_238199

theorem medicine_tablets (num_b : ℕ) (num_a : ℕ) (min_extract : ℕ) : 
  num_b = 14 → 
  min_extract = 16 → 
  min_extract = num_b + 2 →
  num_a = 2 :=
by sorry

end NUMINAMATH_CALUDE_medicine_tablets_l2381_238199


namespace NUMINAMATH_CALUDE_fifth_month_sale_l2381_238136

theorem fifth_month_sale
  (sale1 sale2 sale3 sale4 sale6 : ℕ)
  (average : ℚ)
  (h1 : sale1 = 2500)
  (h2 : sale2 = 6500)
  (h3 : sale3 = 9855)
  (h4 : sale4 = 7230)
  (h6 : sale6 = 11915)
  (h_avg : average = 7500)
  (h_total : (sale1 + sale2 + sale3 + sale4 + sale6 + sale5) / 6 = average) :
  sale5 = 7000 := by
sorry

end NUMINAMATH_CALUDE_fifth_month_sale_l2381_238136


namespace NUMINAMATH_CALUDE_no_valid_n_l2381_238192

theorem no_valid_n : ¬∃ n : ℕ+, 
  (1000 ≤ (n : ℝ) / 4 ∧ (n : ℝ) / 4 < 10000) ∧ 
  (1000 ≤ 4 * (n : ℝ) ∧ 4 * (n : ℝ) < 10000) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_n_l2381_238192


namespace NUMINAMATH_CALUDE_passes_through_origin_symmetric_about_y_axis_l2381_238170

-- Define the quadratic function
def f (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 + m - 2

-- Theorem for passing through the origin
theorem passes_through_origin (m : ℝ) : 
  (f m 0 = 0) ↔ (m = 1 ∨ m = -2) := by sorry

-- Theorem for symmetry about y-axis
theorem symmetric_about_y_axis (m : ℝ) :
  (∀ x, f m x = f m (-x)) ↔ m = 0 := by sorry

end NUMINAMATH_CALUDE_passes_through_origin_symmetric_about_y_axis_l2381_238170


namespace NUMINAMATH_CALUDE_area_of_remaining_rectangle_l2381_238198

theorem area_of_remaining_rectangle (s : ℝ) (h1 : s = 3) : s^2 - (1 * 3 + 1^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_remaining_rectangle_l2381_238198


namespace NUMINAMATH_CALUDE_product_remainder_remainder_98_102_mod_11_l2381_238126

theorem product_remainder (a b n : ℕ) (h : n > 0) : (a * b) % n = ((a % n) * (b % n)) % n := by sorry

theorem remainder_98_102_mod_11 : (98 * 102) % 11 = 1 := by sorry

end NUMINAMATH_CALUDE_product_remainder_remainder_98_102_mod_11_l2381_238126


namespace NUMINAMATH_CALUDE_burger_problem_l2381_238177

theorem burger_problem (total_time : ℕ) (cook_time_per_side : ℕ) (grill_capacity : ℕ) (total_guests : ℕ) :
  total_time = 72 →
  cook_time_per_side = 4 →
  grill_capacity = 5 →
  total_guests = 30 →
  ∃ (burgers_per_half : ℕ),
    burgers_per_half * (total_guests / 2) + (total_guests / 2) = 
      (total_time / (2 * cook_time_per_side)) * grill_capacity ∧
    burgers_per_half = 2 :=
by sorry

end NUMINAMATH_CALUDE_burger_problem_l2381_238177


namespace NUMINAMATH_CALUDE_exactly_one_double_digit_sum_two_l2381_238128

/-- Sum of digits function -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Predicate for two-digit numbers -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The main theorem -/
theorem exactly_one_double_digit_sum_two :
  ∃! x : ℕ, is_two_digit x ∧ digit_sum (digit_sum x) = 2 := by sorry

end NUMINAMATH_CALUDE_exactly_one_double_digit_sum_two_l2381_238128


namespace NUMINAMATH_CALUDE_cubic_sum_zero_l2381_238162

theorem cubic_sum_zero (a b c n : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a^3 + b^3 + c^3 = 0) : 
  a^(2*n+1) + b^(2*n+1) + c^(2*n+1) = 0 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_zero_l2381_238162


namespace NUMINAMATH_CALUDE_race_overtake_equation_l2381_238180

/-- The time it takes for John to overtake Steve in a race --/
def overtake_time (initial_distance : ℝ) (john_initial_speed : ℝ) (john_acceleration : ℝ) (steve_speed : ℝ) (final_distance : ℝ) : ℝ → Prop :=
  λ t => 0.5 * john_acceleration * t^2 + john_initial_speed * t - steve_speed * t - initial_distance - final_distance = 0

theorem race_overtake_equation :
  let initial_distance : ℝ := 15
  let john_initial_speed : ℝ := 3.5
  let john_acceleration : ℝ := 0.25
  let steve_speed : ℝ := 3.8
  let final_distance : ℝ := 2
  ∃ t : ℝ, overtake_time initial_distance john_initial_speed john_acceleration steve_speed final_distance t :=
by sorry

end NUMINAMATH_CALUDE_race_overtake_equation_l2381_238180


namespace NUMINAMATH_CALUDE_forest_area_relationship_l2381_238106

/-- 
Given forest areas a, b, c for three consecutive years,
with constant growth rate in the last two years,
prove that ac = b²
-/
theorem forest_area_relationship (a b c : ℝ) 
  (h : ∃ x : ℝ, b = a * (1 + x) ∧ c = b * (1 + x)) : 
  a * c = b ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_forest_area_relationship_l2381_238106


namespace NUMINAMATH_CALUDE_stones_required_l2381_238134

def hall_length : ℝ := 45
def hall_width : ℝ := 25
def stone_length : ℝ := 1.2  -- 12 dm = 1.2 m
def stone_width : ℝ := 0.7   -- 7 dm = 0.7 m

theorem stones_required :
  ⌈(hall_length * hall_width) / (stone_length * stone_width)⌉ = 1341 := by
  sorry

end NUMINAMATH_CALUDE_stones_required_l2381_238134


namespace NUMINAMATH_CALUDE_rent_reduction_percentage_l2381_238144

-- Define the room prices
def cheap_room_price : ℕ := 40
def expensive_room_price : ℕ := 60

-- Define the total rent
def total_rent : ℕ := 1000

-- Define the number of rooms to be moved
def rooms_to_move : ℕ := 10

-- Define the function to calculate the new total rent
def new_total_rent : ℕ := total_rent - rooms_to_move * (expensive_room_price - cheap_room_price)

-- Define the reduction percentage
def reduction_percentage : ℚ := (total_rent - new_total_rent : ℚ) / total_rent * 100

-- Theorem statement
theorem rent_reduction_percentage :
  reduction_percentage = 20 :=
sorry

end NUMINAMATH_CALUDE_rent_reduction_percentage_l2381_238144


namespace NUMINAMATH_CALUDE_room_width_calculation_l2381_238191

def room_length : ℝ := 25
def room_height : ℝ := 12
def door_length : ℝ := 6
def door_width : ℝ := 3
def window_length : ℝ := 4
def window_width : ℝ := 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 8
def total_cost : ℝ := 7248

theorem room_width_calculation (x : ℝ) :
  (2 * (room_length * room_height + x * room_height) - 
   (door_length * door_width + ↑num_windows * window_length * window_width)) * 
   cost_per_sqft = total_cost →
  x = 15 := by sorry

end NUMINAMATH_CALUDE_room_width_calculation_l2381_238191


namespace NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l2381_238194

theorem arccos_gt_arctan_iff (x : ℝ) : Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1) (Real.sqrt 2 / 2) ∧ x ≠ Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l2381_238194


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2381_238179

theorem complex_equation_solution (a b c : ℂ) 
  (eq : 3*a + 4*b + 5*c = 0) 
  (norm_a : Complex.abs a = 1) 
  (norm_b : Complex.abs b = 1) 
  (norm_c : Complex.abs c = 1) : 
  a * (b + c) = -3/5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2381_238179


namespace NUMINAMATH_CALUDE_prize_distribution_l2381_238113

/-- The number of ways to distribute prizes in a class --/
theorem prize_distribution (n m : ℕ) (hn : n = 28) (hm : m = 4) :
  /- Identical prizes, at most one per student -/
  (n.choose m = 20475) ∧ 
  /- Identical prizes, more than one per student allowed -/
  ((n + m - 1).choose m = 31465) ∧ 
  /- Distinct prizes, at most one per student -/
  (n * (n - 1) * (n - 2) * (n - 3) = 491400) ∧ 
  /- Distinct prizes, more than one per student allowed -/
  (n ^ m = 614656) :=
by sorry

end NUMINAMATH_CALUDE_prize_distribution_l2381_238113


namespace NUMINAMATH_CALUDE_marble_ratio_proof_l2381_238157

theorem marble_ratio_proof (mabel_marbles katrina_marbles amanda_marbles : ℕ) 
  (h1 : mabel_marbles = 5 * katrina_marbles)
  (h2 : mabel_marbles = 85)
  (h3 : mabel_marbles = amanda_marbles + 63)
  : (amanda_marbles + 12) / katrina_marbles = 2 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_proof_l2381_238157


namespace NUMINAMATH_CALUDE_road_construction_equation_l2381_238143

theorem road_construction_equation (x : ℝ) (h : x > 0) :
  let road_length : ℝ := 1200
  let speed_increase : ℝ := 0.2
  let days_saved : ℝ := 2
  (road_length / x) - (road_length / ((1 + speed_increase) * x)) = days_saved :=
by sorry

end NUMINAMATH_CALUDE_road_construction_equation_l2381_238143


namespace NUMINAMATH_CALUDE_schedule_theorem_l2381_238108

-- Define the number of classes
def total_classes : ℕ := 6

-- Define the number of morning slots
def morning_slots : ℕ := 4

-- Define the number of afternoon slots
def afternoon_slots : ℕ := 2

-- Define the function to calculate the number of arrangements
def schedule_arrangements (n : ℕ) (m : ℕ) (a : ℕ) : ℕ :=
  (m.choose 1) * (a.choose 1) * (n - 2).factorial

-- Theorem statement
theorem schedule_theorem :
  schedule_arrangements total_classes morning_slots afternoon_slots = 192 :=
by sorry

end NUMINAMATH_CALUDE_schedule_theorem_l2381_238108


namespace NUMINAMATH_CALUDE_backpack_profit_theorem_l2381_238124

/-- Represents the profit equation for a backpack sale -/
def profit_equation (x : ℝ) : Prop :=
  (1 + 0.5) * x * 0.8 - x = 8

/-- Theorem stating the profit equation holds for a backpack sale with given conditions -/
theorem backpack_profit_theorem (x : ℝ) 
  (h_markup : ℝ → ℝ := λ price => (1 + 0.5) * price)
  (h_discount : ℝ → ℝ := λ price => 0.8 * price)
  (h_profit : ℝ := 8) :
  profit_equation x := by
  sorry

end NUMINAMATH_CALUDE_backpack_profit_theorem_l2381_238124


namespace NUMINAMATH_CALUDE_floor_paving_cost_l2381_238146

theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (cost_per_sqm : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : cost_per_sqm = 600) : 
  length * width * cost_per_sqm = 12375 := by
sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l2381_238146


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2381_238125

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l2381_238125


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2381_238139

theorem solve_exponential_equation :
  ∀ x : ℝ, (64 : ℝ)^(3*x + 1) = (16 : ℝ)^(4*x - 5) ↔ x = -13 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2381_238139


namespace NUMINAMATH_CALUDE_sin_150_cos_30_plus_cos_150_sin_30_eq_zero_l2381_238158

theorem sin_150_cos_30_plus_cos_150_sin_30_eq_zero : 
  Real.sin (150 * π / 180) * Real.cos (30 * π / 180) + 
  Real.cos (150 * π / 180) * Real.sin (30 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_cos_30_plus_cos_150_sin_30_eq_zero_l2381_238158


namespace NUMINAMATH_CALUDE_triangle_side_range_l2381_238176

/-- Given a triangle ABC where c = √2 and a cos C = c sin A, 
    the length of side BC is in the range (√2, 2) -/
theorem triangle_side_range (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  a * Real.cos C = c * Real.sin A →
  ∃ (BC : ℝ), BC > Real.sqrt 2 ∧ BC < 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l2381_238176


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2381_238172

def U : Finset Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Finset Nat := {1, 2, 5, 7}

theorem complement_of_A_in_U : 
  (U \ A : Finset Nat) = {3, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2381_238172


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l2381_238175

-- Define a point in a 2D Cartesian coordinate system
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the origin
def origin : Point2D := ⟨0, 0⟩

-- Define the point P
def P : Point2D := ⟨2, 4⟩

-- Theorem stating that the coordinates of P with respect to the origin are (2,4)
theorem coordinates_wrt_origin :
  (P.x - origin.x = 2) ∧ (P.y - origin.y = 4) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l2381_238175


namespace NUMINAMATH_CALUDE_inequality_proof_l2381_238115

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2381_238115


namespace NUMINAMATH_CALUDE_no_valid_partition_l2381_238103

/-- A partition of integers into three subsets -/
def IntPartition := ℤ → Fin 3

/-- Property that n, n-50, and n+1987 belong to different subsets -/
def ValidPartition (p : IntPartition) : Prop :=
  ∀ n : ℤ, p n ≠ p (n - 50) ∧ p n ≠ p (n + 1987) ∧ p (n - 50) ≠ p (n + 1987)

/-- Theorem stating the impossibility of such a partition -/
theorem no_valid_partition : ¬ ∃ p : IntPartition, ValidPartition p := by
  sorry

end NUMINAMATH_CALUDE_no_valid_partition_l2381_238103


namespace NUMINAMATH_CALUDE_circle_cutting_terminates_l2381_238101

-- Define the circle-cutting process
def circle_cutting_process (m : ℕ) (n : ℕ) : Prop :=
  m ≥ 2 ∧ ∃ (remaining_area : ℝ), 
    remaining_area > 0 ∧
    remaining_area < (1 - 1/m)^n

-- Theorem statement
theorem circle_cutting_terminates (m : ℕ) :
  m ≥ 2 → ∃ n : ℕ, ∀ k : ℕ, k ≥ n → ¬(circle_cutting_process m k) :=
sorry

end NUMINAMATH_CALUDE_circle_cutting_terminates_l2381_238101


namespace NUMINAMATH_CALUDE_negative_half_greater_than_negative_two_thirds_l2381_238164

theorem negative_half_greater_than_negative_two_thirds :
  -0.5 > -(2/3) := by
  sorry

end NUMINAMATH_CALUDE_negative_half_greater_than_negative_two_thirds_l2381_238164


namespace NUMINAMATH_CALUDE_selling_price_calculation_l2381_238105

/-- Calculates the selling price of an article given the gain and gain percentage -/
theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) : 
  gain = 30 ∧ gain_percentage = 20 → 
  (gain / (gain_percentage / 100)) + gain = 180 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l2381_238105


namespace NUMINAMATH_CALUDE_equation_solution_l2381_238188

theorem equation_solution (x : ℝ) (number : ℝ) :
  x = 32 →
  35 - (23 - (15 - x)) = 12 * 2 / (number / 2) →
  number = -2.4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2381_238188


namespace NUMINAMATH_CALUDE_intersection_A_B_l2381_238117

def A : Set ℕ := {0, 1, 2, 3, 4}
def B : Set ℕ := {x | ∃ n ∈ A, x = 2 * n}

theorem intersection_A_B : A ∩ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2381_238117


namespace NUMINAMATH_CALUDE_power_function_inequality_l2381_238171

-- Define the power function
def f (x : ℝ) : ℝ := x^(4/5)

-- State the theorem
theorem power_function_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) :
  f ((x₁ + x₂) / 2) > (f x₁ + f x₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_inequality_l2381_238171


namespace NUMINAMATH_CALUDE_min_colors_theorem_l2381_238151

def is_multiple (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n

def valid_coloring (f : ℕ → ℕ) : Prop :=
  ∀ m n, 2 ≤ n ∧ n < m ∧ m ≤ 31 → is_multiple m n → f m ≠ f n

theorem min_colors_theorem :
  ∃ (k : ℕ) (f : ℕ → ℕ),
    (∀ n, 2 ≤ n ∧ n ≤ 31 → f n < k) ∧
    valid_coloring f ∧
    (∀ k' < k, ¬∃ f', (∀ n, 2 ≤ n ∧ n ≤ 31 → f' n < k') ∧ valid_coloring f') ∧
    k = 4 :=
sorry

end NUMINAMATH_CALUDE_min_colors_theorem_l2381_238151


namespace NUMINAMATH_CALUDE_complete_work_together_l2381_238174

/-- The number of days it takes for two workers to complete a job together,
    given the number of days it takes each worker to complete the job individually. -/
def days_to_complete_together (days_a days_b : ℚ) : ℚ :=
  1 / (1 / days_a + 1 / days_b)

/-- Theorem stating that if worker A takes 9 days and worker B takes 18 days to complete a job individually,
    then together they will complete the job in 6 days. -/
theorem complete_work_together :
  days_to_complete_together 9 18 = 6 := by
  sorry

end NUMINAMATH_CALUDE_complete_work_together_l2381_238174


namespace NUMINAMATH_CALUDE_max_min_product_l2381_238167

theorem max_min_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (sum_eq : a + b + c = 12) (prod_sum_eq : a * b + b * c + c * a = 30) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 9 * Real.sqrt 2 ∧
  ∃ (a' b' c' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    a' + b' + c' = 12 ∧ a' * b' + b' * c' + c' * a' = 30 ∧
    min (a' * b') (min (b' * c') (c' * a')) = 9 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l2381_238167


namespace NUMINAMATH_CALUDE_not_possible_when_70_possible_when_80_l2381_238133

-- Define the original triangle
structure OriginalTriangle where
  alpha : Real
  is_valid : 0 < alpha ∧ alpha < 180

-- Define the resulting triangles after cutting
structure ResultingTriangle where
  angles : Fin 3 → Real
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, 0 < angles i

-- Define the cutting process
def cut (t : OriginalTriangle) : Set ResultingTriangle := sorry

-- Theorem for the case when α = 70°
theorem not_possible_when_70 (t : OriginalTriangle) 
  (h : t.alpha = 70) : 
  ¬∃ (s : Set ResultingTriangle), s = cut t ∧ 
  (∀ rt ∈ s, ∀ i, rt.angles i < t.alpha) :=
sorry

-- Theorem for the case when α = 80°
theorem possible_when_80 (t : OriginalTriangle) 
  (h : t.alpha = 80) : 
  ∃ (s : Set ResultingTriangle), s = cut t ∧ 
  (∀ rt ∈ s, ∀ i, rt.angles i < t.alpha) :=
sorry

end NUMINAMATH_CALUDE_not_possible_when_70_possible_when_80_l2381_238133


namespace NUMINAMATH_CALUDE_zoo_animals_count_l2381_238135

/-- Represents the number of four-legged birds -/
def num_birds : ℕ := 14

/-- Represents the number of six-legged calves -/
def num_calves : ℕ := 22

/-- The total number of heads -/
def total_heads : ℕ := 36

/-- The total number of legs -/
def total_legs : ℕ := 100

/-- The number of legs each bird has -/
def bird_legs : ℕ := 4

/-- The number of legs each calf has -/
def calf_legs : ℕ := 6

theorem zoo_animals_count :
  (num_birds + num_calves = total_heads) ∧
  (num_birds * bird_legs + num_calves * calf_legs = total_legs) := by
  sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l2381_238135


namespace NUMINAMATH_CALUDE_union_of_I_is_odd_integers_l2381_238185

def I : ℕ → Set ℤ
  | 0 => {-1, 1}
  | n + 1 => {x : ℤ | ∃ y ∈ I n, x^2 - 2*x*y + y^2 = 4^(n + 1)}

def OddIntegers : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 2*k + 1}

theorem union_of_I_is_odd_integers :
  (⋃ n : ℕ, I n) = OddIntegers :=
sorry

end NUMINAMATH_CALUDE_union_of_I_is_odd_integers_l2381_238185


namespace NUMINAMATH_CALUDE_number_divided_by_14_5_equals_173_l2381_238109

theorem number_divided_by_14_5_equals_173 (x : ℝ) : 
  x / 14.5 = 173 → x = 2508.5 := by
sorry

end NUMINAMATH_CALUDE_number_divided_by_14_5_equals_173_l2381_238109


namespace NUMINAMATH_CALUDE_seal_earnings_l2381_238145

/-- Calculates the total earnings of a musician over a given number of years -/
def musician_earnings (songs_per_month : ℕ) (earnings_per_song : ℕ) (years : ℕ) : ℕ :=
  songs_per_month * earnings_per_song * 12 * years

/-- Proves that Seal's earnings over 3 years, given the specified conditions, equal $216,000 -/
theorem seal_earnings : musician_earnings 3 2000 3 = 216000 := by
  sorry

end NUMINAMATH_CALUDE_seal_earnings_l2381_238145


namespace NUMINAMATH_CALUDE_coffee_drinkers_possible_values_l2381_238186

def round_table_coffee_problem (n : ℕ) (coffee_drinkers : ℕ) : Prop :=
  n = 14 ∧
  0 < coffee_drinkers ∧
  coffee_drinkers < n ∧
  ∃ (k : ℕ), k > 0 ∧ k < n/2 ∧ coffee_drinkers = n - 2*k

theorem coffee_drinkers_possible_values :
  ∀ (n : ℕ) (coffee_drinkers : ℕ),
    round_table_coffee_problem n coffee_drinkers →
    coffee_drinkers = 6 ∨ coffee_drinkers = 8 ∨ coffee_drinkers = 10 ∨ coffee_drinkers = 12 :=
by sorry

end NUMINAMATH_CALUDE_coffee_drinkers_possible_values_l2381_238186


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l2381_238182

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 25 ∧ initial_mean = 190 ∧ incorrect_value = 130 ∧ correct_value = 165 →
  (n : ℚ) * initial_mean - incorrect_value + correct_value = n * 191.4 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l2381_238182


namespace NUMINAMATH_CALUDE_target_hit_probability_l2381_238149

theorem target_hit_probability (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2) (h_B : p_B = 1/3) (h_C : p_C = 1/4) :
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l2381_238149


namespace NUMINAMATH_CALUDE_c_investment_determination_l2381_238137

/-- Represents the investment and profit distribution in a shop partnership --/
structure ShopPartnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that given the conditions of the problem, C's investment must be 30,000 --/
theorem c_investment_determination (shop : ShopPartnership)
  (h1 : shop.a_investment = 5000)
  (h2 : shop.b_investment = 15000)
  (h3 : shop.total_profit = 5000)
  (h4 : shop.c_profit_share = 3000)
  (h5 : shop.c_profit_share * (shop.a_investment + shop.b_investment + shop.c_investment) = 
        shop.total_profit * shop.c_investment) :
  shop.c_investment = 30000 := by
  sorry

#check c_investment_determination

end NUMINAMATH_CALUDE_c_investment_determination_l2381_238137


namespace NUMINAMATH_CALUDE_women_on_bus_l2381_238132

theorem women_on_bus (total : ℕ) (men : ℕ) (children : ℕ) 
  (h1 : total = 54)
  (h2 : men = 18)
  (h3 : children = 10) :
  total - men - children = 26 := by
  sorry

end NUMINAMATH_CALUDE_women_on_bus_l2381_238132


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_difference_l2381_238141

theorem consecutive_odd_numbers_difference (x : ℤ) : 
  (x + x + 2 + x + 4 + x + 6 + x + 8) / 5 = 55 → 
  (x + 8) - x = 8 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_difference_l2381_238141


namespace NUMINAMATH_CALUDE_smallest_n_for_factorization_l2381_238114

theorem smallest_n_for_factorization : 
  ∀ n : ℤ, (∃ A B : ℤ, ∀ x, 5*x^2 + n*x + 50 = (5*x + A)*(x + B)) → n ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_factorization_l2381_238114


namespace NUMINAMATH_CALUDE_gcf_lcm_300_105_l2381_238187

theorem gcf_lcm_300_105 : ∃ (gcf lcm : ℕ),
  (Nat.gcd 300 105 = gcf) ∧
  (Nat.lcm 300 105 = lcm) ∧
  (gcf = 15) ∧
  (lcm = 2100) := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_300_105_l2381_238187


namespace NUMINAMATH_CALUDE_cos_45_minus_cos_90_l2381_238127

theorem cos_45_minus_cos_90 : Real.cos (π/4) - Real.cos (π/2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_minus_cos_90_l2381_238127


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2381_238168

/-- Acme T-Shirt Company's pricing function -/
def acme_price (x : ℕ) : ℕ := 60 + 8 * x

/-- Delta T-shirt Company's pricing function -/
def delta_price (x : ℕ) : ℕ := 12 * x

/-- The minimum number of shirts for which Acme is cheaper than Delta -/
def min_shirts_for_acme_cheaper : ℕ := 16

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_for_acme_cheaper < delta_price min_shirts_for_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_for_acme_cheaper →
    acme_price n ≥ delta_price n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2381_238168


namespace NUMINAMATH_CALUDE_root_property_l2381_238190

theorem root_property (a : ℝ) : a^2 - 2*a - 5 = 0 → 2*a^2 - 4*a = 10 := by
  sorry

end NUMINAMATH_CALUDE_root_property_l2381_238190


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l2381_238121

def A (m : ℝ) : Set ℝ := {-1, 3, m^2}
def B (m : ℝ) : Set ℝ := {3, 2*m - 1}

theorem subset_implies_m_values (m : ℝ) : B m ⊆ A m → m = 0 ∨ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l2381_238121


namespace NUMINAMATH_CALUDE_salary_change_percentage_l2381_238183

theorem salary_change_percentage (initial_salary : ℝ) (h : initial_salary > 0) :
  let decreased_salary := initial_salary * (1 - 0.3)
  let final_salary := decreased_salary * (1 + 0.3)
  (initial_salary - final_salary) / initial_salary * 100 = 9 := by
sorry

end NUMINAMATH_CALUDE_salary_change_percentage_l2381_238183


namespace NUMINAMATH_CALUDE_difference_of_numbers_l2381_238178

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 221) : 
  |x - y| = 4 := by sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l2381_238178


namespace NUMINAMATH_CALUDE_power_division_addition_l2381_238116

theorem power_division_addition (a : ℝ) : a^4 / a^2 + a^2 = 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_addition_l2381_238116


namespace NUMINAMATH_CALUDE_john_profit_l2381_238181

/-- Calculate the selling price given the cost price and profit percentage -/
def selling_price (cost : ℚ) (profit_percent : ℚ) : ℚ :=
  cost * (1 + profit_percent / 100)

/-- Calculate the overall profit given the cost and selling prices of two items -/
def overall_profit (cost1 cost2 sell1 sell2 : ℚ) : ℚ :=
  (sell1 + sell2) - (cost1 + cost2)

theorem john_profit :
  let grinder_cost : ℚ := 15000
  let mobile_cost : ℚ := 10000
  let grinder_loss_percent : ℚ := 4
  let mobile_profit_percent : ℚ := 10
  let grinder_sell := selling_price grinder_cost (-grinder_loss_percent)
  let mobile_sell := selling_price mobile_cost mobile_profit_percent
  overall_profit grinder_cost mobile_cost grinder_sell mobile_sell = 400 := by
  sorry

end NUMINAMATH_CALUDE_john_profit_l2381_238181


namespace NUMINAMATH_CALUDE_even_function_problem_l2381_238197

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_problem (f : ℝ → ℝ) 
  (h1 : EvenFunction f) (h2 : f (-5) = 9) : f 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_even_function_problem_l2381_238197


namespace NUMINAMATH_CALUDE_partnership_capital_share_l2381_238102

theorem partnership_capital_share (T : ℝ) (x : ℝ) : 
  (x + 1/4 + 1/5 + (11/20 - x) = 1) →  -- Total shares add up to 1
  (810 / 2430 = x) →                   -- A's profit share equals capital share
  (x = 1/3) :=                         -- A's capital share is 1/3
by sorry

end NUMINAMATH_CALUDE_partnership_capital_share_l2381_238102


namespace NUMINAMATH_CALUDE_chord_length_l2381_238163

/-- The parabola equation y² = 8x with focus F -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (2, 0)

/-- The line passing through F with inclination angle 60° -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sqrt 3 * (p.1 - F.1)}

/-- The chord intercepted by the parabola from the line -/
def Chord : Set (ℝ × ℝ) :=
  Parabola ∩ Line

theorem chord_length : 
  ∃ (A B : ℝ × ℝ), A ∈ Chord ∧ B ∈ Chord ∧ A ≠ B ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32 / 3 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l2381_238163


namespace NUMINAMATH_CALUDE_range_of_m_l2381_238195

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 15 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2*x - m^2 + 1 ≤ 0

-- Define the property that ¬p is a necessary but not sufficient condition for ¬q
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  ∀ x, q x m → p x ∧ ∃ y, ¬(p y) ∧ q y m

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, not_p_necessary_not_sufficient_for_not_q m ↔ (m < -4 ∨ m > 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2381_238195


namespace NUMINAMATH_CALUDE_max_profit_at_70_best_selling_price_l2381_238120

/-- Represents the profit function for a product with given pricing and demand characteristics -/
def profit (x : ℕ) : ℝ :=
  (50 + x - 40) * (50 - x)

/-- Theorem stating that the maximum profit occurs when the selling price is 70 yuan -/
theorem max_profit_at_70 :
  ∀ x : ℕ, x < 50 → x > 0 → profit x ≤ profit 20 :=
sorry

/-- Corollary stating that the best selling price is 70 yuan -/
theorem best_selling_price :
  ∃ x : ℕ, x < 50 ∧ x > 0 ∧ ∀ y : ℕ, y < 50 → y > 0 → profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_70_best_selling_price_l2381_238120


namespace NUMINAMATH_CALUDE_triangle_lines_theorem_l2381_238140

-- Define the triangle vertices
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- Define the line equation type
def LineEquation := ℝ → ℝ → ℝ

-- Define the line AC
def line_AC : LineEquation := fun x y => 3 * x + 4 * y - 12

-- Define the altitude from B to AB
def altitude_B : LineEquation := fun x y => 2 * x + 7 * y - 21

-- Theorem statement
theorem triangle_lines_theorem :
  (∀ x y, line_AC x y = 0 ↔ (x - A.1) * (C.2 - A.2) = (y - A.2) * (C.1 - A.1)) ∧
  (∀ x y, altitude_B x y = 0 ↔ (x - B.1) * (B.1 - A.1) + (y - B.2) * (B.2 - A.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_lines_theorem_l2381_238140


namespace NUMINAMATH_CALUDE_tiffany_phone_pictures_l2381_238122

theorem tiffany_phone_pictures :
  ∀ (phone_pics camera_pics total_pics num_albums pics_per_album : ℕ),
    camera_pics = 13 →
    num_albums = 5 →
    pics_per_album = 4 →
    total_pics = num_albums * pics_per_album →
    total_pics = phone_pics + camera_pics →
    phone_pics = 7 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_phone_pictures_l2381_238122


namespace NUMINAMATH_CALUDE_james_ownership_l2381_238159

theorem james_ownership (total : ℕ) (difference : ℕ) (james : ℕ) (ali : ℕ) :
  total = 250 →
  difference = 40 →
  james = ali + difference →
  total = james + ali →
  james = 145 := by
sorry

end NUMINAMATH_CALUDE_james_ownership_l2381_238159


namespace NUMINAMATH_CALUDE_road_length_difference_l2381_238123

/-- The length of Telegraph Road in kilometers -/
def telegraph_road_length : ℝ := 162

/-- The length of Pardee Road in meters -/
def pardee_road_length : ℝ := 12000

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

theorem road_length_difference :
  (telegraph_road_length * km_to_m - pardee_road_length) / km_to_m = 150 := by
  sorry

end NUMINAMATH_CALUDE_road_length_difference_l2381_238123


namespace NUMINAMATH_CALUDE_calculate_expression_no_solution_inequality_system_l2381_238160

-- Problem 1
theorem calculate_expression : (-2)^3 + |(-4)| - Real.sqrt 9 = -7 := by sorry

-- Problem 2
theorem no_solution_inequality_system :
  ¬∃ x : ℝ, (2*x > 3*x - 2) ∧ (x - 1 > (x + 2) / 3) := by sorry

end NUMINAMATH_CALUDE_calculate_expression_no_solution_inequality_system_l2381_238160


namespace NUMINAMATH_CALUDE_solve_system_l2381_238165

theorem solve_system (x y : ℝ) (h1 : 2 * x + y = 4) (h2 : (x + y) / 3 = 1) :
  x + 2 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2381_238165


namespace NUMINAMATH_CALUDE_roots_sum_greater_than_2a_l2381_238112

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.log x + a / x

theorem roots_sum_greater_than_2a
  (h₁ : x₁ > 0)
  (h₂ : x₂ > 0)
  (h₃ : x₁ ≠ x₂)
  (h₄ : f a x₁ = a / 2)
  (h₅ : f a x₂ = a / 2) :
  x₁ + x₂ > 2 * a :=
sorry

end NUMINAMATH_CALUDE_roots_sum_greater_than_2a_l2381_238112


namespace NUMINAMATH_CALUDE_ski_race_minimum_participants_l2381_238138

theorem ski_race_minimum_participants : ∀ n : ℕ,
  (∃ k : ℕ, 
    (k : ℝ) / n ≥ 0.035 ∧ 
    (k : ℝ) / n ≤ 0.045 ∧ 
    k > 0) →
  n ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_ski_race_minimum_participants_l2381_238138


namespace NUMINAMATH_CALUDE_unique_divisible_by_twelve_l2381_238169

/-- A function that constructs a four-digit number in the form x27x -/
def constructNumber (x : Nat) : Nat :=
  1000 * x + 270 + x

/-- Predicate to check if a number is a single digit -/
def isSingleDigit (n : Nat) : Prop :=
  n ≥ 0 ∧ n ≤ 9

theorem unique_divisible_by_twelve :
  ∃! x : Nat, isSingleDigit x ∧ (constructNumber x) % 12 = 0 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_twelve_l2381_238169


namespace NUMINAMATH_CALUDE_train_passing_platform_l2381_238111

/-- A train passes a platform -/
theorem train_passing_platform
  (l : ℝ) -- length of the train
  (t : ℝ) -- time to pass a pole
  (v : ℝ) -- velocity of the train
  (h1 : t > 0) -- time is positive
  (h2 : l > 0) -- length is positive
  (h3 : v > 0) -- velocity is positive
  (h4 : l = v * t) -- relation between length, velocity, and time for passing a pole
  : v * (4 * t) = l + 3 * l := by sorry

end NUMINAMATH_CALUDE_train_passing_platform_l2381_238111


namespace NUMINAMATH_CALUDE_parabola_coefficient_l2381_238130

def quadratic_function (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_coefficient
  (a b c : ℤ)
  (vertex_x vertex_y : ℝ)
  (point_x point_y : ℝ)
  (h_vertex : ∀ x, quadratic_function a b c x ≥ quadratic_function a b c vertex_x)
  (h_vertex_y : quadratic_function a b c vertex_x = vertex_y)
  (h_point : quadratic_function a b c point_x = point_y)
  (h_vertex_coords : vertex_x = 2 ∧ vertex_y = 3)
  (h_point_coords : point_x = 1 ∧ point_y = 0) :
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l2381_238130


namespace NUMINAMATH_CALUDE_P_on_angle_bisector_PQ_parallel_to_x_axis_l2381_238119

-- Define points P and Q
def P (a : ℝ) : ℝ × ℝ := (a + 1, 2 * a - 3)
def Q : ℝ × ℝ := (2, 3)

-- Theorem for the first condition
theorem P_on_angle_bisector :
  ∃ a : ℝ, P a = (5, 5) ∧ (P a).1 = (P a).2 := by sorry

-- Theorem for the second condition
theorem PQ_parallel_to_x_axis :
  ∃ a : ℝ, (P a).2 = Q.2 → |((P a).1 - Q.1)| = 2 := by sorry

end NUMINAMATH_CALUDE_P_on_angle_bisector_PQ_parallel_to_x_axis_l2381_238119


namespace NUMINAMATH_CALUDE_simple_interest_fraction_l2381_238148

/-- 
Given a principal sum P, proves that the simple interest calculated for 8 years 
at a rate of 2.5% per annum is equal to 1/5 of the principal sum.
-/
theorem simple_interest_fraction (P : ℝ) (P_pos : P > 0) : 
  (P * 2.5 * 8) / 100 = P * (1 / 5) := by
  sorry

#check simple_interest_fraction

end NUMINAMATH_CALUDE_simple_interest_fraction_l2381_238148


namespace NUMINAMATH_CALUDE_graph_is_two_parabolas_l2381_238147

/-- The equation of the conic sections -/
def equation (x y : ℝ) : Prop :=
  y^6 - 9*x^6 = 3*y^3 - 1

/-- A parabola in cubic form -/
def cubic_parabola (x y : ℝ) (a b : ℝ) : Prop :=
  y^3 + a*x^3 = b

/-- The graph consists of two parabolas -/
theorem graph_is_two_parabolas :
  ∃ (a₁ b₁ a₂ b₂ : ℝ), 
    (∀ x y, equation x y ↔ (cubic_parabola x y a₁ b₁ ∨ cubic_parabola x y a₂ b₂)) ∧
    a₁ ≠ a₂ :=
  sorry

end NUMINAMATH_CALUDE_graph_is_two_parabolas_l2381_238147


namespace NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l2381_238152

theorem fifteen_percent_of_600_is_90 :
  ∀ x : ℝ, (15 / 100) * x = 90 → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l2381_238152


namespace NUMINAMATH_CALUDE_log_product_theorem_l2381_238189

theorem log_product_theorem (c d : ℕ+) : 
  (d - c = 450) →
  (Real.log d / Real.log c = 3) →
  (c + d = 520) := by sorry

end NUMINAMATH_CALUDE_log_product_theorem_l2381_238189


namespace NUMINAMATH_CALUDE_real_part_of_reciprocal_difference_l2381_238100

open Complex

theorem real_part_of_reciprocal_difference (w : ℂ) (h1 : w ≠ 0) (h2 : w.im ≠ 0) (h3 : abs w = 2) :
  (1 / (2 - w)).re = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_reciprocal_difference_l2381_238100


namespace NUMINAMATH_CALUDE_curve_self_intersects_l2381_238150

/-- The x-coordinate of a point on the curve given a parameter t -/
def x (t : ℝ) : ℝ := t^2 - 4

/-- The y-coordinate of a point on the curve given a parameter t -/
def y (t : ℝ) : ℝ := t^3 - 6*t + 7

/-- The curve intersects itself if there exist two distinct real numbers that yield the same point -/
def self_intersects : Prop :=
  ∃ a b : ℝ, a ≠ b ∧ x a = x b ∧ y a = y b

/-- The point of self-intersection -/
def intersection_point : ℝ × ℝ := (2, 7)

/-- Theorem stating that the curve intersects itself at (2, 7) -/
theorem curve_self_intersects :
  self_intersects ∧ ∃ a b : ℝ, a ≠ b ∧ x a = (intersection_point.1) ∧ y a = (intersection_point.2) :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersects_l2381_238150


namespace NUMINAMATH_CALUDE_exists_number_with_specific_digit_sums_l2381_238196

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number n such that the sum of its digits is 100
    and the sum of the digits of n^3 is 1,000,000 -/
theorem exists_number_with_specific_digit_sums :
  ∃ n : ℕ, sum_of_digits n = 100 ∧ sum_of_digits (n^3) = 1000000 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_specific_digit_sums_l2381_238196


namespace NUMINAMATH_CALUDE_investment_decrease_l2381_238107

theorem investment_decrease (P : ℝ) (x : ℝ) 
  (h1 : P > 0)
  (h2 : 1.60 * P - (x / 100) * (1.60 * P) = 1.12 * P) :
  x = 30 := by
sorry

end NUMINAMATH_CALUDE_investment_decrease_l2381_238107


namespace NUMINAMATH_CALUDE_sqrt_equation_l2381_238142

theorem sqrt_equation (m n : ℝ) : 
  1.55 * Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2)) - 
  Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2)) = 
  2 * Real.sqrt (3 * m - n) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_l2381_238142


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2381_238131

/-- Two vectors in R² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- Given vectors a = (-6, 3) and b = (2, x), if they are perpendicular, then x = -4 -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (-6, 3)
  let b : ℝ → ℝ × ℝ := fun x ↦ (2, x)
  ∀ x : ℝ, perpendicular a (b x) → x = -4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2381_238131


namespace NUMINAMATH_CALUDE_same_color_probability_l2381_238110

/-- Represents a 20-sided die with a specific color distribution -/
structure Die :=
  (maroon : Nat)
  (teal : Nat)
  (cyan : Nat)
  (sparkly : Nat)
  (total : Nat)
  (valid : maroon + teal + cyan + sparkly = total)

/-- The first die with its color distribution -/
def die1 : Die :=
  { maroon := 5
    teal := 6
    cyan := 7
    sparkly := 2
    total := 20
    valid := by simp }

/-- The second die with its color distribution -/
def die2 : Die :=
  { maroon := 4
    teal := 7
    cyan := 8
    sparkly := 1
    total := 20
    valid := by simp }

/-- Calculates the probability of a specific color on a die -/
def colorProbability (d : Die) (color : Nat) : Rat :=
  color / d.total

/-- Calculates the probability of both dice showing the same color -/
def sameProbability (d1 d2 : Die) : Rat :=
  (colorProbability d1 d1.maroon * colorProbability d2 d2.maroon) +
  (colorProbability d1 d1.teal * colorProbability d2 d2.teal) +
  (colorProbability d1 d1.cyan * colorProbability d2 d2.cyan) +
  (colorProbability d1 d1.sparkly * colorProbability d2 d2.sparkly)

theorem same_color_probability :
  sameProbability die1 die2 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2381_238110


namespace NUMINAMATH_CALUDE_triangle_area_l2381_238193

theorem triangle_area (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c = a + b - 12) 
  (h4 : (a + b + c) / 2 = 21) (h5 : c - a = 2) : 
  Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2381_238193


namespace NUMINAMATH_CALUDE_ratio_equality_implies_sum_ratio_l2381_238104

theorem ratio_equality_implies_sum_ratio (x y z : ℝ) :
  x / 3 = y / (-4) ∧ y / (-4) = z / 7 →
  (3 * x + y + z) / y = -3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_implies_sum_ratio_l2381_238104


namespace NUMINAMATH_CALUDE_complex_modulus_l2381_238153

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2381_238153


namespace NUMINAMATH_CALUDE_shift_down_two_units_l2381_238173

def f (x : ℝ) : ℝ := 2 * x + 1

def g (x : ℝ) : ℝ := 2 * x - 1

def vertical_shift (h : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x => h x - shift

theorem shift_down_two_units :
  vertical_shift f 2 = g :=
sorry

end NUMINAMATH_CALUDE_shift_down_two_units_l2381_238173


namespace NUMINAMATH_CALUDE_petya_wins_l2381_238161

/-- Represents the possible moves in the game -/
inductive Move
  | changeOneToPlus
  | eraseOnePlusOneMinus
  | changeTwoToThreePlus

/-- The game state -/
structure GameState where
  minuses : ℕ
  pluses : ℕ

/-- Apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.changeOneToPlus => ⟨state.minuses - 1, state.pluses + 1⟩
  | Move.eraseOnePlusOneMinus => ⟨state.minuses - 1, state.pluses - 1⟩
  | Move.changeTwoToThreePlus => ⟨state.minuses - 2, state.pluses + 3⟩

/-- Check if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  match move with
  | Move.changeOneToPlus => state.minuses ≥ 1
  | Move.eraseOnePlusOneMinus => state.minuses ≥ 1 ∧ state.pluses ≥ 1
  | Move.changeTwoToThreePlus => state.minuses ≥ 2

/-- Check if the game is over (no valid moves left) -/
def isGameOver (state : GameState) : Prop :=
  ¬(isValidMove state Move.changeOneToPlus ∨
    isValidMove state Move.eraseOnePlusOneMinus ∨
    isValidMove state Move.changeTwoToThreePlus)

/-- The winning strategy for the first player -/
def hasWinningStrategy (initialMinuses : ℕ) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (game : List Move),
      let finalState := game.foldl applyMove ⟨initialMinuses, 0⟩
      isGameOver finalState →
      game.length % 2 = 1

/-- Theorem: The first player (Petya) has a winning strategy when starting with 2021 minuses -/
theorem petya_wins : hasWinningStrategy 2021 := by
  sorry

end NUMINAMATH_CALUDE_petya_wins_l2381_238161


namespace NUMINAMATH_CALUDE_field_pond_area_ratio_l2381_238166

/-- Given a rectangular field and a square pond, prove the ratio of their areas -/
theorem field_pond_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
  field_length = 2 * field_width →
  field_length = 32 →
  pond_side = 8 →
  (pond_side^2) / (field_length * field_width) = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_field_pond_area_ratio_l2381_238166


namespace NUMINAMATH_CALUDE_average_temperature_l2381_238156

def temperatures : List ℝ := [52, 62, 55, 59, 50]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 55.6 := by sorry

end NUMINAMATH_CALUDE_average_temperature_l2381_238156
