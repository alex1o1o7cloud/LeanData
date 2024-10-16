import Mathlib

namespace NUMINAMATH_CALUDE_max_value_trig_function_l769_76938

theorem max_value_trig_function :
  (∀ x : ℝ, 3 * Real.sin x - 3 * Real.sqrt 3 * Real.cos x ≤ 6) ∧
  (∃ x : ℝ, 3 * Real.sin x - 3 * Real.sqrt 3 * Real.cos x = 6) := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_function_l769_76938


namespace NUMINAMATH_CALUDE_eliza_almonds_l769_76961

theorem eliza_almonds (eliza_almonds daniel_almonds : ℕ) : 
  eliza_almonds = daniel_almonds + 8 →
  daniel_almonds = eliza_almonds / 3 →
  eliza_almonds = 12 := by
sorry

end NUMINAMATH_CALUDE_eliza_almonds_l769_76961


namespace NUMINAMATH_CALUDE_job_completion_time_l769_76951

theorem job_completion_time (job : ℝ) (days_A : ℝ) (efficiency_C : ℝ) : 
  job > 0 → days_A > 0 → efficiency_C > 0 →
  (job / days_A) * efficiency_C * 16 = job :=
by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_job_completion_time_l769_76951


namespace NUMINAMATH_CALUDE_min_value_theorem_l769_76944

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  b / a + 1 / b ≥ 3 ∧ (b / a + 1 / b = 3 ↔ a = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l769_76944


namespace NUMINAMATH_CALUDE_range_of_a_l769_76940

-- Define propositions A and B
def PropA (x : ℝ) : Prop := (x - 1)^2 < 9
def PropB (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Define the set of x satisfying proposition A
def SetA : Set ℝ := {x | PropA x}

-- Define the set of x satisfying proposition B for a given a
def SetB (a : ℝ) : Set ℝ := {x | PropB x a}

-- Define the condition that A is sufficient but not necessary for B
def ASufficientNotNecessary (a : ℝ) : Prop :=
  SetA ⊂ SetB a ∧ SetA ≠ SetB a

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, ASufficientNotNecessary a ↔ a < -4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l769_76940


namespace NUMINAMATH_CALUDE_no_triangle_from_divisibility_conditions_l769_76905

theorem no_triangle_from_divisibility_conditions (a b c : ℕ+) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 →
  a.val ∣ (b.val - c.val)^2 →
  b.val ∣ (a.val - c.val)^2 →
  c.val ∣ (a.val - b.val)^2 →
  ¬(a.val + b.val > c.val ∧ b.val + c.val > a.val ∧ c.val + a.val > b.val) := by
sorry

end NUMINAMATH_CALUDE_no_triangle_from_divisibility_conditions_l769_76905


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l769_76973

theorem number_of_boys_in_class (num_girls : ℕ) (avg_boys : ℚ) (avg_girls : ℚ) (avg_class : ℚ) :
  num_girls = 4 ∧ avg_boys = 84 ∧ avg_girls = 92 ∧ avg_class = 86 →
  ∃ (num_boys : ℕ), num_boys = 12 ∧
    (avg_boys * num_boys + avg_girls * num_girls) / (num_boys + num_girls) = avg_class :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l769_76973


namespace NUMINAMATH_CALUDE_weightlifter_total_lift_l769_76966

/-- The weight a weightlifter can lift in one hand -/
def weight_per_hand : ℕ := 7

/-- The number of hands a weightlifter has -/
def number_of_hands : ℕ := 2

/-- The total weight a weightlifter can lift at once -/
def total_weight : ℕ := weight_per_hand * number_of_hands

/-- Theorem: The total weight a weightlifter can lift at once is 14 pounds -/
theorem weightlifter_total_lift : total_weight = 14 := by
  sorry

end NUMINAMATH_CALUDE_weightlifter_total_lift_l769_76966


namespace NUMINAMATH_CALUDE_midpoint_coordinate_product_l769_76969

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (5, -2) and (-3, 6) is equal to 2. -/
theorem midpoint_coordinate_product : 
  let x₁ : ℝ := 5
  let y₁ : ℝ := -2
  let x₂ : ℝ := -3
  let y₂ : ℝ := 6
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x * midpoint_y = 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_product_l769_76969


namespace NUMINAMATH_CALUDE_total_revenue_calculation_l769_76980

/-- Calculate the total revenue from vegetable sales --/
theorem total_revenue_calculation :
  let morning_potatoes : ℕ := 29
  let morning_onions : ℕ := 15
  let morning_carrots : ℕ := 12
  let afternoon_potatoes : ℕ := 17
  let afternoon_onions : ℕ := 22
  let afternoon_carrots : ℕ := 9
  let potato_weight : ℕ := 7
  let onion_weight : ℕ := 5
  let carrot_weight : ℕ := 4
  let potato_price : ℚ := 1.75
  let onion_price : ℚ := 2.50
  let carrot_price : ℚ := 3.25

  let total_potatoes : ℕ := morning_potatoes + afternoon_potatoes
  let total_onions : ℕ := morning_onions + afternoon_onions
  let total_carrots : ℕ := morning_carrots + afternoon_carrots

  let potato_revenue : ℚ := (total_potatoes * potato_weight : ℚ) * potato_price
  let onion_revenue : ℚ := (total_onions * onion_weight : ℚ) * onion_price
  let carrot_revenue : ℚ := (total_carrots * carrot_weight : ℚ) * carrot_price

  let total_revenue : ℚ := potato_revenue + onion_revenue + carrot_revenue

  total_revenue = 1299.00 := by sorry

end NUMINAMATH_CALUDE_total_revenue_calculation_l769_76980


namespace NUMINAMATH_CALUDE_not_in_range_quadratic_l769_76977

theorem not_in_range_quadratic (b : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + 3 ≠ -3) ↔ -Real.sqrt 24 < b ∧ b < Real.sqrt 24 := by
  sorry

end NUMINAMATH_CALUDE_not_in_range_quadratic_l769_76977


namespace NUMINAMATH_CALUDE_matching_pair_probability_l769_76979

def black_socks : ℕ := 12
def blue_socks : ℕ := 10
def total_socks : ℕ := black_socks + blue_socks

def matching_pairs : ℕ := (black_socks * (black_socks - 1)) / 2 + (blue_socks * (blue_socks - 1)) / 2
def total_combinations : ℕ := (total_socks * (total_socks - 1)) / 2

theorem matching_pair_probability :
  (matching_pairs : ℚ) / total_combinations = 111 / 231 := by sorry

end NUMINAMATH_CALUDE_matching_pair_probability_l769_76979


namespace NUMINAMATH_CALUDE_competitive_examination_selection_l769_76933

theorem competitive_examination_selection (total_candidates : ℕ) 
  (selection_rate_A : ℚ) (selection_rate_B : ℚ) : 
  total_candidates = 8100 → 
  selection_rate_A = 6 / 100 → 
  selection_rate_B = 7 / 100 → 
  (selection_rate_B - selection_rate_A) * total_candidates = 81 := by
  sorry

end NUMINAMATH_CALUDE_competitive_examination_selection_l769_76933


namespace NUMINAMATH_CALUDE_samantha_last_name_has_seven_letters_l769_76999

/-- The number of letters in Jamie's last name -/
def jamie_last_name_length : ℕ := 4

/-- The number of letters in Bobbie's last name -/
def bobbie_last_name_length : ℕ := jamie_last_name_length * 2 + 2

/-- The number of letters in Samantha's last name -/
def samantha_last_name_length : ℕ := bobbie_last_name_length - 3

theorem samantha_last_name_has_seven_letters :
  samantha_last_name_length = 7 := by
  sorry

end NUMINAMATH_CALUDE_samantha_last_name_has_seven_letters_l769_76999


namespace NUMINAMATH_CALUDE_power_of_four_exponent_l769_76926

theorem power_of_four_exponent (n : ℕ) (x : ℕ) 
  (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (hn : n = 25) : x = 26 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_exponent_l769_76926


namespace NUMINAMATH_CALUDE_cone_unfolded_side_view_is_sector_l769_76929

/-- A shape with one curved side and two straight sides -/
structure ConeUnfoldedSideView where
  curved_side : ℕ
  straight_sides : ℕ
  h_curved : curved_side = 1
  h_straight : straight_sides = 2

/-- Definition of a sector -/
def is_sector (shape : ConeUnfoldedSideView) : Prop :=
  shape.curved_side = 1 ∧ shape.straight_sides = 2

/-- Theorem: The unfolded side view of a cone is a sector -/
theorem cone_unfolded_side_view_is_sector (shape : ConeUnfoldedSideView) :
  is_sector shape :=
by sorry

end NUMINAMATH_CALUDE_cone_unfolded_side_view_is_sector_l769_76929


namespace NUMINAMATH_CALUDE_yanna_apples_l769_76954

theorem yanna_apples (apples_to_zenny apples_to_andrea apples_kept : ℕ) : 
  apples_to_zenny = 18 → 
  apples_to_andrea = 6 → 
  apples_kept = 36 → 
  apples_to_zenny + apples_to_andrea + apples_kept = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_yanna_apples_l769_76954


namespace NUMINAMATH_CALUDE_jet_bar_sales_difference_l769_76984

def weekly_target : ℕ := 90
def monday_sales : ℕ := 45
def remaining_sales : ℕ := 16

theorem jet_bar_sales_difference : 
  monday_sales - (weekly_target - remaining_sales - monday_sales) = 16 := by
  sorry

end NUMINAMATH_CALUDE_jet_bar_sales_difference_l769_76984


namespace NUMINAMATH_CALUDE_smallest_benches_proof_l769_76960

/-- The number of adults that can sit on one bench -/
def adults_per_bench : ℕ := 7

/-- The number of children that can sit on one bench -/
def children_per_bench : ℕ := 11

/-- A function that returns true if the given number of benches can seat an equal number of adults and children -/
def can_seat_equally (n : ℕ) : Prop :=
  ∃ (people : ℕ), people > 0 ∧ 
    n * adults_per_bench = people ∧
    n * children_per_bench = people

/-- The smallest number of benches that can seat an equal number of adults and children -/
def smallest_n : ℕ := 18

theorem smallest_benches_proof :
  (∀ m : ℕ, m > 0 → m < smallest_n → ¬(can_seat_equally m)) ∧
  can_seat_equally smallest_n :=
sorry

end NUMINAMATH_CALUDE_smallest_benches_proof_l769_76960


namespace NUMINAMATH_CALUDE_same_solutions_imply_coefficients_l769_76909

-- Define the absolute value equation
def abs_equation (x : ℝ) : Prop := |x - 3| = 4

-- Define the quadratic equation
def quadratic_equation (x b c : ℝ) : Prop := x^2 + b*x + c = 0

-- Theorem statement
theorem same_solutions_imply_coefficients :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ abs_equation x₁ ∧ abs_equation x₂) →
  (∀ x : ℝ, abs_equation x ↔ ∃ b c : ℝ, quadratic_equation x b c) →
  ∃! (b c : ℝ), ∀ x : ℝ, abs_equation x ↔ quadratic_equation x b c :=
by sorry

end NUMINAMATH_CALUDE_same_solutions_imply_coefficients_l769_76909


namespace NUMINAMATH_CALUDE_custom_op_solution_l769_76901

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- State the theorem
theorem custom_op_solution :
  ∀ y : ℤ, customOp y 10 = 90 → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l769_76901


namespace NUMINAMATH_CALUDE_joshs_initial_money_l769_76981

theorem joshs_initial_money (hat_cost pencil_cost cookie_cost : ℚ)
  (num_cookies : ℕ) (money_left : ℚ) :
  hat_cost = 10 →
  pencil_cost = 2 →
  cookie_cost = 5/4 →
  num_cookies = 4 →
  money_left = 3 →
  hat_cost + pencil_cost + num_cookies * cookie_cost + money_left = 20 :=
by sorry

end NUMINAMATH_CALUDE_joshs_initial_money_l769_76981


namespace NUMINAMATH_CALUDE_three_power_fraction_equals_41_40_l769_76968

theorem three_power_fraction_equals_41_40 :
  (3^1008 + 3^1004) / (3^1008 - 3^1004) = 41/40 := by
  sorry

end NUMINAMATH_CALUDE_three_power_fraction_equals_41_40_l769_76968


namespace NUMINAMATH_CALUDE_simple_interest_rate_l769_76978

/-- Calculates the simple interest rate given loan amounts, durations, and total interest received. -/
theorem simple_interest_rate 
  (loan_b loan_c : ℕ) 
  (duration_b duration_c : ℕ) 
  (total_interest : ℕ) : 
  loan_b = 5000 → 
  loan_c = 3000 → 
  duration_b = 2 → 
  duration_c = 4 → 
  total_interest = 1540 → 
  ∃ (rate : ℚ), 
    rate = 7 ∧ 
    (loan_b * duration_b * rate + loan_c * duration_c * rate) / 100 = total_interest :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l769_76978


namespace NUMINAMATH_CALUDE_exists_surjective_function_with_property_l769_76945

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x else x - 1

-- State the theorem
theorem exists_surjective_function_with_property :
  ∃ (f : ℝ → ℝ), Function.Surjective f ∧
  (∀ x y : ℝ, (f (x + y) - f x - f y) ∈ ({0, 1} : Set ℝ)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_exists_surjective_function_with_property_l769_76945


namespace NUMINAMATH_CALUDE_three_round_layoffs_result_l769_76991

def layoff_round (employees : ℕ) : ℕ :=
  (employees * 10) / 100

def remaining_after_layoff (employees : ℕ) : ℕ :=
  employees - layoff_round employees

def total_layoffs (initial_employees : ℕ) : ℕ :=
  let first_round := layoff_round initial_employees
  let after_first := remaining_after_layoff initial_employees
  let second_round := layoff_round after_first
  let after_second := remaining_after_layoff after_first
  let third_round := layoff_round after_second
  first_round + second_round + third_round

theorem three_round_layoffs_result :
  total_layoffs 1000 = 271 :=
sorry

end NUMINAMATH_CALUDE_three_round_layoffs_result_l769_76991


namespace NUMINAMATH_CALUDE_book_price_calculation_l769_76995

def initial_price : ℝ := 250

def week1_decrease : ℝ := 0.125
def week1_increase : ℝ := 0.30
def week2_decrease : ℝ := 0.20
def week3_increase : ℝ := 0.50

def conversion_rate : ℝ := 3
def sales_tax_rate : ℝ := 0.05

def price_after_fluctuations : ℝ :=
  initial_price * (1 - week1_decrease) * (1 + week1_increase) * (1 - week2_decrease) * (1 + week3_increase)

def price_in_currency_b : ℝ := price_after_fluctuations * conversion_rate

def final_price : ℝ := price_in_currency_b * (1 + sales_tax_rate)

theorem book_price_calculation :
  final_price = 1074.9375 := by sorry

end NUMINAMATH_CALUDE_book_price_calculation_l769_76995


namespace NUMINAMATH_CALUDE_move_right_three_units_l769_76931

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moves a point horizontally by a given distance -/
def moveHorizontal (p : Point) (distance : ℝ) : Point :=
  { x := p.x + distance, y := p.y }

theorem move_right_three_units :
  let P : Point := { x := -2, y := -3 }
  let Q : Point := moveHorizontal P 3
  Q.x = 1 ∧ Q.y = -3 := by
  sorry

end NUMINAMATH_CALUDE_move_right_three_units_l769_76931


namespace NUMINAMATH_CALUDE_remaining_volume_cube_with_cylindrical_hole_l769_76924

/-- The remaining volume of a cube after drilling a cylindrical hole -/
theorem remaining_volume_cube_with_cylindrical_hole :
  let cube_side : ℝ := 6
  let hole_radius : ℝ := 3
  let hole_height : ℝ := 6
  let cube_volume : ℝ := cube_side ^ 3
  let cylinder_volume : ℝ := π * hole_radius ^ 2 * hole_height
  let remaining_volume : ℝ := cube_volume - cylinder_volume
  remaining_volume = 216 - 54 * π := by
  sorry


end NUMINAMATH_CALUDE_remaining_volume_cube_with_cylindrical_hole_l769_76924


namespace NUMINAMATH_CALUDE_systematic_sampling_l769_76982

theorem systematic_sampling 
  (total_employees : Nat) 
  (sample_size : Nat) 
  (fifth_sample : Nat) :
  total_employees = 200 →
  sample_size = 40 →
  fifth_sample = 23 →
  ∃ (start : Nat), 
    start + 4 * (total_employees / sample_size) = fifth_sample ∧
    start + 7 * (total_employees / sample_size) = 38 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_l769_76982


namespace NUMINAMATH_CALUDE_factor_expression_l769_76943

theorem factor_expression (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l769_76943


namespace NUMINAMATH_CALUDE_complex_cube_root_unity_l769_76925

theorem complex_cube_root_unity (i : ℂ) (y : ℂ) :
  i^2 = -1 →
  y = (1 + i * Real.sqrt 3) / 2 →
  1 / (y^3 - y) = -1/2 + (i * Real.sqrt 3) / 6 := by sorry

end NUMINAMATH_CALUDE_complex_cube_root_unity_l769_76925


namespace NUMINAMATH_CALUDE_plains_total_area_l769_76957

/-- The total area of two plains, given their individual areas. -/
def total_area (area_A area_B : ℝ) : ℝ := area_A + area_B

/-- The theorem stating the total area of two plains. -/
theorem plains_total_area :
  ∀ (area_A area_B : ℝ),
  area_B = 200 →
  area_A = area_B - 50 →
  total_area area_A area_B = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_plains_total_area_l769_76957


namespace NUMINAMATH_CALUDE_exam_marks_calculation_l769_76908

theorem exam_marks_calculation (T : ℕ) : 
  (T * 20 / 100 + 40 = 160) → 
  (T * 30 / 100 - 160 = 20) := by
  sorry

end NUMINAMATH_CALUDE_exam_marks_calculation_l769_76908


namespace NUMINAMATH_CALUDE_unpainted_area_crossed_boards_l769_76934

/-- The area of the unpainted region when two boards cross -/
theorem unpainted_area_crossed_boards (width1 width2 : ℝ) (angle : ℝ) :
  width1 = 5 →
  width2 = 8 →
  angle = π / 4 →
  (width1 * width2 * Real.sqrt 2) / 2 = 40 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_unpainted_area_crossed_boards_l769_76934


namespace NUMINAMATH_CALUDE_train_passing_platform_l769_76906

/-- A train passes a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 600) 
  (h2 : tree_crossing_time = 60) 
  (h3 : platform_length = 450) : 
  (train_length + platform_length) / (train_length / tree_crossing_time) = 105 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_platform_l769_76906


namespace NUMINAMATH_CALUDE_janet_freelance_income_difference_l769_76922

/-- Calculates how much more Janet would make per month as a freelancer compared to her current job -/
theorem janet_freelance_income_difference :
  let hours_per_week : ℕ := 40
  let weeks_per_month : ℕ := 4
  let current_hourly_rate : ℚ := 30
  let freelance_hourly_rate : ℚ := 40
  let extra_fica_per_week : ℚ := 25
  let healthcare_premium_per_month : ℚ := 400

  let current_monthly_income := (hours_per_week * weeks_per_month : ℚ) * current_hourly_rate
  let freelance_monthly_income := (hours_per_week * weeks_per_month : ℚ) * freelance_hourly_rate
  let extra_costs_per_month := extra_fica_per_week * weeks_per_month + healthcare_premium_per_month

  freelance_monthly_income - extra_costs_per_month - current_monthly_income = 1100 :=
by sorry

end NUMINAMATH_CALUDE_janet_freelance_income_difference_l769_76922


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l769_76965

theorem product_of_repeating_decimal_and_eight :
  let t : ℚ := 456 / 999
  t * 8 = 48 / 13 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l769_76965


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l769_76927

/-- A quadratic equation with parameter m -/
def quadratic_equation (m : ℤ) (x : ℤ) : Prop :=
  m * x^2 - (m + 1) * x + 1 = 0

/-- The property that the equation has two distinct integer roots -/
def has_two_distinct_integer_roots (m : ℤ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

theorem quadratic_equation_solution :
  ∀ m : ℤ, has_two_distinct_integer_roots m → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l769_76927


namespace NUMINAMATH_CALUDE_trampoline_jumps_l769_76962

theorem trampoline_jumps (ronald_jumps rupert_extra_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : rupert_extra_jumps = 86) : 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 := by
  sorry

end NUMINAMATH_CALUDE_trampoline_jumps_l769_76962


namespace NUMINAMATH_CALUDE_sum_bottle_caps_l769_76916

/-- The number of bottle caps for each child -/
def bottle_caps : Fin 9 → ℕ
  | ⟨0, _⟩ => 5
  | ⟨1, _⟩ => 8
  | ⟨2, _⟩ => 12
  | ⟨3, _⟩ => 7
  | ⟨4, _⟩ => 9
  | ⟨5, _⟩ => 10
  | ⟨6, _⟩ => 15
  | ⟨7, _⟩ => 4
  | ⟨8, _⟩ => 11
  | ⟨n+9, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 9 n))

/-- The theorem stating that the sum of bottle caps is 81 -/
theorem sum_bottle_caps : (Finset.univ.sum bottle_caps) = 81 := by
  sorry

end NUMINAMATH_CALUDE_sum_bottle_caps_l769_76916


namespace NUMINAMATH_CALUDE_different_genre_pairs_count_l769_76921

/-- Represents the number of books in each genre -/
structure BookCollection where
  mystery : Nat
  fantasy : Nat
  biography : Nat

/-- Calculates the number of possible pairs of books from different genres -/
def differentGenrePairs (books : BookCollection) : Nat :=
  books.mystery * books.fantasy +
  books.mystery * books.biography +
  books.fantasy * books.biography

/-- Theorem: Given 4 mystery novels, 3 fantasy novels, and 2 biographies,
    the number of possible pairs of books from different genres is 26 -/
theorem different_genre_pairs_count :
  differentGenrePairs ⟨4, 3, 2⟩ = 26 := by
  sorry

end NUMINAMATH_CALUDE_different_genre_pairs_count_l769_76921


namespace NUMINAMATH_CALUDE_intersection_length_l769_76949

/-- The length of segment AB is 8 when a line y = kx - k intersects 
    the parabola y² = 4x at points A and B, and the distance from 
    the midpoint of segment AB to the y-axis is 3 -/
theorem intersection_length (k : ℝ) (A B : ℝ × ℝ) : 
  (∃ (x y : ℝ), y = k * x - k ∧ y^2 = 4 * x) →  -- line intersects parabola
  (A.1 + B.1) / 2 = 3 →                         -- midpoint x-coordinate is 3
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    A = (x₁, y₁) ∧ 
    B = (x₂, y₂) ∧ 
    y₁ = k * x₁ - k ∧ 
    y₁^2 = 4 * x₁ ∧ 
    y₂ = k * x₂ - k ∧ 
    y₂^2 = 4 * x₂ ∧
    ((x₂ - x₁)^2 + (y₂ - y₁)^2).sqrt = 8 :=
by sorry

end NUMINAMATH_CALUDE_intersection_length_l769_76949


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l769_76952

theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∀ n m : ℕ, a (n + m) = a n * a m) :
  a 6 = 6 → a 9 = 9 → a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l769_76952


namespace NUMINAMATH_CALUDE_meat_market_sales_ratio_l769_76964

/-- Given the sales data for a Meat Market over four days, prove the ratio of Sunday to Saturday sales --/
theorem meat_market_sales_ratio :
  let thursday_sales : ℕ := 210
  let friday_sales : ℕ := 2 * thursday_sales
  let saturday_sales : ℕ := 130
  let planned_total : ℕ := 500
  let actual_total : ℕ := planned_total + 325
  let sunday_sales : ℕ := actual_total - (thursday_sales + friday_sales + saturday_sales)
  (sunday_sales : ℚ) / saturday_sales = 1 / 2 := by
  sorry

#check meat_market_sales_ratio

end NUMINAMATH_CALUDE_meat_market_sales_ratio_l769_76964


namespace NUMINAMATH_CALUDE_cookies_sold_in_morning_l769_76996

/-- Proves the number of cookies sold in the morning given the total cookies,
    cookies sold during lunch and afternoon, and cookies left at the end of the day. -/
theorem cookies_sold_in_morning 
  (total : ℕ) 
  (lunch_sold : ℕ) 
  (afternoon_sold : ℕ) 
  (left_at_end : ℕ) 
  (h1 : total = 120) 
  (h2 : lunch_sold = 57) 
  (h3 : afternoon_sold = 16) 
  (h4 : left_at_end = 11) : 
  total - lunch_sold - afternoon_sold - left_at_end = 36 := by
  sorry

end NUMINAMATH_CALUDE_cookies_sold_in_morning_l769_76996


namespace NUMINAMATH_CALUDE_highway_vehicles_l769_76911

theorem highway_vehicles (total : ℕ) (trucks : ℕ) (cars : ℕ) 
  (h1 : total = 300)
  (h2 : cars = 2 * trucks)
  (h3 : total = cars + trucks) :
  trucks = 100 := by
  sorry

end NUMINAMATH_CALUDE_highway_vehicles_l769_76911


namespace NUMINAMATH_CALUDE_no_function_satisfies_conditions_l769_76990

/-- A function satisfying the given conditions in the problem -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∃ M : ℝ, ∀ x, |f x| ≤ M) ∧ 
  f 1 = 1 ∧
  ∀ x ≠ 0, f (x + 1/x^2) = f x + (f (1/x))^2

/-- Theorem stating that no function satisfies all the given conditions -/
theorem no_function_satisfies_conditions : ¬∃ f : ℝ → ℝ, SatisfiesConditions f := by
  sorry


end NUMINAMATH_CALUDE_no_function_satisfies_conditions_l769_76990


namespace NUMINAMATH_CALUDE_sum_a1_a5_l769_76976

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = n² + a₁/2,
    prove that a₁ + a₅ = 11 -/
theorem sum_a1_a5 (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, n > 0 → S n = n^2 + a 1 / 2) : 
    a 1 + a 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_a1_a5_l769_76976


namespace NUMINAMATH_CALUDE_paulson_spending_percentage_l769_76941

theorem paulson_spending_percentage 
  (income : ℝ) 
  (expenditure : ℝ) 
  (savings : ℝ) 
  (h1 : expenditure + savings = income) 
  (h2 : 1.2 * income - 1.1 * expenditure = 1.5 * savings) : 
  expenditure = 0.75 * income :=
sorry

end NUMINAMATH_CALUDE_paulson_spending_percentage_l769_76941


namespace NUMINAMATH_CALUDE_tina_change_probability_l769_76988

/-- The number of toys in the machine -/
def num_toys : ℕ := 10

/-- The cost of the most expensive toy in cents -/
def max_cost : ℕ := 450

/-- The cost difference between consecutive toys in cents -/
def cost_diff : ℕ := 50

/-- The number of quarters Tina starts with -/
def initial_quarters : ℕ := 12

/-- The cost of Tina's favorite toy in cents -/
def favorite_toy_cost : ℕ := 400

/-- The probability that Tina needs to get change for her twenty-dollar bill -/
def change_probability : ℚ := 999802 / 1000000

theorem tina_change_probability :
  (1 : ℚ) - (Nat.factorial (num_toys - 4) : ℚ) / (Nat.factorial num_toys : ℚ) = change_probability :=
sorry

end NUMINAMATH_CALUDE_tina_change_probability_l769_76988


namespace NUMINAMATH_CALUDE_polynomial_division_l769_76902

-- Define the theorem
theorem polynomial_division (a b : ℝ) (h : b ≠ 2*a) : 
  (4*a^2 - b^2) / (b - 2*a) = -2*a - b :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_l769_76902


namespace NUMINAMATH_CALUDE_sqrt_65_bound_l769_76975

theorem sqrt_65_bound (n : ℕ+) (h : (n : ℝ) < Real.sqrt 65 ∧ Real.sqrt 65 < (n : ℝ) + 1) : n = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_65_bound_l769_76975


namespace NUMINAMATH_CALUDE_odd_prime_properties_l769_76923

theorem odd_prime_properties (p n : ℕ) (hp : Nat.Prime p) (hodd : Odd p) (hform : p = 4 * n + 1) :
  (∃ (x : ℕ), x ^ 2 % p = n % p) ∧ (n ^ n % p = 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_properties_l769_76923


namespace NUMINAMATH_CALUDE_distribute_five_to_three_l769_76920

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    with each box containing at least one object. -/
def distributeObjects (n k : ℕ) : ℕ :=
  sorry

theorem distribute_five_to_three :
  distributeObjects 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_to_three_l769_76920


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l769_76974

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l769_76974


namespace NUMINAMATH_CALUDE_gcd_102_238_l769_76956

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l769_76956


namespace NUMINAMATH_CALUDE_shirt_markup_percentage_l769_76970

theorem shirt_markup_percentage (initial_markup : ℝ) (initial_price : ℝ) (price_increase : ℝ) : 
  initial_markup = 0.8 →
  initial_price = 45 →
  price_increase = 5 →
  let wholesale_price := initial_price / (1 + initial_markup)
  let new_price := initial_price + price_increase
  let markup_percentage := (new_price - wholesale_price) / wholesale_price * 100
  markup_percentage = 100 := by
sorry

end NUMINAMATH_CALUDE_shirt_markup_percentage_l769_76970


namespace NUMINAMATH_CALUDE_q_polynomial_form_l769_76992

theorem q_polynomial_form (q : ℝ → ℝ) :
  (∀ x, q x + (2*x^6 + 4*x^4 + 10*x^2) = (5*x^4 + 15*x^3 + 30*x^2 + 10*x + 10)) →
  (∀ x, q x = -2*x^6 + x^4 + 15*x^3 + 20*x^2 + 10*x + 10) := by
sorry

end NUMINAMATH_CALUDE_q_polynomial_form_l769_76992


namespace NUMINAMATH_CALUDE_max_sum_with_constraints_l769_76903

theorem max_sum_with_constraints (x y : ℝ) 
  (h1 : 3 * x + 2 * y ≤ 7) 
  (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 11 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_with_constraints_l769_76903


namespace NUMINAMATH_CALUDE_successfully_served_pizzas_l769_76915

def pizzas_served : ℕ := 9
def pizzas_returned : ℕ := 6

theorem successfully_served_pizzas : 
  pizzas_served - pizzas_returned = 3 := by sorry

end NUMINAMATH_CALUDE_successfully_served_pizzas_l769_76915


namespace NUMINAMATH_CALUDE_jordons_machine_l769_76997

theorem jordons_machine (x : ℝ) : 2 * x + 3 = 27 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_jordons_machine_l769_76997


namespace NUMINAMATH_CALUDE_hexagon_angle_measure_l769_76914

/-- A hexagon with vertices N, U, M, B, E, S -/
structure Hexagon :=
  (N U M B E S : ℝ)

/-- The property that three angles are congruent -/
def three_angles_congruent (h : Hexagon) : Prop :=
  h.N = h.M ∧ h.M = h.B

/-- The property that two angles are supplementary -/
def supplementary (a b : ℝ) : Prop :=
  a + b = 180

/-- The theorem stating that in a hexagon NUMBERS where ∠N ≅ ∠M ≅ ∠B 
    and ∠U is supplementary to ∠S, the measure of ∠B is 135° -/
theorem hexagon_angle_measure (h : Hexagon) 
  (h_congruent : three_angles_congruent h)
  (h_supplementary : supplementary h.U h.S) :
  h.B = 135 :=
sorry

end NUMINAMATH_CALUDE_hexagon_angle_measure_l769_76914


namespace NUMINAMATH_CALUDE_aisha_shopping_money_l769_76936

theorem aisha_shopping_money (initial_money : ℝ) : 
  let after_first := initial_money - (0.4 * initial_money + 4)
  let after_second := after_first - (0.5 * after_first + 5)
  let after_third := after_second - (0.6 * after_second + 6)
  after_third = 2 → initial_money = 90 := by
sorry

end NUMINAMATH_CALUDE_aisha_shopping_money_l769_76936


namespace NUMINAMATH_CALUDE_gcd_87654321_12345678_l769_76904

theorem gcd_87654321_12345678 : Nat.gcd 87654321 12345678 = 75 := by
  sorry

end NUMINAMATH_CALUDE_gcd_87654321_12345678_l769_76904


namespace NUMINAMATH_CALUDE_no_distinct_complex_numbers_satisfying_equation_l769_76955

theorem no_distinct_complex_numbers_satisfying_equation :
  ∀ (a b c d : ℂ), 
    (a^3 - b*c*d = b^3 - a*c*d) ∧ 
    (b^3 - a*c*d = c^3 - a*b*d) ∧ 
    (c^3 - a*b*d = d^3 - a*b*c) →
    (a = b) ∨ (a = c) ∨ (a = d) ∨ (b = c) ∨ (b = d) ∨ (c = d) :=
by sorry

end NUMINAMATH_CALUDE_no_distinct_complex_numbers_satisfying_equation_l769_76955


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_l769_76935

theorem chinese_remainder_theorem (x : ℤ) : 
  (x ≡ 2 [ZMOD 6] ∧ x ≡ 3 [ZMOD 5] ∧ x ≡ 4 [ZMOD 7]) ↔ 
  (∃ k : ℤ, x = 210 * k - 52) := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_l769_76935


namespace NUMINAMATH_CALUDE_function_transformation_l769_76919

theorem function_transformation (f : ℝ → ℝ) (h : f 1 = 3) : f (-(-1)) + 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_transformation_l769_76919


namespace NUMINAMATH_CALUDE_ellipse_equation_constants_l769_76932

def ellipse_constants (f1 f2 p : ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

theorem ellipse_equation_constants :
  let f1 : ℝ × ℝ := (3, 1)
  let f2 : ℝ × ℝ := (3, 7)
  let p : ℝ × ℝ := (12, 2)
  let (a, b, h, k) := ellipse_constants f1 f2 p
  (a = (Real.sqrt 82 + Real.sqrt 106) / 2) ∧
  (b = Real.sqrt ((Real.sqrt 82 + Real.sqrt 106)^2 / 4 - 9)) ∧
  (h = 3) ∧
  (k = 4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_constants_l769_76932


namespace NUMINAMATH_CALUDE_roberto_outfits_l769_76907

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets hats : ℕ) : ℕ :=
  trousers * shirts * jackets * hats

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits :
  number_of_outfits 5 8 4 2 = 320 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l769_76907


namespace NUMINAMATH_CALUDE_raw_materials_cost_calculation_raw_materials_cost_value_l769_76930

/-- The amount Kanul spent on raw materials -/
def raw_materials_cost : ℝ := sorry

/-- The total amount Kanul had -/
def total_amount : ℝ := 5555.56

/-- The amount Kanul spent on machinery -/
def machinery_cost : ℝ := 2000

/-- The amount Kanul kept as cash -/
def cash : ℝ := 0.1 * total_amount

theorem raw_materials_cost_calculation : 
  raw_materials_cost = total_amount - machinery_cost - cash := by sorry

theorem raw_materials_cost_value : 
  raw_materials_cost = 3000 := by sorry

end NUMINAMATH_CALUDE_raw_materials_cost_calculation_raw_materials_cost_value_l769_76930


namespace NUMINAMATH_CALUDE_intercepts_sum_l769_76912

/-- A line is described by the equation y - 3 = 6(x - 5). -/
def line_equation (x y : ℝ) : Prop := y - 3 = 6 * (x - 5)

/-- The x-intercept of the line. -/
def x_intercept : ℝ := 4.5

/-- The y-intercept of the line. -/
def y_intercept : ℝ := -27

theorem intercepts_sum :
  line_equation x_intercept 0 ∧
  line_equation 0 y_intercept ∧
  x_intercept + y_intercept = -22.5 := by sorry

end NUMINAMATH_CALUDE_intercepts_sum_l769_76912


namespace NUMINAMATH_CALUDE_square_of_101_l769_76983

theorem square_of_101 : 101 * 101 = 10201 := by
  sorry

end NUMINAMATH_CALUDE_square_of_101_l769_76983


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l769_76917

theorem root_sum_reciprocal (p q r : ℝ) (A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ (x : ℝ), x^3 - 23*x^2 + 85*x - 72 = (x - p)*(x - q)*(x - r)) →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 23*s^2 + 85*s - 72) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 248 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l769_76917


namespace NUMINAMATH_CALUDE_temperature_calculation_l769_76958

/-- Given the average temperatures for two sets of four consecutive days and the temperature of the first day, calculate the temperature of the last day. -/
theorem temperature_calculation (M T W Th F : ℝ) : 
  M = 42 →
  (M + T + W + Th) / 4 = 48 →
  (T + W + Th + F) / 4 = 46 →
  F = 34 := by
  sorry

#check temperature_calculation

end NUMINAMATH_CALUDE_temperature_calculation_l769_76958


namespace NUMINAMATH_CALUDE_line_formation_ways_l769_76998

/-- The number of ways to form a line by selecting r people out of n -/
def permutations (n : ℕ) (r : ℕ) : ℕ := (n.factorial) / ((n - r).factorial)

/-- The total number of people -/
def total_people : ℕ := 7

/-- The number of people to select -/
def selected_people : ℕ := 5

theorem line_formation_ways :
  permutations total_people selected_people = 2520 := by
  sorry

end NUMINAMATH_CALUDE_line_formation_ways_l769_76998


namespace NUMINAMATH_CALUDE_joe_weight_lifting_l769_76946

theorem joe_weight_lifting (first_lift second_lift : ℕ) 
  (h1 : first_lift + second_lift = 1500)
  (h2 : 2 * first_lift = second_lift + 300) :
  first_lift = 600 := by
sorry

end NUMINAMATH_CALUDE_joe_weight_lifting_l769_76946


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l769_76986

theorem sum_of_reciprocals_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  ∃ (z : ℝ), z ≥ 2 ∧ (∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + y' = 2 ∧ 1 / x' + 1 / y' = z) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 2 → 1 / a + 1 / b ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l769_76986


namespace NUMINAMATH_CALUDE_equation_roots_l769_76948

theorem equation_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 4 ∧ x₂ = -2.5) ∧ 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → (18 / (x^2 - 4) - 3 / (x - 2) = 2 ↔ (x = x₁ ∨ x = x₂))) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l769_76948


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l769_76918

-- Problem 1
theorem problem_1 : (-3)^2 + (Real.pi - 1/2)^0 - |(-4)| = 6 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ -1) : 
  (1 - 1/(a+1)) * ((a^2 + 2*a + 1)/a) = a + 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l769_76918


namespace NUMINAMATH_CALUDE_circle_square_area_ratio_l769_76910

theorem circle_square_area_ratio (r : ℝ) (h : r > 0) :
  let inner_square_side := 3 * r
  let outer_circle_radius := inner_square_side * Real.sqrt 2 / 2
  let outer_square_side := 2 * outer_circle_radius
  (π * r^2) / (outer_square_side^2) = π / 18 := by
  sorry

end NUMINAMATH_CALUDE_circle_square_area_ratio_l769_76910


namespace NUMINAMATH_CALUDE_unique_solution_square_equation_l769_76994

theorem unique_solution_square_equation :
  ∃! y : ℤ, (2010 + y)^2 = y^2 ∧ y = -1005 := by sorry

end NUMINAMATH_CALUDE_unique_solution_square_equation_l769_76994


namespace NUMINAMATH_CALUDE_relay_race_proof_l769_76950

-- Define the total race distance
def total_distance : ℕ := 2004

-- Define the maximum time allowed (one week in hours)
def max_time : ℕ := 168

-- Define the properties of the race
theorem relay_race_proof :
  ∃ (stage_length : ℕ) (num_stages : ℕ),
    stage_length > 0 ∧
    num_stages > 0 ∧
    num_stages ≤ max_time ∧
    stage_length * num_stages = total_distance ∧
    num_stages = 167 :=
by sorry

end NUMINAMATH_CALUDE_relay_race_proof_l769_76950


namespace NUMINAMATH_CALUDE_garden_length_l769_76928

theorem garden_length (columns : ℕ) (tree_distance : ℝ) (boundary : ℝ) : 
  columns > 0 → 
  tree_distance > 0 → 
  boundary > 0 → 
  (columns - 1) * tree_distance + 2 * boundary = 32 → 
  columns = 12 ∧ tree_distance = 2 ∧ boundary = 5 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l769_76928


namespace NUMINAMATH_CALUDE_cubic_factorization_l769_76972

theorem cubic_factorization (a : ℝ) : a^3 - 4*a^2 + 4*a = a*(a-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l769_76972


namespace NUMINAMATH_CALUDE_min_marked_cells_l769_76939

/-- Represents a board with dimensions m × n -/
structure Board (m n : ℕ) where
  cells : Fin m → Fin n → Bool

/-- Represents an L-shaped piece -/
inductive LPiece
| mk : Fin 2 → Fin 2 → LPiece

/-- Checks if an L-piece placed at (i, j) touches a marked cell -/
def touchesMarked (b : Board m n) (p : LPiece) (i : Fin m) (j : Fin n) : Prop :=
  sorry

/-- Checks if a marking strategy ensures all L-piece placements touch a marked cell -/
def validMarking (b : Board m n) : Prop :=
  ∀ (p : LPiece) (i : Fin m) (j : Fin n), touchesMarked b p i j

/-- Counts the number of marked cells on a board -/
def countMarked (b : Board m n) : ℕ :=
  sorry

/-- Theorem stating that 50 is the smallest number of marked cells required -/
theorem min_marked_cells :
  (∃ (b : Board 10 11), validMarking b ∧ countMarked b = 50) ∧
  (∀ (b : Board 10 11), validMarking b → countMarked b ≥ 50) :=
sorry

end NUMINAMATH_CALUDE_min_marked_cells_l769_76939


namespace NUMINAMATH_CALUDE_midpoint_expression_evaluation_l769_76985

/-- Given two points P and Q in the plane, prove that the expression 3x - 5y 
    evaluates to -36 at their midpoint R(x, y). -/
theorem midpoint_expression_evaluation (P Q : ℝ × ℝ) (h1 : P = (12, 15)) (h2 : Q = (4, 9)) :
  let R : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  3 * R.1 - 5 * R.2 = -36 := by
sorry

end NUMINAMATH_CALUDE_midpoint_expression_evaluation_l769_76985


namespace NUMINAMATH_CALUDE_rod_weight_calculation_l769_76900

/-- Given two rods of different lengths, calculates the weight of the longer rod -/
theorem rod_weight_calculation (length_short : ℝ) (length_long : ℝ) (weight_short : ℝ) 
  (h1 : length_short = 6)
  (h2 : length_long = 12)
  (h3 : weight_short = 6.1) : 
  (weight_short / length_short) * length_long = 12.2 := by
  sorry

end NUMINAMATH_CALUDE_rod_weight_calculation_l769_76900


namespace NUMINAMATH_CALUDE_brenda_friends_count_l769_76959

/-- Prove that Brenda has 9 friends given the pizza ordering scenario -/
theorem brenda_friends_count :
  let slices_per_person : ℕ := 2
  let slices_per_pizza : ℕ := 4
  let pizzas_ordered : ℕ := 5
  let total_slices : ℕ := slices_per_pizza * pizzas_ordered
  let total_people : ℕ := total_slices / slices_per_person
  let brenda_friends : ℕ := total_people - 1
  brenda_friends = 9 := by
  sorry

end NUMINAMATH_CALUDE_brenda_friends_count_l769_76959


namespace NUMINAMATH_CALUDE_exists_expr_2023_l769_76953

/-- An arithmetic expression without parentheses -/
inductive ArithExpr
  | Const : ℤ → ArithExpr
  | Add : ArithExpr → ArithExpr → ArithExpr
  | Sub : ArithExpr → ArithExpr → ArithExpr
  | Mul : ArithExpr → ArithExpr → ArithExpr
  | Div : ArithExpr → ArithExpr → ArithExpr

/-- Evaluation function for ArithExpr -/
def eval : ArithExpr → ℤ
  | ArithExpr.Const n => n
  | ArithExpr.Add a b => eval a + eval b
  | ArithExpr.Sub a b => eval a - eval b
  | ArithExpr.Mul a b => eval a * eval b
  | ArithExpr.Div a b => eval a / eval b

/-- Theorem stating the existence of an arithmetic expression evaluating to 2023 -/
theorem exists_expr_2023 : ∃ e : ArithExpr, eval e = 2023 := by
  sorry


end NUMINAMATH_CALUDE_exists_expr_2023_l769_76953


namespace NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l769_76947

/-- Calculates the tip percentage given the total bill, food price, and sales tax rate. -/
def calculate_tip_percentage (total_bill : ℚ) (food_price : ℚ) (sales_tax_rate : ℚ) : ℚ :=
  let sales_tax := food_price * sales_tax_rate
  let total_before_tip := food_price + sales_tax
  let tip_amount := total_bill - total_before_tip
  (tip_amount / total_before_tip) * 100

/-- Proves that the tip percentage is 20% given the specified conditions. -/
theorem tip_percentage_is_twenty_percent :
  calculate_tip_percentage 158.40 120 0.10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_is_twenty_percent_l769_76947


namespace NUMINAMATH_CALUDE_number_equation_l769_76913

theorem number_equation (x : ℝ) (h : 5 * x = 2 * x + 10) : 5 * x - 2 * x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l769_76913


namespace NUMINAMATH_CALUDE_find_divisor_l769_76993

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) (divisor : Nat) :
  dividend = quotient * divisor + remainder →
  dividend = 172 →
  quotient = 10 →
  remainder = 2 →
  divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l769_76993


namespace NUMINAMATH_CALUDE_rectangle_area_l769_76963

theorem rectangle_area (length width perimeter area : ℝ) : 
  width = (2 / 3) * length →
  perimeter = 2 * (length + width) →
  perimeter = 148 →
  area = length * width →
  area = 1314.24 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l769_76963


namespace NUMINAMATH_CALUDE_gcd_2197_2208_l769_76987

theorem gcd_2197_2208 : Nat.gcd 2197 2208 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2197_2208_l769_76987


namespace NUMINAMATH_CALUDE_dvd_book_problem_l769_76989

theorem dvd_book_problem (total_capacity : ℕ) (empty_spaces : ℕ) (h1 : total_capacity = 126) (h2 : empty_spaces = 45) :
  total_capacity - empty_spaces = 81 := by
  sorry

end NUMINAMATH_CALUDE_dvd_book_problem_l769_76989


namespace NUMINAMATH_CALUDE_linear_equation_condition_l769_76971

/-- 
If (a-6)x - y^(a-6) = 1 is a linear equation in x and y, then a = 7.
-/
theorem linear_equation_condition (a : ℝ) : 
  (∀ x y : ℝ, ∃ k m : ℝ, (a - 6) * x - y^(a - 6) = k * x + m * y + 1) → a = 7 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l769_76971


namespace NUMINAMATH_CALUDE_man_son_age_ratio_l769_76937

/-- The ratio of a man's age to his son's age after two years, given their current ages. -/
theorem man_son_age_ratio (son_age : ℕ) (man_age : ℕ) : 
  son_age = 22 →
  man_age = son_age + 24 →
  (man_age + 2) / (son_age + 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_ratio_l769_76937


namespace NUMINAMATH_CALUDE_binomial_max_prob_l769_76967

/-- The probability mass function of a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The value of k that maximizes the probability mass function for B(200, 1/2) -/
theorem binomial_max_prob (ζ : ℕ → ℝ) (h : ∀ k, ζ k = binomial_pmf 200 (1/2) k) :
  ∃ k : ℕ, k = 100 ∧ ∀ j : ℕ, ζ k ≥ ζ j :=
sorry

end NUMINAMATH_CALUDE_binomial_max_prob_l769_76967


namespace NUMINAMATH_CALUDE_square_sum_from_means_l769_76942

theorem square_sum_from_means (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h_arithmetic : (p + q + r) / 3 = 10)
  (h_geometric : (p * q * r) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/p + 1/q + 1/r) = 4) :
  p^2 + q^2 + r^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l769_76942
