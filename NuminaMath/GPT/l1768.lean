import Mathlib

namespace NUMINAMATH_GPT_investment_ratio_l1768_176872

theorem investment_ratio (X_investment Y_investment : ℕ) (hX : X_investment = 5000) (hY : Y_investment = 15000) : 
  X_investment * 3 = Y_investment :=
by
  sorry

end NUMINAMATH_GPT_investment_ratio_l1768_176872


namespace NUMINAMATH_GPT_exchange_rate_5_CAD_to_JPY_l1768_176883

theorem exchange_rate_5_CAD_to_JPY :
  (1 : ℝ) * 85 * 5 = 425 :=
by
  sorry

end NUMINAMATH_GPT_exchange_rate_5_CAD_to_JPY_l1768_176883


namespace NUMINAMATH_GPT_tangent_line_eqn_when_a_zero_min_value_f_when_a_zero_range_of_a_for_x_ge_zero_exp_x_ln_x_plus_one_gt_x_sq_l1768_176875

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - 1 - x - a * x ^ 2

theorem tangent_line_eqn_when_a_zero :
  (∀ x, y = f 0 x → y - (Real.exp 1 - 2) = (Real.exp 1 - 1) * (x - 1)) :=
sorry

theorem min_value_f_when_a_zero :
  (∀ x : ℝ, f 0 x >= f 0 0) := 
sorry

theorem range_of_a_for_x_ge_zero (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f a x ≥ 0) → (a ≤ 1/2) :=
sorry

theorem exp_x_ln_x_plus_one_gt_x_sq (x : ℝ) :
  x > 0 → ((Real.exp x - 1) * Real.log (x + 1) > x ^ 2) :=
sorry

end NUMINAMATH_GPT_tangent_line_eqn_when_a_zero_min_value_f_when_a_zero_range_of_a_for_x_ge_zero_exp_x_ln_x_plus_one_gt_x_sq_l1768_176875


namespace NUMINAMATH_GPT_jane_20_cent_items_l1768_176863

theorem jane_20_cent_items {x y z : ℕ} (h1 : x + y + z = 50) (h2 : 20 * x + 150 * y + 250 * z = 5000) : x = 31 :=
by
  -- The formal proof would go here
  sorry

end NUMINAMATH_GPT_jane_20_cent_items_l1768_176863


namespace NUMINAMATH_GPT_num_values_between_l1768_176836

theorem num_values_between (x y : ℕ) (h1 : x + y ≥ 200) (h2 : x + y ≤ 1000) 
  (h3 : (x * (x - 1) + y * (y - 1)) * 2 = (x + y) * (x + y - 1)) : 
  ∃ n : ℕ, n - 1 = 17 := by
  sorry

end NUMINAMATH_GPT_num_values_between_l1768_176836


namespace NUMINAMATH_GPT_minimum_value_of_z_l1768_176882

theorem minimum_value_of_z (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) : ∃ min_z, min_z = (1 + Real.sqrt 5) / 4 ∧ ∀ z, z = x^2 + y^2 → min_z ≤ z :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_z_l1768_176882


namespace NUMINAMATH_GPT_quadratic_roots_l1768_176840

theorem quadratic_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + k*x1 + (k - 1) = 0) ∧ (x2^2 + k*x2 + (k - 1) = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1768_176840


namespace NUMINAMATH_GPT_heart_op_ratio_l1768_176834

def heart_op (n m : ℕ) : ℕ := n^3 * m^2

theorem heart_op_ratio : heart_op 3 5 / heart_op 5 3 = 5 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_heart_op_ratio_l1768_176834


namespace NUMINAMATH_GPT_regular_hexagon_area_l1768_176849

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem regular_hexagon_area 
  (A C : ℝ × ℝ)
  (hA : A = (0, 0))
  (hC : C = (8, 2))
  (h_eq_side_length : ∀ x y : ℝ × ℝ, dist A.1 A.2 C.1 C.2 = dist x.1 x.2 y.1 y.2) :
  hexagon_area = 34 * Real.sqrt 3 :=
by
  -- sorry indicates the proof is omitted
  sorry

end NUMINAMATH_GPT_regular_hexagon_area_l1768_176849


namespace NUMINAMATH_GPT_measure_of_smaller_angle_l1768_176866

noncomputable def complementary_angle_ratio_smaller (x : ℝ) (h : 4 * x + x = 90) : ℝ :=
x

theorem measure_of_smaller_angle (x : ℝ) (h : 4 * x + x = 90) : complementary_angle_ratio_smaller x h = 18 :=
sorry

end NUMINAMATH_GPT_measure_of_smaller_angle_l1768_176866


namespace NUMINAMATH_GPT_alpha_range_l1768_176829

theorem alpha_range (α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) : 
  (Real.sin α < Real.sqrt 3 / 2 ∧ Real.cos α > 1 / 2) ↔ 
  (0 < α ∧ α < Real.pi / 3 ∨ 5 * Real.pi / 3 < α ∧ α < 2 * Real.pi) := 
sorry

end NUMINAMATH_GPT_alpha_range_l1768_176829


namespace NUMINAMATH_GPT_Anya_walks_to_school_l1768_176811

theorem Anya_walks_to_school
  (t_f t_b : ℝ)
  (h1 : t_f + t_b = 1.5)
  (h2 : 2 * t_b = 0.5) :
  2 * t_f = 2.5 :=
by
  -- The proof details will go here eventually.
  sorry

end NUMINAMATH_GPT_Anya_walks_to_school_l1768_176811


namespace NUMINAMATH_GPT_impossible_coins_l1768_176862

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end NUMINAMATH_GPT_impossible_coins_l1768_176862


namespace NUMINAMATH_GPT_correct_calculation_l1768_176894

theorem correct_calculation (a b : ℝ) : 
  (¬ (2 * (a - 1) = 2 * a - 1)) ∧ 
  (3 * a^2 - 2 * a^2 = a^2) ∧ 
  (¬ (3 * a^2 - 2 * a^2 = 1)) ∧ 
  (¬ (3 * a + 2 * b = 5 * a * b)) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1768_176894


namespace NUMINAMATH_GPT_determine_a_l1768_176830

theorem determine_a (a : ℝ) (h : 2 * (-1) + a = 3) : a = 5 := sorry

end NUMINAMATH_GPT_determine_a_l1768_176830


namespace NUMINAMATH_GPT_sum_of_roots_of_y_squared_eq_36_l1768_176888

theorem sum_of_roots_of_y_squared_eq_36 :
  (∀ y : ℝ, y^2 = 36 → y = 6 ∨ y = -6) → (6 + (-6) = 0) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_y_squared_eq_36_l1768_176888


namespace NUMINAMATH_GPT_range_of_a_for_inequality_l1768_176846

theorem range_of_a_for_inequality (a : ℝ) : (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ (a ≥ -2) :=
sorry

end NUMINAMATH_GPT_range_of_a_for_inequality_l1768_176846


namespace NUMINAMATH_GPT_factoring_difference_of_squares_l1768_176807

theorem factoring_difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := 
sorry

end NUMINAMATH_GPT_factoring_difference_of_squares_l1768_176807


namespace NUMINAMATH_GPT_min_vans_proof_l1768_176805

-- Define the capacity and availability of each type of van
def capacity_A : Nat := 7
def capacity_B : Nat := 9
def capacity_C : Nat := 12

def available_A : Nat := 3
def available_B : Nat := 4
def available_C : Nat := 2

-- Define the number of people going on the trip
def students : Nat := 40
def adults : Nat := 14

-- Define the total number of people
def total_people : Nat := students + adults

-- Define the minimum number of vans needed
def min_vans_needed : Nat := 6

-- Define the number of each type of van used
def vans_A_used : Nat := 0
def vans_B_used : Nat := 4
def vans_C_used : Nat := 2

-- Prove the minimum number of vans needed to accommodate everyone is 6
theorem min_vans_proof : min_vans_needed = 6 ∧ 
  (vans_A_used * capacity_A + vans_B_used * capacity_B + vans_C_used * capacity_C = total_people) ∧
  vans_A_used <= available_A ∧ vans_B_used <= available_B ∧ vans_C_used <= available_C :=
by 
  sorry

end NUMINAMATH_GPT_min_vans_proof_l1768_176805


namespace NUMINAMATH_GPT_triangle_tangent_half_angle_l1768_176809

theorem triangle_tangent_half_angle (a b c : ℝ) (A : ℝ) (C : ℝ)
  (h : a + c = 2 * b) :
  Real.tan (A / 2) * Real.tan (C / 2) = 1 / 3 := 
sorry

end NUMINAMATH_GPT_triangle_tangent_half_angle_l1768_176809


namespace NUMINAMATH_GPT_total_candy_pieces_l1768_176844

theorem total_candy_pieces : 
  (brother_candy = 6) → 
  (wendy_boxes = 2) → 
  (pieces_per_box = 3) → 
  (brother_candy + (wendy_boxes * pieces_per_box) = 12) 
  := 
  by 
    intros brother_candy wendy_boxes pieces_per_box 
    sorry

end NUMINAMATH_GPT_total_candy_pieces_l1768_176844


namespace NUMINAMATH_GPT_train_cross_time_approx_l1768_176819

noncomputable def length_of_train : ℝ := 100
noncomputable def speed_of_train_km_hr : ℝ := 80
noncomputable def length_of_bridge : ℝ := 142
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def speed_of_train_m_s : ℝ := speed_of_train_km_hr * 1000 / 3600
noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_m_s

theorem train_cross_time_approx :
  abs (time_to_cross_bridge - 10.89) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_time_approx_l1768_176819


namespace NUMINAMATH_GPT_proof_problem_l1768_176856

variables (x y b z a : ℝ)

def condition1 : Prop := x * y + x^2 = b
def condition2 : Prop := (1 / x^2) - (1 / y^2) = a
def z_def : Prop := z = x + y

theorem proof_problem (x y b z a : ℝ) (h1 : condition1 x y b) (h2 : condition2 x y a) (hz : z_def x y z) : (x + y) ^ 2 = z ^ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l1768_176856


namespace NUMINAMATH_GPT_analysis_hours_l1768_176800

-- Define the conditions: number of bones and minutes per bone
def number_of_bones : Nat := 206
def minutes_per_bone : Nat := 45

-- Define the conversion factor: minutes per hour
def minutes_per_hour : Nat := 60

-- Define the total minutes spent analyzing all bones
def total_minutes (number_of_bones minutes_per_bone : Nat) : Nat :=
  number_of_bones * minutes_per_bone

-- Define the total hours required for analysis
def total_hours (total_minutes minutes_per_hour : Nat) : Float :=
  total_minutes.toFloat / minutes_per_hour.toFloat

-- Prove that total_hours equals 154.5 hours
theorem analysis_hours : total_hours (total_minutes number_of_bones minutes_per_bone) minutes_per_hour = 154.5 := by
  sorry

end NUMINAMATH_GPT_analysis_hours_l1768_176800


namespace NUMINAMATH_GPT_hayden_earnings_l1768_176879

theorem hayden_earnings 
  (wage_per_hour : ℕ) 
  (pay_per_ride : ℕ)
  (bonus_per_review : ℕ)
  (number_of_rides : ℕ)
  (hours_worked : ℕ)
  (gas_cost_per_gallon : ℕ)
  (gallons_of_gas : ℕ)
  (positive_reviews : ℕ)
  : wage_per_hour = 15 → 
    pay_per_ride = 5 → 
    bonus_per_review = 20 → 
    number_of_rides = 3 → 
    hours_worked = 8 → 
    gas_cost_per_gallon = 3 → 
    gallons_of_gas = 17 → 
    positive_reviews = 2 → 
    (hours_worked * wage_per_hour + number_of_rides * pay_per_ride + positive_reviews * bonus_per_review + gallons_of_gas * gas_cost_per_gallon) = 226 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Further proof processing with these assumptions
  sorry

end NUMINAMATH_GPT_hayden_earnings_l1768_176879


namespace NUMINAMATH_GPT_kan_krao_park_walkways_l1768_176808

-- Definitions for the given conditions
structure Park (α : Type*) := 
  (entrances : Finset α)
  (walkways : α → α → Prop)
  (brick_paved : α → α → Prop)
  (asphalt_paved : α → α → Prop)
  (no_three_intersections : ∀ (x y z w : α), x ≠ y → y ≠ z → z ≠ w → w ≠ x → (walkways x y ∧ walkways z w) → ¬ (walkways x z ∧ walkways y w))

-- Conditions based on the given problem
variables {α : Type*} [Finite α] [DecidableRel (@walkways α)]
variable (p : Park α)
variables [Fintype α]

-- Translate conditions to definitions
def has_lotuses (p : α → α → Prop) (q : α → α → Prop) (x y : α) : Prop := p x y ∧ p x y
def has_waterlilies (p : α → α → Prop) (q : α → α → Prop) (x y : α) : Prop := (p x y ∧ q x y) ∨ (q x y ∧ p x y)
def is_lit (p : α → α → Prop) (q : α → α → Prop) : Prop := ∃ (x y : α), x ≠ y ∧ (has_lotuses p q x y ∧ has_lotuses p q x y ∧ ∃ sz, sz ≥ 45)

-- Mathematically equivalent proof problem
theorem kan_krao_park_walkways (p : Park α) :
  (∃ walkways_same_material : α → α → Prop, ∃ (lit_walkways : Finset (α × α)), lit_walkways.card ≥ 11) :=
sorry

end NUMINAMATH_GPT_kan_krao_park_walkways_l1768_176808


namespace NUMINAMATH_GPT_complete_square_formula_D_l1768_176867

-- Definitions of polynomial multiplications
def poly_A (a b : ℝ) : ℝ := (a - b) * (a + b)
def poly_B (a b : ℝ) : ℝ := -((a + b) * (b - a))
def poly_C (a b : ℝ) : ℝ := (a + b) * (b - a)
def poly_D (a b : ℝ) : ℝ := (a - b) * (b - a)

theorem complete_square_formula_D (a b : ℝ) : 
  poly_D a b = -(a - b)*(a - b) :=
by sorry

end NUMINAMATH_GPT_complete_square_formula_D_l1768_176867


namespace NUMINAMATH_GPT_intersection_M_N_l1768_176860

def set_M : Set ℝ := { x | x < 2 }
def set_N : Set ℝ := { x | x > 0 }
def set_intersection : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_M_N : set_M ∩ set_N = set_intersection := 
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1768_176860


namespace NUMINAMATH_GPT_probability_of_snow_at_least_once_first_week_l1768_176833

theorem probability_of_snow_at_least_once_first_week :
  let p_first4 := 1 / 4
  let p_next3 := 1 / 3
  let p_no_snow_first4 := (1 - p_first4) ^ 4
  let p_no_snow_next3 := (1 - p_next3) ^ 3
  let p_no_snow_week := p_no_snow_first4 * p_no_snow_next3
  1 - p_no_snow_week = 29 / 32 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_snow_at_least_once_first_week_l1768_176833


namespace NUMINAMATH_GPT_function_value_range_l1768_176839

noncomputable def f (x : ℝ) : ℝ := 9^x - 3^(x+1) + 2

theorem function_value_range :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → -1/4 ≤ f x ∧ f x ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_function_value_range_l1768_176839


namespace NUMINAMATH_GPT_symmetric_y_axis_l1768_176822

theorem symmetric_y_axis (a b : ℝ) (h₁ : a = -4) (h₂ : b = 3) : a - b = -7 :=
by
  rw [h₁, h₂]
  norm_num

end NUMINAMATH_GPT_symmetric_y_axis_l1768_176822


namespace NUMINAMATH_GPT_max_player_salary_l1768_176878

theorem max_player_salary (n : ℕ) (min_salary total_salary : ℕ) (player_count : ℕ)
  (h1 : player_count = 25)
  (h2 : min_salary = 15000)
  (h3 : total_salary = 850000)
  (h4 : n = 24 * min_salary)
  : (total_salary - n) = 490000 := 
by
  -- assumptions ensure that n represents the total minimum salaries paid to 24 players
  sorry

end NUMINAMATH_GPT_max_player_salary_l1768_176878


namespace NUMINAMATH_GPT_length_of_boat_l1768_176821

-- Definitions based on the conditions
def breadth : ℝ := 3
def sink_depth : ℝ := 0.01
def man_mass : ℝ := 120
def g : ℝ := 9.8 -- acceleration due to gravity

-- Derived from the conditions
def weight_man : ℝ := man_mass * g
def density_water : ℝ := 1000

-- Statement to be proved
theorem length_of_boat : ∃ L : ℝ, (breadth * sink_depth * L * density_water * g = weight_man) → L = 4 :=
by
  sorry

end NUMINAMATH_GPT_length_of_boat_l1768_176821


namespace NUMINAMATH_GPT_no_unfenced_area_l1768_176899

noncomputable def area : ℝ := 5000
noncomputable def cost_per_foot : ℝ := 30
noncomputable def budget : ℝ := 120000

theorem no_unfenced_area (area : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  (budget / cost_per_foot) >= 4 * (Real.sqrt (area)) → 0 = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_unfenced_area_l1768_176899


namespace NUMINAMATH_GPT_incenter_x_coordinate_eq_l1768_176890

theorem incenter_x_coordinate_eq (x y : ℝ) :
  (x = y) ∧ 
  (y = -x + 3) → 
  x = 3 / 2 := 
sorry

end NUMINAMATH_GPT_incenter_x_coordinate_eq_l1768_176890


namespace NUMINAMATH_GPT_original_loaf_slices_l1768_176881

-- Define the given conditions
def andy_slices_1 := 3
def andy_slices_2 := 3
def toast_slices_per_piece := 2
def pieces_of_toast := 10
def slices_left_over := 1

-- Define the variables
def total_andy_slices := andy_slices_1 + andy_slices_2
def total_toast_slices := toast_slices_per_piece * pieces_of_toast

-- State the theorem
theorem original_loaf_slices : 
  ∃ S : ℕ, S = total_andy_slices + total_toast_slices + slices_left_over := 
by {
  sorry
}

end NUMINAMATH_GPT_original_loaf_slices_l1768_176881


namespace NUMINAMATH_GPT_determine_CD_l1768_176824

theorem determine_CD (AB : ℝ) (BD : ℝ) (BC : ℝ) (CD : ℝ) (Angle_ADB : ℝ)
  (sin_A : ℝ) (sin_C : ℝ)
  (h1 : AB = 30)
  (h2 : Angle_ADB = 90)
  (h3 : sin_A = 4/5)
  (h4 : sin_C = 1/5)
  (h5 : BD = sin_A * AB)
  (h6 : BC = BD / sin_C) :
  CD = 24 * Real.sqrt 23 := by
  sorry

end NUMINAMATH_GPT_determine_CD_l1768_176824


namespace NUMINAMATH_GPT_company_blocks_l1768_176896

noncomputable def number_of_blocks (workers_per_block total_budget gift_cost : ℕ) : ℕ :=
  (total_budget / gift_cost) / workers_per_block

theorem company_blocks :
  number_of_blocks 200 6000 2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_company_blocks_l1768_176896


namespace NUMINAMATH_GPT_no_real_roots_l1768_176873

-- Define the polynomial Q(x)
def Q (x : ℝ) : ℝ := x^6 - 3 * x^5 + 6 * x^4 - 6 * x^3 - x + 8

-- The problem can be stated as proving that Q(x) has no real roots
theorem no_real_roots : ∀ x : ℝ, Q x ≠ 0 := by
  sorry

end NUMINAMATH_GPT_no_real_roots_l1768_176873


namespace NUMINAMATH_GPT_jesse_pencils_l1768_176865

def initial_pencils : ℕ := 78
def pencils_given : ℕ := 44
def final_pencils : ℕ := initial_pencils - pencils_given

theorem jesse_pencils :
  final_pencils = 34 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_jesse_pencils_l1768_176865


namespace NUMINAMATH_GPT_complex_expression_l1768_176838

-- The condition: n is a positive integer
variable (n : ℕ) (hn : 0 < n)

-- Definition of the problem to be proved
theorem complex_expression (n : ℕ) (hn : 0 < n) : 
  (Complex.I ^ (4 * n) + Complex.I ^ (4 * n + 1) + Complex.I ^ (4 * n + 2) + Complex.I ^ (4 * n + 3)) = 0 :=
sorry

end NUMINAMATH_GPT_complex_expression_l1768_176838


namespace NUMINAMATH_GPT_evaluate_powers_of_i_l1768_176891

noncomputable def imag_unit := Complex.I

theorem evaluate_powers_of_i :
  (imag_unit^11 + imag_unit^16 + imag_unit^21 + imag_unit^26 + imag_unit^31) = -imag_unit :=
by
  sorry

end NUMINAMATH_GPT_evaluate_powers_of_i_l1768_176891


namespace NUMINAMATH_GPT_determine_m_first_degree_inequality_l1768_176868

theorem determine_m_first_degree_inequality (m : ℝ) (x : ℝ) :
  (m + 1) * x ^ |m| + 2 > 0 → |m| = 1 → m = 1 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_determine_m_first_degree_inequality_l1768_176868


namespace NUMINAMATH_GPT_johnnyMoneyLeft_l1768_176828

noncomputable def johnnySavingsSeptember : ℝ := 30
noncomputable def johnnySavingsOctober : ℝ := 49
noncomputable def johnnySavingsNovember : ℝ := 46
noncomputable def johnnySavingsDecember : ℝ := 55

noncomputable def johnnySavingsJanuary : ℝ := johnnySavingsDecember * 1.15

noncomputable def totalSavings : ℝ := johnnySavingsSeptember + johnnySavingsOctober + johnnySavingsNovember + johnnySavingsDecember + johnnySavingsJanuary

noncomputable def videoGameCost : ℝ := 58
noncomputable def bookCost : ℝ := 25
noncomputable def birthdayPresentCost : ℝ := 40

noncomputable def totalSpent : ℝ := videoGameCost + bookCost + birthdayPresentCost

noncomputable def moneyLeft : ℝ := totalSavings - totalSpent

theorem johnnyMoneyLeft : moneyLeft = 120.25 := by
  sorry

end NUMINAMATH_GPT_johnnyMoneyLeft_l1768_176828


namespace NUMINAMATH_GPT_rectangle_perimeter_of_right_triangle_l1768_176815

noncomputable def right_triangle_area (a b: ℕ) : ℝ := (1/2 : ℝ) * a * b

noncomputable def rectangle_length (area width: ℝ) : ℝ := area / width

noncomputable def rectangle_perimeter (length width: ℝ) : ℝ := 2 * (length + width)

theorem rectangle_perimeter_of_right_triangle :
  rectangle_perimeter (rectangle_length (right_triangle_area 7 24) 5) 5 = 43.6 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_of_right_triangle_l1768_176815


namespace NUMINAMATH_GPT_repeating_decimal_calculation_l1768_176893

theorem repeating_decimal_calculation :
  2 * (8 / 9 - 2 / 9 + 4 / 9) = 20 / 9 :=
by
  -- sorry proof will be inserted here.
  sorry

end NUMINAMATH_GPT_repeating_decimal_calculation_l1768_176893


namespace NUMINAMATH_GPT_family_members_l1768_176889

variable (p : ℝ) (i : ℝ) (c : ℝ)

theorem family_members (h1 : p = 1.6) (h2 : i = 0.25) (h3 : c = 16) :
  (c / (2 * (p * (1 + i)))) = 4 := by
  sorry

end NUMINAMATH_GPT_family_members_l1768_176889


namespace NUMINAMATH_GPT_mia_high_school_has_2000_students_l1768_176864

variables (M Z : ℕ)

def mia_high_school_students : Prop :=
  M = 4 * Z ∧ M + Z = 2500

theorem mia_high_school_has_2000_students (h : mia_high_school_students M Z) : 
  M = 2000 := by
  sorry

end NUMINAMATH_GPT_mia_high_school_has_2000_students_l1768_176864


namespace NUMINAMATH_GPT_sum_of_nonnegative_reals_l1768_176803

theorem sum_of_nonnegative_reals (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 27) :
  x + y + z = Real.sqrt 106 :=
sorry

end NUMINAMATH_GPT_sum_of_nonnegative_reals_l1768_176803


namespace NUMINAMATH_GPT_trigonometric_identity_l1768_176885

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α - Real.pi / 4) = 1 / 3) : 
  Real.cos (Real.pi / 4 + α) = - (1 / 3) := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1768_176885


namespace NUMINAMATH_GPT_least_integer_value_l1768_176823

-- Define the condition and then prove the statement
theorem least_integer_value (x : ℤ) (h : 3 * |x| - 2 > 13) : x = -6 :=
by
  sorry

end NUMINAMATH_GPT_least_integer_value_l1768_176823


namespace NUMINAMATH_GPT_commute_time_difference_l1768_176892

theorem commute_time_difference (x y : ℝ) 
  (h1 : x + y = 39)
  (h2 : (x - 10)^2 + (y - 10)^2 = 10) :
  |x - y| = 4 :=
by
  sorry

end NUMINAMATH_GPT_commute_time_difference_l1768_176892


namespace NUMINAMATH_GPT_olivia_correct_answers_l1768_176842

theorem olivia_correct_answers (c w : ℕ) (h1 : c + w = 15) (h2 : 4 * c - 3 * w = 25) : c = 10 :=
by
  sorry

end NUMINAMATH_GPT_olivia_correct_answers_l1768_176842


namespace NUMINAMATH_GPT_greatest_number_of_sets_l1768_176857

-- Definitions based on conditions
def whitney_tshirts := 5
def whitney_buttons := 24
def whitney_stickers := 12
def buttons_per_set := 2
def stickers_per_set := 1

-- The statement to prove the greatest number of identical sets Whitney can make
theorem greatest_number_of_sets : 
  ∃ max_sets : ℕ, 
  max_sets = whitney_tshirts ∧ 
  max_sets ≤ (whitney_buttons / buttons_per_set) ∧
  max_sets ≤ (whitney_stickers / stickers_per_set) :=
sorry

end NUMINAMATH_GPT_greatest_number_of_sets_l1768_176857


namespace NUMINAMATH_GPT_problem_correct_l1768_176801

theorem problem_correct (x : ℝ) : 
  14 * ((150 / 3) + (35 / 7) + (16 / 32) + x) = 777 + 14 * x := 
by
  sorry

end NUMINAMATH_GPT_problem_correct_l1768_176801


namespace NUMINAMATH_GPT_triangle_angle_A_triangle_length_b_l1768_176886

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (m n : ℝ × ℝ)
variable (S : ℝ)

theorem triangle_angle_A (h1 : a = 7) (h2 : c = 8) (h3 : m = (1, 7 * a)) (h4 : n = (-4 * a, Real.sin C))
  (h5 : m.1 * n.1 + m.2 * n.2 = 0) : 
  A = Real.pi / 6 := 
  sorry

theorem triangle_length_b (h1 : a = 7) (h2 : c = 8) (h3 : (7 * 8 * Real.sin B) / 2 = 16 * Real.sqrt 3) :
  b = Real.sqrt 97 :=
  sorry

end NUMINAMATH_GPT_triangle_angle_A_triangle_length_b_l1768_176886


namespace NUMINAMATH_GPT_range_of_m_l1768_176855

namespace MathProof

def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - m * x + m - 1 = 0 }

theorem range_of_m (m : ℝ) (h : A ∪ (B m) = A) : m = 3 :=
  sorry

end MathProof

end NUMINAMATH_GPT_range_of_m_l1768_176855


namespace NUMINAMATH_GPT_minimum_value_of_a_plus_2b_l1768_176845

theorem minimum_value_of_a_plus_2b 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : 2 * a + b = a * b - 1) 
  : a + 2 * b = 5 + 2 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_a_plus_2b_l1768_176845


namespace NUMINAMATH_GPT_find_old_weight_l1768_176870

variable (avg_increase : ℝ) (num_persons : ℕ) (W_new : ℝ) (total_increase : ℝ) (W_old : ℝ)

theorem find_old_weight (h1 : avg_increase = 3.5) 
                        (h2 : num_persons = 7) 
                        (h3 : W_new = 99.5) 
                        (h4 : total_increase = num_persons * avg_increase) 
                        (h5 : W_new = W_old + total_increase) 
                        : W_old = 75 :=
by
  sorry

end NUMINAMATH_GPT_find_old_weight_l1768_176870


namespace NUMINAMATH_GPT_combined_mpg_is_30_l1768_176806

-- Define the constants
def ray_efficiency : ℕ := 50 -- miles per gallon
def tom_efficiency : ℕ := 25 -- miles per gallon
def ray_distance : ℕ := 100 -- miles
def tom_distance : ℕ := 200 -- miles

-- Define the combined miles per gallon calculation and the proof statement.
theorem combined_mpg_is_30 :
  (ray_distance + tom_distance) /
  ((ray_distance / ray_efficiency) + (tom_distance / tom_efficiency)) = 30 :=
by
  -- All proof steps are skipped using sorry
  sorry

end NUMINAMATH_GPT_combined_mpg_is_30_l1768_176806


namespace NUMINAMATH_GPT_part_a_part_b_l1768_176810

-- Define the conditions for part (a)
def psychic_can_guess_at_least_19_cards : Prop :=
  ∃ (deck : Fin 36 → Fin 4) (psychic_guess : Fin 36 → Fin 4)
    (assistant_arrangement : Fin 36 → Bool),
    -- assistant and psychic agree on a method ensuring at least 19 correct guesses
    (∃ n : ℕ, n ≥ 19 ∧
      ∃ correct_guesses_set : Finset (Fin 36),
        correct_guesses_set.card = n ∧
        ∀ i ∈ correct_guesses_set, psychic_guess i = deck i)

-- Prove that the above condition is satisfied
theorem part_a : psychic_can_guess_at_least_19_cards :=
by
  sorry

-- Define the conditions for part (b)
def psychic_can_guess_at_least_23_cards : Prop :=
  ∃ (deck : Fin 36 → Fin 4) (psychic_guess : Fin 36 → Fin 4)
    (assistant_arrangement : Fin 36 → Bool),
    -- assistant and psychic agree on a method ensuring at least 23 correct guesses
    (∃ n : ℕ, n ≥ 23 ∧
      ∃ correct_guesses_set : Finset (Fin 36),
        correct_guesses_set.card = n ∧
        ∀ i ∈ correct_guesses_set, psychic_guess i = deck i)

-- Prove that the above condition is satisfied
theorem part_b : psychic_can_guess_at_least_23_cards :=
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l1768_176810


namespace NUMINAMATH_GPT_expr_eval_l1768_176897

noncomputable def expr_value : ℕ :=
  (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6)

theorem expr_eval : expr_value = 18 := by
  sorry

end NUMINAMATH_GPT_expr_eval_l1768_176897


namespace NUMINAMATH_GPT_prob_sunny_l1768_176812

variables (A B C : Prop) 
variables (P : Prop → ℝ)

-- Conditions
axiom prob_A : P A = 0.45
axiom prob_B : P B = 0.2
axiom mutually_exclusive : P A + P B + P C = 1

-- Proof problem
theorem prob_sunny : P C = 0.35 :=
by sorry

end NUMINAMATH_GPT_prob_sunny_l1768_176812


namespace NUMINAMATH_GPT_pass_rate_correct_l1768_176841

variable {a b : ℝ}

-- Assumptions: defect rates are between 0 and 1
axiom h_a : 0 ≤ a ∧ a ≤ 1
axiom h_b : 0 ≤ b ∧ b ≤ 1

-- Definition: Pass rate is 1 minus the defect rate
def pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

-- Theorem: Proving the pass rate is (1 - a) * (1 - b)
theorem pass_rate_correct : pass_rate a b = (1 - a) * (1 - b) := 
by
  sorry

end NUMINAMATH_GPT_pass_rate_correct_l1768_176841


namespace NUMINAMATH_GPT_percent_of_value_l1768_176827

theorem percent_of_value : (2 / 5) * (1 / 100) * 450 = 1.8 :=
by sorry

end NUMINAMATH_GPT_percent_of_value_l1768_176827


namespace NUMINAMATH_GPT_determine_x_l1768_176837

theorem determine_x (x : ℕ) (hx : 27^3 + 27^3 + 27^3 = 3^x) : x = 10 :=
sorry

end NUMINAMATH_GPT_determine_x_l1768_176837


namespace NUMINAMATH_GPT_bailey_total_spending_l1768_176848

noncomputable def cost_after_discount : ℝ :=
  let guest_sets := 2
  let master_sets := 4
  let guest_price := 40.0
  let master_price := 50.0
  let discount := 0.20
  let total_cost := (guest_sets * guest_price) + (master_sets * master_price)
  let discount_amount := total_cost * discount
  total_cost - discount_amount

theorem bailey_total_spending : cost_after_discount = 224.0 :=
by
  unfold cost_after_discount
  sorry

end NUMINAMATH_GPT_bailey_total_spending_l1768_176848


namespace NUMINAMATH_GPT_find_value_of_x_squared_plus_inverse_squared_l1768_176820

theorem find_value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x + (1/x) = 2) : x^2 + (1/x^2) = 2 :=
sorry

end NUMINAMATH_GPT_find_value_of_x_squared_plus_inverse_squared_l1768_176820


namespace NUMINAMATH_GPT_Lily_balls_is_3_l1768_176818

-- Definitions from conditions
variable (L : ℕ)

def Frodo_balls := L + 8
def Brian_balls := 2 * (L + 8)

axiom Brian_has_22 : Brian_balls L = 22

-- The goal is to prove that Lily has 3 tennis balls
theorem Lily_balls_is_3 : L = 3 :=
by
  sorry

end NUMINAMATH_GPT_Lily_balls_is_3_l1768_176818


namespace NUMINAMATH_GPT_at_least_two_equal_l1768_176880

theorem at_least_two_equal
  {a b c d : ℝ}
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h₁ : a + b + (1 / (a * b)) = c + d + (1 / (c * d)))
  (h₂ : (1 / a) + (1 / b) + (a * b) = (1 / c) + (1 / d) + (c * d)) :
  a = c ∨ a = d ∨ b = c ∨ b = d ∨ a = b ∨ c = d := by
  sorry

end NUMINAMATH_GPT_at_least_two_equal_l1768_176880


namespace NUMINAMATH_GPT_minimum_employees_for_identical_training_l1768_176851

def languages : Finset String := {"English", "French", "Spanish", "German"}

noncomputable def choose_pairings_count (n k : ℕ) : ℕ :=
Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem minimum_employees_for_identical_training 
  (num_languages : ℕ := 4) 
  (employees_per_pairing : ℕ := 4)
  (pairings : ℕ := choose_pairings_count num_languages 2) 
  (total_employees : ℕ := employees_per_pairing * pairings)
  (minimum_employees : ℕ := total_employees + 1):
  minimum_employees = 25 :=
by
  -- We skip the proof details as per the instructions
  sorry

end NUMINAMATH_GPT_minimum_employees_for_identical_training_l1768_176851


namespace NUMINAMATH_GPT_problem_solution_l1768_176843

-- Definitions of the arithmetic sequence a_n and its common difference and first term
variables (a d : ℝ)

-- Definitions of arithmetic sequence conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

-- Required conditions for the proof
variables (h1 : d ≠ 0) (h2 : a ≠ 0)
variables (h3 : arithmetic_sequence a d 2 * arithmetic_sequence a d 8 = (arithmetic_sequence a d 4) ^ 2)

-- The target theorem to prove
theorem problem_solution : 
  (a + (a + 4 * d) + (a + 8 * d)) / ((a + d) + (a + 2 * d)) = 3 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1768_176843


namespace NUMINAMATH_GPT_double_series_evaluation_l1768_176850

theorem double_series_evaluation :
    (∑' m : ℕ, ∑' n : ℕ, if h : n ≥ m then 1 / (m * n * (m + n + 2)) else 0) = (Real.pi ^ 2) / 6 := sorry

end NUMINAMATH_GPT_double_series_evaluation_l1768_176850


namespace NUMINAMATH_GPT_prob_no_1_or_6_l1768_176854

theorem prob_no_1_or_6 :
  ∀ (a b c : ℕ), (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) →
  (8 / 27 : ℝ) = (4 / 6) * (4 / 6) * (4 / 6) :=
by
  intros a b c h
  sorry

end NUMINAMATH_GPT_prob_no_1_or_6_l1768_176854


namespace NUMINAMATH_GPT_daily_wage_of_c_is_71_l1768_176884

theorem daily_wage_of_c_is_71 (x : ℚ) :
  let a_days := 16
  let b_days := 9
  let c_days := 4
  let total_earnings := 1480
  let wage_ratio_a := 3
  let wage_ratio_b := 4
  let wage_ratio_c := 5
  let total_contribution := a_days * wage_ratio_a * x + b_days * wage_ratio_b * x + c_days * wage_ratio_c * x
  total_contribution = total_earnings →
  c_days * wage_ratio_c * x = 71 := by
  sorry

end NUMINAMATH_GPT_daily_wage_of_c_is_71_l1768_176884


namespace NUMINAMATH_GPT_trigonometric_inequality_l1768_176853

open Real

theorem trigonometric_inequality
  (x y z : ℝ)
  (h1 : 0 < x)
  (h2 : x < y)
  (h3 : y < z)
  (h4 : z < π / 2) :
  π / 2 + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_l1768_176853


namespace NUMINAMATH_GPT_soldiers_first_side_l1768_176814

theorem soldiers_first_side (x : ℤ) (h1 : ∀ s1 : ℤ, s1 = 10)
                           (h2 : ∀ s2 : ℤ, s2 = 8)
                           (h3 : ∀ y : ℤ, y = x - 500)
                           (h4 : (10 * x + 8 * (x - 500)) = 68000) : x = 4000 :=
by
  -- Left blank for Lean to fill in the required proof steps
  sorry

end NUMINAMATH_GPT_soldiers_first_side_l1768_176814


namespace NUMINAMATH_GPT_geometric_progression_common_ratio_l1768_176869

-- Define the problem conditions in Lean 4
theorem geometric_progression_common_ratio (a : ℕ → ℝ) (r : ℝ) (n : ℕ)
  (h_pos : ∀ n, a n > 0) 
  (h_rel : ∀ n, a n = (a (n + 1) + a (n + 2)) / 2 + 2 ) : 
  r = 1 :=
sorry

end NUMINAMATH_GPT_geometric_progression_common_ratio_l1768_176869


namespace NUMINAMATH_GPT_tom_age_ratio_l1768_176861

-- Define the variables and conditions
variables (T N : ℕ)

-- Condition 1: Tom's current age is twice the sum of his children's ages
def children_sum_current : ℤ := T / 2

-- Condition 2: Tom's age N years ago was three times the sum of their ages then
def children_sum_past : ℤ := (T / 2) - 2 * N

-- Main theorem statement proving the ratio T/N = 10 assuming given conditions
theorem tom_age_ratio (h1 : T = 2 * (T / 2)) 
                      (h2 : T - N = 3 * ((T / 2) - 2 * N)) : 
                      T / N = 10 :=
sorry

end NUMINAMATH_GPT_tom_age_ratio_l1768_176861


namespace NUMINAMATH_GPT_students_left_early_l1768_176816

theorem students_left_early :
  let initial_groups := 3
  let students_per_group := 8
  let students_remaining := 22
  let total_students := initial_groups * students_per_group
  total_students - students_remaining = 2 :=
by
  -- Define the initial conditions
  let initial_groups := 3
  let students_per_group := 8
  let students_remaining := 22
  let total_students := initial_groups * students_per_group
  -- Proof (to be completed)
  sorry

end NUMINAMATH_GPT_students_left_early_l1768_176816


namespace NUMINAMATH_GPT_arithmetic_sequence_1005th_term_l1768_176813

theorem arithmetic_sequence_1005th_term (p r : ℤ) 
  (h1 : 11 = p + 2 * r)
  (h2 : 11 + 2 * r = 4 * p - r) :
  (5 + 1004 * 6) = 6029 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_1005th_term_l1768_176813


namespace NUMINAMATH_GPT_illumination_ways_l1768_176802

def ways_to_illuminate_traffic_lights (n : ℕ) : ℕ :=
  3^n

theorem illumination_ways (n : ℕ) : ways_to_illuminate_traffic_lights n = 3 ^ n :=
by
  sorry

end NUMINAMATH_GPT_illumination_ways_l1768_176802


namespace NUMINAMATH_GPT_gcd_problem_l1768_176831

theorem gcd_problem :
  ∃ n : ℕ, (80 ≤ n) ∧ (n ≤ 100) ∧ (n % 9 = 0) ∧ (Nat.gcd n 27 = 9) ∧ (n = 90) :=
by sorry

end NUMINAMATH_GPT_gcd_problem_l1768_176831


namespace NUMINAMATH_GPT_total_money_l1768_176817

theorem total_money (total_coins nickels dimes : ℕ) (val_nickel val_dime : ℕ)
  (h1 : total_coins = 8)
  (h2 : nickels = 2)
  (h3 : total_coins = nickels + dimes)
  (h4 : val_nickel = 5)
  (h5 : val_dime = 10) :
  (nickels * val_nickel + dimes * val_dime) = 70 :=
by
  sorry

end NUMINAMATH_GPT_total_money_l1768_176817


namespace NUMINAMATH_GPT_sin_B_plus_pi_over_6_eq_l1768_176895

noncomputable def sin_b_plus_pi_over_6 (B : ℝ) : ℝ :=
  Real.sin B * (Real.sqrt 3 / 2) + (Real.sqrt (1 - (Real.sin B) ^ 2)) * (1 / 2)

theorem sin_B_plus_pi_over_6_eq :
  ∀ (A B : ℝ) (b c : ℝ),
    A = (2 * Real.pi / 3) →
    b = 1 →
    (1 / 2 * b * c * Real.sin A) = Real.sqrt 3 →
    c = 2 →
    sin_b_plus_pi_over_6 B = (2 * Real.sqrt 7 / 7) :=
by
  intros A B b c hA hb hArea hc
  sorry

end NUMINAMATH_GPT_sin_B_plus_pi_over_6_eq_l1768_176895


namespace NUMINAMATH_GPT_find_age_l1768_176852

open Nat

-- Definition of ages
def Teacher_Zhang_age (z : Nat) := z
def Wang_Bing_age (w : Nat) := w

-- Conditions
axiom teacher_zhang_condition (z w : Nat) : z = 3 * w + 4
axiom age_comparison_condition (z w : Nat) : z - 10 = w + 10

-- Proposition to prove
theorem find_age (z w : Nat) (hz : z = 3 * w + 4) (hw : z - 10 = w + 10) : z = 28 ∧ w = 8 := by
  sorry

end NUMINAMATH_GPT_find_age_l1768_176852


namespace NUMINAMATH_GPT_minimum_cuts_for_10_pieces_l1768_176858

theorem minimum_cuts_for_10_pieces :
  ∃ n : ℕ, (n * (n + 1)) / 2 ≥ 10 ∧ ∀ m < n, (m * (m + 1)) / 2 < 10 := sorry

end NUMINAMATH_GPT_minimum_cuts_for_10_pieces_l1768_176858


namespace NUMINAMATH_GPT_perpendicular_lines_l1768_176874

def line1 (m : ℝ) (x y : ℝ) := m * x - 3 * y - 1 = 0
def line2 (m : ℝ) (x y : ℝ) := (3 * m - 2) * x - m * y + 2 = 0

theorem perpendicular_lines (m : ℝ) :
  (∀ x y : ℝ, line1 m x y) →
  (∀ x y : ℝ, line2 m x y) →
  (∀ x y : ℝ, (m / 3) * ((3 * m - 2) / m) = -1) →
  m = 0 ∨ m = -1/3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1768_176874


namespace NUMINAMATH_GPT_false_statement_l1768_176832

theorem false_statement :
  ¬ (∀ x : ℝ, x^2 + 1 > 3 * x) = (∃ x : ℝ, x^2 + 1 ≤ 3 * x) := sorry

end NUMINAMATH_GPT_false_statement_l1768_176832


namespace NUMINAMATH_GPT_tangent_circle_distance_proof_l1768_176859

noncomputable def tangent_circle_distance (R r : ℝ) (tangent_type : String) : ℝ :=
  if tangent_type = "external" then R + r else R - r

theorem tangent_circle_distance_proof (R r : ℝ) (tangent_type : String) (hR : R = 4) (hr : r = 3) :
  tangent_circle_distance R r tangent_type = 7 ∨ tangent_circle_distance R r tangent_type = 1 := by
  sorry

end NUMINAMATH_GPT_tangent_circle_distance_proof_l1768_176859


namespace NUMINAMATH_GPT_raisins_in_other_three_boxes_l1768_176876

-- Definitions of the known quantities
def total_raisins : ℕ := 437
def box1_raisins : ℕ := 72
def box2_raisins : ℕ := 74

-- The goal is to prove that each of the other three boxes has 97 raisins
theorem raisins_in_other_three_boxes :
  total_raisins - (box1_raisins + box2_raisins) = 3 * 97 :=
by
  sorry

end NUMINAMATH_GPT_raisins_in_other_three_boxes_l1768_176876


namespace NUMINAMATH_GPT_largest_pos_integer_binary_op_l1768_176825

def binary_op (n : ℤ) : ℤ := n - n * 5

theorem largest_pos_integer_binary_op :
  ∃ n : ℕ, binary_op n < 14 ∧ ∀ m : ℕ, binary_op m < 14 → m ≤ 1 :=
sorry

end NUMINAMATH_GPT_largest_pos_integer_binary_op_l1768_176825


namespace NUMINAMATH_GPT_equate_operations_l1768_176871

theorem equate_operations :
  (15 * 5) / (10 + 2) = 3 → 8 / 4 = 2 → ((18 * 6) / (14 + 4) = 6) :=
by
sorry

end NUMINAMATH_GPT_equate_operations_l1768_176871


namespace NUMINAMATH_GPT_neg_p_l1768_176898

-- Let's define the original proposition p
def p : Prop := ∃ x : ℝ, x ≥ 2 ∧ x^2 - 2 * x - 2 > 0

-- Now, we state the problem in Lean as requiring the proof of the negation of p
theorem neg_p : ¬p ↔ ∀ x : ℝ, x ≥ 2 → x^2 - 2 * x - 2 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_neg_p_l1768_176898


namespace NUMINAMATH_GPT_hyperbola_focus_l1768_176804

theorem hyperbola_focus :
    ∃ (f : ℝ × ℝ), f = (-2 - Real.sqrt 6, -2) ∧
    ∀ (x y : ℝ), 2 * x^2 - y^2 + 8 * x - 4 * y - 8 = 0 → 
    ∃ a b h k : ℝ, 
        (a = Real.sqrt 2) ∧ (b = 2) ∧ (h = -2) ∧ (k = -2) ∧
        ((2 * (x + h)^2 - (y + k)^2 = 4) ∧ 
         (x, y) = f) :=
sorry

end NUMINAMATH_GPT_hyperbola_focus_l1768_176804


namespace NUMINAMATH_GPT_cone_sphere_ratio_l1768_176847

theorem cone_sphere_ratio (r h : ℝ) (h_r_ne_zero : r ≠ 0)
  (h_vol_cone : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 := 
by
  sorry

end NUMINAMATH_GPT_cone_sphere_ratio_l1768_176847


namespace NUMINAMATH_GPT_correct_operation_l1768_176887

theorem correct_operation :
  (∀ (a : ℤ), 2 * a - a ≠ 1) ∧
  (∀ (a : ℤ), (a^2)^4 ≠ a^6) ∧
  (∀ (a b : ℤ), (a * b)^2 ≠ a * b^2) ∧
  (∀ (a : ℤ), a^3 * a^2 = a^5) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1768_176887


namespace NUMINAMATH_GPT_largest_N_l1768_176835

-- Definition of the problem conditions
def problem_conditions (n : ℕ) (N : ℕ) (a : Fin (N + 1) → ℝ) : Prop :=
  (n ≥ 2) ∧
  (a 0 + a 1 = -(1 : ℝ) / n) ∧  
  (∀ k : ℕ, 1 ≤ k → k ≤ N - 1 → (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1))

-- The theorem stating that the largest integer N is n
theorem largest_N (n : ℕ) (N : ℕ) (a : Fin (N + 1) → ℝ) :
  problem_conditions n N a → N = n :=
sorry

end NUMINAMATH_GPT_largest_N_l1768_176835


namespace NUMINAMATH_GPT_katie_added_new_songs_l1768_176877

-- Definitions for the conditions
def initial_songs := 11
def deleted_songs := 7
def current_songs := 28

-- Definition of the expected answer
def new_songs_added := current_songs - (initial_songs - deleted_songs)

-- Statement of the problem in Lean
theorem katie_added_new_songs : new_songs_added = 24 :=
by
  sorry

end NUMINAMATH_GPT_katie_added_new_songs_l1768_176877


namespace NUMINAMATH_GPT_find_x_value_l1768_176826

theorem find_x_value :
  let a := (2021 : ℝ)
  let b := (2022 : ℝ)
  ∀ x : ℝ, (a / b - b / a + x = 0) → (x = b / a - a / b) :=
  by
    intros a b x h
    sorry

end NUMINAMATH_GPT_find_x_value_l1768_176826
