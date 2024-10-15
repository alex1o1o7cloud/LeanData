import Mathlib

namespace NUMINAMATH_GPT_inequality_for_any_x_l2038_203827

theorem inequality_for_any_x (a : ℝ) (h : ∀ x : ℝ, |3 * x + 2 * a| + |2 - 3 * x| - |a + 1| > 2) :
  a < -1/3 ∨ a > 5 := 
sorry

end NUMINAMATH_GPT_inequality_for_any_x_l2038_203827


namespace NUMINAMATH_GPT_solution_to_system_of_eqns_l2038_203835

theorem solution_to_system_of_eqns (x y z : ℝ) :
  (x = (2 * z ^ 2) / (1 + z ^ 2) ∧ y = (2 * x ^ 2) / (1 + x ^ 2) ∧ z = (2 * y ^ 2) / (1 + y ^ 2)) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_GPT_solution_to_system_of_eqns_l2038_203835


namespace NUMINAMATH_GPT_eliana_additional_steps_first_day_l2038_203862

variables (x : ℝ)

def eliana_first_day_steps := 200 + x
def eliana_second_day_steps := 2 * eliana_first_day_steps
def eliana_third_day_steps := eliana_second_day_steps + 100
def eliana_total_steps := eliana_first_day_steps + eliana_second_day_steps + eliana_third_day_steps

theorem eliana_additional_steps_first_day : eliana_total_steps = 1600 → x = 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_eliana_additional_steps_first_day_l2038_203862


namespace NUMINAMATH_GPT_alices_favorite_number_l2038_203898

theorem alices_favorite_number :
  ∃ n : ℕ, 80 < n ∧ n ≤ 130 ∧ n % 13 = 0 ∧ n % 3 ≠ 0 ∧ ((n / 100) + (n % 100 / 10) + (n % 10)) % 4 = 0 ∧ n = 130 :=
by
  sorry

end NUMINAMATH_GPT_alices_favorite_number_l2038_203898


namespace NUMINAMATH_GPT_unique_positive_integer_satisfies_condition_l2038_203851

def is_positive_integer (n : ℕ) : Prop := n > 0

def condition (n : ℕ) : Prop := 20 - 5 * n ≥ 15

theorem unique_positive_integer_satisfies_condition :
  ∃! n : ℕ, is_positive_integer n ∧ condition n :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_integer_satisfies_condition_l2038_203851


namespace NUMINAMATH_GPT_min_stamps_needed_l2038_203878

theorem min_stamps_needed {c f : ℕ} (h : 3 * c + 4 * f = 33) : c + f = 9 :=
sorry

end NUMINAMATH_GPT_min_stamps_needed_l2038_203878


namespace NUMINAMATH_GPT_distribute_a_eq_l2038_203828

variable (a b c : ℝ)

theorem distribute_a_eq : a * (a + b - c) = a^2 + a * b - a * c := 
sorry

end NUMINAMATH_GPT_distribute_a_eq_l2038_203828


namespace NUMINAMATH_GPT_calculate_unoccupied_volume_l2038_203849

def tank_length : ℕ := 12
def tank_width : ℕ := 10
def tank_height : ℕ := 8
def tank_volume : ℕ := tank_length * tank_width * tank_height

def water_volume : ℕ := tank_volume / 3
def ice_cube_volume : ℕ := 1
def ice_cubes_count : ℕ := 12
def total_ice_volume : ℕ := ice_cubes_count * ice_cube_volume
def occupied_volume : ℕ := water_volume + total_ice_volume

def unoccupied_volume : ℕ := tank_volume - occupied_volume

theorem calculate_unoccupied_volume : unoccupied_volume = 628 := by
  sorry

end NUMINAMATH_GPT_calculate_unoccupied_volume_l2038_203849


namespace NUMINAMATH_GPT_find_original_radius_l2038_203875

theorem find_original_radius (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_original_radius_l2038_203875


namespace NUMINAMATH_GPT_rhombus_area_l2038_203895

theorem rhombus_area (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 20) : 
  (d1 * d2) / 2 = 160 := by
sorry

end NUMINAMATH_GPT_rhombus_area_l2038_203895


namespace NUMINAMATH_GPT_dice_probabilities_relationship_l2038_203805

theorem dice_probabilities_relationship :
  let p1 := 5 / 18
  let p2 := 11 / 18
  let p3 := 1 / 2
  p1 < p3 ∧ p3 < p2
:= by
  sorry

end NUMINAMATH_GPT_dice_probabilities_relationship_l2038_203805


namespace NUMINAMATH_GPT_cos_alpha_minus_beta_cos_alpha_plus_beta_l2038_203821

variables (α β : Real) (h1 : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2)
           (h2 : Real.tan α * Real.tan β = 13/7)
           (h3 : Real.sin (α - β) = sqrt 5 / 3)

-- Part (1): Prove that cos (α - β) = 2/3
theorem cos_alpha_minus_beta : Real.cos (α - β) = 2 / 3 := by
  have h := h1
  have h := h2
  have h := h3
  sorry

-- Part (2): Prove that cos (α + β) = -1/5
theorem cos_alpha_plus_beta : Real.cos (α + β) = -1 / 5 := by
  have h := h1
  have h := h2
  have h := h3
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_beta_cos_alpha_plus_beta_l2038_203821


namespace NUMINAMATH_GPT_units_digit_of_153_base_3_l2038_203847

theorem units_digit_of_153_base_3 :
  (153 % 3 ^ 1) = 2 := by
sorry

end NUMINAMATH_GPT_units_digit_of_153_base_3_l2038_203847


namespace NUMINAMATH_GPT_smallest_positive_integer_a_l2038_203836

theorem smallest_positive_integer_a (a : ℕ) (hpos : a > 0) :
  (∃ k, 5880 * a = k ^ 2) → a = 15 := 
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_a_l2038_203836


namespace NUMINAMATH_GPT_simplify_expression_l2038_203812

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := 4 * x^3 + 5 * x^2 + 2 * x + 8 - (3 * x^3 - 7 * x^2 + 4 * x - 6)

-- Define the simplified form
def simplified_expr (x : ℝ) : ℝ := x^3 + 12 * x^2 - 2 * x + 14

-- State the theorem
theorem simplify_expression (x : ℝ) : initial_expr x = simplified_expr x :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2038_203812


namespace NUMINAMATH_GPT_find_ordered_pair_l2038_203826

theorem find_ordered_pair (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^y + 1 = y^x) (h2 : 2 * x^y = y^x + 13) : (x = 2 ∧ y = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_ordered_pair_l2038_203826


namespace NUMINAMATH_GPT_discount_percentage_l2038_203843

theorem discount_percentage (original_price sale_price : ℝ) (h₁ : original_price = 128) (h₂ : sale_price = 83.2) :
  (original_price - sale_price) / original_price * 100 = 35 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l2038_203843


namespace NUMINAMATH_GPT_solve_for_x_l2038_203807

theorem solve_for_x (x : ℚ) (h : (7 * x) / (x - 2) + 4 / (x - 2) = 6 / (x - 2)) : x = 2 / 7 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l2038_203807


namespace NUMINAMATH_GPT_distance_between_cities_l2038_203831

theorem distance_between_cities 
  (t : ℝ)
  (h1 : 60 * t = 70 * (t - 1 / 4)) 
  (d : ℝ) : 
  d = 105 := by
sorry

end NUMINAMATH_GPT_distance_between_cities_l2038_203831


namespace NUMINAMATH_GPT_vehicle_wax_initial_amount_l2038_203845

theorem vehicle_wax_initial_amount
  (wax_car wax_suv wax_spilled wax_left original_amount : ℕ)
  (h_wax_car : wax_car = 3)
  (h_wax_suv : wax_suv = 4)
  (h_wax_spilled : wax_spilled = 2)
  (h_wax_left : wax_left = 2)
  (h_total_wax_used : wax_car + wax_suv = 7)
  (h_wax_before_waxing : wax_car + wax_suv + wax_spilled = 9) :
  original_amount = 11 := by
  sorry

end NUMINAMATH_GPT_vehicle_wax_initial_amount_l2038_203845


namespace NUMINAMATH_GPT_tank_fraction_before_gas_added_l2038_203863

theorem tank_fraction_before_gas_added (capacity : ℝ) (added_gasoline : ℝ) (fraction_after : ℝ) (initial_fraction : ℝ) :
  capacity = 42 → added_gasoline = 7 → fraction_after = 9 / 10 → (initial_fraction * capacity + added_gasoline = fraction_after * capacity) → initial_fraction = 733 / 1000 :=
by
  intros h_capacity h_added_gasoline h_fraction_after h_equation
  sorry

end NUMINAMATH_GPT_tank_fraction_before_gas_added_l2038_203863


namespace NUMINAMATH_GPT_ratio_of_q_to_r_l2038_203833

theorem ratio_of_q_to_r
  (P Q R : ℕ)
  (h1 : R = 400)
  (h2 : P + Q + R = 1210)
  (h3 : 5 * Q = 4 * P) :
  Q * 10 = R * 9 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_q_to_r_l2038_203833


namespace NUMINAMATH_GPT_c_geq_one_l2038_203802

open Real

theorem c_geq_one (a b : ℕ) (c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h_eqn : (a + 1) / (b + c) = b / a) : c ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_c_geq_one_l2038_203802


namespace NUMINAMATH_GPT_average_brown_MnMs_l2038_203848

theorem average_brown_MnMs 
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 9)
  (h2 : a2 = 12)
  (h3 : a3 = 8)
  (h4 : a4 = 8)
  (h5 : a5 = 3) : 
  (a1 + a2 + a3 + a4 + a5) / 5 = 8 :=
by
  sorry

end NUMINAMATH_GPT_average_brown_MnMs_l2038_203848


namespace NUMINAMATH_GPT_smallest_value_of_3b_plus_2_l2038_203857

theorem smallest_value_of_3b_plus_2 (b : ℝ) (h : 8 * b^2 + 7 * b + 6 = 5) : (∃ t : ℝ, t = 3 * b + 2 ∧ (∀ x : ℝ, 8 * x^2 + 7 * x + 6 = 5 → x = b → t ≤ 3 * x + 2)) :=
sorry

end NUMINAMATH_GPT_smallest_value_of_3b_plus_2_l2038_203857


namespace NUMINAMATH_GPT_find_m_l2038_203872

theorem find_m (m : ℝ) (h : (1 : ℝ) ^ 2 - m * (1 : ℝ) + 2 = 0) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2038_203872


namespace NUMINAMATH_GPT_ratio_closest_to_10_l2038_203887

theorem ratio_closest_to_10 :
  (⌊(10^3000 + 10^3004 : ℝ) / (10^3001 + 10^3003) + 0.5⌋ : ℝ) = 10 :=
sorry

end NUMINAMATH_GPT_ratio_closest_to_10_l2038_203887


namespace NUMINAMATH_GPT_Laura_running_speed_l2038_203870

noncomputable def running_speed (x : ℝ) : Prop :=
  (15 / (3 * x + 2)) + (4 / x) = 1.5 ∧ x > 0

theorem Laura_running_speed : ∃ (x : ℝ), running_speed x ∧ abs (x - 5.64) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_Laura_running_speed_l2038_203870


namespace NUMINAMATH_GPT_slope_intercept_product_l2038_203801

theorem slope_intercept_product (b m : ℤ) (h1 : b = -3) (h2 : m = 3) : m * b = -9 := by
  sorry

end NUMINAMATH_GPT_slope_intercept_product_l2038_203801


namespace NUMINAMATH_GPT_black_cards_remaining_proof_l2038_203876

def initial_black_cards := 26
def black_cards_taken_out := 4
def black_cards_remaining := initial_black_cards - black_cards_taken_out

theorem black_cards_remaining_proof : black_cards_remaining = 22 := 
by sorry

end NUMINAMATH_GPT_black_cards_remaining_proof_l2038_203876


namespace NUMINAMATH_GPT_John_spent_fraction_toy_store_l2038_203856

variable (weekly_allowance arcade_money toy_store_money candy_store_money : ℝ)
variable (spend_fraction : ℝ)

-- John's conditions
def John_conditions : Prop :=
  weekly_allowance = 3.45 ∧
  arcade_money = 3 / 5 * weekly_allowance ∧
  candy_store_money = 0.92 ∧
  toy_store_money = weekly_allowance - arcade_money - candy_store_money

-- Theorem to prove the fraction spent at the toy store
theorem John_spent_fraction_toy_store :
  John_conditions weekly_allowance arcade_money toy_store_money candy_store_money →
  spend_fraction = toy_store_money / (weekly_allowance - arcade_money) →
  spend_fraction = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_John_spent_fraction_toy_store_l2038_203856


namespace NUMINAMATH_GPT_sample_size_l2038_203889

theorem sample_size (f_c f_o N: ℕ) (h1: f_c = 8) (h2: f_c = 1 / 4 * f_o) (h3: f_c + f_o = N) : N = 40 :=
  sorry

end NUMINAMATH_GPT_sample_size_l2038_203889


namespace NUMINAMATH_GPT_correct_comparison_l2038_203816

-- Definitions of conditions based on the problem 
def hormones_participate : Prop := false 
def enzymes_produced_by_living_cells : Prop := true 
def hormones_produced_by_endocrine : Prop := true 
def endocrine_can_produce_both : Prop := true 
def synthesize_enzymes_not_nec_hormones : Prop := true 
def not_all_proteins : Prop := true 

-- Statement of the equivalence between the correct answer and its proof
theorem correct_comparison :  (¬hormones_participate ∧ enzymes_produced_by_living_cells ∧ hormones_produced_by_endocrine ∧ endocrine_can_produce_both ∧ synthesize_enzymes_not_nec_hormones ∧ not_all_proteins) → (endocrine_can_produce_both) :=
by
  sorry

end NUMINAMATH_GPT_correct_comparison_l2038_203816


namespace NUMINAMATH_GPT_balance_blue_balls_l2038_203841

variables (G Y W R B : ℕ)

axiom green_balance : 3 * G = 6 * B
axiom yellow_balance : 2 * Y = 5 * B
axiom white_balance : 6 * B = 4 * W
axiom red_balance : 4 * R = 10 * B

theorem balance_blue_balls : 5 * G + 3 * Y + 3 * W + 2 * R = 27 * B :=
  by
  sorry

end NUMINAMATH_GPT_balance_blue_balls_l2038_203841


namespace NUMINAMATH_GPT_real_part_of_product_l2038_203825

open Complex

theorem real_part_of_product (α β : ℝ) :
  let z1 := Complex.mk (Real.cos α) (Real.sin α)
  let z2 := Complex.mk (Real.cos β) (Real.sin β)
  Complex.re (z1 * z2) = Real.cos (α + β) :=
by
  let z1 := Complex.mk (Real.cos α) (Real.sin α)
  let z2 := Complex.mk (Real.cos β) (Real.sin β)
  sorry

end NUMINAMATH_GPT_real_part_of_product_l2038_203825


namespace NUMINAMATH_GPT_dispatch_plans_l2038_203894

theorem dispatch_plans (students : Finset ℕ) (h : students.card = 6) :
  ∃ (plans : Finset (Finset ℕ)), plans.card = 180 :=
by
  sorry

end NUMINAMATH_GPT_dispatch_plans_l2038_203894


namespace NUMINAMATH_GPT_polynomial_root_solution_l2038_203884

theorem polynomial_root_solution (a b c : ℝ) (h1 : (2:ℝ)^5 + 4*(2:ℝ)^4 + a*(2:ℝ)^2 = b*(2:ℝ) + 4*c) 
  (h2 : (-2:ℝ)^5 + 4*(-2:ℝ)^4 + a*(-2:ℝ)^2 = b*(-2:ℝ) + 4*c) :
  a = -48 ∧ b = 16 ∧ c = -32 :=
sorry

end NUMINAMATH_GPT_polynomial_root_solution_l2038_203884


namespace NUMINAMATH_GPT_slope_correct_l2038_203864

-- Coordinates of the vertices of the polygon
def vertex_A := (0, 0)
def vertex_B := (0, 4)
def vertex_C := (4, 4)
def vertex_D := (4, 2)
def vertex_E := (6, 2)
def vertex_F := (6, 0)

-- Define the total area of the polygon
def total_area : ℝ := 20

-- Define the slope of the line through the origin dividing the area in half
def slope_line_dividing_area (slope : ℝ) : Prop :=
  ∃ l : ℝ, l = 5 / 3 ∧
  ∃ area_divided : ℝ, area_divided = total_area / 2

-- Prove the slope is 5/3
theorem slope_correct :
  slope_line_dividing_area (5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_slope_correct_l2038_203864


namespace NUMINAMATH_GPT_length_of_segment_l2038_203830

theorem length_of_segment : ∃ (a b : ℝ), (|a - (16 : ℝ)^(1/5)| = 3) ∧ (|b - (16 : ℝ)^(1/5)| = 3) ∧ abs (a - b) = 6 :=
by
  sorry

end NUMINAMATH_GPT_length_of_segment_l2038_203830


namespace NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l2038_203888

variable (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) (d : ℝ)
variable (h₁ : S_n 5 = -15) (h₂ : a_n 2 + a_n 5 = -2)

theorem common_difference_of_arithmetic_sequence :
  d = 4 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_of_arithmetic_sequence_l2038_203888


namespace NUMINAMATH_GPT_area_increase_cost_increase_l2038_203861

-- Given definitions based only on the conditions from part a
def original_length := 60
def original_width := 20
def original_fence_cost_per_foot := 15
def original_perimeter := 2 * (original_length + original_width)
def original_fencing_cost := original_perimeter * original_fence_cost_per_foot

def new_fence_cost_per_foot := 20
def new_square_side := original_perimeter / 4
def new_square_area := new_square_side * new_square_side
def new_fencing_cost := original_perimeter * new_fence_cost_per_foot

-- Proof statements using the conditions and correct answers from part b
theorem area_increase : new_square_area - (original_length * original_width) = 400 := by
  sorry

theorem cost_increase : new_fencing_cost - original_fencing_cost = 800 := by
  sorry

end NUMINAMATH_GPT_area_increase_cost_increase_l2038_203861


namespace NUMINAMATH_GPT_impossible_sequence_l2038_203855

def letters_order : List ℕ := [1, 2, 3, 4, 5]

def is_typing_sequence (order : List ℕ) (seq : List ℕ) : Prop :=
  sorry -- This function will evaluate if a sequence is possible given the order

theorem impossible_sequence : ¬ is_typing_sequence letters_order [4, 5, 2, 3, 1] :=
  sorry

end NUMINAMATH_GPT_impossible_sequence_l2038_203855


namespace NUMINAMATH_GPT_range_of_m_l2038_203868

theorem range_of_m {a b c x0 y0 y1 y2 m : ℝ} (h1 : a ≠ 0)
    (A_on_parabola : y1 = a * m^2 + 4 * a * m + c)
    (B_on_parabola : y2 = a * (m + 2)^2 + 4 * a * (m + 2) + c)
    (C_on_parabola : y0 = a * (-2)^2 + 4 * a * (-2) + c)
    (C_is_vertex : x0 = -2)
    (y_relation : y0 ≥ y2 ∧ y2 > y1) :
    m < -3 := 
sorry

end NUMINAMATH_GPT_range_of_m_l2038_203868


namespace NUMINAMATH_GPT_jasmine_additional_cans_needed_l2038_203893

theorem jasmine_additional_cans_needed
  (n_initial : ℕ)
  (n_lost : ℕ)
  (n_remaining : ℕ)
  (additional_can_coverage : ℕ)
  (n_needed : ℕ) :
  n_initial = 50 →
  n_lost = 4 →
  n_remaining = 36 →
  additional_can_coverage = 2 →
  n_needed = 7 :=
by
  sorry

end NUMINAMATH_GPT_jasmine_additional_cans_needed_l2038_203893


namespace NUMINAMATH_GPT_initial_percentage_rise_l2038_203838

-- Definition of the conditions
def final_price_gain (P : ℝ) (x : ℝ) : Prop :=
  P * (1 + x / 100) * 0.9 * 0.85 = P * 1.03275

-- The statement to be proven
theorem initial_percentage_rise (P : ℝ) (x : ℝ) : final_price_gain P x → x = 35.03 :=
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_initial_percentage_rise_l2038_203838


namespace NUMINAMATH_GPT_initial_ratio_zinc_copper_l2038_203896

theorem initial_ratio_zinc_copper (Z C : ℝ) 
  (h1 : Z + C = 6) 
  (h2 : Z + 8 = 3 * C) : 
  Z / C = 5 / 7 := 
sorry

end NUMINAMATH_GPT_initial_ratio_zinc_copper_l2038_203896


namespace NUMINAMATH_GPT_locus_of_centers_l2038_203859

-- The Lean 4 statement
theorem locus_of_centers (a b : ℝ) 
  (C1 : (x y : ℝ) → x^2 + y^2 = 1)
  (C2 : (x y : ℝ) → (x - 3)^2 + y^2 = 25) :
  4 * a^2 + 4 * b^2 - 52 * a - 169 = 0 :=
sorry

end NUMINAMATH_GPT_locus_of_centers_l2038_203859


namespace NUMINAMATH_GPT_great_dane_weight_l2038_203867

theorem great_dane_weight : 
  ∀ (C P G : ℕ), 
    C + P + G = 439 ∧ P = 3 * C ∧ G = 3 * P + 10 → G = 307 := by
    sorry

end NUMINAMATH_GPT_great_dane_weight_l2038_203867


namespace NUMINAMATH_GPT_number_of_students_l2038_203873

theorem number_of_students (n : ℕ) (bow_cost : ℕ) (vinegar_cost : ℕ) (baking_soda_cost : ℕ) (total_cost : ℕ) :
  bow_cost = 5 → vinegar_cost = 2 → baking_soda_cost = 1 → total_cost = 184 → 8 * n = total_cost → n = 23 :=
by
  intros h_bow h_vinegar h_baking_soda h_total_cost h_equation
  sorry

end NUMINAMATH_GPT_number_of_students_l2038_203873


namespace NUMINAMATH_GPT_man_rate_in_still_water_l2038_203809

theorem man_rate_in_still_water (V_m V_s: ℝ) 
(h1 : V_m + V_s = 19) 
(h2 : V_m - V_s = 11) : 
V_m = 15 := 
by
  sorry

end NUMINAMATH_GPT_man_rate_in_still_water_l2038_203809


namespace NUMINAMATH_GPT_triangle_inequality_for_powers_l2038_203877

theorem triangle_inequality_for_powers (a b c : ℝ) :
  (∀ n : ℕ, (a ^ n + b ^ n > c ^ n)) ↔ (a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b = c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_for_powers_l2038_203877


namespace NUMINAMATH_GPT_reciprocal_of_2023_l2038_203860

theorem reciprocal_of_2023 : 1 / 2023 = (1 : ℚ) / 2023 :=
by sorry

end NUMINAMATH_GPT_reciprocal_of_2023_l2038_203860


namespace NUMINAMATH_GPT_total_telephone_bill_second_month_l2038_203834

theorem total_telephone_bill_second_month
  (F C1 : ℝ) 
  (h1 : F + C1 = 46)
  (h2 : F + 2 * C1 = 76) :
  F + 2 * C1 = 76 :=
by
  sorry

end NUMINAMATH_GPT_total_telephone_bill_second_month_l2038_203834


namespace NUMINAMATH_GPT_find_S2_side_length_l2038_203819

theorem find_S2_side_length 
    (x r : ℝ)
    (h1 : 2 * r + x = 2100)
    (h2 : 3 * x + 300 = 3500)
    : x = 1066.67 := 
sorry

end NUMINAMATH_GPT_find_S2_side_length_l2038_203819


namespace NUMINAMATH_GPT_line_equation_through_origin_and_circle_chord_length_l2038_203813

theorem line_equation_through_origin_and_circle_chord_length 
  (x y : ℝ) 
  (h : x^2 + y^2 - 2 * x - 4 * y + 4 = 0) 
  (chord_length : ℝ) 
  (h_chord : chord_length = 2) 
  : 2 * x - y = 0 := 
sorry

end NUMINAMATH_GPT_line_equation_through_origin_and_circle_chord_length_l2038_203813


namespace NUMINAMATH_GPT_equation_of_line_l2038_203880

theorem equation_of_line (θ : ℝ) (b : ℝ) :
  θ = 135 ∧ b = -1 → (∀ x y : ℝ, x + y + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_line_l2038_203880


namespace NUMINAMATH_GPT_bridget_bought_17_apples_l2038_203874

noncomputable def total_apples (x : ℕ) : Prop :=
  (2 * x / 3) - 5 = 6

theorem bridget_bought_17_apples : ∃ x : ℕ, total_apples x ∧ x = 17 :=
  sorry

end NUMINAMATH_GPT_bridget_bought_17_apples_l2038_203874


namespace NUMINAMATH_GPT_f_at_2_f_shifted_range_f_shifted_l2038_203823

def f (x : ℝ) := x^2 - 2*x + 7

-- 1) Prove that f(2) = 7
theorem f_at_2 : f 2 = 7 := sorry

-- 2) Prove the expressions for f(x-1) and f(x+1)
theorem f_shifted (x : ℝ) : f (x-1) = x^2 - 4*x + 10 ∧ f (x+1) = x^2 + 6 := sorry

-- 3) Prove the range of f(x+1) is [6, +∞)
theorem range_f_shifted : ∀ x, f (x+1) ≥ 6 := sorry

end NUMINAMATH_GPT_f_at_2_f_shifted_range_f_shifted_l2038_203823


namespace NUMINAMATH_GPT_f_seven_point_five_l2038_203803

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom f_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem f_seven_point_five : f 7.5 = -0.5 := by
  sorry

end NUMINAMATH_GPT_f_seven_point_five_l2038_203803


namespace NUMINAMATH_GPT_power_subtraction_divisibility_l2038_203853

theorem power_subtraction_divisibility (N : ℕ) (h : N > 1) : 
  ∃ k : ℕ, (N^2)^2014 - (N^11)^106 = k * (N^6 + N^3 + 1) :=
by
  sorry

end NUMINAMATH_GPT_power_subtraction_divisibility_l2038_203853


namespace NUMINAMATH_GPT_find_a_l2038_203804

def M : Set ℝ := {-1, 0, 1}

def N (a : ℝ) : Set ℝ := {a, a^2}

theorem find_a (a : ℝ) : N a ⊆ M → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2038_203804


namespace NUMINAMATH_GPT_train_length_calculation_l2038_203858

theorem train_length_calculation
  (speed_kmph : ℝ)
  (time_seconds : ℝ)
  (train_length : ℝ)
  (h1 : speed_kmph = 80)
  (h2 : time_seconds = 8.999280057595392)
  (h3 : train_length = (80 * 1000) / 3600 * 8.999280057595392) :
  train_length = 200 := by
  sorry

end NUMINAMATH_GPT_train_length_calculation_l2038_203858


namespace NUMINAMATH_GPT_product_of_sisters_and_brothers_l2038_203890

-- Lucy's family structure
def lucy_sisters : ℕ := 4
def lucy_brothers : ℕ := 6

-- Liam's siblings count
def liam_sisters : ℕ := lucy_sisters + 1  -- Including Lucy herself
def liam_brothers : ℕ := lucy_brothers    -- Excluding himself

-- Prove the product of Liam's sisters and brothers is 25
theorem product_of_sisters_and_brothers : liam_sisters * (liam_brothers - 1) = 25 :=
by
  sorry

end NUMINAMATH_GPT_product_of_sisters_and_brothers_l2038_203890


namespace NUMINAMATH_GPT_bottles_per_crate_l2038_203882

theorem bottles_per_crate (num_bottles total_bottles bottles_not_placed num_crates : ℕ) 
    (h1 : total_bottles = 130)
    (h2 : bottles_not_placed = 10)
    (h3 : num_crates = 10) 
    (h4 : num_bottles = total_bottles - bottles_not_placed) :
    (num_bottles / num_crates) = 12 := 
by 
    sorry

end NUMINAMATH_GPT_bottles_per_crate_l2038_203882


namespace NUMINAMATH_GPT_families_seating_arrangements_l2038_203839

theorem families_seating_arrangements : 
  let factorial := Nat.factorial
  let family_ways := factorial 3
  let bundles := family_ways * family_ways * family_ways
  let bundle_ways := factorial 3
  bundles * bundle_ways = (factorial 3) ^ 4 := by
  sorry

end NUMINAMATH_GPT_families_seating_arrangements_l2038_203839


namespace NUMINAMATH_GPT_total_number_of_cards_l2038_203852

theorem total_number_of_cards (groups : ℕ) (cards_per_group : ℕ) (h_groups : groups = 9) (h_cards_per_group : cards_per_group = 8) : groups * cards_per_group = 72 := by
  sorry

end NUMINAMATH_GPT_total_number_of_cards_l2038_203852


namespace NUMINAMATH_GPT_probability_of_spade_then_king_l2038_203832

theorem probability_of_spade_then_king :
  ( (24 / 104) * (8 / 103) + (2 / 104) * (7 / 103) ) = 103 / 5356 :=
sorry

end NUMINAMATH_GPT_probability_of_spade_then_king_l2038_203832


namespace NUMINAMATH_GPT_beach_ball_properties_l2038_203846

theorem beach_ball_properties :
  let d : ℝ := 18
  let r : ℝ := d / 2
  let surface_area : ℝ := 4 * π * r^2
  let volume : ℝ := (4 / 3) * π * r^3
  surface_area = 324 * π ∧ volume = 972 * π :=
by
  sorry

end NUMINAMATH_GPT_beach_ball_properties_l2038_203846


namespace NUMINAMATH_GPT_cooler1_water_left_l2038_203829

noncomputable def waterLeftInFirstCooler (gallons1 gallons2 : ℝ) (chairs rows : ℕ) (ozSmall ozLarge ozPerGallon : ℝ) : ℝ :=
  let totalChairs := chairs * rows
  let totalSmallOunces := totalChairs * ozSmall
  let initialOunces1 := gallons1 * ozPerGallon
  initialOunces1 - totalSmallOunces

theorem cooler1_water_left :
  waterLeftInFirstCooler 4.5 3.25 12 7 4 8 128 = 240 :=
by
  sorry

end NUMINAMATH_GPT_cooler1_water_left_l2038_203829


namespace NUMINAMATH_GPT_leftover_value_correct_l2038_203897

noncomputable def leftover_value (nickels_per_roll pennies_per_roll : ℕ) (sarah_nickels sarah_pennies tom_nickels tom_pennies : ℕ) : ℚ :=
  let total_nickels := sarah_nickels + tom_nickels
  let total_pennies := sarah_pennies + tom_pennies
  let leftover_nickels := total_nickels % nickels_per_roll
  let leftover_pennies := total_pennies % pennies_per_roll
  (leftover_nickels * 5 + leftover_pennies) / 100

theorem leftover_value_correct :
  leftover_value 40 50 132 245 98 203 = 1.98 := 
by
  sorry

end NUMINAMATH_GPT_leftover_value_correct_l2038_203897


namespace NUMINAMATH_GPT_probability_of_negative_l2038_203871

def set_of_numbers : Set ℤ := {-2, 1, 4, -3, 0}
def negative_numbers : Set ℤ := {-2, -3}
def total_numbers : ℕ := 5
def total_negative_numbers : ℕ := 2

theorem probability_of_negative :
  (total_negative_numbers : ℚ) / (total_numbers : ℚ) = 2 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_negative_l2038_203871


namespace NUMINAMATH_GPT_greatest_integer_not_exceeding_a_l2038_203810

theorem greatest_integer_not_exceeding_a (a : ℝ) (h : 3^a + a^3 = 123) : ⌊a⌋ = 4 :=
sorry

end NUMINAMATH_GPT_greatest_integer_not_exceeding_a_l2038_203810


namespace NUMINAMATH_GPT_part_a_part_b_l2038_203814

-- Conditions
def has_three_classmates_in_any_group_of_ten (students : Fin 60 → Type) : Prop :=
  ∀ (g : Finset (Fin 60)), g.card = 10 → ∃ (a b c : Fin 60), a ∈ g ∧ b ∈ g ∧ c ∈ g ∧ students a = students b ∧ students b = students c

-- Part (a)
theorem part_a (students : Fin 60 → Type) (h : has_three_classmates_in_any_group_of_ten students) : ∃ g : Finset (Fin 60), g.card ≥ 15 ∧ ∀ a b : Fin 60, a ∈ g → b ∈ g → students a = students b :=
sorry

-- Part (b)
theorem part_b (students : Fin 60 → Type) (h : has_three_classmates_in_any_group_of_ten students) : ¬ ∃ g : Finset (Fin 60), g.card ≥ 16 ∧ ∀ a b : Fin 60, a ∈ g → b ∈ g → students a = students b :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l2038_203814


namespace NUMINAMATH_GPT_abc_sum_eq_11sqrt6_l2038_203891

theorem abc_sum_eq_11sqrt6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 :=
sorry

end NUMINAMATH_GPT_abc_sum_eq_11sqrt6_l2038_203891


namespace NUMINAMATH_GPT_value_of_a_l2038_203806

theorem value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - x - 2 < 0 ↔ -2 < x ∧ x < a) → (a = 2 ∨ a = 3 ∨ a = 4) :=
by sorry

end NUMINAMATH_GPT_value_of_a_l2038_203806


namespace NUMINAMATH_GPT_neither_coffee_tea_juice_l2038_203842

open Set

theorem neither_coffee_tea_juice (total : ℕ) (coffee : ℕ) (tea : ℕ) (both_coffee_tea : ℕ)
  (juice : ℕ) (juice_and_tea_not_coffee : ℕ) :
  total = 35 → 
  coffee = 18 → 
  tea = 15 → 
  both_coffee_tea = 7 → 
  juice = 6 → 
  juice_and_tea_not_coffee = 3 →
  (total - ((coffee + tea - both_coffee_tea) + (juice - juice_and_tea_not_coffee))) = 6 :=
sorry

end NUMINAMATH_GPT_neither_coffee_tea_juice_l2038_203842


namespace NUMINAMATH_GPT_range_of_f_l2038_203815

-- Define the function f
def f (x : ℕ) : ℤ := x^2 - 2 * x

-- Define the domain
def domain : Finset ℕ := {0, 1, 2, 3}

-- Define the expected range
def expected_range : Finset ℤ := {-1, 0, 3}

-- State the theorem
theorem range_of_f : (domain.image f) = expected_range := by
  sorry

end NUMINAMATH_GPT_range_of_f_l2038_203815


namespace NUMINAMATH_GPT_circle_equation_l2038_203850

-- Define conditions
def on_parabola (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  x^2 = 4 * y

def tangent_to_y_axis (M : ℝ × ℝ) (r : ℝ) : Prop :=
  let (x, _) := M
  abs x = r

def tangent_to_axis_of_symmetry (M : ℝ × ℝ) (r : ℝ) : Prop :=
  let (_, y) := M
  abs (1 + y) = r

-- Main theorem statement
theorem circle_equation (M : ℝ × ℝ) (r : ℝ) (x y : ℝ)
  (h1 : on_parabola M)
  (h2 : tangent_to_y_axis M r)
  (h3 : tangent_to_axis_of_symmetry M r) :
  (x - M.1)^2 + (y - M.2)^2 = r^2 ↔
  x^2 + y^2 + 4 * M.1 * x - 2 * M.2 * y + 1 = 0 := 
sorry

end NUMINAMATH_GPT_circle_equation_l2038_203850


namespace NUMINAMATH_GPT_find_value_l2038_203886

theorem find_value (number : ℕ) (h : number / 5 + 16 = 58) : number / 15 + 74 = 88 :=
sorry

end NUMINAMATH_GPT_find_value_l2038_203886


namespace NUMINAMATH_GPT_revenue_increase_l2038_203881

theorem revenue_increase (R : ℕ) (r2000 r2003 r2005 : ℝ) (h1 : r2003 = r2000 * 1.50) (h2 : r2005 = r2000 * 1.80) :
  ((r2005 - r2003) / r2003) * 100 = 20 :=
by sorry

end NUMINAMATH_GPT_revenue_increase_l2038_203881


namespace NUMINAMATH_GPT_half_angle_in_second_quadrant_l2038_203844

theorem half_angle_in_second_quadrant (α : ℝ) (h : 180 < α ∧ α < 270) : 90 < α / 2 ∧ α / 2 < 135 := 
by
  sorry

end NUMINAMATH_GPT_half_angle_in_second_quadrant_l2038_203844


namespace NUMINAMATH_GPT_pipe_A_fill_time_l2038_203879

theorem pipe_A_fill_time (t : ℕ) : 
  (∀ x : ℕ, x = 40 → (1 * x) = 40) ∧
  (∀ y : ℕ, y = 30 → (15/40) + ((1/t) + (1/40)) * 15 = 1) ∧ t = 60 :=
sorry

end NUMINAMATH_GPT_pipe_A_fill_time_l2038_203879


namespace NUMINAMATH_GPT_algebraic_expression_no_linear_term_l2038_203885

theorem algebraic_expression_no_linear_term (a : ℝ) :
  (∀ x : ℝ, (x + a) * (x - 1/2) = x^2 - a/2 ↔ a = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_no_linear_term_l2038_203885


namespace NUMINAMATH_GPT_max_mondays_in_59_days_l2038_203800

theorem max_mondays_in_59_days (start_day : ℕ) : ∃ d : ℕ, d ≤ 6 ∧ 
  start_day = d → (d = 0 → ∃ m : ℕ, m = 9) :=
by 
  sorry

end NUMINAMATH_GPT_max_mondays_in_59_days_l2038_203800


namespace NUMINAMATH_GPT_two_digit_number_is_42_l2038_203811

theorem two_digit_number_is_42 (a b : ℕ) (ha : a < 10) (hb : b < 10) (h : 10 * a + b = 42) :
  ((10 * a + b) : ℚ) / (10 * b + a) = 7 / 4 := by
  sorry

end NUMINAMATH_GPT_two_digit_number_is_42_l2038_203811


namespace NUMINAMATH_GPT_books_sold_wednesday_l2038_203854

-- Define the conditions of the problem
def total_books : Nat := 1200
def sold_monday : Nat := 75
def sold_tuesday : Nat := 50
def sold_thursday : Nat := 78
def sold_friday : Nat := 135
def percentage_not_sold : Real := 66.5

-- Define the statement to be proved
theorem books_sold_wednesday : 
  let books_sold := total_books * (1 - percentage_not_sold / 100)
  let known_sales := sold_monday + sold_tuesday + sold_thursday + sold_friday
  books_sold - known_sales = 64 :=
by
  sorry

end NUMINAMATH_GPT_books_sold_wednesday_l2038_203854


namespace NUMINAMATH_GPT_angle_y_value_l2038_203808

theorem angle_y_value (ABC ABD ABE BAE y : ℝ) (h1 : ABC = 180) (h2 : ABD = 66) 
  (h3 : ABE = 114) (h4 : BAE = 31) (h5 : 31 + 114 + y = 180) : y = 35 :=
  sorry

end NUMINAMATH_GPT_angle_y_value_l2038_203808


namespace NUMINAMATH_GPT_remaining_yards_correct_l2038_203892

-- Define the conversion constant
def yards_per_mile: ℕ := 1760

-- Define the conditions
def marathon_in_miles: ℕ := 26
def marathon_in_yards: ℕ := 395
def total_marathons: ℕ := 15

-- Define the function to calculate the remaining yards after conversion
def calculate_remaining_yards (marathon_in_miles marathon_in_yards total_marathons yards_per_mile: ℕ): ℕ :=
  let total_yards := total_marathons * marathon_in_yards
  total_yards % yards_per_mile

-- Statement to prove
theorem remaining_yards_correct :
  calculate_remaining_yards marathon_in_miles marathon_in_yards total_marathons yards_per_mile = 645 :=
  sorry

end NUMINAMATH_GPT_remaining_yards_correct_l2038_203892


namespace NUMINAMATH_GPT_train_length_l2038_203899

/-- Given that the jogger runs at 2.5 m/s,
    the train runs at 12.5 m/s, 
    the jogger is initially 260 meters ahead, 
    and the train takes 38 seconds to pass the jogger,
    prove that the length of the train is 120 meters. -/
theorem train_length (speed_jogger speed_train : ℝ) (initial_distance time_passing : ℝ)
  (hjogger : speed_jogger = 2.5) (htrain : speed_train = 12.5)
  (hinitial : initial_distance = 260) (htime : time_passing = 38) :
  ∃ L : ℝ, L = 120 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l2038_203899


namespace NUMINAMATH_GPT_gcd_2814_1806_l2038_203822

def a := 2814
def b := 1806

theorem gcd_2814_1806 : Nat.gcd a b = 42 :=
by
  sorry

end NUMINAMATH_GPT_gcd_2814_1806_l2038_203822


namespace NUMINAMATH_GPT_num_ways_to_select_officers_l2038_203840

def ways_to_select_five_officers (n : ℕ) (k : ℕ) : ℕ :=
  (List.range' (n - k + 1) k).foldl (λ acc x => acc * x) 1

theorem num_ways_to_select_officers :
  ways_to_select_five_officers 12 5 = 95040 :=
by
  -- By definition of ways_to_select_five_officers, this is equivalent to 12 * 11 * 10 * 9 * 8.
  sorry

end NUMINAMATH_GPT_num_ways_to_select_officers_l2038_203840


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l2038_203866

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 4 + a 6 + a 8 + a 10 + a 12 = 60)
  (h2 : ∀ n, a (n + 1) = a n + d) :
  a 7 - (1 / 3) * a 5 = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l2038_203866


namespace NUMINAMATH_GPT_ellipse_equation_l2038_203824

theorem ellipse_equation
  (x y t : ℝ)
  (h1 : x = (3 * (Real.sin t - 2)) / (3 - Real.cos t))
  (h2 : y = (4 * (Real.cos t - 6)) / (3 - Real.cos t))
  (h3 : ∀ t : ℝ, (Real.cos t)^2 + (Real.sin t)^2 = 1) :
  ∃ (A B C D E F : ℤ), (9 * x^2 + 36 * x * y + 9 * y^2 + 216 * x + 432 * y + 1440 = 0) ∧ 
  (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd A B) C) D) E) F = 1) ∧
  (|A| + |B| + |C| + |D| + |E| + |F| = 2142) :=
sorry

end NUMINAMATH_GPT_ellipse_equation_l2038_203824


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l2038_203869

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 + a 4 = 4)
  (h2 : a 2 * a 3 = 3)
  (hS : ∀ n, S n = n * (a 1 + a n) / 2):
  (a 1 = -1 ∧ (∀ n, a n = 2 * n - 3) ∧ (∀ n, S n = n^2 - 2 * n)) ∨ 
  (a 1 = 5 ∧ (∀ n, a n = 7 - 2 * n) ∧ (∀ n, S n = 6 * n - n^2)) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_l2038_203869


namespace NUMINAMATH_GPT_fewest_handshakes_organizer_l2038_203837

theorem fewest_handshakes_organizer (n k : ℕ) (h : k < n) 
  (total_handshakes: n*(n-1)/2 + k = 406) :
  k = 0 :=
sorry

end NUMINAMATH_GPT_fewest_handshakes_organizer_l2038_203837


namespace NUMINAMATH_GPT_find_least_positive_x_l2038_203817

theorem find_least_positive_x :
  ∃ x : ℕ, 0 < x ∧ (x + 5713) % 15 = 1847 % 15 ∧ x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_least_positive_x_l2038_203817


namespace NUMINAMATH_GPT_frost_cakes_total_l2038_203883

-- Conditions
def Cagney_time := 60 -- seconds per cake
def Lacey_time := 40  -- seconds per cake
def total_time := 10 * 60 -- 10 minutes in seconds

-- The theorem to prove
theorem frost_cakes_total (Cagney_time Lacey_time total_time : ℕ) (h1 : Cagney_time = 60) (h2 : Lacey_time = 40) (h3 : total_time = 600):
  (total_time / (Cagney_time * Lacey_time / (Cagney_time + Lacey_time))) = 25 :=
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_frost_cakes_total_l2038_203883


namespace NUMINAMATH_GPT_total_paintable_area_correct_l2038_203865

-- Define the conditions
def warehouse_width := 12
def warehouse_length := 15
def warehouse_height := 7

def window_count_per_longer_wall := 3
def window_width := 2
def window_height := 3

-- Define areas for walls, ceiling, and floor
def area_wall_1 := warehouse_width * warehouse_height
def area_wall_2 := warehouse_length * warehouse_height
def window_area := window_width * window_height
def window_total_area := window_count_per_longer_wall * window_area
def area_wall_2_paintable := 2 * (area_wall_2 - window_total_area) -- both inside and outside
def area_ceiling := warehouse_width * warehouse_length
def area_floor := warehouse_width * warehouse_length

-- Total paintable area calculation
def total_paintable_area := 2 * area_wall_1 + area_wall_2_paintable + area_ceiling + area_floor

-- Final proof statement
theorem total_paintable_area_correct : total_paintable_area = 876 := by
  sorry

end NUMINAMATH_GPT_total_paintable_area_correct_l2038_203865


namespace NUMINAMATH_GPT_rhombus_area_l2038_203818

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 12) : (d1 * d2) / 2 = 90 := by
  sorry

end NUMINAMATH_GPT_rhombus_area_l2038_203818


namespace NUMINAMATH_GPT_lottery_profit_l2038_203820

-- Definitions

def Prob_A := (1:ℚ) / 5
def Prob_B := (4:ℚ) / 15
def Prob_C := (1:ℚ) / 5
def Prob_D := (2:ℚ) / 15
def Prob_E := (1:ℚ) / 5

def customers := 300

def first_prize_value := 9
def second_prize_value := 3
def third_prize_value := 1

-- Proof Problem Statement

theorem lottery_profit : 
  (first_prize_category == "D") ∧ 
  (second_prize_category == "B") ∧ 
  (300 * 3 - ((300 * Prob_D) * 9 + (300 * Prob_B) * 3 + (300 * (Prob_A + Prob_C + Prob_E)) * 1)) == 120 :=
by 
  -- Insert mathematical proof here using given probabilities and conditions
  sorry

end NUMINAMATH_GPT_lottery_profit_l2038_203820
