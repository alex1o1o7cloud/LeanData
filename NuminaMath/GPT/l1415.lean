import Mathlib

namespace NUMINAMATH_GPT_cost_price_per_meter_of_cloth_l1415_141557

theorem cost_price_per_meter_of_cloth
  (meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) (total_profit : ℕ) (cost_price : ℕ)
  (meters_eq : meters = 80)
  (selling_price_eq : selling_price = 10000)
  (profit_per_meter_eq : profit_per_meter = 7)
  (total_profit_eq : total_profit = profit_per_meter * meters)
  (selling_price_calc : selling_price = cost_price + total_profit)
  (cost_price_calc : cost_price = selling_price - total_profit)
  : (selling_price - total_profit) / meters = 118 :=
by
  -- here we would provide the proof, but we skip it with sorry
  sorry

end NUMINAMATH_GPT_cost_price_per_meter_of_cloth_l1415_141557


namespace NUMINAMATH_GPT_problem_statement_l1415_141562

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem problem_statement : f (f (f (f (f (f 2))))) = 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1415_141562


namespace NUMINAMATH_GPT_find_certain_number_l1415_141590

theorem find_certain_number (h1 : 2994 / 14.5 = 171) (h2 : ∃ x : ℝ, x / 1.45 = 17.1) : ∃ x : ℝ, x = 24.795 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1415_141590


namespace NUMINAMATH_GPT_total_weight_full_l1415_141592

theorem total_weight_full {x y p q : ℝ}
    (h1 : x + (3/4) * y = p)
    (h2 : x + (1/3) * y = q) :
    x + y = (8/5) * p - (3/5) * q :=
by
  sorry

end NUMINAMATH_GPT_total_weight_full_l1415_141592


namespace NUMINAMATH_GPT_alcohol_solution_contradiction_l1415_141548

theorem alcohol_solution_contradiction (initial_volume : ℕ) (added_water : ℕ) 
                                        (final_volume : ℕ) (final_concentration : ℕ) 
                                        (initial_concentration : ℕ) : 
                                        initial_volume = 75 → added_water = 50 → 
                                        final_volume = initial_volume + added_water → 
                                        final_concentration = 45 → 
                                        ¬ (initial_concentration * initial_volume = final_concentration * final_volume) :=
by 
  intro h_initial_volume h_added_water h_final_volume h_final_concentration
  sorry

end NUMINAMATH_GPT_alcohol_solution_contradiction_l1415_141548


namespace NUMINAMATH_GPT_sum_of_smallest_ns_l1415_141515

theorem sum_of_smallest_ns : ∀ n1 n2 : ℕ, (n1 ≡ 1 [MOD 4] ∧ n1 ≡ 2 [MOD 7]) ∧ (n2 ≡ 1 [MOD 4] ∧ n2 ≡ 2 [MOD 7]) ∧ n1 < n2 →
  n1 = 9 ∧ n2 = 37 → (n1 + n2 = 46) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_smallest_ns_l1415_141515


namespace NUMINAMATH_GPT_parabola_tangent_perpendicular_m_eq_one_parabola_min_MF_NF_l1415_141596

open Real

theorem parabola_tangent_perpendicular_m_eq_one (k : ℝ) (hk : k > 0) :
  (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁^2 = 4 * y₁) ∧ (x₂^2 = 4 * y₂) ∧ (y₁ = k * x₁ + m) ∧ (y₂ = k * x₂ + m) ∧ ((x₁ / 2) * (x₂ / 2) = -1)) → m = 1 :=
sorry

theorem parabola_min_MF_NF (k : ℝ) (hk : k > 0) :
  (m = 2) → 
  (∃ x₁ x₂ y₁ y₂ : ℝ, (x₁^2 = 4 * y₁) ∧ (x₂^2 = 4 * y₂) ∧ (y₁ = k * x₁ + 2) ∧ (y₂ = k * x₂ + 2) ∧ |(y₁ + 1) * (y₂ + 1)| ≥ 9) :=
sorry

end NUMINAMATH_GPT_parabola_tangent_perpendicular_m_eq_one_parabola_min_MF_NF_l1415_141596


namespace NUMINAMATH_GPT_value_of_3W5_l1415_141537

-- Define the operation W
def W (a b : ℝ) : ℝ := b + 15 * a - a^3

-- State the theorem to prove
theorem value_of_3W5 : W 3 5 = 23 := by
    sorry

end NUMINAMATH_GPT_value_of_3W5_l1415_141537


namespace NUMINAMATH_GPT_smallest_prime_divides_l1415_141587

theorem smallest_prime_divides (p : ℕ) (a : ℕ) 
  (h1 : Prime p) (h2 : p > 100) (h3 : a > 1) (h4 : p ∣ (a^89 - 1) / (a - 1)) :
  p = 179 := 
sorry

end NUMINAMATH_GPT_smallest_prime_divides_l1415_141587


namespace NUMINAMATH_GPT_find_a_plus_b_l1415_141578

def star (a b : ℕ) : ℕ := a^b + a + b

theorem find_a_plus_b (a b : ℕ) (h2a : 2 ≤ a) (h2b : 2 ≤ b) (h_ab : star a b = 20) :
  a + b = 6 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l1415_141578


namespace NUMINAMATH_GPT_discount_difference_l1415_141598

theorem discount_difference (p : ℝ) (single_discount first_discount second_discount : ℝ) :
    p = 12000 →
    single_discount = 0.45 →
    first_discount = 0.35 →
    second_discount = 0.10 →
    (p * (1 - single_discount) - p * (1 - first_discount) * (1 - second_discount) = 420) := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_discount_difference_l1415_141598


namespace NUMINAMATH_GPT_total_crayons_l1415_141579

-- Definitions for conditions
def boxes : Nat := 7
def crayons_per_box : Nat := 5

-- Statement that needs to be proved
theorem total_crayons : boxes * crayons_per_box = 35 := by
  sorry

end NUMINAMATH_GPT_total_crayons_l1415_141579


namespace NUMINAMATH_GPT_hyperbola_range_m_l1415_141571

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (m + 2 > 0 ∧ m - 2 < 0) ∧ (x^2 / (m + 2) + y^2 / (m - 2) = 1)) ↔ (-2 < m ∧ m < 2) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_range_m_l1415_141571


namespace NUMINAMATH_GPT_speed_of_other_train_l1415_141513

theorem speed_of_other_train :
  ∀ (d : ℕ) (v1 v2 : ℕ), d = 120 → v1 = 30 → 
    ∀ (d_remaining : ℕ), d_remaining = 70 → 
    v1 + v2 = d_remaining → 
    v2 = 40 :=
by
  intros d v1 v2 h_d h_v1 d_remaining h_d_remaining h_rel_speed
  sorry

end NUMINAMATH_GPT_speed_of_other_train_l1415_141513


namespace NUMINAMATH_GPT_division_addition_l1415_141536

theorem division_addition (n : ℕ) (h : 32 - 16 = n * 4) : n / 4 + 16 = 17 :=
by 
  sorry

end NUMINAMATH_GPT_division_addition_l1415_141536


namespace NUMINAMATH_GPT_complement_M_l1415_141522

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}
def C (s : Set ℝ) : Set ℝ := sᶜ -- complement of a set

theorem complement_M :
  C M = {x : ℝ | x < -2 ∨ x > 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_M_l1415_141522


namespace NUMINAMATH_GPT_remainder_19_pow_19_plus_19_mod_20_l1415_141550

theorem remainder_19_pow_19_plus_19_mod_20 : (19^19 + 19) % 20 = 18 := 
by
  sorry

end NUMINAMATH_GPT_remainder_19_pow_19_plus_19_mod_20_l1415_141550


namespace NUMINAMATH_GPT_arithmetic_sequence_a20_l1415_141564

theorem arithmetic_sequence_a20 (a : ℕ → ℝ) (d : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 + a 3 + a 5 = 18)
  (h3 : a 2 + a 4 + a 6 = 24) :
  a 20 = 40 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a20_l1415_141564


namespace NUMINAMATH_GPT_tangent_line_equation_l1415_141524

theorem tangent_line_equation (P : ℝ × ℝ) (hP : P.2 = P.1^2)
  (h_perpendicular : ∃ k : ℝ, k * -1/2 = -1) : 
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = -1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_equation_l1415_141524


namespace NUMINAMATH_GPT_trajectory_moving_point_hyperbola_l1415_141551

theorem trajectory_moving_point_hyperbola {n m : ℝ} (h_neg_n : n < 0) :
    (∃ y < 0, (y^2 = 16) ∧ (m^2 = (n^2 / 4 - 4))) ↔ ( ∃ (y : ℝ), (y^2 / 16) - (m^2 / 4) = 1 ∧ y < 0 ) := 
sorry

end NUMINAMATH_GPT_trajectory_moving_point_hyperbola_l1415_141551


namespace NUMINAMATH_GPT_probability_calc_l1415_141543

noncomputable def probability_no_distinct_positive_real_roots : ℚ :=
  let pairs_count := 169
  let valid_pairs_count := 17
  1 - (valid_pairs_count / pairs_count : ℚ)

theorem probability_calc :
  probability_no_distinct_positive_real_roots = 152 / 169 := by sorry

end NUMINAMATH_GPT_probability_calc_l1415_141543


namespace NUMINAMATH_GPT_frank_sales_quota_l1415_141554

theorem frank_sales_quota (x : ℕ) :
  (3 * x + 12 + 23 = 50) → x = 5 :=
by sorry

end NUMINAMATH_GPT_frank_sales_quota_l1415_141554


namespace NUMINAMATH_GPT_find_q_l1415_141570

theorem find_q (p q : ℝ) (h : ∀ x : ℝ, (x^2 + p * x + q) ≥ 1) : q = 1 + (p^2 / 4) :=
sorry

end NUMINAMATH_GPT_find_q_l1415_141570


namespace NUMINAMATH_GPT_intersection_points_l1415_141597

-- Define the four line equations
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 2
def line4 (x y : ℝ) : Prop := 5 * x - 15 * y = 15

-- State the theorem for intersection points
theorem intersection_points : 
  (line1 (18/11) (13/11) ∧ line2 (18/11) (13/11)) ∧ 
  (line2 (21/11) (8/11) ∧ line3 (21/11) (8/11)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l1415_141597


namespace NUMINAMATH_GPT_cone_shorter_height_ratio_l1415_141518

theorem cone_shorter_height_ratio 
  (circumference : ℝ) (original_height : ℝ) (volume_shorter_cone : ℝ) 
  (shorter_height : ℝ) (radius : ℝ) :
  circumference = 24 * Real.pi ∧ 
  original_height = 40 ∧ 
  volume_shorter_cone = 432 * Real.pi ∧ 
  2 * Real.pi * radius = circumference ∧ 
  volume_shorter_cone = (1 / 3) * Real.pi * radius^2 * shorter_height
  → shorter_height / original_height = 9 / 40 :=
by
  sorry

end NUMINAMATH_GPT_cone_shorter_height_ratio_l1415_141518


namespace NUMINAMATH_GPT_find_higher_selling_price_l1415_141566

-- Define the constants and initial conditions
def cost_price : ℕ := 200
def selling_price_1 : ℕ := 340
def gain_1 : ℕ := selling_price_1 - cost_price
def new_gain : ℕ := gain_1 + gain_1 * 5 / 100

-- Define the problem statement
theorem find_higher_selling_price : 
  ∀ P : ℕ, P = cost_price + new_gain → P = 347 :=
by
  intro P
  intro h
  sorry

end NUMINAMATH_GPT_find_higher_selling_price_l1415_141566


namespace NUMINAMATH_GPT_g_at_pi_over_3_l1415_141540

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.cos (ω * x + φ)

noncomputable def g (x : ℝ) (ω φ : ℝ) : ℝ := 3 * Real.sin (ω * x + φ) - 1

theorem g_at_pi_over_3 (ω φ : ℝ) :
  (∀ x : ℝ, f (π / 3 + x) ω φ = f (π / 3 - x) ω φ) →
  g (π / 3) ω φ = -1 :=
by sorry

end NUMINAMATH_GPT_g_at_pi_over_3_l1415_141540


namespace NUMINAMATH_GPT_train_car_passengers_l1415_141517

theorem train_car_passengers (x : ℕ) (h : 60 * x = 732 + 228) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_train_car_passengers_l1415_141517


namespace NUMINAMATH_GPT_proof_correctness_l1415_141533

-- Define the new operation
def new_op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Definitions for the conclusions
def conclusion_1 : Prop := new_op 1 (-2) = -8
def conclusion_2 : Prop := ∀ a b : ℝ, new_op a b = new_op b a
def conclusion_3 : Prop := ∀ a b : ℝ, new_op a b = 0 → a = 0
def conclusion_4 : Prop := ∀ a b : ℝ, a + b = 0 → (new_op a a + new_op b b = 8 * a^2)

-- Specify the correct conclusions
def correct_conclusions : Prop := conclusion_1 ∧ conclusion_2 ∧ ¬conclusion_3 ∧ conclusion_4

-- State the theorem
theorem proof_correctness : correct_conclusions := by
  sorry

end NUMINAMATH_GPT_proof_correctness_l1415_141533


namespace NUMINAMATH_GPT_water_flow_total_l1415_141525

theorem water_flow_total
  (R1 R2 R3 : ℕ)
  (h1 : R2 = 36)
  (h2 : R2 = (3 / 2) * R1)
  (h3 : R3 = (5 / 4) * R2)
  : R1 + R2 + R3 = 105 :=
sorry

end NUMINAMATH_GPT_water_flow_total_l1415_141525


namespace NUMINAMATH_GPT_find_stream_speed_l1415_141531

theorem find_stream_speed (b s : ℝ) 
  (h1 : b + s = 10) 
  (h2 : b - s = 8) : s = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_stream_speed_l1415_141531


namespace NUMINAMATH_GPT_rowing_distance_l1415_141530

def man_rowing_speed_still_water : ℝ := 10
def stream_speed : ℝ := 8
def rowing_time_downstream : ℝ := 5
def effective_speed_downstream : ℝ := man_rowing_speed_still_water + stream_speed

theorem rowing_distance :
  effective_speed_downstream * rowing_time_downstream = 90 := 
by 
  sorry

end NUMINAMATH_GPT_rowing_distance_l1415_141530


namespace NUMINAMATH_GPT_false_statement_divisibility_l1415_141593

-- Definitions for the divisibility conditions
def divisible_by (a b : ℕ) : Prop := ∃ k, b = a * k

-- The problem statement
theorem false_statement_divisibility (N : ℕ) :
  (divisible_by 2 N ∧ divisible_by 4 N ∧ divisible_by 12 N ∧ ¬ divisible_by 24 N) →
  (¬ divisible_by 24 N) :=
by
  -- The proof will need to be filled in here
  sorry

end NUMINAMATH_GPT_false_statement_divisibility_l1415_141593


namespace NUMINAMATH_GPT_least_prime_factor_of_5pow6_minus_5pow4_l1415_141586

def least_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then (Nat.minFac n) else 0

theorem least_prime_factor_of_5pow6_minus_5pow4 : least_prime_factor (5^6 - 5^4) = 2 := by
  sorry

end NUMINAMATH_GPT_least_prime_factor_of_5pow6_minus_5pow4_l1415_141586


namespace NUMINAMATH_GPT_total_expenditure_now_l1415_141567

-- Define the conditions in Lean
def original_student_count : ℕ := 100
def additional_students : ℕ := 25
def decrease_in_average_expenditure : ℤ := 10
def increase_in_total_expenditure : ℤ := 500

-- Let's denote the original average expenditure per student as A rupees
variable (A : ℤ)

-- Define the old and new expenditures
def original_total_expenditure := original_student_count * A
def new_average_expenditure := A - decrease_in_average_expenditure
def new_total_expenditure := (original_student_count + additional_students) * new_average_expenditure

-- The theorem to prove
theorem total_expenditure_now :
  new_total_expenditure A - original_total_expenditure A = increase_in_total_expenditure →
  new_total_expenditure A = 7500 :=
by
  sorry

end NUMINAMATH_GPT_total_expenditure_now_l1415_141567


namespace NUMINAMATH_GPT_probability_digit_9_in_3_over_11_is_zero_l1415_141556

-- Define the repeating block of the fraction 3/11
def repeating_block_3_over_11 : List ℕ := [2, 7]

-- Define the function to count the occurrences of a digit in a list
def count_occurrences (digit : ℕ) (lst : List ℕ) : ℕ :=
  lst.count digit

-- Define the probability function
def probability_digit_9_in_3_over_11 : ℚ :=
  (count_occurrences 9 repeating_block_3_over_11) / repeating_block_3_over_11.length

-- Theorem statement
theorem probability_digit_9_in_3_over_11_is_zero : 
  probability_digit_9_in_3_over_11 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_probability_digit_9_in_3_over_11_is_zero_l1415_141556


namespace NUMINAMATH_GPT_second_storm_duration_l1415_141505

theorem second_storm_duration
  (x y : ℕ)
  (h1 : x + y = 45)
  (h2 : 30 * x + 15 * y = 975) :
  y = 25 := 
sorry

end NUMINAMATH_GPT_second_storm_duration_l1415_141505


namespace NUMINAMATH_GPT_circle_radii_l1415_141500

noncomputable def smaller_circle_radius (r : ℝ) :=
  r = 4

noncomputable def larger_circle_radius (r : ℝ) :=
  r = 9

theorem circle_radii (r : ℝ) (h1 : ∀ (r: ℝ), (r + 5) - r = 5) (h2 : ∀ (r: ℝ), 2.4 * r = 2.4 * r):
  smaller_circle_radius r → larger_circle_radius (r + 5) :=
by
  sorry

end NUMINAMATH_GPT_circle_radii_l1415_141500


namespace NUMINAMATH_GPT_mowing_time_l1415_141595

theorem mowing_time (length width: ℝ) (swath_width_overlap_rate: ℝ)
                    (walking_speed: ℝ) (ft_per_inch: ℝ)
                    (length_eq: length = 100)
                    (width_eq: width = 120)
                    (swath_eq: swath_width_overlap_rate = 24)
                    (walking_eq: walking_speed = 4500)
                    (conversion_eq: ft_per_inch = 1/12) :
                    (length / walking_speed) * (width / (swath_width_overlap_rate * ft_per_inch)) = 1.33 :=
by
    rw [length_eq, width_eq, swath_eq, walking_eq, conversion_eq]
    exact sorry

end NUMINAMATH_GPT_mowing_time_l1415_141595


namespace NUMINAMATH_GPT_middle_schoolers_count_l1415_141532

theorem middle_schoolers_count (total_students : ℕ) (fraction_girls : ℚ) 
  (primary_girls_fraction : ℚ) (primary_boys_fraction : ℚ) 
  (num_girls : ℕ) (num_boys: ℕ) (primary_grade_girls : ℕ) 
  (primary_grade_boys : ℕ) :
  total_students = 800 →
  fraction_girls = 5 / 8 →
  primary_girls_fraction = 7 / 10 →
  primary_boys_fraction = 2 / 5 →
  num_girls = fraction_girls * total_students →
  num_boys = total_students - num_girls →
  primary_grade_girls = primary_girls_fraction * num_girls →
  primary_grade_boys = primary_boys_fraction * num_boys →
  total_students - (primary_grade_girls + primary_grade_boys) = 330 :=
by
  intros
  sorry

end NUMINAMATH_GPT_middle_schoolers_count_l1415_141532


namespace NUMINAMATH_GPT_base_7_to_base_10_l1415_141572

theorem base_7_to_base_10 :
  (3 * 7^2 + 2 * 7^1 + 1 * 7^0) = 162 :=
by
  sorry

end NUMINAMATH_GPT_base_7_to_base_10_l1415_141572


namespace NUMINAMATH_GPT_allergic_reaction_probability_is_50_percent_l1415_141582

def can_have_allergic_reaction (choice : String) : Prop :=
  choice = "peanut_butter"

def percentage_of_allergic_reaction :=
  let total_peanut_butter := 40 + 30
  let total_cookies := 40 + 50 + 30 + 20
  (total_peanut_butter : Float) / (total_cookies : Float) * 100

theorem allergic_reaction_probability_is_50_percent :
  percentage_of_allergic_reaction = 50 := sorry

end NUMINAMATH_GPT_allergic_reaction_probability_is_50_percent_l1415_141582


namespace NUMINAMATH_GPT_round_table_arrangement_l1415_141501

theorem round_table_arrangement :
  ∀ (n : ℕ), n = 10 → (∃ factorial_value : ℕ, factorial_value = Nat.factorial (n - 1) ∧ factorial_value = 362880) := by
  sorry

end NUMINAMATH_GPT_round_table_arrangement_l1415_141501


namespace NUMINAMATH_GPT_first_day_of_month_is_tuesday_l1415_141591

theorem first_day_of_month_is_tuesday (day23_is_wednesday : (23 % 7 = 3)) : (1 % 7 = 2) :=
sorry

end NUMINAMATH_GPT_first_day_of_month_is_tuesday_l1415_141591


namespace NUMINAMATH_GPT_integrate_diff_eq_l1415_141508

noncomputable def particular_solution (x y : ℝ) : Prop :=
  (y^2 - x^2) / 2 + Real.exp y - Real.log ((x + Real.sqrt (1 + x^2)) / (2 + Real.sqrt 5)) = Real.exp 1 - 3 / 2

theorem integrate_diff_eq (x y : ℝ) :
  (∀ x y : ℝ, y' = (x * Real.sqrt (1 + x^2) + 1) / (Real.sqrt (1 + x^2) * (y + Real.exp y))) → 
  (∃ x0 y0 : ℝ, x0 = 2 ∧ y0 = 1) → 
  particular_solution x y :=
sorry

end NUMINAMATH_GPT_integrate_diff_eq_l1415_141508


namespace NUMINAMATH_GPT_total_books_l1415_141576

-- Define the number of books Tim has
def TimBooks : ℕ := 44

-- Define the number of books Sam has
def SamBooks : ℕ := 52

-- Statement to prove that the total number of books is 96
theorem total_books : TimBooks + SamBooks = 96 := by
  sorry

end NUMINAMATH_GPT_total_books_l1415_141576


namespace NUMINAMATH_GPT_solve_for_y_l1415_141514

theorem solve_for_y (y : ℝ) (h : y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) : y = -4 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_y_l1415_141514


namespace NUMINAMATH_GPT_find_f1_l1415_141534

theorem find_f1 (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) + (-x) ^ 2 = -(f x + x ^ 2))
  (h2 : ∀ x, f (-x) + 2 ^ (-x) = f x + 2 ^ x) :
  f 1 = -7 / 4 := by
sorry

end NUMINAMATH_GPT_find_f1_l1415_141534


namespace NUMINAMATH_GPT_total_vehicle_wheels_in_parking_lot_l1415_141502

def vehicles_wheels := (1 * 4) + (1 * 4) + (8 * 4) + (4 * 2) + (3 * 6) + (2 * 4) + (1 * 8) + (2 * 3)

theorem total_vehicle_wheels_in_parking_lot : vehicles_wheels = 88 :=
by {
    sorry
}

end NUMINAMATH_GPT_total_vehicle_wheels_in_parking_lot_l1415_141502


namespace NUMINAMATH_GPT_john_total_money_after_3_years_l1415_141589

def principal : ℝ := 1000
def rate : ℝ := 0.1
def time : ℝ := 3

/-
  We need to prove that the total money after 3 years is $1300
-/
theorem john_total_money_after_3_years (principal : ℝ) (rate : ℝ) (time : ℝ) :
  principal + (principal * rate * time) = 1300 := by
  sorry

end NUMINAMATH_GPT_john_total_money_after_3_years_l1415_141589


namespace NUMINAMATH_GPT_diana_owes_amount_l1415_141599

def principal : ℝ := 75
def rate : ℝ := 0.07
def time : ℝ := 1
def interest : ℝ := principal * rate * time
def total_amount_owed : ℝ := principal + interest

theorem diana_owes_amount :
  total_amount_owed = 80.25 :=
by
  sorry

end NUMINAMATH_GPT_diana_owes_amount_l1415_141599


namespace NUMINAMATH_GPT_range_of_a_l1415_141558

theorem range_of_a (A B : Set ℝ) (a : ℝ)
  (hA : A = {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5})
  (hB : B = {x | 3 ≤ x ∧ x ≤ 22}) :
  A ⊆ (A ∩ B) ↔ (1 ≤ a ∧ a ≤ 9) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1415_141558


namespace NUMINAMATH_GPT_joohyeon_snack_count_l1415_141577

theorem joohyeon_snack_count
  (c s : ℕ)
  (h1 : 300 * c + 500 * s = 3000)
  (h2 : c + s = 8) :
  s = 3 :=
sorry

end NUMINAMATH_GPT_joohyeon_snack_count_l1415_141577


namespace NUMINAMATH_GPT_product_of_five_consecutive_integers_not_square_l1415_141581

theorem product_of_five_consecutive_integers_not_square (n : ℕ) :
  let P := n * (n + 1) * (n + 2) * (n + 3) * (n + 4)
  ∀ k : ℕ, P ≠ k^2 := 
sorry

end NUMINAMATH_GPT_product_of_five_consecutive_integers_not_square_l1415_141581


namespace NUMINAMATH_GPT_fixed_point_always_l1415_141555

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2^x + Real.logb a (x + 1) + 3

theorem fixed_point_always (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f 0 a = 4 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_always_l1415_141555


namespace NUMINAMATH_GPT_arrange_letters_l1415_141509

-- Definitions based on conditions
def total_letters := 6
def identical_bs := 2 -- Number of B's that are identical
def distinct_as := 3  -- Number of A's that are distinct
def distinct_ns := 1  -- Number of N's that are distinct

-- Now formulate the proof statement
theorem arrange_letters :
    (Nat.factorial total_letters) / (Nat.factorial identical_bs) = 360 :=
by
  sorry

end NUMINAMATH_GPT_arrange_letters_l1415_141509


namespace NUMINAMATH_GPT_general_formula_condition1_general_formula_condition2_general_formula_condition3_sum_Tn_l1415_141542

-- Define the conditions
axiom condition1 (n : ℕ) (h : 2 ≤ n) : ∀ (a : ℕ → ℕ), a 1 = 1 → a n = n / (n-1) * a (n-1)
axiom condition2 (n : ℕ) : ∀ (S : ℕ → ℕ), 2 * S n = n^2 + n
axiom condition3 (n : ℕ) : ∀ (a : ℕ → ℕ), a 1 = 1 → a 3 = 3 → (a n + a (n+2)) = 2 * a (n+1)

-- Proof statements
theorem general_formula_condition1 : ∀ (n : ℕ) (a : ℕ → ℕ) (h : 2 ≤ n), (a 1 = 1) → (∀ n, a n = n) :=
by sorry

theorem general_formula_condition2 : ∀ (n : ℕ) (S a : ℕ → ℕ), (2 * S n = n^2 + n) → (∀ n, a n = n) :=
by sorry

theorem general_formula_condition3 : ∀ (n : ℕ) (a : ℕ → ℕ), (a 1 = 1) → (a 3 = 3) → (∀ n, a n + a (n+2) = 2 * a (n+1)) → (∀ n, a n = n) :=
by sorry

theorem sum_Tn : ∀ (b : ℕ → ℕ) (T : ℕ → ℝ), (b 1 = 2) → (b 2 + b 3 = 12) → (∀ n, T n = 2 * (1 - 1 / (n + 1))) :=
by sorry

end NUMINAMATH_GPT_general_formula_condition1_general_formula_condition2_general_formula_condition3_sum_Tn_l1415_141542


namespace NUMINAMATH_GPT_infinitely_many_solutions_eq_l1415_141512

theorem infinitely_many_solutions_eq {a b : ℝ} 
  (H : ∀ x : ℝ, a * (a - x) - b * (b - x) = 0) : a = b :=
sorry

end NUMINAMATH_GPT_infinitely_many_solutions_eq_l1415_141512


namespace NUMINAMATH_GPT_find_r_l1415_141563

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 :=
sorry

end NUMINAMATH_GPT_find_r_l1415_141563


namespace NUMINAMATH_GPT_intermediate_value_theorem_example_l1415_141583

theorem intermediate_value_theorem_example (f : ℝ → ℝ) :
  f 2007 < 0 → f 2008 < 0 → f 2009 > 0 → ∃ x, 2007 < x ∧ x < 2008 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_intermediate_value_theorem_example_l1415_141583


namespace NUMINAMATH_GPT_PQRS_value_l1415_141568

theorem PQRS_value :
  let P := (Real.sqrt 2011 + Real.sqrt 2010)
  let Q := (-Real.sqrt 2011 - Real.sqrt 2010)
  let R := (Real.sqrt 2011 - Real.sqrt 2010)
  let S := (Real.sqrt 2010 - Real.sqrt 2011)
  P * Q * R * S = -1 :=
by
  sorry

end NUMINAMATH_GPT_PQRS_value_l1415_141568


namespace NUMINAMATH_GPT_merchant_cost_price_l1415_141545

theorem merchant_cost_price (x : ℝ) (h₁ : x + (x^2 / 100) = 39) : x = 30 :=
sorry

end NUMINAMATH_GPT_merchant_cost_price_l1415_141545


namespace NUMINAMATH_GPT_math_problem_l1415_141538

theorem math_problem :
  let initial := 180
  let thirty_five_percent := 0.35 * initial
  let one_third_less := thirty_five_percent - (thirty_five_percent / 3)
  let remaining := initial - one_third_less
  let three_fifths_remaining := (3 / 5) * remaining
  (three_fifths_remaining ^ 2) = 6857.84 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1415_141538


namespace NUMINAMATH_GPT_miquels_theorem_l1415_141528

-- Define a triangle ABC with points D, E, F on sides BC, CA, and AB respectively
variables {A B C D E F : Type}

-- Assume we have a function that checks for collinearity of points
def is_on_side (X Y Z: Type) : Bool := sorry

-- Assume a function that returns the circumcircle of a triangle formed by given points
def circumcircle (X Y Z: Type) : Type := sorry 

-- Define the function that checks the intersection of circumcircles
def have_common_point (circ1 circ2 circ3: Type) : Bool := sorry

-- The theorem statement
theorem miquels_theorem (A B C D E F : Type) 
  (hD: is_on_side D B C) 
  (hE: is_on_side E C A) 
  (hF: is_on_side F A B) : 
  have_common_point (circumcircle A E F) (circumcircle B D F) (circumcircle C D E) :=
sorry

end NUMINAMATH_GPT_miquels_theorem_l1415_141528


namespace NUMINAMATH_GPT_simplify_expr_l1415_141520

noncomputable def expr := 1 / (1 + 1 / (Real.sqrt 5 + 2))
noncomputable def simplified := (Real.sqrt 5 + 1) / 4

theorem simplify_expr : expr = simplified :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1415_141520


namespace NUMINAMATH_GPT_find_b_perpendicular_l1415_141535

theorem find_b_perpendicular
  (b : ℝ)
  (line1 : ∀ x y : ℝ, 2 * x - 3 * y + 5 = 0)
  (line2 : ∀ x y : ℝ, b * x - 3 * y + 1 = 0)
  (perpendicular : (2 / 3) * (b / 3) = -1)
  : b = -9/2 :=
sorry

end NUMINAMATH_GPT_find_b_perpendicular_l1415_141535


namespace NUMINAMATH_GPT_a_5_eq_16_S_8_eq_255_l1415_141523

open Nat

-- Definitions from the conditions
def a : ℕ → ℕ
| 0     => 1
| (n+1) => 2 * a n

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum a

-- Proof problem statements
theorem a_5_eq_16 : a 4 = 16 := sorry

theorem S_8_eq_255 : S 8 = 255 := sorry

end NUMINAMATH_GPT_a_5_eq_16_S_8_eq_255_l1415_141523


namespace NUMINAMATH_GPT_total_cows_l1415_141544

theorem total_cows (Matthews Aaron Tyron Marovich : ℕ) 
  (h1 : Matthews = 60)
  (h2 : Aaron = 4 * Matthews)
  (h3 : Tyron = Matthews - 20)
  (h4 : Aaron + Matthews + Tyron = Marovich + 30) :
  Aaron + Matthews + Tyron + Marovich = 650 :=
by
  sorry

end NUMINAMATH_GPT_total_cows_l1415_141544


namespace NUMINAMATH_GPT_sum_single_digit_numbers_l1415_141569

noncomputable def are_single_digit_distinct (a b c d : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem sum_single_digit_numbers :
  ∀ (A B C D : ℕ),
  are_single_digit_distinct A B C D →
  1000 * A + B - (5000 + 10 * C + 9) = 1000 + 100 * D + 93 →
  A + B + C + D = 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_single_digit_numbers_l1415_141569


namespace NUMINAMATH_GPT_chord_intersects_inner_circle_probability_l1415_141559

noncomputable def probability_of_chord_intersecting_inner_circle
  (radius_inner : ℝ) (radius_outer : ℝ)
  (chord_probability : ℝ) : Prop :=
  radius_inner = 3 ∧ radius_outer = 5 ∧ chord_probability = 0.205

theorem chord_intersects_inner_circle_probability :
  probability_of_chord_intersecting_inner_circle 3 5 0.205 :=
by {
  sorry
}

end NUMINAMATH_GPT_chord_intersects_inner_circle_probability_l1415_141559


namespace NUMINAMATH_GPT_student_scores_correct_answers_l1415_141588

variable (c w : ℕ)

theorem student_scores_correct_answers :
  (c + w = 60) ∧ (4 * c - w = 130) → c = 38 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_student_scores_correct_answers_l1415_141588


namespace NUMINAMATH_GPT_first_term_arithmetic_sequence_l1415_141584

def T_n (a d : ℚ) (n : ℕ) := n * (2 * a + (n - 1) * d) / 2

theorem first_term_arithmetic_sequence (a : ℚ)
  (h_const_ratio : ∀ (n : ℕ), n > 0 → 
    (T_n a 5 (4 * n)) / (T_n a 5 n) = (T_n a 5 4 / T_n a 5 1)) : 
  a = -5/2 :=
by 
  sorry

end NUMINAMATH_GPT_first_term_arithmetic_sequence_l1415_141584


namespace NUMINAMATH_GPT_N_8_12_eq_288_l1415_141560

-- Definitions for various polygonal numbers
def N3 (n : ℕ) : ℕ := n * (n + 1) / 2
def N4 (n : ℕ) : ℕ := n^2
def N5 (n : ℕ) : ℕ := 3 * n^2 / 2 - n / 2
def N6 (n : ℕ) : ℕ := 2 * n^2 - n

-- General definition conjectured
def N (n k : ℕ) : ℕ := (k - 2) * n^2 / 2 + (4 - k) * n / 2

-- The problem statement to prove N(8, 12) == 288
theorem N_8_12_eq_288 : N 8 12 = 288 := by
  -- We would need the proofs for the definitional equalities and calculation here
  sorry

end NUMINAMATH_GPT_N_8_12_eq_288_l1415_141560


namespace NUMINAMATH_GPT_compute_expression_l1415_141504

theorem compute_expression (a b c : ℝ) (h : a^3 - 6 * a^2 + 11 * a - 6 = 0 ∧ b^3 - 6 * b^2 + 11 * b - 6 = 0 ∧ c^3 - 6 * c^2 + 11 * c - 6 = 0) :
  (ab / c + bc / a + ca / b) = 49 / 6 := 
  by
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_compute_expression_l1415_141504


namespace NUMINAMATH_GPT_smaller_group_men_l1415_141507

theorem smaller_group_men (M : ℕ) (h1 : 36 * 25 = M * 90) : M = 10 :=
by
  -- Here we would provide the proof. Unfortunately, proving this in Lean 4 requires knowledge of algebra.
  sorry

end NUMINAMATH_GPT_smaller_group_men_l1415_141507


namespace NUMINAMATH_GPT_simplify_expression_l1415_141573

variable (a b : ℚ)

theorem simplify_expression (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  (2 * a + 1) / (1 - b / (2 * b - 1)) = (2 * a + 1) * (2 * b - 1) / (b - 1) :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1415_141573


namespace NUMINAMATH_GPT_joe_left_pocket_initial_l1415_141527

-- Definitions from conditions
def total_money : ℕ := 200
def initial_left_pocket (L : ℕ) : ℕ := L
def initial_right_pocket (R : ℕ) : ℕ := R
def transfer_one_fourth (L : ℕ) : ℕ := L - L / 4
def add_to_right (R : ℕ) (L : ℕ) : ℕ := R + L / 4
def transfer_20 (L : ℕ) : ℕ := transfer_one_fourth L - 20
def add_20_to_right (R : ℕ) (L : ℕ) : ℕ := add_to_right R L + 20

-- Statement to prove
theorem joe_left_pocket_initial (L R : ℕ) (h₁ : L + R = total_money) 
  (h₂ : transfer_20 L = add_20_to_right R L) : 
  initial_left_pocket L = 160 :=
by
  sorry

end NUMINAMATH_GPT_joe_left_pocket_initial_l1415_141527


namespace NUMINAMATH_GPT_correct_operation_l1415_141529

theorem correct_operation (a : ℝ) :
  (2 * a^2) * a = 2 * a^3 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l1415_141529


namespace NUMINAMATH_GPT_square_side_length_l1415_141552

-- Problem conditions as Lean definitions
def length_rect : ℕ := 400
def width_rect : ℕ := 300
def perimeter_rect := 2 * length_rect + 2 * width_rect
def perimeter_square := 2 * perimeter_rect
def length_square := perimeter_square / 4

-- Proof statement
theorem square_side_length : length_square = 700 := 
by 
  -- (Any necessary tactics to complete the proof would go here)
  sorry

end NUMINAMATH_GPT_square_side_length_l1415_141552


namespace NUMINAMATH_GPT_karen_tests_graded_l1415_141503

theorem karen_tests_graded (n : ℕ) (T : ℕ) 
  (avg_score_70 : T = 70 * n)
  (combined_score_290 : T + 290 = 85 * (n + 2)) : 
  n = 8 := 
sorry

end NUMINAMATH_GPT_karen_tests_graded_l1415_141503


namespace NUMINAMATH_GPT_percentage_heavier_l1415_141521

variables (J M : ℝ)

theorem percentage_heavier (hM : M ≠ 0) : 
  100 * ((J + 3) - M) / M = 100 * ((J + 3) - M) / M := 
sorry

end NUMINAMATH_GPT_percentage_heavier_l1415_141521


namespace NUMINAMATH_GPT_three_digit_number_108_l1415_141546

theorem three_digit_number_108 (a b c : ℕ) (ha : a ≠ 0) (h₀ : a < 10) (h₁ : b < 10) (h₂ : c < 10) (h₃: 100*a + 10*b + c = 12*(a + b + c)) : 
  100*a + 10*b + c = 108 := 
by 
  sorry

end NUMINAMATH_GPT_three_digit_number_108_l1415_141546


namespace NUMINAMATH_GPT_simplify_fractions_sum_l1415_141561

theorem simplify_fractions_sum :
  (48 / 72) + (30 / 45) = 4 / 3 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fractions_sum_l1415_141561


namespace NUMINAMATH_GPT_sequence_a_1998_value_l1415_141526

theorem sequence_a_1998_value :
  (∃ (a : ℕ → ℕ),
    (∀ n : ℕ, 0 <= a n) ∧
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ k : ℕ, ∃ i j t : ℕ, k = a i + 2 * a j + 4 * a t) ∧
    a 1998 = 1227096648) := sorry

end NUMINAMATH_GPT_sequence_a_1998_value_l1415_141526


namespace NUMINAMATH_GPT_union_sets_l1415_141580

def setA : Set ℝ := { x | abs (x - 1) < 3 }
def setB : Set ℝ := { x | x^2 - 4 * x < 0 }

theorem union_sets :
  setA ∪ setB = { x : ℝ | -2 < x ∧ x < 4 } :=
sorry

end NUMINAMATH_GPT_union_sets_l1415_141580


namespace NUMINAMATH_GPT_PhenotypicallyNormalDaughterProbability_l1415_141553

-- Definitions based on conditions
def HemophiliaSexLinkedRecessive := true
def PhenylketonuriaAutosomalRecessive := true
def CouplePhenotypicallyNormal := true
def SonWithBothHemophiliaPhenylketonuria := true

-- Definition of the problem
theorem PhenotypicallyNormalDaughterProbability
  (HemophiliaSexLinkedRecessive : Prop)
  (PhenylketonuriaAutosomalRecessive : Prop)
  (CouplePhenotypicallyNormal : Prop)
  (SonWithBothHemophiliaPhenylketonuria : Prop) :
  -- The correct answer from the solution
  ∃ p : ℚ, p = 3/4 :=
  sorry

end NUMINAMATH_GPT_PhenotypicallyNormalDaughterProbability_l1415_141553


namespace NUMINAMATH_GPT_smallest_number_l1415_141541

def binary_101010 : ℕ := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0
def base5_111 : ℕ := 1 * 5^2 + 1 * 5^1 + 1 * 5^0
def octal_32 : ℕ := 3 * 8^1 + 2 * 8^0
def base6_54 : ℕ := 5 * 6^1 + 4 * 6^0

theorem smallest_number : octal_32 < binary_101010 ∧ octal_32 < base5_111 ∧ octal_32 < base6_54 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_l1415_141541


namespace NUMINAMATH_GPT_problem_statement_l1415_141516

theorem problem_statement : 15 * 35 + 50 * 15 - 5 * 15 = 1200 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1415_141516


namespace NUMINAMATH_GPT_sally_turnip_count_l1415_141547

theorem sally_turnip_count (total_turnips : ℕ) (mary_turnips : ℕ) (sally_turnips : ℕ) 
  (h1: total_turnips = 242) 
  (h2: mary_turnips = 129) 
  (h3: total_turnips = mary_turnips + sally_turnips) : 
  sally_turnips = 113 := 
by 
  sorry

end NUMINAMATH_GPT_sally_turnip_count_l1415_141547


namespace NUMINAMATH_GPT_magnitude_z1_pure_imaginary_l1415_141506

open Complex

theorem magnitude_z1_pure_imaginary 
  (a : ℝ)
  (z1 : ℂ := a + 2 * I)
  (z2 : ℂ := 3 - 4 * I)
  (h : (z1 / z2).re = 0) :
  Complex.abs z1 = 10 / 3 := 
sorry

end NUMINAMATH_GPT_magnitude_z1_pure_imaginary_l1415_141506


namespace NUMINAMATH_GPT_arithmetic_and_geometric_mean_l1415_141575

theorem arithmetic_and_geometric_mean (x y : ℝ) (h1: (x + y) / 2 = 20) (h2: Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 :=
sorry

end NUMINAMATH_GPT_arithmetic_and_geometric_mean_l1415_141575


namespace NUMINAMATH_GPT_admission_counts_l1415_141510

-- Define the total number of ways to admit students under given conditions.
def ways_of_admission : Nat := 1518

-- Statement of the problem: given conditions, prove the result
theorem admission_counts (n_colleges : Nat) (n_students : Nat) (admitted_two_colleges : Bool) : 
  n_colleges = 23 → 
  n_students = 3 → 
  admitted_two_colleges = true →
  ways_of_admission = 1518 :=
by
  intros
  sorry

end NUMINAMATH_GPT_admission_counts_l1415_141510


namespace NUMINAMATH_GPT_general_term_correct_S_maximum_value_l1415_141549

noncomputable def general_term (n : ℕ) : ℤ :=
  if n = 1 then -1 + 24 else (-n^2 + 24 * n) - (-(n - 1)^2 + 24 * (n - 1))

noncomputable def S (n : ℕ) : ℤ :=
  -n^2 + 24 * n

theorem general_term_correct (n : ℕ) (h : 1 ≤ n) : general_term n = -2 * n + 25 := by
  sorry

theorem S_maximum_value : ∃ n : ℕ, S n = 144 ∧ ∀ m : ℕ, S m ≤ 144 := by
  existsi 12
  sorry

end NUMINAMATH_GPT_general_term_correct_S_maximum_value_l1415_141549


namespace NUMINAMATH_GPT_percentage_of_students_owning_cats_l1415_141594

def total_students : ℕ := 500
def students_with_cats : ℕ := 75

theorem percentage_of_students_owning_cats (total_students students_with_cats : ℕ) (h_total: total_students = 500) (h_cats: students_with_cats = 75) :
  100 * (students_with_cats / total_students : ℝ) = 15 := by
  sorry

end NUMINAMATH_GPT_percentage_of_students_owning_cats_l1415_141594


namespace NUMINAMATH_GPT_number_of_roses_two_days_ago_l1415_141519

-- Define the conditions
variables (R : ℕ) 
-- Condition 1: Variable R is the number of roses planted two days ago.
-- Condition 2: The number of roses planted yesterday is R + 20.
-- Condition 3: The number of roses planted today is 2R.
-- Condition 4: The total number of roses planted over three days is 220.
axiom condition_1 : 0 ≤ R
axiom condition_2 : (R + (R + 20) + (2 * R)) = 220

-- Proof goal: Prove that R = 50 
theorem number_of_roses_two_days_ago : R = 50 :=
by sorry

end NUMINAMATH_GPT_number_of_roses_two_days_ago_l1415_141519


namespace NUMINAMATH_GPT_num_positive_solutions_eq_32_l1415_141511

theorem num_positive_solutions_eq_32 : 
  ∃ n : ℕ, (∀ x y : ℕ, 4 * x + 7 * y = 888 → x > 0 ∧ y > 0) ∧ n = 32 :=
sorry

end NUMINAMATH_GPT_num_positive_solutions_eq_32_l1415_141511


namespace NUMINAMATH_GPT_find_salary_l1415_141565

theorem find_salary (x y : ℝ) (h1 : x + y = 2000) (h2 : 0.05 * x = 0.15 * y) : x = 1500 :=
sorry

end NUMINAMATH_GPT_find_salary_l1415_141565


namespace NUMINAMATH_GPT_integer_solutions_range_l1415_141574

theorem integer_solutions_range (a : ℝ) :
  (∀ x : ℤ, x^2 - x + a - a^2 < 0 → x + 2 * a > 1) ↔ 1 < a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_integer_solutions_range_l1415_141574


namespace NUMINAMATH_GPT_find_a2023_l1415_141585

theorem find_a2023
  (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 2/5)
  (h3 : a 3 = 1/4)
  (h_rule : ∀ n : ℕ, 0 < n → (1 / a n + 1 / a (n + 2) = 2 / a (n + 1))) :
  a 2023 = 1 / 3034 :=
by sorry

end NUMINAMATH_GPT_find_a2023_l1415_141585


namespace NUMINAMATH_GPT_height_of_building_l1415_141539

-- Define the conditions as hypotheses
def height_of_flagstaff : ℝ := 17.5
def shadow_length_of_flagstaff : ℝ := 40.25
def shadow_length_of_building : ℝ := 28.75

-- Define the height ratio based on similar triangles
theorem height_of_building :
  (height_of_flagstaff / shadow_length_of_flagstaff = 12.47 / shadow_length_of_building) :=
by
  sorry

end NUMINAMATH_GPT_height_of_building_l1415_141539
