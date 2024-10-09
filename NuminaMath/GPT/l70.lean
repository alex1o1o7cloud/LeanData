import Mathlib

namespace solve_equation1_solve_equation2_l70_7012

theorem solve_equation1 (x : ℝ) (h1 : 5 * x - 2 * (x - 1) = 3) : x = 1 / 3 := 
sorry

theorem solve_equation2 (x : ℝ) (h2 : (x + 3) / 2 - 1 = (2 * x - 1) / 3) : x = 5 :=
sorry

end solve_equation1_solve_equation2_l70_7012


namespace volume_of_inscribed_sphere_l70_7059

noncomputable def volume_of_tetrahedron (R : ℝ) (S1 S2 S3 S4 : ℝ) : ℝ :=
  R * (S1 + S2 + S3 + S4)

theorem volume_of_inscribed_sphere (R S1 S2 S3 S4 V : ℝ) :
  V = R * (S1 + S2 + S3 + S4) :=
sorry

end volume_of_inscribed_sphere_l70_7059


namespace average_students_per_bus_l70_7053

-- Definitions
def total_students : ℕ := 396
def students_in_cars : ℕ := 18
def number_of_buses : ℕ := 7

-- Proof problem statement
theorem average_students_per_bus : (total_students - students_in_cars) / number_of_buses = 54 := by
  sorry

end average_students_per_bus_l70_7053


namespace max_consecutive_irreducible_l70_7050

-- Define what it means for a five-digit number to be irreducible
def is_irreducible (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ ¬∃ x y : ℕ, 100 ≤ x ∧ x < 1000 ∧ 100 ≤ y ∧ y < 1000 ∧ x * y = n

-- Prove the maximum number of consecutive irreducible five-digit numbers is 99
theorem max_consecutive_irreducible : ∃ m : ℕ, m = 99 ∧ 
  (∀ n : ℕ, (n ≤ 99901) → (∀ k : ℕ, (n ≤ k ∧ k < n + m) → is_irreducible k)) ∧
  (∀ x y : ℕ, x > 99 → ∀ n : ℕ, (n ≤ 99899) → (∀ k : ℕ, (n ≤ k ∧ k < n + x) → is_irreducible k) → x = 99) :=
by
  sorry

end max_consecutive_irreducible_l70_7050


namespace expression_value_at_2_l70_7048

theorem expression_value_at_2 : (2^2 - 3 * 2 + 2) = 0 :=
by
  sorry

end expression_value_at_2_l70_7048


namespace find_a_value_l70_7008

theorem find_a_value 
  (f : ℝ → ℝ)
  (a : ℝ)
  (h : ∀ x : ℝ, f x = x^3 + a*x^2 + 3*x - 9)
  (extreme_at_minus_3 : ∀ f' : ℝ → ℝ, (∀ x, f' x = 3*x^2 + 2*a*x + 3) → f' (-3) = 0) :
  a = 5 := 
sorry

end find_a_value_l70_7008


namespace lori_beanie_babies_times_l70_7068

theorem lori_beanie_babies_times (l s : ℕ) (h1 : l = 300) (h2 : l + s = 320) : l = 15 * s :=
by
  sorry

end lori_beanie_babies_times_l70_7068


namespace skeleton_ratio_l70_7080

theorem skeleton_ratio (W M C : ℕ) 
  (h1 : W + M + C = 20)
  (h2 : M = C)
  (h3 : 20 * W + 25 * M + 10 * C = 375) :
  (W : ℚ) / (W + M + C) = 1 / 2 :=
by
  sorry

end skeleton_ratio_l70_7080


namespace sally_cards_final_count_l70_7086

def initial_cards : ℕ := 27
def cards_from_Dan : ℕ := 41
def cards_bought : ℕ := 20
def cards_traded : ℕ := 15
def cards_lost : ℕ := 7

def final_cards (initial : ℕ) (from_Dan : ℕ) (bought : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial + from_Dan + bought - traded - lost

theorem sally_cards_final_count :
  final_cards initial_cards cards_from_Dan cards_bought cards_traded cards_lost = 66 := by
  sorry

end sally_cards_final_count_l70_7086


namespace gardening_project_cost_l70_7032

noncomputable def totalCost : Nat :=
  let roseBushes := 20
  let costPerRoseBush := 150
  let gardenerHourlyRate := 30
  let gardenerHoursPerDay := 5
  let gardenerDays := 4
  let soilCubicFeet := 100
  let soilCostPerCubicFoot := 5

  let costOfRoseBushes := costPerRoseBush * roseBushes
  let gardenerTotalHours := gardenerDays * gardenerHoursPerDay
  let costOfGardener := gardenerHourlyRate * gardenerTotalHours
  let costOfSoil := soilCostPerCubicFoot * soilCubicFeet

  costOfRoseBushes + costOfGardener + costOfSoil

theorem gardening_project_cost : totalCost = 4100 := by
  sorry

end gardening_project_cost_l70_7032


namespace total_kids_played_tag_with_l70_7070

theorem total_kids_played_tag_with : 
  let kids_mon : Nat := 12
  let kids_tues : Nat := 7
  let kids_wed : Nat := 15
  let kids_thurs : Nat := 10
  let kids_fri : Nat := 18
  (kids_mon + kids_tues + kids_wed + kids_thurs + kids_fri) = 62 := by
  sorry

end total_kids_played_tag_with_l70_7070


namespace geometric_series_sum_infinity_l70_7010

theorem geometric_series_sum_infinity (a₁ : ℝ) (q : ℝ) (S₆ S₃ : ℝ)
  (h₁ : a₁ = 3)
  (h₂ : S₆ / S₃ = 7 / 8)
  (h₃ : S₆ = a₁ * (1 - q ^ 6) / (1 - q))
  (h₄ : S₃ = a₁ * (1 - q ^ 3) / (1 - q)) :
  ∑' i : ℕ, a₁ * q ^ i = 2 := by
  sorry

end geometric_series_sum_infinity_l70_7010


namespace lottery_probability_exactly_one_common_l70_7023

open Nat

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem lottery_probability_exactly_one_common :
  let total_combinations := binomial 45 6
  let successful_combinations := 6 * binomial 39 5
  let probability := (successful_combinations : ℚ) / total_combinations
  probability = (6 * binomial 39 5 : ℚ) / binomial 45 6 :=
by
  sorry

end lottery_probability_exactly_one_common_l70_7023


namespace isosceles_triangle_vertex_angle_l70_7030

theorem isosceles_triangle_vertex_angle (A B C : ℝ) (hABC : A + B + C = 180) (h_iso : A = B ∨ B = C ∨ A = C) (h_angle : A = 50 ∨ B = 50 ∨ C = 50) : (A = 50 ∨ A = 80) ∨ (B = 50 ∨ B = 80) ∨ (C = 50 ∨ C = 80) :=
by sorry

end isosceles_triangle_vertex_angle_l70_7030


namespace students_in_class_l70_7041

/-- Conditions:
1. 20 hands in Peter’s class, not including his.
2. Every student in the class has 2 hands.

Prove that the number of students in Peter’s class including him is 11.
-/
theorem students_in_class (hands_without_peter : ℕ) (hands_per_student : ℕ) (students_including_peter : ℕ) :
  hands_without_peter = 20 →
  hands_per_student = 2 →
  students_including_peter = (hands_without_peter + hands_per_student) / hands_per_student →
  students_including_peter = 11 :=
by
  intros h₁ h₂ h₃
  sorry

end students_in_class_l70_7041


namespace simplify_expression_l70_7060

variable (x y z : ℝ)

noncomputable def expr1 := (3 * x + y / 3 + 2 * z)⁻¹ * ((3 * x)⁻¹ + (y / 3)⁻¹ + (2 * z)⁻¹)
noncomputable def expr2 := (2 * y + 18 * x * z + 3 * z * x) / (6 * x * y * z * (9 * x + y + 6 * z))

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxyz : 3 * x + y / 3 + 2 * z ≠ 0) :
  expr1 x y z = expr2 x y z := by 
  sorry

end simplify_expression_l70_7060


namespace find_phi_l70_7015

theorem find_phi (ϕ : ℝ) (h0 : 0 ≤ ϕ) (h1 : ϕ < π)
    (H : 2 * Real.cos (π / 3) = 2 * Real.sin (2 * (π / 3) + ϕ)) : ϕ = π / 6 :=
by
  sorry

end find_phi_l70_7015


namespace washing_machine_heavy_wash_usage_l70_7090

-- Definition of variables and constants
variables (H : ℕ)                           -- Amount of water used for a heavy wash
def regular_wash : ℕ := 10                   -- Gallons used for a regular wash
def light_wash : ℕ := 2                      -- Gallons used for a light wash
def extra_light_wash : ℕ := light_wash       -- Extra light wash due to bleach

-- Number of each type of wash
def num_heavy_washes : ℕ := 2
def num_regular_washes : ℕ := 3
def num_light_washes : ℕ := 1
def num_bleached_washes : ℕ := 2

-- Total water usage
def total_water_usage : ℕ := 
  num_heavy_washes * H + 
  num_regular_washes * regular_wash + 
  num_light_washes * light_wash + 
  num_bleached_washes * extra_light_wash

-- Given total water usage
def given_total_water_usage : ℕ := 76

-- Lean statement to prove the amount of water used for a heavy wash
theorem washing_machine_heavy_wash_usage : total_water_usage H = given_total_water_usage → H = 20 :=
by
  sorry

end washing_machine_heavy_wash_usage_l70_7090


namespace time_upstream_equal_nine_hours_l70_7046

noncomputable def distance : ℝ := 126
noncomputable def time_downstream : ℝ := 7
noncomputable def current_speed : ℝ := 2
noncomputable def downstream_speed := distance / time_downstream
noncomputable def boat_speed := downstream_speed - current_speed
noncomputable def upstream_speed := boat_speed - current_speed

theorem time_upstream_equal_nine_hours : (distance / upstream_speed) = 9 := by
  sorry

end time_upstream_equal_nine_hours_l70_7046


namespace woman_work_completion_woman_days_to_complete_l70_7026

theorem woman_work_completion (M W B : ℝ) (h1 : M + W + B = 1/4) (h2 : M = 1/6) (h3 : B = 1/18) : W = 1/36 :=
by
  -- Substitute h2 and h3 into h1 and solve for W
  sorry

theorem woman_days_to_complete (W : ℝ) (h : W = 1/36) : 1 / W = 36 :=
by
  -- Calculate the reciprocal of h
  sorry

end woman_work_completion_woman_days_to_complete_l70_7026


namespace michelle_total_payment_l70_7045
noncomputable def michelle_base_cost := 25
noncomputable def included_talk_time := 40 -- in hours
noncomputable def text_cost := 10 -- in cents per message
noncomputable def extra_talk_cost := 15 -- in cents per minute
noncomputable def february_texts_sent := 200
noncomputable def february_talk_time := 41 -- in hours

theorem michelle_total_payment : 
  25 + ((200 * 10) / 100) + (((41 - 40) * 60 * 15) / 100) = 54 := by
  sorry

end michelle_total_payment_l70_7045


namespace gcd_proof_l70_7005

theorem gcd_proof :
  ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a + b = 33 ∧ Nat.lcm a b = 90 ∧ Nat.gcd a b = 3 :=
sorry

end gcd_proof_l70_7005


namespace husband_age_l70_7091

theorem husband_age (a b : ℕ) (w_age h_age : ℕ) (ha : a > 0) (hb : b > 0) 
  (hw_age : w_age = 10 * a + b) 
  (hh_age : h_age = 10 * b + a) 
  (h_older : h_age > w_age)
  (h_difference : 9 * (b - a) = a + b) :
  h_age = 54 :=
by
  sorry

end husband_age_l70_7091


namespace investment_C_120000_l70_7088

noncomputable def investment_C (P_B P_A_difference : ℕ) (investment_A investment_B : ℕ) : ℕ :=
  let P_A := (P_B * investment_A) / investment_B
  let P_C := P_A + P_A_difference
  (P_C * investment_B) / P_B

theorem investment_C_120000
  (investment_A investment_B P_B P_A_difference : ℕ)
  (hA : investment_A = 8000)
  (hB : investment_B = 10000)
  (hPB : P_B = 1400)
  (hPA_difference : P_A_difference = 560) :
  investment_C P_B P_A_difference investment_A investment_B = 120000 :=
by
  sorry

end investment_C_120000_l70_7088


namespace max_xy_l70_7040

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 18) : xy <= 81 :=
by {
  sorry
}

end max_xy_l70_7040


namespace math_problem_l70_7016

theorem math_problem : (100 - (5050 - 450)) + (5050 - (450 - 100)) = 200 := by
  sorry

end math_problem_l70_7016


namespace count_integers_between_bounds_l70_7071

theorem count_integers_between_bounds : 
  ∃ n : ℤ, n = 15 ∧ ∀ x : ℤ, 3 < Real.sqrt (x : ℝ) ∧ Real.sqrt (x : ℝ) < 5 → 10 ≤ x ∧ x ≤ 24 :=
by
  sorry

end count_integers_between_bounds_l70_7071


namespace arithmetic_geometric_mean_l70_7063

theorem arithmetic_geometric_mean (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_l70_7063


namespace time_for_machine_A_l70_7062

theorem time_for_machine_A (x : ℝ) (T : ℝ) (A B : ℝ) :
  (B = 2 * x / 5) → 
  (A + B = x / 2) → 
  (A = x / T) → 
  T = 10 := 
by 
  intros hB hAB hA
  sorry

end time_for_machine_A_l70_7062


namespace hyperbola_constant_ellipse_constant_l70_7074

variables {a b : ℝ} (a_pos_b_gt_a : 0 < a ∧ a < b)
variables {A B : ℝ × ℝ} (on_hyperbola_A : A.1^2 / a^2 - A.2^2 / b^2 = 1)
variables (on_hyperbola_B : B.1^2 / a^2 - B.2^2 / b^2 = 1) (perp_OA_OB : A.1 * B.1 + A.2 * B.2 = 0)

-- Hyperbola statement
theorem hyperbola_constant :
  (1 / (A.1^2 + A.2^2)) + (1 / (B.1^2 + B.2^2)) = 1 / a^2 - 1 / b^2 :=
sorry

variables {C D : ℝ × ℝ} (on_ellipse_C : C.1^2 / a^2 + C.2^2 / b^2 = 1)
variables (on_ellipse_D : D.1^2 / a^2 + D.2^2 / b^2 = 1) (perp_OC_OD : C.1 * D.1 + C.2 * D.2 = 0)

-- Ellipse statement
theorem ellipse_constant :
  (1 / (C.1^2 + C.2^2)) + (1 / (D.1^2 + D.2^2)) = 1 / a^2 + 1 / b^2 :=
sorry

end hyperbola_constant_ellipse_constant_l70_7074


namespace cost_per_day_additional_weeks_l70_7002

theorem cost_per_day_additional_weeks :
  let first_week_days := 7
  let first_week_cost_per_day := 18.00
  let first_week_cost := first_week_days * first_week_cost_per_day
  let total_days := 23
  let total_cost := 302.00
  let additional_days := total_days - first_week_days
  let additional_cost := total_cost - first_week_cost
  let cost_per_day_additional := additional_cost / additional_days
  cost_per_day_additional = 11.00 :=
by
  sorry

end cost_per_day_additional_weeks_l70_7002


namespace geometric_sequence_a2_l70_7027

noncomputable def geometric_sequence_sum (n : ℕ) (a : ℝ) : ℝ :=
  a * (3^n) - 2

theorem geometric_sequence_a2 (a : ℝ) : (∃ a1 a2 a3 : ℝ, 
  a1 = geometric_sequence_sum 1 a ∧ 
  a1 + a2 = geometric_sequence_sum 2 a ∧ 
  a1 + a2 + a3 = geometric_sequence_sum 3 a ∧ 
  a2 = 6 * a ∧ 
  a3 = 18 * a ∧ 
  (6 * a)^2 = (a1) * (a3) ∧ 
  a = 2) →
  a2 = 12 :=
by
  intros h
  sorry

end geometric_sequence_a2_l70_7027


namespace fred_dark_blue_marbles_count_l70_7018

/-- Fred's Marble Problem -/
def freds_marbles (red green dark_blue : ℕ) : Prop :=
  red = 38 ∧ green = red / 2 ∧ red + green + dark_blue = 63

theorem fred_dark_blue_marbles_count (red green dark_blue : ℕ) (h : freds_marbles red green dark_blue) :
  dark_blue = 6 :=
by
  sorry

end fred_dark_blue_marbles_count_l70_7018


namespace find_length_AB_l70_7006

theorem find_length_AB 
(distance_between_parallels : ℚ)
(radius_of_incircle : ℚ)
(is_isosceles : Prop)
(h_parallel : distance_between_parallels = 18 / 25)
(h_radius : radius_of_incircle = 8 / 3)
(h_isosceles : is_isosceles) :
  ∃ AB : ℚ, AB = 20 := 
sorry

end find_length_AB_l70_7006


namespace atomic_weight_Ca_l70_7013

def molecular_weight_CaH2 : ℝ := 42
def atomic_weight_H : ℝ := 1.008

theorem atomic_weight_Ca : atomic_weight_H * 2 < molecular_weight_CaH2 :=
by sorry

end atomic_weight_Ca_l70_7013


namespace math_problem_l70_7021

theorem math_problem
    (p q s : ℕ)
    (prime_p : Nat.Prime p)
    (prime_q : Nat.Prime q)
    (prime_s : Nat.Prime s)
    (h1 : p * q = s + 6)
    (h2 : 3 < p)
    (h3 : p < q) :
    p = 5 :=
    sorry

end math_problem_l70_7021


namespace average_of_first_6_numbers_l70_7075

-- Definitions extracted from conditions
def average_of_11_numbers := 60
def average_of_last_6_numbers := 65
def sixth_number := 258
def total_sum := 11 * average_of_11_numbers
def sum_of_last_6_numbers := 6 * average_of_last_6_numbers

-- Lean 4 statement for the proof problem
theorem average_of_first_6_numbers :
  (∃ A, 6 * A = (total_sum - (sum_of_last_6_numbers - sixth_number))) →
  (∃ A, 6 * A = 528) :=
by
  intro h
  exact h

end average_of_first_6_numbers_l70_7075


namespace marbles_solution_l70_7098

def marbles_problem : Prop :=
  let total_marbles := 20
  let blue_marbles := 6
  let red_marbles := 9
  let total_prob_red_white := 0.7
  let white_marbles := 5
  total_marbles = blue_marbles + red_marbles + white_marbles ∧
  (white_marbles / total_marbles + red_marbles / total_marbles = total_prob_red_white)

theorem marbles_solution : marbles_problem :=
by {
  sorry
}

end marbles_solution_l70_7098


namespace probability_six_distinct_numbers_l70_7087

theorem probability_six_distinct_numbers :
  let total_outcomes := 6^6
  let distinct_outcomes := Nat.factorial 6
  let probability := (distinct_outcomes:ℚ) / (total_outcomes:ℚ)
  probability = 5 / 324 :=
sorry

end probability_six_distinct_numbers_l70_7087


namespace problem_solution_includes_024_l70_7014

theorem problem_solution_includes_024 (x : ℝ) :
  (2 * 88 * (abs (abs (abs (abs (x - 1) - 1) - 1) - 1)) = 0) →
  x = 0 ∨ x = 2 ∨ x = 4 :=
by
  sorry

end problem_solution_includes_024_l70_7014


namespace divisibility_condition_l70_7017

theorem divisibility_condition (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ab ∣ (a^2 + b^2 - a - b + 1) → (a = 1 ∧ b = 1) :=
by sorry

end divisibility_condition_l70_7017


namespace augmented_matrix_solution_l70_7022

theorem augmented_matrix_solution (a b : ℝ) 
    (h1 : (∀ (x y : ℝ), (a * x = 2 ∧ y = b ↔ x = 2 ∧ y = 1))) : 
    a + b = 2 :=
by
  sorry

end augmented_matrix_solution_l70_7022


namespace two_pow_15000_mod_1250_l70_7092

theorem two_pow_15000_mod_1250 (h : 2 ^ 500 ≡ 1 [MOD 1250]) :
  2 ^ 15000 ≡ 1 [MOD 1250] :=
sorry

end two_pow_15000_mod_1250_l70_7092


namespace triangle_properties_l70_7096

theorem triangle_properties
  (a b : ℝ)
  (C : ℝ)
  (ha : a = 2)
  (hb : b = 3)
  (hC : C = Real.pi / 3)
  :
  let c := Real.sqrt (a^2 + b^2 - 2 * a * b * Real.cos C)
  let area := (1 / 2) * a * b * Real.sin C
  let sin2A := 2 * (a * Real.sin C / c) * Real.sqrt (1 - (a * Real.sin C / c)^2)
  c = Real.sqrt 7 
  ∧ area = (3 * Real.sqrt 3) / 2 
  ∧ sin2A = (4 * Real.sqrt 3) / 7 :=
by
  sorry

end triangle_properties_l70_7096


namespace arithmetic_series_sum_l70_7042

theorem arithmetic_series_sum :
  let a1 := 5
  let an := 105
  let d := 1
  let n := (an - a1) / d + 1
  (n * (a1 + an) / 2) = 5555 := by
  sorry

end arithmetic_series_sum_l70_7042


namespace problem_statement_l70_7035

noncomputable def smallest_x : ℝ :=
  -8 - (Real.sqrt 292 / 2)

theorem problem_statement (x : ℝ) :
  (15 * x ^ 2 - 40 * x + 18) / (4 * x - 3) + 4 * x = 8 * x - 3 ↔ x = smallest_x :=
by
  sorry

end problem_statement_l70_7035


namespace part1_part2_l70_7064

open Real

noncomputable def curve_parametric (α : ℝ) : ℝ × ℝ :=
  (2 + sqrt 10 * cos α, sqrt 10 * sin α)

noncomputable def curve_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * cos θ - 6 = 0

noncomputable def line_polar (ρ θ : ℝ) : Prop :=
  ρ * cos θ + 2 * ρ * sin θ - 12 = 0

theorem part1 (α : ℝ) : ∃ ρ θ : ℝ, curve_polar ρ θ :=
  sorry

theorem part2 : ∃ ρ1 ρ2 : ℝ, curve_polar ρ1 (π / 4) ∧ line_polar ρ2 (π / 4) ∧ abs (ρ1 - ρ2) = sqrt 2 :=
  sorry

end part1_part2_l70_7064


namespace range_of_a_l70_7055

variables {a x : ℝ}

def P (a : ℝ) : Prop := ∀ x, ¬ (x^2 - (a + 1) * x + 1 ≤ 0)

def Q (a : ℝ) : Prop := ∀ x, |x - 1| ≥ a + 2

theorem range_of_a (a : ℝ) : 
  (¬ P a ∧ ¬ Q a) → a ≥ 1 :=
by
  sorry

end range_of_a_l70_7055


namespace correct_calculation_l70_7067

theorem correct_calculation (n : ℕ) (h : n - 59 = 43) : n - 46 = 56 :=
by
  sorry

end correct_calculation_l70_7067


namespace trig_identity_l70_7024

-- Given conditions
variables (α : ℝ) (h_tan : Real.tan (Real.pi - α) = -2)

-- The goal is to prove the desired equality.
theorem trig_identity :
  1 / (Real.cos (2 * α) + Real.cos α * Real.cos α) = -5 / 2 :=
by
  sorry

end trig_identity_l70_7024


namespace sequence_sum_difference_l70_7019

def sum_odd (n : ℕ) : ℕ := n * n
def sum_even (n : ℕ) : ℕ := n * (n + 1)
def sum_triangular (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem sequence_sum_difference :
  sum_even 1500 - sum_odd 1500 + sum_triangular 1500 = 563628000 :=
by
  sorry

end sequence_sum_difference_l70_7019


namespace vertical_shift_d_l70_7007

variable (a b c d : ℝ)

theorem vertical_shift_d (h1: d + a = 5) (h2: d - a = 1) : d = 3 := 
by
  sorry

end vertical_shift_d_l70_7007


namespace expression_equals_6_l70_7079

-- Define the expression as a Lean definition.
def expression : ℤ := 1 - (-2) - 3 - (-4) - 5 - (-6) - 7 - (-8)

-- The statement to prove that the expression equals 6.
theorem expression_equals_6 : expression = 6 := 
by 
  -- This is a placeholder for the actual proof.
  sorry

end expression_equals_6_l70_7079


namespace ratio_of_work_capacity_l70_7011

theorem ratio_of_work_capacity (work_rate_A work_rate_B : ℝ)
  (hA : work_rate_A = 1 / 45)
  (hAB : work_rate_A + work_rate_B = 1 / 18) :
  work_rate_A⁻¹ / work_rate_B⁻¹ = 3 / 2 :=
by
  sorry

end ratio_of_work_capacity_l70_7011


namespace three_digit_number_divisibility_four_digit_number_divisibility_l70_7073

-- Definition of three-digit number
def is_three_digit_number (a : ℕ) : Prop := 100 ≤ a ∧ a ≤ 999

-- Definition of four-digit number
def is_four_digit_number (b : ℕ) : Prop := 1000 ≤ b ∧ b ≤ 9999

-- First proof problem
theorem three_digit_number_divisibility (a : ℕ) (h : is_three_digit_number a) : 
  (1001 * a) % 7 = 0 ∧ (1001 * a) % 11 = 0 ∧ (1001 * a) % 13 = 0 := 
sorry

-- Second proof problem
theorem four_digit_number_divisibility (b : ℕ) (h : is_four_digit_number b) : 
  (10001 * b) % 73 = 0 ∧ (10001 * b) % 137 = 0 := 
sorry

end three_digit_number_divisibility_four_digit_number_divisibility_l70_7073


namespace combined_molecular_weight_l70_7043

theorem combined_molecular_weight {m1 m2 : ℕ} 
  (MW_C : ℝ) (MW_H : ℝ) (MW_O : ℝ)
  (Butanoic_acid : ℕ × ℕ × ℕ)
  (Propanoic_acid : ℕ × ℕ × ℕ)
  (MW_Butanoic_acid : ℝ)
  (MW_Propanoic_acid : ℝ)
  (weight_Butanoic_acid : ℝ)
  (weight_Propanoic_acid : ℝ)
  (total_weight : ℝ) :
MW_C = 12.01 → MW_H = 1.008 → MW_O = 16.00 →
Butanoic_acid = (4, 8, 2) → MW_Butanoic_acid = (4 * MW_C) + (8 * MW_H) + (2 * MW_O) →
Propanoic_acid = (3, 6, 2) → MW_Propanoic_acid = (3 * MW_C) + (6 * MW_H) + (2 * MW_O) →
m1 = 9 → weight_Butanoic_acid = m1 * MW_Butanoic_acid →
m2 = 5 → weight_Propanoic_acid = m2 * MW_Propanoic_acid →
total_weight = weight_Butanoic_acid + weight_Propanoic_acid →
total_weight = 1163.326 :=
by {
  intros;
  sorry
}

end combined_molecular_weight_l70_7043


namespace multiply_by_11_l70_7085

theorem multiply_by_11 (A B : ℕ) (h : A + B < 10) : 
  (10 * A + B) * 11 = 100 * A + 10 * (A + B) + B :=
by
  sorry

end multiply_by_11_l70_7085


namespace find_other_integer_l70_7020

theorem find_other_integer (x y : ℤ) (h_sum : 3 * x + 2 * y = 115) (h_one_is_25 : x = 25 ∨ y = 25) : (x = 25 → y = 20) ∧ (y = 25 → x = 20) :=
by
  sorry

end find_other_integer_l70_7020


namespace product_mod_17_eq_zero_l70_7039

theorem product_mod_17_eq_zero :
    (2001 * 2002 * 2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end product_mod_17_eq_zero_l70_7039


namespace max_third_side_of_triangle_l70_7051

theorem max_third_side_of_triangle (a b : ℕ) (h₁ : a = 7) (h₂ : b = 11) : 
  ∃ c : ℕ, c < a + b ∧ c = 17 :=
by 
  sorry

end max_third_side_of_triangle_l70_7051


namespace circle_equation_standard_l70_7054

open Real

noncomputable def equation_of_circle : Prop :=
  ∃ R : ℝ, R = sqrt 2 ∧ 
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 → x + y - 2 = 0 → 0 ≤ x ∧ x ≤ 2)

theorem circle_equation_standard :
    equation_of_circle := sorry

end circle_equation_standard_l70_7054


namespace bryce_received_15_raisins_l70_7001

theorem bryce_received_15_raisins (x : ℕ) (c : ℕ) (h1 : c = x - 10) (h2 : c = x / 3) : x = 15 :=
by
  sorry

end bryce_received_15_raisins_l70_7001


namespace impossible_circular_arrangement_1_to_60_l70_7033

theorem impossible_circular_arrangement_1_to_60 :
  (∀ (f : ℕ → ℕ), 
      (∀ n, 1 ≤ f n ∧ f n ≤ 60) ∧ 
      (∀ n, f (n + 2) + f n ≡ 0 [MOD 2]) ∧ 
      (∀ n, f (n + 3) + f n ≡ 0 [MOD 3]) ∧ 
      (∀ n, f (n + 7) + f n ≡ 0 [MOD 7]) 
      → false) := 
  sorry

end impossible_circular_arrangement_1_to_60_l70_7033


namespace range_of_set_l70_7097

theorem range_of_set (a b c : ℕ) (h1 : a = 2) (h2 : b = 6) (h3 : 2 ≤ c ∧ c ≤ 10) (h4 : (a + b + c) / 3 = 6) : (c - a) = 8 :=
by
  sorry

end range_of_set_l70_7097


namespace per_capita_income_growth_l70_7056

noncomputable def income2020 : ℝ := 3.2
noncomputable def income2022 : ℝ := 3.7
variable (x : ℝ)

/--
Prove the per capita disposable income model.
-/
theorem per_capita_income_growth :
  income2020 * (1 + x)^2 = income2022 :=
sorry

end per_capita_income_growth_l70_7056


namespace total_hours_is_900_l70_7076

-- Definitions for the video length, speeds, and number of videos watched
def video_length : ℕ := 100
def lila_speed : ℕ := 2
def roger_speed : ℕ := 1
def num_videos : ℕ := 6

-- Definition of total hours watched
def total_hours_watched : ℕ :=
  let lila_time_per_video := video_length / lila_speed
  let roger_time_per_video := video_length / roger_speed
  (lila_time_per_video * num_videos) + (roger_time_per_video * num_videos)

-- Prove that the total hours watched is 900
theorem total_hours_is_900 : total_hours_watched = 900 :=
by
  -- Proving the equation step-by-step
  sorry

end total_hours_is_900_l70_7076


namespace total_faces_is_198_l70_7089

-- Definitions for the number of dice and geometrical shapes brought by each person:
def TomDice : ℕ := 4
def TimDice : ℕ := 5
def TaraDice : ℕ := 3
def TinaDice : ℕ := 2
def TonyCubes : ℕ := 1
def TonyTetrahedrons : ℕ := 3
def TonyIcosahedrons : ℕ := 2

-- Definitions for the number of faces for each type of dice or shape:
def SixSidedFaces : ℕ := 6
def EightSidedFaces : ℕ := 8
def TwelveSidedFaces : ℕ := 12
def TwentySidedFaces : ℕ := 20
def CubeFaces : ℕ := 6
def TetrahedronFaces : ℕ := 4
def IcosahedronFaces : ℕ := 20

-- We want to prove that the total number of faces is 198:
theorem total_faces_is_198 : 
  (TomDice * SixSidedFaces) + 
  (TimDice * EightSidedFaces) + 
  (TaraDice * TwelveSidedFaces) + 
  (TinaDice * TwentySidedFaces) + 
  (TonyCubes * CubeFaces) + 
  (TonyTetrahedrons * TetrahedronFaces) + 
  (TonyIcosahedrons * IcosahedronFaces) 
  = 198 := 
by {
  sorry
}

end total_faces_is_198_l70_7089


namespace find_A_l70_7066

theorem find_A (A B : ℕ) (h1 : A + B = 1149) (h2 : A = 8 * B + 24) : A = 1024 :=
by
  sorry

end find_A_l70_7066


namespace novel_pages_total_l70_7093

-- Definitions based on conditions
def pages_first_two_days : ℕ := 2 * 50
def pages_next_four_days : ℕ := 4 * 25
def pages_six_days : ℕ := pages_first_two_days + pages_next_four_days
def pages_seventh_day : ℕ := 30
def total_pages : ℕ := pages_six_days + pages_seventh_day

-- Statement of the problem as a theorem in Lean 4
theorem novel_pages_total : total_pages = 230 := by
  sorry

end novel_pages_total_l70_7093


namespace distinct_nonzero_real_product_l70_7000

noncomputable section
open Real

theorem distinct_nonzero_real_product
  (a b c d : ℝ)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hcd : c ≠ d)
  (hda : d ≠ a)
  (ha_ne_0 : a ≠ 0)
  (hb_ne_0 : b ≠ 0)
  (hc_ne_0 : c ≠ 0)
  (hd_ne_0 : d ≠ 0)
  (h : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/d ∧ c + 1/d = d + 1/a) :
  |a * b * c * d| = 1 :=
sorry

end distinct_nonzero_real_product_l70_7000


namespace cost_of_seven_CDs_l70_7047

theorem cost_of_seven_CDs (cost_per_two : ℝ) (h1 : cost_per_two = 32) : (7 * (cost_per_two / 2)) = 112 :=
by
  sorry

end cost_of_seven_CDs_l70_7047


namespace matrix_power_application_l70_7099

variable (B : Matrix (Fin 2) (Fin 2) ℝ)
variable (v : Fin 2 → ℝ := ![4, -3])

theorem matrix_power_application :
  (B.mulVec v = ![8, -6]) →
  (B ^ 4).mulVec v = ![64, -48] :=
by
  intro h
  sorry

end matrix_power_application_l70_7099


namespace train_cross_time_approx_24_seconds_l70_7094

open Real

noncomputable def time_to_cross (train_length : ℝ) (train_speed_km_h : ℝ) (man_speed_km_h : ℝ) : ℝ :=
  let train_speed_m_s := train_speed_km_h * (1000 / 3600)
  let man_speed_m_s := man_speed_km_h * (1000 / 3600)
  let relative_speed := train_speed_m_s - man_speed_m_s
  train_length / relative_speed

theorem train_cross_time_approx_24_seconds : 
  abs (time_to_cross 400 63 3 - 24) < 1 :=
by
  sorry

end train_cross_time_approx_24_seconds_l70_7094


namespace female_managers_count_l70_7095

variable (E M F FM : ℕ)

-- Conditions
def female_employees : Prop := F = 750
def fraction_managers : Prop := (2 / 5 : ℚ) * E = FM + (2 / 5 : ℚ) * M
def total_employees : Prop := E = M + F

-- Proof goal
theorem female_managers_count (h1 : female_employees F) 
                              (h2 : fraction_managers E M FM) 
                              (h3 : total_employees E M F) : 
  FM = 300 := 
sorry

end female_managers_count_l70_7095


namespace total_students_correct_l70_7061

theorem total_students_correct (H : ℕ)
  (B : ℕ := 2 * H)
  (P : ℕ := H + 5)
  (S : ℕ := 3 * (H + 5))
  (h1 : B = 30)
  : (B + H + P + S) = 125 := by
  sorry

end total_students_correct_l70_7061


namespace merchant_gross_profit_l70_7083

-- Define the purchase price and markup rate
def purchase_price : ℝ := 42
def markup_rate : ℝ := 0.30
def discount_rate : ℝ := 0.20

-- Define the selling price equation given the purchase price and markup rate
def selling_price (S : ℝ) : Prop := S = purchase_price + markup_rate * S

-- Define the discounted selling price given the selling price and discount rate
def discounted_selling_price (S : ℝ) : ℝ := S - discount_rate * S

-- Define the gross profit as the difference between the discounted selling price and purchase price
def gross_profit (S : ℝ) : ℝ := discounted_selling_price S - purchase_price

theorem merchant_gross_profit : ∃ S : ℝ, selling_price S ∧ gross_profit S = 6 :=
by
  sorry

end merchant_gross_profit_l70_7083


namespace fred_has_9_dimes_l70_7009

-- Fred has 90 cents in his bank.
def freds_cents : ℕ := 90

-- A dime is worth 10 cents.
def value_of_dime : ℕ := 10

-- Prove that the number of dimes Fred has is 9.
theorem fred_has_9_dimes : (freds_cents / value_of_dime) = 9 := by
  sorry

end fred_has_9_dimes_l70_7009


namespace ant_minimum_distance_l70_7037

section
variables (x y z w u : ℝ)

-- Given conditions
axiom h1 : x + y + z = 22
axiom h2 : w + y + z = 29
axiom h3 : x + y + u = 30

-- Prove the ant crawls at least 47 cm to cover all paths
theorem ant_minimum_distance : x + y + z + w ≥ 47 :=
sorry
end

end ant_minimum_distance_l70_7037


namespace mushrooms_used_by_Karla_correct_l70_7058

-- Given conditions
def mushrooms_cut_each_mushroom : ℕ := 4
def mushrooms_cut_total : ℕ := 22 * mushrooms_cut_each_mushroom
def mushrooms_used_by_Kenny : ℕ := 38
def mushrooms_remaining : ℕ := 8
def mushrooms_total_used_by_Kenny_and_remaining : ℕ := mushrooms_used_by_Kenny + mushrooms_remaining
def mushrooms_used_by_Karla : ℕ := mushrooms_cut_total - mushrooms_total_used_by_Kenny_and_remaining

-- Statement to prove
theorem mushrooms_used_by_Karla_correct :
  mushrooms_used_by_Karla = 42 :=
by
  sorry

end mushrooms_used_by_Karla_correct_l70_7058


namespace min_period_and_max_value_l70_7028

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 - (Real.sin x)^2 + 2

theorem min_period_and_max_value :
  (∀ x, f (x + π) = f x) ∧ (∀ x, f x ≤ 4) ∧ (∃ x, f x = 4) :=
by
  sorry

end min_period_and_max_value_l70_7028


namespace car_average_speed_is_correct_l70_7057

noncomputable def average_speed_of_car : ℝ :=
  let d1 := 30
  let s1 := 30
  let d2 := 35
  let s2 := 55
  let t3 := 0.5
  let s3 := 70
  let t4 := 40 / 60 -- 40 minutes converted to hours
  let s4 := 36
  let t1 := d1 / s1
  let t2 := d2 / s2
  let d3 := s3 * t3
  let d4 := s4 * t4
  let total_distance := d1 + d2 + d3 + d4
  let total_time := t1 + t2 + t3 + t4
  total_distance / total_time

theorem car_average_speed_is_correct :
  average_speed_of_car = 44.238 := 
sorry

end car_average_speed_is_correct_l70_7057


namespace tangent_sphere_surface_area_l70_7078

noncomputable def cube_side_length (V : ℝ) : ℝ := V^(1/3)
noncomputable def sphere_radius (a : ℝ) : ℝ := a / 2
noncomputable def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem tangent_sphere_surface_area (V : ℝ) (hV : V = 64) : 
  sphere_surface_area (sphere_radius (cube_side_length V)) = 16 * Real.pi :=
by
  sorry

end tangent_sphere_surface_area_l70_7078


namespace perimeter_square_III_l70_7077

theorem perimeter_square_III (perimeter_I perimeter_II : ℕ) (hI : perimeter_I = 12) (hII : perimeter_II = 24) : 
  let side_I := perimeter_I / 4 
  let side_II := perimeter_II / 4 
  let side_III := side_I + side_II 
  4 * side_III = 36 :=
by
  sorry

end perimeter_square_III_l70_7077


namespace aisha_additional_miles_l70_7069

theorem aisha_additional_miles
  (D : ℕ) (d : ℕ) (v1 : ℕ) (v2 : ℕ) (v_avg : ℕ)
  (h1 : D = 18) (h2 : v1 = 36) (h3 : v2 = 60) (h4 : v_avg = 48)
  (h5 : d = 30) :
  (D + d) / ((D / v1) + (d / v2)) = v_avg :=
  sorry

end aisha_additional_miles_l70_7069


namespace math_problem_l70_7004

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

def a_n (n : ℕ) : ℕ := 3 * n - 5

theorem math_problem (C5_4 : ℕ) (C6_4 : ℕ) (C7_4 : ℕ) :
  C5_4 = binomial 5 4 →
  C6_4 = binomial 6 4 →
  C7_4 = binomial 7 4 →
  C5_4 + C6_4 + C7_4 = 55 →
  ∃ n : ℕ, a_n n = 55 ∧ n = 20 :=
by
  sorry

end math_problem_l70_7004


namespace jake_pure_alcohol_l70_7031

theorem jake_pure_alcohol (total_shots : ℕ) (shots_per_split : ℕ) (ounces_per_shot : ℚ) (purity : ℚ) :
  total_shots = 8 →
  shots_per_split = 2 →
  ounces_per_shot = 1.5 →
  purity = 0.5 →
  (total_shots / shots_per_split) * ounces_per_shot * purity = 3 := 
by
  sorry

end jake_pure_alcohol_l70_7031


namespace cos_values_l70_7084

theorem cos_values (n : ℤ) : (0 ≤ n ∧ n ≤ 360) ∧ (Real.cos (n * Real.pi / 180) = Real.cos (310 * Real.pi / 180)) ↔ (n = 50 ∨ n = 310) :=
by
  sorry

end cos_values_l70_7084


namespace value_of_40th_expression_l70_7034

-- Define the sequence
def minuend (n : ℕ) : ℕ := 100 - (n - 1)
def subtrahend (n : ℕ) : ℕ := n
def expression_value (n : ℕ) : ℕ := minuend n - subtrahend n

-- Theorem: The value of the 40th expression in the sequence is 21
theorem value_of_40th_expression : expression_value 40 = 21 := by
  show 100 - (40 - 1) - 40 = 21
  sorry

end value_of_40th_expression_l70_7034


namespace must_be_true_if_not_all_electric_l70_7052

variable (P : Type) (ElectricCar : P → Prop)

theorem must_be_true_if_not_all_electric (h : ¬ ∀ x : P, ElectricCar x) : 
  ∃ x : P, ¬ ElectricCar x :=
by 
sorry

end must_be_true_if_not_all_electric_l70_7052


namespace unique_friendly_determination_l70_7049

def is_friendly (a b : ℕ → ℕ) : Prop :=
∀ n : ℕ, ∃ i j : ℕ, n = a i * b j ∧ ∀ (k l : ℕ), n = a k * b l → (i = k ∧ j = l)

theorem unique_friendly_determination {a b c : ℕ → ℕ} 
  (h_friend_a_b : is_friendly a b) 
  (h_friend_a_c : is_friendly a c) :
  b = c :=
sorry

end unique_friendly_determination_l70_7049


namespace average_weight_of_boys_l70_7025

theorem average_weight_of_boys 
  (n1 n2 : ℕ) 
  (w1 w2 : ℝ) 
  (h1 : n1 = 22) 
  (h2 : n2 = 8) 
  (h3 : w1 = 50.25) 
  (h4 : w2 = 45.15) : 
  (n1 * w1 + n2 * w2) / (n1 + n2) = 48.89 :=
by
  sorry

end average_weight_of_boys_l70_7025


namespace quadratic_function_expression_l70_7072

theorem quadratic_function_expression : 
  ∃ (a : ℝ), (a ≠ 0) ∧ (∀ x : ℝ, x = -1 → x * (a * (x + 1) * (x - 2)) = 0) ∧
  (∀ x : ℝ, x = 2 → x * (a * (x + 1) * (x - 2)) = 0) ∧
  (∀ y : ℝ, ∃ x : ℝ, x = 0 ∧ y = -2 → y = a * (x + 1) * (x - 2)) 
  → (∀ x : ℝ, ∃ y : ℝ, y = x^2 - x - 2) := 
sorry

end quadratic_function_expression_l70_7072


namespace linear_function_decreases_l70_7003

theorem linear_function_decreases (m b x : ℝ) (h : m < 0) : 
  ∃ y : ℝ, y = m * x + b ∧ ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m * x₁ + b) > (m * x₂ + b) :=
by 
  sorry

end linear_function_decreases_l70_7003


namespace prime_m_l70_7036

theorem prime_m (m : ℕ) (hm : m ≥ 2) :
  (∀ n : ℕ, (m / 3 ≤ n) → (n ≤ m / 2) → (n ∣ Nat.choose n (m - 2 * n))) → Nat.Prime m :=
by
  intro h
  sorry

end prime_m_l70_7036


namespace Cameron_books_proof_l70_7065

noncomputable def Cameron_initial_books :=
  let B : ℕ := 24
  let B_donated := B / 4
  let B_left := B - B_donated
  let C_donated (C : ℕ) := C / 3
  let C_left (C : ℕ) := C - C_donated C
  ∃ C : ℕ, B_left + C_left C = 38 ∧ C = 30

-- Note that we use sorry to indicate the proof is omitted.
theorem Cameron_books_proof : Cameron_initial_books :=
by {
  sorry
}

end Cameron_books_proof_l70_7065


namespace solve_simultaneous_equations_l70_7029

theorem solve_simultaneous_equations (a b : ℚ) : 
  (a + b) * (a^2 - b^2) = 4 ∧ (a - b) * (a^2 + b^2) = 5 / 2 → 
  (a = 3 / 2 ∧ b = 1 / 2) ∨ (a = -1 / 2 ∧ b = -3 / 2) :=
by
  sorry

end solve_simultaneous_equations_l70_7029


namespace ratio_of_vegetables_to_beef_l70_7081

variable (amountBeefInitial : ℕ) (amountBeefUnused : ℕ) (amountVegetables : ℕ)

def amount_beef_used (initial unused : ℕ) : ℕ := initial - unused
def ratio_vegetables_beef (vegetables beef : ℕ) : ℚ := vegetables / beef

theorem ratio_of_vegetables_to_beef 
  (h1 : amountBeefInitial = 4)
  (h2 : amountBeefUnused = 1)
  (h3 : amountVegetables = 6) :
  ratio_vegetables_beef amountVegetables (amount_beef_used amountBeefInitial amountBeefUnused) = 2 :=
by
  sorry

end ratio_of_vegetables_to_beef_l70_7081


namespace quadratic_general_form_l70_7044

theorem quadratic_general_form (x : ℝ) :
  x * (x + 2) = 5 * (x - 2) → x^2 - 3 * x - 10 = 0 := by
  sorry

end quadratic_general_form_l70_7044


namespace find_angle_D_l70_7082

theorem find_angle_D (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : A + B + C + D = 360) : D = 60 :=
sorry

end find_angle_D_l70_7082


namespace perfect_squares_of_nat_l70_7038

theorem perfect_squares_of_nat (a b c : ℕ) (h : a^2 + b^2 + c^2 = (a - b)^2 + (b - c)^2 + (c - a)^2) :
  ∃ m n p q : ℕ, ab = m^2 ∧ bc = n^2 ∧ ca = p^2 ∧ ab + bc + ca = q^2 :=
by sorry

end perfect_squares_of_nat_l70_7038
