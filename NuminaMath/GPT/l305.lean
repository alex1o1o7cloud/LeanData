import Mathlib

namespace NUMINAMATH_GPT_chord_length_of_intersecting_circle_and_line_l305_30531

-- Define the conditions in Lean
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ
def line_equation (ρ θ : ℝ) : Prop := 3 * ρ * Real.cos θ - 4 * ρ * Real.sin θ - 1 = 0

-- Define the problem to prove the length of the chord
theorem chord_length_of_intersecting_circle_and_line 
  (ρ θ : ℝ) (hC : circle_equation ρ θ) (hL : line_equation ρ θ) : 
  ∃ l : ℝ, l = 2 * Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_chord_length_of_intersecting_circle_and_line_l305_30531


namespace NUMINAMATH_GPT_kirsty_initial_models_l305_30513

theorem kirsty_initial_models 
  (x : ℕ)
  (initial_price : ℝ)
  (increased_price : ℝ)
  (models_bought : ℕ)
  (h_initial_price : initial_price = 0.45)
  (h_increased_price : increased_price = 0.5)
  (h_models_bought : models_bought = 27) 
  (h_total_saved : x * initial_price = models_bought * increased_price) :
  x = 30 :=
by 
  sorry

end NUMINAMATH_GPT_kirsty_initial_models_l305_30513


namespace NUMINAMATH_GPT_find_dimensions_l305_30584

theorem find_dimensions (x y : ℝ) 
  (h1 : 90 = (2 * x + y) * (2 * y))
  (h2 : x * y = 10) : x = 2 ∧ y = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_dimensions_l305_30584


namespace NUMINAMATH_GPT_smallest_q_for_5_in_range_l305_30593

theorem smallest_q_for_5_in_range : ∃ q, (q = 9) ∧ (∃ x, (x^2 - 4 * x + q = 5)) := 
by 
  sorry

end NUMINAMATH_GPT_smallest_q_for_5_in_range_l305_30593


namespace NUMINAMATH_GPT_coefficient_of_xy6_eq_one_l305_30586

theorem coefficient_of_xy6_eq_one (a : ℚ) (h : (7 : ℚ) * a = 1) : a = 1 / 7 :=
by sorry

end NUMINAMATH_GPT_coefficient_of_xy6_eq_one_l305_30586


namespace NUMINAMATH_GPT_inscribed_squares_ratio_l305_30576

theorem inscribed_squares_ratio (a b : ℝ) (h_triangle : 5^2 + 12^2 = 13^2)
    (h_square1 : a = 25 / 37) (h_square2 : b = 10) :
    a / b = 25 / 370 :=
by 
  sorry

end NUMINAMATH_GPT_inscribed_squares_ratio_l305_30576


namespace NUMINAMATH_GPT_room_width_is_7_l305_30578

-- Define the conditions of the problem
def room_length : ℝ := 10
def room_height : ℝ := 5
def door_width : ℝ := 1
def door_height : ℝ := 3
def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5
def cost_per_sq_meter : ℝ := 3
def total_cost : ℝ := 474

-- Define the total cost to be painted
def total_area_painted (width : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height) + 2 * (width * room_height)
  let door_area := 2 * (door_width * door_height)
  let window_area := (window1_width * window1_height) + 2 * (window2_width * window2_height)
  wall_area - door_area - window_area

def cost_equation (width : ℝ) : Prop :=
  (total_cost / cost_per_sq_meter) = total_area_painted width

-- Prove that the width required to satisfy the painting cost equation is 7 meters
theorem room_width_is_7 : ∃ w : ℝ, cost_equation w ∧ w = 7 :=
by
  sorry

end NUMINAMATH_GPT_room_width_is_7_l305_30578


namespace NUMINAMATH_GPT_probability_at_least_3_l305_30592

noncomputable def probability_hitting_at_least_3_of_4 (p : ℝ) (n : ℕ) : ℝ :=
  let p3 := (Nat.choose n 3) * (p^3) * ((1 - p)^(n - 3))
  let p4 := (Nat.choose n 4) * (p^4)
  p3 + p4

theorem probability_at_least_3 (h : probability_hitting_at_least_3_of_4 0.8 4 = 0.8192) : 
   True :=
by trivial

end NUMINAMATH_GPT_probability_at_least_3_l305_30592


namespace NUMINAMATH_GPT_initial_mean_of_observations_l305_30523

theorem initial_mean_of_observations (M : ℝ) (h1 : 50 * M + 30 = 50 * 40.66) : M = 40.06 := 
sorry

end NUMINAMATH_GPT_initial_mean_of_observations_l305_30523


namespace NUMINAMATH_GPT_remaining_files_calc_l305_30597

-- Definitions based on given conditions
def music_files : ℕ := 27
def video_files : ℕ := 42
def deleted_files : ℕ := 11

-- Theorem statement to prove the number of remaining files
theorem remaining_files_calc : music_files + video_files - deleted_files = 58 := by
  sorry

end NUMINAMATH_GPT_remaining_files_calc_l305_30597


namespace NUMINAMATH_GPT_plane_hover_central_time_l305_30518

theorem plane_hover_central_time (x : ℕ) (h1 : 3 + x + 2 + 5 + (x + 2) + 4 = 24) : x = 4 := by
  sorry

end NUMINAMATH_GPT_plane_hover_central_time_l305_30518


namespace NUMINAMATH_GPT_coplanar_k_values_l305_30538

noncomputable def coplanar_lines_possible_k (k : ℝ) : Prop :=
  ∃ (t u : ℝ), (2 + t = 1 + k * u) ∧ (3 + t = 4 + 2 * u) ∧ (4 - k * t = 5 + u)

theorem coplanar_k_values :
  ∀ k : ℝ, coplanar_lines_possible_k k ↔ (k = 0 ∨ k = -3) :=
by
  sorry

end NUMINAMATH_GPT_coplanar_k_values_l305_30538


namespace NUMINAMATH_GPT_num_valid_pairs_l305_30535

theorem num_valid_pairs (a b : ℕ) (h1 : b > a) (h2 : a > 4) (h3 : b > 4)
(h4 : a * b = 3 * (a - 4) * (b - 4)) : 
    (1 + (a - 6) = 1 ∧ 72 = b - 6) ∨
    (2 + (a - 6) = 2 ∧ 36 = b - 6) ∨
    (3 + (a - 6) = 3 ∧ 24 = b - 6) ∨
    (4 + (a - 6) = 4 ∧ 18 = b - 6) :=
sorry

end NUMINAMATH_GPT_num_valid_pairs_l305_30535


namespace NUMINAMATH_GPT_no_non_similar_triangles_with_geometric_angles_l305_30505

theorem no_non_similar_triangles_with_geometric_angles :
  ¬∃ (a r : ℕ), a > 0 ∧ r > 0 ∧ a ≠ a * r ∧ a ≠ a * r * r ∧ a * r ≠ a * r * r ∧
  a + a * r + a * r * r = 180 :=
by
  sorry

end NUMINAMATH_GPT_no_non_similar_triangles_with_geometric_angles_l305_30505


namespace NUMINAMATH_GPT_largest_three_digit_geometric_sequence_with_8_l305_30527

theorem largest_three_digit_geometric_sequence_with_8 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n = 842 ∧ (∃ (a b c : ℕ), n = 100*a + 10*b + c ∧ a = 8 ∧ (a * c = b^2) ∧ (a ≠ b ∧ b ≠ c ∧ a ≠ c) ) :=
by
  sorry

end NUMINAMATH_GPT_largest_three_digit_geometric_sequence_with_8_l305_30527


namespace NUMINAMATH_GPT_arithmetic_sequence_a2_a6_l305_30545

theorem arithmetic_sequence_a2_a6 (a : ℕ → ℕ) (d : ℕ) (h_arith_seq : ∀ n, a (n+1) = a n + d)
  (h_a4 : a 4 = 4) : a 2 + a 6 = 8 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2_a6_l305_30545


namespace NUMINAMATH_GPT_jill_study_hours_l305_30552

theorem jill_study_hours (x : ℕ) (h_condition : x + 2*x + (2*x - 1) = 9) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_jill_study_hours_l305_30552


namespace NUMINAMATH_GPT_true_proposition_l305_30557

-- Define proposition p
def p : Prop := ∀ x : ℝ, Real.log (x^2 + 4) / Real.log 2 ≥ 2

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ≥ 0 → x^(1/2) ≤ x^(1/2)

-- Theorem: true proposition is p ∨ ¬q
theorem true_proposition : p ∨ ¬q :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_l305_30557


namespace NUMINAMATH_GPT_suitable_for_comprehensive_survey_l305_30524

-- Define the four survey options as a custom data type
inductive SurveyOption
  | A : SurveyOption -- Survey on the water quality of the Beijiang River
  | B : SurveyOption -- Survey on the quality of rice dumplings in the market during the Dragon Boat Festival
  | C : SurveyOption -- Survey on the vision of 50 students in a class
  | D : SurveyOption -- Survey by energy-saving lamp manufacturers on the service life of a batch of energy-saving lamps

-- Define feasibility for a comprehensive survey
def isComprehensiveSurvey (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.A => False
  | SurveyOption.B => False
  | SurveyOption.C => True
  | SurveyOption.D => False

-- The statement to be proven
theorem suitable_for_comprehensive_survey : ∃! o : SurveyOption, isComprehensiveSurvey o := by
  sorry

end NUMINAMATH_GPT_suitable_for_comprehensive_survey_l305_30524


namespace NUMINAMATH_GPT_inequality_proof_l305_30579

variables (a b c : ℝ)

theorem inequality_proof
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (cond : a^2 + b^2 + c^2 + ab + bc + ca ≤ 2) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l305_30579


namespace NUMINAMATH_GPT_solution_set_f_l305_30501

def f (x a b : ℝ) : ℝ := (x - 2) * (a * x + b)

theorem solution_set_f (a b : ℝ) (h1 : b = 2 * a) (h2 : 0 < a) :
  {x | f (2 - x) a b > 0} = {x | x < 0 ∨ 4 < x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f_l305_30501


namespace NUMINAMATH_GPT_cadence_total_earnings_l305_30534

/-- Cadence's total earnings in both companies. -/
def total_earnings (old_salary_per_month new_salary_per_month : ℕ) (old_company_months new_company_months : ℕ) : ℕ :=
  (old_salary_per_month * old_company_months) + (new_salary_per_month * new_company_months)

theorem cadence_total_earnings :
  let old_salary_per_month := 5000
  let old_company_years := 3
  let months_per_year := 12
  let old_company_months := old_company_years * months_per_year
  let new_salary_per_month := old_salary_per_month + (old_salary_per_month * 20 / 100)
  let new_company_extra_months := 5
  let new_company_months := old_company_months + new_company_extra_months
  total_earnings old_salary_per_month new_salary_per_month old_company_months new_company_months = 426000 := by
sorry

end NUMINAMATH_GPT_cadence_total_earnings_l305_30534


namespace NUMINAMATH_GPT_inhabitable_fraction_of_mars_surface_l305_30599

theorem inhabitable_fraction_of_mars_surface :
  (3 / 5 : ℚ) * (2 / 3) = (2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_inhabitable_fraction_of_mars_surface_l305_30599


namespace NUMINAMATH_GPT_Seokjin_total_problems_l305_30550

theorem Seokjin_total_problems (initial_problems : ℕ) (additional_problems : ℕ)
  (h1 : initial_problems = 12) (h2 : additional_problems = 7) :
  initial_problems + additional_problems = 19 :=
by
  sorry

end NUMINAMATH_GPT_Seokjin_total_problems_l305_30550


namespace NUMINAMATH_GPT_prime_N_k_iff_k_eq_2_l305_30569

-- Define the function to generate the number N_k based on k
def N_k (k : ℕ) : ℕ := (10^(2 * k) - 1) / 99

-- Define the main theorem to prove
theorem prime_N_k_iff_k_eq_2 (k : ℕ) : Nat.Prime (N_k k) ↔ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_prime_N_k_iff_k_eq_2_l305_30569


namespace NUMINAMATH_GPT_vector_minimization_and_angle_condition_l305_30567

noncomputable def find_OC_condition (C_op C_oa C_ob : ℝ × ℝ) 
  (C : ℝ × ℝ) : Prop := 
  let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
  let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
  (CA.1 * CB.1 + CA.2 * CB.2) ≤ (C_op.1 * CB.1 + C_op.2 * CB.2)

theorem vector_minimization_and_angle_condition (C : ℝ × ℝ) 
  (C_op := (2, 1)) (C_oa := (1, 7)) (C_ob := (5, 1)) :
  (C = (4, 2)) → 
  find_OC_condition C_op C_oa C_ob C →
  let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
  let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
  let cos_ACB := (CA.1 * CB.1 + CA.2 * CB.2) / 
                 (Real.sqrt (CA.1^2 + CA.2^2) * Real.sqrt (CB.1^2 + CB.2^2))
  cos_ACB = -4 * Real.sqrt (17) / 17 :=
  by 
    intro h1 find
    let CA := (C_oa.1 - C.1, C_oa.2 - C.2)
    let CB := (C_ob.1 - C.1, C_ob.2 - C.2)
    let cos_ACB := (CA.1 * CB.1 + CA.2 * CB.2) / 
                   (Real.sqrt (CA.1^2 + CA.2^2) * Real.sqrt (CB.1^2 + CB.2^2))
    exact sorry

end NUMINAMATH_GPT_vector_minimization_and_angle_condition_l305_30567


namespace NUMINAMATH_GPT_subset_condition_l305_30500

variable {U : Type}
variables (P Q : Set U)

theorem subset_condition (h : P ∩ Q = P) : ∀ x : U, x ∉ Q → x ∉ P :=
by {
  sorry
}

end NUMINAMATH_GPT_subset_condition_l305_30500


namespace NUMINAMATH_GPT_opposite_2024_eq_neg_2024_l305_30571

def opposite (n : ℤ) : ℤ := -n

theorem opposite_2024_eq_neg_2024 : opposite 2024 = -2024 :=
by
  sorry

end NUMINAMATH_GPT_opposite_2024_eq_neg_2024_l305_30571


namespace NUMINAMATH_GPT_negation_relation_l305_30562

def p (x : ℝ) : Prop := x < -1 ∨ x > 1
def q (x : ℝ) : Prop := x < -2 ∨ x > 1

def not_p (x : ℝ) : Prop := x ≥ -1 ∧ x ≤ 1
def not_q (x : ℝ) : Prop := x ≥ -2 ∧ x ≤ 1

theorem negation_relation : (∀ x, not_p x → not_q x) ∧ ¬ (∀ x, not_q x → not_p x) :=
by 
  sorry

end NUMINAMATH_GPT_negation_relation_l305_30562


namespace NUMINAMATH_GPT_cos_135_eq_neg_sqrt2_div_2_l305_30542

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_cos_135_eq_neg_sqrt2_div_2_l305_30542


namespace NUMINAMATH_GPT_birds_joined_l305_30555

def numBirdsInitially : Nat := 1
def numBirdsNow : Nat := 5

theorem birds_joined : numBirdsNow - numBirdsInitially = 4 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_birds_joined_l305_30555


namespace NUMINAMATH_GPT_right_triangle_c_l305_30528

theorem right_triangle_c (a b c : ℝ) (h1 : a = 3) (h2 : b = 4)
  (h3 : (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2)) :
  c = 5 ∨ c = Real.sqrt 7 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_right_triangle_c_l305_30528


namespace NUMINAMATH_GPT_shooting_challenge_sequences_l305_30560

theorem shooting_challenge_sequences : ∀ (A B C : ℕ), 
  A = 4 → B = 4 → C = 2 →
  (A + B + C = 10) →
  (Nat.factorial (A + B + C) / (Nat.factorial A * Nat.factorial B * Nat.factorial C) = 3150) :=
by
  intros A B C hA hB hC hsum
  sorry

end NUMINAMATH_GPT_shooting_challenge_sequences_l305_30560


namespace NUMINAMATH_GPT_determine_function_l305_30548

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (1/2) * (x^2 + (1/x)) else 0

theorem determine_function (f: ℝ → ℝ) (h : ∀ x ≠ 0, (1/x) * f (-x) + f (1/x) = x ) :
  ∀ x ≠ 0, f x = (1/2) * (x^2 + (1/x)) :=
by
  sorry

end NUMINAMATH_GPT_determine_function_l305_30548


namespace NUMINAMATH_GPT_find_b_value_l305_30521

def perfect_square_trinomial (a b c : ℕ) : Prop :=
  ∃ d, a = d^2 ∧ c = d^2 ∧ b = 2 * d * d

theorem find_b_value (b : ℝ) :
    (∀ x : ℝ, 16 * x^2 - b * x + 9 = (4 * x - 3) * (4 * x - 3) ∨ 16 * x^2 - b * x + 9 = (4 * x + 3) * (4 * x + 3)) -> 
    b = 24 ∨ b = -24 := 
by
  sorry

end NUMINAMATH_GPT_find_b_value_l305_30521


namespace NUMINAMATH_GPT_value_of_x2_plus_4y2_l305_30551

theorem value_of_x2_plus_4y2 (x y : ℝ) (h1 : x + 2 * y = 6) (h2 : x * y = -12) : x^2 + 4*y^2 = 84 := 
  sorry

end NUMINAMATH_GPT_value_of_x2_plus_4y2_l305_30551


namespace NUMINAMATH_GPT_average_speed_ratio_l305_30596

theorem average_speed_ratio
  (time_eddy : ℕ)
  (time_freddy : ℕ)
  (distance_ab : ℕ)
  (distance_ac : ℕ)
  (h1 : time_eddy = 3)
  (h2 : time_freddy = 4)
  (h3 : distance_ab = 570)
  (h4 : distance_ac = 300) :
  (distance_ab / time_eddy) / (distance_ac / time_freddy) = 38 / 15 := 
by
  sorry

end NUMINAMATH_GPT_average_speed_ratio_l305_30596


namespace NUMINAMATH_GPT_prop_p_necessary_but_not_sufficient_for_prop_q_l305_30507

theorem prop_p_necessary_but_not_sufficient_for_prop_q (x y : ℕ) :
  (x ≠ 1 ∨ y ≠ 3) → (x + y ≠ 4) → ((x+y ≠ 4) → (x ≠ 1 ∨ y ≠ 3)) ∧ ¬ ((x ≠ 1 ∨ y ≠ 3) → (x + y ≠ 4)) :=
by
  sorry

end NUMINAMATH_GPT_prop_p_necessary_but_not_sufficient_for_prop_q_l305_30507


namespace NUMINAMATH_GPT_intersection_M_N_l305_30529

noncomputable def M : Set ℝ := {x | x^2 + x - 6 < 0}
noncomputable def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N :
  {x : ℝ | M x ∧ N x } = {x : ℝ | 1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l305_30529


namespace NUMINAMATH_GPT_inequality_logarithms_l305_30508

noncomputable def a : ℝ := Real.log 3.6 / Real.log 2
noncomputable def b : ℝ := Real.log 3.2 / Real.log 4
noncomputable def c : ℝ := Real.log 3.6 / Real.log 4

theorem inequality_logarithms : a > c ∧ c > b :=
by
  -- the proof will be written here
  sorry

end NUMINAMATH_GPT_inequality_logarithms_l305_30508


namespace NUMINAMATH_GPT_find_sum_l305_30506

-- Defining the conditions of the problem
variables (P r t : ℝ) 
theorem find_sum 
  (h1 : (P * r * t) / 100 = 88) 
  (h2 : (P * r * t) / (100 + (r * t)) = 80) 
  : P = 880 := 
sorry

end NUMINAMATH_GPT_find_sum_l305_30506


namespace NUMINAMATH_GPT_trig_identity_l305_30585

open Real

theorem trig_identity (α : ℝ) (hα : α > -π ∧ α < -π/2) :
  (sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α))) = - 2 / tan α :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_l305_30585


namespace NUMINAMATH_GPT_set_intersection_complement_l305_30565

open Set

def I := {n : ℕ | True}
def A := {x ∈ I | 2 ≤ x ∧ x ≤ 10}
def B := {x | Nat.Prime x}

theorem set_intersection_complement :
  A ∩ (I \ B) = {4, 6, 8, 9, 10} := by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l305_30565


namespace NUMINAMATH_GPT_find_k_no_xy_term_l305_30543

theorem find_k_no_xy_term (k : ℝ) :
  (¬ ∃ x y : ℝ, (-x^2 - 3 * k * x * y - 3 * y^2 + 9 * x * y - 8) = (- x^2 - 3 * y^2 - 8)) → k = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_no_xy_term_l305_30543


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l305_30573

variables {a : ℕ → ℤ} {S T : ℕ → ℤ}

theorem arithmetic_sequence_properties 
  (h₁ : a 2 = 11)
  (h₂ : S 10 = 40)
  (h₃ : ∀ n, S n = n * a 1 + (n * (n - 1)) / 2 * (a 2 - a 1)) -- Sum of first n terms of arithmetic sequence
  (h₄ : ∀ k, a k = a 1 + (k - 1) * (a 2 - a 1)) -- General term formula of arithmetic sequence
  : (∀ n, a n = -2 * n + 15) ∧
    ( (∀ n, 1 ≤ n ∧ n ≤ 7 → T n = -n^2 + 14 * n) ∧ 
      (∀ n, n ≥ 8 → T n = n^2 - 14 * n + 98)) :=
by
sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l305_30573


namespace NUMINAMATH_GPT_correct_options_l305_30537

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x + Real.pi / 3) + 1

def option_2 : Prop := ∃ x : ℝ, f (x) = 0 ∧ x = Real.pi / 3
def option_3 : Prop := ∀ T > 0, (∀ x : ℝ, f (x) = f (x + T)) → T = Real.pi
def option_5 : Prop := ∀ x : ℝ, f (x - Real.pi / 6) = f (-(x - Real.pi / 6))

theorem correct_options :
  option_2 ∧ option_3 ∧ option_5 :=
by
  sorry

end NUMINAMATH_GPT_correct_options_l305_30537


namespace NUMINAMATH_GPT_determine_conflicting_pairs_l305_30511

structure EngineerSetup where
  n : ℕ
  barrels : Fin (2 * n) → Reactant
  conflicts : Fin n → (Reactant × Reactant)

def testTubeBurst (r1 r2 : Reactant) (conflicts : Fin n → (Reactant × Reactant)) : Prop :=
  ∃ i, conflicts i = (r1, r2) ∨ conflicts i = (r2, r1)

theorem determine_conflicting_pairs (setup : EngineerSetup) :
  ∃ pairs : Fin n → (Reactant × Reactant),
  (∀ i, pairs i ∈ { p | ∃ j, setup.conflicts j = p ∨ setup.conflicts j = (p.snd, p.fst) }) ∧
  (∀ i j, i ≠ j → pairs i ≠ pairs j) := 
sorry

end NUMINAMATH_GPT_determine_conflicting_pairs_l305_30511


namespace NUMINAMATH_GPT_total_metal_rods_needed_l305_30539

-- Definitions extracted from the conditions
def metal_sheets_per_panel := 3
def metal_beams_per_panel := 2
def panels := 10
def rods_per_sheet := 10
def rods_per_beam := 4

-- Problem statement: Prove the total number of metal rods required is 380
theorem total_metal_rods_needed :
  (panels * ((metal_sheets_per_panel * rods_per_sheet) + (metal_beams_per_panel * rods_per_beam))) = 380 :=
by
  sorry

end NUMINAMATH_GPT_total_metal_rods_needed_l305_30539


namespace NUMINAMATH_GPT_equal_probabilities_l305_30517

-- Definitions based on the conditions in the problem

def total_parts : ℕ := 160
def first_class_parts : ℕ := 48
def second_class_parts : ℕ := 64
def third_class_parts : ℕ := 32
def substandard_parts : ℕ := 16
def sample_size : ℕ := 20

-- Define the probabilities for each sampling method
def p1 : ℚ := sample_size / total_parts
def p2 : ℚ := (6 : ℚ) / first_class_parts  -- Given the conditions, this will hold for all classes
def p3 : ℚ := 1 / 8

theorem equal_probabilities :
  p1 = p2 ∧ p2 = p3 :=
by
  -- This is the end of the statement as no proof is required
  sorry

end NUMINAMATH_GPT_equal_probabilities_l305_30517


namespace NUMINAMATH_GPT_markers_in_desk_l305_30549

theorem markers_in_desk (pens pencils markers : ℕ) 
  (h_ratio : pens = 2 * pencils ∧ pens = 2 * markers / 5) 
  (h_pens : pens = 10) : markers = 25 :=
by
  sorry

end NUMINAMATH_GPT_markers_in_desk_l305_30549


namespace NUMINAMATH_GPT_exists_three_irrationals_l305_30581

theorem exists_three_irrationals
    (x1 x2 x3 : ℝ)
    (h1 : ¬ ∃ q : ℚ, x1 = q)
    (h2 : ¬ ∃ q : ℚ, x2 = q)
    (h3 : ¬ ∃ q : ℚ, x3 = q)
    (sum_integer : ∃ n : ℤ, x1 + x2 + x3 = n)
    (sum_reciprocals_integer : ∃ m : ℤ, (1/x1) + (1/x2) + (1/x3) = m) :
  true :=
sorry

end NUMINAMATH_GPT_exists_three_irrationals_l305_30581


namespace NUMINAMATH_GPT_Ms_Smiths_Class_Books_Distribution_l305_30514

theorem Ms_Smiths_Class_Books_Distribution :
  ∃ (x : ℕ), (20 * 2 * x + 15 * x + 5 * x = 840) ∧ (20 * 2 * x = 560) ∧ (15 * x = 210) ∧ (5 * x = 70) :=
by
  let x := 14
  have h1 : 20 * 2 * x + 15 * x + 5 * x = 840 := by sorry
  have h2 : 20 * 2 * x = 560 := by sorry
  have h3 : 15 * x = 210 := by sorry
  have h4 : 5 * x = 70 := by sorry
  exact ⟨x, h1, h2, h3, h4⟩

end NUMINAMATH_GPT_Ms_Smiths_Class_Books_Distribution_l305_30514


namespace NUMINAMATH_GPT_largest_five_digit_divisible_by_97_l305_30574

theorem largest_five_digit_divisible_by_97 :
  ∃ n, (99999 - n % 97) = 99930 ∧ n % 97 = 0 ∧ 10000 ≤ n ∧ n ≤ 99999 :=
by
  sorry

end NUMINAMATH_GPT_largest_five_digit_divisible_by_97_l305_30574


namespace NUMINAMATH_GPT_max_sum_is_1717_l305_30559

noncomputable def max_arithmetic_sum (a d : ℤ) : ℤ :=
  let n := 34
  let S : ℤ := n * (2*a + (n - 1)*d) / 2
  S

theorem max_sum_is_1717 (a d : ℤ) (h1 : a + 16 * d = 52) (h2 : a + 29 * d = 13) (hd : d = -3) (ha : a = 100) :
  max_arithmetic_sum a d = 1717 :=
by
  unfold max_arithmetic_sum
  rw [hd, ha]
  -- Add the necessary steps to prove max_arithmetic_sum 100 (-3) = 1717
  -- Sorry ensures the theorem can be checked syntactically without proof
  sorry

end NUMINAMATH_GPT_max_sum_is_1717_l305_30559


namespace NUMINAMATH_GPT_geom_seq_sum_l305_30580

theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < q)
  (h2 : ∀ n, a (n+1) = a n * q)
  (h3 : a 0 + a 1 = 3 / 4)
  (h4 : a 2 + a 3 + a 4 + a 5 = 15) :
  a 6 + a 7 + a 8 = 112 := by
  sorry

end NUMINAMATH_GPT_geom_seq_sum_l305_30580


namespace NUMINAMATH_GPT_compare_A_B_l305_30525

noncomputable def A (x : ℝ) := x / (x^2 - x + 1)
noncomputable def B (y : ℝ) := y / (y^2 - y + 1)

theorem compare_A_B (x y : ℝ) (hx : x > y) (hx_val : x = 2.00 * 10^1998 + 4) (hy_val : y = 2.00 * 10^1998 + 2) : 
  A x < B y := 
by 
  sorry

end NUMINAMATH_GPT_compare_A_B_l305_30525


namespace NUMINAMATH_GPT_max_n_for_factorable_polynomial_l305_30512

theorem max_n_for_factorable_polynomial : 
  ∃ n : ℤ, (∀ A B : ℤ, AB = 108 → n = 6 * B + A) ∧ n = 649 :=
by
  sorry

end NUMINAMATH_GPT_max_n_for_factorable_polynomial_l305_30512


namespace NUMINAMATH_GPT_johns_age_in_8_years_l305_30515

theorem johns_age_in_8_years :
  let current_age := 18
  let age_five_years_ago := current_age - 5
  let twice_age_five_years_ago := 2 * age_five_years_ago
  current_age + 8 = twice_age_five_years_ago :=
by
  let current_age := 18
  let age_five_years_ago := current_age - 5
  let twice_age_five_years_ago := 2 * age_five_years_ago
  sorry

end NUMINAMATH_GPT_johns_age_in_8_years_l305_30515


namespace NUMINAMATH_GPT_relationship_of_magnitudes_l305_30577

noncomputable def is_ordered (x : ℝ) (A B C : ℝ) : Prop :=
  0 < x ∧ x < Real.pi / 4 ∧
  A = Real.cos (x ^ Real.sin (x ^ Real.sin x)) ∧
  B = Real.sin (x ^ Real.cos (x ^ Real.sin x)) ∧
  C = Real.cos (x ^ Real.sin (x * (x ^ Real.cos x))) ∧
  B < A ∧ A < C

theorem relationship_of_magnitudes (x A B C : ℝ) : 
  is_ordered x A B C := 
sorry

end NUMINAMATH_GPT_relationship_of_magnitudes_l305_30577


namespace NUMINAMATH_GPT_correct_analogical_reasoning_l305_30503

-- Definitions of the statements in the problem
def statement_A : Prop := ∀ (a b : ℝ), a * 3 = b * 3 → a = b → a * 0 = b * 0 → a = b
def statement_B : Prop := ∀ (a b c : ℝ), (a + b) * c = a * c + b * c → (a * b) * c = a * c * b * c
def statement_C : Prop := ∀ (a b c : ℝ), (a + b) * c = a * c + b * c → c ≠ 0 → (a + b) / c = a / c + b / c
def statement_D : Prop := ∀ (a b : ℝ) (n : ℕ), (a * b)^n = a^n * b^n → (a + b)^n = a^n + b^n

-- The theorem stating that option C is the only correct analogical reasoning
theorem correct_analogical_reasoning : statement_C ∧ ¬statement_A ∧ ¬statement_B ∧ ¬statement_D := by
  sorry

end NUMINAMATH_GPT_correct_analogical_reasoning_l305_30503


namespace NUMINAMATH_GPT_geo_seq_sum_condition_l305_30595

noncomputable def geometric_seq (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^n

noncomputable def sum_geo_seq_3 (a : ℝ) (q : ℝ) : ℝ :=
  geometric_seq a q 0 + geometric_seq a q 1 + geometric_seq a q 2

noncomputable def sum_geo_seq_6 (a : ℝ) (q : ℝ) : ℝ :=
  sum_geo_seq_3 a q + geometric_seq a q 3 + geometric_seq a q 4 + geometric_seq a q 5

theorem geo_seq_sum_condition {a q S₃ S₆ : ℝ} (h_sum_eq : S₆ = 9 * S₃)
  (h_S₃_def : S₃ = sum_geo_seq_3 a q)
  (h_S₆_def : S₆ = sum_geo_seq_6 a q) :
  q = 2 :=
by
  sorry

end NUMINAMATH_GPT_geo_seq_sum_condition_l305_30595


namespace NUMINAMATH_GPT_ap_contains_sixth_power_l305_30570

theorem ap_contains_sixth_power (a d : ℕ) (i j x y : ℕ) 
  (h_positive : ∀ n, a + n * d > 0) 
  (h_square : a + i * d = x^2) 
  (h_cube : a + j * d = y^3) :
  ∃ k z : ℕ, a + k * d = z^6 := 
  sorry

end NUMINAMATH_GPT_ap_contains_sixth_power_l305_30570


namespace NUMINAMATH_GPT_sum_of_numbers_l305_30566

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def tens_digit_zero (n : ℕ) : Prop := (n / 10) % 10 = 0
def units_digit_nonzero (n : ℕ) : Prop := n % 10 ≠ 0
def same_units_digits (m n : ℕ) : Prop := m % 10 = n % 10

theorem sum_of_numbers (a b c : ℕ)
  (h1 : is_perfect_square a) (h2 : is_perfect_square b) (h3 : is_perfect_square c)
  (h4 : tens_digit_zero a) (h5 : tens_digit_zero b) (h6 : tens_digit_zero c)
  (h7 : units_digit_nonzero a) (h8 : units_digit_nonzero b) (h9 : units_digit_nonzero c)
  (h10 : same_units_digits b c)
  (h11 : a % 10 % 2 = 0) :
  a + b + c = 14612 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l305_30566


namespace NUMINAMATH_GPT_checker_move_10_cells_checker_move_11_cells_l305_30575

noncomputable def F : ℕ → Nat 
| 0 => 1
| 1 => 1
| n + 2 => F (n + 1) + F n

theorem checker_move_10_cells : F 10 = 89 := by
  sorry

theorem checker_move_11_cells : F 11 = 144 := by
  sorry

end NUMINAMATH_GPT_checker_move_10_cells_checker_move_11_cells_l305_30575


namespace NUMINAMATH_GPT_coin_collection_problem_l305_30536

theorem coin_collection_problem (n : ℕ) 
  (quarters : ℕ := n / 2)
  (half_dollars : ℕ := 2 * (n / 2))
  (value_nickels : ℝ := 0.05 * n)
  (value_quarters : ℝ := 0.25 * (n / 2))
  (value_half_dollars : ℝ := 0.5 * (2 * (n / 2)))
  (total_value : ℝ := value_nickels + value_quarters + value_half_dollars) :
  total_value = 67.5 ∨ total_value = 135 :=
sorry

end NUMINAMATH_GPT_coin_collection_problem_l305_30536


namespace NUMINAMATH_GPT_relationship_between_problems_geometry_problem_count_steve_questions_l305_30583

variable (x y W A G : ℕ)

def word_problems (x : ℕ) : ℕ := x / 2
def addition_and_subtraction_problems (x : ℕ) : ℕ := x / 3
def geometry_problems (x W A : ℕ) : ℕ := x - W - A

theorem relationship_between_problems :
  W = word_problems x ∧
  A = addition_and_subtraction_problems x ∧
  G = geometry_problems x W A →
  W + A + G = x :=
by
  sorry

theorem geometry_problem_count :
  W = word_problems x ∧
  A = addition_and_subtraction_problems x →
  G = geometry_problems x W A →
  G = x / 6 :=
by
  sorry

theorem steve_questions :
  y = x / 2 - 4 :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_problems_geometry_problem_count_steve_questions_l305_30583


namespace NUMINAMATH_GPT_simplify_expression_l305_30516

theorem simplify_expression (x : ℝ) (h1 : x^3 + 2*x + 1 ≠ 0) (h2 : x^3 - 2*x - 1 ≠ 0) : 
  ( ((x + 2)^2 * (x^2 - x + 2)^2 / (x^3 + 2*x + 1)^2 )^3 * ((x - 2)^2 * (x^2 + x + 2)^2 / (x^3 - 2*x - 1)^2 )^3 ) = 1 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l305_30516


namespace NUMINAMATH_GPT_weight_of_new_person_l305_30563

-- Definitions based on conditions
def average_weight_increase : ℝ := 2.5
def number_of_persons : ℕ := 8
def old_weight : ℝ := 65
def total_weight_increase : ℝ := number_of_persons * average_weight_increase

-- Proposition to prove
theorem weight_of_new_person : (old_weight + total_weight_increase) = 85 := by
  -- add the actual proof here
  sorry

end NUMINAMATH_GPT_weight_of_new_person_l305_30563


namespace NUMINAMATH_GPT_symmetric_polynomial_identity_l305_30554

variable (x y z : ℝ)
def σ1 : ℝ := x + y + z
def σ2 : ℝ := x * y + y * z + z * x
def σ3 : ℝ := x * y * z

theorem symmetric_polynomial_identity : 
  x^3 + y^3 + z^3 = σ1 x y z ^ 3 - 3 * σ1 x y z * σ2 x y z + 3 * σ3 x y z := by
  sorry

end NUMINAMATH_GPT_symmetric_polynomial_identity_l305_30554


namespace NUMINAMATH_GPT_possible_values_of_m_l305_30546

-- Proposition: for all real values of m, if for all real x, x^2 + 2x + 2 - m >= 0 holds, then m must be one of -1, 0, or 1

theorem possible_values_of_m (m : ℝ) 
  (h : ∀ (x : ℝ), x^2 + 2 * x + 2 - m ≥ 0) : m = -1 ∨ m = 0 ∨ m = 1 :=
sorry

end NUMINAMATH_GPT_possible_values_of_m_l305_30546


namespace NUMINAMATH_GPT_sallys_dad_nickels_l305_30590

theorem sallys_dad_nickels :
  ∀ (initial_nickels mother's_nickels total_nickels nickels_from_dad : ℕ), 
    initial_nickels = 7 → 
    mother's_nickels = 2 →
    total_nickels = 18 →
    total_nickels = initial_nickels + mother's_nickels + nickels_from_dad →
    nickels_from_dad = 9 :=
by
  intros initial_nickels mother's_nickels total_nickels nickels_from_dad
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sallys_dad_nickels_l305_30590


namespace NUMINAMATH_GPT_girls_select_same_colored_marble_l305_30589

def probability_same_color (total_white total_black girls boys : ℕ) : ℚ :=
  let prob_white := (total_white * (total_white - 1)) / ((total_white + total_black) * (total_white + total_black - 1))
  let prob_black := (total_black * (total_black - 1)) / ((total_white + total_black) * (total_white + total_black - 1))
  prob_white + prob_black

theorem girls_select_same_colored_marble :
  probability_same_color 2 2 2 2 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_girls_select_same_colored_marble_l305_30589


namespace NUMINAMATH_GPT_find_start_time_l305_30598

def time_first_train_started 
  (distance_pq : ℝ) 
  (speed_train1 : ℝ) 
  (speed_train2 : ℝ) 
  (start_time_train2 : ℝ) 
  (meeting_time : ℝ) 
  (T : ℝ) : ℝ :=
  T

theorem find_start_time 
  (distance_pq : ℝ := 200)
  (speed_train1 : ℝ := 20)
  (speed_train2 : ℝ := 25)
  (start_time_train2 : ℝ := 8)
  (meeting_time : ℝ := 12) 
  : time_first_train_started distance_pq speed_train1 speed_train2 start_time_train2 meeting_time 7 = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_start_time_l305_30598


namespace NUMINAMATH_GPT_symmetric_point_correct_l305_30582

def point : Type := ℝ × ℝ × ℝ

def symmetric_with_respect_to_y_axis (A : point) : point :=
  let (x, y, z) := A
  (-x, y, z)

def A : point := (-4, 8, 6)

theorem symmetric_point_correct :
  symmetric_with_respect_to_y_axis A = (4, 8, 6) := by
  sorry

end NUMINAMATH_GPT_symmetric_point_correct_l305_30582


namespace NUMINAMATH_GPT_polynomial_has_real_root_l305_30588

theorem polynomial_has_real_root (a b : ℝ) :
  ∃ x : ℝ, x^3 + a * x + b = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_has_real_root_l305_30588


namespace NUMINAMATH_GPT_range_of_b_l305_30540

theorem range_of_b (a b x : ℝ) (ha : 0 < a ∧ a ≤ 5 / 4) (hb : 0 < b) :
  (∀ x, |x - a| < b → |x - a^2| < 1 / 2) ↔ 0 < b ∧ b ≤ 3 / 16 :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l305_30540


namespace NUMINAMATH_GPT_ratio_345_iff_arithmetic_sequence_l305_30587

-- Define the variables and the context
variables (a b c : ℕ) -- assuming non-negative integers for simplicity
variable (k : ℕ) -- scaling factor for the 3:4:5 ratio
variable (d : ℕ) -- common difference in the arithmetic sequence

-- Conditions given
def isRightAngledTriangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧ a < b ∧ b < c

def is345Ratio (a b c : ℕ) : Prop :=
  ∃ k, a = 3 * k ∧ b = 4 * k ∧ c = 5 * k

def formsArithmeticSequence (a b c : ℕ) : Prop :=
  ∃ d, b = a + d ∧ c = b + d 

-- The statement to prove: sufficiency and necessity
theorem ratio_345_iff_arithmetic_sequence 
  (h_triangle : isRightAngledTriangle a b c) :
  (is345Ratio a b c ↔ formsArithmeticSequence a b c) :=
sorry

end NUMINAMATH_GPT_ratio_345_iff_arithmetic_sequence_l305_30587


namespace NUMINAMATH_GPT_least_total_acorns_l305_30594

theorem least_total_acorns :
  ∃ a₁ a₂ a₃ : ℕ,
    (∀ k : ℕ, (∃ a₁ a₂ a₃ : ℕ,
      (2 * a₁ / 3 + a₁ % 3 / 3 + a₂ + a₃ / 9) % 6 = 4 * k ∧
      (a₁ / 6 + a₂ / 3 + a₃ / 3 + 8 * a₃ / 18) % 6 = 3 * k ∧
      (a₁ / 6 + 5 * a₂ / 6 + a₃ / 9) % 6 = 2 * k) → k = 630) ∧
    (a₁ + a₂ + a₃) = 630 :=
sorry

end NUMINAMATH_GPT_least_total_acorns_l305_30594


namespace NUMINAMATH_GPT_eliza_is_18_l305_30553

-- Define the relevant ages
def aunt_ellen_age : ℕ := 48
def dina_age : ℕ := aunt_ellen_age / 2
def eliza_age : ℕ := dina_age - 6

-- Theorem to prove Eliza's age is 18
theorem eliza_is_18 : eliza_age = 18 := by
  sorry

end NUMINAMATH_GPT_eliza_is_18_l305_30553


namespace NUMINAMATH_GPT_expression_varies_l305_30510

noncomputable def expr (x : ℝ) : ℝ := (3 * x^2 - 2 * x - 5) / ((x + 2) * (x - 3)) - (5 + x) / ((x + 2) * (x - 3))

theorem expression_varies (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 3) : ∃ y : ℝ, expr x = y ∧ ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → expr x₁ ≠ expr x₂ :=
by
  sorry

end NUMINAMATH_GPT_expression_varies_l305_30510


namespace NUMINAMATH_GPT_cube_volume_l305_30591

/-- Given the perimeter of one face of a cube, proving the volume of the cube -/

theorem cube_volume (h : ∀ (s : ℝ), 4 * s = 28) : (∃ (v : ℝ), v = (7 : ℝ) ^ 3) :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_l305_30591


namespace NUMINAMATH_GPT_arrange_leopards_correct_l305_30544

-- Definitions for conditions
def num_shortest : ℕ := 3
def total_leopards : ℕ := 9
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Calculation of total ways to arrange given conditions
def arrange_leopards (num_shortest : ℕ) (total_leopards : ℕ) : ℕ :=
  let choose2short := (num_shortest * (num_shortest - 1)) / 2
  let arrange2short := 2 * factorial (total_leopards - num_shortest)
  choose2short * arrange2short * factorial (total_leopards - num_shortest)

theorem arrange_leopards_correct :
  arrange_leopards num_shortest total_leopards = 30240 := by
  sorry

end NUMINAMATH_GPT_arrange_leopards_correct_l305_30544


namespace NUMINAMATH_GPT_length_of_second_train_is_correct_l305_30522

noncomputable def convert_kmph_to_mps (speed_kmph: ℕ) : ℝ :=
  speed_kmph * (1000 / 3600)

def train_lengths_and_time
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℕ)
  (speed_second_train_kmph : ℕ)
  (time_to_cross : ℝ)
  (length_second_train : ℝ) : Prop :=
  let speed_first_train_mps := convert_kmph_to_mps speed_first_train_kmph
  let speed_second_train_mps := convert_kmph_to_mps speed_second_train_kmph
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance := relative_speed * time_to_cross
  total_distance = length_first_train + length_second_train

theorem length_of_second_train_is_correct :
  train_lengths_and_time 260 120 80 9 239.95 :=
by
  sorry

end NUMINAMATH_GPT_length_of_second_train_is_correct_l305_30522


namespace NUMINAMATH_GPT_infinitely_many_n_l305_30568

theorem infinitely_many_n (S : Set ℕ) :
  (∀ n ∈ S, n > 0 ∧ (n ∣ 2 ^ (2 ^ n + 1) + 1) ∧ ¬ (n ∣ 2 ^ n + 1)) ∧ S.Infinite :=
sorry

end NUMINAMATH_GPT_infinitely_many_n_l305_30568


namespace NUMINAMATH_GPT_complement_intersection_l305_30547

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3, 4}

-- Define set B
def B : Set ℕ := {2, 3}

-- Define the complement of A with respect to U
def complement_U (s : Set ℕ) : Set ℕ := {x ∈ U | x ∉ s}

-- Define the statement to be proven
theorem complement_intersection :
  (complement_U A ∩ B) = {2} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l305_30547


namespace NUMINAMATH_GPT_brenda_mice_left_l305_30533

theorem brenda_mice_left (litters : ℕ) (mice_per_litter : ℕ) (fraction_to_robbie : ℚ) 
                          (mult_to_pet_store : ℕ) (fraction_to_feeder : ℚ) 
                          (total_mice : ℕ) (to_robbie : ℕ) (to_pet_store : ℕ) 
                          (remaining_after_first_sales : ℕ) (to_feeder : ℕ) (left_after_feeder : ℕ) :
  litters = 3 →
  mice_per_litter = 8 →
  fraction_to_robbie = 1/6 →
  mult_to_pet_store = 3 →
  fraction_to_feeder = 1/2 →
  total_mice = litters * mice_per_litter →
  to_robbie = total_mice * fraction_to_robbie →
  to_pet_store = mult_to_pet_store * to_robbie →
  remaining_after_first_sales = total_mice - to_robbie - to_pet_store →
  to_feeder = remaining_after_first_sales * fraction_to_feeder →
  left_after_feeder = remaining_after_first_sales - to_feeder →
  left_after_feeder = 4 := sorry

end NUMINAMATH_GPT_brenda_mice_left_l305_30533


namespace NUMINAMATH_GPT_base8_arithmetic_l305_30561

def base8_to_base10 (n : Nat) : Nat :=
  sorry -- Placeholder for base 8 to base 10 conversion

def base10_to_base8 (n : Nat) : Nat :=
  sorry -- Placeholder for base 10 to base 8 conversion

theorem base8_arithmetic (n m : Nat) (h1 : base8_to_base10 45 = n) (h2 : base8_to_base10 76 = m) :
  base10_to_base8 ((n * 2) - m) = 14 :=
by
  sorry

end NUMINAMATH_GPT_base8_arithmetic_l305_30561


namespace NUMINAMATH_GPT_sqrt_sequence_convergence_l305_30558

theorem sqrt_sequence_convergence :
  ∃ x : ℝ, (x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_GPT_sqrt_sequence_convergence_l305_30558


namespace NUMINAMATH_GPT_set_intersection_l305_30541

def A (x : ℝ) : Prop := x > 0
def B (x : ℝ) : Prop := x^2 < 4

theorem set_intersection : {x | A x} ∩ {x | B x} = {x | 0 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_set_intersection_l305_30541


namespace NUMINAMATH_GPT_symm_diff_A_B_l305_30502

-- Define sets A and B
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | abs x < 2}

-- Define set difference
def set_diff (S T : Set ℤ) : Set ℤ := {x | x ∈ S ∧ x ∉ T}

-- Define symmetric difference
def symm_diff (S T : Set ℤ) : Set ℤ := (set_diff S T) ∪ (set_diff T S)

-- Define the expression we need to prove
theorem symm_diff_A_B : symm_diff A B = {-1, 0, 2} := by
  sorry

end NUMINAMATH_GPT_symm_diff_A_B_l305_30502


namespace NUMINAMATH_GPT_problem_l305_30526

open Real

def p (x : ℝ) : Prop := 2*x^2 + 2*x + 1/2 < 0

def q (x y : ℝ) : Prop := (x^2)/4 - (y^2)/12 = 1 ∧ x ≥ 2

def x0_condition (x0 : ℝ) : Prop := sin x0 - cos x0 = sqrt 2

theorem problem (h1 : ∀ x : ℝ, ¬ p x)
               (h2 : ∃ x y : ℝ, q x y)
               (h3 : ∃ x0 : ℝ, x0_condition x0) :
               ∀ x : ℝ, ¬ ¬ p x := 
sorry

end NUMINAMATH_GPT_problem_l305_30526


namespace NUMINAMATH_GPT_eggs_in_basket_empty_l305_30520

theorem eggs_in_basket_empty (a : ℕ) : 
  let remaining_after_first := a - (a / 2 + 1 / 2)
  let remaining_after_second := remaining_after_first - (remaining_after_first / 2 + 1 / 2)
  let remaining_after_third := remaining_after_second - (remaining_after_second / 2 + 1 / 2)
  (remaining_after_first = a / 2 - 1 / 2) → 
  (remaining_after_second = remaining_after_first / 2 - 1 / 2) → 
  (remaining_after_third = remaining_after_second / 2 -1 / 2) → 
  (remaining_after_third = 0) → 
  (a = 7) := sorry

end NUMINAMATH_GPT_eggs_in_basket_empty_l305_30520


namespace NUMINAMATH_GPT_solve_equation_l305_30509

-- Define the conditions
def satisfies_equation (n m : ℕ) : Prop :=
  n > 0 ∧ m > 0 ∧ n^5 + n^4 = 7^m - 1

-- Theorem statement
theorem solve_equation : ∀ n m : ℕ, satisfies_equation n m ↔ (n = 2 ∧ m = 2) := 
by { sorry }

end NUMINAMATH_GPT_solve_equation_l305_30509


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_l305_30519

variable (v_m v_s : ℝ)

theorem speed_of_man_in_still_water :
  (v_m + v_s) * 4 = 48 →
  (v_m - v_s) * 6 = 24 →
  v_m = 8 :=
by
  intros h1 h2
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_l305_30519


namespace NUMINAMATH_GPT_sum_of_altitudes_less_than_sum_of_sides_l305_30530

-- Define a triangle with sides and altitudes properties
structure Triangle :=
(A B C : Point)
(a b c : ℝ)
(m_a m_b m_c : ℝ)
(sides : a + b > c ∧ b + c > a ∧ c + a > b) -- Triangle Inequality

axiom altitude_property (T : Triangle) :
  T.m_a < T.b ∧ T.m_b < T.c ∧ T.m_c < T.a

-- The theorem to prove
theorem sum_of_altitudes_less_than_sum_of_sides (T : Triangle) :
  T.m_a + T.m_b + T.m_c < T.a + T.b + T.c :=
sorry

end NUMINAMATH_GPT_sum_of_altitudes_less_than_sum_of_sides_l305_30530


namespace NUMINAMATH_GPT_length_of_bridge_l305_30564

theorem length_of_bridge
  (length_of_train : ℕ)
  (speed_km_per_hr : ℕ)
  (crossing_time_sec : ℕ)
  (h_train_length : length_of_train = 100)
  (h_speed : speed_km_per_hr = 45)
  (h_time : crossing_time_sec = 30) :
  ∃ (length_of_bridge : ℕ), length_of_bridge = 275 :=
by
  -- Convert speed from km/hr to m/s
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600
  -- Total distance the train travels in crossing_time_sec
  let total_distance := speed_m_per_s * crossing_time_sec
  -- Length of the bridge
  let bridge_length := total_distance - length_of_train
  use bridge_length
  -- Skip the detailed proof steps
  sorry

end NUMINAMATH_GPT_length_of_bridge_l305_30564


namespace NUMINAMATH_GPT_no_three_digits_all_prime_l305_30532

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define a function that forms a three-digit number from digits a, b, c
def form_three_digit (a b c : ℕ) : ℕ :=
100 * a + 10 * b + c

-- Define a function to check if all permutations of three digits form prime numbers
def all_permutations_prime (a b c : ℕ) : Prop :=
is_prime (form_three_digit a b c) ∧
is_prime (form_three_digit a c b) ∧
is_prime (form_three_digit b a c) ∧
is_prime (form_three_digit b c a) ∧
is_prime (form_three_digit c a b) ∧
is_prime (form_three_digit c b a)

-- The main theorem stating that there are no three distinct digits making all permutations prime
theorem no_three_digits_all_prime : ¬∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  all_permutations_prime a b c :=
sorry

end NUMINAMATH_GPT_no_three_digits_all_prime_l305_30532


namespace NUMINAMATH_GPT_number_of_sheets_l305_30556

theorem number_of_sheets
  (n : ℕ)
  (h₁ : 2 * n + 2 = 74) :
  n / 4 = 9 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sheets_l305_30556


namespace NUMINAMATH_GPT_problem_l305_30504

theorem problem (a b : ℚ) (h : a / b = 6 / 5) : (5 * a + 4 * b) / (5 * a - 4 * b) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l305_30504


namespace NUMINAMATH_GPT_money_problem_solution_l305_30572

theorem money_problem_solution (a b : ℝ) (h1 : 7 * a + b < 100) (h2 : 4 * a - b = 40) (h3 : b = 0.5 * a) : 
  a = 80 / 7 ∧ b = 40 / 7 :=
by
  sorry

end NUMINAMATH_GPT_money_problem_solution_l305_30572
