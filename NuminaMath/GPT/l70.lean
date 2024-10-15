import Mathlib

namespace NUMINAMATH_GPT_functionG_has_inverse_l70_7016

noncomputable def functionG : ℝ → ℝ := -- function G described in the problem.
sorry

-- Define the horizontal line test
def horizontal_line_test (f : ℝ → ℝ) : Prop :=
∀ y : ℝ, ∃! x : ℝ, f x = y

theorem functionG_has_inverse : horizontal_line_test functionG :=
sorry

end NUMINAMATH_GPT_functionG_has_inverse_l70_7016


namespace NUMINAMATH_GPT_expand_expression_l70_7094

theorem expand_expression : ∀ (x : ℝ), (17 * x + 21) * 3 * x = 51 * x^2 + 63 * x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_expand_expression_l70_7094


namespace NUMINAMATH_GPT_three_liters_to_gallons_l70_7077

theorem three_liters_to_gallons :
  (0.5 : ℝ) * 3 * 0.1319 = 0.7914 := by
  sorry

end NUMINAMATH_GPT_three_liters_to_gallons_l70_7077


namespace NUMINAMATH_GPT_rate_of_interest_per_annum_l70_7042

theorem rate_of_interest_per_annum (SI P : ℝ) (T : ℕ) (hSI : SI = 4016.25) (hP : P = 10040.625) (hT : T = 5) :
  (SI * 100) / (P * T) = 8 :=
by 
  -- Given simple interest formula
  -- SI = P * R * T / 100, solving for R we get R = (SI * 100) / (P * T)
  -- Substitute SI = 4016.25, P = 10040.625, and T = 5
  -- (4016.25 * 100) / (10040.625 * 5) = 8
  sorry

end NUMINAMATH_GPT_rate_of_interest_per_annum_l70_7042


namespace NUMINAMATH_GPT_infinite_primes_4k1_l70_7012

theorem infinite_primes_4k1 : ∀ (P : List ℕ), (∀ (p : ℕ), p ∈ P → Nat.Prime p ∧ ∃ k, p = 4 * k + 1) → 
  ∃ q, Nat.Prime q ∧ ∃ k, q = 4 * k + 1 ∧ q ∉ P :=
sorry

end NUMINAMATH_GPT_infinite_primes_4k1_l70_7012


namespace NUMINAMATH_GPT_coral_three_night_total_pages_l70_7019

-- Definitions based on conditions in the problem
def night1_pages : ℕ := 30
def night2_pages : ℕ := 2 * night1_pages - 2
def night3_pages : ℕ := night1_pages + night2_pages + 3
def total_pages : ℕ := night1_pages + night2_pages + night3_pages

-- The statement we want to prove
theorem coral_three_night_total_pages : total_pages = 179 := by
  sorry

end NUMINAMATH_GPT_coral_three_night_total_pages_l70_7019


namespace NUMINAMATH_GPT_eleanor_distance_between_meetings_l70_7098

-- Conditions given in the problem
def track_length : ℕ := 720
def eric_time : ℕ := 4
def eleanor_time : ℕ := 5
def eric_speed : ℕ := track_length / eric_time
def eleanor_speed : ℕ := track_length / eleanor_time
def relative_speed : ℕ := eric_speed + eleanor_speed
def time_to_meet : ℚ := track_length / relative_speed

-- Proof task: prove that the distance Eleanor runs between consective meetings is 320 meters.
theorem eleanor_distance_between_meetings : eleanor_speed * time_to_meet = 320 := by
  sorry

end NUMINAMATH_GPT_eleanor_distance_between_meetings_l70_7098


namespace NUMINAMATH_GPT_total_cost_l70_7088

-- Definitions:
def amount_beef : ℕ := 1000
def price_per_pound_beef : ℕ := 8
def amount_chicken := amount_beef * 2
def price_per_pound_chicken : ℕ := 3

-- Theorem: The total cost of beef and chicken is $14000.
theorem total_cost : (amount_beef * price_per_pound_beef) + (amount_chicken * price_per_pound_chicken) = 14000 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_l70_7088


namespace NUMINAMATH_GPT_prod_three_consec_cubemultiple_of_504_l70_7066

theorem prod_three_consec_cubemultiple_of_504 (a : ℤ) : (a^3 - 1) * a^3 * (a^3 + 1) % 504 = 0 := by
  sorry

end NUMINAMATH_GPT_prod_three_consec_cubemultiple_of_504_l70_7066


namespace NUMINAMATH_GPT_initial_kittens_l70_7090

theorem initial_kittens (x : ℕ) (h : x + 3 = 9) : x = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_kittens_l70_7090


namespace NUMINAMATH_GPT_total_crayons_l70_7060

noncomputable def original_crayons : ℝ := 479.0
noncomputable def additional_crayons : ℝ := 134.0

theorem total_crayons : original_crayons + additional_crayons = 613.0 := by
  sorry

end NUMINAMATH_GPT_total_crayons_l70_7060


namespace NUMINAMATH_GPT_parabola_c_value_l70_7062

theorem parabola_c_value (a b c : ℝ) (h1 : 3 = a * (-1)^2 + b * (-1) + c)
  (h2 : 1 = a * (-2)^2 + b * (-2) + c) : c = 1 :=
sorry

end NUMINAMATH_GPT_parabola_c_value_l70_7062


namespace NUMINAMATH_GPT_total_work_completed_in_18_days_l70_7036

theorem total_work_completed_in_18_days :
  let amit_work_rate := 1/10
  let ananthu_work_rate := 1/20
  let amit_days := 2
  let amit_work_done := amit_days * amit_work_rate
  let remaining_work := 1 - amit_work_done
  let ananthu_days := remaining_work / ananthu_work_rate
  amit_days + ananthu_days = 18 := 
by
  sorry

end NUMINAMATH_GPT_total_work_completed_in_18_days_l70_7036


namespace NUMINAMATH_GPT_volume_diff_proof_l70_7085

def volume_difference (x y z x' y' z' : ℝ) : ℝ := x * y * z - x' * y' * z'

theorem volume_diff_proof : 
  (∃ (x y z x' y' z' : ℝ),
    2 * (x + y) = 12 ∧ 2 * (x + z) = 16 ∧ 2 * (y + z) = 24 ∧
    2 * (x' + y') = 12 ∧ 2 * (x' + z') = 16 ∧ 2 * (y' + z') = 20 ∧
    volume_difference x y z x' y' z' = -13) :=
by {
  sorry
}

end NUMINAMATH_GPT_volume_diff_proof_l70_7085


namespace NUMINAMATH_GPT_enrolled_percentage_l70_7091

theorem enrolled_percentage (total_students : ℝ) (non_bio_students : ℝ)
    (h_total : total_students = 880)
    (h_non_bio : non_bio_students = 440.00000000000006) : 
    ((total_students - non_bio_students) / total_students) * 100 = 50 := 
by
  rw [h_total, h_non_bio]
  norm_num
  sorry

end NUMINAMATH_GPT_enrolled_percentage_l70_7091


namespace NUMINAMATH_GPT_identity_1_identity_2_identity_3_l70_7020

-- Variables and assumptions
variables (a b c : ℝ)
variables (h_different : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_pos : a > 0 ∧ b > 0 ∧ c > 0)

-- Part 1
theorem identity_1 : 
  (1 / ((a - b) * (a - c))) + (1 / ((b - c) * (b - a))) + (1 / ((c - a) * (c - b))) = 0 := 
by sorry

-- Part 2
theorem identity_2 :
  (a / ((a - b) * (a - c))) + (b / ((b - c) * (b - a))) + (c / ((c - a) * (c - b))) = 0 :=
by sorry

-- Part 3
theorem identity_3 :
  (a^2 / ((a - b) * (a - c))) + (b^2 / ((b - c) * (b - a))) + (c^2 / ((c - a) * (c - b))) = 1 :=
by sorry

end NUMINAMATH_GPT_identity_1_identity_2_identity_3_l70_7020


namespace NUMINAMATH_GPT_polynomial_coeffs_sum_l70_7084

theorem polynomial_coeffs_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (x : ℝ) :
  (2*x - 3)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 = 10 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_coeffs_sum_l70_7084


namespace NUMINAMATH_GPT_simplify_problem_l70_7026

noncomputable def simplify_expression : ℝ :=
  let numer := (Real.sqrt 3 - 1) ^ (1 - Real.sqrt 2)
  let denom := (Real.sqrt 3 + 1) ^ (1 + Real.sqrt 2)
  numer / denom

theorem simplify_problem :
  simplify_expression = 2 ^ (1 - Real.sqrt 2) * (4 - 2 * Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_simplify_problem_l70_7026


namespace NUMINAMATH_GPT_reciprocal_expression_l70_7095

theorem reciprocal_expression :
  (1 / ((1 / 4 : ℚ) + (1 / 5 : ℚ)) / (1 / 3)) = (20 / 27 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_expression_l70_7095


namespace NUMINAMATH_GPT_cube_property_l70_7097

theorem cube_property (x : ℝ) (s : ℝ) 
  (h1 : s^3 = 8 * x)
  (h2 : 6 * s^2 = 4 * x) :
  x = 5400 :=
by
  sorry

end NUMINAMATH_GPT_cube_property_l70_7097


namespace NUMINAMATH_GPT_find_y_values_l70_7018

open Real

-- Problem statement as a Lean statement.
theorem find_y_values (x : ℝ) (hx : x^2 + 2 * (x / (x - 1)) ^ 2 = 20) :
  ∃ y : ℝ, (y = ((x - 1) ^ 3 * (x + 2)) / (2 * x - 1)) ∧ (y = 14 ∨ y = -56 / 3) := 
sorry

end NUMINAMATH_GPT_find_y_values_l70_7018


namespace NUMINAMATH_GPT_train_passing_pole_l70_7040

variables (v L t_platform D_platform t_pole : ℝ)
variables (H1 : L = 500)
variables (H2 : t_platform = 100)
variables (H3 : D_platform = L + 500)
variables (H4 : t_platform = D_platform / v)

theorem train_passing_pole :
  t_pole = L / v := 
sorry

end NUMINAMATH_GPT_train_passing_pole_l70_7040


namespace NUMINAMATH_GPT_zephyr_island_population_capacity_reach_l70_7035

-- Definitions for conditions
def acres := 30000
def acres_per_person := 2
def initial_year := 2023
def initial_population := 500
def population_growth_rate := 4
def growth_period := 20

-- Maximum population supported by the island
def max_population := acres / acres_per_person

-- Function to calculate population after a given number of years
def population (years : ℕ) : ℕ := initial_population * (population_growth_rate ^ (years / growth_period))

-- The Lean statement to prove that the population will reach or exceed max_capacity in 60 years
theorem zephyr_island_population_capacity_reach : ∃ t : ℕ, t ≤ 60 ∧ population t ≥ max_population :=
by
  sorry

end NUMINAMATH_GPT_zephyr_island_population_capacity_reach_l70_7035


namespace NUMINAMATH_GPT_rod_cut_l70_7017

theorem rod_cut (x : ℕ) (h : 3 * x + 5 * x + 7 * x = 120) : 3 * x = 24 :=
by
  sorry

end NUMINAMATH_GPT_rod_cut_l70_7017


namespace NUMINAMATH_GPT_inequality_proof_l70_7083

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬ (a + d > b + c) := sorry

end NUMINAMATH_GPT_inequality_proof_l70_7083


namespace NUMINAMATH_GPT_trajectory_equation_l70_7002

theorem trajectory_equation (x y a : ℝ) (h : x^2 + y^2 = a^2) :
  (x - y)^2 + 2*x*y = a^2 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_equation_l70_7002


namespace NUMINAMATH_GPT_function_properties_l70_7030

noncomputable def f (x : ℝ) : ℝ := Real.sin (x * Real.cos x)

theorem function_properties :
  (f x = -f (-x)) ∧
  (∀ x, 0 < x ∧ x < Real.pi / 2 → 0 < f x) ∧
  ¬(∃ T, ∀ x, f (x + T) = f x) ∧
  (∀ n : ℤ, f (n * Real.pi) = 0) := 
by
  sorry

end NUMINAMATH_GPT_function_properties_l70_7030


namespace NUMINAMATH_GPT_lift_ratio_l70_7038

theorem lift_ratio (total_weight first_lift second_lift : ℕ) (h1 : total_weight = 1500)
(h2 : first_lift = 600) (h3 : first_lift = 2 * (second_lift - 300)) : first_lift / second_lift = 1 := 
by
  sorry

end NUMINAMATH_GPT_lift_ratio_l70_7038


namespace NUMINAMATH_GPT_three_digit_number_is_473_l70_7004

theorem three_digit_number_is_473 (x y z : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 0 ≤ y) (h4 : y ≤ 9) (h5 : 0 ≤ z) (h6 : z ≤ 9)
  (h7 : 100 * x + 10 * y + z - (100 * z + 10 * y + x) = 99)
  (h8 : x + y + z = 14)
  (h9 : x + z = y) : 100 * x + 10 * y + z = 473 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_is_473_l70_7004


namespace NUMINAMATH_GPT_negq_sufficient_but_not_necessary_for_p_l70_7086

variable (p q : Prop)

theorem negq_sufficient_but_not_necessary_for_p
  (h1 : ¬p → q)
  (h2 : ¬(¬q → p)) :
  (¬q → p) ∧ ¬(p → ¬q) :=
sorry

end NUMINAMATH_GPT_negq_sufficient_but_not_necessary_for_p_l70_7086


namespace NUMINAMATH_GPT_graph_passes_through_0_1_l70_7051

theorem graph_passes_through_0_1 (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (0, 1) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^x) } :=
sorry

end NUMINAMATH_GPT_graph_passes_through_0_1_l70_7051


namespace NUMINAMATH_GPT_factorize_expression_l70_7096

theorem factorize_expression (x : ℝ) : x * (x - 3) - x + 3 = (x - 1) * (x - 3) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l70_7096


namespace NUMINAMATH_GPT_smallest_positive_m_l70_7048

theorem smallest_positive_m (m : ℕ) (h : ∀ (n : ℕ), n % 2 = 1 → (529^n + m * 132^n) % 262417 = 0) : m = 1 :=
sorry

end NUMINAMATH_GPT_smallest_positive_m_l70_7048


namespace NUMINAMATH_GPT_solution_set_eq_2m_add_2_gt_zero_l70_7080

theorem solution_set_eq_2m_add_2_gt_zero {m : ℝ} (h : ∀ x : ℝ, mx + 2 > 0 ↔ x < 2) : m = -1 :=
sorry

end NUMINAMATH_GPT_solution_set_eq_2m_add_2_gt_zero_l70_7080


namespace NUMINAMATH_GPT_divisibility_by_37_l70_7075

theorem divisibility_by_37 (a b c : ℕ) :
  (100 * a + 10 * b + c) % 37 = 0 → 
  (100 * b + 10 * c + a) % 37 = 0 ∧
  (100 * c + 10 * a + b) % 37 = 0 :=
by
  sorry

end NUMINAMATH_GPT_divisibility_by_37_l70_7075


namespace NUMINAMATH_GPT_negate_universal_proposition_l70_7006

open Classical

def P (x : ℝ) : Prop := x^3 - 3*x > 0

theorem negate_universal_proposition :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x :=
by sorry

end NUMINAMATH_GPT_negate_universal_proposition_l70_7006


namespace NUMINAMATH_GPT_line_equation_l70_7071

theorem line_equation :
  ∃ m b, m = 1 ∧ b = 5 ∧ (∀ x y, y = m * x + b ↔ x - y + 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l70_7071


namespace NUMINAMATH_GPT_puzzle_pieces_l70_7027

theorem puzzle_pieces (x : ℝ) (h : x + 2 * 1.5 * x = 4000) : x = 1000 :=
  sorry

end NUMINAMATH_GPT_puzzle_pieces_l70_7027


namespace NUMINAMATH_GPT_unique_m_for_prime_condition_l70_7011

theorem unique_m_for_prime_condition :
  ∃ (m : ℕ), m > 0 ∧ (∀ (p : ℕ), Prime p → (∀ (n : ℕ), ¬ p ∣ (n^m - m))) ↔ m = 1 :=
sorry

end NUMINAMATH_GPT_unique_m_for_prime_condition_l70_7011


namespace NUMINAMATH_GPT_change_is_correct_l70_7043

def regular_ticket_cost : ℕ := 109
def child_discount : ℕ := 5
def payment_given : ℕ := 500

-- Prices for different people in the family
def child_ticket_cost (age : ℕ) : ℕ :=
  if age < 12 then regular_ticket_cost - child_discount else regular_ticket_cost

def parent_ticket_cost : ℕ := regular_ticket_cost
def family_ticket_cost : ℕ :=
  (child_ticket_cost 6) + (child_ticket_cost 10) + parent_ticket_cost + parent_ticket_cost

def change_received : ℕ := payment_given - family_ticket_cost

-- Prove that the change received is 74
theorem change_is_correct : change_received = 74 :=
by sorry

end NUMINAMATH_GPT_change_is_correct_l70_7043


namespace NUMINAMATH_GPT_maggie_total_income_l70_7034

def total_income (h_tractor : ℕ) (r_office r_tractor : ℕ) :=
  let h_office := 2 * h_tractor
  (h_tractor * r_tractor) + (h_office * r_office)

theorem maggie_total_income :
  total_income 13 10 12 = 416 := 
  sorry

end NUMINAMATH_GPT_maggie_total_income_l70_7034


namespace NUMINAMATH_GPT_Sam_and_Tina_distance_l70_7099

theorem Sam_and_Tina_distance (marguerite_distance : ℕ) (marguerite_time : ℕ)
  (sam_time : ℕ) (tina_time : ℕ) (sam_distance : ℕ) (tina_distance : ℕ)
  (h1 : marguerite_distance = 150) (h2 : marguerite_time = 3)
  (h3 : sam_time = 4) (h4 : tina_time = 2)
  (h5 : sam_distance = (marguerite_distance / marguerite_time) * sam_time)
  (h6 : tina_distance = (marguerite_distance / marguerite_time) * tina_time) :
  sam_distance = 200 ∧ tina_distance = 100 :=
by
  sorry

end NUMINAMATH_GPT_Sam_and_Tina_distance_l70_7099


namespace NUMINAMATH_GPT_vector_AB_to_vector_BA_l70_7055

theorem vector_AB_to_vector_BA (z : ℂ) (hz : z = -3 + 2 * Complex.I) : -z = 3 - 2 * Complex.I :=
by
  rw [hz]
  sorry

end NUMINAMATH_GPT_vector_AB_to_vector_BA_l70_7055


namespace NUMINAMATH_GPT_fraction_of_total_l70_7039

def total_amount : ℝ := 5000
def r_amount : ℝ := 2000.0000000000002

theorem fraction_of_total
  (h1 : r_amount = 2000.0000000000002)
  (h2 : total_amount = 5000) :
  r_amount / total_amount = 0.40000000000000004 :=
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_fraction_of_total_l70_7039


namespace NUMINAMATH_GPT_range_of_f_l70_7007

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_f :
  Set.range f = Set.Icc (Real.pi / 2 + Real.arctan (-2)) (Real.pi / 2 + Real.arctan 2) :=
sorry

end NUMINAMATH_GPT_range_of_f_l70_7007


namespace NUMINAMATH_GPT_value_of_expression_l70_7053

variables {a b c : ℝ}

theorem value_of_expression (h1 : a * b * c = 10) (h2 : a + b + c = 15) (h3 : a * b + b * c + c * a = 25) :
  (2 + a) * (2 + b) * (2 + c) = 128 := 
sorry

end NUMINAMATH_GPT_value_of_expression_l70_7053


namespace NUMINAMATH_GPT_Josephine_sold_10_liters_l70_7033

def milk_sold (n1 n2 n3 : ℕ) (v1 v2 v3 : ℝ) : ℝ :=
  (v1 * n1) + (v2 * n2) + (v3 * n3)

theorem Josephine_sold_10_liters :
  milk_sold 3 2 5 2 0.75 0.5 = 10 :=
by
  sorry

end NUMINAMATH_GPT_Josephine_sold_10_liters_l70_7033


namespace NUMINAMATH_GPT_geometric_seq_a7_l70_7067

theorem geometric_seq_a7 (a : ℕ → ℤ) (q : ℤ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 2 * a 4 * a 5 = a 3 * a 6)
  (h2 : a 9 * a 10 = -8) : a 7 = -2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_a7_l70_7067


namespace NUMINAMATH_GPT_range_of_t_l70_7031

variable {f : ℝ → ℝ}

theorem range_of_t (h₁ : ∀ x y : ℝ, x < y → f x ≥ f y) (h₂ : ∀ t : ℝ, f (t^2) < f t) : 
  ∀ t : ℝ, f (t^2) < f t ↔ (t < 0 ∨ t > 1) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_t_l70_7031


namespace NUMINAMATH_GPT_sum_a2012_a2013_l70_7015

-- Define the geometric sequence and its conditions
def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n * q

-- Parameters for the problem
variable (a : ℕ → ℚ)
variable (q : ℚ)
variable (h_seq : geometric_sequence a q)
variable (h_q : 1 < q)
variable (h_eq : ∀ x : ℚ, 4 * x^2 - 8 * x + 3 = 0 → x = a 2010 ∨ x = a 2011)

-- Statement to prove
theorem sum_a2012_a2013 : a 2012 + a 2013 = 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_a2012_a2013_l70_7015


namespace NUMINAMATH_GPT_price_per_gaming_chair_l70_7073

theorem price_per_gaming_chair 
  (P : ℝ)
  (price_per_organizer : ℝ := 78)
  (num_organizers : ℕ := 3)
  (num_chairs : ℕ := 2)
  (total_paid : ℝ := 420)
  (delivery_fee_rate : ℝ := 0.05) 
  (cost_organizers : ℝ := num_organizers * price_per_organizer)
  (cost_gaming_chairs : ℝ := num_chairs * P)
  (total_sales : ℝ := cost_organizers + cost_gaming_chairs)
  (delivery_fee : ℝ := delivery_fee_rate * total_sales) :
  total_paid = total_sales + delivery_fee → P = 83 := 
sorry

end NUMINAMATH_GPT_price_per_gaming_chair_l70_7073


namespace NUMINAMATH_GPT_oxygen_atoms_in_compound_l70_7032

-- Define given conditions as parameters in the problem.
def number_of_oxygen_atoms (molecular_weight : ℕ) (weight_Al : ℕ) (weight_H : ℕ) (weight_O : ℕ) (atoms_Al : ℕ) (atoms_H : ℕ) (weight : ℕ) : ℕ := 
  (weight - (atoms_Al * weight_Al + atoms_H * weight_H)) / weight_O

-- Define the actual problem using the defined conditions.
theorem oxygen_atoms_in_compound
  (molecular_weight : ℕ := 78) 
  (weight_Al : ℕ := 27) 
  (weight_H : ℕ := 1) 
  (weight_O : ℕ := 16) 
  (atoms_Al : ℕ := 1) 
  (atoms_H : ℕ := 3) : 
  number_of_oxygen_atoms molecular_weight weight_Al weight_H weight_O atoms_Al atoms_H molecular_weight = 3 := 
sorry

end NUMINAMATH_GPT_oxygen_atoms_in_compound_l70_7032


namespace NUMINAMATH_GPT_find_a_subtract_two_l70_7045

theorem find_a_subtract_two (a b : ℤ) 
    (h1 : 2 + a = 5 - b) 
    (h2 : 5 + b = 8 + a) : 
    2 - a = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_subtract_two_l70_7045


namespace NUMINAMATH_GPT_factorize_1_factorize_2_l70_7000

-- Proof problem 1: Prove x² - 6x + 9 = (x - 3)²
theorem factorize_1 (x : ℝ) : x^2 - 6 * x + 9 = (x - 3)^2 :=
by sorry

-- Proof problem 2: Prove x²(y - 2) - 4(y - 2) = (y - 2)(x + 2)(x - 2)
theorem factorize_2 (x y : ℝ) : x^2 * (y - 2) - 4 * (y - 2) = (y - 2) * (x + 2) * (x - 2) :=
by sorry

end NUMINAMATH_GPT_factorize_1_factorize_2_l70_7000


namespace NUMINAMATH_GPT_calculator_press_count_l70_7064

theorem calculator_press_count : 
  ∃ n : ℕ, n ≥ 4 ∧ (2 ^ (2 ^ n)) > 500 := 
by
  sorry

end NUMINAMATH_GPT_calculator_press_count_l70_7064


namespace NUMINAMATH_GPT_find_number_l70_7046

-- Define the condition given in the problem
def condition (x : ℕ) : Prop :=
  x / 5 + 6 = 65

-- Prove that the solution satisfies the condition
theorem find_number : ∃ x : ℕ, condition x ∧ x = 295 :=
by
  -- Skip the actual proof steps
  sorry

end NUMINAMATH_GPT_find_number_l70_7046


namespace NUMINAMATH_GPT_ellipse_condition_necessary_not_sufficient_l70_7037

theorem ellipse_condition_necessary_not_sufficient {a b : ℝ} (h : a * b > 0):
  (∀ x y : ℝ, a * x^2 + b * y^2 = 1 → a > 0 ∧ b > 0 ∨ a < 0 ∧ b < 0) ∧ 
  ((a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) → a * b > 0) :=
sorry

end NUMINAMATH_GPT_ellipse_condition_necessary_not_sufficient_l70_7037


namespace NUMINAMATH_GPT_inequality_proof_l70_7089

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  (x / Real.sqrt y + y / Real.sqrt x) ≥ (Real.sqrt x + Real.sqrt y) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l70_7089


namespace NUMINAMATH_GPT_number_of_walls_l70_7052

theorem number_of_walls (bricks_per_row rows_per_wall total_bricks : Nat) :
  bricks_per_row = 30 → 
  rows_per_wall = 50 → 
  total_bricks = 3000 → 
  total_bricks / (bricks_per_row * rows_per_wall) = 2 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_number_of_walls_l70_7052


namespace NUMINAMATH_GPT_percentage_no_job_diploma_l70_7050

def percentage_with_university_diploma {total_population : ℕ} (has_diploma : ℕ) : ℕ :=
  (has_diploma / total_population) * 100

variables {total_population : ℕ} (p_no_diploma_and_job : ℕ) (p_with_job : ℕ) (p_diploma : ℕ)

axiom percentage_no_diploma_job :
  p_no_diploma_and_job = 10

axiom percentage_with_job :
  p_with_job = 40

axiom percentage_diploma :
  p_diploma = 39

theorem percentage_no_job_diploma :
  ∃ p : ℕ, p = (9 / 60) * 100 := sorry

end NUMINAMATH_GPT_percentage_no_job_diploma_l70_7050


namespace NUMINAMATH_GPT_critics_voted_same_actor_actress_l70_7044

theorem critics_voted_same_actor_actress :
  ∃ (critic1 critic2 : ℕ) 
  (actor_vote1 actor_vote2 actress_vote1 actress_vote2 : ℕ),
  1 ≤ critic1 ∧ critic1 ≤ 3366 ∧
  1 ≤ critic2 ∧ critic2 ≤ 3366 ∧
  (critic1 ≠ critic2) ∧
  ∃ (vote_count : Fin 100 → ℕ) 
  (actor actress : Fin 3366 → Fin 100),
  (∀ n : Fin 100, ∃ act : Fin 100, vote_count act = n + 1) ∧
  actor critic1 = actor_vote1 ∧ actress critic1 = actress_vote1 ∧
  actor critic2 = actor_vote2 ∧ actress critic2 = actress_vote2 ∧
  actor_vote1 = actor_vote2 ∧ actress_vote1 = actress_vote2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_critics_voted_same_actor_actress_l70_7044


namespace NUMINAMATH_GPT_mike_total_cost_self_correct_l70_7025

-- Definition of the given conditions
def cost_per_rose_bush : ℕ := 75
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def cost_per_tiger_tooth_aloes : ℕ := 100
def total_tiger_tooth_aloes : ℕ := 2

-- Calculate the total cost for Mike's plants
def total_cost_mike_self: ℕ := 
  (total_rose_bushes - friend_rose_bushes) * cost_per_rose_bush + total_tiger_tooth_aloes * cost_per_tiger_tooth_aloes

-- The main proposition to be proved
theorem mike_total_cost_self_correct : total_cost_mike_self = 500 := by
  sorry

end NUMINAMATH_GPT_mike_total_cost_self_correct_l70_7025


namespace NUMINAMATH_GPT_evaluate_expression_l70_7028

theorem evaluate_expression :
  (∃ (a b c : ℕ), a = 18 ∧ b = 3 ∧ c = 54 ∧ c = a * b ∧ (18^36 / 54^18) = (6^18)) :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l70_7028


namespace NUMINAMATH_GPT_placement_proof_l70_7009

def claimed_first_place (p: String) : Prop := 
  p = "Olya" ∨ p = "Oleg" ∨ p = "Pasha"

def odd_places_boys (positions: ℕ → String) : Prop := 
  (positions 1 = "Oleg" ∨ positions 1 = "Pasha") ∧ (positions 3 = "Oleg" ∨ positions 3 = "Pasha")

def olya_wrong (positions : ℕ → String) : Prop := 
  ¬odd_places_boys positions

def always_truthful_or_lying (Olya_st: Prop) (Oleg_st: Prop) (Pasha_st: Prop) : Prop := 
  Olya_st = Oleg_st ∧ Oleg_st = Pasha_st

def competition_placement : Prop :=
  ∃ (positions: ℕ → String),
    claimed_first_place (positions 1) ∧
    claimed_first_place (positions 2) ∧
    claimed_first_place (positions 3) ∧
    (positions 1 = "Oleg") ∧
    (positions 2 = "Pasha") ∧
    (positions 3 = "Olya") ∧
    olya_wrong positions ∧
    always_truthful_or_lying
      ((claimed_first_place "Olya" ∧ odd_places_boys positions))
      ((claimed_first_place "Oleg" ∧ olya_wrong positions))
      (claimed_first_place "Pasha")

theorem placement_proof : competition_placement :=
  sorry

end NUMINAMATH_GPT_placement_proof_l70_7009


namespace NUMINAMATH_GPT_value_this_year_l70_7059

def last_year_value : ℝ := 20000
def depreciation_factor : ℝ := 0.8

theorem value_this_year :
  last_year_value * depreciation_factor = 16000 :=
by
  sorry

end NUMINAMATH_GPT_value_this_year_l70_7059


namespace NUMINAMATH_GPT_possible_values_of_m_l70_7065

theorem possible_values_of_m (a b : ℤ) (h1 : a * b = -14) :
  ∃ m : ℤ, m = a + b ∧ (m = 5 ∨ m = -5 ∨ m = 13 ∨ m = -13) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_m_l70_7065


namespace NUMINAMATH_GPT_find_value_of_k_l70_7074

def line_equation_holds (m n : ℤ) : Prop := m = 2 * n + 5
def second_point_condition (m n k : ℤ) : Prop := m + 4 = 2 * (n + k) + 5

theorem find_value_of_k (m n k : ℤ) 
  (h1 : line_equation_holds m n) 
  (h2 : second_point_condition m n k) : 
  k = 2 :=
by sorry

end NUMINAMATH_GPT_find_value_of_k_l70_7074


namespace NUMINAMATH_GPT_negation_example_l70_7079

theorem negation_example :
  ¬(∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - x₀ > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l70_7079


namespace NUMINAMATH_GPT_solution_of_modified_system_l70_7056

theorem solution_of_modified_system
  (a b x y : ℝ)
  (h1 : 2*a*3 + 3*4 = 18)
  (h2 : -3 + 5*b*4 = 17)
  : (x + y = 7 ∧ x - y = -1) → (2*a*(x+y) + 3*(x-y) = 18 ∧ (x+y) - 5*b*(x-y) = -17) → (x = (7 / 2) ∧ y = (-1 / 2)) :=
by
sorry

end NUMINAMATH_GPT_solution_of_modified_system_l70_7056


namespace NUMINAMATH_GPT_graph_empty_l70_7003

theorem graph_empty (x y : ℝ) : 
  x^2 + 3 * y^2 - 4 * x - 6 * y + 9 ≠ 0 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_graph_empty_l70_7003


namespace NUMINAMATH_GPT_min_q_of_abs_poly_eq_three_l70_7005

theorem min_q_of_abs_poly_eq_three (p q : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (|x1^2 + p * x1 + q| = 3) ∧ (|x2^2 + p * x2 + q| = 3) ∧ (|x3^2 + p * x3 + q| = 3)) →
  q = -3 :=
sorry

end NUMINAMATH_GPT_min_q_of_abs_poly_eq_three_l70_7005


namespace NUMINAMATH_GPT_determinant_inequality_l70_7092

variable (x : ℝ)

def determinant (a b c d : ℝ) : ℝ :=
  a * d - b * c

theorem determinant_inequality (h : determinant 2 (3 - x) 1 x > 0) : x > 1 := by
  sorry

end NUMINAMATH_GPT_determinant_inequality_l70_7092


namespace NUMINAMATH_GPT_intersection_A_B_l70_7072

variable (A : Set ℤ) (B : Set ℤ)

-- Define the set A and B
def set_A : Set ℤ := {0, 1, 2}
def set_B : Set ℤ := {x | 1 < x ∧ x < 4}

theorem intersection_A_B :
  set_A ∩ set_B = {2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l70_7072


namespace NUMINAMATH_GPT_problem1_problem2_l70_7008

theorem problem1 : (1 * (-5) - (-6) + (-7)) = -6 :=
by
  sorry

theorem problem2 : (-1)^2021 + (-18) * abs (-2 / 9) - 4 / (-2) = -3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l70_7008


namespace NUMINAMATH_GPT_g_54_l70_7010

def g : ℕ → ℤ := sorry

axiom g_multiplicative (x y : ℕ) (hx : x > 0) (hy : y > 0) : g (x * y) = g x + g y
axiom g_6 : g 6 = 10
axiom g_18 : g 18 = 14

theorem g_54 : g 54 = 18 := by
  sorry

end NUMINAMATH_GPT_g_54_l70_7010


namespace NUMINAMATH_GPT_coeff_of_x_square_l70_7021

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem coeff_of_x_square :
  (binom 8 3 = 56) ∧ (8 - 2 * 3 = 2) :=
sorry

end NUMINAMATH_GPT_coeff_of_x_square_l70_7021


namespace NUMINAMATH_GPT_directrix_parabola_l70_7047

-- Given the equation of the parabola and required transformations:
theorem directrix_parabola (d : ℚ) : 
  (∀ x : ℚ, y = -4 * x^2 + 4) → d = 65 / 16 :=
by sorry

end NUMINAMATH_GPT_directrix_parabola_l70_7047


namespace NUMINAMATH_GPT_side_length_correct_l70_7058

noncomputable def find_side_length (b : ℝ) (angleB : ℝ) (sinA : ℝ) : ℝ :=
  let sinB := Real.sin angleB
  let a := b * sinA / sinB
  a

theorem side_length_correct (b : ℝ) (angleB : ℝ) (sinA : ℝ) (a : ℝ) 
  (hb : b = 4)
  (hangleB : angleB = Real.pi / 6)
  (hsinA : sinA = 1 / 3)
  (ha : a = 8 / 3) : 
  find_side_length b angleB sinA = a :=
by
  sorry

end NUMINAMATH_GPT_side_length_correct_l70_7058


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l70_7013

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 8) 
  (h2 : (a + b + c + d) / 4 = 5) : 
  (e + f) / 2 = 14 := 
by  
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l70_7013


namespace NUMINAMATH_GPT_cows_C_grazed_l70_7061

/-- Define the conditions for each milkman’s cow-months. -/
def A_cow_months := 24 * 3
def B_cow_months := 10 * 5
def D_cow_months := 21 * 3
def C_cow_months (x : ℕ) := x * 4

/-- Define the cost per cow-month based on A's share. -/
def cost_per_cow_month := 720 / A_cow_months

/-- Define the total rent. -/
def total_rent := 3250

/-- Define the total cow-months including C's cow-months as a variable. -/
def total_cow_months (x : ℕ) := A_cow_months + B_cow_months + C_cow_months x + D_cow_months

/-- Lean 4 statement to prove the number of cows C grazed. -/
theorem cows_C_grazed (x : ℕ) :
  total_rent = total_cow_months x * cost_per_cow_month → x = 35 := by {
  sorry
}

end NUMINAMATH_GPT_cows_C_grazed_l70_7061


namespace NUMINAMATH_GPT_problem1_problem2_l70_7001

-- Proof of Problem 1
theorem problem1 (x y : ℤ) (h1 : x = -2) (h2 : y = -3) : (6 * x - 5 * y + 3 * y - 2 * x) = -2 :=
by
  sorry

-- Proof of Problem 2
theorem problem2 (a : ℚ) (h : a = -1 / 2) : (1 / 4 * (-4 * a^2 + 2 * a - 8) - (1 / 2 * a - 2)) = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l70_7001


namespace NUMINAMATH_GPT_train_length_l70_7076

-- Definitions based on conditions
def faster_train_speed := 46 -- speed in km/hr
def slower_train_speed := 36 -- speed in km/hr
def time_to_pass := 72 -- time in seconds
def relative_speed_kmph := faster_train_speed - slower_train_speed
def relative_speed_mps : ℚ := (relative_speed_kmph * 1000) / 3600

theorem train_length :
  ∃ L : ℚ, (2 * L = relative_speed_mps * time_to_pass / 1) ∧ L = 100 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l70_7076


namespace NUMINAMATH_GPT_number_of_bonnies_l70_7069

theorem number_of_bonnies (B blueberries apples : ℝ) 
  (h1 : blueberries = 3 / 4 * B) 
  (h2 : apples = 3 * blueberries)
  (h3 : B + blueberries + apples = 240) : 
  B = 60 :=
by
  sorry

end NUMINAMATH_GPT_number_of_bonnies_l70_7069


namespace NUMINAMATH_GPT_probability_of_real_roots_is_correct_l70_7063

open Real

def has_real_roots (m : ℝ) : Prop :=
  2 * m^2 - 8 ≥ 0 

def favorable_set : Set ℝ := {m | has_real_roots m}

def interval_length (a b : ℝ) : ℝ := b - a

noncomputable def probability_of_real_roots : ℝ :=
  interval_length (-4) (-2) + interval_length 2 3 / interval_length (-4) 3

theorem probability_of_real_roots_is_correct : probability_of_real_roots = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_real_roots_is_correct_l70_7063


namespace NUMINAMATH_GPT_expression_eval_l70_7014

theorem expression_eval :
  (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) = 125 :=
by
  sorry

end NUMINAMATH_GPT_expression_eval_l70_7014


namespace NUMINAMATH_GPT_not_sqrt2_rational_union_eq_intersection_implies_equal_intersection_eq_b_subset_a_element_in_both_implies_in_intersection_l70_7070

-- Definitions
def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ (x = a / b)
def union (A B : Set α) : Set α := {x | x ∈ A ∨ x ∈ B}
def intersection (A B : Set α) : Set α := {x | x ∈ A ∧ x ∈ B}
def subset (A B : Set α) : Prop := ∀ x, x ∈ A → x ∈ B

-- Statement A
theorem not_sqrt2_rational : ¬ is_rational (Real.sqrt 2) :=
sorry

-- Statement B
theorem union_eq_intersection_implies_equal {α : Type*} {A B : Set α}
  (h : union A B = intersection A B) : A = B :=
sorry

-- Statement C
theorem intersection_eq_b_subset_a {α : Type*} {A B : Set α}
  (h : intersection A B = B) : subset B A :=
sorry

-- Statement D
theorem element_in_both_implies_in_intersection {α : Type*} {A B : Set α} {a : α}
  (haA : a ∈ A) (haB : a ∈ B) : a ∈ intersection A B :=
sorry

end NUMINAMATH_GPT_not_sqrt2_rational_union_eq_intersection_implies_equal_intersection_eq_b_subset_a_element_in_both_implies_in_intersection_l70_7070


namespace NUMINAMATH_GPT_stay_nights_l70_7029

theorem stay_nights (cost_per_night : ℕ) (num_people : ℕ) (total_cost : ℕ) (n : ℕ) 
    (h1 : cost_per_night = 40) (h2 : num_people = 3) (h3 : total_cost = 360) (h4 : cost_per_night * num_people * n = total_cost) :
    n = 3 :=
sorry

end NUMINAMATH_GPT_stay_nights_l70_7029


namespace NUMINAMATH_GPT_color_of_face_opposite_blue_l70_7054

/-- Assume we have a cube with each face painted in distinct colors. -/
structure Cube where
  top : String
  front : String
  right_side : String
  back : String
  left_side : String
  bottom : String

/-- Given three views of a colored cube, determine the color of the face opposite the blue face. -/
theorem color_of_face_opposite_blue (c : Cube)
  (h_top : c.top = "R")
  (h_right : c.right_side = "G")
  (h_view1 : c.front = "W")
  (h_view2 : c.front = "O")
  (h_view3 : c.front = "Y") :
  c.back = "Y" :=
sorry

end NUMINAMATH_GPT_color_of_face_opposite_blue_l70_7054


namespace NUMINAMATH_GPT_spinner_probabilities_l70_7022

noncomputable def prob_A : ℚ := 1 / 3
noncomputable def prob_B : ℚ := 1 / 4
noncomputable def prob_C : ℚ := 5 / 18
noncomputable def prob_D : ℚ := 5 / 36

theorem spinner_probabilities :
  prob_A + prob_B + prob_C + prob_D = 1 ∧
  prob_C = 2 * prob_D :=
by {
  -- The statement of the theorem matches the given conditions and the correct answers.
  -- Proof will be provided later.
  sorry
}

end NUMINAMATH_GPT_spinner_probabilities_l70_7022


namespace NUMINAMATH_GPT_probability_is_pi_over_12_l70_7024

noncomputable def probability_within_two_units_of_origin : ℝ :=
  let radius := 2
  let circle_area := Real.pi * radius^2
  let rectangle_area := 6 * 8
  circle_area / rectangle_area

theorem probability_is_pi_over_12 :
  probability_within_two_units_of_origin = Real.pi / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_is_pi_over_12_l70_7024


namespace NUMINAMATH_GPT_divisibility_by_2k_l70_7049

-- Define the sequence according to the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ ∀ n, 2 ≤ n → a n = 2 * a (n - 1) + a (n - 2)

-- The theorem to be proved
theorem divisibility_by_2k (a : ℕ → ℤ) (k : ℕ) (n : ℕ)
  (h : seq a) :
  2^k ∣ a n ↔ 2^k ∣ n :=
sorry

end NUMINAMATH_GPT_divisibility_by_2k_l70_7049


namespace NUMINAMATH_GPT_transportation_inverse_proportion_l70_7081

theorem transportation_inverse_proportion (V t : ℝ) (h: V * t = 10^5) : V = 10^5 / t :=
by
  sorry

end NUMINAMATH_GPT_transportation_inverse_proportion_l70_7081


namespace NUMINAMATH_GPT_triangle_distance_bisectors_l70_7087

noncomputable def distance_between_bisectors {a b c : ℝ} (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) : ℝ :=
  (2 * a * b * c) / (b^2 - c^2)

theorem triangle_distance_bisectors 
  (a b c : ℝ) (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) :
  ∀ (DD₁ : ℝ), 
  DD₁ = distance_between_bisectors h₁ h₂ h₃ → 
  DD₁ = (2 * a * b * c) / (b^2 - c^2) := by 
  sorry

end NUMINAMATH_GPT_triangle_distance_bisectors_l70_7087


namespace NUMINAMATH_GPT_toothpick_problem_l70_7078

theorem toothpick_problem : 
  ∃ (N : ℕ), N > 5000 ∧ 
            N % 10 = 9 ∧ 
            N % 9 = 8 ∧ 
            N % 8 = 7 ∧ 
            N % 7 = 6 ∧ 
            N % 6 = 5 ∧ 
            N % 5 = 4 ∧ 
            N = 5039 :=
by
  sorry

end NUMINAMATH_GPT_toothpick_problem_l70_7078


namespace NUMINAMATH_GPT_tutors_meet_in_lab_l70_7057

theorem tutors_meet_in_lab (c a j t : ℕ)
  (hC : c = 5) (hA : a = 6) (hJ : j = 8) (hT : t = 9) :
  Nat.lcm (Nat.lcm (Nat.lcm c a) j) t = 360 :=
by
  rw [hC, hA, hJ, hT]
  rfl

end NUMINAMATH_GPT_tutors_meet_in_lab_l70_7057


namespace NUMINAMATH_GPT_two_digit_product_GCD_l70_7023

-- We define the condition for two-digit integer numbers
def two_digit_num (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Lean statement capturing the conditions
theorem two_digit_product_GCD :
  ∃ (a b : ℕ), two_digit_num a ∧ two_digit_num b ∧ a * b = 1728 ∧ Nat.gcd a b = 12 := 
by {
  sorry -- The proof steps would go here
}

end NUMINAMATH_GPT_two_digit_product_GCD_l70_7023


namespace NUMINAMATH_GPT_sufficient_condition_for_inequality_l70_7082

theorem sufficient_condition_for_inequality (m : ℝ) (h : m ≠ 0) : (m > 2) → (m + 4 / m > 4) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_inequality_l70_7082


namespace NUMINAMATH_GPT_sum_of_coordinates_of_other_endpoint_of_segment_l70_7093

theorem sum_of_coordinates_of_other_endpoint_of_segment {x y : ℝ}
  (h1 : (6 + x) / 2 = 3)
  (h2 : (1 + y) / 2 = 7) :
  x + y = 13 := by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_other_endpoint_of_segment_l70_7093


namespace NUMINAMATH_GPT_fraction_is_5_div_9_l70_7041

-- Define the conditions t = f * (k - 32), t = 35, and k = 95
theorem fraction_is_5_div_9 {f k t : ℚ} (h1 : t = f * (k - 32)) (h2 : t = 35) (h3 : k = 95) : f = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_is_5_div_9_l70_7041


namespace NUMINAMATH_GPT_segment_length_C_C_l70_7068

-- Define the points C and C''.
def C : ℝ × ℝ := (-3, 2)
def C'' : ℝ × ℝ := (-3, -2)

-- State the theorem that the length of the segment from C to C'' is 4.
theorem segment_length_C_C'' : dist C C'' = 4 := by
  sorry

end NUMINAMATH_GPT_segment_length_C_C_l70_7068
