import Mathlib

namespace NUMINAMATH_GPT_other_acute_angle_in_right_triangle_l1330_133010

theorem other_acute_angle_in_right_triangle (a : ℝ) (h : a = 25) :
    ∃ b : ℝ, b = 65 :=
by
  sorry

end NUMINAMATH_GPT_other_acute_angle_in_right_triangle_l1330_133010


namespace NUMINAMATH_GPT_avg_expenditure_Feb_to_July_l1330_133086

noncomputable def avg_expenditure_Jan_to_Jun : ℝ := 4200
noncomputable def expenditure_January : ℝ := 1200
noncomputable def expenditure_July : ℝ := 1500
noncomputable def total_months_Jan_to_Jun : ℝ := 6
noncomputable def total_months_Feb_to_July : ℝ := 6

theorem avg_expenditure_Feb_to_July :
  (avg_expenditure_Jan_to_Jun * total_months_Jan_to_Jun - expenditure_January + expenditure_July) / total_months_Feb_to_July = 4250 :=
by sorry

end NUMINAMATH_GPT_avg_expenditure_Feb_to_July_l1330_133086


namespace NUMINAMATH_GPT_mixture_ratio_l1330_133030

theorem mixture_ratio (V : ℝ) (a b c : ℕ)
  (h_pos : V > 0)
  (h_ratio : V = (3/8) * V + (5/11) * V + ((88 - 33 - 40)/88) * V) :
  a = 33 ∧ b = 40 ∧ c = 15 :=
by
  sorry

end NUMINAMATH_GPT_mixture_ratio_l1330_133030


namespace NUMINAMATH_GPT_right_triangle_proportion_l1330_133032

/-- Given a right triangle ABC with ∠C = 90°, AB = c, AC = b, and BC = a, 
    and a point P on the hypotenuse AB (or its extension) such that 
    AP = m, BP = n, and CP = k, prove that a²m² + b²n² = c²k². -/
theorem right_triangle_proportion
  {a b c m n k : ℝ}
  (h_right : ∀ A B C : ℝ, A^2 + B^2 = C^2)
  (h1 : ∀ P : ℝ, m^2 + n^2 = k^2)
  (h_geometry : a^2 + b^2 = c^2) :
  a^2 * m^2 + b^2 * n^2 = c^2 * k^2 := 
sorry

end NUMINAMATH_GPT_right_triangle_proportion_l1330_133032


namespace NUMINAMATH_GPT_prove_union_sets_l1330_133006

universe u

variable {α : Type u}
variable {M N : Set ℕ}
variable (a b : ℕ)

theorem prove_union_sets (h1 : M = {3, 4^a}) (h2 : N = {a, b}) (h3 : M ∩ N = {1}) : M ∪ N = {0, 1, 3} := sorry

end NUMINAMATH_GPT_prove_union_sets_l1330_133006


namespace NUMINAMATH_GPT_hexagon_inequality_l1330_133017

noncomputable def ABCDEF := 3 * Real.sqrt 3 / 2
noncomputable def ACE := Real.sqrt 3
noncomputable def BDF := Real.sqrt 3
noncomputable def R₁ := Real.sqrt 3 / 4
noncomputable def R₂ := -Real.sqrt 3 / 4

theorem hexagon_inequality :
  min ACE BDF + R₂ - R₁ ≤ 3 * Real.sqrt 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_inequality_l1330_133017


namespace NUMINAMATH_GPT_weight_of_original_piece_of_marble_l1330_133085

theorem weight_of_original_piece_of_marble (W : ℝ) 
  (h1 : W > 0)
  (h2 : (0.75 * 0.56 * W) = 105) : 
  W = 250 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_original_piece_of_marble_l1330_133085


namespace NUMINAMATH_GPT_cross_product_u_v_l1330_133007

-- Define the vectors u and v
def u : ℝ × ℝ × ℝ := (3, -4, 7)
def v : ℝ × ℝ × ℝ := (2, 5, -3)

-- Define the cross product function
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

-- State the theorem to be proved
theorem cross_product_u_v : cross_product u v = (-23, 23, 23) :=
  sorry

end NUMINAMATH_GPT_cross_product_u_v_l1330_133007


namespace NUMINAMATH_GPT_race_distance_l1330_133047

theorem race_distance (dA dB dC : ℝ) (h1 : dA = 1000) (h2 : dB = 900) (h3 : dB = 800) (h4 : dC = 700) (d : ℝ) (h5 : d = dA + 127.5) :
  d = 600 :=
sorry

end NUMINAMATH_GPT_race_distance_l1330_133047


namespace NUMINAMATH_GPT_complement_union_l1330_133070

open Set

theorem complement_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 5, 6, 8})
  (hA : A = {1, 5, 8})(hB : B = {2}) :
  (U \ A) ∪ B = {0, 2, 3, 6} :=
by
  rw [hU, hA, hB]
  -- Intermediate steps would go here
  sorry

end NUMINAMATH_GPT_complement_union_l1330_133070


namespace NUMINAMATH_GPT_committee_probability_l1330_133094

def num_boys : ℕ := 10
def num_girls : ℕ := 15
def num_total : ℕ := 25
def committee_size : ℕ := 5

def num_ways_total : ℕ := Nat.choose num_total committee_size
def num_ways_boys_only : ℕ := Nat.choose num_boys committee_size
def num_ways_girls_only : ℕ := Nat.choose num_girls committee_size

def probability_boys_or_girls_only : ℚ :=
  (num_ways_boys_only + num_ways_girls_only) / num_ways_total

def probability_at_least_one_boy_and_one_girl : ℚ :=
  1 - probability_boys_or_girls_only

theorem committee_probability :
  probability_at_least_one_boy_and_one_girl = 475 / 506 :=
sorry

end NUMINAMATH_GPT_committee_probability_l1330_133094


namespace NUMINAMATH_GPT_appropriate_sampling_methods_l1330_133016

-- Conditions for the first survey
structure Population1 where
  high_income_families : Nat
  middle_income_families : Nat
  low_income_families : Nat
  total : Nat := high_income_families + middle_income_families + low_income_families

def survey1_population : Population1 :=
  { high_income_families := 125,
    middle_income_families := 200,
    low_income_families := 95
  }

-- Condition for the second survey
structure Population2 where
  art_specialized_students : Nat

def survey2_population : Population2 :=
  { art_specialized_students := 5 }

-- The main statement to prove
theorem appropriate_sampling_methods :
  (survey1_population.total >= 100 → stratified_sampling_for_survey1) ∧ 
  (survey2_population.art_specialized_students >= 3 → simple_random_sampling_for_survey2) :=
  sorry

end NUMINAMATH_GPT_appropriate_sampling_methods_l1330_133016


namespace NUMINAMATH_GPT_vector_combination_l1330_133035

-- Definitions for vectors a, b, and c with the conditions provided
def a : ℝ × ℝ × ℝ := (-1, 3, 2)
def b : ℝ × ℝ × ℝ := (4, -6, 2)
def c (t : ℝ) : ℝ × ℝ × ℝ := (-3, 12, t)

-- The statement we want to prove
theorem vector_combination (t m n : ℝ)
  (h : c t = m • a + n • b) :
  t = 11 ∧ m + n = 11 / 2 :=
by
  sorry

end NUMINAMATH_GPT_vector_combination_l1330_133035


namespace NUMINAMATH_GPT_valid_a_value_l1330_133092

theorem valid_a_value (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ a = 20 :=
by
  sorry

end NUMINAMATH_GPT_valid_a_value_l1330_133092


namespace NUMINAMATH_GPT_max_value_ahn_operation_l1330_133013

theorem max_value_ahn_operation :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (300 - n)^2 - 10 = 39990 :=
by
  sorry

end NUMINAMATH_GPT_max_value_ahn_operation_l1330_133013


namespace NUMINAMATH_GPT_quadratic_expression_value_l1330_133099

theorem quadratic_expression_value
  (x : ℝ)
  (h : x^2 + x - 2 = 0)
: x^3 + 2*x^2 - x + 2021 = 2023 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_value_l1330_133099


namespace NUMINAMATH_GPT_find_number_l1330_133039

theorem find_number (x : ℝ) (h : 0.40 * x - 11 = 23) : x = 85 :=
sorry

end NUMINAMATH_GPT_find_number_l1330_133039


namespace NUMINAMATH_GPT_find_plaintext_from_ciphertext_l1330_133082

theorem find_plaintext_from_ciphertext : 
  ∃ x : ℕ, ∀ a : ℝ, (a^3 - 2 = 6) → (1022 = a^x - 2) → x = 10 :=
by
  use 10
  intros a ha hc
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_plaintext_from_ciphertext_l1330_133082


namespace NUMINAMATH_GPT_profit_percentage_l1330_133049

theorem profit_percentage (SP CP : ℝ) (H_SP : SP = 1800) (H_CP : CP = 1500) :
  ((SP - CP) / CP) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_profit_percentage_l1330_133049


namespace NUMINAMATH_GPT_remainder_of_sum_mod_l1330_133060

theorem remainder_of_sum_mod (n : ℤ) : ((7 + n) + (n + 5)) % 7 = (5 + 2 * n) % 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_mod_l1330_133060


namespace NUMINAMATH_GPT_factorization_quad_l1330_133067

theorem factorization_quad (c d : ℕ) (h_factor : (x^2 - 18 * x + 77 = (x - c) * (x - d)))
  (h_nonneg : c ≥ 0 ∧ d ≥ 0) (h_lt : c > d) : 4 * d - c = 17 := by
  sorry

end NUMINAMATH_GPT_factorization_quad_l1330_133067


namespace NUMINAMATH_GPT_other_religion_students_l1330_133002

theorem other_religion_students (total_students : ℕ) 
  (muslims_percent hindus_percent sikhs_percent christians_percent buddhists_percent : ℝ) 
  (h1 : total_students = 1200) 
  (h2 : muslims_percent = 0.35) 
  (h3 : hindus_percent = 0.25) 
  (h4 : sikhs_percent = 0.15) 
  (h5 : christians_percent = 0.10) 
  (h6 : buddhists_percent = 0.05) : 
  ∃ other_religion_students : ℕ, other_religion_students = 120 :=
by
  sorry

end NUMINAMATH_GPT_other_religion_students_l1330_133002


namespace NUMINAMATH_GPT_prob_B_at_most_2_shots_prob_B_exactly_2_more_than_A_l1330_133065

-- Definitions of probabilities of making a shot
def p_A : ℚ := 1 / 3
def p_B : ℚ := 1 / 2

-- Number of attempts
def num_attempts : ℕ := 3

-- Probability that B makes at most 2 shots
theorem prob_B_at_most_2_shots : 
  (1 - (num_attempts.choose 3) * (p_B ^ 3) * ((1 - p_B) ^ (num_attempts - 3))) = 7 / 8 :=
by 
  sorry

-- Probability that B makes exactly 2 more shots than A
theorem prob_B_exactly_2_more_than_A : 
  (num_attempts.choose 2) * (p_B ^ 2) * ((1 - p_B) ^ 1) * (num_attempts.choose 0) * ((1 - p_A) ^ num_attempts) +
  (num_attempts.choose 3) * (p_B ^ 3) * (num_attempts.choose 1) * (p_A ^ 1) * ((1 - p_A) ^ (num_attempts - 1)) = 1 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_prob_B_at_most_2_shots_prob_B_exactly_2_more_than_A_l1330_133065


namespace NUMINAMATH_GPT_simplify_expression_l1330_133071

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1330_133071


namespace NUMINAMATH_GPT_camera_guarantee_l1330_133081

def battery_trials (b : Fin 22 → Bool) : Prop :=
  let charged := Finset.filter (λ i => b i) (Finset.univ : Finset (Fin 22))
  -- Ensuring there are exactly 15 charged batteries
  (charged.card = 15) ∧
  -- The camera works if any set of three batteries are charged
  (∀ (trials : Finset (Finset (Fin 22))),
   trials.card = 10 →
   ∃ t ∈ trials, (t.card = 3 ∧ t ⊆ charged))

theorem camera_guarantee :
  ∃ (b : Fin 22 → Bool), battery_trials b := by
  sorry

end NUMINAMATH_GPT_camera_guarantee_l1330_133081


namespace NUMINAMATH_GPT_melanie_food_total_weight_l1330_133077

def total_weight (brie_oz : ℕ) (bread_lb : ℕ) (tomatoes_lb : ℕ) (zucchini_lb : ℕ) 
           (chicken_lb : ℕ) (raspberries_oz : ℕ) (blueberries_oz : ℕ) : ℕ :=
  let brie_lb := brie_oz / 16
  let raspberries_lb := raspberries_oz / 16
  let blueberries_lb := blueberries_oz / 16
  brie_lb + raspberries_lb + blueberries_lb + bread_lb + tomatoes_lb + zucchini_lb + chicken_lb

theorem melanie_food_total_weight : total_weight 8 1 1 2 (3 / 2) 8 8 = 7 :=
by
  -- result placeholder
  sorry

end NUMINAMATH_GPT_melanie_food_total_weight_l1330_133077


namespace NUMINAMATH_GPT_question_1_question_2_l1330_133055

-- Condition: The coordinates of point P are given by the equations x = -3a - 4, y = 2 + a

-- Question 1: Prove coordinates when P lies on the x-axis
theorem question_1 (a : ℝ) (x : ℝ) (y : ℝ) (h1 : x = -3 * a - 4) (h2 : y = 2 + a) (hy0 : y = 0) :
  a = -2 ∧ x = 2 ∧ y = 0 :=
sorry

-- Question 2: Prove coordinates when PQ is parallel to the y-axis
theorem question_2 (a : ℝ) (x : ℝ) (y : ℝ) (h1 : x = -3 * a - 4) (h2 : y = 2 + a) (hx5 : x = 5) :
  a = -3 ∧ x = 5 ∧ y = -1 :=
sorry

end NUMINAMATH_GPT_question_1_question_2_l1330_133055


namespace NUMINAMATH_GPT_xiao_ying_correct_answers_at_least_l1330_133052

def total_questions : ℕ := 20
def points_correct : ℕ := 5
def points_incorrect : ℕ := 2
def excellent_points : ℕ := 80

theorem xiao_ying_correct_answers_at_least (x : ℕ) :
  (5 * x - 2 * (total_questions - x)) ≥ excellent_points → x ≥ 18 := by
  sorry

end NUMINAMATH_GPT_xiao_ying_correct_answers_at_least_l1330_133052


namespace NUMINAMATH_GPT_find_fraction_l1330_133059

theorem find_fraction 
  (f : ℚ) (t k : ℚ)
  (h1 : t = f * (k - 32)) 
  (h2 : t = 75)
  (h3 : k = 167) : 
  f = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l1330_133059


namespace NUMINAMATH_GPT_system_of_equations_proof_l1330_133057

theorem system_of_equations_proof (a b x A B C : ℝ) (h1: a ≠ 0) 
  (h2: a * Real.sin x + b * Real.cos x = 0) 
  (h3: A * Real.sin (2 * x) + B * Real.cos (2 * x) = C) : 
  2 * a * b * A + (b ^ 2 - a ^ 2) * B + (a ^ 2 + b ^ 2) * C = 0 := 
sorry

end NUMINAMATH_GPT_system_of_equations_proof_l1330_133057


namespace NUMINAMATH_GPT_moles_of_Na2SO4_formed_l1330_133088

/-- 
Given the following conditions:
1. 1 mole of H2SO4 reacts with 2 moles of NaOH.
2. In the presence of 0.5 moles of HCl and 0.5 moles of KOH.
3. At a temperature of 25°C and a pressure of 1 atm.
Prove that the moles of Na2SO4 formed is 1 mole.
-/

theorem moles_of_Na2SO4_formed
  (H2SO4 : ℝ) -- moles of H2SO4
  (NaOH : ℝ) -- moles of NaOH
  (HCl : ℝ) -- moles of HCl
  (KOH : ℝ) -- moles of KOH
  (T : ℝ) -- temperature in °C
  (P : ℝ) -- pressure in atm
  : H2SO4 = 1 ∧ NaOH = 2 ∧ HCl = 0.5 ∧ KOH = 0.5 ∧ T = 25 ∧ P = 1 → 
  ∃ Na2SO4 : ℝ, Na2SO4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_Na2SO4_formed_l1330_133088


namespace NUMINAMATH_GPT_probability_10_or_9_probability_at_least_7_l1330_133044

-- Define the probabilities of hitting each ring
def p_10 : ℝ := 0.1
def p_9 : ℝ := 0.2
def p_8 : ℝ := 0.3
def p_7 : ℝ := 0.3
def p_below_7 : ℝ := 0.1

-- Define the events as their corresponding probabilities
def P_A : ℝ := p_10 -- Event of hitting the 10 ring
def P_B : ℝ := p_9 -- Event of hitting the 9 ring
def P_C : ℝ := p_8 -- Event of hitting the 8 ring
def P_D : ℝ := p_7 -- Event of hitting the 7 ring
def P_E : ℝ := p_below_7 -- Event of hitting below the 7 ring

-- Since the probabilities must sum to 1, we have the following fact about their sum
-- P_A + P_B + P_C + P_D + P_E = 1

theorem probability_10_or_9 : P_A + P_B = 0.3 :=
by 
  -- This would be filled in with the proof steps or assumptions
  sorry

theorem probability_at_least_7 : P_A + P_B + P_C + P_D = 0.9 :=
by 
  -- This would be filled in with the proof steps or assumptions
  sorry

end NUMINAMATH_GPT_probability_10_or_9_probability_at_least_7_l1330_133044


namespace NUMINAMATH_GPT_max_and_min_sum_of_vars_l1330_133038

theorem max_and_min_sum_of_vars (x y z w : ℝ) (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ 0 ≤ w)
  (eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17 / 2) :
  ∃ max min : ℝ, max = 3 ∧ min = -2 + 5 / 2 * Real.sqrt 2 ∧
  (∀ (S : ℝ), S = x + y + z + w → S ≤ max ∧ S ≥ min) :=
by sorry

end NUMINAMATH_GPT_max_and_min_sum_of_vars_l1330_133038


namespace NUMINAMATH_GPT_c_seq_formula_l1330_133026

def x_seq (n : ℕ) : ℕ := 2 * n - 1
def y_seq (n : ℕ) : ℕ := n ^ 2
def c_seq (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem c_seq_formula (n : ℕ) : ∀ k, (c_seq k) = (2 * k - 1) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_c_seq_formula_l1330_133026


namespace NUMINAMATH_GPT_three_digit_multiples_of_7_l1330_133093

theorem three_digit_multiples_of_7 :
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  n = 128 :=
by
  let a := 7 * Nat.ceil (100 / 7)
  let l := 7 * Nat.floor (999 / 7)
  let d := 7
  let n := (l - a) / d + 1
  have : a = 105 := sorry
  have : l = 994 := sorry
  have : n = (994 - 105) / 7 + 1 := sorry
  have : n = 128 := sorry
  exact this

end NUMINAMATH_GPT_three_digit_multiples_of_7_l1330_133093


namespace NUMINAMATH_GPT_sum_two_consecutive_sum_three_consecutive_sum_five_consecutive_sum_six_consecutive_l1330_133000

theorem sum_two_consecutive : ∃ x : ℕ, 75 = x + (x + 1) := by
  sorry

theorem sum_three_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) := by
  sorry

theorem sum_five_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) := by
  sorry

theorem sum_six_consecutive : ∃ x : ℕ, 75 = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) := by
  sorry

end NUMINAMATH_GPT_sum_two_consecutive_sum_three_consecutive_sum_five_consecutive_sum_six_consecutive_l1330_133000


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l1330_133054

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (sum_angles : A + B + C = 180) : 
  (¬ (A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60)) = (A > 60 ∧ B > 60 ∧ C > 60) :=
by sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l1330_133054


namespace NUMINAMATH_GPT_geom_seq_sum_l1330_133024

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geometric : ∀ n, a (n + 1) = a n * r)
variable (h_pos : ∀ n, a n > 0)
variable (h_equation : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

theorem geom_seq_sum : a 3 + a 5 = 5 :=
by sorry

end NUMINAMATH_GPT_geom_seq_sum_l1330_133024


namespace NUMINAMATH_GPT_p1a_p1b_l1330_133015

theorem p1a (m : ℕ) (hm : m > 1) : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = m^3 := by
  sorry  -- Proof is omitted

theorem p1b : ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^6 = y^2 + 127 ∧ x = 4 ∧ y = 63 := by
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_p1a_p1b_l1330_133015


namespace NUMINAMATH_GPT_find_unique_function_l1330_133079

theorem find_unique_function (f : ℝ → ℝ) (hf1 : ∀ x, 0 ≤ x → 0 ≤ f x)
    (hf2 : ∀ x, 0 ≤ x → f (f x) + f x = 12 * x) :
    ∀ x, 0 ≤ x → f x = 3 * x := 
  sorry

end NUMINAMATH_GPT_find_unique_function_l1330_133079


namespace NUMINAMATH_GPT_intercept_sum_l1330_133029

theorem intercept_sum (x y : ℝ) :
  (y - 3 = 6 * (x - 5)) →
  (∃ x_intercept, (y = 0) ∧ (x_intercept = 4.5)) →
  (∃ y_intercept, (x = 0) ∧ (y_intercept = -27)) →
  (4.5 + (-27) = -22.5) :=
by
  intros h_eq h_xint h_yint
  sorry

end NUMINAMATH_GPT_intercept_sum_l1330_133029


namespace NUMINAMATH_GPT_surface_area_geometric_mean_volume_geometric_mean_l1330_133022

noncomputable def surfaces_areas_proof (r : ℝ) (π : ℝ) : Prop :=
  let F_1 := 6 * π * r^2
  let F_2 := 4 * π * r^2
  let F_3 := 9 * π * r^2
  F_1^2 = F_2 * F_3

noncomputable def volumes_proof (r : ℝ) (π : ℝ) : Prop :=
  let V_1 := 2 * π * r^3
  let V_2 := (4 / 3) * π * r^3
  let V_3 := π * r^3
  V_1^2 = V_2 * V_3

theorem surface_area_geometric_mean (r : ℝ) (π : ℝ) : surfaces_areas_proof r π := 
  sorry

theorem volume_geometric_mean (r : ℝ) (π : ℝ) : volumes_proof r π :=
  sorry

end NUMINAMATH_GPT_surface_area_geometric_mean_volume_geometric_mean_l1330_133022


namespace NUMINAMATH_GPT_initial_books_l1330_133037

theorem initial_books (total_books_now : ℕ) (books_added : ℕ) (initial_books : ℕ) :
  total_books_now = 48 → books_added = 10 → initial_books = total_books_now - books_added → initial_books = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_initial_books_l1330_133037


namespace NUMINAMATH_GPT_tom_needs_more_blue_tickets_l1330_133090

def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10
def yellow_to_blue : ℕ := yellow_to_red * red_to_blue
def required_yellow_tickets : ℕ := 10
def required_blue_tickets : ℕ := required_yellow_tickets * yellow_to_blue

def toms_yellow_tickets : ℕ := 8
def toms_red_tickets : ℕ := 3
def toms_blue_tickets : ℕ := 7
def toms_total_blue_tickets : ℕ := 
  (toms_yellow_tickets * yellow_to_blue) + 
  (toms_red_tickets * red_to_blue) + 
  toms_blue_tickets

def additional_blue_tickets_needed : ℕ :=
  required_blue_tickets - toms_total_blue_tickets

theorem tom_needs_more_blue_tickets : additional_blue_tickets_needed = 163 := 
by sorry

end NUMINAMATH_GPT_tom_needs_more_blue_tickets_l1330_133090


namespace NUMINAMATH_GPT_initial_books_donations_l1330_133069

variable {X : ℕ} -- Initial number of book donations

def books_donated_during_week := 10 * 5
def books_borrowed := 140
def books_remaining := 210

theorem initial_books_donations :
  X + books_donated_during_week - books_borrowed = books_remaining → X = 300 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_books_donations_l1330_133069


namespace NUMINAMATH_GPT_modular_inverse_of_35_mod_36_l1330_133075

theorem modular_inverse_of_35_mod_36 : 
  ∃ a : ℤ, (35 * a) % 36 = 1 % 36 ∧ a = 35 := 
by 
  sorry

end NUMINAMATH_GPT_modular_inverse_of_35_mod_36_l1330_133075


namespace NUMINAMATH_GPT_little_sister_stole_roses_l1330_133043

/-- Ricky has 40 roses. His little sister steals some roses. He wants to give away the rest of the roses in equal portions to 9 different people, and each person gets 4 roses. Prove how many roses his little sister stole. -/
theorem little_sister_stole_roses (total_roses stolen_roses remaining_roses people roses_per_person : ℕ)
  (h1 : total_roses = 40)
  (h2 : people = 9)
  (h3 : roses_per_person = 4)
  (h4 : remaining_roses = people * roses_per_person)
  (h5 : remaining_roses = total_roses - stolen_roses) :
  stolen_roses = 4 :=
by
  sorry

end NUMINAMATH_GPT_little_sister_stole_roses_l1330_133043


namespace NUMINAMATH_GPT_discount_percentage_l1330_133048

variable (P : ℝ) -- Original price of the dress
variable (D : ℝ) -- Discount percentage

theorem discount_percentage
  (h1 : P * (1 - D / 100) = 68)
  (h2 : 68 * 1.25 = 85)
  (h3 : 85 - P = 5) :
  D = 15 :=
by
  sorry

end NUMINAMATH_GPT_discount_percentage_l1330_133048


namespace NUMINAMATH_GPT_find_x_add_inv_l1330_133066

theorem find_x_add_inv (x : ℝ) (h : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 :=
sorry

end NUMINAMATH_GPT_find_x_add_inv_l1330_133066


namespace NUMINAMATH_GPT_cubic_polynomial_root_l1330_133072

theorem cubic_polynomial_root (a b c : ℕ) (h : 27 * x^3 - 9 * x^2 - 9 * x - 3 = 0) : 
  (a + b + c = 11) :=
sorry

end NUMINAMATH_GPT_cubic_polynomial_root_l1330_133072


namespace NUMINAMATH_GPT_custom_op_4_2_l1330_133025

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem to prove the result
theorem custom_op_4_2 : custom_op 4 2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_custom_op_4_2_l1330_133025


namespace NUMINAMATH_GPT_eggs_in_each_basket_is_four_l1330_133018

theorem eggs_in_each_basket_is_four 
  (n : ℕ)
  (h1 : n ∣ 16) 
  (h2 : n ∣ 28) 
  (h3 : n ≥ 2) : 
  n = 4 :=
sorry

end NUMINAMATH_GPT_eggs_in_each_basket_is_four_l1330_133018


namespace NUMINAMATH_GPT_x_coordinate_of_point_l1330_133028

theorem x_coordinate_of_point (x_1 n : ℝ) 
  (h1 : x_1 = (n / 5) - (2 / 5)) 
  (h2 : x_1 + 3 = ((n + 15) / 5) - (2 / 5)) : 
  x_1 = (n / 5) - (2 / 5) :=
by sorry

end NUMINAMATH_GPT_x_coordinate_of_point_l1330_133028


namespace NUMINAMATH_GPT_math_problem_l1330_133027

theorem math_problem (x : ℕ) (h : (2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x = 512)) : (x + 2) * (x - 2) = 32 :=
sorry

end NUMINAMATH_GPT_math_problem_l1330_133027


namespace NUMINAMATH_GPT_find_m_l1330_133003

noncomputable def f (x : ℝ) : ℝ := 2^x - 5

theorem find_m (m : ℝ) (h : f m = 3) : m = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l1330_133003


namespace NUMINAMATH_GPT_problems_per_page_is_five_l1330_133034

-- Let M and R be the number of problems on each math and reading page respectively
variables (M R : ℕ)

-- Conditions given in problem
def two_math_pages := 2 * M
def four_reading_pages := 4 * R
def total_problems := two_math_pages + four_reading_pages

-- Assume the number of problems per page is the same for both math and reading as P
variable (P : ℕ)
def problems_per_page_equal := (2 * P) + (4 * P) = 30

theorem problems_per_page_is_five :
  (2 * P) + (4 * P) = 30 → P = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problems_per_page_is_five_l1330_133034


namespace NUMINAMATH_GPT_max_volume_tetrahedron_l1330_133046

-- Definitions and conditions
def SA : ℝ := 4
def AB : ℝ := 5
def SB_min : ℝ := 7
def SC_min : ℝ := 9
def BC_max : ℝ := 6
def AC_max : ℝ := 8

-- Proof statement
theorem max_volume_tetrahedron {SB SC BC AC : ℝ} (hSB : SB ≥ SB_min) (hSC : SC ≥ SC_min) (hBC : BC ≤ BC_max) (hAC : AC ≤ AC_max) :
  ∃ V : ℝ, V = 8 * Real.sqrt 6 ∧ V ≤ (1/3) * (1/2) * SA * AB * (2 * Real.sqrt 6) * BC := by
  sorry

end NUMINAMATH_GPT_max_volume_tetrahedron_l1330_133046


namespace NUMINAMATH_GPT_find_a2_an_le_2an_next_sum_bounds_l1330_133014

variable {a : ℕ → ℝ}
variable (S : ℕ → ℝ)

-- Given conditions
axiom seq_condition (n : ℕ) (h_pos : a n > 0) : 
  a n ^ 2 + a n = 3 * (a (n + 1)) ^ 2 + 2 * a (n + 1)
axiom a1_condition : a 1 = 1

-- Question 1: Prove the value of a2
theorem find_a2 : a 2 = (Real.sqrt 7 - 1) / 3 :=
  sorry

-- Question 2: Prove a_n ≤ 2 * a_{n+1} for any n ∈ N*
theorem an_le_2an_next (n : ℕ) (h_n : n > 0) : a n ≤ 2 * a (n + 1) :=
  sorry

-- Question 3: Prove 2 - 1 / 2^(n - 1) ≤ S_n < 3 for any n ∈ N*
theorem sum_bounds (n : ℕ) (h_n : n > 0) : 
  2 - 1 / 2 ^ (n - 1) ≤ S n ∧ S n < 3 :=
  sorry

end NUMINAMATH_GPT_find_a2_an_le_2an_next_sum_bounds_l1330_133014


namespace NUMINAMATH_GPT_sum_remainders_l1330_133051

theorem sum_remainders (a b c : ℕ) (h₁ : a % 30 = 7) (h₂ : b % 30 = 11) (h₃ : c % 30 = 23) : 
  (a + b + c) % 30 = 11 := 
by
  sorry

end NUMINAMATH_GPT_sum_remainders_l1330_133051


namespace NUMINAMATH_GPT_flight_time_l1330_133068

def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def total_distance : ℕ := 248

theorem flight_time : (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed) > 0 → 
                      total_distance / (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed) = 2 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_flight_time_l1330_133068


namespace NUMINAMATH_GPT_incorrect_operation_B_l1330_133033

theorem incorrect_operation_B : (4 + 5)^2 ≠ 4^2 + 5^2 := 
  sorry

end NUMINAMATH_GPT_incorrect_operation_B_l1330_133033


namespace NUMINAMATH_GPT_joe_two_kinds_of_fruit_l1330_133091

-- Definitions based on the conditions
def meals := ["breakfast", "lunch", "snack", "dinner"] -- 4 meals
def fruits := ["apple", "orange", "banana"] -- 3 kinds of fruits

-- Probability that Joe consumes the same fruit for all meals
noncomputable def prob_same_fruit := (1 / 3) ^ 4

-- Probability that Joe eats at least two different kinds of fruits
noncomputable def prob_at_least_two_kinds := 1 - 3 * prob_same_fruit

theorem joe_two_kinds_of_fruit :
  prob_at_least_two_kinds = 26 / 27 :=
by
  -- Proof omitted for this theorem
  sorry

end NUMINAMATH_GPT_joe_two_kinds_of_fruit_l1330_133091


namespace NUMINAMATH_GPT_union_sets_l1330_133020

open Set

variable {α : Type*}

def setA : Set ℝ := { x | -2 < x ∧ x < 0 }
def setB : Set ℝ := { x | -1 < x ∧ x < 1 }
def setC : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem union_sets : setA ∪ setB = setC := 
by {
  sorry
}

end NUMINAMATH_GPT_union_sets_l1330_133020


namespace NUMINAMATH_GPT_range_of_m_l1330_133023

variable {x m : ℝ}

-- Definition of the first condition: ∀ x in ℝ, |x| + |x - 1| > m
def condition1 (m : ℝ) := ∀ x : ℝ, |x| + |x - 1| > m

-- Definition of the second condition: ∀ x in ℝ, (-(7 - 3 * m))^x is decreasing
def condition2 (m : ℝ) := ∀ x : ℝ, (-(7 - 3 * m))^x > (-(7 - 3 * m))^(x + 1)

-- Main theorem to prove m < 1
theorem range_of_m (h1 : condition1 m) (h2 : condition2 m) : m < 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1330_133023


namespace NUMINAMATH_GPT_value_of_x_l1330_133097

theorem value_of_x (x : ℝ) : (2 : ℝ) = 1 / (4 * x + 2) → x = -3 / 8 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_x_l1330_133097


namespace NUMINAMATH_GPT_solve_system_l1330_133083

theorem solve_system :
  ∃ (x y : ℕ), 
    (∃ d : ℕ, d ∣ 42 ∧ x^2 + y^2 = 468 ∧ d + (x * y) / d = 42) ∧ 
    (x = 12 ∧ y = 18) ∨ (x = 18 ∧ y = 12) :=
sorry

end NUMINAMATH_GPT_solve_system_l1330_133083


namespace NUMINAMATH_GPT_value_of_expression_in_third_quadrant_l1330_133074

theorem value_of_expression_in_third_quadrant (α : ℝ) (h1 : 180 < α ∧ α < 270) :
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) = -2 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_in_third_quadrant_l1330_133074


namespace NUMINAMATH_GPT_complement_is_correct_l1330_133021

variable (U : Set ℕ) (A : Set ℕ)

def complement (U : Set ℕ) (A : Set ℕ) : Set ℕ :=
  { x ∈ U | x ∉ A }

theorem complement_is_correct :
  (U = {1, 2, 3, 4, 5, 6, 7}) →
  (A = {2, 4, 5}) →
  complement U A = {1, 3, 6, 7} :=
by
  sorry

end NUMINAMATH_GPT_complement_is_correct_l1330_133021


namespace NUMINAMATH_GPT_max_profit_at_nine_l1330_133045

noncomputable def profit_function (x : ℝ) : ℝ :=
  -(1/3) * x ^ 3 + 81 * x - 234

theorem max_profit_at_nine :
  ∃ x, x = 9 ∧ ∀ y : ℝ, profit_function y ≤ profit_function 9 :=
by
  sorry

end NUMINAMATH_GPT_max_profit_at_nine_l1330_133045


namespace NUMINAMATH_GPT_point_translation_l1330_133073

theorem point_translation :
  ∃ (x y : ℤ), x = -1 ∧ y = -2 ↔ 
  ∃ (x₀ y₀ : ℤ), 
    x₀ = -3 ∧ y₀ = 2 ∧ 
    x = x₀ + 2 ∧ 
    y = y₀ - 4 := by
  sorry

end NUMINAMATH_GPT_point_translation_l1330_133073


namespace NUMINAMATH_GPT_find_f_of_3_l1330_133031

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end NUMINAMATH_GPT_find_f_of_3_l1330_133031


namespace NUMINAMATH_GPT_norm_of_5v_l1330_133078

noncomputable def norm_scale (v : ℝ × ℝ) (c : ℝ) : ℝ := c * (Real.sqrt (v.1^2 + v.2^2))

theorem norm_of_5v (v : ℝ × ℝ) (h : Real.sqrt (v.1^2 + v.2^2) = 6) : norm_scale v 5 = 30 := by
  sorry

end NUMINAMATH_GPT_norm_of_5v_l1330_133078


namespace NUMINAMATH_GPT_find_t_l1330_133095

variable (t : ℝ)

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (1, 0)
def c (t : ℝ) : ℝ × ℝ := (3 + t, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (h : dot_product (a) (c t) = dot_product (b) (c t)) : t = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_t_l1330_133095


namespace NUMINAMATH_GPT_joggers_difference_l1330_133001

-- Define the conditions as per the problem statement
variables (Tyson Alexander Christopher : ℕ)
variable (H1 : Alexander = Tyson + 22)
variable (H2 : Christopher = 20 * Tyson)
variable (H3 : Christopher = 80)

-- The theorem statement to prove Christopher bought 54 more joggers than Alexander
theorem joggers_difference : (Christopher - Alexander) = 54 :=
  sorry

end NUMINAMATH_GPT_joggers_difference_l1330_133001


namespace NUMINAMATH_GPT_buckets_required_l1330_133019

theorem buckets_required (C : ℝ) (N : ℝ):
  (62.5 * (2 / 5) * C = N * C) → N = 25 :=
by
  sorry

end NUMINAMATH_GPT_buckets_required_l1330_133019


namespace NUMINAMATH_GPT_probe_distance_before_refuel_l1330_133011

def total_distance : ℕ := 5555555555555
def distance_from_refuel : ℕ := 3333333333333
def distance_before_refuel : ℕ := 2222222222222

theorem probe_distance_before_refuel :
  total_distance - distance_from_refuel = distance_before_refuel := by
  sorry

end NUMINAMATH_GPT_probe_distance_before_refuel_l1330_133011


namespace NUMINAMATH_GPT_expression_undefined_at_x_l1330_133061

theorem expression_undefined_at_x (x : ℝ) : (x^2 - 18 * x + 81 = 0) → x = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_undefined_at_x_l1330_133061


namespace NUMINAMATH_GPT_find_functions_l1330_133089

open Function

theorem find_functions (f g : ℚ → ℚ) :
  (∀ x y : ℚ, f (g x - g y) = f (g x) - y) →
  (∀ x y : ℚ, g (f x - f y) = g (f x) - y) →
  ∃ c : ℚ, c ≠ 0 ∧ (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) :=
by
  sorry

end NUMINAMATH_GPT_find_functions_l1330_133089


namespace NUMINAMATH_GPT_total_sacks_needed_l1330_133009

def first_bakery_needs : ℕ := 2
def second_bakery_needs : ℕ := 4
def third_bakery_needs : ℕ := 12
def weeks : ℕ := 4

theorem total_sacks_needed :
  first_bakery_needs * weeks + second_bakery_needs * weeks + third_bakery_needs * weeks = 72 :=
by
  sorry

end NUMINAMATH_GPT_total_sacks_needed_l1330_133009


namespace NUMINAMATH_GPT_phone_purchase_initial_max_profit_additional_purchase_l1330_133098

-- Definitions for phone purchase prices and selling prices
def purchase_price_A : ℕ := 3000
def selling_price_A : ℕ := 3400
def purchase_price_B : ℕ := 3500
def selling_price_B : ℕ := 4000

-- Definitions for total expenditure and profit
def total_spent : ℕ := 32000
def total_profit : ℕ := 4400

-- Definitions for initial number of units purchased
def initial_units_A : ℕ := 6
def initial_units_B : ℕ := 4

-- Definitions for the additional purchase constraints and profit calculation
def max_additional_units : ℕ := 30
def additional_units_A : ℕ := 10
def additional_units_B : ℕ := max_additional_units - additional_units_A 
def max_profit : ℕ := 14000

theorem phone_purchase_initial:
  3000 * initial_units_A + 3500 * initial_units_B = total_spent ∧
  (selling_price_A - purchase_price_A) * initial_units_A + (selling_price_B - purchase_price_B) * initial_units_B = total_profit := by
  sorry 

theorem max_profit_additional_purchase:
  additional_units_A + additional_units_B = max_additional_units ∧
  additional_units_B ≤ 2 * additional_units_A ∧
  (selling_price_A - purchase_price_A) * additional_units_A + (selling_price_B - purchase_price_B) * additional_units_B = max_profit := by
  sorry

end NUMINAMATH_GPT_phone_purchase_initial_max_profit_additional_purchase_l1330_133098


namespace NUMINAMATH_GPT_sum_of_non_solutions_l1330_133064

theorem sum_of_non_solutions (A B C : ℝ) :
  (∀ x : ℝ, (x ≠ -C ∧ x ≠ -10) → (x + B) * (A * x + 40) / ((x + C) * (x + 10)) = 2) →
  (A = 2 ∧ B = 10 ∧ C = 20) →
  (-10 + -20 = -30) :=
by sorry

end NUMINAMATH_GPT_sum_of_non_solutions_l1330_133064


namespace NUMINAMATH_GPT_perpendicular_condition_line_through_point_l1330_133063

-- Definitions for lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y = 6
def l2 (m : ℝ) (x y : ℝ) : Prop := m * x + y = 3

-- Part 1: Prove that l1 is perpendicular to l2 if and only if m = -3 or m = 0
theorem perpendicular_condition (m : ℝ) : 
  (∀ (x : ℝ), ∀ (y : ℝ), (l1 m x y ∧ l2 m x y) → (m = 0 ∨ m = -3)) :=
sorry

-- Part 2: Prove the equations of line l given the conditions
theorem line_through_point (m : ℝ) (l : ℝ → ℝ → Prop) : 
  (∀ (P : ℝ × ℝ), (P = (1, 2*m)) → (l2 m P.1 P.2) → 
  ((∀ (x y : ℝ), l x y → 2 * x - y = 0) ∨ (∀ (x y: ℝ), l x y → x + 2 * y - 5 = 0))) :=
sorry

end NUMINAMATH_GPT_perpendicular_condition_line_through_point_l1330_133063


namespace NUMINAMATH_GPT_complex_solution_l1330_133004

theorem complex_solution (x : ℂ) (h : x^2 + 1 = 0) : x = Complex.I ∨ x = -Complex.I :=
by sorry

end NUMINAMATH_GPT_complex_solution_l1330_133004


namespace NUMINAMATH_GPT_solve_for_x_l1330_133036

theorem solve_for_x : ∀ x, (8 * x^2 + 150 * x + 2) / (3 * x + 50) = 4 * x + 2 ↔ x = -7 / 2 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1330_133036


namespace NUMINAMATH_GPT_gain_percent_l1330_133087

-- Let C be the cost price of one chocolate
-- Let S be the selling price of one chocolate
-- Given: 35 * C = 21 * S
-- Prove: The gain percent is 66.67%

theorem gain_percent (C S : ℝ) (h : 35 * C = 21 * S) : (S - C) / C * 100 = 200 / 3 :=
by sorry

end NUMINAMATH_GPT_gain_percent_l1330_133087


namespace NUMINAMATH_GPT_vector_perpendicular_l1330_133056

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 3)
def vec_diff : ℝ × ℝ := (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_perpendicular :
  dot_product vec_a vec_diff = 0 := by
  sorry

end NUMINAMATH_GPT_vector_perpendicular_l1330_133056


namespace NUMINAMATH_GPT_problem_one_problem_two_problem_three_l1330_133096

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def h (x : ℝ) : ℝ := (f x + 1) * g x

noncomputable def M (x : ℝ) : ℝ :=
  if f x >= g x then g x else f x

noncomputable def condition_one : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 8 → -6 ≤ h x ∧ h x ≤ 2

noncomputable def condition_two : Prop :=
  ∃ x, (M x = 1 ∧ 0 < x ∧ x ≤ 2) ∧ (∀ y, 0 < y ∧ y < x → M y < 1)

noncomputable def condition_three : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 8 → f (x^2) * f (Real.sqrt x) ≥ g x * -3

theorem problem_one : condition_one := sorry
theorem problem_two : condition_two := sorry
theorem problem_three : condition_three := sorry

end NUMINAMATH_GPT_problem_one_problem_two_problem_three_l1330_133096


namespace NUMINAMATH_GPT_triangle_perimeter_l1330_133040

theorem triangle_perimeter (a b : ℝ) (f : ℝ → Prop) 
  (h₁ : a = 7) (h₂ : b = 11)
  (eqn : ∀ x, f x ↔ x^2 - 25 = 2 * (x - 5)^2)
  (h₃ : ∃ x, f x ∧ 4 < x ∧ x < 18) :
  ∃ p : ℝ, (p = a + b + 5 ∨ p = a + b + 15) :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1330_133040


namespace NUMINAMATH_GPT_number_of_comic_books_l1330_133076

def fairy_tale_books := 305
def science_and_technology_books := fairy_tale_books + 115
def total_books := fairy_tale_books + science_and_technology_books
def comic_books := total_books * 4

theorem number_of_comic_books : comic_books = 2900 := by
  sorry

end NUMINAMATH_GPT_number_of_comic_books_l1330_133076


namespace NUMINAMATH_GPT_smallest_number_with_unique_digits_summing_to_32_exists_l1330_133050

theorem smallest_number_with_unique_digits_summing_to_32_exists : 
  ∃ n : ℕ, n / 10000 < 10 ∧ (n % 10 ≠ (n / 10) % 10) ∧ 
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧ 
  ((n / 100) % 10 ≠ (n / 1000) % 10) ∧ 
  ((n / 1000) % 10 ≠ (n / 10000) % 10) ∧ 
  (n % 10 + (n / 10) % 10 + (n / 100) % 10 + (n / 1000) % 10 + (n / 10000) % 10 = 32) := 
sorry

end NUMINAMATH_GPT_smallest_number_with_unique_digits_summing_to_32_exists_l1330_133050


namespace NUMINAMATH_GPT_number_of_vans_needed_l1330_133008

theorem number_of_vans_needed (capacity_per_van : ℕ) (students : ℕ) (adults : ℕ)
  (h_capacity : capacity_per_van = 9)
  (h_students : students = 40)
  (h_adults : adults = 14) :
  (students + adults + capacity_per_van - 1) / capacity_per_van = 6 := by
  sorry

end NUMINAMATH_GPT_number_of_vans_needed_l1330_133008


namespace NUMINAMATH_GPT_tan_identity_given_condition_l1330_133005

variable (α : Real)

theorem tan_identity_given_condition :
  (Real.tan α + 1 / Real.tan α = 9 / 4) →
  (Real.tan α ^ 2 + 1 / (Real.sin α * Real.cos α) + 1 / Real.tan α ^ 2 = 85 / 16) := 
by
  sorry

end NUMINAMATH_GPT_tan_identity_given_condition_l1330_133005


namespace NUMINAMATH_GPT_symmetry_sum_zero_l1330_133053

theorem symmetry_sum_zero (v : ℝ → ℝ) 
  (h_sym : ∀ x : ℝ, v (-x) = -v x) : 
  v (-2.00) + v (-1.00) + v (1.00) + v (2.00) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_symmetry_sum_zero_l1330_133053


namespace NUMINAMATH_GPT_max_travel_within_budget_l1330_133080

noncomputable def rental_cost_per_day : ℝ := 30
noncomputable def insurance_fee_per_day : ℝ := 10
noncomputable def mileage_cost_per_mile : ℝ := 0.18
noncomputable def budget : ℝ := 75
noncomputable def minimum_required_travel : ℝ := 100

theorem max_travel_within_budget : ∀ (rental_cost_per_day insurance_fee_per_day mileage_cost_per_mile budget minimum_required_travel), 
  rental_cost_per_day = 30 → 
  insurance_fee_per_day = 10 → 
  mileage_cost_per_mile = 0.18 → 
  budget = 75 →
  minimum_required_travel = 100 →
  (minimum_required_travel + (budget - rental_cost_per_day - insurance_fee_per_day - mileage_cost_per_mile * minimum_required_travel) / mileage_cost_per_mile) = 194 := 
by
  intros rental_cost_per_day insurance_fee_per_day mileage_cost_per_mile budget minimum_required_travel h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end NUMINAMATH_GPT_max_travel_within_budget_l1330_133080


namespace NUMINAMATH_GPT_Nina_can_buy_8_widgets_at_reduced_cost_l1330_133012

def money_Nina_has : ℕ := 48
def widgets_she_can_buy_initially : ℕ := 6
def reduction_per_widget : ℕ := 2

theorem Nina_can_buy_8_widgets_at_reduced_cost :
  let initial_cost_per_widget := money_Nina_has / widgets_she_can_buy_initially
  let reduced_cost_per_widget := initial_cost_per_widget - reduction_per_widget
  money_Nina_has / reduced_cost_per_widget = 8 :=
by
  sorry

end NUMINAMATH_GPT_Nina_can_buy_8_widgets_at_reduced_cost_l1330_133012


namespace NUMINAMATH_GPT_precisely_hundred_million_l1330_133062

-- Defining the options as an enumeration type
inductive Precision
| HundredBillion
| Billion
| HundredMillion
| Percent

-- The given figure in billions
def givenFigure : Float := 21.658

-- The correct precision is HundredMillion
def correctPrecision : Precision := Precision.HundredMillion

-- The theorem to prove the correctness of the figure's precision
theorem precisely_hundred_million : correctPrecision = Precision.HundredMillion :=
by
  sorry

end NUMINAMATH_GPT_precisely_hundred_million_l1330_133062


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_l1330_133042

variable (A b l : ℝ)

theorem breadth_of_rectangular_plot :
  (A = 15 * b) ∧ (l = b + 10) ∧ (A = l * b) → b = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_l1330_133042


namespace NUMINAMATH_GPT_gcf_75_100_l1330_133084

theorem gcf_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_gcf_75_100_l1330_133084


namespace NUMINAMATH_GPT_cut_scene_length_l1330_133041

theorem cut_scene_length (original_length final_length : ℕ) (h1 : original_length = 60) (h2 : final_length = 52) : original_length - final_length = 8 := 
by 
  sorry

end NUMINAMATH_GPT_cut_scene_length_l1330_133041


namespace NUMINAMATH_GPT_condition_suff_and_nec_l1330_133058

def p (x : ℝ) : Prop := |x + 2| ≤ 3
def q (x : ℝ) : Prop := x < -8

theorem condition_suff_and_nec (x : ℝ) : p x ↔ ¬ q x :=
by
  sorry

end NUMINAMATH_GPT_condition_suff_and_nec_l1330_133058
