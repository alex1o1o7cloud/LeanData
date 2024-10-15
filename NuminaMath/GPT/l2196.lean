import Mathlib

namespace NUMINAMATH_GPT_fifth_term_of_arithmetic_sequence_is_minus_three_l2196_219657

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

theorem fifth_term_of_arithmetic_sequence_is_minus_three (a d : ℤ) :
  (arithmetic_sequence a d 11 = 25) ∧ (arithmetic_sequence a d 12 = 29) →
  (arithmetic_sequence a d 4 = -3) :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_fifth_term_of_arithmetic_sequence_is_minus_three_l2196_219657


namespace NUMINAMATH_GPT_find_M_l2196_219665

theorem find_M :
  ∃ (M : ℕ), 1001 + 1003 + 1005 + 1007 + 1009 = 5100 - M ∧ M = 75 :=
by
  sorry

end NUMINAMATH_GPT_find_M_l2196_219665


namespace NUMINAMATH_GPT_min_distance_to_circle_l2196_219618

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 - 1)^2 + Q.2^2 = 4

def P : ℝ × ℝ := (-2, -3)
def center : ℝ × ℝ := (1, 0)
def radius : ℝ := 2

theorem min_distance_to_circle : ∃ Q : ℝ × ℝ, is_on_circle Q ∧ distance P Q = 3 * (Real.sqrt 2) - radius :=
by
  sorry

end NUMINAMATH_GPT_min_distance_to_circle_l2196_219618


namespace NUMINAMATH_GPT_general_term_formula_sum_first_n_terms_l2196_219626

noncomputable def a_n (n : ℕ) : ℕ := 2^(n - 1)

def S (n : ℕ) : ℕ := n * (2^(n - 1))  -- Placeholder function for the sum of the first n terms

theorem general_term_formula (a_3_eq_2a_2 : 2^(3 - 1) = 2 * 2^(2 - 1)) (S3_eq_7 : S 3 = 7) :
  ∀ n, a_n n = 2^(n - 1) :=
sorry

def T (n : ℕ) : ℕ := 4 - ((4 + 2 * n) / 2^n) -- Placeholder function for calculating T_n

theorem sum_first_n_terms (a_3_eq_2a_2 : 2^(3 - 1) = 2 * 2^(2 - 1)) (S3_eq_7 : S 3 = 7) :
  ∀ n, T n = 4 - ((4 + 2*n) / 2^n) :=
sorry

end NUMINAMATH_GPT_general_term_formula_sum_first_n_terms_l2196_219626


namespace NUMINAMATH_GPT_minimum_value_of_f_symmetry_of_f_monotonic_decreasing_f_l2196_219693

noncomputable def f (x : Real) : Real := Real.cos (2*x) - 2*Real.sin x + 1

theorem minimum_value_of_f : ∃ x : Real, f x = -2 := sorry

theorem symmetry_of_f : ∀ x : Real, f x = f (π - x) := sorry

theorem monotonic_decreasing_f : ∀ x y : Real, 0 < x ∧ x < y ∧ y < π / 2 → f y < f x := sorry

end NUMINAMATH_GPT_minimum_value_of_f_symmetry_of_f_monotonic_decreasing_f_l2196_219693


namespace NUMINAMATH_GPT_ants_of_species_X_on_day_6_l2196_219632

/-- Given the initial populations of Species X and Species Y and their growth rates,
    prove the number of Species X ants on Day 6. -/
theorem ants_of_species_X_on_day_6 
  (x y : ℕ)  -- Number of Species X and Y ants on Day 0
  (h1 : x + y = 40)  -- Total number of ants on Day 0
  (h2 : 64 * x + 4096 * y = 21050)  -- Total number of ants on Day 6
  :
  64 * x = 2304 := 
sorry

end NUMINAMATH_GPT_ants_of_species_X_on_day_6_l2196_219632


namespace NUMINAMATH_GPT_geometric_sequence_vertex_property_l2196_219676

theorem geometric_sequence_vertex_property (a b c d : ℝ) 
  (h_geom : ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r)
  (h_vertex : b = 1 ∧ c = 2) : a * d = b * c :=
by sorry

end NUMINAMATH_GPT_geometric_sequence_vertex_property_l2196_219676


namespace NUMINAMATH_GPT_complex_imaginary_unit_sum_l2196_219650

theorem complex_imaginary_unit_sum (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 = -1 := 
by sorry

end NUMINAMATH_GPT_complex_imaginary_unit_sum_l2196_219650


namespace NUMINAMATH_GPT_inequality_one_inequality_two_l2196_219689

-- Definitions of the three positive real numbers and their sum of reciprocals squared is equal to 1
variables {a b c : ℝ}
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1)

-- First proof that (1/a + 1/b + 1/c) <= sqrt(3)
theorem inequality_one (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1) :
  (1 / a) + (1 / b) + (1 / c) ≤ Real.sqrt 3 :=
sorry

-- Second proof that (a^2/b^4) + (b^2/c^4) + (c^2/a^4) >= 1
theorem inequality_two (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1) :
  (a^2 / b^4) + (b^2 / c^4) + (c^2 / a^4) ≥ 1 :=
sorry

end NUMINAMATH_GPT_inequality_one_inequality_two_l2196_219689


namespace NUMINAMATH_GPT_sin_cos_eq_sqrt2_l2196_219660

theorem sin_cos_eq_sqrt2 (x : ℝ) (h0 : 0 ≤ x) (h1 : x ≤ 2 * Real.pi) (h2 : Real.sin x - Real.cos x = Real.sqrt 2) :
  x = (3 * Real.pi) / 4 :=
sorry

end NUMINAMATH_GPT_sin_cos_eq_sqrt2_l2196_219660


namespace NUMINAMATH_GPT_ac_bc_ratios_l2196_219671

theorem ac_bc_ratios (A B C : ℝ) (m n : ℕ) (h : AC / BC = m / n) : 
  if m ≠ n then
    ((AC / AB = m / (m+n) ∧ BC / AB = n / (m+n)) ∨ 
     (AC / AB = m / (n-m) ∧ BC / AB = n / (n-m)))
  else 
    (AC / AB = 1 / 2 ∧ BC / AB = 1 / 2) := sorry

end NUMINAMATH_GPT_ac_bc_ratios_l2196_219671


namespace NUMINAMATH_GPT_three_f_l2196_219672

noncomputable def f (x : ℝ) : ℝ := sorry

theorem three_f (x : ℝ) (hx : 0 < x) (h : ∀ y > 0, f (3 * y) = 5 / (3 + y)) :
  3 * f x = 45 / (9 + x) :=
by
  sorry

end NUMINAMATH_GPT_three_f_l2196_219672


namespace NUMINAMATH_GPT_find_minimum_value_l2196_219610

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + 6*y^2 - 2*x*y - 14*x - 6*y + 72

theorem find_minimum_value :
  let x := 9
  let y := 2
  (∀ x y : ℝ, f x y ≥ 3) ∧ (f 9 2 = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_minimum_value_l2196_219610


namespace NUMINAMATH_GPT_jerry_weighted_mean_l2196_219673

noncomputable def weighted_mean (aunt uncle sister cousin friend1 friend2 friend3 friend4 friend5 : ℝ)
    (eur_to_usd gbp_to_usd cad_to_usd : ℝ) (family_weight friends_weight : ℝ) : ℝ :=
  let uncle_usd := uncle * eur_to_usd
  let friend3_usd := friend3 * eur_to_usd
  let friend4_usd := friend4 * gbp_to_usd
  let cousin_usd := cousin * cad_to_usd
  let family_sum := aunt + uncle_usd + sister + cousin_usd
  let friends_sum := friend1 + friend2 + friend3_usd + friend4_usd + friend5
  family_sum * family_weight + friends_sum * friends_weight

theorem jerry_weighted_mean : 
  weighted_mean 9.73 9.43 7.25 20.37 22.16 23.51 18.72 15.53 22.84 
               1.20 1.38 0.82 0.40 0.60 = 85.4442 := 
sorry

end NUMINAMATH_GPT_jerry_weighted_mean_l2196_219673


namespace NUMINAMATH_GPT_total_students_l2196_219644

-- Define the conditions
def ratio_girls_boys (G B : ℕ) : Prop := G / B = 1 / 2
def ratio_math_girls (M N : ℕ) : Prop := M / N = 3 / 1
def ratio_sports_boys (S T : ℕ) : Prop := S / T = 4 / 1

-- Define the problem statement
theorem total_students (G B M N S T : ℕ) 
  (h1 : ratio_girls_boys G B)
  (h2 : ratio_math_girls M N)
  (h3 : ratio_sports_boys S T)
  (h4 : M = 12)
  (h5 : G = M + N)
  (h6 : G = 16) 
  (h7 : B = 32) : 
  G + B = 48 :=
sorry

end NUMINAMATH_GPT_total_students_l2196_219644


namespace NUMINAMATH_GPT_abc_correct_and_c_not_true_l2196_219647

theorem abc_correct_and_c_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  a^2 > b^2 ∧ ab > b^2 ∧ (1/(a+b) > 1/a) ∧ ¬(1/a < 1/b) :=
  sorry

end NUMINAMATH_GPT_abc_correct_and_c_not_true_l2196_219647


namespace NUMINAMATH_GPT_length_decrease_by_33_percent_l2196_219663

theorem length_decrease_by_33_percent (L W L_new : ℝ) 
  (h1 : L * W = L_new * 1.5 * W) : 
  L_new = (2 / 3) * L ∧ ((1 - (2 / 3)) * 100 = 33.33) := 
by
  sorry

end NUMINAMATH_GPT_length_decrease_by_33_percent_l2196_219663


namespace NUMINAMATH_GPT_trig_identity_proof_l2196_219667

theorem trig_identity_proof :
  (Real.cos (10 * Real.pi / 180) * Real.sin (70 * Real.pi / 180) - Real.cos (80 * Real.pi / 180) * Real.sin (20 * Real.pi / 180)) = (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l2196_219667


namespace NUMINAMATH_GPT_trig_identity_1_trig_identity_2_l2196_219607

theorem trig_identity_1 (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) (h₁ : Real.tan θ = 2) :
  (Real.sin (π - θ) + Real.sin (3 * π / 2 + θ)) / 
  (3 * Real.sin (π / 2 - θ) - 2 * Real.sin (π + θ)) = 1 / 7 :=
by sorry

theorem trig_identity_2 (θ : ℝ) (h₀ : 0 < θ ∧ θ < π / 2) (h₁ : Real.tan θ = 2) :
  (1 - Real.cos (2 * θ)) / 
  (Real.sin (2 * θ) + Real.cos (2 * θ)) = 8 :=
by sorry

end NUMINAMATH_GPT_trig_identity_1_trig_identity_2_l2196_219607


namespace NUMINAMATH_GPT_daves_initial_apps_l2196_219656

theorem daves_initial_apps : ∃ (X : ℕ), X + 11 - 17 = 4 ∧ X = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_daves_initial_apps_l2196_219656


namespace NUMINAMATH_GPT_g_of_2_l2196_219622

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : x * g y = 2 * y * g x 
axiom g_of_10 : g 10 = 5

theorem g_of_2 : g 2 = 2 :=
by
    sorry

end NUMINAMATH_GPT_g_of_2_l2196_219622


namespace NUMINAMATH_GPT_central_angle_measure_l2196_219623

-- Given conditions
def radius : ℝ := 2
def area : ℝ := 4

-- Central angle α
def central_angle : ℝ := 2

-- Theorem statement: The central angle measure is 2 radians
theorem central_angle_measure :
  ∃ α : ℝ, α = central_angle ∧ area = (1/2) * (α * radius) := 
sorry

end NUMINAMATH_GPT_central_angle_measure_l2196_219623


namespace NUMINAMATH_GPT_xiaoming_problem_l2196_219696

theorem xiaoming_problem :
  (- 1 / 24) / (1 / 3 - 1 / 6 + 3 / 8) = - 1 / 13 :=
by
  sorry

end NUMINAMATH_GPT_xiaoming_problem_l2196_219696


namespace NUMINAMATH_GPT_pushups_percentage_l2196_219648

def total_exercises : ℕ := 12 + 8 + 20

def percentage_pushups (total_ex: ℕ) : ℕ := (8 * 100) / total_ex

theorem pushups_percentage (h : total_exercises = 40) : percentage_pushups total_exercises = 20 :=
by
  sorry

end NUMINAMATH_GPT_pushups_percentage_l2196_219648


namespace NUMINAMATH_GPT_percentage_answered_first_correctly_l2196_219619

-- Defining the given conditions
def percentage_answered_second_correctly : ℝ := 0.25
def percentage_answered_neither_correctly : ℝ := 0.20
def percentage_answered_both_correctly : ℝ := 0.20

-- Lean statement for the proof problem
theorem percentage_answered_first_correctly :
  ∃ a : ℝ, a + percentage_answered_second_correctly - percentage_answered_both_correctly = 0.80 ∧ a = 0.75 := by
  sorry

end NUMINAMATH_GPT_percentage_answered_first_correctly_l2196_219619


namespace NUMINAMATH_GPT_find_a_l2196_219620

theorem find_a (a : ℝ) (h_pos : a > 0) :
  (∀ x y : ℤ, x^2 - a * (x : ℝ) + 4 * a = 0) →
  a = 25 ∨ a = 18 ∨ a = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2196_219620


namespace NUMINAMATH_GPT_joe_eats_different_fruits_l2196_219666

noncomputable def joe_probability : ℚ :=
  let single_fruit_prob := (1 / 3) ^ 4
  let all_same_fruit_prob := 3 * single_fruit_prob
  let at_least_two_diff_fruits_prob := 1 - all_same_fruit_prob
  at_least_two_diff_fruits_prob

theorem joe_eats_different_fruits :
  joe_probability = 26 / 27 :=
by
  -- The proof is omitted for this task
  sorry

end NUMINAMATH_GPT_joe_eats_different_fruits_l2196_219666


namespace NUMINAMATH_GPT_no_one_is_always_largest_l2196_219639

theorem no_one_is_always_largest (a b c d : ℝ) :
  a - 2 = b + 3 ∧ a - 2 = c * 2 ∧ a - 2 = d + 5 →
  ∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → (x ≤ c ∨ x ≤ a) :=
by
  -- The proof requires assuming the conditions and showing that no variable is always the largest.
  intro h cond
  sorry

end NUMINAMATH_GPT_no_one_is_always_largest_l2196_219639


namespace NUMINAMATH_GPT_minimum_rows_required_l2196_219636

theorem minimum_rows_required (total_students : ℕ) (max_students_per_school : ℕ) (seats_per_row : ℕ) (num_schools : ℕ) 
    (h_total_students : total_students = 2016) 
    (h_max_students_per_school : max_students_per_school = 45) 
    (h_seats_per_row : seats_per_row = 168) 
    (h_num_schools : num_schools = 46) : 
    ∃ (min_rows : ℕ), min_rows = 16 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_minimum_rows_required_l2196_219636


namespace NUMINAMATH_GPT_geometric_sequence_inequality_l2196_219668

variable (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ)

-- Conditions
def geometric_sequence (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ) : Prop :=
  a₂ = a₁ * q ∧
  a₃ = a₁ * q^2 ∧
  a₄ = a₁ * q^3 ∧
  a₅ = a₁ * q^4 ∧
  a₆ = a₁ * q^5 ∧
  a₇ = a₁ * q^6 ∧
  a₈ = a₁ * q^7

theorem geometric_sequence_inequality
  (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q : ℝ)
  (h_seq : geometric_sequence a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ q)
  (h_a₁_pos : 0 < a₁)
  (h_q_ne_1 : q ≠ 1) :
  a₁ + a₈ > a₄ + a₅ :=
by 
-- Proof omitted
sorry

end NUMINAMATH_GPT_geometric_sequence_inequality_l2196_219668


namespace NUMINAMATH_GPT_harly_initial_dogs_l2196_219698

theorem harly_initial_dogs (x : ℝ) 
  (h1 : 0.40 * x + 0.60 * x + 5 = 53) : 
  x = 80 := 
by 
  sorry

end NUMINAMATH_GPT_harly_initial_dogs_l2196_219698


namespace NUMINAMATH_GPT_negate_universal_prop_l2196_219642

theorem negate_universal_prop :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 - 2*x + 2 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negate_universal_prop_l2196_219642


namespace NUMINAMATH_GPT_value_of_x_plus_inv_x_l2196_219662

theorem value_of_x_plus_inv_x (x : ℝ) (h : x + (1 / x) = v) (hr : x^2 + (1 / x)^2 = 23) : v = 5 :=
sorry

end NUMINAMATH_GPT_value_of_x_plus_inv_x_l2196_219662


namespace NUMINAMATH_GPT_sum_due_is_correct_l2196_219613

-- Define constants for Banker's Discount and True Discount
def BD : ℝ := 288
def TD : ℝ := 240

-- Define Banker's Gain as the difference between BD and TD
def BG : ℝ := BD - TD

-- Define the sum due (S.D.) as the face value including True Discount and Banker's Gain
def SD : ℝ := TD + BG

-- Create a theorem to prove the sum due is Rs. 288
theorem sum_due_is_correct : SD = 288 :=
by
  -- Skipping proof with sorry; expect this statement to be true based on given conditions 
  sorry

end NUMINAMATH_GPT_sum_due_is_correct_l2196_219613


namespace NUMINAMATH_GPT_count_ways_to_choose_one_person_l2196_219615

theorem count_ways_to_choose_one_person (A B : ℕ) (hA : A = 3) (hB : B = 5) : A + B = 8 :=
by
  sorry

end NUMINAMATH_GPT_count_ways_to_choose_one_person_l2196_219615


namespace NUMINAMATH_GPT_find_m_l2196_219601

theorem find_m (x1 x2 m : ℝ)
  (h1 : ∀ x, x^2 - 4 * x + m = 0 → x = x1 ∨ x = x2)
  (h2 : x1 + x2 - x1 * x2 = 1) :
  m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_l2196_219601


namespace NUMINAMATH_GPT_min_value_of_expression_l2196_219621

theorem min_value_of_expression (x : ℝ) (hx : 0 < x) : 4 * x + 1 / x ^ 6 ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l2196_219621


namespace NUMINAMATH_GPT_cost_hour_excess_is_1point75_l2196_219627

noncomputable def cost_per_hour_excess (x : ℝ) : Prop :=
  let total_hours := 9
  let initial_cost := 15
  let excess_hours := total_hours - 2
  let total_cost := initial_cost + excess_hours * x
  let average_cost_per_hour := 3.0277777777777777
  (total_cost / total_hours) = average_cost_per_hour

theorem cost_hour_excess_is_1point75 : cost_per_hour_excess 1.75 :=
by
  sorry

end NUMINAMATH_GPT_cost_hour_excess_is_1point75_l2196_219627


namespace NUMINAMATH_GPT_evaluate_fx_plus_2_l2196_219641

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x - 1)

theorem evaluate_fx_plus_2 (x : ℝ) (h : x ^ 2 ≠ 1) : 
  f (x + 2) = (x + 3) / (x + 1) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fx_plus_2_l2196_219641


namespace NUMINAMATH_GPT_sum_of_max_values_l2196_219690

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - Real.cos x)

theorem sum_of_max_values : (f π + f (3 * π)) = (Real.exp π + Real.exp (3 * π)) := 
by sorry

end NUMINAMATH_GPT_sum_of_max_values_l2196_219690


namespace NUMINAMATH_GPT_gcd_459_357_polynomial_at_neg4_l2196_219652

-- Statement for the GCD problem
theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

-- Definition of the polynomial
def f (x : Int) : Int :=
  12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

-- Statement for the polynomial evaluation problem
theorem polynomial_at_neg4 : f (-4) = 3392 := by
  sorry

end NUMINAMATH_GPT_gcd_459_357_polynomial_at_neg4_l2196_219652


namespace NUMINAMATH_GPT_final_score_is_83_l2196_219683

def running_score : ℕ := 90
def running_weight : ℚ := 0.5

def fancy_jump_rope_score : ℕ := 80
def fancy_jump_rope_weight : ℚ := 0.3

def jump_rope_score : ℕ := 70
def jump_rope_weight : ℚ := 0.2

noncomputable def final_score : ℚ := 
  running_score * running_weight + 
  fancy_jump_rope_score * fancy_jump_rope_weight + 
  jump_rope_score * jump_rope_weight

theorem final_score_is_83 : final_score = 83 := 
  by
    sorry

end NUMINAMATH_GPT_final_score_is_83_l2196_219683


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l2196_219603

theorem quadratic_distinct_real_roots (k : ℝ) : k < 1 / 2 ∧ k ≠ 0 ↔ (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (k * x1^2 - 2 * x1 + 2 = 0) ∧ (k * x2^2 - 2 * x2 + 2 = 0)) := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l2196_219603


namespace NUMINAMATH_GPT_wire_length_ratio_l2196_219614

noncomputable def total_wire_length_bonnie (pieces : Nat) (length_per_piece : Nat) := 
  pieces * length_per_piece

noncomputable def volume_of_cube (edge_length : Nat) := 
  edge_length ^ 3

noncomputable def wire_length_roark_per_cube (edges_per_cube : Nat) (length_per_edge : Nat) (num_cubes : Nat) :=
  edges_per_cube * length_per_edge * num_cubes

theorem wire_length_ratio : 
  let bonnie_pieces := 12
  let bonnie_length_per_piece := 8
  let bonnie_edge_length := 8
  let roark_length_per_edge := 2
  let roark_edges_per_cube := 12
  let bonnie_wire_length := total_wire_length_bonnie bonnie_pieces bonnie_length_per_piece
  let bonnie_cube_volume := volume_of_cube bonnie_edge_length
  let roark_num_cubes := bonnie_cube_volume
  let roark_wire_length := wire_length_roark_per_cube roark_edges_per_cube roark_length_per_edge roark_num_cubes
  bonnie_wire_length / roark_wire_length = 1 / 128 :=
by
  sorry

end NUMINAMATH_GPT_wire_length_ratio_l2196_219614


namespace NUMINAMATH_GPT_total_bricks_proof_l2196_219678

-- Define the initial conditions
def initial_courses := 3
def bricks_per_course := 400
def additional_courses := 2

-- Compute the number of bricks removed from the last course
def bricks_removed_from_last_course (bricks_per_course: ℕ) : ℕ :=
  bricks_per_course / 2

-- Calculate the total number of bricks
def total_bricks (initial_courses : ℕ) (bricks_per_course : ℕ) (additional_courses : ℕ) (bricks_removed : ℕ) : ℕ :=
  (initial_courses + additional_courses) * bricks_per_course - bricks_removed

-- Given values and the proof problem
theorem total_bricks_proof :
  total_bricks initial_courses bricks_per_course additional_courses (bricks_removed_from_last_course bricks_per_course) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_total_bricks_proof_l2196_219678


namespace NUMINAMATH_GPT_rabbit_travel_time_l2196_219684

theorem rabbit_travel_time :
  let distance := 2
  let speed := 5
  let hours_to_minutes := 60
  (distance / speed) * hours_to_minutes = 24 := by
sorry

end NUMINAMATH_GPT_rabbit_travel_time_l2196_219684


namespace NUMINAMATH_GPT_fish_per_black_duck_l2196_219628

theorem fish_per_black_duck :
  ∀ (W_d B_d M_d : ℕ) (fish_per_W fish_per_M total_fish : ℕ),
    (fish_per_W = 5) →
    (fish_per_M = 12) →
    (W_d = 3) →
    (B_d = 7) →
    (M_d = 6) →
    (total_fish = 157) →
    (total_fish - (W_d * fish_per_W + M_d * fish_per_M)) = 70 →
    (70 / B_d) = 10 :=
by
  intros W_d B_d M_d fish_per_W fish_per_M total_fish hW hM hW_d hB_d hM_d htotal_fish hcalculation
  sorry

end NUMINAMATH_GPT_fish_per_black_duck_l2196_219628


namespace NUMINAMATH_GPT_weather_on_july_15_l2196_219654

theorem weather_on_july_15 
  (T: ℝ) (sunny: Prop) (W: ℝ) (crowded: Prop) 
  (h1: (T ≥ 85 ∧ sunny ∧ W < 15) → crowded) 
  (h2: ¬ crowded) : (T < 85 ∨ ¬ sunny ∨ W ≥ 15) :=
sorry

end NUMINAMATH_GPT_weather_on_july_15_l2196_219654


namespace NUMINAMATH_GPT_num_ways_to_convert_20d_l2196_219617

theorem num_ways_to_convert_20d (n d q : ℕ) (h : 5 * n + 10 * d + 25 * q = 2000) (hn : n ≥ 2) (hq : q ≥ 1) :
    ∃ k : ℕ, k = 130 := sorry

end NUMINAMATH_GPT_num_ways_to_convert_20d_l2196_219617


namespace NUMINAMATH_GPT_problem_statement_l2196_219674

variables {a c b d : ℝ} {x y q z : ℕ}

-- Given conditions:
def condition1 (a c : ℝ) (x q : ℕ) : Prop := a^(x + 1) = c^(q + 2)
def condition2 (a c : ℝ) (y z : ℕ) : Prop := c^(y + 3) = a^(z+ 4)

-- Goal statement
theorem problem_statement (a c : ℝ) (x y q z : ℕ) (h1 : condition1 a c x q) (h2 : condition2 a c y z) :
  (q + 2) * (z + 4) = (y + 3) * (x + 1) :=
sorry

end NUMINAMATH_GPT_problem_statement_l2196_219674


namespace NUMINAMATH_GPT_evaluate_expression_l2196_219606

theorem evaluate_expression :
  let a := 24
  let b := 7
  3 * (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 2258 :=
by
  let a := 24
  let b := 7
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2196_219606


namespace NUMINAMATH_GPT_c_value_for_infinite_solutions_l2196_219608

theorem c_value_for_infinite_solutions :
  ∀ (c : ℝ), (∀ (x : ℝ), 3 * (5 + c * x) = 15 * x + 15) ↔ c = 5 :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_c_value_for_infinite_solutions_l2196_219608


namespace NUMINAMATH_GPT_union_complement_eq_l2196_219651

open Set

variable (I A B : Set ℤ)
variable (I_def : I = {-3, -2, -1, 0, 1, 2})
variable (A_def : A = {-1, 1, 2})
variable (B_def : B = {-2, -1, 0})

theorem union_complement_eq :
  A ∪ (I \ B) = {-3, -1, 1, 2} :=
by 
  rw [I_def, A_def, B_def]
  sorry

end NUMINAMATH_GPT_union_complement_eq_l2196_219651


namespace NUMINAMATH_GPT_bobby_toy_cars_in_5_years_l2196_219669

noncomputable def toy_cars_after_n_years (initial_cars : ℕ) (percentage_increase : ℝ) (n : ℕ) : ℝ :=
initial_cars * (1 + percentage_increase)^n

theorem bobby_toy_cars_in_5_years :
  toy_cars_after_n_years 25 0.75 5 = 410 := by
  -- 25 * (1 + 0.75)^5 
  -- = 25 * (1.75)^5 
  -- ≈ 410.302734375
  -- After rounding
  sorry

end NUMINAMATH_GPT_bobby_toy_cars_in_5_years_l2196_219669


namespace NUMINAMATH_GPT_least_three_digit_with_product_l2196_219661

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_product (n : ℕ) (p : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 * d2 * d3 = p

theorem least_three_digit_with_product (p : ℕ) : ∃ n : ℕ, is_three_digit n ∧ digits_product n p ∧ 
  ∀ m : ℕ, is_three_digit m ∧ digits_product m p → n ≤ m :=
by
  use 116
  sorry

end NUMINAMATH_GPT_least_three_digit_with_product_l2196_219661


namespace NUMINAMATH_GPT_min_a2_b2_c2_l2196_219691

theorem min_a2_b2_c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 3 * b + 5 * c = 100) : 
  a^2 + b^2 + c^2 ≥ (5000 / 19) :=
by
  sorry

end NUMINAMATH_GPT_min_a2_b2_c2_l2196_219691


namespace NUMINAMATH_GPT_problem_f_2004_l2196_219605

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_f_2004 (a α b β : ℝ) 
  (h_non_zero : a ≠ 0 ∧ b ≠ 0 ∧ α ≠ 0 ∧ β ≠ 0) 
  (h_condition : f 2003 a α b β = 6) : 
  f 2004 a α b β = 2 := 
by
  sorry

end NUMINAMATH_GPT_problem_f_2004_l2196_219605


namespace NUMINAMATH_GPT_elise_hospital_distance_l2196_219675

noncomputable def distance_to_hospital (total_fare: ℝ) (base_price: ℝ) (toll_price: ℝ) 
(tip_percent: ℝ) (cost_per_mile: ℝ) (increase_percent: ℝ) (toll_count: ℕ) : ℝ :=
let base_and_tolls := base_price + (toll_price * toll_count)
let fare_before_tip := total_fare / (1 + tip_percent)
let distance_fare := fare_before_tip - base_and_tolls
let original_travel_fare := distance_fare / (1 + increase_percent)
original_travel_fare / cost_per_mile

theorem elise_hospital_distance : distance_to_hospital 34.34 3 2 0.15 4 0.20 3 = 5 := 
sorry

end NUMINAMATH_GPT_elise_hospital_distance_l2196_219675


namespace NUMINAMATH_GPT_find_c_l2196_219670

theorem find_c (x c : ℤ) (h1 : 3 * x + 8 = 5) (h2 : c * x - 7 = 1) : c = -8 :=
sorry

end NUMINAMATH_GPT_find_c_l2196_219670


namespace NUMINAMATH_GPT_set_D_cannot_form_triangle_l2196_219688

theorem set_D_cannot_form_triangle : ¬ (∃ a b c : ℝ, a = 2 ∧ b = 4 ∧ c = 6 ∧ 
  (a + b > c ∧ a + c > b ∧ b + c > a)) :=
by {
  sorry
}

end NUMINAMATH_GPT_set_D_cannot_form_triangle_l2196_219688


namespace NUMINAMATH_GPT_minimum_value_of_PQ_l2196_219631

theorem minimum_value_of_PQ {x y : ℝ} (P : ℝ × ℝ) (h₁ : (P.1 - 3)^2 + (P.2 - 4)^2 > 4)
  (h₂ : ∀ Q : ℝ × ℝ, (Q.1 - 3)^2 + (Q.2 - 4)^2 = 4 → (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = (P.1)^2 + (P.2)^2) :
  ∃ PQ_min : ℝ, PQ_min = 17/2 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_PQ_l2196_219631


namespace NUMINAMATH_GPT_max_value_quadratic_l2196_219609

theorem max_value_quadratic :
  (∃ x : ℝ, ∀ y : ℝ, -3*y^2 + 9*y + 24 ≤ -3*x^2 + 9*x + 24) ∧ (∃ x : ℝ, x = 3/2) :=
sorry

end NUMINAMATH_GPT_max_value_quadratic_l2196_219609


namespace NUMINAMATH_GPT_least_integer_value_l2196_219649

theorem least_integer_value :
  ∃ x : ℤ, (∀ x' : ℤ, (|3 * x' + 4| <= 18) → (x' >= x)) ∧ (|3 * x + 4| <= 18) ∧ x = -7 := 
sorry

end NUMINAMATH_GPT_least_integer_value_l2196_219649


namespace NUMINAMATH_GPT_marked_price_l2196_219645

theorem marked_price (original_price : ℝ) 
                     (discount1_rate : ℝ) 
                     (profit_rate : ℝ) 
                     (discount2_rate : ℝ)
                     (marked_price : ℝ) : 
                     original_price = 40 → 
                     discount1_rate = 0.15 → 
                     profit_rate = 0.25 → 
                     discount2_rate = 0.10 → 
                     marked_price = 47.20 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_marked_price_l2196_219645


namespace NUMINAMATH_GPT_cubic_eq_solutions_l2196_219611

theorem cubic_eq_solutions (x : ℝ) :
  x^3 - 4 * x = 0 ↔ x = 0 ∨ x = -2 ∨ x = 2 := by
  sorry

end NUMINAMATH_GPT_cubic_eq_solutions_l2196_219611


namespace NUMINAMATH_GPT_back_wheel_revolutions_calculation_l2196_219634

noncomputable def front_diameter : ℝ := 3 -- Diameter of the front wheel in feet
noncomputable def back_diameter : ℝ := 0.5 -- Diameter of the back wheel in feet
noncomputable def no_slippage : Prop := true -- No slippage condition
noncomputable def front_revolutions : ℕ := 150 -- Number of front wheel revolutions

theorem back_wheel_revolutions_calculation 
  (d_f : ℝ) (d_b : ℝ) (slippage : Prop) (n_f : ℕ) : 
  slippage → d_f = front_diameter → d_b = back_diameter → 
  n_f = front_revolutions → 
  ∃ n_b : ℕ, n_b = 900 := 
by
  sorry

end NUMINAMATH_GPT_back_wheel_revolutions_calculation_l2196_219634


namespace NUMINAMATH_GPT_carrot_cakes_in_february_l2196_219681

theorem carrot_cakes_in_february :
  (∃ (cakes_in_oct : ℕ) (cakes_in_nov : ℕ) (cakes_in_dec : ℕ) (cakes_in_jan : ℕ) (monthly_increase : ℕ),
      cakes_in_oct = 19 ∧
      cakes_in_nov = 21 ∧
      cakes_in_dec = 23 ∧
      cakes_in_jan = 25 ∧
      monthly_increase = 2 ∧
      cakes_in_february = cakes_in_jan + monthly_increase) →
  cakes_in_february = 27 :=
  sorry

end NUMINAMATH_GPT_carrot_cakes_in_february_l2196_219681


namespace NUMINAMATH_GPT_max_value_expression_l2196_219633

theorem max_value_expression : ∀ (a b c d : ℝ), 
  a ∈ Set.Icc (-4.5) 4.5 → 
  b ∈ Set.Icc (-4.5) 4.5 → 
  c ∈ Set.Icc (-4.5) 4.5 → 
  d ∈ Set.Icc (-4.5) 4.5 → 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 90 :=
by sorry

end NUMINAMATH_GPT_max_value_expression_l2196_219633


namespace NUMINAMATH_GPT_find_ticket_cost_l2196_219694

-- Define the initial amount Tony had
def initial_amount : ℕ := 20

-- Define the amount Tony paid for a hot dog
def hot_dog_cost : ℕ := 3

-- Define the amount Tony had after buying the ticket and the hot dog
def remaining_amount : ℕ := 9

-- Define the function to find the baseball ticket cost
def ticket_cost (t : ℕ) : Prop := initial_amount - t - hot_dog_cost = remaining_amount

-- The statement to prove
theorem find_ticket_cost : ∃ t : ℕ, ticket_cost t ∧ t = 8 := 
by 
  existsi 8
  unfold ticket_cost
  simp
  exact sorry

end NUMINAMATH_GPT_find_ticket_cost_l2196_219694


namespace NUMINAMATH_GPT_smallest_integer_in_range_l2196_219646

-- Given conditions
def is_congruent_6 (n : ℕ) : Prop := n % 6 = 1
def is_congruent_7 (n : ℕ) : Prop := n % 7 = 1
def is_congruent_8 (n : ℕ) : Prop := n % 8 = 1

-- Lean statement for the proof problem
theorem smallest_integer_in_range :
  ∃ n : ℕ, (n > 1) ∧ is_congruent_6 n ∧ is_congruent_7 n ∧ is_congruent_8 n ∧ (n = 169) ∧ (120 ≤ n ∧ n < 210) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_in_range_l2196_219646


namespace NUMINAMATH_GPT_cut_ribbon_l2196_219630

theorem cut_ribbon
    (length_ribbon : ℝ)
    (points : ℝ × ℝ × ℝ × ℝ × ℝ)
    (h_length : length_ribbon = 5)
    (h_points : points = (1, 2, 3, 4, 5)) :
    points.2.1 = (11 / 15) * length_ribbon :=
by
    sorry

end NUMINAMATH_GPT_cut_ribbon_l2196_219630


namespace NUMINAMATH_GPT_team_plays_60_games_in_division_l2196_219687

noncomputable def number_of_division_games (N M : ℕ) (hNM : N > M) (hM : M > 5) (h_total : 4 * N + 5 * M = 90) : ℕ :=
  4 * N

theorem team_plays_60_games_in_division (N M : ℕ) 
  (hNM : N > M) (hM : M > 5) (h_total : 4 * N + 5 * M = 90) 
  : number_of_division_games N M hNM hM h_total = 60 := 
sorry

end NUMINAMATH_GPT_team_plays_60_games_in_division_l2196_219687


namespace NUMINAMATH_GPT_sequence_remainder_mod_10_l2196_219629

def T : ℕ → ℕ := sorry -- Since the actual recursive definition is part of solution steps, we abstract it.
def remainder (n k : ℕ) : ℕ := n % k

theorem sequence_remainder_mod_10 (n : ℕ) (h: n = 2023) : remainder (T n) 10 = 6 :=
by 
  sorry

end NUMINAMATH_GPT_sequence_remainder_mod_10_l2196_219629


namespace NUMINAMATH_GPT_geometric_sequence_n_value_l2196_219604

theorem geometric_sequence_n_value (a₁ : ℕ) (q : ℕ) (a_n : ℕ) (n : ℕ) (h1 : a₁ = 1) (h2 : q = 2) (h3 : a_n = 64) (h4 : a_n = a₁ * q^(n-1)) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_n_value_l2196_219604


namespace NUMINAMATH_GPT_simplify_fraction_sum_l2196_219600

variable (a b c : ℝ)
variable (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)

theorem simplify_fraction_sum (x : ℝ) (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ( (x + a) ^ 2 / ((a - b) * (a - c))
  + (x + b) ^ 2 / ((b - a) * (b - c))
  + (x + c) ^ 2 / ((c - a) * (c - b)) )
  = a * x + b * x + c * x - a - b - c :=
sorry

end NUMINAMATH_GPT_simplify_fraction_sum_l2196_219600


namespace NUMINAMATH_GPT_tan_3theta_eq_2_11_sin_3theta_eq_22_125_l2196_219697

variable {θ : ℝ}

-- First, stating the condition \(\tan \theta = 2\)
axiom tan_theta_eq_2 : Real.tan θ = 2

-- Stating the proof problem for \(\tan 3\theta = \frac{2}{11}\)
theorem tan_3theta_eq_2_11 : Real.tan (3 * θ) = 2 / 11 :=
by 
  sorry

-- Stating the proof problem for \(\sin 3\theta = \frac{22}{125}\)
theorem sin_3theta_eq_22_125 : Real.sin (3 * θ) = 22 / 125 :=
by 
  sorry

end NUMINAMATH_GPT_tan_3theta_eq_2_11_sin_3theta_eq_22_125_l2196_219697


namespace NUMINAMATH_GPT_radius_ratio_in_right_triangle_l2196_219692

theorem radius_ratio_in_right_triangle (PQ QR PR PS SR : ℝ)
  (h₁ : PQ = 5) (h₂ : QR = 12) (h₃ : PR = 13)
  (h₄ : PS + SR = PR) (h₅ : PS / SR = 5 / 8)
  (r_p r_q : ℝ)
  (hr_p : r_p = (1 / 2 * PQ * PS / 3) / ((PQ + PS / 3 + PS) / 3))
  (hr_q : r_q = (1 / 2 * QR * SR) / ((PS / 3 + QR + SR) / 3)) :
  r_p / r_q = 175 / 576 :=
sorry

end NUMINAMATH_GPT_radius_ratio_in_right_triangle_l2196_219692


namespace NUMINAMATH_GPT_length_B1C1_l2196_219699

variable (AC BC : ℝ) (A1B1 : ℝ) (T : ℝ)

/-- Given a right triangle ABC with legs AC = 3 and BC = 4, and transformations
  of points to A1, B1, and C1 where A1B1 = 1 and angle B1 = 90 degrees,
  prove that the length of B1C1 is 12. -/
theorem length_B1C1 (h1 : AC = 3) (h2 : BC = 4) (h3 : A1B1 = 1) 
  (TABC : T = 6) (right_triangle_ABC : true) (right_triangle_A1B1C1 : true) : 
  B1C1 = 12 := 
sorry

end NUMINAMATH_GPT_length_B1C1_l2196_219699


namespace NUMINAMATH_GPT_find_k_l2196_219624

theorem find_k (k : ℝ) :
    (∀ x : ℝ, 4 * x^2 + k * x + 4 ≠ 0) → k = 8 :=
sorry

end NUMINAMATH_GPT_find_k_l2196_219624


namespace NUMINAMATH_GPT_range_of_x_plus_y_l2196_219643

theorem range_of_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x * y - (x + y) = 1) : 
  x + y ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_range_of_x_plus_y_l2196_219643


namespace NUMINAMATH_GPT_max_sum_square_pyramid_addition_l2196_219602

def square_pyramid_addition_sum (faces edges vertices : ℕ) : ℕ :=
  let new_faces := faces - 1 + 4
  let new_edges := edges + 4
  let new_vertices := vertices + 1
  new_faces + new_edges + new_vertices

theorem max_sum_square_pyramid_addition :
  square_pyramid_addition_sum 6 12 8 = 34 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_square_pyramid_addition_l2196_219602


namespace NUMINAMATH_GPT_value_of_b_plus_c_l2196_219659

theorem value_of_b_plus_c 
  (b c : ℝ) 
  (f : ℝ → ℝ)
  (h_def : ∀ x, f x = x^2 + 2 * b * x + c)
  (h_solution_set : ∀ x, f x ≤ 0 ↔ -1 ≤ x ∧ x ≤ 1) :
  b + c = -1 :=
sorry

end NUMINAMATH_GPT_value_of_b_plus_c_l2196_219659


namespace NUMINAMATH_GPT_mouse_jump_frog_jump_diff_l2196_219635

open Nat

theorem mouse_jump_frog_jump_diff :
  let grasshopper_jump := 19
  let frog_jump := grasshopper_jump + 10
  let mouse_jump := grasshopper_jump + 30
  mouse_jump - frog_jump = 20 :=
by
  let grasshopper_jump := 19
  let frog_jump := grasshopper_jump + 10
  let mouse_jump := grasshopper_jump + 30
  have h1 : frog_jump = 29 := by decide
  have h2 : mouse_jump = 49 := by decide
  have h3 : mouse_jump - frog_jump = 20 := by decide
  exact h3

end NUMINAMATH_GPT_mouse_jump_frog_jump_diff_l2196_219635


namespace NUMINAMATH_GPT_mass_percentage_O_is_correct_l2196_219658

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def num_Al_atoms : ℕ := 2
noncomputable def num_O_atoms : ℕ := 3

noncomputable def molar_mass_Al2O3 : ℝ :=
  (num_Al_atoms * molar_mass_Al) + (num_O_atoms * molar_mass_O)

noncomputable def mass_percentage_O_in_Al2O3 : ℝ :=
  ((num_O_atoms * molar_mass_O) / molar_mass_Al2O3) * 100

theorem mass_percentage_O_is_correct :
  mass_percentage_O_in_Al2O3 = 47.07 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_O_is_correct_l2196_219658


namespace NUMINAMATH_GPT_faster_train_length_225_l2196_219637

noncomputable def length_of_faster_train (speed_slower speed_faster : ℝ) (time : ℝ) : ℝ :=
  let relative_speed_kmph := speed_slower + speed_faster
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  relative_speed_mps * time

theorem faster_train_length_225 :
  length_of_faster_train 36 45 10 = 225 := by
  sorry

end NUMINAMATH_GPT_faster_train_length_225_l2196_219637


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l2196_219653

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x^2 + m*x + (m + 3) = 0)) ↔ (m < -2 ∨ m > 6) := 
sorry

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l2196_219653


namespace NUMINAMATH_GPT_quadratic_inequality_solution_non_empty_l2196_219686

theorem quadratic_inequality_solution_non_empty
  (a b c : ℝ) (h : a < 0) :
  ∃ x : ℝ, ax^2 + bx + c < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_non_empty_l2196_219686


namespace NUMINAMATH_GPT_student_tickets_second_day_l2196_219616

variable (S T x: ℕ)

theorem student_tickets_second_day (hT : T = 9) (h_eq1 : 4 * S + 3 * T = 79) (h_eq2 : 12 * S + x * T = 246) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_student_tickets_second_day_l2196_219616


namespace NUMINAMATH_GPT_find_acute_angle_x_l2196_219682

def a_parallel_b (x : ℝ) : Prop :=
  let a := (Real.sin x, 3 / 4)
  let b := (1 / 3, 1 / 2 * Real.cos x)
  b.1 * a.2 = a.1 * b.2

theorem find_acute_angle_x (x : ℝ) (h : a_parallel_b x) : x = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_acute_angle_x_l2196_219682


namespace NUMINAMATH_GPT_trigonometric_cos_value_l2196_219638

open Real

theorem trigonometric_cos_value (α : ℝ) (h : sin (α + π / 6) = 1 / 3) : 
  cos (2 * α - 2 * π / 3) = -7 / 9 := 
sorry

end NUMINAMATH_GPT_trigonometric_cos_value_l2196_219638


namespace NUMINAMATH_GPT_marie_finishes_ninth_task_at_730PM_l2196_219695

noncomputable def start_time : ℕ := 8 * 60 -- 8:00 AM in minutes
noncomputable def end_time_task_3 : ℕ := 11 * 60 + 30 -- 11:30 AM in minutes
noncomputable def total_tasks : ℕ := 9
noncomputable def tasks_done_by_1130AM : ℕ := 3
noncomputable def end_time_task_9 : ℕ := 19 * 60 + 30 -- 7:30 PM in minutes

theorem marie_finishes_ninth_task_at_730PM
    (h1 : start_time = 480) -- 8:00 AM
    (h2 : end_time_task_3 = 690) -- 11:30 AM
    (h3 : total_tasks = 9)
    (h4 : tasks_done_by_1130AM = 3)
    (h5 : end_time_task_9 = 1170) -- 7:30 PM
    : end_time_task_9 = start_time + ((end_time_task_3 - start_time) / tasks_done_by_1130AM) * total_tasks :=
sorry

end NUMINAMATH_GPT_marie_finishes_ninth_task_at_730PM_l2196_219695


namespace NUMINAMATH_GPT_part_a_part_b_l2196_219685

noncomputable def arithmetic_progression_a (a₁: ℕ) (r: ℕ) : ℕ :=
  a₁ + 3 * r

theorem part_a (a₁: ℕ) (r: ℕ) (h_a₁ : a₁ = 2) (h_r : r = 3) : arithmetic_progression_a a₁ r = 11 := 
by 
  sorry

noncomputable def arithmetic_progression_formula (d: ℕ) (r: ℕ) (n: ℕ) : ℕ :=
  d + (n - 1) * r

theorem part_b (a3: ℕ) (a6: ℕ) (a9: ℕ) (a4_plus_a7_plus_a10: ℕ) (a_sum: ℕ) (h_a3 : a3 = 3) (h_a6 : a6 = 6) (h_a9 : a9 = 9) 
  (h_a4a7a10 : a4_plus_a7_plus_a10 = 207) (h_asum : a_sum = 553) 
  (h_eqn1: 3 * a3 + a6 * 2 = 207) (h_eqn2: a_sum = 553): 
  arithmetic_progression_formula 9 10 11 = 109 := 
by 
  sorry

end NUMINAMATH_GPT_part_a_part_b_l2196_219685


namespace NUMINAMATH_GPT_repetend_of_5_over_13_l2196_219679

theorem repetend_of_5_over_13 : (∃ r : ℕ, r = 384615) :=
by
  let d := 13
  let n := 5
  let r := 384615
  -- Definitions to use:
  -- d is denominator 13
  -- n is numerator 5
  -- r is the repetend 384615
  sorry

end NUMINAMATH_GPT_repetend_of_5_over_13_l2196_219679


namespace NUMINAMATH_GPT_insurance_compensation_zero_l2196_219677

noncomputable def insured_amount : ℝ := 500000
noncomputable def deductible : ℝ := 0.01
noncomputable def actual_damage : ℝ := 4000

theorem insurance_compensation_zero :
  actual_damage < insured_amount * deductible → 0 = 0 := by
sorry

end NUMINAMATH_GPT_insurance_compensation_zero_l2196_219677


namespace NUMINAMATH_GPT_must_be_negative_when_x_is_negative_l2196_219664

open Real

theorem must_be_negative_when_x_is_negative (x : ℝ) (h : x < 0) : x^3 < 0 ∧ -x^4 < 0 := 
by
  sorry

end NUMINAMATH_GPT_must_be_negative_when_x_is_negative_l2196_219664


namespace NUMINAMATH_GPT_option_A_is_incorrect_option_B_is_incorrect_option_C_is_correct_option_D_is_incorrect_l2196_219640

variable (x y: ℝ)

theorem option_A_is_incorrect : 5 - 3 * (x + 1) ≠ 5 - 3 * x - 1 := 
by sorry

theorem option_B_is_incorrect : 2 - 4 * (x + 1/4) ≠ 2 - 4 * x + 1 := 
by sorry

theorem option_C_is_correct : 2 - 4 * (1/4 * x + 1) = 2 - x - 4 := 
by sorry

theorem option_D_is_incorrect : 2 * (x - 2) - 3 * (y - 1) ≠ 2 * x - 4 - 3 * y - 3 := 
by sorry

end NUMINAMATH_GPT_option_A_is_incorrect_option_B_is_incorrect_option_C_is_correct_option_D_is_incorrect_l2196_219640


namespace NUMINAMATH_GPT_find_remainder_l2196_219680

def dividend : ℝ := 17698
def divisor : ℝ := 198.69662921348313
def quotient : ℝ := 89
def remainder : ℝ := 14

theorem find_remainder :
  dividend = (divisor * quotient) + remainder :=
by 
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_find_remainder_l2196_219680


namespace NUMINAMATH_GPT_ratio_solution_l2196_219612

theorem ratio_solution (x : ℚ) : (1 : ℚ) / 3 = 5 / 3 / x → x = 5 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_ratio_solution_l2196_219612


namespace NUMINAMATH_GPT_total_eggs_l2196_219625

def e0 : ℝ := 47.0
def ei : ℝ := 5.0

theorem total_eggs : e0 + ei = 52.0 := by
  sorry

end NUMINAMATH_GPT_total_eggs_l2196_219625


namespace NUMINAMATH_GPT_original_profit_percentage_l2196_219655

theorem original_profit_percentage
  (C : ℝ) -- original cost
  (S : ℝ) -- selling price
  (y : ℝ) -- original profit percentage
  (hS : S = C * (1 + 0.01 * y)) -- condition for selling price based on original cost
  (hC' : S = 0.85 * C * (1 + 0.01 * (y + 20))) -- condition for selling price based on reduced cost
  : y = -89 :=
by
  sorry

end NUMINAMATH_GPT_original_profit_percentage_l2196_219655
