import Mathlib

namespace NUMINAMATH_GPT_problem_statement_l671_67172

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {4, 5}
def C_U (B : Set ℕ) : Set ℕ := U \ B

-- Statement
theorem problem_statement : A ∩ (C_U B) = {2} :=
  sorry

end NUMINAMATH_GPT_problem_statement_l671_67172


namespace NUMINAMATH_GPT_sum_of_angles_l671_67140

theorem sum_of_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (sin_α : Real.sin α = 2 * Real.sqrt 5 / 5) (sin_beta : Real.sin β = 3 * Real.sqrt 10 / 10) :
  α + β = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_sum_of_angles_l671_67140


namespace NUMINAMATH_GPT_sum_arithmetic_series_base8_l671_67113

theorem sum_arithmetic_series_base8 : 
  let n := 36
  let a := 1
  let l := 30 -- 36_8 in base 10 is 30
  let S := (n * (a + l)) / 2
  let sum_base10 := 558
  let sum_base8 := 1056 -- 558 in base 8 is 1056
  S = sum_base10 ∧ sum_base10 = 1056 :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_series_base8_l671_67113


namespace NUMINAMATH_GPT_painting_clock_57_painting_clock_1913_l671_67114

-- Part (a)
theorem painting_clock_57 (h : ∀ n : ℕ, (n = 12 ∨ (exists k : ℕ, n = (12 + k * 57) % 12))) :
  ∃ m : ℕ, m = 4 :=
by { sorry }

-- Part (b)
theorem painting_clock_1913 (h : ∀ n : ℕ, (n = 12 ∨ (exists k : ℕ, n = (12 + k * 1913) % 12))) :
  ∃ m : ℕ, m = 12 :=
by { sorry }

end NUMINAMATH_GPT_painting_clock_57_painting_clock_1913_l671_67114


namespace NUMINAMATH_GPT_b100_mod_50_l671_67182

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b100_mod_50 : b 100 % 50 = 2 := by
  sorry

end NUMINAMATH_GPT_b100_mod_50_l671_67182


namespace NUMINAMATH_GPT_wine_distribution_l671_67177

theorem wine_distribution (m n k s : ℕ) (h : Nat.gcd m (Nat.gcd n k) = 1) (h_s : s < m + n + k) :
  ∃ g : ℕ, g = s := by
  sorry

end NUMINAMATH_GPT_wine_distribution_l671_67177


namespace NUMINAMATH_GPT_sum_A_B_l671_67131

theorem sum_A_B (A B : ℕ) 
  (h1 : (1 / 4 : ℚ) * (1 / 8) = 1 / (4 * A))
  (h2 : 1 / (4 * A) = 1 / B) : A + B = 40 := 
by
  sorry

end NUMINAMATH_GPT_sum_A_B_l671_67131


namespace NUMINAMATH_GPT_initial_solution_amount_l671_67150

theorem initial_solution_amount (x : ℝ) (h1 : x - 200 + 1000 = 2000) : x = 1200 := by
  sorry

end NUMINAMATH_GPT_initial_solution_amount_l671_67150


namespace NUMINAMATH_GPT_students_per_class_l671_67155

theorem students_per_class
  (cards_per_student : Nat)
  (periods_per_day : Nat)
  (cost_per_pack : Nat)
  (total_spent : Nat)
  (cards_per_pack : Nat)
  (students_per_class : Nat)
  (H1 : cards_per_student = 10)
  (H2 : periods_per_day = 6)
  (H3 : cost_per_pack = 3)
  (H4 : total_spent = 108)
  (H5 : cards_per_pack = 50)
  (H6 : students_per_class = 30)
  :
  students_per_class = (total_spent / cost_per_pack * cards_per_pack / cards_per_student / periods_per_day) :=
sorry

end NUMINAMATH_GPT_students_per_class_l671_67155


namespace NUMINAMATH_GPT_problem_statement_l671_67163

theorem problem_statement (a b : ℝ) (h1 : 1 + b = 0) (h2 : a - 3 = 0) : 
  3 * (a^2 - 2 * a * b + b^2) - (4 * a^2 - 2 * (1 / 2 * a^2 + a * b - 3 / 2 * b^2)) = 12 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l671_67163


namespace NUMINAMATH_GPT_dimes_left_l671_67136

-- Definitions based on the conditions
def Initial_dimes : ℕ := 8
def Sister_borrowed : ℕ := 4
def Friend_borrowed : ℕ := 2

-- The proof problem statement (without the proof)
theorem dimes_left (Initial_dimes Sister_borrowed Friend_borrowed : ℕ) : 
  Initial_dimes = 8 → Sister_borrowed = 4 → Friend_borrowed = 2 →
  Initial_dimes - (Sister_borrowed + Friend_borrowed) = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_dimes_left_l671_67136


namespace NUMINAMATH_GPT_ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth_l671_67178

theorem ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : 0 < a * b * c)
  : a * b + b * c + c * a < (Real.sqrt (a * b * c)) / 2 + 1 / 4 := 
sorry

end NUMINAMATH_GPT_ab_plus_bc_plus_ca_lt_sqrt_abc_over_2_plus_one_fourth_l671_67178


namespace NUMINAMATH_GPT_range_of_x_l671_67183

-- Define the problem conditions and the conclusion to be proved
theorem range_of_x (f : ℝ → ℝ) (h_inc : ∀ x y, -1 ≤ x → x ≤ 1 → -1 ≤ y → y ≤ 1 → x ≤ y → f x ≤ f y)
  (h_ineq : ∀ x, f (x - 2) < f (1 - x)) :
  ∀ x, 1 ≤ x ∧ x < 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l671_67183


namespace NUMINAMATH_GPT_subset_range_a_l671_67164

def setA : Set ℝ := { x | (x^2 - 4 * x + 3) < 0 }
def setB (a : ℝ) : Set ℝ := { x | (2^(1 - x) + a) ≤ 0 ∧ (x^2 - 2*(a + 7)*x + 5) ≤ 0 }

theorem subset_range_a (a : ℝ) : setA ⊆ setB a ↔ -4 ≤ a ∧ a ≤ -1 := 
  sorry

end NUMINAMATH_GPT_subset_range_a_l671_67164


namespace NUMINAMATH_GPT_ingrid_tax_rate_l671_67161

def john_income : ℝ := 57000
def ingrid_income : ℝ := 72000
def john_tax_rate : ℝ := 0.30
def combined_tax_rate : ℝ := 0.35581395348837205

theorem ingrid_tax_rate :
  let john_tax := john_tax_rate * john_income
  let combined_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * combined_income
  let ingrid_tax := total_tax - john_tax
  let ingrid_tax_rate := ingrid_tax / ingrid_income
  ingrid_tax_rate = 0.40 :=
by
  sorry

end NUMINAMATH_GPT_ingrid_tax_rate_l671_67161


namespace NUMINAMATH_GPT_largest_integral_ratio_l671_67132

theorem largest_integral_ratio (P A : ℕ) (rel_prime_sides : ∃ (a b c : ℕ), gcd a b = 1 ∧ gcd b c = 1 ∧ gcd c a = 1 ∧ a^2 + b^2 = c^2 ∧ P = a + b + c ∧ A = a * b / 2) :
  (∃ (k : ℕ), k = 45 ∧ ∀ l, l < 45 → l ≠ (P^2 / A)) :=
sorry

end NUMINAMATH_GPT_largest_integral_ratio_l671_67132


namespace NUMINAMATH_GPT_find_k_for_circle_radius_l671_67112

theorem find_k_for_circle_radius (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ∧ (x + 7)^2 + (y + 4)^2 = 10^2) ↔ k = 35 :=
by
  sorry

end NUMINAMATH_GPT_find_k_for_circle_radius_l671_67112


namespace NUMINAMATH_GPT_greatest_large_chips_l671_67160

theorem greatest_large_chips (s l p : ℕ) (h1 : s + l = 80) (h2 : s = l + p) (hp : Nat.Prime p) : l ≤ 39 :=
by
  sorry

end NUMINAMATH_GPT_greatest_large_chips_l671_67160


namespace NUMINAMATH_GPT_price_decrease_for_original_price_l671_67146

theorem price_decrease_for_original_price (P : ℝ) (h : P > 0) :
  let new_price := 1.25 * P
  let decrease := (new_price - P) / new_price * 100
  decrease = 20 :=
by
  let new_price := 1.25 * P
  let decrease := (new_price - P) / new_price * 100
  sorry

end NUMINAMATH_GPT_price_decrease_for_original_price_l671_67146


namespace NUMINAMATH_GPT_largest_among_options_l671_67106

theorem largest_among_options :
  let A := 15679 + (1 / 3579)
  let B := 15679 - (1 / 3579)
  let C := 15679 * (1 / 3579)
  let D := 15679 / (1 / 3579)
  let E := 15679 * 1.03
  D > A ∧ D > B ∧ D > C ∧ D > E := by
{
  let A := 15679 + (1 / 3579)
  let B := 15679 - (1 / 3579)
  let C := 15679 * (1 / 3579)
  let D := 15679 / (1 / 3579)
  let E := 15679 * 1.03
  sorry
}

end NUMINAMATH_GPT_largest_among_options_l671_67106


namespace NUMINAMATH_GPT_sum_f_alpha_beta_gamma_neg_l671_67138

theorem sum_f_alpha_beta_gamma_neg (f : ℝ → ℝ)
  (h_f : ∀ x, f x = -x - x^3)
  (α β γ : ℝ)
  (h1 : α + β > 0)
  (h2 : β + γ > 0)
  (h3 : γ + α > 0) :
  f α + f β + f γ < 0 := 
sorry

end NUMINAMATH_GPT_sum_f_alpha_beta_gamma_neg_l671_67138


namespace NUMINAMATH_GPT_intersection_point_l671_67123

theorem intersection_point (k : ℚ) :
  (∃ x y : ℚ, x + k * y = 0 ∧ 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0) ↔ (k = -1/2) :=
by sorry

end NUMINAMATH_GPT_intersection_point_l671_67123


namespace NUMINAMATH_GPT_min_value_a_l671_67176

theorem min_value_a (a b : ℕ) (h1: a = b - 2005) 
  (h2: ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ p + q = a ∧ p * q = b) : a ≥ 95 := sorry

end NUMINAMATH_GPT_min_value_a_l671_67176


namespace NUMINAMATH_GPT_max_and_min_A_l671_67117

noncomputable def B := {B : ℕ // B > 22222222 ∧ gcd B 18 = 1}
noncomputable def A (B : B) : ℕ := 10^8 * ((B.val % 10)) + (B.val / 10)

noncomputable def A_max := 999999998
noncomputable def A_min := 122222224

theorem max_and_min_A : 
  (∃ B : B, A B = A_max) ∧ (∃ B : B, A B = A_min) := sorry

end NUMINAMATH_GPT_max_and_min_A_l671_67117


namespace NUMINAMATH_GPT_min_value_AF_BF_l671_67143

noncomputable def parabola_focus : ℝ × ℝ := (0, 1)

noncomputable def parabola_eq (x y : ℝ) : Prop := x^2 = 4 * y

noncomputable def line_eq (k x : ℝ) : ℝ := k * x + 1

theorem min_value_AF_BF :
  ∀ (x1 x2 y1 y2 k : ℝ),
  parabola_eq x1 y1 →
  parabola_eq x2 y2 →
  line_eq k x1 = y1 →
  line_eq k x2 = y2 →
  (x1 ≠ x2) →
  parabola_focus = (0, 1) →
  (|y1 + 2| + 1) * (|y2 + 1|) = 2 * Real.sqrt 2 + 3 := 
by
  intros
  sorry

end NUMINAMATH_GPT_min_value_AF_BF_l671_67143


namespace NUMINAMATH_GPT_chris_did_not_get_A_l671_67128

variable (A : Prop) (MC_correct : Prop) (Essay80 : Prop)

-- The condition provided by professor
axiom condition : A ↔ (MC_correct ∧ Essay80)

-- The theorem we need to prove based on the statement (B) from the solution
theorem chris_did_not_get_A 
    (h : ¬ A) : ¬ MC_correct ∨ ¬ Essay80 :=
by sorry

end NUMINAMATH_GPT_chris_did_not_get_A_l671_67128


namespace NUMINAMATH_GPT_evaluate_expression_l671_67169

theorem evaluate_expression : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + 2 * Real.sqrt 2) = (3 / 2) * (Real.sqrt 6 + Real.sqrt 2 - 0.8 * Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l671_67169


namespace NUMINAMATH_GPT_set_theorem_l671_67105

noncomputable def set_A : Set ℕ := {1, 2}
noncomputable def set_B : Set ℕ := {1, 2, 3}
noncomputable def set_C : Set ℕ := {2, 3, 4}

theorem set_theorem : (set_A ∩ set_B) ∪ set_C = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_GPT_set_theorem_l671_67105


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l671_67184

theorem simplify_and_evaluate_expr (x : Real) (h : x = Real.sqrt 3 - 1) :
  1 - (x / (x + 1)) / (x / (x ^ 2 - 1)) = 3 - Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l671_67184


namespace NUMINAMATH_GPT_find_f_10_l671_67168

variable (f : ℝ → ℝ)
variable (y : ℝ)

-- Conditions
def condition1 : Prop := ∀ x, f x = 2 * x^2 + y
def condition2 : Prop := f 2 = 30

-- Theorem to prove
theorem find_f_10 (h1 : condition1 f y) (h2 : condition2 f) : f 10 = 222 := 
sorry

end NUMINAMATH_GPT_find_f_10_l671_67168


namespace NUMINAMATH_GPT_no_periodic_sequence_first_non_zero_digit_l671_67142

/-- 
Definition of the first non-zero digit from the unit's place in the decimal representation of n! 
-/
def first_non_zero_digit (n : ℕ) : ℕ :=
  -- This function should compute the first non-zero digit from the unit's place in n!
  -- Implementation details are skipped here.
  sorry

/-- 
Prove that no natural number \( N \) exists such that the sequence \( a_{N+1}, a_{N+2}, a_{N+3}, \ldots \) 
forms a periodic sequence, where \( a_n \) is the first non-zero digit from the unit's place in the decimal 
representation of \( n! \). 
-/
theorem no_periodic_sequence_first_non_zero_digit :
  ¬ ∃ (N : ℕ), ∃ (T : ℕ), ∀ (k : ℕ), first_non_zero_digit (N + k * T) = first_non_zero_digit (N + ((k + 1) * T)) :=
by
  sorry

end NUMINAMATH_GPT_no_periodic_sequence_first_non_zero_digit_l671_67142


namespace NUMINAMATH_GPT_polygon_with_interior_sum_1260_eq_nonagon_l671_67187

theorem polygon_with_interior_sum_1260_eq_nonagon :
  ∃ n : ℕ, (n-2) * 180 = 1260 ∧ n = 9 := by
  sorry

end NUMINAMATH_GPT_polygon_with_interior_sum_1260_eq_nonagon_l671_67187


namespace NUMINAMATH_GPT_alan_needs_more_wings_l671_67167

theorem alan_needs_more_wings 
  (kevin_wings : ℕ) (kevin_time : ℕ) (alan_rate : ℕ) (target_wings : ℕ) : 
  kevin_wings = 64 → kevin_time = 8 → alan_rate = 5 → target_wings = 3 → 
  (kevin_wings / kevin_time < alan_rate + target_wings) :=
by
  intros kevin_eq time_eq rate_eq target_eq
  sorry

end NUMINAMATH_GPT_alan_needs_more_wings_l671_67167


namespace NUMINAMATH_GPT_jill_arrives_before_jack_l671_67197

theorem jill_arrives_before_jack {distance speed_jill speed_jack : ℝ} (h1 : distance = 1) 
  (h2 : speed_jill = 10) (h3 : speed_jack = 4) :
  (60 * (distance / speed_jack) - 60 * (distance / speed_jill)) = 9 :=
by
  sorry

end NUMINAMATH_GPT_jill_arrives_before_jack_l671_67197


namespace NUMINAMATH_GPT_joe_initial_tests_l671_67198

theorem joe_initial_tests (S n : ℕ) (h1 : S = 60 * n) (h2 : (S - 45) = 65 * (n - 1)) : n = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_joe_initial_tests_l671_67198


namespace NUMINAMATH_GPT_value_of_b_plus_c_l671_67109

variable {a b c d : ℝ}

theorem value_of_b_plus_c (h1 : a + b = 4) (h2 : c + d = 5) (h3 : a + d = 2) : b + c = 7 :=
sorry

end NUMINAMATH_GPT_value_of_b_plus_c_l671_67109


namespace NUMINAMATH_GPT_quadratic_expression_value_l671_67144

variable (x y : ℝ)

theorem quadratic_expression_value (h1 : 3 * x + y = 6) (h2 : x + 3 * y = 8) :
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 100 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_expression_value_l671_67144


namespace NUMINAMATH_GPT_ratio_of_potatoes_l671_67193

-- Definitions as per conditions
def initial_potatoes : ℕ := 300
def given_to_gina : ℕ := 69
def remaining_potatoes : ℕ := 47
def k : ℕ := 2  -- Identify k is 2 based on the ratio

-- Calculate given_to_tom and total given away
def given_to_tom : ℕ := k * given_to_gina
def given_to_anne : ℕ := given_to_tom / 3

-- Arithmetical conditions derived from the problem
def total_given_away : ℕ := given_to_gina + given_to_tom + given_to_anne + remaining_potatoes

-- Proof statement to show the ratio between given_to_tom and given_to_gina is 2
theorem ratio_of_potatoes :
  k = 2 → total_given_away = initial_potatoes → given_to_tom / given_to_gina = 2 := by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_ratio_of_potatoes_l671_67193


namespace NUMINAMATH_GPT_rug_inner_rectangle_length_l671_67189

theorem rug_inner_rectangle_length
  (width : ℕ)
  (shaded1_width : ℕ)
  (shaded2_width : ℕ)
  (areas_in_ap : ℕ → ℕ → ℕ → Prop)
  (h1 : width = 2)
  (h2 : shaded1_width = 2)
  (h3 : shaded2_width = 2)
  (h4 : ∀ y a1 a2 a3, 
        a1 = 2 * y →
        a2 = 6 * (y + 4) →
        a3 = 10 * (y + 8) →
        areas_in_ap a1 (a2 - a1) (a3 - a2) →
        (a2 - a1 = a3 - a2)) :
  ∃ y, y = 4 :=
by
  sorry

end NUMINAMATH_GPT_rug_inner_rectangle_length_l671_67189


namespace NUMINAMATH_GPT_defective_probability_l671_67185

variable (total_products defective_products qualified_products : ℕ)
variable (first_draw_defective second_draw_defective : Prop)

-- Definitions of the problem
def total_prods := 10
def def_prods := 4
def qual_prods := 6
def p_A := def_prods / total_prods
def p_AB := (def_prods / total_prods) * ((def_prods - 1) / (total_prods - 1))
def p_B_given_A := p_AB / p_A

-- Theorem: The probability of drawing a defective product on the second draw given the first was defective is 1/3.
theorem defective_probability 
  (hp1 : total_products = total_prods)
  (hp2 : defective_products = def_prods)
  (hp3 : qualified_products = qual_prods)
  (pA_eq : p_A = 2 / 5)
  (pAB_eq : p_AB = 2 / 15) : 
  p_B_given_A = 1 / 3 := sorry

end NUMINAMATH_GPT_defective_probability_l671_67185


namespace NUMINAMATH_GPT_point_not_similar_inflection_point_ln_l671_67170

noncomputable def similar_inflection_point (C : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
∃ (m : ℝ → ℝ), (∀ x, m x = (deriv C P.1) * (x - P.1) + P.2) ∧
  ∃ ε > 0, ∀ h : ℝ, |h| < ε → (C (P.1 + h) > m (P.1 + h) ∧ C (P.1 - h) < m (P.1 - h)) ∨ 
                     (C (P.1 + h) < m (P.1 + h) ∧ C (P.1 - h) > m (P.1 - h))

theorem point_not_similar_inflection_point_ln :
  ¬ similar_inflection_point (fun x => Real.log x) (1, 0) :=
sorry

end NUMINAMATH_GPT_point_not_similar_inflection_point_ln_l671_67170


namespace NUMINAMATH_GPT_probability_same_color_l671_67166

theorem probability_same_color :
  let bagA_white := 8
  let bagA_red := 4
  let bagB_white := 6
  let bagB_red := 6
  let totalA := bagA_white + bagA_red
  let totalB := bagB_white + bagB_red
  let prob_white_white := (bagA_white / totalA) * (bagB_white / totalB)
  let prob_red_red := (bagA_red / totalA) * (bagB_red / totalB)
  let total_prob := prob_white_white + prob_red_red
  total_prob = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_probability_same_color_l671_67166


namespace NUMINAMATH_GPT_sum_of_tangents_slopes_at_vertices_l671_67190

noncomputable def curve (x : ℝ) := (x + 3) * (x ^ 2 + 3)

theorem sum_of_tangents_slopes_at_vertices {x_A x_B x_C : ℝ}
  (h1 : curve x_A = x_A * (x_A ^ 2 + 6 * x_A + 9) + 3)
  (h2 : curve x_B = x_B * (x_B ^ 2 + 6 * x_B + 9) + 3)
  (h3 : curve x_C = x_C * (x_C ^ 2 + 6 * x_C + 9) + 3)
  : (3 * x_A ^ 2 + 6 * x_A + 3) + (3 * x_B ^ 2 + 6 * x_B + 3) + (3 * x_C ^ 2 + 6 * x_C + 3) = 237 :=
sorry

end NUMINAMATH_GPT_sum_of_tangents_slopes_at_vertices_l671_67190


namespace NUMINAMATH_GPT_range_of_h_l671_67108

noncomputable def h : ℝ → ℝ
| x => if x = -7 then 0 else 2 * (x - 3)

theorem range_of_h :
  (Set.range h) = Set.univ \ {-20} :=
sorry

end NUMINAMATH_GPT_range_of_h_l671_67108


namespace NUMINAMATH_GPT_sin_double_alpha_l671_67107

variable (α β : ℝ)

theorem sin_double_alpha (h1 : Real.pi / 2 < β ∧ β < α ∧ α < 3 * Real.pi / 4)
        (h2 : Real.cos (α - β) = 12 / 13) 
        (h3 : Real.sin (α + β) = -3 / 5) : 
        Real.sin (2 * α) = -56 / 65 := by
  sorry

end NUMINAMATH_GPT_sin_double_alpha_l671_67107


namespace NUMINAMATH_GPT_boys_meet_once_excluding_start_finish_l671_67171

theorem boys_meet_once_excluding_start_finish 
    (d : ℕ) 
    (h1 : 0 < d) 
    (boy1_speed : ℕ) (boy2_speed : ℕ) 
    (h2 : boy1_speed = 6) (h3 : boy2_speed = 10)
    (relative_speed : ℕ) (h4 : relative_speed = boy1_speed + boy2_speed) 
    (time_to_meet_A_again : ℕ) (h5 : time_to_meet_A_again = d / relative_speed) 
    (boy1_laps_per_sec boy2_laps_per_sec : ℕ) 
    (h6 : boy1_laps_per_sec = boy1_speed / d) 
    (h7 : boy2_laps_per_sec = boy2_speed / d)
    (lcm_laps : ℕ) (h8 : lcm_laps = Nat.lcm 6 10)
    (meetings_per_lap : ℕ) (h9 : meetings_per_lap = lcm_laps / d)
    (total_meetings : ℕ) (h10 : total_meetings = meetings_per_lap * time_to_meet_A_again)
  : total_meetings = 1 := by
  sorry

end NUMINAMATH_GPT_boys_meet_once_excluding_start_finish_l671_67171


namespace NUMINAMATH_GPT_geometric_seq_a9_l671_67130

theorem geometric_seq_a9 
  (a : ℕ → ℤ)  -- The sequence definition
  (h_geometric : ∀ n : ℕ, a (n+1) = a 1 * (a 2 ^ n) / a 1 ^ n)  -- Geometric sequence property
  (h_a1 : a 1 = 2)  -- Given a₁ = 2
  (h_a5 : a 5 = 18)  -- Given a₅ = 18
: a 9 = 162 := sorry

end NUMINAMATH_GPT_geometric_seq_a9_l671_67130


namespace NUMINAMATH_GPT_cricketer_average_increase_l671_67122

theorem cricketer_average_increase (A : ℝ) (H1 : 18 * A + 98 = 19 * 26) :
  26 - A = 4 :=
by
  sorry

end NUMINAMATH_GPT_cricketer_average_increase_l671_67122


namespace NUMINAMATH_GPT_radio_loss_percentage_l671_67111

theorem radio_loss_percentage (cost_price selling_price : ℕ) (h1 : cost_price = 1500) (h2 : selling_price = 1305) : 
  (cost_price - selling_price) * 100 / cost_price = 13 := by
  sorry

end NUMINAMATH_GPT_radio_loss_percentage_l671_67111


namespace NUMINAMATH_GPT_total_raining_time_correct_l671_67102

-- Define individual durations based on given conditions
def duration_day1 : ℕ := 10        -- 17:00 - 07:00 = 10 hours
def duration_day2 : ℕ := duration_day1 + 2    -- Second day: 10 hours + 2 hours = 12 hours
def duration_day3 : ℕ := duration_day2 * 2    -- Third day: 12 hours * 2 = 24 hours

-- Define the total raining time over three days
def total_raining_time : ℕ := duration_day1 + duration_day2 + duration_day3

-- Formally state the theorem to prove the total rain time is 46 hours
theorem total_raining_time_correct : total_raining_time = 46 := by
  sorry

end NUMINAMATH_GPT_total_raining_time_correct_l671_67102


namespace NUMINAMATH_GPT_height_of_middle_brother_l671_67199

theorem height_of_middle_brother (h₁ h₂ h₃ : ℝ) (h₁_le_h₂ : h₁ ≤ h₂) (h₂_le_h₃ : h₂ ≤ h₃)
  (avg_height : (h₁ + h₂ + h₃) / 3 = 1.74) (avg_height_tallest_shortest : (h₁ + h₃) / 2 = 1.75) :
  h₂ = 1.72 :=
by
  -- Proof to be filled here
  sorry

end NUMINAMATH_GPT_height_of_middle_brother_l671_67199


namespace NUMINAMATH_GPT_shaded_area_l671_67137

theorem shaded_area (area_large : ℝ) (area_small : ℝ) (n_small_squares : ℕ) 
  (n_triangles: ℕ) (area_total : ℝ) : 
  area_large = 16 → 
  area_small = 1 → 
  n_small_squares = 4 → 
  n_triangles = 4 → 
  area_total = 4 → 
  4 * area_small = 4 →
  area_large - (area_total + (n_small_squares * area_small)) = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_shaded_area_l671_67137


namespace NUMINAMATH_GPT_overall_percent_change_l671_67104

theorem overall_percent_change (W : ℝ) : 
  (W * 0.6 * 1.3 * 0.8 * 1.1) / W = 0.624 :=
by {
  sorry
}

end NUMINAMATH_GPT_overall_percent_change_l671_67104


namespace NUMINAMATH_GPT_harry_annual_pet_feeding_cost_l671_67181

def monthly_cost_snake := 10
def monthly_cost_iguana := 5
def monthly_cost_gecko := 15
def num_snakes := 4
def num_iguanas := 2
def num_geckos := 3
def months_in_year := 12

theorem harry_annual_pet_feeding_cost :
  (num_snakes * monthly_cost_snake + 
   num_iguanas * monthly_cost_iguana + 
   num_geckos * monthly_cost_gecko) * 
   months_in_year = 1140 := 
sorry

end NUMINAMATH_GPT_harry_annual_pet_feeding_cost_l671_67181


namespace NUMINAMATH_GPT_find_unique_positive_integer_pair_l671_67139

theorem find_unique_positive_integer_pair :
  ∃! (b c : ℕ), b > 0 ∧ c > 0 ∧ c > b^2 ∧ b > c^2 :=
sorry

end NUMINAMATH_GPT_find_unique_positive_integer_pair_l671_67139


namespace NUMINAMATH_GPT_intersection_points_on_circle_l671_67133

theorem intersection_points_on_circle
  (x y : ℝ)
  (h1 : y = (x + 2)^2)
  (h2 : x + 2 = (y - 1)^2) :
  (x + 2)^2 + (y - 1)^2 = 2 :=
sorry

end NUMINAMATH_GPT_intersection_points_on_circle_l671_67133


namespace NUMINAMATH_GPT_quadratic_unique_real_root_l671_67157

theorem quadratic_unique_real_root (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ∃! r : ℝ, x = r) → m = 2/9 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_unique_real_root_l671_67157


namespace NUMINAMATH_GPT_first_player_winning_strategy_l671_67147

noncomputable def optimal_first_move : ℕ := 45

-- Prove that with 300 matches initially and following the game rules,
-- taking 45 matches on the first turn leaves the opponent in a losing position.

theorem first_player_winning_strategy (n : ℕ) (h₀ : n = 300) :
    ∃ m : ℕ, (m ≤ n / 2 ∧ n - m = 255) :=
by
  exists optimal_first_move
  sorry

end NUMINAMATH_GPT_first_player_winning_strategy_l671_67147


namespace NUMINAMATH_GPT_find_formula_l671_67151

variable (x : ℕ) (y : ℕ)

theorem find_formula (h1: (x = 2 ∧ y = 10) ∨ (x = 3 ∧ y = 21) ∨ (x = 4 ∧ y = 38) ∨ (x = 5 ∧ y = 61) ∨ (x = 6 ∧ y = 90)) :
  y = 3 * x^2 - 2 * x + 2 :=
  sorry

end NUMINAMATH_GPT_find_formula_l671_67151


namespace NUMINAMATH_GPT_hyperbola_center_l671_67115

theorem hyperbola_center :
  ∃ (c : ℝ × ℝ), c = (3, 5) ∧
  (9 * (x - c.1)^2 - 36 * (y - c.2)^2 - (1244 - 243 - 1001) = 0) :=
sorry

end NUMINAMATH_GPT_hyperbola_center_l671_67115


namespace NUMINAMATH_GPT_lake_width_l671_67118

theorem lake_width
  (W : ℝ)
  (janet_speed : ℝ) (sister_speed : ℝ) (wait_time : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : wait_time = 3)
  (h4 : W / sister_speed = W / janet_speed + wait_time) :
  W = 60 := 
sorry

end NUMINAMATH_GPT_lake_width_l671_67118


namespace NUMINAMATH_GPT_rectangle_area_l671_67192

theorem rectangle_area (a b k : ℕ)
  (h1 : k = 6 * (a + b) + 36)
  (h2 : k = 114)
  (h3 : a / b = 8 / 5) :
  a * b = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_area_l671_67192


namespace NUMINAMATH_GPT_battery_life_in_standby_l671_67124

noncomputable def remaining_battery_life (b_s : ℝ) (b_a : ℝ) (t_total : ℝ) (t_active : ℝ) : ℝ :=
  let standby_rate := 1 / b_s
  let active_rate := 1 / b_a
  let standby_time := t_total - t_active
  let consumption_active := t_active * active_rate
  let consumption_standby := standby_time * standby_rate
  let total_consumption := consumption_active + consumption_standby
  let remaining_battery := 1 - total_consumption
  remaining_battery * b_s

theorem battery_life_in_standby :
  remaining_battery_life 30 4 10 1.5 = 10.25 := sorry

end NUMINAMATH_GPT_battery_life_in_standby_l671_67124


namespace NUMINAMATH_GPT_avg_speed_4_2_l671_67116

noncomputable def avg_speed_round_trip (D : ℝ) : ℝ :=
  let speed_up := 3
  let speed_down := 7
  let total_distance := 2 * D
  let total_time := D / speed_up + D / speed_down
  total_distance / total_time

theorem avg_speed_4_2 (D : ℝ) (hD : D > 0) : avg_speed_round_trip D = 4.2 := by
  sorry

end NUMINAMATH_GPT_avg_speed_4_2_l671_67116


namespace NUMINAMATH_GPT_sum_of_modified_numbers_l671_67129

theorem sum_of_modified_numbers (x y R : ℝ) (h : x + y = R) : 
  2 * (x + 4) + 2 * (y + 5) = 2 * R + 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_modified_numbers_l671_67129


namespace NUMINAMATH_GPT_roots_equation_l671_67149

theorem roots_equation (m n : ℝ) (h1 : ∀ x, (x - m) * (x - n) = x^2 + 2 * x - 2025) : m^2 + 3 * m + n = 2023 :=
by
  sorry

end NUMINAMATH_GPT_roots_equation_l671_67149


namespace NUMINAMATH_GPT_find_prism_height_l671_67175

variables (base_side_length : ℝ) (density : ℝ) (weight : ℝ) (height : ℝ)

-- Assume the base_side_length is 2 meters, density is 2700 kg/m³, and weight is 86400 kg
def given_conditions := (base_side_length = 2) ∧ (density = 2700) ∧ (weight = 86400)

-- Define the volume based on weight and density
noncomputable def volume (density weight : ℝ) : ℝ := weight / density

-- Define the area of the base
def base_area (side_length : ℝ) : ℝ := side_length * side_length

-- Define the height of the prism
noncomputable def prism_height (volume base_area : ℝ) : ℝ := volume / base_area

-- The proof statement
theorem find_prism_height (h : ℝ) : given_conditions base_side_length density weight → prism_height (volume density weight) (base_area base_side_length) = h :=
by
  intros h_cond
  sorry

end NUMINAMATH_GPT_find_prism_height_l671_67175


namespace NUMINAMATH_GPT_unique_n_for_given_divisors_l671_67188

theorem unique_n_for_given_divisors :
  ∃! (n : ℕ), 
    ∀ (k : ℕ) (d : ℕ → ℕ), 
      k ≥ 22 ∧ 
      d 1 = 1 ∧ d k = n ∧ 
      (∀ i j, i < j → d i < d j) ∧ 
      (d 7) ^ 2 + (d 10) ^ 2 = (n / d 22) ^ 2 →
      n = 2^3 * 3 * 5 * 17 :=
sorry

end NUMINAMATH_GPT_unique_n_for_given_divisors_l671_67188


namespace NUMINAMATH_GPT_percent_increase_equilateral_triangles_l671_67126

theorem percent_increase_equilateral_triangles :
  let s₁ := 3
  let s₂ := 2 * s₁
  let s₃ := 2 * s₂
  let s₄ := 2 * s₃
  let P₁ := 3 * s₁
  let P₄ := 3 * s₄
  (P₄ - P₁) / P₁ * 100 = 700 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_equilateral_triangles_l671_67126


namespace NUMINAMATH_GPT_find_a_if_f_even_l671_67100

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end NUMINAMATH_GPT_find_a_if_f_even_l671_67100


namespace NUMINAMATH_GPT_cos_double_angle_example_l671_67179

theorem cos_double_angle_example (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (2 * θ) = -7 / 9 := by
  sorry

end NUMINAMATH_GPT_cos_double_angle_example_l671_67179


namespace NUMINAMATH_GPT_factorize_polynomial_l671_67194

theorem factorize_polynomial (a b c : ℚ) : 
  b^2 - c^2 + a * (a + 2 * b) = (a + b + c) * (a + b - c) :=
by
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l671_67194


namespace NUMINAMATH_GPT_moles_of_H2O_combined_l671_67186

theorem moles_of_H2O_combined (mole_NH4Cl mole_NH4OH : ℕ) (reaction : mole_NH4Cl = 1 ∧ mole_NH4OH = 1) : 
  ∃ mole_H2O : ℕ, mole_H2O = 1 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_H2O_combined_l671_67186


namespace NUMINAMATH_GPT_eliminate_denominator_correctness_l671_67101

-- Define the initial equality with fractions
def initial_equation (x : ℝ) := (2 * x - 3) / 5 = (2 * x) / 3 - 3

-- Define the resulting expression after eliminating the denominators
def eliminated_denominators (x : ℝ) := 3 * (2 * x - 3) = 5 * 2 * x - 3 * 15

-- The theorem states that given the initial equation, the eliminated denomination expression holds true
theorem eliminate_denominator_correctness (x : ℝ) :
  initial_equation x → eliminated_denominators x := by
  sorry

end NUMINAMATH_GPT_eliminate_denominator_correctness_l671_67101


namespace NUMINAMATH_GPT_johns_number_l671_67153

theorem johns_number (n : ℕ) 
  (h1 : 125 ∣ n) 
  (h2 : 30 ∣ n) 
  (h3 : 800 ≤ n ∧ n ≤ 2000) : 
  n = 1500 :=
sorry

end NUMINAMATH_GPT_johns_number_l671_67153


namespace NUMINAMATH_GPT_xy_square_sum_l671_67154

theorem xy_square_sum (x y : ℝ) (h1 : (x - y)^2 = 49) (h2 : x * y = 8) : x^2 + y^2 = 65 :=
by
  sorry

end NUMINAMATH_GPT_xy_square_sum_l671_67154


namespace NUMINAMATH_GPT_fraction_simplification_l671_67156

theorem fraction_simplification :
  (20 / 21) * (35 / 54) * (63 / 50) = (7 / 9) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l671_67156


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l671_67180

-- Definitions based on problem conditions
def total_people := 12
def choices := 5
def special_people_count := 3

noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Proof problem 1: A, B, and C must be chosen, so select 2 more from the remaining 9 people
theorem problem_1 : choose 9 2 = 36 :=
by sorry

-- Proof problem 2: Only one among A, B, and C is chosen, so select 4 more from the remaining 9 people
theorem problem_2 : choose 3 1 * choose 9 4 = 378 :=
by sorry

-- Proof problem 3: At most two among A, B, and C are chosen
theorem problem_3 : choose 12 5 - choose 9 2 = 756 :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l671_67180


namespace NUMINAMATH_GPT_multiply_inequalities_positive_multiply_inequalities_negative_l671_67173

variable {a b c d : ℝ}

theorem multiply_inequalities_positive (h₁ : a > b) (h₂ : c > d) (h₃ : 0 < a) (h₄ : 0 < b) (h₅ : 0 < c) (h₆ : 0 < d) :
  a * c > b * d :=
sorry

theorem multiply_inequalities_negative (h₁ : a < b) (h₂ : c < d) (h₃ : a < 0) (h₄ : b < 0) (h₅ : c < 0) (h₆ : d < 0) :
  a * c > b * d :=
sorry

end NUMINAMATH_GPT_multiply_inequalities_positive_multiply_inequalities_negative_l671_67173


namespace NUMINAMATH_GPT_jose_share_is_correct_l671_67125

noncomputable def total_profit : ℝ := 
  5000 - 2000 + 7000 + 1000 - 3000 + 10000 + 500 + 4000 - 2500 + 6000 + 8000 - 1000

noncomputable def tom_investment_ratio : ℝ := 30000 * 12
noncomputable def jose_investment_ratio : ℝ := 45000 * 10
noncomputable def maria_investment_ratio : ℝ := 60000 * 8

noncomputable def total_investment_ratio : ℝ := tom_investment_ratio + jose_investment_ratio + maria_investment_ratio

noncomputable def jose_share : ℝ := (jose_investment_ratio / total_investment_ratio) * total_profit

theorem jose_share_is_correct : jose_share = 14658 := 
by 
  sorry

end NUMINAMATH_GPT_jose_share_is_correct_l671_67125


namespace NUMINAMATH_GPT_probability_two_same_number_l671_67110

theorem probability_two_same_number :
  let rolls := 5
  let sides := 8
  let total_outcomes := sides ^ rolls
  let favorable_outcomes := 8 * 7 * 6 * 5 * 4
  let probability_all_different := (favorable_outcomes : ℚ) / total_outcomes
  let probability_at_least_two_same := 1 - probability_all_different
  probability_at_least_two_same = (3256 : ℚ) / 4096 :=
by 
  sorry

end NUMINAMATH_GPT_probability_two_same_number_l671_67110


namespace NUMINAMATH_GPT_smallest_n_l671_67134

theorem smallest_n (n : ℕ) (h1 : n > 2016) (h2 : (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0) : n = 2020 :=
sorry

end NUMINAMATH_GPT_smallest_n_l671_67134


namespace NUMINAMATH_GPT_meal_combinations_l671_67148

theorem meal_combinations :
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let vegetable_combinations := Nat.choose vegetables 3
  meats * vegetable_combinations * desserts = 150 :=
by
  let meats := 3
  let vegetables := 5
  let desserts := 5
  let vegetable_combinations := Nat.choose vegetables 3
  show meats * vegetable_combinations * desserts = 150
  sorry

end NUMINAMATH_GPT_meal_combinations_l671_67148


namespace NUMINAMATH_GPT_total_students_l671_67135

theorem total_students (N : ℕ)
  (h1 : (84 + 128 + 13 = 15 * N))
  : N = 15 :=
sorry

end NUMINAMATH_GPT_total_students_l671_67135


namespace NUMINAMATH_GPT_seventh_observation_l671_67162

-- Declare the conditions with their definitions
def average_of_six (sum6 : ℕ) : Prop := sum6 = 6 * 14
def new_average_decreased (sum6 sum7 : ℕ) : Prop := sum7 = sum6 + 7 ∧ 13 = (sum6 + 7) / 7

-- The main statement to prove that the seventh observation is 7
theorem seventh_observation (sum6 sum7 : ℕ) (h_avg6 : average_of_six sum6) (h_new_avg : new_average_decreased sum6 sum7) :
  sum7 - sum6 = 7 := 
  sorry

end NUMINAMATH_GPT_seventh_observation_l671_67162


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l671_67119

variable (p q : Prop)

theorem sufficient_but_not_necessary : (¬p → ¬(p ∧ q)) ∧ (¬(¬p) → ¬(p ∧ q) → False) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_l671_67119


namespace NUMINAMATH_GPT_min_steps_for_humpty_l671_67195

theorem min_steps_for_humpty (x y : ℕ) (H : 47 * x - 37 * y = 1) : x + y = 59 :=
  sorry

end NUMINAMATH_GPT_min_steps_for_humpty_l671_67195


namespace NUMINAMATH_GPT_max_t_eq_one_l671_67191

theorem max_t_eq_one {x y : ℝ} (hx : x > 0) (hy : y > 0) : 
  max (min x (y / (x^2 + y^2))) 1 = 1 :=
sorry

end NUMINAMATH_GPT_max_t_eq_one_l671_67191


namespace NUMINAMATH_GPT_range_of_a_l671_67159

theorem range_of_a (A M : ℝ × ℝ) (a : ℝ) (C : ℝ × ℝ → ℝ) (hA : A = (-3, 0)) 
(hM : C M = 1) (hMA : dist M A = 2 * dist M (0, 0)) :
  a ∈ (Set.Icc (1/2 : ℝ) (3/2) ∪ Set.Icc (-3/2) (-1/2)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l671_67159


namespace NUMINAMATH_GPT_total_preparation_time_l671_67145

theorem total_preparation_time
    (minutes_per_game : ℕ)
    (number_of_games : ℕ)
    (h1 : minutes_per_game = 10)
    (h2 : number_of_games = 15) :
    minutes_per_game * number_of_games = 150 :=
by
  -- Lean 4 proof goes here
  sorry

end NUMINAMATH_GPT_total_preparation_time_l671_67145


namespace NUMINAMATH_GPT_soccer_team_wins_l671_67158

theorem soccer_team_wins 
  (total_matches : ℕ)
  (total_points : ℕ)
  (points_per_win : ℕ)
  (points_per_draw : ℕ)
  (points_per_loss : ℕ)
  (losses : ℕ)
  (H1 : total_matches = 10)
  (H2 : total_points = 17)
  (H3 : points_per_win = 3)
  (H4 : points_per_draw = 1)
  (H5 : points_per_loss = 0)
  (H6 : losses = 3) : 
  ∃ (wins : ℕ), wins = 5 := 
by
  sorry

end NUMINAMATH_GPT_soccer_team_wins_l671_67158


namespace NUMINAMATH_GPT_max_value_HMMT_l671_67174

theorem max_value_HMMT :
  ∀ (H M T : ℤ), H * M ^ 2 * T = H + 2 * M + T → H * M ^ 2 * T ≤ 8 :=
by
  sorry

end NUMINAMATH_GPT_max_value_HMMT_l671_67174


namespace NUMINAMATH_GPT_yellow_pill_cost_22_5_l671_67120

-- Definitions based on conditions
def number_of_days := 3 * 7
def total_cost := 903
def daily_cost := total_cost / number_of_days
def blue_pill_cost (yellow_pill_cost : ℝ) := yellow_pill_cost - 2

-- Prove that the cost of one yellow pill is 22.5 dollars
theorem yellow_pill_cost_22_5 : 
  ∃ (yellow_pill_cost : ℝ), 
    number_of_days = 21 ∧
    total_cost = 903 ∧ 
    (∀ yellow_pill_cost, daily_cost = yellow_pill_cost + blue_pill_cost yellow_pill_cost → yellow_pill_cost = 22.5) :=
by 
  sorry

end NUMINAMATH_GPT_yellow_pill_cost_22_5_l671_67120


namespace NUMINAMATH_GPT_bushes_for_zucchinis_l671_67165

def bushes_yield := 10 -- containers per bush
def container_to_zucchini := 3 -- containers per zucchini
def zucchinis_required := 60 -- total zucchinis needed

theorem bushes_for_zucchinis (hyld : bushes_yield = 10) (ctz : container_to_zucchini = 3) (zreq : zucchinis_required = 60) :
  ∃ bushes : ℕ, bushes = 60 * container_to_zucchini / bushes_yield :=
sorry

end NUMINAMATH_GPT_bushes_for_zucchinis_l671_67165


namespace NUMINAMATH_GPT_hyperbola_standard_equation_l671_67152

theorem hyperbola_standard_equation (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h_real_axis : 2 * a = 4 * Real.sqrt 2) (h_eccentricity : a / Real.sqrt (a^2 + b^2) = Real.sqrt 6 / 2) :
    (a = 2 * Real.sqrt 2) ∧ (b = 2) → ∀ x y : ℝ, (x^2)/8 - (y^2)/4 = 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_l671_67152


namespace NUMINAMATH_GPT_probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5_l671_67196

open BigOperators

/-- Suppose 30 balls are tossed independently and at random into one 
of the 6 bins. Let p be the probability that one bin ends up with 3 
balls, another with 6 balls, another with 5, another with 4, another 
with 2, and the last one with 10 balls. Let q be the probability 
that each bin ends up with 5 balls. Calculate p / q. 
-/
theorem probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5 :
  (Nat.factorial 5 ^ 6 : ℚ) / ((Nat.factorial 3:ℚ) * Nat.factorial 6 * Nat.factorial 5 * Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 10) = 0.125 := 
sorry

end NUMINAMATH_GPT_probability_ratio_3_6_5_4_2_10_vs_5_5_5_5_5_5_l671_67196


namespace NUMINAMATH_GPT_gross_profit_percentage_l671_67121

theorem gross_profit_percentage :
  ∀ (selling_price wholesale_cost : ℝ),
  selling_price = 28 →
  wholesale_cost = 24.14 →
  (selling_price - wholesale_cost) / wholesale_cost * 100 = 15.99 :=
by
  intros selling_price wholesale_cost h1 h2
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_gross_profit_percentage_l671_67121


namespace NUMINAMATH_GPT_find_F_l671_67103

theorem find_F (F C : ℝ) (hC_eq : C = (4/7) * (F - 40)) (hC_val : C = 35) : F = 101.25 :=
by
  sorry

end NUMINAMATH_GPT_find_F_l671_67103


namespace NUMINAMATH_GPT_find_LN_l671_67127

noncomputable def LM : ℝ := 25
noncomputable def sinN : ℝ := 4 / 5

theorem find_LN (LN : ℝ) (h_sin : sinN = LM / LN) : LN = 125 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_LN_l671_67127


namespace NUMINAMATH_GPT_problem_statement_l671_67141

noncomputable def r (C: ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def A (r: ℝ) : ℝ := Real.pi * r^2

noncomputable def combined_area_difference (C1 C2 C3: ℝ) : ℝ :=
  let r1 := r C1
  let r2 := r C2
  let r3 := r C3
  let A1 := A r1
  let A2 := A r2
  let A3 := A r3
  (A3 - A1) - A2

theorem problem_statement : combined_area_difference 528 704 880 = -9.76 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l671_67141
