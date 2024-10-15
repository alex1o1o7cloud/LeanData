import Mathlib

namespace NUMINAMATH_GPT_intersection_A_B_l1567_156736

noncomputable def A : Set ℝ := { x | (x - 1) / (x + 3) < 0 }
noncomputable def B : Set ℝ := { x | abs x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1567_156736


namespace NUMINAMATH_GPT_period_started_at_7_am_l1567_156747

-- Define the end time of the period
def end_time : ℕ := 16 -- 4 pm in 24-hour format

-- Define the total duration in hours
def duration : ℕ := 9

-- Define the start time of the period
def start_time : ℕ := end_time - duration

-- Prove that the start time is 7 am
theorem period_started_at_7_am : start_time = 7 := by
  sorry

end NUMINAMATH_GPT_period_started_at_7_am_l1567_156747


namespace NUMINAMATH_GPT_tan_beta_eq_neg13_l1567_156750

variables (α β : Real)

theorem tan_beta_eq_neg13 (h1 : Real.tan α = 2) (h2 : Real.tan (α - β) = -3/5) : 
  Real.tan β = -13 := 
by 
  sorry

end NUMINAMATH_GPT_tan_beta_eq_neg13_l1567_156750


namespace NUMINAMATH_GPT_value_of_a_l1567_156755

theorem value_of_a (x : ℝ) (h : (1 - x^32) ≠ 0):
  (8 * a / (1 - x^32) = 
   2 / (1 - x) + 2 / (1 + x) + 
   4 / (1 + x^2) + 8 / (1 + x^4) + 
   16 / (1 + x^8) + 32 / (1 + x^16)) → 
  a = 8 := sorry

end NUMINAMATH_GPT_value_of_a_l1567_156755


namespace NUMINAMATH_GPT_mango_rate_is_50_l1567_156711

theorem mango_rate_is_50 (quantity_grapes kg_grapes_perkg quantity_mangoes total_paid cost_grapes cost_mangoes rate_mangoes : ℕ) 
  (h1 : quantity_grapes = 8) 
  (h2 : kg_grapes_perkg = 70) 
  (h3 : quantity_mangoes = 9) 
  (h4 : total_paid = 1010)
  (h5 : cost_grapes = quantity_grapes * kg_grapes_perkg)
  (h6 : cost_mangoes = total_paid - cost_grapes)
  (h7 : rate_mangoes = cost_mangoes / quantity_mangoes) : 
  rate_mangoes = 50 :=
by sorry

end NUMINAMATH_GPT_mango_rate_is_50_l1567_156711


namespace NUMINAMATH_GPT_factorization_of_cubic_polynomial_l1567_156776

-- Define the elements and the problem
variable (a : ℝ)

theorem factorization_of_cubic_polynomial :
  a^3 - 3 * a = a * (a + Real.sqrt 3) * (a - Real.sqrt 3) := by
  sorry

end NUMINAMATH_GPT_factorization_of_cubic_polynomial_l1567_156776


namespace NUMINAMATH_GPT_sweeties_remainder_l1567_156748

theorem sweeties_remainder (m : ℕ) (h : m % 6 = 4) : (2 * m) % 6 = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_sweeties_remainder_l1567_156748


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1567_156761

variable (x y : ℝ)

theorem simplify_and_evaluate_expression
  (hx : x = 2)
  (hy : y = -0.5) :
  2 * (2 * x - 3 * y) - (3 * x + 2 * y + 1) = 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1567_156761


namespace NUMINAMATH_GPT_B_time_l1567_156735

-- Define the work rates of A, B, and C in terms of how long they take to complete the work
variable (A B C : ℝ)

-- Conditions provided in the problem
axiom A_rate : A = 1 / 3
axiom BC_rate : B + C = 1 / 3
axiom AC_rate : A + C = 1 / 2

-- Prove that B alone will take 6 hours to complete the work
theorem B_time : B = 1 / 6 → (1 / B) = 6 := by
  intro hB
  sorry

end NUMINAMATH_GPT_B_time_l1567_156735


namespace NUMINAMATH_GPT_income_calculation_l1567_156706

theorem income_calculation (savings : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (ratio_condition : income_ratio = 5 ∧ expenditure_ratio = 4) (savings_condition : savings = 3800) :
  income_ratio * savings / (income_ratio - expenditure_ratio) = 19000 :=
by
  sorry

end NUMINAMATH_GPT_income_calculation_l1567_156706


namespace NUMINAMATH_GPT_find_g_of_nine_l1567_156773

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_nine (h : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = x) : g 9 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_g_of_nine_l1567_156773


namespace NUMINAMATH_GPT_woman_waits_for_man_l1567_156772

noncomputable def man_speed := 5 / 60 -- miles per minute
noncomputable def woman_speed := 15 / 60 -- miles per minute
noncomputable def passed_time := 2 -- minutes

noncomputable def catch_up_time (man_speed woman_speed : ℝ) (passed_time : ℝ) : ℝ :=
  (woman_speed * passed_time) / man_speed

theorem woman_waits_for_man
  (man_speed woman_speed : ℝ)
  (passed_time : ℝ)
  (h_man_speed : man_speed = 5 / 60)
  (h_woman_speed : woman_speed = 15 / 60)
  (h_passed_time : passed_time = 2) :
  catch_up_time man_speed woman_speed passed_time = 6 := 
by
  -- actual proof skipped
  sorry

end NUMINAMATH_GPT_woman_waits_for_man_l1567_156772


namespace NUMINAMATH_GPT_angle_measure_l1567_156797

theorem angle_measure (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : P = 206 :=
by
  sorry

end NUMINAMATH_GPT_angle_measure_l1567_156797


namespace NUMINAMATH_GPT_hours_per_shift_l1567_156703

def hourlyWage : ℝ := 4.0
def tipRate : ℝ := 0.15
def shiftsWorked : ℕ := 3
def averageOrdersPerHour : ℝ := 40.0
def totalEarnings : ℝ := 240.0

theorem hours_per_shift :
  (hourlyWage + averageOrdersPerHour * tipRate) * (8 * shiftsWorked) = totalEarnings := 
sorry

end NUMINAMATH_GPT_hours_per_shift_l1567_156703


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l1567_156780

theorem no_real_roots_of_quadratic (a b c : ℝ) (h_eq : a = 1 ∧ b = 1 ∧ c = 1) :
  (b^2 - 4 * a * c < 0) → ¬∃ x : ℝ, a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l1567_156780


namespace NUMINAMATH_GPT_celine_change_l1567_156749

theorem celine_change
  (price_laptop : ℕ)
  (price_smartphone : ℕ)
  (num_laptops : ℕ)
  (num_smartphones : ℕ)
  (total_money : ℕ)
  (h1 : price_laptop = 600)
  (h2 : price_smartphone = 400)
  (h3 : num_laptops = 2)
  (h4 : num_smartphones = 4)
  (h5 : total_money = 3000) :
  total_money - (num_laptops * price_laptop + num_smartphones * price_smartphone) = 200 :=
by
  sorry

end NUMINAMATH_GPT_celine_change_l1567_156749


namespace NUMINAMATH_GPT_discount_percentage_l1567_156787

theorem discount_percentage
  (number_of_fandoms : ℕ)
  (tshirts_per_fandom : ℕ)
  (price_per_shirt : ℝ)
  (tax_rate : ℝ)
  (total_paid : ℝ)
  (total_expected_price_with_discount_without_tax : ℝ)
  (total_expected_price_without_discount : ℝ)
  (discount_amount : ℝ)
  (discount_percentage : ℝ) :

  number_of_fandoms = 4 ∧
  tshirts_per_fandom = 5 ∧
  price_per_shirt = 15 ∧
  tax_rate = 10 / 100 ∧
  total_paid = 264 ∧
  total_expected_price_with_discount_without_tax = total_paid / (1 + tax_rate) ∧
  total_expected_price_without_discount = number_of_fandoms * tshirts_per_fandom * price_per_shirt ∧
  discount_amount = total_expected_price_without_discount - total_expected_price_with_discount_without_tax ∧
  discount_percentage = (discount_amount / total_expected_price_without_discount) * 100 ->

  discount_percentage = 20 :=
sorry

end NUMINAMATH_GPT_discount_percentage_l1567_156787


namespace NUMINAMATH_GPT_walking_rate_on_escalator_l1567_156725

theorem walking_rate_on_escalator (v : ℝ)
  (escalator_speed : ℝ := 12)
  (escalator_length : ℝ := 196)
  (travel_time : ℝ := 14)
  (effective_speed : ℝ := v + escalator_speed)
  (distance_eq : effective_speed * travel_time = escalator_length) :
  v = 2 := by
  sorry

end NUMINAMATH_GPT_walking_rate_on_escalator_l1567_156725


namespace NUMINAMATH_GPT_expression_evaluation_l1567_156718

-- Define the numbers and operations
def expr : ℚ := 10 * (1 / 2) * 3 / (1 / 6)

-- Formalize the proof problem
theorem expression_evaluation : expr = 90 := 
by 
  -- Start the proof, which is not required according to the instruction, so we replace it with 'sorry'
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1567_156718


namespace NUMINAMATH_GPT_complex_solution_l1567_156727

theorem complex_solution (i z : ℂ) (h : i^2 = -1) (hz : (z - 2 * i) * (2 - i) = 5) : z = 2 + 3 * i :=
sorry

end NUMINAMATH_GPT_complex_solution_l1567_156727


namespace NUMINAMATH_GPT_solution_set_inequality_system_l1567_156774

theorem solution_set_inequality_system (x : ℝ) :
  (x - 3 < 2 ∧ 3 * x + 1 ≥ 2 * x) ↔ (-1 ≤ x ∧ x < 5) := by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_system_l1567_156774


namespace NUMINAMATH_GPT_sum_first_n_terms_eq_l1567_156754

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b_n (n : ℕ) : ℕ := 2 ^ (n - 1)

noncomputable def c_n (n : ℕ) : ℕ := a_n n * b_n n

noncomputable def T_n (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ n + 3

theorem sum_first_n_terms_eq (n : ℕ) : 
  (Finset.sum (Finset.range n.succ) (λ k => c_n k) = T_n n) :=
  sorry

end NUMINAMATH_GPT_sum_first_n_terms_eq_l1567_156754


namespace NUMINAMATH_GPT_correctStatements_l1567_156763

-- Definitions based on conditions
def isFunctionalRelationshipDeterministic (S1 : Prop) := 
  S1 = true

def isCorrelationNonDeterministic (S2 : Prop) := 
  S2 = true

def regressionAnalysisFunctionalRelation (S3 : Prop) :=
  S3 = false

def regressionAnalysisCorrelation (S4 : Prop) :=
  S4 = true

-- The translated proof problem statement
theorem correctStatements :
  ∀ (S1 S2 S3 S4 : Prop), 
    isFunctionalRelationshipDeterministic S1 →
    isCorrelationNonDeterministic S2 →
    regressionAnalysisFunctionalRelation S3 →
    regressionAnalysisCorrelation S4 →
    (S1 ∧ S2 ∧ ¬S3 ∧ S4) →
    (S1 ∧ S2 ∧ ¬S3 ∧ S4) = (true ∧ true ∧ true ∧ true) :=
by
  intros S1 S2 S3 S4 H1 H2 H3 H4 H5
  sorry

end NUMINAMATH_GPT_correctStatements_l1567_156763


namespace NUMINAMATH_GPT_ratio_of_board_pieces_l1567_156720

theorem ratio_of_board_pieces (S L : ℕ) (hS : S = 23) (hTotal : S + L = 69) : L / S = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_board_pieces_l1567_156720


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_l1567_156732

theorem sum_of_consecutive_integers (n a : ℕ) (h₁ : 2 ≤ n) (h₂ : (n * (2 * a + n - 1)) = 36) :
    ∃! (a' n' : ℕ), 2 ≤ n' ∧ (n' * (2 * a' + n' - 1)) = 36 :=
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_l1567_156732


namespace NUMINAMATH_GPT_largest_good_number_is_576_smallest_bad_number_is_443_l1567_156781

def is_good_number (M : ℕ) : Prop :=
  ∃ (a b c d : ℤ), M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

def largest_good_number : ℕ := 576

def smallest_bad_number : ℕ := 443

theorem largest_good_number_is_576 : ∀ M : ℕ, is_good_number M → M ≤ 576 := 
by
  sorry

theorem smallest_bad_number_is_443 : ∀ M : ℕ, ¬ is_good_number M → 443 ≤ M :=
by
  sorry

end NUMINAMATH_GPT_largest_good_number_is_576_smallest_bad_number_is_443_l1567_156781


namespace NUMINAMATH_GPT_original_deck_total_l1567_156799

theorem original_deck_total (b y : ℕ) 
    (h1 : (b : ℚ) / (b + y) = 2 / 5)
    (h2 : (b : ℚ) / (b + y + 6) = 5 / 14) :
    b + y = 50 := by
  sorry

end NUMINAMATH_GPT_original_deck_total_l1567_156799


namespace NUMINAMATH_GPT_tan_of_alpha_intersects_unit_circle_l1567_156762

theorem tan_of_alpha_intersects_unit_circle (α : ℝ) (hα : ∃ P : ℝ × ℝ, P = (12 / 13, -5 / 13) ∧ ∀ x y : ℝ, P = (x, y) → x^2 + y^2 = 1) : 
  Real.tan α = -5 / 12 :=
by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_tan_of_alpha_intersects_unit_circle_l1567_156762


namespace NUMINAMATH_GPT_inverse_proportion_l1567_156708

theorem inverse_proportion (a : ℝ) (b : ℝ) (k : ℝ) : 
  (a = k / b^2) → 
  (40 = k / 12^2) → 
  (a = 10) → 
  b = 24 := 
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_l1567_156708


namespace NUMINAMATH_GPT_general_term_sum_formula_l1567_156724

-- Conditions for the sequence
variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (S : ℕ → ℤ)

-- Given conditions
axiom a2_eq_5 : a 2 = 5
axiom S4_eq_28 : S 4 = 28

-- The sequence is an arithmetic sequence
axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d

-- Statement 1: Proof that a_n = 4n - 3
theorem general_term (n : ℕ) : a n = 4 * n - 3 :=
by
  sorry

-- Statement 2: Proof that S_n = 2n^2 - n
theorem sum_formula (n : ℕ) : S n = 2 * n^2 - n :=
by
  sorry

end NUMINAMATH_GPT_general_term_sum_formula_l1567_156724


namespace NUMINAMATH_GPT_fraction_pow_zero_l1567_156789

theorem fraction_pow_zero :
  (4310000 / -21550000 : ℝ) ≠ 0 →
  (4310000 / -21550000 : ℝ) ^ 0 = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_fraction_pow_zero_l1567_156789


namespace NUMINAMATH_GPT_prove_monomial_l1567_156714

-- Definitions and conditions from step a)
def like_terms (x y : ℕ) := 
  x = 2 ∧ x + y = 5

-- Main statement to be proved
theorem prove_monomial (x y : ℕ) (h : like_terms x y) : 
  1 / 2 * x^3 - 1 / 6 * x * y^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_prove_monomial_l1567_156714


namespace NUMINAMATH_GPT_greatest_b_for_no_real_roots_l1567_156707

theorem greatest_b_for_no_real_roots :
  ∀ (b : ℤ), (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ 0) ↔ b ≤ 6 := sorry

end NUMINAMATH_GPT_greatest_b_for_no_real_roots_l1567_156707


namespace NUMINAMATH_GPT_locus_C2_angle_measure_90_l1567_156702

variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a)

-- Conditions for Question 1
def ellipse_C1 (x y : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

variable (x0 y0 x1 y1 : ℝ)
variable (hA : ellipse_C1 a b x0 y0)
variable (hE : ellipse_C1 a b x1 y1)
variable (h_perpendicular : x1 * x0 + y1 * y0 = 0)

theorem locus_C2 :
  ∀ (x y : ℝ), ellipse_C1 a b x y → 
  x ≠ 0 → y ≠ 0 → 
  (x^2 / a^2 + y^2 / b^2 = (a^2 - b^2)^2 / (a^2 + b^2)^2) := 
sorry

-- Conditions for Question 2
def circle_C3 (x y : ℝ) : Prop := 
  x^2 + y^2 = 1

theorem angle_measure_90 :
  (a^2 + b^2)^3 = a^2 * b^2 * (a^2 - b^2)^2 → 
  ∀ (x y : ℝ), ellipse_C1 a b x y → 
  circle_C3 x y → 
  (∃ (theta : ℝ), θ = 90) := 
sorry

end NUMINAMATH_GPT_locus_C2_angle_measure_90_l1567_156702


namespace NUMINAMATH_GPT_average_apples_sold_per_day_l1567_156796

theorem average_apples_sold_per_day (boxes_sold : ℕ) (days : ℕ) (apples_per_box : ℕ) (H1 : boxes_sold = 12) (H2 : days = 4) (H3 : apples_per_box = 25) : (boxes_sold * apples_per_box) / days = 75 :=
by {
  -- Based on given conditions, the total apples sold is 12 * 25 = 300.
  -- Dividing by the number of days, 300 / 4 gives us 75 apples/day.
  -- The proof is omitted as instructed.
  sorry
}

end NUMINAMATH_GPT_average_apples_sold_per_day_l1567_156796


namespace NUMINAMATH_GPT_board_arithmetic_impossibility_l1567_156795

theorem board_arithmetic_impossibility :
  ¬ (∃ (a b : ℕ), a ≡ 0 [MOD 7] ∧ b ≡ 1 [MOD 7] ∧ (a * b + a^3 + b^3) = 2013201420152016) := 
    sorry

end NUMINAMATH_GPT_board_arithmetic_impossibility_l1567_156795


namespace NUMINAMATH_GPT_Larry_wins_probability_l1567_156777

noncomputable def probability_Larry_wins (p_larry: ℚ) (p_paul: ℚ): ℚ :=
  let q_larry := 1 - p_larry
  let q_paul := 1 - p_paul
  p_larry / (1 - q_larry * q_paul)

theorem Larry_wins_probability:
  probability_Larry_wins (1/3 : ℚ) (1/2 : ℚ) = (2/5 : ℚ) :=
by {
  sorry
}

end NUMINAMATH_GPT_Larry_wins_probability_l1567_156777


namespace NUMINAMATH_GPT_brad_amount_l1567_156771

-- Definitions for the conditions
def total_amount (j d b : ℚ) := j + d + b = 68
def josh_twice_brad (j b : ℚ) := j = 2 * b
def josh_three_fourths_doug (j d : ℚ) := j = (3 / 4) * d

-- The theorem we want to prove
theorem brad_amount : ∃ (b : ℚ), (∃ (j d : ℚ), total_amount j d b ∧ josh_twice_brad j b ∧ josh_three_fourths_doug j d) ∧ b = 12 :=
sorry

end NUMINAMATH_GPT_brad_amount_l1567_156771


namespace NUMINAMATH_GPT_not_a_factorization_method_l1567_156741

def factorization_methods : Set String := 
  {"Taking out the common factor", "Cross multiplication method", "Formula method", "Group factorization"}

theorem not_a_factorization_method : 
  ¬ ("Addition and subtraction elimination method" ∈ factorization_methods) :=
sorry

end NUMINAMATH_GPT_not_a_factorization_method_l1567_156741


namespace NUMINAMATH_GPT_cos_30_eq_sqrt3_div_2_l1567_156712

theorem cos_30_eq_sqrt3_div_2 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_30_eq_sqrt3_div_2_l1567_156712


namespace NUMINAMATH_GPT_no_real_satisfies_absolute_value_equation_l1567_156743

theorem no_real_satisfies_absolute_value_equation :
  ∀ x : ℝ, ¬ (|x - 2| = |x - 1| + |x - 5|) :=
by
  sorry

end NUMINAMATH_GPT_no_real_satisfies_absolute_value_equation_l1567_156743


namespace NUMINAMATH_GPT_F_is_even_l1567_156700

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  (x^3 - 2*x) * f x

theorem F_is_even (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_nonzero : f 1 ≠ 0) :
  is_even_function (F f) :=
sorry

end NUMINAMATH_GPT_F_is_even_l1567_156700


namespace NUMINAMATH_GPT_base_conversion_l1567_156760

theorem base_conversion (b : ℕ) (h : 1 * 6^2 + 4 * 6 + 2 = 2 * b^2 + b + 5) : b = 5 :=
by
  sorry

end NUMINAMATH_GPT_base_conversion_l1567_156760


namespace NUMINAMATH_GPT_raul_money_left_l1567_156745

theorem raul_money_left (initial_money : ℕ) (cost_per_comic : ℕ) (number_of_comics : ℕ) (money_left : ℕ)
  (h1 : initial_money = 87)
  (h2 : cost_per_comic = 4)
  (h3 : number_of_comics = 8)
  (h4 : money_left = initial_money - (number_of_comics * cost_per_comic)) :
  money_left = 55 :=
by 
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_raul_money_left_l1567_156745


namespace NUMINAMATH_GPT_xyz_unique_solution_l1567_156722

theorem xyz_unique_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_eq : x + y^2 + z^3 = x * y * z)
  (h_gcd : z = Nat.gcd x y) : x = 5 ∧ y = 1 ∧ z = 1 :=
by
  sorry

end NUMINAMATH_GPT_xyz_unique_solution_l1567_156722


namespace NUMINAMATH_GPT_lines_not_form_triangle_l1567_156790

theorem lines_not_form_triangle {m : ℝ} :
  (∀ x y : ℝ, 2 * x - 3 * y + 1 ≠ 0 → 4 * x + 3 * y + 5 ≠ 0 → mx - y - 1 ≠ 0) →
  (m = -4 / 3 ∨ m = 2 / 3 ∨ m = 4 / 3) :=
sorry

end NUMINAMATH_GPT_lines_not_form_triangle_l1567_156790


namespace NUMINAMATH_GPT_fraction_value_l1567_156759

theorem fraction_value : (1 - 1 / 4) / (1 - 1 / 5) = 15 / 16 := sorry

end NUMINAMATH_GPT_fraction_value_l1567_156759


namespace NUMINAMATH_GPT_total_earnings_l1567_156742

theorem total_earnings (d_a : ℕ) (h : 57 * d_a + 684 + 380 = 1406) : d_a = 6 :=
by {
  -- The proof will involve algebraic manipulations similar to the solution steps
  sorry
}

end NUMINAMATH_GPT_total_earnings_l1567_156742


namespace NUMINAMATH_GPT_length_of_plot_l1567_156765

-- Definitions of the given conditions, along with the question.
def breadth (b : ℝ) : Prop := 2 * (b + 32) + 2 * b = 5300 / 26.50
def length (b : ℝ) := b + 32

theorem length_of_plot (b : ℝ) (h : breadth b) : length b = 66 := by 
  sorry

end NUMINAMATH_GPT_length_of_plot_l1567_156765


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1567_156791

theorem solution_set_of_quadratic_inequality 
  (a b c x₁ x₂ : ℝ)
  (h1 : a > 0) 
  (h2 : a * x₁^2 + b * x₁ + c = 0)
  (h3 : a * x₂^2 + b * x₂ + c = 0)
  : {x : ℝ | a * x^2 + b * x + c > 0} = ({x : ℝ | x > x₁} ∩ {x : ℝ | x > x₂}) ∪ ({x : ℝ | x < x₁} ∩ {x : ℝ | x < x₂}) :=
sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1567_156791


namespace NUMINAMATH_GPT_academy_league_total_games_l1567_156717

theorem academy_league_total_games (teams : ℕ) (plays_each_other_twice games_non_conference : ℕ) 
  (h_teams : teams = 8)
  (h_plays_each_other_twice : plays_each_other_twice = 2 * teams * (teams - 1) / 2)
  (h_games_non_conference : games_non_conference = 6 * teams) :
  (plays_each_other_twice + games_non_conference) = 104 :=
by
  sorry

end NUMINAMATH_GPT_academy_league_total_games_l1567_156717


namespace NUMINAMATH_GPT_part1_part2_l1567_156723

def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

theorem part1 : {x : ℝ | f x ≤ 4} = {x : ℝ | -5 / 3 ≤ x ∧ x ≤ 1} :=
by
  sorry

theorem part2 {a : ℝ} :
  ({x : ℝ | f x ≤ 4} ⊆ {x : ℝ | |x + 3| + |x + a| < x + 6}) ↔ (-4 / 3 < a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1567_156723


namespace NUMINAMATH_GPT_find_arith_seq_params_l1567_156756

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- The conditions given in the problem
theorem find_arith_seq_params :
  ∃ a d : ℤ, 
  (arithmetic_sequence a d 8) = 5 * (arithmetic_sequence a d 1) ∧
  (arithmetic_sequence a d 12) = 2 * (arithmetic_sequence a d 5) + 5 ∧
  a = 3 ∧
  d = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_arith_seq_params_l1567_156756


namespace NUMINAMATH_GPT_probability_of_dime_l1567_156740

noncomputable def num_quarters := 12 / 0.25
noncomputable def num_dimes := 8 / 0.10
noncomputable def num_pennies := 5 / 0.01
noncomputable def total_coins := num_quarters + num_dimes + num_pennies

theorem probability_of_dime : (num_dimes / total_coins) = (40 / 314) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_dime_l1567_156740


namespace NUMINAMATH_GPT_total_length_circle_l1567_156770

-- Definitions based on conditions
def num_strips : ℕ := 16
def length_each_strip : ℝ := 10.4
def overlap_each_strip : ℝ := 3.5

-- Theorem stating the total length of the circle-shaped colored tape
theorem total_length_circle : 
  (num_strips * length_each_strip) - (num_strips * overlap_each_strip) = 110.4 := 
by 
  sorry

end NUMINAMATH_GPT_total_length_circle_l1567_156770


namespace NUMINAMATH_GPT_brochures_per_box_l1567_156782

theorem brochures_per_box (total_brochures : ℕ) (boxes : ℕ) 
  (htotal : total_brochures = 5000) (hboxes : boxes = 5) : 
  (1000 / 5000 : ℚ) = 1 / 5 := 
by sorry

end NUMINAMATH_GPT_brochures_per_box_l1567_156782


namespace NUMINAMATH_GPT_total_transportation_cost_l1567_156793

def weights_in_grams : List ℕ := [300, 450, 600]
def cost_per_kg : ℕ := 15000

def convert_to_kg (w : ℕ) : ℚ :=
  w / 1000

def calculate_cost (weight_in_kg : ℚ) (cost_per_kg : ℕ) : ℚ :=
  weight_in_kg * cost_per_kg

def total_cost (weights_in_grams : List ℕ) (cost_per_kg : ℕ) : ℚ :=
  weights_in_grams.map (λ w => calculate_cost (convert_to_kg w) cost_per_kg) |>.sum

theorem total_transportation_cost :
  total_cost weights_in_grams cost_per_kg = 20250 := by
  sorry

end NUMINAMATH_GPT_total_transportation_cost_l1567_156793


namespace NUMINAMATH_GPT_peter_stamps_l1567_156738

theorem peter_stamps (M : ℕ) (h1 : M % 5 = 2) (h2 : M % 11 = 2) (h3 : M % 13 = 2) (h4 : M > 1) : M = 717 :=
by
  -- proof will be filled in
  sorry

end NUMINAMATH_GPT_peter_stamps_l1567_156738


namespace NUMINAMATH_GPT_slope_of_chord_l1567_156779

theorem slope_of_chord (x1 x2 y1 y2 : ℝ) (P : ℝ × ℝ)
    (hp : P = (3, 2))
    (h1 : 4 * x1 ^ 2 + 9 * y1 ^ 2 = 144)
    (h2 : 4 * x2 ^ 2 + 9 * y2 ^ 2 = 144)
    (h3 : (x1 + x2) / 2 = 3)
    (h4 : (y1 + y2) / 2 = 2) : 
    (y1 - y2) / (x1 - x2) = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_chord_l1567_156779


namespace NUMINAMATH_GPT_expression_inside_absolute_value_l1567_156786

theorem expression_inside_absolute_value (E : ℤ) (x : ℤ) (h1 : x = 10) (h2 : 30 - |E| = 26) :
  E = 4 ∨ E = -4 := 
by
  sorry

end NUMINAMATH_GPT_expression_inside_absolute_value_l1567_156786


namespace NUMINAMATH_GPT_odd_two_digit_combinations_l1567_156734

theorem odd_two_digit_combinations (digits : Finset ℕ) (h_digits : digits = {1, 3, 5, 7, 9}) :
  ∃ n : ℕ, n = 20 ∧ (∃ a b : ℕ, a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ (10 * a + b) % 2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_odd_two_digit_combinations_l1567_156734


namespace NUMINAMATH_GPT_triangle_area_eq_40_sqrt_3_l1567_156753

open Real

theorem triangle_area_eq_40_sqrt_3 
  (a : ℝ) (A : ℝ) (b c : ℝ)
  (h1 : a = 14)
  (h2 : A = π / 3) -- 60 degrees in radians
  (h3 : b / c = 8 / 5) :
  1 / 2 * b * c * sin A = 40 * sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_eq_40_sqrt_3_l1567_156753


namespace NUMINAMATH_GPT_find_tabitha_age_l1567_156728

-- Define the conditions
variable (age_started : ℕ) (colors_started : ℕ) (years_future : ℕ) (future_colors : ℕ)

-- Let's specify the given problem's conditions:
axiom h1 : age_started = 15          -- Tabitha started at age 15
axiom h2 : colors_started = 2        -- with 2 colors
axiom h3 : years_future = 3          -- in three years
axiom h4 : future_colors = 8         -- she will have 8 different colors

-- The proof problem we need to state:
theorem find_tabitha_age : ∃ age_now : ℕ, age_now = age_started + (future_colors - colors_started) - years_future := by
  sorry

end NUMINAMATH_GPT_find_tabitha_age_l1567_156728


namespace NUMINAMATH_GPT_pieces_of_wood_for_chair_is_correct_l1567_156788

-- Define the initial setup and constants
def total_pieces_of_wood := 672
def pieces_of_wood_per_table := 12
def number_of_tables := 24
def number_of_chairs := 48

-- Calculation in the conditions
def pieces_of_wood_used_for_tables := number_of_tables * pieces_of_wood_per_table
def pieces_of_wood_left_for_chairs := total_pieces_of_wood - pieces_of_wood_used_for_tables

-- Question and answer verification
def pieces_of_wood_per_chair := pieces_of_wood_left_for_chairs / number_of_chairs

theorem pieces_of_wood_for_chair_is_correct :
  pieces_of_wood_per_chair = 8 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_pieces_of_wood_for_chair_is_correct_l1567_156788


namespace NUMINAMATH_GPT_min_value_of_algebraic_sum_l1567_156737

theorem min_value_of_algebraic_sum 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : a + 3 * b = 3) :
  ∃ (min_value : ℝ), min_value = 16 / 3 ∧ (∀ a b, a > 0 → b > 0 → a + 3 * b = 3 → 1 / a + 3 / b ≥ min_value) :=
sorry

end NUMINAMATH_GPT_min_value_of_algebraic_sum_l1567_156737


namespace NUMINAMATH_GPT_third_week_cases_l1567_156739

-- Define the conditions as Lean definitions
def first_week_cases : ℕ := 5000
def second_week_cases : ℕ := first_week_cases / 2
def total_cases_after_three_weeks : ℕ := 9500

-- The statement to be proven
theorem third_week_cases :
  first_week_cases + second_week_cases + 2000 = total_cases_after_three_weeks :=
by
  sorry

end NUMINAMATH_GPT_third_week_cases_l1567_156739


namespace NUMINAMATH_GPT_time_to_finish_all_problems_l1567_156709

def mathProblems : ℝ := 17.0
def spellingProblems : ℝ := 15.0
def problemsPerHour : ℝ := 8.0
def totalProblems : ℝ := mathProblems + spellingProblems

theorem time_to_finish_all_problems : totalProblems / problemsPerHour = 4.0 :=
by
  sorry

end NUMINAMATH_GPT_time_to_finish_all_problems_l1567_156709


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l1567_156767

theorem cone_lateral_surface_area (r : ℝ) (V : ℝ) (h : ℝ) (l : ℝ) 
  (h₁ : r = 3)
  (h₂ : V = 12 * Real.pi)
  (h₃ : V = (1 / 3) * Real.pi * r^2 * h)
  (h₄ : l = Real.sqrt (r^2 + h^2)) : 
  ∃ A : ℝ, A = Real.pi * r * l ∧ A = 15 * Real.pi := 
by
  use Real.pi * r * l
  have hr : r = 3 := by exact h₁
  have hV : V = 12 * Real.pi := by exact h₂
  have volume_formula : V = (1 / 3) * Real.pi * r^2 * h := by exact h₃
  have slant_height : l = Real.sqrt (r^2 + h^2) := by exact h₄
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l1567_156767


namespace NUMINAMATH_GPT_shaded_area_inequality_l1567_156730

theorem shaded_area_inequality 
    (A : ℝ) -- All three triangles have the same total area, A.
    {a1 a2 a3 : ℝ} -- a1, a2, a3 are the shaded areas of Triangle I, II, and III respectively.
    (h1 : a1 = A / 6) 
    (h2 : a2 = A / 2) 
    (h3 : a3 = (2 * A) / 3) : 
    a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 :=
by
  -- Proof steps would go here, but they are not required as per the instructions
  sorry

end NUMINAMATH_GPT_shaded_area_inequality_l1567_156730


namespace NUMINAMATH_GPT_range_of_a_l1567_156775

theorem range_of_a (a : ℝ) :
  ¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 ≤ 0 → -1 < a ∧ a < 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1567_156775


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_max_min_values_of_f_on_interval_l1567_156729

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) ^ 2 + Real.cos (2 * x)

theorem smallest_positive_period_of_f : ∀ (x : ℝ), f (x + π) = f x :=
by sorry

theorem max_min_values_of_f_on_interval : ∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ ≤ π / 2 ∧ 0 ≤ x₂ ∧ x₂ ≤ π / 2 ∧
  f x₁ = 0 ∧ f x₂ = 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_max_min_values_of_f_on_interval_l1567_156729


namespace NUMINAMATH_GPT_sequence_general_term_l1567_156757

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = (1 / 2) * a n + 1) :
  ∀ n, a n = 2 - (1 / 2) ^ (n - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1567_156757


namespace NUMINAMATH_GPT_find_value_of_A_l1567_156778

theorem find_value_of_A (x y A : ℝ)
  (h1 : 2^x = A)
  (h2 : 7^(2*y) = A)
  (h3 : 1 / x + 2 / y = 2) : 
  A = 7 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_find_value_of_A_l1567_156778


namespace NUMINAMATH_GPT_sqrt_prime_geometric_progression_impossible_l1567_156766

theorem sqrt_prime_geometric_progression_impossible {p1 p2 p3 : ℕ} (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3) (hneq12 : p1 ≠ p2) (hneq23 : p2 ≠ p3) (hneq31 : p3 ≠ p1) :
  ¬ ∃ (a r : ℝ) (n1 n2 n3 : ℤ), (a * r^n1 = Real.sqrt p1) ∧ (a * r^n2 = Real.sqrt p2) ∧ (a * r^n3 = Real.sqrt p3) := sorry

end NUMINAMATH_GPT_sqrt_prime_geometric_progression_impossible_l1567_156766


namespace NUMINAMATH_GPT_sale_price_with_50_percent_profit_l1567_156733

theorem sale_price_with_50_percent_profit (CP SP₁ SP₃ : ℝ) 
(h1 : SP₁ - CP = CP - 448) 
(h2 : SP₃ = 1.5 * CP) 
(h3 : SP₃ = 1020) : 
SP₃ = 1020 := 
by 
  sorry

end NUMINAMATH_GPT_sale_price_with_50_percent_profit_l1567_156733


namespace NUMINAMATH_GPT_jamie_dimes_l1567_156785

theorem jamie_dimes (y : ℕ) (h : 5 * y + 10 * y + 25 * y = 1440) : y = 36 :=
by 
  sorry

end NUMINAMATH_GPT_jamie_dimes_l1567_156785


namespace NUMINAMATH_GPT_Caitlin_age_l1567_156744

theorem Caitlin_age (Aunt_Anna_age : ℕ) (Brianna_age : ℕ) (Caitlin_age : ℕ)
    (h1 : Aunt_Anna_age = 48)
    (h2 : Brianna_age = Aunt_Anna_age / 3)
    (h3 : Caitlin_age = Brianna_age - 6) : 
    Caitlin_age = 10 := by 
  -- proof here
  sorry

end NUMINAMATH_GPT_Caitlin_age_l1567_156744


namespace NUMINAMATH_GPT_find_principal_amount_l1567_156721

variable (x y : ℝ)

-- conditions given in the problem
def simple_interest_condition : Prop :=
  600 = (x * y * 2) / 100

def compound_interest_condition : Prop :=
  615 = x * ((1 + y / 100)^2 - 1)

-- target statement to be proven
theorem find_principal_amount (h1 : simple_interest_condition x y) (h2 : compound_interest_condition x y) :
  x = 285.7142857 :=
  sorry

end NUMINAMATH_GPT_find_principal_amount_l1567_156721


namespace NUMINAMATH_GPT_presidency_meeting_ways_l1567_156719

theorem presidency_meeting_ways :
  let total_schools := 4
  let members_per_school := 4
  let host_school_choices := total_schools
  let choose_3_from_4 := Nat.choose 4 3
  let choose_1_from_4 := Nat.choose 4 1
  let ways_per_host := choose_3_from_4 * choose_1_from_4 ^ 3
  let total_ways := host_school_choices * ways_per_host
  total_ways = 1024 := by
  sorry

end NUMINAMATH_GPT_presidency_meeting_ways_l1567_156719


namespace NUMINAMATH_GPT_exists_x0_in_interval_l1567_156764

noncomputable def f (x : ℝ) : ℝ := (2 : ℝ) / x + Real.log (1 / (x - 1))

theorem exists_x0_in_interval :
  ∃ x0 ∈ Set.Ioo (2 : ℝ) (3 : ℝ), f x0 = 0 := 
sorry  -- Proof is left as an exercise

end NUMINAMATH_GPT_exists_x0_in_interval_l1567_156764


namespace NUMINAMATH_GPT_solve_for_x_l1567_156768

theorem solve_for_x : ∃ x : ℝ, 3 * x - 6 = |(-20 + 5)| ∧ x = 7 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1567_156768


namespace NUMINAMATH_GPT_cost_of_second_type_of_rice_is_22_l1567_156784

noncomputable def cost_second_type_of_rice (c1 : ℝ) (w1 : ℝ) (w2 : ℝ) (avg : ℝ) (total_weight : ℝ) : ℝ :=
  ((total_weight * avg) - (w1 * c1)) / w2

theorem cost_of_second_type_of_rice_is_22 :
  cost_second_type_of_rice 16 8 4 18 12 = 22 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_second_type_of_rice_is_22_l1567_156784


namespace NUMINAMATH_GPT_class_students_l1567_156758

theorem class_students :
  ∃ n : ℕ,
    (∃ m : ℕ, 2 * m = n) ∧
    (∃ q : ℕ, 4 * q = n) ∧
    (∃ l : ℕ, 7 * l = n) ∧
    (∀ f : ℕ, f < 6 → n - (n / 2) - (n / 4) - (n / 7) = f) ∧
    n = 28 :=
by
  sorry

end NUMINAMATH_GPT_class_students_l1567_156758


namespace NUMINAMATH_GPT_tan_alpha_eq_cos_two_alpha_plus_quarter_pi_sin_beta_eq_l1567_156783

-- Definitions
variables {α β : ℝ}

-- Condition: 0 < α < π / 2
def valid_alpha (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

-- Condition: sin α = 4 / 5
def sin_alpha (α : ℝ) : Prop := Real.sin α = 4 / 5

-- Condition: 0 < β < π / 2
def valid_beta (β : ℝ) : Prop := 0 < β ∧ β < Real.pi / 2

-- Condition: cos (α + β) = -1 / 2
def cos_alpha_add_beta (α β : ℝ) : Prop := Real.cos (α + β) = - 1 / 2

/-- Proofs begin -/
-- Proof for tan α = 4 / 3 given 0 < α < π / 2 and sin α = 4 / 5
theorem tan_alpha_eq (α : ℝ) (h_valid : valid_alpha α) (h_sin : sin_alpha α) : Real.tan α = 4 / 3 := 
  sorry

-- Proof for cos (2α + π / 4) = -31√2 / 50 given 0 < α < π / 2 and sin α = 4 / 5
theorem cos_two_alpha_plus_quarter_pi (α : ℝ) (h_valid : valid_alpha α) (h_sin : sin_alpha α) : 
  Real.cos (2 * α + Real.pi / 4) = -31 * Real.sqrt 2 / 50 := 
  sorry

-- Proof for sin β = 4 + 3√3 / 10 given 0 < α < π / 2, sin α = 4 / 5, 0 < β < π / 2 and cos (α + β) = -1 / 2
theorem sin_beta_eq (α β : ℝ) (h_validα : valid_alpha α) (h_sinα : sin_alpha α) 
  (h_validβ : valid_beta β) (h_cosαβ : cos_alpha_add_beta α β) : Real.sin β = 4 + 3 * Real.sqrt 3 / 10 := 
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_cos_two_alpha_plus_quarter_pi_sin_beta_eq_l1567_156783


namespace NUMINAMATH_GPT_range_of_a_l1567_156710

def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) :=
∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y

theorem range_of_a (a : ℝ) :
  (∀ x y, 3 ≤ x ∧ x ≤ y → (x^2 - 2*a*x + 2) ≤ (y^2 - 2*a*y + 2)) → a ≤ 3 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1567_156710


namespace NUMINAMATH_GPT_axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l1567_156746

-- (1) Axis of symmetry
theorem axis_of_symmetry (t : ℝ) :
  ∀ x y : ℝ, (y = x^2 - 2*t*x + 1) → (x = t) := sorry

-- (2) Comparison of m and n
theorem compare_m_n (t m n : ℝ) :
  (t - 2)^2 - 2*t*(t - 2) + 1 = m*1 →
  (t + 3)^2 - 2*t*(t + 3) + 1 = n*1 →
  n > m := sorry

-- (3) Range of t for y₁ ≤ y₂
theorem range_of_t_for_y1_leq_y2 (t x1 x2 y1 y2 : ℝ) :
  (-1 ≤ x1) → (x1 < 3) → (x2 = 3) → 
  (y1 = x1^2 - 2*t*x1 + 1) → 
  (y2 = x2^2 - 2*t*x2 + 1) → 
  y1 ≤ y2 →
  t ≤ 1 := sorry

-- (4) Maximum value of t
theorem maximum_value_of_t (t y1 y2 : ℝ) :
  (y1 = (t + 1)^2 - 2*t*(t + 1) + 1) →
  (y2 = (2*t - 4)^2 - 2*t*(2*t - 4) + 1) →
  y1 ≥ y2 →
  t = 5 := sorry

end NUMINAMATH_GPT_axis_of_symmetry_compare_m_n_range_of_t_for_y1_leq_y2_maximum_value_of_t_l1567_156746


namespace NUMINAMATH_GPT_adults_wearing_sunglasses_l1567_156751

def total_adults : ℕ := 2400
def one_third_of_adults (total : ℕ) : ℕ := total / 3
def women_wearing_sunglasses (women : ℕ) : ℕ := (15 * women) / 100
def men_wearing_sunglasses (men : ℕ) : ℕ := (12 * men) / 100

theorem adults_wearing_sunglasses : 
  let women := one_third_of_adults total_adults
  let men := total_adults - women
  let women_in_sunglasses := women_wearing_sunglasses women
  let men_in_sunglasses := men_wearing_sunglasses men
  women_in_sunglasses + men_in_sunglasses = 312 :=
by
  sorry

end NUMINAMATH_GPT_adults_wearing_sunglasses_l1567_156751


namespace NUMINAMATH_GPT_carl_took_4_pink_hard_hats_l1567_156716

-- Define the initial number of hard hats
def initial_pink : ℕ := 26
def initial_green : ℕ := 15
def initial_yellow : ℕ := 24

-- Define the number of hard hats John took
def john_pink : ℕ := 6
def john_green : ℕ := 2 * john_pink
def john_total : ℕ := john_pink + john_green

-- Define the total initial number of hard hats
def total_initial : ℕ := initial_pink + initial_green + initial_yellow

-- Define the number of hard hats remaining after John's removal
def remaining_after_john : ℕ := total_initial - john_total

-- Define the total number of hard hats that remained in the truck
def total_remaining : ℕ := 43

-- Define the number of pink hard hats Carl took away
def carl_pink : ℕ := remaining_after_john - total_remaining

-- State the proof problem
theorem carl_took_4_pink_hard_hats : carl_pink = 4 := by
  sorry

end NUMINAMATH_GPT_carl_took_4_pink_hard_hats_l1567_156716


namespace NUMINAMATH_GPT_coin_toss_5_times_same_side_l1567_156701

noncomputable def probability_of_same_side (n : ℕ) : ℝ :=
  (1 / 2) ^ n

theorem coin_toss_5_times_same_side :
  probability_of_same_side 5 = 1 / 32 :=
by 
  -- The goal is to prove (1/2)^5 = 1/32
  sorry

end NUMINAMATH_GPT_coin_toss_5_times_same_side_l1567_156701


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1567_156752

noncomputable def hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (4:ℝ) * a^2 = c^2) : ℝ :=
  c / a

theorem eccentricity_of_hyperbola (a b c : ℝ) (ha : a > 0) (hb : b > 0) (h : (4:ℝ) * a^2 = c^2) :
  hyperbola_eccentricity a b c ha hb h = 2 :=
by
  sorry


end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1567_156752


namespace NUMINAMATH_GPT_area_of_yard_l1567_156769

def length {w : ℝ} : ℝ := 2 * w + 30

def perimeter {w l : ℝ} (cond_len : l = 2 * w + 30) : Prop := 2 * w + 2 * l = 700

theorem area_of_yard {w l A : ℝ} 
  (cond_len : l = 2 * w + 30) 
  (cond_perim : 2 * w + 2 * l = 700) : 
  A = w * l := 
  sorry

end NUMINAMATH_GPT_area_of_yard_l1567_156769


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1567_156731

variables (a b : ℝ)

def p : Prop := a > b ∧ b > 1
def q : Prop := a - b < a^2 - b^2

theorem sufficient_but_not_necessary_condition (h : p a b) : q a b :=
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1567_156731


namespace NUMINAMATH_GPT_father_20_bills_count_l1567_156715

-- Defining the conditions from the problem.
variables (mother50 mother20 mother10 father50 father10 : ℕ)
def mother_total := mother50 * 50 + mother20 * 20 + mother10 * 10
def father_total (x : ℕ) := father50 * 50 + x * 20 + father10 * 10

-- Given conditions
axiom mother_given : mother50 = 1 ∧ mother20 = 2 ∧ mother10 = 3
axiom father_given : father50 = 4 ∧ father10 = 1
axiom school_fee : 350 = 350

-- Theorem to prove
theorem father_20_bills_count (x : ℕ) :
  mother_total 1 2 3 + father_total 4 x 1 = 350 → x = 1 :=
by sorry

end NUMINAMATH_GPT_father_20_bills_count_l1567_156715


namespace NUMINAMATH_GPT_students_in_class_l1567_156713

theorem students_in_class (total_pencils : ℕ) (pencils_per_student : ℕ) (n: ℕ) 
    (h1 : total_pencils = 18) 
    (h2 : pencils_per_student = 9) 
    (h3 : total_pencils = n * pencils_per_student) : 
    n = 2 :=
by 
  sorry

end NUMINAMATH_GPT_students_in_class_l1567_156713


namespace NUMINAMATH_GPT_second_discarded_number_l1567_156705

theorem second_discarded_number (S : ℝ) (X : ℝ) (h1 : S / 50 = 62) (h2 : (S - 45 - X) / 48 = 62.5) : X = 55 := 
by
  sorry

end NUMINAMATH_GPT_second_discarded_number_l1567_156705


namespace NUMINAMATH_GPT_children_playing_tennis_l1567_156726

theorem children_playing_tennis
  (Total : ℕ) (S : ℕ) (N : ℕ) (B : ℕ) (T : ℕ) 
  (hTotal : Total = 38) (hS : S = 21) (hN : N = 10) (hB : B = 12) :
  T = 38 - 21 + 12 - 10 :=
by
  sorry

end NUMINAMATH_GPT_children_playing_tennis_l1567_156726


namespace NUMINAMATH_GPT_largest_A_divisible_by_8_l1567_156792

theorem largest_A_divisible_by_8 (A B C : ℕ) (h1 : A = 8 * B + C) (h2 : B = C) (h3 : C < 8) : A ≤ 9 * 7 :=
by sorry

end NUMINAMATH_GPT_largest_A_divisible_by_8_l1567_156792


namespace NUMINAMATH_GPT_function_y_neg3x_plus_1_quadrants_l1567_156704

theorem function_y_neg3x_plus_1_quadrants :
  ∀ (x : ℝ), (∃ y : ℝ, y = -3 * x + 1) ∧ (
    (x < 0 ∧ y > 0) ∨ -- Second quadrant
    (x > 0 ∧ y > 0) ∨ -- First quadrant
    (x > 0 ∧ y < 0)   -- Fourth quadrant
  )
:= sorry

end NUMINAMATH_GPT_function_y_neg3x_plus_1_quadrants_l1567_156704


namespace NUMINAMATH_GPT_number_of_lines_dist_l1567_156798

theorem number_of_lines_dist {A B : ℝ × ℝ} (hA : A = (3, 0)) (hB : B = (0, 4)) : 
  ∃ n : ℕ, n = 3 ∧
  ∀ l : ℝ → ℝ → Prop, 
  (∀ p : ℝ × ℝ, l p.1 p.2 → p ≠ A → dist A p = 2) ∧ 
  (∀ p : ℝ × ℝ, l p.1 p.2 → p ≠ B → dist B p = 3) → n = 3 := 
by sorry

end NUMINAMATH_GPT_number_of_lines_dist_l1567_156798


namespace NUMINAMATH_GPT_line_intersects_circle_l1567_156794

theorem line_intersects_circle (k : ℝ) :
  ∃ x y : ℝ, y = k * (x - 1) ∧ x^2 + y^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_circle_l1567_156794
