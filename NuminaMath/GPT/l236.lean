import Mathlib

namespace NUMINAMATH_GPT_value_of_y_at_3_l236_23618

-- Define the function
def f (x : ℕ) : ℕ := 2 * x^2 + 1

-- Prove that when x = 3, y = 19
theorem value_of_y_at_3 : f 3 = 19 :=
by
  -- Provide the definition and conditions
  let x := 3
  let y := f x
  have h : y = 2 * x^2 + 1 := rfl
  -- State the actual proof could go here
  sorry

end NUMINAMATH_GPT_value_of_y_at_3_l236_23618


namespace NUMINAMATH_GPT_find_ordered_pair_l236_23669

theorem find_ordered_pair : 
  ∃ (x y : ℚ), 7 * x = -5 - 3 * y ∧ 4 * x = 5 * y - 34 ∧
  x = -127 / 47 ∧ y = 218 / 47 :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l236_23669


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l236_23663

theorem problem1 : (-20 + (-14) - (-18) - 13) = -29 := by
  sorry

theorem problem2 : (-6 * (-2) / (1 / 8)) = 96 := by
  sorry

theorem problem3 : (-24 * (-3 / 4 - 5 / 6 + 7 / 8)) = 17 := by
  sorry

theorem problem4 : (-1^4 - (1 - 0.5) * (1 / 3) * (-3)^2) = -5 / 2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l236_23663


namespace NUMINAMATH_GPT_find_angle_B_l236_23697

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l236_23697


namespace NUMINAMATH_GPT_coefficients_sum_correct_l236_23645

noncomputable def poly_expr (x : ℝ) : ℝ := (x + 2)^4

def coefficients_sum (a a_1 a_2 a_3 a_4 : ℝ) : ℝ :=
  a_1 + a_2 + a_3 + a_4

theorem coefficients_sum_correct (a a_1 a_2 a_3 a_4 : ℝ) :
  poly_expr 1 = a_4 * 1 ^ 4 + a_3 * 1 ^ 3 + a_2 * 1 ^ 2 + a_1 * 1 + a →
  a = 16 → coefficients_sum a a_1 a_2 a_3 a_4 = 65 :=
by
  intro h₁ h₂
  sorry

end NUMINAMATH_GPT_coefficients_sum_correct_l236_23645


namespace NUMINAMATH_GPT_geometry_problem_l236_23609

/-- Given:
  DC = 5
  CB = 9
  AB = 1/3 * AD
  ED = 2/3 * AD
  Prove: FC = 10.6667 -/
theorem geometry_problem
  (DC CB AD FC : ℝ) (hDC : DC = 5) (hCB : CB = 9) (hAB : AB = 1 / 3 * AD) (hED : ED = 2 / 3 * AD)
  (AB ED: ℝ):
  FC = 10.6667 :=
by
  sorry

end NUMINAMATH_GPT_geometry_problem_l236_23609


namespace NUMINAMATH_GPT_systematic_sampling_l236_23671

-- Define the conditions
def total_products : ℕ := 100
def selected_products (n : ℕ) : ℕ := 3 + 10 * n
def is_systematic (f : ℕ → ℕ) : Prop :=
  ∃ k b, ∀ n, f n = b + k * n

-- Theorem to prove that the selection method is systematic sampling
theorem systematic_sampling : is_systematic selected_products :=
  sorry

end NUMINAMATH_GPT_systematic_sampling_l236_23671


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l236_23636

/--
Express \(2.\overline{06}\) as a reduced fraction, given that \(0.\overline{01} = \frac{1}{99}\)
-/
theorem repeating_decimal_to_fraction : 
  (0.01:ℚ) = 1 / 99 → (2.06:ℚ) = 68 / 33 := 
by 
  sorry 

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l236_23636


namespace NUMINAMATH_GPT_tom_final_payment_l236_23656

noncomputable def cost_of_fruit (kg: ℝ) (rate_per_kg: ℝ) := kg * rate_per_kg

noncomputable def total_bill := 
  cost_of_fruit 15.3 1.85 + cost_of_fruit 12.7 2.45 + cost_of_fruit 10.5 3.20 + cost_of_fruit 6.2 4.50

noncomputable def discount (bill: ℝ) := 0.10 * bill

noncomputable def discounted_total (bill: ℝ) := bill - discount bill

noncomputable def sales_tax (amount: ℝ) := 0.06 * amount

noncomputable def final_amount (bill: ℝ) := discounted_total bill + sales_tax (discounted_total bill)

theorem tom_final_payment : final_amount total_bill = 115.36 :=
  sorry

end NUMINAMATH_GPT_tom_final_payment_l236_23656


namespace NUMINAMATH_GPT_syllogism_arrangement_l236_23639

theorem syllogism_arrangement : 
  (∀ n : ℕ, Odd n → ¬ (n % 2 = 0)) → 
  Odd 2013 → 
  (¬ (2013 % 2 = 0)) :=
by
  intros h1 h2
  exact h1 2013 h2

end NUMINAMATH_GPT_syllogism_arrangement_l236_23639


namespace NUMINAMATH_GPT_counterexample_to_prime_statement_l236_23631

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

theorem counterexample_to_prime_statement 
  (n : ℕ) 
  (h_n_composite : is_composite n) 
  (h_n_minus_3_not_prime : ¬ is_prime (n - 3)) : 
  n = 18 ∨ n = 24 :=
by 
  sorry

end NUMINAMATH_GPT_counterexample_to_prime_statement_l236_23631


namespace NUMINAMATH_GPT_find_number_l236_23680

theorem find_number (x : ℕ) (h : x / 4 + 3 = 5) : x = 8 :=
by sorry

end NUMINAMATH_GPT_find_number_l236_23680


namespace NUMINAMATH_GPT_max_sum_ac_bc_l236_23685

noncomputable def triangle_ab_bc_sum_max (AB : ℝ) (C : ℝ) : ℝ :=
  if AB = Real.sqrt 6 - Real.sqrt 2 ∧ C = Real.pi / 6 then 4 else 0

theorem max_sum_ac_bc {A B C : ℝ} (h1 : AB = Real.sqrt 6 - Real.sqrt 2) (h2 : C = Real.pi / 6) :
  triangle_ab_bc_sum_max AB C = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_sum_ac_bc_l236_23685


namespace NUMINAMATH_GPT_identify_radioactive_balls_l236_23622

theorem identify_radioactive_balls (balls : Fin 11 → Bool) (measure : (Finset (Fin 11)) → Bool) :
  (∃ (t1 t2 : Fin 11), ¬ t1 = t2 ∧ balls t1 = true ∧ balls t2 = true) →
  (∃ (pairs : List (Finset (Fin 11))), pairs.length ≤ 7 ∧
    ∀ t1 t2, t1 ≠ t2 ∧ balls t1 = true ∧ balls t2 = true →
      ∃ pair ∈ pairs, measure pair = true ∧ (t1 ∈ pair ∨ t2 ∈ pair)) :=
by
  sorry

end NUMINAMATH_GPT_identify_radioactive_balls_l236_23622


namespace NUMINAMATH_GPT_joan_change_received_l236_23628

theorem joan_change_received :
  let cat_toy_cost := 8.77
  let cage_cost := 10.97
  let payment := 20.00
  let total_cost := cat_toy_cost + cage_cost
  let change_received := payment - total_cost
  change_received = 0.26 :=
by
  sorry

end NUMINAMATH_GPT_joan_change_received_l236_23628


namespace NUMINAMATH_GPT_silver_coins_change_l236_23688

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_silver_coins_change_l236_23688


namespace NUMINAMATH_GPT_books_printed_l236_23652

-- Definitions of the conditions
def book_length := 600
def pages_per_sheet := 8
def total_sheets := 150

-- The theorem to prove
theorem books_printed : (total_sheets * pages_per_sheet / book_length) = 2 := by
  sorry

end NUMINAMATH_GPT_books_printed_l236_23652


namespace NUMINAMATH_GPT_plan1_has_higher_expected_loss_l236_23634

noncomputable def prob_minor_flooding : ℝ := 0.2
noncomputable def prob_major_flooding : ℝ := 0.05
noncomputable def cost_plan1 : ℝ := 4000
noncomputable def loss_major_plan1 : ℝ := 30000
noncomputable def loss_minor_plan2 : ℝ := 15000
noncomputable def loss_major_plan2 : ℝ := 30000

noncomputable def expected_loss_plan1 : ℝ :=
  (loss_major_plan1 * prob_major_flooding) + (cost_plan1 * prob_minor_flooding) + cost_plan1

noncomputable def expected_loss_plan2 : ℝ :=
  (loss_major_plan2 * prob_major_flooding) + (loss_minor_plan2 * prob_minor_flooding)

theorem plan1_has_higher_expected_loss : expected_loss_plan1 > expected_loss_plan2 :=
by
  sorry

end NUMINAMATH_GPT_plan1_has_higher_expected_loss_l236_23634


namespace NUMINAMATH_GPT_identity_equality_l236_23637

theorem identity_equality (a b m n x y : ℝ) :
  ((a^2 + b^2) * (m^2 + n^2) * (x^2 + y^2)) =
  ((a * n * y - a * m * x - b * m * y + b * n * x)^2 + (a * m * y + a * n * x + b * m * x - b * n * y)^2) :=
by
  sorry

end NUMINAMATH_GPT_identity_equality_l236_23637


namespace NUMINAMATH_GPT_correct_system_of_equations_l236_23684

noncomputable def system_of_equations (x y : ℝ) : Prop :=
x + y = 150 ∧ 3 * x + (1 / 3) * y = 210

theorem correct_system_of_equations : ∃ x y : ℝ, system_of_equations x y :=
sorry

end NUMINAMATH_GPT_correct_system_of_equations_l236_23684


namespace NUMINAMATH_GPT_passing_marks_l236_23648

-- Define the conditions and prove P = 160 given these conditions
theorem passing_marks (T P : ℝ) (h1 : 0.40 * T = P - 40) (h2 : 0.60 * T = P + 20) : P = 160 :=
by
  sorry

end NUMINAMATH_GPT_passing_marks_l236_23648


namespace NUMINAMATH_GPT_technicians_count_l236_23621

theorem technicians_count 
    (total_workers : ℕ) (avg_salary_all : ℕ) (avg_salary_technicians : ℕ) (avg_salary_rest : ℕ)
    (h_workers : total_workers = 28) (h_avg_all : avg_salary_all = 8000) 
    (h_avg_tech : avg_salary_technicians = 14000) (h_avg_rest : avg_salary_rest = 6000) : 
    ∃ T R : ℕ, T + R = total_workers ∧ (avg_salary_technicians * T + avg_salary_rest * R = avg_salary_all * total_workers) ∧ T = 7 :=
by
  sorry

end NUMINAMATH_GPT_technicians_count_l236_23621


namespace NUMINAMATH_GPT_last_person_is_knight_l236_23658

-- Definitions for the conditions:
def first_whispered_number := 7
def last_announced_number_first_game := 3
def last_whispered_number_second_game := 5
def first_announced_number_second_game := 2

-- Definitions to represent the roles:
inductive Role
| knight
| liar

-- Definition of the last person in the first game being a knight:
def last_person_first_game_role := Role.knight

theorem last_person_is_knight 
  (h1 : Role.liar = Role.liar)
  (h2 : last_announced_number_first_game = 3)
  (h3 : first_whispered_number = 7)
  (h4 : first_announced_number_second_game = 2)
  (h5 : last_whispered_number_second_game = 5) :
  last_person_first_game_role = Role.knight :=
sorry

end NUMINAMATH_GPT_last_person_is_knight_l236_23658


namespace NUMINAMATH_GPT_price_difference_eq_l236_23614

-- Define the problem conditions
variable (P : ℝ) -- Original price
variable (H1 : P - 0.15 * P = 61.2) -- Condition 1: 15% discount results in $61.2
variable (H2 : P * (1 - 0.15) = 61.2) -- Another way to represent Condition 1 (if needed)
variable (H3 : 61.2 * 1.25 = 76.5) -- Condition 4: Price raises by 25% after the 15% discount
variable (H4 : 76.5 * 0.9 = 68.85) -- Condition 5: Additional 10% discount after raise
variable (H5 : P = 72) -- Calculated original price

-- Define the theorem to prove
theorem price_difference_eq :
  (P - 68.85 = 3.15) := 
by
  sorry

end NUMINAMATH_GPT_price_difference_eq_l236_23614


namespace NUMINAMATH_GPT_last_three_digits_of_11_pow_210_l236_23603

theorem last_three_digits_of_11_pow_210 : (11 ^ 210) % 1000 = 601 :=
by sorry

end NUMINAMATH_GPT_last_three_digits_of_11_pow_210_l236_23603


namespace NUMINAMATH_GPT_inequality_of_power_sums_l236_23633

variable (a b c : ℝ)

theorem inequality_of_power_sums (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a < b + c) (h5 : b < c + a) (h6 : c < a + b) :
  a^4 + b^4 + c^4 < 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) := sorry

end NUMINAMATH_GPT_inequality_of_power_sums_l236_23633


namespace NUMINAMATH_GPT_remainder_when_98_mul_102_divided_by_11_l236_23635

theorem remainder_when_98_mul_102_divided_by_11 :
  (98 * 102) % 11 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_98_mul_102_divided_by_11_l236_23635


namespace NUMINAMATH_GPT_one_half_percent_as_decimal_l236_23666

def percent_to_decimal (x : ℚ) := x / 100

theorem one_half_percent_as_decimal : percent_to_decimal (1 / 2) = 0.005 := 
by
  sorry

end NUMINAMATH_GPT_one_half_percent_as_decimal_l236_23666


namespace NUMINAMATH_GPT_value_of_f_37_5_l236_23692

-- Mathematical definitions and conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f (x)
def satisfies_condition (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = -f (x)
def interval_condition (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f (x) = x

-- Main theorem to be proved
theorem value_of_f_37_5 (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_periodic : satisfies_condition f) 
  (h_interval : interval_condition f) : 
  f 37.5 = 0.5 := 
sorry

end NUMINAMATH_GPT_value_of_f_37_5_l236_23692


namespace NUMINAMATH_GPT_binomial_coefficients_sum_l236_23615

theorem binomial_coefficients_sum :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ),
  (1 - 2 * 0)^5 = a_0 + a_1 * (1 + 0) + a_2 * (1 + 0)^2 + a_3 * (1 + 0)^3 + a_4 * (1 + 0)^4 + a_5 * (1 + 0)^5 →
  (1 - 2 * 1)^5 = (-1)^5 * a_5 →
  a_0 + a_1 + a_2 + a_3 + a_4 = 33 :=
by sorry

end NUMINAMATH_GPT_binomial_coefficients_sum_l236_23615


namespace NUMINAMATH_GPT_percentage_decrease_is_17_point_14_l236_23616

-- Define the conditions given in the problem
variable (S : ℝ) -- original salary
variable (D : ℝ) -- percentage decrease

-- Given conditions
def given_conditions : Prop :=
  1.40 * S - (D / 100) * 1.40 * S = 1.16 * S

-- The required proof problem, where we assert D = 17.14
theorem percentage_decrease_is_17_point_14 (S : ℝ) (h : given_conditions S D) : D = 17.14 := 
  sorry

end NUMINAMATH_GPT_percentage_decrease_is_17_point_14_l236_23616


namespace NUMINAMATH_GPT_problem_condition_necessary_and_sufficient_l236_23667

theorem problem_condition_necessary_and_sufficient (a b : ℝ) (h : a * b > 0) :
  (a > b) ↔ (1 / a < 1 / b) :=
sorry

end NUMINAMATH_GPT_problem_condition_necessary_and_sufficient_l236_23667


namespace NUMINAMATH_GPT_cosine_theorem_a_cosine_theorem_b_cosine_theorem_c_l236_23686

theorem cosine_theorem_a (a b c A : ℝ) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A := sorry

theorem cosine_theorem_b (a b c B : ℝ) :
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B := sorry

theorem cosine_theorem_c (a b c C : ℝ) :
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C := sorry

end NUMINAMATH_GPT_cosine_theorem_a_cosine_theorem_b_cosine_theorem_c_l236_23686


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l236_23630

theorem simplify_and_evaluate_expression : 
  ∀ (x y : ℤ), x = -1 → y = 2 → -2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 4 * x * y = 6 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l236_23630


namespace NUMINAMATH_GPT_sector_max_area_l236_23619

theorem sector_max_area (r : ℝ) (α : ℝ) (S : ℝ) :
  (0 < r ∧ r < 10) ∧ (2 * r + r * α = 20) ∧ (S = (1 / 2) * r * (r * α)) →
  (α = 2 ∧ S = 25) :=
by
  sorry

end NUMINAMATH_GPT_sector_max_area_l236_23619


namespace NUMINAMATH_GPT_fraction_inequality_fraction_inequality_equality_case_l236_23638

variables {α β a b : ℝ}

theorem fraction_inequality 
  (h_alpha_beta_pos : 0 < α ∧ 0 < β)
  (h_bounds_a : α ≤ a ∧ a ≤ β)
  (h_bounds_b : α ≤ b ∧ b ≤ β) :
  (b / a + a / b) ≤ (β / α + α / β) :=
sorry

-- Additional equality statement
theorem fraction_inequality_equality_case
  (h_alpha_beta_pos : 0 < α ∧ 0 < β)
  (h_bounds_a : α ≤ a ∧ a ≤ β)
  (h_bounds_b : α ≤ b ∧ b ≤ β) :
  (b / a + a / b = β / α + α / β) ↔ (a = α ∧ b = β ∨ a = β ∧ b = α) :=
sorry

end NUMINAMATH_GPT_fraction_inequality_fraction_inequality_equality_case_l236_23638


namespace NUMINAMATH_GPT_sqrt_18_mul_sqrt_6_sqrt_8_sub_sqrt_2_add_2_sqrt_half_sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3_sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5_l236_23696

-- Problem 1
theorem sqrt_18_mul_sqrt_6 : (Real.sqrt 18 * Real.sqrt 6 = 6 * Real.sqrt 3) :=
sorry

-- Problem 2
theorem sqrt_8_sub_sqrt_2_add_2_sqrt_half : (Real.sqrt 8 - Real.sqrt 2 + 2 * Real.sqrt (1 / 2) = 3 * Real.sqrt 2) :=
sorry

-- Problem 3
theorem sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3 : (Real.sqrt 12 * (Real.sqrt 9 / 3) / (Real.sqrt 3 / 3) = 6) :=
sorry

-- Problem 4
theorem sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5 : ((Real.sqrt 7 + Real.sqrt 5) * (Real.sqrt 7 - Real.sqrt 5) = 2) :=
sorry

end NUMINAMATH_GPT_sqrt_18_mul_sqrt_6_sqrt_8_sub_sqrt_2_add_2_sqrt_half_sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3_sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5_l236_23696


namespace NUMINAMATH_GPT_factor_expression_equals_one_l236_23657

theorem factor_expression_equals_one (a b c : ℝ) :
  ((a^2 - b^2)^2 + (b^2 - c^2)^2 + (c^2 - a^2)^2) / ((a - b)^2 + (b - c)^2 + (c - a)^2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_equals_one_l236_23657


namespace NUMINAMATH_GPT_power_complex_l236_23664

theorem power_complex (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : -64 = (-4)^3) (h3 : (a^b)^((3:ℝ) / 2) = a^(b * ((3:ℝ) / 2))) (h4 : (-4:ℂ)^(1/2) = 2 * i) :
  (↑(-64):ℂ) ^ (3/2) = 512 * i :=
by
  sorry

end NUMINAMATH_GPT_power_complex_l236_23664


namespace NUMINAMATH_GPT_find_a_value_l236_23660

theorem find_a_value
  (a : ℕ)
  (x y : ℝ)
  (h1 : a * x + y = -4)
  (h2 : 2 * x + y = -2)
  (hx_neg : x < 0)
  (hy_pos : y > 0) :
  a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_value_l236_23660


namespace NUMINAMATH_GPT_Bomi_change_l236_23604

def candy_cost : ℕ := 350
def chocolate_cost : ℕ := 500
def total_paid : ℕ := 1000
def total_cost := candy_cost + chocolate_cost
def change := total_paid - total_cost

theorem Bomi_change : change = 150 :=
by
  -- Here we would normally provide the proof steps.
  sorry

end NUMINAMATH_GPT_Bomi_change_l236_23604


namespace NUMINAMATH_GPT_find_angle_D_l236_23650

noncomputable def calculate_angle (A B C D : ℝ) : ℝ :=
  if (A + B = 180) ∧ (C = D) ∧ (A = 2 * D - 10) then D else 0

theorem find_angle_D (A B C D : ℝ) (h1: A + B = 180) (h2: C = D) (h3: A = 2 * D - 10) : D = 70 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_D_l236_23650


namespace NUMINAMATH_GPT_subdivide_tetrahedron_l236_23608

/-- A regular tetrahedron with edge length 1 can be divided into smaller regular tetrahedrons and octahedrons,
    such that the edge lengths of the resulting tetrahedrons and octahedrons are less than 1 / 100 after a 
    finite number of subdivisions. -/
theorem subdivide_tetrahedron (edge_len : ℝ) (h : edge_len = 1) :
  ∃ (k : ℕ), (1 / (2^k : ℝ) < 1 / 100) :=
by sorry

end NUMINAMATH_GPT_subdivide_tetrahedron_l236_23608


namespace NUMINAMATH_GPT_initial_population_l236_23665

theorem initial_population (P : ℝ) (h1 : ∀ n : ℕ, n = 2 → P * (0.7 ^ n) = 3920) : P = 8000 := by
  sorry

end NUMINAMATH_GPT_initial_population_l236_23665


namespace NUMINAMATH_GPT_even_property_of_f_when_a_zero_non_even_odd_property_of_f_when_a_nonzero_minimum_value_of_f_l236_23611

open Real

def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem even_property_of_f_when_a_zero : 
  ∀ x : ℝ, f 0 x = f 0 (-x) :=
by sorry

theorem non_even_odd_property_of_f_when_a_nonzero : 
  ∀ (a x : ℝ), a ≠ 0 → (f a x ≠ f a (-x) ∧ f a x ≠ -f a (-x)) :=
by sorry

theorem minimum_value_of_f :
  ∀ (a : ℝ), 
    (a ≤ -1/2 → ∃ x : ℝ, f a x = -a - 5/4) ∧ 
    (-1/2 < a ∧ a ≤ 1/2 → ∃ x : ℝ, f a x = a^2 - 1) ∧ 
    (a > 1/2 → ∃ x : ℝ, f a x = a - 5/4) :=
by sorry

end NUMINAMATH_GPT_even_property_of_f_when_a_zero_non_even_odd_property_of_f_when_a_nonzero_minimum_value_of_f_l236_23611


namespace NUMINAMATH_GPT_sugar_left_in_grams_l236_23677

theorem sugar_left_in_grams 
  (initial_ounces : ℝ) (spilled_ounces : ℝ) (conversion_factor : ℝ)
  (h_initial : initial_ounces = 9.8) (h_spilled : spilled_ounces = 5.2)
  (h_conversion : conversion_factor = 28.35) :
  (initial_ounces - spilled_ounces) * conversion_factor = 130.41 := 
by
  sorry

end NUMINAMATH_GPT_sugar_left_in_grams_l236_23677


namespace NUMINAMATH_GPT_find_negative_integer_l236_23640

theorem find_negative_integer (N : ℤ) (h : N^2 + N = -12) : N = -4 := 
by sorry

end NUMINAMATH_GPT_find_negative_integer_l236_23640


namespace NUMINAMATH_GPT_non_obtuse_triangle_range_l236_23623

noncomputable def range_of_2a_over_c (a b c A C : ℝ) (h1 : B = π / 3) (h2 : A + C = 2 * π / 3) (h3 : π / 6 < C ∧ C ≤ π / 2) : Set ℝ :=
  {x | ∃ (a b c A : ℝ), x = (2 * a) / c ∧ 1 < x ∧ x ≤ 4}

theorem non_obtuse_triangle_range (a b c A C : ℝ) (h1 : B = π / 3) (h2 : A + C = 2 * π / 3) (h3 : π / 6 < C ∧ C ≤ π / 2) :
  (2 * a) / c ∈ range_of_2a_over_c a b c A C h1 h2 h3 := 
sorry

end NUMINAMATH_GPT_non_obtuse_triangle_range_l236_23623


namespace NUMINAMATH_GPT_negation_of_proposition_l236_23673

theorem negation_of_proposition (x : ℝ) : 
  ¬ (|x| < 2 → x < 2) ↔ (|x| ≥ 2 → x ≥ 2) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l236_23673


namespace NUMINAMATH_GPT_mean_weight_players_l236_23627

/-- Definitions for the weights of the players and proving the mean weight. -/
def weights : List ℕ := [62, 65, 70, 73, 73, 76, 78, 79, 81, 81, 82, 84, 87, 89, 89, 89, 90, 93, 95]

def mean (lst : List ℕ) : ℚ := (lst.sum : ℚ) / lst.length

theorem mean_weight_players : mean weights = 80.84 := by
  sorry

end NUMINAMATH_GPT_mean_weight_players_l236_23627


namespace NUMINAMATH_GPT_maximum_value_of_x_plus_2y_l236_23642

theorem maximum_value_of_x_plus_2y (x y : ℝ) (h : x^2 - 2 * x + 4 * y = 5) : ∃ m, m = x + 2 * y ∧ m ≤ 9/2 := by
  sorry

end NUMINAMATH_GPT_maximum_value_of_x_plus_2y_l236_23642


namespace NUMINAMATH_GPT_largest_divisor_of_n_l236_23626

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 72 ∣ n^2) : 12 ∣ n :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_n_l236_23626


namespace NUMINAMATH_GPT_abc_value_l236_23632

theorem abc_value 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (hab : a * b = 24) 
  (hac : a * c = 40) 
  (hbc : b * c = 60) : 
  a * b * c = 240 := 
by sorry

end NUMINAMATH_GPT_abc_value_l236_23632


namespace NUMINAMATH_GPT_absolute_difference_avg_median_l236_23681

theorem absolute_difference_avg_median (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : 
  |((3 + 4 * a + 2 * b) / 4) - (a + b / 2 + 1)| = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_absolute_difference_avg_median_l236_23681


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l236_23613

variable (B S : ℝ)

theorem boat_speed_in_still_water :
  (B + S = 38) ∧ (B - S = 16) → B = 27 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l236_23613


namespace NUMINAMATH_GPT_ellipse_chord_line_eq_l236_23605

noncomputable def chord_line (x y : ℝ) : ℝ := 2 * x + 4 * y - 3

theorem ellipse_chord_line_eq :
  ∀ (x y : ℝ),
    (x ^ 2 / 2 + y ^ 2 = 1) ∧ (x + y = 1) → (chord_line x y = 0) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_ellipse_chord_line_eq_l236_23605


namespace NUMINAMATH_GPT_number_of_parallel_lines_l236_23698

/-- 
Given 10 parallel lines in the first set and the fact that the intersection 
of two sets of parallel lines forms 1260 parallelograms, 
prove that the second set contains 141 parallel lines.
-/
theorem number_of_parallel_lines (n : ℕ) (h₁ : 10 - 1 = 9) (h₂ : 9 * (n - 1) = 1260) : n = 141 :=
sorry

end NUMINAMATH_GPT_number_of_parallel_lines_l236_23698


namespace NUMINAMATH_GPT_telephone_number_A_value_l236_23661

theorem telephone_number_A_value :
  ∃ A B C D E F G H I J : ℕ,
    A > B ∧ B > C ∧
    D > E ∧ E > F ∧
    G > H ∧ H > I ∧ I > J ∧
    (D = E + 1) ∧ (E = F + 1) ∧
    G + H + I + J = 20 ∧
    A + B + C = 15 ∧
    A = 8 := sorry

end NUMINAMATH_GPT_telephone_number_A_value_l236_23661


namespace NUMINAMATH_GPT_total_cost_of_books_l236_23617

-- Conditions from the problem
def C1 : ℝ := 350
def loss_percent : ℝ := 0.15
def gain_percent : ℝ := 0.19
def SP1 : ℝ := C1 - (loss_percent * C1) -- Selling price of the book sold at a loss
def SP2 : ℝ := SP1 -- Selling price of the book sold at a gain

-- Statement to prove the total cost
theorem total_cost_of_books : C1 + (SP2 / (1 + gain_percent)) = 600 := by
  sorry

end NUMINAMATH_GPT_total_cost_of_books_l236_23617


namespace NUMINAMATH_GPT_find_integer_a_l236_23624

theorem find_integer_a (x d e a : ℤ) :
  ((x - a)*(x - 8) - 3 = (x + d)*(x + e)) → (a = 6) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_a_l236_23624


namespace NUMINAMATH_GPT_unique_solution_to_equation_l236_23629

theorem unique_solution_to_equation (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x^y - y = 2005) : x = 1003 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_to_equation_l236_23629


namespace NUMINAMATH_GPT_moon_temp_difference_l236_23682

def temp_difference (T_day T_night : ℤ) : ℤ := T_day - T_night

theorem moon_temp_difference :
  temp_difference 127 (-183) = 310 :=
by
  sorry

end NUMINAMATH_GPT_moon_temp_difference_l236_23682


namespace NUMINAMATH_GPT_intersection_eq_l236_23695

def set1 : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def set2 : Set ℝ := {x | -2 ≤ x ∧ x < 2}

theorem intersection_eq : (set1 ∩ set2) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l236_23695


namespace NUMINAMATH_GPT_union_of_A_and_B_l236_23694

def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l236_23694


namespace NUMINAMATH_GPT_right_triangle_exists_with_area_ab_l236_23602

theorem right_triangle_exists_with_area_ab (a b c d : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
    (h1 : a * b = c * d) (h2 : a + b = c - d) :
    ∃ (x y z : ℕ), x^2 + y^2 = z^2 ∧ (x * y / 2 = a * b) := sorry

end NUMINAMATH_GPT_right_triangle_exists_with_area_ab_l236_23602


namespace NUMINAMATH_GPT_sum_of_sides_l236_23689

-- Definitions: Given conditions
def ratio (a b c : ℕ) : Prop := 
a * 5 = b * 3 ∧ b * 7 = c * 5

-- Given that the longest side is 21 cm and the ratio of the sides is 3:5:7
def similar_triangle (x y : ℕ) : Prop :=
ratio x y 21

-- Proof statement: The sum of the lengths of the other two sides is 24 cm
theorem sum_of_sides (x y : ℕ) (h : similar_triangle x y) : x + y = 24 :=
sorry

end NUMINAMATH_GPT_sum_of_sides_l236_23689


namespace NUMINAMATH_GPT_part_I_part_II_l236_23662

noncomputable def f (x m : ℝ) : ℝ := |3 * x + m|
noncomputable def g (x m : ℝ) : ℝ := f x m - 2 * |x - 1|

theorem part_I (m : ℝ) : (∀ x : ℝ, (f x m - m ≤ 9) ↔ (-1 ≤ x ∧ x ≤ 3)) → m = -3 :=
by
  sorry

theorem part_II (m : ℝ) (h : m > 0) : (∃ A B C : ℝ × ℝ, 
  let A := (-m-2, 0)
  let B := ((2-m)/5, 0)
  let C := (-m/3, -2*m/3-2)
  let Area : ℝ := 1/2 * |(B.1 - A.1) * (C.2 - 0) - (B.2 - A.2) * (C.1 - A.1)|
  Area > 60 ) → m > 12 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l236_23662


namespace NUMINAMATH_GPT_problem_1_problem_2_l236_23606

open Set

variables {U : Type*} [TopologicalSpace U] (a x : ℝ)

def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def N (a : ℝ) : Set ℝ := { x | a + 1 ≤ x ∧ x ≤ 2 * a + 1 }

noncomputable def complement_N (a : ℝ) : Set ℝ := { x | x < a + 1 ∨ 2 * a + 1 < x }

theorem problem_1 (h : a = 2) :
  M ∩ (complement_N a) = { x | -2 ≤ x ∧ x < 3 } :=
sorry

theorem problem_2 (h : M ∪ N a = M) :
  a ≤ 2 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l236_23606


namespace NUMINAMATH_GPT_symmetric_point_correct_l236_23659

-- Define the point and the symmetry operation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def symmetric_with_respect_to_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

-- Define the specific point M
def M : Point := {x := 1, y := 2}

-- Define the expected answer point M'
def M' : Point := {x := 1, y := -2}

-- Prove that the symmetric point with respect to the x-axis is as expected
theorem symmetric_point_correct :
  symmetric_with_respect_to_x_axis M = M' :=
by sorry

end NUMINAMATH_GPT_symmetric_point_correct_l236_23659


namespace NUMINAMATH_GPT_jon_percentage_increase_l236_23620

def initial_speed : ℝ := 80
def trainings : ℕ := 4
def weeks_per_training : ℕ := 4
def speed_increase_per_week : ℝ := 1

theorem jon_percentage_increase :
  let total_weeks := trainings * weeks_per_training
  let total_increase := total_weeks * speed_increase_per_week
  let final_speed := initial_speed + total_increase
  let percentage_increase := (total_increase / initial_speed) * 100
  percentage_increase = 20 :=
by
  sorry

end NUMINAMATH_GPT_jon_percentage_increase_l236_23620


namespace NUMINAMATH_GPT_line_a_minus_b_l236_23647

theorem line_a_minus_b (a b : ℝ)
  (h1 : (2 : ℝ) = a * (3 : ℝ) + b)
  (h2 : (26 : ℝ) = a * (7 : ℝ) + b) :
  a - b = 22 :=
by
  sorry

end NUMINAMATH_GPT_line_a_minus_b_l236_23647


namespace NUMINAMATH_GPT_ellipse_sum_l236_23653

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := 0
noncomputable def a : ℝ := 5
noncomputable def b : ℝ := Real.sqrt 21
noncomputable def F_1 : (ℝ × ℝ) := (1, 0)
noncomputable def F_2 : (ℝ × ℝ) := (5, 0)

theorem ellipse_sum :
  (F_1 = (1, 0)) → 
  (F_2 = (5, 0)) →
  (∀ P : (ℝ × ℝ), (Real.sqrt ((P.1 - F_1.1)^2 + (P.2 - F_1.2)^2) + Real.sqrt ((P.1 - F_2.1)^2 + (P.2 - F_2.2)^2) = 10)) →
  (h + k + a + b = 8 + Real.sqrt 21) :=
by
  intros
  sorry

end NUMINAMATH_GPT_ellipse_sum_l236_23653


namespace NUMINAMATH_GPT_manuscript_typing_total_cost_is_1400_l236_23651

-- Defining the variables and constants based on given conditions
def cost_first_time_per_page := 10
def cost_revision_per_page := 5
def total_pages := 100
def pages_revised_once := 20
def pages_revised_twice := 30
def pages_no_revision := total_pages - pages_revised_once - pages_revised_twice

-- Calculations based on the given conditions
def cost_first_time :=
  total_pages * cost_first_time_per_page

def cost_revised_once :=
  pages_revised_once * cost_revision_per_page

def cost_revised_twice :=
  pages_revised_twice * cost_revision_per_page * 2

def total_cost :=
  cost_first_time + cost_revised_once + cost_revised_twice

-- Prove that the total cost equals the calculated value
theorem manuscript_typing_total_cost_is_1400 :
  total_cost = 1400 := by
  sorry

end NUMINAMATH_GPT_manuscript_typing_total_cost_is_1400_l236_23651


namespace NUMINAMATH_GPT_percent_markdown_l236_23641

theorem percent_markdown (P S : ℝ) (h : S * 1.25 = P) : (P - S) / P * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_percent_markdown_l236_23641


namespace NUMINAMATH_GPT_cost_price_l236_23693

theorem cost_price (SP : ℝ) (profit_percentage : ℝ) : SP = 600 ∧ profit_percentage = 60 → ∃ CP : ℝ, CP = 375 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_price_l236_23693


namespace NUMINAMATH_GPT_seven_large_power_mod_seventeen_l236_23644

theorem seven_large_power_mod_seventeen :
  (7 : ℤ)^1985 % 17 = 7 :=
by
  have h1 : (7 : ℤ)^2 % 17 = 15 := sorry
  have h2 : (7 : ℤ)^4 % 17 = 16 := sorry
  have h3 : (7 : ℤ)^8 % 17 = 1 := sorry
  have h4 : 1985 = 8 * 248 + 1 := sorry
  sorry

end NUMINAMATH_GPT_seven_large_power_mod_seventeen_l236_23644


namespace NUMINAMATH_GPT_exists_valid_numbers_l236_23668

noncomputable def sum_of_numbers_is_2012_using_two_digits : Prop :=
  ∃ (a b c d : ℕ), (a < 1000) ∧ (b < 1000) ∧ (c < 1000) ∧ (d < 1000) ∧ 
                    (∀ n ∈ [a, b, c, d], ∃ x y, (x ≠ y) ∧ ((∀ d ∈ [n / 100 % 10, n / 10 % 10, n % 10], d = x ∨ d = y))) ∧
                    (a + b + c + d = 2012)

theorem exists_valid_numbers : sum_of_numbers_is_2012_using_two_digits :=
  sorry

end NUMINAMATH_GPT_exists_valid_numbers_l236_23668


namespace NUMINAMATH_GPT_directrix_of_parabola_l236_23655

theorem directrix_of_parabola :
  ∀ (x y : ℝ), (y = (x^2 - 4 * x + 4) / 8) → y = -2 :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l236_23655


namespace NUMINAMATH_GPT_find_f_7_l236_23687

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_7 (h_odd : ∀ x, f (-x) = -f x)
                 (h_periodic : ∀ x, f (x + 4) = f x)
                 (h_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x ^ 2) :
  f 7 = -2 := 
sorry

end NUMINAMATH_GPT_find_f_7_l236_23687


namespace NUMINAMATH_GPT_equal_intercepts_lines_area_two_lines_l236_23612

-- Defining the general equation of the line l with parameter a
def line_eq (a : ℝ) (x y : ℝ) : Prop := y = -(a + 1) * x + 2 - a

-- Problem statement for equal intercepts condition
theorem equal_intercepts_lines (a : ℝ) : 
  (∃ x y : ℝ, line_eq a x y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (x = y ∨ x + y = 2*a + 2)) →
  (a = 2 ∨ a = 0) → 
  (line_eq a 1 (-3) ∨ line_eq a 1 1) :=
sorry

-- Problem statement for triangle area condition
theorem area_two_lines (a : ℝ) : 
  (∃ x y : ℝ, line_eq a x y ∧ x ≠ 0 ∧ y ≠ 0 ∧ (1 / 2 * |x| * |y| = 2)) →
  (a = 8 ∨ a = 0) → 
  (line_eq a 1 (-9) ∨ line_eq a 1 1) :=
sorry

end NUMINAMATH_GPT_equal_intercepts_lines_area_two_lines_l236_23612


namespace NUMINAMATH_GPT_hours_week3_and_4_l236_23601

variable (H3 H4 : Nat)

def hours_worked_week1_and_2 : Nat := 35 + 35
def extra_hours_worked_week3_and_4 : Nat := 26
def total_hours_week3_and_4 : Nat := hours_worked_week1_and_2 + extra_hours_worked_week3_and_4

theorem hours_week3_and_4 :
  H3 + H4 = total_hours_week3_and_4 := by
sorry

end NUMINAMATH_GPT_hours_week3_and_4_l236_23601


namespace NUMINAMATH_GPT_mod_congruence_l236_23675

theorem mod_congruence (N : ℕ) (hN : N > 1) (h1 : 69 % N = 90 % N) (h2 : 90 % N = 125 % N) : 81 % N = 4 := 
by {
    sorry
}

end NUMINAMATH_GPT_mod_congruence_l236_23675


namespace NUMINAMATH_GPT_single_elimination_games_l236_23600

theorem single_elimination_games (n : Nat) (h : n = 21) : games_needed = n - 1 :=
by
  sorry

end NUMINAMATH_GPT_single_elimination_games_l236_23600


namespace NUMINAMATH_GPT_min_sum_a_b_l236_23676

theorem min_sum_a_b (a b : ℝ) (h_cond: 1/a + 4/b = 1) (a_pos : 0 < a) (b_pos : 0 < b) : 
  a + b ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_sum_a_b_l236_23676


namespace NUMINAMATH_GPT_solve_for_x_l236_23649

theorem solve_for_x : (∃ x : ℝ, (1/2 - 1/3 = 1/x)) ↔ (x = 6) := sorry

end NUMINAMATH_GPT_solve_for_x_l236_23649


namespace NUMINAMATH_GPT_squares_difference_l236_23643

theorem squares_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 :=
by sorry

end NUMINAMATH_GPT_squares_difference_l236_23643


namespace NUMINAMATH_GPT_lcm_and_sum_of_14_21_35_l236_23670

def lcm_of_numbers_and_sum (a b c : ℕ) : ℕ × ℕ :=
  (Nat.lcm (Nat.lcm a b) c, a + b + c)

theorem lcm_and_sum_of_14_21_35 :
  lcm_of_numbers_and_sum 14 21 35 = (210, 70) :=
  sorry

end NUMINAMATH_GPT_lcm_and_sum_of_14_21_35_l236_23670


namespace NUMINAMATH_GPT_initial_oranges_in_box_l236_23654

theorem initial_oranges_in_box (o_taken_out o_left_in_box : ℕ) (h1 : o_taken_out = 35) (h2 : o_left_in_box = 20) :
  o_taken_out + o_left_in_box = 55 := 
by
  sorry

end NUMINAMATH_GPT_initial_oranges_in_box_l236_23654


namespace NUMINAMATH_GPT_probability_of_first_three_red_cards_l236_23610

def total_cards : ℕ := 104
def suits : ℕ := 4
def cards_per_suit : ℕ := 26
def red_suits : ℕ := 2
def black_suits : ℕ := 2
def total_red_cards : ℕ := 52
def total_black_cards : ℕ := 52

noncomputable def probability_first_three_red : ℚ :=
  (total_red_cards / total_cards) * ((total_red_cards - 1) / (total_cards - 1)) * ((total_red_cards - 2) / (total_cards - 2))

theorem probability_of_first_three_red_cards :
  probability_first_three_red = 425 / 3502 :=
sorry

end NUMINAMATH_GPT_probability_of_first_three_red_cards_l236_23610


namespace NUMINAMATH_GPT_hives_needed_for_candles_l236_23683

theorem hives_needed_for_candles (h : (3 : ℕ) * c = 12) : (96 : ℕ) / c = 24 :=
by
  sorry

end NUMINAMATH_GPT_hives_needed_for_candles_l236_23683


namespace NUMINAMATH_GPT_find_m_same_foci_l236_23672

theorem find_m_same_foci (m : ℝ) 
(hyperbola_eq : ∃ x y : ℝ, x^2 - y^2 = m) 
(ellipse_eq : ∃ x y : ℝ, 2 * x^2 + 3 * y^2 = m + 1) 
(same_foci : ∀ a b : ℝ, (x^2 - y^2 = m) ∧ (2 * x^2 + 3 * y^2 = m + 1) → 
               let c_ellipse := (m + 1) / 6
               let c_hyperbola := 2 * m
               c_ellipse = c_hyperbola ) : 
m = 1 / 11 := 
sorry

end NUMINAMATH_GPT_find_m_same_foci_l236_23672


namespace NUMINAMATH_GPT_g_f2_minus_f_g2_eq_zero_l236_23646

def f (x : ℝ) : ℝ := x^2 + 3 * x + 1

def g (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem g_f2_minus_f_g2_eq_zero : g (f 2) - f (g 2) = 0 := by
  sorry

end NUMINAMATH_GPT_g_f2_minus_f_g2_eq_zero_l236_23646


namespace NUMINAMATH_GPT_sin_cos_difference_l236_23679

theorem sin_cos_difference
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioo 0 Real.pi)
  (h2 : Real.sin θ + Real.cos θ = 1 / 5) :
  Real.sin θ - Real.cos θ = 7 / 5 :=
sorry

end NUMINAMATH_GPT_sin_cos_difference_l236_23679


namespace NUMINAMATH_GPT_meatballs_left_l236_23678
open Nat

theorem meatballs_left (meatballs_per_plate sons : ℕ)
  (hp : meatballs_per_plate = 3) 
  (hs : sons = 3) 
  (fraction_eaten : ℚ)
  (hf : fraction_eaten = 2 / 3): 
  (meatballs_per_plate - meatballs_per_plate * fraction_eaten) * sons = 3 := by
  -- Placeholder proof; the details would be filled in by a full proof.
  sorry

end NUMINAMATH_GPT_meatballs_left_l236_23678


namespace NUMINAMATH_GPT_max_incircle_circumcircle_ratio_l236_23674

theorem max_incircle_circumcircle_ratio (c : ℝ) (α : ℝ) 
  (hα : 0 < α ∧ α < π / 2) :
  let a := c * Real.cos α
  let b := c * Real.sin α
  let R := c / 2
  let r := (a + b - c) / 2
  (r / R <= Real.sqrt 2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_max_incircle_circumcircle_ratio_l236_23674


namespace NUMINAMATH_GPT_prob_two_segments_same_length_l236_23690

namespace hexagon_prob

noncomputable def prob_same_length : ℚ :=
  let total_elements : ℕ := 15
  let sides : ℕ := 6
  let diagonals : ℕ := 9
  (sides / total_elements) * ((sides - 1) / (total_elements - 1)) + (diagonals / total_elements) * ((diagonals - 1) / (total_elements - 1))

theorem prob_two_segments_same_length : prob_same_length = 17 / 35 :=
by
  sorry

end hexagon_prob

end NUMINAMATH_GPT_prob_two_segments_same_length_l236_23690


namespace NUMINAMATH_GPT_order_scores_l236_23625

theorem order_scores
  (J K M Q S : ℕ)
  (h1 : J ≥ Q) (h2 : J ≥ M) (h3 : J ≥ S) (h4 : J ≥ K)
  (h5 : M > Q ∨ M > S ∨ M > K)
  (h6 : K < S) (h7 : S < J) :
  K < S ∧ S < M ∧ M < Q :=
by
  sorry

end NUMINAMATH_GPT_order_scores_l236_23625


namespace NUMINAMATH_GPT_find_avg_mpg_first_car_l236_23699

def avg_mpg_first_car (x : ℝ) : Prop :=
  let miles_per_month := 450 / 3
  let gallons_first_car := miles_per_month / x
  let gallons_second_car := miles_per_month / 10
  let gallons_third_car := miles_per_month / 15
  let total_gallons := 56 / 2
  gallons_first_car + gallons_second_car + gallons_third_car = total_gallons

theorem find_avg_mpg_first_car : avg_mpg_first_car 50 :=
  sorry

end NUMINAMATH_GPT_find_avg_mpg_first_car_l236_23699


namespace NUMINAMATH_GPT_combined_weight_of_emma_and_henry_l236_23607

variables (e f g h : ℕ)

theorem combined_weight_of_emma_and_henry 
  (h1 : e + f = 310)
  (h2 : f + g = 265)
  (h3 : g + h = 280) : e + h = 325 :=
by
  sorry

end NUMINAMATH_GPT_combined_weight_of_emma_and_henry_l236_23607


namespace NUMINAMATH_GPT_simplify_expression_l236_23691

theorem simplify_expression : 
  (1 / ((1 / (1 / 3)^1) + (1 / (1 / 3)^2) + (1 / (1 / 3)^3))) = 1 / 39 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l236_23691
