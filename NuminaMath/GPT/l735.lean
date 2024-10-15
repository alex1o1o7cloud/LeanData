import Mathlib

namespace NUMINAMATH_GPT_fraction_cal_handled_l735_73500

theorem fraction_cal_handled (Mabel Anthony Cal Jade : ℕ) 
  (h_Mabel : Mabel = 90)
  (h_Anthony : Anthony = Mabel + Mabel / 10)
  (h_Jade : Jade = 80)
  (h_Cal : Cal = Jade - 14) :
  (Cal : ℚ) / (Anthony : ℚ) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_cal_handled_l735_73500


namespace NUMINAMATH_GPT_hoseok_multiplied_number_l735_73572

theorem hoseok_multiplied_number (n : ℕ) (h : 11 * n = 99) : n = 9 := 
sorry

end NUMINAMATH_GPT_hoseok_multiplied_number_l735_73572


namespace NUMINAMATH_GPT_compute_xy_l735_73529

variable (x y : ℝ)
variable (h1 : x - y = 6)
variable (h2 : x^3 - y^3 = 108)

theorem compute_xy : x * y = 0 := by
  sorry

end NUMINAMATH_GPT_compute_xy_l735_73529


namespace NUMINAMATH_GPT_probability_of_three_primes_from_30_l735_73585

noncomputable def primes_up_to_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_three_primes_from_30 :
  ((primes_up_to_30.card.choose 3) / ((Finset.range 31).card.choose 3)) = (6 / 203) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_three_primes_from_30_l735_73585


namespace NUMINAMATH_GPT_simplify_fraction_l735_73517

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 3 + Real.sqrt 48) = (5 * Real.sqrt 3) / 36 :=
by
  have h1 : Real.sqrt 75 = 5 * Real.sqrt 3 := by sorry
  have h2 : Real.sqrt 48 = 4 * Real.sqrt 3 := by sorry
  sorry

end NUMINAMATH_GPT_simplify_fraction_l735_73517


namespace NUMINAMATH_GPT_quadratic_inequality_l735_73544

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_l735_73544


namespace NUMINAMATH_GPT_pure_imaginary_solutions_l735_73501

theorem pure_imaginary_solutions:
  ∀ (x : ℂ), (x.im ≠ 0 ∧ x.re = 0) → (x ^ 4 - 5 * x ^ 3 + 10 * x ^ 2 - 50 * x - 75 = 0)
         → (x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_solutions_l735_73501


namespace NUMINAMATH_GPT_area_EFGH_l735_73578

theorem area_EFGH (n : ℕ) (n_pos : 1 < n) (S_ABCD : ℝ) (h₁ : S_ABCD = 1) :
  ∃ S_EFGH : ℝ, S_EFGH = (n - 2) / n :=
by sorry

end NUMINAMATH_GPT_area_EFGH_l735_73578


namespace NUMINAMATH_GPT_correct_division_l735_73520

theorem correct_division (x : ℝ) (h : 8 * x + 8 = 56) : x / 8 = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_correct_division_l735_73520


namespace NUMINAMATH_GPT_find_roots_of_polynomial_l735_73564

noncomputable def polynomial_roots : Set ℝ :=
  {x | (6 * x^4 + 25 * x^3 - 59 * x^2 + 28 * x) = 0 }

theorem find_roots_of_polynomial :
  polynomial_roots = {0, 1, (-31 + Real.sqrt 1633) / 12, (-31 - Real.sqrt 1633) / 12} :=
by
  sorry

end NUMINAMATH_GPT_find_roots_of_polynomial_l735_73564


namespace NUMINAMATH_GPT_find_a4_l735_73540

variable {a_n : ℕ → ℝ}
variable (S_n : ℕ → ℝ)

noncomputable def Sn := 1/2 * 5 * (a_n 1 + a_n 5)

axiom h1 : S_n 5 = 25
axiom h2 : a_n 2 = 3

theorem find_a4 : a_n 4 = 5 := sorry

end NUMINAMATH_GPT_find_a4_l735_73540


namespace NUMINAMATH_GPT_quotient_of_division_l735_73550

theorem quotient_of_division (dividend divisor remainder quotient : ℕ) 
  (h_dividend : dividend = 271) (h_divisor : divisor = 30) 
  (h_remainder : remainder = 1) (h_division : dividend = divisor * quotient + remainder) : 
  quotient = 9 := 
by 
  sorry

end NUMINAMATH_GPT_quotient_of_division_l735_73550


namespace NUMINAMATH_GPT_negation_example_l735_73515

theorem negation_example :
  (¬ (∀ x : ℝ, x^2 - 2 * x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) :=
sorry

end NUMINAMATH_GPT_negation_example_l735_73515


namespace NUMINAMATH_GPT_sum_in_range_l735_73576

noncomputable def mixed_number_sum : ℚ :=
  3 + 1/8 + 4 + 3/7 + 6 + 2/21

theorem sum_in_range : 13.5 ≤ mixed_number_sum ∧ mixed_number_sum < 14 := by
  sorry

end NUMINAMATH_GPT_sum_in_range_l735_73576


namespace NUMINAMATH_GPT_find_son_age_l735_73596

theorem find_son_age (F S : ℕ) (h1 : F + S = 55)
  (h2 : ∃ Y, S + Y = F ∧ (F + Y) + (S + Y) = 93)
  (h3 : F = 18 ∨ S = 18) : S = 18 :=
by
  sorry  -- Proof to be filled in

end NUMINAMATH_GPT_find_son_age_l735_73596


namespace NUMINAMATH_GPT_smallest_number_l735_73598

theorem smallest_number:
  ∃ n : ℕ, (∀ d ∈ [12, 16, 18, 21, 28, 35, 39], (n - 7) % d = 0) ∧ n = 65527 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_l735_73598


namespace NUMINAMATH_GPT_div_sub_mult_exp_eq_l735_73506

-- Lean 4 statement for the mathematical proof problem
theorem div_sub_mult_exp_eq :
  8 / 4 - 3 - 9 + 3 * 7 - 2^2 = 7 := 
sorry

end NUMINAMATH_GPT_div_sub_mult_exp_eq_l735_73506


namespace NUMINAMATH_GPT_find_average_after_17th_inning_l735_73541

def initial_average_after_16_inns (A : ℕ) : Prop :=
  let total_runs := 16 * A
  let new_total_runs := total_runs + 87
  let new_average := new_total_runs / 17
  new_average = A + 4

def runs_in_17th_inning := 87

noncomputable def average_after_17th_inning (A : ℕ) : Prop :=
  A + 4 = 23

theorem find_average_after_17th_inning (A : ℕ) :
  initial_average_after_16_inns A →
  average_after_17th_inning A :=
  sorry

end NUMINAMATH_GPT_find_average_after_17th_inning_l735_73541


namespace NUMINAMATH_GPT_solve_eq_l735_73551

theorem solve_eq : ∀ x : ℝ, -2 * (x - 1) = 4 → x = -1 := 
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_eq_l735_73551


namespace NUMINAMATH_GPT_ratio_of_side_lengths_l735_73555

theorem ratio_of_side_lengths
  (pentagon_perimeter square_perimeter : ℕ)
  (pentagon_sides square_sides : ℕ)
  (pentagon_perimeter_eq : pentagon_perimeter = 100)
  (square_perimeter_eq : square_perimeter = 100)
  (pentagon_sides_eq : pentagon_sides = 5)
  (square_sides_eq : square_sides = 4) :
  (pentagon_perimeter / pentagon_sides) / (square_perimeter / square_sides) = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_side_lengths_l735_73555


namespace NUMINAMATH_GPT_part1_even_function_part2_min_value_l735_73519

variable {a x : ℝ}

def f (x a : ℝ) : ℝ := x^2 + |x - a| + 1

theorem part1_even_function (h : a = 0) : 
  ∀ x : ℝ, f x 0 = f (-x) 0 :=
by
  -- This statement needs to be proved to show that f(x) is even when a = 0
  sorry

theorem part2_min_value (h : true) : 
  (a > (1/2) → ∃ x : ℝ, f x a = a + (3/4)) ∧
  (a ≤ -(1/2) → ∃ x : ℝ, f x a = -a + (3/4)) ∧
  ((- (1/2) < a ∧ a ≤ (1/2)) → ∃ x : ℝ, f x a = a^2 + 1) :=
by
  -- This statement needs to be proved to show the different minimum values of the function
  sorry

end NUMINAMATH_GPT_part1_even_function_part2_min_value_l735_73519


namespace NUMINAMATH_GPT_compound_interest_example_l735_73579

theorem compound_interest_example :
  let P := 5000
  let r := 0.08
  let n := 4
  let t := 0.5
  let A := P * (1 + r / n) ^ (n * t)
  A = 5202 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_example_l735_73579


namespace NUMINAMATH_GPT_perfect_square_pairs_l735_73562

-- Definition of a perfect square
def is_perfect_square (k : ℕ) : Prop :=
∃ (n : ℕ), n * n = k

-- Main theorem statement
theorem perfect_square_pairs (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  is_perfect_square ((2^m - 1) * (2^n - 1)) ↔ (m = n) ∨ (m = 3 ∧ n = 6) ∨ (m = 6 ∧ n = 3) :=
sorry

end NUMINAMATH_GPT_perfect_square_pairs_l735_73562


namespace NUMINAMATH_GPT_booster_club_tickets_l735_73583

theorem booster_club_tickets (x : ℕ) : 
  (11 * 9 + x * 7 = 225) → 
  (x + 11 = 29) := 
by
  sorry

end NUMINAMATH_GPT_booster_club_tickets_l735_73583


namespace NUMINAMATH_GPT_correct_divisor_l735_73590

variable (D X : ℕ)

-- Conditions
def condition1 : Prop := X = D * 24
def condition2 : Prop := X = (D - 12) * 42

theorem correct_divisor (D X : ℕ) (h1 : condition1 D X) (h2 : condition2 D X) : D = 28 := by
  sorry

end NUMINAMATH_GPT_correct_divisor_l735_73590


namespace NUMINAMATH_GPT_Anne_mom_toothpaste_usage_l735_73593

theorem Anne_mom_toothpaste_usage
  (total_toothpaste : ℕ)
  (dad_usage_per_brush : ℕ)
  (sibling_usage_per_brush : ℕ)
  (num_brushes_per_day : ℕ)
  (total_days : ℕ)
  (total_toothpaste_used : ℕ)
  (M : ℕ)
  (family_use_model : total_toothpaste = total_toothpaste_used + 3 * num_brushes_per_day * M)
  (total_toothpaste_used_def : total_toothpaste_used = 5 * (dad_usage_per_brush * num_brushes_per_day + 2 * sibling_usage_per_brush * num_brushes_per_day))
  (given_values : total_toothpaste = 105 ∧ dad_usage_per_brush = 3 ∧ sibling_usage_per_brush = 1 ∧ num_brushes_per_day = 3 ∧ total_days = 5)
  : M = 2 := by
  sorry

end NUMINAMATH_GPT_Anne_mom_toothpaste_usage_l735_73593


namespace NUMINAMATH_GPT_base_of_exponent_l735_73552

theorem base_of_exponent (b x y : ℕ) (h1 : x - y = 12) (h2 : x = 12) (h3 : b^x * 4^y = 531441) : b = 3 :=
by
  sorry

end NUMINAMATH_GPT_base_of_exponent_l735_73552


namespace NUMINAMATH_GPT_find_sixth_term_of_geometric_sequence_l735_73571

noncomputable def common_ratio (a b : ℚ) : ℚ := b / a

noncomputable def geometric_sequence_term (a r : ℚ) (k : ℕ) : ℚ := a * (r ^ (k - 1))

theorem find_sixth_term_of_geometric_sequence :
  geometric_sequence_term 5 (common_ratio 5 1.25) 6 = 5 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_find_sixth_term_of_geometric_sequence_l735_73571


namespace NUMINAMATH_GPT_ratio_of_expenditures_l735_73545

theorem ratio_of_expenditures 
  (income_Uma : ℕ) (income_Bala : ℕ) (expenditure_Uma : ℕ) (expenditure_Bala : ℕ)
  (h_ratio_incomes : income_Uma / income_Bala = 4 / 3)
  (h_savings_Uma : income_Uma - expenditure_Uma = 5000)
  (h_savings_Bala : income_Bala - expenditure_Bala = 5000)
  (h_income_Uma : income_Uma = 20000) :
  expenditure_Uma / expenditure_Bala = 3 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_expenditures_l735_73545


namespace NUMINAMATH_GPT_horner_multiplications_additions_l735_73531

-- Define the polynomial
def f (x : ℤ) : ℤ := x^7 + 2 * x^5 + 3 * x^4 + 4 * x^3 + 5 * x^2 + 6 * x + 7

-- Define the number of multiplications and additions required by Horner's method
def horner_method_mults (n : ℕ) : ℕ := n
def horner_method_adds (n : ℕ) : ℕ := n - 1

-- Define the value of x
def x : ℤ := 3

-- Define the degree of the polynomial
def degree_of_polynomial : ℕ := 7

-- Define the statements for the proof
theorem horner_multiplications_additions :
  horner_method_mults degree_of_polynomial = 7 ∧
  horner_method_adds degree_of_polynomial = 6 :=
by
  sorry

end NUMINAMATH_GPT_horner_multiplications_additions_l735_73531


namespace NUMINAMATH_GPT_alice_password_prob_correct_l735_73569

noncomputable def password_probability : ℚ :=
  let even_digit_prob := 5 / 10
  let valid_symbol_prob := 3 / 5
  let non_zero_digit_prob := 9 / 10
  even_digit_prob * valid_symbol_prob * non_zero_digit_prob

theorem alice_password_prob_correct :
  password_probability = 27 / 100 := by
  rfl

end NUMINAMATH_GPT_alice_password_prob_correct_l735_73569


namespace NUMINAMATH_GPT_fraction_paint_remaining_l735_73503

theorem fraction_paint_remaining 
  (original_paint : ℝ)
  (h_original : original_paint = 2) 
  (used_first_day : ℝ)
  (h_used_first_day : used_first_day = (1 / 4) * original_paint) 
  (remaining_after_first : ℝ)
  (h_remaining_first : remaining_after_first = original_paint - used_first_day) 
  (used_second_day : ℝ)
  (h_used_second_day : used_second_day = (1 / 3) * remaining_after_first) 
  (remaining_after_second : ℝ)
  (h_remaining_second : remaining_after_second = remaining_after_first - used_second_day) : 
  remaining_after_second / original_paint = 1 / 2 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_fraction_paint_remaining_l735_73503


namespace NUMINAMATH_GPT_simplify_fractions_l735_73542

theorem simplify_fractions :
  (270 / 18) * (7 / 210) * (9 / 4) = 9 / 8 :=
by sorry

end NUMINAMATH_GPT_simplify_fractions_l735_73542


namespace NUMINAMATH_GPT_probability_of_Y_l735_73527

theorem probability_of_Y (P_X P_both : ℝ) (h1 : P_X = 1/5) (h2 : P_both = 0.13333333333333333) : 
    (0.13333333333333333 / (1 / 5)) = 0.6666666666666667 :=
by sorry

end NUMINAMATH_GPT_probability_of_Y_l735_73527


namespace NUMINAMATH_GPT_lower_bound_for_expression_l735_73523

theorem lower_bound_for_expression :
  ∃ L: ℤ, (∀ n: ℤ, L < 4 * n + 7 ∧ 4 * n + 7 < 120) → L = 5 :=
sorry

end NUMINAMATH_GPT_lower_bound_for_expression_l735_73523


namespace NUMINAMATH_GPT_find_x_y_l735_73568

theorem find_x_y (x y : ℝ) : (3 * x + 4 * -2 = 0) ∧ (3 * 1 + 4 * y = 0) → x = 8 / 3 ∧ y = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_y_l735_73568


namespace NUMINAMATH_GPT_find_function_solution_l735_73514

noncomputable def function_solution (f : ℤ → ℤ) : Prop :=
∀ x y : ℤ, x ≠ 0 → x * f (2 * f y - x) + y^2 * f (2 * x - f y) = f x ^ 2 / x + f (y * f y)

theorem find_function_solution : 
  ∀ f : ℤ → ℤ, function_solution f → (∀ x : ℤ, f x = 0) ∨ (∀ x : ℤ, f x = x^2) :=
sorry

end NUMINAMATH_GPT_find_function_solution_l735_73514


namespace NUMINAMATH_GPT_find_a_l735_73561

theorem find_a (a x : ℝ) 
  (h : x^2 + 3 * x + a = (x + 1) * (x + 2)) : 
  a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l735_73561


namespace NUMINAMATH_GPT_find_a2_l735_73588

theorem find_a2 (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + 2)
  (h_geom : (a 1) * (a 5) = (a 2) * (a 2)) : a 2 = 3 :=
by
  -- We are given the conditions and need to prove the statement.
  sorry

end NUMINAMATH_GPT_find_a2_l735_73588


namespace NUMINAMATH_GPT_sequence_value_l735_73592

theorem sequence_value (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, n > 0 → a n * a (n + 2) = a (n + 1) ^ 2)
  (h2 : a 7 = 16)
  (h3 : a 3 * a 5 = 4) : 
  a 3 = 1 := 
sorry

end NUMINAMATH_GPT_sequence_value_l735_73592


namespace NUMINAMATH_GPT_sum_of_digits_of_square_99999_l735_73516

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_square_99999 : sum_of_digits ((99999 : ℕ)^2) = 45 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_square_99999_l735_73516


namespace NUMINAMATH_GPT_problem1_problem2_l735_73513

-- Problem 1
theorem problem1 : (2 * Real.sqrt 12 - 3 * Real.sqrt (1 / 3)) * Real.sqrt 6 = 9 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h1 : x / (2 * x - 1) = 2 - 3 / (1 - 2 * x)) : x = -1 / 3 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l735_73513


namespace NUMINAMATH_GPT_b7_in_form_l735_73580

theorem b7_in_form (a : ℕ → ℚ) (b : ℕ → ℚ) : 
  a 0 = 3 → 
  b 0 = 5 → 
  (∀ n : ℕ, a (n + 1) = (a n)^2 / (b n)) → 
  (∀ n : ℕ, b (n + 1) = (b n)^2 / (a n)) → 
  b 7 = (5^50 : ℚ) / (3^41 : ℚ) := 
by 
  intros h1 h2 h3 h4 
  sorry

end NUMINAMATH_GPT_b7_in_form_l735_73580


namespace NUMINAMATH_GPT_quadratic_one_real_root_l735_73508

theorem quadratic_one_real_root (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : ∀ x : ℝ, x^2 + 6*m*x - n = 0 → x * x = 0) : n = 9*m^2 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_one_real_root_l735_73508


namespace NUMINAMATH_GPT_snow_volume_l735_73537

theorem snow_volume
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (h_length : length = 15)
  (h_width : width = 3)
  (h_depth : depth = 0.6) :
  length * width * depth = 27 := 
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_snow_volume_l735_73537


namespace NUMINAMATH_GPT_percentage_increase_school_B_l735_73525

theorem percentage_increase_school_B (A B Q_A Q_B : ℝ) 
  (h1 : Q_A = 0.7 * A) 
  (h2 : Q_B = 1.5 * Q_A) 
  (h3 : Q_B = 0.875 * B) :
  (B - A) / A * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_school_B_l735_73525


namespace NUMINAMATH_GPT_milton_sold_15_pies_l735_73559

theorem milton_sold_15_pies
  (apple_pie_slices_per_pie : ℕ) (peach_pie_slices_per_pie : ℕ)
  (ordered_apple_pie_slices : ℕ) (ordered_peach_pie_slices : ℕ)
  (h1 : apple_pie_slices_per_pie = 8) (h2 : peach_pie_slices_per_pie = 6)
  (h3 : ordered_apple_pie_slices = 56) (h4 : ordered_peach_pie_slices = 48) :
  (ordered_apple_pie_slices / apple_pie_slices_per_pie) + (ordered_peach_pie_slices / peach_pie_slices_per_pie) = 15 := 
by
  sorry

end NUMINAMATH_GPT_milton_sold_15_pies_l735_73559


namespace NUMINAMATH_GPT_original_number_conditions_l735_73594

theorem original_number_conditions (a : ℕ) :
  ∃ (y1 y2 : ℕ), (7 * a = 10 * 9 + y1) ∧ (9 * 9 = 10 * 8 + y2) ∧ y2 = 1 ∧ (a = 13 ∨ a = 14) := sorry

end NUMINAMATH_GPT_original_number_conditions_l735_73594


namespace NUMINAMATH_GPT_evaluate_g_at_neg1_l735_73584

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem evaluate_g_at_neg1 : g (-1) = -5 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_neg1_l735_73584


namespace NUMINAMATH_GPT_sum_of_fractions_bounds_l735_73512

theorem sum_of_fractions_bounds (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum_numerators : a + c = 1000) (h_sum_denominators : b + d = 1000) :
  (999 / 969 + 1 / 31) ≤ (a / b + c / d) ∧ (a / b + c / d) ≤ (999 + 1 / 999) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_bounds_l735_73512


namespace NUMINAMATH_GPT_find_m_l735_73563

theorem find_m :
  ∃ m : ℝ, (∀ x : ℝ, x > 0 → (m^2 - m - 5) * x^(m - 1) > 0) ∧ m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_l735_73563


namespace NUMINAMATH_GPT_math_club_partition_l735_73536

def is_played (team : Finset ℕ) (A B C : ℕ) : Bool :=
(A ∈ team ∧ B ∉ team ∧ C ∉ team) ∨ 
(A ∉ team ∧ B ∈ team ∧ C ∉ team) ∨ 
(A ∉ team ∧ B ∉ team ∧ C ∈ team) ∨ 
(A ∈ team ∧ B ∈ team ∧ C ∈ team)

theorem math_club_partition 
  (students : Finset ℕ) (A B C : ℕ) 
  (h_size : students.card = 24)
  (teams : List (Finset ℕ))
  (h_teams : teams.length = 4)
  (h_team_size : ∀ t ∈ teams, t.card = 6)
  (h_partition : ∀ t ∈ teams, t ⊆ students) :
  ∃ (teams_played : List (Finset ℕ)), teams_played.length = 1 ∨ teams_played.length = 3 :=
sorry

end NUMINAMATH_GPT_math_club_partition_l735_73536


namespace NUMINAMATH_GPT_solve_eq_l735_73595

open Real

noncomputable def solution : Set ℝ := { x | ∃ (n : ℤ), x = π / 12 + π * (n : ℝ) }

theorem solve_eq : { x : ℝ | ∃ (n : ℤ), x = π / 12 + π * (n : ℝ) } = solution := by sorry

end NUMINAMATH_GPT_solve_eq_l735_73595


namespace NUMINAMATH_GPT_product_of_three_integers_sum_l735_73511
-- Import necessary libraries

-- Define the necessary conditions and the goal
theorem product_of_three_integers_sum (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
(h4 : a * b * c = 11^3) : a + b + c = 133 :=
sorry

end NUMINAMATH_GPT_product_of_three_integers_sum_l735_73511


namespace NUMINAMATH_GPT_projection_matrix_solution_l735_73554

theorem projection_matrix_solution (a c : ℚ) (Q : Matrix (Fin 2) (Fin 2) ℚ) 
  (hQ : Q = !![a, 18/45; c, 27/45] ) 
  (proj_Q : Q * Q = Q) : 
  (a, c) = (2/5, 3/5) :=
by
  sorry

end NUMINAMATH_GPT_projection_matrix_solution_l735_73554


namespace NUMINAMATH_GPT_percent_value_in_quarters_l735_73565

theorem percent_value_in_quarters
  (num_dimes num_quarters num_nickels : ℕ)
  (value_dime value_quarter value_nickel : ℕ)
  (h_dimes : num_dimes = 70)
  (h_quarters : num_quarters = 30)
  (h_nickels : num_nickels = 40)
  (h_value_dime : value_dime = 10)
  (h_value_quarter : value_quarter = 25)
  (h_value_nickel : value_nickel = 5) :
  ((num_quarters * value_quarter : ℕ) * 100 : ℚ) / 
  (num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel) = 45.45 :=
by
  sorry

end NUMINAMATH_GPT_percent_value_in_quarters_l735_73565


namespace NUMINAMATH_GPT_new_supervisor_salary_l735_73558

-- Definitions
def average_salary_old (W : ℕ) : Prop :=
  (W + 870) / 9 = 430

def average_salary_new (W : ℕ) (S_new : ℕ) : Prop :=
  (W + S_new) / 9 = 430

-- Problem statement
theorem new_supervisor_salary (W : ℕ) (S_new : ℕ) :
  average_salary_old W →
  average_salary_new W S_new →
  S_new = 870 :=
by
  sorry

end NUMINAMATH_GPT_new_supervisor_salary_l735_73558


namespace NUMINAMATH_GPT_no_real_solution_intersection_l735_73581

theorem no_real_solution_intersection :
  ¬ ∃ x y : ℝ, (y = 8 / (x^3 + 4 * x + 3)) ∧ (x + y = 5) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_intersection_l735_73581


namespace NUMINAMATH_GPT_part_I_part_II_l735_73582

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos x) ^ 2 - Real.sin (2 * x - (7 * Real.pi / 6))

theorem part_I :
  (∀ x, f x ≤ 2) ∧ (∃ x, f x = 2 ∧ ∃ k : ℤ, x = k * Real.pi + Real.pi / 6) :=
by
  sorry

theorem part_II (A a b c : ℝ) (h1 : f A = 3 / 2) (h2 : b + c = 2) :
  a >= 1 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l735_73582


namespace NUMINAMATH_GPT_yuna_initial_marbles_l735_73599

theorem yuna_initial_marbles (M : ℕ) :
  (M - 12 + 5) / 2 + 3 = 17 → M = 35 := by
  sorry

end NUMINAMATH_GPT_yuna_initial_marbles_l735_73599


namespace NUMINAMATH_GPT_arithmetic_progression_number_of_terms_l735_73521

variable (a d : ℕ)
variable (n : ℕ) (h_n_even : n % 2 = 0)
variable (h_sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 60)
variable (h_sum_even : (n / 2) * (2 * (a + d) + (n - 2) * d) = 80)
variable (h_diff : (n - 1) * d = 16)

theorem arithmetic_progression_number_of_terms : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_number_of_terms_l735_73521


namespace NUMINAMATH_GPT_total_time_spent_l735_73589

noncomputable def time_per_round : ℕ := 30
noncomputable def saturday_rounds : ℕ := 1 + 10
noncomputable def sunday_rounds : ℕ := 15
noncomputable def total_rounds : ℕ := saturday_rounds + sunday_rounds
noncomputable def total_time : ℕ := total_rounds * time_per_round

theorem total_time_spent :
  total_time = 780 := by sorry

end NUMINAMATH_GPT_total_time_spent_l735_73589


namespace NUMINAMATH_GPT_total_cloth_sold_l735_73587

variable (commissionA commissionB salesA salesB totalWorth : ℝ)

def agentA_commission := 0.025 * salesA
def agentB_commission := 0.03 * salesB
def total_worth_of_cloth_sold := salesA + salesB

theorem total_cloth_sold 
  (hA : agentA_commission = 21) 
  (hB : agentB_commission = 27)
  : total_worth_of_cloth_sold = 1740 :=
by
  sorry

end NUMINAMATH_GPT_total_cloth_sold_l735_73587


namespace NUMINAMATH_GPT_crystal_meal_combinations_l735_73556

-- Definitions for conditions:
def entrees := 4
def drinks := 4
def desserts := 3 -- includes two desserts and the option of no dessert

-- Statement of the problem as a theorem:
theorem crystal_meal_combinations : entrees * drinks * desserts = 48 := by
  sorry

end NUMINAMATH_GPT_crystal_meal_combinations_l735_73556


namespace NUMINAMATH_GPT_gcd_lcm_problem_l735_73549

theorem gcd_lcm_problem (b : ℤ) (x : ℕ) (hx_pos : 0 < x) (hx : x = 12) :
  gcd 30 b = x + 3 ∧ lcm 30 b = x * (x + 3) → b = 90 := 
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_problem_l735_73549


namespace NUMINAMATH_GPT_two_dice_sum_greater_than_four_l735_73510
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end NUMINAMATH_GPT_two_dice_sum_greater_than_four_l735_73510


namespace NUMINAMATH_GPT_G_at_16_l735_73575

noncomputable def G : ℝ → ℝ := sorry

-- Condition 1: G is a polynomial, implicitly stated
-- Condition 2: Given G(8) = 21
axiom G_at_8 : G 8 = 21

-- Condition 3: Given that
axiom G_fraction_condition : ∀ (x : ℝ), 
  (x^2 + 6*x + 8) ≠ 0 ∧ ((x+4)*(x+2)) ≠ 0 → 
  (G (2*x) / G (x+4) = 4 - (16*x + 32) / (x^2 + 6*x + 8))

-- The problem: Prove G(16) = 90
theorem G_at_16 : G 16 = 90 := 
sorry

end NUMINAMATH_GPT_G_at_16_l735_73575


namespace NUMINAMATH_GPT_sqrt_domain_l735_73528

theorem sqrt_domain (x : ℝ) : (∃ y, y * y = x - 2) ↔ (x ≥ 2) :=
by sorry

end NUMINAMATH_GPT_sqrt_domain_l735_73528


namespace NUMINAMATH_GPT_product_of_reverse_numbers_l735_73507

def reverse (n : Nat) : Nat :=
  Nat.ofDigits 10 (List.reverse (Nat.digits 10 n))

theorem product_of_reverse_numbers : 
  ∃ (a b : ℕ), a * b = 92565 ∧ b = reverse a ∧ ((a = 165 ∧ b = 561) ∨ (a = 561 ∧ b = 165)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_reverse_numbers_l735_73507


namespace NUMINAMATH_GPT_inappropriate_survey_method_l735_73543

def survey_method_appropriate (method : String) : Bool :=
  method = "sampling" -- only sampling is considered appropriate in this toy model

def survey_approps : Bool :=
  let A := survey_method_appropriate "sampling"
  let B := survey_method_appropriate "sampling"
  let C := ¬ survey_method_appropriate "census"
  let D := survey_method_appropriate "census"
  C

theorem inappropriate_survey_method :
  survey_approps = true :=
by
  sorry

end NUMINAMATH_GPT_inappropriate_survey_method_l735_73543


namespace NUMINAMATH_GPT_calculate_expression_l735_73567

theorem calculate_expression :
  50 * 24.96 * 2.496 * 500 = (1248)^2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l735_73567


namespace NUMINAMATH_GPT_find_functions_satisfying_condition_l735_73566

noncomputable def function_satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 → a * b * c * d = 1 →
  (f a + f b) * (f c + f d) = (a + b) * (c + d)

theorem find_functions_satisfying_condition :
  ∀ f : ℝ → ℝ, function_satisfies_condition f →
    (∀ x : ℝ, x > 0 → f x = x) ∨ (∀ x : ℝ, x > 0 → f x = 1 / x) :=
sorry

end NUMINAMATH_GPT_find_functions_satisfying_condition_l735_73566


namespace NUMINAMATH_GPT_simplify_expression_l735_73509

-- Definitions derived from the problem statement
variable (x : ℝ)

-- Theorem statement
theorem simplify_expression : 1 - (1 + (1 - (1 + (1 - x)))) = 1 - x :=
sorry

end NUMINAMATH_GPT_simplify_expression_l735_73509


namespace NUMINAMATH_GPT_no_integer_regular_pentagon_l735_73577

theorem no_integer_regular_pentagon 
  (x y : Fin 5 → ℤ) 
  (h_length : ∀ i j : Fin 5, i ≠ j → (x i - x j) ^ 2 + (y i - y j) ^ 2 = (x 0 - x 1) ^ 2 + (y 0 - y 1) ^ 2)
  : False :=
sorry

end NUMINAMATH_GPT_no_integer_regular_pentagon_l735_73577


namespace NUMINAMATH_GPT_cuboid_edge_length_l735_73560

theorem cuboid_edge_length (x : ℝ) (h1 : 5 * 6 * x = 120) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_edge_length_l735_73560


namespace NUMINAMATH_GPT_probability_both_red_l735_73548

-- Definitions for the problem conditions
def total_balls := 16
def red_balls := 7
def blue_balls := 5
def green_balls := 4
def first_red_prob := (red_balls : ℚ) / total_balls
def second_red_given_first_red_prob := (red_balls - 1 : ℚ) / (total_balls - 1)

-- The statement to be proved
theorem probability_both_red : (first_red_prob * second_red_given_first_red_prob) = (7 : ℚ) / 40 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_both_red_l735_73548


namespace NUMINAMATH_GPT_convert_speed_kmh_to_ms_l735_73547

-- Define the given speed in km/h
def speed_kmh : ℝ := 1.1076923076923078

-- Define the conversion factor from km/h to m/s
def conversion_factor : ℝ := 3.6

-- State the theorem
theorem convert_speed_kmh_to_ms (s : ℝ) (h : s = speed_kmh) : (s / conversion_factor) = 0.3076923076923077 := by
  -- Skip the proof as instructed
  sorry

end NUMINAMATH_GPT_convert_speed_kmh_to_ms_l735_73547


namespace NUMINAMATH_GPT_rachels_game_final_configurations_l735_73530

-- Define the number of cells in the grid
def n : ℕ := 2011

-- Define the number of moves needed
def moves_needed : ℕ := n - 3

-- Define a function that counts the number of distinct final configurations
-- based on the number of fights (f) possible in the given moves.
def final_configurations : ℕ := moves_needed + 1

theorem rachels_game_final_configurations : final_configurations = 2009 :=
by
  -- Calculation shows that moves_needed = 2008 and therefore final_configurations = 2008 + 1 = 2009.
  sorry

end NUMINAMATH_GPT_rachels_game_final_configurations_l735_73530


namespace NUMINAMATH_GPT_tom_books_after_transactions_l735_73532

-- Define the initial conditions as variables
def initial_books : ℕ := 5
def sold_books : ℕ := 4
def new_books : ℕ := 38

-- Define the property we need to prove
theorem tom_books_after_transactions : initial_books - sold_books + new_books = 39 := by
  sorry

end NUMINAMATH_GPT_tom_books_after_transactions_l735_73532


namespace NUMINAMATH_GPT_largest_prime_number_largest_composite_number_l735_73538

-- Definitions of prime and composite
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

-- Largest prime and composite numbers less than 20
def largest_prime_less_than_20 := 19
def largest_composite_less_than_20 := 18

theorem largest_prime_number : 
  largest_prime_less_than_20 = 19 ∧ is_prime 19 ∧ 
  (∀ n : ℕ, n < 20 → is_prime n → n < 19) := 
by sorry

theorem largest_composite_number : 
  largest_composite_less_than_20 = 18 ∧ is_composite 18 ∧ 
  (∀ n : ℕ, n < 20 → is_composite n → n < 18) := 
by sorry

end NUMINAMATH_GPT_largest_prime_number_largest_composite_number_l735_73538


namespace NUMINAMATH_GPT_largest_divisor_of_n_l735_73570

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 360 ∣ n^2) : 60 ∣ n := 
sorry

end NUMINAMATH_GPT_largest_divisor_of_n_l735_73570


namespace NUMINAMATH_GPT_find_d_q_l735_73504

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def b_n (b1 q : ℕ) (n : ℕ) : ℕ :=
  b1 * q^(n - 1)

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S_n (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * a1 + (n * (n - 1) / 2) * d

-- Sum of the first n terms of a geometric sequence
noncomputable def T_n (b1 q : ℕ) (n : ℕ) : ℕ :=
  if q = 1 then n * b1
  else b1 * (1 - q^n) / (1 - q)

theorem find_d_q (a1 b1 d q : ℕ) (h1 : ∀ n : ℕ, n > 0 →
  n^2 * (T_n b1 q n + 1) = 2^n * S_n a1 d n) : d = 2 ∧ q = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_d_q_l735_73504


namespace NUMINAMATH_GPT_n_cubed_plus_5_div_by_6_l735_73533

theorem n_cubed_plus_5_div_by_6  (n : ℤ) : 6 ∣ n * (n^2 + 5) :=
sorry

end NUMINAMATH_GPT_n_cubed_plus_5_div_by_6_l735_73533


namespace NUMINAMATH_GPT_standard_equation_of_ellipse_l735_73597

-- Definitions for clarity
def is_ellipse (E : Type) := true
def major_axis (e : is_ellipse E) : ℝ := sorry
def minor_axis (e : is_ellipse E) : ℝ := sorry
def focus (e : is_ellipse E) : ℝ := sorry

theorem standard_equation_of_ellipse (E : Type)
  (e : is_ellipse E)
  (major_sum : major_axis e + minor_axis e = 9)
  (focus_position : focus e = 3) :
  ∀ x y, (x^2 / 25) + (y^2 / 16) = 1 :=
by sorry

end NUMINAMATH_GPT_standard_equation_of_ellipse_l735_73597


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l735_73553

theorem solve_equation_1 (x : ℚ) : 1 - (1 / (x - 5)) = (x / (x + 5)) → x = 15 / 2 := 
by
  sorry

theorem solve_equation_2 (x : ℚ) : (3 / (x - 1)) - (2 / (x + 1)) = (1 / (x^2 - 1)) → x = -4 := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l735_73553


namespace NUMINAMATH_GPT_possible_division_l735_73546

theorem possible_division (side_length : ℕ) (areas : Fin 5 → Set (Fin side_length × Fin side_length))
  (h1 : side_length = 5)
  (h2 : ∀ i, ∃ cells : Finset (Fin side_length × Fin side_length), areas i = cells ∧ Finset.card cells = 5)
  (h3 : ∀ i j, i ≠ j → Disjoint (areas i) (areas j))
  (total_cut_length : ℕ)
  (h4 : total_cut_length ≤ 16) :
  
  ∃ cuts : Finset (Fin side_length × Fin side_length) × Finset (Fin side_length × Fin side_length),
    total_cut_length = (cuts.1.card + cuts.2.card) :=
sorry

end NUMINAMATH_GPT_possible_division_l735_73546


namespace NUMINAMATH_GPT_inequality_solution_l735_73526

theorem inequality_solution (x : ℝ) : 
  (x / (x + 5) ≥ 0) ↔ (x ∈ (Set.Iio (-5)).union (Set.Ici 0)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l735_73526


namespace NUMINAMATH_GPT_days_to_complete_work_together_l735_73586

theorem days_to_complete_work_together :
  (20 * 35) / (20 + 35) = 140 / 11 :=
by
  sorry

end NUMINAMATH_GPT_days_to_complete_work_together_l735_73586


namespace NUMINAMATH_GPT_problem_statement_l735_73591

variable (y1 y2 y3 y4 y5 y6 y7 y8 : ℝ)

theorem problem_statement
  (h1 : y1 + 4 * y2 + 9 * y3 + 16 * y4 + 25 * y5 + 36 * y6 + 49 * y7 + 64 * y8 = 3)
  (h2 : 4 * y1 + 9 * y2 + 16 * y3 + 25 * y4 + 36 * y5 + 49 * y6 + 64 * y7 + 81 * y8 = 15)
  (h3 : 9 * y1 + 16 * y2 + 25 * y3 + 36 * y4 + 49 * y5 + 64 * y6 + 81 * y7 + 100 * y8 = 140) :
  16 * y1 + 25 * y2 + 36 * y3 + 49 * y4 + 64 * y5 + 81 * y6 + 100 * y7 + 121 * y8 = 472 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l735_73591


namespace NUMINAMATH_GPT_nonneg_int_solutions_to_ineq_system_l735_73535

open Set

theorem nonneg_int_solutions_to_ineq_system :
  {x : ℤ | (5 * x - 6 ≤ 2 * (x + 3)) ∧ ((x / 4 : ℚ) - 1 < (x - 2) / 3)} = {0, 1, 2, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_nonneg_int_solutions_to_ineq_system_l735_73535


namespace NUMINAMATH_GPT_charles_housesitting_hours_l735_73539

theorem charles_housesitting_hours :
  ∀ (earnings_per_hour_housesitting earnings_per_hour_walking_dog number_of_dogs_walked total_earnings : ℕ),
  earnings_per_hour_housesitting = 15 →
  earnings_per_hour_walking_dog = 22 →
  number_of_dogs_walked = 3 →
  total_earnings = 216 →
  ∃ h : ℕ, 15 * h + 22 * 3 = 216 ∧ h = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_charles_housesitting_hours_l735_73539


namespace NUMINAMATH_GPT_smallest_b_for_quadratic_inequality_l735_73557

theorem smallest_b_for_quadratic_inequality : 
  ∃ b : ℝ, (b^2 - 16 * b + 63 ≤ 0) ∧ ∀ b' : ℝ, (b'^2 - 16 * b' + 63 ≤ 0) → b ≤ b' := sorry

end NUMINAMATH_GPT_smallest_b_for_quadratic_inequality_l735_73557


namespace NUMINAMATH_GPT_acute_triangle_sin_sum_gt_2_l735_73502

open Real

theorem acute_triangle_sin_sum_gt_2 (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) (h_sum : α + β + γ = π) :
  sin α + sin β + sin γ > 2 :=
sorry

end NUMINAMATH_GPT_acute_triangle_sin_sum_gt_2_l735_73502


namespace NUMINAMATH_GPT_sun_city_population_correct_l735_73522

noncomputable def willowdale_population : Nat := 2000
noncomputable def roseville_population : Nat := 3 * willowdale_population - 500
noncomputable def sun_city_population : Nat := 2 * roseville_population + 1000

theorem sun_city_population_correct : sun_city_population = 12000 := by
  sorry

end NUMINAMATH_GPT_sun_city_population_correct_l735_73522


namespace NUMINAMATH_GPT_find_Q_l735_73573

variable {x P Q : ℝ}

theorem find_Q (h₁ : x + 1 / x = P) (h₂ : P = 1) : x^6 + 1 / x^6 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_Q_l735_73573


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l735_73534

def is_nat (n : ℕ) : Prop := n > 0

theorem right_triangle_hypotenuse (x : ℕ) (x_pos : is_nat x) (consec : x + 1 > x) (h : 11^2 + x^2 = (x + 1)^2) : x + 1 = 61 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l735_73534


namespace NUMINAMATH_GPT_max_students_in_auditorium_l735_73524

def increment (i : ℕ) : ℕ :=
  (i * (i + 1)) / 2

def seats_in_row (i : ℕ) : ℕ :=
  10 + increment i

def max_students_in_row (n : ℕ) : ℕ :=
  (n + 1) / 2

def total_max_students_up_to_row (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => max_students_in_row (seats_in_row (i + 1)))

theorem max_students_in_auditorium : total_max_students_up_to_row 20 = 335 := 
sorry

end NUMINAMATH_GPT_max_students_in_auditorium_l735_73524


namespace NUMINAMATH_GPT_altitude_of_triangle_l735_73574

theorem altitude_of_triangle (x : ℝ) (h : ℝ) 
  (h1 : x^2 = (1/2) * x * h) : h = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_altitude_of_triangle_l735_73574


namespace NUMINAMATH_GPT_committee_count_l735_73505

theorem committee_count (students : Finset ℕ) (Alice : ℕ) (hAlice : Alice ∈ students) (hCard : students.card = 7) :
  ∃ committees : Finset (Finset ℕ), (∀ c ∈ committees, Alice ∈ c ∧ c.card = 4) ∧ committees.card = 20 :=
sorry

end NUMINAMATH_GPT_committee_count_l735_73505


namespace NUMINAMATH_GPT_positive_difference_solutions_of_abs_eq_l735_73518

theorem positive_difference_solutions_of_abs_eq (x1 x2 : ℝ) (h1 : 2 * x1 - 3 = 15) (h2 : 2 * x2 - 3 = -15) : |x1 - x2| = 15 := by
  sorry

end NUMINAMATH_GPT_positive_difference_solutions_of_abs_eq_l735_73518
