import Mathlib

namespace fido_yard_area_fraction_l195_195890

theorem fido_yard_area_fraction (r : ℝ) (h : r > 0) :
  let square_area := (2 * r)^2
  let reachable_area := π * r^2
  let fraction := reachable_area / square_area
  ∃ a b : ℕ, (fraction = (Real.sqrt a) / b * π) ∧ (a * b = 4) := by
  sorry

end fido_yard_area_fraction_l195_195890


namespace school_total_payment_l195_195575

theorem school_total_payment
  (price : ℕ)
  (kindergarten_models : ℕ)
  (elementary_library_multiplier : ℕ)
  (model_reduction_percentage : ℚ)
  (total_models : ℕ)
  (reduced_price : ℚ)
  (total_payment : ℚ)
  (h1 : price = 100)
  (h2 : kindergarten_models = 2)
  (h3 : elementary_library_multiplier = 2)
  (h4 : model_reduction_percentage = 0.05)
  (h5 : total_models = kindergarten_models + (kindergarten_models * elementary_library_multiplier))
  (h6 : total_models > 5)
  (h7 : reduced_price = price - (price * model_reduction_percentage))
  (h8 : total_payment = total_models * reduced_price) :
  total_payment = 570 := 
by
  sorry

end school_total_payment_l195_195575


namespace least_positive_divisible_by_smallest_primes_l195_195132

def smallest_primes := [2, 3, 5, 7, 11]

noncomputable def product_of_smallest_primes :=
  List.foldl (· * ·) 1 smallest_primes

theorem least_positive_divisible_by_smallest_primes :
  product_of_smallest_primes = 2310 :=
by
  sorry

end least_positive_divisible_by_smallest_primes_l195_195132


namespace grazing_time_for_36_cows_l195_195981

-- Defining the problem conditions and the question in Lean 4
theorem grazing_time_for_36_cows :
  ∀ (g r b : ℕ), 
    (24 * 6 * b = g + 6 * r) →
    (21 * 8 * b = g + 8 * r) →
    36 * 3 * b = g + 3 * r :=
by
  intros
  sorry

end grazing_time_for_36_cows_l195_195981


namespace tank_emptying_time_l195_195326

theorem tank_emptying_time (fill_without_leak fill_with_leak : ℝ) (h1 : fill_without_leak = 7) (h2 : fill_with_leak = 8) : 
  let R := 1 / fill_without_leak
  let L := R - 1 / fill_with_leak
  let emptying_time := 1 / L
  emptying_time = 56 :=
by
  sorry

end tank_emptying_time_l195_195326


namespace decimal_to_base8_conversion_l195_195878

theorem decimal_to_base8_conversion : (512 : ℕ) = 8^3 :=
by
  sorry

end decimal_to_base8_conversion_l195_195878


namespace minimum_prime_factorization_sum_l195_195559

theorem minimum_prime_factorization_sum (x y a b c d : ℕ) (hx : x > 0) (hy : y > 0)
  (h : 5 * x^7 = 13 * y^17) (h_pf: x = a ^ c * b ^ d) :
  a + b + c + d = 33 :=
sorry

end minimum_prime_factorization_sum_l195_195559


namespace lunch_to_novel_ratio_l195_195554

theorem lunch_to_novel_ratio 
  (initial_amount : ℕ) 
  (novel_cost : ℕ) 
  (remaining_after_mall : ℕ) 
  (spent_on_lunch : ℕ)
  (h1 : initial_amount = 50) 
  (h2 : novel_cost = 7) 
  (h3 : remaining_after_mall = 29) 
  (h4 : spent_on_lunch = initial_amount - novel_cost - remaining_after_mall) :
  spent_on_lunch / novel_cost = 2 := 
  sorry

end lunch_to_novel_ratio_l195_195554


namespace hat_p_at_1_l195_195474

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^2 - (1 + 1)*x + 1

-- Definition of displeased polynomial
def isDispleased (p : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 x3 x4 : ℝ), p (p x1) = 0 ∧ p (p x2) = 0 ∧ p (p x3) = 0 ∧ p (p x4) = 0

-- Define the specific polynomial hat_p
def hat_p (x : ℝ) : ℝ := p x

-- Theorem statement
theorem hat_p_at_1 : isDispleased hat_p → hat_p 1 = 0 :=
by
  sorry

end hat_p_at_1_l195_195474


namespace product_of_first_five_terms_l195_195079

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ m n p q : ℕ, m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 ∧ m + n = p + q → a m * a n = a p * a q

theorem product_of_first_five_terms 
  (h : geometric_sequence a) 
  (h3 : a 3 = 2) : 
  a 1 * a 2 * a 3 * a 4 * a 5 = 32 :=
sorry

end product_of_first_five_terms_l195_195079


namespace number_property_l195_195438

theorem number_property : ∀ n : ℕ, (∀ q : ℕ, q > 0 → n % q^2 < q^(q^2) / 2) ↔ n = 1 ∨ n = 4 :=
by sorry

end number_property_l195_195438


namespace find_new_curve_l195_195058

-- Define the given curve equation
noncomputable def given_curve (theta : ℝ) : ℝ := 5 * real.sqrt 3 * real.cos theta - 5 * real.sin theta

-- Define the new curve equation to be proven
noncomputable def new_curve (theta : ℝ) : ℝ := 10 * real.cos (theta - π / 6)

-- State the problem
theorem find_new_curve (theta : ℝ) :
    symmetric_polar_axis (given_curve theta) →
    (∀ rho, new_curve theta = rho) := sorry

def symmetric_polar_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ θ ρ, f θ ρ = f (-θ) ρ


end find_new_curve_l195_195058


namespace anna_clara_age_l195_195477

theorem anna_clara_age :
  ∃ x : ℕ, (54 - x) * 3 = 80 - x ∧ x = 41 :=
by
  sorry

end anna_clara_age_l195_195477


namespace age_difference_l195_195589

/-- The age difference between each child d -/
theorem age_difference (d : ℝ) 
  (h1 : ∃ a b c e : ℝ, d = a ∧ 2*d = b ∧ 3*d = c ∧ 4*d = e)
  (h2 : 12 + (12 - d) + (12 - 2*d) + (12 - 3*d) + (12 - 4*d) = 40) : 
  d = 2 := 
sorry

end age_difference_l195_195589


namespace fraction_of_y_l195_195921

theorem fraction_of_y (w x y : ℝ) (h1 : wx = y) 
  (h2 : (w + x) / 2 = 0.5) : 
  (2 / w + 2 / x = 2 / y) := 
by
  sorry

end fraction_of_y_l195_195921


namespace increase_by_percentage_l195_195624

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l195_195624


namespace range_of_a_l195_195998

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x0 : ℝ, x0^2 + 2 * a * x0 + 2 - a = 0) ↔ a ≤ -2 ∨ a = 1 :=
by sorry

end range_of_a_l195_195998


namespace vector_x_value_l195_195946

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def b : ℝ × ℝ := (1, 2)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_x_value (x : ℝ) : (perpendicular (a x) b) → x = -2 / 3 := by
  intro h
  sorry

end vector_x_value_l195_195946


namespace solve_equation_l195_195490

theorem solve_equation (x : ℝ) (h₀ : x ≠ 1) (h₁ : x ≠ -1) :
  ( -15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) + 1 )
  ↔ x = 5 / 4 ∨ x = -2 :=
by sorry

end solve_equation_l195_195490


namespace moles_required_to_form_2_moles_H2O_l195_195192

def moles_of_NH4NO3_needed (moles_of_H2O : ℕ) : ℕ := moles_of_H2O

theorem moles_required_to_form_2_moles_H2O :
  moles_of_NH4NO3_needed 2 = 2 := 
by 
  -- From the balanced equation 1 mole of NH4NO3 produces 1 mole of H2O
  -- Therefore, 2 moles of NH4NO3 are needed to produce 2 moles of H2O
  sorry

end moles_required_to_form_2_moles_H2O_l195_195192


namespace percentage_of_male_students_solved_l195_195248

variable (M F : ℝ)
variable (M_25 F_25 : ℝ)
variable (prob_less_25 : ℝ)

-- Conditions from the problem
def graduation_class_conditions (M F M_25 F_25 prob_less_25 : ℝ) : Prop :=
  M + F = 100 ∧
  M_25 = 0.50 * M ∧
  F_25 = 0.30 * F ∧
  (1 - 0.50) * M + (1 - 0.30) * F = prob_less_25 * 100

-- Theorem to prove
theorem percentage_of_male_students_solved (M F : ℝ) (M_25 F_25 prob_less_25 : ℝ) :
  graduation_class_conditions M F M_25 F_25 prob_less_25 → prob_less_25 = 0.62 → M = 40 :=
by
  sorry

end percentage_of_male_students_solved_l195_195248


namespace possible_combinations_of_scores_l195_195249

theorem possible_combinations_of_scores 
    (scores : Set ℕ := {0, 3, 5})
    (total_scores : ℕ := 32)
    (teams : ℕ := 3)
    : (∃ (number_of_combinations : ℕ), number_of_combinations = 255) := by
  sorry

end possible_combinations_of_scores_l195_195249


namespace simultaneous_equations_solution_l195_195711

-- Definition of the two equations
def eq1 (m x y : ℝ) : Prop := y = m * x + 5
def eq2 (m x y : ℝ) : Prop := y = (3 * m - 2) * x + 6

-- Lean theorem statement to check if the equations have a solution
theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 1) ↔ ∃ x y : ℝ, eq1 m x y ∧ eq2 m x y := 
sorry

end simultaneous_equations_solution_l195_195711


namespace chi_squared_confidence_l195_195321

theorem chi_squared_confidence (K_squared : ℝ) :
  (99.5 / 100 : ℝ) = 0.995 → (K_squared ≥ 7.879) :=
sorry

end chi_squared_confidence_l195_195321


namespace sticks_per_stool_is_two_l195_195097

-- Conditions
def sticks_from_chair := 6
def sticks_from_table := 9
def sticks_needed_per_hour := 5
def num_chairs := 18
def num_tables := 6
def num_stools := 4
def hours_to_keep_warm := 34

-- Question and Answer in Lean 4 statement
theorem sticks_per_stool_is_two : 
  (hours_to_keep_warm * sticks_needed_per_hour) - (num_chairs * sticks_from_chair + num_tables * sticks_from_table) = 2 * num_stools := 
  by
    sorry

end sticks_per_stool_is_two_l195_195097


namespace find_c_l195_195990

-- Given conditions
variables {a b c d e : ℕ} (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e)
variables (h6 : a + b = e - 1) (h7 : a * b = d + 1)

-- Required to prove
theorem find_c : c = 4 := by
  sorry

end find_c_l195_195990


namespace increase_by_percentage_l195_195619

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l195_195619


namespace baby_whales_on_second_trip_l195_195413

def iwishmael_whales_problem : Prop :=
  let male1 := 28
  let female1 := 2 * male1
  let male3 := male1 / 2
  let female3 := female1
  let total_whales := 178
  let total_without_babies := (male1 + female1) + (male3 + female3)
  total_whales - total_without_babies = 24

theorem baby_whales_on_second_trip : iwishmael_whales_problem :=
  by
  sorry

end baby_whales_on_second_trip_l195_195413


namespace neg_p_implies_neg_q_sufficient_but_not_necessary_l195_195510

variables (x : ℝ) (p : Prop) (q : Prop)

def p_condition := (1 < x ∨ x < -3)
def q_condition := (5 * x - 6 > x ^ 2)

theorem neg_p_implies_neg_q_sufficient_but_not_necessary :
  p_condition x → q_condition x → ((¬ p_condition x) → (¬ q_condition x)) :=
by 
  intro h1 h2
  sorry

end neg_p_implies_neg_q_sufficient_but_not_necessary_l195_195510


namespace pascals_triangle_contains_47_once_l195_195205

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l195_195205


namespace increase_by_percentage_l195_195625

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l195_195625


namespace investment_c_is_correct_l195_195992

-- Define the investments of a and b
def investment_a : ℕ := 45000
def investment_b : ℕ := 63000
def profit_c : ℕ := 24000
def total_profit : ℕ := 60000

-- Define the equation to find the investment of c
def proportional_share (x y total : ℕ) : Prop :=
  2 * (x + y + total) = 5 * total

-- The theorem to prove c's investment given the conditions
theorem investment_c_is_correct (c : ℕ) (h_proportional: proportional_share investment_a investment_b c) :
  c = 72000 :=
by
  sorry

end investment_c_is_correct_l195_195992


namespace max_distance_P_to_C2_area_of_triangle_ABC1_l195_195410

noncomputable def C1 : ℝ × ℝ → Prop :=
λ p, ∃ α : ℝ, p.1 = -2 + Real.cos α ∧ p.2 = -1 + Real.sin α

def C2 (p : ℝ × ℝ) : Prop :=
p.1 = 3

noncomputable def C3 (p : ℝ × ℝ) : Prop :=
p.2 = p.1

theorem max_distance_P_to_C2 : 
  ∀ P : ℝ × ℝ, C1 P → ∃ d, d = 6 :=
begin
  sorry
end

theorem area_of_triangle_ABC1 :
  ∀ A B : ℝ × ℝ, 
  C1 A ∧ C1 B ∧ C3 A ∧ C3 B →
  ∃ S, S = 1 / 2 :=
begin
  sorry
end

end max_distance_P_to_C2_area_of_triangle_ABC1_l195_195410


namespace pascal_triangle_47_number_of_rows_containing_47_l195_195227

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l195_195227


namespace smallest_sum_divisible_by_5_l195_195892

-- Definition of a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of four consecutive primes greater than 5
def four_consecutive_primes_greater_than_five (a b c d : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ a > 5 ∧ b > 5 ∧ c > 5 ∧ d > 5 ∧ 
  b = a + 4 ∧ c = b + 6 ∧ d = c + 2

-- The statement to prove
theorem smallest_sum_divisible_by_5 :
  (∃ a b c d : ℕ, four_consecutive_primes_greater_than_five a b c d ∧ (a + b + c + d) % 5 = 0 ∧
   ∀ x y z w : ℕ, four_consecutive_primes_greater_than_five x y z w → (x + y + z + w) % 5 = 0 → a + b + c + d ≤ x + y + z + w) →
  (∃ a b c d : ℕ, four_consecutive_primes_greater_than_five a b c d ∧ (a + b + c + d) = 60) :=
by
  sorry

end smallest_sum_divisible_by_5_l195_195892


namespace roots_quadratic_eq_l195_195056

theorem roots_quadratic_eq (a b : ℝ) (h1 : a^2 + 3*a - 4 = 0) (h2 : b^2 + 3*b - 4 = 0) (h3 : a + b = -3) : a^2 + 4*a + b - 3 = -2 :=
by
  sorry

end roots_quadratic_eq_l195_195056


namespace combined_selling_price_correct_l195_195862

def ArticleA_Cost : ℝ := 500
def ArticleA_Profit_Percent : ℝ := 0.45
def ArticleB_Cost : ℝ := 300
def ArticleB_Profit_Percent : ℝ := 0.30
def ArticleC_Cost : ℝ := 1000
def ArticleC_Profit_Percent : ℝ := 0.20
def Sales_Tax_Percent : ℝ := 0.12

def CombinedSellingPrice (A_cost A_profit_percent B_cost B_profit_percent C_cost C_profit_percent tax_percent : ℝ) : ℝ :=
  let A_selling_price := A_cost * (1 + A_profit_percent)
  let A_final_price := A_selling_price * (1 + tax_percent)
  let B_selling_price := B_cost * (1 + B_profit_percent)
  let B_final_price := B_selling_price * (1 + tax_percent)
  let C_selling_price := C_cost * (1 + C_profit_percent)
  let C_final_price := C_selling_price * (1 + tax_percent)
  A_final_price + B_final_price + C_final_price

theorem combined_selling_price_correct :
  CombinedSellingPrice ArticleA_Cost ArticleA_Profit_Percent ArticleB_Cost ArticleB_Profit_Percent ArticleC_Cost ArticleC_Profit_Percent Sales_Tax_Percent = 2592.8 := by
  sorry

end combined_selling_price_correct_l195_195862


namespace percentage_decrease_hours_worked_l195_195797

theorem percentage_decrease_hours_worked (B H : ℝ) (h₁ : H > 0) (h₂ : B > 0)
  (h_assistant1 : (1.8 * B) = B * 1.8) (h_assistant2 : (2 * (B / H)) = (1.8 * B) / (0.9 * H)) : 
  ((H - (0.9 * H)) / H) * 100 = 10 := 
by
  sorry

end percentage_decrease_hours_worked_l195_195797


namespace minimum_value_of_quadratic_function_l195_195794

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem minimum_value_of_quadratic_function 
  (f : ℝ → ℝ)
  (n : ℕ)
  (h1 : f n = 6)
  (h2 : f (n + 1) = 5)
  (h3 : f (n + 2) = 5)
  (hf : ∃ a b c : ℝ, f = quadratic_function a b c) :
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 5 :=
by
  sorry

end minimum_value_of_quadratic_function_l195_195794


namespace minimize_S_n_l195_195379

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

axiom arithmetic_sequence : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d
axiom sum_first_n_terms : ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * d)
axiom condition1 : a 0 + a 4 = -14
axiom condition2 : S 9 = -27

theorem minimize_S_n : ∃ n, ∀ m, S n ≤ S m := sorry

end minimize_S_n_l195_195379


namespace remainder_when_7n_divided_by_4_l195_195302

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l195_195302


namespace jefferson_high_school_ninth_graders_l195_195972

theorem jefferson_high_school_ninth_graders (total_students science_students arts_students students_taking_both : ℕ):
  total_students = 120 →
  science_students = 85 →
  arts_students = 65 →
  students_taking_both = 150 - 120 →
  science_students - students_taking_both = 55 :=
by
  sorry

end jefferson_high_school_ninth_graders_l195_195972


namespace additional_amount_needed_l195_195261

-- Definitions of the conditions
def shampoo_cost : ℝ := 10.00
def conditioner_cost : ℝ := 10.00
def lotion_cost : ℝ := 6.00
def lotions_count : ℕ := 3
def free_shipping_threshold : ℝ := 50.00

-- Calculating the total amount spent
def total_spent : ℝ :=
  shampoo_cost + conditioner_cost + lotions_count * lotion_cost

-- Required statement for the proof
theorem additional_amount_needed : 
  total_spent + 12.00 = free_shipping_threshold :=
by 
  -- Proof will be here
  sorry

end additional_amount_needed_l195_195261


namespace least_integer_square_condition_l195_195842

theorem least_integer_square_condition (x : ℤ) (h : x^2 = 3 * x + 36) : x = -6 :=
by sorry

end least_integer_square_condition_l195_195842


namespace divisible_by_6_l195_195568

theorem divisible_by_6 (n : ℤ) : 6 ∣ (n * (n + 1) * (n + 2)) :=
sorry

end divisible_by_6_l195_195568


namespace remainder_7n_mod_4_l195_195315

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l195_195315


namespace length_of_segment_AB_l195_195877

theorem length_of_segment_AB :
  ∀ A B : ℝ × ℝ,
  (∃ x y : ℝ, y^2 = 8 * x ∧ y = (y - 0) / (4 - 2) * (x - 2))
  ∧ (A.1 + B.1) / 2 = 4
  → dist A B = 12 := 
by
  sorry

end length_of_segment_AB_l195_195877


namespace sin_double_angle_solution_l195_195784

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l195_195784


namespace true_statement_D_l195_195668

-- Definitions related to the problem conditions
def supplementary_angles (a b : ℝ) : Prop := a + b = 180

def exterior_angle_sum_of_polygon (n : ℕ) : ℝ := 360

def acute_angle (a : ℝ) : Prop := a < 90

def triangle_inequality (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

-- The theorem to be proven based on the correct evaluation
theorem true_statement_D (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0):
  triangle_inequality a b c :=
by 
  sorry

end true_statement_D_l195_195668


namespace increase_by_percentage_l195_195599

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l195_195599


namespace num_ways_dist_6_balls_3_boxes_l195_195533

open Finset

/-- Number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem num_ways_dist_6_balls_3_boxes : 
  let num_ways := 
    let n := 6 in let k := 3 in
    ((choose n 6) + 
    (choose n 5) + 
    (choose n 4) + 
    (choose n 4) + 
    (choose n 3) / 2 + 
    (choose n 3) * (choose (n - 3) 2) + 
    (choose n 2) * (choose (n - 2) 2) / (factorial k)) 
  in num_ways = 122 := 
by 
  sorry

end num_ways_dist_6_balls_3_boxes_l195_195533


namespace increase_by_150_percent_l195_195649

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l195_195649


namespace max_marks_l195_195267

theorem max_marks (M S : ℕ) :
  (267 + 45 = 312) ∧ (312 = (45 * M) / 100) ∧ (292 + 38 = 330) ∧ (330 = (50 * S) / 100) →
  (M + S = 1354) :=
by
  sorry

end max_marks_l195_195267


namespace arccos_cos_of_11_l195_195023

-- Define the initial conditions
def angle_in_radians (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ 2 * Real.pi

def arccos_principal_range (a : ℝ) : Prop :=
  0 ≤ a ∧ a ≤ Real.pi

-- Define the main theorem to be proved
theorem arccos_cos_of_11 :
  angle_in_radians 11 →
  arccos_principal_range (Real.arccos (Real.cos 11)) →
  Real.arccos (Real.cos 11) = 4.71682 :=
by
  -- Proof is not required
  sorry

end arccos_cos_of_11_l195_195023


namespace sin_2phi_l195_195769

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l195_195769


namespace quadratic_roots_correct_l195_195284

theorem quadratic_roots_correct (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) := 
by
  sorry

end quadratic_roots_correct_l195_195284


namespace carols_total_peanuts_l195_195876

-- Define the initial number of peanuts Carol has
def initial_peanuts : ℕ := 2

-- Define the number of peanuts given by Carol's father
def peanuts_given : ℕ := 5

-- Define the total number of peanuts Carol has
def total_peanuts : ℕ := initial_peanuts + peanuts_given

-- The statement we need to prove
theorem carols_total_peanuts : total_peanuts = 7 := by
  sorry

end carols_total_peanuts_l195_195876


namespace probability_sum_odd_l195_195001

theorem probability_sum_odd :
  (∃ (balls : Finset ℕ), balls.card = 13 ∧ ∀ x ∈ balls, x ∈ Finset.range 14) →
  (∃ (drawnBalls : Finset ℕ), drawnBalls.card = 7) →
  (let oddBalls := (Finset.filter (λ x, x % 2 = 1) balls), evenBalls := (Finset.filter (λ x, x % 2 = 0) balls) in
    (∑ b in drawnBalls, b) % 2 = 1 →
    ((Finset.card oddBalls = 7 ∧ Finset.card evenBalls = 6) ∧
     Finset.card (Finset.filter (λ x, x % 2 = 1) drawnBalls) % 2 = 1 ∧
    ∑ _ in Finset.pairs oddBalls drawnBalls, _ + ∑ _ in Finset.pairs evenBalls drawnBalls, _ = 7)) →
  (let favorable := ∑ n in (Finset.filter (λ n, n ∈ [5, 3, 1]) (Finset.range 8)), 
       (Nat.choose 7 n) * (Nat.choose 6 (7 - n))) in
        favorable / Nat.choose 13 7 = 141 / 286) :=
sorry

end probability_sum_odd_l195_195001


namespace percent_nurses_with_neither_l195_195346

-- Define the number of nurses in each category
def total_nurses : ℕ := 150
def nurses_with_hbp : ℕ := 90
def nurses_with_ht : ℕ := 50
def nurses_with_both : ℕ := 30

-- Define a predicate that checks the conditions of the problem
theorem percent_nurses_with_neither :
  ((total_nurses - (nurses_with_hbp + nurses_with_ht - nurses_with_both)) * 100 : ℚ) / total_nurses = 2667 / 100 :=
by sorry

end percent_nurses_with_neither_l195_195346


namespace number_of_parrots_in_each_cage_l195_195342

theorem number_of_parrots_in_each_cage (num_cages : ℕ) (total_birds : ℕ) (parrots_per_cage parakeets_per_cage : ℕ)
    (h1 : num_cages = 9)
    (h2 : parrots_per_cage = parakeets_per_cage)
    (h3 : total_birds = 36)
    (h4 : total_birds = num_cages * (parrots_per_cage + parakeets_per_cage)) :
  parrots_per_cage = 2 :=
by
  sorry

end number_of_parrots_in_each_cage_l195_195342


namespace abs_diff_of_solutions_l195_195034

theorem abs_diff_of_solutions (h : ∀ x : ℝ, x^2 - 7 * x + 10 = 0 → x = 2 ∨ x = 5) :
  |(2 - 5 : ℝ)| = 3 :=
by sorry

end abs_diff_of_solutions_l195_195034


namespace find_b_l195_195829

theorem find_b (b : ℚ) (h : b * (-3) - (b - 1) * 5 = b - 3) : b = 8 / 9 :=
by
  sorry

end find_b_l195_195829


namespace sin_double_angle_l195_195767

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l195_195767


namespace fish_original_count_l195_195110

theorem fish_original_count (F : ℕ) (h : F / 2 - F / 6 = 12) : F = 36 := 
by 
  sorry

end fish_original_count_l195_195110


namespace find_M_l195_195165

theorem find_M : 
  ∃ M : ℚ, 
  (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
  M = 610 / 1657 :=
by
  sorry

end find_M_l195_195165


namespace total_pieces_of_clothing_l195_195936

-- Define Kaleb's conditions
def pieces_in_one_load : ℕ := 19
def num_equal_loads : ℕ := 5
def pieces_per_load : ℕ := 4

-- The total pieces of clothing Kaleb has
theorem total_pieces_of_clothing : pieces_in_one_load + num_equal_loads * pieces_per_load = 39 :=
by
  sorry

end total_pieces_of_clothing_l195_195936


namespace sum_coeffs_odd_exp_l195_195944

noncomputable def polynomial_expansion : Polynomial ℤ := Polynomial.X * 3 - 1

theorem sum_coeffs_odd_exp (a0 a1 a2 a3 a4 a5 a6 a7 : ℤ) :
  (polynomial_expansion ^ 7).coeff 0 = a0 →
  (polynomial_expansion ^ 7).coeff 1 = a1 →
  (polynomial_expansion ^ 7).coeff 2 = a2 →
  (polynomial_expansion ^ 7).coeff 3 = a3 →
  (polynomial_expansion ^ 7).coeff 4 = a4 →
  (polynomial_expansion ^ 7).coeff 5 = a5 →
  (polynomial_expansion ^ 7).coeff 6 = a6 →
  (polynomial_expansion ^ 7).coeff 7 = a7 →
  a1 + a3 + a5 + a7 = 8256 := by
  sorry

end sum_coeffs_odd_exp_l195_195944


namespace distance_is_20_sqrt_6_l195_195836

-- Definitions for problem setup
def distance_between_parallel_lines (r d : ℝ) : Prop :=
  ∃ O C D E F P Q : ℝ, 
  40^2 * 40 + (d / 2)^2 * 40 = 40 * r^2 ∧ 
  15^2 * 30 + (d / 2)^2 * 30 = 30 * r^2

-- The main statement to be proved
theorem distance_is_20_sqrt_6 :
  ∀ r d : ℝ,
  distance_between_parallel_lines r d →
  d = 20 * Real.sqrt 6 :=
sorry

end distance_is_20_sqrt_6_l195_195836


namespace find_two_digit_ab_l195_195552

def digit_range (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def different_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_two_digit_ab (A B C D : ℕ) (hA : digit_range A) (hB : digit_range B)
                         (hC : digit_range C) (hD : digit_range D)
                         (h_diff : different_digits A B C D)
                         (h_eq : (100 * A + 10 * B + C) * (10 * A + B) + C * D = 2017) :
  10 * A + B = 14 :=
sorry

end find_two_digit_ab_l195_195552


namespace find_positive_real_number_l195_195491

theorem find_positive_real_number (x : ℝ) (hx : x = 25 + 2 * Real.sqrt 159) :
  1 / 2 * (3 * x ^ 2 - 1) = (x ^ 2 - 50 * x - 10) * (x ^ 2 + 25 * x + 5) :=
by
  sorry

end find_positive_real_number_l195_195491


namespace seashells_given_joan_to_mike_l195_195414

-- Declaring the context for the problem: Joan's seashells
def initial_seashells := 79
def remaining_seashells := 16

-- Proving how many seashells Joan gave to Mike
theorem seashells_given_joan_to_mike : (initial_seashells - remaining_seashells) = 63 :=
by
  -- This proof needs to be completed
  sorry

end seashells_given_joan_to_mike_l195_195414


namespace least_possible_value_l195_195994

theorem least_possible_value (x y z : ℕ) (hx : 2 * x = 5 * y) (hy : 5 * y = 8 * z) (hz : 8 * z = 2 * x) (hnz_x: x > 0) (hnz_y: y > 0) (hnz_z: z > 0) :
  x + y + z = 33 :=
sorry

end least_possible_value_l195_195994


namespace integer_xyz_zero_l195_195672

theorem integer_xyz_zero (x y z : ℤ) (h : x^2 + y^2 + z^2 = 2 * x * y * z) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end integer_xyz_zero_l195_195672


namespace sum_of_areas_of_rectangles_l195_195834

theorem sum_of_areas_of_rectangles :
  let width := 2
  let lengths := [1, 4, 9, 16, 25, 36]
  let areas := lengths.map (λ l => l * width)
  let total_area := areas.sum
  total_area = 182 := by
  sorry

end sum_of_areas_of_rectangles_l195_195834


namespace total_tour_time_l195_195095

-- Declare constants for distances
def distance1 : ℝ := 55
def distance2 : ℝ := 40
def distance3 : ℝ := 70
def extra_miles : ℝ := 10

-- Declare constants for speeds
def speed1_part1 : ℝ := 60
def speed1_part2 : ℝ := 40
def speed2 : ℝ := 45
def speed3_part1 : ℝ := 45
def speed3_part2 : ℝ := 35
def speed3_part3 : ℝ := 50
def return_speed : ℝ := 55

-- Declare constants for stop times
def stop1 : ℝ := 1
def stop2 : ℝ := 1.5
def stop3 : ℝ := 2

-- Prove the total time required for the tour
theorem total_tour_time :
  (30 / speed1_part1) + (25 / speed1_part2) + stop1 +
  (distance2 / speed2) + stop2 +
  (20 / speed3_part1) + (30 / speed3_part2) + (20 / speed3_part3) + stop3 +
  ((distance1 + distance2 + distance3 + extra_miles) / return_speed) = 11.40 :=
by
  sorry

end total_tour_time_l195_195095


namespace compare_numbers_l195_195483

theorem compare_numbers :
  3 * 10^5 < 2 * 10^6 ∧ -2 - 1 / 3 > -3 - 1 / 2 := by
  sorry

end compare_numbers_l195_195483


namespace symmetry_axis_of_function_l195_195074

noncomputable def f (varphi : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (2 * x + varphi)

theorem symmetry_axis_of_function
  (varphi : ℝ) (h1 : |varphi| < Real.pi / 2)
  (h2 : f varphi (Real.pi / 6) = 1) :
  ∃ k : ℤ, (k * Real.pi / 2 + Real.pi / 3 = Real.pi / 3) :=
sorry

end symmetry_axis_of_function_l195_195074


namespace amount_added_to_doubled_number_l195_195837

theorem amount_added_to_doubled_number (N A : ℝ) (h1 : N = 6.0) (h2 : 2 * N + A = 17) : A = 5.0 :=
by
  sorry

end amount_added_to_doubled_number_l195_195837


namespace pascal_triangle_contains_prime_l195_195214

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l195_195214


namespace max_value_of_sum_of_cubes_l195_195941

theorem max_value_of_sum_of_cubes (a b c d e : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * Real.sqrt 5 := by
  sorry

end max_value_of_sum_of_cubes_l195_195941


namespace increase_by_150_percent_l195_195652

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l195_195652


namespace remainder_7n_mod_4_l195_195316

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l195_195316


namespace quadratic_function_equal_values_l195_195718

theorem quadratic_function_equal_values (a m n : ℝ) (h : a ≠ 0) (hmn : a * m^2 - 4 * a * m - 3 = a * n^2 - 4 * a * n - 3) : m + n = 4 :=
by
  sorry

end quadratic_function_equal_values_l195_195718


namespace prob_score_3_points_l195_195980

-- Definitions for the probabilities
def probability_hit_A := 3/4
def score_hit_A := 1
def score_miss_A := -1

def probability_hit_B := 2/3
def score_hit_B := 2
def score_miss_B := 0

-- Conditional probabilities and their calculations
noncomputable def prob_scenario_1 : ℚ := 
  probability_hit_A * 2 * probability_hit_B * (1 - probability_hit_B)

noncomputable def prob_scenario_2 : ℚ := 
  (1 - probability_hit_A) * probability_hit_B^2

noncomputable def total_prob : ℚ := 
  prob_scenario_1 + prob_scenario_2

-- The final proof statement
theorem prob_score_3_points : total_prob = 4/9 := sorry

end prob_score_3_points_l195_195980


namespace minimum_value_f_l195_195388

noncomputable def f (x : ℕ) (hx : x > 0) : ℝ := (x^2 + 33 : ℝ) / x

theorem minimum_value_f : ∃ x ∈ {x : ℕ | x > 0}, f x (by exact x_pos_proof) = 23 / 2 := 
by
  use 6
  split
  -- 6 is indeed a positive natural number
  norm_num
  -- showing the function evaluated at 6 is 23/2
  unfold f
sorry

end minimum_value_f_l195_195388


namespace least_common_denominator_l195_195874

-- We first need to define the function to compute the LCM of a list of natural numbers.
def lcm_list (ns : List ℕ) : ℕ :=
ns.foldr Nat.lcm 1

theorem least_common_denominator : 
  lcm_list [3, 4, 5, 8, 9, 11] = 3960 := 
by
  -- Here's where the proof would go
  sorry

end least_common_denominator_l195_195874


namespace algebraic_expression_value_l195_195715

theorem algebraic_expression_value (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : a^2 - b^2 = 15 :=
by
  sorry

end algebraic_expression_value_l195_195715


namespace find_values_of_a_l195_195708

noncomputable def has_one_real_solution (a : ℝ) : Prop :=
  ∃ x: ℝ, (x^3 - a*x^2 - 3*a*x + a^2 - 1 = 0) ∧ (∀ y: ℝ, (y^3 - a*y^2 - 3*a*y + a^2 - 1 = 0) → y = x)

theorem find_values_of_a : ∀ a: ℝ, has_one_real_solution a ↔ a < -(5 / 4) :=
by
  sorry

end find_values_of_a_l195_195708


namespace remainder_when_7n_divided_by_4_l195_195305

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l195_195305


namespace sin_double_angle_l195_195756

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l195_195756


namespace range_of_m_l195_195402

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - 4*x ≥ m) → m ≤ -3 :=
by
  intro h
  sorry

end range_of_m_l195_195402


namespace geometric_product_is_geometric_l195_195047

theorem geometric_product_is_geometric (q : ℝ) (a : ℕ → ℝ)
  (h_geo : ∀ n, a (n + 1) = q * a n) :
  ∀ n, (a n) * (a (n + 1)) = (q^2) * (a (n - 1) * a n) := by
  sorry

end geometric_product_is_geometric_l195_195047


namespace find_m_l195_195515

theorem find_m (m : ℝ) 
    (h1 : ∃ (m: ℝ), ∀ x y : ℝ, x - m * y + 2 * m = 0) 
    (h2 : ∃ (m: ℝ), ∀ x y : ℝ, x + 2 * y - m = 0) 
    (perpendicular : (1/m) * (-1/2) = -1) : m = 1/2 :=
sorry

end find_m_l195_195515


namespace perimeter_eq_120_plus_2_sqrt_1298_l195_195408

noncomputable def total_perimeter_of_two_quadrilaterals (AB BC CD : ℝ) (AC : ℝ := Real.sqrt (AB ^ 2 + BC ^ 2)) (AD : ℝ := Real.sqrt (AC ^ 2 + CD ^ 2)) : ℝ :=
2 * (AB + BC + CD + AD)

theorem perimeter_eq_120_plus_2_sqrt_1298 (hAB : AB = 15) (hBC : BC = 28) (hCD : CD = 17) :
  total_perimeter_of_two_quadrilaterals 15 28 17 = 120 + 2 * Real.sqrt 1298 :=
by
  sorry

end perimeter_eq_120_plus_2_sqrt_1298_l195_195408


namespace sin_2phi_l195_195772

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l195_195772


namespace relatively_prime_2n_plus_1_4n2_plus_1_l195_195093

theorem relatively_prime_2n_plus_1_4n2_plus_1 (n : ℕ) (h : n > 0) : 
  Nat.gcd (2 * n + 1) (4 * n^2 + 1) = 1 := 
by
  sorry

end relatively_prime_2n_plus_1_4n2_plus_1_l195_195093


namespace initial_breads_count_l195_195327

theorem initial_breads_count :
  ∃ (X : ℕ), ((((X / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2 = 3 ∧ X = 127 :=
by sorry

end initial_breads_count_l195_195327


namespace group_membership_l195_195971

theorem group_membership (n : ℕ) (h1 : n % 7 = 4) (h2 : n % 11 = 6) (h3 : 100 ≤ n ∧ n ≤ 200) :
  n = 116 ∨ n = 193 :=
sorry

end group_membership_l195_195971


namespace fruits_left_l195_195934

theorem fruits_left (plums guavas apples given : ℕ) (h1 : plums = 16) (h2 : guavas = 18) (h3 : apples = 21) (h4 : given = 40) : 
  (plums + guavas + apples - given = 15) :=
by
  sorry

end fruits_left_l195_195934


namespace find_m_l195_195199

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 36 m = 108) (h2 : Nat.lcm 45 m = 180) : m = 72 := 
by 
  sorry

end find_m_l195_195199


namespace school_total_payment_l195_195574

theorem school_total_payment
  (price : ℕ)
  (kindergarten_models : ℕ)
  (elementary_library_multiplier : ℕ)
  (model_reduction_percentage : ℚ)
  (total_models : ℕ)
  (reduced_price : ℚ)
  (total_payment : ℚ)
  (h1 : price = 100)
  (h2 : kindergarten_models = 2)
  (h3 : elementary_library_multiplier = 2)
  (h4 : model_reduction_percentage = 0.05)
  (h5 : total_models = kindergarten_models + (kindergarten_models * elementary_library_multiplier))
  (h6 : total_models > 5)
  (h7 : reduced_price = price - (price * model_reduction_percentage))
  (h8 : total_payment = total_models * reduced_price) :
  total_payment = 570 := 
by
  sorry

end school_total_payment_l195_195574


namespace least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22_l195_195292

def least_subtrahend (n m : ℕ) (k : ℕ) : Prop :=
  (n - k) % m = 0 ∧ ∀ k' : ℕ, k' < k → (n - k') % m ≠ 0

theorem least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22 :
  least_subtrahend 102932847 25 22 :=
sorry

end least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22_l195_195292


namespace min_value_of_a_l195_195584

noncomputable def smallest_root_sum : ℕ := 78

theorem min_value_of_a (r s t : ℕ) (h1 : r * s * t = 2310) (h2 : r > 0) (h3 : s > 0) (h4 : t > 0) :
  r + s + t = smallest_root_sum :=
sorry

end min_value_of_a_l195_195584


namespace pascal_triangle_47_l195_195216

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l195_195216


namespace star_example_l195_195717

section star_operation

variables (x y z : ℕ) 

-- Define the star operation as a binary function
def star (a b : ℕ) : ℕ := a * b

-- Given conditions
axiom star_idempotent : ∀ x : ℕ, star x x = 0
axiom star_associative : ∀ x y z : ℕ, star x (star y z) = (star x y) + z

-- Main theorem to be proved
theorem star_example : star 1993 1935 = 58 :=
sorry

end star_operation

end star_example_l195_195717


namespace monthly_fixed_cost_is_correct_l195_195337

-- Definitions based on the conditions in the problem
def production_cost_per_component : ℕ := 80
def shipping_cost_per_component : ℕ := 5
def components_per_month : ℕ := 150
def minimum_price_per_component : ℕ := 195

-- Monthly fixed cost definition based on the provided solution
def monthly_fixed_cost := components_per_month * (minimum_price_per_component - (production_cost_per_component + shipping_cost_per_component))

-- Theorem stating that the calculated fixed cost is correct.
theorem monthly_fixed_cost_is_correct : monthly_fixed_cost = 16500 :=
by
  unfold monthly_fixed_cost
  norm_num
  sorry

end monthly_fixed_cost_is_correct_l195_195337


namespace determine_k_for_one_real_solution_l195_195377

theorem determine_k_for_one_real_solution (k : ℝ):
  (∃ x : ℝ, 9 * x^2 + k * x + 49 = 0 ∧ (∀ y : ℝ, 9 * y^2 + k * y + 49 = 0 → y = x)) → k = 42 :=
sorry

end determine_k_for_one_real_solution_l195_195377


namespace num_sides_regular_polygon_l195_195155

-- Define the perimeter and side length of the polygon
def perimeter : ℝ := 160
def side_length : ℝ := 10

-- Theorem to prove the number of sides
theorem num_sides_regular_polygon : 
  (perimeter / side_length) = 16 := by
    sorry  -- Proof is omitted

end num_sides_regular_polygon_l195_195155


namespace quadratic_inequality_solution_range_l195_195916

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1 / 2 > 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end quadratic_inequality_solution_range_l195_195916


namespace sequence_sum_l195_195903

theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  a 1 = 2 ∧ a 2 = 2 ∧
  (∀ n : ℕ, n > 0 → a (n + 2) = (1 + Real.cos (n * Real.pi)) * (a n - 1) + 2) →
  (∀ n : ℕ, S (2 * n) = ∑ k in Finset.range (2 * n + 1), a k) →
  (∀ n : ℕ, S (2 * n) = 2 ^ (n + 1) + 2 * n - 2) :=
sorry

end sequence_sum_l195_195903


namespace Dan_tshirts_total_l195_195361

theorem Dan_tshirts_total :
  (let rate1 := 1 / 12
   let rate2 := 1 / 6
   let hour := 60
   let tshirts_first_hour := hour * rate1
   let tshirts_second_hour := hour * rate2
   let total_tshirts := tshirts_first_hour + tshirts_second_hour
   total_tshirts) = 15 := by
  sorry

end Dan_tshirts_total_l195_195361


namespace sin_double_angle_l195_195766

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l195_195766


namespace whitney_greatest_sets_l195_195323

-- Define the conditions: Whitney has 4 T-shirts and 20 buttons.
def num_tshirts := 4
def num_buttons := 20

-- The problem statement: Prove that the greatest number of sets Whitney can make is 4.
theorem whitney_greatest_sets : Nat.gcd num_tshirts num_buttons = 4 := by
  sorry

end whitney_greatest_sets_l195_195323


namespace remainder_140_div_k_l195_195045

theorem remainder_140_div_k (k : ℕ) (hk : k > 0) :
  (80 % k^2 = 8) → (140 % k = 2) :=
by
  sorry

end remainder_140_div_k_l195_195045


namespace find_number_l195_195000

theorem find_number (x : ℝ) : ((x - 50) / 4) * 3 + 28 = 73 → x = 110 := 
  by 
  sorry

end find_number_l195_195000


namespace range_of_z_l195_195512

theorem range_of_z (x y : ℝ) (h1 : -4 ≤ x - y ∧ x - y ≤ -1) (h2 : -1 ≤ 4 * x - y ∧ 4 * x - y ≤ 5) :
  ∃ (z : ℝ), z = 9 * x - y ∧ -1 ≤ z ∧ z ≤ 20 :=
sorry

end range_of_z_l195_195512


namespace smallest_number_l195_195453

theorem smallest_number (n : ℕ) : 
  (∃ n, ∀ m : ℕ, (m + 2) % 12 = 0 ∧ (m + 2) % 30 = 0 ∧ (m + 2) % 48 = 0 ∧ (m + 2) % 74 = 0 ∧ (m + 2) % 100 = 0 → n = m → m = 44398) :=
begin
  sorry -- the proof is omitted 
end

end smallest_number_l195_195453


namespace remainder_7n_mod_4_l195_195312

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l195_195312


namespace sin_double_angle_l195_195755

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l195_195755


namespace f_g_of_4_l195_195942

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt x + 12 / Real.sqrt x
noncomputable def g (x : ℝ) : ℝ := 3 * x^2 - x - 4

theorem f_g_of_4 : f (g 4) = 23 * Real.sqrt 10 / 5 := by
  sorry

end f_g_of_4_l195_195942


namespace meaningful_expression_iff_l195_195974

noncomputable theory
open Classical -- For dealing with classical logic (if necessary)

-- Define the conditions as Lean terms
def meaningful_expression (x : ℝ) : Prop := (x + 1 ≥ 0) ∧ (x ≠ 0)

-- Theorem stating the equivalent condition
theorem meaningful_expression_iff {x : ℝ} :
  meaningful_expression x ↔ (x ≥ -1 ∧ x ≠ 0) :=
begin
  sorry, -- Proof is left out
end

end meaningful_expression_iff_l195_195974


namespace k_value_l195_195707

open Real

noncomputable def k_from_roots (α β : ℝ) : ℝ := - (α + β)

theorem k_value (k : ℝ) (α β : ℝ) (h1 : α + β = -k) (h2 : α * β = 8) (h3 : (α+3) + (β+3) = k) (h4 : (α+3) * (β+3) = 12) : k = 3 :=
by
  -- Here we skip the proof as instructed.
  sorry

end k_value_l195_195707


namespace paintings_total_l195_195698

def june_paintings : ℕ := 2
def july_paintings : ℕ := 2 * june_paintings
def august_paintings : ℕ := 3 * july_paintings
def total_paintings : ℕ := june_paintings + july_paintings + august_paintings

theorem paintings_total : total_paintings = 18 :=
by {
  sorry
}

end paintings_total_l195_195698


namespace seedlings_planted_by_father_l195_195430

theorem seedlings_planted_by_father (remi_day1_seedlings : ℕ) (total_seedlings : ℕ) :
  remi_day1_seedlings = 200 →
  total_seedlings = 1200 →
  let remi_day2_seedlings := 2 * remi_day1_seedlings in
  total_seedlings = remi_day1_seedlings + remi_day2_seedlings + 600 :=
begin
  assume h1 h2,
  sorry,
end

end seedlings_planted_by_father_l195_195430


namespace area_of_given_field_l195_195117

noncomputable def area_of_field (cost_in_rupees : ℕ) (rate_per_meter_in_paise : ℕ) (ratio_width : ℕ) (ratio_length : ℕ) : ℕ :=
  let cost_in_paise := cost_in_rupees * 100
  let perimeter := (ratio_width + ratio_length) * 2
  let x := cost_in_paise / (perimeter * rate_per_meter_in_paise)
  let width := ratio_width * x
  let length := ratio_length * x
  width * length

theorem area_of_given_field :
  let cost_in_rupees := 105
  let rate_per_meter_in_paise := 25
  let ratio_width := 3
  let ratio_length := 4
  area_of_field cost_in_rupees rate_per_meter_in_paise ratio_width ratio_length = 10800 :=
by
  sorry

end area_of_given_field_l195_195117


namespace interest_earned_is_91_dollars_l195_195963

-- Define the initial conditions
def P : ℝ := 2000
def r : ℝ := 0.015
def n : ℕ := 3

-- Define the compounded amount function
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

-- Prove the interest earned after 3 years is 91 dollars
theorem interest_earned_is_91_dollars : 
  (compound_interest P r n) - P = 91 :=
by
  sorry

end interest_earned_is_91_dollars_l195_195963


namespace pascal_triangle_47_rows_l195_195221

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l195_195221


namespace increasing_iff_a_ge_half_l195_195519

noncomputable def f (a x : ℝ) : ℝ := (2 / 3) * x ^ 3 + (1 / 2) * (a - 1) * x ^ 2 + a * x + 1

theorem increasing_iff_a_ge_half (a : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → (2 * x ^ 2 + (a - 1) * x + a) ≥ 0) ↔ a ≥ -1 / 2 :=
sorry

end increasing_iff_a_ge_half_l195_195519


namespace increase_80_by_150_percent_l195_195612

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l195_195612


namespace find_k_l195_195069

-- Defining the quadratic function
def quadratic (x k : ℝ) := x^2 + (2 * k + 1) * x + k^2 + 1

-- Condition 1: The roots are distinct, implies discriminant > 0
def discriminant_positive (k : ℝ) := (2 * k + 1)^2 - 4 * (k^2 + 1) > 0

-- Condition 2: Product of roots given as 5
def product_of_roots (k : ℝ) := k^2 + 1 = 5

-- Main theorem
theorem find_k (k : ℝ) (hk1 : discriminant_positive k) (hk2 : product_of_roots k) : k = 2 := by
  sorry

end find_k_l195_195069


namespace probability_Rachel_Robert_in_picture_l195_195100

noncomputable def Rachel_lap_time := 75
noncomputable def Robert_lap_time := 70
noncomputable def photo_time_start := 900
noncomputable def photo_time_end := 960
noncomputable def track_fraction := 1 / 5

theorem probability_Rachel_Robert_in_picture :
  let lap_time_Rachel := Rachel_lap_time
  let lap_time_Robert := Robert_lap_time
  let time_start := photo_time_start
  let time_end := photo_time_end
  let interval_Rachel := 15  -- ±15 seconds for Rachel
  let interval_Robert := 14  -- ±14 seconds for Robert
  let probability := (2 * interval_Robert) / (time_end - time_start) 
  probability = 7 / 15 :=
by
  sorry

end probability_Rachel_Robert_in_picture_l195_195100


namespace pascal_triangle_47_l195_195218

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l195_195218


namespace jared_sent_in_november_l195_195935

noncomputable def text_messages (n : ℕ) : ℕ :=
  match n with
  | 0 => 1  -- November
  | 1 => 2  -- December
  | 2 => 4  -- January
  | 3 => 8  -- February
  | 4 => 16 -- March
  | _ => 0

theorem jared_sent_in_november : text_messages 0 = 1 :=
sorry

end jared_sent_in_november_l195_195935


namespace part1_part2_l195_195065

theorem part1 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) : k > 3 / 4 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) (hx1x2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 → x1 * x2 = 5) : k = 2 :=
sorry

end part1_part2_l195_195065


namespace molecular_weight_N2O3_correct_l195_195174

/-- Conditions -/
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

/-- Proof statement -/
theorem molecular_weight_N2O3_correct :
  (2 * atomic_weight_N + 3 * atomic_weight_O) = 76.02 ∧
  name_of_N2O3 = "dinitrogen trioxide" := sorry

/-- Definition of the compound name based on formula -/
def name_of_N2O3 : String := "dinitrogen trioxide"

end molecular_weight_N2O3_correct_l195_195174


namespace find_first_term_of_arithmetic_progression_l195_195193

-- Definitions for the proof
def arithmetic_progression_first_term (L n d : ℕ) : ℕ :=
  L - (n - 1) * d

-- Theorem stating the proof problem
theorem find_first_term_of_arithmetic_progression (L n d : ℕ) (hL : L = 62) (hn : n = 31) (hd : d = 2) :
  arithmetic_progression_first_term L n d = 2 :=
by
  -- proof omitted
  sorry

end find_first_term_of_arithmetic_progression_l195_195193


namespace increase_150_percent_of_80_l195_195628

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l195_195628


namespace prob_A_prob_B_prob_two_computers_l195_195683

open ProbabilityTheory

def repair_events :=
  { A0 := 0.75, A1 := 0.15, A2 := 0.06, A3 := 0.04 }

def P (A : String) : ℝ :=
  match A with
  | "A0" => repair_events.A0
  | "A1" => repair_events.A1
  | "A2" => repair_events.A2
  | "A3" => repair_events.A3
  | _ => 0

theorem prob_A : P("A1") + P("A2") + P("A3") = 0.25 :=
by sorry

theorem prob_B : P("A0") + P("A1") = 0.9 :=
by sorry

def two_computers_prob (X Y : String) : ℝ :=
  P(X) * P(Y)

theorem prob_two_computers : 
  two_computers_prob "A0" "A0"
  + 2 * two_computers_prob "A0" "A1"
  + 2 * two_computers_prob "A0" "A2"
  + two_computers_prob "A1" "A1" = 0.9 :=
by sorry

end prob_A_prob_B_prob_two_computers_l195_195683


namespace arithmetic_progression_contains_sixth_power_l195_195355

theorem arithmetic_progression_contains_sixth_power (a b : ℕ) (h_ap_pos : ∀ t : ℕ, a + b * t > 0)
  (h_contains_square : ∃ n : ℕ, ∃ t : ℕ, a + b * t = n^2)
  (h_contains_cube : ∃ m : ℕ, ∃ t : ℕ, a + b * t = m^3) :
  ∃ k : ℕ, ∃ t : ℕ, a + b * t = k^6 :=
sorry

end arithmetic_progression_contains_sixth_power_l195_195355


namespace minimum_vertical_distance_l195_195828

noncomputable def absolute_value (x : ℝ) : ℝ := abs x

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 - 3 * x - 5

theorem minimum_vertical_distance :
  ∃ x : ℝ, (∀ y : ℝ, |absolute_value y - quadratic_function y| ≥ 4) ∧ (|absolute_value x - quadratic_function x| = 4) := 
sorry

end minimum_vertical_distance_l195_195828


namespace increase_result_l195_195644

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l195_195644


namespace increase_result_l195_195642

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l195_195642


namespace increase_by_150_percent_l195_195653

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l195_195653


namespace find_larger_number_l195_195580

theorem find_larger_number (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : L = 1635 :=
sorry

end find_larger_number_l195_195580


namespace longest_side_of_triangle_l195_195970

theorem longest_side_of_triangle :
  ∃ y : ℚ, 6 + (y + 3) + (3 * y - 2) = 40 ∧ max (6 : ℚ) (max (y + 3) (3 * y - 2)) = 91 / 4 :=
by
  sorry

end longest_side_of_triangle_l195_195970


namespace work_completion_in_days_l195_195682

noncomputable def work_days_needed : ℕ :=
  let A_rate := 1 / 9
  let B_rate := 1 / 18
  let C_rate := 1 / 12
  let D_rate := 1 / 24
  let AB_rate := A_rate + B_rate
  let CD_rate := C_rate + D_rate
  let two_day_work := AB_rate + CD_rate
  let total_cycles := 24 / 7
  let total_days := (if total_cycles % 1 = 0 then total_cycles else total_cycles + 1) * 2
  total_days

theorem work_completion_in_days :
  work_days_needed = 8 :=
by
  sorry

end work_completion_in_days_l195_195682


namespace Annabelle_saved_12_dollars_l195_195018

def weekly_allowance : ℕ := 30
def spent_on_junk_food : ℕ := weekly_allowance / 3
def spent_on_sweets : ℕ := 8
def total_spent : ℕ := spent_on_junk_food + spent_on_sweets
def saved_amount : ℕ := weekly_allowance - total_spent

theorem Annabelle_saved_12_dollars : saved_amount = 12 := by
  -- proof goes here
  sorry

end Annabelle_saved_12_dollars_l195_195018


namespace specific_time_l195_195977

theorem specific_time :
  (∀ (s : ℕ), 0 ≤ s ∧ s ≤ 7 → (∃ (t : ℕ), (t ^ 2 + 2 * t) - (3 ^ 2 + 2 * 3) = 20 ∧ t = 5)) :=
  by sorry

end specific_time_l195_195977


namespace urn_problem_l195_195167

theorem urn_problem : 
  (5 / 12 * 20 / (20 + M) + 7 / 12 * M / (20 + M) = 0.62) → M = 111 :=
by
  intro h
  sorry

end urn_problem_l195_195167


namespace find_x_l195_195070

def vec (x y : ℝ) := (x, y)

def a := vec 1 (-4)
def b (x : ℝ) := vec (-1) x
def c (x : ℝ) := (a.1 + 3 * (b x).1, a.2 + 3 * (b x).2)

theorem find_x (x : ℝ) : a.1 * (c x).2 = (c x).1 * a.2 → x = 4 :=
by
  sorry

end find_x_l195_195070


namespace x_coordinate_of_second_point_l195_195081

variable (m n : ℝ)

theorem x_coordinate_of_second_point
  (h1 : m = 2 * n + 5)
  (h2 : (m + 5) = 2 * (n + 2.5) + 5) :
  (m + 5) = m + 5 :=
by
  sorry

end x_coordinate_of_second_point_l195_195081


namespace A_times_B_is_correct_l195_195802

noncomputable def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {x : ℝ | x ≥ 0}

noncomputable def A_union_B : Set ℝ := {x : ℝ | x ≥ 0}
noncomputable def A_inter_B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

noncomputable def A_times_B : Set ℝ := {x : ℝ | x ∈ A_union_B ∧ x ∉ A_inter_B}

theorem A_times_B_is_correct :
  A_times_B = {x : ℝ | x > 2} := sorry

end A_times_B_is_correct_l195_195802


namespace probability_of_diff_by_three_is_one_eighth_l195_195162

-- Define the problem within a namespace
namespace DiceRoll

-- Define the probability of rolling two integers that differ by 3 on an 8-sided die
noncomputable def prob_diff_by_three : ℚ :=
  let successful_outcomes := 8
  let total_outcomes := 8 * 8
  successful_outcomes / total_outcomes

-- The main theorem
theorem probability_of_diff_by_three_is_one_eighth :
  prob_diff_by_three = 1 / 8 := by
  sorry

end DiceRoll

end probability_of_diff_by_three_is_one_eighth_l195_195162


namespace remainder_7n_mod_4_l195_195318

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l195_195318


namespace range_of_m_l195_195380

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -abs (x + 3) + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x > g x m) → m < 5 :=
by
  sorry

end range_of_m_l195_195380


namespace option_d_correct_l195_195454

theorem option_d_correct (a b : ℝ) : 2 * a^2 * b - 4 * a^2 * b = -2 * a^2 * b :=
by
  sorry

end option_d_correct_l195_195454


namespace increase_80_by_150_percent_l195_195618

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l195_195618


namespace triangle_CD_length_l195_195938

noncomputable def triangle_AB_values : ℝ := 4024
noncomputable def triangle_AC_values : ℝ := 4024
noncomputable def triangle_BC_values : ℝ := 2012
noncomputable def CD_value : ℝ := 504.5

theorem triangle_CD_length 
  (AB AC : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (h1 : AB = triangle_AB_values)
  (h2 : AC = triangle_AC_values)
  (h3 : BC = triangle_BC_values) :
  CD = CD_value := by
  sorry

end triangle_CD_length_l195_195938


namespace volume_conversion_l195_195863

theorem volume_conversion (v_feet : ℕ) (h : v_feet = 250) : (v_feet / 27 : ℚ) = 250 / 27 := by
  sorry

end volume_conversion_l195_195863


namespace isosceles_triangle_perimeter_l195_195740

theorem isosceles_triangle_perimeter
  (a b : ℕ)
  (ha : a = 3)
  (hb : b = 7)
  (h_iso : ∃ (c : ℕ), (c = a ∨ c = b) ∧ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  a + b + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l195_195740


namespace perfect_square_trinomial_implies_value_of_a_l195_195279

theorem perfect_square_trinomial_implies_value_of_a (a : ℝ) :
  (∃ (b : ℝ), (∃ (x : ℝ), (x^2 - ax + 9 = 0) ∧ (x + b)^2 = x^2 - ax + 9)) ↔ a = 6 ∨ a = -6 :=
by
  sorry

end perfect_square_trinomial_implies_value_of_a_l195_195279


namespace math_problem_l195_195418

theorem math_problem (f_star f_ast : ℕ → ℕ → ℕ) (h₁ : f_star 20 5 = 15) (h₂ : f_ast 15 5 = 75) :
  (f_star 8 4) / (f_ast 10 2) = (1:ℚ) / 5 := by
  sorry

end math_problem_l195_195418


namespace jack_evening_emails_l195_195083

theorem jack_evening_emails
  (emails_afternoon : ℕ := 3)
  (emails_morning : ℕ := 6)
  (emails_total : ℕ := 10) :
  emails_total - emails_afternoon - emails_morning = 1 :=
by
  sorry

end jack_evening_emails_l195_195083


namespace girls_in_club_l195_195689

/-
A soccer club has 30 members. For a recent team meeting, only 18 members could attend:
one-third of the girls attended but all of the boys attended. Prove that the number of 
girls in the soccer club is 18.
-/

variables (B G : ℕ)

-- Conditions
def total_members (B G : ℕ) := B + G = 30
def meeting_attendance (B G : ℕ) := (1/3 : ℚ) * G + B = 18

theorem girls_in_club (B G : ℕ) (h1 : total_members B G) (h2 : meeting_attendance B G) : G = 18 :=
  sorry

end girls_in_club_l195_195689


namespace average_speed_l195_195436

def s (t : ℝ) : ℝ := 3 + t^2

theorem average_speed {t1 t2 : ℝ} (h1 : t1 = 2) (h2: t2 = 2.1) :
  (s t2 - s t1) / (t2 - t1) = 4.1 :=
by
  sorry

end average_speed_l195_195436


namespace letter_addition_problem_l195_195412

theorem letter_addition_problem (S I X : ℕ) (E L V N : ℕ) 
  (hS : S = 8) 
  (hX_odd : X % 2 = 1)
  (h_diff_digits : ∀ (a b c d e f : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ a)
  (h_sum : 2 * S * 100 + 2 * I * 10 + 2 * X = E * 10000 + L * 1000 + E * 100 + V * 10 + E + N) :
  I = 3 :=
by
  sorry

end letter_addition_problem_l195_195412


namespace profit_percentage_l195_195325

theorem profit_percentage (selling_price profit : ℝ) (h1 : selling_price = 900) (h2 : profit = 300) : 
  (profit / (selling_price - profit)) * 100 = 50 :=
by
  sorry

end profit_percentage_l195_195325


namespace minimum_length_intersection_l195_195290

def length (a b : ℝ) : ℝ := b - a

def M (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 2/3 }
def N (n : ℝ) : Set ℝ := { x | n - 1/2 ≤ x ∧ x ≤ n }

def IntervalSet : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem minimum_length_intersection (m n : ℝ) (hM : M m ⊆ IntervalSet) (hN : N n ⊆ IntervalSet) :
  length (max m (n - 1/2)) (min (m + 2/3) n) = 1/6 :=
by
  sorry

end minimum_length_intersection_l195_195290


namespace workers_complete_time_l195_195457

theorem workers_complete_time 
  (time_A time_B time_C : ℕ) 
  (hA : time_A = 10)
  (hB : time_B = 12) 
  (hC : time_C = 15) : 
  let rate_A := (1: ℚ) / time_A
  let rate_B := (1: ℚ) / time_B
  let rate_C := (1: ℚ) / time_C
  let total_rate := rate_A + rate_B + rate_C
  1 / total_rate = 4 := 
by
  sorry

end workers_complete_time_l195_195457


namespace inequality_proof_l195_195271

variable (f : ℕ → ℕ → ℕ)

theorem inequality_proof :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
by sorry

end inequality_proof_l195_195271


namespace general_integral_of_ODE_l195_195434

noncomputable def general_solution (x y : ℝ) (m C : ℝ) : Prop :=
  (x^2 * y - x - m) / (x^2 * y - x + m) = C * Real.exp (2 * m / x)

theorem general_integral_of_ODE (m : ℝ) (y : ℝ → ℝ) (C : ℝ) (x : ℝ) (hx : x ≠ 0) :
  (∀ (y' : ℝ → ℝ) (x : ℝ), deriv y x = m^2 / x^4 - (y x)^2) ∧ 
  (y 1 = 1 / x + m / x^2) ∧ 
  (y 2 = 1 / x - m / x^2) →
  general_solution x (y x) m C :=
by 
  sorry

end general_integral_of_ODE_l195_195434


namespace value_a6_l195_195094

noncomputable def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 2, a n - a (n - 1) = n - 1

theorem value_a6 : ∃ a : ℕ → ℕ, seq a ∧ a 6 = 16 := by
  sorry

end value_a6_l195_195094


namespace increase_80_by_150_percent_l195_195656

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l195_195656


namespace greatest_integer_prime_l195_195841

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m < n → n % m ≠ 0

theorem greatest_integer_prime (x : ℤ) :
  is_prime (|8 * x ^ 2 - 56 * x + 21|) → ∀ y : ℤ, (is_prime (|8 * y ^ 2 - 56 * y + 21|) → y ≤ x) :=
by
  sorry

end greatest_integer_prime_l195_195841


namespace black_spools_l195_195795

-- Define the given conditions
def spools_per_beret : ℕ := 3
def red_spools : ℕ := 12
def blue_spools : ℕ := 6
def berets_made : ℕ := 11

-- Define the statement to be proved using the defined conditions
theorem black_spools (spools_per_beret red_spools blue_spools berets_made : ℕ) : (spools_per_beret * berets_made) - (red_spools + blue_spools) = 15 :=
by sorry

end black_spools_l195_195795


namespace megatek_manufacturing_percentage_l195_195275

-- Define the given conditions
def sector_deg : ℝ := 18
def full_circle_deg : ℝ := 360

-- Define the problem as a theorem statement in Lean
theorem megatek_manufacturing_percentage : 
  (sector_deg / full_circle_deg) * 100 = 5 := 
sorry

end megatek_manufacturing_percentage_l195_195275


namespace find_n_l195_195283

def digit_sum (n : ℕ) : ℕ :=
-- This function needs a proper definition for the digit sum, we leave it as sorry for this example.
sorry

def num_sevens (n : ℕ) : ℕ :=
7 * (10^n - 1) / 9

def product (n : ℕ) : ℕ :=
8 * num_sevens n

theorem find_n (n : ℕ) : digit_sum (product n) = 800 ↔ n = 788 :=
sorry

end find_n_l195_195283


namespace part1_part2_mean_part2_variance_l195_195550

open ProbabilityTheory

section Problem

/-- Define the population of doctors. --/
def doctors : Finset ℕ := {0, 1, 2, 3, 4, 5}

/-- The set of surgeons, internists, and ophthalmologists. --/
def surgeons : Finset ℕ := {0, 1}
def internists : Finset ℕ := {2, 3}
def ophthalmologists : Finset ℕ := {4, 5}

/-- The probability of selecting a subset of 3 doctors. --/
def select_doctors (s : Finset ℕ) : Prop :=
  s.card = 3 ∧ s ⊆ doctors

/-- Probability mass function for selecting 3 doctors. --/
noncomputable def pmf_select : PMF (Finset ℕ) :=
  PMF.ofMultisetMultiset (doctors.powerset.filter select_doctors).val
    (by { rw ← Multiset.card, exact (Multiset.card_pos_iff_exists_mem.mpr ⟨_, Multiset.mem_powerset.mpr (by refl)⟩), })

/-- Probability event where the number of selected surgeons is greater than the number of selected internists. --/
def event_more_surgeons (s : Finset ℕ) : Prop :=
  (s ∩ surgeons).card > (s ∩ internists).card

/-- The random variable representing the number of surgeons selected. --/
def num_surgeons (s : Finset ℕ) : ℕ :=
  (s ∩ surgeons).card

theorem part1 :
  PMF.prob (pmf_select {w | event_more_surgeons w}) = 3 / 10 :=
sorry

theorem part2_mean :
  PMF.expectedValue pmf_select num_surgeons = 1 :=
sorry

theorem part2_variance :
  PMF.variance num_surgeons pmf_select = 2 / 5 :=
sorry

end Problem

end part1_part2_mean_part2_variance_l195_195550


namespace sin_alpha_second_quadrant_l195_195053

/-- Given angle α in the second quadrant such that tan(π - α) = 3/4, we need to prove that sin α = 3/5. -/
theorem sin_alpha_second_quadrant (α : ℝ) (hα1 : π / 2 < α ∧ α < π) (hα2 : Real.tan (π - α) = 3 / 4) : Real.sin α = 3 / 5 := by
  sorry

end sin_alpha_second_quadrant_l195_195053


namespace arcsin_neg_one_l195_195179

theorem arcsin_neg_one : Real.arcsin (-1) = -Real.pi / 2 := by
  sorry

end arcsin_neg_one_l195_195179


namespace power_function_is_x_cubed_l195_195401

/-- Define the power function and its property -/
def power_function (a : ℕ) (x : ℝ) : ℝ := x ^ a

/-- The given condition that the function passes through the point (3, 27) -/
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f 3 = 27

/-- Prove that the power function is x^3 -/
theorem power_function_is_x_cubed (f : ℝ → ℝ)
  (h : passes_through_point f) : 
  f = fun x => x ^ 3 := 
by
  sorry -- proof to be filled in

end power_function_is_x_cubed_l195_195401


namespace increase_80_by_150_percent_l195_195617

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l195_195617


namespace line_of_intersection_in_standard_form_l195_195462

noncomputable def plane1 (x y z : ℝ) := 3 * x + 4 * y - 2 * z = 5
noncomputable def plane2 (x y z : ℝ) := 2 * x + 3 * y - z = 3

theorem line_of_intersection_in_standard_form :
  (∃ x y z : ℝ, plane1 x y z ∧ plane2 x y z ∧ (∀ t : ℝ, (x, y, z) = 
  (3 + 2 * t, -1 - t, t))) :=
by {
  sorry
}

end line_of_intersection_in_standard_form_l195_195462


namespace prob_white_first_yellow_second_l195_195445

-- Defining the number of yellow and white balls
def yellow_balls : ℕ := 6
def white_balls : ℕ := 4

-- Defining the total number of balls
def total_balls : ℕ := yellow_balls + white_balls

-- Define the events A and B
def event_A : Prop := true -- event A: drawing a white ball first
def event_B : Prop := true -- event B: drawing a yellow ball second

-- Conditional probability P(B|A)
def prob_B_given_A : ℚ := 6 / (total_balls - 1)

-- Main theorem stating the proof problem
theorem prob_white_first_yellow_second : prob_B_given_A = 2 / 3 :=
by
  sorry

end prob_white_first_yellow_second_l195_195445


namespace sphere_surface_area_l195_195268

-- Define the conditions
def points_on_sphere (A B C : Type) := 
  ∃ (AB BC AC : Real), AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define the distance condition
def distance_condition (R : Real) := 
  ∃ (d : Real), d = R / 2

-- Define the main theorem
theorem sphere_surface_area 
  (A B C : Type) 
  (h_points : points_on_sphere A B C) 
  (h_distance : ∃ R : Real, distance_condition R) : 
  4 * Real.pi * (10 / 3 * Real.sqrt 3) ^ 2 = 400 / 3 * Real.pi := 
by 
  sorry

end sphere_surface_area_l195_195268


namespace rational_numbers_satisfying_conditions_l195_195041

theorem rational_numbers_satisfying_conditions :
  (∃ n : ℕ, n = 166 ∧ ∀ (m : ℚ),
  abs m < 500 → (∃ x : ℤ, 3 * x^2 + m * x + 25 = 0) ↔ n = 166)
:=
sorry

end rational_numbers_satisfying_conditions_l195_195041


namespace center_of_circle_l195_195190

theorem center_of_circle : 
  ∀ x y : ℝ, 4 * x^2 + 8 * x + 4 * y^2 - 12 * y + 29 = 0 → (x = -1 ∧ y = 3 / 2) :=
by
  sorry

end center_of_circle_l195_195190


namespace inequality_proof_l195_195057

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) : 
  (1 / (5 * a^2 - 4 * a + 11)) + (1 / (5 * b^2 - 4 * b + 11)) + (1 / (5 * c^2 - 4 * c + 11)) ≤ 1 / 4 := 
by
  -- proof steps will be here
  sorry

end inequality_proof_l195_195057


namespace gasoline_needed_l195_195856

theorem gasoline_needed (D : ℕ) 
    (fuel_efficiency : ℕ) 
    (fuel_efficiency_proof : fuel_efficiency = 20)
    (gallons_for_130km : ℕ) 
    (gallons_for_130km_proof : gallons_for_130km = 130 / 20) :
    (D : ℕ) / fuel_efficiency = (D : ℕ) / 20 :=
by
  -- The proof is omitted as per the instruction
  sorry

end gasoline_needed_l195_195856


namespace vectors_coplanar_l195_195995

/-- Vectors defined as 3-dimensional Euclidean space vectors. --/
def vector3 := (ℝ × ℝ × ℝ)

/-- Definitions for vectors a, b, c as given in the problem conditions. --/
def a : vector3 := (3, 1, -1)
def b : vector3 := (1, 0, -1)
def c : vector3 := (8, 3, -2)

/-- The scalar triple product of vectors a, b, c is the determinant of the matrix formed. --/
noncomputable def scalarTripleProduct (u v w : vector3) : ℝ :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  u1 * (v2 * w3 - v3 * w2) - u2 * (v1 * w3 - v3 * w1) + u3 * (v1 * w2 - v2 * w1)

/-- Statement to prove that vectors a, b, c are coplanar (i.e., their scalar triple product is zero). --/
theorem vectors_coplanar : scalarTripleProduct a b c = 0 :=
  by sorry

end vectors_coplanar_l195_195995


namespace not_enrolled_eq_80_l195_195251

variable (total_students : ℕ)
variable (french_students : ℕ)
variable (german_students : ℕ)
variable (spanish_students : ℕ)
variable (french_and_german : ℕ)
variable (german_and_spanish : ℕ)
variable (spanish_and_french : ℕ)
variable (all_three : ℕ)

noncomputable def students_not_enrolled_in_any_language 
  (total_students french_students german_students spanish_students french_and_german german_and_spanish spanish_and_french all_three : ℕ) : ℕ :=
  total_students - (french_students + german_students + spanish_students - french_and_german - german_and_spanish - spanish_and_french + all_three)

theorem not_enrolled_eq_80 : 
  students_not_enrolled_in_any_language 180 60 50 35 20 15 10 5 = 80 :=
  by
    unfold students_not_enrolled_in_any_language
    simp
    sorry

end not_enrolled_eq_80_l195_195251


namespace PedoeInequalityHolds_l195_195419

noncomputable def PedoeInequality 
  (a b c a1 b1 c1 : ℝ) (Δ Δ1 : ℝ) :
  Prop :=
  a^2 * (b1^2 + c1^2 - a1^2) + 
  b^2 * (c1^2 + a1^2 - b1^2) + 
  c^2 * (a1^2 + b1^2 - c1^2) >= 16 * Δ * Δ1 

axiom areas_triangle 
  (a b c : ℝ) : ℝ 

axiom areas_triangle1 
  (a1 b1 c1 : ℝ) : ℝ 

theorem PedoeInequalityHolds 
  (a b c a1 b1 c1 : ℝ) 
  (Δ := areas_triangle a b c) 
  (Δ1 := areas_triangle1 a1 b1 c1) :
  PedoeInequality a b c a1 b1 c1 Δ Δ1 :=
sorry

end PedoeInequalityHolds_l195_195419


namespace second_particle_catches_first_l195_195957

open Real

-- Define the distance functions for both particles
def distance_first (t : ℝ) : ℝ := 34 + 5 * t
def distance_second (t : ℝ) : ℝ := 0.25 * t^2 + 2.75 * t

-- The proof statement
theorem second_particle_catches_first : ∃ t : ℝ, distance_second t = distance_first t ∧ t = 17 :=
by
  have : distance_first 17 = 34 + 5 * 17 := by sorry
  have : distance_second 17 = 0.25 * 17^2 + 2.75 * 17 := by sorry
  sorry

end second_particle_catches_first_l195_195957


namespace tens_digit_of_large_power_l195_195546

theorem tens_digit_of_large_power : ∃ a : ℕ, a = 2 ∧ ∀ n ≥ 2, (5 ^ n) % 100 = 25 :=
by
  sorry

end tens_digit_of_large_power_l195_195546


namespace number_of_dogs_on_boat_l195_195335

theorem number_of_dogs_on_boat 
  (initial_sheep : ℕ) (initial_cows : ℕ) (initial_dogs : ℕ)
  (drowned_sheep : ℕ) (drowned_cows : ℕ)
  (made_it_to_shore : ℕ)
  (H1 : initial_sheep = 20)
  (H2 : initial_cows = 10)
  (H3 : drowned_sheep = 3)
  (H4 : drowned_cows = 2 * drowned_sheep)
  (H5 : made_it_to_shore = 35)
  : initial_dogs = 14 := 
sorry

end number_of_dogs_on_boat_l195_195335


namespace num_geography_books_l195_195286

theorem num_geography_books
  (total_books : ℕ)
  (history_books : ℕ)
  (math_books : ℕ)
  (h1 : total_books = 100)
  (h2 : history_books = 32)
  (h3 : math_books = 43) :
  total_books - history_books - math_books = 25 :=
by
  sorry

end num_geography_books_l195_195286


namespace clea_ride_escalator_time_l195_195932

def clea_time_not_walking (x k y : ℝ) : Prop :=
  60 * x = y ∧ 24 * (x + k) = y ∧ 1.5 * x = k ∧ 40 = y / k

theorem clea_ride_escalator_time :
  ∀ (x y k : ℝ), 60 * x = y → 24 * (x + k) = y → (1.5 * x = k) → 40 = y / k :=
by
  intros x y k H1 H2 H3
  sorry

end clea_ride_escalator_time_l195_195932


namespace remainder_of_S_div_1000_l195_195807

theorem remainder_of_S_div_1000 :
  let S := (Finset.filter (λ n : ℕ, ∃ m : ℕ, n^2 + 12 * n - 3007 = m^2)
    (Finset.range 2000)).sum in
  (S % 1000) = 566 := by
  sorry

end remainder_of_S_div_1000_l195_195807


namespace necessary_but_not_sufficient_l195_195542

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*(a+1)*x + 3

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x → f a x - f a 1 ≥ 0) ↔ (a ≤ -2) :=
sorry

end necessary_but_not_sufficient_l195_195542


namespace min_value_of_expression_l195_195722

theorem min_value_of_expression (m n : ℕ) (hm : 0 < m) (hn : 0 < n)
  (hpar : ∀ x y : ℝ, 2 * x + (n - 1) * y - 2 = 0 → ∃ c : ℝ, mx + ny + c = 0) :
  2 * m + n = 9 :=
by
  sorry

end min_value_of_expression_l195_195722


namespace rectangle_perimeter_gt_16_l195_195196

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_area_gt_perim : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
by
  sorry

end rectangle_perimeter_gt_16_l195_195196


namespace john_has_500_dollars_l195_195086

-- Define the initial amount and the condition
def initial_amount : ℝ := 1600
def condition (spent : ℝ) : Prop := (1600 - spent) = (spent - 600)

-- The final amount of money John still has
def final_amount (spent : ℝ) : ℝ := initial_amount - spent

-- The main theorem statement
theorem john_has_500_dollars : ∃ (spent : ℝ), condition spent ∧ final_amount spent = 500 :=
by
  sorry

end john_has_500_dollars_l195_195086


namespace inequality_solution_set_range_of_values_l195_195060

def f (x : ℝ) : ℝ := abs (2 * x - 1)

theorem inequality_solution_set :
  {x : ℝ | -3 / 2 < x ∧ x < 5 / 2} = {x : ℝ | ∃ y : ℝ, y = f x ∧ 0 < y ∧ y < 4} :=
by {
  sorry
}

def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem range_of_values (a : ℝ) (m n : ℝ) (h1 : m + n = a) (h2 : a = 2) (h3 : m > 0) (h4 : n > 0) :
  ∃ t : ℝ, t ∈ Set.Ici (3 / 2 + Real.sqrt 2) ∧ ∀ (m n : ℝ), m > 0 → n > 0 → m + n = 2 → t = 2 / m + 1 / n :=
by {
  sorry
}

end inequality_solution_set_range_of_values_l195_195060


namespace values_of_a_and_b_to_satisfy_condition_l195_195943

noncomputable def f (x a b : ℝ) : ℝ := x^2 + a*x + b*cos x

theorem values_of_a_and_b_to_satisfy_condition :
  {a b : ℝ | ∃ S : set ℝ, S = {x | f x a b = 0} ∧ S.nonempty ∧ S = {x | f (f x a b) a b = 0}} 
  = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 < 4 ∧ p.2 = 0} :=
by
  sorry

end values_of_a_and_b_to_satisfy_condition_l195_195943


namespace carl_candy_bars_l195_195706

/-- 
Carl earns $0.75 every week for taking out his neighbor's trash. 
Carl buys a candy bar every time he earns $0.50. 
After four weeks, Carl will be able to buy 6 candy bars.
-/
theorem carl_candy_bars :
  (0.75 * 4) / 0.50 = 6 := 
  by
    sorry

end carl_candy_bars_l195_195706


namespace no_two_obtuse_angles_in_triangle_l195_195137

theorem no_two_obtuse_angles_in_triangle (A B C : ℝ) 
  (h1 : 0 < A) (h2 : A < 180) 
  (h3 : 0 < B) (h4 : B < 180) 
  (h5 : 0 < C) (h6 : C < 180)
  (h7 : A + B + C = 180) 
  (h8 : A > 90) (h9 : B > 90) : false :=
by
  sorry

end no_two_obtuse_angles_in_triangle_l195_195137


namespace probability_MAME_on_top_l195_195985

theorem probability_MAME_on_top : 
  let num_sections := 8
  let favorable_outcome := 1
  (favorable_outcome : ℝ) / (num_sections : ℝ) = 1 / 8 :=
by 
  sorry

end probability_MAME_on_top_l195_195985


namespace Dan_tshirts_total_l195_195360

theorem Dan_tshirts_total :
  (let rate1 := 1 / 12
   let rate2 := 1 / 6
   let hour := 60
   let tshirts_first_hour := hour * rate1
   let tshirts_second_hour := hour * rate2
   let total_tshirts := tshirts_first_hour + tshirts_second_hour
   total_tshirts) = 15 := by
  sorry

end Dan_tshirts_total_l195_195360


namespace MaryHasBlueMarbles_l195_195885

-- Define the number of blue marbles Dan has
def DanMarbles : Nat := 5

-- Define the relationship of Mary's marbles to Dan's marbles
def MaryMarbles : Nat := 2 * DanMarbles

-- State the theorem that we need to prove
theorem MaryHasBlueMarbles : MaryMarbles = 10 :=
by
  sorry

end MaryHasBlueMarbles_l195_195885


namespace quadratic_min_value_l195_195791

theorem quadratic_min_value (f : ℕ → ℚ) (n : ℕ)
  (h₁ : f n = 6)
  (h₂ : f (n + 1) = 5)
  (h₃ : f (n + 2) = 5) :
  ∃ c : ℚ, c = 39 / 8 ∧ ∀ x : ℕ, f x ≥ c :=
by
  sorry

end quadratic_min_value_l195_195791


namespace daps_equivalent_to_dips_l195_195235

theorem daps_equivalent_to_dips (daps dops dips : ℕ) 
  (h1 : 4 * daps = 3 * dops) 
  (h2 : 2 * dops = 7 * dips) :
  35 * dips = 20 * daps :=
by
  sorry

end daps_equivalent_to_dips_l195_195235


namespace last_digit_base4_of_389_l195_195880

theorem last_digit_base4_of_389 : (389 % 4 = 1) :=
by sorry

end last_digit_base4_of_389_l195_195880


namespace sin_double_angle_solution_l195_195788

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l195_195788


namespace garden_area_remaining_l195_195146

variable (d : ℕ) (w : ℕ) (t : ℕ)

theorem garden_area_remaining (r : Real) (A_circle : Real) 
                              (A_path : Real) (A_remaining : Real) :
  r = 10 →
  A_circle = 100 * Real.pi →
  A_path = 66.66 * Real.pi - 50 * Real.sqrt 3 →
  A_remaining = 33.34 * Real.pi + 50 * Real.sqrt 3 :=
by
  -- Given the radius of the garden
  let r := (d : Real) / 2
  -- Calculate the total area of the garden
  let A_circle := Real.pi * r^2
  -- Area covered by the path computed using circular segments
  let A_path := 66.66 * Real.pi - 50 * Real.sqrt 3
  -- Remaining garden area
  let A_remaining := A_circle - A_path
  -- Statement to prove correct
  sorry 

end garden_area_remaining_l195_195146


namespace median_is_2005_5_l195_195133

noncomputable def median_of_special_list : Rat :=
let list := (List.range (2050 + 1)).append (List.range(2050 + 1)).map (λ x => x * x) in
(list.nth_le 2049 (by simp)).enslave + (list.nth_le 2050 (by simp)) / 2

theorem median_is_2005_5 :
  median_of_special_list = 2005.5 := 
by
  sorry

end median_is_2005_5_l195_195133


namespace pascal_triangle_47_l195_195219

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l195_195219


namespace outdoor_chairs_count_l195_195145

theorem outdoor_chairs_count (indoor_tables outdoor_tables : ℕ) (chairs_per_indoor_table : ℕ) 
  (total_chairs : ℕ) (h1: indoor_tables = 9) (h2: outdoor_tables = 11) 
  (h3: chairs_per_indoor_table = 10) (h4: total_chairs = 123) : 
  (total_chairs - indoor_tables * chairs_per_indoor_table) / outdoor_tables = 3 :=
by 
  sorry

end outdoor_chairs_count_l195_195145


namespace total_time_for_journey_l195_195014

theorem total_time_for_journey (x : ℝ) : 
  let time_first_part := x / 50
  let time_second_part := 3 * x / 80
  time_first_part + time_second_part = 23 * x / 400 :=
by 
  sorry

end total_time_for_journey_l195_195014


namespace find_a2_l195_195092

def S (n : Nat) (a1 d : Int) : Int :=
  n * a1 + (n * (n - 1) * d) / 2

theorem find_a2 (a1 : Int) (d : Int) :
  a1 = -2010 ∧
  (S 2010 a1 d) / 2010 - (S 2008 a1 d) / 2008 = 2 →
  a1 + d = -2008 :=
by
  sorry

end find_a2_l195_195092


namespace people_present_l195_195449

-- Define the number of parents, pupils, and teachers as constants
def p := 73
def s := 724
def t := 744

-- The theorem to prove the total number of people present
theorem people_present : p + s + t = 1541 := 
by
  -- Proof is inserted here
  sorry

end people_present_l195_195449


namespace ratio_of_doctors_to_lawyers_l195_195106

/--
Given the average age of a group consisting of doctors and lawyers is 47,
the average age of doctors is 45,
and the average age of lawyers is 55,
prove that the ratio of the number of doctors to the number of lawyers is 4:1.
-/
theorem ratio_of_doctors_to_lawyers
  (d l : ℕ) -- numbers of doctors and lawyers
  (avg_group_age : ℝ := 47)
  (avg_doctors_age : ℝ := 45)
  (avg_lawyers_age : ℝ := 55)
  (h : (45 * d + 55 * l) / (d + l) = 47) :
  d = 4 * l :=
by
  sorry

end ratio_of_doctors_to_lawyers_l195_195106


namespace polynomial_degree_l195_195088

variable {P : Polynomial ℝ}

theorem polynomial_degree (h1 : ∀ x : ℝ, (x - 4) * P.eval (2 * x) = 4 * (x - 1) * P.eval x) (h2 : P.eval 0 ≠ 0) : P.degree = 2 := 
sorry

end polynomial_degree_l195_195088


namespace f_odd_function_l195_195910

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive (a b : ℝ) : f (a + b) = f a + f b

theorem f_odd_function : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  sorry

end f_odd_function_l195_195910


namespace jessica_speed_last_40_l195_195085

theorem jessica_speed_last_40 
  (total_distance : ℕ)
  (total_time_min : ℕ)
  (first_segment_avg_speed : ℕ)
  (second_segment_avg_speed : ℕ)
  (last_segment_avg_speed : ℕ) :
  total_distance = 120 →
  total_time_min = 120 →
  first_segment_avg_speed = 50 →
  second_segment_avg_speed = 60 →
  last_segment_avg_speed = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end jessica_speed_last_40_l195_195085


namespace school_pays_570_l195_195573

theorem school_pays_570
  (price_per_model : ℕ := 100)
  (models_kindergarten : ℕ := 2)
  (models_elementary_multiple : ℕ := 2)
  (total_models : ℕ := models_kindergarten + models_elementary_multiple * models_kindergarten)
  (price_reduction : ℕ := if total_models > 5 then (price_per_model * 5 / 100) else 0)
  (reduced_price_per_model : ℕ := price_per_model - price_reduction) :
  2 * models_kindergarten * reduced_price_per_model = 570 :=
by
  -- Proof omitted
  sorry

end school_pays_570_l195_195573


namespace largest_divisor_of_expression_l195_195237

theorem largest_divisor_of_expression (x : ℤ) (h : x % 2 = 1) : 
  324 ∣ (12 * x + 3) * (12 * x + 9) * (6 * x + 6) :=
sorry

end largest_divisor_of_expression_l195_195237


namespace increase_by_150_percent_l195_195664

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l195_195664


namespace part1_part2_l195_195064

theorem part1 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) : k > 3 / 4 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) (hx1x2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 → x1 * x2 = 5) : k = 2 :=
sorry

end part1_part2_l195_195064


namespace no_two_obtuse_angles_in_triangle_l195_195136

theorem no_two_obtuse_angles_in_triangle (A B C : ℝ) 
  (h1 : 0 < A) (h2 : A < 180) 
  (h3 : 0 < B) (h4 : B < 180) 
  (h5 : 0 < C) (h6 : C < 180)
  (h7 : A + B + C = 180) 
  (h8 : A > 90) (h9 : B > 90) : false :=
by
  sorry

end no_two_obtuse_angles_in_triangle_l195_195136


namespace value_of_b_l195_195240

theorem value_of_b (b : ℝ) : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^3 - b*x1^2 + 1/2 = 0) ∧ (x2^3 - b*x2^2 + 1/2 = 0)) → b = 3/2 :=
by
  sorry

end value_of_b_l195_195240


namespace fraction_identity_proof_l195_195082

theorem fraction_identity_proof (a b : ℝ) (h : 2 / a - 1 / b = 1 / (a + 2 * b)) :
  4 / (a ^ 2) - 1 / (b ^ 2) = 1 / (a * b) :=
by
  sorry

end fraction_identity_proof_l195_195082


namespace system_of_equations_solution_system_of_inequalities_solution_l195_195677

-- Problem (1): Solve the system of equations
theorem system_of_equations_solution :
  ∃ (x y : ℝ), (3 * (y - 2) = x - 1) ∧ (2 * (x - 1) = 5 * y - 8) ∧ (x = 7) ∧ (y = 4) :=
by
  sorry

-- Problem (2): Solve the system of linear inequalities
theorem system_of_inequalities_solution :
  ∃ (x : ℝ), (3 * x ≤ 2 * x + 3) ∧ ((x + 1) / 6 - 1 < (2 * (x + 1)) / 3) ∧ (-3 < x) ∧ (x ≤ 3) :=
by
  sorry

end system_of_equations_solution_system_of_inequalities_solution_l195_195677


namespace variable_value_l195_195922

theorem variable_value (w x v : ℝ) (h1 : 5 / w + 5 / x = 5 / v) (h2 : w * x = v) (h3 : (w + x) / 2 = 0.5) : v = 0.25 :=
by
  sorry

end variable_value_l195_195922


namespace arrange_numbers_in_ascending_order_l195_195170

noncomputable def S := 222 ^ 2
noncomputable def T := 22 ^ 22
noncomputable def U := 2 ^ 222
noncomputable def V := 22 ^ (2 ^ 2)
noncomputable def W := 2 ^ (22 ^ 2)
noncomputable def X := 2 ^ (2 ^ 22)
noncomputable def Y := 2 ^ (2 ^ (2 ^ 2))

theorem arrange_numbers_in_ascending_order :
  S < Y ∧ Y < V ∧ V < T ∧ T < U ∧ U < W ∧ W < X :=
sorry

end arrange_numbers_in_ascending_order_l195_195170


namespace construct_inaccessible_angle_bisector_l195_195370

-- Definitions for problem context
structure Point :=
  (x y : ℝ)

structure Line :=
  (p1 p2 : Point)

structure Angle := 
  (vertex : Point)
  (ray1 ray2 : Line)

-- Predicate to determine if a line bisects an angle
def IsAngleBisector (L : Line) (A : Angle) : Prop := sorry

-- The inaccessible vertex angle we are considering
-- Let's assume the vertex is defined but we cannot access it physically in constructions
noncomputable def inaccessible_angle : Angle := sorry

-- Statement to prove: Construct a line that bisects the inaccessible angle
theorem construct_inaccessible_angle_bisector :
  ∃ L : Line, IsAngleBisector L inaccessible_angle :=
sorry

end construct_inaccessible_angle_bisector_l195_195370


namespace third_player_matches_l195_195591

theorem third_player_matches (first_player second_player third_player : ℕ) (h1 : first_player = 10) (h2 : second_player = 21) :
  third_player = 11 :=
by
  sorry

end third_player_matches_l195_195591


namespace solution_in_range_for_fraction_l195_195500

theorem solution_in_range_for_fraction (a : ℝ) : 
  (∃ x : ℝ, (2 * x + a) / (x + 1) = 1 ∧ x < 0) ↔ (a > 1 ∧ a ≠ 2) :=
by
  sorry

end solution_in_range_for_fraction_l195_195500


namespace max_value_of_n_l195_195105

theorem max_value_of_n : 
  ∃ n : ℕ, 
    (∀ m : ℕ, m ≤ n → (2 / 3)^(m - 1) * (1 / 3) ≥ 1 / 60) 
      ∧ 
    (∀ k : ℕ, k > n → (2 / 3)^(k - 1) * (1 / 3) < 1 / 60) 
      ∧ 
    n = 8 :=
by
  sorry

end max_value_of_n_l195_195105


namespace log_bounds_l195_195923

-- Definitions and assumptions
def tenCubed : Nat := 1000
def tenFourth : Nat := 10000
def twoNine : Nat := 512
def twoFourteen : Nat := 16384

-- Statement that encapsulates the proof problem
theorem log_bounds (h1 : 10^3 = tenCubed) 
                   (h2 : 10^4 = tenFourth) 
                   (h3 : 2^9 = twoNine) 
                   (h4 : 2^14 = twoFourteen) : 
  (2 / 7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (1 / 3 : ℝ) :=
sorry

end log_bounds_l195_195923


namespace arcsin_neg_one_is_neg_half_pi_l195_195178

noncomputable def arcsine_equality : Prop := 
  real.arcsin (-1) = - (real.pi / 2)

theorem arcsin_neg_one_is_neg_half_pi : arcsine_equality :=
by
  sorry

end arcsin_neg_one_is_neg_half_pi_l195_195178


namespace sin_double_angle_solution_l195_195787

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l195_195787


namespace bucket_weight_full_l195_195002

variable (c d : ℝ)

theorem bucket_weight_full (h1 : ∃ x y, x + (1 / 4) * y = c)
                           (h2 : ∃ x y, x + (3 / 4) * y = d) :
  ∃ x y, x + y = (3 * d - c) / 2 :=
by
  sorry

end bucket_weight_full_l195_195002


namespace roses_in_vase_l195_195448

theorem roses_in_vase (initial_roses added_roses : ℕ) (h₀ : initial_roses = 10) (h₁ : added_roses = 8) : initial_roses + added_roses = 18 :=
by
  sorry

end roses_in_vase_l195_195448


namespace increase_80_by_150_percent_l195_195615

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l195_195615


namespace mary_blue_marbles_l195_195883

theorem mary_blue_marbles (dan_blue_marbles mary_blue_marbles : ℕ)
  (h1 : dan_blue_marbles = 5)
  (h2 : mary_blue_marbles = 2 * dan_blue_marbles) : mary_blue_marbles = 10 := 
by
  sorry

end mary_blue_marbles_l195_195883


namespace polynomial_form_l195_195033

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  ((n.to_digits 10).sum)

theorem polynomial_form (P : ℤ[X]) (hP : ∀ (x y : ℕ), sum_of_digits x = sum_of_digits y → 
  sum_of_digits (|P.eval x|) = sum_of_digits (|P.eval y|)) :
  ∃ k c : ℕ, (P = polynomial.C (-1) * (polynomial.X * 10^k + polynomial.C c)
  ∨ P = polynomial.X * 10^k + polynomial.C c) ∧ 0 ≤ c ∧ c < 10^k :=
  sorry

end polynomial_form_l195_195033


namespace express_y_in_terms_of_x_l195_195551

theorem express_y_in_terms_of_x (x y : ℝ) (h : 4 * x - y = 7) : y = 4 * x - 7 :=
sorry

end express_y_in_terms_of_x_l195_195551


namespace abc_equal_l195_195044

theorem abc_equal (a b c : ℝ) (h : a^2 + b^2 + c^2 = a * b + b * c + c * a) : a = b ∧ b = c :=
by
  sorry

end abc_equal_l195_195044


namespace negation_of_existence_l195_195437

theorem negation_of_existence : 
  (¬ ∃ x_0 : ℝ, (x_0 + 1 < 0) ∨ (x_0^2 - x_0 > 0)) ↔ ∀ x : ℝ, (x + 1 ≥ 0) ∧ (x^2 - x ≤ 0) := 
by
  sorry

end negation_of_existence_l195_195437


namespace balls_in_indistinguishable_boxes_l195_195531

theorem balls_in_indistinguishable_boxes : 
  let balls := 6 in
  let boxes := 3 in
  -- Expression to count ways to distribute balls into indistinguishable boxes
  ((balls.choose 0 * ((balls-0).choose 6)) + 
   (balls.choose 1 * ((balls-1).choose 5)) + 
   (balls.choose 2 * ((balls-2).choose 4)) + 
   (balls.choose 2 * ((balls-2).choose 3) / 2!) + 
   (balls.choose 3 * ((balls-3).choose 3)) + 
   (balls.choose 3 * ((balls-3).choose 2) * ((balls-5).choose 1)) + 
   (balls.choose 2 * ((balls-2).choose 2) / 3!))
  = 92 := 
sorry

end balls_in_indistinguishable_boxes_l195_195531


namespace increase_by_150_percent_l195_195666

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l195_195666


namespace value_of_f_m_plus_one_is_negative_l195_195504

-- Definitions for function and condition
def f (x a : ℝ) := x^2 - x + a 

-- Problem statement: Given that 'f(-m) < 0', prove 'f(m+1) < 0'
theorem value_of_f_m_plus_one_is_negative (a m : ℝ) (h : f (-m) a < 0) : f (m + 1) a < 0 :=
by 
  sorry

end value_of_f_m_plus_one_is_negative_l195_195504


namespace sarah_average_speed_l195_195101

theorem sarah_average_speed :
  ∀ (total_distance race_time : ℕ) 
    (sadie_speed sadie_time ariana_speed ariana_time : ℕ)
    (distance_sarah speed_sarah time_sarah : ℚ),
  sadie_speed = 3 → 
  sadie_time = 2 → 
  ariana_speed = 6 → 
  ariana_time = 1 / 2 → 
  race_time = 9 / 2 → 
  total_distance = 17 →
  distance_sarah = total_distance - (sadie_speed * sadie_time + ariana_speed * ariana_time) →
  time_sarah = race_time - (sadie_time + ariana_time) →
  speed_sarah = distance_sarah / time_sarah →
  speed_sarah = 4 :=
by
  intros total_distance race_time sadie_speed sadie_time ariana_speed ariana_time distance_sarah speed_sarah time_sarah
  intros sadie_speed_eq sadie_time_eq ariana_speed_eq ariana_time_eq race_time_eq total_distance_eq distance_sarah_eq time_sarah_eq speed_sarah_eq
  sorry

end sarah_average_speed_l195_195101


namespace sum_of_coefficients_of_expansion_l195_195897

theorem sum_of_coefficients_of_expansion (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (2 * x - 1)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_1 + a_2 + a_3 + a_4 + a_5 = 2 :=
by
  intro h
  have h0 := h 0
  have h1 := h 1
  sorry

end sum_of_coefficients_of_expansion_l195_195897


namespace isosceles_triangle_perimeter_l195_195746

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3) (h2 : b = 7) : 
∃ (c : ℕ), 
  (c = 7 ∧ a = 3 ∧ b = 7 ∧ a + b + c = 17) ∨ 
  (c = 3 ∧ a = 7 ∧ b = 7 ∧ a + b + c = 17) :=
  sorry

end isosceles_triangle_perimeter_l195_195746


namespace total_camels_l195_195447

theorem total_camels (x y : ℕ) (humps_eq : x + 2 * y = 23) (legs_eq : 4 * (x + y) = 60) : x + y = 15 :=
by
  sorry

end total_camels_l195_195447


namespace StatementA_incorrect_l195_195195

def f (n : ℕ) : ℕ := (n.factorial)^2

def g (x : ℕ) : ℕ := f (x + 1) / f x

theorem StatementA_incorrect (x : ℕ) (h : x = 1) : g x ≠ 4 := sorry

end StatementA_incorrect_l195_195195


namespace sum_second_largest_smallest_l195_195287

theorem sum_second_largest_smallest (a b c : ℕ) (order_cond : a < b ∧ b < c) : a + b = 21 :=
by
  -- Following the correct answer based on the provided conditions:
  -- 10, 11, and 12 with their ordering, we have the smallest a and the second largest b.
  sorry

end sum_second_largest_smallest_l195_195287


namespace scientific_notation_of_203000_l195_195409

-- Define the number
def n : ℝ := 203000

-- Define the representation of the number in scientific notation
def scientific_notation (a b : ℝ) : Prop := n = a * 10^b ∧ 1 ≤ a ∧ a < 10

-- The theorem to state 
theorem scientific_notation_of_203000 : ∃ a b : ℝ, scientific_notation a b ∧ a = 2.03 ∧ b = 5 :=
by
  use 2.03
  use 5
  sorry

end scientific_notation_of_203000_l195_195409


namespace sin_double_angle_l195_195763

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l195_195763


namespace number_of_valid_grids_l195_195368

-- Define the concept of a grid and the necessary properties
structure Grid (n : ℕ) :=
  (cells: Fin (n * n) → ℕ)
  (unique: Function.Injective cells)
  (ordered_rows: ∀ i j : Fin n, i < j → cells ⟨i * n + j, sorry⟩ > cells ⟨i * n + j - 1, sorry⟩)
  (ordered_columns: ∀ i j : Fin n, i < j → cells ⟨j * n + i, sorry⟩ > cells ⟨(j - 1) * n + i, sorry⟩)

-- Define the 4x4 grid
def grid_4x4 := Grid 4

-- Statement of the problem: prove there are 2 valid grid_4x4 configurations
theorem number_of_valid_grids : ∃ g : grid_4x4, (∃ g1 g2 : grid_4x4, (g1 ≠ g2) ∧ (∀ g3 : grid_4x4, g3 = g1 ∨ g3 = g2)) :=
  sorry

end number_of_valid_grids_l195_195368


namespace complement_intersection_eq_l195_195721

variable (U P Q : Set ℕ)
variable (hU : U = {1, 2, 3})
variable (hP : P = {1, 2})
variable (hQ : Q = {2, 3})

theorem complement_intersection_eq : 
  (U \ (P ∩ Q)) = {1, 3} := by
  sorry

end complement_intersection_eq_l195_195721


namespace before_lunch_rush_customers_l195_195159

def original_customers_before_lunch := 29
def added_customers_during_lunch := 20
def customers_no_tip := 34
def customers_tip := 15

theorem before_lunch_rush_customers : 
  original_customers_before_lunch + added_customers_during_lunch = customers_no_tip + customers_tip → 
  original_customers_before_lunch = 29 := 
by
  sorry

end before_lunch_rush_customers_l195_195159


namespace perfect_square_trinomial_l195_195539

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, (∀ x : ℝ, x^2 + 2*(m - 1)*x + 16 = (x + a)^2) ∨ (∀ x : ℝ, x^2 + 2*(m - 1)*x + 16 = (x - a)^2)) ↔ m = 5 ∨ m = -3 :=
sorry

end perfect_square_trinomial_l195_195539


namespace solve_triangle_l195_195243

noncomputable def angle_A := 45
noncomputable def angle_B := 60
noncomputable def side_a := Real.sqrt 2

theorem solve_triangle {A B : ℕ} {a b : Real}
    (hA : A = angle_A)
    (hB : B = angle_B)
    (ha : a = side_a) :
    b = Real.sqrt 3 := 
by sorry

end solve_triangle_l195_195243


namespace sum_of_triangle_areas_in_cube_l195_195442

theorem sum_of_triangle_areas_in_cube :
  let m : ℤ := 48,
      n : ℤ := 4608,
      p : ℤ := 3072
  in m + n + p = 7728 :=
by
  sorry

end sum_of_triangle_areas_in_cube_l195_195442


namespace pascal_triangle_contains_47_l195_195209

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l195_195209


namespace vector_relationship_l195_195902

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
          (A A1 B D E : V) (x y z : ℝ)

-- Given Conditions
def inside_top_face_A1B1C1D1 (E : V) : Prop :=
  ∃ (y z : ℝ), (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧
  E = A1 + y • (B - A) + z • (D - A)

-- Prove the desired relationship
theorem vector_relationship (h : E = x • (A1 - A) + y • (B - A) + z • (D - A))
  (hE : inside_top_face_A1B1C1D1 A A1 B D E) : 
  x = 1 ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) :=
sorry

end vector_relationship_l195_195902


namespace simultaneous_solution_exists_l195_195046

-- Definitions required by the problem
def eqn1 (m x : ℝ) : ℝ := m * x + 2
def eqn2 (m x : ℝ) : ℝ := (3 * m - 2) * x + 5

-- Proof statement
theorem simultaneous_solution_exists (m : ℝ) : 
  (∃ x y : ℝ, y = eqn1 m x ∧ y = eqn2 m x) ↔ (m ≠ 1) := 
sorry

end simultaneous_solution_exists_l195_195046


namespace max_area_triangle_l195_195259

open Real

theorem max_area_triangle (a b : ℝ) (C : ℝ) (h₁ : a + b = 4) (h₂ : C = π / 6) : 
  (1 : ℝ) ≥ (1 / 2 * a * b * sin (π / 6)) := 
by 
  sorry

end max_area_triangle_l195_195259


namespace isosceles_triangle_perimeter_l195_195741

def is_isosceles_triangle (A B C : ℝ) : Prop :=
  (A = B ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (A = C ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (B = C ∧ A + B > C ∧ A + C > B ∧ B + C > A)

theorem isosceles_triangle_perimeter {A B C : ℝ} 
  (h : is_isosceles_triangle A B C) 
  (h1 : A = 3 ∨ A = 7) 
  (h2 : B = 3 ∨ B = 7) 
  (h3 : C = 3 ∨ C = 7) : 
  A + B + C = 17 := 
sorry

end isosceles_triangle_perimeter_l195_195741


namespace points_on_intersecting_lines_l195_195027

def clubsuit (a b : ℝ) := a^3 * b - a * b^3

theorem points_on_intersecting_lines (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = y ∨ x = -y) := 
by
  sorry

end points_on_intersecting_lines_l195_195027


namespace quadratic_transformation_l195_195352

theorem quadratic_transformation :
  ∀ x : ℝ, (x^2 - 6 * x - 5 = 0) → ((x - 3)^2 = 14) :=
by
  intros x h
  sorry

end quadratic_transformation_l195_195352


namespace pascal_triangle_47_rows_l195_195220

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l195_195220


namespace triangle_right_if_angle_difference_l195_195260

noncomputable def is_right_triangle (A B C : ℝ) : Prop := 
  A = 90

theorem triangle_right_if_angle_difference (A B C : ℝ) (h : A - B = C) (sum_angles : A + B + C = 180) :
  is_right_triangle A B C :=
  sorry

end triangle_right_if_angle_difference_l195_195260


namespace largest_four_digit_number_mod_l195_195291

theorem largest_four_digit_number_mod (n : ℕ) : 
  (n < 10000) → 
  (n % 11 = 2) → 
  (n % 7 = 4) → 
  n ≤ 9973 :=
by
  sorry

end largest_four_digit_number_mod_l195_195291


namespace sara_total_payment_l195_195272

structure DecorationCosts where
  balloons: ℝ
  tablecloths: ℝ
  streamers: ℝ
  banners: ℝ
  confetti: ℝ
  change_received: ℝ

noncomputable def total_cost (c : DecorationCosts) : ℝ :=
  c.balloons + c.tablecloths + c.streamers + c.banners + c.confetti

noncomputable def amount_given (c : DecorationCosts) : ℝ :=
  total_cost c + c.change_received

theorem sara_total_payment : 
  ∀ (costs : DecorationCosts), 
    costs = ⟨3.50, 18.25, 9.10, 14.65, 7.40, 6.38⟩ →
    amount_given costs = 59.28 :=
by
  intros
  sorry

end sara_total_payment_l195_195272


namespace combination_count_l195_195896

open Nat

def valid_combinations (cards : List Nat) : Bool :=
  ∀ i j, {i, j} ⊆ Finset.range cards.length → |cards[i] - cards[j]| ≥ 2

theorem combination_count :
  ∃ card_selections : Finset (Finset Fin 7), card_selections.card = 10 ∧
  ∀ cards ∈ card_selections, valid_combinations cards.to_list :=
begin
  sorry,
end

end combination_count_l195_195896


namespace complex_number_real_implies_m_is_5_l195_195400

theorem complex_number_real_implies_m_is_5 (m : ℝ) (h : m^2 - 2 * m - 15 = 0) : m = 5 :=
  sorry

end complex_number_real_implies_m_is_5_l195_195400


namespace simplify_expression_l195_195182

theorem simplify_expression (x : ℝ) : 
  ( ( (x^(16/8))^(1/4) )^3 * ( (x^(16/4))^(1/8) )^5 ) = x^4 := 
by 
  sorry

end simplify_expression_l195_195182


namespace inequality_one_inequality_two_l195_195997

-- Problem (1)
theorem inequality_one {a b : ℝ} (h1 : a ≥ b) (h2 : b > 0) : 2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b :=
sorry

-- Problem (2)
theorem inequality_two {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) : (a ^ 2 / b + b ^ 2 / c + c ^ 2 / a) ≥ 1 :=
sorry

end inequality_one_inequality_two_l195_195997


namespace sqrt_164_between_12_and_13_l195_195465

theorem sqrt_164_between_12_and_13 : 12 < Real.sqrt 164 ∧ Real.sqrt 164 < 13 :=
sorry

end sqrt_164_between_12_and_13_l195_195465


namespace lucas_min_deliveries_l195_195949

theorem lucas_min_deliveries (cost_of_scooter earnings_per_delivery fuel_cost_per_delivery parking_fee_per_delivery : ℕ)
  (cost_eq : cost_of_scooter = 3000)
  (earnings_eq : earnings_per_delivery = 12)
  (fuel_cost_eq : fuel_cost_per_delivery = 4)
  (parking_fee_eq : parking_fee_per_delivery = 1) :
  ∃ d : ℕ, 7 * d ≥ cost_of_scooter ∧ d = 429 := by
  sorry

end lucas_min_deliveries_l195_195949


namespace minimum_value_am_hm_l195_195421

theorem minimum_value_am_hm (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hpqr : p + q + r = 3) :
  \(\frac{1}{p + 3q} + \frac{1}{q + 3r} + \frac{1}{r + 3p} \geq \frac{3}{4}\) :=
by
  sorry

end minimum_value_am_hm_l195_195421


namespace intersection_M_N_l195_195678

def M : Set ℕ := {3, 5, 6, 8}
def N : Set ℕ := {4, 5, 7, 8}

theorem intersection_M_N : M ∩ N = {5, 8} :=
  sorry

end intersection_M_N_l195_195678


namespace increase_by_150_percent_l195_195648

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l195_195648


namespace coconut_grove_problem_l195_195547

variable (x : ℝ)

-- Conditions
def trees_yield_40_nuts_per_year : ℝ := 40 * (x + 2)
def trees_yield_120_nuts_per_year : ℝ := 120 * x
def trees_yield_180_nuts_per_year : ℝ := 180 * (x - 2)
def average_yield_per_tree_per_year : ℝ := 100

-- Problem Statement
theorem coconut_grove_problem
  (yield_40_trees : trees_yield_40_nuts_per_year x = 40 * (x + 2))
  (yield_120_trees : trees_yield_120_nuts_per_year x = 120 * x)
  (yield_180_trees : trees_yield_180_nuts_per_year x = 180 * (x - 2))
  (average_yield : average_yield_per_tree_per_year = 100) :
  x = 7 :=
by
  sorry

end coconut_grove_problem_l195_195547


namespace max_sector_area_l195_195544

theorem max_sector_area (r l : ℝ) (hp : 2 * r + l = 40) : (1 / 2) * l * r ≤ 100 := 
by
  sorry

end max_sector_area_l195_195544


namespace sequence_sum_l195_195197

theorem sequence_sum (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) = (1/3) * a n) (h_a4a5 : a 4 + a 5 = 4) :
    a 2 + a 3 = 36 :=
    sorry

end sequence_sum_l195_195197


namespace largest_fraction_is_D_l195_195455

-- Define the fractions as Lean variables
def A : ℚ := 2 / 6
def B : ℚ := 3 / 8
def C : ℚ := 4 / 12
def D : ℚ := 7 / 16
def E : ℚ := 9 / 24

-- Define a theorem to prove the largest fraction is D
theorem largest_fraction_is_D : max (max (max A B) (max C D)) E = D :=
by
  sorry

end largest_fraction_is_D_l195_195455


namespace suitable_communication_l195_195832

def is_suitable_to_communicate (beijing_time : Nat) (sydney_difference : Int) (los_angeles_difference : Int) : Bool :=
  let sydney_time := beijing_time + sydney_difference
  let los_angeles_time := beijing_time - los_angeles_difference
  sydney_time >= 8 ∧ sydney_time <= 22 -- let's assume suitable time is between 8:00 to 22:00

theorem suitable_communication:
  let beijing_time := 18
  let sydney_difference := 2
  let los_angeles_difference := 15
  is_suitable_to_communicate beijing_time sydney_difference los_angeles_difference = true :=
by
  sorry

end suitable_communication_l195_195832


namespace sin_2phi_l195_195778

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l195_195778


namespace part1_solution_set_a_eq_2_part2_range_of_a_l195_195725

noncomputable def f (a x : ℝ) : ℝ := abs (x - a) + abs (2 * x - 2)

theorem part1_solution_set_a_eq_2 :
  { x : ℝ | f 2 x > 2 } = { x | x < (2 / 3) } ∪ { x | x > 2 } :=
by
  sorry

theorem part2_range_of_a :
  { a : ℝ | ∀ x : ℝ, f a x ≥ 2 } = { a | a ≤ -1 } ∪ { a | a ≥ 3 } :=
by
  sorry

end part1_solution_set_a_eq_2_part2_range_of_a_l195_195725


namespace increase_result_l195_195646

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l195_195646


namespace richard_remaining_distance_l195_195819

noncomputable def remaining_distance : ℝ :=
  let d1 := 45
  let d2 := d1 / 2 - 8
  let d3 := 2 * d2 - 4
  let d4 := (d1 + d2 + d3) / 3 + 3
  let d5 := 0.7 * d4
  let total_walked := d1 + d2 + d3 + d4 + d5
  635 - total_walked

theorem richard_remaining_distance : abs (remaining_distance - 497.5166) < 0.0001 :=
by
  sorry

end richard_remaining_distance_l195_195819


namespace zack_travel_countries_l195_195324

theorem zack_travel_countries (G J P Z : ℕ) 
  (hG : G = 6)
  (hJ : J = G / 2)
  (hP : P = 3 * J)
  (hZ : Z = 2 * P) :
  Z = 18 := by
  sorry

end zack_travel_countries_l195_195324


namespace externally_tangent_circles_m_l195_195524

def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 = 4

def circle2_eqn (x y m : ℝ) : Prop := x^2 + y^2 - 2 * m * x + m^2 - 1 = 0

theorem externally_tangent_circles_m (m : ℝ) :
  (∀ x y : ℝ, circle1_eqn x y) →
  (∀ x y : ℝ, circle2_eqn x y m) →
  m = 3 ∨ m = -3 :=
by sorry

end externally_tangent_circles_m_l195_195524


namespace combined_weight_l195_195566

-- Define the conditions
variables (Ron_weight Roger_weight Rodney_weight : ℕ)

-- Define the conditions as Lean propositions
def conditions : Prop :=
  Rodney_weight = 2 * Roger_weight ∧ 
  Roger_weight = 4 * Ron_weight - 7 ∧ 
  Rodney_weight = 146

-- Define the proof goal
def proof_goal : Prop :=
  Rodney_weight + Roger_weight + Ron_weight = 239

theorem combined_weight (Ron_weight Roger_weight Rodney_weight : ℕ) (h : conditions Ron_weight Roger_weight Rodney_weight) : 
  proof_goal Ron_weight Roger_weight Rodney_weight :=
sorry

end combined_weight_l195_195566


namespace ladder_leaning_distance_l195_195015

variable (m f h : ℝ)
variable (f_pos : f > 0) (h_pos : h > 0)

def distance_to_wall_upper_bound : ℝ := 12.46
def distance_to_wall_lower_bound : ℝ := 8.35

theorem ladder_leaning_distance (m f h : ℝ) (f_pos : f > 0) (h_pos : h > 0) :
  ∃ x : ℝ, x = 12.46 ∨ x = 8.35 := 
sorry

end ladder_leaning_distance_l195_195015


namespace sum_of_k_values_l195_195487

theorem sum_of_k_values (k : ℤ) :
  (∃ (r s : ℤ), (r ≠ s) ∧ (3 * r * s = 9) ∧ (r + s = k / 3)) → k = 0 :=
by sorry

end sum_of_k_values_l195_195487


namespace assignment_correct_l195_195894

noncomputable def task_assignment_count : ℕ :=
  (nat.choose 3 1 * nat.choose 4 2 * nat.perm 3 3) + (nat.choose 3 2 * nat.perm 3 3)

theorem assignment_correct:
  (let A := "A"; B := "B"; C := "C"; D := "D"; E := "E";
       tasks := ["Translation", "TourGuide", "Etiquette", "Driver"];
       count := task_assignment_count in
   count = (nat.choose 3 1 * nat.choose 4 2 * nat.perm 3 3) + (nat.choose 3 2 * nat.perm 3 3)) :=
by
  -- Proof would go here
  sorry

end assignment_correct_l195_195894


namespace sugar_water_inequality_one_sugar_water_inequality_two_l195_195955

variable (a b m : ℝ)

-- Condition constraints
variable (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m)

-- Sugar Water Experiment One Inequality
theorem sugar_water_inequality_one : a / b > a / (b + m) := 
by
  sorry

-- Sugar Water Experiment Two Inequality
theorem sugar_water_inequality_two : a / b < (a + m) / b := 
by
  sorry

end sugar_water_inequality_one_sugar_water_inequality_two_l195_195955


namespace abs_diff_roots_eq_3_l195_195038

theorem abs_diff_roots_eq_3 : ∀ (r1 r2 : ℝ), (r1 ≠ r2) → (r1 + r2 = 7) → (r1 * r2 = 10) → |r1 - r2| = 3 :=
by
  intros r1 r2 hneq hsum hprod
  sorry

end abs_diff_roots_eq_3_l195_195038


namespace other_endpoint_of_diameter_l195_195482

theorem other_endpoint_of_diameter (center endpoint : ℝ × ℝ) (hc : center = (1, 2)) (he : endpoint = (4, 6)) :
    ∃ other_endpoint : ℝ × ℝ, other_endpoint = (-2, -2) :=
by
  sorry

end other_endpoint_of_diameter_l195_195482


namespace sandy_younger_than_molly_l195_195567

variable (s m : ℕ)
variable (h_ratio : 7 * m = 9 * s)
variable (h_sandy : s = 56)

theorem sandy_younger_than_molly : 
  m - s = 16 := 
by
  sorry

end sandy_younger_than_molly_l195_195567


namespace budget_remaining_l195_195866

noncomputable theory
open_locale big_operators

def conversion_rate : ℝ := 1.2
def last_year_budget_euros : ℝ := 6
def this_year_allocation : ℝ := 50
def additional_grant : ℝ := 20
def gift_card : ℝ := 10

def initial_price_textbooks : ℝ := 45
def discount_textbooks : ℝ := 0.15
def tax_textbooks : ℝ := 0.08

def initial_price_notebooks : ℝ := 18
def discount_notebooks : ℝ := 0.10
def tax_notebooks : ℝ := 0.05

def initial_price_pens : ℝ := 27
def discount_pens : ℝ := 0.05
def tax_pens : ℝ := 0.06

def initial_price_art_supplies : ℝ := 35
def discount_art_supplies : ℝ := 0
def tax_art_supplies : ℝ := 0.07

def initial_price_folders : ℝ := 15
def voucher_folders : ℝ := 5
def tax_folders : ℝ := 0.04

def convert_budget (e: ℝ) (r: ℝ): ℝ := e * r

def calculate_discounted_price (price: ℝ) (discount: ℝ) : ℝ :=
  price - price * discount

def calculate_tax_price (price: ℝ) (tax: ℝ) : ℝ :=
  price + price * tax

def compute_final_price (price: ℝ) (discount: ℝ) (tax: ℝ) : ℝ :=
  calculate_tax_price (calculate_discounted_price(price, discount), tax)

def total_budget : ℝ :=
  convert_budget last_year_budget_euros conversion_rate +
  this_year_allocation +
  additional_grant +
  gift_card

def cost_textbooks : ℝ :=
  compute_final_price(initial_price_textbooks, discount_textbooks, tax_textbooks)

def cost_notebooks : ℝ :=
  compute_final_price(initial_price_notebooks, discount_notebooks, tax_notebooks)

def cost_pens : ℝ :=
  compute_final_price(initial_price_pens, discount_pens, tax_pens)

def cost_art_supplies : ℝ :=
  compute_final_price(initial_price_art_supplies, discount_art_supplies, tax_art_supplies)

def cost_folders : ℝ :=
  compute_final_price(initial_price_folders - voucher_folders, 0, tax_folders)

def total_cost : ℝ :=
  cost_textbooks +
  cost_notebooks +
  cost_pens +
  cost_art_supplies +
  cost_folders

theorem budget_remaining :
  total_budget - (total_cost - gift_card) = -36.16 :=
sorry

end budget_remaining_l195_195866


namespace list_price_l195_195830

theorem list_price (P : ℝ) (h₀ : 0.83817 * P = 56.16) : P = 67 :=
sorry

end list_price_l195_195830


namespace terry_total_miles_l195_195825

def total_gasoline_used := 9 + 17
def average_gas_mileage := 30

theorem terry_total_miles (M : ℕ) : 
  total_gasoline_used * average_gas_mileage = M → M = 780 :=
by
  intro h
  rw [←h]
  sorry

end terry_total_miles_l195_195825


namespace problem_condition_l195_195537

theorem problem_condition (x y : ℝ) (h : x^2 + y^2 - x * y = 1) : 
  x + y ≥ -2 ∧ x^2 + y^2 ≤ 2 :=
by
  sorry

end problem_condition_l195_195537


namespace alpha_beta_sum_l195_195446

theorem alpha_beta_sum (α β : ℝ) (h : ∀ x : ℝ, x ≠ 54 → x ≠ -60 → (x - α) / (x + β) = (x^2 - 72 * x + 945) / (x^2 + 45 * x - 3240)) :
  α + β = 81 :=
sorry

end alpha_beta_sum_l195_195446


namespace jane_percentage_decrease_l195_195798

theorem jane_percentage_decrease
  (B H : ℝ) -- Number of bears Jane makes per week and hours she works per week
  (H' : ℝ) -- Number of hours Jane works per week with assistant
  (h1 : H' = 0.9 * H) -- New working hours when Jane works with an assistant
  (h2 : H ≠ 0) -- Ensure H is not zero to avoid division by zero
  : ((H - H') / H) * 100 = 10 := 
by calc
  ((H - H') / H) * 100
      = ((H - 0.9 * H) / H) * 100 : by rw [h1]
  ... = (0.1 * H / H) * 100 : by simp
  ... = 0.1 * 100 : by rw [div_self h2]
  ... = 10 : by norm_num

end jane_percentage_decrease_l195_195798


namespace stratified_sampling_correct_l195_195329

-- Define the total number of students and the ratio of students in grades 10, 11, and 12
def total_students : ℕ := 4000
def ratio_grade10 : ℕ := 32
def ratio_grade11 : ℕ := 33
def ratio_grade12 : ℕ := 35

-- The total sample size
def sample_size : ℕ := 200

-- Define the expected numbers of students drawn from each grade in the sample
def sample_grade10 : ℕ := 64
def sample_grade11 : ℕ := 66
def sample_grade12 : ℕ := 70

-- The theorem to be proved
theorem stratified_sampling_correct :
  (sample_grade10 + sample_grade11 + sample_grade12 = sample_size) ∧
  (sample_grade10 = (ratio_grade10 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) ∧
  (sample_grade11 = (ratio_grade11 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) ∧
  (sample_grade12 = (ratio_grade12 * sample_size) / (ratio_grade10 + ratio_grade11 + ratio_grade12)) :=
by
  sorry

end stratified_sampling_correct_l195_195329


namespace ab_divisible_by_6_l195_195505

theorem ab_divisible_by_6
  (n : ℕ) (a b : ℕ)
  (h1 : 2^n = 10 * a + b)
  (h2 : n > 3)
  (h3 : b < 10) :
  (a * b) % 6 = 0 :=
sorry

end ab_divisible_by_6_l195_195505


namespace distance_from_original_position_l195_195013

/-- Definition of initial problem conditions and parameters --/
def square_area (l : ℝ) : Prop :=
  l * l = 18

def folded_area_relation (x : ℝ) : Prop :=
  0.5 * x^2 = 2 * (18 - 0.5 * x^2)

/-- The main statement that needs to be proved --/
theorem distance_from_original_position :
  ∃ (A_initial A_folded_dist : ℝ),
    square_area A_initial ∧
    (∃ x : ℝ, folded_area_relation x ∧ A_folded_dist = 2 * Real.sqrt 6 * Real.sqrt 2) ∧
    A_folded_dist = 4 * Real.sqrt 3 :=
by
  -- The proof is omitted here; providing structure for the problem.
  sorry

end distance_from_original_position_l195_195013


namespace min_value_d_l195_195426

theorem min_value_d (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (unique_solution : ∃! x y : ℤ, 2 * x + y = 2007 ∧ y = (abs (x - a) + abs (x - b) + abs (x - c) + abs (x - d))) :
  d = 504 :=
sorry

end min_value_d_l195_195426


namespace carrots_picked_by_mother_l195_195032

-- Define the conditions
def faye_picked : ℕ := 23
def good_carrots : ℕ := 12
def bad_carrots : ℕ := 16

-- Define the problem of the total number of carrots
def total_carrots : ℕ := good_carrots + bad_carrots

-- Define the mother's picked carrots
def mother_picked (total_faye : ℕ) (total : ℕ) := total - total_faye

-- State the theorem
theorem carrots_picked_by_mother (faye_picked : ℕ) (total_carrots : ℕ) : mother_picked faye_picked total_carrots = 5 := by
  sorry

end carrots_picked_by_mother_l195_195032


namespace correct_statement_l195_195353

def correct_input_format_1 (s : String) : Prop :=
  s = "INPUT a, b, c"

def correct_input_format_2 (s : String) : Prop :=
  s = "INPUT x="

def correct_output_format_1 (s : String) : Prop :=
  s = "PRINT A="

def correct_output_format_2 (s : String) : Prop :=
  s = "PRINT 3*2"

theorem correct_statement : (correct_input_format_1 "INPUT a; b; c" = false) ∧
                            (correct_input_format_2 "INPUT x=3" = false) ∧
                            (correct_output_format_1 "PRINT“A=4”" = false) ∧
                            (correct_output_format_2 "PRINT 3*2" = true) :=
by sorry

end correct_statement_l195_195353


namespace range_of_a_l195_195917

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, a * x ^ 2 + 2 * a * x + 1 ≤ 0) →
  0 ≤ a ∧ a < 1 :=
by
  -- sorry to skip the proof
  sorry

end range_of_a_l195_195917


namespace check_correct_l195_195857

-- Given the conditions
variable (x y : ℕ) (H1 : 10 ≤ x ∧ x ≤ 81) (H2 : y = x + 18)

-- Rewrite the problem and correct answer for verification in Lean
theorem check_correct (Hx : 10 ≤ x ∧ x ≤ 81) (Hy : y = x + 18) : 
  y = 2 * x ↔ x = 18 := 
by
  sorry

end check_correct_l195_195857


namespace arithmetic_geometric_l195_195059

theorem arithmetic_geometric (a : ℕ → ℤ) (d : ℤ) (h1 : d = 2)
  (h2 : ∀ n, a (n + 1) - a n = d)
  (h3 : ∃ r, a 1 * r = a 3 ∧ a 3 * r = a 4) :
  a 2 = -6 :=
by sorry

end arithmetic_geometric_l195_195059


namespace solve_x_l195_195398

theorem solve_x : ∃ x : ℝ, 2^(Real.log 5 / Real.log 2) = 3 * x + 4 ∧ x = 1 / 3 :=
by
  use 1 / 3
  sorry

end solve_x_l195_195398


namespace increased_number_l195_195605

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l195_195605


namespace increase_80_by_150_percent_l195_195659

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l195_195659


namespace spaceship_journey_time_l195_195345

theorem spaceship_journey_time
  (initial_travel_1 : ℕ)
  (first_break : ℕ)
  (initial_travel_2 : ℕ)
  (second_break : ℕ)
  (travel_per_segment : ℕ)
  (break_per_segment : ℕ)
  (total_break_time : ℕ)
  (remaining_break_time : ℕ)
  (num_segments : ℕ)
  (total_travel_time : ℕ)
  (total_time : ℕ) :
  initial_travel_1 = 10 →
  first_break = 3 →
  initial_travel_2 = 10 →
  second_break = 1 →
  travel_per_segment = 11 →
  break_per_segment = 1 →
  total_break_time = 8 →
  remaining_break_time = total_break_time - (first_break + second_break) →
  num_segments = remaining_break_time / break_per_segment →
  total_travel_time = initial_travel_1 + initial_travel_2 + (num_segments * travel_per_segment) →
  total_time = total_travel_time + total_break_time →
  total_time = 72 :=
by
  intros
  sorry

end spaceship_journey_time_l195_195345


namespace panthers_score_l195_195548

-- Definitions as per the conditions
def total_points (C P : ℕ) : Prop := C + P = 48
def margin (C P : ℕ) : Prop := C = P + 20

-- Theorem statement proving Panthers score 14 points
theorem panthers_score (C P : ℕ) (h1 : total_points C P) (h2 : margin C P) : P = 14 :=
sorry

end panthers_score_l195_195548


namespace largest_multiple_11_lt_neg85_l195_195131

-- Define the conditions: a multiple of 11 and smaller than -85
def largest_multiple_lt (m n : Int) : Int :=
  let k := (m / n) - 1
  n * k

-- Define our specific problem
theorem largest_multiple_11_lt_neg85 : largest_multiple_lt (-85) 11 = -88 := 
  by
  sorry

end largest_multiple_11_lt_neg85_l195_195131


namespace sum_of_areas_of_triangles_l195_195441

theorem sum_of_areas_of_triangles (m n p : ℤ) :
  let vertices := {v : ℝ × ℝ × ℝ | v.1 ∈ {0, 2} ∧ v.2 ∈ {0, 2} ∧ v.3 ∈ {0, 2}},
      triangles := {t : tuple ℝ 9 | 
                     ∃ v1 v2 v3 ∈ vertices, 
                     t = (v1.1, v1.2, v1.3, v2.1, v2.2, v2.3, v3.1, v3.2, v3.3)},
      triangle_area := λ t : tuple ℝ 9, 
        let (x1, y1, z1, x2, y2, z2, x3, y3, z3) := t in
        1 / 2 * real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2) *
                 real.sqrt ((x3 - x1) ^ 2 + (y3 - y1) ^ 2 + (z3 - z1) ^ 2) *
                 real.sqrt ((x3 - x2) ^ 2 + (y3 - y2) ^ 2 + (z3 - z2) ^ 2)
  in (∑ t in triangles, triangle_area t) = 48 + real.sqrt 4608 + real.sqrt 3072 :=
sorry

end sum_of_areas_of_triangles_l195_195441


namespace find_angle_C_l195_195733

theorem find_angle_C (A B C : ℝ) (h1 : A = 88) (h2 : B - C = 20) (angle_sum : A + B + C = 180) : C = 36 :=
by
  sorry

end find_angle_C_l195_195733


namespace increase_80_by_150_percent_l195_195638

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l195_195638


namespace time_since_production_approximate_l195_195099

noncomputable def solve_time (N N₀ : ℝ) (t : ℝ) : Prop :=
  N = N₀ * (1 / 2) ^ (t / 5730) ∧
  N / N₀ = 3 / 8 ∧
  t = 8138

theorem time_since_production_approximate
  (N N₀ : ℝ)
  (h_decay : N = N₀ * (1 / 2) ^ (t / 5730))
  (h_ratio : N / N₀ = 3 / 8) :
  t = 8138 := 
sorry

end time_since_production_approximate_l195_195099


namespace range_my_function_l195_195293

noncomputable def my_function (x : ℝ) := (x^2 + 4 * x + 3) / (x + 2)

theorem range_my_function : 
  Set.range my_function = Set.univ := 
sorry

end range_my_function_l195_195293


namespace find_M_l195_195166

theorem find_M : 
  ∃ M : ℚ, 
  (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
  M = 610 / 1657 :=
by
  sorry

end find_M_l195_195166


namespace increased_number_l195_195608

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l195_195608


namespace pascal_row_contains_prime_47_l195_195230

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l195_195230


namespace last_month_games_l195_195098

-- Definitions and conditions
def this_month := 9
def next_month := 7
def total_games := 24

-- Question to prove
theorem last_month_games : total_games - (this_month + next_month) = 8 := 
by 
  sorry

end last_month_games_l195_195098


namespace remainder_when_7n_divided_by_4_l195_195308

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l195_195308


namespace book_cost_l195_195488

theorem book_cost (b : ℝ) : (11 * b < 15) ∧ (12 * b > 16.20) → b = 1.36 :=
by
  intros h
  sorry

end book_cost_l195_195488


namespace points_do_not_exist_l195_195359

/-- 
  If \( A, B, C, D \) are four points in space and 
  \( AB = 8 \) cm, 
  \( CD = 8 \) cm, 
  \( AC = 10 \) cm, 
  \( BD = 10 \) cm, 
  \( AD = 13 \) cm, 
  \( BC = 13 \) cm, 
  then such points \( A, B, C, D \) cannot exist.
-/
theorem points_do_not_exist 
  (A B C D : Type)
  (AB CD AC BD AD BC : ℝ) 
  (h1 : AB = 8) 
  (h2 : CD = 8) 
  (h3 : AC = 10)
  (h4 : BD = 10)
  (h5 : AD = 13)
  (h6 : BC = 13) : 
  false :=
sorry

end points_do_not_exist_l195_195359


namespace sin_double_angle_l195_195761

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l195_195761


namespace f_g_2_eq_1_l195_195234

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := -2 * x + 5

theorem f_g_2_eq_1 : f (g 2) = 1 :=
by
  sorry

end f_g_2_eq_1_l195_195234


namespace find_m_value_l195_195713

theorem find_m_value (x m : ℝ)
  (h1 : -3 * x = -5 * x + 4)
  (h2 : m^x - 9 = 0) :
  m = 3 ∨ m = -3 := 
sorry

end find_m_value_l195_195713


namespace maximum_matches_l195_195838

theorem maximum_matches (A B C : ℕ) (h1 : A > B) (h2 : B > C) 
    (h3 : A ≥ B + 10) (h4 : B ≥ C + 10) (h5 : B + C > A) : 
    A + B + C - 1 ≤ 62 :=
sorry

end maximum_matches_l195_195838


namespace divide_square_into_equal_parts_l195_195881

-- Given a square with four shaded smaller squares inside
structure SquareWithShaded (n : ℕ) :=
  (squares : Fin n → Fin n → Prop) -- this models the presence of shaded squares
  (shaded : (Fin 2) → (Fin 2) → Prop)

-- To prove: we can divide the square into four equal parts with each containing one shaded square
theorem divide_square_into_equal_parts :
  ∀ (sq : SquareWithShaded 4),
  ∃ (parts : Fin 2 → Fin 2 → Prop),
  (∀ i j, parts i j ↔ 
    ((i = 0 ∧ j = 0) ∨ (i = 1 ∧ j = 0) ∨ 
    (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 1)) ∧
    (∃! k l, sq.shaded k l ∧ parts i j)) :=
sorry

end divide_square_into_equal_parts_l195_195881


namespace increase_by_percentage_l195_195620

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l195_195620


namespace plane_divides_space_into_two_parts_l195_195010

def divides_space : Prop :=
  ∀ (P : ℝ → ℝ → ℝ → Prop), (∀ x y z, P x y z → P x y z) →
  (∃ region1 region2 : ℝ → ℝ → ℝ → Prop,
    (∀ x y z, P x y z → (region1 x y z ∨ region2 x y z)) ∧
    (∀ x y z, region1 x y z → ¬region2 x y z) ∧
    (∃ x1 y1 z1 x2 y2 z2, region1 x1 y1 z1 ∧ region2 x2 y2 z2))

theorem plane_divides_space_into_two_parts (P : ℝ → ℝ → ℝ → Prop) (hP : ∀ x y z, P x y z → P x y z) : 
  divides_space :=
  sorry

end plane_divides_space_into_two_parts_l195_195010


namespace Ms_Smiths_Class_Books_Distribution_l195_195928

theorem Ms_Smiths_Class_Books_Distribution :
  ∃ (x : ℕ), (20 * 2 * x + 15 * x + 5 * x = 840) ∧ (20 * 2 * x = 560) ∧ (15 * x = 210) ∧ (5 * x = 70) :=
by
  let x := 14
  have h1 : 20 * 2 * x + 15 * x + 5 * x = 840 := by sorry
  have h2 : 20 * 2 * x = 560 := by sorry
  have h3 : 15 * x = 210 := by sorry
  have h4 : 5 * x = 70 := by sorry
  exact ⟨x, h1, h2, h3, h4⟩

end Ms_Smiths_Class_Books_Distribution_l195_195928


namespace c_share_l195_195464

theorem c_share (A B C : ℝ) 
  (h1 : A = (1 / 2) * B)
  (h2 : B = (1 / 2) * C)
  (h3 : A + B + C = 392) : 
  C = 224 :=
by
  sorry

end c_share_l195_195464


namespace range_a_f_x_neg_l195_195914

noncomputable def f (a x : ℝ) : ℝ := x^2 + (2 * a - 1) * x - 3

theorem range_a_f_x_neg (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ f a x < 0) → a < 3 / 2 := sorry

end range_a_f_x_neg_l195_195914


namespace remainder_of_n_squared_plus_4n_plus_5_l195_195395

theorem remainder_of_n_squared_plus_4n_plus_5 {n : ℤ} (h : n % 50 = 1) : (n^2 + 4*n + 5) % 50 = 10 :=
by
  sorry

end remainder_of_n_squared_plus_4n_plus_5_l195_195395


namespace area_square_l195_195280

-- Define the conditions
variables (l r s : ℝ)
variable (breadth : ℝ := 10)
variable (area_rect : ℝ := 180)

-- Given conditions
def length_is_two_fifths_radius : Prop := l = (2/5) * r
def radius_is_side_square : Prop := r = s
def area_of_rectangle : Prop := area_rect = l * breadth

-- The theorem statement
theorem area_square (h1 : length_is_two_fifths_radius l r)
                    (h2 : radius_is_side_square r s)
                    (h3 : area_of_rectangle l breadth area_rect) :
  s^2 = 2025 :=
by
  sorry

end area_square_l195_195280


namespace smallest_integer_with_eight_minimal_fibonacci_ones_l195_195839

open Nat

-- Define the minimal Fibonacci representation condition
def is_minimal_fibonacci_representation (k : ℕ) (a : ℕ → ℕ) : Prop :=
  k = ∑ i in finset.range (nat.succ (finset.sup (finset.filter (λ i, a i = 1) finset.univ))), a i * fib i ∧ 
  ∀ i, a i ∈ {0, 1} ∧ a (nat.succ (finset.sup (finset.filter (λ i, a i = 1) finset.univ))) = 1 ∧ 
  -- Ensure non-consecutiveness (Zeckendorf's condition)
  ∀ i, (a i = 1 → a (nat.succ i) = 0) ∧ (a i = 1 → a (nat.succ (nat.succ i)) = 0)

-- Define the condition of exactly eight ones in the representation
def exactly_eight_ones (a : ℕ → ℕ) : Prop :=
  (finset.filter (λ i, a i = 1) finset.univ).card = 8

noncomputable def minimal_fibonacci_representation_eight_ones : ℕ :=
  ∑ i in finset.range 8, fib (2 * i + 2)

theorem smallest_integer_with_eight_minimal_fibonacci_ones : minimal_fibonacci_representation_eight_ones = 1596 :=
by 
  sorry

end smallest_integer_with_eight_minimal_fibonacci_ones_l195_195839


namespace xiangshan_port_investment_scientific_notation_l195_195964

-- Definition of scientific notation
def in_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ x = a * 10^n

-- Theorem stating the equivalence of the investment in scientific notation
theorem xiangshan_port_investment_scientific_notation :
  in_scientific_notation 7.7 9 7.7e9 :=
by {
  sorry
}

end xiangshan_port_investment_scientific_notation_l195_195964


namespace cheesecake_factory_hours_per_day_l195_195565

theorem cheesecake_factory_hours_per_day
  (wage_per_hour : ℝ)
  (days_per_week : ℝ)
  (weeks : ℝ)
  (combined_savings : ℝ)
  (robbie_saves : ℝ)
  (jaylen_saves : ℝ)
  (miranda_saves : ℝ)
  (h : ℝ) :
  wage_per_hour = 10 → days_per_week = 5 → weeks = 4 → combined_savings = 3000 →
  robbie_saves = 2/5 → jaylen_saves = 3/5 → miranda_saves = 1/2 →
  (robbie_saves * (wage_per_hour * h * days_per_week) +
  jaylen_saves * (wage_per_hour * h * days_per_week) +
  miranda_saves * (wage_per_hour * h * days_per_week)) * weeks = combined_savings →
  h = 10 :=
by
  intros hwage hweek hweeks hsavings hrobbie hjaylen hmiranda heq
  sorry

end cheesecake_factory_hours_per_day_l195_195565


namespace problem_1_simplified_problem_2_simplified_l195_195367

noncomputable def problem_1 : ℝ :=
  2 * Real.sqrt 18 - Real.sqrt 50 + (1/2) * Real.sqrt 32

theorem problem_1_simplified : problem_1 = 3 * Real.sqrt 2 :=
  sorry

noncomputable def problem_2 : ℝ :=
  (Real.sqrt 5 + Real.sqrt 6) * (Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - 1)^2

theorem problem_2_simplified : problem_2 = -7 + 2 * Real.sqrt 5 :=
  sorry

end problem_1_simplified_problem_2_simplified_l195_195367


namespace april_rainfall_correct_l195_195734

-- Define the constants for the rainfalls in March and the difference in April
def march_rainfall : ℝ := 0.81
def rain_difference : ℝ := 0.35

-- Define the expected April rainfall based on the conditions
def april_rainfall : ℝ := march_rainfall - rain_difference

-- Theorem to prove that the April rainfall is 0.46 inches
theorem april_rainfall_correct : april_rainfall = 0.46 :=
by
  -- Placeholder for the proof
  sorry

end april_rainfall_correct_l195_195734


namespace distinct_arrangements_of_PHONE_l195_195918

-- Condition: The word PHONE consists of 5 distinct letters
def distinctLetters := 5

-- Theorem: The number of distinct arrangements of the letters in the word PHONE
theorem distinct_arrangements_of_PHONE : Nat.factorial distinctLetters = 120 := sorry

end distinct_arrangements_of_PHONE_l195_195918


namespace arithmetic_sequence_8th_term_l195_195571

theorem arithmetic_sequence_8th_term (a d : ℤ) :
  (a + d = 25) ∧ (a + 5 * d = 49) → (a + 7 * d = 61) :=
by
  sorry

end arithmetic_sequence_8th_term_l195_195571


namespace remainder_when_7n_divided_by_4_l195_195307

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l195_195307


namespace abs_diff_of_solutions_l195_195036

theorem abs_diff_of_solutions (h : ∀ x : ℝ, x^2 - 7 * x + 10 = 0 → x = 2 ∨ x = 5) :
  |(2 - 5 : ℝ)| = 3 :=
by sorry

end abs_diff_of_solutions_l195_195036


namespace sin_double_angle_solution_l195_195785

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l195_195785


namespace increased_number_l195_195609

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l195_195609


namespace circumscribed_circle_radius_l195_195111

theorem circumscribed_circle_radius (r : ℝ) (π : ℝ)
  (isosceles_right_triangle : Type) 
  (perimeter : isosceles_right_triangle → ℝ )
  (area : ℝ → ℝ)
  (h : ∀ (t : isosceles_right_triangle), perimeter t = area r) :
  r = (1 + Real.sqrt 2) / π :=
sorry

end circumscribed_circle_radius_l195_195111


namespace equivalent_contrapositive_l195_195976

-- Given definitions
variables {Person : Type} (possess : Person → Prop) (happy : Person → Prop)

-- The original statement: "If someone is happy, then they possess it."
def original_statement : Prop := ∀ p : Person, happy p → possess p

-- The contrapositive: "If someone does not possess it, then they are not happy."
def contrapositive_statement : Prop := ∀ p : Person, ¬ possess p → ¬ happy p

-- The theorem to prove logical equivalence
theorem equivalent_contrapositive : original_statement possess happy ↔ contrapositive_statement possess happy := 
by sorry

end equivalent_contrapositive_l195_195976


namespace pascals_triangle_contains_47_once_l195_195204

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l195_195204


namespace increase_by_percentage_l195_195602

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l195_195602


namespace percentage_decrease_in_sale_l195_195814

theorem percentage_decrease_in_sale (P Q : ℝ) (D : ℝ)
  (h1 : 1.80 * P * Q * (1 - D / 100) = 1.44 * P * Q) : 
  D = 20 :=
by
  -- Proof goes here
  sorry

end percentage_decrease_in_sale_l195_195814


namespace find_second_month_sale_l195_195151

/-- Given sales for specific months and required sales goal -/
def sales_1 := 4000
def sales_3 := 5689
def sales_4 := 7230
def sales_5 := 6000
def sales_6 := 12557
def avg_goal := 7000
def months := 6

theorem find_second_month_sale (x2 : ℕ) :
  (sales_1 + x2 + sales_3 + sales_4 + sales_5 + sales_6) / months = avg_goal →
  x2 = 6524 :=
by
  sorry

end find_second_month_sale_l195_195151


namespace balls_in_boxes_l195_195528

theorem balls_in_boxes : ∀ (balls boxes : ℕ), balls = 6 → boxes = 3 → 
  (∃! n : ℕ, n = 47 ∧ n = number_of_ways_to_distribute_balls_in_boxes_d(balls, boxes)) :=
by
  intros balls boxes h1 h2
  have h_balls : balls = 6 := h1
  have h_boxes : boxes = 3 := h2
  use 47
  split
  sorry
  sorry

end balls_in_boxes_l195_195528


namespace sick_cows_variance_l195_195861

noncomputable def ξ : ℕ → ℝ := binomial 10 0.02

theorem sick_cows_variance :
  variance (ξ 10) = 0.196 :=
by
  sorry

end sick_cows_variance_l195_195861


namespace increase_150_percent_of_80_l195_195627

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l195_195627


namespace sin_2phi_l195_195777

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l195_195777


namespace inequalities_not_simultaneous_l195_195265

theorem inequalities_not_simultaneous (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (ineq1 : a + b < c + d) (ineq2 : (a + b) * (c + d) < a * b + c * d) (ineq3 : (a + b) * c * d < (c + d) * a * b) :
  false := 
sorry

end inequalities_not_simultaneous_l195_195265


namespace find_multiple_l195_195114

theorem find_multiple (a b m : ℤ) (h1 : b = 7) (h2 : b - a = 2) 
  (h3 : a * b = m * (a + b) + 11) : m = 2 :=
by {
  sorry
}

end find_multiple_l195_195114


namespace puppies_sold_l195_195688

theorem puppies_sold (total_puppies sold_puppies puppies_per_cage total_cages : ℕ)
  (h1 : total_puppies = 102)
  (h2 : puppies_per_cage = 9)
  (h3 : total_cages = 9)
  (h4 : total_puppies - sold_puppies = puppies_per_cage * total_cages) :
  sold_puppies = 21 :=
by {
  -- Proof details would go here
  sorry
}

end puppies_sold_l195_195688


namespace num_pairs_with_math_book_l195_195076

theorem num_pairs_with_math_book (books : Finset String) (h : books = {"Chinese", "Mathematics", "English", "Biology", "History"}):
  (∃ pairs : Finset (Finset String), pairs.card = 4 ∧ ∀ pair ∈ pairs, "Mathematics" ∈ pair) :=
by
  sorry

end num_pairs_with_math_book_l195_195076


namespace gcd_2728_1575_l195_195597

theorem gcd_2728_1575 : Int.gcd 2728 1575 = 1 :=
by sorry

end gcd_2728_1575_l195_195597


namespace hexagon_classroom_students_l195_195858

-- Define the number of sleeping students
def num_sleeping_students (students_detected : Nat → Nat) :=
  students_detected 2 + students_detected 3 + students_detected 6

-- Define the condition that the sum of snore-o-meter readings is 7
def snore_o_meter_sum (students_detected : Nat → Nat) :=
  2 * students_detected 2 + 3 * students_detected 3 + 6 * students_detected 6 = 7

-- Proof that the number of sleeping students is 3 given the conditions
theorem hexagon_classroom_students : 
  ∀ (students_detected : Nat → Nat), snore_o_meter_sum students_detected → num_sleeping_students students_detected = 3 :=
by
  intro students_detected h
  sorry

end hexagon_classroom_students_l195_195858


namespace original_price_of_trouser_l195_195556

theorem original_price_of_trouser (P : ℝ) (sale_price : ℝ) (percent_decrease : ℝ) 
  (h1 : sale_price = 40) (h2 : percent_decrease = 0.60) 
  (h3 : sale_price = P * (1 - percent_decrease)) : P = 100 :=
by
  sorry

end original_price_of_trouser_l195_195556


namespace train_passes_jogger_time_l195_195459

theorem train_passes_jogger_time (speed_jogger_kmph : ℝ) 
                                (speed_train_kmph : ℝ) 
                                (distance_ahead_m : ℝ) 
                                (length_train_m : ℝ) : 
  speed_jogger_kmph = 9 → 
  speed_train_kmph = 45 →
  distance_ahead_m = 250 →
  length_train_m = 120 →
  (distance_ahead_m + length_train_m) / (speed_train_kmph - speed_jogger_kmph) * (1000 / 3600) = 37 :=
by
  intros h1 h2 h3 h4
  sorry

end train_passes_jogger_time_l195_195459


namespace find_root_product_l195_195422

theorem find_root_product :
  (∃ r s t : ℝ, (∀ x : ℝ, (x - r) * (x - s) * (x - t) = x^3 - 15 * x^2 + 26 * x - 8) ∧
  (1 + r) * (1 + s) * (1 + t) = 50) :=
sorry

end find_root_product_l195_195422


namespace part1_part2_l195_195907

variable (a b : ℝ) (x : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) : (a^2 / b) + (b^2 / a) ≥ a + b :=
sorry

theorem part2 (h3 : 0 < x) (h4 : x < 1) : 
(∀ y : ℝ, y = ((1 - x)^2 / x) + (x^2 / (1 - x)) → y ≥ 1) ∧ ((1 - x) = x → y = 1) :=
sorry

end part1_part2_l195_195907


namespace solve_equation_l195_195823

noncomputable def equation (x : ℝ) : ℝ :=
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x))

theorem solve_equation (x : ℝ) (k : ℤ) :
  (equation x = 2 / Real.sqrt 3) ↔
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) :=
sorry

end solve_equation_l195_195823


namespace selling_prices_max_profit_strategy_l195_195684

theorem selling_prices (x y : ℕ) (hx : y - x = 30) (hy : 2 * x + 3 * y = 740) : x = 130 ∧ y = 160 :=
by
  sorry

theorem max_profit_strategy (m : ℕ) (hm : 20 ≤ m ∧ m ≤ 80) 
(hcost : 90 * m + 110 * (80 - m) ≤ 8400) : m = 20 ∧ (80 - m) = 60 :=
by
  sorry

end selling_prices_max_profit_strategy_l195_195684


namespace increase_result_l195_195643

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l195_195643


namespace moment_goal_equality_l195_195982

theorem moment_goal_equality (total_goals_russia total_goals_tunisia : ℕ) (T : total_goals_russia = 9) (T2 : total_goals_tunisia = 5) :
  ∃ n, n ≤ 9 ∧ (9 - n) = total_goals_tunisia :=
by
  sorry

end moment_goal_equality_l195_195982


namespace number_of_kids_l195_195872

theorem number_of_kids (A K : ℕ) (h1 : A + K = 13) (h2 : 7 * A = 28) : K = 9 :=
by
  sorry

end number_of_kids_l195_195872


namespace find_a_b_find_max_m_l195_195724

-- Define the function
def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (3 * x - 2)

-- Conditions
def solution_set_condition (x a : ℝ) : Prop := (-4 * a / 5 ≤ x ∧ x ≤ 3 * a / 5)
def eq_five_condition (x : ℝ) : Prop := f x ≤ 5

-- Prove the values of a and b
theorem find_a_b (a b : ℝ) : (∀ x : ℝ, eq_five_condition x ↔ solution_set_condition x a) → (a = 1 ∧ b = 2) :=
by
  sorry

-- Prove that |x - a| + |x + b| >= m^2 - 3m and find the maximum value of m
theorem find_max_m (a b m : ℝ) : (a = 1 ∧ b = 2) →
  (∀ x : ℝ, abs (x - a) + abs (x + b) ≥ m^2 - 3 * m) →
  m ≤ (3 + Real.sqrt 21) / 2 :=
by
  sorry


end find_a_b_find_max_m_l195_195724


namespace square_area_from_diagonal_l195_195541

theorem square_area_from_diagonal (d : ℝ) (h_d : d = 12) : ∃ (A : ℝ), A = 72 :=
by
  -- we will use the given diagonal to derive the result
  sorry

end square_area_from_diagonal_l195_195541


namespace find_k_l195_195067

-- Defining the quadratic function
def quadratic (x k : ℝ) := x^2 + (2 * k + 1) * x + k^2 + 1

-- Condition 1: The roots are distinct, implies discriminant > 0
def discriminant_positive (k : ℝ) := (2 * k + 1)^2 - 4 * (k^2 + 1) > 0

-- Condition 2: Product of roots given as 5
def product_of_roots (k : ℝ) := k^2 + 1 = 5

-- Main theorem
theorem find_k (k : ℝ) (hk1 : discriminant_positive k) (hk2 : product_of_roots k) : k = 2 := by
  sorry

end find_k_l195_195067


namespace orthogonal_circles_l195_195594

theorem orthogonal_circles (R1 R2 d : ℝ) :
  (d^2 = R1^2 + R2^2) ↔ (d^2 = R1^2 + R2^2) :=
by sorry

end orthogonal_circles_l195_195594


namespace third_median_length_l195_195736

theorem third_median_length (m1 m2 area : ℝ) (h1 : m1 = 5) (h2 : m2 = 10) (h3 : area = 10 * Real.sqrt 10) : 
  ∃ m3 : ℝ, m3 = 3 * Real.sqrt 10 :=
by
  sorry

end third_median_length_l195_195736


namespace units_digit_fraction_l195_195844

-- Given conditions
def numerator : ℕ := 30 * 31 * 32 * 33 * 34 * 35
def denominator : ℕ := 1500
def simplified_fraction : ℕ := 2^5 * 3 * 31 * 33 * 17 * 7

-- Statement of the proof goal
theorem units_digit_fraction :
  (simplified_fraction) % 10 = 2 := by
  sorry

end units_digit_fraction_l195_195844


namespace percentage_less_than_l195_195006

theorem percentage_less_than (T F S : ℝ) 
  (hF : F = 0.70 * T) 
  (hS : S = 0.63 * T) : 
  ((T - S) / T) * 100 = 37 := 
by
  sorry

end percentage_less_than_l195_195006


namespace remainder_when_7n_divided_by_4_l195_195303

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l195_195303


namespace pascal_triangle_contains_prime_l195_195212

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l195_195212


namespace analytic_expression_on_1_2_l195_195804

noncomputable def f : ℝ → ℝ :=
  sorry

theorem analytic_expression_on_1_2 (x : ℝ) (h1 : 1 < x) (h2 : x < 2) :
  f x = Real.logb (1 / 2) (x - 1) :=
sorry

end analytic_expression_on_1_2_l195_195804


namespace inequality_condition_l195_195396

theorem inequality_condition (x y a : ℝ) (h1 : x < y) (h2 : a * x < a * y) : a > 0 :=
sorry

end inequality_condition_l195_195396


namespace min_dot_product_of_vectors_at_fixed_point_l195_195256

noncomputable def point := ℝ × ℝ

def on_ellipse (x y : ℝ) : Prop := 
  (x^2) / 36 + (y^2) / 9 = 1

def dot_product (p q : point) : ℝ := 
  p.1 * q.1 + p.2 * q.2

def vector_magnitude_squared (p : point) : ℝ := 
  p.1^2 + p.2^2

def KM (M : point) : point := 
  (M.1 - 2, M.2)

def NM (N M : point) : point := 
  (M.1 - N.1, M.2 - N.2)

def fixed_point_K : point := 
  (2, 0)

theorem min_dot_product_of_vectors_at_fixed_point (M N : point) 
  (hM_on_ellipse : on_ellipse M.1 M.2) 
  (hN_on_ellipse : on_ellipse N.1 N.2) 
  (h_orthogonal : dot_product (KM M) (KM N) = 0) : 
  ∃ (α : ℝ), dot_product (KM M) (NM N M) = 23 / 3 :=
sorry

end min_dot_product_of_vectors_at_fixed_point_l195_195256


namespace range_of_a_l195_195517

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem range_of_a {a : ℝ} (h : ∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) < 0) : 
  a > 1/7 ∧ a < 1/3 :=
sorry

end range_of_a_l195_195517


namespace market_value_of_stock_l195_195143

variable (face_value : ℝ) (annual_dividend yield : ℝ)

-- Given conditions:
def stock_four_percent := annual_dividend = 0.04 * face_value
def stock_yield_five_percent := yield = 0.05

-- Problem statement:
theorem market_value_of_stock (face_value := 100) (annual_dividend := 4) (yield := 0.05) 
  (h1 : stock_four_percent face_value annual_dividend) 
  (h2 : stock_yield_five_percent yield) : 
  (4 / 0.05) * 100 = 80 :=
by
  sorry

end market_value_of_stock_l195_195143


namespace carol_carrots_l195_195175

def mother_picked := 16
def good_carrots := 38
def bad_carrots := 7
def total_carrots := good_carrots + bad_carrots
def carol_picked : Nat := total_carrots - mother_picked

theorem carol_carrots : carol_picked = 29 := by
  sorry

end carol_carrots_l195_195175


namespace first_digit_after_decimal_correct_l195_195809

noncomputable def first_digit_after_decimal (n: ℕ) : ℕ :=
  if n % 2 = 0 then 9 else 4

theorem first_digit_after_decimal_correct (n : ℕ) :
  (first_digit_after_decimal n = 9 ↔ n % 2 = 0) ∧ (first_digit_after_decimal n = 4 ↔ n % 2 = 1) :=
by
  sorry

end first_digit_after_decimal_correct_l195_195809


namespace alcohol_water_ratio_l195_195288

theorem alcohol_water_ratio
  (V p q : ℝ)
  (hV : V > 0)
  (hp : p > 0)
  (hq : q > 0) :
  let total_alcohol := 3 * V * (p / (p + 1)) + V * (q / (q + 1))
  let total_water := 3 * V * (1 / (p + 1)) + V * (1 / (q + 1))
  total_alcohol / total_water = (3 * p * (q + 1) + q * (p + 1)) / (3 * (q + 1) + (p + 1)) :=
sorry

end alcohol_water_ratio_l195_195288


namespace prime_in_form_x_squared_plus_16y_squared_prime_in_form_4x_squared_plus_4xy_plus_5y_squared_l195_195523

theorem prime_in_form_x_squared_plus_16y_squared (p : ℕ) (hprime : Prime p) (h1 : p % 8 = 1) :
  ∃ x y : ℤ, p = x^2 + 16 * y^2 :=
by
  sorry

theorem prime_in_form_4x_squared_plus_4xy_plus_5y_squared (p : ℕ) (hprime : Prime p) (h5 : p % 8 = 5) :
  ∃ x y : ℤ, p = 4 * x^2 + 4 * x * y + 5 * y^2 :=
by
  sorry

end prime_in_form_x_squared_plus_16y_squared_prime_in_form_4x_squared_plus_4xy_plus_5y_squared_l195_195523


namespace total_dividends_received_l195_195473

theorem total_dividends_received
  (investment : ℝ)
  (share_price : ℝ)
  (nominal_value : ℝ)
  (dividend_rate_year1 : ℝ)
  (dividend_rate_year2 : ℝ)
  (dividend_rate_year3 : ℝ)
  (num_shares : ℝ)
  (total_dividends : ℝ) :
  investment = 14400 →
  share_price = 120 →
  nominal_value = 100 →
  dividend_rate_year1 = 0.07 →
  dividend_rate_year2 = 0.09 →
  dividend_rate_year3 = 0.06 →
  num_shares = investment / share_price → 
  total_dividends = (dividend_rate_year1 * nominal_value * num_shares) +
                    (dividend_rate_year2 * nominal_value * num_shares) +
                    (dividend_rate_year3 * nominal_value * num_shares) →
  total_dividends = 2640 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end total_dividends_received_l195_195473


namespace jason_commute_with_detour_l195_195799

theorem jason_commute_with_detour (d1 d2 d3 d4 d5 : ℝ) 
  (h1 : d1 = 4)     -- Distance from house to first store
  (h2 : d2 = 6)     -- Distance between first and second store
  (h3 : d3 = d2 + (2/3) * d2) -- Distance between second and third store without detour
  (h4 : d4 = 3)     -- Additional distance due to detour
  (h5 : d5 = d1)    -- Distance from third store to work
  : d1 + d2 + (d3 + d4) + d5 = 27 :=
by
  sorry

end jason_commute_with_detour_l195_195799


namespace total_hours_worked_l195_195576

theorem total_hours_worked
  (x : ℕ)
  (h1 : 5 * x = 55)
  : 2 * x + 3 * x + 5 * x = 110 :=
by 
  sorry

end total_hours_worked_l195_195576


namespace no_real_roots_of_quadratic_l195_195456

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem no_real_roots_of_quadratic (h : quadratic_discriminant 1 (-1) 1 < 0) :
  ¬ ∃ x : ℝ, x^2 - x + 1 = 0 :=
by
  sorry

end no_real_roots_of_quadratic_l195_195456


namespace necessary_but_not_sufficient_l195_195966

theorem necessary_but_not_sufficient (a b : ℕ) : 
  (a ≠ 1 ∨ b ≠ 2) → ¬ (a + b = 3) → ¬(a = 1 ∧ b = 2) ∧ ((a = 1 ∧ b = 2) → (a + b = 3)) := sorry

end necessary_but_not_sufficient_l195_195966


namespace expanded_figure_perimeter_l195_195961

def side_length : ℕ := 2
def bottom_row_squares : ℕ := 3
def total_squares : ℕ := 4

def perimeter (side_length : ℕ) (bottom_row_squares : ℕ) (total_squares: ℕ) : ℕ :=
  2 * side_length * (bottom_row_squares + 1)

theorem expanded_figure_perimeter : perimeter side_length bottom_row_squares total_squares = 20 :=
by
  sorry

end expanded_figure_perimeter_l195_195961


namespace reading_homework_pages_eq_three_l195_195816

-- Define the conditions
def pages_of_math_homework : ℕ := 7
def difference : ℕ := 4

-- Define what we need to prove
theorem reading_homework_pages_eq_three (x : ℕ) (h : x + difference = pages_of_math_homework) : x = 3 := by
  sorry

end reading_homework_pages_eq_three_l195_195816


namespace find_m_l195_195543

theorem find_m (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 - 2 * x + m) 
  (h2 : ∀ x ≥ (3 : ℝ), f x ≥ 1) : m = -2 := 
sorry

end find_m_l195_195543


namespace sam_age_l195_195747

theorem sam_age (drew_current_age : ℕ) (drew_future_age : ℕ) (sam_future_age : ℕ) : 
  (drew_current_age = 12) → 
  (drew_future_age = drew_current_age + 5) → 
  (sam_future_age = 3 * drew_future_age) → 
  (sam_future_age - 5 = 46) := 
by sorry

end sam_age_l195_195747


namespace turner_oldest_child_age_l195_195577

theorem turner_oldest_child_age (a b c : ℕ) (avg : ℕ) :
  (a = 6) → (b = 8) → (c = 11) → (avg = 9) → 
  (4 * avg = (a + b + c + x) → x = 11) :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end turner_oldest_child_age_l195_195577


namespace divisible_by_7_imp_coefficients_divisible_by_7_l195_195815

theorem divisible_by_7_imp_coefficients_divisible_by_7
  (a0 a1 a2 a3 a4 a5 a6 : ℤ)
  (h : ∀ x : ℤ, 7 ∣ (a6 * x^6 + a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0)) :
  7 ∣ a0 ∧ 7 ∣ a1 ∧ 7 ∣ a2 ∧ 7 ∣ a3 ∧ 7 ∣ a4 ∧ 7 ∣ a5 ∧ 7 ∣ a6 :=
sorry

end divisible_by_7_imp_coefficients_divisible_by_7_l195_195815


namespace triangle_side_ratio_l195_195904

theorem triangle_side_ratio (a b c : ℝ) (h1 : a + b ≤ 2 * c) (h2 : b + c ≤ 3 * a) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  2 / 3 < c / a ∧ c / a < 2 :=
by
  sorry

end triangle_side_ratio_l195_195904


namespace interval_for_rollers_l195_195164

noncomputable def interval_contains_probability (a σ : ℝ) (p : ℝ) : Prop :=
  ∃ δ : ℝ, 2 * CDF (Normal a σ) δ - 1 = p

theorem interval_for_rollers 
  (a : ℝ) (σ : ℝ) (p : ℝ) (hl: a = 10) (hs: σ = 0.1) (hp: p = 0.9973):
  interval_contains_probability a σ p → (9.7 < a ∧ a < 10.3) :=
sorry

end interval_for_rollers_l195_195164


namespace locus_of_M_equation_of_l_l195_195049
open Real

-- Step 1: Define the given circles
def circle_F1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle_F2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 36

-- Step 2: Define the condition of tangency for the moving circle M
def external_tangent_F1 (cx cy r : ℝ) : Prop := (cx + 2)^2 + cy^2 = (2 + r)^2
def internal_tangent_F2 (cx cy r : ℝ) : Prop := (cx - 2)^2 + cy^2 = (6 - r)^2

-- Step 4: Prove the locus C is an ellipse with the equation excluding x = -4
theorem locus_of_M (cx cy : ℝ) : 
  (∃ r : ℝ, external_tangent_F1 cx cy r ∧ internal_tangent_F2 cx cy r) ↔
  (cx ≠ -4 ∧ (cx^2) / 16 + (cy^2) / 12 = 1) :=
sorry

-- Step 5: Define the conditions for the midpoint of segment AB
def midpoint_Q (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = -1

-- Step 6: Prove the equation of line l
theorem equation_of_l (x1 y1 x2 y2 : ℝ) (h1 : midpoint_Q x1 y1 x2 y2) 
  (h2 : (x1^2 / 16 + y1^2 / 12 = 1) ∧ (x2^2 / 16 + y2^2 / 12 = 1)) :
  3 * (x1 - x2) - 2 * (y1 - y2) = 8 :=
sorry

end locus_of_M_equation_of_l_l195_195049


namespace intersection_lines_l195_195493

theorem intersection_lines (x y : ℝ) :
  (2 * x - y - 10 = 0) ∧ (3 * x + 4 * y - 4 = 0) → (x = 4) ∧ (y = -2) :=
by
  -- The proof is provided here
  sorry

end intersection_lines_l195_195493


namespace sum_abc_eq_neg_ten_thirds_l195_195420

variable (a b c d y : ℝ)

-- Define the conditions
def condition_1 : Prop := a + 2 = y
def condition_2 : Prop := b + 3 = y
def condition_3 : Prop := c + 4 = y
def condition_4 : Prop := d + 5 = y
def condition_5 : Prop := a + b + c + d + 6 = y

-- State the theorem
theorem sum_abc_eq_neg_ten_thirds
    (h1 : condition_1 a y)
    (h2 : condition_2 b y)
    (h3 : condition_3 c y)
    (h4 : condition_4 d y)
    (h5 : condition_5 a b c d y) :
    a + b + c + d = -10 / 3 :=
sorry

end sum_abc_eq_neg_ten_thirds_l195_195420


namespace middleton_sewers_capacity_l195_195587

theorem middleton_sewers_capacity:
  (total_runoff: ℤ) (runoff_per_hour: ℤ) (hours_per_day: ℤ) 
  (h1: total_runoff = 240000) 
  (h2: runoff_per_hour = 1000) 
  (h3: hours_per_day = 24) : 
  total_runoff / runoff_per_hour / hours_per_day = 10 := 
by sorry

end middleton_sewers_capacity_l195_195587


namespace minimum_cuts_to_unit_cubes_l195_195502

def cubes := List (ℕ × ℕ × ℕ)

def cube_cut (c : cubes) (n : ℕ) (dim : ℕ) : cubes :=
  sorry -- Function body not required for the statement

theorem minimum_cuts_to_unit_cubes (c : cubes) (s : ℕ) (dim : ℕ) :
  c = [(4,4,4)] ∧ s = 64 ∧ dim = 3 →
  ∃ (n : ℕ), n = 9 ∧
    (∀ cuts : cubes, cube_cut cuts n dim = [(1,1,1)]) :=
sorry

end minimum_cuts_to_unit_cubes_l195_195502


namespace sam_bought_cards_l195_195183

-- Define the initial number of baseball cards Dan had.
def dan_initial_cards : ℕ := 97

-- Define the number of baseball cards Dan has after selling some to Sam.
def dan_remaining_cards : ℕ := 82

-- Prove that the number of baseball cards Sam bought is 15.
theorem sam_bought_cards : (dan_initial_cards - dan_remaining_cards) = 15 :=
by
  sorry

end sam_bought_cards_l195_195183


namespace storm_deposit_eq_120_billion_gallons_l195_195351

theorem storm_deposit_eq_120_billion_gallons :
  ∀ (initial_content : ℝ) (full_percentage_pre_storm : ℝ) (full_percentage_post_storm : ℝ) (reservoir_capacity : ℝ),
  initial_content = 220 * 10^9 → 
  full_percentage_pre_storm = 0.55 →
  full_percentage_post_storm = 0.85 →
  reservoir_capacity = initial_content / full_percentage_pre_storm →
  (full_percentage_post_storm * reservoir_capacity - initial_content) = 120 * 10^9 :=
by
  intro initial_content full_percentage_pre_storm full_percentage_post_storm reservoir_capacity
  intros h_initial_content h_pre_storm h_post_storm h_capacity
  sorry

end storm_deposit_eq_120_billion_gallons_l195_195351


namespace solve_trig_eq_l195_195822

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  (x = (π / 12) + 2 * k * π ∨
   x = (7 * π / 12) + 2 * k * π ∨
   x = (7 * π / 6) + 2 * k * π ∨
   x = -(5 * π / 6) + 2 * k * π) →
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 :=
sorry

end solve_trig_eq_l195_195822


namespace sin_2phi_l195_195774

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l195_195774


namespace colonization_combinations_l195_195920

/-- Number of different combinations of planets that can be colonized using 15 units. --/
theorem colonization_combinations (units : ℕ) (e_earth : ℕ) (m_mars : ℕ) : 
  units = 15 ∧ e_earth = 5 ∧ m_mars = 8 → 
  ∑ (a : ℕ) in {0, 1, 2, 3, 4, 5}, ∑ (b : ℕ) in {0, 1, 2, 3, 4, 5, 6, 7, 8}, 
  if 2 * a + b = 15 then (Nat.choose e_earth a * Nat.choose m_mars b) else 0 = 96 :=
begin
  sorry
end

end colonization_combinations_l195_195920


namespace sin_double_angle_solution_l195_195789

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l195_195789


namespace smallest_positive_integer_n_l195_195986

def contains_digit_9 (n : ℕ) : Prop := 
  ∃ m : ℕ, (10^m) ∣ n ∧ (n / 10^m) % 10 = 9

theorem smallest_positive_integer_n :
  ∃ n : ℕ, (∀ k : ℕ, k > 0 ∧ k < n → 
  (∃ a b : ℕ, k = 2^a * 5^b * 3) ∧ contains_digit_9 k ∧ (k % 3 = 0))
  → n = 90 :=
sorry

end smallest_positive_integer_n_l195_195986


namespace min_triangle_area_l195_195051

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1
noncomputable def circle_with_diameter_passing_origin (A B : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  let d := (A.1 - B.1)^2 + (A.2 - B.2)^2
  let center := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  center.1^2 + center.2^2 = d / 4

theorem min_triangle_area (A B : ℝ × ℝ)
    (hA : hyperbola A.1 A.2)
    (hB : hyperbola B.1 B.2)
    (hc : circle_with_diameter_passing_origin A B) : 
    ∃ (S : ℝ), S = 2 :=
sorry

end min_triangle_area_l195_195051


namespace solve_inequality_l195_195103

theorem solve_inequality (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  ( if 0 ≤ a ∧ a < 1 / 2 then (x > a ∧ x < 1 - a) else 
    if a = 1 / 2 then false else 
    if 1 / 2 < a ∧ a ≤ 1 then (x > 1 - a ∧ x < a) else false ) ↔ ((x - a) * (x + a - 1) < 0) :=
by
  sorry

end solve_inequality_l195_195103


namespace cost_of_pencil_l195_195472

theorem cost_of_pencil (s n c : ℕ) (h_majority : s > 15) (h_pencils : n > 1) (h_cost : c > n)
  (h_total_cost : s * c * n = 1771) : c = 11 :=
sorry

end cost_of_pencil_l195_195472


namespace cos_of_angle_between_lines_l195_195153

noncomputable def cosTheta (a b : ℝ × ℝ) : ℝ :=
  let dotProduct := a.1 * b.1 + a.2 * b.2
  let magA := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let magB := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dotProduct / (magA * magB)

theorem cos_of_angle_between_lines :
  cosTheta (3, 4) (1, 3) = 3 / Real.sqrt 10 :=
by
  sorry

end cos_of_angle_between_lines_l195_195153


namespace geometry_problem_l195_195432

-- Definitions for geometrical entities
variable {Point : Type} -- type representing points

variable (Line : Type) -- type representing lines
variable (Plane : Type) -- type representing planes

-- Parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop) 
variable (perpendicular : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Given entities
variables (m : Line) (n : Line) (α : Plane) (β : Plane)

-- Given conditions
axiom condition1 : perpendicular α β
axiom condition2 : perpendicular_line_plane m β
axiom condition3 : ¬ contained_in m α

-- Statement of the problem in Lean 4
theorem geometry_problem : parallel m α :=
by
  -- proof will involve using the axioms and definitions
  sorry

end geometry_problem_l195_195432


namespace series_solution_eq_l195_195186

theorem series_solution_eq (x : ℝ) 
  (h : (∃ a : ℕ → ℝ, (∀ n, a n = 1 + 6 * n) ∧ (∑' n, a n * x^n = 100))) :
  x = 23/25 ∨ x = 1/50 :=
sorry

end series_solution_eq_l195_195186


namespace pascal_triangle_47_number_of_rows_containing_47_l195_195225

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l195_195225


namespace pet_fee_is_120_l195_195087

noncomputable def daily_rate : ℝ := 125.00
noncomputable def rental_days : ℕ := 14
noncomputable def service_fee_rate : ℝ := 0.20
noncomputable def security_deposit : ℝ := 1110.00
noncomputable def security_deposit_rate : ℝ := 0.50

theorem pet_fee_is_120 :
  let total_stay_cost := daily_rate * rental_days
  let service_fee := service_fee_rate * total_stay_cost
  let total_before_pet_fee := total_stay_cost + service_fee
  let entire_bill := security_deposit / security_deposit_rate
  let pet_fee := entire_bill - total_before_pet_fee
  pet_fee = 120 := by
  sorry

end pet_fee_is_120_l195_195087


namespace num_double_yolk_eggs_l195_195339

noncomputable def double_yolk_eggs (total_eggs total_yolks : ℕ) (double_yolk_contrib : ℕ) : ℕ :=
(total_yolks - total_eggs + double_yolk_contrib) / double_yolk_contrib

theorem num_double_yolk_eggs (total_eggs total_yolks double_yolk_contrib expected : ℕ)
    (h1 : total_eggs = 12)
    (h2 : total_yolks = 17)
    (h3 : double_yolk_contrib = 2)
    (h4 : expected = 5) :
  double_yolk_eggs total_eggs total_yolks double_yolk_contrib = expected :=
by
  rw [h1, h2, h3, h4]
  dsimp [double_yolk_eggs]
  norm_num
  sorry

end num_double_yolk_eggs_l195_195339


namespace min_guesses_correct_l195_195460

noncomputable def min_guesses (n k : ℕ) (h : n > k) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h : n > k) :
  min_guesses n k h = if n = 2 * k then 2 else 1 :=
by
  sorry

end min_guesses_correct_l195_195460


namespace count_total_coins_l195_195450

theorem count_total_coins (quarters nickels : Nat) (h₁ : quarters = 4) (h₂ : nickels = 8) : quarters + nickels = 12 :=
by sorry

end count_total_coins_l195_195450


namespace find_A_l195_195854

theorem find_A :
  ∃ A B : ℕ, A < 10 ∧ B < 10 ∧ 5 * 100 + A * 10 + 8 - (B * 100 + 1 * 10 + 4) = 364 ∧ A = 7 :=
by
  sorry

end find_A_l195_195854


namespace cos_A_minus_B_eq_nine_eighths_l195_195054

theorem cos_A_minus_B_eq_nine_eighths (A B : ℝ)
  (h1 : Real.sin A + Real.sin B = 1 / 2)
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9 / 8 := 
by
  sorry

end cos_A_minus_B_eq_nine_eighths_l195_195054


namespace student_A_selection_probability_l195_195865

def probability_student_A_selected (total_students : ℕ) (students_removed : ℕ) (representatives : ℕ) : ℚ :=
  representatives / (total_students : ℚ)

theorem student_A_selection_probability :
  probability_student_A_selected 752 2 5 = 5 / 752 :=
by
  sorry

end student_A_selection_probability_l195_195865


namespace distinguishable_balls_in_indistinguishable_boxes_l195_195534

theorem distinguishable_balls_in_indistinguishable_boxes :
  let num_distinguishable_balls := 6
  let num_indistinguishable_boxes := 3
  -- different ways to distribute balls across boxes summarized
  let ways := ∑([{6, 0, 0} ++ {5, 1, 0} ++ {4, 2, 0} ++ {4, 1, 1} ++ {3, 3, 0} ++ {3, 2, 1} ++ {2, 2, 2}], sum)
  -- corresponding ways to remove permutation of distinguishable ball cases
  ways = 222 := sorry

end distinguishable_balls_in_indistinguishable_boxes_l195_195534


namespace at_least_one_pass_l195_195123

variable (n : ℕ) (p : ℝ)

theorem at_least_one_pass (h_p_range : 0 < p ∧ p < 1) :
  (1 - (1 - p) ^ n) = 1 - (1 - p) ^ n :=
sorry

end at_least_one_pass_l195_195123


namespace find_A_coordinates_l195_195254

-- Given conditions
variable (B : (ℝ × ℝ)) (hB1 : B = (1, 2))

-- Definitions to translate problem conditions into Lean
def symmetric_y (P B : ℝ × ℝ) : Prop :=
  P.1 = -B.1 ∧ P.2 = B.2

def symmetric_x (A P : ℝ × ℝ) : Prop :=
  A.1 = P.1 ∧ A.2 = -P.2

-- Theorem statement
theorem find_A_coordinates (A P B : ℝ × ℝ) (hB1 : B = (1, 2))
    (h_symm_y: symmetric_y P B) (h_symm_x: symmetric_x A P) : 
    A = (-1, -2) :=
by
  sorry

end find_A_coordinates_l195_195254


namespace find_number_l195_195340

-- Define given numbers
def a : ℕ := 555
def b : ℕ := 445

-- Define given conditions
def sum : ℕ := a + b
def difference : ℕ := a - b
def quotient : ℕ := 2 * difference
def remainder : ℕ := 30

-- Define the number we're looking for
def number := sum * quotient + remainder

-- The theorem to prove
theorem find_number : number = 220030 := by
  -- Use the let expressions to simplify the calculation for clarity
  let sum := a + b
  let difference := a - b
  let quotient := 2 * difference
  let number := sum * quotient + remainder
  show number = 220030
  -- Placeholder for proof
  sorry

end find_number_l195_195340


namespace pascal_triangle_47_number_of_rows_containing_47_l195_195226

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l195_195226


namespace calculate_expression_l195_195873

theorem calculate_expression :
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5 / 4 :=
by
  sorry

end calculate_expression_l195_195873


namespace fourth_term_of_gp_is_negative_10_point_42_l195_195538

theorem fourth_term_of_gp_is_negative_10_point_42 (x : ℝ) 
  (h : ∃ r : ℝ, r * (5 * x + 5) = (3 * x + 3) * ((3 * x + 3) / x)) :
  r * (5 * x + 5) * ((3 * x + 3) / x) * ((3 * x + 3) / x) = -10.42 :=
by
  sorry

end fourth_term_of_gp_is_negative_10_point_42_l195_195538


namespace price_returns_to_initial_l195_195930

theorem price_returns_to_initial {P₀ P₁ P₂ P₃ P₄ : ℝ} (y : ℝ) (h₁ : P₀ = 100)
  (h₂ : P₁ = P₀ * 1.30) (h₃ : P₂ = P₁ * 0.70) (h₄ : P₃ = P₂ * 1.40) 
  (h₅ : P₄ = P₃ * (1 - y / 100)) : P₄ = P₀ → y = 22 :=
by
  sorry

end price_returns_to_initial_l195_195930


namespace math_problem_l195_195497

theorem math_problem
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : x = z * (1 / y)) : 
  (x - z / x) * (y + 1 / (z * y)) = (x^4 - z^3 + x^2 * (z^2 - z)) / (z * x^2) :=
by
  sorry

end math_problem_l195_195497


namespace solution_eq_l195_195516

theorem solution_eq (a x : ℚ) :
  (2 * (x - 2 * (x - a / 4)) = 3 * x) ∧ ((x + a) / 9 - (1 - 3 * x) / 12 = 1) → 
  a = 65 / 11 ∧ x = 13 / 11 :=
by
  sorry

end solution_eq_l195_195516


namespace total_packs_of_groceries_l195_195812

-- Definitions based on conditions
def packs_of_cookies : Nat := 4
def packs_of_cake : Nat := 22
def packs_of_chocolate : Nat := 16

-- The proof statement
theorem total_packs_of_groceries : packs_of_cookies + packs_of_cake + packs_of_chocolate = 42 :=
by
  -- Proof skipped using sorry
  sorry

end total_packs_of_groceries_l195_195812


namespace distance_from_Q_to_EG_l195_195570

noncomputable def distance_to_line : ℝ :=
  let E := (0, 5)
  let F := (5, 5)
  let G := (5, 0)
  let H := (0, 0)
  let N := (2.5, 0)
  let Q := (25 / 7, 10 / 7)
  let line_y := 5
  let distance := abs (line_y - Q.2)
  distance

theorem distance_from_Q_to_EG : distance_to_line = 25 / 7 :=
by
  sorry

end distance_from_Q_to_EG_l195_195570


namespace union_M_N_eq_U_l195_195898

def U : Set Nat := {2, 3, 4, 5, 6, 7}
def M : Set Nat := {3, 4, 5, 7}
def N : Set Nat := {2, 4, 5, 6}

theorem union_M_N_eq_U : M ∪ N = U := 
by {
  -- Proof would go here
  sorry
}

end union_M_N_eq_U_l195_195898


namespace rationalize_denominator_l195_195817

theorem rationalize_denominator : 
  let a := 32
  let b := 8
  let c := 2
  let d := 4
  (a / (c * Real.sqrt c) + b / (d * Real.sqrt c)) = (9 * Real.sqrt c) :=
by
  sorry

end rationalize_denominator_l195_195817


namespace probability_sum_eq_erika_age_l195_195889

def conditions : Prop :=
  let fair_coin := {15, 20}
  ∧ let die_faces := {1, 2, 3, 4, 5, 6}
  ∧ let erika_age := 16 in
  true -- conditions are contextual and inherently true in this context

theorem probability_sum_eq_erika_age : 
  ∀ (coin_flip : ℕ) (die_roll : ℕ), 
  coin_flip ∈ {15, 20} → 
  die_roll ∈ {1, 2, 3, 4, 5, 6} → 
  (coin_flip + die_roll = 16) → 
  (1 / 2) * (1 / 6) = 1 / 12 := 
by
  intros coin_flip die_roll coin_flip_in coin_flip_in die_roll_in sum_eq
  sorry -- proof goes here

end probability_sum_eq_erika_age_l195_195889


namespace increase_result_l195_195641

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l195_195641


namespace age_of_b_l195_195848

-- Define the conditions as per the problem statement
variables (A B C D E : ℚ)

axiom cond1 : A = B + 2
axiom cond2 : B = 2 * C
axiom cond3 : D = A - 3
axiom cond4 : E = D / 2 + 3
axiom cond5 : A + B + C + D + E = 70

theorem age_of_b : B = 16.625 :=
by {
  -- Placeholder for the proof
  sorry
}

end age_of_b_l195_195848


namespace problem_statement_l195_195113

theorem problem_statement (a b : ℝ) (h1 : a^3 - b^3 = 2) (h2 : a^5 - b^5 ≥ 4) : a^2 + b^2 ≥ 2 := 
sorry

end problem_statement_l195_195113


namespace triangle_largest_angle_l195_195075

theorem triangle_largest_angle 
  (a1 a2 a3 : ℝ) 
  (h_sum : a1 + a2 + a3 = 180)
  (h_arith_seq : 2 * a2 = a1 + a3)
  (h_one_angle : a1 = 28) : 
  max a1 (max a2 a3) = 92 := 
by
  sorry

end triangle_largest_angle_l195_195075


namespace exists_X_Y_sum_not_in_third_subset_l195_195077

open Nat Set

theorem exists_X_Y_sum_not_in_third_subset :
  ∀ (M_1 M_2 M_3 : Set ℕ), 
  Disjoint M_1 M_2 ∧ Disjoint M_2 M_3 ∧ Disjoint M_1 M_3 → 
  ∃ (X Y : ℕ), (X ∈ M_1 ∪ M_2 ∪ M_3) ∧ (Y ∈ M_1 ∪ M_2 ∪ M_3) ∧  
  (X ∈ M_1 → Y ∈ M_2 ∨ Y ∈ M_3) ∧
  (X ∈ M_2 → Y ∈ M_1 ∨ Y ∈ M_3) ∧
  (X ∈ M_3 → Y ∈ M_1 ∨ Y ∈ M_2) ∧
  (X + Y ∉ M_3) :=
by
  intros M_1 M_2 M_3 disj
  sorry

end exists_X_Y_sum_not_in_third_subset_l195_195077


namespace find_frac_sin_cos_l195_195385

theorem find_frac_sin_cos (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin (3 * Real.pi / 2 + α)) : 
  (Real.sin α - 4 * Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = -1 / 6 :=
by
  sorry

end find_frac_sin_cos_l195_195385


namespace Julia_played_with_kids_on_Monday_l195_195415

theorem Julia_played_with_kids_on_Monday (kids_tuesday : ℕ) (more_kids_monday : ℕ) :
  kids_tuesday = 14 → more_kids_monday = 8 → (kids_tuesday + more_kids_monday = 22) :=
by
  sorry

end Julia_played_with_kids_on_Monday_l195_195415


namespace number_of_sampled_medium_stores_is_five_l195_195929

-- Definitions based on the conditions
def total_stores : ℕ := 300
def large_stores : ℕ := 30
def medium_stores : ℕ := 75
def small_stores : ℕ := 195
def sample_size : ℕ := 20

-- Proportion calculation function
def medium_store_proportion := (medium_stores : ℚ) / (total_stores : ℚ)

-- Sampled medium stores calculation
def sampled_medium_stores := medium_store_proportion * (sample_size : ℚ)

-- Theorem stating the number of medium stores drawn using stratified sampling
theorem number_of_sampled_medium_stores_is_five :
  sampled_medium_stores = 5 := 
by 
  sorry

end number_of_sampled_medium_stores_is_five_l195_195929


namespace intersection_M_N_l195_195417

open Set

variable (x y : ℝ)

theorem intersection_M_N :
  let M := {x | x < 1}
  let N := {y | ∃ x, x < 1 ∧ y = 1 - 2 * x}
  M ∩ N = ∅ := sorry

end intersection_M_N_l195_195417


namespace maximum_value_of_expression_is_4_l195_195397

noncomputable def maximimum_integer_value (x : ℝ) : ℝ :=
    (5 * x^2 + 10 * x + 12) / (5 * x^2 + 10 * x + 2)

theorem maximum_value_of_expression_is_4 :
    ∃ x : ℝ, ∀ y : ℝ, maximimum_integer_value y ≤ 4 ∧ maximimum_integer_value x = 4 := 
by 
  -- Proof omitted for now
  sorry

end maximum_value_of_expression_is_4_l195_195397


namespace root_product_is_27_l195_195480

open Real

noncomputable def cube_root (x : ℝ) := x ^ (1 / 3 : ℝ)
noncomputable def fourth_root (x : ℝ) := x ^ (1 / 4 : ℝ)
noncomputable def square_root (x : ℝ) := x ^ (1 / 2 : ℝ)

theorem root_product_is_27 : 
  (cube_root 27) * (fourth_root 81) * (square_root 9) = 27 := 
by
  sorry

end root_product_is_27_l195_195480


namespace probability_roll_differs_by_three_on_two_eight_sided_dies_l195_195163

theorem probability_roll_differs_by_three_on_two_eight_sided_dies : 
  let S := { (x: ℕ, y: ℕ) | 1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8 } in -- sample space
  let E := { (x: ℕ, y: ℕ) | 1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8 ∧ (x = y + 3 ∨ y = x + 3) } in -- event of interest
  ((E.card : ℚ) / S.card) = 1 / 8 := 
by
  sorry

end probability_roll_differs_by_three_on_two_eight_sided_dies_l195_195163


namespace cos_angles_difference_cos_angles_sum_l195_195461

-- Part (a)
theorem cos_angles_difference: 
  (Real.cos (36 * Real.pi / 180) - Real.cos (72 * Real.pi / 180) = 1 / 2) :=
sorry

-- Part (b)
theorem cos_angles_sum: 
  (Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7) = 1 / 2) :=
sorry

end cos_angles_difference_cos_angles_sum_l195_195461


namespace increase_80_by_150_percent_l195_195660

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l195_195660


namespace certain_number_is_50_l195_195331

theorem certain_number_is_50 (x : ℝ) (h : 0.6 * x = 0.42 * 30 + 17.4) : x = 50 :=
by
  sorry

end certain_number_is_50_l195_195331


namespace max_sum_of_factors_l195_195411

theorem max_sum_of_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C)
  (h4 : A * B * C = 3003) : A + B + C ≤ 117 :=
sorry

end max_sum_of_factors_l195_195411


namespace tip_percentage_l195_195262

def julie_food_cost : ℝ := 10
def letitia_food_cost : ℝ := 20
def anton_food_cost : ℝ := 30
def julie_tip : ℝ := 4
def letitia_tip : ℝ := 4
def anton_tip : ℝ := 4

theorem tip_percentage : 
  (julie_tip + letitia_tip + anton_tip) / (julie_food_cost + letitia_food_cost + anton_food_cost) * 100 = 20 :=
by
  sorry

end tip_percentage_l195_195262


namespace ratio_of_speeds_correct_l195_195851

noncomputable def ratio_speeds_proof_problem : Prop :=
  ∃ (v_A v_B : ℝ),
    (∀ t : ℝ, 0 ≤ t ∧ t = 3 → 3 * v_A = abs (-800 + 3 * v_B)) ∧
    (∀ t : ℝ, 0 ≤ t ∧ t = 15 → 15 * v_A = abs (-800 + 15 * v_B)) ∧
    (3 * 15 * v_A / (15 * v_B) = 3 / 4)

theorem ratio_of_speeds_correct : ratio_speeds_proof_problem :=
sorry

end ratio_of_speeds_correct_l195_195851


namespace increase_by_150_percent_l195_195661

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l195_195661


namespace find_a_and_b_l195_195277

theorem find_a_and_b (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = x^3 - a * x^2 - b * x + a^2) →
  f 1 = 10 →
  deriv f 1 = 0 →
  (a = -4 ∧ b = 11) :=
by
  intros hf hf1 hderiv
  sorry

end find_a_and_b_l195_195277


namespace chengdu_gdp_scientific_notation_l195_195257

theorem chengdu_gdp_scientific_notation :
  15000 = 1.5 * 10^4 :=
sorry

end chengdu_gdp_scientific_notation_l195_195257


namespace unique_triple_l195_195187

theorem unique_triple (x y p : ℕ) (hx : 0 < x) (hy : 0 < y) (hp : Nat.Prime p) (h1 : p = x^2 + 1) (h2 : 2 * p^2 = y^2 + 1) :
  (x, y, p) = (2, 7, 5) :=
sorry

end unique_triple_l195_195187


namespace minimum_value_of_function_l195_195831

noncomputable def function_y (x : ℝ) : ℝ := 1 / (Real.sqrt (x - x^2))

theorem minimum_value_of_function : (∀ x : ℝ, 0 < x ∧ x < 1 → function_y x ≥ 2) ∧ (∃ x : ℝ, 0 < x ∧ x < 1 ∧ function_y x = 2) :=
by
  sorry

end minimum_value_of_function_l195_195831


namespace arithmetic_sequence_an_12_l195_195255

theorem arithmetic_sequence_an_12 {a : ℕ → ℝ} (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a3 : a 3 = 9)
  (h_a6 : a 6 = 15) :
  a 12 = 27 := 
sorry

end arithmetic_sequence_an_12_l195_195255


namespace range_of_a_l195_195909

theorem range_of_a (p q : Set ℝ) (a : ℝ) (h1 : ∀ x, 2 * x^2 - 3 * x + 1 ≤ 0 → x ∈ p) 
                             (h2 : ∀ x, x^2 - (2 * a + 1) * x + a^2 + a ≤ 0 → x ∈ q)
                             (h3 : ∀ x, p x → q x ∧ ∃ x, ¬p x ∧ q x) : 
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l195_195909


namespace sphere_radius_l195_195978

theorem sphere_radius {r1 r2 : ℝ} (w1 w2 : ℝ) (S : ℝ → ℝ) 
  (h1 : S r1 = 4 * Real.pi * r1^2)
  (h2 : S r2 = 4 * Real.pi * r2^2)
  (w_s1 : w1 = 8)
  (w_s2 : w2 = 32)
  (r2_val : r2 = 0.3)
  (prop : ∀ r, w_s2 = w1 * S r2 / S r1 → w2 = w1 * S r2 / S r1 ) :
  r1 = 0.15 :=
by sorry

end sphere_radius_l195_195978


namespace increase_by_150_percent_l195_195667

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l195_195667


namespace gen_term_of_a_n_sum_of_first_n_b_n_l195_195386

-- We define the sequence $\{a_n\}$ and $\{b_n\}$ given the conditions and state the problem
def a_n (n : ℕ) := 3 * n - 1
def b_n : ℕ → ℚ 
| 0       := 1 -- 0-indexed version for convenience
| 1       := 1/3
| (n + 1) := (n * b_n n - (3 * n - 1) * b_n (n + 1)) / (3 * (n - 1))

-- Assume the relationship holds for $a_n$ and $b_n$
axiom a_n_b_n_condition : ∀ n : ℕ, a_n n * b_n (n + 1) + b_n (n + 1) = n * b_n n

-- Theorem for the general term of $\{a_n\}$
theorem gen_term_of_a_n : ∀ n : ℕ, a_n n = 3 * n - 1 := sorry

-- Theorem for the sum of the first $n$ terms of $\{b_n\}$
theorem sum_of_first_n_b_n (n : ℕ) : (Σ i in finset.range n, b_n i) = (3/2) - (1/2 * 3 ^ (n - 1)) := sorry

end gen_term_of_a_n_sum_of_first_n_b_n_l195_195386


namespace stratified_sampling_pines_l195_195338

def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

theorem stratified_sampling_pines :
  sample_size * pine_saplings / total_saplings = 20 := by
  sorry

end stratified_sampling_pines_l195_195338


namespace school_spent_440_l195_195011

-- Definition based on conditions listed in part a)
def cost_of_pencils (cartons_pencils : ℕ) (boxes_per_carton_pencils : ℕ) (cost_per_box_pencils : ℕ) : ℕ := 
  cartons_pencils * boxes_per_carton_pencils * cost_per_box_pencils

def cost_of_markers (cartons_markers : ℕ) (cost_per_carton_markers : ℕ) : ℕ := 
  cartons_markers * cost_per_carton_markers

noncomputable def total_cost (cartons_pencils cartons_markers boxes_per_carton_pencils cost_per_box_pencils cost_per_carton_markers : ℕ) : ℕ := 
  cost_of_pencils cartons_pencils boxes_per_carton_pencils cost_per_box_pencils + 
  cost_of_markers cartons_markers cost_per_carton_markers

-- Theorem statement to prove the total cost is $440 given the conditions
theorem school_spent_440 : total_cost 20 10 10 2 4 = 440 := by 
  sorry

end school_spent_440_l195_195011


namespace complement_intersection_l195_195714

def U : Set ℝ := Set.univ
def M : Set ℝ := {y | 0 ≤ y ∧ y ≤ 2}
def N : Set ℝ := {x | (x < -3) ∨ (x > 0)}

theorem complement_intersection :
  (Set.univ \ M) ∩ N = {x | x < -3 ∨ x > 2} :=
by
  sorry

end complement_intersection_l195_195714


namespace factor_transformation_option_C_l195_195276

theorem factor_transformation_option_C (y : ℝ) : 
  4 * y^2 - 4 * y + 1 = (2 * y - 1)^2 :=
sorry

end factor_transformation_option_C_l195_195276


namespace coordinate_P_condition_1_coordinate_P_condition_2_coordinate_P_condition_3_l195_195381

-- Definition of the conditions
def condition_1 (Px: ℝ) (Py: ℝ) : Prop := Px = 0

def condition_2 (Px: ℝ) (Py: ℝ) : Prop := Py = Px + 3

def condition_3 (Px: ℝ) (Py: ℝ) : Prop := 
  abs Py = 2 ∧ Px > 0 ∧ Py < 0

-- Proof problem for condition 1
theorem coordinate_P_condition_1 : ∃ (Px Py: ℝ), condition_1 Px Py ∧ Px = 0 ∧ Py = -7 := 
  sorry

-- Proof problem for condition 2
theorem coordinate_P_condition_2 : ∃ (Px Py: ℝ), condition_2 Px Py ∧ Px = 10 ∧ Py = 13 :=
  sorry

-- Proof problem for condition 3
theorem coordinate_P_condition_3 : ∃ (Px Py: ℝ), condition_3 Px Py ∧ Px = 5/2 ∧ Py = -2 :=
  sorry

end coordinate_P_condition_1_coordinate_P_condition_2_coordinate_P_condition_3_l195_195381


namespace num_white_squares_in_24th_row_l195_195333

-- Define the function that calculates the total number of squares in the nth row
def total_squares (n : ℕ) : ℕ := 1 + 2 * (n - 1)

-- Define the function that calculates the number of white squares in the nth row
def white_squares (n : ℕ) : ℕ := (total_squares n - 2) / 2

-- Problem statement for the Lean 4 theorem
theorem num_white_squares_in_24th_row : white_squares 24 = 23 :=
by {
  -- Lean proof generation will be placed here
  sorry
}

end num_white_squares_in_24th_row_l195_195333


namespace expected_value_of_problems_l195_195102

-- Define the setup
def num_pairs : ℕ := 5
def num_shoes : ℕ := num_pairs * 2
def prob_same_color : ℚ := 1 / (num_shoes - 1)
def days : ℕ := 5

-- Define the expected value calculation using linearity of expectation
def expected_problems_per_day : ℚ := prob_same_color
def expected_total_problems : ℚ := days * expected_problems_per_day

-- Prove the expected number of practice problems Sandra gets to do over 5 days
theorem expected_value_of_problems : expected_total_problems = 5 / 9 := 
by 
  rw [expected_total_problems, expected_problems_per_day, prob_same_color]
  norm_num
  sorry

end expected_value_of_problems_l195_195102


namespace increase_result_l195_195645

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l195_195645


namespace distance_to_convenience_store_l195_195021

def distance_work := 6
def days_work := 5
def distance_dog_walk := 2
def times_dog_walk := 2
def days_week := 7
def distance_friend_house := 1
def times_friend_visit := 1
def total_miles := 95
def trips_convenience_store := 2

theorem distance_to_convenience_store :
  ∃ x : ℝ,
    (distance_work * 2 * days_work) +
    (distance_dog_walk * times_dog_walk * days_week) +
    (distance_friend_house * 2 * times_friend_visit) +
    (x * trips_convenience_store) = total_miles
    → x = 2.5 :=
by
  sorry

end distance_to_convenience_store_l195_195021


namespace num_triangles_with_perimeter_20_l195_195895

theorem num_triangles_with_perimeter_20 : 
  ∃ (triangles : List (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ triangles → a + b + c = 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ b + c > a ∧ c + a > b) ∧ 
    triangles.length = 8 :=
sorry

end num_triangles_with_perimeter_20_l195_195895


namespace average_infected_per_round_is_nine_l195_195871

theorem average_infected_per_round_is_nine (x : ℝ) :
  1 + x + x * (1 + x) = 100 → x = 9 :=
by {
  sorry
}

end average_infected_per_round_is_nine_l195_195871


namespace number_of_BMWs_sold_l195_195469

theorem number_of_BMWs_sold (total_cars : ℕ) (Audi_percent Toyota_percent Acura_percent Ford_percent : ℝ)
  (h_total_cars : total_cars = 250) 
  (h_percentages : Audi_percent = 0.10 ∧ Toyota_percent = 0.20 ∧ Acura_percent = 0.15 ∧ Ford_percent = 0.25) :
  ∃ (BMWs_sold : ℕ), BMWs_sold = 75 := 
by
  sorry

end number_of_BMWs_sold_l195_195469


namespace island_count_l195_195754

-- Defining the conditions
def lakes := 7
def canals := 10

-- Euler's formula for connected planar graph
def euler_characteristic (V E F : ℕ) := V - E + F = 2

-- Determine the number of faces using Euler's formula
def faces (V E : ℕ) :=
  let F := V - E + 2
  F

-- The number of islands is the number of faces minus one for the outer face
def number_of_islands (F : ℕ) :=
  F - 1

-- The given proof problem to be converted to Lean
theorem island_count :
  number_of_islands (faces lakes canals) = 4 :=
by
  unfold lakes canals faces number_of_islands
  sorry

end island_count_l195_195754


namespace original_length_equals_13_l195_195796

-- Definitions based on conditions
def original_width := 18
def increased_length (x : ℕ) := x + 2
def increased_width := 20

-- Total area condition
def total_area (x : ℕ) := 
  4 * ((increased_length x) * increased_width) + 2 * ((increased_length x) * increased_width)

theorem original_length_equals_13 (x : ℕ) (h : total_area x = 1800) : x = 13 := 
by
  sorry

end original_length_equals_13_l195_195796


namespace circles_C1_C2_intersect_C1_C2_l195_195514

noncomputable def center1 : (ℝ × ℝ) := (5, 3)
noncomputable def radius1 : ℝ := 3

noncomputable def center2 : (ℝ × ℝ) := (2, -1)
noncomputable def radius2 : ℝ := Real.sqrt 14

noncomputable def distance : ℝ := Real.sqrt ((5 - 2)^2 + (3 + 1)^2)

def circles_intersect : Prop :=
  radius2 - radius1 < distance ∧ distance < radius2 + radius1

theorem circles_C1_C2_intersect_C1_C2 : circles_intersect :=
by
  -- The proof of this theorem is to be worked out using the given conditions and steps.
  sorry

end circles_C1_C2_intersect_C1_C2_l195_195514


namespace milk_price_per_liter_l195_195109

theorem milk_price_per_liter (M : ℝ) 
  (price_fruit_per_kg : ℝ) (price_each_fruit_kg_eq_2: price_fruit_per_kg = 2)
  (milk_liters_per_batch : ℝ) (milk_liters_per_batch_eq_10: milk_liters_per_batch = 10)
  (fruit_kg_per_batch : ℝ) (fruit_kg_per_batch_eq_3 : fruit_kg_per_batch = 3)
  (cost_three_batches : ℝ) (cost_three_batches_eq_63: cost_three_batches = 63) :
  M = 1.5 :=
by
  sorry

end milk_price_per_liter_l195_195109


namespace percentage_return_l195_195751

theorem percentage_return (income investment : ℝ) (h_income : income = 680) (h_investment : investment = 8160) :
  (income / investment) * 100 = 8.33 :=
by
  rw [h_income, h_investment]
  -- The rest of the proof is omitted.
  sorry

end percentage_return_l195_195751


namespace subtraction_example_l195_195875

theorem subtraction_example : 34.256 - 12.932 - 1.324 = 20.000 := 
by
  sorry

end subtraction_example_l195_195875


namespace pascal_row_contains_prime_47_l195_195229

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l195_195229


namespace original_price_l195_195341

theorem original_price (P : ℝ) (S : ℝ) (h1 : S = 1.3 * P) (h2 : S = P + 650) : P = 2166.67 :=
by
  sorry

end original_price_l195_195341


namespace emily_initial_cards_l195_195888

theorem emily_initial_cards (x : ℤ) (h1 : x + 7 = 70) : x = 63 :=
by
  sorry

end emily_initial_cards_l195_195888


namespace annabelle_savings_l195_195019

theorem annabelle_savings (weekly_allowance : ℕ) (junk_food_fraction : ℚ) (sweets_cost : ℕ) 
    (h1 : weekly_allowance = 30) 
    (h2 : junk_food_fraction = 1 / 3) 
    (h3 : sweets_cost = 8) : 
    weekly_allowance - (weekly_allowance * (junk_food_fraction.numerator / junk_food_fraction.denominator)) - sweets_cost = 12 :=
by
  sorry

end annabelle_savings_l195_195019


namespace expected_successes_in_10_trials_l195_195125

noncomputable def prob_success (p : ℝ) (n : ℕ) : ℝ := n * p

theorem expected_successes_in_10_trials :
  let p := (1 : ℝ) - ((2 / 3) * (2 / 3))
  let n := 10
  in prob_success p n = 50 / 9 :=
by
  let p := (1 : ℝ) - ((2 / 3) * (2 / 3))
  let n := 10
  show prob_success p n = 50 / 9
  sorry

end expected_successes_in_10_trials_l195_195125


namespace find_coordinates_of_C_l195_195564

def Point := (ℝ × ℝ)

def A : Point := (-2, -1)
def B : Point := (4, 7)

/-- A custom definition to express that point C divides the segment AB in the ratio 2:1 from point B. -/
def is_point_C (C : Point) : Prop :=
  ∃ k : ℝ, k = 2 / 3 ∧
  C = (k * A.1 + (1 - k) * B.1, k * A.2 + (1 - k) * B.2)

theorem find_coordinates_of_C (C : Point) (h : is_point_C C) : 
  C = (2, 13 / 3) :=
sorry

end find_coordinates_of_C_l195_195564


namespace ap_contains_sixth_power_l195_195356

theorem ap_contains_sixth_power (a d : ℕ) (i j x y : ℕ) 
  (h_positive : ∀ n, a + n * d > 0) 
  (h_square : a + i * d = x^2) 
  (h_cube : a + j * d = y^3) :
  ∃ k z : ℕ, a + k * d = z^6 := 
  sorry

end ap_contains_sixth_power_l195_195356


namespace remainder_of_7n_mod_4_l195_195298

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l195_195298


namespace sin_2phi_l195_195775

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l195_195775


namespace pentagon_diagonal_probability_l195_195939

theorem pentagon_diagonal_probability :
  let S := set (fin 10) in
  (∃ s1 s2 ∈ S, s1 ≠ s2 ∧ 
   let same_length :=
     (s1 < 5 ∧ s2 < 5) ∨ (s1 ≥ 5 ∧ s2 ≥ 5) in
   (∑ x in S, ∑ y in (S.erase x), if same_length then 1 else 0) = 4 / 9 * (|S| * (|S| - 1))) := sorry

end pentagon_diagonal_probability_l195_195939


namespace average_temperature_l195_195578

def temperatures :=
  ∃ T_tue T_wed T_thu : ℝ,
    (44 + T_tue + T_wed + T_thu) / 4 = 48 ∧
    (T_tue + T_wed + T_thu + 36) / 4 = 46

theorem average_temperature :
  temperatures :=
by
  sorry

end average_temperature_l195_195578


namespace increased_number_l195_195611

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l195_195611


namespace radius_of_cone_l195_195732

theorem radius_of_cone (A : ℝ) (g : ℝ) (R : ℝ) (hA : A = 15 * Real.pi) (hg : g = 5) : R = 3 :=
sorry

end radius_of_cone_l195_195732


namespace black_beans_count_l195_195855

theorem black_beans_count (B G O : ℕ) (h₁ : G = B + 2) (h₂ : O = G - 1) (h₃ : B + G + O = 27) : B = 8 := by
  sorry

end black_beans_count_l195_195855


namespace remainder_when_7n_divided_by_4_l195_195306

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l195_195306


namespace machines_solution_l195_195124

theorem machines_solution (x : ℝ) (h : x > 0) :
  (1 / (x + 10) + 1 / (x + 3) + 1 / (2 * x) = 1 / x) → x = 3 / 2 := 
by
  sorry

end machines_solution_l195_195124


namespace operation_value_l195_195499

def operation1 (y : ℤ) : ℤ := 8 - y
def operation2 (y : ℤ) : ℤ := y - 8

theorem operation_value : operation2 (operation1 15) = -15 := by
  sorry

end operation_value_l195_195499


namespace f_800_l195_195264

-- Definitions of hypothesis from conditions given
def f : ℕ → ℤ := sorry
axiom f_mul (x y : ℕ) : f (x * y) = f x + f y
axiom f_10 : f 10 = 10
axiom f_40 : f 40 = 18

-- Proof problem statement: prove that f(800) = 32
theorem f_800 : f 800 = 32 := 
by
  sorry

end f_800_l195_195264


namespace coefficient_a_for_factor_l195_195294

noncomputable def P (a : ℚ) (x : ℚ) : ℚ := x^3 + 2 * x^2 + a * x + 20

theorem coefficient_a_for_factor (a : ℚ) :
  (∀ x : ℚ, (x - 3) ∣ P a x) → a = -65/3 :=
by
  sorry

end coefficient_a_for_factor_l195_195294


namespace find_certain_number_l195_195545

theorem find_certain_number (x certain_number : ℕ) (h: x = 3) (h2: certain_number = 5 * x + 4) : certain_number = 19 :=
by
  sorry

end find_certain_number_l195_195545


namespace union_M_N_l195_195202

-- Definitions for the sets M and N
def M : Set ℝ := { x | x^2 = x }
def N : Set ℝ := { x | Real.log x / Real.log 2 ≤ 0 }

-- Proof problem statement
theorem union_M_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end union_M_N_l195_195202


namespace find_x_l195_195931

def integers_x_y (x y : ℤ) : Prop :=
  x > y ∧ y > 0 ∧ x + y + x * y = 110

theorem find_x (x y : ℤ) (h : integers_x_y x y) : x = 36 := sorry

end find_x_l195_195931


namespace race_victory_l195_195244

variable (distance : ℕ := 200)
variable (timeA : ℕ := 18)
variable (timeA_beats_B_by : ℕ := 7)

theorem race_victory : ∃ meters_beats_B : ℕ, meters_beats_B = 56 :=
by
  let speedA := distance / timeA
  let timeB := timeA + timeA_beats_B_by
  let speedB := distance / timeB
  let distanceB := speedB * timeA
  let meters_beats_B := distance - distanceB
  use meters_beats_B
  sorry

end race_victory_l195_195244


namespace minimum_value_fraction_l195_195435

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

/-- Given that the function f(x) = log_a(4x-3) + 1 (where a > 0 and a ≠ 1) has a fixed point A(m, n), 
if for any positive numbers x and y, mx + ny = 3, 
then the minimum value of 1/(x+1) + 1/y is 1. -/
theorem minimum_value_fraction (a x y : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (hx : x + y = 3) : 
  (1 / (x + 1) + 1 / y) = 1 := 
sorry

end minimum_value_fraction_l195_195435


namespace marcus_has_210_cards_l195_195951

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the increment of baseball cards Marcus has over Carter
def increment : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + increment

-- Prove that Marcus has 210 baseball cards
theorem marcus_has_210_cards : marcus_cards = 210 :=
by simp [marcus_cards, carter_cards, increment]

end marcus_has_210_cards_l195_195951


namespace find_cost_price_l195_195697

theorem find_cost_price (SP : ℤ) (profit_percent : ℚ) (CP : ℤ) (h1 : SP = CP + (profit_percent * CP)) (h2 : SP = 240) (h3 : profit_percent = 0.25) : CP = 192 :=
by
  sorry

end find_cost_price_l195_195697


namespace matthew_younger_than_freddy_l195_195705

variables (M R F : ℕ)

-- Define the conditions
def sum_of_ages : Prop := M + R + F = 35
def matthew_older_than_rebecca : Prop := M = R + 2
def freddy_age : Prop := F = 15

-- Prove the statement "Matthew is 4 years younger than Freddy."
theorem matthew_younger_than_freddy (h1 : sum_of_ages M R F) (h2 : matthew_older_than_rebecca M R) (h3 : freddy_age F) :
    F - M = 4 := by
  sorry

end matthew_younger_than_freddy_l195_195705


namespace sin_double_angle_l195_195759

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l195_195759


namespace solve_fraction_eqn_l195_195121

def fraction_eqn_solution : Prop :=
  ∃ (x : ℝ), (x + 2) / (x - 1) = 0 ∧ x ≠ 1 ∧ x = -2

theorem solve_fraction_eqn : fraction_eqn_solution :=
sorry

end solve_fraction_eqn_l195_195121


namespace problem_1_problem_2_l195_195913

noncomputable def f (a x : ℝ) : ℝ := 2 * x^3 - a * x^2 + 8
noncomputable def g (a x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 - 12 * a^2 * x + 3 * a^3

theorem problem_1 (a : ℝ) : (∀ x ∈ set.Icc (1 : ℝ) 2, f a x < 0) → 10 < a :=
sorry

theorem problem_2 : ¬ ∃ a : ℤ, ∃ x ∈ set.Ioo (0 : ℝ) 1, is_local_min (g a) x :=
sorry

end problem_1_problem_2_l195_195913


namespace pascal_triangle_contains_prime_l195_195215

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l195_195215


namespace flour_for_each_cupcake_l195_195501

noncomputable def flour_per_cupcake (total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes : ℝ) : ℝ :=
  remaining_flour / num_cupcakes

theorem flour_for_each_cupcake :
  ∀ (total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes : ℝ),
    total_flour = 6 →
    remaining_flour = 2 →
    cake_flour_per_cake = 0.5 →
    cake_price = 2.5 →
    cupcake_price = 1 →
    total_revenue = 30 →
    num_cakes = 4 / 0.5 →
    num_cupcakes = 10 →
    flour_per_cupcake total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes = 0.2 :=
by intros; sorry

end flour_for_each_cupcake_l195_195501


namespace abs_diff_of_solutions_l195_195035

theorem abs_diff_of_solutions (h : ∀ x : ℝ, x^2 - 7 * x + 10 = 0 → x = 2 ∨ x = 5) :
  |(2 - 5 : ℝ)| = 3 :=
by sorry

end abs_diff_of_solutions_l195_195035


namespace sin_double_angle_solution_l195_195786

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l195_195786


namespace part_a_possible_part_b_not_possible_l195_195181

section GroupOfPeople

variable {People : Type} (knows : People → People → Prop) [symm : Symmetric knows]
variable (group : Finset People) [DecidableEq People]
variable (n : ℕ)

-- Condition: There are 15 people in the group
axiom people_count : group.card = 15

-- Definition: Degree of a person is the number of people they know
def degree (person : People) : ℕ := (group.filter (knows person)).card

-- Part (a): Prove it is possible that each person knows exactly 4 other people
theorem part_a_possible (h4 : ∀ p ∈ group, degree knows group p = 4) : ∃ graph, (∀ p ∈ group, degree knows group p = 4) :=
sorry

-- Part (b): Prove it is not possible that each person knows exactly 3 other people
theorem part_b_not_possible (h3 : ∀ p ∈ group, degree knows group p = 3) : ¬ ∃ graph, (∀ p ∈ group, degree knows group p = 3) :=
sorry

end GroupOfPeople

end part_a_possible_part_b_not_possible_l195_195181


namespace balls_in_boxes_l195_195529

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l195_195529


namespace calculate_y_l195_195071

theorem calculate_y (x y : ℝ) (h1 : x = 101) (h2 : x^3 * y - 2 * x^2 * y + x * y = 101000) : y = 1 / 10 :=
by
  sorry

end calculate_y_l195_195071


namespace pascal_triangle_47_rows_l195_195223

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l195_195223


namespace increase_by_percentage_l195_195623

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l195_195623


namespace lucy_crayons_correct_l195_195138

-- Define the number of crayons Willy has.
def willyCrayons : ℕ := 5092

-- Define the number of extra crayons Willy has compared to Lucy.
def extraCrayons : ℕ := 1121

-- Define the number of crayons Lucy has.
def lucyCrayons : ℕ := willyCrayons - extraCrayons

-- Statement to prove
theorem lucy_crayons_correct : lucyCrayons = 3971 := 
by
  -- The proof is omitted as per instructions
  sorry

end lucy_crayons_correct_l195_195138


namespace remainder_17_pow_2047_mod_23_l195_195843

theorem remainder_17_pow_2047_mod_23 : (17 ^ 2047) % 23 = 11 := 
by
  sorry

end remainder_17_pow_2047_mod_23_l195_195843


namespace range_of_a_l195_195520

theorem range_of_a (A B : Set ℝ) (a : ℝ)
  (hA : A = {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5})
  (hB : B = {x | 3 ≤ x ∧ x ≤ 22}) :
  A ⊆ (A ∩ B) ↔ (1 ≤ a ∧ a ≤ 9) :=
by
  sorry

end range_of_a_l195_195520


namespace marie_divided_by_alex_l195_195954

theorem marie_divided_by_alex :
  let maries_sum := (finset.range 300).sum (λ n, 2 * (n + 1))
  let alexs_sum := (finset.range 300).sum (λ n, n + 1)
  (maries_sum : ℚ) / alexs_sum = 2 :=
by
  -- Definitions based on the problem
  let maries_sum := (finset.range 300).sum (λ n, 2 * (n + 1))
  let alexs_sum := (finset.range 300).sum (λ n, n + 1)
  -- sorry added to indicate the proof is not complete
  sorry

end marie_divided_by_alex_l195_195954


namespace solve_trig_eq_l195_195821

theorem solve_trig_eq (x : ℝ) (k : ℤ) :
  (x = (π / 12) + 2 * k * π ∨
   x = (7 * π / 12) + 2 * k * π ∨
   x = (7 * π / 6) + 2 * k * π ∨
   x = -(5 * π / 6) + 2 * k * π) →
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 :=
sorry

end solve_trig_eq_l195_195821


namespace change_calculation_l195_195850

def cost_of_apple : ℝ := 0.75
def amount_paid : ℝ := 5.00

theorem change_calculation : (amount_paid - cost_of_apple) = 4.25 := by
  sorry

end change_calculation_l195_195850


namespace remainder_of_M_div_by_51_is_zero_l195_195090

open Nat

noncomputable def M := 1234567891011121314151617181920212223242526272829303132333435363738394041424344454647484950

theorem remainder_of_M_div_by_51_is_zero :
  M % 51 = 0 :=
sorry

end remainder_of_M_div_by_51_is_zero_l195_195090


namespace number_of_books_in_box_l195_195463

theorem number_of_books_in_box (total_weight : ℕ) (weight_per_book : ℕ) 
  (h1 : total_weight = 42) (h2 : weight_per_book = 3) : total_weight / weight_per_book = 14 :=
by sorry

end number_of_books_in_box_l195_195463


namespace ratio_of_cube_dimensions_l195_195859

theorem ratio_of_cube_dimensions (V_original V_larger : ℝ) (hV_org : V_original = 64) (hV_lrg : V_larger = 512) :
  (∃ r : ℝ, r^3 = V_larger / V_original) ∧ r = 2 := 
sorry

end ratio_of_cube_dimensions_l195_195859


namespace sum_a1_a4_l195_195387

variables (S : ℕ → ℕ) (a : ℕ → ℕ)

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ := n^2 + n + 1

-- Define the individual terms of the sequence
def term_seq (n : ℕ) : ℕ :=
if n = 1 then sum_seq 1 else sum_seq n - sum_seq (n - 1)

-- Prove that the sum of the first and fourth terms equals 11
theorem sum_a1_a4 : 
  (term_seq 1) + (term_seq 4) = 11 :=
by
  -- to be completed with proof steps
  sorry

end sum_a1_a4_l195_195387


namespace pascal_triangle_47_rows_l195_195222

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l195_195222


namespace calc_correct_operation_l195_195466

theorem calc_correct_operation (a : ℕ) :
  (2 : ℕ) * a + (3 : ℕ) * a = (5 : ℕ) * a :=
by
  -- Proof
  sorry

end calc_correct_operation_l195_195466


namespace fraction_of_2d_nails_l195_195020

theorem fraction_of_2d_nails (x : ℝ) (h1 : x + 0.5 = 0.75) : x = 0.25 :=
by
  sorry

end fraction_of_2d_nails_l195_195020


namespace increase_result_l195_195640

-- Given conditions
def original_number : ℝ := 80
def increase_percentage : ℝ := 1.5

-- The result after the increase
theorem increase_result (h1 : original_number = 80) (h2 : increase_percentage = 1.5) : 
  original_number + (increase_percentage * original_number) = 200 := by
  sorry

end increase_result_l195_195640


namespace missing_digit_in_138_x_6_divisible_by_9_l195_195191

theorem missing_digit_in_138_x_6_divisible_by_9 :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 9 ∧ (1 + 3 + 8 + x + 6) % 9 = 0 ∧ x = 0 :=
by
  sorry

end missing_digit_in_138_x_6_divisible_by_9_l195_195191


namespace imaginary_part_of_complex_l195_195709

open Complex

theorem imaginary_part_of_complex (i : ℂ) (z : ℂ) (h1 : i^2 = -1) (h2 : z = (3 - 2 * i^3) / (1 + i)) : z.im = -1 / 2 :=
by {
  -- Proof would go here
  sorry
}

end imaginary_part_of_complex_l195_195709


namespace commute_time_l195_195140

theorem commute_time (d w t : ℝ) (x : ℝ) (h_distance : d = 1.5) (h_walking_speed : w = 3) (h_train_speed : t = 20)
  (h_extra_time : 30 = 4.5 + x + 2) : x = 25.5 :=
by {
  -- Add the statement of the proof
  sorry
}

end commute_time_l195_195140


namespace events_are_mutually_exclusive_but_not_opposite_l195_195428

-- Definitions based on the conditions:
structure BallBoxConfig where
  ball1 : Fin 4 → ℕ     -- Function representing the placement of ball number 1 into one of the 4 boxes
  h_distinct : ∀ i j, i ≠ j → ball1 i ≠ ball1 j

def event_A (cfg : BallBoxConfig) : Prop := cfg.ball1 ⟨0, sorry⟩ = 1
def event_B (cfg : BallBoxConfig) : Prop := cfg.ball1 ⟨0, sorry⟩ = 2

-- The proof problem:
theorem events_are_mutually_exclusive_but_not_opposite (cfg : BallBoxConfig) :
  (event_A cfg ∨ event_B cfg) ∧ ¬ (event_A cfg ∧ event_B cfg) :=
sorry

end events_are_mutually_exclusive_but_not_opposite_l195_195428


namespace find_n_l195_195801

noncomputable def f (x : ℤ) : ℤ := sorry -- f is some polynomial with integer coefficients

theorem find_n (n : ℤ) (h1 : f 1 = -1) (h4 : f 4 = 2) (h8 : f 8 = 34) (hn : f n = n^2 - 4 * n - 18) : n = 3 ∨ n = 6 :=
sorry

end find_n_l195_195801


namespace total_value_of_horse_and_saddle_l195_195563

def saddle_value : ℝ := 12.5
def horse_value : ℝ := 7 * saddle_value

theorem total_value_of_horse_and_saddle : horse_value + saddle_value = 100 := by
  sorry

end total_value_of_horse_and_saddle_l195_195563


namespace find_point_A_equidistant_l195_195492

theorem find_point_A_equidistant :
  ∃ (x : ℝ), (∃ A : ℝ × ℝ × ℝ, A = (x, 0, 0)) ∧
              (∃ B : ℝ × ℝ × ℝ, B = (4, 0, 5)) ∧
              (∃ C : ℝ × ℝ × ℝ, C = (5, 4, 2)) ∧
              (dist (x, 0, 0) (4, 0, 5) = dist (x, 0, 0) (5, 4, 2)) ∧ 
              (x = 2) :=
by
  sorry

end find_point_A_equidistant_l195_195492


namespace remainder_when_divided_by_13_l195_195669

theorem remainder_when_divided_by_13 (N : ℕ) (k : ℕ) (hk : N = 39 * k + 15) : N % 13 = 2 :=
sorry

end remainder_when_divided_by_13_l195_195669


namespace cp_of_apple_l195_195674

theorem cp_of_apple (SP : ℝ) (hSP : SP = 17) (loss_fraction : ℝ) (h_loss_fraction : loss_fraction = 1 / 6) : 
  ∃ CP : ℝ, CP = 20.4 ∧ SP = CP - loss_fraction * CP :=
by
  -- Placeholder for proof
  sorry

end cp_of_apple_l195_195674


namespace inequality_for_five_real_numbers_l195_195558

open Real

theorem inequality_for_five_real_numbers
  (a1 a2 a3 a4 a5 : ℝ)
  (h1 : 1 < a1)
  (h2 : 1 < a2)
  (h3 : 1 < a3)
  (h4 : 1 < a4)
  (h5 : 1 < a5) :
  16 * (a1 * a2 * a3 * a4 * a5 + 1) ≥ (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) * (1 + a5) := 
sorry

end inequality_for_five_real_numbers_l195_195558


namespace mod_squares_eq_one_l195_195899

theorem mod_squares_eq_one
  (n : ℕ)
  (h : n = 5)
  (a : ℤ)
  (ha : ∃ b : ℕ, ↑b = a ∧ b * b ≡ 1 [MOD 5]) :
  (a * a) % n = 1 :=
by
  sorry

end mod_squares_eq_one_l195_195899


namespace quadrilateral_side_inequality_quadrilateral_side_inequality_if_intersect_l195_195005

variable (a b c d : ℝ)
variable (angle_B angle_D : ℝ)
variable (d_intersect_circle : Prop)

-- Condition that angles B and D sum up to more than 180 degrees.
def angle_condition : Prop := angle_B + angle_D > 180

-- Condition for sides of the convex quadrilateral
def side_condition1 : Prop := a + c > b + d

-- Condition for the circle touching sides a, b, and c
def circle_tangent : Prop := True -- Placeholder as no function to verify this directly in Lean

theorem quadrilateral_side_inequality (h1 : angle_condition angle_B angle_D) 
                                      (h2 : circle_tangent) 
                                      (h3 : ¬ d_intersect_circle) 
                                      : a + c > b + d :=
  sorry

theorem quadrilateral_side_inequality_if_intersect (h1 : angle_condition angle_B angle_D) 
                                                   (h2 : circle_tangent) 
                                                   (h3 : d_intersect_circle) 
                                                   : a + c < b + d :=
  sorry

end quadrilateral_side_inequality_quadrilateral_side_inequality_if_intersect_l195_195005


namespace magic_show_l195_195008

theorem magic_show (performances : ℕ) (prob_never_reappear : ℚ) (prob_two_reappear : ℚ)
  (h_performances : performances = 100)
  (h_prob_never_reappear : prob_never_reappear = 1 / 10)
  (h_prob_two_reappear : prob_two_reappear = 1 / 5) :
  let never_reappear := prob_never_reappear * performances,
      two_reappear := prob_two_reappear * performances,
      normal_reappear := performances,
      extra_reappear := two_reappear,
      total_reappear := normal_reappear + extra_reappear - never_reappear in
  total_reappear = 110 := by
  sorry

end magic_show_l195_195008


namespace sufficient_condition_l195_195498

variable (x : ℝ) (a : ℝ)

theorem sufficient_condition (h : ∀ x : ℝ, |x| + |x - 1| ≥ 1) : a < 1 → ∀ x : ℝ, a ≤ |x| + |x - 1| :=
by
  sorry

end sufficient_condition_l195_195498


namespace tree_leaves_l195_195157

theorem tree_leaves (initial_leaves : ℕ) (first_week_fraction : ℚ) (second_week_percentage : ℚ) (third_week_fraction : ℚ) :
  initial_leaves = 1000 →
  first_week_fraction = 2 / 5 →
  second_week_percentage = 40 / 100 →
  third_week_fraction = 3 / 4 →
  let leaves_after_first_week := initial_leaves - (first_week_fraction * initial_leaves).toNat,
      leaves_after_second_week := leaves_after_first_week - (second_week_percentage * leaves_after_first_week).toNat,
      leaves_after_third_week := leaves_after_second_week - (third_week_fraction * leaves_after_second_week).toNat
  in leaves_after_third_week = 90 :=
begin
  intros h1 h2 h3 h4,
  unfold leaves_after_first_week leaves_after_second_week leaves_after_third_week,
  rw [h1, h2, h3, h4],
  norm_num,
end

end tree_leaves_l195_195157


namespace average_minutes_run_l195_195364

theorem average_minutes_run (t : ℕ) (t_pos : 0 < t) 
  (average_first_graders : ℕ := 8) 
  (average_second_graders : ℕ := 12) 
  (average_third_graders : ℕ := 16)
  (num_first_graders : ℕ := 9 * t)
  (num_second_graders : ℕ := 3 * t)
  (num_third_graders : ℕ := t) :
  (8 * 9 * t + 12 * 3 * t + 16 * t) / (9 * t + 3 * t + t) = 10 := 
by
  sorry

end average_minutes_run_l195_195364


namespace sin_17pi_over_6_l195_195893

theorem sin_17pi_over_6 : Real.sin (17 * Real.pi / 6) = 1 / 2 :=
by
  sorry

end sin_17pi_over_6_l195_195893


namespace adult_ticket_cost_l195_195126

/--
Tickets at a local theater cost a certain amount for adults and 2 dollars for kids under twelve.
Given that 175 tickets were sold and the profit was 750 dollars, and 75 kid tickets were sold,
prove that an adult ticket costs 6 dollars.
-/
theorem adult_ticket_cost
  (kid_ticket_price : ℕ := 2)
  (kid_tickets_sold : ℕ := 75)
  (total_tickets_sold : ℕ := 175)
  (total_profit : ℕ := 750)
  (adult_tickets_sold : ℕ := total_tickets_sold - kid_tickets_sold)
  (adult_ticket_revenue : ℕ := total_profit - kid_ticket_price * kid_tickets_sold)
  (adult_ticket_cost : ℕ := adult_ticket_revenue / adult_tickets_sold) :
  adult_ticket_cost = 6 :=
by
  sorry

end adult_ticket_cost_l195_195126


namespace tan_sum_eq_one_l195_195393

theorem tan_sum_eq_one (a b : ℝ) (h1 : Real.tan a = 1 / 2) (h2 : Real.tan b = 1 / 3) :
    Real.tan (a + b) = 1 := 
by
  sorry

end tan_sum_eq_one_l195_195393


namespace frog_reaches_vertical_side_l195_195471

def P (x y : ℕ) : ℝ := 
  if (x = 3 ∧ y = 3) then 0 -- blocked cell
  else if (x = 0 ∨ x = 5) then 1 -- vertical boundary
  else if (y = 0 ∨ y = 5) then 0 -- horizontal boundary
  else sorry -- inner probabilities to be calculated

theorem frog_reaches_vertical_side : P 2 2 = 5 / 8 :=
by sorry

end frog_reaches_vertical_side_l195_195471


namespace evaluated_result_l195_195372

noncomputable def evaluate_expression (y : ℝ) (hy : y ≠ 0) : ℝ :=
  (18 * y^3) * (4 * y^2) * (1 / (2 * y)^3)

theorem evaluated_result (y : ℝ) (hy : y ≠ 0) : evaluate_expression y hy = 9 * y^2 :=
by
  sorry

end evaluated_result_l195_195372


namespace increase_by_150_percent_l195_195650

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l195_195650


namespace exponent_subtraction_l195_195394

variable {a : ℝ} {m n : ℕ}

theorem exponent_subtraction (hm : a ^ m = 12) (hn : a ^ n = 3) : a ^ (m - n) = 4 :=
by
  sorry

end exponent_subtraction_l195_195394


namespace garden_area_l195_195154

/-- A rectangular garden is 350 cm long and 50 cm wide. Determine its area in square meters. -/
theorem garden_area (length_cm width_cm : ℝ) (h_length : length_cm = 350) (h_width : width_cm = 50) : (length_cm / 100) * (width_cm / 100) = 1.75 :=
by
  sorry

end garden_area_l195_195154


namespace increase_80_by_150_percent_l195_195633

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l195_195633


namespace subset_implies_value_l195_195728

theorem subset_implies_value (m : ℝ) (A : Set ℝ) (B : Set ℝ) 
  (hA : A = {-1, 3, 2*m-1}) (hB : B = {3, m}) (hSub : B ⊆ A) : 
  m = -1 ∨ m = 1 := by
  sorry

end subset_implies_value_l195_195728


namespace work_completed_in_initial_days_l195_195933

theorem work_completed_in_initial_days (x : ℕ) : 
  (100 * x = 50 * 40) → x = 20 :=
by
  sorry

end work_completed_in_initial_days_l195_195933


namespace increase_by_150_percent_l195_195647

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l195_195647


namespace quadratic_discriminant_l195_195028

noncomputable def discriminant (a b c : ℚ) : ℚ :=
  b^2 - 4 * a * c

theorem quadratic_discriminant :
  discriminant 6 (6 + 1/6) (1/6) = 1225 / 36 :=
by
  sorry

end quadratic_discriminant_l195_195028


namespace pipe_filling_time_l195_195343

/-- 
A problem involving two pipes filling and emptying a tank. 
Time taken for the first pipe to fill the tank is proven to be 16.8 minutes.
-/
theorem pipe_filling_time :
  ∃ T : ℝ, (∀ T, let r1 := 1 / T
                let r2 := 1 / 24
                let time_both_pipes_open := 36
                let time_first_pipe_only := 6
                (r1 - r2) * time_both_pipes_open + r1 * time_first_pipe_only = 1) ∧
           T = 16.8 :=
by
  sorry

end pipe_filling_time_l195_195343


namespace contradiction_method_conditions_l195_195258

theorem contradiction_method_conditions :
  (using_judgments_contrary_to_conclusion ∧ using_conditions_of_original_proposition ∧ using_axioms_theorems_definitions) =
  (needed_conditions_method_of_contradiction) :=
sorry

end contradiction_method_conditions_l195_195258


namespace village_population_l195_195332

-- Defining the variables and the condition
variable (P : ℝ) (h : 0.9 * P = 36000)

-- Statement of the theorem to prove
theorem village_population : P = 40000 :=
by sorry

end village_population_l195_195332


namespace find_x1_value_l195_195384

theorem find_x1_value (x1 x2 x3 : ℝ) (h1 : 0 ≤ x3) (h2 : x3 ≤ x2) (h3 : x2 ≤ x1) (h4 : x1 ≤ 1) 
  (h_eq : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 3) : 
  x1 = 2 / 3 := 
sorry

end find_x1_value_l195_195384


namespace increase_80_by_150_percent_l195_195657

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l195_195657


namespace radon_nikodym_inv_exists_l195_195266

open MeasureTheory

variables {Ω : Type*} {𝓕 : MeasurableSpace Ω} (λ μ : Measure Ω)
variables (f : Ω → ℝ)
variable (c : ℝ)

noncomputable def radon_nikodym_derivative_inv (hf : Integrable f μ) (hμ : μ (setOf (λ ω, f ω = 0)) = 0) : Ω → ℝ :=
  λ ω, if f ω ≠ 0 then 1 / f ω else c

theorem radon_nikodym_inv_exists 
  (hf : f = (λ μ.toMeasurable λ).rnDeriv μ.toMeasurable)
  (hμ : μ (setOf (λ ω, f ω = 0)) = 0) :
  μ.withDensity (radon_nikodym_derivative_inv λ μ f c hf hμ) = λ :=
sorry

end radon_nikodym_inv_exists_l195_195266


namespace part1_part2_l195_195066

theorem part1 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) : k > 3 / 4 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0) (hx1x2 : ∀ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + (2*k + 1)*x1 + k^2 + 1 = 0 ∧ x2^2 + (2*k + 1)*x2 + k^2 + 1 = 0 → x1 * x2 = 5) : k = 2 :=
sorry

end part1_part2_l195_195066


namespace probability_of_drawing_red_second_draw_l195_195444

theorem probability_of_drawing_red_second_draw :
  (∀ (balls : list ℕ),
    length balls = 5 →
    count (λ x, x = 0) balls = 3 →
    count (λ x, x = 1) balls = 2 →
    noncomputable_prob (draw_with_replacement balls 2) (λ draws, draws.nth 1 = some 0) = 3/5) :=
begin
  sorry
end

end probability_of_drawing_red_second_draw_l195_195444


namespace relationship_and_range_max_profit_find_a_l195_195429

noncomputable def functional_relationship (x : ℝ) : ℝ :=
if 40 ≤ x ∧ x ≤ 50 then 5
else if 50 < x ∧ x ≤ 100 then 10 - 0.1 * x
else 0  -- default case to handle x out of range, though ideally this should not occur in the context.

theorem relationship_and_range : 
  ∀ (x : ℝ), (40 ≤ x ∧ x ≤ 100) →
    (functional_relationship x = 
    (if 40 ≤ x ∧ x ≤ 50 then 5 else if 50 < x ∧ x ≤ 100 then 10 - 0.1 * x else 0)) :=
sorry

noncomputable def monthly_profit (x : ℝ) : ℝ :=
(x - 40) * functional_relationship x

theorem max_profit : 
  (∀ x, 40 ≤ x ∧ x ≤ 100 → monthly_profit x ≤ 90) ∧
  (monthly_profit 70 = 90) :=
sorry

noncomputable def donation_profit (x a : ℝ) : ℝ :=
(x - 40 - a) * (10 - 0.1 * x)

theorem find_a (a : ℝ) : 
  (∀ x, x ≤ 70 → donation_profit x a ≤ 78) ∧
  (donation_profit 70 a = 78) → 
  a = 4 :=
sorry

end relationship_and_range_max_profit_find_a_l195_195429


namespace increased_number_l195_195610

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l195_195610


namespace air_conditioner_sales_l195_195274

/-- Represent the conditions -/
def conditions (x y m : ℕ) : Prop :=
  (3 * x + 5 * y = 23500) ∧
  (4 * x + 10 * y = 42000) ∧
  (x = 2500) ∧
  (y = 3200) ∧
  (700 * (50 - m) + 800 * m ≥ 38000)

/-- Prove that the unit selling prices of models A and B are 2500 yuan and 3200 yuan respectively,
    and at least 30 units of model B need to be purchased for a profit of at least 38000 yuan,
    given the conditions. -/
theorem air_conditioner_sales :
  ∃ (x y m : ℕ), conditions x y m ∧ m ≥ 30 := by
  sorry

end air_conditioner_sales_l195_195274


namespace parabola_focus_distance_l195_195521

theorem parabola_focus_distance (p m : ℝ) (hp : p > 0)
  (P_on_parabola : m^2 = 2 * p)
  (PF_dist : (1 + p / 2) = 3) : p = 4 := 
  sorry

end parabola_focus_distance_l195_195521


namespace pascal_triangle_47_l195_195217

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l195_195217


namespace train_speed_in_kmh_l195_195690

def length_of_train : ℝ := 156
def length_of_bridge : ℝ := 219.03
def time_to_cross_bridge : ℝ := 30
def speed_of_train_kmh : ℝ := 45.0036

theorem train_speed_in_kmh :
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = speed_of_train_kmh :=
by
  sorry

end train_speed_in_kmh_l195_195690


namespace even_sum_probability_l195_195031

-- Define the probabilities for the first wheel
def prob_even_first_wheel : ℚ := 3 / 6
def prob_odd_first_wheel : ℚ := 3 / 6

-- Define the probabilities for the second wheel
def prob_even_second_wheel : ℚ := 3 / 4
def prob_odd_second_wheel : ℚ := 1 / 4

-- Probability that the sum of the two selected numbers is even
def prob_even_sum : ℚ :=
  (prob_even_first_wheel * prob_even_second_wheel) +
  (prob_odd_first_wheel * prob_odd_second_wheel)

-- The theorem to prove
theorem even_sum_probability : prob_even_sum = 13 / 24 := by
  sorry

end even_sum_probability_l195_195031


namespace final_salt_concentration_is_25_l195_195330

-- Define the initial conditions
def original_solution_weight : ℝ := 100
def original_salt_concentration : ℝ := 0.10
def added_salt_weight : ℝ := 20

-- Define the amount of salt in the original solution
def original_salt_weight := original_solution_weight * original_salt_concentration

-- Define the total amount of salt after adding pure salt
def total_salt_weight := original_salt_weight + added_salt_weight

-- Define the total weight of the new solution
def new_solution_weight := original_solution_weight + added_salt_weight

-- Define the final salt concentration
noncomputable def final_salt_concentration := (total_salt_weight / new_solution_weight) * 100

-- Prove the final salt concentration equals 25%
theorem final_salt_concentration_is_25 : final_salt_concentration = 25 :=
by
  sorry

end final_salt_concentration_is_25_l195_195330


namespace eggs_town_hall_l195_195073

-- Definitions of given conditions
def eggs_club_house : ℕ := 40
def eggs_park : ℕ := 25
def total_eggs_found : ℕ := 80

-- Problem statement
theorem eggs_town_hall : total_eggs_found - (eggs_club_house + eggs_park) = 15 := by
  sorry

end eggs_town_hall_l195_195073


namespace total_area_of_combined_shape_l195_195344

theorem total_area_of_combined_shape
  (length_rectangle : ℝ) (width_rectangle : ℝ) (side_square : ℝ)
  (h_length : length_rectangle = 0.45)
  (h_width : width_rectangle = 0.25)
  (h_side : side_square = 0.15) :
  (length_rectangle * width_rectangle + side_square * side_square) = 0.135 := 
by 
  sorry

end total_area_of_combined_shape_l195_195344


namespace total_miles_walked_l195_195553

def weekly_group_walk_miles : ℕ := 3 * 6

def Jamie_additional_walk_miles_per_week : ℕ := 2 * 6
def Sue_additional_walk_miles_per_week : ℕ := 1 * 6 -- half of Jamie's additional walk
def Laura_additional_walk_miles_per_week : ℕ := 1 * 3 -- 1 mile every two days for 6 days
def Melissa_additional_walk_miles_per_week : ℕ := 2 * 2 -- 2 miles every three days for 6 days
def Katie_additional_walk_miles_per_week : ℕ := 1 * 6

def Jamie_weekly_miles : ℕ := weekly_group_walk_miles + Jamie_additional_walk_miles_per_week
def Sue_weekly_miles : ℕ := weekly_group_walk_miles + Sue_additional_walk_miles_per_week
def Laura_weekly_miles : ℕ := weekly_group_walk_miles + Laura_additional_walk_miles_per_week
def Melissa_weekly_miles : ℕ := weekly_group_walk_miles + Melissa_additional_walk_miles_per_week
def Katie_weekly_miles : ℕ := weekly_group_walk_miles + Katie_additional_walk_miles_per_week

def weeks_in_month : ℕ := 4

def Jamie_monthly_miles : ℕ := Jamie_weekly_miles * weeks_in_month
def Sue_monthly_miles : ℕ := Sue_weekly_miles * weeks_in_month
def Laura_monthly_miles : ℕ := Laura_weekly_miles * weeks_in_month
def Melissa_monthly_miles : ℕ := Melissa_weekly_miles * weeks_in_month
def Katie_monthly_miles : ℕ := Katie_weekly_miles * weeks_in_month

def total_monthly_miles : ℕ :=
  Jamie_monthly_miles + Sue_monthly_miles + Laura_monthly_miles + Melissa_monthly_miles + Katie_monthly_miles

theorem total_miles_walked (month_has_30_days : Prop) : total_monthly_miles = 484 :=
by
  unfold total_monthly_miles
  unfold Jamie_monthly_miles Sue_monthly_miles Laura_monthly_miles Melissa_monthly_miles Katie_monthly_miles
  unfold Jamie_weekly_miles Sue_weekly_miles Laura_weekly_miles Melissa_weekly_miles Katie_weekly_miles
  unfold weekly_group_walk_miles Jamie_additional_walk_miles_per_week Sue_additional_walk_miles_per_week Laura_additional_walk_miles_per_week Melissa_additional_walk_miles_per_week Katie_additional_walk_miles_per_week
  unfold weeks_in_month
  sorry

end total_miles_walked_l195_195553


namespace weight_of_new_person_l195_195433

-- Definitions for the conditions given.

-- Average weight increase
def avg_weight_increase : ℝ := 2.5

-- Number of persons
def num_persons : ℕ := 8

-- Weight of the person being replaced
def weight_replaced : ℝ := 65

-- Total weight increase
def total_weight_increase : ℝ := num_persons * avg_weight_increase

-- Statement to prove the weight of the new person
theorem weight_of_new_person : 
  ∃ (W_new : ℝ), W_new = weight_replaced + total_weight_increase :=
sorry

end weight_of_new_person_l195_195433


namespace pascal_triangle_contains_47_l195_195211

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l195_195211


namespace intersection_A_B_l195_195383

-- Defining set A condition
def A : Set ℝ := {x | x - 1 < 2}

-- Defining set B condition
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

-- The goal to prove
theorem intersection_A_B : {x | x > 0 ∧ x < 3} = (A ∩ { x | 0 < x ∧ x < 8 }) :=
by
  sorry

end intersection_A_B_l195_195383


namespace pow_eq_of_pow_sub_eq_l195_195233

theorem pow_eq_of_pow_sub_eq (a : ℝ) (m n : ℕ) (h1 : a^m = 6) (h2 : a^(m-n) = 2) : a^n = 3 := 
by
  sorry

end pow_eq_of_pow_sub_eq_l195_195233


namespace cube_triangle_area_sum_solution_l195_195440

def cube_vertex_triangle_area_sum (m n p : ℤ) : Prop :=
  m + n + p = 121 ∧
  (∀ (a : ℕ) (b : ℕ) (c : ℕ), a * b * c = 8) -- Ensures the vertices are for a 2*2*2 cube

theorem cube_triangle_area_sum_solution :
  cube_vertex_triangle_area_sum 48 64 9 :=
by
  unfold cube_vertex_triangle_area_sum
  split
  · exact rfl -- m + n + p = 121
  · intros a b c h
    sorry -- Conditions ensuring these m, n, p were calculated from a 2x2x2 cube

end cube_triangle_area_sum_solution_l195_195440


namespace pascal_row_contains_prime_47_l195_195231

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l195_195231


namespace ellipse_equation_correct_l195_195870

theorem ellipse_equation_correct :
  ∃ (a b h k : ℝ), 
    h = 4 ∧ 
    k = 0 ∧ 
    a = 10 + 2 * Real.sqrt 10 ∧ 
    b = Real.sqrt (101 + 20 * Real.sqrt 10) ∧ 
    (∀ x y : ℝ, (x, y) = (9, 6) → 
    ((x - h)^2 / a^2 + y^2 / b^2 = 1)) ∧
    (dist (4 - 3, 0) (4 + 3, 0) = 6) := 
sorry

end ellipse_equation_correct_l195_195870


namespace toms_total_score_l195_195127

def regular_enemy_points : ℕ := 10
def elite_enemy_points : ℕ := 25
def boss_enemy_points : ℕ := 50

def regular_enemy_bonus (kills : ℕ) : ℚ :=
  if 100 ≤ kills ∧ kills < 150 then 0.50
  else if 150 ≤ kills ∧ kills < 200 then 0.75
  else if kills ≥ 200 then 1.00
  else 0.00

def elite_enemy_bonus (kills : ℕ) : ℚ :=
  if 15 ≤ kills ∧ kills < 25 then 0.30
  else if 25 ≤ kills ∧ kills < 35 then 0.50
  else if kills >= 35 then 0.70
  else 0.00

def boss_enemy_bonus (kills : ℕ) : ℚ :=
  if 5 ≤ kills ∧ kills < 10 then 0.20
  else if kills ≥ 10 then 0.40
  else 0.00

noncomputable def total_score (regular_kills elite_kills boss_kills : ℕ) : ℚ :=
  let regular_points := regular_kills * regular_enemy_points
  let elite_points := elite_kills * elite_enemy_points
  let boss_points := boss_kills * boss_enemy_points
  let regular_total := regular_points + regular_points * regular_enemy_bonus regular_kills
  let elite_total := elite_points + elite_points * elite_enemy_bonus elite_kills
  let boss_total := boss_points + boss_points * boss_enemy_bonus boss_kills
  regular_total + elite_total + boss_total

theorem toms_total_score :
  total_score 160 20 8 = 3930 := by
  sorry

end toms_total_score_l195_195127


namespace increase_80_by_150_percent_l195_195654

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l195_195654


namespace stamp_blocks_inequalities_l195_195392

noncomputable def b (n : ℕ) : ℕ := sorry

theorem stamp_blocks_inequalities (n : ℕ) (m : ℕ) (hn : 0 < n) :
  ∃ c d : ℝ, c = 2 / 7 ∧ d = (4 * m^2 + 4 * m + 40) / 5 ∧
    (1 / 7 : ℝ) * n^2 - c * n ≤ b n ∧ 
    b n ≤ (1 / 5 : ℝ) * n^2 + d * n := 
  sorry

end stamp_blocks_inequalities_l195_195392


namespace three_power_not_square_l195_195808

theorem three_power_not_square (m n : ℕ) (hm : m ≥ 1) (hn : n ≥ 1) : ¬ ∃ k : ℕ, k * k = 3^m + 3^n + 1 := by 
  sorry

end three_power_not_square_l195_195808


namespace sally_picked_peaches_l195_195820

variable (p_initial p_current p_picked : ℕ)

theorem sally_picked_peaches (h1 : p_initial = 13) (h2 : p_current = 55) :
  p_picked = p_current - p_initial → p_picked = 42 :=
by
  intros
  sorry

end sally_picked_peaches_l195_195820


namespace erasers_total_l195_195096

-- Define the initial amount of erasers
def initialErasers : Float := 95.0

-- Define the amount of erasers Marie buys
def boughtErasers : Float := 42.0

-- Define the total number of erasers Marie ends with
def totalErasers : Float := 137.0

-- The theorem that needs to be proven
theorem erasers_total 
  (initial : Float := initialErasers)
  (bought : Float := boughtErasers)
  (total : Float := totalErasers) :
  initial + bought = total :=
sorry

end erasers_total_l195_195096


namespace magician_act_reappearance_l195_195007

-- Defining the conditions as given in the problem
def total_performances : ℕ := 100

def no_one_reappears (perf : ℕ) : ℕ := perf / 10
def two_reappear (perf : ℕ) : ℕ := perf / 5
def one_reappears (perf : ℕ) : ℕ := perf - no_one_reappears perf - two_reappear perf
def total_reappeared (perf : ℕ) : ℕ := one_reappears perf + 2 * two_reappear perf

-- The statement to be proved
theorem magician_act_reappearance : total_reappeared total_performances = 110 := by
  sorry

end magician_act_reappearance_l195_195007


namespace sin_2phi_l195_195780

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l195_195780


namespace quadratic_min_value_l195_195792

theorem quadratic_min_value (f : ℕ → ℚ) (n : ℕ)
  (h₁ : f n = 6)
  (h₂ : f (n + 1) = 5)
  (h₃ : f (n + 2) = 5) :
  ∃ c : ℚ, c = 39 / 8 ∧ ∀ x : ℕ, f x ≥ c :=
by
  sorry

end quadratic_min_value_l195_195792


namespace inequality_proof_l195_195818

theorem inequality_proof (x y : ℝ) (h1 : x^2 ≥ y) (h2 : y^2 ≥ x) : 
  (x / (y^2 + 1) + y / (x^2 + 1) ≤ 1) :=
sorry

end inequality_proof_l195_195818


namespace find_ordered_pair_l195_195194

theorem find_ordered_pair:
  ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ 10 * m * n = 45 - 5 * m - 3 * n ∧ (m, n) = (1, 11) :=
by
  sorry

end find_ordered_pair_l195_195194


namespace range_of_a_bisection_method_solution_l195_195201

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f a x = 0) :
  (12 * (27 - 4 * Real.sqrt 6) / 211 ≤ a) ∧ (a ≤ 12 * (27 + 4 * Real.sqrt 6) / 211) :=
sorry

theorem bisection_method_solution (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f (32 / 17) x = 0) :
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ (|f (32 / 17) x| < 0.1) :=
sorry

end range_of_a_bisection_method_solution_l195_195201


namespace solve_equation_l195_195824

noncomputable def equation (x : ℝ) : ℝ :=
  (|Real.sin x| + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x))

theorem solve_equation (x : ℝ) (k : ℤ) :
  (equation x = 2 / Real.sqrt 3) ↔
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) :=
sorry

end solve_equation_l195_195824


namespace goldfish_equal_number_after_n_months_l195_195366

theorem goldfish_equal_number_after_n_months :
  ∃ (n : ℕ), 2 * 4^n = 162 * 3^n ∧ n = 6 :=
by
  sorry

end goldfish_equal_number_after_n_months_l195_195366


namespace proof_problem_l195_195945

-- Define sets A and B according to the given conditions
def A : Set ℝ := { x | x ≥ -1 }
def B : Set ℝ := { x | x > 2 }
def complement_B : Set ℝ := { x | ¬ (x > 2) }  -- Complement of B

-- Remaining intersection expression
def intersect_expr : Set ℝ := { x | x ≥ -1 ∧ x ≤ 2 }

-- Statement to prove
theorem proof_problem : (A ∩ complement_B) = intersect_expr :=
sorry

end proof_problem_l195_195945


namespace isosceles_triangle_perimeter_l195_195742

def is_isosceles_triangle (A B C : ℝ) : Prop :=
  (A = B ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (A = C ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (B = C ∧ A + B > C ∧ A + C > B ∧ B + C > A)

theorem isosceles_triangle_perimeter {A B C : ℝ} 
  (h : is_isosceles_triangle A B C) 
  (h1 : A = 3 ∨ A = 7) 
  (h2 : B = 3 ∨ B = 7) 
  (h3 : C = 3 ∨ C = 7) : 
  A + B + C = 17 := 
sorry

end isosceles_triangle_perimeter_l195_195742


namespace goldfish_cost_graph_is_finite_set_of_points_l195_195525

theorem goldfish_cost_graph_is_finite_set_of_points :
  ∀ (n : ℤ), (1 ≤ n ∧ n ≤ 12) → ∃ (C : ℤ), C = 15 * n ∧ ∀ m ≠ n, C ≠ 15 * m :=
by
  -- The proof goes here
  sorry

end goldfish_cost_graph_is_finite_set_of_points_l195_195525


namespace abs_diff_101st_term_l195_195129

theorem abs_diff_101st_term 
  (C D : ℕ → ℤ)
  (hC_start : C 0 = 20)
  (hD_start : D 0 = 20)
  (hC_diff : ∀ n, C (n + 1) = C n + 12)
  (hD_diff : ∀ n, D (n + 1) = D n - 6) :
  |C 100 - D 100| = 1800 :=
by
  sorry

end abs_diff_101st_term_l195_195129


namespace players_have_five_coins_l195_195017

noncomputable def game_probability : ℚ :=
  let totalWays := (4.choose 2) ^ 3  -- ways to choose who gets green and red in each round (binomial coefficient 4C2)
  let favorableWays := 1             -- only one way to maintain balance across three rounds
  favorableWays / totalWays

theorem players_have_five_coins :
  game_probability = 1 / 46656 := by
  sorry

end players_have_five_coins_l195_195017


namespace isosceles_triangle_perimeter_l195_195744

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3) (h2 : b = 7) : 
∃ (c : ℕ), 
  (c = 7 ∧ a = 3 ∧ b = 7 ∧ a + b + c = 17) ∨ 
  (c = 3 ∧ a = 7 ∧ b = 7 ∧ a + b + c = 17) :=
  sorry

end isosceles_triangle_perimeter_l195_195744


namespace percentage_increase_l195_195975

-- Conditions
variables (S_final S_initial : ℝ) (P : ℝ)
def conditions := (S_final = 3135) ∧ (S_initial = 3000) ∧
  (S_final = (S_initial + (P/100) * S_initial) - 0.05 * (S_initial + (P/100) * S_initial))

-- Statement of the problem
theorem percentage_increase (S_final S_initial : ℝ) 
  (cond : conditions S_final S_initial P) : P = 10 := by
  sorry

end percentage_increase_l195_195975


namespace blake_bought_six_chocolate_packs_l195_195701

-- Defining the conditions as hypotheses
variables (lollipops : ℕ) (lollipopCost : ℕ) (packCost : ℕ)
          (cashGiven : ℕ) (changeReceived : ℕ)
          (totalSpent : ℕ) (totalLollipopCost : ℕ) (amountSpentOnChocolates : ℕ)

-- Assertion of the values based on the conditions
axiom h1 : lollipops = 4
axiom h2 : lollipopCost = 2
axiom h3 : packCost = lollipops * lollipopCost
axiom h4 : cashGiven = 6 * 10
axiom h5 : changeReceived = 4
axiom h6 : totalSpent = cashGiven - changeReceived
axiom h7 : totalLollipopCost = lollipops * lollipopCost
axiom h8 : amountSpentOnChocolates = totalSpent - totalLollipopCost
axiom chocolatePacks : ℕ
axiom h9 : chocolatePacks = amountSpentOnChocolates / packCost

-- The statement to be proved
theorem blake_bought_six_chocolate_packs :
    chocolatePacks = 6 :=
by
  subst_vars
  sorry

end blake_bought_six_chocolate_packs_l195_195701


namespace quadratic_distinct_real_roots_range_quadratic_root_product_value_l195_195061

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → k > 3 / 4 :=
sorry

theorem quadratic_root_product_value (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → (∀ x₁ x₂ : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → x₁ * x₂ = 5) 
  → k = 2 :=
sorry

end quadratic_distinct_real_roots_range_quadratic_root_product_value_l195_195061


namespace cheaper_fluid_cost_is_20_l195_195956

variable (x : ℕ) -- Denote the cost per drum of the cheaper fluid as x

-- Given conditions:
variable (total_drums : ℕ) (cheaper_drums : ℕ) (expensive_cost : ℕ) (total_cost : ℕ)
variable (remaining_drums : ℕ) (total_expensive_cost : ℕ)

axiom total_drums_eq : total_drums = 7
axiom cheaper_drums_eq : cheaper_drums = 5
axiom expensive_cost_eq : expensive_cost = 30
axiom total_cost_eq : total_cost = 160
axiom remaining_drums_eq : remaining_drums = total_drums - cheaper_drums
axiom total_expensive_cost_eq : total_expensive_cost = remaining_drums * expensive_cost

-- The equation for the total cost:
axiom total_cost_eq2 : total_cost = cheaper_drums * x + total_expensive_cost

-- The goal: Prove that the cheaper fluid cost per drum is $20
theorem cheaper_fluid_cost_is_20 : x = 20 :=
by
  sorry

end cheaper_fluid_cost_is_20_l195_195956


namespace shooting_to_practice_ratio_l195_195263

variable (practiceTime minutes weightliftingTime runningTime shootingTime : ℕ)
variable (runningWeightliftingRatio : ℕ)

axiom practiceTime_def : practiceTime = 2 * 60 -- converting 2 hours to minutes
axiom weightliftingTime_def : weightliftingTime = 20
axiom runningWeightliftingRatio_def : runningWeightliftingRatio = 2
axiom runningTime_def : runningTime = runningWeightliftingRatio * weightliftingTime
axiom shootingTime_def : shootingTime = practiceTime - (runningTime + weightliftingTime)

theorem shooting_to_practice_ratio (practiceTime minutes weightliftingTime runningTime shootingTime : ℕ) 
                                   (runningWeightliftingRatio : ℕ) :
  practiceTime = 120 →
  weightliftingTime = 20 →
  runningWeightliftingRatio = 2 →
  runningTime = runningWeightliftingRatio * weightliftingTime →
  shootingTime = practiceTime - (runningTime + weightliftingTime) →
  (shootingTime : ℚ) / practiceTime = 1 / 2 :=
by sorry

end shooting_to_practice_ratio_l195_195263


namespace remainder_7n_mod_4_l195_195313

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l195_195313


namespace find_k_l195_195068

-- Defining the quadratic function
def quadratic (x k : ℝ) := x^2 + (2 * k + 1) * x + k^2 + 1

-- Condition 1: The roots are distinct, implies discriminant > 0
def discriminant_positive (k : ℝ) := (2 * k + 1)^2 - 4 * (k^2 + 1) > 0

-- Condition 2: Product of roots given as 5
def product_of_roots (k : ℝ) := k^2 + 1 = 5

-- Main theorem
theorem find_k (k : ℝ) (hk1 : discriminant_positive k) (hk2 : product_of_roots k) : k = 2 := by
  sorry

end find_k_l195_195068


namespace color_copies_comparison_l195_195043

theorem color_copies_comparison (n : ℕ) (pX pY : ℝ) (charge_diff : ℝ) 
  (h₀ : pX = 1.20) (h₁ : pY = 1.70) (h₂ : charge_diff = 35) 
  (h₃ : pY * n = pX * n + charge_diff) : n = 70 :=
by
  -- proof steps would go here
  sorry

end color_copies_comparison_l195_195043


namespace solve_for_x_l195_195569

theorem solve_for_x (x : ℝ) : (x - 20) / 3 = (4 - 3 * x) / 4 → x = 7.08 := by
  sorry

end solve_for_x_l195_195569


namespace sin_double_angle_l195_195768

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l195_195768


namespace distinct_real_roots_max_abs_gt_2_l195_195536

theorem distinct_real_roots_max_abs_gt_2 
  (r1 r2 r3 q : ℝ)
  (h_distinct : r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3)
  (h_sum : r1 + r2 + r3 = -q)
  (h_product : r1 * r2 * r3 = -9)
  (h_sum_prod : r1 * r2 + r2 * r3 + r3 * r1 = 6)
  (h_nonzero_discriminant : q^2 * 6^2 - 4 * 6^3 - 4 * q^3 * 9 - 27 * 9^2 + 18 * q * 6 * (-9) ≠ 0) :
  max (|r1|) (max (|r2|) (|r3|)) > 2 :=
sorry

end distinct_real_roots_max_abs_gt_2_l195_195536


namespace correctness_of_option_C_l195_195987

-- Define the conditions as hypotheses
variable (x y : ℝ)

def condA : Prop := ∀ x: ℝ, x^3 * x^5 = x^15
def condB : Prop := ∀ x y: ℝ, 2 * x + 3 * y = 5 * x * y
def condC : Prop := ∀ x y: ℝ, 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y
def condD : Prop := ∀ x: ℝ, (x - 2)^2 = x^2 - 4

-- State the proof problem is correct
theorem correctness_of_option_C (x y : ℝ) : 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y := by
  sorry

end correctness_of_option_C_l195_195987


namespace min_value_of_y_l195_195886

theorem min_value_of_y (x : ℝ) : ∃ x0 : ℝ, (∀ x : ℝ, 4 * x^2 + 8 * x + 16 ≥ 12) ∧ (4 * x0^2 + 8 * x0 + 16 = 12) :=
sorry

end min_value_of_y_l195_195886


namespace book_distribution_methods_l195_195835

theorem book_distribution_methods :
  let novels := 2
  let picture_books := 2
  let students := 3
  (number_ways : ℕ) = 12 :=
by
  sorry

end book_distribution_methods_l195_195835


namespace central_angle_of_regular_polygon_l195_195399

theorem central_angle_of_regular_polygon (n : ℕ) (h : 360 ∣ 360 - 36 * n) :
  n = 10 :=
by
  sorry

end central_angle_of_regular_polygon_l195_195399


namespace find_unknown_number_l195_195026

theorem find_unknown_number (x : ℝ) (h : (2 / 3) * x + 6 = 10) : x = 6 :=
  sorry

end find_unknown_number_l195_195026


namespace small_cubes_with_two_faces_painted_l195_195425

theorem small_cubes_with_two_faces_painted :
  let n := 5
  let total_small_cubes := n ^ 3
  let small_cube_edge_length := 1
  let small_cubes_with_two_faces := 12 * (n - 2)
  12 * (n - 2) = 36 :=
by
  let n := 5
  let total_small_cubes := n ^ 3
  let small_cube_edge_length := 1
  let small_cubes_with_two_faces := 12 * (n - 2)
  exact sorry

end small_cubes_with_two_faces_painted_l195_195425


namespace ratio_R_U_l195_195592

theorem ratio_R_U : 
  let spacing := 1 / 4
  let R := 3 * spacing
  let U := 6 * spacing
  R / U = 0.5 := 
by
  sorry

end ratio_R_U_l195_195592


namespace original_fraction_l195_195685

theorem original_fraction (n d : ℝ) (h1 : n + d = 5.25) (h2 : (n + 3) / (2 * d) = 1 / 3) : n / d = 2 / 33 :=
by
  sorry

end original_fraction_l195_195685


namespace find_y_when_x_is_7_l195_195833

theorem find_y_when_x_is_7
  (x y : ℝ)
  (h1 : x * y = 384)
  (h2 : x + y = 40)
  (h3 : x - y = 8)
  (h4 : x = 7) :
  y = 384 / 7 :=
by
  sorry

end find_y_when_x_is_7_l195_195833


namespace bono_jelly_beans_l195_195868

variable (t A B C : ℤ)

theorem bono_jelly_beans (h₁ : A + B = 6 * t + 3) 
                         (h₂ : A + C = 4 * t + 5) 
                         (h₃ : B + C = 6 * t) : 
                         B = 4 * t - 1 := by
  sorry

end bono_jelly_beans_l195_195868


namespace find_first_spill_l195_195959

def bottle_capacity : ℕ := 20
def refill_count : ℕ := 3
def days : ℕ := 7
def total_water_drunk : ℕ := 407
def second_spill : ℕ := 8

theorem find_first_spill :
  let total_without_spill := bottle_capacity * refill_count * days
  let total_spilled := total_without_spill - total_water_drunk
  let first_spill := total_spilled - second_spill
  first_spill = 5 :=
by
  -- Proof goes here.
  sorry

end find_first_spill_l195_195959


namespace range_of_a_increasing_function_l195_195389

noncomputable def f (x a : ℝ) := x^3 + a * x + 1 / x

noncomputable def f' (x a : ℝ) := 3 * x^2 - 1 / x^2 + a

theorem range_of_a_increasing_function (a : ℝ) :
  (∀ x : ℝ, x > 1/2 → f' x a ≥ 0) ↔ a ≥ 13 / 4 := 
sorry

end range_of_a_increasing_function_l195_195389


namespace factorization_problem1_factorization_problem2_l195_195373

-- Mathematical statements
theorem factorization_problem1 (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2 := by
  sorry

theorem factorization_problem2 (a : ℝ) : 18 * a^2 - 50 = 2 * (3 * a + 5) * (3 * a - 5) := by
  sorry

end factorization_problem1_factorization_problem2_l195_195373


namespace circle_equation_l195_195048

theorem circle_equation (x y : ℝ) : (x^2 = 16 * y) → (y = 4) → (x, -4) = (x, 4) → x^2 + (y-4)^2 = 64 :=
by
  sorry

end circle_equation_l195_195048


namespace increase_by_150_percent_l195_195651

theorem increase_by_150_percent (n : ℕ) : 
  n = 80 → n + (3 / 2) * n = 200 :=
by
  intros h
  rw [h]
  norm_num
  sorry

end increase_by_150_percent_l195_195651


namespace defective_rate_20_percent_l195_195200

open_locale big_operators

noncomputable def defective_rate (n : ℕ) : ℝ := n / 10.0

theorem defective_rate_20_percent (p : ℝ) (ξ : ℝ)
  (h1 : p = 16/45)
  (h2 : ξ = 1)
  (h3 : ∀ n ≤ 4, P(ξ = n) ≤ p) : defective_rate 2 = 0.2 :=
by
  sorry

end defective_rate_20_percent_l195_195200


namespace range_of_8x_plus_y_l195_195901

theorem range_of_8x_plus_y (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_condition : 1 / x + 2 / y = 2) : 8 * x + y ≥ 9 :=
by
  sorry

end range_of_8x_plus_y_l195_195901


namespace abe_bob_matching_probability_l195_195349

-- Definitions of the conditions
def AbeJellyBeans := {green := 1, red := 2, total := 3}
def BobJellyBeans := {green := 2, yellow := 1, red := 1, total := 4}

-- Statement of the problem
theorem abe_bob_matching_probability :
  let p_green_abe := (1 / AbeJellyBeans.total : ℚ)
  let p_green_bob := (2 / BobJellyBeans.total : ℚ)
  let p_both_green := p_green_abe * p_green_bob

  let p_red_abe := (2 / AbeJellyBeans.total : ℚ)
  let p_red_bob := (1 / BobJellyBeans.total : ℚ)
  let p_both_red := p_red_abe * p_red_bob

  p_both_green + p_both_red = 1 / 3 :=
begin
  let p_green_abe := (1 / AbeJellyBeans.total : ℚ),
  let p_green_bob := (2 / BobJellyBeans.total : ℚ),
  let p_both_green := p_green_abe * p_green_bob,
  
  let p_red_abe := (2 / AbeJellyBeans.total : ℚ),
  let p_red_bob := (1 / BobJellyBeans.total : ℚ),
  let p_both_red := p_red_abe * p_red_bob,

  have h1 : p_both_green = 1 / 6, by norm_num,
  have h2 : p_both_red = 1 / 6, by norm_num,
  rw [h1, h2],
  norm_num
end

end abe_bob_matching_probability_l195_195349


namespace jill_arrives_30_minutes_before_jack_l195_195282

theorem jill_arrives_30_minutes_before_jack
    (d : ℝ) (s_jill : ℝ) (s_jack : ℝ) (t_diff : ℝ)
    (h_d : d = 2)
    (h_s_jill : s_jill = 12)
    (h_s_jack : s_jack = 3)
    (h_t_diff : t_diff = 30) :
    ((d / s_jack) * 60 - (d / s_jill) * 60) = t_diff :=
by
  sorry

end jill_arrives_30_minutes_before_jack_l195_195282


namespace increase_by_percentage_l195_195598

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l195_195598


namespace calculate_expression_l195_195702

theorem calculate_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end calculate_expression_l195_195702


namespace adam_spent_money_on_ferris_wheel_l195_195699

def tickets_bought : ℕ := 13
def tickets_left : ℕ := 4
def ticket_cost : ℕ := 9
def tickets_used : ℕ := tickets_bought - tickets_left

theorem adam_spent_money_on_ferris_wheel :
  tickets_used * ticket_cost = 81 :=
by
  sorry

end adam_spent_money_on_ferris_wheel_l195_195699


namespace last_digit_sum_l195_195582

theorem last_digit_sum (a b : ℕ) (exp : ℕ)
  (h₁ : a = 1993) (h₂ : b = 1995) (h₃ : exp = 2002) :
  ((a ^ exp + b ^ exp) % 10) = 4 := 
by
  sorry

end last_digit_sum_l195_195582


namespace marcus_baseball_cards_l195_195952

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the number of additional cards Marcus has compared to Carter
def additional_cards : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + additional_cards

-- The proof statement asserting Marcus' total number of baseball cards
theorem marcus_baseball_cards : marcus_cards = 210 :=
by {
  -- This is where the proof steps would go, but we are skipping with sorry
  sorry
}

end marcus_baseball_cards_l195_195952


namespace fraction_area_below_diagonal_is_one_l195_195958

noncomputable def fraction_below_diagonal (s : ℝ) : ℝ := 1

theorem fraction_area_below_diagonal_is_one (s : ℝ) :
  let long_side := 2 * s
  let P := (2 * s / 3, 0)
  let Q := (s, s / 2)
  -- Total area of the rectangle
  let total_area := s * 2 * s -- 2s^2
  -- Total area below the diagonal
  let area_below_diagonal := 2 * s * s  -- 2s^2
  -- Fraction of the area below diagonal
  fraction_below_diagonal s = area_below_diagonal / total_area := 
by 
  sorry

end fraction_area_below_diagonal_is_one_l195_195958


namespace matrix_product_is_zero_l195_195176

def vec3 := (ℝ × ℝ × ℝ)

def M1 (a b c : ℝ) : vec3 × vec3 × vec3 :=
  ((0, 2 * c, -2 * b),
   (-2 * c, 0, 2 * a),
   (2 * b, -2 * a, 0))

def M2 (a b c : ℝ) : vec3 × vec3 × vec3 :=
  ((2 * a^2, a^2 + b^2, a^2 + c^2),
   (a^2 + b^2, 2 * b^2, b^2 + c^2),
   (a^2 + c^2, b^2 + c^2, 2 * c^2))

def matrix_mul (m1 m2 : vec3 × vec3 × vec3) : vec3 × vec3 × vec3 := sorry

theorem matrix_product_is_zero (a b c : ℝ) :
  matrix_mul (M1 a b c) (M2 a b c) = ((0, 0, 0), (0, 0, 0), (0, 0, 0)) := by
  sorry

end matrix_product_is_zero_l195_195176


namespace irreducible_fraction_iff_not_congruent_mod_5_l195_195376

theorem irreducible_fraction_iff_not_congruent_mod_5 (n : ℕ) : 
  (Nat.gcd (21 * n + 4) (14 * n + 1) = 1) ↔ (n % 5 ≠ 1) := 
by 
  sorry

end irreducible_fraction_iff_not_congruent_mod_5_l195_195376


namespace increase_by_150_percent_l195_195662

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l195_195662


namespace alice_wrong_questions_l195_195475

theorem alice_wrong_questions :
  ∃ a b e : ℕ,
    (a + b = 6 + 8 + e) ∧
    (a + 8 = b + 6 + 3) ∧
    a = 9 :=
by {
  sorry
}

end alice_wrong_questions_l195_195475


namespace solve_frac_eq_l195_195120

-- Define the fractional function
def frac_eq (x : ℝ) : Prop := (x + 2) / (x - 1) = 0

-- State the theorem
theorem solve_frac_eq : frac_eq (-2) :=
by
  unfold frac_eq
  -- Use sorry to skip the proof
  sorry

end solve_frac_eq_l195_195120


namespace gcd_372_684_l195_195581

theorem gcd_372_684 : Nat.gcd 372 684 = 12 :=
by
  sorry

end gcd_372_684_l195_195581


namespace t_shirts_in_two_hours_l195_195363

-- Definitions for the conditions
def first_hour_rate : Nat := 12
def second_hour_rate : Nat := 6

-- Main statement to prove
theorem t_shirts_in_two_hours : 
  (60 / first_hour_rate + 60 / second_hour_rate) = 15 := by
  sorry

end t_shirts_in_two_hours_l195_195363


namespace num_integers_D_l195_195029

theorem num_integers_D :
  ∃ (D : ℝ) (n : ℕ), 
    (∀ (a b : ℝ), -1/4 < a → a < 1/4 → -1/4 < b → b < 1/4 → abs (a^2 - D * b^2) < 1) → n = 32 :=
sorry

end num_integers_D_l195_195029


namespace negation_of_p_l195_195540

-- Declare the proposition p as a condition
def p : Prop :=
  ∀ (x : ℝ), 0 ≤ x → x^2 + 4 * x + 3 > 0

-- State the problem
theorem negation_of_p : ¬ p ↔ ∃ (x : ℝ), 0 ≤ x ∧ x^2 + 4 * x + 3 ≤ 0 :=
by
  sorry

end negation_of_p_l195_195540


namespace triangle_ABC_area_l195_195840

def point : Type := ℚ × ℚ

def triangle_area (A B C : point) : ℚ :=
  let (x1, y1) := A;
  let (x2, y2) := B;
  let (x3, y3) := C;
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_ABC_area :
  let A : point := (-5, 4)
  let B : point := (1, 7)
  let C : point := (4, -3)
  triangle_area A B C = 34.5 :=
by
  sorry

end triangle_ABC_area_l195_195840


namespace grandma_mushrooms_l195_195172

theorem grandma_mushrooms (M : ℕ) (h₁ : ∀ t : ℕ, t = 2 * M)
                         (h₂ : ∀ p : ℕ, p = 4 * t)
                         (h₃ : ∀ b : ℕ, b = 4 * p)
                         (h₄ : ∀ r : ℕ, r = b / 3)
                         (h₅ : r = 32) :
  M = 3 :=
by
  -- We are expected to fill the steps here to provide the proof if required
  sorry

end grandma_mushrooms_l195_195172


namespace option_C_correct_l195_195846

theorem option_C_correct : 5 + (-6) - (-7) = 5 - 6 + 7 := 
by
  sorry

end option_C_correct_l195_195846


namespace sin_2phi_l195_195770

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l195_195770


namespace quadratic_distinct_real_roots_l195_195924

theorem quadratic_distinct_real_roots (k : ℝ) :
  ((k - 1) * x^2 + 6 * x + 3 = 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k - 1) * x1^2 + 6 * x1 + 3 = 0) ∧ ((k - 1) * x2^2 + 6 * x2 + 3 = 0)) ↔ (k < 4 ∧ k ≠ 1) :=
by {
  sorry
}

end quadratic_distinct_real_roots_l195_195924


namespace train_half_speed_time_l195_195691

-- Definitions for Lean
variables (S T D : ℝ)

-- Conditions
axiom cond1 : D = S * T
axiom cond2 : D = (1 / 2) * S * (T + 4)

-- Theorem Statement
theorem train_half_speed_time : 
  (T = 4) → (4 + 4 = 8) := 
by 
  intros hT
  simp [hT]

end train_half_speed_time_l195_195691


namespace circles_intersect_l195_195585

section PositionalRelationshipCircles

-- Define the first circle O1 with center (1, 0) and radius 1
def Circle1 (p : ℝ × ℝ) : Prop := (p.1 - 1)^2 + p.2^2 = 1

-- Define the second circle O2 with center (0, 3) and radius 3
def Circle2 (p : ℝ × ℝ) : Prop := p.1^2 + (p.2 - 3)^2 = 9

-- Prove that the positional relationship between Circle1 and Circle2 is intersecting
theorem circles_intersect : 
  ∃ p : ℝ × ℝ, Circle1 p ∧ Circle2 p :=
sorry

end PositionalRelationshipCircles

end circles_intersect_l195_195585


namespace urn_problem_l195_195168

theorem urn_problem : 
  (5 / 12 * 20 / (20 + M) + 7 / 12 * M / (20 + M) = 0.62) → M = 111 :=
by
  intro h
  sorry

end urn_problem_l195_195168


namespace increase_80_by_150_percent_l195_195616

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l195_195616


namespace hyperbola_standard_equation_l195_195726

theorem hyperbola_standard_equation :
  (∀ (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
    (h_eccentricity : (sqrt (1 + (b/a)^2)) = sqrt 5),
    ∃ x y : ℝ,
    let C1 := (λ x : ℝ, x^2 = 2*y) in
    let C2 := (λ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) in
    let A := (a, 0) in
    let asymptote := (λ x : ℝ, y = (b/a)*(x - a)) in
    (∀ x, is_tangent at x C1 asymptote) →  (C2 x y) → (x^2 - y^2 / 4 = 1) ) :=
sorry

end hyperbola_standard_equation_l195_195726


namespace MaryHasBlueMarbles_l195_195884

-- Define the number of blue marbles Dan has
def DanMarbles : Nat := 5

-- Define the relationship of Mary's marbles to Dan's marbles
def MaryMarbles : Nat := 2 * DanMarbles

-- State the theorem that we need to prove
theorem MaryHasBlueMarbles : MaryMarbles = 10 :=
by
  sorry

end MaryHasBlueMarbles_l195_195884


namespace integer_solutions_eq_400_l195_195526

theorem integer_solutions_eq_400 : 
  ∃ (s : Finset (ℤ × ℤ)), (∀ x y, (x, y) ∈ s ↔ |3 * x + 2 * y| + |2 * x + y| = 100) ∧ s.card = 400 :=
sorry

end integer_solutions_eq_400_l195_195526


namespace find_m_value_l195_195078

theorem find_m_value (m : ℝ) (h : (m - 4)^2 + 1^2 + 2^2 = 30) : m = 9 ∨ m = -1 :=
by {
  sorry
}

end find_m_value_l195_195078


namespace minimum_cups_needed_l195_195687

theorem minimum_cups_needed (container_capacity cup_capacity : ℕ) (h1 : container_capacity = 980) (h2 : cup_capacity = 80) : 
  Nat.ceil (container_capacity / cup_capacity : ℚ) = 13 :=
by
  sorry

end minimum_cups_needed_l195_195687


namespace sin_double_angle_l195_195232

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ + 1 / Real.tan θ = 4) : Real.sin (2 * θ) = 1 / 2 :=
by
  sorry

end sin_double_angle_l195_195232


namespace apples_in_each_bag_l195_195150

variable (x : ℕ)
variable (total_children : ℕ)
variable (eaten_apples : ℕ)
variable (sold_apples : ℕ)
variable (remaining_apples : ℕ)

theorem apples_in_each_bag
  (h1 : total_children = 5)
  (h2 : eaten_apples = 2 * 4)
  (h3 : sold_apples = 7)
  (h4 : remaining_apples = 60)
  (h5 : total_children * x - eaten_apples - sold_apples = remaining_apples) :
  x = 15 :=
by
  sorry

end apples_in_each_bag_l195_195150


namespace work_finished_earlier_due_to_additional_men_l195_195016

-- Define the conditions as given facts in Lean
def original_men := 10
def original_days := 12
def additional_men := 10

-- State the theorem to be proved
theorem work_finished_earlier_due_to_additional_men :
  let total_men := original_men + additional_men
  let original_work := original_men * original_days
  let days_earlier := original_days - x
  original_work = total_men * days_earlier → x = 6 :=
by
  sorry

end work_finished_earlier_due_to_additional_men_l195_195016


namespace parabola_translation_correct_l195_195452

noncomputable def translate_parabola (x y : ℝ) (h : y = -2 * x^2 - 4 * x - 6) : Prop :=
  let x' := x - 1
  let y' := y + 3
  y' = -2 * x'^2 - 1

theorem parabola_translation_correct (x y : ℝ) (h : y = -2 * x^2 - 4 * x - 6) :
  translate_parabola x y h :=
sorry

end parabola_translation_correct_l195_195452


namespace center_number_is_4_l195_195869

-- Define the numbers and the 3x3 grid
inductive Square
| center | top_middle | left_middle | right_middle | bottom_middle

-- Define the properties of the problem
def isConsecutiveAdjacent (a b : ℕ) : Prop := 
  (a + 1 = b ∨ a = b + 1)

-- The condition to check the sum of edge squares
def sum_edge_squares (grid : Square → ℕ) : Prop := 
  grid Square.top_middle + grid Square.left_middle + grid Square.right_middle + grid Square.bottom_middle = 28

-- The condition that the center square number is even
def even_center (grid : Square → ℕ) : Prop := 
  grid Square.center % 2 = 0

-- The main theorem statement
theorem center_number_is_4 (grid : Square → ℕ) :
  (∀ i j : Square, i ≠ j → isConsecutiveAdjacent (grid i) (grid j)) → 
  (grid Square.top_middle + grid Square.left_middle + grid Square.right_middle + grid Square.bottom_middle = 28) →
  (grid Square.center % 2 = 0) →
  grid Square.center = 4 :=
by sorry

end center_number_is_4_l195_195869


namespace boat_distance_against_water_flow_l195_195467

variable (a : ℝ) -- speed of the boat in still water

theorem boat_distance_against_water_flow 
  (speed_boat_still_water : ℝ := a)
  (speed_water_flow : ℝ := 3)
  (time_travel : ℝ := 3) :
  (speed_boat_still_water - speed_water_flow) * time_travel = 3 * (a - 3) := 
by
  sorry

end boat_distance_against_water_flow_l195_195467


namespace sin_double_angle_solution_l195_195783

theorem sin_double_angle_solution (φ : ℝ) 
  (h : (7 / 13) + Real.sin φ = Real.cos φ) : 
  Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_solution_l195_195783


namespace find_expression_l195_195055

-- Define the polynomial and the fact that a and b are roots.
def poly (x : ℝ) := x^2 + 3 * x - 4

-- Assuming a and b are roots of the polynomial
variables (a b : ℝ)
hypothesis h_a_root : poly a = 0
hypothesis h_b_root : poly b = 0

-- Prove that a^2 + 4a + b - 3 = -2 given the above assumptions
theorem find_expression (a b : ℝ) (h_a_root : poly a = 0) (h_b_root : poly b = 0) : a^2 + 4 * a + b - 3 = -2 :=
by sorry

end find_expression_l195_195055


namespace abs_diff_roots_eq_3_l195_195037

theorem abs_diff_roots_eq_3 : ∀ (r1 r2 : ℝ), (r1 ≠ r2) → (r1 + r2 = 7) → (r1 * r2 = 10) → |r1 - r2| = 3 :=
by
  intros r1 r2 hneq hsum hprod
  sorry

end abs_diff_roots_eq_3_l195_195037


namespace correct_propositions_l195_195476

noncomputable def proposition1 : Prop :=
  (∀ x : ℝ, x^2 - 3 * x + 2 = 0 -> x = 1) ->
  (∀ x : ℝ, x ≠ 1 -> x^2 - 3 * x + 2 ≠ 0)

noncomputable def proposition2 : Prop :=
  (∀ p q : Prop, p ∨ q -> p ∧ q) ->
  (∀ p q : Prop, p ∧ q -> p ∨ q)

noncomputable def proposition3 : Prop :=
  (∀ p q : Prop, ¬(p ∧ q) -> ¬p ∧ ¬q)

noncomputable def proposition4 : Prop :=
  (∃ x : ℝ, x^2 + x + 1 < 0) ->
  (∀ x : ℝ, x^2 + x + 1 ≥ 0)

theorem correct_propositions :
  proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l195_195476


namespace remainder_of_7n_mod_4_l195_195297

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l195_195297


namespace fraction_members_absent_l195_195252

variable (p : ℕ) -- Number of persons in the office
variable (W : ℝ) -- Total work amount
variable (x : ℝ) -- Fraction of members absent

theorem fraction_members_absent (h : W / (p * (1 - x)) = W / p + W / (6 * p)) : x = 1 / 7 :=
by
  sorry

end fraction_members_absent_l195_195252


namespace increase_by_150_percent_l195_195663

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l195_195663


namespace find_AB_l195_195595

theorem find_AB
  (r R : ℝ)
  (h : r < R) :
  ∃ AB : ℝ, AB = (4 * r * (Real.sqrt (R * r))) / (R + r) :=
by
  sorry

end find_AB_l195_195595


namespace bags_of_cookies_l195_195927

theorem bags_of_cookies (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 33) (h2 : cookies_per_bag = 11) : total_cookies / cookies_per_bag = 3 :=
by
  sorry

end bags_of_cookies_l195_195927


namespace circus_dogs_ratio_l195_195171

theorem circus_dogs_ratio :
  ∀ (x y : ℕ), 
  (x + y = 12) → (2 * x + 4 * y = 36) → (x = y) → x / y = 1 :=
by
  intros x y h1 h2 h3
  sorry

end circus_dogs_ratio_l195_195171


namespace pascal_triangle_contains_47_l195_195208

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l195_195208


namespace increase_80_by_150_percent_l195_195655

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l195_195655


namespace probability_10_products_expected_value_of_products_l195_195003

open ProbabilityTheory

/-- Probability calculations for worker assessment. -/
noncomputable def worker_assessment_probability (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p^9 * (10 - 9*p)

/-- Expected value of total products produced and debugged by Worker A -/
noncomputable def expected_products (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  20 - 10*p - 10*p^9 + 10*p^10

/-- Theorem 1: Prove that the probability that Worker A ends the assessment by producing only 10 products is p^9(10 - 9p). -/
theorem probability_10_products (p : ℝ) (h : 0 < p ∧ p < 1) :
  worker_assessment_probability p h = p^9 * (10 - 9*p) := by
  sorry

/-- Theorem 2: Prove the expected value E(X) of the total number of products produced and debugged by Worker A is 20 - 10p - 10p^9 + 10p^{10}. -/
theorem expected_value_of_products (p : ℝ) (h : 0 < p ∧ p < 1) :
  expected_products p h = 20 - 10*p - 10*p^9 + 10*p^10 := by
  sorry

end probability_10_products_expected_value_of_products_l195_195003


namespace solve_inequality_l195_195486

theorem solve_inequality :
  {x : ℝ | x^2 - 9 * x + 14 < 0} = {x : ℝ | 2 < x ∧ x < 7} := sorry

end solve_inequality_l195_195486


namespace probability_multiple_of_3_or_4_l195_195973

theorem probability_multiple_of_3_or_4 : ((15 : ℚ) / 30) = (1 / 2) := by
  sorry

end probability_multiple_of_3_or_4_l195_195973


namespace ratio_of_discounted_bricks_l195_195451

theorem ratio_of_discounted_bricks (total_bricks discounted_price full_price total_spending: ℝ) 
  (h1 : total_bricks = 1000) 
  (h2 : discounted_price = 0.25) 
  (h3 : full_price = 0.50) 
  (h4 : total_spending = 375) : 
  ∃ D : ℝ, (D / total_bricks = 1 / 2) ∧ (0.25 * D + 0.50 * (total_bricks - D) = total_spending) := 
  sorry

end ratio_of_discounted_bricks_l195_195451


namespace carlson_fraction_jam_l195_195800

-- Definitions and conditions.
def total_time (T : ℕ) := T > 0
def time_maloish_cookies (t : ℕ) := t > 0
def equal_cookies (c : ℕ) := c > 0
def carlson_rate := 3

-- Let j_k and j_m be the amounts of jam eaten by Carlson and Maloish respectively.
def fraction_jam_carlson (j_k j_m : ℕ) : ℚ := j_k / (j_k + j_m)

-- The problem statement
theorem carlson_fraction_jam (T t c j_k j_m : ℕ)
  (hT : total_time T)
  (ht : time_maloish_cookies t)
  (hc : equal_cookies c)
  (h_carlson_rate : carlson_rate = 3)
  (h_equal_cookies : c > 0)  -- Both ate equal cookies
  (h_jam : j_k + j_m = j_k * 9 / 10 + j_m / 10) :
  fraction_jam_carlson j_k j_m = 9 / 10 :=
by
  sorry

end carlson_fraction_jam_l195_195800


namespace smallest_number_bob_l195_195867

-- Define the conditions given in the problem
def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_factors (x : ℕ) : Set ℕ := { p | is_prime p ∧ p ∣ x }

-- The problem statement
theorem smallest_number_bob (b : ℕ) (h1 : prime_factors 30 = prime_factors b) : b = 30 :=
by
  sorry

end smallest_number_bob_l195_195867


namespace increased_number_l195_195607

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l195_195607


namespace rise_in_water_level_l195_195670

theorem rise_in_water_level : 
  let edge_length : ℝ := 15
  let volume_cube : ℝ := edge_length ^ 3
  let length : ℝ := 20
  let width : ℝ := 15
  let base_area : ℝ := length * width
  let rise_in_level : ℝ := volume_cube / base_area
  rise_in_level = 11.25 :=
by
  sorry

end rise_in_water_level_l195_195670


namespace sin_2phi_l195_195773

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l195_195773


namespace min_value_expression_l195_195382

open Classical

theorem min_value_expression (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h : 1/x + 1/y = 1) :
  ∃ (m : ℝ), m = 25 ∧ ∀ x y : ℝ, 0 < x → 0 < y → 1/x + 1/y = 1 → (4*x/(x - 1) + 9*y/(y - 1)) ≥ m :=
by 
  sorry

end min_value_expression_l195_195382


namespace luca_loss_years_l195_195700

variable (months_in_year : ℕ := 12)
variable (barbi_kg_per_month : ℚ := 1.5)
variable (luca_kg_per_year : ℚ := 9)
variable (luca_additional_kg : ℚ := 81)

theorem luca_loss_years (barbi_yearly_loss : ℚ :=
                          barbi_kg_per_month * months_in_year) :
  (81 + barbi_yearly_loss) / luca_kg_per_year = 11 := by
  let total_loss_by_luca := 81 + barbi_yearly_loss
  sorry

end luca_loss_years_l195_195700


namespace min_sum_of_distances_eqn_l195_195513

open Real

def parabola (x y : ℝ) : Prop := y^2 = 2 * x

def point_D : ℝ × ℝ := (2, (3 / 2) * sqrt 3)

def distance_to_y_axis (P : ℝ × ℝ) : ℝ := abs P.1

def distance (P Q : ℝ × ℝ) : ℝ := sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

noncomputable def min_distance_sum : ℝ :=
  let f : ℝ × ℝ := (1/2, 0)
  let sum_distance (P : ℝ × ℝ) := distance P point_D + distance_to_y_axis P
  let min_distance := min (sum_distance f) ((distance f point_D) + distance_to_y_axis f - 1/2)
  min_distance

theorem min_sum_of_distances_eqn :
    ∃ (P : ℝ × ℝ), parabola P.1 P.2 ∧
    min_distance_sum = 5 / 2 :=
sorry

end min_sum_of_distances_eqn_l195_195513


namespace sequence_periodic_l195_195508

theorem sequence_periodic (a : ℕ → ℚ) (h1 : a 1 = 4 / 5)
  (h2 : ∀ n, 0 ≤ a n ∧ a n ≤ 1 → 
    (a (n + 1) = if a n ≤ 1 / 2 then 2 * a n else 2 * a n - 1)) :
  a 2017 = 4 / 5 :=
sorry

end sequence_periodic_l195_195508


namespace part_a_l195_195852

theorem part_a (p : ℕ → ℕ → ℝ) (m : ℕ) (hm : m ≥ 1) : p m 0 = (3 / 4) * p (m-1) 0 + (1 / 2) * p (m-1) 2 + (1 / 8) * p (m-1) 4 :=
by
  sorry

end part_a_l195_195852


namespace increase_80_by_150_percent_l195_195635

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l195_195635


namespace calculate_expression_l195_195022

theorem calculate_expression : 
  ∀ (x y z : ℤ), x = 2 → y = -3 → z = 7 → (x^2 + y^2 + z^2 - 2 * x * y) = 74 :=
by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  sorry

end calculate_expression_l195_195022


namespace _l195_195238

/-- This theorem states that if the GCD of 8580 and 330 is diminished by 12, the result is 318. -/
example : (Int.gcd 8580 330) - 12 = 318 :=
by
  sorry

end _l195_195238


namespace man_l195_195671

variable (V_m V_c : ℝ)

theorem man's_speed_against_current :
  (V_m + V_c = 21 ∧ V_c = 2.5) → (V_m - V_c = 16) :=
by
  sorry

end man_l195_195671


namespace difference_between_balls_l195_195579

theorem difference_between_balls (B R : ℕ) (h1 : R - 152 = B + 152 + 346) : R - B = 650 := 
sorry

end difference_between_balls_l195_195579


namespace chord_length_is_correct_l195_195080

noncomputable def length_of_chord {ρ θ : Real} 
 (h_line : ρ * Real.sin (π / 6 - θ) = 2) 
 (h_curve : ρ = 4 * Real.cos θ) : Real :=
  2 * Real.sqrt 3

theorem chord_length_is_correct {ρ θ : Real} 
 (h_line : ρ * Real.sin (π / 6 - θ) = 2) 
 (h_curve : ρ = 4 * Real.cos θ) : 
 length_of_chord h_line h_curve = 2 * Real.sqrt 3 :=
sorry

end chord_length_is_correct_l195_195080


namespace log_sequence_equality_l195_195507

theorem log_sequence_equality (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n + 1) = a n + 1) (h2: a 2 + a 4 + a 6 = 18) : 
  Real.logb 3 (a 5 + a 7 + a 9) = 3 := 
by
  sorry

end log_sequence_equality_l195_195507


namespace rose_bush_cost_correct_l195_195478

-- Definitions of the given conditions
def total_rose_bushes : ℕ := 20
def gardener_rate : ℕ := 30
def gardener_hours_per_day : ℕ := 5
def gardener_days : ℕ := 4
def gardener_cost : ℕ := gardener_rate * gardener_hours_per_day * gardener_days
def soil_cubic_feet : ℕ := 100
def soil_cost_per_cubic_foot : ℕ := 5
def soil_cost : ℕ := soil_cubic_feet * soil_cost_per_cubic_foot
def total_cost : ℕ := 4100

-- Result computed given the conditions
def rose_bush_cost : ℕ := 150

-- The proof goal (statement only, no proof)
theorem rose_bush_cost_correct : 
  total_cost - gardener_cost - soil_cost = total_rose_bushes * rose_bush_cost :=
by
  sorry

end rose_bush_cost_correct_l195_195478


namespace x_midpoint_of_MN_l195_195050

-- Definition: Given the parabola y^2 = 4x
def parabola (y x : ℝ) : Prop := y^2 = 4 * x

-- Definition: Point F is the focus of the parabola y^2 = 4x
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

-- Definition: Points M and N are on the parabola
def on_parabola (M N : ℝ × ℝ) : Prop :=
  parabola M.2 M.1 ∧ parabola N.2 N.1

-- Definition: The sum of distances |MF| + |NF| = 6
def sum_of_distances (M N : ℝ × ℝ) (F : ℝ × ℝ) : Prop :=
  dist M F + dist N F = 6

-- Theorem: Prove that the x-coordinate of the midpoint of MN is 2
theorem x_midpoint_of_MN (M N : ℝ × ℝ) (F : ℝ × ℝ) 
  (hF : focus F) (hM_N : on_parabola M N) (hDist : sum_of_distances M N F) :
  (M.1 + N.1) / 2 = 2 :=
sorry

end x_midpoint_of_MN_l195_195050


namespace wholesale_price_of_milk_l195_195860

theorem wholesale_price_of_milk (W : ℝ) 
  (h1 : ∀ p : ℝ, p = 1.25 * W) 
  (h2 : ∀ q : ℝ, q = 0.95 * (1.25 * W)) 
  (h3 : q = 4.75) :
  W = 4 :=
by
  sorry

end wholesale_price_of_milk_l195_195860


namespace complex_values_l195_195912

open Complex

theorem complex_values (z : ℂ) (h : z ^ 3 + z = 2 * (abs z) ^ 2) :
  z = 0 ∨ z = 1 ∨ z = -1 + 2 * Complex.I ∨ z = -1 - 2 * Complex.I :=
by sorry

end complex_values_l195_195912


namespace value_of_z_l195_195236

theorem value_of_z {x y z : ℤ} (h1 : x = 2) (h2 : y = x^2 - 5) (h3 : z = y^2 - 5) : z = -4 := by
  sorry

end value_of_z_l195_195236


namespace school_spent_total_amount_l195_195012

theorem school_spent_total_amount
  (num_cartons_pencils : ℕ)
  (boxes_per_carton_pencils : ℕ)
  (cost_per_box_pencils : ℕ)
  (num_cartons_markers : ℕ)
  (boxes_per_carton_markers : ℕ)
  (cost_per_box_markers : ℕ)
  (total_spent : ℕ)
  (h1 : num_cartons_pencils = 20)
  (h2 : boxes_per_carton_pencils = 10)
  (h3 : cost_per_box_pencils = 2)
  (h4 : num_cartons_markers = 10)
  (h5 : boxes_per_carton_markers = 5)
  (h6 : cost_per_box_markers = 4)
  (h7 : total_spent = 
        (num_cartons_pencils * boxes_per_carton_pencils * cost_per_box_pencils)
        + (num_cartons_markers * boxes_per_carton_markers * cost_per_box_markers)) :
  total_spent = 600 :=
by
  rw [h1, h2, h3, h4, h5, h6] at h7
  exact h7.mpr rfl

end school_spent_total_amount_l195_195012


namespace pascal_row_contains_prime_47_l195_195228

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l195_195228


namespace min_gloves_proof_l195_195968

-- Let n represent the number of participants
def n : Nat := 63

-- Let g represent the number of gloves per participant
def g : Nat := 2

-- The minimum number of gloves required
def min_gloves : Nat := n * g

theorem min_gloves_proof : min_gloves = 126 :=
by 
  -- Placeholder for the proof
  sorry

end min_gloves_proof_l195_195968


namespace units_digit_fraction_l195_195845

theorem units_digit_fraction :
  let n := 30 * 31 * 32 * 33 * 34 * 35
  let d := 1500
  let fraction := n / d
  (fraction % 10) = 4 := by
  let n := 30 * 31 * 32 * 33 * 34 * 35
  let d := 1500
  let fraction := n / d
  have h₁ : 30 = 2 * 3 * 5 := by rfl
  have h₂ : 31 = 31 := by rfl
  have h₃ : 32 = 2 ^ 5 := by rfl
  have h₄ : 33 = 3 * 11 := by rfl
  have h₅ : 34 = 2 * 17 := by rfl
  have h₆ : 35 = 5 * 7 := by rfl
  have h₇ : 1500 = 2 ^ 2 * 3 * 5 ^ 3 := by rfl
  have num_factorization : n = 2 ^ 7 * 3 ^ 2 * 5 ^ 2 * 31 * 11 * 17 * 7 := by
    rw [← h₁, ← h₂, ← h₃, ← h₄, ← h₅, ← h₆]
    ring
  have den_factorization : d = 2 ^ 2 * 3 * 5 ^ 3 := by rw h₇
  have simplified_fraction : fraction = 2 ^ 5 * 3 * 31 * 11 * 17 * 7 := by
    rw [num_factorization, den_factorization]
    field_simp
    ring
  have : (2 ^ 5 * 3 * 31 * 11 * 17 * 7 % 10) = 4 := by sorry
  exact this

end units_digit_fraction_l195_195845


namespace inequality_solution_l195_195118

theorem inequality_solution (x : ℝ) : x^2 - 2 * x - 5 > 2 * x ↔ x > 5 ∨ x < -1 :=
by
  sorry

end inequality_solution_l195_195118


namespace remainder_when_7n_divided_by_4_l195_195300

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l195_195300


namespace rhombus_area_l195_195141

/-
  We want to prove that the area of a rhombus with given diagonals' lengths is 
  equal to the computed value according to the formula Area = (d1 * d2) / 2.
-/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) : 
  (d1 * d2) / 2 = 60 :=
by
  rw [h1, h2]
  sorry

end rhombus_area_l195_195141


namespace increase_80_by_150_percent_l195_195637

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l195_195637


namespace sam_age_l195_195748

theorem sam_age (drew_current_age : ℕ) (drew_future_age : ℕ) (sam_future_age : ℕ) : 
  (drew_current_age = 12) → 
  (drew_future_age = drew_current_age + 5) → 
  (sam_future_age = 3 * drew_future_age) → 
  (sam_future_age - 5 = 46) := 
by sorry

end sam_age_l195_195748


namespace find_smaller_number_l195_195596

theorem find_smaller_number (x y : ℤ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 :=
by
  sorry

end find_smaller_number_l195_195596


namespace sin_double_angle_l195_195764

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l195_195764


namespace a_square_plus_one_over_a_square_l195_195198

theorem a_square_plus_one_over_a_square (a : ℝ) (h : a + 1/a = 5) : a^2 + 1/a^2 = 23 :=
by 
  sorry

end a_square_plus_one_over_a_square_l195_195198


namespace min_number_of_trials_sum_15_min_number_of_trials_sum_at_least_15_l195_195391

noncomputable def min_trials_sum_of_15 : ℕ :=
  15

noncomputable def min_trials_sum_at_least_15 : ℕ :=
  8

theorem min_number_of_trials_sum_15 (x : ℕ) :
  (∀ (x : ℕ), (103/108 : ℝ)^x < (1/2 : ℝ) → x >= min_trials_sum_of_15) := sorry

theorem min_number_of_trials_sum_at_least_15 (x : ℕ) :
  (∀ (x : ℕ), (49/54 : ℝ)^x < (1/2 : ℝ) → x >= min_trials_sum_at_least_15) := sorry

end min_number_of_trials_sum_15_min_number_of_trials_sum_at_least_15_l195_195391


namespace area_of_TURS_eq_area_of_PQRS_l195_195328

-- Definition of the rectangle PQRS
structure Rectangle where
  length : ℕ
  width : ℕ
  area : ℕ

-- Definition of the trapezoid TURS
structure Trapezoid where
  base1 : ℕ
  base2 : ℕ
  height : ℕ
  area : ℕ

-- Condition: PQRS is a rectangle whose area is 20 square units
def PQRS : Rectangle := { length := 5, width := 4, area := 20 }

-- Question: Prove the area of TURS equals area of PQRS
theorem area_of_TURS_eq_area_of_PQRS (TURS_area : ℕ) : TURS_area = PQRS.area :=
  sorry

end area_of_TURS_eq_area_of_PQRS_l195_195328


namespace sum_of_coefficients_l195_195716

theorem sum_of_coefficients (a : Fin 7 → ℕ) (x : ℕ) : 
  (1 - x) ^ 6 = (a 0) + (a 1) * x + (a 2) * x^2 + (a 3) * x^3 + (a 4) * x^4 + (a 5) * x^5 + (a 6) * x^6 → 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 0 := 
by
  intro h
  by_cases hx : x = 1
  · rw [hx] at h
    sorry
  · sorry

end sum_of_coefficients_l195_195716


namespace compare_a_x_l195_195423

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem compare_a_x (x a b : ℝ) (h1 : a = log_base 5 (3^x + 4^x))
                    (h2 : b = log_base 4 (5^x - 3^x)) (h3 : a ≥ b) : x ≤ a :=
by
  sorry

end compare_a_x_l195_195423


namespace least_number_remainder_l195_195710

theorem least_number_remainder (n : ℕ) (h1 : n % 34 = 4) (h2 : n % 5 = 4) : n = 174 :=
  sorry

end least_number_remainder_l195_195710


namespace compute_permutation_eq_4_l195_195177

def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem compute_permutation_eq_4 :
  (4 * permutation 8 4 + 2 * permutation 8 5) / (permutation 8 6 - permutation 9 5) * 1 = 4 :=
by
  sorry

end compute_permutation_eq_4_l195_195177


namespace remainder_of_7n_mod_4_l195_195299

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l195_195299


namespace triangle_no_two_obtuse_angles_l195_195135

theorem triangle_no_two_obtuse_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90) (h3 : B > 90) (h4 : C > 0) : false :=
by
  sorry

end triangle_no_two_obtuse_angles_l195_195135


namespace remainder_t4_mod7_l195_195185

def T : ℕ → ℕ
| 0 => 0 -- Not used
| 1 => 6
| n+1 => 6 ^ (T n)

theorem remainder_t4_mod7 : (T 4 % 7) = 6 := by
  sorry

end remainder_t4_mod7_l195_195185


namespace Jeremy_age_l195_195696

noncomputable def A : ℝ := sorry
noncomputable def J : ℝ := sorry
noncomputable def C : ℝ := sorry

-- Conditions
axiom h1 : A + J + C = 132
axiom h2 : A = (1/3) * J
axiom h3 : C = 2 * A

-- The goal is to prove J = 66
theorem Jeremy_age : J = 66 :=
sorry

end Jeremy_age_l195_195696


namespace balls_in_boxes_l195_195527

theorem balls_in_boxes :
    (Nat.choose 6 6) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4 * Nat.choose 2 2) + 
    (Nat.choose 6 3 / 2) + 
    (Nat.choose 6 4 * Nat.choose 2 1) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 3.factorial) = 77 :=
by
  sorry

end balls_in_boxes_l195_195527


namespace isosceles_triangle_perimeter_l195_195743

def is_isosceles_triangle (A B C : ℝ) : Prop :=
  (A = B ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (A = C ∧ A + B > C ∧ A + C > B ∧ B + C > A) ∨
  (B = C ∧ A + B > C ∧ A + C > B ∧ B + C > A)

theorem isosceles_triangle_perimeter {A B C : ℝ} 
  (h : is_isosceles_triangle A B C) 
  (h1 : A = 3 ∨ A = 7) 
  (h2 : B = 3 ∨ B = 7) 
  (h3 : C = 3 ∨ C = 7) : 
  A + B + C = 17 := 
sorry

end isosceles_triangle_perimeter_l195_195743


namespace math_equivalence_proof_problem_l195_195130

-- Define the initial radii in L0
def r1 := 50^2
def r2 := 53^2

-- Define the formula for constructing a new circle in subsequent layers
def next_radius (r1 r2 : ℕ) : ℕ :=
  (r1 * r2) / ((Nat.sqrt r1 + Nat.sqrt r2)^2)

-- Compute the sum of reciprocals of the square roots of the radii 
-- of all circles up to and including layer L6
def sum_of_reciprocals_of_square_roots_up_to_L6 : ℚ :=
  let initial_sum := (1 / (50 : ℚ)) + (1 / (53 : ℚ))
  (127 * initial_sum) / (50 * 53)

theorem math_equivalence_proof_problem : 
  sum_of_reciprocals_of_square_roots_up_to_L6 = 13021 / 2650 := 
sorry

end math_equivalence_proof_problem_l195_195130


namespace base10_to_base8_conversion_l195_195879

theorem base10_to_base8_conversion (n : ℕ) (h₁ : n = 512) : nat.to_digits 8 n = [1, 0, 0, 0] :=
by {
  rw h₁,
  simp,
  sorry
}

end base10_to_base8_conversion_l195_195879


namespace increase_by_percentage_l195_195622

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l195_195622


namespace four_racers_meet_l195_195404

/-- In a circular auto race, four racers participate. Their cars start simultaneously from 
the same point and move at constant speeds, and for any three cars, there is a moment 
when they meet. Prove that after the start of the race, there will be a moment when all 
four cars meet. (Assume the race continues indefinitely in time.) -/
theorem four_racers_meet (V1 V2 V3 V4 : ℝ) (L : ℝ) (t : ℝ) 
  (h1 : 0 ≤ t) 
  (h2 : V1 ≤ V2 ∧ V2 ≤ V3 ∧ V3 ≤ V4)
  (h3 : ∀ t1 t2 t3, ∃ t, t1 * V1 = t ∧ t2 * V2 = t ∧ t3 * V3 = t) :
  ∃ t, t > 0 ∧ ∃ t', V1 * t' % L = 0 ∧ V2 * t' % L = 0 ∧ V3 * t' % L = 0 ∧ V4 * t' % L = 0 :=
sorry

end four_racers_meet_l195_195404


namespace polynomial_decomposition_l195_195593

-- Define the given polynomial
def P (x y z : ℝ) : ℝ := x^2 + 2*x*y + 5*y^2 - 6*x*z - 22*y*z + 16*z^2

-- Define the target decomposition
def Q (x y z : ℝ) : ℝ := (x + (y - 3*z))^2 + (2*y - 4*z)^2 - (3*z)^2

theorem polynomial_decomposition (x y z : ℝ) : P x y z = Q x y z :=
  sorry

end polynomial_decomposition_l195_195593


namespace determinant_of_2x2_matrix_l195_195173

theorem determinant_of_2x2_matrix :
  let a := 2
  let b := 4
  let c := 1
  let d := 3
  a * d - b * c = 2 := by
  sorry

end determinant_of_2x2_matrix_l195_195173


namespace students_not_enrolled_in_either_l195_195247

-- Definitions based on conditions
def total_students : ℕ := 120
def french_students : ℕ := 65
def german_students : ℕ := 50
def both_courses_students : ℕ := 25

-- The proof statement
theorem students_not_enrolled_in_either : total_students - (french_students + german_students - both_courses_students) = 30 := by
  sorry

end students_not_enrolled_in_either_l195_195247


namespace slope_range_l195_195052

theorem slope_range (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ k : ℝ, k = (y + 2) / (x + 1) ∧ k ∈ Set.Ici (3 / 4) :=
sorry

end slope_range_l195_195052


namespace boat_distance_against_stream_l195_195407

-- Define the speed of the boat in still water
def speed_boat_still : ℝ := 8

-- Define the distance covered by the boat along the stream in one hour
def distance_along_stream : ℝ := 11

-- Define the time duration for the journey
def time_duration : ℝ := 1

-- Define the speed of the stream
def speed_stream : ℝ := distance_along_stream - speed_boat_still

-- Define the speed of the boat against the stream
def speed_against_stream : ℝ := speed_boat_still - speed_stream

-- Define the distance covered by the boat against the stream in one hour
def distance_against_stream (t : ℝ) : ℝ := speed_against_stream * t

-- The main theorem: The boat travels 5 km against the stream in one hour
theorem boat_distance_against_stream : distance_against_stream time_duration = 5 := by
  sorry

end boat_distance_against_stream_l195_195407


namespace percentage_increase_soda_price_l195_195496

theorem percentage_increase_soda_price
  (C_new : ℝ) (S_new : ℝ) (C_increase : ℝ) (C_total_before : ℝ)
  (h1 : C_new = 20)
  (h2: S_new = 6)
  (h3: C_increase = 0.25)
  (h4: C_new * (1 - C_increase) + S_new * (1 + (S_new / (S_new * (1 + (S_new / (S_new * 0.5)))))) = C_total_before) : 
  (S_new - S_new * (1 - C_increase) * 100 / (S_new * (1 + 0.5)) * C_total_before) = 50 := 
by 
  -- This is where the proof would go.
  sorry

end percentage_increase_soda_price_l195_195496


namespace chess_tournament_games_l195_195679

theorem chess_tournament_games (n : ℕ) (h : n = 16) :
  (n * (n - 1) * 2) / 2 = 480 :=
by
  rw [h]
  simp
  norm_num
  sorry

end chess_tournament_games_l195_195679


namespace median_of_data_set_is_5_5_l195_195468

theorem median_of_data_set_is_5_5 : 
  let data := [3, 4, 5, 6, 6, 7];
  ∃ m, m = 5.5 ∧ m = (data.nth_le 2 sorry + data.nth_le 3 sorry) / 2 :=
by {
  -- Arrange the data in ascending order (already sorted in this case)
  let data_sorted := data.sorted (λ a b, a ≤ b);
  -- Find the middle elements for even length
  let m1 := data_sorted.nth_le 2 sorry;
  let m2 := data_sorted.nth_le 3 sorry;
  -- Calculate the median
  let median := (m1 + m2)/2;
  -- Assert the result
  use median;
  sorry
}

end median_of_data_set_is_5_5_l195_195468


namespace repeating_decimal_to_fraction_l195_195494

/-- The repeating decimal 0.565656... equals the fraction 56/99. -/
theorem repeating_decimal_to_fraction : 
  let a := 56 / 100
      r := 1 / 100
  in (a / (1 - r) = 56 / 99) := 
by
  let a := 56 / 100
  let r := 1 / 100
  have h1 : 0 < r, by norm_num
  have h2 : r < 1, by norm_num
  have sum_inf_geo_series : a / (1 - r) = 56 / 99 := by sorry
  use sum_inf_geo_series
  sorry

end repeating_decimal_to_fraction_l195_195494


namespace dave_deleted_apps_l195_195184

theorem dave_deleted_apps : 
  ∀ (a_initial a_left a_deleted : ℕ), a_initial = 16 → a_left = 5 → a_deleted = a_initial - a_left → a_deleted = 11 :=
by
  intros a_initial a_left a_deleted h_initial h_left h_deleted
  rw [h_initial, h_left] at h_deleted
  exact h_deleted

end dave_deleted_apps_l195_195184


namespace isosceles_triangle_perimeter_l195_195745

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3) (h2 : b = 7) : 
∃ (c : ℕ), 
  (c = 7 ∧ a = 3 ∧ b = 7 ∧ a + b + c = 17) ∨ 
  (c = 3 ∧ a = 7 ∧ b = 7 ∧ a + b + c = 17) :=
  sorry

end isosceles_triangle_perimeter_l195_195745


namespace female_students_count_l195_195405

variable (F M : ℕ)

def numberOfMaleStudents (F : ℕ) : ℕ := 3 * F

def totalStudents (F M : ℕ) : Prop := F + M = 52

theorem female_students_count :
  totalStudents F (numberOfMaleStudents F) → F = 13 :=
by
  intro h
  sorry

end female_students_count_l195_195405


namespace increase_80_by_150_percent_l195_195613

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l195_195613


namespace find_y_ratio_l195_195962

variable {R : Type} [LinearOrderedField R]
variables (x y : R → R) (x1 x2 y1 y2 : R)

-- Condition: x is inversely proportional to y, so xy is constant.
def inversely_proportional (x y : R → R) : Prop := ∀ (a b : R), x a * y a = x b * y b

-- Condition: ∀ nonzero x values, we have these specific ratios
variable (h_inv_prop : inversely_proportional x y)
variable (h_ratio_x : x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 / x2 = 4 / 5)
variable (h_nonzero_y : y1 ≠ 0 ∧ y2 ≠ 0)

-- Claim to prove
theorem find_y_ratio : (y1 / y2) = 5 / 4 :=
by
  sorry

end find_y_ratio_l195_195962


namespace artworks_per_student_in_first_half_l195_195948

theorem artworks_per_student_in_first_half (x : ℕ) (h1 : 10 = 10) (h2 : 20 = 20) (h3 : 5 * x + 5 * 4 = 35) : x = 3 := by
  sorry

end artworks_per_student_in_first_half_l195_195948


namespace article_final_price_l195_195281

theorem article_final_price (list_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : 
  first_discount = 0.1 → 
  second_discount = 0.01999999999999997 → 
  list_price = 70 → 
  ∃ final_price, final_price = 61.74 := 
by {
  sorry
}

end article_final_price_l195_195281


namespace sum_a3_a4_a5_a6_l195_195727

theorem sum_a3_a4_a5_a6 (S : ℕ → ℕ) (h : ∀ n, S n = n^2 + 2 * n) : S 6 - S 2 = 40 :=
by
  sorry

end sum_a3_a4_a5_a6_l195_195727


namespace original_number_exists_l195_195692

theorem original_number_exists : 
  ∃ (t o : ℕ), (10 * t + o = 74) ∧ (t = o * o - 9) ∧ (10 * o + t = 10 * t + o - 27) :=
by
  sorry

end original_number_exists_l195_195692


namespace inequality_problem_l195_195803

variable {R : Type*} [LinearOrderedField R]

theorem inequality_problem
  (a b : R) 
  (ha_pos : 0 < a)
  (hb_pos : 0 < b)
  (hab : a + b = 1) :
  (a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2 := 
sorry

end inequality_problem_l195_195803


namespace increase_80_by_150_percent_l195_195658

theorem increase_80_by_150_percent : 
  ∀ x percent, x = 80 ∧ percent = 1.5 → x + (x * percent) = 200 :=
by
  sorry

end increase_80_by_150_percent_l195_195658


namespace magician_performances_l195_195009

theorem magician_performances (performances : ℕ) (p_no_reappear : ℚ) (p_two_reappear : ℚ) :
  performances = 100 → p_no_reappear = 1/10 → p_two_reappear = 1/5 → 
  let num_no_reappear := performances * p_no_reappear in
  let num_two_reappear := performances * p_two_reappear in
  let num_one_reappear := performances - num_no_reappear - num_two_reappear in
  let total_reappeared := num_one_reappear + 2 * num_two_reappear in
  total_reappeared = 110 :=
by
  intros h1 h2 h3
  let num_no_reappear := performances * p_no_reappear
  let num_two_reappear := performances * p_two_reappear
  let num_one_reappear := performances - num_no_reappear - num_two_reappear
  let total_reappeared := num_one_reappear + 2 * num_two_reappear
  have h4 : num_no_reappear = 10 := by sorry
  have h5 : num_two_reappear = 20 := by sorry
  have h6 : num_one_reappear = 70 := by sorry
  have h7 : total_reappeared = 110 := by sorry
  exact h7

end magician_performances_l195_195009


namespace xena_head_start_l195_195139

theorem xena_head_start
  (xena_speed : ℝ) (dragon_speed : ℝ) (time : ℝ) (burn_distance : ℝ) 
  (xena_speed_eq : xena_speed = 15) 
  (dragon_speed_eq : dragon_speed = 30) 
  (time_eq : time = 32) 
  (burn_distance_eq : burn_distance = 120) :
  (dragon_speed * time - burn_distance) - (xena_speed * time) = 360 := 
  by 
  sorry

end xena_head_start_l195_195139


namespace trig_solutions_l195_195989

theorem trig_solutions (t : ℝ) :
  (sin(2 * t))^3 + (cos(2 * t))^3 + (1 / 2) * sin(4 * t) = 1 →
  ∃ k n : ℤ, t = k * π ∨ t = (π / 4) * (4 * n + 1) :=
by
  sorry

end trig_solutions_l195_195989


namespace a_5_is_9_l195_195720

-- Definition of the sequence sum S_n
def S : ℕ → ℕ
| n => n^2 - 1

-- Define the specific term in the sequence
def a (n : ℕ) :=
  if n = 1 then S 1 else S n - S (n - 1)

-- Theorem to prove
theorem a_5_is_9 : a 5 = 9 :=
sorry

end a_5_is_9_l195_195720


namespace seashells_total_l195_195947

theorem seashells_total (x y z T : ℕ) (m k : ℝ) 
  (h₁ : x = 2) 
  (h₂ : y = 5) 
  (h₃ : z = 9) 
  (h₄ : x + y = T) 
  (h₅ : m * x + k * y = z) : 
  T = 7 :=
by
  -- This is where the proof would go.
  sorry

end seashells_total_l195_195947


namespace remainder_when_7n_divided_by_4_l195_195304

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l195_195304


namespace roots_not_integers_l195_195719

theorem roots_not_integers (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
    ¬ ∃ x₁ x₂ : ℤ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

end roots_not_integers_l195_195719


namespace min_satisfies_condition_only_for_x_eq_1_div_4_l195_195503

theorem min_satisfies_condition_only_for_x_eq_1_div_4 (x : ℝ) (hx_nonneg : 0 ≤ x) :
  (min (Real.sqrt x) (min (x^2) x) = 1/16) ↔ (x = 1/4) :=
by sorry

end min_satisfies_condition_only_for_x_eq_1_div_4_l195_195503


namespace any_nat_as_difference_or_element_l195_195116

noncomputable def seq (q : ℕ → ℕ) : Prop :=
∀ n, q n < 2 * n

theorem any_nat_as_difference_or_element (q : ℕ → ℕ) (h_seq : seq q) (m : ℕ) :
  (∃ k, q k = m) ∨ (∃ k l, q l - q k = m) :=
sorry

end any_nat_as_difference_or_element_l195_195116


namespace triangle_problem_l195_195403

noncomputable def a : ℝ := 2 * Real.sqrt 3
noncomputable def B : ℝ := 45
noncomputable def S : ℝ := 3 + Real.sqrt 3

noncomputable def c : ℝ := Real.sqrt 2 + Real.sqrt 6
noncomputable def C : ℝ := 75

theorem triangle_problem
  (a_val : a = 2 * Real.sqrt 3)
  (B_val : B = 45)
  (S_val : S = 3 + Real.sqrt 3) :
  c = Real.sqrt 2 + Real.sqrt 6 ∧ C = 75 :=
by
  sorry

end triangle_problem_l195_195403


namespace probability_differs_by_three_l195_195161

theorem probability_differs_by_three :
  let outcomes := [(1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 3), (8, 5)],
      num_outcomes := List.length outcomes,
      total_possibilities := 8 * 8
  in
  Rational.mk num_outcomes total_possibilities = Rational.mk 7 64 :=
by
  sorry

end probability_differs_by_three_l195_195161


namespace sin_2phi_l195_195781

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l195_195781


namespace remainder_7n_mod_4_l195_195317

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l195_195317


namespace problem1_problem2_l195_195484

noncomputable def op (a b : ℝ) := 2 * a - (3 / 2) * (a + b)

theorem problem1 (x : ℝ) (h : op x 4 = 0) : x = 12 :=
by sorry

theorem problem2 (x m : ℝ) (h : op x m = op (-2) (x + 4)) (hnn : x ≥ 0) : m ≥ 14 / 3 :=
by sorry

end problem1_problem2_l195_195484


namespace base_k_for_repeating_series_equals_fraction_l195_195378

-- Define the fraction 5/29
def fraction := 5 / 29

-- Define the repeating series in base k
def repeating_series (k : ℕ) : ℚ :=
  (1 / k) / (1 - 1 / k^2) + (3 / k^2) / (1 - 1 / k^2)

-- State the problem
theorem base_k_for_repeating_series_equals_fraction (k : ℕ) (hk1 : 0 < k) (hk2 : k ≠ 1):
  repeating_series k = fraction ↔ k = 8 := sorry

end base_k_for_repeating_series_equals_fraction_l195_195378


namespace abs_diff_roots_eq_3_l195_195039

theorem abs_diff_roots_eq_3 : ∀ (r1 r2 : ℝ), (r1 ≠ r2) → (r1 + r2 = 7) → (r1 * r2 = 10) → |r1 - r2| = 3 :=
by
  intros r1 r2 hneq hsum hprod
  sorry

end abs_diff_roots_eq_3_l195_195039


namespace increased_number_l195_195606

theorem increased_number (original_number : ℕ) (percentage_increase : ℚ) :
  original_number = 80 →
  percentage_increase = 1.5 →
  original_number * (1 + percentage_increase) = 200 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end increased_number_l195_195606


namespace black_car_overtakes_red_car_in_one_hour_l195_195849

def red_car_speed : ℕ := 40
def black_car_speed : ℕ := 50
def initial_gap : ℕ := 10

theorem black_car_overtakes_red_car_in_one_hour (h_red_car_speed : red_car_speed = 40)
                                               (h_black_car_speed : black_car_speed = 50)
                                               (h_initial_gap : initial_gap = 10) :
  initial_gap / (black_car_speed - red_car_speed) = 1 :=
by
  sorry

end black_car_overtakes_red_car_in_one_hour_l195_195849


namespace distribution_of_6_balls_in_3_indistinguishable_boxes_l195_195530

-- Definition of the problem with conditions
def ways_to_distribute_balls_into_boxes
    (balls : ℕ) (boxes : ℕ) (distinguishable : bool)
    (indistinguishable : bool) : ℕ :=
  if (balls = 6) ∧ (boxes = 3) ∧ (distinguishable = true) ∧ (indistinguishable = true) 
  then 122 -- The correct answer given the conditions
  else 0

-- The Lean statement for the proof problem
theorem distribution_of_6_balls_in_3_indistinguishable_boxes :
  ways_to_distribute_balls_into_boxes 6 3 true true = 122 :=
by sorry

end distribution_of_6_balls_in_3_indistinguishable_boxes_l195_195530


namespace least_possible_lcm_l195_195278

-- Definitions of the least common multiples given the conditions
variable (a b c : ℕ)
variable (h₁ : Nat.lcm a b = 20)
variable (h₂ : Nat.lcm b c = 28)

-- The goal is to prove the least possible value of lcm(a, c) given the conditions
theorem least_possible_lcm (a b c : ℕ) (h₁ : Nat.lcm a b = 20) (h₂ : Nat.lcm b c = 28) : Nat.lcm a c = 35 :=
by
  sorry

end least_possible_lcm_l195_195278


namespace perpendicular_lines_slope_l195_195390

theorem perpendicular_lines_slope (a : ℝ) :
  (∀ x1 y1 x2 y2: ℝ, y1 = a * x1 - 2 ∧ y2 = x2 + 1 → (a * 1) = -1) → a = -1 :=
by
  sorry

end perpendicular_lines_slope_l195_195390


namespace sequence_general_term_l195_195439

theorem sequence_general_term (a : ℕ → ℤ) (h₀ : a 0 = 1) (hstep : ∀ n, a (n + 1) = if a n = 1 then 0 else 1) :
  ∀ n, a n = (1 + (-1)^(n + 1)) / 2 :=
sorry

end sequence_general_term_l195_195439


namespace no_family_argument_proportion_l195_195996

noncomputable def probability_no_family_argument (p quarrel_h husband quarreling with mother-in-law) (p_quarrel_w wife quarreling with mother-in-law) : ℝ :=
  let p_h_side_own := 1 / 2
  let p_w_side_own := 1 / 2
  let p_family_argument_h :=
    p quarrel_h husband quarreling with mother-in-law * p_h_side_own
  let p_family_argument_w :=
    p_quarrel_w wife quarreling with mother-in-law * p_w_side_own
  let p_family_argument := p_family_argument_h + p_family_argument_w - p_family_argument_h * p_family_argument_w
  1 - p_family_argument

theorem no_family_argument_proportion :
  probability_no_family_argument (2 / 3) (2 / 3) = 4 / 9 :=
by
  sorry

end no_family_argument_proportion_l195_195996


namespace multiple_choice_questions_count_l195_195562

variable (M F : ℕ)

-- Conditions
def totalQuestions := M + F = 60
def totalStudyTime := 15 * M + 25 * F = 1200

-- Statement to prove
theorem multiple_choice_questions_count (h1 : totalQuestions M F) (h2 : totalStudyTime M F) : M = 30 := by
  sorry

end multiple_choice_questions_count_l195_195562


namespace quadratic_roots_l195_195115

theorem quadratic_roots : ∀ x : ℝ, x * (x - 2) = 2 - x ↔ (x = 2 ∨ x = -1) := by
  intros
  sorry

end quadratic_roots_l195_195115


namespace complement_P_correct_l195_195940

def is_solution (x : ℝ) : Prop := |x + 3| + |x + 6| = 3

def P : Set ℝ := {x | is_solution x}

def C_R (P : Set ℝ) : Set ℝ := {x | x ∉ P}

theorem complement_P_correct : C_R P = {x | x < -6 ∨ x > -3} :=
by
  sorry

end complement_P_correct_l195_195940


namespace increase_80_by_150_percent_l195_195636

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l195_195636


namespace length_of_rect_box_l195_195583

noncomputable def length_of_box (height : ℝ) (width : ℝ) (volume : ℝ) : ℝ :=
  volume / (width * height)

theorem length_of_rect_box :
  (length_of_box 0.5 25 (6000 / 7.48052)) = 64.1624 :=
by
  unfold length_of_box
  norm_num
  sorry

end length_of_rect_box_l195_195583


namespace equipment_B_production_l195_195348

theorem equipment_B_production
  (total_production : ℕ)
  (sample_size : ℕ)
  (A_sample_production : ℕ)
  (B_sample_production : ℕ)
  (A_total_production : ℕ)
  (B_total_production : ℕ)
  (total_condition : total_production = 4800)
  (sample_condition : sample_size = 80)
  (A_sample_condition : A_sample_production = 50)
  (B_sample_condition : B_sample_production = 30)
  (ratio_condition : (A_sample_production / B_sample_production) = (5 / 3))
  (production_condition : A_total_production + B_total_production = total_production) :
  B_total_production = 1800 := 
sorry

end equipment_B_production_l195_195348


namespace computation_problem_points_l195_195347

def num_problems : ℕ := 30
def points_per_word_problem : ℕ := 5
def total_points : ℕ := 110
def num_computation_problems : ℕ := 20

def points_per_computation_problem : ℕ := 3

theorem computation_problem_points :
  ∃ x : ℕ, (num_computation_problems * x + (num_problems - num_computation_problems) * points_per_word_problem = total_points) ∧ x = points_per_computation_problem :=
by
  use points_per_computation_problem
  simp
  sorry

end computation_problem_points_l195_195347


namespace find_x_l195_195336

-- Define the conditions
def is_purely_imaginary (z : Complex) : Prop :=
  z.re = 0

-- Define the problem
theorem find_x (x : ℝ) (z : Complex) (h1 : z = Complex.ofReal (x^2 - 1) + Complex.I * (x + 1)) (h2 : is_purely_imaginary z) : x = 1 :=
sorry

end find_x_l195_195336


namespace find_b_l195_195810

noncomputable def f (b x : ℝ) : ℝ :=
if x < 1 then 2 * x - b else 2 ^ x

theorem find_b (b : ℝ) (h : f b (f b (1 / 2)) = 4) : b = -1 :=
sorry

end find_b_l195_195810


namespace increase_by_percentage_l195_195600

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l195_195600


namespace min_product_sum_l195_195427

theorem min_product_sum (a : Fin 7 → ℕ) (b : Fin 7 → ℕ) 
  (h2 : ∀ i, 2 ≤ a i) 
  (h3 : ∀ i, a i ≤ 166) 
  (h4 : ∀ i, a i ^ b i % 167 = a (i + 1) % 7 + 1 ^ 2 % 167) : 
  b 0 * b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * (b 0 + b 1 + b 2 + b 3 + b 4 + b 5 + b 6) = 675 := sorry

end min_product_sum_l195_195427


namespace john_got_80_percent_of_value_l195_195555

noncomputable def percentage_of_value (P : ℝ) : Prop :=
  let old_system_cost := 250
  let new_system_cost := 600
  let discount_percentage := 0.25
  let pocket_spent := 250
  let discount_amount := discount_percentage * new_system_cost
  let price_after_discount := new_system_cost - discount_amount
  let value_for_old_system := (P / 100) * old_system_cost
  value_for_old_system + pocket_spent = price_after_discount

theorem john_got_80_percent_of_value : percentage_of_value 80 :=
by
  sorry

end john_got_80_percent_of_value_l195_195555


namespace problem_l195_195511

variables {a b c : ℝ}

-- Given positive numbers a, b, c
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c

-- Given conditions
axiom h1 : a * b + a + b = 3
axiom h2 : b * c + b + c = 3
axiom h3 : a * c + a + c = 3

-- Goal statement
theorem problem : (a + 1) * (b + 1) * (c + 1) = 8 := 
by 
  sorry

end problem_l195_195511


namespace correct_number_l195_195112

theorem correct_number : ∃ x : ℤ, 2023 + x = 0 ∧ x = -2023 :=
by
  -- proof starts here
  sorry

end correct_number_l195_195112


namespace unique_solution_linear_system_l195_195108

theorem unique_solution_linear_system
  (a11 a22 a33 : ℝ) (a12 a13 a21 a23 a31 a32 : ℝ) 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 < a11) (h2 : 0 < a22) (h3 : 0 < a33)
  (h4 : a12 < 0) (h5 : a13 < 0) (h6 : a21 < 0) (h7 : a23 < 0) (h8 : a31 < 0) (h9 : a32 < 0)
  (h10 : 0 < a11 + a12 + a13) (h11 : 0 < a21 + a22 + a23) (h12 : 0 < a31 + a32 + a33) :
  (a11 * x1 + a12 * x2 + a13 * x3 = 0) →
  (a21 * x1 + a22 * x2 + a23 * x3 = 0) →
  (a31 * x1 + a32 * x2 + a33 * x3 = 0) →
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := by
  sorry

end unique_solution_linear_system_l195_195108


namespace triangle_perimeter_l195_195737

theorem triangle_perimeter (a b : ℝ) (x : ℝ) 
  (h₁ : a = 3) 
  (h₂ : b = 5) 
  (h₃ : x ^ 2 - 5 * x + 6 = 0)
  (h₄ : 2 < x ∧ x < 8) : a + b + x = 11 :=
by sorry

end triangle_perimeter_l195_195737


namespace find_eighth_number_l195_195107

def average_of_numbers (a b c d e f g h x : ℕ) : ℕ :=
  (a + b + c + d + e + f + g + h + x) / 9

theorem find_eighth_number (a b c d e f g h x : ℕ) (avg : ℕ) 
    (h_avg : average_of_numbers a b c d e f g h x = avg)
    (h_total_sum : a + b + c + d + e + f + g + h + x = 540)
    (h_x_val : x = 65) : a = 53 :=
by
  sorry

end find_eighth_number_l195_195107


namespace find_p_l195_195489

theorem find_p (p : ℕ) : 18^3 = (16^2 / 4) * 2^(8 * p) → p = 0 := 
by 
  sorry

end find_p_l195_195489


namespace pascal_triangle_contains_47_l195_195210

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l195_195210


namespace remainder_of_7n_mod_4_l195_195296

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l195_195296


namespace quadratic_distinct_real_roots_range_quadratic_root_product_value_l195_195063

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → k > 3 / 4 :=
sorry

theorem quadratic_root_product_value (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → (∀ x₁ x₂ : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → x₁ * x₂ = 5) 
  → k = 2 :=
sorry

end quadratic_distinct_real_roots_range_quadratic_root_product_value_l195_195063


namespace trigonometric_identity_l195_195535

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = Real.sqrt 2) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = (5 - Real.sqrt 2) / 3 := 
by
  sorry

end trigonometric_identity_l195_195535


namespace min_value_of_expression_l195_195091

theorem min_value_of_expression (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 10) : 
  ∃ B, B = x^2 + y^2 + z^2 + x^2 * y ∧ B ≥ 4 :=
by
  sorry

end min_value_of_expression_l195_195091


namespace sin_double_angle_l195_195760

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l195_195760


namespace hyperbola_focal_length_l195_195253

/--
In the Cartesian coordinate system \( xOy \),
let the focal length of the hyperbola \( \frac{x^{2}}{2m^{2}} - \frac{y^{2}}{3m} = 1 \) be 6.
Prove that the set of all real numbers \( m \) that satisfy this condition is {3/2}.
-/
theorem hyperbola_focal_length (m : ℝ) (h1 : 2 * m^2 > 0) (h2 : 3 * m > 0) (h3 : 2 * m^2 + 3 * m = 9) :
  m = 3 / 2 :=
sorry

end hyperbola_focal_length_l195_195253


namespace suzhou_visitors_accuracy_l195_195676

/--
In Suzhou, during the National Day holiday in 2023, the city received 17.815 million visitors.
Given that number, prove that it is accurate to the thousands place.
-/
theorem suzhou_visitors_accuracy :
  (17.815 : ℝ) * 10^6 = 17815000 ∧ true := 
by
sorry

end suzhou_visitors_accuracy_l195_195676


namespace part_I_part_II_l195_195560

def f (x a : ℝ) := |2 * x - a| + 5 * x

theorem part_I (x : ℝ) : f x 3 ≥ 5 * x + 1 ↔ (x ≤ 1 ∨ x ≥ 2) := sorry

theorem part_II (a x : ℝ) (h : (∀ x, f x a ≤ 0 ↔ x ≤ -1)) : a = 3 := sorry

end part_I_part_II_l195_195560


namespace remainder_of_7n_mod_4_l195_195295

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l195_195295


namespace ap_contains_sixth_power_l195_195357

theorem ap_contains_sixth_power (a d : ℕ) (i j x y : ℕ) 
  (h_positive : ∀ n, a + n * d > 0) 
  (h_square : a + i * d = x^2) 
  (h_cube : a + j * d = y^3) :
  ∃ k z : ℕ, a + k * d = z^6 := 
  sorry

end ap_contains_sixth_power_l195_195357


namespace polygon_sides_l195_195925

theorem polygon_sides (n : ℕ) 
  (h1 : sum_interior_angles = 180 * (n - 2))
  (h2 : sum_exterior_angles = 360)
  (h3 : sum_interior_angles = 3 * sum_exterior_angles) : 
  n = 8 :=
by
  sorry

end polygon_sides_l195_195925


namespace pascal_triangle_contains_prime_l195_195213

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l195_195213


namespace school_pays_570_l195_195572

theorem school_pays_570
  (price_per_model : ℕ := 100)
  (models_kindergarten : ℕ := 2)
  (models_elementary_multiple : ℕ := 2)
  (total_models : ℕ := models_kindergarten + models_elementary_multiple * models_kindergarten)
  (price_reduction : ℕ := if total_models > 5 then (price_per_model * 5 / 100) else 0)
  (reduced_price_per_model : ℕ := price_per_model - price_reduction) :
  2 * models_kindergarten * reduced_price_per_model = 570 :=
by
  -- Proof omitted
  sorry

end school_pays_570_l195_195572


namespace value_of_x2_plus_inv_x2_l195_195911

theorem value_of_x2_plus_inv_x2 (x : ℝ) (hx : x ≠ 0) (h : x^4 + 1 / x^4 = 47) : x^2 + 1 / x^2 = 7 :=
sorry

end value_of_x2_plus_inv_x2_l195_195911


namespace marcus_has_210_cards_l195_195950

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the increment of baseball cards Marcus has over Carter
def increment : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + increment

-- Prove that Marcus has 210 baseball cards
theorem marcus_has_210_cards : marcus_cards = 210 :=
by simp [marcus_cards, carter_cards, increment]

end marcus_has_210_cards_l195_195950


namespace votes_combined_l195_195030

theorem votes_combined (vote_A vote_B : ℕ) (h_ratio : vote_A = 2 * vote_B) (h_A_votes : vote_A = 14) : vote_A + vote_B = 21 :=
by
  sorry

end votes_combined_l195_195030


namespace max_tan_value_l195_195723

open real

noncomputable def max_tan (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2)
                          (h₂ : 0 < β ∧ β < π / 2)
                          (h₃ : cos (α + β) = sin α / sin β) : ℝ :=
  sup (set_of (λ x : ℝ, 0 < x ∧ x = tan α ∧ x ≤ sqrt 2 / 4))

theorem max_tan_value (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2)
                           (h₂ : 0 < β ∧ β < π / 2)
                           (h₃ : cos (α + β) = sin α / sin β) :
  max_tan α β h₁ h₂ h₃ = sqrt 2 / 4 :=
sorry

end max_tan_value_l195_195723


namespace base7_perfect_square_xy5z_l195_195239

theorem base7_perfect_square_xy5z (n : ℕ) (x y z : ℕ) (hx : x ≠ 0) (hn : n = 343 * x + 49 * y + 35 + z) (hsq : ∃ m : ℕ, n = m * m) : z = 1 ∨ z = 6 :=
sorry

end base7_perfect_square_xy5z_l195_195239


namespace exists_infinitely_many_solutions_l195_195188

theorem exists_infinitely_many_solutions :
  ∃ m : ℕ, m > 0 ∧ (∀ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) →
    (1/a + 1/b + 1/c + 1/(a*b*c) = m / (a + b + c))) :=
sorry

end exists_infinitely_many_solutions_l195_195188


namespace subsets_of_Zmod_prime_l195_195089

theorem subsets_of_Zmod_prime {p : ℕ} [Fact (Nat.Prime p)] :
  {S : Set (ZMod p) | 
    (∀ a b ∈ S, a * b ∈ S) ∧ 
    (∃ r ∈ S, ∀ a ∈ S, r - a ∈ S ∨ r - a = 0)} = 
  {S : Set (ZMod p) | 
    (S \ {0} ⊆ {1}) ∨ 
    ((∃ H : Subgroup (ZMod p)ˣ, H.carrier = S \ {0}) ∧ (-1 : (ZMod p)ˣ ∈ H)} } :=
sorry

end subsets_of_Zmod_prime_l195_195089


namespace rhombus_side_length_15_l195_195375

variable {p : ℝ} (h_p : p = 60)
variable {n : ℕ} (h_n : n = 4)

noncomputable def side_length_of_rhombus (p : ℝ) (n : ℕ) : ℝ :=
p / n

theorem rhombus_side_length_15 (h_p : p = 60) (h_n : n = 4) :
  side_length_of_rhombus p n = 15 :=
by
  sorry

end rhombus_side_length_15_l195_195375


namespace isosceles_triangle_perimeter_l195_195738

theorem isosceles_triangle_perimeter
  (a b : ℕ)
  (ha : a = 3)
  (hb : b = 7)
  (h_iso : ∃ (c : ℕ), (c = a ∨ c = b) ∧ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  a + b + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l195_195738


namespace percentage_of_additional_money_is_10_l195_195988

-- Define the conditions
def months := 11
def payment_per_month := 15
def total_borrowed := 150

-- Define the function to calculate the total amount paid
def total_paid (months payment_per_month : ℕ) : ℕ :=
  months * payment_per_month

-- Define the function to calculate the additional amount paid
def additional_paid (total_paid total_borrowed : ℕ) : ℕ :=
  total_paid - total_borrowed

-- Define the function to calculate the percentage of the additional amount
def percentage_additional (additional_paid total_borrowed : ℕ) : ℕ :=
  (additional_paid * 100) / total_borrowed

-- State the theorem to prove the percentage of the additional money is 10%
theorem percentage_of_additional_money_is_10 :
  percentage_additional (additional_paid (total_paid months payment_per_month) total_borrowed) total_borrowed = 10 :=
by
  sorry

end percentage_of_additional_money_is_10_l195_195988


namespace tyre_punctures_deflation_time_l195_195991

theorem tyre_punctures_deflation_time :
  (1 / (1 / 9 + 1 / 6)) = 3.6 :=
by
  sorry

end tyre_punctures_deflation_time_l195_195991


namespace frank_handed_cashier_amount_l195_195712

-- Place conditions as definitions
def cost_chocolate_bar : ℕ := 2
def cost_bag_chip : ℕ := 3
def num_chocolate_bars : ℕ := 5
def num_bag_chips : ℕ := 2
def change_received : ℕ := 4

-- Define the target theorem (Lean 4 statement)
theorem frank_handed_cashier_amount :
  (num_chocolate_bars * cost_chocolate_bar + num_bag_chips * cost_bag_chip + change_received = 20) := 
sorry

end frank_handed_cashier_amount_l195_195712


namespace increase_150_percent_of_80_l195_195631

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l195_195631


namespace blocks_given_by_father_l195_195479

theorem blocks_given_by_father :
  ∀ (blocks_original total_blocks blocks_given : ℕ), 
  blocks_original = 2 →
  total_blocks = 8 →
  blocks_given = total_blocks - blocks_original →
  blocks_given = 6 :=
by
  intros blocks_original total_blocks blocks_given h1 h2 h3
  sorry

end blocks_given_by_father_l195_195479


namespace impossible_piles_of_three_l195_195269

theorem impossible_piles_of_three (n : ℕ) (h1 : n = 1001)
  (h2 : ∀ p : ℕ, p > 1 → ∃ a b : ℕ, a + b = p - 1 ∧ a ≤ b) : 
  ¬ (∃ piles : List ℕ, ∀ pile ∈ piles, pile = 3 ∧ (piles.sum = n + piles.length)) :=
by
  sorry

end impossible_piles_of_three_l195_195269


namespace quadratic_distinct_real_roots_range_quadratic_root_product_value_l195_195062

theorem quadratic_distinct_real_roots_range (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → k > 3 / 4 :=
sorry

theorem quadratic_root_product_value (k : ℝ) :
  (∀ x : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x - x₁) * (x - x₂) = 0) 
  → (∀ x₁ x₂ : ℝ, (x^2 + (2*k + 1)*x + k^2 + 1 = 0) → x₁ * x₂ = 5) 
  → k = 2 :=
sorry

end quadratic_distinct_real_roots_range_quadratic_root_product_value_l195_195062


namespace six_inch_cube_value_eq_844_l195_195470

-- Definition of the value of a cube in lean
noncomputable def cube_value (s₁ s₂ : ℕ) (value₁ : ℕ) : ℕ :=
  let volume₁ := s₁ ^ 3
  let volume₂ := s₂ ^ 3
  (value₁ * volume₂) / volume₁

-- Theorem stating the equivalence between the volumes and values.
theorem six_inch_cube_value_eq_844 :
  cube_value 4 6 250 = 844 :=
by
  sorry

end six_inch_cube_value_eq_844_l195_195470


namespace remainder_when_7n_divided_by_4_l195_195301

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l195_195301


namespace spending_ratio_l195_195811

theorem spending_ratio 
  (lisa_tshirts : Real)
  (lisa_jeans : Real)
  (lisa_coats : Real)
  (carly_tshirts : Real)
  (carly_jeans : Real)
  (carly_coats : Real)
  (total_spent : Real)
  (hl1 : lisa_tshirts = 40)
  (hl2 : lisa_jeans = lisa_tshirts / 2)
  (hl3 : lisa_coats = 2 * lisa_tshirts)
  (hc1 : carly_tshirts = lisa_tshirts / 4)
  (hc2 : carly_coats = lisa_coats / 4)
  (htotal : total_spent = lisa_tshirts + lisa_jeans + lisa_coats + carly_tshirts + carly_jeans + carly_coats)
  (h_total_spent_val : total_spent = 230) :
  carly_jeans = 3 * lisa_jeans :=
by
  -- Placeholder for theorem's proof
  sorry

end spending_ratio_l195_195811


namespace sin_double_angle_l195_195765

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l195_195765


namespace pascal_triangle_47_number_of_rows_containing_47_l195_195224

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l195_195224


namespace triangle_no_two_obtuse_angles_l195_195134

theorem triangle_no_two_obtuse_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 90) (h3 : B > 90) (h4 : C > 0) : false :=
by
  sorry

end triangle_no_two_obtuse_angles_l195_195134


namespace problem_l195_195730

noncomputable def key_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : Real.sqrt (x * y) ≤ 1) 
    : Prop := ∃ z : ℝ, 0 < z ∧ z = 2 * (x + y) / (x + y + 2)^2

theorem problem (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
    (habcd : a + b + c + d = 2) :
    (a^2 / (a^2 + 1)^2) + (b^2 / (b^2 + 1)^2) + (c^2 / (c^2 + 1)^2) + (d^2 / (d^2 + 1)^2) ≤ 16 / 25 := 
sorry

end problem_l195_195730


namespace intersection_point_x_coordinate_l195_195369

noncomputable def hyperbola (x y b : ℝ) := x^2 - (y^2 / b^2) = 1

noncomputable def c := 1 + Real.sqrt 3

noncomputable def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_point_x_coordinate
  (x y b : ℝ)
  (h_hyperbola : hyperbola x y b)
  (h_distance_foci : distance (2 * c, 0) (0, 0) = 2 * c)
  (h_circle_center : distance (x, y) (0, 0) = c)
  (h_p_distance : distance (x, y) (2 * c, 0) = c + 2) :
  x = (Real.sqrt 3 + 1) / 2 :=
sorry

end intersection_point_x_coordinate_l195_195369


namespace math_problem_l195_195270

theorem math_problem (x y z : ℤ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : (x + y * Complex.I)^2 - 46 * Complex.I = z) :
  x + y + z = 552 :=
by
  sorry

end math_problem_l195_195270


namespace increase_80_by_150_percent_l195_195639

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l195_195639


namespace remainder_7n_mod_4_l195_195314

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l195_195314


namespace multiply_by_nine_l195_195320

theorem multiply_by_nine (x : ℝ) (h : 9 * x = 36) : x = 4 :=
sorry

end multiply_by_nine_l195_195320


namespace increase_by_percentage_l195_195603

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l195_195603


namespace distribute_6_balls_in_3_boxes_l195_195532

def num_ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if balls = 6 ∧ boxes = 3 then 92 else 0

theorem distribute_6_balls_in_3_boxes :
  num_ways_to_distribute_balls 6 3 = 92 :=
by
  rw num_ways_to_distribute_balls
  norm_num
  sorry

end distribute_6_balls_in_3_boxes_l195_195532


namespace quadratic_to_vertex_form_l195_195926

theorem quadratic_to_vertex_form :
  ∀ (x a h k : ℝ), (x^2 - 7*x = a*(x - h)^2 + k) → k = -49 / 4 :=
by
  intros x a h k
  sorry

end quadratic_to_vertex_form_l195_195926


namespace exists_equal_sum_pairs_l195_195675

theorem exists_equal_sum_pairs (n : ℕ) (hn : n > 2009) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≤ 2009 ∧ b ≤ 2009 ∧ c ≤ 2009 ∧ d ≤ 2009 ∧
  (1 / a + 1 / b : ℝ) = 1 / c + 1 / d :=
sorry

end exists_equal_sum_pairs_l195_195675


namespace find_largest_number_l195_195900

theorem find_largest_number
  (a b c d e : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h₁ : a + b = 32)
  (h₂ : a + c = 36)
  (h₃ : b + c = 37)
  (h₄ : c + e = 48)
  (h₅ : d + e = 51) :
  (max a (max b (max c (max d e)))) = 27.5 :=
sorry

end find_largest_number_l195_195900


namespace remainder_when_7n_divided_by_4_l195_195309

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l195_195309


namespace validate_option_B_l195_195847

theorem validate_option_B (a b : ℝ) : 
  (2 * a + 3 * a^2 ≠ 5 * a^3) ∧ 
  ((-a^3)^2 = a^6) ∧ 
  (¬ (-4 * a^3 * b / (2 * a) = -2 * a^2)) ∧ 
  ((5 * a * b)^2 ≠ 10 * a^2 * b^2) := 
by
  sorry

end validate_option_B_l195_195847


namespace increase_150_percent_of_80_l195_195629

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l195_195629


namespace remainder_7n_mod_4_l195_195310

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l195_195310


namespace sequence_periodic_l195_195561

theorem sequence_periodic (a : ℕ → ℝ) (h1 : a 1 = 0) (h2 : ∀ n, a n + a (n + 1) = 2) : a 2011 = 0 := by
  sorry

end sequence_periodic_l195_195561


namespace increase_80_by_150_percent_l195_195634

-- Define the initial number (n) and the percentage increase (p)
def n : ℕ := 80
def p : ℚ := 1.5

-- The theorem stating the expected result after increasing n by 150%
theorem increase_80_by_150_percent : n + (p * n) = 200 := by
  sorry

end increase_80_by_150_percent_l195_195634


namespace number_of_three_digit_numbers_with_5_and_7_l195_195919

def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999
def containsDigit (n : ℕ) (d : ℕ) : Prop := d ∈ (n.digits 10) 
def hasAtLeastOne5andOne7 (n : ℕ) : Prop := containsDigit n 5 ∧ containsDigit n 7
def totalThreeDigitNumbersWith5and7 : ℕ := 50

theorem number_of_three_digit_numbers_with_5_and_7 :
  ∃ n : ℕ, isThreeDigitNumber n ∧ hasAtLeastOne5andOne7 n → n = 50 := sorry

end number_of_three_digit_numbers_with_5_and_7_l195_195919


namespace sum_is_two_l195_195805

noncomputable def compute_sum (x : ℂ) (hx : x^7 = 1) (hx_ne : x ≠ 1) : ℂ :=
  (x^2 / (x - 1)) + (x^4 / (x^2 - 1)) + (x^6 / (x^3 - 1)) + (x^8 / (x^4 - 1)) + (x^10 / (x^5 - 1)) + (x^12 / (x^6 - 1))

theorem sum_is_two (x : ℂ) (hx : x^7 = 1) (hx_ne : x ≠ 1) : compute_sum x hx hx_ne = 2 :=
by
  sorry

end sum_is_two_l195_195805


namespace increase_by_percentage_l195_195601

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l195_195601


namespace minimum_value_of_quadratic_function_l195_195793

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem minimum_value_of_quadratic_function 
  (f : ℝ → ℝ)
  (n : ℕ)
  (h1 : f n = 6)
  (h2 : f (n + 1) = 5)
  (h3 : f (n + 2) = 5)
  (hf : ∃ a b c : ℝ, f = quadratic_function a b c) :
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 5 :=
by
  sorry

end minimum_value_of_quadratic_function_l195_195793


namespace part1_monotonicity_part2_range_a_l195_195518

noncomputable def f (a x : ℝ) : ℝ := Real.log x - a * x + 1

theorem part1_monotonicity (a : ℝ) :
  (∀ x > 0, (0 : ℝ) < x → 0 < 1 / x - a) ∨
  (a > 0 → ∀ x > 0, (0 : ℝ) < x ∧ x < 1 / a → 0 < 1 / x - a ∧ 1 / a < x → 1 / x - a < 0) := sorry

theorem part2_range_a (a : ℝ) :
  (∀ x > 0, Real.log x - a * x + 1 ≤ 0) → 1 ≤ a := sorry

end part1_monotonicity_part2_range_a_l195_195518


namespace amount_spent_l195_195694

-- Definitions
def initial_amount : ℕ := 54
def amount_left : ℕ := 29

-- Proof statement
theorem amount_spent : initial_amount - amount_left = 25 :=
by
  sorry

end amount_spent_l195_195694


namespace sin_2phi_l195_195771

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end sin_2phi_l195_195771


namespace increase_150_percent_of_80_l195_195630

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l195_195630


namespace rachel_milk_amount_l195_195371

theorem rachel_milk_amount : 
  let don_milk := (3 : ℚ) / 7
  let rachel_fraction := 4 / 5
  let rachel_milk := rachel_fraction * don_milk
  rachel_milk = 12 / 35 :=
by sorry

end rachel_milk_amount_l195_195371


namespace remainder_7n_mod_4_l195_195319

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l195_195319


namespace increase_150_percent_of_80_l195_195632

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l195_195632


namespace janet_overtime_multiple_l195_195084

theorem janet_overtime_multiple :
  let hourly_rate := 20
  let weekly_hours := 52
  let regular_hours := 40
  let car_price := 4640
  let weeks_needed := 4
  let normal_weekly_earning := regular_hours * hourly_rate
  let overtime_hours := weekly_hours - regular_hours
  let required_weekly_earning := car_price / weeks_needed
  let overtime_weekly_earning := required_weekly_earning - normal_weekly_earning
  let overtime_rate := overtime_weekly_earning / overtime_hours
  (overtime_rate / hourly_rate = 1.5) :=
by
  sorry

end janet_overtime_multiple_l195_195084


namespace find_symmetric_point_l195_195752

structure Point := (x : Int) (y : Int)

def translate_right (p : Point) (n : Int) : Point :=
  { x := p.x + n, y := p.y }

def symmetric_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem find_symmetric_point : 
  ∀ (A B C : Point),
  A = ⟨-1, 2⟩ →
  B = translate_right A 2 →
  C = symmetric_x_axis B →
  C = ⟨1, -2⟩ :=
by
  intros A B C hA hB hC
  sorry

end find_symmetric_point_l195_195752


namespace charge_per_call_proof_l195_195149

-- Define the conditions as given in the problem
def fixed_rental : ℝ := 350
def free_calls_per_month : ℕ := 200
def charge_per_call_exceed_200 (x : ℝ) (calls : ℕ) : ℝ := 
  if calls > 200 then (calls - 200) * x else 0

def charge_per_call_exceed_400 : ℝ := 1.6
def discount_rate : ℝ := 0.28
def february_calls : ℕ := 150
def march_calls : ℕ := 250
def march_discount (x : ℝ) : ℝ := x * (1 - discount_rate)
def total_march_charge (x : ℝ) : ℝ := 
  fixed_rental + charge_per_call_exceed_200 (march_discount x) march_calls

-- Prove the correct charge per call when calls exceed 200 per month
theorem charge_per_call_proof (x : ℝ) : 
  charge_per_call_exceed_200 x february_calls = 0 ∧ 
  total_march_charge x = fixed_rental + (march_calls - free_calls_per_month) * (march_discount x) → 
  x = x := 
by { 
  sorry 
}

end charge_per_call_proof_l195_195149


namespace increase_by_percentage_l195_195621

theorem increase_by_percentage (initial_amount : ℕ) (percentage_increase : ℕ) :
  initial_amount = 80 → 
  percentage_increase = 150 → 
  initial_amount * (percentage_increase / 100) + initial_amount = 200 :=
by
  intros h_initial h_percentage
  rw [h_initial, h_percentage]
  calc
    80 * (150 / 100) + 80 = 80 * 1.5 + 80 : by norm_num
                     ... = 120 + 80         : by norm_num
                     ... = 200              : by norm_num
  sorry

end increase_by_percentage_l195_195621


namespace pascals_triangle_contains_47_once_l195_195206

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l195_195206


namespace ratio_cars_to_dogs_is_two_l195_195485

-- Definitions of the conditions
def initial_dogs : ℕ := 90
def initial_cars : ℕ := initial_dogs / 3
def additional_cars : ℕ := 210
def current_dogs : ℕ := 120
def current_cars : ℕ := initial_cars + additional_cars

-- The statement to be proven
theorem ratio_cars_to_dogs_is_two :
  (current_cars : ℚ) / (current_dogs : ℚ) = 2 := by
  sorry

end ratio_cars_to_dogs_is_two_l195_195485


namespace maximum_y_coordinate_l195_195915

variable (x y b : ℝ)

def hyperbola (x y b : ℝ) : Prop := (x^2) / 4 - (y^2) / b = 1

def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

def op_condition (x y b : ℝ) : Prop := (x^2 + y^2) = 4 + b

noncomputable def eccentricity (b : ℝ) : ℝ := (Real.sqrt (4 + b)) / 2

theorem maximum_y_coordinate (hb : b > 0) 
                            (h_ec : 1 < eccentricity b ∧ eccentricity b ≤ 2) 
                            (h_hyp : hyperbola x y b) 
                            (h_first : first_quadrant x y) 
                            (h_op : op_condition x y b) 
                            : y ≤ 3 :=
sorry

end maximum_y_coordinate_l195_195915


namespace sin_2phi_l195_195782

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l195_195782


namespace basketball_team_starting_lineups_l195_195681

theorem basketball_team_starting_lineups :
  let total_players := 16
  let starters := 7
  let triplets := 3
  let twins := 2
  let unrestricted_lineups := Nat.choose 16 7
  let invalid_triplets_lineups := Nat.choose 13 4
  let invalid_twins_lineups := Nat.choose 14 5
  let overlap_lineups := Nat.choose 11 2
  unrestricted_lineups - (invalid_triplets_lineups + invalid_twins_lineups - overlap_lineups) = 8778 := by
  sorry

end basketball_team_starting_lineups_l195_195681


namespace increase_80_by_150_percent_l195_195614

-- Original number
def originalNumber : ℕ := 80

-- Percentage increase as a decimal
def increaseFactor : ℚ := 1.5

-- Expected result after the increase
def expectedResult : ℕ := 200

theorem increase_80_by_150_percent :
  originalNumber + (increaseFactor * originalNumber) = expectedResult :=
by
  sorry

end increase_80_by_150_percent_l195_195614


namespace units_digit_17_mul_27_l195_195042

theorem units_digit_17_mul_27 : 
  let u17 := (17 % 10)
  let u27 := (27 % 10)
  let units_product := (u17 * u27) % 10
  units_product = 9 := by
  let u17 := (17 % 10)
  let u27 := (27 % 10)
  let units_product := (u17 * u27) % 10
  sorry

end units_digit_17_mul_27_l195_195042


namespace allocation_problem_l195_195148

-- Definitions based on conditions
variable (employees : Fin 5) (departments : Fin 3)

noncomputable def allocation_methods_count : Nat := 36 -- Correct answer from the problem

-- Statement to prove
theorem allocation_problem :
  ∃ (f : employees → departments),
    (∀ d : departments, ∃ e : employees, f e = d) ∧
    (∃ d : departments, ∀ e : {e : employees // e < 2}, f e.1 = d) →
    allocation_methods_count = 36 :=
by
  sorry

end allocation_problem_l195_195148


namespace union_of_sets_l195_195729

theorem union_of_sets (P Q : Set ℝ) 
  (hP : P = {x | 2 ≤ x ∧ x ≤ 3}) 
  (hQ : Q = {x | x^2 ≤ 4}) : 
  P ∪ Q = {x | -2 ≤ x ∧ x ≤ 3} := 
sorry

end union_of_sets_l195_195729


namespace inequality_relation_l195_195142

theorem inequality_relation (a b : ℝ) :
  (∃ a b : ℝ, a > b ∧ ¬(1/a < 1/b)) ∧ (∃ a b : ℝ, (1/a < 1/b) ∧ ¬(a > b)) :=
by {
  sorry
}

end inequality_relation_l195_195142


namespace min_value_of_squares_l195_195906

theorem min_value_of_squares (x y z : ℝ) (h : x + y + z = 1) : x^2 + y^2 + z^2 ≥ 1 / 3 := sorry

end min_value_of_squares_l195_195906


namespace arithmetic_progression_contains_sixth_power_l195_195354

theorem arithmetic_progression_contains_sixth_power (a b : ℕ) (h_ap_pos : ∀ t : ℕ, a + b * t > 0)
  (h_contains_square : ∃ n : ℕ, ∃ t : ℕ, a + b * t = n^2)
  (h_contains_cube : ∃ m : ℕ, ∃ t : ℕ, a + b * t = m^3) :
  ∃ k : ℕ, ∃ t : ℕ, a + b * t = k^6 :=
sorry

end arithmetic_progression_contains_sixth_power_l195_195354


namespace correct_mark_l195_195965

theorem correct_mark 
  (avg_wrong : ℝ := 60)
  (wrong_mark : ℝ := 90)
  (num_students : ℕ := 30)
  (avg_correct : ℝ := 57.5) :
  (wrong_mark - (avg_wrong * num_students - avg_correct * num_students)) = 15 :=
by
  sorry

end correct_mark_l195_195965


namespace t_shirts_in_two_hours_l195_195362

-- Definitions for the conditions
def first_hour_rate : Nat := 12
def second_hour_rate : Nat := 6

-- Main statement to prove
theorem t_shirts_in_two_hours : 
  (60 / first_hour_rate + 60 / second_hour_rate) = 15 := by
  sorry

end t_shirts_in_two_hours_l195_195362


namespace remainder_sum_of_integers_division_l195_195806

theorem remainder_sum_of_integers_division (n S : ℕ) (hn_cond : n > 0) (hn_sq : n^2 + 12 * n - 3007 ≥ 0) (hn_square : ∃ m : ℕ, n^2 + 12 * n - 3007 = m^2):
  S = n → S % 1000 = 516 := 
sorry

end remainder_sum_of_integers_division_l195_195806


namespace sphere_radius_proportional_l195_195979

theorem sphere_radius_proportional
  (k : ℝ)
  (r1 r2 : ℝ)
  (W1 W2 : ℝ)
  (h_weight_area : ∀ (r : ℝ), W1 = k * (4 * π * r^2))
  (h_given1: W2 = 32)
  (h_given2: r2 = 0.3)
  (h_given3: W1 = 8):
  r1 = 0.15 := 
by
  sorry

end sphere_radius_proportional_l195_195979


namespace mean_score_calculation_l195_195735

noncomputable def class_mean_score (total_students students_1 mean_score_1 students_2 mean_score_2 : ℕ) : ℚ :=
  ((students_1 * mean_score_1 + students_2 * mean_score_2) : ℚ) / total_students

theorem mean_score_calculation :
  class_mean_score 60 54 76 6 82 = 76.6 := 
sorry

end mean_score_calculation_l195_195735


namespace remainder_correct_l195_195495

noncomputable def p : Polynomial ℝ := Polynomial.C 3 * Polynomial.X^5 + Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X + Polynomial.C 8
noncomputable def d : Polynomial ℝ := (Polynomial.X - Polynomial.C 1) ^ 2
noncomputable def r : Polynomial ℝ := Polynomial.C 16 * Polynomial.X - Polynomial.C 8 

theorem remainder_correct : (p % d) = r := by sorry

end remainder_correct_l195_195495


namespace zeros_of_quadratic_l195_195072

theorem zeros_of_quadratic (a b : ℝ) (h : a + b = 0) : 
  ∀ x, (b * x^2 - a * x = 0) ↔ (x = 0 ∨ x = -1) :=
by
  intro x
  sorry

end zeros_of_quadratic_l195_195072


namespace sin_double_angle_l195_195758

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l195_195758


namespace simplify_identity_l195_195431

theorem simplify_identity :
  ∀ θ : ℝ, θ = 160 → (1 / (Real.sqrt (1 + Real.tan (θ : ℝ) ^ 2))) = -Real.cos (θ : ℝ) :=
by
  intro θ h
  rw [h]
  sorry  

end simplify_identity_l195_195431


namespace solve_fraction_eqn_l195_195122

def fraction_eqn_solution : Prop :=
  ∃ (x : ℝ), (x + 2) / (x - 1) = 0 ∧ x ≠ 1 ∧ x = -2

theorem solve_fraction_eqn : fraction_eqn_solution :=
sorry

end solve_fraction_eqn_l195_195122


namespace fractions_product_equals_54_l195_195703

theorem fractions_product_equals_54 :
  (4 / 5) * (9 / 6) * (12 / 4) * (20 / 15) * (14 / 21) * (35 / 28) * (48 / 32) * (24 / 16) = 54 :=
by
  -- Add the proof here
  sorry

end fractions_product_equals_54_l195_195703


namespace inverse_proportion_points_l195_195905

theorem inverse_proportion_points (x1 x2 y1 y2 : ℝ)
  (h1 : x1 < 0)
  (h2 : x2 > 0)
  (h3 : y1 = -8 / x1)
  (h4 : y2 = -8 / x2) :
  y2 < 0 ∧ 0 < y1 :=
by
  sorry

end inverse_proportion_points_l195_195905


namespace tree_leaves_remaining_after_three_weeks_l195_195158

theorem tree_leaves_remaining_after_three_weeks :
  let initial_leaves := 1000
  let leaves_shed_first_week := (2 / 5 : ℝ) * initial_leaves
  let leaves_remaining_after_first_week := initial_leaves - leaves_shed_first_week
  let leaves_shed_second_week := (4 / 10 : ℝ) * leaves_remaining_after_first_week
  let leaves_remaining_after_second_week := leaves_remaining_after_first_week - leaves_shed_second_week
  let leaves_shed_third_week := (3 / 4 : ℝ) * leaves_shed_second_week
  let leaves_remaining_after_third_week := leaves_remaining_after_second_week - leaves_shed_third_week
  leaves_remaining_after_third_week = 180 :=
by
  sorry

end tree_leaves_remaining_after_three_weeks_l195_195158


namespace marcus_baseball_cards_l195_195953

-- Define the number of baseball cards Carter has
def carter_cards : ℕ := 152

-- Define the number of additional cards Marcus has compared to Carter
def additional_cards : ℕ := 58

-- Define the number of baseball cards Marcus has
def marcus_cards : ℕ := carter_cards + additional_cards

-- The proof statement asserting Marcus' total number of baseball cards
theorem marcus_baseball_cards : marcus_cards = 210 :=
by {
  -- This is where the proof steps would go, but we are skipping with sorry
  sorry
}

end marcus_baseball_cards_l195_195953


namespace sin_2phi_l195_195779

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l195_195779


namespace sum_of_areas_of_triangles_in_cube_l195_195443

theorem sum_of_areas_of_triangles_in_cube : 
  let m := 48
  let n := 4608
  let p := 576
  m + n + p = 5232 := by 
    sorry

end sum_of_areas_of_triangles_in_cube_l195_195443


namespace probability_of_equal_digit_counts_l195_195680

noncomputable def probability_equal_digit_counts : ℚ :=
  let p_one_digit := (9 : ℚ) / 20
  let p_two_digit := (11 : ℚ) / 20
  let ways := nat.choose 6 3
  ways * p_one_digit^3 * p_two_digit^3

theorem probability_of_equal_digit_counts :
  (probability_equal_digit_counts = (4851495 : ℚ) / 16000000) :=
by
  -- Mathematical proof skipped
  sorry

end probability_of_equal_digit_counts_l195_195680


namespace find_sam_current_age_l195_195750

def Drew_current_age : ℕ := 12

def Drew_age_in_five_years : ℕ := Drew_current_age + 5

def Sam_age_in_five_years : ℕ := 3 * Drew_age_in_five_years

def Sam_current_age : ℕ := Sam_age_in_five_years - 5

theorem find_sam_current_age : Sam_current_age = 46 := by
  sorry

end find_sam_current_age_l195_195750


namespace trajectory_of_P_l195_195285

-- Define points P, A, and B in a 2D plane
variable {P A B : EuclideanSpace ℝ (Fin 2)}

-- Define the condition that the sum of the distances from P to A and P to B equals the distance between A and B
def sum_of_distances_condition (P A B : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist P A + dist P B = dist A B

-- Main theorem statement: If P satisfies the above condition, then P lies on the line segment AB
theorem trajectory_of_P (P A B : EuclideanSpace ℝ (Fin 2)) (h : sum_of_distances_condition P A B) :
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = t • A + (1 - t) • B :=
  sorry

end trajectory_of_P_l195_195285


namespace increase_by_150_percent_l195_195665

theorem increase_by_150_percent (x : ℝ) (h : x = 80) : x + (1.5 * x) = 200 := 
by
  -- The proof goes here, but is omitted as per instructions.
  sorry

end increase_by_150_percent_l195_195665


namespace stock_price_after_two_years_l195_195160

theorem stock_price_after_two_years 
    (p0 : ℝ) (r1 r2 : ℝ) (p1 p2 : ℝ) 
    (h0 : p0 = 100) (h1 : r1 = 0.50) 
    (h2 : r2 = 0.30) 
    (h3 : p1 = p0 * (1 + r1)) 
    (h4 : p2 = p1 * (1 - r2)) : 
    p2 = 105 :=
by sorry

end stock_price_after_two_years_l195_195160


namespace compute_sqrt_fraction_l195_195180

theorem compute_sqrt_fraction :
  (Real.sqrt ((16^10 + 2^30) / (16^6 + 2^35))) = (256 / Real.sqrt 2049) :=
sorry

end compute_sqrt_fraction_l195_195180


namespace probability_of_satisfaction_l195_195358

-- Definitions for the conditions given in the problem
def dissatisfied_customers_leave_negative_review_probability : ℝ := 0.8
def satisfied_customers_leave_positive_review_probability : ℝ := 0.15
def negative_reviews : ℕ := 60
def positive_reviews : ℕ := 20
def expected_satisfaction_probability : ℝ := 0.64

-- The problem to prove
theorem probability_of_satisfaction :
  ∃ p : ℝ, (dissatisfied_customers_leave_negative_review_probability * (1 - p) = negative_reviews / (negative_reviews + positive_reviews)) ∧
           (satisfied_customers_leave_positive_review_probability * p = positive_reviews / (negative_reviews + positive_reviews)) ∧
           p = expected_satisfaction_probability := 
by
  sorry

end probability_of_satisfaction_l195_195358


namespace complement_of_P_l195_195242

def U : Set ℤ := {-1, 0, 1, 2}
def P : Set ℤ := {x | x^2 < 2}

theorem complement_of_P :
  (U \ P) = {2} :=
by
  sorry

end complement_of_P_l195_195242


namespace shobha_current_age_l195_195993

variable (S B : ℕ)
variable (h_ratio : 4 * B = 3 * S)
variable (h_future_age : S + 6 = 26)

theorem shobha_current_age : B = 15 :=
by
  sorry

end shobha_current_age_l195_195993


namespace always_exists_triangle_l195_195908

variable (a1 a2 a3 a4 d : ℕ)

def arithmetic_sequence (a1 a2 a3 a4 d : ℕ) :=
  a2 = a1 + d ∧ a3 = a1 + 2 * d ∧ a4 = a1 + 3 * d

def positive_terms (a1 a2 a3 a4 : ℕ) :=
  a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0

theorem always_exists_triangle (a1 a2 a3 a4 d : ℕ)
  (h1 : arithmetic_sequence a1 a2 a3 a4 d)
  (h2 : d > 0)
  (h3 : positive_terms a1 a2 a3 a4) :
  a2 + a3 > a4 ∧ a2 + a4 > a3 ∧ a3 + a4 > a2 :=
sorry

end always_exists_triangle_l195_195908


namespace sin_double_angle_l195_195757

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) :
    Real.sin (2 * φ) = 120 / 169 := 
sorry

end sin_double_angle_l195_195757


namespace pascals_triangle_contains_47_once_l195_195207

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l195_195207


namespace Kishore_education_expense_l195_195695

theorem Kishore_education_expense
  (rent milk groceries petrol misc saved : ℝ) -- expenses
  (total_saved_salary : ℝ) -- percentage of saved salary
  (saving_amount : ℝ) -- actual saving
  (total_salary total_expense_children_education : ℝ) -- total salary and expense on children's education
  (H1 : rent = 5000)
  (H2 : milk = 1500)
  (H3 : groceries = 4500)
  (H4 : petrol = 2000)
  (H5 : misc = 3940)
  (H6 : total_saved_salary = 0.10)
  (H7 : saving_amount = 2160)
  (H8 : total_salary = saving_amount / total_saved_salary)
  (H9 : total_expense_children_education = total_salary - (rent + milk + groceries + petrol + misc) - saving_amount) :
  total_expense_children_education = 2600 :=
by 
  simp only [H1, H2, H3, H4, H5, H6, H7] at *
  norm_num at *
  sorry

end Kishore_education_expense_l195_195695


namespace constant_term_expansion_l195_195590

noncomputable def sum_of_coefficients (a : ℕ) : ℕ := sorry

noncomputable def constant_term (a : ℕ) : ℕ := sorry

theorem constant_term_expansion (a : ℕ) (h : sum_of_coefficients a = 2) : constant_term 2 = 10 :=
sorry

end constant_term_expansion_l195_195590


namespace regular_vs_diet_sodas_l195_195152

theorem regular_vs_diet_sodas :
  let regular_cola := 67
  let regular_lemon := 45
  let regular_orange := 23
  let diet_cola := 9
  let diet_lemon := 32
  let diet_orange := 12
  let regular_sodas := regular_cola + regular_lemon + regular_orange
  let diet_sodas := diet_cola + diet_lemon + diet_orange
  regular_sodas - diet_sodas = 82 := sorry

end regular_vs_diet_sodas_l195_195152


namespace bus_speed_express_mode_l195_195969

theorem bus_speed_express_mode (L : ℝ) (t_red : ℝ) (speed_increase : ℝ) (x : ℝ) (normal_speed : ℝ) :
  L = 16 ∧ t_red = 1 / 15 ∧ speed_increase = 8 ∧ normal_speed = x - 8 ∧ 
  (16 / normal_speed - 16 / x = 1 / 15) → x = 48 :=
by
  sorry

end bus_speed_express_mode_l195_195969


namespace triangle_side_lengths_relation_l195_195509

-- Given a triangle ABC with side lengths a, b, c
variables (a b c R d : ℝ)
-- Given orthocenter H and circumcenter O, and the radius of the circumcircle is R,
-- and distance between O and H is d.
-- Prove that a² + b² + c² = 9R² - d²

theorem triangle_side_lengths_relation (a b c R d : ℝ) (H O : Type) (orthocenter : H) (circumcenter : O)
  (radius_circumcircle : O → ℝ)
  (distance_OH : O → H → ℝ) :
  a^2 + b^2 + c^2 = 9 * R^2 - d^2 :=
sorry

end triangle_side_lengths_relation_l195_195509


namespace sum_of_circumferences_eq_28pi_l195_195686

theorem sum_of_circumferences_eq_28pi (R r : ℝ) (h1 : r = (1:ℝ)/3 * R) (h2 : R - r = 7) : 
  2 * Real.pi * R + 2 * Real.pi * r = 28 * Real.pi :=
by
  sorry

end sum_of_circumferences_eq_28pi_l195_195686


namespace probability_of_Z_l195_195250

namespace ProbabilityProof

def P_X : ℚ := 1 / 4
def P_Y : ℚ := 1 / 8
def P_X_or_Y_or_Z : ℚ := 0.4583333333333333

theorem probability_of_Z :
  ∃ P_Z : ℚ, P_Z = 0.0833333333333333 ∧ 
  P_X_or_Y_or_Z = P_X + P_Y + P_Z :=
by
  sorry

end ProbabilityProof

end probability_of_Z_l195_195250


namespace ellipsoid_center_and_axes_sum_l195_195704

theorem ellipsoid_center_and_axes_sum :
  let x₀ := -2
  let y₀ := 3
  let z₀ := 1
  let A := 6
  let B := 4
  let C := 2
  x₀ + y₀ + z₀ + A + B + C = 14 := 
by
  sorry

end ellipsoid_center_and_axes_sum_l195_195704


namespace katie_clock_l195_195937

theorem katie_clock (t_clock t_actual : ℕ) :
  t_clock = 540 →
  t_actual = (540 * 60) / 37 →
  8 * 60 + 875 = 22 * 60 + 36 :=
by
  intros h1 h2
  have h3 : 875 = (540 * 60 / 37) := sorry
  have h4 : 8 * 60 + 875 = 480 + 875 := sorry
  have h5 : 480 + 875 = 22 * 60 + 36 := sorry
  exact h5

end katie_clock_l195_195937


namespace right_triangle_hypotenuse_l195_195549

theorem right_triangle_hypotenuse
  (a b c : ℝ)
  (h₀ : a = 24)
  (h₁ : a^2 + b^2 + c^2 = 2500)
  (h₂ : c^2 = a^2 + b^2) :
  c = 25 * Real.sqrt 2 :=
by
  sorry

end right_triangle_hypotenuse_l195_195549


namespace find_weeks_period_l195_195999

def weekly_addition : ℕ := 3
def bikes_sold : ℕ := 18
def bikes_in_stock : ℕ := 45
def initial_stock : ℕ := 51

theorem find_weeks_period (x : ℕ) :
  initial_stock + weekly_addition * x - bikes_sold = bikes_in_stock ↔ x = 4 := 
by 
  sorry

end find_weeks_period_l195_195999


namespace union_complement_eq_l195_195424

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

end union_complement_eq_l195_195424


namespace first_day_of_month_is_tuesday_l195_195826

theorem first_day_of_month_is_tuesday (day23_is_wednesday : (23 % 7 = 3)) : (1 % 7 = 2) :=
sorry

end first_day_of_month_is_tuesday_l195_195826


namespace arithmetic_sequence_k_l195_195406

theorem arithmetic_sequence_k (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (ha : ∀ n, S (n + 1) = S n + a (n + 1))
  (hS3_S8 : S 3 = S 8) 
  (hS7_Sk : ∃ k, S 7 = S k)
  : ∃ k, k = 4 :=
by
  sorry

end arithmetic_sequence_k_l195_195406


namespace volume_of_each_hemisphere_container_is_correct_l195_195693

-- Define the given conditions
def Total_volume : ℕ := 10936
def Number_containers : ℕ := 2734

-- Define the volume of each hemisphere container
def Volume_each_container : ℕ := Total_volume / Number_containers

-- The theorem to prove, asserting the volume is correct
theorem volume_of_each_hemisphere_container_is_correct :
  Volume_each_container  = 4 := by
  -- placeholder for the actual proof
  sorry

end volume_of_each_hemisphere_container_is_correct_l195_195693


namespace jerry_water_usage_l195_195189

noncomputable def total_water_usage 
  (drinking_cooking : ℕ) 
  (shower_per_gallon : ℕ) 
  (length width height : ℕ) 
  (gallon_per_cubic_ft : ℕ) 
  (number_of_showers : ℕ) 
  : ℕ := 
   drinking_cooking + 
   (number_of_showers * shower_per_gallon) + 
   (length * width * height / gallon_per_cubic_ft)

theorem jerry_water_usage 
  (drinking_cooking : ℕ := 100)
  (shower_per_gallon : ℕ := 20)
  (length : ℕ := 10)
  (width : ℕ := 10)
  (height : ℕ := 6)
  (gallon_per_cubic_ft : ℕ := 1)
  (number_of_showers : ℕ := 15)
  : total_water_usage drinking_cooking shower_per_gallon length width height gallon_per_cubic_ft number_of_showers = 1400 := 
by
  sorry

end jerry_water_usage_l195_195189


namespace simplest_common_denominator_l195_195588

variable (m n a : ℕ)

theorem simplest_common_denominator (h₁ : m > 0) (h₂ : n > 0) (h₃ : a > 0) :
  ∃ l : ℕ, l = 2 * a^2 := 
sorry

end simplest_common_denominator_l195_195588


namespace product_is_cube_l195_195790

/-
  Given conditions:
    - a, b, and c are distinct composite natural numbers.
    - None of a, b, and c are divisible by any of the integers from 2 to 100 inclusive.
    - a, b, and c are the smallest possible numbers satisfying the above conditions.

  We need to prove that their product a * b * c is a cube of a natural number.
-/

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q

theorem product_is_cube (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : is_composite a) (h5 : is_composite b) (h6 : is_composite c)
  (h7 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ a))
  (h8 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ b))
  (h9 : ∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ c))
  (h10 : ∀ (d e f : ℕ), is_composite d → is_composite e → is_composite f → d ≠ e → e ≠ f → d ≠ f → 
         (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ d)) → (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ e)) → (∀ x, 2 ≤ x ∧ x ≤ 100 → ¬ (x ∣ f)) →
         (d * e * f ≥ a * b * c)) :
  ∃ (n : ℕ), a * b * c = n ^ 3 :=
by
  sorry

end product_is_cube_l195_195790


namespace compute_expression_l195_195024

theorem compute_expression : 2 + ((4 * 3 - 2) / 2 * 3) + 5 = 22 :=
by
  -- Place the solution steps if needed
  sorry

end compute_expression_l195_195024


namespace area_between_hexagon_and_square_l195_195864

noncomputable def circleRadius : ℝ := 6

noncomputable def centralAngleSquare : ℝ := Real.pi / 2

noncomputable def centralAngleHexagon : ℝ := Real.pi / 3

noncomputable def areaSegment (r α : ℝ) : ℝ :=
  0.5 * r^2 * (α - Real.sin α)

noncomputable def areaBetweenArcs : ℝ :=
  let r := circleRadius
  let T_AB := areaSegment r centralAngleSquare
  let T_CD := areaSegment r centralAngleHexagon
  2 * (T_AB - T_CD)

theorem area_between_hexagon_and_square :
  abs (areaBetweenArcs - 14.03) < 0.01 :=
by
  sorry

end area_between_hexagon_and_square_l195_195864


namespace feifei_sheep_count_l195_195416

noncomputable def sheep_number (x y : ℕ) : Prop :=
  (y = 3 * x + 15) ∧ (x = y - y / 3)

theorem feifei_sheep_count :
  ∃ x y : ℕ, sheep_number x y ∧ x = 5 :=
sorry

end feifei_sheep_count_l195_195416


namespace coin_probability_l195_195147

theorem coin_probability (p : ℝ) (h1 : p < 1/2) (h2 : (Nat.choose 6 3) * p^3 * (1-p)^3 = 1/20) : p = 1/400 := sorry

end coin_probability_l195_195147


namespace prob_drawing_2_or_3_white_balls_prob_drawing_at_least_1_white_ball_prob_drawing_at_least_1_black_ball_l195_195245

open Finset

-- Define the conditions
def white_balls := 5
def black_balls := 4
def total_balls := white_balls + black_balls
def drawn_balls := 3

-- Define combinations
def choose (n k : ℕ) : ℕ := (nat.choose n k).to_nat

-- Define the events
def event_1 := choose white_balls 2 * choose black_balls 1
def event_2 := choose white_balls 3
def total_event := choose total_balls drawn_balls

-- Define probabilities
def prob_event_1 := (event_1 + event_2) / total_event
def prob_event_2 := 1 - (choose black_balls 3 / total_event)
def prob_event_3 := 1 - (choose white_balls 3 / total_event)

-- Proof problem statements
theorem prob_drawing_2_or_3_white_balls : prob_event_1 = 25 / 42 := by
  sorry

theorem prob_drawing_at_least_1_white_ball : prob_event_2 = 20 / 21 := by
  sorry

theorem prob_drawing_at_least_1_black_ball : prob_event_3 = 37 / 42 := by
  sorry

end prob_drawing_2_or_3_white_balls_prob_drawing_at_least_1_white_ball_prob_drawing_at_least_1_black_ball_l195_195245


namespace min_time_shoe_horses_l195_195853

variable (blacksmiths horses hooves_per_horse minutes_per_hoof : ℕ)
variable (total_time : ℕ)

theorem min_time_shoe_horses (h_blacksmiths : blacksmiths = 48) 
                            (h_horses : horses = 60)
                            (h_hooves_per_horse : hooves_per_horse = 4)
                            (h_minutes_per_hoof : minutes_per_hoof = 5)
                            (h_total_time : total_time = (horses * hooves_per_horse * minutes_per_hoof) / blacksmiths) :
                            total_time = 25 := 
by
  sorry

end min_time_shoe_horses_l195_195853


namespace increase_150_percent_of_80_l195_195626

theorem increase_150_percent_of_80 : 80 * 1.5 + 80 = 200 :=
by
  sorry

end increase_150_percent_of_80_l195_195626


namespace delegate_seating_probability_l195_195128

theorem delegate_seating_probability :
  ∃ m n : ℕ, nat.coprime m n ∧ (m + n = 909) ∧
  (∑ i in range 12, 1 : ℕ) = 12 ∧
  (∑ c in range 3, 4 : ℕ) = 12 ∧
  (∃ (p : ℕ), p = 409 ∧ ∃ (q : ℕ), q = 500 ∧
  ∃ (prob : ℚ), prob = (p / q) ∧ m = p ∧ n = q) :=
by
  sorry

end delegate_seating_probability_l195_195128


namespace solve_for_a_l195_195731

theorem solve_for_a (a : ℚ) (h : a + a/3 + a/4 = 11/4) : a = 33/19 :=
sorry

end solve_for_a_l195_195731


namespace yoongi_correct_calculation_l195_195458

theorem yoongi_correct_calculation (x : ℕ) (h : x + 9 = 30) : x - 7 = 14 :=
sorry

end yoongi_correct_calculation_l195_195458


namespace solve_equation_l195_195273

theorem solve_equation (x : ℝ) (h : x ≠ 1) : 
  1 / (x - 1) + 1 = 3 / (2 * x - 2) ↔ x = 3 / 2 := by
  sorry

end solve_equation_l195_195273


namespace average_math_score_first_year_students_l195_195246

theorem average_math_score_first_year_students 
  (total_male_students : ℕ) (total_female_students : ℕ)
  (sample_size : ℕ) (avg_score_male : ℕ) (avg_score_female : ℕ)
  (male_sample_size female_sample_size : ℕ)
  (weighted_avg : ℚ) :
  total_male_students = 300 → 
  total_female_students = 200 →
  sample_size = 60 → 
  avg_score_male = 110 →
  avg_score_female = 100 →
  male_sample_size = (3 * sample_size) / 5 →
  female_sample_size = (2 * sample_size) / 5 →
  weighted_avg = (male_sample_size * avg_score_male + female_sample_size * avg_score_female : ℕ) / sample_size → 
  weighted_avg = 106 := 
by
  sorry

end average_math_score_first_year_students_l195_195246


namespace remainder_of_m_div_5_l195_195673

theorem remainder_of_m_div_5 (m n : ℕ) (hpos : 0 < m) (hdef : m = 15 * n - 1) : m % 5 = 4 :=
sorry

end remainder_of_m_div_5_l195_195673


namespace solve_frac_eq_l195_195119

-- Define the fractional function
def frac_eq (x : ℝ) : Prop := (x + 2) / (x - 1) = 0

-- State the theorem
theorem solve_frac_eq : frac_eq (-2) :=
by
  unfold frac_eq
  -- Use sorry to skip the proof
  sorry

end solve_frac_eq_l195_195119


namespace last_row_number_l195_195967

/-
Given:
1. Each row forms an arithmetic sequence.
2. The common differences of the rows are:
   - 1st row: common difference = 1
   - 2nd row: common difference = 2
   - 3rd row: common difference = 4
   - ...
   - 2015th row: common difference = 2^2014
3. The nth row starts with \( (n+1) \times 2^{n-2} \).

Prove:
The number in the last row (2016th row) is \( 2017 \times 2^{2014} \).
-/
theorem last_row_number
  (common_diff : ℕ → ℕ)
  (h1 : common_diff 1 = 1)
  (h2 : common_diff 2 = 2)
  (h3 : common_diff 3 = 4)
  (h_general : ∀ n, common_diff n = 2^(n-1))
  (first_number_in_row : ℕ → ℕ)
  (first_number_in_row_def : ∀ n, first_number_in_row n = (n + 1) * 2^(n - 2)) :
  first_number_in_row 2016 = 2017 * 2^2014 := by
    sorry

end last_row_number_l195_195967


namespace no_non_degenerate_triangle_l195_195983

theorem no_non_degenerate_triangle 
  (a b c : ℕ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ c) 
  (h3 : a ≠ c) 
  (h4 : Nat.gcd a (Nat.gcd b c) = 1) 
  (h5 : a ∣ (b - c) * (b - c)) 
  (h6 : b ∣ (a - c) * (a - c)) 
  (h7 : c ∣ (a - b) * (a - b)) : 
  ¬ (a < b + c ∧ b < a + c ∧ c < a + b) := 
sorry

end no_non_degenerate_triangle_l195_195983


namespace find_number_l195_195104

variable (number : ℤ)

theorem find_number (h : number - 44 = 15) : number = 59 := 
sorry

end find_number_l195_195104


namespace scientific_notation_of_5_35_million_l195_195350

theorem scientific_notation_of_5_35_million : 
  (5.35 : ℝ) * 10^6 = 5.35 * 10^6 := 
by
  sorry

end scientific_notation_of_5_35_million_l195_195350


namespace sin_double_angle_l195_195762

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l195_195762


namespace vartan_recreation_l195_195557

noncomputable def vartan_recreation_percent (W : ℝ) (P : ℝ) : Prop := 
  let W_this_week := 0.9 * W
  let recreation_last_week := (P / 100) * W
  let recreation_this_week := 0.3 * W_this_week
  recreation_this_week = 1.8 * recreation_last_week

theorem vartan_recreation (W : ℝ) : ∀ P : ℝ, vartan_recreation_percent W P → P = 15 := 
by
  intro P h
  unfold vartan_recreation_percent at h
  sorry

end vartan_recreation_l195_195557


namespace average_ABC_eq_2A_plus_3_l195_195025

theorem average_ABC_eq_2A_plus_3 (A B C : ℝ) 
  (h1 : 2023 * C - 4046 * A = 8092) 
  (h2 : 2023 * B - 6069 * A = 10115) : 
  (A + B + C) / 3 = 2 * A + 3 :=
sorry

end average_ABC_eq_2A_plus_3_l195_195025


namespace find_a_l195_195522

variable (a : ℝ)

def A := ({1, 2, a} : Set ℝ)
def B := ({1, a^2 - a} : Set ℝ)

theorem find_a (h : B a ⊆ A a) : a = -1 ∨ a = 0 :=
  sorry

end find_a_l195_195522


namespace number_of_participants_l195_195156

theorem number_of_participants (n : ℕ) (hn : n = 862) 
    (h_lower : 575 ≤ n * 2 / 3) 
    (h_upper : n * 7 / 9 ≤ 670) : 
    ∃ p, (575 ≤ p) ∧ (p ≤ 670) ∧ (p % 11 = 0) ∧ ((p - 575) / 11 + 1 = 8) :=
by
  sorry

end number_of_participants_l195_195156


namespace games_within_division_l195_195334

/-- 
Given a baseball league with two four-team divisions,
where each team plays N games against other teams in its division,
and M games against teams in the other division.
Given that N > 2M and M > 6, and each team plays a total of 92 games in a season,
prove that each team plays 60 games within its own division.
-/
theorem games_within_division (N M : ℕ) (hN : N > 2 * M) (hM : M > 6) (h_total : 3 * N + 4 * M = 92) :
  3 * N = 60 :=
by
  -- The proof is omitted.
  sorry

end games_within_division_l195_195334


namespace lassis_from_mangoes_l195_195481

theorem lassis_from_mangoes (L M : ℕ) (h : 2 * L = 11 * M) : 12 * L = 66 :=
by sorry

end lassis_from_mangoes_l195_195481


namespace probability_of_draw_l195_195289

-- Let P be the probability of the game ending in a draw.
-- Let PA be the probability of Player A winning.

def PA_not_losing := 0.8
def PB_not_losing := 0.7

theorem probability_of_draw : ¬ (1 - PA_not_losing + PB_not_losing ≠ 1.5) → PA_not_losing + (1 - PB_not_losing) = 1.5 → PB_not_losing + 0.5 = 1 := by
  intros
  sorry

end probability_of_draw_l195_195289


namespace initial_percentage_correct_l195_195144

noncomputable def percentInitiallyFull (initialWater: ℕ) (waterAdded: ℕ) (fractionFull: ℚ) (capacity: ℕ) : ℚ :=
  (initialWater : ℚ) / (capacity : ℚ) * 100

theorem initial_percentage_correct (initialWater waterAdded capacity: ℕ) (fractionFull: ℚ) :
  waterAdded = 14 →
  fractionFull = 3/4 →
  capacity = 40 →
  initialWater + waterAdded = fractionFull * capacity →
  percentInitiallyFull initialWater waterAdded fractionFull capacity = 40 :=
by
  intros h1 h2 h3 h4
  unfold percentInitiallyFull
  sorry

end initial_percentage_correct_l195_195144


namespace diagramD_non_eulerian_l195_195322

-- Given conditions
def hasEulerianPath (G : SimpleGraph) : Prop :=
  let odd_degree_vertices := {v | G.degree v % 2 = 1}
  odd_degree_vertices.card = 0 ∨ odd_degree_vertices.card = 2

-- Given information for our specific diagrams (We create a specific name DiagramD to be explicit)
variable (DiagramD : SimpleGraph)

-- The theorem to prove
theorem diagramD_non_eulerian (h : {v | DiagramD.degree v % 2 = 1}.card ≠ 0 ∧ {v | DiagramD.degree v % 2 = 1}.card ≠ 2) : ¬ hasEulerianPath DiagramD :=
by sorry

end diagramD_non_eulerian_l195_195322


namespace find_non_negative_integers_l195_195374

def has_exactly_two_distinct_solutions (a : ℕ) (m : ℕ) : Prop :=
  ∃ (x₁ x₂ : ℕ), (x₁ < m) ∧ (x₂ < m) ∧ (x₁ ≠ x₂) ∧ (x₁^2 + a) % m = 0 ∧ (x₂^2 + a) % m = 0

theorem find_non_negative_integers (a : ℕ) (m : ℕ := 2007) : 
  a < m ∧ has_exactly_two_distinct_solutions a m ↔ a = 446 ∨ a = 1115 ∨ a = 1784 :=
sorry

end find_non_negative_integers_l195_195374


namespace heap_holds_20_sheets_l195_195365

theorem heap_holds_20_sheets :
  ∀ (num_bundles num_bunches num_heaps sheets_per_bundle sheets_per_bunch total_sheets : ℕ),
    num_bundles = 3 →
    num_bunches = 2 →
    num_heaps = 5 →
    sheets_per_bundle = 2 →
    sheets_per_bunch = 4 →
    total_sheets = 114 →
    (total_sheets - (num_bundles * sheets_per_bundle + num_bunches * sheets_per_bunch)) / num_heaps = 20 := 
by
  intros num_bundles num_bunches num_heaps sheets_per_bundle sheets_per_bunch total_sheets 
  intros h1 h2 h3 h4 h5 h6 
  sorry

end heap_holds_20_sheets_l195_195365


namespace find_fraction_of_cistern_l195_195004

noncomputable def fraction_initially_full (x : ℝ) : Prop :=
  let rateA := (1 - x) / 12
  let rateB := (1 - x) / 8
  let combined_rate := 1 / 14.4
  combined_rate = rateA + rateB

theorem find_fraction_of_cistern {x : ℝ} (h : fraction_initially_full x) : x = 2 / 3 :=
by
  sorry

end find_fraction_of_cistern_l195_195004


namespace total_cost_of_antibiotics_l195_195169

-- Definitions based on the conditions
def cost_A_per_dose : ℝ := 3
def cost_B_per_dose : ℝ := 4.50
def doses_per_day_A : ℕ := 2
def days_A : ℕ := 3
def doses_per_day_B : ℕ := 1
def days_B : ℕ := 4

-- Total cost calculations
def total_cost_A : ℝ := days_A * doses_per_day_A * cost_A_per_dose
def total_cost_B : ℝ := days_B * doses_per_day_B * cost_B_per_dose

-- Final proof statement
theorem total_cost_of_antibiotics : total_cost_A + total_cost_B = 36 :=
by
  -- The proof goes here
  sorry

end total_cost_of_antibiotics_l195_195169


namespace midpoint_trajectory_l195_195506

-- Define the parabola and line intersection conditions
def parabola (x y : ℝ) : Prop := x^2 = 4 * y

def line_through_focus (A B : ℝ × ℝ) (focus : ℝ × ℝ) : Prop :=
  ∃ m b : ℝ, (∀ P ∈ [A, B, focus], P.2 = m * P.1 + b)

def midpoint (A B M : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

theorem midpoint_trajectory (A B M : ℝ × ℝ) (focus : ℝ × ℝ):
  (parabola A.1 A.2) ∧ (parabola B.1 B.2) ∧ (line_through_focus A B focus) ∧ (midpoint A B M)
  → (M.1 ^ 2 = 2 * M.2 - 2) :=
by
  sorry

end midpoint_trajectory_l195_195506


namespace isosceles_triangle_perimeter_l195_195739

theorem isosceles_triangle_perimeter
  (a b : ℕ)
  (ha : a = 3)
  (hb : b = 7)
  (h_iso : ∃ (c : ℕ), (c = a ∨ c = b) ∧ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  a + b + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l195_195739


namespace cross_product_correct_l195_195040

def u : ℝ × ℝ × ℝ := (4, 3, -2)
def v : ℝ × ℝ × ℝ := (-1, 2, 5)
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_product_correct :
  cross_product u v = (19, -18, 11) :=
by
  have h1 : cross_product u v = (19, -18, 11) := sorry
  exact h1

end cross_product_correct_l195_195040


namespace mary_blue_marbles_l195_195882

theorem mary_blue_marbles (dan_blue_marbles mary_blue_marbles : ℕ)
  (h1 : dan_blue_marbles = 5)
  (h2 : mary_blue_marbles = 2 * dan_blue_marbles) : mary_blue_marbles = 10 := 
by
  sorry

end mary_blue_marbles_l195_195882


namespace gwen_total_books_l195_195203

theorem gwen_total_books
  (mystery_shelves : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ)
  (mystery_shelves_count : mystery_shelves = 3)
  (picture_shelves_count : picture_shelves = 5)
  (each_shelf_books : books_per_shelf = 9) :
  (mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf) = 72 := by
  sorry

end gwen_total_books_l195_195203


namespace find_plaintext_from_ciphertext_l195_195984

theorem find_plaintext_from_ciphertext : 
  ∃ x : ℕ, ∀ a : ℝ, (a^3 - 2 = 6) → (1022 = a^x - 2) → x = 10 :=
by
  use 10
  intros a ha hc
  -- Proof omitted
  sorry

end find_plaintext_from_ciphertext_l195_195984


namespace find_sam_current_age_l195_195749

def Drew_current_age : ℕ := 12

def Drew_age_in_five_years : ℕ := Drew_current_age + 5

def Sam_age_in_five_years : ℕ := 3 * Drew_age_in_five_years

def Sam_current_age : ℕ := Sam_age_in_five_years - 5

theorem find_sam_current_age : Sam_current_age = 46 := by
  sorry

end find_sam_current_age_l195_195749


namespace remainder_7n_mod_4_l195_195311

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l195_195311


namespace regression_correlation_relation_l195_195241

variable (b r : ℝ)

theorem regression_correlation_relation (h : b = 0) : r = 0 := 
sorry

end regression_correlation_relation_l195_195241


namespace earning_80_yuan_represents_l195_195753

-- Defining the context of the problem
def spending (n : Int) : Int := -n
def earning (n : Int) : Int := n

-- The problem statement as a Lean theorem
theorem earning_80_yuan_represents (x : Int) (hx : earning x = 80) : x = 80 := 
by
  sorry

end earning_80_yuan_represents_l195_195753


namespace sewers_handle_rain_l195_195586

theorem sewers_handle_rain (total_capacity : ℕ) (runoff_per_hour : ℕ) : 
  total_capacity = 240000 → 
  runoff_per_hour = 1000 → 
  total_capacity / runoff_per_hour / 24 = 10 :=
by 
  intro h1 h2
  sorry

end sewers_handle_rain_l195_195586


namespace minimum_volume_sum_l195_195827

section pyramid_volume

variables {R : Type*} [OrderedRing R]
variables {V : Type*} [AddCommGroup V] [Module R V]

-- Define the volumes of the pyramids
variables (V_SABR1 V_SR2P2R3Q2 V_SCDR4 : R)
variables (V_SR1P1R2Q1 V_SR3P3R4Q3 : R)

-- Given condition
axiom volume_condition : V_SR1P1R2Q1 + V_SR3P3R4Q3 = 78

-- The theorem to be proved
theorem minimum_volume_sum : 
  V_SABR1^2 + V_SR2P2R3Q2^2 + V_SCDR4^2 ≥ 2028 :=
sorry

end pyramid_volume

end minimum_volume_sum_l195_195827


namespace arithmetic_geometric_mean_inequality_l195_195960

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end arithmetic_geometric_mean_inequality_l195_195960


namespace determine_a_square_binomial_l195_195887

theorem determine_a_square_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, (4 * x^2 + 16 * x + a) = (2 * x + b)^2) → a = 16 := 
by
  sorry

end determine_a_square_binomial_l195_195887


namespace find_triples_l195_195891

theorem find_triples (x y z : ℕ) :
  (1 / x + 2 / y - 3 / z = 1) ↔ 
  ((x = 2 ∧ y = 1 ∧ z = 2) ∨
   (x = 2 ∧ y = 3 ∧ z = 18) ∨
   ∃ (n : ℕ), n ≥ 1 ∧ x = 1 ∧ y = 2 * n ∧ z = 3 * n ∨
   ∃ (k : ℕ), k ≥ 1 ∧ x = k ∧ y = 2 ∧ z = 3 * k) := sorry

end find_triples_l195_195891


namespace increase_by_percentage_l195_195604

theorem increase_by_percentage (a : ℤ) (b : ℚ) : ((b + 1) * a) = 200 :=
by
  -- Definitions based directly on conditions
  let a := (80 : ℤ)
  let b := (1.5 : ℚ)
  -- Assertions about the equivalent proof problem
  have h1 : b + 1 = 2.5 := by norm_num
  have h2 : (b + 1) * a = 200 := by norm_num
  exact h2

end increase_by_percentage_l195_195604


namespace price_per_glass_first_day_l195_195813

theorem price_per_glass_first_day 
(O G : ℝ) (H : 2 * O * G * P₁ = 3 * O * G * 0.5466666666666666 ) : 
  P₁ = 0.82 :=
by
  sorry

end price_per_glass_first_day_l195_195813


namespace sin_2phi_l195_195776

theorem sin_2phi (φ : ℝ) (h : 7 / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_2phi_l195_195776
