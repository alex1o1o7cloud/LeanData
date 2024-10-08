import Mathlib

namespace peacocks_in_zoo_l80_80482

theorem peacocks_in_zoo :
  ∃ p t : ℕ, 2 * p + 4 * t = 54 ∧ p + t = 17 ∧ p = 7 :=
by
  sorry

end peacocks_in_zoo_l80_80482


namespace num_circles_rectangle_l80_80042

structure Rectangle (α : Type*) [Field α] :=
  (A B C D : α × α)
  (AB_parallel_CD : B.1 = A.1 ∧ D.1 = C.1)
  (AD_parallel_BC : D.2 = A.2 ∧ C.2 = B.2)

def num_circles_with_diameter_vertices (R : Rectangle ℝ) : ℕ :=
  sorry

theorem num_circles_rectangle (R : Rectangle ℝ) : num_circles_with_diameter_vertices R = 5 :=
  sorry

end num_circles_rectangle_l80_80042


namespace exists_b_gt_a_divides_l80_80534

theorem exists_b_gt_a_divides (a : ℕ) (h : 0 < a) :
  ∃ b : ℕ, b > a ∧ (1 + 2^a + 3^a) ∣ (1 + 2^b + 3^b) :=
sorry

end exists_b_gt_a_divides_l80_80534


namespace sum_of_roots_of_y_squared_eq_36_l80_80411

theorem sum_of_roots_of_y_squared_eq_36 :
  (∀ y : ℝ, y^2 = 36 → y = 6 ∨ y = -6) → (6 + (-6) = 0) :=
by
  sorry

end sum_of_roots_of_y_squared_eq_36_l80_80411


namespace value_of_a_l80_80521

theorem value_of_a :
  ∀ (g : ℝ → ℝ), (∀ x, g x = 5*x - 7) → ∃ a, g a = 0 ∧ a = 7 / 5 :=
by
  sorry

end value_of_a_l80_80521


namespace jesse_pencils_l80_80414

def initial_pencils : ℕ := 78
def pencils_given : ℕ := 44
def final_pencils : ℕ := initial_pencils - pencils_given

theorem jesse_pencils :
  final_pencils = 34 :=
by
  -- Proof goes here
  sorry

end jesse_pencils_l80_80414


namespace Mildred_heavier_than_Carol_l80_80942

-- Definition of weights for Mildred and Carol
def weight_Mildred : ℕ := 59
def weight_Carol : ℕ := 9

-- Definition of how much heavier Mildred is than Carol
def weight_difference : ℕ := weight_Mildred - weight_Carol

-- The theorem stating the difference in weight
theorem Mildred_heavier_than_Carol : weight_difference = 50 := 
by 
  -- Just state the theorem without providing the actual steps (proof skipped)
  sorry

end Mildred_heavier_than_Carol_l80_80942


namespace unique_solution_values_l80_80064

theorem unique_solution_values (a : ℝ) :
  (∃! x : ℝ, a * x^2 - x + 1 = 0) ↔ (a = 0 ∨ a = 1 / 4) :=
by
  sorry

end unique_solution_values_l80_80064


namespace gum_sharing_l80_80698

theorem gum_sharing (john cole aubrey : ℕ) (sharing_people : ℕ) 
  (hj : john = 54) (hc : cole = 45) (ha : aubrey = 0) 
  (hs : sharing_people = 3) : 
  john + cole + aubrey = 99 ∧ (john + cole + aubrey) / sharing_people = 33 := 
by
  sorry

end gum_sharing_l80_80698


namespace inequality_problem_l80_80296

-- Define the problem conditions and goal
theorem inequality_problem (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) : 
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := 
sorry

end inequality_problem_l80_80296


namespace two_digit_number_representation_l80_80440

-- Define the conditions and the problem statement in Lean 4
def units_digit (n : ℕ) := n % 10
def tens_digit (n : ℕ) := (n / 10) % 10

theorem two_digit_number_representation (x : ℕ) (h : x < 10) :
  ∃ n : ℕ, units_digit n = x ∧ tens_digit n = 2 * x ^ 2 ∧ n = 20 * x ^ 2 + x :=
by {
  sorry
}

end two_digit_number_representation_l80_80440


namespace intersection_infinite_l80_80993

-- Define the equations of the curves
def curve1 (x y : ℝ) : Prop := 2 * x^2 - x * y - y^2 - x - 2 * y - 1 = 0
def curve2 (x y : ℝ) : Prop := 3 * x^2 - 4 * x * y + y^2 - 3 * x + y = 0

-- Theorem statement
theorem intersection_infinite : ∃ (f : ℝ → ℝ), ∀ x, curve1 x (f x) ∧ curve2 x (f x) :=
sorry

end intersection_infinite_l80_80993


namespace same_solution_sets_l80_80732

theorem same_solution_sets (a : ℝ) :
  (∀ x : ℝ, 3 * x - 5 < a ↔ 2 * x < 4) → a = 1 := 
by
  sorry

end same_solution_sets_l80_80732


namespace question_1_question_2_l80_80706

noncomputable def f (x m : ℝ) : ℝ := abs (x + m) - abs (2 * x - 2 * m)

theorem question_1 (x : ℝ) (m : ℝ) (h : m = 1/2) (h_pos : m > 0) : 
  (f x m ≥ 1/2) ↔ (1/3 ≤ x ∧ x < 1) :=
sorry

theorem question_2 (m : ℝ) (h_pos : m > 0) : 
  (∀ x : ℝ, ∃ t : ℝ, f x m + abs (t - 3) < abs (t + 4)) ↔ (0 < m ∧ m < 7/2) :=
sorry

end question_1_question_2_l80_80706


namespace engineering_students_pass_percentage_l80_80840

theorem engineering_students_pass_percentage :
  let num_male_students := 120
  let num_female_students := 100
  let perc_male_eng_students := 0.25
  let perc_female_eng_students := 0.20
  let perc_male_eng_pass := 0.20
  let perc_female_eng_pass := 0.25
  
  let num_male_eng_students := num_male_students * perc_male_eng_students
  let num_female_eng_students := num_female_students * perc_female_eng_students
  
  let num_male_eng_pass := num_male_eng_students * perc_male_eng_pass
  let num_female_eng_pass := num_female_eng_students * perc_female_eng_pass
  
  let total_eng_students := num_male_eng_students + num_female_eng_students
  let total_eng_pass := num_male_eng_pass + num_female_eng_pass
  
  (total_eng_pass / total_eng_students) * 100 = 22 :=
by
  sorry

end engineering_students_pass_percentage_l80_80840


namespace avg_of_6_10_N_is_10_if_even_l80_80797

theorem avg_of_6_10_N_is_10_if_even (N : ℕ) (h1 : 9 ≤ N) (h2 : N ≤ 17) (h3 : (6 + 10 + N) % 2 = 0) : (6 + 10 + N) / 3 = 10 :=
by
-- sorry is placed here since we are not including the actual proof
sorry

end avg_of_6_10_N_is_10_if_even_l80_80797


namespace greatest_number_of_sets_l80_80395

-- Definitions based on conditions
def whitney_tshirts := 5
def whitney_buttons := 24
def whitney_stickers := 12
def buttons_per_set := 2
def stickers_per_set := 1

-- The statement to prove the greatest number of identical sets Whitney can make
theorem greatest_number_of_sets : 
  ∃ max_sets : ℕ, 
  max_sets = whitney_tshirts ∧ 
  max_sets ≤ (whitney_buttons / buttons_per_set) ∧
  max_sets ≤ (whitney_stickers / stickers_per_set) :=
sorry

end greatest_number_of_sets_l80_80395


namespace remaining_miles_l80_80802

theorem remaining_miles (total_miles : ℕ) (driven_miles : ℕ) (h1: total_miles = 1200) (h2: driven_miles = 642) :
  total_miles - driven_miles = 558 :=
by
  sorry

end remaining_miles_l80_80802


namespace sugar_fill_count_l80_80849

noncomputable def sugar_needed_for_one_batch : ℚ := 3 + 1/2
noncomputable def total_batches : ℕ := 2
noncomputable def cup_capacity : ℚ := 1/3
noncomputable def total_sugar_needed : ℚ := total_batches * sugar_needed_for_one_batch

theorem sugar_fill_count : (total_sugar_needed / cup_capacity) = 21 :=
by
  -- Assuming necessary preliminary steps already defined, we just check the equality directly
  sorry

end sugar_fill_count_l80_80849


namespace cubes_difference_divisible_91_l80_80451

theorem cubes_difference_divisible_91 (cubes : Fin 16 → ℤ) (h : ∀ n : Fin 16, ∃ m : ℤ, cubes n = m^3) :
  ∃ (a b : Fin 16), a ≠ b ∧ 91 ∣ (cubes a - cubes b) :=
sorry

end cubes_difference_divisible_91_l80_80451


namespace bailey_total_spending_l80_80422

noncomputable def cost_after_discount : ℝ :=
  let guest_sets := 2
  let master_sets := 4
  let guest_price := 40.0
  let master_price := 50.0
  let discount := 0.20
  let total_cost := (guest_sets * guest_price) + (master_sets * master_price)
  let discount_amount := total_cost * discount
  total_cost - discount_amount

theorem bailey_total_spending : cost_after_discount = 224.0 :=
by
  unfold cost_after_discount
  sorry

end bailey_total_spending_l80_80422


namespace marvin_substitute_correct_l80_80483

theorem marvin_substitute_correct {a b c d f : ℤ} (ha : a = 3) (hb : b = 4) (hc : c = 7) (hd : d = 5) :
  (a + (b - (c + (d - f))) = 5 - f) → f = 5 :=
sorry

end marvin_substitute_correct_l80_80483


namespace find_principal_amount_l80_80668

theorem find_principal_amount
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (h1 : SI = 4034.25)
  (h2 : R = 9)
  (h3 : T = 5) :
  P = 8965 :=
by
  sorry

end find_principal_amount_l80_80668


namespace max_value_of_expr_l80_80657

-- Define the initial conditions and expression 
def initial_ones (n : ℕ) := List.replicate n 1

-- Given that we place "+" or ")(" between consecutive ones
def max_possible_value (n : ℕ) : ℕ := sorry

theorem max_value_of_expr : max_possible_value 2013 = 3 ^ 671 := 
sorry

end max_value_of_expr_l80_80657


namespace shopkeeper_marked_price_l80_80736

theorem shopkeeper_marked_price 
  (L C M S : ℝ)
  (h1 : C = 0.75 * L)
  (h2 : C = 0.75 * S)
  (h3 : S = 0.85 * M) :
  M = 1.17647 * L :=
sorry

end shopkeeper_marked_price_l80_80736


namespace trigonometric_identity_l80_80141

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  (1 + Real.cos α ^ 2) / (Real.sin α * Real.cos α + Real.sin α ^ 2) = 11 / 12 :=
by
  sorry

end trigonometric_identity_l80_80141


namespace floor_add_self_eq_20_5_iff_l80_80737

theorem floor_add_self_eq_20_5_iff (s : ℝ) : (⌊s⌋₊ : ℝ) + s = 20.5 ↔ s = 10.5 :=
by
  sorry

end floor_add_self_eq_20_5_iff_l80_80737


namespace tangent_line_eqn_when_a_zero_min_value_f_when_a_zero_range_of_a_for_x_ge_zero_exp_x_ln_x_plus_one_gt_x_sq_l80_80416

noncomputable def f (a : ℝ) (x : ℝ) := Real.exp x - 1 - x - a * x ^ 2

theorem tangent_line_eqn_when_a_zero :
  (∀ x, y = f 0 x → y - (Real.exp 1 - 2) = (Real.exp 1 - 1) * (x - 1)) :=
sorry

theorem min_value_f_when_a_zero :
  (∀ x : ℝ, f 0 x >= f 0 0) := 
sorry

theorem range_of_a_for_x_ge_zero (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f a x ≥ 0) → (a ≤ 1/2) :=
sorry

theorem exp_x_ln_x_plus_one_gt_x_sq (x : ℝ) :
  x > 0 → ((Real.exp x - 1) * Real.log (x + 1) > x ^ 2) :=
sorry

end tangent_line_eqn_when_a_zero_min_value_f_when_a_zero_range_of_a_for_x_ge_zero_exp_x_ln_x_plus_one_gt_x_sq_l80_80416


namespace hexagon_perimeter_arithmetic_sequence_l80_80948

theorem hexagon_perimeter_arithmetic_sequence :
  let a₁ := 10
  let a₂ := 12
  let a₃ := 14
  let a₄ := 16
  let a₅ := 18
  let a₆ := 20
  let lengths := [a₁, a₂, a₃, a₄, a₅, a₆]
  let perimeter := lengths.sum
  perimeter = 90 :=
by
  sorry

end hexagon_perimeter_arithmetic_sequence_l80_80948


namespace find_pairs_l80_80237

open Nat

-- m and n are odd natural numbers greater than 2009
def is_odd_gt_2009 (x : ℕ) : Prop := (x % 2 = 1) ∧ (x > 2009)

-- condition: m divides n^2 + 8
def divides_m_n_squared_plus_8 (m n : ℕ) : Prop := m ∣ (n ^ 2 + 8)

-- condition: n divides m^2 + 8
def divides_n_m_squared_plus_8 (m n : ℕ) : Prop := n ∣ (m ^ 2 + 8)

-- Final statement
theorem find_pairs :
  ∃ m n : ℕ, is_odd_gt_2009 m ∧ is_odd_gt_2009 n ∧ divides_m_n_squared_plus_8 m n ∧ divides_n_m_squared_plus_8 m n ∧ ((m, n) = (881, 89) ∨ (m, n) = (3303, 567)) :=
sorry

end find_pairs_l80_80237


namespace solve_for_b_l80_80251

theorem solve_for_b (b : ℝ) (h : b + b / 4 = 10 / 4) : b = 2 :=
sorry

end solve_for_b_l80_80251


namespace total_amount_paid_l80_80552

theorem total_amount_paid (B : ℕ) (hB : B = 232) (A : ℕ) (hA : A = 3 / 2 * B) :
  A + B = 580 :=
by
  sorry

end total_amount_paid_l80_80552


namespace sheets_in_stack_l80_80095

theorem sheets_in_stack (n : ℕ) (thickness : ℝ) (height : ℝ) 
  (h1 : n = 400) (h2 : thickness = 4) (h3 : height = 10) : 
  n * height / thickness = 1000 := 
by 
  sorry

end sheets_in_stack_l80_80095


namespace students_making_stars_l80_80032

theorem students_making_stars (total_stars stars_per_student : ℕ) (h1 : total_stars = 372) (h2 : stars_per_student = 3) : 
  total_stars / stars_per_student = 124 :=
by
  sorry

end students_making_stars_l80_80032


namespace weight_of_new_person_l80_80836

/-- The average weight of 10 persons increases by 7.2 kg when a new person
replaces one who weighs 65 kg. Prove that the weight of the new person is 137 kg. -/
theorem weight_of_new_person (W_new : ℝ) (W_old : ℝ) (n : ℝ) (increase : ℝ) 
  (h1 : W_old = 65) (h2 : n = 10) (h3 : increase = 7.2) 
  (h4 : W_new = W_old + n * increase) : W_new = 137 := 
by
  -- proof to be done later
  sorry

end weight_of_new_person_l80_80836


namespace solve_for_x_l80_80040

theorem solve_for_x (x : ℤ) (h : 20 * 14 + x = 20 + 14 * x) : x = 20 := 
by 
  sorry

end solve_for_x_l80_80040


namespace find_slope_l80_80189

theorem find_slope (m : ℝ) : 
    (∀ x : ℝ, (2, 13) = (x, 5 * x + 3)) → 
    (∀ x : ℝ, (2, 13) = (x, m * x + 1)) → 
    m = 6 :=
by 
  intros hP hQ
  have h_inter_p := hP 2
  have h_inter_q := hQ 2
  simp at h_inter_p h_inter_q
  have : 13 = 5 * 2 + 3 := h_inter_p
  have : 13 = m * 2 + 1 := h_inter_q
  linarith

end find_slope_l80_80189


namespace fraction_of_AD_eq_BC_l80_80847

theorem fraction_of_AD_eq_BC (x y : ℝ) (B C D A : ℝ) 
  (h1 : B < C) 
  (h2 : C < D)
  (h3 : D < A) 
  (hBD : B < D)
  (hCD : C < D)
  (hAD : A = D)
  (hAB : A - B = 3 * (D - B)) 
  (hAC : A - C = 7 * (D - C))
  (hx_eq : x = 2 * y) 
  (hADx : A - D = 4 * x)
  (hADy : A - D = 8 * y)
  : (C - B) = 1/8 * (A - D) := 
sorry

end fraction_of_AD_eq_BC_l80_80847


namespace xn_plus_inv_xn_is_integer_l80_80295

theorem xn_plus_inv_xn_is_integer (x : ℝ) (hx : x ≠ 0) (k : ℤ) (h : x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
by sorry

end xn_plus_inv_xn_is_integer_l80_80295


namespace sum_of_squares_of_sums_l80_80474

axiom roots_of_polynomial (p q r : ℝ) : p^3 - 15*p^2 + 25*p - 12 = 0 ∧ q^3 - 15*q^2 + 25*q - 12 = 0 ∧ r^3 - 15*r^2 + 25*r - 12 = 0

theorem sum_of_squares_of_sums (p q r : ℝ)
  (h_roots : p^3 - 15*p^2 + 25*p - 12 = 0 ∧ q^3 - 15*q^2 + 25*q - 12 = 0 ∧ r^3 - 15*r^2 + 25*r - 12 = 0) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := 
sorry

end sum_of_squares_of_sums_l80_80474


namespace determine_m_first_degree_inequality_l80_80413

theorem determine_m_first_degree_inequality (m : ℝ) (x : ℝ) :
  (m + 1) * x ^ |m| + 2 > 0 → |m| = 1 → m = 1 :=
by
  intro h1 h2
  sorry

end determine_m_first_degree_inequality_l80_80413


namespace comparison_of_abc_l80_80235

noncomputable def a : ℝ := (4 - Real.log 4) / Real.exp 2
noncomputable def b : ℝ := Real.log 2 / 2
noncomputable def c : ℝ := 1 / Real.exp 1

theorem comparison_of_abc : b < a ∧ a < c :=
by
  sorry

end comparison_of_abc_l80_80235


namespace elaineExpenseChanges_l80_80262

noncomputable def elaineIncomeLastYear : ℝ := 20000 + 5000
noncomputable def elaineExpensesLastYearRent := 0.10 * elaineIncomeLastYear
noncomputable def elaineExpensesLastYearGroceries := 0.20 * elaineIncomeLastYear
noncomputable def elaineExpensesLastYearHealthcare := 0.15 * elaineIncomeLastYear
noncomputable def elaineTotalExpensesLastYear := elaineExpensesLastYearRent + elaineExpensesLastYearGroceries + elaineExpensesLastYearHealthcare
noncomputable def elaineSavingsLastYear := elaineIncomeLastYear - elaineTotalExpensesLastYear

noncomputable def elaineIncomeThisYear : ℝ := 23000 + 10000
noncomputable def elaineExpensesThisYearRent := 0.30 * elaineIncomeThisYear
noncomputable def elaineExpensesThisYearGroceries := 0.25 * elaineIncomeThisYear
noncomputable def elaineExpensesThisYearHealthcare := (0.15 * elaineIncomeThisYear) * 1.10
noncomputable def elaineTotalExpensesThisYear := elaineExpensesThisYearRent + elaineExpensesThisYearGroceries + elaineExpensesThisYearHealthcare
noncomputable def elaineSavingsThisYear := elaineIncomeThisYear - elaineTotalExpensesThisYear

theorem elaineExpenseChanges :
  ( ((elaineExpensesThisYearRent - elaineExpensesLastYearRent) / elaineExpensesLastYearRent) * 100 = 296)
  ∧ ( ((elaineExpensesThisYearGroceries - elaineExpensesLastYearGroceries) / elaineExpensesLastYearGroceries) * 100 = 65)
  ∧ ( ((elaineExpensesThisYearHealthcare - elaineExpensesLastYearHealthcare) / elaineExpensesLastYearHealthcare) * 100 = 45.2)
  ∧ ( (elaineSavingsLastYear / elaineIncomeLastYear) * 100 = 55)
  ∧ ( (elaineSavingsThisYear / elaineIncomeThisYear) * 100 = 28.5)
  ∧ ( (elaineTotalExpensesLastYear / elaineIncomeLastYear) = 0.45 )
  ∧ ( (elaineTotalExpensesThisYear / elaineIncomeThisYear) = 0.715 )
  ∧ ( (elaineSavingsLastYear - elaineSavingsThisYear) = 4345 ∧ ( (55 - ((elaineSavingsThisYear / elaineIncomeThisYear) * 100)) = 26.5 ))
:= by sorry

end elaineExpenseChanges_l80_80262


namespace compound_interest_l80_80108

variables {a r : ℝ}

theorem compound_interest (a r : ℝ) :
  (a * (1 + r)^10) = a * (1 + r)^(2020 - 2010) :=
by
  sorry

end compound_interest_l80_80108


namespace frequency_interval_20_to_inf_l80_80831

theorem frequency_interval_20_to_inf (sample_size : ℕ)
  (freq_5_10 : ℕ) (freq_10_15 : ℕ) (freq_15_20 : ℕ)
  (freq_20_25 : ℕ) (freq_25_30 : ℕ) (freq_30_35 : ℕ) :
  sample_size = 35 ∧
  freq_5_10 = 5 ∧
  freq_10_15 = 12 ∧
  freq_15_20 = 7 ∧
  freq_20_25 = 5 ∧
  freq_25_30 = 4 ∧
  freq_30_35 = 2 →
  (1 - (freq_5_10 + freq_10_15 + freq_15_20 : ℕ) / (sample_size : ℕ) : ℝ) = 11 / 35 :=
by sorry

end frequency_interval_20_to_inf_l80_80831


namespace triangle_angle_R_measure_l80_80116

theorem triangle_angle_R_measure :
  ∀ (P Q R : ℝ),
  P + Q + R = 180 ∧ P = 70 ∧ Q = 2 * R + 15 → R = 95 / 3 :=
by
  intros P Q R h
  sorry

end triangle_angle_R_measure_l80_80116


namespace calculate_correctly_l80_80215

theorem calculate_correctly (x : ℕ) (h : 2 * x = 22) : 20 * x + 3 = 223 :=
by
  sorry

end calculate_correctly_l80_80215


namespace comb_15_6_eq_5005_perm_6_eq_720_l80_80526

open Nat

-- Prove that \frac{15!}{6!(15-6)!} = 5005
theorem comb_15_6_eq_5005 : (factorial 15) / (factorial 6 * factorial (15 - 6)) = 5005 := by
  sorry

-- Prove that the number of ways to arrange 6 items in a row is 720
theorem perm_6_eq_720 : factorial 6 = 720 := by
  sorry

end comb_15_6_eq_5005_perm_6_eq_720_l80_80526


namespace water_added_l80_80852

theorem water_added (capacity : ℝ) (percentage_initial : ℝ) (percentage_final : ℝ) :
  capacity = 120 →
  percentage_initial = 0.30 →
  percentage_final = 0.75 →
  ((percentage_final * capacity) - (percentage_initial * capacity)) = 54 :=
by intros
   sorry

end water_added_l80_80852


namespace intersection_M_complement_N_l80_80316

open Set Real

def M : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}
def N : Set ℝ := {x | (Real.log 2) ^ (1 - x) < 1}
def complement_N := {x : ℝ | x ≥ 1}

theorem intersection_M_complement_N :
  M ∩ complement_N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_complement_N_l80_80316


namespace divisibility_proof_l80_80491

theorem divisibility_proof (n : ℕ) (hn : 0 < n) (h : n ∣ (10^n - 1)) : 
  n ∣ ((10^n - 1) / 9) :=
  sorry

end divisibility_proof_l80_80491


namespace students_appeared_l80_80676

def passed (T : ℝ) : ℝ := 0.35 * T
def B_grade_range (T : ℝ) : ℝ := 0.25 * T
def failed (T : ℝ) : ℝ := T - passed T

theorem students_appeared (T : ℝ) (hp : passed T = 0.35 * T)
    (hb : B_grade_range T = 0.25 * T) (hf : failed T = 481) :
    T = 740 :=
by
  -- proof goes here
  sorry

end students_appeared_l80_80676


namespace calculate_expression_l80_80355

theorem calculate_expression : (8^5 / 8^2) * 3^6 = 373248 := by
  sorry

end calculate_expression_l80_80355


namespace cost_of_fencing_l80_80843

/-- The sides of a rectangular field are in the ratio 3:4.
If the area of the field is 10092 sq. m and the cost of fencing the field is 25 paise per meter,
then the cost of fencing the field is 101.5 rupees. --/
theorem cost_of_fencing (area : ℕ) (fencing_cost : ℝ) (ratio1 ratio2 perimeter : ℝ)
  (h_area : area = 10092)
  (h_ratio : ratio1 = 3 ∧ ratio2 = 4)
  (h_fencing_cost : fencing_cost = 0.25)
  (h_perimeter : perimeter = 406) :
  perimeter * fencing_cost = 101.5 := by
  sorry

end cost_of_fencing_l80_80843


namespace race_positions_l80_80208

theorem race_positions :
  ∀ (M J T R H D : ℕ),
    (M = J + 3) →
    (J = T + 1) →
    (T = R + 3) →
    (H = R + 5) →
    (D = H + 4) →
    (M = 9) →
    H = 7 :=
by sorry

end race_positions_l80_80208


namespace average_interest_rate_l80_80454

theorem average_interest_rate (x : ℝ) (h1 : 0 < x ∧ x < 6000)
  (h2 : 0.03 * (6000 - x) = 0.055 * x) :
  ((0.03 * (6000 - x) + 0.055 * x) / 6000) = 0.0388 :=
by
  sorry

end average_interest_rate_l80_80454


namespace roots_sum_l80_80101

theorem roots_sum (a b : ℝ) 
  (h₁ : 3^(a-1) = 6 - a)
  (h₂ : 3^(6-b) = b - 1) : 
  a + b = 7 := 
by sorry

end roots_sum_l80_80101


namespace average_annual_growth_rate_l80_80599

-- Definitions of the provided conditions
def initial_amount : ℝ := 200
def final_amount : ℝ := 338
def periods : ℝ := 2

-- Statement of the goal
theorem average_annual_growth_rate :
  (final_amount / initial_amount)^(1 / periods) - 1 = 0.3 := 
sorry

end average_annual_growth_rate_l80_80599


namespace wire_not_used_is_20_l80_80193

def initial_wire_length : ℕ := 50
def number_of_parts : ℕ := 5
def parts_used : ℕ := 3

def length_of_each_part (total_length : ℕ) (parts : ℕ) : ℕ := total_length / parts
def length_used (length_each_part : ℕ) (used_parts : ℕ) : ℕ := length_each_part * used_parts
def wire_not_used (total_length : ℕ) (used_length : ℕ) : ℕ := total_length - used_length

theorem wire_not_used_is_20 : 
  wire_not_used initial_wire_length 
    (length_used 
      (length_of_each_part initial_wire_length number_of_parts) 
    parts_used) = 20 := by
  sorry

end wire_not_used_is_20_l80_80193


namespace total_caps_produced_l80_80549

-- Define the production of each week as given in the conditions.
def week1_caps : ℕ := 320
def week2_caps : ℕ := 400
def week3_caps : ℕ := 300

-- Define the average of the first three weeks.
def average_caps : ℕ := (week1_caps + week2_caps + week3_caps) / 3

-- Define the production increase for the fourth week.
def increase_caps : ℕ := average_caps / 5  -- 20% is equivalent to dividing by 5

-- Calculate the total production for the fourth week (including the increase).
def week4_caps : ℕ := average_caps + increase_caps

-- Calculate the total number of caps produced in four weeks.
def total_caps : ℕ := week1_caps + week2_caps + week3_caps + week4_caps

-- Theorem stating the total production over the four weeks.
theorem total_caps_produced : total_caps = 1428 := by sorry

end total_caps_produced_l80_80549


namespace julia_played_with_34_kids_l80_80242

-- Define the number of kids Julia played with on each day
def kidsMonday : Nat := 17
def kidsTuesday : Nat := 15
def kidsWednesday : Nat := 2

-- Define the total number of kids Julia played with
def totalKids : Nat := kidsMonday + kidsTuesday + kidsWednesday

-- Prove given conditions
theorem julia_played_with_34_kids :
  totalKids = 34 :=
by
  sorry

end julia_played_with_34_kids_l80_80242


namespace half_sum_of_squares_of_even_or_odd_l80_80157

theorem half_sum_of_squares_of_even_or_odd (n1 n2 : ℤ) (a b : ℤ) :
  (n1 % 2 = 0 ∧ n2 % 2 = 0 ∧ n1 = 2*a ∧ n2 = 2*b ∨
   n1 % 2 = 1 ∧ n2 % 2 = 1 ∧ n1 = 2*a + 1 ∧ n2 = 2*b + 1) →
  ∃ x y : ℤ, (n1^2 + n2^2) / 2 = x^2 + y^2 :=
by
  intro h
  sorry

end half_sum_of_squares_of_even_or_odd_l80_80157


namespace second_student_marks_l80_80954

theorem second_student_marks (x y : ℝ) 
  (h1 : x = y + 9) 
  (h2 : x = 0.56 * (x + y)) : 
  y = 33 := 
sorry

end second_student_marks_l80_80954


namespace generalized_inequality_l80_80078

theorem generalized_inequality (n k : ℕ) (h1 : 3 ≤ n) (h2 : 1 ≤ k ∧ k ≤ n) : 
  2^n + 5^n > 2^(n - k) * 5^k + 2^k * 5^(n - k) := 
by 
  sorry

end generalized_inequality_l80_80078


namespace time_to_travel_A_to_C_is_6_l80_80060

-- Assume the existence of a real number t representing the time taken
-- Assume constant speed r for the river current and p for the power boat relative to the river.
variables (t r p : ℝ)

-- Conditions
axiom condition1 : p > 0
axiom condition2 : r > 0
axiom condition3 : t * (1.5 * (p + r)) + (p - r) * (12 - t) = 12 * r

-- Define the time taken for the power boat to travel from A to C
def time_from_A_to_C : ℝ := t

-- The proof problem: Prove time_from_A_to_C = 6 under the given conditions
theorem time_to_travel_A_to_C_is_6 : time_from_A_to_C = 6 := by
  sorry

end time_to_travel_A_to_C_is_6_l80_80060


namespace range_of_m_l80_80807

-- Definitions according to the problem conditions
def p (x : ℝ) : Prop := (-2 ≤ x ∧ x ≤ 10)
def q (x : ℝ) (m : ℝ) : Prop := (1 - m ≤ x ∧ x ≤ 1 + m) ∧ m > 0

-- Rephrasing the problem statement in Lean
theorem range_of_m (x : ℝ) (m : ℝ) :
  (∀ x, p x → q x m) → m ≥ 9 :=
sorry

end range_of_m_l80_80807


namespace af2_plus_bfg_plus_cg2_geq_0_l80_80291

theorem af2_plus_bfg_plus_cg2_geq_0 (a b c : ℝ) (f g : ℝ) :
  (a * f^2 + b * f * g + c * g^2 ≥ 0) ↔ (a ≥ 0 ∧ c ≥ 0 ∧ 4 * a * c ≥ b^2) := 
sorry

end af2_plus_bfg_plus_cg2_geq_0_l80_80291


namespace sphere_surface_area_l80_80707

theorem sphere_surface_area (a b c : ℝ) (r : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : r = (Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2)) / 2):
    4 * Real.pi * r ^ 2 = 50 * Real.pi :=
by
  sorry

end sphere_surface_area_l80_80707


namespace positive_integers_divisible_by_4_5_and_6_less_than_300_l80_80611

open Nat

theorem positive_integers_divisible_by_4_5_and_6_less_than_300 : 
    ∃ n : ℕ, n = 5 ∧ ∀ m, m < 300 → (m % 4 = 0 ∧ m % 5 = 0 ∧ m % 6 = 0) → (m % 60 = 0) :=
by
  sorry

end positive_integers_divisible_by_4_5_and_6_less_than_300_l80_80611


namespace xy_sum_value_l80_80195

theorem xy_sum_value (x y : ℝ) (h1 : x^2 + x * y + 2 * y = 10) (h2 : y^2 + x * y + 2 * x = 14) :
  x + y = -6 ∨ x + y = 4 :=
sorry

end xy_sum_value_l80_80195


namespace platform_length_l80_80241

theorem platform_length (train_speed_kmph : ℕ) (train_time_man_seconds : ℕ) (train_time_platform_seconds : ℕ) (train_speed_mps : ℕ) : 
  train_speed_kmph = 54 →
  train_time_man_seconds = 20 →
  train_time_platform_seconds = 30 →
  train_speed_mps = (54 * 1000 / 3600) →
  (54 * 5 / 18) = 15 →
  ∃ (P : ℕ), (train_speed_mps * train_time_platform_seconds) = (train_speed_mps * train_time_man_seconds) + P ∧ P = 150 :=
by
  sorry

end platform_length_l80_80241


namespace commodity_price_l80_80144

-- Define the variables for the prices of the commodities.
variable (x y : ℝ)

-- Conditions given in the problem.
def total_cost (x y : ℝ) : Prop := x + y = 827
def price_difference (x y : ℝ) : Prop := x = y + 127

-- The main statement to prove.
theorem commodity_price (x y : ℝ) (h1 : total_cost x y) (h2 : price_difference x y) : x = 477 :=
by
  sorry

end commodity_price_l80_80144


namespace mohamed_donated_more_l80_80951

-- Definitions of the conditions
def toysLeilaDonated : ℕ := 2 * 25
def toysMohamedDonated : ℕ := 3 * 19

-- The theorem stating Mohamed donated 7 more toys than Leila
theorem mohamed_donated_more : toysMohamedDonated - toysLeilaDonated = 7 :=
by
  sorry

end mohamed_donated_more_l80_80951


namespace acme_profit_calculation_l80_80248

theorem acme_profit_calculation :
  let initial_outlay := 12450
  let cost_per_set := 20.75
  let selling_price := 50
  let number_of_sets := 950
  let total_revenue := number_of_sets * selling_price
  let total_manufacturing_costs := initial_outlay + cost_per_set * number_of_sets
  let profit := total_revenue - total_manufacturing_costs 
  profit = 15337.50 := 
by
  sorry

end acme_profit_calculation_l80_80248


namespace original_number_l80_80162

theorem original_number (n : ℚ) (h : (3 * (n + 3) - 2) / 3 = 10) : n = 23 / 3 := 
sorry

end original_number_l80_80162


namespace unique_pair_fraction_l80_80592

theorem unique_pair_fraction (p : ℕ) (hprime : Nat.Prime p) (hodd : p % 2 = 1) :
  ∃! (n m : ℕ), (n ≠ m) ∧ (2 / (p : ℚ) = 1 / (n : ℚ) + 1 / (m : ℚ)) ∧
  ((n = (p + 1) / 2 ∧ m = p * (p + 1) / 2) ∨ (m = (p + 1) / 2 ∧ n = p * (p + 1) / 2)) := sorry

end unique_pair_fraction_l80_80592


namespace sequence_problem_l80_80655

/-- Given sequence a_n with specific values for a_2 and a_4 and the assumption that a_(n+1)
    is a geometric sequence, prove that a_6 equals 63. -/
theorem sequence_problem 
  {a : ℕ → ℝ} 
  (h1 : a 2 = 3) 
  (h2 : a 4 = 15) 
  (h3 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n ∧ q^2 = 4) : 
  a 6 = 63 := by
  sorry

end sequence_problem_l80_80655


namespace isosceles_triangle_l80_80594

-- Let ∆ABC be a triangle with angles A, B, and C
variables {A B C : ℝ}

-- Given condition: 2 * cos B * sin A = sin C
def condition (A B C : ℝ) : Prop := 2 * Real.cos B * Real.sin A = Real.sin C

-- Problem: Given the condition, we need to prove that ∆ABC is an isosceles triangle, meaning A = B.
theorem isosceles_triangle (A B C : ℝ) (h : condition A B C) : A = B :=
by
  sorry

end isosceles_triangle_l80_80594


namespace commute_time_difference_l80_80376

theorem commute_time_difference (x y : ℝ) 
  (h1 : x + y = 39)
  (h2 : (x - 10)^2 + (y - 10)^2 = 10) :
  |x - y| = 4 :=
by
  sorry

end commute_time_difference_l80_80376


namespace alice_winning_strategy_l80_80564

theorem alice_winning_strategy (n : ℕ) (h : n ≥ 2) :
  (∃ strategy : Π (s : ℕ), s < n → (ℕ × ℕ), 
    ∀ (k : ℕ) (hk : k < n), ¬(strategy k hk).fst = (strategy k hk).snd) ↔ (n % 4 = 3) :=
sorry

end alice_winning_strategy_l80_80564


namespace chocoBites_mod_l80_80243

theorem chocoBites_mod (m : ℕ) (hm : m % 8 = 5) : (4 * m) % 8 = 4 :=
by
  sorry

end chocoBites_mod_l80_80243


namespace probability_pink_second_marble_l80_80682

def bagA := (5, 5)  -- (red, green)
def bagB := (8, 2)  -- (pink, purple)
def bagC := (3, 7)  -- (pink, purple)

def P (success total : ℕ) := success / total

def probability_red := P 5 10
def probability_green := P 5 10

def probability_pink_given_red := P 8 10
def probability_pink_given_green := P 3 10

theorem probability_pink_second_marble :
  probability_red * probability_pink_given_red +
  probability_green * probability_pink_given_green = 11 / 20 :=
sorry

end probability_pink_second_marble_l80_80682


namespace proof_problem_l80_80119

-- Definitions for the solution sets
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def intersection : Set ℝ := {x | -1 < x ∧ x < 2}

-- The quadratic inequality solution sets
def solution_set (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- The main theorem statement
theorem proof_problem (a b : ℝ) (h : solution_set a b = intersection) : a + b = -3 :=
sorry

end proof_problem_l80_80119


namespace solution_set_quadratic_l80_80139

theorem solution_set_quadratic (a x : ℝ) (h : a < 0) : 
  (x^2 - 2 * a * x - 3 * a^2 < 0) ↔ (3 * a < x ∧ x < -a) := 
by
  sorry

end solution_set_quadratic_l80_80139


namespace tan_ratio_l80_80605

-- Given conditions
variables {p q : ℝ} (h1 : Real.cos (p + q) = 1 / 3) (h2 : Real.cos (p - q) = 2 / 3)

-- The theorem we need to prove
theorem tan_ratio (h1 : Real.cos (p + q) = 1 / 3) (h2 : Real.cos (p - q) = 2 / 3) : 
  Real.tan p / Real.tan q = -1 / 3 :=
sorry

end tan_ratio_l80_80605


namespace sheets_per_day_l80_80826

-- Definitions based on conditions
def total_sheets : ℕ := 60
def total_days_per_week : ℕ := 7
def days_off : ℕ := 2

-- Derived condition from the problem
def work_days_per_week : ℕ := total_days_per_week - days_off

-- The statement to prove
theorem sheets_per_day : total_sheets / work_days_per_week = 12 :=
by
  sorry

end sheets_per_day_l80_80826


namespace find_positive_real_solutions_l80_80079

variable {x_1 x_2 x_3 x_4 x_5 : ℝ}

theorem find_positive_real_solutions
  (h1 : (x_1^2 - x_3 * x_5) * (x_2^2 - x_3 * x_5) ≤ 0)
  (h2 : (x_2^2 - x_4 * x_1) * (x_3^2 - x_4 * x_1) ≤ 0)
  (h3 : (x_3^2 - x_5 * x_2) * (x_4^2 - x_5 * x_2) ≤ 0)
  (h4 : (x_4^2 - x_1 * x_3) * (x_5^2 - x_1 * x_3) ≤ 0)
  (h5 : (x_5^2 - x_2 * x_4) * (x_1^2 - x_2 * x_4) ≤ 0)
  (hx1 : 0 < x_1)
  (hx2 : 0 < x_2)
  (hx3 : 0 < x_3)
  (hx4 : 0 < x_4)
  (hx5 : 0 < x_5) :
  x_1 = x_2 ∧ x_2 = x_3 ∧ x_3 = x_4 ∧ x_4 = x_5 :=
by
  sorry

end find_positive_real_solutions_l80_80079


namespace div_30_div_510_div_66_div_large_l80_80299

theorem div_30 (a : ℤ) : 30 ∣ (a^5 - a) := 
  sorry  

theorem div_510 (a : ℤ) : 510 ∣ (a^17 - a) := 
  sorry

theorem div_66 (a : ℤ) : 66 ∣ (a^11 - a) := 
  sorry

theorem div_large (a : ℤ) : (2 * 3 * 5 * 7 * 13 * 19 * 37 * 73) ∣ (a^73 - a) := 
  sorry  

end div_30_div_510_div_66_div_large_l80_80299


namespace find_constants_for_B_l80_80007
open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2, 4], ![2, 0, 2], ![4, 2, 0]]

def I3 : Matrix (Fin 3) (Fin 3) ℝ := 1

def zeros : Matrix (Fin 3) (Fin 3) ℝ := 0

theorem find_constants_for_B : 
  ∃ (s t u : ℝ), s = 0 ∧ t = -36 ∧ u = -48 ∧ (B^3 + s • B^2 + t • B + u • I3 = zeros) :=
sorry

end find_constants_for_B_l80_80007


namespace largest_final_digit_l80_80545

theorem largest_final_digit (seq : Fin 1002 → Fin 10) 
  (h1 : seq 0 = 2) 
  (h2 : ∀ n : Fin 1001, (17 ∣ (10 * seq n + seq (n + 1))) ∨ (29 ∣ (10 * seq n + seq (n + 1)))) : 
  seq 1001 = 5 :=
sorry

end largest_final_digit_l80_80545


namespace hayden_earnings_l80_80407

theorem hayden_earnings 
  (wage_per_hour : ℕ) 
  (pay_per_ride : ℕ)
  (bonus_per_review : ℕ)
  (number_of_rides : ℕ)
  (hours_worked : ℕ)
  (gas_cost_per_gallon : ℕ)
  (gallons_of_gas : ℕ)
  (positive_reviews : ℕ)
  : wage_per_hour = 15 → 
    pay_per_ride = 5 → 
    bonus_per_review = 20 → 
    number_of_rides = 3 → 
    hours_worked = 8 → 
    gas_cost_per_gallon = 3 → 
    gallons_of_gas = 17 → 
    positive_reviews = 2 → 
    (hours_worked * wage_per_hour + number_of_rides * pay_per_ride + positive_reviews * bonus_per_review + gallons_of_gas * gas_cost_per_gallon) = 226 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  -- Further proof processing with these assumptions
  sorry

end hayden_earnings_l80_80407


namespace coplanar_vertices_sum_even_l80_80181

theorem coplanar_vertices_sum_even (a b c d e f g h : ℤ) :
  (∃ (a b c d : ℤ), true ∧ (a + b + c + d) % 2 = 0) :=
sorry

end coplanar_vertices_sum_even_l80_80181


namespace solve_cubic_eq_a_solve_cubic_eq_b_solve_cubic_eq_c_l80_80191

-- For the first polynomial equation
theorem solve_cubic_eq_a (x : ℝ) : x^3 - 3 * x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

-- For the second polynomial equation
theorem solve_cubic_eq_b (x : ℝ) : x^3 - 19 * x - 30 = 0 ↔ x = 5 ∨ x = -2 ∨ x = -3 :=
by sorry

-- For the third polynomial equation
theorem solve_cubic_eq_c (x : ℝ) : x^3 + 4 * x^2 + 6 * x + 4 = 0 ↔ x = -2 :=
by sorry

end solve_cubic_eq_a_solve_cubic_eq_b_solve_cubic_eq_c_l80_80191


namespace product_identity_l80_80939

theorem product_identity (x y : ℝ) : (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end product_identity_l80_80939


namespace find_E_l80_80537

variable (A H C S M N E : ℕ)
variable (x y z l : ℕ)

theorem find_E (h1 : A * x + H * y + C * z = l)
 (h2 : S * x + M * y + N * z = l)
 (h3 : E * x = l)
 (h4 : A ≠ S ∧ A ≠ H ∧ A ≠ C ∧ A ≠ M ∧ A ≠ N ∧ A ≠ E ∧ H ≠ C ∧ H ≠ M ∧ H ≠ N ∧ H ≠ E ∧ C ≠ M ∧ C ≠ N ∧ C ≠ E ∧ M ≠ N ∧ M ≠ E ∧ N ≠ E)
 : E = (A * M + C * N - S * H - N * H) / (M + N - H) := 
sorry

end find_E_l80_80537


namespace sum_of_first_four_terms_l80_80353

noncomputable def sum_first_n_terms (a q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem sum_of_first_four_terms :
  ∀ (a q : ℝ), a * (1 + q) = 7 → a * (q^6 - 1) / (q - 1) = 91 →
  a * (1 + q + q^2 + q^3) = 28 :=
by
  intros a q h₁ h₂
  -- Proof omitted
  sorry

end sum_of_first_four_terms_l80_80353


namespace eval_P_at_4_over_3_eval_P_at_2_l80_80559

noncomputable def P (a : ℚ) : ℚ := (6 * a^2 - 14 * a + 5) * (3 * a - 4)

theorem eval_P_at_4_over_3 : P (4 / 3) = 0 :=
by sorry

theorem eval_P_at_2 : P 2 = 2 :=
by sorry

end eval_P_at_4_over_3_eval_P_at_2_l80_80559


namespace remainder_of_3_pow_101_plus_4_mod_5_l80_80336

theorem remainder_of_3_pow_101_plus_4_mod_5 :
  (3^101 + 4) % 5 = 2 :=
by
  have h1 : 3 % 5 = 3 := by sorry
  have h2 : (3^2) % 5 = 4 := by sorry
  have h3 : (3^3) % 5 = 2 := by sorry
  have h4 : (3^4) % 5 = 1 := by sorry
  -- more steps to show the pattern and use it to prove the final statement
  sorry

end remainder_of_3_pow_101_plus_4_mod_5_l80_80336


namespace perpendicular_lines_l80_80393

def line1 (m : ℝ) (x y : ℝ) := m * x - 3 * y - 1 = 0
def line2 (m : ℝ) (x y : ℝ) := (3 * m - 2) * x - m * y + 2 = 0

theorem perpendicular_lines (m : ℝ) :
  (∀ x y : ℝ, line1 m x y) →
  (∀ x y : ℝ, line2 m x y) →
  (∀ x y : ℝ, (m / 3) * ((3 * m - 2) / m) = -1) →
  m = 0 ∨ m = -1/3 :=
by
  intros
  sorry

end perpendicular_lines_l80_80393


namespace inequality_proof_l80_80812

variable {a1 a2 a3 a4 a5 : ℝ}

theorem inequality_proof (h1 : 1 < a1) (h2 : 1 < a2) (h3 : 1 < a3) (h4 : 1 < a4) (h5 : 1 < a5) :
  16 * (a1 * a2 * a3 * a4 * a5 + 1) > (1 + a1) * (1 + a2) * (1 + a3) * (1 + a4) * (1 + a5) := 
sorry

end inequality_proof_l80_80812


namespace percentage_increase_in_ear_piercing_l80_80133

def cost_of_nose_piercing : ℕ := 20
def noses_pierced : ℕ := 6
def ears_pierced : ℕ := 9
def total_amount_made : ℕ := 390

def cost_of_ear_piercing : ℕ := (total_amount_made - (noses_pierced * cost_of_nose_piercing)) / ears_pierced

def percentage_increase (original new : ℕ) : ℚ := ((new - original : ℚ) / original) * 100

theorem percentage_increase_in_ear_piercing : 
  percentage_increase cost_of_nose_piercing cost_of_ear_piercing = 50 := 
by 
  sorry

end percentage_increase_in_ear_piercing_l80_80133


namespace minimal_fencing_l80_80956

theorem minimal_fencing (w l : ℝ) (h1 : l = 2 * w) (h2 : w * l ≥ 400) : 
  2 * (w + l) = 60 * Real.sqrt 2 :=
by
  sorry

end minimal_fencing_l80_80956


namespace total_numbers_l80_80643

theorem total_numbers (n : ℕ) (a : ℕ → ℝ) 
  (h1 : (a 0 + a 1 + a 2 + a 3) / 4 = 25)
  (h2 : (a (n - 3) + a (n - 2) + a (n - 1)) / 3 = 35)
  (h3 : a 3 = 25)
  (h4 : (Finset.sum (Finset.range n) a) / n = 30) :
  n = 6 :=
sorry

end total_numbers_l80_80643


namespace measure_of_angle_A_values_of_b_and_c_l80_80325

variable (a b c : ℝ) (A : ℝ)

-- Declare the conditions as hypotheses
def condition1 (a b c : ℝ) := a^2 - c^2 = b^2 - b * c
def condition2 (a : ℝ) := a = 2
def condition3 (b c : ℝ) := b + c = 4

-- Proof that A = 60 degrees when the conditions are satisfied
theorem measure_of_angle_A (h : condition1 a b c) : A = 60 := by
  sorry

-- Proof that b and c are 2 when given conditions are satisfied
theorem values_of_b_and_c (h1 : condition1 2 b c) (h2 : condition3 b c) : b = 2 ∧ c = 2 := by
  sorry

end measure_of_angle_A_values_of_b_and_c_l80_80325


namespace plane_intersects_unit_cubes_l80_80712

-- Definitions:
def isLargeCube (cube : ℕ × ℕ × ℕ) : Prop := cube = (4, 4, 4)
def isUnitCube (size : ℕ) : Prop := size = 1

-- The main theorem we want to prove:
theorem plane_intersects_unit_cubes :
  ∀ (cube : ℕ × ℕ × ℕ) (plane : (ℝ × ℝ × ℝ) → ℝ),
  isLargeCube cube →
  (∀ point : ℝ × ℝ × ℝ, plane point = 0 → 
       ∃ (x y z : ℕ), x < 4 ∧ y < 4 ∧ z < 4 ∧ 
                     (x, y, z) ∈ { coords : ℕ × ℕ × ℕ | true }) →
  (∃ intersects : ℕ, intersects = 16) :=
by
  intros cube plane Hcube Hplane
  sorry

end plane_intersects_unit_cubes_l80_80712


namespace length_of_platform_is_300_meters_l80_80681

-- Definitions used in the proof
def kmph_to_mps (v: ℕ) : ℕ := (v * 1000) / 3600

def speed := kmph_to_mps 72

def time_cross_man := 15

def length_train := speed * time_cross_man

def time_cross_platform := 30

def total_distance_cross_platform := speed * time_cross_platform

def length_platform := total_distance_cross_platform - length_train

theorem length_of_platform_is_300_meters :
  length_platform = 300 :=
by
  sorry

end length_of_platform_is_300_meters_l80_80681


namespace combination_divisible_by_30_l80_80471

theorem combination_divisible_by_30 (n : ℤ) : ∃ k : ℤ, n^5 - n = 30 * k :=
by
  sorry

end combination_divisible_by_30_l80_80471


namespace selling_price_per_machine_l80_80171

theorem selling_price_per_machine (parts_cost patent_cost : ℕ) (num_machines : ℕ) 
  (hc1 : parts_cost = 3600) (hc2 : patent_cost = 4500) (hc3 : num_machines = 45) :
  (parts_cost + patent_cost) / num_machines = 180 :=
by
  sorry

end selling_price_per_machine_l80_80171


namespace minimum_score_for_advanced_course_l80_80345

theorem minimum_score_for_advanced_course (q1 q2 q3 q4 : ℕ) (H1 : q1 = 88) (H2 : q2 = 84) (H3 : q3 = 82) :
  (q1 + q2 + q3 + q4) / 4 ≥ 85 → q4 = 86 := by
  sorry

end minimum_score_for_advanced_course_l80_80345


namespace problem_l80_80455

-- Define the main problem conditions
variables {a b c : ℝ}
axiom h1 : a^2 + b^2 + c^2 = 63
axiom h2 : 2 * a + 3 * b + 6 * c = 21 * Real.sqrt 7

-- Define the goal
theorem problem :
  (a / c) ^ (a / b) = (1 / 3) ^ (2 / 3) :=
sorry

end problem_l80_80455


namespace trajectory_of_point_l80_80922

theorem trajectory_of_point (x y : ℝ) (P A : ℝ × ℝ × ℝ) (hP : P = (x, y, 0)) (hA : A = (0, 0, 4)) (hPA : dist P A = 5) : 
  x^2 + y^2 = 9 :=
by sorry

end trajectory_of_point_l80_80922


namespace mnmn_not_cube_in_base_10_and_find_smallest_base_b_l80_80273

theorem mnmn_not_cube_in_base_10_and_find_smallest_base_b 
    (m n : ℕ) (h1 : m * 10^3 + n * 10^2 + m * 10 + n < 10000) :
    ¬ (∃ k : ℕ, (m * 10^3 + n * 10^2 + m * 10 + n) = k^3) 
    ∧ ∃ b : ℕ, b > 1 ∧ (∃ k : ℕ, (m * b^3 + n * b^2 + m * b + n = k^3)) :=
by sorry

end mnmn_not_cube_in_base_10_and_find_smallest_base_b_l80_80273


namespace trig_identity_l80_80763

theorem trig_identity (α : ℝ) (h : Real.tan α = 1/3) :
  Real.cos α ^ 2 + Real.cos (Real.pi / 2 + 2 * α) = 3 / 10 := 
sorry

end trig_identity_l80_80763


namespace max_area_225_l80_80342

noncomputable def max_area_rect_perim60 (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x ≥ 10) : ℝ :=
max (x * y) (30 - x)

theorem max_area_225 (x y : ℝ) (h1 : 2 * x + 2 * y = 60) (h2 : x ≥ 10) :
  max_area_rect_perim60 x y h1 h2 = 225 :=
sorry

end max_area_225_l80_80342


namespace pencils_per_child_l80_80726

theorem pencils_per_child (children : ℕ) (total_pencils : ℕ) (h1 : children = 2) (h2 : total_pencils = 12) :
  total_pencils / children = 6 :=
by 
  sorry

end pencils_per_child_l80_80726


namespace farmer_sowed_correct_amount_l80_80790

def initial_buckets : ℝ := 8.75
def final_buckets : ℝ := 6
def buckets_sowed : ℝ := initial_buckets - final_buckets

theorem farmer_sowed_correct_amount : buckets_sowed = 2.75 :=
by {
  sorry
}

end farmer_sowed_correct_amount_l80_80790


namespace parabola_intersection_l80_80562

def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 4 * x + 2
def parabola2 (x : ℝ) : ℝ := 9 * x ^ 2 + 6 * x + 2

theorem parabola_intersection :
  ∃ (x y : ℝ), (parabola1 x = y ∧ parabola2 x = y) ∧ 
                ((x = 0 ∧ y = 2) ∨ (x = -5 / 3 ∧ y = 17)) :=
by
  sorry

end parabola_intersection_l80_80562


namespace percentage_difference_wages_l80_80750

variables (W1 W2 : ℝ)
variables (h1 : W1 > 0) (h2 : W2 > 0)
variables (h3 : 0.40 * W2 = 1.60 * 0.20 * W1)

theorem percentage_difference_wages (W1 W2 : ℝ) (h1 : W1 > 0) (h2 : W2 > 0) (h3 : 0.40 * W2 = 1.60 * 0.20 * W1) :
  (W1 - W2) / W1 = 0.20 :=
by
  sorry

end percentage_difference_wages_l80_80750


namespace number_of_solutions_l80_80009

-- Given conditions
def positiveIntSolution (x y : ℤ) : Prop := x > 0 ∧ y > 0 ∧ 4 * x + 7 * y = 2001

-- Theorem statement
theorem number_of_solutions : ∃ (count : ℕ), 
  count = 71 ∧ ∃ f : Fin count → ℤ × ℤ,
    (∀ i, positiveIntSolution (f i).1 (f i).2) :=
by
  sorry

end number_of_solutions_l80_80009


namespace count_not_divisible_by_5_or_7_l80_80923

theorem count_not_divisible_by_5_or_7 :
  let N := 500
  let count_divisible_by_5 := Nat.floor (499 / 5)
  let count_divisible_by_7 := Nat.floor (499 / 7)
  let count_divisible_by_35 := Nat.floor (499 / 35)
  let count_divisible_by_5_or_7 := count_divisible_by_5 + count_divisible_by_7 - count_divisible_by_35
  let total_numbers := 499
  total_numbers - count_divisible_by_5_or_7 = 343 :=
by
  let N := 500
  let count_divisible_by_5 := Nat.floor (499 / 5)
  let count_divisible_by_7 := Nat.floor (499 / 7)
  let count_divisible_by_35 := Nat.floor (499 / 35)
  let count_divisible_by_5_or_7 := count_divisible_by_5 + count_divisible_by_7 - count_divisible_by_35
  let total_numbers := 499
  have h : total_numbers - count_divisible_by_5_or_7 = 343 := by sorry
  exact h

end count_not_divisible_by_5_or_7_l80_80923


namespace proposition_holds_for_odd_numbers_l80_80767

variable (P : ℕ → Prop)

theorem proposition_holds_for_odd_numbers 
  (h1 : P 1)
  (h_ind : ∀ k : ℕ, k ≥ 1 → P k → P (k + 2)) :
  ∀ n : ℕ, n % 2 = 1 → P n :=
by
  sorry

end proposition_holds_for_odd_numbers_l80_80767


namespace clock_angle_4_oclock_l80_80509

theorem clock_angle_4_oclock :
  let total_degrees := 360
  let hours := 12
  let degree_per_hour := total_degrees / hours
  let hour_position := 4
  let minute_hand_position := 0
  let hour_hand_angle := hour_position * degree_per_hour
  hour_hand_angle = 120 := sorry

end clock_angle_4_oclock_l80_80509


namespace total_time_to_4864_and_back_l80_80169

variable (speed_boat : ℝ) (speed_stream : ℝ) (distance : ℝ)
variable (Sboat : speed_boat = 14) (Sstream : speed_stream = 1.2) (Dist : distance = 4864)

theorem total_time_to_4864_and_back :
  let speed_downstream := speed_boat + speed_stream
  let speed_upstream := speed_boat - speed_stream
  let time_downstream := distance / speed_downstream
  let time_upstream := distance / speed_upstream
  let total_time := time_downstream + time_upstream
  total_time = 700 :=
by
  sorry

end total_time_to_4864_and_back_l80_80169


namespace angle_part_a_angle_part_b_l80_80754

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (a : ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1^2 + a.2^2)

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ :=
  Real.arccos ((dot_product a b) / (magnitude a * magnitude b))

theorem angle_part_a :
  angle_between_vectors (4, 0) (2, -2) = Real.arccos (Real.sqrt 2 / 2) :=
by
  sorry

theorem angle_part_b :
  angle_between_vectors (5, -3) (3, 5) = Real.pi / 2 :=
by
  sorry

end angle_part_a_angle_part_b_l80_80754


namespace sum_of_exterior_segment_angles_is_540_l80_80543

-- Define the setup of the problem
def quadrilateral_inscribed_in_circle (A B C D : Type) : Prop := sorry
def angle_externally_inscribed (segment : Type) : ℝ := sorry

-- Main theorem statement
theorem sum_of_exterior_segment_angles_is_540
  (A B C D : Type)
  (h_quad : quadrilateral_inscribed_in_circle A B C D)
  (alpha beta gamma delta : ℝ)
  (h_alpha : alpha = angle_externally_inscribed A)
  (h_beta : beta = angle_externally_inscribed B)
  (h_gamma : gamma = angle_externally_inscribed C)
  (h_delta : delta = angle_externally_inscribed D) :
  alpha + beta + gamma + delta = 540 :=
sorry

end sum_of_exterior_segment_angles_is_540_l80_80543


namespace find_solution_l80_80916

def satisfies_conditions (x y z : ℝ) :=
  (x + 1) * y * z = 12 ∧
  (y + 1) * z * x = 4 ∧
  (z + 1) * x * y = 4

theorem find_solution (x y z : ℝ) :
  satisfies_conditions x y z →
  (x = 1 / 3 ∧ y = 3 ∧ z = 3) ∨ (x = 2 ∧ y = -2 ∧ z = -2) :=
by
  sorry

end find_solution_l80_80916


namespace count_four_digit_numbers_l80_80182

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l80_80182


namespace eugene_payment_correct_l80_80591

noncomputable def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price - (original_price * discount_rate)

noncomputable def total_cost (quantity : ℕ) (price : ℝ) : ℝ :=
  quantity * price

noncomputable def eugene_total_cost : ℝ :=
  let tshirt_price := discounted_price 20 0.10
  let pants_price := discounted_price 80 0.10
  let shoes_price := discounted_price 150 0.15
  let hat_price := discounted_price 25 0.05
  let jacket_price := discounted_price 120 0.20
  let total_cost_before_tax := 
    total_cost 4 tshirt_price + 
    total_cost 3 pants_price + 
    total_cost 2 shoes_price + 
    total_cost 3 hat_price + 
    total_cost 1 jacket_price
  total_cost_before_tax + (total_cost_before_tax * 0.06)

theorem eugene_payment_correct : eugene_total_cost = 752.87 := by
  sorry

end eugene_payment_correct_l80_80591


namespace tiling_possible_l80_80083

theorem tiling_possible (n x : ℕ) (hx : 7 * x = n^2) : ∃ k : ℕ, n = 7 * k :=
by sorry

end tiling_possible_l80_80083


namespace find_first_term_of_geometric_progression_l80_80991

theorem find_first_term_of_geometric_progression
  (a_2 : ℝ) (a_3 : ℝ) (a_1 : ℝ) (q : ℝ)
  (h1 : a_2 = a_1 * q)
  (h2 : a_3 = a_1 * q^2)
  (h3 : a_2 = 5)
  (h4 : a_3 = 1) : a_1 = 25 :=
by
  sorry

end find_first_term_of_geometric_progression_l80_80991


namespace oil_spending_l80_80884

-- Define the original price per kg of oil
def original_price (P : ℝ) := P

-- Define the reduced price per kg of oil
def reduced_price (P : ℝ) := 0.75 * P

-- Define the reduced price as Rs. 60
def reduced_price_fixed := 60

-- State the condition that reduced price enables 5 kgs more oil
def extra_kg := 5

-- The amount of money spent by housewife at reduced price which is to be proven as Rs. 1200
def amount_spent (M : ℝ) := M

-- Define the problem to prove in Lean 4
theorem oil_spending (P X : ℝ) (h1 : reduced_price P = reduced_price_fixed) (h2 : X * original_price P = (X + extra_kg) * reduced_price_fixed) : amount_spent ((X + extra_kg) * reduced_price_fixed) = 1200 :=
  sorry

end oil_spending_l80_80884


namespace line_equation_l80_80568

theorem line_equation (b r S : ℝ) (h : ℝ) (m : ℝ) (eq_one : S = 1/2 * b * h) (eq_two : h = 2*S / b) (eq_three : |m| = r / b) 
  (eq_four : m = r / b) : 
  (∀ x y : ℝ, y = m * (x - b) → b > 0 → r > 0 → S > 0 → rx - bry - rb = 0) := 
sorry

end line_equation_l80_80568


namespace victoria_should_return_22_l80_80957

theorem victoria_should_return_22 :
  let initial_money := 50
  let pizza_cost_per_box := 12
  let pizzas_bought := 2
  let juice_cost_per_pack := 2
  let juices_bought := 2
  let total_spent := (pizza_cost_per_box * pizzas_bought) + (juice_cost_per_pack * juices_bought)
  let money_returned := initial_money - total_spent
  money_returned = 22 :=
by
  sorry

end victoria_should_return_22_l80_80957


namespace factorial_equation_solution_unique_l80_80649

theorem factorial_equation_solution_unique :
  ∀ a b c : ℕ, (0 < a ∧ 0 < b ∧ 0 < c) →
  (a.factorial * b.factorial = a.factorial + b.factorial + c.factorial) →
  (a = 3 ∧ b = 3 ∧ c = 4) := 
by
  intros a b c h_positive h_eq
  sorry

end factorial_equation_solution_unique_l80_80649


namespace horners_rule_correct_l80_80607

open Classical

variables (x : ℤ) (poly_val : ℤ)

def original_polynomial (x : ℤ) : ℤ := 7 * x^3 + 3 * x^2 - 5 * x + 11

def horner_evaluation (x : ℤ) : ℤ := ((7 * x + 3) * x - 5) * x + 11

theorem horners_rule_correct : (poly_val = horner_evaluation 23) ↔ (poly_val = original_polynomial 23) :=
by {
  sorry
}

end horners_rule_correct_l80_80607


namespace subset_ratio_l80_80719

theorem subset_ratio (S T : ℕ) (hS : S = 256) (hT : T = 56) :
  (T / S : ℚ) = 7 / 32 := by
sorry

end subset_ratio_l80_80719


namespace max_value_of_symmetric_function_l80_80308

noncomputable def f (x a b : ℝ) : ℝ := (1 - x^2) * (x^2 + a * x + b)

theorem max_value_of_symmetric_function (a b : ℝ) :
  (∀ x : ℝ, f (-2 - x) a b = f (-2 + x) a b) → ∃ x : ℝ, ∀ y : ℝ, f x a b ≥ f y a b ∧ f x a b = 16 :=
sorry

end max_value_of_symmetric_function_l80_80308


namespace solution_set_l80_80626

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set :
  {x | f x ≥ x^2 - 8 * x + 15} = {2} ∪ {x | x > 6} :=
by
  sorry

end solution_set_l80_80626


namespace encryption_of_hope_is_correct_l80_80905

def shift_letter (c : Char) : Char :=
  if 'a' ≤ c ∧ c ≤ 'z' then
    Char.ofNat ((c.toNat - 'a'.toNat + 4) % 26 + 'a'.toNat)
  else 
    c

def encrypt (s : String) : String :=
  s.map shift_letter

theorem encryption_of_hope_is_correct : encrypt "hope" = "lsti" :=
by
  sorry

end encryption_of_hope_is_correct_l80_80905


namespace rectangle_enclosing_ways_l80_80746

/-- Given five horizontal lines and five vertical lines, the total number of ways to choose four lines (two horizontal, two vertical) such that they form a rectangle is 100 --/
theorem rectangle_enclosing_ways : 
  let horizontal_lines := [1, 2, 3, 4, 5]
  let vertical_lines := [1, 2, 3, 4, 5]
  let ways_horizontal := Nat.choose 5 2
  let ways_vertical := Nat.choose 5 2
  ways_horizontal * ways_vertical = 100 := 
by
  sorry

end rectangle_enclosing_ways_l80_80746


namespace least_positive_integer_divisibility_l80_80813

theorem least_positive_integer_divisibility :
  ∃ n > 1, (∀ k ∈ [2, 3, 4, 5, 6, 7, 8, 9], n % k = 1) ∧ n = 2521 :=
by
  sorry

end least_positive_integer_divisibility_l80_80813


namespace division_remainder_correct_l80_80642

theorem division_remainder_correct :
  ∃ q r, 987670 = 128 * q + r ∧ 0 ≤ r ∧ r < 128 ∧ r = 22 :=
by
  sorry

end division_remainder_correct_l80_80642


namespace axis_of_symmetry_parabola_l80_80967

theorem axis_of_symmetry_parabola (x y : ℝ) :
  y = - (1 / 8) * x^2 → y = 2 :=
sorry

end axis_of_symmetry_parabola_l80_80967


namespace greatest_sum_of_consecutive_integers_l80_80129

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l80_80129


namespace number_of_sandwiches_l80_80811

-- Definitions based on the conditions in the problem
def sandwich_cost : Nat := 3
def water_cost : Nat := 2
def total_cost : Nat := 11

-- Lean statement to prove the number of sandwiches bought is 3
theorem number_of_sandwiches (S : Nat) (h : sandwich_cost * S + water_cost = total_cost) : S = 3 :=
by
  sorry

end number_of_sandwiches_l80_80811


namespace distinct_natural_numbers_circles_sum_equal_impossible_l80_80505

theorem distinct_natural_numbers_circles_sum_equal_impossible :
  ¬∃ (f : ℕ → ℕ) (distinct : ∀ i j, i ≠ j → f i ≠ f j) (equal_sum : ∀ i j k, (f i + f j + f k = f (i+1) + f (j+1) + f (k+1))),
  true :=
  sorry

end distinct_natural_numbers_circles_sum_equal_impossible_l80_80505


namespace mia_high_school_has_2000_students_l80_80385

variables (M Z : ℕ)

def mia_high_school_students : Prop :=
  M = 4 * Z ∧ M + Z = 2500

theorem mia_high_school_has_2000_students (h : mia_high_school_students M Z) : 
  M = 2000 := by
  sorry

end mia_high_school_has_2000_students_l80_80385


namespace mangoes_ratio_l80_80781

theorem mangoes_ratio (a d_a : ℕ)
  (h1 : a = 60)
  (h2 : a + d_a = 75) : a / (75 - a) = 4 := by
  sorry

end mangoes_ratio_l80_80781


namespace angle_P_measure_l80_80805

theorem angle_P_measure (P Q R S : ℝ) 
  (h1 : P = 3 * Q)
  (h2 : P = 4 * R)
  (h3 : P = 6 * S)
  (h_sum : P + Q + R + S = 360) : 
  P = 206 :=
by 
  sorry

end angle_P_measure_l80_80805


namespace simplify_sum_of_square_roots_l80_80081

theorem simplify_sum_of_square_roots : (Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2) :=
by
  sorry

end simplify_sum_of_square_roots_l80_80081


namespace some_seniors_not_club_members_l80_80815

variables {People : Type} (Senior ClubMember : People → Prop) (Punctual : People → Prop)

-- Conditions:
def some_seniors_not_punctual := ∃ x, Senior x ∧ ¬Punctual x
def all_club_members_punctual := ∀ x, ClubMember x → Punctual x

-- Theorem statement to be proven:
theorem some_seniors_not_club_members (h1 : some_seniors_not_punctual Senior Punctual) (h2 : all_club_members_punctual ClubMember Punctual) : 
  ∃ x, Senior x ∧ ¬ ClubMember x :=
sorry

end some_seniors_not_club_members_l80_80815


namespace proof_ab_lt_1_l80_80644

noncomputable def f (x : ℝ) : ℝ := |Real.log x|

theorem proof_ab_lt_1 (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a > f b) : a * b < 1 :=
by
  -- Sorry to skip the proof
  sorry

end proof_ab_lt_1_l80_80644


namespace parabola_focus_distance_x_l80_80773

theorem parabola_focus_distance_x (x y : ℝ) :
  y^2 = 4 * x ∧ y^2 = 4 * (x^2 + 5^2) → x = 4 :=
by
  sorry

end parabola_focus_distance_x_l80_80773


namespace trig_expression_tangent_l80_80161

theorem trig_expression_tangent (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (π - α) - Real.sin (π / 2 + α)) / 
  (Real.cos (3 * π / 2 + α) + Real.cos (π - α)) = 1 :=
sorry

end trig_expression_tangent_l80_80161


namespace pencils_added_l80_80780

theorem pencils_added (initial_pencils total_pencils Mike_pencils : ℕ) 
    (h1 : initial_pencils = 41) 
    (h2 : total_pencils = 71) 
    (h3 : total_pencils = initial_pencils + Mike_pencils) :
    Mike_pencils = 30 := by
  sorry

end pencils_added_l80_80780


namespace ending_number_is_54_l80_80641

def first_even_after_15 : ℕ := 16
def evens_between (a b : ℕ) : ℕ := (b - first_even_after_15) / 2 + 1

theorem ending_number_is_54 (n : ℕ) (h : evens_between 15 n = 20) : n = 54 :=
by {
  sorry
}

end ending_number_is_54_l80_80641


namespace total_balloons_is_72_l80_80728

-- Definitions for the conditions from the problem
def fred_balloons : Nat := 10
def sam_balloons : Nat := 46
def dan_balloons : Nat := 16

-- The total number of red balloons is the sum of Fred's, Sam's, and Dan's balloons
def total_balloons (f s d : Nat) : Nat := f + s + d

-- The theorem stating the problem to be proved
theorem total_balloons_is_72 : total_balloons fred_balloons sam_balloons dan_balloons = 72 := by
  sorry

end total_balloons_is_72_l80_80728


namespace prob_no_1_or_6_l80_80425

theorem prob_no_1_or_6 :
  ∀ (a b c : ℕ), (1 ≤ a ∧ a ≤ 6) ∧ (1 ≤ b ∧ b ≤ 6) ∧ (1 ≤ c ∧ c ≤ 6) →
  (8 / 27 : ℝ) = (4 / 6) * (4 / 6) * (4 / 6) :=
by
  intros a b c h
  sorry

end prob_no_1_or_6_l80_80425


namespace total_profit_correct_l80_80779

noncomputable def total_profit (Cp Cq Cr Tp : ℝ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) (hR : 900 = (6 / (15 + 10 + 6)) * Tp) : ℝ := Tp

theorem total_profit_correct (Cp Cq Cr Tp : ℝ) (h1 : 4 * Cp = 6 * Cq) (h2 : 6 * Cq = 10 * Cr) (hR : 900 = (6 / (15 + 10 + 6)) * Tp) : 
  total_profit Cp Cq Cr Tp h1 h2 hR = 4650 :=
sorry

end total_profit_correct_l80_80779


namespace john_took_11_more_chickens_than_ray_l80_80145

noncomputable def chickens_taken_by_john (mary_chickens : ℕ) : ℕ := mary_chickens + 5
noncomputable def chickens_taken_by_ray (mary_chickens : ℕ) : ℕ := mary_chickens - 6
def ray_chickens : ℕ := 10

-- The theorem to prove:
theorem john_took_11_more_chickens_than_ray :
  ∃ (mary_chickens : ℕ), chickens_taken_by_john mary_chickens - ray_chickens = 11 :=
by
  -- Initial assumptions and derivation steps should be provided here.
  sorry

end john_took_11_more_chickens_than_ray_l80_80145


namespace books_remaining_in_special_collection_l80_80498

theorem books_remaining_in_special_collection
  (initial_books : ℕ)
  (loaned_books : ℕ)
  (returned_percentage : ℕ)
  (initial_books_eq : initial_books = 75)
  (loaned_books_eq : loaned_books = 45)
  (returned_percentage_eq : returned_percentage = 80) :
  ∃ final_books : ℕ, final_books = initial_books - (loaned_books - (loaned_books * returned_percentage / 100)) ∧ final_books = 66 :=
by
  sorry

end books_remaining_in_special_collection_l80_80498


namespace train_passes_platform_in_43_2_seconds_l80_80574

open Real

noncomputable def length_of_train : ℝ := 360
noncomputable def length_of_platform : ℝ := 180
noncomputable def speed_of_train_kmph : ℝ := 45
noncomputable def speed_of_train_mps : ℝ := (45 * 1000) / 3600  -- Converting km/hr to m/s

noncomputable def total_distance : ℝ := length_of_train + length_of_platform
noncomputable def time_to_pass_platform : ℝ := total_distance / speed_of_train_mps

theorem train_passes_platform_in_43_2_seconds :
  time_to_pass_platform = 43.2 := by
  sorry

end train_passes_platform_in_43_2_seconds_l80_80574


namespace Megan_pictures_left_l80_80861

theorem Megan_pictures_left (zoo_pictures museum_pictures deleted_pictures : ℕ) 
  (h1 : zoo_pictures = 15) 
  (h2 : museum_pictures = 18) 
  (h3 : deleted_pictures = 31) : 
  zoo_pictures + museum_pictures - deleted_pictures = 2 := 
by
  sorry

end Megan_pictures_left_l80_80861


namespace find_y_l80_80326

theorem find_y (y : ℕ) (h1 : y % 6 = 5) (h2 : y % 7 = 6) (h3 : y % 8 = 7) : y = 167 := 
by
  sorry  -- Proof is omitted

end find_y_l80_80326


namespace sin_B_plus_pi_over_6_eq_l80_80420

noncomputable def sin_b_plus_pi_over_6 (B : ℝ) : ℝ :=
  Real.sin B * (Real.sqrt 3 / 2) + (Real.sqrt (1 - (Real.sin B) ^ 2)) * (1 / 2)

theorem sin_B_plus_pi_over_6_eq :
  ∀ (A B : ℝ) (b c : ℝ),
    A = (2 * Real.pi / 3) →
    b = 1 →
    (1 / 2 * b * c * Real.sin A) = Real.sqrt 3 →
    c = 2 →
    sin_b_plus_pi_over_6 B = (2 * Real.sqrt 7 / 7) :=
by
  intros A B b c hA hb hArea hc
  sorry

end sin_B_plus_pi_over_6_eq_l80_80420


namespace arithmetic_sequence_sum_l80_80602

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h1 : S 4 = 3) (h2 : S 8 = 7) : S 12 = 12 :=
by
  -- placeholder for the proof, details omitted
  sorry

end arithmetic_sequence_sum_l80_80602


namespace floor_ceiling_sum_l80_80531

theorem floor_ceiling_sum : 
    Int.floor (0.998 : ℝ) + Int.ceil (2.002 : ℝ) = 3 := by
  sorry

end floor_ceiling_sum_l80_80531


namespace cos_value_given_sin_l80_80463

theorem cos_value_given_sin (α : ℝ) (h : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5) :
  Real.cos (π / 3 - α) = 2 * Real.sqrt 5 / 5 := by
  sorry

end cos_value_given_sin_l80_80463


namespace fare_per_1_5_mile_l80_80959

-- Definitions and conditions
def fare_first : ℝ := 1.0
def total_fare : ℝ := 7.3
def increments_per_mile : ℝ := 5.0
def total_miles : ℝ := 3.0
def remaining_increments : ℝ := (total_miles * increments_per_mile) - 1
def remaining_fare : ℝ := total_fare - fare_first

-- Theorem to prove
theorem fare_per_1_5_mile : remaining_fare / remaining_increments = 0.45 :=
by
  sorry

end fare_per_1_5_mile_l80_80959


namespace logarithm_argument_positive_l80_80551

open Real

theorem logarithm_argument_positive (a : ℝ) : 
  (∀ x : ℝ, sin x ^ 6 + cos x ^ 6 + a * sin x * cos x > 0) ↔ -1 / 2 < a ∧ a < 1 / 2 :=
by
  -- placeholder for the proof
  sorry

end logarithm_argument_positive_l80_80551


namespace solve_CD_l80_80065

noncomputable def find_CD : Prop :=
  ∃ C D : ℝ, (C = 11 ∧ D = 0) ∧ (∀ x : ℝ, x ≠ -4 ∧ x ≠ 12 → 
    (7 * x - 3) / ((x + 4) * (x - 12)) = C / (x + 4) + D / (x - 12))

theorem solve_CD : find_CD :=
sorry

end solve_CD_l80_80065


namespace genetic_recombination_does_not_occur_during_dna_replication_l80_80038

-- Definitions based on conditions
def dna_replication_spermatogonial_cells : Prop := 
  ∃ dna_interphase: Prop, ∃ dna_unwinding: Prop, 
    ∃ gene_mutation: Prop, ∃ protein_synthesis: Prop,
      dna_interphase ∧ dna_unwinding ∧ gene_mutation ∧ protein_synthesis

def genetic_recombination_not_occur : Prop :=
  ¬ ∃ genetic_recombination: Prop, genetic_recombination

-- Proof problem statement
theorem genetic_recombination_does_not_occur_during_dna_replication : 
  dna_replication_spermatogonial_cells → genetic_recombination_not_occur :=
by sorry

end genetic_recombination_does_not_occur_during_dna_replication_l80_80038


namespace maximize_profit_l80_80885

/-- A car sales company purchased a total of 130 vehicles of models A and B, 
with x vehicles of model A purchased. The profit y is defined by selling 
prices and factory prices of both models. -/
def total_profit (x : ℕ) : ℝ := -2 * x + 520

theorem maximize_profit :
  ∃ x : ℕ, (130 - x ≤ 2 * x) ∧ (total_profit x = 432) ∧ (∀ y : ℕ, (130 - y ≤ 2 * y) → (total_profit y ≤ 432)) :=
by {
  sorry
}

end maximize_profit_l80_80885


namespace binary_111_to_decimal_l80_80086

-- Define a function to convert binary list to decimal
def binaryToDecimal (bin : List ℕ) : ℕ :=
  bin.reverse.enumFrom 0 |>.foldl (λ acc ⟨i, b⟩ => acc + b * (2 ^ i)) 0

-- Assert the equivalence between the binary number [1, 1, 1] and its decimal representation 7
theorem binary_111_to_decimal : binaryToDecimal [1, 1, 1] = 7 :=
  by
  sorry

end binary_111_to_decimal_l80_80086


namespace tan_double_angle_tan_angle_add_pi_div_4_l80_80999

theorem tan_double_angle (α : ℝ) (h : Real.tan α = -2) : Real.tan (2 * α) = 4 / 3 :=
by
  sorry

theorem tan_angle_add_pi_div_4 (α : ℝ) (h : Real.tan α = -2) : Real.tan (2 * α + Real.pi / 4) = -7 :=
by
  sorry

end tan_double_angle_tan_angle_add_pi_div_4_l80_80999


namespace fourth_square_area_l80_80160

theorem fourth_square_area (AB BC CD AC x : ℝ) 
  (h_AB : AB^2 = 49) 
  (h_BC : BC^2 = 25) 
  (h_CD : CD^2 = 64) 
  (h_AC1 : AC^2 = AB^2 + BC^2) 
  (h_AC2 : AC^2 = CD^2 + x^2) :
  x^2 = 10 :=
by
  sorry

end fourth_square_area_l80_80160


namespace solve_floor_equation_l80_80617

theorem solve_floor_equation (x : ℝ) :
  (⌊⌊2 * x⌋ - 1 / 2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
sorry

end solve_floor_equation_l80_80617


namespace no_real_roots_l80_80392

-- Define the polynomial Q(x)
def Q (x : ℝ) : ℝ := x^6 - 3 * x^5 + 6 * x^4 - 6 * x^3 - x + 8

-- The problem can be stated as proving that Q(x) has no real roots
theorem no_real_roots : ∀ x : ℝ, Q x ≠ 0 := by
  sorry

end no_real_roots_l80_80392


namespace petrol_expenses_l80_80786

-- Definitions based on the conditions stated in the problem
def salary_saved (salary : ℝ) : ℝ := 0.10 * salary
def total_known_expenses : ℝ := 5000 + 1500 + 4500 + 2500 + 3940

-- Main theorem statement that needs to be proved
theorem petrol_expenses (salary : ℝ) (petrol : ℝ) :
  salary_saved salary = 2160 ∧ salary - 2160 = 19440 ∧ 
  5000 + 1500 + 4500 + 2500 + 3940 = total_known_expenses  →
  petrol = 2000 :=
sorry

end petrol_expenses_l80_80786


namespace teairra_shirts_l80_80185

theorem teairra_shirts (S : ℕ) (pants_total : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) (neither_plaid_nor_purple : ℕ)
  (pants_total_eq : pants_total = 24)
  (plaid_shirts_eq : plaid_shirts = 3)
  (purple_pants_eq : purple_pants = 5)
  (neither_plaid_nor_purple_eq : neither_plaid_nor_purple = 21) :
  (S - plaid_shirts + (pants_total - purple_pants) = neither_plaid_nor_purple) → S = 5 :=
by
  sorry

end teairra_shirts_l80_80185


namespace range_of_solutions_l80_80114

-- Define the function f(x) = x^2 - bx - 5
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - b * x - 5

theorem range_of_solutions (b : ℝ) :
  (f b (-2) = 5) ∧ 
  (f b (-1) = -1) ∧ 
  (f b 4 = -1) ∧ 
  (f b 5 = 5) →
  ∃ x1 x2, (-2 < x1 ∧ x1 < -1) ∨ (4 < x2 ∧ x2 < 5) ∧ f b x1 = 0 ∧ f b x2 = 0 :=
by
  sorry

end range_of_solutions_l80_80114


namespace KimFridayToMondayRatio_l80_80926

variable (MondaySweaters : ℕ) (TuesdaySweaters : ℕ) (WednesdaySweaters : ℕ) (ThursdaySweaters : ℕ) (FridaySweaters : ℕ)

def KimSweaterKnittingConditions (MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters : ℕ) : Prop :=
  MondaySweaters = 8 ∧
  TuesdaySweaters = MondaySweaters + 2 ∧
  WednesdaySweaters = TuesdaySweaters - 4 ∧
  ThursdaySweaters = TuesdaySweaters - 4 ∧
  MondaySweaters + TuesdaySweaters + WednesdaySweaters + ThursdaySweaters + FridaySweaters = 34

theorem KimFridayToMondayRatio 
  (MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters : ℕ)
  (h : KimSweaterKnittingConditions MondaySweaters TuesdaySweaters WednesdaySweaters ThursdaySweaters FridaySweaters) :
  FridaySweaters / MondaySweaters = 1/2 :=
  sorry

end KimFridayToMondayRatio_l80_80926


namespace Caroline_lost_4_pairs_of_socks_l80_80508

theorem Caroline_lost_4_pairs_of_socks 
  (initial_pairs : ℕ) (pairs_donated_fraction : ℚ)
  (new_pairs_purchased : ℕ) (new_pairs_gifted : ℕ)
  (final_pairs : ℕ) (L : ℕ) :
  initial_pairs = 40 →
  pairs_donated_fraction = 2/3 →
  new_pairs_purchased = 10 →
  new_pairs_gifted = 3 →
  final_pairs = 25 →
  (initial_pairs - L) * (1 - pairs_donated_fraction) + new_pairs_purchased + new_pairs_gifted = final_pairs →
  L = 4 :=
by {
  sorry
}

end Caroline_lost_4_pairs_of_socks_l80_80508


namespace problem1_problem2_l80_80017

open Set Real

-- Given A and B
def A (a : ℝ) : Set ℝ := {x | x > a}
def B : Set ℝ := {y | y > -1}

-- Problem 1: If A = B, then a = -1
theorem problem1 (a : ℝ) (h : A a = B) : a = -1 := by
  sorry

-- Problem 2: If (complement of A) ∩ B ≠ ∅, find the range of a
theorem problem2 (a : ℝ) (h : (compl (A a)) ∩ B ≠ ∅) : a ∈ Ioi (-1) := by
  sorry

end problem1_problem2_l80_80017


namespace piggy_bank_dimes_diff_l80_80334

theorem piggy_bank_dimes_diff :
  ∃ (a b c : ℕ), a + b + c = 100 ∧ 5 * a + 10 * b + 25 * c = 1005 ∧ (∀ lo hi, 
  (lo = 1 ∧ hi = 101) → (hi - lo = 100)) :=
by
  sorry

end piggy_bank_dimes_diff_l80_80334


namespace total_sodas_bought_l80_80966

-- Condition 1: Number of sodas they drank
def sodas_drank : ℕ := 3

-- Condition 2: Number of extra sodas Robin had
def sodas_extras : ℕ := 8

-- Mathematical equivalence we want to prove: Total number of sodas bought by Robin
theorem total_sodas_bought : sodas_drank + sodas_extras = 11 := by
  sorry

end total_sodas_bought_l80_80966


namespace perimeter_of_AF1B_l80_80745

noncomputable def ellipse_perimeter (a b x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (2 * a)

theorem perimeter_of_AF1B (h : (6:ℝ) = 6) :
  ellipse_perimeter 6 4 0 0 6 0 = 24 :=
by
  sorry

end perimeter_of_AF1B_l80_80745


namespace no_solution_for_triples_l80_80693

theorem no_solution_for_triples :
  ¬ ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ (a * b + b * c = 66) ∧ (a * c + b * c = 35) :=
by {
  sorry
}

end no_solution_for_triples_l80_80693


namespace coprime_pairs_solution_l80_80553

theorem coprime_pairs_solution (x y : ℕ) (hx : x ∣ y^2 + 210) (hy : y ∣ x^2 + 210) (hxy : Nat.gcd x y = 1) :
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) :=
by sorry

end coprime_pairs_solution_l80_80553


namespace intersection_of_A_and_B_l80_80124

def A (x : ℝ) : Prop := x^2 - x - 6 ≤ 0
def B (x : ℝ) : Prop := x > 1

theorem intersection_of_A_and_B :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x ≤ 3} :=
by
  sorry

end intersection_of_A_and_B_l80_80124


namespace range_of_a_l80_80286

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + a * x + 1 < 0) → -2 ≤ a ∧ a ≤ 2 := sorry

end range_of_a_l80_80286


namespace annes_score_l80_80702

theorem annes_score (a b : ℕ) (h1 : a = b + 50) (h2 : (a + b) / 2 = 150) : a = 175 := 
by
  sorry

end annes_score_l80_80702


namespace total_candy_pieces_l80_80431

theorem total_candy_pieces : 
  (brother_candy = 6) → 
  (wendy_boxes = 2) → 
  (pieces_per_box = 3) → 
  (brother_candy + (wendy_boxes * pieces_per_box) = 12) 
  := 
  by 
    intros brother_candy wendy_boxes pieces_per_box 
    sorry

end total_candy_pieces_l80_80431


namespace inequality_problem_l80_80647

variable {a b : ℕ}

theorem inequality_problem (a : ℕ) (b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_neq_1_a : a ≠ 1) (h_neq_1_b : b ≠ 1) :
  ((a^5 - 1:ℚ) / (a^4 - 1)) * ((b^5 - 1) / (b^4 - 1)) > (25 / 64 : ℚ) * (a + 1) * (b + 1) :=
by
  sorry

end inequality_problem_l80_80647


namespace fraction_changes_l80_80224

theorem fraction_changes (x y : ℝ) (h : 0 < x ∧ 0 < y) :
  (x + y) / (x * y) = 2 * ((2 * x + 2 * y) / (2 * x * 2 * y)) :=
by
  sorry

end fraction_changes_l80_80224


namespace regular_18gon_lines_rotational_symmetry_sum_l80_80756

def L : ℕ := 18
def R : ℕ := 20

theorem regular_18gon_lines_rotational_symmetry_sum : L + R = 38 :=
by 
  sorry

end regular_18gon_lines_rotational_symmetry_sum_l80_80756


namespace task_completion_time_l80_80281

theorem task_completion_time :
  let rate_A := 1 / 10
  let rate_B := 1 / 15
  let rate_C := 1 / 15
  let combined_rate := rate_A + rate_B + rate_C
  let working_days_A := 2
  let working_days_B := 1
  let rest_day_A := 1
  let rest_days_B := 2
  let work_done_A := rate_A * working_days_A
  let work_done_B := rate_B * working_days_B
  let work_done_C := rate_C * (working_days_A + rest_day_A)
  let work_done := work_done_A + work_done_B + work_done_C
  let remaining_work := 1 - work_done
  let total_days := (work_done / combined_rate) + rest_day_A + rest_days_B
  total_days = 4 + 1 / 7 := by sorry

end task_completion_time_l80_80281


namespace value_of_k_if_two_equal_real_roots_l80_80481

theorem value_of_k_if_two_equal_real_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x + k = 0 → x^2 - 2 * x + k = 0) → k = 1 :=
by
  sorry

end value_of_k_if_two_equal_real_roots_l80_80481


namespace solution_set_inequality_l80_80156

theorem solution_set_inequality (x : ℝ) : 
  (2 < 1 / (x - 1) ∧ 1 / (x - 1) < 3) ↔ (4 / 3 < x ∧ x < 3 / 2) := 
by
  sorry

end solution_set_inequality_l80_80156


namespace money_equations_l80_80996

theorem money_equations (x y : ℝ) (h1 : x + (1 / 2) * y = 50) (h2 : y + (2 / 3) * x = 50) :
  x + (1 / 2) * y = 50 ∧ y + (2 / 3) * x = 50 :=
by
  exact ⟨h1, h2⟩

-- Please note that by stating the theorem this way, we have restated the conditions and conclusion
-- in Lean 4. The proof uses the given conditions directly without the need for intermediate steps.

end money_equations_l80_80996


namespace min_percentage_both_physics_chemistry_l80_80679

/--
Given:
- A certain school conducted a survey.
- 68% of the students like physics.
- 72% of the students like chemistry.

Prove that the minimum percentage of students who like both physics and chemistry is 40%.
-/
theorem min_percentage_both_physics_chemistry (P C : ℝ)
(hP : P = 0.68) (hC : C = 0.72) :
  ∃ B, B = P + C - 1 ∧ B = 0.40 :=
by
  sorry

end min_percentage_both_physics_chemistry_l80_80679


namespace subtract_one_from_solution_l80_80244

theorem subtract_one_from_solution (x : ℝ) (h : 15 * x = 45) : (x - 1) = 2 := 
by {
  sorry
}

end subtract_one_from_solution_l80_80244


namespace max_planes_15_points_l80_80910

-- Define the total number of points
def total_points : ℕ := 15

-- Define the number of collinear points
def collinear_points : ℕ := 5

-- Compute the binomial coefficient C(n, k)
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of planes formed by any 3 out of 15 points
def total_planes : ℕ := binom total_points 3

-- Number of degenerate planes formed by the collinear points
def degenerate_planes : ℕ := binom collinear_points 3

-- Maximum number of unique planes
def max_unique_planes : ℕ := total_planes - degenerate_planes

-- Lean theorem statement
theorem max_planes_15_points : max_unique_planes = 445 :=
by
  sorry

end max_planes_15_points_l80_80910


namespace sufficient_but_not_necessary_condition_l80_80816

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 2 → x^2 + 2 * x - 8 > 0) ∧ (¬(x > 2) → ¬(x^2 + 2 * x - 8 > 0)) → false :=
by 
  sorry

end sufficient_but_not_necessary_condition_l80_80816


namespace geometric_progression_common_ratio_l80_80419

-- Define the problem conditions in Lean 4
theorem geometric_progression_common_ratio (a : ℕ → ℝ) (r : ℝ) (n : ℕ)
  (h_pos : ∀ n, a n > 0) 
  (h_rel : ∀ n, a n = (a (n + 1) + a (n + 2)) / 2 + 2 ) : 
  r = 1 :=
sorry

end geometric_progression_common_ratio_l80_80419


namespace right_triangle_shorter_leg_l80_80604

theorem right_triangle_shorter_leg (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
sorry

end right_triangle_shorter_leg_l80_80604


namespace find_number_l80_80989

theorem find_number (x : ℤ) (h : 3 * (3 * x) = 18) : x = 2 := 
sorry

end find_number_l80_80989


namespace distance_by_which_A_beats_B_l80_80504

noncomputable def speed_of_A : ℝ := 1000 / 192
noncomputable def time_difference : ℝ := 8
noncomputable def distance_beaten : ℝ := speed_of_A * time_difference

theorem distance_by_which_A_beats_B :
  distance_beaten = 41.67 := by
  sorry

end distance_by_which_A_beats_B_l80_80504


namespace charlotte_age_l80_80667

theorem charlotte_age : 
  ∀ (B C E : ℝ), 
    (B = 4 * C) → 
    (E = C + 5) → 
    (B = E) → 
    C = 5 / 3 :=
by
  intros B C E h1 h2 h3
  /- start of the proof -/
  sorry

end charlotte_age_l80_80667


namespace mom_has_enough_money_l80_80777

def original_price : ℝ := 268
def discount_rate : ℝ := 0.2
def money_brought : ℝ := 230
def discounted_price := original_price * (1 - discount_rate)

theorem mom_has_enough_money : money_brought ≥ discounted_price := by
  sorry

end mom_has_enough_money_l80_80777


namespace ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one_l80_80529

theorem ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one
  (m n : ℕ) : (10 ^ m + 1) % (10 ^ n - 1) ≠ 0 := 
  sorry

end ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one_l80_80529


namespace sequence_a_n_l80_80615

theorem sequence_a_n (a : ℤ) (h : (-1)^1 * 1 + a + (-1)^4 * 4 + a = 3 * ( (-1)^2 * 2 + a )) :
  a = -3 ∧ ((-1)^100 * 100 + a) = 97 :=
by
  sorry  -- proof is omitted

end sequence_a_n_l80_80615


namespace jack_more_emails_morning_than_afternoon_l80_80886

def emails_afternoon := 3
def emails_morning := 5

theorem jack_more_emails_morning_than_afternoon :
  emails_morning - emails_afternoon = 2 :=
by
  sorry

end jack_more_emails_morning_than_afternoon_l80_80886


namespace length_PQ_l80_80890

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def distance (P Q : Point3D) : ℝ :=
  Real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2 + (P.z - Q.z) ^ 2)

def P : Point3D := { x := 3, y := 4, z := 5 }

def Q : Point3D := { x := 3, y := 4, z := 0 }

theorem length_PQ : distance P Q = 5 :=
by
  sorry

end length_PQ_l80_80890


namespace find_x7_plus_32x2_l80_80442

theorem find_x7_plus_32x2 (x : ℝ) (h : x^3 + 2 * x = 4) : x^7 + 32 * x^2 = 64 :=
sorry

end find_x7_plus_32x2_l80_80442


namespace minimum_value_of_z_l80_80430

theorem minimum_value_of_z (x y : ℝ) (h : x^2 + 2*x*y - 3*y^2 = 1) : ∃ min_z, min_z = (1 + Real.sqrt 5) / 4 ∧ ∀ z, z = x^2 + y^2 → min_z ≤ z :=
by
  sorry

end minimum_value_of_z_l80_80430


namespace part1_part2_part3_l80_80696

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 - Real.log x) * (x - Real.log x) + 1

variable {a : ℝ}

-- Prove that for all x > 0, if ax^2 > ln x, then f(x) ≥ ax^2 - ln x + 1
theorem part1 (h : ∀ x > 0, a*x^2 > Real.log x) (x : ℝ) (hx : x > 0) :
  f a x ≥ a*x^2 - Real.log x + 1 := sorry

-- Find the maximum value of a given there exists x₀ ∈ (0, +∞) where f(x₀) = 1 + x₀ ln x₀ - ln² x₀
theorem part2 (h : ∃ x₀ > 0, f a x₀ = 1 + x₀ * Real.log x₀ - (Real.log x₀)^2) :
  a ≤ 1 / Real.exp 1 := sorry

-- Prove that for all 1 < x < 2, we have f(x) > ax(2-ax)
theorem part3 (h : ∀ x, 1 < x ∧ x < 2) (x : ℝ) (hx1 : 1 < x) (hx2 : x < 2) :
  f a x > a * x * (2 - a * x) := sorry

end part1_part2_part3_l80_80696


namespace inscribed_triangle_area_l80_80510

noncomputable def triangle_area (r : ℝ) (A B C : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin A + Real.sin B + Real.sin C)

theorem inscribed_triangle_area :
  ∀ (r : ℝ), r = 12 / Real.pi →
  ∀ (A B C : ℝ), A = 40 * Real.pi / 180 → B = 80 * Real.pi / 180 → C = 120 * Real.pi / 180 →
  triangle_area r A B C = 359.4384 / Real.pi^2 :=
by
  intros
  unfold triangle_area
  sorry

end inscribed_triangle_area_l80_80510


namespace area_difference_l80_80074

theorem area_difference (r1 d2 : ℝ) (h1 : r1 = 30) (h2 : d2 = 15) : 
  π * r1^2 - π * (d2 / 2)^2 = 843.75 * π :=
by
  sorry

end area_difference_l80_80074


namespace toll_for_18_wheel_truck_l80_80054

-- Define the number of axles given the conditions
def num_axles (total_wheels rear_axle_wheels front_axle_wheels : ℕ) : ℕ :=
  let rear_axles := (total_wheels - front_axle_wheels) / rear_axle_wheels
  rear_axles + 1

-- Define the toll calculation given the number of axles
def toll (axles : ℕ) : ℝ :=
  1.50 + 0.50 * (axles - 2)

-- Constants specific to the problem
def total_wheels : ℕ := 18
def rear_axle_wheels : ℕ := 4
def front_axle_wheels : ℕ := 2

-- Calculate the number of axles for the given truck
def truck_axles : ℕ := num_axles total_wheels rear_axle_wheels front_axle_wheels

-- The actual statement to prove
theorem toll_for_18_wheel_truck : toll truck_axles = 3.00 :=
  by
    -- proof will go here
    sorry

end toll_for_18_wheel_truck_l80_80054


namespace imaginary_unit_multiplication_l80_80665

theorem imaginary_unit_multiplication (i : ℂ) (h1 : i * i = -1) : i * (1 + i) = i - 1 :=
by
  sorry

end imaginary_unit_multiplication_l80_80665


namespace half_angle_quadrant_l80_80715

theorem half_angle_quadrant (α : ℝ) (k : ℤ) (h1 : k * 360 + 180 < α) (h2 : α < k * 360 + 270) :
    (∃ n : ℤ, (n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ (n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315)) :=
sorry

end half_angle_quadrant_l80_80715


namespace kilometers_to_meters_kilograms_to_grams_l80_80817

def km_to_meters (km: ℕ) : ℕ := km * 1000
def kg_to_grams (kg: ℕ) : ℕ := kg * 1000

theorem kilometers_to_meters (h: 3 = 3): km_to_meters 3 = 3000 := by {
 sorry
}

theorem kilograms_to_grams (h: 4 = 4): kg_to_grams 4 = 4000 := by {
 sorry
}

end kilometers_to_meters_kilograms_to_grams_l80_80817


namespace triangles_hyperbola_parallel_l80_80268

variable (a b c a1 b1 c1 : ℝ)

-- Defining the property that all vertices lie on the hyperbola y = 1/x
def on_hyperbola (x : ℝ) (y : ℝ) : Prop := y = 1 / x

-- Defining the parallelism condition for line segments
def parallel (slope1 slope2 : ℝ) : Prop := slope1 = slope2

theorem triangles_hyperbola_parallel
  (H1A : on_hyperbola a (1 / a))
  (H1B : on_hyperbola b (1 / b))
  (H1C : on_hyperbola c (1 / c))
  (H2A : on_hyperbola a1 (1 / a1))
  (H2B : on_hyperbola b1 (1 / b1))
  (H2C : on_hyperbola c1 (1 / c1))
  (H_AB_parallel_A1B1 : parallel ((b - a) / (a * b * (a - b))) ((b1 - a1) / (a1 * b1 * (a1 - b1))))
  (H_BC_parallel_B1C1 : parallel ((c - b) / (b * c * (b - c))) ((c1 - b1) / (b1 * c1 * (b1 - c1)))) :
  parallel ((c1 - a) / (a * c1 * (a - c1))) ((c - a1) / (a1 * c * (a1 - c))) :=
sorry

end triangles_hyperbola_parallel_l80_80268


namespace solve_equation_l80_80717

theorem solve_equation:
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ 
    (x - y - x / y - (x^3 / y^3) + (x^4 / y^4) = 2017) ∧ 
    ((x = 2949 ∧ y = 983) ∨ (x = 4022 ∧ y = 2011)) :=
sorry

end solve_equation_l80_80717


namespace sugar_bought_l80_80596

noncomputable def P : ℝ := 0.50
noncomputable def S : ℝ := 2.0

theorem sugar_bought : 
  (1.50 * S + 5 * P = 5.50) ∧ 
  (3 * 1.50 + P = 5) ∧
  ((1.50 : ℝ) ≠ 0) → (S = 2) :=
by
  sorry

end sugar_bought_l80_80596


namespace find_b_d_l80_80227

theorem find_b_d (b d : ℕ) (h1 : b + d = 41) (h2 : b < d) : 
  (∃! x, b * x * x + 24 * x + d = 0) → (b = 9 ∧ d = 32) :=
by 
  sorry

end find_b_d_l80_80227


namespace gcd_relatively_prime_l80_80735

theorem gcd_relatively_prime (a : ℤ) (m n : ℕ) (h_odd : a % 2 = 1) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_diff : n ≠ m) :
  Int.gcd (a ^ 2^m + 2 ^ 2^m) (a ^ 2^n + 2 ^ 2^n) = 1 :=
by
  sorry

end gcd_relatively_prime_l80_80735


namespace problem_stmt_l80_80111

variable (a b : ℝ)

theorem problem_stmt (ha : a > 0) (hb : b > 0) (a_plus_b : a + b = 2):
  3 * a^2 + b^2 ≥ 3 ∧ 4 / (a + 1) + 1 / b ≥ 3 := by
  sorry

end problem_stmt_l80_80111


namespace probability_not_paired_shoes_l80_80337

noncomputable def probability_not_pair (total_shoes : ℕ) (pairs : ℕ) (shoes_drawn : ℕ) : ℚ :=
  let total_ways := Nat.choose total_shoes shoes_drawn
  let pair_ways := pairs * Nat.choose 2 2
  let not_pair_ways := total_ways - pair_ways
  not_pair_ways / total_ways

theorem probability_not_paired_shoes (total_shoes pairs shoes_drawn : ℕ) (h1 : total_shoes = 6) 
(h2 : pairs = 3) (h3 : shoes_drawn = 2) :
  probability_not_pair total_shoes pairs shoes_drawn = 4 / 5 :=
by 
  rw [h1, h2, h3]
  simp [probability_not_pair, Nat.choose]
  sorry

end probability_not_paired_shoes_l80_80337


namespace paint_pyramid_l80_80437

theorem paint_pyramid (colors : Finset ℕ) (n : ℕ) (h : colors.card = 5) :
  let ways_to_paint := 5 * 4 * 3 * 2 * 1
  n = ways_to_paint
:=
sorry

end paint_pyramid_l80_80437


namespace sum_of_consecutive_even_integers_l80_80648

theorem sum_of_consecutive_even_integers
  (a1 a2 a3 a4 : ℤ)
  (h1 : a2 = a1 + 2)
  (h2 : a3 = a1 + 4)
  (h3 : a4 = a1 + 6)
  (h_sum : a1 + a3 = 146) :
  a1 + a2 + a3 + a4 = 296 :=
by sorry

end sum_of_consecutive_even_integers_l80_80648


namespace carnations_percentage_l80_80033

-- Definition of the total number of flowers
def total_flowers (F : ℕ) : Prop := 
  F > 0

-- Definition of the pink roses condition
def pink_roses_condition (F : ℕ) : Prop := 
  (1 / 2) * (3 / 5) * F = (3 / 10) * F

-- Definition of the red carnations condition
def red_carnations_condition (F : ℕ) : Prop := 
  (1 / 3) * (2 / 5) * F = (2 / 15) * F

-- Definition of the total pink flowers
def pink_flowers_condition (F : ℕ) : Prop :=
  (3 / 5) * F > 0

-- Proof that the percentage of the flowers that are carnations is 50%
theorem carnations_percentage (F : ℕ) (h_total : total_flowers F) (h_pink_roses : pink_roses_condition F) (h_red_carnations : red_carnations_condition F) (h_pink_flowers : pink_flowers_condition F) :
  (1 / 2) * 100 = 50 :=
by
  sorry

end carnations_percentage_l80_80033


namespace ticket_distribution_l80_80907

theorem ticket_distribution 
    (A Ad C Cd S : ℕ) 
    (h1 : 25 * A + 20 * 50 + 15 * C + 10 * 30 + 20 * S = 7200) 
    (h2 : A + 50 + C + 30 + S = 400)
    (h3 : A + 50 = 2 * S)
    (h4 : Ad = 50)
    (h5 : Cd = 30) : 
    A = 102 ∧ Ad = 50 ∧ C = 142 ∧ Cd = 30 ∧ S = 76 := 
by 
    sorry

end ticket_distribution_l80_80907


namespace final_grey_cats_l80_80789

def initially_total_cats : Nat := 16
def initial_white_cats : Nat := 2
def percent_black_cats : Nat := 25
def black_cats_left_fraction : Nat := 2
def new_white_cats : Nat := 2
def new_grey_cats : Nat := 1

/- We will calculate the number of grey cats after all specified events -/
theorem final_grey_cats :
  let total_cats := initially_total_cats
  let white_cats := initial_white_cats + new_white_cats
  let black_cats := (percent_black_cats * total_cats / 100) / black_cats_left_fraction
  let initial_grey_cats := total_cats - white_cats - black_cats
  let final_grey_cats := initial_grey_cats + new_grey_cats
  final_grey_cats = 11 := by
  sorry

end final_grey_cats_l80_80789


namespace arithmetic_geometric_ratio_l80_80964

theorem arithmetic_geometric_ratio
  (a : ℕ → ℤ) 
  (d : ℤ)
  (h_seq : ∀ n, a (n+1) = a n + d)
  (h_geometric : (a 3)^2 = a 1 * a 9)
  (h_nonzero_d : d ≠ 0) :
  a 11 / a 5 = 5 / 2 :=
by sorry

end arithmetic_geometric_ratio_l80_80964


namespace product_of_three_consecutive_integers_divisible_by_six_l80_80572

theorem product_of_three_consecutive_integers_divisible_by_six (n : ℕ) : 
  6 ∣ (n * (n + 1) * (n + 2)) :=
sorry

end product_of_three_consecutive_integers_divisible_by_six_l80_80572


namespace equation_of_line_l_l80_80150

-- Define the conditions for the parabola and the line
def parabola_vertex : Prop := 
  ∃ C : ℝ × ℝ, C = (0, 0)

def parabola_symmetry_axis : Prop := 
  ∃ l : ℝ → ℝ, ∀ x, l x = -1

def midpoint_of_AB (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1

def parabola_equation (A B : ℝ × ℝ) : Prop :=
  A.2^2 = 4 * A.1 ∧ B.2^2 = 4 * B.1

-- State the theorem to be proven
theorem equation_of_line_l (A B : ℝ × ℝ) :
  parabola_vertex ∧ parabola_symmetry_axis ∧ midpoint_of_AB A B ∧ parabola_equation A B →
  ∃ l : ℝ → ℝ, ∀ x, l x = 2 * x - 3 :=
by sorry

end equation_of_line_l_l80_80150


namespace river_width_l80_80247

theorem river_width (w : ℕ) (speed_const : ℕ) 
(meeting1_from_nearest_shore : ℕ) (meeting2_from_other_shore : ℕ)
(h1 : speed_const = 1) 
(h2 : meeting1_from_nearest_shore = 720) 
(h3 : meeting2_from_other_shore = 400)
(h4 : 3 * w = 3 * meeting1_from_nearest_shore)
(h5 : 2160 = 2 * w - meeting2_from_other_shore) :
w = 1280 :=
by
  {
      sorry
  }

end river_width_l80_80247


namespace repeating_decimal_calculation_l80_80380

theorem repeating_decimal_calculation :
  2 * (8 / 9 - 2 / 9 + 4 / 9) = 20 / 9 :=
by
  -- sorry proof will be inserted here.
  sorry

end repeating_decimal_calculation_l80_80380


namespace inheritance_amount_l80_80795

-- Define the conditions
def federal_tax_rate : ℝ := 0.2
def state_tax_rate : ℝ := 0.1
def total_taxes_paid : ℝ := 10500

-- Lean statement for the proof
theorem inheritance_amount (I : ℝ)
  (h1 : federal_tax_rate = 0.2)
  (h2 : state_tax_rate = 0.1)
  (h3 : total_taxes_paid = 10500)
  (taxes_eq : total_taxes_paid = (federal_tax_rate * I) + (state_tax_rate * (I - (federal_tax_rate * I))))
  : I = 37500 :=
sorry

end inheritance_amount_l80_80795


namespace number_of_ways_to_tile_dominos_l80_80759

-- Define the dimensions of the shapes and the criteria for the tiling problem
def L_shaped_area := 24
def size_of_square := 4
def size_of_rectangles := 2 * 10
def number_of_ways_to_tile := 208

-- Theorem statement
theorem number_of_ways_to_tile_dominos :
  (L_shaped_area = size_of_square + size_of_rectangles) →
  number_of_ways_to_tile = 208 :=
by
  intros h
  sorry

end number_of_ways_to_tile_dominos_l80_80759


namespace cos_two_thirds_pi_l80_80238

theorem cos_two_thirds_pi : Real.cos (2 / 3 * Real.pi) = -1 / 2 :=
by sorry

end cos_two_thirds_pi_l80_80238


namespace binom_eq_sum_l80_80542

theorem binom_eq_sum (x : ℕ) : (∃ x : ℕ, Nat.choose 7 x = 21) ∧ Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4 :=
by
  sorry

end binom_eq_sum_l80_80542


namespace smallest_possible_product_l80_80757

def digits : Set ℕ := {2, 4, 5, 8}

def is_valid_pair (a b : ℤ) : Prop :=
  let (d1, d2, d3, d4) := (a / 10, a % 10, b / 10, b % 10)
  {d1.toNat, d2.toNat, d3.toNat, d4.toNat} ⊆ digits ∧ {d1.toNat, d2.toNat, d3.toNat, d4.toNat} = digits

def smallest_product : ℤ :=
  1200

theorem smallest_possible_product :
  ∀ (a b : ℤ), is_valid_pair a b → a * b ≥ smallest_product :=
by
  intro a b h
  sorry

end smallest_possible_product_l80_80757


namespace avg_weight_of_13_children_l80_80671

-- Definitions based on conditions:
def boys_avg_weight := 160
def boys_count := 8
def girls_avg_weight := 130
def girls_count := 5

-- Calculation to determine the total weights
def boys_total_weight := boys_avg_weight * boys_count
def girls_total_weight := girls_avg_weight * girls_count

-- Combined total weight
def total_weight := boys_total_weight + girls_total_weight

-- Average weight calculation
def children_count := boys_count + girls_count
def avg_weight := total_weight / children_count

-- The theorem to prove:
theorem avg_weight_of_13_children : avg_weight = 148 := by
  sorry

end avg_weight_of_13_children_l80_80671


namespace iron_per_horseshoe_l80_80113

def num_farms := 2
def num_horses_per_farm := 2
def num_stables := 2
def num_horses_per_stable := 5
def num_horseshoes_per_horse := 4
def iron_available := 400
def num_horses_riding_school := 36

-- Lean theorem statement
theorem iron_per_horseshoe : 
  (iron_available / (num_farms * num_horses_per_farm * num_horseshoes_per_horse 
  + num_stables * num_horses_per_stable * num_horseshoes_per_horse 
  + num_horses_riding_school * num_horseshoes_per_horse)) = 2 := 
by 
  sorry

end iron_per_horseshoe_l80_80113


namespace number_of_books_l80_80335

theorem number_of_books (original_books new_books : ℕ) (h1 : original_books = 35) (h2 : new_books = 56) : 
  original_books + new_books = 91 :=
by {
  -- the proof will go here, but is not required for the statement
  sorry
}

end number_of_books_l80_80335


namespace vasya_days_l80_80943

-- Define the variables
variables (x y z w : ℕ)

-- Given conditions
def conditions :=
  (x + y + z + w = 15) ∧
  (9 * x + 4 * z = 30) ∧
  (2 * y + z = 9)

-- Proof problem statement: prove w = 7 given the conditions
theorem vasya_days (x y z w : ℕ) (h : conditions x y z w) : w = 7 :=
by
  -- Use the conditions to deduce w = 7
  sorry

end vasya_days_l80_80943


namespace extreme_value_l80_80800

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * (x - 1)

theorem extreme_value (a : ℝ) (h : a > 0) :
  ∃ x : ℝ, f x a = a - Real.log a - 1 ∧ (∀ y : ℝ, f y a ≤ f x a) :=
sorry

end extreme_value_l80_80800


namespace sin_seven_pi_over_six_l80_80997

theorem sin_seven_pi_over_six :
  Real.sin (7 * Real.pi / 6) = - 1 / 2 :=
by
  sorry

end sin_seven_pi_over_six_l80_80997


namespace constant_function_of_inequality_l80_80122

theorem constant_function_of_inequality (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end constant_function_of_inequality_l80_80122


namespace max_player_salary_l80_80404

theorem max_player_salary (n : ℕ) (min_salary total_salary : ℕ) (player_count : ℕ)
  (h1 : player_count = 25)
  (h2 : min_salary = 15000)
  (h3 : total_salary = 850000)
  (h4 : n = 24 * min_salary)
  : (total_salary - n) = 490000 := 
by
  -- assumptions ensure that n represents the total minimum salaries paid to 24 players
  sorry

end max_player_salary_l80_80404


namespace find_age_l80_80389

open Nat

-- Definition of ages
def Teacher_Zhang_age (z : Nat) := z
def Wang_Bing_age (w : Nat) := w

-- Conditions
axiom teacher_zhang_condition (z w : Nat) : z = 3 * w + 4
axiom age_comparison_condition (z w : Nat) : z - 10 = w + 10

-- Proposition to prove
theorem find_age (z w : Nat) (hz : z = 3 * w + 4) (hw : z - 10 = w + 10) : z = 28 ∧ w = 8 := by
  sorry

end find_age_l80_80389


namespace second_group_students_l80_80785

theorem second_group_students 
  (total_students : ℕ) 
  (first_group_students : ℕ) 
  (h1 : total_students = 71) 
  (h2 : first_group_students = 34) : 
  total_students - first_group_students = 37 :=
by 
  sorry

end second_group_students_l80_80785


namespace Veenapaniville_high_schools_l80_80518

theorem Veenapaniville_high_schools :
  ∃ (districtA districtB districtC : ℕ),
    districtA + districtB + districtC = 50 ∧
    (districtA + districtB + districtC = 50) ∧
    (∃ (publicB parochialB privateB : ℕ), 
      publicB + parochialB + privateB = 17 ∧ privateB = 2) ∧
    (∃ (publicC parochialC privateC : ℕ),
      publicC = 9 ∧ parochialC = 9 ∧ privateC = 9 ∧ publicC + parochialC + privateC = 27) ∧
    districtB = 17 ∧
    districtC = 27 →
    districtA = 6 := by
  sorry

end Veenapaniville_high_schools_l80_80518


namespace albums_either_but_not_both_l80_80488

-- Definition of the problem conditions
def shared_albums : Nat := 11
def andrew_total_albums : Nat := 20
def bob_exclusive_albums : Nat := 8

-- Calculate Andrew's exclusive albums
def andrew_exclusive_albums : Nat := andrew_total_albums - shared_albums

-- Question: Prove the total number of albums in either Andrew's or Bob's collection but not both is 17
theorem albums_either_but_not_both : 
  andrew_exclusive_albums + bob_exclusive_albums = 17 := 
by
  sorry

end albums_either_but_not_both_l80_80488


namespace operation_ab_equals_nine_l80_80901

variable (a b : ℝ)

def operation (x y : ℝ) : ℝ := a * x + b * y - 1

theorem operation_ab_equals_nine
  (h1 : operation a b 1 2 = 4)
  (h2 : operation a b (-2) 3 = 10)
  : a * b = 9 :=
by
  sorry

end operation_ab_equals_nine_l80_80901


namespace quadratic_function_symmetry_l80_80877

theorem quadratic_function_symmetry
  (p : ℝ → ℝ)
  (h_sym : ∀ x, p (5.5 - x) = p (5.5 + x))
  (h_0 : p 0 = -4) :
  p 11 = -4 :=
by sorry

end quadratic_function_symmetry_l80_80877


namespace inverse_proportion_function_increasing_l80_80066

theorem inverse_proportion_function_increasing (m : ℝ) :
  (∀ x1 x2 : ℝ, (0 < x1) → (x1 < x2) → (y = (m - 5) / x1) < (y = (m - 5) / x2)) ↔ m < 5 :=
by
  sorry

end inverse_proportion_function_increasing_l80_80066


namespace inv_geom_seq_prod_next_geom_seq_l80_80694

variable {a : Nat → ℝ} (q : ℝ) (h_q : q ≠ 0)
variable (h_geom : ∀ n, a (n + 1) = q * a n)

theorem inv_geom_seq :
  ∀ n, ∃ c q_inv, (q_inv ≠ 0) ∧ (1 / a n = c * q_inv ^ n) :=
sorry

theorem prod_next_geom_seq :
  ∀ n, ∃ c q_sq, (q_sq ≠ 0) ∧ (a n * a (n + 1) = c * q_sq ^ n) :=
sorry

end inv_geom_seq_prod_next_geom_seq_l80_80694


namespace sum_squares_l80_80496

theorem sum_squares (w x y z : ℝ) (h1 : w + x + y + z = 0) (h2 : w^2 + x^2 + y^2 + z^2 = 1) :
  -1 ≤ w * x + x * y + y * z + z * w ∧ w * x + x * y + y * z + z * w ≤ 0 := 
by 
  sorry

end sum_squares_l80_80496


namespace triangle_angle_A_triangle_length_b_l80_80401

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (m n : ℝ × ℝ)
variable (S : ℝ)

theorem triangle_angle_A (h1 : a = 7) (h2 : c = 8) (h3 : m = (1, 7 * a)) (h4 : n = (-4 * a, Real.sin C))
  (h5 : m.1 * n.1 + m.2 * n.2 = 0) : 
  A = Real.pi / 6 := 
  sorry

theorem triangle_length_b (h1 : a = 7) (h2 : c = 8) (h3 : (7 * 8 * Real.sin B) / 2 = 16 * Real.sqrt 3) :
  b = Real.sqrt 97 :=
  sorry

end triangle_angle_A_triangle_length_b_l80_80401


namespace distance_M_to_AB_l80_80909

noncomputable def distance_to_ab : ℝ := 5.8

theorem distance_M_to_AB
  (M : Point)
  (A B C : Point)
  (d_AC d_BC : ℝ)
  (AB BC AC : ℝ)
  (H1 : d_AC = 2)
  (H2 : d_BC = 4)
  (H3 : AB = 10)
  (H4 : BC = 17)
  (H5 : AC = 21) :
  distance_to_ab = 5.8 :=
by
  sorry

end distance_M_to_AB_l80_80909


namespace farm_transaction_difference_l80_80747

theorem farm_transaction_difference
  (x : ℕ)
  (h_initial : 6 * x - 15 > 0) -- Ensure initial horses are enough to sell 15
  (h_ratio_initial : 6 * x = x * 6)
  (h_ratio_final : (6 * x - 15) = 3 * (x + 15)) :
  (6 * x - 15) - (x + 15) = 70 :=
by
  sorry

end farm_transaction_difference_l80_80747


namespace determine_function_l80_80439

theorem determine_function (f : ℝ → ℝ)
    (h1 : f 1 = 0)
    (h2 : ∀ x y : ℝ, |f x - f y| = |x - y|) :
    (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) := by
  sorry

end determine_function_l80_80439


namespace range_of_a_l80_80687

variable (a : ℝ)
def proposition_p := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def proposition_q := ∃ x₀ : ℝ, x₀^2 - x₀ + a = 0

theorem range_of_a (h1 : proposition_p a ∨ proposition_q a)
    (h2 : ¬ (proposition_p a ∧ proposition_q a)) :
    a < 0 ∨ (1 / 4) < a ∧ a < 4 :=
  sorry

end range_of_a_l80_80687


namespace problem1_problem2_l80_80686

-- Definitions
variables {a b z : ℝ}

-- Problem 1 translated to Lean
theorem problem1 (h1 : a + 2 * b = 9) (h2 : |9 - 2 * b| + |a + 1| < 3) : -2 < a ∧ a < 1 := 
sorry

-- Problem 2 translated to Lean
theorem problem2 (h1 : a + 2 * b = 9) (ha_pos : 0 < a) (hb_pos : 0 < b) : 
  ∃ z : ℝ, z = a * b^2 ∧ ∀ w : ℝ, (∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a + 2 * b = 9 ∧ w = a * b^2) → w ≤ 27 :=
sorry

end problem1_problem2_l80_80686


namespace deputy_more_enemies_than_friends_l80_80220

theorem deputy_more_enemies_than_friends (deputies : Type) 
  (friendship hostility indifference : deputies → deputies → Prop)
  (h_symm_friend : ∀ (a b : deputies), friendship a b → friendship b a)
  (h_symm_hostile : ∀ (a b : deputies), hostility a b → hostility b a)
  (h_symm_indiff : ∀ (a b : deputies), indifference a b → indifference b a)
  (h_enemy_exists : ∀ (d : deputies), ∃ (e : deputies), hostility d e)
  (h_principle : ∀ (a b c : deputies), hostility a b → friendship b c → hostility a c) :
  ∃ (d : deputies), ∃ (f e : ℕ), f < e :=
sorry

end deputy_more_enemies_than_friends_l80_80220


namespace Anna_phone_chargers_l80_80565

-- Define the conditions and the goal in Lean
theorem Anna_phone_chargers (P L : ℕ) (h1 : L = 5 * P) (h2 : P + L = 24) : P = 4 :=
by
  sorry

end Anna_phone_chargers_l80_80565


namespace actual_plot_area_in_acres_l80_80913

-- Define the conditions
def base1_cm := 18
def base2_cm := 12
def height_cm := 8
def scale_cm_to_miles := 5
def sq_mile_to_acres := 640

-- Prove the question which is to find the actual plot area in acres
theorem actual_plot_area_in_acres : 
  (1/2 * (base1_cm + base2_cm) * height_cm * (scale_cm_to_miles ^ 2) * sq_mile_to_acres) = 1920000 :=
by
  sorry

end actual_plot_area_in_acres_l80_80913


namespace arithmetic_expression_evaluation_l80_80294

theorem arithmetic_expression_evaluation :
  (-18) + (-12) - (-33) + 17 = 20 :=
by
  sorry

end arithmetic_expression_evaluation_l80_80294


namespace mean_of_combined_sets_l80_80236

theorem mean_of_combined_sets
  (S₁ : Finset ℝ) (S₂ : Finset ℝ)
  (h₁ : S₁.card = 7) (h₂ : S₂.card = 8)
  (mean_S₁ : (S₁.sum id) / S₁.card = 15)
  (mean_S₂ : (S₂.sum id) / S₂.card = 26)
  : (S₁.sum id + S₂.sum id) / (S₁.card + S₂.card) = 20.8667 := 
by
  sorry

end mean_of_combined_sets_l80_80236


namespace derivative_of_y_l80_80361

open Real

noncomputable def y (x : ℝ) : ℝ := (cos (log 7) * (sin (7 * x)) ^ 2) / (7 * cos (14 * x))

theorem derivative_of_y (x : ℝ) : deriv y x = (cos (log 7) * tan (14 * x)) / cos (14 * x) := sorry

end derivative_of_y_l80_80361


namespace circumcircle_of_right_triangle_l80_80962

theorem circumcircle_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a = 6) (hb : b = 8) (hc : c = 10) :
  ∃ (x y : ℝ), (x - 0)^2 + (y - 0)^2 = 25 :=
by
  sorry

end circumcircle_of_right_triangle_l80_80962


namespace solve_eq_f_x_plus_3_l80_80740

-- Define the function f with its piecewise definition based on the conditions
noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x^2 - 3 * x
  else -(x^2 - 3 * (-x))

-- Define the main theorem to find the solution set
theorem solve_eq_f_x_plus_3 (x : ℝ) :
  f x = x + 3 ↔ x = 2 + Real.sqrt 7 ∨ x = -1 ∨ x = -3 :=
by sorry

end solve_eq_f_x_plus_3_l80_80740


namespace moles_of_H2_required_l80_80718

theorem moles_of_H2_required 
  (moles_C : ℕ) 
  (moles_O2 : ℕ) 
  (moles_CH4 : ℕ) 
  (moles_CO2 : ℕ) 
  (balanced_reaction_1 : ℕ → ℕ → ℕ → Prop)
  (balanced_reaction_2 : ℕ → ℕ → ℕ → ℕ → Prop)
  (H_balanced : balanced_reaction_2 2 4 2 1)
  (H_form_CO2 : balanced_reaction_1 1 1 1) :
  moles_C = 2 ∧ moles_O2 = 1 ∧ moles_CH4 = 2 ∧ moles_CO2 = 1 → (∃ moles_H2, moles_H2 = 4) :=
by sorry

end moles_of_H2_required_l80_80718


namespace vlad_score_l80_80436

theorem vlad_score :
  ∀ (rounds wins : ℕ) (totalPoints taroPoints vladPoints : ℕ),
    rounds = 30 →
    (wins = 5) →
    (totalPoints = rounds * wins) →
    (taroPoints = (3 * totalPoints) / 5 - 4) →
    (vladPoints = totalPoints - taroPoints) →
    vladPoints = 64 :=
by
  intros rounds wins totalPoints taroPoints vladPoints h1 h2 h3 h4 h5
  sorry

end vlad_score_l80_80436


namespace johns_total_cost_l80_80755

variable (C_s C_d : ℝ)

theorem johns_total_cost (h_s : C_s = 20) (h_d : C_d = 0.5 * C_s) : C_s + C_d = 30 := by
  sorry

end johns_total_cost_l80_80755


namespace quadruple_pieces_sold_l80_80927

theorem quadruple_pieces_sold (split_earnings : (2 : ℝ) * 5 = 10) 
  (single_pieces_sold : 100 * (0.01 : ℝ) = 1) 
  (double_pieces_sold : 45 * (0.02 : ℝ) = 0.9) 
  (triple_pieces_sold : 50 * (0.03 : ℝ) = 1.5) : 
  let total_earnings := 10
  let earnings_from_others := 3.4
  let quadruple_piece_price := 0.04
  total_earnings - earnings_from_others = 6.6 → 
  6.6 / quadruple_piece_price = 165 :=
by 
  intros 
  sorry

end quadruple_pieces_sold_l80_80927


namespace mod_remainder_l80_80850

theorem mod_remainder (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by
  sorry

end mod_remainder_l80_80850


namespace rate_per_kg_for_fruits_l80_80743

-- Definitions and conditions
def total_cost (rate_per_kg : ℝ) : ℝ := 8 * rate_per_kg + 9 * rate_per_kg

def total_paid : ℝ := 1190

theorem rate_per_kg_for_fruits : ∃ R : ℝ, total_cost R = total_paid ∧ R = 70 :=
by
  sorry

end rate_per_kg_for_fruits_l80_80743


namespace initial_number_of_balls_l80_80636

theorem initial_number_of_balls (T B : ℕ) (P : ℚ) (after3_blue : ℕ) (prob : ℚ) :
  B = 7 → after3_blue = B - 3 → prob = after3_blue / T → prob = 1/3 → T = 15 :=
by
  sorry

end initial_number_of_balls_l80_80636


namespace S9_value_l80_80952

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)

-- Define the arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a_n (n + 1) - a_n n) = (a_n 1 - a_n 0)

-- Sum of the first n terms of arithmetic sequence
def sum_first_n_terms (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S_n n = n * (a_n 0 + a_n (n - 1)) / 2

-- Given conditions: 
axiom a4_plus_a6 : a_n 4 + a_n 6 = 12
axiom S_definition : sum_first_n_terms S_n a_n

theorem S9_value : S_n 9 = 54 :=
by
  -- assuming the given conditions and definitions, we aim to prove the desired theorem.
  sorry

end S9_value_l80_80952


namespace diameter_correct_l80_80929

noncomputable def diameter_of_circle (C : ℝ) (hC : C = 36) : ℝ :=
  let r := C / (2 * Real.pi)
  2 * r

theorem diameter_correct (C : ℝ) (hC : C = 36) : diameter_of_circle C hC = 36 / Real.pi := by
  sorry

end diameter_correct_l80_80929


namespace acute_angle_probability_l80_80019

noncomputable def prob_acute_angle : ℝ :=
  let m_values := [1, 2, 3, 4, 5, 6]
  let outcomes_count := (36 : ℝ)
  let good_outcomes_count := (15 : ℝ)
  good_outcomes_count / outcomes_count

theorem acute_angle_probability :
  prob_acute_angle = 5 / 12 :=
by
  sorry

end acute_angle_probability_l80_80019


namespace number_of_geese_more_than_ducks_l80_80126

theorem number_of_geese_more_than_ducks (geese ducks : ℝ) (h1 : geese = 58.0) (h2 : ducks = 37.0) :
  geese - ducks = 21.0 :=
by
  sorry

end number_of_geese_more_than_ducks_l80_80126


namespace evaluate_expr_l80_80021

noncomputable def expr : ℚ :=
  2013 * (5.7 * 4.2 + (21 / 5) * 4.3) / ((14 / 73) * 15 + (5 / 73) * 177 + 656)

theorem evaluate_expr : expr = 126 := by
  sorry

end evaluate_expr_l80_80021


namespace wire_cut_ratio_l80_80176

-- Define lengths a and b
variable (a b : ℝ)

-- Define perimeter equal condition
axiom perimeter_eq : 4 * (a / 4) = 6 * (b / 6)

-- The statement to prove
theorem wire_cut_ratio (h : 4 * (a / 4) = 6 * (b / 6)) : a / b = 1 :=
by
  sorry

end wire_cut_ratio_l80_80176


namespace number_of_members_l80_80297

theorem number_of_members (n : ℕ) (h : n * n = 2025) : n = 45 :=
sorry

end number_of_members_l80_80297


namespace find_original_volume_l80_80217

theorem find_original_volume
  (V : ℝ)
  (h1 : V - (3 / 4) * V = (1 / 4) * V)
  (h2 : (1 / 4) * V - (3 / 4) * ((1 / 4) * V) = (1 / 16) * V)
  (h3 : (1 / 16) * V = 0.2) :
  V = 3.2 :=
by 
  -- Proof skipped, as the assistant is instructed to provide only the statement 
  sorry

end find_original_volume_l80_80217


namespace functional_equation_solution_l80_80352

theorem functional_equation_solution (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  (f x = (1/2) * (x + 1 - 1/x - 1/(1-x))) →
  (f x + f (1 / (1 - x)) = x) :=
sorry

end functional_equation_solution_l80_80352


namespace Jason_spent_correct_amount_l80_80658

def flute_cost : ℝ := 142.46
def music_stand_cost : ℝ := 8.89
def song_book_cost : ℝ := 7.00
def total_cost : ℝ := 158.35

theorem Jason_spent_correct_amount :
  flute_cost + music_stand_cost + song_book_cost = total_cost :=
by
  sorry

end Jason_spent_correct_amount_l80_80658


namespace intersection_points_with_x_axis_l80_80456

theorem intersection_points_with_x_axis (a : ℝ) :
    (∃ x : ℝ, a * x^2 - a * x + 3 * x + 1 = 0 ∧ 
              ∀ x' : ℝ, (x' ≠ x → a * x'^2 - a * x' + 3 * x' + 1 ≠ 0)) ↔ 
    (a = 0 ∨ a = 1 ∨ a = 9) := by 
  sorry

end intersection_points_with_x_axis_l80_80456


namespace proof_problem_l80_80994

theorem proof_problem (p q : Prop) : (p ∧ q) ↔ ¬ (¬ p ∨ ¬ q) :=
sorry

end proof_problem_l80_80994


namespace area_of_triangle_ABC_l80_80527

theorem area_of_triangle_ABC (a b c : ℝ) (h : b^2 - 4 * a * c > 0) :
  let x1 := (-b - Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let x2 := (-b + Real.sqrt (b^2 - 4 * a * c)) / (2 * a)
  let y_vertex := (4 * a * c - b^2) / (4 * a)
  0.5 * (|x2 - x1|) * |y_vertex| = (b^2 - 4 * a * c) * Real.sqrt (b^2 - 4 * a * c) / (8 * a^2) :=
sorry

end area_of_triangle_ABC_l80_80527


namespace infinitely_many_not_2a_3b_5c_l80_80556

theorem infinitely_many_not_2a_3b_5c : ∃ᶠ x : ℤ in Filter.cofinite, ∀ a b c : ℕ, x % 120 ≠ (2^a + 3^b - 5^c) % 120 :=
by
  sorry

end infinitely_many_not_2a_3b_5c_l80_80556


namespace slope_angle_of_line_l80_80979

theorem slope_angle_of_line (α : ℝ) (hα : 0 ≤ α ∧ α < 180) 
    (slope_eq_tan : Real.tan α = 1) : α = 45 :=
by
  sorry

end slope_angle_of_line_l80_80979


namespace part1_part2_part3_l80_80196

def set_A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def set_B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def set_C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

theorem part1 (a : ℝ) : (set_A a ∩ set_B) = (set_A a ∪ set_B) → a = 5 :=
by
  sorry

theorem part2 (a : ℝ) : (∅ ⊂ (set_A a ∩ set_B)) ∧ (set_A a ∩ set_C = ∅) → a = -2 :=
by
  sorry

theorem part3 (a : ℝ) : (set_A a ∩ set_B) = (set_A a ∩ set_C) ∧ (set_A a ∩ set_B ≠ ∅) → a = -3 :=
by
  sorry

end part1_part2_part3_l80_80196


namespace coffee_remaining_after_shrink_l80_80598

-- Definitions of conditions in the problem
def shrink_factor : ℝ := 0.5
def cups_before_shrink : ℕ := 5
def ounces_per_cup_before_shrink : ℝ := 8

-- Definition of the total ounces of coffee remaining after shrinking
def ounces_per_cup_after_shrink : ℝ := ounces_per_cup_before_shrink * shrink_factor
def total_ounces_after_shrink : ℝ := cups_before_shrink * ounces_per_cup_after_shrink

-- The proof statement
theorem coffee_remaining_after_shrink :
  total_ounces_after_shrink = 20 :=
by
  -- Omitting the proof as only the statement is needed
  sorry

end coffee_remaining_after_shrink_l80_80598


namespace slices_per_pack_l80_80661

theorem slices_per_pack (sandwiches : ℕ) (slices_per_sandwich : ℕ) (packs_of_bread : ℕ) (total_slices : ℕ) 
  (h1 : sandwiches = 8) (h2 : slices_per_sandwich = 2) (h3 : packs_of_bread = 4) : 
  total_slices = 4 :=
by
  sorry

end slices_per_pack_l80_80661


namespace problem_statement_l80_80290

theorem problem_statement
  (a b c d : ℕ)
  (h1 : (b + c + d) / 3 + 2 * a = 54)
  (h2 : (a + c + d) / 3 + 2 * b = 50)
  (h3 : (a + b + d) / 3 + 2 * c = 42)
  (h4 : (a + b + c) / 3 + 2 * d = 30) :
  a = 17 ∨ b = 17 ∨ c = 17 ∨ d = 17 :=
by
  sorry

end problem_statement_l80_80290


namespace worker_assignment_l80_80766

theorem worker_assignment (x : ℕ) (y : ℕ) 
  (h1 : x + y = 90)
  (h2 : 2 * 15 * x = 3 * 8 * y) : 
  (x = 40 ∧ y = 50) := by
  sorry

end worker_assignment_l80_80766


namespace impossible_coins_l80_80410

theorem impossible_coins (p1 p2 : ℝ) (hp1 : 0 ≤ p1 ∧ p1 ≤ 1) (hp2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (1 - p1) * (1 - p2) = p1 * p2 ∧ p1 * (1 - p2) + p2 * (1 - p1) = p1 * p2 → false :=
by 
  sorry

end impossible_coins_l80_80410


namespace probability_event_comparison_l80_80165

theorem probability_event_comparison (m n : ℕ) :
  let P_A := (2 * m * n) / (m + n)^2
  let P_B := (m^2 + n^2) / (m + n)^2
  P_A ≤ P_B ∧ (P_A = P_B ↔ m = n) :=
by
  sorry

end probability_event_comparison_l80_80165


namespace fraction_multiplication_subtraction_l80_80166

theorem fraction_multiplication_subtraction :
  (3 + 1 / 117) * (4 + 1 / 119) - (2 - 1 / 117) * (6 - 1 / 119) - (5 / 119) = 10 / 117 :=
by
  sorry

end fraction_multiplication_subtraction_l80_80166


namespace two_perfect_squares_not_two_perfect_cubes_l80_80151

-- Define the initial conditions as Lean assertions
def isSumOfTwoPerfectSquares (n : ℕ) := ∃ a b : ℕ, n = a^2 + b^2

def isSumOfTwoPerfectCubes (n : ℕ) := ∃ a b : ℕ, n = a^3 + b^3

-- Lean 4 statement to show 2005^2005 is a sum of two perfect squares
theorem two_perfect_squares :
  isSumOfTwoPerfectSquares (2005^2005) :=
sorry

-- Lean 4 statement to show 2005^2005 is not a sum of two perfect cubes
theorem not_two_perfect_cubes :
  ¬ isSumOfTwoPerfectCubes (2005^2005) :=
sorry

end two_perfect_squares_not_two_perfect_cubes_l80_80151


namespace solve_system_of_inequalities_l80_80903

theorem solve_system_of_inequalities (x : ℝ) :
  (2 * x + 1 > x) ∧ (x < -3 * x + 8) ↔ -1 < x ∧ x < 2 :=
by
  sorry

end solve_system_of_inequalities_l80_80903


namespace shiela_neighbors_l80_80359

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) (neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : drawings_per_neighbor = 9) : neighbors = total_drawings / drawings_per_neighbor :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end shiela_neighbors_l80_80359


namespace total_games_in_season_l80_80461

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem total_games_in_season
  (teams : ℕ)
  (games_per_pair : ℕ)
  (h_teams : teams = 30)
  (h_games_per_pair : games_per_pair = 6) :
  (choose 30 2 * games_per_pair) = 2610 :=
  by
    sorry

end total_games_in_season_l80_80461


namespace p_satisfies_conditions_l80_80782

noncomputable def p (x : ℕ) : ℕ := sorry

theorem p_satisfies_conditions (h_monic : p 1 = 1 ∧ p 2 = 2 ∧ p 3 = 3 ∧ p 4 = 4 ∧ p 5 = 5) : 
  p 6 = 126 := sorry

end p_satisfies_conditions_l80_80782


namespace product_of_two_numbers_l80_80073

theorem product_of_two_numbers (x y : ℚ) 
  (h1 : x + y = 8 * (x - y)) 
  (h2 : x * y = 15 * (x - y)) : 
  x * y = 100 / 7 := 
by 
  sorry

end product_of_two_numbers_l80_80073


namespace driving_distance_l80_80651

def miles_per_gallon : ℕ := 20
def gallons_of_gas : ℕ := 5

theorem driving_distance :
  miles_per_gallon * gallons_of_gas = 100 :=
  sorry

end driving_distance_l80_80651


namespace marks_change_factor_l80_80370

def total_marks (n : ℕ) (avg : ℝ) : ℝ := n * avg

theorem marks_change_factor 
  (n : ℕ) (initial_avg new_avg : ℝ) 
  (initial_total := total_marks n initial_avg) 
  (new_total := total_marks n new_avg)
  (h1 : initial_avg = 36)
  (h2 : new_avg = 72)
  (h3 : n = 12):
  (new_total / initial_total) = 2 :=
by
  sorry

end marks_change_factor_l80_80370


namespace roots_sum_of_squares_l80_80332

theorem roots_sum_of_squares {p q r : ℝ} 
  (h₁ : ∀ x : ℝ, (x - p) * (x - q) * (x - r) = x^3 - 24 * x^2 + 50 * x - 35) :
  (p + q)^2 + (q + r)^2 + (r + p)^2 = 1052 :=
by
  have h_sum : p + q + r = 24 := by sorry
  have h_product : p * q + q * r + r * p = 50 := by sorry
  sorry

end roots_sum_of_squares_l80_80332


namespace factorial_sum_power_of_two_l80_80663

theorem factorial_sum_power_of_two (a b c : ℕ) (hac : 0 < a) (hbc : 0 < b) (hcc : 0 < c) :
  a! + b! = 2 ^ c! ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 2 ∧ b = 2 ∧ c = 2) :=
by
  sorry

end factorial_sum_power_of_two_l80_80663


namespace dice_probability_sum_15_l80_80533

def is_valid_combination (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ a + b + c = 15

def count_outcomes : ℕ :=
  6 * 6 * 6

def count_valid_combinations : ℕ :=
  10  -- From the list of valid combinations

def probability (valid_count total_count : ℕ) : ℚ :=
  valid_count / total_count

theorem dice_probability_sum_15 : probability count_valid_combinations count_outcomes = 5 / 108 :=
by
  sorry

end dice_probability_sum_15_l80_80533


namespace matchsticks_left_l80_80120

theorem matchsticks_left (total_matchsticks elvis_match_per_square ralph_match_per_square elvis_squares ralph_squares : ℕ)
  (h1 : total_matchsticks = 50)
  (h2 : elvis_match_per_square = 4)
  (h3 : ralph_match_per_square = 8)
  (h4 : elvis_squares = 5)
  (h5 : ralph_squares = 3) :
  total_matchsticks - (elvis_match_per_square * elvis_squares + ralph_match_per_square * ralph_squares) = 6 := 
by
  sorry

end matchsticks_left_l80_80120


namespace eq_abc_gcd_l80_80313

theorem eq_abc_gcd
  (a b c d : ℕ)
  (h1 : a^a * b^(a + b) = c^c * d^(c + d))
  (h2 : Nat.gcd a b = 1)
  (h3 : Nat.gcd c d = 1) : 
  a = c ∧ b = d := 
sorry

end eq_abc_gcd_l80_80313


namespace calories_difference_l80_80721

theorem calories_difference
  (calories_squirrel : ℕ := 300)
  (squirrels_per_hour : ℕ := 6)
  (calories_rabbit : ℕ := 800)
  (rabbits_per_hour : ℕ := 2) :
  ((squirrels_per_hour * calories_squirrel) - (rabbits_per_hour * calories_rabbit)) = 200 :=
by
  sorry

end calories_difference_l80_80721


namespace line_does_not_pass_through_third_quadrant_l80_80625

theorem line_does_not_pass_through_third_quadrant (x y : ℝ) (h : y = -x + 1) :
  ¬(x < 0 ∧ y < 0) :=
sorry

end line_does_not_pass_through_third_quadrant_l80_80625


namespace triangle_side_ratio_triangle_area_l80_80221

-- Definition of Problem 1
theorem triangle_side_ratio {A B C a b c : ℝ} 
  (h1 : 4 * Real.sin A = 3 * Real.sin B)
  (h2 : 2 * a * Real.cos C + 2 * c * Real.cos A = a + c)
  (h3 : a / b = Real.sin A / Real.sin B)
  (h4 : b / c = Real.sin B / Real.sin C)
  : c / b = 5 / 4 :=
sorry

-- Definition of Problem 2
theorem triangle_area {A B C a b c : ℝ} 
  (h1 : C = 2 * Real.pi / 3)
  (h2 : c - a = 8)
  (h3 : 2 * a * Real.cos C + 2 * c * Real.cos A = a + c)
  (h4 : a + c = 2 * b)
  : (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 :=
sorry

end triangle_side_ratio_triangle_area_l80_80221


namespace min_dist_of_PQ_l80_80149

open Real

theorem min_dist_of_PQ :
  ∀ (P Q : ℝ × ℝ),
    (P.fst - 3)^2 + (P.snd + 1)^2 = 4 →
    Q.fst = -3 →
    ∃ (min_dist : ℝ), min_dist = 4 :=
by
  sorry

end min_dist_of_PQ_l80_80149


namespace quadratic_completing_the_square_q_l80_80179

theorem quadratic_completing_the_square_q (x p q : ℝ) (h : 4 * x^2 + 8 * x - 468 = 0) :
  (∃ p, (x + p)^2 = q) → q = 116 := sorry

end quadratic_completing_the_square_q_l80_80179


namespace jigi_scored_55_percent_l80_80270

noncomputable def jigi_percentage (max_score : ℕ) (avg_score : ℕ) (gibi_pct mike_pct lizzy_pct : ℕ) : ℕ := sorry

theorem jigi_scored_55_percent :
  jigi_percentage 700 490 59 99 67 = 55 :=
sorry

end jigi_scored_55_percent_l80_80270


namespace total_items_proof_l80_80339

noncomputable def totalItemsBought (budget : ℕ) (sandwichCost : ℕ) 
  (pastryCost : ℕ) (maxSandwiches : ℕ) : ℕ :=
  let s := min (budget / sandwichCost) maxSandwiches
  let remainingMoney := budget - s * sandwichCost
  let p := remainingMoney / pastryCost
  s + p

theorem total_items_proof : totalItemsBought 50 6 2 7 = 11 := by
  sorry

end total_items_proof_l80_80339


namespace lcm_of_9_12_18_l80_80514

-- Let's declare the numbers involved
def num1 : ℕ := 9
def num2 : ℕ := 12
def num3 : ℕ := 18

-- Define what it means for a number to be the LCM of num1, num2, and num3
def is_lcm (a b c l : ℕ) : Prop :=
  l % a = 0 ∧ l % b = 0 ∧ l % c = 0 ∧
  ∀ m, (m % a = 0 ∧ m % b = 0 ∧ m % c = 0) → l ≤ m

-- Now state the theorem
theorem lcm_of_9_12_18 : is_lcm num1 num2 num3 36 :=
by
  sorry

end lcm_of_9_12_18_l80_80514


namespace orange_balloons_count_l80_80486

variable (original_orange_balloons : ℝ)
variable (found_orange_balloons : ℝ)
variable (total_orange_balloons : ℝ)

theorem orange_balloons_count :
  original_orange_balloons = 9.0 →
  found_orange_balloons = 2.0 →
  total_orange_balloons = original_orange_balloons + found_orange_balloons →
  total_orange_balloons = 11.0 := by
  sorry

end orange_balloons_count_l80_80486


namespace solution_set_of_inequality_l80_80327

variable {R : Type} [LinearOrderedField R]

theorem solution_set_of_inequality (f : R -> R) (h1 : ∀ x, f (-x) = -f x) (h2 : ∀ x y, 0 < x ∧ x < y → f x < f y) (h3 : f 1 = 0) :
  { x : R | (f x - f (-x)) / x < 0 } = { x : R | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) } :=
sorry

end solution_set_of_inequality_l80_80327


namespace sum_of_coordinates_of_A_l80_80048

open Real

theorem sum_of_coordinates_of_A (A B C : ℝ × ℝ) (h1 : B = (2, 8)) (h2 : C = (5, 2))
  (h3 : ∃ (k : ℝ), A = ((2 * (B.1:ℝ) + C.1) / 3, (2 * (B.2:ℝ) + C.2) / 3) ∧ k = 1/3) :
  A.1 + A.2 = 9 :=
sorry

end sum_of_coordinates_of_A_l80_80048


namespace find_x_for_abs_expression_zero_l80_80984

theorem find_x_for_abs_expression_zero (x : ℚ) : |5 * x - 2| = 0 → x = 2 / 5 := by
  sorry

end find_x_for_abs_expression_zero_l80_80984


namespace vertex_angle_measure_l80_80821

-- Define the isosceles triangle and its properties
def is_isosceles_triangle (A B C : ℝ) (a b c : ℝ) :=
  (A = B ∨ B = C ∨ C = A) ∧ (a + b + c = 180)

-- Define the conditions based on the problem statement
def two_angles_sum_to_100 (x y : ℝ) := x + y = 100

-- The measure of the vertex angle
theorem vertex_angle_measure (A B C : ℝ) (a b c : ℝ) 
  (h1 : is_isosceles_triangle A B C a b c) (h2 : two_angles_sum_to_100 A B) :
  C = 20 ∨ C = 80 :=
sorry

end vertex_angle_measure_l80_80821


namespace part_one_part_two_l80_80310

noncomputable def f (x a : ℝ) : ℝ :=
  |x + a| + 2 * |x - 1|

theorem part_one (a : ℝ) (h : a = 1) : 
  ∃ x : ℝ, f x 1 = 2 :=
sorry

theorem part_two (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : ∀ x : ℝ, 1 ≤ x → x ≤ 2 → f x a > x^2 - b + 1) : 
  (a + 1 / 2)^2 + (b + 1 / 2)^2 > 2 :=
sorry

end part_one_part_two_l80_80310


namespace feb1_is_wednesday_l80_80168

-- Define the days of the week as a data type
inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

-- Define a function that models the backward count for days of the week from a given day
def days_backward (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days % 7 with
  | 0 => start
  | 1 => match start with
         | Sunday => Saturday
         | Monday => Sunday
         | Tuesday => Monday
         | Wednesday => Tuesday
         | Thursday => Wednesday
         | Friday => Thursday
         | Saturday => Friday
  | 2 => match start with
         | Sunday => Friday
         | Monday => Saturday
         | Tuesday => Sunday
         | Wednesday => Monday
         | Thursday => Tuesday
         | Friday => Wednesday
         | Saturday => Thursday
  | 3 => match start with
         | Sunday => Thursday
         | Monday => Friday
         | Tuesday => Saturday
         | Wednesday => Sunday
         | Thursday => Monday
         | Friday => Tuesday
         | Saturday => Wednesday
  | 4 => match start with
         | Sunday => Wednesday
         | Monday => Thursday
         | Tuesday => Friday
         | Wednesday => Saturday
         | Thursday => Sunday
         | Friday => Monday
         | Saturday => Tuesday
  | 5 => match start with
         | Sunday => Tuesday
         | Monday => Wednesday
         | Tuesday => Thursday
         | Wednesday => Friday
         | Thursday => Saturday
         | Friday => Sunday
         | Saturday => Monday
  | 6 => match start with
         | Sunday => Monday
         | Monday => Tuesday
         | Tuesday => Wednesday
         | Wednesday => Thursday
         | Thursday => Friday
         | Friday => Saturday
         | Saturday => Sunday
  | _ => start  -- This case is unreachable because days % 7 is always between 0 and 6

-- Proof statement: given February 28 is a Tuesday, prove that February 1 is a Wednesday
theorem feb1_is_wednesday (h : days_backward Tuesday 27 = Wednesday) : True :=
by
  sorry

end feb1_is_wednesday_l80_80168


namespace children_more_than_adults_l80_80862

-- Conditions
def total_members : ℕ := 120
def adult_percentage : ℝ := 0.40
def child_percentage : ℝ := 1 - adult_percentage

-- Proof problem statement
theorem children_more_than_adults : 
  let number_of_adults := adult_percentage * total_members
  let number_of_children := child_percentage * total_members
  let difference := number_of_children - number_of_adults
  difference = 24 :=
by
  sorry

end children_more_than_adults_l80_80862


namespace fred_money_last_week_l80_80205

theorem fred_money_last_week (F_current F_earned F_last_week : ℕ) 
  (h_current : F_current = 86)
  (h_earned : F_earned = 63)
  (h_last_week : F_last_week = 23) :
  F_current - F_earned = F_last_week := 
by
  sorry

end fred_money_last_week_l80_80205


namespace gas_pipe_probability_l80_80163

-- Define the conditions as Lean hypotheses
theorem gas_pipe_probability (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y)
    (hxy : x + y ≤ 100) (h25x : 25 ≤ x) (h25y : 25 ≤ y)
    (h100xy : 75 ≥ x + y) :
  ∃ (p : ℝ), p = 1/16 :=
by
  sorry

end gas_pipe_probability_l80_80163


namespace find_X_value_l80_80164

theorem find_X_value (X : ℝ) : 
  (1.5 * ((3.6 * 0.48 * 2.5) / (0.12 * X * 0.5)) = 1200.0000000000002) → 
  X = 0.225 :=
by
  sorry

end find_X_value_l80_80164


namespace average_of_remaining_numbers_l80_80319

theorem average_of_remaining_numbers 
  (numbers : List ℝ) 
  (h_len : numbers.length = 50) 
  (h_avg : (numbers.sum / 50) = 20)
  (h_disc : 45 ∈ numbers ∧ 55 ∈ numbers) 
  (h_count_45_55 : numbers.count 45 = 1 ∧ numbers.count 55 = 1) :
  (numbers.sum - 45 - 55) / (50 - 2) = 18.75 :=
by
  sorry

end average_of_remaining_numbers_l80_80319


namespace sequence_arithmetic_mean_l80_80629

theorem sequence_arithmetic_mean (a b c d e f g : ℝ)
  (h1 : b = (a + c) / 2)
  (h2 : c = (b + d) / 2)
  (h3 : d = (c + e) / 2)
  (h4 : e = (d + f) / 2)
  (h5 : f = (e + g) / 2) :
  d = (a + g) / 2 :=
sorry

end sequence_arithmetic_mean_l80_80629


namespace probability_at_least_one_unqualified_l80_80953

theorem probability_at_least_one_unqualified :
  let total_products := 6
  let qualified_products := 4
  let unqualified_products := 2
  let products_selected := 2
  (1 - (Nat.choose qualified_products 2 / Nat.choose total_products 2)) = 3/5 :=
by
  sorry

end probability_at_least_one_unqualified_l80_80953


namespace sum_of_squares_of_roots_l80_80844

theorem sum_of_squares_of_roots :
  ∀ (x₁ x₂ : ℝ), (∀ a b c : ℝ, (a ≠ 0) →
  6 * x₁ ^ 2 + 5 * x₁ - 4 = 0 ∧ 6 * x₂ ^ 2 + 5 * x₂ - 4 = 0 →
  x₁ ^ 2 + x₂ ^ 2 = 73 / 36) :=
by
  sorry

end sum_of_squares_of_roots_l80_80844


namespace polar_bear_daily_fish_intake_l80_80172

theorem polar_bear_daily_fish_intake : 
  (0.2 + 0.4 = 0.6) := by
  sorry

end polar_bear_daily_fish_intake_l80_80172


namespace find_a_range_l80_80630

def f (a x : ℝ) : ℝ := x^2 + a * x

theorem find_a_range (a : ℝ) :
  (∃ x : ℝ, f a (f a x) ≤ f a x) → (a ≤ 0 ∨ a ≥ 2) :=
by
  sorry

end find_a_range_l80_80630


namespace largest_even_k_for_sum_of_consecutive_integers_l80_80536

theorem largest_even_k_for_sum_of_consecutive_integers (k n : ℕ) (h_k_even : k % 2 = 0) :
  (3^10 = k * (2 * n + k + 1)) → k ≤ 162 :=
sorry

end largest_even_k_for_sum_of_consecutive_integers_l80_80536


namespace percent_increase_quarter_l80_80705

-- Define the profit changes over each month
def profit_march (P : ℝ) := P
def profit_april (P : ℝ) := 1.40 * P
def profit_may (P : ℝ) := 1.12 * P
def profit_june (P : ℝ) := 1.68 * P

-- Starting Lean theorem statement
theorem percent_increase_quarter (P : ℝ) (hP : P > 0) :
  ((profit_june P - profit_march P) / profit_march P) * 100 = 68 :=
  sorry

end percent_increase_quarter_l80_80705


namespace daily_wage_of_c_is_71_l80_80387

theorem daily_wage_of_c_is_71 (x : ℚ) :
  let a_days := 16
  let b_days := 9
  let c_days := 4
  let total_earnings := 1480
  let wage_ratio_a := 3
  let wage_ratio_b := 4
  let wage_ratio_c := 5
  let total_contribution := a_days * wage_ratio_a * x + b_days * wage_ratio_b * x + c_days * wage_ratio_c * x
  total_contribution = total_earnings →
  c_days * wage_ratio_c * x = 71 := by
  sorry

end daily_wage_of_c_is_71_l80_80387


namespace distinct_triangles_count_l80_80691

def num_points : ℕ := 8
def num_rows : ℕ := 2
def num_cols : ℕ := 4

-- Define the number of ways to choose 3 points from the 8 available points.
def combinations (n k : ℕ) := Nat.choose n k
def total_combinations := combinations num_points 3

-- Define the number of degenerate cases of collinear points in columns.
def degenerate_cases_per_column := combinations num_cols 3
def total_degenerate_cases := num_cols * degenerate_cases_per_column

-- The number of distinct triangles is the total combinations minus the degenerate cases.
def distinct_triangles := total_combinations - total_degenerate_cases

theorem distinct_triangles_count : distinct_triangles = 40 := by
  -- the proof goes here
  sorry

end distinct_triangles_count_l80_80691


namespace water_added_l80_80004

theorem water_added (W : ℝ) : 
  (15 + W) * 0.20833333333333336 = 3.75 → W = 3 :=
by
  intro h
  sorry

end water_added_l80_80004


namespace sqrt_operations_correctness_l80_80700

open Real

theorem sqrt_operations_correctness :
  (sqrt 2 + sqrt 3 ≠ sqrt 5) ∧
  (sqrt (2/3) * sqrt 6 = 2) ∧
  (sqrt 9 = 3) ∧
  (sqrt ((-6) ^ 2) = 6) :=
by
  sorry

end sqrt_operations_correctness_l80_80700


namespace number_of_outfits_l80_80435

def red_shirts : ℕ := 6
def green_shirts : ℕ := 7
def number_pants : ℕ := 9
def blue_hats : ℕ := 10
def red_hats : ℕ := 10

theorem number_of_outfits :
  (red_shirts * number_pants * blue_hats) + (green_shirts * number_pants * red_hats) = 1170 :=
by
  sorry

end number_of_outfits_l80_80435


namespace equate_operations_l80_80423

theorem equate_operations :
  (15 * 5) / (10 + 2) = 3 → 8 / 4 = 2 → ((18 * 6) / (14 + 4) = 6) :=
by
sorry

end equate_operations_l80_80423


namespace intersection_complement_l80_80986

def U : Set ℕ := {0, 1, 2, 4, 6, 8}
def M : Set ℕ := {0, 4, 6}
def N : Set ℕ := {0, 1, 6}
def compl_U_N : Set ℕ := {x ∈ U | x ∉ N}

theorem intersection_complement :
  M ∩ compl_U_N = {4} :=
by
  have h1 : compl_U_N = {2, 4, 8} := by sorry
  have h2 : M ∩ compl_U_N = {4} := by sorry
  exact h2

end intersection_complement_l80_80986


namespace value_less_than_mean_by_std_dev_l80_80312

theorem value_less_than_mean_by_std_dev :
  ∀ (mean value std_dev : ℝ), mean = 16.2 → std_dev = 2.3 → value = 11.6 → 
  (mean - value) / std_dev = 2 :=
by
  intros mean value std_dev h_mean h_std_dev h_value
  -- The proof goes here, but per instructions, it is skipped
  -- So we put 'sorry' to indicate that the proof is intentionally left incomplete
  sorry

end value_less_than_mean_by_std_dev_l80_80312


namespace expectation_of_binomial_l80_80731

noncomputable def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p

theorem expectation_of_binomial :
  binomial_expectation 6 (1/3) = 2 :=
by
  sorry

end expectation_of_binomial_l80_80731


namespace number_divided_is_144_l80_80848

theorem number_divided_is_144 (n divisor quotient remainder : ℕ) (h_divisor : divisor = 11) (h_quotient : quotient = 13) (h_remainder : remainder = 1) (h_division : n = (divisor * quotient) + remainder) : n = 144 :=
by
  sorry

end number_divided_is_144_l80_80848


namespace positive_integer_solutions_count_l80_80955

theorem positive_integer_solutions_count : 
  (∃! (n : ℕ), n > 0 ∧ 25 - 5 * n > 15) :=
sorry

end positive_integer_solutions_count_l80_80955


namespace range_of_f_l80_80934

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem range_of_f :
  ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f x ∈ Set.Icc (1 : ℝ) (Real.sqrt 2) := 
by
  intro x hx
  rw [Set.mem_Icc] at hx
  have : ∀ x ∈ Set.Icc (Real.pi / 2) (3 * Real.pi / 4), f x ∈ Set.Icc 1 (Real.sqrt 2) := sorry
  exact this x hx

end range_of_f_l80_80934


namespace factorize_expression_l80_80115

theorem factorize_expression (m : ℝ) : m^3 - 4 * m^2 + 4 * m = m * (m - 2)^2 :=
by
  sorry

end factorize_expression_l80_80115


namespace two_a_minus_five_d_eq_zero_l80_80833

variables {α : Type*} [Field α]

def f (a b c d x : α) : α :=
  (2*a*x + 3*b) / (4*c*x - 5*d)

theorem two_a_minus_five_d_eq_zero
  (a b c d : α) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (hf : ∀ x, f a b c d (f a b c d x) = x) :
  2*a - 5*d = 0 :=
sorry

end two_a_minus_five_d_eq_zero_l80_80833


namespace problem_statement_l80_80610

-- Define the problem parameters with the constraints
def numberOfWaysToDistributeBalls (totalBalls : Nat) (initialDistribution : List Nat) : Nat :=
  -- Compute the number of remaining balls after the initial distribution
  let remainingBalls := totalBalls - initialDistribution.foldl (· + ·) 0
  -- Use the stars and bars formula to compute the number of ways to distribute remaining balls
  Nat.choose (remainingBalls + initialDistribution.length - 1) (initialDistribution.length - 1)

-- The boxes are to be numbered 1, 2, and 3, and each must contain at least its number of balls
def answer : Nat := numberOfWaysToDistributeBalls 9 [1, 2, 3]

-- Statement of the theorem
theorem problem_statement : answer = 10 := by
  sorry

end problem_statement_l80_80610


namespace solve_ab_c_eq_l80_80670

theorem solve_ab_c_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_eq : 11^a + 3^b = c^2) :
  a = 4 ∧ b = 5 ∧ c = 122 :=
by
  sorry

end solve_ab_c_eq_l80_80670


namespace sequence_26th_term_l80_80941

theorem sequence_26th_term (a d : ℕ) (n : ℕ) (h_a : a = 4) (h_d : d = 3) (h_n : n = 26) :
  a + (n - 1) * d = 79 :=
by
  sorry

end sequence_26th_term_l80_80941


namespace problem1_range_of_f_problem2_range_of_m_l80_80839

noncomputable def f (x : ℝ) : ℝ :=
  (Real.log x / Real.log 2 - 2) * (Real.log x / Real.log 4 - 1/2)

theorem problem1_range_of_f :
  Set.range (fun x => f x) ∩ Set.Icc 1 4 = Set.Icc (-1/8 : ℝ) 1 :=
sorry

theorem problem2_range_of_m :
  ∀ x, x ∈ Set.Icc 4 16 → f x > (m : ℝ) * (Real.log x / Real.log 4) ↔ m < 0 :=
sorry

end problem1_range_of_f_problem2_range_of_m_l80_80839


namespace hyperbola_focus_and_asymptotes_l80_80609

def is_focus_on_y_axis (a b : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
∃ c : ℝ, eq (c^2 * a) (c^2 * b)

def are_asymptotes_perpendicular (eq : ℝ → ℝ → Prop) : Prop :=
∃ k1 k2 : ℝ, (k1 != 0 ∧ k2 != 0 ∧ eq k1 k2 ∧ eq (-k1) k2)

theorem hyperbola_focus_and_asymptotes :
  is_focus_on_y_axis 1 (-1) (fun y x => y^2 - x^2 = 4) ∧ are_asymptotes_perpendicular (fun y x => y = x) :=
by
  sorry

end hyperbola_focus_and_asymptotes_l80_80609


namespace remainder_div_29_l80_80130

theorem remainder_div_29 (k : ℤ) (N : ℤ) (h : N = 899 * k + 63) : N % 29 = 10 :=
  sorry

end remainder_div_29_l80_80130


namespace cost_price_per_meter_l80_80500

theorem cost_price_per_meter (total_cost : ℝ) (total_length : ℝ) (h1 : total_cost = 397.75) (h2 : total_length = 9.25) : total_cost / total_length = 43 :=
by
  -- Proof omitted
  sorry

end cost_price_per_meter_l80_80500


namespace solution_set_l80_80069

noncomputable def f : ℝ → ℝ := sorry

-- The function f is defined to be odd.
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- The function f is increasing on (-∞, 0).
axiom increasing_f : ∀ x y : ℝ, x < y ∧ y < 0 → f x < f y

-- Given f(2) = 0
axiom f_at_2 : f 2 = 0

-- Prove the solution set for x f(x + 1) < 0
theorem solution_set : { x : ℝ | x * f (x + 1) < 0 } = {x : ℝ | (-3 < x ∧ x < -1) ∨ (0 < x ∧ x < 1)} :=
by
  sorry

end solution_set_l80_80069


namespace instantaneous_velocity_at_3_l80_80285

-- Define the displacement function
def displacement (t : ℝ) : ℝ := t^2 - t

-- State the main theorem that we need to prove
theorem instantaneous_velocity_at_3 : (deriv displacement 3 = 5) := by
  sorry

end instantaneous_velocity_at_3_l80_80285


namespace no_such_pairs_l80_80233

theorem no_such_pairs :
  ¬ ∃ (b c : ℕ), b > 0 ∧ c > 0 ∧ (b^2 - 4 * c < 0) ∧ (c^2 - 4 * b < 0) := sorry

end no_such_pairs_l80_80233


namespace milk_jars_good_for_sale_l80_80835

noncomputable def good_whole_milk_jars : ℕ := 
  let initial_jars := 60 * 30
  let short_deliveries := 20 * 30 * 2
  let damaged_jars_1 := 3 * 5
  let damaged_jars_2 := 4 * 6
  let totally_damaged_cartons := 2 * 30
  let received_jars := initial_jars - short_deliveries - damaged_jars_1 - damaged_jars_2 - totally_damaged_cartons
  let spoilage := (5 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_skim_milk_jars : ℕ := 
  let initial_jars := 40 * 40
  let short_delivery := 10 * 40
  let damaged_jars := 5 * 4
  let totally_damaged_carton := 1 * 40
  let received_jars := initial_jars - short_delivery - damaged_jars - totally_damaged_carton
  let spoilage := (3 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_almond_milk_jars : ℕ := 
  let initial_jars := 30 * 20
  let short_delivery := 5 * 20
  let damaged_jars := 2 * 3
  let received_jars := initial_jars - short_delivery - damaged_jars
  let spoilage := (1 * received_jars) / 100
  received_jars - spoilage

theorem milk_jars_good_for_sale : 
  good_whole_milk_jars = 476 ∧
  good_skim_milk_jars = 1106 ∧
  good_almond_milk_jars = 489 :=
by
  sorry

end milk_jars_good_for_sale_l80_80835


namespace smallest_A_divided_by_6_has_third_of_original_factors_l80_80627

theorem smallest_A_divided_by_6_has_third_of_original_factors:
  ∃ A: ℕ, A > 0 ∧ (∃ a b: ℕ, A = 2^a * 3^b ∧ (a + 1) * (b + 1) = 3 * a * b) ∧ A = 12 :=
by
  sorry

end smallest_A_divided_by_6_has_third_of_original_factors_l80_80627


namespace smallest_value_A_B_C_D_l80_80088

theorem smallest_value_A_B_C_D :
  ∃ (A B C D : ℕ), 
  (A < B) ∧ (B < C) ∧ (C < D) ∧ -- A, B, C are in arithmetic sequence and B, C, D in geometric sequence
  (C = B + (B - A)) ∧  -- A, B, C form an arithmetic sequence with common difference d = B - A
  (C = (4 * B) / 3) ∧  -- Given condition
  (D = (4 * C) / 3) ∧ -- B, C, D form geometric sequence with common ratio 4/3
  ((∃ k, D = k * 9) ∧ -- D must be an integer, ensuring B must be divisible by 9
   A + B + C + D = 43) := 
sorry

end smallest_value_A_B_C_D_l80_80088


namespace eval_polynomial_at_2_l80_80612

theorem eval_polynomial_at_2 : 
  ∃ a b c d : ℝ, (∀ x : ℝ, (3 * x^2 - 5 * x + 4) * (7 - 2 * x) = a * x^3 + b * x^2 + c * x + d) ∧ (8 * a + 4 * b + 2 * c + d = 18) :=
by
  sorry

end eval_polynomial_at_2_l80_80612


namespace F_minimum_value_neg_inf_to_0_l80_80940

variable (f g : ℝ → ℝ)

def is_odd (h : ℝ → ℝ) := ∀ x, h (-x) = - (h x)

theorem F_minimum_value_neg_inf_to_0 
  (hf_odd : is_odd f) 
  (hg_odd : is_odd g)
  (hF_max : ∀ x > 0, f x + g x + 2 ≤ 8) 
  (hF_reaches_max : ∃ x > 0, f x + g x + 2 = 8) :
  ∀ x < 0, f x + g x + 2 ≥ -4 :=
by
  sorry

end F_minimum_value_neg_inf_to_0_l80_80940


namespace obtain_2001_from_22_l80_80791

theorem obtain_2001_from_22 :
  ∃ (f : ℕ → ℕ), (∀ n, f (n + 1) = n ∨ f (n) = n + 1) ∧ (f 22 = 2001) := 
sorry

end obtain_2001_from_22_l80_80791


namespace total_votes_is_5000_l80_80947

theorem total_votes_is_5000 :
  ∃ (V : ℝ), 0.45 * V - 0.35 * V = 500 ∧ 0.35 * V - 0.20 * V = 350 ∧ V = 5000 :=
by
  sorry

end total_votes_is_5000_l80_80947


namespace problem1_problem2_l80_80107

-- Problem 1
theorem problem1 (x y : ℤ) (h1 : x = 2) (h2 : y = 2016) :
  (3*x + 2*y)*(3*x - 2*y) - (x + 2*y)*(5*x - 2*y) / (8*x) = -2015 :=
by
  sorry

-- Problem 2
theorem problem2 (x : ℤ) (h1 : x = 2) :
  ((x - 3) / (x^2 - 1)) * ((x^2 + 2*x + 1) / (x - 3)) - (1 / (x - 1) + 1) = 1 :=
by
  sorry

end problem1_problem2_l80_80107


namespace solve_for_y_l80_80787

theorem solve_for_y (x y : ℝ) (h1 : 2 * x - y = 10) (h2 : x + 3 * y = 2) : y = -6 / 7 := 
by
  sorry

end solve_for_y_l80_80787


namespace mario_age_is_4_l80_80121

-- Define the conditions
def sum_of_ages (mario maria : ℕ) : Prop := mario + maria = 7
def mario_older_by_one (mario maria : ℕ) : Prop := mario = maria + 1

-- State the theorem to prove Mario's age is 4 given the conditions
theorem mario_age_is_4 (mario maria : ℕ) (h1 : sum_of_ages mario maria) (h2 : mario_older_by_one mario maria) : mario = 4 :=
sorry -- Proof to be completed later

end mario_age_is_4_l80_80121


namespace geometric_sequence_third_term_l80_80660

theorem geometric_sequence_third_term (a1 a5 a3 : ℕ) (r : ℝ) 
  (h1 : a1 = 4) 
  (h2 : a5 = 1296) 
  (h3 : a5 = a1 * r^4)
  (h4 : a3 = a1 * r^2) : 
  a3 = 36 := 
by 
  sorry

end geometric_sequence_third_term_l80_80660


namespace problem_statement_l80_80043

-- Definition of operation nabla
def nabla (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2

-- Main theorem statement
theorem problem_statement : nabla 2 (nabla 0 (nabla 1 7)) = 71859 :=
by
  -- Computational proof
  sorry

end problem_statement_l80_80043


namespace find_w_when_x_is_six_l80_80096

variable {x w : ℝ}
variable (h1 : x = 3)
variable (h2 : w = 16)
variable (h3 : ∀ (x w : ℝ), x^4 * w^(1 / 4) = 162)

theorem find_w_when_x_is_six : x = 6 → w = 1 / 4096 :=
by
  intro hx
  sorry

end find_w_when_x_is_six_l80_80096


namespace raisins_in_other_three_boxes_l80_80405

-- Definitions of the known quantities
def total_raisins : ℕ := 437
def box1_raisins : ℕ := 72
def box2_raisins : ℕ := 74

-- The goal is to prove that each of the other three boxes has 97 raisins
theorem raisins_in_other_three_boxes :
  total_raisins - (box1_raisins + box2_raisins) = 3 * 97 :=
by
  sorry

end raisins_in_other_three_boxes_l80_80405


namespace modular_inverse_l80_80365

theorem modular_inverse :
  (24 * 22) % 53 = 1 :=
by
  have h1 : (24 * -29) % 53 = (53 * 0 - 29 * 24) % 53 := by sorry
  have h2 : (24 * -29) % 53 = (-29 * 24) % 53 := by sorry
  have h3 : (-29 * 24) % 53 = (-29 % 53 * 24 % 53 % 53) := by sorry
  have h4 : -29 % 53 = 53 - 24 := by sorry
  have h5 : (53 - 29) % 53 = (22 * 22) % 53 := by sorry
  have h6 : (22 * 22) % 53 = (24 * 22) % 53 := by sorry
  have h7 : (24 * 22) % 53 = 1 := by sorry
  exact h7

end modular_inverse_l80_80365


namespace solve_equation_l80_80680

theorem solve_equation :
  ∃! (x y z : ℝ), 2 * x^4 + 2 * y^4 - 4 * x^3 * y + 6 * x^2 * y^2 - 4 * x * y^3 + 7 * y^2 + 7 * z^2 - 14 * y * z - 70 * y + 70 * z + 175 = 0 ∧ x = 0 ∧ y = 0 ∧ z = -5 :=
by
  sorry

end solve_equation_l80_80680


namespace correct_div_value_l80_80029

theorem correct_div_value (x : ℝ) (h : 25 * x = 812) : x / 4 = 8.12 :=
by sorry

end correct_div_value_l80_80029


namespace number_of_single_windows_upstairs_l80_80895

theorem number_of_single_windows_upstairs :
  ∀ (num_double_windows_downstairs : ℕ)
    (glass_panels_per_double_window : ℕ)
    (num_single_windows_upstairs : ℕ)
    (glass_panels_per_single_window : ℕ)
    (total_glass_panels : ℕ),
  num_double_windows_downstairs = 6 →
  glass_panels_per_double_window = 4 →
  glass_panels_per_single_window = 4 →
  total_glass_panels = 80 →
  num_single_windows_upstairs = (total_glass_panels - (num_double_windows_downstairs * glass_panels_per_double_window)) / glass_panels_per_single_window →
  num_single_windows_upstairs = 14 :=
by
  intros
  sorry

end number_of_single_windows_upstairs_l80_80895


namespace minimum_value_of_expression_l80_80628

noncomputable def minimum_value_expression (x y z : ℝ) : ℝ :=
  1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z))

theorem minimum_value_of_expression : ∀ (x y z : ℝ), -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ -1 < z ∧ z < 0 → 
  minimum_value_expression x y z ≥ 2 := 
by
  intro x y z h
  sorry

end minimum_value_of_expression_l80_80628


namespace degrees_to_radians_l80_80128

theorem degrees_to_radians : (800 : ℝ) * (Real.pi / 180) = (40 / 9) * Real.pi :=
by
  sorry

end degrees_to_radians_l80_80128


namespace triangle_perimeter_l80_80170

theorem triangle_perimeter
  (a : ℝ) (a_gt_5 : a > 5)
  (ellipse : ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / 25 = 1)
  (dist_foci : 8 = 2 * 4) :
  4 * Real.sqrt (41) = 4 * Real.sqrt (41) := by
sorry

end triangle_perimeter_l80_80170


namespace samantha_interest_l80_80053

-- Definitions based on problem conditions
def P : ℝ := 2000
def r : ℝ := 0.08
def n : ℕ := 5

-- Compound interest calculation
noncomputable def A : ℝ := P * (1 + r) ^ n
noncomputable def Interest : ℝ := A - P

-- Theorem statement with Lean 4
theorem samantha_interest : Interest = 938.656 := 
by 
  sorry

end samantha_interest_l80_80053


namespace total_students_correct_l80_80192

noncomputable def num_roman_numerals : ℕ := 7
noncomputable def sketches_per_numeral : ℕ := 5
noncomputable def total_students : ℕ := 35

theorem total_students_correct : num_roman_numerals * sketches_per_numeral = total_students := by
  sorry

end total_students_correct_l80_80192


namespace exchange_rate_5_CAD_to_JPY_l80_80386

theorem exchange_rate_5_CAD_to_JPY :
  (1 : ℝ) * 85 * 5 = 425 :=
by
  sorry

end exchange_rate_5_CAD_to_JPY_l80_80386


namespace mean_temperature_is_88_75_l80_80304

def temperatures : List ℕ := [85, 84, 85, 88, 91, 93, 94, 90]

theorem mean_temperature_is_88_75 : (List.sum temperatures : ℚ) / temperatures.length = 88.75 := by
  sorry

end mean_temperature_is_88_75_l80_80304


namespace range_of_m_l80_80810

theorem range_of_m (x m : ℝ) (h1: |x - m| < 1) (h2: x^2 - 8 * x + 12 < 0) (h3: ∀ x, (x^2 - 8 * x + 12 < 0) → ((m - 1) < x ∧ x < (m + 1))) : 
  3 ≤ m ∧ m ≤ 5 := 
sorry

end range_of_m_l80_80810


namespace uncool_students_in_two_classes_l80_80475

theorem uncool_students_in_two_classes
  (students_class1 : ℕ)
  (cool_dads_class1 : ℕ)
  (cool_moms_class1 : ℕ)
  (both_cool_class1 : ℕ)
  (students_class2 : ℕ)
  (cool_dads_class2 : ℕ)
  (cool_moms_class2 : ℕ)
  (both_cool_class2 : ℕ)
  (h1 : students_class1 = 45)
  (h2 : cool_dads_class1 = 22)
  (h3 : cool_moms_class1 = 25)
  (h4 : both_cool_class1 = 11)
  (h5 : students_class2 = 35)
  (h6 : cool_dads_class2 = 15)
  (h7 : cool_moms_class2 = 18)
  (h8 : both_cool_class2 = 7) :
  (students_class1 - ((cool_dads_class1 - both_cool_class1) + (cool_moms_class1 - both_cool_class1) + both_cool_class1) +
   students_class2 - ((cool_dads_class2 - both_cool_class2) + (cool_moms_class2 - both_cool_class2) + both_cool_class2)
  ) = 18 :=
sorry

end uncool_students_in_two_classes_l80_80475


namespace residue_of_neg_1235_mod_29_l80_80762

theorem residue_of_neg_1235_mod_29 : 
  ∃ r, 0 ≤ r ∧ r < 29 ∧ (-1235) % 29 = r ∧ r = 12 :=
by
  sorry

end residue_of_neg_1235_mod_29_l80_80762


namespace abs_inequality_holds_l80_80970

theorem abs_inequality_holds (m x : ℝ) (h : -1 ≤ m ∧ m ≤ 6) : 
  |x - 2| + |x + 4| ≥ m^2 - 5 * m :=
sorry

end abs_inequality_holds_l80_80970


namespace characterization_of_M_l80_80868

noncomputable def M : Set ℂ := {z : ℂ | (z - 1) ^ 2 = Complex.abs (z - 1) ^ 2}

theorem characterization_of_M : M = {z : ℂ | ∃ r : ℝ, z = r} :=
by
  sorry

end characterization_of_M_l80_80868


namespace jump_difference_l80_80601

variable (runningRicciana jumpRicciana runningMargarita : ℕ)

theorem jump_difference :
  (runningMargarita + (2 * jumpRicciana - 1)) - (runningRicciana + jumpRicciana) = 1 :=
by
  -- Given conditions
  let runningRicciana := 20
  let jumpRicciana := 4
  let runningMargarita := 18
  -- The proof is omitted (using 'sorry')
  sorry

end jump_difference_l80_80601


namespace find_q_l80_80771

noncomputable def Q (x p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q (p q d : ℝ) (h₁ : -p / 3 = q) (h₂ : q = 1 + p + q + 5) (h₃ : d = 5) : q = 2 :=
by
  sorry

end find_q_l80_80771


namespace max_band_members_l80_80045

variable (r x m : ℕ)

noncomputable def band_formation (r x m: ℕ) :=
  m = r * x + 4 ∧
  m = (r - 3) * (x + 2) ∧
  m < 100

theorem max_band_members (r x m : ℕ) (h : band_formation r x m) : m = 88 :=
by
  sorry

end max_band_members_l80_80045


namespace circus_capacity_l80_80476

theorem circus_capacity (sections : ℕ) (people_per_section : ℕ) (h1 : sections = 4) (h2 : people_per_section = 246) :
  sections * people_per_section = 984 :=
by
  sorry

end circus_capacity_l80_80476


namespace ben_final_amount_l80_80374

-- Definition of the conditions
def daily_start := 50
def daily_spent := 15
def daily_saving := daily_start - daily_spent
def days := 7
def mom_double (s : ℕ) := 2 * s
def dad_addition := 10

-- Total amount calculation based on the conditions
noncomputable def total_savings := daily_saving * days
noncomputable def after_mom := mom_double total_savings
noncomputable def total_amount := after_mom + dad_addition

-- The final theorem to prove Ben's final amount is $500 after the given conditions
theorem ben_final_amount : total_amount = 500 :=
by sorry

end ben_final_amount_l80_80374


namespace major_minor_axis_lengths_foci_vertices_coordinates_l80_80138

-- Given conditions
def ellipse_eq (x y : ℝ) : Prop := 16 * x^2 + 25 * y^2 = 400

-- Proof Tasks
theorem major_minor_axis_lengths : 
  (∃ a b : ℝ, a = 5 ∧ b = 4 ∧ 2 * a = 10) :=
by sorry

theorem foci_vertices_coordinates : 
  (∃ c : ℝ, 
    (c = 3) ∧ 
    (∀ x y : ℝ, ellipse_eq x y → (x = 0 → y = 4 ∨ y = -4) ∧ (y = 0 → x = 5 ∨ x = -5))) :=
by sorry

end major_minor_axis_lengths_foci_vertices_coordinates_l80_80138


namespace three_number_product_l80_80274

theorem three_number_product
  (x y z : ℝ)
  (h1 : x + y = 18)
  (h2 : x ^ 2 + y ^ 2 = 220)
  (h3 : z = x - y) :
  x * y * z = 104 * Real.sqrt 29 :=
sorry

end three_number_product_l80_80274


namespace find_n_l80_80186

theorem find_n (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) (h3 : ∃ k : ℤ, 721 = n + 360 * k): n = 1 :=
sorry

end find_n_l80_80186


namespace sin_330_eq_neg_half_l80_80973

theorem sin_330_eq_neg_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by sorry

end sin_330_eq_neg_half_l80_80973


namespace negate_proposition_l80_80072

theorem negate_proposition :
    (¬ ∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by
  sorry

end negate_proposition_l80_80072


namespace ratio_of_money_with_Ram_and_Gopal_l80_80920

noncomputable section

variable (R K G : ℕ)

theorem ratio_of_money_with_Ram_and_Gopal 
  (hR : R = 735) 
  (hK : K = 4335) 
  (hRatio : G * 17 = 7 * K) 
  (hGCD : Nat.gcd 735 1785 = 105) :
  R * 17 = 7 * G := 
by
  sorry

end ratio_of_money_with_Ram_and_Gopal_l80_80920


namespace fatima_total_donation_l80_80173

theorem fatima_total_donation :
  let cloth1 := 100
  let cloth1_piece1 := 0.40 * cloth1
  let cloth1_piece2 := 0.30 * cloth1
  let cloth1_piece3 := 0.30 * cloth1
  let donation1 := cloth1_piece2 + cloth1_piece3

  let cloth2 := 65
  let cloth2_piece1 := 0.55 * cloth2
  let cloth2_piece2 := 0.45 * cloth2
  let donation2 := cloth2_piece2

  let cloth3 := 48
  let cloth3_piece1 := 0.60 * cloth3
  let cloth3_piece2 := 0.40 * cloth3
  let donation3 := cloth3_piece2

  donation1 + donation2 + donation3 = 108.45 :=
by
  sorry

end fatima_total_donation_l80_80173


namespace contribution_of_eight_families_l80_80267

/-- Definition of the given conditions --/
def classroom := 200
def two_families := 2 * 20
def ten_families := 10 * 5
def missing_amount := 30

def total_raised (x : ℝ) : ℝ := two_families + ten_families + 8 * x

/-- The main theorem to prove the contribution of each of the eight families --/
theorem contribution_of_eight_families (x : ℝ) (h : total_raised x = classroom - missing_amount) : x = 10 := by
  sorry

end contribution_of_eight_families_l80_80267


namespace total_flowers_in_vases_l80_80684

theorem total_flowers_in_vases :
  let vase_count := 5
  let flowers_per_vase_4 := 5
  let flowers_per_vase_1 := 6
  let vases_with_5_flowers := 4
  let vases_with_6_flowers := 1
  (4 * 5 + 1 * 6 = 26) := by
  let total_flowers := 4 * 5 + 1 * 6
  show total_flowers = 26
  sorry

end total_flowers_in_vases_l80_80684


namespace room_total_space_l80_80228

-- Definitions based on the conditions
def bookshelf_space : ℕ := 80
def reserved_space : ℕ := 160
def number_of_shelves : ℕ := 3

-- The theorem statement
theorem room_total_space : 
  (number_of_shelves * bookshelf_space) + reserved_space = 400 := 
by
  sorry

end room_total_space_l80_80228


namespace solve_equation_l80_80806

theorem solve_equation : ∀ x : ℝ, 4 * x - 2 * x + 1 - 3 = 0 → x = 1 :=
by
  intro x
  intro h
  sorry

end solve_equation_l80_80806


namespace remaining_days_to_complete_job_l80_80535

-- Define the given conditions
def in_10_days (part_of_job_done : ℝ) (days : ℕ) : Prop :=
  part_of_job_done = 1 / 8 ∧ days = 10

-- Define the complete job condition
def complete_job (total_days : ℕ) : Prop :=
  total_days = 80

-- Define the remaining days to finish the job
def remaining_days (total_days_worked : ℕ) (days_worked : ℕ) (remaining : ℕ) : Prop :=
  total_days_worked = 80 ∧ days_worked = 10 ∧ remaining = 70

-- The theorem statement
theorem remaining_days_to_complete_job (part_of_job_done : ℝ) (days : ℕ) (total_days : ℕ) (total_days_worked : ℕ) (days_worked : ℕ) (remaining : ℕ) :
  in_10_days part_of_job_done days → complete_job total_days → remaining_days total_days_worked days_worked remaining :=
sorry

end remaining_days_to_complete_job_l80_80535


namespace solve_for_x_l80_80105

theorem solve_for_x (x : ℝ) (d : ℝ) (h1 : x > 0) (h2 : x^2 = 4 + d) (h3 : 25 = x^2 + d) : x = Real.sqrt 14.5 := 
by 
  sorry

end solve_for_x_l80_80105


namespace original_ticket_price_l80_80883

open Real

theorem original_ticket_price 
  (P : ℝ)
  (total_revenue : ℝ)
  (revenue_equation : total_revenue = 10 * 0.60 * P + 20 * 0.85 * P + 15 * P) 
  (total_revenue_val : total_revenue = 760) : 
  P = 20 := 
by
  sorry

end original_ticket_price_l80_80883


namespace polygon_interior_exterior_relation_l80_80897

theorem polygon_interior_exterior_relation :
  ∃ (n : ℕ), (n > 2) ∧ ((n - 2) * 180 = 4 * 360) ∧ n = 10 :=
by
  sorry

end polygon_interior_exterior_relation_l80_80897


namespace tom_age_l80_80269

theorem tom_age (c : ℕ) (h1 : 2 * c - 1 = tom) (h2 : c + 3 = dave) (h3 : c + (2 * c - 1) + (c + 3) = 30) : tom = 13 :=
  sorry

end tom_age_l80_80269


namespace loss_percentage_l80_80764

theorem loss_percentage (CP SP SP_new : ℝ) (L : ℝ) 
  (h1 : CP = 1428.57)
  (h2 : SP = CP - (L / 100 * CP))
  (h3 : SP_new = CP + 0.04 * CP)
  (h4 : SP_new = SP + 200) :
  L = 10 := by
    sorry

end loss_percentage_l80_80764


namespace required_HCl_moles_l80_80538

-- Definitions of chemical substances:
def HCl: Type := Unit
def NaHCO3: Type := Unit
def H2O: Type := Unit
def CO2: Type := Unit
def NaCl: Type := Unit

-- The reaction as a balanced chemical equation:
def balanced_eq (hcl: HCl) (nahco3: NaHCO3) (h2o: H2O) (co2: CO2) (nacl: NaCl) : Prop :=
  ∃ (m: ℕ), m = 1

-- Given conditions:
def condition1: Prop := balanced_eq () () () () ()
def condition2 (moles_H2O moles_CO2 moles_NaCl: ℕ): Prop :=
  moles_H2O = moles_CO2 ∧ moles_CO2 = moles_NaCl ∧ moles_NaCl = moles_H2O

def condition3: ℕ := 3  -- moles of NaHCO3

-- The theorem statement:
theorem required_HCl_moles (moles_HCl moles_NaHCO3: ℕ)
  (hcl: HCl) (nahco3: NaHCO3) (h2o: H2O) (co2: CO2) (nacl: NaCl)
  (balanced: balanced_eq hcl nahco3 h2o co2 nacl)
  (equal_moles: condition2 moles_H2O moles_CO2 moles_NaCl)
  (nahco3_eq_3: moles_NaHCO3 = condition3):
  moles_HCl = 3 :=
sorry

end required_HCl_moles_l80_80538


namespace find_a_for_even_function_l80_80230

theorem find_a_for_even_function (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x, f x = f (-x)) 
  (h_neg : ∀ x, x < 0 → f x = x^2 + a * x) 
  (h_value : f 3 = 3) : a = 2 :=
sorry

end find_a_for_even_function_l80_80230


namespace simplify_expression_l80_80908

variable {x y : ℝ}
variable (h : x * y ≠ 0)

theorem simplify_expression (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^2 + 1) / y) - ((x^2 - 1) / y) * ((y^3 - 1) / x) =
  (x^3*y^2 - x^2*y^3 + x^3 + x^2 + y^2 + y^3) / (x*y) :=
by sorry

end simplify_expression_l80_80908


namespace slant_asymptote_sum_l80_80254

theorem slant_asymptote_sum (x : ℝ) (hx : x ≠ 5) :
  (5 : ℝ) + (21 : ℝ) = 26 :=
by
  sorry

end slant_asymptote_sum_l80_80254


namespace find_p_l80_80616

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_p (p : ℕ) (h : is_prime p) (hpgt1 : 1 < p) :
  8 * p^4 - 3003 = 1997 ↔ p = 5 :=
by
  sorry

end find_p_l80_80616


namespace intersection_eq_l80_80995

noncomputable def A : Set ℕ := {1, 2, 3, 4}
noncomputable def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_eq : A ∩ B = {2, 3, 4} := 
by
  sorry

end intersection_eq_l80_80995


namespace initial_capacity_of_drum_x_l80_80485

theorem initial_capacity_of_drum_x (C x : ℝ) (h_capacity_y : 2 * x = 2 * 0.75 * C) :
  x = 0.75 * C :=
sorry

end initial_capacity_of_drum_x_l80_80485


namespace point_bisector_second_quadrant_l80_80857

theorem point_bisector_second_quadrant (a : ℝ) : 
  (a < 0 ∧ 2 > 0) ∧ (2 = -a) → a = -2 :=
by sorry

end point_bisector_second_quadrant_l80_80857


namespace max_green_beads_l80_80232

theorem max_green_beads (n : ℕ) (red blue green : ℕ) 
    (total_beads : ℕ)
    (h_total : total_beads = 100)
    (h_colors : n = red + blue + green)
    (h_blue_condition : ∀ i : ℕ, i ≤ total_beads → ∃ j, j ≤ 4 ∧ (i + j) % total_beads = blue)
    (h_red_condition : ∀ i : ℕ, i ≤ total_beads → ∃ k, k ≤ 6 ∧ (i + k) % total_beads = red) :
    green ≤ 65 :=
by
  sorry

end max_green_beads_l80_80232


namespace explicit_formula_is_even_tangent_line_at_1_tangent_line_equation_l80_80453

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 + (2 - a) * x + (a - 1)

-- Proof needed for the first question:
theorem explicit_formula_is_even (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) → a = 2 ∧ ∀ x : ℝ, f x a = x^2 + 1 :=
by sorry

-- Proof needed for the second question:
theorem tangent_line_at_1 (f : ℝ → ℝ) : (∀ x : ℝ, f x = x^2 + 1) → ∀ x : ℝ, deriv f 1 = 2 :=
by sorry

-- The tangent line equation at x = 1 in the required form
theorem tangent_line_equation (f : ℝ → ℝ) : (∀ x : ℝ, f x = x^2 + 1) → ∀ x : ℝ, deriv f 1 = 2 → (f 1 - deriv f 1 * 1 + deriv f 1 * x = 2 * x) :=
by sorry

end explicit_formula_is_even_tangent_line_at_1_tangent_line_equation_l80_80453


namespace pile_limit_exists_l80_80819

noncomputable def log_floor (b x : ℝ) : ℤ :=
  Int.floor (Real.log x / Real.log b)

theorem pile_limit_exists (k : ℝ) (hk : k < 2) : ∃ Nk : ℤ, 
  Nk = 2 * (log_floor (2 / k) 2 + 1) := 
  by
    sorry

end pile_limit_exists_l80_80819


namespace parallel_line_segment_length_l80_80468

theorem parallel_line_segment_length (AB : ℝ) (S : ℝ) (x : ℝ) 
  (h1 : AB = 36) 
  (h2 : S = (S / 2) * 2)
  (h3 : x / AB = (↑(1 : ℝ) / 2 * S / S) ^ (1 / 2)) : 
  x = 18 * Real.sqrt 2 :=
by 
    sorry 

end parallel_line_segment_length_l80_80468


namespace fish_to_apples_l80_80603

variables (f l r a : ℝ)

theorem fish_to_apples (h1 : 3 * f = 2 * l) (h2 : l = 5 * r) (h3 : l = 3 * a) : f = 2 * a :=
by
  -- We assume the conditions as hypotheses and aim to prove the final statement
  sorry

end fish_to_apples_l80_80603


namespace loss_calculation_l80_80809

-- Given conditions: 
-- The ratio of the amount of money Cara, Janet, and Jerry have is 4:5:6
-- The total amount of money they have is $75

theorem loss_calculation :
  let cara_ratio := 4
  let janet_ratio := 5
  let jerry_ratio := 6
  let total_ratio := cara_ratio + janet_ratio + jerry_ratio
  let total_money := 75
  let part_value := total_money / total_ratio
  let cara_money := cara_ratio * part_value
  let janet_money := janet_ratio * part_value
  let combined_money := cara_money + janet_money
  let selling_price := 0.80 * combined_money
  combined_money - selling_price = 9 :=
by
  sorry

end loss_calculation_l80_80809


namespace angle_between_NE_and_SW_l80_80917

theorem angle_between_NE_and_SW
  (n : ℕ) (hn : n = 12)
  (total_degrees : ℚ) (htotal : total_degrees = 360)
  (spaced_rays : ℚ) (hspaced : spaced_rays = total_degrees / n)
  (angles_between_NE_SW : ℕ) (hangles : angles_between_NE_SW = 4) :
  (angles_between_NE_SW * spaced_rays = 120) :=
by
  rw [htotal, hn] at hspaced
  rw [hangles]
  rw [hspaced]
  sorry

end angle_between_NE_and_SW_l80_80917


namespace polynomial_coeff_sum_abs_l80_80963

theorem polynomial_coeff_sum_abs (a a_1 a_2 a_3 a_4 a_5 : ℤ) (x : ℤ) 
  (h : (2*x - 1)^5 + (x + 2)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) :
  |a| + |a_2| + |a_4| = 30 :=
sorry

end polynomial_coeff_sum_abs_l80_80963


namespace donna_total_episodes_per_week_l80_80729

-- Defining the conditions
def episodes_per_weekday : ℕ := 8
def weekday_count : ℕ := 5
def weekend_factor : ℕ := 3
def weekend_count : ℕ := 2

-- Theorem statement
theorem donna_total_episodes_per_week :
  (episodes_per_weekday * weekday_count) + ((episodes_per_weekday * weekend_factor) * weekend_count) = 88 := 
  by sorry

end donna_total_episodes_per_week_l80_80729


namespace avg_annual_growth_rate_optimal_selling_price_l80_80875

theorem avg_annual_growth_rate (v2022 v2024 : ℕ) (x : ℝ) 
  (h1 : v2022 = 200000) 
  (h2 : v2024 = 288000)
  (h3: v2024 = v2022 * (1 + x)^2) :
  x = 0.2 :=
by
  sorry

theorem optimal_selling_price (cost : ℝ) (initial_price : ℝ) (initial_cups : ℕ) 
  (price_drop_effect : ℝ) (initial_profit : ℝ) (daily_profit : ℕ) (y : ℝ)
  (h1 : cost = 6)
  (h2 : initial_price = 25) 
  (h3 : initial_cups = 300)
  (h4 : price_drop_effect = 1)
  (h5 : initial_profit = 6300)
  (h6 : (y - cost) * (initial_cups + 30 * (initial_price - y)) = daily_profit) :
  y = 20 :=
by
  sorry

end avg_annual_growth_rate_optimal_selling_price_l80_80875


namespace consecutive_sum_l80_80256

theorem consecutive_sum (m k : ℕ) (h : (k + 1) * (2 * m + k) = 2000) :
  (m = 1000 ∧ k = 0) ∨ 
  (m = 198 ∧ k = 4) ∨ 
  (m = 28 ∧ k = 24) ∨ 
  (m = 55 ∧ k = 15) :=
by sorry

end consecutive_sum_l80_80256


namespace sum_of_a_for_repeated_root_l80_80255

theorem sum_of_a_for_repeated_root :
  ∀ a : ℝ, (∀ x : ℝ, 2 * x^2 + a * x + 10 * x + 16 = 0 → 
               (a + 10 = 8 * Real.sqrt 2 ∨ a + 10 = -8 * Real.sqrt 2)) → 
               (a = -10 + 8 * Real.sqrt 2 ∨ a = -10 - 8 * Real.sqrt 2) → 
               ((-10 + 8 * Real.sqrt 2) + (-10 - 8 * Real.sqrt 2) = -20) := by
sorry

end sum_of_a_for_repeated_root_l80_80255


namespace exterior_angle_BAC_eq_162_l80_80983

noncomputable def measure_of_angle_BAC : ℝ := 360 - 108 - 90

theorem exterior_angle_BAC_eq_162 :
  measure_of_angle_BAC = 162 := by
  sorry

end exterior_angle_BAC_eq_162_l80_80983


namespace median_production_l80_80864

def production_data : List ℕ := [5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10]

def median (l : List ℕ) : ℕ :=
  if l.length % 2 = 1 then
    l.nthLe (l.length / 2) sorry
  else
    let m := l.length / 2
    (l.nthLe (m - 1) sorry + l.nthLe m sorry) / 2

theorem median_production :
  median (production_data) = 8 :=
by
  sorry

end median_production_l80_80864


namespace percentage_of_valid_votes_l80_80099

theorem percentage_of_valid_votes 
  (total_votes : ℕ) 
  (invalid_percentage : ℕ) 
  (candidate_valid_votes : ℕ)
  (percentage_invalid : invalid_percentage = 15)
  (total_votes_eq : total_votes = 560000)
  (candidate_votes_eq : candidate_valid_votes = 380800) 
  : (candidate_valid_votes : ℝ) / (total_votes * (0.85 : ℝ)) * 100 = 80 := 
by 
  sorry

end percentage_of_valid_votes_l80_80099


namespace milk_in_jugs_l80_80458

theorem milk_in_jugs (x y : ℝ) (h1 : x + y = 70) (h2 : y + 0.125 * x = 0.875 * x) :
  x = 40 ∧ y = 30 := 
sorry

end milk_in_jugs_l80_80458


namespace triangle_sum_l80_80876

-- Define the triangle operation
def triangle (a b c : ℕ) : ℕ := a + b + c

-- State the theorem
theorem triangle_sum :
  triangle 2 4 3 + triangle 1 6 5 = 21 :=
by
  sorry

end triangle_sum_l80_80876


namespace bucket_full_weight_l80_80278

theorem bucket_full_weight (x y c d : ℝ)
  (h1 : x + 3 / 4 * y = c)
  (h2 : x + 1 / 3 * y = d) :
  x + y = (8 / 5) * c - (7 / 5) * d :=
by
  sorry

end bucket_full_weight_l80_80278


namespace determine_x_l80_80688

theorem determine_x
  (total_area : ℝ)
  (side_length_square1 : ℝ)
  (side_length_square2 : ℝ)
  (h1 : total_area = 1300)
  (h2 : side_length_square1 = 3 * x)
  (h3 : side_length_square2 = 7 * x) :
    x = Real.sqrt (2600 / 137) :=
by
  sorry

end determine_x_l80_80688


namespace find_p_of_binomial_distribution_l80_80503

noncomputable def binomial_mean (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem find_p_of_binomial_distribution (p : ℝ) (h : binomial_mean 5 p = 2) : p = 0.4 :=
by
  sorry

end find_p_of_binomial_distribution_l80_80503


namespace equal_triangle_area_l80_80837

theorem equal_triangle_area
  (ABC_area : ℝ)
  (AP PB : ℝ)
  (AB_area : ℝ)
  (PQ_BQ_equal : Prop)
  (AP_ratio: AP / (AP + PB) = 3 / 5)
  (ABC_area_val : ABC_area = 15)
  (AP_val : AP = 3)
  (PB_val : PB = 2)
  (PQ_BQ_equal : PQ_BQ_equal = true) :
  ∃ area, area = 9 ∧ area = 9 :=
by
  sorry

end equal_triangle_area_l80_80837


namespace ellipse_parabola_common_point_l80_80131

theorem ellipse_parabola_common_point (a : ℝ) :
  (∃ (x y : ℝ), x^2 + 4 * (y - a)^2 = 4 ∧ x^2 = 2 * y) ↔ -1 ≤ a ∧ a ≤ 17 / 8 := 
by 
  sorry

end ellipse_parabola_common_point_l80_80131


namespace total_parents_in_auditorium_l80_80311

-- Define the conditions.
def girls : Nat := 6
def boys : Nat := 8
def total_kids : Nat := girls + boys
def parents_per_kid : Nat := 2
def total_parents : Nat := total_kids * parents_per_kid

-- The statement to prove.
theorem total_parents_in_auditorium : total_parents = 28 := by
  sorry

end total_parents_in_auditorium_l80_80311


namespace arc_length_ratio_l80_80577

theorem arc_length_ratio
  (h_circ : ∀ (x y : ℝ), (x - 1)^2 + y^2 = 1)
  (h_line : ∀ x y : ℝ, x - y = 0) :
  let shorter_arc := (1 / 4) * (2 * Real.pi)
  let longer_arc := 2 * Real.pi - shorter_arc
  shorter_arc / longer_arc = 1 / 3 :=
by
  sorry

end arc_length_ratio_l80_80577


namespace eq_satisfied_in_entire_space_l80_80870

theorem eq_satisfied_in_entire_space (x y z : ℝ) : 
  (x + y + z)^2 = x^2 + y^2 + z^2 ↔ xy + xz + yz = 0 :=
by
  sorry

end eq_satisfied_in_entire_space_l80_80870


namespace students_with_both_pets_l80_80494

theorem students_with_both_pets :
  ∀ (total_students students_with_dog students_with_cat students_with_both : ℕ),
    total_students = 45 →
    students_with_dog = 25 →
    students_with_cat = 34 →
    total_students = students_with_dog + students_with_cat - students_with_both →
    students_with_both = 14 :=
by
  intros total_students students_with_dog students_with_cat students_with_both
  sorry

end students_with_both_pets_l80_80494


namespace product_divisible_by_60_l80_80894

open Nat

theorem product_divisible_by_60 (S : Finset ℕ) (h_card : S.card = 10) (h_sum : S.sum id = 62) :
  60 ∣ S.prod id :=
  sorry

end product_divisible_by_60_l80_80894


namespace problem1_problem2_problem3_problem4_l80_80600

-- Defining each problem as a theorem statement
theorem problem1 : 20 + 3 - (-27) + (-5) = 45 :=
by sorry

theorem problem2 : (-7) - (-6 + 5 / 6) + abs (-3) + 1 + 1 / 6 = 4 :=
by sorry

theorem problem3 : (1 / 4 + 3 / 8 - 7 / 12) / (1 / 24) = 1 :=
by sorry

theorem problem4 : -1 ^ 4 - (1 - 0.4) + 1 / 3 * ((-2) ^ 2 - 6) = -2 - 4 / 15 :=
by sorry

end problem1_problem2_problem3_problem4_l80_80600


namespace calculate_teena_speed_l80_80018

noncomputable def Teena_speed (t c t_ahead_in_1_5_hours : ℝ) : ℝ :=
  let distance_initial_gap := 7.5
  let coe_speed := 40
  let time_in_hours := 1.5
  let distance_coe_travels := coe_speed * time_in_hours
  let total_distance_teena_needs := distance_coe_travels + distance_initial_gap + t_ahead_in_1_5_hours
  total_distance_teena_needs / time_in_hours

theorem calculate_teena_speed :
  (Teena_speed 7.5 40 15) = 55 :=
  by
  -- skipped proof
  sorry

end calculate_teena_speed_l80_80018


namespace constant_term_expansion_l80_80260

theorem constant_term_expansion (n : ℕ) (hn : n = 9) :
  y^3 * (x + 1 / (x^2 * y))^n = 84 :=
by sorry

end constant_term_expansion_l80_80260


namespace minimum_toys_to_add_l80_80976

theorem minimum_toys_to_add {T : ℤ} (k m n : ℤ) (h1 : T = 12 * k + 3) (h2 : T = 18 * m + 3) 
  (h3 : T = 36 * n + 3) : 
  ∃ x : ℤ, (T + x) % 7 = 0 ∧ x = 4 :=
sorry

end minimum_toys_to_add_l80_80976


namespace measure_of_angle_A_l80_80023

variables (A B C a b c : ℝ)
variables (triangle_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
variables (sides_relation : (a^2 + b^2 - c^2) * tan A = a * b)

theorem measure_of_angle_A :
  A = π / 6 :=
by 
  sorry

end measure_of_angle_A_l80_80023


namespace all_three_items_fans_l80_80546

theorem all_three_items_fans 
  (h1 : ∀ n, n = 4800 % 80 → n = 0)
  (h2 : ∀ n, n = 4800 % 40 → n = 0)
  (h3 : ∀ n, n = 4800 % 60 → n = 0)
  (h4 : ∀ n, n = 4800):
  (∃ k, k = 20):=
by
  sorry

end all_three_items_fans_l80_80546


namespace no_integer_roots_l80_80711

theorem no_integer_roots (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) : 
  ¬ ∃ x : ℤ, a * x^2 + b * x + c = 0 :=
by {
  sorry
}

end no_integer_roots_l80_80711


namespace tom_age_ratio_l80_80409

-- Define the variables and conditions
variables (T N : ℕ)

-- Condition 1: Tom's current age is twice the sum of his children's ages
def children_sum_current : ℤ := T / 2

-- Condition 2: Tom's age N years ago was three times the sum of their ages then
def children_sum_past : ℤ := (T / 2) - 2 * N

-- Main theorem statement proving the ratio T/N = 10 assuming given conditions
theorem tom_age_ratio (h1 : T = 2 * (T / 2)) 
                      (h2 : T - N = 3 * ((T / 2) - 2 * N)) : 
                      T / N = 10 :=
sorry

end tom_age_ratio_l80_80409


namespace staplers_left_l80_80127

-- Definitions of the conditions
def initialStaplers : ℕ := 50
def dozen : ℕ := 12
def reportsStapled : ℕ := 3 * dozen

-- The proof statement
theorem staplers_left : initialStaplers - reportsStapled = 14 := by
  sorry

end staplers_left_l80_80127


namespace retail_price_before_discount_l80_80020

variable (R : ℝ) -- Let R be the retail price of each machine before the discount

theorem retail_price_before_discount :
    let wholesale_price := 126
    let machines := 10
    let bulk_discount_rate := 0.05
    let profit_margin := 0.20
    let sales_tax_rate := 0.07
    let discount_rate := 0.10

    -- Calculate wholesale total price
    let wholesale_total := machines * wholesale_price

    -- Calculate bulk purchase discount
    let bulk_discount := bulk_discount_rate * wholesale_total

    -- Calculate total amount paid
    let amount_paid := wholesale_total - bulk_discount

    -- Calculate profit per machine
    let profit_per_machine := profit_margin * wholesale_price
    
    -- Calculate total profit
    let total_profit := machines * profit_per_machine

    -- Calculate sales tax on profit
    let tax_on_profit := sales_tax_rate * total_profit

    -- Calculate total amount after paying tax
    let total_amount_after_tax := (amount_paid + total_profit) - tax_on_profit

    -- Express total selling price after discount
    let total_selling_after_discount := machines * (0.90 * R)

    -- Total selling price after discount is equal to total amount after tax
    (9 * R = total_amount_after_tax) →
    R = 159.04 :=
by
  sorry

end retail_price_before_discount_l80_80020


namespace expected_total_rain_l80_80450

theorem expected_total_rain :
  let p_sun := 0.30
  let p_rain5 := 0.30
  let p_rain12 := 0.40
  let rain_sun := 0
  let rain_rain5 := 5
  let rain_rain12 := 12
  let days := 6
  let E_rain := p_sun * rain_sun + p_rain5 * rain_rain5 + p_rain12 * rain_rain12
  E_rain * days = 37.8 :=
by
  -- Proof omitted
  sorry

end expected_total_rain_l80_80450


namespace standard_deviation_of_data_l80_80360

noncomputable def mean (data : List ℝ) : ℝ :=
  data.sum / data.length

noncomputable def variance (data : List ℝ) : ℝ :=
  let m := mean data
  (data.map (fun x => (x - m)^2)).sum / data.length

noncomputable def std_dev (data : List ℝ) : ℝ :=
  Real.sqrt (variance data)

theorem standard_deviation_of_data :
  std_dev [5, 7, 7, 8, 10, 11] = 2 := 
sorry

end standard_deviation_of_data_l80_80360


namespace gcd_2024_1728_l80_80302

theorem gcd_2024_1728 : Int.gcd 2024 1728 = 8 := 
by
  sorry

end gcd_2024_1728_l80_80302


namespace ratio_13_2_l80_80213

def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def total_trees_that_fell : ℕ := 5
def current_total_trees : ℕ := 88

def number_narra_trees_that_fell (N : ℕ) : Prop := N + (N + 1) = total_trees_that_fell
def total_trees_before_typhoon : ℕ := initial_mahogany_trees + initial_narra_trees

def ratio_of_planted_trees_to_narra_fallen (planted : ℕ) (N : ℕ) : Prop := 
  88 - (total_trees_before_typhoon - total_trees_that_fell) = planted ∧ 
  planted / N = 13 / 2

theorem ratio_13_2 : ∃ (planted N : ℕ), 
  number_narra_trees_that_fell N ∧ 
  ratio_of_planted_trees_to_narra_fallen planted N :=
sorry

end ratio_13_2_l80_80213


namespace inequality_proof_l80_80229

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c)) ≥ (1 / (b + c) + 1 / (c + a) + 1 / (a + b)) :=
by sorry

end inequality_proof_l80_80229


namespace num_students_in_class_l80_80828

-- Define the conditions
variables (S : ℕ) (num_boys : ℕ) (num_boys_under_6ft : ℕ)

-- Assume the conditions given in the problem
axiom two_thirds_boys : num_boys = (2 * S) / 3
axiom three_fourths_under_6ft : num_boys_under_6ft = (3 * num_boys) / 4
axiom nineteen_boys_under_6ft : num_boys_under_6ft = 19

-- The statement we want to prove
theorem num_students_in_class : S = 38 :=
by
  -- Proof omitted (insert proof here)
  sorry

end num_students_in_class_l80_80828


namespace gain_percent_l80_80028

theorem gain_percent (CP SP : ℝ) (hCP : CP = 100) (hSP : SP = 115) : 
  ((SP - CP) / CP) * 100 = 15 := 
by 
  sorry

end gain_percent_l80_80028


namespace quadrant_of_angle_l80_80477

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ n : ℤ, n = 1 ∧ α = (n * π + π / 2) :=
sorry

end quadrant_of_angle_l80_80477


namespace mika_stickers_l80_80314

def s1 : ℝ := 20.5
def s2 : ℝ := 26.3
def s3 : ℝ := 19.75
def s4 : ℝ := 6.25
def s5 : ℝ := 57.65
def s6 : ℝ := 15.8

theorem mika_stickers 
  (M : ℝ)
  (hM : M = s1 + s2 + s3 + s4 + s5 + s6) 
  : M = 146.25 :=
sorry

end mika_stickers_l80_80314


namespace cos_angle_sum_eq_negative_sqrt_10_div_10_l80_80034

theorem cos_angle_sum_eq_negative_sqrt_10_div_10 
  (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.tan α = 2) :
  Real.cos (α + π / 4) = - Real.sqrt 10 / 10 := by
  sorry

end cos_angle_sum_eq_negative_sqrt_10_div_10_l80_80034


namespace initial_percent_l80_80988

theorem initial_percent (x : ℝ) :
  (x / 100) * (5 / 100) = 60 / 100 → x = 1200 := 
by 
  sorry

end initial_percent_l80_80988


namespace pears_left_l80_80071

theorem pears_left (jason_pears : ℕ) (keith_pears : ℕ) (mike_ate : ℕ) (total_pears : ℕ) (pears_left : ℕ) 
  (h1 : jason_pears = 46) 
  (h2 : keith_pears = 47) 
  (h3 : mike_ate = 12) 
  (h4 : total_pears = jason_pears + keith_pears) 
  (h5 : pears_left = total_pears - mike_ate) 
  : pears_left = 81 :=
by
  sorry

end pears_left_l80_80071


namespace problem_solution_l80_80427

-- Definitions of the arithmetic sequence a_n and its common difference and first term
variables (a d : ℝ)

-- Definitions of arithmetic sequence conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

-- Required conditions for the proof
variables (h1 : d ≠ 0) (h2 : a ≠ 0)
variables (h3 : arithmetic_sequence a d 2 * arithmetic_sequence a d 8 = (arithmetic_sequence a d 4) ^ 2)

-- The target theorem to prove
theorem problem_solution : 
  (a + (a + 4 * d) + (a + 8 * d)) / ((a + d) + (a + 2 * d)) = 3 :=
sorry

end problem_solution_l80_80427


namespace range_of_x_l80_80992

noncomputable def function_domain (x : ℝ) : Prop :=
x + 2 > 0 ∧ x ≠ 1

theorem range_of_x {x : ℝ} (h : function_domain x) : x > -2 ∧ x ≠ 1 :=
by
  sorry

end range_of_x_l80_80992


namespace simplify_expression_l80_80338

variable (y : ℝ)
variable (h : y ≠ 0)

theorem simplify_expression : (3 / 7) * (7 / y + 14 * y^3) = 3 / y + 6 * y^3 :=
by
  sorry

end simplify_expression_l80_80338


namespace solve_fraction_equation_l80_80720

theorem solve_fraction_equation (x : ℚ) :
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 := 
by
  sorry

end solve_fraction_equation_l80_80720


namespace isosceles_triangle_angle_B_l80_80853

theorem isosceles_triangle_angle_B (A B C : ℝ)
  (h_triangle : (A + B + C = 180))
  (h_exterior_A : 180 - A = 110)
  (h_sum_angles : A + B + C = 180) :
  B = 70 ∨ B = 55 ∨ B = 40 :=
by
  sorry

end isosceles_triangle_angle_B_l80_80853


namespace study_time_for_average_l80_80561

theorem study_time_for_average
    (study_time_exam1 score_exam1 : ℕ)
    (study_time_exam2 score_exam2 average_score desired_average : ℝ)
    (relation : score_exam1 = 20 * study_time_exam1)
    (direct_relation : score_exam2 = 20 * study_time_exam2)
    (total_exams : ℕ)
    (average_condition : (score_exam1 + score_exam2) / total_exams = desired_average) :
    study_time_exam2 = 4.5 :=
by
  have : total_exams = 2 := by sorry
  have : score_exam1 = 60 := by sorry
  have : desired_average = 75 := by sorry
  have : score_exam2 = 90 := by sorry
  sorry

end study_time_for_average_l80_80561


namespace part1_solution_set_part2_range_of_a_l80_80252

-- Part 1
theorem part1_solution_set (x : ℝ) : (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part 2
theorem part2_range_of_a (a : ℝ) : (∀ x : ℝ, |x - a| + |x + 3| > -a) ↔ (a > -3 / 2) :=
sorry

end part1_solution_set_part2_range_of_a_l80_80252


namespace neither_A_B_C_prob_correct_l80_80896

noncomputable def P (A B C : Prop) : Prop :=
  let P_A := 0.25
  let P_B := 0.35
  let P_C := 0.40
  let P_A_and_B := 0.10
  let P_A_and_C := 0.15
  let P_B_and_C := 0.20
  let P_A_and_B_and_C := 0.05
  
  let P_A_or_B_or_C := 
    P_A + P_B + P_C - P_A_and_B - P_A_and_C - P_B_and_C + P_A_and_B_and_C
  
  let P_neither_A_nor_B_nor_C := 1 - P_A_or_B_or_C
    
  P_neither_A_nor_B_nor_C = 0.45

theorem neither_A_B_C_prob_correct :
  P A B C := by
  sorry

end neither_A_B_C_prob_correct_l80_80896


namespace pattern_equation_l80_80479

theorem pattern_equation (n : ℕ) (h : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 := by
  sorry

end pattern_equation_l80_80479


namespace missing_coin_value_l80_80075

-- Definitions based on the conditions
def value_of_dime := 10 -- Value of 1 dime in cents
def value_of_nickel := 5 -- Value of 1 nickel in cents
def num_dimes := 1
def num_nickels := 2
def total_value_found := 45 -- Total value found in cents

-- Statement to prove the missing coin's value
theorem missing_coin_value : 
  (total_value_found - (num_dimes * value_of_dime + num_nickels * value_of_nickel)) = 25 := 
by
  sorry

end missing_coin_value_l80_80075


namespace minimum_value_l80_80524

noncomputable def f (x : ℝ) (a b : ℝ) := a^x - b
noncomputable def g (x : ℝ) := x + 1

theorem minimum_value (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f (0 : ℝ) a b * g 0 ≤ 0)
  (h4 : ∀ x : ℝ, f x a b * g x ≤ 0) : (1 / a + 4 / b) ≥ 4 :=
sorry

end minimum_value_l80_80524


namespace second_train_speed_is_correct_l80_80516

noncomputable def speed_of_second_train (length_first : ℝ) (speed_first : ℝ) (time_cross : ℝ) (length_second : ℝ) : ℝ :=
let total_distance := length_first + length_second
let relative_speed := total_distance / time_cross
let relative_speed_kmph := relative_speed * 3.6
relative_speed_kmph - speed_first

theorem second_train_speed_is_correct :
  speed_of_second_train 270 120 9 230.04 = 80.016 :=
by
  sorry

end second_train_speed_is_correct_l80_80516


namespace numberOfBigBoats_l80_80725

-- Conditions
variable (students : Nat) (bigBoatCapacity : Nat) (smallBoatCapacity : Nat) (totalBoats : Nat)
variable (students_eq : students = 52)
variable (bigBoatCapacity_eq : bigBoatCapacity = 8)
variable (smallBoatCapacity_eq : smallBoatCapacity = 4)
variable (totalBoats_eq : totalBoats = 9)

theorem numberOfBigBoats : bigBoats + smallBoats = totalBoats → 
                         bigBoatCapacity * bigBoats + smallBoatCapacity * smallBoats = students → 
                         bigBoats = 4 := 
by
  intros h1 h2
  -- Proof steps
  sorry


end numberOfBigBoats_l80_80725


namespace infinitely_many_n_l80_80003

theorem infinitely_many_n (h : ℤ) : ∃ (S : Set ℤ), S ≠ ∅ ∧ ∀ n ∈ S, ∃ k : ℕ, ⌊n * Real.sqrt (h^2 + 1)⌋ = k^2 :=
by
  sorry

end infinitely_many_n_l80_80003


namespace total_mile_times_l80_80070

-- Define the conditions
def Tina_time : ℕ := 6  -- Tina runs a mile in 6 minutes

def Tony_time : ℕ := Tina_time / 2  -- Tony runs twice as fast as Tina

def Tom_time : ℕ := Tina_time / 3  -- Tom runs three times as fast as Tina

-- Define the proof statement
theorem total_mile_times : Tony_time + Tina_time + Tom_time = 11 := by
  sorry

end total_mile_times_l80_80070


namespace ratio_of_sam_to_sue_l80_80373

-- Definitions
def Sam_age (S : ℕ) : Prop := 3 * S = 18
def Kendra_age (K : ℕ) : Prop := K = 18
def total_age_in_3_years (S U K : ℕ) : Prop := (S + 3) + (U + 3) + (K + 3) = 36

-- Theorem statement
theorem ratio_of_sam_to_sue (S U K : ℕ) (h1 : Sam_age S) (h2 : Kendra_age K) (h3 : total_age_in_3_years S U K) :
  S / U = 2 :=
sorry

end ratio_of_sam_to_sue_l80_80373


namespace max_n_for_neg_sum_correct_l80_80261

noncomputable def max_n_for_neg_sum (S : ℕ → ℤ) : ℕ :=
  if h₁ : S 19 > 0 then
    if h₂ : S 20 < 0 then
      11
    else 0  -- default value
  else 0  -- default value

theorem max_n_for_neg_sum_correct (S : ℕ → ℤ) (h₁ : S 19 > 0) (h₂ : S 20 < 0) : max_n_for_neg_sum S = 11 :=
by
  sorry

end max_n_for_neg_sum_correct_l80_80261


namespace average_speed_is_correct_l80_80350
noncomputable def average_speed_trip : ℝ :=
  let distance_AB := 240 * 5
  let distance_BC := 300 * 3
  let distance_CD := 400 * 4
  let total_distance := distance_AB + distance_BC + distance_CD
  let flight_time_AB := 5
  let layover_B := 2
  let flight_time_BC := 3
  let layover_C := 1
  let flight_time_CD := 4
  let total_time := (flight_time_AB + flight_time_BC + flight_time_CD) + (layover_B + layover_C)
  total_distance / total_time

theorem average_speed_is_correct :
  average_speed_trip = 246.67 := sorry

end average_speed_is_correct_l80_80350


namespace initial_amount_l80_80540

theorem initial_amount (X : ℝ) (h1 : 0.70 * X = 2800) : X = 4000 :=
by
  sorry

end initial_amount_l80_80540


namespace probability_of_sum_23_l80_80052

def is_valid_time (h m : ℕ) : Prop :=
  0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

def sum_of_time_digits (h m : ℕ) : ℕ :=
  sum_of_digits h + sum_of_digits m

theorem probability_of_sum_23 :
  (∃ h m, is_valid_time h m ∧ sum_of_time_digits h m = 23) →
  (4 / 1440 : ℚ) = (1 / 360 : ℚ) :=
by
  sorry

end probability_of_sum_23_l80_80052


namespace total_spending_correct_l80_80222

-- Define the costs and number of children for each ride and snack
def cost_ferris_wheel := 5 * 5
def cost_roller_coaster := 7 * 3
def cost_merry_go_round := 3 * 8
def cost_bumper_cars := 4 * 6

def cost_ice_cream := 8 * 2 * 5
def cost_hot_dog := 6 * 4
def cost_pizza := 4 * 3

-- Calculate the total cost
def total_cost_rides := cost_ferris_wheel + cost_roller_coaster + cost_merry_go_round + cost_bumper_cars
def total_cost_snacks := cost_ice_cream + cost_hot_dog + cost_pizza
def total_spent := total_cost_rides + total_cost_snacks

-- The statement to prove
theorem total_spending_correct : total_spent = 170 := by
  sorry

end total_spending_correct_l80_80222


namespace unit_digit_15_pow_100_l80_80067

theorem unit_digit_15_pow_100 : ((15^100) % 10) = 5 := 
by sorry

end unit_digit_15_pow_100_l80_80067


namespace length_of_train_correct_l80_80467

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  speed_ms * time_sec

theorem length_of_train_correct :
  length_of_train 60 18 = 300.06 :=
by
  -- Placeholder for proof
  sorry

end length_of_train_correct_l80_80467


namespace probability_of_double_tile_is_one_fourth_l80_80084

noncomputable def probability_double_tile : ℚ :=
  let total_pairs := (7 * 7) / 2
  let double_pairs := 7
  double_pairs / total_pairs

theorem probability_of_double_tile_is_one_fourth :
  probability_double_tile = 1 / 4 :=
by
  sorry

end probability_of_double_tile_is_one_fourth_l80_80084


namespace cos_double_angle_l80_80495

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (2 * θ) = -7 / 25 := sorry

end cos_double_angle_l80_80495


namespace parabola_vertex_sum_l80_80584

theorem parabola_vertex_sum (p q r : ℝ)
  (h1 : ∃ a : ℝ, ∀ x y : ℝ, y = a * (x - 3)^2 + 4 → y = p * x^2 + q * x + r)
  (h2 : ∀ y1 : ℝ, y1 = p * (1 : ℝ)^2 + q * (1 : ℝ) + r → y1 = 10)
  (h3 : ∀ y2 : ℝ, y2 = p * (-1 : ℝ)^2 + q * (-1 : ℝ) + r → y2 = 14) :
  p + q + r = 10 :=
sorry

end parabola_vertex_sum_l80_80584


namespace geometric_series_sum_l80_80554

noncomputable def first_term : ℝ := 6
noncomputable def common_ratio : ℝ := -2 / 3

theorem geometric_series_sum :
  (|common_ratio| < 1) → (first_term / (1 - common_ratio) = 18 / 5) :=
by
  intros h
  simp [first_term, common_ratio]
  sorry

end geometric_series_sum_l80_80554


namespace necessary_not_sufficient_condition_l80_80103

variable {x : ℝ}

theorem necessary_not_sufficient_condition (h : x > 2) : x > 1 :=
by
  sorry

end necessary_not_sufficient_condition_l80_80103


namespace max_value_in_interval_l80_80090

theorem max_value_in_interval :
  ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → x^4 - 2 * x^2 + 5 ≤ 13 :=
by
  sorry

end max_value_in_interval_l80_80090


namespace learning_hours_difference_l80_80223

/-- Define the hours Ryan spends on each language. -/
def hours_learned (lang : String) : ℝ :=
  if lang = "English" then 2 else
  if lang = "Chinese" then 5 else
  if lang = "Spanish" then 4 else
  if lang = "French" then 3 else
  if lang = "German" then 1.5 else 0

/-- Prove that Ryan spends 2.5 more hours learning Chinese and French combined
    than he does learning German and Spanish combined. -/
theorem learning_hours_difference :
  hours_learned "Chinese" + hours_learned "French" - (hours_learned "German" + hours_learned "Spanish") = 2.5 :=
by
  sorry

end learning_hours_difference_l80_80223


namespace correct_scientific_notation_l80_80944

def scientific_notation (n : ℝ) : ℝ × ℝ := 
  (4, 5)

theorem correct_scientific_notation : scientific_notation 400000 = (4, 5) :=
by {
  sorry
}

end correct_scientific_notation_l80_80944


namespace scooter_safety_gear_price_increase_l80_80932

theorem scooter_safety_gear_price_increase :
  let last_year_scooter_price := 200
  let last_year_gear_price := 50
  let scooter_increase_rate := 0.08
  let gear_increase_rate := 0.15
  let total_last_year_price := last_year_scooter_price + last_year_gear_price
  let this_year_scooter_price := last_year_scooter_price * (1 + scooter_increase_rate)
  let this_year_gear_price := last_year_gear_price * (1 + gear_increase_rate)
  let total_this_year_price := this_year_scooter_price + this_year_gear_price
  let total_increase := total_this_year_price - total_last_year_price
  let percent_increase := (total_increase / total_last_year_price) * 100
  percent_increase = 9 :=
by
  -- sorry is added here to skip the proof steps
  sorry

end scooter_safety_gear_price_increase_l80_80932


namespace total_votes_l80_80147

theorem total_votes (bob_votes total_votes : ℕ) (h1 : bob_votes = 48) (h2 : (2 : ℝ) / 5 * total_votes = bob_votes) :
  total_votes = 120 :=
by
  sorry

end total_votes_l80_80147


namespace red_lucky_stars_l80_80792

theorem red_lucky_stars (x : ℕ) : (20 + x + 15 > 0) → (x / (20 + x + 15) : ℚ) = 0.5 → x = 35 := by
  sorry

end red_lucky_stars_l80_80792


namespace smallest_altitude_le_3_l80_80566

theorem smallest_altitude_le_3 (a b c h_a h_b h_c : ℝ) (r : ℝ) (h_r : r = 1)
    (h_a_ge_b : a ≥ b) (h_b_ge_c : b ≥ c) 
    (area_eq1 : (a + b + c) / 2 * r = (a * h_a) / 2) 
    (area_eq2 : (a + b + c) / 2 * r = (b * h_b) / 2) 
    (area_eq3 : (a + b + c) / 2 * r = (c * h_c) / 2) : 
    min h_a (min h_b h_c) ≤ 3 := 
by
  sorry

end smallest_altitude_le_3_l80_80566


namespace base_price_lowered_percentage_l80_80974

theorem base_price_lowered_percentage (P : ℝ) (new_price final_price : ℝ) (x : ℝ)
    (h1 : new_price = P - (x / 100) * P)
    (h2 : final_price = 0.9 * new_price)
    (h3 : final_price = P - (14.5 / 100) * P) :
    x = 5 :=
  sorry

end base_price_lowered_percentage_l80_80974


namespace fractions_product_l80_80878

theorem fractions_product :
  (8 / 4) * (10 / 25) * (20 / 10) * (15 / 45) * (40 / 20) * (24 / 8) * (30 / 15) * (35 / 7) = 64 := by
  sorry

end fractions_product_l80_80878


namespace gcd_polynomials_l80_80520

def even_multiple_of_2927 (a : ℤ) : Prop := ∃ k : ℤ, a = 2 * 2927 * k

theorem gcd_polynomials (a : ℤ) (h : even_multiple_of_2927 a) :
  Int.gcd (3 * a ^ 2 + 61 * a + 143) (a + 19) = 7 :=
by
  sorry

end gcd_polynomials_l80_80520


namespace three_digit_numbers_l80_80633

theorem three_digit_numbers (n : ℕ) :
  n = 4 ↔ ∃ (x y : ℕ), 
  (100 ≤ 101 * x + 10 * y ∧ 101 * x + 10 * y < 1000) ∧ 
  (x ≠ 0 ∧ x ≠ 5) ∧ 
  (2 * x + y = 15) ∧ 
  (y < 10) :=
by { sorry }

end three_digit_numbers_l80_80633


namespace max_f_l80_80322

open Real

noncomputable def f (x : ℝ) : ℝ := 3 + log x + 4 / log x

theorem max_f (h : 0 < x ∧ x < 1) : f x ≤ -1 :=
sorry

end max_f_l80_80322


namespace problem1_l80_80249

theorem problem1 : abs (-3) + (-1: ℤ)^2021 * (Real.pi - 3.14)^0 - (- (1/2: ℝ))⁻¹ = 4 := 
  sorry

end problem1_l80_80249


namespace bryce_raisins_l80_80915

theorem bryce_raisins (x : ℕ) (h1 : x = 2 * (x - 8)) : x = 16 :=
by
  sorry

end bryce_raisins_l80_80915


namespace polynomial_discriminant_l80_80340

theorem polynomial_discriminant (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : (b - 1)^2 - 4 * a * (c + 2) = 0)
  (h3 : (b + 1 / 2)^2 - 4 * a * (c - 1) = 0) :
  b^2 - 4 * a * c = -1 / 2 :=
by
  sorry

end polynomial_discriminant_l80_80340


namespace scores_fraction_difference_l80_80441

theorem scores_fraction_difference (y : ℕ) (white_ratio : ℕ) (black_ratio : ℕ) (total : ℕ) 
(h1 : white_ratio = 7) (h2 : black_ratio = 6) (h3 : total = 78) 
(h4 : y = white_ratio + black_ratio) : 
  ((white_ratio * total / y) - (black_ratio * total / y)) / total = 1 / 13 :=
by
 sorry

end scores_fraction_difference_l80_80441


namespace line_through_point_intersects_yaxis_triangular_area_l80_80027

theorem line_through_point_intersects_yaxis_triangular_area 
  (a T : ℝ) 
  (h : 0 < a) 
  (line_eqn : ∀ x y : ℝ, x = -a * y + a → 2 * T * x + a^2 * y - 2 * a * T = 0) 
  : ∃ (m b : ℝ), (forall x y : ℝ, y = m * x + b) := 
by
  sorry

end line_through_point_intersects_yaxis_triangular_area_l80_80027


namespace customer_pays_correct_amount_l80_80062

def wholesale_price : ℝ := 4
def markup : ℝ := 0.25
def discount : ℝ := 0.05

def retail_price : ℝ := wholesale_price * (1 + markup)
def discount_amount : ℝ := retail_price * discount
def customer_price : ℝ := retail_price - discount_amount

theorem customer_pays_correct_amount : customer_price = 4.75 := by
  -- proof steps would go here, but we are skipping them as instructed
  sorry

end customer_pays_correct_amount_l80_80062


namespace problem_statement_l80_80968

variable (X Y : ℝ)

theorem problem_statement
  (h1 : 0.18 * X = 0.54 * 1200)
  (h2 : X = 4 * Y) :
  X = 3600 ∧ Y = 900 := by
  sorry

end problem_statement_l80_80968


namespace x_can_be_any_sign_l80_80300

theorem x_can_be_any_sign
  (x y p q : ℝ)
  (h1 : abs (x / y) < abs (p) / q^2)
  (h2 : y ≠ 0) (h3 : q ≠ 0) :
  ∃ (x' : ℝ), True :=
by
  sorry

end x_can_be_any_sign_l80_80300


namespace company_blocks_l80_80379

noncomputable def number_of_blocks (workers_per_block total_budget gift_cost : ℕ) : ℕ :=
  (total_budget / gift_cost) / workers_per_block

theorem company_blocks :
  number_of_blocks 200 6000 2 = 15 :=
by
  sorry

end company_blocks_l80_80379


namespace sum_of_solutions_l80_80613

theorem sum_of_solutions :
  (∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    ((x, y) = (-1, 0) ∨ (x, y) = (2, 3) ∨ (x, y) = (7, 2))) →
  ((∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    (1 + 1 = 3 ∨ true)) → 
  (∀ (x y : ℝ), (|x - 4| = |y - 5| ∧ |x - 7| = 3 * |y - 2|) →
    (x, y) = (-1, 0) ∨ (x, y) = (2, 3) ∨ (x, y) = (7, 2))) →
  (-1) + 0 + 2 + 3 + 7 + 2 = 13 :=
by
  sorry

end sum_of_solutions_l80_80613


namespace acute_triangle_condition_l80_80775

theorem acute_triangle_condition (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 0) (h3 : B > 0) (h4 : C > 0)
    (h5 : A + B > 90) (h6 : B + C > 90) (h7 : C + A > 90) : A < 90 ∧ B < 90 ∧ C < 90 :=
sorry

end acute_triangle_condition_l80_80775


namespace rainfall_thursday_l80_80653

theorem rainfall_thursday : 
  let monday_rain := 0.9
  let tuesday_rain := monday_rain - 0.7
  let wednesday_rain := tuesday_rain * 1.5
  let thursday_rain := wednesday_rain * 0.8
  thursday_rain = 0.24 :=
by
  sorry

end rainfall_thursday_l80_80653


namespace colton_stickers_final_count_l80_80146

-- Definitions based on conditions
def initial_stickers := 200
def stickers_given_to_7_friends := 6 * 7
def stickers_given_to_mandy := stickers_given_to_7_friends + 8
def remaining_after_mandy := initial_stickers - stickers_given_to_7_friends - stickers_given_to_mandy
def stickers_distributed_to_4_friends := remaining_after_mandy / 2
def remaining_after_4_friends := remaining_after_mandy - stickers_distributed_to_4_friends
def given_to_justin := 2 * remaining_after_4_friends / 3
def remaining_after_justin := remaining_after_4_friends - given_to_justin
def given_to_karen := remaining_after_justin / 5
def final_stickers := remaining_after_justin - given_to_karen

-- Theorem to state the proof problem
theorem colton_stickers_final_count : final_stickers = 15 := by
  sorry

end colton_stickers_final_count_l80_80146


namespace find_other_percentage_l80_80136

noncomputable def percentage_other_investment
  (total_investment : ℝ)
  (investment_10_percent : ℝ)
  (total_interest : ℝ)
  (interest_rate_10_percent : ℝ)
  (other_investment_interest : ℝ) : ℝ :=
  let interest_10_percent := investment_10_percent * interest_rate_10_percent
  let interest_other_investment := total_interest - interest_10_percent
  let amount_other_percentage := total_investment - investment_10_percent
  interest_other_investment / amount_other_percentage

theorem find_other_percentage :
  ∀ (total_investment : ℝ)
    (investment_10_percent : ℝ)
    (total_interest : ℝ)
    (interest_rate_10_percent : ℝ),
    total_investment = 31000 ∧
    investment_10_percent = 12000 ∧
    total_interest = 1390 ∧
    interest_rate_10_percent = 0.1 →
    percentage_other_investment total_investment investment_10_percent total_interest interest_rate_10_percent 190 = 0.01 :=
by
  intros total_investment investment_10_percent total_interest interest_rate_10_percent h
  sorry

end find_other_percentage_l80_80136


namespace square_root_condition_l80_80595

theorem square_root_condition (x : ℝ) : (6 + x ≥ 0) ↔ (x ≥ -6) :=
by sorry

end square_root_condition_l80_80595


namespace john_total_amount_l80_80640

def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount
def aunt_amount : ℕ := 3 / 2 * grandpa_amount
def uncle_amount : ℕ := 2 / 3 * grandma_amount

def total_amount : ℕ :=
  grandpa_amount + grandma_amount + aunt_amount + uncle_amount

theorem john_total_amount : total_amount = 225 := by sorry

end john_total_amount_l80_80640


namespace travel_rate_on_foot_l80_80673

theorem travel_rate_on_foot
  (total_distance : ℝ)
  (total_time : ℝ)
  (distance_on_foot : ℝ)
  (rate_on_bicycle : ℝ)
  (rate_on_foot : ℝ) :
  total_distance = 80 ∧ total_time = 7 ∧ distance_on_foot = 32 ∧ rate_on_bicycle = 16 →
  rate_on_foot = 8 := by
  sorry

end travel_rate_on_foot_l80_80673


namespace monica_total_savings_l80_80478

noncomputable def weekly_savings : ℕ := 15
noncomputable def weeks_to_fill_moneybox : ℕ := 60
noncomputable def num_repeats : ℕ := 5
noncomputable def total_savings (weekly_savings weeks_to_fill_moneybox num_repeats : ℕ) : ℕ :=
  (weekly_savings * weeks_to_fill_moneybox) * num_repeats

theorem monica_total_savings :
  total_savings 15 60 5 = 4500 := by
  sorry

end monica_total_savings_l80_80478


namespace fruits_left_l80_80892

theorem fruits_left (plums guavas apples given : ℕ) (h1 : plums = 16) (h2 : guavas = 18) (h3 : apples = 21) (h4 : given = 40) : 
  (plums + guavas + apples - given = 15) :=
by
  sorry

end fruits_left_l80_80892


namespace minimize_S_n_l80_80961

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)

axiom arithmetic_sequence : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d
axiom sum_first_n_terms : ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * d)
axiom condition1 : a 0 + a 4 = -14
axiom condition2 : S 9 = -27

theorem minimize_S_n : ∃ n, ∀ m, S n ≤ S m := sorry

end minimize_S_n_l80_80961


namespace expr_eval_l80_80381

noncomputable def expr_value : ℕ :=
  (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6)

theorem expr_eval : expr_value = 18 := by
  sorry

end expr_eval_l80_80381


namespace correct_operation_l80_80382

theorem correct_operation :
  (∀ (a : ℤ), 2 * a - a ≠ 1) ∧
  (∀ (a : ℤ), (a^2)^4 ≠ a^6) ∧
  (∀ (a b : ℤ), (a * b)^2 ≠ a * b^2) ∧
  (∀ (a : ℤ), a^3 * a^2 = a^5) :=
by
  sorry

end correct_operation_l80_80382


namespace total_savings_percentage_l80_80975

theorem total_savings_percentage :
  let coat_price := 100
  let hat_price := 50
  let shoes_price := 75
  let coat_discount := 0.30
  let hat_discount := 0.40
  let shoes_discount := 0.25
  let original_total := coat_price + hat_price + shoes_price
  let coat_savings := coat_price * coat_discount
  let hat_savings := hat_price * hat_discount
  let shoes_savings := shoes_price * shoes_discount
  let total_savings := coat_savings + hat_savings + shoes_savings
  let savings_percentage := (total_savings / original_total) * 100
  savings_percentage = 30.556 :=
by
  sorry

end total_savings_percentage_l80_80975


namespace warehouse_can_release_100kg_l80_80879

theorem warehouse_can_release_100kg (a b c d : ℕ) : 
  24 * a + 23 * b + 17 * c + 16 * d = 100 → True :=
by
  sorry

end warehouse_can_release_100kg_l80_80879


namespace initial_volume_shampoo_l80_80638

theorem initial_volume_shampoo (V : ℝ) 
  (replace_rate : ℝ)
  (use_rate : ℝ)
  (t : ℝ) 
  (hot_sauce_fraction : ℝ) 
  (hot_sauce_amount : ℝ) : 
  replace_rate = 1/2 → 
  use_rate = 1 → 
  t = 4 → 
  hot_sauce_fraction = 0.25 → 
  hot_sauce_amount = t * replace_rate → 
  hot_sauce_amount = hot_sauce_fraction * V → 
  V = 8 :=
by 
  intro h_replace_rate h_use_rate h_t h_hot_sauce_fraction h_hot_sauce_amount h_hot_sauce_amount_eq
  sorry

end initial_volume_shampoo_l80_80638


namespace angle_in_first_quadrant_l80_80788

theorem angle_in_first_quadrant (α : ℝ) (h : 90 < α ∧ α < 180) : 0 < 180 - α ∧ 180 - α < 90 :=
by
  sorry

end angle_in_first_quadrant_l80_80788


namespace geom_seq_general_term_arith_seq_sum_l80_80753

theorem geom_seq_general_term (q : ℕ → ℕ) (a_1 a_2 a_3 : ℕ) (h1 : a_1 = 2)
  (h2 : (a_1 + a_3) / 2 = a_2 + 1) (h3 : a_2 = q 2) (h4 : a_3 = q 3)
  (g : ℕ → ℕ) (Sn : ℕ → ℕ) (gen_term : ∀ n, q n = 2^n) (sum_term : ∀ n, Sn n = 2^(n+1) - 2) :
  q n = g n :=
sorry

theorem arith_seq_sum (a_1 a_2 a_4 : ℕ) (b : ℕ → ℕ) (Tn : ℕ → ℕ) (h1 : a_1 = 2)
  (h2 : a_2 = 4) (h3 : a_4 = 16) (h4 : b 2 = a_1) (h5 : b 8 = a_2 + a_4)
  (gen_term : ∀ n, b n = 1 + 3 * (n - 1)) (sum_term : ∀ n, Tn n = (3 * n^2 - n) / 2) :
  Tn n = (3 * n^2 - 1) / 2 :=
sorry

end geom_seq_general_term_arith_seq_sum_l80_80753


namespace proportion_decrease_l80_80341

open Real

/-- 
Given \(x\) and \(y\) are directly proportional and positive,
if \(x\) decreases by \(q\%\), then \(y\) decreases by \(q\%\).
-/
theorem proportion_decrease (c x q : ℝ) (h_pos : x > 0) (h_q_pos : q > 0)
    (h_direct : ∀ x y, y = c * x) :
    ((x * (1 - q / 100)) = y) → ((y * (1 - q / 100)) = (c * x * (1 - q / 100))) := by
  sorry

end proportion_decrease_l80_80341


namespace ratio_of_perimeters_l80_80832

theorem ratio_of_perimeters (d : ℝ) (s1 s2 P1 P2 : ℝ) (h1 : d^2 = 2 * s1^2)
  (h2 : (3 * d)^2 = 2 * s2^2) (h3 : P1 = 4 * s1) (h4 : P2 = 4 * s2) :
  P2 / P1 = 3 := 
by sorry

end ratio_of_perimeters_l80_80832


namespace octagon_ratio_l80_80198

theorem octagon_ratio (total_area : ℝ) (area_below_PQ : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) (XQ QY : ℝ) :
  total_area = 10 ∧
  area_below_PQ = 5 ∧
  triangle_base = 5 ∧
  triangle_height = 8 / 5 ∧
  area_below_PQ = 1 + (1 / 2) * triangle_base * triangle_height ∧
  XQ + QY = triangle_base ∧
  (1 / 2) * (XQ + QY) * triangle_height = 5
  → (XQ / QY) = 2 / 3 := 
sorry

end octagon_ratio_l80_80198


namespace picture_edge_distance_l80_80958

theorem picture_edge_distance 
    (wall_width : ℕ) 
    (picture_width : ℕ) 
    (centered : Bool) 
    (h_w : wall_width = 22) 
    (h_p : picture_width = 4) 
    (h_c : centered = true) : 
    ∃ (distance : ℕ), distance = 9 := 
by
  sorry

end picture_edge_distance_l80_80958


namespace triangle_AD_eq_8sqrt2_l80_80744

/-- Given a triangle ABC where AB = 13, AC = 20, and
    D is the foot of the perpendicular from A to BC,
    with the ratio BD : CD = 3 : 4, prove that AD = 8√2. -/
theorem triangle_AD_eq_8sqrt2 
  (AB AC : ℝ) (BD CD AD : ℝ) 
  (h₁ : AB = 13)
  (h₂ : AC = 20)
  (h₃ : BD / CD = 3 / 4)
  (h₄ : BD^2 = AB^2 - AD^2)
  (h₅ : CD^2 = AC^2 - AD^2) :
  AD = 8 * Real.sqrt 2 :=
by
  sorry

end triangle_AD_eq_8sqrt2_l80_80744


namespace simplify_expression_l80_80866

theorem simplify_expression (x : ℝ) : 
  (3*x - 4)*(2*x + 9) - (x + 6)*(3*x + 2) = 3*x^2 - x - 48 :=
by
  sorry

end simplify_expression_l80_80866


namespace gaussian_solutions_count_l80_80621

noncomputable def solve_gaussian (x : ℝ) : ℕ :=
  if h : x^2 = 2 * (⌊x⌋ : ℝ) + 1 then 
    1 
  else
    0

theorem gaussian_solutions_count :
  ∀ x : ℝ, solve_gaussian x = 2 :=
sorry

end gaussian_solutions_count_l80_80621


namespace group_sizes_correct_l80_80512

-- Define the number of fruits and groups
def num_bananas : Nat := 527
def num_oranges : Nat := 386
def num_apples : Nat := 319

def groups_bananas : Nat := 11
def groups_oranges : Nat := 103
def groups_apples : Nat := 17

-- Define the expected sizes of each group
def bananas_per_group : Nat := 47
def oranges_per_group : Nat := 3
def apples_per_group : Nat := 18

-- Prove the sizes of the groups are as expected
theorem group_sizes_correct :
  (num_bananas / groups_bananas = bananas_per_group) ∧
  (num_oranges / groups_oranges = oranges_per_group) ∧
  (num_apples / groups_apples = apples_per_group) :=
by
  -- Division in Nat rounds down
  have h1 : num_bananas / groups_bananas = 47 := by sorry
  have h2 : num_oranges / groups_oranges = 3 := by sorry
  have h3 : num_apples / groups_apples = 18 := by sorry
  exact ⟨h1, h2, h3⟩

end group_sizes_correct_l80_80512


namespace range_of_f_minus_2_l80_80200

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^2 + b * x

theorem range_of_f_minus_2 (a b : ℝ) (h1 : 1 ≤ f (-1) a b) (h2 : f (-1) a b ≤ 2) (h3 : 2 ≤ f 1 a b) (h4 : f 1 a b ≤ 4) :
  6 ≤ f (-2) a b ∧ f (-2) a b ≤ 10 :=
sorry

end range_of_f_minus_2_l80_80200


namespace total_bugs_eaten_l80_80825

-- Define the conditions
def gecko_eats : ℕ := 12
def lizard_eats : ℕ := gecko_eats / 2
def frog_eats : ℕ := lizard_eats * 3
def toad_eats : ℕ := frog_eats + (frog_eats / 2)

-- Define the proof
theorem total_bugs_eaten : gecko_eats + lizard_eats + frog_eats + toad_eats = 63 :=
by
  sorry

end total_bugs_eaten_l80_80825


namespace initial_cell_count_l80_80586

theorem initial_cell_count (f : ℕ → ℕ) (h₁ : ∀ n, f (n + 1) = 2 * (f n - 2)) (h₂ : f 5 = 164) : f 0 = 9 :=
sorry

end initial_cell_count_l80_80586


namespace scheduled_conference_games_total_l80_80871

def number_of_teams_in_A := 7
def number_of_teams_in_B := 5
def games_within_division (n : Nat) : Nat := n * (n - 1)
def interdivision_games := 7 * 5
def rivalry_games := 7

theorem scheduled_conference_games_total : 
  let games_A := games_within_division number_of_teams_in_A
  let games_B := games_within_division number_of_teams_in_B
  let total_games := games_A + games_B + interdivision_games + rivalry_games
  total_games = 104 :=
by
  sorry

end scheduled_conference_games_total_l80_80871


namespace reciprocal_of_neg_2023_l80_80012

theorem reciprocal_of_neg_2023 : 1 / (-2023) = -1 / 2023 :=
by
  sorry

end reciprocal_of_neg_2023_l80_80012


namespace savings_calculation_l80_80597

theorem savings_calculation (x : ℕ) (h1 : 15 * x = 15000) : (15000 - 8 * x = 7000) :=
sorry

end savings_calculation_l80_80597


namespace final_balance_l80_80490

noncomputable def initial_balance : ℕ := 10
noncomputable def charity_donation : ℕ := 4
noncomputable def prize_amount : ℕ := 90
noncomputable def lost_at_first_slot : ℕ := 50
noncomputable def lost_at_second_slot : ℕ := 10
noncomputable def lost_at_last_slot : ℕ := 5
noncomputable def cost_of_water : ℕ := 1
noncomputable def cost_of_lottery_ticket : ℕ := 1
noncomputable def lottery_win : ℕ := 65

theorem final_balance : 
  initial_balance - charity_donation + prize_amount - (lost_at_first_slot + lost_at_second_slot + lost_at_last_slot) - (cost_of_water + cost_of_lottery_ticket) + lottery_win = 94 := 
by 
  -- This is the lean statement, the proof is not required as per instructions.
  sorry

end final_balance_l80_80490


namespace original_price_of_cycle_l80_80109

/--
A man bought a cycle for some amount and sold it at a loss of 20%.
The selling price of the cycle is Rs. 1280.
What was the original price of the cycle?
-/
theorem original_price_of_cycle
    (loss_percent : ℝ)
    (selling_price : ℝ)
    (original_price : ℝ)
    (h_loss_percent : loss_percent = 0.20)
    (h_selling_price : selling_price = 1280)
    (h_selling_eqn : selling_price = (1 - loss_percent) * original_price) :
    original_price = 1600 :=
sorry

end original_price_of_cycle_l80_80109


namespace emails_in_morning_and_evening_l80_80716

def morning_emails : ℕ := 3
def afternoon_emails : ℕ := 4
def evening_emails : ℕ := 8

theorem emails_in_morning_and_evening : morning_emails + evening_emails = 11 :=
by
  sorry

end emails_in_morning_and_evening_l80_80716


namespace maddie_episodes_friday_l80_80293

theorem maddie_episodes_friday :
  let total_episodes : ℕ := 8
  let episode_duration : ℕ := 44
  let monday_time : ℕ := 138
  let thursday_time : ℕ := 21
  let weekend_time : ℕ := 105
  let total_time : ℕ := total_episodes * episode_duration
  let non_friday_time : ℕ := monday_time + thursday_time + weekend_time
  let friday_time : ℕ := total_time - non_friday_time
  let friday_episodes : ℕ := friday_time / episode_duration
  friday_episodes = 2 :=
by
  sorry

end maddie_episodes_friday_l80_80293


namespace power_inequality_l80_80530

theorem power_inequality (a b n : ℕ) (h_ab : a > b) (h_b1 : b > 1)
  (h_odd_b : b % 2 = 1) (h_n_pos : 0 < n) (h_div : b^n ∣ a^n - 1) :
  a^b > 3^n / n :=
by
  sorry

end power_inequality_l80_80530


namespace numberOfKidsInOtherClass_l80_80769

-- Defining the conditions as given in the problem
def kidsInSwansonClass := 25
def averageZitsSwansonClass := 5
def averageZitsOtherClass := 6
def additionalZitsInOtherClass := 67

-- Total number of zits in Ms. Swanson's class
def totalZitsSwansonClass := kidsInSwansonClass * averageZitsSwansonClass

-- Total number of zits in the other class
def totalZitsOtherClass := totalZitsSwansonClass + additionalZitsInOtherClass

-- Proof that the number of kids in the other class is 32
theorem numberOfKidsInOtherClass : 
  (totalZitsOtherClass / averageZitsOtherClass = 32) :=
by
  -- Proof is left as an exercise.
  sorry

end numberOfKidsInOtherClass_l80_80769


namespace fruit_punch_total_l80_80357

section fruit_punch
variable (orange_punch : ℝ) (cherry_punch : ℝ) (apple_juice : ℝ) (total_punch : ℝ)

axiom h1 : orange_punch = 4.5
axiom h2 : cherry_punch = 2 * orange_punch
axiom h3 : apple_juice = cherry_punch - 1.5
axiom h4 : total_punch = orange_punch + cherry_punch + apple_juice

theorem fruit_punch_total : total_punch = 21 := sorry

end fruit_punch

end fruit_punch_total_l80_80357


namespace cone_sphere_ratio_l80_80421

theorem cone_sphere_ratio (r h : ℝ) (h_r_ne_zero : r ≠ 0)
  (h_vol_cone : (1 / 3) * π * r^2 * h = (1 / 3) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 := 
by
  sorry

end cone_sphere_ratio_l80_80421


namespace integer_solutions_system_l80_80292

theorem integer_solutions_system :
  {x : ℤ | (4 * (1 + x) / 3 - 1 ≤ (5 + x) / 2) ∧ (x - 5 ≤ (3 * (3 * x - 2)) / 2)} = {0, 1, 2} :=
by
  sorry

end integer_solutions_system_l80_80292


namespace stratified_sampling_num_of_female_employees_l80_80541

theorem stratified_sampling_num_of_female_employees :
  ∃ (total_employees male_employees sample_size female_employees_to_draw : ℕ),
    total_employees = 750 ∧
    male_employees = 300 ∧
    sample_size = 45 ∧
    female_employees_to_draw = (total_employees - male_employees) * sample_size / total_employees ∧
    female_employees_to_draw = 27 :=
by
  sorry

end stratified_sampling_num_of_female_employees_l80_80541


namespace part_I_part_II_l80_80369

noncomputable def vector_a : ℝ × ℝ := (4, 3)
noncomputable def vector_b : ℝ × ℝ := (5, -12)
noncomputable def vector_sum := (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def vector_magnitude_sum := magnitude vector_sum
noncomputable def magnitude_a := magnitude vector_a
noncomputable def magnitude_b := magnitude vector_b
noncomputable def cos_theta := dot_product vector_a vector_b / (magnitude_a * magnitude_b)

-- Prove the magnitude of the sum of vectors is 9√2
theorem part_I : vector_magnitude_sum = 9 * Real.sqrt 2 :=
by
  sorry

-- Prove the cosine of the angle between the vectors is -16/65
theorem part_II : cos_theta = -16 / 65 :=
by
  sorry

end part_I_part_II_l80_80369


namespace total_books_l80_80204

theorem total_books (d k g : ℕ) 
  (h1 : d = 6) 
  (h2 : k = d / 2) 
  (h3 : g = 5 * (d + k)) : 
  d + k + g = 54 :=
by
  sorry

end total_books_l80_80204


namespace map_area_l80_80344

def length : ℕ := 5
def width : ℕ := 2
def area_of_map (length width : ℕ) : ℕ := length * width

theorem map_area : area_of_map length width = 10 := by
  sorry

end map_area_l80_80344


namespace longest_segment_CD_l80_80858

variables (A B C D : Type)
variables (angle_ABD angle_ADB angle_BDC angle_CBD : ℝ)

axiom angle_ABD_eq : angle_ABD = 30
axiom angle_ADB_eq : angle_ADB = 65
axiom angle_BDC_eq : angle_BDC = 60
axiom angle_CBD_eq : angle_CBD = 80

theorem longest_segment_CD
  (h_ABD : angle_ABD = 30)
  (h_ADB : angle_ADB = 65)
  (h_BDC : angle_BDC = 60)
  (h_CBD : angle_CBD = 80) : false :=
sorry

end longest_segment_CD_l80_80858


namespace solve_fractional_equation_1_solve_fractional_equation_2_l80_80231

-- Proof Problem 1
theorem solve_fractional_equation_1 (x : ℝ) (h : 6 * x - 2 ≠ 0) :
  (3 / 2 - 1 / (3 * x - 1) = 5 / (6 * x - 2)) ↔ (x = 10 / 9) :=
sorry

-- Proof Problem 2
theorem solve_fractional_equation_2 (x : ℝ) (h1 : 3 * x - 6 ≠ 0) :
  (5 * x - 4) / (x - 2) = (4 * x + 10) / (3 * x - 6) - 1 → false :=
sorry

end solve_fractional_equation_1_solve_fractional_equation_2_l80_80231


namespace flag_count_l80_80106

def colors := 3

def stripes := 3

noncomputable def number_of_flags (colors stripes : ℕ) : ℕ :=
  colors ^ stripes

theorem flag_count : number_of_flags colors stripes = 27 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end flag_count_l80_80106


namespace largest_three_digit_number_l80_80548

theorem largest_three_digit_number (a b c : ℕ) (h1 : a = 8) (h2 : b = 0) (h3 : c = 7) :
  ∃ (n : ℕ), ∀ (x : ℕ), (x = a * 100 + b * 10 + c) → x = 870 :=
by
  sorry

end largest_three_digit_number_l80_80548


namespace sufficient_condition_not_necessary_condition_l80_80219

variables (p q : Prop)
def φ := ¬p ∧ ¬q
def ψ := ¬p

theorem sufficient_condition : φ p q → ψ p := 
sorry

theorem not_necessary_condition : ψ p → ¬ (φ p q) :=
sorry

end sufficient_condition_not_necessary_condition_l80_80219


namespace weight_of_8_moles_of_AlI3_l80_80713

noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_I : ℝ := 126.90
noncomputable def molecular_weight_AlI3 : ℝ := atomic_weight_Al + 3 * atomic_weight_I

theorem weight_of_8_moles_of_AlI3 : 
  (8 * molecular_weight_AlI3) = 3261.44 := by
sorry

end weight_of_8_moles_of_AlI3_l80_80713


namespace Lennon_total_reimbursement_l80_80567

def mileage_reimbursement (industrial_weekday: ℕ → ℕ) (commercial_weekday: ℕ → ℕ) (weekend: ℕ → ℕ) : ℕ :=
  let industrial_rate : ℕ := 36
  let commercial_weekday_rate : ℕ := 42
  let weekend_rate : ℕ := 45
  (industrial_weekday 1 * industrial_rate + commercial_weekday 1 * commercial_weekday_rate)    -- Monday
  + (industrial_weekday 2 * industrial_rate + commercial_weekday 2 * commercial_weekday_rate + commercial_weekday 3 * commercial_weekday_rate)  -- Tuesday
  + (industrial_weekday 3 * industrial_rate + commercial_weekday 3 * commercial_weekday_rate)    -- Wednesday
  + (commercial_weekday 4 * commercial_weekday_rate + commercial_weekday 5 * commercial_weekday_rate)  -- Thursday
  + (industrial_weekday 5 * industrial_rate + commercial_weekday 6 * commercial_weekday_rate + industrial_weekday 6 * industrial_rate)    -- Friday
  + (weekend 1 * weekend_rate)                                       -- Saturday

def monday_industrial_miles : ℕ := 10
def monday_commercial_miles : ℕ := 8

def tuesday_industrial_miles : ℕ := 12
def tuesday_commercial_miles_1 : ℕ := 9
def tuesday_commercial_miles_2 : ℕ := 5

def wednesday_industrial_miles : ℕ := 15
def wednesday_commercial_miles : ℕ := 5

def thursday_commercial_miles_1 : ℕ := 10
def thursday_commercial_miles_2 : ℕ := 10

def friday_industrial_miles_1 : ℕ := 5
def friday_commercial_miles : ℕ := 8
def friday_industrial_miles_2 : ℕ := 3

def saturday_commercial_miles : ℕ := 12

def reimbursement_total :=
  mileage_reimbursement
    (fun day => if day = 1 then monday_industrial_miles else if day = 2 then tuesday_industrial_miles else if day = 3 then wednesday_industrial_miles else if day = 5 then friday_industrial_miles_1 + friday_industrial_miles_2 else 0)
    (fun day => if day = 1 then monday_commercial_miles else if day = 2 then tuesday_commercial_miles_1 + tuesday_commercial_miles_2 else if day = 3 then wednesday_commercial_miles else if day = 4 then thursday_commercial_miles_1 + thursday_commercial_miles_2 else if day = 6 then friday_commercial_miles else 0)
    (fun day => if day = 1 then saturday_commercial_miles else 0)

theorem Lennon_total_reimbursement : reimbursement_total = 4470 := 
by sorry

end Lennon_total_reimbursement_l80_80567


namespace find_prime_factors_l80_80618

-- Define n and the prime numbers p and q
def n : ℕ := 400000001
def p : ℕ := 20201
def q : ℕ := 19801

-- Main theorem statement
theorem find_prime_factors (hn : n = p * q) 
  (hp : Prime p) 
  (hq : Prime q) : 
  n = 400000001 ∧ p = 20201 ∧ q = 19801 := 
by {
  sorry
}

end find_prime_factors_l80_80618


namespace Natasha_avg_speed_climb_l80_80371

-- Definitions for conditions
def distance_to_top : ℝ := sorry -- We need to find this
def time_up := 3 -- time in hours to climb up
def time_down := 2 -- time in hours to climb down
def avg_speed_journey := 3 -- avg speed in km/hr for the whole journey

-- Equivalent math proof problem statement
theorem Natasha_avg_speed_climb (distance_to_top : ℝ) 
  (h1 : time_up = 3)
  (h2 : time_down = 2)
  (h3 : avg_speed_journey = 3)
  (h4 : (2 * distance_to_top) / (time_up + time_down) = avg_speed_journey) : 
  (distance_to_top / time_up) = 2.5 :=
sorry -- Proof not required

end Natasha_avg_speed_climb_l80_80371


namespace sqrt_sum_l80_80493

theorem sqrt_sum (a b : ℕ) (ha : a = 72) (hb : b = 32) : 
  Real.sqrt a + Real.sqrt b = 10 * Real.sqrt 2 := 
by 
  rw [ha, hb] 
  -- Insert any further required simplifications as a formal proof or leave it abstracted.
  exact sorry -- skipping the proof to satisfy this step.

end sqrt_sum_l80_80493


namespace subset_implies_bound_l80_80708

def setA := {x : ℝ | x < 2}
def setB (m : ℝ) := {x : ℝ | x < m}

theorem subset_implies_bound (m : ℝ) (h : setB m ⊆ setA) : m ≤ 2 :=
by 
  sorry

end subset_implies_bound_l80_80708


namespace john_paid_percentage_l80_80803

theorem john_paid_percentage (SRP WP : ℝ) (h1 : SRP = 1.40 * WP) (h2 : ∀ P, P = (1 / 3) * SRP) : ((1 / 3) * SRP / SRP * 100) = 33.33 :=
by
  sorry

end john_paid_percentage_l80_80803


namespace M_inter_N_eq_M_l80_80059

-- Definitions of the sets M and N
def M : Set ℝ := {x | abs (x - 1) < 1}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- The desired equality
theorem M_inter_N_eq_M : M ∩ N = M := 
by
  sorry

end M_inter_N_eq_M_l80_80059


namespace trapezoid_area_l80_80000

theorem trapezoid_area :
  ∃ S, (S = 6 ∨ S = 10) ∧ 
  ((∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 4 ∧ d = 5 ∧ 
    (∃ (is_isosceles_trapezoid : Prop), is_isosceles_trapezoid)) ∨
   (∃ (a b c d : ℝ), a = 1 ∧ b = 4 ∧ c = 5 ∧ d = 4 ∧ 
    (∃ (is_right_angled_trapezoid : Prop), is_right_angled_trapezoid)) ∨ 
   (∃ (a b c d : ℝ), (a = 1 ∨ b = 1 ∨ c = 1 ∨ d = 1) →
   (∀ (is_impossible_trapezoid : Prop), ¬ is_impossible_trapezoid))) :=
sorry

end trapezoid_area_l80_80000


namespace trip_to_Atlanta_equals_Boston_l80_80209

def distance_to_Boston : ℕ := 840
def daily_distance : ℕ := 40
def num_days (distance : ℕ) (daily : ℕ) : ℕ := distance / daily
def distance_to_Atlanta (days : ℕ) (daily : ℕ) : ℕ := days * daily

theorem trip_to_Atlanta_equals_Boston :
  distance_to_Atlanta (num_days distance_to_Boston daily_distance) daily_distance = distance_to_Boston :=
by
  -- Here we would insert the proof.
  sorry

end trip_to_Atlanta_equals_Boston_l80_80209


namespace Maria_needs_more_l80_80158

def num_mechanics : Nat := 20
def num_thermodynamics : Nat := 50
def num_optics : Nat := 30
def total_questions : Nat := num_mechanics + num_thermodynamics + num_optics

def correct_mechanics : Nat := (80 * num_mechanics) / 100
def correct_thermodynamics : Nat := (50 * num_thermodynamics) / 100
def correct_optics : Nat := (70 * num_optics) / 100
def correct_total : Nat := correct_mechanics + correct_thermodynamics + correct_optics

def correct_for_passing : Nat := (65 * total_questions) / 100
def additional_needed : Nat := correct_for_passing - correct_total

theorem Maria_needs_more:
  additional_needed = 3 := by
  sorry

end Maria_needs_more_l80_80158


namespace problem_statement_l80_80977

noncomputable def f (x : ℝ) : ℝ := x + 1
noncomputable def g (x : ℝ) : ℝ := -x + 1
noncomputable def h (x : ℝ) : ℝ := f x * g x

theorem problem_statement :
  (h (-x) = h x) :=
by
  sorry

end problem_statement_l80_80977


namespace determine_ab_l80_80593

theorem determine_ab (a b : ℕ) (h1: a + b = 30) (h2: 2 * a * b + 14 * a = 5 * b + 290) : a * b = 104 := by
  -- the proof would be written here
  sorry

end determine_ab_l80_80593


namespace terminal_side_third_quadrant_l80_80076

theorem terminal_side_third_quadrant (α : ℝ) (k : ℤ) 
  (hα : (π / 2) + 2 * k * π < α ∧ α < π + 2 * k * π) : 
  ¬(π + 2 * k * π < α / 3 ∧ α / 3 < (3 / 2) * π + 2 * k * π) :=
by
  sorry

end terminal_side_third_quadrant_l80_80076


namespace intersection_A_B_l80_80087

def A : Set ℝ := { y | ∃ x : ℝ, y = |x| }
def B : Set ℝ := { y | ∃ x : ℝ, y = 1 - 2*x - x^2 }

theorem intersection_A_B :
  A ∩ B = { y | 0 ≤ y ∧ y ≤ 2 } :=
sorry

end intersection_A_B_l80_80087


namespace triangle_area_l80_80608

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (7, 4)
def C : ℝ × ℝ := (7, -4)

-- Statement to prove the area of the triangle is 32 square units
theorem triangle_area :
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  (1 / 2 : ℝ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)| = 32 := by
  sorry  -- Proof to be provided

end triangle_area_l80_80608


namespace olivia_correct_answers_l80_80408

theorem olivia_correct_answers (c w : ℕ) (h1 : c + w = 15) (h2 : 4 * c - 3 * w = 25) : c = 10 :=
by
  sorry

end olivia_correct_answers_l80_80408


namespace june_earnings_l80_80362

theorem june_earnings (total_clovers : ℕ) (percent_three : ℝ) (percent_two : ℝ) (percent_four : ℝ) :
  total_clovers = 200 →
  percent_three = 0.75 →
  percent_two = 0.24 →
  percent_four = 0.01 →
  (total_clovers * percent_three + total_clovers * percent_two + total_clovers * percent_four) = 200 := 
by
  intros h1 h2 h3 h4
  sorry

end june_earnings_l80_80362


namespace second_solution_percentage_l80_80741

theorem second_solution_percentage (P : ℝ) : 
  (28 * 0.30 + 12 * P = 40 * 0.45) → P = 0.8 :=
by
  intros h
  sorry

end second_solution_percentage_l80_80741


namespace percentage_decrease_l80_80226

theorem percentage_decrease (purchase_price selling_price decrease gross_profit : ℝ)
  (h_purchase : purchase_price = 81)
  (h_markup : selling_price = purchase_price + 0.25 * selling_price)
  (h_gross_profit : gross_profit = 5.40)
  (h_decrease : decrease = 108 - 102.60) :
  (decrease / 108) * 100 = 5 :=
by sorry

end percentage_decrease_l80_80226


namespace george_max_pencils_l80_80814

-- Define the conditions for the problem
def total_money : ℝ := 9.30
def pencil_cost : ℝ := 1.05
def discount_rate : ℝ := 0.10

-- Define the final statement to prove
theorem george_max_pencils (n : ℕ) :
  (n ≤ 8 ∧ pencil_cost * n ≤ total_money) ∨ 
  (n > 8 ∧ pencil_cost * (1 - discount_rate) * n ≤ total_money) →
  n ≤ 9 :=
by
  sorry

end george_max_pencils_l80_80814


namespace possible_value_of_a_l80_80276

theorem possible_value_of_a (a : ℕ) : (5 + 8 > a ∧ a > 3) → (a = 9 → True) :=
by
  intros h ha
  sorry

end possible_value_of_a_l80_80276


namespace evaluate_expression_at_values_l80_80906

theorem evaluate_expression_at_values (x y : ℤ) (h₁ : x = 1) (h₂ : y = -2) :
  (-2 * x ^ 2 + 2 * x - y) = 2 :=
by
  subst h₁
  subst h₂
  sorry

end evaluate_expression_at_values_l80_80906


namespace initial_men_in_fort_l80_80259

theorem initial_men_in_fort (M : ℕ) 
  (h1 : ∀ N : ℕ, M * 35 = (N - 25) * 42) 
  (h2 : 10 + 42 = 52) : M = 150 :=
sorry

end initial_men_in_fort_l80_80259


namespace ratio_of_inscribed_squares_in_isosceles_right_triangle_l80_80480

def isosceles_right_triangle (a b : ℝ) (leg : ℝ) : Prop :=
  let a_square_inscribed := a = leg
  let b_square_inscribed := b = leg
  a_square_inscribed ∧ b_square_inscribed

theorem ratio_of_inscribed_squares_in_isosceles_right_triangle (a b leg : ℝ)
  (h : isosceles_right_triangle a b leg) :
  leg = 6 ∧ a = leg ∧ b = leg → a / b = 1 := 
by {
  sorry -- the proof will go here
}

end ratio_of_inscribed_squares_in_isosceles_right_triangle_l80_80480


namespace min_AB_plus_five_thirds_BF_l80_80092

theorem min_AB_plus_five_thirds_BF 
  (A : ℝ × ℝ) (onEllipse : ℝ × ℝ → Prop) (F : ℝ × ℝ)
  (B : ℝ × ℝ) (minFunction : ℝ)
  (hf : F = (-3, 0)) (hA : A = (-2,2))
  (hB : onEllipse B) :
  (∀ B', onEllipse B' → (dist A B' + 5/3 * dist B' F) ≥ minFunction) →
  minFunction = (dist A B + 5/3 * dist B F) →
  B = (-(5 * Real.sqrt 3) / 2, 2) := by
  sorry

def onEllipse (B : ℝ × ℝ) : Prop := (B.1^2) / 25 + (B.2^2) / 16 = 1

end min_AB_plus_five_thirds_BF_l80_80092


namespace roots_fourth_pow_sum_l80_80990

theorem roots_fourth_pow_sum :
  (∃ p q r : ℂ, (∀ z, (z = p ∨ z = q ∨ z = r) ↔ z^3 - z^2 + 2*z - 3 = 0) ∧ p^4 + q^4 + r^4 = 13) := by
sorry

end roots_fourth_pow_sum_l80_80990


namespace avg_rate_of_change_interval_1_2_l80_80277

def f (x : ℝ) : ℝ := 2 * x + 1

theorem avg_rate_of_change_interval_1_2 : 
  (f 2 - f 1) / (2 - 1) = 2 :=
by sorry

end avg_rate_of_change_interval_1_2_l80_80277


namespace line_perpendicular_slope_l80_80039

theorem line_perpendicular_slope (m : ℝ) :
  let slope1 := (1 / 2) 
  let slope2 := (-2 / m)
  slope1 * slope2 = -1 → m = 1 := 
by
  -- The proof will go here
  sorry

end line_perpendicular_slope_l80_80039


namespace proof_b_greater_a_greater_c_l80_80872

def a : ℤ := -2 * 3^2
def b : ℤ := (-2 * 3)^2
def c : ℤ := - (2 * 3)^2

theorem proof_b_greater_a_greater_c (ha : a = -18) (hb : b = 36) (hc : c = -36) : b > a ∧ a > c := 
by
  rw [ha, hb, hc]
  exact And.intro (by norm_num) (by norm_num)

end proof_b_greater_a_greater_c_l80_80872


namespace percentage_loss_l80_80578

variable (CP SP : ℝ)
variable (HCP : CP = 1600)
variable (HSP : SP = 1408)

theorem percentage_loss (HCP : CP = 1600) (HSP : SP = 1408) : 
  (CP - SP) / CP * 100 = 12 := by
sorry

end percentage_loss_l80_80578


namespace factor_of_polynomial_l80_80184

def polynomial (x : ℝ) : ℝ := x^4 - 4*x^2 + 16
def q1 (x : ℝ) : ℝ := x^2 + 4
def q2 (x : ℝ) : ℝ := x - 2
def q3 (x : ℝ) : ℝ := x^2 - 4*x + 4
def q4 (x : ℝ) : ℝ := x^2 + 4*x + 4

theorem factor_of_polynomial : (∃ (f g : ℝ → ℝ), polynomial x = f x * g x) ∧ (q4 = f ∨ q4 = g) := by sorry

end factor_of_polynomial_l80_80184


namespace find_number_l80_80438

variable (number x : ℝ)

theorem find_number (h1 : number * x = 1600) (h2 : x = -8) : number = -200 := by
  sorry

end find_number_l80_80438


namespace katie_added_new_songs_l80_80391

-- Definitions for the conditions
def initial_songs := 11
def deleted_songs := 7
def current_songs := 28

-- Definition of the expected answer
def new_songs_added := current_songs - (initial_songs - deleted_songs)

-- Statement of the problem in Lean
theorem katie_added_new_songs : new_songs_added = 24 :=
by
  sorry

end katie_added_new_songs_l80_80391


namespace mod_calculation_l80_80288

theorem mod_calculation :
  (3 * 43 + 6 * 37) % 60 = 51 :=
by
  sorry

end mod_calculation_l80_80288


namespace greatest_three_digit_multiple_of_17_l80_80234

def is_three_digit (n : Int) : Prop := (n ≥ 100 ∧ n ≤ 999)
def is_multiple_of_17 (n : Int) : Prop := (∃ k : Int, n = 17 * k)

theorem greatest_three_digit_multiple_of_17 : ∀ n : Int, is_three_digit n ∧ is_multiple_of_17 n → n ≤ 986 :=
sorry

end greatest_three_digit_multiple_of_17_l80_80234


namespace increase_80_by_50_percent_l80_80528

theorem increase_80_by_50_percent : 
  let original_number := 80
  let percentage_increase := 0.5
  let increase := original_number * percentage_increase
  let final_number := original_number + increase
  final_number = 120 := 
by 
  sorry

end increase_80_by_50_percent_l80_80528


namespace island_knight_majority_villages_l80_80860

def NumVillages := 1000
def NumInhabitants := 99
def TotalKnights := 54054
def AnswersPerVillage : ℕ := 66 -- Number of villagers who answered "more knights"
def RemainingAnswersPerVillage : ℕ := 33 -- Number of villagers who answered "more liars"

theorem island_knight_majority_villages : 
  ∃ n : ℕ, n = 638 ∧ (66 * n + 33 * (NumVillages - n) = TotalKnights) :=
by -- Begin the proof
  sorry -- Proof to be filled in later

end island_knight_majority_villages_l80_80860


namespace infinite_geometric_series_sum_l80_80328

noncomputable def a : ℚ := 5 / 3
noncomputable def r : ℚ := -1 / 2

theorem infinite_geometric_series_sum : 
  ∑' (n : ℕ), a * r^n = 10 / 9 := 
by sorry

end infinite_geometric_series_sum_l80_80328


namespace pairs_count_1432_1433_l80_80575

def PairsCount (n : ℕ) : ℕ :=
  -- The implementation would count the pairs (x, y) such that |x^2 - y^2| = n
  sorry

-- We write down the theorem that expresses what we need to prove
theorem pairs_count_1432_1433 : PairsCount 1432 = 8 ∧ PairsCount 1433 = 4 := by
  sorry

end pairs_count_1432_1433_l80_80575


namespace distance_earth_sun_l80_80446

theorem distance_earth_sun (speed_of_light : ℝ) (time_to_earth: ℝ) 
(h1 : speed_of_light = 3 * 10^8) 
(h2 : time_to_earth = 5 * 10^2) :
  speed_of_light * time_to_earth = 1.5 * 10^11 := 
by 
  -- proof steps can be filled here
  sorry

end distance_earth_sun_l80_80446


namespace greatest_four_digit_number_divisible_by_6_and_12_l80_80323

theorem greatest_four_digit_number_divisible_by_6_and_12 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 6 = 0) ∧ (n % 12 = 0) ∧ 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (m % 6 = 0) ∧ (m % 12 = 0) → m ≤ n) ∧
  n = 9996 := 
by
  sorry

end greatest_four_digit_number_divisible_by_6_and_12_l80_80323


namespace exists_between_elements_l80_80100

noncomputable def M : Set ℝ :=
  { x | ∃ (m n : ℕ), 0 < m ∧ 0 < n ∧ x = (m + n) / Real.sqrt (m^2 + n^2) }

theorem exists_between_elements (x y : ℝ) (hx : x ∈ M) (hy : y ∈ M) (hxy : x < y) :
  ∃ z ∈ M, x < z ∧ z < y :=
by
  sorry

end exists_between_elements_l80_80100


namespace recurring_decimal_to_fraction_l80_80918

noncomputable def recurring_decimal := 0.4 + (37 : ℝ) / (990 : ℝ)

theorem recurring_decimal_to_fraction : recurring_decimal = (433 : ℚ) / (990 : ℚ) :=
sorry

end recurring_decimal_to_fraction_l80_80918


namespace annie_total_spent_l80_80263

-- Define cost of a single television
def cost_per_tv : ℕ := 50
-- Define number of televisions bought
def number_of_tvs : ℕ := 5
-- Define cost of a single figurine
def cost_per_figurine : ℕ := 1
-- Define number of figurines bought
def number_of_figurines : ℕ := 10

-- Define total cost calculation
noncomputable def total_cost : ℕ :=
  number_of_tvs * cost_per_tv + number_of_figurines * cost_per_figurine

theorem annie_total_spent : total_cost = 260 := by
  sorry

end annie_total_spent_l80_80263


namespace jane_20_cent_items_l80_80384

theorem jane_20_cent_items {x y z : ℕ} (h1 : x + y + z = 50) (h2 : 20 * x + 150 * y + 250 * z = 5000) : x = 31 :=
by
  -- The formal proof would go here
  sorry

end jane_20_cent_items_l80_80384


namespace river_joe_collected_money_l80_80272

theorem river_joe_collected_money :
  let price_catfish : ℤ := 600 -- in cents to avoid floating point issues
  let price_shrimp : ℤ := 350 -- in cents to avoid floating point issues
  let total_orders : ℤ := 26
  let shrimp_orders : ℤ := 9
  let catfish_orders : ℤ := total_orders - shrimp_orders
  let total_catfish_sales : ℤ := catfish_orders * price_catfish
  let total_shrimp_sales : ℤ := shrimp_orders * price_shrimp
  let total_money_collected : ℤ := total_catfish_sales + total_shrimp_sales
  total_money_collected = 13350 := -- in cents, so $133.50 is 13350 cents
by
  sorry

end river_joe_collected_money_l80_80272


namespace tangent_line_circle_l80_80132

theorem tangent_line_circle (m : ℝ) (h : ∀ x y : ℝ, (x + y = 0) → ((x - m)^2 + y^2 = 2)) : m = 2 :=
sorry

end tangent_line_circle_l80_80132


namespace chord_length_l80_80930

theorem chord_length (R a b : ℝ) (hR : a + b = R) (hab : a * b = 10) 
    (h_nonneg : 0 ≤ R ∧ 0 ≤ a ∧ 0 ≤ b) : ∃ L : ℝ, L = 2 * Real.sqrt 10 :=
by
  sorry

end chord_length_l80_80930


namespace problem_statement_l80_80585

def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x <= -3}
def R (S : Set ℝ) : Set ℝ := {x | ∃ y ∈ S, x = y}

theorem problem_statement : R (M ∪ N) = {x | x >= 1} :=
by
  sorry

end problem_statement_l80_80585


namespace books_total_l80_80710

def stuBooks : ℕ := 9
def albertBooks : ℕ := 4 * stuBooks
def totalBooks : ℕ := stuBooks + albertBooks

theorem books_total : totalBooks = 45 := by
  sorry

end books_total_l80_80710


namespace base_six_product_correct_l80_80798

namespace BaseSixProduct

-- Definitions of the numbers in base six
def num1_base6 : ℕ := 1 * 6^2 + 3 * 6^1 + 2 * 6^0
def num2_base6 : ℕ := 1 * 6^1 + 4 * 6^0

-- Their product in base ten
def product_base10 : ℕ := num1_base6 * num2_base6

-- Convert the base ten product back to base six
def product_base6 : ℕ := 2 * 6^3 + 3 * 6^2 + 3 * 6^1 + 2 * 6^0

-- Theorem statement
theorem base_six_product_correct : product_base10 = 560 ∧ product_base6 = 2332 := by
  sorry

end BaseSixProduct

end base_six_product_correct_l80_80798


namespace lemonade_quart_calculation_l80_80250

-- Define the conditions
def water_parts := 5
def lemon_juice_parts := 3
def total_parts := water_parts + lemon_juice_parts

def gallons := 2
def quarts_per_gallon := 4
def total_quarts := gallons * quarts_per_gallon
def quarts_per_part := total_quarts / total_parts

-- Proof problem
theorem lemonade_quart_calculation :
  let water_quarts := water_parts * quarts_per_part
  let lemon_juice_quarts := lemon_juice_parts * quarts_per_part
  water_quarts = 5 ∧ lemon_juice_quarts = 3 :=
by
  let water_quarts := water_parts * quarts_per_part
  let lemon_juice_quarts := lemon_juice_parts * quarts_per_part
  have h_w : water_quarts = 5 := sorry
  have h_l : lemon_juice_quarts = 3 := sorry
  exact ⟨h_w, h_l⟩

end lemonade_quart_calculation_l80_80250


namespace luncheon_cost_l80_80085

variable (s c p : ℝ)
variable (eq1 : 5 * s + 9 * c + 2 * p = 6.50)
variable (eq2 : 7 * s + 14 * c + 3 * p = 9.45)
variable (eq3 : 4 * s + 8 * c + p = 5.20)

theorem luncheon_cost :
  s + c + p = 1.30 :=
by
  sorry

end luncheon_cost_l80_80085


namespace women_stockbrokers_2005_l80_80623

-- Define the context and conditions
def women_stockbrokers_2000 : ℕ := 10000
def percent_increase_2005 : ℕ := 100

-- Statement to prove the number of women stockbrokers in 2005
theorem women_stockbrokers_2005 : women_stockbrokers_2000 + women_stockbrokers_2000 * percent_increase_2005 / 100 = 20000 := by
  sorry

end women_stockbrokers_2005_l80_80623


namespace geom_seq_sum_4n_l80_80912

-- Assume we have a geometric sequence with positive terms and common ratio q
variables (a : ℕ → ℝ) (q : ℝ) (n : ℕ)

-- The sum of the first n terms of the geometric sequence is S_n
noncomputable def S_n : ℝ := a 0 * (1 - q^n) / (1 - q)

-- Given conditions
axiom h1 : S_n a q n = 2
axiom h2 : S_n a q (3 * n) = 14

-- We need to prove that S_{4n} = 30
theorem geom_seq_sum_4n : S_n a q (4 * n) = 30 :=
by
  sorry

end geom_seq_sum_4n_l80_80912


namespace adding_books_multiplying_books_l80_80749

-- Define the conditions
def num_books_first_shelf : ℕ := 4
def num_books_second_shelf : ℕ := 5
def num_books_third_shelf : ℕ := 6

-- Define the first question and prove its correctness
theorem adding_books :
  num_books_first_shelf + num_books_second_shelf + num_books_third_shelf = 15 :=
by
  -- The proof steps would go here, but they are replaced with sorry for now
  sorry

-- Define the second question and prove its correctness
theorem multiplying_books :
  num_books_first_shelf * num_books_second_shelf * num_books_third_shelf = 120 :=
by
  -- The proof steps would go here, but they are replaced with sorry for now
  sorry

end adding_books_multiplying_books_l80_80749


namespace truck_loading_time_l80_80776

theorem truck_loading_time (h1_rate h2_rate h3_rate : ℝ)
  (h1 : h1_rate = 1 / 5) (h2 : h2_rate = 1 / 4) (h3 : h3_rate = 1 / 6) :
  (1 / (h1_rate + h2_rate + h3_rate)) = 60 / 37 :=
by simp [h1, h2, h3]; sorry

end truck_loading_time_l80_80776


namespace correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l80_80473

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l80_80473


namespace total_investment_with_interest_l80_80183

theorem total_investment_with_interest
  (total_investment : ℝ)
  (amount_at_3_percent : ℝ)
  (interest_rate_3 : ℝ)
  (interest_rate_5 : ℝ)
  (remaining_amount : ℝ)
  (interest_3 : ℝ)
  (interest_5 : ℝ) :
  total_investment = 1000 →
  amount_at_3_percent = 199.99999999999983 →
  interest_rate_3 = 0.03 →
  interest_rate_5 = 0.05 →
  remaining_amount = total_investment - amount_at_3_percent →
  interest_3 = amount_at_3_percent * interest_rate_3 →
  interest_5 = remaining_amount * interest_rate_5 →
  total_investment + interest_3 + remaining_amount + interest_5 = 1046 :=
by
  intros H1 H2 H3 H4 H5 H6 H7
  sorry

end total_investment_with_interest_l80_80183


namespace savings_percentage_correct_l80_80008

variables (price_jacket : ℕ) (price_shirt : ℕ) (price_hat : ℕ)
          (discount_jacket : ℕ) (discount_shirt : ℕ) (discount_hat : ℕ)

def original_total_cost (price_jacket price_shirt price_hat : ℕ) : ℕ :=
  price_jacket + price_shirt + price_hat

def savings (price : ℕ) (discount : ℕ) : ℕ :=
  price * discount / 100

def total_savings (price_jacket price_shirt price_hat : ℕ)
  (discount_jacket discount_shirt discount_hat : ℕ) : ℕ :=
  (savings price_jacket discount_jacket) + (savings price_shirt discount_shirt) + (savings price_hat discount_hat)

def total_savings_percentage (price_jacket price_shirt price_hat : ℕ)
  (discount_jacket discount_shirt discount_hat : ℕ) : ℕ :=
  total_savings price_jacket price_shirt price_hat discount_jacket discount_shirt discount_hat * 100 /
  original_total_cost price_jacket price_shirt price_hat

theorem savings_percentage_correct :
  total_savings_percentage 100 50 30 30 60 50 = 4167 / 100 :=
sorry

end savings_percentage_correct_l80_80008


namespace amounts_are_correct_l80_80037

theorem amounts_are_correct (P Q R S : ℕ) 
    (h1 : P + Q + R + S = 10000)
    (h2 : R = 2 * P)
    (h3 : R = 3 * Q)
    (h4 : S = P + Q) :
    P = 1875 ∧ Q = 1250 ∧ R = 3750 ∧ S = 3125 := by
  sorry

end amounts_are_correct_l80_80037


namespace sum_of_acute_angles_l80_80863

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (hcosα : Real.cos α = 1 / Real.sqrt 10)
variable (hcosβ : Real.cos β = 1 / Real.sqrt 5)

theorem sum_of_acute_angles :
  α + β = 3 * Real.pi / 4 := by
  sorry

end sum_of_acute_angles_l80_80863


namespace trigonometric_identity_l80_80400

theorem trigonometric_identity (α : ℝ) (h : Real.sin (α - Real.pi / 4) = 1 / 3) : 
  Real.cos (Real.pi / 4 + α) = - (1 / 3) := 
by
  sorry

end trigonometric_identity_l80_80400


namespace discounted_price_l80_80211

variable (marked_price : ℝ) (discount_rate : ℝ)
variable (marked_price_def : marked_price = 150)
variable (discount_rate_def : discount_rate = 20)

theorem discounted_price (hmp : marked_price = 150) (hdr : discount_rate = 20) : 
  marked_price - (discount_rate / 100) * marked_price = 120 := by
  rw [hmp, hdr]
  sorry

end discounted_price_l80_80211


namespace subset_of_difference_empty_l80_80452

theorem subset_of_difference_empty {α : Type*} (A B : Set α) :
  (A \ B = ∅) → (A ⊆ B) :=
by
  sorry

end subset_of_difference_empty_l80_80452


namespace family_members_l80_80418

variable (p : ℝ) (i : ℝ) (c : ℝ)

theorem family_members (h1 : p = 1.6) (h2 : i = 0.25) (h3 : c = 16) :
  (c / (2 * (p * (1 + i)))) = 4 := by
  sorry

end family_members_l80_80418


namespace sequence_closed_form_l80_80838

theorem sequence_closed_form (a : ℕ → ℤ) (h1 : a 1 = 1) (h_rec : ∀ n : ℕ, a (n + 1) = 2 * a n + 3) :
  ∀ n : ℕ, a n = 2^(n + 1) - 3 :=
by 
sorry

end sequence_closed_form_l80_80838


namespace problem_1_problem_2_l80_80738

variable {c : ℝ}

def p (c : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → c ^ x₁ > c ^ x₂

def q (c : ℝ) : Prop := ∀ x₁ x₂ : ℝ, (1 / 2) < x₁ ∧ x₁ < x₂ → (x₁ ^ 2 - 2 * c * x₁ + 1) < (x₂ ^ 2 - 2 * c * x₂ + 1)

theorem problem_1 (hc : 0 < c) (hcn1 : c ≠ 1) (hp : p c) (hnq_false : ¬ ¬ q c) : 0 < c ∧ c ≤ 1 / 2 :=
by
  sorry

theorem problem_2 (hc : 0 < c) (hcn1 : c ≠ 1) (hpq_false : ¬ (p c ∧ q c)) (hp_or_q : p c ∨ q c) : 1 / 2 < c ∧ c < 1 :=
by
  sorry

end problem_1_problem_2_l80_80738


namespace mass_percentage_B_in_H3BO3_l80_80201

noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_B : ℝ := 10.81
noncomputable def atomic_mass_O : ℝ := 16.00
noncomputable def molar_mass_H3BO3 : ℝ := 3 * atomic_mass_H + atomic_mass_B + 3 * atomic_mass_O

theorem mass_percentage_B_in_H3BO3 : (atomic_mass_B / molar_mass_H3BO3) * 100 = 17.48 :=
by
  sorry

end mass_percentage_B_in_H3BO3_l80_80201


namespace machine_transportation_l80_80650

theorem machine_transportation (x y : ℕ) 
  (h1 : x + 6 - y = 10) 
  (h2 : 400 * x + 800 * (20 - x) + 300 * (6 - y) + 500 * y = 16000) : 
  x = 5 ∧ y = 1 := 
sorry

end machine_transportation_l80_80650


namespace average_age_combined_l80_80349

theorem average_age_combined (n1 n2 : ℕ) (avg1 avg2 : ℕ) 
  (h1 : n1 = 45) (h2 : n2 = 60) (h3 : avg1 = 12) (h4 : avg2 = 40) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 28 :=
by
  sorry

end average_age_combined_l80_80349


namespace number_of_classes_l80_80444

variable (s : ℕ) (h_s : s > 0)
-- Define the conditions
def student_books_year : ℕ := 4 * 12
def total_books_read : ℕ := 48
def class_books_year (s : ℕ) : ℕ := s * student_books_year
def total_classes (c s : ℕ) (h_s : s > 0) : ℕ := 1

-- Define the main theorem
theorem number_of_classes (h : total_books_read = 48) (h_s : s > 0)
  (h1 : c * class_books_year s = 48) : c = 1 := by
  sorry

end number_of_classes_l80_80444


namespace find_m_from_power_function_l80_80971

theorem find_m_from_power_function :
  (∃ a : ℝ, (2 : ℝ) ^ a = (Real.sqrt 2) / 2) →
  (∃ m : ℝ, (m : ℝ) ^ (-1 / 2 : ℝ) = 2) →
  ∃ m : ℝ, m = 1 / 4 :=
by
  intro h1 h2
  sorry

end find_m_from_power_function_l80_80971


namespace problem1_problem2_l80_80459

-- Problem 1: Prove that the given expression evaluates to the correct answer
theorem problem1 :
  2 * Real.sin (Real.pi / 6) - (2015 - Real.pi)^0 + abs (1 - Real.tan (Real.pi / 3)) = abs (1 - Real.sqrt 3) :=
sorry

-- Problem 2: Prove that the solutions to the given equation are correct
theorem problem2 (x : ℝ) :
  (x-2)^2 = 3 * (x-2) → x = 2 ∨ x = 5 :=
sorry

end problem1_problem2_l80_80459


namespace sqrt10_solution_l80_80931

theorem sqrt10_solution (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : (1/a) + (1/b) = 2) :
  m = Real.sqrt 10 :=
sorry

end sqrt10_solution_l80_80931


namespace charity_race_finished_racers_l80_80089

theorem charity_race_finished_racers :
  let initial_racers := 50
  let joined_after_20_minutes := 30
  let doubled_after_30_minutes := 2
  let dropped_racers := 30
  let total_racers_after_20_minutes := initial_racers + joined_after_20_minutes
  let total_racers_after_50_minutes := total_racers_after_20_minutes * doubled_after_30_minutes
  let finished_racers := total_racers_after_50_minutes - dropped_racers
  finished_racers = 130 := by
    sorry

end charity_race_finished_racers_l80_80089


namespace depth_of_channel_l80_80855

theorem depth_of_channel (h : ℝ) 
  (top_width : ℝ := 12) (bottom_width : ℝ := 6) (area : ℝ := 630) :
  1 / 2 * (top_width + bottom_width) * h = area → h = 70 :=
sorry

end depth_of_channel_l80_80855


namespace at_least_two_equal_l80_80388

theorem at_least_two_equal
  {a b c d : ℝ}
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h₁ : a + b + (1 / (a * b)) = c + d + (1 / (c * d)))
  (h₂ : (1 / a) + (1 / b) + (a * b) = (1 / c) + (1 / d) + (c * d)) :
  a = c ∨ a = d ∨ b = c ∨ b = d ∨ a = b ∨ c = d := by
  sorry

end at_least_two_equal_l80_80388


namespace range_of_a_for_inequality_l80_80403

theorem range_of_a_for_inequality (a : ℝ) : (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ (a ≥ -2) :=
sorry

end range_of_a_for_inequality_l80_80403


namespace candle_height_comparison_l80_80972

def first_candle_height (t : ℝ) : ℝ := 10 - 2 * t
def second_candle_height (t : ℝ) : ℝ := 8 - 2 * t

theorem candle_height_comparison (t : ℝ) :
  first_candle_height t = 3 * second_candle_height t → t = 3.5 :=
by
  -- the main proof steps would be here
  sorry

end candle_height_comparison_l80_80972


namespace competition_end_time_l80_80796

-- Definitions for the problem conditions
def start_time : ℕ := 15 * 60  -- 3:00 p.m. in minutes from midnight
def duration : ℕ := 1300       -- competition duration in minutes
def end_time : ℕ := start_time + duration

-- The expected end time in minutes from midnight, where 12:40 p.m. is (12*60 + 40) = 760 + 40 = 800 minutes from midnight.
def expected_end_time : ℕ := 12 * 60 + 40 

-- The theorem to prove
theorem competition_end_time : end_time = expected_end_time := by
  sorry

end competition_end_time_l80_80796


namespace find_m_l80_80525

variable (a b m : ℝ)

def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem find_m 
  (h₁ : right_triangle a b 5)
  (h₂ : a + b = 2*m - 1)
  (h₃ : a * b = 4 * (m - 1)) : 
  m = 4 := 
sorry

end find_m_l80_80525


namespace evaluate_at_2_l80_80874

-- Define the polynomial function using Lean
def f (x : ℝ) : ℝ := 3 * x^3 + 2 * x^2 + x + 1

-- State the theorem that f(2) evaluates to 35 using Horner's method
theorem evaluate_at_2 : f 2 = 35 := by
  sorry

end evaluate_at_2_l80_80874


namespace seats_taken_correct_l80_80761

-- Define the conditions
def rows := 40
def chairs_per_row := 20
def unoccupied_seats := 10

-- Define the total number of seats
def total_seats := rows * chairs_per_row

-- Define the number of seats taken
def seats_taken := total_seats - unoccupied_seats

-- Statement of our math proof problem
theorem seats_taken_correct : seats_taken = 790 := by
  sorry

end seats_taken_correct_l80_80761


namespace trigonometric_inequality_l80_80383

open Real

theorem trigonometric_inequality
  (x y z : ℝ)
  (h1 : 0 < x)
  (h2 : x < y)
  (h3 : y < z)
  (h4 : z < π / 2) :
  π / 2 + 2 * sin x * cos y + 2 * sin y * cos z > sin (2 * x) + sin (2 * y) + sin (2 * z) :=
by
  sorry

end trigonometric_inequality_l80_80383


namespace solve_equation_in_natural_numbers_l80_80287

theorem solve_equation_in_natural_numbers (x y : ℕ) :
  2 * y^2 - x * y - x^2 + 2 * y + 7 * x - 84 = 0 ↔ (x = 1 ∧ y = 6) ∨ (x = 14 ∧ y = 13) := 
sorry

end solve_equation_in_natural_numbers_l80_80287


namespace abs_neg_three_l80_80159

theorem abs_neg_three : abs (-3) = 3 :=
by 
  sorry

end abs_neg_three_l80_80159


namespace intersection_point_of_lines_l80_80026

theorem intersection_point_of_lines : 
  (∃ x y : ℚ, (8 * x - 3 * y = 5) ∧ (5 * x + 2 * y = 20)) ↔ (x = 70 / 31 ∧ y = 135 / 31) :=
sorry

end intersection_point_of_lines_l80_80026


namespace find_x_l80_80348

theorem find_x :
  ∃ x : ℝ, ((x * 0.85) / 2.5) - (8 * 2.25) = 5.5 ∧
  x = 69.11764705882353 :=
by
  sorry

end find_x_l80_80348


namespace total_situps_l80_80469

def situps (b c j : ℕ) : ℕ := b * 1 + c * 2 + j * 3

theorem total_situps :
  ∀ (b c j : ℕ),
    b = 45 →
    c = 2 * b →
    j = c + 5 →
    situps b c j = 510 :=
by intros b c j hb hc hj
   sorry

end total_situps_l80_80469


namespace part1_proof_l80_80140

def a : ℚ := 1 / 2
def b : ℚ := -2
def expr : ℚ := 2 * (3 * a^2 * b - a * b^2) - 3 * (2 * a^2 * b - a * b^2 + a * b)

theorem part1_proof : expr = 5 := by
  unfold expr
  unfold a
  unfold b
  sorry

end part1_proof_l80_80140


namespace total_cost_of_items_l80_80216

theorem total_cost_of_items (m n : ℕ) : (8 * m + 5 * n) = 8 * m + 5 * n := 
by sorry

end total_cost_of_items_l80_80216


namespace cost_of_each_box_is_8_33_l80_80010

noncomputable def cost_per_box (boxes pens_per_box pens_packaged price_per_packaged price_per_set profit_total : ℕ) : ℝ :=
  let total_pens := boxes * pens_per_box
  let packaged_pens := pens_packaged * pens_per_box
  let packages := packaged_pens / 6
  let revenue_packages := packages * price_per_packaged
  let remaining_pens := total_pens - packaged_pens
  let sets := remaining_pens / 3
  let revenue_sets := sets * price_per_set
  let total_revenue := revenue_packages + revenue_sets
  let cost_total := total_revenue - profit_total
  cost_total / boxes

theorem cost_of_each_box_is_8_33 :
  cost_per_box 12 30 5 3 2 115 = 100 / 12 :=
by
  unfold cost_per_box
  sorry

end cost_of_each_box_is_8_33_l80_80010


namespace indoor_players_count_l80_80464

theorem indoor_players_count (T O B I : ℕ) 
  (hT : T = 400) 
  (hO : O = 350) 
  (hB : B = 60) 
  (hEq : T = (O - B) + (I - B) + B) : 
  I = 110 := 
by sorry

end indoor_players_count_l80_80464


namespace distinct_bead_arrangements_l80_80050

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n-1)

theorem distinct_bead_arrangements : factorial 8 / (8 * 2) = 2520 := 
  by sorry

end distinct_bead_arrangements_l80_80050


namespace double_series_evaluation_l80_80378

theorem double_series_evaluation :
    (∑' m : ℕ, ∑' n : ℕ, if h : n ≥ m then 1 / (m * n * (m + n + 2)) else 0) = (Real.pi ^ 2) / 6 := sorry

end double_series_evaluation_l80_80378


namespace find_y_l80_80982

def angle_at_W (RWQ RWT QWR TWQ : ℝ) :=  RWQ + RWT + QWR + TWQ

theorem find_y 
  (RWQ RWT QWR TWQ : ℝ)
  (h1 : RWQ = 90) 
  (h2 : RWT = 3 * y)
  (h3 : QWR = y)
  (h4 : TWQ = 90) 
  (h_sum : angle_at_W RWQ RWT QWR TWQ = 360)  
  : y = 67.5 :=
by
  sorry

end find_y_l80_80982


namespace CE_squared_plus_DE_squared_proof_l80_80859

noncomputable def CE_squared_plus_DE_squared (radius : ℝ) (diameter : ℝ) (BE : ℝ) (angle_AEC : ℝ) : ℝ :=
  if radius = 10 ∧ diameter = 20 ∧ BE = 4 ∧ angle_AEC = 30 then 200 else sorry

theorem CE_squared_plus_DE_squared_proof : CE_squared_plus_DE_squared 10 20 4 30 = 200 := by
  sorry

end CE_squared_plus_DE_squared_proof_l80_80859


namespace correct_choice_option_D_l80_80347

theorem correct_choice_option_D : (500 - 9 * 7 = 437) := by sorry

end correct_choice_option_D_l80_80347


namespace value_of_expression_l80_80470

variable (m : ℝ)

theorem value_of_expression (h : 2 * m^2 + 3 * m - 1 = 0) : 
  4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end value_of_expression_l80_80470


namespace square_plot_area_l80_80727

theorem square_plot_area
  (cost_per_foot : ℕ)
  (total_cost : ℕ)
  (s : ℕ)
  (area : ℕ)
  (h1 : cost_per_foot = 55)
  (h3 : total_cost = 3740)
  (h4 : total_cost = 4 * s * cost_per_foot)
  (h5 : area = s * s) :
  area = 289 := sorry

end square_plot_area_l80_80727


namespace rhind_papyrus_smallest_portion_l80_80457

theorem rhind_papyrus_smallest_portion :
  ∀ (a1 d : ℚ),
    5 * a1 + (5 * 4 / 2) * d = 10 ∧
    (3 * a1 + 9 * d) / 7 = a1 + (a1 + d) →
    a1 = 1 / 6 :=
by sorry

end rhind_papyrus_smallest_portion_l80_80457


namespace maximum_positive_factors_l80_80214

theorem maximum_positive_factors (b n : ℕ) (hb : 0 < b ∧ b ≤ 20) (hn : 0 < n ∧ n ≤ 15) :
  ∃ k, (k = b^n) ∧ (∀ m, m = b^n → m.factors.count ≤ 61) :=
sorry

end maximum_positive_factors_l80_80214


namespace original_loaf_slices_l80_80426

-- Define the given conditions
def andy_slices_1 := 3
def andy_slices_2 := 3
def toast_slices_per_piece := 2
def pieces_of_toast := 10
def slices_left_over := 1

-- Define the variables
def total_andy_slices := andy_slices_1 + andy_slices_2
def total_toast_slices := toast_slices_per_piece * pieces_of_toast

-- State the theorem
theorem original_loaf_slices : 
  ∃ S : ℕ, S = total_andy_slices + total_toast_slices + slices_left_over := 
by {
  sorry
}

end original_loaf_slices_l80_80426


namespace value_v3_at_1_horners_method_l80_80057

def f (x : ℝ) : ℝ := 5 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem value_v3_at_1_horners_method :
  let v0 : ℝ := 5
  let v1 : ℝ := v0 * 1 + 2
  let v2 : ℝ := v1 * 1 + 3.5
  let v3 : ℝ := v2 * 1 - 2.6
  let v4 : ℝ := v3 * 1 + 1.7
  let result : ℝ := v4 * 1 - 0.8
  v3 = 7.9 :=
by
  let v0 : ℝ := 5
  let v1 : ℝ := v0 * 1 + 2
  let v2 : ℝ := v1 * 1 + 3.5
  let v3 : ℝ := v2 * 1 - 2.6
  let v4 : ℝ := v3 * 1 + 1.7
  let result : ℝ := v4 * 1 - 0.8
  exact sorry

end value_v3_at_1_horners_method_l80_80057


namespace chess_players_swim_not_every_swimmer_plays_tennis_tennis_players_play_chess_l80_80225

variables (Bodyguards : Type)
variables (U S T : Bodyguards → Prop)

-- Conditions
axiom cond1 : ∀ x, (T x → (S x → U x))
axiom cond2 : ∀ x, (S x → (U x ∨ T x))
axiom cond3 : ∀ x, (¬ U x ∧ T x → S x)

-- To prove
theorem chess_players_swim : ∀ x, (S x → U x) := by
  sorry

theorem not_every_swimmer_plays_tennis : ¬ ∀ x, (U x → T x) := by
  sorry

theorem tennis_players_play_chess : ∀ x, (T x → S x) := by
  sorry

end chess_players_swim_not_every_swimmer_plays_tennis_tennis_players_play_chess_l80_80225


namespace tables_chairs_legs_l80_80801

theorem tables_chairs_legs (t : ℕ) (c : ℕ) (total_legs : ℕ) 
  (h1 : c = 8 * t) 
  (h2 : total_legs = 4 * c + 6 * t) 
  (h3 : total_legs = 798) : 
  t = 21 :=
by
  sorry

end tables_chairs_legs_l80_80801


namespace powers_of_two_l80_80614

theorem powers_of_two (n : ℕ) (h : ∀ n, ∃ m, (2^n - 1) ∣ (m^2 + 9)) : ∃ s, n = 2^s :=
sorry

end powers_of_two_l80_80614


namespace range_of_a_l80_80445

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a + 1) * x > a + 1 ↔ x > 1) : a > -1 := 
by
  sorry

end range_of_a_l80_80445


namespace rosie_pie_count_l80_80834

def total_apples : ℕ := 40
def initial_apples_required : ℕ := 3
def apples_per_pie : ℕ := 5

theorem rosie_pie_count : (total_apples - initial_apples_required) / apples_per_pie = 7 :=
by
  sorry

end rosie_pie_count_l80_80834


namespace find_angle_B_max_value_a_squared_plus_c_squared_l80_80203

variable {A B C : ℝ} -- Angles A, B, C in radians
variable {a b c : ℝ} -- Sides opposite to these angles

-- Problem 1
theorem find_angle_B (h : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos B) : B = Real.pi / 3 :=
by
  sorry -- Proof is not needed

-- Problem 2
theorem max_value_a_squared_plus_c_squared (h : b = Real.sqrt 3)
  (h' : b * Real.cos C + c * Real.cos B = 2 * a * Real.cos B) : (a^2 + c^2) ≤ 6 :=
by
  sorry -- Proof is not needed

end find_angle_B_max_value_a_squared_plus_c_squared_l80_80203


namespace sewers_handle_rain_l80_80606

theorem sewers_handle_rain (total_capacity : ℕ) (runoff_per_hour : ℕ) : 
  total_capacity = 240000 → 
  runoff_per_hour = 1000 → 
  total_capacity / runoff_per_hour / 24 = 10 :=
by 
  intro h1 h2
  sorry

end sewers_handle_rain_l80_80606


namespace product_with_zero_is_zero_l80_80143

theorem product_with_zero_is_zero :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 0) = 0 :=
by
  sorry

end product_with_zero_is_zero_l80_80143


namespace factorize_polynomial_find_value_l80_80363

-- Problem 1: Factorize a^3 - 3a^2 - 4a + 12
theorem factorize_polynomial (a : ℝ) :
  a^3 - 3 * a^2 - 4 * a + 12 = (a - 3) * (a - 2) * (a + 2) :=
sorry

-- Problem 2: Given m + n = 5 and m - n = 1, prove m^2 - n^2 + 2m - 2n = 7
theorem find_value (m n : ℝ) (h1 : m + n = 5) (h2 : m - n = 1) :
  m^2 - n^2 + 2 * m - 2 * n = 7 :=
sorry

end factorize_polynomial_find_value_l80_80363


namespace wall_thickness_is_correct_l80_80303

-- Define the dimensions of the brick.
def brick_length : ℝ := 80
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the number of required bricks.
def num_bricks : ℝ := 2000

-- Define the dimensions of the wall.
def wall_length : ℝ := 800
def wall_height : ℝ := 600

-- The volume of one brick.
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- The volume of the wall.
def wall_volume (T : ℝ) : ℝ := wall_length * wall_height * T

-- The thickness of the wall to be proved.
theorem wall_thickness_is_correct (T_wall : ℝ) (h : num_bricks * brick_volume = wall_volume T_wall) : 
  T_wall = 22.5 :=
sorry

end wall_thickness_is_correct_l80_80303


namespace point_outside_circle_l80_80049

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), (a*x + b*y = 1 ∧ x^2 + y^2 = 1)) : a^2 + b^2 ≥ 1 :=
sorry

end point_outside_circle_l80_80049


namespace suitable_high_jump_athlete_l80_80279

structure Athlete where
  average : ℕ
  variance : ℝ

def A : Athlete := ⟨169, 6.0⟩
def B : Athlete := ⟨168, 17.3⟩
def C : Athlete := ⟨169, 5.0⟩
def D : Athlete := ⟨168, 19.5⟩

def isSuitableCandidate (athlete: Athlete) (average_threshold: ℕ) : Prop :=
  athlete.average = average_threshold

theorem suitable_high_jump_athlete : isSuitableCandidate C 169 ∧
  (∀ a, isSuitableCandidate a 169 → a.variance ≥ C.variance) := by
  sorry

end suitable_high_jump_athlete_l80_80279


namespace minimum_omega_l80_80125

theorem minimum_omega (ω : ℝ) (hω_pos : ω > 0)
  (f : ℝ → ℝ) (hf : ∀ x, f x = Real.sin (ω * x + Real.pi / 3))
  (C : ℝ → ℝ) (hC : ∀ x, C x = Real.sin (ω * (x + Real.pi / 2) + Real.pi / 3)) :
  (∀ x, C x = C (-x)) ↔ ω = 1 / 3 := by
sorry

end minimum_omega_l80_80125


namespace sofia_initial_floor_l80_80178

theorem sofia_initial_floor (x : ℤ) (h1 : x + 7 - 6 + 5 = 20) : x = 14 := 
sorry

end sofia_initial_floor_l80_80178


namespace inequality_abc_l80_80515

theorem inequality_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) : 
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (2 / (1 + a) + 2 / (1 + b) + 2 / (1 + c)) :=
by
  sorry

end inequality_abc_l80_80515


namespace red_candies_difference_l80_80202

def jar1_ratio_red : ℕ := 7
def jar1_ratio_yellow : ℕ := 3
def jar2_ratio_red : ℕ := 5
def jar2_ratio_yellow : ℕ := 4
def total_yellow : ℕ := 108

theorem red_candies_difference :
  ∀ (x y : ℚ), jar1_ratio_yellow * x + jar2_ratio_yellow * y = total_yellow ∧ jar1_ratio_red + jar1_ratio_yellow = jar2_ratio_red + jar2_ratio_yellow → 10 * x = 9 * y → 7 * x - 5 * y = 21 := 
by sorry

end red_candies_difference_l80_80202


namespace triangle_side_c_l80_80167

noncomputable def angle_B_eq_2A (A B : ℝ) := B = 2 * A
noncomputable def side_a_eq_1 (a : ℝ) := a = 1
noncomputable def side_b_eq_sqrt3 (b : ℝ) := b = Real.sqrt 3

noncomputable def find_side_c (A B C a b c : ℝ) :=
  angle_B_eq_2A A B ∧
  side_a_eq_1 a ∧
  side_b_eq_sqrt3 b →
  c = 2

theorem triangle_side_c (A B C a b c : ℝ) : find_side_c A B C a b c :=
by sorry

end triangle_side_c_l80_80167


namespace stratified_sampling_size_l80_80969

theorem stratified_sampling_size (a_ratio b_ratio c_ratio : ℕ) (total_items_A : ℕ) (h_ratio : a_ratio + b_ratio + c_ratio = 10)
  (h_A_ratio : a_ratio = 2) (h_B_ratio : b_ratio = 3) (h_C_ratio : c_ratio = 5) (items_A : total_items_A = 20) : 
  ∃ n : ℕ, n = total_items_A * 5 := 
by {
  -- The proof should go here. Since we only need the statement:
  sorry
}

end stratified_sampling_size_l80_80969


namespace complement_of_A_in_U_l80_80502

def U : Set ℤ := {x | -2 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {x | ∃ n : ℕ, (x = 2 * n ∧ n ≤ 3)}

theorem complement_of_A_in_U : (U \ A) = {-2, -1, 1, 3, 5} :=
by
  sorry

end complement_of_A_in_U_l80_80502


namespace same_number_of_acquaintances_l80_80960

theorem same_number_of_acquaintances (n : ℕ) (h : n ≥ 2) (acquaintances : Fin n → Fin n) :
  ∃ i j : Fin n, i ≠ j ∧ acquaintances i = acquaintances j :=
by
  -- Insert proof here
  sorry

end same_number_of_acquaintances_l80_80960


namespace inequality_proof_l80_80887

theorem inequality_proof (a b : ℝ) (h1 : b < 0) (h2 : 0 < a) : a - b > a + b :=
sorry

end inequality_proof_l80_80887


namespace charge_two_hours_l80_80063

def charge_first_hour (F A : ℝ) : Prop := F = A + 25
def total_charge_five_hours (F A : ℝ) : Prop := F + 4 * A = 250
def total_charge_two_hours (F A : ℝ) : Prop := F + A = 115

theorem charge_two_hours (F A : ℝ) 
  (h1 : charge_first_hour F A)
  (h2 : total_charge_five_hours F A) : 
  total_charge_two_hours F A :=
by
  sorry

end charge_two_hours_l80_80063


namespace circle_diameter_l80_80519

theorem circle_diameter (A : ℝ) (hA : A = 4 * π) : ∃ d : ℝ, d = 4 := by
  sorry

end circle_diameter_l80_80519


namespace tan_sum_of_angles_eq_neg_sqrt_three_l80_80672

theorem tan_sum_of_angles_eq_neg_sqrt_three 
  (A B C : ℝ)
  (h1 : B - A = C - B)
  (h2 : A + B + C = Real.pi) :
  Real.tan (A + C) = -Real.sqrt 3 :=
sorry

end tan_sum_of_angles_eq_neg_sqrt_three_l80_80672


namespace conor_work_times_per_week_l80_80841

-- Definitions for the conditions
def vegetables_per_day (eggplants carrots potatoes : ℕ) : ℕ :=
  eggplants + carrots + potatoes

def total_vegetables_per_week (days vegetables_per_day : ℕ) : ℕ :=
  days * vegetables_per_day

-- Theorem statement to be proven
theorem conor_work_times_per_week :
  let eggplants := 12
  let carrots := 9
  let potatoes := 8
  let weekly_total := 116
  vegetables_per_day eggplants carrots potatoes = 29 →
  total_vegetables_per_week 4 29 = 116 →
  4 = weekly_total / 29 :=
by
  intros _ _ h1 h2
  sorry

end conor_work_times_per_week_l80_80841


namespace belt_length_sufficient_l80_80025

theorem belt_length_sufficient (r O_1O_2 O_1O_3 O_3_plane : ℝ) 
(O_1O_2_eq : O_1O_2 = 12) (O_1O_3_eq : O_1O_3 = 10) (O_3_plane_eq : O_3_plane = 8) (r_eq : r = 2) : 
(∃ L₁ L₂, L₁ = 32 + 4 * Real.pi ∧ L₂ = 22 + 2 * Real.sqrt 97 + 4 * Real.pi ∧ 
L₁ ≠ 54 ∧ L₂ > 54) := 
by 
  sorry

end belt_length_sufficient_l80_80025


namespace common_difference_of_arithmetic_sequence_l80_80865

theorem common_difference_of_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a n = a 0 + n * d) (h2 : a 5 = 10) (h3 : a 10 = -5) : d = -3 := 
by 
  sorry

end common_difference_of_arithmetic_sequence_l80_80865


namespace changed_answers_percentage_l80_80036

variables (n : ℕ) (a b c d : ℕ)

theorem changed_answers_percentage (h1 : a + b + c + d = 100)
  (h2 : a + d + c = 50)
  (h3 : a + c = 60)
  (h4 : b + d = 40) :
  10 ≤ c + d ∧ c + d ≤ 90 :=
  by sorry

end changed_answers_percentage_l80_80036


namespace train_length_calculation_l80_80555

noncomputable def train_length (speed_km_hr : ℕ) (time_sec : ℕ) : ℝ :=
  (speed_km_hr * 1000 / 3600) * time_sec

theorem train_length_calculation :
  train_length 250 6 = 416.67 :=
by
  sorry

end train_length_calculation_l80_80555


namespace smallest_integer_in_range_l80_80583

theorem smallest_integer_in_range :
  ∃ (n : ℕ), n > 1 ∧ n % 3 = 2 ∧ n % 7 = 2 ∧ n % 8 = 2 ∧ 131 ≤ n ∧ n ≤ 170 :=
by
  sorry

end smallest_integer_in_range_l80_80583


namespace Maria_height_in_meters_l80_80752

theorem Maria_height_in_meters :
  let inch_to_cm := 2.54
  let cm_to_m := 0.01
  let height_in_inch := 54
  let height_in_cm := height_in_inch * inch_to_cm
  let height_in_m := height_in_cm * cm_to_m
  let rounded_height_in_m := Float.round (height_in_m * 1000) / 1000
  rounded_height_in_m = 1.372 := 
by
  sorry

end Maria_height_in_meters_l80_80752


namespace quadrilateral_area_is_6_l80_80808

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨3, 1⟩
def D : Point := ⟨5, 5⟩

def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

def quadrilateral_area (A B C D : Point) : ℝ :=
  area_triangle A B C + area_triangle A C D

theorem quadrilateral_area_is_6 : quadrilateral_area A B C D = 6 :=
  sorry

end quadrilateral_area_is_6_l80_80808


namespace no_such_integers_exist_l80_80013

theorem no_such_integers_exist :
  ¬ ∃ (a b : ℕ), a ≥ 1 ∧ b ≥ 1 ∧ ∃ k₁ k₂ : ℕ, (a^5 * b + 3 = k₁^3) ∧ (a * b^5 + 3 = k₂^3) :=
by
  sorry

end no_such_integers_exist_l80_80013


namespace determine_function_l80_80448

theorem determine_function (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / (x + 1)) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 → f x = x / (x + 1)) :=
by
  sorry

end determine_function_l80_80448


namespace sum_of_x_and_y_l80_80210

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 := 
sorry

end sum_of_x_and_y_l80_80210


namespace points_can_move_on_same_line_l80_80123

variable {A B C x y x' y' : ℝ}

def transform_x (x y : ℝ) : ℝ := 3 * x + 2 * y + 1
def transform_y (x y : ℝ) : ℝ := x + 4 * y - 3

noncomputable def points_on_same_line (A B C : ℝ) (x y : ℝ) : Prop :=
  A*x + B*y + C = 0 ∧
  A*(transform_x x y) + B*(transform_y x y) + C = 0

theorem points_can_move_on_same_line :
  ∃ (A B C : ℝ), ∀ (x y : ℝ), points_on_same_line A B C x y :=
sorry

end points_can_move_on_same_line_l80_80123


namespace triangle_side_ratio_impossible_triangle_side_ratio_impossible_2_triangle_side_ratio_impossible_3_l80_80569

theorem triangle_side_ratio_impossible (a b c : ℝ) (h₁ : a = b / 2) (h₂ : a = c / 3) : false :=
by
  sorry

theorem triangle_side_ratio_impossible_2 (a b c : ℝ) (h₁ : b = a / 2) (h₂ : b = c / 3) : false :=
by
  sorry

theorem triangle_side_ratio_impossible_3 (a b c : ℝ) (h₁ : c = a / 2) (h₂ : c = b / 3) : false :=
by
  sorry

end triangle_side_ratio_impossible_triangle_side_ratio_impossible_2_triangle_side_ratio_impossible_3_l80_80569


namespace isosceles_right_triangle_leg_hypotenuse_ratio_l80_80590

theorem isosceles_right_triangle_leg_hypotenuse_ratio (a d k : ℝ) 
  (h_iso : d = a * Real.sqrt 2)
  (h_ratio : k = a / d) : 
  k^2 = 1 / 2 := by sorry

end isosceles_right_triangle_leg_hypotenuse_ratio_l80_80590


namespace simplify_and_evaluate_l80_80487

theorem simplify_and_evaluate (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 2) (hx3 : x ≠ -2) (hx4 : x = -1) :
  (2 / (x^2 - 4)) / (1 / (x^2 - 2*x)) = -2 :=
by
  sorry

end simplify_and_evaluate_l80_80487


namespace range_of_alpha_minus_beta_l80_80284

variable (α β : ℝ)

theorem range_of_alpha_minus_beta (h1 : -90 < α) (h2 : α < β) (h3 : β < 90) : -180 < α - β ∧ α - β < 0 := 
by
  sorry

end range_of_alpha_minus_beta_l80_80284


namespace fraction_equation_solution_l80_80539

theorem fraction_equation_solution (x : ℝ) (h : x ≠ 3) : (2 - x) / (x - 3) + 3 = 2 / (3 - x) ↔ x = 5 / 2 := by
  sorry

end fraction_equation_solution_l80_80539


namespace find_m_b_l80_80047

theorem find_m_b (m b : ℚ) :
  (3 * m - 14 = 2) ∧ (m ^ 2 - 6 * m + 15 = b) →
  m = 16 / 3 ∧ b = 103 / 9 := by
  intro h
  rcases h with ⟨h1, h2⟩
  -- proof steps here
  sorry

end find_m_b_l80_80047


namespace minimum_handshakes_l80_80669

-- Definitions
def people : ℕ := 30
def handshakes_per_person : ℕ := 3

-- Theorem statement
theorem minimum_handshakes : (people * handshakes_per_person) / 2 = 45 :=
by
  sorry

end minimum_handshakes_l80_80669


namespace functional_eq_solution_l80_80002

theorem functional_eq_solution (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (m + f n) = f m + n) : ∀ n, f n = n := 
by
  sorry

end functional_eq_solution_l80_80002


namespace find_certain_number_l80_80155

-- Define the conditions
variable (m : ℕ)
variable (h_lcm : Nat.lcm 24 m = 48)
variable (h_gcd : Nat.gcd 24 m = 8)

-- State the theorem to prove
theorem find_certain_number (h_lcm : Nat.lcm 24 m = 48) (h_gcd : Nat.gcd 24 m = 8) : m = 16 :=
sorry

end find_certain_number_l80_80155


namespace goods_train_speed_is_52_l80_80197

def man_train_speed : ℕ := 60 -- speed of the man's train in km/h
def goods_train_length : ℕ := 280 -- length of the goods train in meters
def time_to_pass : ℕ := 9 -- time for the goods train to pass the man in seconds
def relative_speed_kmph : ℕ := (goods_train_length * 3600) / (time_to_pass * 1000) -- relative speed in km/h, calculated as (0.28 km / (9/3600) h)
def goods_train_speed : ℕ := relative_speed_kmph - man_train_speed -- speed of the goods train in km/h

theorem goods_train_speed_is_52 : goods_train_speed = 52 := by
  sorry

end goods_train_speed_is_52_l80_80197


namespace proof1_l80_80778

def prob1 : Prop :=
  (1 : ℝ) * (Real.sqrt 45 + Real.sqrt 18) - (Real.sqrt 8 - Real.sqrt 125) = 8 * Real.sqrt 5 + Real.sqrt 2

theorem proof1 : prob1 :=
by
  sorry

end proof1_l80_80778


namespace leonardo_nap_duration_l80_80981

theorem leonardo_nap_duration (h : (1 : ℝ) / 5 * 60 = 12) : (1 / 5 : ℝ) * 60 = 12 :=
by 
  exact h

end leonardo_nap_duration_l80_80981


namespace tic_tac_toe_tie_fraction_l80_80282

theorem tic_tac_toe_tie_fraction :
  let amys_win : ℚ := 5 / 12
  let lilys_win : ℚ := 1 / 4
  1 - (amys_win + lilys_win) = 1 / 3 :=
by
  sorry

end tic_tac_toe_tie_fraction_l80_80282


namespace find_a_b_l80_80320

theorem find_a_b (a b : ℤ) (h : ({a, 0, -1} : Set ℤ) = {4, b, 0}) : a = 4 ∧ b = -1 := by
  sorry

end find_a_b_l80_80320


namespace number_of_correct_answers_is_95_l80_80656

variable (x y : ℕ) -- Define x as the number of correct answers and y as the number of wrong answers

-- Define the conditions
axiom h1 : x + y = 150
axiom h2 : 5 * x - 2 * y = 370

-- State the goal we want to prove
theorem number_of_correct_answers_is_95 : x = 95 :=
by
  sorry

end number_of_correct_answers_is_95_l80_80656


namespace addition_subtraction_questions_l80_80794

theorem addition_subtraction_questions (total_questions word_problems answered_questions add_sub_questions : ℕ)
  (h1 : total_questions = 45)
  (h2 : word_problems = 17)
  (h3 : answered_questions = total_questions - 7)
  (h4 : add_sub_questions = answered_questions - word_problems) : 
  add_sub_questions = 21 := 
by 
  -- the proof steps are skipped
  sorry

end addition_subtraction_questions_l80_80794


namespace largest_number_is_correct_l80_80055

theorem largest_number_is_correct (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 3) : c = 33.25 :=
by
  sorry

end largest_number_is_correct_l80_80055


namespace expenditure_of_neg_50_l80_80935

/-- In the book "Nine Chapters on the Mathematical Art," it is noted that
"when two calculations have opposite meanings, they should be named positive
and negative." This means: if an income of $80 is denoted as $+80, then $-50
represents an expenditure of $50. -/
theorem expenditure_of_neg_50 :
  (∀ (income : ℤ), income = 80 → -income = -50 → ∃ (expenditure : ℤ), expenditure = 50) := sorry

end expenditure_of_neg_50_l80_80935


namespace correct_calculation_l80_80051

-- Definitions of the conditions
def condition_A (a : ℝ) : Prop := a^2 + a^2 = a^4
def condition_B (a : ℝ) : Prop := 3 * a^2 + 2 * a^2 = 5 * a^2
def condition_C (a : ℝ) : Prop := a^4 - a^2 = a^2
def condition_D (a : ℝ) : Prop := 3 * a^2 - 2 * a^2 = 1

-- The theorem statement
theorem correct_calculation (a : ℝ) : condition_B a := by 
sorry

end correct_calculation_l80_80051


namespace incenter_x_coordinate_eq_l80_80428

theorem incenter_x_coordinate_eq (x y : ℝ) :
  (x = y) ∧ 
  (y = -x + 3) → 
  x = 3 / 2 := 
sorry

end incenter_x_coordinate_eq_l80_80428


namespace gcd_min_value_l80_80484

-- Definitions of the conditions
def is_positive_integer (x : ℕ) := x > 0

def gcd_cond (m n : ℕ) := Nat.gcd m n = 18

-- The main theorem statement
theorem gcd_min_value (m n : ℕ) (hm : is_positive_integer m) (hn : is_positive_integer n) (hgcd : gcd_cond m n) : 
  Nat.gcd (12 * m) (20 * n) = 72 :=
sorry

end gcd_min_value_l80_80484


namespace product_of_two_integers_l80_80098

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 18) (h2 : x^2 - y^2 = 36) : x * y = 80 :=
by
  sorry

end product_of_two_integers_l80_80098


namespace count_correct_statements_l80_80589

theorem count_correct_statements :
  ∃ (M: ℚ) (M1: ℚ) (M2: ℚ) (M3: ℚ) (M4: ℚ)
    (a b c d e : ℚ) (hacb : c ≠ 0) (habc: a ≠ 0) (hbcb : b ≠ 0) (hdcb: d ≠ 0) (hec: e ≠ 0),
  M = (ac + bd - ce) / c 
  ∧ M1 = (-bc - ad - ce) / c 
  ∧ M2 = (-dc - ab - ce) / c 
  ∧ M3 = (-dc - ab - de) / d 
  ∧ M4 = (ce - bd - ac) / (-c)
  ∧ M4 = M
  ∧ (M ≠ M3)
  ∧ (∀ M1, M1 = (-bc - ad - ce) / c → ((a = c ∨ b = d) ↔ b = d))
  ∧ (M4 = (ac + bd - ce)/c) :=
sorry

end count_correct_statements_l80_80589


namespace correct_mark_proof_l80_80675

-- Define the conditions
def wrong_mark := 85
def increase_in_average : ℝ := 0.5
def number_of_pupils : ℕ := 104

-- Define the correct mark to be proven
noncomputable def correct_mark : ℕ := 33

-- Statement to be proven
theorem correct_mark_proof (x : ℝ) :
  (wrong_mark - x) / number_of_pupils = increase_in_average → x = correct_mark :=
by
  sorry

end correct_mark_proof_l80_80675


namespace frances_towel_weight_in_ounces_l80_80104

theorem frances_towel_weight_in_ounces :
  (∀ Mary_towels Frances_towels : ℕ,
    Mary_towels = 4 * Frances_towels →
    Mary_towels = 24 →
    (Mary_towels + Frances_towels) * 2 = 60 →
    Frances_towels * 2 * 16 = 192) :=
by
  intros Mary_towels Frances_towels h1 h2 h3
  sorry

end frances_towel_weight_in_ounces_l80_80104


namespace contrapositive_negation_l80_80082

-- Define the main condition of the problem
def statement_p (x y : ℝ) : Prop :=
  (x - 1) * (y + 2) = 0 → (x = 1 ∨ y = -2)

-- Prove the contrapositive of statement_p
theorem contrapositive (x y : ℝ) : 
  (x ≠ 1 ∧ y ≠ -2) → ¬ ((x - 1) * (y + 2) = 0) :=
by 
  sorry

-- Prove the negation of statement_p
theorem negation (x y : ℝ) : 
  ((x - 1) * (y + 2) = 0) → ¬ (x = 1 ∨ y = -2) :=
by 
  sorry

end contrapositive_negation_l80_80082


namespace total_balls_l80_80472

theorem total_balls {balls_per_box boxes : ℕ} (h1 : balls_per_box = 3) (h2 : boxes = 2) : balls_per_box * boxes = 6 :=
by
  sorry

end total_balls_l80_80472


namespace arithmetic_sequence_50th_term_l80_80558

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 4
  let a_n (n : ℕ) := a_1 + (n - 1) * d
  a_n 50 = 199 :=
by
  sorry

end arithmetic_sequence_50th_term_l80_80558


namespace work_completion_days_l80_80919

theorem work_completion_days
  (E_q : ℝ) -- Efficiency of q
  (E_p : ℝ) -- Efficiency of p
  (E_r : ℝ) -- Efficiency of r
  (W : ℝ)  -- Total work
  (H1 : E_p = 1.5 * E_q) -- Condition 1
  (H2 : W = E_p * 25) -- Condition 2
  (H3 : E_r = 0.8 * E_q) -- Condition 3
  : (W / (E_p + E_q + E_r)) = 11.36 := -- Prove the days_needed is 11.36
by
  sorry

end work_completion_days_l80_80919


namespace part1_part2_l80_80364

-- Let's define the arithmetic sequence and conditions
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + d * (n - 1)
def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables (a1 a4 a3 a5 : ℕ)
variable (d : ℕ)

-- Additional conditions for the problem  
axiom h1 : a1 = 2
axiom h2 : a4 = 8
axiom h3 : arithmetic_seq a1 d 3 + arithmetic_seq a1 d 5 = a4 + 8

-- Define S7
def S7 : ℕ := sum_arithmetic_seq a1 d 7

-- Part I: Prove S7 = 56
theorem part1 : S7 = 56 := 
by
  sorry

-- Part II: Prove k = 2 given additional conditions
variable (k : ℕ)

-- Given that a_3, a_{k+1}, S_k are a geometric sequence
def is_geom_seq (a b s : ℕ) : Prop := b*b = a * s

axiom h4 : a3 = arithmetic_seq a1 d 3
axiom h5 : ∃ k, 0 < k ∧ is_geom_seq a3 (arithmetic_seq a1 d (k + 1)) (sum_arithmetic_seq a1 d k)

theorem part2 : ∃ k, 0 < k ∧ k = 2 := 
by
  sorry

end part1_part2_l80_80364


namespace range_of_m_l80_80522

variable (f : ℝ → ℝ) (m : ℝ)

-- Given conditions
def condition1 := ∀ x, f (-x) = -f x -- f(x) is an odd function
def condition2 := ∀ x, f (x + 3) = f x -- f(x) has a minimum positive period of 3
def condition3 := f 2015 > 1 -- f(2015) > 1
def condition4 := f 1 = (2 * m + 3) / (m - 1) -- f(1) = (2m + 3) / (m - 1)

-- We aim to prove that -2/3 < m < 1 given these conditions.
theorem range_of_m : condition1 f → condition2 f → condition3 f → condition4 f m → -2 / 3 < m ∧ m < 1 := by
  intros
  sorry

end range_of_m_l80_80522


namespace proof_inequality_l80_80142

noncomputable def proof_problem (p q : ℝ) (m n : ℕ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) : Prop :=
  (1 - p^m)^n + (1 - q^n)^m ≥ 1

theorem proof_inequality (p q : ℝ) (m n : ℕ) (hpq : p + q = 1) (hp : 0 < p) (hq : 0 < q) :
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 :=
by
  sorry

end proof_inequality_l80_80142


namespace rabbit_fraction_l80_80557

theorem rabbit_fraction
  (initial_rabbits : ℕ) (added_rabbits : ℕ) (total_rabbits_seen : ℕ)
  (h_initial : initial_rabbits = 13)
  (h_added : added_rabbits = 7)
  (h_seen : total_rabbits_seen = 60) :
  (initial_rabbits + added_rabbits) / total_rabbits_seen = 1 / 3 :=
by
  -- we will prove this
  sorry

end rabbit_fraction_l80_80557


namespace ramsey_theorem_six_people_l80_80499

theorem ramsey_theorem_six_people (S : Finset Person)
  (hS: S.card = 6)
  (R : Person → Person → Prop): 
  (∃ (has_relation : Person → Person → Prop), 
    ∀ A B : Person, A ≠ B → R A B ∨ ¬ R A B) →
  (∃ (T : Finset Person), T.card = 3 ∧ 
    ((∀ x y : Person, x ∈ T → y ∈ T → x ≠ y → R x y) ∨ 
     (∀ x y : Person, x ∈ T → y ∈ T → x ≠ y → ¬ R x y))) :=
by
  sorry

end ramsey_theorem_six_people_l80_80499


namespace tangent_circle_distance_proof_l80_80417

noncomputable def tangent_circle_distance (R r : ℝ) (tangent_type : String) : ℝ :=
  if tangent_type = "external" then R + r else R - r

theorem tangent_circle_distance_proof (R r : ℝ) (tangent_type : String) (hR : R = 4) (hr : r = 3) :
  tangent_circle_distance R r tangent_type = 7 ∨ tangent_circle_distance R r tangent_type = 1 := by
  sorry

end tangent_circle_distance_proof_l80_80417


namespace cannot_be_sum_of_consecutive_nat_iff_power_of_two_l80_80889

theorem cannot_be_sum_of_consecutive_nat_iff_power_of_two (n : ℕ) : 
  (∀ a b : ℕ, n ≠ (b - a + 1) * (a + b) / 2) ↔ (∃ k : ℕ, n = 2 ^ k) := by
  sorry

end cannot_be_sum_of_consecutive_nat_iff_power_of_two_l80_80889


namespace function_values_l80_80998

noncomputable def f (a b c x : ℝ) : ℝ := a * Real.cos x + b * x^2 + c

theorem function_values (a b c : ℝ) : 
  f a b c 1 = 1 ∧ f a b c (-1) = 1 := 
by
  sorry

end function_values_l80_80998


namespace false_statement_D_l80_80893

theorem false_statement_D :
  ¬ (∀ {α β : ℝ}, α = β → (true → true → true → α = β ↔ α = β)) :=
by
  sorry

end false_statement_D_l80_80893


namespace probability_odd_3_in_6_rolls_l80_80506

-- Definitions based on problem conditions
def probability_of_odd (outcome: ℕ) : ℚ := if outcome % 2 = 1 then 1/2 else 0 

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ := 
  ((Nat.choose n k : ℚ) * (p^k) * ((1 - p)^(n - k)))

-- Given problem
theorem probability_odd_3_in_6_rolls : 
  binomial_probability 6 3 (1/2) = 5 / 16 :=
by
  sorry

end probability_odd_3_in_6_rolls_l80_80506


namespace solve_inequality_l80_80068

theorem solve_inequality (x : ℝ) : 3 * x^2 + 7 * x + 2 < 0 ↔ -1 < x ∧ x < -2/3 := by
  sorry

end solve_inequality_l80_80068


namespace boat_speed_l80_80772

theorem boat_speed (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 7) : b = 9 :=
by
  sorry

end boat_speed_l80_80772


namespace triangle_PQR_min_perimeter_l80_80061

theorem triangle_PQR_min_perimeter (PQ PR QR : ℕ) (QJ : ℕ) 
  (hPQ_PR : PQ = PR) (hQJ_10 : QJ = 10) (h_pos_QR : 0 < QR) :
  QR * 2 + PQ * 2 = 96 :=
  sorry

end triangle_PQR_min_perimeter_l80_80061


namespace range_of_m_l80_80406

namespace MathProof

def A : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }
def B (m : ℝ) : Set ℝ := { x | x^2 - m * x + m - 1 = 0 }

theorem range_of_m (m : ℝ) (h : A ∪ (B m) = A) : m = 3 :=
  sorry

end MathProof

end range_of_m_l80_80406


namespace sequence_general_term_l80_80507

open Nat

/-- Define the sequence recursively -/
def a : ℕ → ℤ
| 0     => -1
| (n+1) => 3 * a n - 1

/-- The general term of the sequence is given by - (3^n - 1) / 2 -/
theorem sequence_general_term (n : ℕ) : a n = - (3^n - 1) / 2 := 
by
  sorry

end sequence_general_term_l80_80507


namespace sum_of_all_two_digit_numbers_l80_80793

theorem sum_of_all_two_digit_numbers : 
  let digits := [0, 1, 2, 3, 4, 5]
  let tens_digits := [1, 2, 3, 4, 5]
  let num_ones_digits := digits.length
  let num_tens_digits := tens_digits.length
  let sum_tens_place := 10 * (tens_digits.sum) * num_ones_digits
  let sum_ones_place := (digits.sum) * num_tens_digits
  sum_tens_place + sum_ones_place = 975 :=
by 
  let digits := [0, 1, 2, 3, 4, 5]
  let tens_digits := [1, 2, 3, 4, 5]
  let num_ones_digits := digits.length
  let num_tens_digits := tens_digits.length
  let sum_tens_place := 10 * (tens_digits.sum) * num_ones_digits
  let sum_ones_place := (digits.sum) * num_tens_digits
  show sum_tens_place + sum_ones_place = 975
  sorry

end sum_of_all_two_digit_numbers_l80_80793


namespace white_surface_fraction_l80_80587

-- Definition of the problem conditions
def larger_cube_surface_area : ℕ := 54
def white_cubes : ℕ := 6
def white_surface_area_minimized : ℕ := 5

-- Theorem statement proving the fraction of white surface area
theorem white_surface_fraction : (white_surface_area_minimized / larger_cube_surface_area : ℚ) = 5 / 54 := 
by
  sorry

end white_surface_fraction_l80_80587


namespace sum_of_fractions_eq_decimal_l80_80253

theorem sum_of_fractions_eq_decimal :
  (3 / 100) + (5 / 1000) + (7 / 10000) = 0.0357 :=
by
  sorry

end sum_of_fractions_eq_decimal_l80_80253


namespace probability_of_both_selected_l80_80245

variable (P_ram : ℚ) (P_ravi : ℚ) (P_both : ℚ)

def selection_probability (P_ram : ℚ) (P_ravi : ℚ) : ℚ :=
  P_ram * P_ravi

theorem probability_of_both_selected (h1 : P_ram = 3/7) (h2 : P_ravi = 1/5) :
  selection_probability P_ram P_ravi = P_both :=
by
  sorry

end probability_of_both_selected_l80_80245


namespace impossible_transformation_l80_80523

def f (x : ℝ) := x^2 + 5 * x + 4
def g (x : ℝ) := x^2 + 10 * x + 8

theorem impossible_transformation :
  (∀ x, f (x) = x^2 + 5 * x + 4) →
  (∀ x, g (x) = x^2 + 10 * x + 8) →
  (¬ ∃ t : ℝ → ℝ → ℝ, (∀ x, t (f x) x = g x)) :=
by
  sorry

end impossible_transformation_l80_80523


namespace evaluate_six_applications_problem_solution_l80_80937

def r (θ : ℚ) : ℚ := 1 / (1 + θ)

theorem evaluate_six_applications (θ : ℚ) : 
  r (r (r (r (r (r θ))))) = (8 + 5 * θ) / (13 + 8 * θ) :=
sorry

theorem problem_solution : r (r (r (r (r (r 30))))) = 158 / 253 :=
by
  have h : r (r (r (r (r (r 30))))) = (8 + 5 * 30) / (13 + 8 * 30) := by
    exact evaluate_six_applications 30
  rw [h]
  norm_num

end evaluate_six_applications_problem_solution_l80_80937


namespace calc_fraction_l80_80580

theorem calc_fraction : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end calc_fraction_l80_80580


namespace convert_spherical_to_rectangular_l80_80264

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 15 (3 * Real.pi / 4) (Real.pi / 2) = 
    (-15 * Real.sqrt 2 / 2, 15 * Real.sqrt 2 / 2, 0) :=
by 
  sorry

end convert_spherical_to_rectangular_l80_80264


namespace Kate_has_223_pennies_l80_80015

-- Definition of the conditions
variables (J K : ℕ)
variable (h1 : J = 388)
variable (h2 : J = K + 165)

-- Prove the question equals the answer
theorem Kate_has_223_pennies : K = 223 :=
by
  sorry

end Kate_has_223_pennies_l80_80015


namespace minimum_value_of_a_plus_2b_l80_80402

theorem minimum_value_of_a_plus_2b 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : 2 * a + b = a * b - 1) 
  : a + 2 * b = 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_of_a_plus_2b_l80_80402


namespace geometric_series_3000_terms_sum_l80_80177

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ := a * (1 - r^n) / (1 - r)

theorem geometric_series_3000_terms_sum
    (a r : ℝ)
    (h_r : r ≠ 1)
    (sum_1000 : geometric_sum a r 1000 = 500)
    (sum_2000 : geometric_sum a r 2000 = 950) :
  geometric_sum a r 3000 = 1355 :=
by 
  sorry

end geometric_series_3000_terms_sum_l80_80177


namespace measure_of_smaller_angle_l80_80415

noncomputable def complementary_angle_ratio_smaller (x : ℝ) (h : 4 * x + x = 90) : ℝ :=
x

theorem measure_of_smaller_angle (x : ℝ) (h : 4 * x + x = 90) : complementary_angle_ratio_smaller x h = 18 :=
sorry

end measure_of_smaller_angle_l80_80415


namespace combined_stripes_is_22_l80_80152

-- Definition of stripes per shoe for each person based on the conditions
def stripes_per_shoe_Olga : ℕ := 3
def stripes_per_shoe_Rick : ℕ := stripes_per_shoe_Olga - 1
def stripes_per_shoe_Hortense : ℕ := stripes_per_shoe_Olga * 2

-- The total combined number of stripes on all shoes for Olga, Rick, and Hortense
def total_stripes : ℕ := 2 * (stripes_per_shoe_Olga + stripes_per_shoe_Rick + stripes_per_shoe_Hortense)

-- The statement to prove that the total number of stripes on all their shoes is 22
theorem combined_stripes_is_22 : total_stripes = 22 :=
by
  sorry

end combined_stripes_is_22_l80_80152


namespace inequality_solution_l80_80137

open Set

theorem inequality_solution :
  {x : ℝ | |x + 1| - 2 > 0} = {x : ℝ | x < -3} ∪ {x : ℝ | x > 1} :=
by
  sorry

end inequality_solution_l80_80137


namespace f_at_2_l80_80818

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
  x ^ 2017 + a * x ^ 3 - b / x - 8

theorem f_at_2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by sorry

end f_at_2_l80_80818


namespace min_value_inequality_l80_80938

theorem min_value_inequality (a b c d e f : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
    (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_pos_e : 0 < e) (h_pos_f : 0 < f)
    (h_sum : a + b + c + d + e + f = 9) : 
    1 / a + 9 / b + 16 / c + 25 / d + 36 / e + 49 / f ≥ 676 / 9 := 
by 
  sorry

end min_value_inequality_l80_80938


namespace selling_price_correct_l80_80056

-- Define the conditions
def cost_per_cupcake : ℝ := 0.75
def total_cupcakes_burnt : ℕ := 24
def total_eaten_first : ℕ := 5
def total_eaten_later : ℕ := 4
def net_profit : ℝ := 24
def total_cupcakes_made : ℕ := 72
def total_cost : ℝ := total_cupcakes_made * cost_per_cupcake
def total_eaten : ℕ := total_eaten_first + total_eaten_later
def total_sold : ℕ := total_cupcakes_made - total_eaten
def revenue (P : ℝ) : ℝ := total_sold * P

-- Prove the correctness of the selling price P
theorem selling_price_correct : 
  ∃ P : ℝ, revenue P - total_cost = net_profit ∧ (P = 1.24) :=
by
  sorry

end selling_price_correct_l80_80056


namespace partitions_equal_l80_80246

namespace MathProof

-- Define the set of natural numbers
def nat := ℕ

-- Define the partition functions (placeholders)
def num_distinct_partitions (n : nat) : nat := sorry
def num_odd_partitions (n : nat) : nat := sorry

-- Statement of the theorem
theorem partitions_equal (n : nat) : 
  num_distinct_partitions n = num_odd_partitions n :=
sorry

end MathProof

end partitions_equal_l80_80246


namespace age_product_difference_l80_80709

theorem age_product_difference (age_today : ℕ) (product_today : ℕ) (product_next_year : ℕ) :
  age_today = 7 →
  product_today = age_today * age_today →
  product_next_year = (age_today + 1) * (age_today + 1) →
  product_next_year - product_today = 15 :=
by
  sorry

end age_product_difference_l80_80709


namespace roots_of_quadratic_l80_80560

theorem roots_of_quadratic (p q : ℝ) (h1 : 3 * p^2 + 9 * p - 21 = 0) (h2 : 3 * q^2 + 9 * q - 21 = 0) :
  (3 * p - 4) * (6 * q - 8) = 122 := by
  -- We don't need to provide the proof here, only the statement
  sorry

end roots_of_quadratic_l80_80560


namespace rational_expression_equals_3_l80_80011

theorem rational_expression_equals_3 (x : ℝ) (hx : x^3 + x - 1 = 0) :
  (x^4 - 2*x^3 + x^2 - 3*x + 5) / (x^5 - x^2 - x + 2) = 3 := 
by
  sorry

end rational_expression_equals_3_l80_80011


namespace inlet_rate_480_l80_80110

theorem inlet_rate_480 (capacity : ℕ) (T_outlet : ℕ) (T_outlet_inlet : ℕ) (R_i : ℕ) :
  capacity = 11520 →
  T_outlet = 8 →
  T_outlet_inlet = 12 →
  R_i = 480 :=
by
  intros
  sorry

end inlet_rate_480_l80_80110


namespace proof_problem_l80_80394

variables (x y b z a : ℝ)

def condition1 : Prop := x * y + x^2 = b
def condition2 : Prop := (1 / x^2) - (1 / y^2) = a
def z_def : Prop := z = x + y

theorem proof_problem (x y b z a : ℝ) (h1 : condition1 x y b) (h2 : condition2 x y a) (hz : z_def x y z) : (x + y) ^ 2 = z ^ 2 :=
by {
  sorry
}

end proof_problem_l80_80394


namespace cos_diff_proof_l80_80367

theorem cos_diff_proof (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2)
  (h2 : Real.cos A + Real.cos B = 1) :
  Real.cos (A - B) = 5 / 8 := 
by
  sorry

end cos_diff_proof_l80_80367


namespace points_on_same_circle_l80_80358
open Real

theorem points_on_same_circle (m : ℝ) :
  ∃ D E F, 
  (2^2 + 1^2 + 2 * D + 1 * E + F = 0) ∧
  (4^2 + 2^2 + 4 * D + 2 * E + F = 0) ∧
  (3^2 + 4^2 + 3 * D + 4 * E + F = 0) ∧
  (1^2 + m^2 + 1 * D + m * E + F = 0) →
  (m = 2 ∨ m = 3) := 
sorry

end points_on_same_circle_l80_80358


namespace range_of_m_l80_80701

theorem range_of_m (x m : ℝ) (h₀ : -2 ≤ x ∧ x ≤ 11)
  (h₁ : 1 - 3 * m ≤ x ∧ x ≤ 3 + m)
  (h₂ : ¬ (-2 ≤ x ∧ x ≤ 11) → ¬ (1 - 3 * m ≤ x ∧ x ≤ 3 + m)) :
  m ≥ 8 :=
by
  sorry

end range_of_m_l80_80701


namespace find_a_l80_80306

noncomputable def A (a : ℝ) : Set ℝ := {2^a, 3}
def B : Set ℝ := {2, 3}
def C : Set ℝ := {1, 2, 3}

theorem find_a (a : ℝ) (h : A a ∪ B = C) : a = 0 :=
sorry

end find_a_l80_80306


namespace length_of_PQ_is_8_l80_80829

-- Define the lengths of the sides and conditions
variables (PQ QR PS SR : ℕ) (perimeter : ℕ)

-- State the conditions
def conditions : Prop :=
  SR = 16 ∧
  perimeter = 40 ∧
  PQ = QR ∧ QR = PS

-- State the goal
theorem length_of_PQ_is_8 (h : conditions PQ QR PS SR perimeter) : PQ = 8 :=
sorry

end length_of_PQ_is_8_l80_80829


namespace basketball_game_total_points_l80_80266

theorem basketball_game_total_points :
  ∃ (a d b: ℕ) (r: ℝ), 
      a = b + 2 ∧     -- Eagles lead by 2 points at the end of the first quarter
      (a + d < 100) ∧ -- Points scored by Eagles in each quarter form an increasing arithmetic sequence
      (b * r < 100) ∧ -- Points scored by Lions in each quarter form an increasing geometric sequence
      (a + (a + d) + (a + 2 * d)) = b * (1 + r + r^2) ∧ -- Aggregate score tied at the end of the third quarter
      (a + (a + d) + (a + 2 * d) + (a + 3 * d) + b * (1 + r + r^2 + r^3) = 144) -- Total points scored by both teams 
   :=
sorry

end basketball_game_total_points_l80_80266


namespace correct_expression_l80_80949

theorem correct_expression (a : ℝ) :
  (a^3 * a^2 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ ¬((a - 1)^2 = a^2 - 1) :=
by
  sorry

end correct_expression_l80_80949


namespace log_base_3_domain_is_minus_infinity_to_3_l80_80570

noncomputable def log_base_3_domain (x : ℝ) : Prop :=
  3 - x > 0

theorem log_base_3_domain_is_minus_infinity_to_3 :
  ∀ x : ℝ, log_base_3_domain x ↔ x < 3 :=
by
  sorry

end log_base_3_domain_is_minus_infinity_to_3_l80_80570


namespace range_of_m_plus_n_l80_80321

noncomputable def f (m n x : ℝ) : ℝ := m * 2^x + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x : ℝ, f m n x = 0 ∧ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end range_of_m_plus_n_l80_80321


namespace highway_total_vehicles_l80_80936

theorem highway_total_vehicles (num_trucks : ℕ) (num_cars : ℕ) (total_vehicles : ℕ)
  (h1 : num_trucks = 100)
  (h2 : num_cars = 2 * num_trucks)
  (h3 : total_vehicles = num_cars + num_trucks) :
  total_vehicles = 300 :=
by
  sorry

end highway_total_vehicles_l80_80936


namespace lizas_final_balance_l80_80845

-- Define the initial condition and subsequent changes
def initial_balance : ℕ := 800
def rent_payment : ℕ := 450
def paycheck_deposit : ℕ := 1500
def electricity_bill : ℕ := 117
def internet_bill : ℕ := 100
def phone_bill : ℕ := 70

-- Calculate the final balance
def final_balance : ℕ :=
  let balance_after_rent := initial_balance - rent_payment
  let balance_after_paycheck := balance_after_rent + paycheck_deposit
  let balance_after_bills := balance_after_paycheck - (electricity_bill + internet_bill)
  balance_after_bills - phone_bill

-- Theorem to prove that the final balance is 1563
theorem lizas_final_balance : final_balance = 1563 :=
by
  sorry

end lizas_final_balance_l80_80845


namespace geometric_series_first_term_l80_80928

theorem geometric_series_first_term (a : ℕ) (r : ℚ) (S : ℕ) (h_r : r = 1 / 4) (h_S : S = 40) (h_sum : S = a / (1 - r)) : a = 30 := sorry

end geometric_series_first_term_l80_80928


namespace stadium_revenue_difference_l80_80827

theorem stadium_revenue_difference :
  let total_capacity := 2000
  let vip_capacity := 200
  let standard_capacity := 1000
  let general_capacity := 800
  let vip_price := 50
  let standard_price := 30
  let general_price := 20
  let three_quarters (n : ℕ) := (3 * n) / 4
  let three_quarter_full := three_quarters total_capacity
  let vip_three_quarter := three_quarters vip_capacity
  let standard_three_quarter := three_quarters standard_capacity
  let general_three_quarter := three_quarters general_capacity
  let revenue_three_quarter := vip_three_quarter * vip_price + standard_three_quarter * standard_price + general_three_quarter * general_price
  let revenue_full := vip_capacity * vip_price + standard_capacity * standard_price + general_capacity * general_price
  revenue_three_quarter = 42000 ∧ (revenue_full - revenue_three_quarter) = 14000 :=
by
  sorry

end stadium_revenue_difference_l80_80827


namespace more_girls_than_boys_l80_80842

theorem more_girls_than_boys (total students : ℕ) (girls boys : ℕ) (h1 : total = 41) (h2 : girls = 22) (h3 : girls + boys = total) : (girls - boys) = 3 :=
by
  sorry

end more_girls_than_boys_l80_80842


namespace min_value_of_squares_l80_80898

theorem min_value_of_squares (a b t : ℝ) (h : a + b = t) : (a^2 + b^2) ≥ t^2 / 2 := 
by
  sorry

end min_value_of_squares_l80_80898


namespace solve_x_l80_80517

theorem solve_x (x : ℝ) (h : x^2 + 6 * x + 8 = -(x + 4) * (x + 6)) : 
  x = -4 := 
by
  sorry

end solve_x_l80_80517


namespace geometric_sequence_first_term_l80_80856

variable (a y z : ℕ)
variable (r : ℕ)
variable (h₁ : 16 = a * r^2)
variable (h₂ : 128 = a * r^4)

theorem geometric_sequence_first_term 
  (h₃ : r = 2) : a = 4 :=
by
  sorry

end geometric_sequence_first_term_l80_80856


namespace discount_price_l80_80309

theorem discount_price (P P_d : ℝ) 
  (h1 : P_d = 0.85 * P) 
  (P_final : ℝ) 
  (h2 : P_final = 1.25 * P_d) 
  (h3 : P - P_final = 5.25) :
  P_d = 71.4 :=
by
  sorry

end discount_price_l80_80309


namespace no_parallelepiped_exists_l80_80005

theorem no_parallelepiped_exists 
  (xyz_half_volume: ℝ)
  (xy_plus_yz_plus_zx_half_surface_area: ℝ) 
  (sum_of_squares_eq_4: ℝ) : 
  ¬(∃ x y z : ℝ, (x * y * z = xyz_half_volume) ∧ 
                 (x * y + y * z + z * x = xy_plus_yz_plus_zx_half_surface_area) ∧ 
                 (x^2 + y^2 + z^2 = sum_of_squares_eq_4)) := 
by
  let xyz_half_volume := 2 * Real.pi / 3
  let xy_plus_yz_plus_zx_half_surface_area := Real.pi
  let sum_of_squares_eq_4 := 4
  sorry

end no_parallelepiped_exists_l80_80005


namespace evaluate_expression_l80_80723

theorem evaluate_expression (x : ℝ) :
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 :=
by
  sorry

end evaluate_expression_l80_80723


namespace correct_calculation_l80_80399

theorem correct_calculation (a b : ℝ) : 
  (¬ (2 * (a - 1) = 2 * a - 1)) ∧ 
  (3 * a^2 - 2 * a^2 = a^2) ∧ 
  (¬ (3 * a^2 - 2 * a^2 = 1)) ∧ 
  (¬ (3 * a + 2 * b = 5 * a * b)) :=
by
  sorry

end correct_calculation_l80_80399


namespace gcd_228_1995_l80_80307

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end gcd_228_1995_l80_80307


namespace percentage_pine_cones_on_roof_l80_80768

theorem percentage_pine_cones_on_roof 
  (num_trees : Nat) 
  (pine_cones_per_tree : Nat) 
  (pine_cone_weight_oz : Nat) 
  (total_pine_cone_weight_on_roof_oz : Nat) 
  : num_trees = 8 ∧ pine_cones_per_tree = 200 ∧ pine_cone_weight_oz = 4 ∧ total_pine_cone_weight_on_roof_oz = 1920 →
    (total_pine_cone_weight_on_roof_oz / pine_cone_weight_oz) / (num_trees * pine_cones_per_tree) * 100 = 30 := 
by
  sorry

end percentage_pine_cones_on_roof_l80_80768


namespace tanC_over_tanA_plus_tanC_over_tanB_l80_80190

theorem tanC_over_tanA_plus_tanC_over_tanB {a b c : ℝ} (A B C : ℝ) (h : a / b + b / a = 6 * Real.cos C) (acute_triangle : A > 0 ∧ A < Real.pi / 2 ∧ B > 0 ∧ B < Real.pi / 2 ∧ C > 0 ∧ C < Real.pi / 2) :
  (Real.tan C / Real.tan A) + (Real.tan C / Real.tan B) = 4 :=
sorry -- Proof not required

end tanC_over_tanA_plus_tanC_over_tanB_l80_80190


namespace final_sale_price_l80_80434

theorem final_sale_price (P P₁ P₂ P₃ : ℝ) (d₁ d₂ d₃ dx : ℝ) (x : ℝ)
  (h₁ : P = 600) 
  (h_d₁ : d₁ = 20) (h_d₂ : d₂ = 15) (h_d₃ : d₃ = 10)
  (h₁₁ : P₁ = P * (1 - d₁ / 100))
  (h₁₂ : P₂ = P₁ * (1 - d₂ / 100))
  (h₁₃ : P₃ = P₂ * (1 - d₃ / 100))
  (h_P₃_final : P₃ = 367.2) :
  P₃ * (100 - dx) / 100 = 367.2 * (100 - x) / 100 :=
by
  sorry

end final_sale_price_l80_80434


namespace possible_six_digit_numbers_divisible_by_3_l80_80094

theorem possible_six_digit_numbers_divisible_by_3 (missing_digit_condition : ∀ k : Nat, (8 + 5 + 5 + 2 + 2 + k) % 3 = 0) : 
  ∃ count : Nat, count = 13 := by
  sorry

end possible_six_digit_numbers_divisible_by_3_l80_80094


namespace total_people_served_l80_80659

variable (total_people : ℕ)
variable (people_not_buy_coffee : ℕ := 10)

theorem total_people_served (H : (2 / 5 : ℚ) * total_people = people_not_buy_coffee) : total_people = 25 := 
by
  sorry

end total_people_served_l80_80659


namespace det_matrixE_l80_80748

def matrixE : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

theorem det_matrixE : (matrixE.det) = 25 := by
  sorry

end det_matrixE_l80_80748


namespace DF_is_5_point_5_l80_80946

variables {A B C D E F : Type}
variables (congruent : triangle A B C ≃ triangle D E F)
variables (ac_length : AC = 5.5)

theorem DF_is_5_point_5 : DF = 5.5 :=
by
  -- skipped proof
  sorry

end DF_is_5_point_5_l80_80946


namespace multiply_polynomials_l80_80497

theorem multiply_polynomials (x : ℝ) : 2 * x * (5 * x ^ 2) = 10 * x ^ 3 := by
  sorry

end multiply_polynomials_l80_80497


namespace fraction_exponentiation_l80_80112

theorem fraction_exponentiation : (3/4 : ℚ)^3 = 27/64 := by
  sorry

end fraction_exponentiation_l80_80112


namespace least_m_for_no_real_roots_l80_80985

theorem least_m_for_no_real_roots : ∃ (m : ℤ), (∀ (x : ℝ), 3 * x * (m * x + 6) - 2 * x^2 + 8 ≠ 0) ∧ m = 4 := 
sorry

end least_m_for_no_real_roots_l80_80985


namespace willie_exchange_rate_l80_80118

theorem willie_exchange_rate :
  let euros := 70
  let normal_exchange_rate := 1 / 5 -- euros per dollar
  let airport_exchange_rate := 5 / 7
  let dollars := euros * normal_exchange_rate * airport_exchange_rate
  dollars = 10 := by
  sorry

end willie_exchange_rate_l80_80118


namespace tangent_line_y_intercept_l80_80265

noncomputable def y_intercept_tangent_line (R1_center R2_center : ℝ × ℝ)
  (R1_radius R2_radius : ℝ) : ℝ :=
if R1_center = (3,0) ∧ R2_center = (8,0) ∧ R1_radius = 3 ∧ R2_radius = 2
then 15 * Real.sqrt 26 / 26
else 0

theorem tangent_line_y_intercept : 
  y_intercept_tangent_line (3,0) (8,0) 3 2 = 15 * Real.sqrt 26 / 26 :=
by
  -- proof goes here
  sorry

end tangent_line_y_intercept_l80_80265


namespace Jina_has_51_mascots_l80_80704

def teddies := 5
def bunnies := 3 * teddies
def koala_bear := 1
def additional_teddies := 2 * bunnies
def total_mascots := teddies + bunnies + koala_bear + additional_teddies

theorem Jina_has_51_mascots : total_mascots = 51 := by
  sorry

end Jina_has_51_mascots_l80_80704


namespace problem_part_1_problem_part_2_l80_80532

noncomputable def f (x : ℝ) (m : ℝ) := |x + 1| + |x - 2| - m

theorem problem_part_1 : 
  {x : ℝ | f x 5 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 3} :=
by sorry

theorem problem_part_2 (h : ∀ x : ℝ, f x m ≥ 2) : m ≤ 1 :=
by sorry

end problem_part_1_problem_part_2_l80_80532


namespace count_three_digit_multiples_of_35_l80_80199

theorem count_three_digit_multiples_of_35 : 
  ∃ n : ℕ, n = 26 ∧ ∀ x : ℕ, (100 ≤ x ∧ x < 1000) → (x % 35 = 0 → x = 35 * (3 + ((x / 35) - 3))) := 
sorry

end count_three_digit_multiples_of_35_l80_80199


namespace range_of_m_l80_80662

noncomputable def problem_statement
  (x y m : ℝ) : Prop :=
  (x - 2 * y + 5 ≥ 0) ∧
  (3 - x ≥ 0) ∧
  (x + y ≥ 0) ∧
  (m > 0)

theorem range_of_m (x y m : ℝ) :
  problem_statement x y m →
  ((∀ x y, problem_statement x y m → x^2 + y^2 ≤ m^2) ↔ m ≥ 3 * Real.sqrt 2) :=
by 
  intro h
  sorry

end range_of_m_l80_80662


namespace oranges_in_bowl_l80_80366

theorem oranges_in_bowl (bananas : Nat) (apples : Nat) (pears : Nat) (total_fruits : Nat) (h_bananas : bananas = 4) (h_apples : apples = 3 * bananas) (h_pears : pears = 5) (h_total_fruits : total_fruits = 30) :
  total_fruits - (bananas + apples + pears) = 9 :=
by
  subst h_bananas
  subst h_apples
  subst h_pears
  subst h_total_fruits
  sorry

end oranges_in_bowl_l80_80366


namespace first_term_geometric_series_l80_80677

theorem first_term_geometric_series (a r S : ℝ) (h1 : r = -1/3) (h2 : S = 9)
  (h3 : S = a / (1 - r)) : a = 12 :=
sorry

end first_term_geometric_series_l80_80677


namespace candy_cost_l80_80965

theorem candy_cost (C : ℝ) 
  (h1 : 20 + 40 = 60) 
  (h2 : 5 * 40 + 20 * C = 60 * 6) : 
  C = 8 :=
by
  sorry

end candy_cost_l80_80965


namespace sum_of_powers_of_minus_one_l80_80465

theorem sum_of_powers_of_minus_one : (-1) ^ 2010 + (-1) ^ 2011 + 1 ^ 2012 - 1 ^ 2013 + (-1) ^ 2014 = -1 := by
  sorry

end sum_of_powers_of_minus_one_l80_80465


namespace even_poly_iff_a_zero_l80_80351

theorem even_poly_iff_a_zero (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 3) = (x^2 - a*x + 3)) → a = 0 :=
by
  sorry

end even_poly_iff_a_zero_l80_80351


namespace arithmetic_sequence_ninth_term_l80_80678

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) :=
  (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2

theorem arithmetic_sequence_ninth_term
  (a: ℕ → ℕ)
  (h_arith: is_arithmetic_sequence a)
  (h_sum_5: sum_of_first_n_terms a 5 = 75)
  (h_a4: a 4 = 2 * a 2) :
  a 9 = 45 :=
sorry

end arithmetic_sequence_ninth_term_l80_80678


namespace card_distribution_count_l80_80035

theorem card_distribution_count : 
  ∃ (methods : ℕ), methods = 18 ∧ 
  ∃ (cards : Finset ℕ),
  ∃ (envelopes : Finset (Finset ℕ)), 
  cards = {1, 2, 3, 4, 5, 6} ∧ 
  envelopes.card = 3 ∧ 
  (∀ e ∈ envelopes, (e.card = 2) ∧ ({1, 2} ⊆ e → ∃ e1 e2, {e1, e2} ∈ envelopes ∧ {e1, e2} ⊆ cards \ {1, 2})) ∧ 
  (∀ c1 ∈ cards, ∃ e ∈ envelopes, c1 ∈ e) :=
by
  sorry

end card_distribution_count_l80_80035


namespace cuboid_edge_length_l80_80511

-- This is the main statement we want to prove
theorem cuboid_edge_length (L : ℝ) (w : ℝ) (h : ℝ) (V : ℝ) (w_eq : w = 5) (h_eq : h = 3) (V_eq : V = 30) :
  V = L * w * h → L = 2 :=
by
  -- Adding the sorry allows us to compile and acknowledge the current placeholder for the proof.
  sorry

end cuboid_edge_length_l80_80511


namespace dot_product_ABC_l80_80733

-- Defining vectors as pairs of real numbers
def vector := (ℝ × ℝ)

-- Defining the vectors AB and AC
def AB : vector := (1, 0)
def AC : vector := (-2, 3)

-- Definition of vector subtraction
def vector_sub (v1 v2 : vector) : vector := (v1.1 - v2.1, v1.2 - v2.2)

-- Definition of dot product
def dot_product (v1 v2 : vector) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define vector BC using the given vectors AB and AC
def BC : vector := vector_sub AC AB

-- The theorem stating the desired dot product result
theorem dot_product_ABC : dot_product AB BC = -3 := by
  sorry

end dot_product_ABC_l80_80733


namespace tan_diff_eq_rat_l80_80820

theorem tan_diff_eq_rat (A : ℝ × ℝ) (B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (5, 1))
  (α β : ℝ)
  (hα : Real.tan α = 2) (hβ : Real.tan β = 1 / 5) :
  Real.tan (α - β) = 9 / 7 := by
  sorry

end tan_diff_eq_rat_l80_80820


namespace joined_toucans_is_1_l80_80343

-- Define the number of toucans initially
def initial_toucans : ℕ := 2

-- Define the total number of toucans after some join
def total_toucans : ℕ := 3

-- Define the number of toucans that joined
def toucans_joined : ℕ := total_toucans - initial_toucans

-- State the theorem to prove that 1 toucan joined
theorem joined_toucans_is_1 : toucans_joined = 1 :=
by
  sorry

end joined_toucans_is_1_l80_80343


namespace sum_of_coefficients_l80_80489

theorem sum_of_coefficients:
  (x^3 + 2*x + 1) * (3*x^2 + 4) = 28 :=
by
  sorry

end sum_of_coefficients_l80_80489


namespace cylinder_height_relationship_l80_80207

variables (π r₁ r₂ h₁ h₂ : ℝ)

theorem cylinder_height_relationship
  (h_volume_eq : π * r₁^2 * h₁ = π * r₂^2 * h₂)
  (h_radius_rel : r₂ = 1.2 * r₁) :
  h₁ = 1.44 * h₂ :=
by {
  sorry -- proof not required as per instructions
}

end cylinder_height_relationship_l80_80207


namespace part1_part2_l80_80751

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) - abs (x - 2)

theorem part1 : 
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} :=
sorry 

noncomputable def g (x : ℝ) : ℝ := f x - x^2 + x

theorem part2 (m : ℝ) : 
  (∃ x : ℝ, f x ≥ x^2 - x + m) → m ≤ 5/4 :=
sorry 

end part1_part2_l80_80751


namespace sum_of_three_numbers_is_98_l80_80882

variable (A B C : ℕ) (h_ratio1 : A = 2 * (B / 3)) (h_ratio2 : B = 30) (h_ratio3 : B = 5 * (C / 8))

theorem sum_of_three_numbers_is_98 : A + B + C = 98 := by
  sorry

end sum_of_three_numbers_is_98_l80_80882


namespace find_n_l80_80632

theorem find_n (n : ℤ) : -180 ≤ n ∧ n ≤ 180 ∧ (Real.sin (n * Real.pi / 180) = Real.cos (690 * Real.pi / 180)) → n = 60 :=
by
  intro h
  sorry

end find_n_l80_80632


namespace design_height_lower_part_l80_80588

theorem design_height_lower_part (H : ℝ) (H_eq : H = 2) (L : ℝ) 
  (ratio : (H - L) / L = L / H) : L = Real.sqrt 5 - 1 :=
by {
  sorry
}

end design_height_lower_part_l80_80588


namespace product_of_roots_of_cubic_l80_80206

theorem product_of_roots_of_cubic :
  let a := 2
  let d := 18
  let product_of_roots := -(d / a)
  product_of_roots = -9 :=
by
  sorry

end product_of_roots_of_cubic_l80_80206


namespace evaluate_powers_of_i_l80_80424

noncomputable def imag_unit := Complex.I

theorem evaluate_powers_of_i :
  (imag_unit^11 + imag_unit^16 + imag_unit^21 + imag_unit^26 + imag_unit^31) = -imag_unit :=
by
  sorry

end evaluate_powers_of_i_l80_80424


namespace veenapaniville_private_independent_district_A_l80_80851

theorem veenapaniville_private_independent_district_A :
  let total_schools := 50
  let public_schools := 25
  let parochial_schools := 16
  let private_schools := 9
  let district_A_schools := 18
  let district_B_schools := 17
  let district_B_private := 2
  let remaining_schools := total_schools - district_A_schools - district_B_schools
  let each_kind_in_C := remaining_schools / 3
  let district_C_private := each_kind_in_C
  let district_A_private := private_schools - district_B_private - district_C_private
  district_A_private = 2 := by
  sorry

end veenapaniville_private_independent_district_A_l80_80851


namespace length_of_symmedian_l80_80462

theorem length_of_symmedian (a b c : ℝ) (AS : ℝ) :
  AS = (2 * b * c^2) / (b^2 + c^2) := sorry

end length_of_symmedian_l80_80462


namespace investment_ratio_l80_80398

theorem investment_ratio (X_investment Y_investment : ℕ) (hX : X_investment = 5000) (hY : Y_investment = 15000) : 
  X_investment * 3 = Y_investment :=
by
  sorry

end investment_ratio_l80_80398


namespace power_product_to_seventh_power_l80_80683

theorem power_product_to_seventh_power :
  (2 ^ 14) * (2 ^ 21) = (32 ^ 7) :=
by
  sorry

end power_product_to_seventh_power_l80_80683


namespace inscribed_circle_theta_l80_80014

/-- Given that a circle inscribed in triangle ABC is tangent to sides BC, CA, and AB at points
    where the tangential angles are 120 degrees, 130 degrees, and theta degrees respectively,
    we need to prove that theta is 110 degrees. -/
theorem inscribed_circle_theta 
  (ABC : Type)
  (A B C : ABC)
  (theta : ℝ)
  (tangent_angle_BC : ℝ)
  (tangent_angle_CA : ℝ) 
  (tangent_angle_AB : ℝ) 
  (h1 : tangent_angle_BC = 120)
  (h2 : tangent_angle_CA = 130) 
  (h3 : tangent_angle_AB = theta) : 
  theta = 110 :=
by
  sorry

end inscribed_circle_theta_l80_80014


namespace binomial_coefficient_divisible_by_p_l80_80091

theorem binomial_coefficient_divisible_by_p (p k : ℕ) (hp : Nat.Prime p) (hk1 : 0 < k) (hk2 : k < p) :
  p ∣ (Nat.factorial p / (Nat.factorial k * Nat.factorial (p - k))) :=
by
  sorry

end binomial_coefficient_divisible_by_p_l80_80091


namespace nutmeg_amount_l80_80714

def amount_of_cinnamon : ℝ := 0.6666666666666666
def difference_cinnamon_nutmeg : ℝ := 0.16666666666666666

theorem nutmeg_amount (x : ℝ) 
  (h1 : amount_of_cinnamon = x + difference_cinnamon_nutmeg) : 
  x = 0.5 :=
by 
  sorry

end nutmeg_amount_l80_80714


namespace travel_allowance_increase_20_l80_80639

def employees_total : ℕ := 480
def employees_no_increase : ℕ := 336
def employees_salary_increase_percentage : ℕ := 10

def employees_salary_increase : ℕ :=
(employees_salary_increase_percentage * employees_total) / 100

def employees_travel_allowance_increase : ℕ :=
employees_total - (employees_salary_increase + employees_no_increase)

def travel_allowance_increase_percentage : ℕ :=
(employees_travel_allowance_increase * 100) / employees_total

theorem travel_allowance_increase_20 :
  travel_allowance_increase_percentage = 20 :=
by sorry

end travel_allowance_increase_20_l80_80639


namespace population_increase_rate_l80_80447

theorem population_increase_rate (P₀ P₁ : ℕ) (rate : ℚ) (h₁ : P₀ = 220) (h₂ : P₁ = 242) :
  rate = ((P₁ - P₀ : ℚ) / P₀) * 100 := by
  sorry

end population_increase_rate_l80_80447


namespace charity_donation_correct_l80_80978

-- Define each donation series for Suzanne, Maria, and James
def suzanne_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 10
  | (n+1)  => 2 * suzanne_donation_per_km n

def maria_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 15
  | (n+1)  => 1.5 * maria_donation_per_km n

def james_donation_per_km (n : ℕ) : ℝ :=
  match n with
  |  0     => 20
  | (n+1)  => 2 * james_donation_per_km n

-- Total donations after 5 kilometers
def total_donation_suzanne : ℝ := (List.range 5).map suzanne_donation_per_km |>.sum
def total_donation_maria : ℝ := (List.range 5).map maria_donation_per_km |>.sum
def total_donation_james : ℝ := (List.range 5).map james_donation_per_km |>.sum

def total_donation_charity : ℝ :=
  total_donation_suzanne + total_donation_maria + total_donation_james

-- Statement to be proven
theorem charity_donation_correct : total_donation_charity = 1127.81 := by
  sorry

end charity_donation_correct_l80_80978


namespace complement_union_eq_l80_80324

open Set

variable (U A B : Set ℤ)

noncomputable def universal_set : Set ℤ := {-2, -1, 0, 1, 2, 3}

noncomputable def setA : Set ℤ := {-1, 0, 3}

noncomputable def setB : Set ℤ := {1, 3}

theorem complement_union_eq :
  A ∪ B = {-1, 0, 1, 3} →
  U = universal_set →
  A = setA →
  B = setB →
  (U \ (A ∪ B)) = {-2, 2} := by
  intros
  sorry

end complement_union_eq_l80_80324


namespace inequality_holds_l80_80945

theorem inequality_holds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  (a + (1 / b))^2 + (b + (1 / c))^2 + (c + (1 / a))^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end inequality_holds_l80_80945


namespace intersect_complementB_l80_80950

def setA (x : ℝ) : Prop := ∃ y : ℝ, y = Real.log (9 - x^2)

def setB (x : ℝ) : Prop := ∃ y : ℝ, y = Real.sqrt (4 * x - x^2)

def complementB (x : ℝ) : Prop := x < 0 ∨ 4 < x

theorem intersect_complementB :
  { x : ℝ | setA x } ∩ { x : ℝ | complementB x } = { x : ℝ | -3 < x ∧ x < 0 } :=
sorry

end intersect_complementB_l80_80950


namespace tim_kittens_count_l80_80654

def initial_kittens : Nat := 6
def kittens_given_to_jessica : Nat := 3
def kittens_received_from_sara : Nat := 9

theorem tim_kittens_count : initial_kittens - kittens_given_to_jessica + kittens_received_from_sara = 12 :=
by
  sorry

end tim_kittens_count_l80_80654


namespace original_cost_l80_80315

theorem original_cost (C : ℝ) (h : 550 = 1.35 * C) : C = 550 / 1.35 :=
by
  sorry

end original_cost_l80_80315


namespace hostel_cost_l80_80784

def first_week_rate : ℝ := 18
def additional_week_rate : ℝ := 12
def first_week_days : ℕ := 7
def total_days : ℕ := 23

theorem hostel_cost :
  (first_week_days * first_week_rate + 
  (total_days - first_week_days) / first_week_days * first_week_days * additional_week_rate + 
  (total_days - first_week_days) % first_week_days * additional_week_rate) = 318 := 
by
  sorry

end hostel_cost_l80_80784


namespace largest_integer_x_l80_80432

theorem largest_integer_x (x : ℤ) : (x / 4 + 3 / 5 < 7 / 4) → x ≤ 4 := sorry

end largest_integer_x_l80_80432


namespace running_time_around_pentagon_l80_80869

theorem running_time_around_pentagon :
  let l₁ := 40
  let l₂ := 50
  let l₃ := 60
  let l₄ := 45
  let l₅ := 55
  let v₁ := 9 * 1000 / 60
  let v₂ := 8 * 1000 / 60
  let v₃ := 7 * 1000 / 60
  let v₄ := 6 * 1000 / 60
  let v₅ := 5 * 1000 / 60
  let t₁ := l₁ / v₁
  let t₂ := l₂ / v₂
  let t₃ := l₃ / v₃
  let t₄ := l₄ / v₄
  let t₅ := l₅ / v₅
  t₁ + t₂ + t₃ + t₄ + t₅ = 2.266 := by
    sorry

end running_time_around_pentagon_l80_80869


namespace Problem_l80_80631

theorem Problem (x y : ℝ) (h1 : 2*x + 2*y = 10) (h2 : x*y = -15) : 4*(x^2) + 4*(y^2) = 220 := 
by
  sorry

end Problem_l80_80631


namespace decreasing_implies_b_geq_4_l80_80093

-- Define the function and its derivative
def function (x : ℝ) (b : ℝ) : ℝ := x^3 - 3*b*x + 1

def derivative (x : ℝ) (b : ℝ) : ℝ := 3*x^2 - 3*b

theorem decreasing_implies_b_geq_4 (b : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → derivative x b ≤ 0) → b ≥ 4 :=
by
  intros h
  sorry

end decreasing_implies_b_geq_4_l80_80093


namespace Adam_picks_apples_days_l80_80830

theorem Adam_picks_apples_days (total_apples remaining_apples daily_pick : ℕ) 
  (h1 : total_apples = 350) 
  (h2 : remaining_apples = 230) 
  (h3 : daily_pick = 4) : 
  (total_apples - remaining_apples) / daily_pick = 30 :=
by {
  sorry
}

end Adam_picks_apples_days_l80_80830


namespace find_a6_a7_l80_80097

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- Given Conditions
axiom cond1 : arithmetic_sequence a d
axiom cond2 : a 2 + a 4 + a 9 + a 11 = 32

-- Proof Problem
theorem find_a6_a7 : a 6 + a 7 = 16 :=
  sorry

end find_a6_a7_l80_80097


namespace cubic_sum_l80_80212

theorem cubic_sum (p q r : ℝ) (h1 : p + q + r = 4) (h2 : p * q + q * r + r * p = 7) (h3 : p * q * r = -10) :
  p ^ 3 + q ^ 3 + r ^ 3 = 154 := 
by sorry

end cubic_sum_l80_80212


namespace sum_of_Ns_l80_80881

theorem sum_of_Ns (N R : ℝ) (hN_nonzero : N ≠ 0) (h_eq : N - 3 * N^2 = R) : 
  ∃ N1 N2 : ℝ, N1 ≠ 0 ∧ N2 ≠ 0 ∧ 3 * N1^2 - N1 + R = 0 ∧ 3 * N2^2 - N2 + R = 0 ∧ (N1 + N2) = 1 / 3 :=
sorry

end sum_of_Ns_l80_80881


namespace original_price_of_cycle_l80_80581

theorem original_price_of_cycle (P : ℝ) (h1 : P * 0.85 = 1190) : P = 1400 :=
by
  sorry

end original_price_of_cycle_l80_80581


namespace product_third_fourth_term_l80_80645

theorem product_third_fourth_term (a d : ℝ) : 
  (a + 7 * d = 20) → (d = 2) → 
  ( (a + 2 * d) * (a + 3 * d) = 120 ) := 
by 
  intros h1 h2
  sorry

end product_third_fourth_term_l80_80645


namespace TriangleInscribedAngle_l80_80770

theorem TriangleInscribedAngle
  (x : ℝ)
  (arc_PQ : ℝ := x + 100)
  (arc_QR : ℝ := 2 * x + 50)
  (arc_RP : ℝ := 3 * x - 40)
  (angle_sum_eq_360 : arc_PQ + arc_QR + arc_RP = 360) :
  ∃ angle_PQR : ℝ, angle_PQR = 70.84 := 
sorry

end TriangleInscribedAngle_l80_80770


namespace negation_of_at_most_four_is_at_least_five_l80_80298

theorem negation_of_at_most_four_is_at_least_five :
  (∀ n : ℕ, n ≤ 4) ↔ (∃ n : ℕ, n ≥ 5) := 
sorry

end negation_of_at_most_four_is_at_least_five_l80_80298


namespace fill_40x41_table_l80_80824

-- Define the condition on integers in the table
def valid_integer_filling (m n : ℕ) (table : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < m → j < n →
    table i j =
    ((if i > 0 then if table i j = table (i - 1) j then 1 else 0 else 0) +
    (if j > 0 then if table i j = table i (j - 1) then 1 else 0 else 0) +
    (if i < m - 1 then if table i j = table (i + 1) j then 1 else 0 else 0) +
    (if j < n - 1 then if table i j = table i (j + 1) then 1 else 0 else 0))

-- Define the specific problem for a 40 × 41 table.
theorem fill_40x41_table :
  ∃ (table : ℕ → ℕ → ℕ), valid_integer_filling 40 41 table :=
by
  sorry

end fill_40x41_table_l80_80824


namespace anyas_hair_loss_l80_80153

theorem anyas_hair_loss (H : ℝ) 
  (washes_hair_loss : H > 0) 
  (brushes_hair_loss : H / 2 > 0) 
  (grows_back : ∃ h : ℝ, h = 49 ∧ H + H / 2 + 1 = h) :
  H = 32 :=
by
  sorry

end anyas_hair_loss_l80_80153


namespace swap_equality_l80_80077

theorem swap_equality {a1 b1 a2 b2 : ℝ} 
  (h1 : a1^2 + b1^2 = 1)
  (h2 : a2^2 + b2^2 = 1)
  (h3 : a1 * a2 + b1 * b2 = 0) :
  b1 = a2 ∨ b1 = -a2 :=
by sorry

end swap_equality_l80_80077


namespace interval_monotonic_increase_axis_of_symmetry_max_and_min_values_l80_80492

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem interval_monotonic_increase (k : ℤ) :
  ∀ x : ℝ, -Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + k * Real.pi →
    ∃ I : Set ℝ, I = Set.Icc (-Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi) ∧
      (∀ x1 x2 : ℝ, x1 ∈ I ∧ x2 ∈ I → x1 ≤ x2 → f x1 ≤ f x2) := sorry

theorem axis_of_symmetry (k : ℤ) :
  ∃ x : ℝ, x = Real.pi / 3 + k * (Real.pi / 2) := sorry

theorem max_and_min_values :
  ∃ (max_val min_val : ℝ), max_val = 2 ∧ min_val = -1 ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      ((f x = 2 ∧ x = Real.pi / 3) ∨ (f x = -1 ∧ x = 0))) := sorry

end interval_monotonic_increase_axis_of_symmetry_max_and_min_values_l80_80492


namespace totalOwlsOnFence_l80_80888

-- Define the conditions given in the problem
def initialOwls : Nat := 3
def joinedOwls : Nat := 2

-- Define the total number of owls
def totalOwls : Nat := initialOwls + joinedOwls

-- State the theorem we want to prove
theorem totalOwlsOnFence : totalOwls = 5 := by
  sorry

end totalOwlsOnFence_l80_80888


namespace length_of_chord_l80_80783

theorem length_of_chord (x y : ℝ) 
  (h1 : (x - 1)^2 + y^2 = 4) 
  (h2 : x + y + 1 = 0) 
  : ∃ (l : ℝ), l = 2 * Real.sqrt 2 := by
  sorry

end length_of_chord_l80_80783


namespace find_f_of_2_l80_80873

theorem find_f_of_2 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x) = 4 * x - 1) : f 2 = 3 :=
by
  sorry

end find_f_of_2_l80_80873


namespace rods_in_mile_l80_80134

theorem rods_in_mile (mile_to_furlongs : 1 = 12) (furlong_to_rods : 1 = 50) : 1 * 12 * 50 = 600 :=
by
  sorry

end rods_in_mile_l80_80134


namespace simplify_abs_value_l80_80987

theorem simplify_abs_value : abs (- 5 ^ 2 + 6) = 19 := by
  sorry

end simplify_abs_value_l80_80987


namespace systems_on_second_street_l80_80854

-- Definitions based on the conditions
def commission_per_system : ℕ := 25
def total_commission : ℕ := 175
def systems_on_first_street (S : ℕ) := S / 2
def systems_on_third_street : ℕ := 0
def systems_on_fourth_street : ℕ := 1

-- Question: How many security systems did Rodney sell on the second street?
theorem systems_on_second_street (S : ℕ) :
  S / 2 + S + 0 + 1 = total_commission / commission_per_system → S = 4 :=
by
  intros h
  sorry

end systems_on_second_street_l80_80854


namespace quarter_sector_area_l80_80722

theorem quarter_sector_area (d : ℝ) (h : d = 10) : (π * (d / 2)^2) / 4 = 6.25 * π :=
by 
  sorry

end quarter_sector_area_l80_80722


namespace width_of_hall_l80_80635

variable (L W H : ℕ) -- Length, Width, Height of the hall
variable (expenditure cost : ℕ) -- Expenditure and cost per square meter

-- Given conditions
def hall_length : L = 20 := by sorry
def hall_height : H = 5 := by sorry
def total_expenditure : expenditure = 28500 := by sorry
def cost_per_sq_meter : cost = 30 := by sorry

-- Derived value
def total_area_to_cover (W : ℕ) : ℕ :=
  (2 * (L * W) + 2 * (L * H) + 2 * (W * H))

theorem width_of_hall (W : ℕ) (h: total_area_to_cover L W H * cost = expenditure) : W = 15 := by
  sorry

end width_of_hall_l80_80635


namespace initial_erasers_in_box_l80_80135

-- Definitions based on the conditions
def erasers_in_bag_jane := 15
def erasers_taken_out_doris := 54
def erasers_left_in_box := 15

-- Theorem statement
theorem initial_erasers_in_box : ∃ B_i : ℕ, B_i = erasers_taken_out_doris + erasers_left_in_box ∧ B_i = 69 :=
by
  use 69
  -- omitted proof steps
  sorry

end initial_erasers_in_box_l80_80135


namespace pet_shop_legs_l80_80666

theorem pet_shop_legs :
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  birds * bird_legs + dogs * dog_legs + snakes * snake_legs + spiders * spider_legs = 34 := 
by
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  sorry

end pet_shop_legs_l80_80666


namespace malia_berries_second_bush_l80_80333

theorem malia_berries_second_bush :
  ∀ (b2 : ℕ), ∃ (d1 d2 d3 d4 : ℕ),
  d1 = 3 → d2 = 7 → d3 = 12 → d4 = 19 →
  d2 - d1 = (d3 - d2) - 2 →
  d3 - d2 = (d4 - d3) - 2 →
  b2 = d1 + (d2 - d1 - 2) →
  b2 = 6 :=
by
  sorry

end malia_berries_second_bush_l80_80333


namespace solve_for_x_l80_80148

theorem solve_for_x (x t : ℝ)
  (h₁ : t = 9)
  (h₂ : (3 * (x + 5)) / 4 = t + (3 - 3 * x) / 2) :
  x = 3 :=
by
  sorry

end solve_for_x_l80_80148


namespace remaining_candy_l80_80356

def initial_candy : ℕ := 36
def ate_candy1 : ℕ := 17
def ate_candy2 : ℕ := 15
def total_ate_candy : ℕ := ate_candy1 + ate_candy2

theorem remaining_candy : initial_candy - total_ate_candy = 4 := by
  sorry

end remaining_candy_l80_80356


namespace route_B_is_faster_by_7_5_minutes_l80_80634

def distance_A := 10  -- miles
def normal_speed_A := 30  -- mph
def construction_distance_A := 2  -- miles
def construction_speed_A := 15  -- mph
def distance_B := 8  -- miles
def normal_speed_B := 40  -- mph
def school_zone_distance_B := 1  -- miles
def school_zone_speed_B := 10  -- mph

noncomputable def time_for_normal_speed_A : ℝ := (distance_A - construction_distance_A) / normal_speed_A * 60  -- minutes
noncomputable def time_for_construction_A : ℝ := construction_distance_A / construction_speed_A * 60  -- minutes
noncomputable def total_time_A : ℝ := time_for_normal_speed_A + time_for_construction_A

noncomputable def time_for_normal_speed_B : ℝ := (distance_B - school_zone_distance_B) / normal_speed_B * 60  -- minutes
noncomputable def time_for_school_zone_B : ℝ := school_zone_distance_B / school_zone_speed_B * 60  -- minutes
noncomputable def total_time_B : ℝ := time_for_normal_speed_B + time_for_school_zone_B

theorem route_B_is_faster_by_7_5_minutes : total_time_B + 7.5 = total_time_A := by
  sorry

end route_B_is_faster_by_7_5_minutes_l80_80634


namespace complex_number_real_l80_80902

theorem complex_number_real (m : ℝ) (z : ℂ) 
  (h1 : z = ⟨1 / (m + 5), 0⟩ + ⟨0, m^2 + 2 * m - 15⟩)
  (h2 : m^2 + 2 * m - 15 = 0)
  (h3 : m ≠ -5) :
  m = 3 :=
sorry

end complex_number_real_l80_80902


namespace abs_diff_eq_0_5_l80_80619

noncomputable def x : ℝ := 3.7
noncomputable def y : ℝ := 4.2

theorem abs_diff_eq_0_5 (hx : ⌊x⌋ + (y - ⌊y⌋) = 3.2) (hy : (x - ⌊x⌋) + ⌊y⌋ = 4.7) :
  |x - y| = 0.5 :=
by
  sorry

end abs_diff_eq_0_5_l80_80619


namespace nature_of_roots_l80_80822

noncomputable def P (x : ℝ) : ℝ := x^6 - 5 * x^5 - 7 * x^3 - 2 * x + 9

theorem nature_of_roots : 
  (∀ x < 0, P x > 0) ∧ ∃ x > 0, P 0 * P x < 0 := 
by {
  sorry
}

end nature_of_roots_l80_80822


namespace smallest_two_digit_prime_with_conditions_l80_80891

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

noncomputable def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

theorem smallest_two_digit_prime_with_conditions :
  ∃ p : ℕ, is_prime p ∧ 10 ≤ p ∧ p < 100 ∧ (p / 10 = 3) ∧ is_composite (((p % 10) * 10) + (p / 10) + 5) ∧ p = 31 :=
by
  sorry

end smallest_two_digit_prime_with_conditions_l80_80891


namespace overall_average_score_l80_80346

theorem overall_average_score (students_total : ℕ) (scores_day1 : ℕ) (avg1 : ℝ)
  (scores_day2 : ℕ) (avg2 : ℝ) (scores_day3 : ℕ) (avg3 : ℝ)
  (h1 : students_total = 45)
  (h2 : scores_day1 = 35)
  (h3 : avg1 = 0.65)
  (h4 : scores_day2 = 8)
  (h5 : avg2 = 0.75)
  (h6 : scores_day3 = 2)
  (h7 : avg3 = 0.85) :
  (scores_day1 * avg1 + scores_day2 * avg2 + scores_day3 * avg3) / students_total = 0.68 :=
by
  -- Lean proof goes here
  sorry

end overall_average_score_l80_80346


namespace initial_liquid_X_percentage_is_30_l80_80271

variable (initial_liquid_X_percentage : ℝ)

theorem initial_liquid_X_percentage_is_30
  (solution_total_weight : ℝ := 8)
  (initial_water_percentage : ℝ := 70)
  (evaporated_water_weight : ℝ := 3)
  (added_solution_weight : ℝ := 3)
  (new_liquid_X_percentage : ℝ := 41.25)
  (total_new_solution_weight : ℝ := 8)
  :
  initial_liquid_X_percentage = 30 :=
sorry

end initial_liquid_X_percentage_is_30_l80_80271


namespace angle_is_20_l80_80624

theorem angle_is_20 (x : ℝ) (h : 180 - x = 2 * (90 - x) + 20) : x = 20 :=
by
  sorry

end angle_is_20_l80_80624


namespace incorrect_height_is_151_l80_80501

def incorrect_height (average_initial correct_height average_corrected : ℝ) : ℝ :=
  (30 * average_initial) - (30 * average_corrected) + correct_height

theorem incorrect_height_is_151 :
  incorrect_height 175 136 174.5 = 151 :=
by
  sorry

end incorrect_height_is_151_l80_80501


namespace age_problem_l80_80760

variables (a b c : ℕ)

theorem age_problem (h₁ : a = b + 2) (h₂ : b = 2 * c) (h₃ : a + b + c = 27) : b = 10 :=
by {
  -- Interactive proof steps can go here.
  sorry
}

end age_problem_l80_80760


namespace cube_fit_count_cube_volume_percentage_l80_80880

-- Definitions based on the conditions in the problem.
def box_length : ℕ := 8
def box_width : ℕ := 4
def box_height : ℕ := 12
def cube_side : ℕ := 4

-- Definitions for the calculated values.
def num_cubes_length : ℕ := box_length / cube_side
def num_cubes_width : ℕ := box_width / cube_side
def num_cubes_height : ℕ := box_height / cube_side

def total_cubes : ℕ := num_cubes_length * num_cubes_width * num_cubes_height

def volume_cube : ℕ := cube_side^3
def volume_cubes_total : ℕ := total_cubes * volume_cube
def volume_box : ℕ := box_length * box_width * box_height

def percentage_volume : ℕ := (volume_cubes_total * 100) / volume_box

-- The proof statements.
theorem cube_fit_count : total_cubes = 6 := by
  sorry

theorem cube_volume_percentage : percentage_volume = 100 := by
  sorry

end cube_fit_count_cube_volume_percentage_l80_80880


namespace quadratic_real_roots_l80_80664

theorem quadratic_real_roots (k : ℝ) (h : k ≠ 0) : 
  (∃ x1 x2 : ℝ, k * x1^2 - 6 * x1 - 1 = 0 ∧ k * x2^2 - 6 * x2 - 1 = 0 ∧ x1 ≠ x2) ↔ k ≥ -9 := 
by
  sorry

end quadratic_real_roots_l80_80664


namespace find_cubic_sum_l80_80257

theorem find_cubic_sum
  {a b : ℝ}
  (h1 : a^5 - a^4 * b - a^4 + a - b - 1 = 0)
  (h2 : 2 * a - 3 * b = 1) :
  a^3 + b^3 = 9 :=
by
  sorry

end find_cubic_sum_l80_80257


namespace circle_occupies_62_8_percent_l80_80275

noncomputable def largestCirclePercentage (length : ℝ) (width : ℝ) : ℝ :=
  let radius := width / 2
  let circle_area := Real.pi * radius^2
  let rectangle_area := length * width
  (circle_area / rectangle_area) * 100

theorem circle_occupies_62_8_percent : largestCirclePercentage 5 4 = 62.8 := 
by 
  /- Sorry, skipping the proof -/
  sorry

end circle_occupies_62_8_percent_l80_80275


namespace sin_eq_one_fifth_l80_80466

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sin_eq_one_fifth (ϕ : ℝ)
  (h : binomial_coefficient 5 3 * (Real.cos ϕ)^2 = 4) :
  Real.sin (2 * ϕ - π / 2) = 1 / 5 := sorry

end sin_eq_one_fifth_l80_80466


namespace minimum_employees_for_identical_training_l80_80397

def languages : Finset String := {"English", "French", "Spanish", "German"}

noncomputable def choose_pairings_count (n k : ℕ) : ℕ :=
Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem minimum_employees_for_identical_training 
  (num_languages : ℕ := 4) 
  (employees_per_pairing : ℕ := 4)
  (pairings : ℕ := choose_pairings_count num_languages 2) 
  (total_employees : ℕ := employees_per_pairing * pairings)
  (minimum_employees : ℕ := total_employees + 1):
  minimum_employees = 25 :=
by
  -- We skip the proof details as per the instructions
  sorry

end minimum_employees_for_identical_training_l80_80397


namespace tate_total_years_eq_12_l80_80571

-- Definitions based on conditions
def high_school_normal_years : ℕ := 4
def high_school_years : ℕ := high_school_normal_years - 1
def college_years : ℕ := 3 * high_school_years
def total_years : ℕ := high_school_years + college_years

-- Statement to prove
theorem tate_total_years_eq_12 : total_years = 12 := by
  sorry

end tate_total_years_eq_12_l80_80571


namespace parabola_ord_l80_80563

theorem parabola_ord {M : ℝ × ℝ} (h1 : M.1 = (M.2 * M.2) / 8) (h2 : dist M (2, 0) = 4) : M.2 = 4 ∨ M.2 = -4 := 
sorry

end parabola_ord_l80_80563


namespace kirill_is_62_5_l80_80449

variable (K : ℝ)

def kirill_height := K
def brother_height := K + 14
def sister_height := 2 * K
def total_height := K + (K + 14) + 2 * K

theorem kirill_is_62_5 (h1 : total_height K = 264) : K = 62.5 := by
  sorry

end kirill_is_62_5_l80_80449


namespace y_intercept_of_line_b_l80_80433

theorem y_intercept_of_line_b
  (m : ℝ) (c₁ : ℝ) (c₂ : ℝ) (x₁ : ℝ) (y₁ : ℝ)
  (h_parallel : m = 3/2)
  (h_point : (4, 2) ∈ { p : ℝ × ℝ | p.2 = m * p.1 + c₂ }) :
  c₂ = -4 := by
  sorry

end y_intercept_of_line_b_l80_80433


namespace complete_square_formula_D_l80_80412

-- Definitions of polynomial multiplications
def poly_A (a b : ℝ) : ℝ := (a - b) * (a + b)
def poly_B (a b : ℝ) : ℝ := -((a + b) * (b - a))
def poly_C (a b : ℝ) : ℝ := (a + b) * (b - a)
def poly_D (a b : ℝ) : ℝ := (a - b) * (b - a)

theorem complete_square_formula_D (a b : ℝ) : 
  poly_D a b = -(a - b)*(a - b) :=
by sorry

end complete_square_formula_D_l80_80412


namespace cristina_running_pace_l80_80685

theorem cristina_running_pace
  (nicky_pace : ℝ) (nicky_headstart : ℝ) (time_nicky_run : ℝ) 
  (distance_nicky_run : ℝ) (time_cristina_catch : ℝ) :
  (nicky_pace = 3) →
  (nicky_headstart = 12) →
  (time_nicky_run = 30) →
  (distance_nicky_run = nicky_pace * time_nicky_run) →
  (time_cristina_catch = time_nicky_run - nicky_headstart) →
  (cristina_pace : ℝ) →
  (cristina_pace = distance_nicky_run / time_cristina_catch) →
  cristina_pace = 5 :=
by
  sorry

end cristina_running_pace_l80_80685


namespace minimum_cuts_for_10_pieces_l80_80396

theorem minimum_cuts_for_10_pieces :
  ∃ n : ℕ, (n * (n + 1)) / 2 ≥ 10 ∧ ∀ m < n, (m * (m + 1)) / 2 < 10 := sorry

end minimum_cuts_for_10_pieces_l80_80396


namespace complete_the_square_eqn_l80_80058

theorem complete_the_square_eqn (x b c : ℤ) (h_eqn : x^2 - 10 * x + 15 = 0) (h_form : (x + b)^2 = c) : b + c = 5 := by
  sorry

end complete_the_square_eqn_l80_80058


namespace sufficient_and_necessary_condition_l80_80258

theorem sufficient_and_necessary_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
by 
  sorry

end sufficient_and_necessary_condition_l80_80258


namespace eggs_left_l80_80699

def initial_eggs := 20
def mother_used := 5
def father_used := 3
def chicken1_laid := 4
def chicken2_laid := 3
def chicken3_laid := 2
def oldest_took := 2

theorem eggs_left :
  initial_eggs - (mother_used + father_used) + (chicken1_laid + chicken2_laid + chicken3_laid) - oldest_took = 19 := 
by
  sorry

end eggs_left_l80_80699


namespace lino_shells_l80_80006

theorem lino_shells (picked_up : ℝ) (put_back : ℝ) (remaining_shells : ℝ) :
  picked_up = 324.0 → 
  put_back = 292.0 → 
  remaining_shells = picked_up - put_back → 
  remaining_shells = 32.0 :=
by
  intros h1 h2 h3
  sorry

end lino_shells_l80_80006


namespace eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256_l80_80187

theorem eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256 :
  11^2 + 2 * 11 * 5 + 5^2 = 256 := by
  sorry

end eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256_l80_80187


namespace team_c_score_l80_80582

theorem team_c_score (points_A points_B total_points : ℕ) (hA : points_A = 2) (hB : points_B = 9) (hTotal : total_points = 15) :
  total_points - (points_A + points_B) = 4 :=
by
  sorry

end team_c_score_l80_80582


namespace maximize_area_l80_80579

variable (x : ℝ)
def fence_length : ℝ := 240 - 2 * x
def area (x : ℝ) : ℝ := x * fence_length x

theorem maximize_area : fence_length 60 = 120 :=
  sorry

end maximize_area_l80_80579


namespace commute_time_variance_l80_80544

theorem commute_time_variance
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : (x - 10)^2 + (y - 10)^2 = 8) :
  x^2 + y^2 = 208 :=
by
  sorry

end commute_time_variance_l80_80544


namespace f_f_1_equals_4_l80_80046

noncomputable def f (x : ℝ) : ℝ :=
  if x > 2 then x + 1 / (x - 2) else x^2 + 2

theorem f_f_1_equals_4 : f (f 1) = 4 := by sorry

end f_f_1_equals_4_l80_80046


namespace find_value_l80_80318

theorem find_value (x : ℤ) (h : 3 * x - 45 = 159) : (x + 32) * 12 = 1200 :=
by
  sorry

end find_value_l80_80318


namespace time_to_cross_first_platform_l80_80867

variable (length_first_platform : ℝ)
variable (length_second_platform : ℝ)
variable (time_to_cross_second_platform : ℝ)
variable (length_of_train : ℝ)

theorem time_to_cross_first_platform :
  length_first_platform = 160 →
  length_second_platform = 250 →
  time_to_cross_second_platform = 20 →
  length_of_train = 110 →
  (270 / (360 / 20) = 15) := 
by
  intro h1 h2 h3 h4
  sorry

end time_to_cross_first_platform_l80_80867


namespace present_age_of_R_l80_80044

variables (P_p Q_p R_p : ℝ)

-- Conditions from the problem
axiom h1 : P_p - 8 = 1/2 * (Q_p - 8)
axiom h2 : Q_p - 8 = 2/3 * (R_p - 8)
axiom h3 : Q_p = 2 * Real.sqrt R_p
axiom h4 : P_p = 3/5 * Q_p

theorem present_age_of_R : R_p = 400 :=
by
  sorry

end present_age_of_R_l80_80044


namespace lottery_probability_correct_l80_80622

/-- The binomial coefficient function -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of matching MegaBall and WinnerBalls in the lottery -/
noncomputable def lottery_probability : ℚ :=
  let megaBall_prob := (1 : ℚ) / 30
  let winnerBalls_prob := (1 : ℚ) / binom 45 6
  megaBall_prob * winnerBalls_prob

theorem lottery_probability_correct : lottery_probability = (1 : ℚ) / 244351800 := by
  sorry

end lottery_probability_correct_l80_80622


namespace sum_C_D_eq_one_fifth_l80_80899

theorem sum_C_D_eq_one_fifth (D C : ℚ) :
  (∀ x : ℚ, (Dx - 13) / (x^2 - 9 * x + 20) = C / (x - 4) + 5 / (x - 5)) →
  (C + D) = 1/5 :=
by
  sorry

end sum_C_D_eq_one_fifth_l80_80899


namespace option_C_correct_l80_80674

-- Define the base a and natural numbers m and n for exponents
variables {a : ℕ} {m n : ℕ}

-- Lean statement to prove (a^5)^3 = a^(5 * 3)
theorem option_C_correct : (a^5)^3 = a^(5 * 3) := 
by sorry

end option_C_correct_l80_80674


namespace units_digit_of_17_pow_549_l80_80921

theorem units_digit_of_17_pow_549 : (17 ^ 549) % 10 = 7 :=
by {
  -- Provide the necessary steps or strategies to prove the theorem
  sorry
}

end units_digit_of_17_pow_549_l80_80921


namespace intersection_M_N_l80_80390

def set_M : Set ℝ := { x | x < 2 }
def set_N : Set ℝ := { x | x > 0 }
def set_intersection : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_M_N : set_M ∩ set_N = set_intersection := 
by
  sorry

end intersection_M_N_l80_80390


namespace binom_six_two_l80_80080

-- Define the binomial coefficient function
def binom (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- State the theorem
theorem binom_six_two : binom 6 2 = 15 := by
  sorry

end binom_six_two_l80_80080


namespace race_distance_l80_80180

theorem race_distance (a b c : ℝ) (d : ℝ) 
  (h1 : d / a = (d - 15) / b)
  (h2 : d / b = (d - 30) / c)
  (h3 : d / a = (d - 40) / c) : 
  d = 90 :=
by sorry

end race_distance_l80_80180


namespace adelaide_ducks_l80_80911

variable (A E K : ℕ)

theorem adelaide_ducks (h1 : A = 2 * E) (h2 : E = K - 45) (h3 : (A + E + K) / 3 = 35) :
  A = 30 := by
  sorry

end adelaide_ducks_l80_80911


namespace derivative_of_f_l80_80024

-- Define the function
def f (x : ℝ) : ℝ := x^2 - x

-- State the theorem to prove
theorem derivative_of_f : ∀ x : ℝ,  (deriv f x = 2 * x - 1) :=
by sorry

end derivative_of_f_l80_80024


namespace min_value_fractions_l80_80194

theorem min_value_fractions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  2 ≤ (5 * z) / (3 * x + y) + (5 * x) / (y + 3 * z) + (2 * y) / (x + z) :=
by sorry

end min_value_fractions_l80_80194


namespace min_sum_ab_l80_80846

theorem min_sum_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / a) + (2 / b) = 2) :
  a + b ≥ (3 + 2 * Real.sqrt 2) / 2 :=
sorry

end min_sum_ab_l80_80846


namespace cost_of_article_l80_80739

theorem cost_of_article 
    (C G : ℝ) 
    (h1 : 340 = C + G) 
    (h2 : 350 = C + G + 0.05 * G) 
    : C = 140 :=
by
    -- We do not need to provide the proof; 'sorry' is sufficient.
    sorry

end cost_of_article_l80_80739


namespace tan_sum_l80_80904

theorem tan_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 96 / 65)
  (h2 : Real.cos x + Real.cos y = 72 / 65) :
  Real.tan x + Real.tan y = 507 / 112 := 
sorry

end tan_sum_l80_80904


namespace production_today_l80_80734

-- Conditions
def average_daily_production_past_n_days (P : ℕ) (n : ℕ) := P = n * 50
def new_average_daily_production (P : ℕ) (T : ℕ) (new_n : ℕ) := (P + T) / new_n = 55

-- Values from conditions
def n := 11
def P := 11 * 50

-- Mathematically equivalent proof problem
theorem production_today :
  ∃ (T : ℕ), average_daily_production_past_n_days P n ∧ new_average_daily_production P T 12 → T = 110 :=
by
  sorry

end production_today_l80_80734


namespace inequality_direction_change_l80_80573

theorem inequality_direction_change :
  ∃ (a b c : ℝ), (a < b) ∧ (c < 0) ∧ (a * c > b * c) :=
by
  sorry

end inequality_direction_change_l80_80573


namespace certain_number_is_310_l80_80001

theorem certain_number_is_310 (x : ℤ) (h : 3005 - x + 10 = 2705) : x = 310 :=
by
  sorry

end certain_number_is_310_l80_80001


namespace problem_statement_l80_80175

noncomputable def expr : ℝ :=
  (1 - Real.sqrt 5)^0 + abs (-Real.sqrt 2) - 2 * Real.cos (Real.pi / 4) + (1 / 4 : ℝ)⁻¹

theorem problem_statement : expr = 5 := by
  sorry

end problem_statement_l80_80175


namespace least_five_digit_congruent_to_8_mod_17_l80_80375

theorem least_five_digit_congruent_to_8_mod_17 : 
  ∃ n : ℕ, n = 10004 ∧ n % 17 = 8 ∧ 10000 ≤ n ∧ n < 100000 ∧ (∀ m, 10000 ≤ m ∧ m < 10004 → m % 17 ≠ 8) :=
sorry

end least_five_digit_congruent_to_8_mod_17_l80_80375


namespace object_reaches_max_height_at_three_l80_80283

theorem object_reaches_max_height_at_three :
  ∀ (h : ℝ) (t : ℝ), h = -15 * (t - 3)^2 + 150 → t = 3 :=
by
  sorry

end object_reaches_max_height_at_three_l80_80283


namespace CD_eq_CE_l80_80823

theorem CD_eq_CE {Point : Type*} [MetricSpace Point]
  (A B C D E : Point) (m : Set Point)
  (hAm : A ∈ m) (hBm : B ∈ m) (hCm : C ∈ m)
  (hDm : D ∉ m) (hEm : E ∉ m) 
  (hAD_AE : dist A D = dist A E)
  (hBD_BE : dist B D = dist B E) :
  dist C D = dist C E :=
sorry

end CD_eq_CE_l80_80823


namespace building_height_l80_80331

-- Definitions of the conditions
def wooden_box_height : ℝ := 3
def wooden_box_shadow : ℝ := 12
def building_shadow : ℝ := 36

-- The statement that needs to be proved
theorem building_height : ∃ (height : ℝ), height = 9 ∧ wooden_box_height / wooden_box_shadow = height / building_shadow :=
by
  sorry

end building_height_l80_80331


namespace elly_candies_l80_80354

theorem elly_candies (a b c : ℝ) (h1 : a * b * c = 216) : 
  24 * 216 = 5184 :=
by
  sorry

end elly_candies_l80_80354


namespace compute_x_squared_y_plus_xy_squared_l80_80301

theorem compute_x_squared_y_plus_xy_squared 
  (x y : ℝ)
  (h1 : (1 / x) + (1 / y) = 4)
  (h2 : x * y + x + y = 7) :
  x^2 * y + x * y^2 = 49 := 
  sorry

end compute_x_squared_y_plus_xy_squared_l80_80301


namespace expense_and_income_calculations_l80_80724

def alexander_salary : ℕ := 125000
def natalia_salary : ℕ := 61000
def utilities_transport_household : ℕ := 17000
def loan_repayment : ℕ := 15000
def theater_cost : ℕ := 5000
def cinema_cost_per_person : ℕ := 1000
def savings_crimea : ℕ := 20000
def dining_weekday_cost : ℕ := 1500
def dining_weekend_cost : ℕ := 3000
def weekdays : ℕ := 20
def weekends : ℕ := 10
def phone_A_cost : ℕ := 57000
def phone_B_cost : ℕ := 37000

def total_expenses : ℕ :=
  utilities_transport_household +
  loan_repayment +
  theater_cost + 2 * cinema_cost_per_person +
  savings_crimea +
  weekdays * dining_weekday_cost +
  weekends * dining_weekend_cost

def net_income : ℕ :=
  alexander_salary + natalia_salary

def can_buy_phones : Prop :=
  net_income - total_expenses < phone_A_cost + phone_B_cost

theorem expense_and_income_calculations :
  total_expenses = 119000 ∧
  net_income = 186000 ∧
  can_buy_phones :=
by
  sorry

end expense_and_income_calculations_l80_80724


namespace fractions_product_l80_80174

theorem fractions_product :
  (4 / 2) * (8 / 4) * (9 / 3) * (18 / 6) * (16 / 8) * (24 / 12) * (30 / 15) * (36 / 18) = 576 := by
  sorry

end fractions_product_l80_80174


namespace power_multiplication_l80_80443

theorem power_multiplication : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := by
  sorry

end power_multiplication_l80_80443


namespace pumps_fill_time_l80_80637

def fill_time {X Y Z : ℝ} (h1 : X + Y = 1/3) (h2 : X + Z = 1/6) (h3 : Y + Z = 2/9) : Prop :=
  1 / (X + Y + Z) = 36 / 13

theorem pumps_fill_time (X Y Z : ℝ) (h1 : X + Y = 1/3) (h2 : X + Z = 1/6) (h3 : Y + Z = 2/9) : 
  1 / (X + Y + Z) = 36 / 13 :=
by
  sorry

end pumps_fill_time_l80_80637


namespace product_xyz_l80_80646

theorem product_xyz {x y z a b c : ℝ} 
  (h1 : x + y + z = a) 
  (h2 : x^2 + y^2 + z^2 = b^2) 
  (h3 : x^3 + y^3 + z^3 = c^3) : 
  x * y * z = (a^3 - 3 * a * b^2 + 2 * c^3) / 6 :=
by
  sorry

end product_xyz_l80_80646


namespace symmetric_line_b_value_l80_80117

theorem symmetric_line_b_value (b : ℝ) : 
  (∃ l1 l2 : ℝ × ℝ → Prop, 
    (∀ (x y : ℝ), l1 (x, y) ↔ y = -2 * x + b) ∧ 
    (∃ p2 : ℝ × ℝ, p2 = (1, 6) ∧ l2 p2) ∧
    l2 (-1, 6) ∧ 
    (∀ (x y : ℝ), l1 (x, y) ↔ l2 (-x, y))) →
  b = 4 := 
by
  sorry

end symmetric_line_b_value_l80_80117


namespace celer_tanks_dimensions_l80_80692

theorem celer_tanks_dimensions :
  ∃ (a v : ℕ), 
    (a * a * v = 200) ∧
    (2 * a ^ 3 + 50 = 300) ∧
    (a = 5) ∧
    (v = 8) :=
sorry

end celer_tanks_dimensions_l80_80692


namespace sqrt_nine_factorial_over_72_eq_l80_80305

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_nine_factorial_over_72_eq : 
  Real.sqrt ((factorial 9) / 72) = 12 * Real.sqrt 35 :=
by
  sorry

end sqrt_nine_factorial_over_72_eq_l80_80305


namespace odd_tiling_numbers_l80_80690

def f (n k : ℕ) : ℕ := sorry -- Assume f(n, 2k) is defined appropriately.

theorem odd_tiling_numbers (n : ℕ) : (∀ k : ℕ, f n (2*k) % 2 = 1) ↔ ∃ i : ℕ, n = 2^i - 1 := sorry

end odd_tiling_numbers_l80_80690


namespace find_k_l80_80758

theorem find_k (angle_BAC : ℝ) (angle_D : ℝ)
  (h1 : 0 < angle_BAC ∧ angle_BAC < π)
  (h2 : 0 < angle_D ∧ angle_D < π)
  (h3 : (π - angle_BAC) / 2 = 3 * angle_D) :
  angle_BAC = (5 / 11) * π :=
by sorry

end find_k_l80_80758


namespace find_old_weight_l80_80429

variable (avg_increase : ℝ) (num_persons : ℕ) (W_new : ℝ) (total_increase : ℝ) (W_old : ℝ)

theorem find_old_weight (h1 : avg_increase = 3.5) 
                        (h2 : num_persons = 7) 
                        (h3 : W_new = 99.5) 
                        (h4 : total_increase = num_persons * avg_increase) 
                        (h5 : W_new = W_old + total_increase) 
                        : W_old = 75 :=
by
  sorry

end find_old_weight_l80_80429


namespace smallest_product_of_non_factors_l80_80620

theorem smallest_product_of_non_factors (a b : ℕ) (h_a : a ∣ 48) (h_b : b ∣ 48) (h_distinct : a ≠ b) (h_prod_non_factor : ¬ (a * b ∣ 48)) : a * b = 18 :=
sorry

end smallest_product_of_non_factors_l80_80620


namespace largest_number_formed_l80_80924

-- Define the digits
def digit1 : ℕ := 2
def digit2 : ℕ := 6
def digit3 : ℕ := 9

-- Define the function to form the largest number using the given digits
def largest_three_digit_number (a b c : ℕ) : ℕ :=
  if a > b ∧ a > c then
    if b > c then 100 * a + 10 * b + c
    else 100 * a + 10 * c + b
  else if b > a ∧ b > c then
    if a > c then 100 * b + 10 * a + c
    else 100 * b + 10 * c + a
  else
    if a > b then 100 * c + 10 * a + b
    else 100 * c + 10 * b + a

-- Statement that this function correctly computes the largest number
theorem largest_number_formed :
  largest_three_digit_number digit1 digit2 digit3 = 962 :=
by
  sorry

end largest_number_formed_l80_80924


namespace exists_arith_prog_5_primes_exists_arith_prog_6_primes_l80_80730

-- Define the condition of being an arithmetic progression
def is_arith_prog (seq : List ℕ) : Prop :=
  ∀ (i : ℕ), i < seq.length - 1 → seq.get! (i + 1) - seq.get! i = seq.get! 1 - seq.get! 0

-- Define the condition of being prime
def all_prime (seq : List ℕ) : Prop :=
  ∀ (n : ℕ), n ∈ seq → Nat.Prime n

-- The main statements
theorem exists_arith_prog_5_primes :
  ∃ (seq : List ℕ), seq.length = 5 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

theorem exists_arith_prog_6_primes :
  ∃ (seq : List ℕ), seq.length = 6 ∧ is_arith_prog seq ∧ all_prime seq := 
sorry

end exists_arith_prog_5_primes_exists_arith_prog_6_primes_l80_80730


namespace find_x_values_l80_80188

theorem find_x_values (x : ℝ) :
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + 2 * x)) ↔ - (12 / 7 : ℝ) ≤ x ∧ x < - (3 / 4 : ℝ) :=
by
  sorry

end find_x_values_l80_80188


namespace least_positive_integer_l80_80695

theorem least_positive_integer (n : ℕ) (h1 : n > 1)
  (h2 : n % 3 = 2) (h3 : n % 4 = 2) (h4 : n % 5 = 2) (h5 : n % 11 = 2) :
  n = 662 :=
sorry

end least_positive_integer_l80_80695


namespace bird_cages_count_l80_80703

/-- 
If each bird cage contains 2 parrots and 2 parakeets,
and the total number of birds is 36,
then the number of bird cages is 9.
-/
theorem bird_cages_count (parrots_per_cage parakeets_per_cage total_birds cages : ℕ)
  (h1 : parrots_per_cage = 2)
  (h2 : parakeets_per_cage = 2)
  (h3 : total_birds = 36)
  (h4 : total_birds = (parrots_per_cage + parakeets_per_cage) * cages) :
  cages = 9 := 
by 
  sorry

end bird_cages_count_l80_80703


namespace quadratic_inequality_l80_80030

theorem quadratic_inequality (x : ℝ) : x^2 - x + 1 ≥ 0 :=
sorry

end quadratic_inequality_l80_80030


namespace harry_book_pages_correct_l80_80697

-- Define the total pages in Selena's book.
def selena_book_pages : ℕ := 400

-- Define Harry's book pages as 20 fewer than half of Selena's book pages.
def harry_book_pages : ℕ := (selena_book_pages / 2) - 20

-- The theorem to prove the number of pages in Harry's book.
theorem harry_book_pages_correct : harry_book_pages = 180 := by
  sorry

end harry_book_pages_correct_l80_80697


namespace average_weight_women_l80_80022

variable (average_weight_men : ℕ) (number_of_men : ℕ)
variable (average_weight : ℕ) (number_of_women : ℕ)
variable (average_weight_all : ℕ) (total_people : ℕ)

theorem average_weight_women (h1 : average_weight_men = 190) 
                            (h2 : number_of_men = 8)
                            (h3 : average_weight_all = 160)
                            (h4 : total_people = 14) 
                            (h5 : number_of_women = 6):
  average_weight = 120 := 
by
  sorry

end average_weight_women_l80_80022


namespace remaining_students_l80_80102

def groups := 3
def students_per_group := 8
def students_left_early := 2

theorem remaining_students : (groups * students_per_group) - students_left_early = 22 := by
  --Proof skipped
  sorry

end remaining_students_l80_80102


namespace average_age_first_and_fifth_dogs_l80_80460

-- Define the conditions
def first_dog_age : ℕ := 10
def second_dog_age : ℕ := first_dog_age - 2
def third_dog_age : ℕ := second_dog_age + 4
def fourth_dog_age : ℕ := third_dog_age / 2
def fifth_dog_age : ℕ := fourth_dog_age + 20

-- Define the goal statement
theorem average_age_first_and_fifth_dogs : 
  (first_dog_age + fifth_dog_age) / 2 = 18 :=
by
  sorry

end average_age_first_and_fifth_dogs_l80_80460


namespace solve_equation_l80_80280

theorem solve_equation (x : ℝ) (h : x ≠ 4) :
  (x - 3) / (4 - x) - 1 = 1 / (x - 4) → x = 3 :=
by
  sorry

end solve_equation_l80_80280


namespace remaining_download_time_l80_80980

-- Define the relevant quantities
def total_size : ℝ := 1250
def downloaded : ℝ := 310
def download_speed : ℝ := 2.5

-- State the theorem
theorem remaining_download_time : (total_size - downloaded) / download_speed = 376 := by
  -- Proof will be filled in here
  sorry

end remaining_download_time_l80_80980


namespace trigonometric_identity_l80_80900

-- Definition for the given condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = 2

-- The proof goal
theorem trigonometric_identity (α : ℝ) (h : tan_alpha α) : 
  Real.cos (π + α) * Real.cos (π / 2 + α) = 2 / 5 :=
by
  sorry

end trigonometric_identity_l80_80900


namespace distinct_real_roots_sum_l80_80933

theorem distinct_real_roots_sum (p r_1 r_2 : ℝ) (h_eq : ∀ x, x^2 + p * x + 18 = 0)
  (h_distinct : r_1 ≠ r_2) (h_root1 : x^2 + p * x + 18 = 0)
  (h_root2 : x^2 + p * x + 18 = 0) : |r_1 + r_2| > 6 :=
sorry

end distinct_real_roots_sum_l80_80933


namespace odd_function_expression_l80_80547

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2*x else -((-x)^2 - 2*(-x))

theorem odd_function_expression (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, 0 ≤ x → f x = x^2 - 2*x) :
  ∀ x : ℝ, f x = x * (|x| - 2) :=
by
  sorry

end odd_function_expression_l80_80547


namespace speed_of_journey_l80_80765

-- Define the conditions
def journey_time : ℕ := 10
def journey_distance : ℕ := 200
def half_journey_distance : ℕ := journey_distance / 2

-- Define the hypothesis that the journey is split into two equal parts, each traveled at the same speed
def equal_speed (v : ℕ) : Prop :=
  (half_journey_distance / v) + (half_journey_distance / v) = journey_time

-- Prove the speed v is 20 km/hr given the conditions
theorem speed_of_journey : ∃ v : ℕ, equal_speed v ∧ v = 20 :=
by
  have h : equal_speed 20 := sorry
  exact ⟨20, h, rfl⟩

end speed_of_journey_l80_80765


namespace parakeets_in_each_cage_l80_80239

variable (num_cages : ℕ) (parrots_per_cage : ℕ) (total_birds : ℕ)

-- Given conditions
def total_parrots (num_cages parrots_per_cage : ℕ) : ℕ := num_cages * parrots_per_cage
def total_parakeets (total_birds total_parrots : ℕ) : ℕ := total_birds - total_parrots
def parakeets_per_cage (total_parakeets num_cages : ℕ) : ℕ := total_parakeets / num_cages

-- Theorem: Number of parakeets in each cage is 7
theorem parakeets_in_each_cage (h1 : num_cages = 8) (h2 : parrots_per_cage = 2) (h3 : total_birds = 72) : 
  parakeets_per_cage (total_parakeets total_birds (total_parrots num_cages parrots_per_cage)) num_cages = 7 :=
by
  sorry

end parakeets_in_each_cage_l80_80239


namespace multiple_of_interest_rate_l80_80513

theorem multiple_of_interest_rate (P r m : ℝ) (h1 : P * r^2 = 40) (h2 : P * (m * r)^2 = 360) : m = 3 :=
by
  sorry

end multiple_of_interest_rate_l80_80513


namespace slope_of_tangent_at_minus_1_l80_80689

theorem slope_of_tangent_at_minus_1
  (c : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (x - 2) * (x^2 + c))
  (h_extremum : deriv f 1 = 0) :
  deriv f (-1) = 8 :=
by
  sorry

end slope_of_tangent_at_minus_1_l80_80689


namespace trigonometric_expression_simplification_l80_80368

theorem trigonometric_expression_simplification
  (α : ℝ) 
  (hα : α = 49 * Real.pi / 48) :
  4 * (Real.sin α ^ 3 * Real.cos (3 * α) + 
       Real.cos α ^ 3 * Real.sin (3 * α)) * 
  Real.cos (4 * α) = 0.75 := 
by 
  sorry

end trigonometric_expression_simplification_l80_80368


namespace jessies_current_weight_l80_80289

theorem jessies_current_weight (initial_weight lost_weight : ℝ) (h1 : initial_weight = 69) (h2 : lost_weight = 35) :
  initial_weight - lost_weight = 34 :=
by sorry

end jessies_current_weight_l80_80289


namespace algebraic_inequality_solution_l80_80154

theorem algebraic_inequality_solution (x : ℝ) : (1 + 2 * x ≤ 8 + 3 * x) → (x ≥ -7) :=
by
  sorry

end algebraic_inequality_solution_l80_80154


namespace regular_hexagon_area_l80_80377

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem regular_hexagon_area 
  (A C : ℝ × ℝ)
  (hA : A = (0, 0))
  (hC : C = (8, 2))
  (h_eq_side_length : ∀ x y : ℝ × ℝ, dist A.1 A.2 C.1 C.2 = dist x.1 x.2 y.1 y.2) :
  hexagon_area = 34 * Real.sqrt 3 :=
by
  -- sorry indicates the proof is omitted
  sorry

end regular_hexagon_area_l80_80377


namespace dentist_age_considered_years_ago_l80_80041

theorem dentist_age_considered_years_ago (A : ℕ) (X : ℕ) (H1 : A = 32) (H2 : (1/6 : ℚ) * (A - X) = (1/10 : ℚ) * (A + 8)) : X = 8 :=
sorry

end dentist_age_considered_years_ago_l80_80041


namespace mike_age_proof_l80_80031

theorem mike_age_proof (a m : ℝ) (h1 : m = 3 * a - 20) (h2 : m + a = 70) : m = 47.5 := 
by {
  sorry
}

end mike_age_proof_l80_80031


namespace reciprocal_neg_2023_l80_80799

theorem reciprocal_neg_2023 : (1 / (-2023: ℤ)) = - (1 / 2023) :=
by
  -- proof goes here
  sorry

end reciprocal_neg_2023_l80_80799


namespace arccos_one_half_l80_80914

theorem arccos_one_half : Real.arccos (1 / 2) = Real.pi / 3 :=
by
  sorry

end arccos_one_half_l80_80914


namespace prime_sum_of_primes_l80_80576

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def distinct_primes (p q r s : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem prime_sum_of_primes (p q r s : ℕ) :
  distinct_primes p q r s →
  is_prime (p + q + r + s) →
  is_square (p^2 + q * s) →
  is_square (p^2 + q * r) →
  (p = 2 ∧ q = 7 ∧ r = 11 ∧ s = 3) ∨ (p = 2 ∧ q = 7 ∧ r = 3 ∧ s = 11) :=
by
  sorry

end prime_sum_of_primes_l80_80576


namespace sequence_formula_l80_80317

theorem sequence_formula (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (h : ∀ n, S_n n = 3 + 2 * a_n n) :
  ∀ n, a_n n = -3 * 2^(n - 1) :=
by
  sorry

end sequence_formula_l80_80317


namespace part1_part2_l80_80652

noncomputable def A_m (m : ℕ) (k : ℕ) : ℕ := (2 * k - 1) * m + k

theorem part1 (m : ℕ) (hm : m ≥ 2) :
  ∃ a : ℕ, 1 ≤ a ∧ a < m ∧ (∃ k : ℕ, 2^a = A_m m k) ∨ (∃ k : ℕ, 2^a + 1 = A_m m k) :=
sorry

theorem part2 {m : ℕ} (hm : m ≥ 2) 
  (a b : ℕ) (ha : ∃ k, 2^a = A_m m k) (hb : ∃ k, 2^b + 1 = A_m m k)
  (hmin_a : ∀ x, (∃ k, 2^x = A_m m k) → a ≤ x) 
  (hmin_b : ∀ y, (∃ k, 2^y + 1 = A_m m k) → b ≤ y) :
  a = 2 * b + 1 :=
sorry

end part1_part2_l80_80652


namespace correct_subtraction_result_l80_80925

theorem correct_subtraction_result (n : ℕ) (h : 40 / n = 5) : 20 - n = 12 := by
sorry

end correct_subtraction_result_l80_80925


namespace count_multiples_of_7_not_14_l80_80330

theorem count_multiples_of_7_not_14 (n : ℕ) : (n < 500 ∧ n % 7 = 0 ∧ n % 14 ≠ 0) → ∃ (k : ℕ), k = 36 :=
by
  sorry

end count_multiples_of_7_not_14_l80_80330


namespace devin_initial_height_l80_80240

theorem devin_initial_height (h : ℝ) (p : ℝ) (p' : ℝ) :
  (p = 10 / 100) →
  (p' = (h - 66) / 100) →
  (h + 3 = 68) →
  (p + p' * (h + 3 - 66) = 30 / 100) →
  h = 68 :=
by
  intros hp hp' hg pt
  sorry

end devin_initial_height_l80_80240


namespace stereos_production_fraction_l80_80804

/-
Company S produces three kinds of stereos: basic, deluxe, and premium.
Of the stereos produced by Company S last month, 2/5 were basic, 3/10 were deluxe, and the rest were premium.
It takes 1.6 as many hours to produce a deluxe stereo as it does to produce a basic stereo, and 2.5 as many hours to produce a premium stereo as it does to produce a basic stereo.
Prove that the number of hours it took to produce the deluxe and premium stereos last month was 123/163 of the total number of hours it took to produce all the stereos.
-/

def stereos_production (total_stereos : ℕ) (basic_ratio deluxe_ratio : ℚ)
  (deluxe_time_multiplier premium_time_multiplier : ℚ) : ℚ :=
  let basic_stereos := total_stereos * basic_ratio
  let deluxe_stereos := total_stereos * deluxe_ratio
  let premium_stereos := total_stereos - basic_stereos - deluxe_stereos
  let basic_time := basic_stereos
  let deluxe_time := deluxe_stereos * deluxe_time_multiplier
  let premium_time := premium_stereos * premium_time_multiplier
  let total_time := basic_time + deluxe_time + premium_time
  (deluxe_time + premium_time) / total_time

-- Given values
def total_stereos : ℕ := 100
def basic_ratio : ℚ := 2 / 5
def deluxe_ratio : ℚ := 3 / 10
def deluxe_time_multiplier : ℚ := 1.6
def premium_time_multiplier : ℚ := 2.5

theorem stereos_production_fraction : stereos_production total_stereos basic_ratio deluxe_ratio deluxe_time_multiplier premium_time_multiplier = 123 / 163 := by
  sorry

end stereos_production_fraction_l80_80804


namespace AllieMoreGrapes_l80_80372

-- Definitions based on conditions
def RobBowl : ℕ := 25
def TotalGrapes : ℕ := 83
def AllynBowl (A : ℕ) : ℕ := A + 4

-- The proof statement that must be shown.
theorem AllieMoreGrapes (A : ℕ) (h1 : A + (AllynBowl A) + RobBowl = TotalGrapes) : A - RobBowl = 2 :=
by {
  sorry
}

end AllieMoreGrapes_l80_80372


namespace pyramid_certain_height_l80_80742

noncomputable def certain_height (h : ℝ) : Prop :=
  let height := h + 20
  let width := height + 234
  (height + width = 1274) → h = 1000 / 3

theorem pyramid_certain_height (h : ℝ) : certain_height h :=
by
  let height := h + 20
  let width := height + 234
  have h_eq : (height + width = 1274) → h = 1000 / 3 := sorry
  exact h_eq

end pyramid_certain_height_l80_80742


namespace grace_walks_distance_l80_80218

theorem grace_walks_distance
  (south_blocks west_blocks : ℕ)
  (block_length_in_miles : ℚ)
  (h_south_blocks : south_blocks = 4)
  (h_west_blocks : west_blocks = 8)
  (h_block_length : block_length_in_miles = 1 / 4)
  : ((south_blocks + west_blocks) * block_length_in_miles = 3) :=
by 
  sorry

end grace_walks_distance_l80_80218


namespace part1_solution_set_part2_value_of_t_l80_80329

open Real

def f (t x : ℝ) : ℝ := x^2 - (t + 1) * x + t

-- Statement for the equivalent proof problem
theorem part1_solution_set (x : ℝ) : 
  (t = 3 → f 3 x > 0 ↔ (x < 1) ∨ (x > 3)) :=
by
  sorry

theorem part2_value_of_t (t : ℝ) :
  (∀ x : ℝ, f t x ≥ 0) → t = 1 :=
by
  sorry

end part1_solution_set_part2_value_of_t_l80_80329


namespace below_sea_level_notation_l80_80016

theorem below_sea_level_notation (depth_above_sea_level : Int) (depth_below_sea_level: Int) 
  (h: depth_above_sea_level = 9050 ∧ depth_below_sea_level = -10907) : 
  depth_below_sea_level = -10907 :=
by 
  sorry

end below_sea_level_notation_l80_80016


namespace circles_internally_tangent_l80_80550

theorem circles_internally_tangent :
  let C1 := (3, -2)
  let r1 := 1
  let C2 := (7, 1)
  let r2 := 6
  let d := Real.sqrt (((7 - 3)^2 + (1 - (-2))^2) : ℝ)
  d = r2 - r1 :=
by
  sorry

end circles_internally_tangent_l80_80550


namespace spaceship_speed_conversion_l80_80774

theorem spaceship_speed_conversion (speed_km_per_sec : ℕ) (seconds_in_hour : ℕ) (correct_speed_km_per_hour : ℕ) :
  speed_km_per_sec = 12 →
  seconds_in_hour = 3600 →
  correct_speed_km_per_hour = 43200 →
  speed_km_per_sec * seconds_in_hour = correct_speed_km_per_hour := by
  sorry

end spaceship_speed_conversion_l80_80774
