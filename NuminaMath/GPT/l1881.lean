import Mathlib

namespace greatest_integer_value_x_l1881_188149

theorem greatest_integer_value_x :
  ∃ x : ℤ, (8 - 3 * (2 * x + 1) > 26) ∧ ∀ y : ℤ, (8 - 3 * (2 * y + 1) > 26) → y ≤ x :=
sorry

end greatest_integer_value_x_l1881_188149


namespace base_conversion_difference_l1881_188147

-- Definitions
def base9_to_base10 (n : ℕ) : ℕ := 3 * (9^2) + 2 * (9^1) + 7 * (9^0)
def base8_to_base10 (m : ℕ) : ℕ := 2 * (8^2) + 5 * (8^1) + 3 * (8^0)

-- Statement
theorem base_conversion_difference :
  base9_to_base10 327 - base8_to_base10 253 = 97 :=
by sorry

end base_conversion_difference_l1881_188147


namespace part1_solution_l1881_188140

theorem part1_solution : ∀ n : ℕ, ∃ k : ℤ, 2^n + 3 = k^2 ↔ n = 0 :=
by sorry

end part1_solution_l1881_188140


namespace range_of_a_l1881_188122

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + (1 / 2) * x ^ 2 + a * x

theorem range_of_a (a : ℝ) : (∃ x > 0, deriv (f a) x = 3) ↔ a < 1 := by
  sorry

end range_of_a_l1881_188122


namespace arithmetic_sequence_statements_l1881_188198

/-- 
Given the arithmetic sequence {a_n} with first term a_1 > 0 and the sum of the first n terms denoted as S_n, 
prove the following statements based on the condition S_8 = S_16:
  1. d > 0
  2. a_{13} < 0
  3. The maximum value of S_n is S_{12}
  4. When S_n < 0, the minimum value of n is 25
--/
theorem arithmetic_sequence_statements (a_1 d : ℤ) (S : ℕ → ℤ)
  (h1 : a_1 > 0)
  (h2 : S 8 = S 16)
  (hS8 : S 8 = 8 * a_1 + 28 * d)
  (hS16 : S 16 = 16 * a_1 + 120 * d) :
  (d > 0) ∨ 
  (a_1 + 12 * d < 0) ∨ 
  (∀ n, n ≠ 12 → S n ≤ S 12) ∨ 
  (∀ n, S n < 0 → n ≥ 25) :=
sorry

end arithmetic_sequence_statements_l1881_188198


namespace find_B_value_l1881_188153

theorem find_B_value (A B : ℕ) : (A * 100 + B * 10 + 2) - 41 = 591 → B = 3 :=
by
  sorry

end find_B_value_l1881_188153


namespace percentage_of_boys_playing_soccer_l1881_188128

theorem percentage_of_boys_playing_soccer
  (total_students : ℕ)
  (boys : ℕ)
  (students_playing_soccer : ℕ)
  (girl_students_not_playing_soccer : ℕ)
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : students_playing_soccer = 250)
  (h4 : girl_students_not_playing_soccer = 89) :
  (students_playing_soccer - (total_students - boys - girl_students_not_playing_soccer)) * 100 / students_playing_soccer = 86 :=
by
  sorry

end percentage_of_boys_playing_soccer_l1881_188128


namespace total_number_of_dresses_l1881_188181

theorem total_number_of_dresses (ana_dresses lisa_more_dresses : ℕ) (h_condition : ana_dresses = 15) (h_more : lisa_more_dresses = ana_dresses + 18) : ana_dresses + lisa_more_dresses = 48 :=
by
  sorry

end total_number_of_dresses_l1881_188181


namespace natalia_total_distance_l1881_188172

theorem natalia_total_distance :
  let dist_mon := 40
  let bonus_mon := 0.05 * dist_mon
  let effective_mon := dist_mon + bonus_mon
  
  let dist_tue := 50
  let bonus_tue := 0.03 * dist_tue
  let effective_tue := dist_tue + bonus_tue
  
  let dist_wed := dist_tue / 2
  let bonus_wed := 0.07 * dist_wed
  let effective_wed := dist_wed + bonus_wed
  
  let dist_thu := dist_mon + dist_wed
  let bonus_thu := 0.04 * dist_thu
  let effective_thu := dist_thu + bonus_thu
  
  let dist_fri := 1.2 * dist_thu
  let bonus_fri := 0.06 * dist_fri
  let effective_fri := dist_fri + bonus_fri
  
  let dist_sat := 0.75 * dist_fri
  let bonus_sat := 0.02 * dist_sat
  let effective_sat := dist_sat + bonus_sat
  
  let dist_sun := dist_sat - dist_wed
  let bonus_sun := 0.10 * dist_sun
  let effective_sun := dist_sun + bonus_sun
  
  effective_mon + effective_tue + effective_wed + effective_thu + effective_fri + effective_sat + effective_sun = 367.05 :=
by
  sorry

end natalia_total_distance_l1881_188172


namespace find_f_2018_l1881_188141

-- Define the function f, its periodicity and even property
variable (f : ℝ → ℝ)

-- Conditions
axiom f_periodicity : ∀ x : ℝ, f (x + 4) = -f x
axiom f_symmetric : ∀ x : ℝ, f x = f (-x)
axiom f_at_two : f 2 = 2

-- Theorem stating the desired property
theorem find_f_2018 : f 2018 = 2 :=
  sorry

end find_f_2018_l1881_188141


namespace initial_apples_count_l1881_188146

theorem initial_apples_count (a b : ℕ) (h₁ : b = 13) (h₂ : b = a + 5) : a = 8 :=
by
  sorry

end initial_apples_count_l1881_188146


namespace unique_solution_p_eq_neg8_l1881_188189

theorem unique_solution_p_eq_neg8 (p : ℝ) (h : ∀ y : ℝ, 2 * y^2 - 8 * y - p = 0 → ∃! y : ℝ, 2 * y^2 - 8 * y - p = 0) : p = -8 :=
sorry

end unique_solution_p_eq_neg8_l1881_188189


namespace evaluate_expression_l1881_188160

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) : (a^3 + b)^2 - (a^3 - b)^2 = 216 :=
by
  -- Proof is not required, add sorry to skip the proof
  sorry

end evaluate_expression_l1881_188160


namespace find_angle_between_vectors_l1881_188118

open Real

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem find_angle_between_vectors 
  (a b : ℝ × ℝ)
  (a_nonzero : a ≠ (0, 0))
  (b_nonzero : b ≠ (0, 0))
  (ha : vector_norm a = 2)
  (hb : vector_norm b = 3)
  (h_sum : vector_norm (a.1 + b.1, a.2 + b.2) = 1)
  : arccos (dot_product a b / (vector_norm a * vector_norm b)) = π :=
sorry

end find_angle_between_vectors_l1881_188118


namespace sum_of_arithmetic_sequence_is_constant_l1881_188114

def is_constant (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, S n = c

theorem sum_of_arithmetic_sequence_is_constant
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 2 + a 6 + a 10 = a 1 + d + a 1 + 5 * d + a 1 + 9 * d)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  is_constant 11 a S :=
by
  sorry

end sum_of_arithmetic_sequence_is_constant_l1881_188114


namespace elderly_people_not_set_l1881_188152

def is_well_defined (S : Set α) : Prop := Nonempty S

def all_positive_numbers : Set ℝ := {x : ℝ | 0 < x}
def real_numbers_non_zero : Set ℝ := {x : ℝ | x ≠ 0}
def four_great_inventions : Set String := {"compass", "gunpowder", "papermaking", "printing"}

def elderly_people_description : String := "elderly people"

theorem elderly_people_not_set :
  ¬ (∃ S : Set α, elderly_people_description = "elderly people" ∧ is_well_defined S) :=
sorry

end elderly_people_not_set_l1881_188152


namespace rectangles_with_equal_perimeters_can_have_different_shapes_l1881_188162

theorem rectangles_with_equal_perimeters_can_have_different_shapes (l₁ w₁ l₂ w₂ : ℝ) 
  (h₁ : l₁ + w₁ = l₂ + w₂) : (l₁ ≠ l₂ ∨ w₁ ≠ w₂) :=
by
  sorry

end rectangles_with_equal_perimeters_can_have_different_shapes_l1881_188162


namespace factors_of_48_are_multiples_of_6_l1881_188119

theorem factors_of_48_are_multiples_of_6 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d, d ∣ 48 → (6 ∣ d ↔ d = 6 ∨ d = 12 ∨ d = 24 ∨ d = 48) := 
by { sorry }

end factors_of_48_are_multiples_of_6_l1881_188119


namespace rainfall_on_wednesday_l1881_188177

theorem rainfall_on_wednesday 
  (rain_on_monday : ℝ)
  (rain_on_tuesday : ℝ)
  (total_rain : ℝ) 
  (hmonday : rain_on_monday = 0.16666666666666666) 
  (htuesday : rain_on_tuesday = 0.4166666666666667) 
  (htotal : total_rain = 0.6666666666666666) :
  total_rain - (rain_on_monday + rain_on_tuesday) = 0.0833333333333333 :=
by
  -- Proof would go here
  sorry

end rainfall_on_wednesday_l1881_188177


namespace fraction_div_add_result_l1881_188138

theorem fraction_div_add_result : 
  (2 / 3) / (4 / 5) + (1 / 2) = (4 / 3) := 
by 
  sorry

end fraction_div_add_result_l1881_188138


namespace gcd_of_three_numbers_l1881_188109

theorem gcd_of_three_numbers :
  Nat.gcd (Nat.gcd 72 120) 168 = 24 :=
sorry

end gcd_of_three_numbers_l1881_188109


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l1881_188105

-- Define the first theorem
theorem solve_quadratic_1 (x : ℝ) : x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the second theorem
theorem solve_quadratic_2 (x : ℝ) : 25*x^2 - 36 = 0 ↔ x = 6/5 ∨ x = -6/5 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the third theorem
theorem solve_quadratic_3 (x : ℝ) : x^2 + 10*x + 21 = 0 ↔ x = -3 ∨ x = -7 := 
by {
  -- We assume this proof is provided
  sorry
}

-- Define the fourth theorem
theorem solve_quadratic_4 (x : ℝ) : (x-3)^2 + 2*x*(x-3) = 0 ↔ x = 3 ∨ x = 1 := 
by {
  -- We assume this proof is provided
  sorry
}

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_solve_quadratic_4_l1881_188105


namespace max_a_is_fractional_value_l1881_188187

theorem max_a_is_fractional_value (a k : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 - (k^2 - 5 * a * k + 3) * x + 7)
  (h_k : 0 ≤ k ∧ k ≤ 2)
  (x1 x2 : ℝ)
  (h_x1 : k ≤ x1 ∧ x1 ≤ k + a)
  (h_x2 : k + 2 * a ≤ x2 ∧ x2 ≤ k + 4 * a)
  (h_fx1_fx2 : f x1 ≥ f x2) :
  a = (2 * Real.sqrt 6 - 4) / 5 :=
sorry

end max_a_is_fractional_value_l1881_188187


namespace investment_amount_l1881_188104

theorem investment_amount (R T V : ℝ) (hT : T = 0.9 * R) (hV : V = 0.99 * R) (total_sum : R + T + V = 6936) : R = 2400 :=
by sorry

end investment_amount_l1881_188104


namespace cos_alpha_l1881_188196

-- Define the conditions
variable (α : Real)
variable (x y r : Real)
-- Given the point (-3, 4)
def point_condition (x : Real) (y : Real) : Prop := x = -3 ∧ y = 4

-- Define r as the distance
def radius_condition (x y r : Real) : Prop := r = Real.sqrt (x ^ 2 + y ^ 2)

-- Prove that cos α and cos 2α are the given values
theorem cos_alpha (α : Real) (x y r : Real) (h1 : point_condition x y) (h2 : radius_condition x y r) :
  Real.cos α = -3 / 5 ∧ Real.cos (2 * α) = -7 / 25 :=
by
  sorry

end cos_alpha_l1881_188196


namespace int_even_bijection_l1881_188144

theorem int_even_bijection :
  ∃ (f : ℤ → ℤ), (∀ n : ℤ, ∃ m : ℤ, f n = m ∧ m % 2 = 0) ∧
                 (∀ m : ℤ, m % 2 = 0 → ∃ n : ℤ, f n = m) := 
sorry

end int_even_bijection_l1881_188144


namespace polar_bear_daily_salmon_consumption_l1881_188176

/-- Polar bear's fish consumption conditions and daily salmon amount calculation -/
theorem polar_bear_daily_salmon_consumption (h1: ℝ) (h2: ℝ) : 
  (h1 = 0.2) → (h2 = 0.6) → (h2 - h1 = 0.4) :=
by
  sorry

end polar_bear_daily_salmon_consumption_l1881_188176


namespace max_val_xa_minus_2x_max_val_ab_plus_bc_plus_ac_l1881_188155

-- First proof problem
theorem max_val_xa_minus_2x (x a : ℝ) (h1 : 0 < x) (h2 : 2 * x < a) :
  ∃ y, (y = x * (a - 2 * x)) ∧ y ≤ a^2 / 8 :=
sorry

-- Second proof problem
theorem max_val_ab_plus_bc_plus_ac (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 4) :
  ab + bc + ac ≤ 4 :=
sorry

end max_val_xa_minus_2x_max_val_ab_plus_bc_plus_ac_l1881_188155


namespace log_expression_eq_l1881_188111

theorem log_expression_eq (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log (y^4)) * 
  (Real.log (y^3) / Real.log (x^6)) * 
  (Real.log (x^4) / Real.log (y^3)) * 
  (Real.log (y^4) / Real.log (x^2)) * 
  (Real.log (x^6) / Real.log y) = 
  16 * Real.log x / Real.log y := 
sorry

end log_expression_eq_l1881_188111


namespace M_inter_N_eq_l1881_188125

-- Definitions based on the problem conditions
def M : Set ℝ := { x | abs x ≥ 3 }
def N : Set ℝ := { y | ∃ x ∈ M, y = x^2 }

-- The statement we want to prove
theorem M_inter_N_eq : M ∩ N = { x : ℝ | x ≥ 3 } :=
by
  sorry

end M_inter_N_eq_l1881_188125


namespace outfit_choices_l1881_188121

noncomputable def calculate_outfits : Nat :=
  let shirts := 6
  let pants := 6
  let hats := 6
  let total_outfits := shirts * pants * hats
  let matching_colors := 4 -- tan, black, blue, gray for matching
  total_outfits - matching_colors

theorem outfit_choices : calculate_outfits = 212 :=
by
  sorry

end outfit_choices_l1881_188121


namespace no_solution_l1881_188170

theorem no_solution (n : ℕ) (k : ℕ) (hn : Prime n) (hk : 0 < k) :
  ¬ (n ≤ n.factorial - k ^ n ∧ n.factorial - k ^ n ≤ k * n) :=
by
  sorry

end no_solution_l1881_188170


namespace region_relation_l1881_188110

theorem region_relation (A B C : ℝ)
  (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39)
  (h_triangle : a^2 + b^2 = c^2)
  (h_right_triangle : true) -- Since the triangle is already confirmed as right-angle
  (h_A : A = (π * (c / 2)^2 / 2 - 270) / 2)
  (h_B : B = (π * (c / 2)^2 / 2 - 270) / 2)
  (h_C : C = π * (c / 2)^2 / 2) :
  A + B + 270 = C :=
by
  sorry

end region_relation_l1881_188110


namespace plane_equation_through_point_and_line_l1881_188150

theorem plane_equation_through_point_and_line :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd A B = 1 ∧ Int.gcd A C = 1 ∧ Int.gcd A D = 1 ∧
  ∀ (x y z : ℝ),
    (A * x + B * y + C * z + D = 0 ↔ 
    (∃ (t : ℝ), x = -3 * t - 1 ∧ y = 2 * t + 3 ∧ z = t - 2) ∨ 
    (x = 0 ∧ y = 7 ∧ z = -7)) :=
by
  -- sorry, implementing proofs is not required.
  sorry

end plane_equation_through_point_and_line_l1881_188150


namespace mixed_fraction_product_l1881_188113

theorem mixed_fraction_product (X Y : ℕ) (hX : X ≠ 0) (hY : Y ≠ 0) :
  (5 + (1 / X : ℚ)) * (Y + (1 / 2 : ℚ)) = 43 ↔ X = 17 ∧ Y = 8 := 
by 
  sorry

end mixed_fraction_product_l1881_188113


namespace maps_skipped_l1881_188190

-- Definitions based on conditions
def total_pages := 372
def pages_read := 125
def pages_left := 231

-- Statement to be proven
theorem maps_skipped : total_pages - (pages_read + pages_left) = 16 :=
by
  sorry

end maps_skipped_l1881_188190


namespace problem_1_l1881_188131

noncomputable def f (a x : ℝ) : ℝ := abs (x + 2) + abs (x - a)

theorem problem_1 (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 3) : a ≤ -5 ∨ a ≥ 1 :=
by
  sorry

end problem_1_l1881_188131


namespace range_of_function_l1881_188167

theorem range_of_function :
  (∀ y : ℝ, (∃ x : ℝ, y = (x + 1) / (x ^ 2 + 1)) ↔ 0 ≤ y ∧ y ≤ 4/3) :=
by
  sorry

end range_of_function_l1881_188167


namespace work_completion_problem_l1881_188173

theorem work_completion_problem :
  (∃ x : ℕ, 9 * (1 / 45 + 1 / x) + 23 * (1 / x) = 1) → x = 40 :=
sorry

end work_completion_problem_l1881_188173


namespace abe_age_sum_l1881_188159

theorem abe_age_sum (h : abe_age = 29) : abe_age + (abe_age - 7) = 51 :=
by
  sorry

end abe_age_sum_l1881_188159


namespace total_horse_food_l1881_188143

theorem total_horse_food (ratio_sh_to_h : ℕ → ℕ → Prop) 
    (sheep : ℕ) 
    (ounce_per_horse : ℕ) 
    (total_ounces_per_day : ℕ) : 
    ratio_sh_to_h 5 7 → sheep = 40 → ounce_per_horse = 230 → total_ounces_per_day = 12880 :=
by
  intros h_ratio h_sheep h_ounce
  sorry

end total_horse_food_l1881_188143


namespace publishing_company_break_even_l1881_188161

theorem publishing_company_break_even : 
  ∀ (F V P : ℝ) (x : ℝ), F = 35630 ∧ V = 11.50 ∧ P = 20.25 →
  (P * x = F + V * x) → x = 4074 :=
by
  intros F V P x h_eq h_rev
  sorry

end publishing_company_break_even_l1881_188161


namespace cosine_sum_identity_l1881_188112

theorem cosine_sum_identity 
  (α : ℝ) 
  (h_sin : Real.sin α = 3 / 5) 
  (h_alpha_first_quad : 0 < α ∧ α < Real.pi / 2) : 
  Real.cos (Real.pi / 3 + α) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end cosine_sum_identity_l1881_188112


namespace books_from_second_shop_l1881_188120

theorem books_from_second_shop (x : ℕ) (h₁ : 6500 + 2000 = 8500)
    (h₂ : 85 = 8500 / (65 + x)) : x = 35 :=
by
  -- proof goes here
  sorry

end books_from_second_shop_l1881_188120


namespace expression_divisible_by_a_square_l1881_188157

theorem expression_divisible_by_a_square (n : ℕ) (a : ℤ) : 
  a^2 ∣ ((a * n - 1) * (a + 1) ^ n + 1) := 
sorry

end expression_divisible_by_a_square_l1881_188157


namespace sum_of_first_110_terms_l1881_188164

theorem sum_of_first_110_terms
  (a d : ℝ)
  (h1 : (10 : ℝ) * (2 * a + (10 - 1) * d) / 2 = 100)
  (h2 : (100 : ℝ) * (2 * a + (100 - 1) * d) / 2 = 10) :
  (110 : ℝ) * (2 * a + (110 - 1) * d) / 2 = -110 :=
  sorry

end sum_of_first_110_terms_l1881_188164


namespace find_q_l1881_188191

def P (q x : ℝ) : ℝ := x^4 + 2 * q * x^3 - 3 * x^2 + 2 * q * x + 1

theorem find_q (q : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 < 0 ∧ x2 < 0 ∧ P q x1 = 0 ∧ P q x2 = 0) → q < 1 / 4 :=
by
  sorry

end find_q_l1881_188191


namespace find_three_digit_number_l1881_188192

noncomputable def three_digit_number := ∃ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ 100 * x + 10 * y + z = 345 ∧
  (100 * z + 10 * y + x = 100 * x + 10 * y + z + 198) ∧
  (100 * x + 10 * z + y = 100 * x + 10 * y + z + 9) ∧
  (x^2 + y^2 + z^2 - 2 = 4 * (x + y + z))

theorem find_three_digit_number : three_digit_number :=
sorry

end find_three_digit_number_l1881_188192


namespace factorization1_factorization2_factorization3_factorization4_l1881_188134

-- Question 1
theorem factorization1 (a b : ℝ) :
  4 * a^2 * b - 6 * a * b^2 = 2 * a * b * (2 * a - 3 * b) :=
by 
  sorry

-- Question 2
theorem factorization2 (x y : ℝ) :
  25 * x^2 - 9 * y^2 = (5 * x + 3 * y) * (5 * x - 3 * y) :=
by 
  sorry

-- Question 3
theorem factorization3 (a b : ℝ) :
  2 * a^2 * b - 8 * a * b^2 + 8 * b^3 = 2 * b * (a - 2 * b)^2 :=
by 
  sorry

-- Question 4
theorem factorization4 (x : ℝ) :
  (x + 2) * (x - 8) + 25 = (x - 3)^2 :=
by 
  sorry

end factorization1_factorization2_factorization3_factorization4_l1881_188134


namespace robins_hair_cut_l1881_188136

theorem robins_hair_cut (x : ℕ) : 16 - x + 12 = 17 → x = 11 := by
  sorry

end robins_hair_cut_l1881_188136


namespace integer_representation_l1881_188127

theorem integer_representation (n : ℤ) : ∃ x y z : ℤ, n = x^2 + y^2 - z^2 :=
by sorry

end integer_representation_l1881_188127


namespace geometric_sequence_problem_l1881_188117

section 
variables (a : ℕ → ℝ) (r : ℝ) 

-- Condition: {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n : ℕ, a (n + 1) = a n * r

-- Condition: a_4 + a_6 = 8
axiom a4_a6_sum : a 4 + a 6 = 8

-- Mathematical equivalent proof problem
theorem geometric_sequence_problem (h : is_geometric_sequence a r) : 
  a 1 * a 7 + 2 * a 3 * a 7 + a 3 * a 9 = 64 :=
sorry

end

end geometric_sequence_problem_l1881_188117


namespace cube_surface_area_of_same_volume_as_prism_l1881_188154

theorem cube_surface_area_of_same_volume_as_prism :
  let prism_length := 10
  let prism_width := 5
  let prism_height := 24
  let prism_volume := prism_length * prism_width * prism_height
  let cube_edge := (prism_volume : ℝ)^(1/3)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 677.76 := by
  sorry

end cube_surface_area_of_same_volume_as_prism_l1881_188154


namespace time_to_produce_one_item_l1881_188116

-- Definitions based on the conditions
def itemsProduced : Nat := 300
def totalTimeHours : ℝ := 2.0
def minutesPerHour : ℝ := 60.0

-- The statement we need to prove
theorem time_to_produce_one_item : (totalTimeHours / itemsProduced * minutesPerHour) = 0.4 := by
  sorry

end time_to_produce_one_item_l1881_188116


namespace inf_many_solutions_to_ineq_l1881_188171

theorem inf_many_solutions_to_ineq (x : ℕ) : (15 < 2 * x + 20) ↔ x ≥ 1 :=
by
  sorry

end inf_many_solutions_to_ineq_l1881_188171


namespace maximize_box_volume_l1881_188100

-- Define the volume function
def volume (x : ℝ) := (48 - 2 * x)^2 * x

-- Define the constraint on x
def constraint (x : ℝ) := 0 < x ∧ x < 24

-- The theorem stating the side length of the removed square that maximizes the volume
theorem maximize_box_volume : ∃ x : ℝ, constraint x ∧ (∀ y : ℝ, constraint y → volume y ≤ volume 8) :=
by
  sorry

end maximize_box_volume_l1881_188100


namespace focal_distance_of_ellipse_l1881_188106

theorem focal_distance_of_ellipse : 
  ∀ (θ : ℝ), (∃ (c : ℝ), (x = 5 * Real.cos θ ∧ y = 4 * Real.sin θ) → 2 * c = 6) :=
by
  sorry

end focal_distance_of_ellipse_l1881_188106


namespace percent_of_l1881_188193

theorem percent_of (Part Whole : ℕ) (Percent : ℕ) (hPart : Part = 120) (hWhole : Whole = 40) :
  Percent = (Part * 100) / Whole → Percent = 300 :=
by
  sorry

end percent_of_l1881_188193


namespace average_letters_per_day_l1881_188123

theorem average_letters_per_day (letters_tuesday : Nat) (letters_wednesday : Nat) (total_days : Nat) 
  (h_tuesday : letters_tuesday = 7) (h_wednesday : letters_wednesday = 3) (h_days : total_days = 2) : 
  (letters_tuesday + letters_wednesday) / total_days = 5 :=
by 
  sorry

end average_letters_per_day_l1881_188123


namespace find_difference_of_roots_l1881_188108

-- Define the conditions for the given problem
def larger_root_of_eq_1 (a : ℝ) : Prop :=
  (1998 * a) ^ 2 - 1997 * 1999 * a - 1 = 0

def smaller_root_of_eq_2 (b : ℝ) : Prop :=
  b ^ 2 + 1998 * b - 1999 = 0

-- Define the main problem with the proof obligation
theorem find_difference_of_roots (a b : ℝ) (h1: larger_root_of_eq_1 a) (h2: smaller_root_of_eq_2 b) : a - b = 2000 :=
sorry

end find_difference_of_roots_l1881_188108


namespace latus_rectum_equation_l1881_188183

theorem latus_rectum_equation (y x : ℝ) :
  y^2 = 4 * x → x = -1 :=
sorry

end latus_rectum_equation_l1881_188183


namespace eval_at_2_l1881_188194

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem eval_at_2 : f 2 = 62 := by
  sorry

end eval_at_2_l1881_188194


namespace methane_hydrate_scientific_notation_l1881_188135

theorem methane_hydrate_scientific_notation :
  (9.2 * 10^(-4)) = 0.00092 :=
by sorry

end methane_hydrate_scientific_notation_l1881_188135


namespace averageFishIs75_l1881_188115

-- Introduce the number of fish in Boast Pool
def BoastPool : ℕ := 75

-- Introduce the number of fish in Onum Lake
def OnumLake : ℕ := BoastPool + 25

-- Introduce the number of fish in Riddle Pond
def RiddlePond : ℕ := OnumLake / 2

-- Define the total number of fish in all three bodies of water
def totalFish : ℕ := BoastPool + OnumLake + RiddlePond

-- Define the average number of fish in all three bodies of water
def averageFish : ℕ := totalFish / 3

-- Prove that the average number of fish is 75
theorem averageFishIs75 : averageFish = 75 :=
by
  -- We need to provide the proof steps here but using sorry to skip
  sorry

end averageFishIs75_l1881_188115


namespace max_profit_l1881_188166

noncomputable def profit_function (x : ℕ) : ℝ :=
  if x ≤ 400 then
    300 * x - (1 / 2) * x^2 - 20000
  else
    60000 - 100 * x

theorem max_profit : 
  (∀ x ≥ 0, profit_function x ≤ 25000) ∧ (profit_function 300 = 25000) :=
by 
  sorry

end max_profit_l1881_188166


namespace angle_A_value_sin_BC_value_l1881_188186

open Real

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ 
  A + B + C = π 

theorem angle_A_value (A B C : ℝ) (h : triangleABC a b c A B C) (h1 : cos 2 * A - 3 * cos (B + C) = 1) : 
  A = π / 3 :=
sorry

theorem sin_BC_value (A B C S b c : ℝ) (h : triangleABC a b c A B C)
  (hA : A = π / 3) (hS : S = 5 * sqrt 3) (hb : b = 5) : 
  sin B * sin C = 5 / 7 :=
sorry

end angle_A_value_sin_BC_value_l1881_188186


namespace inequality_proof_l1881_188151

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ab * (a + b) + bc * (b + c) + ac * (a + c) ≥ 6 * abc := 
sorry

end inequality_proof_l1881_188151


namespace trig_identity_l1881_188188

theorem trig_identity : 4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by sorry

end trig_identity_l1881_188188


namespace determine_m_from_quadratic_l1881_188148

def is_prime (n : ℕ) := 2 ≤ n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem determine_m_from_quadratic (x1 x2 m : ℕ) (hx1_prime : is_prime x1) (hx2_prime : is_prime x2) 
    (h_roots : x1 + x2 = 1999) (h_product : x1 * x2 = m) : 
    m = 3994 := 
by 
    sorry

end determine_m_from_quadratic_l1881_188148


namespace distance_between_Sneezy_and_Grumpy_is_8_l1881_188197

variables (DS DV SP VP: ℕ) (SV: ℕ)

theorem distance_between_Sneezy_and_Grumpy_is_8
  (hDS : DS = 5)
  (hDV : DV = 4)
  (hSP : SP = 10)
  (hVP : VP = 17)
  (hSV_condition1 : SV + SP > VP)
  (hSV_condition2 : SV < DS + DV)
  (hSV_condition3 : 7 < SV) :
  SV = 8 := 
sorry

end distance_between_Sneezy_and_Grumpy_is_8_l1881_188197


namespace polygon_stats_l1881_188185

-- Definitions based on the problem's conditions
def total_number_of_polygons : ℕ := 207
def median_position : ℕ := 104
def m : ℕ := 14
def sum_of_squares_of_sides : ℕ := 2860
def mean_value : ℚ := sum_of_squares_of_sides / total_number_of_polygons
def mode_median : ℚ := 11.5

-- The proof statement
theorem polygon_stats (d μ M : ℚ)
  (h₁ : μ = mean_value)
  (h₂ : d = mode_median)
  (h₃ : M = m) :
  d < μ ∧ μ < M :=
by
  rw [h₁, h₂, h₃]
  -- The exact proof steps are omitted
  sorry

end polygon_stats_l1881_188185


namespace option_A_correct_l1881_188130

theorem option_A_correct (p : ℕ) (h1 : p > 1) (h2 : p % 2 = 1) : 
  (p - 1)^(p/2 - 1) - 1 ≡ 0 [MOD (p - 2)] :=
sorry

end option_A_correct_l1881_188130


namespace pipe_fills_entire_cistern_in_77_minutes_l1881_188169

-- Define the time taken to fill 1/11 of the cistern
def time_to_fill_one_eleven_cistern : ℕ := 7

-- Define the fraction of the cistern filled in a certain time
def fraction_filled (t : ℕ) : ℚ := t / time_to_fill_one_eleven_cistern * (1 / 11)

-- Define the problem statement
theorem pipe_fills_entire_cistern_in_77_minutes : 
  fraction_filled 77 = 1 := by
  sorry

end pipe_fills_entire_cistern_in_77_minutes_l1881_188169


namespace george_change_sum_l1881_188126

theorem george_change_sum :
  ∃ n m : ℕ,
    0 ≤ n ∧ n < 19 ∧
    0 ≤ m ∧ m < 10 ∧
    (7 + 5 * n) = (4 + 10 * m) ∧
    (7 + 5 * 14) + (4 + 10 * 7) = 144 :=
by
  -- We declare the problem stating that there exist natural numbers n and m within
  -- the given ranges such that the sums of valid change amounts add up to 144 cents.
  sorry

end george_change_sum_l1881_188126


namespace problem_solution_l1881_188129

variable (x y : ℝ)

theorem problem_solution :
  (x - y + 1) * (x - y - 1) = x^2 - 2 * x * y + y^2 - 1 :=
by
  sorry

end problem_solution_l1881_188129


namespace Pooja_speed_3_l1881_188107

variable (Roja_speed Pooja_speed : ℝ)
variable (t d : ℝ)

theorem Pooja_speed_3
  (h1 : Roja_speed = 6)
  (h2 : t = 4)
  (h3 : d = 36)
  (h4 : d = t * (Roja_speed + Pooja_speed)) :
  Pooja_speed = 3 :=
by
  sorry

end Pooja_speed_3_l1881_188107


namespace total_revenue_correct_l1881_188168

def KwikETaxCenter : Type := ℕ

noncomputable def federal_return_price : ℕ := 50
noncomputable def state_return_price : ℕ := 30
noncomputable def quarterly_business_taxes_price : ℕ := 80
noncomputable def international_return_price : ℕ := 100
noncomputable def value_added_service_price : ℕ := 75

noncomputable def federal_returns_sold : ℕ := 60
noncomputable def state_returns_sold : ℕ := 20
noncomputable def quarterly_returns_sold : ℕ := 10
noncomputable def international_returns_sold : ℕ := 13
noncomputable def value_added_services_sold : ℕ := 25

noncomputable def international_discount : ℕ := 20

noncomputable def calculate_total_revenue 
   (federal_price : ℕ) (state_price : ℕ) 
   (quarterly_price : ℕ) (international_price : ℕ) 
   (value_added_price : ℕ)
   (federal_sold : ℕ) (state_sold : ℕ) 
   (quarterly_sold : ℕ) (international_sold : ℕ) 
   (value_added_sold : ℕ)
   (discount : ℕ) : ℕ := 
    (federal_price * federal_sold) 
  + (state_price * state_sold) 
  + (quarterly_price * quarterly_sold) 
  + ((international_price - discount) * international_sold) 
  + (value_added_price * value_added_sold)

theorem total_revenue_correct :
  calculate_total_revenue federal_return_price state_return_price 
                          quarterly_business_taxes_price international_return_price 
                          value_added_service_price
                          federal_returns_sold state_returns_sold 
                          quarterly_returns_sold international_returns_sold 
                          value_added_services_sold 
                          international_discount = 7315 := 
  by sorry

end total_revenue_correct_l1881_188168


namespace log_x2y2_l1881_188133

theorem log_x2y2 (x y : ℝ) (h1 : Real.log (x^2 * y^5) = 2) (h2 : Real.log (x^3 * y^2) = 2) :
  Real.log (x^2 * y^2) = 16 / 11 :=
by
  sorry

end log_x2y2_l1881_188133


namespace probability_red_or_blue_l1881_188179

theorem probability_red_or_blue :
  ∀ (total_marbles white_marbles green_marbles red_blue_marbles : ℕ),
    total_marbles = 90 →
    (white_marbles : ℝ) / total_marbles = 1 / 6 →
    (green_marbles : ℝ) / total_marbles = 1 / 5 →
    white_marbles = 15 →
    green_marbles = 18 →
    red_blue_marbles = total_marbles - (white_marbles + green_marbles) →
    (red_blue_marbles : ℝ) / total_marbles = 19 / 30 :=
by
  intros total_marbles white_marbles green_marbles red_blue_marbles
  intros h_total_marbles h_white_prob h_green_prob h_white_count h_green_count h_red_blue_count
  sorry

end probability_red_or_blue_l1881_188179


namespace reduce_expression_l1881_188182

-- Define the variables a, b, c as real numbers
variables (a b c : ℝ)

-- State the theorem with the given condition that expressions are defined and non-zero
theorem reduce_expression :
  (a^2 + b^2 - c^2 - 2*a*b) / (a^2 + c^2 - b^2 - 2*a*c) = ((a - b + c) * (a - b - c)) / ((a - c + b) * (a - c - b)) :=
by
  sorry

end reduce_expression_l1881_188182


namespace range_of_c_for_two_distinct_roots_l1881_188132

theorem range_of_c_for_two_distinct_roots (c : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 3 * x1 + c = x1 + 2) ∧ (x2^2 - 3 * x2 + c = x2 + 2)) ↔ (c < 6) :=
sorry

end range_of_c_for_two_distinct_roots_l1881_188132


namespace total_tickets_sold_l1881_188165

def ticket_prices : Nat := 25
def senior_ticket_price : Nat := 15
def total_receipts : Nat := 9745
def senior_tickets_sold : Nat := 348
def adult_tickets_sold : Nat := (total_receipts - senior_ticket_price * senior_tickets_sold) / ticket_prices

theorem total_tickets_sold : adult_tickets_sold + senior_tickets_sold = 529 :=
by
  sorry

end total_tickets_sold_l1881_188165


namespace largest_angle_of_triangle_l1881_188184

theorem largest_angle_of_triangle (x : ℝ) 
  (h1 : 35 + 70 + x = 180) : 75 = max (max 35 70) x := 
sorry

end largest_angle_of_triangle_l1881_188184


namespace probability_in_given_interval_l1881_188156

noncomputable def length_interval (a b : ℝ) : ℝ := b - a

noncomputable def probability_in_interval (a b c d : ℝ) : ℝ :=
  (length_interval a b) / (length_interval c d)

theorem probability_in_given_interval : 
  probability_in_interval (-1) 1 (-2) 3 = 2 / 5 :=
by
  sorry

end probability_in_given_interval_l1881_188156


namespace solution_set_inequality_l1881_188195

theorem solution_set_inequality (a : ℝ) (x : ℝ) (h : 0 < a ∧ a < 1) : 
  ((a - x) * (x - 1 / a) > 0) ↔ (a < x ∧ x < 1 / a) :=
by
  sorry

end solution_set_inequality_l1881_188195


namespace common_number_of_two_sets_l1881_188158

theorem common_number_of_two_sets (a b c d e f g : ℚ) :
  (a + b + c + d) / 4 = 5 →
  (d + e + f + g) / 4 = 8 →
  (a + b + c + d + e + f + g) / 7 = 46 / 7 →
  d = 6 :=
by
  intros h₁ h₂ h₃
  sorry

end common_number_of_two_sets_l1881_188158


namespace systematic_sampling_method_l1881_188142

-- Define the problem conditions
def total_rows : Nat := 40
def seats_per_row : Nat := 25
def attendees_left (row : Nat) : Nat := if row < total_rows then 18 else 0

-- Problem statement to be proved: The method used is systematic sampling.
theorem systematic_sampling_method :
  (∀ r : Nat, r < total_rows → attendees_left r = 18) →
  (seats_per_row = 25) →
  (∃ k, k > 0 ∧ ∀ r, r < total_rows → attendees_left r = 18 + k * r) →
  True :=
by
  intro h1 h2 h3
  sorry

end systematic_sampling_method_l1881_188142


namespace fraction_subtraction_property_l1881_188175

variable (a b c d : ℚ)

theorem fraction_subtraction_property :
  (a / b - c / d) = ((a - c) / (b + d)) → (a / c) = (b / d) ^ 2 := 
by
  sorry

end fraction_subtraction_property_l1881_188175


namespace right_triangle_acute_angle_ratio_l1881_188102

theorem right_triangle_acute_angle_ratio (A B : ℝ) (h_ratio : A / B = 5 / 4) (h_sum : A + B = 90) :
  min A B = 40 :=
by
  -- Conditions are provided
  sorry

end right_triangle_acute_angle_ratio_l1881_188102


namespace number_of_positive_integers_with_positive_log_l1881_188163

theorem number_of_positive_integers_with_positive_log (b : ℕ) (h : ∃ n : ℕ, n > 0 ∧ b ^ n = 1024) : 
  ∃ L, L = 4 :=
sorry

end number_of_positive_integers_with_positive_log_l1881_188163


namespace part_a_part_b_l1881_188178

-- Part (a)
theorem part_a (x : ℝ) (h : x > 0) : x^3 - 3*x ≥ -2 :=
sorry

-- Part (b)
theorem part_b (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y / z) + (y^2 * z / x) + (z^2 * x / y) + 2 * ((y / (x * z)) + (z / (x * y)) + (x / (y * z))) ≥ 9 :=
sorry

end part_a_part_b_l1881_188178


namespace alyssa_final_money_l1881_188139

-- Definitions based on conditions
def weekly_allowance : Int := 8
def spent_on_movies : Int := weekly_allowance / 2
def earnings_from_washing_car : Int := 8

-- The statement to prove
def final_amount : Int := (weekly_allowance - spent_on_movies) + earnings_from_washing_car

-- The theorem expressing the problem
theorem alyssa_final_money : final_amount = 12 := by
  sorry

end alyssa_final_money_l1881_188139


namespace train_speed_l1881_188174

def length_of_train : ℝ := 250
def length_of_bridge : ℝ := 120
def time_taken : ℝ := 20
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def speed_of_train : ℝ := total_distance / time_taken

theorem train_speed : speed_of_train = 18.5 :=
  by sorry

end train_speed_l1881_188174


namespace triangle_shape_isosceles_or_right_l1881_188101

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the angles

theorem triangle_shape_isosceles_or_right (h1 : a^2 + b^2 ≠ 0) (h2 : 
  (a^2 + b^2) * Real.sin (A - B) 
  = (a^2 - b^2) * Real.sin (A + B))
  (h3 : ∀ (A B C : ℝ), A + B + C = π) :
  ∃ (isosceles : Bool), (isosceles = true) ∨ (isosceles = false ∧ A + B = π / 2) :=
sorry

end triangle_shape_isosceles_or_right_l1881_188101


namespace total_students_l1881_188124

theorem total_students (boys girls : ℕ) (h_ratio : boys / girls = 8 / 5) (h_girls : girls = 120) : boys + girls = 312 :=
by
  sorry

end total_students_l1881_188124


namespace equilateral_triangle_grid_l1881_188199

noncomputable def number_of_triangles (n : ℕ) : ℕ :=
1 + 3 + 5 + 7 + 9 + 1 + 2 + 3 + 4 + 3 + 1 + 2 + 3 + 1 + 2 + 1

theorem equilateral_triangle_grid (n : ℕ) (h : n = 5) : number_of_triangles n = 48 := by
  sorry

end equilateral_triangle_grid_l1881_188199


namespace find_b_squared_l1881_188180

-- Assume a and b are real numbers and positive
variables (a b : ℝ)
-- Given conditions
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom magnitude : a^2 + b^2 = 100
axiom equidistant : 2 * a - 4 * b = 7

-- Main proof statement
theorem find_b_squared : b^2 = 287 / 17 := sorry

end find_b_squared_l1881_188180


namespace cake_sector_chord_length_l1881_188137

noncomputable def sector_longest_chord_square (d : ℝ) (n : ℕ) : ℝ :=
  let r := d / 2
  let theta := (360 : ℝ) / n
  let chord_length := 2 * r * Real.sin (theta / 2 * Real.pi / 180)
  chord_length ^ 2

theorem cake_sector_chord_length :
  sector_longest_chord_square 18 5 = 111.9473 := by
  sorry

end cake_sector_chord_length_l1881_188137


namespace distinct_sum_of_five_integers_l1881_188103

theorem distinct_sum_of_five_integers 
  (a b c d e : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
  (h_condition : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = -120) : 
  a + b + c + d + e = 25 :=
sorry

end distinct_sum_of_five_integers_l1881_188103


namespace find_value_in_table_l1881_188145

theorem find_value_in_table :
  let W := 'W'
  let L := 'L'
  let Q := 'Q'
  let table := [
    [W, '?', Q],
    [L, Q, W],
    [Q, W, L]
  ]
  table[0][1] = L :=
by
  sorry

end find_value_in_table_l1881_188145
