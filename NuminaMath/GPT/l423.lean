import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_S6_l423_42315

noncomputable def sum_of_first_n_terms (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_S6 (a r : ℝ) (h1 : sum_of_first_n_terms a r 2 = 6) (h2 : sum_of_first_n_terms a r 4 = 30) : 
  sum_of_first_n_terms a r 6 = 126 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_S6_l423_42315


namespace NUMINAMATH_GPT_prime_p_perfect_cube_l423_42348

theorem prime_p_perfect_cube (p : ℕ) (hp : Nat.Prime p) (h : ∃ n : ℕ, 13 * p + 1 = n^3) :
  p = 2 ∨ p = 211 :=
by
  sorry

end NUMINAMATH_GPT_prime_p_perfect_cube_l423_42348


namespace NUMINAMATH_GPT_base_5_minus_base_8_in_base_10_l423_42396

def base_5 := 52143
def base_8 := 4310

theorem base_5_minus_base_8_in_base_10 :
  (5 * 5^4 + 2 * 5^3 + 1 * 5^2 + 4 * 5^1 + 3 * 5^0) -
  (4 * 8^3 + 3 * 8^2 + 1 * 8^1 + 0 * 8^0)
  = 1175 := by
  sorry

end NUMINAMATH_GPT_base_5_minus_base_8_in_base_10_l423_42396


namespace NUMINAMATH_GPT_Marilyn_has_40_bananas_l423_42300

-- Definitions of the conditions
def boxes : ℕ := 8
def bananas_per_box : ℕ := 5

-- Statement of the proof problem
theorem Marilyn_has_40_bananas : (boxes * bananas_per_box) = 40 := by
  sorry

end NUMINAMATH_GPT_Marilyn_has_40_bananas_l423_42300


namespace NUMINAMATH_GPT_max_stamps_l423_42340

theorem max_stamps (price_per_stamp : ℕ) (total_money : ℕ) (h_price : price_per_stamp = 37) (h_total : total_money = 4000) : 
  ∃ max_stamps : ℕ, max_stamps = 108 ∧ max_stamps * price_per_stamp ≤ total_money ∧ ∀ n : ℕ, n * price_per_stamp ≤ total_money → n ≤ max_stamps :=
by
  sorry

end NUMINAMATH_GPT_max_stamps_l423_42340


namespace NUMINAMATH_GPT_optionD_is_deductive_l423_42375

-- Conditions related to the reasoning options
inductive ReasoningProcess where
  | optionA : ReasoningProcess
  | optionB : ReasoningProcess
  | optionC : ReasoningProcess
  | optionD : ReasoningProcess

-- Definitions matching the equivalent Lean problem
def isDeductiveReasoning (rp : ReasoningProcess) : Prop :=
  match rp with
  | ReasoningProcess.optionA => False
  | ReasoningProcess.optionB => False
  | ReasoningProcess.optionC => False
  | ReasoningProcess.optionD => True

-- The proposition we need to prove
theorem optionD_is_deductive :
  isDeductiveReasoning ReasoningProcess.optionD = True := by
  sorry

end NUMINAMATH_GPT_optionD_is_deductive_l423_42375


namespace NUMINAMATH_GPT_evaluate_expression_l423_42320

theorem evaluate_expression :
  12 - 5 * 3^2 + 8 / 2 - 7 + 4^2 = -20 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l423_42320


namespace NUMINAMATH_GPT_first_folder_number_l423_42310

theorem first_folder_number (stickers : ℕ) (folders : ℕ) : stickers = 999 ∧ folders = 369 → 100 = 100 :=
by sorry

end NUMINAMATH_GPT_first_folder_number_l423_42310


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l423_42324

theorem arithmetic_sequence_common_difference  (a_n : ℕ → ℝ)
  (h1 : a_n 1 + a_n 6 = 12)
  (h2 : a_n 4 = 7) :
  ∃ d : ℝ, ∀ n : ℕ, a_n n = a_n 1 + (n - 1) * d ∧ d = 2 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l423_42324


namespace NUMINAMATH_GPT_three_digit_subtraction_l423_42301

theorem three_digit_subtraction (c d : ℕ) (H1 : 0 ≤ c ∧ c ≤ 9) (H2 : 0 ≤ d ∧ d ≤ 9) :
  (745 - (300 + c * 10 + 4) = (400 + d * 10 + 1)) ∧ ((4 + 1) - d % 11 = 0) → 
  c + d = 14 := 
sorry

end NUMINAMATH_GPT_three_digit_subtraction_l423_42301


namespace NUMINAMATH_GPT_cost_of_3000_pencils_l423_42358

-- Define the cost per box and the number of pencils per box
def cost_per_box : ℝ := 36
def pencils_per_box : ℕ := 120

-- Define the number of pencils to buy
def pencils_to_buy : ℕ := 3000

-- Define the total cost to prove
def total_cost_to_prove : ℝ := 900

-- The theorem to prove
theorem cost_of_3000_pencils : 
  (cost_per_box / pencils_per_box) * pencils_to_buy = total_cost_to_prove :=
by
  sorry

end NUMINAMATH_GPT_cost_of_3000_pencils_l423_42358


namespace NUMINAMATH_GPT_solution_set_quadratic_inequality_l423_42384

theorem solution_set_quadratic_inequality :
  {x : ℝ | (x^2 - 3*x + 2) < 0} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_solution_set_quadratic_inequality_l423_42384


namespace NUMINAMATH_GPT_trigonometric_ineq_l423_42383

theorem trigonometric_ineq (h₁ : (Real.pi / 4) < 1.5) (h₂ : 1.5 < (Real.pi / 2)) : 
  Real.cos 1.5 < Real.sin 1.5 ∧ Real.sin 1.5 < Real.tan 1.5 := 
sorry

end NUMINAMATH_GPT_trigonometric_ineq_l423_42383


namespace NUMINAMATH_GPT_min_value_sin_sq_l423_42341

theorem min_value_sin_sq (A B : ℝ) (h : A + B = π / 2) :
  4 / (Real.sin A)^2 + 9 / (Real.sin B)^2 ≥ 25 :=
sorry

end NUMINAMATH_GPT_min_value_sin_sq_l423_42341


namespace NUMINAMATH_GPT_combined_value_of_cookies_l423_42385

theorem combined_value_of_cookies
  (total_boxes_sold : ℝ)
  (plain_boxes_sold : ℝ)
  (price_chocolate_chip : ℝ)
  (price_plain : ℝ)
  (h1 : total_boxes_sold = 1585)
  (h2 : plain_boxes_sold = 793.375)
  (h3 : price_chocolate_chip = 1.25)
  (h4 : price_plain = 0.75) :
  (plain_boxes_sold * price_plain) + ((total_boxes_sold - plain_boxes_sold) * price_chocolate_chip) = 1584.5625 :=
by
  sorry

end NUMINAMATH_GPT_combined_value_of_cookies_l423_42385


namespace NUMINAMATH_GPT_rental_lower_amount_eq_50_l423_42362

theorem rental_lower_amount_eq_50 (L : ℝ) (total_rent : ℝ) (reduction : ℝ) (rooms_changed : ℕ) (diff_per_room : ℝ)
  (h1 : total_rent = 400)
  (h2 : reduction = 0.25 * total_rent)
  (h3 : rooms_changed = 10)
  (h4 : diff_per_room = reduction / ↑rooms_changed)
  (h5 : 60 - L = diff_per_room) :
  L = 50 :=
  sorry

end NUMINAMATH_GPT_rental_lower_amount_eq_50_l423_42362


namespace NUMINAMATH_GPT_quadratic_inequality_real_solutions_l423_42393

theorem quadratic_inequality_real_solutions (c : ℝ) (h_pos : 0 < c) : 
  (∃ x : ℝ, x^2 - 10 * x + c < 0) ↔ c < 25 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_real_solutions_l423_42393


namespace NUMINAMATH_GPT_evaluate_expression_l423_42306

theorem evaluate_expression : 
  -3 * 5 - (-4 * -2) + (-15 * -3) / 3 = -8 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l423_42306


namespace NUMINAMATH_GPT_inequality_proof_l423_42342

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l423_42342


namespace NUMINAMATH_GPT_ratio_of_pens_to_pencils_l423_42372

-- Define the conditions
def total_items : ℕ := 13
def pencils : ℕ := 4
def eraser : ℕ := 1
def pens : ℕ := total_items - pencils - eraser

-- Prove the ratio of pens to pencils is 2:1
theorem ratio_of_pens_to_pencils : pens = 2 * pencils :=
by
  -- indicate that the proof is omitted
  sorry

end NUMINAMATH_GPT_ratio_of_pens_to_pencils_l423_42372


namespace NUMINAMATH_GPT_area_of_inscribed_triangle_l423_42345

-- Define the square with a given diagonal
def diagonal (d : ℝ) : Prop := d = 16
def side_length_of_square (s : ℝ) : Prop := s = 8 * Real.sqrt 2
def side_length_of_equilateral_triangle (a : ℝ) : Prop := a = 8 * Real.sqrt 2

-- Define the area of the equilateral triangle
def area_of_equilateral_triangle (area : ℝ) : Prop :=
  area = 32 * Real.sqrt 3

-- The theorem: Given the above conditions, prove the area of the equilateral triangle
theorem area_of_inscribed_triangle (d s a area : ℝ) 
  (h1 : diagonal d) 
  (h2 : side_length_of_square s) 
  (h3 : side_length_of_equilateral_triangle a) 
  (h4 : s = a) : 
  area_of_equilateral_triangle area :=
sorry

end NUMINAMATH_GPT_area_of_inscribed_triangle_l423_42345


namespace NUMINAMATH_GPT_cooking_time_l423_42317

theorem cooking_time
  (total_potatoes : ℕ) (cooked_potatoes : ℕ) (remaining_time : ℕ) (remaining_potatoes : ℕ)
  (h_total : total_potatoes = 15)
  (h_cooked : cooked_potatoes = 8)
  (h_remaining_time : remaining_time = 63)
  (h_remaining_potatoes : remaining_potatoes = total_potatoes - cooked_potatoes) :
  remaining_time / remaining_potatoes = 9 :=
by
  sorry

end NUMINAMATH_GPT_cooking_time_l423_42317


namespace NUMINAMATH_GPT_selection_methods_l423_42350

theorem selection_methods (students : ℕ) (boys : ℕ) (girls : ℕ) (selected : ℕ) (h1 : students = 8) (h2 : boys = 6) (h3 : girls = 2) (h4 : selected = 4) : 
  ∃ methods, methods = 40 :=
by
  have h5 : students = boys + girls := by linarith
  sorry

end NUMINAMATH_GPT_selection_methods_l423_42350


namespace NUMINAMATH_GPT_found_bottle_caps_is_correct_l423_42369

def initial_bottle_caps : ℕ := 6
def total_bottle_caps : ℕ := 28

theorem found_bottle_caps_is_correct : total_bottle_caps - initial_bottle_caps = 22 := by
  sorry

end NUMINAMATH_GPT_found_bottle_caps_is_correct_l423_42369


namespace NUMINAMATH_GPT_proof_problem_l423_42382

variable (a b c d x : ℤ)

-- Conditions
axiom condition1 : a - b = c + d + x
axiom condition2 : a + b = c - d - 3
axiom condition3 : a - c = 3
axiom answer_eq : x = 9

-- Proof statement
theorem proof_problem : (a - b) = (c + d + 9) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l423_42382


namespace NUMINAMATH_GPT_estimate_larger_than_difference_l423_42353

variable {x y : ℝ}

theorem estimate_larger_than_difference (h1 : x > y) (h2 : y > 0) :
    ⌈x⌉ - ⌊y⌋ > x - y := by
  sorry

end NUMINAMATH_GPT_estimate_larger_than_difference_l423_42353


namespace NUMINAMATH_GPT_mixture_replacement_l423_42312

theorem mixture_replacement (A B x : ℕ) (hA : A = 32) (h_ratio1 : A / B = 4) (h_ratio2 : A / (B + x) = 2 / 3) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_mixture_replacement_l423_42312


namespace NUMINAMATH_GPT_smallest_pos_int_ends_in_6_divisible_by_11_l423_42389

theorem smallest_pos_int_ends_in_6_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 6 ∧ 11 ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m % 10 = 6 ∧ 11 ∣ m → n ≤ m := by
  sorry

end NUMINAMATH_GPT_smallest_pos_int_ends_in_6_divisible_by_11_l423_42389


namespace NUMINAMATH_GPT_multiply_polynomials_l423_42339

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end NUMINAMATH_GPT_multiply_polynomials_l423_42339


namespace NUMINAMATH_GPT_factorize_one_factorize_two_l423_42381

variable (m x y : ℝ)

-- Problem statement for Question 1
theorem factorize_one (m : ℝ) : 
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := 
sorry

-- Problem statement for Question 2
theorem factorize_two (x y : ℝ) : 
  (x + y)^2 - 4 * (x + y) + 4 = (x + y - 2)^2 := 
sorry

end NUMINAMATH_GPT_factorize_one_factorize_two_l423_42381


namespace NUMINAMATH_GPT_simplify_tangent_expression_l423_42367

theorem simplify_tangent_expression :
  (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_simplify_tangent_expression_l423_42367


namespace NUMINAMATH_GPT_suff_not_nec_cond_l423_42309

theorem suff_not_nec_cond (a : ℝ) : (a > 6 → a^2 > 36) ∧ (a^2 > 36 → (a > 6 ∨ a < -6)) := by
  sorry

end NUMINAMATH_GPT_suff_not_nec_cond_l423_42309


namespace NUMINAMATH_GPT_work_completion_l423_42330

noncomputable def efficiency (p q: ℕ) := q = 3 * p / 5

theorem work_completion (p q : ℕ) (h1 : efficiency p q) (h2: p * 24 = 100) :
  2400 / (p + q) = 15 :=
by 
  sorry

end NUMINAMATH_GPT_work_completion_l423_42330


namespace NUMINAMATH_GPT_max_value_of_expression_l423_42373

theorem max_value_of_expression (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) : 
  (∃ x : ℝ, x = 3 → 
    ∀ A : ℝ, A = (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4/3)) → 
      A ≤ x) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l423_42373


namespace NUMINAMATH_GPT_fraction_power_multiplication_l423_42360

theorem fraction_power_multiplication :
  ( (1 / 3) ^ 4 * (1 / 5) = 1 / 405 ) :=
by
  sorry

end NUMINAMATH_GPT_fraction_power_multiplication_l423_42360


namespace NUMINAMATH_GPT_no_daily_coverage_l423_42334

theorem no_daily_coverage (ranks : Nat → Nat)
  (h_ranks_ordered : ∀ i, ranks (i+1) ≥ 3 * ranks i)
  (h_cycle : ∀ i, ∃ N : Nat, ranks i = N ∧ ∃ k : Nat, k = N ∧ ∀ m, m % (2 * N) < N → (¬ ∃ j, ranks j ≤ N))
  : ¬ (∀ d : Nat, ∃ j : Nat, (∃ k : Nat, d % (2 * (ranks j)) < ranks j))
  := sorry

end NUMINAMATH_GPT_no_daily_coverage_l423_42334


namespace NUMINAMATH_GPT_sum_of_two_primes_unique_l423_42314

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_two_primes_unique_l423_42314


namespace NUMINAMATH_GPT_equal_binomial_terms_l423_42327

theorem equal_binomial_terms (p q : ℝ) (h1 : 0 < p) (h2 : 0 < q) (h3 : p + q = 1)
    (h4 : 55 * p^9 * q^2 = 165 * p^8 * q^3) : p = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_equal_binomial_terms_l423_42327


namespace NUMINAMATH_GPT_no_integers_p_q_l423_42349

theorem no_integers_p_q :
  ¬ ∃ p q : ℤ, ∀ x : ℤ, 3 ∣ (x^2 + p * x + q) :=
by
  sorry

end NUMINAMATH_GPT_no_integers_p_q_l423_42349


namespace NUMINAMATH_GPT_largest_sum_fraction_l423_42337

open Rat

theorem largest_sum_fraction :
  let a := (2:ℚ) / 5
  let c1 := (1:ℚ) / 6
  let c2 := (1:ℚ) / 3
  let c3 := (1:ℚ) / 7
  let c4 := (1:ℚ) / 8
  let c5 := (1:ℚ) / 9
  max (a + c1) (max (a + c2) (max (a + c3) (max (a + c4) (a + c5)))) = a + c2
  ∧ a + c2 = (11:ℚ) / 15 := by
  sorry

end NUMINAMATH_GPT_largest_sum_fraction_l423_42337


namespace NUMINAMATH_GPT_max_students_l423_42366

theorem max_students (A B C : ℕ) (A_left B_left C_left : ℕ)
  (hA : A = 38) (hB : B = 78) (hC : C = 128)
  (hA_left : A_left = 2) (hB_left : B_left = 6) (hC_left : C_left = 20) :
  gcd (A - A_left) (gcd (B - B_left) (C - C_left)) = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_students_l423_42366


namespace NUMINAMATH_GPT_river_current_speed_l423_42357

noncomputable section

variables {d r w : ℝ}

def time_equation_normal_speed (d r w : ℝ) : Prop :=
  (d / (r + w)) + 4 = (d / (r - w))

def time_equation_tripled_speed (d r w : ℝ) : Prop :=
  (d / (3 * r + w)) + 2 = (d / (3 * r - w))

theorem river_current_speed (d r : ℝ) (h1 : time_equation_normal_speed d r w) (h2 : time_equation_tripled_speed d r w) : w = 2 :=
sorry

end NUMINAMATH_GPT_river_current_speed_l423_42357


namespace NUMINAMATH_GPT_range_of_a_l423_42390

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x^2

theorem range_of_a {a : ℝ} : 
  (∀ x, Real.exp x - 2 * a * x ≥ 0) ↔ 0 ≤ a ∧ a ≤ Real.exp 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l423_42390


namespace NUMINAMATH_GPT_rectangle_area_l423_42399

theorem rectangle_area (length_of_rectangle radius_of_circle side_of_square : ℝ)
  (h1 : length_of_rectangle = (2 / 5) * radius_of_circle)
  (h2 : radius_of_circle = side_of_square)
  (h3 : side_of_square * side_of_square = 1225)
  (breadth_of_rectangle : ℝ)
  (h4 : breadth_of_rectangle = 10) : 
  length_of_rectangle * breadth_of_rectangle = 140 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_l423_42399


namespace NUMINAMATH_GPT_ratio_correct_l423_42380

def my_age : ℕ := 35
def son_age_next_year : ℕ := 8
def son_age_now : ℕ := son_age_next_year - 1
def ratio_of_ages : ℕ := my_age / son_age_now

theorem ratio_correct : ratio_of_ages = 5 :=
by
  -- Add proof here
  sorry

end NUMINAMATH_GPT_ratio_correct_l423_42380


namespace NUMINAMATH_GPT_min_value_frac_sum_l423_42370

open Real

theorem min_value_frac_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 2 * a + b = 1) :
  1 / a + 2 / b = 8 :=
sorry

end NUMINAMATH_GPT_min_value_frac_sum_l423_42370


namespace NUMINAMATH_GPT_problem1_problem2_l423_42313

-- Problem 1: Lean 4 Statement
theorem problem1 (n : ℕ) (hn : n > 0) : 20 ∣ (4 * 6^n + 5^(n + 1) - 9) :=
sorry

-- Problem 2: Lean 4 Statement
theorem problem2 : (3^100 % 7) = 4 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l423_42313


namespace NUMINAMATH_GPT_emily_and_eli_probability_l423_42371

noncomputable def probability_same_number : ℚ :=
  let count_multiples (n k : ℕ) := (k - 1) / n
  let emily_count := count_multiples 20 250
  let eli_count := count_multiples 30 250
  let common_lcm := Nat.lcm 20 30
  let common_count := count_multiples common_lcm 250
  common_count / (emily_count * eli_count : ℚ)

theorem emily_and_eli_probability :
  let probability := probability_same_number
  probability = 1 / 24 :=
by
  sorry

end NUMINAMATH_GPT_emily_and_eli_probability_l423_42371


namespace NUMINAMATH_GPT_solve_equation_l423_42361

noncomputable def a := 3 + Real.sqrt 8
noncomputable def b := 3 - Real.sqrt 8

theorem solve_equation (x : ℝ) :
  (Real.sqrt (a^x) + Real.sqrt (b^x) = 6) ↔ (x = 2 ∨ x = -2) := 
  by
  sorry

end NUMINAMATH_GPT_solve_equation_l423_42361


namespace NUMINAMATH_GPT_maximize_revenue_l423_42326

theorem maximize_revenue (p : ℝ) (h : p ≤ 30) : 
  (∀ q : ℝ, q ≤ 30 → (150 * 18.75 - 4 * (18.75:ℝ)^2) ≥ (150 * q - 4 * q^2)) ↔ p = 18.75 := 
sorry

end NUMINAMATH_GPT_maximize_revenue_l423_42326


namespace NUMINAMATH_GPT_pq_plus_four_mul_l423_42328

open Real

theorem pq_plus_four_mul {p q : ℝ} (h1 : (x - 4) * (3 * x + 11) = x ^ 2 - 19 * x + 72) 
  (hpq1 : 2 * p ^ 2 + 18 * p - 116 = 0) (hpq2 : 2 * q ^ 2 + 18 * q - 116 = 0) (hpq_ne : p ≠ q) : 
  (p + 4) * (q + 4) = -78 := 
sorry

end NUMINAMATH_GPT_pq_plus_four_mul_l423_42328


namespace NUMINAMATH_GPT_polygon_properties_l423_42311

theorem polygon_properties
    (each_exterior_angle : ℝ)
    (h1 : each_exterior_angle = 24) :
    ∃ n : ℕ, n = 15 ∧ (180 * (n - 2) = 2340) :=
  by
    sorry

end NUMINAMATH_GPT_polygon_properties_l423_42311


namespace NUMINAMATH_GPT_polar_area_enclosed_l423_42397

theorem polar_area_enclosed :
  let θ1 := Real.pi / 3
  let θ2 := 2 * Real.pi / 3
  let ρ := 4
  let area := (1/2) * (θ2 - θ1) * ρ^2
  area = 8 * Real.pi / 3 :=
by
  let θ1 := Real.pi / 3
  let θ2 := 2 * Real.pi / 3
  let ρ := 4
  let area := (1/2) * (θ2 - θ1) * ρ^2
  show area = 8 * Real.pi / 3
  sorry

end NUMINAMATH_GPT_polar_area_enclosed_l423_42397


namespace NUMINAMATH_GPT_more_silverfish_than_goldfish_l423_42343

variable (n G S R : ℕ)

-- Condition 1: If the cat eats all the goldfish, the number of remaining fish is \(\frac{2}{3}\)n - 1
def condition1 := n - G = (2 * n) / 3 - 1

-- Condition 2: If the cat eats all the redfish, the number of remaining fish is \(\frac{2}{3}\)n + 4
def condition2 := n - R = (2 * n) / 3 + 4

-- The goal: Silverfish are more numerous than goldfish by 2
theorem more_silverfish_than_goldfish (h1 : condition1 n G) (h2 : condition2 n R) :
  S = (n / 3) + 3 → G = (n / 3) + 1 → S - G = 2 :=
by
  sorry

end NUMINAMATH_GPT_more_silverfish_than_goldfish_l423_42343


namespace NUMINAMATH_GPT_third_box_number_l423_42391

def N : ℕ := 301

theorem third_box_number (N : ℕ) (h1 : N % 3 = 1) (h2 : N % 4 = 1) (h3 : N % 7 = 0) :
  ∃ x : ℕ, x > 4 ∧ x ≠ 7 ∧ N % x = 1 ∧ (∀ y > 4, y ≠ 7 → y < x → N % y ≠ 1) ∧ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_third_box_number_l423_42391


namespace NUMINAMATH_GPT_integer_divisibility_l423_42392

open Nat

theorem integer_divisibility (n : ℕ) (h1 : ∃ m : ℕ, 2^n - 2 = n * m) : ∃ k : ℕ, 2^((2^n) - 1) - 2 = (2^n - 1) * k := by
  sorry

end NUMINAMATH_GPT_integer_divisibility_l423_42392


namespace NUMINAMATH_GPT_total_number_of_shells_l423_42307

variable (David Mia Ava Alice : ℕ)
variable (hd : David = 15)
variable (hm : Mia = 4 * David)
variable (ha : Ava = Mia + 20)
variable (hAlice : Alice = Ava / 2)

theorem total_number_of_shells :
  David + Mia + Ava + Alice = 195 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_shells_l423_42307


namespace NUMINAMATH_GPT_f_values_sum_l423_42368

noncomputable def f : ℝ → ℝ := sorry

-- defining the properties
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- given conditions
axiom f_odd : is_odd f
axiom f_periodic : is_periodic f 2

-- statement to prove
theorem f_values_sum : f 1 + f 2 + f 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_values_sum_l423_42368


namespace NUMINAMATH_GPT_total_ages_l423_42331

variable (Bill_age Caroline_age : ℕ)
variable (h1 : Bill_age = 2 * Caroline_age - 1) (h2 : Bill_age = 17)

theorem total_ages : Bill_age + Caroline_age = 26 :=
by
  sorry

end NUMINAMATH_GPT_total_ages_l423_42331


namespace NUMINAMATH_GPT_cos_beta_half_l423_42303

theorem cos_beta_half (α β : ℝ) (hα_ac : 0 < α ∧ α < π / 2) (hβ_ac : 0 < β ∧ β < π / 2) 
  (h1 : Real.tan α = 4 * Real.sqrt 3) (h2 : Real.cos (α + β) = -11 / 14) : 
  Real.cos β = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_beta_half_l423_42303


namespace NUMINAMATH_GPT_geometric_sequence_sum_inverse_equals_l423_42338

variable (a : ℕ → ℝ)
variable (n : ℕ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃(r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum_inverse_equals (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_sum : a 5 + a 6 + a 7 + a 8 = 15 / 8)
  (h_prod : a 6 * a 7 = -9 / 8) :
  (1 / a 5) + (1 / a 6) + (1 / a 7) + (1 / a 8) = -5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_inverse_equals_l423_42338


namespace NUMINAMATH_GPT_ab_is_square_l423_42363

theorem ab_is_square (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) 
  (h_main : a + b = b * (a - c)) (h_prime : ∃ p : ℕ, Prime p ∧ c + 1 = p^2) :
  ∃ k : ℕ, a + b = k^2 :=
by
  sorry

end NUMINAMATH_GPT_ab_is_square_l423_42363


namespace NUMINAMATH_GPT_proof_problem_l423_42305

def polar_curve_C (ρ : ℝ) : Prop := ρ = 5

def point_P (x y : ℝ) : Prop := x = -3 ∧ y = -3 / 2

def line_l_through_P (x y : ℝ) (k : ℝ) : Prop := y + 3 / 2 = k * (x + 3)

def distance_AB (A B : ℝ × ℝ) : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64

theorem proof_problem
  (ρ : ℝ) (x y : ℝ) (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : polar_curve_C ρ)
  (h2 : point_P (-3) (-3 / 2))
  (h3 : ∃ k, line_l_through_P x y k)
  (h4 : distance_AB A B) :
  ∃ (x y : ℝ), (x^2 + y^2 = 25) ∧ ((x = -3) ∨ (3 * x + 4 * y + 15 = 0)) := 
sorry

end NUMINAMATH_GPT_proof_problem_l423_42305


namespace NUMINAMATH_GPT_river_width_l423_42318

noncomputable def width_of_river (d: ℝ) (f: ℝ) (v: ℝ) : ℝ :=
  v / (d * (f * 1000 / 60))

theorem river_width : width_of_river 2 2 3000 = 45 := by
  sorry

end NUMINAMATH_GPT_river_width_l423_42318


namespace NUMINAMATH_GPT_remainder_avg_is_correct_l423_42332

-- Definitions based on the conditions
variables (total_avg : ℝ) (first_part_avg : ℝ) (second_part_avg : ℝ) (first_part_percent : ℝ) (second_part_percent : ℝ)

-- The conditions stated mathematically
def overall_avg_contribution 
  (remainder_avg : ℝ) : Prop :=
  first_part_percent * first_part_avg + 
  second_part_percent * second_part_avg + 
  (1 - first_part_percent - second_part_percent) * remainder_avg =  total_avg
  
-- The question
theorem remainder_avg_is_correct : overall_avg_contribution 75 80 65 0.25 0.50 90 := sorry

end NUMINAMATH_GPT_remainder_avg_is_correct_l423_42332


namespace NUMINAMATH_GPT_total_tagged_numbers_l423_42321

theorem total_tagged_numbers:
  let W := 200
  let X := W / 2
  let Y := X + W
  let Z := 400
  W + X + Y + Z = 1000 := by 
    sorry

end NUMINAMATH_GPT_total_tagged_numbers_l423_42321


namespace NUMINAMATH_GPT_average_white_paper_per_ton_trees_saved_per_ton_l423_42329

-- Define the given conditions
def waste_paper_tons : ℕ := 5
def produced_white_paper_tons : ℕ := 4
def saved_trees : ℕ := 40

-- State the theorems that need to be proved
theorem average_white_paper_per_ton :
  (produced_white_paper_tons : ℚ) / waste_paper_tons = 0.8 := 
sorry

theorem trees_saved_per_ton :
  (saved_trees : ℚ) / waste_paper_tons = 8 := 
sorry

end NUMINAMATH_GPT_average_white_paper_per_ton_trees_saved_per_ton_l423_42329


namespace NUMINAMATH_GPT_inequality_correct_l423_42376

variable (a b : ℝ)

theorem inequality_correct (h : a < b) : 2 - a > 2 - b :=
by
  sorry

end NUMINAMATH_GPT_inequality_correct_l423_42376


namespace NUMINAMATH_GPT_find_x_l423_42316

def operation (a b : Int) : Int := 2 * a + b

theorem find_x :
  ∃ x : Int, operation 3 (operation 4 x) = -1 :=
  sorry

end NUMINAMATH_GPT_find_x_l423_42316


namespace NUMINAMATH_GPT_initial_red_marbles_l423_42377

theorem initial_red_marbles (R : ℕ) (blue_marbles_initial : ℕ) (red_marbles_removed : ℕ) :
  blue_marbles_initial = 30 →
  red_marbles_removed = 3 →
  (R - red_marbles_removed) + (blue_marbles_initial - 4 * red_marbles_removed) = 35 →
  R = 20 :=
by
  intros h_blue h_red h_total
  sorry

end NUMINAMATH_GPT_initial_red_marbles_l423_42377


namespace NUMINAMATH_GPT_function_additive_of_tangential_property_l423_42354

open Set

variable {f : ℝ → ℝ}

def is_tangential_quadrilateral_sides (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ (a + c = b + d)

theorem function_additive_of_tangential_property
  (h : ∀ (a b c d : ℝ), is_tangential_quadrilateral_sides a b c d → f (a + b + c + d) = f a + f b + f c + f d) :
  ∀ (x y : ℝ), 0 < x → 0 < y → f (x + y) = f x + f y :=
by
  sorry

end NUMINAMATH_GPT_function_additive_of_tangential_property_l423_42354


namespace NUMINAMATH_GPT_suitable_storage_temp_l423_42347

theorem suitable_storage_temp : -5 ≤ -1 ∧ -1 ≤ 1 := by {
  sorry
}

end NUMINAMATH_GPT_suitable_storage_temp_l423_42347


namespace NUMINAMATH_GPT_trigonometric_proof_l423_42388

noncomputable def proof_problem (α β : Real) : Prop :=
  (β = 90 - α) → (Real.sin β = Real.cos α) → 
  (Real.sqrt 3 * Real.sin α + Real.sin β) / Real.sqrt (2 - 2 * Real.cos 100) = 1

-- Statement that incorporates all conditions and concludes the proof problem.
theorem trigonometric_proof :
  proof_problem 20 70 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_trigonometric_proof_l423_42388


namespace NUMINAMATH_GPT_find_divisor_l423_42322

theorem find_divisor (n x : ℕ) (h1 : n = 3) (h2 : (n / x : ℝ) * 12 = 9): x = 4 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l423_42322


namespace NUMINAMATH_GPT_rectangle_area_192_l423_42387

variable (b l : ℝ) (A : ℝ)

-- Conditions
def length_is_thrice_breadth : Prop :=
  l = 3 * b

def perimeter_is_64 : Prop :=
  2 * (l + b) = 64

-- Area calculation
def area_of_rectangle : ℝ :=
  l * b

theorem rectangle_area_192 (h1 : length_is_thrice_breadth b l) (h2 : perimeter_is_64 b l) :
  area_of_rectangle l b = 192 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_192_l423_42387


namespace NUMINAMATH_GPT_percent_profit_l423_42374

theorem percent_profit (C S : ℝ) (h : 60 * C = 50 * S):
  (((S - C) / C) * 100) = 20 :=
by 
  sorry

end NUMINAMATH_GPT_percent_profit_l423_42374


namespace NUMINAMATH_GPT_least_number_modular_l423_42304

theorem least_number_modular 
  (n : ℕ)
  (h1 : n % 34 = 4)
  (h2 : n % 48 = 6)
  (h3 : n % 5 = 2) : n = 4082 :=
by
  sorry

end NUMINAMATH_GPT_least_number_modular_l423_42304


namespace NUMINAMATH_GPT_max_value_y_l423_42335

theorem max_value_y (x y : ℕ) (h₁ : 9 * (x + y) > 17 * x) (h₂ : 15 * x < 8 * (x + y)) :
  y ≤ 112 :=
sorry

end NUMINAMATH_GPT_max_value_y_l423_42335


namespace NUMINAMATH_GPT_elasticity_ratio_is_correct_l423_42346

-- Definitions of the given elasticities
def e_OGBR_QN : ℝ := 1.27
def e_OGBR_PN : ℝ := 0.76

-- Theorem stating the ratio of elasticities equals 1.7
theorem elasticity_ratio_is_correct : (e_OGBR_QN / e_OGBR_PN) = 1.7 := sorry

end NUMINAMATH_GPT_elasticity_ratio_is_correct_l423_42346


namespace NUMINAMATH_GPT_can_capacity_is_14_l423_42398

noncomputable def capacity_of_can 
    (initial_milk: ℝ) (initial_water: ℝ) 
    (added_milk: ℝ) (ratio_initial: ℝ) (ratio_final: ℝ): ℝ :=
  initial_milk + initial_water + added_milk

theorem can_capacity_is_14
    (M W: ℝ) 
    (ratio_initial : M / W = 1 / 5) 
    (added_milk : ℝ := 2) 
    (ratio_final:  (M + 2) / W = 2.00001 / 5.00001): 
    capacity_of_can M W added_milk (1 / 5) (2.00001 / 5.00001) = 14 := 
  by
    sorry

end NUMINAMATH_GPT_can_capacity_is_14_l423_42398


namespace NUMINAMATH_GPT_total_phd_time_l423_42365

-- Definitions for the conditions
def acclimation_period : ℕ := 1
def basics_period : ℕ := 2
def research_period := basics_period + (3 * basics_period / 4)
def dissertation_period := acclimation_period / 2

-- Main statement to prove
theorem total_phd_time : acclimation_period + basics_period + research_period + dissertation_period = 7 := by
  -- Here should be the proof (skipped with sorry)
  sorry

end NUMINAMATH_GPT_total_phd_time_l423_42365


namespace NUMINAMATH_GPT_range_of_a_no_real_roots_l423_42379

theorem range_of_a_no_real_roots (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 + ax + 1 = 0)) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_no_real_roots_l423_42379


namespace NUMINAMATH_GPT_original_pencils_example_l423_42394

-- Statement of the problem conditions
def original_pencils (total_pencils : ℕ) (added_pencils : ℕ) : ℕ :=
  total_pencils - added_pencils

-- Theorem we need to prove
theorem original_pencils_example : original_pencils 5 3 = 2 := 
by
  -- Proof
  sorry

end NUMINAMATH_GPT_original_pencils_example_l423_42394


namespace NUMINAMATH_GPT_solve_for_x_l423_42344

theorem solve_for_x (x : ℝ) : 
  (x - 35) / 3 = (3 * x + 10) / 8 → x = -310 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l423_42344


namespace NUMINAMATH_GPT_relationship_f_2011_2014_l423_42356

noncomputable def quadratic_func : Type := ℝ → ℝ

variable (f : quadratic_func)

-- The function is symmetric about x = 2013
axiom symmetry (x : ℝ) : f (2013 + x) = f (2013 - x)

-- The function opens upward (convexity)
axiom opens_upward (a b : ℝ) : f ((a + b) / 2) ≤ (f a + f b) / 2

theorem relationship_f_2011_2014 :
  f 2011 > f 2014 := 
sorry

end NUMINAMATH_GPT_relationship_f_2011_2014_l423_42356


namespace NUMINAMATH_GPT_base_3_is_most_economical_l423_42352

theorem base_3_is_most_economical (m d : ℕ) (h : d ≥ 1) (h_m_div_d : m % d = 0) :
  3^(m / 3) ≥ d^(m / d) :=
sorry

end NUMINAMATH_GPT_base_3_is_most_economical_l423_42352


namespace NUMINAMATH_GPT_find_U_l423_42323

-- Declare the variables and conditions
def digits : Set ℤ := {1, 2, 3, 4, 5, 6}

theorem find_U (P Q R S T U : ℤ) :
  -- Condition: Digits are distinct and each is in {1, 2, 3, 4, 5, 6}
  (P ∈ digits) ∧ (Q ∈ digits) ∧ (R ∈ digits) ∧ (S ∈ digits) ∧ (T ∈ digits) ∧ (U ∈ digits) ∧
  (P ≠ Q) ∧ (P ≠ R) ∧ (P ≠ S) ∧ (P ≠ T) ∧ (P ≠ U) ∧
  (Q ≠ R) ∧ (Q ≠ S) ∧ (Q ≠ T) ∧ (Q ≠ U) ∧
  (R ≠ S) ∧ (R ≠ T) ∧ (R ≠ U) ∧ (S ≠ T) ∧ (S ≠ U) ∧ (T ≠ U) ∧
  -- Condition: The three-digit number PQR is divisible by 9
  (100 * P + 10 * Q + R) % 9 = 0 ∧
  -- Condition: The three-digit number QRS is divisible by 4
  (10 * Q + R) % 4 = 0 ∧
  -- Condition: The three-digit number RST is divisible by 3
  (10 * R + S) % 3 = 0 ∧
  -- Condition: The sum of the digits is divisible by 5
  (P + Q + R + S + T + U) % 5 = 0
  -- Conclusion: U = 4
  → U = 4 :=
by sorry

end NUMINAMATH_GPT_find_U_l423_42323


namespace NUMINAMATH_GPT_number_of_terriers_groomed_l423_42395

-- Define the initial constants and the conditions from the problem statement
def time_to_groom_poodle := 30
def time_to_groom_terrier := 15
def number_of_poodles := 3
def total_grooming_time := 210

-- Define the problem to prove that the number of terriers groomed is 8
theorem number_of_terriers_groomed (groom_time_poodle groom_time_terrier num_poodles total_time : ℕ) : 
  groom_time_poodle = time_to_groom_poodle → 
  groom_time_terrier = time_to_groom_terrier →
  num_poodles = number_of_poodles →
  total_time = total_grooming_time →
  ∃ n : ℕ, n * groom_time_terrier + num_poodles * groom_time_poodle = total_time ∧ n = 8 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_number_of_terriers_groomed_l423_42395


namespace NUMINAMATH_GPT_partial_fraction_sum_eq_zero_l423_42302

theorem partial_fraction_sum_eq_zero (A B C D E : ℂ) :
  (∀ x : ℂ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 4 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x - 4)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x - 4)) →
  A + B + C + D + E = 0 :=
by
  sorry

end NUMINAMATH_GPT_partial_fraction_sum_eq_zero_l423_42302


namespace NUMINAMATH_GPT_percentage_of_other_investment_l423_42364

theorem percentage_of_other_investment (investment total_interest interest_5 interest_other percentage_other : ℝ) 
  (h1 : investment = 18000)
  (h2 : interest_5 = 6000 * 0.05)
  (h3 : total_interest = 660)
  (h4 : percentage_other / 100 * (investment - 6000) = 360) : 
  percentage_other = 3 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_other_investment_l423_42364


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l423_42351

theorem arithmetic_seq_sum (a : ℕ → ℤ) (h_arith_seq : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q) (h_a5 : a 5 = 15) : a 2 + a 4 + a 6 + a 8 = 60 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l423_42351


namespace NUMINAMATH_GPT_annual_expenditure_l423_42333

theorem annual_expenditure (x y : ℝ) (h1 : y = 0.8 * x + 0.1) (h2 : x = 15) : y = 12.1 :=
by
  sorry

end NUMINAMATH_GPT_annual_expenditure_l423_42333


namespace NUMINAMATH_GPT_Ian_money_left_l423_42308

-- Definitions based on the conditions
def hours_worked : ℕ := 8
def rate_per_hour : ℕ := 18
def total_money_made : ℕ := hours_worked * rate_per_hour
def money_left : ℕ := total_money_made / 2

-- The statement to be proved 
theorem Ian_money_left : money_left = 72 :=
by
  sorry

end NUMINAMATH_GPT_Ian_money_left_l423_42308


namespace NUMINAMATH_GPT_largest_number_of_pangs_largest_number_of_pangs_possible_l423_42386

theorem largest_number_of_pangs (x y z : ℕ) 
  (hx : x ≥ 2) 
  (hy : y ≥ 3) 
  (hcost : 3 * x + 4 * y + 9 * z = 100) : 
  z ≤ 9 :=
by sorry

theorem largest_number_of_pangs_possible (x y z : ℕ) 
  (hx : x = 2) 
  (hy : y = 3) 
  (hcost : 3 * x + 4 * y + 9 * z = 100) : 
  z = 9 :=
by sorry

end NUMINAMATH_GPT_largest_number_of_pangs_largest_number_of_pangs_possible_l423_42386


namespace NUMINAMATH_GPT_relationship_of_squares_and_products_l423_42359

theorem relationship_of_squares_and_products (a b x : ℝ) (h1 : b < x) (h2 : x < a) (h3 : a < 0) : 
  x^2 > ax ∧ ax > b^2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_of_squares_and_products_l423_42359


namespace NUMINAMATH_GPT_total_pencils_children_l423_42325

theorem total_pencils_children :
  let c1 := 6
  let c2 := 9
  let c3 := 12
  let c4 := 15
  let c5 := 18
  c1 + c2 + c3 + c4 + c5 = 60 :=
by
  let c1 := 6
  let c2 := 9
  let c3 := 12
  let c4 := 15
  let c5 := 18
  show c1 + c2 + c3 + c4 + c5 = 60
  sorry

end NUMINAMATH_GPT_total_pencils_children_l423_42325


namespace NUMINAMATH_GPT_simultaneous_equations_solution_l423_42355

theorem simultaneous_equations_solution :
  ∃ x y : ℚ, (3 * x + 4 * y = 20) ∧ (9 * x - 8 * y = 36) ∧ (x = 76 / 15) ∧ (y = 18 / 15) :=
by
  sorry

end NUMINAMATH_GPT_simultaneous_equations_solution_l423_42355


namespace NUMINAMATH_GPT_smallest_natrural_number_cube_ends_888_l423_42336

theorem smallest_natrural_number_cube_ends_888 :
  ∃ n : ℕ, (n^3 % 1000 = 888) ∧ (∀ m : ℕ, (m^3 % 1000 = 888) → n ≤ m) := 
sorry

end NUMINAMATH_GPT_smallest_natrural_number_cube_ends_888_l423_42336


namespace NUMINAMATH_GPT_fraction_division_l423_42319

theorem fraction_division (a b c d : ℚ) (h1 : a = 3) (h2 : b = 8) (h3 : c = 5) (h4 : d = 12) :
  (a / b) / (c / d) = 9 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_division_l423_42319


namespace NUMINAMATH_GPT_two_point_form_eq_l423_42378

theorem two_point_form_eq (x y : ℝ) : 
  let A := (5, 6)
  let B := (-1, 2)
  (y - 6) / (2 - 6) = (x - 5) / (-1 - 5) := 
  sorry

end NUMINAMATH_GPT_two_point_form_eq_l423_42378
