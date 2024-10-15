import Mathlib

namespace NUMINAMATH_GPT_marks_in_biology_l2421_242107

theorem marks_in_biology (marks_english : ℕ) (marks_math : ℕ) (marks_physics : ℕ) (marks_chemistry : ℕ) (average_marks : ℕ) :
  marks_english = 73 → marks_math = 69 → marks_physics = 92 → marks_chemistry = 64 → average_marks = 76 →
  (380 - (marks_english + marks_math + marks_physics + marks_chemistry)) = 82 :=
by
  intros
  sorry

end NUMINAMATH_GPT_marks_in_biology_l2421_242107


namespace NUMINAMATH_GPT_value_of_y_l2421_242122

theorem value_of_y (x y : ℝ) (h₁ : 1.5 * x = 0.75 * y) (h₂ : x = 20) : y = 40 :=
sorry

end NUMINAMATH_GPT_value_of_y_l2421_242122


namespace NUMINAMATH_GPT_suff_not_nec_l2421_242129

variables (a b : ℝ)
def P := (a = 1) ∧ (b = 1)
def Q := (a + b = 2)

theorem suff_not_nec : P a b → Q a b ∧ ¬ (Q a b → P a b) :=
by
  sorry

end NUMINAMATH_GPT_suff_not_nec_l2421_242129


namespace NUMINAMATH_GPT_smallest_four_digit_solution_l2421_242182

theorem smallest_four_digit_solution :
  ∃ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧
  (3 * x ≡ 6 [MOD 12]) ∧
  (5 * x + 20 ≡ 25 [MOD 15]) ∧
  (3 * x - 2 ≡ 2 * x [MOD 35]) ∧
  x = 1274 :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_solution_l2421_242182


namespace NUMINAMATH_GPT_remainder_777_777_mod_13_l2421_242194

theorem remainder_777_777_mod_13 : (777 ^ 777) % 13 = 12 := 
by 
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_remainder_777_777_mod_13_l2421_242194


namespace NUMINAMATH_GPT_expression_value_l2421_242198

noncomputable def expr := (1.90 * (1 / (1 - (3: ℝ)^(1/4)))) + (1 / (1 + (3: ℝ)^(1/4))) + (2 / (1 + (3: ℝ)^(1/2)))

theorem expression_value : expr = -2 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l2421_242198


namespace NUMINAMATH_GPT_ways_to_divide_8_friends_l2421_242196

theorem ways_to_divide_8_friends : (4 : ℕ) ^ 8 = 65536 := by
  sorry

end NUMINAMATH_GPT_ways_to_divide_8_friends_l2421_242196


namespace NUMINAMATH_GPT_least_multiple_greater_than_500_l2421_242175

theorem least_multiple_greater_than_500 : ∃ n : ℕ, n > 500 ∧ n % 32 = 0 := by
  let n := 512
  have h1 : n > 500 := by 
    -- proof omitted, as we're not solving the problem here
    sorry
  have h2 : n % 32 = 0 := by 
    -- proof omitted
    sorry
  exact ⟨n, h1, h2⟩

end NUMINAMATH_GPT_least_multiple_greater_than_500_l2421_242175


namespace NUMINAMATH_GPT_max_distance_circle_to_point_A_l2421_242138

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 2

noncomputable def point_A : ℝ × ℝ := (-1, 3)

noncomputable def max_distance (d : ℝ) : Prop :=
  ∃ x y, circle_eq x y ∧ d = Real.sqrt ((2 + 1)^2 + (0 - 3)^2) + Real.sqrt 2 

theorem max_distance_circle_to_point_A : max_distance (4 * Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_max_distance_circle_to_point_A_l2421_242138


namespace NUMINAMATH_GPT_find_a1_l2421_242148

theorem find_a1 (f : ℝ → ℝ) (a : ℕ → ℝ) (h₀ : ∀ x, f x = (x - 1)^3 + x + 2)
(h₁ : ∀ n, a (n + 1) = a n + 1/2)
(h₂ : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 18) :
a 1 = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_l2421_242148


namespace NUMINAMATH_GPT_original_number_increased_l2421_242167

theorem original_number_increased (x : ℝ) (h : (1.10 * x) * 1.15 = 632.5) : x = 500 :=
sorry

end NUMINAMATH_GPT_original_number_increased_l2421_242167


namespace NUMINAMATH_GPT_kids_on_soccer_field_l2421_242157

theorem kids_on_soccer_field (n f : ℕ) (h1 : n = 14) (h2 : f = 3) :
  n + n * f = 56 :=
by
  sorry

end NUMINAMATH_GPT_kids_on_soccer_field_l2421_242157


namespace NUMINAMATH_GPT_find_c_l2421_242149

-- Definitions based on the conditions in the problem
def is_vertex (h k : ℝ) := (5, 1) = (h, k)
def passes_through (x y : ℝ) := (2, 3) = (x, y)

-- Lean theorem statement
theorem find_c (a b c : ℝ) (h k x y : ℝ) (hv : is_vertex h k) (hp : passes_through x y)
  (heq : ∀ y, x = a * y^2 + b * y + c) : c = 17 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l2421_242149


namespace NUMINAMATH_GPT_probability_even_sum_l2421_242151

def p_even_first_wheel : ℚ := 1 / 3
def p_odd_first_wheel : ℚ := 2 / 3
def p_even_second_wheel : ℚ := 3 / 5
def p_odd_second_wheel : ℚ := 2 / 5

theorem probability_even_sum : 
  (p_even_first_wheel * p_even_second_wheel) + (p_odd_first_wheel * p_odd_second_wheel) = 7 / 15 :=
by
  sorry

end NUMINAMATH_GPT_probability_even_sum_l2421_242151


namespace NUMINAMATH_GPT_transformed_A_coordinates_l2421_242106

open Real

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.fst, p.snd)

def A : ℝ × ℝ := (-3, 2)

theorem transformed_A_coordinates :
  reflect_over_y_axis (rotate_90_clockwise A) = (-2, 3) :=
by
  sorry

end NUMINAMATH_GPT_transformed_A_coordinates_l2421_242106


namespace NUMINAMATH_GPT_gray_region_area_is_96pi_l2421_242199

noncomputable def smaller_circle_diameter : ℝ := 4

noncomputable def smaller_circle_radius : ℝ := smaller_circle_diameter / 2

noncomputable def larger_circle_radius : ℝ := 5 * smaller_circle_radius

noncomputable def area_of_larger_circle : ℝ := Real.pi * (larger_circle_radius ^ 2)

noncomputable def area_of_smaller_circle : ℝ := Real.pi * (smaller_circle_radius ^ 2)

noncomputable def area_of_gray_region : ℝ := area_of_larger_circle - area_of_smaller_circle

theorem gray_region_area_is_96pi : area_of_gray_region = 96 * Real.pi := by
  sorry

end NUMINAMATH_GPT_gray_region_area_is_96pi_l2421_242199


namespace NUMINAMATH_GPT_problem_statement_l2421_242156

noncomputable def G (x : ℝ) : ℝ := ((x + 1) ^ 2) / 2 - 4

theorem problem_statement : G (G (G 0)) = -3.9921875 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2421_242156


namespace NUMINAMATH_GPT_smallest_integer_value_of_x_satisfying_eq_l2421_242114

theorem smallest_integer_value_of_x_satisfying_eq (x : ℤ) (h : |x^2 - 5*x + 6| = 14) : 
  ∃ y : ℤ, (y = -1) ∧ ∀ z : ℤ, (|z^2 - 5*z + 6| = 14) → (y ≤ z) :=
sorry

end NUMINAMATH_GPT_smallest_integer_value_of_x_satisfying_eq_l2421_242114


namespace NUMINAMATH_GPT_wendy_total_sales_l2421_242159

noncomputable def apple_price : ℝ := 1.50
noncomputable def orange_price : ℝ := 1.00
noncomputable def morning_apples : ℕ := 40
noncomputable def morning_oranges : ℕ := 30
noncomputable def afternoon_apples : ℕ := 50
noncomputable def afternoon_oranges : ℕ := 40

theorem wendy_total_sales :
  (morning_apples * apple_price + morning_oranges * orange_price) +
  (afternoon_apples * apple_price + afternoon_oranges * orange_price) = 205 := by
  sorry

end NUMINAMATH_GPT_wendy_total_sales_l2421_242159


namespace NUMINAMATH_GPT_no_perfect_square_for_nnplus1_l2421_242147

theorem no_perfect_square_for_nnplus1 :
  ¬ ∃ (n : ℕ), 0 < n ∧ ∃ (k : ℕ), n * (n + 1) = k * k :=
sorry

end NUMINAMATH_GPT_no_perfect_square_for_nnplus1_l2421_242147


namespace NUMINAMATH_GPT_division_by_repeating_decimal_l2421_242184

-- Define the repeating decimal as a fraction
def repeating_decimal := 4 / 9

-- Prove the main theorem
theorem division_by_repeating_decimal : 8 / repeating_decimal = 18 :=
by
  -- lean implementation steps
  sorry

end NUMINAMATH_GPT_division_by_repeating_decimal_l2421_242184


namespace NUMINAMATH_GPT_isosceles_trapezoid_base_ratio_correct_l2421_242188

def isosceles_trapezoid_ratio (x y a b : ℝ) : Prop :=
  b = 2 * x ∧ a = 2 * y ∧ a + b = 10 ∧ (y * (Real.sqrt 2 + 1) = 5) →

  (a / b = (2 * (Real.sqrt 2) - 1) / 2)

theorem isosceles_trapezoid_base_ratio_correct: ∃ (x y a b : ℝ), 
  isosceles_trapezoid_ratio x y a b := sorry

end NUMINAMATH_GPT_isosceles_trapezoid_base_ratio_correct_l2421_242188


namespace NUMINAMATH_GPT_distinguishable_octahedrons_l2421_242152

noncomputable def number_of_distinguishable_octahedrons (total_colors : ℕ) (used_colors : ℕ) : ℕ :=
  let num_ways_choose_colors := Nat.choose total_colors (used_colors - 1)
  let num_permutations := (used_colors - 1).factorial
  let num_rotations := 3
  (num_ways_choose_colors * num_permutations) / num_rotations

theorem distinguishable_octahedrons (h : number_of_distinguishable_octahedrons 9 8 = 13440) : true := sorry

end NUMINAMATH_GPT_distinguishable_octahedrons_l2421_242152


namespace NUMINAMATH_GPT_mod_product_l2421_242163

theorem mod_product : (198 * 955) % 50 = 40 :=
by sorry

end NUMINAMATH_GPT_mod_product_l2421_242163


namespace NUMINAMATH_GPT_remainder_of_modified_division_l2421_242161

theorem remainder_of_modified_division (x y u v : ℕ) (hx : 0 ≤ v ∧ v < y) (hxy : x = u * y + v) :
  ((x + 3 * u * y) % y) = v := by
  sorry

end NUMINAMATH_GPT_remainder_of_modified_division_l2421_242161


namespace NUMINAMATH_GPT_least_n_exceeds_product_l2421_242166

def product_exceeds (n : ℕ) : Prop :=
  10^(n * (n + 1) / 18) > 10^6

theorem least_n_exceeds_product (n : ℕ) (h : n = 12) : product_exceeds n :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_least_n_exceeds_product_l2421_242166


namespace NUMINAMATH_GPT_Lee_payment_total_l2421_242110

theorem Lee_payment_total 
  (ticket_price : ℝ := 10.00)
  (booking_fee : ℝ := 1.50)
  (youngest_discount : ℝ := 0.40)
  (oldest_discount : ℝ := 0.30)
  (middle_discount : ℝ := 0.20)
  (youngest_tickets : ℕ := 3)
  (oldest_tickets : ℕ := 3)
  (middle_tickets : ℕ := 4) :
  (youngest_tickets * (ticket_price * (1 - youngest_discount)) + 
   oldest_tickets * (ticket_price * (1 - oldest_discount)) + 
   middle_tickets * (ticket_price * (1 - middle_discount)) + 
   (youngest_tickets + oldest_tickets + middle_tickets) * booking_fee) = 86.00 :=
by 
  sorry

end NUMINAMATH_GPT_Lee_payment_total_l2421_242110


namespace NUMINAMATH_GPT_robin_albums_l2421_242130

theorem robin_albums (phone_pics : ℕ) (camera_pics : ℕ) (pics_per_album : ℕ) (total_pics : ℕ) (albums_created : ℕ)
  (h1 : phone_pics = 35)
  (h2 : camera_pics = 5)
  (h3 : pics_per_album = 8)
  (h4 : total_pics = phone_pics + camera_pics)
  (h5 : albums_created = total_pics / pics_per_album) : albums_created = 5 := 
sorry

end NUMINAMATH_GPT_robin_albums_l2421_242130


namespace NUMINAMATH_GPT_no_infinite_subdivision_exists_l2421_242123

theorem no_infinite_subdivision_exists : ¬ ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (∀ n : ℕ,
    ∃ (ai bi : ℝ), ai > bi ∧ bi > 0 ∧ ai * bi = a * b ∧
    (ai / bi = a / b ∨ bi / ai = a / b)) :=
sorry

end NUMINAMATH_GPT_no_infinite_subdivision_exists_l2421_242123


namespace NUMINAMATH_GPT_estimated_value_at_28_l2421_242154

-- Definitions based on the conditions
def regression_equation (x : ℝ) : ℝ := 4.75 * x + 257

-- Problem statement
theorem estimated_value_at_28 : regression_equation 28 = 390 :=
by
  -- Sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_estimated_value_at_28_l2421_242154


namespace NUMINAMATH_GPT_geometric_progression_l2421_242153

theorem geometric_progression (p : ℝ) 
  (a b c : ℝ)
  (h1 : a = p - 2)
  (h2 : b = 2 * Real.sqrt p)
  (h3 : c = -3 - p)
  (h4 : b ^ 2 = a * c) : 
  p = 1 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_progression_l2421_242153


namespace NUMINAMATH_GPT_chosen_number_is_120_l2421_242164

theorem chosen_number_is_120 (x : ℤ) (h : 2 * x - 138 = 102) : x = 120 :=
sorry

end NUMINAMATH_GPT_chosen_number_is_120_l2421_242164


namespace NUMINAMATH_GPT_red_and_purple_probability_l2421_242133

def total_balls : ℕ := 120
def white_balls : ℕ := 30
def green_balls : ℕ := 25
def yellow_balls : ℕ := 24
def red_balls : ℕ := 20
def blue_balls : ℕ := 10
def purple_balls : ℕ := 5
def orange_balls : ℕ := 4
def gray_balls : ℕ := 2

def probability_red_purple : ℚ := 5 / 357

theorem red_and_purple_probability :
  ((red_balls / total_balls) * (purple_balls / (total_balls - 1)) +
  (purple_balls / total_balls) * (red_balls / (total_balls - 1))) = probability_red_purple :=
by
  sorry

end NUMINAMATH_GPT_red_and_purple_probability_l2421_242133


namespace NUMINAMATH_GPT_probability_wheel_l2421_242145

theorem probability_wheel (P : ℕ → ℚ) 
  (hA : P 0 = 1/4) 
  (hB : P 1 = 1/3) 
  (hC : P 2 = 1/6) 
  (hSum : P 0 + P 1 + P 2 + P 3 = 1) : 
  P 3 = 1/4 := 
by 
  -- Proof here
  sorry

end NUMINAMATH_GPT_probability_wheel_l2421_242145


namespace NUMINAMATH_GPT_min_value_expression_l2421_242118

theorem min_value_expression : ∃ x y : ℝ, (x = 2 ∧ y = -3/2) ∧ ∀ a b : ℝ, 2 * a^2 + 2 * b^2 - 8 * a + 6 * b + 28 ≥ 10.5 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l2421_242118


namespace NUMINAMATH_GPT_min_value_expr_l2421_242132

theorem min_value_expr (a d : ℝ) (b c : ℝ) (h_a : 0 ≤ a) (h_d : 0 ≤ d) (h_b : 0 < b) (h_c : 0 < c) (h : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end NUMINAMATH_GPT_min_value_expr_l2421_242132


namespace NUMINAMATH_GPT_sum_of_numbers_with_six_zeros_and_56_divisors_l2421_242108

theorem sum_of_numbers_with_six_zeros_and_56_divisors :
  ∃ N1 N2 : ℕ, (N1 % 10^6 = 0) ∧ (N2 % 10^6 = 0) ∧ (N1_divisors = 56) ∧ (N2_divisors = 56) ∧ (N1 + N2 = 7000000) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_with_six_zeros_and_56_divisors_l2421_242108


namespace NUMINAMATH_GPT_frog_problem_l2421_242168

theorem frog_problem 
  (N : ℕ) 
  (h1 : N < 50) 
  (h2 : N % 2 = 1) 
  (h3 : N % 3 = 1) 
  (h4 : N % 4 = 1) 
  (h5 : N % 5 = 0) : 
  N = 25 := 
  sorry

end NUMINAMATH_GPT_frog_problem_l2421_242168


namespace NUMINAMATH_GPT_probability_either_boy_A_or_girl_B_correct_probability_B_correct_conditional_probability_A_given_B_correct_l2421_242111

-- Define the total number of ways to choose 3 leaders from 6 students
def total_ways : ℕ := Nat.choose 6 3

-- Calculate the number of ways in which boy A or girl B is chosen
def boy_A_chosen_ways : ℕ := Nat.choose 4 2 + 4 * 2
def girl_B_chosen_ways : ℕ := Nat.choose 4 1 + Nat.choose 4 2
def either_boy_A_or_girl_B_chosen_ways : ℕ := boy_A_chosen_ways + girl_B_chosen_ways

-- Calculate the probability that either boy A or girl B is chosen
def probability_either_boy_A_or_girl_B : ℚ := either_boy_A_or_girl_B_chosen_ways / total_ways

-- Calculate the probability that girl B is chosen
def girl_B_total_ways : ℕ := Nat.choose 5 2
def probability_B : ℚ := girl_B_total_ways / total_ways

-- Calculate the probability that both boy A and girl B are chosen
def both_A_and_B_chosen_ways : ℕ := Nat.choose 4 1
def probability_AB : ℚ := both_A_and_B_chosen_ways / total_ways

-- Calculate the conditional probability P(A|B) given P(B)
def conditional_probability_A_given_B : ℚ := probability_AB / probability_B

-- Theorem statements
theorem probability_either_boy_A_or_girl_B_correct : probability_either_boy_A_or_girl_B = (4 / 5) := sorry
theorem probability_B_correct : probability_B = (1 / 2) := sorry
theorem conditional_probability_A_given_B_correct : conditional_probability_A_given_B = (2 / 5) := sorry

end NUMINAMATH_GPT_probability_either_boy_A_or_girl_B_correct_probability_B_correct_conditional_probability_A_given_B_correct_l2421_242111


namespace NUMINAMATH_GPT_expand_polynomials_l2421_242187

def p (z : ℝ) : ℝ := 3 * z ^ 2 + 4 * z - 7
def q (z : ℝ) : ℝ := 4 * z ^ 3 - 3 * z + 2

theorem expand_polynomials :
  (p z) * (q z) = 12 * z ^ 5 + 16 * z ^ 4 - 37 * z ^ 3 - 6 * z ^ 2 + 29 * z - 14 := by
  sorry

end NUMINAMATH_GPT_expand_polynomials_l2421_242187


namespace NUMINAMATH_GPT_value_of_expression_when_x_is_neg2_l2421_242116

theorem value_of_expression_when_x_is_neg2 : 
  ∀ (x : ℤ), x = -2 → (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_when_x_is_neg2_l2421_242116


namespace NUMINAMATH_GPT_parallel_lines_iff_m_eq_neg2_l2421_242102

theorem parallel_lines_iff_m_eq_neg2 (m : ℝ) :
  (∀ x y : ℝ, 2 * x + m * y - 2 * m + 4 = 0 → m * x + 2 * y - m + 2 = 0 ↔ m = -2) :=
sorry

end NUMINAMATH_GPT_parallel_lines_iff_m_eq_neg2_l2421_242102


namespace NUMINAMATH_GPT_ratio_of_money_given_l2421_242137

theorem ratio_of_money_given
  (T : ℕ) (W : ℕ) (Th : ℕ) (m : ℕ)
  (h1 : T = 8) 
  (h2 : W = m * T) 
  (h3 : Th = W + 9)
  (h4 : Th = T + 41) : 
  W / T = 5 := 
sorry

end NUMINAMATH_GPT_ratio_of_money_given_l2421_242137


namespace NUMINAMATH_GPT_min_value_l2421_242193

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
3 * a + 6 * b + 12 * c

theorem min_value (a b c : ℝ) (h : 9 * a ^ 2 + 4 * b ^ 2 + 36 * c ^ 2 = 4) :
  minimum_value a b c = -2 * Real.sqrt 14 := sorry

end NUMINAMATH_GPT_min_value_l2421_242193


namespace NUMINAMATH_GPT_geometric_sequence_problem_l2421_242197

variable (a_n : ℕ → ℝ)

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := λ n => a₁ * q^(n-1)

theorem geometric_sequence_problem (q a_1 : ℝ) (a_1_pos : a_1 = 9)
  (h : ∀ n, a_n n = geometric_sequence a_1 q n)
  (h5 : a_n 5 = a_n 3 * (a_n 4)^2) : 
  a_n 4 = 1/3 ∨ a_n 4 = -1/3 := by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l2421_242197


namespace NUMINAMATH_GPT_prob_exceeds_175_l2421_242139

-- Definitions from the conditions
def prob_less_than_160 (p : ℝ) : Prop := p = 0.2
def prob_160_to_175 (p : ℝ) : Prop := p = 0.5

-- The mathematical equivalence proof we need
theorem prob_exceeds_175 (p₁ p₂ p₃ : ℝ) 
  (h₁ : prob_less_than_160 p₁) 
  (h₂ : prob_160_to_175 p₂) 
  (H : p₃ = 1 - (p₁ + p₂)) :
  p₃ = 0.3 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_prob_exceeds_175_l2421_242139


namespace NUMINAMATH_GPT_min_value_proof_l2421_242192

noncomputable def min_value_of_expression (a b c d e f g h : ℝ) : ℝ :=
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2

theorem min_value_proof (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16) :
  ∃ (x : ℝ), x = 32 ∧ min_value_of_expression a b c d e f g h = x :=
by
  use 32
  sorry

end NUMINAMATH_GPT_min_value_proof_l2421_242192


namespace NUMINAMATH_GPT_initial_number_is_31_l2421_242181

theorem initial_number_is_31 (N : ℕ) (h : ∃ k : ℕ, N - 10 = 21 * k) : N = 31 :=
sorry

end NUMINAMATH_GPT_initial_number_is_31_l2421_242181


namespace NUMINAMATH_GPT_inequality_am_gm_l2421_242126

theorem inequality_am_gm (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (h : a^2 + b^2 + c^2 = 12) :
  1/(a-1) + 1/(b-1) + 1/(c-1) ≥ 3 := 
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l2421_242126


namespace NUMINAMATH_GPT_triangle_obtuse_of_cos_relation_l2421_242146

theorem triangle_obtuse_of_cos_relation
  (a b c : ℝ)
  (A B C : ℝ)
  (hTriangle : A + B + C = Real.pi)
  (hSides : a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
  (hSides' : b^2 = a^2 + c^2 - 2*a*c*Real.cos B)
  (hSides'' : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)
  (hRelation : a * Real.cos C = b + 2/3 * c) :
 ∃ (A' : ℝ), A' = A ∧ A > (Real.pi / 2) := 
sorry

end NUMINAMATH_GPT_triangle_obtuse_of_cos_relation_l2421_242146


namespace NUMINAMATH_GPT_employee_pays_216_l2421_242125

def retail_price (wholesale_cost : ℝ) (markup_percentage : ℝ) : ℝ :=
    wholesale_cost + markup_percentage * wholesale_cost

def employee_payment (retail_price : ℝ) (discount_percentage : ℝ) : ℝ :=
    retail_price - discount_percentage * retail_price

theorem employee_pays_216 (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
    wholesale_cost = 200 ∧ markup_percentage = 0.20 ∧ discount_percentage = 0.10 →
    employee_payment (retail_price wholesale_cost markup_percentage) discount_percentage = 216 :=
by
  intro h
  rcases h with ⟨h_wholesale, h_markup, h_discount⟩
  rw [h_wholesale, h_markup, h_discount]
  -- Now we have to prove the final statement: employee_payment (retail_price 200 0.20) 0.10 = 216
  -- This follows directly by computation, so we leave it as a sorry for now
  sorry

end NUMINAMATH_GPT_employee_pays_216_l2421_242125


namespace NUMINAMATH_GPT_probability_of_picking_letter_from_mathematics_l2421_242180

-- Definition of the problem conditions
def extended_alphabet_size := 30
def distinct_letters_in_mathematics := 8

-- Theorem statement
theorem probability_of_picking_letter_from_mathematics :
  (distinct_letters_in_mathematics / extended_alphabet_size : ℚ) = 4 / 15 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_picking_letter_from_mathematics_l2421_242180


namespace NUMINAMATH_GPT_minimum_value_l2421_242120

noncomputable def min_value_b_plus_4_over_a (a : ℝ) (b : ℝ) :=
  b + 4 / a

theorem minimum_value (a : ℝ) (b : ℝ) (h₁ : a > 0) 
  (h₂ : ∀ x : ℝ, x > 0 → (a * x - 2) * (x^2 + b * x - 5) ≥ 0) :
  min_value_b_plus_4_over_a a b = 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_minimum_value_l2421_242120


namespace NUMINAMATH_GPT_y_directly_proportional_x_l2421_242112

-- Definition for direct proportionality
def directly_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x

-- Theorem stating the relationship between y and x given the condition
theorem y_directly_proportional_x (x y : ℝ) (h : directly_proportional x y) :
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x :=
by
  sorry

end NUMINAMATH_GPT_y_directly_proportional_x_l2421_242112


namespace NUMINAMATH_GPT_factorization_of_2210_l2421_242150

theorem factorization_of_2210 : 
  ∃! (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 2210) :=
sorry

end NUMINAMATH_GPT_factorization_of_2210_l2421_242150


namespace NUMINAMATH_GPT_playground_area_l2421_242155

noncomputable def length (w : ℝ) := 2 * w + 30
noncomputable def perimeter (l w : ℝ) := 2 * (l + w)
noncomputable def area (l w : ℝ) := l * w

theorem playground_area :
  ∃ (w l : ℝ), length w = l ∧ perimeter l w = 700 ∧ area l w = 25955.56 :=
by {
  sorry
}

end NUMINAMATH_GPT_playground_area_l2421_242155


namespace NUMINAMATH_GPT_apple_ratio_simplest_form_l2421_242128

theorem apple_ratio_simplest_form (sarah_apples brother_apples cousin_apples : ℕ) 
  (h1 : sarah_apples = 630)
  (h2 : brother_apples = 270)
  (h3 : cousin_apples = 540)
  (gcd_simplified : Nat.gcd (Nat.gcd sarah_apples brother_apples) cousin_apples = 90) :
  (sarah_apples / 90, brother_apples / 90, cousin_apples / 90) = (7, 3, 6) := 
by
  sorry

end NUMINAMATH_GPT_apple_ratio_simplest_form_l2421_242128


namespace NUMINAMATH_GPT_find_m_n_l2421_242136

noncomputable def A : Set ℝ := {3, 5}
def B (m n : ℝ) : Set ℝ := {x | x^2 + m * x + n = 0}

theorem find_m_n (m n : ℝ) (h_union : A ∪ B m n = A) (h_inter : A ∩ B m n = {5}) :
  m = -10 ∧ n = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_m_n_l2421_242136


namespace NUMINAMATH_GPT_willie_initial_bananas_l2421_242176

/-- Given that Willie will have 13 bananas, we need to prove that the initial number of bananas Willie had was some specific number X. --/
theorem willie_initial_bananas (initial_bananas : ℕ) (final_bananas : ℕ) 
    (h : final_bananas = 13) : initial_bananas = initial_bananas :=
by
  sorry

end NUMINAMATH_GPT_willie_initial_bananas_l2421_242176


namespace NUMINAMATH_GPT_distance_from_highest_point_of_sphere_to_bottom_of_glass_l2421_242191

theorem distance_from_highest_point_of_sphere_to_bottom_of_glass :
  ∀ (x y : ℝ),
  x^2 = 2 * y →
  0 ≤ y ∧ y < 15 →
  ∃ b : ℝ, (x^2 + (y - b)^2 = 9) ∧ b = 5 ∧ (b + 3 = 8) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_highest_point_of_sphere_to_bottom_of_glass_l2421_242191


namespace NUMINAMATH_GPT_hyperbola_condition_l2421_242124

theorem hyperbola_condition (m : ℝ) : (∀ x y : ℝ, x^2 + m * y^2 = 1 → m < 0 ↔ x ≠ 0 ∧ y ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l2421_242124


namespace NUMINAMATH_GPT_largest_exponent_l2421_242190

theorem largest_exponent : 
  ∀ (a b c d e : ℕ), a = 2^5000 → b = 3^4000 → c = 4^3000 → d = 5^2000 → e = 6^1000 → b > a ∧ b > c ∧ b > d ∧ b > e :=
by
  sorry

end NUMINAMATH_GPT_largest_exponent_l2421_242190


namespace NUMINAMATH_GPT_polygon_edges_l2421_242185

theorem polygon_edges :
  ∃ a b : ℕ, a + b = 2014 ∧
              (a * (a - 3) / 2 + b * (b - 3) / 2 = 1014053) ∧
              a ≤ b ∧
              a = 952 :=
by
  sorry

end NUMINAMATH_GPT_polygon_edges_l2421_242185


namespace NUMINAMATH_GPT_rounding_addition_to_tenth_l2421_242142

def number1 : Float := 96.23
def number2 : Float := 47.849

theorem rounding_addition_to_tenth (sum : Float) : 
    sum = number1 + number2 →
    Float.round (sum * 10) / 10 = 144.1 :=
by
  intro h
  rw [h]
  norm_num
  sorry -- Skipping the actual rounding proof

end NUMINAMATH_GPT_rounding_addition_to_tenth_l2421_242142


namespace NUMINAMATH_GPT_sum_consecutive_not_power_of_two_l2421_242172

theorem sum_consecutive_not_power_of_two :
  ∀ n k : ℕ, ∀ x : ℕ, n > 0 → k > 0 → (n * (n + 2 * k - 1)) / 2 ≠ 2 ^ x := by
  sorry

end NUMINAMATH_GPT_sum_consecutive_not_power_of_two_l2421_242172


namespace NUMINAMATH_GPT_inequality_proof_l2421_242158

theorem inequality_proof (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) : y * (y - 1) ≤ x^2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2421_242158


namespace NUMINAMATH_GPT_track_width_l2421_242131

theorem track_width (r_1 r_2 : ℝ) (h1 : r_2 = 20) (h2 : 2 * Real.pi * r_1 - 2 * Real.pi * r_2 = 20 * Real.pi) : r_1 - r_2 = 10 :=
sorry

end NUMINAMATH_GPT_track_width_l2421_242131


namespace NUMINAMATH_GPT_hiring_probability_l2421_242179

noncomputable def combinatorics (n k : ℕ) : ℕ := Nat.choose n k

theorem hiring_probability (n : ℕ) (h1 : combinatorics 2 2 = 1)
                          (h2 : combinatorics (n - 2) 1 = n - 2)
                          (h3 : combinatorics n 3 = n * (n - 1) * (n - 2) / 6)
                          (h4 : (6 : ℕ) / (n * (n - 1) : ℚ) = 1 / 15) :
  n = 10 :=
by
  sorry

end NUMINAMATH_GPT_hiring_probability_l2421_242179


namespace NUMINAMATH_GPT_mikes_earnings_l2421_242171

-- Definitions based on the conditions:
def blade_cost : ℕ := 47
def game_count : ℕ := 9
def game_cost : ℕ := 6

-- The total money Mike made:
def total_money (M : ℕ) : Prop :=
  M - (blade_cost + game_count * game_cost) = 0

theorem mikes_earnings (M : ℕ) : total_money M → M = 101 :=
by
  sorry

end NUMINAMATH_GPT_mikes_earnings_l2421_242171


namespace NUMINAMATH_GPT_unique_solution_p_l2421_242160

theorem unique_solution_p (p : ℚ) :
  (∀ x : ℝ, (2 * x + 3) / (p * x - 2) = x) ↔ p = -4 / 3 := sorry

end NUMINAMATH_GPT_unique_solution_p_l2421_242160


namespace NUMINAMATH_GPT_soccer_team_probability_l2421_242119

theorem soccer_team_probability :
  let total_players := 12
  let forwards := 6
  let defenders := 6
  let total_ways := Nat.choose total_players 2
  let defender_ways := Nat.choose defenders 2
  ∃ p : ℚ, p = defender_ways / total_ways ∧ p = 5 / 22 :=
sorry

end NUMINAMATH_GPT_soccer_team_probability_l2421_242119


namespace NUMINAMATH_GPT_scientist_birth_day_is_wednesday_l2421_242100

noncomputable def calculate_birth_day : String :=
  let years := 150
  let leap_years := 36
  let regular_years := years - leap_years
  let total_days_backward := regular_years + 2 * leap_years -- days to move back
  let days_mod := total_days_backward % 7
  let day_of_birth := (5 + 7 - days_mod) % 7 -- 5 is for backward days from Monday
  match day_of_birth with
  | 0 => "Monday"
  | 1 => "Sunday"
  | 2 => "Saturday"
  | 3 => "Friday"
  | 4 => "Thursday"
  | 5 => "Wednesday"
  | 6 => "Tuesday"
  | _ => "Error"

theorem scientist_birth_day_is_wednesday :
  calculate_birth_day = "Wednesday" :=
  by
    sorry

end NUMINAMATH_GPT_scientist_birth_day_is_wednesday_l2421_242100


namespace NUMINAMATH_GPT_distinct_solutions_eq_four_l2421_242141

theorem distinct_solutions_eq_four : ∃! (x : ℝ), abs (x - abs (3 * x + 2)) = 4 :=
by sorry

end NUMINAMATH_GPT_distinct_solutions_eq_four_l2421_242141


namespace NUMINAMATH_GPT_ball_hits_ground_at_time_l2421_242144

theorem ball_hits_ground_at_time :
  ∃ t : ℚ, -9.8 * t^2 + 5.6 * t + 10 = 0 ∧ t = 131 / 98 :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_at_time_l2421_242144


namespace NUMINAMATH_GPT_find_2a_plus_b_l2421_242101

theorem find_2a_plus_b (a b : ℝ) (h1 : 0 < a ∧ a < π / 2) (h2 : 0 < b ∧ b < π / 2)
  (h3 : 5 * (Real.sin a)^2 + 3 * (Real.sin b)^2 = 2)
  (h4 : 4 * Real.sin (2 * a) + 3 * Real.sin (2 * b) = 3) :
  2 * a + b = π / 2 :=
sorry

end NUMINAMATH_GPT_find_2a_plus_b_l2421_242101


namespace NUMINAMATH_GPT_yellow_surface_area_min_fraction_l2421_242143

/-- 
  Given a larger cube with 4-inch edges, constructed from 64 smaller cubes (each with 1-inch edge),
  where 50 cubes are colored blue, and 14 cubes are colored yellow. 
  If the large cube is crafted to display the minimum possible yellow surface area externally,
  then the fraction of the surface area of the large cube that is yellow is 7/48.
-/
theorem yellow_surface_area_min_fraction (n_smaller_cubes blue_cubes yellow_cubes : ℕ) 
  (edge_small edge_large : ℕ) (surface_area_larger_cube yellow_surface_min : ℕ) :
  edge_small = 1 → edge_large = 4 → n_smaller_cubes = 64 → 
  blue_cubes = 50 → yellow_cubes = 14 →
  surface_area_larger_cube = 96 → yellow_surface_min = 14 → 
  (yellow_surface_min : ℚ) / (surface_area_larger_cube : ℚ) = 7 / 48 := 
by 
  intros h_edge_small h_edge_large h_n h_blue h_yellow h_surface_area h_yellow_surface
  sorry

end NUMINAMATH_GPT_yellow_surface_area_min_fraction_l2421_242143


namespace NUMINAMATH_GPT_complete_the_square_l2421_242127

theorem complete_the_square (x : ℝ) : (x^2 + 2 * x - 1 = 0) -> ((x + 1)^2 = 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complete_the_square_l2421_242127


namespace NUMINAMATH_GPT_problem_a_problem_b_l2421_242104

-- Problem (a): Prove that (1 + 1/x)(1 + 1/y) ≥ 9 given x > 0, y > 0, and x + y = 1
theorem problem_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : 
  (1 + 1 / x) * (1 + 1 / y) ≥ 9 := sorry

-- Problem (b): Prove that 0 < u + v - uv < 1 given 0 < u < 1 and 0 < v < 1
theorem problem_b (u v : ℝ) (hu : 0 < u) (hu1 : u < 1) (hv : 0 < v) (hv1 : v < 1) : 
  0 < u + v - u * v ∧ u + v - u * v < 1 := sorry

end NUMINAMATH_GPT_problem_a_problem_b_l2421_242104


namespace NUMINAMATH_GPT_sweatshirt_cost_l2421_242103

/--
Hannah bought 3 sweatshirts and 2 T-shirts.
Each T-shirt cost $10.
Hannah spent $65 in total.
Prove that the cost of each sweatshirt is $15.
-/
theorem sweatshirt_cost (S : ℝ) (h1 : 3 * S + 2 * 10 = 65) : S = 15 :=
by
  sorry

end NUMINAMATH_GPT_sweatshirt_cost_l2421_242103


namespace NUMINAMATH_GPT_simplify_expression_l2421_242117

theorem simplify_expression (x : ℝ) : 
  x - 2 * (1 + x) + 3 * (1 - x) - 4 * (1 + 2 * x) = -12 * x - 3 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_simplify_expression_l2421_242117


namespace NUMINAMATH_GPT_blue_hat_cost_l2421_242134

theorem blue_hat_cost :
  ∀ (total_hats green_hats total_price green_hat_price blue_hat_price) 
  (B : ℕ),
  total_hats = 85 →
  green_hats = 30 →
  total_price = 540 →
  green_hat_price = 7 →
  blue_hat_price = B →
  (30 * 7) + (55 * B) = 540 →
  B = 6 := sorry

end NUMINAMATH_GPT_blue_hat_cost_l2421_242134


namespace NUMINAMATH_GPT_min_abs_diff_x1_x2_l2421_242121

theorem min_abs_diff_x1_x2 (x1 x2 : ℝ) (f : ℝ → ℝ) (Hf : ∀ x, f x = Real.sin (π * x))
  (Hbounds : ∀ x, f x1 ≤ f x ∧ f x ≤ f x2) : |x1 - x2| = 1 := 
by
  sorry

end NUMINAMATH_GPT_min_abs_diff_x1_x2_l2421_242121


namespace NUMINAMATH_GPT_sin_1320_eq_neg_sqrt_3_div_2_l2421_242105

theorem sin_1320_eq_neg_sqrt_3_div_2 : Real.sin (1320 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_1320_eq_neg_sqrt_3_div_2_l2421_242105


namespace NUMINAMATH_GPT_fraction_simplification_l2421_242186

theorem fraction_simplification (x y z : ℝ) (h : x + y + z ≠ 0) :
  (x^2 + y^2 - z^2 + 2 * x * y) / (x^2 + z^2 - y^2 + 2 * x * z) = (x + y - z) / (x + z - y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l2421_242186


namespace NUMINAMATH_GPT_wicket_keeper_age_difference_l2421_242109

def cricket_team_average_age : Nat := 24
def total_members : Nat := 11
def remaining_members : Nat := 9
def age_difference : Nat := 1

theorem wicket_keeper_age_difference :
  let total_age := cricket_team_average_age * total_members
  let remaining_average_age := cricket_team_average_age - age_difference
  let remaining_total_age := remaining_average_age * remaining_members
  let combined_age := total_age - remaining_total_age
  let average_age := cricket_team_average_age
  let wicket_keeper_age := combined_age - average_age
  wicket_keeper_age - average_age = 9 := 
by
  sorry

end NUMINAMATH_GPT_wicket_keeper_age_difference_l2421_242109


namespace NUMINAMATH_GPT_least_three_digit_multiple_13_l2421_242174

theorem least_three_digit_multiple_13 : 
  ∃ n : ℕ, (n ≥ 100) ∧ (∃ k : ℕ, n = 13 * k) ∧ ∀ m, m < n → (m < 100 ∨ ¬∃ k : ℕ, m = 13 * k) :=
by
  sorry

end NUMINAMATH_GPT_least_three_digit_multiple_13_l2421_242174


namespace NUMINAMATH_GPT_remaining_family_member_age_l2421_242162

variable (total_age father_age sister_age : ℕ) (remaining_member_age : ℕ)

def mother_age := father_age - 2
def brother_age := father_age / 2
def known_total_age := father_age + mother_age + brother_age + sister_age

theorem remaining_family_member_age : 
  total_age = 200 ∧ 
  father_age = 60 ∧ 
  sister_age = 40 ∧ 
  known_total_age = total_age - remaining_member_age → 
  remaining_member_age = 12 := by
  sorry

end NUMINAMATH_GPT_remaining_family_member_age_l2421_242162


namespace NUMINAMATH_GPT_sandwiches_consumption_difference_l2421_242113

theorem sandwiches_consumption_difference :
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let monday_total := monday_lunch + monday_dinner

  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let tuesday_total := tuesday_lunch + tuesday_dinner

  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let wednesday_total := wednesday_lunch + wednesday_dinner

  let combined_monday_tuesday := monday_total + tuesday_total

  combined_monday_tuesday - wednesday_total = -5 :=
by
  sorry

end NUMINAMATH_GPT_sandwiches_consumption_difference_l2421_242113


namespace NUMINAMATH_GPT_triangle_altitudes_perfect_square_l2421_242178

theorem triangle_altitudes_perfect_square
  (a b c : ℤ)
  (h : (2 * (↑a * ↑b * ↑c )) = (2 * (↑a * ↑c ) + 2 * (↑a * ↑b))) :
  ∃ k : ℤ, a^2 + b^2 + c^2 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_altitudes_perfect_square_l2421_242178


namespace NUMINAMATH_GPT_expression_divisible_by_7_l2421_242177

theorem expression_divisible_by_7 (n : ℕ) (hn : n > 0) :
  7 ∣ (3^(3*n+1) + 5^(3*n+2) + 7^(3*n+3)) :=
sorry

end NUMINAMATH_GPT_expression_divisible_by_7_l2421_242177


namespace NUMINAMATH_GPT_remainder_of_expression_l2421_242170

theorem remainder_of_expression (n : ℤ) (h : n % 8 = 3) : (4 * n - 10) % 8 = 2 :=
sorry

end NUMINAMATH_GPT_remainder_of_expression_l2421_242170


namespace NUMINAMATH_GPT_find_other_number_l2421_242169

theorem find_other_number (LCM HCF num1 num2 : ℕ) 
  (h1 : LCM = 2310) 
  (h2 : HCF = 30) 
  (h3 : num1 = 330) 
  (h4 : LCM * HCF = num1 * num2) : 
  num2 = 210 := by 
  sorry

end NUMINAMATH_GPT_find_other_number_l2421_242169


namespace NUMINAMATH_GPT_work_completion_days_l2421_242195

theorem work_completion_days (A B : ℕ) (hB : B = 12) (work_together_days : ℕ) (work_together : work_together_days = 3) (work_alone_days : ℕ) (work_alone : work_alone_days = 3) : 
  (1 / A + 1 / B) * 3 + (1 / B) * 3 = 1 → A = 6 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_work_completion_days_l2421_242195


namespace NUMINAMATH_GPT_line_equation_l2421_242173

theorem line_equation {m : ℤ} :
  (∀ x y : ℤ, 2 * x + y + m = 0) →
  (∀ x y : ℤ, 2 * x + y - 10 = 0) →
  (2 * 1 + 0 + m = 0) →
  m = -2 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l2421_242173


namespace NUMINAMATH_GPT_negation_of_exists_inequality_l2421_242183

theorem negation_of_exists_inequality :
  ¬ (∃ x : ℝ, x * x + 4 * x + 5 ≤ 0) ↔ ∀ x : ℝ, x * x + 4 * x + 5 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_inequality_l2421_242183


namespace NUMINAMATH_GPT_fraction_negative_iff_x_lt_2_l2421_242189

theorem fraction_negative_iff_x_lt_2 (x : ℝ) :
  (-5) / (2 - x) < 0 ↔ x < 2 := by
  sorry

end NUMINAMATH_GPT_fraction_negative_iff_x_lt_2_l2421_242189


namespace NUMINAMATH_GPT_irrational_roots_of_odd_coeff_quad_l2421_242140

theorem irrational_roots_of_odd_coeff_quad (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) :
  ¬ ∃ r : ℚ, a * r^2 + b * r + c = 0 := 
sorry

end NUMINAMATH_GPT_irrational_roots_of_odd_coeff_quad_l2421_242140


namespace NUMINAMATH_GPT_find_trapezoid_bases_l2421_242115

-- Define the conditions of the isosceles trapezoid
variables {AD BC : ℝ}
variables (h1 : ∀ (A B C D : ℝ), is_isosceles_trapezoid A B C D ∧ intersects_at_right_angle A B C D)
variables (h2 : ∀ {A B C D : ℝ}, trapezoid_area A B C D = 12)
variables (h3 : ∀ {A B C D : ℝ}, trapezoid_height A B C D = 2)

-- Prove the bases AD and BC are 8 and 4 respectively under the given conditions
theorem find_trapezoid_bases (AD BC : ℝ) : 
  AD = 8 ∧ BC = 4 :=
  sorry

end NUMINAMATH_GPT_find_trapezoid_bases_l2421_242115


namespace NUMINAMATH_GPT_villager4_truth_teller_l2421_242135

def villager1_statement (liars : Finset ℕ) : Prop := liars = {0, 1, 2, 3}
def villager2_statement (liars : Finset ℕ) : Prop := liars.card = 1
def villager3_statement (liars : Finset ℕ) : Prop := liars.card = 2
def villager4_statement (liars : Finset ℕ) : Prop := 3 ∉ liars

theorem villager4_truth_teller (liars : Finset ℕ) :
  ¬ villager1_statement liars ∧
  ¬ villager2_statement liars ∧
  ¬ villager3_statement liars ∧
  villager4_statement liars ↔
  liars = {0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_villager4_truth_teller_l2421_242135


namespace NUMINAMATH_GPT_sophomores_in_program_l2421_242165

theorem sophomores_in_program (total_students : ℕ) (not_sophomores_nor_juniors : ℕ) 
    (percentage_sophomores_debate : ℚ) (percentage_juniors_debate : ℚ) 
    (eq_debate_team : ℚ) (total_students := 40) 
    (not_sophomores_nor_juniors := 5) 
    (percentage_sophomores_debate := 0.20) 
    (percentage_juniors_debate := 0.25) 
    (eq_debate_team := (percentage_sophomores_debate * S = percentage_juniors_debate * J)) :
    ∀ (S J : ℚ), S + J = total_students - not_sophomores_nor_juniors → 
    (S = 5 * J / 4) → S = 175 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_sophomores_in_program_l2421_242165
