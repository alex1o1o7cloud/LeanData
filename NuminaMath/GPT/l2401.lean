import Mathlib

namespace NUMINAMATH_GPT_unique_f_satisfies_eq_l2401_240156

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * (x^2 + 2 * x - 1)

theorem unique_f_satisfies_eq (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, 2 * f x + f (1 - x) = x^2) : 
  ∀ x : ℝ, f x = (1 / 3) * (x^2 + 2 * x - 1) :=
sorry

end NUMINAMATH_GPT_unique_f_satisfies_eq_l2401_240156


namespace NUMINAMATH_GPT_P_is_necessary_but_not_sufficient_for_Q_l2401_240136

def P (x : ℝ) : Prop := |x - 1| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

theorem P_is_necessary_but_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) :=
by
  sorry

end NUMINAMATH_GPT_P_is_necessary_but_not_sufficient_for_Q_l2401_240136


namespace NUMINAMATH_GPT_sequence_arith_or_geom_l2401_240101

def sequence_nature (a S : ℕ → ℝ) : Prop :=
  ∀ n, 4 * S n = (a n + 1) ^ 2

theorem sequence_arith_or_geom {a : ℕ → ℝ} {S : ℕ → ℝ} (h : sequence_nature a S) (h₁ : a 1 = 1) :
  (∃ d, ∀ n, a (n + 1) = a n + d) ∨ (∃ r, ∀ n, a (n + 1) = a n * r) :=
sorry

end NUMINAMATH_GPT_sequence_arith_or_geom_l2401_240101


namespace NUMINAMATH_GPT_total_value_is_76_percent_of_dollar_l2401_240158

def coin_values : List Nat := [1, 5, 20, 50]

def total_value (coins : List Nat) : Nat :=
  List.sum coins

def percentage_of_dollar (value : Nat) : Nat :=
  value * 100 / 100

theorem total_value_is_76_percent_of_dollar :
  percentage_of_dollar (total_value coin_values) = 76 := by
  sorry

end NUMINAMATH_GPT_total_value_is_76_percent_of_dollar_l2401_240158


namespace NUMINAMATH_GPT_lemonade_cups_count_l2401_240183

theorem lemonade_cups_count :
  ∃ x y : ℕ, x + y = 400 ∧ x + 2 * y = 546 ∧ x = 254 :=
by
  sorry

end NUMINAMATH_GPT_lemonade_cups_count_l2401_240183


namespace NUMINAMATH_GPT_max_value_inequality_l2401_240143

theorem max_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (abc * (a + b + c) / ((a + b)^2 * (b + c)^2) ≤ 1 / 4) :=
sorry

end NUMINAMATH_GPT_max_value_inequality_l2401_240143


namespace NUMINAMATH_GPT_expression_equals_4008_l2401_240111

def calculate_expression : ℤ :=
  let expr := (2004 - (2011 - 196)) + (2011 - (196 - 2004))
  expr

theorem expression_equals_4008 : calculate_expression = 4008 := 
by
  sorry

end NUMINAMATH_GPT_expression_equals_4008_l2401_240111


namespace NUMINAMATH_GPT_max_handshakes_l2401_240177

theorem max_handshakes (n : ℕ) (m : ℕ)
  (h_n : n = 25)
  (h_m : m = 20)
  (h_mem : n - m = 5)
  : ∃ (max_handshakes : ℕ), max_handshakes = 250 :=
by
  sorry

end NUMINAMATH_GPT_max_handshakes_l2401_240177


namespace NUMINAMATH_GPT_distance_between_foci_of_ellipse_l2401_240149

theorem distance_between_foci_of_ellipse :
  ∀ x y : ℝ,
  9 * x^2 - 36 * x + 4 * y^2 + 16 * y + 16 = 0 →
  2 * Real.sqrt (9 - 4) = 2 * Real.sqrt 5 :=
by 
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_ellipse_l2401_240149


namespace NUMINAMATH_GPT_find_base_l2401_240142

theorem find_base (a : ℕ) (ha : a > 11) (hB : 11 = 11) :
  (3 * a^2 + 9 * a + 6) + (5 * a^2 + 7 * a + 5) = (9 * a^2 + 7 * a + 11) → 
  a = 12 :=
sorry

end NUMINAMATH_GPT_find_base_l2401_240142


namespace NUMINAMATH_GPT_find_value_of_M_l2401_240155

variable {C y M A : ℕ}

theorem find_value_of_M (h1 : C + y + 2 * M + A = 11)
                        (h2 : C ≠ y)
                        (h3 : C ≠ M)
                        (h4 : C ≠ A)
                        (h5 : y ≠ M)
                        (h6 : y ≠ A)
                        (h7 : M ≠ A)
                        (h8 : 0 < C)
                        (h9 : 0 < y)
                        (h10 : 0 < M)
                        (h11 : 0 < A) : M = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_M_l2401_240155


namespace NUMINAMATH_GPT_factor_tree_value_l2401_240184

-- Define the values and their relationships
def A := 900
def B := 3 * (3 * 2)
def D := 3 * 2
def C := 5 * (5 * 2)
def E := 5 * 2

-- Define the theorem and provide the conditions
theorem factor_tree_value :
  (B = 3 * D) →
  (D = 3 * 2) →
  (C = 5 * E) →
  (E = 5 * 2) →
  (A = B * C) →
  A = 900 := by
  intros hB hD hC hE hA
  sorry

end NUMINAMATH_GPT_factor_tree_value_l2401_240184


namespace NUMINAMATH_GPT_equal_real_roots_value_of_m_l2401_240194

theorem equal_real_roots_value_of_m (m : ℝ) (h : (x^2 - 4*x + m = 0)) 
  (discriminant_zero : (16 - 4*m) = 0) : m = 4 :=
sorry

end NUMINAMATH_GPT_equal_real_roots_value_of_m_l2401_240194


namespace NUMINAMATH_GPT_max_x_add_inv_x_l2401_240198

variable (x : ℝ) (y : Fin 2022 → ℝ)

-- Conditions
def sum_condition : Prop := x + (Finset.univ.sum y) = 2024
def reciprocal_sum_condition : Prop := (1/x) + (Finset.univ.sum (λ i => 1 / (y i))) = 2024

-- The statement we need to prove
theorem max_x_add_inv_x (h_sum : sum_condition x y) (h_rec_sum : reciprocal_sum_condition x y) : 
  x + (1/x) ≤ 2 := by
  sorry

end NUMINAMATH_GPT_max_x_add_inv_x_l2401_240198


namespace NUMINAMATH_GPT_range_of_n_l2401_240182

theorem range_of_n (x : ℕ) (n : ℝ) : 
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 5 → x - 2 < n + 3) → ∃ n, 0 < n ∧ n ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_n_l2401_240182


namespace NUMINAMATH_GPT_who_is_next_to_boris_l2401_240108

/-- Arkady, Boris, Vera, Galya, Danya, and Egor are standing in a circle such that:
1. Danya stood next to Vera, on her right side.
2. Galya stood opposite Egor.
3. Egor stood next to Danya.
4. Arkady and Galya did not want to stand next to each other.
  Prove that Arkady and Galya are standing next to Boris. -/
theorem who_is_next_to_boris
  (Vera Danya Egor Galya Arkady Boris : Prop)
  (H1 : (Danya ∧ Vera))
  (H2 : (Galya ↔ Egor))
  (H3 : (Egor ∧ Danya))
  (H4 : ¬(Arkady ∧ Galya)) 
  : (Arkady ∧ Galya) := 
sorry

end NUMINAMATH_GPT_who_is_next_to_boris_l2401_240108


namespace NUMINAMATH_GPT_fraction_simplification_l2401_240169

open Real -- Open the Real namespace for real number operations

theorem fraction_simplification (a x : ℝ) : 
  (sqrt (a^2 + x^2) - (x^2 + a^2) / sqrt (a^2 + x^2)) / (a^2 + x^2) = 0 := 
sorry

end NUMINAMATH_GPT_fraction_simplification_l2401_240169


namespace NUMINAMATH_GPT_seven_digit_divisible_by_eleven_l2401_240152

theorem seven_digit_divisible_by_eleven (n : ℕ) (h1 : 0 ≤ n) (h2 : n ≤ 9) 
  (h3 : 10 - n ≡ 0 [MOD 11]) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_seven_digit_divisible_by_eleven_l2401_240152


namespace NUMINAMATH_GPT_sum_of_all_N_l2401_240185

-- Define the machine's processing rules
def process (N : ℕ) : ℕ :=
  if N % 2 = 1 then 4 * N + 2 else N / 2

-- Define the 6-step process starting from N
def six_steps (N : ℕ) : ℕ :=
  process (process (process (process (process (process N)))))

-- Definition for the main theorem
theorem sum_of_all_N (N : ℕ) : six_steps N = 10 → N = 640 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_all_N_l2401_240185


namespace NUMINAMATH_GPT_largest_unshaded_area_l2401_240134

theorem largest_unshaded_area (s : ℝ) (π_approx : ℝ) :
    (let r := s / 2
     let area_square := s^2
     let area_circle := π_approx * r^2
     let area_triangle := (1 / 2) * (s / 2) * (s / 2)
     let unshaded_square := area_square - area_circle
     let unshaded_circle := area_circle - area_triangle
     unshaded_circle) > (unshaded_square) := by
        sorry

end NUMINAMATH_GPT_largest_unshaded_area_l2401_240134


namespace NUMINAMATH_GPT_age_of_female_employee_when_hired_l2401_240199

-- Defining the conditions
def hired_year : ℕ := 1989
def retirement_year : ℕ := 2008
def sum_age_employment : ℕ := 70

-- Given the conditions we found that years of employment (Y):
def years_of_employment : ℕ := retirement_year - hired_year -- 19

-- Defining the age when hired (A)
def age_when_hired : ℕ := sum_age_employment - years_of_employment -- 51

-- Now we need to prove
theorem age_of_female_employee_when_hired : age_when_hired = 51 :=
by
  -- Here should be the proof steps, but we use sorry for now
  sorry

end NUMINAMATH_GPT_age_of_female_employee_when_hired_l2401_240199


namespace NUMINAMATH_GPT_arrangements_count_correct_l2401_240106

noncomputable def count_arrangements : ℕ :=
  -- The total number of different arrangements of students A, B, C, D in 3 communities
  -- such that each community has at least one student, and A and B are not in the same community.
  sorry

theorem arrangements_count_correct : count_arrangements = 30 := by
  sorry

end NUMINAMATH_GPT_arrangements_count_correct_l2401_240106


namespace NUMINAMATH_GPT_time_to_cross_tree_l2401_240137

theorem time_to_cross_tree (length_train : ℝ) (length_platform : ℝ) (time_to_pass_platform : ℝ) (h1 : length_train = 1200) (h2 : length_platform = 1200) (h3 : time_to_pass_platform = 240) : 
  (length_train / ((length_train + length_platform) / time_to_pass_platform)) = 120 := 
by
    sorry

end NUMINAMATH_GPT_time_to_cross_tree_l2401_240137


namespace NUMINAMATH_GPT_no_hamiltonian_cycle_l2401_240104

-- Define the problem constants
def n : ℕ := 2016
def a : ℕ := 2
def b : ℕ := 3

-- Define the circulant graph and the conditions of the Hamiltonian cycle theorem
theorem no_hamiltonian_cycle (s t : ℕ) (h1 : s + t = Int.gcd n (a - b)) :
  ¬ (Int.gcd n (s * a + t * b) = 1) :=
by
  sorry  -- Proof not required as per instructions

end NUMINAMATH_GPT_no_hamiltonian_cycle_l2401_240104


namespace NUMINAMATH_GPT_inequality_solution_set_l2401_240154

theorem inequality_solution_set (x : ℝ) :
  ((1 - x) * (x - 3) < 0) ↔ (x < 1 ∨ x > 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l2401_240154


namespace NUMINAMATH_GPT_real_solutions_eq_pos_neg_2_l2401_240170

theorem real_solutions_eq_pos_neg_2 (x : ℝ) :
  ( (x - 1) ^ 2 * (x - 5) * (x - 5) / (x - 5) = 4) ↔ (x = 3 ∨ x = -1) :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_eq_pos_neg_2_l2401_240170


namespace NUMINAMATH_GPT_number_above_210_is_165_l2401_240132

def triangular_number (k : ℕ) : ℕ := k * (k + 1) / 2
def tetrahedral_number (k : ℕ) : ℕ := k * (k + 1) * (k + 2) / 6
def row_start (k : ℕ) : ℕ := tetrahedral_number (k - 1) + 1

theorem number_above_210_is_165 :
  ∀ k, triangular_number k = 210 →
  ∃ n, n = 165 → 
  ∀ m, row_start (k - 1) ≤ m ∧ m < row_start k →
  m = 210 →
  n = m - triangular_number (k - 1) :=
  sorry

end NUMINAMATH_GPT_number_above_210_is_165_l2401_240132


namespace NUMINAMATH_GPT_solve_quadratic_l2401_240145

theorem solve_quadratic :
  ∀ x, (x^2 - x - 12 = 0) → (x = -3 ∨ x = 4) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_l2401_240145


namespace NUMINAMATH_GPT_evaluate_powers_l2401_240140

theorem evaluate_powers : (81^(1/2:ℝ) * 64^(-1/3:ℝ) * 49^(1/4:ℝ) = 9 * (1/4) * Real.sqrt 7) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_powers_l2401_240140


namespace NUMINAMATH_GPT_sum_of_digits_5_pow_eq_2_pow_l2401_240123

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_5_pow_eq_2_pow (n : ℕ) (h : sum_of_digits (5^n) = 2^n) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_5_pow_eq_2_pow_l2401_240123


namespace NUMINAMATH_GPT_minimum_value_of_sum_l2401_240138

theorem minimum_value_of_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : 1/a + 2/b + 3/c = 2) : a + 2*b + 3*c = 18 ↔ (a = 3 ∧ b = 3 ∧ c = 3) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_sum_l2401_240138


namespace NUMINAMATH_GPT_find_multiple_l2401_240153

-- Definitions based on the conditions provided
def mike_chocolate_squares : ℕ := 20
def jenny_chocolate_squares : ℕ := 65
def extra_squares : ℕ := 5

-- The theorem to prove the multiple
theorem find_multiple : ∃ (multiple : ℕ), jenny_chocolate_squares = mike_chocolate_squares * multiple + extra_squares ∧ multiple = 3 := by
  sorry

end NUMINAMATH_GPT_find_multiple_l2401_240153


namespace NUMINAMATH_GPT_bacteria_after_time_l2401_240175

def initial_bacteria : ℕ := 1
def division_time : ℕ := 20  -- time in minutes for one division
def total_time : ℕ := 180  -- total time in minutes

def divisions := total_time / division_time

theorem bacteria_after_time : (initial_bacteria * 2 ^ divisions) = 512 := by
  exact sorry

end NUMINAMATH_GPT_bacteria_after_time_l2401_240175


namespace NUMINAMATH_GPT_symmetric_points_x_axis_l2401_240107

theorem symmetric_points_x_axis (a b : ℝ) (h_a : a = -2) (h_b : b = -1) : a + b = -3 :=
by
  -- Skipping the proof steps and adding sorry
  sorry

end NUMINAMATH_GPT_symmetric_points_x_axis_l2401_240107


namespace NUMINAMATH_GPT_cards_problem_l2401_240193

-- Define the conditions and goal
theorem cards_problem 
    (L R : ℕ) 
    (h1 : L + 6 = 3 * (R - 6))
    (h2 : R + 2 = 2 * (L - 2)) : 
    L = 66 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_cards_problem_l2401_240193


namespace NUMINAMATH_GPT_unique_solution_l2401_240128

noncomputable def f (x : ℝ) : ℝ := 2^x + 3^x + 6^x

theorem unique_solution : ∀ x : ℝ, f x = 7^x ↔ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l2401_240128


namespace NUMINAMATH_GPT_problem_I_problem_II_l2401_240126

open Set Real

-- Problem (I)
theorem problem_I (x : ℝ) :
  (|x - 2| ≥ 4 - |x - 1|) ↔ x ∈ Iic (-1/2) ∪ Ici (7/2) :=
by
  sorry

-- Problem (II)
theorem problem_II (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : 1/m + 1/2/n = 1) :
  m + 2 * n ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l2401_240126


namespace NUMINAMATH_GPT_remainder_of_12_pow_2012_mod_5_l2401_240141

theorem remainder_of_12_pow_2012_mod_5 : (12 ^ 2012) % 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_12_pow_2012_mod_5_l2401_240141


namespace NUMINAMATH_GPT_num_real_x_l2401_240118

theorem num_real_x (a b : ℝ) (h1 : a = 123) (h2 : b = 11) :
  ∃ n : ℕ, n = 12 ∧
  ∀ k : ℕ, k ≤ b → ∃ x : ℝ, x = (a - k^2)^2 :=
by
  sorry

end NUMINAMATH_GPT_num_real_x_l2401_240118


namespace NUMINAMATH_GPT_cos_squared_sum_sin_squared_sum_l2401_240161

theorem cos_squared_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.cos (A / 2) ^ 2 + Real.cos (B / 2) ^ 2 + Real.cos (C / 2) ^ 2 =
  2 * (1 + Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2)) :=
sorry

theorem sin_squared_sum (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (A / 2) ^ 2 + Real.sin (B / 2) ^ 2 + Real.sin (C / 2) ^ 2 =
  1 - 2 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) :=
sorry

end NUMINAMATH_GPT_cos_squared_sum_sin_squared_sum_l2401_240161


namespace NUMINAMATH_GPT_A_inter_CUB_eq_l2401_240190

noncomputable def U := Set.univ (ℝ)

noncomputable def A := { x : ℝ | 0 ≤ x ∧ x ≤ 2 }

noncomputable def B := { y : ℝ | ∃ x : ℝ, x ∈ A ∧ y = x + 1 }

noncomputable def C_U (s : Set ℝ) := { x : ℝ | x ∉ s }

noncomputable def A_inter_CUB := A ∩ C_U B

theorem A_inter_CUB_eq : A_inter_CUB = { x : ℝ | 0 ≤ x ∧ x < 1 } :=
  by sorry

end NUMINAMATH_GPT_A_inter_CUB_eq_l2401_240190


namespace NUMINAMATH_GPT_An_is_integer_l2401_240168

theorem An_is_integer 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_gt : a > b)
  (θ : ℝ) (h_theta : θ > 0 ∧ θ < Real.pi / 2)
  (h_sin : Real.sin θ = 2 * (a * b) / (a^2 + b^2)) :
  ∀ n : ℕ, ∃ k : ℤ, ((a^2 + b^2)^n * Real.sin (n * θ) : ℝ) = k :=
by sorry

end NUMINAMATH_GPT_An_is_integer_l2401_240168


namespace NUMINAMATH_GPT_ahmed_goats_correct_l2401_240102

-- Definitions based on the conditions given in the problem.
def adam_goats : ℕ := 7
def andrew_goats : ℕ := 5 + 2 * adam_goats
def ahmed_goats : ℕ := andrew_goats - 6

-- The theorem statement that needs to be proven.
theorem ahmed_goats_correct : ahmed_goats = 13 := by
    sorry

end NUMINAMATH_GPT_ahmed_goats_correct_l2401_240102


namespace NUMINAMATH_GPT_least_common_multiple_of_5_to_10_is_2520_l2401_240160

-- Definitions of the numbers
def numbers : List ℤ := [5, 6, 7, 8, 9, 10]

-- Definition of prime factorization for verification (optional, keeping it simple)
def prime_factors (n : ℤ) : List ℤ :=
  if n = 5 then [5]
  else if n = 6 then [2, 3]
  else if n = 7 then [7]
  else if n = 8 then [2, 2, 2]
  else if n = 9 then [3, 3]
  else if n = 10 then [2, 5]
  else []

-- The property to be proved: The least common multiple of numbers is 2520
theorem least_common_multiple_of_5_to_10_is_2520 : ∃ n : ℕ, (∀ m ∈ numbers, m ∣ n) ∧ n = 2520 := by
  use 2520
  sorry

end NUMINAMATH_GPT_least_common_multiple_of_5_to_10_is_2520_l2401_240160


namespace NUMINAMATH_GPT_find_k_l2401_240119

noncomputable def distance_x (x : ℝ) := 5
noncomputable def distance_y (x k : ℝ) := |x^2 - k|
noncomputable def total_distance (x k : ℝ) := distance_x x + distance_y x k

theorem find_k (x k : ℝ) (hk : distance_y x k = 2 * distance_x x) (htot : total_distance x k = 30) :
  k = x^2 - 10 :=
sorry

end NUMINAMATH_GPT_find_k_l2401_240119


namespace NUMINAMATH_GPT_opposite_of_negative_one_fifth_l2401_240146

theorem opposite_of_negative_one_fifth : -(-1 / 5) = (1 / 5) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_negative_one_fifth_l2401_240146


namespace NUMINAMATH_GPT_find_two_digits_l2401_240121

theorem find_two_digits (a b : ℕ) (h₁: a ≤ 9) (h₂: b ≤ 9)
  (h₃: (4 + a + b) % 9 = 0) (h₄: (10 * a + b) % 4 = 0) :
  (a = 3 ∧ b = 2) ∨ (a = 6 ∧ b = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_two_digits_l2401_240121


namespace NUMINAMATH_GPT_parametric_two_rays_l2401_240178

theorem parametric_two_rays (t : ℝ) : (x = t + 1 / t ∧ y = 2) → (x ≤ -2 ∨ x ≥ 2) := by
  sorry

end NUMINAMATH_GPT_parametric_two_rays_l2401_240178


namespace NUMINAMATH_GPT_billy_laundry_loads_l2401_240117

-- Define constants based on problem conditions
def sweeping_minutes_per_room := 3
def washing_minutes_per_dish := 2
def laundry_minutes_per_load := 9

def anna_rooms := 10
def billy_dishes := 6

-- Calculate total time spent by Anna and the time Billy spends washing dishes
def anna_total_time := sweeping_minutes_per_room * anna_rooms
def billy_dishwashing_time := washing_minutes_per_dish * billy_dishes

-- Define the time difference Billy needs to make up with laundry
def time_difference := anna_total_time - billy_dishwashing_time
def billy_required_laundry_loads := time_difference / laundry_minutes_per_load

-- The theorem to prove
theorem billy_laundry_loads : billy_required_laundry_loads = 2 := by 
  sorry

end NUMINAMATH_GPT_billy_laundry_loads_l2401_240117


namespace NUMINAMATH_GPT_fraction_problem_l2401_240181

theorem fraction_problem 
  (x : ℚ)
  (h : x = 45 / (8 - (3 / 7))) : 
  x = 315 / 53 := 
sorry

end NUMINAMATH_GPT_fraction_problem_l2401_240181


namespace NUMINAMATH_GPT_min_value_expression_l2401_240130

theorem min_value_expression (x : ℝ) (h : x > 1) : 
  ∃ min_val, min_val = 6 ∧ ∀ y > 1, 2 * y + 2 / (y - 1) ≥ min_val :=
by  
  use 6
  sorry

end NUMINAMATH_GPT_min_value_expression_l2401_240130


namespace NUMINAMATH_GPT_combined_parent_age_difference_l2401_240120

def father_age_at_sobha_birth : ℕ := 38
def mother_age_at_brother_birth : ℕ := 36
def brother_younger_than_sobha : ℕ := 4
def sister_younger_than_brother : ℕ := 3
def father_age_at_sister_birth : ℕ := 45
def mother_age_at_youngest_birth : ℕ := 34
def youngest_younger_than_sister : ℕ := 6

def mother_age_at_sobha_birth := mother_age_at_brother_birth - brother_younger_than_sobha
def father_age_at_youngest_birth := father_age_at_sister_birth + youngest_younger_than_sister

def combined_age_difference_at_sobha_birth := father_age_at_sobha_birth - mother_age_at_sobha_birth
def compounded_difference_at_sobha_brother_birth := 
  (father_age_at_sobha_birth + brother_younger_than_sobha) - mother_age_at_brother_birth
def mother_age_at_sister_birth := mother_age_at_brother_birth + sister_younger_than_brother
def compounded_difference_at_sobha_sister_birth := father_age_at_sister_birth - mother_age_at_sister_birth
def compounded_difference_at_youngest_birth := father_age_at_youngest_birth - mother_age_at_youngest_birth

def combined_age_difference := 
  combined_age_difference_at_sobha_birth + 
  compounded_difference_at_sobha_brother_birth + 
  compounded_difference_at_sobha_sister_birth + 
  compounded_difference_at_youngest_birth 

theorem combined_parent_age_difference : combined_age_difference = 35 := by
  sorry

end NUMINAMATH_GPT_combined_parent_age_difference_l2401_240120


namespace NUMINAMATH_GPT_find_k_l2401_240164

variable (m n p k : ℝ)

-- Conditions
def cond1 : Prop := m = 2 * n + 5
def cond2 : Prop := p = 3 * m - 4
def cond3 : Prop := m + 4 = 2 * (n + k) + 5
def cond4 : Prop := p + 3 = 3 * (m + 4) - 4

theorem find_k (h1 : cond1 m n)
               (h2 : cond2 m p)
               (h3 : cond3 m n k)
               (h4 : cond4 m p) :
               k = 2 :=
  sorry

end NUMINAMATH_GPT_find_k_l2401_240164


namespace NUMINAMATH_GPT_chef_cooked_potatoes_l2401_240100

theorem chef_cooked_potatoes
  (total_potatoes : ℕ)
  (cooking_time_per_potato : ℕ)
  (remaining_cooking_time : ℕ)
  (left_potatoes : ℕ)
  (cooked_potatoes : ℕ) :
  total_potatoes = 16 →
  cooking_time_per_potato = 5 →
  remaining_cooking_time = 45 →
  remaining_cooking_time / cooking_time_per_potato = left_potatoes →
  total_potatoes - left_potatoes = cooked_potatoes →
  cooked_potatoes = 7 :=
by
  intros h_total h_cooking_time h_remaining_time h_left_potatoes h_cooked_potatoes
  sorry

end NUMINAMATH_GPT_chef_cooked_potatoes_l2401_240100


namespace NUMINAMATH_GPT_freezer_temp_is_correct_l2401_240196

def freezer_temp (temp: ℤ) := temp

theorem freezer_temp_is_correct (temp: ℤ)
  (freezer_below_zero: temp = -18): freezer_temp temp = -18 := 
by
  -- since freezer_below_zero state that temperature is -18
  exact freezer_below_zero

end NUMINAMATH_GPT_freezer_temp_is_correct_l2401_240196


namespace NUMINAMATH_GPT_vector_operation_l2401_240109

variables {α : Type*} [AddCommGroup α] [Module ℝ α]
variables (a b : α)

theorem vector_operation :
  (1 / 2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by sorry

end NUMINAMATH_GPT_vector_operation_l2401_240109


namespace NUMINAMATH_GPT_church_full_capacity_l2401_240187

theorem church_full_capacity
  (chairs_per_row : ℕ)
  (rows : ℕ)
  (people_per_chair : ℕ)
  (h1 : chairs_per_row = 6)
  (h2 : rows = 20)
  (h3 : people_per_chair = 5) :
  (chairs_per_row * rows * people_per_chair) = 600 := by
  sorry

end NUMINAMATH_GPT_church_full_capacity_l2401_240187


namespace NUMINAMATH_GPT_number_of_students_in_Diligence_before_transfer_l2401_240171

-- Define the total number of students and the transfer information
def total_students : ℕ := 50
def transferred_students : ℕ := 2

-- Define the number of students in Diligence before the transfer
def students_in_Diligence_before : ℕ := 23

-- Let's prove that the number of students in Diligence before the transfer is 23
theorem number_of_students_in_Diligence_before_transfer :
  (total_students / 2) - transferred_students = students_in_Diligence_before :=
by {
  -- The proof is omitted as instructed
  sorry
}

end NUMINAMATH_GPT_number_of_students_in_Diligence_before_transfer_l2401_240171


namespace NUMINAMATH_GPT_weight_of_e_l2401_240116

variables (d e f : ℝ)

theorem weight_of_e
  (h_de_f : (d + e + f) / 3 = 42)
  (h_de : (d + e) / 2 = 35)
  (h_ef : (e + f) / 2 = 41) :
  e = 26 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_e_l2401_240116


namespace NUMINAMATH_GPT_triangle_obtuse_of_inequality_l2401_240112

theorem triangle_obtuse_of_inequality
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)
  (ineq : a^2 < (b + c) * (c - b)) :
  ∃ (A B C : ℝ), (A + B + C = π) ∧ (C > π / 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_obtuse_of_inequality_l2401_240112


namespace NUMINAMATH_GPT_focus_of_hyperbola_l2401_240110

-- Define the given hyperbola equation and its conversion to standard form
def hyperbola_eq (x y : ℝ) : Prop := -2 * (x - 2)^2 + 3 * (y + 3)^2 - 28 = 0

-- Define the standard form equation of the hyperbola
def standard_form (x y : ℝ) : Prop :=
  ((y + 3)^2 / (28 / 3)) - ((x - 2)^2 / 14) = 1

-- Define the coordinates of one of the foci of the hyperbola
def focus (x y : ℝ) : Prop :=
  x = 2 ∧ y = -3 + Real.sqrt (70 / 3)

-- The theorem statement proving the given coordinates is a focus of the hyperbola
theorem focus_of_hyperbola :
  ∃ x y, hyperbola_eq x y ∧ standard_form x y → focus x y :=
by
  existsi 2, (-3 + Real.sqrt (70 / 3))
  sorry -- Proof is required to substantiate it, placeholder here.

end NUMINAMATH_GPT_focus_of_hyperbola_l2401_240110


namespace NUMINAMATH_GPT_perfect_square_solutions_l2401_240195

theorem perfect_square_solutions (a b : ℕ) (ha : a > b) (ha_pos : 0 < a) (hb_pos : 0 < b) (hA : ∃ k : ℕ, a^2 + 4 * b + 1 = k^2) (hB : ∃ l : ℕ, b^2 + 4 * a + 1 = l^2) :
  a = 8 ∧ b = 4 ∧ (a^2 + 4 * b + 1 = (a+1)^2) ∧ (b^2 + 4 * a + 1 = (b + 3)^2) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_solutions_l2401_240195


namespace NUMINAMATH_GPT_inequality_proof_l2401_240167

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (b + c)) + (1 / (a + c)) + (1 / (a + b)) ≥ 9 / (2 * (a + b + c)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2401_240167


namespace NUMINAMATH_GPT_train_speed_l2401_240157

theorem train_speed (distance time : ℝ) (h1 : distance = 400) (h2 : time = 10) : 
  distance / time = 40 := 
sorry

end NUMINAMATH_GPT_train_speed_l2401_240157


namespace NUMINAMATH_GPT_find_a_of_inequality_solution_set_l2401_240197

theorem find_a_of_inequality_solution_set :
  (∃ (a : ℝ), (∀ (x : ℝ), |2*x - a| + a ≤ 4 ↔ -1 ≤ x ∧ x ≤ 2) ∧ a = 1) :=
by sorry

end NUMINAMATH_GPT_find_a_of_inequality_solution_set_l2401_240197


namespace NUMINAMATH_GPT_skylar_starting_age_l2401_240122

-- Conditions of the problem
def annual_donation : ℕ := 8000
def current_age : ℕ := 71
def total_amount_donated : ℕ := 440000

-- Question and proof statement
theorem skylar_starting_age :
  (current_age - total_amount_donated / annual_donation) = 16 := 
by
  sorry

end NUMINAMATH_GPT_skylar_starting_age_l2401_240122


namespace NUMINAMATH_GPT_replace_asterisk_l2401_240144

theorem replace_asterisk (star : ℝ) : ((36 / 18) * (star / 72) = 1) → star = 36 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_replace_asterisk_l2401_240144


namespace NUMINAMATH_GPT_complex_number_identity_l2401_240176

theorem complex_number_identity (i : ℂ) (hi : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_number_identity_l2401_240176


namespace NUMINAMATH_GPT_geometric_sequence_a7_l2401_240114

noncomputable def a (n : ℕ) : ℝ := sorry -- Definition of the sequence

theorem geometric_sequence_a7 :
  a 3 = 1 → a 11 = 25 → a 7 = 5 := 
by
  intros h3 h11
  sorry

end NUMINAMATH_GPT_geometric_sequence_a7_l2401_240114


namespace NUMINAMATH_GPT_max_profit_at_grade_9_l2401_240148

def profit (k : ℕ) : ℕ :=
  (8 + 2 * (k - 1)) * (60 - 3 * (k - 1))

theorem max_profit_at_grade_9 : ∀ k, 1 ≤ k ∧ k ≤ 10 → profit k ≤ profit 9 := 
by
  sorry

end NUMINAMATH_GPT_max_profit_at_grade_9_l2401_240148


namespace NUMINAMATH_GPT_part_a_part_b_l2401_240186

-- This definition states that a number p^m is a divisor of a-1
def divides (p : ℕ) (m : ℕ) (a : ℕ) : Prop :=
  (p ^ m) ∣ (a - 1)

-- This definition states that (p^(m+1)) is not a divisor of a-1
def not_divides (p : ℕ) (m : ℕ) (a : ℕ) : Prop :=
  ¬ (p ^ (m + 1) ∣ (a - 1))

-- Part (a): Prove divisibility
theorem part_a (a m : ℕ) (p : ℕ) [hp: Fact p.Prime] (ha: a > 0) (hm: m > 0)
  (h1: divides p m a) (h2: not_divides p m a) (n : ℕ) : 
  p ^ (m + n) ∣ a ^ (p ^ n) - 1 := 
sorry

-- Part (b): Prove non-divisibility
theorem part_b (a m : ℕ) (p : ℕ) [hp: Fact p.Prime] (ha: a > 0) (hm: m > 0)
  (h1: divides p m a) (h2: not_divides p m a) (n : ℕ) : 
  ¬ p ^ (m + n + 1) ∣ a ^ (p ^ n) - 1 := 
sorry

end NUMINAMATH_GPT_part_a_part_b_l2401_240186


namespace NUMINAMATH_GPT_rhombus_perimeter_area_l2401_240103

theorem rhombus_perimeter_area (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) (right_angle : ∀ (x : ℝ), x = d1 / 2 ∧ x = d2 / 2 → x * x + x * x = (d1 / 2)^2 + (d2 / 2)^2) : 
  ∃ (P A : ℝ), P = 52 ∧ A = 120 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_area_l2401_240103


namespace NUMINAMATH_GPT_find_minimal_x_l2401_240139

-- Conditions
variables (x y : ℕ)
variable (pos_x : x > 0)
variable (pos_y : y > 0)
variable (h : 3 * x^7 = 17 * y^11)

-- Proof Goal
theorem find_minimal_x : ∃ a b c d : ℕ, x = a^c * b^d ∧ a + b + c + d = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_minimal_x_l2401_240139


namespace NUMINAMATH_GPT_line_equation_is_correct_l2401_240162

noncomputable def line_has_equal_intercepts_and_passes_through_A (p q : ℝ) : Prop :=
(p, q) = (3, 2) ∧ q ≠ 0 ∧ (∃ c : ℝ, p + q = c ∨ 2 * p - 3 * q = 0)

theorem line_equation_is_correct :
  line_has_equal_intercepts_and_passes_through_A 3 2 → 
  (∃ f g : ℝ, 2 * f - 3 * g = 0 ∨ f + g = 5) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_is_correct_l2401_240162


namespace NUMINAMATH_GPT_simplify_expression_correct_l2401_240188

def simplify_expression : ℚ :=
  15 * (7 / 10) * (1 / 9)

theorem simplify_expression_correct : simplify_expression = 7 / 6 :=
by
  unfold simplify_expression
  sorry

end NUMINAMATH_GPT_simplify_expression_correct_l2401_240188


namespace NUMINAMATH_GPT_range_of_a_l2401_240180

noncomputable def parabola_locus (x : ℝ) : ℝ := x^2 / 4

def angle_sum_property (a k : ℝ) : Prop :=
  2 * a * k^2 + 2 * k + a = 0

def discriminant_nonnegative (a : ℝ) : Prop :=
  4 - 8 * a^2 ≥ 0

theorem range_of_a (a : ℝ) :
  (- (Real.sqrt 2) / 2) ≤ a ∧ a ≤ (Real.sqrt 2) / 2 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l2401_240180


namespace NUMINAMATH_GPT_smaller_of_two_integers_l2401_240173

theorem smaller_of_two_integers (m n : ℕ) (h1 : 100 ≤ m ∧ m < 1000) (h2 : 100 ≤ n ∧ n < 1000)
  (h3 : (m + n) / 2 = m + n / 1000) : min m n = 999 :=
by {
  sorry
}

end NUMINAMATH_GPT_smaller_of_two_integers_l2401_240173


namespace NUMINAMATH_GPT_roots_magnitude_order_l2401_240192

theorem roots_magnitude_order (m : ℝ) (a b c d : ℝ)
  (h1 : m > 0)
  (h2 : a ^ 2 - m * a - 1 = 0)
  (h3 : b ^ 2 - m * b - 1 = 0)
  (h4 : c ^ 2 + m * c - 1 = 0)
  (h5 : d ^ 2 + m * d - 1 = 0)
  (ha_pos : a > 0) (hb_neg : b < 0)
  (hc_pos : c > 0) (hd_neg : d < 0) :
  |a| > |c| ∧ |c| > |b| ∧ |b| > |d| :=
sorry

end NUMINAMATH_GPT_roots_magnitude_order_l2401_240192


namespace NUMINAMATH_GPT_range_of_m_l2401_240159

theorem range_of_m : 
  ∀ m : ℝ, m = 3 * Real.sqrt 2 - 1 → 3 < m ∧ m < 4 := 
by
  -- the proof will go here
  sorry

end NUMINAMATH_GPT_range_of_m_l2401_240159


namespace NUMINAMATH_GPT_living_room_size_l2401_240189

theorem living_room_size :
  let length := 16
  let width := 10
  let total_rooms := 6
  let total_area := length * width
  let unit_size := total_area / total_rooms
  let living_room_size := 3 * unit_size
  living_room_size = 80 := by
    sorry

end NUMINAMATH_GPT_living_room_size_l2401_240189


namespace NUMINAMATH_GPT_polynomial_coefficient_B_l2401_240172

theorem polynomial_coefficient_B : 
  ∃ (A C D : ℤ), 
    (∀ z : ℤ, (z > 0) → (z^6 - 15 * z^5 + A * z^4 + B * z^3 + C * z^2 + D * z + 64 = 0)) ∧ 
    (B = -244) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_coefficient_B_l2401_240172


namespace NUMINAMATH_GPT_find_q_l2401_240124

-- Given conditions
noncomputable def digits_non_zero (p q r : Nat) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0

noncomputable def three_digit_number (p q r : Nat) : Nat :=
  100 * p + 10 * q + r

noncomputable def two_digit_number (q r : Nat) : Nat :=
  10 * q + r

noncomputable def one_digit_number (r : Nat) : Nat := r

noncomputable def numbers_sum_to (p q r sum : Nat) : Prop :=
  three_digit_number p q r + two_digit_number q r + one_digit_number r = sum

-- The theorem to prove
theorem find_q (p q r : Nat) (hpq : digits_non_zero p q r)
  (hsum : numbers_sum_to p q r 912) : q = 5 := sorry

end NUMINAMATH_GPT_find_q_l2401_240124


namespace NUMINAMATH_GPT_no_such_functions_exist_l2401_240115

theorem no_such_functions_exist :
  ¬ ∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y + 1 :=
by
  sorry

end NUMINAMATH_GPT_no_such_functions_exist_l2401_240115


namespace NUMINAMATH_GPT_circumcircle_diameter_triangle_ABC_l2401_240113

theorem circumcircle_diameter_triangle_ABC
  (A : ℝ) (BC : ℝ) (R : ℝ)
  (hA : A = 60) (hBC : BC = 4)
  (hR_formula : 2 * R = BC / Real.sin (A * Real.pi / 180)) :
  2 * R = 8 * Real.sqrt 3 / 3 :=
by
  sorry

end NUMINAMATH_GPT_circumcircle_diameter_triangle_ABC_l2401_240113


namespace NUMINAMATH_GPT_root_product_is_27_l2401_240174

open Real

noncomputable def cube_root (x : ℝ) := x ^ (1 / 3 : ℝ)
noncomputable def fourth_root (x : ℝ) := x ^ (1 / 4 : ℝ)
noncomputable def square_root (x : ℝ) := x ^ (1 / 2 : ℝ)

theorem root_product_is_27 : 
  (cube_root 27) * (fourth_root 81) * (square_root 9) = 27 := 
by
  sorry

end NUMINAMATH_GPT_root_product_is_27_l2401_240174


namespace NUMINAMATH_GPT_union_of_A_and_B_l2401_240127

open Set

def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4}

theorem union_of_A_and_B : A ∪ B = {0, 1, 2, 3, 4} :=
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2401_240127


namespace NUMINAMATH_GPT_find_second_number_l2401_240125

theorem find_second_number 
  (x : ℕ)
  (h1 : (55 + x + 507 + 2 + 684 + 42) / 6 = 223)
  : x = 48 := 
by 
  sorry

end NUMINAMATH_GPT_find_second_number_l2401_240125


namespace NUMINAMATH_GPT_equation_negative_roots_iff_l2401_240150

theorem equation_negative_roots_iff (a : ℝ) :
  (∃ x < 0, 4^x - 2^(x-1) + a = 0) ↔ (-1/2 < a ∧ a ≤ 1/16) := 
sorry

end NUMINAMATH_GPT_equation_negative_roots_iff_l2401_240150


namespace NUMINAMATH_GPT_bowling_tournament_orders_l2401_240191

theorem bowling_tournament_orders :
  let choices := 2
  let number_of_games := 5
  let total_orders := choices ^ number_of_games
  total_orders = 32 :=
by
  let choices := 2
  let number_of_games := 5
  let total_orders := choices ^ number_of_games
  show total_orders = 32
  sorry

end NUMINAMATH_GPT_bowling_tournament_orders_l2401_240191


namespace NUMINAMATH_GPT_smallest_n_for_candy_l2401_240166

theorem smallest_n_for_candy (r g b n : ℕ) (h1 : 10 * r = 18 * g) (h2 : 18 * g = 20 * b) (h3 : 20 * b = 24 * n) : n = 15 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_candy_l2401_240166


namespace NUMINAMATH_GPT_number_of_combinations_l2401_240135

noncomputable def countOddNumbers (n : ℕ) : ℕ := (n + 1) / 2

noncomputable def countPrimesLessThan30 : ℕ := 9 -- {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

noncomputable def countMultiplesOfFour (n : ℕ) : ℕ := n / 4

theorem number_of_combinations : countOddNumbers 40 * countPrimesLessThan30 * countMultiplesOfFour 40 = 1800 := by
  sorry

end NUMINAMATH_GPT_number_of_combinations_l2401_240135


namespace NUMINAMATH_GPT_find_number_l2401_240129

theorem find_number (x : ℝ) : (8^3 * x^3) / 679 = 549.7025036818851 -> x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2401_240129


namespace NUMINAMATH_GPT_minimum_value_of_k_l2401_240151

theorem minimum_value_of_k (m n a k : ℕ) (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) (hk : 1 < k) (h : 5^m + 63 * n + 49 = a^k) : k = 5 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_k_l2401_240151


namespace NUMINAMATH_GPT_zeros_of_f_l2401_240163

noncomputable def f (x : ℝ) : ℝ := x^3 - 16 * x

theorem zeros_of_f :
  ∃ a b c : ℝ, (a = -4) ∧ (b = 0) ∧ (c = 4) ∧ (f a = 0) ∧ (f b = 0) ∧ (f c = 0) :=
by
  sorry

end NUMINAMATH_GPT_zeros_of_f_l2401_240163


namespace NUMINAMATH_GPT_units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared_l2401_240131

-- Define the cycle of the units digits of powers of 7
def units_digit_of_7_power (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 1  -- 7^4, 7^8, ...
  | 1 => 7  -- 7^1, 7^5, ...
  | 2 => 9  -- 7^2, 7^6, ...
  | 3 => 3  -- 7^3, 7^7, ...
  | _ => 0  -- unreachable

-- The main theorem to prove
theorem units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared : 
  units_digit_of_7_power (3 ^ (5 ^ 2)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_seven_to_the_power_of_three_to_the_power_of_five_squared_l2401_240131


namespace NUMINAMATH_GPT_range_of_a_l2401_240133

-- Define the condition p
def p (x : ℝ) : Prop := (2 * x^2 - 3 * x + 1) ≤ 0

-- Define the condition q
def q (x a : ℝ) : Prop := (x^2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0

-- Lean statement for the problem
theorem range_of_a (a : ℝ) : (¬ (∃ x, p x) → ¬ (∃ x, q x a)) → ((0 : ℝ) ≤ a ∧ a ≤ (1 / 2 : ℝ)) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l2401_240133


namespace NUMINAMATH_GPT_Mary_takes_3_children_l2401_240165

def num_children (C : ℕ) : Prop :=
  ∃ (C : ℕ), 2 + C = 5

theorem Mary_takes_3_children (C : ℕ) : num_children C → C = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Mary_takes_3_children_l2401_240165


namespace NUMINAMATH_GPT_worker_b_alone_time_l2401_240105

theorem worker_b_alone_time (A B C : ℝ) (h1 : A + B = 1 / 8)
  (h2 : A = 1 / 12) (h3 : C = 1 / 18) :
  1 / B = 24 :=
sorry

end NUMINAMATH_GPT_worker_b_alone_time_l2401_240105


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2401_240179

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 + 4 * x - 5 > 0 ↔ (x < -5 ∨ x > 1) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2401_240179


namespace NUMINAMATH_GPT_area_of_rhombus_l2401_240147

theorem area_of_rhombus (x : ℝ) :
  let d1 := 3 * x + 5
  let d2 := 2 * x + 4
  (d1 * d2) / 2 = 3 * x^2 + 11 * x + 10 :=
by
  let d1 := 3 * x + 5
  let d2 := 2 * x + 4
  have h1 : d1 = 3 * x + 5 := rfl
  have h2 : d2 = 2 * x + 4 := rfl
  simp [h1, h2]
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l2401_240147
