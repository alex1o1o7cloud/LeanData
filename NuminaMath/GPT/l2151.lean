import Mathlib

namespace NUMINAMATH_GPT_exist_polynomials_unique_polynomials_l2151_215169

-- Problem statement: the function 'f'
variable (f : ℝ → ℝ → ℝ → ℝ)

-- Condition: f(w, w, w) = 0 for all w ∈ ℝ
axiom f_ww_ww_ww (w : ℝ) : f w w w = 0

-- Statement for existence of A, B, C
theorem exist_polynomials (f : ℝ → ℝ → ℝ → ℝ)
  (hf : ∀ w : ℝ, f w w w = 0) : 
  ∃ A B C : ℝ → ℝ → ℝ → ℝ, 
  (∀ w : ℝ, A w w w + B w w w + C w w w = 0) ∧ 
  ∀ x y z : ℝ, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x) :=
sorry

-- Statement for uniqueness of A, B, C
theorem unique_polynomials (f : ℝ → ℝ → ℝ → ℝ) 
  (A B C A' B' C' : ℝ → ℝ → ℝ → ℝ)
  (hf: ∀ w : ℝ, f w w w = 0)
  (h1 : ∀ w : ℝ, A w w w + B w w w + C w w w = 0)
  (h2 : ∀ x y z : ℝ, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x))
  (h3 : ∀ w : ℝ, A' w w w + B' w w w + C' w w w = 0)
  (h4 : ∀ x y z : ℝ, f x y z = A' x y z * (x - y) + B' x y z * (y - z) + C' x y z * (z - x)) : 
  A = A' ∧ B = B' ∧ C = C' :=
sorry

end NUMINAMATH_GPT_exist_polynomials_unique_polynomials_l2151_215169


namespace NUMINAMATH_GPT_gcf_lcm_15_l2151_215133

def GCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcf_lcm_15 : 
  GCF (LCM 9 15) (LCM 10 21) = 15 :=
by 
  sorry

end NUMINAMATH_GPT_gcf_lcm_15_l2151_215133


namespace NUMINAMATH_GPT_gcd_888_1147_l2151_215105

/-- Use the Euclidean algorithm to find the greatest common divisor (GCD) of 888 and 1147. -/
theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end NUMINAMATH_GPT_gcd_888_1147_l2151_215105


namespace NUMINAMATH_GPT_limit_of_sequence_N_of_epsilon_l2151_215168

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) (h : ∀ n, a_n n = (7 * n - 1) / (n + 1)) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) ↔ a = 7 := sorry

theorem N_of_epsilon (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, N = ⌈8 / ε⌉ := sorry

end NUMINAMATH_GPT_limit_of_sequence_N_of_epsilon_l2151_215168


namespace NUMINAMATH_GPT_complex_quadratic_solution_l2151_215192

theorem complex_quadratic_solution (a b : ℝ) (h₁ : ∀ (x : ℂ), 5 * x ^ 2 - 4 * x + 20 = 0 → x = a + b * Complex.I ∨ x = a - b * Complex.I) :
 a + b ^ 2 = 394 / 25 := 
sorry

end NUMINAMATH_GPT_complex_quadratic_solution_l2151_215192


namespace NUMINAMATH_GPT_roots_of_equation_l2151_215198

theorem roots_of_equation {x : ℝ} :
  (12 * x^2 - 31 * x - 6 = 0) →
  (x = (31 + Real.sqrt 1249) / 24 ∨ x = (31 - Real.sqrt 1249) / 24) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_equation_l2151_215198


namespace NUMINAMATH_GPT_product_of_repeating_decimal_l2151_215172

noncomputable def t : ℚ := 152 / 333

theorem product_of_repeating_decimal :
  8 * t = 1216 / 333 :=
by {
  -- Placeholder for proof.
  sorry
}

end NUMINAMATH_GPT_product_of_repeating_decimal_l2151_215172


namespace NUMINAMATH_GPT_find_other_root_l2151_215111

theorem find_other_root (x : ℚ) (h: 63 * x^2 - 100 * x + 45 = 0) (hx: x = 5 / 7) : x = 1 ∨ x = 5 / 7 :=
by 
  -- Insert the proof steps here if needed.
  sorry

end NUMINAMATH_GPT_find_other_root_l2151_215111


namespace NUMINAMATH_GPT_correct_propositions_l2151_215115

-- Definitions of relations between lines and planes
variable {Line : Type}
variable {Plane : Type}

-- Definition of relationships
variable (parallel_lines : Line → Line → Prop)
variable (parallel_plane_with_plane : Plane → Plane → Prop)
variable (parallel_line_with_plane : Line → Plane → Prop)
variable (perpendicular_plane_with_plane : Plane → Plane → Prop)
variable (perpendicular_line_with_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (distinct_lines : Line → Line → Prop)
variable (distinct_planes : Plane → Plane → Prop)

-- The main theorem we are proving with the given conditions
theorem correct_propositions (m n : Line) (α β γ : Plane)
  (hmn : distinct_lines m n) (hαβ : distinct_planes α β) (hαγ : distinct_planes α γ)
  (hβγ : distinct_planes β γ) :
  -- Statement 1
  (parallel_plane_with_plane α β → parallel_plane_with_plane α γ → parallel_plane_with_plane β γ) ∧
  -- Statement 3
  (perpendicular_line_with_plane m α → parallel_line_with_plane m β → perpendicular_plane_with_plane α β) :=
by
  sorry

end NUMINAMATH_GPT_correct_propositions_l2151_215115


namespace NUMINAMATH_GPT_length_of_MN_l2151_215138

-- Define the lengths and trapezoid properties
variables (a b: ℝ)

-- Define the problem statement
theorem length_of_MN (a b: ℝ) :
  ∃ (MN: ℝ), ∀ (M N: ℝ) (is_trapezoid : True),
  (MN = 3 * a * b / (a + 2 * b)) :=
sorry

end NUMINAMATH_GPT_length_of_MN_l2151_215138


namespace NUMINAMATH_GPT_binary_to_decimal_110011_l2151_215119

theorem binary_to_decimal_110011 :
  1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 51 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_110011_l2151_215119


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2151_215170

theorem arithmetic_sequence_sum (a d x y : ℤ) 
  (h1 : a = 3) (h2 : d = 5) 
  (h3 : x = a + d) 
  (h4 : y = x + d) 
  (h5 : y = 18) 
  (h6 : x = 13) : x + y = 31 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2151_215170


namespace NUMINAMATH_GPT_max_projection_sum_l2151_215120

-- Define the given conditions
def edge_length : ℝ := 2

def projection_front_view (length : ℝ) : Prop := length = edge_length
def projection_side_view (length : ℝ) : Prop := ∃ a : ℝ, a = length
def projection_top_view (length : ℝ) : Prop := ∃ b : ℝ, b = length

-- State the theorem
theorem max_projection_sum (a b : ℝ) (ha : projection_side_view a) (hb : projection_top_view b) :
  a + b ≤ 4 := sorry

end NUMINAMATH_GPT_max_projection_sum_l2151_215120


namespace NUMINAMATH_GPT_triangle_inequality_inequality_equality_condition_l2151_215145

variable (a b c : ℝ)

-- indicating triangle inequality conditions
variable (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a)

theorem triangle_inequality_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2*b*(a - b) + b^2*c*(b - c) + c^2*a*(c - a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  a^2*b*(a - b) + b^2*c*(b - c) + c^2*a*(c - a) = 0 ↔ a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_triangle_inequality_inequality_equality_condition_l2151_215145


namespace NUMINAMATH_GPT_find_prime_triple_l2151_215116

def is_prime (n : ℕ) : Prop := n ≠ 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_prime_triple :
  ∃ (I M C : ℕ), is_prime I ∧ is_prime M ∧ is_prime C ∧ I ≤ M ∧ M ≤ C ∧ 
  I * M * C = I + M + C + 1007 ∧ (I = 2 ∧ M = 2 ∧ C = 337) :=
by
  sorry

end NUMINAMATH_GPT_find_prime_triple_l2151_215116


namespace NUMINAMATH_GPT_line_through_midpoint_l2151_215144

theorem line_through_midpoint (x y : ℝ)
  (ellipse : x^2 / 25 + y^2 / 16 = 1)
  (midpoint : P = (2, 1)) :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (A.1^2 / 25 + A.2^2 / 16 = 1) ∧
    (B.1^2 / 25 + B.2^2 / 16 = 1) ∧
    (x = 32*y - 25*x - 89) :=
sorry

end NUMINAMATH_GPT_line_through_midpoint_l2151_215144


namespace NUMINAMATH_GPT_factorize_poly1_factorize_poly2_l2151_215177

theorem factorize_poly1 (x : ℝ) : 2 * x^3 - 8 * x^2 = 2 * x^2 * (x - 4) :=
by
  sorry

theorem factorize_poly2 (x : ℝ) : x^2 - 14 * x + 49 = (x - 7) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_poly1_factorize_poly2_l2151_215177


namespace NUMINAMATH_GPT_probability_A2_l2151_215197

-- Define events and their probabilities
variable (A1 : Prop) (A2 : Prop) (B1 : Prop)
variable (P : Prop → ℝ)
variable [MeasureTheory.MeasureSpace ℝ]

-- Conditions given in the problem
axiom P_A1 : P A1 = 0.5
axiom P_B1 : P B1 = 0.5
axiom P_A2_given_A1 : P (A2 ∧ A1) / P A1 = 0.7
axiom P_A2_given_B1 : P (A2 ∧ B1) / P B1 = 0.8

-- Theorem statement to prove
theorem probability_A2 : P A2 = 0.75 :=
by
  -- Skipping the proof as per instructions
  sorry

end NUMINAMATH_GPT_probability_A2_l2151_215197


namespace NUMINAMATH_GPT_find_constant_k_eq_l2151_215152

theorem find_constant_k_eq : ∃ k : ℤ, (-x^2 - (k + 11)*x - 8 = -(x - 2)*(x - 4)) ↔ (k = -17) :=
by
  sorry

end NUMINAMATH_GPT_find_constant_k_eq_l2151_215152


namespace NUMINAMATH_GPT_six_divides_p_plus_one_l2151_215193

theorem six_divides_p_plus_one 
  (p : ℕ) 
  (prime_p : Nat.Prime p) 
  (gt_three_p : p > 3) 
  (prime_p_plus_two : Nat.Prime (p + 2)) 
  (gt_three_p_plus_two : p + 2 > 3) : 
  6 ∣ (p + 1) := 
sorry

end NUMINAMATH_GPT_six_divides_p_plus_one_l2151_215193


namespace NUMINAMATH_GPT_jade_cal_difference_l2151_215179

def Mabel_transactions : ℕ := 90

def Anthony_transactions : ℕ := Mabel_transactions + (Mabel_transactions / 10)

def Cal_transactions : ℕ := (2 * Anthony_transactions) / 3

def Jade_transactions : ℕ := 85

theorem jade_cal_difference : Jade_transactions - Cal_transactions = 19 := by
  sorry

end NUMINAMATH_GPT_jade_cal_difference_l2151_215179


namespace NUMINAMATH_GPT_quadratic_has_negative_root_condition_l2151_215102

theorem quadratic_has_negative_root_condition (a : ℝ) (h : a ≠ 0) :
  (∃ x : ℝ, ax^2 + 2*x + 1 = 0 ∧ x < 0) ↔ (a < 0 ∨ (0 < a ∧ a ≤ 1)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_negative_root_condition_l2151_215102


namespace NUMINAMATH_GPT_average_speed_is_50_l2151_215181

-- Defining the conditions
def totalDistance : ℕ := 250
def totalTime : ℕ := 5

-- Defining the average speed
def averageSpeed := totalDistance / totalTime

-- The theorem statement
theorem average_speed_is_50 : averageSpeed = 50 := sorry

end NUMINAMATH_GPT_average_speed_is_50_l2151_215181


namespace NUMINAMATH_GPT_total_apples_l2151_215173

theorem total_apples (n : ℕ) (h1 : ∀ k : ℕ, 6 * k = n ∨ 12 * k = n) (h2 : 70 ≤ n) (h3 : n ≤ 80) : 
  n = 72 ∨ n = 78 := 
sorry

end NUMINAMATH_GPT_total_apples_l2151_215173


namespace NUMINAMATH_GPT_find_the_number_l2151_215142

theorem find_the_number (x : ℝ) : (3 * x - 1 = 2 * x^2) ∧ (2 * x = (3 * x - 1) / x) → x = 1 := 
by sorry

end NUMINAMATH_GPT_find_the_number_l2151_215142


namespace NUMINAMATH_GPT_largest_possible_value_of_n_l2151_215131

open Nat

def is_prime (p : ℕ) : Prop :=
  2 ≤ p ∧ ∀ m, m ∣ p → m = 1 ∨ m = p

def largest_product : ℕ :=
  705

theorem largest_possible_value_of_n :
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧
  is_prime x ∧ is_prime y ∧
  is_prime (10 * y - x) ∧
  largest_product = x * y * (10 * y - x) :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_value_of_n_l2151_215131


namespace NUMINAMATH_GPT_num_more_green_l2151_215137

noncomputable def num_people : ℕ := 150
noncomputable def more_blue : ℕ := 90
noncomputable def both_green_blue : ℕ := 40
noncomputable def neither_green_blue : ℕ := 20

theorem num_more_green :
  (num_people + more_blue + both_green_blue + neither_green_blue) ≤ 150 →
  (more_blue - both_green_blue) + both_green_blue + neither_green_blue ≤ num_people →
  (num_people - 
  ((more_blue - both_green_blue) + both_green_blue + neither_green_blue)) + both_green_blue = 80 :=
by
    intros h1 h2
    sorry

end NUMINAMATH_GPT_num_more_green_l2151_215137


namespace NUMINAMATH_GPT_dice_composite_probability_l2151_215148

theorem dice_composite_probability (m n : ℕ) (h : Nat.gcd m n = 1) :
  (∃ m n : ℕ, (m * 36 = 29 * n) ∧ Nat.gcd m n = 1) → m + n = 65 :=
by {
  sorry
}

end NUMINAMATH_GPT_dice_composite_probability_l2151_215148


namespace NUMINAMATH_GPT_three_digit_integers_sat_f_n_eq_f_2005_l2151_215194

theorem three_digit_integers_sat_f_n_eq_f_2005 
  (f : ℕ → ℕ)
  (h1 : ∀ m n : ℕ, f (m + n) = f (f m + n))
  (h2 : f 6 = 2)
  (h3 : f 6 ≠ f 9)
  (h4 : f 6 ≠ f 12)
  (h5 : f 6 ≠ f 15)
  (h6 : f 9 ≠ f 12)
  (h7 : f 9 ≠ f 15)
  (h8 : f 12 ≠ f 15) :
  ∃! n, 100 ≤ n ∧ n ≤ 999 ∧ f n = f 2005 → n = 225 := 
  sorry

end NUMINAMATH_GPT_three_digit_integers_sat_f_n_eq_f_2005_l2151_215194


namespace NUMINAMATH_GPT_min_x_prime_sum_l2151_215191

theorem min_x_prime_sum (x y : ℕ) (h : 3 * x^2 = 5 * y^4) :
  ∃ a b c d : ℕ, x = a^b * c^d ∧ (a + b + c + d = 11) := 
by sorry

end NUMINAMATH_GPT_min_x_prime_sum_l2151_215191


namespace NUMINAMATH_GPT_highest_number_paper_l2151_215117

theorem highest_number_paper (n : ℕ) (h : 1 / (n : ℝ) = 0.01020408163265306) : n = 98 :=
sorry

end NUMINAMATH_GPT_highest_number_paper_l2151_215117


namespace NUMINAMATH_GPT_shifted_polynomial_roots_are_shifted_l2151_215151

noncomputable def original_polynomial : Polynomial ℝ := Polynomial.X ^ 3 - 5 * Polynomial.X + 7
noncomputable def shifted_polynomial : Polynomial ℝ := Polynomial.X ^ 3 + 6 * Polynomial.X ^ 2 + 7 * Polynomial.X + 5

theorem shifted_polynomial_roots_are_shifted :
  (∀ (a b c : ℝ), (original_polynomial.eval a = 0) ∧ (original_polynomial.eval b = 0) ∧ (original_polynomial.eval c = 0) 
    → (shifted_polynomial.eval (a - 2) = 0) ∧ (shifted_polynomial.eval (b - 2) = 0) ∧ (shifted_polynomial.eval (c - 2) = 0)) :=
by
  sorry

end NUMINAMATH_GPT_shifted_polynomial_roots_are_shifted_l2151_215151


namespace NUMINAMATH_GPT_cos_alpha_value_l2151_215186

theorem cos_alpha_value (α β : Real) (hα1 : 0 < α) (hα2 : α < π / 2) 
    (hβ1 : π / 2 < β) (hβ2 : β < π) (hcosβ : Real.cos β = -1/3)
    (hsin_alpha_beta : Real.sin (α + β) = 1/3) : 
    Real.cos α = 4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l2151_215186


namespace NUMINAMATH_GPT_crate_stacking_probability_l2151_215132

theorem crate_stacking_probability :
  ∃ (p q : ℕ), (p.gcd q = 1) ∧ (p : ℚ) / q = 170 / 6561 ∧ (total_height = 50) ∧ (number_of_crates = 12) ∧ (orientation_probability = 1 / 3) :=
sorry

end NUMINAMATH_GPT_crate_stacking_probability_l2151_215132


namespace NUMINAMATH_GPT_multiplier_is_five_l2151_215146

-- condition 1: n = m * (n - 4)
-- condition 2: n = 5
-- question: prove m = 5

theorem multiplier_is_five (n m : ℝ) 
  (h1 : n = m * (n - 4)) 
  (h2 : n = 5) : m = 5 := 
  sorry

end NUMINAMATH_GPT_multiplier_is_five_l2151_215146


namespace NUMINAMATH_GPT_fred_seashells_l2151_215113

-- Define the initial number of seashells Fred found.
def initial_seashells : ℕ := 47

-- Define the number of seashells Fred gave to Jessica.
def seashells_given : ℕ := 25

-- Prove that Fred now has 22 seashells.
theorem fred_seashells : initial_seashells - seashells_given = 22 :=
by
  sorry

end NUMINAMATH_GPT_fred_seashells_l2151_215113


namespace NUMINAMATH_GPT_island_count_l2151_215108

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

end NUMINAMATH_GPT_island_count_l2151_215108


namespace NUMINAMATH_GPT_jane_current_age_l2151_215155

noncomputable def JaneAge : ℕ := 34

theorem jane_current_age : 
  ∃ J : ℕ, 
    (∀ t : ℕ, t ≥ 18 ∧ t - 18 ≤ JaneAge - 18 → t ≤ JaneAge / 2) ∧
    (JaneAge - 12 = 23 - 12 * 2) ∧
    (23 = 23) →
    J = 34 := by
  sorry

end NUMINAMATH_GPT_jane_current_age_l2151_215155


namespace NUMINAMATH_GPT_books_remaining_after_second_day_l2151_215166

theorem books_remaining_after_second_day :
  let initial_books := 100
  let first_day_borrowed := 5 * 2
  let second_day_borrowed := 20
  let total_borrowed := first_day_borrowed + second_day_borrowed
  let remaining_books := initial_books - total_borrowed
  remaining_books = 70 :=
by
  sorry

end NUMINAMATH_GPT_books_remaining_after_second_day_l2151_215166


namespace NUMINAMATH_GPT_average_remaining_two_numbers_l2151_215160

theorem average_remaining_two_numbers 
    (a b c d e f : ℝ)
    (h1 : (a + b + c + d + e + f) / 6 = 3.95)
    (h2 : (a + b) / 2 = 4.4)
    (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 3.6 := 
sorry

end NUMINAMATH_GPT_average_remaining_two_numbers_l2151_215160


namespace NUMINAMATH_GPT_pq_sub_l2151_215156

-- Assuming the conditions
theorem pq_sub (p q : ℚ) 
  (h₁ : 3 / p = 4) 
  (h₂ : 3 / q = 18) : 
  p - q = 7 / 12 := 
  sorry

end NUMINAMATH_GPT_pq_sub_l2151_215156


namespace NUMINAMATH_GPT_find_angle_A_l2151_215128

theorem find_angle_A (A B : ℝ) (a b : ℝ) (h1 : b = 2 * a * Real.sin B) (h2 : a ≠ 0) :
  A = 30 ∨ A = 150 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_l2151_215128


namespace NUMINAMATH_GPT_donuts_selection_l2151_215162

theorem donuts_selection :
  (∃ g c p : ℕ, g + c + p = 6 ∧ g ≥ 1 ∧ c ≥ 1 ∧ p ≥ 1) →
  ∃ k : ℕ, k = 10 :=
by {
  -- The mathematical proof steps are omitted according to the instructions
  sorry
}

end NUMINAMATH_GPT_donuts_selection_l2151_215162


namespace NUMINAMATH_GPT_total_students_l2151_215114

theorem total_students (x : ℕ) (h1 : 3 * x + 8 = 3 * x + 5) (h2 : 5 * (x - 1) + 3 > 3 * x + 8) : x = 6 :=
sorry

end NUMINAMATH_GPT_total_students_l2151_215114


namespace NUMINAMATH_GPT_second_year_students_sampled_l2151_215140

def total_students (f s t : ℕ) : ℕ := f + s + t

def proportion_second_year (s total_stu : ℕ) : ℚ := s / total_stu

def sampled_second_year_students (p : ℚ) (n : ℕ) : ℚ := p * n

theorem second_year_students_sampled
  (f s t : ℕ) (n : ℕ)
  (h1 : f = 600)
  (h2 : s = 780)
  (h3 : t = 720)
  (h4 : n = 35) :
  sampled_second_year_students (proportion_second_year s (total_students f s t)) n = 13 := 
sorry

end NUMINAMATH_GPT_second_year_students_sampled_l2151_215140


namespace NUMINAMATH_GPT_original_number_of_movies_l2151_215101

/-- Suppose a movie buff owns movies on DVD, Blu-ray, and digital copies in a ratio of 7:2:1.
    After purchasing 5 more Blu-ray movies and 3 more digital copies, the ratio changes to 13:4:2.
    She owns movies on no other medium.
    Prove that the original number of movies in her library before the extra purchase was 390. -/
theorem original_number_of_movies (x : ℕ) (h1 : 7 * x != 0) 
  (h2 : 2 * x != 0) (h3 : x != 0)
  (h4 : 7 * x / (2 * x + 5) = 13 / 4)
  (h5 : 7 * x / (x + 3) = 13 / 2) : 10 * x = 390 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_movies_l2151_215101


namespace NUMINAMATH_GPT_shauna_min_test_score_l2151_215134

theorem shauna_min_test_score (score1 score2 score3 : ℕ) (h1 : score1 = 82) (h2 : score2 = 88) (h3 : score3 = 95) 
  (max_score : ℕ) (h4 : max_score = 100) (desired_avg : ℕ) (h5 : desired_avg = 85) :
  ∃ (score4 score5 : ℕ), score4 ≥ 75 ∧ score5 ≥ 75 ∧ (score1 + score2 + score3 + score4 + score5) / 5 = desired_avg :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_shauna_min_test_score_l2151_215134


namespace NUMINAMATH_GPT_parallelogram_ratio_l2151_215112

-- Definitions based on given conditions
def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_ratio (A : ℝ) (B : ℝ) (h : ℝ) (H1 : A = 242) (H2 : B = 11) (H3 : A = parallelogram_area B h) :
  h / B = 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_parallelogram_ratio_l2151_215112


namespace NUMINAMATH_GPT_total_weight_correct_l2151_215118

def weight_male_clothes : ℝ := 2.6
def weight_female_clothes : ℝ := 5.98
def total_weight_clothes : ℝ := weight_male_clothes + weight_female_clothes

theorem total_weight_correct : total_weight_clothes = 8.58 := by
  sorry

end NUMINAMATH_GPT_total_weight_correct_l2151_215118


namespace NUMINAMATH_GPT_sum_of_roots_l2151_215199

theorem sum_of_roots (a b c : ℚ) (h_eq : 6 * a^3 + 7 * a^2 - 12 * a = 0) (h_eq_b : 6 * b^3 + 7 * b^2 - 12 * b = 0) (h_eq_c : 6 * c^3 + 7 * c^2 - 12 * c = 0) : 
  a + b + c = -7/6 := 
by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_sum_of_roots_l2151_215199


namespace NUMINAMATH_GPT_fraction_value_l2151_215143

theorem fraction_value (a : ℕ) (h : a > 0) (h_eq : (a:ℝ) / (a + 35) = 0.7) : a = 82 :=
by
  -- Steps to prove the theorem here
  sorry

end NUMINAMATH_GPT_fraction_value_l2151_215143


namespace NUMINAMATH_GPT_count_valid_sequences_l2151_215163

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_sequence (x : ℕ → ℕ) : Prop :=
  (x 7 % 2 = 0) ∧ (∀ i < 7, (x i % 2 = 0 → x (i + 1) % 2 = 1) ∧ (x i % 2 = 1 → x (i + 1) % 2 = 0))

theorem count_valid_sequences : ∃ n, 
  n = 78125 ∧ 
  ∃ x : ℕ → ℕ, 
    (∀ i < 8, 0 ≤ x i ∧ x i ≤ 9) ∧ valid_sequence x :=
sorry

end NUMINAMATH_GPT_count_valid_sequences_l2151_215163


namespace NUMINAMATH_GPT_max_candies_l2151_215183

theorem max_candies (V M S : ℕ) (hv : V = 35) (hm : 1 ≤ M ∧ M < 35) (hs : S = 35 + M) (heven : Even S) : V + M + S = 136 :=
sorry

end NUMINAMATH_GPT_max_candies_l2151_215183


namespace NUMINAMATH_GPT_equation_solution_l2151_215125

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (3 * x + 6) / (x^2 + 5 * x - 14) = (3 - x) / (x - 2) ↔ x = 3 ∨ x = -5 :=
by 
  sorry

end NUMINAMATH_GPT_equation_solution_l2151_215125


namespace NUMINAMATH_GPT_income_growth_rate_l2151_215182

noncomputable def income_growth_eq (x : ℝ) : Prop :=
  3.2 * (1 + x)^2 = 3.7

theorem income_growth_rate :
  ∃ x : ℝ, income_growth_eq x :=
sorry

end NUMINAMATH_GPT_income_growth_rate_l2151_215182


namespace NUMINAMATH_GPT_remainder_when_divided_by_10_l2151_215100

theorem remainder_when_divided_by_10 :
  (2457 * 6291 * 9503) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_10_l2151_215100


namespace NUMINAMATH_GPT_perpendicular_lines_m_value_l2151_215126

theorem perpendicular_lines_m_value (m : ℝ) (l1_perp_l2 : (m ≠ 0) → (m * (-1 / m^2)) = -1) : m = 0 ∨ m = 1 :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_m_value_l2151_215126


namespace NUMINAMATH_GPT_train_crossing_time_l2151_215187

noncomputable def length_of_train : ℝ := 120 -- meters
noncomputable def speed_of_train_kmh : ℝ := 27 -- kilometers per hour
noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmh * (1000 / 3600) -- converted to meters per second
noncomputable def time_to_cross : ℝ := length_of_train / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross = 16 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2151_215187


namespace NUMINAMATH_GPT_parallel_lines_condition_l2151_215189

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 6 = 0 → (a - 2) * x + 3 * y + 2 * a = 0 → False) ↔ a = -1 :=
sorry

end NUMINAMATH_GPT_parallel_lines_condition_l2151_215189


namespace NUMINAMATH_GPT_smallest_area_of_right_triangle_l2151_215180

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end NUMINAMATH_GPT_smallest_area_of_right_triangle_l2151_215180


namespace NUMINAMATH_GPT_ducks_joined_l2151_215165

theorem ducks_joined (initial_ducks total_ducks ducks_joined : ℕ) 
  (h_initial: initial_ducks = 13)
  (h_total: total_ducks = 33) :
  initial_ducks + ducks_joined = total_ducks → ducks_joined = 20 :=
by
  intros h_equation
  rw [h_initial, h_total] at h_equation
  sorry

end NUMINAMATH_GPT_ducks_joined_l2151_215165


namespace NUMINAMATH_GPT_hours_per_day_l2151_215167

theorem hours_per_day (H : ℕ) : 
  (42 * 12 * H = 30 * 14 * 6) → 
  H = 5 := by
  sorry

end NUMINAMATH_GPT_hours_per_day_l2151_215167


namespace NUMINAMATH_GPT_odd_function_inequality_l2151_215107

-- Define f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_inequality
  (f : ℝ → ℝ) (h1 : is_odd_function f)
  (a b : ℝ) (h2 : f a > f b) :
  f (-a) < f (-b) :=
by
  sorry

end NUMINAMATH_GPT_odd_function_inequality_l2151_215107


namespace NUMINAMATH_GPT_perfect_square_a_value_l2151_215154

theorem perfect_square_a_value (x y a : ℝ) :
  (∃ k : ℝ, x^2 + 2 * x * y + y^2 - a * (x + y) + 25 = k^2) →
  a = 10 ∨ a = -10 :=
sorry

end NUMINAMATH_GPT_perfect_square_a_value_l2151_215154


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2151_215157

theorem hyperbola_eccentricity 
  (a b c e : ℝ)
  (h_hyperbola : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_asymptote_parallel : b = 2 * a)
  (h_c_squared : c^2 = a^2 + b^2)
  (h_e_def : e = c / a) :
  e = Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2151_215157


namespace NUMINAMATH_GPT_apples_to_pears_l2151_215159

theorem apples_to_pears :
  (3 / 4) * 12 = 9 → (2 / 3) * 6 = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_apples_to_pears_l2151_215159


namespace NUMINAMATH_GPT_fraction_equality_l2151_215153

variables {a b : ℝ}

theorem fraction_equality (h : ab * (a + b) = 1) (ha : a > 0) (hb : b > 0) : 
  a / (a^3 + a + 1) = b / (b^3 + b + 1) := 
sorry

end NUMINAMATH_GPT_fraction_equality_l2151_215153


namespace NUMINAMATH_GPT_right_triangle_angles_l2151_215171

theorem right_triangle_angles (c : ℝ) (t : ℝ) (h : t = c^2 / 8) :
  ∃(A B: ℝ), A = 90 ∧ (B = 75 ∨ B = 15) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_angles_l2151_215171


namespace NUMINAMATH_GPT_tan_neg_3780_eq_zero_l2151_215174

theorem tan_neg_3780_eq_zero : Real.tan (-3780 * Real.pi / 180) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_tan_neg_3780_eq_zero_l2151_215174


namespace NUMINAMATH_GPT_find_expression_l2151_215158

theorem find_expression (x y : ℝ) (h1 : 4 * x + y = 17) (h2 : x + 4 * y = 23) :
  17 * x^2 + 34 * x * y + 17 * y^2 = 818 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_l2151_215158


namespace NUMINAMATH_GPT_ribbon_per_box_l2151_215149

def total_ribbon : ℝ := 4.5
def remaining_ribbon : ℝ := 1
def number_of_boxes : ℕ := 5

theorem ribbon_per_box :
  (total_ribbon - remaining_ribbon) / number_of_boxes = 0.7 :=
by
  sorry

end NUMINAMATH_GPT_ribbon_per_box_l2151_215149


namespace NUMINAMATH_GPT_snickers_cost_l2151_215190

variable (S : ℝ)

def cost_of_snickers (n : ℝ) : Prop :=
  2 * n + 3 * (2 * n) = 12

theorem snickers_cost (h : cost_of_snickers S) : S = 1.50 :=
by
  sorry

end NUMINAMATH_GPT_snickers_cost_l2151_215190


namespace NUMINAMATH_GPT_number_of_white_tiles_l2151_215178

-- Definition of conditions in the problem
def side_length_large_square := 81
def area_large_square := side_length_large_square * side_length_large_square
def area_black_tiles := 81
def num_red_tiles := 154
def area_red_tiles := num_red_tiles * 4
def area_covered_by_black_and_red := area_black_tiles + area_red_tiles
def remaining_area_for_white_tiles := area_large_square - area_covered_by_black_and_red
def area_of_one_white_tile := 2
def expected_num_white_tiles := 2932

-- The theorem to prove
theorem number_of_white_tiles :
  remaining_area_for_white_tiles / area_of_one_white_tile = expected_num_white_tiles :=
by
  sorry

end NUMINAMATH_GPT_number_of_white_tiles_l2151_215178


namespace NUMINAMATH_GPT_correct_sum_of_integers_l2151_215129

theorem correct_sum_of_integers :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a - b = 3 ∧ a * b = 63 ∧ a + b = 17 :=
by 
  sorry

end NUMINAMATH_GPT_correct_sum_of_integers_l2151_215129


namespace NUMINAMATH_GPT_meaningful_sqrt_neg_x_squared_l2151_215103

theorem meaningful_sqrt_neg_x_squared (x : ℝ) : (x = 0) ↔ (-(x^2) ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_sqrt_neg_x_squared_l2151_215103


namespace NUMINAMATH_GPT_first_candidate_more_gain_l2151_215109

-- Definitions for the salaries, revenues, training costs, and bonuses
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_cost_per_month : ℕ := 1200
def training_months : ℕ := 3

def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2_percentage : ℕ := 1

-- Calculate net gains
def net_gain1 : ℕ :=
  revenue1 - salary1 - (training_cost_per_month * training_months)

def net_gain2 : ℕ :=
  revenue2 - salary2 - (salary2 * bonus2_percentage / 100)

def difference_in_gain : ℕ :=
  net_gain1 - net_gain2

-- Theorem statement
theorem first_candidate_more_gain :
  difference_in_gain = 850 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_first_candidate_more_gain_l2151_215109


namespace NUMINAMATH_GPT_bus_average_speed_excluding_stoppages_l2151_215176

theorem bus_average_speed_excluding_stoppages :
  ∀ v : ℝ, (32 / 60) * v = 40 → v = 75 :=
by
  intro v
  intro h
  sorry

end NUMINAMATH_GPT_bus_average_speed_excluding_stoppages_l2151_215176


namespace NUMINAMATH_GPT_eval_expression_l2151_215123

def a : ℕ := 4 * 5 * 6
def b : ℚ := 1/4 + 1/5 - 1/10

theorem eval_expression : a * b = 42 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l2151_215123


namespace NUMINAMATH_GPT_gcd_of_items_l2151_215161

def numPens : ℕ := 891
def numPencils : ℕ := 810
def numNotebooks : ℕ := 1080
def numErasers : ℕ := 972

theorem gcd_of_items :
  Nat.gcd (Nat.gcd (Nat.gcd numPens numPencils) numNotebooks) numErasers = 27 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_items_l2151_215161


namespace NUMINAMATH_GPT_carnival_ticket_count_l2151_215175

theorem carnival_ticket_count (ferris_wheel_rides bumper_car_rides ride_cost : ℕ) 
  (h1 : ferris_wheel_rides = 7) 
  (h2 : bumper_car_rides = 3) 
  (h3 : ride_cost = 5) : 
  ferris_wheel_rides + bumper_car_rides * ride_cost = 50 := 
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_carnival_ticket_count_l2151_215175


namespace NUMINAMATH_GPT_find_g_neg_five_l2151_215141

-- Given function and its properties
variables (g : ℝ → ℝ)

-- Conditions
axiom ax1 : ∀ (x y : ℝ), g (x - y) = g x * g y
axiom ax2 : ∀ (x : ℝ), g x ≠ 0
axiom ax3 : g 5 = 2

-- Theorem to prove
theorem find_g_neg_five : g (-5) = 8 :=
sorry

end NUMINAMATH_GPT_find_g_neg_five_l2151_215141


namespace NUMINAMATH_GPT_total_basketballs_l2151_215184

theorem total_basketballs (soccer_balls : ℕ) (soccer_balls_with_holes : ℕ) (basketballs_with_holes : ℕ) (balls_without_holes : ℕ) 
  (h1 : soccer_balls = 40) 
  (h2 : soccer_balls_with_holes = 30) 
  (h3 : basketballs_with_holes = 7) 
  (h4 : balls_without_holes = 18)
  (soccer_balls_without_holes : ℕ) 
  (basketballs_without_holes : ℕ) 
  (total_basketballs : ℕ)
  (h5 : soccer_balls_without_holes = soccer_balls - soccer_balls_with_holes)
  (h6 : basketballs_without_holes = balls_without_holes - soccer_balls_without_holes)
  (h7 : total_basketballs = basketballs_without_holes + basketballs_with_holes) : 
  total_basketballs = 15 := 
sorry

end NUMINAMATH_GPT_total_basketballs_l2151_215184


namespace NUMINAMATH_GPT_opposite_pairs_l2151_215139

theorem opposite_pairs :
  (3^2 = 9) ∧ (-3^2 = -9) ∧
  ¬ ((3^2 = 9 ∧ -2^3 = -8) ∧ 9 = -(-8)) ∧
  ¬ ((3^2 = 9 ∧ (-3)^2 = 9) ∧ 9 = -9) ∧
  ¬ ((-3^2 = -9 ∧ -(-3)^2 = -9) ∧ -9 = -(-9)) :=
by
  sorry

end NUMINAMATH_GPT_opposite_pairs_l2151_215139


namespace NUMINAMATH_GPT_percentage_stock_sold_l2151_215188

/-!
# Problem Statement
Given:
1. The cash realized on selling a certain percentage stock is Rs. 109.25.
2. The brokerage is 1/4%.
3. The cash after deducting the brokerage is Rs. 109.

Prove:
The percentage of the stock sold is 100%.
-/

noncomputable def brokerage_fee (S : ℝ) : ℝ :=
  S * 0.0025

noncomputable def selling_price (realized_cash : ℝ) (fee : ℝ) : ℝ :=
  realized_cash + fee

theorem percentage_stock_sold (S : ℝ) (realized_cash : ℝ) (cash_after_brokerage : ℝ)
  (h1 : realized_cash = 109.25)
  (h2 : cash_after_brokerage = 109)
  (h3 : brokerage_fee S = S * 0.0025) :
  S = 109.25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_stock_sold_l2151_215188


namespace NUMINAMATH_GPT_arithmetic_expression_equality_l2151_215127

theorem arithmetic_expression_equality : 18 * 36 - 27 * 18 = 162 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_equality_l2151_215127


namespace NUMINAMATH_GPT_geometric_series_sum_l2151_215106

theorem geometric_series_sum :
  ∑' (n : ℕ), (1 / 4) * (1 / 2)^n = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l2151_215106


namespace NUMINAMATH_GPT_horner_value_at_3_l2151_215150

noncomputable def horner (x : ℝ) : ℝ :=
  ((((0.5 * x + 4) * x + 0) * x - 3) * x + 1) * x - 1

theorem horner_value_at_3 : horner 3 = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_horner_value_at_3_l2151_215150


namespace NUMINAMATH_GPT_area_spot_can_reach_l2151_215124

noncomputable def area_reachable_by_spot (s : ℝ) (r : ℝ) : ℝ := 
  if s = 1 ∧ r = 3 then 6.5 * Real.pi else 0

theorem area_spot_can_reach : area_reachable_by_spot 1 3 = 6.5 * Real.pi :=
by
  -- The theorem proof should go here.
  sorry

end NUMINAMATH_GPT_area_spot_can_reach_l2151_215124


namespace NUMINAMATH_GPT_regular_tire_price_l2151_215164

theorem regular_tire_price 
  (x : ℝ) 
  (h1 : 3 * x + x / 2 = 300) 
  : x = 600 / 7 := 
sorry

end NUMINAMATH_GPT_regular_tire_price_l2151_215164


namespace NUMINAMATH_GPT_complete_even_square_diff_eqn_even_square_diff_multiple_of_four_odd_square_diff_multiple_of_eight_l2151_215135

theorem complete_even_square_diff_eqn : (10^2 - 8^2 = 4 * 9) :=
by sorry

theorem even_square_diff_multiple_of_four (n : ℕ) : (4 * (n + 1) * (n + 1) - 4 * n * n) % 4 = 0 :=
by sorry

theorem odd_square_diff_multiple_of_eight (m : ℕ) : ((2 * m + 1)^2 - (2 * m - 1)^2) % 8 = 0 :=
by sorry

end NUMINAMATH_GPT_complete_even_square_diff_eqn_even_square_diff_multiple_of_four_odd_square_diff_multiple_of_eight_l2151_215135


namespace NUMINAMATH_GPT_find_a_b_find_extreme_point_g_num_zeros_h_l2151_215185

-- (1) Proving the values of a and b
theorem find_a_b (a b : ℝ)
  (h1 : (3 + 2 * a + b = 0))
  (h2 : (3 - 2 * a + b = 0)) : 
  a = 0 ∧ b = -3 :=
sorry

-- (2) Proving the extreme points of g(x)
theorem find_extreme_point_g (x : ℝ) : 
  x = -2 :=
sorry

-- (3) Proving the number of zeros of h(x)
theorem num_zeros_h (c : ℝ) (h : -2 ≤ c ∧ c ≤ 2) :
  (|c| = 2 → ∃ y, y = 5) ∧ (|c| < 2 → ∃ y, y = 9) :=
sorry

end NUMINAMATH_GPT_find_a_b_find_extreme_point_g_num_zeros_h_l2151_215185


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l2151_215122

theorem arithmetic_expression_evaluation :
  3^2 + 4 * 2 - 6 / 3 + 7 = 22 :=
by 
  -- Use tactics to break down the arithmetic expression evaluation (steps are abstracted)
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l2151_215122


namespace NUMINAMATH_GPT_find_total_cards_l2151_215130

def numCardsInStack (n : ℕ) : Prop :=
  let cards : List ℕ := List.range' 1 (2 * n + 1)
  let pileA := cards.take n
  let pileB := cards.drop n
  let restack := List.zipWith (fun x y => [y, x]) pileA pileB |> List.join
  (restack.take 13).getLastD 0 = 13 ∧ 2 * n = 26

theorem find_total_cards : ∃ (n : ℕ), numCardsInStack n :=
sorry

end NUMINAMATH_GPT_find_total_cards_l2151_215130


namespace NUMINAMATH_GPT_incorrect_statement_B_l2151_215121

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (x - 1)^3 - a * x - b + 2

-- Condition for statement B
axiom eqn_B (a b : ℝ) : (∀ x : ℝ, f (2 - x) a b = 1 - f x a b) → a + b ≠ -1

-- The theorem to prove:
theorem incorrect_statement_B (a b : ℝ) : (∀ x : ℝ, f (2 - x) a b = 1 - f x a b) → a + b ≠ -1 := by
  exact eqn_B a b

end NUMINAMATH_GPT_incorrect_statement_B_l2151_215121


namespace NUMINAMATH_GPT_green_valley_ratio_l2151_215136

variable (j s : ℕ)

theorem green_valley_ratio (h : (3 / 4 : ℚ) * j = (1 / 2 : ℚ) * s) : s = 3 / 2 * j :=
by
  sorry

end NUMINAMATH_GPT_green_valley_ratio_l2151_215136


namespace NUMINAMATH_GPT_find_range_of_m_l2151_215104

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m * x + 1 = 0 ∧ y^2 + m * y + 1 = 0
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem find_range_of_m (m : ℝ) (h1 : ¬(p m ∧ q m)) (h2 : ¬¬p m) : m ≥ 3 ∨ m < -2 :=
by 
  sorry

end NUMINAMATH_GPT_find_range_of_m_l2151_215104


namespace NUMINAMATH_GPT_parallel_lines_condition_l2151_215196

theorem parallel_lines_condition (a : ℝ) (l : ℝ) :
  (∀ (x y : ℝ), ax + 3*y + 3 = 0 → x + (a - 2)*y + l = 0 → a = -1) ∧ (a = -1 → ∀ (x y : ℝ), (ax + 3*y + 3 = 0 ↔ x + (a - 2)*y + l = 0)) :=
sorry

end NUMINAMATH_GPT_parallel_lines_condition_l2151_215196


namespace NUMINAMATH_GPT_ray_initial_cents_l2151_215147

theorem ray_initial_cents :
  ∀ (initial_cents : ℕ), 
    (∃ (peter_cents : ℕ), 
      peter_cents = 30 ∧
      ∃ (randi_cents : ℕ),
        randi_cents = 2 * peter_cents ∧
        randi_cents = peter_cents + 60 ∧
        peter_cents + randi_cents = initial_cents
    ) →
    initial_cents = 90 := 
by
    intros initial_cents h
    obtain ⟨peter_cents, hp, ⟨randi_cents, hr1, hr2, hr3⟩⟩ := h
    sorry

end NUMINAMATH_GPT_ray_initial_cents_l2151_215147


namespace NUMINAMATH_GPT_area_of_inscribed_hexagon_in_square_is_27sqrt3_l2151_215195

noncomputable def side_length_of_triangle : ℝ := 6
noncomputable def radius_of_circle (a : ℝ) : ℝ := (a * Real.sqrt 2) / 2
noncomputable def side_length_of_square (r : ℝ) : ℝ := 2 * r
noncomputable def side_length_of_hexagon_in_square (s : ℝ) : ℝ := s / (Real.sqrt 2)
noncomputable def area_of_hexagon (side_hexagon : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * side_hexagon^2

theorem area_of_inscribed_hexagon_in_square_is_27sqrt3 :
  ∀ (a r s side_hex : ℝ), 
    a = side_length_of_triangle →
    r = radius_of_circle a →
    s = side_length_of_square r →
    side_hex = side_length_of_hexagon_in_square s →
    area_of_hexagon side_hex = 27 * Real.sqrt 3 :=
by
  intros a r s side_hex h_a h_r h_s h_side_hex
  sorry

end NUMINAMATH_GPT_area_of_inscribed_hexagon_in_square_is_27sqrt3_l2151_215195


namespace NUMINAMATH_GPT_triangle_BC_length_l2151_215110

theorem triangle_BC_length (A : ℝ) (AC : ℝ) (S : ℝ) (BC : ℝ)
  (h1 : A = 60) (h2 : AC = 16) (h3 : S = 220 * Real.sqrt 3) :
  BC = 49 :=
by
  sorry

end NUMINAMATH_GPT_triangle_BC_length_l2151_215110
