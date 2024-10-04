import Mathlib

namespace polygon_sides_l489_489599

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l489_489599


namespace percentage_greater_than_l489_489858

theorem percentage_greater_than (M N : ℝ) :
  \( \frac{100 \cdot (M - N)}{M + N} \) = \( \frac{100 \cdot (M - N)}{M + N} \) :=
by
  sorry

end percentage_greater_than_l489_489858


namespace factor_difference_of_squares_l489_489257

theorem factor_difference_of_squares (a b p q : ℝ) :
  (∃ c d : ℝ, -a ^ 2 + 9 = c ^ 2 - d ^ 2) ∧
  (¬(∃ c d : ℝ, -a ^ 2 - b ^ 2 = c ^ 2 - d ^ 2)) ∧
  (¬(∃ c d : ℝ, p ^ 2 - (-q ^ 2) = c ^ 2 - d ^ 2)) ∧
  (¬(∃ c d : ℝ, a ^ 2 - b ^ 3 = c ^ 2 - d ^ 2)) := 
  by 
  sorry

end factor_difference_of_squares_l489_489257


namespace alec_votes_l489_489713

theorem alec_votes (class_size : ℕ) (half_class_votes : ℕ) (remaining_interested : ℕ) (fraction_persuaded : ℕ) :
  class_size = 60 →
  half_class_votes = class_size / 2 →
  remaining_interested = 5 →
  fraction_persuaded = (class_size - half_class_votes - remaining_interested) / 5 →
  (3 * class_size) / 4 - (half_class_votes + fraction_persuaded) = 10 :=
by
  intros h_class_size h_half_class_votes h_remaining_interested h_fraction_persuaded
  rw h_class_size at h_half_class_votes h_remaining_interested h_fraction_persuaded
  rw [h_half_class_votes, h_remaining_interested, h_fraction_persuaded]
  sorry

end alec_votes_l489_489713


namespace find_ratio_CP_PE_l489_489456

theorem find_ratio_CP_PE 
  (A B C D E P : Type)
  (hCE : CE wrong lean parser CE -- there should be an implementation for defining lines CE and AD and their intersection point P)
  (hAD : AD)
  (hCD_DB : CD / DB = 4 / 1)
  (hAE_EB : AE / EB = 4 / 3)
  (hP_intersection : P = intersection_point (CE, AD))
  : let r := CP / PE in
  r = 7 := sorry

end find_ratio_CP_PE_l489_489456


namespace equation_of_locus_line_BC_fixed_point_l489_489906

section
  variable (k : Type) [Field k]
  
  def parabola (M : k × k) : Prop := M.2 ^ 2 = 36 * M.1
  
  def projection_on_x_axis (M N : k × k) : Prop := M.1 = N.1 ∧ N.2 = 0
  
  def vector_condition (P N M : k × k) : Prop := 
    (P.1 - N.1 = (M.1 - N.1) / 3) ∧
    (P.2 - N.2 = (M.2 - N.2) / 3)

  def locus (P : k × k) : Prop := P.2 ^ 2 = 4 * P.1

  theorem equation_of_locus 
    (M P N : k × k)
    (hM : parabola M)
    (hN : projection_on_x_axis M N)
    (hP : vector_condition P N M) :
    locus P := 
  sorry

  variable (D A B C : k × k)

  def line_through_point (p1 p2 : k × k) : Prop := ∃ k : k, p2.2 - p1.2 = k * (p2.1 - p1.1)

  def slope_eq_one (A C : k × k) : Prop := C.2 - A.2 = C.1 - A.1

  theorem line_BC_fixed_point
    (hD : D = (-3, 2))
    (hE : locus A ∧ locus B)
    (hDAB : line_through_point D A ∧ line_through_point D B)
    (hCA : slope_eq_one A C) :
    line_through_point (5, 2) B ∧ line_through_point (5, 2) C :=
  sorry
end

end equation_of_locus_line_BC_fixed_point_l489_489906


namespace impossible_network_of_triangles_l489_489536

-- Define the conditions of the problem, here we could define vertices and properties of the network
structure Vertex :=
(triangles_meeting : Nat)

def five_triangles_meeting (v : Vertex) : Prop :=
v.triangles_meeting = 5

-- The main theorem statement - it's impossible to cover the entire plane with such a network
theorem impossible_network_of_triangles :
  ¬ (∀ v : Vertex, five_triangles_meeting v) :=
sorry

end impossible_network_of_triangles_l489_489536


namespace triangle_ABC_right_triangle_l489_489452

theorem triangle_ABC_right_triangle (A B C : Type) [metric_space A] [metric_space B] [metric_space C] 
(AB BC : ℝ) (angle_A angle_B : ℝ) 
(h1 : AB = 2 * BC) 
(h2 : angle_B = 2 * angle_A) : 
(AB^2 = (BC^2 + (sqrt (3 * BC^2)))^2) :=
sorry

end triangle_ABC_right_triangle_l489_489452


namespace no_integer_solutions_l489_489800

theorem no_integer_solutions (n : ℕ) (h : 2 ≤ n) :
  ¬ ∃ x y z : ℤ, x^2 + y^2 = z^n :=
sorry

end no_integer_solutions_l489_489800


namespace find_polynomials_that_satisfy_condition_l489_489359

noncomputable def polynomial_function_condition (f : ℤ[X]) :=
  ∀ (p : ℕ) [Fact (Nat.Prime p)] (u v : ℕ), p ∣ (u * v - 1) → p ∣ (f.eval u * f.eval v - 1)

theorem find_polynomials_that_satisfy_condition :
  ∀ (f : ℤ[X]), polynomial_function_condition f →
    ∃ (a : ℤ) (n : ℕ), (a = 1 ∨ a = -1) ∧ (f = a * X^n) :=
by
  intros f h
  sorry

end find_polynomials_that_satisfy_condition_l489_489359


namespace arccos_neg_one_eq_pi_l489_489750

theorem arccos_neg_one_eq_pi : real.arccos (-1) = real.pi :=
by sorry

end arccos_neg_one_eq_pi_l489_489750


namespace sum_of_coefficients_is_2_l489_489268

noncomputable def polynomial_expansion_condition (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) :=
  (x^2 + 1) * (x - 2)^9 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + 
                          a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 + a_8 * (x - 1)^8 + 
                          a_9 * (x - 1)^9 + a_10 * (x - 1)^10 + a_11 * (x - 1)^11

theorem sum_of_coefficients_is_2 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) :
  polynomial_expansion_condition 1 a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 →
  polynomial_expansion_condition 2 a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 2 :=
by sorry

end sum_of_coefficients_is_2_l489_489268


namespace Iris_pairs_of_pants_l489_489912

theorem Iris_pairs_of_pants (jacket_cost short_cost pant_cost total_spent n_jackets n_shorts n_pants : ℕ) :
  (jacket_cost = 10) →
  (short_cost = 6) →
  (pant_cost = 12) →
  (total_spent = 90) →
  (n_jackets = 3) →
  (n_shorts = 2) →
  (n_jackets * jacket_cost + n_shorts * short_cost + n_pants * pant_cost = total_spent) →
  (n_pants = 4) := 
by
  intros h_jacket_cost h_short_cost h_pant_cost h_total_spent h_n_jackets h_n_shorts h_eq
  sorry

end Iris_pairs_of_pants_l489_489912


namespace tan_beta_expression_max_tan_beta_l489_489817

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π/2)
variable (hβ : 0 < β ∧ β < π/2)
variable (h : sin β / sin α = cos (α + β))

theorem tan_beta_expression : 
  tan β = sin α * cos α / (1 + sin α ^ 2) :=
by sorry

theorem max_tan_beta : 
  ∃ (α : ℝ), 0 < α ∧ α < π/2 ∧ tan β = sqrt 2 / 4 :=
by sorry

end tan_beta_expression_max_tan_beta_l489_489817


namespace line_intersects_y_axis_at_five_thirds_l489_489289

-- Define the points
def point1 := (2 : ℝ, 3 : ℝ)
def point2 := (5 : ℝ, 5 : ℝ)

-- Define what it means for a point to be on the line between two points
def on_line (p1 p2 p : ℝ × ℝ) : Prop :=
  ∃ m b : ℝ, p2.2 = m * p2.1 + b ∧ p1.2 = m * p1.1 + b ∧ p.2 = m * p.1 + b

-- Define the y-axis intersection point
def y_axis_intersection (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  let m := (p2.2 - p1.2) / (p2.1 - p1.1) in
  let b := p1.2 - m * p1.1 in
  (0, b)

-- Statement of the theorem
theorem line_intersects_y_axis_at_five_thirds :
  y_axis_intersection point1 point2 = (0, 5/3) :=
by
  sorry

end line_intersects_y_axis_at_five_thirds_l489_489289


namespace part_1_part_2_part_3_l489_489488

noncomputable theory

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (b : ℕ → ℕ)

-- Conditions:
axiom condition_1 : ∀ n : ℕ, a (n + 3) = a n + 3
axiom condition_2 : a 1 = 1
axiom condition_3 : ∀ n : ℕ, S n = (finset.range (n + 1)).sum a
axiom condition_4 : S 15 = 15 * a 8
axiom condition_5 : ∀ n : ℕ, b n = n + a * c^(S n + a)
axiom condition_6 : c > 0
axiom condition_7 : c ≠ 1

-- Goals:
def T (n : ℕ) := (finset.range n).sum (λ i, a (3 * i - 2))

theorem part_1 : T n = (3*n^2 - n) / 2 := 
sorry

theorem part_2 : ∀ n : ℕ, a (n + 1) - a n = 1 := 
sorry

theorem part_3 : ∀ n : ℕ, a n = n ∧ a = 0 :=
sorry

end part_1_part_2_part_3_l489_489488


namespace matrix_power_problem_l489_489147

open Matrix
open_locale matrix big_operators

def B : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 4], ![0, 2]]

theorem matrix_power_problem :
  B^15 - 3 • (B^14) = ![![0, 4], ![0, -1]] :=
by
  sorry

end matrix_power_problem_l489_489147


namespace three_digit_numbers_sum_seven_l489_489015

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489015


namespace evaluate_expression_l489_489440

theorem evaluate_expression (a b x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
    (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  sorry

end evaluate_expression_l489_489440


namespace perfect_square_or_subset_product_perfect_square_l489_489410

/-- 
Given a set of 1986 natural numbers where their product has exactly 1985 distinct prime factors,
prove that either one of these numbers is a perfect square or the product of some subset of these numbers is a perfect square.
-/
theorem perfect_square_or_subset_product_perfect_square
  (A : Finset ℕ)
  (h_card : A.card = 1986)
  (h_factors : (A.fold (*) 1).factors.toFinset.card = 1985) :
  (∃ n ∈ A, ∃ k, n = k^2) ∨ (∃ B ⊆ A, ∃ k, (B.fold (*) 1) = k^2) :=
sorry

end perfect_square_or_subset_product_perfect_square_l489_489410


namespace non_black_cows_l489_489994

theorem non_black_cows (total_cows black_increment : ℕ) (h : total_cows = 18) (half_black_add : black_increment = 5) :
  (total_cows - (total_cows / 2 + black_increment)) = 4 :=
by
  rw [h, half_black_add]
  -- Each step from solution can be left as comments for better understanding
  -- Half of the cows: 18 / 2 = 9
  -- Number of black cows: 9 + 5 = 14
  -- Non-black cows: 18 - 14 = 4
  sorry

end non_black_cows_l489_489994


namespace cube_diagonal_length_l489_489686

theorem cube_diagonal_length
  (side_length : ℝ)
  (h_side_length : side_length = 15) :
  ∃ d : ℝ, d = side_length * Real.sqrt 3 :=
by
  sorry

end cube_diagonal_length_l489_489686


namespace angle_ABC_of_unused_sector_l489_489375

theorem angle_ABC_of_unused_sector (r₁ r₂ : ℝ) (V : ℝ) (r_cone : ℝ) 
  (cone_volume : V = 432 * real.pi) (r_cone_length : r_cone = 12):
  let height_cone := (3 * V) / (real.pi * r_cone^2),
      slant_height := real.sqrt (r_cone^2 + height_cone^2),
      original_radius := slant_height,
      angle_AC := 360 * (2 * real.pi * r_cone) / (2 * real.pi * original_radius),
      angle_ABC := 360 - angle_AC
  in  angle_ABC = 72 :=
by
  sorry

end angle_ABC_of_unused_sector_l489_489375


namespace problem_l489_489438

theorem problem (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + (1/r^4) = 7 := 
by
  sorry

end problem_l489_489438


namespace prime_if_lcm_improper_l489_489823

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d ∣ n ∧ d ≠ 0 ∧ d ≠ n)

def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

theorem prime_if_lcm_improper (n : ℕ) (h1 : 2 ≤ n) (h2 : ProperDivisorsLCMNeqN : lcm_list (proper_divisors n) ≠ n) : is_prime n :=
  sorry

end prime_if_lcm_improper_l489_489823


namespace find_x_value_l489_489624

-- Definitions based on the conditions
def z : ℝ := 90
def y : ℝ := z / 3
def x : ℝ := y / 4

-- The goal is to prove that x = 7.5 under the given conditions
theorem find_x_value : x = 7.5 := by
  sorry

end find_x_value_l489_489624


namespace probability_of_four_primes_l489_489244

-- Define the probability of rolling a prime number
def prime_probability : ℚ := 4 / 8  -- There are 4 prime numbers (2, 3, 5, 7) on an 8-sided die

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Use the binomial probability formula to define the probability of getting exactly 4 prime numbers in 6 rolls
def probability_four_primes_in_six_rolls : ℚ :=
  (binom 6 4) * (prime_probability ^ 4) * ((1 - prime_probability) ^ (6 - 4))

theorem probability_of_four_primes (h : probability_four_primes_in_six_rolls = 15 / 64) : Prop :=
  h

end probability_of_four_primes_l489_489244


namespace error_percentage_calculation_l489_489694

theorem error_percentage_calculation (y : ℝ) (hy : 0 < y) : 
    abs_fraction_percentage (a : ℝ) (b : ℝ) (ha : abs a = a) (hb : abs b = b) :
abs (a - b) / b * 100 = (|4 * y + 25 | / (5 * y + 25))* 100 :=
 sorry
 
end error_percentage_calculation_l489_489694


namespace form_of_reasoning_is_wrong_l489_489632

-- Let's define the conditions
def some_rat_nums_are_proper_fractions : Prop :=
  ∃ q : ℚ, (q.num : ℤ) ≠ q.den ∧ (q.den : ℤ) ≠ 1 ∧ q.den ≠ 0

def integers_are_rational_numbers : Prop :=
  ∀ n : ℤ, ∃ q : ℚ, q = n

-- The major premise of the syllogism
def major_premise := some_rat_nums_are_proper_fractions

-- The minor premise of the syllogism
def minor_premise := integers_are_rational_numbers

-- The conclusion of the syllogism
def conclusion := ∀ n : ℤ, ∃ q : ℚ, (q.num : ℤ) ≠ q.den ∧ (q.den : ℤ) ≠ 1 ∧ q.den ≠ 0

-- We need to prove that the form of reasoning is wrong
theorem form_of_reasoning_is_wrong (H1 : major_premise) (H2 : minor_premise) : ¬ conclusion :=
by
  sorry -- proof to be filled in

end form_of_reasoning_is_wrong_l489_489632


namespace smallest_value_of_magnitude_squared_l489_489177

theorem smallest_value_of_magnitude_squared (z : ℂ) (h1 : 0 < z.re) (h2 : 2 * complex.abs (complex.sin (2 * (complex.arg z))) = 12 / 13) :
  ∃ d : ℝ, d^2 = 36 / 13 :=
by
  sorry

end smallest_value_of_magnitude_squared_l489_489177


namespace active_probability_correct_not_active_moderate_probability_correct_chi_square_significance_l489_489283

section survey_analysis

def total_students : ℕ := 50
def students_active : ℕ := 22
def students_not_active_moderate : ℕ := 20

def table_A : ℕ := 17
def table_B : ℕ := 8
def table_C : ℕ := 5
def table_D : ℕ := 20

def active_prob : ℚ := students_active / total_students
def not_active_moderate_prob : ℚ := students_not_active_moderate / total_students

def chi_square (n A B C D : ℕ) : ℚ :=
  (n * (A * D - B * C) ^ 2) / ((A + B) * (C + D) * (A + C) * (B + D))

def chi_square_val : ℚ := chi_square total_students table_A table_B table_C table_D

def critical_value_0_001 : ℚ := 10.8

-- Statements to prove
theorem active_probability_correct : active_prob = (22 : ℚ) / 50 := by
  sorry
  
theorem not_active_moderate_probability_correct : not_active_moderate_prob = (20 : ℚ) / 50 := by
  sorry
  
theorem chi_square_significance : chi_square_val > critical_value_0_001 := by
  sorry

end survey_analysis

end active_probability_correct_not_active_moderate_probability_correct_chi_square_significance_l489_489283


namespace equivalent_problem_l489_489486

-- Define the conditions and objects involved
structure Point (α : Type _) :=
(x : α) (y : α)

def A : Point ℝ := ⟨-1, 0⟩
def B : Point ℝ := ⟨2, 0⟩

def dist (p1 p2 : Point ℝ) : ℝ := 
real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define what it means for the point M to satisfy the given condition
def C_condition (M : Point ℝ) : Prop :=
dist M A / dist M B = 1/2

-- Define the equation of the trajectory C
def curve_C (M : Point ℝ) : Prop :=
(M.x + 2)^2 + M.y^2 = 4

-- Define the fixed point P and the line segment condition
def P (a : ℝ) : Point ℝ := ⟨0, a⟩
def max_dist_MP (a : ℝ) (M P : Point ℝ) : Prop :=
a > 0 ∧ real.sqrt (dist M P ^ 2) + 2 = 2 + 2 * real.sqrt 2

-- Define the tangent line condition
def tangent_line (a k : ℝ) (M : Point ℝ) : Prop :=
abs (-2 * k + a) / real.sqrt (1 + k^2) = 2

-- Equivalent proof that (1) proves curve C and (2) finds the tangent lines
theorem equivalent_problem (M : Point ℝ) (a : ℝ) (k : ℝ) :
  (C_condition M → curve_C M) ∧ (max_dist_MP a M (P a) → (tangent_line a k M ∨ a = 2 ∧ (M.x = 0 ∨ M.y = 2))) :=
sorry

end equivalent_problem_l489_489486


namespace main_inequality_l489_489955

variables {n : ℕ} 
variables {f g : ℝ → ℝ → ℝ} 
variables {a b : ℝ}

-- Conditions
axiom pos_elements : ∀ (i : ℕ) (Hi : 1 ≤ i ∧ i ≤ n), 0 < a i ∧ 0 < b i
axiom func_conditions : 
  (∀ (a b : ℝ), f a b * g a b = a^2 * b^2) ∧
  (∀ (k a b : ℝ), k > 0 → f (k*a) (k*b) = k^2 * f a b) ∧
  (∀ (a b : ℝ), (b * f a 1 / (a * f b 1) + a * f b 1 / (b * f a 1)) ≤ (a / b + b / a))

-- Statement to prove
theorem main_inequality 
    (a b : ℕ → ℝ) 
    (f g : ℝ → ℝ → ℝ) 
    (f_cond : ∀ (a b : ℝ), f a b * g a b = a^2 * b^2) 
    (g_cond : ∀ (k a b : ℝ), k > 0 → f (k*a) (k*b) = k^2 * f a b)  
    (cond_ineq: ∀ (a b : ℝ), (b * f a 1 / (a * f b 1) + a * f b 1 / (b * f a 1)) ≤ (a / b + b / a))
    (pos_ab : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ n → 0 < a i ∧ 0 < b i) :
    (∑ i in finset.range n, a i * b i)^2 
    ≤ (∑ i in finset.range n, f (a i) (b i)) * (∑ i in finset.range n, g (a i) (b i)) ∧ 
    (∑ i in finset.range n, f (a i) (b i)) * (∑ i in finset.range n, g (a i) (b i)) 
    ≤ (∑ i in finset.range n, (a i)^2) * (∑ i in finset.range n, (b i)^2) :=
sorry

end main_inequality_l489_489955


namespace valid_first_coupon_days_l489_489556

-- Defining the days of the week
inductive Weekday
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq, Repr

open Weekday

-- Function to check if a day is a weekend day
def isWeekend (d : Weekday) : Bool :=
  d = Saturday ∨ d = Sunday

-- Function to calculate the day of the week n days after a given start day
def addDays (start : Weekday) (n : Nat) : Weekday :=
  let days := [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]
  let startIndex := days.indexOf start
  days.get! ((startIndex + n) % 7)

-- Function to check if a series of coupon redemptions will fall on weekends
def validStartDay (start : Weekday) : Bool :=
  let redemptions := List.map (fun i => addDays start (i * 15)) [0, 1, 2, 3, 4]
  !redemptions.any isWeekend

theorem valid_first_coupon_days : 
  ∃ d : Weekday, 
    (d = Monday ∨ d = Tuesday ∨ d = Wednesday) ∧ 
    validStartDay d :=
by
  cases Weekday.decidableEq
  use Monday
  split
  { left, refl }
  { unfold validStartDay, simp [isWeekend, addDays, list.get!] }
  sorry

end valid_first_coupon_days_l489_489556


namespace disjoint_subsets_exist_l489_489662

theorem disjoint_subsets_exist (n : ℕ) (h : 0 < n) 
  (A : Fin (n + 1) → Set (Fin n)) (hA : ∀ i : Fin (n + 1), A i ≠ ∅) :
  ∃ (I J : Finset (Fin (n + 1))), I ≠ ∅ ∧ J ≠ ∅ ∧ Disjoint I J ∧ 
    (⋃ i ∈ I, A i) = (⋃ j ∈ J, A j) :=
sorry

end disjoint_subsets_exist_l489_489662


namespace ratio_of_areas_l489_489552

-- Define the sides and conditions
variables (s : ℝ) (P B : ℝ)
-- Given conditions
def AP_eq_3PB : Prop := P = 3 * B
def summing_pieces : Prop := P + B = s

-- Define the areas
def area_ABCD : ℝ := s * s
def area_PQRV : ℝ := (s / 4) * (s / 4)

-- The final ratio condition
def ratio_PQRV_to_ABCD : Prop := area_PQRV s = area_ABCD s / 16

-- The theorem we need to prove
theorem ratio_of_areas (h1 : AP_eq_3PB P B) (h2 : summing_pieces P B s) : ratio_PQRV_to_ABCD s :=
by
    unfold AP_eq_3PB at h1
    unfold summing_pieces at h2
    unfold ratio_PQRV_to_ABCD
    sorry

end ratio_of_areas_l489_489552


namespace calculate_markup_percentage_l489_489306

noncomputable def cost_price : ℝ := 225
noncomputable def profit_percentage : ℝ := 0.25
noncomputable def discount1_percentage : ℝ := 0.10
noncomputable def discount2_percentage : ℝ := 0.15
noncomputable def selling_price : ℝ := cost_price * (1 + profit_percentage)
noncomputable def markup_percentage : ℝ := 63.54

theorem calculate_markup_percentage :
  let marked_price := selling_price / ((1 - discount1_percentage) * (1 - discount2_percentage))
  let calculated_markup_percentage := ((marked_price - cost_price) / cost_price) * 100
  abs (calculated_markup_percentage - markup_percentage) < 0.01 :=
sorry

end calculate_markup_percentage_l489_489306


namespace exists_polynomial_S_l489_489057

noncomputable def P : Polynomial ℝ := sorry
noncomputable def Q : Polynomial ℝ := sorry

theorem exists_polynomial_S (R : Polynomial ℝ × Polynomial ℝ → Polynomial ℝ) (h : ∀ x y, P.eval x - P.eval y = R (x, y) * (Q.eval x - Q.eval y)) :
  ∃ S : Polynomial ℝ, ∀ x, P.eval x = (S.eval (Q.eval x)) :=
sorry

end exists_polynomial_S_l489_489057


namespace num_three_digit_sums7_l489_489003

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l489_489003


namespace angle_BDC_half_A_l489_489468

/-- In a right triangle ABC where angle C is 90 degrees, the bisectors of the exterior angles at B and C meet at point D. Prove that angle BDC is A / 2. -/
theorem angle_BDC_half_A (A B C : ℝ) (hC : C = 90) (h_triangle : A + B + C = 180) :
  ∃ D, ∠BDC = A / 2 :=
by
  -- Formal representation of the conditions
  rcases hC with rfl,
  sorry

end angle_BDC_half_A_l489_489468


namespace isosceles_triangle_base_angle_l489_489471

theorem isosceles_triangle_base_angle (a b c : ℝ) (h : a + b + c = 180) (h_isosceles : b = c) (h_angle_a : a = 120) : b = 30 := 
by
  sorry

end isosceles_triangle_base_angle_l489_489471


namespace polygon_sides_l489_489606

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489606


namespace part1_part2_l489_489852

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := k * x - real.log (4^x + 1) / real.log 4

theorem part1 (h : ∀ x, f x k = f (-x) k) : k = 1 / 2 := by
  sorry -- Proof of part1

noncomputable def g (x : ℝ) (m : ℝ) : ℝ :=
  4^((1 / 2) * x - f x (1 / 2)) - m * 4^(1 + (1 / 2) * x) - 1

theorem part2 (m : ℝ) :
  ∀ x ∈ set.Icc 0 (real.log 3 / real.log 2),
    g x m ≤ if m < 1 then 9 - 12 * m else 1 - 4 * m := by
  sorry -- Proof of part2

end part1_part2_l489_489852


namespace farthest_point_from_origin_l489_489647

noncomputable def distance_from_origin (p : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (p.1^2 + p.2^2 + p.3^2)

theorem farthest_point_from_origin :
  let
    p1 := (0, 5, 1)
    p2 := (1, 2, 3)
    p3 := (4, 0, -3)
    p4 := (6, 1, 0)
    p5 := (-1, -2, 2)
  in distance_from_origin p4 > distance_from_origin p1 ∧
     distance_from_origin p4 > distance_from_origin p2 ∧
     distance_from_origin p4 > distance_from_origin p3 ∧
     distance_from_origin p4 > distance_from_origin p5 :=
by
  sorry

end farthest_point_from_origin_l489_489647


namespace Jake_has_fewer_peaches_l489_489199

def Steven_peaches := 14
def Jill_peaches := 5
def Jake_peaches := Jill_peaches + 3

theorem Jake_has_fewer_peaches : Steven_peaches - Jake_peaches = 6 :=
by
  sorry

end Jake_has_fewer_peaches_l489_489199


namespace sum_of_series_l489_489397

theorem sum_of_series (a : ℕ → ℝ) :
  ((x + 1)^2 * (x + 2)^2016 = ∑ i in range 2019, a i * (x + 2)^i) →
  (a_1 / 2 + a_2 / 2^2 + a_3 / 2^3 + ... + a_2018 / 2^2018 = (1 / 2)^2018) := 
sorry

end sum_of_series_l489_489397


namespace sides_of_polygon_l489_489614

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l489_489614


namespace polygon_sides_l489_489604

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l489_489604


namespace probability_log_condition_correct_l489_489996

noncomputable def probability_log_condition
  (x y : ℕ)
  (hx : x ∈ {1, 2, 3, 4, 5, 6})
  (hy : y ∈ {1, 2, 3, 4, 5, 6})
  (h : (∀ x y, log (2 * x) y = 1 → y = 2 * x)) : ℚ :=
let possible_outcomes := { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} } in
let favorable_outcomes := { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ log (2 * x) y = 1 } in
favorable_outcomes.card / possible_outcomes.card

theorem probability_log_condition_correct
  : probability_log_condition = 1 / 12 :=
sorry

end probability_log_condition_correct_l489_489996


namespace matrix_power_identity_l489_489142

-- Define the matrix B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![0, 2]]

-- Define the identity matrix I
def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

-- Prove that B^15 - 3 * B^14 is equal to the given matrix
theorem matrix_power_identity :
  B ^ 15 - 3 • (B ^ 14) = ![![0, 4], ![0, -1]] :=
by
  -- Sorry is used here so the Lean code is syntactically correct
  sorry

end matrix_power_identity_l489_489142


namespace find_n_l489_489519

theorem find_n (n : ℤ) (h₁ : 50 ≤ n ∧ n ≤ 120)
               (h₂ : n % 8 = 0)
               (h₃ : n % 12 = 4)
               (h₄ : n % 7 = 4) : 
  n = 88 :=
sorry

end find_n_l489_489519


namespace infinite_pseudoprimes_l489_489350

theorem infinite_pseudoprimes :
  ∃ S : Set ℕ, S.infinite ∧ ∀ n ∈ S, ∃ a : ℕ, a^(n-1) ≡ 1 [MOD n] ∧ n = p * q ∧ Prime p ∧ Prime q ∧ p ≠ q :=
by
  sorry

end infinite_pseudoprimes_l489_489350


namespace part1_part2_l489_489051

variable {a : ℕ → ℝ}
variable (A : ∀ n : ℕ, 0 < a n)
variable (B : ∀ n : ℕ, a n < a (n + 1))
variable (C : ∀ k : ℕ, ∀ n : ℕ, n > k → a n > a k)

noncomputable def S (k : ℕ) : ℝ := ∑ i in Finset.range k, a i / a (i + 1)

theorem part1 : ∃ k₀ : ℕ, ∀ k ≥ k₀, S k < k - 1 :=
sorry

theorem part2 : ∃ k₁ : ℕ, ∀ k ≥ k₁, S k < k - 1985 :=
sorry

end part1_part2_l489_489051


namespace parabola_inequality_l489_489055

theorem parabola_inequality {y1 y2 : ℝ} :
  (∀ x1 x2 : ℝ, x1 = -5 → x2 = 2 →
  y1 = x1^2 + 2 * x1 + 3 ∧ y2 = x2^2 + 2 * x2 + 3) → (y1 > y2) :=
by
  intros h
  sorry

end parabola_inequality_l489_489055


namespace determine_condition_l489_489433

theorem determine_condition (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) 
    (h_eq : (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) : 
    b + c = 12 :=
by
  sorry

end determine_condition_l489_489433


namespace proj_CD_on_AC_range_PA_dot_PB_l489_489908

open Real EuclideanGeometry

-- Define the conditions
variables (A B C D P : Point) (h1 : ∠A = π / 2) (hAB : dist A B = 6) (hAC : dist A C = 4)
          (hD : midpoint₀ A B D)

-- Define first proof statement
theorem proj_CD_on_AC (hA : ∠A = π / 2) (hAB : dist A B = 6) (hAC : dist A C = 4)
  (hD : midpoint₀ A B D) : proj (D - C) (C - A) = -(C - A) :=
by sorry

-- Define second proof statement with P specifically
variables (hCP : dist C P = 1)

theorem range_PA_dot_PB (A B C D P : Point) (hA : ∠A = π / 2) (hAB : dist A B = 6) 
  (hAC : dist A C = 4) (hD : midpoint₀ A B D) (hCP : dist C P = 1) :
  ∃ (x : ℝ), 7 ≤ x ∧ x ≤ 27 ∧ x = (A - P) • (B - P) :=
by sorry

end proj_CD_on_AC_range_PA_dot_PB_l489_489908


namespace prevent_red_2x2_square_l489_489734

namespace ChessGame

-- Define the dimensions and cell structure of the chessboard
def Chessboard := Array (Array (Option Bool))

-- Initial state of the chessboard
def initialChessboard : Chessboard := 
  Array.replicate 8 (Array.replicate 8 none)

-- Define a function that represents a move taken by a player
def make_move (board : Chessboard) (pos : Nat × Nat) (color : Bool) : Chessboard := 
  let (r, c) := pos
  board.set! r (board.get! r).set! c (some color)

-- Game strategy for Dinh-Khanh
/-- Prove that Dinh-Khanh can always prevent Chloé from completely coloring any 2x2 square red-/
theorem prevent_red_2x2_square (board : Chessboard := initialChessboard) : 
  ∀ (moves : List (Nat × Nat)), ∃ (strategy : List (Nat × Nat)), 
  ∀ (pos : Nat × Nat), ¬(∃ (r c : Nat), 
  r < 8 ∧ c < 8 ∧
  (board.get! r).get! c = some true ∧ 
  (board.get! (r + 1)).get! c = some true ∧ 
  (board.get! r).get! (c + 1) = some true ∧ 
  (board.get! (r + 1)).get! (c + 1) = some true) := 
by sorry

end ChessGame

end prevent_red_2x2_square_l489_489734


namespace bug_crawl_distance_l489_489679

-- Define the conditions
def initial_position : ℤ := -2
def first_move : ℤ := -6
def second_move : ℤ := 5

-- Define the absolute difference function (distance on a number line)
def abs_diff (a b : ℤ) : ℤ :=
  abs (b - a)

-- Define the total distance crawled function
def total_distance (p1 p2 p3 : ℤ) : ℤ :=
  abs_diff p1 p2 + abs_diff p2 p3

-- Prove that total distance starting at -2, moving to -6, and then to 5 is 15 units
theorem bug_crawl_distance : total_distance initial_position first_move second_move = 15 := by
  sorry

end bug_crawl_distance_l489_489679


namespace assistant_increases_output_by_100_percent_l489_489935

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l489_489935


namespace sum_of_series_equality_l489_489326

theorem sum_of_series_equality :
  (∑ (a b c : ℕ) (ha : 0 < a) (hb : a < b) (hc : b < c), (1:ℚ) / (3^a * 4^b * 6^c)) = 1 / 1265 := 
sorry

end sum_of_series_equality_l489_489326


namespace skips_in_one_meter_l489_489554

variable (p q r s t u : ℕ)

theorem skips_in_one_meter (h1 : p * s * u = q * r * t) : 1 = (p * r * t) / (u * s * q) := by
  sorry

end skips_in_one_meter_l489_489554


namespace geom_seq_sum_half_l489_489126

theorem geom_seq_sum_half (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_sum : ∃ L, L = ∑' n, a n ∧ L = 1 / 2) (h_abs : |q| < 1) :
  a 0 ∈ (Set.Ioo 0 (1 / 2)) ∪ (Set.Ioo (1 / 2) 1) :=
sorry

end geom_seq_sum_half_l489_489126


namespace scooped_water_amount_l489_489709

variables (x : ℝ)

def initial_water_amount : ℝ := 10
def total_amount : ℝ := initial_water_amount
def alcohol_concentration : ℝ := 0.75

theorem scooped_water_amount (h : x / total_amount = alcohol_concentration) : x = 7.5 :=
by sorry

end scooped_water_amount_l489_489709


namespace mimi_spending_adidas_l489_489526

theorem mimi_spending_adidas
  (total_spending : ℤ)
  (nike_to_adidas_ratio : ℤ)
  (adidas_to_skechers_ratio : ℤ)
  (clothes_spending : ℤ)
  (eq1 : total_spending = 8000)
  (eq2 : nike_to_adidas_ratio = 3)
  (eq3 : adidas_to_skechers_ratio = 5)
  (eq4 : clothes_spending = 2600) :
  ∃ A : ℤ, A + nike_to_adidas_ratio * A + adidas_to_skechers_ratio * A + clothes_spending = total_spending ∧ A = 600 := by
  sorry

end mimi_spending_adidas_l489_489526


namespace a2_and_a3_values_geometric_sequence_and_general_term_sum_formula_l489_489129

open Nat

def seq_a : ℕ → ℕ
| 1 := 3
| (n+1) := 2 * seq_a n + n + 1 - 2 -- Note that n+1 means the next term, in Lean n starts from 0 so we use (n+1) instead of directly n >= 2

-- Problem 1
theorem a2_and_a3_values :
  seq_a 2 = 6 ∧ seq_a 3 = 13 :=
  sorry

-- Problem 2
theorem geometric_sequence_and_general_term :
  (∀ n, seq_a n + n = 4 * 2^(n-1)) ∧ (∀ n, seq_a n = 2^(n+1) - n) :=
  sorry

-- Problem 3
def sum_first_n_terms (n : ℕ) : ℕ :=
  ∑ i in range n, seq_a (i+1)

theorem sum_formula (n : ℕ) :
  sum_first_n_terms n = 2^(n+2) - (n^2 + n + 8) / 2 :=
  sorry

end a2_and_a3_values_geometric_sequence_and_general_term_sum_formula_l489_489129


namespace tangent_series_equal_l489_489986

theorem tangent_series_equal (n : ℕ) (α : ℝ) :
  (∑ k in finset.range (n - 1), (Real.tan (k * α) * Real.tan ((k + 1) * α))) = (Real.tan (n * α) / Real.tan α) - n := 
by
  sorry

end tangent_series_equal_l489_489986


namespace matrix_power_problem_l489_489148

open Matrix
open_locale matrix big_operators

def B : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 4], ![0, 2]]

theorem matrix_power_problem :
  B^15 - 3 • (B^14) = ![![0, 4], ![0, -1]] :=
by
  sorry

end matrix_power_problem_l489_489148


namespace function_sum_even_l489_489511

variable (f : ℝ → ℝ)

theorem function_sum_even : ∀ x : ℝ, f(x) + f(-x) = f(-x) + f(x) :=
by 
  intros x
  sorry

end function_sum_even_l489_489511


namespace variance_of_given_data_is_2_l489_489625

-- Define the data set
def data_set : List ℕ := [198, 199, 200, 201, 202]

-- Define the mean function for a given data set
noncomputable def mean (data : List ℕ) : ℝ :=
  (data.sum : ℝ) / data.length

-- Define the variance function for a given data set
noncomputable def variance (data : List ℕ) : ℝ :=
  let μ := mean data
  (data.map (λ x => (x : ℝ) - μ) |>.map (λ x => x^2)).sum / data.length

-- Proposition that the variance of the given data set is 2
theorem variance_of_given_data_is_2 : variance data_set = 2 := by
  sorry

end variance_of_given_data_is_2_l489_489625


namespace positive_difference_of_squares_l489_489623

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 18) : a^2 - b^2 = 1080 :=
by
  sorry

end positive_difference_of_squares_l489_489623


namespace units_digit_of_four_consecutive_l489_489645

noncomputable def units_digit (n : ℕ) : ℕ :=
  let prod := n * (n + 1) * (n + 2) * (n + 3) in
  prod % 10

theorem units_digit_of_four_consecutive (n : ℕ) (h : ∃ k : ℕ, (n = 3^k) ∨ (n+1 = 3^k) ∨ (n+2 = 3^k) ∨ (n+3 = 3^k)) : 
  units_digit n = 4 :=
by
  sorry

end units_digit_of_four_consecutive_l489_489645


namespace decoy_effect_rational_medium_purchase_l489_489972

structure PopcornOption where
  grams : ℕ
  price : ℕ

def small : PopcornOption := ⟨50, 200⟩
def medium : PopcornOption := ⟨70, 400⟩
def large : PopcornOption := ⟨130, 500⟩

-- Hypothesis: Small, Medium and Large options as described.
def options : List PopcornOption := [small, medium, large]

-- We need a theorem that states the usage of the decoy effect.
theorem decoy_effect (o : List PopcornOption) :
  (o = options) →
  (medium.grams < large.grams ∧ medium.price < large.price ∧ 
   (small.price < medium.price ∧ small.grams < medium.grams)) →
  (∃ d : PopcornOption, d = medium ∧ d ≠ small ∧ d ≠ large) →
  (∃ better_option : PopcornOption, better_option = large ∧
    better_option.price - medium.price ≤ 100 ∧
    better_option.grams - medium.grams ≥ 60) :=
begin
  intros hopts hcomp hdc,
  sorry
end

-- Rationality of buying medium-sized popcorn under certain conditions.
theorem rational_medium_purchase (o : List PopcornOption) :
  (o = options) →
  (∃ budget : ℕ, budget = 500 ∧ ∃ drink_price, drink_price = 100 ∧ 
   (medium.price + drink_price ≤ budget) ∧ (large.price > budget ∨ 
   small.grams < medium.grams)) →
  rational_choice : (PopcornOption → ℕ) (d :=
    if medium.price + drink_price ≤ budget then medium else if small.price ≤ budget then small else large) :=
begin
  intros hopts hbudget,
  sorry
end

end decoy_effect_rational_medium_purchase_l489_489972


namespace object_speed_is_approx_34_point_09_mph_l489_489264

noncomputable def object_speed (distance_feet : ℝ) (time_seconds : ℝ) : ℝ :=
  let distance_miles := distance_feet / 5280
  let time_hours := time_seconds / 3600
  distance_miles / time_hours

theorem object_speed_is_approx_34_point_09_mph :
  object_speed 200 4 ≈ 34.09 := sorry

end object_speed_is_approx_34_point_09_mph_l489_489264


namespace find_decreasing_interval_l489_489104

noncomputable def decreasing_interval (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem find_decreasing_interval :
  ∀ (f : ℝ → ℝ) (a b : ℝ),
  (∀ x, f x = Real.log (Real.sin (2 * x + Real.pi / 6)))
  ∧ ∃ (T > 0), ∀ x, f (x + T) = f x 
  ∧ T = Real.pi
  →
  decreasing_interval f (Real.pi / 6) (5 * Real.pi / 12) :=
by
  intro f a b
  assume h
  sorry

end find_decreasing_interval_l489_489104


namespace area_of_square_with_given_diagonal_l489_489881

-- Definition of the conditions
def diagonal := 12
def s := Real
def area (s : Real) := s^2
def diag_relation (d s : Real) := d^2 = 2 * s^2

-- The proof statement
theorem area_of_square_with_given_diagonal :
  ∃ s : Real, diag_relation diagonal s ∧ area s = 72 :=
by
  sorry

end area_of_square_with_given_diagonal_l489_489881


namespace neg_modulus_of_z_l489_489845

-- Defining the given complex number z
def z : ℂ := 6 + 8*complex.I

-- Proving the statement that -|z| = -10
theorem neg_modulus_of_z : -abs z = -10 := by
  sorry

end neg_modulus_of_z_l489_489845


namespace polynomial_divisibility_l489_489131

noncomputable def is_divisible (M N : Polynomial ℤ) : Prop :=
  ∃ Q : Polynomial ℤ, N = M * Q

def poly_M (x : ℤ) (k : ℕ) : Polynomial ℤ :=
  ∑ i in Finset.range (k+1), Polynomial.C 1 * Polynomial.X ^ (4 * i)

def poly_N (x : ℤ) (k : ℕ) : Polynomial ℤ :=
  ∑ i in Finset.range (k+1), Polynomial.C 1 * Polynomial.X ^ (2 * i)

theorem polynomial_divisibility (x : ℤ) (k : ℕ) :
  (k % 2 = 0) ↔ is_divisible (poly_M x k) (poly_N x k) := by
  sorry

end polynomial_divisibility_l489_489131


namespace arc_length_of_given_polar_curve_l489_489316

noncomputable def arc_length_polar (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ (x : ℝ) in a..b, real.sqrt ((f x) ^ 2 + (deriv f x) ^ 2)

theorem arc_length_of_given_polar_curve : 
  arc_length_polar (λ φ : ℝ, 6 * (1 + real.sin φ)) (-real.pi / 2) 0 = 12 * (2 - real.sqrt 2) :=
by
  -- Proof goes here
  sorry

end arc_length_of_given_polar_curve_l489_489316


namespace class_mean_l489_489887

theorem class_mean
  (num_students_1 : ℕ)
  (num_students_2 : ℕ)
  (total_students : ℕ)
  (mean_score_1 : ℚ)
  (mean_score_2 : ℚ)
  (new_mean_score : ℚ)
  (h1 : num_students_1 + num_students_2 = total_students)
  (h2 : total_students = 30)
  (h3 : num_students_1 = 24)
  (h4 : mean_score_1 = 80)
  (h5 : num_students_2 = 6)
  (h6 : mean_score_2 = 85) :
  new_mean_score = 81 :=
by
  sorry

end class_mean_l489_489887


namespace chessboard_tiling_impossible_l489_489345

def square := ℕ
def L_tile := Π (b w : square), b = 2 ∧ w = 2
def T_tile := Π (b w : square), (b = 3 ∧ w = 1) ∨ (b = 1 ∧ w = 3)

theorem chessboard_tiling_impossible :
  ∀ (board : square) (b w : square) (L_tiles : ℕ) (T_tiles : ℕ),
    board = 64 ∧ b = 32 ∧ w = 32 ∧ L_tiles = 15 ∧ T_tiles = 1 →
    ¬(∃ (bl wt : square), 
      (L_tile 2 2 → L_tiles * 2 = bl) ∧ (T_tile bl wt → bl + wt = 64)) :=
by
  sorry

end chessboard_tiling_impossible_l489_489345


namespace problem1_solution_set_problem2_range_of_a_l489_489854

-- Define the functions
def f (x a : ℝ) : ℝ := |2 * x - 1| + |2 * x + a|
def g (x : ℝ) : ℝ := x + 3

-- Problem 1: Proving the solution set when a = -2
theorem problem1_solution_set (x : ℝ) : (f x (-2) < g x) ↔ (0 < x ∧ x < 2) :=
  sorry

-- Problem 2: Proving the range of a
theorem problem2_range_of_a (a : ℝ) : 
  (a > -1) ∧ (∀ x, (x ∈ Set.Icc (-a/2) (1/2) → f x a ≤ g x)) ↔ a ∈ Set.Ioo (-1) (4/3) ∨ a = 4/3 :=
  sorry

end problem1_solution_set_problem2_range_of_a_l489_489854


namespace part1_part2_part3_l489_489946

section
  variable (a : ℝ) (n : ℕ) (a_n S_n : ℕ → ℝ) (b_n T_n : ℕ → ℝ)

  -- Condition: S_n is the sum of the first n terms of the sequence {a_n}
  def S_def (n : ℕ) : ℝ := ∑ i in range (n+1), a_n i

  -- Conditions
  hypothesis h1 : ∀ n, 2 ≤ n → S_n^2 = a_n * (S_n - 1/2)
  hypothesis h2 : S_n = λ n, (1 : ℝ) / (2 * n - 1)
  
  -- Prove that sequence {1/S_n} is arithmetic and find S_n
  theorem part1 : (∀ n, 2 ≤ n → (1 / S_def n - 1 / S_def (n-1)) = 2) ∧ (∀ n, S_n n = (1 : ℝ) / (2 * n - 1)) := sorry

  -- Definition: b_n and T_n
  def b_n (n : ℕ) : ℝ := S_n n / (2 * n + 3)
  def T_n (n : ℕ) : ℝ := ∑ i in range (n+1), b_n i
  
  -- Prove and find T_n
  theorem part2 : ∀ n, T_n n = (1 : ℝ) / 3 - (1 / 4) * ((1 : ℝ) / (2 * n + 1) + 1 / (2 * n + 3)) := sorry

  -- Prove the range of a
  hypothesis h3 : ∀ n, (4 * n ^ 2 - 4 * n + 10) * S_n n > (-1)^n * a 

  theorem part3 : - (34 / 5 : ℝ) < a ∧ a < (6 : ℝ) := sorry
end

end part1_part2_part3_l489_489946


namespace harmonic_series_inequality_l489_489183

theorem harmonic_series_inequality (n : ℕ) : 
  (∑ i in finset.range (2*n+1), 1 / (n + 1 + i)) > 1 :=
sorry

end harmonic_series_inequality_l489_489183


namespace percent_increase_output_l489_489929

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l489_489929


namespace greatest_int_with_gcd_of_24_eq_2_l489_489245

theorem greatest_int_with_gcd_of_24_eq_2 (n : ℕ) (h1 : n < 200) (h2 : Int.gcd n 24 = 2) : n = 194 := 
sorry

end greatest_int_with_gcd_of_24_eq_2_l489_489245


namespace nonzero_term_count_l489_489098

noncomputable def polynomial_expression_1 := (∀ x : ℝ, (x^2 + 2) * (3 * x^3 - x^2 + 4) - 2 * (x^4 - 3 * x^3 + x^2))

theorem nonzero_term_count : ∀ x : ℝ, has_4_nonzero_terms (polynomial_expression_1 x) := by
  sorry

end nonzero_term_count_l489_489098


namespace planted_fraction_correct_l489_489356

def planted_fraction (triangle_area square_area : ℝ) : ℝ := (triangle_area - square_area) / triangle_area

theorem planted_fraction_correct :
  ∀ (a b : ℝ), a = 3 → b = 4 →
  let h := Real.sqrt (a^2 + b^2) in
  ∀ (s_dist : ℝ), s_dist = 2 →
  let square_area := (2 / (a + b)^2) * h^2 in
  planted_fraction ((1/2) * a * b) square_area = 145 / 147 :=
by
  intros a b ha hb h s_dist hs square_area
  subst ha
  subst hb
  subst hs
  have triangle_area : ℝ := (1 / 2) * a * b
  have square_area : ℝ := 4 / 49
  rw [triangle_area, square_area]
  admit -- The detailed math proof would go here

end planted_fraction_correct_l489_489356


namespace find_a_l489_489102

theorem find_a (a : ℝ) (h : (6 * real.sqrt 3) / (3 * real.sqrt 2 - 2 * real.sqrt 3) = 3 * real.sqrt a + 6) : a = 6 :=
sorry

end find_a_l489_489102


namespace area_bounded_by_curves_l489_489730

theorem area_bounded_by_curves :
  (∫ y in 0..1, √(4 - y^2)) = (√3 / 2 + π / 3) :=
by
  sorry

end area_bounded_by_curves_l489_489730


namespace minimum_g_value_l489_489897

-- Define the lengths as conditions
def PQ : ℝ := 40
def SR : ℝ := 40
def PS : ℝ := 60
def RQ : ℝ := 60
def PR : ℝ := 80
def QS : ℝ := 80

-- Function g(X) defined for any point X in space
noncomputable def g (X : ℝ × ℝ × ℝ) : ℝ :=
  let PX : ℝ := dist (X, (0, 0, 0)) -- Placeholder for actual distance
  let QX : ℝ := dist (X, (PQ, 0, 0)) -- Placeholder for actual distance
  let RX : ℝ := dist (X, (PR, 0, 0)) -- Placeholder for actual distance
  let SX : ℝ := dist (X, (PS, SR, 0)) -- Placeholder for actual distance
  PX + QX + RX + SX

-- Prove the minimum value of g(X) is 4 * sqrt(1830)
theorem minimum_g_value : ∀ (X : ℝ × ℝ × ℝ), g(X) ≥ 4 * Real.sqrt 1830 := sorry

end minimum_g_value_l489_489897


namespace unpainted_area_of_parallelogram_l489_489636

theorem unpainted_area_of_parallelogram
  (base_5_inch_board : ℝ)
  (base_8_inch_board : ℝ)
  (angle_between_boards : ℝ)
  (height_under_eight_inch_board : ℝ)
  (unpainted_area : ℝ) :
  base_5_inch_board = 5 →
  base_8_inch_board = 8 →
  angle_between_boards = π / 4 →
  height_under_eight_inch_board = 5 * real.sin (π / 4) →
  unpainted_area = base_8_inch_board * height_under_eight_inch_board →
  unpainted_area = 20 * real.sqrt 2 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4, h5],
  sorry
end

end unpainted_area_of_parallelogram_l489_489636


namespace ratio_answers_to_questions_l489_489697

theorem ratio_answers_to_questions
  (num_members : ℕ)
  (avg_questions_per_hour : ℕ)
  (total_posts_per_day : ℕ)
  (h1 : num_members = 200)
  (h2 : avg_questions_per_hour = 3)
  (h3 : total_posts_per_day = 57600)
  : (43200/14400) = 3 :=
by
  have questions_per_member_per_day := 24 * avg_questions_per_hour,
  have total_questions_per_day := num_members * questions_per_member_per_day,
  have total_answers_per_day := total_posts_per_day - total_questions_per_day,
  have ratio := total_answers_per_day / total_questions_per_day,
  rw [h1, h2, h3] at *,
  rw [show 24 * 3 = 72, from rfl] at *,
  rw [show 200 * 72 = 14400, from rfl] at *,
  rw [show 57600 - 14400 = 43200, from rfl] at *,
  rw [show 43200 / 14400 = 3, from rfl],
  exact rfl

end ratio_answers_to_questions_l489_489697


namespace complex_number_solution_l489_489040

open Complex

theorem complex_number_solution (z : ℂ) (h1 : ∥z∥ = Real.sqrt 2) (h2 : z + conj z = 2) :
  z = 1 + I ∨ z = 1 - I :=
  sorry

end complex_number_solution_l489_489040


namespace team_mean_height_l489_489223

def heights_50s : list ℕ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
def heights_60s : list ℕ := [60, 62, 64, 65, 66, 67, 68, 69]
def heights_70s : list ℕ := [71, 72, 73, 75, 76, 78]

def all_heights : list ℕ := heights_50s ++ heights_60s ++ heights_70s

def num_players : ℕ := list.length all_heights

def total_height : ℕ := list.sum all_heights

def mean_height : ℕ := total_height / num_players
def mean_height_frac : ℚ := total_height / num_players.toRat

theorem team_mean_height : mean_height_frac = 62.9583 := by sorry

end team_mean_height_l489_489223


namespace assistant_increases_output_by_100_percent_l489_489938

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l489_489938


namespace students_with_puppies_and_parrots_l489_489459

theorem students_with_puppies_and_parrots (total_students : ℕ) 
    (percent_with_puppies : ℝ) 
    (percent_with_puppies_and_parrots : ℝ) 
    (h_total_students : total_students = 40) 
    (h_percent_with_puppies : percent_with_puppies = 0.80) 
    (h_percent_with_puppies_and_parrots : percent_with_puppies_and_parrots = 0.25) :
    let students_with_puppies := percent_with_puppies * total_students,
        students_with_both := percent_with_puppies_and_parrots * students_with_puppies
    in students_with_both = 8 :=
by {
  sorry
}

end students_with_puppies_and_parrots_l489_489459


namespace lcm_60_30_40_eq_120_l489_489366

theorem lcm_60_30_40_eq_120 : (Nat.lcm (Nat.lcm 60 30) 40) = 120 := 
sorry

end lcm_60_30_40_eq_120_l489_489366


namespace euler_formula_l489_489997

open Nat

-- Definitions of vertices, edges, faces, and connected planar graphs
def vertices (G : Type) := ℕ
def edges (G : Type) := ℕ
def faces (G : Type) := ℕ

structure PlanarGraph (G : Type) :=
  (V : vertices G)
  (E : edges G)
  (F : faces G)
  (connected : Prop)
  (planar : Prop)

-- Euler's formula for connected planar graphs
theorem euler_formula (G : Type) [pg : PlanarGraph G] (h_conn : pg.connected) (h_planar : pg.planar) : 
  pg.F = pg.E - pg.V + 2 :=
sorry

end euler_formula_l489_489997


namespace three_digit_numbers_sum_seven_l489_489009

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489009


namespace min_adjacent_seat_occupation_l489_489682

def minOccupiedSeats (n : ℕ) : ℕ :=
  n / 3

theorem min_adjacent_seat_occupation (n : ℕ) (h : n = 150) :
  minOccupiedSeats n = 50 :=
by
  -- Placeholder for proof
  sorry

end min_adjacent_seat_occupation_l489_489682


namespace parallel_lines_eq_a2_l489_489573

theorem parallel_lines_eq_a2
  (a : ℝ)
  (h : ∀ x y : ℝ, x + a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0)
  : a = 2 := 
  sorry

end parallel_lines_eq_a2_l489_489573


namespace sum_possible_values_of_x_l489_489251

open Real

noncomputable def mean (x : ℝ) : ℝ := (25 + x) / 7

def mode : ℝ := 2

def median (x : ℝ) : ℝ :=
  if x ≤ 2 then 2
  else if 4 ≤ x ∧ x ≤ 5 then 4
  else x

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem sum_possible_values_of_x 
  (values : set ℝ)
  (h : ∀ (x : ℝ), x ∈ values ↔ (is_arithmetic_progression 
                                    mode 
                                    (median x) 
                                    (mean x))): 
  ∑ x in values, x = 20 :=
by {
  have h1 : (is_arithmetic_progression 2 2 (mean 2)) → false :=
    -- Calculation here would show contradiction
    sorry,
  have h2 : (is_arithmetic_progression 2 4 (mean 17)) :=
    -- Arithmetic progression check here
    sorry,
  have h3 : (is_arithmetic_progression 2 3 (mean 3)) :=
    -- Arithmetic progression check here
    sorry,
  let values := { 17, 3 },
  have values_eq : values = {17, 3} := rfl,
  rw values_eq,
  exact sum_singleton 17 + sum_singleton 3 -- Sum of elements
}

end sum_possible_values_of_x_l489_489251


namespace max_distance_between_points_K_and_M_l489_489981

-- Definitions based on conditions
def point_C := (0, 0) : ℝ × ℝ 
def point_B := (3, 0) : ℝ × ℝ 
def point_A := (-(1/2), (Real.sqrt 3 / 2)) : ℝ × ℝ 

def circle1_center := (27 / 8, 0) : ℝ × ℝ 
def circle1_radius := 9 / 8 : ℝ 

def circle2_center := (-9 / 10, 9 * Real.sqrt 3 / 10) : ℝ × ℝ 
def circle2_radius := 6 / 5 : ℝ 

-- Statement of the problem
theorem max_distance_between_points_K_and_M :
  let D_max := Real.sqrt ( (circle1_center.1 - circle2_center.1)^2 + (circle1_center.2 - circle2_center.2)^2 )
  in D_max + circle1_radius + circle2_radius = (93 + 9 * Real.sqrt 409) / 40 := 
sorry

end max_distance_between_points_K_and_M_l489_489981


namespace num_three_digit_sums7_l489_489004

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l489_489004


namespace hexagon_side_length_is_correct_l489_489125

-- Definition of the problem
def equilateral_hexagon (A B C D E F : Type) : Prop :=
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ E ∧ E ≠ F ∧ F ≠ A) ∧
  (∀ {X Y Z}, (X = A ∨ X = B ∨ X = C ∨ X = D ∨ X = E ∨ X = F) →
    (Y = A ∨ Y = B ∨ Y = C ∨ Y = D ∨ Y = E ∨ Y = F) →
    (Z = A ∨ Z = B ∨ Z = C ∨ X = D ∨ Y = E ∨ Y = F) →
    (dist X Y = dist Y Z))

def acute_interior_angles (A B C D E F : Type) : Prop :=
  (∠B = 45 ∧ ∠D = 45 ∧ ∠F = 45)

def enclosed_area (A B C D E F : Type) : Prop :=
  (area (hexagon A B C D E F) = 9 * sqrt 2)

noncomputable def hexagon_side_length (A B C D E F : Type) : ℝ :=
  classical.some (exists_unique_of_exists _ $ begin
    sorry
  end)

-- The theorem stating the side length of the hexagon
theorem hexagon_side_length_is_correct {A B C D E F : Type} 
  (h1 : equilateral_hexagon A B C D E F) 
  (h2 : acute_interior_angles A B C D E F) 
  (h3 : enclosed_area A B C D E F) :
  hexagon_side_length A B C D E F = 2 * sqrt 3 :=
sorry

end hexagon_side_length_is_correct_l489_489125


namespace count_valid_subsets_l489_489139

def set_S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def valid_subsets (S : Finset ℕ) : Finset (Finset ℕ) :=
  S.powerset.filter (λ A, ∃ a1 a2 a3, a1 ∈ A ∧ a2 ∈ A ∧ a3 ∈ A ∧
    a1 < a2 ∧ a2 < a3 ∧ a3 - a2 ≤ 6)

theorem count_valid_subsets : (valid_subsets set_S).card = 83 :=
by {
  sorry
}

end count_valid_subsets_l489_489139


namespace part_a_l489_489870

noncomputable def M (n k : ℕ) : ℕ :=
  Nat.lcm_list (List.range' (n - k + 1) k)

noncomputable def f (n : ℕ) : ℕ :=
  Nat.find_greatest (λ k, (k ≤ n) ∧ ∀ i, 1 ≤ i → i < k → M n i < M n (i+1)) n

theorem part_a (n : ℕ) (hn.pos: 0 < n) : f n < 3 * Nat.sqrt n :=
  sorry

end part_a_l489_489870


namespace find_a_from_graph_l489_489726

-- Definitions corresponding to conditions
def is_positive (x : ℝ) := 0 < x

def sec (x : ℝ) : ℝ := 1 / real.cos x

noncomputable def y (a b x : ℝ) : ℝ := a * sec (b * x)

-- Main statement
theorem find_a_from_graph (a b : ℝ) 
  (ha : is_positive a) (hb : is_positive b)
  (hmin : ∀ x : ℝ, 0 ≤ x → x < π → y a b x = 3) :
  a = 3 :=
sorry

end find_a_from_graph_l489_489726


namespace min_obtuse_triangles_l489_489976

/--
Given 5 points on a plane with no three points collinear, 
prove that the minimum number of obtuse triangles 
that can be formed using these points as vertices is 2.
-/
theorem min_obtuse_triangles (p : Finset (EuclideanSpace ℝ (Fin 2))) (h₁ : p.card = 5)
  (h₂ : ∀ (a b c : EuclideanSpace ℝ (Fin 2)), a ∈ p → b ∈ p → c ∈ p → a ≠ b → a ≠ c → b ≠ c → ¬ collinear ({a, b, c} : Set (EuclideanSpace ℝ (Fin 2)))) :
  ∃ t : Finset (Finset (EuclideanSpace ℝ (Fin 2))), t.card = 2 ∧ ∀ (τ ∈ t), ∃ (a b c : EuclideanSpace ℝ (Fin 2)),
    a ∈ p ∧ b ∈ p ∧ c ∈ p ∧ τ = {a, b, c} ∧ ∃ (α β γ : ℝ), α + β + γ = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2) :=
sorry

end min_obtuse_triangles_l489_489976


namespace find_point_of_tangency_l489_489242

noncomputable def point_of_tangency (ξ η : ℝ) : Prop :=
ξ^2 + η^2 = 1 ∧ 
(π * (1 - ξ^2)^2 / (3 * ξ) - π * ((1 - ξ)^2 * (2 + ξ)) / 3 = 4 * π / 3) 

theorem find_point_of_tangency : 
  ∃ (ξ η : ℝ), point_of_tangency ξ η ∧ ξ = 3 - 2 * real.sqrt 2 :=
begin
  sorry
end

end find_point_of_tangency_l489_489242


namespace jessica_age_proof_l489_489499

-- Definitions based on conditions
def grandmother_age (j : ℚ) : ℚ := 15 * j
def age_difference (g j : ℚ) : Prop := g - j = 60

-- Proposed age of Jessica
def jessica_age : ℚ := 30 / 7

-- Main statement to prove
theorem jessica_age_proof : ∃ j : ℚ, grandmother_age j = 15 * j ∧ age_difference (grandmother_age j) j ∧ j = jessica_age :=
by sorry

end jessica_age_proof_l489_489499


namespace det_condition_l489_489842

theorem det_condition (a b c d : ℤ) 
    (h_exists : ∀ m n : ℤ, ∃ h k : ℤ, a * h + b * k = m ∧ c * h + d * k = n) :
    |a * d - b * c| = 1 :=
sorry

end det_condition_l489_489842


namespace medium_as_decoy_and_rational_choice_l489_489973

/-- 
  Define the prices and sizes of the popcorn containers:
  Small: 50g for 200 rubles.
  Medium: 70g for 400 rubles.
  Large: 130g for 500 rubles.
-/
structure PopcornContainer where
  size : ℕ -- in grams
  price : ℕ -- in rubles

def small := PopcornContainer.mk 50 200
def medium := PopcornContainer.mk 70 400
def large := PopcornContainer.mk 130 500

/-- 
  The medium-sized popcorn container can be considered a decoy
  in the context of asymmetric dominance.
  Additionally, under certain budget constraints and preferences, 
  rational economic agents may find the medium-sized container optimal.
-/
theorem medium_as_decoy_and_rational_choice :
  (medium.price = 400 ∧ medium.size = 70) ∧ 
  (∃ (budget : ℕ) (pref : ℕ → ℕ → Prop), (budget ≥ medium.price ∧ 
    pref medium.size (budget - medium.price))) :=
by
  sorry

end medium_as_decoy_and_rational_choice_l489_489973


namespace div3_of_div9_l489_489185

theorem div3_of_div9 (u v : ℤ) (h : 9 ∣ (u^2 + u * v + v^2)) : 3 ∣ u ∧ 3 ∣ v :=
sorry

end div3_of_div9_l489_489185


namespace total_teachers_l489_489293

theorem total_teachers (total_individuals sample_size sampled_students : ℕ)
  (H1 : total_individuals = 2400)
  (H2 : sample_size = 160)
  (H3 : sampled_students = 150) :
  ∃ total_teachers, total_teachers * (sample_size / (sample_size - sampled_students)) = 2400 / (sample_size / (sample_size - sampled_students)) ∧ total_teachers = 150 := 
  sorry

end total_teachers_l489_489293


namespace percent_increase_output_per_hour_l489_489916

-- Definitions and conditions
variable (B H : ℝ) -- B is the number of bears per week, H is the number of hours per week

-- Define outputs per hour
def output_per_hour (B H : ℝ) := B / H
def new_output_per_hour (B H : ℝ) := 1.8 * B / (0.9 * H)

-- A mathematical statement to prove the percentage increase of output per hour
theorem percent_increase_output_per_hour (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((new_output_per_hour B H) - (output_per_hour B H)) / (output_per_hour B H) * 100 = 100 :=
by
  sorry

end percent_increase_output_per_hour_l489_489916


namespace sin_sin_sin_sin_sin_eq_x_div_3_three_solutions_l489_489869

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (Real.sin (Real.sin (Real.sin (Real.sin x))))

noncomputable def g (x : ℝ) : ℝ := x / 3

theorem sin_sin_sin_sin_sin_eq_x_div_3_three_solutions :
  ∃ n : ℕ, n = 3 ∧ ∃! x in set.Icc 0 3, f x = g x :=
sorry

end sin_sin_sin_sin_sin_eq_x_div_3_three_solutions_l489_489869


namespace number_of_correct_answers_l489_489114

-- We define variables C (number of correct answers) and W (number of wrong answers).
variables (C W : ℕ)

-- Define the conditions given in the problem.
def conditions :=
  C + W = 75 ∧ 4 * C - W = 125

-- Define the theorem which states that the number of correct answers is 40.
theorem number_of_correct_answers
  (h : conditions C W) :
  C = 40 :=
sorry

end number_of_correct_answers_l489_489114


namespace champion_class_is_3_l489_489890

-- Definitions for the problem
def class := ℕ

def judgeA (c : class) : Prop := c ≠ 3 ∧ c ≠ 4
def judgeB (c : class) : Prop := c ≠ 3 ∧ c = 5
def judgeC (c : class) : Prop := c ≠ 5 ∧ c = 3

def correct_judgments (judge : class → Prop) (c : class) (correct : ℕ) : Prop :=
  (judge c) = (correct = 2)

def incorrect_judgments (judge : class → Prop) (c : class) (incorrect : ℕ) : Prop :=
  (judge c) = (incorrect = 0)

def mixed_judgments (judge : class → Prop) (c : class) (correct : ℕ) (incorrect : ℕ) : Prop :=
  (judge c) = (correct = 1 ∧ incorrect = 1)

-- Main theorem statement
theorem champion_class_is_3 (champion : ℕ) :
  ∃ c : class, (c = 3) ∧ 
  (
    (correct_judgments judgeA c 2 ∧ incorrect_judgments judgeB c 0 ∧ mixed_judgments judgeC c 1 1) ∨
    (correct_judgments judgeB c 2 ∧ incorrect_judgments judgeA c 0 ∧ mixed_judgments judgeC c 1 1) ∨
    (correct_judgments judgeC c 2 ∧ incorrect_judgments judgeA c 0 ∧ mixed_judgments judgeB c 1 1)
  ) :=
begin
  existsi 3,
  sorry
end

end champion_class_is_3_l489_489890


namespace convex_quadrilateral_midpoints_l489_489892

theorem convex_quadrilateral_midpoints
  (A B C D X Y : ℝ^2)
  (h_convex : ConvexQuadrilateral A B C D)
  (h_AB : dist A B = 13)
  (h_BC : dist B C = 13)
  (h_CD : dist C D = 24)
  (h_DA : dist D A = 24)
  (h_angle_D : angle D = 60)
  (h_X_mid : midpoint B C X)
  (h_Y_mid : midpoint D A Y) :
  dist X Y ^ 2 = (1033 / 4) + 30 * Real.sqrt 3 :=
sorry

end convex_quadrilateral_midpoints_l489_489892


namespace find_a9_l489_489076

variable {a : ℕ → ℝ} 
variable {q : ℝ}

-- Conditions
def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def a3_eq_1 (a : ℕ → ℝ) : Prop := 
  a 3 = 1

def a5_a6_a7_eq_8 (a : ℕ → ℝ) : Prop := 
  a 5 * a 6 * a 7 = 8

-- Theorem to prove
theorem find_a9 {a : ℕ → ℝ} {q : ℝ} 
  (geom : geom_seq a q)
  (ha3 : a3_eq_1 a)
  (ha5a6a7 : a5_a6_a7_eq_8 a) : a 9 = 4 := 
sorry

end find_a9_l489_489076


namespace arccos_neg_one_eq_pi_l489_489742

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l489_489742


namespace non_intersecting_circles_concentric_inversion_l489_489989

-- Define a basic geometric setup (placeholders for actual definitions)
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the non-intersecting condition
def non_intersecting (S1 S2 : Circle) : Prop :=
  let d := (S1.center.1 - S2.center.1)^2 + (S1.center.2 - S2.center.2)^2
  d > (S1.radius + S2.radius)^2

-- Problem statement in Lean
theorem non_intersecting_circles_concentric_inversion
  (S1 S2 : Circle) (h : non_intersecting S1 S2) : 
  ∃ O k, -- Possible parameters for the inversion center and power
    let S1' := { center := (O.1, O.2), radius := k } -- Placeholder for actual inversion transformation
    let S2' := { center := (O.1, O.2), radius := k } -- Placeholder for actual inversion transformation
    S1'.center = S2'.center := 
sorry

end non_intersecting_circles_concentric_inversion_l489_489989


namespace polygon_sides_l489_489601

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l489_489601


namespace sum_of_digits_of_expr_l489_489249

theorem sum_of_digits_of_expr : (sum_of_digits (decimal_representation (2 ^ 2010 * 5 ^ 2012 * 3 * 7))) = 18 := sorry

end sum_of_digits_of_expr_l489_489249


namespace distance_between_intersections_l489_489477

-- Define the conditions
def polar_equiv_cartesian (r θ : ℝ) : Prop :=
  r = 2 * (cos θ) / (1 - cos θ ^ 2) →

def line_parametric (t α : ℝ) : Prop :=
  ∃ (x y : ℝ), x = 2 + t * cos α ∧ y = 1 + t * sin α

-- Define the proof problem
theorem distance_between_intersections 
  (α t1 t2 : ℝ)
  (ht : 0 ≠ α)
  (hα : sin α = cos α)
  (hintersect1 : polar_equiv_cartesian (2 + t1 * cos α) (1 + t1 * sin α))
  (hintersect2 : polar_equiv_cartesian (2 + t2 * cos α) (1 + t2 * sin α))
  (midpoint : ∃ (P : ℝ × ℝ), P = ((2, 1) : ℝ × ℝ) ∧ P = ((2 + (t1 + t2) / 2 * cos α, 1 + (t1 + t2) / 2 * sin α) : ℝ × ℝ)) :
  |t1 - t2| = 2 * sqrt 6 := 
sorry

end distance_between_intersections_l489_489477


namespace polygon_sides_l489_489592

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l489_489592


namespace polygon_sides_l489_489596

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l489_489596


namespace proposition_D_l489_489200

theorem proposition_D (a b : ℝ) (h : a > abs b) : a^2 > b^2 :=
by {
    sorry
}

end proposition_D_l489_489200


namespace total_calories_l489_489656

-- Definitions for the quantities of each snack item
def strawberries := 12
def yogurt := 6
def blueberries := 8
def apple := 1
def almonds := 10

-- Caloric content per unit of each snack item
def caloriesPerStrawberry := 4
def caloriesPerOunceYogurt := 17
def caloriesPerBlueberry := 0.8
def caloriesPerApple := 95
def caloriesPerAlmond := 7

-- Lean 4 statement to prove the total calories consumed
theorem total_calories :
  (strawberries * caloriesPerStrawberry +
   yogurt * caloriesPerOunceYogurt +
   blueberries * caloriesPerBlueberry +
   apple * caloriesPerApple +
   almonds * caloriesPerAlmond) = 321.4 := by
  sorry

end total_calories_l489_489656


namespace total_airflow_correct_l489_489630

def airflow_fan_A : ℕ := 10 * 10 * 60 * 7
def airflow_fan_B : ℕ := 15 * 20 * 60 * 5
def airflow_fan_C : ℕ := 25 * 30 * 60 * 5
def airflow_fan_D : ℕ := 20 * 15 * 60 * 2
def airflow_fan_E : ℕ := 30 * 60 * 60 * 6

def total_airflow : ℕ :=
  airflow_fan_A + airflow_fan_B + airflow_fan_C + airflow_fan_D + airflow_fan_E

theorem total_airflow_correct : total_airflow = 1041000 := by
  sorry

end total_airflow_correct_l489_489630


namespace vector_BC_l489_489883

def vector_subtraction (v1 v2 : ℤ × ℤ) : ℤ × ℤ :=
(v1.1 - v2.1, v1.2 - v2.2)

theorem vector_BC (BA CA BC : ℤ × ℤ) (hBA : BA = (2, 3)) (hCA : CA = (4, 7)) :
  BC = vector_subtraction BA CA → BC = (-2, -4) :=
by
  intro hBC
  rw [vector_subtraction, hBA, hCA] at hBC
  simpa using hBC

end vector_BC_l489_489883


namespace john_walks_further_than_nina_l489_489500

theorem john_walks_further_than_nina :
  let john_distance := 0.7
  let nina_distance := 0.4
  john_distance - nina_distance = 0.3 :=
by
  sorry

end john_walks_further_than_nina_l489_489500


namespace expected_defective_chips_l489_489684

noncomputable def defective_ratio (S1_defective S1_total S2_defective S2_total S3_defective S3_total S4_defective S4_total : ℕ) : ℚ :=
  (S1_defective + S2_defective + S3_defective + S4_defective) /
  (S1_total + S2_total + S3_total + S4_total : ℚ)

theorem expected_defective_chips (S1_defective S1_total S2_defective S2_total S3_defective S3_total S4_defective S4_total : ℕ) :
  ∀(total_shipment_chips : ℕ), total_shipment_chips = 60000 →
  let ratio := defective_ratio S1_defective S1_total S2_defective S2_total S3_defective S3_total S4_defective S4_total in
  ∃ expected_defective, expected_defective = ratio * total_shipment_chips :=
by
  intros total_shipment_chips h
  have ratio := defective_ratio _ _ _ _ _ _ _ _
  use ratio * total_shipment_chips
  sorry

end expected_defective_chips_l489_489684


namespace number_of_integer_solutions_l489_489807

theorem number_of_integer_solutions (x : ℤ) : 
  (x^2 < 10 * x) → {x | (x^2 < 10 * x)}.finite
    ∧ {x | (x^2 < 10 * x)}.to_finset.card = 9 :=
by
  sorry

end number_of_integer_solutions_l489_489807


namespace solve_f_l489_489801

def divisors (n : ℕ) : Finset ℕ := (Finset.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0)
def d (n : ℕ) : ℕ := (divisors n).card
def σ (n : ℕ) : ℕ := (divisors n).sum id

noncomputable def f : ℕ → ℕ := sorry

theorem solve_f :
  (∀ n : ℕ, f (d (n + 1)) = d (f n + 1)) ∧ (∀ n : ℕ, f (σ (n + 1)) = σ (f n + 1)) →
  ∀ n : ℕ, f n = n := sorry

end solve_f_l489_489801


namespace solution_set_exponential_inequality_l489_489585

theorem solution_set_exponential_inequality (x : ℝ) : 
  2^(x + 2) > 8 ↔ x > 1 := 
by
  sorry

end solution_set_exponential_inequality_l489_489585


namespace measure_angle_ACB_l489_489481

theorem measure_angle_ACB :
  (∀ (A B C: Type) (a b c : A → ℝ),
    ∠ B A C = 115 ∧
    ∠ A B D = 140 ∧
    (forall d : C, ∠ B A D + ∠ B A C = 180) ∧
    (forall e : D, ∠ B A C + ∠ B C A + ∠ C A B = 180) →
    ∠ C A B = 25) := sorry

end measure_angle_ACB_l489_489481


namespace KodyAgeIs32_l489_489023

-- Definition for Mohamed's current age
def mohamedCurrentAge : ℕ := 2 * 30

-- Definition for Mohamed's age four years ago
def mohamedAgeFourYrsAgo : ℕ := mohamedCurrentAge - 4

-- Definition for Kody's age four years ago
def kodyAgeFourYrsAgo : ℕ := mohamedAgeFourYrsAgo / 2

-- Definition to check Kody's current age
def kodyCurrentAge : ℕ := kodyAgeFourYrsAgo + 4

theorem KodyAgeIs32 : kodyCurrentAge = 32 := by
  sorry

end KodyAgeIs32_l489_489023


namespace min_value_of_M_l489_489086

noncomputable def f (x b : ℝ) : ℝ := |x^2 + b * x|

noncomputable def M (b : ℝ) : ℝ :=
  let max_f := λ x, f x b
  classical.some (exists_max_image (set.Univ ∩ set.Icc (0 : ℝ) 1) max_f)

theorem min_value_of_M :
  ∃ b : ℝ, M b = 3 - 2 * Real.sqrt 2 :=
by {
  sorry
}

end min_value_of_M_l489_489086


namespace LCM_of_numbers_l489_489291

theorem LCM_of_numbers (a b : ℕ) (h1 : a = 20) (h2 : a / b = 5 / 4): Nat.lcm a b = 80 :=
by
  sorry

end LCM_of_numbers_l489_489291


namespace sides_of_polygon_l489_489612

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l489_489612


namespace same_terminal_side_of_neg_527_l489_489582

def same_terminal_side (angle: ℝ) : Set ℝ := 
  {α | ∃ k : ℤ, α = k * 360 + angle}

theorem same_terminal_side_of_neg_527 :
  same_terminal_side (-527) = {α | ∃ k : ℤ, α = k * 360 + 193} := 
sorry

end same_terminal_side_of_neg_527_l489_489582


namespace area_of_triangle_ABD_l489_489234

-- Definitions required for conditions
def Triangle (A B C : Type) := (AB AC BC : ℝ)
def Circle (center : Type) (radius : ℝ)

def isMidpoint (M B C : Type) : Prop := (dist B M = dist M C)
def isDiameter (C1 C2 : Type) (diam : ℝ) : Prop := (dist C1 C2 = diam)
def meets (CLine1 CLine2 : Type) (point1 point2 : Type) : Prop := 
  -- Assume some notion of intersection points
  sorry

def rightAngle (∠ BDC : Prop) : Prop := sorry -- Definition of right angle

-- Math equivalent proof problem
theorem area_of_triangle_ABD : 
  ∀ (A B C D : Type), 
  (Triangle A B C) (10, 16, 10) → 
  (Circle (isMidpoint B C) 5) → 
  (isDiameter B C 10) → 
  (meets AC (C D)) →
  (rightAngle ∠BDC) →
  ∃ area : ℝ, area = 24 := 
by 
  -- Fill in the proof
  sorry

end area_of_triangle_ABD_l489_489234


namespace tangent_line_area_l489_489202

theorem tangent_line_area (x y : ℝ) (h : y = x^3) (hx : x = 3) (hy : y = 27) :
  let slope := 3 * x^2,
      tangent_line_eq := y - 27 = slope * (x - 3),
      x_intercept := (0, 54),
      y_intercept := (2, 0)
  in (2 * 54) / 2 = 54 :=
by
  sorry

end tangent_line_area_l489_489202


namespace reflection_of_midpoint_l489_489124

theorem reflection_of_midpoint :
  let z1 : ℂ := 5 - 4 * complex.i,
      z2 : ℂ := -9 + 15 * complex.i,
      midpoint : ℂ := (z1 + z2) / 2,
      reflection : ℂ := complex.conj midpoint * complex.i
  in reflection = 2 + 5.5 * complex.i :=
by
  let z1 := 5 - 4 * complex.i,
  let z2 := -9 + 15 * complex.i,
  let midpoint := (z1 + z2) / 2,
  let reflection := (-midpoint.re) + midpoint.im * complex.i,
  have midpoint_calculated : midpoint = -2 + 5.5 * complex.i := sorry,
  have reflection_calculated : reflection = 2 + 5.5 * complex.i := sorry,
  exact reflection_calculated

end reflection_of_midpoint_l489_489124


namespace ray_NK_passes_through_center_of_square_l489_489983

theorem ray_NK_passes_through_center_of_square
  (A B C D X N K F O : Type)
  [inhabited A] [has_mem A X] [has_mem X AD]
  (square_ABCD : A × B × C × D)
  (point_on_AD : X ∈ AD)
  (circle_inscribed_in_ABX : ∃ (circle : Type), 
    ∀ (AX BX AB: Type) [tangents AX BX AB x], 
      N ∈ circle ∧ K ∈ circle ∧ F ∈ circle )
  (center_O_of_square : O)
  : ray_NK_passes_through_center_O : Prop :=
sorry

end ray_NK_passes_through_center_of_square_l489_489983


namespace find_all_n_l489_489783

theorem find_all_n (n : ℕ) : 
  (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % n = 0) ↔ (∃ j : ℕ, n = 3^j) :=
by 
  -- proof goes here
  sorry

end find_all_n_l489_489783


namespace total_people_after_four_years_l489_489284

-- Define initial conditions
def initial_total_people : Nat := 9
def board_members : Nat := 3
def regular_members_initial : Nat := initial_total_people - board_members
def years : Nat := 4

-- Define the function for regular members over the years
def regular_members (n : Nat) : Nat :=
  if n = 0 then 
    regular_members_initial
  else 
    2 * regular_members (n - 1)

theorem total_people_after_four_years :
  regular_members years = 96 := 
sorry

end total_people_after_four_years_l489_489284


namespace non_negative_y_implies_x_range_l489_489821

theorem non_negative_y_implies_x_range {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 90) :
    (4 * cos x * sin x + 2 * cos x - 2 * sin x - 1 ≥ 0) ↔ (0 ≤ x ∧ x ≤ 60) :=
by
  sorry

end non_negative_y_implies_x_range_l489_489821


namespace spacy_subsets_of_1_to_15_l489_489338

def spacy_subsets_count : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 3
| 3     := 4
| (n+4) := spacy_subsets_count (n+3) + spacy_subsets_count (n+1)

theorem spacy_subsets_of_1_to_15 : spacy_subsets_count 15 = 406 :=
by
  sorry

end spacy_subsets_of_1_to_15_l489_489338


namespace sequence_formula_l489_489079

theorem sequence_formula :
  ∀ n : ℕ, n > 0 →
    (∀ i : ℕ, i = 1 → a i = 0) →
    (∀ i : ℕ, i = 2 → a i = (Real.sqrt 3) / 3) →
    (∀ i : ℕ, i = 3 → a i = (Real.sqrt 2) / 2) →
    (∀ i : ℕ, i = 4 → a i = (Real.sqrt 15) / 5) →
    (∀ i : ℕ, i = 5 → a i = (Real.sqrt 6) / 3) →
    a n = Real.sqrt ((n - 1) / (n + 1)) :=
sorry

end sequence_formula_l489_489079


namespace y_intercept_is_minus_2_l489_489088

-- Define the given condition about the line and its inclination angle
def line_equation (a : ℝ) : Prop := ∃ y : ℝ, ∀ x : ℝ, a * x + y + 2 = 0
def inclination_angle : ℝ := 3 * Real.pi / 4

-- Define the value of a from tan(inclination angle)
lemma solve_a (a : ℝ) : tan (3 * Real.pi / 4) = -a → a = 1 :=
by sorry

-- Main theorem statement proving the y-intercept of the line
theorem y_intercept_is_minus_2 (a : ℝ) (h : tan inclination_angle = -a) : 
  let equation := (ax + y + 2 = 0),
      intercept := -2
  in 
  solve_a h → intercept = -2 :=
by sorry

end y_intercept_is_minus_2_l489_489088


namespace acute_angle_between_bisectors_l489_489382

theorem acute_angle_between_bisectors 
  (F D C K : Type)
  [is_tangent (circle_circumscribed F D C) F K]
  (h1 : ∠ K F C = 58) 
  (h2 : K ≠ D)
  (h3 : is_on_opposite_sides K D F C):
  acute_angle (angle_bisector ∠ C F D) (angle_bisector ∠ F C D) = 61 :=
sorry

end acute_angle_between_bisectors_l489_489382


namespace concurrency_of_lines_l489_489164
-- Import the necessary library

-- Define the environment with all geometric constructs
variable {Point : Type}

variable (A B C D E F A₁ C₁ : Point)

-- Assume the given conditions
variable [HasSegment Point]

-- Declare the geometric parallel condition and circumcircle properties
noncomputable def is_trapezoid (A B C D : Point) : Prop :=
  parallel (line A D) (line B C)

noncomputable def on_segment (X Y Z : Point) : Prop :=
  lies_on X (segment Y Z)

noncomputable def circumcircle (A X Y : Point) : circle Point :=
  circle.of_3_pts A X Y

noncomputable def other_intersection (line : line Point) (circle : circle Point) (P : Point) : Point :=
  other_intersection_point P (intersection line circle)

-- Create the conditions using definitions
axiom trapezoid_def : is_trapezoid A B C D
axiom E_on_segment_AB : on_segment E A B
axiom F_on_segment_DC : on_segment F D C
axiom A₁_def : A₁ = other_intersection (line A D) (circumcircle A E F) A
axiom C₁_def : C₁ = other_intersection (line B C) (circumcircle C E F) C

theorem concurrency_of_lines :
  concurrent (line B D) (line E F) (line A₁ C₁) := by
  sorry

end concurrency_of_lines_l489_489164


namespace roots_magnitude_one_l489_489036

theorem roots_magnitude_one {a b c : ℂ} (h : ∀ (w : ℂ), w ∈ (polynomial.roots (polynomial.C c + polynomial.C b * polynomial.X + polynomial.C a * polynomial.X ^ 2 + polynomial.X ^ 3)) → |w| = 1) :
  ∀ w ∈ (polynomial.roots (polynomial.C (complex.abs c) + polynomial.C (complex.abs b) * polynomial.X + polynomial.C (complex.abs a) * polynomial.X ^ 2 + polynomial.X ^ 3)), |w| = 1 :=
sorry

end roots_magnitude_one_l489_489036


namespace medium_as_decoy_and_rational_choice_l489_489974

/-- 
  Define the prices and sizes of the popcorn containers:
  Small: 50g for 200 rubles.
  Medium: 70g for 400 rubles.
  Large: 130g for 500 rubles.
-/
structure PopcornContainer where
  size : ℕ -- in grams
  price : ℕ -- in rubles

def small := PopcornContainer.mk 50 200
def medium := PopcornContainer.mk 70 400
def large := PopcornContainer.mk 130 500

/-- 
  The medium-sized popcorn container can be considered a decoy
  in the context of asymmetric dominance.
  Additionally, under certain budget constraints and preferences, 
  rational economic agents may find the medium-sized container optimal.
-/
theorem medium_as_decoy_and_rational_choice :
  (medium.price = 400 ∧ medium.size = 70) ∧ 
  (∃ (budget : ℕ) (pref : ℕ → ℕ → Prop), (budget ≥ medium.price ∧ 
    pref medium.size (budget - medium.price))) :=
by
  sorry

end medium_as_decoy_and_rational_choice_l489_489974


namespace triangular_partition_l489_489188

-- Define triangular number function
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_partition (s : ℕ) (h_pos : s > 0) (h_ne_2 : s ≠ 2) :
  ∃ (t : Finset ℕ), t.card = s ∧ (∀ (n ∈ t), ∃ k, triangular_number k = n) ∧ 
  (∑ n in t, (1 : ℚ) / n) = 1 :=
sorry

end triangular_partition_l489_489188


namespace fred_initial_cards_l489_489815

variables {n : ℕ}

theorem fred_initial_cards (h : n - 22 = 18) : n = 40 :=
by {
  sorry
}

end fred_initial_cards_l489_489815


namespace car_average_mpg_l489_489724

/-- 
Prove that the car's average miles-per-gallon for the entire trip is 25.7,
given the initial odometer reading, gasoline added during the trip, and
final odometer reading.
-/
theorem car_average_mpg
  (initial_odometer : ℕ := 56200)
  (first_fill_gallons : ℕ := 8)
  (second_fill_gallons : ℕ := 15)
  (second_odometer : ℕ := 56590)
  (third_fill_gallons : ℕ := 20)
  (final_odometer : ℕ := 57100)
  : (final_odometer - initial_odometer) / 
     (second_fill_gallons + third_fill_gallons : ℝ) ≈ 25.7 :=
by sorry

end car_average_mpg_l489_489724


namespace locus_of_point_is_circle_l489_489829

theorem locus_of_point_is_circle (x y : ℝ) 
  (h : 10 * Real.sqrt ((x - 1)^2 + (y - 2)^2) = |3 * x - 4 * y|) : 
  ∃ (c : ℝ) (r : ℝ), ∀ (x y : ℝ), (x - c)^2 + (y - c)^2 = r^2 := 
sorry

end locus_of_point_is_circle_l489_489829


namespace eq_b_minus_a_l489_489578

   -- Definition for rotating a point counterclockwise by 180° around another point
   def rotate_180 (h k x y : ℝ) : ℝ × ℝ :=
     (2 * h - x, 2 * k - y)

   -- Definition for reflecting a point about the line y = -x
   def reflect_y_eq_neg_x (x y : ℝ) : ℝ × ℝ :=
     (-y, -x)

   -- Given point Q(a, b)
   variables (a b : ℝ)

   -- Image of Q after the transformations
   def Q_transformed :=
     (5, -1)

   -- Image of Q after reflection about y = -x
   def Q_reflected :=
     reflect_y_eq_neg_x (5) (-1)

   -- Image of Q after 180° rotation around (2,3)
   def Q_original :=
     rotate_180 (2) (3) a b

   -- Statement we want to prove:
   theorem eq_b_minus_a : b - a = 6 :=
   by
     -- Calculation steps
     sorry
   
end eq_b_minus_a_l489_489578


namespace sum_of_possible_x_values_l489_489254

theorem sum_of_possible_x_values : 
  let lst : List ℝ := [10, 2, 5, 2, 4, 2, x]
  let mean := (25 + x) / 7
  let mode := 2
  let median := if x ≤ 2 then 2 else if 2 < x ∧ x < 4 then x else 4
  mean, median, and mode form a non-constant arithmetic progression 
  -> ∃ x_values : List ℝ, sum x_values = 20 :=
by
  sorry

end sum_of_possible_x_values_l489_489254


namespace bug_total_distance_l489_489678

theorem bug_total_distance :
  let start_position := -2
  let first_stop := -6
  let final_position := 5
  abs(first_stop - start_position) + abs(final_position - first_stop) = 15 :=
by
  sorry

end bug_total_distance_l489_489678


namespace volume_of_parallelepiped_l489_489584

-- Given definitions for a and b and condition on the angle

variable (a b : ℝ) (h_angle : ∀ {α : ℝ}, α = 30)

-- Definition of the volume of the rectangular parallelepiped
theorem volume_of_parallelepiped 
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_angle_correct : ∀ {d : ℝ}, d = √(a^2 + b^2) ∧ sin (30) = 1/2) : 
  ∃ (V : ℝ), V = a * b * (√(3 * a^2 - b^2)) :=
by 
  use a * b * (√(3 * a^2 - b^2))
  sorry

end volume_of_parallelepiped_l489_489584


namespace max_true_statements_l489_489949

theorem max_true_statements (a b : ℝ) :
  let S1 := (1 / a > 1 / b),
      S2 := (|a| > |b|),
      S3 := (a > b),
      S4 := (a < 0),
      S5 := (b > 0)
  in (S1, S2, S3, S4, S5).filter (λ s, s).length ≤ 3 :=
sorry

end max_true_statements_l489_489949


namespace paint_needed_for_540_statues_l489_489875

def paint_needed_for_similar_statues 
  (paint_needed_for_one_large_statue : ℕ)
  (height_large_statue height_small_statue : ℕ)
  (number_of_small_statues : ℕ) : ℕ :=
  let ratio := (height_small_statue : ℚ) / height_large_statue
  let small_statue_paint := (paint_needed_for_one_large_statue : ℚ) * ratio^2
  (number_of_small_statues : ℚ) * small_statue_paint

theorem paint_needed_for_540_statues :
  paint_needed_for_similar_statues 1 6 1 540 = 15 := 
by
  sorry

end paint_needed_for_540_statues_l489_489875


namespace problem_statement_l489_489859

variables {a b : EuclideanSpace ℝ (fin 2)}
variable {θ : ℝ}

def unit_vector (v : EuclideanSpace ℝ (fin 2)) : Prop :=
  ∥v∥ = 1

def angle_between (a b : EuclideanSpace ℝ (fin 2)) : ℝ :=
  real.arccos (a ⬝ b / (∥a∥ * ∥b∥))

def sufficient_condition (a b : EuclideanSpace ℝ (fin 2)) (θ : ℝ) : Prop :=
  (π / 6 < θ ∧ θ < π / 3) → ∥a - b∥ < 1

def necessary_condition (a b : EuclideanSpace ℝ (fin 2)) (θ : ℝ) : Prop :=
  ∥a - b∥ < 1 → (π / 6 < θ ∧ θ < π / 3)

noncomputable def is_sufficient_but_not_necessary (a b : EuclideanSpace ℝ (fin 2)) (θ : ℝ) : Prop :=
  (sufficient_condition a b θ) ∧ ¬(necessary_condition a b θ)

theorem problem_statement (a b : EuclideanSpace ℝ (fin 2)) (θ : ℝ) :
  unit_vector a →
  unit_vector b →
  angle_between a b = θ →
  ∥a - b∥ < 1 →
  (π / 6 < θ ∧ θ < π / 3) →
  is_sufficient_but_not_necessary a b θ :=
sorry

end problem_statement_l489_489859


namespace diamond_problem_l489_489758

-- define the operation diamond
def diamond (a b : ℝ) : ℝ := a + 1 / b

-- state the theorem as described in step c)
theorem diamond_problem : 
  (diamond (diamond 3 4) 5) - diamond 3 (diamond 4 5) = 89 / 420 :=
by
  sorry

end diamond_problem_l489_489758


namespace max_total_length_N3_max_total_length_N4_l489_489176

-- Definitions based on the problem's conditions
def equator_length : ℝ := 1
def circumference (road : ℕ) := 1
def max_train_length (n : ℕ) : ℝ := n / 2

-- Theorem statements for the specific cases
theorem max_total_length_N3 : (max_train_length 3 = 1.5) := by
  sorry

theorem max_total_length_N4 : (max_train_length 4 = 2) := by
  sorry

end max_total_length_N3_max_total_length_N4_l489_489176


namespace josh_final_pencils_l489_489941

-- Define the conditions as Lean definitions
def josh_initial_pencils : ℕ := 142
def percent_given_to_dorothy : ℚ := 25 / 100
def pencils_given_to_dorothy : ℕ := floor (percent_given_to_dorothy * josh_initial_pencils)
def pencils_given_to_mark : ℕ := 10
def pencils_given_back_by_dorothy : ℕ := 6

-- Define the final computation as Lean definition
theorem josh_final_pencils : 
  josh_initial_pencils - pencils_given_to_dorothy - pencils_given_to_mark + pencils_given_back_by_dorothy = 103 := 
by
  sorry

end josh_final_pencils_l489_489941


namespace find_expression_value_l489_489064

theorem find_expression_value (m: ℝ) (h: m^2 - 2 * m - 1 = 0) : 
  (m - 1)^2 - (m - 3) * (m + 3) - (m - 1) * (m - 3) = 6 := 
by 
  sorry

end find_expression_value_l489_489064


namespace store_discount_claims_correctness_l489_489703

-- Define the initial conditions
def original_price := ℝ
def first_discount_rate := 0.25
def second_discount_rate := 0.15
def claimed_total_discount := 0.40

-- Compute the true total discount
def true_total_discount : ℝ :=
  1.0 - ((1.0 - first_discount_rate) * (1.0 - second_discount_rate))

-- Compute the difference between the claimed and true discounts
def discount_difference : ℝ :=
  claimed_total_discount - true_total_discount

theorem store_discount_claims_correctness :
  true_total_discount = 0.3625 ∧ discount_difference = 0.0375 :=
by
  sorry

end store_discount_claims_correctness_l489_489703


namespace cuboid_volume_is_correct_l489_489209

-- Definition of cuboid edges and volume calculation
def cuboid_volume (a b c : ℕ) : ℕ := a * b * c

-- Given conditions
def edge1 : ℕ := 2
def edge2 : ℕ := 5
def edge3 : ℕ := 3

-- Theorem statement
theorem cuboid_volume_is_correct : cuboid_volume edge1 edge2 edge3 = 30 := 
by sorry

end cuboid_volume_is_correct_l489_489209


namespace quotient_remains_unchanged_l489_489207

theorem quotient_remains_unchanged (d v : ℚ) :
  ∀ (k ≠ 0), (d * k) / (v * k) = d / v := by
  sorry

end quotient_remains_unchanged_l489_489207


namespace value_corresponds_l489_489435

-- Define the problem
def certain_number (x : ℝ) : Prop :=
  0.30 * x = 120

-- State the theorem to be proved
theorem value_corresponds (x : ℝ) (h : certain_number x) : 0.40 * x = 160 :=
by
  sorry

end value_corresponds_l489_489435


namespace jane_output_increase_l489_489925

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l489_489925


namespace sum_q_t_12_eq_2048_l489_489140

-- Define the set of all 12-tuples where each entry is either 0 or 1
def T : Set (Fin 12 → Bool) :=
  { t | True }

-- Define the polynomial q_t(x) for each 12-tuple t in T
noncomputable def q_t (t : Fin 12 → Bool) (x : ℕ) : ℝ := 
  if h : x < 12 then ite (t ⟨x, h⟩) 1.0 0.0 else 0.0

-- Define the polynomial q(x) as the sum of q_t(x) over all t in T
noncomputable def q (x : ℕ) : ℝ :=
  ∑ t in T, q_t t x

-- The statement that proves the sum
theorem sum_q_t_12_eq_2048 : 
  q 12 = 2048 := sorry

end sum_q_t_12_eq_2048_l489_489140


namespace min_ticket_gates_l489_489348

theorem min_ticket_gates (a x y : ℕ) (h_pos: a > 0) :
  (a = 30 * x) ∧ (y = 2 * x) → ∃ n : ℕ, (n ≥ 4) ∧ (a + 5 * x ≤ 5 * n * y) :=
by
  sorry

end min_ticket_gates_l489_489348


namespace ant_crawling_routes_ratio_l489_489311

theorem ant_crawling_routes_ratio 
  (m n : ℕ) 
  (h1 : m = 2) 
  (h2 : n = 6) : 
  n / m = 3 :=
by
  -- Proof is omitted (we only need the statement as per the instruction)
  sorry

end ant_crawling_routes_ratio_l489_489311


namespace cos_C_sin_A_l489_489451

-- Define the problem conditions
variables {A B C : ℕ}
variable (cos_half_C : ℝ)
-- The given conditions
def conditions := cos_half_C = (√5) / 5 ∧ BC = 1 ∧ AC = 5

-- Statement proving the first part of the question
theorem cos_C (cos_half_C : ℝ) (h : conditions cos_half_C) : cos C = (-3) / 5 :=
by { sorry }

-- Statement proving the second part of the question
theorem sin_A (cos_half_C : ℝ) (h : conditions cos_half_C) : sin A = √2 / 10 :=
by { sorry }

-- Validate that the theorems compile correctly.
#check @cos_C
#check @sin_A

end cos_C_sin_A_l489_489451


namespace problem1_l489_489091

noncomputable def a : ℕ → ℚ
| 0       := 2
| (n + 1) := (a n - 1) / a n

def S (n : ℕ) : ℚ := ∑ k in finset.range n, a k

theorem problem1 : S 2013 = 2013 / 2 :=
by 
  sorry

end problem1_l489_489091


namespace trapezoid_area_formula_l489_489302

noncomputable def trapezoid_circumscribed_area (R α β : ℝ) : ℝ := 
  (4 * R^2 * (Real.sin ((α + β) / 2)) * (Real.cos ((α - β) / 2))) / (Real.sin α * Real.sin β)

theorem trapezoid_area_formula 
  (R α β : ℝ) 
  (h1 : 0 < R) 
  (h2 : 0 < Real.sin α) 
  (h3 : 0 < Real.sin β) 
  (h4 : α ∈ Ioo 0 (2 * Real.pi)) 
  (h5 : β ∈ Ioo 0 (2 * Real.pi)) :
  ∃ S : ℝ,
  S = trapezoid_circumscribed_area R α β :=
by 
  use trapezoid_circumscribed_area R α β
  sorry

end trapezoid_area_formula_l489_489302


namespace arccos_neg_one_eq_pi_l489_489751

theorem arccos_neg_one_eq_pi : real.arccos (-1) = real.pi :=
by sorry

end arccos_neg_one_eq_pi_l489_489751


namespace min_F_on_negative_reals_l489_489841

def F (m n : ℝ) (f g : ℝ → ℝ) (x : ℝ) : ℝ := m * (f x) + n * (g x) + x + 2

axiom odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

theorem min_F_on_negative_reals (f g : ℝ → ℝ) (hf : odd_function f) (hg : odd_function g)
    (m n : ℝ) (hF : ∀ x ∈ set.Ioi 0, F m n f g x ≤ 8) :
    ∃ x ∈ set.Iio 0, F m n f g x = -4 :=
by 
  sorry

end min_F_on_negative_reals_l489_489841


namespace range_f_l489_489343

noncomputable def f (x : ℝ) : ℝ := (x + 1) / x^2

theorem range_f : 
  {y : ℝ | ∃ x : ℝ, x ≠ 0 ∧ f x = y} = set.univ :=
sorry

end range_f_l489_489343


namespace matrix_power_identity_l489_489141

-- Define the matrix B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![0, 2]]

-- Define the identity matrix I
def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

-- Prove that B^15 - 3 * B^14 is equal to the given matrix
theorem matrix_power_identity :
  B ^ 15 - 3 • (B ^ 14) = ![![0, 4], ![0, -1]] :=
by
  -- Sorry is used here so the Lean code is syntactically correct
  sorry

end matrix_power_identity_l489_489141


namespace island_length_l489_489308

/-- Proof problem: Given an island in the Indian Ocean with a width of 4 miles and a perimeter of 22 miles. 
    Assume the island is rectangular in shape. Prove that the length of the island is 7 miles. -/
theorem island_length
  (width length : ℝ) 
  (h_width : width = 4)
  (h_perimeter : 2 * (length + width) = 22) : 
  length = 7 :=
sorry

end island_length_l489_489308


namespace carlos_local_tax_deduction_l489_489323

theorem carlos_local_tax_deduction :
  let hourly_wage_dollars := 25
  let hourly_wage_cents := hourly_wage_dollars * 100
  let tax_rate := 2.5 / 100
  hourly_wage_cents * tax_rate = 62.5 :=
by
  sorry

end carlos_local_tax_deduction_l489_489323


namespace intersection_A_B_l489_489964

open Set

def U := ℝ
def A := { x : ℝ | (2 * x + 3) / (x - 2) > 0 }
def B := { x : ℝ | abs (x - 1) < 2 }

theorem intersection_A_B : (A ∩ B) = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_A_B_l489_489964


namespace smaller_circle_circumference_l489_489044

-- Definitions based on the conditions given in the problem
def AB : ℝ := 24
def BC : ℝ := 45
def CD : ℝ := 28
def DA : ℝ := 53
def smaller_circle_diameter : ℝ := AB

-- Main statement to prove
theorem smaller_circle_circumference :
  let r : ℝ := smaller_circle_diameter / 2
  let circumference := 2 * Real.pi * r
  circumference = 24 * Real.pi := by
  sorry

end smaller_circle_circumference_l489_489044


namespace parallel_lines_and_line_equation_l489_489424

structure Point := 
  (x : ℝ) 
  (y : ℝ)

def M := Point.mk 1 1
def N := Point.mk 3 (-1)
def P := Point.mk 4 0
def Q := Point.mk 2 2

def slope (A B : Point) : ℝ := (B.y - A.y) / (B.x - A.x)

theorem parallel_lines_and_line_equation :
  slope M N = slope P Q ∧ (∀ x y : ℝ, (y = -1 * (x - P.x) + P.y) → (x + y - 4 = 0)) :=
by 
  sorry

end parallel_lines_and_line_equation_l489_489424


namespace arccos_neg_one_eq_pi_l489_489754

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_eq_pi_l489_489754


namespace range_of_a_l489_489879

noncomputable def y (a x : ℝ) : ℝ := a * Real.exp x + 3 * x
noncomputable def y_prime (a x : ℝ) : ℝ := a * Real.exp x + 3

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, a * Real.exp x + 3 = 0 ∧ a * Real.exp x + 3 * x < 0) → a < -3 :=
by
  sorry

end range_of_a_l489_489879


namespace constant_term_in_expansion_l489_489411

theorem constant_term_in_expansion (n : ℕ) (h_sum : (3 : ℤ)^n = 243) :
  let c := binom n 3 * 2^3 in
  n = 5 → c = 80 :=
by
  sorry

end constant_term_in_expansion_l489_489411


namespace project_completion_time_l489_489696

theorem project_completion_time (A B C : ℕ) (hA : A = 4) (hB : B = 6) (hC : C = 5) 
    (start_B : ℕ := 3) (start_C : ℕ := start_B + 4) :
    P = 12 := 
by
  have completion_A := 0 + A,
  have completion_B := start_B + B,
  have completion_C := start_C + C,
  have completion_project := max (max completion_A completion_B) completion_C,
  have P_eq_completion_C : P = completion_C := sorry,
  exact P_eq_completion_C

end project_completion_time_l489_489696


namespace jessica_age_proof_l489_489498

-- Definitions based on conditions
def grandmother_age (j : ℚ) : ℚ := 15 * j
def age_difference (g j : ℚ) : Prop := g - j = 60

-- Proposed age of Jessica
def jessica_age : ℚ := 30 / 7

-- Main statement to prove
theorem jessica_age_proof : ∃ j : ℚ, grandmother_age j = 15 * j ∧ age_difference (grandmother_age j) j ∧ j = jessica_age :=
by sorry

end jessica_age_proof_l489_489498


namespace number_of_integer_solutions_l489_489804

theorem number_of_integer_solutions (x : ℤ) : (x^2 < 10 * x) → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 :=
begin
  assume h,
  sorry
end

#eval List.length [1, 2, 3, 4, 5, 6, 7, 8, 9] -- 9

end number_of_integer_solutions_l489_489804


namespace jane_output_increase_l489_489928

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l489_489928


namespace yanna_change_l489_489652

theorem yanna_change :
  let shirt_cost := 5 in
  let sandel_cost := 3 in
  let total_money := 100 in
  let num_shirts := 10 in
  let num_sandels := 3 in
  let total_cost := (num_shirts * shirt_cost) + (num_sandels * sandel_cost) in
  let change := total_money - total_cost in
  change = 41 :=
by
  sorry

end yanna_change_l489_489652


namespace max_blue_drummers_l489_489312

/-- Given a 50x50 grid of drummers, each drummer is dressed in either blue or red. 
    A blue-dressed drummer can only see red-dressed drummers in all directions (360 degrees). 
    This theorem proves that the maximum number of blue-dressed drummers is 625. 
-/
theorem max_blue_drummers (num_rows num_cols : ℕ) 
  (h_rows : num_rows = 50) (h_cols : num_cols = 50) (drummers : ℕ → ℕ → Prop) :
  (∃ blue_drummers : ℕ, blue_drummers ≤ num_rows * num_cols 
    ∧ (∀ i j, drummers i j → drummers i j = 0 
      ∨ (∀ i' j', ((abs (i - i') ≤ 1 ∧ abs (j - j') ≤ 1) ∧ drummers i' j' = 1) → false)) 
    ∧ blue_drummers = 625) :=
by
  have h : 25 * 25 = 625 := rfl
  have h_sum : num_rows * num_cols = 2500 := by rw [h_rows, h_cols]
  exact ⟨625, by linarith, sorry⟩

end max_blue_drummers_l489_489312


namespace num_paths_l489_489349

theorem num_paths (n r : ℕ) (hr : r = 3) (hn : n = 7) : 
  nat.choose n r = 35 :=
by {
  simp [hr, hn],
  sorry
}

end num_paths_l489_489349


namespace probability_of_selection_l489_489239

theorem probability_of_selection (total_population : ℕ) (sample_size : ℕ) (h1 : total_population = 121) (h2 : sample_size = 12) : 
  (sample_size : ℚ) / total_population = 12 / 121 := 
by {
  rw [h1, h2],
  norm_num,
}

end probability_of_selection_l489_489239


namespace volume_ratio_of_tetrahedron_and_cube_centroids_l489_489400

-- given conditions
def Tetrahedron (s : ℝ) := True
def CubeFromTetrahedronCentroids (T : Tetrahedron) := True
def volume_ratio (VT VC : ℝ) (p q : ℕ) [nat.coprime p q] := VT / VC = p / q

-- the theorem to prove
theorem volume_ratio_of_tetrahedron_and_cube_centroids (s : ℝ) (VT VC : ℝ) (p q : ℕ) 
  [nat.coprime p q] (hT : Tetrahedron s) (hC : CubeFromTetrahedronCentroids hT) 
  (hRatio : volume_ratio VT VC p q) : p + q = 5 :=
by
  sorry

end volume_ratio_of_tetrahedron_and_cube_centroids_l489_489400


namespace perimeter_of_rectangle_l489_489292

theorem perimeter_of_rectangle (area width : ℝ) (h_area : area = 750) (h_width : width = 25) :
  ∃ perimeter length, length = area / width ∧ perimeter = 2 * (length + width) ∧ perimeter = 110 := by
  sorry

end perimeter_of_rectangle_l489_489292


namespace set_intersection_A_B_l489_489856

theorem set_intersection_A_B :
  (A : Set ℤ) ∩ (B : Set ℤ) = { -1, 0, 1, 2 } :=
by
  let A := { x : ℤ | x^2 - x - 2 ≤ 0 }
  let B := {x : ℤ | x ∈ Set.univ}
  sorry

end set_intersection_A_B_l489_489856


namespace determine_radius_of_smaller_circle_l489_489561

noncomputable def radius_of_smaller_circle (S : ℝ) : ℝ :=
  sqrt (S / (π * (4 * π^2 - 1)))

theorem determine_radius_of_smaller_circle (S R_1 R_2 : ℝ) (h1 : R_2 = 2 * π * R_1) (h2 : S = π * R_2^2 - π * R_1^2) :
  R_1 = radius_of_smaller_circle S :=
by
  sorry

end determine_radius_of_smaller_circle_l489_489561


namespace total_marbles_is_correct_l489_489463

-- Declare the relevant variables and constants
variables (yellow purple orange : ℕ)
variables (ratio_yellow ratio_purple ratio_orange : ℕ)
variable (total_marbles : ℕ)

-- State initial conditions
axiom h_ratio  : ratio_yellow = 2
axiom h_pp_ratio : ratio_purple = 4
axiom h_orange_ratio : ratio_orange = 6
axiom h_orange_count : orange = 18

-- Calculated variables and milestones towards the solution
def part_size := orange / ratio_orange
def total_parts := ratio_yellow + ratio_purple + ratio_orange
def total_marbles_calc := part_size * total_parts

-- The theorem to prove
theorem total_marbles_is_correct : 
  total_marbles_calc = 36 :=
by
  sorry

end total_marbles_is_correct_l489_489463


namespace count_integer_solutions_x_sq_lt_10x_l489_489809

theorem count_integer_solutions_x_sq_lt_10x :
  {x : ℤ | x^2 < 10 * x}.card = 9 :=
sorry

end count_integer_solutions_x_sq_lt_10x_l489_489809


namespace complement_N_in_M_l489_489420

open Set

noncomputable def M : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
noncomputable def N : Set ℤ := {1, 2}

theorem complement_N_in_M :
  compl M N = {-1, 0, 3} :=
sorry

end complement_N_in_M_l489_489420


namespace soldiers_count_l489_489717

-- Statements of conditions and proofs
theorem soldiers_count (n : ℕ) (s : ℕ) :
  (n * n + 30 = s) →
  ((n + 1) * (n + 1) - 50 = s) →
  s = 1975 :=
by
  intros h1 h2
  -- We know from h1 and h2 that there should be a unique solution for s and n that satisfies both
  -- conditions. Our goal is to show that s must be 1975.

  -- Initialize the proof structure
  sorry

end soldiers_count_l489_489717


namespace fibonacci_contains_21_l489_489900

-- Definition of the Fibonacci sequence
def fibonacci : ℕ → ℕ 
| 0 => 1
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Theorem statement: Proving that 21 is in the Fibonacci sequence
theorem fibonacci_contains_21 : ∃ n, fibonacci n = 21 :=
by
  sorry

end fibonacci_contains_21_l489_489900


namespace two_digit_numbers_sum_154_l489_489763

theorem two_digit_numbers_sum_154 : 
  {n : ℕ | 10 * (n % 10) + n / 10 = 154}.card = 5 :=
by 
  -- Definitions:
  let dig a := a / 10
  let dig' b := b % 10
  let pairs := {n : ℕ | ∀ (a b : ℕ), n = 10 * a + b ∧ 10 * b + a = 154}

  -- The number of valid pairs (a, b) such that a + b = 14;  a and b are between 0 and 9 inclusive.
  count pairs sorry

end two_digit_numbers_sum_154_l489_489763


namespace shift_graph_l489_489233

def f (x : ℝ) : ℝ := cos (1/2 * x)
def g (x : ℝ) : ℝ := cos (1/2 * x + π / 3)

theorem shift_graph :
  ∀ x : ℝ, g x = f (x + 2 * π / 3) :=
by
  sorry

end shift_graph_l489_489233


namespace pencils_given_l489_489310

-- Define the conditions
def a : Nat := 9
def b : Nat := 65

-- Define the goal statement: the number of pencils Kathryn gave to Anthony
theorem pencils_given (a b : Nat) (h₁ : a = 9) (h₂ : b = 65) : b - a = 56 :=
by
  -- Omitted proof part
  sorry

end pencils_given_l489_489310


namespace increase_in_output_with_assistant_l489_489923

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l489_489923


namespace minimum_value_F_l489_489065

def max (a b : ℝ) : ℝ := if a ≥ b then a else b

theorem minimum_value_F (x y m n : ℝ) (h : m + n = 6) : 
  ∃ (F : ℝ), F = max (abs (x^2 - 4 * y + m)) (abs (y^2 - 2 * x + n)) ∧ F = 1 / 2 :=
by
  sorry

end minimum_value_F_l489_489065


namespace ramesh_selling_price_l489_489189

noncomputable def labelled_price_original : ℝ := 17361.11
noncomputable def final_price_after_discounts : ℝ := 12500
noncomputable def transport_cost : ℝ := 125
noncomputable def installation_cost : ℝ := 250
noncomputable def vat_rate : ℝ := 0.05
noncomputable def luxury_tax_rate : ℝ := 0.10
noncomputable def investment_rate : ℝ := 0.05
noncomputable def investment_period : ℝ := 2
noncomputable def profit_margin : ℝ := 0.15

def selling_price (labelled_price : ℝ) (invested_amount : ℝ) : ℝ :=
  let total_cost_before_luxury_tax := final_price_after_discounts + transport_cost + installation_cost + (vat_rate * final_price_after_discounts)
  let luxury_tax := luxury_tax_rate * total_cost_before_luxury_tax
  let total_amount_paid := total_cost_before_luxury_tax + luxury_tax
  let final_amount_after_investment := total_amount_paid * (1 + investment_rate)^investment_period
  final_amount_after_investment / (1 + profit_margin)

theorem ramesh_selling_price : 
  selling_price labelled_price_original final_price_after_discounts = 19965.28 :=
  sorry

end ramesh_selling_price_l489_489189


namespace sum_a2000_inv_a2000_l489_489318

theorem sum_a2000_inv_a2000 (a : ℂ) (h : a^2 - a + 1 = 0) : a^2000 + 1/(a^2000) = -1 :=
by
    sorry

end sum_a2000_inv_a2000_l489_489318


namespace no_positive_real_solutions_l489_489360

theorem no_positive_real_solutions 
  (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^3 + y^3 + z^3 = x + y + z) (h2 : x^2 + y^2 + z^2 = x * y * z) :
  false :=
by sorry

end no_positive_real_solutions_l489_489360


namespace quadruplets_sets_l489_489723

theorem quadruplets_sets (a b c babies: ℕ) (h1: 2 * a + 3 * b + 4 * c = 1200) (h2: b = 5 * c) (h3: a = 2 * b) :
  4 * c = 123 :=
by
  sorry

end quadruplets_sets_l489_489723


namespace travel_routes_A_to_D_l489_489737

theorem travel_routes_A_to_D :
  let A := 'A'
  let B := 'B'
  let C := 'C'
  let D := 'D'
  let roads := [("A", "B", 3), ("A", "C", 1), ("B", "C", 2), ("C", "D", 3), ("B", "D", 1)]
  ∃ (A B C D : Type), 
    (list.length (roads) = 5) ∧ 
    ((∑ r in roads, if r.1 = A ∧ r.2 = B then r.3 else 0) = 3) ∧ 
    ((∑ r in roads, if r.1 = A ∧ r.2 = C then r.3 else 0) = 1) ∧ 
    ((∑ r in roads, if r.1 = B ∧ r.2 = C then r.3 else 0) = 2) ∧ 
    ((∑ r in roads, if r.1 = C ∧ r.2 = D then r.3 else 0) = 3) ∧ 
    ((∑ r in roads, if r.1 = B ∧ r.2 = D then r.3 else 0) = 1) ∧ 
    (∃ (path1 path2 : list (char × char × nat)),
      (path1 = [("A", "B", 3), ("B", "C", 2), ("C", "D", 3)]) ∧ 
      (path2 = [("A", "C", 1), ("C", "B", 2), ("B", "D", 1)]) ∧ 
      ((list.foldr (λ r acc, r.3 * acc) 1 path1 + 
        list.foldr (λ r acc, r.3 * acc) 1 path2) = 20) ) :=
sorry

end travel_routes_A_to_D_l489_489737


namespace find_angle_B_find_range_y_l489_489454

-- Definitions of the terms
variables (a b c : ℝ) (A B C : ℝ)
variable (y : ℝ)

-- Assumptions/conditions given in the problem
def triangle_conditions : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧
  A + B + C = π ∧
  C = π - A - B ∧
  (√3 / 3) * b * sin (A / 2) * cos (A / 2) = a * sin (B / 2)^2

-- First proof task: B = π / 3
theorem find_angle_B (h : triangle_conditions a b c A B C) : B = π / 3 :=
sorry

-- Second proof task: Define y and find its range
def y_def (A B C : ℝ) : ℝ := sin C - sin A

theorem find_range_y (h : triangle_conditions a b c A B C) (hy : y = y_def A B C) :
  y ∈ Ioo (-√3 / 2) (√3 / 2) :=
sorry

end find_angle_B_find_range_y_l489_489454


namespace five_letter_arrangements_l489_489863

theorem five_letter_arrangements : ∃ n : ℕ, 
  let letters := {A, B, C, D, E, F, G},
      usable_letters := letters.erase 'C' in
  (∀ s : List Char, s.length = 5 ∧ 
    s.head = some 'C' ∧ 
    (s.tail.head ≠ some 'D') ∧ 
    'B' ∈ s.tail ∧ 
    s.nodup_from (0 : ℕ)) →
  n = 408
:= sorry

end five_letter_arrangements_l489_489863


namespace all_mathematicians_play_all_humanities_l489_489469

theorem all_mathematicians_play_all_humanities
  (n m : ℕ) (hnm : n ≠ m) :
  ∀ (M : fin m) (H : fin n), ∃ (t : ℕ),
  ((∃ i, M = i % m) ∧ (∃ j, H = j % n) ∧ (t > 0 ∧ t < m * n) ∧ ∀ t < m * n, true) :=
by
  sorry

end all_mathematicians_play_all_humanities_l489_489469


namespace find_possible_values_d_l489_489190

theorem find_possible_values_d (u v c d : ℝ) (huv : u ≠ v)
  (h_p : ∀ x, x = u → p x = 0)
  (h_q : ∀ x, x = u + 3 → q x = 0) :
  d = -2601 ∨ d = -693 :=
by
  sorry

def p (x : ℝ) : ℝ := x ^ 3 + c * x + d
def q (x : ℝ) : ℝ := x ^ 3 + c * x + d + 360

end find_possible_values_d_l489_489190


namespace parallel_lines_coplanar_l489_489069

variables {Point : Type*} [AffineSpace ℝ Point]

variables {α β : AffineSubspace ℝ Point} (A C : Point) (hAC : A ≠ C) 
variables {B D : Point} (hBD : B ≠ D)

-- Assuming planes α and β are parallel and contain the respective points
variables (hαβ : α.direction = β.direction) (hAα : A ∈ α) (hCα : C ∈ α)
variables (hBβ : B ∈ β) (hDβ : D ∈ β)

theorem parallel_lines_coplanar :
  (span ℝ ({A, C} : set Point)).direction = (span ℝ ({B, D} : set Point)).direction ↔
  ∃ (P : AffineSubspace ℝ Point), A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ D ∈ P := sorry

end parallel_lines_coplanar_l489_489069


namespace segment_length_and_distance_from_midpoint_to_P_l489_489478

theorem segment_length_and_distance_from_midpoint_to_P :
  (∃ (t1 t2 : ℝ),
    (t1 + t2 = -4) ∧ (t1 * t2 = -10) ∧ 
    (|t1 - t2| = 2 * √14) ∧
    (let M := (-2, 2) in -- midpoint M has Cartesian coordinates (-2, 2)
     let P := (-2, 2) in -- point P after converting to Cartesian coordinates
     dist P M = 2)) :=
sorry

end segment_length_and_distance_from_midpoint_to_P_l489_489478


namespace probability_of_spinner_stopping_in_region_G_l489_489303

theorem probability_of_spinner_stopping_in_region_G :
  let pE := (1:ℝ) / 2
  let pF := (1:ℝ) / 4
  let y  := (1:ℝ) / 6
  let z  := (1:ℝ) / 12
  pE + pF + y + z = 1 → y = 2 * z → y = (1:ℝ) / 6 := by
  intros htotal hdouble
  sorry

end probability_of_spinner_stopping_in_region_G_l489_489303


namespace find_original_number_l489_489978

variable {I K S : ℕ}

theorem find_original_number
  (distinct_digits : I ≠ K ∧ K ≠ S ∧ S ≠ I)
  (non_zero_digits : I ≠ 0 ∧ K ≠ 0 ∧ S ≠ 0)
  (product_ends_with_S : (I * 100 + K * 10 + S) * (K * 100 + S * 10 + I) % 10 = S)
  (is_six_digit_number : 100000 ≤ (I * 100 + K * 10 + S) * (K * 100 + S * 10 + I) ∧ (I * 100 + K * 10 + S) * (K * 100 + S * 10 + I) < 1000000)
  (erased_zeros : erase_zeros ((I * 100 + K * 10 + S) * (K * 100 + S * 10 + I)) = I * 100 + K * 10 + S) :
  (I = 1) ∧ (K = 6) ∧ (S = 2) := sorry

end find_original_number_l489_489978


namespace sum_of_two_primes_eq_53_l489_489472

theorem sum_of_two_primes_eq_53 : 
  ∀ (p q : ℕ), prime p ∧ prime q ∧ p + q = 53 → 0 :=
by 
  sorry

end sum_of_two_primes_eq_53_l489_489472


namespace directrix_of_parabola_eq_neg_quarter_l489_489363

open Real

theorem directrix_of_parabola_eq_neg_quarter (x : ℝ) :
  let p := (x^2 - 4*x + 4) / 8
  in ∃ (d : ℝ), d = -1/4 ∧ (d = p - (1/4)) :=
by
  sorry

end directrix_of_parabola_eq_neg_quarter_l489_489363


namespace quotient_base4_l489_489351

def base4_to_base10 (n : ℕ) : ℕ :=
  n % 10 + 4 * (n / 10 % 10) + 4^2 * (n / 100 % 10) + 4^3 * (n / 1000)

def base10_to_base4 (n : ℕ) : ℕ :=
  let rec convert (n acc : ℕ) : ℕ :=
    if n < 4 then n * acc
    else convert (n / 4) ((n % 4) * acc * 10 + acc)
  convert n 1

theorem quotient_base4 (a b : ℕ) (h1 : a = 2313) (h2 : b = 13) :
  base10_to_base4 ((base4_to_base10 a) / (base4_to_base10 b)) = 122 :=
by
  sorry

end quotient_base4_l489_489351


namespace remainder_when_divided_by_100_l489_489137

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

def no_two_digits_are_same (n : ℕ) : Prop :=
  let digits := n.digits 10
  (∀ (d : ℕ), d ∈ digits → d ≠ 0) ∧ (∀ (i j : ℕ), i ≠ j → digits.nth i ≠ digits.nth j)

def largest_integer_multiple_of_9_with_conditions (n : ℕ) : Prop :=
  is_multiple_of_9 n ∧ no_two_digits_are_same n

theorem remainder_when_divided_by_100
  (M : ℕ)
  (h1 : largest_integer_multiple_of_9_with_conditions M) :
  M % 100 = 81 :=
sorry

end remainder_when_divided_by_100_l489_489137


namespace angle_between_lines_is_60_l489_489201

-- Define the two lines
def line1 (x y : ℝ) : Prop := x + sqrt 3 * y + 2 = 0
def line2 (x y : ℝ) : Prop := x + 1 = 0

-- Premises that follow from the given conditions
axiom line1_slope : ∀ x y, line1 x y → ∃ m : ℝ, m = - (1 / sqrt 3)
axiom line1_inclination_angle : ∀ x y, line1 x y → ∃ θ : ℝ, θ = 150

axiom line2_slope : ∀ x y, line2 x y → False
axiom line2_inclination_angle : ∀ x y, line2 x y → ∃ θ : ℝ, θ = 90

-- The theorem to prove
theorem angle_between_lines_is_60 : ∀ x1 y1 x2 y2, line1 x1 y1 → line2 x2 y2 → ∃ θ : ℝ, θ = 60 := by
  sorry

end angle_between_lines_is_60_l489_489201


namespace derivative_f_tangent_line_at_m_l489_489032

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) * Real.cos (3 * x)

theorem derivative_f (x : ℝ) : 
  deriv f x = (2 * Real.cos (3 * x) - 3 * Real.sin (3 * x)) * Real.exp (2 * x) := sorry

theorem tangent_line_at_m : 
  let m := ∫ x in 0..2 * Real.pi, Real.sin x 
  m = 0 → 
  f m = 1 → 
  (m, f m) = (0, 1) →
  let slope := 2
  ∀ (x : ℝ),
  y = 2 * x + 1 := sorry

end derivative_f_tangent_line_at_m_l489_489032


namespace positive_integers_powers_of_3_l489_489780

theorem positive_integers_powers_of_3 (n : ℕ) (h : ∀ k : ℤ, ∃ a : ℤ, n ∣ a^3 + a - k) : ∃ b : ℕ, n = 3^b :=
sorry

end positive_integers_powers_of_3_l489_489780


namespace no_solution_inequalities_l489_489108

theorem no_solution_inequalities (m : ℝ) : 
  (∀ x : ℝ, (2 * x - 1 < 3) → (x > m) → false) ↔ (m ≥ 2) :=
by 
  sorry

end no_solution_inequalities_l489_489108


namespace units_defective_produced_by_B_l489_489288

noncomputable def defective_units_produced_by_B (total_probability : ℝ) (defective_A : ℝ) (prob_A : ℝ) : ℝ :=
  let prob_B := total_probability - prob_A * defective_A in
  1 / (prob_B * (1 / prob_A))

theorem units_defective_produced_by_B : 
  defective_units_produced_by_B 0.0156 0.009 0.4 = 50 := 
sorry

end units_defective_produced_by_B_l489_489288


namespace max_value_sqrt_expression_l489_489504

noncomputable def expression_max_value (a b: ℝ) : ℝ :=
  Real.sqrt (a * b) + Real.sqrt ((1 - a) * (1 - b))

theorem max_value_sqrt_expression : 
  ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 → expression_max_value a b ≤ 1 :=
by
  intros a b h
  sorry

end max_value_sqrt_expression_l489_489504


namespace sum_real_imag_parts_l489_489028

noncomputable section

open Complex

theorem sum_real_imag_parts (z : ℂ) (h : z / (1 + 2 * i) = 2 + i) : 
  ((z + 5).re + (z + 5).im) = 0 :=
  by
  sorry

end sum_real_imag_parts_l489_489028


namespace triangle_cos_theta_l489_489831

-- Lean statement without the proof
theorem triangle_cos_theta (A B C E M : Type) [Triangle A B C] [Midpoint E A B]
  (acute_BEC : AcuteAngle (Angle B E C))
  (theta : Real)
  (angle_condition : Angle B M E = Angle E C A) :
  MC / AB = Cos theta :=
sorry

end triangle_cos_theta_l489_489831


namespace polygon_sides_l489_489595

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l489_489595


namespace rational_root_uniqueness_l489_489196

theorem rational_root_uniqueness (c : ℚ) :
  ∀ x1 x2 : ℚ, (x1 ≠ x2) →
  (x1^3 - 3 * c * x1^2 - 3 * x1 + c = 0) →
  (x2^3 - 3 * c * x2^2 - 3 * x2 + c = 0) →
  false := 
by
  intros x1 x2 h1 h2 h3
  sorry

end rational_root_uniqueness_l489_489196


namespace B_power_15_minus_3_B_power_14_l489_489144

def B : Matrix (Fin 2) (Fin 2) ℝ := !!
  [3, 4]
  [0, 2]

theorem B_power_15_minus_3_B_power_14 :
  B^15 - 3 • B^14 = !!
    [0, 4]
    [0, -1] := by
  sorry

end B_power_15_minus_3_B_power_14_l489_489144


namespace symmetric_function_arithmetic_sequence_l489_489106

theorem symmetric_function_arithmetic_sequence
  {f : ℝ → ℝ} {a b m n : ℝ} (h1 : a ≠ m)
  (symm1 : ∀ x, f(x) + f(2 * a - x) = 2 * b)
  (symm2 : ∀ x, f(x) + f(2 * m - x) = 2 * n) :
  ∀ x k : ℤ, (f (x + 2 * k * (a - m)) - f (x + 2 * (k - 1) * (a - m))) = 2 * (b - n) := 
by   
  intros x k
  sorry

end symmetric_function_arithmetic_sequence_l489_489106


namespace Megan_pays_correct_amount_l489_489969

def original_price : ℝ := 22
def discount : ℝ := 6
def amount_paid := original_price - discount

theorem Megan_pays_correct_amount : amount_paid = 16 := by
  sorry

end Megan_pays_correct_amount_l489_489969


namespace alec_votes_l489_489710

variable (students totalVotes goalVotes neededVotes : ℕ)

theorem alec_votes (h1 : students = 60)
                   (h2 : goalVotes = 3 * students / 4)
                   (h3 : totalVotes = students / 2 + 5 + (students - (students / 2 + 5)) / 5)
                   (h4 : neededVotes = goalVotes - totalVotes) :
                   neededVotes = 5 :=
by sorry

end alec_votes_l489_489710


namespace chess_tournament_total_players_l489_489460

/--
In a chess tournament, each player played exactly one game against each of the other players.
In each game, the winner was awarded 1 point, the loser got 0 points, and each of the two players earned 0.5 points in the case of a tie.
It was discovered that exactly half of the points earned by each player came from games against the eight players with the least number of total points.
Each of these eight lowest-scoring players earned half of their points against the other seven of the eight.
Prove that there are 16 players in total in the tournament.
-/
theorem chess_tournament_total_players 
  (players : ℕ)
  (games_played : ∀ (p1 p2 : ℕ), p1 ≠ p2 → games_played p1 p2 = 1)
  (points_awarded : ∀ (p1 p2 : ℕ), p1 ≠ p2 → (winner_points : ℝ) × (loser_points : ℝ) ∨ (draw_points : ℝ × ℝ) ∧ draw_points = (0.5 , 0.5))
  (points_from_weakest : ∀ p, (p < 8 → (total_points p) / 2 = points_against_weakest p))
  (points_within_weakest : ∀ p, (p < 8 → points_against_weakest p = (7 * total_points p) / 2))
  : players = 16 := 
sorry

end chess_tournament_total_players_l489_489460


namespace arccos_neg_one_eq_pi_proof_l489_489747

noncomputable def arccos_neg_one_eq_pi : Prop :=
  arccos (-1) = π

theorem arccos_neg_one_eq_pi_proof : arccos_neg_one_eq_pi := by
  sorry

end arccos_neg_one_eq_pi_proof_l489_489747


namespace hyperbola_eccentricity_l489_489665

-- Define the hyperbola and its conditions
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Define the focus F and points A and B
def focus (a b : ℝ) : ℝ × ℝ := (sqrt (a^2 + b^2), 0)

-- Define the foot of perpendicular and intersection points
def point_A (a b c : ℝ) : ℝ × ℝ := (3 * c / 4, 3 * b * c / (4 * a))
def point_B (a b c : ℝ) : ℝ × ℝ := (3 * c / 2, -3 * b * c / (2 * a))

-- Define the relationship between the vectors
def vector_relation (a b c : ℝ) : Prop :=
  2 * (c - 3 * c / 4, 3 * b * c / (4 * a)) = (3 * c / 2 - c, -3 * b * c / (2 * a))

-- Define the eccentricity calculation
def eccentricity (a b : ℝ) : ℝ :=
  sqrt (a^2 + b^2) / a

-- Prove the statement
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let c := sqrt (a ^ 2 + b ^ 2) in
  a^2 = 3 * b^2 →
  eccentricity a b = 2 * sqrt 3 / 3 :=
by
  intros ha hb c h
  sorry

end hyperbola_eccentricity_l489_489665


namespace problem1_problem2_l489_489267

-- Problem 1
theorem problem1 : sqrt ((-4 : ℝ) ^ 2) + 2 * (sqrt 2 - 3) - abs (-2 * sqrt 2) = -2 :=
by sorry

-- Problem 2
theorem problem2 (x y : ℝ) (h1 : x / 2 + y / 3 = 4) (h2 : x + 2 * y = 16) : x = 4 ∧ y = 6 :=
by sorry

end problem1_problem2_l489_489267


namespace sum_possible_values_of_x_l489_489252

open Real

noncomputable def mean (x : ℝ) : ℝ := (25 + x) / 7

def mode : ℝ := 2

def median (x : ℝ) : ℝ :=
  if x ≤ 2 then 2
  else if 4 ≤ x ∧ x ≤ 5 then 4
  else x

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem sum_possible_values_of_x 
  (values : set ℝ)
  (h : ∀ (x : ℝ), x ∈ values ↔ (is_arithmetic_progression 
                                    mode 
                                    (median x) 
                                    (mean x))): 
  ∑ x in values, x = 20 :=
by {
  have h1 : (is_arithmetic_progression 2 2 (mean 2)) → false :=
    -- Calculation here would show contradiction
    sorry,
  have h2 : (is_arithmetic_progression 2 4 (mean 17)) :=
    -- Arithmetic progression check here
    sorry,
  have h3 : (is_arithmetic_progression 2 3 (mean 3)) :=
    -- Arithmetic progression check here
    sorry,
  let values := { 17, 3 },
  have values_eq : values = {17, 3} := rfl,
  rw values_eq,
  exact sum_singleton 17 + sum_singleton 3 -- Sum of elements
}

end sum_possible_values_of_x_l489_489252


namespace find_sum_x_y_l489_489962

theorem find_sum_x_y (x y : ℝ) 
  (h1 : x^3 - 3 * x^2 + 2026 * x = 2023)
  (h2 : y^3 + 6 * y^2 + 2035 * y = -4053) : 
  x + y = -1 := 
sorry

end find_sum_x_y_l489_489962


namespace arseniy_cannot_collect_all_water_in_one_bucket_l489_489266

noncomputable def sum_first_n (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem arseniy_cannot_collect_all_water_in_one_bucket :
  let water_amounts := list.range' 1 10 + 1 in 
  let total_amount := water_amounts.sum in 
  total_amount = 55 ∧ ∀ a b : ℕ, a ∈ water_amounts → b ∈ water_amounts → a != b → (a + b = total_amount) → ∃ b' : ℕ, b' % 2 = 0 → total_amount % 2 == 1 :=
begin
  -- Define water amounts as [1,2,3,...,10]
  let water_amounts : list ℕ := [1,2,3,4,5,6,7,8,9,10],
  -- Define total water amount as the sum of the list.
  let total_amount := water_amounts.sum,
  -- Verify if sum == 55
  have water_sum : total_amount = 55,
  {
    norm_num [water_amounts.sum],
  },
  split,
  -- Verify total == 55 is odd.
  { exact water_sum },
  -- Define properties of water amounts
  {
    intros a b ha hb hab hsum,
    -- water_amounts invariant by condition stated.
    sorry
  }
end

end arseniy_cannot_collect_all_water_in_one_bucket_l489_489266


namespace valid_set_l489_489259

-- Define the conditions as types representing the sets
inductive SetOption
| A
| B
| C
| D

-- Definitions of definiteness of elements (based on the problem's conditions)
def isDefinite : SetOption → Prop
| SetOption.A := False
| SetOption.B := True
| SetOption.C := False
| SetOption.D := False

-- The question rephrased to judge which set is well-defined
theorem valid_set : ∃ s, isDefinite s ∧ s = SetOption.B :=
by 
  existsi SetOption.B
  constructor
  . exact True.intro
  . rfl
  sorry

end valid_set_l489_489259


namespace convex_polygon_point_set_l489_489151

theorem convex_polygon_point_set {n : ℕ} (h_n : 3 ≤ n) (P : fin n → ℝ × ℝ) 
  (h_convex : ∀ i j k : fin n, (∃ a b c : ℝ, (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧ 
  P i = (a • P i + b • P j + c • P k)) 
  → (
    (convex ℝ (set.range P)) 
    ∧ (affine_independent ℝ (set.range P))) : 
∃ S : fin (n - 2) → ℝ × ℝ, 
  ∀ (i j k : fin n), 
  (P i, P j, P k).triangle → 
  (∃ x : fin (n - 2), point_in_triangle S x (P i, P j, P k)) :=
sorry

end convex_polygon_point_set_l489_489151


namespace value_of_x3_plus_inv_x3_l489_489100

theorem value_of_x3_plus_inv_x3 (x : ℝ) (h : 728 = x^6 + 1 / x^6) : 
  x^3 + 1 / x^3 = Real.sqrt 730 :=
sorry

end value_of_x3_plus_inv_x3_l489_489100


namespace math_problem_l489_489374

variable {p q r s t : ℝ}
variable (h1 : p < q) (h2 : q < r) (h3 : r < s) (h4 : s < t)

-- Definitions for M and m functions
def M (a b : ℝ) := max a b
def m (a b : ℝ) := min a b

theorem math_problem :
  M (M p (m q s)) (m r (m p t)) = q := 
by 
  sorry

end math_problem_l489_489374


namespace sides_of_polygon_l489_489611

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l489_489611


namespace omega_range_l489_489213

theorem omega_range (ω : ℝ) (hω : ω > 0) :
  (∀ x ∈ Icc (0 : ℝ) (π / 2), 
   ∀ z ∈ Icc (0 : ℝ) (π / 2), 
   (0 ≤ x - z → cos (ω*x + π/6) ≤ cos (ω*z + π/6))) ↔ (0 < ω ∧ ω ≤ 5/3) :=
by sorry

end omega_range_l489_489213


namespace remaining_slices_after_weekend_l489_489992

theorem remaining_slices_after_weekend 
  (initial_pies : ℕ) (slices_per_pie : ℕ) (rebecca_initial_slices : ℕ) 
  (family_fraction : ℚ) (sunday_evening_slices : ℕ) : 
  initial_pies = 2 → 
  slices_per_pie = 8 → 
  rebecca_initial_slices = 2 → 
  family_fraction = 0.5 → 
  sunday_evening_slices = 2 → 
  (initial_pies * slices_per_pie 
   - rebecca_initial_slices 
   - family_fraction * (initial_pies * slices_per_pie - rebecca_initial_slices) 
   - sunday_evening_slices) = 5 :=
by 
  intros initial_pies_eq slices_per_pie_eq rebecca_initial_slices_eq family_fraction_eq sunday_evening_slices_eq
  sorry

end remaining_slices_after_weekend_l489_489992


namespace alec_votes_l489_489712

theorem alec_votes (class_size : ℕ) (half_class_votes : ℕ) (remaining_interested : ℕ) (fraction_persuaded : ℕ) :
  class_size = 60 →
  half_class_votes = class_size / 2 →
  remaining_interested = 5 →
  fraction_persuaded = (class_size - half_class_votes - remaining_interested) / 5 →
  (3 * class_size) / 4 - (half_class_votes + fraction_persuaded) = 10 :=
by
  intros h_class_size h_half_class_votes h_remaining_interested h_fraction_persuaded
  rw h_class_size at h_half_class_votes h_remaining_interested h_fraction_persuaded
  rw [h_half_class_votes, h_remaining_interested, h_fraction_persuaded]
  sorry

end alec_votes_l489_489712


namespace problem_solution_l489_489119

-- Definition of the parametric equations for curve C
def parametric_curve (α : ℝ) : ℝ × ℝ :=
(x = 3 * Real.cos α, y = Real.sin α)

-- General equation of curve C
def curve_general_equation (x y : ℝ) : Prop :=
x^2 / 9 + y^2 = 1

-- Polar equation for line l
def polar_line_equation (ρ θ : ℝ) : Prop :=
ρ * Real.sin (θ - Real.pi / 4) = Real.sqrt 2

-- Inclination angle of line l
def inclination_angle (θ : ℝ) : Prop :=
θ = Real.pi / 4

-- Point on line l
def point_on_line (P : ℝ × ℝ) : Prop :=
P = (0, 2)

-- Total distance from point P to points A and B
def PA_PB_distance (t1 t2 : ℝ) : ℝ :=
|t1| + |t2|

-- The main theorem combining all the above definitions
theorem problem_solution :
  (∀ α : ℝ, parametric_curve α)
  → (∀ x y : ℝ, curve_general_equation x y)
  ∧ (∀ θ ρ : ℝ, polar_line_equation ρ θ)
  ∧ (∀ θ : ℝ, inclination_angle θ)
  ∧ (∀ t1 t2 : ℝ, PA_PB_distance t1 t2 = 18 * Real.sqrt 2 / 5) :=
by 
  intros
  sorry

end problem_solution_l489_489119


namespace positive_integers_powers_of_3_l489_489781

theorem positive_integers_powers_of_3 (n : ℕ) (h : ∀ k : ℤ, ∃ a : ℤ, n ∣ a^3 + a - k) : ∃ b : ℕ, n = 3^b :=
sorry

end positive_integers_powers_of_3_l489_489781


namespace number_of_valid_arrays_l489_489096

-- Define types and helper functions if necessary
def is_valid_array (M : Matrix (Fin 5) (Fin 5) ℤ) : Prop :=
  (∀ i, (∑ j, M i j) = 0) ∧ -- Sum of each row is 0
  (∀ j, (∑ i, M i j) = 0) ∧ -- Sum of each column is 0
  (∀ i, ∃! j, M i j = 0) ∧ -- Each row contains exactly one 0
  (∀ j, ∃! i, M i j = 0)    -- Each column contains exactly one 0

-- The main proof statement
theorem number_of_valid_arrays : 
  ∃ n : ℕ, n = 933120 ∧ (∃ M : Matrix (Fin 5) (Fin 5) ℤ, is_valid_array M) :=
by
  use 933120
  split
  · refl
  · sorry  -- placeholder for the proof

end number_of_valid_arrays_l489_489096


namespace least_possible_BC_l489_489666

-- Define the lengths for \( \triangle ABC \) and \( \triangle BDC \)
variables (AB AC DC BD BC : ℕ)
variables (hAB : AB = 7) (hAC : AC = 15) (hDC : DC = 10) (hBD : BD = 24)

-- Define the conditions for the shared side \( BC \)
def validBC (BC : ℕ) : Prop := BC > AC - AB ∧ BC > BD - DC

-- Statement of the theorem
theorem least_possible_BC
  (h_validBC : validBC BC) :
  BC ≥ 14 :=
sorry

end least_possible_BC_l489_489666


namespace jack_quarantine_days_l489_489250

theorem jack_quarantine_days (total_hours : ℕ) (customs_hours : ℕ) (hours_in_day : ℕ) (H : total_hours = 356) (H1 : customs_hours = 20) (H2 : hours_in_day = 24) :
  (total_hours - customs_hours) / hours_in_day = 14 :=
by
  rw [H, H1, H2]
  sorry

end jack_quarantine_days_l489_489250


namespace locus_of_P_is_circle_l489_489826

def point_locus_is_circle (x y : ℝ) : Prop :=
  10 * real.sqrt ((x - 1)^2 + (y - 2)^2) = abs (3 * x - 4 * y)

theorem locus_of_P_is_circle :
  ∀ (x y : ℝ), point_locus_is_circle x y → (distance (x, y) (1, 2) = r) := 
sorry

end locus_of_P_is_circle_l489_489826


namespace sum_of_coordinates_of_B_is_7_l489_489532

-- Define points and conditions
def A := (0, 0)
def B (x : ℝ) := (x, 3)
def slope (p₁ p₂ : ℝ × ℝ) : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)

-- Main theorem to prove the sum of the coordinates of point B is 7
theorem sum_of_coordinates_of_B_is_7 (x : ℝ) (h_slope : slope A (B x) = 3 / 4) : x + 3 = 7 :=
by
  -- Proof goes here, we use sorry to skip the proof steps.
  sorry

end sum_of_coordinates_of_B_is_7_l489_489532


namespace max_value_of_a_l489_489090

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - a * x + 1

theorem max_value_of_a :
  ∃ (a : ℝ), (∀ (x : ℝ), (0 ≤ x ∧ x ≤ 1) → |f a x| ≤ 1) ∧ a = 8 := by
  sorry

end max_value_of_a_l489_489090


namespace cube_face_probability_l489_489501

-- Define the type of a cube and its vertices
inductive vertex : Type
| V1 | V2 | V3 | V4 | V5 | V6 | V7 | V8

-- Probabilities for Q relative to P
def prob_Q_eq_P : ℝ := 1 / 8
def prob_Q_neighbor_P : ℝ := 3 / 8
def prob_Q_face_diag_P : ℝ := 3 / 8
def prob_Q_space_diag_P : ℝ := 1 / 8

def prob_R_same_face_Q_eq_P : ℝ := 7 / 8
def prob_R_same_face_Q_neighbor_P : ℝ := 6 / 8
def prob_R_same_face_Q_face_diag_P : ℝ := 4 / 8
def prob_R_same_face_Q_space_diag_P : ℝ := 0 / 8

def probability_same_face : ℝ :=
  prob_Q_eq_P * prob_R_same_face_Q_eq_P +
  prob_Q_neighbor_P * prob_R_same_face_Q_neighbor_P +
  prob_Q_face_diag_P * prob_R_same_face_Q_face_diag_P +
  prob_Q_space_diag_P * prob_R_same_face_Q_space_diag_P

theorem cube_face_probability :
  probability_same_face = 37 / 64 :=
by
  sorry

end cube_face_probability_l489_489501


namespace arccos_neg_one_eq_pi_l489_489749

theorem arccos_neg_one_eq_pi : real.arccos (-1) = real.pi :=
by sorry

end arccos_neg_one_eq_pi_l489_489749


namespace min_value_x_3y_l489_489516

noncomputable def min_value (x y : ℝ) : ℝ := x + 3 * y

theorem min_value_x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) :
  ∃ (x y : ℝ), min_value x y = 18 + 21 * Real.sqrt 3 :=
sorry

end min_value_x_3y_l489_489516


namespace price_raise_percentage_correct_l489_489736

def P := 80  -- Original price of the dress.

def discount_price := 0.85 * P  -- Price after 15% discount.
def sale_price : ℝ := 68  -- Given price after discount.

def final_price := P - 5  -- Final price after increase.
def price_increase : ℝ := final_price - sale_price  -- Increase in price.

def percentage_increase := (price_increase / sale_price) * 100  -- Percentage increase in price.

theorem price_raise_percentage_correct :
    discount_price = sale_price →
    final_price = P - 5 →
    price_increase = final_price - sale_price →
    percentage_increase ≈ 10.29 :=
by
    sorry

end price_raise_percentage_correct_l489_489736


namespace inverse_proportion_decreases_l489_489192

theorem inverse_proportion_decreases {x : ℝ} (h : x > 0 ∨ x < 0) : 
  y = 3 / x → ∀ (x1 x2 : ℝ), (x1 > 0 ∨ x1 < 0) → (x2 > 0 ∨ x2 < 0) → x1 < x2 → (3 / x1) > (3 / x2) := 
by
  sorry

end inverse_proportion_decreases_l489_489192


namespace range_a_singleton_intersection_l489_489396

noncomputable def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd = a * p.fst + 2}

def B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd = abs (p.fst + 1)}

theorem range_a_singleton_intersection :
  ∀ (a : ℝ), (∃! p : ℝ × ℝ, p ∈ A a ∧ p ∈ B) ↔ (a ∈ (-∞, -1] ∪ [1, ∞)) := 
by
  sorry

end range_a_singleton_intersection_l489_489396


namespace number_of_married_men_at_least_11_l489_489465

-- Definitions based only on conditions from a)
def total_men := 100
def men_with_tv := 75
def men_with_radio := 85
def men_with_ac := 70
def married_with_tv_radio_ac := 11

-- Theorem that needs to be proven based on the conditions
theorem number_of_married_men_at_least_11 : total_men ≥ married_with_tv_radio_ac :=
by
  sorry

end number_of_married_men_at_least_11_l489_489465


namespace solve_for_x_l489_489344

theorem solve_for_x (x : ℤ) : 3^4 * 3^x = 81 → x = 0 := by
  sorry

end solve_for_x_l489_489344


namespace correct_statements_count_l489_489425

-- Definition of given conditions
variable (Lines Planes : Type) [Liner Line_Space : (Lines → Planes → Prop)]

-- Definition of statements ① - ④
def statement1 (m n : Lines) (α : Planes) : Prop :=
  (m ⊥ n) ∧ (m ⊥ α) → (n ∥ α)

def statement2 (m : Lines) (α β : Planes) : Prop :=
  (α ∥ β) ∧ (m ⊥ α) → (m ⊥ β)

def statement3 (m : Lines) (α β : Planes) : Prop :=
  (m ⊥ β) ∧ (α ⊥ β) → (m ∥ α)

def statement4 (m n : Lines) (α : Planes) : Prop :=
  (m ∥ α) ∧ (n ∥ α) → (m ∥ n)

-- The theorem to be proven
theorem correct_statements_count (m n : Lines) (α β : Planes) :
  (statement1 m n α → false) ∧
  statement2 m α β ∧
  (statement3 m α β → false) ∧
  (statement4 m n α → false) →
  (∃! (i : Nat), i = 1) :=
by
  -- Proof would go here
  sorry

end correct_statements_count_l489_489425


namespace parking_configuration_l489_489708

/-- There are 7 consecutive parking spaces. We need to park 3 different models of cars in such a way that the remaining 4 parking spaces are consecutive. The number of different parking methods is 24. -/
theorem parking_configuration (p : Finset ℕ) (h : p = {0, 1, 2, 3, 4, 5, 6}) :
  (∃ c : Finset ℕ, c.card = 3 ∧ p \ c = {0, 1, 2, 3, 4, 5, 6}.filter (λ x, x ≠ c) ∧ (p \ c).card = 4 ∧
    (p \ c).min ∈ (p \ c) ∧ (p \ c).max ∈ (p \ c)) →
  ∃ n, n = 24 := 
by
  sorry

end parking_configuration_l489_489708


namespace original_percentage_pure_ghee_l489_489896

noncomputable def percentage_pure_ghee (P V : ℕ) : ℚ :=
  (P.to_rat / (P + V).to_rat) * 100

theorem original_percentage_pure_ghee (P V : ℕ) (h₁ : P + V = 30) (h₂ : P + 20 = 35) : 
  percentage_pure_ghee P V = 50 := 
by
  sorry

end original_percentage_pure_ghee_l489_489896


namespace length_of_BE_l489_489909

theorem length_of_BE (A B C E : ℝ)
  (h₀ : 0 < A) 
  (h₁ : 0 < B) 
  (h₂ : 0 < C) 
  (h₃ : 0 < E)
  (h4 : B = 5)
  (h5 : C = 12)
  (h6 : A = 13)
  (h7 : E = sqrt 2)
  (h8: ((A^2) + (B^2)) = (C^2)) :
  E = (65 / 18) * sqrt 2 := 
by
  sorry

end length_of_BE_l489_489909


namespace part_a_part_b_l489_489911

variable (A B C O A₁ B₁ C₁ : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]
  (hA₁ : SegmentIntersect A O (BC : Set (MetricSpace Point)) A₁)
  (hB₁ : SegmentIntersect B O (CA : Set (MetricSpace Point)) B₁)
  (hC₁ : SegmentIntersect C O (AB : Set (MetricSpace Point)) C₁)

theorem part_a :
  (SegmentRatio O A₁ A A₁) + (SegmentRatio O B₁ B B₁) + (SegmentRatio O C₁ C C₁) = 1 :=
sorry

theorem part_b :
  (SegmentRatio AC₁ C₁B) * (SegmentRatio BA₁ A₁C) * (SegmentRatio CB₁ B₁A) = 1 :=
sorry


end part_a_part_b_l489_489911


namespace y_intercept_of_line_is_minus_one_l489_489628

theorem y_intercept_of_line_is_minus_one : 
  (∀ x y : ℝ, y = 2 * x - 1 → y = -1) :=
by
  sorry

end y_intercept_of_line_is_minus_one_l489_489628


namespace find_domain_f_l489_489208

def validDomain (x : ℝ) := 1 < x ∧ x < 2

theorem find_domain_f (x : ℝ) : (f: ℝ → ℝ) (x) = (log (x - 1)) / (sqrt (4 - x ^ 2)) → validDomain x := by
  sorry

end find_domain_f_l489_489208


namespace arccos_neg_one_eq_pi_proof_l489_489745

noncomputable def arccos_neg_one_eq_pi : Prop :=
  arccos (-1) = π

theorem arccos_neg_one_eq_pi_proof : arccos_neg_one_eq_pi := by
  sorry

end arccos_neg_one_eq_pi_proof_l489_489745


namespace each_number_in_column_l489_489052

noncomputable def grid_filled_with_all_numbers (n k m : ℕ) (h_kn : k < n) (h_km : k < m) (h_m_coprime : Nat.gcd m (n - k) = 1) : Prop :=
  ∀ col : fin n, ∀ num : ℕ, num ∈ finset.range n.succ → ∃ row : fin n, grid_entry n k m col.val row.val = num + 1

-- Placeholder for the function that computes the grid entry. You'll need to define this based on the provided cyclic permutation rule.
noncomputable def grid_entry (n k m col row : ℕ) : ℕ := sorry

theorem each_number_in_column (n k m : ℕ) (h_kn : k < n) (h_km : k < m) (h_m_coprime : Nat.gcd m (n - k) = 1) :
  grid_filled_with_all_numbers n k m h_kn h_km h_m_coprime :=
sorry

end each_number_in_column_l489_489052


namespace golden_ratio_expression_l489_489212

theorem golden_ratio_expression :
  let m := 2 * Real.sin (Real.pi / 10) in
  (Real.sin (7 * Real.pi / 30) + m) / Real.cos (7 * Real.pi / 30) = Real.sqrt 3 := 
sorry

end golden_ratio_expression_l489_489212


namespace convert_length_convert_area_convert_time_convert_mass_l489_489777

theorem convert_length (cm : ℕ) : cm = 7 → (cm : ℚ) / 100 = 7 / 100 :=
by sorry

theorem convert_area (dm2 : ℕ) : dm2 = 35 → (dm2 : ℚ) / 100 = 7 / 20 :=
by sorry

theorem convert_time (min : ℕ) : min = 45 → (min : ℚ) / 60 = 3 / 4 :=
by sorry

theorem convert_mass (g : ℕ) : g = 2500 → (g : ℚ) / 1000 = 5 / 2 :=
by sorry

end convert_length_convert_area_convert_time_convert_mass_l489_489777


namespace digits_of_3_pow_100_l489_489243

theorem digits_of_3_pow_100 : 
  ∀ (lg : ℝ → ℝ) (n : ℕ → ℝ) (a : ℝ) (N : ℕ) (h1 : 1 ≤ a ∧ a < 10) 
    (h2 : ∀ N, N = a * 10 ^ N) (h3 : ∀ N, lg N = N + lg a) 
    (h4 : ∀ N, N > 0 → ∃ d, digits N = d + 1) 
    (h5 : lg 3 ≈ 0.4771),
  N = 3^100 → digits (3^100) = 48 :=
by
  intros lg n a N h1 h2 h3 h4 h5 hN
  sorry

end digits_of_3_pow_100_l489_489243


namespace assistant_increases_output_by_100_percent_l489_489936

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l489_489936


namespace alec_votes_l489_489711

variable (students totalVotes goalVotes neededVotes : ℕ)

theorem alec_votes (h1 : students = 60)
                   (h2 : goalVotes = 3 * students / 4)
                   (h3 : totalVotes = students / 2 + 5 + (students - (students / 2 + 5)) / 5)
                   (h4 : neededVotes = goalVotes - totalVotes) :
                   neededVotes = 5 :=
by sorry

end alec_votes_l489_489711


namespace arccos_neg_one_eq_pi_l489_489743

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l489_489743


namespace coordinates_of_P_l489_489070

theorem coordinates_of_P (x0 y0 : ℝ) (h1 : y0 = 3 * x0^2) (h2 : deriv (λ x : ℝ, 3 * x^2) x0 = 6) :
(P (x0) (y0) : ℝ × ℝ) = (1, 3) :=
by {
  -- conditions from the problem
  have h1 : y0 = 3 * x0^2,
  have h2 : deriv (λ x : ℝ, 3 * x^2) x0 = 6,

  -- initial goal
  sorry
}

end coordinates_of_P_l489_489070


namespace sum_of_two_primes_eq_53_l489_489473

theorem sum_of_two_primes_eq_53 : 
  ∀ (p q : ℕ), prime p ∧ prime q ∧ p + q = 53 → 0 :=
by 
  sorry

end sum_of_two_primes_eq_53_l489_489473


namespace bmw_sold_l489_489687

-- Define the total number of cars sold.
def total_cars : ℕ := 500

-- Define the percentages of cars sold.
def mercedes_percentage : ℝ := 0.10
def toyota_percentage : ℝ := 0.30
def ford_percentage : ℝ := 0.20

-- Calculate the percentage of BMWs sold.
def bmw_percentage : ℝ := 1 - (mercedes_percentage + toyota_percentage + ford_percentage)

-- Calculate the number of BMWs sold.
def number_of_bmws_sold : ℕ := total_cars * bmw_percentage.toInt

theorem bmw_sold (total_cars_eq : total_cars = 500)
                 (mercedes_percentage_eq : mercedes_percentage = 0.10)
                 (toyota_percentage_eq : toyota_percentage = 0.30)
                 (ford_percentage_eq : ford_percentage = 0.20) :
  number_of_bmws_sold = 200 :=
by
  rw [total_cars_eq, mercedes_percentage_eq, toyota_percentage_eq, ford_percentage_eq]
  dsimp only [number_of_bmws_sold, total_cars, bmw_percentage]
  norm_num
  sorry

end bmw_sold_l489_489687


namespace eccentricity_of_ellipse_l489_489167

noncomputable def ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (c_def : c = sqrt (a^2 - b^2)) : ℝ :=
let e := c / a in
if h : (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ x^2 + y^2 = c^2 ∧ (∃ Q_x Q_y : ℝ, (Q_x - c)^2 + Q_y^2 = (c / 2)^2))
then sqrt 3 - 1
else e

theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0) (c : ℝ)
  (c_def : c = sqrt (a^2 - b^2))
  (hPQ_QF2 : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ x^2 + y^2 = c^2 ∧ (∃ Q_x Q_y : ℝ, (Q_x - c)^2 + Q_y^2 = (c / 2)^2)) :
  (ellipse_eccentricity a b c h1 h2 c_def) = sqrt 3 - 1 := by
  sorry

end eccentricity_of_ellipse_l489_489167


namespace rotation_reflection_matrix_l489_489787

/--
The matrix that first rotates a point by 150 degrees counter-clockwise around the origin 
and then reflects it over the x-axis is given by:
\[
\begin{pmatrix}
-\sqrt{3} / 2 & 1 / 2 \\
1 / 2 & \sqrt{3} / 2
\end{pmatrix}
\]
-/
theorem rotation_reflection_matrix :
  let θ := 150 * Real.pi / 180,
      R := Matrix.fromBlocks (Matrix.fromBlocks (Matrix.scalar 2 (Real.cos θ)) (Matrix.scalar 2 (-(Real.sin θ)))
                                               (-(Matrix.scalar 2 (Real.sin θ))) (Matrix.scalar 2 (Real.cos θ))),
      M_x := Matrix.fromBlocks (Matrix.scalar 2 1) (Matrix.zero 2 2)
                               (Matrix.zero 2 2) (Matrix.scalar 2 (-1))
  in R * M_x = Matrix.fromBlocks
                          (Matrix.scalar 2 (-Real.sqrt 3 / 2)) (Matrix.scalar 2 (1 / 2))
                          (Matrix.scalar 2 (1 / 2)) (Matrix.scalar 2 (Real.sqrt 3 / 2)) :=
by
  sorry

end rotation_reflection_matrix_l489_489787


namespace finite_cyclic_even_order_l489_489502

variables {F : Type*} [Field F] [ne_zero (Nat := 2)] 

def finite_order_elements (F : Type*) [Field F] : set F :=
{x : F | x ≠ 0 ∧ ∃ n : ℕ, n > 0 ∧ x^n = 1}

lemma finite_order_elements_is_subgroup (F : Type*) [Field F] :
  is_subgroup (finite_order_elements F) := sorry

theorem finite_cyclic_even_order
  (F : Type*) [Field F] 
  (ne_char_2 : ∀ (x : F), x ≠ 1 → x^2 ≠ 1)
  (T : set F)
  (hT : T = { x ∈ (finite_order_elements F) | x ≠ 0 } ∧ T.finite) : 
  is_cyclic_group T ∧ even T.order := 
sorry

end finite_cyclic_even_order_l489_489502


namespace theme_park_ratio_l489_489566

theorem theme_park_ratio (a c : ℕ) (h_cost_adult : 20 * a + 15 * c = 1600) (h_eq_ratio : a * 28 = c * 59) :
  a / c = 59 / 28 :=
by
  /-
  Proof steps would go here.
  -/
  sorry

end theme_park_ratio_l489_489566


namespace arccos_neg_one_eq_pi_l489_489755

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_eq_pi_l489_489755


namespace matrix_power_identity_l489_489143

-- Define the matrix B
def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![0, 2]]

-- Define the identity matrix I
def I : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 0], ![0, 1]]

-- Prove that B^15 - 3 * B^14 is equal to the given matrix
theorem matrix_power_identity :
  B ^ 15 - 3 • (B ^ 14) = ![![0, 4], ![0, -1]] :=
by
  -- Sorry is used here so the Lean code is syntactically correct
  sorry

end matrix_power_identity_l489_489143


namespace arrangement_count_l489_489719

/-- Given numbers 12, 13, ..., 20, there are 216 ways to arrange them
such that the sum of every three consecutive numbers is a multiple of 3. -/
theorem arrangement_count :
  ∃ count : ℕ, count = 216 ∧
  (∀ (arr : Fin 9 → ℕ) (h : ∀ i, 0 ≤ arr i ∧ arr i < 9),
  (arr 0 + arr 1 + arr 2) % 3 = 0 ∧ (arr 1 + arr 2 + arr 3) % 3 = 0 ∧ 
  (arr 2 + arr 3 + arr 4) % 3 = 0 ∧ (arr 3 + arr 4 + arr 5) % 3 = 0 ∧ 
  (arr 4 + arr 5 + arr 6) % 3 = 0 ∧ (arr 5 + arr 6 + arr 7) % 3 = 0 ∧ 
  (arr 6 + arr 7 + arr 8) % 3 = 0) :=
    ∃ perm_count : ℕ, perm_count = 216 :=
  sorry

end arrangement_count_l489_489719


namespace find_x_coordinate_l489_489485

-- Define the conditions
def line_through_point (x y : ℝ) (m b : ℝ) : Prop := y = m * x + b
def slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

-- Define the problem statement
theorem find_x_coordinate :
  ∀ (x1 x2 y2 : ℝ),
  line_through_point x2 y2 (slope x1 (-6) x2 y2) (y2 - slope x1 (-6) x2 y2 * x2) →
  x1 = 4 →
  x2 = 10 →
  y2 = 3 →
  slope x1 (-6) x2 y2 = 1/2 →
  (∃ x, line_through_point x (-6) (1/2) (-2) ∧ x = -8) :=
by
  intros x1 x2 y2 h_line h_x1 h_x2 h_y2 h_slope
  use -8
  simp [line_through_point, slope] at *
  sorry

end find_x_coordinate_l489_489485


namespace find_x_value_l489_489522

theorem find_x_value
  (y₁ y₂ z₁ z₂ x₁ x w k : ℝ)
  (h₁ : y₁ = 3) (h₂ : z₁ = 2) (h₃ : x₁ = 1)
  (h₄ : y₂ = 6) (h₅ : z₂ = 5)
  (inv_rel : ∀ y z k, x = k * (z / y^2))
  (const_prod : ∀ x w, x * w = 1) :
  x = 5 / 8 :=
by
  -- omitted proof steps
  sorry

end find_x_value_l489_489522


namespace three_digit_numbers_sum_seven_l489_489016

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489016


namespace minimum_value_is_6_l489_489087

noncomputable def minimum_value : ℝ :=
  real.min (λ (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) (h₃ : a + b = 1), 1 / a + 2 / b)

theorem minimum_value_is_6 :
  ∀ (a b : ℝ), (a ≥ b) → (b > 0) → (a + b = 1) → (1 / a + 2 / b = 6) :=
begin
  intros a b h₁ h₂ h₃,
  sorry
end

end minimum_value_is_6_l489_489087


namespace victors_friend_decks_l489_489240

theorem victors_friend_decks:
  ∀ (deck_cost : ℕ) (victor_decks : ℕ) (total_spent : ℕ)
  (friend_decks : ℕ),
  deck_cost = 8 →
  victor_decks = 6 →
  total_spent = 64 →
  (victor_decks * deck_cost + friend_decks * deck_cost = total_spent) →
  friend_decks = 2 :=
by
  intros deck_cost victor_decks total_spent friend_decks hc hv ht heq
  sorry

end victors_friend_decks_l489_489240


namespace sum_of_repeating_decimals_l489_489732

noncomputable def repeating_decimal_4 : ℚ := 4 / 9
noncomputable def repeating_decimal_7 : ℚ := 7 / 9
noncomputable def repeating_decimal_3 : ℚ := 1 / 3

def sum_repeating_decimals : ℚ :=
  repeating_decimal_4 + repeating_decimal_7 - repeating_decimal_3

theorem sum_of_repeating_decimals : sum_repeating_decimals = 8 / 9 := by
  sorry

end sum_of_repeating_decimals_l489_489732


namespace percent_increase_output_per_hour_l489_489915

-- Definitions and conditions
variable (B H : ℝ) -- B is the number of bears per week, H is the number of hours per week

-- Define outputs per hour
def output_per_hour (B H : ℝ) := B / H
def new_output_per_hour (B H : ℝ) := 1.8 * B / (0.9 * H)

-- A mathematical statement to prove the percentage increase of output per hour
theorem percent_increase_output_per_hour (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((new_output_per_hour B H) - (output_per_hour B H)) / (output_per_hour B H) * 100 = 100 :=
by
  sorry

end percent_increase_output_per_hour_l489_489915


namespace num_valid_19_tuples_l489_489790

def S (a : Fin 19 → ℤ) : ℤ := Finset.univ.sum a

noncomputable def satisfies_condition (a : Fin 19 → ℤ) : Prop :=
  ∀ i : Fin 19, (a i)^2 = S a - a i

theorem num_valid_19_tuples : {a : Fin 19 → ℤ // satisfies_condition a}.card = 15506 := by
  sorry

end num_valid_19_tuples_l489_489790


namespace radian_measure_60_degree_l489_489220

-- Definition of the conversion factor
def degree_to_radian_factor := π / 180

-- Statement of the main theorem
theorem radian_measure_60_degree : 60 * degree_to_radian_factor = π / 3 :=
by
  sorry

end radian_measure_60_degree_l489_489220


namespace count_integer_solutions_x_sq_lt_10x_l489_489810

theorem count_integer_solutions_x_sq_lt_10x :
  {x : ℤ | x^2 < 10 * x}.card = 9 :=
sorry

end count_integer_solutions_x_sq_lt_10x_l489_489810


namespace trailing_zeros_30_l489_489317

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

-- Define a function to count the number of times a prime factor (like 5) appears in the factorial
def count_factors (n p : ℕ) : ℕ :=
  if p = 0 then 0 else
  if n = 0 then 0 else
  let rec_count := count_factors (n / p) p
  in n / p + rec_count

-- Theorem: number of trailing zeros in 30!
theorem trailing_zeros_30! : count_factors 30 5 = 7 := 
  sorry

end trailing_zeros_30_l489_489317


namespace exactly_two_succeed_probability_l489_489305

-- Define the probabilities of events A, B, and C decrypting the code
def P_A_decrypts : ℚ := 1/5
def P_B_decrypts : ℚ := 1/4
def P_C_decrypts : ℚ := 1/3

-- Define the probabilities of events A, B, and C not decrypting the code
def P_A_not_decrypts : ℚ := 1 - P_A_decrypts
def P_B_not_decrypts : ℚ := 1 - P_B_decrypts
def P_C_not_decrypts : ℚ := 1 - P_C_decrypts

-- Define the probability that exactly two out of A, B, and C decrypt the code
def P_exactly_two_succeed : ℚ :=
  (P_A_decrypts * P_B_decrypts * P_C_not_decrypts) +
  (P_A_decrypts * P_B_not_decrypts * P_C_decrypts) +
  (P_A_not_decrypts * P_B_decrypts * P_C_decrypts)

-- Prove that this probability is equal to 3/20
theorem exactly_two_succeed_probability : P_exactly_two_succeed = 3 / 20 := by
  sorry

end exactly_two_succeed_probability_l489_489305


namespace log_diff_values_l489_489026

theorem log_diff_values :
  let S := {3, 5, 7, 11}
  (∃ (vals : Finset ℝ), ∀ a b ∈ S, a ≠ b → vals = Finset.image (λ (p : ℝ × ℝ), (Real.log p.1 - Real.log p.2)) (S.product S) ∧ 
    vals.card = 12) :=
by
  sorry

end log_diff_values_l489_489026


namespace minimize_BC_l489_489160

variable (A B C M : Point)
variable (BC : Line)
variable (H : Line)
variable (P : Plane)

-- Conditions
variable (h1 : is_triangle P A B C)
variable (h2 : M ∈ BC)
variable (h3 : H = orthogonal_projection A BC)
variable (h4 : B' = orthogonal_projection M (line_through A C))
variable (h5 : C' = orthogonal_projection M (line_through A B))

-- Conclusion
theorem minimize_BC' : 
  min_length (line_segment B' C') = 
  if H ∈ BC then 
    length (line_segment A H) 
  else 
    min (length (line_segment B C)) (length (line_segment C B)) :=
sorry

end minimize_BC_l489_489160


namespace infinitely_many_coprime_pairs_divisibility_l489_489186

theorem infinitely_many_coprime_pairs_divisibility :
  ∃ᶠ (a b : ℕ) in Filter.atTop, Nat.Coprime a b ∧ a > 1 ∧ b > 1 ∧ ((a + b) ∣ (a ^ b + b ^ a)) :=
by sorry

end infinitely_many_coprime_pairs_divisibility_l489_489186


namespace vector_magnitude_sum_l489_489822

theorem vector_magnitude_sum
  (a b : EuclideanSpace ℝ (Fin 3))
  (h1 : ∥a∥ = 4)
  (h2 : ∥b∥ = 3)
  (h3 : real.angle.cos (real.angle_of_vectors a b) = 1 / 2) :
  ∥a + b∥ = real.sqrt 37 := 
sorry

end vector_magnitude_sum_l489_489822


namespace cubical_room_side_length_l489_489572

theorem cubical_room_side_length (d : ℝ) (sqrt3_approx : ℝ) (h_d : d = 6.92820323027551) (h_sqrt3 : sqrt3_approx = 1.7320508075688772) :
  let a := d / sqrt3_approx in a = 4 :=
by
  have h_a : a = 6.92820323027551 / 1.7320508075688772 := by rw [h_d, h_sqrt3]
  exact h_a

end cubical_room_side_length_l489_489572


namespace baked_by_brier_correct_l489_489970

def baked_by_macadams : ℕ := 20
def baked_by_flannery : ℕ := 17
def total_baked : ℕ := 55

def baked_by_brier : ℕ := total_baked - (baked_by_macadams + baked_by_flannery)

-- Theorem statement
theorem baked_by_brier_correct : baked_by_brier = 18 := 
by
  -- proof will go here 
  sorry

end baked_by_brier_correct_l489_489970


namespace check_roots_l489_489793

noncomputable def roots_of_quadratic_eq (a b : ℂ) : list ℂ :=
  [(-b + complex.sqrt(b ^ 2 - 4 * a * (3 - 4 * complex.i))) / (2 * a),
   (-b - complex.sqrt(b ^ 2 - 4 * a * (3 - 4 * complex.i))) / (2 * a)]

theorem check_roots :
  ∀ (z : ℂ), z^2 + 2 * z + (3 - 4 * complex.i) = 0 ↔ (z = complex.i ∨ z = -3 - 2 * complex.i) :=
begin
  sorry
end

end check_roots_l489_489793


namespace more_white_birds_than_grey_l489_489228

def num_grey_birds_in_cage : ℕ := 40
def num_remaining_birds : ℕ := 66

def num_grey_birds_freed : ℕ := num_grey_birds_in_cage / 2
def num_grey_birds_left_in_cage : ℕ := num_grey_birds_in_cage - num_grey_birds_freed
def num_white_birds : ℕ := num_remaining_birds - num_grey_birds_left_in_cage

theorem more_white_birds_than_grey : num_white_birds - num_grey_birds_in_cage = 6 := by
  sorry

end more_white_birds_than_grey_l489_489228


namespace prime_solution_l489_489798

theorem prime_solution (p q r s : ℕ) (hp : p.prime) (hq : q.prime) (hr : r.prime) (hs : s.prime) : 
  p^4 + q^4 + r^4 + 119 = s^2 → {p, q, r} = {2, 3, 5} ∧ s = 29 := 
by
  sorry

end prime_solution_l489_489798


namespace molecular_weight_of_H2O_l489_489642

theorem molecular_weight_of_H2O : 
  (∀ n : ℕ, n > 0 → molecular_weight(7 * n) = 126 * n → molecular_weight(1) = 18) :=
by
  intro n hn h
  sorry

end molecular_weight_of_H2O_l489_489642


namespace polynomial_expansion_propositions_l489_489029

theorem polynomial_expansion_propositions 
  (a : ℕ → ℤ)
  (h_binomial : ∀ x : ℤ, (1 - 2*x)^2023 = ∑ i in Finset.range 2024, a i * x^i) :
  (∑ i in Finset.range 2024, a i * (-2)^i = 2^2023) ∧
  (∑ i in Finset.range 2024, a i = -1) ∧
  (∑ i in Finset.filter (λ i, odd i) (Finset.range 2024), a i = (3^2023 - 1) / 2) ∧
  (∑ i in Finset.range 2024, a i / 2^i ≠ 0) := 
by 
  sorry

end polynomial_expansion_propositions_l489_489029


namespace arccos_neg_one_eq_pi_l489_489752

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_eq_pi_l489_489752


namespace percent_increase_output_l489_489932

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l489_489932


namespace rectangle_area_l489_489895

-- Define the rectangle ABCD with a diagonal divided into segments
theorem rectangle_area (DE EF FB : ℕ) (hDE : DE = 2) (hEF : EF = 1) (hFB : FB = 1) :
  let DB := DE + EF + FB in
  let area : ℝ := 6.9 in
  (2 * 2 * Real.sqrt 3).round(1) = area :=
by
  let DB := DE + EF + FB
  have hDB : DB = 4 := by simp [hDE, hEF, hFB]
  let AE := Real.sqrt 3
  let tri_area := (1 / 2) * DB * AE
  let rect_area := 2 * tri_area
  have h_rect_area : rect_area = 4 * Real.sqrt 3 := by norm_num [tri_area, AE]
  have h_approx : (4 * Real.sqrt 3).round(1) = 6.9 := by 
    -- Approximation and rounding logic here, potentially introduce intermediate steps
    sorry
  exact h_approx

end rectangle_area_l489_489895


namespace general_term_formula_T_n_upper_bound_l489_489387

/- Define the arithmetic sequence and conditions -/
def is_arithmetic_sequence_positive (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (a (n + 1) = a n + 1) ∧ (a n > 0)

/- Define the sum of terms condition -/
def sum_condition (sum_seq : ℕ → ℚ): Prop :=
  ∀ n : ℕ, n > 0 → (sum_seq n = (n : ℚ) / (2 * (n + 2)))

/- Define the general term verification -/
theorem general_term_formula (a : ℕ → ℕ) (sum_seq : ℕ → ℚ)
  (h1 : is_arithmetic_sequence_positive a)
  (h2 : sum_condition (λ n, 1 / (a n * a (n + 1)))) :
  ∀ n : ℕ, n > 0 → a n = n + 1 :=
sorry

/- Define the sum of first n terms S_n -/
def S_n (a : ℕ → ℕ) (n : ℕ) : ℕ := (n * (n + 3)) / 2

/- Define the sum of reciprocals T_n -/
def T_n (S_n : ℕ → ℕ) (n : ℕ) : ℚ := (2 / 3) * 
  (∑ i in finset.range n, 1 / S_n i)

theorem T_n_upper_bound (a : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → a n = n + 1) :
  ∀ n : ℕ, n > 0 → T_n (S_n a) n < (11 / 9 : ℚ) :=
sorry

end general_term_formula_T_n_upper_bound_l489_489387


namespace vector_sum_l489_489319

def v1 : ℤ × ℤ := (5, -3)
def v2 : ℤ × ℤ := (-2, 4)
def scalar : ℤ := 3

theorem vector_sum : 
  (v1.1 + scalar * v2.1, v1.2 + scalar * v2.2) = (-1, 9) := 
by 
  sorry

end vector_sum_l489_489319


namespace proof_G_eq_BC_eq_D_eq_AB_AC_l489_489901

-- Let's define the conditions of the problem first
variables (A B C O D G F E : Type) [Field A] [Field B] [Field C] [Field O] [Field D] [Field G] [Field F] [Field E]

-- Given triangle ABC with circumcenter O
variable {triangle_ABC: Prop}

-- Given point D on line segment BC
variable (D_on_BC : Prop)

-- Given circle Gamma with diameter OD
variable (circle_Gamma : Prop)

-- Given circles Gamma_1 and Gamma_2 are circumcircles of triangles ABD and ACD respectively
variable (circle_Gamma1 : Prop)
variable (circle_Gamma2 : Prop)

-- Given points F and E as intersection points
variable (intersect_F : Prop)
variable (intersect_E : Prop)

-- Given G as the second intersection point of the circumcircles of triangles BED and DFC
variable (second_intersect_G : Prop)

-- Prove that the condition for point G to be equidistant from points B and C is that point D is equidistant from lines AB and AC
theorem proof_G_eq_BC_eq_D_eq_AB_AC : 
  triangle_ABC ∧ D_on_BC ∧ circle_Gamma ∧ circle_Gamma1 ∧ circle_Gamma2 ∧ intersect_F ∧ intersect_E ∧ second_intersect_G → 
  G_dist_BC ↔ D_dist_AB_AC :=
by
  sorry

end proof_G_eq_BC_eq_D_eq_AB_AC_l489_489901


namespace square_center_sum_l489_489298

noncomputable def sum_of_center_coordinates (A B C D : ℝ × ℝ) : ℝ :=
  let center : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  center.1 + center.2

theorem square_center_sum
  (A B C D : ℝ × ℝ)
  (h1 : 9 = A.1) (h2 : 0 = A.2)
  (h3 : 4 = B.1) (h4 : 0 = B.2)
  (h5 : 0 = C.1) (h6 : 3 = C.2)
  (h7: A.1 < B.1) (h8: A.2 < C.2) :
  sum_of_center_coordinates A B C D = 8 := 
by
  sorry

end square_center_sum_l489_489298


namespace cone_radius_l489_489221

theorem cone_radius (h : ℝ) (V : ℝ) (π : ℝ) (r : ℝ)
    (h_def : h = 21)
    (V_def : V = 2199.114857512855)
    (volume_formula : V = (1/3) * π * r^2 * h) : r = 10 :=
by {
  sorry
}

end cone_radius_l489_489221


namespace general_term_of_sequence_l489_489130

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n, a (n + 2) = a n + 2

theorem general_term_of_sequence (a : ℕ → ℕ) (h : sequence a) : ∀ n, a n = n :=
by { sorry }

end general_term_of_sequence_l489_489130


namespace sum_even_integers_between_400_600_l489_489248

theorem sum_even_integers_between_400_600 : 
  let a := 402,
      d := 2,
      l := 600,
      n := (l - a) / d + 1
  in 
  n = 100 ∧ (n / 2) * (a + l) = 50100 := 
by 
  let a := 402;
  let d := 2;
  let l := 600;
  let n := (l - a) / d + 1;
  show n = 100 ∧ (n / 2) * (a + l) = 50100,
  from sorry

end sum_even_integers_between_400_600_l489_489248


namespace pair1_relationship_pair2_relationship_pair3_relationship_pair4_relationship_l489_489659

-- Define normal vector calculation
def normal_vector (A B C : ℝ) : ℝ × ℝ × ℝ := (A, B, C)

-- Define proportionality check for three values
def proportional (a b c : ℝ) (x y z : ℝ) : Prop :=
  (a / x = b / y) ∧ (b / y = c / z)

-- Define pair 1
def plane1_pair1 : ℝ × ℝ × ℝ := normal_vector 4 (-6) 3
def plane2_pair1 : ℝ × ℝ × ℝ := normal_vector 2 (-3) 1
def pair1_intersect : Prop :=
  ¬ proportional 4 (-6) 3 2 (-3) 1

-- Define pair 2
def plane1_pair2 : ℝ × ℝ × ℝ := normal_vector 6 8 (-4)
def plane2_pair2 : ℝ × ℝ × ℝ := normal_vector 3 4 (-2)
def pair2_parallel : Prop :=
  proportional 6 8 (-4) 3 4 (-2) ∧ (6 / 3 ≠ -6 / 3)

-- Define pair 3
def plane1_pair3 : ℝ × ℝ × ℝ := normal_vector 3 (-6) 3
def plane2_pair3 : ℝ × ℝ × ℝ := normal_vector (-1) 2 (-1)
def pair3_coincide : Prop :=
  proportional 3 (-6) 3 (-1) 2 (-1) ∧ (3 / (-1) = -6 / 2 = 3 / (-1))

-- Define pair 4
def plane1_pair4 : ℝ × ℝ × ℝ := normal_vector 6 0 (-9)
def plane2_pair4 : ℝ × ℝ × ℝ := normal_vector 2 0 (-3)
def pair4_parallel : Prop :=
  proportional 6 0 (-9) 2 0 (-3)

-- Theorem statements
theorem pair1_relationship : pair1_intersect := sorry
theorem pair2_relationship : pair2_parallel := sorry
theorem pair3_relationship : pair3_coincide := sorry
theorem pair4_relationship : pair4_parallel := sorry

end pair1_relationship_pair2_relationship_pair3_relationship_pair4_relationship_l489_489659


namespace coeff_expansion_l489_489483

theorem coeff_expansion : 
  (finset.univ.sum 
    (λ (s : finset (fin 6)),
      (((if s.card = 3 then 1 else 0) * 
        (if s.filter (λ i, i ≥ 3).card = 2 then (-2)^2 else 0) * 
        (if s.filter (λ i, i > 4).card = 1 then 3 else 0)) : ℤ))) =
  720 := 
sorry

end coeff_expansion_l489_489483


namespace bounded_area_l489_489328

-- Define the given equation as a predicate
def equation (x y : ℝ) : Prop := (y - 10) ^ 2 + 2 * x * (y - 10) + 20 * |x| = 200

-- Define the bounded region to calculate its area
noncomputable def bounded_region_area : ℝ :=
  let pos_x_y1 (x : ℝ) := -10 + 2 * x in
  let pos_x_y2 (x : ℝ) := 20 in
  let neg_x_y1 (x : ℝ) := 10 - 2 * x in
  let neg_x_y2 (x : ℝ) := -10 in
  let base := 15 + 5 in  -- Width from x = -5 to x = 15
  let height := 20 - (-10) in  -- Difference between y = 20 and y = -10
  base * height

-- Proving that the bounded region area is 600
theorem bounded_area : bounded_region_area = 600 := by 
  sorry

end bounded_area_l489_489328


namespace polygon_sides_l489_489620

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489620


namespace find_value_of_a_l489_489569

theorem find_value_of_a (a : ℝ) :
  (∀ x y : ℝ, y = a * (x - 3)^2 + 2 → (x, y) = (3, 2) ∨ (x, y) = (-2, -18)) →
  a = -4 / 5 :=
by
  assume h
  have : (-18 : ℝ) = a * (( -2 : ℝ ) - 3)^2 + 2,
  {
    specialize h (-2) (-18),
    exact h rfl,
  }
  linarith

end find_value_of_a_l489_489569


namespace polynomial_expansion_identity_l489_489037

theorem polynomial_expansion_identity :
  ∃ a : Fin 2019 → ℕ, 
  ((x + 1)^2 * (x + 2)^2016 = 
    (Finset.sum (Finset.range 2019) (λ i, (a i) * (x + 2)^i)) ∧ 
    (Finset.sum (Finset.range 2018) (λ i, (a i) / 2^(i + 1)) = (1 / 2)^2018)) :=
sorry

end polynomial_expansion_identity_l489_489037


namespace longest_side_length_l489_489365

open Set

def region := {p : ℝ × ℝ | p.1 + p.2 ≤ 4 ∧ 2 * p.1 + p.2 ≥ 0 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

theorem longest_side_length : ∃ (p₁ p₂ : ℝ × ℝ), p₁ ∈ region ∧ p₂ ∈ region ∧ (dist p₁ p₂ = 4) ∧ 
  ∀ (p₃ p₄ : ℝ × ℝ), p₃ ∈ region → p₄ ∈ region → dist p₃ p₄ ≤ 4 :=
by
  sorry

end longest_side_length_l489_489365


namespace chord_length_l489_489364

theorem chord_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let c := Real.sqrt (a^2 + b^2) in
  let y_int := b^2 / a in
  ((c - c)^2 + ((y_int - (-y_int))^2)) = (2 * b^2 / a)^2 :=
by
  sorry

end chord_length_l489_489364


namespace qualified_flour_l489_489707

-- Define the acceptable weight range
def acceptable_range (w : ℝ) : Prop :=
  24.75 ≤ w ∧ w ≤ 25.25

-- Define the weight options
def optionA : ℝ := 24.70
def optionB : ℝ := 24.80
def optionC : ℝ := 25.30
def optionD : ℝ := 25.51

-- The statement to be proved
theorem qualified_flour : acceptable_range optionB ∧ ¬acceptable_range optionA ∧ ¬acceptable_range optionC ∧ ¬acceptable_range optionD :=
by
  sorry

end qualified_flour_l489_489707


namespace five_digit_with_4_or_7_l489_489429

def five_digit_numbers := {n : ℕ | 10000 ≤ n ∧ n ≤ 99999}

def numbers_without_4_or_7 : ℕ := 7 * 8^4

theorem five_digit_with_4_or_7 : ∃ count : ℕ, count = (90000 - numbers_without_4_or_7) ∧ count = 61328 := by
  use 61328
  simp [numbers_without_4_or_7]
  sorry

end five_digit_with_4_or_7_l489_489429


namespace solve_for_xy_l489_489447

theorem solve_for_xy (x y : ℝ) (h : 2 * x - 3 ≤ Real.log (x + y + 1) + Real.log (x - y - 2)) : x * y = -9 / 4 :=
by sorry

end solve_for_xy_l489_489447


namespace half_angle_quadrant_l489_489060

theorem half_angle_quadrant (α : ℝ) (k : ℤ) (h : 2 * k * real.pi + real.pi < α ∧ α < 2 * k * real.pi + 3 / 2 * real.pi) :
  (∃ m : ℤ, 2 * m * real.pi + real.pi / 2 < α / 2 ∧ α / 2 < 2 * m * real.pi + real.pi) ∨
  (∃ n : ℤ, 2 * n * real.pi + real.pi < α / 2 ∧ α / 2 < 2 * n * real.pi + 3 / 2 * real.pi) :=
sorry

end half_angle_quadrant_l489_489060


namespace polygon_properties_l489_489216

theorem polygon_properties
  (n : ℕ)
  (h_exterior_angle : 360 / 20 = n)
  (h_n_sides : n = 18) :
  (180 * (n - 2) = 2880) ∧ (n * (n - 3) / 2 = 135) :=
by
  sorry

end polygon_properties_l489_489216


namespace max_and_min_cos_beta_minus_gamma_values_l489_489836

variables {α β γ : ℝ} {k : ℝ}

def complex_number (magnitude : ℝ) (arg : ℝ) : ℂ :=
  magnitude * (complex.exp (arg * complex.I))

def z1 := complex_number 1 α
def z2 := complex_number k β
def z3 := complex_number (2 - k) γ

-- Defining the condition
def sum_is_zero : Prop := z1 + z2 + z3 = 0

-- Define the maximum value of cos (β - γ)
def cos_beta_minus_gamma_max : ℝ := -1/2

-- Theorem statement
theorem max_and_min_cos_beta_minus_gamma_values 
  (hz : sum_is_zero) :
  k = 1 + sqrt 7 / 2 ∨ k = 1 - sqrt 7 / 2 ∧ cos (β - γ) = cos_beta_minus_gamma_max :=
sorry

end max_and_min_cos_beta_minus_gamma_values_l489_489836


namespace locus_of_P_is_circle_l489_489827

def point_locus_is_circle (x y : ℝ) : Prop :=
  10 * real.sqrt ((x - 1)^2 + (y - 2)^2) = abs (3 * x - 4 * y)

theorem locus_of_P_is_circle :
  ∀ (x y : ℝ), point_locus_is_circle x y → (distance (x, y) (1, 2) = r) := 
sorry

end locus_of_P_is_circle_l489_489827


namespace remainder_of_S_div_9_l489_489506

open BigOperators

def S : ℕ := ∑ k in finset.range 28 \ {0}, nat.choose 27 k

theorem remainder_of_S_div_9 : S % 9 = 7 := 
by sorry

end remainder_of_S_div_9_l489_489506


namespace max_roses_l489_489542

theorem max_roses (budget : ℝ) (indiv_price : ℝ) (dozen_1_price : ℝ) (dozen_2_price : ℝ) (dozen_5_price : ℝ) (hundred_price : ℝ) 
  (budget_eq : budget = 1000) (indiv_price_eq : indiv_price = 5.30) (dozen_1_price_eq : dozen_1_price = 36) 
  (dozen_2_price_eq : dozen_2_price = 50) (dozen_5_price_eq : dozen_5_price = 110) (hundred_price_eq : hundred_price = 180) : 
  ∃ max_roses : ℕ, max_roses = 548 :=
by
  sorry

end max_roses_l489_489542


namespace students_taking_German_l489_489461

-- Defining the conditions
variables (students_total students_French students_both students_neither : ℕ)
variables (students_German : ℕ)

-- Total number of students
def condition1 : students_total = 60 := sorry

-- Number of students taking French
def condition2 : students_French = 41 := sorry

-- Number of students taking both French and German
def condition3 : students_both = 9 := sorry

-- Number of students not taking either course
def condition4 : students_neither = 6 := sorry

-- The number of students taking German is 22
theorem students_taking_German : students_German = 22 :=
by
  have h1 : students_total = 60 := condition1
  have h2 : students_French = 41 := condition2
  have h3 : students_both = 9 := condition3
  have h4 : students_neither = 6 := condition4

  -- Calculate supporting intermediate values (all other steps)
  have students_at_least_one := students_total - students_neither
  have students_only_French := students_French - students_both
  have students_German_or_both := students_at_least_one - students_only_French
  have students_only_German := students_German_or_both - students_both
  have students_German := students_only_German + students_both

  -- Assert the result
  -- From the context, asserting from intermediate supporting steps (students_German) == 22
  exact congrArg2 (λ (_ : students_total = 60) (_ : students_neither = 6), students_German)
      sorry

end students_taking_German_l489_489461


namespace x_coordinate_of_point_P_l489_489843

theorem x_coordinate_of_point_P :
  ∃ x y : ℝ, (x^2 / 16 - y^2 / 9 = 1) ∧ 
             ∃ d : ℝ, d = (1 / 2) * (|sqrt (x^2 + y^2) - sqrt ((x - 5)^2 + y^2)|)  ∧ 
                    x = -((16 / 5) + d) :=
sorry

end x_coordinate_of_point_P_l489_489843


namespace max_tickets_l489_489174

theorem max_tickets (ticket_cost : ℕ) (total_money : ℕ) (n : ℕ) :
  ticket_cost = 15 → total_money = 120 → ticket_cost * n ≤ total_money → n ≤ 8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  have : n ≤ 120 / 15 := le_div_iff_mul_le' zero_lt_fifteen
  norm_num
  sorry

end max_tickets_l489_489174


namespace odd_numbers_in_A_even_4k_minus_2_not_in_A_product_in_A_l489_489159

def is_in_A (a : ℤ) : Prop := ∃ (x y : ℤ), a = x^2 - y^2

theorem odd_numbers_in_A :
  ∀ (n : ℤ), n % 2 = 1 → is_in_A n :=
sorry

theorem even_4k_minus_2_not_in_A :
  ∀ (k : ℤ), ¬ is_in_A (4 * k - 2) :=
sorry

theorem product_in_A :
  ∀ (a b : ℤ), is_in_A a → is_in_A b → is_in_A (a * b) :=
sorry

end odd_numbers_in_A_even_4k_minus_2_not_in_A_product_in_A_l489_489159


namespace average_temperature_for_july_4th_l489_489529

def avg_temperature_july_4th : ℤ := 
  let temperatures := [90, 90, 90, 79, 71]
  let sum := List.sum temperatures
  sum / temperatures.length

theorem average_temperature_for_july_4th :
  avg_temperature_july_4th = 84 := 
by
  sorry

end average_temperature_for_july_4th_l489_489529


namespace ratio_of_weights_l489_489942

def initial_weight : ℝ := 2
def weight_after_brownies (w : ℝ) : ℝ := w * 3
def weight_after_more_jelly_beans (w : ℝ) : ℝ := w + 2
def final_weight : ℝ := 16
def weight_before_adding_gummy_worms : ℝ := weight_after_more_jelly_beans (weight_after_brownies initial_weight)

theorem ratio_of_weights :
  final_weight / weight_before_adding_gummy_worms = 2 := 
by
  sorry

end ratio_of_weights_l489_489942


namespace remainder_of_eggs_is_2_l489_489336

-- Define the number of eggs each person has
def david_eggs : ℕ := 45
def emma_eggs : ℕ := 52
def fiona_eggs : ℕ := 25

-- Define total eggs and remainder function
def total_eggs : ℕ := david_eggs + emma_eggs + fiona_eggs
def remainder (a b : ℕ) : ℕ := a % b

-- Prove that the remainder of total eggs divided by 10 is 2
theorem remainder_of_eggs_is_2 : remainder total_eggs 10 = 2 := by
  sorry

end remainder_of_eggs_is_2_l489_489336


namespace range_of_a_exists_perpendicular_tangent_l489_489963

theorem range_of_a_exists_perpendicular_tangent
  (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ)
  (h_f : ∀ x : ℝ, f x = -exp x - x) 
  (h_g : ∀ x : ℝ, g x = 3 * a * x + 2 * cos x) :
  ∃ l1 l2, (∀ x, l1 x = f x ∧ tangent_to l1) ∧ (∃ x, l2 x = g x ∧ tangent_to l2 ∧ perpendicular l1 l2) → 
  - (1 / 3) ≤ a ∧ a ≤ (2 / 3) := 
sorry

end range_of_a_exists_perpendicular_tangent_l489_489963


namespace decoy_effect_rational_medium_purchase_l489_489971

structure PopcornOption where
  grams : ℕ
  price : ℕ

def small : PopcornOption := ⟨50, 200⟩
def medium : PopcornOption := ⟨70, 400⟩
def large : PopcornOption := ⟨130, 500⟩

-- Hypothesis: Small, Medium and Large options as described.
def options : List PopcornOption := [small, medium, large]

-- We need a theorem that states the usage of the decoy effect.
theorem decoy_effect (o : List PopcornOption) :
  (o = options) →
  (medium.grams < large.grams ∧ medium.price < large.price ∧ 
   (small.price < medium.price ∧ small.grams < medium.grams)) →
  (∃ d : PopcornOption, d = medium ∧ d ≠ small ∧ d ≠ large) →
  (∃ better_option : PopcornOption, better_option = large ∧
    better_option.price - medium.price ≤ 100 ∧
    better_option.grams - medium.grams ≥ 60) :=
begin
  intros hopts hcomp hdc,
  sorry
end

-- Rationality of buying medium-sized popcorn under certain conditions.
theorem rational_medium_purchase (o : List PopcornOption) :
  (o = options) →
  (∃ budget : ℕ, budget = 500 ∧ ∃ drink_price, drink_price = 100 ∧ 
   (medium.price + drink_price ≤ budget) ∧ (large.price > budget ∨ 
   small.grams < medium.grams)) →
  rational_choice : (PopcornOption → ℕ) (d :=
    if medium.price + drink_price ≤ budget then medium else if small.price ≤ budget then small else large) :=
begin
  intros hopts hbudget,
  sorry
end

end decoy_effect_rational_medium_purchase_l489_489971


namespace problem_statement_l489_489503

-- Given conditions

variables (n : ℕ) (a : Fin (2 * n + 1) → ℝ)

-- Assume n is a positive integer
axiom n_pos : 0 < n

-- Assume a₁, a₂, ..., a₂n+1 are positive reals
axiom a_pos : ∀ i, 0 < a i

-- Definition of b_k as given
noncomputable def b (k : Fin (2 * n + 1)) : ℝ :=
  Finset.univ.sup (λ m : Fin (n + 1), 
    (1 : ℝ) / (2 * m.val + 1) * (Finset.Ico (k.val - m.val) (k.val + m.val + 1)).sum (λ i, a (i % (2 * n + 1))))

-- The Lean proposition to be proved
theorem problem_statement : 
  Finset.filter (λ k : Fin (2 * n + 1), b n a k ≥ 1) Finset.univ.card ≤ 2 * (Finset.univ.sum (λ i, a i)) :=
sorry

end problem_statement_l489_489503


namespace simon_candies_l489_489998

-- Defining the conditions Simon's actions over the days.

def first_day (x : ℝ) : ℝ := (3/4) * x - 3
def second_day (x : ℝ) : ℝ := (3/8) * x - (13 / 2)
def third_day (x : ℝ) : ℝ := (3/32) * x - (61 / 8)
def fourth_day (x : ℝ) : ℝ := fourth_day := 4

theorem simon_candies (x : ℝ) (h1 : fourth_day x) : 
  (3/32) * x - (61 / 8) = 4 → x = 124 :=
sorry

end simon_candies_l489_489998


namespace sin_of_sum_of_roots_l489_489084

-- Definition of the function f(x)
def f (x : ℝ) : ℝ :=
  3 * sin (2 * x - π / 3) - 2 * cos^2 (x - π / 6) + 1

-- Definition of the function g(x) which is f(x) shifted left by π/6
def g (x : ℝ) : ℝ :=
  3 * sin (2 * (x + π / 6) - π / 3) - 2 * cos^2 ((x + π / 6) - π / 6) + 1

-- Define a to be any given value where g(x) = a has roots x1 and x2 in [0, π/2]
variable (a : ℝ)

-- Define the roots x1 and x2 of g(x) = a in the interval [0, π / 2]
variable (x1 x2 : ℝ)
variable hx1 : 0 ≤ x1 ∧ x1 ≤ π / 2
variable hx2 : 0 ≤ x2 ∧ x2 ≤ π / 2
variable hroots : g x1 = a ∧ g x2 = a

-- Define θ such that cos θ = 3 / sqrt 10 and sin θ = 1 / sqrt 10
def θ : ℝ := acos (3 / sqrt 10)

-- Statement to prove
theorem sin_of_sum_of_roots : sin (2 * x1 + 2 * x2) = - 3 / 5 :=
by
  -- Placeholder for the proof
  sorry

end sin_of_sum_of_roots_l489_489084


namespace set_elements_arithmetic_progressions_same_ratio_l489_489961

open Set

noncomputable def is_arithmetic_progression (seq : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ d : ℝ, ∀ i : ℕ, 1 ≤ i ∧ i < n → seq (i + 1) = seq i + d

theorem set_elements_arithmetic_progressions_same_ratio
  (m n : ℕ)
  (a b : ℕ → ℝ) 
  (hmn : 2 ≤ m ∧ 2 ≤ n) 
  (ha : ∀ i j : ℕ, 1 ≤ i ∧ i < n → a (i + 1) > a i)
  (hb : ∀ i j : ℕ, 1 ≤ i ∧ i < m → b (i + 1) > b i) :
  (∃ S : Set ℝ, S = { x | ∃ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ m ∧ x = a i + b j } ∧ S.card = n + m - 1)
  ↔ (is_arithmetic_progression a n ∧ is_arithmetic_progression b m ∧ ∃ d : ℝ, ∀ i : ℕ, 1 ≤ i ∧ i < n ∧ 1 ≤ i ∧ i < m → (a (i + 1) - a i = b (i + 1) - b i)) := sorry

end set_elements_arithmetic_progressions_same_ratio_l489_489961


namespace isosceles_triangle_roots_l489_489115

theorem isosceles_triangle_roots (k : ℝ) (a b : ℝ) 
  (h1 : a = 2 ∨ b = 2)
  (h2 : a^2 - 6 * a + k = 0)
  (h3 : b^2 - 6 * b + k = 0) :
  k = 9 :=
by
  sorry

end isosceles_triangle_roots_l489_489115


namespace keychain_savings_l489_489523

def cost_per_keychain_10_pack (price : ℝ) (quantity : ℝ) : ℝ := price / quantity
def cost_per_keychain_4_pack (price : ℝ) (quantity : ℝ) : ℝ := price / quantity
def savings_per_keychain (cost_4_pack : ℝ) (cost_10_pack : ℝ) : ℝ := cost_4_pack - cost_10_pack
def savings_for_20_keychains (savings_per_keychain : ℝ) (amount : ℝ) : ℝ := savings_per_keychain * amount

theorem keychain_savings :
    let price_10_pack := 20
    let quantity_10_pack := 10
    let price_4_pack := 12
    let quantity_4_pack := 4
    let amount := 20
    savings_for_20_keychains
        (savings_per_keychain
            (cost_per_keychain_4_pack price_4_pack quantity_4_pack)
            (cost_per_keychain_10_pack price_10_pack quantity_10_pack))
        amount = 20 := 
by
    sorry

end keychain_savings_l489_489523


namespace paco_sweet_cookies_left_l489_489980

theorem paco_sweet_cookies_left (initial_sweet_cookies : ℕ) (eaten_sweet_cookies : ℕ) :
  initial_sweet_cookies = 34 →
  eaten_sweet_cookies = 15 →
  initial_sweet_cookies - eaten_sweet_cookies = 19 :=
by
  intros h1 h2
  rw [h1, h2]
  rfl

end paco_sweet_cookies_left_l489_489980


namespace max_gangsters_is_35_l489_489886

-- Define the problem conditions
noncomputable def gangsters_in_chicago (num_gangs : ℕ) :=
  ∃ (gangster_set : Set (Set ℕ)), 
  (∀ gangster ∈ gangster_set, ∃ (gangs : Set ℕ),
    gangs ⊆ Finset.range num_gangs ∧ gangster = gangs ∧
    (∀ g1 g2 ∈ gangs, g1 ≠ g2 → ¬ gangs_in_conflict g1 g2) ∧
    (∀ g ∈ Finset.range num_gangs, g ∉ gangs → ∃ g' ∈ gangs, gangs_in_conflict g g'))

-- Condition function to check if two gangs are in conflict
constant gangs_in_conflict : ℕ → ℕ → Prop

-- Prove that the maximum number of gangsters is 35 given 36 gangs
theorem max_gangsters_is_35 : gangsters_in_chicago 36 → ∀ S : Set (Set ℕ), gangster_count S → S.card ≤ 35 :=
sorry

end max_gangsters_is_35_l489_489886


namespace black_cars_in_parking_lot_l489_489629

theorem black_cars_in_parking_lot :
  let total_cars := 3000
  let blue_percent := 0.40
  let red_percent := 0.25
  let green_percent := 0.15
  let yellow_percent := 0.10
  let black_percent := 1 - (blue_percent + red_percent + green_percent + yellow_percent)
  let number_of_black_cars := total_cars * black_percent
  number_of_black_cars = 300 :=
by
  sorry

end black_cars_in_parking_lot_l489_489629


namespace kw_price_percentage_l489_489878

theorem kw_price_percentage (A B : ℝ) (Price_KW_start : ℝ) 
    (A_end : ℝ) (B_end : ℝ) :
    Price_KW_start = 1.9 * A → 
    Price_KW_start = 2 * B → 
    A_end = 1.2 * A → 
    B_end = 0.9 * B → 
    Price_KW_start / (A_end + B_end) ≈ 0.9246 :=
by
  intros h1 h2 h3 h4
  sorry

end kw_price_percentage_l489_489878


namespace tennis_players_l489_489462

-- Define the known quantities
def total_people : ℕ := 310
def play_baseball : ℕ := 255
def play_both : ℕ := 94
def not_playing_sport : ℕ := 11

-- Define the proof statement
theorem tennis_players : Σ (T : ℕ), T = 138 :=
  by
  let play_sport := (play_baseball - play_both)
  -- Calculate the total playing at least one sport
  have h1 : total_people - not_playing_sport = T + play_sport := sorry
  let T := total_people - not_playing_sport - play_sport
  have h2 : T = 138 := sorry
  use T
  exact h2

end tennis_players_l489_489462


namespace count_integer_solutions_x_sq_lt_10x_l489_489811

theorem count_integer_solutions_x_sq_lt_10x :
  {x : ℤ | x^2 < 10 * x}.card = 9 :=
sorry

end count_integer_solutions_x_sq_lt_10x_l489_489811


namespace square_area_l489_489721

theorem square_area (l w x : ℝ) (h1 : 2 * (l + w) = 20) (h2 : l = x / 2) (h3 : w = x / 4) :
  x^2 = 1600 / 9 :=
by
  sorry

end square_area_l489_489721


namespace objective_fn_range_l489_489792

-- Given assumptions
variables (x y : ℝ)

-- Condition: 3 ≤ x + y ≤ 5
def constraint (x y : ℝ) : Prop :=
  3 ≤ x + y ∧ x + y ≤ 5

-- Objective function
def objective_fn (x y : ℝ) : ℝ :=
  3 * x + 2 * y

-- Theorem statement
theorem objective_fn_range :
  (∀ x y, constraint x y → 9 ≤ objective_fn x y ∧ objective_fn x y ≤ 15) :=
begin
  intros,
  sorry
end

end objective_fn_range_l489_489792


namespace max_m_decreasing_l489_489445

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem max_m_decreasing (m : ℝ) :
  (∀ x ∈ Ioo (1 : ℝ) m, f x < f (x + 1)) → m ≤ Real.exp 1 :=
by
  sorry

end max_m_decreasing_l489_489445


namespace smallest_perimeter_of_triangle_with_conditions_l489_489668

noncomputable def triangle_with_conditions (A B C : ℝ) :=
∃ a b c R : ℝ, 
  R > 0 ∧ 
  a = 2 * R * sin A ∧ 
  b = 2 * R * sin B ∧ 
  c = 2 * R * sin C ∧ 
  ((sin A * sin B * sin C) / (sin A + sin B + sin C)) = 1 / 4 ∧
  (sqrt 3 / 4) * a * b * sin C = 144 * sqrt 3

theorem smallest_perimeter_of_triangle_with_conditions (A B C : ℝ) :
  triangle_with_conditions A B C → 
  ∃ p : ℝ, p = a + b + c ∧ p = 72 :=
begin 
  sorry
end

end smallest_perimeter_of_triangle_with_conditions_l489_489668


namespace compounding_frequency_l489_489995

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compounding_frequency (P A r t n : ℝ) 
  (principal : P = 6000) 
  (amount : A = 6615)
  (rate : r = 0.10)
  (time : t = 1) 
  (comp_freq : n = 2) :
  compound_interest P r n t = A := 
by 
  simp [compound_interest, principal, rate, time, comp_freq, amount]
  -- calculations and proof omitted
  sorry

end compounding_frequency_l489_489995


namespace concurrency_BB_l489_489165

variables {A B C B' C' H H' : Type} 
[geometry A] [geometry B] [geometry C] [geometry B'] [geometry C'] [geometry H] [geometry H']

axiom is_triangle : is_triangle ABC
axiom circle : passes_through [B, C]
axiom intersects_AB : intersects circle AB C'
axiom intersects_AC : intersects circle AC B'
axiom orthocenter_ABC : orthocenter ABC = H
axiom orthocenter_AB'C' : orthocenter AB'C' = H'

theorem concurrency_BB'_CC'_HH' : concurrent BB' CC' HH' :=
sorry

end concurrency_BB_l489_489165


namespace jason_total_hours_l489_489494

variables (hours_after_school hours_total : ℕ)

def earnings_after_school := 4 * hours_after_school
def earnings_saturday := 6 * 8
def total_earnings := earnings_after_school + earnings_saturday

theorem jason_total_hours :
  4 * hours_after_school + earnings_saturday = 88 →
  hours_total = hours_after_school + 8 →
  total_earnings = 88 →
  hours_total = 18 :=
by
  intros h1 h2 h3
  sorry

end jason_total_hours_l489_489494


namespace increase_in_output_with_assistant_l489_489921

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l489_489921


namespace puzzles_sold_eq_36_l489_489558

def n_science_kits : ℕ := 45
def n_puzzles : ℕ := n_science_kits - 9

theorem puzzles_sold_eq_36 : n_puzzles = 36 := by
  sorry

end puzzles_sold_eq_36_l489_489558


namespace polygon_sides_l489_489603

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l489_489603


namespace sqrt_inequality_sqrt_equality_l489_489987

theorem sqrt_inequality (x : ℝ) (hx : x > 0) : 
  sqrt (1 / (3 * x + 1)) + sqrt (x / (x + 3)) ≥ 1 := 
sorry

theorem sqrt_equality (x : ℝ) (hx : x = 1) : 
  sqrt (1 / (3 * x + 1)) + sqrt (x / (x + 3)) = 1 := 
sorry

end sqrt_inequality_sqrt_equality_l489_489987


namespace curve_touches_all_Ca_l489_489767

theorem curve_touches_all_Ca (a : ℝ) (h : a > 0) : ∃ C : ℝ → ℝ, ∀ x y, (y - a^2)^2 = x^2 * (a^2 - x^2) → y = C x ∧ C x = 3 * x^2 / 4 :=
sorry

end curve_touches_all_Ca_l489_489767


namespace circumscribed_circle_radius_correct_l489_489700

-- Define necessary variables and conditions
variables (θ : ℝ) (hθ : 0 < θ ∧ θ < π)

-- Define the given circle's radius
noncomputable def given_circle_radius : ℝ := 6

-- Define the expression for the radius of the circumscribed circle
noncomputable def circumscribed_circle_radius : ℝ := 3 * (Real.sec (θ / 2))

-- The theorem to be proven
theorem circumscribed_circle_radius_correct :
  ∀ (θ : ℝ), 0 < θ ∧ θ < π → circumscribed_circle_radius θ = 3 * (Real.sec (θ / 2)) :=
by
  sorry

end circumscribed_circle_radius_correct_l489_489700


namespace volume_of_pyramid_is_one_sixth_l489_489664

noncomputable def volume_of_pyramid_ABCH (cube_volume : ℚ) : ℚ :=
let edge_length := 1 in -- Since cube volume is 1, each edge length is 1
let base_area := (1 / 2) * edge_length * edge_length in
let height := edge_length in
(1 / 3) * base_area * height

theorem volume_of_pyramid_is_one_sixth (cube_volume : ℚ) (h_cube_volume : cube_volume = 1) : 
  volume_of_pyramid_ABCH cube_volume = 1 / 6 := by
  -- We are given the cube has volume 1
  have edge_length : ℚ := 1
  have base_area : ℚ := (1 / 2) * 1 * 1
  have height : ℚ := 1
  show (1 / 3) * base_area * height = 1 / 6
  sorry

end volume_of_pyramid_is_one_sixth_l489_489664


namespace firefighter_hourly_pay_l489_489688

theorem firefighter_hourly_pay 
  (hours_per_week : ℕ) (weeks_per_month : ℕ) (monthly_rent_fraction : ℚ) 
  (monthly_food_expense : ℚ) (monthly_tax : ℚ) (remaining_money : ℚ) 
  (wage_per_hour : ℚ) 
  (weekly_hours_eq : hours_per_week = 48)
  (weeks_eq : weeks_per_month = 4)
  (rent_fraction_eq : monthly_rent_fraction = 1/3) 
  (food_expense_eq : monthly_food_expense = 500) 
  (tax_eq : monthly_tax = 1000)
  (remaining_money_eq : remaining_money = 2340) 
  (monthly_income : ℚ) 
  (monthly_income_eq : monthly_income = weeks_per_month * hours_per_week * wage_per_hour) :

    (monthly_income - monthly_rent_fraction * monthly_income - monthly_food_expense - monthly_tax = remaining_money) 
    → (wage_per_hour = 30) :=
begin 
  sorry 
end

end firefighter_hourly_pay_l489_489688


namespace exists_natural_number_half_true_statements_l489_489913

theorem exists_natural_number_half_true_statements :
  ∃ x : ℕ, ((
    ((x + 1) % 19 = 0).to_nat +
    ((x + 2) % 18 = 0).to_nat +
    ((x + 3) % 17 = 0).to_nat +
    ((x + 4) % 16 = 0).to_nat +
    ((x + 5) % 15 = 0).to_nat +
    ((x + 6) % 14 = 0).to_nat +
    ((x + 7) % 13 = 0).to_nat +
    ((x + 8) % 12 = 0).to_nat +
    ((x + 9) % 11 = 0).to_nat +
    ((x + 10) % 10 = 0).to_nat +
    ((x + 11) % 9 = 0).to_nat +
    ((x + 12) % 8 = 0).to_nat +
    ((x + 13) % 7 = 0).to_nat +
    ((x + 14) % 6 = 0).to_nat +
    ((x + 15) % 5 = 0).to_nat +
    ((x + 16) % 4 = 0).to_nat +
    ((x + 17) % 3 = 0).to_nat +
    ((x + 18) % 2 = 0).to_nat) = 9) := 
λ x, x = 4849825 := sorry

end exists_natural_number_half_true_statements_l489_489913


namespace solve_equation_l489_489953

def integer_part (x : ℝ) : ℤ := int.floor x
def fractional_part (x : ℝ) : ℝ := x - integer_part x

theorem solve_equation (x : ℝ) (h : integer_part x * fractional_part x = 1991 * x) :
  x = 0 ∨ x = -1 / 1992 :=
by
  sorry

end solve_equation_l489_489953


namespace min_value_of_fraction_l489_489107

theorem min_value_of_fraction (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) 
    (h3 : (m * (-3) + n * (-1) + 2 = 0)) 
    (h4 : (m * (-2) + n * 0 + 2 = 0)) : 
    (1 / m + 3 / n) = 6 :=
by
  sorry

end min_value_of_fraction_l489_489107


namespace area_trapezoid_EFBA_l489_489269

open Finset

-- Define the rectangle and the grid
structure Rectangle :=
(area : ℝ)
(width : ℝ)
(height : ℝ)

noncomputable def small_rectangle_area (rect : Rectangle) (grid_width grid_height : ℝ) : ℝ :=
rect.area / (grid_width * grid_height)

-- Define points E and F at specific positions along AD
structure Point :=
(x : ℝ)
(y : ℝ)

def point_E (rect : Rectangle) : Point :=
{ x := 0,
  y := rect.height * 1 / 4 }

def point_F (rect : Rectangle) : Point :=
{ x := 0,
  y := rect.height * 3 / 4 }

-- Define the area function for a trapezoid given the coordinates of its vertices 
def area_trapezoid (A B : Point) : ℝ :=
  let height := A.y - B.y
  let base1 := A.x
  let base2 := B.x
  (base1 + base2) * height / 2

-- Define the problem
def rectangle : Rectangle :=
{ area := 20,
  width := 5,
  height := 4 }

theorem area_trapezoid_EFBA : 
  area_trapezoid (Point.mk 0 0) (Point.mk 0 (rectangle.height * 2 / 4)) = 10 :=
sorry

end area_trapezoid_EFBA_l489_489269


namespace determine_value_of_m_l489_489073

-- Definitions for the problem conditions
def ellipse_eqn_major_axis_length (a b : ℝ) : Prop := 
  b = 8 / 2 * sqrt a

theorem determine_value_of_m (m : ℝ) (h : ellipse_eqn_major_axis_length m 8) : m = 16 := 
by
  sorry

end determine_value_of_m_l489_489073


namespace _l489_489046

noncomputable theorem find_prob (X : ℝ → ℝ) (μ σ : ℝ) (hX : ∀ x, X x ~ ℕ (μ, σ^2))
  (hμ : μ = 2) 
  (P_X_lt_4 : P(X < 4) = 0.8) :
  P(0 < X < 2) = 0.3 :=
sorry

end _l489_489046


namespace jam_jars_weight_l489_489492

noncomputable def jars_weight 
    (initial_suitcase_weight : ℝ) 
    (perfume_weight_oz : ℝ) (num_perfume : ℕ)
    (chocolate_weight_lb : ℝ)
    (soap_weight_oz : ℝ) (num_soap : ℕ)
    (total_return_weight : ℝ)
    (oz_to_lb : ℝ) : ℝ :=
  initial_suitcase_weight 
  + (num_perfume * perfume_weight_oz) / oz_to_lb 
  + chocolate_weight_lb 
  + (num_soap * soap_weight_oz) / oz_to_lb

theorem jam_jars_weight
    (initial_suitcase_weight : ℝ := 5)
    (perfume_weight_oz : ℝ := 1.2) (num_perfume : ℕ := 5)
    (chocolate_weight_lb : ℝ := 4)
    (soap_weight_oz : ℝ := 5) (num_soap : ℕ := 2)
    (total_return_weight : ℝ := 11)
    (oz_to_lb : ℝ := 16) :
    jars_weight initial_suitcase_weight perfume_weight_oz num_perfume
      chocolate_weight_lb soap_weight_oz num_soap total_return_weight oz_to_lb + (jars_weight initial_suitcase_weight perfume_weight_oz num_perfume
      chocolate_weight_lb soap_weight_oz num_soap total_return_weight oz_to_lb) = 1 :=
by
  sorry

end jam_jars_weight_l489_489492


namespace moles_of_ethane_used_l489_489789

-- Definitions based on conditions
def ethane : Type := Nat  -- Represents moles of Ethane (C2H6)
def chlorine : Type := Nat  -- Represents moles of Chlorine (Cl2)
def chloroethane : Type := Nat  -- Represents moles of Chloroethane (C2H5Cl)
def HCl : Type := Nat  -- Represents moles of Hydrogen Chloride (HCl)

-- Reaction relationship based on the balanced equation
def reaction (e : ethane) (cl : chlorine) (ce : chloroethane) (hcl : HCl) : Prop :=
  e = cl ∧ cl = ce ∧ ce = hcl

-- Theorem to prove that 1 mole of chlorine reacting to produce 1 mole of chloroethane implies 1 mole of ethane used
theorem moles_of_ethane_used (e : ethane) (cl : chlorine) (ce : chloroethane) (hcl : HCl) : 
  reaction e cl ce hcl → cl = 1 → ce = 1 → e = 1 :=
by
  sorry

end moles_of_ethane_used_l489_489789


namespace coins_amount_correct_l489_489775

-- Definitions based on the conditions
def cost_of_flour : ℕ := 5
def cost_of_cake_stand : ℕ := 28
def amount_given_in_bills : ℕ := 20 + 20
def change_received : ℕ := 10

-- Total cost of items
def total_cost : ℕ := cost_of_flour + cost_of_cake_stand

-- Total money given
def total_money_given : ℕ := total_cost + change_received

-- Amount given in loose coins
def loose_coins_given : ℕ := total_money_given - amount_given_in_bills

-- Proposition statement
theorem coins_amount_correct : loose_coins_given = 3 := by
  sorry

end coins_amount_correct_l489_489775


namespace sequence_sum_formula_l489_489158

theorem sequence_sum_formula (n : ℕ) (h : n > 0) : 
  let x := λ k : ℕ, (3 + (k-1) * k / 2) in
  ∑ i in Finset.range n, x (i + 1) = n * (n^2 + 3 * n + 18) / 6 :=
by sorry

end sequence_sum_formula_l489_489158


namespace circle_intersection_property_l489_489163

theorem circle_intersection_property (n : ℕ) (h : 2 ≤ n) :
  let k := 3 * n^2 - 10 * n + 10 in
  ∃ circles : Fin k → metric.ball ℝ (0 : ℝ) 1,
      (∃ S : Finset (Fin k), S.card = n ∧ ∀ i ∈ S, ∀ j ∈ S, i ≠ j → disjoint (circles i) (circles j)) ∨
      (∃ T : Finset (Fin k), T.card = n ∧ ∀ i ∈ T, ∀ j ∈ T, i ≠ j → ¬ disjoint (circles i) (circles j)) :=
by
  sorry

end circle_intersection_property_l489_489163


namespace shares_ratio_l489_489543

theorem shares_ratio (A B C : ℝ)
  (h_total : A + B + C = 116000)
  (h_A : A = 29491.525423728814)
  (h_ratio_BC : B / C = 5 / 6)
  : A / B = 3 / 4 :=
by
  have h1 : 29491.525423728814 + B + C = 116000, from sorry,
  have h2 : C = 6 / 5 * B, from sorry,
  have h3 : 11 * C = 86508.474576271186, from sorry,
  have h_x : C = 7864.406779652835, from sorry,
  have h_B : B = 5 * 7864.406779652835, from sorry,
  have h_B_val : B = 39322.033898264175, from sorry,
  have : 29491.525423728814 / 39322.033898264175 = 3 / 4, from sorry,
  exact this

end shares_ratio_l489_489543


namespace tv_cost_l489_489966

def original_savings : ℝ := 500
def fraction_spent_on_furniture : ℝ := 4 / 5
def amount_spent_on_furniture : ℝ := fraction_spent_on_furniture * original_savings
def amount_left_for_tv : ℝ := original_savings - amount_spent_on_furniture

theorem tv_cost (h: amount_left_for_tv = 100) : amount_left_for_tv = 100 :=
  by
  exact h

end tv_cost_l489_489966


namespace solve_arcsin_arccos_l489_489547

open Real

theorem solve_arcsin_arccos (x : ℝ) (h_condition : - (1 / 2 : ℝ) ≤ x ∧ x ≤ 1 / 2) :
  arcsin x + arcsin (2 * x) = arccos x ↔ x = 0 :=
sorry

end solve_arcsin_arccos_l489_489547


namespace number_of_integer_solutions_l489_489806

theorem number_of_integer_solutions (x : ℤ) : 
  (x^2 < 10 * x) → {x | (x^2 < 10 * x)}.finite
    ∧ {x | (x^2 < 10 * x)}.to_finset.card = 9 :=
by
  sorry

end number_of_integer_solutions_l489_489806


namespace perpendicular_lines_sufficient_but_not_necessary_l489_489426

theorem perpendicular_lines_sufficient_but_not_necessary (a : ℝ) :
  let l1 := λ x y : ℝ, a * x + (a + 1) * y + 1 = 0,
      l2 := λ x y : ℝ, x + a * y + 2 = 0,
      slope1 := -a / (a + 1),
      slope2 := -1 / a
  in (slope1 * slope2 = -1) ↔ a = -2 :=
begin
  sorry
end

end perpendicular_lines_sufficient_but_not_necessary_l489_489426


namespace simplify_expression_l489_489545

open Real

theorem simplify_expression (α : ℝ) : 
  (cos (4 * α - π / 2) * sin (5 * π / 2 + 2 * α)) / ((1 + cos (2 * α)) * (1 + cos (4 * α))) = tan α :=
by
  sorry

end simplify_expression_l489_489545


namespace catalyst_second_addition_is_882_l489_489681

-- Constants for the problem
def lower_bound : ℝ := 500
def upper_bound : ℝ := 1500
def golden_ratio_method : ℝ := 0.618

-- Calculated values
def first_addition : ℝ := lower_bound + golden_ratio_method * (upper_bound - lower_bound)
def second_bound : ℝ := first_addition - lower_bound
def second_addition : ℝ := lower_bound + golden_ratio_method * second_bound

theorem catalyst_second_addition_is_882 :
  lower_bound = 500 → upper_bound = 1500 → golden_ratio_method = 0.618 → second_addition = 882 := by
  -- Proof goes here
  sorry

end catalyst_second_addition_is_882_l489_489681


namespace solve_for_a_l489_489034

theorem solve_for_a (f : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x, f (x^3) = Real.log x / Real.log a)
  (h2 : f 8 = 1) :
  a = 2 :=
sorry

end solve_for_a_l489_489034


namespace number_of_terms_divisibility_l489_489557

theorem number_of_terms_divisibility : 
    let seq := { n | 1 ≤ n ∧ n ≤ 2019 ∧ (n % 5 = 2 ∧ n % 7 = 2) } in 
    seq.card = 58 :=
by sorry

end number_of_terms_divisibility_l489_489557


namespace age_ratio_holds_l489_489231

variables (e s : ℕ)

-- Conditions based on the problem statement
def condition_1 : Prop := e - 3 = 2 * (s - 3)
def condition_2 : Prop := e - 5 = 3 * (s - 5)

-- Proposition to prove that in 1 year, the age ratio will be 3:2
def age_ratio_in_one_year : Prop := (e + 1) * 2 = (s + 1) * 3

theorem age_ratio_holds (h1 : condition_1 e s) (h2 : condition_2 e s) : age_ratio_in_one_year e s :=
by {
  sorry
}

end age_ratio_holds_l489_489231


namespace locus_of_point_corresponding_to_z_on_circle_locus_of_point_corresponding_to_z_on_line_l489_489510

def complex_circle_condition (z a b c d w₀ r w : ℂ) : Prop :=
  (a ≠ 0) ∧ (a * d ≠ b * c) ∧
  ∃ (w : ℂ), w = (c * z + d) / (a * z + b) ∧ 
  (abs (w - w₀) = r) ∧ (
  (c - a * w₀ ≠ 0 → 
    ∃(z1 z2 : ℂ), z1 = (b * w₀ - d) / (c - a * w₀) ∧ z2 = - b / a ∧ 
    (λ = r * abs a / abs (c - a * w₀) → abs (z - z1) = λ * abs (z + z2)) ∧ 
    (λ = 1 → abs (z - z1) = abs (z - z2))) ∨ 
  (c - a * w₀ = 0 → abs (z + b / a) = 1 / (r * abs a) * abs (d - b * w₀)))

def complex_line_condition (z a b c d α β w : ℂ) : Prop :=
  (a ≠ 0) ∧ (a * d ≠ b * c) ∧ 
  ∃ (w : ℂ), w = (c * z + d) / (a * z + b) ∧ 
  (w - α).abs = (w - β).abs ∧ 
  ∃ (z1 z2 : ℂ), z1 = (b * α - d) / (c - a * α) ∧ z2 = (b * β - d) / (c - a * β) ∧ 
  ∃(λ : ℝ), λ = abs (c - a * β) / abs (c - a * α) →
    (λ = 1 → abs (z - z1) = abs (z - z2)) ∧ 
    (λ ≠ 1 → abs (z - z1) = λ * abs (z - z2))

theorem locus_of_point_corresponding_to_z_on_circle (a b c d w₀ r z: ℂ) :
complex_circle_condition z a b c d w₀ r (c * z + d) / (a * z + b) :=
sorry

theorem locus_of_point_corresponding_to_z_on_line (a b c d α β z: ℂ) :
complex_line_condition z a b c d α β (c * z + d) / (a * z + b) :=
sorry

end locus_of_point_corresponding_to_z_on_circle_locus_of_point_corresponding_to_z_on_line_l489_489510


namespace range_of_a_l489_489880

theorem range_of_a (a : ℝ) : 
  let P := ((fun x => x + 2 * a) = (fun x => 2 * x + a + 1)) in 
  P = (a - 1, 3 * a - 1) → ((a - 1)^2 + (3 * a - 1)^2 < 4) → 
  (-1 / 5 < a ∧ a < 1) :=
by
  intros _
  intro ha
  intros hb
  sorry

end range_of_a_l489_489880


namespace tan_sin_identity_l489_489739

theorem tan_sin_identity : 
  (tan (20 * Real.pi / 180))^2 - (sin (20 * Real.pi / 180))^2 / (tan (20 * Real.pi / 180))^2 * (sin (20 * Real.pi / 180))^2 = 1 :=
by 
  sorry

end tan_sin_identity_l489_489739


namespace find_all_n_l489_489782

theorem find_all_n (n : ℕ) : 
  (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % n = 0) ↔ (∃ j : ℕ, n = 3^j) :=
by 
  -- proof goes here
  sorry

end find_all_n_l489_489782


namespace no_net_profit_or_loss_l489_489297

theorem no_net_profit_or_loss (C : ℝ) : 
  let cost1 := C
  let cost2 := C
  let selling_price1 := 1.10 * C
  let selling_price2 := 0.90 * C
  let total_cost := cost1 + cost2
  let total_selling_price := selling_price1 + selling_price2
  let net_profit_loss := (total_selling_price - total_cost) / total_cost * 100
  net_profit_loss = 0 :=
by
  let cost1 := C
  let cost2 := C
  let selling_price1 := 1.10 * C
  let selling_price2 := 0.90 * C
  let total_cost := cost1 + cost2
  let total_selling_price := selling_price1 + selling_price2
  let net_profit_loss := (total_selling_price - total_cost) / total_cost * 100
  sorry

end no_net_profit_or_loss_l489_489297


namespace total_fish_caught_l489_489315

theorem total_fish_caught (C_trips : ℕ) (B_fish_per_trip : ℕ) (C_fish_per_trip : ℕ) (D_fish_per_trip : ℕ) (B_trips D_trips : ℕ) :
  C_trips = 10 →
  B_trips = 2 * C_trips →
  B_fish_per_trip = 400 →
  C_fish_per_trip = B_fish_per_trip * (1 + 2/5) →
  D_trips = 3 * C_trips →
  D_fish_per_trip = C_fish_per_trip * (1 + 50/100) →
  B_trips * B_fish_per_trip + C_trips * C_fish_per_trip + D_trips * D_fish_per_trip = 38800 := 
by
  sorry

end total_fish_caught_l489_489315


namespace opposite_vector_AB_to_BA_l489_489394

noncomputable def A : ℝ × ℝ × ℝ := (-2, 3, 5)
noncomputable def B : ℝ × ℝ × ℝ := (1, -1, -7)

theorem opposite_vector_AB_to_BA :
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3) in
  let BA := (-AB.1, -AB.2, -AB.3) in
  BA = (-3, 4, 12) :=
by
  sorry

end opposite_vector_AB_to_BA_l489_489394


namespace number_of_throwers_l489_489975

theorem number_of_throwers
    (total_players : ℕ)
    (right_handed_players : ℕ)
    (one_third_fraction : ℚ)
    (throwers_are_right_handed : Prop)
    (throwers : ℕ)
    (left_handed_players : ℕ)
    (no_throwers : total_players - throwers)
    (left_handed_players_eq : left_handed_players = ((total_players - throwers) / 3)) :
    (right_handed_players - throwers = 2 * left_handed_players) ∧ 
    total_players = 70 ∧ 
    right_handed_players = 60 ∧ 
    one_third_fraction = 1/3 ∧ 
    throwers_are_right_handed → 
    throwers = 40 := by    
sorry

end number_of_throwers_l489_489975


namespace calculate_x_l489_489218

theorem calculate_x (a b x : ℕ) (h1 : b = 9) (h2 : b - a = 5) (h3 : a * b = 2 * (a + b) + x) : x = 10 :=
by
  sorry

end calculate_x_l489_489218


namespace solution_l489_489066

noncomputable def seq (n : ℕ) : ℚ :=
  if n = 1 ∨ n = 2 then 1 else if n = 3 then 2 else (3 + seq (n-1) * seq (n-2)) / (seq (n-1) - 2)

theorem solution (n : ℕ) : seq n = (5 + 2 * Real.sqrt 5) / 10 * (3 + Real.sqrt 5) / 2 ^ n + (5 - 2 * Real.sqrt 5) / 10 * (3 - Real.sqrt 5) / 2 ^ n := 
by sorry

end solution_l489_489066


namespace election_winners_l489_489466

def population : ℕ := 10000

def perc_adults_over_30 : ℝ := 0.60
def perc_adults_18_30 : ℝ := 0.25
def perc_elderly : ℝ := 0.15

def votes_over_30 : ℕ := (perc_adults_over_30 * population : ℝ).toNat
def votes_18_30 : ℕ := (perc_adults_18_30 * population : ℝ).toNat
def votes_elderly : ℕ := (perc_elderly * population : ℝ).toNat

def perc_vote_over_30 (cand : string) : ℝ :=
  match cand with
  | "A" => 0.25
  | "B" => 0.20
  | "C" => 0.05
  | "D" => 0.30
  | "E" => 0.20
  | _ => 0.0

def perc_vote_18_30 (cand : string) : ℝ :=
  match cand with
  | "A" => 0.22
  | "B" => 0.20
  | "C" => 0.08
  | "D" => 0.40
  | "E" => 0.10
  | _ => 0.0

def perc_vote_elderly (cand : string) : ℝ :=
  match cand with
  | "A" => 0.20
  | "B" => 0.15
  | "C" => 0.10
  | "D" => 0.45
  | "E" => 0.10
  | _ => 0.0

def total_votes (cand : string) : ℕ :=
  ((perc_vote_over_30 cand * votes_over_30 : ℝ).toNat
   + (perc_vote_18_30 cand * votes_18_30 : ℝ).toNat
   + (perc_vote_elderly cand * votes_elderly : ℝ).toNat)

theorem election_winners :
  total_votes "D" = 3475 ∧ total_votes "A" = 2350 :=
sorry

end election_winners_l489_489466


namespace graph_passes_through_fixed_point_l489_489031

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + a^(x-1)

theorem graph_passes_through_fixed_point (a : ℝ) : f a 1 = 5 :=
by
  -- sorry is a placeholder for the proof
  sorry

end graph_passes_through_fixed_point_l489_489031


namespace polygon_sides_l489_489621

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489621


namespace find_sin_cos_find_cos_phi_l489_489095

open Real

variables {θ φ : ℝ}

-- Definition of vectors a and b and the orthogonality condition
def orthogonal_vectors (θ : ℝ) := (sin θ, -2) =ᵥ (1, cos θ) ∧ θ ∈ Ioo 0 (π / 2)

-- First part: Finding sin θ and cos θ
theorem find_sin_cos (h : orthogonal_vectors θ) : 
  sin θ = 2 * sqrt 5 / 5 ∧ cos θ = sqrt 5 / 5 :=
sorry

-- Second part: Finding cos φ
theorem find_cos_phi (h : orthogonal_vectors θ) 
  (Hθ : sin θ = 2 * sqrt 5 / 5 ∧ cos θ = sqrt 5 / 5) 
  (Hφ : 5 * cos (θ - φ) = 3 * sqrt 5 * cos φ ∧ φ ∈ Ioo 0 (π / 2)) : 
  cos φ = sqrt 2 / 2 :=
sorry

end find_sin_cos_find_cos_phi_l489_489095


namespace train_crossing_pole_time_l489_489301

theorem train_crossing_pole_time :
  ∀ (speed_kmph length_m: ℝ), speed_kmph = 160 → length_m = 400.032 → 
  length_m / (speed_kmph * 1000 / 3600) = 9.00072 :=
by
  intros speed_kmph length_m h_speed h_length
  rw [h_speed, h_length]
  -- The proof is omitted as per instructions
  sorry

end train_crossing_pole_time_l489_489301


namespace remainder_of_3_pow_500_mod_17_l489_489255

theorem remainder_of_3_pow_500_mod_17 : (3 ^ 500) % 17 = 13 := 
by
  sorry

end remainder_of_3_pow_500_mod_17_l489_489255


namespace midpoint_CD_l489_489982

theorem midpoint_CD :
  let A : (ℝ × ℝ) := (0, 0),
      B : (ℝ × ℝ) := (2, 3),
      D : (ℝ × ℝ) := (10, 0) in
  (∃ C : (ℝ × ℝ), B = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) →
  ((4 + 10) / 2, (6 + 0) / 2) = (7, 3) := sorry

end midpoint_CD_l489_489982


namespace jane_output_increase_l489_489926

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l489_489926


namespace haley_zoo_pics_l489_489260

-- Define the conditions
def z_museum := 8
def number_deleted := 38
def remaining_pics := 20

-- State the theorem
theorem haley_zoo_pics : ∃ (z_zoo : ℕ), (z_zoo + z_museum - number_deleted = remaining_pics) ↔ z_zoo = 50 :=
by
  existsi 50
  apply Iff.intro
  {
    intro h
    exact h
  }
  {
    intro h
    rw h
    norm_num
   }
  sorry

end haley_zoo_pics_l489_489260


namespace period_of_abs_sine_function_l489_489766

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) :=
  ∀ x : ℝ, f (x + T) = f x ∧ (∀ ε > 0, ε ≠ T → f (x + ε) ≠ f x)

theorem period_of_abs_sine_function :
  smallest_positive_period (λ x, abs (5 * sin (2 * x + π / 3))) (π / 2) :=
  sorry

end period_of_abs_sine_function_l489_489766


namespace symmetric_circle_eq_l489_489392

theorem symmetric_circle_eq (x y : ℝ) :
  (x + 1)^2 + (y - 1)^2 = 1 → x - y = 1 → (x - 2)^2 + (y + 2)^2 = 1 :=
by
  sorry

end symmetric_circle_eq_l489_489392


namespace angle_A_is_pi_div_2_triangle_area_l489_489061

theorem angle_A_is_pi_div_2 (a b c : ℝ) (h : (b - c)^2 = a^2 - b * c) : 
  ∠A = π / 2 := 
by 
  sorry

theorem triangle_area (a b c : ℝ) (h1 : a = 3) (h2 : sin C = 2 * sin B) : 
  area_of_triangle a b c = 1 := 
by 
  sorry

end angle_A_is_pi_div_2_triangle_area_l489_489061


namespace three_digit_numbers_sum_seven_l489_489011

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489011


namespace triangle_probability_l489_489770

theorem triangle_probability :
  let sticks := {3, 4, 6, 8, 10, 12, 15, 18}
  let valid_triplet (a b c : ℤ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)
  let valid_combinations := {{4, 6, 8}, {6, 8, 10}, {8, 10, 12}, {10, 12, 15}}
  let total_combinations := Nat.choose 8 3
  let valid_combinations_count := 4
  let probability := valid_combinations_count / total_combinations
  probability = 1 / 14 :=
by
  let sticks := {3, 4, 6, 8, 10, 12, 15, 18}
  let valid_triplet (a b c : ℤ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)
  let valid_combinations := {{4, 6, 8}, {6, 8, 10}, {8, 10, 12}, {10, 12, 15}}
  let total_combinations := Nat.choose 8 3
  let valid_combinations_count := 4
  let probability := valid_combinations_count / total_combinations
  have : probability = 1 / 14 := sorry
  exact this

end triangle_probability_l489_489770


namespace product_of_roots_of_minimal_polynomial_with_rational_coefficients_l489_489138

noncomputable def minimal_polynomial_with_rational_coefficients (x : ℝ) : Polynomial ℚ :=
  sorry

theorem product_of_roots_of_minimal_polynomial_with_rational_coefficients :
  (minimal_polynomial_with_rational_coefficients (ℝ.cbrt 7 + ℝ.cbrt 14)).roots.prod = 7 :=
by
  sorry

end product_of_roots_of_minimal_polynomial_with_rational_coefficients_l489_489138


namespace divided_scale_length_l489_489699

/-
  The problem definition states that we have a scale that is 6 feet 8 inches long, 
  and we need to prove that when the scale is divided into two equal parts, 
  each part is 3 feet 4 inches long.
-/

/-- Given length conditions in feet and inches --/
def total_length_feet : ℕ := 6
def total_length_inches : ℕ := 8

/-- Convert total length to inches --/
def total_length_in_inches := total_length_feet * 12 + total_length_inches

/-- Proof that if a scale is 6 feet 8 inches long and divided into 2 parts, each part is 3 feet 4 inches --/
theorem divided_scale_length :
  (total_length_in_inches / 2) = 40 ∧ (40 / 12 = 3 ∧ 40 % 12 = 4) :=
by
  sorry

end divided_scale_length_l489_489699


namespace jason_cards_l489_489495

theorem jason_cards :
  (initial_cards - bought_cards = remaining_cards) →
  initial_cards = 676 →
  bought_cards = 224 →
  remaining_cards = 452 :=
by
  intros h1 h2 h3
  sorry

end jason_cards_l489_489495


namespace polygon_sides_l489_489618

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489618


namespace matrix_power_problem_l489_489149

open Matrix
open_locale matrix big_operators

def B : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 4], ![0, 2]]

theorem matrix_power_problem :
  B^15 - 3 • (B^14) = ![![0, 4], ![0, -1]] :=
by
  sorry

end matrix_power_problem_l489_489149


namespace sum_reciprocals_roots_eq_67_l489_489019

noncomputable def polynomial (a b c : ℝ) := (3 * (X^7)) - (291 * (X^6)) + (a * (X^5)) + (b * (X^4)) + (c * (X^2)) + (134 * X) - 2

theorem sum_reciprocals_roots_eq_67 (a b c : ℝ) 
  (h1 : (∃ r : list ℝ, r.length = 7 ∧ polynomial a b c = 0))
  (h2 : (∑ x in h1.some, x = 97))
  : (∑ x in h1.some, x⁻¹) = 67 := sorry

end sum_reciprocals_roots_eq_67_l489_489019


namespace ines_bought_3_pounds_l489_489491

-- Define initial and remaining money of Ines
def initial_money : ℕ := 20
def remaining_money : ℕ := 14

-- Define the cost per pound of peaches
def cost_per_pound : ℕ := 2

-- The total money spent on peaches
def money_spent := initial_money - remaining_money

-- The number of pounds of peaches bought
def pounds_of_peaches := money_spent / cost_per_pound

-- The proof problem
theorem ines_bought_3_pounds :
  pounds_of_peaches = 3 :=
by
  sorry

end ines_bought_3_pounds_l489_489491


namespace polygon_sides_l489_489608

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489608


namespace varphi_value_l489_489105

theorem varphi_value (φ : ℝ) (h₀ : 0 < φ) (h₁ : φ < π):
  (∀ x, sin (2 * x + φ) = sin (2 * (2 * π / 3 - x) + φ)) → φ = 5 * π / 6 :=
by sorry

end varphi_value_l489_489105


namespace max_possible_value_l489_489902

def in_set_0_to_4 (x : ℕ) := x ∈ {0, 1, 2, 3, 4}

def distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem max_possible_value (a b c d : ℕ) (h_distinct : distinct a b c d)
  (ha : in_set_0_to_4 a) (hb : in_set_0_to_4 b)
  (hc : in_set_0_to_4 c) (hd : in_set_0_to_4 d) :
  c * a^(b + d) ≤ 1024 :=
by
  sorry

end max_possible_value_l489_489902


namespace geese_survived_first_year_l489_489175

theorem geese_survived_first_year (eggs : ℕ) (h1 : eggs = 100) 
  (h2 : (1 / 2 : ℝ) * eggs = 50) 
  (h3 : (3 / 4 : ℝ) * 50 = 37.5) 
  (h4 : 37.5.floor = 37)
  (h5 : (2 / 5 : ℝ) * 37 = 14.8)
  (h6 : 14.8.floor = 14) 
  : (14 : ℕ) := 
begin
  sorry
end

end geese_survived_first_year_l489_489175


namespace prob_at_least_two_days_prob_at_least_one_consecutive_l489_489127

noncomputable def accuracy : ℝ := 0.8

/-- The probability that at least 2 days are accurately predicted. -/
theorem prob_at_least_two_days :
  (choose 3 2) * accuracy^2 * (1 - accuracy) + (choose 3 3) * accuracy^3 = 0.896 :=
by sorry

/-- The probability that there is at least one instance of 2 consecutive days being accurately predicted. -/
theorem prob_at_least_one_consecutive :
  2 * accuracy^2 * (1 - accuracy) + accuracy^3 = 0.768 :=
by sorry

end prob_at_least_two_days_prob_at_least_one_consecutive_l489_489127


namespace N_positive_l489_489436

def N (a b : ℝ) : ℝ :=
  4 * a^2 - 12 * a * b + 13 * b^2 - 6 * a + 4 * b + 13

theorem N_positive (a b : ℝ) : N a b > 0 :=
by
  sorry

end N_positive_l489_489436


namespace calculate_final_amount_l489_489362

def initial_amount : ℝ := 7500
def first_year_rate : ℝ := 0.20
def second_year_rate : ℝ := 0.25

def first_year_interest (p : ℝ) (r : ℝ) : ℝ := p * r
def amount_after_first_year (p : ℝ) (i : ℝ) : ℝ := p + i

def second_year_interest (p : ℝ) (r : ℝ) : ℝ := p * r
def amount_after_second_year (p : ℝ) (i : ℝ) : ℝ := p + i

theorem calculate_final_amount :
  let initial : ℝ := initial_amount
  let interest1 : ℝ := first_year_interest initial first_year_rate
  let amount1 : ℝ := amount_after_first_year initial interest1
  let interest2 : ℝ := second_year_interest amount1 second_year_rate
  let final_amount : ℝ := amount_after_second_year amount1 interest2
  final_amount = 11250 := by
  sorry

end calculate_final_amount_l489_489362


namespace B_finishes_work_in_54_days_l489_489657

-- The problem statement rewritten in Lean 4.
theorem B_finishes_work_in_54_days
  (A_eff : ℕ) -- amount of work A can do in one day
  (B_eff : ℕ) -- amount of work B can do in one day
  (work_days_together : ℕ) -- number of days A and B work together to finish the work
  (h1 : A_eff = 2 * B_eff)
  (h2 : A_eff + B_eff = 3)
  (h3 : work_days_together = 18) :
  work_days_together * (A_eff + B_eff) / B_eff = 54 :=
by
  sorry

end B_finishes_work_in_54_days_l489_489657


namespace sum_areas_of_tangent_circles_l489_489225

theorem sum_areas_of_tangent_circles : 
  ∃ r s t : ℝ, 
    (r + s = 6) ∧ 
    (r + t = 8) ∧ 
    (s + t = 10) ∧ 
    (π * (r^2 + s^2 + t^2) = 36 * π) :=
by
  sorry

end sum_areas_of_tangent_circles_l489_489225


namespace identify_quadratic_l489_489256

def is_quadratic_equation (eq : Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ eq = (a * (x : ℝ) ^ 2 + b * x + c = 0)

theorem identify_quadratic (x : ℝ)
    (eq1 : Prop := x^2 - 2*x - 3 = 0)
    (eq2 : Prop := x^2 - x*y = 2)
    (eq3 : Prop := x^2 + 1/x = 2)
    (eq4 : Prop := 2*(x - 1) = x) :
  is_quadratic_equation eq1 ∧
  ¬is_quadratic_equation eq2 ∧
  ¬is_quadratic_equation eq3 ∧
  ¬is_quadratic_equation eq4 := 
by
  sorry

end identify_quadratic_l489_489256


namespace range_of_derivative_at_1_l489_489521

noncomputable def f (θ : ℝ) (x : ℝ) : ℝ := x^3 + x^2 + Real.tan θ

theorem range_of_derivative_at_1 :
  ∀ θ ∈ Set.Icc 0 Real.pi, (∃ y ∈ Icc (Real.deriv (f θ) 1).left (Real.deriv (f θ) 1).right, 
  (Real.deriv (f θ) 1 = y) ∧ y ∈ Set.Iic 2) :=
by
  sorry

end range_of_derivative_at_1_l489_489521


namespace polygon_sides_l489_489602

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l489_489602


namespace find_lambda_l489_489412

-- Define the required vectors and conditions
variables {V : Type} [AddCommGroup V] [Module ℝ V]
variables (e1 e2 : V) (a b : V)
variables (λ : ℝ)
axiom non_collinear : ¬ collinear ℝ ({e1, e2} : Set V)

-- Definitions of a and b
def vec_a : V := 2 • e1 - e2
def vec_b : V := 3 • e1 + λ • e2

-- Collinearity condition between a and b
axiom collinear_ab : collinear ℝ ({vec_a, vec_b} : Set V)

-- The theorem we need to prove
theorem find_lambda : λ = -3 / 2 :=
by
  sorry

end find_lambda_l489_489412


namespace polygon_sides_l489_489605

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489605


namespace mass_percentage_Ca_in_CaBr2_l489_489367

def molar_mass_Ca : ℝ := 40.08
def molar_mass_Br : ℝ := 79.904
def molar_mass_CaBr2 : ℝ := molar_mass_Ca + 2 * molar_mass_Br
def mass_percentage_Ca : ℝ := (molar_mass_Ca / molar_mass_CaBr2) * 100

theorem mass_percentage_Ca_in_CaBr2 : mass_percentage_Ca = 20.04 := by
  sorry

end mass_percentage_Ca_in_CaBr2_l489_489367


namespace total_distance_from_start_is_zero_total_fuel_consumption_is_correct_total_cost_is_correct_l489_489979

-- Define the itinerary
def itinerary : List Int := [5, -4, -8, 10, 3, -6]

-- Define fuel consumption per km and price per liter
def fuel_consumption_per_km : Float := 0.2
def price_per_liter : Float := 6.2

-- Prove the total distance from starting point is 0 km
theorem total_distance_from_start_is_zero :
  itinerary.sum = 0 :=
sorry

-- Prove the total fuel consumption given the itinerary
theorem total_fuel_consumption_is_correct :
  (itinerary.map Int.natAbs).sum * fuel_consumption_per_km = 7.2 :=
sorry

-- Prove the total cost given the fuel consumption and price per liter
theorem total_cost_is_correct :
  7.2 * price_per_liter = 44.64 :=
sorry

end total_distance_from_start_is_zero_total_fuel_consumption_is_correct_total_cost_is_correct_l489_489979


namespace function_behaviour_l489_489195

noncomputable def dec_function (x : ℝ) : ℝ := 3 / x

theorem function_behaviour :
  ∀ x > 0, ∀ y > 0, (dec_function x = y) → (x₂ > x₁) → (dec_function x₂ < dec_function x₁) :=
by
  intros x y hxy hinc
  rw dec_function at hxy
  sorry

end function_behaviour_l489_489195


namespace assistant_increases_output_by_100_percent_l489_489934

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l489_489934


namespace two_lines_perpendicular_to_plane_are_parallel_l489_489380

variables {m n : Type} [line m] [line n]
variables {β : Type} [plane β]

-- Assuming different lines and plane where m and n are the lines and β is the plane
variable (hmn : m ≠ n)
variable (hmb : m ⊥ β)
variable (hnb : n ⊥ β)

theorem two_lines_perpendicular_to_plane_are_parallel :
  m ∥ n :=
sorry

end two_lines_perpendicular_to_plane_are_parallel_l489_489380


namespace area_of_red_flowers_is_54_l489_489907

noncomputable def total_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def red_yellow_area (total : ℝ) : ℝ :=
  total / 2

noncomputable def red_area (red_yellow : ℝ) : ℝ :=
  red_yellow / 2

theorem area_of_red_flowers_is_54 :
  total_area 18 12 / 2 / 2 = 54 := 
  by
    sorry

end area_of_red_flowers_is_54_l489_489907


namespace steve_probability_l489_489198

theorem steve_probability :
  let p := 1 / 4 in
  let k := 10 in
  let n := 20 in
  finset.Ico k (n + 1).sum (λ k, nat.choose n k * p^k * (1 - p)^(n - k)) = 0.000289 := sorry

end steve_probability_l489_489198


namespace plane_equation_l489_489027

-- Define the point P and the points M1, M2, M3 as discussed.
def P : ℝ × ℝ × ℝ := (2, 3, -5)
def M1 : ℝ × ℝ × ℝ := (2, 3, 0)
def M2 : ℝ × ℝ × ℝ := (2, 0, -5)
def M3 : ℝ × ℝ × ℝ := (0, 3, -5)

-- Prove that the plane passing through M1, M2, M3 has the equation 15x + 10y - 6z - 60 = 0.
theorem plane_equation (x y z : ℝ) :
  let determinant := (15 * (x - 2)) - (10 * (y - 3)) + (6 * z)
  in determinant = 60 :=
sorry

end plane_equation_l489_489027


namespace length_of_best_day_l489_489377

theorem length_of_best_day
  (len_raise_the_roof : Nat)
  (len_rap_battle : Nat)
  (len_best_day : Nat)
  (total_ride_duration : Nat)
  (playlist_count : Nat)
  (total_songs_length : Nat)
  (h_len_raise_the_roof : len_raise_the_roof = 2)
  (h_len_rap_battle : len_rap_battle = 3)
  (h_total_ride_duration : total_ride_duration = 40)
  (h_playlist_count : playlist_count = 5)
  (h_total_songs_length : len_raise_the_roof + len_rap_battle + len_best_day = total_songs_length)
  (h_playlist_length : total_ride_duration / playlist_count = total_songs_length) :
  len_best_day = 3 := 
sorry

end length_of_best_day_l489_489377


namespace min_socks_for_pair_l489_489464

variables (num_red_socks num_blue_socks : ℕ)

theorem min_socks_for_pair (h_red : num_red_socks = 24) (h_blue : num_blue_socks = 24) :
  ∃ n, n = 3 ∧ (∀ m < 3, (m mod 2 = 1 → (num_red_socks * (m div 2) + num_blue_socks * (m div 2) < m))) :=
by
  sorry

end min_socks_for_pair_l489_489464


namespace difference_before_exchange_l489_489544

--Definitions
variables {S B : ℤ}

-- Conditions
axiom h1 : S - 2 = B + 2
axiom h2 : B > S

theorem difference_before_exchange : B - S = 2 :=
by
-- Proof will go here
sorry

end difference_before_exchange_l489_489544


namespace tan_beta_minus_2alpha_l489_489378

noncomputable def tan_alpha := 1 / 2
noncomputable def tan_beta_minus_alpha := 2 / 5
theorem tan_beta_minus_2alpha (α β : ℝ) (h1 : Real.tan α = tan_alpha) (h2 : Real.tan (β - α) = tan_beta_minus_alpha) :
  Real.tan (β - 2 * α) = -1 / 12 := 
by
  sorry

end tan_beta_minus_2alpha_l489_489378


namespace sum_areas_of_tangent_circles_l489_489224

theorem sum_areas_of_tangent_circles : 
  ∃ r s t : ℝ, 
    (r + s = 6) ∧ 
    (r + t = 8) ∧ 
    (s + t = 10) ∧ 
    (π * (r^2 + s^2 + t^2) = 36 * π) :=
by
  sorry

end sum_areas_of_tangent_circles_l489_489224


namespace sequence_general_term_l489_489958

noncomputable def floor_part (x : ℝ) : ℤ := Int.floor x
noncomputable def frac_part (x : ℝ) : ℝ := x - Int.floor x

theorem sequence_general_term (x : ℝ) (n : ℕ) (k : ℕ+) :
  (sqrt (floor_part x * floor_part (x^3 : ℝ)) + sqrt (frac_part x * frac_part (x^3 : ℝ)) = x^2) ∧ (x ≥ 1) →
  (x_n = (if n % 2 = 1 then k else sqrt (k^2 + 1 / k))) :=
sorry

end sequence_general_term_l489_489958


namespace remainder_when_divided_l489_489384

theorem remainder_when_divided (P D Q R D'' Q'' R'' : ℕ) (h1 : P = Q * D + R) (h2 : Q = D'' * Q'' + R'') :
  P % (2 * D * D'') = D * R'' + R := sorry

end remainder_when_divided_l489_489384


namespace min_orange_chips_l489_489277

theorem min_orange_chips (p g o : ℕ)
    (h1: g ≥ (1 / 3) * p)
    (h2: g ≤ (1 / 4) * o)
    (h3: p + g ≥ 75) : o = 76 :=
    sorry

end min_orange_chips_l489_489277


namespace correct_option_A_l489_489812

noncomputable def chi_squared_test_hypothesis : Prop :=
  ∀ events : Set Event, events are_mutually_independent events

-- Chi-squared statistic chi_squared cannot be negative given that all data in a contingency table are positive integers.
noncomputable def chi_squared_nonnegative (entries : List ℕ) : Prop :=
  ∀ entry ∈ entries, 0 < entry → chi_squared entries ≥ 0

-- The relationship indicated by the chi-squared test is not absolute certainty.
noncomputable def chi_squared_test_interpretation : Prop :=
  ∀ (smoking_habit bronchitis : Event), related (smoking_habit) (bronchitis) → ¬(definitely related (smoking_habit) (bronchitis))

-- Entries in a 2x2 contingency table cannot be any numbers if they are to reflect a specific dataset's statistical relationship.
noncomputable def contingency_table_entries (entries : vector (fin 4)) : Prop :=
  ¬(∃ entries: Vector (fin 4), ∀ entry, positive number ∈ entries)

theorem correct_option_A :
  let option_A := chi_squared_test_hypothesis
  let option_B := chi_squared_nonnegative
  let option_C := chi_squared_test_interpretation
  let option_D := contingency_table_entries
  option_A = True
  ∧ option_B = False
  ∧ option_C = False
  ∧ option_D = False :=
    by
      let option_A := chi_squared_test_hypothesis
      let option_B := chi_squared_nonnegative
      let option_C := chi_squared_test_interpretation
      let option_D := contingency_table_entries
      sorry -- proof is not provided here

end correct_option_A_l489_489812


namespace second_discount_percentage_l489_489581

variables (P : ℝ) (D1 : ℝ) (S : ℝ) (D2 : ℝ)

theorem second_discount_percentage
  (hP : P = 400)
  (hD1 : D1 = 0.30)
  (hS : S = 224)
  (h1 : D2 = 0.20) :
  let after_first_discount := P * (1 - D1),
      after_second_discount := after_first_discount * (1 - D2)
  in after_second_discount = S :=
sorry

end second_discount_percentage_l489_489581


namespace bus_driver_hours_l489_489282

theorem bus_driver_hours (h_r h_o t c x : ℕ) (h_r_val : h_r = 16) (h_o_val : h_o = 16 * 175 / 100) (t_val : t = 65) (c_val : c = 1340) :
  16 * x + 28 * (65 - x) = 1340 → x = 40 :=
by
  intros h_eq h_r_val h_o_val t_val c_val
  sorry

end bus_driver_hours_l489_489282


namespace inverse_proportion_decreases_l489_489193

theorem inverse_proportion_decreases {x : ℝ} (h : x > 0 ∨ x < 0) : 
  y = 3 / x → ∀ (x1 x2 : ℝ), (x1 > 0 ∨ x1 < 0) → (x2 > 0 ∨ x2 < 0) → x1 < x2 → (3 / x1) > (3 / x2) := 
by
  sorry

end inverse_proportion_decreases_l489_489193


namespace bushes_needed_for_60_zucchinis_l489_489769

variable number_of_containers_per_bush : ℕ := 11
variable number_of_containers_per_zucchini : ℕ := 3
variable number_of_zucchinis_needed : ℕ := 60

theorem bushes_needed_for_60_zucchinis : 
  let bushes : ℕ := Nat.ceil ((number_of_zucchinis_needed * number_of_containers_per_zucchini) / number_of_containers_per_bush);
  bushes = 17 :=
by
  sorry

end bushes_needed_for_60_zucchinis_l489_489769


namespace ab_value_l489_489180

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : a^4 + b^4 = 5 / 8) : ab = (Real.sqrt 3) / 4 :=
by
  sorry

end ab_value_l489_489180


namespace probability_sum_is_odd_l489_489109

-- Define the list of the first eight prime numbers
def first_eight_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Count the successful outcomes, where the sum of two distinct primes is odd
def num_successful_outcomes : ℕ :=
  List.length $ List.filter (λ pair: ℕ × ℕ, 
      pair.1 ≠ pair.2 ∧ (pair.1 + pair.2) % 2 = 1) $ 
      first_eight_primes.product first_eight_primes

-- Total possible outcomes for selecting 2 distinct numbers
def total_outcomes : ℕ := first_eight_primes.length.choose 2

-- Prove that the probability of their sum being odd is 1/4
theorem probability_sum_is_odd :
  (num_successful_outcomes : ℚ) / total_outcomes = 1 / 4 := sorry

end probability_sum_is_odd_l489_489109


namespace locus_of_B_with_orthocenter_and_angle_condition_l489_489093

variable {A H : Point}
variable {B : Point}
variable {α : ℝ}

/-- A condition ensuring the existence of a point B such that triangle ABC has H as orthocenter 
     and all angles greater than α, where α < π/4. -/
theorem locus_of_B_with_orthocenter_and_angle_condition
  (h1 : 0 < α)
  (h2 : α < π / 4)
  (h3 : is_orthocenter H A B C)
  (h4 : ∀ (angleA angleB angleC : ℝ), (angleA > α) ∧ (angleB > α) ∧ (angleC > α)) :
  ∃ B, ∃ C, is_orthocenter H A B C ∧ (angle A > α) ∧ (angle B > α) ∧ (angle C > α) :=
sorry

end locus_of_B_with_orthocenter_and_angle_condition_l489_489093


namespace polar_equation_of_curve_line_curve_intersection_product_l489_489128

section
variables {α : Type*} [RealField α]

-- Defining parametric equations of the curve C
noncomputable def x (a : α) : α := 1 + 2 * Real.cos a
noncomputable def y (a : α) : α := 2 * Real.sin a

-- polar equation transformation
theorem polar_equation_of_curve 
    (ρ θ : α) : ρ^2 - 2 * ρ * Real.cos θ - 3 = 0 → 
    (∃ a, x a = ρ * Real.cos θ ∧ y a = ρ * Real.sin θ) :=
begin
    sorry
end

-- Line intersection and product of distances theorem
theorem line_curve_intersection_product 
    (k : α) (h : k > 0) (β ρ1 ρ2 : α) 
    (hρ1 : ρ1^2 - 2 * ρ1 * Real.cos β - 3 = 0)
    (hρ2 : ρ2^2 - 2 * ρ2 * Real.cos β - 3 = 0) :
    |ρ1| * |ρ2| = 3 :=
begin
    sorry
end
end

end polar_equation_of_curve_line_curve_intersection_product_l489_489128


namespace polygon_sides_l489_489619

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489619


namespace probability_two_females_chosen_l489_489467

theorem probability_two_females_chosen (total_contestants : ℕ) (female_contestants : ℕ) (chosen_contestants : ℕ) :
  total_contestants = 8 → female_contestants = 5 → chosen_contestants = 2 → 
  (finset.card (finset.filter (λ p : Finset (Fin 8), ∀ q ∈ p, q.val < 5) ((finset.univ : Finset (Fin 8)).powersetLen 2))) / 
  (finset.card ((finset.univ : Finset (Fin 8)).powersetLen 2).to_float) = 5 / 14 := 
by 
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end probability_two_females_chosen_l489_489467


namespace planted_fraction_correct_l489_489355

def planted_fraction (triangle_area square_area : ℝ) : ℝ := (triangle_area - square_area) / triangle_area

theorem planted_fraction_correct :
  ∀ (a b : ℝ), a = 3 → b = 4 →
  let h := Real.sqrt (a^2 + b^2) in
  ∀ (s_dist : ℝ), s_dist = 2 →
  let square_area := (2 / (a + b)^2) * h^2 in
  planted_fraction ((1/2) * a * b) square_area = 145 / 147 :=
by
  intros a b ha hb h s_dist hs square_area
  subst ha
  subst hb
  subst hs
  have triangle_area : ℝ := (1 / 2) * a * b
  have square_area : ℝ := 4 / 49
  rw [triangle_area, square_area]
  admit -- The detailed math proof would go here

end planted_fraction_correct_l489_489355


namespace system_solution_ratio_l489_489372

theorem system_solution_ratio (x y z : ℝ) (h_xyz_nonzero: x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h1 : x + (95/9)*y + 4*z = 0) (h2 : 4*x + (95/9)*y - 3*z = 0) (h3 : 3*x + 5*y - 4*z = 0) :
  (x * z) / (y ^ 2) = 175 / 81 := 
by sorry

end system_solution_ratio_l489_489372


namespace num_solutions_ggx_eq_5_l489_489518

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 1 then -x + 4 else 3 * x - 7

theorem num_solutions_ggx_eq_5 : 
(∃ (x : ℝ), g (g x) = 5) ↔ 2 :=
begin
  sorry
end

end num_solutions_ggx_eq_5_l489_489518


namespace find_f_half_l489_489072

variable {f : ℝ → ℝ}

axiom f_eq : ∀ x : ℝ, f(x) = 1 - f(2) * Real.logb 2 x

theorem find_f_half (h : f_eq) : f(1/2) = 3/2 :=
sorry

end find_f_half_l489_489072


namespace negation_of_exists_l489_489575

theorem negation_of_exists (h : ¬ ∃ x : ℝ, x^2 + x + 1 < 0) : ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by
  sorry

end negation_of_exists_l489_489575


namespace train_length_correct_l489_489705

noncomputable def train_length (speed_kmh : ℝ) (time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  (speed_kmh / 3.6) * time_s - bridge_length_m

theorem train_length_correct :
  train_length 72 12.299016078713702 136 = 110.98032157427404 := by
  unfold train_length
  -- Direct calculation should match the provided value
  have h1 : (72 / 3.6 : ℝ) = 20 := by norm_num
  rw [h1]
  have h2 : 20 * 12.299016078713702 = 246.98032157427404 := by norm_num
  rw [h2]
  norm_num
  sorry

end train_length_correct_l489_489705


namespace sqrt_range_l489_489449

theorem sqrt_range (x : ℝ) : (1 - x ≥ 0) ↔ (x ≤ 1) := sorry

end sqrt_range_l489_489449


namespace hot_dogs_served_for_dinner_l489_489698

theorem hot_dogs_served_for_dinner
  (l t : ℕ) 
  (h_cond1 : l = 9) 
  (h_cond2 : t = 11) :
  ∃ d : ℕ, d = t - l ∧ d = 2 := by
  sorry

end hot_dogs_served_for_dinner_l489_489698


namespace parallel_lines_distance_l489_489427

theorem parallel_lines_distance 
  (a c : ℝ)
  (h0 : a = -4)
  (h1 : c ≠ -2)
  (h2 : abs(2 * c + 4) / √13 = 4 / 13) :
  (c + 2) / a = 1 :=
  sorry

end parallel_lines_distance_l489_489427


namespace area_enclosed_by_curves_l489_489271

theorem area_enclosed_by_curves :
  ∫ x in 0..1, sqrt x + ∫ x in 1..2, 2 - x = 7 / 6 :=
by
  sorry

end area_enclosed_by_curves_l489_489271


namespace function_behaviour_l489_489194

noncomputable def dec_function (x : ℝ) : ℝ := 3 / x

theorem function_behaviour :
  ∀ x > 0, ∀ y > 0, (dec_function x = y) → (x₂ > x₁) → (dec_function x₂ < dec_function x₁) :=
by
  intros x y hxy hinc
  rw dec_function at hxy
  sorry

end function_behaviour_l489_489194


namespace eval_expr1_eval_expr2_l489_489320

-- (1) Proof that evaluates to -3
theorem eval_expr1 : 
  Real.cbrt ((-4)^3) - (1/2)^0 + (0.25)^(1/2) * ((-1)/(Real.sqrt 2))^(-4) = -3 := 
by
  sorry

-- (2) Proof for the given condition
theorem eval_expr2 (x : ℝ) (h : x^(1/2) + x^(-1/2) = 3) : x^(3/2) + x^(-3/2) = 18 := 
by
  sorry

end eval_expr1_eval_expr2_l489_489320


namespace percentage_of_x_l489_489434

-- Define variables
variables {x y : ℝ}

-- State the condition
def condition : Prop := 0.20 * (x - y) = 0.15 * (x + y)

-- State the theorem
theorem percentage_of_x (h : condition) : y = (100 / 7) * (x / 100) := by
  sorry

end percentage_of_x_l489_489434


namespace alcohol_percentage_in_new_mixture_l489_489669

-- Define the initial conditions.
variable (initial_solution_volume : ℝ) (initial_alcohol_percentage : ℝ) (added_water_volume : ℝ)
variable (new_total_volume : ℝ) (remaining_alcohol_volume : ℝ) (new_alcohol_percentage : ℝ)

-- Define the given values.
def condition1 := initial_solution_volume = 9
def condition2 := initial_alcohol_percentage = 0.57
def condition3 := added_water_volume = 3

-- Define the calculations based on conditions.
def calc_alcohol_volume := remaining_alcohol_volume = initial_solution_volume * initial_alcohol_percentage
def calc_new_volume := new_total_volume = initial_solution_volume + added_water_volume
def calc_new_percentage := new_alcohol_percentage = (remaining_alcohol_volume / new_total_volume) * 100

-- The theorem we need to prove.
theorem alcohol_percentage_in_new_mixture :
  condition1 ∧ condition2 ∧ condition3 ∧ calc_alcohol_volume ∧ calc_new_volume ∧ calc_new_percentage → new_alcohol_percentage = 42.75 :=
by 
  -- The proof details are skipped.
  sorry

end alcohol_percentage_in_new_mixture_l489_489669


namespace children_picking_apples_l489_489226

theorem children_picking_apples (total_apples initial_apples remaining_apples : ℕ) (baskets : Fin 11 → ℕ) 
  (basket_sum total_picked : ℕ) (c : ℕ) :
  (∀ (i : Fin 11), baskets i = i + 1) →
  total_apples = initial_apples →
  remaining_apples = 340 →
  initial_apples = 1000 →
  basket_sum = ∑ i, baskets i →
  total_picked = total_apples - remaining_apples →
  total_picked = basket_sum * c →
  basket_sum = 66 →
  c = 10 :=
by
  intros
  sorry

end children_picking_apples_l489_489226


namespace arccos_neg_one_eq_pi_l489_489740

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l489_489740


namespace fifth_decimal_place_of_x_l489_489274

theorem fifth_decimal_place_of_x :
  ∃ x : ℝ, (8 * 5.4 - 0.6 * 10 / x = 31.000000000000004) ∧ 
  (floor (10^5 * x) % 10 = 3) := 
sorry

end fifth_decimal_place_of_x_l489_489274


namespace Rahul_batting_average_l489_489540

theorem Rahul_batting_average 
  (A : ℕ) (current_matches : ℕ := 12) (new_matches : ℕ := 13) (scored_today : ℕ := 78) (new_average : ℕ := 54)
  (h1 : (A * current_matches + scored_today) = new_average * new_matches) : A = 52 := 
by
  sorry

end Rahul_batting_average_l489_489540


namespace angle_DBE_values_l489_489136

noncomputable theory

variables {O : Type} [Circle O]
variables {A B C D E : O}
variables {O_center : Point O}
variables (triangle_ABC : Triangle O where angle_B = 36)
variables (D_diameter : Diameter O A D)
variables (E_diameter : Diameter O C E)

theorem angle_DBE_values :
  (∠ D B E = 36) ∨ (∠ D B E = 144) := sorry

end angle_DBE_values_l489_489136


namespace min_xy_min_x_plus_y_l489_489820

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : xy ≥ 4 := sorry

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : x + y ≥ 9 / 2 := sorry

end min_xy_min_x_plus_y_l489_489820


namespace ternary_predecessor_l489_489876

theorem ternary_predecessor (M : ℕ) (h : M = 2 * 3^4 + 1 * 3^3 + 0 * 3^2 + 2 * 3^1 + 1 * 3^0) : 
  prev_ternary(M) = "21020" :=
by
  sorry

end ternary_predecessor_l489_489876


namespace omega_value_min_value_x_set_smallest_positive_phi_l489_489081

variables (ω : ℝ) (x : ℝ) (φ : ℝ)
noncomputable def a := (sqrt 3, 2 * Real.sin (ω * x / 2))
noncomputable def b := (Real.sin (ω * x), -Real.sin (ω * x / 2))
noncomputable def f := (a.1 * b.1 + a.2 * b.2 + 1)

-- Theorem statements
theorem omega_value (h1 : ∀ x, f x = 2 * Real.sin (ω * x + (Real.pi / 6))) : ω = 2 := sorry

theorem min_value_x_set
  (h1 : ω = 2)
  (h2 : ∀ x, f x = 2 * Real.sin (2 * x + (Real.pi / 6))) :
  (∀ k : ℤ, ∃ x, f x = -2 ∧ x = -(Real.pi / 3) + k * Real.pi) := sorry

theorem smallest_positive_phi
  (h1 : ω = 2)
  (h2 : ∀ x, f x = 2 * Real.sin (2 * x + (Real.pi / 6)))
  (h3 : ∀ x, Real.sin (2 * (x + φ) + (Real.pi / 6)) = Real.sin (2 * x + 2 * φ + (Real.pi / 6)))
  (h4 : 2 * (Real.pi / 3) + 2 * φ + (Real.pi / 6) = k * Real.pi) :
  (∃ k : ℤ, φ = (k * Real.pi / 2) - (5 * Real.pi / 12) ∧ φ > 0) :=
  ∃ φ_min, φ_min = Real.pi / 12 ∧ φ_min > 0 := sorry

end omega_value_min_value_x_set_smallest_positive_phi_l489_489081


namespace find_k_l489_489476

noncomputable theory

def curve_C1 (θ : ℝ) : ℝ × ℝ :=
  (3 + cos θ, 4 + sin θ)

def curve_C1_eq (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 4)^2 = 1

def curve_C2 (k : ℝ) (ρ θ : ℝ) : Prop :=
  ρ * (sin θ - k * cos θ) = 3

def curve_C2_eq (k x y : ℝ) : Prop :=
  y = k * x + 3

def calculate_distance (x0 y0 A B C : ℝ) : ℝ :=
  abs (A * x0 + B * y0 + C) / sqrt (A^2 + B^2)

def tangent_minimum_length := 2 * sqrt 2

theorem find_k (k : ℝ) :
  (∃ P_on_C2 : ℝ × ℝ, curve_C2_eq k P_on_C2.1 P_on_C2.2) ∧
  (∃ P_on_C2 : ℝ × ℝ, calculate_distance 3 4 (-k) 1 3 = 3) ∧
calculate_distance 3 4 (-k) 1 3 = 3 → k = -4 / 3 :=
sorry

end find_k_l489_489476


namespace max_unique_coin_sums_l489_489689

theorem max_unique_coin_sums :
  let coins := [1, 1, 1, 5, 10, 25] in
  let sums := finset.bind (finset.powerset (finset.of_multiset (coins.to_multiset))) 
                          (λ s, if s.card = 2 then {s.sum} else ∅) in
  sums.card = 7 := 
by
  let coins := [1, 1, 1, 5, 10, 25] 
  let sums := finset.bind (finset.powerset (finset.of_multiset (coins.to_multiset))) 
                          (λ s, if s.card = 2 then {s.sum} else ∅)
  have : sums = {2, 6, 11, 15, 26, 30, 35} := sorry
  have : sums.card = 7 := 
    by { calc
           sums.card = {2, 6, 11, 15, 26, 30, 35}.card : sorry <| this
                ... = 7                                 : by norm_num }
  exact this

end max_unique_coin_sums_l489_489689


namespace relationship_R2_residuals_l489_489580

-- Definitions:
def coeff_determination (residuals_sq_sum : ℝ) (total_sq_sum : ℝ) : ℝ :=
  1 - (residuals_sq_sum / total_sq_sum)

-- Conditions and theorem to prove:
theorem relationship_R2_residuals (R2 : ℝ) (residuals_sq_sum total_sq_sum : ℝ) :
  R2 = coeff_determination residuals_sq_sum total_sq_sum →
  (R2 > 0) →
  (residuals_sq_sum < total_sq_sum) →
  (R2 > coeff_determination (residuals_sq_sum + 1) total_sq_sum) :=
by
  sorry -- Proof is not required

end relationship_R2_residuals_l489_489580


namespace reduced_rates_fraction_l489_489262

theorem reduced_rates_fraction (total_hours_week : ℕ) (weekdays_reduced_hours : ℕ) (weekends_reduced_hours : ℕ) 
(hr_wkdays : weekdays_reduced_hours = 12 * 5) (hr_wkends : weekends_reduced_hours = 24 * 2) :
  (weekdays_reduced_hours + weekends_reduced_hours) / total_hours_week = 9 / 14 :=
by
  have total_hours_week_eq : total_hours_week = 7 * 24 := rfl
  have total_reduced_hours_eq : weekdays_reduced_hours + weekends_reduced_hours = 60 + 48 :=
    by rw [hr_wkdays, hr_wkends]; rfl
  have reduced_fraction_simplified : (60 + 48 : ℝ) / 168 = 9 / 14 := 
    by norm_num
  rw [total_hours_week_eq, total_reduced_hours_eq] at reduced_fraction_simplified
  exact reduced_fraction_simplified

end reduced_rates_fraction_l489_489262


namespace students_with_puppies_and_parrots_l489_489458

theorem students_with_puppies_and_parrots (total_students : ℕ) 
    (percent_with_puppies : ℝ) 
    (percent_with_puppies_and_parrots : ℝ) 
    (h_total_students : total_students = 40) 
    (h_percent_with_puppies : percent_with_puppies = 0.80) 
    (h_percent_with_puppies_and_parrots : percent_with_puppies_and_parrots = 0.25) :
    let students_with_puppies := percent_with_puppies * total_students,
        students_with_both := percent_with_puppies_and_parrots * students_with_puppies
    in students_with_both = 8 :=
by {
  sorry
}

end students_with_puppies_and_parrots_l489_489458


namespace calcTotalProfit_l489_489304

noncomputable def totalProfit (investmentA investmentB investmentC investmentD investmentE: ℕ)
                              (rateA rateB rateC rateD rateE: ℝ)
                              (timeA timeE: ℝ)
                              (shareA shareB shareC shareD shareE: ℝ)
                              (profitC: ℕ): ℝ :=
  let interestA := investmentA * rateA * timeA
  let interestB := investmentB * rateB
  let interestC := investmentC * rateC
  let interestD := investmentD * rateD
  let interestE := investmentE * rateE * timeE
  let totalInterest := interestA + interestB + interestC + interestD + interestE
  let totalProfit := profitC / shareC
  totalProfit

theorem calcTotalProfit (investmentA investmentB investmentC investmentD investmentE: ℕ)
                        (rateA rateB rateC rateD rateE: ℝ)
                        (timeA timeE: ℝ)
                        (shareA shareB shareC shareD shareE: ℝ)
                        (profitC: ℕ)
                        (hA: investmentA = 12000)
                        (hB: investmentB = 16000)
                        (hC: investmentC = 20000)
                        (hD: investmentD = 24000)
                        (hE: investmentE = 18000)
                        (hRA: rateA = 0.05)
                        (hRB: rateB = 0.06)
                        (hRC: rateC = 0.07)
                        (hRD: rateD = 0.08)
                        (hRE: rateE = 0.065)
                        (hTimeA: timeA = 0.5)
                        (hTimeE: timeE = 0.5)
                        (hShareA: shareA = 0.10)
                        (hShareB: shareB = 0.15)
                        (hShareC: shareC = 0.20)
                        (hShareD: shareD = 0.25)
                        (hShareE: shareE = 0.30)
                        (hProfitC: profitC = 36000):
  totalProfit investmentA investmentB investmentC investmentD investmentE
              rateA rateB rateC rateD rateE
              timeA timeE
              shareA shareB shareC shareD shareE
              profitC = 180000 := by
  -- Assume necessary conditions
  rw [←hA, ←hB, ←hC, ←hD, ←hE, ←hRA, ←hRB, ←hRC, ←hRD, ←hRE, ←hTimeA, ←hTimeE, ←hShareA, ←hShareB, ←hShareC, ←hShareD, ←hShareE, ←hProfitC]
  -- Actual proof goes here
  sorry

end calcTotalProfit_l489_489304


namespace common_difference_l489_489479

def arith_seq_common_difference (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference {a : ℕ → ℤ} (h₁ : a 5 = 3) (h₂ : a 6 = -2) : arith_seq_common_difference a (-5) :=
by
  intros n
  cases n with
  | zero => sorry -- base case: a 1 = a 0 + (-5), requires additional initial condition
  | succ n' => sorry -- inductive step

end common_difference_l489_489479


namespace circumcenter_of_A₁B₁C₁_is_incenter_of_ABC_l489_489660

-- Definitions from condition a)
variables {A B C O A₀ B₀ C₀ A₁ B₁ C₁ I : Type}
variables (ABC : Triangle A B C)
variables (acute_scalene : acute_angled_scalene ABC)
variables (altitudes : (Altitude A ABC A₀) ∧ (Altitude B ABC B₀) ∧ (Altitude C ABC C₀))
variables (R : Real)
variables (distance_equality : (AA₁ = BB₁ ∧ AA₁ = CC₁) ∧ (AA₁ = R))

-- Question (correct answer extracted from solution step b))
theorem circumcenter_of_A₁B₁C₁_is_incenter_of_ABC
  (circumcenter_ABC : Circumcenter ABC O)
  (A₁B₁C₁ : Triangle A₁ B₁ C₁)
  (points_on_altitudes : OnAltitude ABC A A₁ ∧ OnAltitude ABC B B₁ ∧ OnAltitude ABC C C₁)
  (radius_equality : Distance A O A₁ = R ∧ Distance B O B₁ = R ∧ Distance C O C₁ = R)
  (incenter : Incenter ABC I) :
  Circumcenter A₁B₁C₁ I :=
sorry

end circumcenter_of_A₁B₁C₁_is_incenter_of_ABC_l489_489660


namespace vec_op_comm_vec_op_bound_l489_489339

variables {V : Type} [InnerProductSpace ℝ V]
variable (a b : V)

-- Define the operation ⊗
noncomputable def vec_op (a b : V) : ℝ :=
  ‖a‖ * ‖b‖ * Real.sin (inner a b)

-- Prove the conclusions
theorem vec_op_comm : vec_op a b = vec_op b a :=
  sorry

theorem vec_op_bound : abs (vec_op a b) ≤ ‖a‖ * ‖b‖ :=
  sorry

end vec_op_comm_vec_op_bound_l489_489339


namespace transform_to_concentric_circles_l489_489990

theorem transform_to_concentric_circles (S1 S2 : set ℝ) (h1 : is_circle S1) (h2 : is_circle S2 ∨ is_line S2) (h3 : ¬(S1 ∩ S2).nonempty) :
  ∃ (O : ℝ) (radius : ℝ), is_inverted_concentric S1 S2 O radius :=
sorry

end transform_to_concentric_circles_l489_489990


namespace lizz_team_loses_by_8_points_l489_489170

-- Definitions of the given conditions
def initial_deficit : ℕ := 20
def free_throw_points : ℕ := 5 * 1
def three_pointer_points : ℕ := 3 * 3
def jump_shot_points : ℕ := 4 * 2
def liz_points : ℕ := free_throw_points + three_pointer_points + jump_shot_points
def other_team_points : ℕ := 10
def points_caught_up : ℕ := liz_points - other_team_points
def final_deficit : ℕ := initial_deficit - points_caught_up

-- Theorem proving Liz's team loses by 8 points
theorem lizz_team_loses_by_8_points : final_deficit = 8 :=
  by
    -- Proof will be here
    sorry

end lizz_team_loses_by_8_points_l489_489170


namespace compound_proposition_truths_l489_489395

noncomputable def sqrt_five_div_two_real : ℝ := real.sqrt 5 / 2

def prop_p : Prop := ∃ x : ℝ, real.sin x = sqrt_five_div_two_real
def prop_q : Prop := ∀ x : ℝ, x^2 + x + 1 > 0

theorem compound_proposition_truths :
  ¬ prop_p ∧ prop_q ∧ (p ∧ ¬q → false) ∧ (¬p ∨ q) :=
by {
  sorry
}

end compound_proposition_truths_l489_489395


namespace num_palindromes_is_correct_l489_489430

section Palindromes

def num_alphanumeric_chars : ℕ := 10 + 26

def num_four_char_palindromes : ℕ := num_alphanumeric_chars * num_alphanumeric_chars

theorem num_palindromes_is_correct : num_four_char_palindromes = 1296 :=
by
  sorry

end Palindromes

end num_palindromes_is_correct_l489_489430


namespace minimum_distance_PQ_l489_489043

theorem minimum_distance_PQ :
  let P := fun x : ℝ => (x, (1 / 2) * Real.exp x)
  let Q := fun x : ℝ => (y, Real.log (2 * x))
  ∀ x > 0, y > 0, P x, Q y ∃ dmin,
  |P.2 - Q.2| = dmin → dmin = (1 - Real.log 2) / Real.sqrt 2 :=
sorry

end minimum_distance_PQ_l489_489043


namespace sides_of_polygon_l489_489616

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l489_489616


namespace line_intersections_product_l489_489877

open_locale big_operators

variables {A B C O C_1 B_1 A_1: Type*}
variables [field A] [field B] [field C] [field O] [field C_1] [field B_1] [field A_1]

/-- Given a triangle ABC with an arbitrary point O in its plane, and lines through vertices A, B, C 
    and O intersecting sides AB, AC, BC at points C_1, B_1, A_1 respectively, the products of any 
    three segments not sharing a common vertex are equal. -/
theorem line_intersections_product :
  AC_1 * BA_1 * CB_1 = AB_1 * CA_1 * BC_1 :=
sorry

end line_intersections_product_l489_489877


namespace jane_output_increase_l489_489927

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l489_489927


namespace smallest_b_value_l489_489985

theorem smallest_b_value (a b c : ℕ) (h0 : a > 0) (h1 : b > 0) (h2 : c > 0)
  (h3 : (31 : ℚ) / 72 = (a : ℚ) / 8 + (b : ℚ) / 9 - c) :
  b = 5 :=
sorry

end smallest_b_value_l489_489985


namespace fg_pi_is_zero_l489_489033

def f (x : ℝ) : ℤ :=
  if x > 0 then 1
  else if x = 0 then 0
  else -1

def g (x : ℝ) : ℤ :=
  if ∃ q : ℚ, ↑q = x then 1
  else 0

theorem fg_pi_is_zero : f (g (Real.pi)) = 0 :=
by
  sorry

end fg_pi_is_zero_l489_489033


namespace min_colors_needed_min_colors_needed_l489_489830

theorem min_colors_needed (n : ℕ) (h : n ≥ 3) : odd n → (∃ m, ∀ v : ℕ, v < n → v ≥ 0 → (∀ i < n-1, ∃ c : ℕ, 1 ≤ c ∧ c ≤ m)) → m = n
| n h odd n → (∃ m, ∀ v : ℕ, v < n → v ≥ 0 → (∀ i < n-1, ∃ c : ℕ, 1 ≤ c ∧ c ≤ m)) → m = n :=
sorry

theorem min_colors_needed (n : ℕ) (h : n ≥ 3) : even n → (∃ m, ∀ v : ℕ, v < n → v ≥ 0 → (∀ i < n-1, ∃ c : ℕ, 1 ≤ c ∧ c ≤ m)) → m = n-1
| n h even n → (∃ m, ∀ v : ℕ, v < n → v ≥ 0 → (∀ i < n-1, ∃ c : ℕ, 1 ≤ c ∧ c ≤ m)) → m = n-1 :=
sorry

end min_colors_needed_min_colors_needed_l489_489830


namespace num_three_digit_sums7_l489_489000

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l489_489000


namespace trig_expression_value_l489_489401

-- Define the problem's conditions and the result statement
theorem trig_expression_value
  (theta : ℝ)
  (h1 : cos (π / 4 + theta) = -3 / 5)
  (h2 : 11 * π / 12 < theta ∧ theta < 5 * π / 4) :
  (sin (2 * theta) + 2 * sin (theta)^2) / (1 - tan(theta)) = 28 / 75 :=
sorry

end trig_expression_value_l489_489401


namespace remainder_of_2345678_div_5_l489_489247

theorem remainder_of_2345678_div_5 : (2345678 % 5) = 3 :=
by
  sorry

end remainder_of_2345678_div_5_l489_489247


namespace cos_squared_alpha_plus_pi_div_4_l489_489402

noncomputable def alpha : ℝ := sorry -- α is a real number.

axiom sin_two_alpha : real.sin (2 * alpha) = 2 / 3

theorem cos_squared_alpha_plus_pi_div_4 :
  real.cos (alpha + real.pi / 4) ^ 2 = 1 / 6 :=
by
  have h := sin_two_alpha
  sorry

end cos_squared_alpha_plus_pi_div_4_l489_489402


namespace least_positive_integer_n_exceeds_1024_l489_489089

theorem least_positive_integer_n_exceeds_1024 :
  ∃ n : ℕ, (∀ m : ℕ, m < n → 2^((∑ k in (finset.range m), (2 * k + 1) / 7) ≤ 1024) ∧ 2^((∑ k in (finset.range n), (2 * k + 1) / 7) > 1024)) ∧ n = 9 :=
sorry

end least_positive_integer_n_exceeds_1024_l489_489089


namespace remainder_172_pow_172_mod_13_l489_489246

theorem remainder_172_pow_172_mod_13 :
  (172 : ℤ) ^ 172 % 13 = 3 := 
by {
  have h1 : 172 % 13 = 2 := sorry,
  have h2 : (2 : ℤ) ^ 172 % 13 = 3 := sorry,
  rw <- h1,
  exact h2,
}

end remainder_172_pow_172_mod_13_l489_489246


namespace non_intersecting_circles_concentric_inversion_l489_489988

-- Define a basic geometric setup (placeholders for actual definitions)
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the non-intersecting condition
def non_intersecting (S1 S2 : Circle) : Prop :=
  let d := (S1.center.1 - S2.center.1)^2 + (S1.center.2 - S2.center.2)^2
  d > (S1.radius + S2.radius)^2

-- Problem statement in Lean
theorem non_intersecting_circles_concentric_inversion
  (S1 S2 : Circle) (h : non_intersecting S1 S2) : 
  ∃ O k, -- Possible parameters for the inversion center and power
    let S1' := { center := (O.1, O.2), radius := k } -- Placeholder for actual inversion transformation
    let S2' := { center := (O.1, O.2), radius := k } -- Placeholder for actual inversion transformation
    S1'.center = S2'.center := 
sorry

end non_intersecting_circles_concentric_inversion_l489_489988


namespace increase_in_output_with_assistant_l489_489920

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l489_489920


namespace area_ratio_l489_489408

theorem area_ratio (O A B C D : Point) (h1: O ∈ interior (triangle A B C))
  (h2: midpoint D A B)
  (h3: vector_eq (vector_add (vector_add (vector O A) (vector O B))
    (vector_smul 2 (vector O C))) (vector_zero)) : 
  (area (triangle A B C) / area (triangle A O C)) = 4 := 
sorry

end area_ratio_l489_489408


namespace milkshakes_total_l489_489313

theorem milkshakes_total (A_rate L_rate time : ℕ) (hA : A_rate = 3) (hL : L_rate = 7) (hT : time = 8) :
  A_rate * time + L_rate * time = 80 :=
by 
  rw [hA, hL, hT]
  norm_num

end milkshakes_total_l489_489313


namespace smallest_integer_satisfying_conditions_l489_489797

-- Define the conditions explicitly as hypotheses
def satisfies_congruence_3_2 (n : ℕ) : Prop :=
  n % 3 = 2

def satisfies_congruence_7_2 (n : ℕ) : Prop :=
  n % 7 = 2

def satisfies_congruence_8_2 (n : ℕ) : Prop :=
  n % 8 = 2

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- Define the smallest positive integer satisfying the above conditions
theorem smallest_integer_satisfying_conditions : ∃ (n : ℕ), n > 1 ∧ satisfies_congruence_3_2 n ∧ satisfies_congruence_7_2 n ∧ satisfies_congruence_8_2 n ∧ is_perfect_square n :=
  by
    sorry

end smallest_integer_satisfying_conditions_l489_489797


namespace parabola_intersection_difference_l489_489577

theorem parabola_intersection_difference:
  ∃ a c : ℝ, (∃ b d : ℝ, 
    b = 3*a^2 - 6*a + 6 ∧ 
    b = -2*a^2 - 4*a + 7 ∧ 
    d = 3*c^2 - 6*c + 6 ∧ 
    d = -2*c^2 - 4*c + 7) ∧ 
    c ≥ a ∧ 
    c - a = (2 * Real.sqrt 6) / 5 :=
begin
  sorry
end

end parabola_intersection_difference_l489_489577


namespace solve_sqrt_equation_l489_489551

theorem solve_sqrt_equation :
  (∃ x : ℝ, (sqrt (3 + sqrt (4 + sqrt x)) = real.cbrt (2 + sqrt x)) ∧  x = 625) :=
sorry

end solve_sqrt_equation_l489_489551


namespace projection_ratio_l489_489150

variables {V : Type*} [inner_product_space ℝ V]
variables (v w p q : V)
variable (hv : v ≠ 0)
variable (hw : w ≠ 0)

-- Definitions of projections
def proj_w_v : V := inner_product_space.proj w v
def proj_w_p : V := inner_product_space.proj w p

-- Hypotheses
axiom hp : p = proj_w_v
axiom hq : q = proj_w_p
axiom hratio : ∥p∥ / ∥v∥ = 4 / 5

-- Theorem to prove
theorem projection_ratio : ∥q∥ / ∥w∥ = 4 / 5 :=
by sorry

end projection_ratio_l489_489150


namespace exists_alternating_multiple_l489_489693

def is_alternating (n : ℕ) : Prop :=
  let digits := n.digits 10
  (∀i, i < digits.length - 1 → (digits.nth i % 2 ≠ digits.nth (i + 1) % 2))

theorem exists_alternating_multiple (n : ℕ) (h : ¬ (20 ∣ n)) : 
  ∃ m : ℕ, m % n = 0 ∧ is_alternating m :=
sorry

end exists_alternating_multiple_l489_489693


namespace sum_binom_not_divisible_by_5_l489_489535

theorem sum_binom_not_divisible_by_5 (n : ℕ) : 
  ¬ (5 ∣ (∑ k in Finset.range (n + 1), Nat.choose (2 * n + 1) (2 * k + 1) * 2^(3 * k))) := 
by
  sorry

end sum_binom_not_divisible_by_5_l489_489535


namespace sum_cubes_div_xyz_eq_thirteen_l489_489157

noncomputable def complex_numbers := ℂ

theorem sum_cubes_div_xyz_eq_thirteen
  (x y z : complex_numbers)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (h1 : x + y + z = 10)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 13 := 
by {
  sorry
}

end sum_cubes_div_xyz_eq_thirteen_l489_489157


namespace count_even_multiples_of_5_as_perfect_squares_lt_2500_l489_489867

theorem count_even_multiples_of_5_as_perfect_squares_lt_2500 : 
  ∃ n, n = 4 ∧ ∀ k : ℕ, 0 < k → 100 * k * k < 2500 → k ∈ {1, 2, 3, 4} :=
by
  sorry

end count_even_multiples_of_5_as_perfect_squares_lt_2500_l489_489867


namespace prime_digit_A_for_20240x_l489_489583

theorem prime_digit_A_for_20240x (A : ℕ) (hA : A < 10) : (20240 * 10 + A).prime ↔ A = 9 :=
by
  -- We assert the main mathematical translation here
  sorry

end prime_digit_A_for_20240x_l489_489583


namespace Crossing13Dots_l489_489334

def Points := Fin 13 -- Represents the 13 dots

-- Definition of the problem's conditions
structure ConnectDots (dots : Points) where
  lines : fin 5 → List Points -- 5 line segments each connecting dots
  connected : ∀ (i j : fin 5), i ≠ j → lines i ≠ lines j -- No lines are traced twice
  cover_all_dots : ∀ (dot : Points), ∃ (i : fin 5), dot ∈ lines i -- All dots must be intersected by some line
  no_lifting_pencil : ∀ (i : fin 5), (lines i).head? ≠ none -- Pencil cannot be lifted

theorem Crossing13Dots (D : ConnectDots Points) : 
  ∃ (L : (fin 5) → List Points), (∀ (i j : fin 5), i ≠ j → L i ≠ L j) ∧ 
    (∀ (dot : Points), ∃ (i : fin 5), dot ∈ L i) ∧ 
    (∀ (i : fin 5), (L i).head? ≠ none) :=
sorry

end Crossing13Dots_l489_489334


namespace veronica_pitting_time_is_correct_l489_489634

def veronica_pitting_time : ℝ :=
  let cherries_per_pound := 80
  let time_first_pound := 10 * (cherries_per_pound / 20)
  let time_second_pound := 8 * (cherries_per_pound / 20)
  let time_third_pound := 12 * (cherries_per_pound / 20)
  let interruption_time := 15 * 2
  let total_pitting_time := time_first_pound + time_second_pound + time_third_pound
  let total_time := total_pitting_time + interruption_time
  (total_time / 60 : ℝ)

theorem veronica_pitting_time_is_correct : veronica_pitting_time = 2.5 := by
  sorry

end veronica_pitting_time_is_correct_l489_489634


namespace men_absent_l489_489685

theorem men_absent (x : ℕ) (H1 : 10 * 6 = 60) (H2 : (10 - x) * 10 = 60) : x = 4 :=
by
  sorry

end men_absent_l489_489685


namespace company_employees_after_hiring_l489_489627

theorem company_employees_after_hiring (T : ℕ) (H : T = 286) :
  let initial_female_workers := 0.60 * T,
      initial_male_workers := 0.40 * T,
      additional_male_workers := 26,
      final_total_employees := T + additional_male_workers,
      new_female_percentage := 55,
      new_male_workers := initial_male_workers + additional_male_workers,
      new_male_percentage := 0.45 * final_total_employees
  in final_total_employees = 312 :=
by
  -- Proof goes here.
  sorry

end company_employees_after_hiring_l489_489627


namespace salary_increase_l489_489716

theorem salary_increase (new_salary : ℝ) (percent_increase : ℝ) (increase_amount : ℝ) :
  new_salary = 90000 ∧ percent_increase = 0.3846153846153846 → increase_amount = 25000 :=
by
  intros h,
  sorry

end salary_increase_l489_489716


namespace fraction_of_boys_in_clubs_l489_489528

def total_students : ℕ := 150
def girls_percent : ℚ := 0.6
def boys_percent : ℚ := 0.4
def boys_not_in_clubs : ℕ := 40

theorem fraction_of_boys_in_clubs :
  let total_boys := total_students * boys_percent
  let boys_in_clubs := total_boys - boys_not_in_clubs
  boys_in_clubs / total_boys = 1 / 3 := 
  by
    sorry

end fraction_of_boys_in_clubs_l489_489528


namespace zeros_in_nine_digit_square_l489_489399

theorem zeros_in_nine_digit_square :
  (∀ (n : ℕ), (n ≥ 1) → (∃ (k : ℕ), (number_of_nines n ^ 2 = k) ∧ (number_of_zeros k = n - 1))) →
  (number_of_zeros (999999999^2) = 8) :=
by
sorry

end zeros_in_nine_digit_square_l489_489399


namespace polygon_sides_l489_489610

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489610


namespace caltech_equilateral_triangles_l489_489733

theorem caltech_equilateral_triangles (n : ℕ) (h : n = 900) :
  let total_triangles := (n * (n - 1) / 2) * 2
  let overcounted_triangles := n / 3
  total_triangles - overcounted_triangles = 808800 :=
by
  sorry

end caltech_equilateral_triangles_l489_489733


namespace transform_to_concentric_circles_l489_489991

theorem transform_to_concentric_circles (S1 S2 : set ℝ) (h1 : is_circle S1) (h2 : is_circle S2 ∨ is_line S2) (h3 : ¬(S1 ∩ S2).nonempty) :
  ∃ (O : ℝ) (radius : ℝ), is_inverted_concentric S1 S2 O radius :=
sorry

end transform_to_concentric_circles_l489_489991


namespace range_of_a_l489_489417

open Set

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → x ≤ 3 → x^2 - a * x - a + 1 ≥ 0) ↔ a ≤ 5 / 2 :=
sorry

end range_of_a_l489_489417


namespace star_in_S_star_associative_l489_489661

def S (x : ℕ) : Prop :=
  x > 1 ∧ x % 2 = 1

def f (x : ℕ) : ℕ :=
  Nat.log2 x

def star (a b : ℕ) : ℕ :=
  a + 2 ^ (f a) * (b - 3)

theorem star_in_S (a b : ℕ) (h_a : S a) (h_b : S b) : S (star a b) :=
  sorry

theorem star_associative (a b c : ℕ) (h_a : S a) (h_b : S b) (h_c : S c) :
  star (star a b) c = star a (star b c) :=
  sorry

end star_in_S_star_associative_l489_489661


namespace percent_increase_output_per_hour_l489_489917

-- Definitions and conditions
variable (B H : ℝ) -- B is the number of bears per week, H is the number of hours per week

-- Define outputs per hour
def output_per_hour (B H : ℝ) := B / H
def new_output_per_hour (B H : ℝ) := 1.8 * B / (0.9 * H)

-- A mathematical statement to prove the percentage increase of output per hour
theorem percent_increase_output_per_hour (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((new_output_per_hour B H) - (output_per_hour B H)) / (output_per_hour B H) * 100 = 100 :=
by
  sorry

end percent_increase_output_per_hour_l489_489917


namespace sum_of_first_4_terms_l489_489450

variable {a : ℕ → ℝ} -- Define the arithmetic sequence

-- Axiom: Definition of an arithmetic sequence
axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n, a (n + 1) = a n + d

-- Axiom: Given condition a₂ + a₃ = 6
axiom a2_a3_sum (a : ℕ → ℝ) : a 1 + a 2 = 6 -- Note: a₂ is a 1, and a₃ is a 2 because Lean is 0-based

-- Theorem: We need to prove S₄ = 12
theorem sum_of_first_4_terms (a : ℕ → ℝ) (d : ℝ) [arithmetic_sequence a d] [a2_a3_sum a] : 
  a 0 + a 1 + a 2 + a 3 = 12 := 
sorry

end sum_of_first_4_terms_l489_489450


namespace polygon_sides_l489_489600

theorem polygon_sides (n : ℕ) 
    (h1 : (n-2) * 180 = 3 * 360 - 180) 
    (h2 : ∀ k, k > 2 → (k-2) * 180 = 180 * (k - 2)) 
    (h3 : 360 = 360) : n = 5 := 
by
  sorry

end polygon_sides_l489_489600


namespace current_speed_1_5_kmh_l489_489279

/--
A boat moves upstream at the rate of 1 km in 30 minutes and downstream 1 km in 12 minutes.
-/
def speed_of_current (upstream_distance : ℝ) (upstream_time : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) : ℝ :=
  let speed_upstream := upstream_distance / (upstream_time / 60)
  let speed_downstream := downstream_distance / (downstream_time / 60)
  (speed_downstream - speed_upstream) / 2

theorem current_speed_1_5_kmh :
  speed_of_current 1 30 1 12 = 1.5 :=
by
  sorry

end current_speed_1_5_kmh_l489_489279


namespace modular_expression_l489_489357

open scoped nat

def mod_inverse (a n : ℕ) : ℕ :=
  if h : gcd a n = 1 then xgcd.val (gcd.a h) % n else 0

theorem modular_expression :
  3 * mod_inverse 4 65 + 7 * mod_inverse 13 65 ≡ 47 [MOD 65] :=
by
  sorry

end modular_expression_l489_489357


namespace f_periodic_l489_489947

def floor (x : ℝ) : ℤ := int.floor x

def f (x : ℝ) : ℝ := floor x - x

theorem f_periodic : ∀ (x : ℝ), f (x + 1) = f x :=
by
  intro x
  sorry

end f_periodic_l489_489947


namespace polygon_sides_l489_489598

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l489_489598


namespace sum_infinite_geometric_series_l489_489771

theorem sum_infinite_geometric_series :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 3
  (a / (1 - r) = (3 : ℚ) / 8) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 3
  sorry

end sum_infinite_geometric_series_l489_489771


namespace basketball_starting_lineups_l489_489674

theorem basketball_starting_lineups (n_players n_guards n_forwards n_centers : ℕ)
  (h_players : n_players = 12)
  (h_guards : n_guards = 2)
  (h_forwards : n_forwards = 2)
  (h_centers : n_centers = 1) :
  (Nat.choose n_players n_guards) * (Nat.choose (n_players - n_guards) n_forwards) * (Nat.choose (n_players - n_guards - n_forwards) n_centers) = 23760 := by
  sorry

end basketball_starting_lineups_l489_489674


namespace chord_eq_diameter_l489_489182

-- Define a curve of constant width
structure CurveOfConstantWidth (R : Type) [Real] :=
  (width : ℝ)
  (is_constant : ∀ (P Q : R), P ≠ Q → (∀ tangent1 tangent2 : R, width = (distance P Q) ))

-- Define the required theorem
theorem chord_eq_diameter (R : Type) [Real] (C : CurveOfConstantWidth R) 
  (A B : R) (hAB : ∀ A B : R, C.width = (distance A B)) : 
  (is_diameter A B) :=
by 
  sorry

end chord_eq_diameter_l489_489182


namespace arithmetic_sequence_properties_l489_489077

noncomputable def a_n (a1 : ℤ) (d : ℤ) (n : ℕ): ℤ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_properties : 
  ∃ (a1 d : ℤ), a_n a1 d 5 = 3 ∧ a_n a1 d 6 = -2 ∧ 
    a1 = 23 ∧ d = -5 ∧ 
    ∀ n : ℕ, a_n 23 (-5) n = 28 - 5 * n :=
by
  use 23
  use -5
  split
  · rw [a_n, ← int.coe_nat_succ, ← int.coe_nat_succ]
    norm_num
  split
  · rw [a_n]
    norm_num
  split
  · rfl
  split
  · rfl
  intro n
  rw [a_n]
  norm_num
  sorry

end arithmetic_sequence_properties_l489_489077


namespace real_roots_imp_k_value_l489_489799

def polynomial (x : ℝ) (k : ℝ) : ℝ :=
  x^4 - 4 * x^3 + 4 * x^2 + k * x - 4

theorem real_roots_imp_k_value (h: ∀ x : ℝ, polynomial x k = 0 → is_real x) : k = -8 :=
  sorry

end real_roots_imp_k_value_l489_489799


namespace trajectory_equation_minimum_dot_product_l489_489383

open Real

def trajectory_condition (P : ℝ × ℝ) : Prop :=
  (sqrt ((P.1 - 1)^2 + P.2^2) - abs P.1) = 1

theorem trajectory_equation {P : ℝ × ℝ} (h : trajectory_condition P) :
  (P.1 ≥ 0 → P.2^2 = 4 * P.1) ∧ (P.1 < 0 → P.2 = 0) :=
sorry

theorem minimum_dot_product (k : ℝ) (hk : k ≠ 0) :
  let x₁ : ℝ := 1 + 2/k^2
  let x₂ : ℝ := 1 + 2/k
  min_val : ℝ := 16 :=
sorry

end trajectory_equation_minimum_dot_product_l489_489383


namespace product_simplification_l489_489731

theorem product_simplification :
  (10 * (1 / 5) * (1 / 2) * 4 / 2 : ℝ) = 2 :=
by
  sorry

end product_simplification_l489_489731


namespace find_ABD_l489_489555

-- Conditions and variables
variables {A B D : ℕ}
hypothesis h1 : A ≠ 0 ∧ B ≠ 0 ∧ D ≠ 0
hypothesis h2 : A ≠ B ∧ A ≠ D ∧ B ≠ D
hypothesis h3 : A < 7 ∧ B < 7 ∧ D < 7
hypothesis h4 : A * 7 + B + D = D * 7
hypothesis h5 : A * 7 + B + B * 7 + A = D * 7 + D

-- Statement to be proved
theorem find_ABD : A * 100 + B * 10 + D = 434 :=
by
  sorry

end find_ABD_l489_489555


namespace y1_y2_difference_l489_489021

-- Definitions and conditions for the problem
def ellipse (x y : ℝ) := (x^2 / 25) + (y^2 / 16) = 1
def foci := ((-3, 0), (3, 0))
def chord_passes_through_focus (A B : ℝ × ℝ) := B = (-3, 0)
def inscribed_circle_circumference := π

-- Coordinates of points A and B
variables (x1 y1 x2 y2 : ℝ)

-- Main theorem for the proof problem
theorem y1_y2_difference : 
  ellipse x1 y1 → ellipse x2 y2 → chord_passes_through_focus (x1, y1) (x2, y2) → 
  inscribed_circle_circumference = π → 
  |y1 - y2| = 5 / 4 := 
sorry

end y1_y2_difference_l489_489021


namespace B_days_to_finish_l489_489280

-- Define the conditions
def A_work_rate : ℝ := 1 / 20
def B_work_rate : ℝ := 1 / 30
def A_work_days : ℝ := 10

-- Define the question as a theorem
theorem B_days_to_finish : ℝ := 
  let remaining_work := 1 - (A_work_rate * A_work_days) in
  remaining_work / B_work_rate = 15 :=
by 
  let remaining_work := 1 - (A_work_rate * A_work_days)
  show remaining_work / B_work_rate = 15
  sorry

end B_days_to_finish_l489_489280


namespace solve_arcsin_arccos_l489_489548

open Real

theorem solve_arcsin_arccos (x : ℝ) (h_condition : - (1 / 2 : ℝ) ≤ x ∧ x ≤ 1 / 2) :
  arcsin x + arcsin (2 * x) = arccos x ↔ x = 0 :=
sorry

end solve_arcsin_arccos_l489_489548


namespace solution_set_of_inequality_l489_489849

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1 / 2)^(x - 1) else Real.logBase 3 (x + 1)

theorem solution_set_of_inequality :
  {x : ℝ | f (x + 1) - 1 > 0} = {x : ℝ | x < 0 ∨ x > 1} :=
by
  sorry

end solution_set_of_inequality_l489_489849


namespace quadratic_roots_l489_489796

-- Define the given conditions of the equation
def eqn (z : ℂ) : Prop := z^2 + 2 * z + (3 - 4 * Complex.I) = 0

-- State the theorem to prove that the roots of the equation are 2i and -2 + 2i.
theorem quadratic_roots :
  ∃ z1 z2 : ℂ, (z1 = 2 * Complex.I ∧ z2 = -2 + 2 * Complex.I) ∧ 
  (∀ z : ℂ, eqn z → z = z1 ∨ z = z2) :=
by
  sorry

end quadratic_roots_l489_489796


namespace sum_binom_eq_one_l489_489371

theorem sum_binom_eq_one (n : ℕ) (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : 
  ∑ r in Finset.range (n + 1), (Nat.choose (n + 1 + r) r) * ((1 - x) ^ (n + 1) * x ^ r + x ^ (n + 1) * (1 - x) ^ r) = 1 := 
by {
  sorry
}

end sum_binom_eq_one_l489_489371


namespace sum_of_perimeters_l489_489179

-- Given conditions
variables (A B C D : Type) 
variables (AB BC x y : ℕ)
variable [MetricSpace A]
variables (h : ℕ) (AC : ℕ := 28)
variables (AD CD : ℕ)

-- Equivalent proof problem statement
theorem sum_of_perimeters (h^2_x : h^2 = x^2 - 196)
  (h^2_y : h^2 = y^2 - 64)
  (x_y_rel : x^2 - y^2 = 132)
  (valid_pair1 : x = 34 ∧ y = 32)
  (valid_pair2 : x = 14 ∧ y = 8) : 
  2 * x + AC = 152 := by
  sorry

end sum_of_perimeters_l489_489179


namespace coordinates_of_point_P_l489_489409

theorem coordinates_of_point_P (x y : ℝ) (h1 : x > 0) (h2 : y < 0) (h3 : abs y = 2) (h4 : abs x = 4) : (x, y) = (4, -2) :=
by
  sorry

end coordinates_of_point_P_l489_489409


namespace yanna_change_l489_489651

theorem yanna_change :
  let shirt_cost := 5 in
  let sandel_cost := 3 in
  let total_money := 100 in
  let num_shirts := 10 in
  let num_sandels := 3 in
  let total_cost := (num_shirts * shirt_cost) + (num_sandels * sandel_cost) in
  let change := total_money - total_cost in
  change = 41 :=
by
  sorry

end yanna_change_l489_489651


namespace relationship_f_cos_alpha_f_sin_beta_l489_489075

variables {f : ℝ → ℝ}
variables {α β : ℝ}

-- f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

-- f is monotonically decreasing on the interval [-1, 0]
def monotonic_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b → x < y → f(x) ≥ f(y)

-- α and β are acute angles
def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < π / 2

-- Main theorem
theorem relationship_f_cos_alpha_f_sin_beta 
  (h_odd : odd_function f)
  (h_decreasing : monotonic_decreasing f (-1) 0)
  (h_alpha : acute_angle α)
  (h_beta : acute_angle β)
  (h_relation : β = π / 2 - α) :
  f (cos α) > f (sin β) := 
  sorry

end relationship_f_cos_alpha_f_sin_beta_l489_489075


namespace original_cost_of_dress_l489_489725

theorem original_cost_of_dress (x: ℝ) 
  (h1: x / 2 - 10 < x) 
  (h2: x - (x / 2 - 10) = 80) : 
  x = 140 :=
sorry

end original_cost_of_dress_l489_489725


namespace correct_expression_count_l489_489352

def expression_1 (x y : ℝ) : Prop := x^2 + xy + y^2 = (x + y)^2
def expression_2 (x y : ℝ) : Prop := -x^2 + 2xy - y^2 = -(x - y)^2
def expression_3 (x y : ℝ) : Prop := x^2 + 6xy - 9y^2 = (x - 3y)^2
def expression_4 (x : ℝ) : Prop := -x^2 + 1/4 = (1/2 + x) * (1/2 - x)

theorem correct_expression_count :
  (∃ x y, expression_2 x y) ∧ (∃ x, expression_4 x) ∧ 
  ¬ (∃ x y, expression_1 x y) ∧ ¬ (∃ x y, expression_3 x y) :=
by
  sorry

end correct_expression_count_l489_489352


namespace sum_of_coordinates_of_point_B_l489_489533

theorem sum_of_coordinates_of_point_B
  (A : ℝ × ℝ) (hA : A = (0, 0))
  (B : ℝ × ℝ) (hB : ∃ x : ℝ, B = (x, 3))
  (slope_AB : ∃ x : ℝ, (3 - 0)/(x - 0) = 3/4) :
  (∃ x : ℝ, B = (x, 3)) ∧ x + 3 = 7 :=
by
  sorry

end sum_of_coordinates_of_point_B_l489_489533


namespace range_of_f_l489_489270

def f (x : ℕ) : ℕ := 2 * x + 1

theorem range_of_f : 
  ({f x | x ∈ {1, 2, 3, 4}} = {3, 5, 7, 9}) :=
by {
  sorry
}

end range_of_f_l489_489270


namespace find_a_b_l489_489514

theorem find_a_b (a b : ℝ) (h1 : (2 - 3 * complex.I) ∈ {z : ℂ | Polynomial.aeval z (Polynomial.C a * Polynomial.X^2 + 3 * Polynomial.X + Polynomial.C b) = 0}) :
    (a, b) = (-3/2, 65/2) :=
sorry

end find_a_b_l489_489514


namespace rush_order_fee_l489_489728

theorem rush_order_fee :
  let main_meal_cost := 12.0
  let appetizer_cost := 6.0
  let num_people := 4
  let num_appetizers := 2
  let tip_rate := 0.20
  let total_spent := 77.0
  let total_meal_cost := num_people * main_meal_cost
  let total_appetizer_cost := num_appetizers * appetizer_cost
  let subtotal := total_meal_cost + total_appetizer_cost
  let tip := subtotal * tip_rate
  let total_before_rush := subtotal + tip
  total_spent - total_before_rush = 5.0 :=
by 
  let main_meal_cost := 12.0
  let appetizer_cost := 6.0
  let num_people := 4
  let num_appetizers := 2
  let tip_rate := 0.20
  let total_spent := 77.0
  let total_meal_cost := num_people * main_meal_cost
  let total_appetizer_cost := num_appetizers * appetizer_cost
  let subtotal := total_meal_cost + total_appetizer_cost
  let tip := subtotal * tip_rate
  let total_before_rush := subtotal + tip
  show total_spent - total_before_rush = 5.0 from
  sorry  -- The proof is omitted.

end rush_order_fee_l489_489728


namespace find_y_for_slope_l489_489418

theorem find_y_for_slope (y : ℝ) :
  let P := (-3 : ℝ, 5 : ℝ)
  let Q := (5 : ℝ, y)
  (y - 5) / (5 - (-3)) = -4 / 3 → y = -17 / 3 :=
sorry

end find_y_for_slope_l489_489418


namespace binomial_odd_terms_sum_l489_489122

theorem binomial_odd_terms_sum (x : ℝ) (S : ℝ) :
  let f(x) := (x - real.sqrt 2)^2006 in
  (∑ k in finset.range 2007, if k % 2 = 1 then (nat.choose 2006 k) * x^(2006 - k) * (-real.sqrt 2)^k else 0)
   = S → 
   S = -2^3008 :=
by
  sorry

end binomial_odd_terms_sum_l489_489122


namespace sum_of_three_consecutive_integers_l489_489227

theorem sum_of_three_consecutive_integers (a b c : ℤ) (h1 : a + 1 = b) (h2 : b + 1 = c) (h3 : c = 7) : a + b + c = 18 :=
sorry

end sum_of_three_consecutive_integers_l489_489227


namespace quadratic_expression_result_l489_489839

theorem quadratic_expression_result (x y : ℚ) 
  (h1 : 4 * x + y = 11) 
  (h2 : x + 4 * y = 15) : 
  13 * x^2 + 14 * x * y + 13 * y^2 = 275.2 := 
by 
  sorry

end quadratic_expression_result_l489_489839


namespace fraction_of_field_planted_l489_489353

open Real

def planted_fraction (a b : ℝ) (d : ℝ) : ℝ :=
  let area_triangle := (a * b) / 2
  let s := (2 / 7) ^ 2
  let area_planted := area_triangle - s
  area_planted / area_triangle

theorem fraction_of_field_planted :
  planted_fraction 3 4 2 = 145 / 147 :=
by sorry

end fraction_of_field_planted_l489_489353


namespace jana_winning_strategy_l489_489517

theorem jana_winning_strategy (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  (m + n) % 2 = 1 ∨ m = 1 ∨ n = 1 := sorry

end jana_winning_strategy_l489_489517


namespace smallest_collection_l489_489653

def Yoongi_collected : ℕ := 4
def Jungkook_collected : ℕ := 6 * 3
def Yuna_collected : ℕ := 5

theorem smallest_collection : Yoongi_collected = 4 ∧ Yoongi_collected ≤ Jungkook_collected ∧ Yoongi_collected ≤ Yuna_collected := by
  sorry

end smallest_collection_l489_489653


namespace digits_C_not_make_1C34_divisible_by_4_l489_489802

theorem digits_C_not_make_1C34_divisible_by_4 :
  ∀ (C : ℕ), (C ≥ 0) ∧ (C ≤ 9) → ¬ (1034 + 100 * C) % 4 = 0 :=
by sorry

end digits_C_not_make_1C34_divisible_by_4_l489_489802


namespace optimality_theorem_l489_489860

def sequence_1 := "[[[a1, a2], a3], a4]" -- 22 symbols sequence
def sequence_2 := "[[a1, a2], [a3, a4]]" -- 16 symbols sequence

def optimal_sequence := sequence_2

theorem optimality_theorem : optimal_sequence = "[[a1, a2], [a3, a4]]" :=
by
  sorry

end optimality_theorem_l489_489860


namespace area_of_enclosed_region_l489_489899

noncomputable def f (a x : ℝ) : ℝ := a * Real.sin(a * x) + Real.cos(a * x)
noncomputable def g (a : ℝ) : ℝ := Real.sqrt(a^2 + 1)

theorem area_of_enclosed_region (a : ℝ) (h : a > 0) : 
  (∃ P T : ℝ, 
     (P = (2 * Real.pi / a)) ∧ 
     (T = Real.sqrt(a^2 + 1)) ∧ 
     (∫ x in 0..P, g a - f a x) = (2 * Real.pi / a) * Real.sqrt(a^2 + 1)) :=
sorry

end area_of_enclosed_region_l489_489899


namespace geometric_sequence_sum_l489_489903

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (ha1 : q ≠ 0)
  (h1 : a 1 + a 2 = 3) 
  (h2 : a 3 + a 4 = (a 1 + a 2) * q^2)
  : a 5 + a 6 = 48 :=
by
  sorry

end geometric_sequence_sum_l489_489903


namespace parallelepiped_l489_489759

variables (i j k : ℝ → ℝ)
variables (u v w : ℝ → ℝ)

def PQ : ℝ → ℝ := λ t, 2 * i t + j t
def PR : ℝ → ℝ := λ t, i t + 2 * j t + k t
def PT : ℝ → ℝ := λ t, 3 * k t

def norm_squared (v : ℝ → ℝ) : ℝ := (v 1) • (v 1)

def QT_squared : ℝ := norm_squared (λ t, 3 * k t + i t + 2 * j t + k t - (2 * i t + j t))
def RV_squared : ℝ := norm_squared (λ t, 3 * k t + (2 * i t + j t) - (i t + 2 * j t + k t)) 
def PS_squared : ℝ := norm_squared (λ t, (2 * i t + j t) + (i t + 2 * j t + k t))
def UW_squared : ℝ := norm_squared (λ t, 3 * k t)

def sum_diagonal_squares : ℝ := QT_squared + RV_squared + PS_squared + UW_squared
def sum_side_squares : ℝ := norm_squared PT + norm_squared PQ + norm_squared PR

theorem parallelepiped : (sum_diagonal_squares / sum_side_squares) = 3.55 := by
  sorry

end parallelepiped_l489_489759


namespace mike_and_john_lunch_total_cost_l489_489172

theorem mike_and_john_lunch_total_cost :
  let side_salad := 2.50
  let cheesy_fries := 4.25
  let diet_cola := 2.00
  let guacamole := 3.00
  let lemonade := 1.75
  let nachos := 3.50
  let m_additional := side_salad + cheesy_fries + diet_cola + guacamole
  let j_additional := lemonade + nachos
  let j_total := 23.50  -- From the solution steps
  let m_total := 1.5 * j_total
  m_total + j_total = 58.75 :=
by
  let side_salad := 2.50
  let cheesy_fries := 4.25
  let diet_cola := 2.00
  let guacamole := 3.00
  let lemonade := 1.75
  let nachos := 3.50
  let m_additional := side_salad + cheesy_fries + diet_cola + guacamole
  let j_additional := lemonade + nachos
  let j_total := 23.50 
  let m_total := 1.5 * j_total
  calc
  m_total + j_total = 35.25 + 23.50 : by sorry
  ... = 58.75 : by sorry

end mike_and_john_lunch_total_cost_l489_489172


namespace original_radius_eq_l489_489490

-- Definitions based on the conditions
variables (r y : ℝ)
def original_height : ℝ := 4
def volume_increase_with_radius (r : ℝ) : ℝ := 16 * π * r + 16 * π
def volume_increase_with_height (r : ℝ) : ℝ := 4 * π * r^2

-- Theorem statement to prove the original radius
theorem original_radius_eq :
  (volume_increase_with_radius r = y ∧ volume_increase_with_height r = y) → 
  r = 2 + 2 * real.sqrt 2 :=
sorry

end original_radius_eq_l489_489490


namespace sum_of_digits_d_l489_489527

theorem sum_of_digits_d (d : ℕ) (h₁ : d = 200) : (2 + 0 + 0 = 2) :=
by
  have hd : digit_sum d = 2 := by sorry
  exact hd

end sum_of_digits_d_l489_489527


namespace length_of_single_row_l489_489314

-- Define smaller cube properties and larger cube properties
def side_length_smaller_cube : ℕ := 5  -- in cm
def side_length_larger_cube : ℕ := 100  -- converted from 1 meter to cm

-- Prove that the row of smaller cubes is 400 meters long
theorem length_of_single_row :
  let num_smaller_cubes := (side_length_larger_cube / side_length_smaller_cube) ^ 3
  let length_in_cm := num_smaller_cubes * side_length_smaller_cube
  let length_in_m := length_in_cm / 100
  length_in_m = 400 :=
by
  sorry

end length_of_single_row_l489_489314


namespace no_solution_x_to_2n_plus_y_to_2n_eq_z_sq_l489_489405

theorem no_solution_x_to_2n_plus_y_to_2n_eq_z_sq (n : ℕ) (h : ∀ (x y z : ℕ), x^n + y^n ≠ z^n) : ∀ (x y z : ℕ), x^(2*n) + y^(2*n) ≠ z^2 :=
by 
  intro x y z
  sorry

end no_solution_x_to_2n_plus_y_to_2n_eq_z_sq_l489_489405


namespace ellipse_perimeter_l489_489833

noncomputable def perimeter_of_triangle (a b : ℝ) (e : ℝ) : ℝ :=
  if (b = 4 ∧ e = 3 / 5 ∧ a = b / (1 - e^2) ^ (1 / 2))
  then 4 * a
  else 0

theorem ellipse_perimeter :
  let a : ℝ := 5
  let b : ℝ := 4
  let e : ℝ := 3 / 5
  4 * a = 20 :=
by
  sorry

end ellipse_perimeter_l489_489833


namespace non_similar_triangles_in_arith_prog_l489_489097

theorem non_similar_triangles_in_arith_prog (d : ℕ) :
  (∃ (angles : ℕ × ℕ × ℕ),
    let a := angles.1, let b := angles.2.1, let c := angles.2.2 in
    0 < a ∧ a < b ∧ b < c ∧ c < 180 ∧ b % 20 = 0 ∧ 
    a + b + c = 180 ∧
    b - a = c - b ∧ 
    b = 60 ∧ 
    ∀ (d : ℕ), 1 ≤ d ∧ d < 60) → 
    (∑ d in (finset.range 59).filter (λ x, x + 1 < 60), 1) = 59 :=
by
  sorry

end non_similar_triangles_in_arith_prog_l489_489097


namespace no_such_function_exists_l489_489346

theorem no_such_function_exists (f : ℕ → ℕ) : ¬ (∀ n : ℕ, n ≥ 2 → f (f (n - 1)) = f (n + 1) - f n) :=
sorry

end no_such_function_exists_l489_489346


namespace construct_largest_area_triangle_l489_489390

noncomputable def largest_area_triangle (a b : ℝ) (i : ℝ × ℝ) : Prop :=
∃ P Q : ℝ × ℝ, 
  (is_parallel P Q i) ∧ 
  (forms_triangle_with_center P Q (0, 0)) ∧ 
  (area_of_triangle (0, 0) P Q = (a * b) / 2)

theorem construct_largest_area_triangle 
  (a b : ℝ) (i : ℝ × ℝ) (h_ellipse : True) : largest_area_triangle a b i := 
by
  sorry

end construct_largest_area_triangle_l489_489390


namespace first_student_percentage_l489_489428

theorem first_student_percentage :
  ∀ (n_correct total: ℕ), 
  n_correct = 38 → total = 40 → 
  (n_correct / total.to_float) * 100 = 95 := 
by
  intros n_correct total hn ht
  sorry

end first_student_percentage_l489_489428


namespace arccos_neg_one_eq_pi_l489_489741

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi_l489_489741


namespace find_first_year_l489_489641

-- Define sum of digits
def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100) % 10 + (n / 10) % 10 + n % 10

-- Define the conditions
def after_2020 (n : ℕ) : Prop := n > 2020
def sum_of_digits_eq (n required_sum : ℕ) : Prop := sum_of_digits n = required_sum

noncomputable def first_year_after_2020_with_digit_sum_15 : ℕ :=
  2049

-- The statement to be proved
theorem find_first_year : 
  ∃ y : ℕ, after_2020 y ∧ sum_of_digits_eq y 15 ∧ y = first_year_after_2020_with_digit_sum_15 :=
by
  sorry

end find_first_year_l489_489641


namespace Wayne_initially_collected_blocks_l489_489241

-- Let's denote the initial blocks collected by Wayne as 'w'.
-- According to the problem:
-- - Wayne's father gave him 6 more blocks.
-- - He now has 15 blocks in total.
--
-- We need to prove that the initial number of blocks Wayne collected (w) is 9.

theorem Wayne_initially_collected_blocks : 
  ∃ w : ℕ, (w + 6 = 15) ↔ (w = 9) := by
  sorry

end Wayne_initially_collected_blocks_l489_489241


namespace divergence_of_vector_field_l489_489238

noncomputable def vector_field (x : ℝ) : ℝ → ℝ := λ x, x

def sphere_radius (ε : ℝ) : ℝ := (4 / 3) * π * ε^3

def divergence_at_origin (ε : ℝ) (a : ℝ → ℝ) : ℝ :=
  let surface_integral := (∫ (σ : set ℝ), (a σ) * (σ / |σ|)) in
  (surface_integral / sphere_radius ε)

theorem divergence_of_vector_field : ∀ (a : ℝ → ℝ),
  (a = vector_field) →
  ∀ ε > 0,
  (divergence_at_origin ε a) = 1 := 
by
  intros a ha ε ε_pos
  have h_surface_integral : (∫ (σ : set ℝ), (a σ) * (σ / |σ|)) = (4 / 3) * π * ε^3,
  sorry
  rw [divergence_at_origin, h_surface_integral, sphere_radius]
  field_simp [ε_pos],
  rw [divergence_at_origin],
  field_simp [sphere_radius, ε_pos],
  sorry -- further simplifications here.

end divergence_of_vector_field_l489_489238


namespace expression_approx_l489_489884

noncomputable def x : ℝ := 478593.65 / 785412.55
noncomputable def y : ℝ := 695745.22 / 375421.77
noncomputable def expression := (y^3 / x^2) * real.sqrt (x * y)

theorem expression_approx : abs (expression - 18.2215) < 0.0001 := by
  sorry

end expression_approx_l489_489884


namespace g_g_x_eq_3_solutions_count_l489_489959

def g (x : ℝ) : ℝ :=
  if x < 0 then -x + 2 else 3 * x - 7

theorem g_g_x_eq_3_solutions_count :
  ∃ xs : ℝ, g (g xs) = 3 ∧ (xs = 2 ∨ xs = -4/3 ∨ xs = 31/9) :=
by
  sorry

end g_g_x_eq_3_solutions_count_l489_489959


namespace extended_triangle_area_l489_489720

theorem extended_triangle_area (S : ℝ) (k : ℝ) (h₁ : k = 2):
  let A :=  S in
  let A' := (k^2) * A in
  A' = 4 * A :=
by
  intro
  have h₂ := calc
    A' = k^2 * A : by sorry
  have h₃ := show k^2 = 4, from by
    calc
      k^2 = 2^2 : by rw [h₁]
      ... = 4 : by sorry
  rw [h₃] at h₂
  sorry

end extended_triangle_area_l489_489720


namespace population_ratio_X_to_Z_l489_489738

variable {Z : ℕ} -- Assume the population is a natural number for simplicity

-- Definitions based on conditions
def population_city_Z := Z
def population_city_Y := 2 * population_city_Z
def population_city_X := 3 * population_city_Y

-- The goal is to prove the ratio of populations
theorem population_ratio_X_to_Z (Z : ℕ) : population_city_X / population_city_Z = 6 :=
by
  -- Sorry added to skip the proof
  sorry

end population_ratio_X_to_Z_l489_489738


namespace value_of_g_at_3_l489_489439

def g (x : ℝ) := x^2 - 2*x + 1

theorem value_of_g_at_3 : g 3 = 4 :=
by
  sorry

end value_of_g_at_3_l489_489439


namespace polygon_sides_l489_489594

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l489_489594


namespace percent_increase_output_l489_489930

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l489_489930


namespace sum_of_values_of_z_l489_489153

def f (x : ℝ) := x^2 - 2*x + 3

theorem sum_of_values_of_z (z : ℝ) (h : f (5 * z) = 7) : z = 2 / 25 :=
sorry

end sum_of_values_of_z_l489_489153


namespace smallest_prime_digit_sum_28_l489_489644

theorem smallest_prime_digit_sum_28 : ∃ p : ℕ, p.prime ∧ digits_sum p = 28 ∧ ∀ q : ℕ, q < p → q.prime → digits_sum q ≠ 28 :=
by
  sorry

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

end smallest_prime_digit_sum_28_l489_489644


namespace todd_time_difference_l489_489729

theorem todd_time_difference {Brian_time Todd_time : ℝ} (hBrian : Brian_time = 96) (hTodd : Todd_time = 88) :
  Brian_time - Todd_time = 8 :=
by
  rw [hBrian, hTodd]
  norm_num
  sorry

end todd_time_difference_l489_489729


namespace assistant_increases_output_by_100_percent_l489_489937

theorem assistant_increases_output_by_100_percent (B H : ℝ) (H_pos : H > 0) (B_pos : B > 0) :
  (1.8 * B) / (0.9 * H) = 2 * (B / H) := 
sorry

end assistant_increases_output_by_100_percent_l489_489937


namespace find_ellipse_standard_eq_find_line_eq_l489_489834

noncomputable def ellipse_standard_eq (a b c : ℝ) (h1 : a = sqrt 2 * b) (h2 : c = 1) (h3 : b^2 + c^2 = a^2) : Prop :=
  ∃ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

theorem find_ellipse_standard_eq :
  ellipse_standard_eq (sqrt 2 * 1) 1 1
  (by { rw sqrt_eq_rpow, norm_num, }) 
  (by norm_num) 
  (by { norm_num }) :=
  sorry

noncomputable def line_eq (k : ℝ) : Prop :=
  ∃ x y : ℝ, ∃ P Q : ℝ × ℝ, P = (2, 0) ∧ x = Q.1 ∧ (y = Q.2 ∧ ((3 / 2 * abs(y - y)) = sqrt 10 / 2) ∧
  ((Q = (x, k * (x +1)) ∧ ((x + sqrt 2 * y + 1 = 0) ∨ (x - sqrt 2 * y + 1 = 0)))))

theorem find_line_eq (k : ℝ) :
  line_eq k :=
  sorry

end find_ellipse_standard_eq_find_line_eq_l489_489834


namespace polygon_sides_l489_489609

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489609


namespace three_digit_numbers_sum_seven_l489_489013

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489013


namespace find_alpha_l489_489342

-- Define the problem in Lean terms
variable (x y α : ℝ)

-- Conditions
def condition1 : Prop := 3 + α + y = 4 + α + x
def condition2 : Prop := 1 + x + 3 + 3 + α + y + 4 + 1 = 2 * (4 + α + x)

-- The theorem to prove
theorem find_alpha (h1 : condition1 x y α) (h2 : condition2 x y α) : α = 5 := 
  sorry

end find_alpha_l489_489342


namespace ellipse_equation_l489_489071

-- Definitions from conditions
def ecc (e : ℝ) := e = Real.sqrt 3 / 2
def parabola_focus (c : ℝ) (a : ℝ) := c = Real.sqrt 3 ∧ a = 2
def b_val (b a c : ℝ) := b = Real.sqrt (a^2 - c^2)

-- Main problem statement
theorem ellipse_equation (e a b c : ℝ) (x y : ℝ) :
  ecc e → parabola_focus c a → b_val b a c → (x^2 + y^2 / 4 = 1) := 
by
  intros h1 h2 h3
  sorry

end ellipse_equation_l489_489071


namespace min_value_frac_l489_489570

theorem min_value_frac (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2 * m + n = 1) : 
  (1 / m + 2 / n) = 8 :=
sorry

end min_value_frac_l489_489570


namespace collinear_iff_perpendicular_iff_l489_489054

variables {a : ℝ}
def A (a : ℝ) := (1, -2 * a)
def B (a : ℝ) := (2, a)
def C (a : ℝ) := (2 + a, 0)
def D (a : ℝ) := (2 * a, 1)

def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

theorem collinear_iff (a : ℝ) : 
  slope (A a) (B a) = slope (B a) (C a) ↔ a = -1/3 :=
by sorry

theorem perpendicular_iff (a : ℝ) : 
  slope (A a) (B a) * slope (C a) (D a) = -1 ↔ a = 1/2 :=
by sorry

end collinear_iff_perpendicular_iff_l489_489054


namespace square_area_l489_489571

noncomputable def side_length1 (x : ℝ) : ℝ := 5 * x - 20
noncomputable def side_length2 (x : ℝ) : ℝ := 25 - 2 * x

theorem square_area (x : ℝ) (h : side_length1 x = side_length2 x) :
  (side_length1 x)^2 = 7225 / 49 :=
by
  sorry

end square_area_l489_489571


namespace range_of_m_l489_489419

def p (m : ℝ) : Prop :=
  (real.sqrt ((m + 3) / 3) > real.sqrt 2)

def q (m : ℝ) : Prop :=
  (m / 2 > m - 2 ∧ m - 2 > 0)

theorem range_of_m {m : ℝ} (hp : p m) (hq : q m) : 3 < m ∧ m < 4 :=
by
  have h1 : (m > 3) := sorry
  have h2 : (m < 4) := sorry
  exact ⟨h1, h2⟩ 

end range_of_m_l489_489419


namespace question_1_question_2a_question_2b_question_3_l489_489416

-- Define the given functions
def f (x : ℝ) : ℝ := log x - real.exp (1 - x)
def g (a : ℝ) (x : ℝ) : ℝ := a * (x^2 - 1) - 1 / x

-- Define h function as per question (2)
def h (a : ℝ) (x : ℝ) : ℝ := g(a,x) - f(x) + (real.exp(x) - real.exp(1) * x) / (x * real.exp(x))

-- Define the conditions and the proof problem
theorem question_1 : ∃! x : ℝ, 1 < x ∧ x < real.exp 1 ∧ f(x) = 0 := sorry

theorem question_2a {a : ℝ} (h_le_nonpos : a ≤ 0) : ∀ x : ℝ, 0 < x → deriv (h a) x < 0 := sorry

theorem question_2b {a : ℝ} (h_pos : a > 0) : ∀ x : ℝ, 
  if x < 1 / real.sqrt (2 * a)
  then (deriv (h a)) x < 0
  else (deriv (h a)) x > 0 := sorry

theorem question_3 (a : ℝ) : (∀ x : ℝ, 1 < x → f(x) < g(a, x)) ↔ a ∈ set.Ici (1/2) := sorry

end question_1_question_2a_question_2b_question_3_l489_489416


namespace circle_center_is_21_l489_489785

theorem circle_center_is_21 : ∀ x y : ℝ, x^2 + y^2 - 4 * x - 2 * y - 5 = 0 →
                                      ∃ h k : ℝ, h = 2 ∧ k = 1 ∧ (x - h)^2 + (y - k)^2 = 10 :=
by
  intro x y h_eq
  sorry

end circle_center_is_21_l489_489785


namespace polygon_sides_l489_489589

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l489_489589


namespace find_angle_A_find_max_area_l489_489470

/-
  In an acute triangle ΔABC, with sides opposite angles A, B, and C being a, b, and c respectively,
  it is given that b/2 is the arithmetic mean of 2a sin A cos C and c sin 2A.
  Prove that the measure of angle A is π/6.
-/
theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h1 : sin A = 1 / 2) (h2 : 0 < A) (h3 : A < π / 2)
(h4: b / 2 = (2 * a * sin A * cos C + c * sin (2 * A)) / 2) : A = π / 6 := 
sorry

/-
  If a = 2, prove that the maximum area of the triangle ΔABC is 2+√3.
-/
theorem find_max_area (a b c : ℝ) (A B C : ℝ)
(h1: a = 2)
(h2: ∀ (b c : ℝ), b^2 + c^2 - 2*b*c*cos A ≥ b^2 + c^2 - sqrt 3 * b * c)
(h3: S = 1/2 * b * c * sin A)
: S ≤ 2 + sqrt 3 :=
sorry

end find_angle_A_find_max_area_l489_489470


namespace angle_OCB_30_degrees_l489_489025

-- Define the points and tangent properties based on the given conditions
variables {A O B C D F E : Type*}
variables {circle : Set O} [IsCircle circle]
variables (A O B C D F E : Point circle)

-- Define the specific conditions given in the problem
-- 1. Point A is outside a circle with center O
variable (outside_circle : ∃ (P : Point circle), P = A ∧ ¬(A ∈ circle))

-- 2. Tangents AB and AC from point A touching the circle at B and C respectively
variable (tangent1 : IsTangent A B circle)
variable (tangent2 : IsTangent A C circle)

-- 3. Line segment AO intersects the circle at point D and segment BC at point F
variable (intersect_AO_circle : ∃ (P : Point circle), P = D ∧ LineSegment A O ∩ circle = {P})
variable (intersect_AO_BC : ∃ (P : LineSegment B C), P = F ∧ LineSegment A O ∩ LineSegment B C = {P})

-- 4. Line BD intersects segment AC at point E
variable (intersect_BD_AC : ∃ (P : LineSegment A C), P = E ∧ LineSegment B D ∩ LineSegment A C = {P})

-- 5. The area of quadrilateral DECF is equal to the area of triangle ABD
variable (area_equality : Area quadrilateral D E C F = Area triangle A B D)

-- Question: Prove that the angle OCB is 30 degrees
theorem angle_OCB_30_degrees : angle O C B = 30 := sorry

end angle_OCB_30_degrees_l489_489025


namespace unique_not_in_range_l489_489568

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ :=
  (p * x + q) / (r * x + s)

theorem unique_not_in_range (p q r s : ℝ) (h₁ : p ≠ 0) (h₂ : q ≠ 0) (h₃ : r ≠ 0) (h₄ : s ≠ 0)
  (h₅ : g p q r s 5 = 5) (h₆ : g p q r s 13 = 13) (h₇ : ∀ x, x ≠ (-s / r) → g p q r s (g p q r s x) = x) :
  ∃! c : ℝ, ∀ x : ℝ, g p q r s x ≠ c := 
begin
  use 9,
  intro x,
  sorry
end

end unique_not_in_range_l489_489568


namespace prove_angle_CNB_l489_489891

noncomputable def isosceles_triangle_angle (α β : ℝ) (x y : ℝ) (A B C N : Type) 
  [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint N] :=
  IsIsoscelesTriangle A B C ∧ AB = BC ∧ ∠BAC = α ∧
  ∠BAN = β ∧ ∠ABN = x ∧ x + y + ∠BNA = 180 ∧ 
  y + ∠BNA + ∠CNB = 360

theorem prove_angle_CNB : 
  ∀ (A B C N : Type) [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint N],
  isosceles_triangle_angle 100 13 17 170 A B C N :=
begin
  sorry,
end

end prove_angle_CNB_l489_489891


namespace trigo_condition_and_value_l489_489508

noncomputable def tan_x (x : ℝ) : ℝ := Real.tan x
noncomputable def tan_2x (x : ℝ) : ℝ := Real.tan (2 * x)
noncomputable def tan_3x (x : ℝ) : ℝ := Real.tan (3 * x)

theorem trigo_condition_and_value (x a b c : ℝ) (h₁ : tan_x x = a) 
  (h₂ : tan_2x x = b) (h₃ : tan_3x x = c) 
  (h₄ : Real.cos (3 * x) ≠ 0) (h₅ : Real.cos (2 * x) ≠ 0) 
  (h₆ : Real.cos x ≠ 0) :
  (a^3 + b^3 + c^3 = (a + b + c)^3 ↔ (a+b)(a+c)(b+c) = 0) ∧ 
  (∃ m n : ℤ, x = (↑m * Real.pi) / 3 ∨ x = (↑n * Real.pi) / 5) :=
by
  sorry

end trigo_condition_and_value_l489_489508


namespace more_kids_stayed_home_l489_489768

-- Define the given data
def kids_10_12 : ℕ := 450000
def kids_13_15 : ℕ := 600000
def kids_16_18 : ℕ := 300000

def camp_perc_10_12 : ℝ := 0.12
def camp_perc_13_15 : ℝ := 0.20
def camp_perc_16_18 : ℝ := 0.05

-- Calculate kids attending camps and staying home
def attend_camp (total : ℕ) (perc : ℝ) : ℕ :=
  (perc * (total : ℝ)).to_nat

def stay_home (total attend : ℕ) : ℕ := 
  total - attend

def camp_10_12 := attend_camp kids_10_12 camp_perc_10_12
def camp_13_15 := attend_camp kids_13_15 camp_perc_13_15
def camp_16_18 := attend_camp kids_16_18 camp_perc_16_18

def home_10_12 := stay_home kids_10_12 camp_10_12
def home_13_15 := stay_home kids_13_15 camp_13_15
def home_16_18 := stay_home kids_16_18 camp_16_18

-- Calculate the total difference
def diff (home attend : ℕ) : ℕ := 
  home - attend

def total_diff : ℕ :=
  diff home_10_12 camp_10_12 + 
  diff home_13_15 camp_13_15 + 
  diff home_16_18 camp_16_18

-- The theorem to be proven
theorem more_kids_stayed_home :
  total_diff = 972000 :=
by
  sorry

end more_kids_stayed_home_l489_489768


namespace find_inner_circle_radius_l489_489299

-- Definitions based on the problem conditions
def square_side_length : ℝ := 4
def semicircle_radius : ℝ := 2 / 3
def expected_circle_radius : ℝ := Real.sqrt 2

-- Statement of the problem in Lean
theorem find_inner_circle_radius :
  ∃ r : ℝ,
    (square_side_length = 4) ∧
    (semicircle_radius = 2 / 3) ∧
    (∀ i : Fin 12, tangent_to_all_semicircles r) ∧
    (r = expected_circle_radius) :=
sorry

-- Placeholder for a definition of "tangent_to_all_semicircles"
-- which must be defined in terms of problem conditions
def tangent_to_all_semicircles (r : ℝ) : Prop :=
sorry

end find_inner_circle_radius_l489_489299


namespace bug_crawl_distance_l489_489680

-- Define the conditions
def initial_position : ℤ := -2
def first_move : ℤ := -6
def second_move : ℤ := 5

-- Define the absolute difference function (distance on a number line)
def abs_diff (a b : ℤ) : ℤ :=
  abs (b - a)

-- Define the total distance crawled function
def total_distance (p1 p2 p3 : ℤ) : ℤ :=
  abs_diff p1 p2 + abs_diff p2 p3

-- Prove that total distance starting at -2, moving to -6, and then to 5 is 15 units
theorem bug_crawl_distance : total_distance initial_position first_move second_move = 15 := by
  sorry

end bug_crawl_distance_l489_489680


namespace largest_real_root_l489_489275

noncomputable def mediterranean_polynomial (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) : Polynomial ℝ :=
Polynomial.C a0 + Polynomial.C a1 * X + Polynomial.C a2 * X^2 +
Polynomial.C a3 * X^3 + Polynomial.C a4 * X^4 +
Polynomial.C a5 * X^5 + Polynomial.C a6 * X^6 +
Polynomial.C a7 * X^7 + Polynomial.C 135 * X^8 +
Polynomial.C (-20) * X^9 + Polynomial.C 1 * X^10

theorem largest_real_root (a0 a1 a2 a3 a4 a5 a6 a7 : ℝ) (α : ℝ) (h : α ∈ (mediterranean_polynomial a0 a1 a2 a3 a4 a5 a6 a7).roots) :
  α ≤ 11 :=
by
  sorry

end largest_real_root_l489_489275


namespace positive_difference_perimeters_l489_489235

-- Definition for the perimeter of the first figure
def perimeter_figure1 := 2 * (4 + 2) + 3 

-- Definition for the perimeter of the second figure
def perimeter_figure2 := 2 * (6 + 2)

-- Definition for the positive difference between the perimeters
def positive_difference := abs (perimeter_figure1 - perimeter_figure2)

-- The main theorem stating that the positive difference is 1 unit
theorem positive_difference_perimeters : positive_difference = 1 := 
by 
  sorry -- proof not required

end positive_difference_perimeters_l489_489235


namespace no_valid_decomposition_l489_489123

def cell_value : Type := Prop

def dominos_prop (board : matrix (fin 6) (fin 6) cell_value)
                 (pair : (fin 6 × fin 6) → (fin 6 × fin 6)) : Prop :=
∀ (i j : fin 6), cell_value (board i j) ∧
               (i ≠ (pair (i, j)).1 ∨ j ≠ (pair (i, j)).2) ∧
               ((i, j) = ((pair (i, j)).1, (pair (i, j)).2))

def board_sum (board : matrix (fin 6) (fin 6) cell_value) : Prop :=
∃ (pair : (fin 6 × fin 6) → (fin 6 × fin 6)),
  dominos_prop board pair

theorem no_valid_decomposition :
  ¬ ∃ (board : matrix (fin 6) (fin 6) cell_value), board_sum board :=
sorry

end no_valid_decomposition_l489_489123


namespace number_of_five_digit_even_numbers_l489_489816

theorem number_of_five_digit_even_numbers : 
  let digits := {1, 2, 3, 4, 5, 6}
  let n := 5
  let evens := {2, 4, 6}
  (∃ (selected : Finset ℕ), selected.card = n ∧ selected ⊆ digits) → 
  (∃ (even_unit : ℕ), even_unit ∈ evens) →
  ∃ number_count : ℕ, number_count = 360 :=
by
  sorry

end number_of_five_digit_even_numbers_l489_489816


namespace quadratic_roots_l489_489795

-- Define the given conditions of the equation
def eqn (z : ℂ) : Prop := z^2 + 2 * z + (3 - 4 * Complex.I) = 0

-- State the theorem to prove that the roots of the equation are 2i and -2 + 2i.
theorem quadratic_roots :
  ∃ z1 z2 : ℂ, (z1 = 2 * Complex.I ∧ z2 = -2 + 2 * Complex.I) ∧ 
  (∀ z : ℂ, eqn z → z = z1 ∨ z = z2) :=
by
  sorry

end quadratic_roots_l489_489795


namespace total_expense_l489_489735

theorem total_expense (tanya_face_cost : ℕ) (tanya_face_qty : ℕ) (tanya_body_cost : ℕ) (tanya_body_qty : ℕ) 
  (tanya_total_expense : ℕ) (christy_multiplier : ℕ) (christy_total_expense : ℕ) (total_expense : ℕ) :
  tanya_face_cost = 50 →
  tanya_face_qty = 2 →
  tanya_body_cost = 60 →
  tanya_body_qty = 4 →
  tanya_total_expense = tanya_face_qty * tanya_face_cost + tanya_body_qty * tanya_body_cost →
  christy_multiplier = 2 →
  christy_total_expense = christy_multiplier * tanya_total_expense →
  total_expense = christy_total_expense + tanya_total_expense →
  total_expense = 1020 :=
by
  intros
  sorry

end total_expense_l489_489735


namespace P_contains_zero_and_two_l489_489701

noncomputable def P : Set ℤ := {x | x ∈ {y | y > 0} ∪ {y | y < 0} ∪ {y | even y} ∪ {y | odd y}}  -- A set of integers

axiom P_property1 : ∀ x y : ℤ, x ∈ P → y ∈ P → x + y ∈ P
axiom P_property2 : ∃ x : ℤ, x > 0 ∧ x ∈ P ∧ ∃ y : ℤ, y < 0 ∧ y ∈ P
axiom P_property3 : ∃ x : ℤ, odd x ∧ x ∈ P ∧ ∃ y : ℤ, even y ∧ y ∈ P
axiom P_property4 : -1 ∉ P

theorem P_contains_zero_and_two : 0 ∈ P ∧ 2 ∈ P :=
  sorry

end P_contains_zero_and_two_l489_489701


namespace inequality_solution_l489_489361

theorem inequality_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : (x * (x + 1)) / ((x - 3)^2) ≥ 8) : 3 < x ∧ x ≤ 24/7 :=
sorry

end inequality_solution_l489_489361


namespace find_N_l489_489101

theorem find_N (N : ℕ) : 
  981 + 983 + 985 + 987 + 989 + 991 + 993 = 7000 - N → N = 91 :=
by
  assume h : 981 + 983 + 985 + 987 + 989 + 991 + 993 = 7000 - N
  sorry

end find_N_l489_489101


namespace max_alpha_value_l489_489509

variable (a b x y α : ℝ)

theorem max_alpha_value (h1 : a = 2 * b)
    (h2 : a^2 + y^2 = b^2 + x^2)
    (h3 : b^2 + x^2 = (a - x)^2 + (b - y)^2)
    (h4 : 0 ≤ x) (h5 : x < a) (h6 : 0 ≤ y) (h7 : y < b) :
    α = a / b → α^2 = 4 := 
by
  sorry

end max_alpha_value_l489_489509


namespace exists_ray_K_l489_489637

variables (O_1 O_2 : E) (r_1 r_2 : ℝ) (vertex : E)

-- Define the plane geometric context
variable [metric_space E]

-- Assume properties of circles and the infinite angle
variables (gray_side black_side : set E)
variables (angle_position : set E)

-- Additional conditions: the circles are non-intersecting and the infinite angle satisfies the problem's constraints
axiom non_intersecting : disjoint (metric.ball O_1 r_1) (metric.ball O_2 r_2)
axiom gray_touches_gray_circle : ∀ pos ∈ angle_position, ¬(vertex ∈ gray_side ∩ ¬gray_side ∩ metric.sphere O_1 r_1)
axiom black_touches_black_circle : ∀ pos ∈ angle_position, ¬(vertex ∈ black_side ∩ black_side ∩ metric.sphere O_2 r_2)
axiom tangency_not_at_vertex : ∀ pos ∈ angle_position, (∀ p ∈ gray_side ∩ black_side, p ≠ vertex)

-- Theorem statement: existence of such point K
theorem exists_ray_K (A' K : E) :
  (∃ K, ∀ pos ∈ angle_position,
    ∃ (K_pos : set E), K = K_pos ∧ ∀ (d1 d2 : ℝ), 
    d1 / d2 = r_1 / r_2 ∧ 
    ∃ (p1 ∈ gray_side ∩ metric.sphere O_1 r_1) (p2 ∈ black_side ∩ metric.sphere O_2 r_2), 
    metric.d K p1 / metric.d K p2 = r_1 / r_2) :=
sorry

end exists_ray_K_l489_489637


namespace K9_le_89_K9_example_171_l489_489229

section weights_proof

def K (n : ℕ) (P : ℕ) : ℕ := sorry -- Assume the definition of K given by the problem

theorem K9_le_89 : ∀ P, K 9 P ≤ 89 := by
  sorry -- Proof to be filled

def example_weight : ℕ := 171

theorem K9_example_171 : K 9 example_weight = 89 := by
  sorry -- Proof to be filled

end weights_proof

end K9_le_89_K9_example_171_l489_489229


namespace square_areas_l489_489905

theorem square_areas (s1 s2 s3 : ℕ)
  (h1 : s3 = s2 + 1)
  (h2 : s3 = s1 + 2)
  (h3 : s2 = 18)
  (h4 : s1 = s2 - 1) :
  s3^2 = 361 ∧ s2^2 = 324 ∧ s1^2 = 289 :=
by {
sorry
}

end square_areas_l489_489905


namespace minimum_value_am_bn_l489_489667

-- Definitions and conditions
variables {a b m n : ℝ}
variables (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < m) (h₃ : 0 < n)
variables (h₄ : a + b = 1) (h₅ : m * n = 2)

-- Statement of the proof problem
theorem minimum_value_am_bn :
  ∃ c, (∀ a b m n : ℝ, 0 < a → 0 < b → 0 < m → 0 < n → a + b = 1 → m * n = 2 → (am * bn) * (bm * an) ≥ c) ∧ c = 2 :=
sorry

end minimum_value_am_bn_l489_489667


namespace largest_number_of_cakes_l489_489232

theorem largest_number_of_cakes : ∃ (c : ℕ), c = 65 :=
by
  sorry

end largest_number_of_cakes_l489_489232


namespace num_three_digit_sums7_l489_489001

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l489_489001


namespace volume_of_pyramid_l489_489222

noncomputable
def V (x y z : ℝ) : ℝ :=
  255 * 115 / (3 * Real.sqrt 193)

theorem volume_of_pyramid :
  ∀ (Q E F G H : ℝ × ℝ × ℝ),
  E = (0, Real.sqrt 578, 0) →
  G = (7.5 * Real.sqrt 2, 0, 0) →
  H = (-7.5 * Real.sqrt 2, 0, 0) →
  Q = (0, 351 / (2 * Real.sqrt 578), 115 / Real.sqrt 193) →
  let base_area := (1/2) * 15 * Real.sqrt 2 * 17 * Real.sqrt 2 in
  V 0 (351 / (2 * Real.sqrt 578)) (115 / (Real.sqrt 193)) = 255 * 115 / (3 * Real.sqrt 193) :=
by
  intros
  sorry -- the proof goes here

end volume_of_pyramid_l489_489222


namespace intervals_of_monotonicity_max_min_values_log_inequality_l489_489083

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x - Real.log x

theorem intervals_of_monotonicity :
  (∀ x > 0, x < 1 → 0 < (1 - x) / (x^2)) ∧ (∀ x > 1, (1 - x) / (x^2) < 0) :=
by sorry

theorem max_min_values :
  (f 1 = 0) ∧ (f (1 / Real.exp 1) = 2 - Real.exp 1) ∧ (f (Real.exp 1) = -1 / Real.exp 1) ∧ (2 - Real.exp 1 < -1 / Real.exp 1) :=
by sorry

theorem log_inequality (x : ℝ) (hx1 : x > 0) (hx2 : x ≠ 1) :
  (Real.log (Real.exp 2 / x) ≤ (1 + x) / x) :=
by 
  let g (x : ℝ) := 1 - Real.log x - 1 / x
  have h_deriv : ∀ x > 0, deriv g x = (1 - x) / (x^2) := sorry,
  show g x ≤ 0 := sorry

end intervals_of_monotonicity_max_min_values_log_inequality_l489_489083


namespace ratio_BD_BP_l489_489117

-- Declare necessary geometric structures
variables (A B C D M N P : Point)
variable u : ℝ
variables (AB AD AM AN BD BP : ℝ)

-- Assume the conditions given
def conditions (u AB AD AM AN : ℝ) : Prop :=
  AB = 7 * u ∧ AD = 9 * u ∧ AM = 1 * u ∧ AN = 1 * u

-- Define the main theorem to prove
theorem ratio_BD_BP (h : conditions u AB AD AM AN) (BD BP : ℝ) : BD / BP = 2 :=
sorry

end ratio_BD_BP_l489_489117


namespace k1k2_constant_l489_489846

noncomputable def ellipse_standard_eq (a b : ℝ) (h1 : a > b) (h2 : e = (Real.sqrt 3) / 3) 
(line_eq : ∀ x y : ℝ, (x - y + 2 = 0) → dist (0, 0) (x, y) = b): Prop :=
  a = Real.sqrt 3 ∧ b = Real.sqrt 2 ∧ ( ∀ x y : ℝ, ( x * x / (a * a) + y * y / (b * b) = 1 )) 

theorem k1k2_constant (P A B : ℝ × ℝ) (hP : P ∈ {p : ℝ × ℝ | (p.1 ^ 2 / 3 + p.2 ^ 2 / 2 = 1)} ) 
(hA : A = (- Real.sqrt 3, 0)) (hB : B = (Real.sqrt 3, 0)) (hP_non_vertex : P ≠ A ∧ P ≠ B) : 
  let k1 := (P.snd / (P.fst + Real.sqrt 3))
  let k2 := (P.snd / (P.fst - Real.sqrt 3))
  k1 * k2 = -2 / 3 := 
sorry

noncomputable def trajectory_eq_M  (k λ : ℝ) (P M : ℝ × ℝ) (hP_pos : λ > 0 ∧ λ ≠ 1 / Real.sqrt 3):
  (M ∈ {m : ℝ × ℝ | m.1 ^ 2 / (6 / (3 * λ ^ 2 - 1)) + m.2 ^ 2 / (6 / (3 * λ ^ 2)) = 1}) ∧ 
  (λ = 1 / Real.sqrt 3 → ∀ x : ℝ, abs x <= 3 → M.2 = Real.sqrt 6) :=
sorry

end k1k2_constant_l489_489846


namespace number_of_valid_subsets_l489_489760

def set_a := {1, 2, 3, 4}
def set_b := {1, 2}

-- Define a predicate for the property B ⊆ C ⊆ A
def isValidSubset (C : Set ℕ) : Prop :=
  set_b ⊆ C ∧ C ⊆ set_a

-- Prove that the number of subsets meeting the conditions is 12
theorem number_of_valid_subsets : (Set.filter isValidSubset (Set.powerset set_a)).card = 12 :=
  sorry

end number_of_valid_subsets_l489_489760


namespace sum_f_neg_l489_489041

def f (x : ℝ) := -x^3 - Real.sin x

theorem sum_f_neg
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 > 0)
  (h2 : x2 + x3 > 0)
  (h3 : x3 + x1 > 0) :
  f x1 + f x2 + f x3 < 0 :=
by
  sorry

end sum_f_neg_l489_489041


namespace negation_of_r_truth_of_propositions_l489_489168

def p : Prop := ∃ x_0 : ℝ, x_0 > -2 ∧ 6 + abs x_0 = 5
def q : Prop := ∀ x : ℝ, x < 0 → x^2 + 4 / x^2 ≥ 4
def r (a : ℝ) : Prop := a ≥ 1 → ∀ x y : ℝ, x < y → ax + cos x ≤ ay + cos y

theorem negation_of_r (a : ℝ) : ¬(r a) ↔ (a < 1 → ∃ x y : ℝ, x < y ∧ ax + cos x > ay + cos y) := 
by sorry

theorem truth_of_propositions (a : ℝ) : 
  (¬p) ∧ (p ∨ r a) ∧ (¬(p ∧ q)) := 
by sorry

end negation_of_r_truth_of_propositions_l489_489168


namespace arccos_neg_one_eq_pi_l489_489748

theorem arccos_neg_one_eq_pi : real.arccos (-1) = real.pi :=
by sorry

end arccos_neg_one_eq_pi_l489_489748


namespace range_of_a_l489_489838

noncomputable def A := {x : ℝ | x^2 - 2*x - 8 < 0}
noncomputable def B := {x : ℝ | x^2 + 2*x - 3 > 0}
noncomputable def C (a : ℝ) := {x : ℝ | x^2 - 3*a*x + 2*a^2 < 0}

theorem range_of_a (a : ℝ) :
  (C a ⊆ A ∩ B) ↔ (1 ≤ a ∧ a ≤ 2 ∨ a = 0) :=
sorry

end range_of_a_l489_489838


namespace max_value_ab_l489_489074

theorem max_value_ab (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_decreasing : ∀ (x m : ℝ), x ≥ 0 → m > 0 → f (x + m) < f x)
  (h_inequality : ∀ (x a b : ℝ), a > 0 → f (2 * Real.exp x - a * x) - f (Real.exp x + b) ≤ 0) :
  ∃ (a b : ℝ), a = Real.sqrt Real.exp 1 ∧ b = a - a * Real.log a ∧ a * b = Real.exp 1 / 2 :=
by sorry

end max_value_ab_l489_489074


namespace BR_perp_CR_l489_489110

theorem BR_perp_CR {A B C I P Q R : Type*}
  [EuclideanGeometry A B C I P Q R] -- Assuming a Euclidean geometry context.
  (h_iso_triangle : is_isosceles_triangle A B C)
  (h_incenter : is_incenter I A B C)
  (h_circ_A : is_circle A A B)
  (h_circ_I : is_circle I I B)
  (h_gamma : is_intersects_at_points Γ B I P Q)
  (h_intersect : is_intersect IP BQ R) :
  is_perpendicular B R C R :=
sorry

end BR_perp_CR_l489_489110


namespace least_multiple_x_correct_l489_489885

noncomputable def least_multiple_x : ℕ :=
  let x := 20
  let y := 8
  let z := 5
  5 * y

theorem least_multiple_x_correct (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 33) (h5 : 5 * y = 8 * z) : least_multiple_x = 40 :=
by
  sorry

end least_multiple_x_correct_l489_489885


namespace car_r_average_speed_l489_489265

-- Define the given conditions as assumptions in Lean
variables (d t v : ℝ)
variable hP_greater : ∀ vR : ℝ, vP = vR + 10
variable hTime_diff : ∀ vR : ℝ, (600 / vP = 600 / vR - 2)

-- Define the problem statement
theorem car_r_average_speed : ∃ vR : ℝ, (hP_greater vR) ∧ (hTime_diff vR) ∧ (vR = 50) :=
begin
  sorry
end

end car_r_average_speed_l489_489265


namespace number_of_possible_values_of_n_l489_489579

theorem number_of_possible_values_of_n:
  ∃ (n : ℤ), (∃ (a : ℤ) (r : ℝ), 
  (P(x) = x^3 - 2003 * x^2 + m * x + n) ∧ 
  (P(a) = 0) ∧ 
  (a = 2 * ((a / 3) + r)) ∧ 
  (r ∈ ℝ \ ℚ) ∧ 
  (x^3 - (a + 2*(a/3))*x^2 + (2*(a^2)/9 + r^2)*x - (a^3/27 - ar^2/3)= 
  x^3 - 2003 * x^2 + m * x + n)
  ) → (160800 : ℕ) :=
sorry

end number_of_possible_values_of_n_l489_489579


namespace cannot_transform_to_all_pluses_l489_489824

def initial_grid : list (list char) := [
  ['+', '-', '+', '+'],
  ['+', '+', '+', '+'],
  ['+', '+', '+', '+'],
  ['+', '-', '+', '+']
]

def toggle_row (grid : list (list char)) (r : ℕ) : list (list char) :=
  grid.map_with_index (λ i row, if i = r then row.map (λ c, if c = '+' then '-' else '+') else row)

def toggle_col (grid : list (list char)) (c : ℕ) : list (list char) :=
  grid.map (λ row, row.map_with_index (λ j cell, if j = c then if cell = '+' then '-' else '+' else cell))

def all_pluses (grid : list (list char)) : Prop :=
  grid.all (λ row, row.all (λ c, c = '+'))

theorem cannot_transform_to_all_pluses : ¬ ∃ (grid : list (list char)) (moves : list (ℕ × bool)), 
  let final_grid := moves.foldl (λ g move, if move.2 then toggle_row g move.1 else toggle_col g move.1) initial_grid
  in all_pluses final_grid :=
sorry

end cannot_transform_to_all_pluses_l489_489824


namespace sqrt_5th_of_x_sqrt_4th_x_l489_489722

theorem sqrt_5th_of_x_sqrt_4th_x (x : ℝ) (hx : 0 < x) : Real.sqrt (x * Real.sqrt (x ^ (1 / 4))) = x ^ (1 / 4) :=
by
  sorry

end sqrt_5th_of_x_sqrt_4th_x_l489_489722


namespace min_k_acute_angled_triangle_l489_489376

theorem min_k_acute_angled_triangle : 
  ∃ k : ℕ, 
    (∀ (s : finset ℕ), s.card = k → 
      s ⊆ finset.range 2004 → 
      ∃ a b c : ℕ, 
        a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ 
        a < b ∧ b < c ∧ 
        (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)) 
  ∧ k = 29 :=
begin
  sorry
end

end min_k_acute_angled_triangle_l489_489376


namespace find_digits_l489_489067

noncomputable def digit_a : ℕ :=
  2

noncomputable def digit_b : ℕ :=
  0

noncomputable def digit_c : ℕ :=
  0

noncomputable def digit_d : ℕ :=
  3

theorem find_digits (a b c d : ℕ) 
    (h1 : 34! = Nat.factorial 34)
    (h2 : 35 * digit_a % 8 = 0) 
    (h3 : (digit_c + digit_d) % 9 = 3) 
    (h4 : (61 + digit_c) % 11 = (80 + digit_d) % 11) 
    (h5 : digit_a ≠ 0)
    : a = digit_a ∧ b = digit_b ∧ c = digit_c ∧ d = digit_d := by
  sorry

end find_digits_l489_489067


namespace three_digit_numbers_sum_seven_l489_489008

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489008


namespace prob_B_hits_once_prob_hits_with_ABC_l489_489230

section
variable (P_A P_B P_C : ℝ)
variable (hA : P_A = 1 / 2)
variable (hB : P_B = 1 / 3)
variable (hC : P_C = 1 / 4)

-- Part (Ⅰ): Probability of hitting the target exactly once when B shoots twice
theorem prob_B_hits_once : 
  (P_B * (1 - P_B) + (1 - P_B) * P_B) = 4 / 9 := 
by
  rw [hB]
  sorry

-- Part (Ⅱ): Probability of hitting the target when A, B, and C each shoot once
theorem prob_hits_with_ABC :
  (1 - ((1 - P_A) * (1 - P_B) * (1 - P_C))) = 3 / 4 := 
by
  rw [hA, hB, hC]
  sorry

end

end prob_B_hits_once_prob_hits_with_ABC_l489_489230


namespace inverse_of_f_l489_489786

-- defining the function f
def f (x : ℝ) : ℝ := 3 - 4 * x

-- defining the candidate inverse function g
def g (x : ℝ) : ℝ := (3 - x) / 4

-- stating the theorem that g is indeed the inverse of f
theorem inverse_of_f : ∀ x : ℝ, f (g x) = x := 
by
  intro x
  sorry

end inverse_of_f_l489_489786


namespace polygon_sides_l489_489588

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l489_489588


namespace largest_possible_distance_l489_489951

noncomputable def largest_distance (z : ℂ) (hz : complex.abs z = 3) : ℝ :=
  complex.abs ((4 + 3 * complex.I) * z^3 + 2 * z - (z^4 + 1))

theorem largest_possible_distance (z : ℂ) (hz : complex.abs z = 3) :
  largest_distance z hz = real.sqrt 7290 :=
sorry

end largest_possible_distance_l489_489951


namespace choose_sandwich_and_gimbap_l489_489286

theorem choose_sandwich_and_gimbap (sandwiches gimbap : ℕ) (h1 : sandwiches = 4) (h2 : gimbap = 3) : sandwiches * gimbap = 12 := by
  rw [h1, h2]
  norm_num

end choose_sandwich_and_gimbap_l489_489286


namespace exists_set_A_with_property_l489_489187

theorem exists_set_A_with_property :
  ∃ A : Set ℕ, (∀ S : Set ℕ, infinite S → (∀ (p : ℕ), p ∈ S → Nat.Prime p) →
  ∃ (k : ℕ) (m n : ℕ), k ≥ 2 ∧ m ∈ A ∧ n ∉ A ∧
  ∃ (S' : Finset ℕ), S' ⊆ S ∧ S'.card = k ∧ m = S'.prod id ∧
    ∃ (S'' : Finset ℕ), S'' ⊆ S ∧ S''.card = k ∧ n = S''.prod id) := sorry

end exists_set_A_with_property_l489_489187


namespace smallest_n_for_nonzero_constant_term_l489_489443

theorem smallest_n_for_nonzero_constant_term : 
  ∃ n : ℕ, (∃ r : ℕ, n = 5 * r / 3) ∧ (n > 0) ∧ ∀ m : ℕ, (m > 0) → (∃ s : ℕ, m = 5 * s / 3) → n ≤ m :=
by sorry

end smallest_n_for_nonzero_constant_term_l489_489443


namespace cost_per_pound_of_mixed_candy_l489_489683

def w1 := 10
def p1 := 8
def w2 := 20
def p2 := 5

theorem cost_per_pound_of_mixed_candy : 
    (w1 * p1 + w2 * p2) / (w1 + w2) = 6 := by
  sorry

end cost_per_pound_of_mixed_candy_l489_489683


namespace extreme_points_inequality_l489_489853

noncomputable def g (x : ℝ) (a : ℝ) := log x + 2 * x + a / x
noncomputable def f (x : ℝ) (a : ℝ) := x * g(x, a) - (a / 2 + 2) * x^2 - x

theorem extreme_points_inequality 
    {a : ℝ} {m : ℝ} (h_cond : 0 < a ∧ a < 1 / Real.exp 1) (m_ge_1 : m ≥ 1)
    (h_extreme_points : ∃ (x1 x2 : ℝ), f'(x1, a) = 0 ∧ f'(x2, a) = 0 ∧ x1 < x2) :
    ∃ (x1 x2 : ℝ), x1 * x2^m > Real.exp (1 + m) :=
sorry

end extreme_points_inequality_l489_489853


namespace unique_solution_f_l489_489957

theorem unique_solution_f (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x + f y) ≥ f (f x + y))
  (h2 : f 0 = 0) :
  ∀ x : ℝ, f x = x :=
sorry

end unique_solution_f_l489_489957


namespace find_total_quantities_l489_489203

theorem find_total_quantities (n S S_3 S_2 : ℕ) (h1 : S = 8 * n) (h2 : S_3 = 4 * 3) (h3 : S_2 = 14 * 2) (h4 : S = S_3 + S_2) : n = 5 :=
by
  sorry

end find_total_quantities_l489_489203


namespace yanna_change_l489_489649

theorem yanna_change :
  let shirt_cost := 5
  let sandal_cost := 3
  let num_shirts := 10
  let num_sandals := 3
  let given_amount := 100
  (given_amount - (num_shirts * shirt_cost + num_sandals * sandal_cost)) = 41 :=
by
  sorry

end yanna_change_l489_489649


namespace number_of_integer_solutions_l489_489808

theorem number_of_integer_solutions (x : ℤ) : 
  (x^2 < 10 * x) → {x | (x^2 < 10 * x)}.finite
    ∧ {x | (x^2 < 10 * x)}.to_finset.card = 9 :=
by
  sorry

end number_of_integer_solutions_l489_489808


namespace noodles_initial_l489_489761

-- Definitions of our conditions
def given_away : ℝ := 12.0
def noodles_left : ℝ := 42.0
def initial_noodles : ℝ := 54.0

-- Theorem statement
theorem noodles_initial (a b : ℝ) (x : ℝ) (h₁ : a = 12.0) (h₂ : b = 42.0) (h₃ : x = a + b) : x = initial_noodles :=
by
  -- Placeholder for the proof
  sorry

end noodles_initial_l489_489761


namespace evaluate_at_x_l489_489772

noncomputable def evaluate_expression (x : ℝ) := ((x^2 / (x - 3)) - (2 * x / (x - 3))) / (x / (x - 3))

noncomputable def simplify_expression (x : ℝ) := x - 2

-- Given condition
def x_value : ℝ := Real.sqrt 7 + 1

-- The statement we want to prove
theorem evaluate_at_x : (Float.ofReal (evaluate_expression x_value)).round (some 2) = 1.65 :=
by
  sorry

end evaluate_at_x_l489_489772


namespace range_positive_f_l489_489391

-- Define the conditions of the problem
variables {f : ℝ → ℝ}
variable H_even : ∀ x, f (-x) = f x
variable H_diff : ∀ x, HasDerivAt f (f' x) x
variable H_f1 : f 1 = 0
variable H_ineq : ∀ x, 0 < x → x * (f' x) < 2 * (f x)

-- The range of x for which f(x) > 0
theorem range_positive_f : ∀ x, (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) → f x > 0 := by
  sorry

end range_positive_f_l489_489391


namespace n_times_s_eq_neg2_l489_489956

-- Conditions on the function f
variables (f : ℝ → ℝ)
variable (h1 : f 2 = 2)
variable (h2 : ∀ x y : ℝ, f (x * y + f x) = x * f y + f x + x ^ 2)

-- The main theorem to prove the mathematically equivalent statement
theorem n_times_s_eq_neg2 : 
  let n := {z : ℝ | ∃ y : ℝ, f y = z}.to_set.count f 0 in
  let s := {z : ℝ | ∃ y : ℝ, f y = z}.to_set.sum in
  n * s = -2 := sorry

end n_times_s_eq_neg2_l489_489956


namespace f_odd_period_pi_l489_489211

def f (x : ℝ) : ℝ := sin (x + π / 4) ^ 2 - sin (x - π / 4) ^ 2

theorem f_odd_period_pi : (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, f (x + π) = f x) :=
by
  sorry

end f_odd_period_pi_l489_489211


namespace right_triangle_legs_l489_489047

theorem right_triangle_legs (a b : ℕ) (h : a^2 + b^2 = 100) (h_r: a + b - 10 = 4) : (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) :=
sorry

end right_triangle_legs_l489_489047


namespace rectangle_dimension_l489_489407

theorem rectangle_dimension (x : ℝ) (h : (x^2) * (x + 5) = 3 * (2 * (x^2) + 2 * (x + 5))) : x = 3 :=
by
  have eq1 : (x^2) * (x + 5) = x^3 + 5 * x^2 := by ring
  have eq2 : 3 * (2 * (x^2) + 2 * (x + 5)) = 6 * x^2 + 6 * x + 30 := by ring
  rw [eq1, eq2] at h
  sorry  -- Proof details omitted

end rectangle_dimension_l489_489407


namespace probability_one_person_hits_probability_plane_is_hit_l489_489673
noncomputable def P_A := 0.7
noncomputable def P_B := 0.6

theorem probability_one_person_hits : P_A * (1 - P_B) + (1 - P_A) * P_B = 0.46 :=
by
  sorry

theorem probability_plane_is_hit : 1 - (1 - P_A) * (1 - P_B) = 0.88 :=
by
  sorry

end probability_one_person_hits_probability_plane_is_hit_l489_489673


namespace boat_speed_in_still_water_l489_489893

theorem boat_speed_in_still_water (b s : ℝ) 
  (h1 : b + s = 11) 
  (h2 : b - s = 3) : b = 7 :=
by
  sorry

end boat_speed_in_still_water_l489_489893


namespace max_kings_on_chessboard_l489_489788

theorem max_kings_on_chessboard (n : ℕ) (board : matrix (fin 12) (fin 12) bool) : 
  (∀ i j, board i j → 
      (∃ i' j', (i', j') ≠ (i, j) ∧ board i' j' ∧ 
       (abs (i - i') ≤ 1 ∧ abs (j - j') ≤ 1))) → 
  ∃ K, (∀ i j, board i j → ∃ i' j', board i' j' ∧ 
       (abs (i - i') ≤ 1 ∧ abs (j - j') ≤ 1) ∧ 
       (i', j') ≠ (i, j)) ∧ 
     K ≤ 56 :=
begin
  sorry
end

end max_kings_on_chessboard_l489_489788


namespace proof_problem_l489_489406

open Real

noncomputable def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  { p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1 }

noncomputable def foci (a b : ℝ) (h : a > b ∧ b > 0) : ℝ := 
  sqrt (a^2 - b^2)

noncomputable def triangle_perimeter (P F1 F2 : (ℝ × ℝ)) : ℝ :=
  dist P F1 + dist P F2 + dist F1 F2

noncomputable def minimum_value (a c : ℝ) : Prop :=
  (4 / a + 1 / c) = 3

noncomputable def range_QA_QB (m : ℝ) : Set ℝ :=
  Ioo (45 / 4) 12

theorem proof_problem 
  (a b c : ℝ)
  (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : (4 / a + 1 / c) = 3)
  (h₄ : ∃ P : ℝ × ℝ, (P ∈ ellipse a b ⟨h₁, h₂⟩) ∧ triangle_perimeter P (-foci a b ⟨h₁, h₂⟩, 0) (foci a b ⟨h₁, h₂⟩, 0) = 6)
  (h₅ : ∃ A B : ℝ × ℝ, A ≠ B ∧ A ∈ ellipse a b ⟨h₁, h₂⟩ ∧ B ∈ ellipse a b ⟨h₁, h₂⟩ 
    ∧ A.1 = B.1 ∧ A.1 = -4) :
  ∃ m : ℝ, m ∈ range_QA_QB m :=
sorry

end proof_problem_l489_489406


namespace num_three_digit_sums7_l489_489006

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l489_489006


namespace area_of_circle_outside_triangle_l489_489944

-- Define the conditions and problem
theorem area_of_circle_outside_triangle
  (A B C X Y : Point)
  (r : ℝ)
  (h_triangle : right_triangle A B C)
  (h_angle : angle A B C = 90)
  (h_tangent_AB : is_tangent_circle AB X r)
  (h_tangent_AC : is_tangent_circle AC Y r)
  (h_diametric_X_BC : diametric_opposite X BC)
  (h_diametric_Y_BC : diametric_opposite Y BC)
  (AB_len : dist A B = 8)
  : area_of_circle_outside_triangle = (16 * π - 32) / 9 := sorry

end area_of_circle_outside_triangle_l489_489944


namespace nine_point_two_minus_star_l489_489658

def greatest_even_le (y : ℝ) : ℝ :=
  if (0 : ℝ) ≤ y then
    let n := floor (y / 2) in 2 * n
  else 0

theorem nine_point_two_minus_star :
  9.2 - greatest_even_le 9.2 = 1.2 :=
by
  -- We skip the proof here
  sorry

end nine_point_two_minus_star_l489_489658


namespace percentage_increase_mario_salary_is_zero_l489_489171

variable (M : ℝ) -- Mario's salary last year
variable (P : ℝ) -- Percentage increase in Mario's salary

-- Condition 1: Mario's salary increased to $4000 this year
def mario_salary_increase (M P : ℝ) : Prop :=
  M * (1 + P / 100) = 4000 

-- Condition 2: Bob's salary last year was 3 times Mario's salary this year
def bob_salary_last_year (M : ℝ) : Prop :=
  3 * 4000 = 12000 

-- Condition 3: Bob's current salary is 20% more than his salary last year
def bob_current_salary : Prop :=
  12000 * 1.2 = 14400

-- Theorem : The percentage increase in Mario's salary is 0%
theorem percentage_increase_mario_salary_is_zero
  (h1 : mario_salary_increase M P)
  (h2 : bob_salary_last_year M)
  (h3 : bob_current_salary) : 
  P = 0 := 
sorry

end percentage_increase_mario_salary_is_zero_l489_489171


namespace qualified_flour_l489_489706

-- Define the acceptable weight range
def acceptable_range (w : ℝ) : Prop :=
  24.75 ≤ w ∧ w ≤ 25.25

-- Define the weight options
def optionA : ℝ := 24.70
def optionB : ℝ := 24.80
def optionC : ℝ := 25.30
def optionD : ℝ := 25.51

-- The statement to be proved
theorem qualified_flour : acceptable_range optionB ∧ ¬acceptable_range optionA ∧ ¬acceptable_range optionC ∧ ¬acceptable_range optionD :=
by
  sorry

end qualified_flour_l489_489706


namespace sides_of_polygon_l489_489613

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l489_489613


namespace percent_increase_output_per_hour_l489_489914

-- Definitions and conditions
variable (B H : ℝ) -- B is the number of bears per week, H is the number of hours per week

-- Define outputs per hour
def output_per_hour (B H : ℝ) := B / H
def new_output_per_hour (B H : ℝ) := 1.8 * B / (0.9 * H)

-- A mathematical statement to prove the percentage increase of output per hour
theorem percent_increase_output_per_hour (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((new_output_per_hour B H) - (output_per_hour B H)) / (output_per_hour B H) * 100 = 100 :=
by
  sorry

end percent_increase_output_per_hour_l489_489914


namespace unique_root_of_ln_eq_4tsq_minus_a_l489_489813

noncomputable def unique_root_value (a : ℝ) : Prop :=
∃ x : ℝ, (∀ y : ℝ, ln (y + a) - 4 * (y + a)^2 + a = 0 → y = x)

theorem unique_root_of_ln_eq_4tsq_minus_a :
  (unique_root_value ((3 * Real.log 2 + 1) / 2)) :=
begin
  sorry
end

end unique_root_of_ln_eq_4tsq_minus_a_l489_489813


namespace diagonals_in_polygon_with_150_deg_angles_l489_489287

def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_polygon_with_150_deg_angles :
  (∀ (n : ℕ), (n - 2) * 180 = 150 * n → num_diagonals n = 54) := 
by
  intro n
  have h : (n - 2) * 180 = 150 * n
  have h1 : 180 * n - 360 = 150 * n
  have h2 : 30 * n = 360
  have n_eq_12 : n = 12
  rw [n_eq_12]
  unfold num_diagonals
  norm_num
  exact eq.refl 54
  sorry

end diagonals_in_polygon_with_150_deg_angles_l489_489287


namespace odd_function_condition_l489_489414

def f (x : ℝ) (a : ℝ) := (2^x - a) / (2^x + a)

theorem odd_function_condition (a : ℝ) : (∀ x : ℝ, f (-x) a = - f x a) ↔ (a = 1 ∨ a = -1) :=
by
  sorry

end odd_function_condition_l489_489414


namespace maximize_profit_l489_489695

noncomputable def profit (x : ℝ) : ℝ :=
  let selling_price := 10 + 0.5 * x
  let sales_volume := 200 - 10 * x
  (selling_price - 8) * sales_volume

theorem maximize_profit : ∃ x : ℝ, x = 8 → profit x = profit 8 ∧ (∀ y : ℝ, profit y ≤ profit 8) := 
  sorry

end maximize_profit_l489_489695


namespace region_area_is_correct_l489_489329

open Real

noncomputable def region_area : ℝ :=
  let A := Set.Icc (2 : ℝ) ((13 : ℝ) / 3)
  let B := Set.Ici 3
  Set.integral (λ x, abs (x - 2)) (λ x, 5 - 2 * abs (x - 3)) A + 
  Set.integral (λ x, abs (x - 2)) (λ x, 5 - 2 * abs (x - 3)) B

theorem region_area_is_correct : region_area = 35 / 9 := 
by sorry

end region_area_is_correct_l489_489329


namespace problem_inequality_I_problem_inequality_II_l489_489030

theorem problem_inequality_I (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1):
  1 / a + 1 / b ≥ 4 := sorry

theorem problem_inequality_II (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1):
  (a + 1 / a)^2 + (b + 1 / b)^2 ≥ 25 / 2 := sorry

end problem_inequality_I_problem_inequality_II_l489_489030


namespace median_squared_formula_l489_489347

theorem median_squared_formula (a b c m : ℝ) (AC_is_median : 2 * m^2 + c^2 = a^2 + b^2) : 
  m^2 = (1/4) * (2 * a^2 + 2 * b^2 - c^2) := 
by
  sorry

end median_squared_formula_l489_489347


namespace number_of_valid_pairs_l489_489340

theorem number_of_valid_pairs :
  (∀ (m n : ℕ), 1 ≤ m ∧ m ≤ 2044 ∧ 5^n < 2^m ∧ 2^m < 2^(m + 1) ∧ 2^(m + 1) < 5^(n + 1)) ↔
  ((∃ (x y : ℕ), 2^2100 < 5^900 ∧ 5^900 < 2^2101)) → 
  (∃ (count : ℕ), count = 900) :=
by sorry

end number_of_valid_pairs_l489_489340


namespace jane_output_increase_l489_489924

theorem jane_output_increase (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  let original_output := B / H
  let new_output := (1.8 * B) / (0.9 * H)
  let percent_increase := ((new_output - original_output) / original_output) * 100
  percent_increase = 100 := by
  sorry

end jane_output_increase_l489_489924


namespace sphere_surface_area_correct_l489_489626

-- Define the condition and the known formulas.
def sphere_volume (r : ℝ) : ℝ := (4/3) * Real.pi * r^3
def sphere_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Specify the given volume.
def given_volume : ℝ := 72 * Real.pi

-- Declare the theorem to prove the surface area given the condition.
theorem sphere_surface_area_correct (r : ℝ) (h : sphere_volume r = given_volume) : 
    sphere_surface_area r = 36 * Real.pow 2 (2 / 3) * Real.pi := 
by
  sorry


end sphere_surface_area_correct_l489_489626


namespace polygon_sides_l489_489617

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489617


namespace polygon_sides_l489_489590

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l489_489590


namespace arccos_neg_one_eq_pi_proof_l489_489746

noncomputable def arccos_neg_one_eq_pi : Prop :=
  arccos (-1) = π

theorem arccos_neg_one_eq_pi_proof : arccos_neg_one_eq_pi := by
  sorry

end arccos_neg_one_eq_pi_proof_l489_489746


namespace total_income_generated_is_correct_l489_489335

noncomputable def cupcake_original_price : ℝ := 3.00
noncomputable def cupcake_discount : ℝ := 0.30
noncomputable def cupcake_quantity : ℕ := 25
noncomputable def cookie_original_price : ℝ := 2.00
noncomputable def cookie_discount : ℝ := 0.45
noncomputable def cookie_quantity : ℕ := 18
noncomputable def brownie_original_price : ℝ := 4.00
noncomputable def brownie_discount : ℝ := 0.25
noncomputable def brownie_quantity : ℕ := 15
noncomputable def macaron_original_price : ℝ := 1.50
noncomputable def macaron_discount : ℝ := 0.50
noncomputable def macaron_quantity : ℕ := 30

theorem total_income_generated_is_correct :
  let cupcake_reduced_price := cupcake_original_price * (1 - cupcake_discount)
      cupcake_income := cupcake_reduced_price * cupcake_quantity
      cookie_reduced_price := cookie_original_price * (1 - cookie_discount)
      cookie_income := cookie_reduced_price * cookie_quantity
      brownie_reduced_price := brownie_original_price * (1 - brownie_discount)
      brownie_income := brownie_reduced_price * brownie_quantity
      macaron_reduced_price := macaron_original_price * (1 - macaron_discount)
      macaron_income := macaron_reduced_price * macaron_quantity
      total_income := cupcake_income + cookie_income + brownie_income + macaron_income
  in total_income = 139.80 := by
begin
  let cupcake_reduced_price := cupcake_original_price * (1 - cupcake_discount),
  let cupcake_income := cupcake_reduced_price * cupcake_quantity,
  let cookie_reduced_price := cookie_original_price * (1 - cookie_discount),
  let cookie_income := cookie_reduced_price * cookie_quantity,
  let brownie_reduced_price := brownie_original_price * (1 - brownie_discount),
  let brownie_income := brownie_reduced_price * brownie_quantity,
  let macaron_reduced_price := macaron_original_price * (1 - macaron_discount),
  let macaron_income := macaron_reduced_price * macaron_quantity,
  let total_income := cupcake_income + cookie_income + brownie_income + macaron_income,
  rw [cupcake_reduced_price, cookie_reduced_price, brownie_reduced_price, macaron_reduced_price] at total_income,
  sorry
end

end total_income_generated_is_correct_l489_489335


namespace zero_is_integer_not_positive_l489_489541

theorem zero_is_integer_not_positive :
  (Int.zero ∈ Int ∧ ¬ (Int.zero > 0)) :=
by
  sorry

end zero_is_integer_not_positive_l489_489541


namespace count_digits_product_l489_489864

theorem count_digits_product : 
  let a := (3 : ℕ) ^ 7
  let b := (6 : ℕ) ^ 14
  let c := a * b
  let log2 := 0.301
  let log3 := 0.477
  let digits := Int.floor (14 * log2 + 21 * log3) + 1
  in digits = 15 :=
by 
  sorry

end count_digits_product_l489_489864


namespace polygon_sides_l489_489607

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489607


namespace arcsin_arccos_solution_l489_489550

theorem arcsin_arccos_solution (x : ℝ) (hx1 : |x| ≤ 1) (hx2 : |2*x| ≤ 1) :
  arcsin x + arcsin (2*x) = arccos x ↔ x = 0 ∨ x = 2 / Real.sqrt 5 ∨ x = - (2 / Real.sqrt 5) := 
sorry

end arcsin_arccos_solution_l489_489550


namespace parabola_intersects_x_axis_at_two_distinct_points_point_between_x1_and_x2_find_x1_and_x2_when_M_is_given_l489_489042

variables {p q x x0 y0 x1 x2 : ℝ}

-- Given a parabolic equation of the form y = x^2 + px + q and a point (x0, y0) located below the x-axis
def parabola (x : ℝ) : ℝ := x^2 + p * x + q

-- Assume the point M(x0, y0) is below the x-axis
def point_M_below_x_axis (x0 y0 : ℝ) (h : y0 < 0) : Prop :=
  parabola x0 = y0

-- Statements to be proven
theorem parabola_intersects_x_axis_at_two_distinct_points {p q x0 y0 x1 x2 : ℝ} (h_y0 : y0 < 0) 
  (hM : point_M_below_x_axis x0 y0 h_y0) : x1 < x2 ∧ parabola x1 = 0 ∧ parabola x2 = 0 :=
sorry

theorem point_between_x1_and_x2 {p q x0 y0 x1 x2 : ℝ} (h_y0 : y0 < 0) 
  (hM : point_M_below_x_axis x0 y0 h_y0) (h_intersects : x1 < x2 ∧ parabola x1 = 0 ∧ parabola x2 = 0) : 
  x1 < x0 ∧ x0 < x2 :=
sorry

theorem find_x1_and_x2_when_M_is_given (x0 : ℝ) (y0 : ℝ) (h : x0 = 1 ∧ y0 = -1999) : 
  ∃ x1 x2 : ℤ, (x1 * x2 = ⌊q⌋ ∧ x1 + x2 = -⌊p⌋ ∧ ((x1 - 1) * (x2 - 1) = -1999) ∧ (x1 = -1998 ∧ x2 = 2 ∨ x1 = 0 ∧ x2 = 2000)) :=
sorry

end parabola_intersects_x_axis_at_two_distinct_points_point_between_x1_and_x2_find_x1_and_x2_when_M_is_given_l489_489042


namespace find_original_number_l489_489977

variable {I K S : ℕ}

theorem find_original_number
  (distinct_digits : I ≠ K ∧ K ≠ S ∧ S ≠ I)
  (non_zero_digits : I ≠ 0 ∧ K ≠ 0 ∧ S ≠ 0)
  (product_ends_with_S : (I * 100 + K * 10 + S) * (K * 100 + S * 10 + I) % 10 = S)
  (is_six_digit_number : 100000 ≤ (I * 100 + K * 10 + S) * (K * 100 + S * 10 + I) ∧ (I * 100 + K * 10 + S) * (K * 100 + S * 10 + I) < 1000000)
  (erased_zeros : erase_zeros ((I * 100 + K * 10 + S) * (K * 100 + S * 10 + I)) = I * 100 + K * 10 + S) :
  (I = 1) ∧ (K = 6) ∧ (S = 2) := sorry

end find_original_number_l489_489977


namespace sum_original_and_correct_value_l489_489655

theorem sum_original_and_correct_value (x : ℕ) (h : x + 14 = 68) :
  x + (x + 41) = 149 := by
  sorry

end sum_original_and_correct_value_l489_489655


namespace find_angle_GYH_l489_489484

theorem find_angle_GYH 
  (AB CD : Line) (EF GH : Line) (X Y : Point) 
  (A B C D : Point) (α β γ δ : ℝ) :
  parallel AB CD →
  parallel EF GH →
  angle AXF = 132 →
  angle FYG = 110 →
  α = 180 - 132 →
  β = α →
  γ = angle GYH →
  δ = 48 →
  γ = δ := 
by {
  intros h1 h2 h3 h4 h5 h6 h7 h8,
  sorry
}

end find_angle_GYH_l489_489484


namespace vector_sum_magnitude_l489_489094

open Real

variables (a b : ℝ × ℝ)
variables (angle_a_b : ℝ)
variables (magnitude_b : ℝ)

-- The angle between a and b is 60 degrees
def condition_angle : angle_a_b = π / 3 := by sorry

-- The vector a is (2, 0)
def condition_a : a = (2, 0) := by sorry

-- The magnitude of vector b is 1
def condition_b_magnitude : |b| = 1 := by sorry

-- Define the magnitude function for a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Prove that |a + 2b| = 2√3
theorem vector_sum_magnitude :
  |(a.1 + 2 * b.1, a.2 + 2 * b.2)| = 2 * sqrt 3 :=
by {
  rw [condition_a, condition_b_magnitude, condition_angle],
  sorry
}

end vector_sum_magnitude_l489_489094


namespace worst_is_father_l489_489111

-- Definitions for players
inductive Player
| father
| sister
| daughter
| son
deriving DecidableEq

open Player

def opposite_sex (p1 p2 : Player) : Bool :=
match p1, p2 with
| father, sister => true
| father, daughter => true
| sister, father => true
| daughter, father => true
| son, sister => true
| son, daughter => true
| daughter, son => true
| sister, son => true
| _, _ => false 

-- Problem conditions
variables (worst best : Player)
variable (twins : Player → Player)
variable (worst_best_twins : twins worst = best)
variable (worst_twin_conditions : opposite_sex (twins worst) best)

-- Goal: Prove that the worst player is the father
theorem worst_is_father : worst = Player.father := by
  sorry

end worst_is_father_l489_489111


namespace range_of_a_l489_489415

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → (a - 1 / 2) * x ^ 2 + log x < 2 * a * x) ↔ a ∈ set.Icc (-1 / 2) (1 / 2) :=
by
  sorry

end range_of_a_l489_489415


namespace contractor_absent_days_l489_489285

theorem contractor_absent_days :
  ∃ (x y : ℝ), x + y = 30 ∧ 25 * x - 7.5 * y = 490 ∧ y = 8 :=
by {
  sorry
}

end contractor_absent_days_l489_489285


namespace hyperbola_conditions_l489_489825

-- Define the conditions as a Lean 4 statement
theorem hyperbola_conditions 
    {a b c : ℝ} (eccentricity : a * 2 = c)
    (foci_position : F = (c, 0))
    (distance_to_asymptote : dist (c, 0) asymptote = sqrt(3))
    (right_vertex : A1 = (-a, 0))
    (left_vertex : A2 = (a, 0))
    (point_on_hyperbola : ∃ x0 y0, P = (x0, y0) ∧ (x0, y0) ∉ {(-a, 0), (a, 0)})
    (intersections : ∃ M N, M = (0, y0/(x0 + 1)) ∧ N = (0, -y0/(x0 - 1)))
    : 
    -- 1. Prove the equation of Hyperbola C
    (eq_hyperbola : C = { p : ℝ × ℝ // p ∈ { (x, y) | x^2 - y^2 / 3 = 1 } }) 
    -- 2. Prove the circle D with diameter MN passes through the two foci
    (circle_D_contains_foci : ∀ foci, foci ∈ { (sqrt(3), 0), (-sqrt(3), 0) } → foci ∈ circle_D)
 := 
sorry  -- Implementation of the proof not required.

end hyperbola_conditions_l489_489825


namespace angle_of_inclination_l489_489341

theorem angle_of_inclination (θ : ℝ) : 
  √3 * x + y - 3 = 0 → θ = 2 * ℝ.re / 3 → tan θ = -√3 -> θ = 2 * Real.pi / 3 :=
by
  sorry

end angle_of_inclination_l489_489341


namespace percentage_rotten_oranges_l489_489296

variable (total_oranges total_bananas : ℕ)
variable (perc_rotten_bananas perc_good_fruits : ℚ)

theorem percentage_rotten_oranges :
  total_oranges = 600 →
  total_bananas = 400 →
  perc_rotten_bananas = 0.06 →
  perc_good_fruits = 0.886 →
  (∃ perc_rotten_oranges : ℚ, perc_rotten_oranges = 0.15) := 
by
  intro h_oranges h_bananas h_rotten_bananas h_good_fruits
  use 0.15
  sorry

end percentage_rotten_oranges_l489_489296


namespace problem_proof_l489_489404

variable { n : ℕ }
variable ( C : ℕ → ℕ → ℕ ) -- binomial coefficient function
variable ( a : ℕ → ℤ ) -- coefficients in binomial expansion

axiom nat_star (n : ℕ) : n > 0
axiom binomial_eq_sum (n : ℕ) : (∑ i in range (n + 1), i * C n i) = 192

def expansion_sum (n : ℕ) (a : ℕ → ℤ) : Prop :=
  (3 - 2 * (1 : ℕ))^n = ∑ i in range (n + 1), a i * (1 : ℕ)^i

def sum_binom_coeff : ℕ := 2^6
def val_a0_a2_a4_a6 : ℕ := 7813
def abs_sum_coeffs : ℕ := 15625

theorem problem_proof (h : n = 6) (htot : C n 1 + 2 * C n 2 + 3 * C n 3 + ... + n * C n n = 192)
  : sum_binom_coeff = 2^n ∧ ∑ i in [0,2,4,6], a i = 7813 ∧ ∑ j in range (n + 1), |a j| = 15625 :=
by
  sorry

end problem_proof_l489_489404


namespace miles_reads_100_pages_l489_489173

def reading_speed (genre : String) (focus : String) : Float :=
  match genre, focus with
  | "novels", "low"     => 21
  | "novels", "medium"  => 25
  | "novels", "high"    => 30
  | "graphic novels", "low"    => 30
  | "graphic novels", "medium" => 36
  | "graphic novels", "high"   => 42
  | "comic books", "low"    => 45
  | "comic books", "medium" => 54
  | "comic books", "high"   => 60
  | "non-fiction", "low"    => 18
  | "non-fiction", "medium" => 22
  | "non-fiction", "high"   => 28
  | "biographies", "low"   => 20
  | "biographies", "medium" => 24
  | "biographies", "high"   => 29
  | _, _ => 0

def reading_segments : List (String × String × Float) :=
  [ ("novels", "high", 20.0 / 60.0)
  , ("graphic novels", "low", 10.0 / 60.0)
  , ("non-fiction", "medium", 15.0 / 60.0)
  , ("biographies", "low", 15.0 / 60.0)
  , ("comic books", "medium", 25.0 / 60.0)
  , ("graphic novels", "high", 15.0 / 60.0)
  , ("novels", "low", 20.0 / 60.0)
  , ("non-fiction", "high", 10.0 / 60.0)
  , ("biographies", "medium", 20.0 / 60.0)
  , ("comic books", "low", 30.0 / 60.0) ]

def total_pages_read : Float :=
  reading_segments.foldr (fun (segment : (String × String × Float)) acc =>
    let (genre, focus, time_fraction) := segment
    acc + reading_speed genre focus * time_fraction
  ) 0

theorem miles_reads_100_pages :
  ⌊total_pages_read⌋ = 100 :=
sorry

end miles_reads_100_pages_l489_489173


namespace even_function_implies_a_eq_zero_f_cannot_be_odd_for_any_a_l489_489152

noncomputable def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

theorem even_function_implies_a_eq_zero {a : ℝ} (h : ∀ x : ℝ, f a x = f a (-x)) : a = 0 := 
begin
  sorry
end

theorem f_cannot_be_odd_for_any_a (a : ℝ) : ¬(∀ x : ℝ, f a (-x) = -f a x) :=
begin
  sorry
end

end even_function_implies_a_eq_zero_f_cannot_be_odd_for_any_a_l489_489152


namespace arithmetic_seq_a5_zero_l489_489844

axiom arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n + m * d

theorem arithmetic_seq_a5_zero (a : ℕ → ℤ) (d : ℤ) (h : arithmetic_seq a d) (h1 : d ≠ 0) 
  (h2 : a 3 + a 9 = a 10 - a 8) : a 5 = 0 :=
by
  sorry

end arithmetic_seq_a5_zero_l489_489844


namespace radical_center_on_OI_l489_489386

-- Definitions based on given conditions

variable {ABC : Triangle} (A1 A2 B1 B2 C1 C2 : Point)
  (O : Point) -- Circumcenter of triangle ABC
  (I : Point) -- Incenter of triangle ABC
  [ScaleneTriangle ABC] -- ABC is a scalene triangle
  [AngleBisector AA1 ABC] -- AA1 is the angle bisector of angle A in triangle ABC
  [IncircleTouching A2 BC ABC] -- A2 is the point where the incircle touches side BC

/-- Radical center of the circumcircles of triangles AA1A2, BB1B2, CC1C2 lies on the line OI -/
theorem radical_center_on_OI
  (radical_center : Point) : 
  (Circumcircle AA1A2).radical_center (Circumcircle BB1B2) (Circumcircle CC1C2) = some radical_center ->
  LiesOnLine radical_center (LineThru O I) := 
sorry

end radical_center_on_OI_l489_489386


namespace b_plus_1_is_prime_l489_489954

variable (a b : ℕ)

def is_good_for_a (b a : ℕ) : Prop :=
  ∀ n : ℕ, (an ≥ b) → (binom an b - 1) % (an + 1) = 0

theorem b_plus_1_is_prime 
  (h1 : is_good_for_a b a)
  (h2 : ¬ is_good_for_a (b + 2) a) :
  Prime (b + 1) :=
sorry

end b_plus_1_is_prime_l489_489954


namespace area_spot_can_reach_l489_489197

noncomputable def area_reachable_by_spot (s : ℝ) (r : ℝ) : ℝ := 
  if s = 1 ∧ r = 3 then 6.5 * Real.pi else 0

theorem area_spot_can_reach : area_reachable_by_spot 1 3 = 6.5 * Real.pi :=
by
  -- The theorem proof should go here.
  sorry

end area_spot_can_reach_l489_489197


namespace sequence_bound_l489_489487

noncomputable def x : ℕ → ℤ
| 0       := 1
| 1       := 1
| 2       := 3
| (n + 3) := 4 * x (n + 2) - 2 * x (n + 1) - 3 * x n

theorem sequence_bound (n : ℕ) (hn : n ≥ 3) : 
  x n > (3 / 2 : ℚ) * (1 + 3 ^ (n - 2)) := 
sorry

end sequence_bound_l489_489487


namespace fraction_of_field_planted_l489_489354

open Real

def planted_fraction (a b : ℝ) (d : ℝ) : ℝ :=
  let area_triangle := (a * b) / 2
  let s := (2 / 7) ^ 2
  let area_planted := area_triangle - s
  area_planted / area_triangle

theorem fraction_of_field_planted :
  planted_fraction 3 4 2 = 145 / 147 :=
by sorry

end fraction_of_field_planted_l489_489354


namespace number_of_negations_l489_489331

variables (r s : Prop)

def statement_1 := ¬r ∧ ¬s
def statement_2 := r ∧ ¬s
def statement_3 := ¬r ∧ s
def statement_4 := r ∧ s
def negate_and (r s : Prop) := ¬(r ∧ s) -- This is equivalent to ¬r ∨ ¬s

theorem number_of_negations : 
  (statement_1 r s → negate_and r s) ∧ 
  (statement_2 r s → negate_and r s) ∧ 
  (statement_3 r s → negate_and r s) ∧ 
  ¬(statement_4 r s → negate_and r s) → 
  3 :=
by sorry

end number_of_negations_l489_489331


namespace no_two_primes_sum_to_53_l489_489474

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_two_primes_sum_to_53 :
  ¬ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_two_primes_sum_to_53_l489_489474


namespace saved_per_bagel_l489_489562

-- Definitions of the conditions
def bagel_cost_each : ℝ := 3.50
def dozen_cost : ℝ := 38
def bakers_dozen : ℕ := 13
def discount : ℝ := 0.05

-- The conjecture we need to prove
theorem saved_per_bagel : 
  let total_cost_without_discount := dozen_cost + bagel_cost_each
  let discount_amount := discount * total_cost_without_discount
  let total_cost_with_discount := total_cost_without_discount - discount_amount
  let cost_per_bagel_without_discount := dozen_cost / 12
  let cost_per_bagel_with_discount := total_cost_with_discount / bakers_dozen
  let savings_per_bagel := cost_per_bagel_without_discount - cost_per_bagel_with_discount
  let savings_in_cents := savings_per_bagel * 100
  savings_in_cents = 13.36 :=
by
  -- Placeholder for the actual proof
  sorry

end saved_per_bagel_l489_489562


namespace participant_in_all_competitions_l489_489273

variables (Participants : Type) [finite Participants]

noncomputable def competitions : fin 22 → finset (Participants) :=
λ i, sorry -- Placeholder for the set of winners in the i-th competition

lemma common_winner (i j : fin 22) (hij : i ≠ j) :
  ∃! p : Participants, p ∈ competitions i ∧ p ∈ competitions j :=
sorry

theorem participant_in_all_competitions :
  ∃ p : Participants, ∀ i : fin 22, p ∈ competitions i :=
sorry

end participant_in_all_competitions_l489_489273


namespace linear_function_through_point_zero_one_l489_489261

theorem linear_function_through_point_zero_one :
  ∃ k : ℝ, ∀ x : ℝ, (k * x + 1) = y ↔ (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1 = 0 → p.2 = 1) :=
by
  sorry

end linear_function_through_point_zero_one_l489_489261


namespace countless_lines_intersect_A1D1_EF_CD_l489_489480

/-- In the cube ABCD-A₁B₁C₁D₁, E and F are the midpoints of edges AA₁ and CC₁ respectively.
Given that lines A₁D₁, EF, and CD are three non-coplanar lines, 
prove there are countless lines that intersect with these lines in space. --/
theorem countless_lines_intersect_A1D1_EF_CD 
  (A B C D A₁ B₁ C₁ D₁ E F : Point) 
  (H1 : midpoint A A₁ E) 
  (H2 : midpoint C C₁ F) 
  (H3 : non_coplanar A₁ D₁ E F C D) 
  : ¬ finite (set_of (λ l : Line, ∃ p q r, l ∈ intersect_lines A₁D₁ EF CD)) := 
sorry

end countless_lines_intersect_A1D1_EF_CD_l489_489480


namespace parallelogram_ratio_AB_AD_l489_489564

variables (A B C D E F : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D] [AddGroup E] [AddGroup F]
variables (ABCD_E : B)
variables (DAE_EBC_F : C)
variables (ECFD_parallelogram : D)

theorem parallelogram_ratio_AB_AD (ABCD : Type)
  (h_ABCD_parallelogram : Parallelogram ABCD)
  (h_diagonals_intersect_E : IntersectAtMidpointDiagonals ABCD E)
  (h_bisectors_intersect_F : BisectorsAngleDAEandEBCIntersectAtF DAE_EBC_F E F)
  (h_ECFD_parallelogram : Parallelogram ECFD) :
  ratio (AB : AD) = sqrt 3 := sorry

end parallelogram_ratio_AB_AD_l489_489564


namespace distinct_prime_factors_of_144_l489_489865

theorem distinct_prime_factors_of_144 : (finset.card (nat.factors 144).to_finset) = 2 :=
by
  sorry

end distinct_prime_factors_of_144_l489_489865


namespace arccos_neg_one_eq_pi_l489_489753

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_eq_pi_l489_489753


namespace property_check_l489_489337

noncomputable def f (x : ℝ) : ℤ := ⌈x⌉ -- Define the ceiling function

theorem property_check :
  (¬ (∀ x : ℝ, f (2 * x) = 2 * f x)) ∧
  (∀ x1 x2 : ℝ, f x1 = f x2 → |x1 - x2| < 1) ∧
  (∀ x1 x2 : ℝ, f (x1 + x2) ≤ f x1 + f x2) ∧
  (¬ (∀ x : ℝ, f x + f (x + 0.5) = f (2 * x))) :=
by
  sorry

end property_check_l489_489337


namespace smallest_S_condition_l489_489646

noncomputable def smallest_S := 338

theorem smallest_S_condition (n : ℕ) : 
  (∀ d : ℕ, 1 ≤ d ∧ d ≤ 6) →
  (prob_sum_2000 : ℝ) > 0 →
  n ≥ 334 →
  sum_prob_eq : (prob_sum_2000 = prob_sum n smallest_S) →
  smallest_S = 338 := 
by
  -- proof omitted
  sorry

-- Definitions of placeholders (these wouldn't typically be provided
-- in the same file in real-world use but are necessary for a complete
-- Lean 4 statement)
def prob_sum (n : ℕ) (s: ℕ) := sorry
def prob_sum_2000 := sorry

end smallest_S_condition_l489_489646


namespace derivative_at_one_l489_489444

noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def f (x : ℝ) (n : ℕ) : ℝ :=
  ∑ r in Finset.range (n + 1), (C n r) * (-1)^r * x^(2*n - 1 + r)

theorem derivative_at_one (n : ℕ) (h : 0 < n) : 
  deriv (f x n) 1 = 0 :=
by
  sorry

end derivative_at_one_l489_489444


namespace max_students_with_perfect_square_sums_l489_489888

theorem max_students_with_perfect_square_sums : ∀ (n : ℤ), 
  let S := (∑ i in (Finset.range 10), (n + i)) in
  let sums := finset.image (λ k, S - (n + k)) (Finset.range 10) in
  (∑ k in Finset.range 10, if ∃ m : ℤ, (S - (n + k)) = m^2 then 1 else 0) ≤ 4 :=
by
  sorry

end max_students_with_perfect_square_sums_l489_489888


namespace angle_neg_a_b_l489_489442

open Real

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Definition and hypothesis
def angle_between_vectors (u v : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  Real.acos ((inner u v) / (∥u∥ * ∥v∥))

def angle_equiv (d : ℝ) : Prop :=
  angle_between_vectors (-a) b = d

theorem angle_neg_a_b (h : angle_between_vectors a b = π / 3) :
  angle_equiv (2 * π / 3) :=
sorry

end angle_neg_a_b_l489_489442


namespace coeff_a8_in_expansion_l489_489398

theorem coeff_a8_in_expansion :
  let P : ℕ → Polynomial ℚ := λ n => Polynomial.monomial n (1 : ℚ)
  let Q : (Polynomial ℚ) := (Polynomial.X + 1) ^ 10
  let R : (Polynomial ℚ -> Polynomial ℚ) := Polynomial.eval (1 - Polynomial.X)
  let expanded : Polynomial ℚ := R Q
  expanded.coeff 8 = 180 :=
by
  sorry

end coeff_a8_in_expansion_l489_489398


namespace jake_bitcoin_final_amount_l489_489493

-- Definitions based on the conditions
def initial_bitcoins : ℕ := 120
def investment_value (bitcoins : ℕ) := bitcoins * 2
def donation_value_half (bitcoins : ℕ) := bitcoins / 2
def donates_to_charity (bitcoins : ℕ) := 25
def brother_returns (bitcoins : ℕ) := 5
def quadruples (bitcoins : ℕ) := bitcoins * 4
def final_donation (bitcoins : ℕ) := 15 * 4

-- Lean 4 statement for the proof problem
theorem jake_bitcoin_final_amount : 
  let remaining_after_investment := initial_bitcoins - 40 in
  let remaining_after_donation := remaining_after_investment - donation_value_half(donates_to_charity(25)) in
  let remaining_after_giving_to_brother := remaining_after_donation / 2 in
  let remaining_after_returned_by_brother := remaining_after_giving_to_brother + brother_returns(5) in
  let remaining_after_quadruple := quadruples(remaining_after_returned_by_brother) in
  let final_remaining := remaining_after_quadruple - final_donation(15) in
  final_remaining = 95 :=
begin
  sorry
end

end jake_bitcoin_final_amount_l489_489493


namespace european_chess_cup_possible_seating_l489_489515

theorem european_chess_cup_possible_seating 
  (k : ℕ) (hk : k > 0) 
  (P : Type) [fintype P] [decidable_eq P]
  (played_all_pairs : ∀ (a b : P), a ≠ b → (a ≠ b ∧ (∀ p : P, p ≠ a ∧ p ≠ b → (∃ q : P, q ≠ a ∧ q ≠ b ∧ q ≠ p))))
  (exists_defeated_all : ∀ (S : finset P), S.card = k → (∃ p : P, ∀ s ∈ S, (∃ (win : p ≠ s) (lost : p ≠ s), q ≠ p ≠ s)))
  (least_number_players : ∀ (Q : finset P), Q.nonempty → Q.card < finset.card ↔ Q ⊆ P) :
  ∃ seating : list P, ((∀ i, (∃ j, i + 1 = j ∧ j < seating.length) ∧
  winning (seating.nth (i % seating.length)) (seating.nth ((i + 1) % seating.length)) ∧
  losing (seating.nth (i % seating.length)) (seating.nth ((i - 1 + seating.length) % seating.length)))) :=
sorry

end european_chess_cup_possible_seating_l489_489515


namespace scientific_notation_correct_l489_489546

def million : ℝ := 10^6
def num : ℝ := 1.06
def num_in_million : ℝ := num * million
def scientific_notation : ℝ := 1.06 * 10^6

theorem scientific_notation_correct : num_in_million = scientific_notation :=
by 
  -- The proof is skipped, indicated by sorry
  sorry

end scientific_notation_correct_l489_489546


namespace laura_drives_per_week_234_l489_489943

def LauraProblem : Type := 
  {house_school_round_trip : Nat,
   house_supermarket_distance : Nat,
   gym_visits_per_week : Nat,
   house_gym_distance : Nat,
   friend_visit_per_week : Nat,
   house_friend_distance : Nat,
   house_workplace_distance : Nat,
   school_days_per_week : Nat,
   supermarket_visits_per_week : Nat}

axiom laura_conditions : LauraProblem :=
  { house_school_round_trip := 20,
    house_supermarket_distance := 10,
    gym_visits_per_week := 3,
    house_gym_distance := 5,
    friend_visit_per_week := 1,
    house_friend_distance := 12,
    house_workplace_distance := 8,
    school_days_per_week := 5,
    supermarket_visits_per_week := 2 }

theorem laura_drives_per_week_234 (L : LauraProblem) : 
  (8 + (L.house_school_round_trip / 2 - L.house_workplace_distance) + (L.house_school_round_trip / 2)) * L.school_days_per_week
  + ((2 * (L.house_supermarket_distance + L.house_school_round_trip / 2))) * L.supermarket_visits_per_week
  + (2 * (L.house_gym_distance)) * L.gym_visits_per_week
  + (2 * (L.house_friend_distance)) * L.friend_visit_per_week = 234 := 
by
  cases L
  exact sorry

end laura_drives_per_week_234_l489_489943


namespace people_in_group_l489_489205

theorem people_in_group (N : ℕ) (h1 : avg_weight_increase = 2.5) (h2 : old_weight = 65) (h3 : new_weight = 87.5)
  (h4 : increase_in_weight = new_weight - old_weight) (h5 : total_increase = avg_weight_increase * N) : N = 9 := by
  have avg_weight_increase := 2.5
  have old_weight := 65
  have new_weight := 87.5
  have increase_in_weight := 22.5
  have total_increase :=  (avg_weight_increase * N)
  sorry

end people_in_group_l489_489205


namespace max_pages_within_budget_l489_489132

-- Definitions based on the problem conditions
def page_cost_in_cents : ℕ := 5
def total_budget_in_cents : ℕ := 5000
def max_expenditure_in_cents : ℕ := 4500

-- Proof problem statement
theorem max_pages_within_budget : 
  ∃ (pages : ℕ), pages = max_expenditure_in_cents / page_cost_in_cents ∧ 
                  pages * page_cost_in_cents ≤ total_budget_in_cents :=
by {
  sorry
}

end max_pages_within_budget_l489_489132


namespace minimum_n_pairwise_coprime_contains_prime_l489_489161

-- Definition of the set S
def S : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2005}

-- Definition of pairwise co-prime
def pairwise_coprime (A : Set ℕ) : Prop :=
  ∀ x y ∈ A, x ≠ y → Nat.coprime x y

-- Theorem statement
theorem minimum_n_pairwise_coprime_contains_prime :
  (∀ A ⊆ S, pairwise_coprime A → A.Finite → A.card ≥ 16 → (∃ p ∈ A, Nat.Prime p)) ∧
  (∃ A ⊆ S, pairwise_coprime A ∧ A.card = 15 ∧ (∀ p ∈ A, ¬Nat.Prime p)) :=
by
  sorry

end minimum_n_pairwise_coprime_contains_prime_l489_489161


namespace triangle_MPE_area_fraction_l489_489455

variables {A B C : Type} [EuclideanGeometry A B C] 
variables (M : Point) (P : Point) (D : Point) (E : Point)
variables {triangle_area : finset A → ℝ} 
variable {is_incenter : Point → Bool}
variable {is_midpoint : Point → Point → Point → Bool}
variable {are_similar : Triangle → Triangle → Bool}

-- Given incenter M of triangle ABC where AD and CE are angle bisectors
axiom angle_bisectors_intersect : is_incenter M = true

-- Given P is the midpoint of BC
axiom midpoint_BC : is_midpoint P B C = true

-- Goal: area of triangle MPE is 1/12 of the area of triangle ABC
theorem triangle_MPE_area_fraction
  {ΔABC : finset A}
  (h1 : is_incenter M = true)
  (h2 : is_midpoint P B C = true) :
  ∃ k : ℝ, k = 1/12 ∧ triangle_area (finset.image id {M, P, E}) = k * triangle_area ΔABC := sorry

end triangle_MPE_area_fraction_l489_489455


namespace unique_even_three_digit_numbers_l489_489237

/-- 
The number of unique three-digit even numbers that can be formed using 
the digits 0 to 9 without repeating any digits is 296.
-/
theorem unique_even_three_digit_numbers : ∃ n : ℤ, n = 296 ∧ 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
      evens := {2, 4, 6, 8};
  (∃ units ∈ evens,
    ∃ tens ∈ digits \ {units},
    ∃ hundreds ∈ digits \ {units, tens},
    hundreds ≠ 0) ∧
  (∃ units = 0,
    ∃ tens ∈ digits \ {units, 0},
    ∃ hundreds ∈ digits \ {units, tens},
    tens ≠ 0) := 
sorry

end unique_even_three_digit_numbers_l489_489237


namespace two_numbers_equal_l489_489756

variables {a b c : ℝ}
variable (h1 : a + b^2 + c^2 = a^2 + b + c^2)
variable (h2 : a^2 + b + c^2 = a^2 + b^2 + c)

theorem two_numbers_equal (h1 : a + b^2 + c^2 = a^2 + b + c^2) (h2 : a^2 + b + c^2 = a^2 + b^2 + c) :
  a = b ∨ a = c ∨ b = c :=
by
  sorry

end two_numbers_equal_l489_489756


namespace intersection_is_correct_l489_489422

-- Define set M
def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- Define set N
def N : Set ℕ := { 0, 1, 2 }

-- Define the intersection of M and N
def intersection : Set ℝ := { x ∈ M | x ∈ N }

-- The target proof problem:
theorem intersection_is_correct : intersection = {0, 1} :=
sorry

end intersection_is_correct_l489_489422


namespace inequality_proof_l489_489381

variable (x y z : ℝ)

theorem inequality_proof
  (h : x + 2*y + 3*z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 14 :=
by
  sorry

end inequality_proof_l489_489381


namespace area_of_tangency_triangle_l489_489325

theorem area_of_tangency_triangle (r1 r2 r3 : ℕ) (h1 : r1 = 1) (h2 : r2 = 3) (h3 : r3 = 4) 
  (tangential: ∀ p q, p = r1 + r2 ∧ q = r2 + r3 ∧ p = r1 + r3) : real :=
  let a := r1 + r2 in
  let b := r2 + r3 in
  let c := r1 + r3 in
  let s := (a + b + c) / 2 in
  let area := real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area := 4 * real.sqrt 6

end area_of_tangency_triangle_l489_489325


namespace angle_COD_triangle_l489_489489

theorem angle_COD_triangle (Q C B D : Type) [InCircumCircle Q C B D] 
  (tangent_to_circle : TangentToCircle Q C B D)
  (angle_CQB : Angle Q C B = 60) :
  Angle C O D = 120 :=
by
  sorry

end angle_COD_triangle_l489_489489


namespace num_valid_7tuples_l489_489368

theorem num_valid_7tuples : 
  ∃ (S : Finset (Fin 7 → ℤ)), 
  (∀ v ∈ S, (∑ i, (v i)^6) = 96957) ∧ S.card = 2688 := 
sorry

end num_valid_7tuples_l489_489368


namespace kath_total_cost_l489_489113

def admission_cost : ℝ := 8
def discount_percentage_pre6pm : ℝ := 0.25
def discount_percentage_student : ℝ := 0.10
def time_of_movie : ℝ := 4
def num_people : ℕ := 6
def num_students : ℕ := 2

theorem kath_total_cost :
  let discounted_price := admission_cost * (1 - discount_percentage_pre6pm)
  let student_price := discounted_price * (1 - discount_percentage_student)
  let num_non_students := num_people - num_students - 1 -- remaining people (total - 2 students - Kath)
  let kath_and_siblings_cost := 3 * discounted_price
  let student_friends_cost := num_students * student_price
  let non_student_friend_cost := num_non_students * discounted_price
  let total_cost := kath_and_siblings_cost + student_friends_cost + non_student_friend_cost
  total_cost = 34.80 := by
  let discounted_price := admission_cost * (1 - discount_percentage_pre6pm)
  let student_price := discounted_price * (1 - discount_percentage_student)
  let num_non_students := num_people - num_students - 1
  let kath_and_siblings_cost := 3 * discounted_price
  let student_friends_cost := num_students * student_price
  let non_student_friend_cost := num_non_students * discounted_price
  let total_cost := kath_and_siblings_cost + student_friends_cost + non_student_friend_cost
  sorry

end kath_total_cost_l489_489113


namespace three_digit_numbers_sum_seven_l489_489010

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489010


namespace polynomial_roots_identity_l489_489538

-- Let α and β be the roots of the polynomial x^2 + px + 1
def roots_of_polynomial_p (p : ℝ) : ℝ × ℝ :=
  let α := (-p + math.sqrt (p^2 - 4)) / 2
  let β := (-p - math.sqrt (p^2 - 4)) / 2
  (α, β)

-- Let γ and δ be the roots of the polynomial x^2 + qx + 1
def roots_of_polynomial_q (q : ℝ) : ℝ × ℝ :=
  let γ := (-q + math.sqrt (q^2 - 4)) / 2
  let δ := (-q - math.sqrt (q^2 - 4)) / 2
  (γ, δ)

theorem polynomial_roots_identity 
  (p q : ℝ)
  (α β : ℝ) 
  (γ δ : ℝ)
  (hαβ : roots_of_polynomial_p p = (α, β))
  (hγδ : roots_of_polynomial_q q = (γ, δ)) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 := 
sorry

end polynomial_roots_identity_l489_489538


namespace increase_in_output_with_assistant_l489_489919

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l489_489919


namespace orthocentric_tetrahedron_distance_l489_489513

open EuclideanGeometry

/-- Given H as the orthocenter of an orthocentric tetrahedron, M as the centroid of one of its faces,
    and N as one of the points where the line HM intersects the circumsphere of the tetrahedron
    (with M between H and N), prove that |MN| = 2|HM|. -/
theorem orthocentric_tetrahedron_distance (H M N : Point ℝ)
  (tetrahedron : ∀ (V : ℕ), V < 4 → Point ℝ)
  (orthocentric : orthocenter_δοκ_textahedron tetrahedron H)
  (centroid_face : ∃ face, is_face_of tetrahedron face ∧ face_center face = M)
  (circumsphere : N ∈ circumsphere tetrahedron)
  (H_collinear_MN : collinear {H, M, N})
  (M_between_HN : between H M N) :
  dist M N = 2 * dist H M :=
sorry

end orthocentric_tetrahedron_distance_l489_489513


namespace polygon_sides_l489_489591

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l489_489591


namespace john_shirts_after_donation_l489_489940

noncomputable def initial_shirts := 150
noncomputable def designer_percentage := 0.40
noncomputable def non_designer_percentage := 0.60
noncomputable def additional_designer := 8
noncomputable def additional_non_designer := 5
noncomputable def buy_non_designer := 6
noncomputable def free_non_designer := 6
noncomputable def donate_designer_percentage := 0.50
noncomputable def donate_non_designer_percentage := 0.30

theorem john_shirts_after_donation :
  let total_shirts_initial := initial_shirts,
      designer_initial := initial_shirts * designer_percentage,
      non_designer_initial := initial_shirts * non_designer_percentage,
      
      designer_after_purchase := designer_initial + additional_designer,
      non_designer_after_gift := non_designer_initial + additional_non_designer,
      non_designer_after_buy_one_get_one := non_designer_after_gift + buy_non_designer + free_non_designer,
      
      total_before_donation := designer_after_purchase + non_designer_after_buy_one_get_one,
      
      designer_donated := designer_after_purchase * donate_designer_percentage,
      non_designer_donated := non_designer_after_buy_one_get_one * donate_non_designer_percentage,
      
      designer_left := designer_after_purchase - designer_donated,
      non_designer_left := non_designer_after_buy_one_get_one - non_designer_donated
  in designer_left + non_designer_left = 109 :=
begin
  sorry,
end

end john_shirts_after_donation_l489_489940


namespace find_a_extremum_and_min_value_find_max_k_l489_489851

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x + 1

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a

theorem find_a_extremum_and_min_value :
  (∀ a : ℝ, f' a 0 = 0 → a = -1) ∧
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → f (-1) x ≥ 2) :=
by sorry

theorem find_max_k (k : ℤ) :
  (∀ x : ℝ, 0 < x → k * (Real.exp x - 1) < x * Real.exp x + 1) →
  k ≤ 2 :=
by sorry

end find_a_extremum_and_min_value_find_max_k_l489_489851


namespace area_of_CEF_l489_489118

-- Definitions of points and triangles based on given ratios
def is_right_triangle (A B C : Type) : Prop := sorry -- Placeholder for right triangle condition

def divides_ratio (A B : Type) (ratio : ℚ) : Prop := sorry -- Placeholder for ratio division condition

def area_of_triangle (A B C : Type) : ℚ := sorry -- Function to calculate area of triangle - placeholder

theorem area_of_CEF {A B C E F : Type} 
  (h1 : is_right_triangle A B C)
  (h2 : divides_ratio A C (1/4))
  (h3 : divides_ratio A B (2/3))
  (h4 : area_of_triangle A B C = 50) : 
  area_of_triangle C E F = 25 :=
sorry

end area_of_CEF_l489_489118


namespace probability_correct_match_l489_489294

/-- 
In a scenario where a school magazine features four students and also showcases four childhood pictures of these students, 
prove that the probability of a student randomly guessing and correctly associating all students with their childhood pictures is 1/24.
-/
theorem probability_correct_match : 
  let total_arrangements := (4! : ℤ),
      correct_arrangements := 1 in
  (1 : ℚ) / total_arrangements = 1 / 24 :=
by
  let total_arrangements := factorial 4,
  let correct_arrangements := 1,
  have h1 : total_arrangements = 24 := by simp [total_arrangements],
  have h2 : (correct_arrangements : ℚ) / total_arrangements = 1 / 24 := by norm_num [correct_arrangements, total_arrangements],
  exact h2
  sorry

end probability_correct_match_l489_489294


namespace travel_time_l489_489214

noncomputable def distance (time: ℝ) (rate: ℝ) : ℝ := time * rate

theorem travel_time
  (initial_time: ℝ)
  (initial_speed: ℝ)
  (reduced_speed: ℝ)
  (stopover: ℝ)
  (h1: initial_time = 4)
  (h2: initial_speed = 80)
  (h3: reduced_speed = 50)
  (h4: stopover = 0.5) :
  (distance initial_time initial_speed) / reduced_speed + stopover = 6.9 := 
by
  sorry

end travel_time_l489_489214


namespace x_lt_1_iff_x_abs_x_lt_1_l489_489035

theorem x_lt_1_iff_x_abs_x_lt_1 (x : ℝ) : x < 1 ↔ x * |x| < 1 :=
sorry

end x_lt_1_iff_x_abs_x_lt_1_l489_489035


namespace distinct_integer_line_segments_l489_489757

-- Define the isosceles right triangle with given sides
def isosceles_right_triangle (a : ℕ) := ∃ (DE DF EF DG : ℝ),
  DE = a ∧ DF = a ∧ EF = (real.sqrt 2) * a ∧ DG = (real.sqrt 2 / 2) * a

-- Prove the number of distinct integer line segments
theorem distinct_integer_line_segments (a : ℕ) (h : isosceles_right_triangle 24)
  : ∃ n : ℕ, n = 8 :=
by
  sorry

end distinct_integer_line_segments_l489_489757


namespace intersection_of_intervals_l489_489099

theorem intersection_of_intervals (m n x : ℝ) (h1 : -1 < m) (h2 : m < 0) (h3 : 0 < n) :
  (m < x ∧ x < n) ∧ (-1 < x ∧ x < 0) ↔ -1 < x ∧ x < 0 :=
by sorry

end intersection_of_intervals_l489_489099


namespace dot_product_a_b_l489_489948

variables (e1 e2 : Vector ℝ)
variable h1 : norm e1 = 1
variable h2 : norm e2 = 1
variable h3 : ⟪e1, e2⟫ = -1/5
let a := 2 • e1 - e2
let b := e1 + 3 • e2

theorem dot_product_a_b : ⟪a, b⟫ = -2 :=
by
  sorry

end dot_product_a_b_l489_489948


namespace circle_line_intersection_points_l489_489764

theorem circle_line_intersection_points :
  let circle_eqn : ℝ × ℝ → Prop := fun p => (p.1 - 1)^2 + p.2^2 = 16
  let line_eqn  : ℝ × ℝ → Prop := fun p => p.1 = 4
  ∃ (p₁ p₂ : ℝ × ℝ), 
    circle_eqn p₁ ∧ line_eqn p₁ ∧ circle_eqn p₂ ∧ line_eqn p₂ ∧ p₁ ≠ p₂ 
      → ∀ (p : ℝ × ℝ), circle_eqn p ∧ line_eqn p → 
        p = p₁ ∨ p = p₂ ∧ (p₁ ≠ p ∨ p₂ ≠ p)
 := sorry

end circle_line_intersection_points_l489_489764


namespace minimum_value_of_expression_l489_489058

theorem minimum_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 := 
sorry

end minimum_value_of_expression_l489_489058


namespace B_power_15_minus_3_B_power_14_l489_489145

def B : Matrix (Fin 2) (Fin 2) ℝ := !!
  [3, 4]
  [0, 2]

theorem B_power_15_minus_3_B_power_14 :
  B^15 - 3 • B^14 = !!
    [0, 4]
    [0, -1] := by
  sorry

end B_power_15_minus_3_B_power_14_l489_489145


namespace bug_total_distance_l489_489677

theorem bug_total_distance :
  let start_position := -2
  let first_stop := -6
  let final_position := 5
  abs(first_stop - start_position) + abs(final_position - first_stop) = 15 :=
by
  sorry

end bug_total_distance_l489_489677


namespace problem_l489_489085

noncomputable def f (x : ℝ) : ℝ := |x + 1| - |x|

theorem problem :
  (∀ x, f x ≤ 1) ∧
  (∃ x, f x = 1) ∧
  (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 → 
    ∃ x, (x = (a^2 / (b + 1) + b^2 / (a + 1)) ∧ x = 1 / 3)) :=
by {
  sorry
}

end problem_l489_489085


namespace expected_value_after_8_minutes_floor_div_10_eq_13_l489_489539

-- Define the initial number Rachel has.
def initial_number : ℝ := 1000

-- Define the transformation when Rachel puts the number in her left pocket.
def left_pocket (x : ℝ) : ℝ := x + 1

-- Define the transformation when Rachel puts the number in her right pocket.
def right_pocket (x : ℝ) : ℝ := x⁻¹

-- Define the expected value formula for one minute.
def expected_value_one_minute (X : ℝ) : ℝ :=
  (left_pocket X) / 2 + (right_pocket X) / 2

-- Define the recurrence relation for the expected value after n minutes.
noncomputable def expected_value_n_minutes (n : ℕ) : ℝ :=
  Nat.recOn n initial_number
    (λ _ E_n, (left_pocket E_n) / 2 + (right_pocket E_n) / 2)

-- Prove the final statement.
theorem expected_value_after_8_minutes_floor_div_10_eq_13 : 
  (⌊ expected_value_n_minutes 8 / 10 ⌋) = 13 := 
by
  sorry

end expected_value_after_8_minutes_floor_div_10_eq_13_l489_489539


namespace area_of_TRIangle_EIF_l489_489453

variables {ABC : Type*} [triangle ABC] 
variable {A B C D E F I : point ABC}
variables [is_isosceles ABC A B C] [angle_bisector A D B C] [angle_bisector B E A C]
variable [incenter I ABC] [midpoint F A C]

theorem area_of_TRIangle_EIF (ABC : Type*) [triangle ABC] 
  (A B C D E F I : point ABC)
  [is_isosceles ABC A B C]
  [angle_bisector A D B C]
  [angle_bisector B E A C]
  [incenter I ABC] [midpoint F A C] :
  area (triangle EIF) = (1 / 8) * area (triangle ABC) :=
sorry

end area_of_TRIangle_EIF_l489_489453


namespace refuel_stops_required_l489_489210

noncomputable def tank_capacity : ℕ := 50
noncomputable def distance : ℕ := 2560
noncomputable def consumption_rate : ℕ := 8
noncomputable def safety_reserve : ℕ := 6

theorem refuel_stops_required :
  let total_consumption := (distance * consumption_rate) / 100,
      usable_fuel_per_tank := tank_capacity - safety_reserve,
      full_tanks_needed := total_consumption / usable_fuel_per_tank in
  full_tanks_needed.ceil - 1 = 4 :=
by sorry

end refuel_stops_required_l489_489210


namespace number_of_integer_pairs_l489_489791

-- Definitions for conditions
def f (x : ℝ) : ℝ := Real.log x / Real.log 4
def g (x : ℝ) : ℝ := 70 + x - (4^70)

-- Statement of the problem
theorem number_of_integer_pairs :
  let S := { p : ℤ × ℤ | let (x, y) := p in y ≥ 70 + x - 4^70 ∧ y ≤ Real.log x / Real.log 4 } in
  S.card = (1 / 2) * 4^140 - (5 / 6) * 4^70 + 214 / 3 :=
by sorry

end number_of_integer_pairs_l489_489791


namespace length_of_BC_l489_489446

noncomputable def perimeter (a b c : ℝ) := a + b + c
noncomputable def area (b c : ℝ) (A : ℝ) := 0.5 * b * c * (Real.sin A)

theorem length_of_BC
  (a b c : ℝ)
  (h_perimeter : perimeter a b c = 20)
  (h_area : area b c (Real.pi / 3) = 10 * Real.sqrt 3) :
  a = 7 :=
by
  sorry

end length_of_BC_l489_489446


namespace fibonacci_properties_l489_489776

theorem fibonacci_properties (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 1) (h_rec : ∀ n : ℕ, a (n+2) = a (n+1) + a n) :
  (∃ k, a 1005 + a 1001 = 3 * a k ∧ k = 1003) ∧
  (∃ m, (∑ i in (Finset.range 1001), a i * a (i+1)) = a m ^ 2 ∧ m = 1002) :=
by
  sorry

end fibonacci_properties_l489_489776


namespace problem_l489_489960

noncomputable def A (m : ℝ) := {y : ℝ | ∃ x : ℝ, y = Real.sin x - Real.cos (x + Real.pi / 6) + m}
def B := {y : ℝ | ∃ x : ℝ, x ∈ Icc 1 2 ∧ y = -x^2 + 2 * x}
def p (m : ℝ) := ∃ x : ℝ, x ∈ A m
def q := ∃ x : ℝ, x ∈ B

theorem problem (m : ℝ) (h : ¬ p m → ¬ q) : 1 - Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
by
  sorry

end problem_l489_489960


namespace radius_of_circle_formed_by_spherical_coordinates_l489_489330

theorem radius_of_circle_formed_by_spherical_coordinates :
  ∀ θ : ℝ, let ρ := 2 in let φ := real.pi / 4 in
   let x := ρ * real.sin φ * real.cos θ in 
   let y := ρ * real.sin φ * real.sin θ in
   let radius := real.sqrt (x^2 + y^2) in
   radius = real.sqrt 2 := 
by 
  sorry

end radius_of_circle_formed_by_spherical_coordinates_l489_489330


namespace real_roots_m_range_find_value_of_m_l489_489385

-- Part 1: Prove the discriminant condition for real roots
theorem real_roots_m_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - (2 * m + 3) * x + m^2 + 2 = 0) ↔ m ≥ -1/12 := 
sorry

-- Part 2: Prove the value of m given the condition on roots
theorem find_value_of_m (m : ℝ) (x1 x2 : ℝ) 
  (h : x1^2 + x2^2 = 3 * x1 * x2 - 14)
  (h_roots : x^2 - (2 * m + 3) * x + m^2 + 2 = 0 → (x = x1 ∨ x = x2)) :
  m = 13 := 
sorry

end real_roots_m_range_find_value_of_m_l489_489385


namespace max_notebooks_l489_489324

theorem max_notebooks (money : ℝ) (cost_per_notebook : ℝ) (n : ℕ) (h1 : money = 12) (h2 : cost_per_notebook = 1.45) : n ≤ 8 :=
by
  -- Given conditions
  have money_nonnegative : money ≥ 0 := by linarith
  have cost_nonnegative : cost_per_notebook > 0 := by linarith
  
  -- Calculate the maximum number of notebooks
  have max_n_real := money / cost_per_notebook
  
  -- Prove the ceiling of (money / cost_per_notebook) is 8
  have max_n_ceil := Real.floor max_n_real

  -- Since max_n_ceil is a whole number which rounds down the division
  have max_n_ceil_nat : max_n_ceil = 8 := sorry

  -- Therefore, the maximum integer n ≤ 8
  exact le_of_lt (Int.lt_of_le_of_lt (Int.ofNat_le.2 (Int.le_of_lt max_n_ceil_nat)) (by norm_num : (8 : ℤ) < max_n_ceil))

end max_notebooks_l489_489324


namespace range_of_m_l489_489872

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∀ x y : ℝ, 0 < x → 0 < y → (2 * y / x + 9 * x / (2 * y) ≥ m^2 + m))
  ↔ (-3 ≤ m ∧ m ≤ 2) := sorry

end range_of_m_l489_489872


namespace limit_ratio_limit_ratio_eq_limit_ratio_zero_l489_489162

def A_n (a : ℝ) (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ j in finRange n.succ, x j * a^j

def B_n (b : ℝ) (x : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ j in finRange n.succ, x j * b^j

theorem limit_ratio (a b : ℝ) (x : ℕ → ℝ) (h_pos : ∀ n, 0 ≤ x n):
  (a > b) → (filter.tendsto (λ n, (A_n a x n) / (B_n b x n)) filter.at_top filter.at_top) :=
sorry

theorem limit_ratio_eq (a b : ℝ) (x : ℕ → ℝ) (h_pos : ∀ n, 0 ≤ x n):
  (a = b) → (filter.tendsto (λ n, (A_n a x n) / (B_n b x n)) filter.at_top (nhds 1)) :=
sorry

theorem limit_ratio_zero (a b : ℝ) (x : ℕ → ℝ) (h_pos : ∀ n, 0 ≤ x n):
  (a < b) → (filter.tendsto (λ n, (A_n a x n) / (B_n b x n)) filter.at_top (nhds 0)) :=
sorry

end limit_ratio_limit_ratio_eq_limit_ratio_zero_l489_489162


namespace log9_6_eq_mn_over_2_l489_489873

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log9_6_eq_mn_over_2
  (m n : ℝ)
  (h1 : log_base 7 4 = m)
  (h2 : log_base 4 6 = n) : 
  log_base 9 6 = (m * n) / 2 := by
  sorry

end log9_6_eq_mn_over_2_l489_489873


namespace prop_D_l489_489432

variable (a b : ℝ)

theorem prop_D (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
  by
    sorry

end prop_D_l489_489432


namespace num_divisors_of_3d_plus_15_l489_489553

theorem num_divisors_of_3d_plus_15 (c d : ℤ) (h : 4 * d = 10 - 3 * c) :
  (finset.range 8).filter (λ n, (3 * d + 15) % (n + 1) = 0) = {0, 1, 2, 5} :=
by {
  sorry
}

end num_divisors_of_3d_plus_15_l489_489553


namespace folder_cost_l489_489939

theorem folder_cost (cost_pens : ℕ) (cost_notebooks : ℕ) (total_spent : ℕ) (folders : ℕ) :
  cost_pens = 3 → cost_notebooks = 12 → total_spent = 25 → folders = 2 →
  ∃ (cost_per_folder : ℕ), cost_per_folder = 5 :=
by
  intros
  sorry

end folder_cost_l489_489939


namespace fortieth_term_is_81_l489_489133

noncomputable def arithmetic_sequence := λ (n : ℕ), 3 + (n - 1) * 2

theorem fortieth_term_is_81 : arithmetic_sequence 40 = 81 :=
  sorry

end fortieth_term_is_81_l489_489133


namespace ryan_learning_hours_l489_489773

theorem ryan_learning_hours :
  ∀ (e c s : ℕ) , (e = 6) → (s = 58) → (e = c + 3) → (c = 3) :=
by
  intros e c s he hs hc
  sorry

end ryan_learning_hours_l489_489773


namespace check_roots_l489_489794

noncomputable def roots_of_quadratic_eq (a b : ℂ) : list ℂ :=
  [(-b + complex.sqrt(b ^ 2 - 4 * a * (3 - 4 * complex.i))) / (2 * a),
   (-b - complex.sqrt(b ^ 2 - 4 * a * (3 - 4 * complex.i))) / (2 * a)]

theorem check_roots :
  ∀ (z : ℂ), z^2 + 2 * z + (3 - 4 * complex.i) = 0 ↔ (z = complex.i ∨ z = -3 - 2 * complex.i) :=
begin
  sorry
end

end check_roots_l489_489794


namespace books_read_l489_489654

theorem books_read (total_books remaining_books read_books : ℕ)
  (h_total : total_books = 14)
  (h_remaining : remaining_books = 6)
  (h_eq : read_books = total_books - remaining_books) : read_books = 8 := 
by 
  sorry

end books_read_l489_489654


namespace convert_degrees_to_minutes_and_seconds_l489_489332

/-- Definition of conversion from degrees to minutes -/
def degreesToMinutes (d : ℝ) : ℝ := d * 60

/-- Definition of conversion from minutes to seconds -/
def minutesToSeconds (m : ℝ) : ℝ := m * 60

theorem convert_degrees_to_minutes_and_seconds (d : ℝ) :
  d = 1.45 → degreesToMinutes d = 87 ∧ minutesToSeconds 0.45 = 27 :=
by
  intro h
  rw [h]
  split
  · calc
      degreesToMinutes 1.45 = 1.45 * 60 : rfl
                        ... = 87       : by norm_num
  · calc
      minutesToSeconds 0.45 = 0.45 * 60 : rfl
                        ... = 27       : by norm_num
  sorry

end convert_degrees_to_minutes_and_seconds_l489_489332


namespace intersection_points_distance_l489_489053

noncomputable def ellipse_eqn : (ℝ × ℝ) → ℝ := λ p, (p.1^2 / 2) + p.2^2 - 1
def line_eqn : (ℝ × ℝ) → ℝ := λ p, p.2 - p.1 - 1/2

theorem intersection_points_distance :
  let A := (x1, x1 + 1/2),
      B := (x2, x2 + 1/2),
      roots := quadratic_roots (3 * x^2 + 2 * x - 3/2)
  in 
  ellipse_eqn A = 0 ∧ ellipse_eqn B = 0 ∧ line_eqn A = 0 ∧ line_eqn B = 0 →
  |AB| = (2 * sqrt 11) / 3 :=
by 
  sorry

end intersection_points_distance_l489_489053


namespace cost_per_scarf_l489_489633

-- Define the cost of each earring
def cost_of_earring : ℕ := 6000

-- Define the number of earrings
def num_earrings : ℕ := 2

-- Define the cost of the iPhone
def cost_of_iphone : ℕ := 2000

-- Define the number of scarves
def num_scarves : ℕ := 4

-- Define the total value of the swag bag
def total_swag_bag_value : ℕ := 20000

-- Define the total value of diamond earrings and the iPhone
def total_value_of_earrings_and_iphone : ℕ := (num_earrings * cost_of_earring) + cost_of_iphone

-- Define the total value of the scarves
def total_value_of_scarves : ℕ := total_swag_bag_value - total_value_of_earrings_and_iphone

-- Define the cost of each designer scarf
def cost_of_each_scarf : ℕ := total_value_of_scarves / num_scarves

-- Prove that each designer scarf costs $1,500
theorem cost_per_scarf : cost_of_each_scarf = 1500 := by
  sorry

end cost_per_scarf_l489_489633


namespace smallest_whole_number_l489_489370

theorem smallest_whole_number (a b c d : ℤ)
  (h₁ : a = 3 + 1 / 3)
  (h₂ : b = 4 + 1 / 4)
  (h₃ : c = 5 + 1 / 6)
  (h₄ : d = 6 + 1 / 8)
  (h₅ : a + b + c + d - 2 > 16)
  (h₆ : a + b + c + d - 2 < 17) :
  17 > 16 + (a + b + c + d - 18) - 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 8 :=
  sorry

end smallest_whole_number_l489_489370


namespace locus_of_point_is_circle_l489_489828

theorem locus_of_point_is_circle (x y : ℝ) 
  (h : 10 * Real.sqrt ((x - 1)^2 + (y - 2)^2) = |3 * x - 4 * y|) : 
  ∃ (c : ℝ) (r : ℝ), ∀ (x y : ℝ), (x - c)^2 + (y - c)^2 = r^2 := 
sorry

end locus_of_point_is_circle_l489_489828


namespace cos_C_equal_l489_489457

theorem cos_C_equal (ABC : Triangle) (A B C : Point)
  (hRt : angle_deg A B C = 90)
  (hTan : tan_deg C = 6) : 
  cos_deg C = sqrt 37 / 37 :=
  sorry

end cos_C_equal_l489_489457


namespace arithmetic_sequence_sum_l489_489121

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n : ℕ, a (n+1) - a n = a 1 - a 0

theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a) (S : ℝ) 
  (hsum : ∑ i in Finset.range 10, a i = 15) : a 1 + a 8 = 3 := 
by
  sorry

end arithmetic_sequence_sum_l489_489121


namespace integer_values_count_eq_four_l489_489576

theorem integer_values_count_eq_four :
  (∃ (a : ℤ) (b c : ℕ), 
    b + c = 8 ∧ 
    if b = 2*int.natAbs(a) + 7 
    then int.natAbs(a) else 0) ≥ 4 := 
sorry

end integer_values_count_eq_four_l489_489576


namespace geometry_problem_l489_489307

variables {A B₀ B₁ C₀ C₁ P₀ Q₀ P₁ Q₁ P₀' : Type*}
variable [linear_ordered_field ℝ]

/-- Given an ellipse with foci B₀ and B₁, and intersections C₀, C₁ with the sides of triangle
    AB₀B₁ at AB₀, AB₁ respectively, along with constructions as described, prove the following: -/
theorem geometry_problem 
  (H₁ : B₀ ≠ B₁) 
  (H₂ : C₀ ≠ C₁)
  (H3 : ∃ ℓ : ℝ, is_ray (A, B₀) (P₀, ℓ))
  (H4 : is_circle_segment (B₀, P₀) (P₀, Q₀))
  (H5 : is_circle_segment (C₁, Q₀) (Q₀, P₁))
  (H6 : is_circle_segment (B₁, P₁) (P₁, Q₁))
  (H7 : is_circle_segment (C₀, Q₁) (Q₁, P₀')) :
  P₀' = P₀ ∧ tangent_at_point (P₀, Q₀) (P₀, Q₁) P₀ ∧ concyclic_points P₀ Q₀ Q₁ P₁ :=
sorry

end geometry_problem_l489_489307


namespace sum_of_coordinates_of_B_is_7_l489_489531

-- Define points and conditions
def A := (0, 0)
def B (x : ℝ) := (x, 3)
def slope (p₁ p₂ : ℝ × ℝ) : ℝ := (p₂.2 - p₁.2) / (p₂.1 - p₁.1)

-- Main theorem to prove the sum of the coordinates of point B is 7
theorem sum_of_coordinates_of_B_is_7 (x : ℝ) (h_slope : slope A (B x) = 3 / 4) : x + 3 = 7 :=
by
  -- Proof goes here, we use sorry to skip the proof steps.
  sorry

end sum_of_coordinates_of_B_is_7_l489_489531


namespace perfect_squares_mult_36_l489_489868

theorem perfect_squares_mult_36 (n : ℕ) (h : n < 40000000) (hn : ∃ k : ℕ, n = k^2) : 
  (∃ m : ℕ, n = m * 36) → card { k : ℕ | (k * k < 40000000) ∧ (∃ m : ℕ, k = m * 36) } = 175 :=
sorry

end perfect_squares_mult_36_l489_489868


namespace pentagon_PT_value_l489_489116

-- Given conditions
def length_QR := 3
def length_RS := 3
def length_ST := 3
def angle_T := 90
def angle_P := 120
def angle_Q := 120
def angle_R := 120

-- The target statement to prove
theorem pentagon_PT_value (a b : ℝ) (h : PT = a + 3 * Real.sqrt b) : a + b = 6 :=
sorry

end pentagon_PT_value_l489_489116


namespace bottle_caps_total_l489_489134

theorem bottle_caps_total (original : ℕ) (bought : ℕ) : original = 40 → bought = 7 → original + bought = 47 := by
  intros h1 h2
  rw [h1, h2]
  rfl

end bottle_caps_total_l489_489134


namespace variance_of_eta_l489_489045

noncomputable def X : ℕ → ℕ → ℝ → ℝ → Prop := λ n k p q, binomial n k

def eta (X : ℝ) : ℝ := -2 * X + 1

theorem variance_of_eta :
  ∀ (X : ℝ), X → D(eta X) = 5.76 :=
begin
  sorry
end

end variance_of_eta_l489_489045


namespace circle_centers_tangent_to_diameter_l489_489039

theorem circle_centers_tangent_to_diameter 
(center_k : ℝ × ℝ) (radius_k : ℝ) (diameter_k : ℝ) (point_k : ℝ × ℝ) 
(h_center : center_k = (0, 0)) 
(h_radius : radius_k = 1) 
(h_diameter : diameter_k = 2) 
(h_nearest_point : ∃ r L, L ∈ {L : ℝ × ℝ | 
    (L.1)^2 + (L.2)^2 = (2 * L.2 + 1)^2 ∨ (L.1)^2 + (L.2)^2 = (1 - 2 * L.2)^2 }):
    ∀ L : ℝ × ℝ, (L.1, L.2) satisfies 
    (9 * (L.2 + 2/3)^2 - 3 * L.1^2 = 1) ∨ 
    (9 * (L.2 - 2/3)^2 - 3 * L.1^2 = 1) := 
by 
    sorry

end circle_centers_tangent_to_diameter_l489_489039


namespace sum_of_possible_x_values_l489_489253

theorem sum_of_possible_x_values : 
  let lst : List ℝ := [10, 2, 5, 2, 4, 2, x]
  let mean := (25 + x) / 7
  let mode := 2
  let median := if x ≤ 2 then 2 else if 2 < x ∧ x < 4 then x else 4
  mean, median, and mode form a non-constant arithmetic progression 
  -> ∃ x_values : List ℝ, sum x_values = 20 :=
by
  sorry

end sum_of_possible_x_values_l489_489253


namespace compare_variables_l489_489840

theorem compare_variables (a b c : ℝ) (h1 : a = 2 ^ (1 / 2)) (h2 : b = Real.log 3 / Real.log π) (h3 : c = Real.log (1 / 3) / Real.log 2) : 
  a > b ∧ b > c :=
by
  sorry

end compare_variables_l489_489840


namespace limit_fraction_l489_489181

open Filter
open Topology

theorem limit_fraction {x : ℝ} :
  tendsto (λ x, (5 * x + 6) / (6 * x)) atTop (𝓝 (5 / 6)) :=
sorry

end limit_fraction_l489_489181


namespace KodyAgeIs32_l489_489024

-- Definition for Mohamed's current age
def mohamedCurrentAge : ℕ := 2 * 30

-- Definition for Mohamed's age four years ago
def mohamedAgeFourYrsAgo : ℕ := mohamedCurrentAge - 4

-- Definition for Kody's age four years ago
def kodyAgeFourYrsAgo : ℕ := mohamedAgeFourYrsAgo / 2

-- Definition to check Kody's current age
def kodyCurrentAge : ℕ := kodyAgeFourYrsAgo + 4

theorem KodyAgeIs32 : kodyCurrentAge = 32 := by
  sorry

end KodyAgeIs32_l489_489024


namespace factorial_divisibility_l489_489762

/-- Define the sequence [n]! as the product of integers consisting of n ones. -/
def factorial_n (n : ℕ) : ℕ :=
  ∏ i in finset.range n, (10^((i + 1)) - 1) / 9

/-- Prove that [n + m]! is divisible by [n]! * [m]! -/
theorem factorial_divisibility (n m : ℕ) :
  factorial_n (n + m) % (factorial_n n * factorial_n m) = 0 :=
sorry

end factorial_divisibility_l489_489762


namespace cats_left_l489_489692

theorem cats_left (siamese house persian sold_first sold_second : ℕ) (h1 : siamese = 23) (h2 : house = 17) (h3 : persian = 29) (h4 : sold_first = 40) (h5 : sold_second = 12) :
  siamese + house + persian - sold_first - sold_second = 17 :=
by sorry

end cats_left_l489_489692


namespace carla_chickens_disease_l489_489322

theorem carla_chickens_disease :
  ∃ (P : ℕ), 
    let died := P * 400 / 100 in
    let bought := 10 * died in
    P * 400 / 100 ≤ 400 ∧ 
    400 - died + bought = 1840 ∧
    P = 40 :=
by
  sorry

end carla_chickens_disease_l489_489322


namespace integer_part_sum_l489_489635

theorem integer_part_sum :
  (⌊ (2010 / 1000) + (1219 / 100) + (27 / 10) ⌋ : ℤ) = 16 :=
by
  sorry

end integer_part_sum_l489_489635


namespace right_triangles_product_hypotenuses_square_l489_489993

/-- 
Given two right triangles T₁ and T₂ with areas 2 and 8 respectively. 
The hypotenuse of T₁ is congruent to one leg of T₂.
The shorter leg of T₁ is congruent to the hypotenuse of T₂.
Prove that the square of the product of the lengths of their hypotenuses is 4624.
-/
theorem right_triangles_product_hypotenuses_square :
  ∃ x y z u : ℝ, 
    (1 / 2) * x * y = 2 ∧
    (1 / 2) * y * u = 8 ∧
    x^2 + y^2 = z^2 ∧
    y^2 + (16 / y)^2 = z^2 ∧ 
    (z^2)^2 = 4624 := 
sorry

end right_triangles_product_hypotenuses_square_l489_489993


namespace geometric_progression_product_l489_489586

theorem geometric_progression_product (r : ℝ) (b_1 : ℝ) (h1 : b_1 > 0) (h2 : r > 0)
  (sum_squares : (∑ k in finset.range(2013) + 8, (b_1 * r ^ (k-1)) ^ 2) = 4)
  (sum_reciprocals : (∑ k in finset.range(2013) + 8, 1 / ((b_1 * r ^ (k-1)) ^ 2)) = 1)
  : (finset.prod (finset.range(2013) + 8) (λ k, (b_1 * r ^ (k-1)) ^ 2)) = 2 ^ 2013 :=
begin
  sorry
end

end geometric_progression_product_l489_489586


namespace count_pairs_l489_489369

open Set

noncomputable def median (s : Finset ℕ) : ℕ := (s.sort (· < ·)).get ⟨s.card / 2, by sorry⟩

theorem count_pairs :
  ∃ (A B : Finset ℕ),
    A ∩ B = ∅ ∧
    A ∪ B = Finset.range 50 ∧
    A.card = 25 ∧
    B.card = 25 ∧
    median B = median A + 1 ∧
    ∃ n : ℕ, n = (Nat.choose 24 12) ^ 2 := 
  sorry

end count_pairs_l489_489369


namespace sin2alpha_plus_cosalpha_l489_489818

theorem sin2alpha_plus_cosalpha (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) + Real.cos α = (4 + Real.sqrt 5) / 5 :=
by
  sorry

end sin2alpha_plus_cosalpha_l489_489818


namespace problem_l489_489871

theorem problem (n : ℤ) (k : ℤ) (hk : k = 1000) (hn_ge : n ≥ 2^k) (hn_lt : n < 2^(k + 1)) :
  n - ((∑ i in Finset.range k.toNat, ⌊(n - 2^i) / 2^(i + 1)⌋ : ℤ)) = n - ((∑ i in Finset.range k.toNat, ⌊(n / 2^(i + 1))⌋ : ℤ)) + k :=
sorry

end problem_l489_489871


namespace arrangements_A_B_l489_489631

theorem arrangements_A_B (front_row back_row : Finset ℕ) 
  (front_exclusions : Finset ℕ) 
  (cannot_sit_next_to : ℕ → ℕ → Prop)
  (h_count_front : front_row.card = 11) 
  (h_count_back : back_row.card = 12) 
  (h_middle_exclusions : front_exclusions.card = 3) 
  (h_no_middle_seats : ∀ x ∈ front_exclusions, x ∈ front_row) 
  (h_not_next_to : ∀ x y, cannot_sit_next_to x y ↔ abs (x - y) ≠ 1) :
  ∃ arrangements : ℕ, arrangements = 346 := by
  sorry

end arrangements_A_B_l489_489631


namespace trigonometric_identity_l489_489999

theorem trigonometric_identity (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (Real.cos (α / 2) ^ 2) = 4 * Real.sin α :=
by
  sorry

end trigonometric_identity_l489_489999


namespace polygon_sides_l489_489622

theorem polygon_sides (n : ℕ) :
  let interior_sum := (n - 2) * 180 
  let exterior_sum := 360
  interior_sum = 3 * exterior_sum - 180 → n = 7 :=
by
  sorry

end polygon_sides_l489_489622


namespace exists_fixed_point_Q_l489_489068

noncomputable def Q : ℝ × ℝ := (0, 1 / 18)

theorem exists_fixed_point_Q (P A B C : ℝ × ℝ) 
  (h_triangle : ∀ ⦃x y : ℝ⦄, y = x^2 - 71 / 36)
  (h_equilateral :  ∑ v in [A, B, C], v = (0, 0) ) :
  ∃ Q : ℝ × ℝ, Q = (0, 1 / 18) ∧ ∀ (P : ℝ × ℝ), (P = (A + B + C) / 3) → dist P Q = abs (prod.snd P) :=
begin
  sorry
end

end exists_fixed_point_Q_l489_489068


namespace max_square_le_sum_of_squares_l489_489191

theorem max_square_le_sum_of_squares
  (n : ℕ) 
  (a : ℕ → ℝ)
  (hcond : ∑ i in Finset.range n, a i = 0) :
  max (Finset.range n).image (λ k, (a k)^2) ≤ (n / 3 : ℝ) * ∑ i in Finset.range (n - 1), (a i - a (i + 1))^2 :=
by sorry

end max_square_le_sum_of_squares_l489_489191


namespace hexacontagon_triangle_count_l489_489431

-- Define the problem conditions in Lean
def hexacontagon_vertices : ℕ := 60

def num_triangles_without_consec (n : ℕ) : ℕ :=
  if 3 ≤ n then (nat.choose n 3) - n else 0

-- Define the theorem to prove the problem statement
theorem hexacontagon_triangle_count : num_triangles_without_consec hexacontagon_vertices = 34160 :=
by
  sorry

end hexacontagon_triangle_count_l489_489431


namespace prove_dollar_op_l489_489020

variable {a b x y : ℝ}

def dollar_op (a b : ℝ) : ℝ := (a - b) ^ 2

theorem prove_dollar_op :
  dollar_op (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2 * x^2 * y^2 + y^4) := by
  sorry

end prove_dollar_op_l489_489020


namespace line_always_intersects_circle_shortest_chord_line_equation_l489_489038

open Real

noncomputable def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x - 6 * y + 9 = 0

noncomputable def line_eqn (m x y : ℝ) : Prop := 2 * m * x - 3 * m * y + x - y - 1 = 0

theorem line_always_intersects_circle (m : ℝ) : 
  ∀ (x y : ℝ), circle_eqn x y → line_eqn m x y → True := 
by
  sorry

theorem shortest_chord_line_equation : 
  ∃ (m x y : ℝ), line_eqn m x y ∧ (∀ x y, line_eqn m x y → x - y - 1 = 0) :=
by
  sorry

end line_always_intersects_circle_shortest_chord_line_equation_l489_489038


namespace yanna_change_l489_489650

theorem yanna_change :
  let shirt_cost := 5
  let sandal_cost := 3
  let num_shirts := 10
  let num_sandals := 3
  let given_amount := 100
  (given_amount - (num_shirts * shirt_cost + num_sandals * sandal_cost)) = 41 :=
by
  sorry

end yanna_change_l489_489650


namespace contribution_per_student_l489_489894

theorem contribution_per_student (total_contribution : ℝ) (class_funds : ℝ) (num_students : ℕ) 
(h1 : total_contribution = 90) (h2 : class_funds = 14) (h3 : num_students = 19) : 
  (total_contribution - class_funds) / num_students = 4 :=
by
  sorry

end contribution_per_student_l489_489894


namespace num_points_P_l489_489215

noncomputable def ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 9) = 1
noncomputable def line (x y : ℝ) := (x / 4) + (y / 3) = 1

theorem num_points_P : ∃ P_count : ℕ, P_count = 2 ∧
  (∀ P : ℝ × ℝ, ellipse P.1 P.2 → 
  let A := (a₁, a₂), B := (b₁, b₂) in
  A ∈ { (x,y) | line x y } ∧ B ∈ { (x,y) | line x y } →
  abs ((1 / 2) * (P.1 * (a₂ - b₂) + a₁ * (b₂ - P.2) + b₁ * (P.2 - a₂))) = 3)
  :=
sorry

end num_points_P_l489_489215


namespace roots_real_and_distinct_l489_489952

-- Definitions from the conditions
def P1 (x : ℝ) : ℝ := x^2 - 2
def P (j : ℕ) (x : ℝ) : ℝ :=
if j = 1 then P1 x else P1 (P (j-1) x)

theorem roots_real_and_distinct (n : ℕ) : ∀ x y : ℝ,
  (P n x = x → x ∈ set.Icc (-2 : ℝ) 2) ∧
  (P n y = y → x ≠ y → x ≠ y) :=
sorry

end roots_real_and_distinct_l489_489952


namespace twelve_sided_region_area_l489_489530

-- Non-computable because we are dealing with real numbers and transcendental functions
noncomputable def A := (1, 0)
noncomputable def B := (1/2, (Real.sqrt 3)/2)
noncomputable def C := (-1/2, (Real.sqrt 3)/2)
noncomputable def D := (-1, 0)
noncomputable def E := (-1/2, -(Real.sqrt 3)/2)
noncomputable def F := (1/2, -(Real.sqrt 3)/2)

-- The theorem statement
theorem twelve_sided_region_area :
  let area := 1/2 * abs (
     A.1 * D.2 + D.1 * F.2 + F.1 * B.2 + B.1 * E.2 + E.1 * C.2 + C.1 * A.2
     - (A.2 * D.1 + D.2 * F.1 + F.2 * B.1 + B.2 * E.1 + E.2 * C.1 + C.2 * A.1)
   )
  in area = 5 * (Real.sqrt 3) / 6 := 
sorry -- Proof to be completed


end twelve_sided_region_area_l489_489530


namespace value_of_a_is_2_l489_489056

def point_symmetric_x_axis (a b : ℝ) : Prop :=
  (2 * a + b = 1 - 2 * b) ∧ (a - 2 * b = -(-2 * a - b - 1))

theorem value_of_a_is_2 (a b : ℝ) (h : point_symmetric_x_axis a b) : a = 2 :=
by sorry

end value_of_a_is_2_l489_489056


namespace son_age_is_9_l489_489448

-- Definitions for the conditions in the problem
def son_age (S F : ℕ) : Prop := S = (1 / 4 : ℝ) * F - 1
def father_age (S F : ℕ) : Prop := F = 5 * S - 5

-- Main statement of the equivalent problem
theorem son_age_is_9 : ∃ S F : ℕ, son_age S F ∧ father_age S F ∧ S = 9 :=
by
  -- We will leave the proof as an exercise
  sorry

end son_age_is_9_l489_489448


namespace eqidistant_point_on_x_axis_l489_489640

theorem eqidistant_point_on_x_axis (x : ℝ) : 
    (dist (x, 0) (-3, 0) = dist (x, 0) (2, 5)) → 
    x = 2 := by
  sorry

end eqidistant_point_on_x_axis_l489_489640


namespace find_m_l489_489059

open Set Real

noncomputable def setA : Set ℝ := {x | x < 2}
noncomputable def setB : Set ℝ := {x | x > 4}
noncomputable def setC (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ m - 1}

theorem find_m (m : ℝ) : setC m ⊆ (setA ∪ setB) → m < 3 :=
by
  sorry

end find_m_l489_489059


namespace polygon_sides_l489_489587

theorem polygon_sides (n : ℕ) :
  (n - 2) * 180 = 3 * 360 - 180 → n = 5 := by
  intro h
  sorry

end polygon_sides_l489_489587


namespace num_three_digit_sums7_l489_489002

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l489_489002


namespace compute_expression_l489_489512

theorem compute_expression (w : ℂ) (hw : w = Complex.exp (Complex.I * (6 * Real.pi / 11))) (hwp : w^11 = 1) :
  (w / (1 + w^3) + w^2 / (1 + w^6) + w^3 / (1 + w^9) = -2) :=
sorry

end compute_expression_l489_489512


namespace smallest_four_digit_number_correct_l489_489643

/-- The smallest positive four-digit number divisible by 9 which has three odd and one even digit. -/
noncomputable def smallest_four_digit_divisible_by_9_with_odd_even_digits : ℕ :=
  1215

theorem smallest_four_digit_number_correct :
  ∃ n: ℕ, (n = smallest_four_digit_divisible_by_9_with_odd_even_digits) ∧
           (n >= 1000) ∧ (n < 10000) ∧ 
           (n % 9 = 0) ∧
           ((n.digits 10).count (λ d, d % 2 = 1) = 3) ∧
           ((n.digits 10).count (λ d, d % 2 = 0) = 1) := sorry

end smallest_four_digit_number_correct_l489_489643


namespace modulus_of_z_l489_489441

theorem modulus_of_z (z : ℂ) (h : 3 * z ^ 6 + 2 * complex.I * z ^ 5 - 2 * z - 3 * complex.I = 0) : |z| = 1 := sorry

end modulus_of_z_l489_489441


namespace problem_geometric_sequence_l489_489050

variable {α : Type*} [LinearOrderedField α]

noncomputable def geom_sequence_5_8 (a : α) (h : a + 8 * a = 2) : α :=
  (a * 2^4 + a * 2^7)

theorem problem_geometric_sequence : ∃ (a : α), (a + 8 * a = 2) ∧ geom_sequence_5_8 a (sorry) = 32 := 
by sorry

end problem_geometric_sequence_l489_489050


namespace point_not_in_region_l489_489217

theorem point_not_in_region (m : ℝ) :
  ¬ (∃ x y, x = 1 ∧ y = 1 ∧ x - (m^2 - 2 * m + 4) * y + 6 > 0) ↔ m ∈ (-∞, -1] ∪ [3, ∞) :=
by
  sorry

end point_not_in_region_l489_489217


namespace three_digit_numbers_sum_seven_l489_489014

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489014


namespace sin_add_polynomial_l489_489166

-- Define basic trigonometric assumptions and the polynomial to be proven
theorem sin_add_polynomial (α β : ℝ) :
  let a := Real.sin α,
      b := Real.sin β,
      c := Real.sin (α + β) in
  ∃ p : Polynomial ℝ, -- p is the polynomial with integer coefficients
    p.eval c = 0 ∧    -- c satisfies the polynomial equation
    p.coeffs.all (λ coef, ∃ m : ℤ, (coef : ℝ) = ↑m) -- coefficients are integers
    ∧ (
      (a = 0 ∨ a = 1 ∨ a = -1) ∨
      (b = 0 ∨ b = 1 ∨ b = -1) ∨
      (a = b) ∨ (a = -b) → c = a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2)).
Proof
  sorry -- the proof goes here

end sin_add_polynomial_l489_489166


namespace number_of_integer_solutions_l489_489803

theorem number_of_integer_solutions (x : ℤ) : (x^2 < 10 * x) → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 :=
begin
  assume h,
  sorry
end

#eval List.length [1, 2, 3, 4, 5, 6, 7, 8, 9] -- 9

end number_of_integer_solutions_l489_489803


namespace extreme_value_sum_l489_489874

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  4 * x ^ 3 - a * x ^ 2 - 2 * b * x

theorem extreme_value_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : has_deriv_at (f a b) 0 1) : a + b = 6 :=
by sorry

end extreme_value_sum_l489_489874


namespace problem_statement_l489_489327

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 8

theorem problem_statement : 3 * g 2 + 4 * g (-2) = 152 := by
  sorry

end problem_statement_l489_489327


namespace integral_problem_five_digit_even_numbers_floor_function_pattern_range_of_m_l489_489272

open Real

-- Problem 1
theorem integral_problem : ∫ x in 0..1, (√(1 - x^2) + exp x) = (π/4) + exp 1 - 1 := 
by 
  sorry

-- Problem 2
theorem five_digit_even_numbers : 
    ∃! n, (∃ digits : Finset (Fin 5), (digits.card = 5 
    ∧ (digits.sum (λ d, ite (d.val % 2 = 0) 1 0) - 1) = n 
    ∧ (digits.prod (λ d, d.val) % 2 = 0)) → n = 60 :=
by 
  sorry

-- Problem 3
theorem floor_function_pattern (n : ℕ) : 
    ∑ k in Finset.range (2 * n + 1), nat.floor (real.sqrt (n^2 + k)) = n * (2 * n + 1) :=
by 
  sorry

-- Problem 4
theorem range_of_m {f : ℝ → ℝ} (m : ℝ) 
(hf : ∀ x, f x = (x^3 - 2 * real.exp 1 * x^2 + m * x - log x) / x) :
    ∃ x, f x = 0 ↔ m ≤ exp 1^2 + 1 / exp 1 := 
by 
  sorry

end integral_problem_five_digit_even_numbers_floor_function_pattern_range_of_m_l489_489272


namespace intersection_A_B_l489_489837

def A : Set ℕ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | x^2 ≤ 3}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end intersection_A_B_l489_489837


namespace problem1_problem2_l489_489482

section Problem1

-- Given conditions
variable (α : Real) (hcosα_ne_0 : cos α ≠ 0)

-- Definition of the quadratic function's vertex
def vertex_x := Real.sec α
def vertex_y := Real.tan α

-- The proof problem
theorem problem1 : vertex_x α hcosα_ne_0 ^ 2 - vertex_y α hcosα_ne_0 ^ 2 = 1 := sorry

end Problem1

section Problem2

-- Given conditions
variable (a : Real)
variable (ρ θ : Real) -- variables for polar coordinates

-- Definition of the line segment and location of point P
def line_segment_length := 2 * a
def P_ρ := ρ
def P_θ := θ

-- The proof problem
theorem problem2 : P_ρ * cos P_θ * sin P_θ = a → ρ = a * sin (2 * θ) := sorry

end Problem2

end problem1_problem2_l489_489482


namespace special_collection_books_l489_489691

theorem special_collection_books (loaned_books : ℕ) (returned_percentage : ℝ) (end_of_month_books : ℕ)
    (H1 : loaned_books = 160)
    (H2 : returned_percentage = 0.65)
    (H3 : end_of_month_books = 244) :
    let books_returned := returned_percentage * loaned_books
    let books_not_returned := loaned_books - books_returned
    let original_books := end_of_month_books + books_not_returned
    original_books = 300 :=
by
  sorry

end special_collection_books_l489_489691


namespace f_at_2_l489_489080

def f (x : ℝ) : ℝ :=
if x ≤ 1 then x + 1 else -x + 3

theorem f_at_2 : f 2 = 1 :=
by
  sorry

end f_at_2_l489_489080


namespace car_travel_distance_l489_489281

theorem car_travel_distance (v d : ℕ) 
  (h1 : d = v * 7)
  (h2 : d = (v + 12) * 5) : 
  d = 210 := by 
  sorry

end car_travel_distance_l489_489281


namespace hatching_probability_calculation_fry_count_calculation_l489_489290

noncomputable def hatching_probability : ℚ := 8513 / 10000

theorem hatching_probability_calculation : hatching_probability = 0.8513 := 
by {
  have h: hatching_probability = 8513 / 10000 := rfl,
  norm_num at h,
  exact h,
}

noncomputable def expected_fry (eggs : ℚ) : ℚ := eggs * hatching_probability

theorem fry_count_calculation : expected_fry 30000 = 25539 := 
by {
  have h: expected_fry 30000 = 30000 * (8513 / 10000) := rfl,
  norm_num at h,
  exact h,
}

#eval hatching_probability  -- Should output 0.8513
#eval expected_fry 30000    -- Should output 25539

end hatching_probability_calculation_fry_count_calculation_l489_489290


namespace marksman_maximum_hits_l489_489690

-- Define the problem conditions and the conclusion
theorem marksman_maximum_hits (T : Type) [fintype T]
  (hits : T → ℕ) (adjacent : T → T → Prop)
  (shots : ℕ)
  (H1 : ∀ t : T, hits t ≤ 5)
  (H2 : ∀ t1 t2 : T, adjacent t1 t2 → hits t1 + hits t2 ≤ 10)
  (H3 : ∃ t : T, shots = (5 - hits t) * 25):
  ∑ t : T, if hits t = 5 then 1 else 0 ≤ 25 := sorry

end marksman_maximum_hits_l489_489690


namespace subset_exists_l489_489421

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 3, x^2}
def B (x : ℝ) : Set ℝ := {x + 2, 1}

-- Statement of the theorem
theorem subset_exists (x : ℝ) : B 2 ⊆ A 2 :=
by
  sorry

end subset_exists_l489_489421


namespace find_theta_l489_489565

def equilateral_triangle_angle : ℝ := 60
def square_angle : ℝ := 90
def pentagon_angle : ℝ := 108
def total_round_angle : ℝ := 360

theorem find_theta (θ : ℝ)
  (h_eq_tri : equilateral_triangle_angle = 60)
  (h_squ : square_angle = 90)
  (h_pen : pentagon_angle = 108)
  (h_round : total_round_angle = 360) :
  θ = total_round_angle - (equilateral_triangle_angle + square_angle + pentagon_angle) :=
sorry

end find_theta_l489_489565


namespace arccos_neg_one_eq_pi_proof_l489_489744

noncomputable def arccos_neg_one_eq_pi : Prop :=
  arccos (-1) = π

theorem arccos_neg_one_eq_pi_proof : arccos_neg_one_eq_pi := by
  sorry

end arccos_neg_one_eq_pi_proof_l489_489744


namespace circle_area_l489_489984

def point := ℝ × ℝ

def A : point := (5, 16)
def B : point := (13, 14)

def is_on_circle (p : point) (center : point) (radius : ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

noncomputable def point_on_x_axis : point := (5.25, 0)

def are_tangents_intersect_on_x_axis (A B : point) (intersect_point : point) : Prop :=
  intersect_point.2 = 0

def calculate_circle_area (radius : ℝ) : ℝ := π * radius^2

theorem circle_area :
  ∃ (center : point) (radius : ℝ),
    is_on_circle A center radius ∧
    is_on_circle B center radius ∧
    are_tangents_intersect_on_x_axis A B point_on_x_axis ∧
    calculate_circle_area radius = (1024.25 * π / 4) :=
by
  sorry

end circle_area_l489_489984


namespace random_chord_intersects_C1_l489_489639

-- Define the circles
structure Circle :=
  (radius : ℝ)

def C1 : Circle := ⟨1⟩
def C2 : Circle := ⟨2⟩

-- Assume we have a method to choose a random chord
-- Here we simplify and state the probability result directly

theorem random_chord_intersects_C1 :
  (∃ (C1 C2 : Circle), C1.radius = 1 ∧ C2.radius = 2) →
  (P(\text{random chord of C2 intersects C1}) = 1 / 2) :=
by
  sorry

end random_chord_intersects_C1_l489_489639


namespace vertex_of_parabola_l489_489206

theorem vertex_of_parabola : ∀ x y : ℝ, y = 2 * (x - 1) ^ 2 + 2 → (1, 2) = (1, 2) :=
by
  sorry

end vertex_of_parabola_l489_489206


namespace parallel_lines_slope_equality_l489_489882

theorem parallel_lines_slope_equality {a : ℝ} 
  (h1 : ∀ x y : ℝ, ax + 2*y = 0) 
  (h2 : ∀ x y : ℝ, x + y = 1) 
  (h3 : ∀ m1 m2 : ℝ, m1 = m2) 
   : a = 2 := 
begin
  sorry
end

end parallel_lines_slope_equality_l489_489882


namespace concurrent_midpoint_parallel_concurrent_midpoint_segments_l489_489967

-- Defining the initial setup conditions
variable {A B C P : Point}
variable {A1 B1 C1 A2 B2 C2 M_A M_B M_C : Point}
variable {AP BP CP : Line}
variable [Metric P]
variable [Triangle ABC A B C]

-- Define midpoints A2, B2, C2 of BC, CA, AB respectively
def midpoint_BC := midpoint A2 B C
def midpoint_CA := midpoint B2 C A
def midpoint_AB := midpoint C2 A B

-- Define midpoints M_A, M_B, M_C of segments AA1, BB1, CC1 respectively
def midpoint_AA1 := midpoint M_A A A1
def midpoint_BB1 := midpoint M_B B B1
def midpoint_CC1 := midpoint M_C C C1

-- Theorem statement for part (a)
theorem concurrent_midpoint_parallel (h1 : Line (A2 . A2 P)) (h2 : Line (B2 . B2 P)) (h3 : Line (C2 . C2 P)) :
  (∃ P, concurrent (Line A2 P) (Line B2 P) (Line C2 P)) :=
sorry

-- Theorem statement for part (b)
theorem concurrent_midpoint_segments (h1 : Line (A2 M_A)) (h2 : Line (B2 M_B)) (h3 : Line (C2 M_C)) :
  (∃ P, concurrent (Line A2 M_A) (Line B2 M_B) (Line C2 M_C)) :=
sorry

end concurrent_midpoint_parallel_concurrent_midpoint_segments_l489_489967


namespace barbed_wire_total_cost_l489_489702

open Real

noncomputable def square_field_area := 3136
noncomputable def gate_width := 1
noncomputable def section_length := 3
noncomputable def wire_length1 := 5
noncomputable def wire_length2 := 7
noncomputable def cost_per_meter1 := 2.25
noncomputable def cost_per_meter2 := 1.75
noncomputable def number_of_gates := 2

def total_cost_needed : Real :=
  let side_length := sqrt square_field_area
  let perimeter := 4 * side_length
  let adjusted_perimeter := perimeter - number_of_gates * gate_width
  let number_of_sections := adjusted_perimeter / section_length
  let number_of_wire_length1_sections := number_of_sections / 2
  let number_of_wire_length2_sections := number_of_sections / 2
  let total_length_wire1 := number_of_wire_length1_sections * wire_length1
  let total_length_wire2 := number_of_wire_length2_sections * wire_length2
  let total_cost1 := total_length_wire1 * cost_per_meter1
  let total_cost2 := total_length_wire2 * cost_per_meter2
  total_cost1 + total_cost2

theorem barbed_wire_total_cost : total_cost_needed = 869.50 := by
  sorry

end barbed_wire_total_cost_l489_489702


namespace joshua_finishes_later_than_malcolm_l489_489968

variable (m_speed flat j_speed flat2 length last_slowdown_m last_slowdown_j : ℕ)
variable (initial_distance uphill_distance : ℕ)

def race_time (speed flat slowdown : ℕ) (initial_distance uphill_distance : ℕ) : ℕ :=
  (speed flat * initial_distance) + ((speed flat + slowdown) * uphill_distance)

theorem joshua_finishes_later_than_malcolm :
  (let malcolm_time := race_time 4 2 10 5;
       joshua_time := race_time 6 3 10 5 in
     joshua_time - malcolm_time = 35) :=
by
  sorry

end joshua_finishes_later_than_malcolm_l489_489968


namespace part_I_part_II_l489_489850

noncomputable def f (a x : ℝ) := a * exp x - x - 1

-- Part I: Prove that ∀ x ∈ (0, +∞), a ≥ 1 → f(a, x) > 0
theorem part_I (a : ℝ) (h : a ≥ 1) : ∀ x : ℝ, 0 < x → f a x > 0 := by
  sorry

-- Part II: Prove that ∀ x ∈ (0, +∞), ln ((exp x - 1) / x) > x / 2
theorem part_II : ∀ x : ℝ, 0 < x → log ((exp x - 1) / x) > x / 2 := by
  sorry

end part_I_part_II_l489_489850


namespace correct_statement_is_A_l489_489648

def statement_A : Prop :=
  ∀ (Xiaoming : Type) (crossroads : Type) (traffic_lights : Type), random_event Xiaoming crossroads traffic_lights

def statement_B : Prop :=
  ∀ (E : Type), certain_event E → probability E = 1

def statement_C : Prop :=
  ∀ (throws : ℕ), fair_dice six_sided throws → frequency (outcome throws 1) = frequency (outcome throws 6)

def statement_D : Prop :=
  ∀ (students : ℕ) (fail_rate : students → ℝ), random_selection students 2 → 
  ∃ (x y : students), (x fails ∨ y fails) → fail_rate = 0.5

theorem correct_statement_is_A : statement_A ∧ ¬statement_D :=
by
  sorry

end correct_statement_is_A_l489_489648


namespace subject_difference_l489_489525

-- Define the problem in terms of conditions and question
theorem subject_difference (C R M : ℕ) (hC : C = 10) (hR : R = C + 4) (hM : M + R + C = 41) : M - R = 3 :=
by
  -- Lean expects a proof here, we skip it with sorry
  sorry

end subject_difference_l489_489525


namespace rounding_to_one_decimal_place_l489_489670

def number_to_round : Float := 5.049

def rounded_value : Float := 5.0

theorem rounding_to_one_decimal_place :
  (Float.round (number_to_round * 10) / 10) = rounded_value :=
by
  sorry

end rounding_to_one_decimal_place_l489_489670


namespace major_axis_range_l489_489078

theorem major_axis_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∀ x M N : ℝ, (x + (1 - x)) = 1 → x * (1 - x) = 0) 
  (e : ℝ) (h4 : (Real.sqrt 3 / 3) ≤ e ∧ e ≤ (Real.sqrt 2 / 2)) :
  ∃ a : ℝ, 2 * (Real.sqrt 5) ≤ 2 * a ∧ 2 * a ≤ 2 * (Real.sqrt 6) := 
sorry

end major_axis_range_l489_489078


namespace sequence_increasing_l489_489062

theorem sequence_increasing (a : ℕ → ℤ) (h : ∀ n, a (n + 1) - a n = 3) : 
  ∀ n, a n < a (n + 1) :=
begin
  sorry
end

end sequence_increasing_l489_489062


namespace increase_in_output_with_assistant_l489_489922

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end increase_in_output_with_assistant_l489_489922


namespace larger_integer_value_l489_489219

theorem larger_integer_value (a b : ℕ) (h1 : a/b = 3/2) (h2 : a * b = 180) :
  ∃ x : ℝ, a = 3 * real.sqrt 30 ∧ b = 2 * real.sqrt 30 := 
begin
  sorry
end

end larger_integer_value_l489_489219


namespace minimum_value_sum_of_distances_l489_489393

/-- 
  Given point A(4, 4), if the focus of the parabola y^2 = 2px 
  coincides with the right focus of the ellipse x^2/a^2 + y^2/b^2 = 1, 
  and there is a point M on the parabola whose projection on the y-axis 
  is point N, find the minimum value of |MA| + |MN|.
-/
theorem minimum_value_sum_of_distances
  (A : ℝ × ℝ) (hA : A = (4, 4))
  (a b : ℝ) (hb : b < a)
  (focus_coincide : (a + real.sqrt(a^2 - b^2), 0) = (1 / 2 * p, 0))
  (p : ℝ) (hy : 0 < y)
  (M : ℝ × ℝ) (hM1 : M = (y^2 / (2 * p), y))
  (hM2 : y^2 = 2 * p * (M.1))
  (N : ℝ × ℝ) (hN : N = (0, y)) : 
  |M - A| + |N - A| = 4 :=
sorry

end minimum_value_sum_of_distances_l489_489393


namespace percent_increase_output_l489_489931

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l489_489931


namespace game_cost_l489_489524

theorem game_cost (initial_money spent_money games : ℕ) (h1 : initial_money = 69) (h2 : spent_money = 24) (h3 : games = 9) :
  (initial_money - spent_money) / games = 5 :=
by
  -- Given equations and simple arithmetic
  have h4 : initial_money - spent_money = 45 := by linarith [h1, h2]
  have h5 : games = 9 := h3
  have h6 : 45 / games = 5 := by norm_num
  rw [h4, h5, h6]
  sorry

end game_cost_l489_489524


namespace range_of_m_l489_489819

def f (x : ℝ) : ℝ := x ^ 3 - 3 * x

def tangent_points (m : ℝ) (x₀ : ℝ) : Prop := 
  2 * x₀ ^ 3 - 3 * x₀ ^ 2 + m + 3 = 0

theorem range_of_m (m : ℝ) :
  (∀ x₀, tangent_points m x₀) ∧ m ≠ -2 → (-3 < m ∧ m < -2) :=
sorry

end range_of_m_l489_489819


namespace remainder_of_number_of_minimally_intersecting_triples_l489_489505

noncomputable def number_of_minimally_intersecting_triples : Nat :=
  let n := (8 * 7 * 6) * (4 ^ 5)
  n % 1000

theorem remainder_of_number_of_minimally_intersecting_triples :
  number_of_minimally_intersecting_triples = 64 := by
  sorry

end remainder_of_number_of_minimally_intersecting_triples_l489_489505


namespace find_angle_C_find_max_area_l489_489379

-- Defining the conditions of the problem
variables {A B C : ℝ} (a b c : ℝ)
hypothesis (h : a * Real.sin A - c * Real.sin C = (a - b) * Real.sin B)

-- Theorem statement part 1: Prove that ∠C = π / 3 given the conditions
theorem find_angle_C (h : a * Real.sin A - c * Real.sin C = (a - b) * Real.sin B) :
  C = π / 3 :=
sorry

-- Theorem statement part 2: Prove that the maximum area S is 3√3/2 when c = √6
theorem find_max_area (h : a * Real.sin A - c * Real.sin C = (a - b) * Real.sin B)
  (c_eq : c = Real.sqrt 6) :
  ∃ S : ℝ, S = 3 * Real.sqrt 3 / 2 :=
sorry

end find_angle_C_find_max_area_l489_489379


namespace ursula_hourly_wage_l489_489236

def annual_salary : ℝ := 16320
def hours_per_day : ℝ := 8
def days_per_month : ℝ := 20
def months_per_year : ℝ := 12

theorem ursula_hourly_wage : 
  (annual_salary / months_per_year) / (hours_per_day * days_per_month) = 8.50 := by 
  sorry

end ursula_hourly_wage_l489_489236


namespace find_fourth_score_l489_489814

theorem find_fourth_score
  (a b c : ℕ) (d : ℕ)
  (ha : a = 70) (hb : b = 80) (hc : c = 90)
  (average_eq : (a + b + c + d) / 4 = 70) :
  d = 40 := 
sorry

end find_fourth_score_l489_489814


namespace num_pos_divisors_7560_multiple_5_l489_489866

theorem num_pos_divisors_7560_multiple_5 :
  let n := 7560 in
  let factorization := (2^3 * 3^3 * 5 * 7) in
  (∃ a b c : ℕ,
      0 ≤ a ∧ a ≤ 3 ∧
      0 ≤ b ∧ b ≤ 3 ∧
      0 ≤ c ∧ c ≤ 1 ∧
      n = 2^a * 3^b * 5 * 7^c) →
  (count (λ d, d ∣ n ∧ 5 ∣ d) (finset.range (n + 1))) = 32 :=
by
  intros
  sorry

end num_pos_divisors_7560_multiple_5_l489_489866


namespace sum_f_eq_201_l489_489082

noncomputable def f (x : ℚ) : ℚ :=
  if x = 1 then 1 else (x + 2) / (x - 1)

theorem sum_f_eq_201 : 
  ∑ k in (Finset.range 201).map (Function.Embedding.coerce ((λ k : ℕ, (k + 1) / 101) : ℕ → ℚ)), f k = 201 := 
by
  sorry

end sum_f_eq_201_l489_489082


namespace profit_percentage_is_20_l489_489675

noncomputable def selling_price : ℝ := 200
noncomputable def cost_price : ℝ := 166.67
noncomputable def profit : ℝ := selling_price - cost_price

theorem profit_percentage_is_20 :
  (profit / cost_price) * 100 = 20 := by
  sorry

end profit_percentage_is_20_l489_489675


namespace tangent_line_evaluation_l489_489063

theorem tangent_line_evaluation (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h1 : ∀ x, has_deriv_at f (f' x) x)
  (h2 : ∀ y, y = f 5 + f' 5 * (5 - 5))
  (tangent_line_eq : ∀ x y, x + y - 5 = 0) :
  f(5) + f'(5) = -1 := 
sorry

end tangent_line_evaluation_l489_489063


namespace factorize_expression_l489_489774

theorem factorize_expression (a : ℝ) : 
  a^3 - 16 * a = a * (a + 4) * (a - 4) :=
sorry

end factorize_expression_l489_489774


namespace sum_of_digits_corrected_l489_489560

theorem sum_of_digits_corrected (d e : ℕ) :
    (397154 + 518629 = 1061835) →
    (∀ x : ℕ, if x = d then e else x) = 
    (∀ y : ℕ, if y = e then y = 1061835 - x else y) →
    d + e = 1 :=
sorry

end sum_of_digits_corrected_l489_489560


namespace three_digit_numbers_sum_seven_l489_489018

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489018


namespace domain_of_f_2x_minus_3_l489_489520

noncomputable def f (x : ℝ) : ℝ := 2 * x + 1

-- Conditions: f(x) has a domain of [1, 5]
def domain_f : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}

-- Translating our problem
def domain_f_2x_minus_3 : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}

theorem domain_of_f_2x_minus_3 :
  ∀ x, (1 ≤ 2 * x - 3 ∧ 2 * x - 3 ≤ 5) ↔ (2 ≤ x ∧ x ≤ 4) :=
begin
  intro x,
  split,
  { intro h,
    cases h with h1 h2,
    linarith,
  },
  { intro h,
    cases h with h1 h2,
    split; linarith,
  }
end

end domain_of_f_2x_minus_3_l489_489520


namespace fill_time_l489_489718

def old_pump_time : ℝ := 600
def second_pump_time : ℝ := 200
def third_pump_time : ℝ := 400
def fourth_pump_time : ℝ := 300
def leak_time : ℝ := 1200

def old_pump_rate := 1 / old_pump_time
def second_pump_rate := 1 / second_pump_time
def third_pump_rate := 1 / third_pump_time
def fourth_pump_rate := 1 / fourth_pump_time
def leak_rate := 1 / leak_time

def net_rate := old_pump_rate + second_pump_rate + third_pump_rate + fourth_pump_rate - leak_rate

theorem fill_time : (1 / net_rate) = 600 / 7 :=
by
  unfold old_pump_rate second_pump_rate third_pump_rate fourth_pump_rate leak_rate net_rate
  sorry

end fill_time_l489_489718


namespace no_two_primes_sum_to_53_l489_489475

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_two_primes_sum_to_53 :
  ¬ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 53 :=
by
  sorry

end no_two_primes_sum_to_53_l489_489475


namespace set_equality_iff_inter_compl_eq_l489_489423

variables {U : Type*} [decidable_eq U] {A B D : set U}

theorem set_equality_iff_inter_compl_eq (h : A ∪ B = A ∪ D) :
  B ∩ (U \ A) = D ∩ (U \ A) :=
sorry

end set_equality_iff_inter_compl_eq_l489_489423


namespace three_digit_numbers_sum_seven_l489_489007

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489007


namespace blocks_differ_in_three_ways_l489_489295

/-
  Given a set of 96 distinct blocks where each block is one of:
  - 2 materials (plastic, wood),
  - 3 sizes (small, medium, large),
  - 5 colors (blue, green, red, yellow, purple),
  - 4 shapes (circle, hexagon, square, triangle),

  Prove that the number of blocks that differ from the 'plastic medium red circle' in exactly 3 ways is 26.
-/

noncomputable def blocks := 96
def materials := 2
def sizes := 3
def colors := 5
def shapes := 4

theorem blocks_differ_in_three_ways : ∃ (num_diff_blocks : ℕ), num_diff_blocks = 26 ∧ 
  (num_diff_blocks = coefficient (expand ((1 + x) * (1 + 2 * x) * (1 + 4 * x) * (1 + 3 * x))) 3) := 
sorry

end blocks_differ_in_three_ways_l489_489295


namespace monotonic_increasing_interval_l489_489574

noncomputable def f (x : ℝ) := Real.log (5 + 4 * x - x ^ 2)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, -1 < x ∧ x < 2 → 
    is_monotonic_increasing (λ x, Real.log (5 + 4 * x - x ^ 2)) :=
begin
  sorry
end

end monotonic_increasing_interval_l489_489574


namespace banana_no_adjacent_As_l489_489765

open Nat Combinatorics

theorem banana_no_adjacent_As :
  let letters : List Char := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := 6
  let a_count := 3
  ∃ (ways : ℕ), ways = 12 ∧ 
    (∀ l : List Char, (l.length = n ∧ (l.count 'A' = a_count)) → valid_banana_arrangement l ways)
:= by
  sorry

-- Auxiliary definition to check validity of arrangement
def valid_banana_arrangement (l : List Char) (ways : ℕ) : Prop :=
  (∃ n : ℕ, n = 12) ∧
  all_no_adjacent_As l

-- Auxiliary function to check if no 'A's are adjacent
def all_no_adjacent_As (l : List Char) : Prop :=
  ∀ i, i < l.length - 1 -> (l.get! i = 'A' -> l.get! (i + 1) ≠ 'A')

end banana_no_adjacent_As_l489_489765


namespace least_prime_P_with_integer_roots_of_quadratic_l489_489437

theorem least_prime_P_with_integer_roots_of_quadratic :
  ∃ P : ℕ, P.Prime ∧ (∃ m : ℤ,  m^2 = 12 * P + 60) ∧ P = 7 :=
by
  sorry

end least_prime_P_with_integer_roots_of_quadratic_l489_489437


namespace continuous_func_unique_l489_489358

theorem continuous_func_unique (f : ℝ → ℝ) (hf_cont : Continuous f)
  (hf_eqn : ∀ x : ℝ, f x + f (x^2) = 2) :
  ∀ x : ℝ, f x = 1 :=
by
  sorry

end continuous_func_unique_l489_489358


namespace line_through_circle_condition_l489_489567

theorem line_through_circle_condition
  (x y : ℝ) (k : ℝ) (l : ℝ → ℝ → Prop)
  (P : x = -4 ∧ y = 0)
  (circle : ∀ x y, (x + 1)^2 + (y - 2)^2 = 25)
  (dist_AB : ∀ A B, |A - B| = 8) :
  (l = (λ x y, 5 * x + 12 * y + 20 = 0) ∨ l = (λ x y, x + 4 = 0)) := 
  sorry

end line_through_circle_condition_l489_489567


namespace three_digit_numbers_sum_seven_l489_489012

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489012


namespace smallest_n_for_429_digits_in_decimal_l489_489563

theorem smallest_n_for_429_digits_in_decimal (m n : ℕ) (h1 : Nat.Coprime m n) (h2 : m < n) 
  (h3 : ∃ k, (429 * n ≤ 1000 * m) ∧ (1000 * m < 430 * n)) : 
  n = 43 :=
sorry

end smallest_n_for_429_digits_in_decimal_l489_489563


namespace geometric_sequence_property_l489_489388

noncomputable def geometric_sequence (b : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n * r

theorem geometric_sequence_property :
  ∀ (b : ℕ → ℝ) (r : ℝ) (b1 : ℝ),
  geometric_sequence b r →
  b 1 = b1 →
  (∏ n in finset.range 1006, b (2 * n + 1)) / (∏ n in finset.range 1005, b (2 * n + 2)) = b 1006 :=
by
  intros b r b1 h_seq h_b1
  sorry

end geometric_sequence_property_l489_489388


namespace problem_statement_l489_489373

def θ (m v : ℕ) : ℕ := m % v

theorem problem_statement : ((θ 90 33) θ 17) - (θ 99 (θ 33 17)) = 4 :=
by
  sorry

end problem_statement_l489_489373


namespace jessica_age_l489_489497

theorem jessica_age 
  (j g : ℚ)
  (h1 : g = 15 * j) 
  (h2 : g - j = 60) : 
  j = 30 / 7 :=
by
  sorry

end jessica_age_l489_489497


namespace probability_of_D_l489_489278

variable p_A : ℚ := 1/4
variable p_B : ℚ := 1/3
variable p_C : ℚ := 1/6
variable p_D : ℚ

theorem probability_of_D : p_A + p_B + p_C + p_D = 1 → p_D = 1/4 :=
by
  intro h,
  sorry

end probability_of_D_l489_489278


namespace arithmetic_sequence_sum_six_l489_489120

open Nat

noncomputable def sum_first_six_terms (a : ℕ → ℚ) : ℚ :=
  let a1 : ℚ := a 1
  let d : ℚ := a 2 - a1
  3 * (2 * a1 + 5 * d) / 3

theorem arithmetic_sequence_sum_six (a : ℕ → ℚ) (h : a 2 + a 5 = 2 / 3) : sum_first_six_terms a = 2 :=
by
  let a1 : ℚ := a 1
  let d : ℚ := a 2 - a1
  have eq1 : a 5 = a1 + 4 * d := by sorry
  have eq2 : 3 * (2 * a1 + 5 * d) / 3 = (2 : ℚ) := by sorry
  sorry

end arithmetic_sequence_sum_six_l489_489120


namespace number_of_integer_solutions_l489_489805

theorem number_of_integer_solutions (x : ℤ) : (x^2 < 10 * x) → x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 :=
begin
  assume h,
  sorry
end

#eval List.length [1, 2, 3, 4, 5, 6, 7, 8, 9] -- 9

end number_of_integer_solutions_l489_489805


namespace sides_of_polygon_l489_489615

theorem sides_of_polygon (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 :=
by
  sorry

end sides_of_polygon_l489_489615


namespace three_digit_numbers_sum_seven_l489_489017

-- Define the problem in Lean
theorem three_digit_numbers_sum_seven : 
  ∃ (s : Finset (Fin 10 × Fin 10 × Fin 10)), 
  (∀ (a b c : Fin 10), (a, b, c) ∈ s → a ≥ 1 ∧ a + b + c = 7) 
  ∧ s.card = 28 :=
by
  let s := { n | let (a, b, c) := (n / 100, (n / 10) % 10, n % 10) in 1 ≤ a ∧ a + b + c = 7 }.to_finset
  use s
  split
  { intros a b c h, exact h }
  sorry

end three_digit_numbers_sum_seven_l489_489017


namespace lowest_price_l489_489263

theorem lowest_price (cost_per_component shipping_cost_per_unit fixed_costs number_of_components produced_cost total_variable_cost total_cost lowest_price : ℝ)
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 2)
  (h3 : fixed_costs = 16200)
  (h4 : number_of_components = 150)
  (h5 : total_variable_cost = cost_per_component + shipping_cost_per_unit)
  (h6 : produced_cost = total_variable_cost * number_of_components)
  (h7 : total_cost = produced_cost + fixed_costs)
  (h8 : lowest_price = total_cost / number_of_components) :
  lowest_price = 190 :=
  by
  sorry

end lowest_price_l489_489263


namespace cost_per_taco_is_1_50_l489_489300

namespace TacoTruck

def total_beef : ℝ := 100
def beef_per_taco : ℝ := 0.25
def taco_price : ℝ := 2
def profit : ℝ := 200

theorem cost_per_taco_is_1_50 :
  let total_tacos := total_beef / beef_per_taco
  let total_revenue := total_tacos * taco_price
  let total_cost := total_revenue - profit
  total_cost / total_tacos = 1.50 := 
by
  sorry

end TacoTruck

end cost_per_taco_is_1_50_l489_489300


namespace max_knights_no_only_liars_l489_489671

namespace KnightsAndLiars

variables (N : ℕ)

-- Condition: There are 2N students of different heights.
def students_count := 2 * N

-- Condition: Students are lined up in pairs forming two columns.
-- Condition: Specific students' statements based on their positions.

-- Proposition: The maximum number of knights in the school is N.
theorem max_knights (h : 2 * N > 0) : ∃ k ≤ 2 * N, k = N :=
sorry

-- Proposition: It is not possible for only liars to study in the school.
theorem no_only_liars (h : 2 * N > 0) : ∀ (L ≤ 2 * N), L ≠ 2 * N :=
sorry

end KnightsAndLiars

end max_knights_no_only_liars_l489_489671


namespace part_one_part_two_l489_489169

-- Definitions based on the conditions
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 1}
def C (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 - a}

-- Prove intersection A ∩ B = (0, 1)
theorem part_one : A ∩ B = { x | 0 < x ∧ x < 1 } := by
  sorry

-- Prove range of a when A ∪ C = A
theorem part_two (a : ℝ) (h : A ∪ C a = A) : 1 < a := by
  sorry

end part_one_part_two_l489_489169


namespace correct_propositions_l489_489403

-- Define the conditions and propositions
variables {a b c : Type}

-- Conditions and Propositions
def condition1 (a b c : Type) : Prop := (a.parallel b) ∧ (b.parallel c) → (a.parallel c)
def condition2 (a b c : Type) : Prop := (a.perpendicular b) ∧ (b.perpendicular c) → (a.perpendicular c)
def condition3 (a b c : Type) : Prop := (a.perpendicular b) ∧ (b.perpendicular c) → (a.parallel c)
def condition4 (a b c : Type) : Prop := (a.parallel b) ∧ (b.perpendicular c) → (a.perpendicular c)

-- The theorem proving that the correct propositions are (1) and (4)
theorem correct_propositions : 
  (condition1 a b c) ∧ (condition4 a b c) ∧ ¬(condition2 a b c) ∧ ¬(condition3 a b c) :=
by sorry

end correct_propositions_l489_489403


namespace sum_of_coordinates_of_point_B_l489_489534

theorem sum_of_coordinates_of_point_B
  (A : ℝ × ℝ) (hA : A = (0, 0))
  (B : ℝ × ℝ) (hB : ∃ x : ℝ, B = (x, 3))
  (slope_AB : ∃ x : ℝ, (3 - 0)/(x - 0) = 3/4) :
  (∃ x : ℝ, B = (x, 3)) ∧ x + 3 = 7 :=
by
  sorry

end sum_of_coordinates_of_point_B_l489_489534


namespace geometric_sequence_a5_value_l489_489904

theorem geometric_sequence_a5_value :
  ∃ (a : ℕ → ℝ) (r : ℝ), (a 3)^2 - 4 * a 3 + 3 = 0 ∧ 
                         (a 7)^2 - 4 * a 7 + 3 = 0 ∧ 
                         (a 3) * (a 7) = 3 ∧ 
                         (a 3) + (a 7) = 4 ∧ 
                         a 5 = (a 3 * a 7).sqrt :=
sorry

end geometric_sequence_a5_value_l489_489904


namespace arcsin_arccos_solution_l489_489549

theorem arcsin_arccos_solution (x : ℝ) (hx1 : |x| ≤ 1) (hx2 : |2*x| ≤ 1) :
  arcsin x + arcsin (2*x) = arccos x ↔ x = 0 ∨ x = 2 / Real.sqrt 5 ∨ x = - (2 / Real.sqrt 5) := 
sorry

end arcsin_arccos_solution_l489_489549


namespace thirteen_digit_sequences_l489_489638

def good_sequences_of_length (n : ℕ) : ℕ :=
  let A : ℕ → ℕ := sorry    -- Number of sequences ending in 0 or 4
  let B : ℕ → ℕ := sorry    -- Number of sequences ending in 1 or 3
  let C : ℕ → ℕ := sorry    -- Number of sequences ending in 2
  A n + B n + C n

theorem thirteen_digit_sequences : good_sequences_of_length 13 = 3402 :=
by {
  let A : ℕ → ℕ := sorry,
  let B : ℕ → ℕ := sorry,
  let C : ℕ → ℕ := sorry,
  have A_13 := sorry,
  have B_13 := sorry,
  have C_13 := sorry,
  show good_sequences_of_length 13 = 3402, from sorry
}

end thirteen_digit_sequences_l489_489638


namespace total_seats_in_theater_l489_489704

theorem total_seats_in_theater : 
  let R := 20
  let first_row_seats := 28
  let seat_increment := 2
  let sum_seats := (first_row_seats + (first_row_seats + (R - 1) * seat_increment)) * R / 2
  in sum_seats = 940 :=
by
  sorry

end total_seats_in_theater_l489_489704


namespace part_a_part_b_part_c_part_d_part_e_l489_489537

variable (a : ℕ → ℝ)
variable (P Q : ℕ → ℝ)
variable (k : ℕ)

-- Conditions
axiom a_pos : ∀ n, a n > 0
axiom P_rec : ∀ n ≥ 2, P n = a n * P (n - 1) + P (n - 2)
axiom Q_rec : ∀ n ≥ 2, Q n = a n * Q (n - 1) + Q (n - 2)

-- Part (a)
theorem part_a (h : k ≥ 2) : P k * Q (k - 2) - P (k - 2) * Q k = (-1) ^ k * a k := sorry

-- Part (b)
theorem part_b (h : k ≥ 1) : P k / Q k - P (k - 1) / Q (k - 1) = (-1) ^ (k + 1) / (Q k * Q (k - 1)) := sorry

-- Part (c)
theorem part_c (n : ℕ) (h1 : n ≥ 1) (h2 : a 1 > 0) : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → Q k < Q (k + 1) := sorry

-- Part (d)
theorem part_d (n : ℕ) : (P 0 / Q 0) < (P 2 / Q 2) ∧
                        (P 2 / Q 2) < (P 4 / Q 4) ∧
                        ... ∧
                        (Q n) ≤ ... ∧
                        (P 5 / Q 5) < (P 3 / Q 3) ∧
                        (P 3 / Q 3) < (P 1 / Q 1) := sorry
                                
-- Part (e)
theorem part_e (k l : ℕ) : P (2 * k) / Q (2 * k) < P (2 * l + 1) / Q (2 * l + 1) := sorry

end part_a_part_b_part_c_part_d_part_e_l489_489537


namespace Angelina_time_difference_l489_489309

open Real

def time_diff (d₁ s₁ d₂ s₂ : ℝ) : ℝ :=
  (d₁ / s₁) - (d₂ / s₂)

theorem Angelina_time_difference :
  let home_to_grocery_distance := 200
  let grocery_to_gym_distance := 300
  let grocery_to_gym_speed := 2
  let home_to_grocery_speed := grocery_to_gym_speed / 2
  time_diff home_to_grocery_distance home_to_grocery_speed grocery_to_gym_distance grocery_to_gym_speed = 50 := 
by 
  sorry

end Angelina_time_difference_l489_489309


namespace remainder_S_l489_489945

def R' : Set ℕ := {n % 500 | n ∈ range (100)}
def S' : ℕ := (Finset.range 100).sum (λ n, 2 ^ n % 500)

theorem remainder_S'_mod_500 : S' % 500 = 499 :=
by
  -- Proof placeholder
  sorry

end remainder_S_l489_489945


namespace coffee_cost_l489_489965

theorem coffee_cost :
  ∃ y : ℕ, 
  (∃ x : ℕ, 3 * x + 2 * y = 630 ∧ 2 * x + 3 * y = 690) → y = 162 :=
by
  sorry

end coffee_cost_l489_489965


namespace S_10_eq_1991_l489_489507

def sequence_a : ℕ → ℕ
| 0     := 1
| (n+1) := 2 * sequence_a n + n - 1

def S (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), sequence_a k

theorem S_10_eq_1991 : S 10 = 1991 :=
by
  sorry

end S_10_eq_1991_l489_489507


namespace B_power_15_minus_3_B_power_14_l489_489146

def B : Matrix (Fin 2) (Fin 2) ℝ := !!
  [3, 4]
  [0, 2]

theorem B_power_15_minus_3_B_power_14 :
  B^15 - 3 • B^14 = !!
    [0, 4]
    [0, -1] := by
  sorry

end B_power_15_minus_3_B_power_14_l489_489146


namespace problem_sets_classification_l489_489778

theorem problem_sets_classification :
  ∃ (positive negative integer rational : set ℝ),
    positive = {0.3, Real.pi / 3, 22 / 7} ∧
    negative = {-2, -3.14, -0.1212212221...} ∧
    integer = {-2, 0} ∧
    rational = {-2, 0, 0.3, 22 / 7, -0.1212212221...} :=
begin
  sorry
end

end problem_sets_classification_l489_489778


namespace bisection_interval_contains_root_l489_489847

def f (x : ℝ) : ℝ := 2^x + 2*x - 6

theorem bisection_interval_contains_root :
  ∃ a b : ℝ, (1 < a ∧ a < 2 ∧ 2 < b ∧ b < 3) ∧ 
             f(1) < 0 ∧ f(2) > 0 ∧ f(3) > 0 ∧ 
             (forall x, (a < x ∧ x < b) -> f(x) = 0) :=
by
  sorry

end bisection_interval_contains_root_l489_489847


namespace percent_increase_output_per_hour_l489_489918

-- Definitions and conditions
variable (B H : ℝ) -- B is the number of bears per week, H is the number of hours per week

-- Define outputs per hour
def output_per_hour (B H : ℝ) := B / H
def new_output_per_hour (B H : ℝ) := 1.8 * B / (0.9 * H)

-- A mathematical statement to prove the percentage increase of output per hour
theorem percent_increase_output_per_hour (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((new_output_per_hour B H) - (output_per_hour B H)) / (output_per_hour B H) * 100 = 100 :=
by
  sorry

end percent_increase_output_per_hour_l489_489918


namespace ratio_of_investments_l489_489276

theorem ratio_of_investments (I B_profit total_profit : ℝ) (x : ℝ)
  (h1 : B_profit = 4000) (h2 : total_profit = 28000) (h3 : I * (2 * B_profit / 4000 - 1) = total_profit - B_profit) :
  x = 3 :=
by
  sorry

end ratio_of_investments_l489_489276


namespace square_carpet_side_length_l489_489672

theorem square_carpet_side_length (length width : ℝ) (uncovered_area : ℝ) 
  (h1 : length = 10) (h2 : width = 8) (h3 : uncovered_area = 64) : 
  ∃ (s : ℝ), s = 4 :=
by
  let total_area := length * width
  have h_total_area : total_area = 80 := by rw [h1, h2]; norm_num
  let carpet_area := total_area - uncovered_area
  have h_carpet_area : carpet_area = 16 := by rw [h_total_area, h3]; norm_num
  use real.sqrt carpet_area
  have h_s : real.sqrt 16 = 4 := by norm_num
  rw [h_carpet_area, h_s]
  norm_num

end square_carpet_side_length_l489_489672


namespace ice_cream_per_milkshake_l489_489727

variable (oz_milk_per_milkshake : ℕ) (oz_milk_initial : ℕ) (oz_ice_cream_initial : ℕ) (oz_milk_leftover : ℕ)

axiom oz_milk_per_milkshake_eq : oz_milk_per_milkshake = 4
axiom oz_milk_initial_eq : oz_milk_initial = 72
axiom oz_ice_cream_initial_eq : oz_ice_cream_initial = 192
axiom oz_milk_leftover_eq : oz_milk_leftover = 8

theorem ice_cream_per_milkshake : 
  (oz_ice_cream_initial / ((oz_milk_initial - oz_milk_leftover) / oz_milk_per_milkshake)) = 12 :=
by 
  rw [oz_milk_per_milkshake_eq, oz_milk_initial_eq, oz_ice_cream_initial_eq, oz_milk_leftover_eq]
  norm_num

end ice_cream_per_milkshake_l489_489727


namespace amy_created_albums_l489_489714

theorem amy_created_albums (total_photos : ℕ) (photos_per_album : ℕ) 
  (h1 : total_photos = 180)
  (h2 : photos_per_album = 20) : 
  (total_photos / photos_per_album = 9) :=
by
  sorry

end amy_created_albums_l489_489714


namespace determinant_M_l489_489135

open Matrix

-- Defining the set and subsets
def nonempty_subsets (n : ℕ) : List (Set ℕ) :=
  (List.sublists' (List.range n.succ)).tail

-- Defining the matrix M
def matrix_entry (S_i S_j : Set ℕ) : ℕ :=
  if S_i ∩ S_j = ∅ then 0 else 1

def M (n : ℕ) : Matrix (Fin (2^n - 1)) (Fin (2^n - 1)) ℕ :=
  let subsets := nonempty_subsets n
  matrix.ofFun (λ i j => matrix_entry (subsets.get i.val) (subsets.get j.val))

-- The final theorem statement
theorem determinant_M (n : ℕ) : 
  det (M n) = (-1) ^ (if n ≠ 1 then 1 else 0) :=
by
  sorry

end determinant_M_l489_489135


namespace min_value_functions_l489_489258

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 1 / x^2
noncomputable def f_B (x : ℝ) : ℝ := 2 * x + 2 / x
noncomputable def f_C (x : ℝ) : ℝ := (x - 1) / (x + 1)
noncomputable def f_D (x : ℝ) : ℝ := Real.log (Real.sqrt x + 1)

theorem min_value_functions :
  (∃ x : ℝ, ∀ y : ℝ, f_A x ≤ f_A y) ∧
  (∃ x : ℝ, ∀ y : ℝ, f_D x ≤ f_D y) ∧
  ¬ (∃ x : ℝ, ∀ y : ℝ, f_B x ≤ f_B y) ∧
  ¬ (∃ x : ℝ, ∀ y : ℝ, f_C x ≤ f_C y) :=
by
  sorry

end min_value_functions_l489_489258


namespace length_of_chord_EF_l489_489889

-- Assuming basic geometry axioms, definitions, and necessary theorems are available

-- Conditions given in the problem
variables {A B C D G E F : Point} - Points involved
variables {O N P : Circle} - Circles involved
variables (rO rN rP : ℝ) -- Radii of the circles

hypothesis (h1 : rO = 12) -- Radius of circle O
hypothesis (h2 : rN = 20) -- Radius of circle N
hypothesis (h3 : rP = 15) -- Radius of circle P

hypothesis (h4 : line_segment A D) -- Line segment AD
hypothesis (h5 : lies_on B A D) -- Point B lies on line segment AD
hypothesis (h6 : lies_on C A D) -- Point C lies on line segment AD
hypothesis (h7 : diameter O A B) -- AB is the diameter of circle O
hypothesis (h8 : diameter N B C) -- BC is the diameter of circle N
hypothesis (h9 : diameter P C D) -- CD is the diameter of circle P
hypothesis (h10 : enclosed P N) -- Circle P is enclosed within circle N
hypothesis (h11 : tangent_line AG N G) -- Line AG is tangent to circle N at point G
hypothesis (h12 : intersects AG P E F) -- Line AG intersects circle P at points E and F

-- The proof problem
theorem length_of_chord_EF :
  length_chord P E F = 16 * sqrt 6 :=
sorry -- Proof to be completed

end length_of_chord_EF_l489_489889


namespace sequence_formula_l489_489092

theorem sequence_formula (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h : ∀ n : ℕ, S n = 3 * a n + (-1)^n) :
  ∀ n : ℕ, a n = (1/10) * (3/2)^(n-1) - (2/5) * (-1)^n :=
by sorry

end sequence_formula_l489_489092


namespace num_three_digit_sums7_l489_489005

theorem num_three_digit_sums7 : 
  { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 7 }.card = 28 :=
sorry

end num_three_digit_sums7_l489_489005


namespace find_b_l489_489835

noncomputable def circle_center_radius : Prop :=
  let C := (2, 0) -- center
  let r := 2 -- radius
  C.1 = 2 ∧ C.2 = 0 ∧ r = 2

noncomputable def line (b : ℝ) : Prop :=
  ∃ M N : ℝ × ℝ, M ≠ N ∧ 
  (M.2 = M.1 + b) ∧ (N.2 = N.1 + b) -- points on the line are M = (x1, x1 + b) and N = (x2, x2 + b)

noncomputable def perpendicular_condition (M N center: ℝ × ℝ) : Prop :=
  (M.1 - center.1) * (N.1 - center.1) + (M.2 - center.2) * (N.2 - center.2) = 0 -- CM ⟂ CN

theorem find_b (b : ℝ) : 
  circle_center_radius ∧
  (∃ M N, line b ∧ perpendicular_condition M N (2, 0)) →
  b = 0 ∨ b = -4 :=
by {
  -- Proof omitted
  sorry
}

end find_b_l489_489835


namespace angle_DAC_l489_489784

theorem angle_DAC (A B C D : Type) (l1 l2 l3 l4 : Type)
  [point A] [point B] [point C] [point D]
  [line l1] [line l2] [line l3] [line l4]
  (AB : segment A B) (BC : segment B C) (AC : segment A C) (CD : segment C D)
  (parallel_lines : l1 || l2 || l3 || l4)
  (equal_distance : ∀ l1 l2, dist_between_lines l1 l2)
  (AB_eq_BC : length AB = length BC)
  (AC_eq_CD : length AC = length CD)
  : measure_angle A D C = 30 :=
sorry

end angle_DAC_l489_489784


namespace arithmetic_geometric_sequences_l489_489832

theorem arithmetic_geometric_sequences (a_n b_n : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a_n = 2 * (n + 1) - 1)
  (h2 : ∀ n : ℕ, b_n = 1 / (2^(n + 1))) :
  (∀ n : ℕ, a_n * b_n = 2n - 1 / 2^(n + 1)) ∧
  (∀ n : ℕ, ∑ i in finset.range (n + 1), a_n i * b_n i = 3 - (2n + 3) / 2^(n + 1)) := 
by sorry

end arithmetic_geometric_sequences_l489_489832


namespace triangle_uncovered_area_l489_489910

theorem triangle_uncovered_area :
  let side_length : ℝ := 4 * Real.sqrt 3
  let circle_diameter : ℝ := 1
  let altitude (s : ℝ) : ℝ := (Real.sqrt 3 / 2) * s
  let inner_triangle_side_length (v' : ℝ) : ℝ := (2 * v') / Real.sqrt 3
  let equilateral_triangle_area (a : ℝ) : ℝ := (Real.sqrt 3 / 4) * a^2
  let sector_area (r : ℝ) : ℝ := (r^2 * Real.pi) / 4
  let small_triangle_area (a : ℝ) : ℝ := a^2 * Real.sqrt 3 / 4
  let v := altitude side_length
  let v' := v - 1.5
  let a' := inner_triangle_side_length v'
  let S1 := equilateral_triangle_area a'
  let S2 := 3 * ((Real.sqrt 3 / 4) * side_length^2 - sector_area (circle_diameter / 2))
  in
  (S1 + S2) = (15 / 4 * Real.sqrt 3 - Real.pi / 4)
:= 
by 
  sorry

end triangle_uncovered_area_l489_489910


namespace range_of_a_l489_489413

noncomputable def f (a x : ℝ) : ℝ :=
if h : a ≤ x ∧ x < 0 then -((1/2)^x)
else if h' : 0 ≤ x ∧ x ≤ 4 then -(x^2) + 2*x
else 0

theorem range_of_a (a : ℝ) (h : ∀ x, f a x ∈ Set.Icc (-8 : ℝ) (1 : ℝ)) : 
  a ∈ Set.Ico (-3 : ℝ) 0 :=
sorry

end range_of_a_l489_489413


namespace segment_inequality_l489_489184

theorem segment_inequality (A B C D E : Point) :
  dist A B + dist C D + dist D E + dist E C ≤
  dist A C + dist A D + dist A E + dist B C + dist B D + dist B E := 
  sorry

end segment_inequality_l489_489184


namespace total_basketballs_l489_489559

theorem total_basketballs (players : ℕ) (basketballs_per_player : ℕ) : 
  players = 35 → basketballs_per_player = 15 → players * basketballs_per_player = 525 :=
by
  intros h_players h_basketballs_per_player
  rw [h_players, h_basketballs_per_player]
  exact (Nat.mul_eq .refl 35 .refl 15).trans (Nat.mul_eq_iff_eq_div.mpr (by norm_num))

end total_basketballs_l489_489559


namespace louisa_second_day_distance_l489_489178

variables 
  (distance_first_day : ℝ)
  (speed : ℝ)
  (time_first_day : ℝ)
  (time_second_day : ℝ)
  (distance_second_day : ℝ)

def conditions : Prop :=
  distance_first_day = 250 ∧
  speed = 33.333333333333336 ∧
  time_first_day = distance_first_day / speed ∧
  time_second_day = time_first_day + 3 ∧
  distance_second_day = speed * time_second_day

theorem louisa_second_day_distance (h : conditions) : distance_second_day = 350 := 
sorry

end louisa_second_day_distance_l489_489178


namespace triangle_max_area_l489_489022

theorem triangle_max_area (a b c : ℝ) 
  (h1 : a = 2) 
  (h2 : (a - b + c) / c = b / (a + b - c)) : 
  ∃ A : ℝ, 
    (A = Real.sqrt(3) := 
sorry

end triangle_max_area_l489_489022


namespace equilateral_triangle_cover_l489_489663

variables (T : Type) [metric_space T] [eq_triangle : ∀ t : T, equilateral_triangle t]
variables (S : set T)
  (H1 : ∀ t1 t2 ∈ S, parallel_translation t1 t2)
  (H2 : ∀ t1 t2 ∈ S, t1 ∩ t2 ≠ ∅)

theorem equilateral_triangle_cover :
  ∃ P1 P2 P3 : T, ∀ t ∈ S, t.contains P1 ∨ t.contains P2 ∨ t.contains P3 :=
sorry

end equilateral_triangle_cover_l489_489663


namespace highest_score_is_96_l489_489112

theorem highest_score_is_96 :
  let standard_score := 85
  let deviations := [-9, -4, 11, -7, 0]
  let actual_scores := deviations.map (λ x => standard_score + x)
  actual_scores.maximum = 96 :=
by
  sorry

end highest_score_is_96_l489_489112


namespace trajectory_equation_l489_489103

def distance_from_point_to_line (P : ℝ × ℝ) (y_line : ℝ) : ℝ :=
  abs (P.2 - y_line)

def distance_from_point_to_point (P1 P2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

def is_valid_trajectory (P : ℝ × ℝ) : Prop :=
  distance_from_point_to_line P (-1) + 2 = distance_from_point_to_point P (0, 3)

theorem trajectory_equation :
  ∀ P : ℝ × ℝ, is_valid_trajectory P ↔ P.1^2 = 12 * P.2 :=
sorry

end trajectory_equation_l489_489103


namespace polar_to_rectangular_correct_l489_489333

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
let x := r * Real.cos θ in
let y := r * Real.sin θ in
(x, y)

theorem polar_to_rectangular_correct :
  polar_to_rectangular (-3) (5 * Real.pi / 6) = (3 * Real.sqrt 3 / 2, -3 / 2) :=
by
  sorry

end polar_to_rectangular_correct_l489_489333


namespace words_with_at_least_one_consonant_l489_489862

-- Define the sets of vowels and consonants
def vowels : Set Char := {'A', 'E', 'I'}
def consonants : Set Char := {'B', 'C', 'D'}
def all_letters : Set Char := vowels ∪ consonants

-- Total number of 5-letter words that can be formed from the 6 letters
def total_combinations := (all_letters.card)^5

-- Total number of 5-letter words that can be formed using only vowels
def vowel_combinations := (vowels.card)^5

-- Number of 5-letter words with at least one consonant
def words_with_consonant := total_combinations - vowel_combinations

theorem words_with_at_least_one_consonant : words_with_consonant = 7533 := by sorry

end words_with_at_least_one_consonant_l489_489862


namespace number_of_valid_z_l489_489950

def f (z : ℂ) : ℂ := z^2 + complex.I * z + 2

def valid_re_im (a b : ℤ) : Prop :=
  abs a ≤ 5 ∧ abs b ≤ 5 ∧ a + 5 / 4 < b^2

theorem number_of_valid_z :
  { (a, b) : ℤ × ℤ // valid_re_im a b }.card = 36 :=
sorry

end number_of_valid_z_l489_489950


namespace min_chord_length_l489_489779

theorem min_chord_length (p t : ℝ) (A B : ℝ → ℝ) :
  (∀ t, A = (2 * p * t^2, 2 * p * t)) → (∀ s, B = (2 * p * s^2, 2 * p * s)) →
  (∀ t s, t + s = -1 / (2 * t)) →
  (∃ t, t = sqrt(2) / 2 ∨ t = -sqrt(2) / 2) →
  (min (3 * sqrt(3) * p) = 3 * sqrt(3) * p) :=
begin
  sorry
end

end min_chord_length_l489_489779


namespace polygon_sides_l489_489593

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l489_489593


namespace problem_I_problem_II_l489_489848

-- Definition of the function and necessary conditions
noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  4 * (Real.cos (ω * x - Real.pi / 6) * Real.sin (ω * x)) - Real.cos (2 * ω * x + Real.pi)

def condition1 (ω : ℝ) : Prop := (ω > 0)

-- Problem (I): Find the range of the function y = f(x)
theorem problem_I (ω : ℝ) (h : ω > 0) : set.range (λ x, f x ω) = set.Icc (1 - Real.sqrt 3) (1 + Real.sqrt 3) :=
sorry

-- Problem (II): Find the maximum value of ω given the function is increasing on [-3π/2, π/2]
theorem problem_II (h : MonotoneOn (λ x, f x ω) (set.Icc (-3* Real.pi / 2) (Real.pi / 2))) : ω ≤ 1/6 :=
sorry

end problem_I_problem_II_l489_489848


namespace remainder_when_b_divided_by_11_l489_489156

theorem remainder_when_b_divided_by_11 (n : ℕ) (hn : n = 1) (b : ℕ) 
  (hb : b ≡ (5^(3*n) + 4)⁻¹ [MOD 11]) : b % 11 = 7 :=
sorry

end remainder_when_b_divided_by_11_l489_489156


namespace handbag_monday_price_l489_489676

theorem handbag_monday_price (initial_price : ℝ) (primary_discount : ℝ) (additional_discount : ℝ)
(h_initial_price : initial_price = 250)
(h_primary_discount : primary_discount = 0.4)
(h_additional_discount : additional_discount = 0.1) :
(initial_price - initial_price * primary_discount) - ((initial_price - initial_price * primary_discount) * additional_discount) = 135 := by
  sorry

end handbag_monday_price_l489_489676


namespace polygon_sides_l489_489597

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end polygon_sides_l489_489597


namespace find_m_plus_n_l489_489857

theorem find_m_plus_n
  (m n : ℝ)
  (l1 : ∀ x y : ℝ, 2 * x + m * y + 2 = 0)
  (l2 : ∀ x y : ℝ, 2 * x + y - 1 = 0)
  (l3 : ∀ x y : ℝ, x + n * y + 1 = 0)
  (parallel_l1_l2 : ∀ x y : ℝ, (2 * x + m * y + 2 = 0) → (2 * x + y - 1 = 0))
  (perpendicular_l1_l3 : ∀ x y : ℝ, (2 * x + m * y + 2 = 0) ∧ (x + n * y + 1 = 0) → true) :
  m + n = -1 :=
by
  sorry

end find_m_plus_n_l489_489857


namespace run_up_and_down_when_escalator_moves_up_l489_489861

def time_vasya_runs_up_and_down (escalator_stopped esclev_downtime: ℝ) : ℝ :=
  let y: ℝ := 1 / sqrt(45)
  let x: ℝ := 1 / 2 
  have h₁ : 6 = 3 / x, by sorry
  have h₂: 13.5 = 1 / (x + y) + 1 / (x / 2 - y), by sorry
  5.4 * 60 = 324

theorem run_up_and_down_when_escalator_moves_up (y positive)  : 
  (time_vasya_runs_up_and_down 6 6 = 324) :=
by
  sorry

end run_up_and_down_when_escalator_moves_up_l489_489861


namespace number_of_valid_grids_l489_489898

-- Define the grid as a list of lists of booleans, where true represents a black cell
def grid := list (list bool)

-- Define the constraints: rows and columns with specific numbers of black cells
def row_constraints : list nat := [_, _, _, _] -- Specify the actual numbers for each row
def column_constraints : list nat := [_, _, _, _] -- Specify the actual numbers for each column

-- Define a function to check if a grid satisfies the given constraints
def satisfies_constraints (g : grid) : bool :=
  (∀ (r : nat), r < 4 → list.count true (g.get r) = row_constraints.get r) ∧
  (∀ (c : nat), c < 4 → list.count true (list.map (λ row, row.get c) g) = column_constraints.get c)

-- Define the main theorem to prove the number of valid grids
theorem number_of_valid_grids : ∃ configs : list grid, list.length configs = 5 ∧ ∀ g ∈ configs, satisfies_constraints g :=
sorry

end number_of_valid_grids_l489_489898


namespace ellipse_problem_l489_489389

noncomputable def ellipse_foci : Prop := 
  ∀ (a b c : ℝ) (A : ℝ × ℝ), 
  0 < b ∧ b < a ∧ 2 * a = 8 ∧
  (c = a * (√3) / 2) ∧
  (a^2 - b^2 = c^2) ∧
  (A.2 = b^2 / a) ∧
  ((A.1 = c) ∧ ((A.1 - c) * (F₂ - F₁) = 0)) ∧
  (1 / √3 = A.2 / A.1) → 
  (a = 4 ∧ b = 2).

noncomputable def ellipse_equation : Prop := 
  ellipse_foci → 
  ∀ (x y : ℝ), 
  ((x^2 / 16) + (y^2 / 4)) = 1.

noncomputable def line_intersection : Prop :=
  ∀ (x y k : ℝ) (B : ℝ × ℝ),
  (B = (0, -2)) ∧ (y = k * x + 3/2) ∧
  (x^2 / 16 + y^2 / 4 = 1) ↔
  ((1 + 4 * k^2) * x^2 + 12 * k * x - 7 = 0) →
  (∀ D : ℝ × ℝ, 
   (D = (- 6 * k / (1 + 4 * k^2), 3 / (2 * (1 + 4 * k^2)))) 
    ∧ (B.2 - D.2)/(D.1 - B.1) = -1/k) →
  k = sqrt(5)/4 ∨ k = -sqrt(5)/4.

theorem ellipse_problem : Prop :=
  ellipse_equation ∧ line_intersection.

end ellipse_problem_l489_489389


namespace a₂_value_general_formula_for_aₙ_minimum_value_of_S_when_a_is_minus_9_minimum_value_S_n_l489_489049

-- Definitions based on conditions
def a₁ (a : ℕ) : ℕ := a
def S (n : ℕ) : ℕ := sorry

-- (I) Prove that a₂ = 2
theorem a₂_value (a : ℕ) (h₁ : a ≠ 0) : S 1 = a₁ a → 2 * S 1 = a₁ a * a₂ :=
by sorry

-- (II) Prove the general formula for the n-th term of the sequence
theorem general_formula_for_aₙ (a : ℕ) (h₁ : a ≠ 0) (n : ℕ) :
  a n = if odd n then n + a - 1 else n :=
by sorry

-- (III) Prove the minimum value of Sₙ when a = -9
theorem minimum_value_of_S_when_a_is_minus_9 : S n = if odd n then (n - 10) * (n + 1) / 2 else n * (n - 9) / 2 :=
by sorry

theorem minimum_value_S_n (h₁ : a = -9) (n : ℕ) : min_S = -15 :=
by sorry

end a₂_value_general_formula_for_aₙ_minimum_value_of_S_when_a_is_minus_9_minimum_value_S_n_l489_489049


namespace total_number_of_workers_l489_489204

theorem total_number_of_workers 
  (W : ℕ) 
  (h_all_avg : W * 8000 = 10 * 12000 + (W - 10) * 6000) : 
  W = 30 := 
by
  sorry

end total_number_of_workers_l489_489204


namespace sum_first_2017_terms_l489_489048

theorem sum_first_2017_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → S (n + 1) - S n = 3^n / a n) :
  S 2017 = 3^1009 - 2 := sorry

end sum_first_2017_terms_l489_489048


namespace problem_statement_l489_489154

variable {a b x : ℝ} {f g : ℝ → ℝ}

theorem problem_statement (h_f_diff : ∀ x ∈ set.Icc a b, DifferentiableAt ℝ f x)
  (h_g_diff : ∀ x ∈ set.Icc a b, DifferentiableAt ℝ g x) 
  (h_f_prime_gt_g_prime : ∀ x ∈ set.Ioo a b, (deriv f x) > (deriv g x))
  (h_ax_in_interval : a < x) (h_xb_in_interval : x < b) :
  f x + g a > g x + f a := by
  sorry

end problem_statement_l489_489154


namespace percent_increase_output_l489_489933

theorem percent_increase_output (B H : ℝ) (hB : B > 0) (hH : H > 0) :
  ((1.8 * B / (0.9 * H) - B / H) / (B / H)) * 100 = 100 := 
by
  sorry

end percent_increase_output_l489_489933


namespace calculation_correct_l489_489321

theorem calculation_correct :
  (2 - 2^2 - 2^3 - 2^4 - ... - 2^2006 + 2^2007) = 6 :=
sorry

end calculation_correct_l489_489321


namespace f_neg_l489_489155

/-- Define f(x) as an odd function --/
def f : ℝ → ℝ := sorry

/-- The property of odd functions: f(-x) = -f(x) --/
axiom odd_fn_property (x : ℝ) : f (-x) = -f x

/-- Define the function for non-negative x --/
axiom f_nonneg (x : ℝ) (hx : 0 ≤ x) : f x = x + 1

/-- The goal is to determine f(x) when x < 0 --/
theorem f_neg (x : ℝ) (h : x < 0) : f x = x - 1 :=
by
  sorry

end f_neg_l489_489155


namespace range_of_a_l489_489855

theorem range_of_a (a : ℝ) :
  (∀ (x y z: ℝ), x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2 * y + 2 * z) ↔ (a ≤ -2 ∨ a ≥ 4) :=
by
sorry

end range_of_a_l489_489855


namespace vasim_share_l489_489715

theorem vasim_share (x : ℕ) (F V R : ℕ) (h1 : F = 3 * x) (h2 : V = 5 * x) (h3 : R = 11 * x) (h4 : R - F = 2400) : V = 1500 :=
by sorry

end vasim_share_l489_489715


namespace jessica_age_l489_489496

theorem jessica_age 
  (j g : ℚ)
  (h1 : g = 15 * j) 
  (h2 : g - j = 60) : 
  j = 30 / 7 :=
by
  sorry

end jessica_age_l489_489496
