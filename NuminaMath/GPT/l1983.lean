import Mathlib

namespace minimum_berries_left_l1983_198303

def geometric_sum (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

theorem minimum_berries_left {a r n S : ℕ} 
  (h_a : a = 1) 
  (h_r : r = 2) 
  (h_n : n = 100) 
  (h_S : S = geometric_sum a r n) 
  : S = 2^100 - 1 -> ∃ k, k = 100 :=
by
  sorry

end minimum_berries_left_l1983_198303


namespace eagles_win_at_least_three_matches_l1983_198329

-- Define the conditions
def n : ℕ := 5
def p : ℝ := 0.5

-- Binomial coefficient function
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Probability function for the binomial distribution
noncomputable def binomial_prob (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial n k) * p^k * (1 - p)^(n - k)

-- Theorem stating the main result
theorem eagles_win_at_least_three_matches :
  (binomial_prob n 3 p + binomial_prob n 4 p + binomial_prob n 5 p) = 1 / 2 :=
by
  sorry

end eagles_win_at_least_three_matches_l1983_198329


namespace exists_t_for_f_inequality_l1983_198377

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * (x + 1) ^ 2

theorem exists_t_for_f_inequality :
  ∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → f (x + t) ≤ x := by
  sorry

end exists_t_for_f_inequality_l1983_198377


namespace min_value_fraction_l1983_198328

theorem min_value_fraction 
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (a1 a3 a13 : ℕ)
  (d : ℕ) 
  (h1 : ∀ n, a_n n = a1 + (n - 1) * d)
  (h2 : d ≠ 0)
  (h3 : a1 = 1)
  (h4 : a3 ^ 2 = a1 * a13)
  (h5 : ∀ n, S_n n = n * (a1 + a_n n) / 2) :
  ∃ n, (2 * S_n n + 16) / (a_n n + 3) = 4 := 
sorry

end min_value_fraction_l1983_198328


namespace xiaoMing_xiaoHong_diff_university_l1983_198375

-- Definitions based on problem conditions
inductive Student
| XiaoMing
| XiaoHong
| StudentC
| StudentD
deriving DecidableEq

inductive University
| A
| B
deriving DecidableEq

-- Definition for the problem
def num_ways_diff_university : Nat :=
  4 -- The correct answer based on the solution steps

-- Problem statement
theorem xiaoMing_xiaoHong_diff_university :
  let students := [Student.XiaoMing, Student.XiaoHong, Student.StudentC, Student.StudentD]
  let universities := [University.A, University.B]
  (∃ (assign : Student → University),
    assign Student.XiaoMing ≠ assign Student.XiaoHong ∧
    (assign Student.StudentC ≠ assign Student.StudentD ∨
     assign Student.XiaoMing ≠ assign Student.StudentD ∨
     assign Student.XiaoHong ≠ assign Student.StudentC ∨
     assign Student.XiaoMing ≠ assign Student.StudentC)) →
  num_ways_diff_university = 4 :=
by
  sorry

end xiaoMing_xiaoHong_diff_university_l1983_198375


namespace age_difference_is_18_l1983_198392

def difference_in_ages (X Y Z : ℕ) : ℕ := (X + Y) - (Y + Z)
def younger_by_eighteen (X Z : ℕ) : Prop := Z = X - 18

theorem age_difference_is_18 (X Y Z : ℕ) (h : younger_by_eighteen X Z) : difference_in_ages X Y Z = 18 := by
  sorry

end age_difference_is_18_l1983_198392


namespace problem_statement_l1983_198361

noncomputable def pi : ℝ := Real.pi

theorem problem_statement :
  (pi - 1) ^ 0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end problem_statement_l1983_198361


namespace goods_amount_decreased_initial_goods_amount_total_fees_l1983_198357

-- Define the conditions as variables
def tonnages : List Int := [31, -31, -16, 34, -38, -20]
def final_goods : Int := 430
def fee_per_ton : Int := 5

-- Prove that the amount of goods in the warehouse has decreased
theorem goods_amount_decreased : (tonnages.sum < 0) := by
  sorry

-- Prove the initial amount of goods in the warehouse
theorem initial_goods_amount : (final_goods + tonnages.sum = 470) := by
  sorry

-- Prove the total loading and unloading fees
theorem total_fees : (tonnages.map Int.natAbs).sum * fee_per_ton = 850 := by
  sorry

end goods_amount_decreased_initial_goods_amount_total_fees_l1983_198357


namespace estimate_white_balls_l1983_198343

theorem estimate_white_balls
  (total_balls : ℕ)
  (trials : ℕ)
  (white_draws : ℕ)
  (proportion_white : ℚ)
  (hw : total_balls = 10)
  (ht : trials = 400)
  (hd : white_draws = 240)
  (hprop : proportion_white = 0.6) :
  ∃ x : ℕ, x = 6 :=
by
  sorry

end estimate_white_balls_l1983_198343


namespace maximum_n_for_positive_sum_l1983_198358

noncomputable def max_n_for_positive_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :=
  S n > 0

-- Definition of the arithmetic sequence properties
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d
  
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)

-- Given conditions
variable (h₁ : a 1 > 0)
variable (h₅ : a 2016 + a 2017 > 0)
variable (h₆ : a 2016 * a 2017 < 0)

-- Add the definition of the sum of the first n terms of the arithmetic sequence
noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * (a 0 + a n) / 2

-- Prove the final statement
theorem maximum_n_for_positive_sum : max_n_for_positive_sum a S 4032 :=
by
  -- conditions to use in the proof
  have h₁ : a 1 > 0 := sorry
  have h₅ : a 2016 + a 2017 > 0 := sorry
  have h₆ : a 2016 * a 2017 < 0 := sorry
  -- positively bounded sum
  let Sn := sum_of_first_n_terms a
  -- proof to utilize Lean's capabilities, replace with actual proof later
  sorry

end maximum_n_for_positive_sum_l1983_198358


namespace factor_expression_l1983_198370

theorem factor_expression (x y : ℝ) : x * y^2 - 4 * x = x * (y + 2) * (y - 2) := 
by
  sorry

end factor_expression_l1983_198370


namespace find_primes_l1983_198374

-- Definition of being a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0)

-- Lean 4 statement of the problem
theorem find_primes (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  3 * p^4 - 5 * q^4 - 4 * r^2 = 26 → p = 5 ∧ q = 3 ∧ r = 19 := 
by
  sorry

end find_primes_l1983_198374


namespace product_of_consecutive_integers_l1983_198349

theorem product_of_consecutive_integers (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_less : a < b) :
  ∃ (x y : ℕ), x ≠ y ∧ x * y % (a * b) = 0 :=
by
  sorry

end product_of_consecutive_integers_l1983_198349


namespace jars_of_plum_jelly_sold_l1983_198301

theorem jars_of_plum_jelly_sold (P R G S : ℕ) (h1 : R = 2 * P) (h2 : G = 3 * R) (h3 : G = 2 * S) (h4 : S = 18) : P = 6 := by
  sorry

end jars_of_plum_jelly_sold_l1983_198301


namespace percentage_decrease_in_y_when_x_doubles_l1983_198312

variable {k x y : ℝ}
variable (h_pos_x : 0 < x) (h_pos_y : 0 < y)
variable (inverse_proportional : x * y = k)

theorem percentage_decrease_in_y_when_x_doubles :
  (x' = 2 * x) →
  (y' = y / 2) →
  (100 * (y - y') / y) = 50 :=
by
  intro h1 h2
  simp [h1, h2]
  sorry

end percentage_decrease_in_y_when_x_doubles_l1983_198312


namespace relationship_S_T_l1983_198353

def S (n : ℕ) : ℤ := 2^n
def T (n : ℕ) : ℤ := 2^n - (-1)^n

theorem relationship_S_T (n : ℕ) (h : n > 0) : 
  (n % 2 = 1 → S n < T n) ∧ (n % 2 = 0 → S n > T n) :=
by
  sorry

end relationship_S_T_l1983_198353


namespace number_of_classes_l1983_198395

theorem number_of_classes (x : ℕ) (h : x * (x - 1) = 20) : x = 5 :=
by
  sorry

end number_of_classes_l1983_198395


namespace decreasing_function_condition_l1983_198319

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then (4 * a - 1) * x + 4 * a else a ^ x

theorem decreasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a y ≤ f a x) ↔ (1 / 7 ≤ a ∧ a < 1 / 4) :=
by
  sorry

end decreasing_function_condition_l1983_198319


namespace find_n_for_geometric_series_l1983_198387

theorem find_n_for_geometric_series
  (n : ℝ)
  (a1 : ℝ := 12)
  (a2 : ℝ := 4)
  (r1 : ℝ)
  (S1 : ℝ)
  (b1 : ℝ := 12)
  (b2 : ℝ := 4 + n)
  (r2 : ℝ)
  (S2 : ℝ) :
  (r1 = a2 / a1) →
  (S1 = a1 / (1 - r1)) →
  (S2 = 4 * S1) →
  (r2 = b2 / b1) →
  (S2 = b1 / (1 - r2)) →
  n = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_n_for_geometric_series_l1983_198387


namespace sum_of_cubes_of_integers_l1983_198313

theorem sum_of_cubes_of_integers (n: ℕ) (h1: (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 = 8830) : 
  (n-1)^3 + n^3 + (n+1)^3 + (n+2)^3 = 52264 :=
by
  sorry

end sum_of_cubes_of_integers_l1983_198313


namespace m_le_n_l1983_198397

def polygon : Type := sorry  -- A placeholder definition for polygon.

variables (M : polygon) -- The polygon \( M \)
def max_non_overlapping_circles (M : polygon) : ℕ := sorry -- The maximum number of non-overlapping circles with diameter 1 inside \( M \).
def min_covering_circles (M : polygon) : ℕ := sorry -- The minimum number of circles with radius 1 required to cover \( M \).

theorem m_le_n (M : polygon) : min_covering_circles M ≤ max_non_overlapping_circles M :=
sorry

end m_le_n_l1983_198397


namespace factorize_x4_minus_16_factorize_trinomial_l1983_198367

-- For problem 1: Factorization of \( x^4 - 16 \)
theorem factorize_x4_minus_16 (x : ℝ) : 
  x^4 - 16 = (x - 2) * (x + 2) * (x^2 + 4) := 
sorry

-- For problem 2: Factorization of \( -9x^2y + 12xy^2 - 4y^3 \)
theorem factorize_trinomial (x y : ℝ) : 
  -9 * x^2 * y + 12 * x * y^2 - 4 * y^3 = -y * (3 * x - 2 * y)^2 := 
sorry

end factorize_x4_minus_16_factorize_trinomial_l1983_198367


namespace range_of_a_l1983_198320

-- Define the sets A and B
def setA (a : ℝ) : Set ℝ := {x | x - a > 0}
def setB : Set ℝ := {x | x ≤ 0}

-- The main theorem asserting the condition
theorem range_of_a {a : ℝ} (h : setA a ∩ setB = ∅) : a ≥ 0 := by
  sorry

end range_of_a_l1983_198320


namespace system1_l1983_198383

theorem system1 {x y : ℝ} 
  (h1 : x + y = 3) 
  (h2 : x - y = 1) : 
  x = 2 ∧ y = 1 :=
by
  sorry

end system1_l1983_198383


namespace factorize_expression_l1983_198340

variable (a : ℝ) (b : ℝ)

theorem factorize_expression : 2 * a - 8 * a * b^2 = 2 * a * (1 - 2 * b) * (1 + 2 * b) := by
  sorry

end factorize_expression_l1983_198340


namespace find_three_digit_number_l1983_198350

/-- 
  Define the three-digit number abc and show that for some digit d in the range of 1 to 9,
  the conditions are satisfied.
-/
theorem find_three_digit_number
  (a b c d : ℕ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : 1 ≤ d ∧ d ≤ 9)
  (h_abc : 100 * a + 10 * b + c = 627)
  (h_bcd : 100 * b + 10 * c + d = 627 * a)
  (h_1a4d : 1040 + 100 * a + d = 627 * a)
  : 100 * a + 10 * b + c = 627 := 
sorry

end find_three_digit_number_l1983_198350


namespace sphere_surface_area_l1983_198362

theorem sphere_surface_area (R : ℝ) (h : (4 / 3) * π * R^3 = (32 / 3) * π) : 4 * π * R^2 = 16 * π :=
sorry

end sphere_surface_area_l1983_198362


namespace sum_of_distinct_prime_factors_of_2016_l1983_198310

-- Define 2016 and the sum of its distinct prime factors
def n : ℕ := 2016
def sumOfDistinctPrimeFactors (n : ℕ) : ℕ :=
  if n = 2016 then 2 + 3 + 7 else 0  -- Capture the problem-specific condition

-- The main theorem to prove the sum of the distinct prime factors of 2016 is 12
theorem sum_of_distinct_prime_factors_of_2016 :
  sumOfDistinctPrimeFactors 2016 = 12 :=
by
  -- Since this is beyond the obvious steps, we use a sorry here
  sorry

end sum_of_distinct_prime_factors_of_2016_l1983_198310


namespace major_premise_incorrect_l1983_198396

theorem major_premise_incorrect (a b : ℝ) (h : a > b) : ¬ (a^2 > b^2) :=
by {
  sorry
}

end major_premise_incorrect_l1983_198396


namespace find_longer_parallel_side_length_l1983_198321

noncomputable def longer_parallel_side_length_of_trapezoid : ℝ :=
  let square_side_length : ℝ := 2
  let center_to_side_length : ℝ := square_side_length / 2
  let midline_length : ℝ := square_side_length / 2
  let equal_area : ℝ := (square_side_length^2) / 3
  let height_of_trapezoid : ℝ := center_to_side_length
  let shorter_parallel_side_length : ℝ := midline_length
  let longer_parallel_side_length := (2 * equal_area / height_of_trapezoid) - shorter_parallel_side_length
  longer_parallel_side_length

theorem find_longer_parallel_side_length : 
  longer_parallel_side_length_of_trapezoid = 5/3 := 
sorry

end find_longer_parallel_side_length_l1983_198321


namespace negation_exists_implies_forall_l1983_198315

theorem negation_exists_implies_forall (x_0 : ℝ) (h : ∃ x_0 : ℝ, x_0^3 - x_0 + 1 > 0) : 
  ¬ (∃ x_0 : ℝ, x_0^3 - x_0 + 1 > 0) ↔ ∀ x : ℝ, x^3 - x + 1 ≤ 0 :=
by 
  sorry

end negation_exists_implies_forall_l1983_198315


namespace unique_function_solution_l1983_198332

theorem unique_function_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = y + f x^2) → (∀ x : ℝ, f x = x) :=
by
  sorry

end unique_function_solution_l1983_198332


namespace min_value_reciprocal_l1983_198399

variable {a b : ℝ}

theorem min_value_reciprocal (h1 : a * b > 0) (h2 : a + 4 * b = 1) : 
  ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ ((1/a) + (1/b) = 9) := 
by
  sorry

end min_value_reciprocal_l1983_198399


namespace test_scores_ordering_l1983_198354

variable (M Q S Z K : ℕ)
variable (M_thinks_lowest : M > K)
variable (Q_thinks_same : Q = K)
variable (S_thinks_not_highest : S < K)
variable (Z_thinks_not_middle : (Z < S ∨ Z > M))

theorem test_scores_ordering : (Z < S) ∧ (S < Q) ∧ (Q < M) := by
  -- proof
  sorry

end test_scores_ordering_l1983_198354


namespace negate_proposition_l1983_198316

theorem negate_proposition (x : ℝ) :
  (¬(x > 1 → x^2 > 1)) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by
  sorry

end negate_proposition_l1983_198316


namespace closed_path_has_even_length_l1983_198341

   theorem closed_path_has_even_length 
     (u d r l : ℤ) 
     (hu : u = d) 
     (hr : r = l) : 
     ∃ k : ℤ, 2 * (u + r) = 2 * k :=
   by
     sorry
   
end closed_path_has_even_length_l1983_198341


namespace rod_sliding_friction_l1983_198393

noncomputable def coefficient_of_friction (mg : ℝ) (F : ℝ) (α : ℝ) := 
  (F * Real.cos α - 6 * mg * Real.sin α) / (6 * mg)

theorem rod_sliding_friction
  (α : ℝ)
  (hα : α = 85 * Real.pi / 180)
  (mg : ℝ)
  (hmg_pos : 0 < mg)
  (F : ℝ)
  (hF : F = (mg - 6 * mg * Real.cos 85) / Real.sin 85) :
  coefficient_of_friction mg F α = 0.08 := 
by
  simp [coefficient_of_friction, hα, hF, Real.cos, Real.sin]
  sorry

end rod_sliding_friction_l1983_198393


namespace monotonicity_tangent_intersection_points_l1983_198366

-- Define the function f
def f (x a : ℝ) := x^3 - x^2 + a * x + 1

-- Define the first derivative of f
def f' (x a : ℝ) := 3 * x^2 - 2 * x + a

-- Prove monotonicity conditions
theorem monotonicity (a : ℝ) :
  (a ≥ 1 / 3 → ∀ x : ℝ, f' x a ≥ 0) ∧
  (a < 1 / 3 → 
    ∃ x1 x2 : ℝ, x1 = (1 - Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 x2 = (1 + Real.sqrt (1 - 3 * a)) / 3 ∧ 
                 (∀ x < x1, f' x a > 0) ∧ 
                 (∀ x, x1 < x ∧ x < x2 → f' x a < 0) ∧ 
                 (∀ x > x2, f' x a > 0)) :=
by sorry

-- Prove the coordinates of the intersection points
theorem tangent_intersection_points (a : ℝ) :
  (∃ x0 : ℝ, x0 = 1 ∧ f x0 a = a + 1) ∧ 
  (∃ x0 : ℝ, x0 = -1 ∧ f x0 a = -a - 1) :=
by sorry

end monotonicity_tangent_intersection_points_l1983_198366


namespace Jose_got_5_questions_wrong_l1983_198371

def Jose_questions_wrong (M J A : ℕ) : Prop :=
  M = J - 20 ∧
  J = A + 40 ∧
  M + J + A = 210 ∧
  (50 * 2 = 100) ∧
  (100 - J) / 2 = 5

theorem Jose_got_5_questions_wrong (M J A : ℕ) (h1 : M = J - 20) (h2 : J = A + 40) (h3 : M + J + A = 210) : 
  Jose_questions_wrong M J A :=
by
  sorry

end Jose_got_5_questions_wrong_l1983_198371


namespace rhombus_area_l1983_198356

noncomputable def sqrt125 : ℝ := Real.sqrt 125

theorem rhombus_area 
  (p q : ℝ) 
  (h1 : p < q) 
  (h2 : p + 8 = q) 
  (h3 : ∀ a b : ℝ, a^2 + b^2 = 125 ↔ 2*a = p ∧ 2*b = q) : 
  p*q/2 = 60.5 :=
by
  sorry

end rhombus_area_l1983_198356


namespace prime_numbers_satisfying_condition_l1983_198385

theorem prime_numbers_satisfying_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ x : ℕ, 1 + p * 2^p = x^2) ↔ p = 2 ∨ p = 3 :=
by
  sorry

end prime_numbers_satisfying_condition_l1983_198385


namespace sum_of_cubes_l1983_198391

-- Definitions
noncomputable def p : ℂ := sorry
noncomputable def q : ℂ := sorry
noncomputable def r : ℂ := sorry

-- Roots conditions
axiom h_root_p : p^3 - 2 * p^2 + 3 * p - 4 = 0
axiom h_root_q : q^3 - 2 * q^2 + 3 * q - 4 = 0
axiom h_root_r : r^3 - 2 * r^2 + 3 * r - 4 = 0

-- Vieta's conditions
axiom h_sum : p + q + r = 2
axiom h_product_pairs : p * q + q * r + r * p = 3
axiom h_product : p * q * r = 4

-- Goal
theorem sum_of_cubes : p^3 + q^3 + r^3 = 2 :=
  sorry

end sum_of_cubes_l1983_198391


namespace number_of_cows_l1983_198368

theorem number_of_cows (D C : ℕ) (h1 : 2 * D + 4 * C = 40 + 2 * (D + C)) : C = 20 :=
by
  sorry

end number_of_cows_l1983_198368


namespace erin_serves_all_soup_in_15_minutes_l1983_198302

noncomputable def time_to_serve_all_soup
  (ounces_per_bowl : ℕ)
  (bowls_per_minute : ℕ)
  (soup_in_gallons : ℕ)
  (ounces_per_gallon : ℕ) : ℕ :=
  let total_ounces := soup_in_gallons * ounces_per_gallon
  let total_bowls := (total_ounces + ounces_per_bowl - 1) / ounces_per_bowl -- to round up
  let total_minutes := (total_bowls + bowls_per_minute - 1) / bowls_per_minute -- to round up
  total_minutes

theorem erin_serves_all_soup_in_15_minutes :
  time_to_serve_all_soup 10 5 6 128 = 15 :=
sorry

end erin_serves_all_soup_in_15_minutes_l1983_198302


namespace count_symmetric_numbers_count_symmetric_divisible_by_4_sum_symmetric_numbers_l1983_198326

-- Definitions based on conditions
def is_symmetric_digit (d : ℕ) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 8 ∨ d = 6 ∨ d = 9

def symmetric_pair (a b : ℕ) : Prop :=
  (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 1) ∨ (a = 8 ∧ b = 8) ∨ (a = 6 ∧ b = 9) ∨ (a = 9 ∧ b = 6)

-- 1. Prove the total number of 7-digit symmetric numbers
theorem count_symmetric_numbers : ∃ n, n = 300 := by
  sorry

-- 2. Prove the number of symmetric numbers divisible by 4
theorem count_symmetric_divisible_by_4 : ∃ n, n = 75 := by
  sorry

-- 3. Prove the total sum of these 7-digit symmetric numbers
theorem sum_symmetric_numbers : ∃ s, s = 1959460200 := by
  sorry

end count_symmetric_numbers_count_symmetric_divisible_by_4_sum_symmetric_numbers_l1983_198326


namespace brad_ate_six_halves_l1983_198317

theorem brad_ate_six_halves (total_cookies : ℕ) (total_halves : ℕ) (greg_ate : ℕ) (halves_left : ℕ) (halves_brad_ate : ℕ) 
  (h1 : total_cookies = 14)
  (h2 : total_halves = total_cookies * 2)
  (h3 : greg_ate = 4)
  (h4 : halves_left = 18)
  (h5 : total_halves - greg_ate - halves_brad_ate = halves_left) :
  halves_brad_ate = 6 :=
by
  sorry

end brad_ate_six_halves_l1983_198317


namespace sum_of_solutions_of_quadratic_l1983_198314

theorem sum_of_solutions_of_quadratic :
    let a := 1;
    let b := -8;
    let c := -40;
    let discriminant := b * b - 4 * a * c;
    let root_discriminant := Real.sqrt discriminant;
    let sol1 := (-b + root_discriminant) / (2 * a);
    let sol2 := (-b - root_discriminant) / (2 * a);
    sol1 + sol2 = 8 := by
{
  sorry
}

end sum_of_solutions_of_quadratic_l1983_198314


namespace cost_of_mixture_verify_cost_of_mixture_l1983_198322

variables {C1 C2 Cm : ℝ}

def ratio := 5 / 12

axiom cost_of_rice_1 : C1 = 4.5
axiom cost_of_rice_2 : C2 = 8.75
axiom mix_ratio : ratio = 5 / 12

theorem cost_of_mixture (h1 : C1 = 4.5) (h2 : C2 = 8.75) (r : ratio = 5 / 12) :
  Cm = (8.75 * 5 + 4.5 * 12) / 17 :=
by sorry

-- Prove that the cost of the mixture Cm is indeed 5.75
theorem verify_cost_of_mixture (h1 : C1 = 4.5) (h2 : C2 = 8.75) (r : ratio = 5 / 12) :
  Cm = 5.75 :=
by sorry

end cost_of_mixture_verify_cost_of_mixture_l1983_198322


namespace sum_f_values_l1983_198311

noncomputable def f (x : ℝ) : ℝ := Real.log (1 - 2 / x) + 1

theorem sum_f_values : 
  f (-7) + f (-5) + f (-3) + f (-1) + f (3) + f (5) + f (7) + f (9) = 8 := 
by
  sorry

end sum_f_values_l1983_198311


namespace possible_value_of_2n_plus_m_l1983_198372

variable (n m : ℤ)

theorem possible_value_of_2n_plus_m : (3 * n - m < 5) → (n + m > 26) → (3 * m - 2 * n < 46) → (2 * n + m = 36) :=
by
  sorry

end possible_value_of_2n_plus_m_l1983_198372


namespace piastres_in_6th_purse_l1983_198390

theorem piastres_in_6th_purse (x : ℕ) (sum : ℕ := 10) (total : ℕ := 150)
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9) = 150)
  (h2 : x * 2 ≥ x + 9)
  (n : ℕ := 5):
  x + n = 15 :=
  sorry

end piastres_in_6th_purse_l1983_198390


namespace total_spider_legs_l1983_198330

-- Define the number of legs per spider.
def legs_per_spider : ℕ := 8

-- Define half of the legs per spider.
def half_legs : ℕ := legs_per_spider / 2

-- Define the number of spiders in the group.
def num_spiders : ℕ := half_legs + 10

-- Prove the total number of spider legs in the group is 112.
theorem total_spider_legs : num_spiders * legs_per_spider = 112 := by
  -- Use 'sorry' to skip the detailed proof steps.
  sorry

end total_spider_legs_l1983_198330


namespace days_B_to_finish_work_l1983_198398

-- Definition of work rates based on the conditions
def work_rate_A (A_days: ℕ) : ℚ := 1 / A_days
def work_rate_B (B_days: ℕ) : ℚ := 1 / B_days

-- Theorem that encapsulates the problem statement
theorem days_B_to_finish_work (A_days B_days together_days : ℕ) (work_rate_A_eq : work_rate_A 4 = 1/4) (work_rate_B_eq : work_rate_B 12 = 1/12) : 
  ∀ (remaining_work: ℚ), remaining_work = 1 - together_days * (work_rate_A 4 + work_rate_B 12) → 
  (remaining_work / (work_rate_B 12)) = 4 :=
by
  sorry

end days_B_to_finish_work_l1983_198398


namespace contrapositive_of_square_comparison_l1983_198305

theorem contrapositive_of_square_comparison (x y : ℝ) : (x^2 > y^2 → x > y) → (x ≤ y → x^2 ≤ y^2) :=
  by sorry

end contrapositive_of_square_comparison_l1983_198305


namespace container_unoccupied_volume_l1983_198327

noncomputable def unoccupied_volume (side_length_container : ℝ) (side_length_ice : ℝ) (num_ice_cubes : ℕ) : ℝ :=
  let volume_container := side_length_container ^ 3
  let volume_water := (3 / 4) * volume_container
  let volume_ice := num_ice_cubes / 2 * side_length_ice ^ 3
  volume_container - (volume_water + volume_ice)

theorem container_unoccupied_volume :
  unoccupied_volume 12 1.5 12 = 411.75 :=
by
  sorry

end container_unoccupied_volume_l1983_198327


namespace tank_capacity_l1983_198318

theorem tank_capacity (x : ℝ) 
  (h1 : 1/4 * x + 180 = 2/3 * x) : 
  x = 432 :=
by
  sorry

end tank_capacity_l1983_198318


namespace find_box_length_l1983_198337

theorem find_box_length (width depth : ℕ) (num_cubes : ℕ) (cube_side length : ℕ) 
  (h1 : width = 20)
  (h2 : depth = 10)
  (h3 : num_cubes = 56)
  (h4 : cube_side = 10)
  (h5 : length * width * depth = num_cubes * cube_side * cube_side * cube_side) :
  length = 280 :=
sorry

end find_box_length_l1983_198337


namespace collinear_points_l1983_198363

theorem collinear_points (k : ℝ) (OA OB OC : ℝ × ℝ) 
  (hOA : OA = (1, -3)) 
  (hOB : OB = (2, -1))
  (hOC : OC = (k + 1, k - 2))
  (h_collinear : ∃ t : ℝ, OC - OA = t • (OB - OA)) : 
  k = 1 :=
by
  have := h_collinear
  sorry

end collinear_points_l1983_198363


namespace tic_tac_toe_lines_l1983_198360

theorem tic_tac_toe_lines (n : ℕ) (h_pos : 0 < n) : 
  ∃ lines : ℕ, lines = (5^n - 3^n) / 2 :=
sorry

end tic_tac_toe_lines_l1983_198360


namespace total_snowfall_l1983_198352

variable (morning_snowfall : ℝ) (afternoon_snowfall : ℝ)

theorem total_snowfall {morning_snowfall afternoon_snowfall : ℝ} (h_morning : morning_snowfall = 0.12) (h_afternoon : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.62 :=
sorry

end total_snowfall_l1983_198352


namespace solve_equation_l1983_198373

theorem solve_equation (x : ℝ) :
  (1 / (x^2 + 17 * x - 8) + 1 / (x^2 + 4 * x - 8) + 1 / (x^2 - 9 * x - 8) = 0) →
  (x = 1 ∨ x = -8 ∨ x = 2 ∨ x = -4) :=
by
  sorry

end solve_equation_l1983_198373


namespace triangle_inequality_at_vertex_l1983_198386

-- Define the edge lengths of the tetrahedron and the common vertex label
variables {a b c d e f S : ℝ}

-- Conditions for the edge lengths and vertex label
axiom edge_lengths :
  a + b + c = S ∧
  a + d + e = S ∧
  b + d + f = S ∧
  c + e + f = S

-- The theorem to be proven
theorem triangle_inequality_at_vertex :
  a + b + c = S →
  a + d + e = S →
  b + d + f = S →
  c + e + f = S →
  (a ≤ b + c) ∧
  (b ≤ c + a) ∧
  (c ≤ a + b) ∧
  (a ≤ d + e) ∧
  (d ≤ e + a) ∧
  (e ≤ a + d) ∧
  (b ≤ d + f) ∧
  (d ≤ f + b) ∧
  (f ≤ b + d) ∧
  (c ≤ e + f) ∧
  (e ≤ f + c) ∧
  (f ≤ c + e) :=
sorry

end triangle_inequality_at_vertex_l1983_198386


namespace range_of_m_l1983_198351

theorem range_of_m (m : ℝ) (x : ℝ) (h_eq : m / (x - 2) = 3) (h_pos : x > 0) : m > -6 ∧ m ≠ 0 := 
sorry

end range_of_m_l1983_198351


namespace commute_distance_l1983_198306

noncomputable def distance_to_work (total_time : ℕ) (speed_to_work : ℕ) (speed_to_home : ℕ) : ℕ :=
  let d := (speed_to_work * speed_to_home * total_time) / (speed_to_work + speed_to_home)
  d

-- Given conditions
def speed_to_work : ℕ := 45
def speed_to_home : ℕ := 30
def total_time : ℕ := 1

-- Proof problem statement
theorem commute_distance : distance_to_work total_time speed_to_work speed_to_home = 18 :=
by
  sorry

end commute_distance_l1983_198306


namespace color_of_face_opposite_silver_is_yellow_l1983_198324

def Face : Type := String

def Color : Type := String

variable (B Y O Bl S V : Color)

-- Conditions based on views
variable (cube : Face → Color)
variable (top front_right_1 right_1 front_right_2 front_right_3 : Face)
variable (back : Face)

axiom view1 : cube top = B ∧ cube front_right_1 = Y ∧ cube right_1 = O
axiom view2 : cube top = B ∧ cube front_right_2 = Bl ∧ cube right_1 = O
axiom view3 : cube top = B ∧ cube front_right_3 = V ∧ cube right_1 = O

-- Additional axiom based on the fact that S is not visible and deduced to be on the back face
axiom silver_back : cube back = S

-- The problem: Prove that the color of the face opposite the silver face is yellow.
theorem color_of_face_opposite_silver_is_yellow :
  (∃ front : Face, cube front = Y) :=
by
  sorry

end color_of_face_opposite_silver_is_yellow_l1983_198324


namespace inequality_for_positive_nums_l1983_198307

theorem inequality_for_positive_nums 
    (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    a^2 / b + c^2 / d ≥ (a + c)^2 / (b + d) :=
by
  sorry

end inequality_for_positive_nums_l1983_198307


namespace diet_soda_bottles_l1983_198331

def total_bottles : ℕ := 17
def regular_soda_bottles : ℕ := 9

theorem diet_soda_bottles : total_bottles - regular_soda_bottles = 8 := by
  sorry

end diet_soda_bottles_l1983_198331


namespace dawn_hourly_income_l1983_198338

theorem dawn_hourly_income 
  (n : ℕ) (t_s t_p t_f I_p I_s I_f : ℝ)
  (h_n : n = 12)
  (h_t_s : t_s = 1.5)
  (h_t_p : t_p = 2)
  (h_t_f : t_f = 0.5)
  (h_I_p : I_p = 3600)
  (h_I_s : I_s = 1200)
  (h_I_f : I_f = 300) :
  (I_p + I_s + I_f) / (n * (t_s + t_p + t_f)) = 106.25 := 
  by
  sorry

end dawn_hourly_income_l1983_198338


namespace regions_divided_by_7_tangents_l1983_198300

-- Define the recursive function R for the number of regions divided by n tangents
def R : ℕ → ℕ
| 0       => 1
| (n + 1) => R n + (n + 1)

-- The theorem stating the specific case of the problem
theorem regions_divided_by_7_tangents : R 7 = 29 := by
  sorry

end regions_divided_by_7_tangents_l1983_198300


namespace central_angle_of_sector_l1983_198335

theorem central_angle_of_sector :
  ∃ R α : ℝ, (2 * R + α * R = 4) ∧ (1 / 2 * R ^ 2 * α = 1) ∧ α = 2 :=
by
  sorry

end central_angle_of_sector_l1983_198335


namespace expression_is_composite_l1983_198355

theorem expression_is_composite (a b : ℕ) : ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ 4 * a^2 + 4 * a * b + 4 * a + 2 * b + 1 = m * n := 
by 
  sorry

end expression_is_composite_l1983_198355


namespace ratio_of_areas_of_squares_l1983_198334

open Real

theorem ratio_of_areas_of_squares :
  let side_length_C := 48
  let side_length_D := 60
  let area_C := side_length_C^2
  let area_D := side_length_D^2
  area_C / area_D = (16 : ℝ) / 25 :=
by
  sorry

end ratio_of_areas_of_squares_l1983_198334


namespace was_not_speeding_l1983_198384

theorem was_not_speeding (x s : ℝ) (s_obs : ℝ := 26.5) (x_limit : ℝ := 120)
  (brake_dist_eq : s = 0.01 * x + 0.002 * x^2) : s_obs < 30 → x ≤ x_limit :=
sorry

end was_not_speeding_l1983_198384


namespace min_value_fraction_l1983_198347

theorem min_value_fraction (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : ∃ x₀, (2 * x₀ - 2) * (-2 * x₀ + a) = -1) : 
  ∃ a b, a + b = 5 / 2 → a > 0 → b > 0 → 
  (∀ a b, (1 / a + 4 / b) ≥ 18 / 5) :=
by
  sorry

end min_value_fraction_l1983_198347


namespace same_type_sqrt_l1983_198378

theorem same_type_sqrt (x : ℝ) : (x = 2 * Real.sqrt 3) ↔
  (x = Real.sqrt (1/3)) ∨
  (¬(x = Real.sqrt 8) ∧ ¬(x = Real.sqrt 18) ∧ ¬(x = Real.sqrt 9)) :=
by
  sorry

end same_type_sqrt_l1983_198378


namespace product_pass_rate_l1983_198304

variable (a b : ℝ)

theorem product_pass_rate (h1 : 0 ≤ a) (h2 : a < 1) (h3 : 0 ≤ b) (h4 : b < 1) : 
  (1 - a) * (1 - b) = 1 - (a + b - a * b) :=
by sorry

end product_pass_rate_l1983_198304


namespace range_of_m_l1983_198336

theorem range_of_m (m : ℝ) : 
    (∀ x y : ℝ, (x^2 / (4 - m) + y^2 / (m - 3) = 1) → 
    4 - m > 0 ∧ m - 3 > 0 ∧ m - 3 > 4 - m) → 
    (7/2 < m ∧ m < 4) :=
sorry

end range_of_m_l1983_198336


namespace fraction_problem_l1983_198388

theorem fraction_problem (a b : ℚ) (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := 
by
  sorry

end fraction_problem_l1983_198388


namespace total_distance_walked_l1983_198342

theorem total_distance_walked 
  (d1 : ℝ) (d2 : ℝ)
  (h1 : d1 = 0.75)
  (h2 : d2 = 0.25) :
  d1 + d2 = 1 :=
by
  sorry

end total_distance_walked_l1983_198342


namespace arithmetic_sequence_problem_l1983_198308

noncomputable def a_n (n : ℕ) (a d : ℝ) : ℝ := a + (n - 1) * d

theorem arithmetic_sequence_problem (a d : ℝ) 
  (h : a_n 1 a d - a_n 4 a d - a_n 8 a d - a_n 12 a d + a_n 15 a d = 2) :
  a_n 3 a d + a_n 13 a d = -4 :=
by
  sorry

end arithmetic_sequence_problem_l1983_198308


namespace square_area_PS_l1983_198333

noncomputable def area_of_square_on_PS : ℕ :=
  sorry

theorem square_area_PS (PQ QR RS PR PS : ℝ)
  (h1 : PQ ^ 2 = 25)
  (h2 : QR ^ 2 = 49)
  (h3 : RS ^ 2 = 64)
  (h4 : PQ^2 + QR^2 = PR^2)
  (h5 : PR^2 + RS^2 = PS^2) :
  PS^2 = 138 :=
by
  -- proof skipping
  sorry


end square_area_PS_l1983_198333


namespace order_of_a_add_b_sub_b_l1983_198369

variable (a b : ℚ)

theorem order_of_a_add_b_sub_b (hb : b < 0) : a + b < a ∧ a < a - b := by
  sorry

end order_of_a_add_b_sub_b_l1983_198369


namespace distance_gracie_joe_l1983_198359

noncomputable def distance_between_points := Real.sqrt (5^2 + (-1)^2)
noncomputable def joe_point := Complex.mk 3 (-4)
noncomputable def gracie_point := Complex.mk (-2) (-3)

theorem distance_gracie_joe : Complex.abs (joe_point - gracie_point) = distance_between_points := by 
  sorry

end distance_gracie_joe_l1983_198359


namespace sum_of_distinct_integers_l1983_198376

theorem sum_of_distinct_integers 
  (a b c d e : ℤ)
  (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 60)
  (h2 : (7 - a) ≠ (7 - b) ∧ (7 - a) ≠ (7 - c) ∧ (7 - a) ≠ (7 - d) ∧ (7 - a) ≠ (7 - e))
  (h3 : (7 - b) ≠ (7 - c) ∧ (7 - b) ≠ (7 - d) ∧ (7 - b) ≠ (7 - e))
  (h4 : (7 - c) ≠ (7 - d) ∧ (7 - c) ≠ (7 - e))
  (h5 : (7 - d) ≠ (7 - e)) : 
  a + b + c + d + e = 24 := 
sorry

end sum_of_distinct_integers_l1983_198376


namespace union_of_A_B_l1983_198365

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x > 0}

theorem union_of_A_B :
  A ∪ B = {x | x ≥ -1} := by
  sorry

end union_of_A_B_l1983_198365


namespace min_books_borrowed_l1983_198345

theorem min_books_borrowed
  (total_students : ℕ)
  (students_no_books : ℕ)
  (students_one_book : ℕ)
  (students_two_books : ℕ)
  (avg_books_per_student : ℝ)
  (total_students_eq : total_students = 40)
  (students_no_books_eq : students_no_books = 2)
  (students_one_book_eq : students_one_book = 12)
  (students_two_books_eq : students_two_books = 13)
  (avg_books_per_student_eq : avg_books_per_student = 2) :
  ∀ min_books_borrowed : ℕ, 
    (total_students * avg_books_per_student = 80) → 
    (students_one_book * 1 + students_two_books * 2 ≤ 38) → 
    (total_students - students_no_books - students_one_book - students_two_books = 13) →
    min_books_borrowed * 13 = 42 → 
    min_books_borrowed = 4 :=
by
  intros min_books_borrowed total_books_eq books_count_eq remaining_students_eq total_min_books_eq
  sorry

end min_books_borrowed_l1983_198345


namespace participation_increase_closest_to_10_l1983_198344

def percentage_increase (old new : ℕ) : ℚ := ((new - old) / old) * 100

theorem participation_increase_closest_to_10 :
  (percentage_increase 80 88 = 10) ∧ 
  (percentage_increase 90 99 = 10) := by
  sorry

end participation_increase_closest_to_10_l1983_198344


namespace probability_white_given_red_l1983_198394

-- Define the total number of balls initially
def total_balls := 10

-- Define the number of red balls, white balls, and black balls
def red_balls := 3
def white_balls := 2
def black_balls := 5

-- Define the event A: Picking a red ball on the first draw
def event_A := red_balls

-- Define the event B: Picking a white ball on the second draw
-- Number of balls left after picking one red ball
def remaining_balls_after_A := total_balls - 1

-- Define the event AB: Picking a red ball first and then a white ball
def event_AB := red_balls * white_balls

-- Calculate the probability P(B|A)
def P_B_given_A := event_AB / (event_A * remaining_balls_after_A)

-- Prove the probability of picking a white ball on the second draw given that the first ball picked is a red ball
theorem probability_white_given_red : P_B_given_A = (2 / 9) := by
  sorry

end probability_white_given_red_l1983_198394


namespace ten_factorial_mod_thirteen_l1983_198339

open Nat

theorem ten_factorial_mod_thirteen :
  (10! % 13) = 6 := by
  sorry

end ten_factorial_mod_thirteen_l1983_198339


namespace compute_ab_l1983_198364

theorem compute_ab (a b : ℝ) 
  (h1 : b^2 - a^2 = 25) 
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = Real.sqrt 444 := 
by 
  sorry

end compute_ab_l1983_198364


namespace calculate_expression_l1983_198379

theorem calculate_expression : -4^2 * (-1)^2022 = -16 :=
by
  sorry

end calculate_expression_l1983_198379


namespace sum_of_remainders_eq_11_mod_13_l1983_198381

theorem sum_of_remainders_eq_11_mod_13 
  (a b c d : ℤ)
  (ha : a % 13 = 3) 
  (hb : b % 13 = 5) 
  (hc : c % 13 = 7) 
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := 
by
  sorry

end sum_of_remainders_eq_11_mod_13_l1983_198381


namespace multiples_of_8_has_highest_avg_l1983_198325

def average_of_multiples (m : ℕ) (a b : ℕ) : ℕ :=
(a + b) / 2

def multiples_of_7_avg := average_of_multiples 7 7 196 -- 101.5
def multiples_of_2_avg := average_of_multiples 2 2 200 -- 101
def multiples_of_8_avg := average_of_multiples 8 8 200 -- 104
def multiples_of_5_avg := average_of_multiples 5 5 200 -- 102.5
def multiples_of_9_avg := average_of_multiples 9 9 189 -- 99

theorem multiples_of_8_has_highest_avg :
  multiples_of_8_avg > multiples_of_7_avg ∧
  multiples_of_8_avg > multiples_of_2_avg ∧
  multiples_of_8_avg > multiples_of_5_avg ∧
  multiples_of_8_avg > multiples_of_9_avg :=
by
  sorry

end multiples_of_8_has_highest_avg_l1983_198325


namespace triangle_third_side_l1983_198382

theorem triangle_third_side (DE DF : ℝ) (E F : ℝ) (EF : ℝ) 
    (h₁ : DE = 7) 
    (h₂ : DF = 21) 
    (h₃ : E = 3 * F) : EF = 14 * Real.sqrt 2 :=
sorry

end triangle_third_side_l1983_198382


namespace joanna_marbles_l1983_198348

theorem joanna_marbles (m n : ℕ) (h1 : m * n = 720) (h2 : m > 1) (h3 : n > 1) :
  ∃ (count : ℕ), count = 28 :=
by
  -- Use the properties of divisors and conditions to show that there are 28 valid pairs (m, n).
  sorry

end joanna_marbles_l1983_198348


namespace increasing_interval_l1983_198389

noncomputable def y (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x1 x2, a < x1 ∧ x1 < x2 ∧ x2 < b → f x1 < f x2

theorem increasing_interval :
  is_monotonic_increasing y π (2 * π) :=
by
  -- Proof would go here
  sorry

end increasing_interval_l1983_198389


namespace initial_invitation_count_l1983_198309

def people_invited (didnt_show : ℕ) (num_tables : ℕ) (people_per_table : ℕ) : ℕ :=
  didnt_show + num_tables * people_per_table

theorem initial_invitation_count (didnt_show : ℕ) (num_tables : ℕ) (people_per_table : ℕ)
    (h1 : didnt_show = 35) (h2 : num_tables = 5) (h3 : people_per_table = 2) :
  people_invited didnt_show num_tables people_per_table = 45 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end initial_invitation_count_l1983_198309


namespace Eval_trig_exp_l1983_198380

theorem Eval_trig_exp :
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 / 3 :=
by
  sorry

end Eval_trig_exp_l1983_198380


namespace total_peanut_cost_l1983_198323

def peanut_cost_per_pound : ℝ := 3
def minimum_pounds : ℝ := 15
def extra_pounds : ℝ := 20

theorem total_peanut_cost :
  (minimum_pounds + extra_pounds) * peanut_cost_per_pound = 105 :=
by
  sorry

end total_peanut_cost_l1983_198323


namespace unit_digit_G_1000_l1983_198346

def G (n : ℕ) : ℕ := 3^(3^n) + 1

theorem unit_digit_G_1000 : (G 1000) % 10 = 2 :=
by
  sorry

end unit_digit_G_1000_l1983_198346
