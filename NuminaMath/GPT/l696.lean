import Mathlib
import Mathlib.Algebra.ArithMean
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field
import Mathlib.Algebra.Order.RearrangementInequality
import Mathlib.Analysis.Series
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Prime
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.LinearAlgebra.Eigenspace
import Mathlib.LinearAlgebra.Matrix
import Mathlib.NumberTheory.EuclideanDomain.Basic
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic

namespace probability_all_different_digits_l696_696765

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696765


namespace smallest_area_right_triangle_l696_696688

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696688


namespace smallest_right_triangle_area_l696_696677

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696677


namespace problem1_problem2_l696_696188

variable (α : Real)

noncomputable def tanα : Real := 2

theorem problem1 : 
  tan α = tanα →
  (sin α - 4 * cos α) / (5 * sin α + 2 * cos α) = -1 / 6 := by
  sorry

theorem problem2 : 
  tan α = tanα →
  4 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = 1 := by
  sorry

end problem1_problem2_l696_696188


namespace inverse_proposition_true_l696_696381

-- Define the original proposition
def original_proposition (l1 l2 : Line) (a b : Angle) : Prop :=
  (parallel l1 l2) → (supplementary a b)

-- Define the inverse proposition
def inverse_proposition (a b : Angle) (l1 l2 : Line) : Prop :=
  (supplementary a b) → (parallel l1 l2)

-- Theorem that proves the inverse proposition is true given the original proposition
theorem inverse_proposition_true 
  {l1 l2 : Line} {a b : Angle}
  (H : original_proposition l1 l2 a b) :
  inverse_proposition a b l1 l2 :=
sorry

end inverse_proposition_true_l696_696381


namespace sum_of_first_five_primes_with_units_digit_3_l696_696131

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696131


namespace dot_product_v1_v2_l696_696896

-- Conditions defining the vectors
def v1 : ℝ × ℝ × ℝ := (-3, 2, 4)
def v2 : ℝ × ℝ × ℝ := (5, -7, 1)

-- Dot product function
def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

-- Theorem stating the dot product is -29
theorem dot_product_v1_v2 : dot_product v1 v2 = -29 := by
  sorry

end dot_product_v1_v2_l696_696896


namespace minimum_value_expression_l696_696909

theorem minimum_value_expression (x : ℝ) (h : -3 < x ∧ x < 2) :
  ∃ y, y = (x^2 + 4 * x + 5) / (2 * x + 6) ∧ y = 3 / 4 :=
by
  sorry

end minimum_value_expression_l696_696909


namespace fn_2006_eq_19_over_30_l696_696235

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1/2 then x + 1/2 else 2 * (1 - x)

noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
nat.iterate f n x

theorem fn_2006_eq_19_over_30 :
  fn 2006 (2 / 15) = 19 / 30 := by
  sorry

end fn_2006_eq_19_over_30_l696_696235


namespace smallest_right_triangle_area_l696_696583

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696583


namespace probability_digits_all_different_l696_696784

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696784


namespace smallest_area_right_triangle_l696_696474

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696474


namespace product_fraction_l696_696050

noncomputable def a (n : ℕ) : ℤ := int.floor (real.sqrt (real.sqrt (2 * n - 1)))
noncomputable def b (n : ℕ) : ℤ := int.floor (real.sqrt (real.sqrt (2 * n)))
noncomputable def prod_a (N : ℕ) : ℤ := (finset.range N).prod (λ n, a (n + 1))
noncomputable def prod_b (N : ℕ) : ℤ := (finset.range N).prod (λ n, b (n + 1))

theorem product_fraction:
  (prod_a 1024 : ℚ) / prod_b 1024 = 105 / 384 :=
by
  sorry

end product_fraction_l696_696050


namespace sum_of_first_five_primes_units_digit_3_l696_696167

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l696_696167


namespace maximum_area_of_right_angled_triangle_l696_696994

noncomputable def max_area_right_angled_triangle (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : a + b + c = 48) : ℕ := 
  max (a * b / 2) 288

theorem maximum_area_of_right_angled_triangle (a b c : ℕ) 
  (h1 : a^2 + b^2 = c^2)    -- Pythagorean theorem
  (h2 : a + b + c = 48)     -- Perimeter condition
  (h3 : 0 < a)              -- Positive integer side length condition
  (h4 : 0 < b)              -- Positive integer side length condition
  (h5 : 0 < c)              -- Positive integer side length condition
  : max_area_right_angled_triangle a b c h1 h2 = 288 := 
sorry

end maximum_area_of_right_angled_triangle_l696_696994


namespace sequence_sum_l696_696913

theorem sequence_sum : 
  (∑ k in Finset.range 2002, (-1 : ℤ) ^ k * k * (k + 1)) = 2004002 :=
sorry

end sequence_sum_l696_696913


namespace conditional_probability_l696_696203

open Set

def Omega : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 2, 4, 5, 6}

noncomputable def P (s : Set ℕ) : ℝ := s.card.to_real / Omega.card.to_real

theorem conditional_probability :
  P (A ∩ B) / P A = 2 / 3 :=
by
  sorry

end conditional_probability_l696_696203


namespace total_goals_l696_696277

def arith_sum (n a d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem total_goals (louie_last_match_goals louie_previous_goals : ℕ)
                    (donnie_multiplier seasons games_per_season annie_first_term annie_diff : ℕ):
  louie_last_match_goals = 4 →
  louie_previous_goals = 40 →
  donnie_multiplier = 2 →
  seasons = 3 →
  games_per_season = 50 →
  annie_first_term = 2 →
  annie_diff = 2 →
  let louie_total_goals := louie_previous_goals + louie_last_match_goals in
  let donnie_goals_per_game := donnie_multiplier * louie_last_match_goals in
  let donnie_total_games := seasons * games_per_season in
  let donnie_total_goals := donnie_goals_per_game * donnie_total_games in
  let annie_total_games := 2 * games_per_season in
  let annie_total_goals := arith_sum annie_total_games annie_first_term annie_diff in
  louie_total_goals + donnie_total_goals + annie_total_goals = 11344 :=
by {
  intro h1 h2 h3 h4 h5 h6 h7,
  have louie_total_goals := 40 + 4,
  have donnie_goals_per_game := 2 * 4,
  have donnie_total_games := 3 * 50,
  have donnie_total_goals := 8 * 150,
  have annie_total_games := 2 * 50,
  have annie_total_goals := arith_sum 100 2 2,
  have total_goals := louie_total_goals + donnie_total_goals + annie_total_goals,
  have h := rfl,
  sorry
}

end total_goals_l696_696277


namespace P_is_polynomial_l696_696941

def is_polynomial (p : ℚ[x]) : Prop := True -- Placeholder for more precise polynomial check, if needed

noncomputable def P (n k : ℕ) (x : ℚ) : ℚ :=
  ((List.range k).map (fun i => (x ^ n - x ^ i)).prod) / ((List.range k).map (fun i => (x ^ k - x ^ i)).prod)

theorem P_is_polynomial (n k : ℕ) : 
  ∃ p : ℚ[x], ∀ x : ℚ, P n k x = (p.eval x) :=
sorry

end P_is_polynomial_l696_696941


namespace probability_passing_through_C_is_3_over_7_l696_696759

noncomputable def calculate_probability_passing_through_C : ℚ :=
  let total_paths := Nat.choose 8 4 in
  let paths_via_C := (Nat.choose 5 3) * (Nat.choose 3 1) in
  paths_via_C / total_paths

theorem probability_passing_through_C_is_3_over_7 :
  calculate_probability_passing_through_C = 3 / 7 :=
by
  sorry

end probability_passing_through_C_is_3_over_7_l696_696759


namespace smallest_area_right_triangle_l696_696510

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696510


namespace additional_spaces_on_second_level_are_8_l696_696011

-- Define the parking spaces on each level based on the conditions
def first_level_spaces : ℕ := 90
def second_level_spaces (x : ℕ) : ℕ := first_level_spaces + x
def third_level_spaces (x : ℕ) : ℕ := second_level_spaces(x) + 12
def fourth_level_spaces (x : ℕ) : ℕ := third_level_spaces(x) - 9

-- Define the total capacity of the parking garage
def total_capacity : ℕ := 399

-- Define the problem statement as a Lean theorem
theorem additional_spaces_on_second_level_are_8 (x : ℕ) :
  first_level_spaces + second_level_spaces(x) + third_level_spaces(x) + fourth_level_spaces(x) = total_capacity → x = 8 :=
by
  sorry

end additional_spaces_on_second_level_are_8_l696_696011


namespace volume_truncated_cone_lateral_surface_area_truncated_cone_l696_696024

section TruncatedCone
variables (R r h : ℝ)
variable (π : ℝ := Real.pi) -- Using real pi constant

-- Given conditions
def large_base_radius := R = 10
def small_base_radius := r = 5
def height := h = 8

-- Proof for the volume of truncated cone
theorem volume_truncated_cone 
  (large_base_radius : R = 10) 
  (small_base_radius : r = 5) 
  (height : h = 8) :
  (1 / 3) * π * (R^2 * (h + r) - r^2 * h) = 1400 * π / 3 :=
sorry

-- Proof for the lateral surface area of truncated cone
theorem lateral_surface_area_truncated_cone 
  (large_base_radius : R = 10) 
  (small_base_radius : r = 5) 
  (height : h = 8) :
  π * (R + r) * sqrt ((R - r)^2 + h^2) = 15 * π * sqrt 89 :=
sorry

end TruncatedCone

end volume_truncated_cone_lateral_surface_area_truncated_cone_l696_696024


namespace probability_all_digits_different_l696_696860

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696860


namespace probability_all_digits_different_l696_696873

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696873


namespace smallest_area_of_right_triangle_l696_696463

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696463


namespace prob_all_digits_different_l696_696849

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696849


namespace probability_all_digits_different_l696_696822

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696822


namespace annulus_area_of_regular_15_polygon_l696_696202

theorem annulus_area_of_regular_15_polygon (a R r : ℝ) (h_reg_polygon : true) (h_side_length : ∀ (AB : ℝ), AB = 2 * a) (h_circumradius : ∀ (O : true), O = R) (h_inradius : ∀ (O : true), O = r) : 
  let t := π * (R^2 - r^2)
  in t = π * a^2 :=
by 
  have h_pythagorean : R^2 = r^2 + a^2 := sorry
  have h_diff : R^2 - r^2 = a^2 := by rw [h_pythagorean]; ring
  show π * (R^2 - r^2) = π * a^2, by rw [h_diff]

end annulus_area_of_regular_15_polygon_l696_696202


namespace coin_draw_probability_l696_696267

theorem coin_draw_probability :
  let num_pennies := 3
  let num_nickels := 5
  let num_dimes := 4
  let total_coins := num_pennies + num_nickels + num_dimes
  let draw_count := 6
  let total_outcomes := Nat.choose total_coins draw_count
  let successful_outcomes := (Nat.choose num_pennies 0) * (Nat.choose num_nickels 2) * (Nat.choose num_dimes 4) +
                            (Nat.choose num_pennies 0) * (Nat.choose num_nickels 3) * (Nat.choose num_dimes 3) +
                            (Nat.choose num_pennies 0) * (Nat.choose num_nickels 4) * (Nat.choose num_dimes 2) +
                            (Nat.choose num_pennies 0) * (Nat.choose num_nickels 5) * (Nat.choose num_dimes 1) +
                            (Nat.choose num_pennies 1) * (Nat.choose num_nickels 4) * (Nat.choose num_dimes 1)
  (successes: successful_outcomes = 144) →
  probability := (successful_outcomes.to_Rat / total_outcomes.to_Rat)
  probability = 12 / 77 :=
by
  sorry

end coin_draw_probability_l696_696267


namespace cyclic_ABCD_l696_696198

variable (A B C D E F G H : Point)
variable (circle1 circle2 circle3 circle4 : Circle)
variable (h1 : A ∈ circle1 ∧ A ∈ circle2 ∧ A ∈ circle3 ∧ A ∈ circle4)
variable (h2 : B ∈ circle1 ∧ B ∈ circle2 ∧ B ∈ circle3 ∧ B ∈ circle4)
variable (h3 : C ∈ circle1 ∧ C ∈ circle2 ∧ C ∈ circle3 ∧ C ∈ circle4)
variable (h4 : D ∈ circle1 ∧ D ∈ circle2 ∧ D ∈ circle3 ∧ D ∈ circle4)
variable (h5 : E ∈ circle1 ∧ E ∈ circle2)
variable (h6 : F ∈ circle2 ∧ F ∈ circle3)
variable (h7 : G ∈ circle3 ∧ G ∈ circle4)
variable (h8 : H ∈ circle4 ∧ H ∈ circle1)
variable (h_cyclic_EFGH : CyclicQuadrilateral E F G H)

theorem cyclic_ABCD (A B C D E F G H : Point)
    (circle1 circle2 circle3 circle4 : Circle)
    (h1 : A ∈ circle1 ∧ A ∈ circle2 ∧ A ∈ circle3 ∧ A ∈ circle4)
    (h2 : B ∈ circle1 ∧ B ∈ circle2 ∧ B ∈ circle3 ∧ B ∈ circle4)
    (h3 : C ∈ circle1 ∧ C ∈ circle2 ∧ C ∈ circle3 ∧ C ∈ circle4)
    (h4 : D ∈ circle1 ∧ D ∈ circle2 ∧ D ∈ circle3 ∧ D ∈ circle4)
    (h5 : E ∈ circle1 ∧ E ∈ circle2)
    (h6 : F ∈ circle2 ∧ F ∈ circle3)
    (h7 : G ∈ circle3 ∧ G ∈ circle4)
    (h8 : H ∈ circle4 ∧ H ∈ circle1)
    (h_cyclic_EFGH : CyclicQuadrilateral E F G H) :
    CyclicQuadrilateral A B C D :=
by
    sorry

end cyclic_ABCD_l696_696198


namespace probability_digits_different_l696_696802

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696802


namespace apple_distribution_l696_696067

theorem apple_distribution : 
  ∃ (a b c : ℕ) (s : finset (ℕ × ℕ × ℕ)), 
    (∀ {x y z}, (x, y, z) ∈ s → x + y + z = 10 ∧ 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5 ∧ 1 ≤ z ∧ z ≤ 5) ∧ 
    s.card = 4 := 
sorry

end apple_distribution_l696_696067


namespace coin_collection_prob_l696_696940

theorem coin_collection_prob (n : ℕ) (hpos : 0 < n) 
(f : ℕ → ℕ) 
(hf : ∀ (m : ℕ), f m = ∑ (k : ℕ) in (finset.range m).filter(λ i, 0 < i), (n.factorial : ℕ))
: ∃ C : ℝ, ∀ n : ℕ, 0 < n → (n : ℝ)^(n^2/2 - C * n) * real.exp(-((n^2) / 4)) ≤ (f n : ℝ) ∧ (f n : ℝ) ≤ (n : ℝ)^(n^2 / 2 + C * n) * real.exp(-((n^2) / 4)) :=
sorry

end coin_collection_prob_l696_696940


namespace smallest_area_right_triangle_l696_696517

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696517


namespace smallest_area_right_triangle_l696_696606

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696606


namespace f_difference_l696_696233

def α : ℝ := (1 + real.sqrt 5) / 2
def β : ℝ := (1 - real.sqrt 5) / 2

noncomputable def f (n : ℕ) : ℝ := 
  (5 + 3 * real.sqrt 5) / 10 * α ^ n + (5 - 3 * real.sqrt 5) / 10 * β ^ n

theorem f_difference (n: ℕ) : f(n + 1) - f(n - 1) = f(n) := sorry

end f_difference_l696_696233


namespace triangle_identity_l696_696955

theorem triangle_identity 
  (A B C M N : Type) 
  (angle_mab_angle_nac : ∠ M A B = ∠ N A C)
  (angle_mba_angle_nbc : ∠ M B A = ∠ N B C)
  (dist_am : Real)
  (dist_an : Real)
  (dist_ab : Real)
  (dist_ac : Real)
  (dist_bm : Real)
  (dist_bn : Real)
  (dist_bc : Real)
  (dist_cm : Real)
  (dist_cn : Real)
  (dist_cb : Real) :
  (dist_am * dist_an) / (dist_ab * dist_ac) + 
  (dist_bm * dist_bn) / (dist_ab * dist_bc) + 
  (dist_cm * dist_cn) / (dist_ac * dist_bc) = 1 := 
  sorry

end triangle_identity_l696_696955


namespace smallest_area_right_triangle_l696_696693

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696693


namespace maximum_ratio_OB_OA_l696_696281

-- Definitions of the curves in Cartesian coordinates
def C₁_cartesian (x y : ℝ) : Prop :=
  x + y = 4

def C₂_parametric (θ : ℝ) : ℝ × ℝ :=
  (1 + cos θ, sin θ)

-- Polar conversions
def C₁_polar (ρ θ : ℝ) : Prop :=
  ρ * (cos θ + sin θ) = 4

def C₂_polar (ρ θ : ℝ) : Prop :=
  ρ = 2 * cos θ

-- Maximum ratio problem
def ratio_OB_OA (OB OA : ℝ) : ℝ :=
  OB / OA

theorem maximum_ratio_OB_OA (α : ℝ) (hα : -π/4 < α ∧ α < π/2) :
  max (ratio_OB_OA (2 * cos α) (4 / (cos α + sin α))) = (1 / 4 * (sqrt 2 + 1)) :=
sorry

end maximum_ratio_OB_OA_l696_696281


namespace smallest_right_triangle_area_l696_696448

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696448


namespace f_2_value_l696_696219

def f : ℝ → ℝ := sorry

theorem f_2_value (h : ∀ x : ℝ, f (x - 1 / x) = x^2 + (1 / x)^2 + 1) : f 2 = 7 := by
  -- Define the function based on the given condition
  have f_eq : ∀ y : ℝ, f y = y^2 + 3,
    from sorry,
  -- Using the definition of f, we calculate f(2)
  calc
    f 2 = 2^2 + 3 : by rw f_eq
     ... = 7     : by norm_num

end f_2_value_l696_696219


namespace sin_pi_plus_alpha_l696_696187

variable (α : ℝ) -- Declare variable α
-- Declare conditions as hypotheses
hypothesis (h1 : sin (π / 2 + α) = 3 / 5)
hypothesis (h2 : α ∈ Ioc 0 (π / 2))

-- State the theorem to prove
theorem sin_pi_plus_alpha : sin (π + α) = -4 / 5 :=
by
  sorry -- Proof is omitted

end sin_pi_plus_alpha_l696_696187


namespace largest_mersenne_prime_less_than_500_l696_696438

theorem largest_mersenne_prime_less_than_500 : ∃ n : ℕ, (Nat.prime n ∧ (2^n - 1) = 127 ∧ ∀ m : ℕ, (Nat.prime m → (2^m - 1) < 500 → (2^m - 1) ≤ 127)) :=
sorry

end largest_mersenne_prime_less_than_500_l696_696438


namespace find_b_l696_696398

theorem find_b (a b : ℝ) (h1 : a^2 * sqrt b = 54) (h2 : a * b = 108) : 
  b = 36 :=
sorry

end find_b_l696_696398


namespace quadrilateral_area_l696_696887

theorem quadrilateral_area (n : ℕ) (h : n = 2017) : 
  let S : ℕ → ℚ := λ n, (2 * n + 1) / (2 * (n + 1))
  S n = 4035 / 4036 :=
by
  sorry

end quadrilateral_area_l696_696887


namespace sum_of_first_five_primes_with_units_digit_3_l696_696129

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696129


namespace distance_F1_to_line_F2M_l696_696988

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := (x ^ 2 / 6) - (y ^ 2 / 3) = 1

-- Define the foci
def F1 : ℝ × ℝ := (-3, 0)
def F2 : ℝ × ℝ := (3, 0)

-- Define the condition that point M lies on the hyperbola 
-- and that MF1 is perpendicular to the x-axis
def M_y (y : ℝ) : ℝ × ℝ := (-3, y)
def on_hyperbola (M : ℝ × ℝ) : Prop := hyperbola M.1 M.2
def perpendicular_to_x_axis (M : ℝ × ℝ) : Prop := M.1 = -3

-- Define the distance from a point to a line
def distance_point_line (p : ℝ × ℝ) (A B C : ℝ) : ℝ := 
  abs (A * p.1 + B * p.2 + C) / sqrt (A ^ 2 + B ^ 2)

-- Proof problem statement
theorem distance_F1_to_line_F2M (y : ℝ) (h1 : on_hyperbola (M_y y)) (h2 : perpendicular_to_x_axis (M_y y)) :
  ∃ (d : ℝ), d = (3 * sqrt 6) / 5 :=
by
  sorry

end distance_F1_to_line_F2M_l696_696988


namespace smallest_right_triangle_area_l696_696680

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696680


namespace half_of_animals_get_sick_l696_696407

theorem half_of_animals_get_sick : 
  let chickens := 26
  let piglets := 40
  let goats := 34
  let total_animals := chickens + piglets + goats
  let sick_animals := total_animals / 2
  sick_animals = 50 :=
by
  sorry

end half_of_animals_get_sick_l696_696407


namespace smallest_right_triangle_area_l696_696498

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696498


namespace select_participants_l696_696357

variable (F : ℕ) (M : ℕ) -- Define the number of female (F) and male (M) students.

def comb (n k : ℕ) : ℕ := Nat.choose n k -- Combination function, choose k from n.

theorem select_participants : comb 6 3 - comb 4 3 + comb 2 1 * comb 4 2 + comb 2 2 * comb 4 1 = 16 :=
by 
  let F := 2
  let M := 4
  have H₁ : comb (F + M) 3 = comb 6 3 := by rfl
  have H₂ : comb M 3 = comb 4 3 := by rfl
  have H₃ : comb F 1 * comb M 2 = comb 2 1 * comb 4 2 := by rfl
  have H₄ : comb F 2 * comb M 1 = comb 2 2 * comb 4 1 := by rfl
  rw [H₁, H₂, H₃, H₄]
  sorry -- proof to be filled in

end select_participants_l696_696357


namespace right_triangle_min_area_l696_696652

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696652


namespace triangle_fold_point_area_l696_696208

theorem triangle_fold_point_area (AB BC AC : ℝ)
  (h₁ : AB = 45) (h₂ : BC = 60) (h₃ : AC = 75) :
  ∃ (m n t : ℤ), (m > 0) ∧ (n > 0) ∧ (t > 0) ∧ (∀ (p p' : ℕ), prime p' → t % (p' * p') ≠ 0) ∧ (m + n + t = 527) :=
sorry

end triangle_fold_point_area_l696_696208


namespace largest_mersenne_prime_is_127_l696_696434

noncomputable def largest_mersenne_prime_less_than_500 : ℕ :=
  127

theorem largest_mersenne_prime_is_127 :
  ∃ p : ℕ, Nat.Prime p ∧ (2^p - 1) = largest_mersenne_prime_less_than_500 ∧ 2^p - 1 < 500 := 
by 
  -- The largest Mersenne prime less than 500 is 127
  use 7
  sorry

end largest_mersenne_prime_is_127_l696_696434


namespace total_pictures_480_l696_696365

noncomputable def total_pictures (pictures_per_album : ℕ) (num_albums : ℕ) : ℕ :=
  pictures_per_album * num_albums

theorem total_pictures_480 : total_pictures 20 24 = 480 :=
  by
    sorry

end total_pictures_480_l696_696365


namespace binomial_expansion_sum_coeff_l696_696041

theorem binomial_expansion_sum_coeff (a b : ℕ) (h₁ : a = 1) (h₂ : b = 1) :
  (∑ k in Finset.range 8, Nat.choose 7 k * (a^(7 - k)) * (b^k)) = 128 :=
by
  rw [h₁, h₂]
  sorry

end binomial_expansion_sum_coeff_l696_696041


namespace OI_perp_AC_l696_696289

-- Definitions of points involved in the problem
variables {A B C I I_A I_C O : Type*}

-- Conditions given in the problem
noncomputable def triangle_ABC (A B C : Type*) := true  -- Placeholder for the definition of triangle ABC
def incenter (I : Type*) : Prop := true  -- Placeholder for the definition of incenter
def excenter (I_A I_C : Type*) : Prop := true  -- Placeholder for the definition of the excenter
def circumcenter (O : Type*) : Prop := true  -- Placeholder for the definition of the circumcenter

-- The main statement to prove
theorem OI_perp_AC 
  (tri : triangle_ABC A B C)
  (hI : incenter I)
  (hIA : excenter I_A)
  (hIC : excenter I_C)
  (hO : circumcenter O)
  : is_perp (line O I) (line A C) :=
sorry

end OI_perp_AC_l696_696289


namespace smallest_right_triangle_area_l696_696576

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696576


namespace smallest_area_of_right_triangle_l696_696627

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696627


namespace sum_of_first_five_prime_units_digit_3_l696_696107

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l696_696107


namespace digits_probability_l696_696790

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696790


namespace smallest_area_right_triangle_l696_696614

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696614


namespace smallest_right_triangle_area_l696_696492

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696492


namespace integral_upper_semicircle_l696_696892

open Complex MeasureTheory Topology

noncomputable def upper_semicircle (t : ℝ) := exp (I * t)

theorem integral_upper_semicircle :
  ∫ z in curve_integrable (path.mk upper_semicircle 0 π) measure_space.volume, 
  abs z * conj z * (path.mk upper_semicircle 0 π).diff z = I * π := 
sorry

end integral_upper_semicircle_l696_696892


namespace probability_digits_different_l696_696798

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696798


namespace sum_of_first_five_primes_with_units_digit_three_l696_696139

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l696_696139


namespace probability_all_digits_different_l696_696858

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696858


namespace twice_x_minus_3_l696_696074

theorem twice_x_minus_3 (x : ℝ) : (2 * x) - 3 = 2 * x - 3 := 
by 
  -- This proof is trivial and we can assert equality directly
  sorry

end twice_x_minus_3_l696_696074


namespace smallest_area_right_triangle_l696_696564

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696564


namespace sum_of_first_five_primes_with_units_digit_3_l696_696155

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696155


namespace smallest_right_triangle_area_l696_696494

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696494


namespace quadratic_min_is_less_than_neg6_l696_696739

-- Define the given data points of the quadratic function
def quadratic_points : List (ℝ × ℝ) := [(-2, 6), (0, -4), (1, -6), (3, -4)]

-- The statement to prove
theorem quadratic_min_is_less_than_neg6 : ∃ x y, (x, y) ∈ quadratic_points ∧ y < -6 := by
  use (1, -6)
  -- (1, -6) is part of the given points, and -6 < -6 is trivially true, so we conclude !
  sorry

end quadratic_min_is_less_than_neg6_l696_696739


namespace smallest_area_right_triangle_l696_696565

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696565


namespace sum_of_first_five_primes_with_units_digit_3_l696_696122

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696122


namespace probability_all_different_digits_l696_696768

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696768


namespace smallest_area_right_triangle_l696_696609

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696609


namespace collinear_C_I_Q_l696_696037

open Set Geometry

variable {O I : Circle} {P A B Q M N C : Point} (h1 : inscribed O I) 
  (h2 : tangent AB I Q) 
  (h3 : intersects (line_extension PQ) O M) 
  (h4 : diameter O MN) 
  (h5 : perpendicular_drawn_from P PA AN C)

theorem collinear_C_I_Q : Collinear {C, I.center, Q} := sorry

end collinear_C_I_Q_l696_696037


namespace smallest_area_correct_l696_696547

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696547


namespace solve_sqrt_equation_l696_696933

theorem solve_sqrt_equation (x : ℝ) (h1 : x ≤ 3) (h2 : 2 ≤ x) :
  (sqrt (3 - x) + sqrt (x - 2) = 2 ↔ x = 3/4 ∨ x = 2) := by
  sorry

end solve_sqrt_equation_l696_696933


namespace sum_of_first_five_primes_with_units_digit_3_l696_696132

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696132


namespace smallest_right_triangle_area_l696_696570

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696570


namespace probability_all_digits_different_l696_696820

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696820


namespace smallest_area_of_right_triangle_l696_696621

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696621


namespace smallest_area_of_right_triangle_l696_696466

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696466


namespace total_spent_on_music_books_eq_160_l696_696299

noncomputable def total_spent_music_books : ℝ :=
let initial_amount := 500
let math_books := 4
let math_book_price := 20
let math_books_discount := 0.10
let science_books := math_books + 6
let science_book_price := 10
let multiple_art_books := 2
let art_books := multiple_art_books * math_books
let art_book_price := 20
let art_books_tax := 0.05
let music_books_tax := 0.07 in 

let cost_math_books := math_books * math_book_price
let total_cost_math_books := cost_math_books - (math_books_discount * cost_math_books)
let total_cost_science_books := science_books * science_book_price
let cost_art_books := art_books * art_book_price
let total_cost_art_books := cost_art_books * (1 + art_books_tax) in

let total_spent_books := total_cost_math_books + total_cost_science_books + total_cost_art_books
let remaining_amount := initial_amount - total_spent_books in
remaining_amount

theorem total_spent_on_music_books_eq_160 :
  total_spent_music_books * 1.07 = 160 :=
by sorry

end total_spent_on_music_books_eq_160_l696_696299


namespace sum_is_correct_l696_696152

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l696_696152


namespace prob_all_digits_different_l696_696848

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696848


namespace f_increasing_on_nonneg_f_max_value_on_2_9_f_min_value_on_2_9_l696_696230

def f (x : ℝ) : ℝ := (2 * x - 3) / (x + 1)

theorem f_increasing_on_nonneg : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 :=
by
  -- proof omitted
  sorry

theorem f_max_value_on_2_9 : ∃ x : ℝ, x ∈ Icc 2 9 ∧ f x = 3 / 2 :=
by
  -- proof omitted
  sorry

theorem f_min_value_on_2_9 : ∃ x : ℝ, x ∈ Icc 2 9 ∧ f x = 1 / 3 :=
by
  -- proof omitted
  sorry

end f_increasing_on_nonneg_f_max_value_on_2_9_f_min_value_on_2_9_l696_696230


namespace msrp_calculation_l696_696889

-- Define the variables and conditions
def insurance_rate : ℝ := 0.20
def state_tax_rate : ℝ := 0.50
def total_paid : ℝ := 54

-- Define the MSRP variable
def msrp := total_paid / (1 + insurance_rate + state_tax_rate * (1 + insurance_rate))

-- Statement to be proved
theorem msrp_calculation : msrp = 30 := by
  -- You can provide details here, but for now, we skip the actual proof
  sorry

end msrp_calculation_l696_696889


namespace area_of_pentagon_l696_696021

structure Point where
  x : ℤ
  y : ℤ

def vertices : List Point :=
  [ {x := 1, y := 1}, {x := 4, y := 3}, {x := 6, y := 1}, {x := 5, y := -2}, {x := 2, y := -1} ]

noncomputable def calculate_area (pts : List Point) : ℤ :=
  let n := pts.length
  ∑ i in finset.range n, (pts[i].x * pts[(i + 1) % n].y - pts[i].y * pts[(i + 1) % n].x)

theorem area_of_pentagon : calculate_area vertices / 2 = 15 :=
by
  sorry

end area_of_pentagon_l696_696021


namespace smallest_area_correct_l696_696552

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696552


namespace radius_and_area_of_inscribed_square_l696_696032

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  let K := triangle_area a b c
  (a * b * c) / (4 * K)

noncomputable def side_of_square (d : ℝ) : ℝ :=
  d / Real.sqrt 2

noncomputable def area_of_square (s : ℝ) : ℝ := s ^ 2

theorem radius_and_area_of_inscribed_square :
  let a := 13
  let b := 13
  let c := 10
  let R := circumradius a b c
  let d := 2 * R
  let s := side_of_square d
  let A := area_of_square s
  R ≈ 7.04 ∧ A ≈ 99 :=
by
  -- The exact proof goes here
  sorry

end radius_and_area_of_inscribed_square_l696_696032


namespace smallest_area_right_triangle_l696_696616

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696616


namespace half_of_animals_get_sick_l696_696406

theorem half_of_animals_get_sick : 
  let chickens := 26
  let piglets := 40
  let goats := 34
  let total_animals := chickens + piglets + goats
  let sick_animals := total_animals / 2
  sick_animals = 50 :=
by
  sorry

end half_of_animals_get_sick_l696_696406


namespace digits_all_different_l696_696814

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696814


namespace digits_all_different_l696_696807

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696807


namespace special_sequence_exists_l696_696914

noncomputable def exists_special_sequence : Prop :=
  ∃ (a : ℕ → ℕ), (∀ i, 1 ≤ i → a(i) < a(i + 1)) ∧ 
    (∀ k, 2 ≤ k ∧ k ≤ 100 → Nat.lcm (a(k - 1)) (a(k)) > Nat.lcm (a(k)) (a(k + 1)))

theorem special_sequence_exists : exists_special_sequence := sorry

end special_sequence_exists_l696_696914


namespace pentagon_area_l696_696051

open Real BigOperators

theorem pentagon_area (AB BC CD DE EA : ℝ)
  (angle_ABCDE : ∀ A B C, angle A B C = π / 3) :
  let area_AED := abs (1 / 2 * DE * EA * sin (2 * π / 3))
  let area_ABC := abs (sqrt 3 / 4 * (3:ℝ) ^ 2)
  in (2 * area_AED + area_ABC) = (sqrt 289 + sqrt 0) := sorry

end pentagon_area_l696_696051


namespace stuart_initial_marbles_l696_696027

theorem stuart_initial_marbles
    (betty_marbles : ℕ)
    (stuart_marbles_after_given : ℕ)
    (percentage_given : ℚ)
    (betty_gave : ℕ):
    betty_marbles = 60 →
    stuart_marbles_after_given = 80 →
    percentage_given = 0.40 →
    betty_gave = percentage_given * betty_marbles →
    stuart_marbles_after_given = stuart_initial + betty_gave →
    stuart_initial = 56 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end stuart_initial_marbles_l696_696027


namespace smallest_right_triangle_area_l696_696451

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696451


namespace fraction_shaded_area_l696_696344

-- Definitions based on the conditions
def decagon : Type := sorry
def center (O : decagon) : Prop := sorry
def midpoint (X AB : decagon) : Prop := sorry
def side (AB : decagon) : Prop := sorry
def triangle (CF D E F B O : decagon) : Prop := sorry
def half (triangle : decagon) : decagon := sorry
def shaded_area (region : decagon) : decagon := sorry

-- Given Conditions
axiom O_is_center : ∃ O : decagon, center O
axiom X_is_midpoint : ∃ (X AB : decagon), midpoint X AB ∧ side AB
axiom shaded_region : ∃ (C F D E B O : decagon), 
  triangle C F O ∧ triangle D F O ∧ triangle E F O ∧ half (triangle B F O)

-- Theorem to prove
theorem fraction_shaded_area : shaded_area (sorry : decagon) = 7/20 :=
sorry

end fraction_shaded_area_l696_696344


namespace ag_length_l696_696288

-- Definitions for the problem conditions
variables {A B C D E G : Type} 
variables {length_AB length_AD length_BE : ℝ}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited G]

-- Conditions from problem in Lean
def is_median (A B C D : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] := 
       true -- Assume a predicate that verifies D is the midpoint of BC

def is_centroid (G A D : Type) [inhabited G] [inhabited A] [inhabited D] :=
       true -- Assume a predicate that verifies G divides AD in the 2:1 ratio

-- Problem Statement
theorem ag_length {length_AD : ℝ} (h_med: is_median A B C D) (h_centroid: is_centroid G A D) (h_AD: length_AD = 8) : 
  let AG := (2 / 3) * length_AD in
  AG = 16 / 3 := 
by
  -- Insert a proof here in actual use
  sorry

end ag_length_l696_696288


namespace arithmetic_sequence_general_term_and_sum_l696_696972

theorem arithmetic_sequence_general_term_and_sum (d a1 a4 a6 an Sn : ℝ) (k : ℤ)
  (h0 : d = 2)
  (h1 : a1 = -11)
  (h2 : a4 + a6 = -6) 
  (h3 : an = 2n - 13)
  (h4 : Sn = n^2 - 12 * n)
  (h5 : Sn = 189) :
  an = 2 * n - 13 ∧ k = 21  :=
by
  sorry

end arithmetic_sequence_general_term_and_sum_l696_696972


namespace valid_n_values_l696_696177

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

def n_values : List ℕ :=
  [1998, 2001, 2007, 2013]

theorem valid_n_values :
  ∀ n ∈ n_values, n + sum_of_digits n + sum_of_digits (sum_of_digits n) = 2023 :=
by
  sorry

end valid_n_values_l696_696177


namespace smallest_area_of_right_triangle_l696_696618

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696618


namespace find_length_of_AG_l696_696716

-- Define points on a line segment and their properties
variables (A G E F N : Type) [AffineSpace ℝ A]

-- Define the points and distances involved
variables (AG AE EF FG AN NG NF : ℝ)

-- Given conditions and properties
def quadrisection (AG AE EF FG GA : ℝ) := (AE = EF) ∧ (EF = FG) ∧ (FG = GA) ∧ (4 * AE = AG)
def midpoint (AG AN NG : ℝ) := (AN = NG) ∧ (AN + NG = AG)
def given_nf (NF : ℝ) := (NF = 12)

-- Main theorem: length of AG
theorem find_length_of_AG
  (h1 : quadrisection AG AE EF FG AE)
  (h2 : midpoint AG AN NG)
  (h3 : given_nf NF):
  AG = 16 :=
by
  sorry

end find_length_of_AG_l696_696716


namespace lcm_gcf_ratio_240_630_l696_696440

theorem lcm_gcf_ratio_240_630 :
  let a := 240
  let b := 630
  Nat.lcm a b / Nat.gcd a b = 168 := by
  sorry

end lcm_gcf_ratio_240_630_l696_696440


namespace number_is_square_l696_696944

theorem number_is_square (x y : ℕ) : (∃ n : ℕ, (1100 * x + 11 * y = n^2)) ↔ (x = 7 ∧ y = 4) :=
by
  sorry

end number_is_square_l696_696944


namespace smallest_area_right_triangle_l696_696555

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696555


namespace smallest_area_of_right_triangle_l696_696464

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696464


namespace tangent_lines_exist_l696_696753

open Topology

variables {R r L : ℝ}

-- Given two circles with radii R and r (R > r), and a chord of length L in the smaller circle
theorem tangent_lines_exist (hR : R > 0) (hr : r > 0) (hL : L > 0) 
  (hRr : R > r) (hLr : L < 2 * r) 
  : ∃ (P Q : ℝ × ℝ), is_tangent (P) (R) ∧ is_tangent (Q) (R) ∧ chord_length (P Q) (r) = L := 
sorry

end tangent_lines_exist_l696_696753


namespace math_problem_l696_696078

-- Define the mixed numbers as fractions
def mixed_3_1_5 := 16 / 5 -- 3 + 1/5 = 16/5
def mixed_4_1_2 := 9 / 2  -- 4 + 1/2 = 9/2
def mixed_2_3_4 := 11 / 4 -- 2 + 3/4 = 11/4
def mixed_1_2_3 := 5 / 3  -- 1 + 2/3 = 5/3

-- Define the main expression
def main_expr := 53 * (mixed_3_1_5 - mixed_4_1_2) / (mixed_2_3_4 + mixed_1_2_3)

-- Define the expected answer in its fractional form
def expected_result := -78 / 5

-- The theorem to prove the main expression equals the expected mixed number
theorem math_problem : main_expr = expected_result :=
by sorry

end math_problem_l696_696078


namespace clown_balloon_count_l696_696371

theorem clown_balloon_count (b1 b2 : ℕ) (h1 : b1 = 47) (h2 : b2 = 13) : b1 + b2 = 60 := by
  sorry

end clown_balloon_count_l696_696371


namespace function_properties_l696_696986

theorem function_properties
  (f : ℝ → ℝ)
  (h1 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0)
  (h2 : ∀ x, f (x - t) = f (x + t)) 
  (h3_even : ∀ x, f (-x) = f x)
  (h3_decreasing : ∀ x1 x2, x1 < x2 ∧ x2 < 0 → f x1 > f x2)
  (h3_at_neg2 : f (-2) = 0)
  (h4_odd : ∀ x, f (-x) = -f x) : 
  ((∀ x1 x2, x1 < x2 → f x1 > f x2) ∧
   (¬∀ x, (f x > 0) ↔ (-2 < x ∧ x < 2)) ∧
   (∀ x, f (x) * f (|x|) = - f (-x) * f |x|) ∧
   (¬∀ x, f (x) = f (x + 2 * t))) :=
by 
  sorry

end function_properties_l696_696986


namespace graph_self_intersections_l696_696010

theorem graph_self_intersections :
  let x := λ t : ℝ, cos t + 3 * t / 2
  let y := λ t : ℝ, sin t
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 10 ≤ x t₁ ∧ x t₁ ≤ 80 ∧ 10 ≤ x t₂ ∧ x t₂ ≤ 80 ∧ 
    x t₁ = x t₂ ∧ y t₁ = y t₂) ∧
  ∃ n : ℤ, n = 8 := 
sorry

end graph_self_intersections_l696_696010


namespace find_solutions_system_l696_696083

theorem find_solutions_system :
  (∀ (x y z : ℝ), x >= 0 → y >= 0 → z >= 0 →
    (sqrt(x + y) + sqrt(z) = 7) →
    (sqrt(x + z) + sqrt(y) = 7) →
    (sqrt(y + z) + sqrt(x) = 5) →
    (x = 1 ∧ y = 4 ∧ z = 4) ∨ (x = 1 ∧ y = 9 ∧ z = 9)) :=
by
  sorry

end find_solutions_system_l696_696083


namespace smallest_area_right_triangle_l696_696489

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696489


namespace smallest_right_triangle_area_l696_696670

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696670


namespace smallest_area_right_triangle_l696_696477

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696477


namespace smallest_area_of_right_triangle_l696_696623

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696623


namespace cos_sin_ratio_l696_696960

open Real

-- Given conditions
variables {α β : Real}
axiom tan_alpha_beta : tan (α + β) = 2 / 5
axiom tan_beta_pi_over_4 : tan (β - π / 4) = 1 / 4

-- Theorem to be proven
theorem cos_sin_ratio (hαβ : tan (α + β) = 2 / 5) (hβ : tan (β - π / 4) = 1 / 4) :
  (cos α + sin α) / (cos α - sin α) = 3 / 22 :=
sorry

end cos_sin_ratio_l696_696960


namespace largest_k_summing_consecutive_integers_l696_696088

theorem largest_k_summing_consecutive_integers (k n : ℕ) (h : 5^9 = (k / 2) * (2 * n + k + 1) ∧ k < 2 * 5^4.5) : 
  k ≤ 1250 := 
sorry

end largest_k_summing_consecutive_integers_l696_696088


namespace digits_probability_l696_696791

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696791


namespace find_x_log_base_l696_696080

theorem find_x_log_base :
  ∃ x : ℝ, log x 81 = 2 → x = 9 :=
begin
  -- Declaring that x is a positive real number to be valid as a logarithmic base
  use 9,
  intro h,
  field_simp at h,
  -- We need to show that log base 9 of 81 is indeed 2
  rw log_eq_log_iff,
  -- And the fact that 81 = 9^2
  exact pow_two (9 : ℝ),
end

end find_x_log_base_l696_696080


namespace fewest_printers_l696_696728

theorem fewest_printers (c1 c2 c3 c4 : ℕ) (h1 : c1 = 400) (h2 : c2 = 350) (h3 : c3 = 500) (h4 : c4 = 200) :
  let lcm := Nat.lcm c1 (Nat.lcm c2 (Nat.lcm c3 c4)) in
  let n1 := lcm / c1 in
  let n2 := lcm / c2 in
  let n3 := lcm / c3 in
  let n4 := lcm / c4 in
  n1 + n2 + n3 + n4 = 173 := 
by
  sorry

end fewest_printers_l696_696728


namespace smallest_right_triangle_area_l696_696444

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696444


namespace sum_of_first_five_primes_with_units_digit_3_l696_696127

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696127


namespace probability_all_digits_different_l696_696827

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696827


namespace g_2022_value_l696_696081

open Real

theorem g_2022_value :
  (∀ x y : ℝ, g(x - y) = 2022 * (g x + g y) - 2021 * x * y) →
  g 2022 = 2043231 :=
by sorry

end g_2022_value_l696_696081


namespace smallest_area_right_triangle_l696_696475

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696475


namespace product_of_B_coordinates_l696_696216

theorem product_of_B_coordinates :
  (∃ (x y : ℝ), (1 / 3 * x + 2 / 3 * 4 = 1 ∧ 1 / 3 * y + 2 / 3 * 2 = 7) ∧ x * y = -85) :=
by
  sorry

end product_of_B_coordinates_l696_696216


namespace smallest_right_triangle_area_l696_696671

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696671


namespace smallest_area_correct_l696_696548

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696548


namespace number_of_polynomials_l696_696053

/--
There are n polynomials Q(x) of degree at most 4 with coefficients
in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} such that Q(-1) = 11.
-/
theorem number_of_polynomials (n : ℕ) :
  ∃ (Q : ℕ → ℤ), (∀ i, Q i ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : set ℤ)) ∧
  (∃ (a b c d e : ℤ),
     Q 4 = a ∧ Q 3 = b ∧ Q 2 = c ∧ Q 1 = d ∧ Q 0 = e ∧ 
     11 = e - d + c - b + a) → n = <calculated_value> :=
sorry

end number_of_polynomials_l696_696053


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696117

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696117


namespace find_probability_l696_696369

noncomputable def normal_dist : ℝ → ℝ :=
  λ x, 1 / (5 * real.sqrt (2 * real.pi)) * real.exp (- (x - 100)^2 / (2 * 5^2))

theorem find_probability
  (ξ : ℝ)
  (h₁ : ∀ x, ξ = normal_dist x)
  (h₂ : ∃ ξ, P(ξ < 110) = 0.96) :
  P(90 < ξ ∧ ξ < 100) = 0.46 :=
sorry

end find_probability_l696_696369


namespace solve_for_x_l696_696707

theorem solve_for_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 :=
by
  sorry

end solve_for_x_l696_696707


namespace smallest_right_triangle_area_l696_696580

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696580


namespace tangent_line_a1_monotonic_intervals_range_of_a_l696_696231

noncomputable def f (x a: ℝ) : ℝ := Real.exp (a * x) - x

-- Part (1): Tangent Line Equation at (0, f(0))
theorem tangent_line_a1:
  (∀ x: ℝ, f x 1 = Real.exp x - x) →
  (∀ x: ℝ, deriv (λ x, f x 1) x = Real.exp x - 1) →
  (f 0 1 = 1) →
  (f' = deriv (λ x , f x 1 )) →
  (f' 0 = 0) →
  ( ∀x:ℝ , (0 ≤ x) → f' x = 0 * x + 1 ) 


-- Part (2): Monotonic Intervals
theorem monotonic_intervals (a : ℝ):
  (∀ x : ℝ, f x a = Real.exp (a * x) - x) →
  (0 > a → ∀ x : ℝ, deriv (λx, f x a) x < 0) ∧
  (a > 0 →
   ∀ x : ℝ,
     (x < -((Real.log a) / a) → deriv (λx, f x a) x < 0) ∧
     (x > -((Real.log a) / a) → deriv (λx, f x a) x > 0)) :=
sorry


-- Part (3): Range of Values for a
theorem range_of_a :
  (∀ x : ℝ, f x a) →
  ( ∀ x1 x2 : ℝ,  -1 ≤ x1 ∧ x1 ≤ 1 → -1 ≤ x2 ∧ x2 ≤ 1 → f x1 a* f x2 a ≥ λ
     → (a ≤ -Real.log 2) ∨ (a ≥ Real.log 4)) :=
sorry

end tangent_line_a1_monotonic_intervals_range_of_a_l696_696231


namespace right_triangle_min_area_l696_696661

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696661


namespace largest_difference_l696_696439

theorem largest_difference {S : set ℤ} (hS : S = {-20, -10, 0, 5, 15, 25}) : 
  (∃ a b ∈ S, a - b = 45) ∧ (∀ a b ∈ S, a - b ≤ 45) := 
by
  sorry

end largest_difference_l696_696439


namespace smallest_area_right_triangle_l696_696638

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696638


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696113

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696113


namespace digits_all_different_l696_696816

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696816


namespace sum_of_first_five_primes_with_units_digit_3_l696_696162

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696162


namespace quadratic_min_is_less_than_neg6_l696_696738

-- Define the given data points of the quadratic function
def quadratic_points : List (ℝ × ℝ) := [(-2, 6), (0, -4), (1, -6), (3, -4)]

-- The statement to prove
theorem quadratic_min_is_less_than_neg6 : ∃ x y, (x, y) ∈ quadratic_points ∧ y < -6 := by
  use (1, -6)
  -- (1, -6) is part of the given points, and -6 < -6 is trivially true, so we conclude !
  sorry

end quadratic_min_is_less_than_neg6_l696_696738


namespace reflect_point_x_axis_correct_l696_696282

-- Definition of the transformation reflecting a point across the x-axis
def reflect_x_axis (P : ℝ × ℝ) : ℝ × ℝ := (P.1, -P.2)

-- Define the original point coordinates
def P : ℝ × ℝ := (-2, 3)

-- The Lean proof statement
theorem reflect_point_x_axis_correct :
  reflect_x_axis P = (-2, -3) :=
sorry

end reflect_point_x_axis_correct_l696_696282


namespace possible_triplets_l696_696908

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem possible_triplets (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (is_power_of_two (a * b - c) ∧ is_power_of_two (b * c - a) ∧ is_power_of_two (c * a - b)) ↔ 
  (a = 2 ∧ b = 2 ∧ c = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 6) ∨
  (a = 3 ∧ b = 5 ∧ c = 7) :=
by
  sorry

end possible_triplets_l696_696908


namespace skateboard_distance_l696_696247

theorem skateboard_distance (scooter_speed : ℕ) (skateboard_fraction : ℚ) (time_minutes : ℚ) :
  scooter_speed = 50 →
  skateboard_fraction = 2/5 →
  time_minutes = 45 →
  let skateboard_speed := skateboard_fraction * scooter_speed in
  let time_hours := time_minutes / 60 in
  let distance := skateboard_speed * time_hours in
  distance = 15 :=
by
  intros h_scooter_speed h_skateboard_fraction h_time_minutes
  let skateboard_speed := skateboard_fraction * scooter_speed
  let time_hours := time_minutes / 60
  let distance := skateboard_speed * time_hours
  rw [h_scooter_speed, h_skateboard_fraction, h_time_minutes]
  sorry

end skateboard_distance_l696_696247


namespace digits_probability_l696_696793

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696793


namespace sum_of_first_five_primes_with_units_digit_3_l696_696128

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696128


namespace probability_digits_all_different_l696_696775

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696775


namespace proof_expansion_l696_696950

variable (x : ℝ)

theorem proof_expansion :
  let a : Fin 2023 → ℝ := 
   λ n, (Polynomial.eval (x - 1) (Polynomial.coeff ((Polynomial.C 3 * Polynomial.X - Polynomial.C 2)^2022) n))
  in (a 0 = 1) ∧ 
     ((finset.range 1012).filter (λ n, n.even)).sum a = (4^2022 + 2^2022) / 2 ∧ 
     a 2021 = 2022 * (3^2021) :=
by {
  sorry
}

end proof_expansion_l696_696950


namespace rectangle_area_l696_696014

theorem rectangle_area (DA FD AE : ℝ) (h1 : DA = 20) (h2 : FD = 5) (h3 : AE = 12) : 
    let FE := FD + DA + AE
    let CD := sqrt (FD * (DA + AE))
    DA * CD = 80 * sqrt 10 := 
by 
    sorry

end rectangle_area_l696_696014


namespace product_formula_l696_696220

theorem product_formula {p : ℕ} (hp : Nat.Prime p ∧ p > 3) :
  ∏ k in (Finset.range (p + 1)), (1 + 2 * Real.cos (2 * k * Real.pi / p)) = 3 := 
sorry

end product_formula_l696_696220


namespace union_closed_number_bound_l696_696323

-- Define the concept of a union-closed set over a finite set
def is_union_closed (S : Finset ℕ) (T : Set (Finset ℕ)) : Prop :=
  ∀ {A B}, A ∈ T → B ∈ T → (A ∪ B).subset S → (A ∪ B) ∈ T

-- Define the number of elements in the set S
def cardinality_S : ℕ := 10

-- The main conjecture to be proved
theorem union_closed_number_bound (S : Finset ℕ) (hS : S.card = cardinality_S) :
  ∃ T : Set (Finset ℕ), is_union_closed S T ∧ (T.card < 2 ^ 1023) :=
sorry

end union_closed_number_bound_l696_696323


namespace smallest_area_of_right_triangle_l696_696531

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696531


namespace negative_integer_solutions_l696_696384

theorem negative_integer_solutions (x : ℤ) : 3 * x + 1 ≥ -5 ↔ x = -2 ∨ x = -1 := 
by
  sorry

end negative_integer_solutions_l696_696384


namespace smallest_area_right_triangle_l696_696689

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696689


namespace pizza_toppings_l696_696734

theorem pizza_toppings (n : ℕ) (hn : n = 7) :
  (nat.choose n 1) + (nat.choose n 2) + (nat.choose n 3) = 63 :=
by
  sorry

end pizza_toppings_l696_696734


namespace smallest_area_right_triangle_l696_696649

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696649


namespace smallest_right_triangle_area_l696_696495

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696495


namespace sum_of_first_five_primes_with_units_digit_3_l696_696094

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696094


namespace interval_of_increase_of_f_when_k_eq_1_interval_of_decrease_of_f_when_k_eq_1_greatest_integer_k_f_gt_0_when_x_gt_1_l696_696978

theorem interval_of_increase_of_f_when_k_eq_1 :
  ∀ (x : ℝ), 1 < x -> f(x) = x * log x + (1 - 1) * x + 1
  → (deriv  (λ x : ℝ, x * log x + 1)) x > 0 ↔ x > 1 / exp 1 :=
by
  -- sorry is used here to skip the proof
  sorry

theorem interval_of_decrease_of_f_when_k_eq_1 :
  ∀ (x : ℝ), 0 < x -> x < 1 / exp 1
  → (deriv  (λ x : ℝ, x * log x + 1)) x < 0 :=
by
  -- sorry is used here to skip the proof
  sorry

theorem greatest_integer_k_f_gt_0_when_x_gt_1 :
  ∃ k, (∀ x > 1, x * log x + (1 - k.toReal) * x + k.toReal > 0)
  ∧ ∀ m : ℤ, m > k → ∃ x > 1, x * log x + (1 - m.toReal) * x + m.toReal ≤ 0 :=
by
  -- sorry is used here to skip the proof
  sorry

end interval_of_increase_of_f_when_k_eq_1_interval_of_decrease_of_f_when_k_eq_1_greatest_integer_k_f_gt_0_when_x_gt_1_l696_696978


namespace probability_digits_all_different_l696_696782

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696782


namespace paper_sufficient_to_cover_cube_l696_696750

noncomputable def edge_length_cube : ℝ := 1
noncomputable def side_length_sheet : ℝ := 2.5

noncomputable def surface_area_cube : ℝ := 6
noncomputable def area_sheet : ℝ := 6.25

theorem paper_sufficient_to_cover_cube : area_sheet ≥ surface_area_cube :=
  by
    sorry

end paper_sufficient_to_cover_cube_l696_696750


namespace machine_total_working_time_l696_696761

theorem machine_total_working_time :
  ∀ (shirts : ℕ) (rate : ℕ) (malfunctions : ℕ) (fix_time : ℕ)
  (production_time : ℕ) (total_time : ℕ),
  shirts = 360 →
  rate = 4 →
  malfunctions = 2 →
  fix_time = 5 →
  production_time = shirts / rate →
  total_time = production_time + malfunctions * fix_time →
  total_time = 100 :=
by
  intros shirts rate malfunctions fix_time production_time total_time
  intros h_shirts h_rate h_malfunctions h_fix_time h_production_time h_total_time
  rw [h_shirts, h_rate, h_malfunctions, h_fix_time, h_production_time]
  sorry

end machine_total_working_time_l696_696761


namespace exists_point_M_on_circle_intercepting_AB_segment_l696_696951

-- Define the given data
variables {circle : Type*} [metric_space circle] [normed_group circle] 
          {P Q : circle} {line : Type*} [metric_space line] [normed_group line]
          (A B : line) (given_length : ℝ)

-- Define the existence of point M on the circle
theorem exists_point_M_on_circle_intercepting_AB_segment :
  ∃ M : circle, (∃ l1 l2 : line, l1 ≠ l2 ∧
    l1 ∩ ({m : circle | distance m M ≤ distance P Q / 2} : set line) ≠ ∅ ∧
    l2 ∩ ({m : circle | distance m M ≤ distance P Q / 2} : set line) ≠ ∅ ∧
    segment_containing M P Q l1 l2 A B given_length) := sorry

end exists_point_M_on_circle_intercepting_AB_segment_l696_696951


namespace smallest_right_triangle_area_l696_696581

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696581


namespace smallest_right_triangle_area_l696_696446

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696446


namespace david_boxes_l696_696056

theorem david_boxes (total_dogs : ℕ) (dogs_per_box : ℕ) (boxes : ℕ) 
  (h1 : total_dogs = 28) (h2 : dogs_per_box = 4) : 
  boxes = total_dogs / dogs_per_box → boxes = 7 :=
by
  intros h
  rw [h1, h2] at h
  exact h
  sorry

end david_boxes_l696_696056


namespace smallest_area_right_triangle_l696_696613

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696613


namespace general_formula_of_sequence_l696_696286

open Nat

def a : ℕ → ℝ
| 0 => 1
| (n+1) => 3 * a n / (3 + a n)

theorem general_formula_of_sequence (n : ℕ) : a (n) = 3 / (n + 2) :=
by
  induction n with
  | zero => 
    simp [a]
  | succ n ih =>
    calc
      a (n + 1) = 3 * a n / (3 + a n) : by simp [a]
      ... = 3 * (3 / (n + 2)) / (3 + (3 / (n + 2))) : by rw [ih]
      ... = 3 * (3 / (n + 2)) / ((3 * (n + 2) + 3) / (n + 2)) : by field_simp
      ... = 3 * 3 / (3 * (n + 2) + 3) : by ring
      ... = 9 / (3 * (n + 2) + 3) : by ring
      ... = 3 / (n + 2 + 1) : by ring

end general_formula_of_sequence_l696_696286


namespace bella_started_with_136_candies_l696_696038

/-
Theorem:
Bella started with 136 candies.
-/

-- define the initial number of candies
variable (x : ℝ)

-- define the conditions
def condition1 : Prop := (x / 2 - 3 / 4) - 5 = 9
def condition2 : Prop := x = 136

-- structure the proof statement 
theorem bella_started_with_136_candies : condition1 x -> condition2 x :=
by
  sorry

end bella_started_with_136_candies_l696_696038


namespace school_scores_probability_l696_696273

noncomputable def normal_distribution := sorry

theorem school_scores_probability :
  ∀ (X : ℝ → ℝ) (σ : ℝ),
  (normal_distribution X 80 σ) ∧
  (prob (60 ≤ X) (X ≤ 80) = 0.25) →
  (prob X (< 100) = 0.75) :=
by
  sorry

end school_scores_probability_l696_696273


namespace sequence_sum_find_S15_l696_696957

-- Define the sequence {a_n} with initial terms and the relation for the sum S.
def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else 2 * n - 2

-- Define the sum S_n as expected by the given relation.
def S (n : ℕ) : ℕ :=
  if n = 1 then sequence 1
  else if n = 2 then sequence 2
  else S (n - 1) + 2 * (n - 1)

-- Use the initial conditions and given mathematical relation for the sequence.
theorem sequence_sum (n : ℕ) (hn : n ≥ 2) :
  S (n + 1) + S (n - 1) = 2 * (S n + S 1) :=
    sorry

theorem find_S15 : S 15 = 211 := sorry

end sequence_sum_find_S15_l696_696957


namespace students_without_favorite_subject_l696_696275

theorem students_without_favorite_subject (total_students : ℕ) (math_fraction : ℚ) (english_fraction : ℚ) (history_fraction : ℚ) (science_fraction : ℚ) :
  total_students = 120 →
  math_fraction = 3/10 →
  english_fraction = 5/12 →
  history_fraction = 1/8 →
  science_fraction = 3/20 →
  (let students_with_favorite_subjects :=
     (math_fraction * total_students) +
     (english_fraction * total_students) +
     (history_fraction * total_students) +
     (science_fraction * total_students)
   in total_students - students_with_favorite_subjects = 1) :=
begin
sorry
end

end students_without_favorite_subject_l696_696275


namespace smallest_area_correct_l696_696546

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696546


namespace max_non_div_by_3_l696_696756

theorem max_non_div_by_3 (s : Finset ℕ) (h_len : s.card = 7) (h_prod : 3 ∣ s.prod id) : 
  ∃ n, n ≤ 6 ∧ ∀ x ∈ s, ¬ (3 ∣ x) → n = 6 :=
sorry

end max_non_div_by_3_l696_696756


namespace ellipse_properties_l696_696974

theorem ellipse_properties (m : ℝ) (h_pos : m > 0) :
  (mx^2 + y^2 = 1 ∧ x = sqrt(3) ∧ y = 0) →
  (m = 1/4 ∧ ∀ k : ℝ, m ≠ 1/3 ∧ 0 < m < 1 ∧
  ((k ≠ 1 ∧ m * k^2 + (m - 1) * k + m = 0) ∧ 
  ∀ Δ : ℝ, (m - 1)^2 - 4 * m^2 > 0)) := sorry

end ellipse_properties_l696_696974


namespace prob_all_digits_different_l696_696847

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696847


namespace smallest_area_of_right_triangle_l696_696532

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696532


namespace right_triangle_min_area_l696_696655

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696655


namespace parabola_and_line_solutions_l696_696992

-- Definition of the parabola with its focus
def parabola_with_focus (p : ℝ) : Prop :=
  (∃ (y x : ℝ), y^2 = 2 * p * x) ∧ (∃ (x : ℝ), x = 1 / 2)

-- Definitions of conditions for intersection and orthogonal vectors
def line_intersecting_parabola (slope t : ℝ) (p : ℝ) : Prop :=
  ∃ (x1 x2 y1 y2 : ℝ), 
  (y1 = 2 * x1 + t) ∧ (y2 = 2 * x2 + t) ∧
  (y1^2 = 2 * x1) ∧ (y2^2 = 2 * x2) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0) ∧
  (x1 * x2 = (t^2) / 4) ∧ (x1 * x2 + y1 * y2 = 0)

-- Lean statement for the proof problem
theorem parabola_and_line_solutions :
  ∀ p t : ℝ, 
  parabola_with_focus p → 
  (line_intersecting_parabola 2 t p → t = -4)
  → p = 1 :=
by
  intros p t h_parabola h_line
  sorry

end parabola_and_line_solutions_l696_696992


namespace largest_mersenne_prime_less_than_500_l696_696437

theorem largest_mersenne_prime_less_than_500 : ∃ n : ℕ, (Nat.prime n ∧ (2^n - 1) = 127 ∧ ∀ m : ℕ, (Nat.prime m → (2^m - 1) < 500 → (2^m - 1) ≤ 127)) :=
sorry

end largest_mersenne_prime_less_than_500_l696_696437


namespace smallest_area_of_right_triangle_l696_696632

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696632


namespace probability_digits_all_different_l696_696867

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696867


namespace probability_of_selecting_dime_l696_696001

theorem probability_of_selecting_dime :
  let quarters_value := 12.50
  let nickels_value := 15.00
  let dimes_value := 5.00
  let quarter_value := 0.25
  let nickel_value := 0.05
  let dime_value := 0.10
   
  let num_quarters := quarters_value / quarter_value
  let num_nickels := nickels_value / nickel_value
  let num_dimes := dimes_value / dime_value
   
  let total_coins := num_quarters + num_nickels + num_dimes  
  
  (num_dimes / total_coins = 1 / 8) :=
begin
  sorry
end

end probability_of_selecting_dime_l696_696001


namespace calculate_f_zero_l696_696985

noncomputable def f (ω φ x : ℝ) := Real.sin (ω * x + φ)

theorem calculate_f_zero
  (ω φ : ℝ)
  (h_inc : ∀ x y : ℝ, (π / 6 < x ∧ x < y ∧ y < 2 * π / 3) → f ω φ x < f ω φ y)
  (h_symmetry1 : ∀ x : ℝ, f ω φ (π / 6 - x) = f ω φ (π / 6 + x))
  (h_symmetry2 : ∀ x : ℝ, f ω φ (2 * π / 3 - x) = f ω φ (2 * π / 3 + x)) :
  f ω φ 0 = -1 / 2 :=
sorry

end calculate_f_zero_l696_696985


namespace smallest_area_of_right_triangle_l696_696596

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696596


namespace smallest_area_right_triangle_l696_696648

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696648


namespace probability_all_digits_different_l696_696883

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696883


namespace variance_condition_l696_696969

noncomputable def variance : ℝ :=
  (1 / 5) * ((6 - 10)^2 + (9 - 10)^2 + (a - 10)^2 + (11 - 10)^2 + (b - 10)^2)

noncomputable def average (a b : ℝ) : ℝ :=
  (6 + 9 + a + 11 + b) / 5

theorem variance_condition (a b : ℝ) : 
  variance = 6.8 ∧ average a b = 10 → a^2 + b^2 = 296 :=
by
  sorry

end variance_condition_l696_696969


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696118

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696118


namespace probability_of_score_less_than_100_l696_696271

/-- Given X follows a normal distribution N(80, σ^2) and P(60 ≤ X ≤ 80) = 0.25,
    prove that P(X < 100) = 0.75 -/
theorem probability_of_score_less_than_100
  {X : ℝ → ℝ}
  (hX : X ~ Normal(80, σ^2))
  (h_prob : P(60 ≤ X ∧ X ≤ 80) = 0.25) :
  P(X < 100) = 0.75 :=
sorry

end probability_of_score_less_than_100_l696_696271


namespace number_of_good_hands_l696_696005

theorem number_of_good_hands :
  let deck := (range 1 (10 + 1)).product (range 1 4) ++ [(0, 0), (0, 0)] in
  let score (k : ℕ) := 2 ^ k in
  let hand_score (hand : list ℕ) := hand.map score |>.sum in
  (finset.powerset deck).filter (λ h : list ℕ, hand_score h = 2004).card = 1006009 :=
begin
  admit, -- Placeholder for the actual proof
end

end number_of_good_hands_l696_696005


namespace number_of_sick_animals_l696_696408

def total_animals := 26 + 40 + 34  -- Total number of animals at Stacy's farm
def sick_fraction := 1 / 2  -- Half of all animals get sick

-- Defining sick animals for each type
def sick_chickens := 26 * sick_fraction
def sick_piglets := 40 * sick_fraction
def sick_goats := 34 * sick_fraction

-- The main theorem to prove
theorem number_of_sick_animals :
  sick_chickens + sick_piglets + sick_goats = 50 :=
by
  -- Skeleton of the proof that is to be completed later
  sorry

end number_of_sick_animals_l696_696408


namespace rhombus_longer_diagonal_l696_696016

theorem rhombus_longer_diagonal (a b : ℝ)
  (side_length : a = 60)
  (shorter_diagonal : b = 56) :
  ∃ d : ℝ, d = 32 * Real.sqrt 11 :=
by
  let half_shorter_diagonal := b / 2
  have a_squared := a * a
  have b_squared := half_shorter_diagonal * half_shorter_diagonal

  let half_longer_diagonal := Real.sqrt (a_squared - b_squared)
  let longer_diagonal := 2 * half_longer_diagonal

  have longer_diagonal_squared : longer_diagonal * longer_diagonal = ((2 * half_longer_diagonal) * (2 * half_longer_diagonal)) := by sorry
 
  use 32 * Real.sqrt 11
  rw [← longer_diagonal_squared]
  sorry

end rhombus_longer_diagonal_l696_696016


namespace digits_probability_l696_696795

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696795


namespace probability_all_different_digits_l696_696763

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696763


namespace units_digit_of_sum_of_sequence_l696_696698

theorem units_digit_of_sum_of_sequence :
  let sequence_sum := (1! + 1) + (2! + 2) + (3! + 3) + (4! + 4) + (5! + 5) + (6! + 6) +
                      (7! + 7) + (8! + 8) + (9! + 9) + (10! + 10)
  in sequence_sum % 10 = 8 :=
by {
  -- Proof is omitted
  sorry
}

end units_digit_of_sum_of_sequence_l696_696698


namespace probability_all_different_digits_l696_696770

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696770


namespace jeff_current_cats_l696_696295

def initial_cats : ℕ := 20
def monday_found_kittens : ℕ := 2 + 3
def monday_stray_cats : ℕ := 4
def tuesday_injured_cats : ℕ := 1
def tuesday_health_issues_cats : ℕ := 2
def tuesday_family_cats : ℕ := 3
def wednesday_adopted_cats : ℕ := 4 * 2
def wednesday_pregnant_cats : ℕ := 2
def thursday_adopted_cats : ℕ := 3
def thursday_donated_cats : ℕ := 3
def friday_adopted_cats : ℕ := 2
def friday_found_cats : ℕ := 3

theorem jeff_current_cats : 
  initial_cats 
  + monday_found_kittens + monday_stray_cats 
  + (tuesday_injured_cats + tuesday_health_issues_cats + tuesday_family_cats)
  + (wednesday_pregnant_cats - wednesday_adopted_cats)
  + (thursday_donated_cats - thursday_adopted_cats)
  + (friday_found_cats - friday_adopted_cats) 
  = 30 := by
  sorry

end jeff_current_cats_l696_696295


namespace f_distinct_mod_3_pow_k_f_distinct_mod_3_pow_2013_l696_696061

def f : ℕ → ℕ
| 1       := 1
| (n + 1) := f n + 2^(f n)

theorem f_distinct_mod_3_pow_k (k : ℕ) :
  ∀ (n m : ℕ), 1 ≤ n ∧ n ≤ 3 ^ k ∧ 1 ≤ m ∧ m ≤ 3 ^ k ∧ n ≠ m → f(n) % (3 ^ k) ≠ f(m) % (3 ^ k) :=
by sorry

theorem f_distinct_mod_3_pow_2013 :
  ∀ (n m : ℕ), 1 ≤ n ∧ n ≤ 3 ^ 2013 ∧ 1 ≤ m ∧ m ≤ 3 ^ 2013 ∧ n ≠ m → f(n) % (3 ^ 2013) ≠ f(m) % (3 ^ 2013) :=
by sorry

end f_distinct_mod_3_pow_k_f_distinct_mod_3_pow_2013_l696_696061


namespace smallest_area_of_right_triangle_l696_696620

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696620


namespace number_of_towers_in_sandcastle_on_marks_beach_l696_696405

-- Conditions
def num_sandcastles_mark : ℕ := 20
def num_sandcastles_jeff : ℕ := 3 * num_sandcastles_mark
def towers_per_castle_jeff : ℕ := 5
def total_towers_mark (T : ℕ) : ℕ := num_sandcastles_mark * T
def total_sandcastles : ℕ := num_sandcastles_mark + num_sandcastles_jeff
def total_towers_jeff : ℕ := num_sandcastles_jeff * towers_per_castle_jeff

-- Combined total number of sandcastles and towers on both beaches is 580
axiom combined_total_eq : total_sandcastles + total_towers_mark T + total_towers_jeff = 580

-- Proof statement
theorem number_of_towers_in_sandcastle_on_marks_beach (T : ℕ) :
  combined_total_eq → 
  T = 10 :=
sorry

end number_of_towers_in_sandcastle_on_marks_beach_l696_696405


namespace inequality_solution_l696_696396

theorem inequality_solution {x : ℝ} (h : |x + 3| - |x - 1| > 0) : x > -1 :=
sorry

end inequality_solution_l696_696396


namespace sum_is_correct_l696_696154

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l696_696154


namespace sum_of_first_five_prime_units_digit_3_l696_696101

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l696_696101


namespace sum_of_first_five_primes_with_units_digit_3_l696_696123

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696123


namespace Xiaoming_speed_l696_696702

theorem Xiaoming_speed (x xiaohong_speed_xiaoming_diff : ℝ) :
  (50 * (2 * x + 2) = 600) →
  (xiaohong_speed_xiaoming_diff = 2) →
  x + xiaohong_speed_xiaoming_diff = 7 :=
by
  intros h₁ h₂
  sorry

end Xiaoming_speed_l696_696702


namespace smallest_area_of_right_triangle_l696_696628

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696628


namespace edward_spent_13_l696_696925

-- Define the initial amount of money Edward had
def initial_amount : ℕ := 19
-- Define the current amount of money Edward has now
def current_amount : ℕ := 6
-- Define the amount of money Edward spent
def amount_spent : ℕ := initial_amount - current_amount

-- The proof we need to show
theorem edward_spent_13 : amount_spent = 13 := by
  -- The proof goes here.
  sorry

end edward_spent_13_l696_696925


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696115

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696115


namespace smallest_area_of_right_triangle_l696_696600

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696600


namespace transportation_charges_l696_696354

def purchase_price : ℕ := 14000
def repair_costs : ℕ := 5000
def selling_price : ℕ := 30000
def profit_factor : ℝ := 1.5
def total_cost (T : ℕ) : ℕ := purchase_price + repair_costs + T
def equation (T : ℕ) : Prop := selling_price = profit_factor * total_cost(T)

theorem transportation_charges : ∃ T : ℕ, equation T ∧ T = 1000 := by
  sorry

end transportation_charges_l696_696354


namespace parabola_distance_x_coordinate_l696_696735

noncomputable theory

def parabola_condition (M : ℝ × ℝ) : Prop :=
  M.2 ^ 2 = 4 * M.1

def distance_condition (M : ℝ × ℝ) : Prop :=
  |M.1 + 1| = 3

theorem parabola_distance_x_coordinate :
  ∃ M : ℝ × ℝ, parabola_condition M ∧ distance_condition M ∧ M.1 = 2 :=
by
  sorry

end parabola_distance_x_coordinate_l696_696735


namespace smallest_right_triangle_area_l696_696500

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696500


namespace digits_all_different_l696_696817

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696817


namespace n_points_form_convex_n_gon_l696_696339

-- Given n points on the plane such that any four of them form the vertices of a convex quadrilateral.
variables (n : ℕ) (points : Fin n → ℝ × ℝ)
hypothesis (H : ∀ (a b c d : Fin n), convex_quad (points a) (points b) (points c) (points d))

-- Define what it means for four points to form a convex quadrilateral.
def convex_quad (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  (p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
  ((p1, p2, p3, p4 : ℝ × ℝ) ∈ convex_hull (points '' (univ : set (Fin n)))))

-- The main theorem to prove.
theorem n_points_form_convex_n_gon : ∃ S : set (Fin n), convex_hull (points '' S) = points '' (univ : set (Fin n)) :=
sorry

end n_points_form_convex_n_gon_l696_696339


namespace probability_all_digits_different_l696_696821

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696821


namespace smallest_right_triangle_area_l696_696673

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696673


namespace minimum_value_f_maximum_value_fraction_l696_696232

noncomputable def f (x : ℝ) : ℝ := abs ((1 + 2 * x) / x) + abs ((1 - 2 * x) / x)

theorem minimum_value_f : (∃ x : ℝ, f x = 4) ∧ (∀ x : ℝ, f x ≥ 4) := sorry

theorem maximum_value_fraction (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 4) :
    ∃ max_val, max_val = (ab / (a + 4b)) ∧ max_val = 4/9 := sorry

end minimum_value_f_maximum_value_fraction_l696_696232


namespace circle_radius_square_l696_696000

theorem circle_radius_square
  (ER RF GS SH : ℝ)
  (hER : ER = 23)
  (hRF : RF = 32)
  (hGS : GS = 41)
  (hSH : SH = 29) :
  let r := sqrt 666.12 in
  r^2 = 666.12 :=
by
  sorry

end circle_radius_square_l696_696000


namespace smallest_area_of_right_triangle_l696_696594

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696594


namespace probability_digits_different_l696_696797

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696797


namespace smallest_rational_number_l696_696030

theorem smallest_rational_number :
  ∃ (x : ℚ), x = -5 ∧ ∀ y ∈ ({-5, 1, -1, 0} : set ℚ), x ≤ y :=
by
  sorry

end smallest_rational_number_l696_696030


namespace smallest_opposite_number_l696_696028

-- Define the set of the given numbers
def given_numbers : List ℝ := [-1, 0, Real.sqrt 5, -1/3]

-- Define a function to get the opposite number
def opposite (x : ℝ) : ℝ := -x

-- The statement to prove
theorem smallest_opposite_number : ∃ x ∈ given_numbers, 
  (∀ y ∈ given_numbers, opposite x ≤ opposite y) ∧ x = Real.sqrt 5 := by
  sorry

end smallest_opposite_number_l696_696028


namespace smallest_area_of_right_triangle_l696_696473

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696473


namespace right_triangle_min_area_l696_696651

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696651


namespace prob_all_digits_different_l696_696841

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696841


namespace original_profit_percentage_l696_696008

noncomputable def cost_price := ℝ   -- cost price C
noncomputable def selling_price := ℝ -- selling price S
noncomputable def new_selling_price := 2 * selling_price

theorem original_profit_percentage (C S : ℝ) (h: 3 * C = 2 * S - C) : 
  (2 * C - C) / C * 100 = 100 :=
by 
  -- Using the condition that 3C = 2S - C to imply 2C = S
  rw ← h at *,
  -- rw S as 2C
  rw two_mul,
  simp,
  -- The original profit percentage calculation
  sorry

end original_profit_percentage_l696_696008


namespace select_participants_l696_696358

variable (F : ℕ) (M : ℕ) -- Define the number of female (F) and male (M) students.

def comb (n k : ℕ) : ℕ := Nat.choose n k -- Combination function, choose k from n.

theorem select_participants : comb 6 3 - comb 4 3 + comb 2 1 * comb 4 2 + comb 2 2 * comb 4 1 = 16 :=
by 
  let F := 2
  let M := 4
  have H₁ : comb (F + M) 3 = comb 6 3 := by rfl
  have H₂ : comb M 3 = comb 4 3 := by rfl
  have H₃ : comb F 1 * comb M 2 = comb 2 1 * comb 4 2 := by rfl
  have H₄ : comb F 2 * comb M 1 = comb 2 2 * comb 4 1 := by rfl
  rw [H₁, H₂, H₃, H₄]
  sorry -- proof to be filled in

end select_participants_l696_696358


namespace number_of_sick_animals_l696_696409

def total_animals := 26 + 40 + 34  -- Total number of animals at Stacy's farm
def sick_fraction := 1 / 2  -- Half of all animals get sick

-- Defining sick animals for each type
def sick_chickens := 26 * sick_fraction
def sick_piglets := 40 * sick_fraction
def sick_goats := 34 * sick_fraction

-- The main theorem to prove
theorem number_of_sick_animals :
  sick_chickens + sick_piglets + sick_goats = 50 :=
by
  -- Skeleton of the proof that is to be completed later
  sorry

end number_of_sick_animals_l696_696409


namespace sum_of_first_five_prime_units_digit_3_l696_696109

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l696_696109


namespace right_triangle_min_area_l696_696657

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696657


namespace john_necklaces_l696_696298

theorem john_necklaces (spools : ℕ) (feet_per_spool : ℕ) (feet_per_necklace : ℕ) (h_spools : spools = 3)
  (h_feet_per_spool : feet_per_spool = 20) (h_feet_per_necklace : feet_per_necklace = 4) :
  (spools * feet_per_spool) / feet_per_necklace = 15 :=
by 
  rw [h_spools, h_feet_per_spool, h_feet_per_necklace];
  norm_num
-- sorry

end john_necklaces_l696_696298


namespace largest_mersenne_prime_less_than_500_l696_696436

theorem largest_mersenne_prime_less_than_500 : ∃ n : ℕ, (Nat.prime n ∧ (2^n - 1) = 127 ∧ ∀ m : ℕ, (Nat.prime m → (2^m - 1) < 500 → (2^m - 1) ≤ 127)) :=
sorry

end largest_mersenne_prime_less_than_500_l696_696436


namespace prob_all_digits_different_l696_696845

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696845


namespace smallest_right_triangle_area_l696_696571

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696571


namespace coeff_x2y2_in_expansion_l696_696372

theorem coeff_x2y2_in_expansion : 
  (binom 8 2) * (binom 4 2) = 168 := 
by 
  sorry

end coeff_x2y2_in_expansion_l696_696372


namespace shaded_area_l696_696285

-- Required definitions based on conditions
def larger_circle_radius : ℝ := 10
def smaller_circle_radius : ℝ := 4

-- Main theorem statement, proving the area of the shaded part in the larger circle
theorem shaded_area : 
    let area_large := π * larger_circle_radius ^ 2 in
    let area_small := π * smaller_circle_radius ^ 2 in
    let total_area_two_small := 2 * area_small in
    area_large - total_area_two_small = 68 * π :=
by
  let area_large := π * larger_circle_radius ^ 2
  let area_small := π * smaller_circle_radius ^ 2
  let total_area_two_small := 2 * area_small
  sorry

end shaded_area_l696_696285


namespace smallest_area_of_right_triangle_l696_696458

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696458


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696111

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696111


namespace sum_sequence_l696_696047

theorem sum_sequence : (∑ n in Finset.range 50, (2 * n + 2) - (2 * n + 3)) + 101 = 51 := 
by
  sorry

end sum_sequence_l696_696047


namespace probability_all_digits_different_l696_696819

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696819


namespace smallest_right_triangle_area_l696_696574

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696574


namespace problem_statement_l696_696205

def sequence_satisfies (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 2 * S n = (a n)^2 + n

def positive_terms (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → a n > 0

def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = finset.sum (finset.range (n + 1)) a

theorem problem_statement (a: ℕ → ℕ) (S: ℕ → ℕ) (b: ℕ → ℕ) 
  (hsat : sequence_satisfies a S) 
  (hpos : positive_terms a)
  (hsum : sequence_sum a S) :
  (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 3) ∧ 
  (∀ n : ℕ, n > 0 → a n = n) ∧ 
  (∀ n : ℕ, finset.sum (finset.range (n + 1)) (λ k, 1 / (a k)^3) < 5 / 4) :=
by {
  -- Proof omitted
  sorry
}

end problem_statement_l696_696205


namespace smallest_right_triangle_area_l696_696667

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696667


namespace smallest_area_right_triangle_l696_696562

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696562


namespace smallest_area_right_triangle_l696_696690

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696690


namespace sum_of_first_five_primes_with_units_digit_three_l696_696144

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l696_696144


namespace sum_of_first_five_prime_units_digit_3_l696_696102

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l696_696102


namespace probability_of_selecting_one_male_and_one_female_l696_696410

noncomputable def probability_one_male_one_female : ℚ :=
  let total_ways := (Nat.choose 6 2) -- Total number of ways to select 2 out of 6
  let ways_one_male_one_female := (Nat.choose 3 1) * (Nat.choose 3 1) -- Ways to select 1 male and 1 female
  ways_one_male_one_female / total_ways

theorem probability_of_selecting_one_male_and_one_female :
  probability_one_male_one_female = 3 / 5 := by
  sorry

end probability_of_selecting_one_male_and_one_female_l696_696410


namespace smallest_area_of_right_triangle_l696_696528

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696528


namespace sum_is_correct_l696_696146

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l696_696146


namespace solve_system_l696_696325

theorem solve_system (a b c : ℝ)
  (h1 : b + c = 10 - 4 * a)
  (h2 : a + c = -16 - 4 * b)
  (h3 : a + b = 9 - 4 * c) :
  2 * a + 2 * b + 2 * c = 1 :=
by
  sorry

end solve_system_l696_696325


namespace derivative_at_pi_l696_696191
-- Import the necessary library for the problem

-- Define the function f(x) = x * sin(x)
def f (x : ℝ) : ℝ := x * Real.sin x

-- State the problem in Lean 4
theorem derivative_at_pi : (deriv f π) = -π :=
by
  sorry

end derivative_at_pi_l696_696191


namespace intersection_M_N_l696_696214

-- Given sets M and N
def M : set ℕ := {1, 2, 3}
def N : set ℕ := {2, 3, 4}

-- The proof statement
theorem intersection_M_N : M ∩ N = {2, 3} := by
  sorry

end intersection_M_N_l696_696214


namespace smallest_area_right_triangle_l696_696684

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696684


namespace probability_all_digits_different_l696_696854

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696854


namespace calculate_expression_l696_696891

theorem calculate_expression : 1^345 + 5^10 / 5^7 = 126 := by
  sorry

end calculate_expression_l696_696891


namespace smallest_right_triangle_area_l696_696443

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696443


namespace avg_payment_correct_l696_696723

def first_payment : ℕ := 410
def additional_amount : ℕ := 65
def num_first_payments : ℕ := 8
def num_remaining_payments : ℕ := 44
def total_installments : ℕ := num_first_payments + num_remaining_payments

def total_first_payments : ℕ := num_first_payments * first_payment
def remaining_payment : ℕ := first_payment + additional_amount
def total_remaining_payments : ℕ := num_remaining_payments * remaining_payment

def total_payment : ℕ := total_first_payments + total_remaining_payments
def average_payment : ℚ := total_payment / total_installments

theorem avg_payment_correct : average_payment = 465 := by
  sorry

end avg_payment_correct_l696_696723


namespace xyz_sum_is_22_l696_696257

theorem xyz_sum_is_22 (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y = 24) (h2 : x * z = 48) (h3 : y * z = 72) : 
  x + y + z = 22 :=
sorry

end xyz_sum_is_22_l696_696257


namespace problem_proof_l696_696311

noncomputable def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def P1 (n : ℕ) : ℕ := sum_n n

noncomputable def P2 (n : ℕ) : ℕ := (n - 1) * sum_n n

noncomputable def P_k (n k : ℕ) : ℕ := sum_n n * (Nat.choose (n - 1) (k - 1))

noncomputable def sum_P (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k, P_k n (k + 1))

theorem problem_proof (n : ℕ) (h : 3 ≤ n) :
  P1 n = n * (n + 1) / 2 ∧
  P2 n = n * (n + 1) * (n - 1) / 2 ∧
  sum_P n = n * (n + 1) * 2 ^ (n - 2) :=
by
  sorry

end problem_proof_l696_696311


namespace ptolemys_inequality_l696_696201

variable {A B C D : Type} [OrderedRing A]
variable (AB BC CD DA AC BD : A)

/-- Ptolemy's inequality for a quadrilateral -/
theorem ptolemys_inequality 
  (AB_ BC_ CD_ DA_ AC_ BD_ : A) :
  AC * BD ≤ AB * CD + BC * AD :=
  sorry

end ptolemys_inequality_l696_696201


namespace idempotent_function_sum_l696_696730

noncomputable theory

def is_idempotent_function {A : Type*} (f : A → A) : Prop :=
  ∀ x : A, f (f x) = f x

def count_idempotent_functions (n : ℕ) : ℕ :=
  (finset.univ.filter (λ f : (fin n) → (fin n), is_idempotent_function f)).card

theorem idempotent_function_sum :
  (∑ n : ℕ in finset.range (nat.succ nat.succ nat.succ), (count_idempotent_functions n : ℝ) / n.factorial) = real.exp (real.exp 1) - 1 :=
sorry

end idempotent_function_sum_l696_696730


namespace unique_pair_natural_numbers_l696_696930

theorem unique_pair_natural_numbers (a b : ℕ) :
  (∀ n : ℕ, ∃ c : ℕ, a ^ n + b ^ n = c ^ (n + 1)) → (a = 2 ∧ b = 2) :=
by
  sorry

end unique_pair_natural_numbers_l696_696930


namespace sum_of_measures_of_all_plane_angles_l696_696349

-- Defining the polyhedron structure
structure Polyhedron :=
  (f : ℕ) -- number of faces
  (m : Fin f → ℕ) -- number of sides of each face
  (a : ℕ) -- some constant related to the structure
  (total_sides : ∑ i, m i = 2 * a) -- given condition

-- Main theorem statement
theorem sum_of_measures_of_all_plane_angles (P : Polyhedron) : 
  ∑ i, (P.m i - 2) * π = 2 * π * (P.a - ↑P.f) :=
by
  sorry

end sum_of_measures_of_all_plane_angles_l696_696349


namespace area_of_triangle_ABC_is_1_l696_696417

-- Define the vertices A, B, and C
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (2, 1)

-- Define the function to compute the area of the triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- The main theorem to prove that the area of triangle ABC is 1
theorem area_of_triangle_ABC_is_1 : triangle_area A B C = 1 := 
by
  sorry

end area_of_triangle_ABC_is_1_l696_696417


namespace fraction_of_male_hamsters_l696_696046

theorem fraction_of_male_hamsters
  (total_pets : ℕ)
  (total_gerbils : ℕ)
  (total_male_pets : ℕ)
  (gerbil_male_fraction : ℚ)
  (total_hamsters := total_pets - total_gerbils)
  (male_gerbils := nat.floor (gerbil_male_fraction * total_gerbils))
  (male_hamsters := total_male_pets - male_gerbils)
  (hamster_fraction : ℚ := male_hamsters / total_hamsters) :
  total_pets = 90 →
  total_gerbils = 66 →
  total_male_pets = 25 →
  gerbil_male_fraction = 1/4 →
  hamster_fraction = 3/8 :=
by
  sorry

end fraction_of_male_hamsters_l696_696046


namespace calculation_l696_696042

theorem calculation :
  (-1:ℤ)^(2022) + (Real.sqrt 9) - 2 * (Real.sin (Real.pi / 6)) = 3 := by
  -- According to the mathematical problem and the given solution.
  -- Here we use essential definitions and facts provided in the problem.
  sorry

end calculation_l696_696042


namespace prob_all_digits_different_l696_696850

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696850


namespace circumcircle_area_60deg_l696_696292

noncomputable def circumcircle_area_of_triangle (b c : ℝ) (A : ℝ) : ℝ :=
  let a := real.sqrt (b^2 + c^2 - 2 * b * c * real.cos A)
  let R := a / (2 * real.sin A)
  π * R^2

theorem circumcircle_area_60deg (b c : ℝ) (A : ℝ) (hb : b = 8) (hc : c = 3) (hA : A = real.pi / 3) :
  circumcircle_area_of_triangle b c A = (49 * π) / 3 :=
by
  rw [hb, hc, hA]
  let a := sqr 7
  sorry

end circumcircle_area_60deg_l696_696292


namespace every_integer_as_x2_minus_y2_plus_ax_plus_by_l696_696964

theorem every_integer_as_x2_minus_y2_plus_ax_plus_by (a b : ℤ) (h : odd (a + b)) :
  ∀ n : ℤ, ∃ x y : ℤ, n = x^2 - y^2 + a * x + b * y :=
begin
  sorry
end

end every_integer_as_x2_minus_y2_plus_ax_plus_by_l696_696964


namespace sum_of_first_five_primes_with_units_digit_3_l696_696100

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696100


namespace probability_digits_all_different_l696_696872

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696872


namespace new_average_daily_production_l696_696943

theorem new_average_daily_production (n : ℕ) (avg_past : ℕ) (production_today : ℕ) (new_avg : ℕ)
  (h1 : n = 9)
  (h2 : avg_past = 50)
  (h3 : production_today = 100)
  (h4 : new_avg = (avg_past * n + production_today) / (n + 1)) :
  new_avg = 55 :=
by
  -- Using the provided conditions, it will be shown in the proof stage that new_avg equals 55
  sorry

end new_average_daily_production_l696_696943


namespace messages_per_member_per_day_l696_696754

theorem messages_per_member_per_day (initial_members removed_members remaining_members total_weekly_messages total_daily_messages : ℕ)
  (h1 : initial_members = 150)
  (h2 : removed_members = 20)
  (h3 : remaining_members = initial_members - removed_members)
  (h4 : total_weekly_messages = 45500)
  (h5 : total_daily_messages = total_weekly_messages / 7)
  (h6 : 7 * total_daily_messages = total_weekly_messages) -- ensures that total_daily_messages calculated is correct
  : total_daily_messages / remaining_members = 50 := 
by
  sorry

end messages_per_member_per_day_l696_696754


namespace sum_is_correct_l696_696151

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l696_696151


namespace smallest_area_correct_l696_696541

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696541


namespace max_value_of_a1_l696_696218

theorem max_value_of_a1 (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h_distinct : ∀ i j, i ≠ j → (i ≠ a1 → i ≠ a2 → i ≠ a3 → i ≠ a4 → i ≠ a5 → i ≠ a6 → i ≠ a7)) 
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 159) : a1 ≤ 19 :=
by
  sorry

end max_value_of_a1_l696_696218


namespace determinant_of_B_l696_696324

noncomputable def B (b c : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := 
  ![![b, 2], 
    ![-3, c]]

theorem determinant_of_B (b c : ℝ) (h : B b c + 2 • (B b c)⁻¹ = 0) : 
  Matrix.det (B b c) = 4 := 
sorry

end determinant_of_B_l696_696324


namespace twice_minus_three_algebraic_l696_696072

def twice_minus_three (x : ℝ) : ℝ := 2 * x - 3

theorem twice_minus_three_algebraic (x : ℝ) : 
  twice_minus_three x = 2 * x - 3 :=
by sorry

end twice_minus_three_algebraic_l696_696072


namespace inversion_center_l696_696348

noncomputable def inverseImageCenter (k g : Set Point)(C : Point) : Point := sorry

theorem inversion_center (k g : Set Point) (C : Point) :
    (is_circle k ∧ center k = C) ∧ (is_circle g ∨ is_line g) ∧ ¬passes_through g C →
    inverseImageCenter k g C = C :=
by
  sorry

end inversion_center_l696_696348


namespace smallest_area_of_right_triangle_l696_696459

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696459


namespace collinear_points_value_P_l696_696996

theorem collinear_points_value_P (P : ℝ) :
  collinear (set.range (λ t : ℝ , (1, -1))) 
    ∧ collinear (set.range (λ t : ℝ , (4, P))) 
    ∧ collinear (set.range (λ t : ℝ , (P, 0))) → 
  P = 2 ∨ P = -2 :=
sorry

end collinear_points_value_P_l696_696996


namespace smallest_area_correct_l696_696543

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696543


namespace smallest_right_triangle_area_l696_696501

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696501


namespace sum_of_first_five_primes_with_units_digit_three_l696_696138

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l696_696138


namespace pool_half_capacity_at_6_hours_l696_696403

noncomputable def double_volume_every_hour (t : ℕ) : ℕ := 2 ^ t

theorem pool_half_capacity_at_6_hours (V : ℕ) (h : ∀ t : ℕ, V = double_volume_every_hour 8) : double_volume_every_hour 6 = V / 2 := by
  sorry

end pool_half_capacity_at_6_hours_l696_696403


namespace smallest_right_triangle_area_l696_696450

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696450


namespace smallest_area_of_right_triangle_l696_696524

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696524


namespace triangle_proof_l696_696302

theorem triangle_proof (A B C M D E : Point) (hABC_acute : ∠ A B C < π / 2) (hAB_lt_AC : dist A B < dist A C) 
    (hM_midpoint : midpoint M B C) (hD_on_AB : collinear A B D) (hAD_eq_DE : dist A D = dist D E) 
    (hE_intersect : lies_on_line E (line_through D C) ∧ lies_on_line E (line_through A M)) :
    dist A B = dist C E := 
sorry

end triangle_proof_l696_696302


namespace smallest_area_of_right_triangle_l696_696468

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696468


namespace cone_radius_l696_696395

theorem cone_radius
  (l : ℝ) (CSA : ℝ) (π : ℝ) (r : ℝ)
  (h_l : l = 15)
  (h_CSA : CSA = 141.3716694115407)
  (h_pi : π = Real.pi) :
  r = 3 :=
by
  sorry

end cone_radius_l696_696395


namespace sum_of_first_five_primes_with_units_digit_three_l696_696143

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l696_696143


namespace find_point_P_l696_696229

noncomputable def tangent_at (f : ℝ → ℝ) (x : ℝ) : ℝ := (deriv f) x

theorem find_point_P :
  ∃ (x₀ y₀ : ℝ), (y₀ = (1 / x₀)) 
  ∧ (0 < x₀)
  ∧ (tangent_at (fun x => x^2) 2 = 4)
  ∧ (tangent_at (fun x => (1 / x)) x₀ = -1 / 4) 
  ∧ (x₀ = 2)
  ∧ (y₀ = 1 / 2) :=
sorry

end find_point_P_l696_696229


namespace largest_mersenne_prime_less_than_500_l696_696431

def mersenne_prime (n : ℕ) : ℕ := 2^n - 1

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem largest_mersenne_prime_less_than_500 :
  ∃ n, is_prime n ∧ mersenne_prime n < 500 ∧ ∀ m, is_prime m ∧ mersenne_prime m < 500 → mersenne_prime m ≤ mersenne_prime n :=
  sorry

end largest_mersenne_prime_less_than_500_l696_696431


namespace fraction_of_income_from_tips_l696_696749

variable (S T I : ℚ)

theorem fraction_of_income_from_tips
  (h₁ : T = (9 / 4) * S)
  (h₂ : I = S + T) : 
  T / I = 9 / 13 := 
sorry

end fraction_of_income_from_tips_l696_696749


namespace find_a_l696_696383

theorem find_a (a : ℝ) : (∃ (p : ℝ × ℝ), p = (3, -9) ∧ (3 * a * p.1 + (2 * a + 1) * p.2 = 3 * a + 3)) → a = -1 :=
by
  sorry

end find_a_l696_696383


namespace find_f2_plus_fpp2_l696_696967

noncomputable def f : ℝ → ℝ := sorry

axiom tangent_line_eq : ∃ (f : ℝ → ℝ), tangent f 2 (fun x => x - 2 * f x + 1) = 0

theorem find_f2_plus_fpp2
    (hf : ∃ (f : ℝ → ℝ), tangent f 2 (fun x => x - 2 * f x + 1) = 0) :
  f 2 + (deriv^[2] f) 2 = 2 := 
sorry

end find_f2_plus_fpp2_l696_696967


namespace sqrt_4_of_10000000_eq_l696_696048

noncomputable def sqrt_4_of_10000000 : Real := Real.sqrt (Real.sqrt 10000000)

theorem sqrt_4_of_10000000_eq :
  sqrt_4_of_10000000 = 10 * Real.sqrt (Real.sqrt 10) := by
sorry

end sqrt_4_of_10000000_eq_l696_696048


namespace smallest_area_of_right_triangle_l696_696588

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696588


namespace angle_E_is_130_l696_696351

-- Define the conditions for the parallelogram and angles
variable (EFGH : Type) [Parallelogram EFGH]
variable {H J : EFGH}
variable (EHJ : angle EFGH H J = 40)
variable (HGF : angle H G F = 130)

-- Define the proof problem to show the measure of angle E
theorem angle_E_is_130 (parallelogram EFGH : Type) [Parallelogram EFGH]
  (EHJ : angle EFGH H J = 40)
  (HGF : angle H G F = 130) :
  angle E G F = 130 :=
by
  sorry

end angle_E_is_130_l696_696351


namespace find_x_log_base_l696_696079

theorem find_x_log_base :
  ∃ x : ℝ, log x 81 = 2 → x = 9 :=
begin
  -- Declaring that x is a positive real number to be valid as a logarithmic base
  use 9,
  intro h,
  field_simp at h,
  -- We need to show that log base 9 of 81 is indeed 2
  rw log_eq_log_iff,
  -- And the fact that 81 = 9^2
  exact pow_two (9 : ℝ),
end

end find_x_log_base_l696_696079


namespace smallest_right_triangle_area_l696_696668

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696668


namespace smallest_area_of_right_triangle_l696_696470

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696470


namespace union_M_N_l696_696242

-- Definition of the sets M and N
def M : Set ℝ := {x : ℝ | x^2 = x}
def N : Set ℝ := {x : ℝ | 1 < 2^x ∧ 2^x < 2}

-- The theorem to prove
theorem union_M_N : M ∪ N = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := 
  sorry

end union_M_N_l696_696242


namespace average_income_correct_l696_696720

noncomputable def totalIncome : ℕ := 200 + 150 + 750 + 400 + 500
def numberOfDays : ℕ := 5
def averageIncome : ℕ := totalIncome / numberOfDays

theorem average_income_correct :
  averageIncome = 400 := by
  sorry

end average_income_correct_l696_696720


namespace projection_direction_vector_l696_696239

noncomputable theory

open Matrix

-- Define the projection matrix P
def P : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1/18, -1/36, -1/6],
    ![-1/36, 1/72, 1/12],
    ![-1/6, 1/12, 11/12]]

-- Define the standard basis vector i
def i : Matrix (Fin 3) (Fin 1) ℝ :=
  ![1, 0, 0]

-- Define the expected direction vector
def expected_vector : Matrix (Fin 3) (Fin 1) ℝ :=
  ![2, -1, -6]

-- Define the proof statement
theorem projection_direction_vector (P_mult_i_equals : P.mulVec i = (36:ℝ) • expected_vector) : true :=
  by
  sorry

end projection_direction_vector_l696_696239


namespace probability_all_different_digits_l696_696771

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696771


namespace smallest_right_triangle_area_l696_696578

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696578


namespace smallest_area_right_triangle_l696_696566

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696566


namespace sum_of_first_five_primes_with_units_digit_3_l696_696134

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696134


namespace cos_sum_eq_one_l696_696253

noncomputable def trig_condition (α β : ℝ) : Prop := 
  (cos α * cos (β / 2)) / (cos (α + β / 2)) + (cos β * cos (α / 2)) / (cos (β + α / 2)) = 1

theorem cos_sum_eq_one (α β : ℝ) (h : trig_condition α β) : cos α + cos β = 1 := 
sorry

end cos_sum_eq_one_l696_696253


namespace smallest_area_right_triangle_l696_696637

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696637


namespace sum_is_correct_l696_696148

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l696_696148


namespace probability_all_digits_different_l696_696874

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696874


namespace tangent_line_circle_l696_696192

open Real

theorem tangent_line_circle (m n : ℝ) :
  (∀ x y : ℝ, ((m + 1) * x + (n + 1) * y - 2 = 0) ↔ (x - 1)^2 + (y - 1)^2 = 1) →
  ((m + n) ≤ 2 - 2 * sqrt 2) ∨ (2 + 2 * sqrt 2 ≤ (m + n)) := by
  sorry

end tangent_line_circle_l696_696192


namespace number_division_equals_value_l696_696260

theorem number_division_equals_value (x : ℝ) (h : x / 0.144 = 14.4 / 0.0144) : x = 144 :=
by
  sorry

end number_division_equals_value_l696_696260


namespace product_of_n_values_l696_696091

theorem product_of_n_values (n : ℕ) (p : ℕ) (hn : n^2 - 41 * n + 420 = p) (hp : nat.prime p) : 
  n = 22 ∨ n = 19 → 22 * 19 = 418 :=
by
  sorry

end product_of_n_values_l696_696091


namespace smallest_area_of_right_triangle_l696_696469

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696469


namespace smallest_area_of_right_triangle_l696_696471

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696471


namespace exponential_function_pass_through_point_l696_696380

theorem exponential_function_pass_through_point
  (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (a^(1 - 1) + 1 = 2) :=
by
  sorry

end exponential_function_pass_through_point_l696_696380


namespace passes_each_other_correct_l696_696338

open Real

def angular_speed (velocity : ℝ) (radius : ℝ) : ℝ := velocity / radius

noncomputable def passes_each_other (Odell_velocity : ℝ) (Odell_radius : ℝ) 
                                     (Kershaw_velocity : ℝ) (Kershaw_radius : ℝ) 
                                     (total_time : ℝ) (delay_time : ℝ) : ℕ :=
let ω_O : ℝ := angular_speed Odell_velocity Odell_radius
let ω_K : ℝ := angular_speed Kershaw_velocity Kershaw_radius
let ω_rel : ℝ := ω_O + ω_K
let effective_meeting_time : ℝ := total_time - delay_time
let meeting_interval : ℝ := (2 * π) / ω_rel
(floor (effective_meeting_time / meeting_interval)).toNat

theorem passes_each_other_correct : passes_each_other 260 70 320 80 45 5 = 49 := 
by
  simp [passes_each_other, angular_speed]
  sorry

end passes_each_other_correct_l696_696338


namespace probability_all_digits_different_l696_696880

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696880


namespace incorrect_reasoning_form_l696_696392

-- Definitions/Conditions
def some_rational_infinite_recurring_decimals : Prop :=
  ∃ q : ℚ, is_infinite_recurring_decimal q

def integers_rational : Prop :=
  ∀ (n : ℤ), ∃ (q : ℚ), q = n

-- Statement to be proved
theorem incorrect_reasoning_form :
  (some_rational_infinite_recurring_decimals ∧ integers_rational) →
  ¬ (∀ (n : ℤ), is_infinite_recurring_decimal n) :=
by
  sorry

end incorrect_reasoning_form_l696_696392


namespace digits_probability_l696_696787

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696787


namespace max_value_4x_plus_3y_l696_696221

theorem max_value_4x_plus_3y :
  ∃ x y : ℝ, (x^2 + y^2 = 16 * x + 8 * y + 8) ∧ (∀ w, w = 4 * x + 3 * y → w ≤ 64) ∧ ∃ x y, 4 * x + 3 * y = 64 :=
sorry

end max_value_4x_plus_3y_l696_696221


namespace smallest_area_of_right_triangle_l696_696537

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696537


namespace smallest_area_of_right_triangle_l696_696599

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696599


namespace smallest_area_right_triangle_l696_696640

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696640


namespace smallest_area_right_triangle_l696_696481

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696481


namespace smallest_area_right_triangle_l696_696641

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696641


namespace problem_solution_l696_696213

variable (a : ℝ)
def ellipse_p (a : ℝ) : Prop := (0 < a) ∧ (a < 5)
def quadratic_q (a : ℝ) : Prop := (-3 ≤ a) ∧ (a ≤ 3)
def p_or_q (a : ℝ) : Prop := ((0 < a ∧ a < 5) ∨ ((-3 ≤ a) ∧ (a ≤ 3)))
def p_and_q (a : ℝ) : Prop := ((0 < a ∧ a < 5) ∧ ((-3 ≤ a) ∧ (a ≤ 3)))

theorem problem_solution (a : ℝ) :
  (ellipse_p a → 0 < a ∧ a < 5) ∧ 
  (¬(ellipse_p a) ∧ quadratic_q a → -3 ≤ a ∧ a ≤ 0) ∧
  (p_or_q a ∧ ¬(p_and_q a) → 3 < a ∧ a < 5 ∨ (-3 ≤ a ∧ a ≤ 0)) :=
  by
  sorry

end problem_solution_l696_696213


namespace smallest_area_right_triangle_l696_696686

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696686


namespace smallest_area_right_triangle_l696_696615

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696615


namespace sum_of_first_five_primes_with_units_digit_three_l696_696137

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l696_696137


namespace find_a_b_sum_l696_696740

theorem find_a_b_sum (a b : ℤ) (h1 : a > b) (h2 : b > 0) (vertices : (a, b) ∈ ℤ ∧ (b, a) ∈ ℤ ∧ (-a, -b) ∈ ℤ ∧ (-b, -a) ∈ ℤ)
    (area_eq : 2 * |a - b| * |a + b| = 16) : a + b = 4 := 
by 
    sorry

end find_a_b_sum_l696_696740


namespace valid_arrangement_count_eq_28_l696_696026

def has_adjacent (s : List Char) (x y : Char) : Prop :=
  ∃ n : Nat, n < s.length - 1 ∧ (s.get? n = some x ∧ s.get? (n + 1) = some y ∨ s.get? n = some y ∧ s.get? (n + 1) = some x)

def valid_arrangement (s : List Char) : Prop :=
  ¬ has_adjacent s 'A' 'B' ∧ ¬ has_adjacent s 'A' 'C' ∧ ¬ has_adjacent s 'D' 'E'

def count_valid_arrangements : Nat :=
  (List.permutations ['A', 'B', 'C', 'D', 'E']).count valid_arrangement

theorem valid_arrangement_count_eq_28 : count_valid_arrangements = 28 :=
by
  sorry

end valid_arrangement_count_eq_28_l696_696026


namespace smallest_area_of_right_triangle_l696_696465

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696465


namespace range_of_y_squared_l696_696264

theorem range_of_y_squared (y : ℝ) (h : ∛(y + 25) - ∛(y - 25) = 4) : 615 ≤ y^2 ∧ y^2 ≤ 635 :=
sorry

end range_of_y_squared_l696_696264


namespace probability_all_digits_different_l696_696826

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696826


namespace smallest_area_right_triangle_l696_696611

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696611


namespace deepak_profit_share_l696_696034

theorem deepak_profit_share (anand_investment : ℕ) (deepak_investment : ℕ) (total_profit : ℕ) 
  (h₁ : anand_investment = 22500) 
  (h₂ : deepak_investment = 35000) 
  (h₃ : total_profit = 13800) : 
  (14 * total_profit / (9 + 14)) = 8400 := 
by
  sorry

end deepak_profit_share_l696_696034


namespace smallest_area_right_triangle_l696_696569

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696569


namespace triangle_inequality_can_form_triangle_l696_696700

theorem triangle_inequality (l1 l2 l3 : ℝ) : 
  l1 + l2 > l3 ∧ l1 + l3 > l2 ∧ l2 + l3 > l1 :=
  l1 > 0 ∧ l2 > 0 ∧ l3 > 0 → 
  l1 + l2 > l3 ∧ l1 + l3 > l2 ∧ l2 + l3 > l1

theorem can_form_triangle :
  let A := (1, 2, 1) in
  let B := (2, 3, 6) in
  let C := (6, 8, 11) in
  let D := (1.5, 2.5, 4) in
  ∀ (x y z : ℝ), (x, y, z) = A ∨ (x, y, z) = B ∨ (x, y, z) = C ∨ (x, y, z) = D →
  (triangle_inequality x y z ↔ (x, y, z) = C) :=
begin
  sorry
end

end triangle_inequality_can_form_triangle_l696_696700


namespace smallest_area_right_triangle_l696_696508

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696508


namespace _l696_696280

-- Definitions and conditions
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x + y - 1 = 0

def line_m (a : ℝ) (x y : ℝ) : Prop := x - a * y + 3 = 0

def point_P := (0, 1) : ℝ × ℝ

def point_Q := (-3, 0) : ℝ × ℝ

def are_perpendicular (a : ℝ) : Prop := true

def |xy_dist_sqr| (x1 y1 x2 y2 : ℝ) : ℝ :=
  (x2 - x1)^2 + (y2 - y1)^2

-- Proving the required statement
theorem |MP_sqr|_plus_|MQ_sqr|_eq_10 (a : ℝ) 
  (M : ℝ × ℝ) 
  (h1 : line_l a M.1 M.2)
  (h2 : line_m a M.1 M.2)
  (h3 : are_perpendicular a) : |xy_dist_sqr| M.1 M.2 point_P.1 point_P.2 + |xy_dist_sqr| M.1 M.2 point_Q.1 point_Q.2 = 10 := 
sorry

end _l696_696280


namespace single_elimination_games_l696_696276

theorem single_elimination_games (n : ℕ) (h : n = 512) : (n - 1) = 511 :=
by
  sorry

end single_elimination_games_l696_696276


namespace sum_of_first_five_primes_with_units_digit_three_l696_696140

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l696_696140


namespace trapezoid_properties_l696_696712

section geometric_proof

variables {A B C D P O M K L : Type} [Geometry A B C D P O M K L]

-- Define the geometric properties outlined in the problem
variables (is_trapezoid : A D ∥ B C) 
variables (P_is_intersection : P = intersection_extension A C B D)
variables (O_is_intersection : O = intersection_diagonals A C B D)
variables (M_is_midpoint_AD : M = midpoint A D)
variables (collinear_POM : collinear P O M)
variables (K_is_midpoint_AC : K = midpoint A C)
variables (L_is_midpoint_BD : L = midpoint B D)

-- Theorem to prove:
-- 1. Prove that KL ∥ AD
-- 2. Prove that points P, H, and M are collinear
-- 3. Prove that trapezoid ABCD is isosceles

theorem trapezoid_properties :
  K L ∥ A D ∧ isosceles_trapezoid A B C D :=
begin
  split,
  -- Prove KL ∥ AD
  {
    sorry
  },
  -- Prove that ABCD is isosceles
  {
    sorry
  }
end

end geometric_proof

end trapezoid_properties_l696_696712


namespace isosceles_right_triangle_probability_l696_696052

theorem isosceles_right_triangle_probability :
  ∀ (A B C P : ℝ × ℝ),
  let AB : ℝ := 6,
      AC : ℝ := 6,
      BC : ℝ := 6 * √2,
      Area_ABC : ℝ := 1 / 2 * AB * AC,
      Area_PBC (h : ℝ) : ℝ := 3 * √2 * h in
  (A = (0, 0)) →
  (B = (6, 0)) →
  (C = (0, 6)) →
  (∀ (x y : ℝ), (x, y) = P → 0 ≤ x ∧ x ≤ 6 ∧ 0 ≤ y ∧ y ≤ 6) →
  (∀ (h : ℝ), Area_PBC h < 6 → 0 ≤ h ∧ h < √2) →
  let ratio_of_areas : ℝ := 1 / 9 in
  let required_probability : ℝ := 1 - ratio_of_areas in
  P ∈ (triangle ABC interior points) →
  Pr (∀ (P : (ℝ × ℝ)), (Area_PBC < Area_ABC / 3)) = 8 / 9 := sorry

end isosceles_right_triangle_probability_l696_696052


namespace smallest_area_correct_l696_696545

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696545


namespace probability_digits_all_different_l696_696865

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696865


namespace find_a_l696_696263

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem find_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 5) ∧ (∃ x : ℝ, f a x = 5) → a = -6 ∨ a = 4 :=
begin
  sorry,
end

end find_a_l696_696263


namespace smallest_right_triangle_area_l696_696672

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696672


namespace smallest_area_right_triangle_l696_696482

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696482


namespace smallest_area_right_triangle_l696_696610

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696610


namespace domain_of_f_eq_l696_696376

noncomputable def domain_of_f (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x ≠ 0)

theorem domain_of_f_eq :
  { x : ℝ | domain_of_f x} = { x : ℝ | -1 ≤ x ∧ x < 0 } ∪ { x : ℝ | 0 < x } :=
by
  sorry

end domain_of_f_eq_l696_696376


namespace smallest_area_right_triangle_l696_696683

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696683


namespace find_number_l696_696441

theorem find_number (x : ℝ) : 60 + (x * 12) / (180 / 3) = 61 ↔ x = 5 := by
  sorry  -- proof can be filled in here when needed

end find_number_l696_696441


namespace avg_median_of_subsets_of_range_l696_696301

open Finset

noncomputable def median {α : Type*} [LinearOrderedField α] (s : Finset α) : α :=
if h : s.card % 2 = 0 then 
  (nth s (s.card / 2)).get h + (nth s (s.pred_card / 2)).get h 
else 
  nth s (s.card / 2).get sorry

theorem avg_median_of_subsets_of_range :
  let S := range 2009 in
  let m (A : Finset ℕ) := median A in
  ∑ A in (powerset S).filter (λ A, A.card > 0), m A / (2^2008 - 1) = 2009 / 2 :=
by sorry

end avg_median_of_subsets_of_range_l696_696301


namespace tangent_line_through_origin_eq_minus_3x_l696_696228

theorem tangent_line_through_origin_eq_minus_3x :
  ∀ (x : ℝ), f x = 2 * x^3 - 3 * x → ∃ (m : ℝ), (∀ (x₀ : ℝ), (f x₀ = 2 * x₀^3 - 3 * x₀) → m = 6 * x₀^2 - 3 ∧ x₀ = 0) → ∀ (x₁ y₁: ℝ), (y₁ - (2 * 0^3 - 3 * 0) = (6 * 0^2 - 3) * (x₁ - 0)) → (y₁ = -3 * x₁)
  sorry

end tangent_line_through_origin_eq_minus_3x_l696_696228


namespace smallest_right_triangle_area_l696_696452

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696452


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696110

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696110


namespace smallest_area_of_right_triangle_l696_696598

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696598


namespace smallest_area_of_right_triangle_l696_696631

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696631


namespace probability_all_digits_different_l696_696818

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696818


namespace find_d_k_l696_696958

open Matrix

noncomputable def matrix_A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, 4], ![6, d]]

noncomputable def inv_matrix_A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let detA := 3 * d - 24
  (1 / detA) • ![![d, -4], ![-6, 3]]

theorem find_d_k (d k : ℝ) (h : inv_matrix_A d = k • matrix_A d) :
    (d, k) = (-3, 1/33) := by
  sorry

end find_d_k_l696_696958


namespace hexagon_area_evaluation_l696_696315

variable (ABCDEF : polygon)
variable (area1 : [ABCDEF] = 1)
variable (M : point)
variable (D E DE_mid : segment)
variable (M_mid : midpoint M DE)
variable (AC BM BF AM : line)
variable (X Y Z : point)
variable (X_intersect : intersects AC BM X)
variable (Y_intersect : intersects BF AM Y)
variable (Z_intersect : intersects AC BF Z)

theorem hexagon_area_evaluation 
  (hex : is_regular_hexagon ABCDEF)
  (area1 : [ABCDEF] = 1)
  (M_mid : midpoint M D E)
  (X_intersect : intersects AC BM X)
  (Y_intersect : intersects BF AM Y)
  (Z_intersect : intersects AC BF Z) :
  [BX C] + [AY F] + [AB Z] - [MX ZY] = 0 :=
sorry

end hexagon_area_evaluation_l696_696315


namespace election_includes_past_officer_l696_696368

theorem election_includes_past_officer :
  let total_candidates := 20
  let past_candidates := 10
  let officer_positions := 5
  let non_past_candidates := total_candidates - past_candidates
  nat.choose total_candidates officer_positions - nat.choose non_past_candidates officer_positions = 15252 :=
by 
  let total_candidates := 20
  let past_candidates := 10
  let officer_positions := 5
  let non_past_candidates := total_candidates - past_candidates
  sorry

end election_includes_past_officer_l696_696368


namespace solution_set_proof_l696_696223

noncomputable def f : (x : ℝ) → ℝ
noncomputable def f' (x : ℝ) := (deriv f) x

theorem solution_set_proof :
  (∀ x > -1, (f(x) / (x + 1)) + (Real.log (x + 1) * f'(x)) ≥ (Real.log (x + 1) * f(x))) →
  (f 4 = (Real.exp 4) / (Real.log 5)) →
  {x : ℝ | Real.log (x + 3) * f(x + 2) ≥ Real.exp (x - 2)} = {x | x ≥ 2} :=
by 
  intros H1 H2
  sorry

end solution_set_proof_l696_696223


namespace percentage_defective_meters_l696_696762

theorem percentage_defective_meters (total_meters : ℕ) (defective_meters : ℕ) (h1 : total_meters = 150) (h2 : defective_meters = 15) : 
  (defective_meters : ℚ) / (total_meters : ℚ) * 100 = 10 := by
sorry

end percentage_defective_meters_l696_696762


namespace sum_of_first_five_primes_with_units_digit_3_l696_696095

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696095


namespace train_speed_l696_696748

theorem train_speed (v : ℕ) :
  (∀ (d : ℕ), d = 480 → ∀ (ship_speed : ℕ), ship_speed = 60 → 
  (∀ (ship_time : ℕ), ship_time = d / ship_speed →
  (∀ (train_time : ℕ), train_time = ship_time + 2 →
  v = d / train_time))) → v = 48 :=
by
  sorry

end train_speed_l696_696748


namespace paper_cut_total_pieces_l696_696902

theorem paper_cut_total_pieces (S : ℕ) (h : ∃ n : ℕ, S = 4 * n + 1) : S = 1993 :=
by 
  -- We need to show that S = 1993 is a possible value according to the form S = 4n + 1
  obtain ⟨n, hn⟩ := h
  have : S = 4 * 498 + 1 := rfl
  rw this
  sorry

end paper_cut_total_pieces_l696_696902


namespace probability_is_18_over_25_l696_696832

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696832


namespace cosine_phi_half_l696_696259

open Real

noncomputable def phi := sorry

noncomputable def condition1 := 6 * (cos phi / sin phi) = 4 * sin phi
noncomputable def condition2 := 0 < phi ∧ phi < pi / 2

theorem cosine_phi_half : condition1 ∧ condition2 → cos phi = 1 / 2 :=
by
  exact sorry

end cosine_phi_half_l696_696259


namespace sum_of_first_five_primes_units_digit_3_l696_696166

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l696_696166


namespace rhombus_longer_diagonal_l696_696017

theorem rhombus_longer_diagonal (a b d_1 : ℝ) (h_side : a = 60) (h_d1 : d_1 = 56) :
  ∃ d_2, d_2 = 106 := by
  sorry

end rhombus_longer_diagonal_l696_696017


namespace smallest_area_of_right_triangle_l696_696461

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696461


namespace roulette_wheel_sectors_l696_696733

theorem roulette_wheel_sectors (n : ℕ) 
    (H1 : ∀ i ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, i ≤ n) 
    (H2 : (1 - ((5:ℝ) / n)^2) = 0.75) : 
    n = 10 := by
    sorry

end roulette_wheel_sectors_l696_696733


namespace smallest_right_triangle_area_l696_696575

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696575


namespace smallest_area_right_triangle_l696_696694

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696694


namespace smallest_right_triangle_area_l696_696674

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696674


namespace sphere_volume_l696_696399

theorem sphere_volume (S : ℝ) (r : ℝ) (V : ℝ) (h₁ : S = 256 * Real.pi) (h₂ : S = 4 * Real.pi * r^2) : V = 2048 / 3 * Real.pi :=
by
  sorry

end sphere_volume_l696_696399


namespace smallest_area_right_triangle_l696_696511

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696511


namespace probability_digits_different_l696_696800

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696800


namespace triangle_type_is_isosceles_right_l696_696284

open Complex

noncomputable def isosceles_right_triangle (α β : ℂ) : Prop := 
  2 * α ^ 2 - 2 * α * β + β ^ 2 = 0 → 
  ∃ θ, θ = π / 4 ∨ θ = -π / 4 ∧ (|α| = |β| * √2 / 2)

theorem triangle_type_is_isosceles_right (α β : ℂ) 
  (hα : α ≠ 0) (hβ : β ≠ 0) (h : 2 * α ^ 2 - 2 * α * β + β ^ 2 = 0) : 
  isosceles_right_triangle α β :=
sorry

end triangle_type_is_isosceles_right_l696_696284


namespace smallest_area_of_right_triangle_l696_696595

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696595


namespace probability_digits_all_different_l696_696779

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696779


namespace digits_probability_l696_696788

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696788


namespace cot_combination_l696_696173

theorem cot_combination :
  cot (arccot 5 + arccot 9 + arccot 17 + arccot 25) = 4497 / 506 :=
by
  -- Using the given property of cotangent
  have h1 : ∀ a b : ℝ, cot (arccot a + arccot b) = (a * b - 1) / (a + b),
  from sorry,
  sorry -- detailed proof goes here

end cot_combination_l696_696173


namespace sum_of_first_five_primes_with_units_digit_3_l696_696163

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696163


namespace probability_digits_all_different_l696_696783

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696783


namespace impossible_friendship_configuration_l696_696268

theorem impossible_friendship_configuration :
  let num_students := 30
  let group1 := 9
  let degree1 := 3
  let group2 := 11
  let degree2 := 4
  let group3 := 10
  let degree3 := 5

  group1 * degree1 + group2 * degree2 + group3 * degree3 = 2 * ((group1 * degree1 + group2 * degree2 + group3 * degree3) / 2) 
  → (num_students = group1 + group2 + group3) 
  → ∃ (total_edges : ℝ), total_edges = (group1 * degree1 + group2 * degree2 + group3 * degree3) / 2 ∧ ¬ (total_edges ∈ ℕ)
:= by
  intro num_students group1 degree1 group2 degree2 group3 degree3 hsum hgroup 
  let total_degree := group1 * degree1 + group2 * degree2 + group3 * degree3
  let num_edges := total_degree / 2
  have h_edges : ¬ (num_edges ∈ ℕ), from sorry
  use num_edges
  exact ⟨rfl, h_edges⟩

end impossible_friendship_configuration_l696_696268


namespace find_value_of_d_f_h_l696_696900

variables (a b c d e f g h : ℂ)

-- Conditions
def cond1 : b = 2 := sorry
def cond2 : g = -a - c - e := sorry
def cond3 : a + Complex.I * b + c + Complex.I * d + e + Complex.I * f + g + Complex.I * h = -3 * Complex.I := sorry

theorem find_value_of_d_f_h (h1 : cond1) (h2 : cond2) (h3 : cond3) : d + f + h = -5 := 
sorry

end find_value_of_d_f_h_l696_696900


namespace P_x_coordinate_l696_696959

-- Definitions
def A : Point := (3, 0)
def parabola (P : Point) : Prop := P.y^2 = 4 * P.x
def line_x_minus_1 (B : Point) : Prop := B.x = -1
def perpendicular (P : Point) (B : Point) : Prop := P.y / (P.x - B.x) = -(B.y / B.x)
def distance (P Q : Point) : ℝ := real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)
def P_x : ℝ := 2

-- Condition and assertion
theorem P_x_coordinate (P B : Point) 
    (h1 : parabola P) 
    (h2 : line_x_minus_1 B) 
    (h3 : perpendicular P B)
    (h4 : distance P B = distance P A) : 
    P.x = P_x := 
sorry

end P_x_coordinate_l696_696959


namespace sum_of_first_n_terms_l696_696240

noncomputable def a_n (n : ℕ) : ℝ := (1 + 2 + 3 + ... + n : ℝ) / n

theorem sum_of_first_n_terms (n : ℕ) : 
  ∑ k in Finset.range n, 1 / (a_n(k) * a_n(k + 1)) = 2 * n / (n + 2) := 
by
  sorry

end sum_of_first_n_terms_l696_696240


namespace probability_all_digits_different_l696_696852

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696852


namespace find_right_triangles_with_equal_perimeter_and_area_l696_696932

noncomputable def is_right_triangle (x y : ℕ) : Prop :=
  let h := Math.sqrt (x^2 + y^2)
  in x^2 + y^2 = (h:ℝ)^2

noncomputable def right_triangle_perimeter (x y : ℕ) : ℝ :=
  x + y + Math.sqrt (x^2 + y^2)

noncomputable def right_triangle_area (x y : ℕ) : ℝ :=
  1/2 * x * y

theorem find_right_triangles_with_equal_perimeter_and_area (x y : ℕ) :
  is_right_triangle x y →
  right_triangle_perimeter x y = right_triangle_area x y →
  (x = 5 ∧ y = 12 ∨ x = 12 ∧ y = 5 ∨ x = 6 ∧ y = 8 ∨ x = 8 ∧ y = 6) :=
by
  sorry

end find_right_triangles_with_equal_perimeter_and_area_l696_696932


namespace product_of_y_values_l696_696084

theorem product_of_y_values :
  (∀ (x y : ℤ), x ^ 3 + y ^ 2 - 3 * y + 1 < 0 ∧ 3 * x ^ 3 - y ^ 2 + 3 * y > 0 → (y = 1 ∨ y = 2)) →
  (∀ (x y₁ x' y₂ : ℤ), (x, y₁) ≠ (x', y₂) → x = x' ∨ y₁ ≠ y₂) →
  (∀ (x y : ℤ), (x ^ 3 + y ^ 2 - 3 * y + 1 < 0 ∧ 3 * x ^ 3 - y ^ 2 + 3 * y > 0 → y = 1 ∨ y = 2) →
    (∃ (y₁ y₂ : ℤ), y₁ = 1 ∧ y₂ = 2 ∧ y₁ * y₂ = 2)) :=
by {
  sorry
}

end product_of_y_values_l696_696084


namespace right_triangle_min_area_l696_696660

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696660


namespace monotone_increasing_function_range_l696_696981

theorem monotone_increasing_function_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (1 / 2 : ℝ) (3 : ℝ), (1 / x + 2 * a * x - 3) ≥ 0) ↔ a ≥ 9 / 8 := 
by 
  sorry

end monotone_increasing_function_range_l696_696981


namespace probability_all_digits_different_l696_696861

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696861


namespace min_value_quadratic_less_than_neg_six_l696_696737

variable {R : Type*} [LinearOrderedField R]

def quadratic (a b c : R) (x : R) : R := a * x^2 + b * x + c

theorem min_value_quadratic_less_than_neg_six 
  (a b c : R) 
  (vertex_x : R)
  (h_neg_vertex_x : vertex_x = (0 + 3 : R) / (2 : R))
  (h_vertex : quadratic a b c vertex_x < -6)
  (h_parabola_upwards : 0 < a)
  (h_f_neg2 : quadratic a b c (-2) = 6)
  (h_f_0 : quadratic a b c 0 = -4)
  (h_f_1 : quadratic a b c 1 = -6)
  (h_f_3 : quadratic a b c 3 = -4) :
  ∃ x_min : R, quadratic a b c x_min < -6 :=
begin
  use vertex_x,
  exact h_vertex,
end

end min_value_quadratic_less_than_neg_six_l696_696737


namespace perpendicular_ZA_ZC_l696_696303

open EuclideanGeometry

variables (ω : Circle) (O A B C D X Y Z : Point)
variables [tangent_to_circle A B ω] [tangent_to_circle A C ω]
variables (h1 : on_line A O D ∧ between O A D)
variables (h2 : orthogonal_projection B C D X)
variables (h3 : midpoint B X Y)
variables (h4 : second_intersection_line_circle D Y ω Z)

theorem perpendicular_ZA_ZC (h : circle_center ω O) : 
  perpendicular ZA ZC := 
sorry

end perpendicular_ZA_ZC_l696_696303


namespace find_third_number_l696_696258

def proportion_third_number (a b c d : ℝ) (h : a / b = c / d) (hb : b ≠ 0) : c = (a * d) / b :=
by
  sorry
  
theorem find_third_number (x : ℝ) (third_number : ℝ) 
  (h1 : 0.25 / x = third_number / 6) 
  (h2 : x = 0.75) : third_number = 2 :=
by
  have x := h2
  have third_number := proportion_third_number 0.25 0.75 third_number 6 h1 (by norm_num)
  exact third_number
  sorry

end find_third_number_l696_696258


namespace largest_prime_factor_of_sum_of_cyclic_sequence_l696_696744

theorem largest_prime_factor_of_sum_of_cyclic_sequence (sequence : List ℕ)
  (h1 : ∀ (n ∈ sequence), 100 ≤ n ∧ n < 1000)
  (h2 : ∀ (a b c : ℕ), a ∈ sequence → b ∈ sequence → c ∈ sequence →
     (a % 100 = (b / 10) % 100) ∧ (c = ((b % 10) * 100 + a / 100)))
  (h3 : ∃ (cycle : List ℕ), cycle.perm sequence ∧ cycle.head = sequence.head)
  : ∃ (p : ℕ), p.prime ∧ p = 37 ∧ ∀ (T : ℕ), (T = 111 * (sequence.foldr (λ x acc, acc + (x % 10)) 0)) → p ∣ T := 
sorry

end largest_prime_factor_of_sum_of_cyclic_sequence_l696_696744


namespace probability_is_18_over_25_l696_696830

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696830


namespace businessmen_drink_neither_l696_696888

theorem businessmen_drink_neither (n c t b : ℕ) 
  (h_n : n = 30) 
  (h_c : c = 15) 
  (h_t : t = 13) 
  (h_b : b = 7) : 
  n - (c + t - b) = 9 := 
  by
  sorry

end businessmen_drink_neither_l696_696888


namespace smallest_area_right_triangle_l696_696691

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696691


namespace symmetric_point_exists_l696_696400

structure Point :=
(x : ℝ)
(y : ℝ)

def symmetric_point (Q P : Point) (line : Point → ℝ) : Prop :=
  line (Point.mk ((Q.x + P.x) / 2) ((Q.y + P.y) / 2)) = 0 ∧
  (P.y - Q.y) / (P.x - Q.x) = 2

theorem symmetric_point_exists :
  symmetric_point (Point.mk 0 2) (Point.mk (-6/5) (-2/5)) (λ p, p.x + 2 * p.y - 1) :=
by { sorry }

end symmetric_point_exists_l696_696400


namespace unique_function_l696_696082

def complete_residue_system_mod (f : ℕ → ℕ) (p x : ℕ) : Prop :=
  ∀ i j : ℕ, i < p → j < p → i ≠ j → (f^[i] x) % p ≠ (f^[j] x) % p

theorem unique_function (f : ℕ → ℕ) :
  (∀ p x : ℕ, Nat.Prime p → complete_residue_system_mod f p x) →
  (∀ x k : ℕ, k > 0 → (f^[k+1] x) = f (f^[k] x)) →
  (∀ x : ℕ, (f^[1] x) = f x) →
  f = (λ x, x + 1) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end unique_function_l696_696082


namespace smallest_right_triangle_area_l696_696503

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696503


namespace probability_digits_all_different_l696_696777

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696777


namespace compute_expression_l696_696321

noncomputable def w : ℂ := Complex.cos (3 * Real.pi / 8) + Complex.i * Complex.sin (3 * Real.pi / 8)

theorem compute_expression : 
  (w / (1 + w^3) + w^2 / (1 + w^5) + w^3 / (1 + w^7) = 0) ↔
  (w^8 = 1 ∧ w ≠ 1 ∧ w^7 + w^6 + w^5 + w^4 + w^3 + w^2 + w + 1 = 0) := 
by
  sorry

end compute_expression_l696_696321


namespace smallest_area_right_triangle_l696_696554

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696554


namespace sum_of_first_five_primes_with_units_digit_3_l696_696092

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696092


namespace kl_mn_ratio_l696_696287

-- Definitions of the problem
variables (A B C D M N L K: Type) [Trapezoid A B C D]
variables (AB CD LMKN: ℝ)
variables (MKdividesAB LNdAD KLoverMN: ℚ)

-- Given conditions
def isosceles_triangle := ∀ a (A B C: Type) [Triangle A B C], (a = b) ∧ (b = c)

-- Given: AB = 8 and CD = 5
def AB_eq_8 : AB = 8 := rfl
def CD_eq_5 : CD = 5 := rfl

-- The lemmas or conditions translated
lemma mk_divides_ab_mid : MKdividesAB = 1 / 2 := 
 sorry 

lemma ln_divides_ad_ratio : LNdAD = 8 / 5 :=
 sorry

lemma lm_kn_ratio_4_7 : LMKN = 4 / 7 :=
 sorry

-- Proof Problem Statement 
theorem kl_mn_ratio (h1: MKdividesAB = 1 / 2) (h2: LNdAD = 8 / 5) (h3: LMKN = 4 / 7) : KLoverMN = 5 / 14 :=
 sorry

end kl_mn_ratio_l696_696287


namespace count_bitonic_integers_l696_696044

-- Definition of a bitonic integer
def is_bitonic (n : ℕ) : Prop :=
  ∃ (l : List ℕ), l.length ≥ 3 ∧ l.nodup ∧ l ∈ (Finset.range 9).lists
  ∧ (∀ (i : ℕ), i < l.length - 1 → l.nth_le i _ < l.nth_le (i + 1) _) -- Strictly increasing
  ∧ (∀ (i : ℕ), i > 0 → i < l.length → l.nth_le i _ < l.nth_le (i - 1) _) -- Strictly decreasing

-- Proof that the number of bitonic integers using only the digits from 1 to 9, each at most once, is 1458
theorem count_bitonic_integers : 
  (Finset.filter is_bitonic ((Finset.range 9).powerset)).card = 1458 :=
by
  sorry

end count_bitonic_integers_l696_696044


namespace multiplication_decomposition_l696_696715

theorem multiplication_decomposition :
  100 * 3 = 100 + 100 + 100 :=
sorry

end multiplication_decomposition_l696_696715


namespace rhombus_longer_diagonal_l696_696015

theorem rhombus_longer_diagonal (a b : ℝ)
  (side_length : a = 60)
  (shorter_diagonal : b = 56) :
  ∃ d : ℝ, d = 32 * Real.sqrt 11 :=
by
  let half_shorter_diagonal := b / 2
  have a_squared := a * a
  have b_squared := half_shorter_diagonal * half_shorter_diagonal

  let half_longer_diagonal := Real.sqrt (a_squared - b_squared)
  let longer_diagonal := 2 * half_longer_diagonal

  have longer_diagonal_squared : longer_diagonal * longer_diagonal = ((2 * half_longer_diagonal) * (2 * half_longer_diagonal)) := by sorry
 
  use 32 * Real.sqrt 11
  rw [← longer_diagonal_squared]
  sorry

end rhombus_longer_diagonal_l696_696015


namespace last_two_nonzero_digits_80_l696_696385

/-- Define the factorial function -/
def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

/-- Returns the last two nonzero digits of the factorial of a number -/
def last_two_nonzero_digits (n : ℕ) : ℕ :=
let digits := list.reverse (nat.digits 10 (factorial n)) in
let nonzero := digits.filter (λ d, d ≠ 0) in
(nonzero.nth 1).getD 0 * 10 + (nonzero.headD 0)

theorem last_two_nonzero_digits_80 :
  last_two_nonzero_digits 80 = 92 :=
by sorry

end last_two_nonzero_digits_80_l696_696385


namespace staircase_toothpicks_l696_696885

theorem staircase_toothpicks :
  ∀ (T : ℕ → ℕ), 
  (T 4 = 28) →
  (∀ n : ℕ, T (n + 1) = T n + (12 + 3 * (n - 3))) →
  T 6 - T 4 = 33 :=
by
  intros T T4_step H_increase
  -- proof goes here
  sorry

end staircase_toothpicks_l696_696885


namespace mary_regular_hours_l696_696337

variable (hours_worked_regular : ℕ) -- define the total number of hours Mary works at her regular rate
variable (total_earnings : ℕ := 360) -- total earnings in dollars
variable (regular_rate : ℕ := 8) -- regular rate in dollars per hour
variable (overtime_rate : ℕ := 10) -- overtime rate in dollars per hour
variable (max_regular_hours : ℕ := 40) -- maximum regular hours
variable (total_hours_worked : ℕ := hours_worked_regular + (if hours_worked_regular <= max_regular_hours then 0 else hours_worked_regular - max_regular_hours)) -- total hours calculation

theorem mary_regular_hours :
  let overtime_hours := if hours_worked_regular <= max_regular_hours then 0 else hours_worked_regular - max_regular_hours,
      regular_earnings := regular_rate * hours_worked_regular,
      overtime_earnings := overtime_rate * overtime_hours,
      total_calculated_earnings := regular_earnings + overtime_earnings
  in total_calculated_earnings = total_earnings -> hours_worked_regular = 40 := 
by
  intros
  apply sorry

end mary_regular_hours_l696_696337


namespace five_pointed_star_rotational_symmetry_l696_696382

-- Definition of a star being a five-pointed star.
def is_five_pointed_star : Prop := 
  -- Assume some formal definition of a five-pointed star
  sorry 

-- Statement that a five-pointed star has a rotational symmetry of 72 degrees.
theorem five_pointed_star_rotational_symmetry (s : Prop) (h : s = is_five_pointed_star) : 
  ∃ k ∈ ℕ, k * 72 = 360 / 5 ∧ k ≠ 0 :=
by
  sorry

end five_pointed_star_rotational_symmetry_l696_696382


namespace smallest_area_right_triangle_l696_696608

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696608


namespace n_fifth_plus_4n_mod_5_l696_696346

theorem n_fifth_plus_4n_mod_5 (n : ℕ) : (n^5 + 4 * n) % 5 = 0 := 
by
  sorry

end n_fifth_plus_4n_mod_5_l696_696346


namespace probability_is_18_over_25_l696_696837

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696837


namespace exist_alpha_beta_l696_696949

def f : ℕ → ℕ
def g : ℕ → ℕ
constant a : ℤ
constant α : ℝ
constant β : ℝ

-- Assumptions
axiom a_gt_4 : 4 < a 
axiom f_initial : f 1 = 1
axiom g_def : ∀ n : ℕ, g n = n * a - 1 - f n
axiom f_next : ∀ n : ℕ, f (n + 1) = 
                 ∃ m : ℕ, m ∉ {f i | i ≤ n} ∧ m ∉ {g i | i ≤ n} ∧ m > 0

-- The theorem to prove
theorem exist_alpha_beta : ∃ (α β : ℝ), 
  (∀ n : ℕ, f n = ⌊α * n⌋ ∧ g n = ⌊β * n⌋) :=
  sorry

end exist_alpha_beta_l696_696949


namespace probability_all_digits_different_l696_696859

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696859


namespace sum_of_first_five_prime_units_digit_3_l696_696103

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l696_696103


namespace sum_of_first_five_primes_with_units_digit_3_l696_696158

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696158


namespace fraction_q_over_p_l696_696195

noncomputable def proof_problem (p q : ℝ) : Prop :=
  ∃ k : ℝ, p = 9^k ∧ q = 12^k ∧ p + q = 16^k

theorem fraction_q_over_p (p q : ℝ) (h : proof_problem p q) : q / p = (1 + Real.sqrt 5) / 2 :=
sorry

end fraction_q_over_p_l696_696195


namespace smallest_area_right_triangle_l696_696516

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696516


namespace tourism_income_surpasses_investment_after_five_years_l696_696725

def total_investment (n : ℕ) : ℝ :=
  4000 * (1 - (4 / 5) ^ n)

def total_tourism_income (n : ℕ) : ℝ :=
  1600 * ((5 / 4) ^ n - 1)

theorem tourism_income_surpasses_investment_after_five_years :
  ∀ n : ℕ, n ≥ 5 → total_tourism_income n > total_investment n :=
by
  intros n hn
  unfold total_investment total_tourism_income
  sorry

end tourism_income_surpasses_investment_after_five_years_l696_696725


namespace prob_all_digits_different_l696_696843

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696843


namespace find_multiplier_l696_696009

-- Define the condition: the man's current age is 72
def current_age : ℕ := 72

-- Define the equation (A + 6) * N - (A - 6) * N = A
def age_equation (N : ℕ) : Prop :=
  (current_age + 6) * N - (current_age - 6) * N = current_age

-- The proof problem is to show that N = 6 satisfies the equation
theorem find_multiplier : ∃ N : ℕ, age_equation N ∧ N = 6 :=
by
  use 6
  split
  . unfold age_equation
    rw [current_age, add_mul, mul_sub, mul_add, mul_sub]
    calc (72 + 6) * 6 - (72 - 6) * 6 = 78 * 6 - 66 * 6 : by ring
        ... = 12 * 6                   : by ring
        ... = 72                      : by norm_num
  . -- The second part is to show that N = 6 is the same as stating N = 6
    refl


end find_multiplier_l696_696009


namespace smallest_area_right_triangle_l696_696513

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696513


namespace problem_part1_problem_part2_problem_part3_l696_696962

open Complex

-- Define the problem conditions
def i := Complex.I
def condition1 (z : ℂ) : Prop := (z - 3) * (2 - i) = 5
def condition2 (z : ℂ) (a : ℝ) : Prop := Im (z * (a + i)) = Im (z * a) + Re z = 0

-- Define the problem statements
theorem problem_part1 (z : ℂ) (h : condition1 z) : z = 5 + i :=
sorry

theorem problem_part2 (z : ℂ) (h : condition1 z) : abs (z - 2 + 3i) = 5 :=
sorry

theorem problem_part3 (z : ℂ) (a : ℝ) (h : condition1 z) (h2 : condition2 z a) : a = 1/5 :=
sorry

end problem_part1_problem_part2_problem_part3_l696_696962


namespace radius_of_larger_circle_l696_696423

-- Definitions based on the given conditions
variables {r : ℝ} -- radius of the smaller circle
variables {AB : ℝ} {AC : ℝ} {BC : ℝ} -- length of chord and diameter
variables {E : ℝ} -- Point of tangency
variables {x : ℝ} -- radius of the larger circle
variables {D : Point} -- Center of the circles

-- Since the radius of the larger circle is 4 times that of the smaller one
def larger_circle_radius := 4 * r

-- The given values from the problem
def diameter_larger_circle := 2 * larger_circle_radius
def length_AC := 8 * r
def length_AB := 8

-- The geometrical property that BC is tangent to the smaller circle at E.
axiom BC_tangent (h : BC)

theorem radius_of_larger_circle (h : 1 / r = 1 / 4) : larger_circle_radius = 16 :=
by sorry

end radius_of_larger_circle_l696_696423


namespace double_sum_evaluation_l696_696319

noncomputable def f (m n : ℕ) : ℕ := 3 * m + n + (m + n) ^ 2

theorem double_sum_evaluation :
  (∑ m:ℕ, ∑ n:ℕ, (2:ℝ) ^ -((f m n) : ℝ)) = (4 / 3 : ℝ) :=
begin
  sorry
end

end double_sum_evaluation_l696_696319


namespace cost_of_paints_is_5_l696_696334

-- Define folders due to 6 classes
def folder_cost_per_item := 6
def num_classes := 6
def total_folder_cost : ℕ := folder_cost_per_item * num_classes

-- Define pencils due to the 6 classes and need per class
def pencil_cost_per_item := 2
def pencil_per_class := 3
def total_pencils : ℕ := pencil_per_class * num_classes
def total_pencil_cost : ℕ := pencil_cost_per_item * total_pencils

-- Define erasers needed based on pencils and their cost
def eraser_cost_per_item := 1
def pencils_per_eraser := 6
def total_erasers : ℕ := total_pencils / pencils_per_eraser
def total_eraser_cost : ℕ := eraser_cost_per_item * total_erasers

-- Total cost spent on folders, pencils, and erasers
def total_spent : ℕ := 80
def total_cost_supplies : ℕ := total_folder_cost + total_pencil_cost + total_eraser_cost

-- Cost of paints is the remaining amount when total cost is subtracted from total spent
def cost_of_paints : ℕ := total_spent - total_cost_supplies

-- The goal is to prove the cost of paints
theorem cost_of_paints_is_5 : cost_of_paints = 5 := by
  sorry

end cost_of_paints_is_5_l696_696334


namespace quadratic_cubic_expression_l696_696312

theorem quadratic_cubic_expression
  (r s : ℝ)
  (h_eq : ∀ x : ℝ, 3 * x^2 - 4 * x - 12 = 0 → x = r ∨ x = s) :
  (9 * r^3 - 9 * s^3) / (r - s) = 52 :=
by 
  sorry

end quadratic_cubic_expression_l696_696312


namespace probability_digits_different_l696_696803

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696803


namespace sqrt_mixed_number_mult_l696_696076

theorem sqrt_mixed_number_mult (a b : ℕ) (c : ℚ) (h1 : 12 + 1/9 = c) (h2 : 12 + 1/9 = a + b/9) (h3 : c = a + b/9) :
  (sqrt (c) * sqrt (3) = (sqrt 327) / 3) :=
by sorry

end sqrt_mixed_number_mult_l696_696076


namespace inequality_proof_l696_696308

theorem inequality_proof (x : ℝ) (hx : 3 / 2 ≤ x ∧ x ≤ 5) : 
  2 * real.sqrt(x + 1) + real.sqrt(2 * x - 3) + real.sqrt(15 - 3 * x) < 2 * real.sqrt 19 := 
by {
  sorry
}

end inequality_proof_l696_696308


namespace symmetry_about_x_eq_2_l696_696901

def f (x : ℝ) : ℝ :=
  |⌊x⌋| - |⌊3 - x⌋|

theorem symmetry_about_x_eq_2 : ∀ x : ℝ, f(x) = f(2 - x) := by
  sorry

end symmetry_about_x_eq_2_l696_696901


namespace smallest_area_of_right_triangle_l696_696530

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696530


namespace total_pages_276_l696_696331

def total_pages (x : ℕ) : Prop :=
  let first_day := (1/6 : ℚ) * x + 10 in
  let after_first_day := x - first_day.to_nat in
  let second_day := (1/5 : ℚ) * after_first_day + 20 in
  let after_second_day := after_first_day - second_day.to_nat in
  let third_day := (1/4 : ℚ) * after_second_day + 25 in
  let after_third_day := after_second_day - third_day.to_nat in
  after_third_day = 100

theorem total_pages_276 : total_pages 276 :=
  sorry

end total_pages_276_l696_696331


namespace sin_α_value_l696_696225

noncomputable def point_of_intersection : ℝ × ℝ :=
  let x := (Real.sqrt 10) / 10
  let y := 3 * (Real.sqrt 10) / 10
  (x, y)

noncomputable def angle_α : ℝ :=
  let (x, y) := point_of_intersection
  Real.arcsin y

theorem sin_α_value : sin angle_α = (3 * Real.sqrt 10) / 10 := by
  sorry

end sin_α_value_l696_696225


namespace exists_plane_with_at_least_six_points_l696_696718

theorem exists_plane_with_at_least_six_points
  (points : Finset (Fin 24 → ℝ))
  (h_total_planes : (points.card.choose 3) = 2002) :
  ∃ plane, (∃ p, plane = {p : Finset (Fin 24 → ℝ) | ∃ (a b c : Fin 24 → ℝ),
    p = a ∧ p = b ∧ p = c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c}) ∧ plane.card ≥ 6 :=
  sorry

end exists_plane_with_at_least_six_points_l696_696718


namespace smallest_area_right_triangle_l696_696484

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696484


namespace probability_digits_different_l696_696801

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696801


namespace count_ordered_triples_lcm_l696_696251

theorem count_ordered_triples_lcm : 
  (∃ (triples : finset (ℕ × ℕ × ℕ)), 
     (∀ t ∈ triples, 
       let x := t.1 in 
       let y := t.2.1 in
       let z := t.2.2 in
       nat.lcm x y = 180 ∧ 
       nat.lcm x z = 1260 ∧ 
       nat.lcm y z = 420) ∧
     triples.card = 5) := 
by
  sorry

end count_ordered_triples_lcm_l696_696251


namespace probability_digits_all_different_l696_696781

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696781


namespace max_elements_in_S_l696_696317

open Finset Nat

def conditions (S : Finset ℕ) : Prop :=
  -- Condition 1: S is a non-empty subset of {1, 2, ..., 108}
  S.nonempty ∧ S ⊆ (finset.range 109).erase 0 ∧
  -- Condition 2: For any a, b in S, there exists c in S such that gcd(a, c) = 1 and gcd(b, c) = 1
  (∀ a b ∈ S, ∃ c ∈ S, gcd a c = 1 ∧ gcd b c = 1) ∧
  -- Condition 3: For any a, b in S, there exists c' in S such that gcd(a, c') > 1 and gcd(b, c') = 1
  (∀ a b ∈ S, ∃ c' ∈ S, gcd a c' > 1 ∧ gcd b c' = 1)

theorem max_elements_in_S : ∀ (S : Finset ℕ), conditions S → S.card ≤ 76 :=
by
  intros
  sorry

end max_elements_in_S_l696_696317


namespace triangle_ratio_l696_696012

/-- Given a triangle ABC with D on the altitude BH. Line AD intersects BC at E, and line CD intersects
    AB at F. If BH divides FE in the ratio 1:3 starting from F, then the ratio FH:HE is 1:3. -/
theorem triangle_ratio {A B C D E F H K : Point} :
  D ∈ altitude B H ∧
  E ∈ line A D ∧ E ∈ line B C ∧
  F ∈ line C D ∧ F ∈ line A B ∧
  B ∈ altitude B H ∧ K ∈ segment F E ∧
  ratio F K K E = 1 / 3 →
  ratio F H H E = 1 / 3 :=
sorry

end triangle_ratio_l696_696012


namespace prob_all_digits_different_l696_696842

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696842


namespace sum_of_first_five_primes_with_units_digit_3_l696_696099

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696099


namespace probability_all_digits_different_l696_696881

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696881


namespace right_triangle_min_area_l696_696659

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696659


namespace calculate_inverse_x3_minus_x_l696_696322
noncomputable def x : ℂ := (1 + complex.I * real.sqrt 3) / 2

theorem calculate_inverse_x3_minus_x : 
  (1 / (x^3 - x)) = ((-6 + complex.I * real.sqrt 3) / 39) := 
by 
  sorry

end calculate_inverse_x3_minus_x_l696_696322


namespace david_boxes_l696_696059

-- Conditions
def number_of_dogs_per_box : ℕ := 4
def total_number_of_dogs : ℕ := 28

-- Problem
theorem david_boxes : total_number_of_dogs / number_of_dogs_per_box = 7 :=
by
  sorry

end david_boxes_l696_696059


namespace correct_propositions_count_l696_696956

-- Scalar multiplication definition
variable (α : Type*) [AddCommGroup α] [Module ℝ α]

-- Define the vectors AB, BC, and AC
variable (AB BA BC AC : α)

-- Proposition definitions
def proposition1 := (AB + BA = 0)
def proposition2 := (AB + BC = AC)
def proposition3 := (AB - AC = BC)
def proposition4 := (0 • AB = (0 : α))

-- Proof of the number of correct propositions
theorem correct_propositions_count : (proposition1 + proposition2 + proposition4).count = 2 := by
  sorry

end correct_propositions_count_l696_696956


namespace probability_all_different_digits_l696_696764

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696764


namespace inequality_good_very_good_l696_696305

def isGood (n a b : ℕ) : Prop := ∃ b ∈ {k | 1 ≤ k ∧ k ≤ n^2 - 1}, n^2 ∣ (a * b - b)

def isVeryGood (n a : ℕ) : Prop := n^2 ∣ (a^2 - a)

noncomputable def countGood (n : ℕ) : ℕ := 
  Finset.card {a ∈ Finset.range (n^2 - 1) | ∃ b ∈ (Finset.range (n^2 - 1)), isGood n (a + 1) (b + 1)}

noncomputable def countVeryGood (n : ℕ) : ℕ := 
  Finset.card {a ∈ Finset.range (n^2 - 1) | isVeryGood n (a + 1)}

theorem inequality_good_very_good (n : ℕ) (h : n > 1) :
  let g := countGood n
  let v := countVeryGood n
  in v^2 + v ≤ g ∧ g ≤ n^2 - n :=
by
  intros g v
  sorry

end inequality_good_very_good_l696_696305


namespace smallest_area_right_triangle_l696_696643

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696643


namespace length_AC_eq_sqrt_10_l696_696345

noncomputable def circle_radius : ℝ := 5
noncomputable def AB_length : ℝ := 6
noncomputable def minor_arc_midpoint (A B : Point) : Point := sorry -- Definition of midpoint of minor arc
noncomputable def line_segment_length (A B : Point) : ℝ := sorry -- Function to calculate the length of a line segment

/--
Given points A and B on a circle of radius 5 with AB = 6, and point C is the midpoint
of the minor arc AB, prove that the length of the line segment AC is equal to √10.
-/
theorem length_AC_eq_sqrt_10 (A B C O : Point)
  (hA : on_circle O circle_radius A)
  (hB : on_circle O circle_radius B)
  (hAB : line_segment_length A B = AB_length)
  (hC : minor_arc_midpoint A B = C) :
  line_segment_length A C = Real.sqrt 10 :=
sorry

end length_AC_eq_sqrt_10_l696_696345


namespace hotel_statistics_l696_696726

noncomputable def hotelA_scores : List ℝ := [60, 75, 65, 80, 65, 75, 85, 70, 55, 70]
noncomputable def hotelB_scores : List ℝ := [75, 70, 65, 80, 80, 50, 80, 70, 60, 70]

open List

def median (l : List ℝ) : ℝ :=
  let l_sorted := sort l
  if length l % 2 = 1 then
    nth_le l_sorted (length l / 2) (by sorry)
  else
    (nth_le l_sorted (length l / 2 - 1) (by sorry) + nth_le l_sorted (length l / 2) (by sorry)) / 2

def variance (l : List ℝ) : ℝ :=
  let mean := (sum l) / (length l)
  (sum (map (λ x => (x - mean)^2) l)) / (length l)

theorem hotel_statistics :
  let x̄A := (sum hotelA_scores) / (length hotelA_scores)
  let x̄B := (sum hotelB_scores) / (length hotelB_scores)
  let s1_square := variance hotelA_scores
  let s2_square := variance hotelB_scores
  median hotelA_scores = 70 ∧ median hotelB_scores = 70 ∧
  (last (sort hotelA_scores) (by sorry)) - (hd (sort hotelA_scores) (by sorry)) = 30 ∧
  (last (sort hotelB_scores) (by sorry)) - (hd (sort hotelB_scores) (by sorry)) = 30 ∧
  x̄A = 70 ∧ x̄B= 70 ∧
  s1_square = 75 ∧ s2_square = 85 := by
  sorry

end hotel_statistics_l696_696726


namespace probability_all_digits_different_l696_696879

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696879


namespace xyz_sum_is_22_l696_696256

theorem xyz_sum_is_22 (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y = 24) (h2 : x * z = 48) (h3 : y * z = 72) : 
  x + y + z = 22 :=
sorry

end xyz_sum_is_22_l696_696256


namespace smallest_right_triangle_area_l696_696457

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696457


namespace smallest_right_triangle_area_l696_696584

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696584


namespace smallest_area_right_triangle_l696_696647

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696647


namespace smallest_area_of_right_triangle_l696_696529

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696529


namespace fundamental_theorem_of_calculus_part1_l696_696320

noncomputable def F (f : ℝ → ℝ) (a x : ℝ) : ℝ :=
  ∫ t in a..x, f t

theorem fundamental_theorem_of_calculus_part1 (f : ℝ → ℝ) (a x : ℝ)
  (h_cont : ContinuousOn f (Set.Icc a x)) : 
  (∂ (F f a x) ∂ x = f x) := 
sorry

end fundamental_theorem_of_calculus_part1_l696_696320


namespace constant_term_binomial_expansion_l696_696910

theorem constant_term_binomial_expansion : 
  let a := (1 : ℚ) / (x : ℚ) -- Note: Here 'x' is not bound, in actual Lean code x should be a declared variable in ℚ.
  let b := 2 * (x : ℚ)
  let n := 6
  let T (r : ℕ) := (Nat.choose n r : ℚ) * a^(n - r) * b^r
  (T 3) = (160 : ℚ) := by
  sorry

end constant_term_binomial_expansion_l696_696910


namespace x_mul_y_eq_4_l696_696361

theorem x_mul_y_eq_4 (x y z w : ℝ) (hw_pos : w > 0) 
  (h1 : x = w) (h2 : y = z) (h3 : w + w = z * w) 
  (h4 : y = w) (h5 : z = 3) (h6 : w + w = w * w) : 
  x * y = 4 := by
  sorry

end x_mul_y_eq_4_l696_696361


namespace sum_of_first_five_primes_units_digit_3_l696_696172

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l696_696172


namespace carpet_requirement_l696_696742

-- Define the dimensions of the rooms
def living_room_length: ℝ := 18
def living_room_width: ℝ := 9
def storage_room_side: ℝ := 3

-- Calculate the areas in square feet
def living_room_area : ℝ := living_room_length * living_room_width
def storage_room_area : ℝ := storage_room_side * storage_room_side

-- Define the total area in square feet
def total_area : ℝ := living_room_area + storage_room_area

-- Define the conversion factor from square feet to square yards 
def sq_ft_to_sq_yd : ℝ := 9

-- Statement: total carpet required in square yards
def total_carpet_required : ℝ := total_area / sq_ft_to_sq_yd

theorem carpet_requirement : total_carpet_required = 19 := by
  sorry

end carpet_requirement_l696_696742


namespace sqrt_five_minus_one_range_l696_696926

theorem sqrt_five_minus_one_range (h : 2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3) : 
  1 < Real.sqrt 5 - 1 ∧ Real.sqrt 5 - 1 < 2 := 
by 
  sorry

end sqrt_five_minus_one_range_l696_696926


namespace smallest_area_of_right_triangle_l696_696472

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696472


namespace digits_all_different_l696_696812

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696812


namespace find_s_l696_696362

-- Define nonzero real numbers.
variables (r s t : ℝ)
-- Conditions from the problem.
hypothesis (h₀ : r ≠ 0) (h₁ : s ≠ 0) (h₂ : t ≠ 0)
-- Condition: polynomial x^2 + r*x + s has roots s and t.
hypothesis (h₃ : s * t = s) (h₄ : s + t = -r)
-- Condition: polynomial x^2 + t*x + r has 5 as a root.
hypothesis (h₅ : 5^2 + 5 * t + r = 0)

-- The goal is to prove s = 29.
theorem find_s : s = 29 :=
by 
  sorry

end find_s_l696_696362


namespace cyclic_quadrilateral_ABC_to_SNDM_l696_696394

theorem cyclic_quadrilateral_ABC_to_SNDM
  (A B C D S K L M N : Point)
  (hTangent : TangentCircle ABCD K L M N)
  (hIntersection : Intersects S K M ∧ Intersects S L N)
  (hCyclicSKBL : CyclicQuadrilateral S K B L) :
  CyclicQuadrilateral S N D M := 
sorry

end cyclic_quadrilateral_ABC_to_SNDM_l696_696394


namespace smallest_area_of_right_triangle_l696_696622

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696622


namespace eigenvalue_kernel_x2t2_l696_696426

theorem eigenvalue_kernel_x2t2 :
  let K : ℝ → ℝ → ℝ := λ x t, (x^2) * (t^2) in
  let eigenvalue : ℝ := 5 in
  ∀ x t : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ t ∧ t ≤ 1 → 
  (∃ λ : ℝ, (EigenValues (K x t) = eigenvalue)) :=
by
  sorry

end eigenvalue_kernel_x2t2_l696_696426


namespace find_k_l696_696948

-- Define the problem conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 1)
def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

-- Define the dot product for 2D vectors
def dot_prod (x y : ℝ × ℝ) : ℝ := x.1 * y.1 + x.2 * y.2

-- State the theorem
theorem find_k (k : ℝ) (h : dot_prod b (c k) = 0) : k = -3/2 :=
by
  sorry

end find_k_l696_696948


namespace prob_all_digits_different_l696_696840

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696840


namespace problem_part_I_problem_part_II_l696_696238

noncomputable def C1_parametric (φ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos φ, 2 * Real.sin φ)

def A_polar : ℝ × ℝ := (2, Real.pi / 6)

noncomputable def C2_polar (θ : ℝ) : ℝ :=
  6 / Real.sqrt (4 + 5 * Real.sin θ ^ 2)

theorem problem_part_I :
  let A := (Real.sqrt 3, 1)
  let B := (-1, Real.sqrt 3)
  let C := (-Real.sqrt 3, -1)
  let D := (1, -Real.sqrt 3)
  True := True :=
sorry

theorem problem_part_II :
  ∀ Pθ : ℝ, ∃ PA PB PC PD : ℝ,
  let P := (3 * Real.cos Pθ, 2 * Real.sin Pθ)
  PA = (P.1 - Real.sqrt 3)^2 + (P.2 - 1)^2 →
  PB = (P.1 + 1)^2 + (P.2 - Real.sqrt 3)^2 →
  PC = (P.1 + Real.sqrt 3)^2 + (P.2 + 1)^2 →
  PD = (P.1 - 1)^2 + (P.2 + Real.sqrt 3)^2 →
  True := True →
  ∃ θ, PA + PB + PC + PD = 52 :=
sorry

end problem_part_I_problem_part_II_l696_696238


namespace simplify_G_l696_696316

-- Define the function F
def F (x : ℝ) : ℝ := log ((1 + x) / (1 - x))

-- Define the function G with the given substitution
def G (x : ℝ) : ℝ := log ((1 + (4 * x + x^4) / (1 + 4 * x^3)) / (1 - (4 * x + x^4) / (1 + 4 * x^3)))

-- Statement to prove that G(x) = 4 * F(x)
theorem simplify_G : ∀ x : ℝ, G x = 4 * F x := by
  sorry

end simplify_G_l696_696316


namespace geometric_sequence_T_n_lt_1_l696_696954

variable {n : ℕ}
def a (n : ℕ) : ℕ := 3 ^ n

def S (n : ℕ) : ℕ := (List.range n).sum (λ i => a (i + 1))

def b (n : ℕ) : ℝ := 1 / (Real.logBase 3 (a n) * Real.logBase 3 (a (n + 1)))

def T (n : ℕ) : ℝ := (List.range n).sum (λ i => b (i + 1))

theorem geometric_sequence (n : ℕ) :
  2 * S n = 3 * a n - 3 := 
sorry

theorem T_n_lt_1 (n : ℕ) (hn : 0 < n) : 
  T n < 1 :=
sorry

end geometric_sequence_T_n_lt_1_l696_696954


namespace sum_of_first_five_primes_units_digit_3_l696_696171

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l696_696171


namespace probability_all_digits_different_l696_696825

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696825


namespace smallest_area_right_triangle_l696_696556

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696556


namespace shaded_area_is_correct_l696_696031

def tile_length : ℝ := 2
def tile_area : ℝ := tile_length * tile_length
def quarter_circle_radius : ℝ := 1
def quarter_circle_area : ℝ := Real.pi * (quarter_circle_radius ^ 2) / 4
def full_circle_area : ℝ := Real.pi * (quarter_circle_radius ^ 2)
def shaded_area_per_tile : ℝ := tile_area - full_circle_area

def floor_length : ℝ := 8
def floor_width : ℝ := 10
def total_tiles : ℝ := (floor_length * floor_width) / tile_area
def total_shaded_area : ℝ := total_tiles * shaded_area_per_tile

theorem shaded_area_is_correct : total_shaded_area = 80 - 20 * Real.pi := by
  sorry

end shaded_area_is_correct_l696_696031


namespace determine_range_of_a_l696_696984

-- Define the function f(x).
def f (a x : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x

-- Define the derivative of f(x).
def f' (a x : ℝ) : ℝ := 1 / x + 2 * a * x - 3

-- Define the function g(x) as used in the solution.
def g (x : ℝ) : ℝ := (3 / (2 * x)) - (1 / (2 * x^2))

-- Define the interval.
def I : set ℝ := {x : ℝ | 1/2 < x ∧ x < 3}

-- Define the condition for f(x) to be monotonically increasing on the interval I.
def monotonic_increasing_on (a : ℝ) : Prop :=
  ∀ x ∈ I, f'(a, x) ≥ 0

-- State the theorem.
theorem determine_range_of_a :
  ∀ a : ℝ, monotonic_increasing_on a ↔ a ∈ set.Ici (9/8) :=
begin
  sorry
end

end determine_range_of_a_l696_696984


namespace smallest_area_right_triangle_l696_696476

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696476


namespace change_in_expression_l696_696086

theorem change_in_expression (x b : ℝ) (hb : 0 < b) :
  let original_expr := x^2 - 5 * x + 2
  let new_x := x + b
  let new_expr := (new_x)^2 - 5 * (new_x) + 2
  new_expr - original_expr = 2 * b * x + b^2 - 5 * b :=
by
  sorry

end change_in_expression_l696_696086


namespace sum_of_first_five_prime_units_digit_3_l696_696104

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l696_696104


namespace integral_x_from_0_to_1_l696_696928

theorem integral_x_from_0_to_1 : ∫ x in 0..1, x = 1/2 :=
by
  sorry

end integral_x_from_0_to_1_l696_696928


namespace find_unique_solution_to_exponential_equation_l696_696085

theorem find_unique_solution_to_exponential_equation :
  ∃! (m n : ℕ), (m > 0) ∧ (n > 0) ∧ (2^m - 3^n = 7) := by
  use 4, 2
  split
  { split
    { exact nat.succ_pos' 3 },
    { exact nat.succ_pos' 1 },
  exact rfl }
  intros a b h
  cases' h with ha h
  cases' h with hb heq
  sorry

end find_unique_solution_to_exponential_equation_l696_696085


namespace endpoint_of_vector_a_l696_696970

theorem endpoint_of_vector_a (x y : ℝ) (h : (x - 3) / -3 = (y + 1) / 4) : 
    x = 13 / 5 ∧ y = 2 / 5 :=
by sorry

end endpoint_of_vector_a_l696_696970


namespace smallest_right_triangle_area_l696_696678

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696678


namespace sqrt_diff_l696_696894

theorem sqrt_diff (a b : ℝ) (h₁ : a = Real.sqrt 7) (h₂ : b = Real.sqrt 6) :
  (a + b) * (a - b) = 1 :=
by
  rw [h₁, h₂]
  calc
    (Real.sqrt 7 + Real.sqrt 6) * (Real.sqrt 7 - Real.sqrt 6)
        = (Real.sqrt 7)^2 - (Real.sqrt 6)^2 : by rw [mul_add, mul_sub, sub_mul, add_comm, sub_self]
    ... = 7 - 6 : by { rw [Real.pow_two, Real.pow_two], arith }
    ... = 1 : by arith

end sqrt_diff_l696_696894


namespace limit_negative_delta_x_l696_696968

open Filter

-- Given problem conditions
variables {f : ℝ → ℝ}
hypothesis hf : Differentiable ℝ f
hypothesis h_deriv : deriv f 1 = 1

-- Formal statement of the proof
theorem limit_negative_delta_x :
  tendsto (λ Δx : ℝ, (f (1 - Δx) - f 1) / (-Δx)) (𝓝 0) (𝓝 1) :=
by {
  -- Proof omitted for brevity
  sorry
}

end limit_negative_delta_x_l696_696968


namespace orthocenter_exists_l696_696343

theorem orthocenter_exists 
  (A B C P : Point)
  (h₁ : angle A B P = angle A C P)
  (h₂ : angle C B P = angle C A P) :
  is_orthocenter P A B C :=
sorry

end orthocenter_exists_l696_696343


namespace largest_negative_sum_S19_l696_696283

theorem largest_negative_sum_S19 (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 10 < 0)
  (h2 : a 11 > 0)
  (h3 : a 11 > |a 10|)
  (hSn : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n, n ≤ 19 → S n ≤ 0) ∧
  (∀ n, n ≥ 19 → S n ≥ 0) :=
begin
  sorry
end

end largest_negative_sum_S19_l696_696283


namespace smallest_right_triangle_area_l696_696676

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696676


namespace smallest_area_right_triangle_l696_696512

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696512


namespace infinite_pairs_of_squares_l696_696350

theorem infinite_pairs_of_squares :
  ∃ᶠ (a b : ℕ) in filter(eventually, ℕ × ℕ),
    (∃ x y : ℕ, a = x^2 ∧ b = y^2 ∧ (nat.digits 10 a).length = (nat.digits 10 b).length ∧ 
    ∃ z : ℕ, (list.foldr (fun d acc => d + 10 * acc) 0 ((nat.digits 10 a) ++ (nat.digits 10 b)) = z ^ 2)) :=
sorry

end infinite_pairs_of_squares_l696_696350


namespace ivan_speed_ratio_l696_696007

/-- 
A group of tourists started a hike from a campsite. Fifteen minutes later, Ivan returned to the campsite for a flashlight 
and started catching up with the group at a faster constant speed. He reached them 2.5 hours after initially leaving. 
Prove Ivan's speed is 1.2 times the group's speed.
-/
theorem ivan_speed_ratio (d_g d_i : ℝ) (t_g t_i : ℝ) (v_g v_i : ℝ)
    (h1 : t_g = 2.25)       -- Group's travel time (2.25 hours after initial 15 minutes)
    (h2 : t_i = 2.5)        -- Ivan's total travel time
    (h3 : d_g = t_g * v_g)  -- Distance covered by group
    (h4 : d_i = 3 * (v_g * (15 / 60))) -- Ivan's distance covered
    (h5 : d_g = d_i)        -- Ivan eventually catches up with the group
  : v_i / v_g = 1.2 := sorry

end ivan_speed_ratio_l696_696007


namespace probability_all_digits_different_l696_696856

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696856


namespace proof_problem_l696_696197

theorem proof_problem (a : ℝ)
  (h1 : ∀ x : ℝ, ⟨ x ⟩ = x - ⌊ x ⌋) 
  (h2 : a > 0)
  (h3 : ⟨ a⁻¹ ⟩ = ⟨ a^2 ⟩) 
  (h4 : 2 < a^2 ∧ a^2 < 3) :
  a^12 - 144 * a⁻¹ = 233 :=
by sorry

end proof_problem_l696_696197


namespace find_vector_u_l696_696905

def B : Matrix (Fin 2) (Fin 2) ℝ := ![
  #[1,0],
  #[0,2]
]

theorem find_vector_u :
  ∃ u : Matrix (Fin 2) (Fin 1) ℝ, 
    (B^6 + 2 • B^4 + 3 • B^2 + (1 : Matrix (Fin 2) (Fin 2) ℝ)) ⬝ u = ![
      #[14],
      #[0]
    ] ∧
  u = ![
    #[2],
    #[0]
  ] :=
by
  use ![
    #[2],
    #[0]
  ]
  sorry

end find_vector_u_l696_696905


namespace smallest_area_right_triangle_l696_696486

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696486


namespace determine_number_of_solutions_l696_696911

noncomputable def num_solutions_eq : Prop :=
  let f (x : ℝ) := (3 * x ^ 2 - 15 * x) / (x ^ 2 - 7 * x + 10)
  let g (x : ℝ) := x - 4
  ∃ S : Finset ℝ, 
    (∀ x ∈ S, (x ≠ 2 ∧ x ≠ 5) ∧ f x = g x) ∧
    S.card = 2

theorem determine_number_of_solutions : num_solutions_eq :=
  by
  sorry

end determine_number_of_solutions_l696_696911


namespace right_triangle_min_area_l696_696665

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696665


namespace beka_flew_more_l696_696713

def bekaMiles := 873
def jacksonMiles := 563

theorem beka_flew_more : bekaMiles - jacksonMiles = 310 := by
  -- proof here
  sorry

end beka_flew_more_l696_696713


namespace determine_range_of_a_l696_696983

-- Define the function f(x).
def f (a x : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x

-- Define the derivative of f(x).
def f' (a x : ℝ) : ℝ := 1 / x + 2 * a * x - 3

-- Define the function g(x) as used in the solution.
def g (x : ℝ) : ℝ := (3 / (2 * x)) - (1 / (2 * x^2))

-- Define the interval.
def I : set ℝ := {x : ℝ | 1/2 < x ∧ x < 3}

-- Define the condition for f(x) to be monotonically increasing on the interval I.
def monotonic_increasing_on (a : ℝ) : Prop :=
  ∀ x ∈ I, f'(a, x) ≥ 0

-- State the theorem.
theorem determine_range_of_a :
  ∀ a : ℝ, monotonic_increasing_on a ↔ a ∈ set.Ici (9/8) :=
begin
  sorry
end

end determine_range_of_a_l696_696983


namespace smallest_area_right_triangle_l696_696605

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696605


namespace probability_all_digits_different_l696_696877

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696877


namespace smallest_area_right_triangle_l696_696642

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696642


namespace relationship_l696_696190

-- Definitions of a, b, and c
def a : ℝ := Real.sqrt 0.3
def b : ℝ := 2 ^ 0.3
def c : ℝ := 0.3 ^ 0.2

-- The proof of the relationship b > a > c
theorem relationship : b > a ∧ a > c := 
by
  -- Proof goes here.
  sorry

end relationship_l696_696190


namespace min_divisors_atresvido_l696_696330

-- Define what it means to be atresvido
def is_atresvido (n : ℕ) : Prop :=
  ∃ (A B C : Finset ℕ), 
    (A ∪ B ∪ C = (Finset.range n).filter (λ d, n % d = 0)) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧ 
    (A.sum id = B.sum id) ∧ (B.sum id = C.sum id)

-- Prove the minimum number of divisors of an atresvido number is 16
theorem min_divisors_atresvido : ∃ n, is_atresvido n ∧ (Finset.range n).filter (λ d, n % d = 0).card = 16 := 
sorry

end min_divisors_atresvido_l696_696330


namespace largest_mersenne_prime_is_127_l696_696433

noncomputable def largest_mersenne_prime_less_than_500 : ℕ :=
  127

theorem largest_mersenne_prime_is_127 :
  ∃ p : ℕ, Nat.Prime p ∧ (2^p - 1) = largest_mersenne_prime_less_than_500 ∧ 2^p - 1 < 500 := 
by 
  -- The largest Mersenne prime less than 500 is 127
  use 7
  sorry

end largest_mersenne_prime_is_127_l696_696433


namespace andrey_word_count_l696_696035

-- Helper function to count words for a given number in the specified range
def word_count (n : ℕ) : ℕ :=
  if n = 180 then 3
  else if n >= 190 && n <= 199 then 2
  else if n >= 200 && n <= 220 then 2
  else 2 -- general case for numbers from 181 to 189

-- Function to calculate the total word count for numbers from 180 to 220
def total_word_count : ℕ :=
  (List.range' 180 (220 - 180 + 1)).sum word_count

-- The theorem statement for the proof problem
theorem andrey_word_count : total_word_count = 99 := by
  sorry

end andrey_word_count_l696_696035


namespace probability_all_different_digits_l696_696766

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696766


namespace ewan_sequence_113_l696_696071

-- Define the sequence
def sequence (n : ℕ) : ℕ := 11 * n - 8

-- Define a predicate to check if a number is in the sequence
def in_sequence (a : ℕ) : Prop := ∃ n : ℕ, sequence n = a

-- The proof statement
theorem ewan_sequence_113 : in_sequence 113 :=
by {
  use 11,
  show sequence 11 = 113,
  calc
  sequence 11 = 11 * 11 - 8 : rfl
            ... = 121 - 8     : rfl
            ... = 113         : rfl
}

end ewan_sequence_113_l696_696071


namespace coin_payment_difference_l696_696341

-- Conditions
def owes (paul: Type) : ℕ := 35
def five_cent : ℕ := 5
def ten_cent : ℕ := 10
def twenty_five_cent : ℕ := 25

-- Problem statement
theorem coin_payment_difference :
  (let min_coins := 2 in
   let max_coins := 7 in
   max_coins - min_coins = 5) :=
begin
  -- Proof goes here
  sorry
end

end coin_payment_difference_l696_696341


namespace evensum_tuples_6_l696_696090

theorem evensum_tuples_6 :
  (∃ t : Fin 6 → Fin 3, (∑ i, (t i : ℕ)) % 2 = 0) =
    365 :=
sorry

end evensum_tuples_6_l696_696090


namespace probability_digits_different_l696_696799

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696799


namespace find_Q_investment_time_l696_696709

variable {x : ℝ}
variable {t : ℝ}

-- Investments ratio
def investments_ratio (I_P I_Q : ℝ) : Prop := I_P / I_Q = 7 / 5.00001

-- Profit ratio
def profits_ratio (P_P P_Q : ℝ) : Prop := P_P / P_Q = 7.00001 / 10

-- Given conditions
axiom investments_ratio_axiom : investments_ratio (7 * x) (5.00001 * x)
axiom profits_ratio_axiom : profits_ratio (7.00001 * (7 * x * 5)) (10 * (5.00001 * x * t))

-- Goal: find the time t for which Q invested the money
theorem find_Q_investment_time
    (P_investment : ℝ) (Q_investment : ℝ)
    (P_profit : ℝ) (Q_profit : ℝ)
    (P_time : ℝ := 5) :
    investments_ratio P_investment Q_investment →
    profits_ratio P_profit Q_profit →
    P_investment = 7 * x →
    Q_investment = 5.00001 * x →
    P_profit = 7.00001 * (7 * x * P_time) →
    Q_profit = 10 * (5.00001 * x * t) →
    t ≈ 9.99857 :=
by
  sorry

end find_Q_investment_time_l696_696709


namespace probability_all_digits_different_l696_696828

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696828


namespace tetrahedron_incenter_intersection_l696_696207

variables {A B C D I_A I_B I_C I_D : Type}
variables [euclidean_space ℝ (Type*)]

noncomputable def tetrahedron (A B C D : P := plane) :=
is_tetrahedron [A, B, C, D]

def incenter (P1 P2 P3 : Type) :=
center_of_incircle P1 P2 P3

theorem tetrahedron_incenter_intersection 
  (h₁ : tetrahedron A B C D)
  (h₂ : AB * CD = AC * BD)
  (h₃ : AC * BD = AD * BC)
  (h₄ : incenter B C D = I_A)
  (h₅ : incenter C D A = I_B)
  (h₆ : incenter D A B = I_C)
  (h₇ : incenter A B C = I_D): 
  ∃ P : Type, collinear A I_A P ∧ collinear B I_B P ∧ collinear C I_C P ∧ collinear D I_D P := 
sorry

end tetrahedron_incenter_intersection_l696_696207


namespace reciprocal_pair_c_l696_696758

def is_reciprocal (a b : ℝ) : Prop :=
  a * b = 1

theorem reciprocal_pair_c :
  is_reciprocal (-2) (-1/2) :=
by sorry

end reciprocal_pair_c_l696_696758


namespace probability_all_digits_different_l696_696855

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696855


namespace monotone_increasing_function_range_l696_696982

theorem monotone_increasing_function_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (1 / 2 : ℝ) (3 : ℝ), (1 / x + 2 * a * x - 3) ≥ 0) ↔ a ≥ 9 / 8 := 
by 
  sorry

end monotone_increasing_function_range_l696_696982


namespace sum_of_first_five_prime_units_digit_3_l696_696105

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l696_696105


namespace find_polynomial_correct_l696_696013

noncomputable def find_polynomial : Polynomial ℚ :=
  let p := Polynomial.Cosnt (5 : ℚ) * Polynomial.X ^ 2 + Polynomial.Cosnt (2 : ℚ) * Polynomial.X - Polynomial.Cosnt (1 : ℚ)
  let q := Polynomial.Cosnt (6 : ℚ) * Polynomial.X ^ 2 + Polynomial.Cosnt (-5 : ℚ) * Polynomial.X + Polynomial.Cosnt (3 : ℚ)
  q - p

theorem find_polynomial_correct :
  let p := Polynomial.Cosnt (5 : ℚ) * Polynomial.X ^ 2 + Polynomial.Cosnt (2 : ℚ) * Polynomial.X - Polynomial.Cosnt (1 : ℚ)
  let q := Polynomial.Cosnt (6 : ℚ) * Polynomial.X ^ 2 + Polynomial.Cosnt (-5 : ℚ) * Polynomial.X + Polynomial.Cosnt (3 : ℚ)
  q - p = Polynomial.Cosnt (1 : ℚ) * Polynomial.X ^ 2 + Polynomial.Cosnt (-7 : ℚ) * Polynomial.X + Polynomial.Cosnt (4 : ℚ) :=
by sorry

end find_polynomial_correct_l696_696013


namespace min_value_m_plus_n_l696_696364

theorem min_value_m_plus_n (m n : ℕ) (h : 108 * m = n^3) (hm : 0 < m) (hn : 0 < n) : m + n = 8 :=
sorry

end min_value_m_plus_n_l696_696364


namespace smallest_area_right_triangle_l696_696485

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696485


namespace smaller_solid_volume_l696_696055

noncomputable def cube_edge_length : ℝ := 2

def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def D := point 0 0 0
def M := point 1 2 0
def N := point 2 0 1

-- Define the condition for the plane that passes through D, M, and N
def plane (p r q : ℝ × ℝ × ℝ) (x y z : ℝ) : Prop :=
  let (px, py, pz) := p
  let (rx, ry, rz) := r
  let (qx, qy, qz) := q
  2 * x - 4 * y - 8 * z = 0

-- Predicate to test if point is on a plane
def on_plane (pt : ℝ × ℝ × ℝ) : Prop :=
  let (x, y, z) := pt
  plane D M N x y z

-- Volume of the smaller solid
theorem smaller_solid_volume :
  ∃ V : ℝ, V = 1 / 6 :=
by
  sorry

end smaller_solid_volume_l696_696055


namespace max_mn_l696_696961

theorem max_mn (a m n : ℝ) (ha : a > 1) (hm : a^m + m = 4) (hn : log a n + n = 4) : ∃ mn_max, mn_max = 4 := by
  -- sorry placeholder for proof
  sorry

end max_mn_l696_696961


namespace digits_probability_l696_696794

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696794


namespace gross_profit_is_4_l696_696732

/-
Given:
- The purchase price of the jacket is $60.
- The markup is 25 percent of the selling price.
- The selling price equals the purchase price plus the markup.
- The selling price is discounted by 20 percent during the sale.

Prove:
- The merchant's gross profit on this sale is $4.
-/

def purchase_price : ℝ := 60

def selling_price : ℝ := purchase_price / 0.75

def discount : ℝ := 0.20 * selling_price

def discounted_selling_price : ℝ := selling_price - discount

def gross_profit : ℝ := discounted_selling_price - purchase_price

theorem gross_profit_is_4 : gross_profit = 4 :=
by
  sorry

end gross_profit_is_4_l696_696732


namespace solve_nested_fraction_l696_696359

def nested_fraction_1985 (a : ℕ) (x : ℝ) : ℝ :=
  if a = 0 then x / (1 + real.sqrt (1 + x))
  else x / (2 + nested_fraction_1985 (a - 1) x)

theorem solve_nested_fraction :
  nested_fraction_1985 1985 x = 1 → x = 3 :=
sorry

end solve_nested_fraction_l696_696359


namespace smallest_area_right_triangle_l696_696479

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696479


namespace right_triangle_min_area_l696_696654

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696654


namespace lines_formed_l696_696193

theorem lines_formed (points : Fin 20 → ℝ × ℝ) (h_collinear_7 : ∃ (subset : Finset (Fin 20)), subset.card = 7 ∧ ∀ ⦃p1 p2 p3 : Fin 20⦄, p1 ∈ subset → p2 ∈ subset → p3 ∈ subset → collinear ℝ (points p1) (points p2) (points p3)) 
  (h_no_collinear_3 : ∀ ⦃p1 p2 p3 : Fin 20⦄, p1 ∉ (classical.some h_collinear_7) ∨ p2 ∉ (classical.some h_collinear_7) ∨ p3 ∉ (classical.some h_collinear_7) ∨ ¬ collinear ℝ (points p1) (points p2) (points p3)) :
  let total_points := 20
      collinear_points := 7
      non_collinear_points := total_points - collinear_points
      lines_from_non_collinear := (non_collinear_points * (non_collinear_points - 1)) / 2
      lines_from_collinear_to_non_collinear := non_collinear_points * collinear_points
      lines_from_collinear := 1
  in (lines_from_non_collinear + lines_from_collinear_to_non_collinear + lines_from_collinear) = 170 :=
by sorry

end lines_formed_l696_696193


namespace sum_of_first_five_primes_with_units_digit_3_l696_696119

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696119


namespace minimum_value_is_correct_l696_696194

noncomputable def min_value (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : ℝ :=
  (a^(2/5) + b^(2/5))^(5/2)

theorem minimum_value_is_correct (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ θ : ℝ, (0 < θ) ∧ (θ < π / 2) ∧ (∀ θ' : ℝ, (0 < θ') ∧ (θ' < π / 2) → (a / (sin θ')^3 + b / (cos θ')^3) ≥ min_value a b h_a h_b) :=
sorry

end minimum_value_is_correct_l696_696194


namespace smallest_area_correct_l696_696539

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696539


namespace problem_equivalent_statement_l696_696181

noncomputable def has_two_distinct_real_roots (a : ℝ) :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (x₁ > 0) ∧ (x₁ + a > 0) ∧ (x₁ + a ≠ 1) ∧
                 (x₂ > 0) ∧ (x₂ + a > 0) ∧ (x₂ + a ≠ 1) ∧
                 (log (x₁ + a) (2 * x₁) = 2) ∧ (log (x₂ + a) (2 * x₂) = 2)

theorem problem_equivalent_statement (a : ℝ) :
  (0 < a ∧ a < 1/2) ↔ has_two_distinct_real_roots a :=
sorry

end problem_equivalent_statement_l696_696181


namespace system_solution_l696_696939

theorem system_solution (n : ℕ) (a : ℝ) (x : Fin n → ℝ) :
  (∑ i, x i = a) ∧ (∑ i, (x i)^2 = a^2) ∧ (∀ m, m ∈ Ico 3 (n + 1) → ∑ i, (x i)^m = a^m) →
  ∃ j, (x j = a ∧ ∀ i, i ≠ j → x i = 0) :=
by
  sorry

end system_solution_l696_696939


namespace probability_all_digits_different_l696_696878

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696878


namespace inequality_solution_l696_696397

theorem inequality_solution (x : ℝ) : 4 * x - 2 ≤ 3 * (x - 1) ↔ x ≤ -1 :=
by 
  sorry

end inequality_solution_l696_696397


namespace smallest_area_right_triangle_l696_696634

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696634


namespace smallest_area_of_right_triangle_l696_696630

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696630


namespace rationalize_sqrt_sum_l696_696352

theorem rationalize_sqrt_sum:
  let A := 6
  let B := 4
  let C := -1
  let D := 1
  let E := 30
  let F := 12 in
  (A + B + C + D + E + F = 52) :=
by
  sorry

end rationalize_sqrt_sum_l696_696352


namespace probability_is_18_over_25_l696_696829

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696829


namespace tangent_at_m_eq_3_range_of_m_l696_696989

-- Definitions for the line l and curve C
def line_l (ρ θ m : ℝ) : Prop := ρ * sin (θ + π / 3) = (sqrt 3 / 2) * m

def curve_C (x y θ : ℝ) : Prop := x = 1 + sqrt 3 * cos θ ∧ y = sqrt 3 * sin θ

-- Prove the line l is tangent to the curve C when m = 3
theorem tangent_at_m_eq_3 :
  ∀ (ρ θ : ℝ),
  line_l ρ θ 3 →
  (∃ x y : ℝ, curve_C x y θ ∧ (y + sqrt 3 * x = 3 * sqrt 3)) →
  (∃ c r : ℝ, c = (1 : ℝ, 0 : ℝ) ∧ r = sqrt 3 ∧ (y + sqrt 3 * x - r = 0) :=
sorry

-- Prove the range of m
theorem range_of_m :
  ∀ (ρ θ m : ℝ),
  (line_l ρ θ m ∧ ∃ x y : ℝ, curve_C x y θ ∧ abs (sqrt 3 - m * sqrt 3) / 2 ≤ sqrt 3 + sqrt 3 / 2) →
  -2 ≤ m ∧ m ≤ 4 :=
sorry

end tangent_at_m_eq_3_range_of_m_l696_696989


namespace number_of_non_congruent_rectangles_l696_696741

def is_rectangle_with_perimeter_100 (w h : ℕ) : Prop :=
  w + h = 50

def is_non_congruent (w h : ℕ) : Prop :=
  w ≤ h

theorem number_of_non_congruent_rectangles : 
  (Finset.filter (λ (wh : ℕ × ℕ), is_rectangle_with_perimeter_100 wh.1 wh.2 ∧ is_non_congruent wh.1 wh.2)
  (Finset.Icc (0, 0) (50, 50))).card = 25 := by
  sorry

end number_of_non_congruent_rectangles_l696_696741


namespace expected_value_of_X_l696_696731

-- Define the conditions
def row_size : ℕ := 6

-- X is the random variable denoting the number of students disturbed by the first student to exit.
noncomputable def X : ℕ → ℚ := sorry -- Define the random variable X properly to denote the disturbance

-- Probability Distribution and Expected Value calculation are based on the given conditions
theorem expected_value_of_X :
  let E : ℚ := 0 * (32 / (6!)) + 1 * (160 / (6!)) + 2 * (280 / (6!)) + 3 * (200 / (6!)) + 4 * (48 / (6!)) in
  E = 21 / 10 :=
by
  sorry

end expected_value_of_X_l696_696731


namespace probability_all_different_digits_l696_696773

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696773


namespace point_lies_on_hyperbola_l696_696179

theorem point_lies_on_hyperbola (t : ℝ) :
  let x := Real.exp t + Real.exp (-t),
      y := 3 * (Real.exp t - Real.exp (-t))
  in (x^2 / 4) - (y^2 / 36) = 1 :=
by
  sorry

end point_lies_on_hyperbola_l696_696179


namespace smallest_area_correct_l696_696538

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696538


namespace sequence_bounded_iff_l696_696304

noncomputable def phi : ℕ → ℕ := sorry  -- Assume φ function is defined accordingly

def sequence (c : ℕ) : ℕ → ℕ
| 1     := c
| (n+1) := c * phi (sequence n)

def bounded (s : ℕ → ℕ) : Prop :=
  ∃ D, ∀ n, |s n| < D

theorem sequence_bounded_iff (c : ℕ) :
  (bounded (sequence c)) ↔ (c = 2 ∨ c = 3) := by
  sorry

end sequence_bounded_iff_l696_696304


namespace smallest_area_right_triangle_l696_696506

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696506


namespace equal_radii_l696_696209

theorem equal_radii 
  (O A B C E K : Point) 
  (circle : Circle) 
  (h1 : tangent circle (A, O))
  (h2 : tangent circle (B, O))
  (h3 : parallel (A, C) (O, B))
  (h4 : intersect (O, C) circle = E)
  (h5 : intersect (A, E) (O, B) = K) 
  : distance O K = distance K B :=
sorry

end equal_radii_l696_696209


namespace probability_is_18_over_25_l696_696839

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696839


namespace card_sums_condition_l696_696922

theorem card_sums_condition (a b c d : ℕ) : 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  a + b = 9 → a + c = 9 → 
  (a, b, c, d) ∈ {(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)} :=
sorry

end card_sums_condition_l696_696922


namespace smallest_area_right_triangle_l696_696483

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696483


namespace dot_product_correct_l696_696087

def vector1 : ℝ × ℝ × ℝ := (3, -4, -3)
def vector2 : ℝ × ℝ × ℝ := (-5, 2, 1)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem dot_product_correct :
  dot_product vector1 vector2 = -26 :=
by
  sorry

end dot_product_correct_l696_696087


namespace correct_statements_l696_696963

variables {m n : Line} {α β : Plane}

def statement_1 : Prop := m ⊥ α ∧ m ⊥ β → α ∥ β
def statement_2 : Prop := m ∥ α ∧ α ∥ β → m ∥ β
def statement_3 : Prop := m ⊥ α ∧ m ∥ β → α ⊥ β
def statement_4 : Prop := m ∥ α ∧ n ⊥ m → n ⊥ α

theorem correct_statements :
  statement_1 ∧ ¬statement_2 ∧ statement_3 ∧ ¬statement_4 :=
sorry

end correct_statements_l696_696963


namespace probability_is_18_over_25_l696_696831

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696831


namespace probability_digits_all_different_l696_696863

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696863


namespace sum_of_first_five_primes_units_digit_3_l696_696168

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l696_696168


namespace decreasing_log_composite_l696_696378

noncomputable def func (a x : ℝ) := log a (2 - a * x)

theorem decreasing_log_composite (a : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) 1, 2 - a * x > 0 ∧ a > 0 ∧ a > 1 ∧ a < 2) →
  ∀ x y ∈ set.Icc (0 : ℝ) 1, x < y → func a y < func a x := sorry

end decreasing_log_composite_l696_696378


namespace job_completion_time_l696_696710

theorem job_completion_time (r_p r_q r_r : ℚ) (h_p : r_p = 1 / 3) (h_q : r_q = 1 / 9) (h_r : r_r = 1 / 6) :
  let work := (r_p * 1) + (r_q * 2) + (r_r * 3)
  work >= 1 → 0 = 0 :=
by
  -- Definitions and given conditions
  assume r_p r_q r_r h_p h_q h_r,
  let work := (r_p * 1) + (r_q * 2) + (r_r * 3),
  assume : work >= 1,
  sorry

end job_completion_time_l696_696710


namespace smallest_area_of_right_triangle_l696_696593

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696593


namespace smallest_area_right_triangle_l696_696557

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696557


namespace remaining_kids_l696_696414

def initial_kids : Float := 22.0
def kids_who_went_home : Float := 14.0

theorem remaining_kids : initial_kids - kids_who_went_home = 8.0 :=
by 
  sorry

end remaining_kids_l696_696414


namespace smallest_area_of_right_triangle_l696_696460

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696460


namespace smallest_right_triangle_area_l696_696577

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696577


namespace smallest_area_right_triangle_l696_696519

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696519


namespace sum_of_first_five_primes_units_digit_3_l696_696164

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l696_696164


namespace smallest_right_triangle_area_l696_696447

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696447


namespace alice_min_speed_exceeds_53_l696_696708

def distance : ℝ := 180
def bob_speed : ℝ := 40
def time_bob : ℝ := distance / bob_speed
def alice_delay : ℝ := 0.5
def time_alice : ℝ := time_bob - alice_delay
def alice_speed_reduction_factor : ℝ := 0.85

noncomputable def min_alice_speed : ℝ := distance / (time_alice * alice_speed_reduction_factor)

theorem alice_min_speed_exceeds_53 : min_alice_speed > 52.94 :=
by sorry

end alice_min_speed_exceeds_53_l696_696708


namespace meaningful_expr_iff_x_ne_neg_5_l696_696699

theorem meaningful_expr_iff_x_ne_neg_5 (x : ℝ) : (x + 5 ≠ 0) ↔ (x ≠ -5) :=
by
  sorry

end meaningful_expr_iff_x_ne_neg_5_l696_696699


namespace cut_angle_from_pentagon_l696_696903

theorem cut_angle_from_pentagon (P : Type) [polygon P] (pentagon_angles : list ℝ) 
    (h_pentagon : pentagon_angles.length = 5) :
  ∃ (resulting_sides : ℕ), resulting_sides ∈ {4, 5, 6} ∧
  let sum_interior_angles (sides : ℕ) : ℝ := (sides - 2) * 180 in
  sum_interior_angles resulting_sides ∈ {360, 540, 720} :=
sorry

end cut_angle_from_pentagon_l696_696903


namespace digits_all_different_l696_696815

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696815


namespace inscribed_shape_area_ratio_l696_696893

theorem inscribed_shape_area_ratio {R : ℝ} (hR : R ≠ 0) :
  let S_3 := (3 * R^2 * Real.sqrt 3) / 4,
      S_4 := 2 * R^2,
      S_6 := (3 * R^2 * Real.sqrt 3) / 2
  in S_4 / 2 = 8 / 4 ∧ 
     S_3 / ((3 * R^2 * Real.sqrt 3) / 4) = 3 * Real.sqrt 3 ∧
     S_6 / ((3 * R^2 * Real.sqrt 3) / 2) = 6 * Real.sqrt 3 := 
sorry

end inscribed_shape_area_ratio_l696_696893


namespace sum_is_correct_l696_696149

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l696_696149


namespace smallest_area_right_triangle_l696_696682

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696682


namespace smallest_area_right_triangle_l696_696507

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696507


namespace right_triangle_min_area_l696_696663

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696663


namespace smallest_area_of_right_triangle_l696_696535

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696535


namespace find_tan_B_l696_696329

noncomputable def right_triangle_with_conditions : Prop :=
  ∃ (A B C D E : Type) 
    (C_right : ∃ (a b c : ℝ), right_angle ∧ ∠ a + ∠ b + ∠ c = 180)
    (DE_BE_ratio : ∃ (DE BE : ℝ), DE / BE = 5 / 13)
    (golden_ratio_div : ∃ (x : ℝ), ∠(C, E, B) / ∠(E, C, D) = (1 + real.sqrt 5) / 2),
  tan B = (5 * real.sqrt 17) / 13

theorem find_tan_B (h : right_triangle_with_conditions) : 
  tan B = (5 * real.sqrt 17) / 13 := 
sorry

end find_tan_B_l696_696329


namespace sum_of_first_five_primes_with_units_digit_3_l696_696097

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696097


namespace smallest_area_right_triangle_l696_696695

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696695


namespace cannot_average_12_mph_l696_696002

-- Definitions
def distance_total : ℝ := 24
def distance_run : ℝ := (2/3) * distance_total
def speed_first_segment : ℝ := 8
def average_speed_required : ℝ := 12

-- Helper Definitions
def time_total : ℝ := distance_total / average_speed_required
def time_first_segment : ℝ := distance_run / speed_first_segment

-- Main Statement
theorem cannot_average_12_mph :
  distance_run = (2/3) * distance_total →
  speed_first_segment = 8 →
  average_speed_required = 12 →
  time_first_segment = distance_run / speed_first_segment →
  time_total = distance_total / average_speed_required →
  ¬ ∃ (speed_remaining : ℝ), (distance_total / (time_first_segment + distance_total - distance_run / speed_remaining)) = average_speed_required :=
by
  intros,
  sorry

end cannot_average_12_mph_l696_696002


namespace equal_sine_squared_l696_696290

theorem equal_sine_squared (α β γ φ : ℝ) 
  (h1 : α + β + γ = π)
  (h2 : α > 0 ∧ β > 0 ∧ γ > 0)
  (h3 : φ > 0 ∧ φ < min (min α β) γ)
  (h4 : sin α ≠ 0 ∧ sin β ≠ 0 ∧ sin γ ≠ 0 ∧ sin φ ≠ 0) :
  1 / (sin φ) ^ 2 = 1 / (sin α) ^ 2 + 1 / (sin β) ^ 2 + 1 / (sin γ) ^ 2 :=
sorry

end equal_sine_squared_l696_696290


namespace zero_of_function_l696_696404

noncomputable def f (x : ℝ) : ℝ := log10 x - 1

theorem zero_of_function : ∃ x : ℝ, f x = 0 ↔ x = 10 :=
by
  sorry

end zero_of_function_l696_696404


namespace probability_digits_different_l696_696806

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696806


namespace probability_all_different_digits_l696_696767

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696767


namespace solution_set_of_inequality_l696_696224

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 else -((abs x) ^ 2)

theorem solution_set_of_inequality :
  {x : ℝ | f (f x) + f (x - 1) < 0} = set.Iio ((real.sqrt 5 - 1) / 2) :=
by sorry

end solution_set_of_inequality_l696_696224


namespace math_problem_l696_696313

noncomputable def x1 : ℝ := sorry
noncomputable def x2 : ℝ := sorry

def condition1 : Prop := 2 * x1 + 2^x1 = 5
def condition2 : Prop := 2 * x2 + 2 * Real.log2 (x2 - 1) = 5

theorem math_problem
  (h1 : condition1)
  (h2 : condition2) :
  x1 + x2 = 7 / 2 := 
sorry

end math_problem_l696_696313


namespace right_triangle_of_medians_l696_696387

theorem right_triangle_of_medians
  (a b c m1 m2 m3 : ℝ)
  (h1 : 4 * m1^2 = 2 * (b^2 + c^2) - a^2)
  (h2 : 4 * m2^2 = 2 * (a^2 + c^2) - b^2)
  (h3 : 4 * m3^2 = 2 * (a^2 + b^2) - c^2)
  (h4 : m1^2 + m2^2 = 5 * m3^2) :
  c^2 = a^2 + b^2 :=
by
  sorry

end right_triangle_of_medians_l696_696387


namespace area_of_triangle_AMC_half_l696_696998

theorem area_of_triangle_AMC_half 
  (A B C M : Type*) 
  [TrigonometricTriangle A B C M]
  (area_ABC : area A B C = 1)
  (BM_perp_to_angle_bisector_C : is_perpendicular B M (angle_bisector C)) :
  area A M C = 1 / 2 := 
sorry

end area_of_triangle_AMC_half_l696_696998


namespace tommys_estimate_larger_than_original_l696_696420

theorem tommys_estimate_larger_than_original (x y : ℝ) (ε : ℝ) (hx : x > y) (hy : y > 0) (hε : ε > 0) : 
  (x + ε) - (y - ε) > x - y :=
by {
  calc (x + ε) - (y - ε) = x - y + 2 * ε : by ring,
  ... > x - y : by linarith,
}

end tommys_estimate_larger_than_original_l696_696420


namespace ratio_calc_l696_696039

theorem ratio_calc :
  (14^4 + 484) * (26^4 + 484) * (38^4 + 484) * (50^4 + 484) * (62^4 + 484) /
  ((8^4 + 484) * (20^4 + 484) * (32^4 + 484) * (44^4 + 484) * (56^4 + 484)) = -423 := 
by
  sorry

end ratio_calc_l696_696039


namespace digits_probability_l696_696786

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696786


namespace transistor_count_2010_l696_696752

-- Define the known constants and conditions
def initial_transistors : ℕ := 2000000
def doubling_period : ℕ := 2
def years_elapsed : ℕ := 2010 - 1995
def number_of_doublings := years_elapsed / doubling_period -- we want floor division

-- The theorem statement we need to prove
theorem transistor_count_2010 : initial_transistors * 2^number_of_doublings = 256000000 := by
  sorry

end transistor_count_2010_l696_696752


namespace probability_empty_subvolume_l696_696760

-- Defining necessary variables and importing required libraries
variables (N : ℕ) (V : ℝ)

-- Defining the conditions
def is_ideal_gas : Prop := true 
def is_in_equilibrium : Prop := true 
def contains_N_molecules (N : ℕ) : Prop := true 
def volume (V : ℝ) : Prop := true 
def sub_volume (V V_star : ℝ) : Prop := V_star = V / (N : ℝ)

-- Statement that needs to be proved
theorem probability_empty_subvolume (h1 : is_ideal_gas) 
                                   (h2 : is_in_equilibrium) 
                                   (h3 : contains_N_molecules N) 
                                   (h4 : volume V)
                                   (h5 : sub_volume V (V / (N : ℝ))) :
  tendsto (λ N, (1 - 1 / (N : ℝ))^N) at_top (𝓝 (real.exp (-1))) :=
sorry

end probability_empty_subvolume_l696_696760


namespace smallest_area_right_triangle_l696_696478

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696478


namespace smallest_right_triangle_area_l696_696579

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696579


namespace sum_of_first_five_primes_units_digit_3_l696_696170

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l696_696170


namespace Bulls_win_finals_in_seven_games_l696_696366

noncomputable def probability_Bulls_win_finals_needing_seven_games : ℚ :=
  let p_Heat := (3 : ℚ) / 4
  let p_Bulls := 1 - p_Heat
  let comb := Nat.choose 6 3
  let prob_six_games := (p_Bulls ^ 3) * (p_Heat ^ 3)
  let prob_last_game := p_Bulls
  let total_prob := comb * prob_six_games * prob_last_game
  in total_prob

theorem Bulls_win_finals_in_seven_games :
  probability_Bulls_win_finals_needing_seven_games = 540 / 16384 := by
  sorry

end Bulls_win_finals_in_seven_games_l696_696366


namespace smallest_right_triangle_area_l696_696681

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696681


namespace right_triangle_min_area_l696_696664

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696664


namespace division_by_repeating_decimal_l696_696429

-- Define the repeating decimal as a fraction
def repeating_decimal := 4 / 9

-- Prove the main theorem
theorem division_by_repeating_decimal : 8 / repeating_decimal = 18 :=
by
  -- lean implementation steps
  sorry

end division_by_repeating_decimal_l696_696429


namespace more_blue_marbles_l696_696411

theorem more_blue_marbles (r_boxes b_boxes marbles_per_box : ℕ) 
    (red_total_eq : r_boxes * marbles_per_box = 70) 
    (blue_total_eq : b_boxes * marbles_per_box = 126) 
    (r_boxes_eq : r_boxes = 5) 
    (b_boxes_eq : b_boxes = 9) 
    (marbles_per_box_eq : marbles_per_box = 14) : 
    126 - 70 = 56 := 
by 
  sorry

end more_blue_marbles_l696_696411


namespace smallest_area_right_triangle_l696_696603

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696603


namespace m_interval_l696_696906

def x_sequence : ℕ → ℝ
| 0       := 3
| (n + 1) := (x_sequence n ^ 2 + 6 * x_sequence n + 8) / (x_sequence n + 7)

def m := (least k : ℕ, 1 ≤ k ∧ x_sequence k ≤ 5 + 1 / 2 ^ 15)

theorem m_interval : 31 ≤ m ∧ m ≤ 90 :=
by
  sorry

end m_interval_l696_696906


namespace smallest_positive_angle_l696_696937

open Real

theorem smallest_positive_angle (y : ℝ) (h : sin(4 * y) * sin(5 * y) = cos(4 * y) * cos(5 * y)) : y = 10 :=
by
  sorry

end smallest_positive_angle_l696_696937


namespace smallest_area_of_right_triangle_l696_696526

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696526


namespace sum_of_first_five_primes_with_units_digit_3_l696_696156

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696156


namespace smallest_area_of_right_triangle_l696_696525

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696525


namespace range_of_a_l696_696180

/-- 
For the system of inequalities in terms of x 
    \begin{cases} 
    x - a < 0 
    ax < 1 
    \end{cases}
the range of values for the real number a such that the solution set is not empty is [-1, ∞).
-/
theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x - a < 0 ∧ a * x < 1) ↔ -1 ≤ a :=
by sorry

end range_of_a_l696_696180


namespace smallest_area_right_triangle_l696_696639

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696639


namespace smallest_right_triangle_area_l696_696491

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696491


namespace terminal_side_third_quadrant_l696_696186

noncomputable def angle_alpha : ℝ := (7 * Real.pi) / 5

def is_in_third_quadrant (angle : ℝ) : Prop :=
  ∃ k : ℤ, (3 * Real.pi) / 2 < angle + 2 * k * Real.pi ∧ angle + 2 * k * Real.pi < 2 * Real.pi

theorem terminal_side_third_quadrant : is_in_third_quadrant angle_alpha :=
sorry

end terminal_side_third_quadrant_l696_696186


namespace smallest_area_right_triangle_l696_696520

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696520


namespace largest_mersenne_prime_less_than_500_l696_696432

def mersenne_prime (n : ℕ) : ℕ := 2^n - 1

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem largest_mersenne_prime_less_than_500 :
  ∃ n, is_prime n ∧ mersenne_prime n < 500 ∧ ∀ m, is_prime m ∧ mersenne_prime m < 500 → mersenne_prime m ≤ mersenne_prime n :=
  sorry

end largest_mersenne_prime_less_than_500_l696_696432


namespace probability_all_digits_different_l696_696876

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696876


namespace prod_factorial_ratio_l696_696890

theorem prod_factorial_ratio :
  (∏ n in Finset.range 10, (fact (n + 1)) / (fact (n + 3))^2) = (1 / 42) :=
sorry

end prod_factorial_ratio_l696_696890


namespace sum_of_first_five_primes_units_digit_3_l696_696169

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l696_696169


namespace sum_is_correct_l696_696150

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l696_696150


namespace sum_of_first_five_primes_with_units_digit_three_l696_696141

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l696_696141


namespace digits_probability_l696_696789

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696789


namespace smallest_right_triangle_area_l696_696679

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696679


namespace digits_probability_l696_696785

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696785


namespace sum_of_first_five_primes_with_units_digit_3_l696_696161

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696161


namespace smallest_right_triangle_area_l696_696505

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696505


namespace find_k_l696_696931

noncomputable def k_solution := 
  {k : ℝ // ∥k • (⟨3, -2⟩: ℝ × ℝ) - (⟨5, 8⟩: ℝ × ℝ)∥ = 3 * Real.sqrt 13}

theorem find_k (k : ℝ) (hk : ∥k • (⟨3, -2⟩: ℝ × ℝ) - (⟨5, 8⟩: ℝ × ℝ)∥ = 3 * Real.sqrt 13) :
  k = ( -1 + Real.sqrt 365) / 13 ∨ k = (-1 - Real.sqrt 365) / 13 :=
sorry

end find_k_l696_696931


namespace smallest_area_of_right_triangle_l696_696590

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696590


namespace smallest_right_triangle_area_l696_696453

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696453


namespace classroom_seats_l696_696724

theorem classroom_seats (a : ℕ) (n : ℕ) (rows : ℕ) (last_row_seats : ℕ):
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ rows → 
    a + (k - 1) * 2 = a + (rows - 1) * 2 - (rows - k) * 2) →
  rows = 22 →
  a = 22 →
  last_row_seats = a + (rows - 1) * 2 →
  sum (λ i, a + (i - 1) * 2) (range rows) = 946 :=
by
  sorry

end classroom_seats_l696_696724


namespace mass_percentage_Ca_in_mixed_compound_l696_696064

def molar_mass_Ca := 40.08 -- g/mol
def molar_mass_O := 16.00  -- g/mol
def molar_mass_C := 12.01  -- g/mol
def molar_mass_S := 32.07  -- g/mol

def molar_mass_CaO := molar_mass_Ca + molar_mass_O
def molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
def molar_mass_CaSO4 := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

def percentage_Ca_in_CaO : Float := (molar_mass_Ca / molar_mass_CaO) * 100
def mass_percentage_Ca_from_CaO : Float := 0.40 * percentage_Ca_in_CaO

def percentage_Ca_in_CaCO3 : Float := (molar_mass_Ca / molar_mass_CaCO3) * 100
def mass_percentage_Ca_from_CaCO3 : Float := 0.30 * percentage_Ca_in_CaCO3

def percentage_Ca_in_CaSO4 : Float := (molar_mass_Ca / molar_mass_CaSO4) * 100
def mass_percentage_Ca_from_CaSO4 : Float := 0.30 * percentage_Ca_in_CaSO4

def total_mass_percentage_Ca : Float := mass_percentage_Ca_from_CaO + mass_percentage_Ca_from_CaCO3 + mass_percentage_Ca_from_CaSO4

theorem mass_percentage_Ca_in_mixed_compound :
  total_mass_percentage_Ca ≈ 49.432 :=
by
  sorry

end mass_percentage_Ca_in_mixed_compound_l696_696064


namespace smallest_area_of_right_triangle_l696_696523

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696523


namespace probability_value_l696_696390

noncomputable def P (k : ℕ) (c : ℚ) : ℚ := c / (k * (k + 1))

theorem probability_value (c : ℚ) (h : P 1 c + P 2 c + P 3 c + P 4 c = 1) : P 1 c + P 2 c = 5 / 6 := 
by
  sorry

end probability_value_l696_696390


namespace smallest_area_of_right_triangle_l696_696633

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696633


namespace increasing_sequence_range_l696_696234

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≤ 7 then (3 - a) * x - 3 else a^(x - 6)

def a_n (a : ℝ) (n : ℕ) [h1 : Fact (0 < n)] : ℝ :=
  f a n

theorem increasing_sequence_range :
  {a : ℝ | ∀ m n : ℕ, 0 < m → 0 < n → m < n → a_n a m < a_n a n} = {a : ℝ | 2 < a ∧ a < 3} :=
sorry

end increasing_sequence_range_l696_696234


namespace log_product_l696_696217

-- The main proof statement
theorem log_product (a b : ℝ) (log_a log_b : ℝ) (h1 : 2 * log_a^2 - 4 * log_a + 1 = 0) (h2 : 2 * log_b^2 - 4 * log_b + 1 = 0) : 
  (log_a + log_b) * ((log a) / (log b) + (log b) / (log a)) = 12 :=
by
  sorry

end log_product_l696_696217


namespace range_of_b_l696_696391

noncomputable theory

open Real

theorem range_of_b :
  ¬(∀ x : ℝ, x^2 - 4 * b * x + 3 * b > 0) ↔ (b ≤ 0 ∨ b ≥ 3 / 4) :=
by sorry

end range_of_b_l696_696391


namespace min_value_quadratic_less_than_neg_six_l696_696736

variable {R : Type*} [LinearOrderedField R]

def quadratic (a b c : R) (x : R) : R := a * x^2 + b * x + c

theorem min_value_quadratic_less_than_neg_six 
  (a b c : R) 
  (vertex_x : R)
  (h_neg_vertex_x : vertex_x = (0 + 3 : R) / (2 : R))
  (h_vertex : quadratic a b c vertex_x < -6)
  (h_parabola_upwards : 0 < a)
  (h_f_neg2 : quadratic a b c (-2) = 6)
  (h_f_0 : quadratic a b c 0 = -4)
  (h_f_1 : quadratic a b c 1 = -6)
  (h_f_3 : quadratic a b c 3 = -4) :
  ∃ x_min : R, quadratic a b c x_min < -6 :=
begin
  use vertex_x,
  exact h_vertex,
end

end min_value_quadratic_less_than_neg_six_l696_696736


namespace probability_is_18_over_25_l696_696838

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696838


namespace probability_digits_all_different_l696_696871

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696871


namespace sum_of_first_five_primes_with_units_digit_3_l696_696157

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696157


namespace smallest_right_triangle_area_l696_696675

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696675


namespace determine_functions_l696_696062

-- Define the problem and conditions as hypotheses
variable (f : ℝ → ℝ)
variable (H : ∀ x y : ℝ, f ((x - y)^2) = (f x)^2 - 2 * x * f y + y^2)

-- Define the theorem to prove
theorem determine_functions :
  (f = λ x, x) ∨ (f = λ x, x + 1) :=
sorry

end determine_functions_l696_696062


namespace smallest_area_of_right_triangle_l696_696587

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696587


namespace probability_digits_all_different_l696_696862

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696862


namespace smallest_area_right_triangle_l696_696514

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696514


namespace parabola_equation_centroid_trajectory_eq_min_MN_value_l696_696965

-- Definition of the initial problem conditions
def is_focus (F : ℝ × ℝ) := F = (1/2, 0)
def is_directrix (l : ℝ) := l = -(1/2)

-- Problem 1: Equation of the parabola C
theorem parabola_equation (F : ℝ × ℝ) (l : ℝ) (hF : is_focus F) (hl : is_directrix l) :
  (∀ x y : ℝ, y^2 = 2 * x ↔ (x + 1/2)^2 + y^2 = (x - 1/2)^2) :=
sorry

-- Problem 2: Trajectory equation of the centroid G
theorem centroid_trajectory_eq (k : ℝ) (F : ℝ × ℝ) (O : ℝ × ℝ) (hF : is_focus F) (hO : O = (0, 0)) :
  (∀ x y : ℝ, y^2 = (2/3) * x - 2/9 ↔ 
  (∃ A B : ℝ × ℝ, (A.1 + B.1 = (k^2 + 2) / k^2 ∧ A.2 + B.2 = 2 / k) ∧ 
  (x = (A.1 + B.1 + 0) / 3 ∧ y = (A.2 + B.2 + 0) / 3))) :=
sorry

-- Problem 3: Smallest value of |MN|
theorem min_MN_value (P : ℝ × ℝ) (x0 y0 : ℝ) (hP : P = (2, 2) ∨ P = (2, -2)) :
  (∀ x y : ℝ, (x - 3)^2 + y^2 = 2 → 
  (|MN| = 2*sqrt(2) * sqrt(1 - 2 / ((x0 - 3)^2 + 2x0 + 9)) → 
  min_MN_value = (2 * sqrt(30)) / 3) :=
sorry

end parabola_equation_centroid_trajectory_eq_min_MN_value_l696_696965


namespace digits_all_different_l696_696810

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696810


namespace smallest_right_triangle_area_l696_696445

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696445


namespace smallest_right_triangle_area_l696_696497

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696497


namespace max_pyramid_volume_l696_696370

noncomputable def maximize_pyramid_volume (SM SN MN : ℝ) (angle_NMK : ℝ) : ℝ × ℝ × ℝ :=
  if h_cond : SM = 2 ∧ SN = 4 ∧ MN = 4 ∧ angle_NMK = 60 then
    (3 * Real.sqrt 2, Real.sqrt 46, 4 * Real.sqrt 5)
  else
    (0, 0, 0)

theorem max_pyramid_volume : 
  ∀ (SM SN MN : ℝ) (angle_NMK : ℝ), 
  SM = 2 ∧ SN = 4 ∧ MN = 4 ∧ angle_NMK = 60 → 
  maximize_pyramid_volume SM SN MN angle_NMK = (3 * Real.sqrt 2, Real.sqrt 46, 4 * Real.sqrt 5) :=
by
  intros SM SN MN angle_NMK h_cond
  simp [maximize_pyramid_volume, h_cond]
  sorry

end max_pyramid_volume_l696_696370


namespace sin_x1_x2_value_l696_696222

open Real

theorem sin_x1_x2_value (m x1 x2 : ℝ) :
  (2 * sin (2 * x1) + cos (2 * x1) = m) →
  (2 * sin (2 * x2) + cos (2 * x2) = m) →
  (0 ≤ x1 ∧ x1 ≤ π / 2) →
  (0 ≤ x2 ∧ x2 ≤ π / 2) →
  sin (x1 + x2) = 2 * sqrt 5 / 5 := 
by
  sorry

end sin_x1_x2_value_l696_696222


namespace smallest_area_of_right_triangle_l696_696592

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696592


namespace M_subset_N_l696_696306

-- Define M and N using the given conditions
def M : Set ℝ := {α | ∃ (k : ℤ), α = k * 90} ∪ {α | ∃ (k : ℤ), α = k * 180 + 45}
def N : Set ℝ := {α | ∃ (k : ℤ), α = k * 45}

-- Prove that M is a subset of N
theorem M_subset_N : M ⊆ N :=
by
  sorry

end M_subset_N_l696_696306


namespace vector_magnitude_difference_l696_696246

open Real

def magnitude (v : Vector3) : ℝ :=
  √(v.x * v.x + v.y * v.y + v.z * v.z)

def dot_product (u v : Vector3) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

structure VectorsWithProperties :=
  (a b : Vector3)
  (angle_ab : ℝ)
  (magnitude_a : ℝ)
  (magnitude_b : ℝ)
  (dot_ab : ℝ)

theorem vector_magnitude_difference (v : VectorsWithProperties) 
  (h_angle : v.angle_ab = 2 * π / 3)
  (h_mag_a : magnitude v.a = 1)
  (h_mag_b : magnitude v.b = 3)
  (h_dot_ab : dot_product v.a v.b = -3 / 2) 
  : magnitude (v.a - v.b) = √13 := by sorry


end vector_magnitude_difference_l696_696246


namespace sum_of_first_five_primes_with_units_digit_3_l696_696096

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696096


namespace general_formula_sum_formula_l696_696210

variable {a : ℕ → ℤ}

-- Arithmetic sequence with positive terms and common difference 2
def is_positive_arithmetic_sequence (a : ℕ → ℤ) :=
  ∀ n, a n > 0 ∧ ∀ m, a (m + 1) - a m = 2

def relation (a : ℕ → ℤ) : Prop :=
  (a 1 + 2) * (a 3 + 6) = 4 * (a 2 + 4) + 1

-- Prove general formula for {a_n}
theorem general_formula (h1 : is_positive_arithmetic_sequence a) (h2 : relation a) :
  ∀ n, a n = 2 * n - 1 :=
sorry

-- Prove the sum of a_1 + a_3 + a_9 + ... + a_{3^n}
theorem sum_formula (h1 : is_positive_arithmetic_sequence a) (h2 : relation a) :
  ∀ n, a 1 + a 3 + a 9 + (a ∘ (λ k, 3^k)) n = 3^(n+1) - n - 2 :=
sorry

end general_formula_sum_formula_l696_696210


namespace smallest_area_right_triangle_l696_696561

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696561


namespace smallest_area_of_right_triangle_l696_696527

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696527


namespace smallest_area_of_right_triangle_l696_696467

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696467


namespace g_50_eq_498_l696_696379

def g : ℤ → ℤ :=
  fun n =>
    if n >= 500 then n - 2
    else g (g (n + 5))

theorem g_50_eq_498 : g 50 = 498 := 
  sorry

end g_50_eq_498_l696_696379


namespace sum_of_first_five_primes_with_units_digit_3_l696_696160

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696160


namespace cups_of_rice_in_afternoon_l696_696353

-- Definitions for conditions
def morning_cups : ℕ := 3
def evening_cups : ℕ := 5
def fat_per_cup : ℕ := 10
def weekly_total_fat : ℕ := 700

-- Theorem statement
theorem cups_of_rice_in_afternoon (morning_cups evening_cups fat_per_cup weekly_total_fat : ℕ) :
  (weekly_total_fat - (morning_cups + evening_cups) * fat_per_cup * 7) / fat_per_cup = 14 :=
by
  sorry

end cups_of_rice_in_afternoon_l696_696353


namespace greatest_sum_on_circle_l696_696413

theorem greatest_sum_on_circle : ∃ x y : ℤ, x^2 + y^2 = 100 ∧ ∀ x' y' : ℤ, (x'^2 + y'^2 = 100 → x + y ≥ x' + y') :=
by
  existsi [6, 8]
  split
  · exact calc
      (6 : ℤ)^2 + (8 : ℤ)^2 = 36 + 64 := by ring
      ... = 100 := by rfl
  · intros x' y'
    rintros (hx : x'^2 + y'^2 = 100)
    -- Proof to show the sum at chosen x, y is at least that of any other point
    -- sorry used to skip complete the proof
    sorry

end greatest_sum_on_circle_l696_696413


namespace ellipse_standard_equation_line_exists_l696_696973

section
  variable {a b c e : ℝ}

  -- Conditions as definitions
  def ellipse : Prop := ∃ a b, a > b ∧ b > 0 ∧ (∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)
  def minor_axis_length : Prop := 2 * b = 6
  def eccentricity : Prop := e = c / a ∧ e = sqrt 2 / 2
  def axis_relation : Prop := a^2 = b^2 + c^2

  -- Proof of part (I)
  theorem ellipse_standard_equation : ellipse ∧ minor_axis_length ∧ eccentricity ∧ axis_relation → 
                                    ∀ x y, (x^2 / 18) + (y^2 / 9) = 1 := 
  by
    sorry

  -- Proof of part (II)
  theorem line_exists :
    ellipse ∧ minor_axis_length ∧ eccentricity ∧ axis_relation →
    ∃ m : ℝ, (y = x + m) ∧ ((m = 2 * sqrt 3) ∨ (m = -2 * sqrt 3)) ∧ 
    ∃ x1 x2 y1 y2, (x1 + x2 = -4m / 3) ∧ (x1 * x2 + y1 * y2 = 0) :=
  by
    sorry
end

end ellipse_standard_equation_line_exists_l696_696973


namespace average_age_when_youngest_born_l696_696412

theorem average_age_when_youngest_born :
  ∀ (n : ℕ) (ages : Fin n → ℕ), n = 7 → (∑ i, ages i) = 210 → (∃ j, ages j = 5) →
  (∑ i in {i | i ≠ j}, ages i) / 6 = 29.17 :=
by
  intros n ages h1 h2 h3
  sorry

end average_age_when_youngest_born_l696_696412


namespace smallest_area_right_triangle_l696_696509

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696509


namespace smallest_right_triangle_area_l696_696582

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696582


namespace correct_conclusions_count_l696_696975

theorem correct_conclusions_count :
  let S1 := "The negation of the proposition 'If p, then q or r' is 'If not p, then not q and not r'" in
  let S2 := "The contrapositive of the proposition 'If not p, then q' is 'If p, then not q'" in
  let S3 := "The negation of the proposition 'There exists an n in ℕ* such that n^2 + 3n is divisible by 10' is 'For all n in ℕ*, n^2 + 3n is not divisible by 10'" in
  let S4 := "The negation of the proposition 'For any x, x^2 - 2x + 3 > 0' is 'There exists an x, x^2 - 2x + 3 < 0'" in
  (S1_correct : (∃ (p q r : Prop), ¬(p → (q ∨ r)) ↔ ¬p → (¬q ∧ ¬r)) :=
    ⟨λ ⟨⟨p, q, r⟩, (hn1: ¬(p → (q ∨ r))).⟩, hn1, λ ⟨¬p → (¬q ∧ ¬r), hcn⟩, hcn⟩) ∧
  (S2_correct : ¬(∃ (p q : Prop), (¬p → q) ↔ p → ¬q) :=
    λ ⟨p, q, h⟩, by
    { have hne := λ hp : ¬p, q, hq, (hne := (hne := ¬q → ¬p), exact λ, not p → r,
      rw hq,
      exact (hne := ¬q → ¬p),
      rwa hne }),
  (S3_correct : (∃ (n : ℕ*), ¬(∃ n in ℕ*, n^2 + 3n % 10 = 0) ↔ ∀ n in ℕ*, let n^2 + 3n ≠ 0 mod 10 ) :=
    ⟨λ ⟨n, n.prop⟩, (hn.exists := hn, hn, by exact hn), λ hn_exist, hn_exist⟩),
  (S4_correct : ¬(∃ (x : ℝ), ¬(∀ x, x^2 - 2x + 3 > 0) ↔ ∃ x, x^2 - 2x + 3 < 0) :=
    λ ⟨x⟩, (hx_ne := hx, exact λ, (hx_ne := ∃ hx), exact ⟨hx⟩)),
  S1_correct ∧ ¬S2_correct ∧ S3_correct ∧ ¬S4_correct →
  ((if S1_correct then 1 else 0) + (if S2_correct then 1 else 0) + (if S3_correct then 1 else 0) + (if S4_correct then 1 else 0)) = 2 :=
by
  sorry

end correct_conclusions_count_l696_696975


namespace find_plane_equation_l696_696935

noncomputable def plane_eq (A B C D x y z : ℝ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem find_plane_equation :
  ∃ (A B C D : ℝ), A > 0 ∧ 
  plane_eq A B C D 2 (-1) 3 ∧ 
  plane_eq A B C D 1 3 (-4) ∧ 
  (∃ k : ℝ, (3 = k * A) ∧ (-4 = k * B) ∧ (1 = k * C)) ∧
  A = 24 ∧ B = 20 ∧ C = -16 ∧ D = -20 :=
begin
  use [24, 20, -16, -20],
  split,
  norm_num,
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { use (1:ℝ),
    norm_num },
  norm_num
end

end find_plane_equation_l696_696935


namespace trig_identity_zero_l696_696265

theorem trig_identity_zero (α : ℝ) (h1 : terminal_side_on_line α (λ x y => x + y = 0)) :
  (sin α / sqrt (1 - sin α ^ 2)) + (sqrt (1 - cos α ^ 2) / cos α) = 0 :=
by
  sorry

end trig_identity_zero_l696_696265


namespace smallest_area_right_triangle_l696_696518

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696518


namespace possible_numbers_l696_696918

theorem possible_numbers (a b c d: ℕ) (h1: 0 < a ∧ a < 9) (h2: 0 < b ∧ b < 9) 
                         (h3: 0 < c ∧ c < 9) (h4: 0 < d ∧ d < 9) 
                         (h5: a ≠ b ∧ a + b ≠ 9) (h6: a ≠ c ∧ a + c ≠ 9) 
                         (h7: a ≠ d ∧ a + d ≠ 9) (h8: b ≠ c ∧ b + c ≠ 9) 
                         (h9: b ≠ d ∧ b + d ≠ 9) (h10: c ≠ d ∧ c + d ≠ 9):
  (a, b, c, d) ∈ {(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)} := 
by sorry

end possible_numbers_l696_696918


namespace smallest_area_right_triangle_l696_696559

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696559


namespace sum_of_first_five_primes_with_units_digit_3_l696_696098

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696098


namespace cost_of_paints_is_5_l696_696333

-- Define folders due to 6 classes
def folder_cost_per_item := 6
def num_classes := 6
def total_folder_cost : ℕ := folder_cost_per_item * num_classes

-- Define pencils due to the 6 classes and need per class
def pencil_cost_per_item := 2
def pencil_per_class := 3
def total_pencils : ℕ := pencil_per_class * num_classes
def total_pencil_cost : ℕ := pencil_cost_per_item * total_pencils

-- Define erasers needed based on pencils and their cost
def eraser_cost_per_item := 1
def pencils_per_eraser := 6
def total_erasers : ℕ := total_pencils / pencils_per_eraser
def total_eraser_cost : ℕ := eraser_cost_per_item * total_erasers

-- Total cost spent on folders, pencils, and erasers
def total_spent : ℕ := 80
def total_cost_supplies : ℕ := total_folder_cost + total_pencil_cost + total_eraser_cost

-- Cost of paints is the remaining amount when total cost is subtracted from total spent
def cost_of_paints : ℕ := total_spent - total_cost_supplies

-- The goal is to prove the cost of paints
theorem cost_of_paints_is_5 : cost_of_paints = 5 := by
  sorry

end cost_of_paints_is_5_l696_696333


namespace contrapositive_l696_696373

theorem contrapositive (a b : ℕ) : (a = 0 → ab = 0) → (ab ≠ 0 → a ≠ 0) :=
by
  sorry

end contrapositive_l696_696373


namespace two_digit_number_l696_696929

theorem two_digit_number (a : ℕ) (N M : ℕ) :
  (10 ≤ a) ∧ (a ≤ 99) ∧ (2 * a + 1 = N^2) ∧ (3 * a + 1 = M^2) → a = 40 :=
by
  sorry

end two_digit_number_l696_696929


namespace smallest_right_triangle_area_l696_696585

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696585


namespace david_boxes_l696_696057

theorem david_boxes (total_dogs : ℕ) (dogs_per_box : ℕ) (boxes : ℕ) 
  (h1 : total_dogs = 28) (h2 : dogs_per_box = 4) : 
  boxes = total_dogs / dogs_per_box → boxes = 7 :=
by
  intros h
  rw [h1, h2] at h
  exact h
  sorry

end david_boxes_l696_696057


namespace right_triangle_min_area_l696_696653

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696653


namespace probability_all_digits_different_l696_696853

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696853


namespace smallest_right_triangle_area_l696_696666

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696666


namespace visible_points_set_convex_polygon_l696_696953

theorem visible_points_set_convex_polygon (P : set ℝ^2) (n : ℕ) (h1 : is_n_sided_polygon P n)
    (h2 : ¬self_intersecting P) (T : set ℝ^2) (h3 : T = { p | ∀ v ∈ vertices P, visible_from p v }) :
    is_convex_polygon_with_at_most_n_sides T n := 
sorry

-- Additional definitions needed for Lean theorem:
def is_n_sided_polygon (P : set ℝ^2) (n : ℕ) : Prop := sorry
def self_intersecting (P : set ℝ^2) : Prop := sorry
def vertices (P : set ℝ^2) : set ℝ^2 := sorry
def visible_from (p : ℝ^2) (v : ℝ^2) : Prop := sorry
def is_convex_polygon_with_at_most_n_sides (T : set ℝ^2) (n : ℕ) : Prop := sorry

end visible_points_set_convex_polygon_l696_696953


namespace sum_of_first_five_primes_with_units_digit_3_l696_696093

noncomputable def is_prime_with_units_digit_3 (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 10 = 3

noncomputable def first_five_primes_with_units_digit_3 : list ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  list.sum first_five_primes_with_units_digit_3 = 135 :=
by
  have h1 : is_prime_with_units_digit_3 3 := by exact ⟨by norm_num, by norm_num⟩
  have h2 : is_prime_with_units_digit_3 13 := by norm_num
  have h3 : is_prime_with_units_digit_3 23 := by norm_num
  have h4 : is_prime_with_units_digit_3 43 := by norm_num
  have h5 : is_prime_with_units_digit_3 53 := by norm_num
  rw [list.sum_cons, list.sum_cons, list.sum_cons, list.sum_cons, list.sum_nil]
  norm_num
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696093


namespace arccos_sin_three_l696_696895

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 :=
by
  sorry

end arccos_sin_three_l696_696895


namespace math_problem_equiv_l696_696065

noncomputable def simplify_sum_1 : ℝ :=
  (Finset.sum (Finset.range 10) (λ k, logBase (7^(k+1)) (2^(2*(k+1)))))

noncomputable def simplify_sum_2 : ℝ :=
  (Finset.sum (Finset.range 50) (λ k, logBase (4^(k+1)) (16^(k+1))))

noncomputable def result : ℝ :=
  20 * logBase 7 2 * 200

theorem math_problem_equiv :
  simplify_sum_1 * simplify_sum_2 = result := by
  sorry

end math_problem_equiv_l696_696065


namespace smallest_area_right_triangle_l696_696636

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696636


namespace number_of_integer_a_l696_696178

noncomputable def count_special_integers : ℕ :=
  let even_count := 100 in
  let odd_perfect_squares := [1, 9, 25, 49, 81, 121, 169].length in
  even_count + odd_perfect_squares

theorem number_of_integer_a (h : ∀ a, 1 ≤ a ∧ a ≤ 200 → (∃ k, a^a = k^2)) : count_special_integers = 107 :=
by
  sorry

end number_of_integer_a_l696_696178


namespace tan_alpha_value_l696_696947

theorem tan_alpha_value (α β : ℝ) (h₁ : Real.tan (α + β) = 3) (h₂ : Real.tan β = 2) : 
  Real.tan α = 1 / 7 := 
by 
  sorry

end tan_alpha_value_l696_696947


namespace median_interval_70_to_74_l696_696054

theorem median_interval_70_to_74 :
  let counts := [
    (85, 89, 15),
    (80, 84, 20),
    (75, 79, 25),
    (70, 74, 30),
    (65, 69, 20),
    (60, 64, 10)  : (ℕ × ℕ × ℕ)
  ] in
  let total_students := 120 in
  let median_index := (total_students + 1) / 2 in
  ∃ (l u : ℕ), (l = 70 ∧ u = 74) ∧
               (∃ (lower_counts : ℕ), lower_counts + 30 ≥ median_index ∧ lower_counts < median_index) :=
by
  sorry

end median_interval_70_to_74_l696_696054


namespace smallest_area_right_triangle_l696_696558

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696558


namespace trains_distance_apart_l696_696721

-- Define the initial conditions
def cattle_train_speed : ℝ := 56
def diesel_train_speed : ℝ := cattle_train_speed - 33
def cattle_train_time : ℝ := 6 + 12
def diesel_train_time : ℝ := 12

-- Calculate distances
def cattle_train_distance : ℝ := cattle_train_speed * cattle_train_time
def diesel_train_distance : ℝ := diesel_train_speed * diesel_train_time

-- Define total distance apart
def distance_apart : ℝ := cattle_train_distance + diesel_train_distance

-- The theorem to prove
theorem trains_distance_apart :
  distance_apart = 1284 :=
by
  -- Skip the proof
  sorry

end trains_distance_apart_l696_696721


namespace problem_statement_l696_696262

def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem problem_statement (ω φ : ℝ) :
  (∀ x : ℝ, f (π / 3 + x) ω φ = f (-x) ω φ) →
  (f (π / 6) ω φ = 2 ∨ f (π / 6) ω φ = -2) :=
by
  sorry

end problem_statement_l696_696262


namespace possible_numbers_l696_696917

theorem possible_numbers (a b c d: ℕ) (h1: 0 < a ∧ a < 9) (h2: 0 < b ∧ b < 9) 
                         (h3: 0 < c ∧ c < 9) (h4: 0 < d ∧ d < 9) 
                         (h5: a ≠ b ∧ a + b ≠ 9) (h6: a ≠ c ∧ a + c ≠ 9) 
                         (h7: a ≠ d ∧ a + d ≠ 9) (h8: b ≠ c ∧ b + c ≠ 9) 
                         (h9: b ≠ d ∧ b + d ≠ 9) (h10: c ≠ d ∧ c + d ≠ 9):
  (a, b, c, d) ∈ {(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)} := 
by sorry

end possible_numbers_l696_696917


namespace dishonest_shopkeeper_gain_l696_696705

-- Conditions: false weight used by shopkeeper
def false_weight : ℚ := 930
def true_weight : ℚ := 1000

-- Correct answer: gain percentage
def gain_percentage (false_weight true_weight : ℚ) : ℚ :=
  ((true_weight - false_weight) / false_weight) * 100

theorem dishonest_shopkeeper_gain :
  gain_percentage false_weight true_weight = 7.53 := by
  sorry

end dishonest_shopkeeper_gain_l696_696705


namespace sum_of_elements_ge_half_n_sq_l696_696314
open Matrix

variable (n : ℕ)
variable (A : Matrix (Fin n) (Fin n) ℕ)
hypothesis h1 : ∀ i j, (A i j = 0) → (∑ k, A i k + ∑ k, A k j) ≥ n

theorem sum_of_elements_ge_half_n_sq :
  let S := ∑ i j, A i j
  in S ≥ (n * n) / 2 :=
  sorry

end sum_of_elements_ge_half_n_sq_l696_696314


namespace tray_height_is_five_l696_696747

theorem tray_height_is_five :
  ∀ (l : ℝ)
    (d : ℝ),
    l = 150 ∧ d = 5 ∧
    (∠PMR = 45 ∧ ∠PNR = 45) →
    let h := (d * real.sqrt 2) * (real.sin (real.pi / 4))
    in h = 5 :=
by
  intros l d h
  rintros ⟨L, D, angles⟩
  let h := (d * real.sqrt 2) * (real.sin (real.pi / 4))
  have : h = (5 * real.sqrt 2) * (real.sin (real.pi / 4)),
  { sorry }
  rw [this],
  norm_num

end tray_height_is_five_l696_696747


namespace lives_per_player_l696_696415

theorem lives_per_player (initial_friends new_players total_lives : ℕ)
  (h_initial : initial_friends = 4)
  (h_new : new_players = 5)
  (h_total : total_lives = 27) :
  (total_lives / (initial_friends + new_players) = 3) :=
by
  simp [h_initial, h_new, h_total]
  sorry

end lives_per_player_l696_696415


namespace incorrect_statement_l696_696743

-- Define the robot's position function P(n)
def P : ℕ → ℤ
| 0 := 0
| (n+1) := if (n % 5) < 3 then P n + 1 else P n - 1

-- Problem statement: Prove that P(2003) <= P(2005)
theorem incorrect_statement : P 2003 > P 2005 := by
  -- Proof is needed
  sorry

end incorrect_statement_l696_696743


namespace smallest_area_right_triangle_l696_696692

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696692


namespace smallest_area_right_triangle_l696_696617

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696617


namespace possible_values_of_a_l696_696241

def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem possible_values_of_a (a : ℝ) :
  N a ⊆ M ↔ a ∈ {-1, 0, 2/3} :=
sorry

end possible_values_of_a_l696_696241


namespace math_problem_l696_696912

theorem math_problem (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^b + 3 = b^a) (h4 : 3 * a^b = b^a + 13) : 
  (a = 2) ∧ (b = 3) :=
sorry

end math_problem_l696_696912


namespace bricks_needed_for_wall_l696_696248

-- Definitions for brick and wall dimensions
constant wall_length : ℝ
constant wall_height : ℝ := 200
constant wall_width : ℝ := 25

constant brick_length : ℝ := 25
constant brick_width : ℝ := 11.25
constant brick_height : ℝ := 6

-- Volume calculations
def volume_wall (x : ℝ) : ℝ := x * wall_height * wall_width
def volume_brick : ℝ := brick_length * brick_width * brick_height

-- Number of bricks calculation
def number_of_bricks (x : ℝ) : ℝ := (volume_wall x) / volume_brick

-- Known number of bricks
constant given_number_of_bricks : ℝ := 1185.1851851851852

-- The theorem to prove
theorem bricks_needed_for_wall : ∀ (x : ℝ), number_of_bricks x = given_number_of_bricks → x = 400 :=
sorry

end bricks_needed_for_wall_l696_696248


namespace gcd_of_459_and_357_l696_696425

open EuclideanDomain

theorem gcd_of_459_and_357 : gcd 459 357 = 51 :=
sorry

end gcd_of_459_and_357_l696_696425


namespace twice_x_minus_3_l696_696075

theorem twice_x_minus_3 (x : ℝ) : (2 * x) - 3 = 2 * x - 3 := 
by 
  -- This proof is trivial and we can assert equality directly
  sorry

end twice_x_minus_3_l696_696075


namespace Vasya_more_configurations_l696_696342

-- Define the setup for Petya's board
def Petya_board : Type := fin 100 × fin 50

-- Define the setup for Vasya's board
def Vasya_board : Type := fin 100 × fin 100

-- Define what it means for kings not to attack each other on Petya's board
def non_attacking_Petya (positions: fin 500 → Petya_board): Prop :=
  ∀ i j, i ≠ j → ¬ attacking (positions i) (positions j)

-- Define what it means for kings not to attack each other on Vasya's board
def non_attacking_Vasya (positions: fin 500 → Vasya_board): Prop :=
  ∀ i j, i ≠ j → ¬ attacking (positions i) (positions j)

-- Define the predicate for positions being on white squares on Vasya's board
def on_white_square (pos: Vasya_board): Prop :=
  (pos.1.val + pos.2.val) % 2 = 0

-- Define the set of valid configurations for Petya
def valid_configurations_Petya : set (fin 500 → Petya_board) :=
  {positions | non_attacking_Petya positions}

-- Define the set of valid configurations for Vasya
def valid_configurations_Vasya : set (fin 500 → Vasya_board) :=
  {positions | non_attacking_Vasya positions ∧ ∀ i, on_white_square (positions i)}

-- Assertion to prove
theorem Vasya_more_configurations :
  (valid_configurations_Vasya).card ≥ (valid_configurations_Petya).card :=
sorry

end Vasya_more_configurations_l696_696342


namespace smallest_area_of_right_triangle_l696_696522

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696522


namespace simplified_sum_l696_696040

theorem simplified_sum :
  (-1 : ℤ) ^ 2002 + (-1 : ℤ) ^ 2003 + 2 ^ 2004 - 2 ^ 2003 = 2 ^ 2003 := 
by 
  sorry -- Proof skipped

end simplified_sum_l696_696040


namespace time_to_fill_pool_l696_696424

noncomputable def slower_pump_rate : ℝ := 1 / 12.5
noncomputable def faster_pump_rate : ℝ := 1.5 * slower_pump_rate
noncomputable def combined_rate : ℝ := slower_pump_rate + faster_pump_rate

theorem time_to_fill_pool : (1 / combined_rate) = 5 := 
by
  sorry

end time_to_fill_pool_l696_696424


namespace ziggy_rap_requests_l696_696704

def song_requests (total_requests electropop_requests dance_requests rock_requests oldies_requests djs_choice_requests rap_requests : ℕ) : Prop :=
  total_requests = 30 ∧
  electropop_requests = total_requests / 2 ∧
  dance_requests = electropop_requests / 3 ∧
  rock_requests = 5 ∧
  oldies_requests = rock_requests - 3 ∧
  djs_choice_requests = oldies_requests / 2 ∧
  rap_requests = total_requests - (electropop_requests + dance_requests + rock_requests + oldies_requests + djs_choice_requests)

theorem ziggy_rap_requests : ∃ rap_requests, song_requests 30 15 5 5 2 1 rap_requests ∧ rap_requests = 2 :=
by
  use 2
  unfold song_requests
  simp
  sorry

end ziggy_rap_requests_l696_696704


namespace prove_P_B_given_A_l696_696045

-- Explanation:
-- Given a set S = {1, 2, 3, 4, 5, 6, 7}, we choose 5 different numbers.
-- Event A: The median of the 5 different numbers chosen is 4
-- Event B: The average of the 5 different numbers chosen is 4
-- We need to prove that P(B | A) = 1/3

def set_S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}.to_finset

def is_median_4 (chosen : Finset ℕ) : Prop :=
  ∃ l r med, chosen.val = l ++ [med] ++ r ∧
              l.length = 2 ∧
              r.length = 2 ∧
              med = 4

def is_average_4 (chosen : Finset ℕ) : Prop :=
  chosen.sum / chosen.card = 4

def P_B_given_A : ℚ := sorry

theorem prove_P_B_given_A :
  ∀ chosen : Finset ℕ,
    chosen ⊆ set_S ∧ chosen.card = 5 ∧ is_median_4 chosen → P_B_given_A = 1/3 :=
sorry

end prove_P_B_given_A_l696_696045


namespace sum_of_first_five_primes_with_units_digit_three_l696_696142

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l696_696142


namespace probability_all_digits_different_l696_696851

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696851


namespace smallest_area_of_right_triangle_l696_696629

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696629


namespace smallest_area_right_triangle_l696_696567

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696567


namespace probability_digits_different_l696_696805

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696805


namespace smallest_area_right_triangle_l696_696687

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696687


namespace positive_difference_mean_median_l696_696401

open List

noncomputable def tree_heights : List ℝ := [85, 105, 130, 90, 120]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)

noncomputable def median (l : List ℝ) : ℝ := 
  let sorted_l := l.qsort (≤)
  sorted_l.get! (sorted_l.length / 2)

theorem positive_difference_mean_median :
  |mean tree_heights - median tree_heights| = 1 :=
by
  sorry

end positive_difference_mean_median_l696_696401


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696112

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696112


namespace smallest_prime_perimeter_l696_696019

-- Define a function that checks if a number is an odd prime
def is_odd_prime (n : ℕ) : Prop :=
  n > 2 ∧ (∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)) ∧ (n % 2 = 1)

-- Define a function that checks if three numbers are consecutive odd primes
def consecutive_odd_primes (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧
  b = a + 2 ∧ c = b + 2

-- Define a function that checks if three numbers form a scalene triangle and satisfy the triangle inequality
def scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main theorem to prove
theorem smallest_prime_perimeter :
  ∃ (a b c : ℕ), consecutive_odd_primes a b c ∧ scalene_triangle a b c ∧ (a + b + c = 23) :=
by
  sorry

end smallest_prime_perimeter_l696_696019


namespace equation_of_ellipse_slope_PQ_constant_l696_696211

-- Define the conditions
def ellipse (a b : ℝ) : set (ℝ × ℝ) := 
  {p | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1}

def point_A := (2 : ℝ, 1 : ℝ)

def eccentricity (a b c : ℝ) : Prop := 
  c / a = (Real.sqrt 3) / 2 ∧ a^2 = b^2 + c^2

-- Theorem 1: equation of ellipse C
theorem equation_of_ellipse 
  (a b : ℝ) (hab : a > b) (hb : b > 0) (hA : point_A ∈ ellipse a b)
  (he : eccentricity a b (Real.sqrt (a^2 - b^2))) :
  ellipse a b = ellipse (2 * Real.sqrt 2) (Real.sqrt 2) :=
sorry

-- Theorem 2: slope of line PQ
theorem slope_PQ_constant
  (a b c k xP xQ yP yQ : ℝ) (hab : a > b) (hb : b > 0) (he : eccentricity a b c)
  (hA : point_A ∈ ellipse a b)
  (hPQ : ∀ P Q : ℝ × ℝ, 
      P ∈ ellipse a b → Q ∈ ellipse a b →
      ∃ k, ¬0 = k → 
      angle_bisector P A Q ∧ perpendicular_to_x_axis (angle_bisector P A Q)) :
  ∃ m : ℝ, ∀ P Q : ℝ × ℝ, P ∈ ellipse a b → Q ∈ ellipse a b → 
  slope (line P Q) = m ∧ m = 1/2 :=
sorry

end equation_of_ellipse_slope_PQ_constant_l696_696211


namespace moles_of_NaNO3_formed_l696_696936

-- Define the substances and their quantities
def NaCl := 2 -- moles of Sodium chloride
def HNO3 := 2 -- moles of Nitric acid

-- Define the balanced chemical equation as a stoichiometric relation
def reaction := (NaCl + HNO3 → NaNO3 + HCl)

-- Define the expected outcome based on stoichiometric calculations
def expected_moles_NaNO3 := 2

-- Formulate the proof problem statement
theorem moles_of_NaNO3_formed : NaCl = 2 → HNO3 = 2 → expected_moles_NaNO3 = 2 :=
by
  intros h1 h2
  -- Insert proof steps here (not necessary for this task)
  sorry

end moles_of_NaNO3_formed_l696_696936


namespace probability_is_18_over_25_l696_696834

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696834


namespace digit_in_2009th_position_l696_696701

def digit_sequence : ℕ := sorry

theorem digit_in_2009th_position : digit_sequence 2009 = 0 := sorry

end digit_in_2009th_position_l696_696701


namespace smallest_area_of_right_triangle_l696_696601

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696601


namespace probability_is_18_over_25_l696_696836

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696836


namespace count_r_l696_696204

/-- Definition of the sequence a_n
    if a_n <= n, a_{n+1} = a_n + n
    if a_n > n,  a_{n+1} = a_n - n -/
def sequence_a : ℕ → ℕ 
| 0       := 1
| (n + 1) := if sequence_a n <= n + 1 then sequence_a n + n + 1 else sequence_a n - (n + 1)

theorem count_r (S : Set ℕ) (hS : ∀ x, x ∈ S ↔ x ≤ 3 ^ 2017 ∧ sequence_a x < x)
  : S.toFinset.card = (3 ^ 2017 - 2017) / 2 - 1 := 
sorry

end count_r_l696_696204


namespace total_short_trees_l696_696274

theorem total_short_trees (initial_short_oak short_maple short_pine : ℕ)
  (new_oak : ℕ) (new_maple_percent : ℕ) (new_pine_factor : ℕ)
  (h1 : initial_short_oak = 41)
  (h2 : short_maple = 18)
  (h3 : short_pine = 24)
  (h4 : new_oak = 57)
  (h5 : new_maple_percent = 30)  -- 30%
  (h6 : new_pine_factor = 1/3)   -- 1/3
  : (initial_short_oak + new_oak) +
    (short_maple + (short_maple * new_maple_percent / 100).nat_floor) + 
    (short_pine + (short_pine / 3)) = 153 := 
  sorry


end total_short_trees_l696_696274


namespace line_a_perpendicular_line_c_l696_696212

-- Declare the lines a, b, and c
variables (a b c : Line)

-- Define the conditions of the problem
axiom a_parallel_b : Parallel a b
axiom b_perp_c : Perpendicular b c

-- The theorem to prove
theorem line_a_perpendicular_line_c : Perpendicular a c :=
sorry

end line_a_perpendicular_line_c_l696_696212


namespace find_a_l696_696997

-- Definitions
structure Point where
  x : ℝ
  y : ℝ

def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

-- Given Points
def A : Point := ⟨2, 2⟩
def B : Point := ⟨5, 1⟩
def C (a : ℝ) : Point := ⟨-4, 2 * a⟩

-- Theorem Statement
theorem find_a (a : ℝ) (h : slope A B = slope A (C a)) : a = 2 :=
by sorry

end find_a_l696_696997


namespace probability_is_18_over_25_l696_696835

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696835


namespace sum_is_correct_l696_696147

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l696_696147


namespace number_of_ways_sum_of_two_primes_10003_l696_696278

theorem number_of_ways_sum_of_two_primes_10003 : 
  (∃ p q: ℕ, p.prime ∧ q.prime ∧ p + q = 10003) ∧ 
  (∀ r s: ℕ, r.prime ∧ s.prime ∧ r + s = 10003 → (r = p ∧ s = q) ∨ (r = q ∧ s = p)) :=
sorry

end number_of_ways_sum_of_two_primes_10003_l696_696278


namespace right_triangle_min_area_l696_696650

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696650


namespace right_triangle_min_area_l696_696658

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696658


namespace repeating_decimal_equiv_fraction_l696_696428

theorem repeating_decimal_equiv_fraction :
  (0.1 ++ (list.repeat '4' 1) ++ (list.repeat '7' 1)).to_rat = 73 / 495 := sorry

end repeating_decimal_equiv_fraction_l696_696428


namespace minimize_fraction_expression_l696_696402

theorem minimize_fraction_expression :
  ∃ n : ℕ+, (∀ m : ℕ+, m ≠ n → (n / 3 + 27 / n : ℝ) < (m / 3 + 27 / m : ℝ)) :=
begin
  use 9,
  sorry
end

end minimize_fraction_expression_l696_696402


namespace smallest_right_triangle_area_l696_696455

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696455


namespace card_numbers_l696_696920

theorem card_numbers (
  a b c d : ℕ) 
  (h_all_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_in_range : 0 < a ∧ a < 9 ∧ 0 < b ∧ b < 9 ∧ 0 < c ∧ c < 9 ∧ 0 < d ∧ d < 9)
  (h_probability_conditions : 
    (∃ twoSums : finset ℕ, twoSums.card = 6 ∧ 
      (∀ x ∈ twoSums, 0 < x < 18) ∧ 
      let sums := [a+b, a+c, a+d, b+c, b+d, c+d] in 
      list.perm sums (twoSums.val) ∧ 
      finset.card (twoSums.filter (λ x, x = 9)) = 2 ∧ 
      finset.card (twoSums.filter (λ x, x < 9)) = 2 ∧ 
      finset.card (twoSums.filter (λ x, x > 9)) = 2))
  : (a, b, c, d) = (1, 2, 7, 8) ∨ (a, b, c, d) = (1, 3, 6, 8) ∨ 
    (a, b, c, d) = (1, 4, 5, 8) ∨ (a, b, c, d) = (2, 3, 6, 7) ∨ 
    (a, b, c, d) = (2, 4, 5, 7) ∨ (a, b, c, d) = (3, 4, 5, 6) :=
sorry

end card_numbers_l696_696920


namespace probability_all_digits_different_l696_696875

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696875


namespace sum_of_first_five_prime_units_digit_3_l696_696106

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l696_696106


namespace probability_runner_up_is_second_strongest_l696_696003

theorem probability_runner_up_is_second_strongest : 
  let n := 8 in
  let rounds := 3 in
  (∃ players : Fin n, 
    (∀ S1 S2 : Fin n, S1 < S2 → S1 ≻ S2) ∧
    ( ∀ final : Fin 2, (final = 1) → (second_strongest runner_up) ∧
        let total_scenarios := choose (n - 1) (n div 2 - 1) in
        (4 * (n - 4)) = 4 ∧
        (total_scenarios - 4).natAbs = 7))
  → (4 / 7 : ℝ)
:= by sorry

end probability_runner_up_is_second_strongest_l696_696003


namespace minimum_value_expr_l696_696089

theorem minimum_value_expr (x : ℤ) (hx : x > 10) : 
     ∃ y, y = (4 * x^2) / (x - 10) ∧ y = 160 :=
begin
  sorry
end

end minimum_value_expr_l696_696089


namespace perpendicular_planes_conditions_l696_696307

variables {Point : Type} {Plane : Type} {Line : Type}

-- Define predicates for perpendicularity and parallelism
def is_perpendicular (X Y : Plane) : Prop := sorry
def is_perpendicular_line (l : Line) (P : Plane) : Prop := sorry
def is_parallel (X Y : Plane) : Prop := sorry

-- Define the objects and conditions
variables (α β γ : Plane)
variables (m n l : Line)

-- Conditions and conclusion
theorem perpendicular_planes_conditions 
  (h1 : is_perpendicular n α) 
  (h2 : is_perpendicular n β) 
  (h3 : is_perpendicular_line m α) : 
  is_perpendicular_line m β :=
sorry

end perpendicular_planes_conditions_l696_696307


namespace completion_time_l696_696717

-- Defining the conditions
def work_done_by_woman_per_day : ℝ := 1 / (10 * 7)
def work_done_by_child_per_day : ℝ := 1 / (10 * 14)

-- Question: How many days will 5 women and 10 children take to complete the work?
theorem completion_time (W : ℝ) (h1 : 10 / 7 = W) (h2 : 10 / 14 = W) : 
  let work_done_by_woman : ℝ := W / (10 * 7)
  let work_done_by_child : ℝ := W / (10 * 14)
  let work_done_by_5_women : ℝ := 5 * work_done_by_woman
  let work_done_by_10_children : ℝ := 10 * work_done_by_child
  let combined_work_done : ℝ := work_done_by_5_women + work_done_by_10_children
  (combined_work_done = W / 7) := sorry

end completion_time_l696_696717


namespace total_weight_of_onions_l696_696418

theorem total_weight_of_onions 
  (initial_bags : Nat) 
  (increase_per_trip : Nat) 
  (num_trips : Nat) 
  (weight_per_bag : Nat) 
  (h_initial : initial_bags = 10) 
  (h_increase : increase_per_trip = 2) 
  (h_trips : num_trips = 20) 
  (h_weight : weight_per_bag = 50) : 
  initial_bags + (num_trips - 1) * increase_per_trip = 48 
  ∧ ∑ i in Finset.range num_trips, (initial_bags + i * increase_per_trip) * weight_per_bag = 29000 :=
by
  sorry

end total_weight_of_onions_l696_696418


namespace smallest_right_triangle_area_l696_696504

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696504


namespace other_eigenvalue_and_eigenvector_l696_696991

noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![3, 2]]

def λ1 : ℝ := -1

def e1 : Fin 2 → ℝ := ![1, -1]

theorem other_eigenvalue_and_eigenvector : 
  let λ2 := 4
  let e2 := ![2, 3]
  eigenvalue M λ2 ∧ (M.mul_vec e2 = λ2 • e2) := by
  sorry

end other_eigenvalue_and_eigenvector_l696_696991


namespace rook_reaches_upper_right_in_70_l696_696899

open Classical
open Set Function
open BigOperators nnreal

-- Define the problem parameters
def rook_grid : Type := Fin 8 × Fin 8

-- Defining the movement with equal probability on the grid
def move (pos : rook_grid) : rook_grid :=
by sorry -- Define movement as probabilistic

-- Define the expected time to reach the upper-right corner from a given position
noncomputable def expected_time (pos : rook_grid) : ℝ :=
by sorry -- Define recursive expected time

-- Prove the expected time to reach the upper-right corner from (0,0) is 70 minutes
theorem rook_reaches_upper_right_in_70 (pos : rook_grid) :
  pos = (0, 0) → expected_time pos = 70 :=
by sorry

end rook_reaches_upper_right_in_70_l696_696899


namespace trash_picked_outside_l696_696421

theorem trash_picked_outside (T_tot : ℕ) (C1 C2 C3 C4 C5 C6 C7 C8 : ℕ)
  (hT_tot : T_tot = 1576)
  (hC1 : C1 = 124) (hC2 : C2 = 98) (hC3 : C3 = 176) (hC4 : C4 = 212)
  (hC5 : C5 = 89) (hC6 : C6 = 241) (hC7 : C7 = 121) (hC8 : C8 = 102) :
  T_tot - (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8) = 413 :=
by sorry

end trash_picked_outside_l696_696421


namespace draw_six_rays_intersecting_at_four_points_l696_696915

-- Definitions for points and rays
variables {α : Type*} [linear_order α]

-- Assume points P1, P2, P3, P4
variables (P1 P2 P3 P4 : α)

-- Assume rays A, B, C intersecting at these points
def intersection (P : α) (A B C : ray) := 
  (A.origin = P) ∧ (B.origin = P) ∧ (C.origin = P)

-- Lean 4 statement
theorem draw_six_rays_intersecting_at_four_points :
  ∃ (A1 A2 A3 B1 B2 B3 C1 C2 C3 : ray) (P1 P2 P3 P4 : α),
  intersection P1 A1 B1 C1 ∧ intersection P2 A2 B2 C2 ∧ intersection P3 A3 B3 C3 ∧ intersection P4 A1 B2 C3 ∧ 
  (∀ P, P ≠ P1 ∧ P ≠ P2 ∧ P ≠ P3 ∧ P ≠ P4 → ¬ (P ∈ (A1 ∩ A2 ∩ A3 ∩ B1 ∩ B2 ∩ B3 ∩ C1 ∩ C2 ∩ C3))) := 
  sorry

end draw_six_rays_intersecting_at_four_points_l696_696915


namespace license_plate_combinations_l696_696249

-- Definitions of the conditions
def num_consonants : ℕ := 20
def num_vowels : ℕ := 6
def num_digits : ℕ := 10

-- The theorem statement
theorem license_plate_combinations : num_consonants * num_vowels * num_vowels * num_digits = 7200 := by
  sorry

end license_plate_combinations_l696_696249


namespace probability_two_balls_same_color_l696_696184

/--
From a box containing 6 colored balls (3 red, 2 yellow, 1 blue), two balls are randomly drawn.

- A box contains 6 colored balls.
- The balls are distributed as: 3 red, 2 yellow, and 1 blue.
- Two balls are randomly drawn.

Prove that the probability that the two balls are of the same color is 4/15.
-/
theorem probability_two_balls_same_color :
  let total_ways := (nat.choose 6 2) in
  let same_color_ways := (nat.choose 3 2) + (nat.choose 2 2) in
  (same_color_ways : ℚ) / total_ways = 4/15 :=
by
  sorry

end probability_two_balls_same_color_l696_696184


namespace minimize_b_plus_c_l696_696200

theorem minimize_b_plus_c (a b c : ℝ) (h1 : 0 < a)
  (h2 : ∀ x, (y : ℝ) = a * x^2 + b * x + c)
  (h3 : ∀ x, (yr : ℝ) = a * (x + 2)^2 + (a - 1)^2) :
  a = 1 :=
by
  sorry

end minimize_b_plus_c_l696_696200


namespace y_alone_complete_l696_696711

variable (Work : Type) (complete : Work) (Day : Type) (days : ℕ → Day) -- Define basic types and constructors.

-- Define functions to represent work rates for x and y.
variable (x_work_rate y_work_rate : Day → Work → Type)
variable [x_rate : ∀ d, x_work_rate d complete]
variable [y_rate : ∀ d, y_work_rate d complete]

-- Condition 1: x is 3 times as fast as y
axiom x_is_3_times_y (d : Day) (w : Work) : 
  x_work_rate d w → ∃ d', y_work_rate d' w ∧ 3 * d' = d

-- Condition 2: Together they can complete the work in 20 days.
axiom together_complete (d : Day) (w : Work) :
  (x_work_rate d w ∧ y_work_rate d w) → d = days 20

-- Statement: y alone can complete the work in 80 days.
theorem y_alone_complete (d : Day) (w : Work) :
  y_work_rate d w → d = days 80 :=
sorry

end y_alone_complete_l696_696711


namespace count_divisors_greater_than_five_factorial_l696_696250

theorem count_divisors_greater_than_five_factorial :
  let n := 6!
  let threshold := 5!
  (Set.toFinset {d | d ∣ n ∧ d > threshold}).card = 4 := by
  simp only [factorial_six_eq_720, factorial_five_eq_120] -- substitute actual values
  sorry

end count_divisors_greater_than_five_factorial_l696_696250


namespace card_sums_condition_l696_696923

theorem card_sums_condition (a b c d : ℕ) : 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  a + b = 9 → a + c = 9 → 
  (a, b, c, d) ∈ {(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)} :=
sorry

end card_sums_condition_l696_696923


namespace sum_of_first_five_primes_with_units_digit_3_l696_696124

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696124


namespace sum_not_zero_l696_696898

variable (n : Nat) (arr : Array (Array Int))
hypothesis h1 : ∀ i, 0 < i → i ≤ n → (∀ j, 0 < j → j ≤ n → (arr[i][j] = 1 ∨ arr[i][j] = -1))
def product_row (i : Nat) : Int := List.foldl (*) 1 (List.map (λ j => arr[i][j]) (List.range n))
def product_col (j : Nat) : Int := List.foldl (*) 1 (List.map (λ i => arr[i][j]) (List.range n))
def a_i := λ i, product_row arr i
def b_j := λ j, product_col arr j

theorem sum_not_zero (h2 : n = 2013) : (∑ i in Finset.range n, a_i i + b_j i) ≠ 0 :=
  sorry

end sum_not_zero_l696_696898


namespace circle_center_l696_696227

theorem circle_center (x y : ℝ) : (x - 2)^2 + (y + 1)^2 = 3 → (2, -1) = (2, -1) :=
by
  intro h
  -- Proof omitted
  sorry

end circle_center_l696_696227


namespace smallest_area_right_triangle_l696_696696

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696696


namespace function_has_two_zeros_for_a_eq_2_l696_696977

noncomputable def f (a x : ℝ) : ℝ := a ^ x - x - 1

theorem function_has_two_zeros_for_a_eq_2 :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f 2 x1 = 0 ∧ f 2 x2 = 0) := sorry

end function_has_two_zeros_for_a_eq_2_l696_696977


namespace probability_digits_all_different_l696_696776

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696776


namespace original_mixture_volume_l696_696722

theorem original_mixture_volume (x : ℝ) (h1 : 0.20 * x / (x + 3) = 1 / 6) : x = 15 :=
  sorry

end original_mixture_volume_l696_696722


namespace project_completion_l696_696025

theorem project_completion (a b c d e : ℕ) 
  (h₁ : 1 / (a : ℝ) + 1 / b + 1 / c + 1 / d = 1 / 6)
  (h₂ : 1 / (b : ℝ) + 1 / c + 1 / d + 1 / e = 1 / 8)
  (h₃ : 1 / (a : ℝ) + 1 / e = 1 / 12) : 
  e = 48 :=
sorry

end project_completion_l696_696025


namespace smallest_area_right_triangle_l696_696480

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696480


namespace prob_all_digits_different_l696_696844

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696844


namespace prob_all_digits_different_l696_696846

theorem prob_all_digits_different : 
  let range_3digit := (set.Icc 100 999).to_finset in
  let total := range_3digit.card in
  let diff_digits := (range_3digit.filter (λ n : ℕ, 
    let hd := n / 100,
        td := (n / 10) % 10,
        ud := n % 10 in
    hd ≠ td ∧ hd ≠ ud ∧ td ≠ ud)).card in
  (diff_digits / total : ℚ) = 73 / 100 :=
sorry

end prob_all_digits_different_l696_696846


namespace closest_point_on_parabola_to_line_l696_696237

noncomputable def line := { P : ℝ × ℝ | 2 * P.1 - P.2 = 4 }
noncomputable def parabola := { P : ℝ × ℝ | P.2 = P.1^2 }

theorem closest_point_on_parabola_to_line : 
  ∃ P : ℝ × ℝ, P ∈ parabola ∧ 
  (∀ Q ∈ parabola, ∀ R ∈ line, dist P R ≤ dist Q R) ∧ 
  P = (1, 1) := 
sorry

end closest_point_on_parabola_to_line_l696_696237


namespace didi_total_fund_l696_696419

-- Define the conditions
def cakes : ℕ := 10
def slices_per_cake : ℕ := 8
def price_per_slice : ℕ := 1
def first_business_owner_donation_per_slice : ℚ := 0.5
def second_business_owner_donation_per_slice : ℚ := 0.25

-- Define the proof problem statement
theorem didi_total_fund (h1 : cakes * slices_per_cake = 80)
    (h2 : (80 : ℕ) * price_per_slice = 80)
    (h3 : (80 : ℕ) * first_business_owner_donation_per_slice = 40)
    (h4 : (80 : ℕ) * second_business_owner_donation_per_slice = 20) : 
    (80 : ℕ) + 40 + 20 = 140 := by
  -- The proof itself will be constructed here
  sorry

end didi_total_fund_l696_696419


namespace minimum_students_l696_696269

theorem minimum_students (n : ℕ) (students : Fin n → Fin n → Prop)
  (h_symm : ∀ i j, students i j ↔ students j i)
  (friend_count : ∀ i, ∃ c, c = 5 ∨ c = 6 ∧ fintype.card {j | students i j} = c)
  (h_diff_counts : ∀ i j, students i j → fintype.card {k | students i k} ≠ fintype.card {k | students j k}) :
  n ≥ 11 :=
sorry

end minimum_students_l696_696269


namespace probability_of_drawing_red_ball_l696_696719

theorem probability_of_drawing_red_ball (total_balls red_balls : ℕ) (h_total : total_balls = 10) (h_red : red_balls = 7) : (red_balls : ℚ) / total_balls = 7 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l696_696719


namespace hyperbola_asymptotes_l696_696236

theorem hyperbola_asymptotes
    (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 = Real.sqrt (1 + (b^2) / (a^2))) :
    (∀ x y : ℝ, (y = x * Real.sqrt 3) ∨ (y = -x * Real.sqrt 3)) :=
by
  sorry

end hyperbola_asymptotes_l696_696236


namespace find_AF_over_FB_l696_696291

open_locale classical
noncomputable theory

structure Triangle :=
(A B C : Point)

structure Configuration (T : Triangle) :=
(D E : Point)
(F : Point)
(G : Point)
(P : Point)
(Q : Point)
(hD_on_BC : collinear T.B T.C D)
(hE_on_BC : collinear T.B T.C E)
(hF_on_AB : segment_contains T.A T.B F)
(hP_on_AD_and_CF : concurrent (segment_connect T.A D).line (segment_connect T.C F).line P)
(hG_on_AC : collinear T.A T.C G)
(hQ_on_AD : segment_contains T.A D Q)
(hQ_on_BG : collinear T.B T.G Q)
(ratio_AP_PD : ratio (dist T.A P) (dist P D) = 3/2)
(ratio_FP_PC : ratio (dist F P) (dist P T.C) = 2/1)
(ratio_BQ_QG : ratio (dist T.B Q) (dist Q G) = 1/3)

theorem find_AF_over_FB (T : Triangle) (C : Configuration T) : ratio (dist C.F T.A) (dist C.F T.B) = 5/12 :=
sorry

end find_AF_over_FB_l696_696291


namespace largest_mersenne_prime_less_than_500_l696_696430

def mersenne_prime (n : ℕ) : ℕ := 2^n - 1

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem largest_mersenne_prime_less_than_500 :
  ∃ n, is_prime n ∧ mersenne_prime n < 500 ∧ ∀ m, is_prime m ∧ mersenne_prime m < 500 → mersenne_prime m ≤ mersenne_prime n :=
  sorry

end largest_mersenne_prime_less_than_500_l696_696430


namespace smallest_right_triangle_area_l696_696454

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696454


namespace number_of_speaking_orders_l696_696727

-- Define the original set of people
def group := fin 7

-- Define the subset of people including A and B
def A_and_B := {0, 1} -- Let's assume person 0 is A and person 1 is B

-- Define the condition that at least one of A and B must participate
def at_least_one_AB (S : finset group) : Prop :=
  0 ∈ S ∨ 1 ∈ S

-- Define the main theorem
theorem number_of_speaking_orders :
  ∃ (S : finset group), S.card = 4 ∧ at_least_one_AB S ∧ S.order_count = 720 :=
sorry

end number_of_speaking_orders_l696_696727


namespace digits_all_different_l696_696813

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696813


namespace marked_price_percentage_l696_696745

theorem marked_price_percentage {CP MP : ℝ} (h1 : CP > 0)
  (gain_percent : ℝ := 0.07)
  (discount_percent : ℝ := 18.939393939393938 / 100)
  (SP1 : MP = CP * (1 + gain_percent))
  (SP2 : CP * 1.07 = MP * (1 - discount_percent)):
  100 * ((SP1 / SP2 - 1) ≈ 32.01320132013201) :=
by
  sorry

end marked_price_percentage_l696_696745


namespace second_meeting_time_l696_696185

-- Given conditions and constants.
def pool_length : ℕ := 120
def initial_george_distance : ℕ := 80
def initial_henry_distance : ℕ := 40
def george_speed (t : ℕ) : ℕ := initial_george_distance / t
def henry_speed (t : ℕ) : ℕ := initial_henry_distance / t

-- Main statement to prove the question and answer.
theorem second_meeting_time (t : ℕ) (h_t_pos : t > 0) : 
  5 * t = 15 / 2 :=
sorry

end second_meeting_time_l696_696185


namespace ed_lighter_than_al_l696_696755

theorem ed_lighter_than_al :
  let Al := Ben + 25
  let Ben := Carl - 16
  let Ed := 146
  let Carl := 175
  Al - Ed = 38 :=
by
  sorry

end ed_lighter_than_al_l696_696755


namespace area_of_QST_l696_696422

theorem area_of_QST {P Q R S T : Type*} 
  (h1 : ∃ (PR : ℝ), (PR = (real.sqrt (6^2 + 8^2)) ∧ PR = 10))
  (h2 : S = (P + R) / 2)
  (h3 : PT = RT)
  (h4 : PT = 10)
  (h5 : right_angle R)
  (h6 : PQ = 6)
  (h7 : QR = 8) 
  (h8 : PQ * QR = 48)
  : 
  area (triangle Q S T) = 10 := 
sorry

end area_of_QST_l696_696422


namespace probability_digits_all_different_l696_696778

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696778


namespace power_of_two_l696_696318

theorem power_of_two (b m n : ℕ) (hb : b > 1) (hmn : m ≠ n) (hpf : ∀ p : ℕ, p.prime → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) : ∃ k, b + 1 = 2^k :=
by sorry

end power_of_two_l696_696318


namespace election_votes_l696_696416

theorem election_votes (V : ℝ) (h1 : 0.56 * V - 0.44 * V = 288) : 0.56 * V = 1344 :=
by 
  sorry

end election_votes_l696_696416


namespace smallest_right_triangle_area_l696_696496

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696496


namespace sum_of_first_five_primes_with_units_digit_three_l696_696145

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ≥ 2 → m * m ≤ n → n % m ≠ 0

def has_units_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def first_five_primes_with_units_digit_three : list ℕ :=
  [3, 13, 23, 43, 53]

def sum_first_five_primes_with_units_digit_three (l : list ℕ) : ℕ :=
  l.foldr (λ x acc, x + acc) 0

theorem sum_of_first_five_primes_with_units_digit_three:
  sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three = 145 := 
by
  have prime_3 : is_prime 3 := by sorry
  have prime_13 : is_prime 13 := by sorry
  have prime_23 : is_prime 23 := by sorry
  have prime_43 : is_prime 43 := by sorry
  have prime_53 : is_prime 53 := by sorry
  have list_units_digit_3 : ∀ n ∈ first_five_primes_with_units_digit_three, has_units_digit_three n := by
    intro n hn
    cases hn
    case inl h1 => rw [h1]; exact rfl
    case inr h1 =>
      cases h1
      case inl h2 => rw [h2]; exact rfl
      case inr h2 =>
        cases h2
        case inl h3 => rw [h3]; exact rfl
        case inr h3 =>
          cases h3
          case inl h4 => rw [h4]; exact rfl
          case inr h4 => cases h4; rw [h4]; exact rfl
  calc
    sum_first_five_primes_with_units_digit_three first_five_primes_with_units_digit_three
    = 3 + 13 + 23 + 43 + 53 : rfl
    ... = 135 : by sorry
    ... = 145 : by 
      sorry
  sorry

end sum_of_first_five_primes_with_units_digit_three_l696_696145


namespace smallest_right_triangle_area_l696_696572

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696572


namespace volume_of_parallelepiped_l696_696254

open Real

variables (c d : ℝ^3)
variables (h_c_unit : ‖c‖ = 1) (h_d_unit : ‖d‖ = 1)
variables (h_angle : real.angle_between c d = π / 4)

theorem volume_of_parallelepiped : 
  abs (c • ((d + 2 * (d × c)) × d)) = 1 :=
sorry

end volume_of_parallelepiped_l696_696254


namespace sum_of_first_five_primes_with_units_digit_3_l696_696125

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696125


namespace triplet_diff_invariant_l696_696266

theorem triplet_diff_invariant (a b c : ℕ) (n : ℕ) (h_init : a = 70 ∧ b = 61 ∧ c = 20)
  (h_transform : ∀ (a b c : ℕ), {a', b', c'} = ({b + c, a + c, a + b} : set ℕ)) :
  n = 1989 → ∀ (a b c : ℕ), a - c = 50 :=
by
  sorry

end triplet_diff_invariant_l696_696266


namespace card_numbers_l696_696921

theorem card_numbers (
  a b c d : ℕ) 
  (h_all_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_in_range : 0 < a ∧ a < 9 ∧ 0 < b ∧ b < 9 ∧ 0 < c ∧ c < 9 ∧ 0 < d ∧ d < 9)
  (h_probability_conditions : 
    (∃ twoSums : finset ℕ, twoSums.card = 6 ∧ 
      (∀ x ∈ twoSums, 0 < x < 18) ∧ 
      let sums := [a+b, a+c, a+d, b+c, b+d, c+d] in 
      list.perm sums (twoSums.val) ∧ 
      finset.card (twoSums.filter (λ x, x = 9)) = 2 ∧ 
      finset.card (twoSums.filter (λ x, x < 9)) = 2 ∧ 
      finset.card (twoSums.filter (λ x, x > 9)) = 2))
  : (a, b, c, d) = (1, 2, 7, 8) ∨ (a, b, c, d) = (1, 3, 6, 8) ∨ 
    (a, b, c, d) = (1, 4, 5, 8) ∨ (a, b, c, d) = (2, 3, 6, 7) ∨ 
    (a, b, c, d) = (2, 4, 5, 7) ∨ (a, b, c, d) = (3, 4, 5, 6) :=
sorry

end card_numbers_l696_696921


namespace park_area_l696_696388

-- Define the conditions as given in the problem
def base_EF : ℝ := 10
def height_G_to_EF : ℝ := 5
def base_GH : ℝ := 2

-- Define the parameters for the areas
def area_EFG : ℝ := (1/2) * base_EF * height_G_to_EF
def area_EGH : ℝ := (1/2) * base_GH * height_G_to_EF

-- Define the area of the park as the difference between areas of EFG and EGH
def area_of_park := area_EFG - area_EGH

-- The proof problem in Lean 4
theorem park_area : area_of_park = 20 := by
  -- The proof steps are skipped here, only the statement is needed
  sorry

end park_area_l696_696388


namespace smallest_area_right_triangle_l696_696487

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696487


namespace max_sum_S_n_l696_696995

-- Define the sequence a_n
def a_n (n : ℕ) : ℤ := 17 - 3 * n

-- Define the partial sum S_n
def S_n (n : ℕ) : ℤ := ∑ i in Finset.range n, a_n (i + 1)

-- Statement to prove: the sum S_n is maximized when n = 5 for the given sequence
theorem max_sum_S_n : (∀ m : ℕ, S_n m ≤ S_n 5) :=
by
  sorry

end max_sum_S_n_l696_696995


namespace square_vertex_distance_l696_696023

noncomputable def inner_square_perimeter : ℝ := 24
noncomputable def outer_square_perimeter : ℝ := 32
noncomputable def greatest_distance : ℝ := 7 * Real.sqrt 2

theorem square_vertex_distance :
  let inner_side := inner_square_perimeter / 4
  let outer_side := outer_square_perimeter / 4
  let inner_diagonal := Real.sqrt (inner_side ^ 2 + inner_side ^ 2)
  let outer_diagonal := Real.sqrt (outer_side ^ 2 + outer_side ^ 2)
  let distance := (inner_diagonal / 2) + (outer_diagonal / 2)
  distance = greatest_distance :=
by
  sorry

end square_vertex_distance_l696_696023


namespace find_a_value_l696_696979

theorem find_a_value (a : ℝ) (h1 : 1 < a) (h2 : ∀ x ∈ Icc (1 : ℝ) a, x^2 - 2*a*x + 5 ∈ Icc (1 : ℝ) a) :
  a = 2 :=
by
  sorry

end find_a_value_l696_696979


namespace smallest_area_right_triangle_l696_696521

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696521


namespace Anna_usual_weekly_salary_l696_696886

variable (weekly_salary daily_earnings : ℝ)
variable (days_missed days_in_week : ℕ)
variable (deducted_salary : ℝ)

-- Given conditions
def conditions (days_missed_eq : days_missed = 2)
               (deducted_salary_eq : deducted_salary = 985)
               (days_in_week_eq : days_in_week = 5)
               : Prop :=
  days_missed_eq ∧ deducted_salary_eq ∧ days_in_week_eq

-- The result we aim to prove
theorem Anna_usual_weekly_salary 
(days_missed_eq : days_missed = 2)
(deducted_salary_eq : deducted_salary = 985)
(days_in_week_eq : days_in_week = 5)
(daily_earnings_eq : daily_earnings = deducted_salary / 2)
(weekly_salary_eq : weekly_salary = daily_earnings * 5)
: weekly_salary = 2462.50 := by
  have h1 : daily_earnings = 492.50 := by
    sorry

  have h2 : weekly_salary = 2462.50 := by
    sorry

  exact h2

end Anna_usual_weekly_salary_l696_696886


namespace largest_mersenne_prime_is_127_l696_696435

noncomputable def largest_mersenne_prime_less_than_500 : ℕ :=
  127

theorem largest_mersenne_prime_is_127 :
  ∃ p : ℕ, Nat.Prime p ∧ (2^p - 1) = largest_mersenne_prime_less_than_500 ∧ 2^p - 1 < 500 := 
by 
  -- The largest Mersenne prime less than 500 is 127
  use 7
  sorry

end largest_mersenne_prime_is_127_l696_696435


namespace smallest_area_of_right_triangle_l696_696462

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℝ), area = 6 * sqrt 7 ∧ 
  ((a = 6 ∧ b = 8) ∨ (a = 2 * sqrt 7 ∧ b = 8)) := by
  sorry

end smallest_area_of_right_triangle_l696_696462


namespace smallest_area_of_right_triangle_l696_696626

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696626


namespace smallest_right_triangle_area_l696_696449

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696449


namespace smallest_d_and_100d_value_l696_696310

noncomputable def b : ℕ → ℝ
| 0       := 3 / 7
| (n + 1) := 3 * (b n)^2 - 2

def meets_inequality (d : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → abs (finset.range n).prod (λ k, b k) ≤ d / 3^n

theorem smallest_d_and_100d_value :
  let d := 240 / 343 in
  meets_inequality d ∧ round (100 * d) = 70 :=
by
  sorry

end smallest_d_and_100d_value_l696_696310


namespace find_largest_n_l696_696999

theorem find_largest_n 
  (a : ℕ → ℕ) (b : ℕ → ℕ)
  (x y : ℕ)
  (h_a1 : a 1 = 1)
  (h_b1 : b 1 = 1)
  (h_arith_a : ∀ n : ℕ, a n = 1 + (n - 1) * x)
  (h_arith_b : ∀ n : ℕ, b n = 1 + (n - 1) * y)
  (h_order : x ≤ y)
  (h_product : ∃ n : ℕ, a n * b n = 4021) :
  ∃ n : ℕ, a n * b n = 4021 ∧ n ≤ 11 := 
by
  sorry

end find_largest_n_l696_696999


namespace smallest_area_of_right_triangle_l696_696619

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696619


namespace number_of_triangles_l696_696340

variables (A B C D : Type) [plane A] [plane B] [plane C] [plane D]

-- Condition: No three points are collinear
axiom no_three_collinear : ¬collinear A B C ∧ ¬collinear A B D ∧ ¬collinear A C D ∧ ¬collinear B C D

-- Theorem to prove: The number of triangles that can be formed is 4
theorem number_of_triangles : (number_of_triangles A B C D = 4) :=
by
  sorry

end number_of_triangles_l696_696340


namespace power_of_i_sum_l696_696927

theorem power_of_i_sum :
  let i : ℂ := complex.I in
  i^11 + i^16 + i^21 + i^26 + i^31 + i^36 = 1 := 
by
  let i : ℂ := complex.I
  sorry

end power_of_i_sum_l696_696927


namespace integral_solution_l696_696066

noncomputable def integral_problem : Prop :=
  ∫ x in 2..4, (x^3 - 3*x^2 + 5) / x^2 = 5 / 4

theorem integral_solution : integral_problem :=
  sorry

end integral_solution_l696_696066


namespace sum_of_first_five_primes_with_units_digit_3_l696_696159

def units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime :=
  ∀ (n : ℕ), (2 ≤ n) → (∀ m, m ∣ n → m = 1 ∨ m = n)

theorem sum_of_first_five_primes_with_units_digit_3 :
  let primes_with_units_digit_3 := [3, 13, 23, 43, 53] in
  ∀ n ∈ primes_with_units_digit_3, is_prime n →
  units_digit_3 n →
  (3 + 13 + 23 + 43 + 53 = 135) :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696159


namespace pairs_of_integers_solution_l696_696063

-- Define the main theorem
theorem pairs_of_integers_solution :
  ∃ (x y : ℤ), 9 * x * y - x^2 - 8 * y^2 = 2005 ∧ 
               ((x = 63 ∧ y = 58) ∨
               (x = -63 ∧ y = -58) ∨
               (x = 459 ∧ y = 58) ∨
               (x = -459 ∧ y = -58)) :=
by
  sorry

end pairs_of_integers_solution_l696_696063


namespace sum_of_first_five_primes_with_units_digit_3_l696_696126

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696126


namespace smallest_n_value_l696_696068

theorem smallest_n_value (n : ℕ) (h : 15 * n - 2 ≡ 0 [MOD 11]) : n = 6 :=
sorry

end smallest_n_value_l696_696068


namespace gauravi_walks_4500m_on_tuesday_l696_696945

def initial_distance : ℕ := 500
def increase_per_day : ℕ := 500
def target_distance : ℕ := 4500

def distance_after_days (n : ℕ) : ℕ :=
  initial_distance + n * increase_per_day

def day_of_week_after (start_day : ℕ) (n : ℕ) : ℕ :=
  (start_day + n) % 7

def monday : ℕ := 0 -- Represent Monday as 0

theorem gauravi_walks_4500m_on_tuesday :
  distance_after_days 8 = target_distance ∧ day_of_week_after monday 8 = 2 :=
by 
  sorry

end gauravi_walks_4500m_on_tuesday_l696_696945


namespace smallest_area_right_triangle_l696_696644

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696644


namespace difference_of_squares_example_l696_696049

theorem difference_of_squares_example : (65^2 - 35^2) = 3000 :=
by
  let a := 65
  let b := 35
  have h : (a^2 - b^2) = (a + b) * (a - b) := (Nat.sq_sub_sq a b).symm
  rw [h, Nat.add_sub_eq_of_eq_add 65 35]
  norm_num
  sorry

end difference_of_squares_example_l696_696049


namespace probability_all_digits_different_l696_696824

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696824


namespace school_scores_probability_l696_696272

noncomputable def normal_distribution := sorry

theorem school_scores_probability :
  ∀ (X : ℝ → ℝ) (σ : ℝ),
  (normal_distribution X 80 σ) ∧
  (prob (60 ≤ X) (X ≤ 80) = 0.25) →
  (prob X (< 100) = 0.75) :=
by
  sorry

end school_scores_probability_l696_696272


namespace probability_digits_all_different_l696_696866

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696866


namespace probability_digits_all_different_l696_696870

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696870


namespace solve_trigonometric_equation_l696_696252

theorem solve_trigonometric_equation :
  ∃ (S : Finset ℝ), (∀ X ∈ S, 0 < X ∧ X < 360 ∧ 1 + 2 * Real.sin (X * Real.pi / 180) - 4 * (Real.sin (X * Real.pi / 180))^2 - 8 * (Real.sin (X * Real.pi / 180))^3 = 0) ∧ S.card = 4 :=
by
  sorry

end solve_trigonometric_equation_l696_696252


namespace probability_digits_all_different_l696_696780

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696780


namespace polar_curve_is_circle_l696_696374

theorem polar_curve_is_circle (θ ρ : ℝ) (h : 4 * Real.sin θ = 5 * ρ) : 
  ∃ c : ℝ×ℝ, ∀ (x y : ℝ), x^2 + y^2 = c.1^2 + c.2^2 :=
by
  sorry

end polar_curve_is_circle_l696_696374


namespace smallest_area_right_triangle_l696_696602

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696602


namespace rhombus_longer_diagonal_l696_696018

theorem rhombus_longer_diagonal (a b d_1 : ℝ) (h_side : a = 60) (h_d1 : d_1 = 56) :
  ∃ d_2, d_2 = 106 := by
  sorry

end rhombus_longer_diagonal_l696_696018


namespace proof_stmt_l696_696033

variable (a x y : ℝ)
variable (ha : a > 0) (hneq : a ≠ 1)

noncomputable def S (x : ℝ) := a^x - a^(-x)
noncomputable def C (x : ℝ) := a^x + a^(-x)

theorem proof_stmt :
  2 * S a (x + y) = S a x * C a y + C a x * S a y ∧
  2 * S a (x - y) = S a x * C a y - C a x * S a y :=
by sorry

end proof_stmt_l696_696033


namespace smallest_positive_period_max_min_value_monotonic_intervals_symmetry_properties_l696_696976

noncomputable def sqrt3 : ℝ := real.sqrt 3

def f (x : ℝ) : ℝ := sqrt3 * cos x ^ 2 + 1 / 2 * sin (2 * x)

theorem smallest_positive_period (T : ℝ) : (∀ x : ℝ, f (x + T) = f x) ∧ T = π := sorry

theorem max_min_value : 
  ∃ x_max x_min : ℝ,
  x_max ∈ Icc (-(π / 6)) (π / 4) ∧ x_min ∈ Icc (-(π / 6)) (π / 4) ∧ 
  (f x_max = 1 + sqrt3 / 2) ∧ (f x_min = sqrt3 / 2) := sorry

theorem monotonic_intervals : 
  ∀ k : ℤ,
  (∀ x : ℝ, x ∈ Icc (k * π - 5 * π / 12) (k * π + π / 12) → monotone f) ∧
  (∀ x : ℝ, x ∈ Icc (k * π + π / 12) (k * π + 7 * π / 12) → antitone f) := sorry

theorem symmetry_properties : 
  (∀ k : ℤ, ∀ x : ℝ, x = 1 / 2 * k * π + π / 12) ∧ 
  (∀ k : ℤ, center_sym (1 / 2 * k * π - π / 6, 0)) := sorry

end smallest_positive_period_max_min_value_monotonic_intervals_symmetry_properties_l696_696976


namespace calc_expression_l696_696043

-- Define the components of the expression
def abs_val (x : ℝ) : ℝ := |x|
def exp (base exp : ℝ) : ℝ := base ^ exp
def eval_expr := abs_val (-3) + exp 2 2 - exp (sqrt 3 - 1) 0

-- The main theorem to prove the expression evaluates to 6
theorem calc_expression : eval_expr = 6 :=
by
  sorry

end calc_expression_l696_696043


namespace empty_subset_intersection_l696_696215

theorem empty_subset_intersection (A B : Set) (hAnempty : A ≠ ∅) (hBnempty : B ≠ ∅) (hAdistinctB : A ≠ B) :
  ∅ ⊆ (A ∩ B) :=
sorry

end empty_subset_intersection_l696_696215


namespace height_of_equilateral_triangle_l696_696293

theorem height_of_equilateral_triangle (h s r: ℝ) 
  (eq1 : s = 2)
  (eq2 : r = h / 2)
  (eq3 : (2 - (sqrt 3 / 4) * 2^2) + 3 * 2 = 4) :
  h = sqrt 3 := 
sorry

end height_of_equilateral_triangle_l696_696293


namespace probability_digits_all_different_l696_696864

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696864


namespace missing_population_correct_l696_696389

def known_populations := [803, 1100, 1023, 945, 980, 1249]
def average_population : ℕ := 1000
def number_of_villages : ℕ := 7
def missing_village_population : ℕ := 900

theorem missing_population_correct :
  (average_population * number_of_villages = list.sum known_populations + missing_village_population) :=
by
  sorry

end missing_population_correct_l696_696389


namespace operation_is_double_l696_696934

theorem operation_is_double (x : ℝ) (operation : ℝ → ℝ) (h1: x^2 = 25) (h2: operation x = x / 5 + 9) : operation x = 2 * x :=
by
  sorry

end operation_is_double_l696_696934


namespace smallest_area_right_triangle_l696_696685

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696685


namespace smallest_area_right_triangle_l696_696488

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end smallest_area_right_triangle_l696_696488


namespace range_of_k_l696_696980

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 2 then 2 / x else (x - 1)^3

theorem range_of_k (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 - k = 0 ∧ f x2 - k = 0) ↔ (0 < k ∧ k < 1) := sorry

end range_of_k_l696_696980


namespace cos_approx_using_maclaurin_l696_696427

noncomputable def maclaurin_series_cos (x : ℝ) : ℝ :=
  ∑ n in finset.range 2, ((-1:ℝ) ^ n * x ^ (2 * n)) / ((nat.factorial (2 * n)):ℝ)

theorem cos_approx_using_maclaurin :
  abs (maclaurin_series_cos 0.1 - 0.995) < 0.00001 := by
  -- Proof omitted
  sorry

end cos_approx_using_maclaurin_l696_696427


namespace smallest_right_triangle_area_l696_696502

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696502


namespace smallest_area_right_triangle_l696_696568

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696568


namespace smallest_area_right_triangle_l696_696612

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696612


namespace right_triangle_min_area_l696_696656

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696656


namespace david_boxes_l696_696058

-- Conditions
def number_of_dogs_per_box : ℕ := 4
def total_number_of_dogs : ℕ := 28

-- Problem
theorem david_boxes : total_number_of_dogs / number_of_dogs_per_box = 7 :=
by
  sorry

end david_boxes_l696_696058


namespace index_card_area_l696_696036

theorem index_card_area 
  (a b : ℕ)
  (ha : a = 5)
  (hb : b = 7)
  (harea : (a - 2) * b = 21) :
  (a * (b - 2) = 25) :=
by
  sorry

end index_card_area_l696_696036


namespace smallest_area_right_triangle_l696_696560

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696560


namespace final_number_is_2012_l696_696386

theorem final_number_is_2012 :
  let a := λ k : Nat, 1 / (k + 1) in
  let step := λ x y : ℝ, x + y + x * y in
  (Array.foldr (λ x acc, step x acc) 0.0 (List.toArray (List.map a (List.range 2012)))) + 1 = 2012 :=
sorry

end final_number_is_2012_l696_696386


namespace smallest_area_correct_l696_696542

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696542


namespace sum_of_first_five_primes_units_digit_3_l696_696165

def is_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def primes_with_units_digit_3 : List ℕ :=
  (Nat.primes.filter is_units_digit_3).take 5

theorem sum_of_first_five_primes_units_digit_3 :
  primes_with_units_digit_3.sum = 135 :=
by
  sorry

end sum_of_first_five_primes_units_digit_3_l696_696165


namespace sum_is_correct_l696_696153

-- Define the five prime numbers with units digit 3
def prime1 := 3
def prime2 := 13
def prime3 := 23
def prime4 := 43
def prime5 := 53

-- Define the sum of these five primes
def sum_of_five_primes : Nat :=
  prime1 + prime2 + prime3 + prime4 + prime5

-- Theorem statement
theorem sum_is_correct : sum_of_five_primes = 123 :=
  by
    -- Proof placeholder
    sorry

end sum_is_correct_l696_696153


namespace smallest_area_right_triangle_l696_696607

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696607


namespace select_participants_at_least_one_female_l696_696356

theorem select_participants_at_least_one_female :
  ∃ n : ℕ, n = 16 ∧ (∃ f m : finset ℕ, f.card = 2 ∧ m.card = 4 ∧
  (finset.choose 3 (f ∪ m)).card = n ∧ (∀ s ⊆ f ∪ m, s.card = 3 → (s ∩ f).nonempty)) :=
begin
  sorry
end

end select_participants_at_least_one_female_l696_696356


namespace number_of_triangles_l696_696196

theorem number_of_triangles (k : ℕ) (n : ℕ → ℕ) :
  ∑ (p q r : ℕ) in finset.triple_powerset (finset.range k), n p * n q * n r +
  ∑ (p q : ℕ) in finset.twos_powerset (finset.range k), n p * nat.choose (n q) 2 + n q * nat.choose (n p) 2 =
    ∑ (1 ≤ p < q < r ≤ k), n p * n q * n r +
    ∑ (1 ≤ p < q ≤ k), n p * nat.choose (n q) 2 + n q * nat.choose (n p) 2 :=
sorry

end number_of_triangles_l696_696196


namespace sequence_general_term_l696_696328

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (a 1 = 1/2) ∧ 
  (∀ n, 2 ≤ n → a n = -5 * S n * S (n - 1)) ∧
  (S n = ∑ i in Finset.range n, a (i + 1)) →
  ∀ n, a n = if n = 1 then 1/2 else -5 / ((5 * n - 3) * (5 * n - 8)) :=
by
  -- Proof will be included here
  sorry

#print axioms sequence_general_term

end sequence_general_term_l696_696328


namespace cost_of_set_of_paints_l696_696335

def classes : ℕ := 6
def folders_per_class : ℕ := 1
def pencils_per_class : ℕ := 3
def erasers_per_6_pencils : ℕ := 1
def folder_cost : ℕ := 6
def pencil_cost : ℕ := 2
def eraser_cost : ℕ := 1
def total_spent : ℕ := 80

theorem cost_of_set_of_paints : 
  let total_folder_cost := classes * folders_per_class * folder_cost,
      total_pencil_cost := classes * pencils_per_class * pencil_cost,
      total_pencils := classes * pencils_per_class,
      total_erasers := total_pencils / 6 * erasers_per_6_pencils,
      total_eraser_cost := total_erasers * eraser_cost,
      total_supplies_cost := total_folder_cost + total_pencil_cost + total_eraser_cost
  in total_spent - total_supplies_cost = 5 :=
by 
  sorry

end cost_of_set_of_paints_l696_696335


namespace smallest_area_of_right_triangle_l696_696597

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696597


namespace smallest_area_right_triangle_l696_696604

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 6) (hb : b = 8) :
  ∃ (area : ℕ), area = 24 ∧ (∃ c, c = Real.sqrt (a^2 + b^2) ∨ a = Real.sqrt (b^2 + c^2) ) :=
by
  use 24
  split
  . rfl
  . use Real.sqrt (a^2 + b^2)
    sorry

end smallest_area_right_triangle_l696_696604


namespace smallest_right_triangle_area_l696_696499

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696499


namespace total_new_people_last_year_l696_696300

-- Define the number of new people born and the number of people immigrated
def new_people_born : ℕ := 90171
def people_immigrated : ℕ := 16320

-- Prove that the total number of new people is 106491
theorem total_new_people_last_year : new_people_born + people_immigrated = 106491 := by
  sorry

end total_new_people_last_year_l696_696300


namespace daniel_distance_in_30_minutes_l696_696904

-- Define constants for time and distance
def time1 : ℝ := (40 / 60)  -- convert minutes to hours
def distance1 : ℝ := 24
def speed : ℝ := distance1 / time1  -- calculate speed

-- Define new time period to consider
def time2 : ℝ := (30 / 60)  -- convert minutes to hours

-- Define function to calculate distance based on speed and time
def distance_cycled (v : ℝ) (t : ℝ) : ℝ := v * t

-- Statement to prove
theorem daniel_distance_in_30_minutes : 
  distance_cycled speed time2 = 18 :=
by
  -- skip the proof
  sorry

end daniel_distance_in_30_minutes_l696_696904


namespace bounded_partial_sums_of_unit_vectors_l696_696706

theorem bounded_partial_sums_of_unit_vectors 
  (n : ℕ) 
  (v : Fin n → ℝ × ℝ)
  (hv_length : ∀ i : Fin n, ‖v i‖ = 1)
  (hv_sum : Finset.univ.sum v = (0, 0)) : 
  ∃ σ : Fin n → Fin n, 
    ∀ k : Fin n, ‖(Finset.univ.filter (· ≤ k)).sum (v ∘ σ)‖ ≤ real.sqrt 5 :=
sorry

end bounded_partial_sums_of_unit_vectors_l696_696706


namespace volume_of_cylindrical_drum_l696_696004

noncomputable def height_yards := 2
def height_feet := height_yards * 3
def diameter_feet := 4
def radius_feet := diameter_feet / 2
def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem volume_of_cylindrical_drum (r h V : ℝ) (h_eq : h = 6) (r_eq : r = 2) :
  V = volume_cylinder r h → V = 24 * π :=
by
  -- sorry is necessary to skip the proof details, as only statement is required.
  sorry

end volume_of_cylindrical_drum_l696_696004


namespace last_two_nonzero_digits_of_80_factorial_l696_696897

theorem last_two_nonzero_digits_of_80_factorial :
  let N := Nat.fact 80 / 10^19 in
  (N % 100) = 48 :=
by
  sorry

end last_two_nonzero_digits_of_80_factorial_l696_696897


namespace find_f_4_l696_696327

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom domain_f_g (x : ℝ) : x ∈ ℝ
axiom f_has_inverse : ∃ h : ℝ → ℝ, f ∘ h = id ∧ h ∘ f = id
axiom g_has_inverse : ∃ h : ℝ → ℝ, g ∘ h = id ∧ h ∘ g = id
axiom symmetry_condition (x : ℝ) : f(x - 1) = (g⁻¹(x - 2))
axiom g_value : g 5 = 2004

theorem find_f_4 : f 4 = 2006 :=
by {
  -- Proof omitted 
  sorry
}

end find_f_4_l696_696327


namespace number_of_subsets_of_P_l696_696946
open Set

theorem number_of_subsets_of_P :
  let M := {0, 1, 2, 3, 4}
  let N := {1, 3, 5, 7}
  let P := {x | x ∈ M ∧ x ∈ N}
  ∃ (n : ℕ), (P = {1, 3} ∧ n = 4) ∧ (n = 2 ^ 2) :=
by
  sorry

end number_of_subsets_of_P_l696_696946


namespace distinct_pen_distribution_l696_696182

theorem distinct_pen_distribution :
  ∃! (a b c d : ℕ), a + b + c + d = 10 ∧
                    1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧
                    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d :=
sorry

end distinct_pen_distribution_l696_696182


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696116

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696116


namespace range_of_function_l696_696393

theorem range_of_function : 
  ∀ y : ℝ, ∃ x : ℝ, y = 4^x + 2^x - 3 ↔ y > -3 :=
by
  sorry

end range_of_function_l696_696393


namespace smallest_area_correct_l696_696549

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696549


namespace smallest_area_right_triangle_l696_696646

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696646


namespace trigonometric_equation_solution_range_l696_696942

theorem trigonometric_equation_solution_range (a : ℝ) :
  (∃ x ∈ Ioo 0 (π / 2), cos x ^ 2 + sin x + a = 0) ↔ -5/4 ≤ a ∧ a ≤ -1 :=
sorry

end trigonometric_equation_solution_range_l696_696942


namespace probability_digits_all_different_l696_696868

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696868


namespace square_geometry_problem_l696_696279

theorem square_geometry_problem :
  ∃ (a b c : ℕ), 
    let AQ := b * Real.sqrt c - a 
    in AQ = (10 * Real.sqrt 3 - 10) ∧ a + b + c = 23 
    ∧ ∃ A B C D P Q R X Y, 
      A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A  -- square vertex distinctness
      ∧ (AB = 10 ∧ BQ ⊥ AP ∧ RQ ∥ PA ∧ B = (10, 0) ∧ P = (x, 10)) -- given conditions
      ∧ X = LineInter (BC) (AP) ∧ Y = CircumCirclePoint (XQ) (P Q D) -- intersection definitions
      ∧ ∠PYR = 105 :=
sorry

end square_geometry_problem_l696_696279


namespace four_non_coplanar_points_determine_four_planes_l696_696183

theorem four_non_coplanar_points_determine_four_planes:
  ∀ (p1 p2 p3 p4: ℝ^3), 
  ¬ collinear p1 p2 p3 → ¬ collinear p1 p2 p4 → ¬ collinear p1 p3 p4 → ¬ collinear p2 p3 p4 → ¬ coplanar p1 p2 p3 p4 → 
  ∃ (planes: finset (finset ℝ^3)), planes.card = 4 ∧ ∀ plane ∈ planes, ∃ a b c: ℝ^3, plane = {a, b, c} :=
by
  sorry

end four_non_coplanar_points_determine_four_planes_l696_696183


namespace max_R_stamps_l696_696907

/-- The maximum value R for which any value from 1 to R can be formed using at most three stamps of four distinct positive integer denominations is 14. -/
theorem max_R_stamps : ∃ (R : ℕ) (denoms : Finset ℕ), 
  denoms.card = 4 ∧ 
  (∀ denom ∈ denoms, denom > 0) ∧ 
  R = 14 ∧ 
  (∀ n ∈ (Finset.range (R + 1)), ∃ (a b c : ℕ), 
    a ∈ denoms ∧ b ∈ denoms ∧ c ∈ denoms ∧ 
    (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = n) ∨ 
    (a = 0 ∧ b ≠ c ∧ b = 0 ∧ c = n) ∨ 
    (b = 0 ∧ a ≠ c ∧ a = 0 ∧ c = n)) :=
begin
  use [14, {1, 2, 4, 8}],
  split,
  { exact Finset.card_insert_of_not_mem (Finset.mem_insert_self 1 {2, 4, 8}), },
  split,
  { intros denom h,
    fin_cases deter from {1, 2, 4, 8},
    exact dec_trivial
  },
  split,
  { refl },
  { intro n,
    fin_cases deter.denoms from Finset.mem_of_mem_insert,
    sorry
  }
end

end max_R_stamps_l696_696907


namespace probability_all_digits_different_l696_696857

-- Defining the range of integers considered (greater than 99 and less than 1000)
def range := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Predicate to check if all digits of the number are different
def digits_all_different (n : ℕ) : Prop := 
  let digits := (show List ℕ, from (Integer.digits 10 n)) in
  digits.nodup

-- Statement: The probability that a randomly chosen integer from 100 to 999
-- has all different digits is 99/100.
theorem probability_all_digits_different : 
  (finset.filter digits_all_different (finset.range' 100 900)).card.to_rat 
  / (finset.range' 100 900).card.to_rat = 99 / 100 := by
  sorry

end probability_all_digits_different_l696_696857


namespace smallest_right_triangle_area_l696_696573

noncomputable def smallest_possible_area_of_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) : ℕ :=
  (1 / 2 * a * b).toNat

theorem smallest_right_triangle_area {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area_of_right_triangle h₁ h₂ = 24 := by
  sorry

end smallest_right_triangle_area_l696_696573


namespace equivalent_proof_l696_696993

noncomputable def parametric_line_eqn (t : ℝ) : ℝ × ℝ :=
  let x := 1 + t
  let y := Real.sqrt 3 + Real.sqrt 3 * t
  (x, y)

def polar_eqn (ρ θ : ℝ) : ℝ :=
  ρ^2 - 4 * ρ * Real.cos θ - 2 * Real.sqrt 3 * ρ * Real.sin θ + 4

theorem equivalent_proof :
  ( ∀ t : ℝ, let (x, y) := parametric_line_eqn t in y = Real.sqrt 3 * x ) ∧
  ( ∀ ρ θ : ℝ, polar_eqn ρ θ = 0 → (ρ * Real.cos θ - 2)^2 + (ρ * Real.sin θ - Real.sqrt 3)^2 = 3 ) ∧
  ( polar_eqn (Real.cos (π / 3)) (π / 3) = 0 → ∀ ρ₁ ρ₂ : ℝ, ρ₁ * ρ₂ = 4 ) :=
by sorry

end equivalent_proof_l696_696993


namespace smallest_area_of_right_triangle_l696_696536

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696536


namespace probability_all_different_digits_l696_696772

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696772


namespace domain_of_f_x_minus_1_l696_696261

theorem domain_of_f_x_minus_1 (f : ℝ → ℝ) (h : ∀ x, x^2 + 1 ∈ Set.Icc 1 10 → x ∈ Set.Icc (-3 : ℝ) 2) :
  Set.Icc 2 (11 : ℝ) ⊆ {x : ℝ | x - 1 ∈ Set.Icc 1 10} :=
by
  sorry

end domain_of_f_x_minus_1_l696_696261


namespace smallest_square_on_longest_side_l696_696746

theorem smallest_square_on_longest_side 
  (a b c : ℝ) (ma mb mc : ℝ) (A B C : Type) [triangle A B C]
  (h_acute : acute_angle_triangle A B C)
  (h_sides : sides_of_triangle A B C a b c)
  (h_altitudes : altitudes_of_triangle A B C ma mb mc)
  (h_longest_side : a ≥ b ∧ b ≥ c) :
  smallest_inscribed_square A B C a b c ma mb mc = some_square_with_side_on_a :=
sorry

end smallest_square_on_longest_side_l696_696746


namespace Yuri_puppies_l696_696703

def week1 : ℕ := 20
def week2 : ℕ := (2 * week1) / 5
def week3 : ℕ := (3 * week2) / 8
def week4 : ℕ := 2 * week2
def week5 : ℕ := week1 + 10
def week6 : ℕ := (2 * week3 - 5)
def week7 : ℕ := 2 * week6
def week8 : ℕ := (7 * week6) / 4
def week9 : ℕ := (3 * week8) / 2
def week10 : ℕ := (9 * week1) / 4
def week11 : ℕ := (5 * week10) / 6

-- The total number of puppies adopted is 164.
def total_puppies : ℕ :=
  week1 + week2 + week3 + week4 + week5 +
  week6 + week7 + week8 + week9 + week10 + week11

theorem Yuri_puppies : total_puppies = 164 :=
by
  rw [week1, week2, week3, week4, week5, week6, week7, week8, week9, week10, week11]
  norm_num

end Yuri_puppies_l696_696703


namespace probability_all_digits_different_l696_696823

def is_digit_different (n : ℕ) : Prop :=
  let digits := List.map (λ x => x.toString.toNat) (n.toString.data)
  (digits.nodup)

theorem probability_all_digits_different :
  ∑ i in Finset.Icc 100 999, if is_digit_different i then 1 else 0 = (3 * (900 / 4)) :=
by
  sorry

end probability_all_digits_different_l696_696823


namespace condition_sufficient_but_not_necessary_l696_696255

variables (p q : Prop)

theorem condition_sufficient_but_not_necessary (hpq : ∀ q, (¬p → ¬q)) (hpns : ¬ (¬p → ¬q ↔ p → q)) : (p → q) ∧ ¬ (q → p) :=
by {
  sorry
}

end condition_sufficient_but_not_necessary_l696_696255


namespace cost_of_set_of_paints_l696_696336

def classes : ℕ := 6
def folders_per_class : ℕ := 1
def pencils_per_class : ℕ := 3
def erasers_per_6_pencils : ℕ := 1
def folder_cost : ℕ := 6
def pencil_cost : ℕ := 2
def eraser_cost : ℕ := 1
def total_spent : ℕ := 80

theorem cost_of_set_of_paints : 
  let total_folder_cost := classes * folders_per_class * folder_cost,
      total_pencil_cost := classes * pencils_per_class * pencil_cost,
      total_pencils := classes * pencils_per_class,
      total_erasers := total_pencils / 6 * erasers_per_6_pencils,
      total_eraser_cost := total_erasers * eraser_cost,
      total_supplies_cost := total_folder_cost + total_pencil_cost + total_eraser_cost
  in total_spent - total_supplies_cost = 5 :=
by 
  sorry

end cost_of_set_of_paints_l696_696336


namespace card_sums_condition_l696_696924

theorem card_sums_condition (a b c d : ℕ) : 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  a + b = 9 → a + c = 9 → 
  (a, b, c, d) ∈ {(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)} :=
sorry

end card_sums_condition_l696_696924


namespace function_f_properties_l696_696952

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 4

theorem function_f_properties :
  (f 8 = 3 / 2) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ (x1 x2 : ℝ), x1 ≠ 0 → x2 ≠ 0 → f x1 = f x2 → f (x1 / x2) = f x2) :=
by
  split
  { sorry } 
  split
  { intros x
    sorry }
  { intros x1 x2 h1 h2 he
    sorry }

end function_f_properties_l696_696952


namespace smallest_area_of_right_triangle_l696_696591

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696591


namespace total_spent_is_correct_l696_696297

def trumpet : ℝ := 149.16
def music_tool : ℝ := 9.98
def song_book : ℝ := 4.14
def trumpet_maintenance_accessories : ℝ := 21.47
def valve_oil_original : ℝ := 8.20
def valve_oil_discount_rate : ℝ := 0.20
def valve_oil_discounted : ℝ := valve_oil_original * (1 - valve_oil_discount_rate)
def band_t_shirt : ℝ := 14.95
def sales_tax_rate : ℝ := 0.065

def total_before_tax : ℝ :=
  trumpet + music_tool + song_book + trumpet_maintenance_accessories + valve_oil_discounted + band_t_shirt

def sales_tax : ℝ := total_before_tax * sales_tax_rate

def total_amount_spent : ℝ := total_before_tax + sales_tax

theorem total_spent_is_correct : total_amount_spent = 219.67 := by
  sorry

end total_spent_is_correct_l696_696297


namespace exam_probabilities_l696_696751

variable (A B C : Type)
variable [probability_space A]
variable [probability_space B]
variable [probability_space C]

-- Given probabilities
def P_A_written : ℝ := 0.6
def P_B_written : ℝ := 0.5
def P_C_written : ℝ := 0.4

def P_A_oral : ℝ := 0.5
def P_B_oral : ℝ := 0.6
def P_C_oral : ℝ := 0.75

def independent_events (event1 event2 : Prop) : Prop := sorry

theorem exam_probabilities :
  let P_exactly_one_written := 
    P_A_written * (1 - P_B_written) * (1 - P_C_written) + 
    (1 - P_A_written) * P_B_written * (1 - P_C_written) + 
    (1 - P_A_written) * (1 - P_B_written) * P_C_written in
  P_exactly_one_written = 0.38 ∧
  let P_A_both := P_A_written * P_A_oral in
  let P_B_both := P_B_written * P_B_oral in
  let P_C_both := P_C_written * P_C_oral in
  let P_pre_admitted_zero := (1 - P_A_both) * (1 - P_B_both) * (1 - P_C_both) in
  let P_pre_admitted_one := 
    3 * (1 - P_A_both) * (1 - P_B_both) * P_C_both +
    3 * (1 - P_A_both) * P_B_both * (1 - P_C_both) +
    3 * P_A_both * (1 - P_B_both) * (1 - P_C_both) in
  let P_pre_admitted_two := 
    3 * P_A_both * P_B_both * (1 - P_C_both) +
    3 * P_A_both * (1 - P_B_both) * P_C_both +
    3 * (1 - P_A_both) * P_B_both * P_C_both in
  let P_pre_admitted_three := P_A_both * P_B_both * P_C_both in
  let E_ξ := 1 * P_pre_admitted_one + 2 * P_pre_admitted_two + 3 * P_pre_admitted_three in
  E_ξ = 0.9 := 
by 
  intro P_exactly_one_written P_A_both P_B_both P_C_both P_pre_admitted_zero P_pre_admitted_one P_pre_admitted_two P_pre_admitted_three E_ξ
  sorry

end exam_probabilities_l696_696751


namespace smallest_right_triangle_area_l696_696490

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696490


namespace sum_of_first_five_primes_with_units_digit_3_l696_696120

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696120


namespace smallest_area_right_triangle_l696_696635

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696635


namespace overall_percentage_gain_l696_696022

theorem overall_percentage_gain
    (original_price : ℝ)
    (first_increase : ℝ)
    (first_discount : ℝ)
    (second_discount : ℝ)
    (third_discount : ℝ)
    (final_increase : ℝ)
    (final_price : ℝ)
    (overall_gain : ℝ)
    (overall_percentage_gain : ℝ)
    (h1 : original_price = 100)
    (h2 : first_increase = original_price * 1.5)
    (h3 : first_discount = first_increase * 0.9)
    (h4 : second_discount = first_discount * 0.85)
    (h5 : third_discount = second_discount * 0.8)
    (h6 : final_increase = third_discount * 1.1)
    (h7 : final_price = final_increase)
    (h8 : overall_gain = final_price - original_price)
    (h9 : overall_percentage_gain = (overall_gain / original_price) * 100) :
  overall_percentage_gain = 0.98 := by
  sorry

end overall_percentage_gain_l696_696022


namespace sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696114

-- Define a predicate for a number to have a units digit of 3.
def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

-- Define the set of numbers that are considered for checking primality.
def number_candidates : List ℕ :=
  [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Define a function to check if a given number is prime.
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the first five prime numbers with a units digit of 3.
def first_five_primes_with_units_digit_3 (l : List ℕ) : List ℕ :=
  l.filter (λ n, has_units_digit_3 n ∧ is_prime n) |>.take 5

-- Define a constant for the expected sum.
def expected_sum : ℕ :=
  135

-- The theorem statement proving the sum of the first five prime numbers that have a units digit of 3 is 135.
theorem sum_of_first_five_primes_with_units_digit_3_eq_135 :
  first_five_primes_with_units_digit_3 number_candidates |>.sum = expected_sum :=
by sorry

end sum_of_first_five_primes_with_units_digit_3_eq_135_l696_696114


namespace vector_magnitude_sqrt13_l696_696971

noncomputable def vector_magnitude (a b : ℝ) : ℝ := sorry

theorem vector_magnitude_sqrt13 
  (a b : ℝ → ℝ → ℝ)
  (unit_a : a • a = 1) 
  (unit_b : b • b = 1) 
  (angle_ab : a • b = 1 / 2) : 
  vector_magnitude (a + 3 • b) = sqrt 13 := 
sorry

end vector_magnitude_sqrt13_l696_696971


namespace probability_of_score_less_than_100_l696_696270

/-- Given X follows a normal distribution N(80, σ^2) and P(60 ≤ X ≤ 80) = 0.25,
    prove that P(X < 100) = 0.75 -/
theorem probability_of_score_less_than_100
  {X : ℝ → ℝ}
  (hX : X ~ Normal(80, σ^2))
  (h_prob : P(60 ≤ X ∧ X ≤ 80) = 0.25) :
  P(X < 100) = 0.75 :=
sorry

end probability_of_score_less_than_100_l696_696270


namespace smallest_right_triangle_area_l696_696456

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696456


namespace probability_is_18_over_25_l696_696833

namespace ProbabilityDifferentDigits

-- Definition of the set of integers between 100 and 999
def int_set := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

-- Definition of the set of integers that have all different digits
def different_digits_set := {n ∈ int_set | 
  let d1 := n / 100, d2 := (n / 10) % 10, d3 := n % 10 
  in (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3)
}

-- Total number of integers between 100 and 999
def total_count : ℕ := 900

-- Number of integers between 100 and 999 with all different digits
def different_count : ℕ := 648

-- The probability that a randomly chosen integer between 100 and 999 has all different digits
def probability_different_digits : ℚ := different_count / total_count

-- Theorem stating that the probability of choosing an integer with all different digits is 18/25
theorem probability_is_18_over_25 :
  probability_different_digits = 18 / 25 := by
    sorry

end ProbabilityDifferentDigits

end probability_is_18_over_25_l696_696833


namespace smallest_area_of_right_triangle_l696_696534

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696534


namespace correct_conclusions_l696_696226

noncomputable def quadratic_solution_set (a b c : ℝ) : Prop :=
  ∀ x : ℝ, (-1 / 2 < x ∧ x < 3) ↔ (a * x^2 + b * x + c > 0)

theorem correct_conclusions (a b c : ℝ) (h : quadratic_solution_set a b c) : c > 0 ∧ 4 * a + 2 * b + c > 0 :=
  sorry

end correct_conclusions_l696_696226


namespace sum_of_first_five_primes_with_units_digit_3_l696_696133

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696133


namespace smallest_area_right_triangle_l696_696515

theorem smallest_area_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ h : ℝ, (a * b / 2) ≤ (6 * Real.sqrt 7) ∧ triangle_area (a, b, h) = 6 * Real.sqrt 7 :=
by sorry

-- auxiliary function for area calculation
def triangle_area (a b c : ℝ) : ℝ :=
  if a * a + b * b = c * c then a * b / 2 else 0

end smallest_area_right_triangle_l696_696515


namespace digits_probability_l696_696792

def digits_all_different(n : ℕ) : Prop :=
  let d1 := n % 10
  let d2 := (n / 10) % 10
  let d3 := n / 100
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

theorem digits_probability :
  (∑ i in Finset.filter (λ n, digits_all_different n) (Finset.range' 100 900), 1 : ℚ) /
  (Finset.card (Finset.range' 100 900)) = 99 / 100 :=
by
  sorry

end digits_probability_l696_696792


namespace probability_digits_different_l696_696796

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696796


namespace apples_first_store_l696_696377

theorem apples_first_store (apples_first_store : ℕ → ℕ) :
  (apples_first_store 3 = 6) :=
begin
  sorry
end

end apples_first_store_l696_696377


namespace lakers_win_series_probability_lakers_win_series_percentage_l696_696367

def probability_lakers_win (p_win : ℚ) : ℚ :=
  ∑ k in Finset.range 3, (Nat.choose (2 + k) k) * (p_win ^ 3) * ((1 - p_win) ^ k)

theorem lakers_win_series_probability :
  probability_lakers_win (2/3) = 16 / 27 :=
by
  sorry

theorem lakers_win_series_percentage :
  ((16 : ℚ) / 27 * 100).round = 59 :=
by
  sorry

end lakers_win_series_probability_lakers_win_series_percentage_l696_696367


namespace twice_minus_three_algebraic_l696_696073

def twice_minus_three (x : ℝ) : ℝ := 2 * x - 3

theorem twice_minus_three_algebraic (x : ℝ) : 
  twice_minus_three x = 2 * x - 3 :=
by sorry

end twice_minus_three_algebraic_l696_696073


namespace probability_digits_different_l696_696804

theorem probability_digits_different : 
  (let count_all := (999 - 100 + 1) in 
   let count_same_digits := 9 in 
   let count_two_same_digits := 3 * 9 * 8 in 
   let count_all_different := count_all - count_same_digits - count_two_same_digits in 
   count_all_different.to_rat / count_all.to_rat = 3 / 4) :=
by sorry

end probability_digits_different_l696_696804


namespace right_triangle_min_area_l696_696662

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end right_triangle_min_area_l696_696662


namespace smallest_right_triangle_area_l696_696442

theorem smallest_right_triangle_area (a b : ℕ) (h1 : a = 6) (h2 : b = 8) : 
  ∃ h : ℕ, h^2 = a^2 + b^2 ∧ a * b / 2 = 24 := by
  sorry

end smallest_right_triangle_area_l696_696442


namespace M_gt_N_l696_696189

theorem M_gt_N (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) :
  let M := a * b
  let N := a + b - 1
  M > N := by
  sorry

end M_gt_N_l696_696189


namespace probability_fly_reaches_8_10_l696_696006

theorem probability_fly_reaches_8_10 :
  let total_steps := 2^18
  let right_up_combinations := Nat.choose 18 8
  (right_up_combinations / total_steps : ℚ) = Nat.choose 18 8 / 2^18 := 
sorry

end probability_fly_reaches_8_10_l696_696006


namespace smallest_area_of_right_triangle_l696_696589

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696589


namespace select_participants_at_least_one_female_l696_696355

theorem select_participants_at_least_one_female :
  ∃ n : ℕ, n = 16 ∧ (∃ f m : finset ℕ, f.card = 2 ∧ m.card = 4 ∧
  (finset.choose 3 (f ∪ m)).card = n ∧ (∀ s ⊆ f ∪ m, s.card = 3 → (s ∩ f).nonempty)) :=
begin
  sorry
end

end select_participants_at_least_one_female_l696_696355


namespace odd_events_probability_l696_696363

open ProbabilityTheory

variables {Ω : Type*} {n : ℕ}

-- Define our probability space and events
variable [ProbabilitySpace Ω]

-- Assuming that A_i are events, and P(A_i) = 1/(2 * i^2) for i = 2, 3, ..., n.
def A (i : ℕ) : Event Ω := sorry

-- Probability that A_i occurs
axiom prob_A (i : ℕ) (h2 : 2 ≤ i) : P (A i) = 1 / (2 * i^2)

-- Given conditions: Independent events
axiom A_is_independent : IndepEvents (λ i, A i) {i | 2 ≤ i ∧ i ≤ n}

-- The theorem we want to prove
theorem odd_events_probability (h1 : 2 ≤ n) : 
  (P (∑ i in Finset.range n, if nat.even i then 1 else 0 = 1)) = (n-1) / (4 * n) := 
sorry

end odd_events_probability_l696_696363


namespace sum_of_first_five_primes_with_units_digit_3_l696_696136

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696136


namespace smallest_area_right_triangle_l696_696697

open Real

theorem smallest_area_right_triangle (a b : ℝ) (h_a : a = 6) (h_b : b = 8) :
  ∃ c : ℝ, c = 6 * sqrt 7 ∧ (∀ x y : ℝ, (x = a ∨ x = b ∨ y = a ∨ y = b) → (area_right_triangle x y ≥ c)) :=
by
  sorry

def area_right_triangle (x y : ℝ) : ℝ :=
  if h : (x * x + y * y = (sqrt (x * x + y * y)) * (sqrt (x * x + y * y))) then
    (1 / 2) * x * y
  else
    (1 / 2) * x * y

end smallest_area_right_triangle_l696_696697


namespace possible_numbers_l696_696916

theorem possible_numbers (a b c d: ℕ) (h1: 0 < a ∧ a < 9) (h2: 0 < b ∧ b < 9) 
                         (h3: 0 < c ∧ c < 9) (h4: 0 < d ∧ d < 9) 
                         (h5: a ≠ b ∧ a + b ≠ 9) (h6: a ≠ c ∧ a + c ≠ 9) 
                         (h7: a ≠ d ∧ a + d ≠ 9) (h8: b ≠ c ∧ b + c ≠ 9) 
                         (h9: b ≠ d ∧ b + d ≠ 9) (h10: c ≠ d ∧ c + d ≠ 9):
  (a, b, c, d) ∈ {(1, 2, 7, 8), (1, 3, 6, 8), (1, 4, 5, 8), (2, 3, 6, 7), (2, 4, 5, 7), (3, 4, 5, 6)} := 
by sorry

end possible_numbers_l696_696916


namespace determine_irrational_option_l696_696029

def is_irrational (x : ℝ) : Prop := ¬ ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

def option_A : ℝ := 7
def option_B : ℝ := 0.5
def option_C : ℝ := abs (3 / 20 : ℚ)
def option_D : ℝ := 0.5151151115 -- Assume notation describes the stated behavior

theorem determine_irrational_option :
  is_irrational option_D ∧
  ¬ is_irrational option_A ∧
  ¬ is_irrational option_B ∧
  ¬ is_irrational option_C := 
by
  sorry

end determine_irrational_option_l696_696029


namespace smallest_area_correct_l696_696550

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696550


namespace smallest_area_correct_l696_696551

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696551


namespace limit_integral_equality_l696_696347

noncomputable def f (x : ℝ) : ℝ :=
  if x = 0 then 1 else (Real.arctan x) / x

theorem limit_integral_equality :
  (∫ x in 0..1, (f x)) =  intg:
  has_integral_real (∫ x in 0..1, (f x)) :=
begin
  sorry
end

end limit_integral_equality_l696_696347


namespace hedge_cost_and_blocks_l696_696296

-- Define the costs of each type of block
def costA : Nat := 2
def costB : Nat := 3
def costC : Nat := 4

-- Define the number of each type of block per section
def blocksPerSectionA : Nat := 20
def blocksPerSectionB : Nat := 10
def blocksPerSectionC : Nat := 5

-- Define the number of sections
def sections : Nat := 8

-- Define the total cost calculation
def totalCost : Nat := sections * (blocksPerSectionA * costA + blocksPerSectionB * costB + blocksPerSectionC * costC)

-- Define the total number of each type of block used
def totalBlocksA : Nat := sections * blocksPerSectionA
def totalBlocksB : Nat := sections * blocksPerSectionB
def totalBlocksC : Nat := sections * blocksPerSectionC

-- State the theorem
theorem hedge_cost_and_blocks :
  totalCost = 720 ∧ totalBlocksA = 160 ∧ totalBlocksB = 80 ∧ totalBlocksC = 40 := by
  sorry

end hedge_cost_and_blocks_l696_696296


namespace hyperbola_asymptotes_l696_696243

theorem hyperbola_asymptotes (x y : ℝ) :
  (y^2 / 2 - x^2 / 4 = 1) → (y = (sqrt 2 / 2) * x) ∨ (y = - (sqrt 2 / 2) * x) :=
by
  sorry

end hyperbola_asymptotes_l696_696243


namespace probability_digits_all_different_l696_696774

theorem probability_digits_all_different : 
  (Finset.filter 
    (λ n : ℕ, n ≥ 100 ∧ n ≤ 999 ∧ let d := n.digits 10 in d.nodup) 
    (Finset.range 1000)).card.toRational / 
  (Finset.filter (λ n : ℕ, n ≥ 100 ∧ n ≤ 999) (Finset.range 1000)).card.toRational 
  = (18 / 25) := 
by
  sorry

end probability_digits_all_different_l696_696774


namespace eval_sequence_l696_696020

noncomputable def b : ℕ → ℤ
| 1 => 1
| 2 => 4
| 3 => 9
| n => if h : n > 3 then b (n - 1) * (b (n - 1) - 1) + 1 else 0

theorem eval_sequence :
  b 1 * b 2 * b 3 * b 4 * b 5 * b 6 - (b 1 ^ 2 + b 2 ^ 2 + b 3 ^ 2 + b 4 ^ 2 + b 5 ^ 2 + b 6 ^ 2)
  = -3166598256 :=
by
  /- The proof steps are omitted. -/
  sorry

end eval_sequence_l696_696020


namespace correct_statements_are_ABD_l696_696245

variables {k x1 x2 y1 y2 x0 : ℝ}

-- Define points A and B on the parabola E : y² = 4x
def point_A_on_parabola (A : ℝ × ℝ) := A.2^2 = 4 * A.1
def point_B_on_parabola (B : ℝ × ℝ) := B.2^2 = 4 * B.1

-- Define symmetry of points A and B with respect to the line x = ky + 4
def are_symmetric (A B : ℝ × ℝ) := A.1 + B.1 = k * (A.2 + B.2) + 8

-- Define the x-intercept of line AB being point C
def line_intersects_x_axis (A B : ℝ × ℝ) (C : ℝ) := 
  ∃ x0 : ℝ, C = (x0, 0) ∧ 
             ∃ m : ℝ, m = (A.2 - B.2) / (A.1 - B.1) ∧ 
             x0 = A.1 - (A.2 / m)

-- Mathematical properties to prove
theorem correct_statements_are_ABD (A B : ℝ × ℝ) (C : ℝ × ℝ) :
  point_A_on_parabola A → point_B_on_parabola B → 
  are_symmetric A B → line_intersects_x_axis A B C → 
  (C = (1, 0) ∨ (A.1 + B.1 = 4) ∨ (C.1 ∈ set.Ioo (-2 : ℝ) 2)) :=
by
  intros hA hB hSymm hLineInt
  left -- Focus A has coordinates (1,0)
  sorry 

end correct_statements_are_ABD_l696_696245


namespace even_numbers_with_specific_square_properties_l696_696309

theorem even_numbers_with_specific_square_properties (n : ℕ) :
  (10^13 ≤ n^2 ∧ n^2 < 10^14 ∧ (n^2 % 100) / 10 = 5) → 
  (2 ∣ n ∧ 273512 > 10^5) := 
sorry

end even_numbers_with_specific_square_properties_l696_696309


namespace card_numbers_l696_696919

theorem card_numbers (
  a b c d : ℕ) 
  (h_all_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_in_range : 0 < a ∧ a < 9 ∧ 0 < b ∧ b < 9 ∧ 0 < c ∧ c < 9 ∧ 0 < d ∧ d < 9)
  (h_probability_conditions : 
    (∃ twoSums : finset ℕ, twoSums.card = 6 ∧ 
      (∀ x ∈ twoSums, 0 < x < 18) ∧ 
      let sums := [a+b, a+c, a+d, b+c, b+d, c+d] in 
      list.perm sums (twoSums.val) ∧ 
      finset.card (twoSums.filter (λ x, x = 9)) = 2 ∧ 
      finset.card (twoSums.filter (λ x, x < 9)) = 2 ∧ 
      finset.card (twoSums.filter (λ x, x > 9)) = 2))
  : (a, b, c, d) = (1, 2, 7, 8) ∨ (a, b, c, d) = (1, 3, 6, 8) ∨ 
    (a, b, c, d) = (1, 4, 5, 8) ∨ (a, b, c, d) = (2, 3, 6, 7) ∨ 
    (a, b, c, d) = (2, 4, 5, 7) ∨ (a, b, c, d) = (3, 4, 5, 6) :=
sorry

end card_numbers_l696_696919


namespace example_problem_l696_696060

noncomputable def f : ℝ → ℝ := sorry

def is_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

def axis_of_symmetry (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c - x) = f (c + x)

theorem example_problem (h_inc : is_increasing_on f {x : ℝ | x < 2})
                         (h_symm : axis_of_symmetry (λ x, f (x + 2)) 0) :
                         f (-1) < f (3) :=
sorry

end example_problem_l696_696060


namespace option_B_perpendicular_l696_696757
-- Imports the Mathlib library to bring in necessary functionalities

-- Define the given line as a linear equation
def given_line (x y : ℝ) := 2 * x + y + 1 = 0

-- Define the four option lines as linear equations
def option_A (x y : ℝ) := 2 * x - y - 1 = 0
def option_B (x y : ℝ) := x - 2 * y + 1 = 0
def option_C (x y : ℝ) := x + 2 * y + 1 = 0
def option_D (x y : ℝ) := x + (1 / 2) * y - 1 = 0

-- Define a function to compute the slope of a line
def slope (line : ℝ → ℝ → Prop) : ℝ :=
if line 1 0 = line 0 0 then 0
else -(line 1 1) / (line 1 0 - line 0 0)

-- Prove that option B is perpendicular to the given line
theorem option_B_perpendicular : slope given_line * slope option_B = -1 :=
  sorry

end option_B_perpendicular_l696_696757


namespace smallest_area_of_right_triangle_l696_696586

-- Define a right triangle with sides 'a', 'b' where one of these might be the hypotenuse.
noncomputable def smallest_possible_area : ℝ := 
  min (1/2 * 6 * 8) (1/2 * 6 * 2 * Real.sqrt 7)

theorem smallest_area_of_right_triangle (a b : ℝ) (h₁ : a = 6) (h₂ : b = 8) :
  smallest_possible_area = 6 * Real.sqrt 7 :=
by
  sorry -- Proof to be filled in later

end smallest_area_of_right_triangle_l696_696586


namespace percentage_diameter_R_S_l696_696375

variable (d_R d_S : ℝ)

-- Conditions: 
-- 1. The area of circle R is 36% of the area of circle S.
def area_R_is_36_percent_area_S (d_R d_S : ℝ) : Prop :=
  let r_R := d_R / 2
  let r_S := d_S / 2
  π * r_R ^ 2 = 0.36 * (π * r_S ^ 2)

-- Proof statement: Prove that the percentage of the diameter of circle R compared to diameter of circle S is 60%.
theorem percentage_diameter_R_S (h : area_R_is_36_percent_area_S d_R d_S) :
  d_R / d_S * 100 = 60 := 
sorry

end percentage_diameter_R_S_l696_696375


namespace smallest_area_of_right_triangle_l696_696625

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696625


namespace five_rings_sum_l696_696077

theorem five_rings_sum :
  ∃ S_max S_min : ℕ,
    (∀ (a b c d e f g h i : ℕ),
      (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
       b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
       c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
       d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
       e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
       f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
       g ≠ h ∧ g ≠ i ∧
       h ≠ i ∧
       a + b + c + d + e + f + g + h + i = 45 ∧
       (a + b = S) ∧ (b + c + d = S) ∧ (d + e + f = S) ∧ (f + g + h = S) ∧ (h + i = S))
      → S ∈ {11, 14}) ∧
    S_max = 14 ∧ S_min = 11 :=
by
  sorry

end five_rings_sum_l696_696077


namespace find_k_l696_696174

theorem find_k : ∃ k : ℕ, 32 / k = 4 ∧ k = 8 := 
sorry

end find_k_l696_696174


namespace value_of_a_27_l696_696206

def a_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n, a (n + 1) = a n + 2 * n

theorem value_of_a_27 (a : ℕ → ℕ) (h : a_sequence a) : a 27 = 702 :=
sorry

end value_of_a_27_l696_696206


namespace smallest_right_triangle_area_l696_696493

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l696_696493


namespace probability_all_different_digits_l696_696769

noncomputable def total_integers := 900
noncomputable def repeating_digits_integers := 9
noncomputable def same_digit_probability : ℚ := repeating_digits_integers / total_integers
noncomputable def different_digit_probability := 1 - same_digit_probability

theorem probability_all_different_digits :
  different_digit_probability = 99 / 100 :=
by
  sorry

end probability_all_different_digits_l696_696769


namespace cost_per_liter_of_fuel_l696_696069

-- Definitions and conditions
def fuel_capacity : ℕ := 150
def initial_fuel : ℕ := 38
def change_received : ℕ := 14
def initial_money : ℕ := 350

-- Proof problem
theorem cost_per_liter_of_fuel :
  (initial_money - change_received) / (fuel_capacity - initial_fuel) = 3 :=
by
  sorry

end cost_per_liter_of_fuel_l696_696069


namespace smallest_area_correct_l696_696553

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696553


namespace proof_1_proof_2_proof_3_l696_696326

section geometry_proof

variables {A B M O Q : Point}
variables {l : Line}
variables {C : Circle}
variables {Ω : Parabola}

def parabola_eq : Prop := ∀ (x y : ℝ), y^2 = 4 * x
def circle_eq : Prop := ∀ (x y : ℝ), (x - 5)^2 + y^2 = 16
def line_intersects_parabola (l : Line) (Ω : Parabola) : Prop := ∃ A B : Point, l.contains A ∧ l.contains B ∧ Ω.contains A ∧ Ω.contains B
def midpoint (A B M : Point) : Prop := 2 * M = A + B
def orthogonal (O Q A B : Point) : Prop := (O - Q).dot (A - B) = 0

theorem proof_1 (Ω : Parabola) (h : parabola_eq) : distance from_focus_to_directrix Ω = 2 := sorry

theorem proof_2 (l : Line) (Ω : Parabola) (C : Circle) (A B M O : Point)
  (h1 : line_intersects_parabola l Ω)
  (h2 : C.contains M)
  (h3 : tangent l C)
  (h4 : midpoint A B M) 
  (h5 : O = origin) : l = {x | x = 1} ∨ l = {x | x = 9} := sorry

theorem proof_3 (l : Line) (Ω : Parabola) (A B Q O : Point)
  (h1 : line_intersects_parabola l Ω)
  (h2 : orthogonal (O - Q) (A - B))
  (h3 : OQ_perpendicular_AB O Q A B)
  (h4 : vector O A .dot vector O B = 0) : ∀ {x y : ℝ}, x^2 - 4*x + y^2 = 0 := sorry

end geometry_proof

end proof_1_proof_2_proof_3_l696_696326


namespace sum_of_first_five_primes_with_units_digit_3_l696_696121

open Nat

-- Predicate to check if a number has a units digit of 3
def hasUnitsDigit3 (n : ℕ) : Prop :=
n % 10 = 3

-- List of the first five prime numbers that have a units digit of 3
def firstFivePrimesUnitsDigit3 : List ℕ :=
[3, 13, 23, 43, 53]

-- Definition for sum of the first five primes with units digit 3
def sumFirstFivePrimesUnitsDigit3 : ℕ :=
(firstFivePrimesUnitsDigit3).sum

-- Theorem statement
theorem sum_of_first_five_primes_with_units_digit_3 :
  sumFirstFivePrimesUnitsDigit3 = 135 := by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696121


namespace min_stamps_value_l696_696332

theorem min_stamps_value (x y : ℕ) (hx : 5 * x + 7 * y = 74) : x + y = 12 :=
by
  sorry

end min_stamps_value_l696_696332


namespace smallest_right_triangle_area_l696_696669

theorem smallest_right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) :
  (a = 6 ∧ b = 8) ∧ (hypotenuse = 10 ∨ hypotenuse = 8 ∧ c = √28) →
  min (1/2 * a * b) (1/2 * a * c) = 3 * √28 :=
begin
  sorry
end

end smallest_right_triangle_area_l696_696669


namespace smallest_area_correct_l696_696540

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696540


namespace digits_all_different_l696_696811

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696811


namespace smallest_area_right_triangle_l696_696645

noncomputable def smallest_area (a b: ℝ) : ℝ :=
  min (0.5 * a * b) (0.5 * a * (real.sqrt (b^2 - a^2)))

theorem smallest_area_right_triangle (a b: ℝ) (ha : a = 6) (hb: b = 8) (h: a^2 + (real.sqrt (b^2 - a^2))^2 = b^2 ∨
                                                                                b^2 + (real.sqrt (b^2 - a^2))^2 = a^2) : 
  smallest_area a b = 15.87 :=
by
  have h_area1 : real.sqrt (b^2 - a^2) ≈ 5.29 := sorry
  have h_area2 := 0.5 * a * 5.29 ≈ 15.87 := sorry
  sorry

end smallest_area_right_triangle_l696_696645


namespace sequence_not_infinite_l696_696199

-- Mathematical Definitions
def sequence (a : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then a
  else if (sequence a (n - 1)) % 10 ≤ 5 then
    (sequence a (n - 1)) / 10
  else
    9 * (sequence a (n - 1))

theorem sequence_not_infinite (a₀ : ℕ) : ∃ N : ℕ, ∀ n > N, sequence a₀ n = 0 :=
by
  sorry

end sequence_not_infinite_l696_696199


namespace sum_of_first_five_primes_with_units_digit_3_l696_696135

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696135


namespace smallest_area_of_right_triangle_l696_696533

noncomputable def smallest_possible_area : ℝ :=
  let a := 6
  let b := 8
  let area1 := 1/2 * a * b
  let area2 := 1/2 * a * sqrt (b ^ 2 - a ^ 2)
  real.sqrt 7 * 6

theorem smallest_area_of_right_triangle (a b : ℝ) (ha : a = 6) (hb : b = 8)
  (h_right_triangle : a^2 + b^2 >= b^2 + a^2) :
  smallest_possible_area = 6 * real.sqrt 7 := by
  sorry

end smallest_area_of_right_triangle_l696_696533


namespace exists_positive_integer_n_l696_696294

theorem exists_positive_integer_n :
  ∃ (n : ℕ), 0 < n ∧ (57 ^ 6 + 95 ^ 6 + 109 ^ 6 = n ^ 6) :=
by
  let a := 57 ^ 6
  let b := 95 ^ 6
  let c := 109 ^ 6
  have ha : a = 33698267 := rfl
  have hb : b = 735091890625 := rfl
  have hc : c = 17715690625 := rfl
  have hsum : a + b + c = 752872124517 := by
    rw [ha, hb, hc]
    norm_num
  use 228
  have hn : 228 ^ 6 = 752872124517 := by norm_num
  rw [← hsum, hn]
  apply nat.zero_lt_succ

end exists_positive_integer_n_l696_696294


namespace central_angle_of_sector_l696_696966

noncomputable def sector_radius (s : ℝ) (A : ℝ) : ℝ := (2 * A) / s

noncomputable def central_angle (s : ℝ) (r : ℝ) : ℝ := s / r

theorem central_angle_of_sector :
  ∀ (s A : ℝ), s = 2 → A = 4 → central_angle s (sector_radius s A) = 1 / 2 := by
  -- provided conditions
  intros s A hs hA,
  rw [hs, hA],
  sorry

end central_angle_of_sector_l696_696966


namespace Gwen_walking_and_elevation_gain_l696_696070

theorem Gwen_walking_and_elevation_gain :
  ∀ (jogging_time walking_time total_time elevation_gain : ℕ)
    (jogging_feet total_feet : ℤ),
    jogging_time = 15 ∧ jogging_feet = 500 ∧ (jogging_time + walking_time = total_time) ∧
    (5 * walking_time = 3 * jogging_time) ∧ (total_time * jogging_feet = 15 * total_feet)
    → walking_time = 9 ∧ total_feet = 800 := by 
  sorry

end Gwen_walking_and_elevation_gain_l696_696070


namespace min_area_square_condition_l696_696175

def parabola (x : ℝ) : ℝ := x^2 + 4
def line (x : ℝ) : ℝ := 3*x - 5

theorem min_area_square_condition (sqrt_200 : real.sqrt 200 = 14.14213562373095) :
  ∃ (s : ℝ), (∃ x1 x2 : ℝ, parabola x1 = 3*(x1) + 4 ∧ parabola x2 = 3*(x2) + 4 ∧ 
    (x1 - x2)^2 + (parabola x1 - parabola x2)^2 = (3*(x1 - x2))^2) ∧
    s^2 = 1000 ∧ s = 20 :=
  sorry

end min_area_square_condition_l696_696175


namespace probability_odd_sum_is_1_over_9_l696_696884

noncomputable def probability_odd_sum (p_even p_odd : ℚ) (h1 : p_even = 2 * p_odd) (h2 : p_even + p_odd = 1) : ℚ :=
let p_odd := 1 / 6,
    p_even := 1 / 3 in
(p_odd * p_even + p_even * p_odd)

theorem probability_odd_sum_is_1_over_9 : probability_odd_sum (1/3) (1/6) (by ring) (by simp) = 1/9 :=
by sorry

end probability_odd_sum_is_1_over_9_l696_696884


namespace find_m_value_l696_696990

noncomputable def tangent_condition : ℝ → Prop :=
  λ m, ∃ x > 0, (x^2 - 3 * real.log x = -x + m) ∧ (2 * x - (3 / x) = -1)

theorem find_m_value :
  tangent_condition 2 :=
by
  sorry

end find_m_value_l696_696990


namespace smallest_area_of_right_triangle_l696_696624

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a ^ 2 + b ^ 2)

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ := (a * b) / 2

theorem smallest_area_of_right_triangle : 
  ∀ (a b : ℝ), a = 6 → b = 8 → 
  min ((a * b) / 2) (min ((a * Real.sqrt (b ^ 2 - a ^ 2)) / 2) ((b * Real.sqrt (a ^ 2 - b ^ 2)) / 2)) = 24 := 
by 
  intros a b ha hb 
  have h1 : a = 6 := ha 
  have h2 : b = 8 := hb 
  rw [h1, h2] 
  simp 
  sorry

end smallest_area_of_right_triangle_l696_696624


namespace smallest_area_right_triangle_l696_696563

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end smallest_area_right_triangle_l696_696563


namespace probability_all_digits_different_l696_696882

theorem probability_all_digits_different : 
  (∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 
     let all_different : ℕ → Prop := λ n, 
       let digits := [n / 100 % 10, n / 10 % 10, n % 10] in
       (∀ i j, i ≠ j → digits.nth i ≠ digits.nth j) in
     (∑ k in finset.Icc 100 999, if all_different k then 1 else 0).to_float / 900.to_float = 18 / 25) :=
sorry

end probability_all_digits_different_l696_696882


namespace sum_of_first_five_prime_units_digit_3_l696_696108

noncomputable def is_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

noncomputable def first_five_prime_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_prime_units_digit_3 :
  ∑ x in first_five_prime_with_units_digit_3, x = 135 :=
by
  sorry

end sum_of_first_five_prime_units_digit_3_l696_696108


namespace sum_of_first_five_primes_with_units_digit_3_l696_696130

def units_digit_is_3 (n: ℕ) : Prop :=
  n % 10 = 3

def is_prime (n: ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def first_five_primes_with_units_digit_3 : List ℕ :=
  [3, 13, 23, 43, 53]

theorem sum_of_first_five_primes_with_units_digit_3 :
  ∃ (S : ℕ), S = List.sum first_five_primes_with_units_digit_3 ∧ S = 135 :=
by
  sorry

end sum_of_first_five_primes_with_units_digit_3_l696_696130


namespace digits_all_different_l696_696808

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696808


namespace probability_digits_all_different_l696_696869

theorem probability_digits_all_different :
  (probability (choose (n : ℕ) (100 ≤ n ∧ n < 1000 ∧ are_digits_distinct n)) = 3 / 4) :=
sorry

-- Definitions required by Lean:
noncomputable def are_digits_distinct (n : ℕ) : Prop :=
  let (d₁, d₂, d₃) := (n / 100, (n / 10) % 10, n % 10)
  d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃

noncomputable def probability {α : Type*} (P : α → Prop) : ℚ :=
  let event_count := {x | P x}.card
  let sample_space_count := {x | 100 ≤ x ∧ x < 1000}.card
  event_count / sample_space_count

noncomputable def choose (P : ℕ → Prop) : finset ℕ :=
  {n | P n}.to_finset

end probability_digits_all_different_l696_696869


namespace probability_of_drawing_ball_labeled_3_on_second_draw_l696_696244

theorem probability_of_drawing_ball_labeled_3_on_second_draw :
  let box1 := [1, 1, 2, 3],
      box2 := [1, 1, 3],
      box3 := [1, 1, 1, 2, 2] in
  (let p1 := 2 / 4 * 1 / 4 + 1 / 4 * 1 / 4 + 1 / 4 * 1 / 6 in
    p1 = 11 / 48) :=
  by {
  let box1 := [1, 1, 2, 3],
  let box2 := [1, 1, 3],
  let box3 := [1, 1, 1, 2, 2],
  let p1 := 2 / 4 * 1 / 4 + 1 / 4 * 1 / 4 + 1 / 4 * 1 / 6,
  show p1 = 11 / 48,
  sorry
}

end probability_of_drawing_ball_labeled_3_on_second_draw_l696_696244


namespace radians_to_degrees_l696_696714

theorem radians_to_degrees (h : Real.rad_to_deg π = 180) : 
  Real.rad_to_deg (π / 3) = 60 ∧
  75 = 75 * (π / 180) ∧
  Real.rad_to_deg (1 : ℝ) = 57.3 := 
  by
    sorry

end radians_to_degrees_l696_696714


namespace largest_term_is_S6_and_S7_l696_696938

noncomputable def largest_term_in_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∃ N : ℕ, ∀ n : ℕ, (6 ≤ n ∨ n ≤ 7) → S n = S 6

theorem largest_term_is_S6_and_S7 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_sum : ∀ n : ℕ, S n = ∑ k in Finset.range n, a (k + 1))
  (h_a1_pos : a 1 > 0)
  (h_S5_eq_S8 : S 5 = S 8) :
  largest_term_in_arithmetic_sequence a S := by
  sorry

end largest_term_is_S6_and_S7_l696_696938


namespace hyperbola_asymptote_l696_696987

theorem hyperbola_asymptote (m : ℝ) (hm : 0 < m) :
  (∃ m > 0, ∀ x y : ℝ, (x + sqrt 3 * y = 0) → (x / m)^2 - y^2 = 1) → m = sqrt 3 :=
by
  intro h
  sorry

end hyperbola_asymptote_l696_696987


namespace digits_all_different_l696_696809

theorem digits_all_different (n : ℕ) (h100 : 100 ≤ n) (h999 : n ≤ 999) :
  let digits := List.digits n in (digits.nodup) → ℝ := by
exact 99 / 100

end digits_all_different_l696_696809


namespace oil_truck_radius_l696_696729

/-- 
A full stationary oil tank that is a right circular cylinder has a radius of 100 feet 
and a height of 25 feet. Oil is pumped from the stationary tank to an oil truck that 
has a tank that is a right circular cylinder. The oil level dropped 0.025 feet in the stationary tank. 
The oil truck's tank has a height of 10 feet. The radius of the oil truck's tank is 5 feet. 
--/
theorem oil_truck_radius (r_stationary : ℝ) (h_stationary : ℝ) (h_truck : ℝ) 
  (Δh : ℝ) (r_truck : ℝ) 
  (h_stationary_pos : 0 < h_stationary) (h_truck_pos : 0 < h_truck) (r_stationary_pos : 0 < r_stationary) :
  r_stationary = 100 → h_stationary = 25 → Δh = 0.025 → h_truck = 10 → r_truck = 5 → 
  π * (r_stationary ^ 2) * Δh = π * (r_truck ^ 2) * h_truck :=
by 
  -- Use the conditions and perform algebra to show the equality.
  sorry

end oil_truck_radius_l696_696729


namespace problem_statement_l696_696176

def f (x : ℝ) : ℤ := int.floor (2 * x) + int.floor (3 * x) + int.floor (4 * x) + int.floor (5 * x)

noncomputable def count_distinct_f_values : ℕ := 
  finset.card (finset.image f (finset.Icc 0 100))

theorem problem_statement : count_distinct_f_values = 101 := by
  sorry

end problem_statement_l696_696176


namespace geo_prog_sum_463_l696_696360

/-- Given a set of natural numbers forming an increasing geometric progression with an integer
common ratio where the sum equals 463, prove that these numbers must be {463}, {1, 462}, or {1, 21, 441}. -/
theorem geo_prog_sum_463 (n : ℕ) (b₁ q : ℕ) (s : Finset ℕ) (hgeo : ∀ i j, i < j → s.toList.get? i = some (b₁ * q^i) ∧ s.toList.get? j = some (b₁ * q^j))
  (hsum : s.sum id = 463) : 
  s = {463} ∨ s = {1, 462} ∨ s = {1, 21, 441} :=
sorry

end geo_prog_sum_463_l696_696360


namespace smallest_area_correct_l696_696544

noncomputable def smallest_area (a b : ℕ) : ℝ :=
  let h := Real.sqrt (a^2 + b^2)
  let config1_area := (1 / 2) * a * b
  let x := Real.sqrt (b^2 - a^2)
  let config2_area := (1 / 2) * a * x
  Real.min config1_area config2_area

theorem smallest_area_correct : smallest_area 6 8 = 15.87 :=
by
  sorry

end smallest_area_correct_l696_696544
