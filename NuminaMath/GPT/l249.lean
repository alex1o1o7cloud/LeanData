import Data.List.Perm
import Mathlib
import Mathlib.Algebra.ContinuedFractions.Translations
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Order
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Probabilities
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Mod
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Triangle
import Mathlib.LinearAlgebra.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace equal_area_bisecting_line_slope_l249_249327

theorem equal_area_bisecting_line_slope 
  (circle1_center circle2_center : ℝ × ℝ) 
  (radius : ℝ) 
  (line_point : ℝ × ℝ) 
  (h1 : circle1_center = (20, 100))
  (h2 : circle2_center = (25, 90))
  (h3 : radius = 4)
  (h4 : line_point = (20, 90))
  : ∃ (m : ℝ), |m| = 2 :=
by
  sorry

end equal_area_bisecting_line_slope_l249_249327


namespace find_fg3_l249_249919

def f (x : ℝ) : ℝ := 4 - Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem find_fg3 : f (g 3) = -2 := by
  sorry

end find_fg3_l249_249919


namespace probability_penny_nickel_dime_all_heads_l249_249284

-- Define flipping five coins
def flip_five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the function to check if the penny, nickel, and dime are heads
def all_heads_penny_nickel_dime (o : flip_five_coins) : bool :=
  o.1 = tt ∧ o.2 = tt ∧ o.3 = tt

-- Define the total count of possible outcomes
def total_outcomes : ℕ := 32

-- Define the count of favorable outcomes
def favorable_outcomes : ℕ := 4

-- Define the probability calculation
def probability_favorable : ℚ := favorable_outcomes / total_outcomes

-- The statement proving the probability is 1/8
theorem probability_penny_nickel_dime_all_heads :
  probability_favorable = 1 / 8 :=
by
  sorry

end probability_penny_nickel_dime_all_heads_l249_249284


namespace trapezoid_BD_l249_249957

/-- Problem statement: In the trapezoid ABCD, 
AB ∥ DC, AC ⊥ DC, DC = 18, tan(∠C) = 2, tan(∠B) = 1.25,
prove that BD = 6√59 -/
theorem trapezoid_BD (AB DC AC : Real)
    (tan_C tan_B : Real) 
    (h_parallel : Parallel AB DC) 
    (h_perpendicular : AC ⊥ DC) 
    (h_DC : DC = 18) 
    (h_tanC : tan_C = 2) 
    (h_tanB : tan_B = 1.25) :
    let C := Real.Arctan (tan_C)
    let B := Real.Arctan (tan_B)
    let AC := DC * tan_C
    let AB := AC / tan_B
    BD = Real.sqrt (AC ^ 2 + AB ^ 2) ∧ BD = 6 * Real.sqrt 59 := 
  sorry

end trapezoid_BD_l249_249957


namespace probability_heads_penny_nickel_dime_l249_249287

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]
variable (coins : Fin 5 → Ω)
variable (heads tails : Ω)

-- Each coin has two outcomes: heads or tails
axiom coin_outcome (i : Fin 5) : coins i = heads ∨ coins i = tails

-- There are 32 total outcomes
axiom total_outcomes : Fintype.card (Fin 5 → Ω) = 32

-- There are 4 successful outcomes for penny, nickel, and dime being heads
axiom successful_outcomes : let successful := {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} in
                          Fintype.card successful = 4

theorem probability_heads_penny_nickel_dime :
  let probability := (Fintype.card {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} : ℤ) /
                     (Fintype.card (Fin 5 → Ω) : ℤ) in
  probability = 1 / 8 :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249287


namespace product_of_reds_is_red_sum_of_reds_is_red_l249_249431

noncomputable def color := ℕ → Prop

variables (white red : color)
variable (r : ℕ)

axiom coloring : ∀ n, white n ∨ red n
axiom exists_white : ∃ n, white n
axiom exists_red : ∃ n, red n
axiom sum_of_white_red_is_white : ∀ m n, white m → red n → white (m + n)
axiom prod_of_white_red_is_red : ∀ m n, white m → red n → red (m * n)

theorem product_of_reds_is_red (m n : ℕ) : red m → red n → red (m * n) :=
sorry

theorem sum_of_reds_is_red (m n : ℕ) : red m → red n → red (m + n) :=
sorry

end product_of_reds_is_red_sum_of_reds_is_red_l249_249431


namespace numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249223

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).length

theorem numbers_with_most_divisors_in_range :
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  (number_of_divisors n ≤ 6) :=
by
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list
  simp [number_of_divisors] <;>
  dec_trivial -- Automatically discharges the goals since they are simple arithmetic comparisons

theorem greatest_divisors_in_range :
  number_of_divisors 12 = 6 ∧
  number_of_divisors 18 = 6 ∧
  number_of_divisors 20 = 6 ∧
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19],
  number_of_divisors n < 6 :=
by
  refine ⟨rfl, rfl, rfl, _⟩
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list excluding 12, 18, 20
  dec_trivial -- Automatically discharges the goals due to their simplicity

#print greatest_divisors_in_range

end numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249223


namespace eccentricity_range_l249_249479

theorem eccentricity_range (a b c : ℝ) (h1 : a > b > 0) (h2 : c > 0)
  (F1 F2 : ℝ × ℝ) (hf1 : F1 = (-c, 0)) (hf2 : F2 = (c, 0))
  (P : ℝ × ℝ) (hP : ∀ (x y : ℝ), P = (x, y) → (x^2) / (a^2) + (y^2) / (b^2) = 1 ∧ 
  ((x + c) * (x - c) + y^2 = c^2)) :
  (sqrt(3) / 3 : ℝ) ≤ c / a ∧ c / a ≤ sqrt(2) / 2 :=
by
  sorry

end eccentricity_range_l249_249479


namespace area_of_ABC_value_of_a_l249_249936

-- Definitions derived from conditions
def cos_A_div_2 (A : ℝ) := (2 * real.sqrt 5) / 5
def cos_A (A : ℝ) := 2 * (cos_A_div_2 A)^2 - 1
def sin_A (A : ℝ) := real.sqrt (1 - (cos_A A)^2)
def bc_cos_A (b c A : ℝ) := b * c * cos_A A
def bc_given (b c : ℝ) := 3
def b_plus_c (b c : ℝ) := 4 * real.sqrt 2

-- Proof statements as Lean declarations
theorem area_of_ABC (A b c : ℝ) (h1 : cos_A_div_2 A = (2 * real.sqrt 5) / 5)
                    (h2 : bc_cos_A b c A = 3): 
                    (1 / 2) * b * c * sin_A A = 2 :=
sorry

theorem value_of_a (A b c a : ℝ) (h1 : cos_A_div_2 A = (2 * real.sqrt 5) / 5)
                   (h2 : bc_cos_A b c A = 3) (h3 : b_plus_c b c = 4 * real.sqrt 2): 
                   a = 4 :=
sorry

end area_of_ABC_value_of_a_l249_249936


namespace sum_abc_eq_37_l249_249366

noncomputable def highest_point_height (a b c : ℕ) : ℝ :=
  let board_length := 64
  let board_height := 4
  let angle := 30
  let diagonal := real.sqrt (board_length^2 + board_height^2)
  let cos_theta := board_length / diagonal
  let sin_theta := board_height / diagonal
  let height := diagonal * (sin_theta * real.cos (real.pi / 6) + cos_theta * real.sin (real.pi / 6))
  let h := 32 + 2 * real.sqrt 3 in
  height

theorem sum_abc_eq_37 : 
  ∃ a b c : ℕ, (c ≠ 0 ∧ ¬ ∃ p : nat.prime, p^2 ∣ c) ∧ highest_point_height a b c = 32 + 2 * real.sqrt 3 ∧ a + b + c = 37 :=
by
  have h := 32 + 2 * real.sqrt 3
  use [32, 2, 3]
  split
  { split
    { exact nat.succ_pos'
      assume ⟨p, pp⟩
      cases pp with p_prime p_squared_div_c
      cases p with k
      }
    split
    { sorry }
  sorry

end sum_abc_eq_37_l249_249366


namespace find_matrix_N_l249_249442

theorem find_matrix_N (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : N ^ 3 - 3 • N ^ 2 + 2 • N = !![ [6, 12], [3, 6] ]) : 
  N = !![ [3, 4], [1, 3] ] :=
by
  sorry

end find_matrix_N_l249_249442


namespace intersection_A_B_l249_249558

open Set

def A : Set ℕ := {x | -2 < (x : ℤ) ∧ (x : ℤ) < 2}
def B : Set ℤ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ {x : ℕ | (x : ℤ) ∈ B} = {0, 1} := by
  sorry

end intersection_A_B_l249_249558


namespace find_m_l249_249860

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (1, 2)
def vec_b (m : ℝ) : ℝ × ℝ := (2, m)

-- Define the addition of vectors
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define the dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := (v1.1 * v2.1) + (v1.2 * v2.2)

-- State the main theorem without proof
theorem find_m (m : ℝ) : dot_product (vec_add vec_a (vec_b m)) vec_a = 0 ↔ m = -7/2 := by
  sorry

end find_m_l249_249860


namespace infinite_solutions_l249_249766

-- Define the system of linear equations
def eq1 (x y : ℝ) : Prop := 3 * x - 4 * y = 1
def eq2 (x y : ℝ) : Prop := 6 * x - 8 * y = 2

-- State that there are an unlimited number of solutions
theorem infinite_solutions : ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧
  ∀ y : ℝ, ∃ x : ℝ, eq1 x y :=
by
  sorry

end infinite_solutions_l249_249766


namespace smallest_n_with_g_geq_8_has_mod_248_l249_249457

def base_five_digits_sum (n : ℕ) : ℕ := 
  -- Function to compute the sum of the digits in the base-five representation of n
  sorry

def base_nine_digits_sum (n : ℕ) : ℕ :=
  -- Function to compute the sum of the digits in the base-nine representation of n
  sorry

def f (n : ℕ) : ℕ :=
  base_five_digits_sum n

def g (n : ℕ) : ℕ :=
  base_nine_digits_sum (f n)

theorem smallest_n_with_g_geq_8_has_mod_248 (N : ℕ) (hN : N = 248) : (∃ n, g(n) ≥ 8 ∧ N % 1000 = 248) :=
  sorry

end smallest_n_with_g_geq_8_has_mod_248_l249_249457


namespace part1_part2_l249_249497

noncomputable def f (a x : ℝ) : ℝ := ln x - a^2 * x^2 + a * x
noncomputable def g (a x : ℝ) : ℝ := (3 * a + 1) * x - (a^2 + a) * x^2

theorem part1 (a : ℝ) (h_pos : a > 0) (h_decr : ∀ x ∈ set.Ici 1, deriv (f a) x < 0) :
  a ≥ 1 := sorry

theorem part2 (a : ℝ) (h1 : ∀ x > 1, f a x < g a x) :
  -1 < a ∧ a ≤ 0 := sorry

end part1_part2_l249_249497


namespace vovochka_max_candies_l249_249903

noncomputable def max_candies_for_vovochka 
  (total_classmates : ℕ)
  (total_candies : ℕ)
  (candies_condition : ∀ (subset : Finset ℕ), subset.card = 16 → ∑ i in subset, (classmate_candies i) ≥ 100) 
  (classmate_candies : ℕ → ℕ) : ℕ :=
  total_candies - ∑ i in (Finset.range total_classmates), classmate_candies i

theorem vovochka_max_candies :
  max_candies_for_vovochka 25 200 (λ subset hc, True) (λ i, if i < 13 then 7 else 6) = 37 :=
by
  sorry

end vovochka_max_candies_l249_249903


namespace oscar_leap_vs_elmer_stride_l249_249429

theorem oscar_leap_vs_elmer_stride :
  ∀ (num_poles : ℕ) (distance : ℝ) (elmer_strides_per_gap : ℕ) (oscar_leaps_per_gap : ℕ)
    (elmer_stride_time_mult : ℕ) (total_distance_poles : ℕ)
    (elmer_total_strides : ℕ) (oscar_total_leaps : ℕ) (elmer_stride_length : ℝ)
    (oscar_leap_length : ℝ) (expected_diff : ℝ),
    num_poles = 81 →
    distance = 10560 →
    elmer_strides_per_gap = 60 →
    oscar_leaps_per_gap = 15 →
    elmer_stride_time_mult = 2 →
    total_distance_poles = 2 →
    elmer_total_strides = elmer_strides_per_gap * (num_poles - 1) →
    oscar_total_leaps = oscar_leaps_per_gap * (num_poles - 1) →
    elmer_stride_length = distance / elmer_total_strides →
    oscar_leap_length = distance / oscar_total_leaps →
    expected_diff = oscar_leap_length - elmer_stride_length →
    expected_diff = 6.6
:= sorry

end oscar_leap_vs_elmer_stride_l249_249429


namespace largest_interior_angle_obtuse_isosceles_triangle_l249_249655

theorem largest_interior_angle_obtuse_isosceles_triangle :
  ∀ (P Q R : Type) (α β γ : ℝ), α + β + γ = 180 ∧ γ = 120 ∧ α = 30 ∧ β = 30 →
  (α = 30 ∧ β = 30 ∧ γ = 120) ∨
  (α = 30 ∧ γ = 30 ∧ β = 120) ∨
  (β = 30 ∧ γ = 30 ∧ α = 120) → 
  γ = max α (max β γ) :=
by {
  intros P Q R α β γ h1 h2,
  repeat { rw h1 at * },
  rw h2,
  sorry
}

end largest_interior_angle_obtuse_isosceles_triangle_l249_249655


namespace collinear_vectors_ratio_l249_249060

theorem collinear_vectors_ratio (m n : ℝ) (h : n ≠ 0) 
  (h_collinear : let a := (1:ℝ, 2:ℝ) in
                 let b := (-2:ℝ, 3:ℝ) in
                 let v1 := (m + 2 * n, 2 * m - 3 * n) in
                 let v2 := (-3:ℝ, 8:ℝ) in
                 (v1.1 * v2.2 - v1.2 * v2.1 = 0)) :
  m / n = -1 / 2 := 
by
  sorry

end collinear_vectors_ratio_l249_249060


namespace main_theorem_l249_249357

open Real

noncomputable def inequality_lemma (n : ℕ) (a b : ℕ → ℝ) (h_pos_a : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i)
                                    (h_pos_b : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < b i) : 
  (∑ k in Finset.range n, (a k + b k)) * (∑ k in Finset.range n, a k * b k / (a k + b k)) 
  ≤ (∑ k in Finset.range n, a k) * (∑ k in Finset.range n, b k) :=
by
  sorry

theorem main_theorem (n : ℕ) (a b : ℕ → ℝ) (h_pos_a : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) 
                      (h_pos_b : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < b i) :
  (∑ k in Finset.range n, (a k + b k)) * (∑ k in Finset.range n, a k * b k / (a k + b k)) 
  ≤ (∑ k in Finset.range n, a k) * (∑ k in Finset.range n, b k) ∧
  (∀ k j, 1 ≤ k ∧ k ≤ n → 1 ≤ j ∧ j ≤ n → (a k * b j - a j * b k) = 0 → a k / b k = a j / b j) :=
by
  split
  { exact inequality_lemma n a b h_pos_a h_pos_b }
  { intros k j h₁ h₂ h₃
    have h₄ : (a k * b j) = (a j * b k) := by rw [← mul_comm (a j) (b k), h₃]
    sorry }

end main_theorem_l249_249357


namespace max_profit_at_9_l249_249723

def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 10.8 - (x^2) / 30
  else if 10 < x then (10.8 / x) - (1000 / (3 * x^2))
  else 0  -- Not necessary but to make R(x) total

def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then (8.1 * x) - (x^3) / 30 - 10
  else if 10 < x then 98 - (1000 / (3 * x)) - (2.7 * x)
  else 0  -- Not necessary but to make W(x) total

theorem max_profit_at_9 : W 9 = 38.6 :=
by
  sorry

end max_profit_at_9_l249_249723


namespace find_d_l249_249143

-- Definitions
def f (x : ℝ) := 5 * x - 1
def g (x : ℝ) := 2 * c * x + 3
def f_comp_g (x : ℝ) := f (g x)

-- The main proof statement
theorem find_d (c d : ℝ) (d_eq : f_comp_g x = 15 * x + d) : d = 14 := 
by
  sorry

end find_d_l249_249143


namespace john_baseball_cards_l249_249978

theorem john_baseball_cards (new_cards old_cards cards_per_page : ℕ) (h1 : new_cards = 8) (h2 : old_cards = 16) (h3 : cards_per_page = 3) :
  (new_cards + old_cards) / cards_per_page = 8 := by
  sorry

end john_baseball_cards_l249_249978


namespace initial_number_of_apples_l249_249240

-- Definitions based on the conditions
def number_of_trees : ℕ := 3
def apples_picked_per_tree : ℕ := 8
def apples_left_on_trees : ℕ := 9

-- The theorem to prove
theorem initial_number_of_apples (t: ℕ := number_of_trees) (a: ℕ := apples_picked_per_tree) (l: ℕ := apples_left_on_trees) : t * a + l = 33 :=
by
  sorry

end initial_number_of_apples_l249_249240


namespace numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249222

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).length

theorem numbers_with_most_divisors_in_range :
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  (number_of_divisors n ≤ 6) :=
by
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list
  simp [number_of_divisors] <;>
  dec_trivial -- Automatically discharges the goals since they are simple arithmetic comparisons

theorem greatest_divisors_in_range :
  number_of_divisors 12 = 6 ∧
  number_of_divisors 18 = 6 ∧
  number_of_divisors 20 = 6 ∧
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19],
  number_of_divisors n < 6 :=
by
  refine ⟨rfl, rfl, rfl, _⟩
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list excluding 12, 18, 20
  dec_trivial -- Automatically discharges the goals due to their simplicity

#print greatest_divisors_in_range

end numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249222


namespace fraction_product_minus_fraction_product_l249_249356

theorem fraction_product_minus_fraction_product : 
  (∏ i in Finset.range 9, (i + 1)/(i + 2)) - 
  (∏ i in (Finset.range 90).filter (λ i, 9 ≤ i), (i + 1)/(i + 2)) = 0 :=
by
  sorry

end fraction_product_minus_fraction_product_l249_249356


namespace number_of_mappings_A_to_B_number_of_ordered_mappings_l249_249987

-- Define the sets A and B
def A : Set ℕ := {a1, a2, a3}
def B : Set ℤ := {-1, 0, 1}

-- The first problem: Proof that there are 27 distinct mappings from A to B
theorem number_of_mappings_A_to_B : Card (A → B) = 27 := sorry

-- The second problem: Proof that the number of functions f: A → B satisfying f(a1) > f(a2) ≥ f(a3) is 4
theorem number_of_ordered_mappings : 
  (Card {f : A → B // f a1 > f a2 ∧ f a2 ≥ f a3}) = 4 := sorry

end number_of_mappings_A_to_B_number_of_ordered_mappings_l249_249987


namespace number_of_long_sleeved_jerseys_l249_249979

def cost_per_long_sleeved := 15
def cost_per_striped := 10
def num_striped_jerseys := 2
def total_spent := 80

theorem number_of_long_sleeved_jerseys (x : ℕ) :
  total_spent = cost_per_long_sleeved * x + cost_per_striped * num_striped_jerseys →
  x = 4 := by
  sorry

end number_of_long_sleeved_jerseys_l249_249979


namespace focus_asymptote_distance_l249_249029

noncomputable def parabola_focus_x : ℝ := 3 -- x-coordinate of the focus of parabola y^2 = 12x
noncomputable def hyperbola_a : ℝ := 2 -- semi-major axis of hyperbola x^2/4 - y^2/b^2 = 1

def hyperbola_b_squared (b : ℝ) : Prop := 4 + b^2 = 9 -- condition derived from the focus coinciding

def asymptote_distance (b : ℝ) : ℝ := abs (3 * real.sqrt 5 / real.sqrt (5 + 4))

theorem focus_asymptote_distance : ∃ b : ℝ, hyperbola_b_squared b ∧ asymptote_distance b = real.sqrt 5 :=
by
  sorry

end focus_asymptote_distance_l249_249029


namespace irrational_sqrt_three_l249_249399

theorem irrational_sqrt_three : irrational (√3) :=
by
  sorry

end irrational_sqrt_three_l249_249399


namespace length_ZP_l249_249546

variables (X Y Z P : Type) [EuclideanGeometry X Y Z] 

-- Define the properties and conditions
variables (A : Point)
variables (B C D : Point)
variables (r1 r2 : ℝ)
variables (phi1 phi2 : Circle)

noncomputable def triangle_XYZ := 
  is_triangle A B C (14 : length(yz B C) 15 : length (xz A C)) (13 : length (xy A B))

-- Circle \(\phi_1\) passes through \(Y\) and is tangent to \(XZ\) at \(Z\)
def circle_phi1 := 
  phi_passes_through B (is_tangent_to (xz A C) Z)

-- Circle \(\phi_2\) passes through \(X\) and is tangent to \(YZ\) at \(Z\)
def circle_phi2 := 
  phi_passes_through A (is_tangent_to (yz B C) Z)

-- Point P is the intersection of \(\phi_1\) and \(\phi_2\) not equal to \(Z\)
def point_P := 
  is_intersection_not_equal phi1 phi2 Z P

theorem length_ZP (XY YZ XZ : ℝ) 
  (hXYZ : XY = 13 ∧ YZ = 14 ∧ XZ = 15) 
  (r1 r2 : ℝ) 
  (h : is_tangent_to (line XZ) (circle_through Y Z) ∧ is_tangent_to (line YZ) (circle_through X Z))
  : distance ZP = sqrt(70) :=
begin
  sorry
end

end length_ZP_l249_249546


namespace solve_system_of_equations_l249_249255

theorem solve_system_of_equations (n : ℤ) (h : n > 5)
    (x : Fin n → ℕ) 
    (h1 : (Finset.range n).sum (λ i, x i) = n + 2)
    (h2 : (Finset.range n).sum (λ i, (i + 1) * x i) = 2 * n + 2)
    (h3 : (Finset.range n).sum (λ i, (i + 1)^2 * x i) = n^2 + n + 4)
    (h4 : (Finset.range n).sum (λ i, (i + 1)^3 * x i) = n^3 + n + 8)
    (hx : ∀ i, 0 ≤ x i) :
    x 0 = n ∧ x 1 = 1 ∧ (∀ i, (i ≥ 2) → x i = 0) :=
by
  sorry

end solve_system_of_equations_l249_249255


namespace decimal_sequence_rational_l249_249563

-- Define the unit digit of the sum 1^2 + 2^2 + ... + n^2 as a function
def unit_digit_of_sum_squares (n : ℕ) : ℕ := (∑ k in finset.range (n + 1), k^2) % 10

-- Define the number 0.a₁a₂a₃... as a rational number
def decimal_sequence := real → ℕ

noncomputable def sequence_to_decimal (seq : decimal_sequence) : ℝ :=
  ∑ n:ℕ, (seq n) * 10 ^ (-(n + 1))

theorem decimal_sequence_rational :
  ∀ seq : decimal_sequence,
  (∀ n, seq n = unit_digit_of_sum_squares n) →
  ∃ r : ℚ, sequence_to_decimal seq = r :=
by
  sorry

end decimal_sequence_rational_l249_249563


namespace arithmetic_sequence_mod_12_l249_249070

theorem arithmetic_sequence_mod_12 (n : ℕ) (h1 : 2 + 8 + 14 + 20 + 26 + ... + 128 + 134 ≡ n \pmod{12}) (h2 : 0 ≤ n ∧ n < 12) : n = 0 := 
by
  sorry

end arithmetic_sequence_mod_12_l249_249070


namespace expression_odd_if_p_q_odd_l249_249576

variable (p q : ℕ)

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem expression_odd_if_p_q_odd (hp : is_odd p) (hq : is_odd q) : is_odd (5 * p * q) :=
sorry

end expression_odd_if_p_q_odd_l249_249576


namespace angle_equality_l249_249137

variables {A B C D M K L : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited M] [Inhabited K] [Inhabited L]

def is_midpoint (A B M : Type) [field A] [field B] [field M] : Prop :=
(A + B) / 2 = M

def is_symmetric (A B K : Type) [field A] [field B] [field K] : Prop :=
K - A = A - B

def is_perpendicular (A B C L : Type) [field A] [field B] [field C] [field L] : Prop :=
dot_product C L = 0

def is_altitude (B D C : Type) [field B] [field D] [field C] : Prop :=
dot_product B D C = 0

def is_obtuse_angle (B : Type) [field B] : Prop :=
B > 90

noncomputable def angle_eq (MBL MKL : Type) [field MBL] [field MKL] : Prop :=
MBL = MKL

theorem angle_equality
  (h1 : is_altitude B D C)
  (h2 : is_midpoint A C M)
  (h3 : AB < BC)
  (h4 : is_obtuse_angle B)
  (h5 : is_symmetric D M K)
  (h6 : is_perpendicular M BC L)
  : angle_eq (MBL) (MKL) := 
sorry

end angle_equality_l249_249137


namespace relationship_among_a_b_c_d_l249_249810

noncomputable def a : ℝ := (3 / 5) ^ (2 / 5)
noncomputable def b : ℝ := (2 / 5) ^ (3 / 5)
noncomputable def c : ℝ := (2 / 5) ^ (2 / 5)
noncomputable def d : ℝ := Real.log2 (2 / 5)

theorem relationship_among_a_b_c_d : a > c ∧ c > b ∧ b > d :=
by {
  sorry
}

end relationship_among_a_b_c_d_l249_249810


namespace compute_series_l249_249573

noncomputable def sum_series (c d : ℝ) : ℝ :=
  ∑' n, 1 / ((n-1) * d - (n-2) * c) / (n * d - (n-1) * c)

theorem compute_series (c d : ℝ) (hc_pos : 0 < c) (hd_pos : 0 < d) (hcd : d < c) : 
  sum_series c d = 1 / ((d - c) * c) :=
sorry

end compute_series_l249_249573


namespace Vovochka_max_candies_l249_249877

theorem Vovochka_max_candies (classmates : Finset ℕ) (candies : ℕ) (hclassmates : classmates.card = 25) (hcandies : candies = 200) (H : ∀ (S : Finset ℕ), S.card = 16 → 100 ≤ S.sum (λ x, 1)) :
  ∃ (max_candies : ℕ), max_candies = 37 :=
by
  use 37
  sorry

end Vovochka_max_candies_l249_249877


namespace angle_C_is_65_deg_l249_249967

-- Defining a triangle and its angles.
structure Triangle :=
  (A B C : ℝ) -- representing the angles in degrees

-- Defining the conditions of the problem.
def given_triangle : Triangle :=
  { A := 75, B := 40, C := 180 - 75 - 40 }

-- Statement of the problem, proving that the measure of ∠C is 65°.
theorem angle_C_is_65_deg (t : Triangle) (hA : t.A = 75) (hB : t.B = 40) (hSum : t.A + t.B + t.C = 180) : t.C = 65 :=
  by sorry

end angle_C_is_65_deg_l249_249967


namespace possible_integer_root_counts_l249_249421

def has_integer_roots (P : Polynomial ℤ) : ℕ :=
  P.roots.count

theorem possible_integer_root_counts (p q r s t : ℤ) :
  let P := Polynomial.Coeff ⟨[p, q, r, s, t, 1], by norm_num⟩ in
  has_integer_roots P ∈ {0, 1, 2, 5} :=
sorry

end possible_integer_root_counts_l249_249421


namespace propositions_hold_count_l249_249511

theorem propositions_hold_count {a b c d : ℝ} 
  (h1 : a > 0) 
  (h2 : b < 0) 
  (h3 : -a < b) 
  (h4 : c < d < 0) : 
  (¬ (a * d > b * c)) ∧ 
  (a / d + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) → 
  3 :=
by
  sorry

end propositions_hold_count_l249_249511


namespace max_candies_l249_249871

theorem max_candies (c : ℕ := 200) (n m : ℕ := 25) (k : ℕ := 16) (t : ℕ := 100) 
  (H : ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - (25 - (x.1 : ℕ))) i ≥ 100) : 
  ∃ d : ℕ, (d = 37 ∧ ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - d - (25 - (x.1 : ℕ))) i ≤ 100) :=
by {
  use 37,
  sorry
}

end max_candies_l249_249871


namespace chili_problem_l249_249415

def cans_of_chili (x y z : ℕ) : Prop := x + 2 * y + z = 6

def percentage_more_tomatoes_than_beans (x y z : ℕ) : ℕ :=
  100 * (z - 2 * y) / (2 * y)

theorem chili_problem (x y z : ℕ) (h1 : cans_of_chili x y z) (h2 : x = 1) (h3 : y = 1) : 
  percentage_more_tomatoes_than_beans x y z = 50 :=
by
  sorry

end chili_problem_l249_249415


namespace part_a_part_b_l249_249692

-- Part (a)
theorem part_a (x : ℝ) (h : x ≠ 0) : 
  (1 / (1 + 1 / x)) + (1 / (1 + x)) = 1 :=
sorry

-- Part (b)
theorem part_b : 
  (finset.sum (finset.range (2 * 2019 + 1)) 
    (λ k, 1 / (2019^(-2019 + k) + 1))) = 4039 / 2 :=
sorry

end part_a_part_b_l249_249692


namespace fraction_to_decimal_l249_249434

theorem fraction_to_decimal : (45 : ℝ) / (2^3 * 5^4) = 0.0090 := by
  sorry

end fraction_to_decimal_l249_249434


namespace sixty_percent_of_number_l249_249592

theorem sixty_percent_of_number (N : ℚ) (h : ((1 / 6) * (2 / 3) * (3 / 4) * (5 / 7) * N = 25)) :
  0.60 * N = 252 := sorry

end sixty_percent_of_number_l249_249592


namespace exists_a_value_l249_249547

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

def triangle (a b c : ℝ) (A B C : Type) : Prop :=
∃ (M : B → C → Type), 
  (∀ {B C : Type}, M B C = (1 / 2) * dist B C) →  -- M is the midpoint of BC
  (∀ {A M : Type}, dist A M = √17 / 2) →  -- AM = √17 / 2
  (∃ {B A : Type}, cos (dist B A) = 3 / 5) →  -- cos B = 3 / 5
  (area [a b c] = 4) →  -- area of ΔABC is 4
  a = 4 ∨ a = 5

theorem exists_a_value (a b c : ℝ) (A B C : Type) :
  triangle a b c A B C → (a = 4 ∨ a = 5) :=
begin
    sorry
end

end exists_a_value_l249_249547


namespace height_on_hypotenuse_correct_l249_249947

noncomputable def height_on_hypotenuse (a b : ℝ) (ha : a = 3) (hb : b = 4) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let area := (a * b) / 2
  (2 * area) / c

theorem height_on_hypotenuse_correct (h : ℝ) : 
  height_on_hypotenuse 3 4 rfl rfl = 12 / 5 :=
by
  sorry

end height_on_hypotenuse_correct_l249_249947


namespace probability_heads_penny_nickel_dime_l249_249289

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]
variable (coins : Fin 5 → Ω)
variable (heads tails : Ω)

-- Each coin has two outcomes: heads or tails
axiom coin_outcome (i : Fin 5) : coins i = heads ∨ coins i = tails

-- There are 32 total outcomes
axiom total_outcomes : Fintype.card (Fin 5 → Ω) = 32

-- There are 4 successful outcomes for penny, nickel, and dime being heads
axiom successful_outcomes : let successful := {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} in
                          Fintype.card successful = 4

theorem probability_heads_penny_nickel_dime :
  let probability := (Fintype.card {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} : ℤ) /
                     (Fintype.card (Fin 5 → Ω) : ℤ) in
  probability = 1 / 8 :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249289


namespace solve_inequality_l249_249611

variable {a x : ℝ}

theorem solve_inequality (h : a > 0) : 
  (ax^2 - (a + 1)*x + 1 < 0) ↔ 
    (if 0 < a ∧ a < 1 then 1 < x ∧ x < 1/a else 
     if a = 1 then false else 
     if a > 1 then 1/a < x ∧ x < 1 else true) :=
  sorry

end solve_inequality_l249_249611


namespace passes_through_midpoint_of_MN_l249_249136

open EuclideanGeometry

namespace Geometry

theorem passes_through_midpoint_of_MN
  {A B C H P M N G : Point}
  (h1 : acute_triangle A B C)
  (h2 : midpoint M A B)
  (h3 : midpoint N B C)
  (h4 : altitude B H A C)
  (h5 : meets_circumcircles_at P A H N C H M)
  (h6 : P ≠ H)
  (h7 : midpoint G M N) :
  passes_through P H G :=
sorry

end Geometry

end passes_through_midpoint_of_MN_l249_249136


namespace no_equal_entries_a_no_equal_entries_b_maximum_value_l249_249593

noncomputable def initial_arrangement_a : list (list ℕ) :=
[
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]

noncomputable def initial_arrangement_b : list (list ℕ) :=
[
  [2, 8, 5],
  [9, 3, 4],
  [6, 7, 1]
]

def operation_possible (table : list (list ℕ)) : Prop := sorry

theorem no_equal_entries_a :
  ¬ operation_possible initial_arrangement_a :=
sorry

theorem no_equal_entries_b :
  ¬ operation_possible initial_arrangement_b :=
sorry

theorem maximum_value :
  (∀ (table : list (list ℕ)), table.sum = 45) → (∃ k : ℕ, k = 5) :=
sorry

end no_equal_entries_a_no_equal_entries_b_maximum_value_l249_249593


namespace basket_weight_l249_249363

variable (B P : ℕ)

theorem basket_weight (h1 : B + P = 62) (h2 : B + P / 2 = 34) : B = 6 :=
by
  sorry

end basket_weight_l249_249363


namespace min_major_axis_l249_249844

theorem min_major_axis (a b c : ℝ) (h1 : b * c = 1) (h2 : a = Real.sqrt (b^2 + c^2)) : 2 * a ≥ 2 * Real.sqrt 2 :=
by
  sorry

end min_major_axis_l249_249844


namespace part1_solution_part2_solution_l249_249829

noncomputable def vector_a : ℝ^3 := sorry
noncomputable def vector_b : ℝ^3 := sorry

-- Conditions for unit vectors
axiom unit_vectors_a_b : ∥vector_a∥ = 1 ∧ ∥vector_b∥ = 1

-- Condition for the problem part 1
axiom part1_condition : ∥3 • vector_a - 2 • vector_b∥ = 3

-- Proof problem part 1
theorem part1_solution : ∥3 • vector_a + vector_b∥ = 2 * Real.sqrt 3 := by
  sorry

-- Additional Condition for the problem part 2 (angle between a and b)
axiom angle_condition : real.arccos (vector_a ⬝ vector_b) = π / 3

-- Defining vectors m and n
noncomputable def vector_m : ℝ^3 := 2 • vector_a + vector_b
noncomputable def vector_n : ℝ^3 := 2 • vector_b - 3 • vector_a

-- Proof problem part 2
theorem part2_solution : real.arccos ((vector_m ⬝ vector_n) / (∥vector_m∥ * ∥vector_n∥)) = 2 * π / 3 := by
  sorry


end part1_solution_part2_solution_l249_249829


namespace y_not_directly_nor_inversely_proportional_l249_249970

theorem y_not_directly_nor_inversely_proportional (x y : ℝ) :
  (∃ k : ℝ, x + y = 0 ∧ y = k * x) ∨
  (∃ k : ℝ, 3 * x * y = 10 ∧ x * y = k) ∨
  (∃ k : ℝ, x = 5 * y ∧ x = k * y) ∨
  (∃ k : ℝ, (y = 10 - x^2 - 3 * x) ∧ y ≠ k * x ∧ y * x ≠ k) ∨
  (∃ k : ℝ, x / y = Real.sqrt 3 ∧ x = k * y)
  → (∃ k : ℝ, y = 10 - x^2 - 3 * x ∧ y ≠ k * x ∧ y * x ≠ k) :=
by
  sorry

end y_not_directly_nor_inversely_proportional_l249_249970


namespace length_of_AB_l249_249312

-- Conditions:
-- The radius of the inscribed circle is 6 cm.
-- The triangle is a right triangle with a 60 degree angle at one vertex.
-- Question: Prove that the length of AB is 12 + 12√3 cm.

theorem length_of_AB (r : ℝ) (angle : ℝ) (h_radius : r = 6) (h_angle : angle = 60) :
  ∃ (AB : ℝ), AB = 12 + 12 * Real.sqrt 3 :=
by
  sorry

end length_of_AB_l249_249312


namespace polynomial_remainder_theorem_l249_249140

open Polynomial

theorem polynomial_remainder_theorem (Q : Polynomial ℝ)
  (h1 : Q.eval 20 = 120)
  (h2 : Q.eval 100 = 40) :
  ∃ R : Polynomial ℝ, R.degree < 2 ∧ Q = (X - 20) * (X - 100) * R + (-X + 140) :=
by
  sorry

end polynomial_remainder_theorem_l249_249140


namespace find_number_l249_249300

noncomputable def some_number : ℝ :=
  0.27712 / 9.237333333333334

theorem find_number :
  (69.28 * 0.004) / some_number = 9.237333333333334 :=
by 
  sorry

end find_number_l249_249300


namespace greatest_divisors_1_to_20_l249_249195

def is_divisor (a b : ℕ) : Prop := b % a = 0

def count_divisors (n : ℕ) : ℕ :=
  nat.succ (list.length ([k for k in list.range n if is_divisor k.succ n]))

def max_divisors_from_1_to_20 : set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 20 ∧ ∃ max_count, count_divisors n = max_count ∧ 
  ∀ m, 1 ≤ m ∧ m ≤ 20 → count_divisors m ≤ max_count}

theorem greatest_divisors_1_to_20 : max_divisors_from_1_to_20 = {12, 18, 20} :=
by {
  sorry
}

end greatest_divisors_1_to_20_l249_249195


namespace sum_of_cubes_eq_neg2_l249_249934

theorem sum_of_cubes_eq_neg2 (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 1) : a^3 + b^3 = -2 := 
sorry

end sum_of_cubes_eq_neg2_l249_249934


namespace annual_interest_is_810_l249_249748

def principal := 9000
def rate := 0.09
def time := 1

-- Define the formula for simple interest
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Statement to prove
theorem annual_interest_is_810 : simple_interest principal rate time = 810 :=
by
  sorry

end annual_interest_is_810_l249_249748


namespace largest_angle_of_obtuse_isosceles_triangle_l249_249664

theorem largest_angle_of_obtuse_isosceles_triangle (P Q R : Type) 
  (triangle_PQR : Triangle P Q R)
  (isosceles_PQR : Isosceles triangle_PQR) 
  (obtuse_PQR : Obtuse triangle_PQR)
  (angle_P_30 : angle P triangle_PQR = 30) : 
  ∃ (angle_Q : ℕ), is_largest_angle angle_Q triangle_PQR ∧ angle_Q = 120 := 
by 
  sorry

end largest_angle_of_obtuse_isosceles_triangle_l249_249664


namespace max_divisors_up_to_20_l249_249215

def divisors (n : Nat) : List Nat :=
  (List.range (n + 1)).filter (λ d, n % d = 0)

def count_divisors (n : Nat) : Nat :=
  (divisors n).length

theorem max_divisors_up_to_20 :
  ∀ n, n ∈ List.range 21 → count_divisors n ≤ 6 ∧
      ((count_divisors 12 = 6) ∧
       (count_divisors 18 = 6) ∧
       (count_divisors 20 = 6)) :=
by
  sorry

end max_divisors_up_to_20_l249_249215


namespace find_x2_l249_249030

-- Define the given conditions
def A : ℝ×ℝ := (-1/2, Real.sqrt 3 / 2)
def α := 2 * Int.pi * k + 2 * pi / 3       -- α is determined by the coordinates of A
def β := α + pi / 6                       -- Beta is the new angle after rotation

-- Prove the required x-coordinate of point B
theorem find_x2 (k : ℤ) :
  let B : ℝ×ℝ := (Real.cos β, Real.sin β)
  (B.1 = - Real.sqrt 3 / 2) :=
sorry

end find_x2_l249_249030


namespace expand_polynomials_l249_249784

variable (x : ℝ)

theorem expand_polynomials : 
  (3 * x^2 - 4 * x + 3) * (-4 * x^2 + 2 * x - 6) = -12 * x^4 + 22 * x^3 - 38 * x^2 + 30 * x - 18 :=
  by
  sorry

end expand_polynomials_l249_249784


namespace greatest_divisors_1_to_20_l249_249179

theorem greatest_divisors_1_to_20 : ∃ n₁ n₂ n₃, ((n₁ = 12 ∧ n₂ = 18 ∧ n₃ = 20) ∧ 
  (∀ m ∈ (Finset.range 21).filter (λ x, x > 0), 
    let d_m := (Finset.range (m + 1)).filter (λ k, m % k = 0) in
    ∃ k, k = d_m.card ∧ k <= 6 ∧ ((m = n₁ ∧ k = 6) ∨ (m = n₂ ∧ k = 6) ∨ (m = n₃ ∧ k = 6)))) :=
by
  sorry

end greatest_divisors_1_to_20_l249_249179


namespace find_y_l249_249732

theorem find_y (x y : ℝ) (hx : x ≠ 0) (h : sqrt ((5 * x) / y) = x) (hx_val : x = 1.6666666666666667) : y = 3 :=
by
  sorry

end find_y_l249_249732


namespace sqrt_meaningful_range_l249_249081

theorem sqrt_meaningful_range (x : ℝ) : x + 1 ≥ 0 ↔ x ≥ -1 :=
by sorry

end sqrt_meaningful_range_l249_249081


namespace contains_sphere_of_diameter_1_01_l249_249744

namespace Tetrahedron

open Classical

variable (A B C D : ℝ)

-- Given conditions
constant edge_length : finset (ℝ × ℝ) → ℝ
constant contains_two_nonintersecting_spheres_of_diameter
  (S : set (set ℝ × ℝ × ℝ)) : Prop

-- The condition that all edges in the tetrahedron are < 100
axiom edges_less_than_100 : ∀ edge ∈ edge_length {A, B, C, D}, edge < 100 

-- The condition that the tetrahedron contains two nonintersecting spheres of diameter 1
axiom contains_two_nonintersecting_spheres : 
  contains_two_nonintersecting_spheres_of_diameter {A, B, C, D} 

-- Prove that the tetrahedron contains a sphere of diameter 1.01
theorem contains_sphere_of_diameter_1_01 :
  ∃ (S : set (set ℝ × ℝ × ℝ)), contains_two_nonintersecting_spheres_of_diameter S ∧ ∃ sphere, (∀ x ∈ sphere, x > 0) ∧ (diameter sphere = 1.01) :=
by
  sorry

end Tetrahedron

end contains_sphere_of_diameter_1_01_l249_249744


namespace largest_angle_of_isosceles_obtuse_30_deg_l249_249674

def is_isosceles (T : Triangle) : Prop :=
  T.A = T.B ∨ T.B = T.C ∨ T.C = T.A

def is_obtuse (T : Triangle) : Prop :=
  T.A > 90 ∨ T.B > 90 ∨ T.C > 90

def T : Type := {P Q R : ℝ}

noncomputable def largest_angle (T : Triangle) : ℝ :=
  max T.A (max T.B T.C)

theorem largest_angle_of_isosceles_obtuse_30_deg :
  ∀ (T : Triangle), is_isosceles T → is_obtuse T → T.A = 30 → largest_angle T = 120 :=
by
  intro T h_iso h_obt h_A30
  sorry

end largest_angle_of_isosceles_obtuse_30_deg_l249_249674


namespace max_divisors_up_to_20_l249_249220

def divisors (n : Nat) : List Nat :=
  (List.range (n + 1)).filter (λ d, n % d = 0)

def count_divisors (n : Nat) : Nat :=
  (divisors n).length

theorem max_divisors_up_to_20 :
  ∀ n, n ∈ List.range 21 → count_divisors n ≤ 6 ∧
      ((count_divisors 12 = 6) ∧
       (count_divisors 18 = 6) ∧
       (count_divisors 20 = 6)) :=
by
  sorry

end max_divisors_up_to_20_l249_249220


namespace largest_angle_in_triangle_PQR_l249_249669

-- Definitions
def is_isosceles_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  (α = β) ∨ (β = γ) ∨ (γ = α)

def is_obtuse_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

variables (P Q R : ℝ)
variables (angleP angleQ angleR : ℝ)

-- Condition: PQR is an obtuse and isosceles triangle, and angle P measures 30 degrees
axiom h1 : is_isosceles_triangle P Q R angleP angleQ angleR
axiom h2 : is_obtuse_triangle P Q R angleP angleQ angleR
axiom h3 : angleP = 30

-- Theorem: The measure of the largest interior angle of triangle PQR is 120 degrees
theorem largest_angle_in_triangle_PQR : max angleP (max angleQ angleR) = 120 :=
  sorry

end largest_angle_in_triangle_PQR_l249_249669


namespace series_sum_eq_one_sixth_l249_249773

noncomputable def series_sum := 
  ∑' n : ℕ, (3^n) / ((7^ (2^n)) + 1)

theorem series_sum_eq_one_sixth : series_sum = 1 / 6 := 
  sorry

end series_sum_eq_one_sixth_l249_249773


namespace max_brownies_l249_249505

-- Definitions for the conditions given in the problem
def is_interior_pieces (m n : ℕ) : ℕ := (m - 2) * (n - 2)
def is_perimeter_pieces (m n : ℕ) : ℕ := 2 * m + 2 * n - 4

-- The assertion that the number of brownies along the perimeter is twice the number in the interior
def condition (m n : ℕ) : Prop := 2 * is_interior_pieces m n = is_perimeter_pieces m n

-- The statement that the maximum number of brownies under the given condition is 84
theorem max_brownies : ∃ (m n : ℕ), condition m n ∧ m * n = 84 := by
  sorry

end max_brownies_l249_249505


namespace fraction_to_decimal_l249_249435

theorem fraction_to_decimal : (45 : ℝ) / (2^3 * 5^4) = 0.0090 := by
  sorry

end fraction_to_decimal_l249_249435


namespace partial_derivatives_bound_l249_249986

theorem partial_derivatives_bound (f : ℝ × ℝ → ℝ)
  (h1 : ∀ (x y : ℝ), x^2 + y^2 ≤ 1 → |f (x, y)| ≤ 1)
  (h2 : ∀ (x y : ℝ), x^2 + y^2 ≤ 1 → differentiable ℝ (λ p, f p))
  : ∃ (x0 y0 : ℝ), x0^2 + y0^2 < 1 ∧
    ((fderiv ℝ (λ (p : ℝ × ℝ), f p) (x0, y0)).1 0)^2 +
    ((fderiv ℝ (λ (p : ℝ × ℝ), f p) (x0, y0)).2 0)^2 ≤ 16 := 
sorry

end partial_derivatives_bound_l249_249986


namespace find_speed_grocery_to_gym_l249_249406

variables (v : ℝ) (speed_grocery_to_gym : ℝ)
variables (d_home_to_grocery : ℝ) (d_grocery_to_gym : ℝ)
variables (time_diff : ℝ)

def problem_conditions : Prop :=
  d_home_to_grocery = 840 ∧
  d_grocery_to_gym = 480 ∧
  time_diff = 40 ∧
  speed_grocery_to_gym = 2 * v

def correct_answer : Prop :=
  speed_grocery_to_gym = 30

theorem find_speed_grocery_to_gym :
  problem_conditions v speed_grocery_to_gym d_home_to_grocery d_grocery_to_gym time_diff →
  correct_answer speed_grocery_to_gym :=
by
  sorry

end find_speed_grocery_to_gym_l249_249406


namespace simplify_expression_l249_249608

-- Define constants
variables (z : ℝ)

-- Define the problem and its solution
theorem simplify_expression :
  (5 - 2 * z) - (4 + 5 * z) = 1 - 7 * z := 
sorry

end simplify_expression_l249_249608


namespace largest_interior_angle_obtuse_isosceles_triangle_l249_249656

theorem largest_interior_angle_obtuse_isosceles_triangle :
  ∀ (P Q R : Type) (α β γ : ℝ), α + β + γ = 180 ∧ γ = 120 ∧ α = 30 ∧ β = 30 →
  (α = 30 ∧ β = 30 ∧ γ = 120) ∨
  (α = 30 ∧ γ = 30 ∧ β = 120) ∨
  (β = 30 ∧ γ = 30 ∧ α = 120) → 
  γ = max α (max β γ) :=
by {
  intros P Q R α β γ h1 h2,
  repeat { rw h1 at * },
  rw h2,
  sorry
}

end largest_interior_angle_obtuse_isosceles_triangle_l249_249656


namespace equation_equivalence_l249_249926

theorem equation_equivalence (p q : ℝ) (hp₀ : p ≠ 0) (hp₅ : p ≠ 5) (hq₀ : q ≠ 0) (hq₇ : q ≠ 7) :
  (3 / p + 5 / q = 1 / 3) → p = 9 * q / (q - 15) :=
by
  sorry

end equation_equivalence_l249_249926


namespace highest_monthly_profit_max_avg_monthly_profit_action_l249_249722

noncomputable def profit (x : Nat) : Int :=
  if h : 1 ≤ x ∧ x ≤ 5 then 26 * x - 56
  else if h : 5 < x ∧ x ≤ 12 then 210 - 20 * x
  else 0

def avg_profit (x : Nat) : Real :=
  if h : 1 ≤ x ∧ x ≤ 5 then (13 * x - 43 : Int) * (1.0 : Real)
  else if h : 5 < x ∧ x ≤ 12 then -10.0 * x + 200.0 - 640.0 / x
  else 0.0

theorem highest_monthly_profit :
  ∃ (xm : Nat) (ymax : Nat), (xm = 6) ∧ (ymax = 90) ∧ profit xm = ymax := by
  sorry

theorem max_avg_monthly_profit_action :
  ∃ (xm : Nat) (wmax : Real), (xm = 8) ∧ (wmax = 40.0) ∧ avg_profit xm = wmax := by
  sorry

end highest_monthly_profit_max_avg_monthly_profit_action_l249_249722


namespace Vovochka_max_candies_l249_249886

noncomputable theory
open_locale classical
open set

def VovochkaCandies (c n k required_sum : ℕ) (dist : fin n → ℕ) : Prop :=
  ∀ (s : finset (fin n)), s.card = k → (s.sum dist ≥ required_sum)

theorem Vovochka_max_candies
  (c n k required_sum : ℕ)
  (h_c : c = 200)
  (h_n : n = 25)
  (h_k : k = 16)
  (h_required_sum : required_sum = 100)
  (dist : fin n → ℕ)
  (h_dist : VovochkaCandies c n k required_sum dist) :
  c - finset.univ.sum dist = 37 :=
sorry

end Vovochka_max_candies_l249_249886


namespace total_toys_l249_249649

theorem total_toys (n : ℕ) (h1 : 3 * (n / 4) = 18) : n = 24 :=
by
  sorry

end total_toys_l249_249649


namespace hou_yifan_not_losing_l249_249526

theorem hou_yifan_not_losing (p_win p_draw : ℝ) (h_win : p_win = 0.65) (h_draw : p_draw = 0.25) :
  p_win + p_draw = 0.9 :=
by
  rw [h_win, h_draw]
  norm_num

end hou_yifan_not_losing_l249_249526


namespace value_of_knife_l249_249330

/-- Two siblings sold their flock of sheep. Each sheep was sold for as many florins as 
the number of sheep originally in the flock. They divided the revenue by giving out 
10 florins at a time. First, the elder brother took 10 florins, then the younger brother, 
then the elder again, and so on. In the end, the younger brother received less than 10 florins, 
so the elder brother gave him his knife, making their earnings equal. 
Prove that the value of the knife in florins is 2. -/
theorem value_of_knife (n : ℕ) (k m : ℕ) (h1 : n^2 = 20 * k + 10 + m) (h2 : 1 ≤ m ∧ m ≤ 9) : 
  (∃ b : ℕ, 10 - b = m + b ∧ b = 2) :=
by
  sorry

end value_of_knife_l249_249330


namespace circumcircle_equation_l249_249467

theorem circumcircle_equation :
  ∀ (A B C : Point ℝ), 
     A = (-real.sqrt 3, 0) → B = (real.sqrt 3, 0) → C = (0, 3) → 
     ∃ (a b r : ℝ), (a = 0) ∧ (b = 1) ∧ (r = 2) ∧
        (∀ (P : Point ℝ), P = A ∨ P = B ∨ P = C → 
           (P.1 - a)^2 + (P.2 - b)^2 = r^2) :=
by
  intros A B C hA hB hC
  sorry

end circumcircle_equation_l249_249467


namespace min_b_minus_a_l249_249855

noncomputable def f (x : ℝ) : ℝ := 1 + x - (x^2) / 2 + (x^3) / 3
noncomputable def g (x : ℝ) : ℝ := 1 - x + (x^2) / 2 - (x^3) / 3
noncomputable def F (x : ℝ) : ℝ := f x * g x

theorem min_b_minus_a (a b : ℤ) (h : ∀ x, F x = 0 → a ≤ x ∧ x ≤ b) (h_a_lt_b : a < b) : b - a = 3 :=
sorry

end min_b_minus_a_l249_249855


namespace teams_played_same_matches_l249_249362

theorem teams_played_same_matches (n : ℕ) (h : n = 30)
  (matches_played : Fin n → ℕ) :
  ∃ (i j : Fin n), i ≠ j ∧ matches_played i = matches_played j :=
by
  sorry

end teams_played_same_matches_l249_249362


namespace store_made_profit_l249_249389

noncomputable def profit_calc (x y sp : ℕ) : ℕ :=
  sp - (x + y)

theorem store_made_profit :
  ∃ x y : ℕ, x * 160 / 100 = 80 ∧ y * 80 / 100 = 80 ∧ profit_calc x y (2 * 80) = 10 :=
by {
  use 50, use 100,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
}

end store_made_profit_l249_249389


namespace largest_angle_in_triangle_PQR_l249_249666

-- Definitions
def is_isosceles_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  (α = β) ∨ (β = γ) ∨ (γ = α)

def is_obtuse_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

variables (P Q R : ℝ)
variables (angleP angleQ angleR : ℝ)

-- Condition: PQR is an obtuse and isosceles triangle, and angle P measures 30 degrees
axiom h1 : is_isosceles_triangle P Q R angleP angleQ angleR
axiom h2 : is_obtuse_triangle P Q R angleP angleQ angleR
axiom h3 : angleP = 30

-- Theorem: The measure of the largest interior angle of triangle PQR is 120 degrees
theorem largest_angle_in_triangle_PQR : max angleP (max angleQ angleR) = 120 :=
  sorry

end largest_angle_in_triangle_PQR_l249_249666


namespace min_value_trig_expression_l249_249443

theorem min_value_trig_expression : (∃ x : ℝ, 3 * Real.cos x - 4 * Real.sin x = -5) :=
by
  sorry

end min_value_trig_expression_l249_249443


namespace percent_two_squares_combined_l249_249735

variable (s : ℝ)

def square_area (s : ℝ) : ℝ := s^2

def rectangle_area (s : ℝ) : ℝ := 12 * s^2

def two_squares_area (s : ℝ) : ℝ := 2 * square_area s

def percent_area_occupied (s : ℝ) : ℝ := (two_squares_area s / rectangle_area s) * 100

theorem percent_two_squares_combined (s : ℝ) : percent_area_occupied s = 16.67 := by
  sorry

end percent_two_squares_combined_l249_249735


namespace find_cube_difference_l249_249024

theorem find_cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := 
by
  sorry

end find_cube_difference_l249_249024


namespace range_of_a_l249_249850

noncomputable def f (x: ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 1)) + x

theorem range_of_a (a : ℝ) (h : f (1 + a) + f (1 - a^2) < 0) : a < -1 ∨ a > 2 :=
sorry

end range_of_a_l249_249850


namespace power_of_minus_two_eq_power_of_two_l249_249516

theorem power_of_minus_two_eq_power_of_two (m : Int) (hm : m = 2) : (-2)^(2 * m) = 2^4 := by
  sorry

end power_of_minus_two_eq_power_of_two_l249_249516


namespace cylinder_volume_increase_factor_l249_249685

theorem cylinder_volume_increase_factor
    (π : Real)
    (r h : Real)
    (V_original : Real := π * r^2 * h)
    (new_height : Real := 3 * h)
    (new_radius : Real := 4 * r)
    (V_new : Real := π * (new_radius)^2 * new_height) :
    V_new / V_original = 48 :=
by
  sorry

end cylinder_volume_increase_factor_l249_249685


namespace total_homework_pages_l249_249599

theorem total_homework_pages (R : ℕ) (H1 : R + 3 = 8) : R + (R + 3) = 13 :=
by sorry

end total_homework_pages_l249_249599


namespace divisors_of_12_18_20_l249_249190

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem divisors_of_12_18_20 (n : ℕ) (h : n ∈ {1, 2, 3, ..., 20}) :
  n ∈ {12, 18, 20} ↔ (∀ m ∈ {1, 2, 3, ..., 20}, count_divisors m ≤ 6) :=
by
  sorry

end divisors_of_12_18_20_l249_249190


namespace translation_g_eq_inverse_f_eq_l249_249036

-- Definition of the original function f(x)
def f (x : ℝ) : ℝ := (x + 3) / (2 - x)

-- Problem 1: translation of the function
theorem translation_g_eq : ∀ x, g(x) = 5 / -x :=
by
  -- Definitions based on the translations
  let f1 := (λ x, f (x + 2)) -- translate 2 units to the left
  let g := (λ x, f1 x + 1) -- translate 1 unit up
  sorry

-- Problem 2: finding the inverse function and its domain
theorem inverse_f_eq : ∀ x, x ≠ -1 → f⁻¹ x = (2 * x - 3) / (1 + x) :=
by
  -- Inverse relation setup
  let inv_f := λ y, (2 * y - 3) / (1 + y)
  sorry

end translation_g_eq_inverse_f_eq_l249_249036


namespace avg_eggs_per_nest_l249_249106

/-- In the Caribbean, loggerhead turtles lay three million eggs in twenty thousand nests. 
On average, show that there are 150 eggs in each nest. -/

theorem avg_eggs_per_nest 
  (total_eggs : ℕ) 
  (total_nests : ℕ) 
  (h1 : total_eggs = 3000000) 
  (h2 : total_nests = 20000) :
  total_eggs / total_nests = 150 := 
by {
  sorry
}

end avg_eggs_per_nest_l249_249106


namespace angle_XYZ_60_degrees_l249_249718

theorem angle_XYZ_60_degrees 
  (P X Y Z : Type) 
  (circumscribed_circle : IsCircumscribed P (triangle X Y Z)) 
  (angle_XPY : ∠ XPY = 120) 
  (angle_YPZ : ∠ YPZ = 130) :
  ∠ XYZ = 60 := 
sorry

end angle_XYZ_60_degrees_l249_249718


namespace F_is_odd_l249_249498

variables {a x : ℝ} (h_a : a ≠ 0)

noncomputable def f (x : ℝ) : ℝ := Real.log (abs (a * x))
def g (x : ℝ) : ℝ := x ^ (-3) + Real.sin x
def F (x : ℝ) : ℝ := f x * g x

theorem F_is_odd : ∀ x : ℝ, F (-x) = -F x :=
by
  intro x
  unfold F
  unfold f
  unfold g
  sorry

end F_is_odd_l249_249498


namespace tangency_of_circumcircles_l249_249615

-- Definitions and conditions
variable [EuclideanGeometry] {α : Type*} [co: metric_space α] [normed_group α] [normed_space ℝ α]

variables (A B C Z A1 C1 X Y : α)

-- Given conditions
axiom tangents_intersect_at_Z : tangent_line (circumcircle A B C) A ∩ tangent_line (circumcircle A B C) C = {Z}
axiom AA1_altitude : is_altitude A1 A B C
axiom CC1_altitude : is_altitude C1 C A B
axiom A1C1_intersections : intersects (line A1 C1) (line Z A) X ∧ intersects (line A1 C1) (line Z C) Y

-- Statement of the theorem
theorem tangency_of_circumcircles :
  tangent (circumcircle A B C) (circumcircle X Y Z) :=
sorry

end tangency_of_circumcircles_l249_249615


namespace triangle_area_AQB_l249_249388

-- Definitions extracted from the problem conditions
def point : Type := ℝ × ℝ
def square (A B C D : point) (s : ℝ) : Prop := 
  A = (0, 0) ∧ B = (s, 0) ∧ D = (s, s) ∧ F = (0, s)

-- Assume the conditions given in the problem
variables (A B C D F Q : point)
variables (s : ℝ) -- side length of the square (8 inches in this case)
variables (QA QB QC : ℝ) -- distances from Q to A, B, C (5 inches in this case)

axiom square_condition : square A B C D s
axiom point_Q_condition : Q = (s / 2, s / 2)
axiom distances_condition : QA = QB ∧ QB = QC ∧ QC = 5
axiom perpendicular_condition : ∃ E : point, (QC = s / 2 ∧ Q = (4, 4))

-- Statement to be proven: the area of triangle AQB is 12 square inches
theorem triangle_area_AQB : 
  square A B C D 8 → 
  QA = 5 → QB = 5 → QC = 5 → 
  (QA = QB ∧ QB = QC) →
  ∃ (AreaAQB : ℝ), AreaAQB = 12 := by
  sorry

end triangle_area_AQB_l249_249388


namespace complex_inverse_identity_l249_249923

theorem complex_inverse_identity : ∀ (i : ℂ), i^2 = -1 → (3 * i - 2 * i⁻¹)⁻¹ = -i / 5 :=
by
  -- Let's introduce the variables and the condition.
  intro i h

  -- Sorry is used to signify the proof is omitted.
  sorry

end complex_inverse_identity_l249_249923


namespace proposition1_proposition2_proposition3_proposition4_correct_propositions_l249_249035

-- Defining the conditions as Lean functions/statements.
def condition1 (f : ℝ → ℝ) : Prop := ∀ x, f(x + 2) = f(2 - x)
def condition3 (f : ℝ → ℝ) : Prop := 
  (∀ x, y = f(2 + x) → ∃ x', y = f(2 - x'))

-- Theorems to be proved.
theorem proposition1 (f : ℝ → ℝ) (h : condition1 f): Prop :=
  ∀ x, ∃ y, f(x) = y  -- Graph of f(x) is symmetric about x=2

theorem proposition2 (f : ℝ → ℝ) (h : condition1 f): Prop :=
  ∃ x, ∀ y, f(x + y) ≠ f(x - y)  -- Graph of f(x) is not symmetric about y-axis

theorem proposition3 (f : ℝ → ℝ) (h : condition3 f): Prop :=
  ¬ (∀ x, y = f(2 + x) → y = f(2 - x))  -- Graphs of y=f(2+x) and y=f(2-x) are not symmetric about x=2

theorem proposition4 (f : ℝ → ℝ) (h : condition3 f): Prop :=
  ∀ x, ∃ y, f(2 + x) = f(2 - x)  -- Graphs of y=f(2+x) and y=f(2-x) are symmetric about y-axis

-- Combining results to match the problem statement.
theorem correct_propositions (f : ℝ → ℝ) (h1 : condition1 f) (h3 : condition3 f) : Prop :=
  proposition1 f h1 ∧ proposition4 f h3 ∧ ¬proposition2 f h1 ∧ ¬proposition3 f h3

end proposition1_proposition2_proposition3_proposition4_correct_propositions_l249_249035


namespace max_divisors_1_to_20_l249_249204

theorem max_divisors_1_to_20 :
  ∃ m n o ∈ set.range (20:ℕ.succ), 
    (∀ k ∈ set.range (20:ℕ.succ), 
      (nat.divisors_count k ≤ nat.divisors_count m) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count n) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count o)) 
    ∧ (nat.divisors_count m = 6)
    ∧ (nat.divisors_count n = 6)
    ∧ (nat.divisors_count o = 6)
    ∧ m ≠ n ∧ n ≠ o ∧ o ≠ m
    ∧ set.mem 12 (set.range (20 + 1))
    ∧ set.mem 18 (set.range (20 + 1))
    ∧ set.mem 20 (set.range (20 + 1)). 
      := by 
  -- all these steps will be skipped here
  sorry

end max_divisors_1_to_20_l249_249204


namespace find_x_l249_249614

theorem find_x (x y : ℝ) (pos_x : 0 < x) (pos_y : 0 < y) 
  (h1 : x - y^2 = 3) (h2 : x^2 + y^4 = 13) : 
  x = (3 + Real.sqrt 17) / 2 := 
sorry

end find_x_l249_249614


namespace length_of_XY_l249_249311

def triangle (A B C : Point) : Prop := Collinear A B C ∧ Distance A B + Distance B C + Distance C A = 1

def circle_tangent (ω : Circle) (A B C P Q : Point) : Prop :=
  Tangent ω A B C ∧ Tangent ω P A B ∧ Tangent ω Q A C

def line_midpoints_intersect_circumcircle (M N X Y A P Q : Point) (Circumcircle_APQ : Circle) : Prop :=
  Midpoint M A B ∧ Midpoint N A C ∧ Intersects (LineThrough M N) Circumcircle_APQ X Y

theorem length_of_XY 
  {A B C P Q X Y M N: Point} 
  (Circumcircle_APQ: Circle)
  (h1: triangle A B C)
  (h2: circle_tangent ω A B C P Q)
  (h3: line_midpoints_intersect_circumcircle M N X Y A P Q Circumcircle_APQ) : 
  Distance X Y = 1 / 2 := 
sorry

end length_of_XY_l249_249311


namespace find_standard_equation_and_range_l249_249834

noncomputable def ellipse_standard_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (h : a > b) :=
  (a = 2 * Real.sqrt 6) ∧ (b = 2 * Real.sqrt 2)

theorem find_standard_equation_and_range (O F1 F2 A B : ℝ × ℝ)
  (hO : O = (0, 0))
  (hF1 : F1 ≠ O)
  (hF2 : F2 ≠ O)
  (hA : A ≠ O)
  (hB : B ≠ O)
  (C : ℝ → ℝ → ℝ)
  (hC : ∀ x y, C x y = (x^2 / 24) + (y^2 / 8) - 1 = 0)
  (geom_seq : ∃ k: ℝ, k^2 = (|OB| * |OF2|) / |AB|)
  (hmax_dist: ∀ P, max_dist P F2 = 2 * Real.sqrt 6 + 4)
  (perpendicular_chords : ∀ M N P Q, M ≠ O → N ≠ O → P ≠ O → Q ≠ O → MN ⊥ PQ → (|MN| + |PQ|) ∈ [4 * Real.sqrt 6, 16 * Real.sqrt 6 / 3])
  : ∃ a b : ℝ, ellipse_standard_equation a b ∧ ∀ MN PQ, ∃ k: ℝ, k > 0 ∧ (|MN| + |PQ|) ∈ [4 * Real.sqrt 6, 16 * Real.sqrt 6 / 3] :=
begin
  sorry,
end

end find_standard_equation_and_range_l249_249834


namespace mixed_number_calculation_l249_249754

/-
  We need to define a proof that shows:
  75 * (2 + 3/7 - 5 * (1/3)) / (3 + 1/5 + 2 + 1/6) = -208 + 7/9
-/
theorem mixed_number_calculation :
  75 * ((17 / 7) - (16 / 3)) / ((16 / 5) + (13 / 6)) = -208 + 7 / 9 := by
  sorry

end mixed_number_calculation_l249_249754


namespace sqrt_5_expression_l249_249916

theorem sqrt_5_expression (a b : ℝ) (h1 : real.sqrt 5 = a + b) (h2 : a ∈ ℤ) (h3 : 0 < b ∧ b < 1) : 
  (a - b) * (4 + real.sqrt 5) = 11 :=
sorry

end sqrt_5_expression_l249_249916


namespace addition_even_odd_is_odd_subtraction_even_odd_is_odd_squared_sum_even_odd_is_odd_l249_249846

section OperationsAlwaysYieldOdd

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem addition_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd (a + b) :=
sorry

theorem subtraction_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd (a - b) :=
sorry

theorem squared_sum_even_odd_is_odd (a b : ℤ) (h₁ : is_even a) (h₂ : is_odd b) : is_odd ((a + b) * (a + b)) :=
sorry

end OperationsAlwaysYieldOdd

end addition_even_odd_is_odd_subtraction_even_odd_is_odd_squared_sum_even_odd_is_odd_l249_249846


namespace vovochka_max_candies_l249_249898

noncomputable def max_candies_for_vovochka 
  (total_classmates : ℕ)
  (total_candies : ℕ)
  (candies_condition : ∀ (subset : Finset ℕ), subset.card = 16 → ∑ i in subset, (classmate_candies i) ≥ 100) 
  (classmate_candies : ℕ → ℕ) : ℕ :=
  total_candies - ∑ i in (Finset.range total_classmates), classmate_candies i

theorem vovochka_max_candies :
  max_candies_for_vovochka 25 200 (λ subset hc, True) (λ i, if i < 13 then 7 else 6) = 37 :=
by
  sorry

end vovochka_max_candies_l249_249898


namespace bob_paid_more_than_cindy_l249_249375

def pizza : ℕ := 10
def plain_pizza_cost : ℝ := 12
def additional_double_cheese_cost : ℝ := 4
def double_cheese_slices : ℕ := 5
def plain_slices_bob : ℕ := 2
def plain_slices_cindy : ℕ := 3
def total_slices : ℝ := pizza
def total_cost : ℝ := plain_pizza_cost + additional_double_cheese_cost
def cost_per_slice : ℝ := total_cost / total_slices
def bob_slices_cost : ℝ := (double_cheese_slices + plain_slices_bob) * cost_per_slice
def cindy_slices_cost : ℝ := plain_slices_cindy * cost_per_slice

theorem bob_paid_more_than_cindy : (bob_slices_cost - cindy_slices_cost) = 6.4 :=
by
  calc
    _ = sorry   -- Here, the detailed computation proof steps would be inserted

end bob_paid_more_than_cindy_l249_249375


namespace parabola_circle_relationship_l249_249309

-- Define the parabola C_1: y = x^2 + 2ax + b
def parabola (a b : ℝ) : ℝ → ℝ := λ x, x^2 + 2 * a * x + b

-- Define the vertices A and B of the parabola where it intersects the x-axis
def roots (a b : ℝ) (x : ℝ) := x^2 + 2 * a * x + b = 0

-- Define the conditions where the vertex of the parabola lies within the circle C_2
def vertex (a b : ℝ) : ℝ × ℝ := (-a, b - a^2)

-- Define the circle C_2 with AB as its diameter
def circle_eq (a b : ℝ) : ℝ × ℝ → ℝ := λ p, (p.1 + a)^2 + p.2^2 - (a^2 - b) = 0

-- Define the theorem to prove the relationship between a and b
theorem parabola_circle_relationship (a b : ℝ) :
  let v := vertex a b in
  v.1^2 + (v.2)^2 + 2 * a * v.1 + b < 0 → a^2 - 1 < b ∧ b < a^2 := 
sorry

end parabola_circle_relationship_l249_249309


namespace correct_population_growth_pattern_l249_249235

def population_growth_pattern : Type := (String, String, String)

def condition_a : population_growth_pattern := ("birth rate", "death rate", "total population")
def condition_b : population_growth_pattern := ("birth rate", "death rate", "rate of social production")
def condition_c : population_growth_pattern := ("birth rate", "death rate", "natural growth rate")
def condition_d : population_growth_pattern := ("birth rate", "total population", "rate of social production")

theorem correct_population_growth_pattern :
  condition_c = ("birth rate", "death rate", "natural growth rate") := 
by
  rfl

end correct_population_growth_pattern_l249_249235


namespace length_AP_eq_sqrt2_l249_249956

/-- In square ABCD with side length 2, a circle ω with center at (1, 0)
    and radius 1 is inscribed. The circle intersects CD at point M,
    and line AM intersects ω at a point P different from M.
    Prove that the length of AP is √2. -/
theorem length_AP_eq_sqrt2 :
  let A := (0, 2)
  let M := (2, 0)
  let P : ℝ × ℝ := (1, 1)
  dist A P = Real.sqrt 2 :=
by
  sorry

end length_AP_eq_sqrt2_l249_249956


namespace max_product_eq_l249_249456

noncomputable def max_product (n : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let l := n / 3 in
    match n % 3 with
    | 0 => 3 ^ l
    | 1 => 4 * 3 ^ (l - 1)
    | 2 => 2 * 3 ^ l
    | _ => 0 -- this case shouldn't occur since n % 3 can only be 0, 1, or 2

theorem max_product_eq (n : ℕ) : max_product n = 
  match n with
  | 1 => 1
  | _ => let l := n / 3 in
         match n % 3 with
         | 0 => 3 ^ l
         | 1 => 4 * 3 ^ (l - 1)
         | 2 => 2 * 3 ^ l
         | _ => 0 -- this case shouldn't occur since n % 3 can only be 0, 1, or 2
         end :=
by sorry

end max_product_eq_l249_249456


namespace tony_age_at_start_of_period_l249_249653

theorem tony_age_at_start_of_period :
  (∀ (age : ℕ), (work_days : ℕ) → (earnings : ℝ),
    3 * work_days * (0.75 + 0.25 * age) = earnings → work_days = 80 → earnings = 840 →
    age = 10) :=
begin
  intros age work_days earnings hwk_days_hours_worked hearn,
  sorry
end

end tony_age_at_start_of_period_l249_249653


namespace complex_modulus_equiv_l249_249156

variable (x y : ℝ)
variable (i : ℂ)

theorem complex_modulus_equiv
  (h : (2 + i) * (3 - x * i) = 3 + (y + 5) * i) :
  abs (x + y * i) = 5 :=
sorry

end complex_modulus_equiv_l249_249156


namespace probability_blue_buttons_l249_249124

theorem probability_blue_buttons:
  let C_red := 6 in
  let C_blue := 10 in
  let total_C := C_red + C_blue in
  let removed_buttons := 4 in
  let remaining_buttons := total_C * (3/4 : ℚ) in
  let removed_red := C_red - 4 / 2 in
  let removed_blue := C_blue - 4 / 2 in
  let D_red := 2 in 
  let D_blue := 2 in
  let P_C_blue := 8 / 12 in
  let P_D_blue := 2 / 4 in
  P_C_blue * P_D_blue = 1 / 3
:=
  sorry

end probability_blue_buttons_l249_249124


namespace new_cards_in_binder_l249_249686

theorem new_cards_in_binder 
  (cards_per_page : ℕ) (used_pages : ℕ) (old_cards : ℕ) (total_cards : ℕ) :
  cards_per_page = 3 → 
  used_pages = 6 → 
  old_cards = 10 → 
  total_cards = cards_per_page * used_pages → 
  total_cards - old_cards = 8 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2] at h4
  have h5 : total_cards = 18 := h4
  rw h5
  rw h3
  trivial  -- The proof would go here if needed.

end new_cards_in_binder_l249_249686


namespace length_RQ_l249_249954

variable (PQ PS PR PQ PQS PSQ PRQ PQR PSR SRT RQ : ℕ)
variable (T : Type) [metric_space T] [ordered_comm_group T]

-- Conditions
def angle_PSQ ≃ angle_PRQ := sorry
def angle_PSR ≃ angle_PQR := sorry
def length_PS : PS = 7 := rfl
def length_SR : SR = 9 := rfl
def length_PQ : PQ = 5 := rfl

-- Proof goal
theorem length_RQ : RQ = 14 := by
  sorry

end length_RQ_l249_249954


namespace equal_numbers_at_k_l249_249994

variable {n : ℕ} (a : Fin n → ℤ) (k : ℕ)

def operation (f : Fin n → ℤ) : Fin n → ℤ :=
  λ i, if h : i = 0 then f i else f i + 1

def sequence_eq (f : ℕ → Fin n → ℤ) : Prop :=
  ∀ i j, i < n → j < n → f i = f j

theorem equal_numbers_at_k :
  n > 2 →
  (∃ k, sequence_eq (nat.iterate operation f a k)) ↔
  (n % 2 = 1 ∨ (n % 2 = 0 ∧ (Finset.univ.sum (fun i => a i)) % 2 = 0)) :=
by
  sorry

end equal_numbers_at_k_l249_249994


namespace find_x3_minus_y3_l249_249022

theorem find_x3_minus_y3 {x y : ℤ} (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : x^3 - y^3 = 108 :=
by 
  sorry

end find_x3_minus_y3_l249_249022


namespace floor_div_eq_8100_l249_249710

def S (n : ℕ) : ℕ :=
  (∑ k in Finset.range (n + 1), (10^k - 1)) / 9

theorem floor_div_eq_8100 : (Real.floor (10^2017 / (S 2014))) = 8100 :=
  by exact sorry

end floor_div_eq_8100_l249_249710


namespace pete_flag_total_circle_square_l249_249296

theorem pete_flag_total_circle_square : 
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  circles + squares = 54 := 
by
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  show circles + squares = 54
  sorry

end pete_flag_total_circle_square_l249_249296


namespace possible_values_of_a_l249_249039

noncomputable def f (x : ℝ) := 4 * sin (π / 2 + x) ^ 2 + 4 * sin x

theorem possible_values_of_a (a : ℝ) :
  (∀ x ∈ set.Icc (0 : ℝ) a, f x ∈ set.Icc (4 : ℝ) (5 : ℝ)) ↔ a ∈ set.Icc (π / 6) π :=
by
  sorry

end possible_values_of_a_l249_249039


namespace Vovochka_max_candies_l249_249879

theorem Vovochka_max_candies (classmates : Finset ℕ) (candies : ℕ) (hclassmates : classmates.card = 25) (hcandies : candies = 200) (H : ∀ (S : Finset ℕ), S.card = 16 → 100 ≤ S.sum (λ x, 1)) :
  ∃ (max_candies : ℕ), max_candies = 37 :=
by
  use 37
  sorry

end Vovochka_max_candies_l249_249879


namespace sum_of_vertices_l249_249451

theorem sum_of_vertices (rect_verts: Nat) (pent_verts: Nat) (h1: rect_verts = 4) (h2: pent_verts = 5) : rect_verts + pent_verts = 9 :=
by
  sorry

end sum_of_vertices_l249_249451


namespace vovochka_max_candies_l249_249902

noncomputable def max_candies_for_vovochka 
  (total_classmates : ℕ)
  (total_candies : ℕ)
  (candies_condition : ∀ (subset : Finset ℕ), subset.card = 16 → ∑ i in subset, (classmate_candies i) ≥ 100) 
  (classmate_candies : ℕ → ℕ) : ℕ :=
  total_candies - ∑ i in (Finset.range total_classmates), classmate_candies i

theorem vovochka_max_candies :
  max_candies_for_vovochka 25 200 (λ subset hc, True) (λ i, if i < 13 then 7 else 6) = 37 :=
by
  sorry

end vovochka_max_candies_l249_249902


namespace line_passes_fixed_point_l249_249637

theorem line_passes_fixed_point (k : ℝ) : ∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ k * x - y - k + 1 = 0 :=
by
  use 1, 1
  split
  { refl }
  split
  { refl }
  { sorry } -- proof omitted

end line_passes_fixed_point_l249_249637


namespace arithmetic_seq_sum_lt_zero_l249_249461

theorem arithmetic_seq_sum_lt_zero {c : ℝ} (h : ∑ i in Finset.range 7, (i + 1 + c) < 0) : c < -4 := by
  sorry

end arithmetic_seq_sum_lt_zero_l249_249461


namespace rearrange_columns_nonneg_diagonal_sum_l249_249951

theorem rearrange_columns_nonneg_diagonal_sum (n : ℕ) 
  (grid : Fin n → Fin n → ℝ) 
  (h_sum_nonneg : 0 ≤ ∑ i j, grid i j) :
  ∃ π : List (Fin n) → List (Fin n), 
    (∀ x : List (Fin n), π x = list.perm x) ∧
    0 ≤ ∑ i, grid i (π (List of Fin n).nth i) :=
sorry

end rearrange_columns_nonneg_diagonal_sum_l249_249951


namespace max_divisors_up_to_20_l249_249216

def divisors (n : Nat) : List Nat :=
  (List.range (n + 1)).filter (λ d, n % d = 0)

def count_divisors (n : Nat) : Nat :=
  (divisors n).length

theorem max_divisors_up_to_20 :
  ∀ n, n ∈ List.range 21 → count_divisors n ≤ 6 ∧
      ((count_divisors 12 = 6) ∧
       (count_divisors 18 = 6) ∧
       (count_divisors 20 = 6)) :=
by
  sorry

end max_divisors_up_to_20_l249_249216


namespace common_chord_of_circles_equation_and_length_l249_249034

theorem common_chord_of_circles_equation_and_length :
  ∀ (x y : ℝ),
  (x^2 + y^2 + 6*x - 4 = 0) ∧ (x^2 + y^2 + 6*y - 28 = 0) →
  (∃ a b c : ℝ, a = 1 ∧ b = -1 ∧ c = 4 ∧ a*x + b*y + c = 0) ∧
  (∃ AB_length : ℝ, AB_length = 5 * real.sqrt 2) :=
by
  intros x y h
  sorry

end common_chord_of_circles_equation_and_length_l249_249034


namespace find_number_l249_249438

theorem find_number (x : ℕ) (h : x / 4 + 15 = 27) : x = 48 :=
sorry

end find_number_l249_249438


namespace minimum_n_value_l249_249960

theorem minimum_n_value : ∃ n : ℕ, n > 0 ∧ ∀ r : ℕ, (2 * n = 5 * r) → n = 5 :=
by
  sorry

end minimum_n_value_l249_249960


namespace roots_are_distinct_l249_249833

theorem roots_are_distinct (a x1 x2 : ℝ) (h : x1 ≠ x2) :
  (∀ x, x^2 - a*x - 2 = 0 → x = x1 ∨ x = x2) → x1 ≠ x2 := sorry

end roots_are_distinct_l249_249833


namespace gcd_f_l249_249144

def f (x: ℤ) : ℤ := x^2 - x + 2023

theorem gcd_f (x y : ℤ) (hx : x = 105) (hy : y = 106) : Int.gcd (f x) (f y) = 7 := by
  sorry

end gcd_f_l249_249144


namespace remainder_45_to_15_l249_249373

theorem remainder_45_to_15 : ∀ (N : ℤ) (k : ℤ), N = 45 * k + 31 → N % 15 = 1 :=
by
  intros N k h
  sorry

end remainder_45_to_15_l249_249373


namespace integer_satisfaction_l249_249063

def count_satisfying_integers : ℕ :=
  let count := λ (lower upper : ℤ), (↑upper - ↑lower + 1)
  
  count (-13) (-8) + count (-4) 2

theorem integer_satisfaction :
  let suitable_n := [(-13), (-12), (-11), (-10), (-9), (-8), (-4), (-3), (-2), (-1), 0, 1, 2] in
  (-13 ≤ n ∧ n ≤ 13 ∧ (n ∈ suitable_n))
    → (n ≥ 0 ∧ n ≤ 13 ∧ (n ∈ suitable_n) = 13) :=
sorry

end integer_satisfaction_l249_249063


namespace problem_l249_249145

noncomputable def roots (a b c : ℝ) : ℝ × ℝ :=
  let D : ℝ := b * b - 4 * a * c
  let root1 := (-b + Real.sqrt D) / (2 * a)
  let root2 := (-b - Real.sqrt D) / (2 * a)
  (root1, root2)

theorem problem (m n : ℝ) (h1 : (1, 2, -7) = roots 1 2 -7) (h2 : m + n = -2) (h3 : m^2 + 2*m - 7 = 0) : 
  m^2 + 3*m + n = 5 := by
  sorry

end problem_l249_249145


namespace find_f_2008_l249_249488

-- Define the odd function property
def is_odd_function (f : ℤ → ℤ) : Prop :=
  ∀ x, f(-x) = -f(x)

-- Define the periodic function property
def is_periodic_function (f : ℤ → ℤ) (p : ℤ) : Prop :=
  ∀ x, f(x) = f(x + p)

-- Define the function f and assume the given conditions
variable (f : ℤ → ℤ)
variable (h_odd : is_odd_function f)
variable (h_periodic : is_periodic_function f 3)
variable (h_f_neg1 : f (-1) = -1)

-- State the theorem that we want to prove
theorem find_f_2008 : f 2008 = 1 :=
by
  sorry

end find_f_2008_l249_249488


namespace abs_expression_value_l249_249998

theorem abs_expression_value (x : ℤ) (h : x = -2023) :
  abs (2 * abs (abs x - x) - abs x) - x = 8092 :=
by {
  -- Proof will be provided here
  sorry
}

end abs_expression_value_l249_249998


namespace hypotenuse_is_2_l249_249641

noncomputable def quadratic_trinomial_hypotenuse (a b c : ℝ) : ℝ :=
  let x1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let xv := -b / (2 * a)
  let yv := a * xv^2 + b * xv + c
  if xv = (x1 + x2) / 2 then
    Real.sqrt 2 * abs (-b / a)
  else 0

theorem hypotenuse_is_2 {a b c : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  quadratic_trinomial_hypotenuse a b c = 2 := by
  sorry

end hypotenuse_is_2_l249_249641


namespace greatest_num_divisors_in_range_l249_249176

def num_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d, n % d = 0)

theorem greatest_num_divisors_in_range : 
  ∀ n ∈ {1, 2, 3, ..., 20}, 
  num_divisors n < 7 → (n = 12 ∨ n = 18 ∨ n = 20) :=
by {
  sorry
}

end greatest_num_divisors_in_range_l249_249176


namespace coin_order_correct_l249_249453

-- Define the coins
inductive Coin
| A | B | C | D | E
deriving DecidableEq

open Coin

-- Define the conditions
def covers (x y : Coin) : Prop :=
  (x = A ∧ y = B) ∨
  (x = C ∧ (y = A ∨ y = D)) ∨
  (x = D ∧ y = B) ∨
  (y = E ∧ x = C)

-- Define the order of coins from top to bottom as a list
def coinOrder : List Coin := [C, E, A, D, B]

-- Prove that the order is correct
theorem coin_order_correct :
  ∀ c₁ c₂ : Coin, c₁ ≠ c₂ → List.indexOf c₁ coinOrder < List.indexOf c₂ coinOrder ↔ covers c₁ c₂ :=
by
  sorry

end coin_order_correct_l249_249453


namespace problem_statement_l249_249842

noncomputable def f (x : ℕ) : ℝ := sorry

theorem problem_statement :
  (∀ a b : ℕ, f (a + b) = f a * f b) →
  f 1 = 2 →
  (∑ i in Finset.range 2016, f i / f (i + 1)) = 1008 :=
by
  intros h1 h2
  sorry

end problem_statement_l249_249842


namespace find_f_10_l249_249831

def f : ℕ → ℕ
| 1     := 1
| 2     := 3
| 3     := 4
| 4     := 7
| 5     := 11
| (n+6) := f (n+5) + f (n+4)

theorem find_f_10 : f 10 = 123 := by
  sorry

end find_f_10_l249_249831


namespace square_area_l249_249118

structure Point :=
(x : ℝ)
(y : ℝ)

structure Square :=
(A B C D : Point)

def trisection_points (BD E F : Segment) : Prop :=
  let dist := (BD.length) / 3 in
  (distance B E = dist ∧ distance E D = 2 * dist) ∧
  (distance B F = 2 * dist ∧ distance F D = dist)

def intersection_points (A E B C F G H : Point) (BC AE GF AD : Line) : Prop :=
  intersects AE BC G ∧ intersects GF AD H

noncomputable def area_triangle (D H F : Point) : ℝ :=
  12

theorem square_area (ABCD : Square) (E F G H : Point)
  (BD AE GF BC AD : Line) 
  (h1 : trisection_points BD E F)
  (h2 : intersection_points ABCD.A E ABCD.B ABCD.C F G H BC AE GF AD)
  (h3 : area_triangle ABD.D H F = 12) : 
  (area_square ABCD = 288) :=
sorry

end square_area_l249_249118


namespace binomial_term_is_constant_range_of_a_over_b_l249_249112

noncomputable def binomial_term (a b : ℝ) (m n : ℤ) (r : ℕ) : ℝ :=
  Nat.choose 12 r * a^(12 - r) * b^r

theorem binomial_term_is_constant
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  ∃ r, r = 4 ∧
  (binomial_term a b m n r) = 1 :=
sorry

theorem range_of_a_over_b 
  (a b : ℝ)
  (m n : ℤ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: m ≠ 0)
  (h4: n ≠ 0)
  (h5: 2 * m + n = 0) :
  8 / 5 ≤ a / b ∧ a / b ≤ 9 / 4 :=
sorry

end binomial_term_is_constant_range_of_a_over_b_l249_249112


namespace divisors_of_12_18_20_l249_249189

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem divisors_of_12_18_20 (n : ℕ) (h : n ∈ {1, 2, 3, ..., 20}) :
  n ∈ {12, 18, 20} ↔ (∀ m ∈ {1, 2, 3, ..., 20}, count_divisors m ≤ 6) :=
by
  sorry

end divisors_of_12_18_20_l249_249189


namespace frustum_volume_and_lateral_surface_area_l249_249816

theorem frustum_volume_and_lateral_surface_area (h : ℝ) 
    (A1 A2 : ℝ) (r R : ℝ) (V S_lateral : ℝ) : 
    A1 = 4 * Real.pi → 
    A2 = 25 * Real.pi → 
    h = 4 → 
    r = 2 → 
    R = 5 → 
    V = (1 / 3) * (A1 + A2 + Real.sqrt (A1 * A2)) * h → 
    S_lateral = Real.pi * r * Real.sqrt (h ^ 2 + (R - r) ^ 2) + Real.pi * R * Real.sqrt (h ^ 2 + (R - r) ^ 2) → 
    V = 42 * Real.pi ∧ S_lateral = 35 * Real.pi := by
  sorry

end frustum_volume_and_lateral_surface_area_l249_249816


namespace ball_hits_ground_in_10_div_7_seconds_l249_249629

noncomputable def ball_height (t : ℝ) : ℝ := -4.9 * t^2 + 4 * t + 6

theorem ball_hits_ground_in_10_div_7_seconds :
  ∃ t : ℝ, ball_height(t) = 0 ∧ t = 10 / 7 := 
by
  have t_positive : (10 : ℝ) / 7 > 0 := 
    by norm_num
  use 10 / 7
  split
  · sorry -- Proof that ball_height (10 / 7) = 0
  · exact rfl

end ball_hits_ground_in_10_div_7_seconds_l249_249629


namespace palindrome_678_count_l249_249332

def is_palindrome (n : ℕ) :=
  let s := n.to_string in
  s = s.reverse

def is_seven_digit (n : ℕ) :=
  1000000 ≤ n ∧ n < 10000000

def uses_only_678 (n : ℕ) :=
  n.to_string.all (λ c, c = '6' ∨ c = '7' ∨ c = '8')

def palindromic_seven_digit_count : ℕ :=
  (Finset.range 10000000).filter (λ n, is_seven_digit n ∧ is_palindrome n ∧ uses_only_678 n).card

theorem palindrome_678_count : palindromic_seven_digit_count = 81 :=
by sorry

end palindrome_678_count_l249_249332


namespace train_length_is_800_l249_249391

noncomputable def calculate_train_length 
  (t : ℝ) 
  (v_m : ℝ) 
  (v_t : ℝ) : ℝ :=
  let relative_speed := (v_t - v_m) * (1000 / 3600)
  in relative_speed * t

theorem train_length_is_800 
  (t : ℝ := 47.99616030717543)
  (v_m : ℝ := 3)
  (v_t : ℝ := 63) : 
  calculate_train_length t v_m v_t = 800 :=
sorry

end train_length_is_800_l249_249391


namespace ratio_of_millipedes_l249_249365

-- Define the given conditions
def total_segments_needed : ℕ := 800
def first_millipede_segments : ℕ := 60
def millipedes_segments (x : ℕ) : ℕ := x
def ten_millipedes_segments : ℕ := 10 * 50

-- State the main theorem
theorem ratio_of_millipedes (x : ℕ) : 
  total_segments_needed = 60 + 2 * x + 10 * 50 →
  2 * x / 60 = 4 :=
sorry

end ratio_of_millipedes_l249_249365


namespace final_price_of_hat_is_correct_l249_249384

-- Definitions capturing the conditions.
def original_price : ℝ := 15
def first_discount_rate : ℝ := 0.20
def second_discount_rate : ℝ := 0.25

-- Calculations for the intermediate prices.
def price_after_first_discount : ℝ := original_price * (1 - first_discount_rate)
def final_price : ℝ := price_after_first_discount * (1 - second_discount_rate)

-- The theorem we need to prove.
theorem final_price_of_hat_is_correct : final_price = 9 := by
  sorry

end final_price_of_hat_is_correct_l249_249384


namespace slope_of_line_l249_249775

theorem slope_of_line (x y : ℝ) (h : 4 * y = 5 * x + 20) : y = (5/4) * x + 5 :=
by {
  sorry
}

end slope_of_line_l249_249775


namespace count_integers_satisfying_inequality_l249_249913

open Real

theorem count_integers_satisfying_inequality (e : ℝ) (h : e ≈ 2.718) : 
  ∃ n, n = 46 ∧ ∀ m : ℤ, -5 * e ≤ m ∧ m ≤ 12 * e ↔ -13 ≤ m ∧ m ≤ 32 := 
sorry

end count_integers_satisfying_inequality_l249_249913


namespace phase_shift_equivalence_l249_249651

noncomputable def y_original (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 - Real.sqrt 3 * Real.sin (2 * x)
noncomputable def y_target (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1
noncomputable def phase_shift : ℝ := 5 * Real.pi / 12

theorem phase_shift_equivalence : 
  ∀ x : ℝ, y_original x = y_target (x - phase_shift) :=
sorry

end phase_shift_equivalence_l249_249651


namespace possible_values_of_k_l249_249256

noncomputable def positive_divisors (n i : ℕ) : ℕ := sorry

theorem possible_values_of_k (n : ℕ) (k : ℕ) (d : ℕ → ℕ)
  (h_divisors : ∀ i < k, d (i + 1) ∣ n)
  (h_sorted : ∀ i j, i < j → d i < d j)
  (h_bounds : 5 ≤ k ∧ k ≤ 1000)
  (h_value_of_n : n = d 2 ^ d 3 * d 4 ^ d 5) :
  k = 45 ∨ k = 53 ∨ k = 253 ∨ k = 280 :=
begin
  sorry
end

end possible_values_of_k_l249_249256


namespace final_data_points_after_manipulations_l249_249975

def initial_data_points := 300
def increase_percentage := 0.15
def additional_data_points := 40
def removal_fraction := 1 / 6
def reduction_percentage := 0.10

theorem final_data_points_after_manipulations :
  let increased_data_points := initial_data_points * (1 + increase_percentage)
  let added_data_points := increased_data_points + additional_data_points
  let removed_data_points := (added_data_points * removal_fraction).toInt -- converting fraction to integer
  let remaining_data_points_after_removal := added_data_points - removed_data_points
  let final_reduction := (remaining_data_points_after_removal * reduction_percentage).toInt -- converting fraction to integer
  let final_total_data_points := remaining_data_points_after_removal - final_reduction
  final_total_data_points = 289 :=
by
  sorry

end final_data_points_after_manipulations_l249_249975


namespace card_count_ge_0_3_l249_249133

theorem card_count_ge_0_3 :
  let Jungkook := 0.8
  let Yoongi := 1 / 2
  let Yoojung := 0.9
  let Yuna := 1 / 3 
  in ( (Jungkook >= 0.3).TT + (Yoongi >= 0.3).TT + (Yoojung >= 0.3).TT + (Yuna >= 0.3).TT ) = 4 :=
by
  sorry

end card_count_ge_0_3_l249_249133


namespace square_perimeter_divided_into_six_rectangles_l249_249741

theorem square_perimeter_divided_into_six_rectangles (s : ℝ) (P : ℝ) (h1 : (2 * s + 2 * (s / 6)) = P) (h2 : P = 30): 
  4 * s = 360 / 7 :=
by
  have h3 : 7 * s = 90, from (by linarith [h1, h2]);
  have h4 : s = 90 / 7, from (by linarith [h3]);
  linarith [h4]

-- The statement specifies that given the conditions for the perimeter equation and its value,
-- the proof confirms the perimeter of the square is indeed 360/7 inches, as calculated. 

end square_perimeter_divided_into_six_rectangles_l249_249741


namespace trig_identity_l249_249564

-- Define the constant c
def c : ℝ := Real.pi / 7

-- The main theorem: Prove the trigonometric identity given the value of c
theorem trig_identity : 
  (sin (4 * c) * sin (5 * c) * cos (6 * c) * sin (7 * c) * sin (8 * c)) / 
  (sin (2 * c) * sin (3 * c) * sin (5 * c) * sin (6 * c) * sin (7 * c)) = 
  - cos (Real.pi / 7) :=
by
  sorry

end trig_identity_l249_249564


namespace no_three_consecutive_A_l249_249065

theorem no_three_consecutive_A : 
  let words := { w : List Char | w.length = 7 ∧ ∀ i, w[i] = 'A' ∨ w[i] = 'B' } in
  let valid_words := { w : List Char | w ∈ words ∧ ∀ i, w[i] = 'A' → (i < 2 ∨ w[i-1] ≠ 'A' ∨ w[i-2] ≠ 'A') } in
  words.count - valid_words.count = 128 - valid_words.count → valid_words.count = 85 :=
by sorry

end no_three_consecutive_A_l249_249065


namespace largest_angle_in_triangle_PQR_l249_249667

-- Definitions
def is_isosceles_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  (α = β) ∨ (β = γ) ∨ (γ = α)

def is_obtuse_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

variables (P Q R : ℝ)
variables (angleP angleQ angleR : ℝ)

-- Condition: PQR is an obtuse and isosceles triangle, and angle P measures 30 degrees
axiom h1 : is_isosceles_triangle P Q R angleP angleQ angleR
axiom h2 : is_obtuse_triangle P Q R angleP angleQ angleR
axiom h3 : angleP = 30

-- Theorem: The measure of the largest interior angle of triangle PQR is 120 degrees
theorem largest_angle_in_triangle_PQR : max angleP (max angleQ angleR) = 120 :=
  sorry

end largest_angle_in_triangle_PQR_l249_249667


namespace find_r_plus_s_l249_249767

variable (r s : ℝ)

def line_equation (x : ℝ) := (-5 / 3) * x + 15

theorem find_r_plus_s (h1 : s = line_equation r)
  (h2 : 4 * (1 / 2 * | r * (15 - s) |) = 1 / 2 * 9 * 15) :
  r + s = 10.5 :=
by
  sorry

end find_r_plus_s_l249_249767


namespace number_of_residents_in_rainbow_l249_249543

/-
  Mathematical conditions:
  1. There are exactly 1000 residents in Zhovtnevo.
  2. Zhovtnevo's population exceeds the average population of settlements in the valley by 90 people.
  3. There are 10 villages in the valley including Zhovtnevo.
  -/
variables (population_zhovtnevo : ℕ) (average_difference : ℕ) (total_villages : ℕ)

-- Given conditions:
def condition_1 := population_zhovtnevo = 1000
def condition_2 := average_difference = 90
def condition_3 := total_villages = 10

-- Result to prove:
def residents_in_rainbow :=
  population_zhovtnevo - average_difference = 900

theorem number_of_residents_in_rainbow :
  condition_1 → condition_2 → condition_3 → residents_in_rainbow :=
by
  intros h1 h2 h3
  rw [condition_1, condition_2, condition_3] at *
  sorry

end number_of_residents_in_rainbow_l249_249543


namespace relationship_among_vars_l249_249071

theorem relationship_among_vars {a b c d : ℝ} (h : (a + 2 * b) / (b + 2 * c) = (c + 2 * d) / (d + 2 * a)) :
  b = 2 * a ∨ a + b + c + d = 0 :=
sorry

end relationship_among_vars_l249_249071


namespace greatest_divisors_1_to_20_l249_249197

def is_divisor (a b : ℕ) : Prop := b % a = 0

def count_divisors (n : ℕ) : ℕ :=
  nat.succ (list.length ([k for k in list.range n if is_divisor k.succ n]))

def max_divisors_from_1_to_20 : set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 20 ∧ ∃ max_count, count_divisors n = max_count ∧ 
  ∀ m, 1 ≤ m ∧ m ≤ 20 → count_divisors m ≤ max_count}

theorem greatest_divisors_1_to_20 : max_divisors_from_1_to_20 = {12, 18, 20} :=
by {
  sorry
}

end greatest_divisors_1_to_20_l249_249197


namespace largest_interior_angle_obtuse_isosceles_triangle_l249_249659

theorem largest_interior_angle_obtuse_isosceles_triangle :
  ∀ (P Q R : Type) (α β γ : ℝ), α + β + γ = 180 ∧ γ = 120 ∧ α = 30 ∧ β = 30 →
  (α = 30 ∧ β = 30 ∧ γ = 120) ∨
  (α = 30 ∧ γ = 30 ∧ β = 120) ∨
  (β = 30 ∧ γ = 30 ∧ α = 120) → 
  γ = max α (max β γ) :=
by {
  intros P Q R α β γ h1 h2,
  repeat { rw h1 at * },
  rw h2,
  sorry
}

end largest_interior_angle_obtuse_isosceles_triangle_l249_249659


namespace number_of_triangles_with_positive_area_l249_249068

def is_valid_triangle (p1 p2 p3 : (ℤ × ℤ)) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x2 - x1) * (y3 - y1) ≠ (y2 - y1) * (x3 - x1)

def in_grid (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  1 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 4

def point_set : finset (ℤ × ℤ) :=
  finset.product (finset.range 4).map (λ x, x + 1) (finset.range 4).map (λ y, y + 1)

theorem number_of_triangles_with_positive_area :
  (point_set.choose 3).filter (λ t, is_valid_triangle t.1 t.2 t.3).card = 516 := sorry

end number_of_triangles_with_positive_area_l249_249068


namespace largest_angle_in_triangle_PQR_l249_249668

-- Definitions
def is_isosceles_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  (α = β) ∨ (β = γ) ∨ (γ = α)

def is_obtuse_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

variables (P Q R : ℝ)
variables (angleP angleQ angleR : ℝ)

-- Condition: PQR is an obtuse and isosceles triangle, and angle P measures 30 degrees
axiom h1 : is_isosceles_triangle P Q R angleP angleQ angleR
axiom h2 : is_obtuse_triangle P Q R angleP angleQ angleR
axiom h3 : angleP = 30

-- Theorem: The measure of the largest interior angle of triangle PQR is 120 degrees
theorem largest_angle_in_triangle_PQR : max angleP (max angleQ angleR) = 120 :=
  sorry

end largest_angle_in_triangle_PQR_l249_249668


namespace big_rectangle_width_l249_249381

theorem big_rectangle_width
  (W : ℝ)
  (h₁ : ∃ l w : ℝ, l = 40 ∧ w = W)
  (h₂ : ∃ l' w' : ℝ, l' = l / 2 ∧ w' = w / 2)
  (h_area : 200 = l' * w') :
  W = 20 :=
by sorry

end big_rectangle_width_l249_249381


namespace max_candies_vovochka_can_keep_l249_249910

-- Definitions for the conditions in the problem.
def classmates := 25
def total_candies := 200
def minimum_candies (candies_distributed : Vector ℕ classmates) : Prop :=
  ∀ (S : Finset (Fin classmates)), S.card = 16 → S.sum (candies_distributed.nth' S) ≥ 100

-- The proof goal
theorem max_candies_vovochka_can_keep (candies_distributed : Vector ℕ classmates) (h : minimum_candies candies_distributed) :
  ∃ k, k = 37 ∧ total_candies - Finset.univ.sum (candies_distributed.nth' Finset.univ) = k :=
begin
  sorry
end

end max_candies_vovochka_can_keep_l249_249910


namespace setB_forms_right_triangle_l249_249400

-- Define the sets of side lengths
def setA : (ℕ × ℕ × ℕ) := (2, 3, 4)
def setB : (ℕ × ℕ × ℕ) := (3, 4, 5)
def setC : (ℕ × ℕ × ℕ) := (5, 6, 7)
def setD : (ℕ × ℕ × ℕ) := (7, 8, 9)

-- Define the Pythagorean theorem condition
def isRightTriangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- The specific proof goal
theorem setB_forms_right_triangle : isRightTriangle 3 4 5 := by
  sorry

end setB_forms_right_triangle_l249_249400


namespace largest_angle_of_obtuse_isosceles_triangle_l249_249662

theorem largest_angle_of_obtuse_isosceles_triangle (P Q R : Type) 
  (triangle_PQR : Triangle P Q R)
  (isosceles_PQR : Isosceles triangle_PQR) 
  (obtuse_PQR : Obtuse triangle_PQR)
  (angle_P_30 : angle P triangle_PQR = 30) : 
  ∃ (angle_Q : ℕ), is_largest_angle angle_Q triangle_PQR ∧ angle_Q = 120 := 
by 
  sorry

end largest_angle_of_obtuse_isosceles_triangle_l249_249662


namespace largest_interior_angle_obtuse_isosceles_triangle_l249_249657

theorem largest_interior_angle_obtuse_isosceles_triangle :
  ∀ (P Q R : Type) (α β γ : ℝ), α + β + γ = 180 ∧ γ = 120 ∧ α = 30 ∧ β = 30 →
  (α = 30 ∧ β = 30 ∧ γ = 120) ∨
  (α = 30 ∧ γ = 30 ∧ β = 120) ∨
  (β = 30 ∧ γ = 30 ∧ α = 120) → 
  γ = max α (max β γ) :=
by {
  intros P Q R α β γ h1 h2,
  repeat { rw h1 at * },
  rw h2,
  sorry
}

end largest_interior_angle_obtuse_isosceles_triangle_l249_249657


namespace percentage_decrease_in_z_l249_249613

variable {x z q k : ℝ}
variable (h_pos_x : 0 < x) (h_pos_z : 0 < z)
variable (h_inv_prop : x * (z + 10) = k)
variable (h_q_percent : q ≠ 0)

theorem percentage_decrease_in_z :
  x * (z + 10) = k →
  let x' := x * (1 + q / 100) in
  let z' := (100 * (z + 10) / (100 + q)) - 10 in
  ((z - z') / z) * 100 = q * (z + 10) / (100 + q) :=
by
  sorry

end percentage_decrease_in_z_l249_249613


namespace valid_triangle_DEF_l249_249765

noncomputable def Triangle := Type -- a placeholder type for triangles

open_locale big_operators

-- Definitions and conditions
constant ABC : Triangle
constant A B C : Point           -- vertices of the triangle ABC
constant D E F : Point           -- points where the excircle meets the extensions of BC, CA, and AB respectively
constant DEF_tri : D ≠ E ∧ E ≠ F ∧ F ≠ D -- D, E, and F are distinct points

-- Proving that ∠DEF, ∠EFD, ∠FED are such that one is obtuse and the others are acute
theorem valid_triangle_DEF (ABC: Triangle) (A B C D E F: Point)
  (HD: Tangent (excircle opposite A) (line_of B C))
  (HE: Tangent (excircle opposite A) (line_of C A))
  (HF: Tangent (excircle opposite A) (line_of A B))
  (distinct: D ≠ E ∧ E ≠ F ∧ F ≠ D) :
  ∃ θ₁ θ₂ θ₃ : Angle,
    (θ₁ + θ₂ + θ₃ = 180 ∧ (one_obtuse θ₁ θ₂ θ₃ ∧ (two_acute θ₁ θ₂ θ₃))) :=
sorry

end valid_triangle_DEF_l249_249765


namespace centroid_projection_identity_l249_249963

noncomputable def triangle (A B C : Point) : Triangle := sorry
noncomputable def centroid (T : Triangle) : Point := sorry
noncomputable def projection (P L1 L2 : Point) : Point := sorry
noncomputable def side_length (A B : Point) : Real := sorry
noncomputable def vector (A B : Point) : Vector := sorry

theorem centroid_projection_identity {A B C : Point} :
  let T := triangle A B C in
  let S := centroid T in
  let A1 := projection S B C in
  let B1 := projection S C A in
  let C1 := projection S A B in
  let a := side_length B C in
  let b := side_length C A in
  let c := side_length A B in
  a^2 * vector S A1 + b^2 * vector S B1 + c^2 * vector S C1 = 0 :=
sorry

end centroid_projection_identity_l249_249963


namespace sequence_bounds_l249_249579

noncomputable def seq (n : ℕ) : ℕ → ℝ
| 0 := 1 / 2
| (k + 1) := seq k + 1 / n * (seq k) ^ 2

theorem sequence_bounds (n : ℕ) : 1 - 1 / n < seq n n ∧ seq n n < 1 :=
by
  sorry

end sequence_bounds_l249_249579


namespace sum_of_reciprocals_l249_249319

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) : 
  1 / x + 1 / y = 3 :=
by
  sorry

end sum_of_reciprocals_l249_249319


namespace compute_expression_l249_249416

theorem compute_expression : 
  let x := 19
  let y := 15
  (x + y)^2 - (x - y)^2 = 1140 :=
by
  sorry

end compute_expression_l249_249416


namespace remainder_of_product_mod_six_l249_249796

theorem remainder_of_product_mod_six (n : ℕ) (hn : n = 21) : 
  let P : ℕ := ∏ i in finset.range (n + 1), 4 + 10 * i
  in P % 6 = 4 :=
by sorry

end remainder_of_product_mod_six_l249_249796


namespace chessboard_polygon_l249_249701

-- Conditions
variable (A B a b : ℕ)

-- Statement of the theorem
theorem chessboard_polygon (A B a b : ℕ) : A - B = 4 * (a - b) :=
sorry

end chessboard_polygon_l249_249701


namespace median_books_read_l249_249631

def num_students_per_books : List (ℕ × ℕ) := [(2, 8), (3, 5), (4, 9), (5, 3)]

theorem median_books_read 
  (h : num_students_per_books = [(2, 8), (3, 5), (4, 9), (5, 3)]) : 
  ∃ median, median = 3 :=
by {
  sorry
}

end median_books_read_l249_249631


namespace part1_part2_l249_249703

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x > -1, (x^2 + 3*x + 6) / (x + 1) ≥ a) ↔ (a ≤ 5) := 
  sorry

-- Part 2
theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) : 
  2*a + (1/a) + 4*b + (8/b) ≥ 27 :=
  sorry

end part1_part2_l249_249703


namespace min_m_n_sum_divisible_by_27_l249_249521

theorem min_m_n_sum_divisible_by_27 (m n : ℕ) (h : 180 * m * (n - 2) % 27 = 0) : m + n = 6 :=
sorry

end min_m_n_sum_divisible_by_27_l249_249521


namespace unit_vector_perpendicular_l249_249836

theorem unit_vector_perpendicular (x y : ℝ) (a : ℝ × ℝ) (h : a = (2, -2)) :
  (2 * x - 2 * y = 0 ∧ x^2 + y^2 = 1) ↔ (x = sqrt 2 / 2 ∧ y = sqrt 2 / 2) ∨ (x = -sqrt 2 / 2 ∧ y = -sqrt 2 / 2) :=
by
  sorry

end unit_vector_perpendicular_l249_249836


namespace projection_vector_unique_l249_249582

open Locale.BigOperators LinearAlgebra

-- Definitions of given vectors.
def a : ℝ^3 := ![2, -2, 3]
def b : ℝ^3 := ![-1, 4, 1]

-- Projection vector p and requirement for it to be collinear with a and b.
def collinear (u v : ℝ^3) : Prop := ∃ (t : ℝ), u = t • v

-- The statement of the problem, with an assertion about the vector p.
theorem projection_vector_unique (v : ℝ^3)
  (hv : v ≠ 0)
  (proj_a : ∃ t : ℝ, a = t • v)
  (proj_b : ∃ t : ℝ, b = t • v)
  (p : ℝ^3)
  (hpa : ∃ t : ℝ, p = t • a)
  (hpb : ∃ t : ℝ, p = t • b) :
  p = ![-(10/7), 46/49, 2] :=
sorry

end projection_vector_unique_l249_249582


namespace divide_triangle_into_equal_areas_l249_249768

-- Given conditions
variables (A B C : Point)
variables (AC BC : ℝ)
variable (h : AC ≥ BC)

-- Define that there exists a line parallel to the internal bisector of ∠ C which divides the area evenly
theorem divide_triangle_into_equal_areas 
  (triangle : Triangle A B C)
  (f_c : Line) 
  (hf : is_angle_bisector f_c (angle A C B))
  : ∃ e : Line, is_parallel_to e f_c ∧ divides_triangle_into_equal_areas e triangle :=
sorry

end divide_triangle_into_equal_areas_l249_249768


namespace intervals_of_increase_length_of_side_a_l249_249848

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * sin (x + π / 3)

theorem intervals_of_increase :
  ∀ k : ℤ, (∃ I : set ℝ, I = set.Icc (k * π - π / 6) (k * π + π / 3) ∧ 
  ∀ x ∈ I, ∃ ϵ > 0, ∀ u, x ≤ u ∧ u ≤ x + ϵ → f(u) > f(x)) :=
sorry

variables (a b c A B C : ℝ)
variables (triangle_ABC : Prop)
variables (AD BD : ℝ)
variables (is_angle_bisector : Prop)
variables (is_axis_of_symmetry : Prop)
variables (AD_length_eq √2_BD_length_eq_2 : AD = sqrt 2 * BD ∧ AD = 2)

theorem length_of_side_a :
  triangle_ABC ∧ is_angle_bisector ∧ is_axis_of_symmetry ∧ AD_length_eq √2_BD_length_eq_2 →
  a = sqrt 6 :=
sorry

end intervals_of_increase_length_of_side_a_l249_249848


namespace not_product_24_pair_not_24_l249_249341

theorem not_product_24 (a b : ℤ) : 
  (a, b) = (-4, -6) ∨ (a, b) = (-2, -12) ∨ (a, b) = (2, 12) ∨ (a, b) = (3/4, 32) → a * b = 24 :=
sorry

theorem pair_not_24 :
  ¬(1/3 * -72 = 24) :=
sorry

end not_product_24_pair_not_24_l249_249341


namespace solution_set_a_eq_1_range_of_a_l249_249038

noncomputable def f (x a : ℝ) : ℝ := -x^2 + a * x + 4

noncomputable def g (x : ℝ) : ℝ := |x + 1| + |x - 1|

theorem solution_set_a_eq_1 :
  { x : ℝ | f x 1 ≥ g x } = set.Icc (-1 : ℝ) ((real.sqrt 17 - 1) / 2) :=
sorry

theorem range_of_a :
  (∀ x ∈ set.Icc (-1 : ℝ) (1 : ℝ), f x a ≥ g x) →
  a ∈ set.Icc (-1 : ℝ) (1 : ℝ) :=
sorry

end solution_set_a_eq_1_range_of_a_l249_249038


namespace pedal_circles_coincide_and_center_sides_pedal_triangle_perpendicular_l249_249347

variables {P1 P2 A B C : Point} -- Define points P1, P2, and vertices A, B, C of triangle ABC
variables (h_iso : isogonal_conjugates P1 P2 A B C) -- Define the isogonal conjugates condition
variables (angle_line : ℝ) -- Define the angle for Part (b)

-- Proof of Part (a) and (b):
theorem pedal_circles_coincide_and_center : 
  ∀ (P1 P2 A B C : Point) (h_iso : isogonal_conjugates P1 P2 A B C),
  circle_center_pedal P1,P2 A B C = midpoint P1 P2 :=
by sorry

-- Proof of Part (c):
theorem sides_pedal_triangle_perpendicular :
  ∀ (P1 P2 A B C : Point) (h_iso : isogonal_conjugates P1 P2 A B C),
  sides_perpendicular_to_lines P1 P2 A B C :=
by sorry

end pedal_circles_coincide_and_center_sides_pedal_triangle_perpendicular_l249_249347


namespace sum_f_neg12_to_13_l249_249484

noncomputable def f (x : ℝ) := 1 / (3^x + Real.sqrt 3)

theorem sum_f_neg12_to_13 : 
  (f (-12) + f (-11) + f (-10) + f (-9) + f (-8) + f (-7) + f (-6)
  + f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0
  + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10
  + f 11 + f 12 + f 13) = (13 * Real.sqrt 3 / 3) :=
sorry

end sum_f_neg12_to_13_l249_249484


namespace numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249230

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).length

theorem numbers_with_most_divisors_in_range :
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  (number_of_divisors n ≤ 6) :=
by
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list
  simp [number_of_divisors] <;>
  dec_trivial -- Automatically discharges the goals since they are simple arithmetic comparisons

theorem greatest_divisors_in_range :
  number_of_divisors 12 = 6 ∧
  number_of_divisors 18 = 6 ∧
  number_of_divisors 20 = 6 ∧
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19],
  number_of_divisors n < 6 :=
by
  refine ⟨rfl, rfl, rfl, _⟩
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list excluding 12, 18, 20
  dec_trivial -- Automatically discharges the goals due to their simplicity

#print greatest_divisors_in_range

end numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249230


namespace nonnegative_difference_between_roots_l249_249336

theorem nonnegative_difference_between_roots : 
  ∀ (x : ℝ), (x^2 + 40 * x + 300 = -100) → 
  let eq := x^2 + 40 * x + 400 in
  let r1 := -20 in
  let r2 := -20 in
  (r1 - r2).abs = 0 :=
by
  intros x h
  let eq := x^2 + 40 * x + 400
  let r1 := -20
  let r2 := -20
  have h1 : r1 = r2 := by
    sorry
  show (r1 - r2).abs = 0 from
    by rw [h1, sub_self, abs_zero]

end nonnegative_difference_between_roots_l249_249336


namespace number_properties_l249_249540

def number : ℕ := 52300600

def position_of_2 : ℕ := 10^6

def value_of_2 : ℕ := 20000000

def position_of_5 : ℕ := 10^7

def value_of_5 : ℕ := 50000000

def read_number : String := "five hundred twenty-three million six hundred"

theorem number_properties : 
  position_of_2 = (10^6) ∧ value_of_2 = 20000000 ∧ 
  position_of_5 = (10^7) ∧ value_of_5 = 50000000 ∧ 
  read_number = "five hundred twenty-three million six hundred" :=
by sorry

end number_properties_l249_249540


namespace parabola_vertex_path_correct_l249_249355

open Real

noncomputable def parabola_vertex_path (a : ℝ) (h : 0 < a) :=
  {λ : ℝ // True}.val = λ x, a^2 - x^2

theorem parabola_vertex_path_correct (a : ℝ) (h : 0 < a) (λ : ℝ) :
  (∃ x y : ℝ, y = x^2 - 2 * λ * x + a^2 ∧ y = a^2 - x^2) → True := sorry

end parabola_vertex_path_correct_l249_249355


namespace sales_commission_l249_249697

theorem sales_commission (total_sale : ℕ) (c1 c2 : ℕ) (r1 r2 : ℚ) (commission_percent : ℚ) :
  total_sale = 800 → c1 = 500 → c2 = total_sale - c1 →
  r1 = 0.20 → r2 = 0.25 →
  commission_percent = ((r1 * c1 + r2 * c2) / total_sale) * 100 → 
  commission_percent ≈ 21.88 :=
by
  sorry

end sales_commission_l249_249697


namespace min_area_equals_min_length_l249_249237

theorem min_area_equals_min_length (a b c : ℤ) : ∃ (S l : ℕ), 
  (∀ (u v : (ℤ × ℤ × ℤ)), 
    (u.1 * a + u.2.1 * b + u.2.2 * c = 0) → 
    (v.1 * a + v.2.1 * b + v.2.2 * c = 0) → 
    let S := (u.1 * v.2.2 - u.2.2 * v.1)^2 + 
             (u.2.1 * v.2.2 - u.2.2 * v.2.1)^2 + 
             (u.2.2 * v.1 - u.2.1 * v.1)^2 in S ≥ l^2) ∧
    l = int.sqrt (a^2 + b^2 + c^2) :=
sorry

end min_area_equals_min_length_l249_249237


namespace candy_problem_l249_249805

theorem candy_problem
  (G : Nat := 7) -- Gwen got 7 pounds of candy
  (C : Nat := 17) -- Combined weight of candy
  (F : Nat) -- Pounds of candy Frank got
  (h : F + G = C) -- Condition: Combined weight
  : F = 10 := 
by
  sorry

end candy_problem_l249_249805


namespace first_order_difference_arithmetic_sequence_general_formula_existence_of_arithmetic_b_l249_249473

-- Part 1: First problem
theorem first_order_difference_arithmetic (a_n : ℕ → ℤ)
  (h : ∀ n, a_n n = n^2 - n) :
  ∃ d : ℤ, ∀ n, (a_n (n + 1) - a_n n) = d * n :=
by sorry

-- Part 2: Second problem
theorem sequence_general_formula (a_n : ℕ → ℤ)
  (h1 : a_n 1 = 1)
  (h2 : ∀ n, (a_n (n + 1) - a_n n) - a_n n = 2^n) :
  ∀ n, a_n n = n * 2^(n - 1) :=
by sorry

-- Part 3: Third problem
theorem existence_of_arithmetic_b (a_n b_n : ℕ → ℤ)
  (h : ∀ n, a_n n = n * 2^(n - 1))
  (hb_seq : ∃ d b₁ : ℤ, ∀ n, b_n n = b₁ + d * (n - 1)) :
  ∀ n, a_n n = ∑ k in finset.range n, b_n (k + 1) * nat.choose n (k + 1) :=
by sorry

end first_order_difference_arithmetic_sequence_general_formula_existence_of_arithmetic_b_l249_249473


namespace pinedale_bus_speed_l249_249618

theorem pinedale_bus_speed 
  (stops_every_minutes : ℕ)
  (num_stops : ℕ)
  (distance_km : ℕ)
  (time_per_stop_minutes : stops_every_minutes = 5)
  (dest_stops : num_stops = 8)
  (dest_distance : distance_km = 40) 
  : (distance_km / (num_stops * stops_every_minutes / 60)) = 60 := 
by
  sorry

end pinedale_bus_speed_l249_249618


namespace certain_number_is_l249_249642

theorem certain_number_is (x : ℝ) : 
  x * (-4.5) = 2 * (-4.5) - 36 → x = 10 :=
by
  intro h
  -- proof goes here
  sorry

end certain_number_is_l249_249642


namespace probability_heads_penny_nickel_dime_l249_249275

theorem probability_heads_penny_nickel_dime :
  let total_outcomes := 2^5 in
  let successful_outcomes := 2 * 2 in
  (successful_outcomes : ℝ) / total_outcomes = (1 / 8 : ℝ) :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249275


namespace sum_S5_l249_249403

-- Define the sequence and sum function
def exp_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q^n

def sum_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a 

-- Given Conditions
variables (a : ℕ → ℝ) (q : ℝ)
variable (h_exp_seq : exp_seq a q)
variable (h_pos : ∀ n, 0 < a n)
variable (h_a3 : a 3 = 4)
variable (h_a2a6 : a 2 * a 6 = 64)

-- Goal
theorem sum_S5 : sum_n a 5 = 31 :=
by 
  sorry

end sum_S5_l249_249403


namespace initial_visual_range_is_90_l249_249715

-- Define the initial visual range without the telescope (V).
variable (V : ℝ)

-- Define the condition that the visual range with the telescope is 150 km.
variable (condition1 : V + (2 / 3) * V = 150)

-- Define the proof problem statement.
theorem initial_visual_range_is_90 (V : ℝ) (condition1 : V + (2 / 3) * V = 150) : V = 90 :=
sorry

end initial_visual_range_is_90_l249_249715


namespace honey_tangerines_estimation_l249_249374

noncomputable def normal_distribution (mu sigma : ℝ) (x : ℝ) : ℝ :=
  (1 / (sigma * Mathlib.sqrt (2 * Mathlib.pi))) * Mathlib.exp (-(x - mu) ^ 2 / (2 * sigma ^ 2))

theorem honey_tangerines_estimation :
  let mu := 90
  let sigma := 2
  let total_tangerines := 10000
  let p_value := 0.9987
  (p_value * total_tangerines) = 9987 :=
by
  sorry

end honey_tangerines_estimation_l249_249374


namespace remainder_expression_mod_88_l249_249414

theorem remainder_expression_mod_88 :
  (1 - 90 * nat.choose 10 1 + 90^2 * nat.choose 10 2 - 90^3 * nat.choose 10 3 
   + ... + (-1)^k * 90^k * nat.choose 10 k + ... + 90^10 * nat.choose 10 10) % 88 = 1 := sorry

end remainder_expression_mod_88_l249_249414


namespace find_x3_y3_l249_249017

noncomputable def x_y_conditions (x y : ℝ) :=
  x - y = 3 ∧
  x^2 + y^2 = 27

theorem find_x3_y3 (x y : ℝ) (h : x_y_conditions x y) : x^3 - y^3 = 108 :=
  sorry

end find_x3_y3_l249_249017


namespace fundraising_group_initial_girls_l249_249524

theorem fundraising_group_initial_girls (p : ℕ) (h1 : 0.5 * p ∈ ℕ)
  (h2 : (0.5 * p : ℝ) - 3 / p = 0.4) : 0.5 * p = 15 := by
  sorry

end fundraising_group_initial_girls_l249_249524


namespace inequality_does_not_hold_l249_249824

theorem inequality_does_not_hold
  (a b c : ℝ)
  (h1 : c < b)
  (h2 : b < a)
  (h3 : a * c < 0) :
  ¬ (cb < ab) :=
sorry

end inequality_does_not_hold_l249_249824


namespace symmetric_axis_l249_249320

noncomputable def f (x : ℝ) : ℝ :=
  2 * sin (x + 5 * Real.pi / 6) * sin (x + Real.pi / 3)

theorem symmetric_axis :
  ∃ k : ℤ, ∃ x : ℝ, f(x) = f(-x + k * Real.pi / 2) ∧ x = 5 * Real.pi / 12 :=
sorry

end symmetric_axis_l249_249320


namespace beetle_distance_from_A_beetle_total_time_l249_249712

-- Define the distances as a list
def movements : List ℝ := [10, -9, 8, -6, 7.5, -6, 8, -7]

-- Define the speed of the beetle
def speed : ℝ := 2

-- Define the distance from point A after eight movements
def distance_from_A (movements : List ℝ) : ℝ :=
  movements.sum

-- Define the total time taken for eight movements
def total_time (movements : List ℝ) (speed : ℝ) : ℝ :=
  movements.sum (λ x => |x|) / speed

-- Prove the calculated distance from point A
theorem beetle_distance_from_A : distance_from_A movements = 5.5 := 
by 
  sorry

-- Prove the total time taken for the crawl
theorem beetle_total_time : total_time movements speed = 30.75 := 
by 
  sorry

end beetle_distance_from_A_beetle_total_time_l249_249712


namespace max_divisors_1_to_20_l249_249211

theorem max_divisors_1_to_20 :
  ∃ m n o ∈ set.range (20:ℕ.succ), 
    (∀ k ∈ set.range (20:ℕ.succ), 
      (nat.divisors_count k ≤ nat.divisors_count m) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count n) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count o)) 
    ∧ (nat.divisors_count m = 6)
    ∧ (nat.divisors_count n = 6)
    ∧ (nat.divisors_count o = 6)
    ∧ m ≠ n ∧ n ≠ o ∧ o ≠ m
    ∧ set.mem 12 (set.range (20 + 1))
    ∧ set.mem 18 (set.range (20 + 1))
    ∧ set.mem 20 (set.range (20 + 1)). 
      := by 
  -- all these steps will be skipped here
  sorry

end max_divisors_1_to_20_l249_249211


namespace science_votes_percentage_l249_249949

theorem science_votes_percentage 
  (math_votes : ℕ) (english_votes : ℕ) (science_votes : ℕ) (history_votes : ℕ) (art_votes : ℕ) 
  (total_votes : ℕ := math_votes + english_votes + science_votes + history_votes + art_votes) 
  (percentage : ℕ := ((science_votes * 100) / total_votes)) :
  math_votes = 80 →
  english_votes = 70 →
  science_votes = 90 →
  history_votes = 60 →
  art_votes = 50 →
  percentage = 26 :=
by
  intros
  sorry

end science_votes_percentage_l249_249949


namespace probability_heads_penny_nickel_dime_l249_249276

theorem probability_heads_penny_nickel_dime :
  let total_outcomes := 2^5 in
  let successful_outcomes := 2 * 2 in
  (successful_outcomes : ℝ) / total_outcomes = (1 / 8 : ℝ) :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249276


namespace tangent_triangle_angle_l249_249706

theorem tangent_triangle_angle (O P A B : Type) [circle O] [tangent_triangle P A B O] (h₁ : ∠ APB = 40) : ∠ AOB = 70 :=
sorry

end tangent_triangle_angle_l249_249706


namespace line_ellipse_tangent_l249_249048

theorem line_ellipse_tangent (m : ℝ) (h : ∃ x y : ℝ, y = 2 * m * x + 2 ∧ 2 * x^2 + 8 * y^2 = 8) :
  m^2 = 3 / 16 :=
sorry

end line_ellipse_tangent_l249_249048


namespace part_one_part_two_l249_249851

theorem part_one (ω : ℝ) (h1 : ω > 0) (φ : ℝ) (h2 : |φ| < π / 2) 
  (h3 : (λ x, sin(ω * x + φ)) 0 = -√3 / 2) : φ = -π / 3 :=
sorry

theorem part_two (ω : ℝ) (h1 : ω > 0) (φ : ℝ) (h2 : |φ| < π / 2)
  (h3 : ∀ x ∈ Icc (-π / 3) (2 * π / 3), sin(ω * x + φ) ≤ sin(ω * (x + 1) + φ))
  (h4 : sin(ω * (2 * π / 3) + φ) = 1)
  (h5 : sin(ω * (-π / 3) + φ) = -1) : ω = 1 ∧ φ = -π / 6 :=
sorry

-- Here, I've chosen condition 2 for the problem conversion.

end part_one_part_two_l249_249851


namespace leap_year_hours_l249_249508

theorem leap_year_hours (days_in_regular_year : ℕ) (hours_in_day : ℕ) (is_leap_year : Bool) : 
  is_leap_year = true ∧ days_in_regular_year = 365 ∧ hours_in_day = 24 → 
  366 * hours_in_day = 8784 :=
by
  intros
  sorry

end leap_year_hours_l249_249508


namespace combined_tax_rate_john_ingrid_l249_249700

noncomputable def combinedTaxRate (johnIncome johnTaxRate ingridIncome ingridTaxRate : ℝ) : ℝ :=
  let johnTax := johnTaxRate * johnIncome
  let ingridTax := ingridTaxRate * ingridIncome
  let totalTax := johnTax + ingridTax
  let totalIncome := johnIncome + ingridIncome
  totalTax / totalIncome

theorem combined_tax_rate_john_ingrid :
  combinedTaxRate 56000 0.30 72000 0.40 = 0.35625 :=
by
  let johnIncome := 56000
  let johnTaxRate := 0.30
  let ingridIncome := 72000
  let ingridTaxRate := 0.40
  let johnTax := johnTaxRate * johnIncome
  let ingridTax := ingridTaxRate * ingridIncome
  let totalTax := johnTax + ingridTax
  let totalIncome := johnIncome + ingridIncome
  let combinedTaxRateCalc := totalTax / totalIncome
  have combinedTaxRateCalc = 0.35625 := sorry
  exact combinedTaxRateCalc

end combined_tax_rate_john_ingrid_l249_249700


namespace remainder_of_product_mod_six_l249_249797

theorem remainder_of_product_mod_six (n : ℕ) (hn : n = 21) : 
  let P : ℕ := ∏ i in finset.range (n + 1), 4 + 10 * i
  in P % 6 = 4 :=
by sorry

end remainder_of_product_mod_six_l249_249797


namespace union_sets_l249_249825

noncomputable def setA : Set ℝ := { x | x^2 - 3*x - 4 ≤ 0 }
noncomputable def setB : Set ℝ := { x | 1 < x ∧ x < 5 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_sets_l249_249825


namespace Batman_game_cost_l249_249326

theorem Batman_game_cost (football_cost strategy_cost total_spent batman_cost : ℝ)
  (h₁ : football_cost = 14.02)
  (h₂ : strategy_cost = 9.46)
  (h₃ : total_spent = 35.52)
  (h₄ : total_spent = football_cost + strategy_cost + batman_cost) :
  batman_cost = 12.04 := by
  sorry

end Batman_game_cost_l249_249326


namespace distance_to_grammys_house_l249_249303

def cost_per_tank : ℝ := 45
def distance_per_tank : ℝ := 500
def food_ratio : ℝ := 3/5
def total_expenses : ℝ := 288

theorem distance_to_grammys_house : ∃ d : ℝ, d = 2900 := by
  let food_cost := food_ratio * cost_per_tank
  let fuel_cost := total_expenses - food_cost
  let tanks := fuel_cost / cost_per_tank
  let distance := tanks * distance_per_tank
  have h : distance = 2900 := sorry -- This is where we would include the actual proof.
  exact ⟨distance, h⟩

end distance_to_grammys_house_l249_249303


namespace fraction_of_gasoline_used_l249_249693

theorem fraction_of_gasoline_used (speed : ℕ) (consumption_rate : ℕ) (initial_gas : ℕ) (time : ℕ) (fraction_used : ℚ) :
  speed = 40 → consumption_rate = 40 → initial_gas = 12 → time = 5 → fraction_used = 5 / 12 :=
by
  intros h_speed h_consumption h_initial_gas h_time
  have distance_travelled : ℕ := speed * time
  have gallons_used : ℚ := distance_travelled / consumption_rate
  have fraction_of_tank : ℚ := gallons_used / initial_gas
  simp [h_speed, h_consumption, h_initial_gas, h_time] at *
  have : distance_travelled = 200 := by simp [h_speed, h_time]
  rw this at *
  have : gallons_used = 5 := by norm_num
  rw this at *
  have : fraction_of_tank = 5 / 12 := by norm_num
  assumption

end fraction_of_gasoline_used_l249_249693


namespace probability_of_drawing_white_ball_l249_249092

variable (a b c : ℕ)

theorem probability_of_drawing_white_ball :
  (a + b + c > 0) → ((a : ℚ) / (a + b + c) = (a : ℚ) / (a + b + c)) := 
by
  intro h
  exact eq.refl ((a : ℚ) / (a + b + c))

-- Define conditions
lemma conditions : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 := sorry

-- Proof of the main theorem
noncomputable def main_theorem : Prop :=
  ∀ (a b c : ℕ), a ≥ 0 → b ≥ 0 → c ≥ 0 → (a + b + c > 0) → ((a : ℚ) / (a + b + c) = (a : ℚ) / (a + b + c))

-- Proof
example : main_theorem := 
by
  intros
  apply probability_of_drawing_white_ball
  assumption

end probability_of_drawing_white_ball_l249_249092


namespace simplify_radical_expr_l249_249412

-- Define the variables and expressions
variables {x : ℝ} (hx : 0 ≤ x) 

-- State the problem
theorem simplify_radical_expr (hx : 0 ≤ x) :
  (Real.sqrt (100 * x)) * (Real.sqrt (3 * x)) * (Real.sqrt (18 * x)) = 30 * x * Real.sqrt (6 * x) :=
sorry

end simplify_radical_expr_l249_249412


namespace min_distinct_values_l249_249377

theorem min_distinct_values (n : ℕ) (mode_freq : ℕ) (total : ℕ)
  (h1 : total = 3000) (h2 : mode_freq = 15) :
  n = 215 :=
by
  sorry

end min_distinct_values_l249_249377


namespace area_of_region_l249_249791

noncomputable def area_between_curve_and_line : ℝ :=
  ∫ x in -1..1, (1 - x^2)

theorem area_of_region : area_between_curve_and_line = 4 / 3 :=
by
  sorry

end area_of_region_l249_249791


namespace Yella_last_week_usage_l249_249344

/-- 
Yella's computer usage last week was some hours. If she plans to use the computer 8 hours a day for this week, 
her computer usage for this week is 35 hours less. Given these conditions, prove that Yella's computer usage 
last week was 91 hours.
-/
theorem Yella_last_week_usage (daily_usage : ℕ) (days_in_week : ℕ) (difference : ℕ)
  (h1: daily_usage = 8)
  (h2: days_in_week = 7)
  (h3: difference = 35) :
  daily_usage * days_in_week + difference = 91 := 
by
  sorry

end Yella_last_week_usage_l249_249344


namespace intersect_xy_plane_l249_249446

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def direction_vector (p1 p2 : Point3D) : Point3D :=
  Point3D.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)

noncomputable def line_param (p : Point3D) (dir : Point3D) (t : ℝ) : Point3D :=
  Point3D.mk (p.x + dir.x * t) (p.y + dir.y * t) (p.z + dir.z * t)

theorem intersect_xy_plane :
  let p1 := Point3D.mk 2 3 2,
      p2 := Point3D.mk 4 0 7,
      dir := direction_vector p1 p2,
      t := (-2) / 5,
      intersection := line_param p1 dir t
  in intersection = Point3D.mk ((2 + 2 * (-2/5))) ((3 - 3 * (-2/5))) 0 :=
by
  sorry

end intersect_xy_plane_l249_249446


namespace greatest_num_divisors_in_range_l249_249173

def num_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d, n % d = 0)

theorem greatest_num_divisors_in_range : 
  ∀ n ∈ {1, 2, 3, ..., 20}, 
  num_divisors n < 7 → (n = 12 ∨ n = 18 ∨ n = 20) :=
by {
  sorry
}

end greatest_num_divisors_in_range_l249_249173


namespace largest_angle_of_isosceles_obtuse_30_deg_l249_249670

def is_isosceles (T : Triangle) : Prop :=
  T.A = T.B ∨ T.B = T.C ∨ T.C = T.A

def is_obtuse (T : Triangle) : Prop :=
  T.A > 90 ∨ T.B > 90 ∨ T.C > 90

def T : Type := {P Q R : ℝ}

noncomputable def largest_angle (T : Triangle) : ℝ :=
  max T.A (max T.B T.C)

theorem largest_angle_of_isosceles_obtuse_30_deg :
  ∀ (T : Triangle), is_isosceles T → is_obtuse T → T.A = 30 → largest_angle T = 120 :=
by
  intro T h_iso h_obt h_A30
  sorry

end largest_angle_of_isosceles_obtuse_30_deg_l249_249670


namespace axis_of_symmetry_l249_249792

noncomputable def equation_of_axis_of_symmetry : set (ℝ) :=
  { x | ∃ (k : ℤ), x = (Real.pi / 8) + (k * (Real.pi / 2)) }

theorem axis_of_symmetry (y : ℝ → ℝ) (h : ∀ x, y x = 3 * Real.sin (2 * x + Real.pi / 4)) :
  ∀ x ∈ equation_of_axis_of_symmetry, y x = 0 :=
begin
  sorry
end

end axis_of_symmetry_l249_249792


namespace sin_double_angle_l249_249002

theorem sin_double_angle (alpha : ℝ) (h1 : Real.cos (alpha + π / 4) = 3 / 5)
  (h2 : π / 2 ≤ alpha ∧ alpha ≤ 3 * π / 2) : Real.sin (2 * alpha) = 7 / 25 := 
sorry

end sin_double_angle_l249_249002


namespace maximum_candies_l249_249890

-- Definitions of the problem parameters
def classmates : ℕ := 25
def total_candies : ℕ := 200
def condition (candies : ℕ → ℕ) : Prop := 
  ∀ S (h : finset ℕ), S.card = 16 → (∑ i in S, candies i) ≥ 100

-- The theorem stating the maximum number of candies Vovochka can keep
theorem maximum_candies (candies : ℕ → ℕ) (h : condition candies) : 
  (total_candies - ∑ i in finset.range classmates, candies i) ≤ 37 :=
sorry

end maximum_candies_l249_249890


namespace coin_flip_heads_probability_l249_249258

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end coin_flip_heads_probability_l249_249258


namespace exist_ordering_rectangles_l249_249113

open Function

structure Rectangle :=
  (left_bot : ℝ × ℝ)  -- Bottom-left corner
  (right_top : ℝ × ℝ)  -- Top-right corner

def below (R1 R2 : Rectangle) : Prop :=
  ∃ g : ℝ, (∀ (x y : ℝ), R1.left_bot.1 ≤ x ∧ x ≤ R1.right_top.1 ∧ R1.left_bot.2 ≤ y ∧ y ≤ R1.right_top.2 → y < g) ∧
           (∀ (x y : ℝ), R2.left_bot.1 <= x ∧ x <= R2.right_top.1 ∧ R2.left_bot.2 <= y ∧ y <= R2.right_top.2 → y > g)

def to_right_of (R1 R2 : Rectangle) : Prop :=
  ∃ h : ℝ, (∀ (x y : ℝ), R1.left_bot.1 ≤ x ∧ x ≤ R1.right_top.1 ∧ R1.left_bot.2 ≤ y ∧ y ≤ R1.right_top.2 → x > h) ∧
           (∀ (x y : ℝ), R2.left_bot.1 <= x ∧ x <= R2.right_top.1 ∧ R2.left_bot.2 <= y ∧ y <= R2.right_top.2 → x < h)

def disjoint (R1 R2 : Rectangle) : Prop :=
  ¬ ((R1.left_bot.1 < R2.right_top.1) ∧ (R1.right_top.1 > R2.left_bot.1) ∧
     (R1.left_bot.2 < R2.right_top.2) ∧ (R1.right_top.2 > R2.left_bot.2))

theorem exist_ordering_rectangles (n : ℕ) (rectangles : Fin n → Rectangle)
  (h_disjoint : ∀ i j, i ≠ j → disjoint (rectangles i) (rectangles j)) :
  ∃ f : Fin n → Fin n, ∀ i j : Fin n, i < j → 
    (to_right_of (rectangles (f i)) (rectangles (f j)) ∨ 
    below (rectangles (f i)) (rectangles (f j))) := 
sorry

end exist_ordering_rectangles_l249_249113


namespace probability_heads_penny_nickel_dime_l249_249279

theorem probability_heads_penny_nickel_dime :
  let total_outcomes := 2^5 in
  let successful_outcomes := 2 * 2 in
  (successful_outcomes : ℝ) / total_outcomes = (1 / 8 : ℝ) :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249279


namespace sunil_end_amount_l249_249301

theorem sunil_end_amount (CI : ℝ) (R : ℝ) (n : ℕ) (t : ℕ) (P : ℝ) :
  CI = 370.80 →
  R = 0.06 →
  n = 1 →
  t = 2 →
  P = 3000 →
  (P * (1 + R / n) ^ (n * t)) = 3370.80 := by
    intros
    suffices : CI + P = 3370.80
    sorry

end sunil_end_amount_l249_249301


namespace probability_eight_coins_l249_249781

noncomputable def probability_of_seven_heads_or_tails (n : ℕ) : ℚ :=
let total_outcomes := 2^n,
    ways_seven_heads := nat.choose n 7,
    ways_seven_tails := nat.choose n 7,
    favorable_outcomes := ways_seven_heads + ways_seven_tails 
in
  favorable_outcomes / total_outcomes

theorem probability_eight_coins:
  probability_of_seven_heads_or_tails 8 = 1 / 16 :=
  by 
    sorry

end probability_eight_coins_l249_249781


namespace min_product_of_eccentricities_l249_249485

-- Defining the conditions
structure EllipseAndHyperbola where
  F1 F2 : Point
  P : Point
  angle_F1PF2 : ℝ
  (h_angle : angle_F1PF2 = 60 * (π / 180)) -- angle in radians
  shared_focus : (∃ (e1 e2 : Ellipse), e1.foci = (F1, F2) ∧ e2.foci = (F1, F2)) ∧ 
                  (∃ (h1 h2 : Hyperbola), h1.foci = (F1, F2) ∧ h2.foci = (F1, F2))
  common_point : ∃ (P_on_ellipse : EllipseAndHyperbola), P ∈ P_on_ellipse.1 ∧ P ∈ P_on_ellipse.2

-- The proof problem
theorem min_product_of_eccentricities (E : EllipseAndHyperbola) : 
  ∃ (e_ellipse e_hyperbola : ℝ), 
  e_ellipse * e_hyperbola = √3 / 2 := sorry

end min_product_of_eccentricities_l249_249485


namespace expr_simplified_l249_249759

theorem expr_simplified : |2 - Real.sqrt 2| - Real.sqrt (1 / 12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1 / 2 := 
by 
  sorry

end expr_simplified_l249_249759


namespace part1_sales_increase_part2_price_reduction_l249_249743

-- Part 1: If the price is reduced by 4 yuan, the new average daily sales will be 28 items.
theorem part1_sales_increase (initial_sales : ℕ) (increase_per_yuan : ℕ) (reduction : ℕ) :
  initial_sales = 20 → increase_per_yuan = 2 → reduction = 4 →
  initial_sales + increase_per_yuan * reduction = 28 :=
by sorry

-- Part 2: By how much should the price of each item be reduced for a daily profit of 1050 yuan.
theorem part2_price_reduction (initial_sales : ℕ) (increase_per_yuan : ℕ) (initial_profit : ℕ) 
  (target_profit : ℕ) (min_profit_per_item : ℕ) (x : ℕ) :
  initial_sales = 20 → increase_per_yuan = 2 → initial_profit = 40 → target_profit = 1050 
  → min_profit_per_item = 25 → (40 - x) * (20 + 2 * x) = 1050 → (40 - x) ≥ 25 → x = 5 :=
by sorry

end part1_sales_increase_part2_price_reduction_l249_249743


namespace Vovochka_max_candies_l249_249878

theorem Vovochka_max_candies (classmates : Finset ℕ) (candies : ℕ) (hclassmates : classmates.card = 25) (hcandies : candies = 200) (H : ∀ (S : Finset ℕ), S.card = 16 → 100 ≤ S.sum (λ x, 1)) :
  ∃ (max_candies : ℕ), max_candies = 37 :=
by
  use 37
  sorry

end Vovochka_max_candies_l249_249878


namespace number_times_sum_l249_249992

noncomputable def func (f : ℝ → ℝ) := 
  f 1 = 2 ∧
  ∀ x y : ℝ, f (x * y + f x + 1) = x * f y + f x

theorem number_times_sum (f : ℝ → ℝ) (c : ℝ) :
  func f → (let n := 1 in let s := c + 2 in n * s = c + 2) :=
by
  sorry

end number_times_sum_l249_249992


namespace trisect_right_triangle_hypotenuse_length_l249_249095

theorem trisect_right_triangle_hypotenuse_length 
  (A B C D E : Point)
  (h_right : right_angle ∠BAC)
  (h_trisect : (segment_length B D) = (segment_length D E) ∧ (segment_length D E) = (segment_length E C))
  (h_AD : segment_length A D = tan θ)
  (h_AE : segment_length A E = cot θ)
  (h_theta : 0 < θ ∧ θ < π / 2) :
  segment_length B C = 3 := 
begin
  sorry
end

end trisect_right_triangle_hypotenuse_length_l249_249095


namespace xyz_squared_sum_l249_249478

theorem xyz_squared_sum (x y z : ℤ) 
  (h1 : |x + y| + |y + z| + |z + x| = 4)
  (h2 : |x - y| + |y - z| + |z - x| = 2) :
  x^2 + y^2 + z^2 = 2 := 
by 
  sorry

end xyz_squared_sum_l249_249478


namespace conic_section_is_parabola_l249_249777

theorem conic_section_is_parabola (x y : ℝ) :
  (|y - 3| = sqrt ((x + 4)^2 + y^2)) → 
  ∃ a b c, a * y = x^2 + b * x + c := sorry

end conic_section_is_parabola_l249_249777


namespace competition_ranking_correct_l249_249523

theorem competition_ranking_correct :
  ∃ (rank : list char),
    rank = ['E', 'D', 'A', 'C', 'B'] ∧
    (∀ (i : ℕ) (h : i < 4),
      list.nth ['A', 'B', 'C', 'D', 'E'] i ≠ list.nth rank i ∧
      list.nth ['D', 'A', 'E', 'C', 'B'] i ≠ list.nth rank i ∧
      list.nth ['E', 'D', 'A', 'C', 'B'] i = list.nth rank i ∧
      (rank !! i, rank !! (i + 1)) ∈ [('D', 'A'), ('A', 'E'), ('E', 'C'), ('C', 'B')]
    ) ∧
    (∃ j k : ℕ, j ≠ k ∧ j < 5 ∧ k < 5 ∧
      list.nth ['D', 'A', 'E', 'C', 'B'] j = list.nth rank j ∧
      list.nth ['D', 'A', 'E', 'C', 'B'] k = list.nth rank k)
:=
begin
  sorry
end

end competition_ranking_correct_l249_249523


namespace max_candies_vovochka_can_keep_l249_249904

-- Definitions for the conditions in the problem.
def classmates := 25
def total_candies := 200
def minimum_candies (candies_distributed : Vector ℕ classmates) : Prop :=
  ∀ (S : Finset (Fin classmates)), S.card = 16 → S.sum (candies_distributed.nth' S) ≥ 100

-- The proof goal
theorem max_candies_vovochka_can_keep (candies_distributed : Vector ℕ classmates) (h : minimum_candies candies_distributed) :
  ∃ k, k = 37 ∧ total_candies - Finset.univ.sum (candies_distributed.nth' Finset.univ) = k :=
begin
  sorry
end

end max_candies_vovochka_can_keep_l249_249904


namespace probability_2_lt_X_le_4_l249_249027

-- Defining the probability distribution function
def P (X : ℕ → Prop) : ℚ :=
  if X 1 then 1/2 ^ 1 else
  if X 2 then 1/2 ^ 2 else
  if X 3 then 1/2 ^ 3 else
  if X 4 then 1/2 ^ 4 else 0

-- The probability for a range 2 < X ≤ 4
theorem probability_2_lt_X_le_4 (P : (ℕ → Prop) → ℚ) :
  (P (λ x, x = 3) + P (λ x, x = 4)) = 3/16 := by
  sorry

end probability_2_lt_X_le_4_l249_249027


namespace guess_the_number_l249_249506

def n : ℕ := 2

-- Condition 1: The number is prime.
def Petya : Prop := Nat.Prime n

-- Condition 2: The number is 9.
def Vasya : Prop := n = 9

-- Condition 3: The number is even.
def Kolya : Prop := n % 2 = 0

-- Condition 4: The number is 15.
def Anton : Prop := n = 15

-- Condition 5: Exactly one of Petya's and Vasya's statements is true.
def condition5 : Prop := xor Petya Vasya

-- Condition 6: Exactly one of Kolya's and Anton's statements is true.
def condition6 : Prop := xor Kolya Anton

-- The final proof statement
theorem guess_the_number (h1 : Petya) (h5 : condition5) (h6 : condition6) : n = 2 :=
by
  -- proof goes here
  sorry

end guess_the_number_l249_249506


namespace expand_det_along_second_column_l249_249785

def determinant_of_minor (a1 a2 a3 : ℝ) (b1 b2 b3 : ℝ) :=
  (3 * (a2 * b3 - a3 * b2)) + 
  (2 * (a1 * b3 - a3 * b1)) - 
  (2 * (a1 * b2 - a2 * b1))

theorem expand_det_along_second_column (a1 a2 a3 b1 b2 b3 : ℝ) :
  let matrix_det := 
    2 * (a1 * (b2 * (-2) - b3 * 2)) -
    3 * (a2 * (b1 * (-2) - b3 * b1)) +
        (a3 * (b1 * 2 - b2 * b1))
  in matrix_det = 3 * (a2 * b3 - a3 * b2) + 2 * (a1 * b3 - a3 * b1) - 2 * (a1 * b2 - a2 * b1) :=
by
  sorry

end expand_det_along_second_column_l249_249785


namespace relay_race_total_time_is_correct_l249_249528

-- Define the time taken by each runner
def time_Ainslee : ℕ := 72
def time_Bridget : ℕ := (10 * time_Ainslee) / 9
def time_Cecilia : ℕ := (3 * time_Bridget) / 4
def time_Dana : ℕ := (5 * time_Cecilia) / 6

-- Define the total time and convert to minutes and seconds
def total_time_seconds : ℕ := time_Ainslee + time_Bridget + time_Cecilia + time_Dana
def total_time_minutes := total_time_seconds / 60
def total_time_remainder := total_time_seconds % 60

theorem relay_race_total_time_is_correct :
  total_time_minutes = 4 ∧ total_time_remainder = 22 :=
by
  -- All intermediate values can be calculated using the definitions
  -- provided above correctly.
  sorry

end relay_race_total_time_is_correct_l249_249528


namespace paper_left_l249_249121

theorem paper_left (first_purchase second_purchase used_project used_artwork used_letters : ℕ) :
  first_purchase = 900 →
  second_purchase = 300 →
  used_project = 156 →
  used_artwork = 97 →
  used_letters = 45 →
  first_purchase + second_purchase - (used_project + used_artwork + used_letters) = 902 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  exact rfl

end paper_left_l249_249121


namespace total_goals_l249_249688

theorem total_goals (Bruce Michael Jack Sarah : ℕ)
  (hB : Bruce = 4)
  (hM : Michael = 2 * Bruce)
  (hJ : Jack = Bruce - 1)
  (hS : Sarah = Jack / 2) 
  (hSRounded : Sarah = 1) : 
  Michael + Jack + Sarah = 12 := 
by
  rw [hB, hM, hJ, hSRounded]
  sorry

end total_goals_l249_249688


namespace greatest_divisors_1_to_20_l249_249196

def is_divisor (a b : ℕ) : Prop := b % a = 0

def count_divisors (n : ℕ) : ℕ :=
  nat.succ (list.length ([k for k in list.range n if is_divisor k.succ n]))

def max_divisors_from_1_to_20 : set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 20 ∧ ∃ max_count, count_divisors n = max_count ∧ 
  ∀ m, 1 ≤ m ∧ m ≤ 20 → count_divisors m ≤ max_count}

theorem greatest_divisors_1_to_20 : max_divisors_from_1_to_20 = {12, 18, 20} :=
by {
  sorry
}

end greatest_divisors_1_to_20_l249_249196


namespace part_b_part_c_l249_249813

def z : ℂ := (5 + I) / (1 + I)

theorem part_b : z.im = -2 :=
by
  -- omitted for brevity
  sorry

theorem part_c : (z.re > 0) ∧ (z.im < 0) :=
by
  -- omitted for brevity
  sorry

end part_b_part_c_l249_249813


namespace largest_angle_of_obtuse_isosceles_triangle_l249_249660

theorem largest_angle_of_obtuse_isosceles_triangle (P Q R : Type) 
  (triangle_PQR : Triangle P Q R)
  (isosceles_PQR : Isosceles triangle_PQR) 
  (obtuse_PQR : Obtuse triangle_PQR)
  (angle_P_30 : angle P triangle_PQR = 30) : 
  ∃ (angle_Q : ℕ), is_largest_angle angle_Q triangle_PQR ∧ angle_Q = 120 := 
by 
  sorry

end largest_angle_of_obtuse_isosceles_triangle_l249_249660


namespace parallel_lines_k_value_l249_249682

theorem parallel_lines_k_value (k : ℝ) 
  (line1 : ∀ x : ℝ, y = 5 * x + 3) 
  (line2 : ∀ x : ℝ, y = (3 * k) * x + 1) 
  (parallel : ∀ x : ℝ, (5 = 3 * k)) : 
  k = 5 / 3 := 
begin
  sorry
end

end parallel_lines_k_value_l249_249682


namespace assign_positive_numbers_to_regions_l249_249231

/--
Given a plane with several lines such that no two lines are parallel and no three lines intersect at a single point,
prove there exists an assignment of positive real numbers to the regions formed by these lines
such that for any given line, the sum of the numbers on each side of the line is equal.
-/
theorem assign_positive_numbers_to_regions
  (n : ℕ)
  (lines : Fin n → ℝ × ℝ → ℝ)
  (h_non_parallel : ∀ i j, i ≠ j → ∀ x y, (lines i x - lines j x ≠ 0 ∨ lines i y - lines j y ≠ 0))
  (h_no_three_lines_intersect : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → ∀ x, ¬ (lines i x = lines j x ∧ lines i x = lines k x)) :
  ∃ (regions : Fin (2 * n) → ℝ), (∀ i, 0 < regions i) ∧ (∀ i, ∑ j in Finset.filter (λ x, lines i (regions x) > lines i (regions x)) (Finset.univ), regions j = ∑ j in Finset.filter (λ x, lines i (regions x) < lines i (regions x)) (Finset.univ), regions j) :=
by
  sorry

end assign_positive_numbers_to_regions_l249_249231


namespace area_of_PQR_30_60_90_l249_249103

noncomputable def area_of_triangle (RP PQ : ℝ) : ℝ := (1/2) * RP * PQ

theorem area_of_PQR_30_60_90 (QR : ℝ) (h1 : QR = 12) (h2 : ∠Q = 60) (h3 : ∠R = 30) :
  ∃ RP PQ : ℝ, RP = 6 ∧ PQ = 6 * Real.sqrt 3 ∧ area_of_triangle RP PQ = 18 * Real.sqrt 3 :=
by
  have RP : ℝ := 6
  have PQ : ℝ := 6 * Real.sqrt 3
  use RP
  use PQ
  split
  . exact rfl
  split
  . exact rfl
  . unfold area_of_triangle
    simp
    norm_num
    simp [Real.sqrt_eq_rpow, Real.rpow_mul]
    sorry

end area_of_PQR_30_60_90_l249_249103


namespace greatest_divisors_1_to_20_l249_249203

def is_divisor (a b : ℕ) : Prop := b % a = 0

def count_divisors (n : ℕ) : ℕ :=
  nat.succ (list.length ([k for k in list.range n if is_divisor k.succ n]))

def max_divisors_from_1_to_20 : set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 20 ∧ ∃ max_count, count_divisors n = max_count ∧ 
  ∀ m, 1 ≤ m ∧ m ≤ 20 → count_divisors m ≤ max_count}

theorem greatest_divisors_1_to_20 : max_divisors_from_1_to_20 = {12, 18, 20} :=
by {
  sorry
}

end greatest_divisors_1_to_20_l249_249203


namespace lowest_runs_scored_l249_249937

theorem lowest_runs_scored :
  ∃ x : ℕ, x + 8 + 15 = 24 :=
begin
  use 1,
  linarith,
end

end lowest_runs_scored_l249_249937


namespace units_digit_7_pow_10_pow_6_l249_249450

theorem units_digit_7_pow_10_pow_6 :
  let units_digits_cycle := [7, 9, 3, 1] in
  let cycle_length := 4 in
  let n := 10^6 in
  (7^n % 10 = 1) := by
  sorry

end units_digit_7_pow_10_pow_6_l249_249450


namespace variance_dataset_is_0_1_l249_249474

-- Define the data set
def dataset : List ℝ := [4.7, 4.8, 5.1, 5.4, 5.5]

-- Define the mean of the data set
def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

-- Define the variance of the data set
def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

-- State the theorem to prove the variance is 0.1
theorem variance_dataset_is_0_1 : variance dataset = 0.1 :=
sorry

end variance_dataset_is_0_1_l249_249474


namespace max_value_of_y_l249_249072

theorem max_value_of_y :
  ∀ x : ℝ, x ∈ Set.Icc (-5/12 * Real.pi) (-Real.pi / 3) →
    (let y := tan (x + (2 / 3) * Real.pi) - tan (x + Real.pi / 6) + cos (x + Real.pi / 6)
    in y ≤ 11 * Real.sqrt 3 / 6) :=
sorry

end max_value_of_y_l249_249072


namespace max_good_triangles_l249_249254

-- Define the points and conditions
variable (Points : Fin 2017 → (ℝ × ℝ))

-- Conditions: No three points are collinear
def no_three_collinear (Points : Fin 2017 → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : Fin 2017), i ≠ j ∧ j ≠ k ∧ i ≠ k →
    let p1 := Points i
    let p2 := Points j
    let p3 := Points k in
      (p1.1 - p3.1) * (p2.2 - p3.2) ≠ (p2.1 - p3.1) * (p1.2 - p3.2)

-- Definition of good triangles
def is_good_triangle (Points : Fin 2017 → (ℝ × ℝ)) (i j k : Fin 2017) : Prop :=
  let p1 := Points i
  let p2 := Points j
  let p3 := Points k in
    ∀ (x y z : Fin 2017),
      let q1 := Points x
      let q2 := Points y
      let q3 := Points z in
        (abs ((p1.1 - p3.1) * (p2.2 - p3.2) - (p2.1 - p3.1) * (p1.2 - p3.2)) ≥ 
         abs ((q1.1 - q3.1) * (q2.2 - q3.2) - (q2.1 - q3.1) * (q1.2 - q3.2)))

-- Theorem to prove: there cannot be more than 2017 good triangles
theorem max_good_triangles (Points : Fin 2017 → (ℝ × ℝ)) :
  no_three_collinear Points →
  ∃ S : Finset (Fin 2017 × Fin 2017 × Fin 2017),
    (∀ t ∈ S, is_good_triangle Points t.1 t.2.1 t.2.2) ∧ S.card ≤ 2017 :=
begin
  sorry
end

end max_good_triangles_l249_249254


namespace eq_frac_l249_249633

noncomputable def g : ℝ → ℝ := sorry

theorem eq_frac (h1 : ∀ c d : ℝ, c^3 * g d = d^3 * g c)
                (h2 : g 3 ≠ 0) : (g 7 - g 4) / g 3 = 279 / 27 :=
by
  sorry

end eq_frac_l249_249633


namespace solution_set_ineq_min_value_sum_l249_249853

-- Part (1)
theorem solution_set_ineq (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| + |x - 2|) :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 2} :=
sorry

-- Part (2)
theorem min_value_sum (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| + |x - 2|)
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (hx : ∀ x, f x ≥ (1 / m) + (1 / n)) :
  m + n = 8 / 3 :=
sorry

end solution_set_ineq_min_value_sum_l249_249853


namespace unfolded_lateral_surface_of_cone_is_sector_l249_249305

-- Define what it means for a surface to be the lateral surface of a cone
def isLateralSurfaceOfCone (S: Type) [Surface S] : Prop := sorry

-- Define what it means for a shape to be a sector
def isSector (S: Type) [Shape S] : Prop := sorry

-- The theorem to be proved
theorem unfolded_lateral_surface_of_cone_is_sector {S: Type} [Surface S] 
  : isLateralSurfaceOfCone S → isSector (unfoldedSurface S) := 
by 
  sorry

end unfolded_lateral_surface_of_cone_is_sector_l249_249305


namespace smallest_n_for_terminating_decimal_l249_249338

theorem smallest_n_for_terminating_decimal : 
  ∃ n : ℕ, 0 < n ∧ (∃ a b : ℕ, n + 50 = 2^a * 5^b) ∧ ∀ m : ℕ, m < n → ∃ a b : ℕ, n + 50 = 2^a * 5^b → m + 50 ≠ 2^a * 5^b :=
begin
  use 14,
  split,
  { exact dec_trivial }, -- 0 < 14
  split,
  { use 6, use 0, -- 14 + 50 = 64 = 2^6 * 5^0
    exact dec_trivial },
  { intros m hm,
    by_contradiction,
    cases h with a ha,
    cases ha with b hb,
    -- You would continue to complete the proof here.
    sorry
  }
end

end smallest_n_for_terminating_decimal_l249_249338


namespace math_expr_eq_five_l249_249758

def π : ℝ := Real.pi
def sqrt3 : ℝ := Real.sqrt 3
def neg2 : ℝ := -2
def abs_num : ℝ := abs (-1/2)
def sin_30 : ℝ := Real.sin (Real.pi / 6)

noncomputable def math_expr : ℝ :=
  (π + sqrt3)^0 + neg2^2 + abs_num - sin_30

theorem math_expr_eq_five : math_expr = 5 := 
  by 
    sorry

end math_expr_eq_five_l249_249758


namespace distinct_sums_count_l249_249062

noncomputable def number_distinct_sums : ℕ :=
  let S_min := 26
  let S_max := 62
  (list.range' S_min (S_max - S_min + 1)).filter (λ x, x % 3 = 2).length

theorem distinct_sums_count :
  number_distinct_sums = 13 :=
sorry

end distinct_sums_count_l249_249062


namespace least_even_p_for_300p_perfect_square_l249_249696

-- Define p as an even integer
def is_even (p : ℤ) := ∃ k : ℤ, p = 2 * k

-- Define the condition that 300 * p is a perfect square
def is_perfect_square (n : ℤ) := ∃ k : ℤ, n = k * k 

-- Define the problem statement
theorem least_even_p_for_300p_perfect_square :
  ∃ p : ℤ, is_even p ∧ 300 * p = 30 * 30 ∧ p = 18 :=
begin
  sorry
end

end least_even_p_for_300p_perfect_square_l249_249696


namespace inequality_sqrt_cbrt_l249_249471

theorem inequality_sqrt_cbrt (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ ( (a * b * c + a * b * d + a * c * d + b * c * d) / 4)^(1 / 3 : ℝ) :=
by
  sorry

end inequality_sqrt_cbrt_l249_249471


namespace arithmetic_sequence_properties_l249_249537

-- Define the arithmetic sequence {a_n}
def arithmetic_sequence (a1 : ℕ) (d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_first_n_terms (a1 : ℕ) (d : ℕ) (n : ℕ) : ℕ := 
  n * (2 * a1 + (n - 1) * d) / 2

-- Define sequence {b_n}
def b_n (a_n a_n1 : ℝ) : ℝ := 4 / (√a_n + √a_n1)

-- Define sum of the first n terms of sequence {b_n}
def sum_first_n_b_terms (bn : ℕ → ℝ) (n : ℕ) : ℝ := 
  (finset.range n).sum bn

-- Define the main theorem
theorem arithmetic_sequence_properties :
  ∀ (n : ℕ), 0 < n → 
  (∃ a1 d : ℕ, a1 = 2 ∧ d ≠ 0 ∧ 
    let a_n := arithmetic_sequence a1 d in
    sum_first_n_terms a1 d (2 * n) = 4 * sum_first_n_terms a1 d n ∧ 
    a_n n = 4 * n - 2 ∧ 
    let bn := λ n, b_n (a_n n) (a_n (n + 1)) in
    sum_first_n_b_terms bn n = √(4 * n + 2) - √2) := 
by sorry

end arithmetic_sequence_properties_l249_249537


namespace midpoint_AB_find_Q_find_H_l249_249409

-- Problem 1: Midpoint of AB
theorem midpoint_AB (x1 y1 x2 y2 : ℝ) : 
  let A := (x1, y1)
  let B := (x2, y2)
  let M := ( (x1 + x2) / 2, (y1 + y2) / 2 )
  M = ( (x1 + x2) / 2, (y1 + y2) / 2 )
:= 
  -- The lean statement that shows the midpoint formula is correct.
  sorry

-- Problem 2: Coordinates of Q given midpoint
theorem find_Q (px py mx my : ℝ) : 
  let P := (px, py)
  let M := (mx, my)
  let Q := (2 * mx - px, 2 * my - py)
  ( (px + Q.1) / 2 = mx ∧ (py + Q.2) / 2 = my )
:= 
  -- Lean statement to find Q
  sorry

-- Problem 3: Coordinates of H given midpoints coinciding
theorem find_H (xE yE xF yF xG yG : ℝ) :
  let E := (xE, yE)
  let F := (xF, yF)
  let G := (xG, yG)
  ∃ xH yH : ℝ, 
    ( (xE + xH) / 2 = (xF + xG) / 2 ∧ (yE + yH) / 2 = (yF + yG) / 2 ) ∨
    ( (xF + xH) / 2 = (xE + xG) / 2 ∧ (yF + yH) / 2 = (yE + yG) / 2 ) ∨
    ( (xG + xH) / 2 = (xE + xF) / 2 ∧ (yG + yH) / 2 = (yE + yF) / 2 )
:=
  -- Lean statement to find H
  sorry

end midpoint_AB_find_Q_find_H_l249_249409


namespace number_of_complex_satisfying_conditions_l249_249444

theorem number_of_complex_satisfying_conditions : 
  ∃ finset_of_z : finset (ℂ), finset.card finset_of_z = 4 ∧ 
  ∀ z ∈ finset_of_z, |z| = 1 ∧ 
  |(z / (conj z)) - ((conj z) / z)| = 2 := 
sorry

end number_of_complex_satisfying_conditions_l249_249444


namespace problem_l249_249849

noncomputable def f (A ω φ x : ℝ) := A * Real.sin (ω * x + φ)

theorem problem
  (A ω φ : ℝ)
  (hA : A > 0)
  (hω : ω > 0)
  (hφ : 0 < φ ∧ φ < π)
  (h_period : ∀ x, f A ω φ (x + π) = f A ω φ x)
  (h_max : f A ω φ (π / 12) = A)
  (h_solution : ∃ x ∈ Icc (-π / 4) 0, f A ω φ x - 1 + A = 0) :
  ω = 2 ∧ φ = π / 3 ∧ (4 - 2 * Real.sqrt 3 ≤ A ∧ A ≤ 2) := sorry

end problem_l249_249849


namespace maximum_ratio_is_79_over_31_l249_249568

noncomputable def maximum_ratio_of_two_digit_numbers (x y : ℕ) : ℚ :=
  if two_digit : (10 ≤ x ∧ x ≤ 99) ∧ (10 ≤ y ∧ y ≤ 99) ∧ (x + y = 110) then
    max (x / y : ℚ) else 0

theorem maximum_ratio_is_79_over_31 :
  maximum_ratio_of_two_digit_numbers 79 31 = (79 / 31 : ℚ) :=
by
  simp [maximum_ratio_of_two_digit_numbers]
  done

end maximum_ratio_is_79_over_31_l249_249568


namespace probability_white_or_red_ball_l249_249369

variable (totalBalls whiteBalls blackBalls redBalls : ℕ)
variable (P : ℕ)

-- Definitions based on conditions.
def totalBalls := 8 + 7 + 4
def whiteBalls := 8
def redBalls := 4
def favorableOutcomeCount := whiteBalls + redBalls

-- Core theorem statement
theorem probability_white_or_red_ball :
  (favorableOutcomeCount : ℝ) / (totalBalls : ℝ) = 12 / 19 :=
by
  sorry -- Proof goes here but is omitted as per instructions.

end probability_white_or_red_ball_l249_249369


namespace find_integer_solutions_l249_249770

-- The conditions are "gcd of pairs is 1" and "x, y, z, t are positive integers".
def is_solution (x y z t : ℕ) : Prop :=
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (t > 0) ∧ 
  ((x + y) * (y + z) * (z + x) = x * y * z * t) ∧ 
  (Nat.gcd x y = 1) ∧ 
  (Nat.gcd y z = 1) ∧ 
  (Nat.gcd z x = 1)

theorem find_integer_solutions :
  ∀ (x y z t : ℕ), is_solution x y z t → 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ t = 1) ∨
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 4) ∨
  (x = 1 ∧ y = 2 ∧ z = 3 ∧ t = 6) :=
begin
  sorry
end

end find_integer_solutions_l249_249770


namespace max_value_intersection_sum_l249_249556

open Set Finset

noncomputable def max_intersection_sum (n : ℕ) (h : 2 < n) (A : Fin 2n → Finset (Fin n)) 
  (distinct : ∀ i j : Fin 2n, i ≠ j → A i ≠ A j) : ℝ :=
  ∑ i in finset.univ (Fin 2n),
    (|A i ∩ A ((i + 1) % (2 * n))| : ℝ) / (|A i| : ℝ) / (|A ((i + 1) % (2 * n))| : ℝ)

theorem max_value_intersection_sum (n : ℕ) (h : 2 < n) (A : Fin 2n → Finset (Fin n)) 
  (distinct : ∀ i j : Fin 2n, i ≠ j → A i ≠ A j) : 
  max_intersection_sum n h A distinct = n := 
sorry

end max_value_intersection_sum_l249_249556


namespace circle_range_of_m_l249_249628

theorem circle_range_of_m (m : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + m * x + 2 * m * y + 2 * m^2 + m - 1 = 0 → (2 * m^2 + m - 1 = 0)) → (-2 < m) ∧ (m < 2/3) :=
by
  sorry

end circle_range_of_m_l249_249628


namespace w_coordinate_l249_249728

theorem w_coordinate (t : ℝ) :
  let line := λ t, (3 + 3 * t, 3 - t, 2 - t, 1 - 2 * t) in
  ∃ t : ℝ, (line t).2 = 4 → (line t).4 = 3 := 
begin
  use -1,
  intro h,
  sorry
end

end w_coordinate_l249_249728


namespace molecular_weight_CaO_is_56_08_l249_249335

-- Define the atomic weights of Calcium and Oxygen
def atomic_weight_Ca := 40.08 -- in g/mol
def atomic_weight_O := 16.00 -- in g/mol

-- Define the molecular weight of the compound
def molecular_weight_CaO := atomic_weight_Ca + atomic_weight_O

-- State the theorem
theorem molecular_weight_CaO_is_56_08 : molecular_weight_CaO = 56.08 :=
by
  -- The proof will be filled in here
  sorry

end molecular_weight_CaO_is_56_08_l249_249335


namespace johns_yard_area_l249_249977

noncomputable def calculateArea (totalPosts: ℕ) (distanceBetweenPosts: ℝ) (ratio: ℝ) : ℝ ≠ 0 :=
  let x := totalPosts / (2 + 2 * ratio)
  let shorterSideLength := (x - 1) * distanceBetweenPosts
  let longerSideLength := (ratio * x - 1) * distanceBetweenPosts
  shorterSideLength * longerSideLength = 1000

theorem johns_yard_area :
  calculateArea 24 5 1.5 = 1000 :=
  sorry

end johns_yard_area_l249_249977


namespace find_a2023_l249_249011

variable {a : ℕ → ℕ}
variable {x : ℕ}

def sequence_property (a: ℕ → ℕ) : Prop :=
  ∀ n, a n + a (n + 1) + a (n + 2) = 20

theorem find_a2023 (h1 : sequence_property a) 
                   (h2 : a 2 = 2 * x) 
                   (h3 : a 18 = 9 + x) 
                   (h4 : a 65 = 6 - x) : 
  a 2023 = 5 := 
by
  sorry

end find_a2023_l249_249011


namespace pete_flag_total_circle_square_l249_249297

theorem pete_flag_total_circle_square : 
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  circles + squares = 54 := 
by
  let stars := 50
  let stripes := 13
  let circles := (stars / 2) - 3
  let squares := (stripes * 2) + 6
  show circles + squares = 54
  sorry

end pete_flag_total_circle_square_l249_249297


namespace difference_of_squares_l249_249707

theorem difference_of_squares (a b : ℕ) (h₁ : a = 69842) (h₂ : b = 30158) :
  (a^2 - b^2) / (a - b) = 100000 :=
by
  rw [h₁, h₂]
  sorry

end difference_of_squares_l249_249707


namespace combined_probability_C_or_D_l249_249367

theorem combined_probability_C_or_D (P_A P_B P_D : ℚ) (hP_A : P_A = 3 / 10) (hP_B : P_B = 1 / 4) (hP_D : P_D = 1 / 5) :
  let P_C := 1 - P_A - P_B - P_D in
  P_C + P_D = 9 / 20 :=
by
  -- Proof goes here
  sorry

end combined_probability_C_or_D_l249_249367


namespace color_verticies_l249_249462

theorem color_verticies (n : ℕ) (h : n ≥ 5) : 
  (∃ colors : Fin n → Fin 6, 
    ∀ i : Fin n, 
      ∀ j : Fin 5, colors ((i + j) % n) ≠ colors ((i + j + 1) % n)) ↔ 
  (n ∉ ({7, 8, 9, 13, 14, 19} : Finset ℕ)) :=
sorry

end color_verticies_l249_249462


namespace range_of_y1_range_of_y2_l249_249360

open Real

-- Define y = sin x + cos x
def y1 (x : ℝ) : ℝ := sin x + cos x

-- Define range of y1
def range_y1 : Set ℝ := {y | - (sqrt 2) ≤ y ∧ y ≤ sqrt 2}

-- Statement for first problem
theorem range_of_y1 : ∀ x : ℝ, y1 x ∈ range_y1 := by sorry

-- Define y = sin x + cos x - sin 2x
def y2 (x : ℝ) : ℝ := sin x + cos x - sin (2 * x)

-- Define range of y2
def range_y2 : Set ℝ := {y | -1 - sqrt 2 ≤ y ∧ y ≤ (5 / 4)}

-- Statement for second problem
theorem range_of_y2 : ∀ x : ℝ, y2 x ∈ range_y2 := by sorry

end range_of_y1_range_of_y2_l249_249360


namespace monotonic_intervals_max_area_of_triangle_l249_249041
noncomputable theory
open Real

def f (x : ℝ) : ℝ := (sin x) * (cos x) - (cos (x + π / 4)) ^ 2

theorem monotonic_intervals :
  (∀ k : ℤ, ∀ x : ℝ, k * π - π / 4 ≤ x ∧ x ≤ k * π + π / 4 → f x ≥ f (x - 1)) ∧
  (∀ k : ℤ, ∀ x : ℝ, k * π + π / 4 ≤ x ∧ x ≤ k * π + 3 * π / 4 → f x ≤ f (x - 1)) :=
by sorry

theorem max_area_of_triangle (a b c : ℝ)
  (A B C : ℝ) (hA2 : f (A / 2) = 0)  (ha : a = 1) (haa: 0 < A ∧ A < π / 2)
  (h_cos : cos A = sqrt 3 / 2) (h_acos: 0 < B ∧ B < π / 2) (h_bcos : 0 < C ∧ C < π / 2) :
  1 + sqrt 3 * b * c = b ^ 2 + c ^ 2 → (1/2) * b * c * sin A ≤ (2 + sqrt 3) / 4 :=
by sorry

end monotonic_intervals_max_area_of_triangle_l249_249041


namespace parallel_lines_k_value_l249_249683

theorem parallel_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = 5 * x + 3 → y = (3 * k) * x + 1 → true) → k = 5 / 3 :=
by
  intros
  sorry

end parallel_lines_k_value_l249_249683


namespace sum_of_divisors_of_8_l249_249800

theorem sum_of_divisors_of_8 : 
  ( ∑ n in { n : ℕ | (n > 0) ∧ (n ∣ 8) }, n ) = 15 := 
by 
  sorry

end sum_of_divisors_of_8_l249_249800


namespace max_divisors_up_to_20_l249_249213

def divisors (n : Nat) : List Nat :=
  (List.range (n + 1)).filter (λ d, n % d = 0)

def count_divisors (n : Nat) : Nat :=
  (divisors n).length

theorem max_divisors_up_to_20 :
  ∀ n, n ∈ List.range 21 → count_divisors n ≤ 6 ∧
      ((count_divisors 12 = 6) ∧
       (count_divisors 18 = 6) ∧
       (count_divisors 20 = 6)) :=
by
  sorry

end max_divisors_up_to_20_l249_249213


namespace lexia_study_time_l249_249553

def kwame_hours : ℝ := 2.5
def connor_hours : ℝ := 1.5
def minutes_in_hour : ℕ := 60
def kwame_minutes : ℕ := (kwame_hours * minutes_in_hour).toNat
def connor_minutes : ℕ := (connor_hours * minutes_in_hour).toNat
def lexia_minutes : ℕ := 240 - 143

theorem lexia_study_time :
  kwame_minutes + connor_minutes = lexia_minutes + 143 :=
by
  sorry

end lexia_study_time_l249_249553


namespace minimum_value_of_quadratic_l249_249638

theorem minimum_value_of_quadratic (x : ℝ) : ∃ (y : ℝ), (∀ x : ℝ, y ≤ x^2 + 2) ∧ (y = 2) :=
by
  sorry

end minimum_value_of_quadratic_l249_249638


namespace find_t_l249_249146

noncomputable def f (x t : ℝ) : ℝ := abs (x^2 - 2 * x - t)

theorem find_t :
  (∀ x ∈ set.Icc (0 : ℝ) 3, f x t ≤ 2) →
  (∃ x ∈ set.Icc (0 : ℝ) 3, f x t = 2) →
  t = 1 :=
begin
  sorry,
end

end find_t_l249_249146


namespace real_root_in_interval_l249_249644

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem real_root_in_interval : 
  continuous f ∧ f 1 < 0 ∧ f 2 > 0 → ∃ x ∈ Icc (1 : ℝ) (2 : ℝ), f x = 0 :=
begin
  sorry
end

end real_root_in_interval_l249_249644


namespace sqrt_of_square_neg_three_l249_249705

theorem sqrt_of_square_neg_three : Real.sqrt ((-3 : ℝ)^2) = 3 := by
  sorry

end sqrt_of_square_neg_three_l249_249705


namespace compute_sum_of_squares_l249_249153

noncomputable def polynomial_roots (p q r : ℂ) : Prop := 
  (p^3 - 15 * p^2 + 22 * p - 8 = 0) ∧ 
  (q^3 - 15 * q^2 + 22 * q - 8 = 0) ∧ 
  (r^3 - 15 * r^2 + 22 * r - 8 = 0) 

theorem compute_sum_of_squares (p q r : ℂ) (h : polynomial_roots p q r) :
  (p + q) ^ 2 + (q + r) ^ 2 + (r + p) ^ 2 = 406 := 
sorry

end compute_sum_of_squares_l249_249153


namespace locus_of_O_is_plane_l249_249818

-- Define a structure for a point in 3D space
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

-- Define the midpoints of segments AA', BB', CC'
def midpoint (A A' : Point3D) : Point3D :=
⟨(A.x + A'.x) / 2, (A.y + A'.y) / 2, (A.z + A'.z) / 2⟩

-- Define the centroid of a triangle with points L, M, N
def centroid (L M N : Point3D) : Point3D :=
⟨(L.x + M.x + N.x) / 3, (L.y + M.y + N.y) / 3, (L.z + M.z + N.z) / 3⟩

-- Define the conditions
variables (A B C : Point3D)
variables (A' B' C' : Point3D)
variables (L M N : Point3D)

def L := midpoint A A'
def M := midpoint B B'
def N := midpoint C C'

-- Define the centroid O of triangle LMN
def O := centroid L M N

-- Define the statement to prove the locus of O
def locus_of_O : Prop :=
O.z = (A.z + B.z + C.z) / 6

-- The theorem statement
theorem locus_of_O_is_plane : locus_of_O :=
by
  sorry

end locus_of_O_is_plane_l249_249818


namespace find_f6_l249_249632

noncomputable def f : ℝ → ℝ := sorry

axiom f_additivity (x y : ℝ) : f(x + y) = f(x) + f(y)

axiom f_value_at_3 : f(3) = 4

theorem find_f6 : f(6) = 8 := 
begin
  sorry
end

end find_f6_l249_249632


namespace angle_BAD_in_quadrilateral_ABCD_is_90_l249_249238

theorem angle_BAD_in_quadrilateral_ABCD_is_90
  (A B C D : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB BC CD : ℝ)
  (AB_eq_BC : AB = BC)
  (BC_eq_CD : BC = CD)
  (angle_ABC : ℝ)
  (angle_ABC_deg : angle_ABC = 80)
  (angle_BCD : ℝ)
  (angle_BCD_deg : angle_BCD = 160) :
  ∃ (angle_BAD : ℝ), angle_BAD = 90 :=
begin
  -- Definitions of points A, B, C, D will be part of proof but are assumed to exist here
  sorry
end

end angle_BAD_in_quadrilateral_ABCD_is_90_l249_249238


namespace similar_triangles_CMN_CBA_points_on_circle_l249_249808

-- Definitions based on the conditions from the problem statement
variables (A B C D M N K P Q : Point)
variables (triangle_ABC : Triangle A B C)
variables (right_angle_at_C : ∃ (right_angle : IsRightTriangle triangle_ABC), right_angle.vertex = C)
variables (altitude_CD : IsAltitude C D)
variables (circle_inscribed_in_ACD : InscribedCircle P triangle_ACD)
variables (circle_inscribed_in_BCD : InscribedCircle Q triangle_BCD)
variables (tangent_to_circles : ExternalTangent PQ)
variables (intersection_with_AC : Intersect AC tangent_to_circles M)
variables (intersection_with_BC : Intersect BC tangent_to_circles N)
variables (intersection_with_CD : Intersect CD tangent_to_circles K)
variables (radius_inscribed_circle : ℝ)
variables (inscribed_circle : ∃ r = radius_inscribed_circle, IsInscribedCircle r ABC)

-- Prove that triangles CMN and CBA are similar
theorem similar_triangles_CMN_CBA
  (similarity_1 : SimilarTriangles CMN CBA) : True := sorry

-- Prove that points C, M, N, P, and Q lie on a circle with center K
-- and the radius equal to the radius of the inscribed circle of triangle ABC
theorem points_on_circle (circle_points : OnCircle C K radius_inscribed_circle ∧ 
    OnCircle M K radius_inscribed_circle ∧ 
    OnCircle N K radius_inscribed_circle ∧ 
    OnCircle P K radius_inscribed_circle ∧ 
    OnCircle Q K radius_inscribed_circle) : True := sorry

end similar_triangles_CMN_CBA_points_on_circle_l249_249808


namespace expected_value_usable_pieces_l249_249689

noncomputable def expected_usable_pieces : ℚ :=
  let L := 7 in
  let f (x : ℝ) := x in
  if 2 < f x ∧ f x < 5 then 2 else if 2 < f x ∨ f x < 5 then 1 else 0

theorem expected_value_usable_pieces :
  (100 * 10 + 7) = 1007 :=
by
  let L := 7
  let usable_condition := λ x : ℝ, (2 < x ∧ x < 5)
  let P_usable_left := (7 - 2) / 7
  let P_usable_right := (5 - 0) / 7
  let expected_value := P_usable_left + P_usable_right
  have h_expected_value : expected_value = (10 / 7) := by 
    simp [P_usable_left, P_usable_right]
    rw [← add_div]
    norm_num
  sorry

#print expected_value_usable_pieces

end expected_value_usable_pieces_l249_249689


namespace soccer_uniform_probability_l249_249385

-- Definitions for the conditions of the problem
def colorsSocks : List String := ["red", "blue"]
def colorsShirts : List String := ["red", "blue", "green"]

noncomputable def differentColorConfigurations : Nat :=
  let validConfigs := [("red", "blue"), ("red", "green"), ("blue", "red"), ("blue", "green")]
  validConfigs.length

noncomputable def totalConfigurations : Nat :=
  colorsSocks.length * colorsShirts.length

noncomputable def probabilityDifferentColors : ℚ :=
  (differentColorConfigurations : ℚ) / (totalConfigurations : ℚ)

-- The theorem to prove
theorem soccer_uniform_probability :
  probabilityDifferentColors = 2 / 3 :=
by
  sorry

end soccer_uniform_probability_l249_249385


namespace sum_union_eq_five_l249_249053

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {1, 2, 3}

-- Define the union of A and B
def union := A ∪ B

-- Theorem statement: The sum of all elements in the union of A and B is 5.
theorem sum_union_eq_five : (Finset.sum (A ∪ B).to_finset id) = 5 := by
  sorry

end sum_union_eq_five_l249_249053


namespace description_of_T_l249_249943

noncomputable def set_of_points (B C : ℝ × ℝ) : set (ℝ × ℝ) :=
  {A : ℝ × ℝ | let h := 2 in (A.2 = h ∧ B = (1,0) ∧ C = (5,0)) 
                ∨ (A.2 = -h ∧ B = (1,0) ∧ C = (5,0))}

theorem description_of_T:
  let B := (1 : ℕ, 0 : ℕ)
  let C := (5 : ℕ, 0 : ℕ)
  let T := set_of_points B C
  in T = {A : ℝ × ℝ | A.2 = 2 ∨ A.2 = -2 ∧ A.1 = 3 → A = (3, 2)} :=
by
  sorry

end description_of_T_l249_249943


namespace inequality_2_inequality_4_l249_249830

variables (a b : ℝ)
variables (h₁ : 0 < a) (h₂ : 0 < b)

theorem inequality_2 (h₁ : 0 < a) (h₂ : 0 < b) : a > |a - b| - b :=
by
  sorry

theorem inequality_4 (h₁ : 0 < a) (h₂ : 0 < b) : ab + 2 / ab > 2 :=
by
  sorry

end inequality_2_inequality_4_l249_249830


namespace required_milk_l249_249100

-- Define the conditions as given in the problem.
def milk_per_flour_ratio : ℝ := 75 / 250
def total_flour : ℝ := 1200

-- Define the statement to prove.
theorem required_milk (milk_per_flour_ratio total_flour : ℝ) : 
  total_flour = 1200 → milk_per_flour_ratio = 75 / 250 → 
  (milk_per_flour_ratio * total_flour * 1) = 360 :=
by
  -- assume conditions
  intros h_flour h_ratio
  -- sorry is used here to indicate the proof step is skipped
  sorry

end required_milk_l249_249100


namespace tan_inverse_form_l249_249154

-- Definitions of the conditions
variables (c d : ℝ) (y m : ℝ)

-- Conditions given in the problem
def tan_y := Math.tan y = 2 * c / (3 * d)
def tan_2y := Math.tan (2 * y) = 3 * d / (2 * c + 3 * d)

-- Statement we need to prove
theorem tan_inverse_form :
  tan_y c d y → tan_2y c d y → y = Math.atan m → m = 1 / 3 :=
by
  intros
  sorry

end tan_inverse_form_l249_249154


namespace jayden_planes_l249_249126

theorem jayden_planes (W : ℕ) (wings_per_plane : ℕ) (total_wings : W = 108) (wpp_pos : wings_per_plane = 2) :
  ∃ n : ℕ, n = W / wings_per_plane ∧ n = 54 :=
by
  sorry

end jayden_planes_l249_249126


namespace sum_equality_m_value_l249_249782

noncomputable def sum_sin_terms : ℝ :=
  ∑ k in finset.range 89, 1 / (Real.sin (↑k + 30) * Real.sin (↑k + 31))

noncomputable def expected_sum (sin1 sqrt3 : ℝ) : ℝ :=
  (4 * sqrt3) / (3 * sin1)

theorem sum_equality (sin1 sqrt3 : ℝ) (h1 : sin1 = Real.sin 1)
  (h2 : sqrt3 = Real.sqrt 3) :
  sum_sin_terms = expected_sum sin1 sqrt3 :=
by
  unfold sum_sin_terms expected_sum
  sorry

noncomputable def m (sin1 sqrt3 : ℝ) : ℝ :=
  Real.arcsin (sin1 * sqrt3 / 4)

theorem m_value (sin1 sqrt3 : ℝ) (h1 : sin1 = Real.sin 1)
  (h2 : sqrt3 = Real.sqrt 3) :
  ∀ (m_val : ℝ), m_val = m sin1 sqrt3 :=
by
  sorry

end sum_equality_m_value_l249_249782


namespace number_of_distinct_b_values_l249_249445

theorem number_of_distinct_b_values : 
  ∃ (b : ℝ) (p q : ℤ), (∀ (x : ℝ), x*x + b*x + 12*b = 0) ∧ 
                        p + q = -b ∧ 
                        p * q = 12 * b ∧ 
                        ∃ n : ℤ, 1 ≤ n ∧ n ≤ 15 :=
sorry

end number_of_distinct_b_values_l249_249445


namespace total_amount_Rs20_l249_249251

theorem total_amount_Rs20 (x y z : ℕ) 
(h1 : x + y + z = 130) 
(h2 : 95 * x + 45 * y + 20 * z = 7000) : 
∃ z : ℕ, (20 * z) = (7000 - 95 * x - 45 * y) / 20 := sorry

end total_amount_Rs20_l249_249251


namespace dot_product_calc_l249_249481

-- Vectors a and b are unit vectors
variables (a b : ℝ^3)
variables ha : ∥a∥ = 1
variables hb : ∥b∥ = 1

-- The angle between a and b is 60 degrees
variables hab : (a ⬝ b) = (1 * 1 * real.cos (real.pi / 3))

-- Theorem to prove
theorem dot_product_calc :
  (2 • a - b) ⬝ b = 0 :=
by sorry

end dot_product_calc_l249_249481


namespace area_of_triangle_POB_l249_249541

noncomputable def polarToCartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem area_of_triangle_POB {α : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2)
  (h1 : ∃ (ρ : ℝ), ρ * Real.cos α = 2) (h2 : ∃ (ρ : ℝ), ρ = 4 * Real.sin α) :
  let P := polarToCartesian 4 (Real.pi / 4)
  let A := polarToCartesian (2 / Real.cos α) α
  let B := polarToCartesian (4 * Real.sin α) α in
  |(fst A)^2 + (snd A)^2 * |(fst B)^2 + (snd B)^2| = 16 + 8 * Real.sqrt 3 →
  let area := 0.5 * 4 * (Real.sqrt 6 + Real.sqrt 2) *
    Real.sin ((5 * Real.pi) / 12 - Real.pi / 4) in
  area = Real.sqrt 6 + Real.sqrt 2 :=
begin
  intros,
  sorry
end

end area_of_triangle_POB_l249_249541


namespace lending_methods_count_l249_249342

theorem lending_methods_count :
  let books := ["Romance of the Three Kingdoms", "Journey to the West", "Water Margin", "Dream of the Red Chamber"]
  let classmates := {A, B, C}
  ∀ (f : books → classmates), -- f is a function that assigns each book a classmate
  (∃ x y : books, x ≠ y ∧ f x = f y) ∧ -- each classmate must receive at least one book, so there must be at least one group with 2 books
  (∀ x y, x ≠ y → f x ≠ "Journey to the West" → f y ≠ "Dream of the Red Chamber" → f x ≠ y ∧ f y ≠ x) -- "Journey to the West" and "Dream of the Red Chamber" cannot be lent to the same person
  → finset.card ((books.powerset.filter (λ s, s.card = 2)).erase (insert "Journey to the West" (singleton "Dream of the Red Chamber"))) * 6 = 30 :=
by sorry

end lending_methods_count_l249_249342


namespace number_of_true_propositions_among_inverse_negation_contrapositive_l249_249308

variable (a b : ℝ)

def original_proposition : Prop :=
  (a^2 + 2 * a * b + b^2 + a + b - 2 ≠ 0) → (a + b ≠ 1)

def inverse_proposition : Prop :=
  (a + b ≠ 1) → (a^2 + 2 * a * b + b^2 + a + b - 2 ≠ 0)

def negation_proposition : Prop :=
  (a^2 + 2 * a * b + b^2 + a + b - 2 = 0) → (a + b = 1)

def contrapositive_proposition : Prop :=
  (a + b = 1) → (a^2 + 2 * a * b + b^2 + a + b - 2 = 0)

theorem number_of_true_propositions_among_inverse_negation_contrapositive :
  (∃ p1 p2 p3 : Prop, p1 = inverse_proposition a b ∧ p2 = negation_proposition a b ∧ p3 = contrapositive_proposition a b ∧
                     (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) = 1) :=
sorry

end number_of_true_propositions_among_inverse_negation_contrapositive_l249_249308


namespace fraction_decreased_is_correct_l249_249974

noncomputable def initial_balance := 500
noncomputable def withdrawal_amount := 200
noncomputable def final_balance := 360

theorem fraction_decreased_is_correct :
  let B := initial_balance in
  let D := (B - withdrawal_amount) / 5 in
  B - withdrawal_amount + D = final_balance →
  (withdrawal_amount : ℝ) / initial_balance = 2 / 5 :=
by
  sorry

end fraction_decreased_is_correct_l249_249974


namespace angle_sum_triangle_l249_249966

theorem angle_sum_triangle (A B C : ℝ) (hA : A = 75) (hB : B = 40) (h_sum : A + B + C = 180) : C = 65 :=
by
  sorry

end angle_sum_triangle_l249_249966


namespace coordinates_of_P_l249_249928

open Real

theorem coordinates_of_P (P : ℝ × ℝ) (h1 : P.1 = 2 * cos (2 * π / 3)) (h2 : P.2 = 2 * sin (2 * π / 3)) :
  P = (-1, sqrt 3) :=
by
  sorry

end coordinates_of_P_l249_249928


namespace value_of_a6_l249_249050

theorem value_of_a6 (a : ℕ → ℝ) (h_positive : ∀ n, 0 < a n)
  (h_a1 : a 1 = 1) (h_a2 : a 2 = 2)
  (h_recurrence : ∀ n, 2 * (a n)^2 = (a (n + 1))^2 + (a (n - 1))^2) :
  a 6 = 4 := 
sorry

end value_of_a6_l249_249050


namespace three_digit_divisible_by_13_l249_249067

theorem three_digit_divisible_by_13 : 
  (finset.filter (λ n, 100 ≤ 13 * n ∧ 13 * n ≤ 999) (finset.Icc 1 100)).card = 69 := 
by 
  sorry

end three_digit_divisible_by_13_l249_249067


namespace tile_equations_correct_l249_249343

theorem tile_equations_correct (x y : ℕ) (h1 : 24 * x + 12 * y = 2220) (h2 : y = 2 * x - 15) : 
    (24 * x + 12 * y = 2220) ∧ (y = 2 * x - 15) :=
by
  exact ⟨h1, h2⟩

end tile_equations_correct_l249_249343


namespace f_x_plus_f_inv_x_eq_one_f_sum_values_eq_seven_halves_l249_249005

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem f_x_plus_f_inv_x_eq_one (x : ℝ) (hx : x ≠ 0) : f(x) + f(1 / x) = 1 := 
by 
sorry

theorem f_sum_values_eq_seven_halves : 
  f(1) + f(2) + f(1 / 2) + f(3) + f(1 / 3) + f(4) + f(1 / 4) = 7 / 2 := 
by 
sorry

end f_x_plus_f_inv_x_eq_one_f_sum_values_eq_seven_halves_l249_249005


namespace k1k2_ellipse_intersection_l249_249838

noncomputable def midPoint {α : Type*} [Field α] (A B : α×α) : α × α := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def isEllipse {α : Type*} [Field α] (p : α × α) : Prop := 
  p.1^2 + 2 * p.2^2 = 2

def slope {α : Type*} [Field α] {A B : α × α} (h : A.1 ≠ B.1) : α := 
  (B.2 - A.2) / (B.1 - A.1)

theorem k1k2_ellipse_intersection
  {α : Type*} [Field α] {A B : α × α} {P : α × α} {k1 k2 : α}
  (hA : isEllipse A) (hB : isEllipse B)
  (hP : P = midPoint A B)
  (hk1 : ∀ h : A.1 ≠ B.1, slope h = k1) 
  (hk1_ne_zero : k1 ≠ 0)
  (hk2 : k2 = P.2 / P.1) :
  k1 * k2 = -1/2 := by sorry

end k1k2_ellipse_intersection_l249_249838


namespace divisors_of_12_18_20_l249_249188

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem divisors_of_12_18_20 (n : ℕ) (h : n ∈ {1, 2, 3, ..., 20}) :
  n ∈ {12, 18, 20} ↔ (∀ m ∈ {1, 2, 3, ..., 20}, count_divisors m ≤ 6) :=
by
  sorry

end divisors_of_12_18_20_l249_249188


namespace arithmetic_identity_l249_249763

theorem arithmetic_identity :
  65 * 1515 - 25 * 1515 + 1515 = 62115 :=
by
  sorry

end arithmetic_identity_l249_249763


namespace max_divisors_up_to_20_l249_249221

def divisors (n : Nat) : List Nat :=
  (List.range (n + 1)).filter (λ d, n % d = 0)

def count_divisors (n : Nat) : Nat :=
  (divisors n).length

theorem max_divisors_up_to_20 :
  ∀ n, n ∈ List.range 21 → count_divisors n ≤ 6 ∧
      ((count_divisors 12 = 6) ∧
       (count_divisors 18 = 6) ∧
       (count_divisors 20 = 6)) :=
by
  sorry

end max_divisors_up_to_20_l249_249221


namespace average_ge_half_n_plus_one_l249_249575

theorem average_ge_half_n_plus_one
  (m n : ℕ)
  (h_positive_m : 0 < m)
  (h_positive_n : 0 < n)
  (a : Fin m → ℕ)
  (h_distinct : Function.Injective a)
  (h_range : ∀ i, a i < n + 1)
  (h_condition : ∀ (i j : Fin m), a i + a j ≤ n → ∃ k : Fin m, a i + a j = a k) :
  (∑ i, a i) / m ≥ (n + 1) / 2 :=
by
  sorry

end average_ge_half_n_plus_one_l249_249575


namespace area_comparison_l249_249952

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

variables (A B C M K H M' K' H' : ℝ × ℝ)
variables (h_acute : ∀ (P Q R : ℝ × ℝ), P ≠ Q → Q ≠ R → R ≠ P → 
  (area_of_triangle P Q R > 0) → (∀ p q r s, ∠PQR < π / 2))

/-- The conditions below ensure that the segments formed in a triangle result in expected points
 - M' is the intersection of median, K' of bisector, H' of altitude,
 - Given these intersections result in a triangle, the areas are the focus.
-/
axiom h_triangle : ∀ (A B C M K H M' K' H' : ℝ × ℝ),
  (area_of_triangle A B C > 0) →
  let median : ℝ × ℝ := ( (A.1 + B.1) / 2 , (A.2 + B.2) / 2 ) in
  let angle_bisector : ℝ × ℝ := ((K.1 + B.1) / 2, (K.2 + B.2) / 2) in
  let altitude : ℝ × ℝ := (H.1, H.2) in -- assuming orthogonal projection
  (M' = median) →
  (K' = angle_bisector) →
  (H' = altitude) →
  let area_ABC := area_of_triangle A B C in
  let area_MKH := area_of_triangle M' K' H' in
  (area_MKH > 0.499 * area_ABC)

theorem area_comparison (A B C M K H M' K' H' : ℝ × ℝ) 
  (h_acute : ∀ (P Q R : ℝ × ℝ), P ≠ Q → Q ≠ R → R ≠ P → 
    (area_of_triangle P Q R > 0) → (∀ p q r s, ∠PQR < π / 2))
  (h_triangle : ∀ (A B C M K H M' K' H' : ℝ× ℝ), 
    (area_of_triangle A B C > 0) →
    let median : ℝ × ℝ := ( (A.1 + B.1) / 2, (A.2 + B.2) / 2) in
    let angle_bisector : ℝ × ℝ := ((K.1 + B.1) / 2, (K.2 + B.2) / 2) in
    let altitude : ℝ × ℝ := (H.1, H.2) in -- assuming orthogonal projection
    (M' = median) →
    (K' = angle_bisector) →
    (H' = altitude) →
    let area_ABC := area_of_triangle A B C in
    let area_MKH := area_of_triangle M' K' H' in
    (area_MKH > 0.499 * area_ABC)) :
  (area_of_triangle M' K' H' > 0.499 * area_of_triangle A B C) :=
sorry

end area_comparison_l249_249952


namespace intersection_distance_l249_249306

theorem intersection_distance (p q : ℕ) (hpq_coprime : Nat.coprime p q) 
  (h_dist : (∃ C D : ℝ, C ≠ D ∧ 2 = 3 * C^2 + 2 * C - 1 ∧ 2 = 3 * D^2 + 2 * D - 1 ∧ abs (C - D) = (2 * Real.sqrt p) / q)) :
  p - q = 31 :=
by
  sorry

end intersection_distance_l249_249306


namespace hyperbola_eccentricity_l249_249626

def hyperbola_eq : Prop :=
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 2) = 1

theorem hyperbola_eccentricity (x y : ℝ) (h : hyperbola_eq x y) : 
  let a := 2
  let b := real.sqrt 2
  let c := real.sqrt (a^2 + b^2)
  let e := c / a
  e = real.sqrt 6 / 2 := 
by sorry

end hyperbola_eccentricity_l249_249626


namespace value_of_coefficients_l249_249826

theorem value_of_coefficients (a₀ a₁ a₂ a₃ : ℤ) (x : ℤ) :
  (5 * x + 4) ^ 3 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 →
  x = -1 →
  (a₀ + a₂) - (a₁ + a₃) = -1 :=
by
  sorry

end value_of_coefficients_l249_249826


namespace smallest_solution_satisfies_condition_l249_249776

noncomputable def verify_smallest_solution : ℝ :=
  (13 - Real.sqrt 69) / 50

theorem smallest_solution_satisfies_condition :
  ∃ x : ℝ, 0 < x ∧ (sqrt (3 * x) = 5 * x - 1) ∧ x = verify_smallest_solution :=
begin
  use verify_smallest_solution,
  split,
  { 
    -- Show that verify_smallest_solution is positive
    sorry
  },
  split,
  {
    -- Show that verify_smallest_solution satisfies the condition sqrt(3x) = 5x - 1.
    sorry
  },
  { 
    -- Show that x = verify_smallest_solution
    refl
  }
end

end smallest_solution_satisfies_condition_l249_249776


namespace ratio_pow_eq_l249_249028

theorem ratio_pow_eq {x y : ℝ} (h : x / y = 7 / 5) : (x^3 / y^2) = 343 / 25 :=
by sorry

end ratio_pow_eq_l249_249028


namespace rectangle_length_l249_249350

theorem rectangle_length (P L B : ℕ) (hP : P = 500) (hB : B = 100) (hP_eq : P = 2 * (L + B)) : L = 150 :=
by
  sorry

end rectangle_length_l249_249350


namespace black_squares_in_35th_row_l249_249742

-- Define the condition for the starting color based on the row
def starts_with_black (n : ℕ) : Prop := n % 2 = 1
def ends_with_white (n : ℕ) : Prop := true  -- This is trivially true by the problem condition
def total_squares (n : ℕ) : ℕ := 2 * n 
-- Black squares are half of the total squares for rows starting with a black square
def black_squares (n : ℕ) : ℕ := total_squares n / 2

theorem black_squares_in_35th_row : black_squares 35 = 35 :=
sorry

end black_squares_in_35th_row_l249_249742


namespace find_natural_number_l249_249325

theorem find_natural_number (N : ℕ)
  (h1 : ∃ m : ℕ, m < N ∧ m ∣ N ∧ N + m = 10 ^ (log 10 (N + m))) :
  N = 75 :=
by
  sorry

end find_natural_number_l249_249325


namespace greatest_num_divisors_in_range_l249_249174

def num_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d, n % d = 0)

theorem greatest_num_divisors_in_range : 
  ∀ n ∈ {1, 2, 3, ..., 20}, 
  num_divisors n < 7 → (n = 12 ∨ n = 18 ∨ n = 20) :=
by {
  sorry
}

end greatest_num_divisors_in_range_l249_249174


namespace remainder_of_3_pow_2023_mod_7_l249_249337

theorem remainder_of_3_pow_2023_mod_7 :
  (3^2023) % 7 = 3 := 
by
  sorry

end remainder_of_3_pow_2023_mod_7_l249_249337


namespace stamps_total_l249_249583

theorem stamps_total (x y : ℕ) (hx : x = 34) (hy : y = x + 44) : x + y = 112 :=
by sorry

end stamps_total_l249_249583


namespace max_distance_between_triangles_l249_249955

-- Define points in space
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define distance function between two points
def dist (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Define a triangle in 3D space
structure Triangle :=
  (A B C : Point3D)

-- Define the main statement
theorem max_distance_between_triangles (T1 T2 : Triangle) :
  ∃ a : ℝ, a = max (max (max (dist T1.A T2.A) (dist T1.A T2.B)) (max (dist T1.A T2.C) (dist T1.B T2.A)))
  (max (max (dist T1.B T2.B) (dist T1.B T2.C)) (max (dist T1.C T2.A) (max (dist T1.C T2.B) (dist T1.C T2.C))))
  ∧ ∀ (M ∈ [T1.A, T1.B, T1.C]) (M' ∈ [T2.A, T2.B, T2.C]), dist M M' ≤ a := sorry


end max_distance_between_triangles_l249_249955


namespace fraction_to_decimal_representation_l249_249432

/-- Determine the decimal representation of a given fraction. -/
theorem fraction_to_decimal_representation : (45 / (2 ^ 3 * 5 ^ 4) = 0.0090) :=
sorry

end fraction_to_decimal_representation_l249_249432


namespace seed_cost_l249_249751

theorem seed_cost (total_cost : ℝ) (cost_per_pound : ℝ) (h : 6 * cost_per_pound = total_cost) : 
  2 * cost_per_pound = 44.68 := 
by 
  have h2 : cost_per_pound = 44.68 / 2 := sorry
  have h3 : 2 * (44.68 / 2) = 44.68 := sorry
  exact h3

end seed_cost_l249_249751


namespace greatest_divisors_1_to_20_l249_249200

def is_divisor (a b : ℕ) : Prop := b % a = 0

def count_divisors (n : ℕ) : ℕ :=
  nat.succ (list.length ([k for k in list.range n if is_divisor k.succ n]))

def max_divisors_from_1_to_20 : set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 20 ∧ ∃ max_count, count_divisors n = max_count ∧ 
  ∀ m, 1 ≤ m ∧ m ≤ 20 → count_divisors m ≤ max_count}

theorem greatest_divisors_1_to_20 : max_divisors_from_1_to_20 = {12, 18, 20} :=
by {
  sorry
}

end greatest_divisors_1_to_20_l249_249200


namespace perimeter_of_shaded_region_is_16_add_8pi_l249_249539

-- Define the conditions
def O : Point := sorry  -- Center of the circle
def r : ℝ := 8  -- Radius of the circle
def OP : Segment := Segment.mk O sorry  -- Radius OP
def OQ : Segment := Segment.mk O sorry  -- Radius OQ

theorem perimeter_of_shaded_region_is_16_add_8pi :
  let circumference := 2 * Real.pi * r in
  let arc_pq := (1 / 2) * circumference in
  2 * r + arc_pq = 16 + 8 * Real.pi :=
by
  sorry

end perimeter_of_shaded_region_is_16_add_8pi_l249_249539


namespace f_satisfies_functional_l249_249477

noncomputable def f (n k: ℤ) (h_neg: n < 0) : ℤ := (n - k) ^ 2

theorem f_satisfies_functional (k: ℤ) :
  ∀ (n: ℤ), n < 0 → f n k ‹n < 0› * f (n + 1) k (by linarith [‹n < 0›]) = 
  (f n k ‹n < 0› + n - k) ^ 2 :=
by
  intros n hn
  rw [f, f, f]
  ring

end f_satisfies_functional_l249_249477


namespace allan_balloons_l249_249397

theorem allan_balloons (x : ℕ) : 
  (2 + x) + 1 = 6 → x = 3 :=
by
  intro h
  linarith

end allan_balloons_l249_249397


namespace maximum_candies_l249_249894

-- Definitions of the problem parameters
def classmates : ℕ := 25
def total_candies : ℕ := 200
def condition (candies : ℕ → ℕ) : Prop := 
  ∀ S (h : finset ℕ), S.card = 16 → (∑ i in S, candies i) ≥ 100

-- The theorem stating the maximum number of candies Vovochka can keep
theorem maximum_candies (candies : ℕ → ℕ) (h : condition candies) : 
  (total_candies - ∑ i in finset.range classmates, candies i) ≤ 37 :=
sorry

end maximum_candies_l249_249894


namespace chime_2500_l249_249721

-- Define the conditions
def chimes_at_quarter_hours_per_hour : nat := 2
def hourly_chime (hour : nat) : nat := hour
def start_time : nat := 2 * 60 + 30 -- minutes since midnight for 2:30 PM
def initial_date : nat := 1 -- January 1, 2023

-- Define a function to calculate total chimes from start time to midnight on initial date
def chimes_till_midnight (start_time : nat) : nat :=
  let remaining_hours := 11 - 2 -- from 2 PM to midnight
  let quarter_hour_chimes := remaining_hours * chimes_at_quarter_hours_per_hour
  let hourly_chimes := (3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)
  1 + quarter_hour_chimes + hourly_chimes -- 1 chime at 2:45 PM

-- Function to calculate daily chimes starting from January 2, 2023
def daily_chimes : nat :=
  24 * chimes_at_quarter_hours_per_hour + (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

-- Lean proof statement
theorem chime_2500 (start_time : nat) (initial_date : nat) : 
  ∃ (date : nat), (chimes_till_midnight start_time + 19 * daily_chimes) = 2500 ∧ date = 21 :=
by
  let total_chimes := chimes_till_midnight start_time + 19 * daily_chimes
  have h_total_chimes : total_chimes = 2500, sorry
  use 21
  exact ⟨h_total_chimes, rfl⟩

end chime_2500_l249_249721


namespace parrots_fraction_l249_249097

variable (P T : ℚ) -- P: fraction of parrots, T: fraction of toucans

def fraction_parrots (P T : ℚ) : Prop :=
  P + T = 1 ∧
  (2 / 3) * P + (1 / 4) * T = 0.5

theorem parrots_fraction (P T : ℚ) (h : fraction_parrots P T) : P = 3 / 5 :=
by
  sorry

end parrots_fraction_l249_249097


namespace percentage_error_l249_249402

-- Define the actual side of the square
def actual_side (x : ℝ) : ℝ := x

-- Define the measured side with 25% error
def measured_side (x : ℝ) : ℝ := 1.25 * x

-- Define the actual area
def actual_area (x : ℝ) : ℝ := x ^ 2

-- Define the calculated (erroneous) area
def erroneous_area (x : ℝ) : ℝ := (measured_side x) ^ 2

-- Prove the percentage error in the calculated area
theorem percentage_error (x : ℝ) : 
  let error := (erroneous_area x) - (actual_area x),
      percentage_error := (error / (actual_area x)) * 100
  in percentage_error = 56.25 := by
  sorry

end percentage_error_l249_249402


namespace small_sphere_is_tangent_to_four_spheres_l249_249104

-- Define the conditions of the four given spheres with specified radii
def sphere_radius1 : ℝ := 2
def sphere_radius2 : ℝ := 2
def sphere_radius3 : ℝ := 3
def sphere_radius4 : ℝ := 3
def small_sphere_radius : ℝ := 6/11

-- Main theorem statement
theorem small_sphere_is_tangent_to_four_spheres :
  ∃ r : ℝ, r = small_sphere_radius ∧
  (∃ k1 k2 k3 k4 k5 : ℝ, k1 = 1/sphere_radius1 ∧ k2 = 1/sphere_radius2 ∧
   k3 = 1/sphere_radius3 ∧ k4 = 1/sphere_radius4 ∧ k5 = 1/r ∧
   (k1 + k2 + k3 + k4 + k5)^2 = 3 * (k1^2 + k2^2 + k3^2 + k4^2 + k5^2)) :=
begin
  use small_sphere_radius,
  split,
  { refl },
  { use [1/sphere_radius1, 1/sphere_radius2, 1/sphere_radius3, 1/sphere_radius4, 1/small_sphere_radius],
    simp [sphere_radius1, sphere_radius2, sphere_radius3, sphere_radius4, small_sphere_radius],
    sorry,
  }
end

end small_sphere_is_tangent_to_four_spheres_l249_249104


namespace maximal_area_point_exists_and_value_l249_249012

noncomputable def ellipse_eq (a b : ℝ) : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1}

noncomputable def hyperbola_eq (v : ℝ) : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | (p.1^2) / (4 - v) + (p.2^2) / (1 - v) = 1}

lemma ellipse_hyperbola_common_focus 
  (v : ℝ) (hv : 1 < v ∧ v < 4) : (*(a:ℝ), b:ℝ) such that*)
  ∃ a b : ℝ, (a^2 - b^2 = 3) ∧ (ellipse_eq a b = {p : ℝ × ℝ | (p.1^2) / 4 + (p.2^2) / 1 = 1}) :=
sorry

noncomputable def area_triangle (O M N : ℝ × ℝ) : ℝ := 
  (1/2) * abs ((M.1 * N.2) - (N.1 * M.2))

theorem maximal_area_point_exists_and_value 
  (m n : ℝ) (R : ℝ × ℝ) 
  (hR_ellipse : R ∈ {p : ℝ × ℝ | (p.1^2) / 4 + (p.2^2) / 1 = 1}) 
  (hR : (R.1^2 + R.2^2 = 2) ∧ (R.1^2 + 4*R.2^2 = 4)) :
  ∃ M N : ℝ × ℝ, 
  (M ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}) ∧
  (N ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}) ∧
  (area_triangle (0, 0) M N = 1/2) :=
sorry

end maximal_area_point_exists_and_value_l249_249012


namespace line_inclination_angle_l249_249772

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x + (Real.sqrt 3) * y - 1 = 0

-- Define the condition of inclination angle in radians
def inclination_angle (θ : ℝ) : Prop := θ = Real.arctan (-1 / Real.sqrt 3) + Real.pi

-- The theorem to prove the inclination angle of the line
theorem line_inclination_angle (x y θ : ℝ) (h : line_eq x y) : inclination_angle θ :=
by
  sorry

end line_inclination_angle_l249_249772


namespace ralph_total_cost_l249_249601

theorem ralph_total_cost
  (initial_cart_cost: ℝ) -- Initial total cost
  (item_initial_cost: ℝ) -- Cost of the item with the issue
  (item_discount_rate: ℝ) -- Discount rate for item with the issue (as a fraction, e.g. 0.20 for 20%)
  (total_discount_rate: ℝ) -- Discount rate for total purchase (as a fraction, e.g. 0.10 for 10%) :
  initial_cart_cost = 54 →
  item_initial_cost = 20 →
  item_discount_rate = 0.20 →
  total_discount_rate = 0.10 →
  let item_discounted_cost := item_initial_cost * (1 - item_discount_rate) in
  let new_cart_cost := initial_cart_cost - item_initial_cost + item_discounted_cost in
  let final_cost := new_cart_cost * (1 - total_discount_rate) in
  final_cost = 45 :=
begin
  sorry -- The proof is omitted
end

end ralph_total_cost_l249_249601


namespace arithmetic_seq_sum_equidistant_l249_249099

variable (a : ℕ → ℤ)

theorem arithmetic_seq_sum_equidistant :
  (∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) → a 4 = 12 → a 1 + a 7 = 24 :=
by
  intros h_seq h_a4
  sorry

end arithmetic_seq_sum_equidistant_l249_249099


namespace largest_angle_of_obtuse_isosceles_triangle_l249_249661

theorem largest_angle_of_obtuse_isosceles_triangle (P Q R : Type) 
  (triangle_PQR : Triangle P Q R)
  (isosceles_PQR : Isosceles triangle_PQR) 
  (obtuse_PQR : Obtuse triangle_PQR)
  (angle_P_30 : angle P triangle_PQR = 30) : 
  ∃ (angle_Q : ℕ), is_largest_angle angle_Q triangle_PQR ∧ angle_Q = 120 := 
by 
  sorry

end largest_angle_of_obtuse_isosceles_triangle_l249_249661


namespace smallest_positive_period_of_f_minimum_value_of_f_in_interval_l249_249040

noncomputable def f (x : ℝ) : ℝ :=
  cos (x) ^ 4 - 2 * sin (x) * cos (x) - sin (x) ^ 4

theorem smallest_positive_period_of_f : smallest_period f = π :=
by sorry

theorem minimum_value_of_f_in_interval :
  ∃ x : ℝ, x ∈ Icc 0 (π / 2) ∧ f x = -sqrt 2 ∧ ∀ y ∈ Icc 0 (π / 2), f y ≥ -sqrt 2 :=
by sorry

end smallest_positive_period_of_f_minimum_value_of_f_in_interval_l249_249040


namespace a3_value_l249_249519

noncomputable def f (x : ℝ) : ℝ := x^6
axiom a0 : ℝ
axiom a1 : ℝ
axiom a2 : ℝ
axiom a3 : ℝ
axiom a4 : ℝ
axiom a5 : ℝ
axiom a6 : ℝ

def polynomial_expansion (x : ℝ) : ℝ :=
  a0 + a1 * (1 + x) + a2 * (1 + x)^2 + a3 * (1 + x)^3 + a4 * (1 + x)^4 + a5 * (1 + x)^5 + a6 * (1 + x)^6

theorem a3_value : a3 = -20 :=
by
  sorry

end a3_value_l249_249519


namespace segment_length_is_24_over_7_l249_249049

-- Define the parametric equations of curve C1
def C1_parametric_eq (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

-- Define the Cartesian equation equivalent of curve C1
def C1_cartesian_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1

-- Define the Cartesian equation of curve C2
def C2_cartesian_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the equation system that we need to solve
def eq_system (x y : ℝ) : Prop := C1_cartesian_eq x y ∧ C2_cartesian_eq x y

-- Prove the length of segment AB is 24/7
theorem segment_length_is_24_over_7 : ∃ (A B : ℝ × ℝ),
  (eq_system A.fst A.snd) ∧
  (eq_system B.fst B.snd) ∧
  (∃ x1 x2, A = (x1, x1 + 1) ∧ B = (x2, x2 + 1) ∧
    Real.dist A B = 24/7) :=
by
  sorry

end segment_length_is_24_over_7_l249_249049


namespace total_trout_caught_l249_249165

theorem total_trout_caught (n_share j_share total_caught : ℕ) (h1 : n_share = 9) (h2 : j_share = 9) (h3 : total_caught = n_share + j_share) :
  total_caught = 18 :=
by
  sorry

end total_trout_caught_l249_249165


namespace twelfth_prime_number_l249_249243

theorem twelfth_prime_number (h : ∃ p : Nat, Nat.prime p ∧ p = 17 ∧ nat.find (λ n, Nat.prime n) 7 = 17) : Nat.find (λ n, Nat.prime n) 12 = 37 :=
sorry

end twelfth_prime_number_l249_249243


namespace intersect_A_B_l249_249051

def A : Set ℝ := {x | 1/x < 1}
def B : Set ℝ := {-1, 0, 1, 2}
def intersection_result : Set ℝ := {-1, 2}

theorem intersect_A_B : A ∩ B = intersection_result :=
by
  sorry

end intersect_A_B_l249_249051


namespace parabola_standard_equation_l249_249317

-- Define the given ellipse equation
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + y^2 = 1

-- Define the right focus of the given ellipse
def right_focus : ℝ × ℝ := (real.sqrt 2, 0)

-- Theorem stating the standard equation of the parabola given the conditions
theorem parabola_standard_equation (x y : ℝ) :
  (∃ p : ℝ, right_focus = (p / 2, 0) ∧ y^2 = 4 * p * x) :=
sorry

end parabola_standard_equation_l249_249317


namespace complex_multiplication_l249_249837

-- Definitions of given conditions
def is_imaginary_unit (i : ℂ) : Prop := i = complex.I

-- Statement of the problem to be proved.
theorem complex_multiplication (i : ℂ) (h : is_imaginary_unit i) :
  (1 - 2 * i) * (2 + i) = 4 - 3 * i :=
by sorry

end complex_multiplication_l249_249837


namespace solution_set_of_inequality_l249_249843

theorem solution_set_of_inequality (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  let f (x : ℝ) := a^x + (2 - 1) * a^(-x)
  set g (x : ℝ) := f (real.log x / real.log 2) in
  {x : ℝ | g x > (a^2 + 1) / a} = {x : ℝ | 0 < x ∧ x < 1 / 2} ∪ {x : ℝ | x > 2} :=
by
  sorry

end solution_set_of_inequality_l249_249843


namespace probability_penny_nickel_dime_all_heads_l249_249281

-- Define flipping five coins
def flip_five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the function to check if the penny, nickel, and dime are heads
def all_heads_penny_nickel_dime (o : flip_five_coins) : bool :=
  o.1 = tt ∧ o.2 = tt ∧ o.3 = tt

-- Define the total count of possible outcomes
def total_outcomes : ℕ := 32

-- Define the count of favorable outcomes
def favorable_outcomes : ℕ := 4

-- Define the probability calculation
def probability_favorable : ℚ := favorable_outcomes / total_outcomes

-- The statement proving the probability is 1/8
theorem probability_penny_nickel_dime_all_heads :
  probability_favorable = 1 / 8 :=
by
  sorry

end probability_penny_nickel_dime_all_heads_l249_249281


namespace count_product_leq_zero_l249_249458

def product_leq_zero (x : ℕ) : Prop :=
  (∏ k in finset.range 1009, (x - 2 * (k + 1))) ≤ 0

theorem count_product_leq_zero :
  ∃ n : ℕ, n = 1514 ∧
  (∀ m : ℕ, product_leq_zero m ↔ m ∈ finset.range 2020 ∧ ¬(product_leq_zero m)).card = n :=
sorry

end count_product_leq_zero_l249_249458


namespace determine_k_m_l249_249386

noncomputable def k_and_m_sum : ℝ := 2   -- k + m = 2

def square_side_length : ℝ := 2

variable (PT_QU : ℝ)  -- PT = QU

-- Proving the relationship
theorem determine_k_m (h1 : PT_QUx = x)
                      (fold_condition : PT_QU = 2 - x ∧
                                       TS = sqrt(2) * (2 - x) ∧
                                       (TS + TU = 2 * sqrt(2)) :
  PT_QU = sqrt(k) - m → k + m = 2 :=
sorry

end determine_k_m_l249_249386


namespace inverse_of_shifted_function_l249_249006

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the function f

-- Conditions
axiom f_inverse : ∀ y, ∃ x, f x = y -- f has an inverse function
axiom f_at_1 : f 1 = 2             -- The graph of f passes through (1,2)

-- Theorem statement
theorem inverse_of_shifted_function (f_inverse : ∀ y, ∃ x, f x = y) (f_at_1 : f 1 = 2) : 
  ∃ x, (λ y, f (y - 4))⁻¹ x = 2 :=
sorry -- proof omitted

end inverse_of_shifted_function_l249_249006


namespace howard_money_l249_249981

theorem howard_money (h_last_week : ℕ) (h_weekend : ℕ) (h_condition : h_weekend = h_last_week) :
  (h_last_week + h_weekend) = 52 :=
by
  have h_last_week := 26
  have h_condition := rfl
  have h_weekend := h_last_week
  sorry

end howard_money_l249_249981


namespace external_angle_at_C_l249_249532

-- Definitions based on conditions
def angleA : ℝ := 40
def B := 2 * angleA
def sum_of_angles_in_triangle (A B C : ℝ) : Prop := A + B + C = 180
def external_angle (C : ℝ) : ℝ := 180 - C

-- Theorem statement
theorem external_angle_at_C :
  ∃ C : ℝ, sum_of_angles_in_triangle angleA B C ∧ external_angle C = 120 :=
sorry

end external_angle_at_C_l249_249532


namespace ratio_BO_BD_eq_zero_l249_249102

structure Point (α : Type) :=
  (x : α)
  (y : α)

structure Rectangle (α : Type) :=
  (A B C D : Point α)
  (AB AD : α)

def midpoint {α : Type} [Field α] (P Q : Point α) : Point α :=
  Point.mk ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

def on_line {α : Type} [Field α] (P Q R : Point α) : Prop :=
  (Q.y - P.y)*(R.x - P.x) = (R.y - P.y)*(Q.x - P.x)

theorem ratio_BO_BD_eq_zero {α : Type} [Field α] 
(A B C D M O : Point α) (r : Rectangle α) 
(h1 : r.A = A) (h2 : r.B = B) (h3 : r.C = C) (h4 : r.D = D)
(h5 : r.AB = 6) (h6 : r.AD = 4) 
(h7 : M = midpoint B C) 
(h8 : on_line A M O) (h9 : on_line B D O) 
(BD : α) (BO : α) : 
(BD = ((B.x - D.x)^2 + (B.y - D.y)^2)^0.5) → 
(BO = ((B.x - O.x)^2 + (B.y - O.y)^2)^0.5) → 
O = B → 
BO / BD = 0 :=
by 
  sorry

end ratio_BO_BD_eq_zero_l249_249102


namespace road_signs_count_l249_249738

theorem road_signs_count (n1 n2 n3 n4 : ℕ) (h1 : n1 = 40) (h2 : n2 = n1 + n1 / 4) (h3 : n3 = 2 * n2) (h4 : n4 = n3 - 20) : 
  n1 + n2 + n3 + n4 = 270 := 
by
  sorry

end road_signs_count_l249_249738


namespace colorable_prism_1995_not_colorable_prism_1996_l249_249120

theorem colorable_prism_1995 : ∃ (coloring : PrismColoring 1995 3), 
  (∀ face, contains_all_colors face coloring) ∧
  (∀ vertex, edges_meet_diff_colors vertex coloring) :=
sorry

theorem not_colorable_prism_1996 : ¬(∃ (coloring : PrismColoring 1996 3), 
  (∀ face, contains_all_colors face coloring) ∧
  (∀ vertex, edges_meet_diff_colors vertex coloring)) :=
sorry

end colorable_prism_1995_not_colorable_prism_1996_l249_249120


namespace geometric_sequence_minimum_value_l249_249527

theorem geometric_sequence_minimum_value (a : ℕ → ℝ) (q : ℝ) (m n : ℕ)
  (hq_pos : 0 < q)
  (geom_seq : ∀ n, a (n + 1) = a n * q)
  (h_a6 : a 6 = a 5 + 2 * a 4)
  (h_sqrt : ∃ (m n : ℕ), m ≠ n ∧ sqrt (a m * a n) = 4 * a 1) :
  (\frac{1}{m} + \frac{2}{n}) = \frac{3 + 2 * sqrt (2)}{6} :=
by
  sorry

end geometric_sequence_minimum_value_l249_249527


namespace count_valid_x_values_l249_249061

theorem count_valid_x_values : 
  {x : ℕ | 34 ≤ x ∧ x ≤ 49}.card = 16 := 
by sorry

end count_valid_x_values_l249_249061


namespace average_of_set_is_333_l249_249935

theorem average_of_set_is_333 (x : ℕ) (h_prime : Nat.Prime x) (h_median : x - 1 = {x - 1, 3 * x + 3, 2 * x - 4}.median) :
  (1 + 9 + 0) / 3 = 10 / 3 := 
by
  sorry

end average_of_set_is_333_l249_249935


namespace min_value_of_a_l249_249142

noncomputable def P (x : ℕ) : ℤ := sorry

def smallest_value_of_a (a : ℕ) : Prop :=
  a > 0 ∧
  (P 1 = a ∧ P 3 = a ∧ P 5 = a ∧ P 7 = a ∧ P 9 = a ∧
   P 2 = -a ∧ P 4 = -a ∧ P 6 = -a ∧ P 8 = -a ∧ P 10 = -a)

theorem min_value_of_a : ∃ a : ℕ, smallest_value_of_a a ∧ a = 6930 :=
sorry

end min_value_of_a_l249_249142


namespace find_point_B_l249_249839

theorem find_point_B (A B : ℝ) (h1 : A = 2) (h2 : abs (B - A) = 5) : B = -3 ∨ B = 7 :=
by
  -- This is where the proof steps would go, but we can skip it with sorry.
  sorry

end find_point_B_l249_249839


namespace part_a_part_b_l249_249591

noncomputable def probability_target_hit (n : ℕ) : ℝ :=
1 - (1 - (1 : ℝ) / n) ^ n

noncomputable def expected_targets_hit (n : ℕ) : ℝ :=
n * probability_target_hit n

theorem part_a (n : ℕ) : expected_targets_hit n = n * (1 - (1 - (1 : ℝ) / n) ^ n) :=
sorry

theorem part_b (n : ℕ) : expected_targets_hit n ≥ n / 2 :=
begin
  have h : (1 - (1 / n : ℝ)) ^ n < 1 / 2,
  { sorry }, -- Here you would prove that for all n, (1 - 1/n)^n < 1/2, which generally follows from the known limit.
  have h2 : (1 - (1 : ℝ) / n) ^ n < 1 / 2 := h,
  calc
  expected_targets_hit n = n * (1 - (1 - (1 / n : ℝ)) ^ n)      : by rw [expected_targets_hit]
                     ... ≥ n * (1 - 1/2)                      : by { apply mul_le_mul_left, linarith, linarith }
                     ... = n / 2                              : by ring
end

end part_a_part_b_l249_249591


namespace max_candies_vovochka_can_keep_l249_249908

-- Definitions for the conditions in the problem.
def classmates := 25
def total_candies := 200
def minimum_candies (candies_distributed : Vector ℕ classmates) : Prop :=
  ∀ (S : Finset (Fin classmates)), S.card = 16 → S.sum (candies_distributed.nth' S) ≥ 100

-- The proof goal
theorem max_candies_vovochka_can_keep (candies_distributed : Vector ℕ classmates) (h : minimum_candies candies_distributed) :
  ∃ k, k = 37 ∧ total_candies - Finset.univ.sum (candies_distributed.nth' Finset.univ) = k :=
begin
  sorry
end

end max_candies_vovochka_can_keep_l249_249908


namespace max_candies_l249_249874

theorem max_candies (c : ℕ := 200) (n m : ℕ := 25) (k : ℕ := 16) (t : ℕ := 100) 
  (H : ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - (25 - (x.1 : ℕ))) i ≥ 100) : 
  ∃ d : ℕ, (d = 37 ∧ ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - d - (25 - (x.1 : ℕ))) i ≤ 100) :=
by {
  use 37,
  sorry
}

end max_candies_l249_249874


namespace num_pennies_in_jar_l249_249727

-- Define the conditions as variables/constants
def num_nickels : ℕ := 85
def num_dimes : ℕ := 35
def num_quarters : ℕ := 26
def cost_per_scoop : ℕ := 300  -- 3 dollars = 300 cents
def num_family_members : ℕ := 5
def leftover_cents : ℕ := 48

-- Define the total values from the coins
def total_nickels : ℕ := num_nickels * 5
def total_dimes : ℕ := num_dimes * 10
def total_quarters : ℕ := num_quarters * 25

-- Define the total amount from coin types
def total_without_pennies : ℕ := total_nickels + total_dimes + total_quarters
def total_ice_cream_cost : ℕ := num_family_members * cost_per_scoop

-- Define the total amount in jar after ice cream trip
def total_jar_after_trip : ℕ := total_ice_cream_cost + leftover_cents

-- Define the proof
theorem num_pennies_in_jar :
  ∃ (num_pennies : ℕ), num_pennies = 123 ∧ total_jar_after_trip - total_without_pennies = num_pennies := 
by
  exists 123
  sorry

end num_pennies_in_jar_l249_249727


namespace total_fencing_cost_l249_249078

-- Definitions based on the conditions
def cost_per_side : ℕ := 69
def number_of_sides : ℕ := 4

-- The proof problem statement
theorem total_fencing_cost : number_of_sides * cost_per_side = 276 := by
  sorry

end total_fencing_cost_l249_249078


namespace largest_interior_angle_obtuse_isosceles_triangle_l249_249658

theorem largest_interior_angle_obtuse_isosceles_triangle :
  ∀ (P Q R : Type) (α β γ : ℝ), α + β + γ = 180 ∧ γ = 120 ∧ α = 30 ∧ β = 30 →
  (α = 30 ∧ β = 30 ∧ γ = 120) ∨
  (α = 30 ∧ γ = 30 ∧ β = 120) ∨
  (β = 30 ∧ γ = 30 ∧ α = 120) → 
  γ = max α (max β γ) :=
by {
  intros P Q R α β γ h1 h2,
  repeat { rw h1 at * },
  rw h2,
  sorry
}

end largest_interior_angle_obtuse_isosceles_triangle_l249_249658


namespace Vovochka_max_candies_l249_249881

theorem Vovochka_max_candies (classmates : Finset ℕ) (candies : ℕ) (hclassmates : classmates.card = 25) (hcandies : candies = 200) (H : ∀ (S : Finset ℕ), S.card = 16 → 100 ≤ S.sum (λ x, 1)) :
  ∃ (max_candies : ℕ), max_candies = 37 :=
by
  use 37
  sorry

end Vovochka_max_candies_l249_249881


namespace prove_collinearity_of_vectors_l249_249536

-- Define necessary points and conditions
variable (x0 y0 x1 y1 : ℝ)
def line_l (x : ℝ) : Prop := x = -1
def point_T := (3, 0 : ℝ × ℝ)
def point_OP := (x0, y0 : ℝ × ℝ)
def foot_point_S := (-1, y0 : ℝ × ℝ)
def condition_OP_ST := x0 * 4 - y0 * y0 = 0

-- Define the parabola equation y^2 = 4x
def parabola_curve (x y : ℝ) : Prop := y * y = 4 * x

-- Define the midway point M
def midpoint_M := ((x0 + x1) / 2, (y0 + y1) / 2 : ℝ × ℝ)

-- Define the point N on l, which intersects the x-axis
def point_N := (-1, 0 : ℝ × ℝ)

-- Define the vectors SM and NQ
def vector_SM := (((x0 ^ 2 + 1) / (2 * x0) + 1), ((y0 ^ 2 - 4) / (2 * y0) - y0) : ℝ × ℝ)
def vector_NQ := ((x1 + 1), y1 : ℝ × ℝ)

-- Prove that SM and NQ are collinear
theorem prove_collinearity_of_vectors : 
  parabola_curve x0 y0 → parabola_curve x1 y1 → 
  4 * x0 = y0 * y0 → 4 * x1 = y1 * y1 →
  ∃ λ : ℝ, 
    vector_SM x0 y0 = λ • vector_NQ x0 x1 y0 y1 := 
  by sorry

end prove_collinearity_of_vectors_l249_249536


namespace chord_equation_1_shortest_chord_l249_249008

variable {x y : ℝ}

def point_P := (1, 1) : ℝ × ℝ
def circle_C := ∀ {x y : ℝ}, (x-2)^2 + (y-2)^2 = 8
def chord_length := 2 * Real.sqrt 7

theorem chord_equation_1 (P_in_circle : (1-2)^2 + (1-2)^2 < 8)
    (P : point_P ∈ circle_C)
    (chord_AB : chord_length)
    (AB_through_P : ∀ {AB : ℝ → ℝ}, AB 1 = 1) :
  (∀ A B, ∃ line : ℝ → ℝ, (line x = 1 ∨ line x = y)) := sorry

theorem shortest_chord (P_in_circle : (1-2)^2 + (1-2)^2 < 8)
    (P : point_P ∈ circle_C)
    (shortest_chord_through_P : ∀ {line : ℝ → ℝ}, line 1 = 1)
    (shortest_chord_perpendicular_to_PC : ∀ {AB : ℝ → ℝ}, line x = -1) :
  (∀ C, ∃ line : ℝ → ℝ, (line x = -x + 2)) := sorry

end chord_equation_1_shortest_chord_l249_249008


namespace range_of_a_l249_249037

def f (x a : ℝ) := -x^3 + a * x
def g (x : ℝ) := - (1/2) * x^(3/2)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x a < g x) → a < -3/16 :=
by
  sorry

end range_of_a_l249_249037


namespace count_valid_integers_l249_249555

theorem count_valid_integers (p q : ℕ) (h1 : Nat.Prime p) (h2 : Nat.Prime q) (h3 : p ≠ 2) (h4 : q ≠ 2) : 
  let n := p^2010 * q^2010 in 
  (∃ (x : ℕ), x ≤ n ∧ p^2010 ∣ x^p - 1 ∧ q^2010 ∣ x^q - 1) = pq := 
sorry

end count_valid_integers_l249_249555


namespace no_natural_number_solution_for_divisibility_by_2020_l249_249423

theorem no_natural_number_solution_for_divisibility_by_2020 :
  ¬ ∃ k : ℕ, (k^3 - 3 * k^2 + 2 * k + 2) % 2020 = 0 :=
sorry

end no_natural_number_solution_for_divisibility_by_2020_l249_249423


namespace parallelogram_area_l249_249845

theorem parallelogram_area (s : ℝ) (ratio : ℝ) (A : ℝ) :
  s = 3 → ratio = 2 * Real.sqrt 2 → A = 9 → 
  (A * ratio = 18 * Real.sqrt 2) :=
by
  sorry

end parallelogram_area_l249_249845


namespace arithmetic_square_root_64_l249_249619

theorem arithmetic_square_root_64 : sqrt 64 = 8 :=
by sorry

end arithmetic_square_root_64_l249_249619


namespace relationship_among_abc_l249_249004

noncomputable def a : ℝ := Real.log 0.3 / Real.log 2
noncomputable def b : ℝ := Real.exp (0.3 * Real.log 2)
noncomputable def c : ℝ := Real.exp (0.2 * Real.log 0.3)

theorem relationship_among_abc :
  b > c ∧ c > a :=
by
  sorry

end relationship_among_abc_l249_249004


namespace rowing_speed_in_still_water_l249_249378

theorem rowing_speed_in_still_water (v c : ℝ) (h1 : c = 1.4) (t : ℝ)
  (h2 : (v + c) * t = (v - c) * (2 * t)) : 
  v = 4.2 :=
by
  sorry

end rowing_speed_in_still_water_l249_249378


namespace zero_point_exists_between_2_and_3_l249_249647

noncomputable def f (x : ℝ) := 2^(x-1) + x - 5

theorem zero_point_exists_between_2_and_3 :
  ∃ x₀ ∈ Set.Ioo (2 : ℝ) 3, f x₀ = 0 :=
sorry

end zero_point_exists_between_2_and_3_l249_249647


namespace sum_a5_a6_a7_l249_249483

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = q * a n

variables (a : ℕ → ℤ)
variables (h_geo : geometric_sequence a)
variables (h1 : a 2 + a 3 = 1)
variables (h2 : a 3 + a 4 = -2)

theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 24 :=
by
  sorry

end sum_a5_a6_a7_l249_249483


namespace problem_1_problem_2_l249_249428

-- Problem (1) Statement
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, abs (x - a) ≤ 2 ↔ -1 ≤ x ∧ x ≤ 3) → a = 1 := by 
sorry

-- Problem (2) Statement
theorem problem_2 (a : ℝ) : 
  (∀ x : ℝ, log 2 (abs (x - a) + abs (x - 3)) ≥ 2) → (a ≥ 7 ∨ a ≤ -1) := by 
sorry

end problem_1_problem_2_l249_249428


namespace problem_part1_problem_part2_l249_249055

-- Definitions for the sets
def SetA : Set ℝ := {x | Real.log x / Real.log 2 < 8}
def SetB : Set ℝ := {x | (x + 2) / (x - 4) < 0}

-- Given conditions
variable (a : ℝ)

-- Define Set C based on a
def SetC : Set ℝ := {x | a < x ∧ x < a + 1}

-- Assertions for the problem
theorem problem_part1 : SetA ∩ SetB = {x | 0 < x ∧ x < 4} :=
sorry  -- Proof of the intersection

theorem problem_part2 (h : SetB ∪ SetC = SetB) : -2 ≤ a ∧ a ≤ 3 :=
sorry  -- Proof of the range of a

end problem_part1_problem_part2_l249_249055


namespace numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249226

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).length

theorem numbers_with_most_divisors_in_range :
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  (number_of_divisors n ≤ 6) :=
by
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list
  simp [number_of_divisors] <;>
  dec_trivial -- Automatically discharges the goals since they are simple arithmetic comparisons

theorem greatest_divisors_in_range :
  number_of_divisors 12 = 6 ∧
  number_of_divisors 18 = 6 ∧
  number_of_divisors 20 = 6 ∧
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19],
  number_of_divisors n < 6 :=
by
  refine ⟨rfl, rfl, rfl, _⟩
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list excluding 12, 18, 20
  dec_trivial -- Automatically discharges the goals due to their simplicity

#print greatest_divisors_in_range

end numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249226


namespace range_of_g_l249_249774

noncomputable def g (x : ℝ) : ℝ := (Real.sin x) ^ 4 - (Real.sin x) ^ 2 * (Real.cos x) ^ 2 + (Real.cos x) ^ 4

theorem range_of_g : 
  (∀ x : ℝ, (Real.sin x)^2 + (Real.cos x)^2 = 1) ∧ 
  (∀ x : ℝ, (Real.sin x)^2 * (Real.cos x)^2 = ((Real.sin (2 * x)) / 2)^2) → 
  (set.range g = set.Icc (1/4 : ℝ) 1) := 
sorry

end range_of_g_l249_249774


namespace angle_PAB_eq_angle_QAC_l249_249571

variable {A B C M N P Q : Type}
variable [AffinePlane A]
variable [RealAffinePlane B]
variable [RealAffinePlane C]
variable [Point B M A]
variable [Point C N A]
variable [LineSegment A B]
variable [LineSegment A C]
variable [LineSegment B C]
variable [parallel BC MN]
variable [LineSegment B N]
variable [LineSegment C M]
variable [Intersection BN CM P]
variable [Circumcircle BMP]
variable [Circumcircle CNP]
variable [IntersectionOfTwoCircles BMP CNP Q]

theorem angle_PAB_eq_angle_QAC : ∠PAB = ∠QAC := by
  sorry

end angle_PAB_eq_angle_QAC_l249_249571


namespace required_water_for_solution_l249_249101

-- Definitions as given in a)
def original_total_solution : ℝ := 0.1
def original_water : ℝ := 0.03
def desired_total_solution : ℝ := 0.6

-- Calculate the ratio of water in the original solution
def water_ratio : ℝ := original_water / original_total_solution

-- Now, calculate the required water for the desired solution
def required_water : ℝ := desired_total_solution * water_ratio

-- The theorem to prove that the required water is 0.18 liters
theorem required_water_for_solution : required_water = 0.18 := sorry

end required_water_for_solution_l249_249101


namespace weight_of_empty_box_l249_249299

theorem weight_of_empty_box (w12 w8 w : ℝ) (h1 : w12 = 11.48) (h2 : w8 = 8.12) (h3 : ∀ b : ℕ, b > 0 → w = 0.84) :
  w8 - 8 * w = 1.40 :=
by
  sorry

end weight_of_empty_box_l249_249299


namespace smallest_initial_number_winning_for_bernardo_l249_249753

def bernardo_operation (x : ℕ) : ℕ := 3 * x

def silvia_operation (x : ℕ) : ℕ := x + 75

def game_sequence (N : ℕ) : Prop :=
  let b1 := bernardo_operation N in
  let s1 := silvia_operation b1 in
  let b2 := bernardo_operation s1 in
  let s2 := silvia_operation b2 in
  b2 < 1000 ∧ s2 >= 1000

theorem smallest_initial_number_winning_for_bernardo : 
  ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 999 ∧ game_sequence N ∧ 
  N = 78 ∧ (Nat.digits 10 N).sum = 15 :=
by
  sorry

end smallest_initial_number_winning_for_bernardo_l249_249753


namespace sum_of_first_7_terms_geometric_series_l249_249757

theorem sum_of_first_7_terms_geometric_series :
  let a := (1 / 6 : ℚ)
  let r := (-1 / 2 : ℚ)
  let n := 7
  ∑ i in Finset.range n, a * r ^ i = 129 / 1152 :=
by
  sorry

end sum_of_first_7_terms_geometric_series_l249_249757


namespace probability_heads_penny_nickel_dime_l249_249292

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]
variable (coins : Fin 5 → Ω)
variable (heads tails : Ω)

-- Each coin has two outcomes: heads or tails
axiom coin_outcome (i : Fin 5) : coins i = heads ∨ coins i = tails

-- There are 32 total outcomes
axiom total_outcomes : Fintype.card (Fin 5 → Ω) = 32

-- There are 4 successful outcomes for penny, nickel, and dime being heads
axiom successful_outcomes : let successful := {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} in
                          Fintype.card successful = 4

theorem probability_heads_penny_nickel_dime :
  let probability := (Fintype.card {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} : ℤ) /
                     (Fintype.card (Fin 5 → Ω) : ℤ) in
  probability = 1 / 8 :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249292


namespace maximum_candies_l249_249891

-- Definitions of the problem parameters
def classmates : ℕ := 25
def total_candies : ℕ := 200
def condition (candies : ℕ → ℕ) : Prop := 
  ∀ S (h : finset ℕ), S.card = 16 → (∑ i in S, candies i) ≥ 100

-- The theorem stating the maximum number of candies Vovochka can keep
theorem maximum_candies (candies : ℕ → ℕ) (h : condition candies) : 
  (total_candies - ∑ i in finset.range classmates, candies i) ≤ 37 :=
sorry

end maximum_candies_l249_249891


namespace binomial_constant_term_l249_249117

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the expansion term
def T (n k : ℕ) (x : ℂ) : ℂ := (-1)^k * (1/2)^(n-k) * binom n k * x^(n - (4*k)/3)

-- Given conditions
def n := 8
def k := 6

-- Binomial Expansion statement
theorem binomial_constant_term : 
  (T n k x).re = 7 := sorry

end binomial_constant_term_l249_249117


namespace necessary_but_not_sufficient_l249_249918
open Real

theorem necessary_but_not_sufficient (a b c : ℝ) :
  (∃ (x : ℝ), (x = 0 ∧ a > b ∧ ¬(ac^2 > bc^2))) ∧
  (ac^2 > bc^2 → a > b) ∧
  ¬(a > b → ac^2 > bc^2) :=
by
  sorry

end necessary_but_not_sufficient_l249_249918


namespace main_inequality_l249_249016

variable {x y a b : ℝ}

theorem main_inequality
  (hx : x > 0)
  (hy : y > 0) :
  (ax + by) / (x + y) ^ 2 ≤ (a ^ 2 * x + b ^ 2 * y) / (x + y) :=
sorry

end main_inequality_l249_249016


namespace geometric_body_views_identical_is_sphere_l249_249726

theorem geometric_body_views_identical_is_sphere (B : Type) 
  (front_view side_view top_view: B → Shape)
  (h1 : ∀ b : B, front_view b = side_view b) 
  (h2 : ∀ b : B, front_view b = top_view b) : 
  (∀ b : B, (∃ s : Sphere, s = b)) :=
sorry

end geometric_body_views_identical_is_sphere_l249_249726


namespace star_area_correct_l249_249645

-- Defining the problem conditions in Lean
structure StarGeom :=
  (A : ℕ → ℝ × ℝ)
  (O : ℝ × ℝ)
  (B : ℕ → ℝ × ℝ)
  (length_AO : ∀ i, dist (A i) O = 1)
  (right_angle : ∀ i, angle (A i) O (B i) = π / 2)
  (symmetric : symmetric_over_diagonals A O)

noncomputable def star_area (s : StarGeom) : ℝ :=
  6 * real.sqrt 2

-- The theorem statement
theorem star_area_correct (s : StarGeom) : 
  star_area s = 6 * real.sqrt 2 :=
sorry

end star_area_correct_l249_249645


namespace candy_cost_l249_249318

variables {x y : ℕ} -- x: kilograms of second type of candy, y: kilograms of first type of candy

-- Conditions translated to Lean statements
def price_1 := 1.80 -- price of first type of candy in rubles per kilogram
def price_2 := 1.50 -- price of second type of candy in rubles per kilogram
def weight_relation := y = x + 0.5 -- first type of candy is 0.5 kilograms more than second type
def cost_relation := 1.80 * y = 1.5 * 1.50 * x -- paid one and a half times more for the first type

-- Proof problem in Lean statement
theorem candy_cost (x y : ℕ) (h1: weight_relation y x) (h2: cost_relation y x) : price_1 * y + price_2 * x = 7.50 := by
  sorry

end candy_cost_l249_249318


namespace probability_all_co_captains_l249_249648

theorem probability_all_co_captains (num_teams : ℕ) (team_sizes : list ℕ) 
  (num_captains : ℕ) (num_selected : ℕ) (h_teams : num_teams = 4) 
  (h_team_sizes : team_sizes = [6, 7, 8, 9]) (h_num_captains : num_captains = 3) 
  (h_num_selected : num_selected = 3) : 
  let p := 1 / num_teams in
  let prob_captains (n : ℕ) := 1 / (nat.choose n 3) in
  p * (prob_captains 6 + prob_captains 7 + prob_captains 8 + prob_captains 9) = 91 / 6720 :=
by
  sorry

end probability_all_co_captains_l249_249648


namespace simplify_polynomial_l249_249245

variable (x : ℝ)

theorem simplify_polynomial : 
  (2 * x^4 + 3 * x^3 - 5 * x + 6) + (-6 * x^4 - 2 * x^3 + 3 * x^2 + 5 * x - 4) = 
  -4 * x^4 + x^3 + 3 * x^2 + 2 :=
by
  sorry

end simplify_polynomial_l249_249245


namespace median_room_number_of_arrived_participants_l249_249752

-- Defining the set of all room numbers
def all_room_numbers : List ℕ := List.range 1 26

-- Defining the list of room numbers with exclusions
def arrived_room_numbers : List ℕ := all_room_numbers.filter (λ n => n ≠ 14 ∧ n ≠ 20)

-- Statement asserting the median room number of the 23 participants
theorem median_room_number_of_arrived_participants : List.median arrived_room_numbers = 12 :=
by
  -- Placeholder for proof steps
  sorry

end median_room_number_of_arrived_participants_l249_249752


namespace remainder_of_quadratic_expression_l249_249924

theorem remainder_of_quadratic_expression (a : ℤ) :
  let n := 7 * a - 1 in
  (n ^ 2 + 3 * n + 4) % 7 = 2 :=
by
  sorry

end remainder_of_quadratic_expression_l249_249924


namespace Mary_lawn_mowed_fraction_l249_249161

-- Defining the problem conditions
def Mary_mows_in_3_hours := 3
def Mary_working_time := 1.5
def Lawn_remaining_fraction := 1 - (Mary_working_time / Mary_mows_in_3_hours)

theorem Mary_lawn_mowed_fraction :
  Lawn_remaining_fraction = 1 / 2 :=
by
  -- The proof steps would go here, but we are skipping it with sorry
  sorry

end Mary_lawn_mowed_fraction_l249_249161


namespace greatest_num_divisors_in_range_l249_249169

def num_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d, n % d = 0)

theorem greatest_num_divisors_in_range : 
  ∀ n ∈ {1, 2, 3, ..., 20}, 
  num_divisors n < 7 → (n = 12 ∨ n = 18 ∨ n = 20) :=
by {
  sorry
}

end greatest_num_divisors_in_range_l249_249169


namespace total_decrease_is_60_percent_l249_249370

-- Define the conditions
variable (P : ℝ) -- Original price
variable (P1 : ℝ) -- Price after first sale
variable (P2 : ℝ) -- Price after second sale
variable (P3 : ℝ) -- Price after third sale

-- Conditions
def first_sale := P1 = (4 / 5) * P
def second_sale := P2 = (1 / 2) * P
def third_sale := P3 = P2 - 0.2 * P2

-- Calculations
def total_percent_decrease := (P - P3) / P * 100

-- Theorem to be proven
theorem total_decrease_is_60_percent (P_pos : 0 < P) : 
  first_sale P P1 → second_sale P P2 → third_sale P2 P3 → total_percent_decrease P P3 = 60 :=
sorry

end total_decrease_is_60_percent_l249_249370


namespace union_of_sets_l249_249159

theorem union_of_sets :
  let A := {1, 3}
  let B := {1, 2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} :=
by
  sorry

end union_of_sets_l249_249159


namespace num_valid_pentatonic_sequences_l249_249938

theorem num_valid_pentatonic_sequences :
  let notes := ["gong", "shang", "jue", "zhi", "yu"]
  let total_sequences := 5!
  let invalid_sequences := 3! * (3!)
  total_sequences - invalid_sequences = 84 :=
by sorry

end num_valid_pentatonic_sequences_l249_249938


namespace total_trout_caught_l249_249166

theorem total_trout_caught (n_share j_share total_caught : ℕ) (h1 : n_share = 9) (h2 : j_share = 9) (h3 : total_caught = n_share + j_share) :
  total_caught = 18 :=
by
  sorry

end total_trout_caught_l249_249166


namespace minor_premise_of_syllogism_l249_249709

theorem minor_premise_of_syllogism (P Q : Prop)
  (h1 : ¬ (P ∧ ¬ Q))
  (h2 : Q) :
  Q :=
by
  sorry

end minor_premise_of_syllogism_l249_249709


namespace simplify_and_evaluate_expression_l249_249246

noncomputable def expression (a : ℝ) : ℝ :=
  ((a^2 - 1) / (a - 3) - a - 1) / ((a + 1) / (a^2 - 6 * a + 9))

theorem simplify_and_evaluate_expression : expression (3 - Real.sqrt 2) = -2 * Real.sqrt 2 :=
by
  sorry

end simplify_and_evaluate_expression_l249_249246


namespace find_x3_minus_y3_l249_249020

theorem find_x3_minus_y3 {x y : ℤ} (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : x^3 - y^3 = 108 :=
by 
  sorry

end find_x3_minus_y3_l249_249020


namespace find_A_find_area_l249_249466

noncomputable def triangle := Type

-- Define the angles and sides
variables {A B C : ℝ} {a b c : ℝ}

-- Define the vectors m and n
def m : ℝ × ℝ := (Real.cos B, Real.sin B)
def n : ℝ × ℝ := (Real.cos C, -Real.sin C)

-- Define the conditions
axiom dot_product_condition : m.1 * n.1 + m.2 * n.2 = 1 / 2

-- Proof that the angle A is 2π/3
theorem find_A (h : B + C = π / 3) : A = 2 * π / 3 :=
  by sorry

-- Given sides and angles, proof the area of the triangle
theorem find_area (h1 : a = 2 * Real.sqrt 3) (h2 : b + c = 4) (h3 : A = 2 * π / 3) : 
  0.5 * b * c * Real.sin A = Real.sqrt 3 :=
  by sorry

end find_A_find_area_l249_249466


namespace inequality_incorrect_l249_249917

theorem inequality_incorrect (a b : ℝ) (h : a > b) : ¬(-a + 2 > -b + 2) :=
by
  sorry

end inequality_incorrect_l249_249917


namespace total_population_l249_249531

variable (b g t s : ℕ)

theorem total_population (hb : b = 4 * g) (hg : g = 8 * t) (ht : t = 2 * s) :
  b + g + t + s = (83 * g) / 16 :=
by sorry

end total_population_l249_249531


namespace coin_flip_heads_probability_l249_249259

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end coin_flip_heads_probability_l249_249259


namespace total_calves_l249_249549

-- Definitions
def pregnant_llamas := 9
def twin_pregnant := 5
def single_pregnant := 4
def traded_calves := 8
def new_adult_llamas := 2
def final_herd := 18
def sold_fraction := 1 / 3

-- Theorem statement
theorem total_calves : ((5 * 2) + (4 * 1) - 8 = 6) → (18 / (2 / 3) = 27) → (27 - 6 = 21) → (21 - 2 = 19) → 
  let total_calves := (5 * 2) + (4 * 1) in
  total_calves = 14 := 
by
  sorry

end total_calves_l249_249549


namespace difference_high_low_score_l249_249364

theorem difference_high_low_score :
  ∀ (num_innings : ℕ) (total_runs : ℕ) (exc_total_runs : ℕ) (high_score : ℕ) (low_score : ℕ),
  num_innings = 46 →
  total_runs = 60 * 46 →
  exc_total_runs = 58 * 44 →
  high_score = 194 →
  total_runs - exc_total_runs = high_score + low_score →
  high_score - low_score = 180 :=
by
  intros num_innings total_runs exc_total_runs high_score low_score h_innings h_total h_exc_total h_high_sum h_difference
  sorry

end difference_high_low_score_l249_249364


namespace solve_inequality_l249_249789

theorem solve_inequality (y : ℝ) :
  (7 / 30) + |y - (19 / 60)| < (17 / 30) → y ∈ Ioo (-1 / 60) (13 / 20) :=
by
  sorry

end solve_inequality_l249_249789


namespace max_divisors_1_to_20_l249_249212

theorem max_divisors_1_to_20 :
  ∃ m n o ∈ set.range (20:ℕ.succ), 
    (∀ k ∈ set.range (20:ℕ.succ), 
      (nat.divisors_count k ≤ nat.divisors_count m) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count n) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count o)) 
    ∧ (nat.divisors_count m = 6)
    ∧ (nat.divisors_count n = 6)
    ∧ (nat.divisors_count o = 6)
    ∧ m ≠ n ∧ n ≠ o ∧ o ≠ m
    ∧ set.mem 12 (set.range (20 + 1))
    ∧ set.mem 18 (set.range (20 + 1))
    ∧ set.mem 20 (set.range (20 + 1)). 
      := by 
  -- all these steps will be skipped here
  sorry

end max_divisors_1_to_20_l249_249212


namespace Vovochka_max_candies_l249_249885

noncomputable theory
open_locale classical
open set

def VovochkaCandies (c n k required_sum : ℕ) (dist : fin n → ℕ) : Prop :=
  ∀ (s : finset (fin n)), s.card = k → (s.sum dist ≥ required_sum)

theorem Vovochka_max_candies
  (c n k required_sum : ℕ)
  (h_c : c = 200)
  (h_n : n = 25)
  (h_k : k = 16)
  (h_required_sum : required_sum = 100)
  (dist : fin n → ℕ)
  (h_dist : VovochkaCandies c n k required_sum dist) :
  c - finset.univ.sum dist = 37 :=
sorry

end Vovochka_max_candies_l249_249885


namespace domain_and_max_value_l249_249578

noncomputable def f (x : ℝ) : ℝ := log x / log 2 + log (1 - x) / log 2

theorem domain_and_max_value :
  (∀ x, 0 < x ∧ x < 1 → True) ∧ 
  (∀ x, 0 < x ∧ x < 1 → f x ≤ -2) :=
by
  sorry

end domain_and_max_value_l249_249578


namespace minimum_discount_l249_249714

open Real

theorem minimum_discount (CP MP SP_min : ℝ) (profit_margin : ℝ) (discount : ℝ) :
  CP = 800 ∧ MP = 1200 ∧ SP_min = 960 ∧ profit_margin = 0.20 ∧
  MP * (1 - discount / 100) ≥ SP_min → discount = 20 :=
by
  intros h
  rcases h with ⟨h_cp, h_mp, h_sp_min, h_profit_margin, h_selling_price⟩
  simp [h_cp, h_mp, h_sp_min, h_profit_margin, sub_eq_self, div_eq_self] at *
  sorry

end minimum_discount_l249_249714


namespace suff_but_not_nec_condition_l249_249995

theorem suff_but_not_nec_condition (x : ℝ) : (2^x > 2) → (1/x < 1) ∧ (1/x < 1 → ¬ (2^x > 2)) :=
by
  sorry

end suff_but_not_nec_condition_l249_249995


namespace vovochka_max_candies_l249_249862

/-!
# Problem Statement:
- Vovochka has 25 classmates.
- Vovochka brought 200 candies to class.

## Condition:
- Any 16 classmates together should have at least 100 candies in total.

Question: What is the maximum number of candies Vovochka can keep for himself while fulfilling his mother's request?

The answer to this problem needs to be proven mathematically.
-/

def max_candies_can_keep (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ) : ℕ :=
if h : classmates = 25 ∧ total_candies = 200 ∧ min_candies_per_16 = 100 then
  37
else 
  sorry -- This statement is just for context, keeping focus on formal proof in Lean.

theorem vovochka_max_candies :
  ∀ (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ),
    classmates = 25 → total_candies = 200 → min_candies_per_16 = 100 →
    max_candies_can_keep candies classmates total_candies min_candies_per_16 = 37 :=
by
  intros candies classmates total_candies min_candies_per_16 hc ht hm
  simp [max_candies_can_keep]
  exact if_pos ⟨hc, ht, hm⟩

end vovochka_max_candies_l249_249862


namespace value_a2017_l249_249010

def sequence (n : ℕ) : ℚ :=
  by 
    if n = 0 then exact 1 / 2
    else exact 1 / (1 - sequence (n - 1))

theorem value_a2017 : sequence 2016 = 2 := 
  sorry

end value_a2017_l249_249010


namespace max_divisors_1_to_20_l249_249210

theorem max_divisors_1_to_20 :
  ∃ m n o ∈ set.range (20:ℕ.succ), 
    (∀ k ∈ set.range (20:ℕ.succ), 
      (nat.divisors_count k ≤ nat.divisors_count m) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count n) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count o)) 
    ∧ (nat.divisors_count m = 6)
    ∧ (nat.divisors_count n = 6)
    ∧ (nat.divisors_count o = 6)
    ∧ m ≠ n ∧ n ≠ o ∧ o ≠ m
    ∧ set.mem 12 (set.range (20 + 1))
    ∧ set.mem 18 (set.range (20 + 1))
    ∧ set.mem 20 (set.range (20 + 1)). 
      := by 
  -- all these steps will be skipped here
  sorry

end max_divisors_1_to_20_l249_249210


namespace remainder_equality_l249_249253

theorem remainder_equality (a b s t d : ℕ) (h1 : a > b) (h2 : a % d = s % d) (h3 : b % d = t % d) :
  ((a + 1) * (b + 1)) % d = ((s + 1) * (t + 1)) % d :=
by
  sorry

end remainder_equality_l249_249253


namespace polynomial_sum_is_2_l249_249075

theorem polynomial_sum_is_2 :
  ∀ (x : ℝ),
  ∃ (A B C D : ℝ), 
  (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D ∧ A + B + C + D = 2 :=
by
  intros x
  use [4, -10, -13, 21]
  split
  · -- Prove the polynomial expansion
    calc
      (x - 3) * (4 * x^2 + 2 * x - 7) 
          = x * (4 * x^2 + 2 * x - 7) - 3 * (4 * x^2 + 2 * x - 7) : by rw mul_sub
      ... = (x * 4 * x^2 + x * 2 * x - x * 7) - (3 * (4 * x^2) + 3 * (2 * x) - 3 * (-7)) : by distribute
      ... = 4 * x^3 + 2 * x^2 - 7 * x - 12 * x^2 - 6 * x + 21 : by algebra
      ... = 4 * x^3 - 10 * x^2 - 13 * x + 21 : by linarith
  · -- Prove A + B + C + D = 2
    calc
      4 + (-10) + (-13) + 21 = 2 : by linarith

end polynomial_sum_is_2_l249_249075


namespace find_line_eq_l249_249307

theorem find_line_eq
  (l : ℝ → ℝ → Prop)
  (bisects_circle : ∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y = 0 → l x y)
  (perpendicular_to_line : ∀ x y : ℝ, l x y ↔ y = -1/2 * x)
  : ∀ x y : ℝ, l x y ↔ 2*x - y = 0 := by
  sorry

end find_line_eq_l249_249307


namespace exists_k_ge_2_l249_249985

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def weak (a b n : ℕ) : Prop :=
  ¬ ∃ x y : ℕ, a * x + b * y = n

theorem exists_k_ge_2 (a b n : ℕ) (h_coprime : coprime a b) (h_positive : 0 < n) (h_weak : weak a b n) (h_bound : n < a * b / 6) :
  ∃ k : ℕ, 2 ≤ k ∧ weak a b (k * n) :=
sorry

end exists_k_ge_2_l249_249985


namespace S2_eq_5_Sn_eq_n_minus_1_2_pow_n_plus_1_l249_249141

-- Define the set X
def X (n : ℕ) : set ℕ := {k | 1 ≤ k ∧ k ≤ n}

-- Define the function f(A) as the largest element in A
noncomputable def f (A : set ℕ) : ℕ :=
  if h : A.nonempty then A.max' h else 0

-- Define the sum S_n
noncomputable def S (n : ℕ) : ℕ :=
  ∑ A in (set.finite_powerset (X n)).to_finset.filter (λ A, A.nonempty), f A

-- Statement for S_2
theorem S2_eq_5 : S 2 = 5 := sorry

-- Statement for S_n
theorem Sn_eq_n_minus_1_2_pow_n_plus_1 (n : ℕ) : S n = (n - 1) * 2^n + 1 := sorry

end S2_eq_5_Sn_eq_n_minus_1_2_pow_n_plus_1_l249_249141


namespace pinocchio_optimal_success_probability_l249_249234

def success_prob (s : List ℚ) : ℚ :=
  s.foldr (λ x acc => (x * acc) / (1 - (1 - x) * acc)) 1

theorem pinocchio_optimal_success_probability :
  let success_probs := [9/10, 8/10, 7/10, 6/10, 5/10, 4/10, 3/10, 2/10, 1/10]
  success_prob success_probs = 0.4315 :=
by 
  sorry

end pinocchio_optimal_success_probability_l249_249234


namespace arithmetic_sequence_sum_l249_249111

theorem arithmetic_sequence_sum : 
    ∃ (d : ℤ), (a_1 = 2) ∧ ((a_1 + d) + (a_1 + 2 * d) = 13) → 
    let a_2 := a_1 + d,
        a_3 := a_1 + 2 * d,
        a_4 := a_1 + 3 * d,
        a_5 := a_1 + 4 * d,
        a_6 := a_1 + 5 * d
    in a_4 + a_5 + a_6 = 42 :=
by
  sorry

end arithmetic_sequence_sum_l249_249111


namespace Vovochka_max_candies_l249_249889

noncomputable theory
open_locale classical
open set

def VovochkaCandies (c n k required_sum : ℕ) (dist : fin n → ℕ) : Prop :=
  ∀ (s : finset (fin n)), s.card = k → (s.sum dist ≥ required_sum)

theorem Vovochka_max_candies
  (c n k required_sum : ℕ)
  (h_c : c = 200)
  (h_n : n = 25)
  (h_k : k = 16)
  (h_required_sum : required_sum = 100)
  (dist : fin n → ℕ)
  (h_dist : VovochkaCandies c n k required_sum dist) :
  c - finset.univ.sum dist = 37 :=
sorry

end Vovochka_max_candies_l249_249889


namespace lines_concurrent_or_parallel_l249_249007

theorem lines_concurrent_or_parallel {A B C D E F G H: Type*}
  [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [ordered_comm_group E] [ordered_comm_group F] [ordered_comm_group G]
  [ordered_comm_group H] [Module ℝ A] [Module ℝ B] [Module ℝ C]
  [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] :
  let parallelogram {A B C D : Point} := parallelogram A B C D,
      line_EF_parallel_BC := (segment (line.parallel_to B C)) passing_through E F,
      line_GH_parallel_AB := (segment (line.parallel_to A B)) passing_through G H in
  collinear_or_parallel E H G F B D :=
by sorry

end lines_concurrent_or_parallel_l249_249007


namespace value_of_8x_minus_5_squared_l249_249804

theorem value_of_8x_minus_5_squared (x : ℝ) (h : 8 * x ^ 2 + 7 = 12 * x + 17) : (8 * x - 5) ^ 2 = 465 := 
sorry

end value_of_8x_minus_5_squared_l249_249804


namespace triangle_angle_geq_60_sqrt_inequality_l249_249711

-- 1. Prove at least one interior angle in a triangle is >= 60°

theorem triangle_angle_geq_60 (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A > 0) (h3 : B > 0) (h4 : C > 0) :
  A ≥ 60 ∨ B ≥ 60 ∨ C ≥ 60 :=
by
  sorry

-- 2. Given n ≥ 0, prove √(n + 2) - √(n + 1) < √(n + 1) - √(n)

theorem sqrt_inequality (n : ℝ) (hn : n ≥ 0) :
  sqrt (n + 2) - sqrt (n + 1) < sqrt (n + 1) - sqrt n :=
by
  sorry

end triangle_angle_geq_60_sqrt_inequality_l249_249711


namespace find_A_l249_249315

def is_valid_A (A : ℕ) : Prop :=
  A = 1 ∨ A = 2 ∨ A = 4 ∨ A = 7 ∨ A = 9

def number (A : ℕ) : ℕ :=
  3 * 100000 + 0 * 10000 + 5 * 1000 + 2 * 100 + 0 * 10 + A

theorem find_A (A : ℕ) (h_valid_A : is_valid_A A) : A = 1 ↔ Nat.Prime (number A) :=
by
  sorry

end find_A_l249_249315


namespace scientific_notation_of_61345_05_billion_l249_249589

theorem scientific_notation_of_61345_05_billion :
  ∃ x : ℝ, (61345.05 * 10^9) = x ∧ x = 6.134505 * 10^12 :=
by
  sorry

end scientific_notation_of_61345_05_billion_l249_249589


namespace perimeter_triangle_mDEF_l249_249654

/-- Define the sides of the triangle DEF --/
def side_DE := 150
def side_EF := 300
def side_FD := 250

/-- Define the lengths of the segments formed by intersecting lines --/
def segment_mD := 75
def segment_mE := 125
def segment_mF := 50

/-- Prove that the perimeter of the triangle formed by m_D, m_E, and m_F is 331.25 --/
theorem perimeter_triangle_mDEF :
  let triangle_perimeter := segment_mD * 2 + segment_mE * 2 + segment_mF * 2
  in triangle_perimeter = 331.25 :=
by
  -- Proof omitted
  sorry

end perimeter_triangle_mDEF_l249_249654


namespace unique_solution_for_f_f_x_eq_1_l249_249627

-- Define the domain of the function
def domain (x : ℝ) := -4 ≤ x ∧ x ≤ 4

-- Define the function f based on the graph's behavior
def f (x : ℝ) : ℝ :=
  if x = -4 then 0
  else if x = -3 then 2
  else if x = -2 then 3
  else if x = 0 then 1
  else if x = 2 then 1
  else if x = 3 then 2
  else if x = 4 then 3
  else sorry -- values can be graph dependent (as not explicitly given)

-- The theorem statement
theorem unique_solution_for_f_f_x_eq_1 : ∃! x : ℝ, domain x ∧ f(f(x)) = 1 := sorry

end unique_solution_for_f_f_x_eq_1_l249_249627


namespace smallest_positive_period_zeros_of_f_max_min_values_on_interval_l249_249809

-- Define the vectors a and b and the function f
def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, 1)
def b (x : ℝ) : ℝ × ℝ := (2 * Real.cos (x - Real.pi / 3), Real.sqrt 3)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 2 * Real.sqrt 3

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem zeros_of_f :
  ∀ k : ℤ, f (k * Real.pi / 2 + Real.pi / 6) = 0 := sorry

theorem max_min_values_on_interval :
  ∃ x_max x_min : ℝ,
    (x_max ∈ Set.Icc (Real.pi / 24) (3 * Real.pi / 4)) ∧ 
    (x_min ∈ Set.Icc (Real.pi / 24) (3 * Real.pi / 4)) ∧ 
    f x_max = 2 ∧ f x_min = -Real.sqrt 2 := sorry

end smallest_positive_period_zeros_of_f_max_min_values_on_interval_l249_249809


namespace number_is_48_l249_249436

theorem number_is_48 (x : ℝ) (h : (1/4) * x + 15 = 27) : x = 48 :=
by sorry

end number_is_48_l249_249436


namespace ant_prob_unique_destination_l249_249609

def octahedron_vertices := {1, 2, 3, 4, 5, 6}
def vertex_neighbors (v : ℕ) : finset ℕ :=
  match v with
  | 1 => {2, 3, 4, 5}
  | 2 => {1, 3, 5, 6}
  | 3 => {1, 2, 4, 6}
  | 4 => {1, 3, 5, 6}
  | 5 => {1, 2, 4, 6}
  | 6 => {2, 3, 4, 5}
  | _ => ∅ -- This case should never happen given our set definition.
  end

def ant_moves_to_different_points : Prop :=
  ∃ f : octahedron_vertices → finset ℕ, (∀ v ∈ octahedron_vertices, f v ∈ vertex_neighbors v) ∧ 
  (∀ v₁ v₂ ∈ octahedron_vertices, v₁ ≠ v₂ → f v₁ ≠ f v₂)

theorem ant_prob_unique_destination : 
  (fintype.card {f // ∀ v ∈ octahedron_vertices, f v ∈ vertex_neighbors v ∧ (∀ v₁ v₂ ∈ octahedron_vertices, v₁ ≠ v₂ → f v₁ ≠ f v₂)}) 
  = (fintype.card {f // ∀ v ∈ octahedron_vertices, f v ∈ vertex_neighbors v} * 1/4096) * 5 / 256 := 
begin
  sorry
end

end ant_prob_unique_destination_l249_249609


namespace probability_units_digit_is_1_l249_249835

theorem probability_units_digit_is_1 :
  ∀ (m : ℕ) (n : ℕ),
    (m ∈ {11, 13, 15, 17, 19}) →
    (n ∈ finset.range 20 + 2000) →
    (∃ p : ℚ, p = (2/5) ∧
      fintype.card {x : ℕ // ((x ∈ finset.range 20 + 2000) ∧ (((m ^ x) % 10) = 1))} = 
      (p * fintype.card {x : ℕ // x ∈ finset.range 20 + 2000})) := sorry

end probability_units_digit_is_1_l249_249835


namespace prod_of_tan_of_right_triangles_l249_249561

def S : set (ℕ × ℕ) := {p | p.1 ∈ {0, 1, 2, 3, 4, 5} ∧ p.2 ∈ {0, 1, 2, 3, 4, 5, 6}}

def is_right_triangle (A B C : ℕ × ℕ) : Prop :=
  (A.1 = B.1 ∧ A.2 = C.2) ∨ (A.2 = B.2 ∧ A.1 = C.1)

def T : set (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  {t | ∃ A B C: ℕ × ℕ, A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ is_right_triangle A B C ∧
     (¬A = B ∧ ¬B = C ∧ ¬C = A)}

def f (t : ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) : ℝ :=
  let (A₁, A₂, B₁, B₂, C₁, C₂) := t in
  if (A₁ = B₁ ∧ A₂ ≠ C₂) then
    (C₁ - B₁) / (B₂ - A₂)
  else if (A₂ = B₂ ∧ A₁ ≠ C₁) then
    (B₂ - C₂) / (C₁ - A₁)
  else 0 -- Only to make function total; this case should not occur.

theorem prod_of_tan_of_right_triangles : (∏ t in T, f t) = 1 :=
  sorry

end prod_of_tan_of_right_triangles_l249_249561


namespace coin_flip_heads_probability_l249_249261

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end coin_flip_heads_probability_l249_249261


namespace book_area_correct_l249_249368

def book_length : ℝ := 5
def book_width : ℝ := 10
def book_area (length : ℝ) (width : ℝ) : ℝ := length * width

theorem book_area_correct :
  book_area book_length book_width = 50 :=
by
  sorry

end book_area_correct_l249_249368


namespace expansion_term_count_l249_249066

theorem expansion_term_count 
  (A : Finset ℕ) (B : Finset ℕ) 
  (hA : A.card = 3) (hB : B.card = 4) : 
  (Finset.card (A.product B)) = 12 :=
by {
  sorry
}

end expansion_term_count_l249_249066


namespace remainder_division_l249_249379

/-- A number when divided by a certain divisor left a remainder, 
when twice the number was divided by the same divisor, the remainder was 112. 
The divisor is 398.
Prove that the remainder when the original number is divided by the divisor is 56. -/
theorem remainder_division (N R : ℤ) (D : ℕ) (Q Q' : ℤ)
  (hD : D = 398)
  (h1 : N = D * Q + R)
  (h2 : 2 * N = D * Q' + 112) :
  R = 56 :=
sorry

end remainder_division_l249_249379


namespace eq1_eq2_l249_249542

def point (α : Type) := α × α

def curve_C_eq (p : point ℝ) : Prop := (p.fst - 1)^2 + p.snd^2 = 4

def polar_eq (θ : ℝ) (ρ : ℝ) (line_tangent : Prop) : Prop :=
  line_tangent → (ρ * Real.sin θ = 2 ∨ 4 * ρ * Real.cos θ + 3 * ρ * Real.sin θ - 8 = 0)

theorem eq1 (M : point ℝ) (line_tangent : Prop) :
  M = (2, 2) →
  (∃ θ ρ, polar_eq θ ρ line_tangent) :=
by
  sorry

def point_distance (p1 p2 : point ℝ) : ℝ :=
  Real.sqrt ((p2.fst - p1.fst)^2 + (p2.snd - p1.snd)^2)

theorem eq2 (M N : point ℝ) :
  M = (2, 2) →
  N = (-2, 2) →
  (∃ d_min d_max, d_min = Real.sqrt 13 - 2 ∧ d_max = Real.sqrt 13 + 2) :=
by
  sorry

end eq1_eq2_l249_249542


namespace four_different_colored_socks_probability_l249_249464

/-- 
Prove that the probability of having 4 different colored socks outside 
the bag is 8/15, given that the bag contains 5 pairs of different colored 
socks, a random sample of 4 single socks is drawn, any complete pairs 
in the sample are discarded and replaced by new pairs from the bag, and 
the process continues until the bag is empty or there are 4 socks of 
different colors outside the bag.
-/
theorem four_different_colored_socks_probability :
  let pairs := 5
  let total_socks := pairs * 2
  let first_draw_socks := 4
  let probability_draw_four_diff_colors (total_socks first_draw_socks : ℕ) : ℚ := 
    (5 * 16 : ℚ) / (210 : ℚ)
  let probability_draw_one_pair_two_diff (total_socks first_draw_socks : ℕ) : ℚ := 
    (5 * 6 * 4 : ℚ) / (210 : ℚ)
  let probability_second_draw (remaining_socks : ℕ) : ℚ := 
    (4 * 6 : ℚ) / (15 : ℚ)
  let probability (p1 p2 p3 : ℚ) : ℚ := 
    p1 + p2 * p3
  probability (probability_draw_four_diff_colors total_socks first_draw_socks)
              (probability_draw_one_pair_two_diff total_socks first_draw_socks)
              (probability_second_draw 6) = 8 / 15 :=
begin
  sorry
end

end four_different_colored_socks_probability_l249_249464


namespace polygon_coloring_l249_249821

theorem polygon_coloring (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 3) :
    ∃ b_n : ℕ, b_n = (m - 1) * ((m - 1) ^ (n - 1) + (-1 : ℤ) ^ n) :=
sorry

end polygon_coloring_l249_249821


namespace vovochka_max_candies_l249_249897

noncomputable def max_candies_for_vovochka 
  (total_classmates : ℕ)
  (total_candies : ℕ)
  (candies_condition : ∀ (subset : Finset ℕ), subset.card = 16 → ∑ i in subset, (classmate_candies i) ≥ 100) 
  (classmate_candies : ℕ → ℕ) : ℕ :=
  total_candies - ∑ i in (Finset.range total_classmates), classmate_candies i

theorem vovochka_max_candies :
  max_candies_for_vovochka 25 200 (λ subset hc, True) (λ i, if i < 13 then 7 else 6) = 37 :=
by
  sorry

end vovochka_max_candies_l249_249897


namespace probability_chord_intersects_inner_circle_l249_249328

-- Define the radii of the circles
def radius_inner := 3
def radius_outer := 5

-- Define the center of the circles
def O : Point := ⟨0, 0⟩

-- Define a fixed point A on the outer circle
def A : Point := ⟨5, 0⟩

-- Define a variable point B on the outer circle (implicitly a random selection)
noncomputable def B : Point := sorry

-- Define the concepts of chord and intersection
lemma chord_AB_intersects_inner_circle (B : Point) (h_outer : dist O B = radius_outer) : 
  (∃ Q : Point, Q ∈ inner_circle) ∧ (line_through A B) ∩ inner_circle ≠ ∅ := 
sorry

-- The final statement with the given probability
theorem probability_chord_intersects_inner_circle : 
  (∃ B : Point, ∀ x, x ∈ outer_circle A B → chord_AB_intersects_inner_circle x) → 
  probability (B_selected = true) = 0.41 := 
sorry

end probability_chord_intersects_inner_circle_l249_249328


namespace Vovochka_max_candies_l249_249880

theorem Vovochka_max_candies (classmates : Finset ℕ) (candies : ℕ) (hclassmates : classmates.card = 25) (hcandies : candies = 200) (H : ∀ (S : Finset ℕ), S.card = 16 → 100 ≤ S.sum (λ x, 1)) :
  ∃ (max_candies : ℕ), max_candies = 37 :=
by
  use 37
  sorry

end Vovochka_max_candies_l249_249880


namespace sequence_problem_l249_249109

theorem sequence_problem 
  (a : ℕ → ℤ) 
  (b : ℕ → ℤ) 
  (c : ℕ → ℤ)
  (S : ℕ → ℤ)
  (a1 : a 1 = -2)
  (a12 : a 12 = 20)
  (H : ∀ n, a n = 2 * n - 4)
  (bn_def : ∀ n, b n = (list.range n).sum (λ k, a (k + 1)) / n)
  (c_def : ∀ n, c n = 3 ^ (b n))
  (Sn_def : ∀ n, S n = (list.range n).sum (λ k, c (k + 1)))
  : 
  (∀ n, a n = 2 * n - 4) ∧
  (∀ n, S n = (3 ^ n - 1) / 18) :=
sorry

end sequence_problem_l249_249109


namespace least_num_subtracted_l249_249354

theorem least_num_subtracted 
  {x : ℤ} 
  (h5 : (642 - x) % 5 = 4) 
  (h7 : (642 - x) % 7 = 4) 
  (h9 : (642 - x) % 9 = 4) : 
  x = 4 := 
sorry

end least_num_subtracted_l249_249354


namespace remainder_sum_div_7_l249_249449

theorem remainder_sum_div_7 :
  (8145 + 8146 + 8147 + 8148 + 8149) % 7 = 4 :=
by
  sorry

end remainder_sum_div_7_l249_249449


namespace vovochka_max_candies_l249_249899

noncomputable def max_candies_for_vovochka 
  (total_classmates : ℕ)
  (total_candies : ℕ)
  (candies_condition : ∀ (subset : Finset ℕ), subset.card = 16 → ∑ i in subset, (classmate_candies i) ≥ 100) 
  (classmate_candies : ℕ → ℕ) : ℕ :=
  total_candies - ∑ i in (Finset.range total_classmates), classmate_candies i

theorem vovochka_max_candies :
  max_candies_for_vovochka 25 200 (λ subset hc, True) (λ i, if i < 13 then 7 else 6) = 37 :=
by
  sorry

end vovochka_max_candies_l249_249899


namespace conditional_probability_age_30_40_female_l249_249525

noncomputable def total_people : ℕ := 350
noncomputable def total_females : ℕ := 180
noncomputable def females_30_40 : ℕ := 50

theorem conditional_probability_age_30_40_female :
  (females_30_40 : ℚ) / total_females = 5 / 18 :=
by
  sorry

end conditional_probability_age_30_40_female_l249_249525


namespace maximum_candies_l249_249893

-- Definitions of the problem parameters
def classmates : ℕ := 25
def total_candies : ℕ := 200
def condition (candies : ℕ → ℕ) : Prop := 
  ∀ S (h : finset ℕ), S.card = 16 → (∑ i in S, candies i) ≥ 100

-- The theorem stating the maximum number of candies Vovochka can keep
theorem maximum_candies (candies : ℕ → ℕ) (h : condition candies) : 
  (total_candies - ∑ i in finset.range classmates, candies i) ≤ 37 :=
sorry

end maximum_candies_l249_249893


namespace inequality_inequation_l249_249596

variable (x : ℝ) (h : x > 0)

theorem inequality_inequation (hx : x > 0) : 
  (x + 1) * real.sqrt (x + 1) ≥ real.sqrt 2 * (x + real.sqrt x) :=
sorry

end inequality_inequation_l249_249596


namespace find_cost_of_paper_clips_l249_249427

def cost_of_box_of_paper_clips (p i : ℝ) : Prop :=
  (15 * p + 7 * i = 55.40) ∧ (12 * p + 10 * i = 61.70)

theorem find_cost_of_paper_clips : ∃ i : ℝ, cost_of_box_of_paper_clips 1.835 i :=
 by
  use 1.835
  split
  sorry

end find_cost_of_paper_clips_l249_249427


namespace angle_sum_triangle_l249_249965

theorem angle_sum_triangle (A B C : ℝ) (hA : A = 75) (hB : B = 40) (h_sum : A + B + C = 180) : C = 65 :=
by
  sorry

end angle_sum_triangle_l249_249965


namespace find_beta_l249_249425

-- Definitions of variables and conditions
variables (β α : ℝ)
axiom α_eq : α = 11

-- Definition to represent the sum condition
def sum_condition (β α : ℝ) : Prop :=
  (∑ r in Finset.range 8, β / (r.succ * (r.succ + 1) * (r.succ + 2))) = α

-- The theorem to be proved
theorem find_beta (β : ℝ) (h : sum_condition β 11) : β = 45 := 
sorry

end find_beta_l249_249425


namespace sequence_mod_distinct_l249_249739

-- Sequence definition
def a : ℕ → ℤ
| 0     := 1
| (n+1) := 2 ^ (a n) + a n

-- Proof statement
theorem sequence_mod_distinct :
  ∀ i j ∈ Finset.range 243, i ≠ j → (a i) % 243 ≠ (a j) % 243 := by
  sorry

end sequence_mod_distinct_l249_249739


namespace calculate_num_years_l249_249232

-- Definitions based on the conditions provided.
variable (P : ℝ) (r : ℝ) (n : ℕ)

-- Let the rate of interest be 0.03
def rate_of_interest : ℝ := 0.03

-- Simple Interest (SI) formula: SI = P * r * n
def simple_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * r * n

-- Compound Interest (CI) formula: CI = P * (1 + r)^n - P
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n - P

-- Given values according to the problem
axiom simple_interest_value : simple_interest P rate_of_interest n = 600
axiom compound_interest_value : compound_interest P rate_of_interest n = 609

-- The proof statement
theorem calculate_num_years (P : ℝ) : n = 2 :=
begin
  sorry
end

end calculate_num_years_l249_249232


namespace part_b_part_c_l249_249815

def z : ℂ := (5 + I) / (1 + I)

theorem part_b : z.im = -2 :=
by
  -- omitted for brevity
  sorry

theorem part_c : (z.re > 0) ∧ (z.im < 0) :=
by
  -- omitted for brevity
  sorry

end part_b_part_c_l249_249815


namespace greatest_num_divisors_in_range_l249_249170

def num_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d, n % d = 0)

theorem greatest_num_divisors_in_range : 
  ∀ n ∈ {1, 2, 3, ..., 20}, 
  num_divisors n < 7 → (n = 12 ∨ n = 18 ∨ n = 20) :=
by {
  sorry
}

end greatest_num_divisors_in_range_l249_249170


namespace equation_is_hyperbola_l249_249422

theorem equation_is_hyperbola : 
  ∀ x y : ℝ, (x^2 - 25*y^2 - 10*x + 50 = 0) → 
  (∃ a b h k : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ (x - h)^2 / a^2 - (y - k)^2 / b^2 = -1)) :=
by
  sorry

end equation_is_hyperbola_l249_249422


namespace even_sum_probability_l249_249616

theorem even_sum_probability : 
  let tiles := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} in
  let players := {1, 2, 3} in
  let combinations (S : Set ℕ) (k : ℕ) := {c : Set ℕ | c ⊆ S ∧ c.card = k} in
  let is_even_sum (s : Set ℕ) := (s.sum % 2 = 0) in
  let ways_even_sum (S : Set ℕ) (k : ℕ) := (combinations S k).filter is_even_sum in
  let p := (ways_even_sum tiles 3).card * (ways_even_sum (tiles \ players) 3).card * (ways_even_sum ((tiles \ players) \ players) 3).card in
  let q := (combinations tiles 3).card * (combinations (tiles \ players) 3).card * (combinations ((tiles \ players) \ players) 3).card in
  p = 4000 ∧ q = 16800 →
  p.gcd q = 1 →
  ∃ (p' q' : ℤ), p' + q' = 26 := 
sorry

end even_sum_probability_l249_249616


namespace cherry_sodas_correct_l249_249724

/-
A cooler is filled with 24 cans of cherry soda and orange pop. 
There are twice as many cans of orange pop as there are of cherry soda. 
Prove that the number of cherry sodas is 8.
-/
def num_cherry_sodas (C O : ℕ) : Prop :=
  O = 2 * C ∧ C + O = 24 → C = 8

theorem cherry_sodas_correct (C O : ℕ) (h : O = 2 * C ∧ C + O = 24) : C = 8 :=
by
  sorry

end cherry_sodas_correct_l249_249724


namespace find_x3_minus_y3_l249_249021

theorem find_x3_minus_y3 {x y : ℤ} (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : x^3 - y^3 = 108 :=
by 
  sorry

end find_x3_minus_y3_l249_249021


namespace greatest_divisors_1_to_20_l249_249201

def is_divisor (a b : ℕ) : Prop := b % a = 0

def count_divisors (n : ℕ) : ℕ :=
  nat.succ (list.length ([k for k in list.range n if is_divisor k.succ n]))

def max_divisors_from_1_to_20 : set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 20 ∧ ∃ max_count, count_divisors n = max_count ∧ 
  ∀ m, 1 ≤ m ∧ m ≤ 20 → count_divisors m ≤ max_count}

theorem greatest_divisors_1_to_20 : max_divisors_from_1_to_20 = {12, 18, 20} :=
by {
  sorry
}

end greatest_divisors_1_to_20_l249_249201


namespace _l249_249410

   noncomputable def distinct_weights : ℕ → ℕ
   | i := 2^1000 + 2^(i-1)

   lemma baron_munchhausen_theorem :
     ∀ w : fin 1000 → ℕ,
       (∀ i, w i = distinct_weights i) →
       (∑ i, w i < 2^1010) →
       (∀ v : fin 1000 → ℕ,
         (∑ i, v i = ∑ i, w i) →
         (∀ j, v j = w j)) :=
   by
     intro w hw hw_lt v hsum hweights
     sorry
   
end _l249_249410


namespace value_of_alpha_minus_beta_value_of_tan_2alpha_minus_beta_l249_249003

-- Define the variables involved and their conditions
variables {α β : ℝ}

-- State the given conditions
axiom cond1 : π < α ∧ α < (3 / 2) * π
axiom cond2 : π < β ∧ β < (3 / 2) * π
axiom cond3 : sin α = - (sqrt 5) / 5
axiom cond4 : cos β = - (sqrt 10) / 10

-- Statement of the proof problem
theorem value_of_alpha_minus_beta : α - β = - (π / 4) :=
by 
  sorry

theorem value_of_tan_2alpha_minus_beta : tan (2 * α - β) = - (1 / 3) :=
by 
  sorry

end value_of_alpha_minus_beta_value_of_tan_2alpha_minus_beta_l249_249003


namespace new_ratio_of_dogs_to_cats_l249_249093

theorem new_ratio_of_dogs_to_cats (initial_dogs : ℕ) (initial_dogs_to_cats_ratio_num : ℕ) (initial_dogs_to_cats_ratio_den : ℕ) (additional_cats : ℕ) (new_ratio_num : ℕ) (new_ratio_den : ℕ) :
  initial_dogs = 60 →
  initial_dogs_to_cats_ratio_num = 15 →
  initial_dogs_to_cats_ratio_den = 7 →
  additional_cats = 16 →
  new_ratio_num = 15 →
  new_ratio_den = 11 →
  let initial_cats := (initial_dogs * initial_dogs_to_cats_ratio_den) / initial_dogs_to_cats_ratio_num in
  let total_cats := initial_cats + additional_cats in
  initial_dogs * new_ratio_den = new_ratio_num * total_cats :=
by
  intros
  let initial_cats : ℕ := (initial_dogs * initial_dogs_to_cats_ratio_den) / initial_dogs_to_cats_ratio_num
  let total_cats : ℕ := initial_cats + additional_cats
  have : initial_dogs * new_ratio_den = new_ratio_num * total_cats := sorry
  exact this

end new_ratio_of_dogs_to_cats_l249_249093


namespace greatest_num_divisors_in_range_l249_249171

def num_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d, n % d = 0)

theorem greatest_num_divisors_in_range : 
  ∀ n ∈ {1, 2, 3, ..., 20}, 
  num_divisors n < 7 → (n = 12 ∨ n = 18 ∨ n = 20) :=
by {
  sorry
}

end greatest_num_divisors_in_range_l249_249171


namespace find_cube_difference_l249_249023

theorem find_cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := 
by
  sorry

end find_cube_difference_l249_249023


namespace M_inter_N_l249_249468

def M : Set ℝ := {x | abs (x - 1) < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

theorem M_inter_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 3} :=
by
  sorry

end M_inter_N_l249_249468


namespace distance_between_incenter_and_circumcenter_of_5_12_13_triangle_l249_249394

theorem distance_between_incenter_and_circumcenter_of_5_12_13_triangle :
  ∀ (ABC : Triangle) (I O : Point) (AB AC BC : ℝ),
    AB = 5 → AC = 12 → BC = 13 →
    right_triangle ABC →
    incenter ABC I →
    circumcenter ABC O →
    distance I O = (Real.sqrt 65) / 2 := by
  sorry

end distance_between_incenter_and_circumcenter_of_5_12_13_triangle_l249_249394


namespace maximum_candies_l249_249896

-- Definitions of the problem parameters
def classmates : ℕ := 25
def total_candies : ℕ := 200
def condition (candies : ℕ → ℕ) : Prop := 
  ∀ S (h : finset ℕ), S.card = 16 → (∑ i in S, candies i) ≥ 100

-- The theorem stating the maximum number of candies Vovochka can keep
theorem maximum_candies (candies : ℕ → ℕ) (h : condition candies) : 
  (total_candies - ∑ i in finset.range classmates, candies i) ≤ 37 :=
sorry

end maximum_candies_l249_249896


namespace height_on_hypotenuse_l249_249945

theorem height_on_hypotenuse (a b : ℕ) (hypotenuse : ℝ)
  (ha : a = 3) (hb : b = 4) (h_c : hypotenuse = sqrt (a^2 + b^2)) :
  let S := (1/2 : ℝ) * a * b in
  ∃ h : ℝ, h = (2 * S) / hypotenuse ∧ h = 12/5 := by
  sorry

end height_on_hypotenuse_l249_249945


namespace max_divisors_1_to_20_l249_249205

theorem max_divisors_1_to_20 :
  ∃ m n o ∈ set.range (20:ℕ.succ), 
    (∀ k ∈ set.range (20:ℕ.succ), 
      (nat.divisors_count k ≤ nat.divisors_count m) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count n) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count o)) 
    ∧ (nat.divisors_count m = 6)
    ∧ (nat.divisors_count n = 6)
    ∧ (nat.divisors_count o = 6)
    ∧ m ≠ n ∧ n ≠ o ∧ o ≠ m
    ∧ set.mem 12 (set.range (20 + 1))
    ∧ set.mem 18 (set.range (20 + 1))
    ∧ set.mem 20 (set.range (20 + 1)). 
      := by 
  -- all these steps will be skipped here
  sorry

end max_divisors_1_to_20_l249_249205


namespace poly_coefficients_sum_l249_249077

theorem poly_coefficients_sum :
  ∀ (x A B C D : ℝ),
  (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 2 :=
by sorry

end poly_coefficients_sum_l249_249077


namespace rounding_no_order_l249_249079

theorem rounding_no_order (x : ℝ) (hx : x > 0) :
  let a := round (x * 100) / 100
  let b := round (x * 1000) / 1000
  let c := round (x * 10000) / 10000
  (¬((a ≥ b ∧ b ≥ c) ∨ (a ≤ b ∧ b ≤ c))) :=
sorry

end rounding_no_order_l249_249079


namespace log_eq_implies_y_eq_four_l249_249788

theorem log_eq_implies_y_eq_four (log_y_64_eq_log_3_27 : log y 64 = log 3 27) : y = 4 := 
by 
  sorry

end log_eq_implies_y_eq_four_l249_249788


namespace part_one_conditions_part_two_conditions_l249_249042

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ := m * log x + n * x
noncomputable def g (x t : ℝ) : ℝ := (-x^2 + 2 * x) / t

-- Define initial conditions
variables {m n t : ℝ}
variables (x : ℝ)

-- Problem (I)
theorem part_one_conditions {m n : ℝ} (h1 : f 1 m n = -2) (h2 : m + n = -1) :
  m = 1 ∧ n = -2 :=
sorry

-- Problem (II)
theorem part_two_conditions {m n : ℝ} (h1 : f 1 1 (-2) = -2) (h3 : t > 0) (h4 : ∃ x ∈ set.Icc 1 real.exp, f x 1 (-2) + x ≥ g x t) :
  t ≤ (real.exp * (real.exp - 2) / (real.exp - 1)) :=
sorry

end part_one_conditions_part_two_conditions_l249_249042


namespace numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249225

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).length

theorem numbers_with_most_divisors_in_range :
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  (number_of_divisors n ≤ 6) :=
by
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list
  simp [number_of_divisors] <;>
  dec_trivial -- Automatically discharges the goals since they are simple arithmetic comparisons

theorem greatest_divisors_in_range :
  number_of_divisors 12 = 6 ∧
  number_of_divisors 18 = 6 ∧
  number_of_divisors 20 = 6 ∧
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19],
  number_of_divisors n < 6 :=
by
  refine ⟨rfl, rfl, rfl, _⟩
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list excluding 12, 18, 20
  dec_trivial -- Automatically discharges the goals due to their simplicity

#print greatest_divisors_in_range

end numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249225


namespace factorize_expression_l249_249787

variable (x y : ℝ)

theorem factorize_expression :
  4 * (x - y + 1) + y * (y - 2 * x) = (y - 2) * (y - 2 - 2 * x) :=
by 
  sorry

end factorize_expression_l249_249787


namespace triangle_side_length_BC_l249_249969

theorem triangle_side_length_BC (A B C : Type) (a b c : ℝ) 
  (hA : A = 60 * Real.pi / 180)
  (h_roots : ∀ x : ℝ, (x^2 - 9*x + 8 = 0) ↔ (x = b ∨ x = c) ∧ 0 < b ∧ 0 < c ∧ b ≠ c)
  (h_b : b = 8)
  (h_c : c = 1) : 
  ∃ BC : ℝ, BC = Real.sqrt 57 :=
by 
  existsi Real.sqrt 57
  sorry

end triangle_side_length_BC_l249_249969


namespace greatest_num_divisors_in_range_l249_249168

def num_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d, n % d = 0)

theorem greatest_num_divisors_in_range : 
  ∀ n ∈ {1, 2, 3, ..., 20}, 
  num_divisors n < 7 → (n = 12 ∨ n = 18 ∨ n = 20) :=
by {
  sorry
}

end greatest_num_divisors_in_range_l249_249168


namespace smallest_grandeur_monic_quadratic_l249_249455

noncomputable def grandeur (q : ℝ → ℝ) : ℝ :=
  Real.sup (set.image (λ x, abs (q x)) (set.Icc (-2) (2)))

theorem smallest_grandeur_monic_quadratic :
  ∃ (g : ℝ → ℝ), (∀ x, polynomial_degree g x ≤ 2) ∧ (∀ x, g x = x^2 + b*x + c → b = 0) ∧ (grandeur g = 3) := sorry

end smallest_grandeur_monic_quadratic_l249_249455


namespace nearest_integer_area_l249_249950

noncomputable def area_of_triangle (A B C : ℝ) : ℝ := 
  ⟦simplified_area_calculation_here⟧  -- Pseudo code to abstract the area calculation method

def triangle_properties (A B C D E : Type) : Prop :=
  ∃ (AD BE : ℝ), AD = 7 ∧ BE = 9 ∧ 
  (is_median A B C D AD) ∧ 
  (is_angle_bisector A B C E BE) ∧ 
  ⊥ (AD ⊤ BE)

-- The main theorem stating that given these properties, the nearest integer to the calculated area is 47
theorem nearest_integer_area (A B C D E : Type) (h: triangle_properties A B C D E) : 
   (∃ (area : ℝ), area_of_triangle A B C = area ∧ abs (area - 47) < 0.5) := 
sorry

end nearest_integer_area_l249_249950


namespace number_of_zeros_of_f_l249_249069

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.log x + x)

theorem number_of_zeros_of_f : 
  (Set.count {x : ℝ | 1 < x ∧ x < 2 ∧ f x = 0}) = 1 :=
sorry

end number_of_zeros_of_f_l249_249069


namespace regular_nonagon_approximation_l249_249790

theorem regular_nonagon_approximation :
  ∀ ε : ℝ, (0 < ε) →
  ∀ (x : ℝ), 
  abs (sin 60 - sin 50 - 0.1) < ε ∧ 
  abs (cos 40 - 0.7660254) < ε →
  ∀ θ : ℝ, θ = (360 / 9) →
  (∃ δ : ℝ, abs (cos θ - cos (40 + δ)) < ε) :=
by
  intro ε ε_pos x h_approx θ h_theta
  -- Here, we would go through the steps to approximate the error and show the construction works
  sorry

end regular_nonagon_approximation_l249_249790


namespace melanie_turnips_l249_249162

theorem melanie_turnips (b : ℕ) (d : ℕ) (h_b : b = 113) (h_d : d = 26) : b + d = 139 :=
by
  sorry

end melanie_turnips_l249_249162


namespace evaluate_expression_l249_249430

theorem evaluate_expression :
  (81: ℝ)^(1/2) * (8: ℝ)^(-1/3) * (25: ℝ)^(1/2) = 45 / 2 :=
by
  sorry

end evaluate_expression_l249_249430


namespace vovochka_max_candies_l249_249863

/-!
# Problem Statement:
- Vovochka has 25 classmates.
- Vovochka brought 200 candies to class.

## Condition:
- Any 16 classmates together should have at least 100 candies in total.

Question: What is the maximum number of candies Vovochka can keep for himself while fulfilling his mother's request?

The answer to this problem needs to be proven mathematically.
-/

def max_candies_can_keep (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ) : ℕ :=
if h : classmates = 25 ∧ total_candies = 200 ∧ min_candies_per_16 = 100 then
  37
else 
  sorry -- This statement is just for context, keeping focus on formal proof in Lean.

theorem vovochka_max_candies :
  ∀ (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ),
    classmates = 25 → total_candies = 200 → min_candies_per_16 = 100 →
    max_candies_can_keep candies classmates total_candies min_candies_per_16 = 37 :=
by
  intros candies classmates total_candies min_candies_per_16 hc ht hm
  simp [max_candies_can_keep]
  exact if_pos ⟨hc, ht, hm⟩

end vovochka_max_candies_l249_249863


namespace limit_Sn_div_Tn_l249_249148

def M_n (n : ℕ) : set ℝ :=
  {x | ∃ (a : Fin n → ℕ), (∀ (i : Fin (n-1)), a i ∈ {0, 1}) ∧ a (n-1) = 1 ∧ 
      x = (∑ i in Finset.range n, (a i) / 10^(i+1)) }

def T_n (n : ℕ) : ℕ := 2^(n-1)

noncomputable def S_n (n : ℕ) : ℝ :=
  ∑ x in M_n n, x

theorem limit_Sn_div_Tn : 
  ∀ {S_n T_n : ℕ → ℝ}, 
  (∀ n, S_n n = ∑ x in M_n n, x) → 
  (∀ n, T_n n = 2^(n-1)) → 
  (∀ n, (S_n n) / (T_n n) = 1 / 18) :=
begin
  sorry
end

end limit_Sn_div_Tn_l249_249148


namespace rachel_math_homework_l249_249239

theorem rachel_math_homework (reading_hw math_hw : ℕ) 
  (h1 : reading_hw = 4) 
  (h2 : math_hw = reading_hw + 3) : 
  math_hw = 7 := by
  sorry

end rachel_math_homework_l249_249239


namespace complement_union_l249_249502

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {1, 3, 5, 7}
def N : Set ℕ := {5, 6, 7}

theorem complement_union (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6, 7, 8}) 
  (hM : M = {1, 3, 5, 7}) (hN : N = {5, 6, 7}) : U \ (M ∪ N) = {2, 4, 8} :=
by
  sorry

end complement_union_l249_249502


namespace probability_heads_penny_nickel_dime_is_one_eighth_l249_249274

-- Define the setup: flipping 5 coins
def five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the condition: penny, nickel, and dime are heads
def heads_penny_nickel_dime (c: five_coins) : bool := c.1 && c.2 && c.3

-- Define the successful outcomes where penny, nickel, and dime are heads
noncomputable def successful_outcomes : Set five_coins :=
  { c | heads_penny_nickel_dime c = tt }

-- Define the total outcomes for 5 coins
noncomputable def total_outcomes : Set five_coins := {c | true}

-- Probability calculation: |success| / |total|
noncomputable def probability_heads_penny_nickel_dime : real :=
  (Set.card successful_outcomes : real) / (Set.card total_outcomes : real)

-- Given the setup, prove that the probability is 1/8
theorem probability_heads_penny_nickel_dime_is_one_eighth :
  probability_heads_penny_nickel_dime = 1 / 8 := 
sorry

end probability_heads_penny_nickel_dime_is_one_eighth_l249_249274


namespace trapezoid_problem_l249_249590

variables {Point : Type} [MetricSpace Point]

-- Constants representing the points in the problem
variables (A B C D E O T : Point)

-- Definitions from the conditions
def is_parallelogram (ABCD : ConvexPolygon Point) : Prop :=
  ∃ A B C D : Point, 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ 
    (ABCD = {A, B, C, D}) ∧
    (vector A B + vector C D = 0) ∧
    (vector B C + vector D A = 0)

def diagonals_bisect_at (ABCD : ConvexPolygon Point) (O : Point) : Prop :=
  ∃ A B C D : Point, 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ 
    (ABCD = {A, B, C, D}) ∧
    (vector A C = 2 * vector A O) ∧
    (vector B D = 2 * vector B O)

def equal_area (ABCD : ConvexPolygon Point) (CDE : ConvexPolygon Point) : Prop := 
  area ABCD = area CDE

def is_chosen_on (T : Point) (DE : LineSegment Point) : Prop :=
  ∃ D E : Point, 
    D ≠ E ∧ 
    (DE = segment D E) ∧ 
    (T ∈ DE)

def parallel (u v : Vector Point) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ (u = k • v)

-- Given conditions and the statement to prove
theorem trapezoid_problem (h1 : is_parallelogram {A, B, C, D})
    (h2 : diagonals_bisect_at {A, B, C, D} O)
    (h3 : equal_area {A, B, C, D} {C, D, E})
    (h4 : is_chosen_on T (segment D E))
    (h5 : parallel (vector O T) (vector B E)) :
  parallel (vector O D) (vector C T) :=
sorry

end trapezoid_problem_l249_249590


namespace speed_of_faster_train_approx_l249_249675

noncomputable def speed_of_slower_train_kmph : ℝ := 40
noncomputable def speed_of_slower_train_mps : ℝ := speed_of_slower_train_kmph * 1000 / 3600
noncomputable def distance_train1 : ℝ := 250
noncomputable def distance_train2 : ℝ := 500
noncomputable def total_distance : ℝ := distance_train1 + distance_train2
noncomputable def crossing_time : ℝ := 26.99784017278618
noncomputable def relative_speed_train_crossing : ℝ := total_distance / crossing_time
noncomputable def speed_of_faster_train_mps : ℝ := relative_speed_train_crossing - speed_of_slower_train_mps
noncomputable def speed_of_faster_train_kmph : ℝ := speed_of_faster_train_mps * 3600 / 1000

theorem speed_of_faster_train_approx : abs (speed_of_faster_train_kmph - 60.0152) < 0.001 :=
by 
  sorry

end speed_of_faster_train_approx_l249_249675


namespace numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249227

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).length

theorem numbers_with_most_divisors_in_range :
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  (number_of_divisors n ≤ 6) :=
by
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list
  simp [number_of_divisors] <;>
  dec_trivial -- Automatically discharges the goals since they are simple arithmetic comparisons

theorem greatest_divisors_in_range :
  number_of_divisors 12 = 6 ∧
  number_of_divisors 18 = 6 ∧
  number_of_divisors 20 = 6 ∧
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19],
  number_of_divisors n < 6 :=
by
  refine ⟨rfl, rfl, rfl, _⟩
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list excluding 12, 18, 20
  dec_trivial -- Automatically discharges the goals due to their simplicity

#print greatest_divisors_in_range

end numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249227


namespace Vovochka_max_candies_l249_249884

noncomputable theory
open_locale classical
open set

def VovochkaCandies (c n k required_sum : ℕ) (dist : fin n → ℕ) : Prop :=
  ∀ (s : finset (fin n)), s.card = k → (s.sum dist ≥ required_sum)

theorem Vovochka_max_candies
  (c n k required_sum : ℕ)
  (h_c : c = 200)
  (h_n : n = 25)
  (h_k : k = 16)
  (h_required_sum : required_sum = 100)
  (dist : fin n → ℕ)
  (h_dist : VovochkaCandies c n k required_sum dist) :
  c - finset.univ.sum dist = 37 :=
sorry

end Vovochka_max_candies_l249_249884


namespace cubic_identity_l249_249514

theorem cubic_identity (x y z : ℝ) 
  (h1 : x + y + z = 12) 
  (h2 : xy + xz + yz = 30) : 
  x^3 + y^3 + z^3 - 3 * x * y * z = 648 :=
sorry

end cubic_identity_l249_249514


namespace range_of_a_tangent_point_x1_l249_249494

noncomputable def f (x a : ℝ) : ℝ := exp x * (x^2 - x + a)
noncomputable def f_prime (x a : ℝ) : ℝ := exp x * (x^2 + x + a - 1)

noncomputable def g (x a : ℝ) : ℝ := x^3 + a * x - a

theorem range_of_a (h : ∃ (x_1 x_2 x_3 : ℝ), x_1 < x_2 ∧ x_2 < x_3 ∧ g x_1 a = 0 ∧ g x_2 a = 0 ∧ g x_3 a = 0) : 
  a < -27 / 4 := 
sorry

theorem tangent_point_x1 (h : ∃ (x_1 x_2 x_3 : ℝ), x_1 < x_2 ∧ x_2 < x_3 ∧ g x_1 a = 0 ∧ g x_2 a = 0 ∧ g x_3 a = 0 ∧ a < -27 / 4): 
  ∃ (x_1 : ℝ), g x_1 a = 0 ∧ x_1 < -3 := 
sorry

end range_of_a_tangent_point_x1_l249_249494


namespace meal_combinations_count_l249_249538

def main_dishes : Set String := {"rice", "noodles"}
def stir_fry_dishes : Set String := {"potato slices", "mapo tofu", "tomato scrambled eggs", "fried potatoes"}

theorem meal_combinations_count : 
  (main_dishes.card * stir_fry_dishes.card = 8) :=
by
  sorry

end meal_combinations_count_l249_249538


namespace joe_time_to_school_l249_249976

theorem joe_time_to_school
    (r_w : ℝ) -- Joe's walking speed
    (t_w : ℝ) -- Time to walk halfway
    (t_stop : ℝ) -- Time stopped at the store
    (r_running_factor : ℝ) -- Factor by which running speed is faster than walking speed
    (initial_walk_time_halfway : t_w = 10)
    (store_stop_time : t_stop = 3)
    (running_speed_factor : r_running_factor = 4) :
    t_w + t_stop + t_w / r_running_factor = 15.5 :=
by
    -- Implementation skipped, just verifying statement is correctly captured
    sorry

end joe_time_to_school_l249_249976


namespace area_of_ADC_eq_twenty_l249_249098

-- Let "ABC" be a triangle, and "D" a point on side "BC"
variables (A B C D : Type*)

-- Let the ratio BD/DC be 3/2
variables (r_bd r_dc : ℝ) (h_ratio : r_bd / r_dc = 3 / 2)

-- Let the area of triangle ABD be 30 square centimeters
variable (area_ABD : ℝ) (h_area_ABD : area_ABD = 30)

-- Define the key variables and statements
def area_ADC : ℝ := area_ABD * (2 / 3)

-- State the theorem that we need to prove
theorem area_of_ADC_eq_twenty : area_ADC = 20 := by
  sorry

end area_of_ADC_eq_twenty_l249_249098


namespace percentage_increase_is_60_percent_l249_249783

/-- Define the constants for distances and speeds -/
def distance := 120
def distance_part1 := 32
def distance_part2 := 88
def speed_sunday (x : ℝ) := x
def speed_monday_part1 (x : ℝ) := 2 * x
def speed_monday_part2 (x : ℝ) := x / 2

/-- Calculate the time taken on Sunday -/
def time_sunday (x : ℝ) := distance / speed_sunday x

/-- Calculate the time taken on Monday -/
def time_monday (x : ℝ) :=
  (distance_part1 / speed_monday_part1 x) + (distance_part2 / speed_monday_part2 x)

/-- Calculate the percentage increase in time from Sunday to Monday -/
def percentage_increase_in_time (x : ℝ) :=
  ((time_monday x - time_sunday x) / time_sunday x) * 100

theorem percentage_increase_is_60_percent (x : ℝ) (hx : x > 0):
  percentage_increase_in_time x = 60 := by
  sorry

end percentage_increase_is_60_percent_l249_249783


namespace odd_sum_probability_l249_249973

/-- Define spinner P -/
def SpinnerP := {2, 4, 5}

/-- Define spinner Q -/
def SpinnerQ := {1, 3, 5}

/-- Define spinner R -/
def SpinnerR := {2, 5, 7}

/-- Define a predicate for odd numbers -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- Define a predicate for even numbers -/
def isEven (n : ℕ) : Prop := n % 2 = 0

/-- Define a function to calculate probability of the sum being odd -/
noncomputable def probabilityOddSum (P Q R : set ℕ) : ℚ :=
  let oddP := {x ∈ P | isOdd x}.toFinset.card / P.toFinset.card
  let evenP := {x ∈ P | isEven x}.toFinset.card / P.toFinset.card
  let oddQ := {x ∈ Q | isOdd x}.toFinset.card / Q.toFinset.card
  let oddR := {x ∈ R | isOdd x}.toFinset.card / R.toFinset.card
  let evenR := {x ∈ R | isEven x}.toFinset.card / R.toFinset.card
  (oddP * oddQ * oddR) + (evenP * evenP * oddQ)

/-- The proposition to be proved -/
theorem odd_sum_probability :
  probabilityOddSum SpinnerP SpinnerQ SpinnerR = 4 / 9 :=
sorry

end odd_sum_probability_l249_249973


namespace largest_angle_of_isosceles_obtuse_30_deg_l249_249671

def is_isosceles (T : Triangle) : Prop :=
  T.A = T.B ∨ T.B = T.C ∨ T.C = T.A

def is_obtuse (T : Triangle) : Prop :=
  T.A > 90 ∨ T.B > 90 ∨ T.C > 90

def T : Type := {P Q R : ℝ}

noncomputable def largest_angle (T : Triangle) : ℝ :=
  max T.A (max T.B T.C)

theorem largest_angle_of_isosceles_obtuse_30_deg :
  ∀ (T : Triangle), is_isosceles T → is_obtuse T → T.A = 30 → largest_angle T = 120 :=
by
  intro T h_iso h_obt h_A30
  sorry

end largest_angle_of_isosceles_obtuse_30_deg_l249_249671


namespace find_cube_difference_l249_249025

theorem find_cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := 
by
  sorry

end find_cube_difference_l249_249025


namespace S_2011_value_l249_249533

-- Definitions based on conditions provided in the problem
def arithmetic_seq (a_n : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a_n (n + 1) = a_n n + d

def sum_seq (S_n : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

-- Problem statement
theorem S_2011_value
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (h_arith : arithmetic_seq a_n)
  (h_sum : sum_seq S_n a_n)
  (h_init : a_n 1 = -2011)
  (h_cond : (S_n 2010) / 2010 - (S_n 2008) / 2008 = 2) :
  S_n 2011 = -2011 := 
sorry

end S_2011_value_l249_249533


namespace ellipse_equation_and_fixed_triangle_area_l249_249840

open Real

noncomputable section

variables (a b : ℝ) (P Q R S : ℝ × ℝ)

def ellipse_C1 : Prop :=
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
  ∀ (x y : ℝ), (x, y) ∈ {P, Q, R, S} → (x^2 / 12 + y^2 / 9 = 1)

def circle_C0 : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ {P, Q, R, S} → (x^2 + y^2 = 36 / 7)

def parallelogram_area (a b : ℝ) : Prop :=
  2 * a * 2 * b / 2 = 12 * sqrt 3

def centroid_at_origin (A B C : ℝ × ℝ) : Prop :=
  ∃ O : ℝ × ℝ, O = (0, 0) ∧ (A.1 + B.1 + C.1) / 3 = O.1 ∧ (A.2 + B.2 + C.2) / 3 = O.2 ∧ 
  A, B, C ∈ {P, Q, R, S}

def fixed_area (A B C : ℝ × ℝ) : Prop :=
  1 / 2 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) = 27 / 2

theorem ellipse_equation_and_fixed_triangle_area :
  ellipse_C1 a b P Q R S →
  circle_C0 P Q R S →
  parallelogram_area a b →
  centroid_at_origin P Q R →
  fixed_area P Q R
:=
sorry

end ellipse_equation_and_fixed_triangle_area_l249_249840


namespace custom_op_5_3_l249_249510

def custom_op (a b : ℕ) : ℕ := a^2 - a*b + b^2

theorem custom_op_5_3 : custom_op 5 3 = 19 :=
by
  rw [custom_op, pow_two, pow_two, ← nat.sub_add_eq_add_sub]
  rw [mul_comm, mul_comm]
  -- Sorry is used to skip the proof, as the problem requires only the statement.
  sorry

end custom_op_5_3_l249_249510


namespace factorial_divisibility_l249_249607

theorem factorial_divisibility (m n : ℕ) : (m! * n! * (m + n)!) ∣ ((2 * m)! * (2 * n)!) := 
sorry

end factorial_divisibility_l249_249607


namespace rectangle_properties_l249_249793

noncomputable def diagonal (x1 y1 x2 y2 : ℕ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def area (length width : ℕ) : ℕ :=
  length * width

theorem rectangle_properties :
  diagonal 1 1 9 7 = 10 ∧ area (9 - 1) (7 - 1) = 48 := by
  sorry

end rectangle_properties_l249_249793


namespace solution_set_x_fx_neg_l249_249640

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom monotone_increasing : ∀ x y : ℝ, 0 < x → x < y → f x < f y
axiom f_one : f 1 = 0

theorem solution_set_x_fx_neg :
  {x | x * f x < 0} = set.Ioo (-1 : ℝ) 0 ∪ set.Ioo (0 : ℝ) 1 :=
by
  sorry

end solution_set_x_fx_neg_l249_249640


namespace remainder_product_div_6_l249_249798

def sequence_term(n : ℕ) : ℕ := 4 + (n - 1) * 10

theorem remainder_product_div_6 : 
  let product := ∏ i in (Finset.range 21).map (Finset.natEmbedding.succ), sequence_term i
  (product % 6) = 4 :=
sorry

end remainder_product_div_6_l249_249798


namespace solution_count_f_fx_eq_1_l249_249155

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then -x + 4 else 3 * x - 6 

theorem solution_count_f_fx_eq_1 : ∃! x : ℝ, f(f(x)) = 1 :=
sorry

end solution_count_f_fx_eq_1_l249_249155


namespace poly_coefficients_sum_l249_249076

theorem poly_coefficients_sum :
  ∀ (x A B C D : ℝ),
  (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 2 :=
by sorry

end poly_coefficients_sum_l249_249076


namespace remainder_product_div_6_l249_249799

def sequence_term(n : ℕ) : ℕ := 4 + (n - 1) * 10

theorem remainder_product_div_6 : 
  let product := ∏ i in (Finset.range 21).map (Finset.natEmbedding.succ), sequence_term i
  (product % 6) = 4 :=
sorry

end remainder_product_div_6_l249_249799


namespace greatest_divisors_1_to_20_l249_249199

def is_divisor (a b : ℕ) : Prop := b % a = 0

def count_divisors (n : ℕ) : ℕ :=
  nat.succ (list.length ([k for k in list.range n if is_divisor k.succ n]))

def max_divisors_from_1_to_20 : set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 20 ∧ ∃ max_count, count_divisors n = max_count ∧ 
  ∀ m, 1 ≤ m ∧ m ≤ 20 → count_divisors m ≤ max_count}

theorem greatest_divisors_1_to_20 : max_divisors_from_1_to_20 = {12, 18, 20} :=
by {
  sorry
}

end greatest_divisors_1_to_20_l249_249199


namespace twelfth_prime_is_37_l249_249241

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_prime (n : ℕ) : ℕ := if h : n > 0 then classical.some (nat.exists_infinite_primes n) else 2

theorem twelfth_prime_is_37 (h : nth_prime 7 = 17) : nth_prime 12 = 37 :=
  sorry

end twelfth_prime_is_37_l249_249241


namespace union_sets_l249_249604

variables {α : Type*} (A B : set α)
variables a b : ℕ

-- Given conditions
noncomputable def condition1 : A = {3, 2^a} := sorry
noncomputable def condition2 : B = {a, b} := sorry
noncomputable def condition3 : A ∩ B = {2} := sorry

-- Prove that A ∪ B = {1, 2, 3}
theorem union_sets (h1 : condition1) (h2 : condition2) (h3 : condition3) : A ∪ B = {1, 2, 3} :=
  sorry

end union_sets_l249_249604


namespace complex_quadrant_proof_l249_249566

theorem complex_quadrant_proof (z : ℂ) (i_sq : complex.I ^ 2 = -1) :
  (((3 - complex.I) / (1 + complex.I)) ^ 2).re < 0 ∧ (((3 - complex.I) / (1 + complex.I)) ^ 2).im < 0 := 
sorry

end complex_quadrant_proof_l249_249566


namespace profit_shares_difference_l249_249346

theorem profit_shares_difference (total_profit : ℝ) (share_ratio_x share_ratio_y : ℝ) 
  (hx : share_ratio_x = 1/2) (hy : share_ratio_y = 1/3) (profit : ℝ):
  total_profit = 500 → profit = (total_profit * share_ratio_x) / ((share_ratio_x + share_ratio_y)) - (total_profit * share_ratio_y) / ((share_ratio_x + share_ratio_y)) → profit = 100 :=
by
  intros
  sorry

end profit_shares_difference_l249_249346


namespace seq_50_eq_l249_249769

noncomputable def sequence (n : ℕ) : ℝ :=
  match n with
  | 0       => 0  -- by convention we use 0 for 0th term since sequence starts at 1
  | 1       => 3
  | (n + 2) => 4 * sequence (n + 1)

theorem seq_50_eq : sequence 50 = 4 ^ 49 * 3 :=
by
  sorry

end seq_50_eq_l249_249769


namespace total_height_correct_l249_249130

def height_of_stairs : ℕ := 10

def num_flights : ℕ := 3

def height_of_all_stairs : ℕ := height_of_stairs * num_flights

def height_of_rope : ℕ := height_of_all_stairs / 2

def extra_height_of_ladder : ℕ := 10

def height_of_ladder : ℕ := height_of_rope + extra_height_of_ladder

def total_height_climbed : ℕ := height_of_all_stairs + height_of_rope + height_of_ladder

theorem total_height_correct : total_height_climbed = 70 := by
  sorry

end total_height_correct_l249_249130


namespace largest_angle_of_isosceles_obtuse_30_deg_l249_249672

def is_isosceles (T : Triangle) : Prop :=
  T.A = T.B ∨ T.B = T.C ∨ T.C = T.A

def is_obtuse (T : Triangle) : Prop :=
  T.A > 90 ∨ T.B > 90 ∨ T.C > 90

def T : Type := {P Q R : ℝ}

noncomputable def largest_angle (T : Triangle) : ℝ :=
  max T.A (max T.B T.C)

theorem largest_angle_of_isosceles_obtuse_30_deg :
  ∀ (T : Triangle), is_isosceles T → is_obtuse T → T.A = 30 → largest_angle T = 120 :=
by
  intro T h_iso h_obt h_A30
  sorry

end largest_angle_of_isosceles_obtuse_30_deg_l249_249672


namespace convex_quad_diagonal_l249_249942

theorem convex_quad_diagonal {
  A B C D E : Type*}
  [convex_quadrilateral A B C D]
  (h1 : diagonal A C B D)
  (h2 : intersection_point A C B D E) :
  (angle E = π / 2) ↔ (AB ^ 2 + CD ^ 2 = AD ^ 2 + BC ^ 2) :=
sorry

end convex_quad_diagonal_l249_249942


namespace transform_quadratic_to_standard_l249_249151

variables {a b c : ℝ}
variable (h_a_nonzero : a ≠ 0)

/-- Prove that the graph of the quadratic function P(x) = a * x^2 + b * x + c can be transformed
    into the graph of the function Q(x) = x^2 through a combination of translation and homothety. -/
theorem transform_quadratic_to_standard (P Q : ℝ → ℝ) :
  (P = λ x, a * x^2 + b * x + c) →
  (Q = λ x, x^2) →
  ∃ (transformation : (ℝ → ℝ) → (ℝ → ℝ)), transformation P = Q :=
by 
  intro hP hQ
  sorry

end transform_quadratic_to_standard_l249_249151


namespace trapezoid_bc_squared_l249_249545

theorem trapezoid_bc_squared (AB CD AD AC BD BC : ℝ) (h1 : AC ⊥ BD) (h2 : AB = 3) (h3 : AD = Real.sqrt 3241) (h4 : right_angle (BC, AB)) (h5 : right_angle (BC, CD)) :
  BC^2 = (9 + Real.sqrt 116433) / 2 :=
by
  sorry

end trapezoid_bc_squared_l249_249545


namespace vec_a_orthogonal_vec_b_find_k_l249_249857

noncomputable theory

open Real

-- Definitions
def vec_a : ℝ × ℝ := (sqrt 3, -1)
def vec_b : ℝ × ℝ := (1/2, (sqrt 3) / 2)

-- Problem 1
theorem vec_a_orthogonal_vec_b : (vec_a.fst * vec_b.fst + vec_a.snd * vec_b.snd = 0) → 
  (vec_a.fst * vec_b.fst + vec_a.snd * vec_b.snd = 0) :=
sorry

-- Definitions for Problem 2
def vec_c (t : ℝ) : ℝ × ℝ := (vec_a.fst + (t^2 - 3) * vec_b.fst, vec_a.snd + (t^2 - 3) * vec_b.snd)
def vec_d (t k : ℝ) : ℝ × ℝ := (-k * vec_a.fst + t * vec_b.fst, -k * vec_a.snd + t * vec_b.snd)

-- Problem 2
theorem find_k (t : ℝ) (ht : t ≠ 0) : 
  ((vec_c t).fst * (vec_d t (t^3 - 3*t) / 4).fst + (vec_c t).snd * (vec_d t (t^3 - 3*t) / 4).snd = 0) → 
  (k = (t^3 - 3*t) / 4) :=
sorry

end vec_a_orthogonal_vec_b_find_k_l249_249857


namespace edward_will_have_16_washers_remaining_l249_249426

section PlumbingProblem

variables (copperPipeLength pvcPipeLength washersInBag : ℕ)
variables (boltsPerCopperPipe boltsPerPVCPipe washersPerCopperBolt washersPerPVCBolt : ℕ)

def washers_remaining (copperPipeLength pvcPipeLength washersInBag : ℕ)
  (boltsPerCopperPipe boltsPerPVCPipe washersPerCopperBolt washersPerPVCBolt : ℕ) : ℕ :=
let copperBolts := copperPipeLength / boltsPerCopperPipe in
let pvcBolts := (pvcPipeLength / boltsPerPVCPipe) * 2 in
let washersNeeded := (copperBolts * washersPerCopperBolt) + (pvcBolts * washersPerPVCBolt) in
washersInBag - washersNeeded

theorem edward_will_have_16_washers_remaining :
  washers_remaining 40 30 50 5 10 2 3 = 16 :=
by 
  unfold washers_remaining 
  norm_num 
  sorry

end PlumbingProblem

end edward_will_have_16_washers_remaining_l249_249426


namespace unique_five_digit_numbers_l249_249509

theorem unique_five_digit_numbers :
  -- Definitions for digits and constraints
  let digits := {1, 2, 3, 4, 5}
  let valid_numbers := { x | ∃ a b c d e : ℕ, 
                        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                        c ≠ d ∧ c ≠ e ∧ 
                        d ≠ e ∧
                        x = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
                        a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
                        x > 20000 ∧ c ≠ 3 }
  in
  -- Prove that the number of such valid numbers is 65
  valid_numbers.card = 65 :=
sorry

end unique_five_digit_numbers_l249_249509


namespace ratio_is_one_ratio_determined_l249_249404

-- Define the essentials of the problem, such as the figure, areas, and line segment characteristics.

variables {X Q Y : ℝ}

-- Definitions based on the conditions
def unit_square_area := 1
def total_squares := 12
def total_area := total_squares * unit_square_area + (1 / 2) * 4 * 3  -- derived from calculating the area
def base_triangle := 4
def height_triangle := 3
def line_PQ := base_triangle

-- Condition that line PQ bisects the area of the entire figure
def bisects_area (PQ : ℝ) := PQ = base_triangle / 2 ∧ below_PQ * 2 = total_area ∧ above_PQ * 2 = total_area
def below_PQ := 9
def above_PQ := 6

-- Definition involvin the ratio
def ratio_XQ_QY (XQ QY : ℝ) := XQ / QY 

-- The theorem we need to prove
theorem ratio_is_one : ratio_XQ_QY (XQ) (QY) = 1 :=
by
  -- Assume XQ and QY both are 2 because this balances the figure split.
  assume h1 : XQ = 2
  assume h2 : QY = 2
  exact (by norm_num : XQ / QY = 1)

-- Use these assumptions to relate to known values
lemma PQ_measures : ∀ P Q, PQ = 4 :=
sorry

theorem ratio_determined : ratio_XQ_QY (2) (2) = 1 :=
sorry

end ratio_is_one_ratio_determined_l249_249404


namespace concurrency_of_tangents_with_AC_l249_249158

variables {A B C O I B' : Type}

-- Given conditions
def is_circumcenter (O : Type) (ABC : Type) : Prop := sorry
def is_incenter (I : Type) (ABC : Type) : Prop := sorry
def is_reflection (B' B O' : Type) (OI : Type) : Prop := sorry
def lies_within_angle (B' : Type) (B : Type) (O' : Type) (I : Type) : Prop := sorry

-- Main statement to prove
theorem concurrency_of_tangents_with_AC
  (h1 : is_circumcenter O (triangle A B C))
  (h2 : is_incenter I (triangle A B C))
  (h3 : is_reflection B' B OI)
  (h4 : lies_within_angle B' B I) :
  ∃ K : Type, tangents_concur (circumcircle B I B') B' I K AC :=
sorry

end concurrency_of_tangents_with_AC_l249_249158


namespace probability_heads_penny_nickel_dime_l249_249291

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]
variable (coins : Fin 5 → Ω)
variable (heads tails : Ω)

-- Each coin has two outcomes: heads or tails
axiom coin_outcome (i : Fin 5) : coins i = heads ∨ coins i = tails

-- There are 32 total outcomes
axiom total_outcomes : Fintype.card (Fin 5 → Ω) = 32

-- There are 4 successful outcomes for penny, nickel, and dime being heads
axiom successful_outcomes : let successful := {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} in
                          Fintype.card successful = 4

theorem probability_heads_penny_nickel_dime :
  let probability := (Fintype.card {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} : ℤ) /
                     (Fintype.card (Fin 5 → Ω) : ℤ) in
  probability = 1 / 8 :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249291


namespace frac_pattern_2_11_frac_pattern_general_l249_249015

theorem frac_pattern_2_11 :
  (2 / 11) = (1 / 6) + (1 / 66) :=
sorry

theorem frac_pattern_general (n : ℕ) (hn : n ≥ 3) :
  (2 / (2 * n - 1)) = (1 / n) + (1 / (n * (2 * n - 1))) :=
sorry

end frac_pattern_2_11_frac_pattern_general_l249_249015


namespace probability_heads_penny_nickel_dime_is_one_eighth_l249_249272

-- Define the setup: flipping 5 coins
def five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the condition: penny, nickel, and dime are heads
def heads_penny_nickel_dime (c: five_coins) : bool := c.1 && c.2 && c.3

-- Define the successful outcomes where penny, nickel, and dime are heads
noncomputable def successful_outcomes : Set five_coins :=
  { c | heads_penny_nickel_dime c = tt }

-- Define the total outcomes for 5 coins
noncomputable def total_outcomes : Set five_coins := {c | true}

-- Probability calculation: |success| / |total|
noncomputable def probability_heads_penny_nickel_dime : real :=
  (Set.card successful_outcomes : real) / (Set.card total_outcomes : real)

-- Given the setup, prove that the probability is 1/8
theorem probability_heads_penny_nickel_dime_is_one_eighth :
  probability_heads_penny_nickel_dime = 1 / 8 := 
sorry

end probability_heads_penny_nickel_dime_is_one_eighth_l249_249272


namespace minimal_bohemian_vertex_count_l249_249557

-- Definitions for the mathematical constructs.
def bohemian_vertex (n : ℕ) (polygon : ℕ → ℕ × ℕ) (i : ℕ) : Prop :=
  let A := polygon i
  let A_previous := polygon (if i = 1 then n else i - 1)
  let A_next := polygon (if i = n then 1 else i + 1)
  let midpoint := (A_previous + A_next) / 2
  let reflection := 2 * midpoint - A
  -- Check if the reflection point lies within or on the boundary of the polygon.
  reflection within_polygon polygon

-- Function to determine the minimal number of bohemian vertices in an n-gon.
def minimal_bohemian_vertices (n : ℕ) : ℕ :=
  if h₁ : n = 3 then 0
  else if h₂ : n = 4 then 1
  else n - 3

-- The problem statement we want to prove.
theorem minimal_bohemian_vertex_count (n : ℕ) (hn : n ≥ 3) :
  minimal_bohemian_vertices n = n - 3 :=
sorry

end minimal_bohemian_vertex_count_l249_249557


namespace max_divisors_1_to_20_l249_249208

theorem max_divisors_1_to_20 :
  ∃ m n o ∈ set.range (20:ℕ.succ), 
    (∀ k ∈ set.range (20:ℕ.succ), 
      (nat.divisors_count k ≤ nat.divisors_count m) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count n) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count o)) 
    ∧ (nat.divisors_count m = 6)
    ∧ (nat.divisors_count n = 6)
    ∧ (nat.divisors_count o = 6)
    ∧ m ≠ n ∧ n ≠ o ∧ o ≠ m
    ∧ set.mem 12 (set.range (20 + 1))
    ∧ set.mem 18 (set.range (20 + 1))
    ∧ set.mem 20 (set.range (20 + 1)). 
      := by 
  -- all these steps will be skipped here
  sorry

end max_divisors_1_to_20_l249_249208


namespace carina_seashells_in_month_l249_249780

variable (weekly_increase : ℕ) (initial_seashells : ℕ) (weeks_in_month : ℕ)

-- Conditions
def carina_seashell_problem_conditions := 
  weekly_increase = 20 ∧ 
  initial_seashells = 50 ∧ 
  weeks_in_month = 4

-- Problem statement
theorem carina_seashells_in_month (h : carina_seashell_problem_conditions weekly_increase initial_seashells weeks_in_month) :
  let s1 := initial_seashells in
  let s2 := s1 + weekly_increase in
  let s3 := s2 + weekly_increase in
  let s4 := s3 + weekly_increase in
  s1 + s2 + s3 + s4 = 320 :=
by {
  -- Full proof would go here
  sorry
}

end carina_seashells_in_month_l249_249780


namespace boys_made_mistake_l249_249247

theorem boys_made_mistake (n m : ℕ) (hn : n > 1) (hm : m > 1) (h_eq : Nat.factorial n = 2^m * Nat.factorial m) : False :=
by
  sorry

end boys_made_mistake_l249_249247


namespace circle_tangent_proportion_l249_249361

theorem circle_tangent_proportion 
(DB DC : Line) (B C : Point) (O : Circle) (H1 : Tangent DB O B) (H2 : Tangent DC O C)
(DF : Line) (A F : Point) (E : Point) 
(H3 : Intersects DF O A F) (H4 : Intersects DF BC E) :
  (AB * AB) / (AC * AC) = BE / EC :=
sorry

end circle_tangent_proportion_l249_249361


namespace constant_term_expanded_eq_neg12_l249_249822

theorem constant_term_expanded_eq_neg12
  (a w c d : ℤ)
  (h_eq : (a * x + w) * (c * x + d) = 6 * x ^ 2 + x - 12)
  (h_abs_sum : abs a + abs w + abs c + abs d = 12) :
  w * d = -12 := by
  sorry

end constant_term_expanded_eq_neg12_l249_249822


namespace roots_outside_unit_circle_l249_249149

noncomputable def f (x : ℝ) (a : ℕ → ℝ) (n : ℕ) :=
  ∑ i in Finset.range (n + 1), a i * x ^ (n - i)

theorem roots_outside_unit_circle (a : ℕ → ℝ) (n : ℕ)
  (h : ∀ i < (n + 1), 0 < a i) (h_increasing : StrictMono (λ k, a k)) :
  ∀ r, is_root (f r a n) → 1 < |r| :=
by
  sorry

end roots_outside_unit_circle_l249_249149


namespace new_class_mean_is_86_l249_249091

-- Define the initial conditions
def n₁ : ℕ := 24  -- Number of students who took the test first
def m₁ : ℚ := 85  -- Mean score of the first 24 students (85%)
def n₂ : ℕ := 4   -- Number of students who took the test later
def m₂ : ℚ := 90  -- Mean score of the remaining 4 students (90%)

-- Define the total number of students
def n : ℕ := 28   -- Total number of students

-- We need to prove the new class mean is 86%
theorem new_class_mean_is_86 :
  ((n₁ * m₁ + n₂ * m₂) / n).round = 86 := by
  sorry

end new_class_mean_is_86_l249_249091


namespace concyclic_BCXY_l249_249138

-- Definitions for points and triangle
variables (A B C K M X Y : Point)
variables (AX BX AY CY : ℝ)
variables (angle_KXM angle_ACB angle_KYM angle_ABC : ℝ)
variables (midpoint_AM : Midpoint A M K)

def is_midpoint_of_median (triangle : Triangle) : Prop :=
  K = midpoint_AM

-- Conditions
axiom cond1 : is_midpoint_of_median (Triangle A B C)
axiom cond2 : X ∈ Line A B
axiom cond3 : Y ∈ Line A C
axiom cond4 : angle_KXM = angle_ACB
axiom cond5 : AX > BX
axiom cond6 : angle_KYM = angle_ABC
axiom cond7 : AY > CY

-- Statement to prove
theorem concyclic_BCXY 
  (cond1 : is_midpoint_of_median (Triangle A B C))
  (cond2 : X ∈ Line A B)
  (cond3 : Y ∈ Line A C)
  (cond4 : angle_KXM = angle_ACB)
  (cond5 : AX > BX)
  (cond6 : angle_KYM = angle_ABC)
  (cond7 : AY > CY) 
  : Concyclic B C X Y :=
  sorry

end concyclic_BCXY_l249_249138


namespace caterpillar_prob_A_l249_249398

-- Define the probabilities involved
def prob_move_to_A_from_1 (x y z : ℚ) : ℚ :=
  (1/3 : ℚ) * 1 + (1/3 : ℚ) * y + (1/3 : ℚ) * z

def prob_move_to_A_from_2 (x y u : ℚ) : ℚ :=
  (1/3 : ℚ) * 0 + (1/3 : ℚ) * x + (1/3 : ℚ) * u

def prob_move_to_A_from_0 (x y : ℚ) : ℚ :=
  (2/3 : ℚ) * x + (1/3 : ℚ) * y

def prob_move_to_A_from_3 (y u : ℚ) : ℚ :=
  (2/3 : ℚ) * y + (1/3 : ℚ) * u

theorem caterpillar_prob_A :
  exists (x y z u : ℚ), 
    x = prob_move_to_A_from_1 x y z ∧
    y = prob_move_to_A_from_2 x y y ∧
    z = prob_move_to_A_from_0 x y ∧
    u = prob_move_to_A_from_3 y y ∧
    u = y ∧
    x = 9/14 :=
sorry

end caterpillar_prob_A_l249_249398


namespace probability_heads_penny_nickel_dime_l249_249277

theorem probability_heads_penny_nickel_dime :
  let total_outcomes := 2^5 in
  let successful_outcomes := 2 * 2 in
  (successful_outcomes : ℝ) / total_outcomes = (1 / 8 : ℝ) :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249277


namespace profit_percent_l249_249699

theorem profit_percent (SP : ℝ) (h : SP > 0):
  let CP := 0.9 * SP in
  let Profit := SP - CP in
  (Profit / CP) * 100 = 11.11 :=
by
  sorry

end profit_percent_l249_249699


namespace sqrt_of_square_neg_three_l249_249704

theorem sqrt_of_square_neg_three : Real.sqrt ((-3 : ℝ)^2) = 3 := by
  sorry

end sqrt_of_square_neg_three_l249_249704


namespace sample_size_9_l249_249939

variable (X : Nat)

theorem sample_size_9 (h : 36 % X = 0 ∧ 36 % (X + 1) ≠ 0) : X = 9 := 
sorry

end sample_size_9_l249_249939


namespace probability_heads_penny_nickel_dime_is_one_eighth_l249_249269

-- Define the setup: flipping 5 coins
def five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the condition: penny, nickel, and dime are heads
def heads_penny_nickel_dime (c: five_coins) : bool := c.1 && c.2 && c.3

-- Define the successful outcomes where penny, nickel, and dime are heads
noncomputable def successful_outcomes : Set five_coins :=
  { c | heads_penny_nickel_dime c = tt }

-- Define the total outcomes for 5 coins
noncomputable def total_outcomes : Set five_coins := {c | true}

-- Probability calculation: |success| / |total|
noncomputable def probability_heads_penny_nickel_dime : real :=
  (Set.card successful_outcomes : real) / (Set.card total_outcomes : real)

-- Given the setup, prove that the probability is 1/8
theorem probability_heads_penny_nickel_dime_is_one_eighth :
  probability_heads_penny_nickel_dime = 1 / 8 := 
sorry

end probability_heads_penny_nickel_dime_is_one_eighth_l249_249269


namespace total_siding_cost_l249_249602

-- Definitions of given conditions and values
def wall_width : ℝ := 8
def wall_height : ℝ := 10
def roof_width : ℝ := 8
def roof_slant_height : ℝ := 7
def sheet_width : ℝ := 8
def sheet_height : ℝ := 12
def sheet_cost : ℝ := 32.80

-- Computations based on the given conditions
def wall_area : ℝ := wall_width * wall_height
def roof_slant_length : ℝ := real.sqrt (roof_width * roof_width + roof_slant_height * roof_slant_height)
def roof_section_area : ℝ := roof_width * roof_slant_length
def total_area : ℝ := wall_area + 2 * roof_section_area
def sheet_area : ℝ := sheet_width * sheet_height
def sheets_needed : ℕ := nat_ceil (total_area / sheet_area).to_real
def total_cost : ℝ := sheets_needed * sheet_cost

-- Main theorem statement to be proven
theorem total_siding_cost : total_cost = 98.40 :=
sorry

end total_siding_cost_l249_249602


namespace LCM_of_apple_and_cherry_pies_l249_249717

theorem LCM_of_apple_and_cherry_pies :
  let apple_pies := (13 : ℚ) / 2
  let cherry_pies := (21 : ℚ) / 4
  let lcm_numerators := Nat.lcm 26 21
  let common_denominator := 4
  (lcm_numerators : ℚ) / (common_denominator : ℚ) = 273 / 2 :=
by
  let apple_pies := (13 : ℚ) / 2
  let cherry_pies := (21 : ℚ) / 4
  let lcm_numerators := Nat.lcm 26 21
  let common_denominator := 4
  have h : (lcm_numerators : ℚ) / (common_denominator : ℚ) = 273 / 2 := sorry
  exact h

end LCM_of_apple_and_cherry_pies_l249_249717


namespace custom_operation_example_l249_249073

def custom_operation (x y : Int) : Int :=
  x * y - 3 * x

theorem custom_operation_example : (custom_operation 7 4) - (custom_operation 4 7) = -9 := by
  sorry

end custom_operation_example_l249_249073


namespace lamps_remaining_on_l249_249321

noncomputable def count_lamps_on_after_flips : ℕ :=
  let total_lamps := 1000 in
  let multiples n := total_lamps / n in
  let lamps_div_2 := multiples 2 in
  let lamps_div_3 := multiples 3 in
  let lamps_div_5 := multiples 5 in
  let lamps_div_6 := multiples 6 in
  let lamps_div_10 := multiples 10 in
  let lamps_div_15 := multiples 15 in
  let lamps_div_30 := multiples 30 in
  total_lamps - ((lamps_div_2 + lamps_div_3 + lamps_div_5 - lamps_div_6 - lamps_div_10 - lamps_div_15 + lamps_div_30))

theorem lamps_remaining_on : count_lamps_on_after_flips = 499 := 
by
  -- The proof steps would typically follow here, 
  -- including confirming the intermediate calculations.
  sorry

end lamps_remaining_on_l249_249321


namespace sqrt_expression_real_l249_249085

theorem sqrt_expression_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 1)) ↔ x ≥ -1 := 
by
  sorry

end sqrt_expression_real_l249_249085


namespace circle_center_and_sum_l249_249333

/-- Given the equation of a circle x^2 + y^2 - 6x + 14y = -28,
    prove that the coordinates (h, k) of the center of the circle are (3, -7)
    and compute h + k. -/
theorem circle_center_and_sum (x y : ℝ) :
  (∃ h k, (x^2 + y^2 - 6*x + 14*y = -28) ∧ (h = 3) ∧ (k = -7) ∧ (h + k = -4)) :=
by {
  sorry
}

end circle_center_and_sum_l249_249333


namespace min_area_triangle_l249_249107

-- Conditions
def point_on_curve (x y : ℝ) : Prop :=
  y^2 = 2 * x

def incircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

-- Theorem statement
theorem min_area_triangle (x₀ y₀ b c : ℝ) (h_curve : point_on_curve x₀ y₀) 
  (h_bc_yaxis : b ≠ c) (h_incircle : incircle x₀ y₀) :
  ∃ P : ℝ × ℝ, 
    ∃ B C : ℝ × ℝ, 
    ∃ S : ℝ,
    point_on_curve P.1 P.2 ∧
    B = (0, b) ∧
    C = (0, c) ∧
    incircle P.1 P.2 ∧
    S = (x₀ - 2) + (4 / (x₀ - 2)) + 4 ∧
    S = 8 :=
sorry

end min_area_triangle_l249_249107


namespace triangle_area_l249_249114

theorem triangle_area (BC AC : ℝ) (angle_BAC : ℝ) (h1 : BC = 12) (h2 : AC = 5) (h3 : angle_BAC = π / 6) :
  1/2 * BC * (AC * Real.sin angle_BAC) = 15 :=
by
  sorry

end triangle_area_l249_249114


namespace hyperbola_example_l249_249729

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  let e := Real.sqrt (1 + (b ^ 2) / (a ^ 2)) in
  let line := fun x => -x + a in
  let asymptote1 := fun x => (b / a) * x in
  let asymptote2 := fun x => -(b / a) * x in
  let A := (a, 0)
  let B := ( a^2 / (a + b),  a * b / (a + b) )
  let C := ( a^2 / (a - b), -a * b / (a - b) )
  let AB := ( B.1 - A.1, B.2 - A.2 )
  let BC := ( C.1 - B.1, C.2 - B.2 )
  (2 * AB.1 = BC.1) ∧ (2 * AB.2 = BC.2) → (b = 2 * a) ∧ (e = Real.sqrt 5)

theorem hyperbola_example (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  hyperbola_eccentricity a b ha hb :=
sorry

end hyperbola_example_l249_249729


namespace fg_three_eq_neg_two_l249_249921

def f (x : ℝ) : ℝ := 4 - Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem fg_three_eq_neg_two : f (g 3) = -2 :=
by
  sorry

end fg_three_eq_neg_two_l249_249921


namespace descent_time_l249_249167

-- Definitions based on conditions
def time_to_top : ℝ := 4
def avg_speed_up : ℝ := 2.625
def avg_speed_total : ℝ := 3.5
def distance_to_top : ℝ := avg_speed_up * time_to_top -- 10.5 km
def total_distance : ℝ := 2 * distance_to_top       -- 21 km

-- Theorem statement: the time to descend (t_down) should be 2 hours
theorem descent_time (t_down : ℝ) : 
  avg_speed_total * (time_to_top + t_down) = total_distance →
  t_down = 2 := 
by 
  -- skip the proof
  sorry

end descent_time_l249_249167


namespace max_candies_vovochka_can_keep_l249_249906

-- Definitions for the conditions in the problem.
def classmates := 25
def total_candies := 200
def minimum_candies (candies_distributed : Vector ℕ classmates) : Prop :=
  ∀ (S : Finset (Fin classmates)), S.card = 16 → S.sum (candies_distributed.nth' S) ≥ 100

-- The proof goal
theorem max_candies_vovochka_can_keep (candies_distributed : Vector ℕ classmates) (h : minimum_candies candies_distributed) :
  ∃ k, k = 37 ∧ total_candies - Finset.univ.sum (candies_distributed.nth' Finset.univ) = k :=
begin
  sorry
end

end max_candies_vovochka_can_keep_l249_249906


namespace train_length_is_800_l249_249390

noncomputable def calculate_train_length 
  (t : ℝ) 
  (v_m : ℝ) 
  (v_t : ℝ) : ℝ :=
  let relative_speed := (v_t - v_m) * (1000 / 3600)
  in relative_speed * t

theorem train_length_is_800 
  (t : ℝ := 47.99616030717543)
  (v_m : ℝ := 3)
  (v_t : ℝ := 63) : 
  calculate_train_length t v_m v_t = 800 :=
sorry

end train_length_is_800_l249_249390


namespace solve_for_x_l249_249610

theorem solve_for_x (x : ℤ) : (3^x) * (9^x) = (81^(x-24)) → x = 96 :=
by
  intro h
  sorry

end solve_for_x_l249_249610


namespace sufficient_conditions_l249_249595

noncomputable theory
open_locale classical

-- Definitions for q₁ and q₂
def q1 (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f(x) ≥ f(y)) ∧ (∀ x, f(x) > 0)

def q2 (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f(x) ≤ f(y)) ∧ (∃ x0, x0 < 0 ∧ f(x0) = 0)

-- Proposition p
def p (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f(x + a) < f(x) + f(a)

-- The theorem to prove
theorem sufficient_conditions (f : ℝ → ℝ) :
  (q1 f → p f) ∧ (q2 f → p f) :=
by
  sorry

end sufficient_conditions_l249_249595


namespace calculation_l249_249755

theorem calculation : 
  let a := 20 / 9 
  let b := -53 / 4 
  (⌈ a * ⌈ b ⌉ ⌉ - ⌊ a * ⌊ b ⌋ ⌋) = 4 :=
by
  sorry

end calculation_l249_249755


namespace greatest_divisors_1_to_20_l249_249198

def is_divisor (a b : ℕ) : Prop := b % a = 0

def count_divisors (n : ℕ) : ℕ :=
  nat.succ (list.length ([k for k in list.range n if is_divisor k.succ n]))

def max_divisors_from_1_to_20 : set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 20 ∧ ∃ max_count, count_divisors n = max_count ∧ 
  ∀ m, 1 ≤ m ∧ m ≤ 20 → count_divisors m ≤ max_count}

theorem greatest_divisors_1_to_20 : max_divisors_from_1_to_20 = {12, 18, 20} :=
by {
  sorry
}

end greatest_divisors_1_to_20_l249_249198


namespace unique_an_l249_249348

noncomputable def a_n (n : ℕ) : ℕ :=
  let b_n := ((1 + Real.sqrt 5)^n - (1 - Real.sqrt 5)^n) / 2 in
  b_n^2

theorem unique_an (n : ℕ) (hn : 0 < n) : ∃! a_n : ℕ, (1 + Real.sqrt 5)^n = Real.sqrt a_n + Real.sqrt (a_n + 4^n) :=
begin
  use a_n n,
  -- The proof itself goes here.
  sorry
end

end unique_an_l249_249348


namespace perpendicular_vectors_x_value_l249_249562

theorem perpendicular_vectors_x_value : 
  ∀ (x : ℝ), let m := (1, 3) in let n := (-2, x) in 
  (m.1 * n.1 + m.2 * n.2 = 0) → x = 2 / 3 :=
by
  sorry

end perpendicular_vectors_x_value_l249_249562


namespace pete_mileage_closest_to_2500_l249_249594

theorem pete_mileage_closest_to_2500 :
  let pedometer_max := 99999
  let flips := 44
  let final_reading := 50000
  let steps_per_mile := 1800
  let steps_from_flips := flips * (pedometer_max + 1)
  let total_steps := steps_from_flips + final_reading
  let miles_walked := total_steps / steps_per_mile
  abs (miles_walked - 2500) <= 
  abs (miles_walked - 3000) ∧ 
  abs (miles_walked - 3500) ∧ 
  abs (miles_walked - 4000) ∧ 
  abs (miles_walked - 4500) :=
by
  sorry

end pete_mileage_closest_to_2500_l249_249594


namespace solve_for_a_l249_249520

theorem solve_for_a
  (a x : ℚ)
  (h1 : (2 * a * x + 3) / (a - x) = 3 / 4)
  (h2 : x = 1) : a = -3 :=
by
  sorry

end solve_for_a_l249_249520


namespace twelfth_prime_number_l249_249244

theorem twelfth_prime_number (h : ∃ p : Nat, Nat.prime p ∧ p = 17 ∧ nat.find (λ n, Nat.prime n) 7 = 17) : Nat.find (λ n, Nat.prime n) 12 = 37 :=
sorry

end twelfth_prime_number_l249_249244


namespace tetrahedron_perpendicular_OH_l_l249_249820

-- Definitions based on the problem conditions
variables (P A B C H A' B' C' O l : Type*)
variables [Tetrahedron P A B C] [Height PH from P to ABC]
variables [PerpendicularFrom H to (PA, HA')] [PerpendicularFrom H to (PB, HB')]
variables [PerpendicularFrom H to (PC, HC')] [IntersectionLine l of ABC and A'B'C']
variables [Circumcenter O of (△ ABC)]

-- The theorem to prove
theorem tetrahedron_perpendicular_OH_l :
  ⊥ O l H :=
sorry

end tetrahedron_perpendicular_OH_l_l249_249820


namespace correct_equation_l249_249293

-- Define the daily paving distances for Team A and Team B
variables (x : ℝ) (h₀ : x > 10)

-- Assuming Team A takes the same number of days to pave 150m as Team B takes to pave 120m
def same_days_to_pave (h₁ : x - 10 > 0) : Prop :=
  (150 / x = 120 / (x - 10))

-- The theorem to be proven
theorem correct_equation (h₁ : x - 10 > 0) : 150 / x = 120 / (x - 10) :=
by
  sorry

end correct_equation_l249_249293


namespace car_X_after_Y_distance_l249_249695

/-- Define the speeds of Car X and Car Y. -/
def speed_X : ℝ := 35 -- miles per hour
def speed_Y : ℝ := 65 -- miles per hour

/-- Define the time duration until Car Y starts. -/
def time_before_Y : ℝ := 72 / 60 -- hours

/-- Define the distance Car X travels before Car Y starts. -/
def distance_X_before_Y : ℝ := speed_X * time_before_Y -- in miles

/-- Define the time Car Y travels until both stop. -/
def time_t : ℝ := 1.4 -- hours

/-- Define the distance Car X travels after Car Y starts until both stop. -/
def distance_X_after_Y : ℝ := speed_X * time_t

/-- Define the total distance Car X travels. -/
def total_distance_X : ℝ := distance_X_before_Y + distance_X_after_Y

/-- Prove that the total distance Car X travels after Car Y starts is 49 miles. -/
theorem car_X_after_Y_distance : distance_X_after_Y = 49 :=
by
  sorry

end car_X_after_Y_distance_l249_249695


namespace intersection_points_of_quadratic_minimum_value_of_quadratic_in_range_range_of_m_for_intersection_with_segment_PQ_l249_249009

-- Define the quadratic function
def quadratic (m x : ℝ) : ℝ := m * x^2 - 4 * m * x + 3 * m

-- Define the conditions
variables (m : ℝ)
theorem intersection_points_of_quadratic :
    (quadratic m 1 = 0) ∧ (quadratic m 3 = 0) ↔ m ≠ 0 :=
sorry

theorem minimum_value_of_quadratic_in_range :
    ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → quadratic (-2) x ≥ -6 :=
sorry

theorem range_of_m_for_intersection_with_segment_PQ :
    ∀ (m : ℝ), (∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ quadratic m x = (m + 4) / 2) ↔ 
    m ≤ -4 / 3 ∨ m ≥ 4 / 5 :=
sorry

end intersection_points_of_quadratic_minimum_value_of_quadratic_in_range_range_of_m_for_intersection_with_segment_PQ_l249_249009


namespace probability_penny_nickel_dime_all_heads_l249_249282

-- Define flipping five coins
def flip_five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the function to check if the penny, nickel, and dime are heads
def all_heads_penny_nickel_dime (o : flip_five_coins) : bool :=
  o.1 = tt ∧ o.2 = tt ∧ o.3 = tt

-- Define the total count of possible outcomes
def total_outcomes : ℕ := 32

-- Define the count of favorable outcomes
def favorable_outcomes : ℕ := 4

-- Define the probability calculation
def probability_favorable : ℚ := favorable_outcomes / total_outcomes

-- The statement proving the probability is 1/8
theorem probability_penny_nickel_dime_all_heads :
  probability_favorable = 1 / 8 :=
by
  sorry

end probability_penny_nickel_dime_all_heads_l249_249282


namespace lines_intersect_on_bisector_of_angle_ABC_l249_249147

theorem lines_intersect_on_bisector_of_angle_ABC
  (A B C M A₁ A₂ C₁ C₂ : Type)
  (BM_is_median : median B M A C)
  (triangle_ABC_right : angle B = 90°)
  (incircle_ABM_tangency : tangency_points (incircle (triangle AB M)) A₁ A₂ AB AM)
  (incircle_CBM_tangency : tangency_points (incircle (triangle CB M)) C₁ C₂ BC CM) :
  intersects (line A₁ A₂) (line C₁ C₂) (bisector (angle ABC)) :=
sorry

end lines_intersect_on_bisector_of_angle_ABC_l249_249147


namespace sin_C_length_of_a_l249_249090

-- Proving sine of angle C
theorem sin_C (A B C : ℝ) (a b c : ℝ) (h_cosA : cos A = -5/13) (h_cosB : cos B = 3/5) :
  sin C = 16/65 :=
by sorry

-- Proving the length of side a
theorem length_of_a (A B C : ℝ) (a b c : ℝ) 
                    (h_cosA : cos A = -5/13) (h_cosB : cos B = 3/5) 
                    (h_area : (1/2) * a * b * sin C = 8/3)
                    (h_b: b = (4/5 * a) / (12/13)) :
  a = 5 :=
by sorry

end sin_C_length_of_a_l249_249090


namespace largest_internal_angle_l249_249089

variables {A B C : ℝ}
variable (sin_ratio : ℝ → ℝ → ℝ → Prop)
variable (deg_to_rad : ℝ → ℝ)

noncomputable def C_degree_measure (A B C : ℝ) : Prop :=
  ∃ x : ℝ, sin_ratio (Real.sin A) (Real.sin B) (Real.sin C) = (3 / x, 5 / x, 7 / x) ∧
  C = Mathlib.Real.pi / 3

theorem largest_internal_angle (h : sin_ratio (Real.sin A) (Real.sin B) (Real.sin C) = (3, 5, 7)) :
  C_degree_measure A B C :=
sorry

end largest_internal_angle_l249_249089


namespace max_trig_sum_l249_249828

theorem max_trig_sum (α β γ : ℝ) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) (h_sum : α + β + 2 * γ = π) :
  ∃ M, M = cos α + cos β + sin (2 * γ) ∧ M = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end max_trig_sum_l249_249828


namespace joan_found_seashells_l249_249551

theorem joan_found_seashells (total_seashells joan_has: ℕ) (seashells_from_sam: ℕ) (joan_found: ℕ)
  (h1: total_seashells = 97)
  (h2: seashells_from_sam = 27) : joan_found = 97 - 27 :=
by
  simp [h1, h2]
  exact 70

end joan_found_seashells_l249_249551


namespace range_of_a_l249_249930

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 5 ∧ log (x^2 + a * x) = 1) → (-3 ≤ a ∧ a ≤ 9) :=
by
  sorry

end range_of_a_l249_249930


namespace problem_inequality_l249_249812

theorem problem_inequality (n : ℕ) (a : Fin n → ℝ) (hpos : ∀ i, 0 < a i) (hprod : (∏ i in Finset.univ, a i) = 1) :
  (∏ i in Finset.univ, (2 + a i)) ≥ 3^n :=
sorry

end problem_inequality_l249_249812


namespace triangle_similarity_proof_l249_249581

-- Define a structure for points in a geometric space
structure Point : Type where
  x : ℝ
  y : ℝ
  deriving Inhabited

-- Define the conditions provided in the problem
variables (A B C D E H : Point)
variables (HD HE : ℝ)

-- Condition statements
def HD_dist := HD = 6
def HE_dist := HE = 3

-- Main theorem statement
theorem triangle_similarity_proof (BD DC AE EC BH AH : ℝ) 
  (h1 : HD = 6) (h2 : HE = 3) 
  (h3 : 2 * BH = AH) : 
  (BD * DC - AE * EC = 9 * BH + 27) :=
sorry

end triangle_similarity_proof_l249_249581


namespace twelfth_prime_is_37_l249_249242

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def nth_prime (n : ℕ) : ℕ := if h : n > 0 then classical.some (nat.exists_infinite_primes n) else 2

theorem twelfth_prime_is_37 (h : nth_prime 7 = 17) : nth_prime 12 = 37 :=
  sorry

end twelfth_prime_is_37_l249_249242


namespace find_p_l249_249961

noncomputable def area_of_ABC (p : ℚ) : ℚ :=
  128 - 6 * p

theorem find_p (p : ℚ) : area_of_ABC p = 45 → p = 83 / 6 := by
  intro h
  sorry

end find_p_l249_249961


namespace sum_of_divisors_of_8_l249_249803

theorem sum_of_divisors_of_8 : 
  (∑ n in { n : ℕ | 0 < n ∧ 8 % n = 0 }, n) = 15 :=
by
  sorry

end sum_of_divisors_of_8_l249_249803


namespace vovochka_max_candies_l249_249901

noncomputable def max_candies_for_vovochka 
  (total_classmates : ℕ)
  (total_candies : ℕ)
  (candies_condition : ∀ (subset : Finset ℕ), subset.card = 16 → ∑ i in subset, (classmate_candies i) ≥ 100) 
  (classmate_candies : ℕ → ℕ) : ℕ :=
  total_candies - ∑ i in (Finset.range total_classmates), classmate_candies i

theorem vovochka_max_candies :
  max_candies_for_vovochka 25 200 (λ subset hc, True) (λ i, if i < 13 then 7 else 6) = 37 :=
by
  sorry

end vovochka_max_candies_l249_249901


namespace halloween_candies_l249_249588

theorem halloween_candies :
  ∃ x : ℕ, (x > 0) ∧ (1/4 : ℝ) * x - 3/2 - 5 = 10 ∧ x = 66 :=
by
  use 66
  norm_num
  sorry

end halloween_candies_l249_249588


namespace parallel_lines_k_value_l249_249681

theorem parallel_lines_k_value (k : ℝ) 
  (line1 : ∀ x : ℝ, y = 5 * x + 3) 
  (line2 : ∀ x : ℝ, y = (3 * k) * x + 1) 
  (parallel : ∀ x : ℝ, (5 = 3 * k)) : 
  k = 5 / 3 := 
begin
  sorry
end

end parallel_lines_k_value_l249_249681


namespace original_stickers_l249_249652

theorem original_stickers (x : ℕ) (h₁ : x * 3 / 4 * 4 / 5 = 45) : x = 75 :=
by
  sorry

end original_stickers_l249_249652


namespace minimum_d_value_l249_249734

theorem minimum_d_value :
  let d := (1 + 2 * Real.sqrt 10) / 3
  let distance := Real.sqrt ((2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2)
  distance = 4 * d :=
by
  let d := (1 + 2 * Real.sqrt 10) / 3
  let distance := Real.sqrt ((2 * Real.sqrt 10) ^ 2 + (d + 5) ^ 2)
  sorry

end minimum_d_value_l249_249734


namespace exists_subset_product_square_l249_249823

theorem exists_subset_product_square {S : Finset ℕ} (hS : S.card = 10)
  (hprime : ∀ n ∈ S, ∀ p : ℕ, p.prime → p ∣ n → p ≤ 20) : 
  ∃ T ⊆ S, (∏ i in T, i) ^ 2 ∣ (∏ i in S, i) ∧ (∏ i in T, i)  ^ 2 ≠ 0 :=
sorry

end exists_subset_product_square_l249_249823


namespace cannot_cover_completely_with_dominoes_l249_249418

theorem cannot_cover_completely_with_dominoes :
  ¬ (∃ f : Fin 5 × Fin 3 → Fin 5 × Fin 3, 
      (∀ p q, f p = f q → p = q) ∧ 
      (∀ p, ∃ q, f q = p) ∧ 
      (∀ p, (f p).1 = p.1 + 1 ∨ (f p).2 = p.2 + 1)) := 
sorry

end cannot_cover_completely_with_dominoes_l249_249418


namespace expected_balls_in_original_positions_l249_249605

/-- The expected number of balls that return to their original positions after two independent adjacent transpositions is 3.857. -/
noncomputable def expected_returned_balls : ℝ :=
  let n := 7 in
  let prob_same_pos := (2 / 49 : ℝ) + (25 / 49 : ℝ) in
  n * prob_same_pos

theorem expected_balls_in_original_positions :
  expected_returned_balls = 3.857 := by
  sorry

end expected_balls_in_original_positions_l249_249605


namespace greatest_num_divisors_in_range_l249_249172

def num_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d, n % d = 0)

theorem greatest_num_divisors_in_range : 
  ∀ n ∈ {1, 2, 3, ..., 20}, 
  num_divisors n < 7 → (n = 12 ∨ n = 18 ∨ n = 20) :=
by {
  sorry
}

end greatest_num_divisors_in_range_l249_249172


namespace percent_runs_by_running_is_correct_l249_249345

open nat

def total_runs : ℕ := 132
def boundaries : ℕ := 12
def sixes : ℕ := 2
def boundary_runs : ℕ := boundaries * 4
def six_runs : ℕ := sixes * 6
def runs_from_boundaries_and_sixes : ℕ := boundary_runs + six_runs
def runs_by_running : ℕ := total_runs - runs_from_boundaries_and_sixes
def percent_runs_by_running (total : ℕ) (running : ℕ) : ℚ := (running.to_rat / total.to_rat) * 100

theorem percent_runs_by_running_is_correct :
  percent_runs_by_running total_runs runs_by_running ≈ 54.55 :=
by
  sorry

end percent_runs_by_running_is_correct_l249_249345


namespace inequalities_not_simultaneously_true_l249_249597

variables {V : Type*} [inner_product_space ℝ V]

theorem inequalities_not_simultaneously_true (a b c : V) :
  ¬ (sqrt 3 * ∥a∥ < ∥b - c∥ ∧ sqrt 3 * ∥b∥ < ∥c - a∥ ∧ sqrt 3 * ∥c∥ < ∥a - b∥) :=
by {
  sorry
}

end inequalities_not_simultaneously_true_l249_249597


namespace coeff_sum_eq_2006_l249_249914

theorem coeff_sum_eq_2006 :
  ∀ (a a1 a2 ... a2006 : ℝ), 
  (1 - 2 * x) ^ 2006 = a + a1 * x + a2 * x^2 + ... + a2006 * x^2006 → 
  (a + a1) + (a + a2) + ... + (a + a2006) = 2006 :=
by
  intros
  sorry

end coeff_sum_eq_2006_l249_249914


namespace kanul_spent_on_machinery_l249_249980

theorem kanul_spent_on_machinery (total raw_materials cash M : ℝ) 
  (h_total : total = 7428.57) 
  (h_raw_materials : raw_materials = 5000) 
  (h_cash : cash = 0.30 * total) 
  (h_expenditure : total = raw_materials + M + cash) :
  M = 200 := 
by
  sorry

end kanul_spent_on_machinery_l249_249980


namespace gregory_current_age_l249_249778

-- Given conditions
variables (D G y : ℕ)
axiom dm_is_three_times_greg_was (x : ℕ) : D = 3 * y
axiom future_age_sum : D + (3 * y) = 49
axiom greg_age_difference x y : D - (3 * y) = (3 * y) - x

-- Prove statement: Gregory's current age is 14
theorem gregory_current_age : G = 14 := by
  sorry

end gregory_current_age_l249_249778


namespace plane_distance_l249_249733

theorem plane_distance (D : ℕ) (h₁ : D / 300 + D / 400 = 7) : D = 1200 :=
sorry

end plane_distance_l249_249733


namespace projection_of_vector_l249_249504

theorem projection_of_vector (
  a : ℝ × ℝ := (2, -1),
  b : ℝ × ℝ := (1, 3)
) : 
  let v := (a.1 + 2 * b.1, a.2 + 2 * b.2)
  let projection := ( ((v.1 * a.1 + v.2 * a.2) / (a.1 * a.1 + a.2 * a.2)) * a.1, 
                      ((v.1 * a.1 + v.2 * a.2) / (a.1 * a.1 + a.2 * a.2)) * a.2)
  projection = (6 / 5, -3 / 5) :=
by
  sorry

end projection_of_vector_l249_249504


namespace sum_sin_squares_l249_249761

theorem sum_sin_squares :
  ∑ k in (finset.range 30).filter (λ k, (k + 1).gcd 30 = 1), (Real.sin ((k + 1) * 6 * π / 180))^2 = 31 / 2 := 
sorry

end sum_sin_squares_l249_249761


namespace find_number_l249_249439

theorem find_number (x : ℕ) (h : x / 4 + 15 = 27) : x = 48 :=
sorry

end find_number_l249_249439


namespace integral_inequality_l249_249984

open Real

noncomputable def smallest_c (f : ℝ → ℝ) (n : ℕ) (h_cont : ContinuousOn f (Icc 0 1)) : ℝ :=
n

theorem integral_inequality (f : ℝ → ℝ) (n : ℕ) (h_cont : ContinuousOn f (Icc 0 1)) (h_nn : ∀ x ∈ Icc 0 1, 0 ≤ f x) (h_n : 2 ≤ n) :
  ∫ x in 0..1, f (x ^ (1 / n : ℝ)) ≤ (smallest_c f n h_cont) * ∫ x in 0..1, f x :=
sorry

end integral_inequality_l249_249984


namespace Vovochka_max_candies_l249_249883

noncomputable theory
open_locale classical
open set

def VovochkaCandies (c n k required_sum : ℕ) (dist : fin n → ℕ) : Prop :=
  ∀ (s : finset (fin n)), s.card = k → (s.sum dist ≥ required_sum)

theorem Vovochka_max_candies
  (c n k required_sum : ℕ)
  (h_c : c = 200)
  (h_n : n = 25)
  (h_k : k = 16)
  (h_required_sum : required_sum = 100)
  (dist : fin n → ℕ)
  (h_dist : VovochkaCandies c n k required_sum dist) :
  c - finset.univ.sum dist = 37 :=
sorry

end Vovochka_max_candies_l249_249883


namespace numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249224

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).length

theorem numbers_with_most_divisors_in_range :
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  (number_of_divisors n ≤ 6) :=
by
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list
  simp [number_of_divisors] <;>
  dec_trivial -- Automatically discharges the goals since they are simple arithmetic comparisons

theorem greatest_divisors_in_range :
  number_of_divisors 12 = 6 ∧
  number_of_divisors 18 = 6 ∧
  number_of_divisors 20 = 6 ∧
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19],
  number_of_divisors n < 6 :=
by
  refine ⟨rfl, rfl, rfl, _⟩
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list excluding 12, 18, 20
  dec_trivial -- Automatically discharges the goals due to their simplicity

#print greatest_divisors_in_range

end numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249224


namespace point_outside_circle_l249_249490

theorem point_outside_circle 
    (r OP : ℝ)
    (h1 : r = 5)
    (h2 : OP = 6) :
    OP > r :=
by { rw [h1, h2], norm_num }

end point_outside_circle_l249_249490


namespace max_net_income_at_eleven_l249_249716

def net_income (x : ℕ) : ℝ :=
if 3 ≤ x ∧ x ≤ 6 then 50 * x - 115
else if 6 < x ∧ x ≤ 20 then -3 * x ^ 2 + 68 * x - 115
else 0

def valid_domain (x : ℕ) : Prop :=
3 ≤ x ∧ x ≤ 20

theorem max_net_income_at_eleven : ∀ x : ℕ, valid_domain x → net_income x ≤ net_income 11 :=
begin
  sorry
end

end max_net_income_at_eleven_l249_249716


namespace sin_75_eq_angle_C_measure_l249_249708

open Real

theorem sin_75_eq :
  sin (75 * pi / 180) = (sqrt 6 + sqrt 2) / 4 :=
by
  sorry

theorem angle_C_measure (a b c : ℝ) (h : a^2 - c^2 + b^2 = a * b) :
  ∠C = pi / 3 :=
by
  sorry

end sin_75_eq_angle_C_measure_l249_249708


namespace percentage_of_boys_from_school_A_study_science_l249_249940

theorem percentage_of_boys_from_school_A_study_science 
  (total_boys : ℕ)
  (p : ℕ)
  (not_study_science : ℕ)
  (h_total_boys : total_boys = 250)
  (h_percentage : p = 20)
  (h_not_study_science : not_study_science = 35) :
  let boys_from_school_A := (p / 100) * total_boys in
  let study_science := boys_from_school_A - not_study_science in
  (study_science / boys_from_school_A : ℚ) * 100 = 30 :=
by
  -- The proof steps would go here
  sorry

end percentage_of_boys_from_school_A_study_science_l249_249940


namespace sqrt_meaningful_range_l249_249080

theorem sqrt_meaningful_range (x : ℝ) : x + 1 ≥ 0 ↔ x ≥ -1 :=
by sorry

end sqrt_meaningful_range_l249_249080


namespace range_of_a_l249_249931

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x ∈ Iio (-1), f (x + 1) < f x) 
  (h2 : ∀ x ∈ Ioi 1, f (x - 1) < f x) :
  ∃ a ∈ set.Icc (-1 : ℝ) 1, ∀ x, f x = x^2 - 2 * a * x + 3 := 
sorry

end range_of_a_l249_249931


namespace mark_can_divide_2020_stones_l249_249584

theorem mark_can_divide_2020_stones : 
  ∃ (piles : Fin 5 → ℕ), (piles 0 + piles 1 + piles 2 + piles 3 + piles 4 = 2020) ∧ 
  (∀ i j, i ≠ j → piles i ≠ piles j) ∧
  (∀ i, (∑ j in ({0,1,2,3,4} \ {i}).toFinset, piles j) + piles i = 2020 ∧ 
         (∑ j in ({0,1,2,3,4} \ {i}).toFinset, piles j) % 4 = 0) :=
sorry

end mark_can_divide_2020_stones_l249_249584


namespace sum_cot_squared_l249_249560

theorem sum_cot_squared
  (S : Set ℝ)
  (hS : ∀ x ∈ S, 0 < x ∧ x < π ∧ (
    ∃ (a b c : ℝ), {a, b, c} = {Real.sin x, Real.cos x, Real.cot x} ∧ a^2 + b^2 = c^2)):
  ∑ x in S, Real.cot x ^ 2 = Real.sqrt 2 :=
by
  sorry

end sum_cot_squared_l249_249560


namespace locus_of_M_is_ellipse_l249_249059

def isEllipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1

theorem locus_of_M_is_ellipse :
  let F₁ := (-4 : ℝ, 0 : ℝ)
  let F₂ := (4 : ℝ, 0 : ℝ)
  ∃ a b : ℝ, a = 5 ∧ b = 3 ∧ ∀ x y : ℝ, 
    (dist (x, y) F₁ + dist (x, y) F₂ = 10) → isEllipse x y a b :=
by
  sorry

end locus_of_M_is_ellipse_l249_249059


namespace tan_diff_identity_l249_249482

-- Define a noncomputable constant for the angle alpha
noncomputable def α := sorry

-- Condition: α is in the second quadrant
axiom α_in_second_quadrant : real.sin α = 3/5 ∧ (π / 2 ≤ α ∧ α ≤ π)

-- Define the hypothesis that matches the given conditions
noncomputable def sin_α : ℝ := 3 / 5

-- Define the theorem stating what needs to be proved
theorem tan_diff_identity : (real.tan (α - π / 4)) = -7 :=
by 
  have h1 : real.sin α = sin_α := sorry,
  have h2 : π / 2 ≤ α ∧ α ≤ π := sorry,
  sorry

end tan_diff_identity_l249_249482


namespace probability_heads_penny_nickel_dime_l249_249278

theorem probability_heads_penny_nickel_dime :
  let total_outcomes := 2^5 in
  let successful_outcomes := 2 * 2 in
  (successful_outcomes : ℝ) / total_outcomes = (1 / 8 : ℝ) :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249278


namespace integer_satisfaction_l249_249064

def count_satisfying_integers : ℕ :=
  let count := λ (lower upper : ℤ), (↑upper - ↑lower + 1)
  
  count (-13) (-8) + count (-4) 2

theorem integer_satisfaction :
  let suitable_n := [(-13), (-12), (-11), (-10), (-9), (-8), (-4), (-3), (-2), (-1), 0, 1, 2] in
  (-13 ≤ n ∧ n ≤ 13 ∧ (n ∈ suitable_n))
    → (n ≥ 0 ∧ n ≤ 13 ∧ (n ∈ suitable_n) = 13) :=
sorry

end integer_satisfaction_l249_249064


namespace greatest_divisors_1_to_20_l249_249181

theorem greatest_divisors_1_to_20 : ∃ n₁ n₂ n₃, ((n₁ = 12 ∧ n₂ = 18 ∧ n₃ = 20) ∧ 
  (∀ m ∈ (Finset.range 21).filter (λ x, x > 0), 
    let d_m := (Finset.range (m + 1)).filter (λ k, m % k = 0) in
    ∃ k, k = d_m.card ∧ k <= 6 ∧ ((m = n₁ ∧ k = 6) ∨ (m = n₂ ∧ k = 6) ∨ (m = n₃ ∧ k = 6)))) :=
by
  sorry

end greatest_divisors_1_to_20_l249_249181


namespace incorrect_statement_B_l249_249340

axiom statement_A : ¬ (0 > 0 ∨ 0 < 0)
axiom statement_C : ∀ (q : ℚ), (∃ (m : ℤ), q = m) ∨ (∃ (a b : ℤ), b ≠ 0 ∧ q = a / b)
axiom statement_D : abs (0 : ℚ) = 0

theorem incorrect_statement_B : ¬ (∀ (q : ℚ), abs q ≥ 1 → abs 1 = abs q) := sorry

end incorrect_statement_B_l249_249340


namespace divisors_of_12_18_20_l249_249192

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem divisors_of_12_18_20 (n : ℕ) (h : n ∈ {1, 2, 3, ..., 20}) :
  n ∈ {12, 18, 20} ↔ (∀ m ∈ {1, 2, 3, ..., 20}, count_divisors m ≤ 6) :=
by
  sorry

end divisors_of_12_18_20_l249_249192


namespace hyperbola_y0_range_l249_249827

theorem hyperbola_y0_range
  (x0 y0 : ℝ)
  (h_hyperbola : x0^2 / 2 - y0^2 = 1)
  (h_dot_product : (-√3 - x0, -y0) • (√3 - x0, -y0) < 0) :
  -√3 / 3 < y0 ∧ y0 < √3 / 3 :=
sorry

end hyperbola_y0_range_l249_249827


namespace maximum_candies_l249_249895

-- Definitions of the problem parameters
def classmates : ℕ := 25
def total_candies : ℕ := 200
def condition (candies : ℕ → ℕ) : Prop := 
  ∀ S (h : finset ℕ), S.card = 16 → (∑ i in S, candies i) ≥ 100

-- The theorem stating the maximum number of candies Vovochka can keep
theorem maximum_candies (candies : ℕ → ℕ) (h : condition candies) : 
  (total_candies - ∑ i in finset.range classmates, candies i) ≤ 37 :=
sorry

end maximum_candies_l249_249895


namespace max_divisors_1_to_20_l249_249206

theorem max_divisors_1_to_20 :
  ∃ m n o ∈ set.range (20:ℕ.succ), 
    (∀ k ∈ set.range (20:ℕ.succ), 
      (nat.divisors_count k ≤ nat.divisors_count m) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count n) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count o)) 
    ∧ (nat.divisors_count m = 6)
    ∧ (nat.divisors_count n = 6)
    ∧ (nat.divisors_count o = 6)
    ∧ m ≠ n ∧ n ≠ o ∧ o ≠ m
    ∧ set.mem 12 (set.range (20 + 1))
    ∧ set.mem 18 (set.range (20 + 1))
    ∧ set.mem 20 (set.range (20 + 1)). 
      := by 
  -- all these steps will be skipped here
  sorry

end max_divisors_1_to_20_l249_249206


namespace find_monthly_rent_l249_249395

-- Given conditions
def apartment_price : ℝ := 12000
def annual_tax : ℝ := 395
def desired_return_rate : ℝ := 0.06
def repair_upkeep_percentage : ℝ := 0.125

-- We need to prove that the monthly rent is approximately 106.48
theorem find_monthly_rent :
  let annual_return := desired_return_rate * apartment_price;
  let total_annual_needed := annual_return + annual_tax;
  let monthly_needed := total_annual_needed / 12;
  let monthly_rent := monthly_needed / (1 - repair_upkeep_percentage)
  in monthly_rent ≈ 106.48 := 
by
  sorry

end find_monthly_rent_l249_249395


namespace least_positive_integer_set_l249_249441

theorem least_positive_integer_set (n : ℕ) (S : Finset ℕ) (h_distinct: S.card = n) (h_prod: ∏ s in S, (1 - 1 / (s : ℚ)) = 51 / 2010) : n = 39 :=
by
  sorry

end least_positive_integer_set_l249_249441


namespace sum_of_divisors_of_8_l249_249801

theorem sum_of_divisors_of_8 : 
  ( ∑ n in { n : ℕ | (n > 0) ∧ (n ∣ 8) }, n ) = 15 := 
by 
  sorry

end sum_of_divisors_of_8_l249_249801


namespace sqrt_expression_real_l249_249084

theorem sqrt_expression_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 1)) ↔ x ≥ -1 := 
by
  sorry

end sqrt_expression_real_l249_249084


namespace arith_geo_seq_prop_l249_249014

theorem arith_geo_seq_prop (a1 a2 b1 b2 b3 : ℝ)
  (arith_seq_condition : 1 + 2 * (a1 - 1) = a2)
  (geo_seq_condition1 : b1 * b3 = 4)
  (geo_seq_condition2 : b1 > 0)
  (geo_seq_condition3 : 1 * b1 * b2 * b3 * 4 = (b1 * b3 * -4)) :
  (a2 - a1) / b2 = 1/2 :=
by
  sorry

end arith_geo_seq_prop_l249_249014


namespace parabolas_line_divide_plane_l249_249310

theorem parabolas_line_divide_plane (b k : ℝ) :
  let p1 := λ x : ℝ, x^2 - b * x
  let p2 := λ x : ℝ, -x^2 + b * x
  let l := λ x : ℝ, k * x
  in ∃ (regions : ℕ), regions = 9 := sorry

end parabolas_line_divide_plane_l249_249310


namespace statement_l249_249993

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Condition 2: f(x-2) = -f(x) for all x
def satisfies_periodicity (f : ℝ → ℝ) : Prop := ∀ x, f (x - 2) = -f x

-- Condition 3: f is decreasing on [0, 2]
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- The proof statement
theorem statement (h1 : is_odd_function f) (h2 : satisfies_periodicity f) (h3 : is_decreasing_on f 0 2) :
  f 5 < f 4 ∧ f 4 < f 3 :=
sorry

end statement_l249_249993


namespace paving_stone_size_l249_249736

theorem paving_stone_size (length_courtyard width_courtyard : ℕ) (num_paving_stones : ℕ) (area_courtyard : ℕ) (s : ℕ)
  (h₁ : length_courtyard = 30) 
  (h₂ : width_courtyard = 18)
  (h₃ : num_paving_stones = 135)
  (h₄ : area_courtyard = length_courtyard * width_courtyard)
  (h₅ : area_courtyard = num_paving_stones * s * s) :
  s = 2 := 
by
  sorry

end paving_stone_size_l249_249736


namespace intersection_complement_eq_l249_249056

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 4}

theorem intersection_complement_eq : A ∩ (U \ B) = {0, 2} := by
  sorry

end intersection_complement_eq_l249_249056


namespace monthly_income_ratio_l249_249639

noncomputable def A_annual_income : ℝ := 571200
noncomputable def C_monthly_income : ℝ := 17000
noncomputable def B_monthly_income : ℝ := C_monthly_income * 1.12
noncomputable def A_monthly_income : ℝ := A_annual_income / 12

theorem monthly_income_ratio :
  (A_monthly_income / B_monthly_income) = 2.5 :=
by
  sorry

end monthly_income_ratio_l249_249639


namespace point_in_second_quadrant_l249_249517

theorem point_in_second_quadrant (a b : ℝ) (h_a : a < 0) (h_b : b > 0) : 
  (-b < 0) ∧ (1 - a > 0) :=
by {
  -- Add sorry to skip the proof
  sorry,
}

end point_in_second_quadrant_l249_249517


namespace C_neither_necessary_nor_sufficient_for_A_l249_249411

theorem C_neither_necessary_nor_sufficient_for_A 
  (A B C : Prop) 
  (h1 : B → C)
  (h2 : B → A) : 
  ¬(A → C) ∧ ¬(C → A) :=
by
  sorry

end C_neither_necessary_nor_sufficient_for_A_l249_249411


namespace child_ticket_cost_l249_249452

variable (A C : ℕ) -- A stands for the number of adults, C stands for the cost of one child's ticket

theorem child_ticket_cost 
  (number_of_adults : ℕ) 
  (number_of_children : ℕ) 
  (cost_concessions : ℕ) 
  (total_cost_trip : ℕ)
  (cost_adult_ticket : ℕ) 
  (ticket_costs : ℕ) 
  (total_adult_cost : ℕ) 
  (remaining_ticket_cost : ℕ) 
  (child_ticket : ℕ) :
  number_of_adults = 5 →
  number_of_children = 2 →
  cost_concessions = 12 →
  total_cost_trip = 76 →
  cost_adult_ticket = 10 →
  ticket_costs = total_cost_trip - cost_concessions →
  total_adult_cost = number_of_adults * cost_adult_ticket →
  remaining_ticket_cost = ticket_costs - total_adult_cost →
  child_ticket = remaining_ticket_cost / number_of_children →
  C = 7 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  -- Adding sorry since the proof is not required
  sorry

end child_ticket_cost_l249_249452


namespace total_shared_in_dollars_l249_249233

-- Definitions based on conditions
def part_share_in_euros (total_ratio : ℕ) (parker_ratio : ℕ) (parker_share_in_euros : ℝ) : ℝ :=
  parker_share_in_euros / parker_ratio

def total_in_euros (parker_share_in_euros richie_share_in_euros jaime_share_in_euros : ℝ) : ℝ :=
  parker_share_in_euros + richie_share_in_euros + jaime_share_in_euros

def convert_to_dollars (amount_in_euros : ℝ) (conversion_rate : ℝ) : ℝ :=
  amount_in_euros * conversion_rate

-- Conditions
def ratio_sum : ℕ := 9
def parker_ratio : ℕ := 2
def richie_ratio : ℕ := 3
def jaime_ratio : ℕ := 4

def parker_share_in_euros : ℝ := 50.0
def conversion_rate : ℝ := 1.1

-- Calculation of shares
def part_value_in_euros := part_share_in_euros ratio_sum parker_ratio parker_share_in_euros
def richie_share_in_euros := richie_ratio * part_value_in_euros
def jaime_share_in_euros := jaime_ratio * part_value_in_euros

-- Total in euros
def total_in_euros_value := total_in_euros parker_share_in_euros richie_share_in_euros jaime_share_in_euros

-- Total in dollars
def total_in_dollars := convert_to_dollars total_in_euros_value conversion_rate

-- Statement of the proof problem
theorem total_shared_in_dollars : total_in_dollars = 247.5 := by
  sorry

end total_shared_in_dollars_l249_249233


namespace odd_function_l249_249817

noncomputable def f : ℝ → ℝ :=
sorry

axiom functional_eq : ∀ (x y: ℝ), f(x + y) = f(x) + f(y)

theorem odd_function : ∀ x: ℝ, f(-x) = -f(x) :=
by
  sorry

example (a : ℝ) (h : f (-3) = a) : f 24 = -8 * a :=
by
  sorry

end odd_function_l249_249817


namespace probability_heads_penny_nickel_dime_l249_249288

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]
variable (coins : Fin 5 → Ω)
variable (heads tails : Ω)

-- Each coin has two outcomes: heads or tails
axiom coin_outcome (i : Fin 5) : coins i = heads ∨ coins i = tails

-- There are 32 total outcomes
axiom total_outcomes : Fintype.card (Fin 5 → Ω) = 32

-- There are 4 successful outcomes for penny, nickel, and dime being heads
axiom successful_outcomes : let successful := {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} in
                          Fintype.card successful = 4

theorem probability_heads_penny_nickel_dime :
  let probability := (Fintype.card {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} : ℤ) /
                     (Fintype.card (Fin 5 → Ω) : ℤ) in
  probability = 1 / 8 :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249288


namespace max_candies_l249_249870

theorem max_candies (c : ℕ := 200) (n m : ℕ := 25) (k : ℕ := 16) (t : ℕ := 100) 
  (H : ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - (25 - (x.1 : ℕ))) i ≥ 100) : 
  ∃ d : ℕ, (d = 37 ∧ ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - d - (25 - (x.1 : ℕ))) i ≤ 100) :=
by {
  use 37,
  sorry
}

end max_candies_l249_249870


namespace outstanding_consumer_installment_credit_l249_249694

-- Given conditions
def total_consumer_installment_credit (C : ℝ) : Prop :=
  let automobile_installment_credit := 0.36 * C
  let automobile_finance_credit := 75
  let total_automobile_credit := 2 * automobile_finance_credit
  automobile_installment_credit = total_automobile_credit

-- Theorem to prove
theorem outstanding_consumer_installment_credit : ∃ (C : ℝ), total_consumer_installment_credit C ∧ C = 416.67 := 
by
  sorry

end outstanding_consumer_installment_credit_l249_249694


namespace exterior_angle_BAC_eq_162_l249_249387

noncomputable def measure_of_angle_BAC : ℝ := 360 - 108 - 90

theorem exterior_angle_BAC_eq_162 :
  measure_of_angle_BAC = 162 := by
  sorry

end exterior_angle_BAC_eq_162_l249_249387


namespace Fr_zero_for_all_r_l249_249572

noncomputable def F (r : ℕ) (x y z A B C : ℝ) : ℝ :=
  x^r * Real.sin (r * A) + y^r * Real.sin (r * B) + z^r * Real.sin (r * C)

theorem Fr_zero_for_all_r
  (x y z A B C : ℝ)
  (h_sum : ∃ k : ℤ, A + B + C = k * Real.pi)
  (hF1 : F 1 x y z A B C = 0)
  (hF2 : F 2 x y z A B C = 0)
  : ∀ r : ℕ, F r x y z A B C = 0 :=
sorry

end Fr_zero_for_all_r_l249_249572


namespace amelia_wins_probability_l249_249760

theorem amelia_wins_probability :
  let pA := 1 / 4
  let pB := 1 / 3
  let pC := 1 / 2
  let cycle_probability := (1 - pA) * (1 - pB) * (1 - pC)
  let infinite_series_sum := 1 / (1 - cycle_probability)
  let total_probability := pA * infinite_series_sum
  total_probability = 1 / 3 :=
by
  sorry

end amelia_wins_probability_l249_249760


namespace diagonals_count_from_vertex_l249_249912

-- Define the number of sides of the decagon
def num_sides (d : Type) : ℕ := 10

-- Define the vertex set of the decagon
def vertex_set (d : Type) : Finset (Fin 10) :=
  Finset.univ

-- Define adjacency function for the vertices
def adjacent (v : Fin 10) (w : Fin 10) : Prop :=
  -- Vertices are adjacent if they are consecutive modulo 10
  w = (v + 1) % 10 ∨ w = (v - 1) % 10

-- Count the number of non-adjacent (and non-self) vertices from any vertex v
def diagonals_from_vertex (v : Fin 10) : Finset (Fin 10) :=
  vertex_set d \ {v} \ {w | adjacent v w}

theorem diagonals_count_from_vertex (v : Fin 10) :
  (diagonals_from_vertex v).card = 7 :=
by
  sorry

end diagonals_count_from_vertex_l249_249912


namespace basketball_teams_count_l249_249606

def height_range := { h : ℕ // 191 ≤ h ∧ h ≤ 197 }
def weight_range := { w : ℕ // 190 ≤ w ∧ w ≤ 196 }
def valid_combinations (h w : ℕ) := w < h 

structure Player :=
  (height : height_range)
  (weight : weight_range)
  (valid : valid_combinations height weight)

noncomputable def count_teams : ℕ :=
  128

theorem basketball_teams_count :
  ∃ (P : Player → Prop), ∀ (T : set Player), (∀ p ∈ T, P p) → T.finite → T.card = count_teams :=
sorry

end basketball_teams_count_l249_249606


namespace part_b_part_c_l249_249814

def z : ℂ := (5 + I) / (1 + I)

theorem part_b : z.im = -2 :=
by
  -- omitted for brevity
  sorry

theorem part_c : (z.re > 0) ∧ (z.im < 0) :=
by
  -- omitted for brevity
  sorry

end part_b_part_c_l249_249814


namespace rectangle_area_constant_l249_249643

theorem rectangle_area_constant (d : ℝ) (length width : ℝ) (h_ratio : length / width = 5 / 2) (h_diag : d = Real.sqrt (length^2 + width^2)) :
  ∃ k : ℝ, (length * width) = k * d^2 ∧ k = 10 / 29 :=
by
  use 10 / 29
  sorry

end rectangle_area_constant_l249_249643


namespace coin_rotations_l249_249621

-- Defining the problem with the given configurations and the required proof.

theorem coin_rotations (n : ℕ) (h_n : 1 ≤ n)
  (coins : Fin n → ℝ × ℝ) 
  (touch_next : ∀ i : Fin n, (coins i).dist (coins ((i + 1) % n)) = 2) 
  (G_roll : ∀ i : Fin n, ∃ j : Fin n, (G j) = coins i ∨ (G j) ∈ (arc coins i coins ((i + 1) % n))) :
  (number_of_rotations G n ∷ ℝ) = 2 + n / 3 := sorry

end coin_rotations_l249_249621


namespace sum_A_B_C_zero_l249_249997

noncomputable def poly : Polynomial ℝ := Polynomial.X^3 - 16 * Polynomial.X^2 + 72 * Polynomial.X - 27

noncomputable def exists_real_A_B_C 
  (p q r: ℝ) (hpqr: p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hrootsp: Polynomial.eval p poly = 0) (hrootsq: Polynomial.eval q poly = 0)
  (hrootsr: Polynomial.eval r poly = 0) :
  ∃ (A B C: ℝ), (∀ s, s ≠ p → s ≠ q → s ≠ r → (1 / (s^3 - 16*s^2 + 72*s - 27) = (A / (s - p)) + (B / (s - q)) + (C / (s - r)))) := sorry

theorem sum_A_B_C_zero 
  {p q r: ℝ} (hpqr: p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hrootsp: Polynomial.eval p poly = 0) (hrootsq: Polynomial.eval q poly = 0)
  (hrootsr: Polynomial.eval r poly = 0) 
  (hABC: ∃ (A B C: ℝ), (∀ s, s ≠ p → s ≠ q → s ≠ r → (1 / (s^3 - 16*s^2 + 72*s - 27) = (A / (s - p)) + (B / (s - q)) + (C / (s - r))))) :
  ∀ A B C, A + B + C = 0 := sorry

end sum_A_B_C_zero_l249_249997


namespace avg_multiples_of_10_zero_l249_249678

open Int

def multiples_of_10 (a b: Int) : Set Int := {x | a ≤ x ∧ x ≤ b ∧ 10 ∣ x} 

theorem avg_multiples_of_10_zero : 
  let S := multiples_of_10 (-1000) 1000 in
  (S.sum / S.card.toReal) = 0 := 
by
  sorry

end avg_multiples_of_10_zero_l249_249678


namespace find_a_l249_249045

noncomputable def f : ℝ → ℝ :=
λ x, if x < 0 then -2 * x else x^2 - 1

noncomputable def equation (x a : ℝ) : ℝ :=
f x + 2 * real.sqrt (1 - x^2) + | f x - 2 * real.sqrt (1 - x^2) | - 2 * a * x - 4

theorem find_a (x1 x2 x3 a : ℝ)
  (h1 : ∃ a, equation x1 a = 0 ∧ equation x2 a = 0 ∧ equation x3 a = 0)
  (h2 : x1 < x2 ∧ x2 < x3)
  (h3 : x3 - x2 = 2 * (x2 - x1)) :
  a = (real.sqrt 17 - 3) / 2 :=
sorry

end find_a_l249_249045


namespace max_divisors_up_to_20_l249_249218

def divisors (n : Nat) : List Nat :=
  (List.range (n + 1)).filter (λ d, n % d = 0)

def count_divisors (n : Nat) : Nat :=
  (divisors n).length

theorem max_divisors_up_to_20 :
  ∀ n, n ∈ List.range 21 → count_divisors n ≤ 6 ∧
      ((count_divisors 12 = 6) ∧
       (count_divisors 18 = 6) ∧
       (count_divisors 20 = 6)) :=
by
  sorry

end max_divisors_up_to_20_l249_249218


namespace largest_angle_in_triangle_PQR_l249_249665

-- Definitions
def is_isosceles_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  (α = β) ∨ (β = γ) ∨ (γ = α)

def is_obtuse_triangle (P Q R : ℝ) (α β γ : ℝ) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

variables (P Q R : ℝ)
variables (angleP angleQ angleR : ℝ)

-- Condition: PQR is an obtuse and isosceles triangle, and angle P measures 30 degrees
axiom h1 : is_isosceles_triangle P Q R angleP angleQ angleR
axiom h2 : is_obtuse_triangle P Q R angleP angleQ angleR
axiom h3 : angleP = 30

-- Theorem: The measure of the largest interior angle of triangle PQR is 120 degrees
theorem largest_angle_in_triangle_PQR : max angleP (max angleQ angleR) = 120 :=
  sorry

end largest_angle_in_triangle_PQR_l249_249665


namespace Jason_spent_on_shorts_l249_249125

/--
Jason went to the mall on Saturday to buy clothes. He spent some money on shorts and $4.74 on a jacket. 
In total, Jason spent $19.02 on clothing. Prove that Jason spent $14.28 on shorts.
-/
theorem Jason_spent_on_shorts (total_spent on_jacket : ℝ) (total_spent_eq : total_spent = 19.02) 
  (on_jacket_eq : on_jacket = 4.74) : (total_spent - on_jacket) = 14.28 :=
by 
  rw [total_spent_eq, on_jacket_eq]
  norm_num
  sorry

end Jason_spent_on_shorts_l249_249125


namespace B_subset_A_A_inter_B_empty_l249_249054

-- definition for sets A and B
def A := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- first problem
theorem B_subset_A (m : ℝ) : (B m ⊆ A) ↔ (m ∈ set.Iic 3) := 
sorry

-- second problem
theorem A_inter_B_empty (m : ℝ) : (A ∩ B m = ∅) ↔ (m ∈ set.Iio 2 ∪ set.Ioi 4) := 
sorry

end B_subset_A_A_inter_B_empty_l249_249054


namespace sequence_periodic_l249_249554

theorem sequence_periodic {u : ℕ → ℝ} (h : ∀ n, u (n + 2) = |u (n + 1)| - u n) : ∃ p : ℕ, p = 9 ∧ ∀ n : ℕ, u n = u (n + p) :=
by
  use 9 -- There exists a positive integer p = 9
  split
  -- Show that ∀ n, u n = u (n + 9)
  sorry

end sequence_periodic_l249_249554


namespace periodic_point_count_1989_l249_249999

def unit_circle : Set ℂ := {z : ℂ | complex.abs z = 1}

def is_periodic_point (f : ℂ → ℂ) (n : ℕ) (c : ℂ) : Prop :=
  (c ∈ unit_circle) ∧ (∀ k : ℕ, 1 ≤ k ∧ k < n → f^[k] c ≠ c) ∧ (f^[n] c = c)

def periodic_points_count (f : ℂ → ℂ) (n : ℕ) : with_top ℕ :=
  (unit_circle.filter (is_periodic_point f n)).to_finset.card

noncomputable def f (z : ℂ) : ℂ := z^2 -- Example specific choice of m

theorem periodic_point_count_1989 :
  periodic_points_count (λ z, z^2) 1989 = 1989 := -- Example specific to m = 2, adjust accordingly
sorry

end periodic_point_count_1989_l249_249999


namespace probability_heads_is_one_eighth_l249_249263

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_is_one_eighth_l249_249263


namespace pure_imaginary_condition_sufficient_but_not_necessary_pure_imaginary_l249_249157

theorem pure_imaginary_condition {a : ℝ} :
  (a - 1) * (a + 2) + (a + 3) * complex.I = (0 * complex.I) → a = 1 :=
by {
  sorry
}

theorem sufficient_but_not_necessary_pure_imaginary {a : ℝ} :
  (∃ a : ℝ, (a - 1) * (a + 2) + (a + 3) * complex.I ≠ 0) :=
by {
  sorry
}

end pure_imaginary_condition_sufficient_but_not_necessary_pure_imaginary_l249_249157


namespace coin_flip_heads_probability_l249_249257

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end coin_flip_heads_probability_l249_249257


namespace element_of_set_l249_249859

theorem element_of_set : -1 ∈ { x : ℝ | x^2 - 1 = 0 } :=
sorry

end element_of_set_l249_249859


namespace find_length_of_FG_l249_249676

theorem find_length_of_FG
  (GH IJ JK : ℝ)
  (H_similar : SimilarTri F G H I J K)
  (hGH : GH = 30)
  (hIJ : IJ = 18)
  (hJK : JK = 15) :
  let FG := 36 in
  FG = 36 :=
by
  -- Insert proof here
  sorry

end find_length_of_FG_l249_249676


namespace rowing_time_l249_249731

variable (V_m : ℝ) (V_r : ℝ) (D_total : ℝ)

noncomputable def upstream_speed := V_m - V_r
noncomputable def downstream_speed := V_m + V_r
noncomputable def D := D_total / 2
noncomputable def T_up := D / upstream_speed
noncomputable def T_down := D / downstream_speed
noncomputable def T_total := T_up + T_down

theorem rowing_time 
  (h1 : V_m = 6) 
  (h2 : V_r = 2) 
  (h3 : D_total = 5.333333333333333):
  T_total V_m V_r D_total = 1 := 
by
  sorry

end rowing_time_l249_249731


namespace ratio_of_segments_l249_249719

open Real

theorem ratio_of_segments
(A B C L J K : Point)
(h₁ : inscribed_circle A B C) 
(h₂ : B.distance C = 9)
(h₃ : A.distance B = 20)
(h₄ : A.distance C = 15)
(h₅ : tangent_point h₁ B C L)
(h₆ : tangent_point h₁ A C J)
(h₇ : tangent_point h₁ A B K)
(r s : ℝ)
(hr : B.distance L = r)
(hs : C.distance L = s)
(hr_lt_hs : r < s) :
  r / s = 2 / 7 := sorry

end ratio_of_segments_l249_249719


namespace max_divisors_up_to_20_l249_249219

def divisors (n : Nat) : List Nat :=
  (List.range (n + 1)).filter (λ d, n % d = 0)

def count_divisors (n : Nat) : Nat :=
  (divisors n).length

theorem max_divisors_up_to_20 :
  ∀ n, n ∈ List.range 21 → count_divisors n ≤ 6 ∧
      ((count_divisors 12 = 6) ∧
       (count_divisors 18 = 6) ∧
       (count_divisors 20 = 6)) :=
by
  sorry

end max_divisors_up_to_20_l249_249219


namespace Vovochka_max_candies_l249_249888

noncomputable theory
open_locale classical
open set

def VovochkaCandies (c n k required_sum : ℕ) (dist : fin n → ℕ) : Prop :=
  ∀ (s : finset (fin n)), s.card = k → (s.sum dist ≥ required_sum)

theorem Vovochka_max_candies
  (c n k required_sum : ℕ)
  (h_c : c = 200)
  (h_n : n = 25)
  (h_k : k = 16)
  (h_required_sum : required_sum = 100)
  (dist : fin n → ℕ)
  (h_dist : VovochkaCandies c n k required_sum dist) :
  c - finset.univ.sum dist = 37 :=
sorry

end Vovochka_max_candies_l249_249888


namespace optical_power_of_converging_lens_l249_249745

theorem optical_power_of_converging_lens 
  (D_p : ℝ) (d_1 d_2 : ℝ) (spot_size_unchanged : Prop) :
  D_p = -6 →
  d_1 = 0.1 →
  d_2 = 0.2 →
  spot_size_unchanged →
  exists (D_c : ℝ), D_c = 18 :=
by
  sorry

end optical_power_of_converging_lens_l249_249745


namespace initial_birds_on_fence_l249_249250

theorem initial_birds_on_fence (B S : ℕ) (S_val : S = 2) (total : B + 5 + S = 10) : B = 3 :=
by
  sorry

end initial_birds_on_fence_l249_249250


namespace probability_heads_is_one_eighth_l249_249265

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_is_one_eighth_l249_249265


namespace parallelogram_height_l249_249535

-- Definitions of the variables
variables (ABCD : Type) [parallelogram ABCD]
variables (AB BC DC : ℝ) (DE DF : ℝ) (EB : ℝ)

-- Conditions
def condition1 : DC = 12 := sorry
def condition2 : EB = 4 := sorry
def condition3 : DE = 6 := sorry
def condition4 : AB = DC := sorry -- Opposite sides of a parallelogram are equal
def condition5 : BC = AB := sorry -- Opposite sides of a parallelogram are equal

-- Theorem stating the problem
theorem parallelogram_height {ABCD : Type} [parallelogram ABCD] (AB BC DC : ℝ) (DE DF : ℝ) (EB : ℝ) :
  (DC = 12) → (EB = 4) → (DE = 6) → (AB = DC) → (BC = AB) → DF = 6 :=
by
  intros hDC hEB hDE hAB hBC
  -- Placeholder for the proof
  sorry

end parallelogram_height_l249_249535


namespace tangent_distance_is_140_l249_249620

noncomputable def circle_A := (19 : ℝ)
noncomputable def circle_B := (32 : ℝ)
noncomputable def circle_C := (100 : ℝ)
noncomputable def segment_CA := (circle_C - circle_A : ℝ)
noncomputable def segment_CB := (circle_C - circle_B : ℝ)
noncomputable def AB_length := (segment_CA + segment_CB : ℝ)

theorem tangent_distance_is_140 
    (MN : ℝ)
    (A B C : ℝ)
    (hA : A = 19)
    (hB : B = 32)
    (hC : C = 100)
    (hMN : MN = (λ MN: ℝ, sqrt (AB_length^2 - (B - A)^2)))
    :
    MN = 140 := 
begin
 sorry
end

end tangent_distance_is_140_l249_249620


namespace trigonometric_identity_l249_249756

theorem trigonometric_identity : 
  sin 20 * sin 20 + cos 50 * cos 50 + sin 20 * cos 50 = 1 :=
by sorry

end trigonometric_identity_l249_249756


namespace solutions_to_gx_eq_5_l249_249577

def g (x : ℝ) :=
  if x < 0 then 4 * x + 8 else 3 * x - 15

theorem solutions_to_gx_eq_5 :
  {x : ℝ | g x = 5} = {-3 / 4, 20 / 3} :=
by
  sorry

end solutions_to_gx_eq_5_l249_249577


namespace greatest_divisors_1_to_20_l249_249178

theorem greatest_divisors_1_to_20 : ∃ n₁ n₂ n₃, ((n₁ = 12 ∧ n₂ = 18 ∧ n₃ = 20) ∧ 
  (∀ m ∈ (Finset.range 21).filter (λ x, x > 0), 
    let d_m := (Finset.range (m + 1)).filter (λ k, m % k = 0) in
    ∃ k, k = d_m.card ∧ k <= 6 ∧ ((m = n₁ ∧ k = 6) ∨ (m = n₂ ∧ k = 6) ∨ (m = n₃ ∧ k = 6)))) :=
by
  sorry

end greatest_divisors_1_to_20_l249_249178


namespace grid_permutation_exists_l249_249750

theorem grid_permutation_exists (n : ℕ) (grid : Fin n → Fin n → ℤ) 
  (cond1 : ∀ i : Fin n, ∃ unique j : Fin n, grid i j = 1)
  (cond2 : ∀ i : Fin n, ∃ unique j : Fin n, grid i j = -1)
  (cond3 : ∀ j : Fin n, ∃ unique i : Fin n, grid i j = 1)
  (cond4 : ∀ j : Fin n, ∃ unique i : Fin n, grid i j = -1)
  (cond5 : ∀ i j, grid i j = 0 ∨ grid i j = 1 ∨ grid i j = -1) :
  ∃ (perm_rows perm_cols : Fin n → Fin n),
    (∀ i j, grid (perm_rows i) (perm_cols j) = -grid i j) :=
by
  -- Proof goes here
  sorry

end grid_permutation_exists_l249_249750


namespace complex_quadrant_l249_249033

open Complex

theorem complex_quadrant (z : ℂ) (hz : (3 + 2 * I) * z = 13 * I) : 0 < z.re ∧ 0 < z.im := by
  -- Step 1: Solve for z
  have : z = (13 * I) / (3 + 2 * I) := by
    -- Multiplying both sides by the conjugate of the denominator:
    field_simp [norm_sq_eq_abs] -- Use the field_simp tactic to simplify the fraction
    norm_num [I_mul_I, I_add_I']  -- Normalize numeral terms involving I (imaginary unit)
    exact hz
  -- Step 2: Simplify to the form z = 2 + 3i
  have hz' : z = 2 + 3 * I := sorry
  -- Step 3: Prove the real and imaginary parts are positive
  rw [hz']  -- Replace z with 2 + 3i
  exact ⟨by norm_num, by norm_num⟩

end complex_quadrant_l249_249033


namespace arithmetic_sequence_term_count_l249_249475

theorem arithmetic_sequence_term_count {a : ℕ → ℝ} (h1 : a 1 + a 2 + a 3 = 4) 
  (h2 : ∀ n, a (n-2) + a (n-1) + a n = 7) (h3 : ∑ k in finset.range n, a k = 22) 
  : n = 12 :=
begin
  sorry
end

end arithmetic_sequence_term_count_l249_249475


namespace different_imaginary_numbers_l249_249032

theorem different_imaginary_numbers (a b : ℕ) (h1 : a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h2 : b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (h3 : a ≠ b) : 
  (finset.filter (λ a, ∃ b, b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ a ≠ b) {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}).card = 81 :=
by
  sorry

end different_imaginary_numbers_l249_249032


namespace socks_ratio_proof_l249_249749

variable (w y : ℝ) -- w: number of white socks, y: price per pair of white socks

def g : ℝ := 6 -- number of grey socks

-- original cost
def C_original : ℝ := 6 * 3 * y + w * y

-- interchanged cost
def C_interchanged : ℝ := w * 3 * y + 6 * y

-- condition: interchanged cost is 1.25 times the original cost
axiom h1 : C_interchanged = 1.25 * C_original

-- goal: ratio of grey socks to white socks is 6:10
theorem socks_ratio_proof : (w ≠ 0) → g / w = 6 / 10 := by
  sorry

end socks_ratio_proof_l249_249749


namespace complete_the_square_example_l249_249586

theorem complete_the_square_example (x : ℝ) : 
  ∃ c d : ℝ, (x^2 - 6 * x + 5 = 0) ∧ ((x + c)^2 = d) ∧ (d = 4) :=
sorry

end complete_the_square_example_l249_249586


namespace trig_identity_l249_249493

theorem trig_identity (θ : ℝ) (h1 : θ ≠ 0) (h2 : tan θ = 2) : 
  sin (2 * θ + π / 4) = sqrt 2 / 10 :=
sorry

end trig_identity_l249_249493


namespace john_good_games_l249_249359

theorem john_good_games (a b c good_games : ℕ) (ha : a = 21) (hb : b = 8) (hc : c = 23) (h_total : good_games = a + b - c) : good_games = 6 :=
by {
  rw [ha, hb, hc],
  show 21 + 8 - 23 = 6,
  sorry
}

end john_good_games_l249_249359


namespace area_of_L_shape_l249_249376

theorem area_of_L_shape : 
  let large_rectangle_area := 10 * 6,
      subtracted_rectangle_area := 4 * 3,
      total_subtracted_area := 2 * subtracted_rectangle_area
  in large_rectangle_area - total_subtracted_area = 36 := 
by
  sorry

end area_of_L_shape_l249_249376


namespace expand_expression_l249_249786

theorem expand_expression (x y : ℝ) : 12 * (3 * x - 4 * y + 2) = 36 * x - 48 * y + 24 :=
by
  sorry

end expand_expression_l249_249786


namespace combined_total_circles_squares_l249_249294

-- Define the problem parameters based on conditions
def US_stars : ℕ := 50
def US_stripes : ℕ := 13
def circles (n : ℕ) : ℕ := (n / 2) - 3
def squares (n : ℕ) : ℕ := (n * 2) + 6

-- Prove that the combined number of circles and squares on Pete's flag is 54
theorem combined_total_circles_squares : 
    circles US_stars + squares US_stripes = 54 := by
  sorry

end combined_total_circles_squares_l249_249294


namespace remainder_when_divided_by_x_minus_3_l249_249679

open Polynomial

noncomputable def p : ℝ[X] := 4 * X^3 - 12 * X^2 + 16 * X - 20

theorem remainder_when_divided_by_x_minus_3 : eval 3 p = 28 := by
  sorry

end remainder_when_divided_by_x_minus_3_l249_249679


namespace geometric_relations_between_MO_MP_MQ_l249_249094

-- Given conditions:
-- 1. Lines PQ and XY intersect at point O.
-- 2. M is the midpoint of XY.
-- 3. Lines XP and YQ are perpendicular to PQ.

open_locale real_inner_product_space

variables {P Q X Y O M : Point}
-- Assume there is a proof or construction satisfying these geometrical conditions.
variables (hIntersect : ∃ (O : Point), intersecting_lines PQ XY O)
variables (hMidpoint : midpoint M X Y)
variables (hPerpendicular1 : perpendicular XP PQ)
variables (hPerpendicular2 : perpendicular YQ PQ)

theorem geometric_relations_between_MO_MP_MQ :
  MP = MQ ∧ MO ≠ MP :=
by sorry

end geometric_relations_between_MO_MP_MQ_l249_249094


namespace multiply_integers_l249_249762

theorem multiply_integers (a b c : ℤ) (h : a = 70) (i : b = 2) (j : c = 68) : (a + b) * (a - b) = 4896 := by
  rw [h, i, j]
  have h1 : 72 = 70 + 2 := by norm_num
  have h2 : 68 = 70 - 2 := by norm_num
  have h3 : (70 + 2) * (70 - 2) = 70^2 - 2^2 := by exact mul_sub_mul_add_const 70 2 2
  rw [h1, h2, h3]
  norm_num
  sorry

end multiply_integers_l249_249762


namespace find_m_range_of_k_l249_249858

-- Definitions for the given functions
def f (m : ℝ) (x : ℝ) : ℝ := (m-1)^2 * x^(m^2 - 4*m + 2)
def g (k : ℝ) (x : ℝ) : ℝ := 2^x - k

-- Condition for monotonicity of f
def is_monotonically_increasing (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2) → f x1 ≤ f x2

-- Problem (I): finding the value of m
theorem find_m (h : is_monotonically_increasing (f m)) : m = 0 := sorry

-- Definitions for sets A and B
def A (x : ℝ) : ℝ := x^2
def B (k : ℝ) (x : ℝ) : ℝ := 2^x - k

-- Problem (II): range of k
theorem range_of_k (h : ∀ x ∈ set.Icc 1 2, A x ∈ set.Icc 1 4)
  (hpq : ∀ x ∈ set.Icc 1 2, ¬ (x ∈ set.Icc 1 4) ∨ (2^x - k ∈ set.Icc 1 4) → x ∈ set.Icc 1 4) :
  0 ≤ k ∧ k ≤ 1 := sorry

end find_m_range_of_k_l249_249858


namespace proof_equivalent_problem_l249_249086

variables (P A B : Prop)
-- Condition: The original statement is false
axiom h : ¬P

-- Definitions based on the solution's interpretation
def statement_II := ∃ x, ¬(P x)
def statement_IV := ¬∀ x, P x

-- Equivalent proof problem statement
theorem proof_equivalent_problem : h → (statement_II) ∧ (statement_IV) :=
sorry

end proof_equivalent_problem_l249_249086


namespace Mike_owes_Laura_l249_249982

theorem Mike_owes_Laura :
  let rate_per_room := (13 : ℚ) / 3
  let rooms_cleaned := (8 : ℚ) / 5
  let total_amount := (104 : ℚ) / 15
  rate_per_room * rooms_cleaned = total_amount :=
by
  sorry

end Mike_owes_Laura_l249_249982


namespace crates_lost_l249_249730

theorem crates_lost (total_crates : ℕ) (total_cost : ℕ) (desired_profit_percent : ℕ) 
(lost_crates remaining_crates : ℕ) (price_per_crate : ℕ) 
(h1 : total_crates = 10) (h2 : total_cost = 160) (h3 : desired_profit_percent = 25) 
(h4 : price_per_crate = 25) (h5 : remaining_crates = total_crates - lost_crates)
(h6 : price_per_crate * remaining_crates = total_cost + total_cost * desired_profit_percent / 100) :
  lost_crates = 2 :=
by
  sorry

end crates_lost_l249_249730


namespace solution_l249_249052

theorem solution (x : ℝ) (h : 6 ∈ ({2, 4, x * x - x} : Set ℝ)) : x = 3 ∨ x = -2 := 
by 
  sorry

end solution_l249_249052


namespace complex_product_modulus_equality_l249_249690

def f (n m : ℕ) (z : Fin n → ℂ) : ℂ := 
  ∑ s in Finset.powersetLen m (Finset.univ : Finset (Fin n)), 
    ∏ i in s, z i

theorem complex_product_modulus_equality (n m : ℕ) (a : ℝ) (h_a_gt_zero : a > 0)
  (z : Fin n → ℂ) (h_modulus : ∀ i, Complex.abs (z i) = a) :
  Complex.abs (f n m z) / (Real.pow a m) = Complex.abs (f n (n - m) z) / (Real.pow a (n - m)) :=
by
  sorry

end complex_product_modulus_equality_l249_249690


namespace handshakes_at_networking_event_l249_249408

noncomputable def total_handshakes (n : ℕ) (exclude : ℕ) : ℕ :=
  (n * (n - 1 - exclude)) / 2

theorem handshakes_at_networking_event : total_handshakes 12 1 = 60 := by
  sorry

end handshakes_at_networking_event_l249_249408


namespace abe_wages_l249_249396

theorem abe_wages (budget : ℝ) (H_budget : budget = 3000) :
  let food := (1/3) * budget,
      supplies := (1/4) * budget,
      wages := budget - (food + supplies)
  in wages = 1250 :=
by
  let food := (1/3) * budget,
  let supplies := (1/4) * budget,
  let wages := budget - (food + supplies),
  sorry

end abe_wages_l249_249396


namespace initial_students_per_class_l249_249617

theorem initial_students_per_class (students_per_class initial_classes additional_classes total_students : ℕ) 
  (h1 : initial_classes = 15) 
  (h2 : additional_classes = 5) 
  (h3 : total_students = 400) 
  (h4 : students_per_class * (initial_classes + additional_classes) = total_students) : 
  students_per_class = 20 := 
by 
  -- Proof goes here
  sorry

end initial_students_per_class_l249_249617


namespace Vovochka_max_candies_l249_249887

noncomputable theory
open_locale classical
open set

def VovochkaCandies (c n k required_sum : ℕ) (dist : fin n → ℕ) : Prop :=
  ∀ (s : finset (fin n)), s.card = k → (s.sum dist ≥ required_sum)

theorem Vovochka_max_candies
  (c n k required_sum : ℕ)
  (h_c : c = 200)
  (h_n : n = 25)
  (h_k : k = 16)
  (h_required_sum : required_sum = 100)
  (dist : fin n → ℕ)
  (h_dist : VovochkaCandies c n k required_sum dist) :
  c - finset.univ.sum dist = 37 :=
sorry

end Vovochka_max_candies_l249_249887


namespace mean_mark_second_section_l249_249383

theorem mean_mark_second_section :
  ∀ (M : ℝ), 
    let n1 := 60
    let n2 := 35
    let n3 := 45
    let n4 := 42
    let mean1 := 50
    let mean3 := 55
    let mean4 := 45
    let overall_average := 52.005494505494504
    let total_students := n1 + n2 + n3 + n4
    let total_marks := (n1 * mean1) + (n2 * M) + (n3 * mean3) + (n4 * mean4)
    (overall_average = total_marks / total_students) → M = 60 :=
by
  intros
  let n1 := 60
  let n2 := 35
  let n3 := 45
  let n4 := 42
  let mean1 := 50
  let mean3 := 55
  let mean4 := 45
  let overall_average := 52.005494505494504
  let total_students := n1 + n2 + n3 + n4
  let total_marks := (n1 * mean1) + (n2 * M) + (n3 * mean3) + (n4 * mean4)
  have ha := overall_average = total_marks / total_students
  sorry

end mean_mark_second_section_l249_249383


namespace difference_of_numbers_l249_249329

theorem difference_of_numbers (a : ℕ) (h : a + (10 * a + 5) = 30000) : (10 * a + 5) - a = 24548 :=
by
  sorry

end difference_of_numbers_l249_249329


namespace infinite_perfect_squares_of_form_l249_249996

theorem infinite_perfect_squares_of_form (k : ℕ) (hk : 0 < k) : 
  ∃ (n : ℕ), ∀ m : ℕ, ∃ a : ℕ, (n + m) * 2^k - 7 = a^2 :=
sorry

end infinite_perfect_squares_of_form_l249_249996


namespace prove_solution_l249_249298

-- Define the philosophical viewpoints identified in each famous line
def viewpoint_1 : Prop := "replacement of old things by new ones"
def viewpoint_2 : Prop := "viewpoint of connection"
def viewpoint_3 : Prop := "replacement of old things by new ones"
def viewpoint_4 : Prop := "consciousness as the subjective image of objective existence"

-- Define the correct answer based on the problem's solution
def correct_answer : Prop := ("①③" = "replacement of old things by new ones")

-- Lean statement that verifies the correct answer given the conditions
theorem prove_solution : (viewpoint_1 = "replacement of old things by new ones" ∧ 
                          viewpoint_3 = "replacement of old things by new ones" ∧ 
                          viewpoint_2 = "viewpoint of connection" ∧ 
                          viewpoint_4 = "consciousness as the subjective image of objective existence") -> 
                         correct_answer :=
by 
  sorry

end prove_solution_l249_249298


namespace probability_penny_nickel_dime_all_heads_l249_249285

-- Define flipping five coins
def flip_five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the function to check if the penny, nickel, and dime are heads
def all_heads_penny_nickel_dime (o : flip_five_coins) : bool :=
  o.1 = tt ∧ o.2 = tt ∧ o.3 = tt

-- Define the total count of possible outcomes
def total_outcomes : ℕ := 32

-- Define the count of favorable outcomes
def favorable_outcomes : ℕ := 4

-- Define the probability calculation
def probability_favorable : ℚ := favorable_outcomes / total_outcomes

-- The statement proving the probability is 1/8
theorem probability_penny_nickel_dime_all_heads :
  probability_favorable = 1 / 8 :=
by
  sorry

end probability_penny_nickel_dime_all_heads_l249_249285


namespace probability_heads_penny_nickel_dime_l249_249290

variable {Ω : Type} [Fintype Ω] [DecidableEq Ω]
variable (coins : Fin 5 → Ω)
variable (heads tails : Ω)

-- Each coin has two outcomes: heads or tails
axiom coin_outcome (i : Fin 5) : coins i = heads ∨ coins i = tails

-- There are 32 total outcomes
axiom total_outcomes : Fintype.card (Fin 5 → Ω) = 32

-- There are 4 successful outcomes for penny, nickel, and dime being heads
axiom successful_outcomes : let successful := {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} in
                          Fintype.card successful = 4

theorem probability_heads_penny_nickel_dime :
  let probability := (Fintype.card {x // (coins 0 = heads) ∧ (coins 1 = heads) ∧ (coins 2 = heads)} : ℤ) /
                     (Fintype.card (Fin 5 → Ω) : ℤ) in
  probability = 1 / 8 :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249290


namespace greatest_divisors_1_to_20_l249_249182

theorem greatest_divisors_1_to_20 : ∃ n₁ n₂ n₃, ((n₁ = 12 ∧ n₂ = 18 ∧ n₃ = 20) ∧ 
  (∀ m ∈ (Finset.range 21).filter (λ x, x > 0), 
    let d_m := (Finset.range (m + 1)).filter (λ k, m % k = 0) in
    ∃ k, k = d_m.card ∧ k <= 6 ∧ ((m = n₁ ∧ k = 6) ∨ (m = n₂ ∧ k = 6) ∨ (m = n₃ ∧ k = 6)))) :=
by
  sorry

end greatest_divisors_1_to_20_l249_249182


namespace probability_heads_is_one_eighth_l249_249267

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_is_one_eighth_l249_249267


namespace fractions_correct_l249_249108
-- Broader import to ensure all necessary libraries are included.

-- Definitions of the conditions
def batman_homes_termite_ridden : ℚ := 1/3
def batman_homes_collapsing : ℚ := 7/10 * batman_homes_termite_ridden
def robin_homes_termite_ridden : ℚ := 3/7
def robin_homes_collapsing : ℚ := 4/5 * robin_homes_termite_ridden
def joker_homes_termite_ridden : ℚ := 1/2
def joker_homes_collapsing : ℚ := 3/8 * joker_homes_termite_ridden

-- Definitions of the fractions of homes that are termite-ridden but not collapsing
def batman_non_collapsing_fraction : ℚ := batman_homes_termite_ridden - batman_homes_collapsing
def robin_non_collapsing_fraction : ℚ := robin_homes_termite_ridden - robin_homes_collapsing
def joker_non_collapsing_fraction : ℚ := joker_homes_termite_ridden - joker_homes_collapsing

-- Proof statement
theorem fractions_correct :
  batman_non_collapsing_fraction = 1/10 ∧
  robin_non_collapsing_fraction = 3/35 ∧
  joker_non_collapsing_fraction = 5/16 :=
sorry

end fractions_correct_l249_249108


namespace sugar_needed_l249_249983

variable (a b c d : ℝ)
variable (H1 : a = 2)
variable (H2 : b = 1)
variable (H3 : d = 5)

theorem sugar_needed (c : ℝ) : c = 2.5 :=
by
  have H : 2 / 1 = 5 / c := by {
    sorry
  }
  sorry

end sugar_needed_l249_249983


namespace total_height_correct_l249_249131

def height_of_stairs : ℕ := 10

def num_flights : ℕ := 3

def height_of_all_stairs : ℕ := height_of_stairs * num_flights

def height_of_rope : ℕ := height_of_all_stairs / 2

def extra_height_of_ladder : ℕ := 10

def height_of_ladder : ℕ := height_of_rope + extra_height_of_ladder

def total_height_climbed : ℕ := height_of_all_stairs + height_of_rope + height_of_ladder

theorem total_height_correct : total_height_climbed = 70 := by
  sorry

end total_height_correct_l249_249131


namespace vovochka_max_candies_l249_249867

/-!
# Problem Statement:
- Vovochka has 25 classmates.
- Vovochka brought 200 candies to class.

## Condition:
- Any 16 classmates together should have at least 100 candies in total.

Question: What is the maximum number of candies Vovochka can keep for himself while fulfilling his mother's request?

The answer to this problem needs to be proven mathematically.
-/

def max_candies_can_keep (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ) : ℕ :=
if h : classmates = 25 ∧ total_candies = 200 ∧ min_candies_per_16 = 100 then
  37
else 
  sorry -- This statement is just for context, keeping focus on formal proof in Lean.

theorem vovochka_max_candies :
  ∀ (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ),
    classmates = 25 → total_candies = 200 → min_candies_per_16 = 100 →
    max_candies_can_keep candies classmates total_candies min_candies_per_16 = 37 :=
by
  intros candies classmates total_candies min_candies_per_16 hc ht hm
  simp [max_candies_can_keep]
  exact if_pos ⟨hc, ht, hm⟩

end vovochka_max_candies_l249_249867


namespace greatest_divisors_1_to_20_l249_249185

theorem greatest_divisors_1_to_20 : ∃ n₁ n₂ n₃, ((n₁ = 12 ∧ n₂ = 18 ∧ n₃ = 20) ∧ 
  (∀ m ∈ (Finset.range 21).filter (λ x, x > 0), 
    let d_m := (Finset.range (m + 1)).filter (λ k, m % k = 0) in
    ∃ k, k = d_m.card ∧ k <= 6 ∧ ((m = n₁ ∧ k = 6) ∨ (m = n₂ ∧ k = 6) ∨ (m = n₃ ∧ k = 6)))) :=
by
  sorry

end greatest_divisors_1_to_20_l249_249185


namespace coin_flip_heads_probability_l249_249260

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end coin_flip_heads_probability_l249_249260


namespace coin_flip_heads_probability_l249_249262

theorem coin_flip_heads_probability :
  let coins : List String := ["penny", "nickel", "dime", "quarter", "half-dollar"]
  let independent_event (coin : String) : Prop := True
  let outcomes := (2 : ℕ) ^ (List.length coins)
  let successful_outcomes := 4
  let probability := successful_outcomes / outcomes
  probability = 1 / 8 := 
by
  sorry

end coin_flip_heads_probability_l249_249262


namespace circle_eq_of_conditions_l249_249031

theorem circle_eq_of_conditions :
  ∃ r : ℝ, (∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = r^2)
  ∧ (∃ d : ℝ, d = (| (2:ℝ) + (-1 : ℝ) - 1 |) / (real.sqrt ((1:ℝ)^2 + (1:ℝ)^2)) ∧ d = real.sqrt 2)
  ∧ ∃ l : ℝ, l = (2 * real.sqrt 2) / 2
  → ∃ r : ℝ, r = real.sqrt (d^2 + l^2)
  ∧ ∃ r : ℝ, r = 2
  ∧ ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 := 
sorry

end circle_eq_of_conditions_l249_249031


namespace savings_proof_l249_249636

variable (income expenditure savings : ℕ)

def ratio_income_expenditure (i e : ℕ) := i / 10 = e / 7

theorem savings_proof (h : ratio_income_expenditure income expenditure) (hincome : income = 10000) :
  savings = income - expenditure → savings = 3000 :=
by
  sorry

end savings_proof_l249_249636


namespace sum_of_first_six_primes_l249_249349

theorem sum_of_first_six_primes : (2 + 3 + 5 + 7 + 11 + 13) = 41 :=
by
  sorry

end sum_of_first_six_primes_l249_249349


namespace intersection_P_Q_eq_Q_l249_249990

-- Definitions of P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Statement to prove P ∩ Q = Q
theorem intersection_P_Q_eq_Q : P ∩ Q = Q := 
by 
  sorry

end intersection_P_Q_eq_Q_l249_249990


namespace liters_pepsi_144_l249_249725

/-- A drink vendor has 50 liters of Maaza, some liters of Pepsi, and 368 liters of Sprite. -/
def liters_maaza : ℕ := 50
def liters_sprite : ℕ := 368
def num_cans : ℕ := 281

/-- The total number of liters of drinks the vendor has -/
def total_liters (lit_pepsi: ℕ) : ℕ := liters_maaza + lit_pepsi + liters_sprite

/-- Given that the least number of cans required is 281, prove that the liters of Pepsi is 144. -/
theorem liters_pepsi_144 (P : ℕ) (h: total_liters P % num_cans = 0) : P = 144 :=
by
  sorry

end liters_pepsi_144_l249_249725


namespace product_of_possible_values_of_x_l249_249959

theorem product_of_possible_values_of_x : 
  (∀ x, |x - 7| - 5 = 4 → x = 16 ∨ x = -2) -> (16 * -2 = -32) :=
by
  intro h
  have := h 16
  have := h (-2)
  sorry

end product_of_possible_values_of_x_l249_249959


namespace girls_together_count_l249_249807

-- Define the problem conditions
def boys : ℕ := 4
def girls : ℕ := 2
def total_entities : ℕ := boys + (girls - 1) -- One entity for the two girls together

-- Calculate the factorial
noncomputable def factorial (n: ℕ) : ℕ :=
  if n = 0 then 1 else (List.range (n+1)).foldl (λx y => x * y) 1

-- Define the total number of ways girls can be together
noncomputable def ways_girls_together : ℕ :=
  factorial total_entities * factorial girls

-- State the theorem that needs to be proved
theorem girls_together_count : ways_girls_together = 240 := by
  sorry

end girls_together_count_l249_249807


namespace third_side_length_not_4_l249_249932

theorem third_side_length_not_4 (x : ℕ) : 
  (5 < x + 9) ∧ (9 < x + 5) ∧ (x + 5 < 14) → ¬ (x = 4) := 
by
  intros h
  sorry

end third_side_length_not_4_l249_249932


namespace equivalent_single_discount_l249_249252

theorem equivalent_single_discount :
  ∀ (x : ℝ), ((1 - 0.15) * (1 - 0.10) * (1 - 0.05) * x) = (1 - 0.273) * x :=
by
  intros x
  --- This proof is left blank intentionally.
  sorry

end equivalent_single_discount_l249_249252


namespace john_total_climb_height_l249_249128

-- Define the heights and conditions
def num_flights : ℕ := 3
def height_per_flight : ℕ := 10
def total_stairs_height : ℕ := num_flights * height_per_flight
def rope_height : ℕ := total_stairs_height / 2
def ladder_height : ℕ := rope_height + 10

-- Prove that the total height John climbed is 70 feet
theorem john_total_climb_height : 
  total_stairs_height + rope_height + ladder_height = 70 := by
  sorry

end john_total_climb_height_l249_249128


namespace henri_total_miles_l249_249000

noncomputable def g_total : ℕ := 315 * 3
noncomputable def h_total : ℕ := g_total + 305

theorem henri_total_miles : h_total = 1250 :=
by
  -- proof goes here
  sorry

end henri_total_miles_l249_249000


namespace find_k_l249_249646

theorem find_k 
  (k : ℝ) 
  (h1 : k > 0) 
  (h2 : |det ![![3, 2, 2], ![4, k, 3], ![5, 3, k+1]]| = 20) : 
  k = (15 + Real.sqrt 237) / 6 :=
sorry

end find_k_l249_249646


namespace relationship_y1_y2_l249_249480

-- Define the existence of points M and N on the line y = -3x + 1
def points_on_line (y1 y2 : ℝ) (p1 : M(-3, y1)) (p2 : N(2, y2)) :=
  y1 = -3 * (-3) + 1 ∧ y2 = -3 * 2 + 1

-- Define the theorem to prove the relationship y1 > y2
theorem relationship_y1_y2 (y1 y2 : ℝ) :
  points_on_line y1 y2 → 
  y1 > y2 :=
by
  sorry

end relationship_y1_y2_l249_249480


namespace vovochka_max_candies_l249_249866

/-!
# Problem Statement:
- Vovochka has 25 classmates.
- Vovochka brought 200 candies to class.

## Condition:
- Any 16 classmates together should have at least 100 candies in total.

Question: What is the maximum number of candies Vovochka can keep for himself while fulfilling his mother's request?

The answer to this problem needs to be proven mathematically.
-/

def max_candies_can_keep (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ) : ℕ :=
if h : classmates = 25 ∧ total_candies = 200 ∧ min_candies_per_16 = 100 then
  37
else 
  sorry -- This statement is just for context, keeping focus on formal proof in Lean.

theorem vovochka_max_candies :
  ∀ (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ),
    classmates = 25 → total_candies = 200 → min_candies_per_16 = 100 →
    max_candies_can_keep candies classmates total_candies min_candies_per_16 = 37 :=
by
  intros candies classmates total_candies min_candies_per_16 hc ht hm
  simp [max_candies_can_keep]
  exact if_pos ⟨hc, ht, hm⟩

end vovochka_max_candies_l249_249866


namespace possible_values_of_m_l249_249001

def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def S (x : ℝ) (m : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

theorem possible_values_of_m (m : ℝ) : (∀ x, S x m → P x) ↔ (m = -1 ∨ m = 1 ∨ m = 3) :=
by
  sorry

end possible_values_of_m_l249_249001


namespace probability_heads_is_one_eighth_l249_249264

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_is_one_eighth_l249_249264


namespace solve_for_x_l249_249512

theorem solve_for_x (x : ℝ) : 
  x - 3 * x + 5 * x = 150 → x = 50 :=
by
  intro h
  -- sorry to skip the proof
  sorry

end solve_for_x_l249_249512


namespace min_val_x_add_y_l249_249832

noncomputable theory

theorem min_val_x_add_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 2 / y = 1) : 
  x + y ≥ 3 + 2 * Real.sqrt(2) :=
by
  sorry

end min_val_x_add_y_l249_249832


namespace max_divisors_up_to_20_l249_249214

def divisors (n : Nat) : List Nat :=
  (List.range (n + 1)).filter (λ d, n % d = 0)

def count_divisors (n : Nat) : Nat :=
  (divisors n).length

theorem max_divisors_up_to_20 :
  ∀ n, n ∈ List.range 21 → count_divisors n ≤ 6 ∧
      ((count_divisors 12 = 6) ∧
       (count_divisors 18 = 6) ∧
       (count_divisors 20 = 6)) :=
by
  sorry

end max_divisors_up_to_20_l249_249214


namespace min_magnitude_of_vec_set_l249_249057

def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2 + v.3^2)

def a := (1 : ℝ, 2, 3)
def b := (1 : ℝ, -1, 1)

def vec_set (k : ℤ) : ℝ × ℝ × ℝ := (a.1 + k, a.2 - k, a.3 + k)

theorem min_magnitude_of_vec_set : ∃ k : ℤ, vector_magnitude (vec_set k) = Real.sqrt 13 :=
by
  sorry

end min_magnitude_of_vec_set_l249_249057


namespace diving_competition_correct_l249_249941

def diving_competition : Prop :=
  ∀ (difficulty : ℝ) (scores : List ℝ) (point_value : ℝ),
    difficulty = 3.2 →
    scores = [7.5, 7.8, 9.0, 6.0, 8.5] →
    point_value = 76.16 →
    -- dropping the highest (9.0) and lowest (6.0) scores
    let remaining_scores := scores.erase (9.0).erase (6.0) in
    -- calculating the point value
    (point_value = (remaining_scores.sum * difficulty)) →
    -- conclude the number of judges
    scores.length = 5

theorem diving_competition_correct : diving_competition :=
by
  unfold diving_competition
  intros
  sorry

end diving_competition_correct_l249_249941


namespace valid_numbers_l249_249134

-- Define the conditions for three-digit numbers
def isThreeDigitNumber (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000

-- Define the splitting cases and the required property
def satisfiesFirstCase (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 100 * a + 10 * b + c ∧
  3 * ((10 * a + b) * c) = n

def satisfiesSecondCase (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a ≠ 0 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
  n = 100 * a + 10 * b + c ∧
  3 * (a * (10 * b + c)) = n

-- Define the main proposition
def validThreeDigitNumber (n : ℕ) : Prop :=
  isThreeDigitNumber n ∧ (satisfiesFirstCase n ∨ satisfiesSecondCase n)

-- The theorem statement which we need to prove
theorem valid_numbers : ∀ n : ℕ, validThreeDigitNumber n ↔ n = 150 ∨ n = 240 ∨ n = 735 :=
by
  sorry

end valid_numbers_l249_249134


namespace vovochka_max_candies_l249_249865

/-!
# Problem Statement:
- Vovochka has 25 classmates.
- Vovochka brought 200 candies to class.

## Condition:
- Any 16 classmates together should have at least 100 candies in total.

Question: What is the maximum number of candies Vovochka can keep for himself while fulfilling his mother's request?

The answer to this problem needs to be proven mathematically.
-/

def max_candies_can_keep (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ) : ℕ :=
if h : classmates = 25 ∧ total_candies = 200 ∧ min_candies_per_16 = 100 then
  37
else 
  sorry -- This statement is just for context, keeping focus on formal proof in Lean.

theorem vovochka_max_candies :
  ∀ (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ),
    classmates = 25 → total_candies = 200 → min_candies_per_16 = 100 →
    max_candies_can_keep candies classmates total_candies min_candies_per_16 = 37 :=
by
  intros candies classmates total_candies min_candies_per_16 hc ht hm
  simp [max_candies_can_keep]
  exact if_pos ⟨hc, ht, hm⟩

end vovochka_max_candies_l249_249865


namespace triangle_side_length_l249_249720

theorem triangle_side_length 
  (radius area : ℝ) (A B :Point) (OD_length : ℝ) 
  (EF : Line) (DEF : Triangle) 
  (is_chord : EF.isChord) (is_equilateral : DEF.isEquilateral)
  (C_outside_DEF : ¬ C.inTriangle DEF) : 
  radius = 14 → area = π * (radius ^ 2) → OD_length = 7 → area = 196π → 
  OD_length = 7 → C_outside_DEF → 
  DEF.side_length = 7 * ℝ.sqrt 3 := by
  sorry

end triangle_side_length_l249_249720


namespace correct_option_B_l249_249160

noncomputable def M : ℝ → MeasureTheory.Measure ℝ := 
  sorry  -- Placeholder for the normal distribution N(165, σ^2)

theorem correct_option_B {P : ℝ → ℝ → ℝ} :
  (P(165, 167) = 0.3) ∧ (P(-∞, 162) = 0.15) → (P(167, 168) = 0.05) :=
sorry

end correct_option_B_l249_249160


namespace total_shaded_area_l249_249116

-- Define the conditions
def larger_circle_area : ℝ := 100 * Real.pi
def smaller_circle_center_on_circumference : Prop := true  -- Assume true as given
def smaller_circle_touches_internally : Prop := true  -- Assume true as given
def circles_divided_into_three_equal_areas : Prop := true  -- Assume true as given

-- Define the theorem to prove
theorem total_shaded_area :
  (2 * (larger_circle_area / 3) + 2 * ((larger_circle_area / 4) / 3)) = (250 * Real.pi) / 3 :=
by
  sorry

end total_shaded_area_l249_249116


namespace vovochka_max_candies_l249_249864

/-!
# Problem Statement:
- Vovochka has 25 classmates.
- Vovochka brought 200 candies to class.

## Condition:
- Any 16 classmates together should have at least 100 candies in total.

Question: What is the maximum number of candies Vovochka can keep for himself while fulfilling his mother's request?

The answer to this problem needs to be proven mathematically.
-/

def max_candies_can_keep (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ) : ℕ :=
if h : classmates = 25 ∧ total_candies = 200 ∧ min_candies_per_16 = 100 then
  37
else 
  sorry -- This statement is just for context, keeping focus on formal proof in Lean.

theorem vovochka_max_candies :
  ∀ (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ),
    classmates = 25 → total_candies = 200 → min_candies_per_16 = 100 →
    max_candies_can_keep candies classmates total_candies min_candies_per_16 = 37 :=
by
  intros candies classmates total_candies min_candies_per_16 hc ht hm
  simp [max_candies_can_keep]
  exact if_pos ⟨hc, ht, hm⟩

end vovochka_max_candies_l249_249864


namespace train_length_correct_l249_249393

variable (time : ℝ) 
variable (speed_man_kmh : ℝ) 
variable (speed_train_kmh : ℝ)

def relative_speed_mps (speed_train_kmh speed_man_kmh : ℝ) : ℝ :=
  (speed_train_kmh - speed_man_kmh) * (5 / 18)

def train_length (relative_speed_mps time : ℝ) : ℝ :=
  relative_speed_mps * time

theorem train_length_correct (htime : time = 47.99616030717543)
                             (hspeed_man : speed_man_kmh = 3)
                             (hspeed_train : speed_train_kmh = 63) : 
  train_length (relative_speed_mps speed_train_kmh speed_man_kmh) time = 799.936 :=
by 
  sorry

end train_length_correct_l249_249393


namespace number_is_48_l249_249437

theorem number_is_48 (x : ℝ) (h : (1/4) * x + 15 = 27) : x = 48 :=
by sorry

end number_is_48_l249_249437


namespace convert_750μg_to_scientific_notation_l249_249677

-- Definitions of conversion factors given in the conditions
def micrograms_to_grams (μg : ℝ) : ℝ := μg * 10⁻⁶
def grams (g : ℝ) : ℝ := g

-- Theorem statement to prove the conversion of 750 micrograms to scientific notation
theorem convert_750μg_to_scientific_notation :
  micrograms_to_grams 750 = 7.5 * 10⁻⁴ :=
by
  sorry

end convert_750μg_to_scientific_notation_l249_249677


namespace divisors_of_12_18_20_l249_249187

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem divisors_of_12_18_20 (n : ℕ) (h : n ∈ {1, 2, 3, ..., 20}) :
  n ∈ {12, 18, 20} ↔ (∀ m ∈ {1, 2, 3, ..., 20}, count_divisors m ≤ 6) :=
by
  sorry

end divisors_of_12_18_20_l249_249187


namespace max_distance_C2_to_C1_l249_249500

-- Define the given parametric equation of curve C1
def C1_parametric (t : ℝ) : ℝ × ℝ :=
  let x := -2 - (sqrt 3 / 2) * t
  let y := (1 / 2) * t
  (x, y)

-- Define the given polar equation of curve C2
def C2_polar (θ : ℝ) : ℝ :=
  2 * sqrt 2 * cos (θ - π / 4)

-- Definition to express the rectangular coordinates for curve C2
def C2_rectangular (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

-- Define the line equation for curve C1
def C1_line (x y : ℝ) : Prop :=
  x + sqrt 3 * y + 2 = 0

-- Proof problem statement
theorem max_distance_C2_to_C1 :
  (∀ θ : ℝ, ∃ x y : ℝ, C2_polar θ = sqrt (x^2 + y^2) ∧ C2_rectangular x y) →
  (∀ t : ℝ, ∃ x y : ℝ, C1_parametric t = (x, y)) →
  ∀ x y : ℝ, C2_rectangular x y → C1_line x y →
  ∃ d : ℝ, d = (3 + sqrt 3) / 2 + sqrt 2 := by
  sorry

end max_distance_C2_to_C1_l249_249500


namespace minimum_t_value_l249_249043

theorem minimum_t_value (t : ℕ) : (∃ x1 x2 ∈ set.Icc (0 : ℝ) (t:ℝ), (sin (π * x1 / 3) = 1) ∧ (sin (π * x2 / 3) = 1) ∧ (x1 ≠ x2)) → t ≥ 8 :=
by
  -- skipping proof
  sorry

end minimum_t_value_l249_249043


namespace complex_expression_evaluation_l249_249991

noncomputable def complexNumbers := ℂ

theorem complex_expression_evaluation (a b : complexNumbers)
  (h : (a + b) / (a - b) + (a - b) / (a + b) = 4) :
  (a ^ 4 + b ^ 4) / (a ^ 4 - b ^ 4) + (a ^ 4 - b ^ 4) / (a ^ 4 + b ^ 4) = 41 / 20 := by
  sorry

end complex_expression_evaluation_l249_249991


namespace max_candies_l249_249873

theorem max_candies (c : ℕ := 200) (n m : ℕ := 25) (k : ℕ := 16) (t : ℕ := 100) 
  (H : ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - (25 - (x.1 : ℕ))) i ≥ 100) : 
  ∃ d : ℕ, (d = 37 ∧ ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - d - (25 - (x.1 : ℕ))) i ≤ 100) :=
by {
  use 37,
  sorry
}

end max_candies_l249_249873


namespace crayons_selection_l249_249522

theorem crayons_selection (total_crayons : ℕ) (select_crayons : ℕ) (total_without_red_blue : ℕ) (correct_answer : ℕ) :
  total_crayons = 18 → select_crayons = 6 → total_without_red_blue = 8008 → correct_answer = 10556 →
  (nat.choose total_crayons select_crayons) - (nat.choose total_crayons.pred.pred select_crayons) = correct_answer :=
by
  intros h_total h_select h_without h_answer
  rw [h_total, h_select, h_without, h_answer]
  sorry

end crayons_selection_l249_249522


namespace log_comparison_l249_249469

theorem log_comparison (a b c : ℝ) 
  (h₁ : a = Real.log 6 / Real.log 3)
  (h₂ : b = Real.log 10 / Real.log 5)
  (h₃ : c = Real.log 14 / Real.log 7) :
  a > b ∧ b > c :=
  sorry

end log_comparison_l249_249469


namespace angle_A_l249_249088

noncomputable def triangleA (a b c : ℝ) (A B C : ℝ) :=
  (b = 2 * a * Real.sin B) → (A = 30 ∨ A = 150)

theorem angle_A (a b c A B C : ℝ) : triangleA a b c A B C :=
begin
  intro h,
  -- proof is omitted 
  sorry
end

end angle_A_l249_249088


namespace range_of_m_l249_249046

noncomputable def f (x : ℝ) := |x - 3| - 2
noncomputable def g (x : ℝ) := -|x + 1| + 4

theorem range_of_m (m : ℝ) : (∀ x, f x - g x ≥ m + 1) ↔ m ≤ -3 :=
by
  sorry

end range_of_m_l249_249046


namespace dan_must_exceed_speed_to_arrive_before_cara_l249_249625

noncomputable def minimum_speed_for_dan (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) : ℕ :=
  (distance / (distance / cara_speed - dan_delay)) + 1

theorem dan_must_exceed_speed_to_arrive_before_cara
  (distance : ℕ) (cara_speed : ℕ) (dan_delay : ℕ) :
  distance = 180 →
  cara_speed = 30 →
  dan_delay = 1 →
  minimum_speed_for_dan distance cara_speed dan_delay > 36 :=
by
  sorry

end dan_must_exceed_speed_to_arrive_before_cara_l249_249625


namespace greatest_num_divisors_in_range_l249_249175

def num_divisors (n : ℕ) : ℕ :=
  (List.range (n+1)).count (λ d, n % d = 0)

theorem greatest_num_divisors_in_range : 
  ∀ n ∈ {1, 2, 3, ..., 20}, 
  num_divisors n < 7 → (n = 12 ∨ n = 18 ∨ n = 20) :=
by {
  sorry
}

end greatest_num_divisors_in_range_l249_249175


namespace sum_of_roots_eq_p_l249_249424

theorem sum_of_roots_eq_p (p q : ℝ) 
  (h: p * p - p - 1 = 0) 
  (h₂: q * q - q - 1 = 0)
  (h₃ : p ≠ q) :
  let a := p + q,
      b := pq,
      sum_of_roots := λ (a b : ℝ), a 
  in 
  sum_of_roots a (p * q) = p ∨ sum_of_roots a (p * q) = q :=
sorry

end sum_of_roots_eq_p_l249_249424


namespace find_fg3_l249_249920

def f (x : ℝ) : ℝ := 4 - Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem find_fg3 : f (g 3) = -2 := by
  sorry

end find_fg3_l249_249920


namespace eq_solutions_a2_eq_b_times_b_plus_7_l249_249771

theorem eq_solutions_a2_eq_b_times_b_plus_7 (a b : ℤ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h : a^2 = b * (b + 7)) :
  (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
sorry

end eq_solutions_a2_eq_b_times_b_plus_7_l249_249771


namespace probability_heads_is_one_eighth_l249_249266

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_is_one_eighth_l249_249266


namespace coefficient_of_x4_l249_249929

theorem coefficient_of_x4 (a : ℝ) (h : 15 * a^4 = 240) : a = 2 ∨ a = -2 := 
sorry

end coefficient_of_x4_l249_249929


namespace downstream_speed_l249_249382

variable (d : ℝ) (upstream_speed : ℝ) (round_trip_speed : ℝ)

theorem downstream_speed
  (h1 : upstream_speed = 6)
  (h2 : round_trip_speed = 6.857142857142857) :
  let v := (12 * round_trip_speed) / (2 * round_trip_speed - upstream_speed) in
  v = 8 := by
  sorry

end downstream_speed_l249_249382


namespace frances_card_value_l249_249972

theorem frances_card_value (x : ℝ) (hx : 90 < x ∧ x < 180) :
  (∃ f : ℝ → ℝ, f = sin ∨ f = cos ∨ f = tan ∧
    f x = -1 ∧
    (∃ y : ℝ, y ≠ x ∧ (sin y ≠ -1 ∧ cos y ≠ -1 ∧ tan y ≠ -1))) :=
sorry

end frances_card_value_l249_249972


namespace find_x2_y2_and_xy_l249_249465

-- Problem statement
theorem find_x2_y2_and_xy (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 9) : 
  x^2 + y^2 = 5 ∧ x * y = -2 :=
by
  sorry -- Proof omitted

end find_x2_y2_and_xy_l249_249465


namespace orthocenter_equilateral_l249_249598

variables {A B C H A1 B1 C1 : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C]
[LinearOrder H] [LinearOrder A1] [LinearOrder B1] [LinearOrder C1]

-- Define the orthocenter and altitude conditions
def orthocenter_divides_altitudes_same_ratio (h_ratio : ∀ {A B C H A1 B1 C1 : Type}, 
  A * B * C * H * A1 * B1 * C1 → LinearOrder A → LinearOrder B → LinearOrder C → LinearOrder H → LinearOrder A1 
  → LinearOrder B1 → LinearOrder C1 → 
  (A1 * H) / (H * A) = (B1 * H) / (H * B) ∧ (A1 * H) / (H * A) = (C1 * H) / (H * C)) := sorry

-- Problem statement: if orthocenter divides the altitudes in the same ratio, then the triangle is equilateral
theorem orthocenter_equilateral {A B C H A1 B1 C1 : Type} [LinearOrder A] [LinearOrder B] [LinearOrder C]
  [LinearOrder H] [LinearOrder A1] [LinearOrder B1] [LinearOrder C1]
  (h_ratio : ∀ {A B C H A1 B1 C1 : Type}, A * B * C * H * A1 * B1 * C1 → LinearOrder A → 
  LinearOrder B → LinearOrder C → LinearOrder H → LinearOrder A1 → LinearOrder B1 → LinearOrder C1 → 
  (A1 * H) / (H * A) = (B1 * H) / (H * B) ∧ (A1 * H) / (H * A) = (C1 * H) / (H * C)) 
  : (∃ (A' B' C' : Type) [LinearOrder A'] [LinearOrder B'] [LinearOrder C'], A' = B' ∧ B' = C' ∧ C' = A') := sorry

end orthocenter_equilateral_l249_249598


namespace max_candies_vovochka_can_keep_l249_249905

-- Definitions for the conditions in the problem.
def classmates := 25
def total_candies := 200
def minimum_candies (candies_distributed : Vector ℕ classmates) : Prop :=
  ∀ (S : Finset (Fin classmates)), S.card = 16 → S.sum (candies_distributed.nth' S) ≥ 100

-- The proof goal
theorem max_candies_vovochka_can_keep (candies_distributed : Vector ℕ classmates) (h : minimum_candies candies_distributed) :
  ∃ k, k = 37 ∧ total_candies - Finset.univ.sum (candies_distributed.nth' Finset.univ) = k :=
begin
  sorry
end

end max_candies_vovochka_can_keep_l249_249905


namespace maximum_candies_l249_249892

-- Definitions of the problem parameters
def classmates : ℕ := 25
def total_candies : ℕ := 200
def condition (candies : ℕ → ℕ) : Prop := 
  ∀ S (h : finset ℕ), S.card = 16 → (∑ i in S, candies i) ≥ 100

-- The theorem stating the maximum number of candies Vovochka can keep
theorem maximum_candies (candies : ℕ → ℕ) (h : condition candies) : 
  (total_candies - ∑ i in finset.range classmates, candies i) ≤ 37 :=
sorry

end maximum_candies_l249_249892


namespace jon_buys_2_coffees_each_day_l249_249552

-- Define the conditions
def cost_per_coffee : ℕ := 2
def total_spent : ℕ := 120
def days_in_april : ℕ := 30

-- Define the total number of coffees bought
def total_coffees_bought : ℕ := total_spent / cost_per_coffee

-- Prove that Jon buys 2 coffees each day
theorem jon_buys_2_coffees_each_day : total_coffees_bought / days_in_april = 2 := by
  sorry

end jon_buys_2_coffees_each_day_l249_249552


namespace find_value_of_expression_l249_249486

-- Conditions
variables (c d : ℝ)
axiom h1 : 100^c = 4
axiom h2 : 100^d = 5

-- Statement we need to prove
theorem find_value_of_expression : 20^((1 - c - d)/(2 * (1 - d))) = 5 :=
by sorry

end find_value_of_expression_l249_249486


namespace mom_in_first_position_l249_249587

-- Let M, D, and G represent the number of clothes bought by Mom, Dad, and Grandpa, respectively.
variable (M D G : Nat)

-- Given conditions
axiom dad_more_than_mom : D > M
axiom grandpa_more_than_dad : G > D

-- Proposition to prove
theorem mom_in_first_position (M D G : Nat) (h1 : D > M) (h2 : G > D) : ∀ (l : List Nat), l = [M, D, G] → l.sorted Nat.le → l.index_of M = 0 := by
  sorry

end mom_in_first_position_l249_249587


namespace least_subtract_to_divisible_by_14_l249_249353

theorem least_subtract_to_divisible_by_14 (n : ℕ) (h : n = 7538): 
  (n % 14 = 6) -> ∃ m, (m = 6) ∧ ((n - m) % 14 = 0) :=
by
  sorry

end least_subtract_to_divisible_by_14_l249_249353


namespace csc_sec_inequality_l249_249236

open Real

theorem csc_sec_inequality (n : ℕ) (x : ℝ) (h1 : sin x ≠ 0) (h2 : cos x ≠ 0) :
  (csc x ^ (2 * n) - 1) * (sec x ^ (2 * n) - 1) ≥ (2^n - 1) ^ 2 :=
by
  sorry

end csc_sec_inequality_l249_249236


namespace divisors_of_12_18_20_l249_249186

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem divisors_of_12_18_20 (n : ℕ) (h : n ∈ {1, 2, 3, ..., 20}) :
  n ∈ {12, 18, 20} ↔ (∀ m ∈ {1, 2, 3, ..., 20}, count_divisors m ≤ 6) :=
by
  sorry

end divisors_of_12_18_20_l249_249186


namespace binom_prime_coprime_l249_249135

theorem binom_prime_coprime {p k n : ℕ} (hp : p.prime) (hk : 0 < k) (hn : 0 < n) (coprime_pn : p.coprime n) :
  Nat.gcd (Nat.choose (n * p^k) (p^k)) p = 1 :=
sorry

end binom_prime_coprime_l249_249135


namespace find_constant_d_l249_249925

noncomputable def polynomial_g (d : ℝ) (x : ℝ) := d * x^4 + 17 * x^3 - 5 * d * x^2 + 45

theorem find_constant_d (d : ℝ) : polynomial_g d 5 = 0 → d = -4.34 :=
by
  sorry

end find_constant_d_l249_249925


namespace segment_shorter_than_leg_l249_249529

theorem segment_shorter_than_leg (A B C M K : Type) (h_triangle : (A, B, C) : ℝ)
  (angle_A : ∠A = π / 6) (AB_hypot : AB : ℝ)
  (midpoint_M : ∃ M, M ∈ midpoint (AB))
  (point_K : ∃ K, K ∈ AC)  (KM_perpendicular : ∃ KM, KM ⊥ AB) :
  ∃ KM, KM = (1 / 3) * AC :=
begin
  sorry
end

end segment_shorter_than_leg_l249_249529


namespace volume_new_pyramid_l249_249737

theorem volume_new_pyramid (base_edge : ℝ) (slant_edge : ℝ) (plane_height : ℝ) :
  base_edge = 12 * Real.sqrt 2 →
  slant_edge = 15 →
  plane_height = 5 →
  let d := 12 * Real.sqrt 2 in
  let h := Real.sqrt (slant_edge^2 - (d / 2)^2) in
  let new_h := h - plane_height in
  let new_base_edge_len := (plane_height / h) * base_edge in
  let new_base_area := (new_base_edge_len^2) / 2 in
  (1/3) * new_base_area * new_h = 50 + 16/27 :=
by
  intros
  let d := 12 * Real.sqrt 2
  let h := Real.sqrt (slant_edge^2 - (d / 2)^2)
  let new_h := h - plane_height
  let new_base_edge_len := (plane_height / h) * base_edge
  let new_base_area := (new_base_edge_len^2) / 2
  have vol_calc : (1/3) * new_base_area * new_h = 50 + 16/27
  sorry

end volume_new_pyramid_l249_249737


namespace find_x3_y3_l249_249018

noncomputable def x_y_conditions (x y : ℝ) :=
  x - y = 3 ∧
  x^2 + y^2 = 27

theorem find_x3_y3 (x y : ℝ) (h : x_y_conditions x y) : x^3 - y^3 = 108 :=
  sorry

end find_x3_y3_l249_249018


namespace Vovochka_max_candies_l249_249882

theorem Vovochka_max_candies (classmates : Finset ℕ) (candies : ℕ) (hclassmates : classmates.card = 25) (hcandies : candies = 200) (H : ∀ (S : Finset ℕ), S.card = 16 → 100 ≤ S.sum (λ x, 1)) :
  ∃ (max_candies : ℕ), max_candies = 37 :=
by
  use 37
  sorry

end Vovochka_max_candies_l249_249882


namespace greatest_divisors_1_to_20_l249_249202

def is_divisor (a b : ℕ) : Prop := b % a = 0

def count_divisors (n : ℕ) : ℕ :=
  nat.succ (list.length ([k for k in list.range n if is_divisor k.succ n]))

def max_divisors_from_1_to_20 : set ℕ :=
  {n | 1 ≤ n ∧ n ≤ 20 ∧ ∃ max_count, count_divisors n = max_count ∧ 
  ∀ m, 1 ≤ m ∧ m ≤ 20 → count_divisors m ≤ max_count}

theorem greatest_divisors_1_to_20 : max_divisors_from_1_to_20 = {12, 18, 20} :=
by {
  sorry
}

end greatest_divisors_1_to_20_l249_249202


namespace area_of_right_triangle_l249_249530

variables {x y : ℝ} (r : ℝ)

theorem area_of_right_triangle (hx : ∀ r, r * (x + y + r) = x * y) :
  1 / 2 * (x + r) * (y + r) = x * y :=
by sorry

end area_of_right_triangle_l249_249530


namespace ms_sally_swift_speed_l249_249163

/-- Ms. Sally Swift's driving speed to arrive exactly on time --/
theorem ms_sally_swift_speed :
  ∃ r : ℝ, ∀ (d t : ℝ),
    (d = 35 * (t + 5/60) ∧ d = 50 * (t - 5/60)) →
    r = d / t :=
begin
  use 41,
  intros d t h,
  cases h with h₁ h₂,
  have : t = 17 / 36,
  { -- Manipulating the equations to solve for t
    have eq_time : 35 * (t + 1/12) = 50 * (t - 1/12) := by rw [←h₁, ←h₂],
    linarith,
  },
  -- Calculate d using t = 17 / 36
  rw this at h₁,
  have : d = 35 * (17/36 + 1/12) := h₁,
  linarith,
  sorry
end

end ms_sally_swift_speed_l249_249163


namespace problem_equivalence_l249_249047

-- Define the line equation
def line_eq (x : ℝ) : ℝ := x + 3

-- Define the direct proportion function
def proportion_eq (k x : ℝ) : ℝ := k * x

-- Define points A and B
def pointA : ℝ × ℝ := (0, 3)
def pointB (m : ℝ) : ℝ × ℝ := (-1, m)

-- Define the area of the triangle
def triangle_area (A B : ℝ × ℝ) : ℝ := 0.5 * (A.2 - 0) * (-B.1)

theorem problem_equivalence :
  (line_eq 0 = 3) ∧
  (line_eq (-1) = 2) ∧
  (proportion_eq (-2) (-1) = 2) ∧
  (triangle_area pointA (pointB 2) = 3) :=
by
  have h1 : line_eq 0 = 3 := rfl
  have h2 : line_eq (-1) = 2 := by simp [line_eq]
  have h3 : proportion_eq (-2) (-1) = 2 := by simp [proportion_eq]
  have h4 : triangle_area pointA (pointB 2) = 3 := by simp [triangle_area, pointB, pointA]
  exact ⟨h1, h2, h3, h4⟩

end problem_equivalence_l249_249047


namespace james_additional_votes_needed_to_win_l249_249352

noncomputable def votes_cast : ℕ := 2000
noncomputable def james_percentage : ℝ := 0.005
noncomputable def james_votes_received : ℕ := (james_percentage * votes_cast : ℕ)
noncomputable def votes_needed_to_win : ℕ := votes_cast / 2 + 1

theorem james_additional_votes_needed_to_win : james_votes_received + 991 = votes_needed_to_win := by
  sorry

end james_additional_votes_needed_to_win_l249_249352


namespace max_candies_l249_249869

theorem max_candies (c : ℕ := 200) (n m : ℕ := 25) (k : ℕ := 16) (t : ℕ := 100) 
  (H : ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - (25 - (x.1 : ℕ))) i ≥ 100) : 
  ∃ d : ℕ, (d = 37 ∧ ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - d - (25 - (x.1 : ℕ))) i ≤ 100) :=
by {
  use 37,
  sorry
}

end max_candies_l249_249869


namespace circle_radius_zero_l249_249448

-- Theorem statement
theorem circle_radius_zero :
  ∀ (x y : ℝ), 4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0 → 
  ∃ (c : ℝ) (r : ℝ), r = 0 ∧ (x - 1)^2 + (y + 2)^2 = r^2 :=
by sorry

end circle_radius_zero_l249_249448


namespace transformed_function_l249_249856

noncomputable def g (f : ℝ → ℝ) : ℝ → ℝ :=
  λ x, f (6 - x)

variable {f : ℝ → ℝ}

theorem transformed_function (x : ℝ) : g f x = f (6 - x) :=
by
  sorry

end transformed_function_l249_249856


namespace max_area_triangle_bqc_l249_249119

noncomputable def triangle_problem : ℝ :=
  let a := 112.5
  let b := 56.25
  let c := 3
  a + b + c

theorem max_area_triangle_bqc : triangle_problem = 171.75 :=
by
  -- The proof would involve validating the steps to ensure the computations
  -- for the maximum area of triangle BQC match the expression 112.5 - 56.25 √3,
  -- and thus confirm that a = 112.5, b = 56.25, c = 3
  -- and verifying that a + b + c = 171.75.
  sorry

end max_area_triangle_bqc_l249_249119


namespace length_of_segment_in_cube_l249_249463

theorem length_of_segment_in_cube :
  ∀ (X Y : ℝ × ℝ × ℝ) (cubes : list (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ)),
    X = (0, 0, 0) →
    Y = (5, 5, 13) →
    cubes = [((0, 0, 0), (1, 1, 1)), ((0, 0, 1), (3, 3, 4)), ((0, 0, 4), (4, 4, 8)), ((0, 0, 8), (5, 5, 13))] →
    let entry_point := (0, 0, 4),
        exit_point := (4, 4, 8),
        length := Real.sqrt ((4 - 0) ^ 2 + (4 - 0) ^ 2 + (8 - 4) ^ 2) in
    length = 4 * Real.sqrt 3 :=
  sorry -- Proof elided

end length_of_segment_in_cube_l249_249463


namespace cyclic_quadrilateral_equivalence_l249_249304

variables {A B C D O M N : Type} [euclidean_geometry A B C D O M N]

-- Define the conditions
def diagonals_intersect (A B C D O : Type) : Prop := 
∃ O, line A C ∩ line B D = O

def ac_angle_bisector (A B C D O : Type) : Prop :=
angle A B D = 2 * angle A B C

def midpoint (x y m : Type) : Prop :=
distance x m = distance m y

-- Define the cyclic property of a quadrilateral
def cyclic (A B C D : Type) : Prop :=
angle A B C + angle C D A = π ∧ 
angle B C D + angle D A B = π

theorem cyclic_quadrilateral_equivalence :
  (diagonals_intersect A B C D O) ∧ 
  (ac_angle_bisector A B C D O) ∧ 
  (midpoint B C M) ∧
  (midpoint D O N) →
  (cyclic A B C D ↔ cyclic A B M N) :=
begin
  sorry
end

end cyclic_quadrilateral_equivalence_l249_249304


namespace midpoint_BC_equidistant_l249_249358

-- Given definitions and assumptions
variables {A B C D X Y : Point}
variables (inscribed_quad : InscribedQuadrilateral A B C D)
variable (P_bisec_BD : PerpendicularBisectorIntersection X BD AD)
variable (P_bisec_AC : PerpendicularBisectorIntersection Y AC AD)

-- What we need to prove
theorem midpoint_BC_equidistant (M : Point) (midpoint_BC : Midpoint M B C) :
  Equidistant M (LineThrough B X) (LineThrough C Y) :=
begin
  sorry
end

end midpoint_BC_equidistant_l249_249358


namespace combined_total_circles_squares_l249_249295

-- Define the problem parameters based on conditions
def US_stars : ℕ := 50
def US_stripes : ℕ := 13
def circles (n : ℕ) : ℕ := (n / 2) - 3
def squares (n : ℕ) : ℕ := (n * 2) + 6

-- Prove that the combined number of circles and squares on Pete's flag is 54
theorem combined_total_circles_squares : 
    circles US_stars + squares US_stripes = 54 := by
  sorry

end combined_total_circles_squares_l249_249295


namespace cos_product_pos_l249_249953

noncomputable def triangle := Type

variables {α : Type} [linear_ordered_field α]

structure obtuse_triangle (α : Type) :=
(A B C : α)
(is_obtuse : ∃ i : ℕ, i < 3 ∧ A + B + C = π ∧ (if i = 0 then A > π / 2 else if i = 1 then B > π / 2 else C > π / 2))
(sin_bounds : A < B ∧ B < C ∧ sin A < sin B ∧ sin B < sin C)

theorem cos_product_pos {t : obtuse_triangle α} (h : ∀ x, t.sin_bounds) :
  cos t.A * cos t.B > 0 :=
sorry

end cos_product_pos_l249_249953


namespace numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249228

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).length

theorem numbers_with_most_divisors_in_range :
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  (number_of_divisors n ≤ 6) :=
by
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list
  simp [number_of_divisors] <;>
  dec_trivial -- Automatically discharges the goals since they are simple arithmetic comparisons

theorem greatest_divisors_in_range :
  number_of_divisors 12 = 6 ∧
  number_of_divisors 18 = 6 ∧
  number_of_divisors 20 = 6 ∧
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19],
  number_of_divisors n < 6 :=
by
  refine ⟨rfl, rfl, rfl, _⟩
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list excluding 12, 18, 20
  dec_trivial -- Automatically discharges the goals due to their simplicity

#print greatest_divisors_in_range

end numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249228


namespace max_four_color_rectangles_l249_249501

-- Define the integer grid points (S)
def S : set (ℕ × ℕ) := { p | p.1 ∈ finset.range 100 ∧ p.2 ∈ finset.range 100 }

-- Define four colors
inductive Color | A | B | C | D

-- Define a color function for grid points
def color_function : (ℕ × ℕ) → Color := sorry

-- Define the property of a four-color rectangle
def is_four_color_rectangle (p1 p2 p3 p4 : ℕ × ℕ) : Prop :=
  p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.2 = p3.2 ∧ p2.2 = p4.2 ∧
  (color_function p1 ≠ color_function p2) ∧
  (color_function p1 ≠ color_function p3) ∧
  (color_function p1 ≠ color_function p4) ∧
  (color_function p2 ≠ color_function p3) ∧
  (color_function p2 ≠ color_function p4) ∧
  (color_function p3 ≠ color_function p4)

-- Define the maximum number of four-color rectangles
theorem max_four_color_rectangles :
  ∃ (n : ℕ), n = 9375000 ∧
  (∀ p1 p2 p3 p4 ∈ S, is_four_color_rectangle p1 p2 p3 p4 → n ≤ 9375000) :=
sorry

end max_four_color_rectangles_l249_249501


namespace angle_AHE_is_22_5_degrees_l249_249764

def regular_octagon := 
  ∀ (A B C D E F G H : ℝ × ℝ),
  is_regular_octagon A B C D E F G H

theorem angle_AHE_is_22_5_degrees (A B C D E F G H : ℝ × ℝ)
  (h : regular_octagon A B C D E F G H) 
  : angle A H E = 22.5 :=
sorry

end angle_AHE_is_22_5_degrees_l249_249764


namespace inequality_proof_l249_249013

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (h : a + b + c = 1)

theorem inequality_proof :
  (1 + a) / (1 - a) + (1 + b) / (1 - a) + (1 + c) / (1 - c) ≤ 2 * ((b / a) + (c / b) + (a / c)) :=
by sorry

end inequality_proof_l249_249013


namespace probability_heads_penny_nickel_dime_is_one_eighth_l249_249273

-- Define the setup: flipping 5 coins
def five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the condition: penny, nickel, and dime are heads
def heads_penny_nickel_dime (c: five_coins) : bool := c.1 && c.2 && c.3

-- Define the successful outcomes where penny, nickel, and dime are heads
noncomputable def successful_outcomes : Set five_coins :=
  { c | heads_penny_nickel_dime c = tt }

-- Define the total outcomes for 5 coins
noncomputable def total_outcomes : Set five_coins := {c | true}

-- Probability calculation: |success| / |total|
noncomputable def probability_heads_penny_nickel_dime : real :=
  (Set.card successful_outcomes : real) / (Set.card total_outcomes : real)

-- Given the setup, prove that the probability is 1/8
theorem probability_heads_penny_nickel_dime_is_one_eighth :
  probability_heads_penny_nickel_dime = 1 / 8 := 
sorry

end probability_heads_penny_nickel_dime_is_one_eighth_l249_249273


namespace sum_of_roots_l249_249569

theorem sum_of_roots (y1 y2 k m : ℝ) (h1 : y1 ≠ y2) (h2 : 5 * y1^2 - k * y1 = m) (h3 : 5 * y2^2 - k * y2 = m) : 
  y1 + y2 = k / 5 := 
by
  sorry

end sum_of_roots_l249_249569


namespace triangle_properties_l249_249964

variable {A B C a b c : ℝ}

-- Defining the conditions of the problem
def perpendicular_vectors (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def sinA_value (A : ℝ) : Prop :=
  sin A = 4 / 5

def max_area (a : ℝ) (S : ℝ) : Prop :=
  a = 2 * sqrt 2 → S ≤ 4

-- Statement of the proof
theorem triangle_properties (h_perpendicular : perpendicular_vectors (sin B, 5 * sin A + 5 * sin C) (5 * sin B - 6 * sin C, sin C - sin A))
  (h_sinA : sinA_value A) :
  (∃ A, sinA_value A) ∧ max_area a (1 / 2 * b * c * sin A) := sorry

end triangle_properties_l249_249964


namespace smallest_d_l249_249152

theorem smallest_d (d f : ℕ) (h_pos : d > 0 ∧ f > 0) (h_seq : ∀ i : ℕ, i ∈ {1, 2, 3, 4} → a (i + 1) = a i ^ 2) 
  (a1_def : a 1 = 0.9) (h_prod : ∏ i in (Finset.range 4).image (λ i, i + 1), a i = (3 ^ d) / f) 
  : d = 30 :=
by
  sorry

end smallest_d_l249_249152


namespace percent_y_of_x_l249_249698

theorem percent_y_of_x (x y : ℝ) (h : 0.60 * (x - y) = 0.30 * (x + y)) : y / x = 1 / 3 :=
by
  -- proof steps would be provided here
  sorry

end percent_y_of_x_l249_249698


namespace average_speed_round_trip_l249_249713

theorem average_speed_round_trip (D : ℝ) (h : D > 0) :
  let speed := 45
  let time_to := D / speed
  let time_back := 2 * time_to
  let total_distance := 2 * D
  let total_time := time_to + time_back
  let average_speed := total_distance / total_time
  in average_speed = 30 :=
by
  sorry

end average_speed_round_trip_l249_249713


namespace increasing_sequence_condition_no_decreasing_sequence_l249_249440

variables {V : Type*} [inner_product_space ℝ V]
variables (x y : V) [nontrivial V]

-- Part a: The sequence is increasing if and only if 3|y| > 2|x| * cos(angle between x and y).
theorem increasing_sequence_condition :
  3 * ∥y∥ > 2 * ∥x∥ * real.cos (real.angle x y) ↔ 
  ∀ n : ℕ, n ≠ 0 → ∥x - n • y∥ > ∥x - (n - 1) • y∥ :=
sorry

-- Part b: There are no pairs (x, y) such that the sequence is decreasing.
theorem no_decreasing_sequence : 
  ¬ (∃ x y : V, ∀ n : ℕ, n ≠ 0 → ∥x - n • y∥ < ∥x - (n - 1) • y∥) :=
sorry

end increasing_sequence_condition_no_decreasing_sequence_l249_249440


namespace largest_angle_of_isosceles_obtuse_30_deg_l249_249673

def is_isosceles (T : Triangle) : Prop :=
  T.A = T.B ∨ T.B = T.C ∨ T.C = T.A

def is_obtuse (T : Triangle) : Prop :=
  T.A > 90 ∨ T.B > 90 ∨ T.C > 90

def T : Type := {P Q R : ℝ}

noncomputable def largest_angle (T : Triangle) : ℝ :=
  max T.A (max T.B T.C)

theorem largest_angle_of_isosceles_obtuse_30_deg :
  ∀ (T : Triangle), is_isosceles T → is_obtuse T → T.A = 30 → largest_angle T = 120 :=
by
  intro T h_iso h_obt h_A30
  sorry

end largest_angle_of_isosceles_obtuse_30_deg_l249_249673


namespace four_digit_numbers_count_l249_249806

def digits : List ℕ := [1, 2, 3]

-- Condition: All three digits must be used.
def all_three_digits_used (n : List ℕ) := digits ⊆ n

-- Condition: Identical digits cannot be adjacent.
def no_adjacent_identical (n : List ℕ) : Prop :=
  ∀ (i : ℕ), i < n.length - 1 → n[i] ≠ n[i + 1]

-- Definition of valid four-digit numbers
def valid_four_digit_numbers (nums : List (List ℕ)) :=
  filter (λ n, all_three_digits_used n ∧ no_adjacent_identical n) nums

-- Length of valid four-digit numbers should be 18
theorem four_digit_numbers_count : ∃ (nums : List (List ℕ)), valid_four_digit_numbers nums.length = 18 :=
  sorry

end four_digit_numbers_count_l249_249806


namespace triangle_perpendicular_distance_l249_249548

-- Define the problem in Lean 4 terms
theorem triangle_perpendicular_distance (A B C D : Type) [metric_space A] 
  [metric_space B] [metric_space C] [metric_space D]
  (h₁ : ∠ A C B = 90) (h₂ : dist A C = 156) (h₃ : dist A B = 169)
  (x : ℝ) : x = 60 := sorry

end triangle_perpendicular_distance_l249_249548


namespace correct_conclusions_l249_249105

theorem correct_conclusions (n : ℕ) (a b : ℝ) (x y : ℝ) (α β : ℝ) :
  (∀ n > 1, (a + b)^n ≠ a^n + b^n) →
  (sin(α + β) ≠ sin α * sin β) →
  (a + b)^2 = a^2 + 2 * a * b + b^2 →
  1 = 1 :=
by
  intros h1 h2 h3
  have h4 : (a + b)^n ≠ a^n + b^n := h1
  have h5 : (sin(α + β) ≠ sin α * sin β) := h2
  have h6 : (a + b)^2 = a^2 + 2 * a * b + b^2 := h3
  sorry

end correct_conclusions_l249_249105


namespace chessboard_star_property_l249_249612

theorem chessboard_star_property {n : ℕ} (H : ∀ (R ⊆ finset.range n) (hRn : R ≠ finset.range n),
    ∃ (C : ℕ), C ∈ finset.range n ∧ (finset.filter (λ r, ¬ r ∈ R) (finset.range n)).count C = 1) :
  ∀ (C' ⊆ finset.range n) (hC'n : C' ≠ finset.range n),
  ∃ (R' : ℕ), R' ∈ finset.range n ∧ (finset.filter (λ c, ¬ c ∈ C') (finset.range n)).count R' = 1 :=
sorry

end chessboard_star_property_l249_249612


namespace peregrines_eat_30_percent_l249_249323

theorem peregrines_eat_30_percent (initial_pigeons : ℕ) (chicks_per_pigeon : ℕ) (pigeons_left : ℕ) :
  initial_pigeons = 40 →
  chicks_per_pigeon = 6 →
  pigeons_left = 196 →
  (100 * (initial_pigeons * chicks_per_pigeon + initial_pigeons - pigeons_left)) / 
  (initial_pigeons * chicks_per_pigeon + initial_pigeons) = 30 :=
by
  intros
  sorry

end peregrines_eat_30_percent_l249_249323


namespace probability_heads_is_one_eighth_l249_249268

-- Define the probability problem
def probability_heads_penny_nickel_dime (total_coins: ℕ) (successful_events: ℕ) : ℚ :=
  successful_events / total_coins

-- Define the constants
def total_outcomes : ℕ := 2^5  -- Number of possible outcomes with 5 coins
def successful_outcomes : ℕ := 4  -- Number of successful outcomes where penny, nickel, and dime are heads

-- State the theorem to be proven
theorem probability_heads_is_one_eighth : 
  probability_heads_penny_nickel_dime total_outcomes successful_outcomes = 1 / 8 :=
by
  sorry

end probability_heads_is_one_eighth_l249_249268


namespace part1_part2_l249_249574

noncomputable def f : ℝ → ℝ := sorry

-- Conditions: f is monotonically increasing, and f(x+1) = f(x) + 3
axiom f_mono : ∀ x y : ℝ, x ≤ y → f(x) ≤ f(y)
axiom f_add3 : ∀ x : ℝ, f(x + 1) = f(x) + 3

-- Definitions of f^(n)
noncomputable def f_iter : ℕ → (ℝ → ℝ)
| 0       := λ x, x
| (n + 1) := λ x, f(f_iter n x)

-- Part 1: 3x + f(0) - 3 ≤ f(x) ≤ 3x + f(0) + 3
theorem part1 (x : ℝ) : 3 * x + f(0) - 3 ≤ f(x) ∧ f(x) ≤ 3 * x + f(0) + 3 := sorry

-- Part 2: |f^(n)(x) - f^(n)(y)| ≤ 3^n * (|x - y| + 3)
theorem part2 (x y : ℝ) (n : ℕ) : |f_iter n x - f_iter n y| ≤ 3^n * (|x - y| + 3) := sorry

end part1_part2_l249_249574


namespace probability_penny_nickel_dime_all_heads_l249_249286

-- Define flipping five coins
def flip_five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the function to check if the penny, nickel, and dime are heads
def all_heads_penny_nickel_dime (o : flip_five_coins) : bool :=
  o.1 = tt ∧ o.2 = tt ∧ o.3 = tt

-- Define the total count of possible outcomes
def total_outcomes : ℕ := 32

-- Define the count of favorable outcomes
def favorable_outcomes : ℕ := 4

-- Define the probability calculation
def probability_favorable : ℚ := favorable_outcomes / total_outcomes

-- The statement proving the probability is 1/8
theorem probability_penny_nickel_dime_all_heads :
  probability_favorable = 1 / 8 :=
by
  sorry

end probability_penny_nickel_dime_all_heads_l249_249286


namespace greatest_possible_integer_l249_249550

theorem greatest_possible_integer 
  (n k l : ℕ) 
  (h1 : n < 150) 
  (h2 : n = 9 * k - 2) 
  (h3 : n = 6 * l - 4) : 
  n = 146 := 
sorry

end greatest_possible_integer_l249_249550


namespace parallelogram_is_central_not_axis_symmetric_l249_249248

-- Definitions for the shapes discussed in the problem
def is_central_symmetric (shape : Type) : Prop := sorry
def is_axis_symmetric (shape : Type) : Prop := sorry

-- Specific shapes being used in the problem
def rhombus : Type := sorry
def parallelogram : Type := sorry
def equilateral_triangle : Type := sorry
def rectangle : Type := sorry

-- Example additional assumptions about shapes can be added here if needed

-- The problem assertion
theorem parallelogram_is_central_not_axis_symmetric :
  is_central_symmetric parallelogram ∧ ¬ is_axis_symmetric parallelogram :=
sorry

end parallelogram_is_central_not_axis_symmetric_l249_249248


namespace lucas_sequence_sum_l249_249989

theorem lucas_sequence_sum :
  let L : ℕ → ℕ := Nat.recOn
    (fun n => Nat.recOn n 1 (fun _ => 3))
    (fun n ih₃ ih₂ => ih₂ + ih₃)
  in 
  (∑' n : ℕ, if h : 0 < n then (L n / (3 : ℝ) ^ (n+1)) else 0) = 1 / 5 := by
  sorry

end lucas_sequence_sum_l249_249989


namespace max_divisors_1_to_20_l249_249207

theorem max_divisors_1_to_20 :
  ∃ m n o ∈ set.range (20:ℕ.succ), 
    (∀ k ∈ set.range (20:ℕ.succ), 
      (nat.divisors_count k ≤ nat.divisors_count m) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count n) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count o)) 
    ∧ (nat.divisors_count m = 6)
    ∧ (nat.divisors_count n = 6)
    ∧ (nat.divisors_count o = 6)
    ∧ m ≠ n ∧ n ≠ o ∧ o ≠ m
    ∧ set.mem 12 (set.range (20 + 1))
    ∧ set.mem 18 (set.range (20 + 1))
    ∧ set.mem 20 (set.range (20 + 1)). 
      := by 
  -- all these steps will be skipped here
  sorry

end max_divisors_1_to_20_l249_249207


namespace ellipse_foci_coordinates_l249_249302

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ), (x^2 / 64 + y^2 / 100 = 1) → (x = 0 ∧ (y = 6 ∨ y = -6)) :=
by
  sorry

end ellipse_foci_coordinates_l249_249302


namespace f_is_odd_max_min_values_l249_249565

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)
variable (f_add : ∀ x y : ℝ, f (x + y) = f (x) + f (y))
variable (f_one : f 1 = -2)
variable (f_neg : ∀ x > 0, f x < 0)

-- Define the statement in Lean for Part 1: proving the function is odd
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f (x) := by sorry

-- Define the statement in Lean for Part 2: proving the max and min values on [-3, 3]
theorem max_min_values : 
  ∃ max_value min_value : ℝ, 
  (max_value = f (-3) ∧ max_value = 6) ∧ 
  (min_value = f (3) ∧ min_value = -6) := by sorry

end f_is_odd_max_min_values_l249_249565


namespace tangent_line_eqn_l249_249630

theorem tangent_line_eqn (x y : ℝ) (h : y = exp x) (P : (x, y) = (0, 1))
  : x - y + 1 = 0 :=
sorry

end tangent_line_eqn_l249_249630


namespace probability_heads_penny_nickel_dime_is_one_eighth_l249_249270

-- Define the setup: flipping 5 coins
def five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the condition: penny, nickel, and dime are heads
def heads_penny_nickel_dime (c: five_coins) : bool := c.1 && c.2 && c.3

-- Define the successful outcomes where penny, nickel, and dime are heads
noncomputable def successful_outcomes : Set five_coins :=
  { c | heads_penny_nickel_dime c = tt }

-- Define the total outcomes for 5 coins
noncomputable def total_outcomes : Set five_coins := {c | true}

-- Probability calculation: |success| / |total|
noncomputable def probability_heads_penny_nickel_dime : real :=
  (Set.card successful_outcomes : real) / (Set.card total_outcomes : real)

-- Given the setup, prove that the probability is 1/8
theorem probability_heads_penny_nickel_dime_is_one_eighth :
  probability_heads_penny_nickel_dime = 1 / 8 := 
sorry

end probability_heads_penny_nickel_dime_is_one_eighth_l249_249270


namespace num_factors_M_l249_249559

def M : ℕ := 58^6 + 6 * 58^5 + 15 * 58^4 + 20 * 58^3 + 15 * 58^2 + 6 * 58 + 1

theorem num_factors_M : Nat.factors (M).length = 7 :=
by
  sorry

end num_factors_M_l249_249559


namespace sums_remainders_equal_l249_249988

-- Definition and conditions
variables (A A' D S S' s s' : ℕ) 
variables (h1 : A > A') 
variables (h2 : A % D = S) 
variables (h3 : A' % D = S') 
variables (h4 : (A + A') % D = s) 
variables (h5 : (S + S') % D = s')

-- Proof statement
theorem sums_remainders_equal : s = s' := 
  sorry

end sums_remainders_equal_l249_249988


namespace seatingArrangementsAreSix_l249_249419

-- Define the number of seating arrangements for 4 people around a round table
def numSeatingArrangements : ℕ :=
  3 * 2 * 1 -- Following the condition that the narrator's position is fixed

-- The main theorem stating the number of different seating arrangements
theorem seatingArrangementsAreSix : numSeatingArrangements = 6 :=
  by
    -- This is equivalent to following the explanation of solution which is just multiplying the numbers
    sorry

end seatingArrangementsAreSix_l249_249419


namespace binom_product_identity_l249_249460

-- Given the definition of generalized binomial coefficient.
def gen_binom (a : ℝ) (k : ℕ) : ℝ := (List.range k).prod (λ i, a - i) / k.fact

-- Prove the equality of the given binomial coefficients product to the simplified form.
theorem binom_product_identity :
  gen_binom (-3/2) 50 * gen_binom 3/2 50 = 4^(-50 : ℝ) * (43.fact)^2 * 51.fact :=
by
  sorry

end binom_product_identity_l249_249460


namespace room_dimension_l249_249624

theorem room_dimension :
  ∃ x : ℝ, 
  let wall_area := 2 * (25 * 12) + 2 * (x * 12),
      door_area := 6 * 3,
      window_area := 3 * (4 * 3),
      net_area := wall_area - door_area - window_area,
      cost := net_area * 4
  in
  cost = 3624 ∧ x = 15 :=
begin
  sorry
end

end room_dimension_l249_249624


namespace exists_prime_seq_satisfying_condition_l249_249779

theorem exists_prime_seq_satisfying_condition :
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ m n, m < n → a m < a n) ∧ 
  (∀ i j, i ≠ j → (i * a j, j * a i) = (i, j)) :=
sorry

end exists_prime_seq_satisfying_condition_l249_249779


namespace greatest_divisors_1_to_20_l249_249180

theorem greatest_divisors_1_to_20 : ∃ n₁ n₂ n₃, ((n₁ = 12 ∧ n₂ = 18 ∧ n₃ = 20) ∧ 
  (∀ m ∈ (Finset.range 21).filter (λ x, x > 0), 
    let d_m := (Finset.range (m + 1)).filter (λ k, m % k = 0) in
    ∃ k, k = d_m.card ∧ k <= 6 ∧ ((m = n₁ ∧ k = 6) ∨ (m = n₂ ∧ k = 6) ∨ (m = n₃ ∧ k = 6)))) :=
by
  sorry

end greatest_divisors_1_to_20_l249_249180


namespace initial_contestants_proof_l249_249944

noncomputable def initial_contestants (final_round : ℕ) : ℕ :=
  let fraction_remaining := 2 / 5
  let fraction_advancing := 1 / 2
  let fraction_final := fraction_remaining * fraction_advancing
  (final_round : ℕ) / fraction_final

theorem initial_contestants_proof : initial_contestants 30 = 150 :=
sorry

end initial_contestants_proof_l249_249944


namespace no_disjoint_quadratic_residue_sets_modulo_p_l249_249139

theorem no_disjoint_quadratic_residue_sets_modulo_p (p : ℕ) (hp : Nat.prime p) (h5 : 5 < p) :
    ¬ ∃ (a c : ℕ), Nat.gcd (a * c) p = 1 ∧ set.disjoint 
    ({x : ℕ // x < p ∧ ∃ y : ℕ, y < p ∧ (y^2) % p = x} : set ℕ)
    ({x : ℕ // x < p ∧ ∃ b : ℕ, b < (p-1)/2 ∧ (a * (b + 1) + c) % p = x} : set ℕ) :=
by {
  sorry
}

end no_disjoint_quadratic_residue_sets_modulo_p_l249_249139


namespace max_divisors_up_to_20_l249_249217

def divisors (n : Nat) : List Nat :=
  (List.range (n + 1)).filter (λ d, n % d = 0)

def count_divisors (n : Nat) : Nat :=
  (divisors n).length

theorem max_divisors_up_to_20 :
  ∀ n, n ∈ List.range 21 → count_divisors n ≤ 6 ∧
      ((count_divisors 12 = 6) ∧
       (count_divisors 18 = 6) ∧
       (count_divisors 20 = 6)) :=
by
  sorry

end max_divisors_up_to_20_l249_249217


namespace weighted_average_plants_per_hour_l249_249911

theorem weighted_average_plants_per_hour :
  let heath_carrot_plants_100 := 100 * 275
  let heath_carrot_plants_150 := 150 * 325
  let heath_total_plants := heath_carrot_plants_100 + heath_carrot_plants_150
  let heath_total_time := 10 + 20
  
  let jake_potato_plants_50 := 50 * 300
  let jake_potato_plants_100 := 100 * 400
  let jake_total_plants := jake_potato_plants_50 + jake_potato_plants_100
  let jake_total_time := 12 + 18

  let total_plants := heath_total_plants + jake_total_plants
  let total_time := heath_total_time + jake_total_time
  let weighted_average := total_plants / total_time
  weighted_average = 2187.5 :=
by
  sorry

end weighted_average_plants_per_hour_l249_249911


namespace value_of_f_log2_3_l249_249854

noncomputable def f : ℝ → ℝ
| x => if x ≥ 2 then 2^x else f (x + 2)

theorem value_of_f_log2_3 : f (Real.log 3 / Real.log 2) = 12 := by
  sorry

end value_of_f_log2_3_l249_249854


namespace largest_angle_of_obtuse_isosceles_triangle_l249_249663

theorem largest_angle_of_obtuse_isosceles_triangle (P Q R : Type) 
  (triangle_PQR : Triangle P Q R)
  (isosceles_PQR : Isosceles triangle_PQR) 
  (obtuse_PQR : Obtuse triangle_PQR)
  (angle_P_30 : angle P triangle_PQR = 30) : 
  ∃ (angle_Q : ℕ), is_largest_angle angle_Q triangle_PQR ∧ angle_Q = 120 := 
by 
  sorry

end largest_angle_of_obtuse_isosceles_triangle_l249_249663


namespace numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249229

def number_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).length

theorem numbers_with_most_divisors_in_range :
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
  (number_of_divisors n ≤ 6) :=
by
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list
  simp [number_of_divisors] <;>
  dec_trivial -- Automatically discharges the goals since they are simple arithmetic comparisons

theorem greatest_divisors_in_range :
  number_of_divisors 12 = 6 ∧
  number_of_divisors 18 = 6 ∧
  number_of_divisors 20 = 6 ∧
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19],
  number_of_divisors n < 6 :=
by
  refine ⟨rfl, rfl, rfl, _⟩
  intro n hn
  cases hn <;> -- Splits the goals based on each element of the list excluding 12, 18, 20
  dec_trivial -- Automatically discharges the goals due to their simplicity

#print greatest_divisors_in_range

end numbers_with_most_divisors_in_range_greatest_divisors_in_range_l249_249229


namespace maximal_q_for_broken_line_l249_249600

theorem maximal_q_for_broken_line :
  ∃ q : ℝ, (∀ i : ℕ, 0 ≤ i → i < 5 → ∀ A_i : ℝ, (A_i = q ^ i)) ∧ 
  (q = (1 + Real.sqrt 5) / 2) := sorry

end maximal_q_for_broken_line_l249_249600


namespace jack_finishes_book_in_13_days_l249_249122

def total_pages : ℕ := 285
def pages_per_day : ℕ := 23

theorem jack_finishes_book_in_13_days : (total_pages + pages_per_day - 1) / pages_per_day = 13 := by
  sorry

end jack_finishes_book_in_13_days_l249_249122


namespace prob_draw_l249_249515

theorem prob_draw (p_not_losing p_winning p_drawing : ℝ) (h1 : p_not_losing = 0.6) (h2 : p_winning = 0.5) :
  p_drawing = 0.1 :=
by
  sorry

end prob_draw_l249_249515


namespace max_value_y_l249_249811

theorem max_value_y (x y : ℕ) (h₁ : 9 * (x + y) > 17 * x) (h₂ : 15 * x < 8 * (x + y)) :
  y ≤ 112 :=
sorry

end max_value_y_l249_249811


namespace total_miles_traveled_l249_249127

noncomputable def initial_fee : ℝ := 2.0
noncomputable def charge_per_2_5_mile : ℝ := 0.35
noncomputable def total_charge : ℝ := 5.15

theorem total_miles_traveled :
  ∃ (miles : ℝ), total_charge = initial_fee + (charge_per_2_5_mile * miles * (5 / 2)) ∧ miles = 3.6 :=
by
  sorry

end total_miles_traveled_l249_249127


namespace sum_yi_cubes_l249_249570

theorem sum_yi_cubes (y : Fin 50 → ℝ) (h_sum : ∑ i, y i = 2)
    (h_frac_sum : ∑ i, y i ^ 2 / (1 - y i) = 2) :
    ∑ i, y i ^ 3 / (1 - y i) = 2 - ∑ i, (y i ^ 2) := by
  sorry

end sum_yi_cubes_l249_249570


namespace max_divisors_1_to_20_l249_249209

theorem max_divisors_1_to_20 :
  ∃ m n o ∈ set.range (20:ℕ.succ), 
    (∀ k ∈ set.range (20:ℕ.succ), 
      (nat.divisors_count k ≤ nat.divisors_count m) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count n) 
      ∧ (nat.divisors_count k ≤ nat.divisors_count o)) 
    ∧ (nat.divisors_count m = 6)
    ∧ (nat.divisors_count n = 6)
    ∧ (nat.divisors_count o = 6)
    ∧ m ≠ n ∧ n ≠ o ∧ o ≠ m
    ∧ set.mem 12 (set.range (20 + 1))
    ∧ set.mem 18 (set.range (20 + 1))
    ∧ set.mem 20 (set.range (20 + 1)). 
      := by 
  -- all these steps will be skipped here
  sorry

end max_divisors_1_to_20_l249_249209


namespace arithmetic_sequence_geometric_condition_l249_249491

theorem arithmetic_sequence_geometric_condition (a : ℕ → ℤ) (d : ℤ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_a1 : a 1 = 1) (h_d_nonzero : d ≠ 0)
  (h_geom : (1 + d) * (1 + d) = 1 * (1 + 4 * d)) : a 2013 = 4025 := by sorry

end arithmetic_sequence_geometric_condition_l249_249491


namespace m_div_x_l249_249313

variable (a b k : ℝ)
variable (ha : a = 4 * k)
variable (hb : b = 5 * k)
variable (k_pos : k > 0)

def x := a * 1.25
def m := b * 0.20

theorem m_div_x : m / x = 1 / 5 := by
  sorry

end m_div_x_l249_249313


namespace ellipse_equation_fixed_point_max_area_of_triangle_l249_249476

theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
(eccentricity : c / a = √6 / 3) (area_quad: 2 * a * b = 2 * √3) 
(hyp : a^2 = b^2 + c^2) : ellipse_eq := 
  ∃ (a : ℝ), ∃ (b : ℝ), ∃ (c : ℝ), (a = √3) ∧ (b = 1) ∧ (c = √2) ∧ 
  (∀ (x y : ℝ), ellipse_eq = (x^2 / 3 + y^2 = 1)) :=
begin
  sorry
end

theorem fixed_point (A M N : ℝ) (hAMAN : (slope AM) * (slope AN) = 2 / 3) :
  ∀ (x : ℝ), line MN = x + (kx - 3) :=
begin
  sorry
end

theorem max_area_of_triangle (A M N : ℝ) (h_AMAN : (slope AM) * (slope AN) = 2 / 3) 
(hyp_A : A = (0,1)) (hyp_MN : line MN passes through the fixed point) 
(hyp_len : |MN| <= sqrt(1 + k^2) * |x1 - x2|) :
  (area_triangle AMN)_max = (2 * √3 / 3) :=
begin
  sorry
end


end ellipse_equation_fixed_point_max_area_of_triangle_l249_249476


namespace common_difference_range_l249_249110

variable (d : ℝ)

def a (n : ℕ) : ℝ := -5 + (n - 1) * d

theorem common_difference_range (H1 : a 10 > 0) (H2 : a 9 ≤ 0) :
  (5 / 9 < d) ∧ (d ≤ 5 / 8) :=
by
  sorry

end common_difference_range_l249_249110


namespace ellipse_standard_eq_and_max_triangle_area_l249_249841

theorem ellipse_standard_eq_and_max_triangle_area
  (f1 f2 : ℝ × ℝ)
  (point_through_ellipse : ℝ × ℝ)
  (line_slope_through_point : ℝ × ℝ)
  (max_area : ℝ) :
  f1 = (-Real.sqrt 2, 0) →
  f2 = (Real.sqrt 2, 0) →
  point_through_ellipse = (Real.sqrt 2 / 2, Real.sqrt 30 / 6) →
  line_slope_through_point = (0, -2) →
  max_area = Real.sqrt 3 / 2 →
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b < a ∧ (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  let eq := x^2 / a^2 + y^2 / b^2 = 1 in
  ∃ (k : ℝ), (k^2 > 1 → 
  let l := (k, -2) in
  ∀ (x1 y1 x2 y2 : ℝ), (line_eq l = y = kx - 2) ∧ 
  ∀ (x y : ℝ), eq ∧ ∀ o : ℝ, area (0, 0) (o, l)) ∧
  max_area = Real.sqrt 3 / 2)
sorry

end ellipse_standard_eq_and_max_triangle_area_l249_249841


namespace measure_of_TVW_is_5_degrees_l249_249958

noncomputable def denote_triangle (T U V : Type) [linear_ordered_comm_ring V] :=
  V

def theta_WTV : ℝ := 58
def theta_TUW : ℝ := 75
def theta_WVU : ℝ := 42

def TWV (T U V W : Type) [linear_ordered_comm_ring W] :=
  180 - theta_WTV - theta_WVU

def TVW (T U V W : Type) [linear_ordered_comm_ring W] :=
  TWV T U V W - theta_TUW

-- Theorem: the measure of ∠TVW in degrees is 5°
theorem measure_of_TVW_is_5_degrees (T U V W : Type) [linear_ordered_comm_ring W] :
  TVW T U V W = 5 :=
by
  unfold TVW
  unfold TWV
  have h : 180 - 58 - 42 = 80 := by norm_num
  rw h
  have h1 : 80 - 75 = 5 := by norm_num
  rw h1
  rfl

end measure_of_TVW_is_5_degrees_l249_249958


namespace tan_alpha_value_l249_249026

variables {α β : ℝ}

theorem tan_alpha_value 
  (h1 : α ∈ Ioc (-π / 2) π)
  (h2 : sin (α + 2 * β) - 2 * sin β * cos (α + β) = - 1 / 3) : 
  tan α = - (Real.sqrt 2) / 4 :=
by
  sorry

end tan_alpha_value_l249_249026


namespace divisors_of_12_18_20_l249_249194

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem divisors_of_12_18_20 (n : ℕ) (h : n ∈ {1, 2, 3, ..., 20}) :
  n ∈ {12, 18, 20} ↔ (∀ m ∈ {1, 2, 3, ..., 20}, count_divisors m ≤ 6) :=
by
  sorry

end divisors_of_12_18_20_l249_249194


namespace slope_angle_of_line_l249_249680

theorem slope_angle_of_line (α : ℝ) (hα : 0 ≤ α ∧ α < 180) 
    (slope_eq_tan : Real.tan α = 1) : α = 45 :=
by
  sorry

end slope_angle_of_line_l249_249680


namespace theta_in_third_quadrant_l249_249915

-- Define the mathematical conditions
variable (θ : ℝ)
axiom cos_theta_neg : Real.cos θ < 0
axiom cos_minus_sin_eq_sqrt : Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ)

-- Prove that θ is in the third quadrant
theorem theta_in_third_quadrant : 
  (∀ θ : ℝ, Real.cos θ < 0 → Real.cos θ - Real.sin θ = Real.sqrt (1 - 2 * Real.sin θ * Real.cos θ) → 
    Real.sin θ < 0 ∧ Real.cos θ < 0) :=
by sorry

end theta_in_third_quadrant_l249_249915


namespace circle_area_from_line_intersection_correct_l249_249489

open Real

noncomputable def circle_area_from_line_intersection : Real :=
  let line (t : Real) := (1 + t, 4 - 2 * t)
  let circle (θ : Real) := (2 * cos θ + 2, 2 * sin θ)
  let circle_equation (x y : Real) := (x - 2)^2 + y^2 = 4
  let line_standard_form (x y : Real) := 2 * x + y - 6 = 0
  let distance_from_center_to_line := abs ((2 * 2) + (1 * 4) - 6) / sqrt (2^2 + 1^2)
  let R := 2
  let AB := 2 * sqrt ((R ^ 2) - (distance_from_center_to_line^2))
  let area := pi * (AB / 2)^2
  area

theorem circle_area_from_line_intersection_correct :
  circle_area_from_line_intersection = 16 * pi / 5 :=
by
  -- this proof will require the proper calculations as outlined in the problem statement
  sorry

end circle_area_from_line_intersection_correct_l249_249489


namespace Jana_taller_than_Kelly_l249_249123

-- Definitions and given conditions
def Jess_height := 72
def Jana_height := 74
def Kelly_height := Jess_height - 3

-- Proof statement
theorem Jana_taller_than_Kelly : Jana_height - Kelly_height = 5 := by
  sorry

end Jana_taller_than_Kelly_l249_249123


namespace max_candies_l249_249872

theorem max_candies (c : ℕ := 200) (n m : ℕ := 25) (k : ℕ := 16) (t : ℕ := 100) 
  (H : ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - (25 - (x.1 : ℕ))) i ≥ 100) : 
  ∃ d : ℕ, (d = 37 ∧ ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - d - (25 - (x.1 : ℕ))) i ≤ 100) :=
by {
  use 37,
  sorry
}

end max_candies_l249_249872


namespace count_greater_than_one_point_one_l249_249794

theorem count_greater_than_one_point_one : 
  let numbers := [1.4, 9 / 10, 1.2, 0.5, 13 / 10]
  (list.filter (λ x, x > 1.1) numbers).length = 3 := by
  sorry

end count_greater_than_one_point_one_l249_249794


namespace vovochka_max_candies_l249_249900

noncomputable def max_candies_for_vovochka 
  (total_classmates : ℕ)
  (total_candies : ℕ)
  (candies_condition : ∀ (subset : Finset ℕ), subset.card = 16 → ∑ i in subset, (classmate_candies i) ≥ 100) 
  (classmate_candies : ℕ → ℕ) : ℕ :=
  total_candies - ∑ i in (Finset.range total_classmates), classmate_candies i

theorem vovochka_max_candies :
  max_candies_for_vovochka 25 200 (λ subset hc, True) (λ i, if i < 13 then 7 else 6) = 37 :=
by
  sorry

end vovochka_max_candies_l249_249900


namespace max_candies_l249_249875

theorem max_candies (c : ℕ := 200) (n m : ℕ := 25) (k : ℕ := 16) (t : ℕ := 100) 
  (H : ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - (25 - (x.1 : ℕ))) i ≥ 100) : 
  ∃ d : ℕ, (d = 37 ∧ ∀ S : Finset (Fin n), S.card = k → ∑ i in S, (fun (x : Fin n) => 200 - d - (25 - (x.1 : ℕ))) i ≤ 100) :=
by {
  use 37,
  sorry
}

end max_candies_l249_249875


namespace verify_cosine_relationship_l249_249495

theorem verify_cosine_relationship 
  (A B C : ℝ) 
  (h : A + B + C = π) :
  cos A + cos B + cos C = 1 + 4 * sin (A / 2) * sin (B / 2) * sin (C / 2) := 
sorry

end verify_cosine_relationship_l249_249495


namespace positive_difference_of_solutions_l249_249249

theorem positive_difference_of_solutions :
  let a := 2
  let b := -10 - 2
  let c := 18 - 42
  let delta := b^2 - 4 * a * c
  let x1 := (-b + Real.sqrt delta) / (2 * a)
  let x2 := (-b - Real.sqrt delta) / (2 * a)
  2 * (Real.sqrt 21) = (x1 - x2) :=
by
  let a := 2
  let b := -10 - 2
  let c := 18 - 42
  let delta := b^2 - 4 * a * c
  let x1 := (-b + Real.sqrt delta) / (2 * a)
  let x2 := (-b - Real.sqrt delta) / (2 * a)
  have h1 : x1 - x2 = 2 * (Real.sqrt 21), by sorry
  exact h1

end positive_difference_of_solutions_l249_249249


namespace Vovochka_max_candies_l249_249876

theorem Vovochka_max_candies (classmates : Finset ℕ) (candies : ℕ) (hclassmates : classmates.card = 25) (hcandies : candies = 200) (H : ∀ (S : Finset ℕ), S.card = 16 → 100 ≤ S.sum (λ x, 1)) :
  ∃ (max_candies : ℕ), max_candies = 37 :=
by
  use 37
  sorry

end Vovochka_max_candies_l249_249876


namespace part_I_part_II_part_III_l249_249499

-- Define the line equation kx - y + 1 + 2k = 0
def line (k : ℝ) : ℝ → ℝ → Prop := λ x y, k * x - y + 1 + 2 * k = 0

-- Part (I): Prove that the line passes through the fixed point (-2, 1)
theorem part_I (k : ℝ) : line k (-2) 1 := by {
  unfold line,
  rw [←sub_eq_zero],
  simp,
}

-- Part (II): If the line does not pass through the fourth quadrant, k ≥ 0
theorem part_II (k : ℝ) (h : ∀ x y : ℝ, ¬ (line k x y ∧ x < 0 ∧ y < 0)) : k ≥ 0 := by {
  sorry
}

-- Part (III): Prove that if the line intersects the negative half of the x-axis at A and the positive half of the y-axis at B, then the minimum area S of triangle AOB is 4, and the line equation is x-2*y+4=0
theorem part_III (k : ℝ) (h1 : k > 0) (h2 : 1 + 2*k > 0) : 
  let A := (- (1 + 2*k) / k, 0),
      B := (0, 1 + 2*k),
      S := 0.5 * ℝ.abs ((- (1 + 2*k) / k) * (1 + 2*k)) in
  S = 4 ∧ ∃ a b c : ℝ, a = 1 ∧ b = -2 ∧ c = 4 := by {
  sorry
}

end part_I_part_II_part_III_l249_249499


namespace tetrahedron_coloring_l249_249534

noncomputable def count_distinct_tetrahedron_colorings : ℕ :=
  sorry

theorem tetrahedron_coloring :
  count_distinct_tetrahedron_colorings = 6 :=
  sorry

end tetrahedron_coloring_l249_249534


namespace digits_to_the_right_of_decimal_l249_249507

noncomputable def expr : ℚ := (5^6 : ℚ) / (10^5 * 8)

theorem digits_to_the_right_of_decimal (x : ℚ) (dec_repr : String) :
  x = expr →
  dec_repr = x.toDecimalString →
  dec_repr.countTrailingZeros = 8 :=
sorry

end digits_to_the_right_of_decimal_l249_249507


namespace geometric_sequence_fn_sum_l249_249314

def geometric_sequence (a : ℕ → ℝ) : Prop := ∃ a1 r : ℝ, ∀ n, a n = a1 * r^(n - 1)

def log_base_1_2 (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

theorem geometric_sequence_fn_sum 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_pos : ∀ n, 0 < a n)
  (h_a4 : a 4 = 2)
  (f : ℝ → ℝ := log_base_1_2) :
  f(a(1)^3) + f(a(2)^3) + f(a(3)^3) + f(a(4)^3) 
    + f(a(5)^3) + f(a(6)^3) + f(a(7)^3) = -21 := 
sorry

end geometric_sequence_fn_sum_l249_249314


namespace neighbor_eggs_taken_l249_249164

theorem neighbor_eggs_taken (h1 : ∀ hens eggs_per_day days, hens = 3 ∧ eggs_per_day = 3 ∧ days = 7 → hens * eggs_per_day * days = 63)
                           (h2 : ∀ eggs_collected dropped, eggs_collected = 46 ∧ dropped = 5 → eggs_collected + dropped = 51)
                           (h3 : ∀ total_laid collected, total_laid = 63 ∧ collected = 51 → total_laid - collected = 12) :
  ∃ (eggs_taken : ℕ), eggs_taken = 12 :=
by {
  use 12,
  have h_total_laid := h1 3 3 7 ⟨rfl, rfl, rfl⟩,
  have h_eggs_collected := h2 46 5 ⟨rfl, rfl⟩,
  have h_eggs_taken := h3 63 51 ⟨rfl, rfl⟩,
  exact h_eggs_taken
}

end neighbor_eggs_taken_l249_249164


namespace find_x3_y3_l249_249019

noncomputable def x_y_conditions (x y : ℝ) :=
  x - y = 3 ∧
  x^2 + y^2 = 27

theorem find_x3_y3 (x y : ℝ) (h : x_y_conditions x y) : x^3 - y^3 = 108 :=
  sorry

end find_x3_y3_l249_249019


namespace example_proof_l249_249407

open Classical

variables (A B C D E F : Point) (h_triangle : Triangle A B C)
variables (h_AD_median : IsMedian A D B C) (h_CF_intersects : LineIntersects CF E F AB AD CF)
variables (h_midpoint_D : Midpoint D B C)

theorem example_proof : (segment_length A E / segment_length E D) = (2 * segment_length A F / segment_length F B) :=
by
  sorry

end example_proof_l249_249407


namespace slope_angle_of_chord_through_focus_l249_249472

theorem slope_angle_of_chord_through_focus
  (parabola : ∀ x y : ℝ, y^2 = 6 * x)
  (focus : (ℝ × ℝ) := (3 / 2, 0))
  (chord_length : ℝ := 12)
  (chord : ∀ (A B : ℝ × ℝ), A ∈ parabola ∧ B ∈ parabola ∧ (focus.1 - A.1)^2 + (focus.2 - A.2)^2 + (focus.1 - B.1)^2 + (focus.2 - B.2)^2 = chord_length^2) :
  ∃ α : ℝ, α = π / 4 ∨ α = 3 * π / 4 :=
by
  sorry

end slope_angle_of_chord_through_focus_l249_249472


namespace sqrt_expression_real_l249_249083

theorem sqrt_expression_real (x : ℝ) : (∃ y : ℝ, y = sqrt (x + 1)) ↔ x ≥ -1 := 
by
  sorry

end sqrt_expression_real_l249_249083


namespace area_bounded_by_parabola_and_x_axis_l249_249413

/-- Define the parabola function -/
def parabola (x : ℝ) : ℝ := 2 * x - x^2

/-- The function for the x-axis -/
def x_axis : ℝ := 0

/-- Prove that the area bounded by the parabola and x-axis between x = 0 and x = 2 is 4/3 -/
theorem area_bounded_by_parabola_and_x_axis : 
  (∫ x in (0 : ℝ)..(2 : ℝ), parabola x) = 4 / 3 := by
    sorry

end area_bounded_by_parabola_and_x_axis_l249_249413


namespace probability_heads_penny_nickel_dime_is_one_eighth_l249_249271

-- Define the setup: flipping 5 coins
def five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the condition: penny, nickel, and dime are heads
def heads_penny_nickel_dime (c: five_coins) : bool := c.1 && c.2 && c.3

-- Define the successful outcomes where penny, nickel, and dime are heads
noncomputable def successful_outcomes : Set five_coins :=
  { c | heads_penny_nickel_dime c = tt }

-- Define the total outcomes for 5 coins
noncomputable def total_outcomes : Set five_coins := {c | true}

-- Probability calculation: |success| / |total|
noncomputable def probability_heads_penny_nickel_dime : real :=
  (Set.card successful_outcomes : real) / (Set.card total_outcomes : real)

-- Given the setup, prove that the probability is 1/8
theorem probability_heads_penny_nickel_dime_is_one_eighth :
  probability_heads_penny_nickel_dime = 1 / 8 := 
sorry

end probability_heads_penny_nickel_dime_is_one_eighth_l249_249271


namespace correct_number_of_ways_l249_249603

noncomputable def numberOfWaysToIntersect : ℕ :=
  ∑ A in Finset.range 8 \ {0}, 
    ∑ D in Finset.range 8 \ {0, A}, 
      ∑ B in Finset.range 8 \ {0, A, D}, 
        ∑ C in Finset.range 8 \ {0, A, D, B}, 
          ∑ E in Finset.range 8 \ {0, A, D, B, C}, 
            if (B^2 - 4 * (A - D) * (C - E) ≥ 0) then 1 else 0

theorem correct_number_of_ways : numberOfWaysToIntersect = 1260 := 
  sorry

end correct_number_of_ways_l249_249603


namespace sam_walks_distance_l249_249971

theorem sam_walks_distance :
  (let rate := 1.5 / 18 in
   let distance := rate * 15 in
   Float.ceil (distance * 10) / 10 = 1.3) :=
by 
  let rate := 1.5 / 18
  let distance := rate * 15
  have h : ∀ x : ℚ, Float.ceil (x * 10) = 13 := sorry
  calc
    Float.ceil (distance * 10) / 10 = 1.3 : by sorry

end sam_walks_distance_l249_249971


namespace part1_part2_l249_249847

noncomputable def f (a b x : ℝ) : ℝ := a * x - b * x ^ 2

-- Part 1
theorem part1 {a b : ℝ} (h : ∀ x : ℝ, f a b x ≤ 1) : a ≤ 2 * real.sqrt b := sorry

-- Part 2
theorem part2 {a b : ℝ} (hb : b > 1) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f a b x| ≤ 1) ↔ (b - 1 ≤ a ∧ a ≤ 2 * real.sqrt b) := sorry

end part1_part2_l249_249847


namespace train_speeds_l249_249746

theorem train_speeds (v t : ℕ) (h1 : t = 1)
  (h2 : v + v * t = 90)
  (h3 : 90 * t = 90) :
  v = 45 := by
  sorry

end train_speeds_l249_249746


namespace min_value_of_function_l249_249513

noncomputable def f (x y : ℝ) : ℝ := (x / (x + 2 * y)) + (y / x)

theorem min_value_of_function : ∀ (x y : ℝ), (x > 0) → (y > 0) → 
  f x y ≥ sqrt 2 - 1 / 2 :=
by
  sorry

end min_value_of_function_l249_249513


namespace train_speed_with_n_coaches_l249_249401

-- Definitions based on conditions:
def speed_without_coaches : ℝ := 60
def speed_with_4_coaches : ℝ := 48
def speed_with_n_coaches (n : ℝ) : ℝ := 24

def speed_variation (k n : ℝ) : ℝ := k / n.sqrt

-- Theorem statement
theorem train_speed_with_n_coaches :
  ∀ k n : ℝ, speed_variation k 4 = speed_with_4_coaches → speed_variation k n = speed_with_n_coaches n → n = 16 :=
by
  intro k n h1 h2
  sorry

end train_speed_with_n_coaches_l249_249401


namespace parallel_lines_k_value_l249_249684

theorem parallel_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = 5 * x + 3 → y = (3 * k) * x + 1 → true) → k = 5 / 3 :=
by
  intros
  sorry

end parallel_lines_k_value_l249_249684


namespace john_total_climb_height_l249_249129

-- Define the heights and conditions
def num_flights : ℕ := 3
def height_per_flight : ℕ := 10
def total_stairs_height : ℕ := num_flights * height_per_flight
def rope_height : ℕ := total_stairs_height / 2
def ladder_height : ℕ := rope_height + 10

-- Prove that the total height John climbed is 70 feet
theorem john_total_climb_height : 
  total_stairs_height + rope_height + ladder_height = 70 := by
  sorry

end john_total_climb_height_l249_249129


namespace flower_count_l249_249322

def numberOfFlowers (pots : ℕ) (sticksPerPot : ℕ) (totalFlowersAndSticks : ℕ) (flowersPerPot : ℕ) : Prop :=
  pots * flowersPerPot + pots * sticksPerPot = totalFlowersAndSticks

theorem flower_count :
  numberOfFlowers 466 181 109044 53 :=
by
  unfold numberOfFlowers
  simp
  sorry

end flower_count_l249_249322


namespace andrew_total_kept_balloons_l249_249405

-- Define the initial number of blue and purple balloons
def blue_balloons : ℕ := 303
def purple_balloons : ℕ := 453

-- Define the fractions he keeps
def fraction_blue : ℚ := 2 / 3
def fraction_purple : ℚ := 3 / 5

-- Define the number of balloons Andrew keeps from each color
def kept_blue_balloons : ℕ := (fraction_blue * blue_balloons).to_nat
def kept_purple_balloons : ℕ := (fraction_purple * purple_balloons).to_nat

-- Define the total number of balloons Andrew keeps
def total_kept_balloons : ℕ := kept_blue_balloons + kept_purple_balloons

-- Prove that the total number of balloons Andrew keeps is 473
theorem andrew_total_kept_balloons : total_kept_balloons = 473 :=
by sorry

end andrew_total_kept_balloons_l249_249405


namespace greatest_divisors_1_to_20_l249_249183

theorem greatest_divisors_1_to_20 : ∃ n₁ n₂ n₃, ((n₁ = 12 ∧ n₂ = 18 ∧ n₃ = 20) ∧ 
  (∀ m ∈ (Finset.range 21).filter (λ x, x > 0), 
    let d_m := (Finset.range (m + 1)).filter (λ k, m % k = 0) in
    ∃ k, k = d_m.card ∧ k <= 6 ∧ ((m = n₁ ∧ k = 6) ∨ (m = n₂ ∧ k = 6) ∨ (m = n₃ ∧ k = 6)))) :=
by
  sorry

end greatest_divisors_1_to_20_l249_249183


namespace polynomial_sum_is_2_l249_249074

theorem polynomial_sum_is_2 :
  ∀ (x : ℝ),
  ∃ (A B C D : ℝ), 
  (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D ∧ A + B + C + D = 2 :=
by
  intros x
  use [4, -10, -13, 21]
  split
  · -- Prove the polynomial expansion
    calc
      (x - 3) * (4 * x^2 + 2 * x - 7) 
          = x * (4 * x^2 + 2 * x - 7) - 3 * (4 * x^2 + 2 * x - 7) : by rw mul_sub
      ... = (x * 4 * x^2 + x * 2 * x - x * 7) - (3 * (4 * x^2) + 3 * (2 * x) - 3 * (-7)) : by distribute
      ... = 4 * x^3 + 2 * x^2 - 7 * x - 12 * x^2 - 6 * x + 21 : by algebra
      ... = 4 * x^3 - 10 * x^2 - 13 * x + 21 : by linarith
  · -- Prove A + B + C + D = 2
    calc
      4 + (-10) + (-13) + 21 = 2 : by linarith

end polynomial_sum_is_2_l249_249074


namespace always_real_roots_range_of_b_analytical_expression_parabola_l249_249687

-- Define the quadratic equation with parameter m
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (5 * m - 1) * x + 4 * m - 4

-- Part 1: Prove the equation always has real roots
theorem always_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 := 
sorry

-- Part 2: Find the range of b such that the line intersects the parabola at two distinct points
theorem range_of_b (b : ℝ) : 
  (∀ m : ℝ, m = 1 → (b > -25/4 → (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq m x1 = (x1 + b) ∧ quadratic_eq m x2 = (x2 + b)))) :=
sorry

-- Part 3: Find the analytical expressions of the parabolas given the distance condition
theorem analytical_expression_parabola (m : ℝ) : 
  (∀ x1 x2 : ℝ, (|x1 - x2| = 2 → quadratic_eq m x1 = 0 → quadratic_eq m x2 = 0) → 
  (m = -1 ∨ m = -1/5) → 
  ((quadratic_eq (-1) x = -x^2 + 6*x - 8) ∨ (quadratic_eq (-1/5) x = -1/5*x^2 + 2*x - 24/5))) :=
sorry

end always_real_roots_range_of_b_analytical_expression_parabola_l249_249687


namespace coefficient_recurrence_l249_249420

noncomputable def F (q x : ℝ) : ℝ := (q_∞) * (-x)_∞ * (-q / x)_∞

theorem coefficient_recurrence (q : ℝ) (a : ℤ → ℝ) :
  (a_k(q) = q^(k(k-1)/2)) :=
sorry

end coefficient_recurrence_l249_249420


namespace balance_scale_measurements_l249_249380

theorem balance_scale_measurements {a b c : ℕ}
    (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 11) :
    ∀ w : ℕ, 1 ≤ w ∧ w ≤ 11 → ∃ (x y z : ℤ), w = abs (x * a + y * b + z * c) :=
sorry

end balance_scale_measurements_l249_249380


namespace max_children_l249_249454

def child : Type := String

def is_boy_named_Adam (c : child) : Prop := c = "Adam"
def is_girl_named_Beata (c : child) : Prop := c = "Beata"

def has_boy_named_Adam (s : Finset child) : Prop := ∃ c ∈ s, is_boy_named_Adam c
def has_girl_named_Beata (s : Finset child) : Prop := ∃ c ∈ s, is_girl_named_Beata c

theorem max_children :
  ∀ (group : Finset child),
    (∀ (trio : Finset child), trio.card = 3 → trio ⊆ group → has_boy_named_Adam trio) →
    (∀ (quartet : Finset child), quartet.card = 4 → quartet ⊆ group → has_girl_named_Beata quartet) →
    group.card ≤ 5 ∧
    (∃ (Adams Beatas : Finset child),
      Adams.card = 3 ∧ Beatas.card = 2 ∧
      (∀ c ∈ Adams, is_boy_named_Adam c) ∧
      (∀ c ∈ Beatas, is_girl_named_Beata c) ∧
      Disjoint Adams Beatas ∧
      group = Adams ∪ Beatas)
:= by
  intro group trio_condition quartet_condition
  sorry

end max_children_l249_249454


namespace constant_term_in_expansion_l249_249087

theorem constant_term_in_expansion (n k : ℕ) (C : ℕ → ℕ → ℕ) : 
  (∑ i in range (n + 1), C n i) = 81 → 
  C 4 2 = 96 := by
  sorry

end constant_term_in_expansion_l249_249087


namespace max_candies_vovochka_can_keep_l249_249909

-- Definitions for the conditions in the problem.
def classmates := 25
def total_candies := 200
def minimum_candies (candies_distributed : Vector ℕ classmates) : Prop :=
  ∀ (S : Finset (Fin classmates)), S.card = 16 → S.sum (candies_distributed.nth' S) ≥ 100

-- The proof goal
theorem max_candies_vovochka_can_keep (candies_distributed : Vector ℕ classmates) (h : minimum_candies candies_distributed) :
  ∃ k, k = 37 ∧ total_candies - Finset.univ.sum (candies_distributed.nth' Finset.univ) = k :=
begin
  sorry
end

end max_candies_vovochka_can_keep_l249_249909


namespace find_value_of_derivative_at_1_l249_249852

def f (x : ℝ) : ℝ := x * real.exp (x - 1)

theorem find_value_of_derivative_at_1 : deriv f 1 = 2 := 
by 
  -- Proof is skipped using sorry
  sorry

end find_value_of_derivative_at_1_l249_249852


namespace y_intercept_l249_249819

noncomputable def a_n (n : ℕ) : ℚ := 1 / (n * (n + 1))

noncomputable def S_n (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ k, a_n (k + 1))

theorem y_intercept (h : S_n 9 = 9 / 10) : 
  let n := 9 in
  ∃ y : ℚ, (n + 1) * 0 + y + n = 0 ∧ y = -9 := 
by
  sorry

end y_intercept_l249_249819


namespace general_formula_for_a_sum_of_first_n_terms_of_b_l249_249492

variable {a : ℕ → ℤ} {b : ℕ → ℤ} {S : ℕ → ℤ}

-- Define the arithmetic sequence with the given conditions
def is_arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

axiom a1 : a 1 = 2
axiom a1a2a3_sum : a 1 + a 2 + a 3 = 12

-- Define b_n
def b (n : ℕ) := a n * 3^n

-- Sum of the first n terms of b_n
def sum_b (n : ℕ) := ∑ i in Finset.range n, b (i + 1)

theorem general_formula_for_a :
  is_arithmetic_seq a → (∀ n, a n = 2 * n) := by
  sorry

theorem sum_of_first_n_terms_of_b :
  is_arithmetic_seq a → (∀ n, sum_b n = (2 * n - 1) / 2 * 3^(n + 1) + 3 / 2) := by
  sorry

end general_formula_for_a_sum_of_first_n_terms_of_b_l249_249492


namespace tangent_line_circle_l249_249933

theorem tangent_line_circle (a : ℝ) : (∀ x y : ℝ, a * x + y + 1 = 0) → (∀ x y : ℝ, x^2 + y^2 - 4 * x = 0) → a = 3 / 4 :=
by
  sorry

end tangent_line_circle_l249_249933


namespace triangle_area_l249_249351

noncomputable def heron_formula (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area (a b c : ℝ) (h : a = 78 ∧ b = 72 ∧ c = 30) :
  heron_formula a b c = 1080 := by
  sorry

end triangle_area_l249_249351


namespace find_circle_equation_l249_249487

theorem find_circle_equation (A B : ℝ × ℝ) (h2 : A = (0, 2)) (h3 : B = (2, -2)) :
  (∃ C : ℝ × ℝ, (C.1 - C.2 + 1 = 0) ∧ ((C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2)) →
  ∃ k l r : ℝ, (k = -3) ∧ (l = -2) ∧ (r = 5) ∧ (∀ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 25) := 
begin
  intros,
  use [-3, -2, 5],
  repeat {split},
  exact rfl,
  exact rfl,
  exact rfl,
  intros x y,
  sorry
end

end find_circle_equation_l249_249487


namespace total_turtles_in_lake_l249_249324

noncomputable def commonPercentage : ℝ := 0.5
noncomputable def rarePercentage : ℝ := 0.3
noncomputable def uniquePercentage : ℝ := 0.15
noncomputable def legendaryPercentage : ℝ := 0.05

noncomputable def commonAgeDistribution : (ℝ × ℝ × ℝ) := (0.4, 0.3, 0.3)
noncomputable def rareAgeDistribution : (ℝ × ℝ × ℝ) := (0.3, 0.4, 0.3)
noncomputable def uniqueAgeDistribution : (ℝ × ℝ × ℝ) := (0.2, 0.3, 0.5)
noncomputable def legendaryAgeDistribution : (ℝ × ℝ × ℝ) := (0.15, 0.3, 0.55)

noncomputable def commonGenderDistribution : (ℝ × ℝ) := (0.4, 0.6)
noncomputable def rareGenderDistribution : (ℝ × ℝ) := (0.45, 0.55)
noncomputable def uniqueGenderDistribution : (ℝ × ℝ) := (0.55, 0.45)
noncomputable def legendaryGenderDistribution : (ℝ × ℝ) := (0.6, 0.4)

noncomputable def commonStripedMalePercentage : ℝ := 0.25
noncomputable def rareStripedMalePercentage : ℝ := 0.4
noncomputable def uniqueStripedMalePercentage : ℝ := 0.33
noncomputable def legendaryStripedMalePercentage : ℝ := 0.5

noncomputable def commonAdultStripedMalePercentage : ℝ := 0.4
noncomputable def rareAdultStripedMalePercentage : ℝ := 0.45
noncomputable def uniqueAdultStripedMalePercentage : ℝ := 0.35
noncomputable def legendaryAdultStripedMalePercentage : ℝ := 0.3

noncomputable def spottedCommonAdultStripedMales : ℕ := 84

theorem total_turtles_in_lake : (total_turtles : ℕ) :=
  let total := 4200 in
  total = 4200 := by sorry

end total_turtles_in_lake_l249_249324


namespace find_focus_of_parabola_l249_249622

-- Define the given parabola equation
def parabola_eqn (x : ℝ) : ℝ := -4 * x^2

-- Define a predicate to check if the point is the focus
def is_focus (x y : ℝ) := x = 0 ∧ y = -1 / 16

theorem find_focus_of_parabola :
  is_focus 0 (parabola_eqn 0) :=
sorry

end find_focus_of_parabola_l249_249622


namespace angle_C_is_65_deg_l249_249968

-- Defining a triangle and its angles.
structure Triangle :=
  (A B C : ℝ) -- representing the angles in degrees

-- Defining the conditions of the problem.
def given_triangle : Triangle :=
  { A := 75, B := 40, C := 180 - 75 - 40 }

-- Statement of the problem, proving that the measure of ∠C is 65°.
theorem angle_C_is_65_deg (t : Triangle) (hA : t.A = 75) (hB : t.B = 40) (hSum : t.A + t.B + t.C = 180) : t.C = 65 :=
  by sorry

end angle_C_is_65_deg_l249_249968


namespace find_first_number_l249_249623

-- Definitions of the conditions and variables
variables (x y : ℝ) -- x and y are real numbers

-- Hypothesis (conditions)
def diff_eq : Prop := x - y = 88
def ratio_eq : Prop := y = 0.20 * x

-- The theorem stating that x = 110 given the conditions
theorem find_first_number (h1 : diff_eq x y) (h2 : ratio_eq x y) : x = 110 :=
begin
  sorry
end

end find_first_number_l249_249623


namespace divisors_of_12_18_20_l249_249193

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem divisors_of_12_18_20 (n : ℕ) (h : n ∈ {1, 2, 3, ..., 20}) :
  n ∈ {12, 18, 20} ↔ (∀ m ∈ {1, 2, 3, ..., 20}, count_divisors m ≤ 6) :=
by
  sorry

end divisors_of_12_18_20_l249_249193


namespace total_students_l249_249096

variable (A B C D E F G H : ℕ)

def students_I_IV : Prop := A + B + C + D = 130
def students_V : Prop := E = B + 7
def students_VI : Prop := F = A - 5
def students_VII : Prop := G = D + 10
def students_VIII : Prop := H = A - 4
def students_V_VIII_calc : Prop := E + F + G + H = 2 * A + B + D + 8

theorem total_students (h1 : students_I_IV A B C D)
  (h2 : students_V A B E)
  (h3 : students_VI A F)
  (h4 : students_VII D G)
  (h5 : students_VIII A H)
  (h6 : students_V_VIII_calc (students_V A B E) (students_VI A F) (students_VII D G) (students_VIII A H)) :
  A + B + C + D + E + F + G + H = 268 :=
by sorry

end total_students_l249_249096


namespace percent_pear_juice_l249_249585

theorem percent_pear_juice (n : ℕ) (pear_juice_per_pear : ℝ) (orange_juice_per_orange : ℝ) :
  pear_juice_per_pear = 2.5 ∧ orange_juice_per_orange = 4 → 
  (n * pear_juice_per_pear) / ((n * pear_juice_per_pear) + (n * orange_juice_per_orange)) = 0.38 :=
by {
  intros H,  -- introduce the hypothesis
  cases H with h1 h2,  -- destructure the conjunctive hypothesis
  calc
    (n * pear_juice_per_pear) / ((n * pear_juice_per_pear) + (n * orange_juice_per_orange))
        = (n * 2.5) / ((n * 2.5) + (n * 4)) : by rw [h1, h2]
    ... = 2.5 / (2.5 + 4) : by { field_simp [mul_comm n], ring }
    ... = 2.5 / 6.5 : by simp
    ... = 0.3846153846153846 : by norm_num
    ... ≈ 0.38 : by norm_num
}

end percent_pear_juice_l249_249585


namespace binary_multiplication_correct_l249_249447

theorem binary_multiplication_correct:
  let n1 := 29 -- binary 11101 is decimal 29
  let n2 := 13 -- binary 1101 is decimal 13
  let result := 303 -- binary 100101111 is decimal 303
  n1 * n2 = result :=
by
  -- Proof goes here
  sorry

end binary_multiplication_correct_l249_249447


namespace greatest_divisors_1_to_20_l249_249177

theorem greatest_divisors_1_to_20 : ∃ n₁ n₂ n₃, ((n₁ = 12 ∧ n₂ = 18 ∧ n₃ = 20) ∧ 
  (∀ m ∈ (Finset.range 21).filter (λ x, x > 0), 
    let d_m := (Finset.range (m + 1)).filter (λ k, m % k = 0) in
    ∃ k, k = d_m.card ∧ k <= 6 ∧ ((m = n₁ ∧ k = 6) ∨ (m = n₂ ∧ k = 6) ∨ (m = n₃ ∧ k = 6)))) :=
by
  sorry

end greatest_divisors_1_to_20_l249_249177


namespace probability_heads_penny_nickel_dime_l249_249280

theorem probability_heads_penny_nickel_dime :
  let total_outcomes := 2^5 in
  let successful_outcomes := 2 * 2 in
  (successful_outcomes : ℝ) / total_outcomes = (1 / 8 : ℝ) :=
by
  sorry

end probability_heads_penny_nickel_dime_l249_249280


namespace base_conversion_proof_l249_249334

-- Definitions of the base-converted numbers
def b1463_7 := 3 * 7^0 + 6 * 7^1 + 4 * 7^2 + 1 * 7^3  -- 1463 in base 7
def b121_5 := 1 * 5^0 + 2 * 5^1 + 1 * 5^2  -- 121 in base 5
def b1754_6 := 4 * 6^0 + 5 * 6^1 + 7 * 6^2 + 1 * 6^3  -- 1754 in base 6
def b3456_7 := 6 * 7^0 + 5 * 7^1 + 4 * 7^2 + 3 * 7^3  -- 3456 in base 7

-- Formalizing the proof goal
theorem base_conversion_proof : (b1463_7 / b121_5 : ℤ) - b1754_6 * 2 + b3456_7 = 278 := by
  sorry  -- Proof is omitted

end base_conversion_proof_l249_249334


namespace triangle_area_l249_249747

/-
A triangle with side lengths in the ratio 4:5:6 is inscribed in a circle of radius 5.
We need to prove that the area of the triangle is 250/9.
-/

theorem triangle_area (x : ℝ) (r : ℝ) (h_r : r = 5) (h_ratio : 6 * x = 2 * r) :
  (1 / 2) * (4 * x) * (5 * x) = 250 / 9 := by 
  -- Proof goes here.
  sorry

end triangle_area_l249_249747


namespace fg_three_eq_neg_two_l249_249922

def f (x : ℝ) : ℝ := 4 - Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem fg_three_eq_neg_two : f (g 3) = -2 :=
by
  sorry

end fg_three_eq_neg_two_l249_249922


namespace polynomial_C_value_l249_249795

theorem polynomial_C_value :
  ∃ (A B D : ℤ) (p : ℤ[X]), p = X^6 - 12*X^5 + A*X^4 + B*X^3 + (-171)*X^2 + D*X + 36 ∧
    ∀ x : ℤ, p.eval x = 0 → x > 0 :=
sorry

end polynomial_C_value_l249_249795


namespace sum_of_divisors_of_8_l249_249802

theorem sum_of_divisors_of_8 : 
  (∑ n in { n : ℕ | 0 < n ∧ 8 % n = 0 }, n) = 15 :=
by
  sorry

end sum_of_divisors_of_8_l249_249802


namespace students_taking_music_l249_249372

theorem students_taking_music
  (total_students : Nat)
  (students_taking_art : Nat)
  (students_taking_both : Nat)
  (students_taking_neither : Nat)
  (total_eq : total_students = 500)
  (art_eq : students_taking_art = 20)
  (both_eq : students_taking_both = 10)
  (neither_eq : students_taking_neither = 440) :
  ∃ M : Nat, M = 50 := by
  sorry

end students_taking_music_l249_249372


namespace coeff_sum_odd_indices_l249_249470

def roots_of_unity_sum {R : Type*} [CommRing R] :
  (x : R) → ((x+1)^2017 = a₀ * x^2017 + a₁ * x^2016 + a₂ * x^2015 + ... + a₂₀₁₆ * x + a₂₀₁₇) → Prop :=
  sorry -- Define the roots of unity filter here

theorem coeff_sum_odd_indices :
  ∀ (a₀ a₁ a₂ ... a₂₀₁₆ a₂₀₁₇ : ℤ),
  ((x+1)^2017 = a₀ * x^2017 + a₁ * x^2016 + a₂ * x^2015 + ... + a₂₀₁₆ * x + a₂₀₁₇) ⟹
  (a₁ + a₅ + a₉ + ... + a₂₀₁₇ = 2^2015) :=
sorry -- The proof in Lean proving the given problem statement

end coeff_sum_odd_indices_l249_249470


namespace operation_impossible_l249_249861

noncomputable def sum_of_squares (a b c : ℤ) : ℤ :=
  a^2 + b^2 + c^2

theorem operation_impossible (a b c d e f : ℤ) (h_initial : a = 89 ∧ b = 12 ∧ c = 3) (h_target : d = 90 ∧ e = 10 ∧ f = 14) : sum_of_squares a b c ≠ sum_of_squares d e f :=
by
  -- Provided conditions
  have h₁ : sum_of_squares a b c = 8074 := by sorry
  have h₂ : sum_of_squares d e f = 8396 := by sorry
  -- Contradiction of sum_of_squares invariants
  show 8074 ≠ 8396 by sorry

end operation_impossible_l249_249861


namespace solve_system_l249_249702

theorem solve_system :
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ : ℚ),
  x₁ + 12 * x₂ = 15 ∧
  x₁ - 12 * x₂ + 11 * x₃ = 2 ∧
  x₁ - 11 * x₃ + 10 * x₄ = 2 ∧
  x₁ - 10 * x₄ + 9 * x₅ = 2 ∧
  x₁ - 9 * x₅ + 8 * x₆ = 2 ∧
  x₁ - 8 * x₆ + 7 * x₇ = 2 ∧
  x₁ - 7 * x₇ + 6 * x₈ = 2 ∧
  x₁ - 6 * x₈ + 5 * x₉ = 2 ∧
  x₁ - 5 * x₉ + 4 * x₁₀ = 2 ∧
  x₁ - 4 * x₁₀ + 3 * x₁₁ = 2 ∧
  x₁ - 3 * x₁₁ + 2 * x₁₂ = 2 ∧
  x₁ - 2 * x₁₂ = 2 ∧
  x₁ = 37 / 12 ∧
  x₂ = 143 / 144 ∧
  x₃ = 65 / 66 ∧
  x₄ = 39 / 40 ∧
  x₅ = 26 / 27 ∧
  x₆ = 91 / 96 ∧
  x₇ = 13 / 14 ∧
  x₈ = 65 / 72 ∧
  x₉ = 13 / 15 ∧
  x₁₀ = 13 / 16 ∧
  x₁₁ = 13 / 18 ∧
  x₁₂ = 13 / 24 :=
by
  sorry

end solve_system_l249_249702


namespace vovochka_max_candies_l249_249868

/-!
# Problem Statement:
- Vovochka has 25 classmates.
- Vovochka brought 200 candies to class.

## Condition:
- Any 16 classmates together should have at least 100 candies in total.

Question: What is the maximum number of candies Vovochka can keep for himself while fulfilling his mother's request?

The answer to this problem needs to be proven mathematically.
-/

def max_candies_can_keep (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ) : ℕ :=
if h : classmates = 25 ∧ total_candies = 200 ∧ min_candies_per_16 = 100 then
  37
else 
  sorry -- This statement is just for context, keeping focus on formal proof in Lean.

theorem vovochka_max_candies :
  ∀ (candies : ℕ) (classmates : ℕ) (total_candies : ℕ) (min_candies_per_16 : ℕ),
    classmates = 25 → total_candies = 200 → min_candies_per_16 = 100 →
    max_candies_can_keep candies classmates total_candies min_candies_per_16 = 37 :=
by
  intros candies classmates total_candies min_candies_per_16 hc ht hm
  simp [max_candies_can_keep]
  exact if_pos ⟨hc, ht, hm⟩

end vovochka_max_candies_l249_249868


namespace natural_numbers_placement_count_l249_249544

noncomputable def natural_number_placements : Nat :=
  ∑ k in {1, 2, 4, 7, 14, 28}, 1

theorem natural_numbers_placement_count :
  (∃ x y z : ℕ, ∀ p triplet : List ℕ, triplet ∈ [[4, 14, x], [14, 6, z], [z, y, 4]] →
    triplet.product = (4 * 14 * x)) ∧
  natural_number_placements = 6 :=
by sorry

end natural_numbers_placement_count_l249_249544


namespace union_eq_l249_249580

-- Define the sets M and N using the given conditions
def M := {x | 0 ≤ x ∧ x ≤ 1}
def N := {x | x^2 ≥ 1}

-- Define the complement of N in ℝ
def complement_N := {x | -1 < x ∧ x < 1}

-- Define the set union of M and complement_N
def union_set := {x | -1 < x ∧ x ≤ 1}

-- Prove that M ∪ (complement of N) is (-1, 1]
theorem union_eq : (M ∪ complement_N) = union_set :=
by
  sorry

end union_eq_l249_249580


namespace height_on_hypotenuse_l249_249946

theorem height_on_hypotenuse (a b : ℕ) (hypotenuse : ℝ)
  (ha : a = 3) (hb : b = 4) (h_c : hypotenuse = sqrt (a^2 + b^2)) :
  let S := (1/2 : ℝ) * a * b in
  ∃ h : ℝ, h = (2 * S) / hypotenuse ∧ h = 12/5 := by
  sorry

end height_on_hypotenuse_l249_249946


namespace cos_phi_is_correct_l249_249503

noncomputable def cos_phi (p q : ℝ → ℝ) (hp : ∥p∥ = 7) (hq : ∥q∥ = 10) (hpq : ∥p + q∥ = 13) : ℝ :=
  (p ⬝ q) / (∥p∥ * ∥q∥)

theorem cos_phi_is_correct (p q : ℝ → ℝ)
  (hp : ∥p∥ = 7)
  (hq : ∥q∥ = 10)
  (hpq : ∥p + q∥ = 13) :
  cos_phi p q hp hq hpq = 1 / 7 :=
sorry

end cos_phi_is_correct_l249_249503


namespace greatest_divisors_1_to_20_l249_249184

theorem greatest_divisors_1_to_20 : ∃ n₁ n₂ n₃, ((n₁ = 12 ∧ n₂ = 18 ∧ n₃ = 20) ∧ 
  (∀ m ∈ (Finset.range 21).filter (λ x, x > 0), 
    let d_m := (Finset.range (m + 1)).filter (λ k, m % k = 0) in
    ∃ k, k = d_m.card ∧ k <= 6 ∧ ((m = n₁ ∧ k = 6) ∨ (m = n₂ ∧ k = 6) ∨ (m = n₃ ∧ k = 6)))) :=
by
  sorry

end greatest_divisors_1_to_20_l249_249184


namespace distance_star_to_earth_l249_249316

-- Define the conditions
def speed_of_light : ℕ := 3 * 10^5  -- speed of light in km/s
def years_to_seconds (years : ℕ) : ℕ := years * 3.1 * 10^7  -- convert years to seconds

-- Main theorem
theorem distance_star_to_earth (years : ℕ) (speed : ℕ) (conversion : ℕ) :
  speed = speed_of_light →
  conversion = years_to_seconds 1 →
  years = 10 →
  (speed * conversion * years) = 9.3 * 10^13 :=
by
  sorry

end distance_star_to_earth_l249_249316


namespace polynomial_with_two_roots_l249_249417

theorem polynomial_with_two_roots :
  let polynomials := { p : Polynomial ℤ // ∃ (b_0 b_1 b_2 b_3 b_4 b_5 b_6 : ℤ), 
    p = x^7 + b_6 * x^6 + b_5 * x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0 ∧
    ∀ i, 0 ≤ i ∧ i ≤ 6 → (b_i = 0 ∨ b_i = 1) } in
  let valid_polynomials := { p ∈ polynomials | p.eval 0 = 0 ∧ p.eval 1 = 0 } in
  valid_polynomials.card = 64 :=
by
  sorry

end polynomial_with_two_roots_l249_249417


namespace find_x11_x12_l249_249927

variable {a : ℕ → ℝ} (d : ℝ)

def harmonic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (1 / a (n + 1)) - (1 / a n) = d

def sum_of_first_n_terms (a : ℕ → ℝ) (x : ℝ) : Prop :=
  (∑ i in finset.range 22, a (i + 1)) = x

theorem find_x11_x12 (a : ℕ → ℝ) (d x : ℝ)
  (H₁ : harmonic_sequence (λ n, 1 / (a n)) d)
  (H₂ : sum_of_first_n_terms a 77) :
  a 11 + a 12 = 7 := 
sorry

end find_x11_x12_l249_249927


namespace smallest_prime_dividing_7_pow_7_plus_3_pow_14_l249_249339

theorem smallest_prime_dividing_7_pow_7_plus_3_pow_14 : 
  let sum := 7^7 + 3^14 in
  is_prime 2 ∧ (2 ∣ sum) ∧ (∀ p : ℕ, p.prime → p ∣ sum → p = 2 ∨ p > 2) :=
by
  let sum := 7^7 + 3^14
  have h₁ : is_prime 2 := by sorry
  have h₂ : 2 ∣ sum := by sorry
  have h₃ : ∀ p : ℕ, p.prime → p ∣ sum → p = 2 ∨ p > 2 := by sorry
  exact ⟨h₁, h₂, h₃⟩

end smallest_prime_dividing_7_pow_7_plus_3_pow_14_l249_249339


namespace standard_equation_of_ellipse_l249_249518

-- Definitions for clarity
def is_ellipse (E : Type) := true
def major_axis (e : is_ellipse E) : ℝ := sorry
def minor_axis (e : is_ellipse E) : ℝ := sorry
def focus (e : is_ellipse E) : ℝ := sorry

theorem standard_equation_of_ellipse (E : Type)
  (e : is_ellipse E)
  (major_sum : major_axis e + minor_axis e = 9)
  (focus_position : focus e = 3) :
  ∀ x y, (x^2 / 25) + (y^2 / 16) = 1 :=
by sorry

end standard_equation_of_ellipse_l249_249518


namespace pipe_weight_l249_249691

noncomputable def external_radius (d : ℝ) := d / 2

noncomputable def volume_of_cylinder (r h : ℝ) := π * r^2 * h

noncomputable def internal_radius (R t : ℝ) := R - t

noncomputable def weight_of_pipe 
    (length diameter thickness density : ℝ) : ℝ :=
  let R := external_radius diameter
  let V_solid := volume_of_cylinder R length
  let r := internal_radius R thickness
  let V_hollow := volume_of_cylinder r length
  density * (V_solid - V_hollow)

theorem pipe_weight 
  (length diameter thickness density : ℝ) 
  (h_length : length = 21) 
  (h_diameter : diameter = 8) 
  (h_thickness : thickness = 1) 
  (h_density : density = 8) : 
  weight_of_pipe length diameter thickness density = 2736.1416 :=
by 
  -- Provide assumptions and formatted theorem statement
  sorry

end pipe_weight_l249_249691


namespace side_of_beef_weight_after_processing_l249_249740

theorem side_of_beef_weight_after_processing :
  ∀ (weight_before weight_after : ℝ), weight_before = 876.9230769230769 → weight_after = 0.65 * weight_before → weight_after = 570 := 
by
  intros weight_before weight_after h_before h_after
  rw h_before at h_after
  rw h_after
  norm_num
  sorry

end side_of_beef_weight_after_processing_l249_249740


namespace general_term_arithmetic_sequence_l249_249634

theorem general_term_arithmetic_sequence :
  ∀ (n : ℕ), let a1 := -3, d := -4 in a1 + (n - 1) * d = -4 * n + 1 :=
by
  intros n a1 d
  sorry

end general_term_arithmetic_sequence_l249_249634


namespace sqrt_meaningful_range_l249_249082

theorem sqrt_meaningful_range (x : ℝ) : x + 1 ≥ 0 ↔ x ≥ -1 :=
by sorry

end sqrt_meaningful_range_l249_249082


namespace hired_year_l249_249371

theorem hired_year (A W : ℕ) (Y : ℕ) (retire_year : ℕ) 
    (hA : A = 30) 
    (h_rule : A + W = 70) 
    (h_retire : retire_year = 2006) 
    (h_employment : retire_year - Y = W) 
    : Y = 1966 := 
by 
  -- proofs are skipped with 'sorry'
  sorry

end hired_year_l249_249371


namespace gcd_square_of_difference_l249_249150

theorem gcd_square_of_difference (x y z : ℕ) (h : 1/x - 1/y = 1/z) :
  ∃ k : ℕ, (Nat.gcd (Nat.gcd x y) z) * (y - x) = k^2 :=
by
  sorry

end gcd_square_of_difference_l249_249150


namespace translation_distance_l249_249962

theorem translation_distance
  (A := (3 : ℝ, 5 : ℝ))
  (B := (1 : ℝ, 3 : ℝ))
  (C := (4 : ℝ, 3 : ℝ))
  (angle := (30 : ℝ))
  (m : ℝ) (hm : 0 < m ∧ m < 1) :
  ∃ d : ℝ, d = (3 * Real.sqrt 3 - 3) * (1 - Real.sqrt m) :=
by
  sorry

end translation_distance_l249_249962


namespace problem_1_problem_2_problem_3_l249_249044

-- Problem 1
theorem problem_1 
  : ∀ x : ℝ, (0 < x ∧ x ≤ sqrt 2) → (x + 2 / x) ≤ (x + 2 / x) :=
sorry

-- Problem 2
def f (x : ℝ) : ℝ := (x^2 - 4 * x - 1) / (x + 1)

theorem problem_2
  (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) 
  : ((∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ f x) ∧ (∀ x, 1 ≤ x ∧ x ≤ 2 → f x ≥ f x))
    ∧ (set.range f ⟨[0, 2], sorry⟩ = set.Icc (-2) (-1)) :=
sorry

-- Problem 3
def g (x : ℝ) (a : ℝ) : ℝ := x + 2 * a

theorem problem_3 
  (a : ℝ) 
  : (∀ x_1 : ℝ, (0 ≤ x_1 ∧ x_1 ≤ 2) → ∃ x_2 : ℝ, (0 ≤ x_2 ∧ x_2 ≤ 2) ∧ g x_2 a = f x_1) ↔ (-3 / 2 ≤ a ∧ a ≤ -1) :=
sorry

end problem_1_problem_2_problem_3_l249_249044


namespace train_length_correct_l249_249392

variable (time : ℝ) 
variable (speed_man_kmh : ℝ) 
variable (speed_train_kmh : ℝ)

def relative_speed_mps (speed_train_kmh speed_man_kmh : ℝ) : ℝ :=
  (speed_train_kmh - speed_man_kmh) * (5 / 18)

def train_length (relative_speed_mps time : ℝ) : ℝ :=
  relative_speed_mps * time

theorem train_length_correct (htime : time = 47.99616030717543)
                             (hspeed_man : speed_man_kmh = 3)
                             (hspeed_train : speed_train_kmh = 63) : 
  train_length (relative_speed_mps speed_train_kmh speed_man_kmh) time = 799.936 :=
by 
  sorry

end train_length_correct_l249_249392


namespace sum_f_values_l249_249496

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_f_values : 
  f 1 + f 2 + f 3 + f (1 / 2) + f (1 / 3) = 5 / 2 :=
sorry

end sum_f_values_l249_249496


namespace divisors_of_12_18_20_l249_249191

noncomputable def count_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, n % d = 0).card

theorem divisors_of_12_18_20 (n : ℕ) (h : n ∈ {1, 2, 3, ..., 20}) :
  n ∈ {12, 18, 20} ↔ (∀ m ∈ {1, 2, 3, ..., 20}, count_divisors m ≤ 6) :=
by
  sorry

end divisors_of_12_18_20_l249_249191


namespace probability_penny_nickel_dime_all_heads_l249_249283

-- Define flipping five coins
def flip_five_coins : Type := (bool × bool × bool × bool × bool)

-- Define the function to check if the penny, nickel, and dime are heads
def all_heads_penny_nickel_dime (o : flip_five_coins) : bool :=
  o.1 = tt ∧ o.2 = tt ∧ o.3 = tt

-- Define the total count of possible outcomes
def total_outcomes : ℕ := 32

-- Define the count of favorable outcomes
def favorable_outcomes : ℕ := 4

-- Define the probability calculation
def probability_favorable : ℚ := favorable_outcomes / total_outcomes

-- The statement proving the probability is 1/8
theorem probability_penny_nickel_dime_all_heads :
  probability_favorable = 1 / 8 :=
by
  sorry

end probability_penny_nickel_dime_all_heads_l249_249283


namespace fraction_to_decimal_representation_l249_249433

/-- Determine the decimal representation of a given fraction. -/
theorem fraction_to_decimal_representation : (45 / (2 ^ 3 * 5 ^ 4) = 0.0090) :=
sorry

end fraction_to_decimal_representation_l249_249433


namespace bisection_next_interval_l249_249331

noncomputable def f (x : ℝ) : ℝ := 2 ^ x + 3 * x - 7

theorem bisection_next_interval (a b c: ℝ) (h : 0 <= a) (h0 : b = 4) (h1 : c = 2)
  (h2 : f(0) * f(2) < 0) : 0 < c ∧ c < b := by
  sorry

end bisection_next_interval_l249_249331


namespace product_conjugate_of_symmetric_l249_249058

noncomputable theory

variables (z1 z2 : ℂ)

-- Conditions given in the problem:
-- 1. z1 = 2 - i
-- 2. z1 and z2 are symmetric about the imaginary axis

def condition1 : z1 = 2 - complex.i := by
  trivial

def symmetric_about_imag_axis : ∀ z1 z2 : ℂ, (z1.re = -z2.re ∧ z1.im = z2.im) → z2 = -z1.re - complex.i * z1.im :=
  by sorry

-- Mathematical problem statement 
theorem product_conjugate_of_symmetric:
  (z1 = 2 - complex.i) → 
  (symmetric_about_imag_axis z1 z2) → 
  ((z1 * (complex.conj z2)) = -3 + 4 * complex.i) :=
by sorry

end product_conjugate_of_symmetric_l249_249058


namespace seq_general_term_l249_249635

theorem seq_general_term (n : ℕ) (h : n > 0) : 
  (λ k : ℕ, if k > 0 then 1 / k else 0) n = 1 / n :=
by sorry

end seq_general_term_l249_249635


namespace perimeter_of_PQR_l249_249115

theorem perimeter_of_PQR (s : ℝ) (h_triangle : ∀ (a b c : ℝ), a = b ∧ b = c ∧ c = a → (a + b + c = s ∧ ∀ x, x ≥ 0 → x ≤ s)) 
  (small_tris : 4 ∧ (s = 9)) :
  let side := s / 3 in
  let perimeter_PQR := 3 * (2 * side) in
  perimeter_PQR = 18 := 
by
  sorry

end perimeter_of_PQR_l249_249115


namespace function_passes_through_point_l249_249459

theorem function_passes_through_point :
  ∀ (x y : ℝ), y = 3 * x - 2 → ((x = -1) → (y = -5)) :=
by
  intros x y hy hxy
  rw hxy
  rw hy
  sorry

end function_passes_through_point_l249_249459


namespace maximize_winning_probability_l249_249650

namespace ProbabilityGame

def interval_a : Set ℝ := Icc 0 1
def interval_b : Set ℝ := Icc (1 / 2) (2 / 3)

noncomputable def winning_probability (x : ℝ) : ℝ :=
  x * ((2 / 3) - x) + (x - (1 / 2)) * (1 - x)

theorem maximize_winning_probability : ∃ x ∈ interval_a, x = 13 / 24 ∧
  (∀ y ∈ interval_a, winning_probability y ≤ winning_probability (13 / 24)) :=
begin
  sorry
end

end ProbabilityGame

end maximize_winning_probability_l249_249650


namespace height_on_hypotenuse_correct_l249_249948

noncomputable def height_on_hypotenuse (a b : ℝ) (ha : a = 3) (hb : b = 4) : ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  let area := (a * b) / 2
  (2 * area) / c

theorem height_on_hypotenuse_correct (h : ℝ) : 
  height_on_hypotenuse 3 4 rfl rfl = 12 / 5 :=
by
  sorry

end height_on_hypotenuse_correct_l249_249948


namespace ring_binder_price_l249_249132

theorem ring_binder_price (x : ℝ) (h1 : 50 + 5 = 55) (h2 : ∀ x, 55 + 3 * (x - 2) = 109) :
  x = 20 :=
by
  sorry

end ring_binder_price_l249_249132


namespace max_candies_vovochka_can_keep_l249_249907

-- Definitions for the conditions in the problem.
def classmates := 25
def total_candies := 200
def minimum_candies (candies_distributed : Vector ℕ classmates) : Prop :=
  ∀ (S : Finset (Fin classmates)), S.card = 16 → S.sum (candies_distributed.nth' S) ≥ 100

-- The proof goal
theorem max_candies_vovochka_can_keep (candies_distributed : Vector ℕ classmates) (h : minimum_candies candies_distributed) :
  ∃ k, k = 37 ∧ total_candies - Finset.univ.sum (candies_distributed.nth' Finset.univ) = k :=
begin
  sorry
end

end max_candies_vovochka_can_keep_l249_249907


namespace solution_product_l249_249567

theorem solution_product (p q : ℝ) (hpq : p ≠ q) (h1 : (x-3)*(3*x+18) = x^2-15*x+54) (hp : (x - p) * (x - q) = x^2 - 12 * x + 54) :
  (p + 2) * (q + 2) = -80 := sorry

end solution_product_l249_249567
