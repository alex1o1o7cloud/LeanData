import Mathlib
import Mathlib.Algebra.Factorial.Basic
import Mathlib.Analysis.Analytic.Basic
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Date.Basic
import Mathlib.Data.Fact
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Prime
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace banana_arrangements_l578_578891

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578891


namespace functional_relationship_selling_price_800_yuan_maximizing_profit_l578_578141

section Chengdu_Universiade

variables (x y : ℕ)

/-- Condition: The daily sales volume y is a linear function of the selling price x -/
def sales_function (x : ℕ) : ℕ := -2 * x + 160

/-- Condition: The cost price is 30 yuan and daily earnings is 800 yuan -/
def daily_earnings (x : ℕ) : ℤ := (x - 30 : ℕ) * (sales_function x : ℕ) - 800

/-- Part 1: Proof that sales_function gives the correct relationship -/
theorem functional_relationship : y = -2 * x + 160 := sorry

/-- Part 2: Proof that selling price 40 yuan gives 800 yuan earnings per day -/
theorem selling_price_800_yuan : ∃ x : ℕ, 33 ≤ x ∧ x ≤ 58 ∧ daily_earnings x = 0 := sorry

/-- Part 3: Proof that selling price 55 yuan maximizes the daily profit -/
theorem maximizing_profit :
  ∃ w_max : ℕ, ∀ x : ℕ, 33 ≤ x ∧ x ≤ 58 → 
  (x ≠ 55 → profit_func x < 1250) ∧ profit_func 55 = 1250 := sorry

end Chengdu_Universiade

end functional_relationship_selling_price_800_yuan_maximizing_profit_l578_578141


namespace BANANA_arrangement_l578_578459

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578459


namespace distinct_permutations_BANANA_l578_578655

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578655


namespace positive_multiple_with_four_digits_l578_578130

/-- 
  The statement asserts that for every positive integer n greater than 1, 
  there exists a positive multiple of n that is less than n^4 
  and uses at most 4 different digits.
-/
theorem positive_multiple_with_four_digits (n : ℕ) (h : n > 1) :
  ∃ k : ℕ, k > 0 ∧ k < n^4 ∧ ∃ (digits : Finset ℕ), digits.card ≤ 4 ∧ ∀ d ∈ (finset.range (k.digits 10).length), (k.digits 10).get d ∈ digits :=
sorry

end positive_multiple_with_four_digits_l578_578130


namespace numberOfWaysToArrangeBANANA_l578_578729

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578729


namespace banana_unique_permutations_l578_578856

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578856


namespace banana_arrangements_l578_578617

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578617


namespace arrange_banana_l578_578605

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578605


namespace banana_arrangements_l578_578906

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578906


namespace permutations_of_BANANA_l578_578381

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578381


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l578_578068

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l578_578068


namespace permutations_BANANA_l578_578553

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578553


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l578_578075

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l578_578075


namespace length_of_second_train_correct_l578_578179

noncomputable def length_of_second_train : ℝ :=
  let speed_first_train := 60 / 3.6
  let speed_second_train := 90 / 3.6
  let relative_speed := speed_first_train + speed_second_train
  let time_to_clear := 6.623470122390208
  let total_distance := relative_speed * time_to_clear
  let length_first_train := 111
  total_distance - length_first_train

theorem length_of_second_train_correct :
  length_of_second_train = 164.978 :=
by
  unfold length_of_second_train
  sorry

end length_of_second_train_correct_l578_578179


namespace total_visible_combinations_l578_578240

def faces_visible_from_point (faces : ℕ) : ℕ :=
  if faces = 3 then 8
  else if faces = 2 then 12
  else if faces = 1 then 6
  else 0

theorem total_visible_combinations : ℕ :=
  faces_visible_from_point 3 + faces_visible_from_point 2 + faces_visible_from_point 1 = 26 :=
by
  sorry

end total_visible_combinations_l578_578240


namespace digit_8_count_1_to_800_l578_578012

theorem digit_8_count_1_to_800 : (count_digit 8 (range 1 801)) = 160 :=
sorry

-- Assuming helper functions are already defined:
-- count_digit d lst: counts how many times the digit d appears in the list of integers lst.
-- range a b: generates a list of integers from a to b-1.

end digit_8_count_1_to_800_l578_578012


namespace permutations_BANANA_l578_578930

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578930


namespace number_of_arrangements_BANANA_l578_578779

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578779


namespace distinct_permutations_BANANA_l578_578669

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578669


namespace banana_arrangements_l578_578972

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578972


namespace find_S11_l578_578018

-- Define the notion of an arithmetic sequence
def arithmetic_seq (a₁ d : ℕ → ℕ) (n : ℕ) : ℕ :=
a₁ + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_seq (a₁ d : ℕ → ℕ) (n : ℕ) : ℕ :=
n * a₁ + (n * (n - 1) / 2) * d

-- Conditions and theorem
theorem find_S11 (a₁ d : ℕ → ℕ) (h : sum_arithmetic_seq a₁ d 8 - sum_arithmetic_seq a₁ d 3 = 10) : 
  sum_arithmetic_seq a₁ d 11 = 22 :=
sorry

end find_S11_l578_578018


namespace BANANA_arrangements_l578_578998

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l578_578998


namespace arrangement_count_BANANA_l578_578694

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578694


namespace number_of_arrangements_l578_578499

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578499


namespace banana_unique_permutations_l578_578870

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578870


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l578_578073

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l578_578073


namespace number_of_ways_to_arrange_BANANA_l578_578813

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578813


namespace arrange_banana_l578_578580

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578580


namespace number_of_ways_to_arrange_BANANA_l578_578809

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578809


namespace right_triangle_side_lengths_l578_578196

/-- 
Given four sets of three numbers:
Set A: {3, 5, 7},
Set B: {6, 8, 10},
Set C: {5, 12, 13},
Set D: {1, sqrt(3), 2},

Prove that set A cannot form the side lengths of a right triangle, 
while sets B, C, and D can.
-/
theorem right_triangle_side_lengths :
  (∀ a b c : ℕ, (a = 3 ∧ b = 5 ∧ c = 7) → a^2 + b^2 ≠ c^2) ∧
  (∀ a b c : ℕ, (a = 6 ∧ b = 8 ∧ c = 10) → a^2 + b^2 = c^2) ∧
  (∀ a b c : ℕ, (a = 5 ∧ b = 12 ∧ c = 13) → a^2 + b^2 = c^2) ∧
  (∀ a b c : ℝ, (a = 1 ∧ b = sqrt 3 ∧ c = 2) → a^2 + b^2 = c^2) :=
by
  sorry

end right_triangle_side_lengths_l578_578196


namespace BANANA_arrangement_l578_578477

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578477


namespace permutations_BANANA_l578_578938

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578938


namespace coin_ratio_l578_578230

theorem coin_ratio (coins_1r coins_50p coins_25p : ℕ) (value_1r value_50p value_25p : ℕ) :
  coins_1r = 120 → coins_50p = 120 → coins_25p = 120 →
  value_1r = coins_1r * 1 → value_50p = coins_50p * 50 → value_25p = coins_25p * 25 →
  value_1r + value_50p + value_25p = 210 →
  (coins_1r : ℚ) / (coins_50p + coins_25p : ℚ) = (1 / 1) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end coin_ratio_l578_578230


namespace distinct_permutations_BANANA_l578_578650

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578650


namespace arrangements_of_BANANA_l578_578411

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578411


namespace permutations_of_BANANA_l578_578389

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578389


namespace largest_cube_side_length_largest_rect_parallelepiped_dimensions_l578_578082

variables {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (a): Side length of the largest cube
theorem largest_cube_side_length :
  let a₀ := a * b * c / (a * b + b * c + a * c) in
  ∃ a₀, a₀ = a * b * c / (a * b + b * c + a * c) :=
begin
  sorry
end

-- Part (b): Dimensions of the largest rectangular parallelepiped
theorem largest_rect_parallelepiped_dimensions :
  let x := a / 3, y := b / 3, z := c / 3 in
  ∃ x y z, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
begin
  sorry
end

end largest_cube_side_length_largest_rect_parallelepiped_dimensions_l578_578082


namespace arrangements_of_BANANA_l578_578414

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578414


namespace permutations_BANANA_l578_578558

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578558


namespace banana_arrangements_l578_578893

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578893


namespace arrange_banana_l578_578587

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578587


namespace time_after_3333_minutes_l578_578194

-- Definition of the problem's conditions
def starting_time := (year : Nat, month : Nat, day : Nat, hour : Nat, minute : Nat) := 
  (2020, 2, 29, 12, 0)  -- Noon on February 29, 2020

def minutes_after : Nat := 3333

-- Translated definition and theorem
theorem time_after_3333_minutes : 
  time_after minutes_after starting_time = (2020, 3, 2, 19, 33) := 
sorry

end time_after_3333_minutes_l578_578194


namespace binomial_60_3_eq_34220_l578_578299

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l578_578299


namespace number_of_ways_to_arrange_BANANA_l578_578814

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578814


namespace max_distinct_integers_l578_578038

theorem max_distinct_integers (M : ℕ → ℕ → ℕ) (h_row : ∀ i : ℕ, i < 16 → (finset.univ.image (M i)).card ≤ 4)
    (h_col : ∀ j : ℕ, j < 16 → (finset.univ.image (λ i, M i j)).card ≤ 4) : (finset.univ.bUnion (λ i, finset.univ.image (λ j, M i j))).card ≤ 49 :=
sorry

end max_distinct_integers_l578_578038


namespace number_of_unique_permutations_BANANA_l578_578352

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578352


namespace permutations_BANANA_l578_578560

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578560


namespace number_of_arrangements_BANANA_l578_578798

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578798


namespace arrange_banana_l578_578604

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578604


namespace BANANA_arrangement_l578_578462

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578462


namespace banana_arrangements_l578_578971

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578971


namespace number_of_unique_permutations_BANANA_l578_578339

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578339


namespace binomial_60_3_eq_34220_l578_578296

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l578_578296


namespace max_people_on_rectangular_table_l578_578178

/-- 
Twelve people can sit evenly spaced around a large square table. 
Eight square tables are arranged in a row to form one long rectangular table.
Prove that the maximum number of people that can sit evenly spaced around the long rectangular table is 60.
--/
theorem max_people_on_rectangular_table : 
  (∀ t : ℕ, t = 12 → evenly_spaced_around_one_square_table t → 
  (∃ total_people : ℕ, total_people = 60 ∧ evenly_spaced_around_long_table (8 * t) total_people)) := sorry

end max_people_on_rectangular_table_l578_578178


namespace find_f_of_pi_by_4_l578_578027

theorem find_f_of_pi_by_4 (ω : ℝ) (φ : ℝ) (hω : ω > 0) (hφ1 : |φ| < π / 2) 
  (h_mono : ∀ x y : ℝ, (x ∈ Icc (π / 6) (2 * π / 3)) → (y ∈ Icc (π / 6) (2 * π / 3)) → x < y → sin(ω * y + φ) < sin(ω * x + φ))
  (h_fx1 : sin(ω * (π / 6) + φ) = 1) (h_fx2 : sin(ω * (2 * π / 3) + φ) = -1) :
  sin(ω * (π / 4) + φ) = sqrt 3 / 2 := 
by {
  sorry
}

end find_f_of_pi_by_4_l578_578027


namespace permutations_BANANA_l578_578535

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578535


namespace number_of_arrangements_BANANA_l578_578782

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578782


namespace permutations_BANANA_l578_578922

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578922


namespace banana_arrangements_l578_578913

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578913


namespace remaining_geom_body_volume_l578_578249

theorem remaining_geom_body_volume 
    (a h1 h2 h3 : ℝ) 
    (h1_h3_eq_2h2 : h1 + h3 = 2 * h2) :
    let S := (sqrt 3 / 4) * a^2 in
    S * h2 / 2 = sqrt 3 / 4 * a^2 * h2 :=
by
    sorry

end remaining_geom_body_volume_l578_578249


namespace arrangement_count_BANANA_l578_578687

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578687


namespace number_of_ways_to_arrange_BANANA_l578_578805

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578805


namespace percentage_of_engineering_students_passed_l578_578206

def total_students := 220
def male_students := 120
def female_students := 100
def percent_male_engineering_students := 0.25
def percent_female_engineering_students := 0.20
def percent_male_engineering_passed := 0.20
def percent_female_engineering_passed := 0.25

def male_engineering_students := percent_male_engineering_students * male_students
def female_engineering_students := percent_female_engineering_students * female_students

def male_engineering_passed := percent_male_engineering_passed * male_engineering_students
def female_engineering_passed := percent_female_engineering_passed * female_engineering_students

def total_engineering_passed := male_engineering_passed + female_engineering_passed
def total_engineering_students := male_engineering_students + female_engineering_students
def percentage_passed := (total_engineering_passed / total_engineering_students) * 100

theorem percentage_of_engineering_students_passed :
  percentage_passed = 22 :=
by
  sorry

end percentage_of_engineering_students_passed_l578_578206


namespace banana_arrangements_l578_578632

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578632


namespace nineteen_power_six_l578_578291

theorem nineteen_power_six :
    19^11 / 19^5 = 47045881 := by
  sorry

end nineteen_power_six_l578_578291


namespace BANANA_arrangements_l578_578996

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l578_578996


namespace total_interest_approx_l578_578245

-- Definitions based on conditions in a)
def principal1 : ℝ := 800
def rate1 : ℝ := 0.03
def principal2 : ℝ := 1400
def rate2 : ℝ := 0.05
def years : ℝ := 3.723404255319149

-- Defining the expected total interest
def expected_total_interest : ℝ := 350

-- Lean statement to prove the problem
theorem total_interest_approx :
  (principal1 * rate1 * years) + (principal2 * rate2 * years) ≈ expected_total_interest := by
  sorry

end total_interest_approx_l578_578245


namespace number_of_unique_permutations_BANANA_l578_578346

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578346


namespace distinct_permutations_BANANA_l578_578651

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578651


namespace determine_a_if_derivative_is_even_l578_578147

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem determine_a_if_derivative_is_even (a : ℝ) :
  (∀ x : ℝ, f' x a = f' (-x) a) → a = 0 :=
by
  intros h
  sorry

end determine_a_if_derivative_is_even_l578_578147


namespace part_a_prob_part_b_expected_time_l578_578215

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l578_578215


namespace selection_methods_count_l578_578048

theorem selection_methods_count (T M F : ℕ) (hT : T = 3) (hM : M = 8) (hF : F = 5) :
  T * (M + F) = 39 :=
by
  rw [hT, hM, hF]
  norm_num

end selection_methods_count_l578_578048


namespace arrangements_of_BANANA_l578_578424

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578424


namespace BANANA_arrangement_l578_578484

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578484


namespace sum_of_two_greatest_values_of_b_sum_of_two_greatest_values_l578_578204

theorem sum_of_two_greatest_values_of_b (b : Real) 
  (h : 4 * b ^ 4 - 41 * b ^ 2 + 100 = 0) :
  b = 2.5 ∨ b = 2 ∨ b = -2.5 ∨ b = -2 :=
sorry

theorem sum_of_two_greatest_values (b1 b2 : Real)
  (hb1 : 4 * b1 ^ 4 - 41 * b1 ^ 2 + 100 = 0)
  (hb2 : 4 * b2 ^ 4 - 41 * b2 ^ 2 + 100 = 0) :
  b1 = 2.5 → b2 = 2 → b1 + b2 = 4.5 :=
sorry

end sum_of_two_greatest_values_of_b_sum_of_two_greatest_values_l578_578204


namespace fraction_zero_power_equals_one_l578_578186

noncomputable def numerator := -574839201
noncomputable def denominator := 1357924680
def fraction_nonzero : Prop := numerator ≠ 0 ∧ denominator ≠ 0
def given_fraction : ℚ := numerator / denominator

theorem fraction_zero_power_equals_one (h : fraction_nonzero) : (given_fraction)^0 = 1 := by
  sorry

end fraction_zero_power_equals_one_l578_578186


namespace BANANA_arrangement_l578_578464

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578464


namespace banana_unique_permutations_l578_578859

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578859


namespace number_of_arrangements_l578_578503

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578503


namespace banana_unique_permutations_l578_578871

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578871


namespace circle_equation_l578_578153

theorem circle_equation (x y : ℝ) : (3 * x - 4 * y + 12 = 0) → (x^2 + 4 * x + y^2 - 3 * y = 0) :=
sorry

end circle_equation_l578_578153


namespace fifth_term_power_of_five_sequence_l578_578325

theorem fifth_term_power_of_five_sequence : 5^0 + 5^1 + 5^2 + 5^3 + 5^4 = 781 := 
by
sorry

end fifth_term_power_of_five_sequence_l578_578325


namespace banana_arrangements_l578_578981

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578981


namespace numberOfWaysToArrangeBANANA_l578_578748

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578748


namespace distinct_permutations_BANANA_l578_578657

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578657


namespace number_of_ways_to_arrange_BANANA_l578_578821

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578821


namespace banana_arrangements_l578_578963

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578963


namespace max_volume_cube_max_volume_parallelepiped_l578_578084

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l578_578084


namespace banana_arrangements_l578_578901

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578901


namespace arrange_banana_l578_578593

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578593


namespace banana_arrangements_l578_578613

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578613


namespace banana_arrangements_l578_578890

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578890


namespace sum_of_g1_values_l578_578109

noncomputable def g : ℝ → ℝ := sorry

axiom g_property (x : ℝ) (hx : x ≠ 0) : 
  g(x - 1) + g(x) + g(x + 1) = (g(x))^2 / (2017 * x)

theorem sum_of_g1_values : g(1) = 6051 := by
  sorry

end sum_of_g1_values_l578_578109


namespace exists_polynomial_primes_l578_578127

noncomputable def find_polynomial (n : ℕ) : ℕ → ℤ :=
sorry

theorem exists_polynomial_primes (n : ℕ) (hn : 0 < n) :
  ∃ f : ℕ → ℤ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → Nat.Prime (f k)) ∧
               (∀ k₁ k₂ : ℕ, 1 ≤ k₁ ∧ k₁ ≤ n → 1 ≤ k₂ ∧ k₂ ≤ n → k₁ < k₂ → f k₁ < f k₂) :=
sorry

end exists_polynomial_primes_l578_578127


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l578_578070

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l578_578070


namespace number_of_arrangements_BANANA_l578_578790

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578790


namespace number_of_arrangements_BANANA_l578_578776

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578776


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l578_578072

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l578_578072


namespace categorize_numbers_l578_578199

-- Define the given numbers
def a : ℝ := -1 / 7
def b : ℝ := 2.5
def c : ℝ := (sqrt (4 / 5))^2
def d : ℝ := 0
def e : ℝ := -sqrt ((-3)^2)
def f : ℝ := real.cbrt 9
def g : ℝ := real.pi / 2
def h : ℝ := 0.5757757775 --representation for the non-repressenting decimal number

-- Statement to prove numbers are either rational or irrational
theorem categorize_numbers :
  a ∈ set_of is_rational ∧
  b ∈ set_of is_rational ∧
  c ∈ set_of is_rational ∧
  d ∈ set_of is_rational ∧
  e ∈ set_of is_rational ∧
  f ∈ set_of is_irrational ∧
  g ∈ set_of is_irrational ∧
  h ∈ set_of is_irrational := by sorry

end categorize_numbers_l578_578199


namespace banana_unique_permutations_l578_578850

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578850


namespace binom_60_3_eq_34220_l578_578312

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l578_578312


namespace numberOfWaysToArrangeBANANA_l578_578755

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578755


namespace binom_60_3_eq_34220_l578_578318

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l578_578318


namespace banana_arrangements_l578_578960

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578960


namespace banana_unique_permutations_l578_578866

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578866


namespace banana_arrangements_l578_578897

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578897


namespace arrange_banana_l578_578595

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578595


namespace banana_arrangements_l578_578990

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578990


namespace parabola_vertex_location_l578_578028

theorem parabola_vertex_location (a b c : ℝ) (h1 : ∀ x < 0, a * x^2 + b * x + c ≤ 0) (h2 : a < 0) : 
  -b / (2 * a) ≥ 0 :=
by
  sorry

end parabola_vertex_location_l578_578028


namespace distinct_permutations_BANANA_l578_578653

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578653


namespace numberOfWaysToArrangeBANANA_l578_578737

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578737


namespace binom_60_3_eq_34220_l578_578311

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l578_578311


namespace permutations_BANANA_l578_578944

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578944


namespace permutations_BANANA_l578_578532

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578532


namespace Jen_distance_from_start_l578_578093

-- Define the rate of Jen's walking (in miles per hour)
def walking_rate : ℝ := 4

-- Define the time Jen walks forward (in hours)
def forward_time : ℝ := 2

-- Define the time Jen walks back (in hours)
def back_time : ℝ := 1

-- Define the distance walked forward
def distance_forward : ℝ := walking_rate * forward_time

-- Define the distance walked back
def distance_back : ℝ := walking_rate * back_time

-- Define the net distance from the starting point
def net_distance : ℝ := distance_forward - distance_back

-- Theorem stating the net distance from the starting point is 4.0 miles
theorem Jen_distance_from_start : net_distance = 4.0 := by
  sorry

end Jen_distance_from_start_l578_578093


namespace jason_age_when_joined_military_l578_578092

-- Definitions based on conditions
def years_to_chief : ℕ := 8
def years_to_master_chief : ℕ := years_to_chief + (years_to_chief / 4)
def years_after_master_chief : ℕ := 10
def age_at_retirement : ℕ := 46
def total_years_in_military : ℕ := years_to_chief + years_to_master_chief + years_after_master_chief

-- Lean statement of the problem
theorem jason_age_when_joined_military : 
  age_at_retirement - total_years_in_military = 18 :=
by 
  -- Various conditions used as definitions in Lean 4
  have h1 : years_to_chief = 8, by rfl
  have h2 : years_to_master_chief = 10, by rfl
  have h3 : years_after_master_chief = 10, by rfl
  have h4 : age_at_retirement = 46, by rfl
  have h5 : total_years_in_military = 28, by 
  -- calculate the total years in military step by step
     calc
      years_to_chief + years_to_master_chief + years_after_master_chief 
      = 8 + 10 + 10 : by simp [h1, h2, h3]
  -- now calculate the final result
  show 46 - 28 = 18, by norm_num

end jason_age_when_joined_military_l578_578092


namespace permutations_of_BANANA_l578_578408

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578408


namespace workers_complete_time_l578_578002

theorem workers_complete_time
  (A : ℝ) -- Total work
  (x1 x2 x3 : ℝ) -- Productivities of the workers
  (h1 : x3 = (x1 + x2) / 2)
  (h2 : 10 * x1 = 15 * x2) :
  (A / x1 = 50) ∧ (A / x2 = 75) ∧ (A / x3 = 60) :=
by
  sorry  -- Proof not required

end workers_complete_time_l578_578002


namespace arrange_banana_l578_578600

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578600


namespace number_of_arrangements_l578_578509

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578509


namespace sequence_a10_is_1_over_19_l578_578060

noncomputable def sequence (n : Nat) : ℚ :=
  if n = 0 then 1 else 
    let rec aux : Nat → ℚ → ℚ
      | 0, a_n => a_n
      | n+1, a_n => aux n (a_n / (1 + 2 * a_n))
    aux n 1

theorem sequence_a10_is_1_over_19 : sequence 10 = 1 / 19 :=
by
  sorry

end sequence_a10_is_1_over_19_l578_578060


namespace min_value_one_over_x_plus_one_over_y_l578_578020

theorem min_value_one_over_x_plus_one_over_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = 4) : 
  (1 / x + 1 / y) ≥ 1 :=
by
  sorry -- Proof goes here

end min_value_one_over_x_plus_one_over_y_l578_578020


namespace banana_arrangements_l578_578635

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578635


namespace arrangements_of_BANANA_l578_578418

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578418


namespace shanghai_expo_problem_l578_578142

open Nat

theorem shanghai_expo_problem (year : ℕ) (may_1_weekday : ℕ → ℕ) :
  year = 2010 ∧ (may_1_weekday 2010 = 6) →
  let common_year := year % 4 ≠ 0 in
  common_year ∧
  let days_in_year := 365 in
  days_in_year = 365 ∧
  let may_31_weekday := (may_1_weekday 2010 + 30) % 7 in
  may_31_weekday = 1 ∧
  let total_days_from_may_1_to_oct_31 := 31 + 30 + 31 + 31 + 30  in
  total_days_from_may_1_to_oct_31 = 184 :=
by
  assume h,
  cases h with h_year h_may_1,
  have h_common_year : year % 4 ≠ 0 := by
    rw h_year
    norm_num,
  have h_days_in_year : 365 = 365 := rfl,
  have h_may_31_weekday : (may_1_weekday 2010 + 30) % 7 = 1 := by
    rw [h_may_1]
    norm_num,
  have h_total_days : 31 + 30 + 31 + 31 + 30 = 184 := by
    norm_num,
  exact ⟨h_common_year, ⟨h_days_in_year, ⟨h_may_31_weekday, h_total_days⟩⟩⟩

end shanghai_expo_problem_l578_578142


namespace arrangement_count_BANANA_l578_578714

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578714


namespace number_of_arrangements_BANANA_l578_578765

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578765


namespace BANANA_arrangement_l578_578456

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578456


namespace arrangement_count_BANANA_l578_578696

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578696


namespace find_p_plus_q_l578_578265

noncomputable def probability_only_one (factor : ℕ → Prop) : ℚ := 0.08 -- Condition 1
noncomputable def probability_exaclty_two (factor1 factor2 : ℕ → Prop) : ℚ := 0.12 -- Condition 2
noncomputable def probability_all_three_given_two (factor1 factor2 factor3 : ℕ → Prop) : ℚ := 1 / 4 -- Condition 3
def women_without_D_has_no_risk_factors (total_women women_with_D women_with_all_factors women_without_D_no_risk_factors : ℕ) : ℚ :=
  women_without_D_no_risk_factors / (total_women - women_with_D)

theorem find_p_plus_q : ∃ (p q : ℕ), (women_without_D_has_no_risk_factors 100 (8 + 2 * 12 + 4) 4 28 = p / q) ∧ (Nat.gcd p q = 1) ∧ p + q = 23 :=
by
  sorry

end find_p_plus_q_l578_578265


namespace arrangements_of_BANANA_l578_578442

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578442


namespace BANANA_arrangement_l578_578450

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578450


namespace find_circle_radius_l578_578150

-- Definitions of given distances and the parallel chord condition
def isChordParallelToDiameter (c d : ℝ × ℝ) (radius distance1 distance2 : ℝ) : Prop :=
  let p1 := distance1
  let p2 := distance2
  p1 = 5 ∧ p2 = 12 ∧ 
  -- Assuming distances from the end of the diameter to the ends of the chord
  true

-- The main theorem which states the radius of the circle given the conditions
theorem find_circle_radius
  (diameter chord : ℝ × ℝ)
  (R p1 p2 : ℝ)
  (h1 : isChordParallelToDiameter diameter chord R p1 p2) :
  R = 6.5 :=
  by
    sorry

end find_circle_radius_l578_578150


namespace banana_arrangements_l578_578970

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578970


namespace banana_unique_permutations_l578_578877

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578877


namespace arrangements_of_BANANA_l578_578427

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578427


namespace banana_arrangements_l578_578915

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578915


namespace arrange_banana_l578_578572

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578572


namespace remainder_when_divided_by_40_l578_578023

theorem remainder_when_divided_by_40 (b : ℤ) : let m := 40 * b - 1 in (m^2 + 3 * m + 5) % 40 = 3 := 
by
  let m := 40 * b - 1
  sorry

end remainder_when_divided_by_40_l578_578023


namespace problem_statement_l578_578110

variables {n : ℕ} (n_pos : 0 < n)
variables (f : Fin (2 * n + 1) → ℝ)

def x (i : ℕ) : ℝ := (2 * Real.pi * i) / (2 * n + 1)

def α (i : ℕ) : ℝ :=
  2 / (2 * n + 1) * (∑ j in Finset.range (2 * n + 1), f ⟨j, sorry⟩ * Real.cos (j * x i))

def β (i : ℕ) : ℝ :=
  2 / (2 * n + 1) * (∑ j in Finset.range (2 * n + 1), f ⟨j, sorry⟩ * Real.sin (j * x i))

theorem problem_statement :
  ∀ i, 0 ≤ i ∧ i ≤ 2 * n →
  f ⟨i, sorry⟩ = α n f 0 / 2 + ∑ j in Finset.range (n + 1), α n f j * Real.cos (j * x i) + β n f j * Real.sin (j * x i) :=
sorry

end problem_statement_l578_578110


namespace distinct_permutations_BANANA_l578_578676

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578676


namespace henry_gym_workout_limit_l578_578001

noncomputable def a_sequence (n : ℕ) : ℝ :=
  if n = 0 then 1.5
  else 3 - 0.5 * b_sequence (n - 1)

noncomputable def b_sequence (n : ℕ) : ℝ :=
  if n = 0 then 1.5
  else 0.5 * a_sequence (n - 1) + 1.5

theorem henry_gym_workout_limit :
  (∀ (n : ℕ), abs (a_sequence n - 2) = 0) ∧ (∀ (n : ℕ), abs (b_sequence n - 1) = 0) →
  abs (\lim (n → a_sequence n) - \lim (n → b_sequence n)) = 1 :=
by
  sorry

end henry_gym_workout_limit_l578_578001


namespace banana_arrangements_l578_578623

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578623


namespace banana_arrangements_l578_578959

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578959


namespace banana_arrangements_l578_578900

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578900


namespace BANANA_arrangement_l578_578451

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578451


namespace candles_used_l578_578259

theorem candles_used (starting_candles used_candles remaining_candles : ℕ) (h1 : starting_candles = 44) (h2 : remaining_candles = 12) : used_candles = 32 :=
by
  sorry

end candles_used_l578_578259


namespace smallest_six_digit_divisible_by_6_l578_578114

theorem smallest_six_digit_divisible_by_6 : ∃ n, 
  (∀ m, (List.permutations [1, 2, 3, 4, 5, 6]).any (λ l, Nat.ofDigits 10 l = m) → 
    (m % 6 = 0) → n ≤ m) ∧ 
  n = 123456 := 
sorry

end smallest_six_digit_divisible_by_6_l578_578114


namespace binom_60_3_eq_34220_l578_578322

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l578_578322


namespace number_of_ways_to_arrange_BANANA_l578_578802

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578802


namespace simplify_and_evaluate_expression_l578_578132

theorem simplify_and_evaluate_expression (m : ℝ) (h1 : m^2 - 4 = 0) (h2 : m ≠ 2) : 
  (m + 3) / m = -1 / 2 :=
by
  have h_m := eq_or_neg_eq_of_sq_eq_sq 2 m h1
  cases h_m with h_pos h_neg
  · contradiction
  · rw [h_neg]
    simp
    linarith

end simplify_and_evaluate_expression_l578_578132


namespace banana_arrangements_l578_578629

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578629


namespace arrange_banana_l578_578598

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578598


namespace arrange_banana_l578_578582

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578582


namespace blanch_lunch_slices_l578_578274

theorem blanch_lunch_slices (s b L sn d r : ℤ) (h1 : s = 15) (h2 : b = 4)
  (h3 : sn = 2) (h4 : d = 5) (h5 : r = 2) (h6 : s - b - L - sn - d = r) :
  L = 2 :=
by {
  -- Initial statements from conditions
  rw [h1, h2, h3, h4, h5] at h6,
  -- Simplification to isolate L
  linarith [h6],
  -- The goal follows immediately
  sorry,
}

end blanch_lunch_slices_l578_578274


namespace card_area_l578_578000

theorem card_area (original_length original_width : ℕ) (h₁ : original_length = 5) (h₂ : original_width = 7)
    (reduced_side_area : ℕ) (h₃ : reduced_side_area = 21) :
    let new_length := original_length - 2 in
    let new_width := original_width - 2 in
    original_length * new_width = 25 :=
by
  sorry

end card_area_l578_578000


namespace number_of_arrangements_l578_578505

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578505


namespace numberOfWaysToArrangeBANANA_l578_578732

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578732


namespace banana_arrangements_l578_578992

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578992


namespace number_of_unique_permutations_BANANA_l578_578340

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578340


namespace permutations_BANANA_l578_578543

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578543


namespace banana_arrangements_l578_578608

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578608


namespace number_of_ways_to_arrange_BANANA_l578_578818

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578818


namespace numberOfWaysToArrangeBANANA_l578_578756

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578756


namespace number_of_arrangements_l578_578513

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578513


namespace expected_value_Z_l578_578183

-- Assume the given conditions
variables (p : ℝ) (h : 0 < p ∧ p < 1)

-- Define Z as a random variable with the given probability function
def P_Z (k : ℕ) : ℝ :=
if k ≥ 2 then p * (1 - p)^(k - 1) + (1 - p) * p^(k - 1) else 0

-- Define the expected value of a geometric distribution E(Y) = 1/p
def E_Y : ℝ := 1 / p

-- Formalize the expected value of Z
def E_Z : ℝ := ∑' k, (k : ℝ) * P_Z p k

-- The proof statement
theorem expected_value_Z : E_Z p = 1 / (p * (1 - p)) - 1 := sorry

end expected_value_Z_l578_578183


namespace banana_arrangements_l578_578626

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578626


namespace simple_interest_sum_l578_578254

variable {P R : ℝ}

theorem simple_interest_sum :
  P * (R + 6) = P * R + 3000 → P = 500 :=
by
  intro h
  sorry

end simple_interest_sum_l578_578254


namespace solve_for_x_l578_578158

theorem solve_for_x (x y z : ℚ) (h1 : x * y = 2 * (x + y)) (h2 : y * z = 4 * (y + z)) (h3 : x * z = 8 * (x + z)) (hx0 : x ≠ 0) (hy0 : y ≠ 0) (hz0 : z ≠ 0) : x = 16 / 3 := 
sorry

end solve_for_x_l578_578158


namespace permutations_BANANA_l578_578954

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578954


namespace max_distinct_integers_l578_578037

theorem max_distinct_integers (M : ℕ → ℕ → ℕ) (h_row : ∀ i : ℕ, i < 16 → (finset.univ.image (M i)).card ≤ 4)
    (h_col : ∀ j : ℕ, j < 16 → (finset.univ.image (λ i, M i j)).card ≤ 4) : (finset.univ.bUnion (λ i, finset.univ.image (λ j, M i j))).card ≤ 49 :=
sorry

end max_distinct_integers_l578_578037


namespace permutations_BANANA_l578_578551

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578551


namespace probability_blue_or_green_l578_578182

def faces : Type := {faces : ℕ // faces = 6}
noncomputable def blue_faces : ℕ := 3
noncomputable def red_faces : ℕ := 2
noncomputable def green_faces : ℕ := 1

theorem probability_blue_or_green :
  (blue_faces + green_faces) / 6 = (2 / 3) := by
  sorry

end probability_blue_or_green_l578_578182


namespace consistent_system_l578_578242

variable (x y : ℕ)

def condition1 := x + y = 40
def condition2 := 2 * 15 * x = 20 * y

theorem consistent_system :
  condition1 x y ∧ condition2 x y ↔ 
  (x + y = 40 ∧ 2 * 15 * x = 20 * y) :=
by
  sorry

end consistent_system_l578_578242


namespace numberOfWaysToArrangeBANANA_l578_578741

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578741


namespace binom_60_3_eq_34220_l578_578304

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l578_578304


namespace Peter_can_guarantee_victory_l578_578126

structure Board :=
  (size : ℕ)
  (cells : Fin size × Fin size → Option Color)

inductive Player
  | Peter
  | Victor
deriving DecidableEq

inductive Color
  | Red
  | Green
  | White
deriving DecidableEq

structure Move :=
  (player : Player)
  (rectangle : Fin 2 × Fin 2)
  (position : Fin 7 × Fin 7)

def isValidMove (board : Board) (move : Move) : Prop := sorry

def applyMove (board : Board) (move : Move) : Board := sorry

def allCellsColored (board : Board) : Prop := sorry

theorem Peter_can_guarantee_victory :
  ∀ (initialBoard : Board),
    (∀ (move : Move), move.player = Player.Victor → isValidMove initialBoard move) →
    Player.Peter = Player.Peter →
    (∃ finalBoard : Board,
       allCellsColored finalBoard ∧ 
       ¬ (∃ (move : Move), move.player = Player.Victor ∧ isValidMove finalBoard move)) :=
sorry

end Peter_can_guarantee_victory_l578_578126


namespace number_of_arrangements_l578_578494

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578494


namespace fraction_zero_iff_numerator_zero_l578_578151

variable (x : ℝ)

def numerator (x : ℝ) : ℝ := x - 5
def denominator (x : ℝ) : ℝ := 6 * x + 12

theorem fraction_zero_iff_numerator_zero (h_denominator_nonzero : denominator 5 ≠ 0) : 
  numerator x / denominator x = 0 ↔ x = 5 :=
by sorry

end fraction_zero_iff_numerator_zero_l578_578151


namespace arrange_banana_l578_578567

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578567


namespace arrangement_count_BANANA_l578_578707

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578707


namespace banana_unique_permutations_l578_578865

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578865


namespace permutations_of_BANANA_l578_578377

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578377


namespace max_diagonal_of_rectangle_l578_578160

theorem max_diagonal_of_rectangle (l w : ℝ) (h : 2 * l + 2 * w = 40) : 
  let d := Real.sqrt (l^2 + w^2) in d ≤ Real.sqrt 200 :=
by
  sorry

end max_diagonal_of_rectangle_l578_578160


namespace two_functions_with_domain_and_range_R_l578_578331

noncomputable def f1 (x : ℝ) := 3 - x
noncomputable def f2 (x : {x // x > 0}) := 2^(x - 1)
noncomputable def f3 (x : ℝ) := x^2 + 2*x - 10
noncomputable def f4 (x : ℝ) := if x ≤ 0 then x else 1 / x

theorem two_functions_with_domain_and_range_R :
  (∀ f1, (∃ (df1 rf1 : set ℝ), df1 = rf1 ∧ df1 = set.univ) ∧
   ∀ f4, (∃ (df4 rf4 : set ℝ), df4 = rf4 ∧ df4 = set.univ)) :=
by {
  sorry,
}

end two_functions_with_domain_and_range_R_l578_578331


namespace permutations_BANANA_l578_578538

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578538


namespace arrangements_of_BANANA_l578_578440

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578440


namespace banana_arrangements_l578_578892

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578892


namespace find_percentage_difference_l578_578162

noncomputable def initial_price := 25.0
noncomputable def expected_price := 33.0625
noncomputable def annual_inflation_rate := 0.12 -- 12% as a decimal

def price_increase_rate (P_1994 P_1996 : Float) : Float :=
  ((P_1996 - P_1994) / P_1994) * 100

def compounded_inflation_rate (r_inf : Float) (n : Nat) : Float :=
  (1 + r_inf) ^ n - 1

theorem find_percentage_difference :
  let rate_increase_sugar := price_increase_rate initial_price expected_price
  let total_inflation := compounded_inflation_rate annual_inflation_rate 2
  rate_increase_sugar - (total_inflation * 100) = 6.81 := by
  sorry

end find_percentage_difference_l578_578162


namespace binom_60_3_eq_34220_l578_578320

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l578_578320


namespace numberOfWaysToArrangeBANANA_l578_578754

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578754


namespace max_volume_cube_max_volume_parallelepiped_l578_578086

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l578_578086


namespace number_of_ways_to_arrange_BANANA_l578_578822

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578822


namespace banana_arrangements_l578_578625

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578625


namespace total_chocolates_and_candies_l578_578135

-- Define initial parameters and conditions
variable (x y c c_c : ℕ)

-- Equations derived from the problem:
def equation1 : Prop := 3 * x + 5 * y + 25 = c
def equation2 : Prop := 7 * x + 5 * y = c_c
def equation3 : Prop := c - 3 * x - 5 * y = 25
def equation4 : Prop := c_c - 7 * x - 5 * y = 4
def equation5 : Prop := c - 3 * x - 5 * y - 4 = 1

-- Proof that total chocolates and candies equal 35
theorem total_chocolates_and_candies :
  equation1 →
  equation2 →
  equation3 →
  equation4 →
  equation5 →
  c + c_c = 35 :=
by
  intros
  sorry -- skipping the proof steps

end total_chocolates_and_candies_l578_578135


namespace cube_not_splittable_l578_578090

theorem cube_not_splittable (V : ℕ) (hV : V = 40 * 40 * 40) : ¬ ∃ (l w h : ℕ), 
    (l % 2 = 1) ∧ (w % 2 = 1) ∧ (h % 2 = 1) ∧
    (l + 2 = w ∨ l + 2 = h ∨ w + 2 = h ∨ w + 2 = l ∨ h + 2 = l ∨ h + 2 = w) ∧
    (l * w * h = V / N) where N is the number of parallelepipeds : sorry

end cube_not_splittable_l578_578090


namespace number_of_ways_to_arrange_BANANA_l578_578803

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578803


namespace number_of_ways_to_arrange_BANANA_l578_578815

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578815


namespace distinct_permutations_BANANA_l578_578683

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578683


namespace permutations_BANANA_l578_578931

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578931


namespace remainder_ab_ac_bc_l578_578104

theorem remainder_ab_ac_bc {n : ℕ} (a b : ℤ) (ha : IsUnit (a : ZMod n)) (hb : IsUnit (b : ZMod n))
    (h : (a : ZMod n) = (b : ZMod n)⁻¹) :
    ab + ac + bc ≡ 7 [MOD n] where
    ab := a * b
    ac := a * (2 * a + b)
    bc := b * (2 * a + b) :=
by
  sorry

end remainder_ab_ac_bc_l578_578104


namespace problem_inequality_l578_578327

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * log (1 + x)

theorem problem_inequality
  (m : ℝ) (x_1 x_2 : ℝ)
  (H_m_pos : 0 < m)
  (H_extremes : ∃ (x_1 x_2 : ℝ), x_1 < x_2 ∧ (derivative (λ x, f x m) x_1 = 0) ∧ (derivative (λ x, f x m) x_2 = 0)) :
  2 * f x_2 m > -x_1 + 2 * x_1 * log 2 := 
sorry

end problem_inequality_l578_578327


namespace banana_unique_permutations_l578_578848

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578848


namespace coefficients_sum_l578_578161

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

theorem coefficients_sum (a b c d: ℝ) (g: ℝ → ℝ):
  (g 3i = 0) ∧ (g (3 + I) = 0) → a + b + c + d = 49 :=
sorry

end coefficients_sum_l578_578161


namespace arrangement_count_BANANA_l578_578706

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578706


namespace value_of_m_l578_578019

theorem value_of_m :
  ∀ (m : ℤ), (∀ x : ℤ, (x - 4) * (x + 3) = x^2 + m * x - 12) → m = -1 :=
by 
  intros m h,
  sorry

end value_of_m_l578_578019


namespace number_of_unique_permutations_BANANA_l578_578349

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578349


namespace permutations_BANANA_l578_578547

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578547


namespace arrangements_of_BANANA_l578_578447

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578447


namespace permutations_BANANA_l578_578955

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578955


namespace train_length_l578_578255

noncomputable def L_train : ℝ :=
  let speed_kmph : ℝ := 60
  let speed_mps : ℝ := (speed_kmph * 1000 / 3600)
  let time : ℝ := 30
  let length_bridge : ℝ := 140
  let total_distance : ℝ := speed_mps * time
  total_distance - length_bridge

theorem train_length : L_train = 360.1 :=
by
  -- Sorry statement to skip the proof
  sorry

end train_length_l578_578255


namespace number_of_ways_to_arrange_BANANA_l578_578812

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578812


namespace number_of_arrangements_l578_578515

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578515


namespace number_of_arrangements_l578_578516

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578516


namespace number_of_unique_permutations_BANANA_l578_578356

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578356


namespace tan_of_angle_on_line_l578_578029

variable (α : ℝ)

def lies_on_line (α : ℝ) : Prop :=
  ∃ (x y : ℝ), x + y = 0 ∧ tan α = y / x

theorem tan_of_angle_on_line (h : lies_on_line α) : tan α = -1 := 
sorry

end tan_of_angle_on_line_l578_578029


namespace num_students_59_l578_578243

theorem num_students_59 (apples : ℕ) (taken_each : ℕ) (students : ℕ) 
  (h_apples : apples = 120) 
  (h_taken_each : taken_each = 2) 
  (h_students_divisors : ∀ d, d = 59 → d ∣ (apples / taken_each)) : students = 59 :=
sorry

end num_students_59_l578_578243


namespace digit_8_count_1_to_800_l578_578011

theorem digit_8_count_1_to_800 : (count_digit 8 (range 1 801)) = 160 :=
sorry

-- Assuming helper functions are already defined:
-- count_digit d lst: counts how many times the digit d appears in the list of integers lst.
-- range a b: generates a list of integers from a to b-1.

end digit_8_count_1_to_800_l578_578011


namespace integral_comparison_l578_578100

noncomputable def a : ℝ := ∫ (x : ℝ) in 1..Real.exp 1, 1 / x

noncomputable def b : ℝ := ∫ (x : ℝ) in 0..1, Real.cos x

theorem integral_comparison : a > b :=
by
  sorry

end integral_comparison_l578_578100


namespace like_terms_sum_three_l578_578017

theorem like_terms_sum_three (m n : ℤ) (h1 : 2 * m = 4 - n) (h2 : m = n - 1) : m + n = 3 :=
sorry

end like_terms_sum_three_l578_578017


namespace permutations_BANANA_l578_578545

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578545


namespace BANANA_arrangement_l578_578457

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578457


namespace permutations_of_BANANA_l578_578391

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578391


namespace integral_abs_exp_eq_l578_578170

theorem integral_abs_exp_eq : 
  ∫ x in -4..2, exp (-|x|) = 2 - exp (-2) - exp (-4) := 
by
  sorry

end integral_abs_exp_eq_l578_578170


namespace number_of_arrangements_BANANA_l578_578788

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578788


namespace numberOfWaysToArrangeBANANA_l578_578760

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578760


namespace jacket_purchase_price_l578_578244

theorem jacket_purchase_price (S D P : ℝ) 
  (h1 : S = P + 0.30 * S)
  (h2 : D = 0.80 * S)
  (h3 : 6.000000000000007 = D - P) :
  P = 42 :=
by
  sorry

end jacket_purchase_price_l578_578244


namespace banana_arrangements_l578_578903

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578903


namespace distinct_permutations_BANANA_l578_578647

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578647


namespace ages_correct_l578_578042

def ages : List ℕ := [5, 8, 13, 15]
def Tanya : ℕ := 13
def Yura : ℕ := 8
def Sveta : ℕ := 5
def Lena : ℕ := 15

theorem ages_correct (h1 : Tanya ∈ ages) 
                     (h2: Yura ∈ ages)
                     (h3: Sveta ∈ ages)
                     (h4: Lena ∈ ages)
                     (h5: Tanya ≠ Yura)
                     (h6: Tanya ≠ Sveta)
                     (h7: Tanya ≠ Lena)
                     (h8: Yura ≠ Sveta)
                     (h9: Yura ≠ Lena)
                     (h10: Sveta ≠ Lena)
                     (h11: Sveta = 5)
                     (h12: Tanya > Yura)
                     (h13: (Tanya + Sveta) % 3 = 0) :
                     Tanya = 13 ∧ Yura = 8 ∧ Sveta = 5 ∧ Lena = 15 := by
  sorry

end ages_correct_l578_578042


namespace part_a_prob_part_b_expected_time_l578_578213

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l578_578213


namespace binom_60_3_eq_34220_l578_578317

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l578_578317


namespace interior_diagonal_length_l578_578164

-- Definitions from Conditions
def surface_area (a b c : ℝ) : ℝ := 2 * (a * b + a * c + b * c)
def edge_length (a b c : ℝ) : ℝ := 4 * (a + b + c)
def diagonal_length (a b c : ℝ) : ℝ := Real.sqrt (a^2 + b^2 + c^2)

theorem interior_diagonal_length
  (a b c : ℝ)
  (h1 : surface_area a b c = 54)
  (h2 : edge_length a b c = 40) :
  diagonal_length a b c = Real.sqrt 46 :=
by
  sorry

end interior_diagonal_length_l578_578164


namespace count_8_up_to_800_l578_578010

def count_digit_occurrences (digit : Nat) (n : Nat) : Nat :=
  let rec countDigit := (digit n : Nat) -> Nat
    | 0 => 0
    | k + 1 => (if k + 1 / 10 = digit then 1 else 0) + countDigit (k / 10)
  countDigit n

theorem count_8_up_to_800 :
  count_digit_occurrences 8 800 = 161 :=
by
  sorry

end count_8_up_to_800_l578_578010


namespace max_cube_side_length_max_parallelepiped_dimensions_l578_578064

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l578_578064


namespace arrangements_of_BANANA_l578_578419

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578419


namespace binom_60_3_eq_34220_l578_578306

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l578_578306


namespace permutations_BANANA_l578_578565

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578565


namespace minimal_side_length_of_room_l578_578326

theorem minimal_side_length_of_room (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ S : ℕ, S = 10 :=
by {
  sorry
}

end minimal_side_length_of_room_l578_578326


namespace permutations_BANANA_l578_578555

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578555


namespace equal_segments_on_equilateral_triangle_l578_578124

open EuclideanGeometry

-- Definitions based on given conditions
def triangles_congruent (ABC A1B1C1 : Triangle) : Prop :=
  congruent ABC A1B1C1

def points_divide_segment_eq (B C B1 C1 M M1 : Point) (ratio : ℝ) : Prop :=
  (∃ r : ℝ, r > 0 ∧ r = (BM / MC) ∧ r = (B1M1 / M1C1))

/- The main proof statement in Lean 4 -/
theorem equal_segments_on_equilateral_triangle
  (ABC A1B1C1 : Triangle) (B C B1 C1 M M1 : Point)
  (h1 : triangles_congruent ABC A1B1C1)
  (h2 : points_divide_segment_eq B C B1 C1 M M1 (BM / MC)) :
  distance (A, M) = distance (A1, M1) := 
sorry

end equal_segments_on_equilateral_triangle_l578_578124


namespace number_of_arrangements_l578_578526

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578526


namespace permutations_BANANA_l578_578923

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578923


namespace arrangement_count_BANANA_l578_578718

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578718


namespace arrangement_count_BANANA_l578_578686

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578686


namespace gcd_m_n_l578_578103

def m : ℕ := 333333
def n : ℕ := 7777777

theorem gcd_m_n : Nat.gcd m n = 1 :=
by
  -- Mathematical steps have been omitted as they are not needed
  sorry

end gcd_m_n_l578_578103


namespace number_of_arrangements_l578_578502

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578502


namespace number_of_unique_permutations_BANANA_l578_578348

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578348


namespace number_of_ways_to_arrange_BANANA_l578_578831

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578831


namespace number_of_unique_permutations_BANANA_l578_578365

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578365


namespace binom_60_3_eq_34220_l578_578314

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l578_578314


namespace BANANA_arrangement_l578_578455

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578455


namespace simplify_fraction_l578_578280

variable (x y : ℝ)

theorem simplify_fraction (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x^2 / y) * (y^2 / (2 * x)) = 3 * x * y / 2 :=
by sorry

end simplify_fraction_l578_578280


namespace problem_statement_l578_578197

theorem problem_statement 
  (H1 : ∀ x, 2^x = y ↔ x = log 2 y) -- symmetry in the graph for inverse functions
  (H2 : ∀ a, a > 0 ∧ a ≠ 1 → (a^0 + 1 = 2)) -- condition for function passing through (0, 2)
  (H3 : ∀ x, (1 / 2)^|x| ≤ 1) -- condition for maximum value of function
  (H4 : ¬∀ x, 3^x > 2^x) -- negating the fourth condition
  
  : (H1 ∧ H2 ∧ H3 ∧ H4) = true := 
by 
  sorry

end problem_statement_l578_578197


namespace banana_arrangements_l578_578911

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578911


namespace BANANA_arrangement_l578_578463

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578463


namespace distinct_permutations_BANANA_l578_578672

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578672


namespace largest_cube_side_length_largest_rect_parallelepiped_dimensions_l578_578078

variables {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (a): Side length of the largest cube
theorem largest_cube_side_length :
  let a₀ := a * b * c / (a * b + b * c + a * c) in
  ∃ a₀, a₀ = a * b * c / (a * b + b * c + a * c) :=
begin
  sorry
end

-- Part (b): Dimensions of the largest rectangular parallelepiped
theorem largest_rect_parallelepiped_dimensions :
  let x := a / 3, y := b / 3, z := c / 3 in
  ∃ x y z, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
begin
  sorry
end

end largest_cube_side_length_largest_rect_parallelepiped_dimensions_l578_578078


namespace permutations_BANANA_l578_578940

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578940


namespace number_of_arrangements_BANANA_l578_578784

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578784


namespace number_of_unique_permutations_BANANA_l578_578367

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578367


namespace number_of_arrangements_l578_578493

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578493


namespace parallelogram_half_quadrilateral_area_l578_578157

variables {A B C D E F G H : Type} [add_comm_group A]

/-- Define the function for midpoint -/
def midpoint (x y : A) : A := (x + y) / 2

/-- Define the quadrilateral with midpoints -/
structure Quadrilateral (A B C D E F G H : A) : Prop where
  midpoint_AB : E = midpoint A B
  midpoint_BC : F = midpoint B C
  midpoint_CD : G = midpoint C D
  midpoint_DA : H = midpoint D A

/-- Define the area function for a parallelogram and quadrilateral -/
noncomputable def Area (x y z w : A) : ℝ := sorry
noncomputable def ParallelogramArea (E F G H : A) : ℝ := sorry

/-- The theorem statement that needs to be proven -/
theorem parallelogram_half_quadrilateral_area 
  (A B C D E F G H : A) 
  (h : Quadrilateral A B C D E F G H) : 
  ParallelogramArea E F G H = Area A B C D / 2 := sorry

end parallelogram_half_quadrilateral_area_l578_578157


namespace distinct_permutations_BANANA_l578_578654

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578654


namespace flour_more_than_salt_l578_578117

open Function

-- Definitions based on conditions
def flour_needed : ℕ := 12
def flour_added : ℕ := 2
def salt_needed : ℕ := 7
def salt_added : ℕ := 0

-- Given that these definitions hold, prove the following theorem
theorem flour_more_than_salt : (flour_needed - flour_added) - (salt_needed - salt_added) = 3 :=
by
  -- Here you would include the proof, but as instructed, we skip it with "sorry".
  sorry

end flour_more_than_salt_l578_578117


namespace banana_unique_permutations_l578_578858

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578858


namespace number_of_arrangements_BANANA_l578_578783

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578783


namespace distinct_permutations_BANANA_l578_578662

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578662


namespace permutations_BANANA_l578_578548

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578548


namespace cos_diff_simplification_l578_578131

theorem cos_diff_simplification :
  (cos (30 * Real.pi / 180) - cos (60 * Real.pi / 180)) = (Real.sqrt 3 - 1) / 2 :=
  by
    sorry

end cos_diff_simplification_l578_578131


namespace compare_fractions_neg_l578_578287

theorem compare_fractions_neg : (- (2 / 3) > - (3 / 4)) :=
  sorry

end compare_fractions_neg_l578_578287


namespace number_of_arrangements_l578_578524

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578524


namespace banana_unique_permutations_l578_578862

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578862


namespace ratio_of_surface_areas_l578_578041

theorem ratio_of_surface_areas (s : ℝ) :
  let cube_surface_area := 6 * s^2
  let tetrahedron_edge := s * Real.sqrt 2
  let tetrahedron_face_area := (Real.sqrt 3 / 4) * (tetrahedron_edge)^2
  let tetrahedron_surface_area := 4 * tetrahedron_face_area
  (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3 :=
by
  let cube_surface_area := 6 * s^2
  let tetrahedron_edge := s * Real.sqrt 2
  let tetrahedron_face_area := (Real.sqrt 3 / 4) * (tetrahedron_edge)^2
  let tetrahedron_surface_area := 4 * tetrahedron_face_area
  show (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3
  sorry

end ratio_of_surface_areas_l578_578041


namespace permutations_BANANA_l578_578948

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578948


namespace banana_arrangements_l578_578984

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578984


namespace permutations_of_BANANA_l578_578387

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578387


namespace distance_between_circumcenters_l578_578106

variable (A B C O : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace O]

-- Define the lengths of sides AB, BC, and CA
variable (AB BC CA : ℝ)
variable h_AB : AB = 13
variable h_BC : BC = 14
variable h_CA : CA = 15

-- Define the circumcenter of triangle ABC
variable h_O : O -- O is the circumcenter of triangle ABC

-- Define the circumcenters of triangles AOB and AOC
variable O1 O2 : Type* -- circumcenters of triangles AOB and AOC

-- The problem statement to prove
theorem distance_between_circumcenters :
  dist O1 O2 = 91 / 6 :=
sorry

end distance_between_circumcenters_l578_578106


namespace permutations_BANANA_l578_578953

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578953


namespace banana_arrangements_l578_578619

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578619


namespace BANANA_arrangement_l578_578481

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578481


namespace number_of_mappings_l578_578111

noncomputable def A : set ℕ := {0, 1, 2}
noncomputable def B : set ℤ := {-1, 0, 1}
noncomputable def f (g : ℕ → ℤ) : Prop :=
  g 0 ∈ B ∧ g 1 ∈ B ∧ g 2 ∈ B ∧ (g 0 - g 1 = g 2)

theorem number_of_mappings : ∃ (count : ℕ), count = 7 ∧
  ∃ (fns : set (ℕ → ℤ)),
    (∀ g ∈ fns, f g) ∧ 
    count = set.card fns :=
begin
  have hA: finite A := by simp [A],
  have hB: finite B := by simp [B],
  sorry
end

end number_of_mappings_l578_578111


namespace permutations_of_BANANA_l578_578401

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578401


namespace distinct_permutations_BANANA_l578_578656

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578656


namespace number_of_arrangements_BANANA_l578_578794

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578794


namespace number_of_arrangements_BANANA_l578_578780

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578780


namespace numberOfWaysToArrangeBANANA_l578_578742

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578742


namespace combined_machines_complete_job_in_2_hours_l578_578234

-- Definitions based on conditions
def work_rate (hours: ℕ) : ℝ := 1 / hours

def type_R_work_rate : ℝ := work_rate 36
def type_S_work_rate : ℝ := work_rate 36

def machines (n: ℕ) (rate: ℝ) : ℝ := n * rate

-- Stating the problem
theorem combined_machines_complete_job_in_2_hours (n: ℕ) :
  n = 9 →
  (machines n type_R_work_rate + machines n type_S_work_rate) = 1 / 2 →
  1 / (machines n type_R_work_rate + machines n type_S_work_rate) = 2 :=
begin
  intros h1 h2,
  sorry
end

end combined_machines_complete_job_in_2_hours_l578_578234


namespace number_of_arrangements_BANANA_l578_578763

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578763


namespace geometric_sequence_a9_l578_578055

theorem geometric_sequence_a9 (a : ℕ → ℝ) (h1 : a 5 = 4) (h2 : a 7 = 6) : a 9 = 9 :=
by
  have prop : (a 7) ^ 2 = (a 5) * (a 9), from sorry
  rw [h1, h2] at prop
  calc
    a 9 = 6 ^ 2 / 4 : by sorry
        ... = 9 : by sorry

end geometric_sequence_a9_l578_578055


namespace BANANA_arrangement_l578_578473

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578473


namespace sum_is_constant_l578_578099

variable (a b c d : ℚ) -- declare variables states as rational numbers

theorem sum_is_constant :
  (a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) →
  a + b + c + d = -(14 / 3) :=
by
  intros h
  sorry

end sum_is_constant_l578_578099


namespace find_starting_number_l578_578025

theorem find_starting_number (a b c : ℕ) 
  (h_digit_a : 1 ≤ a ∧ a ≤ 9)
  (h_digit_b : 0 ≤ b ∧ b ≤ 9)
  (h_digit_c : 0 ≤ c ∧ c ≤ 9)
  (h_orig : 100 * a + 10 * b + c)
  (h_swapped : 100 * a + 10 * c + b)
  (h_sum : 1730 ≤ 200 * a + 11 * (b + c) ∧ 200 * a + 11 * (b + c) ≤ 1739) :
  100 * a + 10 * b + c = 1732 := 
sorry

end find_starting_number_l578_578025


namespace numberOfWaysToArrangeBANANA_l578_578750

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578750


namespace arrange_banana_l578_578573

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578573


namespace assemble_puzzle_three_pieces_l578_578125

theorem assemble_puzzle_three_pieces :
  ∀ (initial_pieces : ℕ), 
  (gluing_time : ℕ) 
  (pieces_reduction_per_operation : ℕ), 
  initial_pieces = 121 → 
  gluing_time = 120 → 
  pieces_reduction_per_operation = 2 → 
  (gluing_time / pieces_reduction_per_operation = 1 hour) := 
by 
  intros initial_pieces gluing_time pieces_reduction_per_operation hp hc hr 
  sorry

end assemble_puzzle_three_pieces_l578_578125


namespace problem1_solution_l578_578210

noncomputable def beta (α β : ℝ) : ℝ := 60

theorem problem1_solution (α β : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2: 0 < β ∧ β < π / 2)
  (h_cos_α : cos α = 1 / 7)
  (h_cos_alpha_beta : cos (α + β) = -11 / 14) : β = 60 :=
sorry

end problem1_solution_l578_578210


namespace number_of_arrangements_l578_578510

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578510


namespace arrangement_count_BANANA_l578_578711

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578711


namespace arrangements_of_BANANA_l578_578434

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578434


namespace number_of_arrangements_BANANA_l578_578772

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578772


namespace BANANA_arrangements_l578_578997

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l578_578997


namespace distinct_permutations_BANANA_l578_578645

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578645


namespace banana_unique_permutations_l578_578878

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578878


namespace arrangement_count_BANANA_l578_578717

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578717


namespace max_volume_cube_max_volume_parallelepiped_l578_578087

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l578_578087


namespace BANANA_arrangement_l578_578470

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578470


namespace permutations_BANANA_l578_578531

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578531


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l578_578224

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l578_578224


namespace number_of_arrangements_BANANA_l578_578769

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578769


namespace permutations_BANANA_l578_578949

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578949


namespace total_marbles_l578_578040

variable (b : ℝ)
variable (r : ℝ) (g : ℝ)
variable (h₁ : r = 1.3 * b)
variable (h₂ : g = 1.5 * b)

theorem total_marbles (b : ℝ) (r : ℝ) (g : ℝ) (h₁ : r = 1.3 * b) (h₂ : g = 1.5 * b) : r + b + g = 3.8 * b :=
by
  sorry

end total_marbles_l578_578040


namespace arrangement_count_BANANA_l578_578721

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578721


namespace max_number_of_extensions_l578_578205

theorem max_number_of_extensions : 
  ∀ (digits : Finset ℕ), digits = {1, 2, 3, 8} → 
  (∀ (x ∈ digits), x mod 2 = 0 → 
   ∀ (x₁ x₂ x₃ x₄ : ℕ), {x₁, x₂, x₃, x₄} = digits ∧ 
   (x₁ * 1000 + x₂ * 100 + x₃ * 10 + x₄) mod 2 = 0 → 
   ∃ n, n = 12) :=
by
  sorry

end max_number_of_extensions_l578_578205


namespace banana_unique_permutations_l578_578868

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578868


namespace arrange_banana_l578_578603

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578603


namespace arrangements_of_BANANA_l578_578439

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578439


namespace distinct_permutations_BANANA_l578_578678

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578678


namespace number_of_arrangements_l578_578527

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578527


namespace permutations_BANANA_l578_578926

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578926


namespace tangent_product_l578_578016

theorem tangent_product (P : ℕ) (tan : ℝ → ℝ) (H : ∀ x : ℝ, tan x = Real.tan x) :
  (∏ (k : ℕ) in finset.range (45) | k + 1, (1 + tan (k + 1) * π / 180)) = 2 ^ 23 :=
by
  let t : ℝ → ℝ := λ x, Real.tan (x * π / 180)
  have Ht : ∀ x : ℝ, t x = Real.tan (x * π / 180) := λ x, rfl

  dsimp at *
  rename_i k hk
  rw Ht
  dsimp
  sorry

end tangent_product_l578_578016


namespace permutations_of_BANANA_l578_578406

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578406


namespace distinct_permutations_BANANA_l578_578661

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578661


namespace permutations_of_BANANA_l578_578384

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578384


namespace permutations_BANANA_l578_578542

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578542


namespace compare_neg_fractions_l578_578282

theorem compare_neg_fractions : (- (2/3 : ℚ)) > - (3/4 : ℚ) := by
  sorry

end compare_neg_fractions_l578_578282


namespace max_distinct_integers_16x16_table_l578_578036

noncomputable def max_distinct_integers (T : Matrix (Fin 16) (Fin 16) ℕ) : ℕ :=
  let row_distinct := ∀ i : Fin 16, (Finset.univ.map (Function.Embedding.subtype $ T i)).card ≤ 4
  let col_distinct := ∀ j : Fin 16, (Finset.univ.map (Function.Embedding.subtype $ T <$> j)).card ≤ 4
  if row_distinct ∧ col_distinct then (Finset.univ.bUnion (λ i => Finset.univ.image (λ j => T i j))).card else 0

theorem max_distinct_integers_16x16_table :
  ∀ T : Matrix (Fin 16) (Fin 16) ℕ, 
  (∀ i, (Finset.univ.image (λ j => T i j)).card ≤ 4) →
  (∀ j, (Finset.univ.image (λ i => T i j)).card ≤ 4) →
  max_distinct_integers T ≤ 49 :=
by sorry

end max_distinct_integers_16x16_table_l578_578036


namespace banana_arrangements_l578_578621

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578621


namespace E_mirrors_C_about_O_l578_578209

-- Define the problem conditions
variables {α : Type*} [field α] [euclidean_space α]
variables (A B O C E : α) (ellipse : set α)

noncomputable def is_center (O : α) (ellipse : set α) : Prop :=
-- Define O as the center of the ellipse
sorry

noncomputable def is_major_axis (A B : α) (ellipse : set α) : Prop :=
-- Define AB as the major axis of the ellipse
sorry

noncomputable def is_on_ellipse (C : α) (ellipse : set α) : Prop :=
-- Define C as any point on the ellipse
sorry

noncomputable def draw_chord (C D : α) (ellipse : set α) : Prop :=
-- Define drawing chord CD from C such that it intersects the major axis at a variable angle
sorry

noncomputable def line_through_O_C_intersects_at_E (O C E : α) (ellipse : set α) : Prop :=
-- Define the line through O and C that intersects the ellipse at another point E
sorry

-- Define the theorem to prove: E always mirrors C about O
theorem E_mirrors_C_about_O
  (h_center : is_center O ellipse)
  (h_major_axis : is_major_axis A B ellipse)
  (h_on_ellipse : is_on_ellipse C ellipse)
  (h_chord : ∃ D, draw_chord C D ellipse)
  (h_line : line_through_O_C_intersects_at_E O C E ellipse) : 
  reflects_about O C E :=
sorry

end E_mirrors_C_about_O_l578_578209


namespace number_of_arrangements_l578_578495

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578495


namespace arrangements_of_BANANA_l578_578449

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578449


namespace banana_arrangements_l578_578641

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578641


namespace distinct_permutations_BANANA_l578_578665

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578665


namespace number_of_unique_permutations_BANANA_l578_578341

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578341


namespace arrange_banana_l578_578576

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578576


namespace numberOfWaysToArrangeBANANA_l578_578728

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578728


namespace travel_time_with_heavy_traffic_l578_578094

-- Define the conditions as constants or variables
constant t : ℝ := 4 -- hours with no traffic
constant d : ℝ := 200 -- distance in miles
constant traffic_speed_diff : ℝ := 10 -- mph difference

-- Define the speeds
def v : ℝ := d / t -- speed with no traffic
def v_traffic : ℝ := v - traffic_speed_diff -- speed with heavy traffic

-- Define the time with heavy traffic
def t_traffic : ℝ := d / v_traffic

-- Statement of the theorem
theorem travel_time_with_heavy_traffic : t_traffic = 5 := by
  -- Insert the proof later
  sorry

end travel_time_with_heavy_traffic_l578_578094


namespace simplify_expression_l578_578133

theorem simplify_expression (y : ℝ) :
  4 * y - 8 * y^2 + 6 - (3 - 6 * y - 9 * y^2 + 2 * y^3) = -2 * y^3 + y^2 + 10 * y + 3 := 
by
  -- Proof goes here, but we just state sorry for now
  sorry

end simplify_expression_l578_578133


namespace contractor_days_l578_578235

def days_engaged (days_worked days_absent : ℕ) (earnings_per_day : ℝ) (fine_per_absent_day : ℝ) : ℝ :=
  earnings_per_day * days_worked - fine_per_absent_day * days_absent

theorem contractor_days
  (days_absent : ℕ)
  (earnings_per_day : ℝ)
  (fine_per_absent_day : ℝ)
  (total_amount : ℝ)
  (days_worked : ℕ)
  (h1 : days_absent = 12)
  (h2 : earnings_per_day = 25)
  (h3 : fine_per_absent_day = 7.50)
  (h4 : total_amount = 360)
  (h5 : days_engaged days_worked days_absent earnings_per_day fine_per_absent_day = total_amount) :
  days_worked = 18 :=
by sorry

end contractor_days_l578_578235


namespace binom_60_3_eq_34220_l578_578321

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l578_578321


namespace distinct_permutations_BANANA_l578_578659

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578659


namespace number_of_arrangements_l578_578508

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578508


namespace arrangement_count_BANANA_l578_578702

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578702


namespace arrange_banana_l578_578583

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578583


namespace aluminum_cans_goal_l578_578115

variable (cans_martha : ℕ) (cans_diego : ℕ) (cans_leah : ℕ)
variable (aluminum_martha : ℕ) (aluminum_diego : ℕ) (aluminum_leah : ℕ)
variable (total_needed : ℕ)

def more_aluminum_cans_needed (cans_martha cans_diego cans_leah : ℕ) (aluminum_martha aluminum_diego aluminum_leah total_needed : ℕ) : ℕ :=
  let total_collected := aluminum_martha + aluminum_diego + aluminum_leah in
  total_needed - total_collected

theorem aluminum_cans_goal :
  ∀ (cans_martha cans_diego cans_leah aluminum_martha aluminum_diego aluminum_leah total_needed : ℕ),
  cans_martha = 90 →
  cans_diego = 55 →
  cans_leah = 25 →
  aluminum_martha = 63 →
  aluminum_diego = 27 →
  aluminum_leah = 20 →
  total_needed = 200 →
  more_aluminum_cans_needed cans_martha cans_diego cans_leah aluminum_martha aluminum_diego aluminum_leah total_needed = 90 :=
by
  intros
  sorry

end aluminum_cans_goal_l578_578115


namespace banana_arrangements_l578_578879

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578879


namespace BANANA_arrangement_l578_578453

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578453


namespace jill_peaches_l578_578091

variable (S J : ℕ)

theorem jill_peaches (h1 : S = 19) (h2 : S = J + 13) : J = 6 :=
by
  sorry

end jill_peaches_l578_578091


namespace number_approx_l578_578044

noncomputable def findNumber (N : ℝ) : ℝ := N

theorem number_approx (N : ℝ) (h1 : (7 / 13) * N = (5 / 16) * N + 500) : N ≈ 2213 := sorry

end number_approx_l578_578044


namespace number_of_ways_to_arrange_BANANA_l578_578801

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578801


namespace arrangements_of_BANANA_l578_578433

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578433


namespace number_of_arrangements_l578_578498

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578498


namespace number_of_ways_to_arrange_BANANA_l578_578828

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578828


namespace numberOfWaysToArrangeBANANA_l578_578745

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578745


namespace numberOfWaysToArrangeBANANA_l578_578723

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578723


namespace hyenas_difference_l578_578095

theorem hyenas_difference :
  let antelopes := 80
  let rabbits := antelopes + 34
  let leopards := rabbits / 2
  let wild_dogs := hyenas + 50
  let total_animals := antelopes + rabbits + hyenas + wild_dogs + leopards
  total_animals = 605 ->
  (antelopes + rabbits) - hyenas = 42 :=
by
  intro antelopes rabbits leopards wild_dogs total_animals h_total
  have h_rabbits : rabbits = antelopes + 34 := rfl
  have h_leopards : leopards = rabbits / 2 := rfl
  have h_wild_dogs : wild_dogs = hyenas + 50 := rfl

  let total_calculated := by
    simp only [h_rabbits, h_leopards, h_wild_dogs]

  have h_animals : total_animals = antelopes + rabbits + hyenas + wild_dogs + leopards := rfl

  sorry

end hyenas_difference_l578_578095


namespace count_eight_in_1_to_800_l578_578007

theorem count_eight_in_1_to_800 : 
  let count_digit (d n : Nat) : Nat := 
    (List.range (n + 1)).filter (λ x => List.contains (Nat.digits 10 x) d).length
  in count_digit 8 800 = 161 :=
by
  sorry

end count_eight_in_1_to_800_l578_578007


namespace number_of_arrangements_l578_578506

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578506


namespace remaining_squares_total_area_removed_l578_578167

-- Define the initial conditions
def initial_square_area : ℝ := 1
def side_length (n : ℕ) : ℝ := 1 / 3^n

-- Prove the number of remaining squares with side length 1/3^n after n steps is 8^n
theorem remaining_squares (n : ℕ) : (8 : ℝ)^n = 8^n :=
  by sorry

-- Prove the total area of the removed squares after n steps is 1 - (8/9)^n
theorem total_area_removed (n : ℕ) : (1 : ℝ) - (8 / 9)^n = 1 - (8 / 9)^n :=
  by sorry

end remaining_squares_total_area_removed_l578_578167


namespace smallest_n_inverse_mod_2310_l578_578190

theorem smallest_n_inverse_mod_2310 :
  ∃ n : ℕ, n > 1 ∧ nat.gcd n 2310 = 1 ∧ ∀ m : ℕ, m > 1 ∧ nat.gcd m 2310 = 1 → m ≥ n :=
begin
  use 13, -- Provide the witness value
  split,
  { -- n > 1
    norm_num,
  },
  split,
  { -- gcd(13, 2310) = 1
    norm_num,
  },
  { -- Minimality of 13
    intros m hm1 hm2,
    by_contra h,
    cases m,
    { -- Case m = 0 is a contradiction as m > 1
      norm_num at hm1,
    },
    cases m,
    { -- Case m = 1 is a contradiction as m > 1
      norm_num at hm1,
    },
    { -- Case m ≥ 2
      have : m < 13,
      { linarith, },
      interval_cases m,
      norm_num at hm2,
      sorry
    }
  }
end

end smallest_n_inverse_mod_2310_l578_578190


namespace banana_arrangements_l578_578885

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578885


namespace numberOfWaysToArrangeBANANA_l578_578735

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578735


namespace banana_arrangements_l578_578618

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578618


namespace even_function_derivative_is_odd_l578_578120

-- Assume f is a function defined on ℝ such that f is even
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- The derivative of f, denoted as f'
noncomputable def f' (f : ℝ → ℝ) : ℝ → ℝ := λ x, derivative f x

-- The statement to prove
theorem even_function_derivative_is_odd (f : ℝ → ℝ) (h_even : is_even_function f) : 
  is_even_function (λ x, -f' f x) :=
sorry

end even_function_derivative_is_odd_l578_578120


namespace number_of_arrangements_l578_578489

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578489


namespace number_of_arrangements_BANANA_l578_578777

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578777


namespace a_2016_is_neg_one_l578_578173

noncomputable def seq : ℕ → ℚ
| 1       := 2
| (n + 1) := 1 - 1 / seq n

theorem a_2016_is_neg_one : seq 2016 = -1 := 
by {
    sorry
}

end a_2016_is_neg_one_l578_578173


namespace intersection_points_rectangular_coords_l578_578058

theorem intersection_points_rectangular_coords :
  ∃ (x y : ℝ),
    (∃ (ρ θ : ℝ), ρ = 2 * Real.cos θ ∧ ρ^2 * (Real.cos θ)^2 - 4 * ρ^2 * (Real.sin θ)^2 = 4 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
    (x = (1 + Real.sqrt 13) / 3 ∧ y = 0) := 
sorry

end intersection_points_rectangular_coords_l578_578058


namespace permutations_BANANA_l578_578546

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578546


namespace permutations_BANANA_l578_578533

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578533


namespace permutations_of_BANANA_l578_578380

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578380


namespace arrangements_of_BANANA_l578_578441

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578441


namespace BANANA_arrangement_l578_578478

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578478


namespace numberOfWaysToArrangeBANANA_l578_578757

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578757


namespace complex_exchange_of_apartments_in_two_days_l578_578053

theorem complex_exchange_of_apartments_in_two_days :
  ∀ (n : ℕ) (p : Fin n → Fin n), ∃ (day1 day2 : Fin n → Fin n),
    (∀ x : Fin n, p (day1 x) = day2 x ∨ day1 (p x) = day2 x) ∧
    (∀ x : Fin n, day1 x ≠ x) ∧
    (∀ x : Fin n, day2 x ≠ x) :=
by
  sorry

end complex_exchange_of_apartments_in_two_days_l578_578053


namespace money_distribution_l578_578257

-- Declare the variables and the conditions as hypotheses
theorem money_distribution (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 40) :
  B + C = 340 :=
by
  sorry

end money_distribution_l578_578257


namespace arrange_banana_l578_578579

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578579


namespace arrangement_count_BANANA_l578_578685

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578685


namespace number_of_unique_permutations_BANANA_l578_578333

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578333


namespace banana_arrangements_l578_578982

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578982


namespace arrangements_of_BANANA_l578_578430

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578430


namespace BANANA_arrangement_l578_578475

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578475


namespace number_of_arrangements_BANANA_l578_578781

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578781


namespace permutations_BANANA_l578_578530

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578530


namespace cricketer_stats_l578_578237

variables (W B : ℕ)

-- Conditions from the problem
def runs_before_match (W : ℕ) : ℕ := 12.4 * W
def balls_before_match (W : ℕ) : ℕ := 30 * W

-- New average and total after the new match
def total_wickets_after (W : ℕ) : ℕ := W + 5
def total_runs_after (W : ℕ) : ℕ := runs_before_match W + 26
def new_average (W : ℕ) : ℕ := total_runs_after W / total_wickets_after W

-- Given improved strike rate
def new_strike_rate : ℕ := 28

-- Initial Strike Rate
def initial_strike_rate (W : ℕ) : ℕ := 30 * W

-- Proof Goal
theorem cricketer_stats :
  (12.4 * W + 26) / (W + 5) = 12 → W = 85 ∧ 30 * W = 2550 := 
by sorry

end cricketer_stats_l578_578237


namespace banana_arrangements_l578_578974

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578974


namespace distinct_permutations_BANANA_l578_578646

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578646


namespace count_8_up_to_800_l578_578008

def count_digit_occurrences (digit : Nat) (n : Nat) : Nat :=
  let rec countDigit := (digit n : Nat) -> Nat
    | 0 => 0
    | k + 1 => (if k + 1 / 10 = digit then 1 else 0) + countDigit (k / 10)
  countDigit n

theorem count_8_up_to_800 :
  count_digit_occurrences 8 800 = 161 :=
by
  sorry

end count_8_up_to_800_l578_578008


namespace max_value_of_exp_l578_578101

theorem max_value_of_exp (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) : 
  a^2 * b^3 * c ≤ 27 / 16 := 
  sorry

end max_value_of_exp_l578_578101


namespace banana_unique_permutations_l578_578860

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578860


namespace numberOfWaysToArrangeBANANA_l578_578746

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578746


namespace permutations_BANANA_l578_578934

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578934


namespace banana_arrangements_l578_578638

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578638


namespace numberOfWaysToArrangeBANANA_l578_578727

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578727


namespace arrange_banana_l578_578599

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578599


namespace arrangement_count_BANANA_l578_578705

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578705


namespace BANANA_arrangement_l578_578466

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578466


namespace correct_propositions_l578_578261

def Line := Type
def Plane := Type

variables (m n: Line) (α β γ: Plane)

-- Conditions from the problem statement
axiom perp (x: Line) (y: Plane): Prop -- x ⊥ y
axiom parallel (x: Line) (y: Plane): Prop -- x ∥ y
axiom perp_planes (x: Plane) (y: Plane): Prop -- x ⊥ y
axiom parallel_planes (x: Plane) (y: Plane): Prop -- x ∥ y

-- Given the conditions
axiom h1: perp m α
axiom h2: parallel n α
axiom h3: perp_planes α γ
axiom h4: perp_planes β γ
axiom h5: parallel_planes α β
axiom h6: parallel_planes β γ
axiom h7: parallel m α
axiom h8: parallel n α
axiom h9: perp m n
axiom h10: perp m γ

-- Lean statement for the problem: Prove that Propositions ① and ④ are correct.
theorem correct_propositions : (perp m n) ∧ (perp m γ) :=
by sorry -- Proof steps are not required.

end correct_propositions_l578_578261


namespace number_of_unique_permutations_BANANA_l578_578353

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578353


namespace banana_arrangements_l578_578991

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578991


namespace number_of_unique_permutations_BANANA_l578_578360

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578360


namespace arrangement_count_BANANA_l578_578713

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578713


namespace find_OH_squared_l578_578105

variables {O H : Type} {a b c R : ℝ}

-- Given conditions
def is_circumcenter (O : Type) (ABC : Type) := true -- Placeholder definition
def is_orthocenter (H : Type) (ABC : Type) := true -- Placeholder definition
def circumradius (O : Type) (R : ℝ) := true -- Placeholder definition
def sides_squared_sum (a b c : ℝ) := a^2 + b^2 + c^2

-- The theorem to be proven
theorem find_OH_squared (O H : Type) (a b c : ℝ) (R : ℝ) 
  (circ : is_circumcenter O ABC) 
  (orth: is_orthocenter H ABC) 
  (radius : circumradius O R) 
  (terms_sum : sides_squared_sum a b c = 50)
  (R_val : R = 10) 
  : OH^2 = 850 := 
sorry

end find_OH_squared_l578_578105


namespace distinct_permutations_BANANA_l578_578658

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578658


namespace find_g_of_conditions_l578_578138

noncomputable def g : ℚ → ℚ := sorry -- g is a rational function, precise definition omitted with sorry

theorem find_g_of_conditions :
  (∀ x : ℚ, x ≠ 0 → 5 * g ((2 : ℚ) / x) - (3 * g x) / x = x^3) →
  g (-3) = 67 / 20 :=
begin
  intros h,
  sorry -- Proof omitted as it's not required here
end

end find_g_of_conditions_l578_578138


namespace banana_arrangements_l578_578967

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578967


namespace banana_unique_permutations_l578_578845

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578845


namespace banana_arrangements_l578_578612

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578612


namespace permutations_BANANA_l578_578564

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578564


namespace banana_arrangements_l578_578610

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578610


namespace binom_60_3_eq_34220_l578_578302

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l578_578302


namespace banana_arrangements_l578_578639

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578639


namespace BANANA_arrangement_l578_578487

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578487


namespace no_sum_of_two_cubes_l578_578128

theorem no_sum_of_two_cubes (n : ℕ) :
  ¬ (∃ a b : ℕ, 10^(3*n+1) = a^3 + b^3) := by
  have h_mod_7 : (∃ a b : ℕ, 10^(3*n+1) = a^3 + b^3) → (∃ a b : ℕ, (10^(3*n+1)) % 7 = (a^3 + b^3) % 7) := by
    sorry -- We'll use modulo arithmetic here
  have h_10_mod_7 : 10 % 7 = 3 := by norm_num
  have h3n1_mod_7 : 10^(3*n+1) % 7 = 3 ^ (3 * n + 1) % 7 := by
    rw [←pow_mod, h_10_mod_7]
    norm_num
  have cubes_mod_7 : ∀ a : ℕ, a^3 % 7 = 0 ∨ a^3 % 7 = 1 ∨ a^3 % 7 = 6 := by
    intro a
    fin_cases (a % 7) with b;
    norm_num
  have sum_cubes_mod_7 : ∀ a b : ℕ, (a^3 + b^3) % 7 ∈ {0, 1, 2, 5, 6} := by
    intros a b
    rcases cubes_mod_7 a with ⟨h1 | h1 | h1⟩;
    rcases cubes_mod_7 b with ⟨h2 | h2 | h2⟩;
    norm_num;
    finish
  have no_overlap : ∀ k : ℕ, (k % 7 = 3 ∨ k % 7 = 4) → k % 7 ∉ {0, 1, 2, 5, 6} := by
    finish
  intros h
  rcases h_mod_7 h with ⟨a, b, h_eq⟩
  rw h_eq at h3n1_mod_7
  cases (3 ^ (3 * n + 1) % 7) with h3_ | h4_
  { exact no_overlap (a^3 + b^3) h3_ (sum_cubes_mod_7 a b) }
  { exact no_overlap (a^3 + b^3) h4_ (sum_cubes_mod_7 a b) }

end no_sum_of_two_cubes_l578_578128


namespace cost_whitewashing_l578_578149

theorem cost_whitewashing
  (length : ℝ) (breadth : ℝ) (height : ℝ)
  (door_height : ℝ) (door_width : ℝ)
  (window_height : ℝ) (window_width : ℝ)
  (num_windows : ℕ) (cost_per_square_foot : ℝ)
  (room_dimensions : length = 25 ∧ breadth = 15 ∧ height = 12)
  (door_dimensions : door_height = 6 ∧ door_width = 3)
  (window_dimensions : window_height = 4 ∧ window_width = 3)
  (num_windows_condition : num_windows = 3)
  (cost_condition : cost_per_square_foot = 8) :
  (2 * (length + breadth) * height - (door_height * door_width + num_windows * window_height * window_width)) * cost_per_square_foot = 7248 := 
by
  sorry

end cost_whitewashing_l578_578149


namespace binom_60_3_eq_34220_l578_578305

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l578_578305


namespace permutations_of_BANANA_l578_578374

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578374


namespace banana_unique_permutations_l578_578840

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578840


namespace range_of_m_in_circle_l578_578239

open Real

noncomputable def P_1 (a b : ℕ) : ℚ :=
  if (a = 2 ∧ b = 4) ∨ (a = 3 ∧ b = 6) then 1 / 36 else 0

noncomputable def P_2 (a b : ℕ) : ℚ :=
  if ¬((a = 2 ∧ b = 4) ∨ (a = 3 ∧ b = 6) ∨ (a = 1 ∧ b = 2)) then 1 else 33 / 36

theorem range_of_m_in_circle :
  let m : ℝ := sorry
  ∀ (P₁ P₂ : ℚ),
    P₁ = 1 / 18 →
    P₂ = 11 / 12 →
    (P₁ - m)^2 + P₂^2 < 137 / 144 ↔ -5 / 18 < m ∧ m < 7 / 18 :=
begin
  intros,
  sorry,
end

end range_of_m_in_circle_l578_578239


namespace numberOfWaysToArrangeBANANA_l578_578724

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578724


namespace smallest_number_divided_into_18_and_60_groups_l578_578191

theorem smallest_number_divided_into_18_and_60_groups : ∃ n : ℕ, (∀ m : ℕ, (m % 18 = 0 ∧ m % 60 = 0) → n ≤ m) ∧ (n % 18 = 0 ∧ n % 60 = 0) ∧ n = 180 :=
by
  use 180
  sorry

end smallest_number_divided_into_18_and_60_groups_l578_578191


namespace number_of_arrangements_l578_578517

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578517


namespace number_of_arrangements_BANANA_l578_578768

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578768


namespace number_of_ways_to_arrange_BANANA_l578_578835

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578835


namespace numberOfWaysToArrangeBANANA_l578_578743

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578743


namespace banana_arrangements_l578_578909

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578909


namespace banana_arrangements_l578_578978

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578978


namespace distinct_permutations_BANANA_l578_578681

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578681


namespace arrange_banana_l578_578575

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578575


namespace distinct_permutations_BANANA_l578_578670

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578670


namespace arrangement_count_BANANA_l578_578684

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578684


namespace banana_arrangements_l578_578628

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578628


namespace brooklyn_total_contribution_over_year_l578_578275

-- Define the series
def monthly_donation (n : ℕ) : ℝ :=
  1453 * (1.04 ^ (n - 1))

-- Define the sum of the first n terms in the series
def total_contribution (n : ℕ) : ℝ :=
  (finset.range n).sum (λ i => monthly_donation (i + 1))

-- Define the specific problem statement for 12 months
theorem brooklyn_total_contribution_over_year :
  abs (total_contribution 12 - 21839.99) < 1 :=
by
  sorry

end brooklyn_total_contribution_over_year_l578_578275


namespace permutations_of_BANANA_l578_578404

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578404


namespace number_of_unique_permutations_BANANA_l578_578361

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578361


namespace banana_arrangements_l578_578912

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578912


namespace number_of_arrangements_BANANA_l578_578766

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578766


namespace sign_pyramid_solution_l578_578046

-- Each cell in the pyramid is either +1 or -1. 
-- The rule for the sign in the cell is as follows:
-- "+" if the same, "-" if different.
-- We need to fill the bottom row such that the topmost cell is "+".

def sign_pyramid_top (a b c d e : ℤ) : ℤ :=
  (a * b * c * c * d) * (b * c * c * d * e)

def is_positive (x : ℤ) : Prop := x = 1

theorem sign_pyramid_solution :
  (finset.univ.filter (λ (t : (ℤ × ℤ × ℤ × ℤ × ℤ)), 
    let (a, b, c, d, e) := t in (a = 1 ∨ a = -1) ∧
    (b = 1 ∨ b = -1) ∧
    (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧
    (e = 1 ∨ e = -1) ∧
    is_positive (sign_pyramid_top a b c d e))).card = 12 := 
sorry

end sign_pyramid_solution_l578_578046


namespace permutations_BANANA_l578_578556

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578556


namespace total_animals_seen_correct_l578_578269

-- Define the number of beavers in the morning
def beavers_morning : ℕ := 35

-- Define the number of chipmunks in the morning
def chipmunks_morning : ℕ := 60

-- Define the number of beavers in the afternoon (tripled)
def beavers_afternoon : ℕ := 3 * beavers_morning

-- Define the number of chipmunks in the afternoon (decreased by 15)
def chipmunks_afternoon : ℕ := chipmunks_morning - 15

-- Calculate the total number of animals seen in the morning
def total_morning : ℕ := beavers_morning + chipmunks_morning

-- Calculate the total number of animals seen in the afternoon
def total_afternoon : ℕ := beavers_afternoon + chipmunks_afternoon

-- The total number of animals seen that day
def total_animals_seen : ℕ := total_morning + total_afternoon

theorem total_animals_seen_correct :
  total_animals_seen = 245 :=
by
  -- skipping the proof
  sorry

end total_animals_seen_correct_l578_578269


namespace permutations_BANANA_l578_578537

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578537


namespace banana_unique_permutations_l578_578854

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578854


namespace trig_simplify_l578_578134

theorem trig_simplify (x : ℝ) :
  (cos (3 * x))^2 = 1 - (sin (3 * x))^2 →
  (cos x)^2 = 1 - (sin x)^2 →
  (sin x + sin (3 * x)) / (1 + (cos x)^2 + (cos (3 * x))^2) = 
  (2 * sin (2 * x) * cos x) / (3 - (sin x)^2 - (sin (3 * x))^2) :=
by
  intros h1 h2
  sorry

end trig_simplify_l578_578134


namespace banana_arrangements_l578_578987

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578987


namespace permutations_of_BANANA_l578_578376

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578376


namespace cauchy_inequality_inequality_b_l578_578181

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable {b : Fin n → ℝ}

theorem cauchy_inequality 
  (h_pos_a : ∀ i, 0 < a i) :
  Real.sqrt (∏ i, a i) ≤ (∑ i, a i) / n := 
sorry

theorem inequality_b 
  (h_pos_b : ∀ i, 0 < b i) :
  (∑ i, b i / n)^(∑ i, b i) ≤ ∏ i, (b i)^ b i :=
sorry

end cauchy_inequality_inequality_b_l578_578181


namespace arrangement_count_BANANA_l578_578689

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578689


namespace sum_of_digits_eq_4_l578_578051

theorem sum_of_digits_eq_4 (A B C D X Y : ℕ) (h1 : A + B + C + D = 22) (h2 : B + D = 9) (h3 : X = 1) (h4 : Y = 3) :
    X + Y = 4 :=
by
  sorry

end sum_of_digits_eq_4_l578_578051


namespace smooth_number_sum_l578_578227

def is_smooth_number (x : ℕ) : Prop :=
  ∃ m n : ℕ, x = 2 ^ m * 3 ^ n

def S : set (ℕ × ℕ × ℕ) :=
  { (a, b, c) | is_smooth_number a ∧ is_smooth_number b ∧ is_smooth_number c ∧
    nat.gcd a b ≠ nat.gcd b c ∧ nat.gcd b c ≠ nat.gcd c a ∧ nat.gcd c a ≠ nat.gcd a b }

noncomputable def sum_inv_abc : ℚ :=
  ∑ (p : ℕ × ℕ × ℕ) in S.to_finset, (1 : ℚ) / (p.1 * p.2.1 * p.2.2)

theorem smooth_number_sum : sum_inv_abc = 162 / 91 := sorry

end smooth_number_sum_l578_578227


namespace numberOfWaysToArrangeBANANA_l578_578725

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578725


namespace number_of_ways_to_arrange_BANANA_l578_578810

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578810


namespace store_sale_money_left_l578_578252

/--
A store decides to shut down and sell all of its inventory. They have 2000 different items divided into 3 categories:
Category A, Category B, and Category C. Category A has 1000 items with a normal retail price of $50 each,
Category B has 700 items with a normal retail price of $75 each, and Category C has 300 items with a normal retail price of $100 each.

For Category A, they offer an 80% discount and manage to sell 85% of the items, for Category B, they offer 70% discount and manage to sell 75% of the items,
and for Category C, they offer 60% discount and manage to sell 90% of the items. They owe $15,000 to their creditors. Prove that the money they have left after the sale is $16112.5.
-/
theorem store_sale_money_left : 
  let category_A_items := 1000
      category_B_items := 700
      category_C_items := 300
      price_A := 50
      price_B := 75
      price_C := 100
      discount_A := 0.80
      discount_B := 0.70
      discount_C := 0.60
      sold_percentage_A := 0.85
      sold_percentage_B := 0.75
      sold_percentage_C := 0.90
      creditors := 15000
      sale_price_A := price_A * (1 - discount_A)
      sale_price_B := price_B * (1 - discount_B)
      sale_price_C := price_C * (1 - discount_C)
      items_sold_A := category_A_items * sold_percentage_A
      items_sold_B := category_B_items * sold_percentage_B
      items_sold_C := category_C_items * sold_percentage_C
      revenue_A := sale_price_A * items_sold_A
      revenue_B := sale_price_B * items_sold_B
      revenue_C := sale_price_C * items_sold_C
      total_revenue := revenue_A + revenue_B + revenue_C
      money_left := total_revenue - creditors
  in 
  money_left = 16112.5 := 
by
  sorry

end store_sale_money_left_l578_578252


namespace banana_unique_permutations_l578_578852

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578852


namespace permutations_of_BANANA_l578_578399

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578399


namespace irrational_count_l578_578263

def is_irrational (x : ℝ) : Prop :=
  ¬ ∃ p q : ℤ, q ≠ 0 ∧ x = p / q

def irrationals_among_given_numbers : ℕ :=
by
  let a := (1 / 2 : ℝ)
  let b := Real.sqrt 8
  let c := (3.14159 : ℝ)
  let d := - (Real.cbrt 27)
  let e := (0 : ℝ)
  let f := Real.sqrt 2 + 1
  let g := Real.pi / 3

  have hab : is_irrational b := 
  by
    sorry

  have haf : is_irrational f := 
  by
    sorry

  have hag : is_irrational g := 
  by
    sorry

  have hna : ¬is_irrational a := 
  by
    sorry

  have hnc : ¬is_irrational c := 
  by
    sorry

  have hnd : ¬is_irrational d := 
  by
    sorry

  have hne : ¬is_irrational e := 
  by
    sorry

  exact 3

theorem irrational_count : irrationals_among_given_numbers = 3 :=
by
  have h1 := hab
  have h2 := haf
  have h3 := hag
  have h4 := hna
  have h5 := hnc
  have h6 := hnd
  have h7 := hne

  exact rfl

end irrational_count_l578_578263


namespace number_of_arrangements_l578_578514

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578514


namespace number_of_arrangements_BANANA_l578_578770

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578770


namespace banana_arrangements_l578_578896

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578896


namespace arrangements_of_BANANA_l578_578443

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578443


namespace numberOfWaysToArrangeBANANA_l578_578753

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578753


namespace distinct_permutations_BANANA_l578_578675

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578675


namespace banana_unique_permutations_l578_578869

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578869


namespace BANANA_arrangement_l578_578468

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578468


namespace number_of_arrangements_l578_578521

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578521


namespace numberOfWaysToArrangeBANANA_l578_578747

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578747


namespace permutations_BANANA_l578_578941

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578941


namespace distinct_permutations_BANANA_l578_578677

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578677


namespace banana_arrangements_l578_578916

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578916


namespace number_of_unique_permutations_BANANA_l578_578347

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578347


namespace banana_arrangements_l578_578973

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578973


namespace arrange_banana_l578_578591

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578591


namespace triangle_problem_l578_578033

theorem triangle_problem (a b c : ℝ) (A B C : ℝ)
  (h1 : A + B + C = π)
  (h2 : a^2 + c^2 = b^2 + sqrt 2 * a * c)
  (ha_pos : 0 < A) (hA_lt_π : A < π)
  (hb_pos : 0 < B) (hB_lt_π : B < π)
  (hc_pos : 0 < C) (hC_lt_π : C < π)
  (habc : a^2 = b^2 + c^2 - 2 * b * c * cos A)
  (hbca : b^2 = a^2 + c^2 - 2 * a * c * cos B)
  (hcab : c^2 = a^2 + b^2 - 2 * a * b * cos C) :
  (B = π / 4) ∧ (∃ A, ∃ C, sqrt 2 * cos A + cos C = 1) :=
by
  sorry

end triangle_problem_l578_578033


namespace arrangement_count_BANANA_l578_578712

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578712


namespace arrange_banana_l578_578570

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578570


namespace number_of_ways_to_arrange_BANANA_l578_578838

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578838


namespace banana_arrangements_l578_578888

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578888


namespace arrangements_of_BANANA_l578_578428

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578428


namespace arrangement_count_BANANA_l578_578708

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578708


namespace num_users_in_china_in_2022_l578_578258

def num_users_scientific (n : ℝ) : Prop :=
  n = 1.067 * 10^9

theorem num_users_in_china_in_2022 :
  num_users_scientific 1.067e9 :=
by
  sorry

end num_users_in_china_in_2022_l578_578258


namespace permutations_of_BANANA_l578_578410

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578410


namespace binom_60_3_eq_34220_l578_578315

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l578_578315


namespace distinct_permutations_BANANA_l578_578671

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578671


namespace arithmetic_sequence_proof_l578_578052

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_proof
  (a₁ d : ℝ)
  (h : a 4 a₁ d + a 6 a₁ d + a 8 a₁ d + a 10 a₁ d + a 12 a₁ d = 120) :
  a 7 a₁ d - (1 / 3) * a 5 a₁ d = 16 :=
by
  sorry

end arithmetic_sequence_proof_l578_578052


namespace permutations_BANANA_l578_578562

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578562


namespace permutations_BANANA_l578_578945

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578945


namespace banana_arrangements_l578_578994

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578994


namespace polynomial_value_at_neg2_l578_578180

def polynomial (x : ℝ) : ℝ :=
  x^6 - 5 * x^5 + 6 * x^4 + x^2 + 0.3 * x + 2

theorem polynomial_value_at_neg2 :
  polynomial (-2) = 325.4 :=
by
  sorry

end polynomial_value_at_neg2_l578_578180


namespace banana_arrangements_l578_578642

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578642


namespace binom_60_3_eq_34220_l578_578300

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l578_578300


namespace numberOfWaysToArrangeBANANA_l578_578730

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578730


namespace permutations_of_BANANA_l578_578394

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578394


namespace digit_8_count_1_to_800_l578_578013

theorem digit_8_count_1_to_800 : (count_digit 8 (range 1 801)) = 160 :=
sorry

-- Assuming helper functions are already defined:
-- count_digit d lst: counts how many times the digit d appears in the list of integers lst.
-- range a b: generates a list of integers from a to b-1.

end digit_8_count_1_to_800_l578_578013


namespace max_interchanges_l578_578208

theorem max_interchanges (n : ℕ) (coins : ℕ := 2^n) 
  (kids : list ℕ) (h_len : kids.length = 2) 
  (initial_state : kids.sum = coins)
  (interchange_possible : ∀ k1 k2, k1 ∈ kids → k2 ∈ kids → 
    (k1 ≥ coins / 2 ∨ k2 ≥ coins / 2 → k1 + k2 = coins)) : 
  (∃ max_interchange_steps, max_possible_interchanges n kids max_interchange_steps ∧ 
  max_interchange_steps = n) :=
sorry

end max_interchanges_l578_578208


namespace permutations_BANANA_l578_578936

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578936


namespace banana_arrangements_l578_578609

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578609


namespace numberOfWaysToArrangeBANANA_l578_578749

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578749


namespace arrangement_count_BANANA_l578_578715

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578715


namespace distinct_permutations_BANANA_l578_578666

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578666


namespace number_of_ways_to_arrange_BANANA_l578_578824

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578824


namespace roots_of_quadratic_l578_578163

theorem roots_of_quadratic (x : ℝ) : x^2 - 5 * x = 0 ↔ (x = 0 ∨ x = 5) := by 
  sorry

end roots_of_quadratic_l578_578163


namespace number_of_ways_to_arrange_BANANA_l578_578819

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578819


namespace number_of_unique_permutations_BANANA_l578_578343

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578343


namespace banana_arrangements_l578_578961

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578961


namespace compare_fractions_l578_578288

-- Define the fractions
def frac1 : ℚ := -2/3
def frac2 : ℚ := -3/4

-- Prove that -2/3 > -3/4
theorem compare_fractions : frac1 > frac2 :=
by {
  sorry
}

end compare_fractions_l578_578288


namespace dihedral_angle_of_folded_ellipse_l578_578156

noncomputable def ellipse_dihedral_angle : Prop :=
  let a := 4 in
  let c := 2 in
  let cos_theta := c / a in
  cos_theta = 1/2 ∧ θ = 60

theorem dihedral_angle_of_folded_ellipse :
  (∃ θ : ℝ, θ = 60) := sorry

end dihedral_angle_of_folded_ellipse_l578_578156


namespace number_of_unique_permutations_BANANA_l578_578357

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578357


namespace arrangement_count_BANANA_l578_578716

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578716


namespace banana_arrangements_l578_578637

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578637


namespace count_eight_in_1_to_800_l578_578005

theorem count_eight_in_1_to_800 : 
  let count_digit (d n : Nat) : Nat := 
    (List.range (n + 1)).filter (λ x => List.contains (Nat.digits 10 x) d).length
  in count_digit 8 800 = 161 :=
by
  sorry

end count_eight_in_1_to_800_l578_578005


namespace numberOfWaysToArrangeBANANA_l578_578751

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578751


namespace permutations_BANANA_l578_578550

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578550


namespace number_of_arrangements_l578_578525

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578525


namespace ABD_F_tangential_quad_radius_inequality_l578_578088

variable {A B C D F : Type}
variable [Triangle A B C]
variable [AC_eq_2AB : AC = 2 * AB]
variable [F_midpoint_AC : F = midpoint A C]
variable [D_angle_bisector_A : is_angle_bisector A D BC]

-- Part (a)
theorem ABD_F_tangential_quad (AC_eq_2AB : AC = 2 * AB) (F_midpoint_AC : F = midpoint A C) (D_angle_bisector_A : is_angle_bisector A D BC) : 
  is_tangential_quadrilateral A B D F :=
sorry

-- Part (b)
variable {r1 r2 : ℝ}

theorem radius_inequality (AC_eq_2AB : AC = 2 * AB) (F_midpoint_AC : F = midpoint A C) (D_angle_bisector_A : is_angle_bisector A D BC) (r1_gt_r2 : r1 > r2) :
  1 < r1 / r2 ∧ r1 / r2 < 2 :=
sorry

end ABD_F_tangential_quad_radius_inequality_l578_578088


namespace number_of_factors_of_M_l578_578332

-- Define M based on the given condition
def M : ℕ := 2^4 * 3^3 * 5^2 * 7^1

-- Define the statement to be proved
theorem number_of_factors_of_M : 
  let a_range := (range 5).toFinset
  let b_range := (range 4).toFinset
  let c_range := (range 3).toFinset
  let d_range := (range 2).toFinset
  a_range.card * b_range.card * c_range.card * d_range.card = 120 := 
by {
  -- We can use sorry here as the proof is not required
  sorry
}

end number_of_factors_of_M_l578_578332


namespace number_of_arrangements_BANANA_l578_578796

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578796


namespace arrangements_of_BANANA_l578_578423

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578423


namespace number_of_arrangements_l578_578497

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578497


namespace number_of_ways_to_arrange_BANANA_l578_578833

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578833


namespace circle_area_eq_l578_578177

theorem circle_area_eq {P Q R I : Type} [MetricSpace P] [MetricSpace Q] 
  (equilateral : EquilateralTriangle P Q R) (side_length : ∀ x y ∈ {P, Q, R}, dist x y = 8) 
  (incenter : Incenter I P Q R) 
  : circle_area P I R = 64 * π / 3 := 
sorry

end circle_area_eq_l578_578177


namespace banana_unique_permutations_l578_578841

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578841


namespace arrangement_count_BANANA_l578_578719

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578719


namespace arrangements_of_BANANA_l578_578413

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578413


namespace banana_unique_permutations_l578_578846

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578846


namespace permutations_BANANA_l578_578549

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578549


namespace banana_arrangements_l578_578887

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578887


namespace banana_arrangements_l578_578904

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578904


namespace arrangements_of_BANANA_l578_578444

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578444


namespace banana_arrangements_l578_578883

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578883


namespace a_beats_b_by_seconds_l578_578043

noncomputable def speed_of_A (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

noncomputable def time_to_cover_distance (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

theorem a_beats_b_by_seconds :
  let total_distance := 1000.0
  let remaining_distance_B := 920.0
  let time_A := 92.0
  let A_speed := speed_of_A total_distance time_A
  let distance_A_beats_B := 80.0
  let time_A_beats_B := time_to_cover_distance distance_A_beats_B A_speed
  in time_A_beats_B = 7.36 :=
by
  sorry

end a_beats_b_by_seconds_l578_578043


namespace permutations_of_BANANA_l578_578396

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578396


namespace find_incorrect_statement_l578_578198

theorem find_incorrect_statement :
  ¬ (∀ a b c : ℝ, c ≠ 0 → (a < b → a * c^2 < b * c^2)) :=
by
  sorry

end find_incorrect_statement_l578_578198


namespace permutations_BANANA_l578_578928

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578928


namespace banana_arrangements_l578_578980

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578980


namespace banana_arrangements_l578_578633

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578633


namespace sum_of_fractions_eq_l578_578192

noncomputable def sum_of_fractions : ℚ :=
  ∑ n in (finset.range 2023).map (λ x, x + 1), 3 / (n * (n + 3))

theorem sum_of_fractions_eq :
  sum_of_fractions = (11 * 2024 * 2025 * 2026 - (2025 * 2026 + 2024 * 2026 + 2024 * 2025)) / (6 * 2024 * 2025 * 2026) :=
sorry

end sum_of_fractions_eq_l578_578192


namespace mike_score_l578_578118

variables (max_marks passing_percentage shortfall : ℕ)

def passing_marks : ℕ := (passing_percentage * max_marks) / 100
def mike_marks : ℕ := passing_marks - shortfall

theorem mike_score 
  (max_marks = 800) 
  (passing_percentage = 30) 
  (shortfall = 28) 
  : mike_marks = 212 :=
by
  sorry

end mike_score_l578_578118


namespace arrange_banana_l578_578592

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578592


namespace arrangement_count_BANANA_l578_578710

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578710


namespace number_of_arrangements_l578_578504

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578504


namespace parallel_line_distance_l578_578056

theorem parallel_line_distance 
    (A_upper : ℝ) (A_middle : ℝ) (A_lower : ℝ)
    (A_total : ℝ) (A_half : ℝ)
    (h_upper : A_upper = 3)
    (h_middle : A_middle = 5)
    (h_lower : A_lower = 2) 
    (h_total : A_total = A_upper + A_middle + A_lower)
    (h_half : A_half = A_total / 2) :
    ∃ d : ℝ, d = 2 + 0.6 ∧ A_middle * 0.6 = 3 := 
sorry

end parallel_line_distance_l578_578056


namespace number_of_arrangements_BANANA_l578_578793

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578793


namespace binomial_60_3_eq_34220_l578_578292

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l578_578292


namespace number_of_ways_to_arrange_BANANA_l578_578836

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578836


namespace BANANA_arrangement_l578_578471

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578471


namespace banana_arrangements_l578_578979

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578979


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l578_578221

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l578_578221


namespace a_2017_value_l578_578061

noncomputable def a : ℕ → ℚ
| 1 := -1
| (n+1) := a n + 1/(n * (n+1))

theorem a_2017_value : a 2017 = -1/2017 := 
sorry

end a_2017_value_l578_578061


namespace mass_percentage_of_H_in_CaOH2_is_correct_l578_578328

def mass_percentage_H_in_CaOH2 : ℕ :=
  let Ca_mass := 40.08
  let O_mass := 16.00
  let H_mass := 1.01
  let CaOH2_molar_mass := Ca_mass + 2 * O_mass + 2 * H_mass
  let H_mass_in_CaOH2 := 2 * H_mass
  ((H_mass_in_CaOH2 / CaOH2_molar_mass) * 100).to_nat

theorem mass_percentage_of_H_in_CaOH2_is_correct :
  mass_percentage_H_in_CaOH2 = 2.73 :=
by
  -- proofs go here
  sorry

end mass_percentage_of_H_in_CaOH2_is_correct_l578_578328


namespace distinct_permutations_BANANA_l578_578673

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578673


namespace arrangement_count_BANANA_l578_578698

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578698


namespace integer_satisfies_l578_578185

theorem integer_satisfies (n : ℤ) (h1 : 0 ≤ n ∧ n < 9) (h2 : -1651 ≡ n [MOD 9]) : n = 5 :=
sorry

end integer_satisfies_l578_578185


namespace largest_cube_side_length_largest_rect_parallelepiped_dimensions_l578_578080

variables {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (a): Side length of the largest cube
theorem largest_cube_side_length :
  let a₀ := a * b * c / (a * b + b * c + a * c) in
  ∃ a₀, a₀ = a * b * c / (a * b + b * c + a * c) :=
begin
  sorry
end

-- Part (b): Dimensions of the largest rectangular parallelepiped
theorem largest_rect_parallelepiped_dimensions :
  let x := a / 3, y := b / 3, z := c / 3 in
  ∃ x y z, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
begin
  sorry
end

end largest_cube_side_length_largest_rect_parallelepiped_dimensions_l578_578080


namespace regression_fitting_correct_l578_578050

/-- 
  A theorem representing the correctness of statements about the fitting effect in regression analysis.
  Conditions:
  - ① The fitting effect is judged by the value of the correlation coefficient \( r \).
  - ② The fitting effect is judged by the value of the coefficient of determination \( R^2 \).
  - ③ The fitting effect is judged by the sum of squared residuals.
  - ④ The fitting effect is judged by the residual plot.
  The correct judgments are ② and ④.
-/
theorem regression_fitting_correct (condition1 condition2 condition3 condition4 : Prop)
  (h1 : condition1 = "The smaller the correlation coefficient \(r\), the better")
  (h2 : condition2 = "The larger the coefficient of determination \(R^2\), the better")
  (h3 : condition3 = "The larger the sum of squared residuals, the better")
  (h4 : condition4 = "Even distribution of residuals in a horizontal band indicates better fitting, narrower band is better"):
  (condition2 ∧ condition4) ∧ ¬condition1 ∧ ¬condition3 :=
by 
  -- The proof details would go here
  sorry

end regression_fitting_correct_l578_578050


namespace numberOfWaysToArrangeBANANA_l578_578744

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578744


namespace permutations_of_BANANA_l578_578378

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578378


namespace permutations_of_BANANA_l578_578385

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578385


namespace binom_60_3_eq_34220_l578_578313

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l578_578313


namespace banana_unique_permutations_l578_578876

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578876


namespace soccer_championship_l578_578047

theorem soccer_championship (teams : Finset ℕ) (games : ℕ → ℕ → Prop) 
  (h_teams : teams.card = 18)
  (h_rounds : ∀ t ∈ teams, (Finset.filter (λ s, games t s) teams).card = 8) :
  ∃ (t1 t2 t3 : ℕ), t1 ∈ teams ∧ t2 ∈ teams ∧ t3 ∈ teams ∧ t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧ 
    ¬ games t1 t2 ∧ ¬ games t1 t3 ∧ ¬ games t2 t3 :=
sorry

end soccer_championship_l578_578047


namespace number_of_unique_permutations_BANANA_l578_578334

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578334


namespace arrange_banana_l578_578588

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578588


namespace game_winner_least_N_sum_digits_l578_578273

theorem game_winner_least_N_sum_digits (N : ℕ) (h1: 0 ≤ N) (h2 : N ≤ 999) :
  (∀ (k : ℕ), (2 ^ k * N + 100 * (k - 1) < 1000) ∧ k = 3 → (sum_digits N = 11)) :=
by
  sorry

def sum_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10)

end game_winner_least_N_sum_digits_l578_578273


namespace arrange_banana_l578_578574

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578574


namespace arrange_banana_l578_578596

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578596


namespace numberOfWaysToArrangeBANANA_l578_578740

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578740


namespace banana_arrangements_l578_578976

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578976


namespace emmanuel_gets_expected_share_l578_578270

-- Define the initial conditions
def total_jelly_beans := 500
def thomas_share_percent := 8
def sarah_share_percent := 12

-- Define the ratio parts
def barry_ratio_part := 4
def emmanuel_ratio_part := 5
def miguel_ratio_part := 6
def total_ratio_parts := barry_ratio_part + emmanuel_ratio_part + miguel_ratio_part

noncomputable def emmanuel_share : ℕ :=
  let thomas_share := (thomas_share_percent * total_jelly_beans) / 100 in
  let sarah_share := (sarah_share_percent * total_jelly_beans) / 100 in
  let remaining_jelly_beans := total_jelly_beans - thomas_share - sarah_share in
  let part_value := remaining_jelly_beans / total_ratio_parts in
  emmanuel_ratio_part * part_value

-- Formulate the theorem stating the expected result
theorem emmanuel_gets_expected_share : emmanuel_share = 133 := by
  -- Placeholder for the proof
  sorry

end emmanuel_gets_expected_share_l578_578270


namespace banana_arrangements_l578_578958

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578958


namespace permutations_BANANA_l578_578544

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578544


namespace permutations_BANANA_l578_578566

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578566


namespace arrangements_of_BANANA_l578_578431

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578431


namespace correct_choice_l578_578021

theorem correct_choice (x y m : ℝ) (h1 : x > y) (h2 : m > 0) : x - y > 0 := by
  sorry

end correct_choice_l578_578021


namespace permutations_BANANA_l578_578935

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578935


namespace arrangements_of_BANANA_l578_578425

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578425


namespace BANANA_arrangement_l578_578479

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578479


namespace probability_businessmen_wait_two_minutes_l578_578217

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l578_578217


namespace arrangement_count_BANANA_l578_578709

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578709


namespace number_of_arrangements_BANANA_l578_578773

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578773


namespace price_difference_is_correct_l578_578165

noncomputable def total_cost : ℝ := 70.93
noncomputable def cost_of_pants : ℝ := 34.0
noncomputable def cost_of_belt : ℝ := total_cost - cost_of_pants
noncomputable def price_difference : ℝ := cost_of_belt - cost_of_pants

theorem price_difference_is_correct :
  price_difference = 2.93 := by
  sorry

end price_difference_is_correct_l578_578165


namespace BANANA_arrangement_l578_578465

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578465


namespace BANANA_arrangement_l578_578482

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578482


namespace number_of_unique_permutations_BANANA_l578_578370

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578370


namespace permutations_of_BANANA_l578_578390

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578390


namespace distinct_permutations_BANANA_l578_578664

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578664


namespace arrangements_of_BANANA_l578_578417

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578417


namespace number_of_ways_to_arrange_BANANA_l578_578839

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578839


namespace arrange_banana_l578_578585

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578585


namespace arrange_banana_l578_578586

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578586


namespace permutations_BANANA_l578_578952

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578952


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l578_578225

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l578_578225


namespace fixed_cost_calculation_l578_578152

theorem fixed_cost_calculation (TC MC n FC : ℕ) (h1 : TC = 16000) (h2 : MC = 200) (h3 : n = 20) (h4 : TC = FC + MC * n) : FC = 12000 :=
by
  sorry

end fixed_cost_calculation_l578_578152


namespace banana_arrangements_l578_578607

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578607


namespace count_8_up_to_800_l578_578009

def count_digit_occurrences (digit : Nat) (n : Nat) : Nat :=
  let rec countDigit := (digit n : Nat) -> Nat
    | 0 => 0
    | k + 1 => (if k + 1 / 10 = digit then 1 else 0) + countDigit (k / 10)
  countDigit n

theorem count_8_up_to_800 :
  count_digit_occurrences 8 800 = 161 :=
by
  sorry

end count_8_up_to_800_l578_578009


namespace value_of_k_l578_578022

theorem value_of_k (k : ℚ) : 8 = 2^(3 * k + 2) → k = 1 / 3 := by
  sorry

end value_of_k_l578_578022


namespace number_of_ways_to_make_divisible_by_18_l578_578057

theorem number_of_ways_to_make_divisible_by_18 : 
  ∃ (n : ℕ), n = 26244 ∧ 
  let last_even_digits := {2, 4, 6, 8} in 
  let possible_digits := {1, 2, 3, 4, 5, 6, 7, 8, 9} in 
  let sum_non_asterisk := 2 + 0 + 1 + 6 + 0 + 2 in 
  (((sum_non_asterisk + (d_1 + d_2 + d_3 + d_4 + d_5 + d_6 + d_7 + d_8) % 9) = 7) ∧ 
  (∃ d_last ∈ last_even_digits, 
   (∀ d ∈ possible_digits, count_ways (d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_last) = n))) :=
sorry

end number_of_ways_to_make_divisible_by_18_l578_578057


namespace banana_unique_permutations_l578_578875

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578875


namespace arrangement_count_BANANA_l578_578692

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578692


namespace permutations_of_BANANA_l578_578400

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578400


namespace number_of_unique_permutations_BANANA_l578_578366

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578366


namespace distinct_permutations_BANANA_l578_578679

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578679


namespace arrange_banana_l578_578578

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578578


namespace binomial_60_3_eq_34220_l578_578298

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l578_578298


namespace breadth_of_rectangle_l578_578154

noncomputable def length (radius : ℝ) : ℝ := (1/4) * radius
noncomputable def side (sq_area : ℝ) : ℝ := Real.sqrt sq_area
noncomputable def radius (side : ℝ) : ℝ := side
noncomputable def breadth (rect_area length : ℝ) : ℝ := rect_area / length

theorem breadth_of_rectangle :
  breadth 200 (length (radius (side 1225))) = 200 / (1/4 * Real.sqrt 1225) :=
by
  sorry

end breadth_of_rectangle_l578_578154


namespace numberOfWaysToArrangeBANANA_l578_578738

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578738


namespace banana_arrangements_l578_578969

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578969


namespace banana_arrangements_l578_578985

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578985


namespace banana_arrangements_l578_578631

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578631


namespace number_of_ways_to_arrange_BANANA_l578_578808

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578808


namespace arrangement_count_BANANA_l578_578704

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578704


namespace permutations_BANANA_l578_578557

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578557


namespace exists_positive_integers_abc_l578_578097

theorem exists_positive_integers_abc (m n : ℕ) (hmn : Nat.Coprime m n) (hm_gt : m > 1) (hn_gt : n > 1) :
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ m^a = 1 + n^b * c ∧ Nat.coprime n c :=
by
  sorry

end exists_positive_integers_abc_l578_578097


namespace point_p_below_and_left_of_l2_l578_578238

theorem point_p_below_and_left_of_l2 (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) :
  let P1 := ((if a = 1 ∨ a = 2 ∨ a = 3 then 1 else 0) : ℚ) / 36,
      P2 := 1 - P1,
      P := ⟨P1, P2⟩ in
  ((P.1 + 2 * P.2 < 2) ∧ (P.1 = 1 / 12) ∧ (P.2 = 33 / 36)).

end point_p_below_and_left_of_l2_l578_578238


namespace numberOfWaysToArrangeBANANA_l578_578752

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578752


namespace number_of_arrangements_l578_578519

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578519


namespace part_a_prob_part_b_expected_time_l578_578214

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l578_578214


namespace cat_weights_ratio_l578_578172

variable (meg_cat_weight : ℕ) (anne_extra_weight : ℕ) (meg_cat_weight := 20) (anne_extra_weight := 8)

/-- The ratio of the weight of Meg's cat to the weight of Anne's cat -/
theorem cat_weights_ratio : (meg_cat_weight / Nat.gcd meg_cat_weight (meg_cat_weight + anne_extra_weight)) 
                            = 5 ∧ ((meg_cat_weight + anne_extra_weight) / Nat.gcd meg_cat_weight (meg_cat_weight + anne_extra_weight)) 
                            = 7 := by
  sorry

end cat_weights_ratio_l578_578172


namespace banana_arrangements_l578_578914

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578914


namespace number_of_arrangements_BANANA_l578_578799

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578799


namespace number_of_arrangements_BANANA_l578_578774

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578774


namespace banana_arrangements_l578_578611

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578611


namespace BANANA_arrangement_l578_578469

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578469


namespace distinct_permutations_BANANA_l578_578682

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578682


namespace arrangements_of_BANANA_l578_578432

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578432


namespace production_line_B_units_l578_578241

theorem production_line_B_units
  (total_units : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
  (h_total_units : total_units = 5000)
  (h_ratio : ratio_A = 1 ∧ ratio_B = 2 ∧ ratio_C = 2) :
  (2 * (total_units / (ratio_A + ratio_B + ratio_C))) = 2000 :=
by
  sorry

end production_line_B_units_l578_578241


namespace min_additional_squares_for_symmetry_l578_578228

def is_shaded (i j : ℕ) (shaded : List (ℕ × ℕ)) : Bool :=
  (i, j) ∈ shaded

-- Definition of 180-degree rotational symmetry for the grid.
def rot_sym (i j : ℕ) (n : ℕ) : ℕ × ℕ :=
  (n + 1 - i, n + 1 - j)

-- The grid dimension
def grid_size : ℕ := 6

-- The initially shaded squares
def initial_shaded_squares : List (ℕ × ℕ) := [(2, 5), (4, 2), (5, 6)]

-- The additional shaded squares needed for 180-degree rotational symmetry
def additional_shaded_squares : List (ℕ × ℕ) := [(5, 2), (3, 5), (2, 1)]

theorem min_additional_squares_for_symmetry :
  ∃ additional : List (ℕ × ℕ),
    initial_shaded_squares ++ additional = 
    List.map (λ (c : ℕ × ℕ), rot_sym c.1 c.2 grid_size) initial_shaded_squares ++ 
    additional ∧
    additional.length = 3 :=
  sorry

end min_additional_squares_for_symmetry_l578_578228


namespace simplify_expression_l578_578279

variable (a b c : ℝ)

theorem simplify_expression :
  (-32 * a^4 * b^5 * c) / ((-2 * a * b)^3) * (-3 / 4 * a * c) = -3 * a^2 * b^2 * c^2 :=
  by
    sorry

end simplify_expression_l578_578279


namespace permutations_BANANA_l578_578554

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578554


namespace banana_arrangements_l578_578894

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578894


namespace permutations_of_BANANA_l578_578386

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578386


namespace charity_activity_arrangements_l578_578233

theorem charity_activity_arrangements :
  let n := 6 in
  let arr1 := Nat.choose n 2 * Nat.perm 2 2 in
  let arr2 := Nat.choose n 3 in
  arr1 + arr2 = 50 :=
by
  sorry

end charity_activity_arrangements_l578_578233


namespace banana_unique_permutations_l578_578864

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578864


namespace binom_60_3_eq_34220_l578_578323

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l578_578323


namespace number_of_unique_permutations_BANANA_l578_578345

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578345


namespace C_and_D_complete_work_together_in_2_86_days_l578_578231

def work_rate (days : ℕ) : ℚ := 1 / days

def A_rate := work_rate 4
def B_rate := work_rate 10
def D_rate := work_rate 5

noncomputable def C_rate : ℚ :=
  let combined_A_B_C_rate := A_rate + B_rate + (1 / (2 : ℚ))
  let C_rate := 1 / (20 / 3 : ℚ)  -- Solved from the equations provided in the solution
  C_rate

noncomputable def combined_C_D_rate := C_rate + D_rate

noncomputable def days_for_C_and_D_to_complete_work : ℚ :=
  1 / combined_C_D_rate

theorem C_and_D_complete_work_together_in_2_86_days :
  abs (days_for_C_and_D_to_complete_work - 2.86) < 0.01 := sorry

end C_and_D_complete_work_together_in_2_86_days_l578_578231


namespace banana_unique_permutations_l578_578855

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578855


namespace number_of_unique_permutations_BANANA_l578_578369

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578369


namespace banana_unique_permutations_l578_578842

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578842


namespace ensure_A_win_product_l578_578229

theorem ensure_A_win_product {s : Finset ℕ} (h1 : s = {1, 2, 3, 4, 5, 6, 7, 8, 9}) (h2 : 8 ∈ s) (h3 : 5 ∈ s) :
  (4 ∈ s ∧ 6 ∈ s ∧ 7 ∈ s) →
  4 * 6 * 7 = 168 := 
by 
  intro _ 
  exact Nat.mul_assoc 4 6 7

end ensure_A_win_product_l578_578229


namespace permutations_BANANA_l578_578559

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578559


namespace permutations_BANANA_l578_578929

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578929


namespace matrix_product_eq_l578_578277

def matrixA : Matrix (Fin 2) (Fin 2) ℤ := ![
![(3 : ℤ), -1],
![(6 : ℤ), -4]
]

def matrixB : Matrix (Fin 2) (Fin 2) ℤ := ![
![(9 : ℤ), -3],
![(2 : ℤ), 2]
]

def matrixC : Matrix (Fin 2) (Fin 2) ℤ := ![
![(25 : ℤ), -11],
![(46 : ℤ), -26]
]

theorem matrix_product_eq : matrixA ⬝ matrixB = matrixC :=
by {
  sorry
}

end matrix_product_eq_l578_578277


namespace arrangements_of_BANANA_l578_578438

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578438


namespace cartesian_equation_of_curve1_polar_equation_of_curve2_distance_between_intersections_l578_578059

noncomputable def curve1_parametric (α : ℝ) : ℝ × ℝ :=
  (Real.sqrt 7 * Real.cos α, 2 + Real.sqrt 7 * Real.sin α)

noncomputable def curve2_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

theorem cartesian_equation_of_curve1 :
  ∀ x y, (∃ α, x = Real.sqrt 7 * Real.cos α ∧ y = 2 + Real.sqrt 7 * Real.sin α) ↔ (x^2 + (y - 2)^2 = 7) :=
by
  intros x y
  constructor
  { intro h
    cases h with α hα
    rw [←hα.left, ←hα.right]
    sorry
  }
  { intro h
    sorry
  }

theorem polar_equation_of_curve2 :
  ∀ ρ θ, curve2_cartesian (ρ * Real.cos θ) (ρ * Real.sin θ) ↔ (ρ = 2 * Real.cos θ) :=
by
  intros ρ θ
  constructor
  { intro h
    have h_eq : ((ρ * Real.cos θ - 1)^2 + (ρ * Real.sin θ)^2) = 1 := by { rw [curve2_cartesian, h] }
    sorry
  }
  { intro h
    rw h
    ring
    sorry
  }

theorem distance_between_intersections :
  let θ := Real.pi / 6 in
  let ρ1 := 3 in  -- Derived from C1
  let ρ2 := Real.sqrt 3 in  -- Derived from C2
  |ρ1 - ρ2| = 3 - Real.sqrt 3 :=
by
  intros _ _ _
  have hρ1 : ρ1 = 3 := rfl
  have hρ2 : ρ2 = Real.sqrt 3 := rfl
  rw [hρ1, hρ2]
  exact abs_of_nonneg (sub_nonneg.mpr (by norm_num))

end cartesian_equation_of_curve1_polar_equation_of_curve2_distance_between_intersections_l578_578059


namespace banana_unique_permutations_l578_578857

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578857


namespace number_of_arrangements_l578_578507

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578507


namespace banana_arrangements_l578_578957

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578957


namespace darryl_books_l578_578113

variable (l m d : ℕ)

theorem darryl_books (h1 : l + m + d = 97) (h2 : l = m - 3) (h3 : m = 2 * d) : d = 20 := 
by
  sorry

end darryl_books_l578_578113


namespace arrangements_of_BANANA_l578_578421

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578421


namespace number_of_unique_permutations_BANANA_l578_578337

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578337


namespace permutations_BANANA_l578_578529

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578529


namespace BANANA_arrangement_l578_578485

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578485


namespace find_angle_ABO_l578_578281

noncomputable def angle {α : Type} [MetricSpace α] (a b c : α) : Real := sorry

variables {α : Type} [MetricSpace α] 

-- Define points A, B, C, D, O
variables (A B C D O : α)

-- Define circles ω1 and ω2
variables (ω1 ω2 : set α)

-- Conditions
variables (h1 : ω1 ∩ ω2 = {A, B})
variables (h2 : ∀ (X ∈ ω1), dist O X < dist O C) 
variables (h3 : ∀ (Y ∈ ω2), dist O Y < dist O D)
variables (h4 : collinear ℝ {A, C, D})

-- Problem statement
theorem find_angle_ABO (h1 : ω1 ∩ ω2 = {A, B})
                       (h2 : ∀ (X ∈ ω1), dist O X < dist O C)
                       (h3 : ∀ (Y ∈ ω2), dist O Y < dist O D)
                       (h4 : collinear ℝ {A, C, D}) :
                       angle A B O = 90 :=
sorry

end find_angle_ABO_l578_578281


namespace permutations_of_BANANA_l578_578398

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578398


namespace banana_unique_permutations_l578_578861

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578861


namespace arrange_banana_l578_578571

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578571


namespace michael_balls_proof_l578_578176

-- Definitions for given conditions
def small_ball_rubber_bands : ℕ := 50
def medium_ball_rubber_bands : ℕ := 150
def large_ball_rubber_bands : ℕ := 300
def total_rubber_bands : ℕ := 10000
def initial_small_balls : ℕ := 35
def initial_medium_balls : ℕ := 15
def desired_remaining_bands : ℕ := 250

-- Statement to prove
theorem michael_balls_proof :
  ∃ remaining_bands small_balls medium_balls large_balls,
    remaining_bands = 250 ∧
    small_balls = 36 ∧
    medium_balls = 15 ∧
    large_balls = 19 ∧
    initial_small_balls * small_ball_rubber_bands +
    initial_medium_balls * medium_ball_rubber_bands +
    large_balls * large_ball_rubber_bands +
    (small_balls - initial_small_balls) * small_ball_rubber_bands = total_rubber_bands - remaining_bands :=
begin
  sorry
end

end michael_balls_proof_l578_578176


namespace same_money_after_days_l578_578030

theorem same_money_after_days 
  (kyuwon_initial: ℕ) (seokgi_initial: ℕ) 
  (kyuwon_daily: ℕ) (seokgi_daily: ℕ) 
  (d: ℕ) 
  (kyuwon_total: ℕ := kyuwon_initial + d * kyuwon_daily) 
  (seokgi_total: ℕ := seokgi_initial + d * seokgi_daily) :
  kyuwon_initial = 8000 → seokgi_initial = 5000 → 
  kyuwon_daily = 300 → seokgi_daily = 500 → 
  d = 15 → 
  kyuwon_total = seokgi_total :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  -- Now the goal is to prove: 8000 + 15 * 300 = 5000 + 15 * 500
  sorry

end same_money_after_days_l578_578030


namespace permutations_BANANA_l578_578933

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578933


namespace number_of_arrangements_BANANA_l578_578778

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578778


namespace number_of_arrangements_BANANA_l578_578787

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578787


namespace permutations_BANANA_l578_578534

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578534


namespace time_after_hours_l578_578146

theorem time_after_hours (h : 3) (n : 3333) : (h + n) % 12 = 12 := by
  have h := 3
  have n := 3333
  calc
    (h + n) % 12
      = (3 + 3333) % 12 : by rw [h, n]
      ... = 3336 % 12  : by norm_num
      ... = 0          : by norm_num
      ... = 12         : sorry

end time_after_hours_l578_578146


namespace convex_100gon_intersection_50_triangles_l578_578260

def is_convex_polygon {α : Type} [OrderedRing α] (polygon : List (Point α)) : Prop :=
  sorry -- Placeholder for actual implementation of convex polygon check

def represents_with_triangles {α : Type} [OrderedRing α] (polygon : List (Point α)) (n : ℕ) : Prop :=
  sorry -- Placeholder for actual implementation of polygon representation with n triangles

theorem convex_100gon_intersection_50_triangles :
  (∀ (polygon : List (Point ℝ)), is_convex_polygon polygon → polygon.length = 100 → 
    ∃ triangles : List (List (Point ℝ)), triangles.length = 50 ∧ represents_with_triangles polygon 50) :=
sorry

end convex_100gon_intersection_50_triangles_l578_578260


namespace banana_arrangements_l578_578902

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578902


namespace probability_businessmen_wait_two_minutes_l578_578220

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l578_578220


namespace arrangement_count_BANANA_l578_578722

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578722


namespace coefficient_x5_l578_578145

theorem coefficient_x5 : 
  (∑ k in Finset.range (5 + 1), (Polynomial.C (Nat.choose 5 k) * (Polynomial.X ^ 2 + Polynomial.X - 1) ^ (5 - k) * (-1 : ℤ) ^ k).coeff 5) = 11 :=
by
  sorry

end coefficient_x5_l578_578145


namespace permutations_BANANA_l578_578951

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578951


namespace arrangement_count_BANANA_l578_578690

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578690


namespace banana_arrangements_l578_578622

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578622


namespace number_of_arrangements_BANANA_l578_578771

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578771


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l578_578074

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l578_578074


namespace binomial_60_3_eq_34220_l578_578294

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l578_578294


namespace permutations_of_BANANA_l578_578373

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578373


namespace tina_total_time_l578_578175

-- Define constants for the problem conditions
def assignment_time : Nat := 20
def dinner_time : Nat := 17 * 60 + 30 -- 5:30 PM in minutes
def clean_time_per_key : Nat := 7
def total_keys : Nat := 30
def remaining_keys : Nat := total_keys - 1
def dry_time_per_key : Nat := 10
def break_time : Nat := 3
def keys_per_break : Nat := 5

-- Define a function to compute total cleaning time for remaining keys
def total_cleaning_time (keys : Nat) (clean_time : Nat) : Nat :=
  keys * clean_time

-- Define a function to compute total drying time for all keys
def total_drying_time (keys : Nat) (dry_time : Nat) : Nat :=
  keys * dry_time

-- Define a function to compute total break time
def total_break_time (keys : Nat) (keys_per_break : Nat) (break_time : Nat) : Nat :=
  (keys / keys_per_break) * break_time

-- Define a function to compute the total time including cleaning, drying, breaks, and assignment
def total_time (cleaning_time drying_time break_time assignment_time : Nat) : Nat :=
  cleaning_time + drying_time + break_time + assignment_time

-- The theorem to be proven
theorem tina_total_time : 
  total_time (total_cleaning_time remaining_keys clean_time_per_key) 
              (total_drying_time total_keys dry_time_per_key)
              (total_break_time total_keys keys_per_break break_time)
              assignment_time = 541 :=
by sorry

end tina_total_time_l578_578175


namespace numberOfWaysToArrangeBANANA_l578_578739

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578739


namespace arrange_banana_l578_578594

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578594


namespace arrange_banana_l578_578597

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578597


namespace bert_money_problem_l578_578202

-- Define the conditions as hypotheses
theorem bert_money_problem
  (n : ℝ)
  (h1 : n > 0)  -- Since he can't have negative or zero dollars initially
  (h2 : (1/2) * ((3/4) * n - 9) = 15) :
  n = 52 :=
sorry

end bert_money_problem_l578_578202


namespace permutations_BANANA_l578_578561

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578561


namespace BANANA_arrangement_l578_578461

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578461


namespace minimum_distance_between_P_and_Q_l578_578248

noncomputable def distance (P Q : ℝ × ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

-- Definitions of vertices
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def C : ℝ × ℝ × ℝ := (1/2, real.sqrt 3 / 2, 0)

-- Definitions of points P and Q
def P : ℝ × ℝ × ℝ := (3/4, 0, 0)
def Q : ℝ × ℝ × ℝ := (3/8, 3 * real.sqrt 3 / 8, 0)

-- The main proposition we want to prove
theorem minimum_distance_between_P_and_Q : distance P Q = 3 / 4 :=
by
  sorry

end minimum_distance_between_P_and_Q_l578_578248


namespace remaining_squares_count_total_area_removed_l578_578168

theorem remaining_squares_count (n : ℕ) : 
  let remaining_squares := 8^n 
  in (number_of_remaining_squares_after_n_steps n = remaining_squares) := sorry

theorem total_area_removed (n : ℕ) : 
  let removed_area := 1 - (8 / 9)^n 
  in (area_of_removed_squares_after_n_steps n = removed_area) := sorry

end remaining_squares_count_total_area_removed_l578_578168


namespace counseling_rooms_l578_578049

theorem counseling_rooms (n : ℕ) (x : ℕ)
  (h1 : n = 20 * x + 32)
  (h2 : n = 24 * (x - 1)) : x = 14 :=
by
  sorry

end counseling_rooms_l578_578049


namespace probability_businessmen_wait_two_minutes_l578_578216

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l578_578216


namespace portion_of_profit_divided_equally_l578_578116

-- Definitions for the given conditions
def total_investment_mary : ℝ := 600
def total_investment_mike : ℝ := 400
def total_profit : ℝ := 7500
def profit_diff : ℝ := 1000

-- Main statement
theorem portion_of_profit_divided_equally (E P : ℝ) 
  (h1 : total_profit = E + P)
  (h2 : E + (3/5) * P = E + (2/5) * P + profit_diff) :
  E = 2500 :=
by
  sorry

end portion_of_profit_divided_equally_l578_578116


namespace distinct_permutations_BANANA_l578_578652

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578652


namespace number_of_ways_to_arrange_BANANA_l578_578820

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578820


namespace count_eight_in_1_to_800_l578_578006

theorem count_eight_in_1_to_800 : 
  let count_digit (d n : Nat) : Nat := 
    (List.range (n + 1)).filter (λ x => List.contains (Nat.digits 10 x) d).length
  in count_digit 8 800 = 161 :=
by
  sorry

end count_eight_in_1_to_800_l578_578006


namespace approx_ants_in_field_l578_578246

noncomputable def ants_in_field (width_ft length_ft : ℕ) (ants_per_sq_inch : ℕ) : ℕ :=
  let width_inch := width_ft * 12
  let length_inch := length_ft * 12
  let area_sq_inch := width_inch * length_inch
  ants_per_sq_inch * area_sq_inch

theorem approx_ants_in_field :
  ants_in_field 300 400 3 ≈ 50000000 := by
  sorry

end approx_ants_in_field_l578_578246


namespace number_of_ways_to_arrange_BANANA_l578_578816

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578816


namespace banana_arrangements_l578_578634

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578634


namespace number_of_arrangements_l578_578520

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578520


namespace banana_arrangements_l578_578882

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578882


namespace false_statement_two_l578_578102

-- Define the necessary elements: Planes and Lines, and their relationships
variable {Plane : Type} {Line : Type}
variable α β γ : Plane
variable l : Line

-- Define necessary relations
class Perpendicular (P Q : Plane) : Prop := (perp : ∀ l : Line, l ∈ P → l ∈ Q → False)
class Intersection (P Q : Plane) (l : Line) : Prop := (intersec : l ∈ P ∧ l ∈ Q)
class Parallel (l : Line) (P : Plane) : Prop := (par : ∀ l' : Line, l ∈ P → l' ∈ P → l = l' ∨ ∀ p ∈ l, p ∉ l')

-- Statement expressing that statement (2) is false
theorem false_statement_two
  (h1 : ¬Perpendicular α β)
  (h2 : ∃ l : Line, Intersection α β l)
  (h3 : ∃ l : Line, Perpendicular l β) :
  False := sorry

end false_statement_two_l578_578102


namespace numberOfWaysToArrangeBANANA_l578_578758

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578758


namespace number_of_arrangements_l578_578523

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578523


namespace age_of_15th_student_l578_578039

-- Definitions and conditions
def total_students : ℕ := 15
def avg_age_total : ℕ := 15
def group1_students : ℕ := 5
def avg_age_group1 : ℕ := 13
def group2_students : ℕ := 6
def avg_age_group2 : ℕ := 15
def group3_students : ℕ := 3
def avg_age_group3 : ℕ := 17

theorem age_of_15th_student :
  let total_age_group1 := group1_students * avg_age_group1,
      total_age_group2 := group2_students * avg_age_group2,
      total_age_group3 := group3_students * avg_age_group3,
      total_age_14_students := total_age_group1 + total_age_group2 + total_age_group3,
      total_age_all_students := total_students * avg_age_total,
      age_15th_student := total_age_all_students - total_age_14_students
  in age_15th_student = 19 :=
by 
  sorry

end age_of_15th_student_l578_578039


namespace rhombus_area_l578_578148

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 12) : (d1 * d2) / 2 = 90 := by
  sorry

end rhombus_area_l578_578148


namespace compare_neg_fractions_l578_578284

theorem compare_neg_fractions : (- (2/3 : ℚ)) > - (3/4 : ℚ) := by
  sorry

end compare_neg_fractions_l578_578284


namespace largest_cube_side_length_largest_rect_parallelepiped_dimensions_l578_578079

variables {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (a): Side length of the largest cube
theorem largest_cube_side_length :
  let a₀ := a * b * c / (a * b + b * c + a * c) in
  ∃ a₀, a₀ = a * b * c / (a * b + b * c + a * c) :=
begin
  sorry
end

-- Part (b): Dimensions of the largest rectangular parallelepiped
theorem largest_rect_parallelepiped_dimensions :
  let x := a / 3, y := b / 3, z := c / 3 in
  ∃ x y z, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
begin
  sorry
end

end largest_cube_side_length_largest_rect_parallelepiped_dimensions_l578_578079


namespace permutations_BANANA_l578_578920

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578920


namespace banana_arrangements_l578_578908

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578908


namespace enclosed_by_eq_area_l578_578187

noncomputable def enclosed_area : ℝ :=
  let eq := λ (x y : ℝ), x^2 + y^2 = |x| + |y| + 1 in
  π * (√(3 / 2))^2 -- total circle area
  / 2 -- adjustment for quadrant coverage
  + 2 -- the area of triangles fixed by symmetry

theorem enclosed_by_eq_area :
  (enclosed_area = (3 / 2) * π + 2) :=
by
  sorry

end enclosed_by_eq_area_l578_578187


namespace largest_tile_size_l578_578236

def courtyard_length : ℤ := 378
def courtyard_width : ℤ := 595

theorem largest_tile_size : Nat.gcd courtyard_length courtyard_width = 7 := by
  sorry

end largest_tile_size_l578_578236


namespace permutations_of_BANANA_l578_578382

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578382


namespace permutations_of_BANANA_l578_578409

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578409


namespace arrangement_count_BANANA_l578_578703

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578703


namespace max_cube_side_length_max_parallelepiped_dimensions_l578_578065

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l578_578065


namespace permutations_BANANA_l578_578536

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578536


namespace number_of_ways_to_arrange_BANANA_l578_578817

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578817


namespace max_distinct_integers_16x16_table_l578_578035

noncomputable def max_distinct_integers (T : Matrix (Fin 16) (Fin 16) ℕ) : ℕ :=
  let row_distinct := ∀ i : Fin 16, (Finset.univ.map (Function.Embedding.subtype $ T i)).card ≤ 4
  let col_distinct := ∀ j : Fin 16, (Finset.univ.map (Function.Embedding.subtype $ T <$> j)).card ≤ 4
  if row_distinct ∧ col_distinct then (Finset.univ.bUnion (λ i => Finset.univ.image (λ j => T i j))).card else 0

theorem max_distinct_integers_16x16_table :
  ∀ T : Matrix (Fin 16) (Fin 16) ℕ, 
  (∀ i, (Finset.univ.image (λ j => T i j)).card ≤ 4) →
  (∀ j, (Finset.univ.image (λ i => T i j)).card ≤ 4) →
  max_distinct_integers T ≤ 49 :=
by sorry

end max_distinct_integers_16x16_table_l578_578035


namespace number_of_ways_to_arrange_BANANA_l578_578829

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578829


namespace arrange_banana_l578_578577

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578577


namespace number_of_ways_to_arrange_BANANA_l578_578804

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578804


namespace permutations_BANANA_l578_578947

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578947


namespace banana_arrangements_l578_578965

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578965


namespace probability_businessmen_wait_two_minutes_l578_578219

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l578_578219


namespace banana_arrangements_l578_578977

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578977


namespace MN_parallel_AD_l578_578266

noncomputable def midpoint (p1 p2 : Point) : Point := {
  x := (p1.x + p2.x) / 2,
  y := (p1.y + p2.y) / 2
}

theorem MN_parallel_AD (A B C D E M N: Point) (h_triangle : Triangle A B C)
  (h_angle_bisector : is_angle_bisector A D (Angle B A C))
  (h_AB_lt_AC : AB < AC)
  (h_CE_EQ_AB : CE = AB)
  (h_M_midpoint : M = midpoint B C)
  (h_N_midpoint : N = midpoint A E) :
  parallel (Line M N) (Line A D) := sorry

end MN_parallel_AD_l578_578266


namespace BANANA_arrangement_l578_578480

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578480


namespace number_of_unique_permutations_BANANA_l578_578359

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578359


namespace banana_arrangements_l578_578624

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578624


namespace number_of_unique_permutations_BANANA_l578_578344

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578344


namespace teddy_bears_ordered_l578_578136

theorem teddy_bears_ordered (days : ℕ) (T : ℕ)
  (h1 : 20 * days + 100 = T)
  (h2 : 23 * days - 20 = T) :
  T = 900 ∧ days = 40 := 
by 
  sorry

end teddy_bears_ordered_l578_578136


namespace product_of_invertible_functions_labels_l578_578062

theorem product_of_invertible_functions_labels :
  ∀ (f2 f3 f4 f5 : ℝ → ℝ),
    (∀ x y : ℝ, f2 x = -x^2 - 1 ∧ x ≠ y → f2 x ≠ f2 y) →
    (∀ x, x ∈ {-4, -2, 0, 2, 4} → f3 x = -f3 (-x)) →
    (∀ x, x ∈ [0, 2*π] → f4 x = sin x) →
    (∀ x, x ≠ 0 → f5 x = 5/x) →
  ((∃ g3 : ℝ → ℝ, ∀ x : ℝ, x ∈ {-4, -2, 0, 2, 4} → g3 (f3 x) = x) ∧
   (∃ g4 : ℝ → ℝ, ∀ y : ℝ, y ∈ [-1, 1] → g4 (sin y) = y) ∧
   (∃ g5 : ℝ → ℝ, ∀ x : ℝ, x ≠ 0 → g5 (5 / x) = x)) →
  3 * 4 * 5 = 60 :=
by
  intros f2 f3 f4 f5 h2 h3 h4 h5.
  sorry

end product_of_invertible_functions_labels_l578_578062


namespace binom_60_3_eq_34220_l578_578319

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l578_578319


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l578_578069

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l578_578069


namespace sample_mean_correct_probability_calc_overall_defect_rate_calc_defective_first_line_calc_l578_578119

noncomputable def sample_mean : ℝ :=
  (56 * 10 + 67 * 20 + 70 * 48 + 78 * 19 + 86 * 3) / 100

def variance := 36
def std_dev := Real.sqrt variance
def mu := sample_mean

noncomputable def normal_distribution_probability (a b μ σ : ℝ) : ℝ :=
  -- Here we calculate the probability for normal distribution
  sorry-- Implementation of the probability function.

def defect_rate_line1 := 0.015
def defect_rate_line2 := 0.018
def production_eff_line1 := 2 / 3
def production_eff_line2 := 1 / 3

def overall_defect_rate : ℝ :=
  production_eff_line1 * defect_rate_line1 + production_eff_line2 * defect_rate_line2

def defective_from_first_line (pa pb : ℝ) : ℝ :=
  (production_eff_line1 * defect_rate_line1) / pa
  
theorem sample_mean_correct : sample_mean = 70 := by
  sorry

theorem probability_calc : normal_distribution_probability 64 82 mu std_dev ≈ 0.8186 := by
  sorry

theorem overall_defect_rate_calc : overall_defect_rate = 0.016 := by
  sorry

theorem defective_first_line_calc : defective_from_first_line overall_defect_rate defect_rate_line1 = 5 / 8 := by
  sorry

end sample_mean_correct_probability_calc_overall_defect_rate_calc_defective_first_line_calc_l578_578119


namespace binomial_60_3_eq_34220_l578_578293

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l578_578293


namespace banana_arrangements_l578_578964

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578964


namespace percent_non_swimmers_play_basketball_eq_98_l578_578267

open Locale.Percent

-- Define the initial percentages for basketball and swimming players
def perc_basketball_players : ℝ := 70
def perc_swimmers : ℝ := 50
def perc_basketball_and_swim : ℝ := 30

-- Define the total number of children
variable (N : ℝ) (hN : 0 < N)

-- Define the number of children playing basketball
def num_basketball_players : ℝ := perc_basketball_players / 100 * N

-- Define the number of children swimming
def num_swimmers : ℝ := perc_swimmers / 100 * N

-- Define the number of basketball players who also swim
def num_basketball_and_swim : ℝ := perc_basketball_and_swim / 100 * num_basketball_players

-- Define the number of non-swimming basketball players
def num_non_swimming_basketball_players : ℝ := num_basketball_players - num_basketball_and_swim

-- Define the number of non-swimmers
def num_non_swimmers : ℝ := N - num_swimmers

-- Define the percentage of non-swimmers who play basketball
def perc_non_swimmers_play_basketball : ℝ := (num_non_swimming_basketball_players / num_non_swimmers) * 100

theorem percent_non_swimmers_play_basketball_eq_98 :
  perc_non_swimmers_play_basketball = 98 := by
  sorry

end percent_non_swimmers_play_basketball_eq_98_l578_578267


namespace permutations_BANANA_l578_578539

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578539


namespace number_of_arrangements_l578_578512

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578512


namespace permutations_BANANA_l578_578932

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578932


namespace banana_unique_permutations_l578_578863

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578863


namespace arrange_banana_l578_578581

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578581


namespace banana_arrangements_l578_578881

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578881


namespace banana_arrangements_l578_578620

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578620


namespace sum_of_two_digit_divisors_l578_578108

theorem sum_of_two_digit_divisors :
  let d_values := [12, 22, 33, 44, 66] in
  ∑ d in d_values, d = 177 :=
by
  -- Sum of the list [12, 22, 33, 44, 66] should be 177
  sorry

end sum_of_two_digit_divisors_l578_578108


namespace permutations_of_BANANA_l578_578375

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578375


namespace arrangements_of_BANANA_l578_578436

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578436


namespace banana_arrangements_l578_578889

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578889


namespace permutations_BANANA_l578_578918

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578918


namespace permutations_BANANA_l578_578939

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578939


namespace arrange_banana_l578_578569

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578569


namespace BANANA_arrangement_l578_578483

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578483


namespace permutations_BANANA_l578_578541

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578541


namespace banana_arrangements_l578_578905

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578905


namespace banana_arrangements_l578_578895

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578895


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l578_578222

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l578_578222


namespace banana_arrangements_l578_578988

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578988


namespace max_ones_in_chessboard_l578_578184

theorem max_ones_in_chessboard (rows : ℕ) (cols : ℕ) (f : ℕ → ℕ → ℕ) :
  rows = 40 → cols = 7 →
  (∀ i j, i ≠ j → f i cols ≠ f j cols) →
  ∑ i in finset.range rows, (nat.bitwise.band (f i cols) cols) = 198 :=
by
  intros
  sorry

end max_ones_in_chessboard_l578_578184


namespace arrange_banana_l578_578584

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578584


namespace binom_60_3_eq_34220_l578_578307

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l578_578307


namespace third_class_males_eq_nineteen_l578_578174

def first_class_males : ℕ := 17
def first_class_females : ℕ := 13
def second_class_males : ℕ := 14
def second_class_females : ℕ := 18
def third_class_females : ℕ := 17
def students_unable_to_partner : ℕ := 2
def total_males_from_first_two_classes : ℕ := first_class_males + second_class_males
def total_females_from_first_two_classes : ℕ := first_class_females + second_class_females
def total_females : ℕ := total_females_from_first_two_classes + third_class_females

theorem third_class_males_eq_nineteen (M : ℕ) : 
  total_males_from_first_two_classes + M - (total_females + students_unable_to_partner) = 0 → M = 19 :=
by
  sorry

end third_class_males_eq_nineteen_l578_578174


namespace banana_arrangements_l578_578907

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578907


namespace banana_arrangements_l578_578898

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578898


namespace arrange_banana_l578_578589

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578589


namespace arrangements_of_BANANA_l578_578420

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578420


namespace permutations_BANANA_l578_578942

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578942


namespace permutations_of_BANANA_l578_578372

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578372


namespace longest_tape_length_l578_578189

theorem longest_tape_length (a b c : ℕ) (h1 : a = 600) (h2 : b = 500) (h3 : c = 1200) : Nat.gcd (Nat.gcd a b) c = 100 :=
by
  sorry

end longest_tape_length_l578_578189


namespace banana_unique_permutations_l578_578851

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578851


namespace banana_arrangements_l578_578962

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578962


namespace compare_fractions_l578_578290

-- Define the fractions
def frac1 : ℚ := -2/3
def frac2 : ℚ := -3/4

-- Prove that -2/3 > -3/4
theorem compare_fractions : frac1 > frac2 :=
by {
  sorry
}

end compare_fractions_l578_578290


namespace BANANA_arrangement_l578_578474

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578474


namespace function_properties_l578_578253

-- Define the function
def f (x : ℝ) : ℝ := x / (1 + |x|)

-- Problem: Prove the correctness of conclusions 1, 2, and 3
theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧                             -- Conclusion 1
  (∀ y : ℝ, ∃ x : ℝ, f x = y ∧ y ∈ Ioo (-1 : ℝ) 1) ∧    -- Conclusion 2
  (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2)                  -- Conclusion 3
  :=
by
  sorry

end function_properties_l578_578253


namespace binom_60_3_eq_34220_l578_578303

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l578_578303


namespace BANANA_arrangement_l578_578460

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578460


namespace number_of_arrangements_BANANA_l578_578800

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578800


namespace two_digit_primes_ending_with_3_count_l578_578014

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_with_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → n % m ≠ 0

theorem two_digit_primes_ending_with_3_count :
  {n : ℕ | is_two_digit n ∧ ends_with_3 n ∧ is_prime n}.to_finset.card = 6 :=
by sorry

end two_digit_primes_ending_with_3_count_l578_578014


namespace probability_businessmen_wait_two_minutes_l578_578218

theorem probability_businessmen_wait_two_minutes :
  let total_suitcases := 200
  let business_suitcases := 10
  let time_to_wait_seconds := 120
  let suitcases_in_120_seconds := time_to_wait_seconds / 2
  let prob := (Nat.choose 59 9) / (Nat.choose total_suitcases business_suitcases)
  suitcases_in_120_seconds = 60 ->
  prob = (Nat.choose 59 9) / (Nat.choose 200 10) :=
by 
  sorry

end probability_businessmen_wait_two_minutes_l578_578218


namespace binomial_60_3_eq_34220_l578_578297

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l578_578297


namespace solve_quadratic_identity_l578_578159

theorem solve_quadratic_identity (y : ℝ) (h : 7 * y^2 + 2 = 5 * y + 13) :
  (14 * y - 5) ^ 2 = 333 :=
by sorry

end solve_quadratic_identity_l578_578159


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l578_578077

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l578_578077


namespace number_of_unique_permutations_BANANA_l578_578368

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578368


namespace arrangements_of_BANANA_l578_578415

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578415


namespace banana_unique_permutations_l578_578843

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578843


namespace permutations_BANANA_l578_578937

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578937


namespace number_of_ways_to_arrange_BANANA_l578_578834

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578834


namespace curvilinear_quadrilateral_area_l578_578251

-- Conditions: Define radius R, and plane angles of the tetrahedral angle.
noncomputable def radius (R : Real) : Prop :=
  R > 0

noncomputable def angle (theta : Real) : Prop :=
  theta = 60

-- Establishing the final goal based on the given conditions and solution's correct answer.
theorem curvilinear_quadrilateral_area
  (R : Real)     -- given radius of the sphere
  (hR : radius R) -- the radius of the sphere touching all edges
  (theta : Real)  -- given angle in degrees
  (hθ : angle theta) -- the plane angle of 60 degrees
  :
  ∃ A : Real, 
    A = π * R^2 * (16/3 * (Real.sqrt (2/3)) - 2) := 
  sorry

end curvilinear_quadrilateral_area_l578_578251


namespace berengere_contribution_l578_578272

noncomputable def euros_needed (P : ℝ) (L_CAD : ℝ) (R : ℝ) : ℝ :=
  P - (L_CAD / R)

theorem berengere_contribution : 
  euros_needed 8 10 1.5 = 1.33 :=
by
  rw [euros_needed, (8 : ℝ), (10 : ℝ), (1.5 : ℝ)]   -- Apply the definitions and given values
  -- Calculation steps are skipped
  sorry   -- Provide a placeholder to denote the proof is not completed

end berengere_contribution_l578_578272


namespace permutations_BANANA_l578_578946

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578946


namespace BANANA_arrangement_l578_578476

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578476


namespace permutations_of_BANANA_l578_578383

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578383


namespace compare_neg_fractions_l578_578283

theorem compare_neg_fractions : (- (2/3 : ℚ)) > - (3/4 : ℚ) := by
  sorry

end compare_neg_fractions_l578_578283


namespace minimum_loaves_arithmetic_sequence_l578_578140

theorem minimum_loaves_arithmetic_sequence :
  ∃ a d : ℚ, 
    (5 * a = 100) ∧ (3 * a + 3 * d = 7 * (2 * a - 3 * d)) ∧ (a - 2 * d = 5/3) :=
sorry

end minimum_loaves_arithmetic_sequence_l578_578140


namespace banana_unique_permutations_l578_578847

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578847


namespace number_of_arrangements_BANANA_l578_578792

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578792


namespace number_of_unique_permutations_BANANA_l578_578351

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578351


namespace permutations_of_BANANA_l578_578397

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578397


namespace finger_motion_due_to_friction_and_weight_distribution_l578_578139

theorem finger_motion_due_to_friction_and_weight_distribution
  (fingers_joined_at_midpoint : Prop)
  (fingers_move_apart_towards_ends : Prop)
  (one_finger_moves_faster : Prop)
  (slower_finger_supports_more_weight : Prop)
  (center_of_mass_balanced : Prop):
  (one_finger_moves_due_to_varying_friction_and_weight : Prop) :=
begin
  sorry
end

end finger_motion_due_to_friction_and_weight_distribution_l578_578139


namespace pairs_divisible_by_7_l578_578004

theorem pairs_divisible_by_7 :
  (∃ (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, (1 ≤ p.fst ∧ p.fst ≤ 1000) ∧ (1 ≤ p.snd ∧ p.snd ≤ 1000) ∧ (p.fst^2 + p.snd^2) % 7 = 0) ∧ 
    pairs.length = 20164) :=
sorry

end pairs_divisible_by_7_l578_578004


namespace permutations_BANANA_l578_578925

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578925


namespace first_term_exceeds_15000_l578_578250

theorem first_term_exceeds_15000 : 
  ∃ n : ℕ, ∀ (a : ℕ → ℕ), 
    (a 1 = 3) ∧
    (∀ n, n ≥ 2 → a (n + 1) = 3 * ∑ i in finset.range n, a (i + 1)) ∧
    (a n > 15000) → 
    n = 8 ∧ a 8 = 36864 :=
begin
  sorry
end

end first_term_exceeds_15000_l578_578250


namespace number_of_unique_permutations_BANANA_l578_578358

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578358


namespace number_of_unique_permutations_BANANA_l578_578335

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578335


namespace sum_segment_ratio_l578_578089

variable {A B C D M N P Q : Type}
variable [Triangle A B C]
variable [Height A D B C]
variable [CircleTangentBC A D B C]
variable [IntersectsCircleAB M N A D B]
variable [IntersectsCircleAC P Q A D C]

theorem sum_segment_ratio (A B C D M N P Q : ℝ)
    (h_height : height A D B C)
    (h_tangent : circle_tangent BC A D)
    (h_intersectAB : intersects_circle AB M N A D B)
    (h_intersectAC : intersects_circle AC P Q A D C) :
    (AM + AN) / AC = (AP + AQ) / AB := 
  sorry

end sum_segment_ratio_l578_578089


namespace arrangements_of_BANANA_l578_578435

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578435


namespace number_of_arrangements_BANANA_l578_578767

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578767


namespace arrangements_of_BANANA_l578_578437

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578437


namespace arrange_banana_l578_578590

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578590


namespace compare_fractions_l578_578289

-- Define the fractions
def frac1 : ℚ := -2/3
def frac2 : ℚ := -3/4

-- Prove that -2/3 > -3/4
theorem compare_fractions : frac1 > frac2 :=
by {
  sorry
}

end compare_fractions_l578_578289


namespace banana_arrangements_l578_578615

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578615


namespace arrange_banana_l578_578602

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578602


namespace number_of_arrangements_BANANA_l578_578797

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578797


namespace BANANA_arrangement_l578_578458

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578458


namespace number_of_unique_permutations_BANANA_l578_578336

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578336


namespace permutations_BANANA_l578_578528

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578528


namespace stratified_sampling_from_C_l578_578034

noncomputable def number_of_teachers_selected_from_C (teachers_A : ℕ) (teachers_B : ℕ) (teachers_C : ℕ) (total_selected : ℕ) : ℕ :=
  (teachers_C * total_selected) / (teachers_A + teachers_B + teachers_C)

theorem stratified_sampling_from_C :
  ∀ (teachers_A teachers_B teachers_C total_selected : ℕ), 
  teachers_A = 180 → 
  teachers_B = 270 → 
  teachers_C = 90 → 
  total_selected = 60 → 
  number_of_teachers_selected_from_C teachers_A teachers_B teachers_C total_selected = 10 :=
by
  intros teachers_A teachers_B teachers_C total_selected hA hB hC hTotal
  unfold number_of_teachers_selected_from_C
  rw [hA, hB, hC, hTotal]
  sorry

end stratified_sampling_from_C_l578_578034


namespace max_volume_cube_max_volume_parallelepiped_l578_578083

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l578_578083


namespace banana_arrangements_l578_578643

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578643


namespace banana_arrangements_l578_578993

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578993


namespace number_of_ways_to_arrange_BANANA_l578_578830

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578830


namespace arrangement_count_BANANA_l578_578691

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578691


namespace arrangements_of_BANANA_l578_578445

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578445


namespace max_volume_cube_max_volume_parallelepiped_l578_578085

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l578_578085


namespace compare_fractions_neg_l578_578286

theorem compare_fractions_neg : (- (2 / 3) > - (3 / 4)) :=
  sorry

end compare_fractions_neg_l578_578286


namespace petes_average_speed_is_correct_l578_578271

-- Definition of the necessary constants
def map_distance := 5.0 -- inches
def scale := 0.023809523809523808 -- inches per mile
def travel_time := 3.5 -- hours

-- The real distance calculation based on the given map scale
def real_distance := map_distance / scale -- miles

-- Proving the average speed calculation
def average_speed := real_distance / travel_time -- miles per hour

-- Theorem statement: Pete's average speed calculation is correct
theorem petes_average_speed_is_correct : average_speed = 60 :=
by
  -- Proof outline
  -- The real distance is 5 / 0.023809523809523808 ≈ 210
  -- The average speed is 210 / 3.5 ≈ 60
  sorry

end petes_average_speed_is_correct_l578_578271


namespace part_a_prob_part_b_expected_time_l578_578211

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l578_578211


namespace arrangements_of_BANANA_l578_578448

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578448


namespace number_of_unique_permutations_BANANA_l578_578338

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578338


namespace remaining_squares_count_total_area_removed_l578_578169

theorem remaining_squares_count (n : ℕ) : 
  let remaining_squares := 8^n 
  in (number_of_remaining_squares_after_n_steps n = remaining_squares) := sorry

theorem total_area_removed (n : ℕ) : 
  let removed_area := 1 - (8 / 9)^n 
  in (area_of_removed_squares_after_n_steps n = removed_area) := sorry

end remaining_squares_count_total_area_removed_l578_578169


namespace number_of_arrangements_l578_578501

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578501


namespace number_of_ways_to_arrange_BANANA_l578_578825

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578825


namespace banana_arrangements_l578_578917

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578917


namespace banana_unique_permutations_l578_578874

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578874


namespace lowest_fraction_by_two_people_l578_578203

theorem lowest_fraction_by_two_people 
  (rate_A rate_B rate_C : ℚ)
  (hA : rate_A = 1 / 4) 
  (hB : rate_B = 1 / 6) 
  (hC : rate_C = 1 / 8) : 
  ∃ (r : ℚ), r = 7 / 24 ∧ 
    ∀ (r1 r2 : ℚ), (r1 = rate_A ∧ r2 = rate_B ∨ r1 = rate_A ∧ r2 = rate_C ∨ r1 = rate_B ∧ r2 = rate_C) → 
      r ≤ r1 + r2 := 
sorry

end lowest_fraction_by_two_people_l578_578203


namespace number_of_arrangements_BANANA_l578_578764

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578764


namespace banana_arrangements_l578_578989

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578989


namespace number_of_ways_to_arrange_BANANA_l578_578832

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578832


namespace infinite_common_prime_factor_l578_578129

theorem infinite_common_prime_factor (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) :
  ∃ p : ℕ, prime p ∧ ∀ n : ℕ, ∃ m : ℕ, p ∣ (a * m + b) :=
sorry

end infinite_common_prime_factor_l578_578129


namespace length_of_EF_l578_578054

variables {Point : Type} [metric_space Point]
variables (A B C D E F : Point)

-- Conditions
variables (AD BC AB DC : ℝ)
variables (DE DF : ℝ)

-- Geometry conditions
variables (is_isosceles_trapezoid : AD = 7 ∧ BC = 7 ∧ AB = 6 ∧ DC = 12)
variables (is_isosceles_triangle_DEF : DE = DF)
variables (on_line_DC : ∀ {X}, (X = E ∨ X = F) → ∃ t : ℝ, DC = t • X)
variables (B_midpoint_DE : B = midpoint DE)
variables (C_midpoint_DF : C = midpoint DF)

-- Proof goal
theorem length_of_EF : dist E F = 6 :=
by
  have h1 : AD = 7 := is_isosceles_trapezoid.1
  have h2 : BC = 7 := is_isosceles_trapezoid.2.1
  have h3 : AB = 6 := is_isosceles_trapezoid.2.2.1
  have h4 : DC = 12 := is_isosceles_trapezoid.2.2.2
  have h5 : DE = DF := is_isosceles_triangle_DEF
  sorry

end length_of_EF_l578_578054


namespace permutations_BANANA_l578_578927

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578927


namespace arrangement_count_BANANA_l578_578701

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578701


namespace arrangements_of_BANANA_l578_578412

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578412


namespace permutations_of_BANANA_l578_578392

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578392


namespace banana_arrangements_l578_578899

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578899


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l578_578223

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l578_578223


namespace banana_arrangements_l578_578630

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578630


namespace arrange_banana_l578_578568

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578568


namespace permutations_of_BANANA_l578_578393

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578393


namespace permutations_BANANA_l578_578919

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578919


namespace number_of_valid_pairs_l578_578107

theorem number_of_valid_pairs : ∃ p : Finset (ℕ × ℕ), 
  (∀ (a b : ℕ), (a, b) ∈ p ↔ a ≤ 10 ∧ b ≤ 10 ∧ 3 * b < a ∧ a < 4 * b) ∧ p.card = 2 :=
by
  sorry

end number_of_valid_pairs_l578_578107


namespace banana_arrangements_l578_578636

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578636


namespace number_of_unique_permutations_BANANA_l578_578342

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578342


namespace distinct_permutations_BANANA_l578_578663

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578663


namespace arrangement_count_BANANA_l578_578693

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578693


namespace number_of_ways_to_arrange_BANANA_l578_578806

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578806


namespace permutations_of_BANANA_l578_578403

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578403


namespace expression_evaluation_l578_578276

theorem expression_evaluation :
  let a := (0.0081)^(1/4)
  let b := (4^(-3/4))^2
  let c := (sqrt 8)^(-4/3)
  let d := 16^(-0.75)
  let e := 2^(log 2 5)
  a + b + c - d + e = 5.55 := by
  have h1 : (0.0081)^(1/4) = 0.3 := sorry
  have h2 : (4^(-3/4))^2 = 1/8 := sorry
  have h3 : (sqrt 8)^(-4/3) = 1/4 := sorry
  have h4 : 16^(-0.75) = 1/8 := sorry
  have h5 : 2^(log 2 5) = 5 := sorry
  calc
    a + b + c - d + e = 0.3 + 1/8 + 1/4 - 1/8 + 5 : by rw [h1, h2, h3, h4, h5]
                   ... = 5.55 : by norm_num

end expression_evaluation_l578_578276


namespace distinct_permutations_BANANA_l578_578648

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578648


namespace number_of_arrangements_l578_578522

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578522


namespace max_cube_side_length_max_parallelepiped_dimensions_l578_578066

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l578_578066


namespace numberOfWaysToArrangeBANANA_l578_578759

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578759


namespace permutations_BANANA_l578_578950

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578950


namespace number_of_arrangements_l578_578490

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578490


namespace number_of_arrangements_l578_578500

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578500


namespace number_of_unique_permutations_BANANA_l578_578363

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578363


namespace banana_unique_permutations_l578_578844

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578844


namespace arrangements_of_BANANA_l578_578429

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578429


namespace arrangement_count_BANANA_l578_578695

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578695


namespace banana_arrangements_l578_578644

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578644


namespace number_of_arrangements_l578_578491

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578491


namespace banana_arrangements_l578_578966

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578966


namespace max_cube_side_length_max_rect_parallelepiped_dimensions_l578_578076

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end max_cube_side_length_max_rect_parallelepiped_dimensions_l578_578076


namespace max_cube_side_length_max_parallelepiped_dimensions_l578_578067

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l578_578067


namespace arrangements_of_BANANA_l578_578426

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578426


namespace cos_C_eq_sqrt_6_div_3_l578_578031

noncomputable theory

open Real

-- Define the problem conditions
variables (A B C : Type) 
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] 

def triangle_ABC := (AB = 2) ∧ (AC = 3) ∧ (angle_B = 60)

-- Define the proof goal
theorem cos_C_eq_sqrt_6_div_3 (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (h : triangle_ABC A B C) : cos C = (sqrt 6) / 3 :=
  sorry

end cos_C_eq_sqrt_6_div_3_l578_578031


namespace common_chord_length_proof_l578_578155

-- Define the first circle equation
def first_circle (x y : ℝ) : Prop := x^2 + y^2 = 50

-- Define the second circle equation
def second_circle (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 6*y + 40 = 0

-- Define the property that the length of the common chord is equal to 2 * sqrt(5)
noncomputable def common_chord_length : ℝ := 2 * Real.sqrt 5

-- The theorem statement
theorem common_chord_length_proof :
  ∀ x y : ℝ, first_circle x y → second_circle x y → common_chord_length = 2 * Real.sqrt 5 :=
by
  intros x y h1 h2
  sorry

end common_chord_length_proof_l578_578155


namespace arrangement_count_BANANA_l578_578700

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578700


namespace three_digit_numbers_with_2_left_of_3_l578_578264

theorem three_digit_numbers_with_2_left_of_3 : 
  ∃ n : ℕ, 
    n = 23 ∧ 
    ∀ d1 d2 d3 : ℕ, 
      (d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3) ∧ 
      (d1 ≠ 0 ∨ d2 ≠ 0 ∨ d3 ≠ 0) ∧ 
      (d1 = 2 → d2 = 3 ∨ d3 = 3 → d1 = 2 → d2 ≠ 2) ↔ n = 23 :=
begin
  sorry
end

end three_digit_numbers_with_2_left_of_3_l578_578264


namespace average_minutes_run_l578_578045

-- Definitions
def third_graders (fi : ℕ) : ℕ := 6 * fi
def fourth_graders (fi : ℕ) : ℕ := 2 * fi
def fifth_graders (fi : ℕ) : ℕ := fi

-- Number of minutes run by each grade
def third_graders_minutes : ℕ := 10
def fourth_graders_minutes : ℕ := 18
def fifth_graders_minutes : ℕ := 8

-- Main theorem
theorem average_minutes_run 
  (fi : ℕ) 
  (t := third_graders fi) 
  (fr := fourth_graders fi) 
  (f := fifth_graders fi) 
  (minutes_total := 10 * t + 18 * fr + 8 * f) 
  (students_total := t + fr + f) :
  (students_total > 0) →
  (minutes_total : ℚ) / students_total = 104 / 9 :=
by
  sorry

end average_minutes_run_l578_578045


namespace number_of_arrangements_BANANA_l578_578789

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578789


namespace chess_team_arrangements_l578_578143

theorem chess_team_arrangements :
  (∃ boys girls : List α, boys.length = 3 ∧ girls.length = 3 ∧
   ∀ arrangement : List α, arrangement.length = 6 ∧
   (arrangement.head = boys.head ∨ arrangement.head = girls.head) ∧
   (∀ (i : ℕ), i < 6 → arrangement.get i = boys.get (i / 2) ∨
                      arrangement.get i = girls.get (i / 2) →
   [Arrangement with alternating genders starting with either boy or girl])) →
  ∃ (num_arrangements : ℕ), num_arrangements = 72 := by
  sorry

end chess_team_arrangements_l578_578143


namespace number_of_arrangements_BANANA_l578_578775

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578775


namespace distinct_permutations_BANANA_l578_578649

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578649


namespace max_cube_side_length_max_parallelepiped_dimensions_l578_578063

theorem max_cube_side_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (a0 : ℝ), a0 = a * b * c / (a * b + b * c + a * c) := 
sorry

theorem max_parallelepiped_dimensions (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
    ∃ (x y z : ℝ), (x = a / 3) ∧ (y = b / 3) ∧ (z = c / 3) :=
sorry

end max_cube_side_length_max_parallelepiped_dimensions_l578_578063


namespace number_of_outfits_l578_578200

-- Define the conditions
def s := 5
def p := 6
def h := 3

-- State the theorem
theorem number_of_outfits : s * p * h = 90 :=
by
  let s := 5
  let p := 6
  let h := 3
  have hs : s = 5 := rfl
  have hp : p = 6 := rfl
  have hh : h = 3 := rfl
  show s * p * h = 90,
  calc s * p * h
      = 5 * 6 * 3 : by rw [hs, hp, hh]
  ... = 90 : by norm_num

end number_of_outfits_l578_578200


namespace remaining_squares_total_area_removed_l578_578166

-- Define the initial conditions
def initial_square_area : ℝ := 1
def side_length (n : ℕ) : ℝ := 1 / 3^n

-- Prove the number of remaining squares with side length 1/3^n after n steps is 8^n
theorem remaining_squares (n : ℕ) : (8 : ℝ)^n = 8^n :=
  by sorry

-- Prove the total area of the removed squares after n steps is 1 - (8/9)^n
theorem total_area_removed (n : ℕ) : (1 : ℝ) - (8 / 9)^n = 1 - (8 / 9)^n :=
  by sorry

end remaining_squares_total_area_removed_l578_578166


namespace part_a_prob_part_b_expected_time_l578_578212

/--  
Let total_suitcases be 200.
Let business_suitcases be 10.
Let total_wait_time be 120.
Let arrival_interval be 2.
--/

def total_suitcases : ℕ := 200
def business_suitcases : ℕ := 10
def total_wait_time : ℕ := 120 
def arrival_interval : ℕ := 2

def two_minutes_suitcases : ℕ := total_wait_time / arrival_interval
def prob_last_suitcase_at_n_minutes (n : ℕ) : ℚ := 
  (Nat.choose (two_minutes_suitcases - 1) (business_suitcases - 1) : ℚ) / 
  (Nat.choose total_suitcases business_suitcases : ℚ)

theorem part_a_prob : 
  prob_last_suitcase_at_n_minutes 2 = 
  (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) := sorry

noncomputable def expected_position_last_suitcase (total_pos : ℕ) (suitcases_per_group : ℕ) : ℚ :=
  (total_pos * business_suitcases : ℚ) / (business_suitcases + 1)

theorem part_b_expected_time : 
  expected_position_last_suitcase 201 11 * arrival_interval = 
  (4020 : ℚ) / 11 := sorry

end part_a_prob_part_b_expected_time_l578_578212


namespace banana_arrangements_l578_578880

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578880


namespace distinct_permutations_BANANA_l578_578674

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578674


namespace total_juice_v_is_25_l578_578226

variable (p_total v_total : ℚ)
variable (p_m v_m p_y v_y : ℚ)

-- Conditions of the problem
#eval let condition1 : p_total = 24
condition1 

#eval let condition2 : p_m / v_m = 4
condition2

#eval let condition3 : p_y / v_y = 1 / 5
condition3

#eval let condition4 : p_m = 20
condition4

-- Definition of total amount of juice v
def total_juice_v (p_total v_total p_m v_m p_y v_y : ℚ) := v_m + v_y

-- The main theorem to prove
theorem total_juice_v_is_25 :
  condition1 → condition2 → condition3 → condition4 → total_juice_v p_total v_total p_m v_m p_y v_y = 25 :=
by
  intro h1 h2 h3 h4
  -- It remains to prove the theorem
  sorry

end total_juice_v_is_25_l578_578226


namespace total_distance_from_points_l578_578193

theorem total_distance_from_points :
  let p1 := (2 : ℝ, -3 : ℝ)
  let p2 := (8 : ℝ, 9 : ℝ)
  let p3 := (3 : ℝ, 2 : ℝ)
  let distance (a b : ℝ × ℝ) := Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)
  distance p1 p2 + distance p2 p3 = 6 * Real.sqrt 5 + Real.sqrt 74 :=
by
  sorry

end total_distance_from_points_l578_578193


namespace banana_arrangements_l578_578886

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578886


namespace banana_arrangements_l578_578640

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578640


namespace distinct_permutations_BANANA_l578_578680

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578680


namespace arrangements_of_BANANA_l578_578416

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578416


namespace numberOfWaysToArrangeBANANA_l578_578734

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578734


namespace number_of_unique_permutations_BANANA_l578_578371

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578371


namespace louisa_first_day_distance_l578_578122

theorem louisa_first_day_distance :
  (∀ t2 t1 : ℕ, t2 = 280 / 40 ∧ t1 = t2 - 3 → 40 * t1 = 160) :=
begin
  intros t2 t1 h,
  cases h with ht2 ht1,
  have ht2' : t2 = 7, by norm_num,
  rw ht2'.symm at ht2,
  norm_num at ht2,
  rw ht2.symm at ht1,
  norm_num at ht1,
  simp [ht1],
end

end louisa_first_day_distance_l578_578122


namespace banana_arrangements_l578_578616

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578616


namespace BANANA_arrangement_l578_578454

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578454


namespace distinct_permutations_BANANA_l578_578660

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578660


namespace numberOfWaysToArrangeBANANA_l578_578726

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578726


namespace banana_arrangements_l578_578995

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578995


namespace number_of_unique_permutations_BANANA_l578_578354

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578354


namespace exists_good_pair_l578_578112

theorem exists_good_pair (m : ℕ) : ∃ n : ℕ, n > m ∧ (∃ k1 k2 : ℕ, (m * n = k1^2) ∧ ((m + 1) * (n + 1) = k2^2)) :=
by
  let n := m * (4 * m + 3)^2
  have h1 : n > m := sorry
  have h2 : ∃ k1 : ℕ, m * n = k1^2 := sorry
  have h3 : ∃ k2 : ℕ, (m + 1) * (n + 1) = k2^2 := sorry
  exact ⟨n, h1, ⟨k1, k2, h2, h3⟩⟩


end exists_good_pair_l578_578112


namespace distinct_permutations_BANANA_l578_578667

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578667


namespace school_students_l578_578268

theorem school_students
  (total_students : ℕ)
  (students_in_both : ℕ)
  (students_chemistry : ℕ)
  (students_biology : ℕ)
  (students_only_chemistry : ℕ)
  (students_only_biology : ℕ)
  (h1 : total_students = students_only_chemistry + students_only_biology + students_in_both)
  (h2 : students_chemistry = 3 * students_biology)
  (students_in_both_eq : students_in_both = 5)
  (total_students_eq : total_students = 43) :
  students_only_chemistry + students_in_both = 36 :=
by
  sorry

end school_students_l578_578268


namespace smallest_k_for_partitioned_set_l578_578098

theorem smallest_k_for_partitioned_set (k : ℕ) (h_k : k ≥ 1) :
  ∃ (A : Finset ℕ) (s1 s2 : Finset ℕ),
  (card A = 4 * k) ∧
  (A = s1 ∪ s2) ∧
  (s1 ∩ s2 = ∅) ∧
  (card s1 = card s2) ∧
  (s1.sum id = s2.sum id) ∧
  (s1.sum (λ x, x^2) = s2.sum (λ x, x^2)) ∧
  (s1.sum (λ x, x^3) = s2.sum (λ x, x^3)) ∧
  k = 4 :=
begin
  sorry
end

end smallest_k_for_partitioned_set_l578_578098


namespace compare_fractions_neg_l578_578285

theorem compare_fractions_neg : (- (2 / 3) > - (3 / 4)) :=
  sorry

end compare_fractions_neg_l578_578285


namespace BANANA_arrangement_l578_578488

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578488


namespace BANANA_arrangement_l578_578452

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578452


namespace permutations_BANANA_l578_578924

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578924


namespace number_of_ways_to_arrange_BANANA_l578_578827

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578827


namespace binom_60_3_eq_34220_l578_578316

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l578_578316


namespace cube_root_simplification_l578_578195

theorem cube_root_simplification (c d : ℕ) (h1 : c = 3) (h2 : d = 100) : c + d = 103 :=
by
  sorry

end cube_root_simplification_l578_578195


namespace banana_arrangements_l578_578606

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578606


namespace BANANA_arrangement_l578_578472

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578472


namespace largest_cube_side_length_largest_rect_parallelepiped_dimensions_l578_578081

variables {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Part (a): Side length of the largest cube
theorem largest_cube_side_length :
  let a₀ := a * b * c / (a * b + b * c + a * c) in
  ∃ a₀, a₀ = a * b * c / (a * b + b * c + a * c) :=
begin
  sorry
end

-- Part (b): Dimensions of the largest rectangular parallelepiped
theorem largest_rect_parallelepiped_dimensions :
  let x := a / 3, y := b / 3, z := c / 3 in
  ∃ x y z, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
begin
  sorry
end

end largest_cube_side_length_largest_rect_parallelepiped_dimensions_l578_578081


namespace banana_arrangements_l578_578983

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578983


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l578_578071

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l578_578071


namespace permutations_of_BANANA_l578_578407

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578407


namespace banana_arrangements_l578_578614

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578614


namespace find_x_l578_578096

theorem find_x (x : ℝ)
    (h : 2^x = (∏ k in (finset.range 4500).filter (λ k, k ≠ 0), 1 + real.tan (k * real.pi / 180 / 100)))
    : x = 2249.5 := 
  by sorry

end find_x_l578_578096


namespace numberOfWaysToArrangeBANANA_l578_578736

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578736


namespace permutations_of_BANANA_l578_578402

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578402


namespace sum_double_factorial_eq_l578_578329

-- Definition of double factorial for odd and even cases
def odd_factorial (n : ℕ) : ℕ :=
  if n % 2 = 1 then (List.prod (List.range' 1 ((n + 1) / 2)).map(λ x => x * 2 - 1)) else 1

def even_factorial (n : ℕ) : ℕ :=
  if n % 2 = 0 then (List.prod (List.range' 1 ((n / 2) + 1)).map(λ x => x * 2)) else 1

def double_factorial (n : ℕ) : ℕ :=
  if n % 2 = 1 then odd_factorial n else even_factorial n

-- Definition of each term in the sum
def term (i : ℕ) : ℚ :=
  (Nat.choose (2 * i) i : ℚ) / (2 ^ (2 * i))

-- Sum of the terms for i from 1 to 5
def sum_terms : ℚ :=
  (List.range' 1 5).sumBy term

-- The theorem to be proved
theorem sum_double_factorial_eq : sum_terms = 437 / 256 :=
  by sorry

end sum_double_factorial_eq_l578_578329


namespace permutations_of_BANANA_l578_578395

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578395


namespace BANANA_arrangement_l578_578486

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578486


namespace integral_approximation_l578_578278

theorem integral_approximation :
  | ∫ x in 0..1, (exp (-x^2) - 1) / x - 0.659 | ≤ 10^(-3) :=
by
  sorry

end integral_approximation_l578_578278


namespace total_women_attendees_l578_578262

theorem total_women_attendees 
  (adults : ℕ) (adult_women : ℕ) (student_offset : ℕ) (total_students : ℕ)
  (male_students : ℕ) :
  adults = 1518 →
  adult_women = 536 →
  student_offset = 525 →
  total_students = adults + student_offset →
  total_students = 2043 →
  male_students = 1257 →
  (adult_women + (total_students - male_students) = 1322) :=
by
  sorry

end total_women_attendees_l578_578262


namespace solve_inequality_l578_578137

theorem solve_inequality (x : ℝ) : 3 - 2 / (3 * x + 4) ≤ 5 ↔ x ∈ set.Ioo (-∞ : ℝ) (-4 / 3) ∪ set.Ioo (-5 / 3) (∞ : ℝ) := 
sorry

end solve_inequality_l578_578137


namespace arc_length_of_sector_l578_578144

theorem arc_length_of_sector (r θ : ℝ) (A : ℝ) (h₁ : r = 4)
  (h₂ : A = 7) : (1 / 2) * r^2 * θ = A → r * θ = 3.5 :=
by
  sorry

end arc_length_of_sector_l578_578144


namespace triangulation_triangle_count_l578_578207

theorem triangulation_triangle_count :
  ∀ (n m : ℕ), n = 1000 → m = 500 →
  (∀ (x y z : ℝ × ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z) → 
  let V := n + m in
  (∃ T : ℕ, T = 2 * V - 2) :=
begin
  sorry
end

end triangulation_triangle_count_l578_578207


namespace number_of_ways_to_arrange_BANANA_l578_578837

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578837


namespace banana_unique_permutations_l578_578853

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578853


namespace color_numbers_proper_divisors_l578_578015

-- Definitions based on the conditions
def color : Type := {c : ℕ // c < 4}
def pool : list ℕ := [2, 3, 4, 5, 6, 7, 8, 9, 10]
def proper_divisors : ℕ → list ℕ
| 2 := []
| 3 := []
| 4 := [2]
| 5 := []
| 6 := [2, 3]
| 7 := []
| 8 := [2, 4]
| 9 := [3]
| 10 := [2, 5]
| _ := []

-- Define the problem statement with the correct answer
theorem color_numbers_proper_divisors : 
  ∃ (f : ℕ → color), 
    (∀ n ∈ pool, ∀ d ∈ proper_divisors n, f n ≠ f d) ∧
    ∏ n in (pool.to_finset), 4 - proper_divisors n.length = 294912 := 
sorry

end color_numbers_proper_divisors_l578_578015


namespace permutations_BANANA_l578_578563

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578563


namespace arrangement_count_BANANA_l578_578697

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578697


namespace angle_bisector_of_C_l578_578123

-- Define the main objects used in the conditions
variables {A B C M N : Type} [triangle ABC : Triangle A B C]

-- Hypotheses as per the problem statement
variables (h1 : OnSegment A B M) (h2 : OnExtensionsSide A B N)
variables (h3 : (AM:Length) (MB:Length) (AN:Length) (NB:Length))
variables (prop : AM / MB = AN / NB)

-- Angle M C N is a right angle
variables (angleMCN : IsRightAngle (∠ M C N))

-- Proof that C M is the angle bisector of ∠ ACB
theorem angle_bisector_of_C 
  {CM_bisector : IsAngleBisector A C B M} :
  CM_bisector := by 
  sorry

end angle_bisector_of_C_l578_578123


namespace number_of_poles_needed_l578_578201

theorem number_of_poles_needed :
  ∀ (L W D : ℕ), L = 60 → W = 50 → D = 5 → (2 * (L + W)) / D = 44 :=
by 
  intros L W D hL hW hD
  rw [hL, hW, hD]
  unfold Nat.div
  dsimp
  sorry

end number_of_poles_needed_l578_578201


namespace number_of_arrangements_l578_578492

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578492


namespace banana_arrangements_l578_578986

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578986


namespace number_of_arrangements_BANANA_l578_578786

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578786


namespace banana_arrangements_l578_578975

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578975


namespace numberOfWaysToArrangeBANANA_l578_578761

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578761


namespace fifth_term_power_of_five_sequence_l578_578324

theorem fifth_term_power_of_five_sequence : 5^0 + 5^1 + 5^2 + 5^3 + 5^4 = 781 := 
by
sorry

end fifth_term_power_of_five_sequence_l578_578324


namespace banana_unique_permutations_l578_578867

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578867


namespace number_of_arrangements_BANANA_l578_578762

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578762


namespace permutations_of_BANANA_l578_578379

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578379


namespace banana_arrangements_l578_578968

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l578_578968


namespace number_of_unique_permutations_BANANA_l578_578355

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578355


namespace diagonals_from_vertex_l578_578024

-- Define the given conditions
def exterior_angle (polygon : Type) [Polygon polygon] : ℝ := 30

-- Define the property that we want to prove
theorem diagonals_from_vertex {polygon : Type} [Polygon polygon] 
  (h : ∀ angle, exterior_angle polygon = 30) :
  ∃ (n : ℕ), (∑_angles angle = 360) ∧ (each_angle_30 angle) → (number_of_sides polygon = 12)
  ∧ (number_of_diags_from_vertex n = 9) :=
sorry

end diagonals_from_vertex_l578_578024


namespace value_of_f_minus_g_at_7_l578_578330

def f : ℝ → ℝ := λ x, 3
def g : ℝ → ℝ := λ x, 5

theorem value_of_f_minus_g_at_7 : f 7 - g 7 = -2 := by
  sorry

end value_of_f_minus_g_at_7_l578_578330


namespace binomial_60_3_eq_34220_l578_578295

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l578_578295


namespace banana_arrangements_l578_578627

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578627


namespace permutations_of_BANANA_l578_578388

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578388


namespace number_of_unique_permutations_BANANA_l578_578364

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578364


namespace permutations_BANANA_l578_578552

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578552


namespace neg_sqrt_of_0_81_l578_578171

theorem neg_sqrt_of_0_81 : -real.sqrt 0.81 = -0.9 :=
by
  sorry

end neg_sqrt_of_0_81_l578_578171


namespace banana_unique_permutations_l578_578849

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578849


namespace permutations_BANANA_l578_578943

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578943


namespace banana_unique_permutations_l578_578873

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578873


namespace segment_length_OI_is_3_l578_578188

-- Define the points along the path
def point (n : ℕ) : ℝ × ℝ := (n, n)

-- Use the Pythagorean theorem to calculate the distance from point O to point I
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the points O and I
def O : ℝ × ℝ := point 0
def I : ℝ × ℝ := point 3

-- The proposition to prove: 
-- The distance between points O and I is 3
theorem segment_length_OI_is_3 : distance O I = 3 := 
  sorry

end segment_length_OI_is_3_l578_578188


namespace arrangements_of_BANANA_l578_578422

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578422


namespace permutations_BANANA_l578_578540

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l578_578540


namespace banana_unique_permutations_l578_578872

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l578_578872


namespace numberOfWaysToArrangeBANANA_l578_578733

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578733


namespace number_of_unique_permutations_BANANA_l578_578350

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578350


namespace arrangements_of_BANANA_l578_578446

theorem arrangements_of_BANANA : 
  let letters := 6
  let N_repeats := 2
  let A_repeats := 3
  number_of_arrangements = (letters! / (N_repeats! * A_repeats!)) :=
by
  have letters_eq := 6
  have N_repeats_eq := 2
  have A_repeats_eq := 3
  have factorial := Nat.factorial

  have number_of_arrangements := factorial letters_eq / (factorial N_repeats_eq * factorial A_repeats_eq)

  exact number_of_arrangements = 60

end arrangements_of_BANANA_l578_578446


namespace binom_60_3_eq_34220_l578_578310

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l578_578310


namespace binom_60_3_eq_34220_l578_578309

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l578_578309


namespace permutations_BANANA_l578_578956

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578956


namespace arrangement_count_BANANA_l578_578688

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578688


namespace permutations_BANANA_l578_578921

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l578_578921


namespace compute_area_ratio_l578_578247

noncomputable def area_ratio (K : ℝ) : ℝ :=
  let small_triangle_area := 2 * K
  let large_triangle_area := 8 * K
  small_triangle_area / large_triangle_area

theorem compute_area_ratio (K : ℝ) : area_ratio K = 1 / 4 :=
by
  unfold area_ratio
  sorry

end compute_area_ratio_l578_578247


namespace permutations_of_BANANA_l578_578405

/-- Number of distinct permutations of the letters in the word BANANA -/
theorem permutations_of_BANANA : nat.perm_count "BANANA" = 60 := by
  have fact_6 : nat.fact 6 = 720 := by norm_num
  have fact_3 : nat.fact 3 = 6 := by norm_num
  have fact_2 : nat.fact 2 = 2 := by norm_num
  calc
    nat.perm_count "BANANA"
        -- Overall factorial: 6!
        = nat.fact 6 / (nat.fact 3 * nat.fact 2) : by sorry
    ... = 720 / (6 * 2)                         : by rw [fact_6, fact_3, fact_2]
    ... = 720 / 12                            : by norm_num
    ... = 60                                  : by norm_num

end permutations_of_BANANA_l578_578405


namespace number_of_arrangements_BANANA_l578_578791

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578791


namespace banana_arrangements_l578_578884

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578884


namespace lattice_points_on_graph_l578_578003

theorem lattice_points_on_graph :
  ∃ S : Finset (ℤ × ℤ), 
    (∀ p ∈ S, let (x, y) := p in x^2 - y^2 = 53) 
    ∧ (Finset.card S = 4) := 
by 
  sorry

end lattice_points_on_graph_l578_578003


namespace arrangement_count_BANANA_l578_578699

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578699


namespace sum_of_sides_l578_578032

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cosB cosC : ℝ)
variable (sinB : ℝ)
variable (area : ℝ)

-- Given conditions
axiom h1 : b = 2
axiom h2 : b * cosC + c * cosB = 3 * a * cosB
axiom h3 : area = 3 * Real.sqrt 2 / 2
axiom h4 : sinB = Real.sqrt (1 - cosB ^ 2)

-- Prove the desired result
theorem sum_of_sides (A B C a b c cosB cosC sinB : ℝ) (area : ℝ)
  (h1 : b = 2)
  (h2 : b * cosC + c * cosB = 3 * a * cosB)
  (h3 : area = 3 * Real.sqrt 2 / 2)
  (h4 : sinB = Real.sqrt (1 - cosB ^ 2)) :
  a + c = 4 := 
sorry

end sum_of_sides_l578_578032


namespace number_of_ways_to_arrange_BANANA_l578_578811

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578811


namespace number_of_ways_to_arrange_BANANA_l578_578823

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578823


namespace initial_customers_proof_l578_578256

-- Define the conditions as variables
variables (initial_customers : ℕ) (left_customers new_customers remaining_customers : ℕ)

-- Specify the conditions given in the problem
def conditions :=
  left_customers = 3 ∧
  remaining_customers = 5 ∧
  new_customers = 99 ∧
  remaining_customers = initial_customers - left_customers

-- The statement that we need to prove:
theorem initial_customers_proof (h : conditions initial_customers left_customers new_customers remaining_customers) :
  initial_customers = 8 :=
begin
  -- Use sorry to skip the proof part
  sorry
end

end initial_customers_proof_l578_578256


namespace number_of_arrangements_l578_578511

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578511


namespace BANANA_arrangement_l578_578467

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l578_578467


namespace arrange_banana_l578_578601

theorem arrange_banana : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let permutations := Nat.factorial total_letters
  let adjust_A := Nat.factorial count_A
  let adjust_N := Nat.factorial count_N
  permutations / (adjust_A * adjust_N) = 60 := by
  -- proof steps will be filled in here
  sorry

end arrange_banana_l578_578601


namespace number_of_arrangements_BANANA_l578_578785

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578785


namespace BANANA_arrangements_l578_578999

theorem BANANA_arrangements : 
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  (n.factorial / (n1.factorial * n2.factorial * n3.factorial)) = 60 :=
by
  let n := 6
  let n1 := 3
  let n2 := 2
  let n3 := 1
  have h : n.factorial / (n1.factorial * n2.factorial * n3.factorial) = 60 := sorry
  exact h

end BANANA_arrangements_l578_578999


namespace binom_60_3_eq_34220_l578_578308

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l578_578308


namespace number_of_ways_to_arrange_BANANA_l578_578807

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578807


namespace banana_arrangements_l578_578910

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l578_578910


namespace arrangement_count_BANANA_l578_578720

theorem arrangement_count_BANANA : 
  let total := 6!
  let factorial_B := 1!
  let factorial_A := 3!
  let factorial_N := 2!
  total / (factorial_B * factorial_A * factorial_N) = 60 :=
by
  -- Definitions of factorial functions
  have h1 : factorial 6 = 720 := by sorry
  have h2 : factorial 1 = 1 := by sorry
  have h3 : factorial 3 = 6 := by sorry
  have h4 : factorial 2 = 2 := by sorry
  -- Calculation verification
  have h5 : 720 / (1 * 6 * 2) = 60 := by sorry
  exact h5

end arrangement_count_BANANA_l578_578720


namespace distinct_permutations_BANANA_l578_578668

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l578_578668


namespace sequence_problem_proof_l578_578121

-- Define the sequence terms, using given conditions
def a_1 : ℕ := 1
def a_2 : ℕ := 2
def a_3 : ℕ := a_1 + a_2
def a_4 : ℕ := a_2 + a_3
def x : ℕ := a_3 + a_4

-- Prove that x = 8
theorem sequence_problem_proof : x = 8 := 
by
  sorry

end sequence_problem_proof_l578_578121


namespace number_of_ways_to_arrange_BANANA_l578_578826

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l578_578826


namespace binom_60_3_eq_34220_l578_578301

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l578_578301


namespace number_of_arrangements_BANANA_l578_578795

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l578_578795


namespace certain_number_is_one_l578_578026

theorem certain_number_is_one (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) 
    (h_rem : (p ^ 2) % 12 = 1) : ∃ n : ℕ, ((p ^ 2 + n) % 12 = 2) ∧ n = 1 :=
by
    use 1
    split
    sorry

end certain_number_is_one_l578_578026


namespace number_of_arrangements_l578_578518

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578518


namespace students_per_class_l578_578232

-- Define the conditions
variables (c : ℕ) (h_c : c ≥ 1) (s : ℕ)

-- Define the total number of books read by one student per year
def books_per_student_per_year := 5 * 12

-- Define the total number of students
def total_number_of_students := c * s

-- Define the total number of books read by the entire student body
def total_books_read := total_number_of_students * books_per_student_per_year

-- The given condition that the entire student body reads 60 books in one year
axiom total_books_eq_60 : total_books_read = 60

theorem students_per_class (h_c : c ≥ 1) : s = 1 / c :=
by sorry

end students_per_class_l578_578232


namespace number_of_arrangements_l578_578496

-- Define the parameters of the problem
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- The main theorem we want to prove
theorem number_of_arrangements : (fact total_letters) / ((fact count_A) * (fact count_N)) = 60 :=
sorry

end number_of_arrangements_l578_578496


namespace numberOfWaysToArrangeBANANA_l578_578731

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l578_578731


namespace number_of_unique_permutations_BANANA_l578_578362

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l578_578362
