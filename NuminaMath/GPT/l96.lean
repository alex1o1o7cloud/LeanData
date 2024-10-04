import Mathlib

namespace BANANA_permutations_l96_96199

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96199


namespace banana_arrangements_l96_96573

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96573


namespace binomial_60_3_eq_34220_l96_96039

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l96_96039


namespace permutations_of_BANANA_l96_96708

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96708


namespace partial_fraction_product_zero_l96_96816

theorem partial_fraction_product_zero
  (A B C : ℚ)
  (partial_fraction_eq : ∀ x : ℚ,
    x^2 - 25 = A * (x + 3) * (x - 5) + B * (x - 3) * (x - 5) + C * (x - 3) * (x + 3))
  (fact_3 : C = 0)
  (fact_neg3 : B = 1/3)
  (fact_5 : A = 0) :
  A * B * C = 0 := 
sorry

end partial_fraction_product_zero_l96_96816


namespace banana_arrangements_l96_96106

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96106


namespace train_length_l96_96011

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

end train_length_l96_96011


namespace production_line_B_units_l96_96004

theorem production_line_B_units
  (total_units : ℕ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
  (h_total_units : total_units = 5000)
  (h_ratio : ratio_A = 1 ∧ ratio_B = 2 ∧ ratio_C = 2) :
  (2 * (total_units / (ratio_A + ratio_B + ratio_C))) = 2000 :=
by
  sorry

end production_line_B_units_l96_96004


namespace banana_permutations_l96_96461

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96461


namespace distinct_permutations_BANANA_l96_96291

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96291


namespace BANANA_arrangement_l96_96163

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96163


namespace number_of_ways_to_arrange_BANANA_l96_96433

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96433


namespace area_of_rectangle_l96_96865

-- Definitions of the conditions
def length (w : ℝ) : ℝ := 4 * w
def perimeter_eq_200 (w l : ℝ) : Prop := 2 * l + 2 * w = 200

-- Main theorem statement
theorem area_of_rectangle (w l : ℝ) (h1 : length w = l) (h2 : perimeter_eq_200 w l) : l * w = 1600 :=
by
  -- Skip the proof
  sorry

end area_of_rectangle_l96_96865


namespace banana_arrangements_l96_96689

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96689


namespace binom_60_3_eq_34220_l96_96036

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l96_96036


namespace fraction_to_terminating_decimal_l96_96783

theorem fraction_to_terminating_decimal :
  (47 / (2^3 * 5^4) : ℝ) = 0.0094 := by
  sorry

end fraction_to_terminating_decimal_l96_96783


namespace arrange_BANANA_l96_96527

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96527


namespace BANANA_permutation_l96_96184

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96184


namespace number_of_negative_x_l96_96946

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l96_96946


namespace banana_arrangements_l96_96659

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96659


namespace numberOfWaysToArrangeBANANA_l96_96086

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96086


namespace number_of_unique_permutations_BANANA_l96_96535

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96535


namespace BANANA_arrangements_l96_96718

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

end BANANA_arrangements_l96_96718


namespace banana_arrangements_l96_96565

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96565


namespace negative_values_of_x_l96_96951

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l96_96951


namespace BANANA_permutation_l96_96178

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96178


namespace lcm_12_20_correct_l96_96791

def lcm_12_20_is_60 : Nat := Nat.lcm 12 20

theorem lcm_12_20_correct : Nat.lcm 12 20 = 60 := by
  -- assumed factorization conditions as prerequisites
  have h₁ : Nat.primeFactors 12 = {2, 3} := sorry
  have h₂ : Nat.primeFactors 20 = {2, 5} := sorry
  -- the main proof goal
  exact sorry

end lcm_12_20_correct_l96_96791


namespace BANANA_arrangements_l96_96725

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

end BANANA_arrangements_l96_96725


namespace permutations_BANANA_l96_96411

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96411


namespace permutations_of_BANANA_l96_96125

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

end permutations_of_BANANA_l96_96125


namespace BANANA_arrangements_correct_l96_96331

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96331


namespace arrangement_count_BANANA_l96_96489

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96489


namespace banana_arrangements_l96_96442

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96442


namespace product_of_divisors_36_l96_96794

-- Problem statement: Prove that the product of the divisors of 36 is 10077696
theorem product_of_divisors_36 : 
  let n := 36 in
  let p := 2 in
  let q := 3 in
  n = p * p * q * q →
  (∏ d in (Multiset.to_finset (Multiset.divisors n)).val, d) = 10077696 :=
by
  sorry

end product_of_divisors_36_l96_96794


namespace permutations_BANANA_l96_96361

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96361


namespace petes_average_speed_is_correct_l96_96020

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

end petes_average_speed_is_correct_l96_96020


namespace banana_arrangements_l96_96566

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96566


namespace banana_arrangements_l96_96572

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96572


namespace numberOfWaysToArrangeBANANA_l96_96094

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96094


namespace simplify_fraction_l96_96022

variable (x y : ℝ)

theorem simplify_fraction (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 * x^2 / y) * (y^2 / (2 * x)) = 3 * x * y / 2 :=
by sorry

end simplify_fraction_l96_96022


namespace BANANA_permutations_l96_96198

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96198


namespace permutations_BANANA_l96_96399

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96399


namespace BANANA_permutations_l96_96195

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96195


namespace cucumber_weight_l96_96927

theorem cucumber_weight (W : ℝ)
  (h1 : W * 0.99 + W * 0.01 = W)
  (h2 : (W * 0.01) / 20 = 1 / 95) :
  W = 100 :=
by
  sorry

end cucumber_weight_l96_96927


namespace BANANA_arrangements_l96_96727

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

end BANANA_arrangements_l96_96727


namespace number_of_negative_x_l96_96950

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l96_96950


namespace num_ways_to_arrange_BANANA_l96_96351

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96351


namespace arrangement_count_BANANA_l96_96231

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

end arrangement_count_BANANA_l96_96231


namespace banana_arrangements_l96_96095

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96095


namespace slope_intercept_parallel_line_l96_96936

def is_parallel (m1 m2 : ℝ) : Prop :=
  m1 = m2

theorem slope_intercept_parallel_line (A : ℝ × ℝ) (hA₁ : A.1 = 3) (hA₂ : A.2 = 2) 
  (m : ℝ) (h_parallel : is_parallel m (-4)) : ∃ b : ℝ, ∀ x y : ℝ, y = -4 * x + b :=
by
  use 14
  intro x y
  sorry

end slope_intercept_parallel_line_l96_96936


namespace BANANA_permutations_l96_96632

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96632


namespace permutations_BANANA_l96_96359

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96359


namespace arrange_banana_l96_96149

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

end arrange_banana_l96_96149


namespace orthogonal_vectors_l96_96995

open Real

variables (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : (a + b)^2 = (a - b)^2)

theorem orthogonal_vectors (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (h : (a + b)^2 = (a - b)^2) : a * b = 0 :=
by 
  sorry

end orthogonal_vectors_l96_96995


namespace BANANA_permutations_l96_96200

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96200


namespace find_k_l96_96826

theorem find_k (x y k : ℝ)
  (h1 : x - 4 * y + 3 ≤ 0)
  (h2 : 3 * x + 5 * y - 25 ≤ 0)
  (h3 : x ≥ 1)
  (h4 : ∃ z, z = k * x + y ∧ z = 12)
  (h5 : ∃ z', z' = k * x + y ∧ z' = 3) :
  k = 2 :=
by sorry

end find_k_l96_96826


namespace BANANA_arrangements_correct_l96_96330

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96330


namespace negative_values_count_l96_96982

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l96_96982


namespace number_of_arrangements_l96_96241

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

end number_of_arrangements_l96_96241


namespace number_of_arrangements_BANANA_l96_96760

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96760


namespace train_length_l96_96891

theorem train_length (v : ℝ) (t : ℝ) (conversion_factor : ℝ) : v = 45 → t = 16 → conversion_factor = 1000 / 3600 → (v * (conversion_factor) * t) = 200 :=
  by
  intros hv ht hcf
  rw [hv, ht, hcf]
  -- Proof steps skipped
  sorry

end train_length_l96_96891


namespace BANANA_arrangement_l96_96172

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96172


namespace num_of_negative_x_l96_96977

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l96_96977


namespace number_of_unique_permutations_BANANA_l96_96537

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96537


namespace banana_arrangements_l96_96258

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96258


namespace arrangement_count_BANANA_l96_96488

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96488


namespace number_of_repeating_decimals_l96_96818

open Nat

theorem number_of_repeating_decimals :
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 15) → (¬ ∃ k : ℕ, k * 18 = n) :=
by
  intros n h
  sorry

end number_of_repeating_decimals_l96_96818


namespace arrange_banana_l96_96153

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

end arrange_banana_l96_96153


namespace BANANA_permutations_l96_96617

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96617


namespace banana_arrangements_l96_96112

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96112


namespace num_ways_to_arrange_BANANA_l96_96338

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96338


namespace banana_arrangements_l96_96438

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96438


namespace quadratic_has_two_distinct_real_roots_l96_96870

theorem quadratic_has_two_distinct_real_roots : 
  ∃ α β : ℝ, (α ≠ β) ∧ (2 * α^2 - 3 * α + 1 = 0) ∧ (2 * β^2 - 3 * β + 1 = 0) :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l96_96870


namespace banana_permutations_l96_96467

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96467


namespace BANANA_permutations_l96_96206

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96206


namespace banana_arrangements_l96_96569

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96569


namespace banana_arrangements_l96_96686

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96686


namespace number_of_arrangements_BANANA_l96_96769

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96769


namespace banana_arrangements_l96_96660

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96660


namespace number_of_unique_permutations_BANANA_l96_96546

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96546


namespace BANANA_arrangement_l96_96169

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96169


namespace banana_arrangements_l96_96668

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96668


namespace line_passes_through_fixed_point_minimal_triangle_area_eq_line_l96_96812

-- Part (1)
theorem line_passes_through_fixed_point (m : ℝ) :
  ∃ M : ℝ × ℝ, M = (-1, -2) ∧
    (∀ m : ℝ, (2 + m) * (-1) + (1 - 2 * m) * (-2) + (4 - 3 * m) = 0) := by
  sorry

-- Part (2)
theorem minimal_triangle_area_eq_line :
  ∃ k : ℝ, k = -2 ∧ 
    (∀ x y : ℝ, y = k * (x + 1) - 2 ↔ y = 2 * x + 4) := by
  sorry

end line_passes_through_fixed_point_minimal_triangle_area_eq_line_l96_96812


namespace arrangement_count_BANANA_l96_96223

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

end arrangement_count_BANANA_l96_96223


namespace candles_used_l96_96014

theorem candles_used (starting_candles used_candles remaining_candles : ℕ) (h1 : starting_candles = 44) (h2 : remaining_candles = 12) : used_candles = 32 :=
by
  sorry

end candles_used_l96_96014


namespace number_of_arrangements_BANANA_l96_96770

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96770


namespace banana_permutations_l96_96472

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96472


namespace binom_60_3_eq_34220_l96_96037

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l96_96037


namespace min_distance_point_to_line_l96_96793

theorem min_distance_point_to_line :
    ∀ (x y : ℝ), (x^2 + y^2 - 6 * x - 4 * y + 12 = 0) -> 
    (3 * x + 4 * y - 2 = 0) -> 
    ∃ d: ℝ, d = 2 :=
by sorry

end min_distance_point_to_line_l96_96793


namespace arrange_BANANA_l96_96534

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96534


namespace arrangement_count_BANANA_l96_96225

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

end arrangement_count_BANANA_l96_96225


namespace banana_arrangements_l96_96657

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96657


namespace BANANA_permutations_l96_96620

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96620


namespace number_of_arrangements_banana_l96_96382

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96382


namespace banana_arrangements_l96_96109

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96109


namespace permutations_BANANA_l96_96414

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96414


namespace banana_arrangements_l96_96684

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96684


namespace min_value_AP_AQ_l96_96993

noncomputable def min_distance (A P Q : ℝ × ℝ) : ℝ := dist A P + dist A Q

theorem min_value_AP_AQ :
  ∀ (A P Q : ℝ × ℝ),
    (∀ (x : ℝ), A = (x, 0)) →
    ((P.1 - 1) ^ 2 + (P.2 - 3) ^ 2 = 1) →
    ((Q.1 - 7) ^ 2 + (Q.2 - 5) ^ 2 = 4) →
    min_distance A P Q = 7 :=
by
  intros A P Q hA hP hQ
  -- Proof is to be provided here
  sorry

end min_value_AP_AQ_l96_96993


namespace banana_permutations_l96_96507

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96507


namespace banana_arrangements_l96_96663

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96663


namespace banana_arrangements_l96_96681

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96681


namespace BANANA_arrangement_l96_96156

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96156


namespace height_of_larger_triangle_l96_96881

theorem height_of_larger_triangle 
  (area_ratio : ℝ)
  (height_small_triangle : ℝ)
  (similar_triangles : Prop)
  (height_large_triangle : ℝ) :
  area_ratio = 1 / 9 →
  height_small_triangle = 5 →
  similar_triangles →
  height_large_triangle = height_small_triangle * 3 :=
begin
  intros h_ratio h_height_small h_similar,
  rw h_ratio at *,
  rw h_height_small at *,
  exact eq.symm (mul_eq_mul_left_iff.1 (eq.trans (sqrt_eq (by norm_num) (by norm_num)) (by norm_num))),
sorry,
end

# The above code imports the necessary library, defines the theorem with the conditions and concludes with the height of the larger triangle.

end height_of_larger_triangle_l96_96881


namespace permutations_banana_l96_96643

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96643


namespace numberOfWaysToArrangeBANANA_l96_96080

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96080


namespace BANANA_arrangements_l96_96716

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

end BANANA_arrangements_l96_96716


namespace permutations_BANANA_l96_96312

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96312


namespace banana_unique_permutations_l96_96751

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96751


namespace fraction_to_terminating_decimal_l96_96784

theorem fraction_to_terminating_decimal :
  (47 / (2^3 * 5^4) : ℝ) = 0.0094 := by
  sorry

end fraction_to_terminating_decimal_l96_96784


namespace arrange_banana_l96_96138

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

end arrange_banana_l96_96138


namespace banana_permutations_l96_96504

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96504


namespace banana_unique_permutations_l96_96743

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96743


namespace banana_permutations_l96_96469

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96469


namespace BANANA_arrangements_correct_l96_96317

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96317


namespace permutations_of_BANANA_l96_96130

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

end permutations_of_BANANA_l96_96130


namespace arrange_banana_l96_96135

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

end arrange_banana_l96_96135


namespace permutations_BANANA_l96_96306

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96306


namespace banana_permutations_l96_96474

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96474


namespace arrange_BANANA_l96_96596

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96596


namespace BANANA_arrangements_l96_96724

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

end BANANA_arrangements_l96_96724


namespace BANANA_arrangements_correct_l96_96319

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96319


namespace side_length_percentage_error_l96_96916

variable (s s' : Real)
-- Conditions
-- s' = s * 1.06 (measured side length is 6% more than actual side length)
-- (s'^2 - s^2) / s^2 * 100% = 12.36% (percentage error in area)

theorem side_length_percentage_error 
    (h1 : s' = s * 1.06)
    (h2 : (s'^2 - s^2) / s^2 * 100 = 12.36) :
    ((s' - s) / s) * 100 = 6 := 
sorry

end side_length_percentage_error_l96_96916


namespace jacket_purchase_price_l96_96007

theorem jacket_purchase_price (S D P : ℝ) 
  (h1 : S = P + 0.30 * S)
  (h2 : D = 0.80 * S)
  (h3 : 6.000000000000007 = D - P) :
  P = 42 :=
by
  sorry

end jacket_purchase_price_l96_96007


namespace BANANA_permutation_l96_96177

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96177


namespace arrangement_count_BANANA_l96_96226

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

end arrangement_count_BANANA_l96_96226


namespace arrange_banana_l96_96151

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

end arrange_banana_l96_96151


namespace distinct_permutations_BANANA_l96_96290

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96290


namespace arrangements_of_BANANA_l96_96577

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

end arrangements_of_BANANA_l96_96577


namespace arrangement_count_BANANA_l96_96228

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

end arrangement_count_BANANA_l96_96228


namespace num_students_59_l96_96006

theorem num_students_59 (apples : ℕ) (taken_each : ℕ) (students : ℕ) 
  (h_apples : apples = 120) 
  (h_taken_each : taken_each = 2) 
  (h_students_divisors : ∀ d, d = 59 → d ∣ (apples / taken_each)) : students = 59 :=
sorry

end num_students_59_l96_96006


namespace banana_arrangements_l96_96257

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96257


namespace permutations_of_BANANA_l96_96701

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96701


namespace maxRegions_four_planes_maxRegions_n_planes_l96_96874

noncomputable def maxRegions (n : ℕ) : ℕ :=
  1 + (n * (n + 1)) / 2

theorem maxRegions_four_planes : maxRegions 4 = 11 := by
  sorry

theorem maxRegions_n_planes (n : ℕ) : maxRegions n = 1 + (n * (n + 1)) / 2 := by
  sorry

end maxRegions_four_planes_maxRegions_n_planes_l96_96874


namespace number_of_unique_permutations_BANANA_l96_96543

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96543


namespace count_negative_values_correct_l96_96965

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l96_96965


namespace number_of_arrangements_BANANA_l96_96756

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96756


namespace banana_arrangements_l96_96568

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96568


namespace binary_arithmetic_l96_96785

theorem binary_arithmetic :
  (110010:ℕ) * (1100:ℕ) / (100:ℕ) / (10:ℕ) = 100100 :=
by sorry

end binary_arithmetic_l96_96785


namespace banana_permutations_l96_96508

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96508


namespace BANANA_arrangements_correct_l96_96320

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96320


namespace arrangements_of_BANANA_l96_96587

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

end arrangements_of_BANANA_l96_96587


namespace largest_n_factorization_l96_96933

theorem largest_n_factorization :
  ∃ (n : ℤ), (∀ A B : ℤ, 3 * B + A = n -> A * B = 90 -> n ≤ 271) ∧ (∃ A B : ℤ, 3 * B + A = 271 ∧ A * B = 90) :=
by {
  apply Exists.intro 271,
  constructor,
  {
    intros A B eqn₁ eqn₂,
    have aux : 3 * B + A ≤ 271,
    sorry, -- Proof steps would go here
    exact aux,
  },
  {
    apply Exists.intro 1,
    apply Exists.intro 90,
    split,
    exact rfl,
    exact rfl,
  }
}

end largest_n_factorization_l96_96933


namespace meal_cost_l96_96987

theorem meal_cost (x : ℝ) (h1 : ∀ (x : ℝ), (x / 4) - 6 = x / 9) : 
  x = 43.2 :=
by
  have h : (∀ (x : ℝ), (x / 4) - (x / 9) = 6) := sorry
  exact sorry

end meal_cost_l96_96987


namespace calculate_expression_l96_96999

variable (x : ℝ)

def quadratic_condition : Prop := x^2 + x - 1 = 0

theorem calculate_expression (h : quadratic_condition x) : 2*x^3 + 3*x^2 - x = 1 := by
  sorry

end calculate_expression_l96_96999


namespace distinct_permutations_BANANA_l96_96277

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96277


namespace banana_permutations_l96_96459

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96459


namespace triangle_shortest_side_l96_96859

theorem triangle_shortest_side (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (base : Real) (base_angle : Real) (sum_other_sides : Real)
    (h1 : base = 80) 
    (h2 : base_angle = 60) 
    (h3 : sum_other_sides = 90) : 
    ∃ shortest_side : Real, shortest_side = 17 :=
by 
    sorry

end triangle_shortest_side_l96_96859


namespace total_strength_of_college_l96_96833

-- Declare the variables for number of students playing each sport
variables (C B Both : ℕ)

-- Given conditions in the problem
def cricket_players : ℕ := 500
def basketball_players : ℕ := 600
def both_players : ℕ := 220

-- Theorem stating the total strength of the college
theorem total_strength_of_college (h_C : C = cricket_players) 
                                  (h_B : B = basketball_players) 
                                  (h_Both : Both = both_players) : 
                                  C + B - Both = 880 :=
by
  sorry

end total_strength_of_college_l96_96833


namespace BANANA_arrangements_correct_l96_96325

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96325


namespace banana_arrangements_l96_96685

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96685


namespace tan_alpha_sin_cos_half_alpha_l96_96800

variable (α : ℝ)

-- Conditions given in the problem
def cond1 : Real.sin α = 1 / 3 := sorry
def cond2 : 0 < α ∧ α < Real.pi := sorry

-- Lean proof that given the conditions, the solutions are as follows:
theorem tan_alpha (h1 : Real.sin α = 1 / 3) (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = Real.sqrt 2 / 4 ∨ Real.tan α = - Real.sqrt 2 / 4 := sorry

theorem sin_cos_half_alpha (h1 : Real.sin α = 1 / 3) (h2 : 0 < α ∧ α < Real.pi) : 
  Real.sin (α / 2) + Real.cos (α / 2) = 2 * Real.sqrt 3 / 3 := sorry

end tan_alpha_sin_cos_half_alpha_l96_96800


namespace parabola_equation_line_intersection_proof_l96_96815

-- Define the parabola and its properties
def parabola (p x y : ℝ) := y^2 = 2 * p * x

-- Define point A
def A_point (x y₀ : ℝ) := (x, y₀)

-- Define the conditions
axiom p_pos (p : ℝ) : p > 0
axiom passes_A (y₀ : ℝ) (p : ℝ) : parabola p 2 y₀
axiom distance_A_axis (p : ℝ) : 2 + p / 2 = 4

-- Prove the equation of the parabola given the conditions
theorem parabola_equation : ∃ p, parabola p x y ∧ p = 4 := sorry

-- Define line l and its intersection properties
def line_l (m x y : ℝ) := y = x + m
def intersection_PQ (m x₁ x₂ y₁ y₂ : ℝ) := 
  line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧ y₁^2 = 8 * x₁ ∧ y₂^2 = 8 * x₂ ∧ 
  x₁ + x₂ = 8 - 2 * m ∧ x₁ * x₂ = m^2 ∧ y₁ + y₂ = 8 ∧ y₁ * y₂ = 8 * m ∧ 
  x₁ * x₂ + y₁ * y₂ = 0

-- Prove the value of m
theorem line_intersection_proof : ∃ m, ∀ (x₁ x₂ y₁ y₂ : ℝ), 
  intersection_PQ m x₁ x₂ y₁ y₂ -> m = -8 := sorry

end parabola_equation_line_intersection_proof_l96_96815


namespace intersection_count_l96_96813

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem intersection_count (ω φ : ℝ) (hω : ω > 0) (hφ : |φ| < Real.pi / 2) 
  (h_max : ∀ x, f x ω φ ≤ f (Real.pi / 6) ω φ)
  (h_period : ∀ x, f x ω φ = f (x + 2 * Real.pi / ω) ω φ) :
  ∃! x : ℝ, f x ω φ = -x + 2 * Real.pi / 3 :=
sorry

end intersection_count_l96_96813


namespace arrange_BANANA_l96_96526

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96526


namespace probability_at_least_four_same_value_l96_96938

-- Define the event of rolling five fair eight-sided dice
def roll_outcomes : ℕ := 8 -- There are 8 possible outcomes per die

-- Calculate the probability
theorem probability_at_least_four_same_value :
  let at_least_four_same_value := (35 / (8^4)) + (1 / (8^4))
  in at_least_four_same_value = (9 / 1024) :=
by
  let p1 := 1 / roll_outcomes^4
  let p2 := 5 * (1 / roll_outcomes^3) * (7 / roll_outcomes)
  have h : p1 + p2 = 36 / 4096 := sorry
  have h_uniform := (9 : ℚ) / 1024
  -- Ensure that the calculated probability matches the expected value
  show h_uniform = (p1 + p2)
  sorry

end probability_at_least_four_same_value_l96_96938


namespace permutations_BANANA_l96_96362

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96362


namespace C_and_D_complete_work_together_in_2_86_days_l96_96001

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

end C_and_D_complete_work_together_in_2_86_days_l96_96001


namespace arrange_BANANA_l96_96058

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96058


namespace banana_arrangements_l96_96263

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96263


namespace dot_product_sum_eq_fifteen_l96_96990

-- Define the vectors a, b, and c
def vec_a (x : ℝ) : ℝ × ℝ := (x, 1)
def vec_b (y : ℝ) : ℝ × ℝ := (1, y)
def vec_c : ℝ × ℝ := (3, -6)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Conditions from the problem
def cond_perpendicular (x : ℝ) : Prop :=
  dot_product (vec_a x) vec_c = 0

def cond_parallel (y : ℝ) : Prop :=
  1 / 3 = y / -6

-- Lean statement for the problem
theorem dot_product_sum_eq_fifteen (x y : ℝ)
  (h1 : cond_perpendicular x) 
  (h2 : cond_parallel y) :
  dot_product (vec_a x + vec_b y) vec_c = 15 :=
sorry

end dot_product_sum_eq_fifteen_l96_96990


namespace person6_number_l96_96879

theorem person6_number (a : ℕ → ℕ) (x : ℕ → ℕ) 
  (mod12 : ∀ i, a (i % 12) = a i)
  (h5 : x 5 = 5)
  (h6 : x 6 = 8)
  (h7 : x 7 = 11) 
  (h_avg : ∀ i, x i = (a (i-1) + a (i+1)) / 2) : 
  a 6 = 6 := sorry

end person6_number_l96_96879


namespace permutations_BANANA_l96_96398

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96398


namespace permutations_BANANA_l96_96397

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96397


namespace numberOfWaysToArrangeBANANA_l96_96093

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96093


namespace arrangements_of_BANANA_l96_96594

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

end arrangements_of_BANANA_l96_96594


namespace numberOfWaysToArrangeBANANA_l96_96088

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96088


namespace number_of_arrangements_l96_96251

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

end number_of_arrangements_l96_96251


namespace no_positive_integer_solutions_l96_96789

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), (a > 0) ∧ (b > 0) → 3 * a^2 ≠ b^2 + 1 :=
by
  sorry

end no_positive_integer_solutions_l96_96789


namespace banana_unique_permutations_l96_96749

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96749


namespace arrange_BANANA_l96_96067

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96067


namespace binom_60_3_eq_34220_l96_96034

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l96_96034


namespace numberOfWaysToArrangeBANANA_l96_96090

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96090


namespace banana_unique_permutations_l96_96741

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96741


namespace permutations_BANANA_l96_96371

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96371


namespace BANANA_permutations_l96_96207

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96207


namespace distinct_permutations_BANANA_l96_96286

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96286


namespace banana_arrangements_l96_96682

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96682


namespace permutations_BANANA_l96_96408

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96408


namespace BANANA_permutation_l96_96185

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96185


namespace num_ways_to_arrange_BANANA_l96_96348

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96348


namespace BANANA_arrangements_correct_l96_96324

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96324


namespace arrange_BANANA_l96_96597

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96597


namespace permutations_BANANA_l96_96299

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96299


namespace binomial_60_3_eq_34220_l96_96041

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l96_96041


namespace arrange_banana_l96_96145

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

end arrange_banana_l96_96145


namespace fraction_to_decimal_terminating_l96_96778

theorem fraction_to_decimal_terminating : 
  (47 / (2^3 * 5^4) : ℝ) = 0.5875 :=
by 
  sorry

end fraction_to_decimal_terminating_l96_96778


namespace arrange_BANANA_l96_96613

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96613


namespace arrange_BANANA_l96_96519

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96519


namespace negative_values_of_x_l96_96952

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l96_96952


namespace num_ways_to_arrange_BANANA_l96_96340

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96340


namespace number_of_arrangements_banana_l96_96380

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96380


namespace distinct_permutations_BANANA_l96_96280

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96280


namespace permutations_banana_l96_96651

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96651


namespace banana_permutations_l96_96465

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96465


namespace f_at_neg_one_l96_96828

def f (x : ℝ) : ℝ := sorry

theorem f_at_neg_one (h : ∀ x : ℝ, f (x - 1) = x^2 + 1) : f (-1) = 1 :=
by sorry

end f_at_neg_one_l96_96828


namespace arrange_BANANA_l96_96055

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96055


namespace arrangements_of_BANANA_l96_96578

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

end arrangements_of_BANANA_l96_96578


namespace permutations_BANANA_l96_96302

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96302


namespace arrangement_count_BANANA_l96_96234

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

end arrangement_count_BANANA_l96_96234


namespace arrange_BANANA_l96_96515

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96515


namespace BANANA_permutations_l96_96627

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96627


namespace algebra_problem_l96_96820

variable (a : ℝ)

-- Condition: Given (a + 1/a)^3 = 4
def condition : Prop := (a + 1/a)^3 = 4

-- Statement: Prove a^4 + 1/a^4 = -158/81
theorem algebra_problem (h : condition a) : a^4 + 1/a^4 = -158/81 := 
sorry

end algebra_problem_l96_96820


namespace banana_permutations_l96_96471

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96471


namespace BANANA_arrangement_l96_96171

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96171


namespace banana_permutations_l96_96501

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96501


namespace min_value_expression_l96_96803

theorem min_value_expression :
  (∀ y : ℝ, abs y ≤ 1 → ∃ x : ℝ, 2 * x + y = 1 ∧ ∀ x : ℝ, (0 ≤ x ∧ x ≤ 1 → 
    (∃ y : ℝ, 2 * x + y = 1 ∧ abs y ≤ 1 ∧ (2 * x ^ 2 + 16 * x + 3 * y ^ 2) = 3))) :=
sorry

end min_value_expression_l96_96803


namespace banana_arrangements_l96_96272

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96272


namespace banana_arrangements_l96_96693

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96693


namespace binom_60_3_eq_34220_l96_96035

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l96_96035


namespace part1_assoc_eq_part2_k_range_part3_m_range_l96_96905

-- Part 1
theorem part1_assoc_eq (x : ℝ) :
  (2 * (x + 1) - x = -3 ∧ (-4 < x ∧ x ≤ 4)) ∨ 
  ((x+1)/3 - 1 = x ∧ (-4 < x ∧ x ≤ 4)) ∨ 
  (2 * x - 7 = 0 ∧ (-4 < x ∧ x ≤ 4)) :=
sorry

-- Part 2
theorem part2_k_range (k : ℝ) :
  (∀ (x : ℝ), (x = (k + 6) / 2) → -5 < x ∧ x ≤ -3) ↔ (-16 < k) ∧ (k ≤ -12) :=
sorry 

-- Part 3
theorem part3_m_range (m : ℝ) :
  (∀ (x : ℝ), (x = 6 * m - 5) → (0 < x) ∧ (x ≤ 3 * m + 1) ∧ (1 ≤ x) ∧ (x ≤ 3)) ↔ (5/6 < m) ∧ (m < 1) :=
sorry

end part1_assoc_eq_part2_k_range_part3_m_range_l96_96905


namespace arrange_BANANA_l96_96530

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96530


namespace permutations_BANANA_l96_96358

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96358


namespace permutations_BANANA_l96_96407

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96407


namespace banana_permutations_l96_96511

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96511


namespace similar_triangles_height_l96_96885

theorem similar_triangles_height (h₁ h₂ : ℝ) (ratio_areas : ℝ) 
  (h₁_eq : h₁ = 5) (ratio_areas_eq : ratio_areas = 1 / 9)
  (similar : h₂^2 = (√ratio_areas)^2 * h₁^2) : h₂ = 15 :=
by {
  have ratio_areas_pos : ratio_areas > 0 := by (simp [ratio_areas_eq]),
  have k := √ratio_areas,
  have k_eq : k = 3 := by {
    rw [ratio_areas_eq, sqrt_div, sqrt_one, sqrt_nat_eq_iff_eq_sq, one_div_eq_inv] at *,
    norm_num },
  have h₂_def : h₂ = 3 * h₁ := by rw [h₁_eq, mul_comm, k_eq],
  rw [h₂_def],
  norm_num,
}

end similar_triangles_height_l96_96885


namespace banana_permutations_l96_96458

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96458


namespace number_of_arrangements_l96_96238

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

end number_of_arrangements_l96_96238


namespace num_of_negative_x_l96_96976

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l96_96976


namespace banana_permutations_l96_96470

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96470


namespace decimal_representation_of_fraction_l96_96781

theorem decimal_representation_of_fraction :
  (47 : ℝ) / (2^3 * 5^4) = 0.0094 :=
by
  sorry

end decimal_representation_of_fraction_l96_96781


namespace permutations_of_BANANA_l96_96117

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

end permutations_of_BANANA_l96_96117


namespace permutations_BANANA_l96_96368

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96368


namespace number_of_arrangements_banana_l96_96388

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96388


namespace arrangements_of_BANANA_l96_96581

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

end arrangements_of_BANANA_l96_96581


namespace numberOfWaysToArrangeBANANA_l96_96081

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96081


namespace students_first_day_l96_96910

-- Definitions based on conditions
def total_books : ℕ := 120
def books_per_student : ℕ := 5
def students_second_day : ℕ := 5
def students_third_day : ℕ := 6
def students_fourth_day : ℕ := 9

-- Main goal
theorem students_first_day (total_books_eq : total_books = 120)
                           (books_per_student_eq : books_per_student = 5)
                           (students_second_day_eq : students_second_day = 5)
                           (students_third_day_eq : students_third_day = 6)
                           (students_fourth_day_eq : students_fourth_day = 9) :
  let books_given_second_day := students_second_day * books_per_student
  let books_given_third_day := students_third_day * books_per_student
  let books_given_fourth_day := students_fourth_day * books_per_student
  let total_books_given_after_first_day := books_given_second_day + books_given_third_day + books_given_fourth_day
  let books_first_day := total_books - total_books_given_after_first_day
  let students_first_day := books_first_day / books_per_student
  students_first_day = 4 :=
by sorry

end students_first_day_l96_96910


namespace number_of_unique_permutations_BANANA_l96_96539

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96539


namespace arrange_BANANA_l96_96533

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96533


namespace number_of_ways_to_arrange_BANANA_l96_96415

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96415


namespace banana_unique_permutations_l96_96746

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96746


namespace permutations_BANANA_l96_96298

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96298


namespace BANANA_permutations_l96_96197

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96197


namespace permutations_BANANA_l96_96365

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96365


namespace number_of_arrangements_banana_l96_96379

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96379


namespace arrange_BANANA_l96_96066

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96066


namespace negative_values_count_l96_96981

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l96_96981


namespace banana_arrangements_l96_96435

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96435


namespace BANANA_arrangements_correct_l96_96318

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96318


namespace BANANA_permutation_l96_96188

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96188


namespace numberOfWaysToArrangeBANANA_l96_96085

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96085


namespace arrange_BANANA_l96_96072

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96072


namespace permutations_BANANA_l96_96369

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96369


namespace permutations_BANANA_l96_96374

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96374


namespace permutations_of_BANANA_l96_96129

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

end permutations_of_BANANA_l96_96129


namespace white_balls_in_bag_l96_96858

   theorem white_balls_in_bag (m : ℕ) (h : m ≤ 7) :
     (2 * (m * (m - 1) / 2) / (7 * 6 / 2)) + ((m * (7 - m)) / (7 * 6 / 2)) = 6 / 7 → m = 3 :=
   by
     intros h_eq
     sorry
   
end white_balls_in_bag_l96_96858


namespace BANANA_arrangements_correct_l96_96316

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96316


namespace banana_arrangements_l96_96101

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96101


namespace BANANA_permutations_l96_96624

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96624


namespace certain_number_mod_l96_96888

theorem certain_number_mod (n : ℤ) : (73 * n) % 8 = 7 → n % 8 = 7 := 
by sorry

end certain_number_mod_l96_96888


namespace distinct_permutations_BANANA_l96_96282

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96282


namespace arrangement_count_BANANA_l96_96492

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96492


namespace arrangement_count_BANANA_l96_96478

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96478


namespace arrange_BANANA_l96_96601

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96601


namespace number_of_unique_permutations_BANANA_l96_96542

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96542


namespace number_of_ways_to_arrange_BANANA_l96_96421

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96421


namespace permutations_of_BANANA_l96_96121

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

end permutations_of_BANANA_l96_96121


namespace count_negative_values_of_x_l96_96970

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l96_96970


namespace binom_60_3_eq_34220_l96_96047

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l96_96047


namespace arrange_BANANA_l96_96531

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96531


namespace banana_arrangements_l96_96688

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96688


namespace arrange_banana_l96_96137

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

end arrange_banana_l96_96137


namespace permutations_of_BANANA_l96_96712

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96712


namespace permutations_of_BANANA_l96_96133

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

end permutations_of_BANANA_l96_96133


namespace nineteen_power_six_l96_96032

theorem nineteen_power_six :
    19^11 / 19^5 = 47045881 := by
  sorry

end nineteen_power_six_l96_96032


namespace binom_60_3_eq_34220_l96_96050

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l96_96050


namespace darius_scores_less_l96_96924

variable (D M Ma : ℕ)

-- Conditions
def condition1 := D = 10
def condition2 := Ma = D + 3
def condition3 := D + M + Ma = 38

-- Theorem to prove
theorem darius_scores_less (D M Ma : ℕ) (h1 : condition1 D) (h2 : condition2 D Ma) (h3 : condition3 D M Ma) : M - D = 5 :=
by
  sorry

end darius_scores_less_l96_96924


namespace permutations_of_BANANA_l96_96704

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96704


namespace permutations_BANANA_l96_96396

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96396


namespace similar_triangles_height_ratio_l96_96882

theorem similar_triangles_height_ratio (area_ratio : ℝ) (h₁ : ℝ) (h₂ : ℝ) 
  (similar : Boolean) (h₁_value : h₁ = 5) (area_ratio_value : area_ratio = 9) :
  similar = true → area_ratio = (h₂ / h₁) ^ 2 → h₂ = 15 :=
by
  intro h_similar area_eq
  rw [h₁_value, area_ratio_value]
  sorry

end similar_triangles_height_ratio_l96_96882


namespace tom_total_cost_l96_96877

theorem tom_total_cost :
  let vaccines_cost := 10 * 45 in
  let total_medical_cost := vaccines_cost + 250 in
  let insurance_covered := 0.80 * total_medical_cost in
  let tom_pay_medical := total_medical_cost - insurance_covered in
  let trip_cost := 1200 in
  let total_cost := tom_pay_medical + trip_cost in
  total_cost = 1340 :=
by
  dsimp
  sorry

end tom_total_cost_l96_96877


namespace permutations_BANANA_l96_96301

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96301


namespace number_of_ways_to_arrange_BANANA_l96_96424

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96424


namespace banana_unique_permutations_l96_96744

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96744


namespace count_negative_x_with_sqrt_pos_int_l96_96961

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l96_96961


namespace BANANA_permutations_l96_96621

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96621


namespace banana_arrangements_l96_96440

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96440


namespace permutations_of_BANANA_l96_96703

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96703


namespace banana_arrangements_l96_96448

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96448


namespace banana_arrangements_l96_96667

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96667


namespace correct_propositions_l96_96015

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

end correct_propositions_l96_96015


namespace permutations_banana_l96_96647

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96647


namespace permutations_BANANA_l96_96300

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96300


namespace coin_ratio_l96_96000

theorem coin_ratio (coins_1r coins_50p coins_25p : ℕ) (value_1r value_50p value_25p : ℕ) :
  coins_1r = 120 → coins_50p = 120 → coins_25p = 120 →
  value_1r = coins_1r * 1 → value_50p = coins_50p * 50 → value_25p = coins_25p * 25 →
  value_1r + value_50p + value_25p = 210 →
  (coins_1r : ℚ) / (coins_50p + coins_25p : ℚ) = (1 / 1) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end coin_ratio_l96_96000


namespace BANANA_arrangements_l96_96723

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

end BANANA_arrangements_l96_96723


namespace number_of_ways_to_arrange_BANANA_l96_96427

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96427


namespace banana_arrangements_l96_96563

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96563


namespace BANANA_permutations_l96_96628

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96628


namespace BANANA_permutations_l96_96625

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96625


namespace permutations_BANANA_l96_96296

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96296


namespace number_of_arrangements_banana_l96_96384

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96384


namespace BANANA_arrangement_l96_96161

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96161


namespace cannot_lie_on_line_l96_96920

open Real

theorem cannot_lie_on_line (m b : ℝ) (h1 : m * b > 0) (h2 : b > 0) :
  (0, -2023) ≠ (0, b) :=
by
  sorry

end cannot_lie_on_line_l96_96920


namespace BANANA_arrangements_correct_l96_96329

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96329


namespace number_of_arrangements_l96_96239

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

end number_of_arrangements_l96_96239


namespace number_of_arrangements_BANANA_l96_96761

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96761


namespace arrange_BANANA_l96_96065

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96065


namespace BANANA_permutation_l96_96187

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96187


namespace BANANA_arrangements_correct_l96_96328

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96328


namespace BANANA_permutations_l96_96212

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96212


namespace determine_k_l96_96925

variable (x y z k : ℝ)

theorem determine_k (h : 9 / (x + y) = k / (y + z) ∧ k / (y + z) = 15 / (x - z)) : k = 0 := by
  sorry

end determine_k_l96_96925


namespace binomial_60_3_eq_34220_l96_96038

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l96_96038


namespace number_of_ways_to_arrange_BANANA_l96_96418

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96418


namespace numberOfWaysToArrangeBANANA_l96_96089

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96089


namespace banana_arrangements_l96_96270

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96270


namespace arrange_BANANA_l96_96060

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96060


namespace number_of_ways_to_arrange_BANANA_l96_96426

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96426


namespace number_of_arrangements_l96_96249

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

end number_of_arrangements_l96_96249


namespace side_of_beef_weight_after_processing_l96_96912

theorem side_of_beef_weight_after_processing (initial_weight : ℝ) (lost_percentage : ℝ) (final_weight : ℝ) 
  (h1 : initial_weight = 400) 
  (h2 : lost_percentage = 0.4) 
  (h3 : final_weight = initial_weight * (1 - lost_percentage)) : 
  final_weight = 240 :=
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end side_of_beef_weight_after_processing_l96_96912


namespace arrangements_of_BANANA_l96_96575

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

end arrangements_of_BANANA_l96_96575


namespace arrangement_count_BANANA_l96_96227

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

end arrangement_count_BANANA_l96_96227


namespace sequence_propositions_l96_96804

theorem sequence_propositions (a : ℕ → ℝ) (h_seq : a 1 > a 2 ∧ a 2 > a 3 ∧ a 3 > a 4 ∧ a 4 ≥ 0) 
  (h_sub : ∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 4 → ∃ k, a i - a j = a k) :
  (∀ k, ∃ d, a k = a 1 - d * (k - 1)) ∧
  (∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ i * a i = j * a j) ∧
  (∃ i, a i = 0) :=
by
  sorry

end sequence_propositions_l96_96804


namespace permutations_of_BANANA_l96_96709

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96709


namespace weight_difference_end_of_year_l96_96832

theorem weight_difference_end_of_year :
  let labrador_starting_weight := 40
  let dachshund_starting_weight := 12
  let weight_gain_percent := 0.25
  let labrador_end_weight := labrador_starting_weight + labrador_starting_weight * weight_gain_percent
  let dachshund_end_weight := dachshund_starting_weight + dachshund_starting_weight * weight_gain_percent
  (labrador_end_weight - dachshund_end_weight) = 35 :=
by
  let labrador_starting_weight := 40
  let dachshund_starting_weight := 12
  let weight_gain_percent := 0.25
  let labrador_end_weight := labrador_starting_weight + labrador_starting_weight * weight_gain_percent
  let dachshund_end_weight := dachshund_starting_weight + dachshund_starting_weight * weight_gain_percent
  -- Statement: difference in weight at end of year is 35 pounds
  have h : (labrador_end_weight - dachshund_end_weight) = 35 := sorry
  exact h

end weight_difference_end_of_year_l96_96832


namespace permutations_of_BANANA_l96_96124

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

end permutations_of_BANANA_l96_96124


namespace BANANA_arrangements_l96_96719

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

end BANANA_arrangements_l96_96719


namespace number_of_arrangements_l96_96236

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

end number_of_arrangements_l96_96236


namespace arrangements_of_BANANA_l96_96590

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

end arrangements_of_BANANA_l96_96590


namespace arrange_BANANA_l96_96062

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96062


namespace banana_permutations_l96_96463

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96463


namespace BANANA_arrangements_correct_l96_96315

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96315


namespace num_ways_to_arrange_BANANA_l96_96335

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96335


namespace num_ways_to_arrange_BANANA_l96_96350

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96350


namespace permutations_of_BANANA_l96_96118

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

end permutations_of_BANANA_l96_96118


namespace number_of_arrangements_banana_l96_96377

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96377


namespace banana_arrangements_l96_96103

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96103


namespace permutations_BANANA_l96_96311

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96311


namespace number_of_negative_x_l96_96945

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l96_96945


namespace arrangement_count_BANANA_l96_96479

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96479


namespace banana_arrangements_l96_96562

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96562


namespace complement_angle_l96_96801

theorem complement_angle (A : Real) (h : A = 55) : 90 - A = 35 := by
  sorry

end complement_angle_l96_96801


namespace arrangement_count_BANANA_l96_96477

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96477


namespace numberOfWaysToArrangeBANANA_l96_96078

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96078


namespace time_to_reach_ship_l96_96911

-- Define the conditions
def rate_of_descent := 30 -- feet per minute
def depth_to_ship := 2400 -- feet

-- Define the proof statement
theorem time_to_reach_ship : (depth_to_ship / rate_of_descent) = 80 :=
by
  -- The proof will be inserted here in practice
  sorry

end time_to_reach_ship_l96_96911


namespace banana_permutations_l96_96502

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96502


namespace banana_arrangements_l96_96267

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96267


namespace arrangement_count_BANANA_l96_96487

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96487


namespace arrange_BANANA_l96_96604

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96604


namespace binom_60_3_eq_34220_l96_96046

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l96_96046


namespace count_negative_values_correct_l96_96963

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l96_96963


namespace BANANA_arrangements_l96_96726

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

end BANANA_arrangements_l96_96726


namespace banana_arrangements_l96_96111

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96111


namespace simplify_fraction_126_11088_l96_96854

theorem simplify_fraction_126_11088 : (126 / 11088 : ℚ) = 1 / 88 := 
by {
  /- 
  Given:
  1. 126 can be factored as 2 * 3^2 * 7.
  2. 11088 can be factored as 2^4 * 3^2 * 7 * 11.
  3. The greatest common divisor (GCD) of 126 and 11088 is 126.
  Simplify the fraction.
  -/
  sorry 
}

end simplify_fraction_126_11088_l96_96854


namespace equivalent_fraction_l96_96928

theorem equivalent_fraction : (8 / (5 * 46)) = (0.8 / 23) := 
by sorry

end equivalent_fraction_l96_96928


namespace number_of_arrangements_BANANA_l96_96762

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96762


namespace BANANA_arrangement_l96_96165

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96165


namespace banana_arrangements_l96_96673

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96673


namespace BANANA_arrangement_l96_96174

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96174


namespace permutations_of_BANANA_l96_96707

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96707


namespace arrange_BANANA_l96_96522

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96522


namespace number_of_unique_permutations_BANANA_l96_96538

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96538


namespace banana_arrangements_l96_96564

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96564


namespace permutations_of_BANANA_l96_96700

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96700


namespace number_of_unique_permutations_BANANA_l96_96547

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96547


namespace arrange_BANANA_l96_96607

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96607


namespace num_ways_to_arrange_BANANA_l96_96354

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96354


namespace BANANA_arrangements_l96_96730

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

end BANANA_arrangements_l96_96730


namespace Sara_pears_left_l96_96851

def Sara_has_left (initial_pears : ℕ) (given_to_Dan : ℕ) (given_to_Monica : ℕ) (given_to_Jenny : ℕ) : ℕ :=
  initial_pears - given_to_Dan - given_to_Monica - given_to_Jenny

theorem Sara_pears_left :
  Sara_has_left 35 28 4 1 = 2 :=
by
  sorry

end Sara_pears_left_l96_96851


namespace negation_of_prop_p_l96_96825

open Classical

variable (p : Prop)

def prop_p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_prop_p : ¬prop_p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end negation_of_prop_p_l96_96825


namespace weight_difference_proof_l96_96831

theorem weight_difference_proof
  (labrador_start_weight : ℕ) (dachshund_start_weight : ℕ)
  (weight_gain_percentage : ℕ)
  (labrador_start_weight = 40)
  (dachshund_start_weight = 12)
  (weight_gain_percentage = 25) :
  (labrador_start_weight + labrador_start_weight * weight_gain_percentage / 100) -
  (dachshund_start_weight + dachshund_start_weight * weight_gain_percentage / 100) =
  35 := 
  sorry

end weight_difference_proof_l96_96831


namespace banana_permutations_l96_96513

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96513


namespace value_of_x_squared_minus_y_squared_l96_96823

theorem value_of_x_squared_minus_y_squared 
  (x y : ℚ)
  (h1 : x + y = 5 / 8) 
  (h2 : x - y = 3 / 8) :
  x^2 - y^2 = 15 / 64 :=
by 
  sorry

end value_of_x_squared_minus_y_squared_l96_96823


namespace banana_arrangements_l96_96656

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96656


namespace number_of_arrangements_BANANA_l96_96765

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96765


namespace degrees_of_interior_angles_l96_96805

-- Definitions for the problem conditions
variables {a b c h_a h_b S : ℝ} 
variables (ABC : Triangle) 
variables (height_to_bc height_to_ac : ℝ)
variables (le_a_ha : a ≤ height_to_bc)
variables (le_b_hb : b ≤ height_to_ac)
variables (area : S = 1 / 2 * a * height_to_bc)
variables (area_eq : S = 1 / 2 * b * height_to_ac)
variables (ha_eq : height_to_bc = 2 * S / a)
variables (hb_eq : height_to_ac = 2 * S / b)
variables (height_pos : 0 < 2 * S)
variables (length_pos : 0 < a ∧ 0 < b ∧ 0 < c)

-- Conclude the degrees of the interior angles
theorem degrees_of_interior_angles : 
  ∃ A B C : ℝ, A = 45 ∧ B = 45 ∧ C = 90 :=
sorry

end degrees_of_interior_angles_l96_96805


namespace banana_arrangements_l96_96266

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96266


namespace similar_triangles_height_l96_96883

open_locale classical

theorem similar_triangles_height
  (h_small : ℝ)
  (A_small A_large : ℝ)
  (area_ratio : ℝ)
  (h_large : ℝ) :
  A_small / A_large = 1 / 9 →
  h_small = 5 →
  area_ratio = A_large / A_small →
  area_ratio = 9 →
  h_large = h_small * sqrt area_ratio →
  h_large = 15 :=
by
  sorry

end similar_triangles_height_l96_96883


namespace banana_arrangements_l96_96273

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96273


namespace banana_permutations_l96_96509

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96509


namespace banana_arrangements_l96_96255

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96255


namespace BANANA_permutation_l96_96176

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96176


namespace total_peaches_l96_96848

-- Definitions based on the given conditions
def initial_peaches : ℕ := 13
def picked_peaches : ℕ := 55

-- The proof goal stating the total number of peaches now
theorem total_peaches : initial_peaches + picked_peaches = 68 := by
  sorry

end total_peaches_l96_96848


namespace BANANA_permutations_l96_96208

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96208


namespace number_of_arrangements_l96_96253

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

end number_of_arrangements_l96_96253


namespace banana_arrangements_l96_96447

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96447


namespace banana_permutations_l96_96466

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96466


namespace BANANA_permutations_l96_96629

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96629


namespace banana_arrangements_l96_96674

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96674


namespace permutations_banana_l96_96639

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96639


namespace arrange_BANANA_l96_96056

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96056


namespace number_of_unique_permutations_BANANA_l96_96548

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96548


namespace arrange_BANANA_l96_96610

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96610


namespace number_of_arrangements_BANANA_l96_96763

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96763


namespace permutations_of_BANANA_l96_96702

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96702


namespace compare_neg_fractions_l96_96030

theorem compare_neg_fractions : (- (2/3 : ℚ)) > - (3/4 : ℚ) := by
  sorry

end compare_neg_fractions_l96_96030


namespace weight_difference_at_end_of_year_l96_96830

-- Conditions
def labrador_initial_weight : ℝ := 40
def dachshund_initial_weight : ℝ := 12
def weight_gain_percentage : ℝ := 0.25

-- Question: Difference in weight at the end of the year
theorem weight_difference_at_end_of_year : 
  let labrador_final_weight := labrador_initial_weight * (1 + weight_gain_percentage)
  let dachshund_final_weight := dachshund_initial_weight * (1 + weight_gain_percentage)
  labrador_final_weight - dachshund_final_weight = 35 :=
by
  sorry

end weight_difference_at_end_of_year_l96_96830


namespace arrangement_count_BANANA_l96_96220

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

end arrangement_count_BANANA_l96_96220


namespace number_of_arrangements_l96_96240

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

end number_of_arrangements_l96_96240


namespace numberOfWaysToArrangeBANANA_l96_96092

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96092


namespace BANANA_permutations_l96_96203

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96203


namespace gary_money_after_sale_l96_96989

theorem gary_money_after_sale :
  let initial_money := 73.0
  let sale_amount := 55.0
  initial_money + sale_amount = 128.0 :=
by
  let initial_money := 73.0
  let sale_amount := 55.0
  show initial_money + sale_amount = 128.0
  sorry

end gary_money_after_sale_l96_96989


namespace permutations_banana_l96_96641

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96641


namespace rectangle_area_l96_96864

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 :=
by
  sorry

end rectangle_area_l96_96864


namespace arrange_banana_l96_96148

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

end arrange_banana_l96_96148


namespace banana_permutations_l96_96498

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96498


namespace banana_permutations_l96_96468

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96468


namespace permutations_banana_l96_96646

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96646


namespace negative_values_count_l96_96984

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l96_96984


namespace jake_weight_l96_96824

variable (J K : ℕ)

-- Conditions given in the problem
axiom h1 : J - 8 = 2 * K
axiom h2 : J + K = 293

-- Statement to prove
theorem jake_weight : J = 198 :=
by
  sorry

end jake_weight_l96_96824


namespace binom_60_3_eq_34220_l96_96045

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l96_96045


namespace arrangement_count_BANANA_l96_96218

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

end arrangement_count_BANANA_l96_96218


namespace numberOfWaysToArrangeBANANA_l96_96079

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96079


namespace arrange_BANANA_l96_96528

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96528


namespace arrange_BANANA_l96_96608

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96608


namespace simplify_expression_l96_96021

variable (a b c : ℝ)

theorem simplify_expression :
  (-32 * a^4 * b^5 * c) / ((-2 * a * b)^3) * (-3 / 4 * a * c) = -3 * a^2 * b^2 * c^2 :=
  by
    sorry

end simplify_expression_l96_96021


namespace banana_arrangements_l96_96666

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96666


namespace permutations_of_BANANA_l96_96126

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

end permutations_of_BANANA_l96_96126


namespace banana_arrangements_l96_96437

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96437


namespace binom_60_3_eq_34220_l96_96033

theorem binom_60_3_eq_34220 : (nat.choose 60 3) = 34220 := sorry

end binom_60_3_eq_34220_l96_96033


namespace triangle_area_l96_96914

theorem triangle_area (a b c : ℕ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) 
  (h₄ : a^2 + b^2 = c^2) : (1 / 2 : ℝ) * a * b = 30 := 
by 
  rw [h₁, h₂]
  norm_num

end triangle_area_l96_96914


namespace arrange_BANANA_l96_96598

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96598


namespace count_negative_values_of_x_l96_96971

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l96_96971


namespace banana_arrangements_l96_96555

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96555


namespace permutations_BANANA_l96_96413

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96413


namespace banana_permutations_l96_96499

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96499


namespace banana_arrangements_l96_96269

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96269


namespace BANANA_permutations_l96_96619

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96619


namespace arrange_BANANA_l96_96606

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96606


namespace banana_arrangements_l96_96268

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96268


namespace banana_unique_permutations_l96_96750

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96750


namespace number_of_unique_permutations_BANANA_l96_96544

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96544


namespace binomial_60_3_eq_34220_l96_96042

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l96_96042


namespace find_solution_l96_96929

open Nat

def binomial (n : ℕ) (k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

noncomputable def expression (n : ℕ) : ℕ :=
  1 + binomial n 1 + binomial n 2 + binomial n 3

theorem find_solution (n : ℕ) (h : n > 3) :
  expression n ∣ 2 ^ 2000 ↔ n = 7 ∨ n = 23 :=
by
  sorry

end find_solution_l96_96929


namespace banana_arrangements_l96_96259

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96259


namespace banana_permutations_l96_96455

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96455


namespace compare_fractions_l96_96026

-- Define the fractions
def frac1 : ℚ := -2/3
def frac2 : ℚ := -3/4

-- Prove that -2/3 > -3/4
theorem compare_fractions : frac1 > frac2 :=
by {
  sorry
}

end compare_fractions_l96_96026


namespace number_of_unique_permutations_BANANA_l96_96550

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96550


namespace arrange_BANANA_l96_96605

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96605


namespace compute_area_ratio_l96_96008

noncomputable def area_ratio (K : ℝ) : ℝ :=
  let small_triangle_area := 2 * K
  let large_triangle_area := 8 * K
  small_triangle_area / large_triangle_area

theorem compute_area_ratio (K : ℝ) : area_ratio K = 1 / 4 :=
by
  unfold area_ratio
  sorry

end compute_area_ratio_l96_96008


namespace collinear_condition_l96_96908

theorem collinear_condition {a b c d : ℝ} (h₁ : a < b) (h₂ : c < d) (h₃ : a < d) (h₄ : c < b) :
  (a / d) + (c / b) = 1 := 
sorry

end collinear_condition_l96_96908


namespace no_positive_integer_solutions_l96_96788

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), (a > 0) ∧ (b > 0) → 3 * a^2 ≠ b^2 + 1 :=
by
  sorry

end no_positive_integer_solutions_l96_96788


namespace binom_60_3_eq_34220_l96_96052

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l96_96052


namespace x_intercept_of_translated_line_l96_96867

theorem x_intercept_of_translated_line :
  let line_translation (y : ℝ) := y + 4
  let new_line_eq := fun (x : ℝ) => 2 * x - 2
  new_line_eq 1 = 0 :=
by
  sorry

end x_intercept_of_translated_line_l96_96867


namespace fraction_to_terminating_decimal_l96_96782

theorem fraction_to_terminating_decimal :
  (47 / (2^3 * 5^4) : ℝ) = 0.0094 := by
  sorry

end fraction_to_terminating_decimal_l96_96782


namespace banana_arrangements_l96_96452

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96452


namespace binom_60_3_eq_34220_l96_96043

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l96_96043


namespace distinct_permutations_BANANA_l96_96288

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96288


namespace roots_of_quadratic_eq_l96_96871

theorem roots_of_quadratic_eq (a b : ℝ) (h1 : a * (-2)^2 + b * (-2) = 6) (h2 : a * 3^2 + b * 3 = 6) :
    ∃ (x1 x2 : ℝ), x1 = -2 ∧ x2 = 3 ∧ ∀ x, a * x^2 + b * x = 6 ↔ (x = x1 ∨ x = x2) :=
by
  use -2, 3
  sorry

end roots_of_quadratic_eq_l96_96871


namespace count_negative_values_of_x_l96_96969

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l96_96969


namespace money_distribution_l96_96012

-- Declare the variables and the conditions as hypotheses
theorem money_distribution (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 40) :
  B + C = 340 :=
by
  sorry

end money_distribution_l96_96012


namespace BANANA_arrangement_l96_96164

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96164


namespace BANANA_permutation_l96_96183

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96183


namespace banana_arrangements_l96_96559

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96559


namespace permutations_of_BANANA_l96_96713

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96713


namespace number_of_ways_to_arrange_BANANA_l96_96434

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96434


namespace arrangements_of_BANANA_l96_96583

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

end arrangements_of_BANANA_l96_96583


namespace number_of_arrangements_l96_96242

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

end number_of_arrangements_l96_96242


namespace num_ways_to_arrange_BANANA_l96_96347

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96347


namespace count_negative_x_with_sqrt_pos_int_l96_96958

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l96_96958


namespace value_of_expression_l96_96998

def f (x : ℝ) : ℝ := x^2 - 3*x + 7
def g (x : ℝ) : ℝ := x + 2

theorem value_of_expression : f (g 3) - g (f 3) = 8 :=
by
  sorry

end value_of_expression_l96_96998


namespace negative_values_count_l96_96986

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l96_96986


namespace num_ways_to_arrange_BANANA_l96_96352

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96352


namespace banana_arrangements_l96_96256

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96256


namespace count_negative_values_correct_l96_96964

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l96_96964


namespace banana_arrangements_l96_96441

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96441


namespace banana_permutations_l96_96500

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96500


namespace banana_permutations_l96_96462

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96462


namespace BANANA_arrangements_l96_96721

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

end BANANA_arrangements_l96_96721


namespace arrangements_of_BANANA_l96_96584

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

end arrangements_of_BANANA_l96_96584


namespace greatest_radius_l96_96827

theorem greatest_radius (r : ℕ) (h : π * (r : ℝ)^2 < 50 * π) : r = 7 :=
sorry

end greatest_radius_l96_96827


namespace numberOfWaysToArrangeBANANA_l96_96075

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96075


namespace number_of_arrangements_banana_l96_96390

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96390


namespace arrange_banana_l96_96144

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

end arrange_banana_l96_96144


namespace minimal_side_length_of_room_l96_96054

theorem minimal_side_length_of_room (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ S : ℕ, S = 10 :=
by {
  sorry
}

end minimal_side_length_of_room_l96_96054


namespace arrange_BANANA_l96_96516

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96516


namespace solve_root_equation_l96_96931

noncomputable def sqrt4 (x : ℝ) : ℝ := x^(1/4)

theorem solve_root_equation (x : ℝ) :
  sqrt4 (43 - 2 * x) + sqrt4 (39 + 2 * x) = 4 ↔ x = 21 ∨ x = -13.5 :=
by
  sorry

end solve_root_equation_l96_96931


namespace arrange_BANANA_l96_96074

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96074


namespace permutations_BANANA_l96_96313

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96313


namespace number_of_arrangements_BANANA_l96_96768

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96768


namespace BANANA_permutations_l96_96634

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96634


namespace permutations_banana_l96_96652

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96652


namespace cost_price_of_watch_l96_96892

theorem cost_price_of_watch (CP : ℝ) (h1 : SP1 = CP * 0.64) (h2 : SP2 = CP * 1.04) (h3 : SP2 = SP1 + 140) : CP = 350 :=
by
  sorry

end cost_price_of_watch_l96_96892


namespace permutations_of_BANANA_l96_96695

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96695


namespace time_juan_ran_l96_96841

variable (Distance Speed : ℝ)
variable (h1 : Distance = 80)
variable (h2 : Speed = 10)

theorem time_juan_ran : (Distance / Speed) = 8 := by
  sorry

end time_juan_ran_l96_96841


namespace banana_arrangements_l96_96561

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96561


namespace permutations_BANANA_l96_96367

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96367


namespace number_of_arrangements_BANANA_l96_96764

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96764


namespace permutations_BANANA_l96_96310

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96310


namespace banana_permutations_l96_96456

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96456


namespace permutations_banana_l96_96640

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96640


namespace permutations_BANANA_l96_96308

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96308


namespace BANANA_arrangements_correct_l96_96332

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96332


namespace permutations_of_BANANA_l96_96127

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

end permutations_of_BANANA_l96_96127


namespace arrange_BANANA_l96_96063

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96063


namespace banana_unique_permutations_l96_96739

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96739


namespace number_of_arrangements_banana_l96_96394

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96394


namespace numberOfWaysToArrangeBANANA_l96_96082

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96082


namespace tom_pays_1340_l96_96878

def vaccine_cost := 45
def number_of_vaccines := 10
def doctor_visit_cost := 250
def insurance_coverage := 0.8
def trip_cost := 1200

def total_vaccine_cost := vaccine_cost * number_of_vaccines
def total_medical_cost := total_vaccine_cost + doctor_visit_cost
def insurance_cover_amount := total_medical_cost * insurance_coverage
def amount_paid_after_insurance := total_medical_cost - insurance_cover_amount
def total_amount_tom_pays := amount_paid_after_insurance + trip_cost

theorem tom_pays_1340 :
  total_amount_tom_pays = 1340 :=
by
  sorry

end tom_pays_1340_l96_96878


namespace permutations_banana_l96_96642

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96642


namespace intersection_eq_l96_96921

namespace Proof

universe u

-- Define the natural number set M
def M : Set ℕ := { x | x > 0 ∧ x < 6 }

-- Define the set N based on the condition |x-1| ≤ 2
def N : Set ℝ := { x | abs (x - 1) ≤ 2 }

-- Define the complement of N with respect to the real numbers
def ComplementN : Set ℝ := { x | x < -1 ∨ x > 3 }

-- Define the intersection of M and the complement of N
def IntersectMCompN : Set ℕ := { x | x ∈ M ∧ (x : ℝ) ∈ ComplementN }

-- Provide the theorem to be proved
theorem intersection_eq : IntersectMCompN = { 4, 5 } :=
by
  sorry

end Proof

end intersection_eq_l96_96921


namespace banana_arrangements_l96_96260

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96260


namespace BANANA_permutations_l96_96618

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96618


namespace number_of_ways_to_arrange_BANANA_l96_96430

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96430


namespace simplify_expression_l96_96799

theorem simplify_expression :
  ((2 + 3 + 4 + 5) / 2) + ((2 * 5 + 8) / 3) = 13 :=
by
  sorry

end simplify_expression_l96_96799


namespace banana_arrangements_l96_96690

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96690


namespace distinct_permutations_BANANA_l96_96281

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96281


namespace arrange_banana_l96_96150

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

end arrange_banana_l96_96150


namespace permutations_of_BANANA_l96_96710

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96710


namespace number_of_arrangements_l96_96237

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

end number_of_arrangements_l96_96237


namespace arrangement_count_BANANA_l96_96475

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96475


namespace average_price_over_3_months_l96_96896

theorem average_price_over_3_months (dMay : ℕ) 
  (pApril pMay pJune : ℝ) 
  (h1 : pApril = 1.20) 
  (h2 : pMay = 1.20) 
  (h3 : pJune = 3.00) 
  (h4 : dApril = 2 / 3 * dMay) 
  (h5 : dJune = 2 * dApril) :
  ((dApril * pApril + dMay * pMay + dJune * pJune) / (dApril + dMay + dJune) = 2) := 
by sorry

end average_price_over_3_months_l96_96896


namespace banana_arrangements_l96_96436

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96436


namespace arrange_BANANA_l96_96614

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96614


namespace binom_60_3_eq_34220_l96_96044

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end binom_60_3_eq_34220_l96_96044


namespace numberOfWaysToArrangeBANANA_l96_96084

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96084


namespace banana_permutations_l96_96512

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96512


namespace banana_arrangements_l96_96691

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96691


namespace BANANA_arrangements_correct_l96_96323

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96323


namespace arrange_BANANA_l96_96517

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96517


namespace BANANA_permutation_l96_96175

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96175


namespace BANANA_permutations_l96_96213

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96213


namespace banana_unique_permutations_l96_96747

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96747


namespace number_of_arrangements_banana_l96_96385

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96385


namespace permutations_BANANA_l96_96401

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96401


namespace arrange_BANANA_l96_96521

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96521


namespace arrangement_count_BANANA_l96_96480

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96480


namespace total_animals_seen_correct_l96_96019

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

end total_animals_seen_correct_l96_96019


namespace contractor_days_l96_96003

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

end contractor_days_l96_96003


namespace number_of_arrangements_banana_l96_96378

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96378


namespace banana_arrangements_l96_96683

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96683


namespace negative_values_count_l96_96942

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l96_96942


namespace BANANA_arrangement_l96_96166

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96166


namespace banana_arrangements_l96_96665

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96665


namespace number_of_arrangements_l96_96248

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

end number_of_arrangements_l96_96248


namespace BANANA_arrangements_l96_96733

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

end BANANA_arrangements_l96_96733


namespace permutations_BANANA_l96_96357

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96357


namespace distinct_permutations_BANANA_l96_96284

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96284


namespace BANANA_arrangements_l96_96717

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

end BANANA_arrangements_l96_96717


namespace num_ways_to_arrange_BANANA_l96_96343

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96343


namespace compare_fractions_neg_l96_96024

theorem compare_fractions_neg : (- (2 / 3) > - (3 / 4)) :=
  sorry

end compare_fractions_neg_l96_96024


namespace permutations_banana_l96_96649

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96649


namespace height_of_larger_triangle_l96_96887

-- Definitions from the conditions
variables (height_small height_large : ℝ)
variables (area_ratio : ℝ)
variables (k : ℝ)

-- Given conditions
def triangles_similar : Prop := area_ratio = 9
def height_small_defined : Prop := height_small = 5
def scale_factor : Prop := k = real.sqrt area_ratio

-- Proof problem statement
theorem height_of_larger_triangle
  (h_similar : triangles_similar)
  (h_height_small : height_small_defined)
  (h_scale_factor : scale_factor) :
  height_large = 15 := sorry

end height_of_larger_triangle_l96_96887


namespace permutations_BANANA_l96_96356

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96356


namespace count_negative_x_with_sqrt_pos_int_l96_96960

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l96_96960


namespace number_of_arrangements_l96_96247

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

end number_of_arrangements_l96_96247


namespace permutations_BANANA_l96_96366

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96366


namespace BANANA_permutations_l96_96211

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96211


namespace permutations_of_BANANA_l96_96123

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

end permutations_of_BANANA_l96_96123


namespace permutations_of_BANANA_l96_96119

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

end permutations_of_BANANA_l96_96119


namespace BANANA_arrangements_l96_96732

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

end BANANA_arrangements_l96_96732


namespace permutations_of_BANANA_l96_96696

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96696


namespace binom_60_3_eq_34220_l96_96048

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l96_96048


namespace polynomial_identity_l96_96775

theorem polynomial_identity (x : ℝ) :
  (x - 2)^5 + 5 * (x - 2)^4 + 10 * (x - 2)^3 + 10 * (x - 2)^2 + 5 * (x - 2) + 1 = (x - 1)^5 := 
by 
  sorry

end polynomial_identity_l96_96775


namespace permutations_banana_l96_96654

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96654


namespace count_negative_x_with_sqrt_pos_int_l96_96957

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l96_96957


namespace arrange_BANANA_l96_96603

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96603


namespace BANANA_permutations_l96_96622

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96622


namespace BANANA_permutation_l96_96191

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96191


namespace sandy_correct_sums_l96_96850

theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 45) : c = 21 :=
  sorry

end sandy_correct_sums_l96_96850


namespace student_weight_loss_l96_96913

variables (S R L : ℕ)

theorem student_weight_loss :
  S = 75 ∧ S + R = 110 ∧ S - L = 2 * R → L = 5 :=
by
  sorry

end student_weight_loss_l96_96913


namespace number_of_arrangements_banana_l96_96389

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96389


namespace number_of_arrangements_BANANA_l96_96774

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96774


namespace find_third_month_sale_l96_96902

theorem find_third_month_sale
  (sale_1 sale_2 sale_3 sale_4 sale_5 sale_6 : ℕ)
  (h1 : sale_1 = 800)
  (h2 : sale_2 = 900)
  (h4 : sale_4 = 700)
  (h5 : sale_5 = 800)
  (h6 : sale_6 = 900)
  (h_avg : (sale_1 + sale_2 + sale_3 + sale_4 + sale_5 + sale_6) / 6 = 850) : 
  sale_3 = 1000 :=
by
  sorry

end find_third_month_sale_l96_96902


namespace percentage_error_in_side_length_l96_96915

theorem percentage_error_in_side_length 
  (A A' s s' : ℝ) (h₁ : A = s^2)
  (h₂ : A' = s'^2)
  (h₃ : ((A' - A) / A * 100) = 12.36) :
  ∃ E : ℝ, (s' = s * (1 + E / 100)) ∧ (E = (real.sqrt 1.1236 - 1) * 100) := 
by
  sorry

end percentage_error_in_side_length_l96_96915


namespace cos_alpha_value_l96_96837

theorem cos_alpha_value (θ α : Real) (P : Real × Real)
  (hP : P = (-3/5, 4/5))
  (hθ : θ = Real.arccos (-3/5))
  (hαθ : α = θ - Real.pi / 3) :
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 := 
by 
  sorry

end cos_alpha_value_l96_96837


namespace BANANA_permutations_l96_96202

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96202


namespace permutations_BANANA_l96_96404

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96404


namespace distinct_permutations_BANANA_l96_96289

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96289


namespace count_negative_values_of_x_l96_96973

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l96_96973


namespace solve_x_l96_96997

theorem solve_x (x : ℝ) (h : (x - 1)^2 = 4) : x = 3 ∨ x = -1 :=
by
  -- proof goes here
  sorry

end solve_x_l96_96997


namespace distinct_permutations_BANANA_l96_96279

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96279


namespace BANANA_arrangements_l96_96715

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

end BANANA_arrangements_l96_96715


namespace BANANA_permutation_l96_96180

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96180


namespace number_of_ways_to_arrange_BANANA_l96_96417

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96417


namespace arrangement_count_BANANA_l96_96219

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

end arrangement_count_BANANA_l96_96219


namespace fraction_to_decimal_terminating_l96_96776

theorem fraction_to_decimal_terminating : 
  (47 / (2^3 * 5^4) : ℝ) = 0.5875 :=
by 
  sorry

end fraction_to_decimal_terminating_l96_96776


namespace BANANA_arrangement_l96_96167

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96167


namespace BANANA_arrangements_l96_96728

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

end BANANA_arrangements_l96_96728


namespace permutations_BANANA_l96_96370

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96370


namespace Daniel_had_more_than_200_marbles_at_day_6_l96_96922

noncomputable def marbles (k : ℕ) : ℕ :=
  5 * 2^k

theorem Daniel_had_more_than_200_marbles_at_day_6 :
  ∃ k : ℕ, marbles k > 200 ∧ ∀ m < k, marbles m ≤ 200 :=
by
  sorry

end Daniel_had_more_than_200_marbles_at_day_6_l96_96922


namespace BANANA_permutations_l96_96631

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96631


namespace number_of_arrangements_l96_96244

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

end number_of_arrangements_l96_96244


namespace numberOfWaysToArrangeBANANA_l96_96076

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96076


namespace grandma_can_give_cherry_exists_better_grand_strategy_l96_96836

variable (Packet1 : Finset String) (Packet2 : Finset String) (Packet3 : Finset String)
variable (isCabbage : String → Prop) (isCherry : String → Prop)
variable (wholePie : String → Prop)

-- Conditions
axiom Packet1_cond : ∀ p ∈ Packet1, isCabbage p
axiom Packet2_cond : ∀ p ∈ Packet2, isCherry p
axiom Packet3_cond_cabbage : ∃ p ∈ Packet3, isCabbage p
axiom Packet3_cond_cherry : ∃ p ∈ Packet3, isCherry p

-- Question (a)
theorem grandma_can_give_cherry (h1 : ∃ p1 ∈ Packet3, wholePie p1 ∧ isCherry p1 ∨
    ∃ p2 ∈ Packet1, wholePie p2 ∧ (∃ q ∈ Packet2 ∪ Packet3, isCherry q ∧ wholePie q) ∨
    ∃ p3 ∈ Packet2, wholePie p3 ∧ isCherry p3) :
  ∃ grand_strategy, grand_strategy = (2 / 3) * (1 : ℝ) :=
by
  sorry

-- Question (b)
theorem exists_better_grand_strategy (h2 : ∃ p ∈ Packet3, wholePie p ∧ isCherry p ∨
    ∃ p2 ∈ Packet1, wholePie p2 ∧ (∃ q ∈ Packet2 ∪ Packet3, isCherry q ∧ wholePie q) ∨
    ∃ p3 ∈ Packet2, wholePie p3 ∧ isCherry p3) :
  ∃ grand_strategy, grand_strategy > (2 / 3) * (1 : ℝ) :=
by
  sorry

end grandma_can_give_cherry_exists_better_grand_strategy_l96_96836


namespace find_point_coordinates_l96_96996

theorem find_point_coordinates :
  ∃ P : ℝ × ℝ × ℝ, 
    P = (0, P.2, 0) ∧
    (let A := (2, -1, 4) in let B := (-1, 2, 5) in 
    real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2 + (A.3 - P.3)^2) = 
    real.sqrt ((B.1 - P.1)^2 + (B.2 - P.2)^2 + (B.3 - P.3)^2)) ∧ 
    P = (0, 3/2, 0) :=
sorry

end find_point_coordinates_l96_96996


namespace complex_division_l96_96810

theorem complex_division (z1 z2 : ℂ) (h1 : z1 = 1 + 1 * Complex.I) (h2 : z2 = 0 + 2 * Complex.I) :
  z2 / z1 = 1 + Complex.I :=
by
  sorry

end complex_division_l96_96810


namespace number_of_ways_to_arrange_BANANA_l96_96428

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96428


namespace banana_arrangements_l96_96664

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96664


namespace arrange_BANANA_l96_96064

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96064


namespace fraction_to_decimal_terminating_l96_96777

theorem fraction_to_decimal_terminating : 
  (47 / (2^3 * 5^4) : ℝ) = 0.5875 :=
by 
  sorry

end fraction_to_decimal_terminating_l96_96777


namespace banana_arrangements_l96_96675

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96675


namespace BANANA_permutation_l96_96181

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96181


namespace number_of_arrangements_banana_l96_96375

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96375


namespace arrange_BANANA_l96_96520

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96520


namespace number_of_unique_permutations_BANANA_l96_96553

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96553


namespace BANANA_permutations_l96_96630

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96630


namespace solve_for_y_l96_96855

theorem solve_for_y (y : ℝ) (h : (4/7) * (1/5) * y - 2 = 14) : y = 140 := 
sorry

end solve_for_y_l96_96855


namespace negative_values_of_x_l96_96954

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l96_96954


namespace banana_arrangements_l96_96104

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96104


namespace range_of_a_l96_96862

   noncomputable section

   variable {f : ℝ → ℝ}

   /-- The requried theorem based on the given conditions and the correct answer -/
   theorem range_of_a (even_f : ∀ x, f (-x) = f x)
                      (increasing_f : ∀ x y, x ≤ y → y ≤ 0 → f x ≤ f y)
                      (h : f a ≤ f 2) : a ≤ -2 ∨ a ≥ 2 :=
   sorry
   
end range_of_a_l96_96862


namespace banana_arrangements_l96_96679

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96679


namespace arrange_BANANA_l96_96061

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96061


namespace number_of_arrangements_banana_l96_96386

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96386


namespace BANANA_permutation_l96_96190

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96190


namespace banana_arrangements_l96_96274

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96274


namespace BANANA_permutation_l96_96179

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96179


namespace ab_value_l96_96893

theorem ab_value (a b : ℤ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) : a * b = 7 := by
  sorry

end ab_value_l96_96893


namespace arrangement_count_BANANA_l96_96481

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96481


namespace arrange_banana_l96_96140

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

end arrange_banana_l96_96140


namespace permutations_of_BANANA_l96_96714

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96714


namespace range_of_function_l96_96868

theorem range_of_function :
  ∃ (S : Set ℝ), (∀ x : ℝ, (1 / 2)^(x^2 - 2) ∈ S) ∧ S = Set.Ioc 0 4 := by
  sorry

end range_of_function_l96_96868


namespace negative_values_count_l96_96941

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l96_96941


namespace BANANA_permutations_l96_96205

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96205


namespace permutations_of_BANANA_l96_96132

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

end permutations_of_BANANA_l96_96132


namespace BANANA_arrangements_l96_96729

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

end BANANA_arrangements_l96_96729


namespace average_percent_score_l96_96846

theorem average_percent_score :
    let students := 120
    let score_95 := 95 * 12
    let score_85 := 85 * 24
    let score_75 := 75 * 30
    let score_65 := 65 * 20
    let score_55 := 55 * 18
    let score_45 := 45 * 10
    let score_35 := 35 * 6
    let total_score := score_95 + score_85 + score_75 + score_65 + score_55 + score_45 + score_35
    (total_score.toFloat / students.toFloat) = 69.8333 :=
by
  sorry

end average_percent_score_l96_96846


namespace arrangements_of_BANANA_l96_96580

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

end arrangements_of_BANANA_l96_96580


namespace number_of_ways_to_arrange_BANANA_l96_96432

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96432


namespace permutations_BANANA_l96_96364

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96364


namespace permutations_BANANA_l96_96410

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96410


namespace kaleb_balance_l96_96897

theorem kaleb_balance (springEarnings : ℕ) (summerEarnings : ℕ) (suppliesCost : ℕ) (totalBalance : ℕ)
  (h1 : springEarnings = 4)
  (h2 : summerEarnings = 50)
  (h3 : suppliesCost = 4)
  (h4 : totalBalance = (springEarnings + summerEarnings) - suppliesCost) : totalBalance = 50 := by
  sorry

end kaleb_balance_l96_96897


namespace banana_arrangements_l96_96100

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96100


namespace number_of_ways_to_arrange_BANANA_l96_96423

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96423


namespace permutations_BANANA_l96_96314

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96314


namespace arrange_BANANA_l96_96071

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96071


namespace number_of_negative_x_l96_96947

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l96_96947


namespace num_ways_to_arrange_BANANA_l96_96344

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96344


namespace negative_values_count_l96_96944

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l96_96944


namespace banana_permutations_l96_96473

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96473


namespace arrange_banana_l96_96147

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

end arrange_banana_l96_96147


namespace permutations_of_BANANA_l96_96131

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

end permutations_of_BANANA_l96_96131


namespace banana_arrangements_l96_96110

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96110


namespace arrange_BANANA_l96_96532

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96532


namespace probability_prime_or_square_sum_l96_96880

open Nat

theorem probability_prime_or_square_sum : 
  let primes := [2, 3, 5, 7, 11] in
  let perfect_squares := [4, 9] in
  (∃ (a b : ℕ), a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} 
    ∧ (a + b) ∈ (primes ∪ perfect_squares)) →
  (22 / 36 = 11 / 18) :=
begin
  sorry
end

end probability_prime_or_square_sum_l96_96880


namespace find_a_plus_b_l96_96822

theorem find_a_plus_b (x a b : ℝ) (hx : x = a + real.sqrt b)
  (h : x ^ 2 + 5 * x + 5 / x + 1 / x ^ 2 = 40)
  (ha : a ∈ ℤ)
  (hb : b ∈ ℤ)
  (ha_pos : a > 0)
  (hb_pos : b > 0) :
  a + b = 11 :=
sorry

end find_a_plus_b_l96_96822


namespace tunnel_depth_l96_96860

theorem tunnel_depth (topWidth : ℝ) (bottomWidth : ℝ) (area : ℝ) (h : ℝ)
  (h1 : topWidth = 15)
  (h2 : bottomWidth = 5)
  (h3 : area = 400)
  (h4 : area = (1 / 2) * (topWidth + bottomWidth) * h) :
  h = 40 := 
sorry

end tunnel_depth_l96_96860


namespace number_of_arrangements_banana_l96_96376

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96376


namespace permutations_BANANA_l96_96305

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96305


namespace banana_arrangements_l96_96557

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96557


namespace similar_triangles_height_l96_96886

theorem similar_triangles_height
  (a b : ℕ)
  (area_ratio: ℕ)
  (height_smaller : ℕ)
  (height_relation: height_smaller = 5)
  (area_relation: area_ratio = 9)
  (similarity: a / b = 1 / area_ratio):
  (∃ height_larger : ℕ, height_larger = 15) :=
by
  sorry

end similar_triangles_height_l96_96886


namespace banana_arrangements_l96_96658

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96658


namespace number_of_arrangements_banana_l96_96387

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96387


namespace permutations_BANANA_l96_96409

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96409


namespace largest_n_for_factoring_l96_96932

theorem largest_n_for_factoring :
  ∃ (n : ℤ), (∀ (A B : ℤ), (3 * A + B = n) → (3 * A * B = 90) → n = 271) :=
by sorry

end largest_n_for_factoring_l96_96932


namespace count_negative_values_correct_l96_96968

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l96_96968


namespace sandy_puppies_l96_96849

theorem sandy_puppies :
  ∀ (initial_puppies puppies_given_away remaining_puppies : ℕ),
  initial_puppies = 8 →
  puppies_given_away = 4 →
  remaining_puppies = initial_puppies - puppies_given_away →
  remaining_puppies = 4 :=
by
  intros initial_puppies puppies_given_away remaining_puppies
  intro h_initial
  intro h_given_away
  intro h_remaining
  rw [h_initial, h_given_away] at h_remaining
  exact h_remaining

end sandy_puppies_l96_96849


namespace BANANA_permutations_l96_96633

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96633


namespace number_of_arrangements_l96_96254

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

end number_of_arrangements_l96_96254


namespace number_of_arrangements_banana_l96_96383

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96383


namespace banana_arrangements_l96_96677

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96677


namespace arrangements_of_BANANA_l96_96592

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

end arrangements_of_BANANA_l96_96592


namespace permutations_banana_l96_96645

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96645


namespace count_negative_values_correct_l96_96967

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l96_96967


namespace solve_quartic_eqn_l96_96930

noncomputable def solutionSet : Set ℂ :=
  {x | x^2 = 6 ∨ x^2 = -6}

theorem solve_quartic_eqn (x : ℂ) : (x^4 - 36 = 0) ↔ (x ∈ solutionSet) := 
sorry

end solve_quartic_eqn_l96_96930


namespace arrange_banana_l96_96142

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

end arrange_banana_l96_96142


namespace banana_arrangements_l96_96444

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96444


namespace arrangement_count_BANANA_l96_96222

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

end arrangement_count_BANANA_l96_96222


namespace banana_arrangements_l96_96271

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96271


namespace number_of_ways_to_arrange_BANANA_l96_96422

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96422


namespace distinct_permutations_BANANA_l96_96278

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96278


namespace arrangements_of_BANANA_l96_96586

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

end arrangements_of_BANANA_l96_96586


namespace binom_60_3_eq_34220_l96_96051

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l96_96051


namespace banana_arrangements_l96_96662

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96662


namespace sum_of_p_q_r_s_t_l96_96843

theorem sum_of_p_q_r_s_t (p q r s t : ℤ) (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t)
  (h_product : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = 120) : 
  p + q + r + s + t = 32 := 
sorry

end sum_of_p_q_r_s_t_l96_96843


namespace banana_unique_permutations_l96_96736

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96736


namespace arrange_banana_l96_96136

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

end arrange_banana_l96_96136


namespace fifth_term_power_of_five_sequence_l96_96053

theorem fifth_term_power_of_five_sequence : 5^0 + 5^1 + 5^2 + 5^3 + 5^4 = 781 := 
by
sorry

end fifth_term_power_of_five_sequence_l96_96053


namespace arrange_banana_l96_96146

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

end arrange_banana_l96_96146


namespace compare_fractions_neg_l96_96025

theorem compare_fractions_neg : (- (2 / 3) > - (3 / 4)) :=
  sorry

end compare_fractions_neg_l96_96025


namespace arrangement_count_BANANA_l96_96216

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

end arrangement_count_BANANA_l96_96216


namespace banana_arrangements_l96_96114

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96114


namespace BANANA_arrangement_l96_96157

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96157


namespace number_of_arrangements_BANANA_l96_96773

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96773


namespace permutations_of_BANANA_l96_96698

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96698


namespace BANANA_arrangements_correct_l96_96334

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96334


namespace permutations_of_BANANA_l96_96115

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

end permutations_of_BANANA_l96_96115


namespace BANANA_permutations_l96_96201

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96201


namespace find_certain_number_l96_96901

theorem find_certain_number (D S X : ℕ): 
  D = 20 → 
  S = 55 → 
  X + (D - S) = 3 * D - 90 →
  X = 5 := 
by
  sorry

end find_certain_number_l96_96901


namespace daniel_pages_to_read_l96_96834

-- Definitions from conditions
def total_pages : ℕ := 980
def daniel_read_time_per_page : ℕ := 50
def emma_read_time_per_page : ℕ := 40

-- The theorem that states the solution
theorem daniel_pages_to_read (d : ℕ) :
  d = 436 ↔ daniel_read_time_per_page * d = emma_read_time_per_page * (total_pages - d) :=
by sorry

end daniel_pages_to_read_l96_96834


namespace number_of_unique_permutations_BANANA_l96_96554

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96554


namespace BANANA_permutation_l96_96182

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96182


namespace number_of_unique_permutations_BANANA_l96_96540

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96540


namespace qualified_light_bulb_from_factory_A_l96_96838

variable (P : Set (Set ℝ))
variable (A : Prop)
variable (B : Prop)

-- Given conditions
axiom prob_A : P A = 0.7
axiom prob_B_given_A : P (B ∩ A) / P A = 0.95

-- The theorem to be proven
theorem qualified_light_bulb_from_factory_A :
  P (B ∩ A) = 0.665 := by
  sorry

end qualified_light_bulb_from_factory_A_l96_96838


namespace banana_arrangements_l96_96262

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96262


namespace banana_arrangements_l96_96560

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96560


namespace arrangement_count_BANANA_l96_96230

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

end arrangement_count_BANANA_l96_96230


namespace no_positive_integers_satisfy_l96_96787

theorem no_positive_integers_satisfy (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ¬ (3 * a^2 = b^2 + 1) := 
sorry

end no_positive_integers_satisfy_l96_96787


namespace arrange_BANANA_l96_96070

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96070


namespace arrange_BANANA_l96_96529

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96529


namespace BANANA_arrangements_l96_96720

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

end BANANA_arrangements_l96_96720


namespace compare_fractions_neg_l96_96023

theorem compare_fractions_neg : (- (2 / 3) > - (3 / 4)) :=
  sorry

end compare_fractions_neg_l96_96023


namespace arrange_BANANA_l96_96524

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96524


namespace arrange_BANANA_l96_96600

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96600


namespace banana_arrangements_l96_96670

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96670


namespace number_of_arrangements_l96_96246

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

end number_of_arrangements_l96_96246


namespace banana_arrangements_l96_96571

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96571


namespace permutations_BANANA_l96_96297

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96297


namespace arrangement_count_BANANA_l96_96233

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

end arrangement_count_BANANA_l96_96233


namespace permutations_BANANA_l96_96406

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96406


namespace arrange_BANANA_l96_96068

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96068


namespace BANANA_arrangements_l96_96722

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

end BANANA_arrangements_l96_96722


namespace compare_fractions_l96_96028

-- Define the fractions
def frac1 : ℚ := -2/3
def frac2 : ℚ := -3/4

-- Prove that -2/3 > -3/4
theorem compare_fractions : frac1 > frac2 :=
by {
  sorry
}

end compare_fractions_l96_96028


namespace number_of_arrangements_l96_96250

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

end number_of_arrangements_l96_96250


namespace banana_arrangements_l96_96105

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96105


namespace arrangements_of_BANANA_l96_96591

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

end arrangements_of_BANANA_l96_96591


namespace find_k_l96_96808

theorem find_k (x₁ x₂ k : ℝ) (h1 : x₁ * x₁ - 6 * x₁ + k = 0) (h2 : x₂ * x₂ - 6 * x₂ + k = 0) (h3 : (1 / x₁) + (1 / x₂) = 3) :
  k = 2 :=
by
  sorry

end find_k_l96_96808


namespace banana_arrangements_l96_96096

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96096


namespace arrangement_count_BANANA_l96_96215

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

end arrangement_count_BANANA_l96_96215


namespace num_users_in_china_in_2022_l96_96013

def num_users_scientific (n : ℝ) : Prop :=
  n = 1.067 * 10^9

theorem num_users_in_china_in_2022 :
  num_users_scientific 1.067e9 :=
by
  sorry

end num_users_in_china_in_2022_l96_96013


namespace negative_values_of_x_l96_96953

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l96_96953


namespace distinct_permutations_BANANA_l96_96275

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96275


namespace largest_divisor_of_n4_minus_n_l96_96795

theorem largest_divisor_of_n4_minus_n (n : ℕ) (h : ¬(Prime n) ∧ n ≠ 1) : 6 ∣ (n^4 - n) :=
by sorry

end largest_divisor_of_n4_minus_n_l96_96795


namespace arrangements_of_BANANA_l96_96579

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

end arrangements_of_BANANA_l96_96579


namespace number_of_ways_to_arrange_BANANA_l96_96420

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96420


namespace permutations_banana_l96_96635

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96635


namespace permutations_BANANA_l96_96400

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96400


namespace number_of_arrangements_BANANA_l96_96766

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96766


namespace banana_arrangements_l96_96445

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96445


namespace arrange_BANANA_l96_96523

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96523


namespace banana_arrangements_l96_96655

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96655


namespace banana_arrangements_l96_96451

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96451


namespace num_ways_to_arrange_BANANA_l96_96336

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96336


namespace nuts_in_tree_l96_96873

def num_squirrels := 4
def num_nuts := 2

theorem nuts_in_tree :
  ∀ (S N : ℕ), S = num_squirrels → S - N = 2 → N = num_nuts :=
by
  intros S N hS hDiff
  sorry

end nuts_in_tree_l96_96873


namespace BANANA_arrangement_l96_96158

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96158


namespace compare_neg_fractions_l96_96031

theorem compare_neg_fractions : (- (2/3 : ℚ)) > - (3/4 : ℚ) := by
  sorry

end compare_neg_fractions_l96_96031


namespace number_of_arrangements_banana_l96_96381

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96381


namespace find_M_plus_N_l96_96819

theorem find_M_plus_N (M N : ℕ) (h1 : 3 / 5 = M / 30) (h2 : 3 / 5 = 90 / N) : M + N = 168 := 
by
  sorry

end find_M_plus_N_l96_96819


namespace permutations_BANANA_l96_96304

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96304


namespace banana_permutations_l96_96497

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96497


namespace permutations_of_BANANA_l96_96122

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

end permutations_of_BANANA_l96_96122


namespace BANANA_arrangements_correct_l96_96322

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96322


namespace rectangle_shaded_area_equal_l96_96917

theorem rectangle_shaded_area_equal {x : ℝ} :
  let total_area := 72
  let shaded_area := 24 + 6*x
  let non_shaded_area := total_area / 2
  shaded_area = non_shaded_area → x = 2 := 
by 
  intros h
  sorry

end rectangle_shaded_area_equal_l96_96917


namespace number_of_ways_to_arrange_BANANA_l96_96431

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96431


namespace BANANA_arrangements_l96_96731

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

end BANANA_arrangements_l96_96731


namespace BANANA_arrangements_correct_l96_96321

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96321


namespace permutations_of_BANANA_l96_96706

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96706


namespace banana_unique_permutations_l96_96748

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96748


namespace find_angle_l96_96829

theorem find_angle (A : ℝ) (h : 0 < A ∧ A < π) 
  (c : 4 * π * Real.sin A - 3 * Real.arccos (-1/2) = 0) :
  A = π / 6 ∨ A = 5 * π / 6 :=
sorry

end find_angle_l96_96829


namespace books_arrangement_l96_96835

noncomputable def num_ways_arrange_books : Nat :=
  3 * 2 * (Nat.factorial 6)

theorem books_arrangement :
  num_ways_arrange_books = 4320 :=
by
  sorry

end books_arrangement_l96_96835


namespace banana_permutations_l96_96457

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96457


namespace BANANA_arrangement_l96_96162

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96162


namespace permutations_BANANA_l96_96403

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96403


namespace arrange_BANANA_l96_96518

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96518


namespace BANANA_permutations_l96_96616

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96616


namespace work_completion_days_l96_96890

theorem work_completion_days (h1 : (1:ℝ)/4 = (1:ℝ)/12 + (1:ℝ)/x) : 
  x = 6 :=
by sorry

end work_completion_days_l96_96890


namespace banana_arrangements_l96_96671

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96671


namespace banana_permutations_l96_96506

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96506


namespace BANANA_arrangement_l96_96159

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96159


namespace unique_solution_l96_96934

-- Given conditions in the problem:
def prime (p : ℕ) : Prop := Nat.Prime p
def is_solution (p n k : ℕ) : Prop :=
  3 ^ p + 4 ^ p = n ^ k ∧ k > 1 ∧ prime p

-- The only solution:
theorem unique_solution (p n k : ℕ) :
  is_solution p n k → (p, n, k) = (2, 5, 2) := 
by
  sorry

end unique_solution_l96_96934


namespace banana_unique_permutations_l96_96745

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96745


namespace number_of_unique_permutations_BANANA_l96_96551

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96551


namespace arrangements_of_BANANA_l96_96585

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

end arrangements_of_BANANA_l96_96585


namespace number_of_arrangements_BANANA_l96_96772

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96772


namespace probability_of_one_common_point_l96_96994

def f1 (x : ℝ) : ℝ := -x
def f2 (x : ℝ) : ℝ := -1 / x
def f3 (x : ℝ) : ℝ := x^3
def f4 (x : ℝ) : ℝ := x^(1 / 2)

noncomputable def event_probability : ℚ :=
  let functions := [f1, f2, f3, f4]
  let pairs := functions.product functions |>.filter (λ pair, pair.1 ≠ pair.2)
  let valid_pairs := pairs.filter (λ pair, (∃ x : ℝ, pair.1 x = pair.2 x) ∧
    (¬∃ x y : ℝ, x ≠ y ∧ pair.1 x = pair.2 x ∧ pair.1 y = pair.2 y))
  ⟨valid_pairs.length, pairs.length⟩

theorem probability_of_one_common_point :
    event_probability = 1 / 3 :=
by
  sorry

end probability_of_one_common_point_l96_96994


namespace banana_unique_permutations_l96_96742

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96742


namespace arrange_BANANA_l96_96609

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96609


namespace banana_permutations_l96_96464

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96464


namespace find_p_plus_q_l96_96017

noncomputable def probability_only_one (factor : ℕ → Prop) : ℚ := 0.08 -- Condition 1
noncomputable def probability_exaclty_two (factor1 factor2 : ℕ → Prop) : ℚ := 0.12 -- Condition 2
noncomputable def probability_all_three_given_two (factor1 factor2 factor3 : ℕ → Prop) : ℚ := 1 / 4 -- Condition 3
def women_without_D_has_no_risk_factors (total_women women_with_D women_with_all_factors women_without_D_no_risk_factors : ℕ) : ℚ :=
  women_without_D_no_risk_factors / (total_women - women_with_D)

theorem find_p_plus_q : ∃ (p q : ℕ), (women_without_D_has_no_risk_factors 100 (8 + 2 * 12 + 4) 4 28 = p / q) ∧ (Nat.gcd p q = 1) ∧ p + q = 23 :=
by
  sorry

end find_p_plus_q_l96_96017


namespace BANANA_arrangements_correct_l96_96327

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96327


namespace number_of_arrangements_banana_l96_96393

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96393


namespace permutations_BANANA_l96_96355

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96355


namespace banana_unique_permutations_l96_96753

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96753


namespace BANANA_arrangement_l96_96170

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96170


namespace identify_counterfeit_in_three_weighings_l96_96872

def CoinType := {x // x = "gold" ∨ x = "silver"}

structure Coins where
  golds: Fin 13
  silvers: Fin 14
  is_counterfeit: CoinType
  counterfeit_weight: Int

def is_lighter (c1 c2: Coins): Prop := sorry
def is_heavier (c1 c2: Coins): Prop := sorry
def balance (c1 c2: Coins): Prop := sorry

def find_counterfeit_coin (coins: Coins): Option Coins := sorry

theorem identify_counterfeit_in_three_weighings (coins: Coins) :
  ∃ (identify: Coins → Option Coins),
  ∀ coins, ( identify coins ≠ none ) :=
sorry

end identify_counterfeit_in_three_weighings_l96_96872


namespace banana_permutations_l96_96495

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96495


namespace count_negative_x_with_sqrt_pos_int_l96_96959

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l96_96959


namespace trig_identity_cosine_powers_l96_96898

theorem trig_identity_cosine_powers :
  12 * (Real.cos (Real.pi / 8)) ^ 4 + 
  (Real.cos (3 * Real.pi / 8)) ^ 4 + 
  (Real.cos (5 * Real.pi / 8)) ^ 4 + 
  (Real.cos (7 * Real.pi / 8)) ^ 4 = 
  3 / 2 := 
  sorry

end trig_identity_cosine_powers_l96_96898


namespace arrangement_count_BANANA_l96_96229

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

end arrangement_count_BANANA_l96_96229


namespace distinct_permutations_BANANA_l96_96293

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96293


namespace banana_arrangements_l96_96567

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96567


namespace count_negative_x_with_sqrt_pos_int_l96_96962

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l96_96962


namespace permutations_BANANA_l96_96360

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96360


namespace arrangement_count_BANANA_l96_96221

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

end arrangement_count_BANANA_l96_96221


namespace BANANA_permutations_l96_96209

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96209


namespace binomial_60_3_eq_34220_l96_96040

theorem binomial_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_eq_34220_l96_96040


namespace sum_of_products_mod_7_l96_96918

-- Define the numbers involved
def a := 1789
def b := 1861
def c := 1945
def d := 1533
def e := 1607
def f := 1688

-- Define the sum of products
def sum_of_products := a * b * c + d * e * f

-- The statement to prove:
theorem sum_of_products_mod_7 : sum_of_products % 7 = 3 := 
by sorry

end sum_of_products_mod_7_l96_96918


namespace joan_paid_230_l96_96895

theorem joan_paid_230 (J K : ℝ) (h1 : J + K = 600) (h2 : 2 * J = K + 90) : J = 230 := 
by 
  sorry

end joan_paid_230_l96_96895


namespace Danny_found_11_wrappers_l96_96923

theorem Danny_found_11_wrappers :
  ∃ wrappers_at_park : ℕ,
  (wrappers_at_park = 11) ∧
  (∃ bottle_caps : ℕ, bottle_caps = 12) ∧
  (∃ found_bottle_caps : ℕ, found_bottle_caps = 58) ∧
  (wrappers_at_park + 1 = bottle_caps) :=
by
  sorry

end Danny_found_11_wrappers_l96_96923


namespace man_l96_96904

noncomputable def speed_of_current : ℝ := 3 -- in kmph
noncomputable def time_to_cover_100_meters_downstream : ℝ := 19.99840012798976 -- in seconds
noncomputable def distance_covered : ℝ := 0.1 -- in kilometers (100 meters)

noncomputable def speed_in_still_water : ℝ :=
  (distance_covered / (time_to_cover_100_meters_downstream / 3600)) - speed_of_current

theorem man's_speed_in_still_water :
  speed_in_still_water = 14.9997120913593 :=
  by
    sorry

end man_l96_96904


namespace number_of_arrangements_BANANA_l96_96771

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96771


namespace arrange_BANANA_l96_96073

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96073


namespace curvilinear_quadrilateral_area_l96_96009

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

end curvilinear_quadrilateral_area_l96_96009


namespace arrangements_of_BANANA_l96_96582

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

end arrangements_of_BANANA_l96_96582


namespace permutations_of_BANANA_l96_96705

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96705


namespace parallelogram_area_l96_96790

theorem parallelogram_area (base height : ℝ) (h_base : base = 36) (h_height : height = 24) : 
    base * height = 864 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l96_96790


namespace num_ways_to_arrange_BANANA_l96_96345

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96345


namespace BANANA_permutation_l96_96186

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96186


namespace arrange_BANANA_l96_96612

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96612


namespace negative_values_count_l96_96985

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l96_96985


namespace num_ways_to_arrange_BANANA_l96_96339

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96339


namespace permutations_of_BANANA_l96_96116

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

end permutations_of_BANANA_l96_96116


namespace find_sin_double_angle_l96_96821

theorem find_sin_double_angle (θ : ℝ) (tanθ : ℝ) (h : tanθ = 1 / 3) : 
  Real.sin (2 * θ) = 3 / 5 :=
sorry

end find_sin_double_angle_l96_96821


namespace find_a_for_opposite_roots_l96_96811

-- Define the equation and condition using the given problem details
theorem find_a_for_opposite_roots (a : ℝ) 
  (h : ∀ (x : ℝ), x^2 - (a^2 - 2 * a - 15) * x + a - 1 = 0 
    → (∃! (x1 x2 : ℝ), x1 + x2 = 0)) :
  a = -3 := 
sorry

end find_a_for_opposite_roots_l96_96811


namespace number_of_ways_to_arrange_BANANA_l96_96425

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96425


namespace arrangement_count_BANANA_l96_96217

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

end arrangement_count_BANANA_l96_96217


namespace permutations_BANANA_l96_96309

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96309


namespace arrange_BANANA_l96_96595

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96595


namespace number_of_arrangements_BANANA_l96_96755

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96755


namespace number_of_negative_x_l96_96948

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l96_96948


namespace negative_values_count_l96_96940

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l96_96940


namespace result_is_0_85_l96_96869

noncomputable def calc_expression := 1.85 - 1.85 / 1.85

theorem result_is_0_85 : calc_expression = 0.85 :=
by 
  sorry

end result_is_0_85_l96_96869


namespace banana_arrangements_l96_96108

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96108


namespace similar_triangles_height_l96_96884

theorem similar_triangles_height (h₁ h₂ : ℝ) 
  (similar : ∀ (A₁ B₁ C₁ A₂ B₂ C₂ : Triangle), 
                (∃ k, k = 3 ∧ A₁ ≈ A₂ ∧ B₁ ≈ B₂ ∧ C₁ ≈ C₂) →
                (area A₁ / area A₂ = 1 / 9)) 
  (height_smaller : h₁ = 5)
  (area_ratio : area (Triangle.mk A₁ B₁ C₁) / area (Triangle.mk A₂ B₂ C₂) = 1 / 9) :
  h₂ = 15 := 
sorry

end similar_triangles_height_l96_96884


namespace BANANA_permutations_l96_96214

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96214


namespace supplements_delivered_l96_96845

-- Define the conditions as given in the problem
def total_medicine_boxes : ℕ := 760
def vitamin_boxes : ℕ := 472

-- Define the number of supplement boxes
def supplement_boxes : ℕ := total_medicine_boxes - vitamin_boxes

-- State the theorem to be proved
theorem supplements_delivered : supplement_boxes = 288 :=
by
  -- The actual proof is not required, so we use "sorry"
  sorry

end supplements_delivered_l96_96845


namespace BANANA_permutations_l96_96623

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96623


namespace number_of_arrangements_l96_96243

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

end number_of_arrangements_l96_96243


namespace BANANA_permutations_l96_96204

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96204


namespace carrie_first_day_miles_l96_96847

theorem carrie_first_day_miles
  (x : ℕ)
  (h1 : ∀ y : ℕ, y = x + 124) -- Second day
  (h2 : ∀ y : ℕ, y = 159) -- Third day
  (h3 : ∀ y : ℕ, y = 189) -- Fourth day
  (h4 : ∀ z : ℕ, z = 106) -- Phone charge interval
  (h5 : ∀ n : ℕ, n = 7) -- Number of charges
  (h_total : 106 * 7 = x + (x + 124) + 159 + 189)
  : x = 135 :=
by sorry

end carrie_first_day_miles_l96_96847


namespace num_of_negative_x_l96_96979

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l96_96979


namespace num_ways_to_arrange_BANANA_l96_96342

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96342


namespace banana_arrangements_l96_96676

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96676


namespace arrangement_count_BANANA_l96_96486

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96486


namespace negative_values_of_x_l96_96956

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l96_96956


namespace banana_arrangements_l96_96556

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96556


namespace num_ways_to_arrange_BANANA_l96_96346

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96346


namespace number_of_unique_permutations_BANANA_l96_96541

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96541


namespace remainder_when_divided_l96_96894

theorem remainder_when_divided (x : ℤ) (k : ℤ) (h: x = 82 * k + 5) : 
  ((x + 17) % 41) = 22 := by
  sorry

end remainder_when_divided_l96_96894


namespace banana_arrangements_l96_96672

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96672


namespace banana_arrangements_l96_96453

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96453


namespace arrange_BANANA_l96_96602

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96602


namespace num_of_negative_x_l96_96975

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l96_96975


namespace banana_arrangements_l96_96694

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96694


namespace banana_unique_permutations_l96_96737

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96737


namespace BANANA_arrangement_l96_96173

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96173


namespace permutations_banana_l96_96653

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96653


namespace number_of_unique_permutations_BANANA_l96_96536

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96536


namespace jenny_ate_65_chocolates_l96_96840

noncomputable def chocolates_eaten_by_Jenny : ℕ :=
  let chocolates_mike := 20
  let chocolates_john := chocolates_mike / 2
  let combined_chocolates := chocolates_mike + chocolates_john
  let twice_combined_chocolates := 2 * combined_chocolates
  5 + twice_combined_chocolates

theorem jenny_ate_65_chocolates :
  chocolates_eaten_by_Jenny = 65 :=
by
  -- Skipping the proof details
  sorry

end jenny_ate_65_chocolates_l96_96840


namespace distinct_permutations_BANANA_l96_96276

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96276


namespace banana_arrangements_l96_96680

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96680


namespace find_j_l96_96889

theorem find_j (n j : ℕ) (h1 : n % j = 28) (h2 : (n : ℝ) / j = 142.07) : j = 400 :=
by
  sorry

end find_j_l96_96889


namespace banana_permutations_l96_96496

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96496


namespace arrangement_count_BANANA_l96_96491

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96491


namespace numberOfWaysToArrangeBANANA_l96_96091

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96091


namespace permutations_of_BANANA_l96_96697

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96697


namespace locus_of_points_l96_96807

open EuclideanGeometry

-- Define the points A and B in 2D space.
variables (A B : EuclideanSpace ℝ (fin 2))

-- Define the perpendicular bisector of A and B.
def perp_bisector (A B : EuclideanSpace ℝ (fin 2)) : AffineSubspace ℝ (EuclideanSpace ℝ (fin 2)) :=
{ carrier := { M | dist A M = dist B M },
  direction := sorry }

-- Define the half-plane containing point B excluding the perpendicular bisector.
def half_plane_containing_B_excluding_bisector (A B : EuclideanSpace ℝ (fin 2)) : Set (EuclideanSpace ℝ (fin 2)) :=
{ M | dist A M > dist B M }

-- The main theorem statement.
theorem locus_of_points (A B : EuclideanSpace ℝ (fin 2)) :
  { M | dist A M > dist B M } = half_plane_containing_B_excluding_bisector A B :=
sorry

end locus_of_points_l96_96807


namespace number_of_arrangements_banana_l96_96391

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96391


namespace num_of_negative_x_l96_96980

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l96_96980


namespace permutations_BANANA_l96_96303

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96303


namespace bathroom_width_l96_96900

def length : ℝ := 4
def area : ℝ := 8
def width : ℝ := 2

theorem bathroom_width :
  area = length * width :=
by
  sorry

end bathroom_width_l96_96900


namespace distinct_permutations_BANANA_l96_96294

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96294


namespace banana_arrangements_l96_96099

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96099


namespace num_ways_to_arrange_BANANA_l96_96349

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96349


namespace banana_arrangements_l96_96450

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96450


namespace intersect_at_single_point_l96_96926

theorem intersect_at_single_point :
  (∃ (x y : ℝ), y = 3 * x + 5 ∧ y = -5 * x + 20 ∧ y = 4 * x + p) → p = 25 / 8 :=
by
  sorry

end intersect_at_single_point_l96_96926


namespace distinct_permutations_BANANA_l96_96285

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96285


namespace garden_fencing_needed_l96_96909

/-- Given a rectangular garden where the length is 300 yards and the length is twice the width,
prove that the total amount of fencing needed to enclose the garden is 900 yards. -/
theorem garden_fencing_needed :
  ∃ (W L P : ℝ), L = 300 ∧ L = 2 * W ∧ P = 2 * (L + W) ∧ P = 900 :=
by
  sorry

end garden_fencing_needed_l96_96909


namespace permutations_of_BANANA_l96_96134

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

end permutations_of_BANANA_l96_96134


namespace permutations_banana_l96_96648

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96648


namespace square_of_1027_l96_96919

theorem square_of_1027 :
  1027 * 1027 = 1054729 :=
by
  sorry

end square_of_1027_l96_96919


namespace number_of_negative_x_l96_96949

theorem number_of_negative_x :
  ∃ n, (∀ m : ℕ, m ≤ n ↔ m^2 < 200) ∧ n = 14 :=
by
  -- n = 14 is the largest integer such that n^2 < 200,
  -- and n ranges from 1 to 14.
  sorry

end number_of_negative_x_l96_96949


namespace BANANA_permutation_l96_96194

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96194


namespace arrangements_of_BANANA_l96_96589

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

end arrangements_of_BANANA_l96_96589


namespace banana_arrangements_l96_96113

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96113


namespace BANANA_arrangements_correct_l96_96326

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96326


namespace number_of_arrangements_banana_l96_96392

-- Define the number of ways to arrange the letters of the word BANANA
def uniqueArrangements := 
  let letters_total := 6
  let num_A := 3
  let num_N := 2
  nat.factorial letters_total / (nat.factorial num_A * nat.factorial num_N)

-- The theorem to prove that this number is 60
theorem number_of_arrangements_banana : uniqueArrangements = 60 := by
  sorry

end number_of_arrangements_banana_l96_96392


namespace num_of_negative_x_l96_96978

theorem num_of_negative_x (n : ℕ) (h : 1 ≤ n ∧ n ≤ 14) : 
    ∃ (x : ℤ), x < 0 ∧ x + 200 = n^2 :=
sorry

end num_of_negative_x_l96_96978


namespace banana_arrangements_l96_96102

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96102


namespace permutations_of_BANANA_l96_96128

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

end permutations_of_BANANA_l96_96128


namespace arrange_banana_l96_96139

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

end arrange_banana_l96_96139


namespace permutations_banana_l96_96650

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96650


namespace cone_volume_l96_96875

theorem cone_volume (p q : ℕ) (a α : ℝ) :
  V = (2 * π * a^3) / (3 * (Real.sin (2 * α)) * (Real.cos (180 * q / (p + q)))^2 * (Real.cos α)) :=
sorry

end cone_volume_l96_96875


namespace number_of_unique_permutations_BANANA_l96_96549

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96549


namespace number_of_arrangements_BANANA_l96_96757

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96757


namespace arrange_banana_l96_96152

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

end arrange_banana_l96_96152


namespace arrangement_count_BANANA_l96_96490

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96490


namespace banana_arrangements_l96_96669

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96669


namespace distinct_permutations_BANANA_l96_96292

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96292


namespace distinct_permutations_BANANA_l96_96287

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96287


namespace banana_arrangements_l96_96558

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96558


namespace arrangement_count_BANANA_l96_96485

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96485


namespace num_ways_to_arrange_BANANA_l96_96353

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96353


namespace compare_neg_fractions_l96_96029

theorem compare_neg_fractions : (- (2/3 : ℚ)) > - (3/4 : ℚ) := by
  sorry

end compare_neg_fractions_l96_96029


namespace number_of_ways_to_arrange_BANANA_l96_96416

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96416


namespace distinct_permutations_BANANA_l96_96283

open Nat

def word_length : Nat := 6
def count_A : Nat := 3
def count_N : Nat := 2

theorem distinct_permutations_BANANA : 
  (fact word_length) / (fact count_A * fact count_N) = 60 := 
sorry

end distinct_permutations_BANANA_l96_96283


namespace count_negative_values_correct_l96_96966

noncomputable def count_negative_values :=
  let count_values (n : ℕ) : ℕ := 
    if (n*n < 200) then n else n-1
  count_values 14

theorem count_negative_values_correct :
    count_negative_values = 14 := by
sorry

end count_negative_values_correct_l96_96966


namespace banana_arrangements_l96_96446

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96446


namespace permutations_of_BANANA_l96_96711

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96711


namespace arrange_BANANA_l96_96059

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96059


namespace permutations_banana_l96_96637

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96637


namespace banana_arrangements_l96_96454

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96454


namespace permutations_BANANA_l96_96395

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96395


namespace permutations_of_BANANA_l96_96120

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

end permutations_of_BANANA_l96_96120


namespace BANANA_arrangements_l96_96734

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

end BANANA_arrangements_l96_96734


namespace arrange_banana_l96_96141

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

end arrange_banana_l96_96141


namespace BANANA_permutation_l96_96193

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96193


namespace numberOfWaysToArrangeBANANA_l96_96083

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96083


namespace BANANA_arrangement_l96_96160

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96160


namespace BANANA_permutation_l96_96192

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96192


namespace no_positive_integers_satisfy_l96_96786

theorem no_positive_integers_satisfy (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ¬ (3 * a^2 = b^2 + 1) := 
sorry

end no_positive_integers_satisfy_l96_96786


namespace banana_arrangements_l96_96678

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96678


namespace negative_values_of_x_l96_96955

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l96_96955


namespace number_of_arrangements_l96_96252

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

end number_of_arrangements_l96_96252


namespace dog_cat_food_difference_l96_96907

theorem dog_cat_food_difference :
  let dogFood := 600
  let catFood := 327
  dogFood - catFood = 273 :=
by
  let dogFood := 600
  let catFood := 327
  show dogFood - catFood = 273
  sorry

end dog_cat_food_difference_l96_96907


namespace negative_values_count_l96_96939

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l96_96939


namespace area_of_rectangle_l96_96866

-- Definitions of the conditions
def length (w : ℝ) : ℝ := 4 * w
def perimeter_eq_200 (w l : ℝ) : Prop := 2 * l + 2 * w = 200

-- Main theorem statement
theorem area_of_rectangle (w l : ℝ) (h1 : length w = l) (h2 : perimeter_eq_200 w l) : l * w = 1600 :=
by
  -- Skip the proof
  sorry

end area_of_rectangle_l96_96866


namespace banana_arrangements_l96_96098

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96098


namespace remainder_when_divided_by_r_minus_2_l96_96935

-- Define polynomial p(r)
def p (r : ℝ) : ℝ := r ^ 11 - 3

-- The theorem stating the problem
theorem remainder_when_divided_by_r_minus_2 : p 2 = 2045 := by
  sorry

end remainder_when_divided_by_r_minus_2_l96_96935


namespace banana_unique_permutations_l96_96735

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96735


namespace permutations_BANANA_l96_96372

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96372


namespace arrange_banana_l96_96154

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

end arrange_banana_l96_96154


namespace permutations_BANANA_l96_96363

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96363


namespace BANANA_permutations_l96_96210

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96210


namespace friends_meet_first_time_at_4pm_l96_96797

def lcm_four_times (a b c d : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

def first_meeting_time (start_time_minutes: ℕ) (lap_anna lap_stephanie lap_james lap_carlos: ℕ) : ℕ :=
  start_time_minutes + lcm_four_times lap_anna lap_stephanie lap_james lap_carlos

theorem friends_meet_first_time_at_4pm :
  first_meeting_time 600 5 8 9 12 = 960 :=
by
  -- where 600 represents 10:00 AM in minutes since midnight and 960 represents 4:00 PM
  sorry

end friends_meet_first_time_at_4pm_l96_96797


namespace banana_arrangements_l96_96107

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96107


namespace g_monotonicity_intervals_f_decreasing_min_a_range_a_l96_96814

open Set

noncomputable def g (x : ℝ) := x / (Real.log x)

noncomputable def f (x a : ℝ) := g x - a * x

-- Problem (I): Proving intervals of monotonicity for g(x)
theorem g_monotonicity_intervals :
  (∀ x ∈ Ioi (Real.exp 1), 0 < (g x - g (Real.exp 1))) ∧ 
  (∀ x ∈ Iio (Real.exp 1), x ≠ 1 → g (Real.exp 1) > g x ∨ g x > g 1) := sorry

-- Problem (II): Prove minimum value of a for f(x) decreasing on (1, +∞)
theorem f_decreasing_min_a (a : ℝ) :
  (∀ x ∈ Ioi 1, f x a < f 1 a) ↔ a ≥ 1/4 := sorry

-- Problem (III): Range of a given g(x1) ≤ f'(x2) + 2a condition
theorem range_a (a : ℝ) :
  (∀ x1 ∈ Icc (Real.exp 1) (Real.exp 2), ∃ x2 ∈ Icc (Real.exp 1) (Real.exp 2), g x1 ≤ f' x2 + 2 * a) ↔ 
  a ∈ Icc ((Real.exp 2)^2 / 2 - 1/4) (Real.exp 2^2 / 2 - 1/4) := sorry

end g_monotonicity_intervals_f_decreasing_min_a_range_a_l96_96814


namespace banana_unique_permutations_l96_96754

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96754


namespace arrangement_count_BANANA_l96_96482

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96482


namespace permutations_BANANA_l96_96295

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96295


namespace half_MN_correct_l96_96817

noncomputable def OM : ℝ × ℝ := (-2, 3)
noncomputable def ON : ℝ × ℝ := (-1, -5)
noncomputable def MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)
noncomputable def half_MN : ℝ × ℝ := (MN.1 / 2, MN.2 / 2)

theorem half_MN_correct : half_MN = (1 / 2, -4) :=
by
  -- define the values of OM and ON
  let OM : ℝ × ℝ := (-2, 3)
  let ON : ℝ × ℝ := (-1, -5)
  -- calculate MN
  let MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)
  -- calculate half of MN
  let half_MN : ℝ × ℝ := (MN.1 / 2, MN.2 / 2)
  -- assert the expected value
  exact sorry

end half_MN_correct_l96_96817


namespace arrangements_of_BANANA_l96_96593

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

end arrangements_of_BANANA_l96_96593


namespace karl_sticker_count_l96_96842

theorem karl_sticker_count : 
  ∀ (K R B : ℕ), 
    (R = K + 20) → 
    (B = R - 10) → 
    (K + R + B = 105) → 
    K = 25 := 
by
  intros K R B hR hB hSum
  sorry

end karl_sticker_count_l96_96842


namespace students_per_class_l96_96002

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

end students_per_class_l96_96002


namespace banana_arrangements_l96_96574

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96574


namespace number_of_arrangements_l96_96235

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

end number_of_arrangements_l96_96235


namespace negative_values_count_l96_96983

theorem negative_values_count :
  ∃ x_vals : Finset ℤ, (∀ x ∈ x_vals, x < 0 ∧ ∃ n : ℕ, 0 < n ∧ n ≤ 14 ∧ x + 200 = n^2) ∧ x_vals.card = 14 :=
by sorry

end negative_values_count_l96_96983


namespace banana_permutations_l96_96505

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96505


namespace compare_polynomials_l96_96802

variable (x : ℝ)
variable (h : x > 1)

theorem compare_polynomials (h : x > 1) : x^3 + 6 * x > x^2 + 6 := 
by
  sorry

end compare_polynomials_l96_96802


namespace centers_distance_ABC_l96_96839

-- Define triangle ABC with the given properties
structure RightTriangle (ABC : Type) :=
(angle_A : ℝ)
(angle_C : ℝ)
(shorter_leg : ℝ)

-- Given: angle A is 30 degrees, angle C is 90 degrees, and shorter leg AC is 1
def triangle_ABC : RightTriangle ℝ := {
  angle_A := 30,
  angle_C := 90,
  shorter_leg := 1
}

-- Define the distance between the centers of the inscribed circles of triangles ACD and BCD
noncomputable def distance_between_centers (ABC : RightTriangle ℝ): ℝ :=
  sorry  -- placeholder for the actual proof

-- Example problem statement
theorem centers_distance_ABC (ABC : RightTriangle ℝ) (h_ABC : ABC = triangle_ABC) :
  distance_between_centers ABC = (Real.sqrt 3 - 1) / Real.sqrt 2 :=
sorry

end centers_distance_ABC_l96_96839


namespace permutations_BANANA_l96_96402

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96402


namespace compare_fractions_l96_96027

-- Define the fractions
def frac1 : ℚ := -2/3
def frac2 : ℚ := -3/4

-- Prove that -2/3 > -3/4
theorem compare_fractions : frac1 > frac2 :=
by {
  sorry
}

end compare_fractions_l96_96027


namespace banana_arrangements_l96_96265

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96265


namespace find_k_l96_96906

-- Definitions based on the conditions
def number := 24
def bigPart := 13
  
theorem find_k (x y k : ℕ) 
  (original_number : x + y = 24)
  (big_part : x = 13 ∨ y = 13)
  (equation : k * x + 5 * y = 146) : k = 7 := 
  sorry

end find_k_l96_96906


namespace school_students_l96_96018

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

end school_students_l96_96018


namespace banana_permutations_l96_96514

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96514


namespace permutations_BANANA_l96_96307

theorem permutations_BANANA : 
  let word := "BANANA" 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let expected_permutations := 60
  (Nat.factorial total_letters) / (Nat.factorial a_count * Nat.factorial n_count) = expected_permutations := 
by
  sorry

end permutations_BANANA_l96_96307


namespace decimal_representation_of_fraction_l96_96780

theorem decimal_representation_of_fraction :
  (47 : ℝ) / (2^3 * 5^4) = 0.0094 :=
by
  sorry

end decimal_representation_of_fraction_l96_96780


namespace BANANA_arrangements_correct_l96_96333

/-- 
  Determine the number of distinct arrangements of the letters 
  in the word BANANA, where B occurs 1 time, A occurs 3 times, 
  and N occurs 2 times.
-/
def num_arrangements_BANANA : Nat :=
  6.factorial / (1.factorial * 3.factorial * 2.factorial)

theorem BANANA_arrangements_correct : num_arrangements_BANANA = 60 := by 
  sorry

end BANANA_arrangements_correct_l96_96333


namespace volume_of_cuboid_l96_96991

variable (a b c : ℝ)

def is_cuboid_adjacent_faces (a b c : ℝ) := a * b = 3 ∧ a * c = 5 ∧ b * c = 15

theorem volume_of_cuboid (a b c : ℝ) (h : is_cuboid_adjacent_faces a b c) :
  a * b * c = 15 := by
  sorry

end volume_of_cuboid_l96_96991


namespace numberOfWaysToArrangeBANANA_l96_96087

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96087


namespace number_of_ways_to_arrange_BANANA_l96_96429

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96429


namespace number_of_arrangements_BANANA_l96_96758

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96758


namespace banana_arrangements_l96_96687

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96687


namespace num_ways_to_arrange_BANANA_l96_96337

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96337


namespace banana_unique_permutations_l96_96752

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96752


namespace part1_part2_l96_96992

-- Let's define the arithmetic sequence and conditions
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + d * (n - 1)
def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables (a1 a4 a3 a5 : ℕ)
variable (d : ℕ)

-- Additional conditions for the problem  
axiom h1 : a1 = 2
axiom h2 : a4 = 8
axiom h3 : arithmetic_seq a1 d 3 + arithmetic_seq a1 d 5 = a4 + 8

-- Define S7
def S7 : ℕ := sum_arithmetic_seq a1 d 7

-- Part I: Prove S7 = 56
theorem part1 : S7 = 56 := 
by
  sorry

-- Part II: Prove k = 2 given additional conditions
variable (k : ℕ)

-- Given that a_3, a_{k+1}, S_k are a geometric sequence
def is_geom_seq (a b s : ℕ) : Prop := b*b = a * s

axiom h4 : a3 = arithmetic_seq a1 d 3
axiom h5 : ∃ k, 0 < k ∧ is_geom_seq a3 (arithmetic_seq a1 d (k + 1)) (sum_arithmetic_seq a1 d k)

theorem part2 : ∃ k, 0 < k ∧ k = 2 := 
by
  sorry

end part1_part2_l96_96992


namespace count_negative_values_of_x_l96_96974

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l96_96974


namespace arrangements_of_BANANA_l96_96576

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

end arrangements_of_BANANA_l96_96576


namespace permutations_BANANA_l96_96373

theorem permutations_BANANA : 
  let n := 6 -- total letters
  let a := 3 -- occurrences of 'A'
  let b := 2 -- occurrences of 'N'
  ∏ i in Finset.range (n + 1), (if i = a ∨ i = b then factorial i else 1) 
    = 60 := 
by 
  sorry

end permutations_BANANA_l96_96373


namespace arrangement_count_BANANA_l96_96476

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96476


namespace find_intersection_l96_96809

variable (A : Set ℝ)
variable (B : Set ℝ := {1, 2})
variable (f : ℝ → ℝ := λ x => x^2)

theorem find_intersection (h : ∀ x, x ∈ A → f x ∈ B) : A ∩ B = ∅ ∨ A ∩ B = {1} :=
by
  sorry

end find_intersection_l96_96809


namespace solve_for_a_l96_96861

theorem solve_for_a (a : ℕ) (h : a > 0) (eqn : a / (a + 37) = 925 / 1000) : a = 455 :=
sorry

end solve_for_a_l96_96861


namespace banana_arrangements_l96_96264

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96264


namespace arrangement_count_BANANA_l96_96232

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

end arrangement_count_BANANA_l96_96232


namespace arrange_BANANA_l96_96599

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96599


namespace arrangement_count_BANANA_l96_96483

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96483


namespace count_negative_values_of_x_l96_96972

theorem count_negative_values_of_x :
  {x : ℤ // x < 0 ∧ ∃ n : ℕ, n > 0 ∧ n * n = x + 200}.to_finset.card = 14 :=
by
  sorry

end count_negative_values_of_x_l96_96972


namespace BANANA_permutations_l96_96626

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96626


namespace total_cost_tom_pays_for_trip_l96_96876

/-- Tom needs to get 10 different vaccines and a doctor's visit to go to Barbados.
    Each vaccine costs $45.
    The doctor's visit costs $250.
    Insurance will cover 80% of these medical bills.
    The trip itself costs $1200.
    Prove that the total amount Tom has to pay for his trip to Barbados, including medical expenses, is $1340. -/
theorem total_cost_tom_pays_for_trip : 
  let cost_per_vaccine := 45
  let number_of_vaccines := 10
  let cost_doctor_visit := 250
  let insurance_coverage_rate := 0.8
  let trip_cost := 1200
  let total_medical_cost := (number_of_vaccines * cost_per_vaccine) + cost_doctor_visit
  let insurance_coverage := insurance_coverage_rate * total_medical_cost
  let net_medical_cost := total_medical_cost - insurance_coverage
  let total_cost := trip_cost + net_medical_cost
  total_cost = 1340 := 
by 
  sorry

end total_cost_tom_pays_for_trip_l96_96876


namespace mass_percentage_Al_in_AlI3_l96_96792

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_AlI3 : ℝ := molar_mass_Al + 3 * molar_mass_I

theorem mass_percentage_Al_in_AlI3 : 
  (molar_mass_Al / molar_mass_AlI3) * 100 = 6.62 := 
  sorry

end mass_percentage_Al_in_AlI3_l96_96792


namespace banana_arrangements_l96_96449

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96449


namespace arrange_BANANA_l96_96069

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96069


namespace simplify_expression_l96_96853

theorem simplify_expression : 
  18 * (8 / 15) * (3 / 4) = 12 / 5 := 
by 
  sorry

end simplify_expression_l96_96853


namespace banana_arrangements_l96_96661

theorem banana_arrangements : 
  let total_permutations := Nat.fact 6
  let repeated_A := Nat.fact 3
  let repeated_N := Nat.fact 2
  let repeated_B := Nat.fact 1
  total_permutations / (repeated_A * repeated_N * repeated_B) = 60 :=
by
  sorry

end banana_arrangements_l96_96661


namespace A_alone_work_days_l96_96899

noncomputable def A_and_B_together : ℕ := 40
noncomputable def A_and_B_worked_together_days : ℕ := 10
noncomputable def B_left_and_C_joined_after_days : ℕ := 6
noncomputable def A_and_C_finish_remaining_work_days : ℕ := 15
noncomputable def C_alone_work_days : ℕ := 60

theorem A_alone_work_days (h1 : A_and_B_together = 40)
                          (h2 : A_and_B_worked_together_days = 10)
                          (h3 : B_left_and_C_joined_after_days = 6)
                          (h4 : A_and_C_finish_remaining_work_days = 15)
                          (h5 : C_alone_work_days = 60) : ∃ (n : ℕ), n = 30 :=
by {
  sorry -- Proof goes here
}

end A_alone_work_days_l96_96899


namespace rectangle_area_l96_96863

theorem rectangle_area (l w : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 200) : l * w = 1600 :=
by
  sorry

end rectangle_area_l96_96863


namespace negative_values_count_l96_96943

open Nat

theorem negative_values_count : 
  {x : ℤ // ∃ n : ℕ, (1 ≤ n ∧ n ≤ 14) ∧ x = n^2 - 200 ∧ x < 0}.card = 14 :=
by
  sorry

end negative_values_count_l96_96943


namespace permutations_BANANA_l96_96412

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96412


namespace arrangement_count_BANANA_l96_96494

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96494


namespace banana_arrangements_l96_96439

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96439


namespace BANANA_permutations_l96_96615

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end BANANA_permutations_l96_96615


namespace number_of_unique_permutations_BANANA_l96_96552

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96552


namespace binom_60_3_eq_34220_l96_96049

theorem binom_60_3_eq_34220 : Nat.choose 60 3 = 34220 := by
  sorry

end binom_60_3_eq_34220_l96_96049


namespace banana_arrangements_l96_96261

theorem banana_arrangements : 
  let totalLetters := 6
  let repeatedA := 3
  let repeatedN := 2
  let repeatedB := 1
  let totalArrangements := Nat.factorial totalLetters / (Nat.factorial repeatedA * Nat.factorial repeatedN * Nat.factorial repeatedB)
  totalArrangements = 60 := by
  sorry

end banana_arrangements_l96_96261


namespace number_of_arrangements_BANANA_l96_96759

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96759


namespace banana_arrangements_l96_96097

theorem banana_arrangements : 
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  Nat.fact total_letters / (Nat.fact count_A * Nat.fact count_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96097


namespace arrange_BANANA_l96_96525

theorem arrange_BANANA : ∃ n : ℕ, n = 6! / (3! * 2!) ∧ n = 60 :=
by
  sorry

end arrange_BANANA_l96_96525


namespace arrangements_of_BANANA_l96_96588

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

end arrangements_of_BANANA_l96_96588


namespace banana_permutations_l96_96510

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96510


namespace arrangement_count_BANANA_l96_96224

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

end arrangement_count_BANANA_l96_96224


namespace number_of_unique_permutations_BANANA_l96_96545

theorem number_of_unique_permutations_BANANA : 
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  ∑ (multiset.to_list (multiset.range total_letters)), 
    (perm_cardinality freq_A freq_N total_letters),
  60 := 
  sorry

end number_of_unique_permutations_BANANA_l96_96545


namespace numberOfWaysToArrangeBANANA_l96_96077

noncomputable def numberOfArrangements : Nat :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

theorem numberOfWaysToArrangeBANANA : numberOfArrangements = 60 := by
  sorry

end numberOfWaysToArrangeBANANA_l96_96077


namespace rod_center_of_gravity_shift_l96_96988

noncomputable def rod_shift (l : ℝ) (s : ℝ) : ℝ := 
  |(l / 2) - ((l - s) / 2)| 

theorem rod_center_of_gravity_shift : 
  rod_shift l 80 = 40 := by
  sorry

end rod_center_of_gravity_shift_l96_96988


namespace banana_arrangements_l96_96570

theorem banana_arrangements : 
  ∀ (letters : List Char), (letters = ['B', 'A', 'N', 'A', 'N', 'A']) →
  (letters.count 'A' = 3) →
  (letters.count 'N' = 2) →
  (letters.length = 6) →
  (6.factorial / (3.factorial * 2.factorial) = 60) :=
by
  intros letters h1 h2 h3 h4
  sorry

end banana_arrangements_l96_96570


namespace solve_fraction_eq_l96_96856

theorem solve_fraction_eq (x : ℝ) (h : x ≠ -2) : (x = -1) ↔ ((x^2 + 2 * x + 3) / (x + 2) = x + 3) := 
by 
  sorry

end solve_fraction_eq_l96_96856


namespace banana_permutations_l96_96460

-- Number of ways to arrange the letters of the word "BANANA"
theorem banana_permutations : (Nat.factorial 6) / (Nat.factorial 3 * Nat.factorial 2) = 60 := by
  sorry

end banana_permutations_l96_96460


namespace permutations_banana_l96_96636

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96636


namespace arrangement_count_BANANA_l96_96493

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96493


namespace simple_interest_sum_l96_96010

variable {P R : ℝ}

theorem simple_interest_sum :
  P * (R + 6) = P * R + 3000 → P = 500 :=
by
  intro h
  sorry

end simple_interest_sum_l96_96010


namespace duration_period_l96_96903

-- Define the conditions and what we need to prove
theorem duration_period (t : ℝ) (h : 3200 * 0.025 * t = 400) : 
  t = 5 :=
sorry

end duration_period_l96_96903


namespace number_neither_9_nice_nor_10_nice_500_l96_96796

def is_k_nice (N k : ℕ) : Prop := ∃ a : ℕ, a > 0 ∧ (∃ m : ℕ, N = (k * m) + 1)

def count_k_nice (N k : ℕ) : ℕ :=
  (N - 1) / k + 1

def count_neither_9_nice_nor_10_nice (N : ℕ) : ℕ :=
  let count_9_nice := count_k_nice N 9
  let count_10_nice := count_k_nice N 10
  let lcm_9_10 := 90  -- lcm of 9 and 10
  let count_both := count_k_nice N lcm_9_10
  N - (count_9_nice + count_10_nice - count_both)

theorem number_neither_9_nice_nor_10_nice_500 : count_neither_9_nice_nor_10_nice 500 = 400 :=
  sorry

end number_neither_9_nice_nor_10_nice_500_l96_96796


namespace BANANA_arrangement_l96_96155

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96155


namespace BANANA_permutations_l96_96196

def count_permutations (word : String) (total: ℕ) (freq : List (Char × ℕ)) : ℕ :=
  if h : total = word.length ∧ (∀ ch in word.to_list, (ch, word.to_list.countp (λ x => x = ch)) ∈ freq) then
    Nat.factorial total / (freq.map (λ ⟨_, c⟩ => Nat.factorial c)).prod
  else
    0

theorem BANANA_permutations :
  count_permutations "BANANA" 6 [('A', 3), ('N', 2), ('B', 1)] = 60 :=
by sorry

end BANANA_permutations_l96_96196


namespace permutations_banana_l96_96644

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96644


namespace banana_permutations_l96_96503

theorem banana_permutations : 
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  in total_permutations = 60 :=
by
  let total_letters := 6
  let a_count := 3
  let n_count := 2
  let b_count := 1
  let total_permutations := Nat.factorial total_letters / (Nat.factorial a_count * Nat.factorial n_count * Nat.factorial b_count)
  have h: total_permutations = 60 := by sorry
  exact h

end banana_permutations_l96_96503


namespace min_distance_l96_96844

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance :
  ∃ m : ℝ, (∀ x > 0, x ≠ m → (f m - g m) ≤ (f x - g x)) ∧ m = Real.sqrt 2 / 2 :=
by
  sorry

end min_distance_l96_96844


namespace banana_arrangements_l96_96692

variable (A N B : ℕ)
variable (hA : A = 3)
variable (hN : N = 2)
variable (hB : B = 1)
variable (hTotal : A + N + B = 6)

theorem banana_arrangements (hFact : factorial 6 = 720) (hFactA : factorial A = 6) (hFactN : factorial N = 2) : 
  ∃ n : ℕ, n = 720 / (6 * 2) ∧ n = 60 :=
by {
  -- Given variable/assumption relations:
  have hTotal6 : 6 = A + N + B, from hTotal, -- Total letters
  have hFact3 : factorial 3 = 6, from hFactA,
  have hFact2 : factorial 2 = 2, from hFactN,

  -- Calculation
  have hArrangement : 720 / (6 * 2) = 60, by calc
    720 / (6 * 2)
      = 720 / 12  : by simp
    ... = 60      : by norm_num,

  -- Solution
  existsi 60,
  constructor,
  { exact hArrangement },
  { refl }
}

end banana_arrangements_l96_96692


namespace banana_unique_permutations_l96_96738

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96738


namespace BANANA_arrangement_l96_96168

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end BANANA_arrangement_l96_96168


namespace permutations_banana_l96_96638

open Finset

theorem permutations_banana : 
  let permutations (n m1 m2 m3 : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial m1) * (nat.factorial m2) * (nat.factorial m3)) in 
  permutations 6 1 3 2 = 60 :=
by
  sorry

end permutations_banana_l96_96638


namespace banana_unique_permutations_l96_96740

theorem banana_unique_permutations (total_letters repeated_A repeated_N : ℕ) 
  (h1 : total_letters = 6) (h2 : repeated_A = 3) (h3 : repeated_N = 2) : 
  ∃ (unique_permutations : ℕ), 
    unique_permutations = Nat.factorial total_letters / 
    (Nat.factorial repeated_A * Nat.factorial repeated_N) ∧ 
    unique_permutations = 60 :=
by
  sorry

end banana_unique_permutations_l96_96740


namespace consistent_system_l96_96005

variable (x y : ℕ)

def condition1 := x + y = 40
def condition2 := 2 * 15 * x = 20 * y

theorem consistent_system :
  condition1 x y ∧ condition2 x y ↔ 
  (x + y = 40 ∧ 2 * 15 * x = 20 * y) :=
by
  sorry

end consistent_system_l96_96005


namespace arrangement_count_BANANA_l96_96484

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l96_96484


namespace number_of_arrangements_BANANA_l96_96767

theorem number_of_arrangements_BANANA : (Nat.factorial 6) / ((Nat.factorial 2) * (Nat.factorial 2)) = 180 := 
by
  sorry

end number_of_arrangements_BANANA_l96_96767


namespace sum_of_angles_l96_96937

theorem sum_of_angles (x u v : ℝ) (h1 : u = Real.sin x) (h2 : v = Real.cos x)
  (h3 : 0 ≤ x ∧ x ≤ 2 * Real.pi) 
  (h4 : Real.sin x ^ 4 - Real.cos x ^ 4 = (u - v) / (u * v)) 
  : x = Real.pi / 4 ∨ x = 5 * Real.pi / 4 → (Real.pi / 4 + 5 * Real.pi / 4) = 3 * Real.pi / 2 := 
by
  intro h
  sorry

end sum_of_angles_l96_96937


namespace banana_arrangements_l96_96443

/-- The number of ways to arrange the letters of the word "BANANA". -/
theorem banana_arrangements : 
  let total_letters := 6
  let repeats_A := 3
  let repeats_N := 2
  fact total_letters / (fact repeats_A * fact repeats_N) = 60 := 
by
  sorry

end banana_arrangements_l96_96443


namespace students_drawn_from_A_l96_96852

-- Define the conditions as variables (number of students in each school)
def studentsA := 3600
def studentsB := 5400
def studentsC := 1800
def sampleSize := 90

-- Define the total number of students
def totalStudents := studentsA + studentsB + studentsC

-- Define the proportion of students in School A
def proportionA := studentsA / totalStudents

-- Define the number of students to be drawn from School A using stratified sampling
def drawnFromA := sampleSize * proportionA

-- The theorem to prove
theorem students_drawn_from_A : drawnFromA = 30 :=
by
  sorry

end students_drawn_from_A_l96_96852


namespace problem_equivalent_l96_96857

theorem problem_equivalent (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) (hz_eq : z = 10 * y) :
  (x + 4 * y + z) / (4 * x - y - z) = 0 :=
by
  sorry

end problem_equivalent_l96_96857


namespace arrange_BANANA_l96_96611

theorem arrange_BANANA : 
  (∃ word : List Char, word = ['B', 'A', 'N', 'A', 'N', 'A']) →
  ∃ letters_count : List (Char × ℕ), 
  letters_count = [('B', 1), ('A', 3), ('N', 2)] →
  (6.factorial / ((3.factorial) * (2.factorial) * (1.factorial)) = 60) :=
by 
  intros _ _ 
  simp 
  sorry

end arrange_BANANA_l96_96611


namespace arrange_banana_l96_96143

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

end arrange_banana_l96_96143


namespace total_women_attendees_l96_96016

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

end total_women_attendees_l96_96016


namespace arrange_BANANA_l96_96057

-- Definitions based on conditions
def total_letters : ℕ := 6
def count_A : ℕ := 3
def count_N : ℕ := 2
def count_B : ℕ := 1

-- The main theorem to prove
theorem arrange_BANANA : 
  (Nat.factorial total_letters) / 
  ((Nat.factorial count_A) * (Nat.factorial count_N) * (Nat.factorial count_B)) = 60 :=
by
  sorry

end arrange_BANANA_l96_96057


namespace decimal_representation_of_fraction_l96_96779

theorem decimal_representation_of_fraction :
  (47 : ℝ) / (2^3 * 5^4) = 0.0094 :=
by
  sorry

end decimal_representation_of_fraction_l96_96779


namespace notebooks_distributed_l96_96798

theorem notebooks_distributed  (C : ℕ) (N : ℕ) 
  (h1 : N = C^2 / 8) 
  (h2 : N = 8 * C) : 
  N = 512 :=
by 
  sorry

end notebooks_distributed_l96_96798


namespace number_of_arrangements_l96_96245

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

end number_of_arrangements_l96_96245


namespace number_of_ways_to_arrange_BANANA_l96_96419

theorem number_of_ways_to_arrange_BANANA : 
  ∃ (n : ℕ ), n = (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) ∧ n = 60 :=
by
  sorry

end number_of_ways_to_arrange_BANANA_l96_96419


namespace permutations_BANANA_l96_96405

theorem permutations_BANANA : 
  let letters_total := 6
  let A_repeats := 3
  let N_repeats := 2
  (Nat.fact letters_total) / ((Nat.fact A_repeats) * (Nat.fact N_repeats)) = 60 :=
by
  sorry

end permutations_BANANA_l96_96405


namespace permutations_of_BANANA_l96_96699

theorem permutations_of_BANANA : ∃ (ways : ℕ), ways = nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2) ∧ ways = 60 := by
  use nat.factorial 6 / (nat.factorial 1 * nat.factorial 3 * nat.factorial 2)
  split
  · rfl
  · sorry

end permutations_of_BANANA_l96_96699


namespace BANANA_permutation_l96_96189

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem BANANA_permutation : ∀ (n : ℕ) (a_count : ℕ) (n_count : ℕ) (b_count : ℕ),
  n = 6 → a_count = 3 → n_count = 2 → b_count = 1 →
  (factorial n) / ((factorial a_count) * (factorial n_count) * (factorial b_count)) = 60 :=
by intros n a_count n_count b_count hn ha hn hn hb
   rw [hn, ha, hn, hb]
   rw [factorial, factorial, factorial, factorial, factorial, factorial]
   rw [factorial, factorial, factorial, factorial]
   sorry

end BANANA_permutation_l96_96189


namespace num_ways_to_arrange_BANANA_l96_96341

theorem num_ways_to_arrange_BANANA : 
  let total_letters := 6
  let num_A := 3
  let num_N := 2
  let total_arrangements := Nat.factorial total_letters / (Nat.factorial num_A * Nat.factorial num_N)
  total_arrangements = 60 :=
by
  sorry

end num_ways_to_arrange_BANANA_l96_96341


namespace remainder_when_divided_l96_96806

theorem remainder_when_divided (P K Q R K' Q' S' T : ℕ)
  (h1 : P = K * Q + R)
  (h2 : Q = K' * Q' + S')
  (h3 : R * Q' = T) :
  P % (K * K') = K * S' + (T / Q') :=
by
  sorry

end remainder_when_divided_l96_96806
