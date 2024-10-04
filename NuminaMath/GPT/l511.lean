import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.GCDMonoid.Basic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Probability
import Mathlib.Analysis.Calculus.Extrema
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Partitions
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Prob.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Data.String.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.NumberTheory.Prime
import Mathlib.Probability.ProbabilityMassFunction.Finite
import Mathlib.ProbabilityTheory.Independent
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import analysis.calculus.mean_value
import data.polynomial

namespace Roe_saved_10_per_month_l511_511809

noncomputable def Roe_savings (x : ℕ) : Prop :=
  let January_to_July := 7 * x
  let August_to_November := 4 * 15
  let December := 20
  let Total_savings := 150
  January_to_July + August_to_November + December = Total_savings

theorem Roe_saved_10_per_month : ∃ x, Roe_savings x ∧ x = 10 := 
by
  use 10
  unfold Roe_savings
  norm_num
  sorry

end Roe_saved_10_per_month_l511_511809


namespace five_letter_words_with_vowel_l511_511236

-- Define the set of letters and identify vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels := {'A', 'E'}

-- Define the problem statement
theorem five_letter_words_with_vowel : 
  (number of 5-letter words with at least one vowel) = 6752 := 
sorry

end five_letter_words_with_vowel_l511_511236


namespace sufficient_but_not_necessary_condition_l511_511644

theorem sufficient_but_not_necessary_condition (a b : ℝ) : (b ≥ 0 → a^2 + b ≥ 0) ∧ ¬(∀ a b, a^2 + b ≥ 0 → b ≥ 0) := by
  sorry

end sufficient_but_not_necessary_condition_l511_511644


namespace ratio_of_x_to_y_l511_511917

theorem ratio_of_x_to_y (x y : ℝ) (R : ℝ) (h1 : x = R * y) (h2 : x - y = 0.909090909090909 * x) : R = 11 := by
  sorry

end ratio_of_x_to_y_l511_511917


namespace unique_root_in_intervals_l511_511745

theorem unique_root_in_intervals 
  {a : ℕ → ℝ} {p : polynomial ℝ} (h_sorted : ∀ j, a j < a (j + 1))
  (h_0 : a 0 = 0) (h_deg : p.degree = nat_degree p)
  (h_int : ∀ j (hj : 0 ≤ j ∧ j < (finrank ℝ (polynomial ℝ)) - 1), ∫ x in Ioc (a j) (a (j + 1)), p.real x = 0) :
  ∀ {j : ℕ} (hj : j < (finrank ℝ (polynomial ℝ)) - 1), ∃! c ∈ Ioc (a j) (a (j + 1)), p.real c = 0 := 
begin
  sorry
end

end unique_root_in_intervals_l511_511745


namespace neg_product_B_l511_511930

def expr_A := (-1 / 3) * (1 / 4) * (-6)
def expr_B := (-9) * (1 / 8) * (-4 / 7) * 7 * (-1 / 3)
def expr_C := (-3) * (-1 / 2) * 7 * 0
def expr_D := (-1 / 5) * 6 * (-2 / 3) * (-5) * (-1 / 2)

theorem neg_product_B :
  expr_B < 0 :=
by
  sorry

end neg_product_B_l511_511930


namespace anchor_concrete_ratio_l511_511739

-- Definitions of the given conditions
def roadwayDeckConcrete : ℕ := 1600
def builtAnchorConcrete : ℕ := 700
def totalBridgeConcrete : ℕ := 4800
def supportingPillarsConcrete : ℕ := 1800

-- Define an axiom to state the given conditions mathematically
axiom bridge_conditions : 
  ∀ (a1 a2 : ℕ), 
  totalBridgeConcrete = roadwayDeckConcrete + supportingPillarsConcrete + a1 + a2 ∧ 
  builtAnchorConcrete = a1 ∧ 
  a2 = builtAnchorConcrete

-- The proof statement
theorem anchor_concrete_ratio : 
  ∀ (a1 a2 : ℕ), 
  bridge_conditions a1 a2 → 
  a1 = a2 → 
  a1 / a2 = 1 := 
by 
  sorry

end anchor_concrete_ratio_l511_511739


namespace largest_five_digit_number_with_product_120_l511_511554

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def prod_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldr (· * ·) 1

def max_five_digit_prod_120 : ℕ := 85311

theorem largest_five_digit_number_with_product_120 :
  is_five_digit max_five_digit_prod_120 ∧ prod_of_digits max_five_digit_prod_120 = 120 :=
by
  sorry

end largest_five_digit_number_with_product_120_l511_511554


namespace count_valid_sets_l511_511336

def A := {1, 2, 3, 4, 5}
def B := {3, 4, 5, 6, 7}

def valid_sets (S : Set ℕ) : Prop :=
  S ⊆ A ∧ (S ∩ B).Nonempty

theorem count_valid_sets : 
  {S : Set ℕ | valid_sets S}.toFinset.card = 28 :=
sorry

end count_valid_sets_l511_511336


namespace sum_digit_differences_l511_511607

def first_digit (n : ℕ) : ℕ := 
  (n / 10 ^ ((Nat.log10 n) : ℕ))

def last_digit (n : ℕ) : ℕ := n % 10

def digit_difference (n : ℕ) : ℤ :=
  (first_digit n : ℤ) - (last_digit n : ℤ)

theorem sum_digit_differences :
  (∑ n in Finset.range 1000, digit_difference n) = 495 := 
sorry

end sum_digit_differences_l511_511607


namespace combined_volume_l511_511472

-- Defining the given conditions
def diameter_cylinder := 4
def height_cylinder := 5
def radius_cylinder := diameter_cylinder / 2

def diameter_cone := 4
def height_cone := 2
def radius_cone := diameter_cone / 2

-- Defining the volumes based on the conditions
def volume_cylinder := Real.pi * radius_cylinder^2 * height_cylinder
def volume_cone := (1 / 3) * Real.pi * radius_cone^2 * height_cone

-- The theorem to prove given the conditions
theorem combined_volume : volume_cylinder - volume_cone = (52 / 3) * Real.pi :=
by
  sorry

end combined_volume_l511_511472


namespace feet_of_medians_collinear_l511_511754

-- Definitions
variable {A B C H : Type} -- Points representing the vertices of the triangle and the orthocenter
variable [line : Type] -- Type for lines
variable (ell1 ell2 : line) -- Two lines intersecting perpendicularly at H

-- Main theorem
theorem feet_of_medians_collinear
  (ABC_triangle : is_triangle A B C)
  (H_is_orthocenter : is_orthocenter H A B C)
  (ell1_ell2_perpendicular : is_perpendicular ell1 ell2)
  (ell1_passes_through_H : passes_through ell1 H)
  (ell2_passes_through_H : passes_through ell2 H) :
  collinear (feet_of_medians H ell1 ell2 A B C) :=
  sorry

end feet_of_medians_collinear_l511_511754


namespace sum_of_differences_l511_511585

open Nat
open BigOperators

theorem sum_of_differences (n : ℕ) (h : n ≥ 1 ∧ n ≤ 999) : 
  let differences := (fun x => 
                        let first_digit := x / 10;
                        let last_digit := x % 10;
                        first_digit - last_digit) in
  ∑ i in (finset.range (n + 1)).filter (λ x, 1 ≤ x ∧ x ≤ 999), differences i = 495 :=
by
  -- Acknowledge the need for a more refined filtering criteria for numbers between 1 and 999
  sorry

end sum_of_differences_l511_511585


namespace intersection_point_sum_l511_511048

def f (x : ℝ) : ℝ := 5 - (x - 1)^2 / 3

theorem intersection_point_sum :
  ∃ a b : ℝ, f(a) = f(a-4) ∧ a + b = 20 / 3 :=
by
  use 3
  use 11 / 3
  split
  { sorry } -- proof that f(a) = f(a-4)
  { sorry } -- proof that a + b = 20 / 3

end intersection_point_sum_l511_511048


namespace rhombus_fourth_vertex_l511_511649

theorem rhombus_fourth_vertex (a b : ℝ) :
  ∃ x y : ℝ, (x, y) = (a - b, a + b) ∧ dist (a, b) (x, y) = dist (-b, a) (x, y) ∧ dist (-b, a) (x, y) = dist (0, 0) (x, y) :=
by
  use (a - b)
  use (a + b)
  sorry

end rhombus_fourth_vertex_l511_511649


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511228

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let total_words := 6^5
  let words_without_vowels := 4^5
  total_words - words_without_vowels = 6752 :=
by
  let total_words := 6^5
  let words_without_vowels := 4^5
  have h_total_words : total_words = 7776 := by norm_num
  have h_words_without_vowels : words_without_vowels = 1024 := by norm_num
  calc
    7776 - 1024 = 6752 : by norm_num

end number_of_5_letter_words_with_at_least_one_vowel_l511_511228


namespace collinear_points_C_coordinates_C_l511_511130

open_locale classical

variables {a b : ℝ}

-- Condition for problem (1)
def point_A := (1, 1 : ℝ)
def point_B := (3, -1 : ℝ)
def point_C (a b : ℝ) := (a, b)

def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst (B) - fst (A), snd (B) - snd (A)) = k • (fst (C) - fst (A), snd (C) - snd (A))

-- Problem (1)
theorem collinear_points_C (h : collinear point_A point_B (point_C a b)) :
  a = 2 - b :=
sorry

-- Condition for problem (2)
def vector_eq (A B C : ℝ × ℝ) : Prop :=
  (fst (C) - fst (A), snd (C) - snd (A)) = 2 • (fst (B) - fst (A), snd (B) - snd (A))

-- Problem (2)
theorem coordinates_C (h : vector_eq point_A point_B (point_C a b)) :
  (a, b) = (5, -3 : ℝ) :=
sorry

end collinear_points_C_coordinates_C_l511_511130


namespace second_tap_empty_time_l511_511907

theorem second_tap_empty_time :
  ∃ T : ℝ, (1 / 4 - 1 / T = 3 / 28) → T = 7 :=
by
  sorry

end second_tap_empty_time_l511_511907


namespace prove_perpendicular_l511_511290

noncomputable def isosceles_triangle (A B C : Type) [EuclideanGeometry A B C] (hAB_AC : AB = AC) : Prop :=
  sorry

noncomputable def midpoint {A B C : Type} [EuclideanGeometry A B C] (H : Type) (h_mid : is_midpoint H B C) : Prop :=
  sorry

noncomputable def perpendicular {A B : Type} [EuclideanGeometry A B] (H E : Type) (h_perp : is_perpendicular H E AC) : Prop :=
  sorry

noncomputable def is_midpoint {A B : Type} [EuclideanGeometry A B] (O : Type) (h_mid : is_midpoint O H E) : Prop :=
  sorry

noncomputable theory

open EuclideanGeometry

theorem prove_perpendicular {A B C H E O : Type} [EuclideanGeometry A B C H E O]
  (h_isosceles : isosceles_triangle A B C hAB_AC)
  (hH_mid : midpoint H (midpoint H B C h_mid))
  (hHE_perp : perpendicular HE (HE_perpendicular_to_AC H E AC hHE_perp))
  (hO_mid : is_midpoint O (O_mid H E hO_mid)) :
  is_perpendicular AO BE :=
sorry

end prove_perpendicular_l511_511290


namespace percentage_after_2_years_l511_511511

noncomputable def initial_participants : ℕ := 1000

noncomputable def participants_28_32_initial : ℕ := initial_participants * 40 / 100
noncomputable def participants_23_27_initial : ℕ := initial_participants * 30 / 100
noncomputable def participants_33_37_initial : ℕ := initial_participants * 20 / 100
noncomputable def participants_18_22_initial : ℕ := initial_participants * 10 / 100

noncomputable def participants_28_32_final : ℕ := 
  let after_first_year := participants_28_32_initial + (participants_28_32_initial * 5 / 100)
  in after_first_year + (after_first_year * 5 / 100)

noncomputable def participants_23_27_final : ℕ := 
  let after_first_year := participants_23_27_initial - (participants_23_27_initial * 3 / 100)
  in after_first_year - (after_first_year * 3 / 100)

noncomputable def participants_33_37_final : ℕ := 
  let after_first_year := participants_33_37_initial + (participants_33_37_initial * 2 / 100)
  in after_first_year + (after_first_year * 2 / 100)

noncomputable def participants_18_22_final : ℕ := participants_18_22_initial

noncomputable def total_final : ℕ := participants_28_32_final + participants_23_27_final + participants_33_37_final + participants_18_22_final

noncomputable def percentage_28_32_final : ℝ := (participants_28_32_final : ℝ) / (total_final : ℝ) * 100
noncomputable def percentage_23_27_final : ℝ := (participants_23_27_final : ℝ) / (total_final : ℝ) * 100
noncomputable def percentage_33_37_final : ℝ := (participants_33_37_final : ℝ) / (total_final : ℝ) * 100
noncomputable def percentage_18_22_final : ℝ := (participants_18_22_final : ℝ) / (total_final : ℝ) * 100

theorem percentage_after_2_years :
  percentage_28_32_final ≈ 42.77 ∧
  percentage_23_27_final ≈ 27.35 ∧
  percentage_33_37_final ≈ 20.17 ∧
  percentage_18_22_final ≈ 9.70 := sorry

end percentage_after_2_years_l511_511511


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511184

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the vowels
def vowels := {'A', 'E'}

-- Define the number of n-length words constructible from a set of letters
def num_words (n : ℕ) (alphabet : Set Char) : ℕ :=
  (alphabet.size ^ n)

-- The total number of 5-letter words (unrestricted)
def total_words := num_words 5 letters

-- The number of 5-letter words with no vowels
def no_vowel_words := num_words 5 (letters \ vowels)

-- The number of 5-letter words with at least one vowel
def at_least_one_vowel_words := total_words - no_vowel_words

-- The statement to prove that the number of 5-letter words with at least one vowel is 6752
theorem number_of_5_letter_words_with_at_least_one_vowel : 
  at_least_one_vowel_words = 6752 :=
by 
  -- Proof will be provided here
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l511_511184


namespace distance_is_fifteen_l511_511548

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_is_fifteen : distance 0 12 9 0 = 15 := 
by
  sorry

end distance_is_fifteen_l511_511548


namespace arrange_365_cards_l511_511056

theorem arrange_365_cards (k : ℕ) (N : ℕ) (x : ℕ) (rubles : ℕ) :
  (∀ k, N ≤ 3^k → (∃ x, arrange_cards x N) → (∃ k, k ≤ rubles → arrange_cards (x + k) (N + 1)))  →
  (∀ k, k = 1 → arrange_cards 1 1) →
  (∀ N, (3^5 = 243) → (365 ≤ 3^6) → arrange_cards 1845 365) →
  (∀ rubles, 1845 ≤ 2000 → arrange_cards 2000 365) :=
by
  sorry


end arrange_365_cards_l511_511056


namespace neg_number_among_set_l511_511039

theorem neg_number_among_set :
  ∃ n ∈ ({5, 1, -2, 0} : Set ℤ), n < 0 ∧ n = -2 :=
by
  sorry

end neg_number_among_set_l511_511039


namespace five_letter_words_with_vowels_l511_511176

noncomputable def num_5_letter_words_with_vowels : Nat := 7776 - 1024

theorem five_letter_words_with_vowels
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'})
  (len : Nat := 5) :
  (letters.card ^ len) - ((letters.filter (λ c, ¬ c ∈ vowels)).card ^ len) = 6752 := by
  sorry

end five_letter_words_with_vowels_l511_511176


namespace weapon_arrangements_l511_511402

-- Conditions as definitions
def post_count : ℕ := 20
def weapon_count : ℕ := 5

def valid_positions (positions : Finset ℕ) : Prop :=
  (positions.card = weapon_count) ∧            -- Exactly 5 posts are equipped
  (1 ∉ positions) ∧ (post_count ∉ positions) ∧ -- First and last posts are not equipped
  (∀ k, k ∈ positions → k + 1 ∉ positions) ∧  -- No two adjacent posts are both equipped
  (∀ k, (1 ≤ k ∧ k ≤ post_count - 4) →        -- Every set of 5 consecutive posts
       ∃ i ∈ positions, i ∈ (Finset.range 5).image (+ k - 1)) -- contains at least one equipped post

-- Question: Number of valid ways
def number_of_valid_ways : ℕ := 69600

-- The final statement to prove
theorem weapon_arrangements : ∃ positions : Finset ℕ, valid_positions positions ∧ 
  Finset.card { positions | valid_positions positions } = number_of_valid_ways :=
sorry

end weapon_arrangements_l511_511402


namespace benny_hours_l511_511050

theorem benny_hours (hours_per_day : ℕ) (days : ℕ) (h1 : hours_per_day = 5) (h2 : days = 12) : 
  hours_per_day * days = 60 := by
  rw [h1, h2]
  sorry

end benny_hours_l511_511050


namespace sum_of_factorials_of_digits_l511_511446

theorem sum_of_factorials_of_digits :
  nat.factorial 7 + nat.factorial 2 + nat.factorial 1 = 5043 :=
by
  sorry

end sum_of_factorials_of_digits_l511_511446


namespace arithmetic_sequence_properties_l511_511122

noncomputable def a_n (n : ℕ) : ℤ :=
  2 * n - 10

def T_n (n : ℕ) : ℤ :=
  if 1 ≤ n ∧ n ≤ 5 then
    9 * n - n ^ 2
  else if 6 ≤ n then
    n ^ 2 - 9 * n + 40
  else
    0 -- To handle cases where n < 1

theorem arithmetic_sequence_properties (d : ℕ) (a : ℕ → ℕ) (a_3 a_7 : ℕ) (a_2 a_8 : ℤ) (n : ℕ) (h1 : d > 0) 
  (h2 : a 3 * a 7 = -16) (h3 : a 2 + a 8 = 0):
  (∀ n, a n = 2 * n - 10) ∧
  (∀ n, T_n n = 
    if 1 ≤ n ∧ n ≤ 5 then
      9 * n - n ^ 2
    else if 6 ≤ n then
      n ^ 2 - 9 * n + 40
    else
      0) :=
by
  sorry

end arithmetic_sequence_properties_l511_511122


namespace price_of_other_frisbees_proof_l511_511030

noncomputable def price_of_other_frisbees (P : ℝ) : Prop :=
  ∃ x : ℝ, x + (60 - x) = 60 ∧ x ≥ 0 ∧ P * x + 4 * (60 - x) = 204 ∧ (60 - x) ≥ 24

theorem price_of_other_frisbees_proof : price_of_other_frisbees 3 :=
by
  sorry

end price_of_other_frisbees_proof_l511_511030


namespace largest_element_in_list_l511_511019

def list_conditions (l : List ℕ) : Prop :=
  ∃ (a b d e : ℕ), l = [8, 8, 10, d, e] ∧ 
  10 = l.sorted.nthLe 2 sorry ∧
  60 = l.sum

theorem largest_element_in_list : ∀ (l : List ℕ), 
  list_conditions l → 
  l.maximum = some 23 :=
begin
  intros l h,
  sorry
end

end largest_element_in_list_l511_511019


namespace hyperbola_no_intersection_l511_511322

theorem hyperbola_no_intersection (a b e : ℝ)
  (ha : 0 < a) (hb : 0 < b)
  (h_e : e = (Real.sqrt (a^2 + b^2)) / a) :
  (√5 ≥ e ∧ 1 < e) → ∀ x y : ℝ, ¬ (y = 2 * x ∧ (x^2 / a^2 - y^2 / b^2 = 1)) :=
begin
  intros h_intersect x y,
  sorry,
end

end hyperbola_no_intersection_l511_511322


namespace inverse_of_A_l511_511085

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ := ![![4, -2], ![5, 11]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  if h : det A ≠ 0 then inv A else 0

theorem inverse_of_A :
  A_inv = if det A ≠ 0 then
    ![![11/54, -5/54], ![1/27, 2/27]]
  else 0 := 
sorry

end inverse_of_A_l511_511085


namespace find_k_l511_511834

-- Define the conditions
def equation : polynomial ℝ := polynomial.X^2 + 8 * polynomial.X + k
def ratio (r s : ℝ) := r = 3 * s

-- The main theorem
theorem find_k (k r s : ℝ) (h : polynomial.roots equation = {r, s}) (hratio: ratio r s) (hnonzero : r ≠ 0 ∧ s ≠ 0) : 
  k = 12 :=
sorry

end find_k_l511_511834


namespace count_5_letter_words_with_at_least_one_vowel_l511_511208

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := (letters.card) ^ 5

def non_vowel_letters : Finset Char := letters \ vowels

def total_non_vowel_5_letter_words : ℕ := (non_vowel_letters.card) ^ 5

theorem count_5_letter_words_with_at_least_one_vowel :
  total_5_letter_words - total_non_vowel_5_letter_words = 6752 :=
by
  sorry

end count_5_letter_words_with_at_least_one_vowel_l511_511208


namespace subtraction_result_l511_511646

open Matrix

namespace Vector

def a : (Fin 3 → ℝ) :=
  ![5, -3, 2]

def b : (Fin 3 → ℝ) :=
  ![-2, 4, 1]

theorem subtraction_result : a - (2 • b) = ![9, -11, 0] :=
by
  -- Skipping the proof
  sorry

end Vector

end subtraction_result_l511_511646


namespace bus_trip_cost_l511_511796

-- Problem Statement Definitions
def distance_AB : ℕ := 4500
def cost_per_kilometer_bus : ℚ := 0.20

-- Theorem Statement
theorem bus_trip_cost : distance_AB * cost_per_kilometer_bus = 900 := by
  sorry

end bus_trip_cost_l511_511796


namespace find_radius_l511_511006

theorem find_radius (abbc: ℝ) (adbd: ℝ) (bccc: ℝ) (dcdd: ℝ) (R: ℝ)
  (h1: abbc = 4) (h2: adbd = 4) (h3: bccc = 2) (h4: dcdd = 1) :
  R = 5 :=
sorry

end find_radius_l511_511006


namespace cube_inequality_l511_511748

theorem cube_inequality (a b : ℝ) : a > b ↔ a^3 > b^3 :=
sorry

end cube_inequality_l511_511748


namespace extra_yellow_balls_dispatched_eq_49_l511_511495

-- Define the given conditions
def ordered_balls : ℕ := 114
def white_balls : ℕ := ordered_balls / 2
def yellow_balls := ordered_balls / 2

-- Define the additional yellow balls dispatched and the ratio condition
def dispatch_error_ratio : ℚ := 8 / 15

-- The statement to prove the number of extra yellow balls dispatched
theorem extra_yellow_balls_dispatched_eq_49
  (ordered_balls_rounded : ordered_balls = 114)
  (white_balls_57 : white_balls = 57)
  (yellow_balls_57 : yellow_balls = 57)
  (ratio_condition : white_balls / (yellow_balls + x) = dispatch_error_ratio) :
  x = 49 :=
  sorry

end extra_yellow_balls_dispatched_eq_49_l511_511495


namespace train_pass_bridge_time_l511_511884

/-- A train is 460 meters long and runs at a speed of 45 km/h. The bridge is 140 meters long. 
Prove that the time it takes for the train to pass the bridge is 48 seconds. -/
theorem train_pass_bridge_time (train_length : ℝ) (bridge_length : ℝ) (speed_kmh : ℝ) 
  (h_train_length : train_length = 460) 
  (h_bridge_length : bridge_length = 140)
  (h_speed_kmh : speed_kmh = 45)
  : (train_length + bridge_length) / (speed_kmh * 1000 / 3600) = 48 := 
by
  sorry

end train_pass_bridge_time_l511_511884


namespace total_exercise_time_l511_511740

-- Definition of constants and speeds for each day
def monday_speed := 2 -- miles per hour
def wednesday_speed := 3 -- miles per hour
def friday_speed := 6 -- miles per hour
def distance := 6 -- miles

-- Function to calculate time given distance and speed
def time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Prove the total time spent in a week
theorem total_exercise_time :
  time distance monday_speed + time distance wednesday_speed + time distance friday_speed = 6 :=
by
  -- Insert detailed proof steps here
  sorry

end total_exercise_time_l511_511740


namespace part1_a_b_part2_estimation_part3_factory_choice_l511_511004

-- Definitions based on the given conditions
def factoryA_weights : List ℝ := [74, 74, 74, 75, 73, 77, 78, 72, 76, 77]
def factoryB_weights : List ℝ := [78, 74, 77, 73, 75, 75, 74, 74, 75, 75]

def mean (l: List ℝ) : ℝ := (l.sum) / (l.length.toFloat)
def median (l: List ℝ) : ℝ := 
  let sorted := l.sort
  if (sorted.length % 2 = 0) then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

def mode (l: List ℝ) : ℝ := 
  l.groupBy id
  |> List.sortBy (λ g => (-(g.length), g.head!.toFloat))
  |> List.head!.head!

def variance (l: List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / (l.length.toFloat)

-- Formulating the proof problems
theorem part1_a_b : median factoryB_weights = 75 ∧ mode factoryA_weights = 74 := by
  sorry

theorem part2_estimation : 
  let count := (factoryB_weights.filter (λ x => x = 75)).length
  (count / 10 * 100).toInt = 40 := by
  sorry

theorem part3_factory_choice : 
  variance factoryB_weights < variance factoryA_weights := by
  sorry

end part1_a_b_part2_estimation_part3_factory_choice_l511_511004


namespace equation_of_ellipse_l511_511123

theorem equation_of_ellipse :
  ∀ (a c b : ℝ),
  (a = Real.sqrt 2 * c) →
  (a - c = Real.sqrt 2 - 1) →
  (b^2 = a^2 - c^2) →
  (b = 1) →
  (∃ (equation : ℝ → ℝ → Prop), equation = λ x y, (y^2) / 2 + x^2 = 1) :=
by {
  intros a c b ha hc hb hb2,
  use λ x y, (y^2) / 2 + x^2 = 1,
  sorry
}

end equation_of_ellipse_l511_511123


namespace dr_strange_eats_12_days_l511_511791

noncomputable def num_ways_to_eat_items : ℕ := 2^12

theorem dr_strange_eats_12_days :
  num_ways_to_eat_items = 2048 :=
by
  def items := (1:ℕ) -- Definition to avoid undefined constants if needed in complex cases
  sorry

end dr_strange_eats_12_days_l511_511791


namespace equilateral_parallelogram_iff_rhombus_l511_511475

-- A figure is defined as an equilateral parallelogram if and only if it is a rhombus

variables {α : Type*} [EuclideanGeometry α] {P Q R S : α}
-- Define a parallelogram
def parallelogram (P Q R S : α) : Prop :=
  parallel P Q R S ∧ parallel Q R S P

-- Define an equilateral shape
def equilateral (P Q R S : α) : Prop :=
  dist P Q = dist Q R ∧ dist Q R = dist R S ∧ dist R S = dist S P

-- Define a rhombus (equilateral parallelogram)
def rhombus (P Q R S : α) : Prop :=
  parallelogram P Q R S ∧ equilateral P Q R S

theorem equilateral_parallelogram_iff_rhombus :
  (parallelogram P Q R S ∧ equilateral P Q R S) ↔ rhombus P Q R S :=
sorry

end equilateral_parallelogram_iff_rhombus_l511_511475


namespace one_third_recipe_quantities_l511_511024

-- Definitions for the initial quantities of flour and sugar
def flour_quantity : ℚ := 7 + 3/4
def sugar_quantity : ℚ := 2 + 1/2

-- The quantities after making one-third of the recipe
def flour_one_third : ℚ := flour_quantity / 3
def sugar_one_third : ℚ := sugar_quantity / 3

-- The expected results as given in the solution
def expected_flour : ℚ := 2 + 7/12
def expected_sugar : ℚ := 5/6

-- Theorem to prove the quantities match the expected values
theorem one_third_recipe_quantities :
  flour_one_third = expected_flour ∧ sugar_one_third = expected_sugar :=
by {
  -- Convert mixed numbers to improper fractions
  have h_flour : flour_quantity = 31/4, by norm_num [flour_quantity],
  have h_sugar : sugar_quantity = 5/2, by norm_num [sugar_quantity],

  -- Calculate one-third of the quantities
  have h_flour_one_third : flour_one_third = 31/12, by rw [h_flour]; ring,
  have h_sugar_one_third : sugar_one_third = 5/6, by rw [h_sugar]; ring,

  -- Convert fractions back to mixed numbers and compare to expected values
  have h_expected_flour : expected_flour = 31/12, by norm_num [expected_flour],
  have h_expected_sugar : expected_sugar = 5/6, by norm_num [expected_sugar],

  -- Prove the equalities
  exact ⟨by rw [h_flour_one_third, h_expected_flour], by rw [h_sugar_one_third, h_expected_sugar]⟩
}

end one_third_recipe_quantities_l511_511024


namespace number_of_true_statements_l511_511358

-- Given function f(x)
def f (x : ℝ) := sqrt 3 * (cos x)^2 + 2 * sin x * cos x - sqrt 3 * (sin x)^2

-- Definition of the symmetric axis
def symmetric_axis : Prop := f (π / 12) = f (π / 6 - π / 12) 

-- Definition for the symmetric property
def symmetric_property : Prop := ∀ x : ℝ, f (π / 3 + x) = -f (π / 3 - x)

-- Definition for translation property and checking it is not odd
def translation_property_is_not_odd : Prop :=
  ∀ x : ℝ, ¬ (2 * sin (2 * (x - π / 3) + π / 3) = -2 * sin (2 * x + π / 3))

-- Definition for the function value difference
def function_value_difference_at_least_four : Prop :=
  ∃ x1 x2 : ℝ, |f x1 - f x2| ≥ 4

-- Lean statement asserting the number of true statements is exactly 3
theorem number_of_true_statements :
  [symmetric_axis, symmetric_property, translation_property_is_not_odd, function_value_difference_at_least_four].count (λ p, p) = 3 := 
  by sorry

end number_of_true_statements_l511_511358


namespace acute_triangle_inequality_l511_511292

theorem acute_triangle_inequality
  {A B C : ℝ}
  {a b c : ℝ}
  (hA : a > 0) (hB : b > 0) (hC : c > 0)
  (hacute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (ha : a = sqrt (b^2 + c^2 - 2 * b * c * cos A))
  (hb : b = sqrt (a^2 + c^2 - 2 * a * c * cos B))
  (hc : c = sqrt (a^2 + b^2 - 2 * a * b * cos C))
  :
  4 * a * b * c < (a^2 + b^2 + c^2) * (a * cos A + b * cos B + c * cos C) ∧
  (a^2 + b^2 + c^2) * (a * cos A + b * cos B + c * cos C) ≤ (9 / 2) * a * b * c :=
by
  sorry

end acute_triangle_inequality_l511_511292


namespace solution_l511_511088

def is_prime (n : ℕ) : Prop := ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

noncomputable def find_pairs : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ is_prime (a * b^2 / (a + b)) ∧ ((a = 6 ∧ b = 2) ∨ (a = 2 ∧ b = 6))

theorem solution :
  find_pairs := sorry

end solution_l511_511088


namespace determine_m_for_divisibility_by_11_l511_511964

def is_divisible_by_11 (n : ℤ) : Prop :=
  n % 11 = 0

def sum_digits_odd_pos : ℤ :=
  8 + 6 + 2 + 8

def sum_digits_even_pos (m : ℤ) : ℤ :=
  5 + m + 4

theorem determine_m_for_divisibility_by_11 :
  ∃ m : ℤ, is_divisible_by_11 (sum_digits_odd_pos - sum_digits_even_pos m) ∧ m = 4 := 
by
  sorry

end determine_m_for_divisibility_by_11_l511_511964


namespace max_pies_without_ingredients_l511_511959

theorem max_pies_without_ingredients (total_pies half_chocolate two_thirds_marshmallows three_fifths_cayenne one_eighth_peanuts : ℕ) 
  (h1 : total_pies = 48) 
  (h2 : half_chocolate = total_pies / 2)
  (h3 : two_thirds_marshmallows = 2 * total_pies / 3) 
  (h4 : three_fifths_cayenne = 3 * total_pies / 5)
  (h5 : one_eighth_peanuts = total_pies / 8) : 
  ∃ pies_without_any_ingredients, pies_without_any_ingredients = 16 :=
  by 
    sorry

end max_pies_without_ingredients_l511_511959


namespace ruler_decree_l511_511892

noncomputable def market_supply (P : ℝ) : ℝ := 6 * P - 312

noncomputable def market_demand (P : ℝ) : ℝ := 688 - 4 * P

def tax_revenue (t Q_d : ℝ) : ℝ := Q_d * t

def max_tax_revenue (t : ℝ) : ℝ := 288 * t - 2.4 * t^2

theorem ruler_decree :
  let t_opt := 60 in
  max_tax_revenue t_opt = 8640 :=
by
  -- Proving the correctness of the maximum tax revenue
  have h1 : deriv (max_tax_revenue t_opt) = 288 - 4.8 * t_opt := by
    sorry
  -- Solving derivative equal to zero for critical point
  have h2 : t_opt = (288 / 4.8) := by
    sorry
  -- Substitute t_opt into max_tax_revenue function to get maximum revenue
  have h3 : max_tax_revenue t_opt = 288 * 60 - 2.4 * 60 ^ 2 := by
    sorry
  -- Confirm the final tax revenue equals 8640
  show max_tax_revenue 60 = 8640 by sorry

end ruler_decree_l511_511892


namespace james_total_spent_l511_511468

open_locale big_operators

noncomputable theory

def entry_fee : ℕ := 20
def drink_cost : ℕ := 6
def drinks_for_friends : ℕ := 2 * 5
def drinks_for_himself : ℕ := 6
def food_cost : ℕ := 14
def tip_rate : ℚ := 0.3

def total_drinks : ℕ := drinks_for_friends + drinks_for_himself

def drink_total_cost : ℕ := total_drinks * drink_cost

def food_and_drink_total : ℕ := drink_total_cost + food_cost

def tip : ℚ := food_and_drink_total * tip_rate

def total_cost_with_tip : ℚ := food_and_drink_total + tip

def final_cost : ℚ := total_cost_with_tip + entry_fee

theorem james_total_spent : final_cost = 163 := by 
  sorry

end james_total_spent_l511_511468


namespace t_range_t_value_l511_511666

-- Define the conditions
def circle_eq (t : ℝ) : Prop :=
  ∃ x y : ℝ, (x^2 + y^2 + (sqrt 3 * t + 1) * x + t * y + t^2 - 2 = 0)

def radius_3 (t : ℝ) : Prop :=
  let r := (1 / 2) * real.sqrt (2 * sqrt 3 * t + 9)
  in r = 3

-- Prove the statements based on the conditions
theorem t_range (t : ℝ) (h : circle_eq t) : 2 * sqrt 3 * t + 9 > 0 → t > - (3 * sqrt 3) / 2 := 
sorry

theorem t_value (t : ℝ) (h : radius_3 t) : t = (9 * sqrt 3) / 2 := 
sorry

end t_range_t_value_l511_511666


namespace chicken_leg_analysis_l511_511002

-- Define the weights from both factories
def weights_A : List ℝ := [74, 74, 74, 75, 73, 77, 78, 72, 76, 77]
def weights_B : List ℝ := [78, 74, 77, 73, 75, 75, 74, 74, 75, 75]

-- Define the conditions given in the problem
def mean_A : ℝ := 75
def median_A : ℝ := 74.5
def variance_A : ℝ := 3.4

def mean_B : ℝ := 75
def variance_B : ℝ := 2

-- Define the correct answers found in the solution.
def median_B : ℝ := 75
def mode_A : ℝ := 74
def estimation_B : ℝ := 40
def preferred_factory := "Factory B"

-- Lean statement to prove the calculated values and preference for Factory B
theorem chicken_leg_analysis :
  (∀ (weights : List ℝ), (weights = weights_B) ∧ (median weights = median_B) → (mode weights_A = mode_A) → 
  ((weights_B.filter (λ x, x = 75)).length / weights_B.length * 100 = estimation_B) →
  (variance_B < variance_A) → preferred_factory = "Factory B") := 
by
  intro weights h1 h2 h3 h4
  simp [median, mode, estimation_B, preferred_factory, h1, h2, h3, h4]
  sorry

end chicken_leg_analysis_l511_511002


namespace min_vector_magnitude_l511_511263

variables {V : Type*} [inner_product_space ℝ V]

def is_unit_vector (v : V) : Prop := ∥v∥ = 1

theorem min_vector_magnitude (a b c : V)
  (ha : is_unit_vector a)
  (hb : is_unit_vector b)
  (hc : is_unit_vector c)
  (hab : ⟪a, b⟫ = 0)
  (habc : ⟪a + c, b + c⟫ ≤ 0) :
  ∥a + b - c∥ ≥ sqrt 5 :=
sorry

end min_vector_magnitude_l511_511263


namespace exists_quadratic_g_l511_511096

noncomputable def quadratic_polynomial (a b c : ℝ) : (ℝ → ℝ) :=
  λ x, a * x ^ 2 + b * x + c

theorem exists_quadratic_g {f g : ℝ → ℝ} (hf : ∃ a b c : ℝ, f = quadratic_polynomial a b c)
    (hg : ∃ d e : ℝ, g = λ x, x * (x - d))
    (hr : ∀ x, f x * g x = 0 ↔ g (f x) = 0)
    (ha : ∃ a1 a2 a3 : ℝ, a2 - a1 = a3 - a2) :
    ∃ a : ℝ, f = λ x, (3 / (2 * a)) * (x - a) * (x - 2 * a) := 
sorry

end exists_quadratic_g_l511_511096


namespace max_slope_of_OQ_l511_511678

-- Assuming the conditions and problem setup
theorem max_slope_of_OQ :
  let C := {p : ℝ // p = 2}
  let equation_C := ∀ (x y : ℝ), y^2 = 4 * x
  let F : ℝ × ℝ := (1, 0)
  let O : ℝ × ℝ := (0, 0)
  let P := ∀ (m n : ℝ), (10 * m - 9, 10 * n)
  let Q := ∀ (m n : ℝ), (m, n)
  let K : ℝ := ∀ (m n : ℝ), (10 * n) / (25 * n^2 + 9)
  max_slope_of_line_OQ : ∃ (K_max : ℝ), K_max = 1 / 3 :=
  sorry

end max_slope_of_OQ_l511_511678


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511235

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let total_words := 6^5
  let words_without_vowels := 4^5
  total_words - words_without_vowels = 6752 :=
by
  let total_words := 6^5
  let words_without_vowels := 4^5
  have h_total_words : total_words = 7776 := by norm_num
  have h_words_without_vowels : words_without_vowels = 1024 := by norm_num
  calc
    7776 - 1024 = 6752 : by norm_num

end number_of_5_letter_words_with_at_least_one_vowel_l511_511235


namespace max_slope_of_OQ_l511_511676

-- Assuming the conditions and problem setup
theorem max_slope_of_OQ :
  let C := {p : ℝ // p = 2}
  let equation_C := ∀ (x y : ℝ), y^2 = 4 * x
  let F : ℝ × ℝ := (1, 0)
  let O : ℝ × ℝ := (0, 0)
  let P := ∀ (m n : ℝ), (10 * m - 9, 10 * n)
  let Q := ∀ (m n : ℝ), (m, n)
  let K : ℝ := ∀ (m n : ℝ), (10 * n) / (25 * n^2 + 9)
  max_slope_of_line_OQ : ∃ (K_max : ℝ), K_max = 1 / 3 :=
  sorry

end max_slope_of_OQ_l511_511676


namespace total_contribution_is_1040_l511_511939

-- Definitions of contributions based on conditions.
def Niraj_contribution : ℕ := 80
def Brittany_contribution : ℕ := 3 * Niraj_contribution
def Angela_contribution : ℕ := 3 * Brittany_contribution

-- Statement to prove that total contribution is $1040.
theorem total_contribution_is_1040 : Niraj_contribution + Brittany_contribution + Angela_contribution = 1040 := by
  sorry

end total_contribution_is_1040_l511_511939


namespace five_letter_words_with_vowel_l511_511244

-- Define the set of letters and identify vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels := {'A', 'E'}

-- Define the problem statement
theorem five_letter_words_with_vowel : 
  (number of 5-letter words with at least one vowel) = 6752 := 
sorry

end five_letter_words_with_vowel_l511_511244


namespace integral_solution_unique_l511_511363

theorem integral_solution_unique (x y z n : ℤ) 
  (h1 : xy + yz + zx = 3 * n ^ 2 - 1) 
  (h2 : x + y + z = 3 * n) 
  (h3 : x ≥ y)
  (h4 : y ≥ z) 
  : (x = n + 1 ∧ y = n ∧ z = n - 1) :=
begin
  sorry
end

end integral_solution_unique_l511_511363


namespace tenth_permutation_is_6785_l511_511839

def digits : List ℕ := [5, 6, 7, 8]

def tenth_permutation : ℕ := 6785

theorem tenth_permutation_is_6785 :
  let permutations := List.permutations digits
  let ordered_permutations := List.sort (≤) permutations
  let tenth := ordered_permutations.get! 9
  List.to_nat tenth = tenth_permutation :=
by
  sorry

end tenth_permutation_is_6785_l511_511839


namespace five_letter_words_with_vowel_l511_511239

-- Define the set of letters and identify vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels := {'A', 'E'}

-- Define the problem statement
theorem five_letter_words_with_vowel : 
  (number of 5-letter words with at least one vowel) = 6752 := 
sorry

end five_letter_words_with_vowel_l511_511239


namespace sample_size_calculation_l511_511735

theorem sample_size_calculation 
  (total_male_students : ℕ)
  (total_female_students : ℕ)
  (sampled_male_students : ℕ)
  (sample_size : ℕ) : 
  total_male_students = 310 → 
  total_female_students = 290 → 
  sampled_male_students = 31 → 
  sample_size * total_male_students = 31 * (total_male_students + total_female_students) →
  sample_size = 60 := by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  repeat {rw [show 31 = sampled_male_students, from h3]} at h4
  have h : sample_size * 310 = 31 * 600, { exact h4 }
  sorry

end sample_size_calculation_l511_511735


namespace leg_ratio_of_right_triangle_l511_511872

theorem leg_ratio_of_right_triangle (a b c m : ℝ) (h1 : a ≤ b)
  (h2 : a * b = c * m) (h3 : c^2 = a^2 + b^2) (h4 : a^2 + m^2 = b^2) :
  (a / b) = Real.sqrt ((-1 + Real.sqrt 5) / 2) :=
sorry

end leg_ratio_of_right_triangle_l511_511872


namespace curve_is_line_l511_511544

theorem curve_is_line (r θ : ℝ) (h : r = 1 / (2 * real.sin θ - real.cos θ)) :
  ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ ∀ x y : ℝ, (∃ (r θ : ℝ), x = r * real.cos θ ∧ y = r * real.sin θ ∧ r = 1 / (2 * real.sin θ - real.cos θ)) → a * x + b * y + c = 0 :=
sorry

end curve_is_line_l511_511544


namespace P_ne_1_3_5_7_9_l511_511756

noncomputable def P (x : ℤ) : ℤ := sorry -- The actual polynomial P(x) is defined with integer coefficients

theorem P_ne_1_3_5_7_9 (x : ℤ) (a_k a_0 : ℤ) (x₁ x₂ x₃ x₄ : ℤ)
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)
  (h_roots : P x₁ = 2 ∧ P x₂ = 2 ∧ P x₃ = 2 ∧ P x₄ = 2)
  (h_P : x ∈ {1, 3, 5, 7, 9} → P x ≠ x) : Prop :=
  ∀ x : ℤ, P x ≠ 1 ∧ P x ≠ 3 ∧ P x ≠ 5 ∧ P x ≠ 7 ∧ P x ≠ 9

-- Proof of the theorem will be provided here
sorry

end P_ne_1_3_5_7_9_l511_511756


namespace fraction_power_simplification_l511_511950

theorem fraction_power_simplification:
  (81000/9000)^3 = 729 → (81000^3) / (9000^3) = 729 :=
by 
  intro h
  rw [<- h]
  sorry

end fraction_power_simplification_l511_511950


namespace largest_n_l511_511293

theorem largest_n (n : ℕ) (h_pos : 0 < n) (h_odd : n % 2 = 1) :
  (∃ (b w : ℕ), b + w = n^2 ∧ 
    b + (n^2 / 2).floor ∈ finset.range 2009) → 
  n ≤ 63 :=
by sorry

end largest_n_l511_511293


namespace hyperbola_no_common_points_l511_511320

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) : ℝ :=
  real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_no_common_points (a b e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_ecc : e = real.sqrt (1 + (b^2 / a^2)))
  (h_slope : b / a < 2) :
  e = 2 :=
sorry

end hyperbola_no_common_points_l511_511320


namespace hyperbola_no_intersection_l511_511323

theorem hyperbola_no_intersection (a b e : ℝ)
  (ha : 0 < a) (hb : 0 < b)
  (h_e : e = (Real.sqrt (a^2 + b^2)) / a) :
  (√5 ≥ e ∧ 1 < e) → ∀ x y : ℝ, ¬ (y = 2 * x ∧ (x^2 / a^2 - y^2 / b^2 = 1)) :=
begin
  intros h_intersect x y,
  sorry,
end

end hyperbola_no_intersection_l511_511323


namespace discount_per_coupon_l511_511810

-- Definitions and conditions from the problem
def num_cans : ℕ := 9
def cost_per_can : ℕ := 175 -- in cents
def num_coupons : ℕ := 5
def total_payment : ℕ := 2000 -- $20 in cents
def change_received : ℕ := 550 -- $5.50 in cents
def amount_paid := total_payment - change_received

-- Mathematical proof problem
theorem discount_per_coupon :
  let total_cost_without_coupons := num_cans * cost_per_can 
  let total_discount := total_cost_without_coupons - amount_paid
  let discount_per_coupon := total_discount / num_coupons
  discount_per_coupon = 25 :=
by
  sorry

end discount_per_coupon_l511_511810


namespace range_of_m_l511_511998

-- Defining conditions to describe the problem setting
def condition_p (m : ℝ) : Prop :=
  x - 2 * y + 3 = 0 → y^2 = m * x → (4 * m^2 - 12 * m < 0)

def condition_q (m : ℝ) : Prop :=
  m ≠ 0 → (5 - 2 * m) * m < 0

-- The proof statement
theorem range_of_m (m : ℝ) : (condition_p m ∨ condition_q m) ∧ ¬(condition_p m ∧ condition_q m) → 
  m < 0 ∨ (0 < m ∧ m <= 5/2) ∨ m ≥ 3 :=
sorry

end range_of_m_l511_511998


namespace largest_five_digit_product_120_l511_511551

theorem largest_five_digit_product_120 : 
  ∃ n : ℕ, n = 85311 ∧ (nat.digits 10 n).product = 120 ∧ 10000 ≤ n ∧ n < 100000 :=
by
  sorry

end largest_five_digit_product_120_l511_511551


namespace five_letter_words_with_vowels_l511_511220

/-
How many 5-letter words with at least one vowel can be constructed from the letters 
A, B, C, D, E, and F? (Note that A and E are vowels, any word is valid, not just English language 
words, and letters may be used more than once.)
-/

theorem five_letter_words_with_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := ['A', 'E'],
      consonants := ['B', 'C', 'D', 'F'] in
  let total_words := 6 ^ 5,
      consonant_only_words := 4 ^ 5,
      at_least_one_vowel_words := total_words - consonant_only_words in
  at_least_one_vowel_words = 6752 :=
by
  intro letters vowels consonants total_words consonant_only_words at_least_one_vowel_words
  sorry

end five_letter_words_with_vowels_l511_511220


namespace hexagon_sum_balanced_assignment_exists_l511_511435

-- Definitions based on the conditions
def is_valid_assignment (a b c d e f g : ℕ) : Prop :=
a + b + g = a + c + g ∧ a + b + g = a + d + g ∧ a + b + g = a + e + g ∧
a + b + g = b + c + g ∧ a + b + g = b + d + g ∧ a + b + g = b + e + g ∧
a + b + g = c + d + g ∧ a + b + g = c + e + g ∧ a + b + g = d + e + g

-- The theorem we want to prove
theorem hexagon_sum_balanced_assignment_exists :
  ∃ (a b c d e f g : ℕ), 
  (a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 2 ∨ b = 3 ∨ b = 5) ∧ 
  (c = 2 ∨ c = 3 ∨ c = 5) ∧ 
  (d = 2 ∨ d = 3 ∨ d = 5) ∧ 
  (e = 2 ∨ e = 3 ∨ e = 5) ∧
  (f = 2 ∨ f = 3 ∨ f = 5) ∧
  (g = 2 ∨ g = 3 ∨ g = 5) ∧
  is_valid_assignment a b c d e f g :=
sorry

end hexagon_sum_balanced_assignment_exists_l511_511435


namespace find_monotonic_intervals_max_min_on_interval_l511_511670

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

noncomputable def f' (x : ℝ) : ℝ := (Real.cos x - Real.sin x) * Real.exp x - 1

theorem find_monotonic_intervals (k : ℤ) : 
  ((2 * k * Real.pi - Real.pi < x ∧ x < 2 * k * Real.pi) → 0 < (f' x)) ∧
  ((2 * k * Real.pi < x ∧ x < 2 * k * Real.pi + Real.pi) → (f' x) < 0) :=
sorry

theorem max_min_on_interval : 
  (∀ x, 0 ≤ x ∧ x ≤ (2 * Real.pi / 3) → f 0 = 1 ∧ f (2 * Real.pi / 3) =  -((1/2) * Real.exp (2/3 * Real.pi)) - (2 * Real.pi / 3)) :=
sorry

end find_monotonic_intervals_max_min_on_interval_l511_511670


namespace volume_of_wedge_l511_511029

-- Conditions
def sphere_circumference (r : ℝ) : ℝ :=
  2 * Real.pi * r

def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * (r^3)

def volume_of_one_wedge (V : ℝ) : ℝ :=
  V / 6

-- Main Theorem
theorem volume_of_wedge (radius : ℝ)
  (h1 : sphere_circumference radius = 18 * Real.pi) :
  volume_of_one_wedge (volume_of_sphere radius) = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l511_511029


namespace partition_medals_l511_511101

open Real

noncomputable theory

def bronze_medals : Π (n : ℕ), nat → ℕ := sorry

theorem partition_medals :
  ∃ (A B : finset ℕ), 
    A.card = 50 ∧ B.card = 50 ∧
    A ∪ B = {1, 2, ..., 100} ∧
    ∀ (i j : ℕ) (hi : i ∈ A) (hj : j ∈ B), 
      |(bronze_medals i) - (bronze_medals j)| ≤ 202 :=
sorry

end partition_medals_l511_511101


namespace remainder_when_divided_by_2_l511_511889

theorem remainder_when_divided_by_2 (n : ℕ) (h₁ : n > 0) (h₂ : (n + 1) % 6 = 4) : n % 2 = 1 :=
by sorry

end remainder_when_divided_by_2_l511_511889


namespace player_A_wins_after_4_rounds_geometric_sequence_differences_find_P_2_l511_511869

-- Define probabilities of shots
def shooting_probability_A : ℝ := 0.5
def shooting_probability_B : ℝ := 0.6

-- Define initial points for questions
def initial_points_question_1 : ℝ := 0
def initial_points_question_2 : ℝ := 2

-- Given initial probabilities
def P_0 : ℝ := 0
def P_4 : ℝ := 1

-- Probability that player A wins after exactly 4 rounds
def probability_A_wins_after_4_rounds : ℝ :=
  let P_A := shooting_probability_A * (1 - shooting_probability_B)
  let P_B := shooting_probability_B * (1 - shooting_probability_A)
  let P_C := 1 - P_A - P_B
  P_A * P_C^2 * P_A + P_A * P_B * P_A^2

-- Define the probabilities P(i) for i=0..4
def P (i : ℕ) : ℝ := sorry -- Placeholder for the function

-- Define the proof problem
theorem player_A_wins_after_4_rounds : probability_A_wins_after_4_rounds = 0.0348 :=
sorry

theorem geometric_sequence_differences :
  ∀ i : ℕ, i < 4 → (P (i + 1) - P i) / (P (i + 2) - P (i + 1)) = 2/3 :=
sorry

theorem find_P_2 : P 2 = 4/13 :=
sorry

end player_A_wins_after_4_rounds_geometric_sequence_differences_find_P_2_l511_511869


namespace custom_mul_expansion_l511_511636

variable {a b x y : ℝ}

def custom_mul (a b : ℝ) : ℝ := (a - b)^2

theorem custom_mul_expansion (x y : ℝ) : custom_mul (x^2) (y^2) = (x + y)^2 * (x - y)^2 := by
  sorry

end custom_mul_expansion_l511_511636


namespace total_amount_received_is_correct_l511_511020

def purchase_price1 : ℝ := 600
def purchase_price2 : ℝ := 800
def purchase_price3 : ℝ := 1000

def loss_percentage1 : ℝ := 20
def loss_percentage2 : ℝ := 25
def loss_percentage3 : ℝ := 30

def loss_amount (purchase_price: ℝ) (loss_percentage: ℝ) : ℝ :=
  (loss_percentage / 100) * purchase_price

def selling_price (purchase_price: ℝ) (loss_percentage: ℝ) : ℝ :=
  purchase_price - loss_amount purchase_price loss_percentage

def total_received : ℝ :=
  selling_price purchase_price1 loss_percentage1 +
  selling_price purchase_price2 loss_percentage2 +
  selling_price purchase_price3 loss_percentage3

theorem total_amount_received_is_correct :
  total_received = 1780 := by
  sorry

end total_amount_received_is_correct_l511_511020


namespace five_letter_words_with_vowel_l511_511199

theorem five_letter_words_with_vowel : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let n := 5 
  (∃ (w : list(char)), w.length = n ∧ ∀ (i : fin n), w[i] ∈ letters ∧ (∃ (j : fin n), w[j] ∈ vowels)) → 
  (6^5 - 4^5 = 6752) := 
by
  sorry

end five_letter_words_with_vowel_l511_511199


namespace sum_of_squares_is_42_l511_511856

variables (D T H : ℕ)

theorem sum_of_squares_is_42
  (h1 : 3 * D + T = 2 * H)
  (h2 : 2 * H^3 = 3 * D^3 + T^3)
  (coprime : Nat.gcd (Nat.gcd D T) H = 1) :
  (T^2 + D^2 + H^2 = 42) :=
sorry

end sum_of_squares_is_42_l511_511856


namespace Mrs_Amaro_roses_l511_511777

theorem Mrs_Amaro_roses :
  ∀ (total_roses red_roses yellow_roses pink_roses remaining_roses white_and_purple white_roses purple_roses : ℕ),
    total_roses = 500 →
    5 * total_roses % 8 = 0 →
    red_roses = total_roses * 5 / 8 →
    yellow_roses = (total_roses - red_roses) * 1 / 8 →
    pink_roses = (total_roses - red_roses) * 2 / 8 →
    remaining_roses = total_roses - red_roses - yellow_roses - pink_roses →
    remaining_roses % 2 = 0 →
    white_roses = remaining_roses / 2 →
    purple_roses = remaining_roses / 2 →
    red_roses + white_roses + purple_roses = 430 :=
by
  intros total_roses red_roses yellow_roses pink_roses remaining_roses white_and_purple white_roses purple_roses
  intro total_roses_eq
  intro red_roses_divisible
  intro red_roses_def
  intro yellow_roses_def
  intro pink_roses_def
  intro remaining_roses_def
  intro remaining_roses_even
  intro white_roses_def
  intro purple_roses_def
  sorry

end Mrs_Amaro_roses_l511_511777


namespace ramesh_share_l511_511454

variable (SureshInvestment RameshInvestment TotalProfit : ℝ)

def SureshInvestment := 24000
def RameshInvestment := 40000
def TotalProfit := 19000

theorem ramesh_share : (RameshInvestment / (SureshInvestment + RameshInvestment)) * TotalProfit = 11875 := by
sorry

end ramesh_share_l511_511454


namespace range_of_a_opposite_sides_l511_511138

theorem range_of_a_opposite_sides (a : ℝ) :
  (3 * (-2) - 2 * 1 - a) * (3 * 1 - 2 * 1 - a) < 0 ↔ -8 < a ∧ a < 1 := by
  sorry

end range_of_a_opposite_sides_l511_511138


namespace parallel_lines_solution_l511_511691

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a = 0 → (x + 2 * a * y - 1 = 0 ∧ (2 * a - 1) * x - a * y - 1 = 0) → (x = y)) ∨ 
  (∀ x y : ℝ, a = 1/4 → (x + 2 * a * y - 1 = 0 ∧ (2 * a - 1) * x - a * y - 1 = 0) → (x = y)) :=
sorry

end parallel_lines_solution_l511_511691


namespace five_letter_words_with_vowel_l511_511251

theorem five_letter_words_with_vowel :
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  total_words - words_no_vowel = 6752 :=
by
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  have h1 : total_words = 7776 := by norm_num
  have h2 : words_no_vowel = 1024 := by norm_num
  show total_words - words_no_vowel = 6752
  calc
    total_words - words_no_vowel
    = 7776 - 1024 : by rw [h1, h2]
    ... = 6752 : by norm_num

end five_letter_words_with_vowel_l511_511251


namespace area_of_triangle_BED_l511_511045

noncomputable def area (T : Triangle) : ℝ := sorry

structure Triangle :=
(A B C : Point)

structure Point :=
(x y : ℝ)

def midpoint (A B : Point) : Point :=
{ x := (A.x + B.x) / 2,
  y := (A.y + B.y) / 2 }

def is_perpendicular (A B C : Point) : Prop :=
(A.x - B.x) * (A.y - C.y) + (A.x - C.x) * (A.y - B.y) = 0

theorem area_of_triangle_BED
  (ABC : Triangle)
  (A B C M D E : Point)
  (h1 : ABC.A = A)
  (h2 : ABC.B = B)
  (h3 : ABC.C = C)
  (h_midpoint : midpoint A B = M)
  (h_perpendicular_MD : is_perpendicular M D C)
  (h_perpendicular_EC : is_perpendicular E C M)
  (h_area_ABC : area ABC = 24) :
  area (Triangle.mk B E D) = 12 :=
sorry

end area_of_triangle_BED_l511_511045


namespace count_5_letter_words_with_at_least_one_vowel_l511_511201

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := (letters.card) ^ 5

def non_vowel_letters : Finset Char := letters \ vowels

def total_non_vowel_5_letter_words : ℕ := (non_vowel_letters.card) ^ 5

theorem count_5_letter_words_with_at_least_one_vowel :
  total_5_letter_words - total_non_vowel_5_letter_words = 6752 :=
by
  sorry

end count_5_letter_words_with_at_least_one_vowel_l511_511201


namespace percent_gain_is_4_58_l511_511488

-- Definitions based on the conditions
def initial_sheep := 1024
def sold_sheep := 980
def remaining_sheep := initial_sheep - sold_sheep
def cost_per_sheep (c : ℝ) : ℝ := c
def total_cost (c : ℝ) := initial_sheep * cost_per_sheep c

-- Derived concepts
def price_per_sheep (c : ℝ) := total_cost c / sold_sheep
def revenue_sold_sheep (c : ℝ) := sold_sheep * price_per_sheep c
def revenue_remaining_sheep (c : ℝ) := remaining_sheep * price_per_sheep c
def total_revenue (c : ℝ) := revenue_sold_sheep c + revenue_remaining_sheep c
def profit (c : ℝ) := total_revenue c - total_cost c
def percentage_gain (c : ℝ) := (profit c / total_cost c) * 100

-- Assertion based on the final answer to be proven
theorem percent_gain_is_4_58 (c : ℝ) : percentage_gain c ≈ 4.58 := sorry

end percent_gain_is_4_58_l511_511488


namespace total_contribution_l511_511937

theorem total_contribution : 
  ∀ (Niraj_contribution : ℕ) (Brittany_contribution Angela_contribution : ℕ),
    (Brittany_contribution = 3 * Niraj_contribution) →
    (Angela_contribution = 3 * Brittany_contribution) →
    (Niraj_contribution = 80) →
    (Niraj_contribution + Brittany_contribution + Angela_contribution = 1040) :=
  by assumption sorry

end total_contribution_l511_511937


namespace prob_train_or_airplane_prob_not_ship_l511_511924

variables (P : Set (Set ℝ)) -- Type for probability spaces

-- Define the probabilities for each means of transportation
def P_T : ℝ := 0.3
def P_S : ℝ := 0.1
def P_C : ℝ := 0.2
def P_A : ℝ := 0.4

-- Proving the two statements:
-- 1. Probability of going by train or airplane
theorem prob_train_or_airplane : P_T + P_A = 0.7 := 
sorry

-- 2. Probability of not going by ship
theorem prob_not_ship : 1 - P_S = 0.9 := 
sorry

end prob_train_or_airplane_prob_not_ship_l511_511924


namespace return_trip_time_l511_511486

-- Define the main parameters
variables (d : ℝ) (p : ℝ) (w : ℝ) (h : ℝ)

-- Define conditions as hypotheses
def condition_1 := d / (p - w) = 120
def condition_2 := (d / (p + w) = (d / (1.1 * p)) - 15)

-- Define the proof problem statement
theorem return_trip_time (h1 : condition_1) (h2 : condition_2) : 
  d / (p + w) = 19 ∨ d / (p + w) = 98 :=
sorry

end return_trip_time_l511_511486


namespace five_letter_words_with_vowels_l511_511174

noncomputable def num_5_letter_words_with_vowels : Nat := 7776 - 1024

theorem five_letter_words_with_vowels
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'})
  (len : Nat := 5) :
  (letters.card ^ len) - ((letters.filter (λ c, ¬ c ∈ vowels)).card ^ len) = 6752 := by
  sorry

end five_letter_words_with_vowels_l511_511174


namespace find_number_l511_511021

theorem find_number (x : ℝ) (h : 20 * (x / 5) = 40) : x = 10 :=
by
  sorry

end find_number_l511_511021


namespace gcd_probability_is_one_l511_511418

open Set Nat

theorem gcd_probability_is_one :
  let S := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := (finset.powerset_len 2 (finset.image id S.to_finset)).card
  let non_rel_prime_pairs := 6
  (finset.card (finset.filter (λ (p : Finset ℕ), p.gcdₓ = 1) 
                                (finset.powerset_len 2 (finset.image id S.to_finset)))) / 
  total_pairs = 11 / 14 :=
sorry

end gcd_probability_is_one_l511_511418


namespace membership_percentage_change_l511_511480

noncomputable def membership_change (initial : ℝ) : ℝ :=
let after_fall := initial * 1.06 in
let after_winter := after_fall * 1.10 in
let after_spring := after_winter * 0.81 in
let after_summer := after_spring * 1.05 in
after_summer

theorem membership_percentage_change : 
  ∃ change : ℝ, change = (membership_change 100 - 100) / 100 * 100 ∧ abs (change + 0.8317) < 0.001 :=
sorry

end membership_percentage_change_l511_511480


namespace solution_set_inequality_l511_511395

theorem solution_set_inequality (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x > 3 / 2 ∨ x < -1 := 
by sorry

end solution_set_inequality_l511_511395


namespace peanuts_calculation_l511_511800

theorem peanuts_calculation (initial_peanuts : ℕ) (peanuts_lost : ℕ) (bags : ℕ) (peanuts_per_bag : ℕ) :
  initial_peanuts = 2650 →
  peanuts_lost = 1379 →
  bags = 4 →
  peanuts_per_bag = 450 →
  initial_peanuts - peanuts_lost + bags * peanuts_per_bag = 3071 :=
by
  intros h_initial h_lost h_bags h_per_bag
  rw [h_initial, h_lost, h_bags, h_per_bag]
  sorry

end peanuts_calculation_l511_511800


namespace general_formula_smallest_integer_k_l511_511663

def sequence_sum (n : ℕ) : ℕ := 
  finset.sum (finset.range n) (λ i, 2^i * (i + 1))

axiom sequence_condition (n : ℕ) : sequence_sum n = (n-1) * 2^n + 1

def sequence_a (n : ℕ) : ℕ := n

def fraction_sum (n : ℕ) : ℝ := 
  finset.sum (finset.range n) (λ i, (i + 1 : ℝ) / (2 ^ (i + 1)))

theorem general_formula (n : ℕ) : sequence_a n = n := 
  sorry 

theorem smallest_integer_k : ∃ k : ℕ, k > fraction_sum 2023 ∧ (∀ m : ℕ, m > fraction_sum 2023 → k ≤ m) :=
  by {
    use 2,
    -- Complete the proof here for proper Lean formalization
    sorry
  }

end general_formula_smallest_integer_k_l511_511663


namespace locus_is_ellipse_perimeter_constant_l511_511140

noncomputable def locus_equation (x y : ℝ) : Prop :=
  ((x - 3)^2 + y^2) / (abs (x - 25 / 3))^2 = (3 / 5)^2

theorem locus_is_ellipse :
  ∀ x y : ℝ, locus_equation x y ↔ (x^2 / 25 + y^2 / 16 = 1) := 
by sorry

theorem perimeter_constant (k m : ℝ) (tangent : ∀ x y : ℝ, x^2 + y^2 = 16 → y = k * x + m) :
  ∀ x1 y1 x2 y2 : ℝ, 
    (x1^2 / 25 + y1^2 / 16 = 1) →
    (x2^2 / 25 + y2^2 / 16 = 1) →
    let A : ℝ × ℝ := (x1, y1);
    let B : ℝ × ℝ := (x2, y2);
    (AB : ℝ := (A.1 - B.1)^2 + (A.2 - B.2)^2) in
    let F : ℝ × ℝ := (3, 0) in
    let FA : ℝ := (F.1 - A.1)^2 + (F.2 - A.2)^2 in
    let FB : ℝ := (F.1 - B.1)^2 + (F.2 - B.2)^2 in
    FA.sqrt + FB.sqrt + AB.sqrt = 10 :=
by sorry

end locus_is_ellipse_perimeter_constant_l511_511140


namespace sum_of_differences_l511_511616

/-- Proving that the sum of all differences (first digit - last digit) for natural numbers from 
    1 to 999 is 495. -/ 
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, 
    let str := n.toString
    let first := if str.length > 1 then str.head!.toNat - '0'.toNat else 0
    let last := if str.length > 1 then str.getLast.toNat - '0'.toNat else 0
    first - last
  ) = 495 := 
by
  sorry

end sum_of_differences_l511_511616


namespace vector_subtraction_correct_l511_511054

theorem vector_subtraction_correct:
  let v1 := (3, -4) -- Vector 1
  let v2 := (2, 6)  -- Vector 2
  let scalar := 3   -- Scalar
  (v1.1 - scalar * v2.1, v1.2 - scalar * v2.2) = (-3, -22) :=
by {
  -- Define the vectors and scalar
  let v1 : ℤ × ℤ := (3, -4)
  let v2 : ℤ × ℤ := (2, 6)
  let scalar : ℤ := 3

  -- Perform the scalar multiplication and subtraction
  have step1 : ℤ × ℤ := (scalar * v2.1, scalar * v2.2)
  have step2 : ℤ × ℤ := (v1.1 - step1.1, v1.2 - step1.2)

  -- Assert and finalize the proof
  show step2 = (-3, -22), by sorry
}

end vector_subtraction_correct_l511_511054


namespace unchanged_number_in_position_100_l511_511506

theorem unchanged_number_in_position_100 :
  ∀ (f : Fin 1982 → ℕ) (l : List (Fin 1982)) (n : ℕ),
    (∀ i j, i < j → f (l.nth_le i sorry) ≤ f (l.nth_le j sorry)) ∧ 
    (∀ i j, i > j → f (l.nth_le i sorry) ≥ f (l.nth_le j sorry)) ∧
    l.length = 1982 ∧ l.nth 99 = n ∧ f (l.nth_le 99 sorry) = n → 
    n = 100 := by
  sorry

end unchanged_number_in_position_100_l511_511506


namespace range_of_ab_l511_511094

noncomputable def circle_equation (x y : ℝ) : Prop := (x^2 + y^2 + 2*x - 4*y + 1 = 0)

noncomputable def line_equation (a b x y : ℝ) : Prop := (2*a*x - b*y - 2 = 0)

def symmetric_with_respect_to (center_x center_y a b : ℝ) : Prop :=
  line_equation a b center_x center_y  -- check if the line passes through the center

theorem range_of_ab (a b : ℝ) (h_symm : symmetric_with_respect_to (-1) 2 a b) : 
  ∃ ab_max : ℝ, ab_max = 1/4 ∧ ∀ ab : ℝ, ab = (a * b) → ab ≤ ab_max :=
sorry

end range_of_ab_l511_511094


namespace total_seats_in_theater_l511_511496

def theater_charges_adults : ℝ := 3.0
def theater_charges_children : ℝ := 1.5
def total_income : ℝ := 510
def number_of_children : ℕ := 60

theorem total_seats_in_theater :
  ∃ (A C : ℕ), C = number_of_children ∧ theater_charges_adults * A + theater_charges_children * C = total_income ∧ A + C = 200 :=
by
  sorry

end total_seats_in_theater_l511_511496


namespace ratio_simplified_l511_511284

def number_red_balls : ℕ := 16
def number_white_balls : ℕ := 20

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem ratio_simplified : 
  (number_red_balls / gcd number_red_balls number_white_balls : ℕ) / 
  (number_white_balls / gcd number_red_balls number_white_balls : ℕ) = 4 / 5 :=
by 
  sorry

end ratio_simplified_l511_511284


namespace toys_produced_each_day_l511_511452

-- Define the conditions
def total_weekly_production : ℕ := 8000
def days_worked_per_week : ℕ := 4
def daily_production : ℕ := total_weekly_production / days_worked_per_week

-- The statement to be proved
theorem toys_produced_each_day : daily_production = 2000 := sorry

end toys_produced_each_day_l511_511452


namespace min_f_x_l511_511957

/-- Consider the function f(x) = x + 1/x + 1/sqrt(x + 1/x) for x > 0.
    This theorem states that the minimum value of f(x) is 2 + 1/sqrt(2). -/
theorem min_f_x (x : ℝ) (hx : x > 0) : 
  (∀ y > 0, (y + 1/y + 1/Real.sqrt(y + 1/y)) ≥ 2 + 1/Real.sqrt(2)) :=
sorry

end min_f_x_l511_511957


namespace find_m_value_l511_511665

noncomputable def imaginary_unit := complex.I

theorem find_m_value (m : ℝ) :
  let z := (m / (1 - imaginary_unit) + (1 - imaginary_unit) / 2 * imaginary_unit) in
  (z.re + z.im = 1) -> m = 1 := by
  -- Here we assume the conditions on z
  sorry

end find_m_value_l511_511665


namespace intersection_A_B_l511_511131

variable (x : ℝ)

def setA : Set ℝ := { x | x^2 - 4*x - 5 < 0 }
def setB : Set ℝ := { x | -2 < x ∧ x < 2 }

theorem intersection_A_B :
  { x | x^2 - 4*x - 5 < 0 } ∩ { x | -2 < x ∧ x < 2 } = { x | -1 < x ∧ x < 2 } :=
by
  -- Here would be the proof, but we use sorry to skip it
  sorry

end intersection_A_B_l511_511131


namespace number_of_BMWs_sold_l511_511913

theorem number_of_BMWs_sold (total_cars : ℕ) (audi_percent toyota_percent acura_percent : ℕ) :
  total_cars = 500 →
  audi_percent = 20 →
  toyota_percent = 25 →
  acura_percent = 15 →
  ∃ (bmw_cars : ℕ), bmw_cars = 200 :=
by
  intros htotal haudi htoyota hacura
  let used_percent := audi_percent + toyota_percent + acura_percent
  have hused_percent : used_percent = 60 := by
    rw [haudi, htoyota, hacura]
    exact nat.add_assoc 20 25 15 60
  let bmw_percent := 100 - used_percent
  have hbmw_percent : bmw_percent = 40 := by
    rw [hused_percent]
    exact nat.sub_self 60 100 40
  let bmw_cars := total_cars * bmw_percent / 100
  have hbmw_cars : bmw_cars = 200 := by
    rw [htotal, hbmw_percent]
    exact nat.div_self 500 40 100 200
  existsi 200
  exact hbmw_cars

end number_of_BMWs_sold_l511_511913


namespace count_perfect_squares_mul_36_l511_511256

theorem count_perfect_squares_mul_36 (n : ℕ) (h1 : n < 10^7) (h2 : ∃k, n = k^2) (h3 : 36 ∣ n) :
  ∃ m : ℕ, m = 263 :=
by
  sorry

end count_perfect_squares_mul_36_l511_511256


namespace johns_average_speed_l511_511302

-- Define the given distances and speeds.
def distance1 : ℝ := 45
def speed1 : ℝ := 15
def distance2 : ℝ := 15
def speed2 : ℝ := 45

-- Define the total distance.
def total_distance : ℝ := distance1 + distance2

-- Define the times for each segment.
def time1 : ℝ := distance1 / speed1
def time2 : ℝ := distance2 / speed2

-- Define the total time.
def total_time : ℝ := time1 + time2

-- Define the average speed.
def average_speed : ℝ := total_distance / total_time

-- Prove that the average speed is 18 miles per hour.
theorem johns_average_speed : average_speed = 18 := by
  -- Sorry is used to skip the actual proof.
  sorry

end johns_average_speed_l511_511302


namespace average_words_per_hour_l511_511500

theorem average_words_per_hour
  (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ)
  (h1 : total_words = 55000)
  (h2 : total_hours = 120)
  (h3 : break_hours = 20) :
  (total_words / (total_hours - break_hours)) = 550 :=
by
  -- Definitions based on conditions
  have h_actual_hours : total_hours - break_hours = 100 := by linarith [h2, h3]
  rw [h1, h_actual_hours]
  norm_num
  exact h_actual_hours

end average_words_per_hour_l511_511500


namespace part1_part2_l511_511750

def f (x : ℝ) : ℝ := x / (1 + x^2)

-- condition for part 1
axiom a_cond (a : ℝ) : 0 < a ∧ a < 1

-- part (1) statement
theorem part1 (a : ℝ) (h : a_cond a) : f(a) + f(1 - a) = 1 := 
  sorry

-- part (2) statement
theorem part2 : ∑ n in finset.range 500, f (1 / (n + 1)) = 500 := 
  sorry

end part1_part2_l511_511750


namespace noah_total_bill_l511_511783

def call_duration := 30 -- in minutes
def charge_per_minute := 0.05 -- in dollars per minute
def calls_per_week := 1 -- calls per week
def weeks_per_year := 52 -- weeks per year

theorem noah_total_bill:
  (calls_per_week * weeks_per_year * call_duration * charge_per_minute) = 78 :=
by
  sorry

end noah_total_bill_l511_511783


namespace sum_of_differences_l511_511603

theorem sum_of_differences : 
  let first_digit (n : ℕ) : ℕ := n / 10^(nat.log10 n)
  let last_digit (n : ℕ) : ℕ := n % 10
  (finset.range 1000).sum (λ n, first_digit n - last_digit n) = 495 :=
by
  sorry

end sum_of_differences_l511_511603


namespace parabola_distance_problem_l511_511118

noncomputable def distance_from_origin_to_point : ℝ :=
  let p := 4
  let x := 2
  let y0 := real.sqrt(8)
  let distance := real.sqrt(x^2 + y0^2)
  distance

theorem parabola_distance_problem
  (x : ℝ)
  (y0 : ℝ)
  (p : ℝ)
  (h1 : y0^2 = 4 * x)
  (h2 : x = 2)
  (h3 : p = 4)
  (h4 : real.dist (2, y0) (p / 2, 0) = 3) :
  real.dist (0,0) (x, y0) = 2 * real.sqrt 3 :=
by {
  rw [h2, h3] at h1,
  unfold real.dist at h4,
  sorry
}

end parabola_distance_problem_l511_511118


namespace product_of_segments_constant_l511_511119

-- Let O be a point inside the circle
variables {α : Type*} [euclidean_geometry α] (O : α)
-- Let A, B, C, D be points on the circle
variables (A B C D : α)
-- Let AC and BD be chords intersecting at O
variable (AC BD : set α)
-- Let h1 and h2 denote the segment lengths
variable (AO OC BO OD : ℝ)

-- Assume O is inside the circle and AC, BD are chords intersecting at O forming segments H1 and H2
def is_chord_intersection (AC : set α) (BD : set α) (O : α) (AO : ℝ) (OC : ℝ) (BO : ℝ) (OD : ℝ) :=
  ∃ AC BD, O ∈ AC ∧ O ∈ BD ∧
           AC ∩ BD = {O} ∧
           AO > 0 ∧ OC > 0 ∧ BO > 0 ∧ OD > 0

-- Define the proof problem 
theorem product_of_segments_constant (AC BD : set α) (O : α) (AO OC BO OD : ℝ) :
  is_chord_intersection AC BD O AO OC BO OD →
  AO * OC = BO * OD :=
by
sorry

end product_of_segments_constant_l511_511119


namespace five_letter_words_with_vowel_l511_511253

theorem five_letter_words_with_vowel :
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  total_words - words_no_vowel = 6752 :=
by
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  have h1 : total_words = 7776 := by norm_num
  have h2 : words_no_vowel = 1024 := by norm_num
  show total_words - words_no_vowel = 6752
  calc
    total_words - words_no_vowel
    = 7776 - 1024 : by rw [h1, h2]
    ... = 6752 : by norm_num

end five_letter_words_with_vowel_l511_511253


namespace right_triangle_acute_angles_l511_511439

variable (α β : ℝ)

noncomputable def prove_acute_angles (α β : ℝ) : Prop :=
  α + β = 90 ∧ 4 * α = 90

theorem right_triangle_acute_angles : 
  prove_acute_angles α β → α = 22.5 ∧ β = 67.5 := by
  sorry

end right_triangle_acute_angles_l511_511439


namespace khali_total_snow_volume_l511_511741

def length1 : ℝ := 25
def width1 : ℝ := 3
def depth1 : ℝ := 0.75

def length2 : ℝ := 15
def width2 : ℝ := 3
def depth2 : ℝ := 1

def volume1 : ℝ := length1 * width1 * depth1
def volume2 : ℝ := length2 * width2 * depth2
def total_volume : ℝ := volume1 + volume2

theorem khali_total_snow_volume : total_volume = 101.25 := by
  sorry

end khali_total_snow_volume_l511_511741


namespace chang_mixture_l511_511733

noncomputable def pure_alcohol_in_mixture (a b : ℕ) (ratioA ratioB : ℕ) (difference : ℕ) (amountA : ℕ) : ℕ :=
  let amountB := amountA + difference
  let pureA := (ratioA * amountA) / 100
  let pureB := (ratioB * amountB) / 100
  pureA + pureB

theorem chang_mixture :
  pure_alcohol_in_mixture 16 10 500 600 = 206 :=
begin
  sorry
end

end chang_mixture_l511_511733


namespace parabola_vertex_l511_511373

theorem parabola_vertex :
  ∀ (x : ℝ), y = 2 * (x + 9)^2 - 3 → 
  (∃ h k, h = -9 ∧ k = -3 ∧ y = 2 * (x - h)^2 + k) :=
by
  sorry

end parabola_vertex_l511_511373


namespace pencils_bought_at_cost_price_l511_511377

variable (C S : ℝ)
variable (n : ℕ)

theorem pencils_bought_at_cost_price (h1 : n * C = 8 * S) (h2 : S = 1.5 * C) : n = 12 := 
by sorry

end pencils_bought_at_cost_price_l511_511377


namespace simplify_expression_l511_511273

theorem simplify_expression (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) :
  (x^2 / y^3) / (y / x) = 3 / 16 :=
by {
  rw [h₁, h₂],
  sorry
}

end simplify_expression_l511_511273


namespace max_hours_wednesday_l511_511775

theorem max_hours_wednesday (x : ℕ) 
    (h1 : ∀ (d w : ℕ), w = x → d = x → d + w + (x + 3) = 3 * 3) 
    (h2 : ∀ (a b c : ℕ), a = b → b = c → (a + b + (c + 3))/3 = 3) :
  x = 2 := 
by
  sorry

end max_hours_wednesday_l511_511775


namespace fraction_product_l511_511944

theorem fraction_product :
  (∏ n in range 1 56, (n / (n + 3))) = (1 / 30728) :=
by
  sorry

end fraction_product_l511_511944


namespace not_perpendicular_AB_A_l511_511414

open Classical

-- Define points and the properties of the first quadrant and reflection
structure Point where
  x : ℝ
  y : ℝ
  deriving DecidableEq

def isFirstQuadrant (A : Point) : Prop :=
  0 ≤ A.x ∧ 0 ≤ A.y

def reflectAcrossYEqualsX (A : Point) : Point :=
  ⟨A.y, A.x⟩

noncomputable def slope (A B : Point) : ℝ :=
  if h : A.x ≠ B.x then (B.y - A.y) / (B.x - A.x) else 0

-- Formalize the conditions in Lean
variable {A B C : Point}
variable hA : isFirstQuadrant A
variable hB : isFirstQuadrant B
variable hC : isFirstQuadrant C
variable h_not_on_line_y_eq_x : A.x ≠ A.y ∧ B.x ≠ B.y ∧ C.x ≠ C.y

-- Statement to prove
theorem not_perpendicular_AB_A'B' :
  ¬ (slope A B * slope (reflectAcrossYEqualsX A) (reflectAcrossYEqualsX B) = -1) :=
  sorry

end not_perpendicular_AB_A_l511_511414


namespace johns_commute_distance_l511_511075

theorem johns_commute_distance
  (y : ℝ)  -- distance in miles
  (h1 : 200 * (y / 200) = y)  -- John usually takes 200 minutes, so usual speed is y/200 miles per minute
  (h2 : 320 = (y / (2 * (y / 200))) + (y / (2 * ((y / 200) - 15/60)))) -- Total journey time on the foggy day
  : y = 92 :=
sorry

end johns_commute_distance_l511_511075


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511185

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the vowels
def vowels := {'A', 'E'}

-- Define the number of n-length words constructible from a set of letters
def num_words (n : ℕ) (alphabet : Set Char) : ℕ :=
  (alphabet.size ^ n)

-- The total number of 5-letter words (unrestricted)
def total_words := num_words 5 letters

-- The number of 5-letter words with no vowels
def no_vowel_words := num_words 5 (letters \ vowels)

-- The number of 5-letter words with at least one vowel
def at_least_one_vowel_words := total_words - no_vowel_words

-- The statement to prove that the number of 5-letter words with at least one vowel is 6752
theorem number_of_5_letter_words_with_at_least_one_vowel : 
  at_least_one_vowel_words = 6752 :=
by 
  -- Proof will be provided here
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l511_511185


namespace independence_test_l511_511295

theorem independence_test 
  (H0 : ¬related X Y)
  (H1 : P(K^2 >= 6.635) ≈ 0.01) :
  confidence_level (related X Y) = 0.99 :=
sorry

end independence_test_l511_511295


namespace relationship_of_y1_y2_l511_511710

theorem relationship_of_y1_y2 (y1 y2 : ℝ) : 
  (∃ y1 y2, (y1 = 2 / -2) ∧ (y2 = 2 / -1)) → (y1 > y2) :=
by
  sorry

end relationship_of_y1_y2_l511_511710


namespace tenth_integer_is_6785_l511_511837

-- Definitions
def four_digit_numbers (digits : List ℕ) : List ℕ := 
  List.permutations digits
  |> List.map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)
  |> List.filter (λ n, 1000 ≤ n ∧ n < 10000)

def tenth_integer (l : List ℕ) : ℕ :=
  l.nth 9 -- List indexing is zero-based, so the 10th element is the 9th index

-- Conditions
def used_digits := [5, 6, 7, 8]

-- Theorem statement
theorem tenth_integer_is_6785 : 
  ∀ (l : List ℕ), l = four_digit_numbers used_digits → (tenth_integer (l.sorted)) = 6785 :=
by
  sorry

end tenth_integer_is_6785_l511_511837


namespace most_stable_yield_l511_511859

theorem most_stable_yield (S_A S_B S_C S_D : ℝ)
  (h₁ : S_A = 3.6)
  (h₂ : S_B = 2.89)
  (h₃ : S_C = 13.4)
  (h₄ : S_D = 20.14) : 
  S_B < S_A ∧ S_B < S_C ∧ S_B < S_D :=
by {
  sorry -- Proof skipped as per instructions
}

end most_stable_yield_l511_511859


namespace product_zero_when_a_is_three_l511_511973

theorem product_zero_when_a_is_three (a : ℤ) (h : a = 3) :
  (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  cases h
  sorry

end product_zero_when_a_is_three_l511_511973


namespace max_value_f_when_a_eq_1_number_of_zeros_of_f_proof_l511_511115

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) := log (2 * x + 1) + 2 * a * x - 4 * a * exp x + 4

-- Condition: a > 0
variables (a : ℝ) (ha : a > 0)

-- 1. Maximum value of f(x) when a = 1
def max_value_f_at_a_eq_1 : ℝ := 0

theorem max_value_f_when_a_eq_1 
  (x : ℝ)
  (hf : ∀ x, f x 1 ≤ max_value_f_at_a_eq_1) :
  f 0 1 = max_value_f_at_a_eq_1 :=
sorry

-- 2. Number of zeros of f(x) based on different intervals of a
def number_of_zeros_of_f (a : ℝ) : ℕ :=
if (0 < a ∧ a < 1) then 2 else if a = 1 then 1 else if a > 1 then 0 else 0

theorem number_of_zeros_of_f_proof :
  (0 < a ∧ a < 1 → number_of_zeros_of_f a = 2) ∧ 
  (a = 1 → number_of_zeros_of_f a = 1) ∧ 
  (a > 1 → number_of_zeros_of_f a = 0) :=
sorry

end max_value_f_when_a_eq_1_number_of_zeros_of_f_proof_l511_511115


namespace max_distance_with_optimal_tire_swapping_l511_511102

theorem max_distance_with_optimal_tire_swapping
  (front_tires_last : ℕ)
  (rear_tires_last : ℕ)
  (front_tires_last_eq : front_tires_last = 20000)
  (rear_tires_last_eq : rear_tires_last = 30000) :
  ∃ D : ℕ, D = 30000 :=
by
  sorry

end max_distance_with_optimal_tire_swapping_l511_511102


namespace no_infinite_subset_of_natural_numbers_l511_511968

theorem no_infinite_subset_of_natural_numbers {
  S : Set ℕ 
} (hS_infinite : S.Infinite) :
  ¬ (∀ a b : ℕ, a ∈ S → b ∈ S → a^2 - a * b + b^2 ∣ (a * b)^2) :=
sorry

end no_infinite_subset_of_natural_numbers_l511_511968


namespace problem1_problem2_l511_511901

noncomputable def expr1 : ℂ :=
  ((1 + complex.i)^3) / complex.i + (((real.sqrt 3) + complex.i) * ((real.sqrt 3) - complex.i) - 4 * complex.i^2016) / (3 + 4 * complex.i)^2

theorem problem1 : expr1 = 2 + complex.i :=
  sorry

variable (z : ℂ) (z_conj : ℂ)
variable (h_conj : z_conj = complex.conj z)
variable (h_eq : 4 * z + 2 * z_conj = 3 * real.sqrt 3 + complex.i)

theorem problem2 : z = (real.sqrt 3) / 2 + (1 / 2 : ℚ) * complex.i :=
  sorry

end problem1_problem2_l511_511901


namespace correct_probability_l511_511568

-- Five people taking a true-or-false test with five questions
def random_guess (n : ℕ) : ℕ := 2^n

-- Majority correct condition 
def majority_correct (people questions : ℕ) := ∀ q : ℕ, q < questions → (3 ≤ people) ∧ (people ≤ 5)

-- Probability that every person answers exactly three questions correctly
def all_answer_three_correctly (people questions : ℕ) : ℕ := nat.choose questions 3

-- Calculation bases 
def p (a b : ℕ) : ℚ := (a : ℚ) / 2^b

-- Final lean theorem
theorem correct_probability :
  (p 255 17 = 255 / 2^17) ∧ (100 * 255 + 17 = 25517) :=
begin
  sorry
end

end correct_probability_l511_511568


namespace milk_left_in_storage_l511_511854

-- Define initial and rate conditions
def initialMilk : ℕ := 30000
def pumpedRate : ℕ := 2880
def pumpedHours : ℕ := 4
def addedRate : ℕ := 1500
def addedHours : ℕ := 7

-- The proof problem: Prove the final amount in storage tank == 28980 gallons
theorem milk_left_in_storage : 
  initialMilk - (pumpedRate * pumpedHours) + (addedRate * addedHours) = 28980 := 
sorry

end milk_left_in_storage_l511_511854


namespace max_length_DE_l511_511508

theorem max_length_DE (A B C D E : Point) (p b : ℝ) (h_triangle : is_triangle A B C)
    (h_perimeter : perimeter A B C = 2 * p)
    (h_parallel : parallel DE AC)
    (h_tangent : tangent_incicle A B C DE) :
    (exists b, b = p / 2 ∧ segment_length DE = p / 4) :=
by sorry

end max_length_DE_l511_511508


namespace hyperbola_eccentricity_l511_511316

theorem hyperbola_eccentricity 
  (a b e : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : e = Real.sqrt (1 + (b^2 / a^2))) 
  (h4 : e ≤ Real.sqrt 5) : 
  e = 2 := 
sorry

end hyperbola_eccentricity_l511_511316


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511233

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let total_words := 6^5
  let words_without_vowels := 4^5
  total_words - words_without_vowels = 6752 :=
by
  let total_words := 6^5
  let words_without_vowels := 4^5
  have h_total_words : total_words = 7776 := by norm_num
  have h_words_without_vowels : words_without_vowels = 1024 := by norm_num
  calc
    7776 - 1024 = 6752 : by norm_num

end number_of_5_letter_words_with_at_least_one_vowel_l511_511233


namespace XPYC_is_parallelogram_l511_511867

variables (ω1 ω2 : Type*) [circle ω1] [circle ω2]
variables (O1 O2 A B P X Y C : Type*)
variables [point O1] [point O2] [point A] [point B] [point P] [point X] [point Y] [point C]
variables (lies_on_ω2 : O1 ∈ ω2) (P_on_ω1 : P ∈ ω1)
variables (second_intersection_BP_ω2 : ∀ P, X ∈ ω2 ∧ X ≠ B) 
variables (second_intersection_AP_ω2 : ∀ P, Y ∈ ω2 ∧ Y ≠ A) 
variables (second_intersection_O1O2_ω2 : C ∈ ω2 ∧ C ≠ O1)

theorem XPYC_is_parallelogram :
  parallelogram X P Y C :=
sorry

end XPYC_is_parallelogram_l511_511867


namespace smallest_perimeter_of_scalene_triangle_with_conditions_l511_511921

def is_odd_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 2 = 1

-- Define a scalene triangle
structure ScaleneTriangle :=
  (a b c : ℕ)
  (a_ne_b : a ≠ b)
  (a_ne_c : a ≠ c)
  (b_ne_c : b ≠ c)
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a)

-- Define the problem conditions
def problem_conditions (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧
  a < b ∧ b < c ∧
  Nat.Prime (a + b + c) ∧
  (∃ (t : ScaleneTriangle), t.a = a ∧ t.b = b ∧ t.c = c)

-- Define the proposition
theorem smallest_perimeter_of_scalene_triangle_with_conditions :
  ∃ (a b c : ℕ), problem_conditions a b c ∧ a + b + c = 23 :=
sorry

end smallest_perimeter_of_scalene_triangle_with_conditions_l511_511921


namespace binomial_coeff_sum_l511_511705

theorem binomial_coeff_sum (a : ℝ → ℝ) (x : ℝ) :
  (∀ x : ℝ, (1 - 3 * x) ^ 2016 = a 0 + a 1 * x + a 2 * x^2 + ... + a 2016 * x^2016) →
  a 0 = 1 →
  a 0 + (a 1 / 3) + (a 2 / 3^2) + ... + (a 2016 / 3^2016) = 0 →
  (a 1 / 3) + (a 2 / 3^2) + ... + (a 2016 / 3^2016) = -1 := by
  intros h_expansion h_a0 h_sum
  sorry

end binomial_coeff_sum_l511_511705


namespace fraction_of_area_above_line_l511_511061

noncomputable def square_vertices : set (ℝ × ℝ) := { (4, 0), (10, 0), (10, 6), (4, 6) }
noncomputable def line_points : set (ℝ × ℝ) := { (4, 2), (10, 1) }

theorem fraction_of_area_above_line :
  let square_area := 36 in
  let line := λ (x : ℝ), (-1/6) * x + 14 / 3 in
  let triangle_area := (1/2) * (10 - 4) * ((line 4 + line 10) / 2) in
  let area_above_line := square_area - triangle_area in
  area_above_line / square_area = 7 / 8 :=
by
  sorry

end fraction_of_area_above_line_l511_511061


namespace cylinder_diameter_l511_511828

noncomputable def diameter_of_cylinder (height : ℝ) (volume : ℝ) : ℝ :=
  let pi_approx : ℝ := 3.14159
  let radius := Real.sqrt (volume / (pi_approx * height))
  2 * radius

theorem cylinder_diameter :
  diameter_of_cylinder 14 1099.5574287564277 ≈ 9.99528 :=
by
  sorry

end cylinder_diameter_l511_511828


namespace probability_gcd_one_l511_511431

-- Defining the domain of our problem: the set {1, 2, 3, ..., 8}
def S := {1, 2, 3, 4, 5, 6, 7, 8}

-- Defining the selection of two distinct natural numbers from S
def select_two_distinct_from_S (x y : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ x ≠ y

-- Defining the greatest common factor condition
def is_rel_prime (x y : ℕ) : Prop :=
  Nat.gcd x y = 1

-- Defining the probability computation (relatively prime pairs over total pairs)
def probability_rel_prime : ℚ :=
  (21 : ℚ) / 28  -- since 21 pairs are relatively prime out of 28 total pairs

-- The main theorem statement
theorem probability_gcd_one :
  probability_rel_prime = 3 / 4 :=
sorry

end probability_gcd_one_l511_511431


namespace area_of_square_l511_511291

-- Define the points of the square ABCD
variables {s : ℝ} (A B C D G H : ℝ × ℝ)

-- Define the square condition
def is_square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (s, 0) ∧ C = (s, s) ∧ D = (0, s) ∧ G = (s / 2, 0)

-- Define the midpoint condition
def midpoint_condition (A B G : ℝ × ℝ) : Prop :=
  G = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the intersection condition
def intersection_condition (D G B C H : ℝ × ℝ) : Prop :=
  let DG_slope := (G.2 - D.2) / (G.1 - D.1)
  in DG_slope = -2 ∧ H.1 = s / 3 ∧ H.2 = s / 3

-- Define the area condition
def area_BGHC (B G H C : ℝ × ℝ) : Prop :=
  let area_DGH := 1 / 2 * (s / 2) * s / 2
  let area_BHC := 1 / 2 * (2 * s / 3) * (2 * s / 3)
  in (area_DGH + area_BHC) = 50

-- Prove the area of the square ABCD
theorem area_of_square :
  is_square A B C D G ∧ midpoint_condition A B G ∧ intersection_condition D G B C H ∧ area_BGHC B G H C → s^2 = 106 :=
by
  sorry

end area_of_square_l511_511291


namespace original_number_is_9_l511_511482

theorem original_number_is_9 (x : ℤ) (h : 10 * x = x + 81) : x = 9 :=
sorry

end original_number_is_9_l511_511482


namespace number_of_equilateral_triangle_vertices_l511_511254

/-- Prove that the number of distinct nonzero complex numbers z 
such that 0, z, and z^3 form an equilateral triangle is equal to 4. -/
theorem number_of_equilateral_triangle_vertices : 
  {z : ℂ | z ≠ 0 ∧ (∃ p : {0, z, z^3}, is_equilateral 0 z (z^3))}.size = 4 := 
by 
  sorry

end number_of_equilateral_triangle_vertices_l511_511254


namespace a_2029_is_1_l511_511121

noncomputable def sequence (n : ℕ) : ℕ :=
  if n % 6 = 1 then 1 else
  if n % 6 = 2 then 2 else
  if n % 6 = 3 then sorry else
  if n % 6 = 4 then sorry else
  if n % 6 = 5 then sorry else
  sorry

theorem a_2029_is_1 : sequence 2029 = 1 :=
by
  -- Proof is omitted.
  sorry

end a_2029_is_1_l511_511121


namespace sphere_volume_doubles_l511_511879

theorem sphere_volume_doubles (r : ℝ) (V : ℝ) (h : V = (4/3) * π * r^3) : 
  let r_new := 2 * r in
  let V_new := (4/3) * π * r_new^3 in
  V_new = 8 * V := 
by 
  sorry

end sphere_volume_doubles_l511_511879


namespace functional_equation_zero_l511_511541

open Function

theorem functional_equation_zero (f : ℕ+ → ℝ) 
  (h : ∀ (m n : ℕ+), n ≥ m → f (n + m) + f (n - m) = f (3 * n)) :
  ∀ n : ℕ+, f n = 0 := sorry

end functional_equation_zero_l511_511541


namespace find_hyperbola_with_given_eccentricity_l511_511931

theorem find_hyperbola_with_given_eccentricity :
  (∃ (e : ℝ) (a b : ℝ),
    e = (Real.sqrt 6) / 2 ∧
    (e^2 = 3 / 2) ∧
    (a^2 = 2 * b^2) ∧
    (∃ (h : ℝ → ℝ → Prop),
      (h = λ x y, x^2 / 4 - y^2 / 2 = 1) ))
sorry

end find_hyperbola_with_given_eccentricity_l511_511931


namespace encore_songs_l511_511842

-- Definitions corresponding to the conditions
def repertoire_size : ℕ := 30
def first_set_songs : ℕ := 5
def second_set_songs : ℕ := 7
def average_songs_per_set_3_and_4 : ℕ := 8

-- The statement to prove
theorem encore_songs : (repertoire_size - (first_set_songs + second_set_songs)) - (2 * average_songs_per_set_3_and_4) = 2 := by
  sorry

end encore_songs_l511_511842


namespace sum_of_differences_l511_511627

theorem sum_of_differences : 
  (∑ n in Finset.range 1000, let first_digit := n / 10 ^ (Nat.log10 n) in
                             let last_digit := n % 10 in
                             first_digit - last_digit) = 495 :=
by
  sorry

end sum_of_differences_l511_511627


namespace total_paint_area_l511_511770

def room_width := 20 -- feet
def room_length := 20 -- feet
def room_height := 8 -- feet

def doorway1_width := 3 -- feet
def doorway1_height := 7 -- feet

def window_width := 6 -- feet
def window_height := 4 -- feet

def doorway2_width := 5 -- feet
def doorway2_height := 7 -- feet

theorem total_paint_area :
  let wall_area := 4 * (room_width * room_height),
      doorway1_area := doorway1_width * doorway1_height,
      window_area := window_width * window_height,
      doorway2_area := doorway2_width * doorway2_height in
  wall_area - (doorway1_area + window_area + doorway2_area) = 560 := 
by 
  unfold wall_area
  unfold doorway1_area
  unfold window_area
  unfold doorway2_area
  sorry

end total_paint_area_l511_511770


namespace pot_water_temperature_131_F_l511_511437

def celsius_to_fahrenheit (C : ℝ) : ℝ :=
  (C * 9 / 5) + 32

theorem pot_water_temperature_131_F (C : ℝ) (P : ℝ) (H1 : C = 55) (H2 : P = 0.8) : 
  celsius_to_fahrenheit C = 131 :=
by
  rw [H1]
  unfold celsius_to_fahrenheit
  norm_num
  done

end pot_water_temperature_131_F_l511_511437


namespace range_of_x_l511_511267

-- Problem Statement
theorem range_of_x (x : ℝ) (h : 0 ≤ x - 8) : 8 ≤ x :=
by {
  sorry
}

end range_of_x_l511_511267


namespace sum_differences_1_to_999_l511_511594

-- Define a utility function to compute the first digit of a number
def first_digit (n : ℕ) : ℕ :=
if n < 10 then n else first_digit (n / 10)

-- Define a utility function to compute the last digit of a number
def last_digit (n : ℕ) : ℕ :=
n % 10

-- Define the operation performed by Damir
def difference (n : ℕ) : ℤ :=
(first_digit n : ℤ) - (last_digit n : ℤ)

-- Define the sum of all differences from 1 to 999
def sum_differences : ℤ :=
(1).to (999).sum (λ n, difference n)

-- State the main theorem to be proved using the previous definitions
theorem sum_differences_1_to_999 : sum_differences = 495 :=
sorry

end sum_differences_1_to_999_l511_511594


namespace find_k_l511_511702

theorem find_k (k : ℝ) (h : (-3 : ℝ)^2 + (-3 : ℝ) - k = 0) : k = 6 :=
by
  sorry

end find_k_l511_511702


namespace finish_lollipops_in_6_days_l511_511171

variables (henry_alison_diff : ℕ) (alison_lollipops : ℕ) (diane_alison_ratio : ℕ) (lollipops_eaten_per_day : ℕ)
variables (days_needed : ℕ) (henry_lollipops : ℕ) (diane_lollipops : ℕ) (total_lollipops : ℕ)

-- Conditions as definitions
def condition_1 : Prop := henry_alison_diff = 30
def condition_2 : Prop := alison_lollipops = 60
def condition_3 : Prop := alison_lollipops * 2 = diane_lollipops
def condition_4 : Prop := lollipops_eaten_per_day = 45

-- Total lollipops calculation
def total_lollipops_calculated : ℕ := alison_lollipops + diane_lollipops + henry_lollipops

-- Days to finish lollipops calculation
def days_needed_calculated : ℕ := total_lollipops / lollipops_eaten_per_day

-- The theorem to prove
theorem finish_lollipops_in_6_days :
  condition_1 →
  condition_2 →
  condition_3 →
  condition_4 →
  henry_lollipops = alison_lollipops + 30 →
  total_lollipops_calculated = 270 →
  days_needed_calculated = 6 :=
by {
  sorry
}

end finish_lollipops_in_6_days_l511_511171


namespace minimize_f_sum_l511_511110

noncomputable def f (x : ℝ) : ℝ := x^2 - 8*x + 10

theorem minimize_f_sum :
  ∃ a₁ : ℝ, (∀ a₂ a₃ : ℝ, a₂ = a₁ + 1 ∧ a₃ = a₁ + 2 →
    f(a₁) + f(a₂) + f(a₃) = 3 * a₁^2 - 18 * a₁ + 30) →
    (∀ b₁ : ℝ, (∀ b₂ b₃ : ℝ, b₂ = b₁ + 1 ∧ b₃ = b₁ + 2 →
      f(b₁) + f(b₂) + f(b₃) ≥ f(a₁) + f(a₂) + f(a₃)) ∧ a₁ = 3) :=
by 
  sorry

end minimize_f_sum_l511_511110


namespace hyperbola_no_common_points_l511_511317

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) : ℝ :=
  real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_no_common_points (a b e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_ecc : e = real.sqrt (1 + (b^2 / a^2)))
  (h_slope : b / a < 2) :
  e = 2 :=
sorry

end hyperbola_no_common_points_l511_511317


namespace ab_value_l511_511673

noncomputable def func (x : ℝ) (a b : ℝ) : ℝ := 4 * x ^ 3 - a * x ^ 2 - 2 * b * x + 2

theorem ab_value 
  (a b : ℝ)
  (h_max : func 1 a b = -3)
  (h_deriv : (12 - 2 * a - 2 * b) = 0) :
  a * b = 9 :=
by
  sorry

end ab_value_l511_511673


namespace damian_serena_passing_times_l511_511065

/-- 
  Damian and Serena are running on a circular track for 40 minutes.
  Damian runs clockwise at 220 m/min on the inner lane with a radius of 45 meters.
  Serena runs counterclockwise at 260 m/min on the outer lane with a radius of 55 meters.
  They start on the same radial line.
  Prove that they pass each other exactly 184 times in 40 minutes. 
-/
theorem damian_serena_passing_times
  (time_run : ℕ)
  (damian_speed : ℕ)
  (serena_speed : ℕ)
  (damian_radius : ℝ)
  (serena_radius : ℝ)
  (start_same_line : Prop) :
  time_run = 40 →
  damian_speed = 220 →
  serena_speed = 260 →
  damian_radius = 45 →
  serena_radius = 55 →
  start_same_line →
  ∃ n : ℕ, n = 184 :=
by
  sorry

end damian_serena_passing_times_l511_511065


namespace child_wants_to_buy_3_toys_l511_511906

/- 
  Problem Statement:
  There are 10 toys, and the number of ways to select a certain number 
  of those toys in any order is 120. We need to find out how many toys 
  were selected.
-/

def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem child_wants_to_buy_3_toys :
  ∃ r : ℕ, r ≤ 10 ∧ comb 10 r = 120 :=
by
  use 3
  -- Here you would write the proof
  sorry

end child_wants_to_buy_3_toys_l511_511906


namespace melanie_trout_l511_511341

theorem melanie_trout (M : ℕ) (h1 : 2 * M = 16) : M = 8 :=
by
  sorry

end melanie_trout_l511_511341


namespace total_animals_l511_511514

theorem total_animals (chickens ducks geese quails turkeys cows pigs : ℕ) 
                      (hc : chickens = 60) 
                      (hd : ducks = 40) 
                      (hg : geese = 20) 
                      (hq : quails = 50) 
                      (ht : turkeys = 10) 
                      (hsheds : cows = 3 * 8) 
                      (hp : pigs = 15) : 
                      chickens + ducks + geese + quails + turkeys + cows + pigs = 219 :=
by
  calc 
  chickens + ducks + geese + quails + turkeys + cows + pigs
      = 60 + 40 + 20 + 50 + 10 + (3 * 8) + 15 : by rw [hc, hd, hg, hq, ht, hsheds, hp]
  ... = 60 + 40 + 20 + 50 + 10 + 24 + 15 : by rfl
  ... = 219 : by rfl

end total_animals_l511_511514


namespace five_letter_words_with_at_least_one_vowel_l511_511217

open Finset

theorem five_letter_words_with_at_least_one_vowel :
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := ((letters \ vowels).card : ℕ) ^ 5
  total_words - no_vowel_words = 6752 :=
by
  let letters := insert 'A' (insert 'B' (insert 'C' (insert 'D' (insert 'E' (singleton 'F')))))
  let vowels := insert 'A' (singleton 'E')
  let consonants := letters \ vowels
  have : letters.card = 6 := by simp
  have : vowels.card = 2 := by simp
  have : consonants.card = 4 := by simp
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := (consonants.card : ℕ) ^ 5
  calc
    total_words - no_vowel_words
        = 6^5 - 4^5         : by simp
    ... = 7776 - 1024       : by norm_num
    ... = 6752             : by norm_num

end five_letter_words_with_at_least_one_vowel_l511_511217


namespace sum_differences_1_to_999_l511_511595

-- Define a utility function to compute the first digit of a number
def first_digit (n : ℕ) : ℕ :=
if n < 10 then n else first_digit (n / 10)

-- Define a utility function to compute the last digit of a number
def last_digit (n : ℕ) : ℕ :=
n % 10

-- Define the operation performed by Damir
def difference (n : ℕ) : ℤ :=
(first_digit n : ℤ) - (last_digit n : ℤ)

-- Define the sum of all differences from 1 to 999
def sum_differences : ℤ :=
(1).to (999).sum (λ n, difference n)

-- State the main theorem to be proved using the previous definitions
theorem sum_differences_1_to_999 : sum_differences = 495 :=
sorry

end sum_differences_1_to_999_l511_511595


namespace noah_total_bill_l511_511784

def call_duration := 30 -- in minutes
def charge_per_minute := 0.05 -- in dollars per minute
def calls_per_week := 1 -- calls per week
def weeks_per_year := 52 -- weeks per year

theorem noah_total_bill:
  (calls_per_week * weeks_per_year * call_duration * charge_per_minute) = 78 :=
by
  sorry

end noah_total_bill_l511_511784


namespace correct_mean_of_values_l511_511390

theorem correct_mean_of_values
    (mean_incorrect: ℝ) (n: ℕ) (incorrect_value: ℝ) (correct_value: ℝ) (mean_correct: ℝ)
    (h1: mean_incorrect = 250) (h2: n = 30) (h3: incorrect_value = 135) (h4: correct_value = 165)
    (h5: mean_correct = 251) :
    (mean_correct := (mean_incorrect * n + (correct_value - incorrect_value)) / n) = 251 :=
by
  -- Proof here
  sorry

end correct_mean_of_values_l511_511390


namespace area_of_triangle_ABC_l511_511871

-- Given conditions
variables (A B C D : Type) [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables (coplanar : A × B × C × D) (angle_right : IsRightAngle D) (AC : ℝ) (AB : ℝ) (DC : ℝ)

-- Define what we need to prove
theorem area_of_triangle_ABC : coplanar ∧ angle_right ∧ AC = 15 ∧ AB = 17 ∧ DC = 9 
  → area_of_triangle A B C = 54 + 6 * real.sqrt 145 := 
sorry

end area_of_triangle_ABC_l511_511871


namespace value_of_k_l511_511831

noncomputable def roots_in_ratio_equation {k : ℝ} (h : k ≠ 0) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ 
  (r₁ / r₂ = 3) ∧ 
  (r₁ + r₂ = -8) ∧ 
  (r₁ * r₂ = k)

theorem value_of_k (k : ℝ) (h : k ≠ 0) (hr : roots_in_ratio_equation h) : k = 12 :=
sorry

end value_of_k_l511_511831


namespace initial_number_divisible_by_15_l511_511560

theorem initial_number_divisible_by_15 (N : ℕ) (h : (N - 7) % 15 = 0) : N = 22 := 
by
  sorry

end initial_number_divisible_by_15_l511_511560


namespace sum_digit_differences_l511_511610

def first_digit (n : ℕ) : ℕ := 
  (n / 10 ^ ((Nat.log10 n) : ℕ))

def last_digit (n : ℕ) : ℕ := n % 10

def digit_difference (n : ℕ) : ℤ :=
  (first_digit n : ℤ) - (last_digit n : ℤ)

theorem sum_digit_differences :
  (∑ n in Finset.range 1000, digit_difference n) = 495 := 
sorry

end sum_digit_differences_l511_511610


namespace increasing_ω_l511_511657

noncomputable def f (ω x : ℝ) : ℝ := (1 / 2) * (Real.sin ((ω * x) / 2)) * (Real.cos ((ω * x) / 2))

theorem increasing_ω (ω : ℝ) (hω : 0 < ω) :
  (∀ x y, - (Real.pi / 3) ≤ x → x ≤ y → y ≤ (Real.pi / 4) → f ω x ≤ f ω y)
  ↔ 0 < ω ∧ ω ≤ (3 / 2) :=
sorry

end increasing_ω_l511_511657


namespace general_term_of_sequence_l511_511651

variable (a : ℕ → ℕ)
variable (h1 : ∀ m : ℕ, a (m^2) = a m ^ 2)
variable (h2 : ∀ m k : ℕ, a (m^2 + k^2) = a m * a k)

theorem general_term_of_sequence : ∀ n : ℕ, n > 0 → a n = 1 :=
by
  intros n hn
  sorry

end general_term_of_sequence_l511_511651


namespace solve_beas_farm_problem_l511_511047

def beas_farm_problem : Prop :=
  ∃ (M N trees_planted : ℕ),
    (50 + 30 - 5) + trees_planted = 88 ∧  -- Total trees before planting + trees planted = 88
    M + N = 5 ∧                          -- Total fallen trees = 5
    M = N + 1 ∧                          -- One more Mahogany tree fell than Narra trees
    trees_planted = 13 ∧                 -- Trees planted after typhoon
    M = 3 ∧                              -- Number of Mahogany trees that fell
    (13 : 3) = 13 : 3                    -- Ratio of trees planted to Mahogany trees that fell

theorem solve_beas_farm_problem : beas_farm_problem :=
  sorry

end solve_beas_farm_problem_l511_511047


namespace probability_same_theme_l511_511026

theorem probability_same_theme :
  ∀ (themes : Finset String) (students : Finset String),
  themes.card = 2 →
  students.card = 2 →
  let n := themes.card * students.card in
  let m := themes.card * 1 in
  (m : ℚ) / n = 1 / 2 :=
by
  intros themes students h1 h2
  let n := themes.card * students.card
  let m := themes.card * 1
  sorry

end probability_same_theme_l511_511026


namespace count_arrangements_l511_511007

def num_chairs : ℕ := 10
def num_benches : ℕ := 4
def total_seats : ℕ := num_chairs + num_benches

theorem count_arrangements : nat.choose total_seats num_benches = 1001 :=
by sorry

end count_arrangements_l511_511007


namespace curve_is_line_l511_511545

theorem curve_is_line (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b c : ℝ), a * (r * Real.cos θ) + b * (r * Real.sin θ) + c = 0 ∧
  (a, b, c) = (-1, 2, -1) := sorry

end curve_is_line_l511_511545


namespace rabbit_catch_up_time_l511_511861

-- Define the conditions
def rabbit_speed : ℝ := 25 -- Rabbit's speed in miles per hour
def cat_speed : ℝ := 20 -- Cat's speed in miles per hour
def head_start_time : ℝ := 0.25 -- Head start time in hours

-- Define the distance the cat covers in the head start
def distance_cat : ℝ := cat_speed * head_start_time

-- Define the relative speed between the rabbit and the cat
def relative_speed : ℝ := rabbit_speed - cat_speed

-- Define the time for the rabbit to catch up
def time_to_catch_up : ℝ := distance_cat / relative_speed

-- Prove that the time to catch up is 1 hour
theorem rabbit_catch_up_time : time_to_catch_up = 1 :=
by
  -- proof will go here
  -- but this is a properly structured Lean statement based on given conditions 
  sorry

end rabbit_catch_up_time_l511_511861


namespace part1_part2_part3_l511_511076

-- Part 1
theorem part1 (x y : ℕ) (h1 : x + y = 30) (h2 : 30 * x + 25 * y = 850) :
  x = 20 ∧ y = 10 :=
sorry

-- Part 2
theorem part2 (m : ℕ) (h1 : m ≤ 40) :
  let w := 3 * m + 960 in 
  w = 2520 :=
sorry

-- Part 3
theorem part3 (a : ℕ) (h1 : (12 - a) * (4 + 2 * a) = 90) :
  a = 3 ∨ a = 7 :=
sorry

end part1_part2_part3_l511_511076


namespace brazilian_triples_l511_511925

def is_brazilian (a b c : ℕ) : Prop :=
  a ∣ (b * c + 1) ∧ b ∣ (a * c + 1) ∧ c ∣ (a * b + 1)

theorem brazilian_triples :
  ∀ a b c : ℕ, is_brazilian a b c → 
  (a = 3 ∧ b = 2 ∧ c = 1) ∨
  (a = 7 ∧ b = 3 ∧ c = 2) ∨
  (a = 2 ∧ b = 1 ∧ c = 1) ∨
  (a = 1 ∧ b = 1 ∧ c = 1) ∨
  (a = 2 ∧ (b, c) = (a, b) ∨ (a = c ∧ (b, a) = (a, c)) ∨ (a, b) = (c, a)) ∨
  (a = 2 ∧ (c, b) = (a, b) ∨ (a = c ∧ (c, a) = (a, c)) ∨ (a, c) = (b, a)) ∨
  (b = 2 ∧ (c, a) = (b, a) ∨ (b = c ∧ (c, b) = (b, c)) ∨ (b, a) = (c, b)) ∨ 
  sorry

end brazilian_triples_l511_511925


namespace perfect_square_divisor_probability_l511_511023

theorem perfect_square_divisor_probability (m n : ℕ) (h_mn_coprime : Nat.coprime m n) (h_m : m = 1) (h_n : n = 42) :
  (15.factorial : ℕ).divisors.count (λ d, Nat.is_square d) / (15.factorial : ℕ).divisors.count (λ _ , true) = 1 / 42 ∧ m + n = 43 :=
by
  sorry

end perfect_square_divisor_probability_l511_511023


namespace value_of_k_l511_511832

noncomputable def roots_in_ratio_equation {k : ℝ} (h : k ≠ 0) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ 
  (r₁ / r₂ = 3) ∧ 
  (r₁ + r₂ = -8) ∧ 
  (r₁ * r₂ = k)

theorem value_of_k (k : ℝ) (h : k ≠ 0) (hr : roots_in_ratio_equation h) : k = 12 :=
sorry

end value_of_k_l511_511832


namespace probability_allison_wins_l511_511034

theorem probability_allison_wins :
  let allison_cube := [6, 6, 6, 6, 6, 6]
  let brian_cube := [1, 2, 3, 4, 5, 6]
  let noah_cube := [3, 3, 3, 5, 5, 5]
  (allison_roll > brian_roll ∧ allison_roll > noah_roll) :=
  ∑ allison_roll in allison_cube, 
  ∑ brian_roll in brian_cube, 
  ∑ noah_roll in noah_cube, 
  (allison_roll > brian_roll ∧ allison_roll > noah_roll) / (6 * 6 * 6) = 5 / 6 :=
by
  sorry

end probability_allison_wins_l511_511034


namespace ab_product_function_range_l511_511136

theorem ab_product (a b : ℝ) (h1 : 8 * a + 2 * b = 2) (h2 : 12 * a + b = 9) : a * b = -3 :=
sorry

theorem function_range (a b : ℝ) (h1 : 8 * a + 2 * b = 2) (h2 : 12 * a + b = 9) :
  set.range (λ x : ℝ, a * x^3 + b * x) ∩ Icc (-3/2) 3 = Icc (-2 : ℝ) 18 :=
sorry

end ab_product_function_range_l511_511136


namespace five_letter_words_with_vowel_l511_511192

theorem five_letter_words_with_vowel : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let n := 5 
  (∃ (w : list(char)), w.length = n ∧ ∀ (i : fin n), w[i] ∈ letters ∧ (∃ (j : fin n), w[j] ∈ vowels)) → 
  (6^5 - 4^5 = 6752) := 
by
  sorry

end five_letter_words_with_vowel_l511_511192


namespace count_5_letter_words_with_at_least_one_vowel_l511_511206

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := (letters.card) ^ 5

def non_vowel_letters : Finset Char := letters \ vowels

def total_non_vowel_5_letter_words : ℕ := (non_vowel_letters.card) ^ 5

theorem count_5_letter_words_with_at_least_one_vowel :
  total_5_letter_words - total_non_vowel_5_letter_words = 6752 :=
by
  sorry

end count_5_letter_words_with_at_least_one_vowel_l511_511206


namespace no_interval_without_points_l511_511965

theorem no_interval_without_points (r : ℝ) (P : ℤ → ℝ → ℝ) (ε : ℝ) (π_irrational : ¬(∃ p q : ℤ, q ≠ 0 ∧ (p:ℝ) / (q:ℝ) = π)) :
  (circle_center P 0 = r) → 
  (∀ n : ℤ, P n r = (n : ℝ) * 1) →
  ¬(∃ (I : Set ℝ), (is_interval I ∧ I.length = ε ∧ (∀ i : ℤ, P i r ∉ I))) := 
sorry

end no_interval_without_points_l511_511965


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511186

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the vowels
def vowels := {'A', 'E'}

-- Define the number of n-length words constructible from a set of letters
def num_words (n : ℕ) (alphabet : Set Char) : ℕ :=
  (alphabet.size ^ n)

-- The total number of 5-letter words (unrestricted)
def total_words := num_words 5 letters

-- The number of 5-letter words with no vowels
def no_vowel_words := num_words 5 (letters \ vowels)

-- The number of 5-letter words with at least one vowel
def at_least_one_vowel_words := total_words - no_vowel_words

-- The statement to prove that the number of 5-letter words with at least one vowel is 6752
theorem number_of_5_letter_words_with_at_least_one_vowel : 
  at_least_one_vowel_words = 6752 :=
by 
  -- Proof will be provided here
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l511_511186


namespace line_equation_l511_511084

-- Define fixed point P
def P : ℝ × ℝ := (0, 1)

-- Define line l1 : x - 3y + 10 = 0
def l1 (x y : ℝ) : Prop := x - 3 * y + 10 = 0

-- Define line l2 : 2x + y - 8 = 0
def l2 (x y : ℝ) : Prop := 2 * x + y - 8 = 0

-- Define the line l passing through P
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- State the main theorem
theorem line_equation (k : ℝ) (l passes_through_P l1_intersection l2_intersection midpoint_condition : Prop) :
  passes_through_P → l1_intersection → l2_intersection → midpoint_condition → (∀ x y, line_l k x y = x + 4 * y - 4 = 0) :=
begin
  -- We define each condition here
  let passes_through_P : Prop := ∃ m n, m = 0 ∧ n = 1 ∧ line_l k m n,
  let l1_intersection : Prop := ∃ x y, l1 x y ∧ line_l k x y,
  let l2_intersection : Prop := ∃ x y, l2 x y ∧ line_l k x y,
  let midpoint_condition : Prop := ∃ x1 y1 x2 y2, l1 x1 y1 ∧ l2 x2 y2 ∧ P = ((x1 + x2) / 2, (y1 + y2) / 2),

  sorry
end

end line_equation_l511_511084


namespace demand_function_is_correct_tax_revenue_collected_at_30_optimal_tax_rate_is_60_max_tax_revenue_is_8640_l511_511891

variables {P t : ℝ}

noncomputable def demandFunction : ℝ → ℝ :=
λ P, 688 - 4 * P

noncomputable def taxRevenue (t : ℝ) : ℝ :=
let Pd := 118 in
let Qd := demandFunction Pd in
Qd * t

theorem demand_function_is_correct :
  ∀ P, demandFunction P = 688 - 4 * P := sorry

theorem tax_revenue_collected_at_30 :
  taxRevenue 30 = 6480 := sorry

noncomputable def optimalTaxRate : ℝ :=
288 / 4.8

theorem optimal_tax_rate_is_60 :
  optimalTaxRate = 60 := sorry

noncomputable def maxTaxRevenue : ℝ :=
let t := optimalTaxRate in
288 * t - 2.4 * t^2

theorem max_tax_revenue_is_8640 :
  maxTaxRevenue = 8640 := sorry

end demand_function_is_correct_tax_revenue_collected_at_30_optimal_tax_rate_is_60_max_tax_revenue_is_8640_l511_511891


namespace five_letter_words_with_vowel_l511_511197

theorem five_letter_words_with_vowel : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let n := 5 
  (∃ (w : list(char)), w.length = n ∧ ∀ (i : fin n), w[i] ∈ letters ∧ (∃ (j : fin n), w[j] ∈ vowels)) → 
  (6^5 - 4^5 = 6752) := 
by
  sorry

end five_letter_words_with_vowel_l511_511197


namespace linda_paint_area_l511_511767

noncomputable def total_paint_area
  (room_width room_length room_height door_width door_height window_width window_height closet_door_width closet_door_height : ℝ) 
  (walls_count : ℕ) : ℝ :=
let wall_area := room_width * room_height in
let doorway_area := door_width * door_height in
let window_area := window_width * window_height in
let closet_door_area := closet_door_width * closet_door_height in
let total_door_window_area := doorway_area + window_area + closet_door_area in
let total_wall_area_before := walls_count * wall_area in
total_wall_area_before - total_door_window_area

theorem linda_paint_area :
  total_paint_area 20 20 8 3 7 6 4 5 7 4 = 560 :=
by simp [total_paint_area]; norm_num; sorry

end linda_paint_area_l511_511767


namespace reduced_dozen_price_l511_511902

-- Define the conditions
variables (P : ℝ) -- Original price of a banana
noncomputable def reduced_price := 0.4 * P -- Reduced price after 60% reduction

-- Number of bananas original price and reduced price for Rs. 150
def original_bananas := 150 / P
def reduced_bananas := 150 / reduced_price + 120

-- The main proof statement
theorem reduced_dozen_price
  (h : 150 / reduced_price = 150 / P + 120) :
  12 * reduced_price = 48 / 17 :=
sorry

end reduced_dozen_price_l511_511902


namespace noah_total_bill_l511_511782

def call_duration := 30 -- in minutes
def charge_per_minute := 0.05 -- in dollars per minute
def calls_per_week := 1 -- calls per week
def weeks_per_year := 52 -- weeks per year

theorem noah_total_bill:
  (calls_per_week * weeks_per_year * call_duration * charge_per_minute) = 78 :=
by
  sorry

end noah_total_bill_l511_511782


namespace cover_parallelepiped_l511_511349

-- Define the conditions
def cube_side_length : ℕ := 1
def cubes_count : ℕ := 4
def parallelepiped_width : ℕ := 1
def parallelepiped_depth : ℕ := 1
def parallelepiped_height : ℕ := cubes_count

-- State the main theorem
theorem cover_parallelepiped : 
  ∃ (square1 square2 square3 : ℕ), 
    (square1 = 4 ∧ square2 = 1 ∧ square3 = 1) ∧
    (∀ (x y z : ℕ), 
      (x = 1 ∧ y = 1 ∧ z = 4) → 
      (covers_parallelepiped cube_side_length x y z square1 square2 square3)) := 
sorry

end cover_parallelepiped_l511_511349


namespace find_d_value_l511_511718

theorem find_d_value (A B C M N F : Type*)
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space M] [metric_space N] [metric_space F]
  (h1 : equilateral_triangle A B C)
  (h2 : is_midpoint M A B)
  (h3 : is_midpoint N A C)
  (h4 : is_on_circumcircle F A B C)
  (h5 : is_parallel MN BC)
  (h6 : MN_meets_circumcircle_at F) :
  let d := (MF / MN) in
  d = (1 + real.sqrt 5) / 2 :=
sorry

end find_d_value_l511_511718


namespace monotonic_intervals_of_f_g_minus_f_less_than_3_l511_511127

noncomputable def f (x : ℝ) : ℝ := -x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - x

theorem monotonic_intervals_of_f :
  ∀ x : ℝ, x < -1 / Real.exp 1 → f x < f (-1 / Real.exp 1) ∧ x > -1 / Real.exp 1 → f x > f (-1 / Real.exp 1) := sorry

theorem g_minus_f_less_than_3 :
  ∀ x : ℝ, x < 0 → g x - f x < 3 := sorry

end monotonic_intervals_of_f_g_minus_f_less_than_3_l511_511127


namespace num_ways_to_insert_plus_signs_l511_511259

theorem num_ways_to_insert_plus_signs : 
  let n := 15
  let m := 10
  ∃ k : ℕ, k = 14.choose 9 ∧ k = 2002 :=
begin
  -- Definitions based on the conditions
  -- n = 15 representing fifteen 1's
  -- m = 10 representing groups summing to a multiple of 10
  let n := 15,
  let m := 10,

  -- The number of ways to insert plus signs so the sum is multiple of 30
  exact ⟨14.choose 9, by norm_num⟩,
  sorry
end

end num_ways_to_insert_plus_signs_l511_511259


namespace parabola_problem_l511_511682

-- Let C be the parabola
def parabola_C (y x : ℝ) : Prop := y^2 = 4 * x

-- Define point F
def focus_F : (ℝ × ℝ) := (1, 0)

-- Define vector equality condition
def vector_condition (P Q F : ℝ × ℝ) : Prop :=
  let (px, py) := P in let (qx, qy) := Q in let (fx, fy) := F in
  (qx - px = 9 * (fx - qx)) ∧ (qy - py = 9 * (fy - qy))

-- Define line slope
def line_slope (P Q : ℝ × ℝ) : ℝ := 
  let (px, py) := P in let (qx, qy) := Q in
  if qx = 0 then 0 else qy / qx

-- Main theorem
theorem parabola_problem :
  (∀ y x : ℝ, parabola_C y x) ∧ 
  (∀ P Q : ℝ × ℝ, (parabola_C (P.2) (P.1)) ∧ (vector_condition P Q focus_F) → line_slope (0, 0) Q = 1 / 3) :=
by {
  sorry
}

end parabola_problem_l511_511682


namespace sphere_tangency_relation_l511_511406

noncomputable def sphere_tangents (r R : ℝ) (h : R > r) :=
  (R >= (2 / (Real.sqrt 3) - 1) * r) ∧
  (∃ x, x = (R * (R + r - Real.sqrt (R^2 + 2 * R * r - r^2 / 3))) /
            (r + Real.sqrt (R^2 + 2 * R * r - r^2 / 3) - R)) 

theorem sphere_tangency_relation (r R: ℝ) (h : R > r) :
  sphere_tangents r R h :=
by
  sorry

end sphere_tangency_relation_l511_511406


namespace balls_in_boxes_l511_511698

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 104 ∧
  let balls := 7;
      boxes := 4 in
  -- Here should be the formal definition of the number of ways to distribute the balls into the boxes,
  -- but we state it as an existential statement acknowledging the result.
  ways = (∑ p in (finset.powerset len_le_boxes (univ (finset.range balls + 1))),
               if (∑ x in p, x) = balls then multinomial p else 0) := sorry

end balls_in_boxes_l511_511698


namespace james_spent_163_for_the_night_l511_511471

noncomputable def totalSpent : ℕ :=
  let club_entry_fee := 20
  let rounds_bought_for_friends := 2
  let friends := 5
  let cost_per_drink := 6
  let drinks_for_himself := 6
  let cost_of_chicken := 14
  let tip_percentage := 0.30 in
  let cost_of_drinks_for_friends := rounds_bought_for_friends * friends * cost_per_drink
  let cost_of_drinks_for_himself := drinks_for_himself * cost_per_drink
  let total_cost_of_drinks := cost_of_drinks_for_friends + cost_of_drinks_for_himself
  let subtotal := club_entry_fee + total_cost_of_drinks + cost_of_chicken
  let tip := tip_percentage * (total_cost_of_drinks + cost_of_chicken) in
  subtotal + tip

theorem james_spent_163_for_the_night : totalSpent = 163 :=
sorry

end james_spent_163_for_the_night_l511_511471


namespace intercept_sum_l511_511896

theorem intercept_sum (x y : ℤ) (h1 : 0 ≤ x) (h2 : x < 42) (h3 : 0 ≤ y) (h4 : y < 42)
  (h : 5 * x ≡ 3 * y - 2 [ZMOD 42]) : (x + y) = 36 :=
by
  sorry

end intercept_sum_l511_511896


namespace simplest_square_root_l511_511880

noncomputable def sqrt8 : ℝ := Real.sqrt 8
noncomputable def inv_sqrt2 : ℝ := 1 / Real.sqrt 2
noncomputable def sqrt2 : ℝ := Real.sqrt 2
noncomputable def sqrt_inv2 : ℝ := Real.sqrt (1 / 2)

theorem simplest_square_root : sqrt2 = Real.sqrt 2 := 
  sorry

end simplest_square_root_l511_511880


namespace monotonic_increasing_implies_a_range_l511_511276

theorem monotonic_increasing_implies_a_range (a : ℝ) :
  (∀ x : ℝ, 1 - (2/3) * Real.cos (2 * x) + a * Real.cos x ≥ 0) →
  - (1 / 3) ≤ a ∧ a ≤ 1 / 3 :=
begin
  sorry
end

end monotonic_increasing_implies_a_range_l511_511276


namespace magic_8_ball_probability_l511_511738

theorem magic_8_ball_probability :
  let p_pos := 1 / 3
  let p_neg := 2 / 3
  let n := 6
  let k := 3
  (Nat.choose n k * (p_pos ^ k) * (p_neg ^ (n - k)) = 160 / 729) :=
by
  sorry

end magic_8_ball_probability_l511_511738


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511188

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the vowels
def vowels := {'A', 'E'}

-- Define the number of n-length words constructible from a set of letters
def num_words (n : ℕ) (alphabet : Set Char) : ℕ :=
  (alphabet.size ^ n)

-- The total number of 5-letter words (unrestricted)
def total_words := num_words 5 letters

-- The number of 5-letter words with no vowels
def no_vowel_words := num_words 5 (letters \ vowels)

-- The number of 5-letter words with at least one vowel
def at_least_one_vowel_words := total_words - no_vowel_words

-- The statement to prove that the number of 5-letter words with at least one vowel is 6752
theorem number_of_5_letter_words_with_at_least_one_vowel : 
  at_least_one_vowel_words = 6752 :=
by 
  -- Proof will be provided here
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l511_511188


namespace find_x_l511_511846

theorem find_x :
  ∀ (x : ℝ), 4.7 * x + 4.7 * 9.43 + 4.7 * 77.31 = 470 → x = 13.26 :=
by
  intro x
  intro h
  sorry

end find_x_l511_511846


namespace five_letter_words_with_vowels_l511_511173

noncomputable def num_5_letter_words_with_vowels : Nat := 7776 - 1024

theorem five_letter_words_with_vowels
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'})
  (len : Nat := 5) :
  (letters.card ^ len) - ((letters.filter (λ c, ¬ c ∈ vowels)).card ^ len) = 6752 := by
  sorry

end five_letter_words_with_vowels_l511_511173


namespace exist_5_sets_disjoint_generalization_n_sets_l511_511453

noncomputable def f_k (k : ℕ) := nat.ceil (5 * k / 2)

noncomputable def f_general (n k : ℕ) : ℕ :=
  if even n then 2 * k 
  else nat.ceil ((2 * n + 1) * k / n)

theorem exist_5_sets_disjoint (k : ℕ) :
  ∃ (f : ℕ) (s1 s2 s3 s4 s5 : finset ℕ), 
    (∀ i : fin 5, (s1 ∪ s2 ∪ s3 ∪ s4 ∪ s5).card = f) ∧
    (∀ i : fin 5, s1.card = k) ∧ 
    (∀ i : fin 5, disjoint s1 (s2 ∪ s4 ∪ s5 ∪ s1 ∪ s3)) ∧
    (f = f_k k) :=
by
  sorry

theorem generalization_n_sets (n k : ℕ) (h : 3 ≤ n) :
  ∃ (f : ℕ) (sets : fin (2 * n + 1) → finset ℕ), 
    (∀ i : fin (2 * n + 1), (univ.bUnion sets).card = f) ∧ 
    (∀ i : fin (2 * n + 1), (sets i).card = k) ∧ 
    (∀ i : fin (2 * n + 1), disjoint (sets i) (sets ((i + 1) % (2 * n + 1)))) ∧ 
    (f = f_general n k) :=
by
  sorry

end exist_5_sets_disjoint_generalization_n_sets_l511_511453


namespace marco_weight_is_15_l511_511339

variable (total_weight father_weight : ℕ)

-- Definition of the condition that together their strawberries weighed 37 pounds, and Dad's strawberries weighed 22 pounds.
axiom condition1 : total_weight = 37
axiom condition2 : father_weight = 22

-- Define the weight of Marco's strawberries
def marco_weight := total_weight - father_weight

-- Prove that Marco's strawberries weighed 15 pounds
theorem marco_weight_is_15 : marco_weight total_weight father_weight = 15 := by
  rw [condition1, condition2]
  rfl

-- Sorry statement to ignore the actual proof
sorry

end marco_weight_is_15_l511_511339


namespace infinite_arithmetic_progression_contains_infinite_squares_l511_511852

noncomputable def infinite_arithmetic_progression (a d : ℕ) : ℕ → ℕ := λ n, a + n * d

theorem infinite_arithmetic_progression_contains_infinite_squares 
  (a d : ℕ) (h1 : ∀ n : ℕ, ∃ k : ℕ, infinite_arithmetic_progression a d n = k * k) :
  ∀ m : ℕ, ∃ k : ℕ, ∃ n : ℕ, infinite_arithmetic_progression a d (m + n) = k * k :=
by
  sorry

end infinite_arithmetic_progression_contains_infinite_squares_l511_511852


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511187

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the vowels
def vowels := {'A', 'E'}

-- Define the number of n-length words constructible from a set of letters
def num_words (n : ℕ) (alphabet : Set Char) : ℕ :=
  (alphabet.size ^ n)

-- The total number of 5-letter words (unrestricted)
def total_words := num_words 5 letters

-- The number of 5-letter words with no vowels
def no_vowel_words := num_words 5 (letters \ vowels)

-- The number of 5-letter words with at least one vowel
def at_least_one_vowel_words := total_words - no_vowel_words

-- The statement to prove that the number of 5-letter words with at least one vowel is 6752
theorem number_of_5_letter_words_with_at_least_one_vowel : 
  at_least_one_vowel_words = 6752 :=
by 
  -- Proof will be provided here
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l511_511187


namespace bob_spending_l511_511078

theorem bob_spending :
  let bread_price := 2
  let bread_count := 4
  let cheese_price := 6
  let cheese_count := 2
  let cheese_discount := 0.25
  let chocolate_price := 3
  let chocolate_count := 3
  let olive_oil_price := 10
  let coupon_threshold := 30
  let coupon_amount := 10
  let original_cheese_total := cheese_price * cheese_count
  let discounted_cheese_total := original_cheese_total * (1 - cheese_discount)
  let total_before_coupon := (bread_price * bread_count) +
                             discounted_cheese_total +
                             (chocolate_price * chocolate_count) +
                             olive_oil_price
  in total_before_coupon >= coupon_threshold →
     total_before_coupon - coupon_amount = 26 :=
by
  sorry

end bob_spending_l511_511078


namespace eccentricity_bound_l511_511309

variables {a b c e : ℝ}

-- Definitions of the problem conditions
def hyperbola (x y : ℝ) (a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def line (x : ℝ) := 2 * x
def eccentricity (c a : ℝ) := c / a

-- Proof statement in Lean
theorem eccentricity_bound (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (c : ℝ)
  (h₃ : hyperbola x y a b)
  (h₄ : ∀ x, line x ≠ y) :
  1 < eccentricity c a ∧ eccentricity c a ≤ sqrt 5 :=
sorry

end eccentricity_bound_l511_511309


namespace infinite_squares_form_l511_511761

theorem infinite_squares_form (k : ℕ) (hk : 0 < k) : ∃ f : ℕ → ℕ, ∀ n, ∃ a, a^2 = f n * 2^k - 7 :=
by
  sorry

end infinite_squares_form_l511_511761


namespace sum_of_differences_l511_511621

theorem sum_of_differences : 
  (∑ n in Finset.range 1000, let first_digit := n / 10 ^ (Nat.log10 n) in
                             let last_digit := n % 10 in
                             first_digit - last_digit) = 495 :=
by
  sorry

end sum_of_differences_l511_511621


namespace day_of_week_299th_day_2004_l511_511711

noncomputable def day_of_week (day: ℕ): ℕ := day % 7

theorem day_of_week_299th_day_2004 : 
  ∀ (d: ℕ), day_of_week d = 3 → d = 45 → day_of_week 299 = 5 :=
by
  sorry

end day_of_week_299th_day_2004_l511_511711


namespace range_of_p_l511_511148

def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

def A : Set ℝ := {x | 3*x^2 - 2*x - 10 ≤ 0}

def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem range_of_p (p : ℝ) (h : A ∪ B p = A) : p ≤ 3 :=
by
  sorry

end range_of_p_l511_511148


namespace steve_sleeping_fraction_l511_511818

def hours_in_a_day : ℝ := 24
def fraction_school : ℝ := 1 / 6
def fraction_assignments : ℝ := 1 / 12
def hours_family : ℝ := 10

theorem steve_sleeping_fraction : (hours_in_a_day - (fraction_school * hours_in_a_day + fraction_assignments * hours_in_a_day + hours_family)) / hours_in_a_day = 1 / 3 := 
by
  sorry

end steve_sleeping_fraction_l511_511818


namespace find_special_set_l511_511079

-- Definitions of the conditions
def increasing_seq (a : list ℤ) : Prop :=
  ∀ (i j : ℕ), i < j → j < a.length → a.get i < a.get j

def all_divides_sum (a : list ℤ) : Prop :=
  let S := a.foldr (+) 0 in
  ∀ (k : ℕ), k < a.length → a.get k ∣ S

-- The main theorem statement
theorem find_special_set (n : ℕ) (a : list ℤ) :
  (n = 3 ∧ a = [1, 2, 3]) ↔
  (2 ≤ n ∧ increasing_seq a ∧ prime (a.get (n - 1)) ∧ all_divides_sum a) :=
begin
  sorry
end

end find_special_set_l511_511079


namespace max_B_at_125_l511_511436

noncomputable def binomial_coefficient (n k : ℕ) : ℝ :=
if h : k ≤ n then (real.fact n) / ((real.fact k) * (real.fact (n - k))) else 0

noncomputable def B (k : ℕ) : ℝ :=
binomial_coefficient 500 k * (0.3)^k

theorem max_B_at_125 : ∃ k : ℕ, k = 125 ∧ (∀ k' : ℕ, 0 ≤ k' ∧ k' ≤ 500 → B k' ≤ B k) :=
by
  sorry

end max_B_at_125_l511_511436


namespace cos_pi_minus_2alpha_eq_seven_over_twentyfive_l511_511106

variable (α : ℝ)

theorem cos_pi_minus_2alpha_eq_seven_over_twentyfive 
  (h : Real.sin (π / 2 - α) = 3 / 5) :
  Real.cos (π - 2 * α) = 7 / 25 := 
by
  sorry

end cos_pi_minus_2alpha_eq_seven_over_twentyfive_l511_511106


namespace five_letter_words_with_vowels_l511_511221

/-
How many 5-letter words with at least one vowel can be constructed from the letters 
A, B, C, D, E, and F? (Note that A and E are vowels, any word is valid, not just English language 
words, and letters may be used more than once.)
-/

theorem five_letter_words_with_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := ['A', 'E'],
      consonants := ['B', 'C', 'D', 'F'] in
  let total_words := 6 ^ 5,
      consonant_only_words := 4 ^ 5,
      at_least_one_vowel_words := total_words - consonant_only_words in
  at_least_one_vowel_words = 6752 :=
by
  intro letters vowels consonants total_words consonant_only_words at_least_one_vowel_words
  sorry

end five_letter_words_with_vowels_l511_511221


namespace dot_product_b_c_l511_511693

variables a b c : ℝ × ℝ
variables m : ℝ

def a := (-1, 2)
def b := (2, m)
def c := (7, 1)

-- If vectors a and b are parallel
axiom a_parallel_b : a.1 * b.2 = a.2 * b.1

theorem dot_product_b_c : b.1 * c.1 + b.2 * c.2 = 10 :=
by
  sorry

end dot_product_b_c_l511_511693


namespace sum_of_squares_not_7_mod_8_l511_511356

theorem sum_of_squares_not_7_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 :=
sorry

end sum_of_squares_not_7_mod_8_l511_511356


namespace sticker_ratio_l511_511776

theorem sticker_ratio (gold : ℕ) (silver : ℕ) (bronze : ℕ)
  (students : ℕ) (stickers_per_student : ℕ)
  (h1 : gold = 50)
  (h2 : bronze = silver - 20)
  (h3 : students = 5)
  (h4 : stickers_per_student = 46)
  (h5 : gold + silver + bronze = students * stickers_per_student) :
  silver / gold = 2 / 1 :=
by
  sorry

end sticker_ratio_l511_511776


namespace sum_of_differences_l511_511582

open Nat
open BigOperators

theorem sum_of_differences (n : ℕ) (h : n ≥ 1 ∧ n ≤ 999) : 
  let differences := (fun x => 
                        let first_digit := x / 10;
                        let last_digit := x % 10;
                        first_digit - last_digit) in
  ∑ i in (finset.range (n + 1)).filter (λ x, 1 ≤ x ∧ x ≤ 999), differences i = 495 :=
by
  -- Acknowledge the need for a more refined filtering criteria for numbers between 1 and 999
  sorry

end sum_of_differences_l511_511582


namespace relationship_abc_l511_511124

noncomputable def odd_increasing_function : Type := 
  { f : ℝ → ℝ // (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x < f y) }

theorem relationship_abc (f : odd_increasing_function) :
  let a := -f.val (Real.logBase 2 (1/5))
  let b := f.val (Real.logBase 2 4.1)
  let c := f.val (2^0.8)
  c < b ∧ b < a :=
by
  sorry

end relationship_abc_l511_511124


namespace reduction_percentage_40_l511_511270

theorem reduction_percentage_40 (P : ℝ) : 
  1500 * 1.20 - (P / 100 * (1500 * 1.20)) = 1080 ↔ P = 40 :=
by
  sorry

end reduction_percentage_40_l511_511270


namespace parabola_problem_l511_511679

-- Let C be the parabola
def parabola_C (y x : ℝ) : Prop := y^2 = 4 * x

-- Define point F
def focus_F : (ℝ × ℝ) := (1, 0)

-- Define vector equality condition
def vector_condition (P Q F : ℝ × ℝ) : Prop :=
  let (px, py) := P in let (qx, qy) := Q in let (fx, fy) := F in
  (qx - px = 9 * (fx - qx)) ∧ (qy - py = 9 * (fy - qy))

-- Define line slope
def line_slope (P Q : ℝ × ℝ) : ℝ := 
  let (px, py) := P in let (qx, qy) := Q in
  if qx = 0 then 0 else qy / qx

-- Main theorem
theorem parabola_problem :
  (∀ y x : ℝ, parabola_C y x) ∧ 
  (∀ P Q : ℝ × ℝ, (parabola_C (P.2) (P.1)) ∧ (vector_condition P Q focus_F) → line_slope (0, 0) Q = 1 / 3) :=
by {
  sorry
}

end parabola_problem_l511_511679


namespace exists_positive_integer_k_l511_511895

noncomputable def f (u x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ u then 0 else 1 - (Real.sqrt (u * x) + Real.sqrt ((1 - u) * (1 - x)))^2

def u_seq (u : ℝ) : ℕ → ℝ
| 0     := u
| (n+1) := f u (u_seq u n)

theorem exists_positive_integer_k (u : ℝ) (h : 0 < u ∧ u < 1) : ∃ k : ℕ, k > 0 ∧ u_seq u k = 0 :=
sorry

end exists_positive_integer_k_l511_511895


namespace sum_of_differences_l511_511599

theorem sum_of_differences : 
  let first_digit (n : ℕ) : ℕ := n / 10^(nat.log10 n)
  let last_digit (n : ℕ) : ℕ := n % 10
  (finset.range 1000).sum (λ n, first_digit n - last_digit n) = 495 :=
by
  sorry

end sum_of_differences_l511_511599


namespace number_of_elements_in_intersection_l511_511337

-- Definitions of sets A and B based on given conditions
def A := {x : ℕ | (1 / 4 : ℝ) ≤ (2 : ℝ)^x ∧ (2 : ℝ)^x ≤ 16}
def B := {x : ℕ | ∃ (y : ℝ), y = Real.log (x^2 - 3*x)}

-- The theorem to be proven: The number of elements in A ∩ B is 1
theorem number_of_elements_in_intersection :
  Finset.card (Finset.filter (λ x, x ∈ B) (Finset.filter (λ x, x ∈ A) Finset.univ)) = 1 :=
by sorry

end number_of_elements_in_intersection_l511_511337


namespace f_even_function_f_periodic_function_f_values_at_integers_l511_511757

noncomputable def f : ℝ → ℝ :=
-- assume the existence of such a function that satisfies the conditions
sorry

theorem f_even_function (x : ℝ) :
  f(x) = f(-x) :=
sorry

theorem f_periodic_function (x : ℝ) :
  f(x + 4) = f(x) :=
sorry

theorem f_values_at_integers (n : ℕ) :
  f(n) = if n % 4 == 0 then 1
         else if n % 4 == 1 || n % 4 == 3 then 0
         else -1 :=
sorry

end f_even_function_f_periodic_function_f_values_at_integers_l511_511757


namespace gcd_probability_is_one_l511_511419

open Set Nat

theorem gcd_probability_is_one :
  let S := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := (finset.powerset_len 2 (finset.image id S.to_finset)).card
  let non_rel_prime_pairs := 6
  (finset.card (finset.filter (λ (p : Finset ℕ), p.gcdₓ = 1) 
                                (finset.powerset_len 2 (finset.image id S.to_finset)))) / 
  total_pairs = 11 / 14 :=
sorry

end gcd_probability_is_one_l511_511419


namespace smallest_n_for_gn_gt_15_l511_511268

def digit_sum (n : ℕ) : ℕ :=
  n.digits_base 10 |>.sum

def g (n : ℕ) : ℕ :=
  digit_sum (2^n + 3^n)

theorem smallest_n_for_gn_gt_15 : ∃ n : ℕ, n > 0 ∧ g(n) > 15 ∧ ∀ m : ℕ, m > 0 ∧ m < n → g(m) ≤ 15 := by
  sorry

end smallest_n_for_gn_gt_15_l511_511268


namespace max_soap_boxes_l511_511476

def volume_carton : ℕ := 63000
def base_area_carton : ℕ := 2100
def min_height_carton : ℕ := 24
def dim_l_soap : ℕ := 7
def dim_w_soap : ℕ := 6
def dim_h_soap : ℕ := 6

theorem max_soap_boxes (V_carton : ℕ) (A_base : ℕ) (h_min : ℕ) (l_soap : ℕ) (w_soap : ℕ) (h_soap : ℕ) :
  V_carton = volume_car & A_base = base_area_carton & h_min = min_height_carton &
  l_soap = dim_l_soap & w_soap = dim_w_soap & h_soap = dim_h_soap) → 
  ⌊V_carton / (l_soap * w_soap * h_soap)⌋ = 200 :=
sorry

end max_soap_boxes_l511_511476


namespace product_of_possible_x_values_l511_511704

theorem product_of_possible_x_values : 
  (∃ x1 x2 : ℚ, 
    (|15 / x1 + 4| = 3 ∧ |15 / x2 + 4| = 3) ∧
    -15 * -(15 / 7) = (225 / 7)) :=
sorry

end product_of_possible_x_values_l511_511704


namespace sum_of_differences_l511_511622

theorem sum_of_differences : 
  (∑ n in Finset.range 1000, let first_digit := n / 10 ^ (Nat.log10 n) in
                             let last_digit := n % 10 in
                             first_digit - last_digit) = 495 :=
by
  sorry

end sum_of_differences_l511_511622


namespace triangle_ABC_angles_l511_511082

theorem triangle_ABC_angles :
  ∃ (θ φ ω : ℝ), θ = 36 ∧ φ = 72 ∧ ω = 72 ∧
  (ω + φ + θ = 180) ∧
  (2 * ω + θ = 180) ∧
  (φ = 2 * θ) :=
by
  sorry

end triangle_ABC_angles_l511_511082


namespace library_system_finish_time_l511_511018

def start_time := 9.5 -- 9:30 AM as 9.5 hours
def half_time := 12.0 -- 12:00 PM as 12 hours
def half_duration := half_time - start_time
def total_duration := 2 * half_duration

def completion_time := start_time + total_duration

theorem library_system_finish_time : completion_time = 14.5 := by
  unfold start_time half_time half_duration total_duration completion_time
  rw [←add_assoc, half_duration, start_time, half_time]
  norm_num
  sorry

end library_system_finish_time_l511_511018


namespace five_letter_words_with_vowel_l511_511242

-- Define the set of letters and identify vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels := {'A', 'E'}

-- Define the problem statement
theorem five_letter_words_with_vowel : 
  (number of 5-letter words with at least one vowel) = 6752 := 
sorry

end five_letter_words_with_vowel_l511_511242


namespace maximum_value_of_expression_l511_511985

theorem maximum_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (xyz * (x + y + z)) / ((x + y)^2 * (y + z)^2) ≤ (1 / 4) :=
sorry

end maximum_value_of_expression_l511_511985


namespace five_letter_words_with_at_least_one_vowel_l511_511210

open Finset

theorem five_letter_words_with_at_least_one_vowel :
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := ((letters \ vowels).card : ℕ) ^ 5
  total_words - no_vowel_words = 6752 :=
by
  let letters := insert 'A' (insert 'B' (insert 'C' (insert 'D' (insert 'E' (singleton 'F')))))
  let vowels := insert 'A' (singleton 'E')
  let consonants := letters \ vowels
  have : letters.card = 6 := by simp
  have : vowels.card = 2 := by simp
  have : consonants.card = 4 := by simp
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := (consonants.card : ℕ) ^ 5
  calc
    total_words - no_vowel_words
        = 6^5 - 4^5         : by simp
    ... = 7776 - 1024       : by norm_num
    ... = 6752             : by norm_num

end five_letter_words_with_at_least_one_vowel_l511_511210


namespace f_5_eq_2_l511_511960

def f : ℕ → ℤ :=
sorry

axiom f_initial_condition : f 1 = 2

axiom f_functional_eq (a b : ℕ) : f (a + b) = 2 * f a + 2 * f b - 3 * f (a * b)

theorem f_5_eq_2 : f 5 = 2 :=
sorry

end f_5_eq_2_l511_511960


namespace five_letter_words_with_vowels_l511_511181

noncomputable def num_5_letter_words_with_vowels : Nat := 7776 - 1024

theorem five_letter_words_with_vowels
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'})
  (len : Nat := 5) :
  (letters.card ^ len) - ((letters.filter (λ c, ¬ c ∈ vowels)).card ^ len) = 6752 := by
  sorry

end five_letter_words_with_vowels_l511_511181


namespace soccer_tickets_l511_511984

theorem soccer_tickets:
  ∃ (a b c : ℕ), 
  a + b + c = 400 ∧ 
  50 * a + 40 * b + 30 * c = 15500 ∧ 
  b = c ∧ 
  a = 100 ∧ 
  b = 150 ∧ 
  c = 150 :=
by 
  use 100, 150, 150
  simp
  split
  { 
    -- Total tickets equation
    guard_target = 100 + 150 + 150 = 400
    exact dec_trivial 
  }
  split
  {
    -- Total revenue equation
    guard_target = 50 * 100 + 40 * 150 + 30 * 150 = 15500
    exact dec_trivial
  }
  split 
  {
    -- b equals c
    guard_target = 150 = 150
    exact dec_trivial
  }
  split 
  {
    -- Explicit value for a
    guard_target = 100 = 100
    exact dec_trivial
  }
  split 
  {
    -- Explicit value for b
    guard_target = 150 = 150
    exact dec_trivial
  } 
  {
    -- Explicit value for c
    guard_target = 150 = 150
    exact dec_trivial
  }

end soccer_tickets_l511_511984


namespace compute_vector_expression_l511_511058

theorem compute_vector_expression :
  4 • (⟨3, -5⟩ : ℝ × ℝ) - 3 • (⟨2, -6⟩ : ℝ × ℝ) + 2 • (⟨0, 3⟩ : ℝ × ℝ) = (⟨6, 4⟩ : ℝ × ℝ) := 
sorry

end compute_vector_expression_l511_511058


namespace cyclist_speed_l511_511016

def hiker_speed : ℝ := 7 -- miles per hour
def cyclist_stop_time : ℝ := 5 / 60 -- hours
def cyclist_wait_time : ℝ := 15 / 60 -- hours
def distance_hiker_after_wait := hiker_speed * cyclist_wait_time -- distance covered by hiker during cyclist wait time

theorem cyclist_speed : ∃ c : ℝ, distance_hiker_after_wait = (c * cyclist_stop_time) ∧ c = 21 :=
by
  -- dist_hiker_wait is miles hiker walks in 15 minutes
  let dist_hiker_wait := distance_hiker_after_wait
  -- We know distance_hiker_after_wait should equal cyclist_speed (c) x cyclist_stop_time in miles
  let c := 21
  -- prove distance_here is consistent with speed 21
  have h : dist_hiker_wait = c * cyclist_stop_time := sorry
  exact ⟨c, h, rfl⟩

end cyclist_speed_l511_511016


namespace angle_relationship_l511_511296

theorem angle_relationship:
  ∀ (A B C D E F : Type)
    (a b c : ℝ),
    (equilateral_triangle A B C) →
    (is_isosceles_triangle DEF with DE = EF) →
    (angle BFD = a) →
    (angle ADE = b) →
    (angle FEC = c) →
    a = (c + 90) / 2 :=
by
  sorry

end angle_relationship_l511_511296


namespace decreasing_on_interval_l511_511355

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

theorem decreasing_on_interval : ∀ (x : ℝ), x ∈ set.Ioo (-2 : ℝ) 1 → (derivative f x) < 0 := 
by 
  -- Here we use notation to indicate that the proof is omitted
  sorry

end decreasing_on_interval_l511_511355


namespace ways_to_distribute_balls_l511_511258

theorem ways_to_distribute_balls :
  let balls : Finset ℕ := {0, 1, 2, 3, 4, 5, 6}
  let boxes : Finset ℕ := {0, 1, 2, 3}
  let choose_distinct (n k : ℕ) : ℕ := Nat.choose n k
  let distribution_patterns : List (ℕ × ℕ × ℕ × ℕ) := 
    [(6,0,0,0), (5,1,0,0), (4,2,0,0), (4,1,1,0), 
     (3,3,0,0), (3,2,1,0), (3,1,1,1), (2,2,2,0), (2,2,1,1)]
  let ways_to_pattern (pattern : ℕ × ℕ × ℕ × ℕ) : ℕ :=
    match pattern with
    | (6,0,0,0) => 1
    | (5,1,0,0) => choose_distinct 6 5
    | (4,2,0,0) => choose_distinct 6 4 * choose_distinct 2 2
    | (4,1,1,0) => choose_distinct 6 4
    | (3,3,0,0) => choose_distinct 6 3 * choose_distinct 3 3 / 2
    | (3,2,1,0) => choose_distinct 6 3 * choose_distinct 3 2 * choose_distinct 1 1
    | (3,1,1,1) => choose_distinct 6 3
    | (2,2,2,0) => choose_distinct 6 2 * choose_distinct 4 2 * choose_distinct 2 2 / 6
    | (2,2,1,1) => choose_distinct 6 2 * choose_distinct 4 2 / 2
    | _ => 0
  let total_ways : ℕ := distribution_patterns.foldl (λ acc x => acc + ways_to_pattern x) 0
  total_ways = 182 := by
  sorry

end ways_to_distribute_balls_l511_511258


namespace five_letter_words_with_vowels_l511_511177

noncomputable def num_5_letter_words_with_vowels : Nat := 7776 - 1024

theorem five_letter_words_with_vowels
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'})
  (len : Nat := 5) :
  (letters.card ^ len) - ((letters.filter (λ c, ¬ c ∈ vowels)).card ^ len) = 6752 := by
  sorry

end five_letter_words_with_vowels_l511_511177


namespace parabola_equation_and_max_slope_l511_511685

-- Define the parabola parameters and conditions
def parabola (p : ℝ) (hp : p > 0) : set (ℝ × ℝ) := 
{xy | xy.2 ^ 2 = 2 * p * xy.1}

def focus_distance (d : ℝ) : Prop := 
d = 2

-- Define the points O, P, and Q and the given vector relationship
def point_o : ℝ × ℝ := (0, 0)

def on_parabola (p : ℝ) (hp : p > 0) (P : ℝ × ℝ) : Prop :=
P ∈ parabola p hp

def vector_relationship (P Q F : ℝ × ℝ) : Prop :=
P.1 - Q.1 = 9 * (Q.1 - F.1) ∧ P.2 - Q.2 = 9 * (Q.2 - F.2)

-- Define the conditions and the proof goals
theorem parabola_equation_and_max_slope :
  ∃ p (hp : p > 0) (F : ℝ × ℝ),
  focus_distance (2 * p) →
  (∀ P, on_parabola p hp P → 
       ∃ Q, vector_relationship P Q F →
             (parabola p hp = (λ xy, xy.2^2 = 4 * xy.1) ∧
             (real.slope point_o Q ≤ (1/3))) :=
by 
  -- Proof is omitted
  sorry

end parabola_equation_and_max_slope_l511_511685


namespace cube_surface_area_given_diagonal_distance_l511_511824

theorem cube_surface_area_given_diagonal_distance
  (x d : ℝ)
  (hx_face_diag : ∀ x, face_diagonal x = x * Real.sqrt 2)
  (hx_space_diag : ∀ x, space_diagonal x = x * Real.sqrt 3)
  (h_distance : distance_between_diagonals x = d)
  (h_height : ∀ x, height_to_diagonal x = x / Real.sqrt 3)
  (h_relation : ∀ x d, x / Real.sqrt 3 = (x * Real.sqrt 3 - d) / 2) :
  surface_area x = 18 * d^2 :=
by
  sorry

noncomputable def face_diagonal (x : ℝ) : ℝ := x * Real.sqrt 2

noncomputable def space_diagonal (x : ℝ) : ℝ := x * Real.sqrt 3

noncomputable def distance_between_diagonals (x : ℝ) : ℝ := d

noncomputable def height_to_diagonal (x : ℝ) : ℝ := x / Real.sqrt 3

noncomputable def surface_area (x : ℝ) : ℝ := 6 * x^2

end cube_surface_area_given_diagonal_distance_l511_511824


namespace proof_problem_l511_511158

-- Condition for the first part: a quadratic inequality having a solution set
def quadratic_inequality (a : ℝ) :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → a * x^2 - 3 * x + 2 ≤ 0

-- Condition for the second part: the solution set of a rational inequality
def rational_inequality_solution (a : ℝ) (b : ℝ) :=
  ∀ x : ℝ, (x + 3) / (a * x - b) > 0 ↔ (x < -3 ∨ x > 2)

theorem proof_problem {a : ℝ} {b : ℝ} :
  (quadratic_inequality a → a = 1 ∧ b = 2) ∧ 
  (rational_inequality_solution 1 2) :=
by
  sorry

end proof_problem_l511_511158


namespace range_of_p_l511_511150

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

-- A = { x | f'(x) ≤ 0 }
def A : Set ℝ := { x | deriv f x ≤ 0 }

-- B = { x | p + 1 ≤ x ≤ 2p - 1 }
def B (p : ℝ) : Set ℝ := { x | p + 1 ≤ x ∧ x ≤ 2 * p - 1 }

-- Given that A ∪ B = A, prove the range of values for p is ≤ 3.
theorem range_of_p (p : ℝ) : (A ∪ B p = A) → p ≤ 3 := sorry

end range_of_p_l511_511150


namespace monotonicity_of_f_range_of_k_minimum_value_of_a_l511_511116

-- Definition of the functions and derivatives
def f (x a : ℝ) := -x^3 + (1 / 2) * a * x^2
def g (x a : ℝ) := a * log x

def f' (x a : ℝ) := -3 * x^2 + a * x
def g' (x a : ℝ) := a / x

-- Proof of monotonicity of f(x)
theorem monotonicity_of_f (a : ℝ) : 
  (a < 0 → monotonic_decreasing_on ℝ (f a) ((-\infty, a/3) ∪ (0, +∞)) ∧ monotonic_increasing_on ℝ (f a) ((a/3, 0))) ∧
  (a = 0 → monotonic_decreasing ℝ (f a)) ∧
  (a > 0 → monotonic_decreasing_on ℝ (f a) ((-\infty, 0) ∪ (a/3, +∞)) ∧ monotonic_increasing_on ℝ (f a) ((0, a/3))) :=
sorry

-- Proof for the range of k when a=3
theorem range_of_k (k : ℝ) : 
  (∀ x > 1, f' x 3 ≤ k * g x 3) ↔ k ≥ -1 := 
sorry

-- Proof for the minimum value of a when a > 0
theorem minimum_value_of_a (a : ℝ) (h : a > 0) :
  (∃ x0 > 0, √(f' x0 a) ≥ g' x0 a) → a ≥ 16 :=
sorry

end monotonicity_of_f_range_of_k_minimum_value_of_a_l511_511116


namespace sin_double_angle_cos_double_angle_tan_double_angle_l511_511642

-- Given conditions
variable (α : ℝ)
hypothesis (h1 : Real.sin α = 3/5)
hypothesis (h2 : π/2 ≤ α ∧ α ≤ π)

-- Statements to be proved
theorem sin_double_angle : Real.sin (2 * α) = -24 / 25 := 
by sorry

theorem cos_double_angle : Real.cos (2 * α) = 7 / 25 :=
by sorry

theorem tan_double_angle : Real.tan (2 * α) = -24 / 7 :=
by sorry

end sin_double_angle_cos_double_angle_tan_double_angle_l511_511642


namespace calculate_expr_l511_511523

noncomputable def expr := (37.921 / 3.569) * 11.798 - (5.392 + 2.659)^2 + (4.075 * 8.391) / 17.243 

theorem calculate_expr :
  expr ≈ 62.563 := 
by 
  sorry

end calculate_expr_l511_511523


namespace shaded_area_is_correct_l511_511956

noncomputable def calculateShadedArea : ℝ := 
  let radius_small := 3
  let radius_large := 5
  let height := 2 * (radius_large + 2)
  let width := (1 + radius_large + 1) + (1 + radius_small + 1)
  let rect_area := width * height
  let area_small_circle := Real.pi * radius_small^2
  let area_large_circle := Real.pi * radius_large^2
  rect_area - (area_small_circle + area_large_circle)

theorem shaded_area_is_correct : calculateShadedArea = 168 - 34 * Real.pi :=
by
  sorry

end shaded_area_is_correct_l511_511956


namespace directrix_parabola_l511_511083

theorem directrix_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 8 * x^2 + 5) : 
  ∃ c : ℝ, ∀ x, y x = 8 * x^2 + 5 ∧ c = 159 / 32 :=
by
  use 159 / 32
  repeat { sorry }

end directrix_parabola_l511_511083


namespace arccos_cos_10_l511_511059

noncomputable def arccos_cos_10_eq : Prop :=
  real.arccos (real.cos 10) = 3.716814

theorem arccos_cos_10 : arccos_cos_10_eq :=
by
  sorry

end arccos_cos_10_l511_511059


namespace no_neighboring_beads_same_color_probability_l511_511988

theorem no_neighboring_beads_same_color_probability : 
  let total_beads := 9
  let count_red := 4
  let count_white := 3
  let count_blue := 2
  let total_permutations := Nat.factorial total_beads / (Nat.factorial count_red * Nat.factorial count_white * Nat.factorial count_blue)
  ∃ valid_permutations : ℕ,
  valid_permutations = 100 ∧
  valid_permutations / total_permutations = 5 / 63 := by
  sorry

end no_neighboring_beads_same_color_probability_l511_511988


namespace sum_differences_1_to_999_l511_511591

-- Define a utility function to compute the first digit of a number
def first_digit (n : ℕ) : ℕ :=
if n < 10 then n else first_digit (n / 10)

-- Define a utility function to compute the last digit of a number
def last_digit (n : ℕ) : ℕ :=
n % 10

-- Define the operation performed by Damir
def difference (n : ℕ) : ℤ :=
(first_digit n : ℤ) - (last_digit n : ℤ)

-- Define the sum of all differences from 1 to 999
def sum_differences : ℤ :=
(1).to (999).sum (λ n, difference n)

-- State the main theorem to be proved using the previous definitions
theorem sum_differences_1_to_999 : sum_differences = 495 :=
sorry

end sum_differences_1_to_999_l511_511591


namespace fraction_power_simplification_l511_511951

theorem fraction_power_simplification:
  (81000/9000)^3 = 729 → (81000^3) / (9000^3) = 729 :=
by 
  intro h
  rw [<- h]
  sorry

end fraction_power_simplification_l511_511951


namespace parabola_problem_l511_511680

-- Let C be the parabola
def parabola_C (y x : ℝ) : Prop := y^2 = 4 * x

-- Define point F
def focus_F : (ℝ × ℝ) := (1, 0)

-- Define vector equality condition
def vector_condition (P Q F : ℝ × ℝ) : Prop :=
  let (px, py) := P in let (qx, qy) := Q in let (fx, fy) := F in
  (qx - px = 9 * (fx - qx)) ∧ (qy - py = 9 * (fy - qy))

-- Define line slope
def line_slope (P Q : ℝ × ℝ) : ℝ := 
  let (px, py) := P in let (qx, qy) := Q in
  if qx = 0 then 0 else qy / qx

-- Main theorem
theorem parabola_problem :
  (∀ y x : ℝ, parabola_C y x) ∧ 
  (∀ P Q : ℝ × ℝ, (parabola_C (P.2) (P.1)) ∧ (vector_condition P Q focus_F) → line_slope (0, 0) Q = 1 / 3) :=
by {
  sorry
}

end parabola_problem_l511_511680


namespace complement_intersection_eq_l511_511689

-- Definitions of sets
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 5, 8}
def B : Set ℕ := {1, 3, 5, 7}

-- Definition of complement of A in U
def complement_U_A : Set ℕ := U \ A

-- The main statement to prove
theorem complement_intersection_eq :
  (complement_U_A ∩ B) = {1, 3, 7} :=
by sorry

end complement_intersection_eq_l511_511689


namespace infinite_solutions_no_solutions_l511_511648

-- Define the geometric sequence with first term a1 = 1 and common ratio q
def a1 : ℝ := 1
def a2 (q : ℝ) : ℝ := a1 * q
def a3 (q : ℝ) : ℝ := a1 * q^2
def a4 (q : ℝ) : ℝ := a1 * q^3

-- Define the system of linear equations
def system_of_eqns (x y q : ℝ) : Prop :=
  a1 * x + a3 q * y = 3 ∧ a2 q * x + a4 q * y = -2

-- Conditions for infinitely many solutions
theorem infinite_solutions (q x y : ℝ) :
  q = -2 / 3 → ∃ x y, system_of_eqns x y q :=
by
  sorry

-- Conditions for no solutions
theorem no_solutions (q : ℝ) :
  q ≠ -2 / 3 → ¬∃ x y, system_of_eqns x y q :=
by
  sorry

end infinite_solutions_no_solutions_l511_511648


namespace new_percentage_of_girls_is_5_l511_511732

theorem new_percentage_of_girls_is_5
  (initial_children : ℕ)
  (percentage_boys : ℕ)
  (added_boys : ℕ)
  (initial_total_boys : ℕ)
  (initial_total_girls : ℕ)
  (new_total_boys : ℕ)
  (new_total_children : ℕ)
  (new_percentage_girls : ℕ)
  (h1 : initial_children = 60)
  (h2 : percentage_boys = 90)
  (h3 : added_boys = 60)
  (h4 : initial_total_boys = (percentage_boys * initial_children / 100))
  (h5 : initial_total_girls = initial_children - initial_total_boys)
  (h6 : new_total_boys = initial_total_boys + added_boys)
  (h7 : new_total_children = initial_children + added_boys)
  (h8 : new_percentage_girls = (initial_total_girls * 100 / new_total_children)) :
  new_percentage_girls = 5 :=
by sorry

end new_percentage_of_girls_is_5_l511_511732


namespace relatively_prime_probability_l511_511426

open Finset

theorem relatively_prime_probability :
  let s := ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)
  in let pairs := s.val.powerset.filter (λ t, t.card = 2)
  in (pairs.count (λ t, (t : Multiset ℕ).gcd = 1)).toRational / pairs.card.toRational = 3 / 4 := 
by
  -- Prove that the probability is 3/4
  sorry

end relatively_prime_probability_l511_511426


namespace valid_words_no_adjacent_a_l511_511160

def a_n (n : ℕ) : ℝ :=
  let A := (2 + Real.sqrt 3) / (2 * Real.sqrt 3)
  let B := (-2 + Real.sqrt 3) / (2 * Real.sqrt 3)
  in A * ((1 + Real.sqrt 3) ^ n) + B * ((1 - Real.sqrt 3) ^ n)

theorem valid_words_no_adjacent_a (n : ℕ) (h : n ≥ 1) :
  a_n n = (2 + Real.sqrt 3) / (2 * Real.sqrt 3) * ((1 + Real.sqrt 3) ^ n) + 
          (-2 + Real.sqrt 3) / (2 * Real.sqrt 3) * ((1 - Real.sqrt 3) ^ n) :=
by sorry

end valid_words_no_adjacent_a_l511_511160


namespace temperature_analysis_l511_511364

theorem temperature_analysis 
  (temps : List ℝ) 
  (h_len : temps.length = 8) 
  (h_vals : temps = [38, 39, 39, 41, 40, 39, 37, 37]) : 
  (List.sum temps / 8 = 38.75) ∧ 
  (let mean := 38.75 in 
   List.sum (temps.map (λ x, (x - mean)^2)) / 8 = 27 / 16) ∧ 
  (List.nth_le (List.sort temps) 6 sorry + List.nth_le (List.sort temps) 5 sorry) / 2 ≠ 39 ∧
  (List.maximum' h_len temps - List.minimum' h_len temps = 4) :=
by 
  sorry

end temperature_analysis_l511_511364


namespace sum_differences_1_to_999_l511_511592

-- Define a utility function to compute the first digit of a number
def first_digit (n : ℕ) : ℕ :=
if n < 10 then n else first_digit (n / 10)

-- Define a utility function to compute the last digit of a number
def last_digit (n : ℕ) : ℕ :=
n % 10

-- Define the operation performed by Damir
def difference (n : ℕ) : ℤ :=
(first_digit n : ℤ) - (last_digit n : ℤ)

-- Define the sum of all differences from 1 to 999
def sum_differences : ℤ :=
(1).to (999).sum (λ n, difference n)

-- State the main theorem to be proved using the previous definitions
theorem sum_differences_1_to_999 : sum_differences = 495 :=
sorry

end sum_differences_1_to_999_l511_511592


namespace maximum_triangles_l511_511747

variable {α : Type*} [RealField α] (S : Finset (α × α)) (n : ℕ)
(hS : S.card = n) (no_three_collinear : ∀ (A B C : α × α), A ∈ S → B ∈ S → C ∈ S → (A, B, C) non_collinear)
(area1 : ∀ (A B C : α × α), A ∈ S → B ∈ S → C ∈ S → triangle_area A B C = 1)

theorem maximum_triangles : 
  ∃ k, k = S.triangle_count 1 ∧ k ≤ (2 * n * (n - 1)) / 3 :=
sorry

end maximum_triangles_l511_511747


namespace minimum_distance_from_parabola_to_circle_l511_511133

noncomputable def minimum_distance_sum : ℝ :=
  let focus : ℝ × ℝ := (1, 0)
  let center : ℝ × ℝ := (0, 4)
  let radius : ℝ := 1
  let distance_from_focus_to_center : ℝ := Real.sqrt ((focus.1 - center.1)^2 + (focus.2 - center.2)^2)
  distance_from_focus_to_center - radius

theorem minimum_distance_from_parabola_to_circle : minimum_distance_sum = Real.sqrt 17 - 1 := by
  sorry

end minimum_distance_from_parabola_to_circle_l511_511133


namespace distance_to_line_range_l511_511653

theorem distance_to_line_range (a : ℝ) : 
  (abs (15 - 3 * a) / 5 ≤ 3) → (0 ≤ a ∧ a ≤ 10) :=
begin
  sorry -- proof goes here
end

end distance_to_line_range_l511_511653


namespace sum_diff_1_to_999_l511_511629

def subtract_last_from_first (n : ℕ) : ℤ :=
  let str_n := n.toString
  if str_n.length = 1 then 0
  else
    let first_digit := str_n.toList.head!.digitToInt!
    let last_digit := str_n.toList.reverse.head!.digitToInt!
    first_digit - last_digit

def numbers : List ℕ := List.range 1000.tail

def sum_of_differences : ℤ := (numbers.map subtract_last_from_first).sum

theorem sum_diff_1_to_999 :
  sum_of_differences = 495 := 
sorry

end sum_diff_1_to_999_l511_511629


namespace linda_paint_area_l511_511768

noncomputable def total_paint_area
  (room_width room_length room_height door_width door_height window_width window_height closet_door_width closet_door_height : ℝ) 
  (walls_count : ℕ) : ℝ :=
let wall_area := room_width * room_height in
let doorway_area := door_width * door_height in
let window_area := window_width * window_height in
let closet_door_area := closet_door_width * closet_door_height in
let total_door_window_area := doorway_area + window_area + closet_door_area in
let total_wall_area_before := walls_count * wall_area in
total_wall_area_before - total_door_window_area

theorem linda_paint_area :
  total_paint_area 20 20 8 3 7 6 4 5 7 4 = 560 :=
by simp [total_paint_area]; norm_num; sorry

end linda_paint_area_l511_511768


namespace sum_of_differences_l511_511625

theorem sum_of_differences : 
  (∑ n in Finset.range 1000, let first_digit := n / 10 ^ (Nat.log10 n) in
                             let last_digit := n % 10 in
                             first_digit - last_digit) = 495 :=
by
  sorry

end sum_of_differences_l511_511625


namespace polar_coordinate_eq_C1_rectangular_coordinate_eq_l_length_MN_after_move_l511_511008

-- Given conditions
def curve_C1 := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}
def point_A := (3*real.sqrt 2, real.pi / 4)
def polar_eq_l (ρ θ a : ℝ) := ρ * real.cos (θ - real.pi / 4) = a

-- Proving the polar coordinate equation of curve C1
theorem polar_coordinate_eq_C1 {ρ θ : ℝ} (h : curve_C1 (ρ * real.cos θ, ρ * real.sin θ)) :
  ρ = 4 * real.cos θ :=
by
  sorry

-- Proving the rectangular coordinate equation of line l
theorem rectangular_coordinate_eq_l (a : ℝ) (h : polar_eq_l (3 * real.sqrt 2) (real.pi / 4) a) :
  ∃ (x y : ℝ), x + y - 6 = 0 :=
by
  sorry

-- Proving the length of |MN| after moving the line segment
theorem length_MN_after_move :
  ∃ (ρ1 ρ2 : ℝ), (ρ1 = 0 ∧ ρ2 = -2 * real.sqrt 2) ∧ |ρ2 - ρ1| = 2 * real.sqrt 2 :=
by
  sorry

end polar_coordinate_eq_C1_rectangular_coordinate_eq_l_length_MN_after_move_l511_511008


namespace rotated_ln_graph_identity_l511_511381

theorem rotated_ln_graph_identity :
  (∀ x : ℝ, 0 < x → ∃ y : ℝ, y = -exp x ∧ (x = ln x)) :=
by
  sorry

end rotated_ln_graph_identity_l511_511381


namespace evaluate_expression_l511_511566

theorem evaluate_expression : 3 + 2 * (8 - 3) = 13 := by
  sorry

end evaluate_expression_l511_511566


namespace sum_of_differences_l511_511612

/-- Proving that the sum of all differences (first digit - last digit) for natural numbers from 
    1 to 999 is 495. -/ 
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, 
    let str := n.toString
    let first := if str.length > 1 then str.head!.toNat - '0'.toNat else 0
    let last := if str.length > 1 then str.getLast.toNat - '0'.toNat else 0
    first - last
  ) = 495 := 
by
  sorry

end sum_of_differences_l511_511612


namespace scalene_triangle_angle_difference_l511_511929

def scalene_triangle (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem scalene_triangle_angle_difference (x y : ℝ) :
  (x + y = 100) → scalene_triangle x y 80 → (x - y = 80) :=
by
  intros h1 h2
  sorry

end scalene_triangle_angle_difference_l511_511929


namespace smallest_integer_satisfying_inequality_l511_511444

theorem smallest_integer_satisfying_inequality :
  ∃ (n : ℤ), n^2 - 15 * n + 56 ≤ 0 ∧ ∀ m : ℤ, m^2 - 15 * m + 56 ≤ 0 → n ≤ m :=
begin
  use 7,
  split,
  { -- n = 7 satisfies the inequality
    calc
      7^2 - 15 * 7 + 56 = 49 - 105 + 56 : by norm_num
      ... = 0 : by norm_num,
    exact le_refl 0, },
  { -- 7 is the smallest such integer
    intros m hm,
    have A : (n - 7) * (n - 8) ≤ 0, from by {
      calc
        n^2 - 15 * n + 56 = (n - 7) * (n - 8) : by ring,
      exact hm },
    interval_cases m,
    { exact le_of_lt (lt_of_not_ge (λ h, not_lt_of_le h (lt_of_le_of_lt (le_of_eq A) (lt_of_not_ge h)))) },
    { exact le_refl 7 },
    { exact le_of_not_lt (λ h, (lt_irrefl _ h)) sorry } },
end

end smallest_integer_satisfying_inequality_l511_511444


namespace matrix_multiplication_is_zero_l511_511948

variable (k d e f c a : ℝ)

def A : Matrix (Fin 3) (Fin 3) ℝ :=
!![ [0, k * d, -k * c],
    [-k * d, 0, k * a],
    [k * c, -k * a, 0] ]

def B : Matrix (Fin 3) (Fin 3) ℝ :=
!![ [d^2, d * e, d * f],
    [d * e, e^2, e * f],
    [d * f, e * f, f^2] ]

theorem matrix_multiplication_is_zero : A k d e f c a * B d e f = 0 := by
  sorry

end matrix_multiplication_is_zero_l511_511948


namespace minimize_sum_of_f_seq_l511_511112

def f (x : ℝ) : ℝ := x^2 - 8 * x + 10

def isArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem minimize_sum_of_f_seq
  (a : ℕ → ℝ)
  (h₀ : isArithmeticSequence a 1)
  (h₁ : a 1 = a₁)
  : f (a 1) + f (a 2) + f (a 3) = 3 * a₁^2 - 18 * a₁ + 30 →

  (∀ x, 3 * x^2 - 18 * x + 30 ≥ 3 * 3^2 - 18 * 3 + 30) →
  a₁ = 3 :=
by
  sorry

end minimize_sum_of_f_seq_l511_511112


namespace solution_set_of_inequality_l511_511652

theorem solution_set_of_inequality (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_continuous : continuous f)
  (h_deriv_lt_zero : ∀ x, 1 < x → deriv f x < 0) (h_deriv_gt_zero : ∀ x, 0 < x ∧ x < 1 → deriv f x > 0)
  (h_f2_zero : f 2 = 0) :
  { x : ℝ | (x + 1) * f x > 0 } = set.union (set.Ioo (-2) (-1)) (set.Ioo 0 2) :=
sorry

end solution_set_of_inequality_l511_511652


namespace circumcenter_on_line_l511_511394

-- Define the quadrilateral ABCD and the center O of the circumscribed circle.
variables {A B C D O : Type} [metric_space O] [euclidean_space O]

-- Define the lines AC and BD do not pass through O.
variables {AC BD : set O} [is_line AC] [is_line BD]
variables (h_ac : O ∉ AC) (h_bd : O ∉ BD)

-- Define the circumcenters of triangles AOC and BOD.
variables {circumcenter_AOC circumcenter_BOD : O}

-- Define the condition that the circumcenter of triangle AOC lies on line BD.
variable (h1 : circumcenter_AOC ∈ BD)

-- The theorem statement.
theorem circumcenter_on_line 
  (h2 : is_circumcenter A O C circumcenter_AOC)
  (h3 : is_circumcenter B O D circumcenter_BOD) :
  circumcenter_BOD ∈ AC := 
sorry

end circumcenter_on_line_l511_511394


namespace twin_functions_sum_is_72_l511_511708

-- Define the expression for the "twin functions".
def twin_function (x : ℤ) : ℤ := 2 * x ^ 2 - 1

-- Define the set of possible domains.
def domains : set (set ℤ) := {
  { -2, -1 }, { -2, 1 }, { 2, -1 }, { 2, 1 }, 
  { -2, -1, 1 }, { -2, -1, 2 }, { -1, 1, 2 }, 
  { -2, 1, 2 }, { -2, -1, 1, 2 }
}

-- Define the function value sum for a given domain.
def function_value_sum (domain : set ℤ) : ℤ :=
  domain.sum (λ x, twin_function x)

-- Define the total sum of function values for all twin functions.
def total_function_value_sum : ℤ :=
  domains.to_finset.sum function_value_sum

-- The theorem to prove:
theorem twin_functions_sum_is_72 : total_function_value_sum = 72 :=
by sorry

end twin_functions_sum_is_72_l511_511708


namespace max_time_for_taxiing_is_15_l511_511367

-- Declare the function representing the distance traveled by the plane with respect to time
def distance (t : ℝ) : ℝ := 60 * t - 2 * t ^ 2

-- The main theorem stating the maximum time s the plane uses for taxiing
theorem max_time_for_taxiing_is_15 : ∃ s, ∀ t, distance t ≤ distance s ∧ s = 15 :=
by
  sorry

end max_time_for_taxiing_is_15_l511_511367


namespace rationalize_sqrt_5_div_18_l511_511801

theorem rationalize_sqrt_5_div_18 :
  (Real.sqrt (5 / 18) = Real.sqrt 10 / 6) :=
sorry

end rationalize_sqrt_5_div_18_l511_511801


namespace trajectory_of_moving_circle_l511_511479

theorem trajectory_of_moving_circle :
  let O := (0, 0)
  let C := (3, 0)
  let R1 := 1
  let R2 := 1
  ∀ (M : ℝ×ℝ), 
    (dist M O = dist M C + 2) ∧
    (dist O C = 3) →
    ∃ a b : ℝ, is_hyperbola_branch_with_foci O C M :=
sorry

end trajectory_of_moving_circle_l511_511479


namespace correct_statement_four_l511_511932

variable {α : Type*} (A B S : Set α) (U : Set α)

theorem correct_statement_four (h1 : U = Set.univ) (h2 : A ∩ B = U) : A = U ∧ B = U := by
  sorry

end correct_statement_four_l511_511932


namespace total_interest_is_350_l511_511478

-- Define the principal amounts, rates, and time
def principal1 : ℝ := 1000
def rate1 : ℝ := 0.03
def principal2 : ℝ := 1200
def rate2 : ℝ := 0.05
def time : ℝ := 3.888888888888889

-- Calculate the interest for one year for each loan
def interest_per_year1 : ℝ := principal1 * rate1
def interest_per_year2 : ℝ := principal2 * rate2

-- Calculate the total interest for the time period for each loan
def total_interest1 : ℝ := interest_per_year1 * time
def total_interest2 : ℝ := interest_per_year2 * time

-- Finally, calculate the total interest amount
def total_interest_amount : ℝ := total_interest1 + total_interest2

-- The proof problem: Prove that total_interest_amount == 350 Rs
theorem total_interest_is_350 : total_interest_amount = 350 := by
  sorry

end total_interest_is_350_l511_511478


namespace curve_is_line_l511_511543

theorem curve_is_line (r θ : ℝ) (h : r = 1 / (2 * real.sin θ - real.cos θ)) :
  ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ ∀ x y : ℝ, (∃ (r θ : ℝ), x = r * real.cos θ ∧ y = r * real.sin θ ∧ r = 1 / (2 * real.sin θ - real.cos θ)) → a * x + b * y + c = 0 :=
sorry

end curve_is_line_l511_511543


namespace scarlett_initial_oil_amount_l511_511404

theorem scarlett_initial_oil_amount (x : ℝ) (h : x + 0.67 = 0.84) : x = 0.17 :=
by sorry

end scarlett_initial_oil_amount_l511_511404


namespace molar_mass_of_substance_l511_511260

theorem molar_mass_of_substance (
    (W : ℝ) (n : ℝ) : W = 264 ∧ n = 3 → M = 88)
    (W : ℝ) (n : ℝ) (M : ℝ) : W / n = M) : W / 3 = 88 := by
  sorry

end molar_mass_of_substance_l511_511260


namespace range_f_l511_511980

noncomputable def f (x : ℝ) : ℝ :=
  1 / (λ g : ℝ → ℝ, g (64 * g (g (Real.log x)) / 1025))
  (λ x : ℝ, x ^ 5 + 1 / x ^ 5)

theorem range_f :
  (set.range (λ x : ℝ, f x)) = (set.Icc (-32 / 1025) 0 ∪ set.Ioc 0 (32 / 1025)) :=
sorry

end range_f_l511_511980


namespace sandy_siding_cost_l511_511811

theorem sandy_siding_cost : 
  let width_wall := 10
  let height_wall := 8
  let base_triangle := 10
  let height_triangle := 6
  let section_area := 100
  let section_cost := 30
  let area_wall := width_wall * height_wall
  let area_triangle := (base_triangle * height_triangle) / 2
  let total_area := area_wall + 2 * area_triangle
  let sections_needed := Int.ceil (total_area / section_area)
  let total_cost := sections_needed * section_cost
  total_cost = 60 :=
by
  sorry

end sandy_siding_cost_l511_511811


namespace posters_total_l511_511772

theorem posters_total (Mario_Samantha_diff : ℕ) (Mario_posters : ℕ)
    (Samantha_posters : ℕ) (Jonathan_posters : ℕ) (total_posters : ℕ) :
  Mario_posters = 36 →
  Mario_Samantha_diff = 45 →
  Samantha_posters = Mario_posters + Mario_Samantha_diff →
  Jonathan_posters = 2 * Samantha_posters →
  total_posters = Mario_posters + Samantha_posters + Jonathan_posters →
  total_posters = 279 :=
by
  intro hMario hDiff hSamantha hJonathan hTotal
  rw [hMario, hDiff] at hSamantha
  rw [←hSamantha, ←hSamantha, hMario, hDiff] at hJonathan
  rw [←hMario, hTotal]
  sorry

end posters_total_l511_511772


namespace odd_three_digit_numbers_count_l511_511639

theorem odd_three_digit_numbers_count :
  let set_1 := {1, 3, 5}
  let set_2 := {2, 4}
  ∃ (s1 : Finset ℤ) (s2: Finset ℤ), 
    s1 = set_1 ∧ s2 = set_2 ∧ 
    ∀ C_2_3 A_1_2 C_1_2 A_2_2, 
      C_2_3 = (Finset.card set_1.choose 2)
      ∧ A_1_2 = 2
      ∧ C_1_2 = (Finset.card set_2.choose 1)
      ∧ A_2_2 = 2
      → (C_2_3 * C_1_2 * A_1_2 * A_2_2 = 24) :=
by
  sorry

end odd_three_digit_numbers_count_l511_511639


namespace sum_differences_1_to_999_l511_511593

-- Define a utility function to compute the first digit of a number
def first_digit (n : ℕ) : ℕ :=
if n < 10 then n else first_digit (n / 10)

-- Define a utility function to compute the last digit of a number
def last_digit (n : ℕ) : ℕ :=
n % 10

-- Define the operation performed by Damir
def difference (n : ℕ) : ℤ :=
(first_digit n : ℤ) - (last_digit n : ℤ)

-- Define the sum of all differences from 1 to 999
def sum_differences : ℤ :=
(1).to (999).sum (λ n, difference n)

-- State the main theorem to be proved using the previous definitions
theorem sum_differences_1_to_999 : sum_differences = 495 :=
sorry

end sum_differences_1_to_999_l511_511593


namespace graphs_intersect_at_y_axis_l511_511345

noncomputable def intersect_at_y_axis (a b c : ℝ) (h_b : b ≠ 0) (h_c : c ≠ 0) : Prop :=
  ∃ x y, (y = a * x^2 + b^2 * x^3 + c) ∧ 
         (y = a * x^2 - b^2 * x^3 + c) ∧ 
         x = 0 ∧ 
         y = c

theorem graphs_intersect_at_y_axis (a b c : ℝ) (h_b : b ≠ 0) (h_c : c ≠ 0) :
  intersect_at_y_axis a b c h_b h_c :=
begin 
  sorry
end

end graphs_intersect_at_y_axis_l511_511345


namespace percentage_of_profit_if_no_discount_l511_511028

-- Conditions
def discount : ℝ := 0.05
def profit_w_discount : ℝ := 0.216
def cost_price : ℝ := 100
def expected_profit : ℝ := 28

-- Proof statement
theorem percentage_of_profit_if_no_discount :
  ∃ (marked_price selling_price_no_discount : ℝ),
    selling_price_no_discount = marked_price ∧
    (marked_price - cost_price) / cost_price * 100 = expected_profit :=
by
  -- Definitions and logic will go here
  sorry

end percentage_of_profit_if_no_discount_l511_511028


namespace divisors_of_power_l511_511696

theorem divisors_of_power {
  let n := 1806,
  let prime_factors := [(2, 1), (3, 2), (101, 1)],
  let n_pow := 1806,
  ∀ (n : ℕ) (p : list (ℕ × ℕ)) (k : ℕ),
  k = p.foldr (λ x acc, (x.snd + 1) * acc) 1,
  let m := 1806,
  let required_divisors := 54
} :
  (number_of_divisors (m^m) divisible_by n) = required_divisors := by
{
  sorry
}

end divisors_of_power_l511_511696


namespace curve_is_line_l511_511546

theorem curve_is_line (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b c : ℝ), a * (r * Real.cos θ) + b * (r * Real.sin θ) + c = 0 ∧
  (a, b, c) = (-1, 2, -1) := sorry

end curve_is_line_l511_511546


namespace relatively_prime_probability_l511_511427

open Finset

theorem relatively_prime_probability :
  let s := ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)
  in let pairs := s.val.powerset.filter (λ t, t.card = 2)
  in (pairs.count (λ t, (t : Multiset ℕ).gcd = 1)).toRational / pairs.card.toRational = 3 / 4 := 
by
  -- Prove that the probability is 3/4
  sorry

end relatively_prime_probability_l511_511427


namespace count_5_letter_words_with_at_least_one_vowel_l511_511207

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := (letters.card) ^ 5

def non_vowel_letters : Finset Char := letters \ vowels

def total_non_vowel_5_letter_words : ℕ := (non_vowel_letters.card) ^ 5

theorem count_5_letter_words_with_at_least_one_vowel :
  total_5_letter_words - total_non_vowel_5_letter_words = 6752 :=
by
  sorry

end count_5_letter_words_with_at_least_one_vowel_l511_511207


namespace find_digits_l511_511808

theorem find_digits : ∃ AB CD : ℕ, (AB * CD = 121) ∧ (10 ≤ AB ∧ AB < 100) ∧ (10 ≤ CD ∧ CD < 100) :=
by {
  use [11, 11],
  split; try { split },
  repeat { sorry }
}

end find_digits_l511_511808


namespace number_of_intersections_l511_511963

theorem number_of_intersections : 
  (∃ p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ 9 * p.1^2 + p.2^2 = 1) 
  ∧ (∃! p₁ p₂ : ℝ × ℝ, p₁ ≠ p₂ ∧ p₁.1^2 + 9 * p₁.2^2 = 9 ∧ 9 * p₁.1^2 + p₁.2^2 = 1 ∧
    p₂.1^2 + 9 * p₂.2^2 = 9 ∧ 9 * p₂.1^2 + p₂.2^2 = 1) :=
by
  -- The proof will be here
  sorry

end number_of_intersections_l511_511963


namespace sin_cos_theta_l511_511641

theorem sin_cos_theta (θ : ℝ) (h : (sin θ + cos θ) / (sin θ - cos θ) = 2) : sin θ * cos θ = 3 / 10 :=
by sorry

end sin_cos_theta_l511_511641


namespace Tod_drove_time_l511_511410

section
variable (distance_north: ℕ) (distance_west: ℕ) (speed: ℕ)

theorem Tod_drove_time :
  distance_north = 55 → distance_west = 95 → speed = 25 → 
  (distance_north + distance_west) / speed = 6 :=
by
  intros
  sorry
end

end Tod_drove_time_l511_511410


namespace vertex_of_parabola_l511_511376

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x + 9)^2 - 3

-- State the theorem to prove
theorem vertex_of_parabola : ∃ h k : ℝ, (h = -9 ∧ k = -3) ∧ (parabola h = k) :=
by sorry

end vertex_of_parabola_l511_511376


namespace field_area_l511_511490

-- Define a rectangular field
structure RectangularField where
  length : ℕ
  width : ℕ
  fencing : ℕ := 2 * width + length
  
-- Given conditions
def field_conditions (L W F : ℕ) : Prop :=
  L = 30 ∧ 2 * W + L = F

-- Theorem stating the required proof
theorem field_area : ∀ (L W F : ℕ), field_conditions L W F → F = 84 → (L * W) = 810 :=
by
  intros L W F h1 h2
  sorry

end field_area_l511_511490


namespace letter_in_repetitive_sequence_l511_511870

def sequence : String := "ABCDEFGHIJKLMNOPQRSTUVWXYZABC"

def sequence_length : Nat := 29

def position_mod (n : Nat) (m : Nat) : Nat := n % m

def nth_letter (s : String) (n : Nat) : Char := 
  s.get (n % s.length)

theorem letter_in_repetitive_sequence (n : Nat) (s : String) (length_s : Nat) (k : Nat) 
  (h1 : s.length = length_s) (h2 : n = k) (h3 : k % length_s = 4) : 
  nth_letter s n = 'E' :=
sorry

end letter_in_repetitive_sequence_l511_511870


namespace eccentricity_bound_l511_511307

variables {a b c e : ℝ}

-- Definitions of the problem conditions
def hyperbola (x y : ℝ) (a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def line (x : ℝ) := 2 * x
def eccentricity (c a : ℝ) := c / a

-- Proof statement in Lean
theorem eccentricity_bound (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (c : ℝ)
  (h₃ : hyperbola x y a b)
  (h₄ : ∀ x, line x ≠ y) :
  1 < eccentricity c a ∧ eccentricity c a ≤ sqrt 5 :=
sorry

end eccentricity_bound_l511_511307


namespace log_range_l511_511262

noncomputable def range_of_a (a : ℝ) : Set ℝ :=
  { x | 0 < x ∧ x < 3/4 } ∪ { x | 1 < x }

theorem log_range (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : Real.logBase a (3/4) < 1) :
  a ∈ range_of_a :=
  sorry

end log_range_l511_511262


namespace distance_is_fifteen_l511_511547

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_is_fifteen : distance 0 12 9 0 = 15 := 
by
  sorry

end distance_is_fifteen_l511_511547


namespace difference_of_integers_subtracts_to_zero_l511_511069

theorem difference_of_integers_subtracts_to_zero :
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 2 3) 4) (Nat.lcm 5 6)) (Nat.lcm (Nat.lcm 7 (Nat.lcm 8 9)) (Nat.lcm 10 (Nat.lcm 11 (Nat.lcm 12 13)))) in
  (lcm_val - 2^3 * 3^2 * 5 * 7 * 11 * 13) = 0 :=
by
  sorry

end difference_of_integers_subtracts_to_zero_l511_511069


namespace find_profits_maximize_profit_week3_l511_511371

-- Defining the conditions of the problems
def week1_sales_A := 10
def week1_sales_B := 12
def week1_profit := 2000

def week2_sales_A := 20
def week2_sales_B := 15
def week2_profit := 3100

def total_sales_week3 := 25

-- Condition: Sales of type B exceed sales of type A but do not exceed twice the sales of type A
def sales_condition (x : ℕ) := (total_sales_week3 - x) > x ∧ (total_sales_week3 - x) ≤ 2 * x

-- Define the profits for types A and B
def profit_A (a b : ℕ) := week1_sales_A * a + week1_sales_B * b = week1_profit
def profit_B (a b : ℕ) := week2_sales_A * a + week2_sales_B * b = week2_profit

-- Define the profit function for week 3
def profit_week3 (a b x : ℕ) := a * x + b * (total_sales_week3 - x)

theorem find_profits : ∃ a b, profit_A a b ∧ profit_B a b :=
by
  use 80, 100
  sorry

theorem maximize_profit_week3 : 
  ∃ x y, 
  sales_condition x ∧ 
  x + y = total_sales_week3 ∧ 
  profit_week3 80 100 x = 2320 :=
by
  use 9, 16
  sorry

end find_profits_maximize_profit_week3_l511_511371


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511230

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let total_words := 6^5
  let words_without_vowels := 4^5
  total_words - words_without_vowels = 6752 :=
by
  let total_words := 6^5
  let words_without_vowels := 4^5
  have h_total_words : total_words = 7776 := by norm_num
  have h_words_without_vowels : words_without_vowels = 1024 := by norm_num
  calc
    7776 - 1024 = 6752 : by norm_num

end number_of_5_letter_words_with_at_least_one_vowel_l511_511230


namespace eccentricity_bound_l511_511310

variables {a b c e : ℝ}

-- Definitions of the problem conditions
def hyperbola (x y : ℝ) (a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def line (x : ℝ) := 2 * x
def eccentricity (c a : ℝ) := c / a

-- Proof statement in Lean
theorem eccentricity_bound (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (c : ℝ)
  (h₃ : hyperbola x y a b)
  (h₄ : ∀ x, line x ≠ y) :
  1 < eccentricity c a ∧ eccentricity c a ≤ sqrt 5 :=
sorry

end eccentricity_bound_l511_511310


namespace remainder_x4_plus_4_div_by_x_minus_3_squared_l511_511563

theorem remainder_x4_plus_4_div_by_x_minus_3_squared :
  ∀ x : ℝ, ∃ q r : ℝ[x], r.degree < 2 ∧ (X^4 + 4 : ℝ[X]) = (X - 3)^2 * q + r ∧ r = (31 * X - 56) :=
by
  sorry

end remainder_x4_plus_4_div_by_x_minus_3_squared_l511_511563


namespace problem1_problem2_l511_511524

theorem problem1 : (Real.pi ^ 0) - ((1 / 2) ^ (-2)) + (3 ^ 2) = 6 := 
  sorry

theorem problem2 (x : ℝ) : (2 * x^2)^2 - x * x^3 - (x^5 / x) = 2 * x^4 := 
  sorry

end problem1_problem2_l511_511524


namespace five_letter_words_with_at_least_one_vowel_l511_511215

open Finset

theorem five_letter_words_with_at_least_one_vowel :
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := ((letters \ vowels).card : ℕ) ^ 5
  total_words - no_vowel_words = 6752 :=
by
  let letters := insert 'A' (insert 'B' (insert 'C' (insert 'D' (insert 'E' (singleton 'F')))))
  let vowels := insert 'A' (singleton 'E')
  let consonants := letters \ vowels
  have : letters.card = 6 := by simp
  have : vowels.card = 2 := by simp
  have : consonants.card = 4 := by simp
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := (consonants.card : ℕ) ^ 5
  calc
    total_words - no_vowel_words
        = 6^5 - 4^5         : by simp
    ... = 7776 - 1024       : by norm_num
    ... = 6752             : by norm_num

end five_letter_words_with_at_least_one_vowel_l511_511215


namespace lollipop_problem_l511_511170

def Henry_lollipops (A : Nat) : Nat := A + 30
def Diane_lollipops (A : Nat) : Nat := 2 * A
def Total_days (H A D : Nat) (daily_rate : Nat) : Nat := (H + A + D) / daily_rate

theorem lollipop_problem
  (A : Nat) (H : Nat) (D : Nat) (daily_rate : Nat)
  (h₁ : A = 60)
  (h₂ : H = Henry_lollipops A)
  (h₃ : D = Diane_lollipops A)
  (h₄ : daily_rate = 45)
  : Total_days H A D daily_rate = 6 := by
  sorry

end lollipop_problem_l511_511170


namespace monochromatic_bound_l511_511527

open Set Finset

noncomputable def A (points : Finset (ℝ × ℝ)) : ℝ := 
  1 / 2 * (∑ t in points.triangleSets, abs (det (Matrix (Vector.map prod.snd t) : Matrix (Fin n.succ) (Fin n.succ) ℝ) 
                                            * 1 / 2))

noncomputable def A1 (points : Finset (ℝ × ℝ)) (coloring : ℝ × ℝ → ℕ) : ℝ := 
  1 / 2 * (∑ t in points.triangleSets, 
             if ∀ v in t, coloring v = coloring (Finset.choose id t) 
             then abs (det (Matrix (Vector.map prod.snd t) : Matrix (Fin n.succ) (Fin n.succ) ℝ) 
                      * 1 / 2) 
             else 0)

theorem monochromatic_bound (points : Finset (ℝ × ℝ)) (coloring : ℝ × ℝ → ℕ)
  (h_no_three_collinear : ∀ {a b c}, a ≠ b → b ≠ c → c ≠ a → a ∈ points → b ∈ points → c ∈ points → orientation a b c ≠ 0)
  (h_colors : (point : ℝ × ℝ) ∈ points → coloring point = 0 ∨ coloring point = 1 ∨ coloring point = 2)
  (h_balance : ∀ c, points.filter (λ v, coloring v = c)).card = 6) :
  A1 points coloring ≤ A points / 4 := 
by { sorry }

end monochromatic_bound_l511_511527


namespace noah_yearly_bill_l511_511780

-- Define the length of each call in minutes
def call_duration : ℕ := 30

-- Define the cost per minute in dollars
def cost_per_minute : ℝ := 0.05

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Define the cost per call in dollars
def cost_per_call : ℝ := call_duration * cost_per_minute

-- Define the total cost for a year in dollars
def yearly_cost : ℝ := cost_per_call * weeks_in_year

-- State the theorem
theorem noah_yearly_bill : yearly_cost = 78 := by
  -- Proof follows here
  sorry

end noah_yearly_bill_l511_511780


namespace hyperbola_eccentricity_l511_511313

theorem hyperbola_eccentricity 
  (a b e : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : e = Real.sqrt (1 + (b^2 / a^2))) 
  (h4 : e ≤ Real.sqrt 5) : 
  e = 2 := 
sorry

end hyperbola_eccentricity_l511_511313


namespace eccentricity_of_hyperbola_l511_511137

variables {a b c e : ℝ}
variables {m n x y : ℝ}

-- Conditions
def on_hyperbola (x y a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def symmetric_about_origin (A B : ℝ × ℝ) := A.1 = -B.1 ∧ A.2 = -B.2
def slope_product (P A B : ℝ × ℝ) := (P.2 - A.2) / (P.1 - A.1) * (P.2 - B.2) / (P.1 - B.1)

variables (A B P A₀ : ℝ × ℝ)
axiom hyperbola_eq : on_hyperbola A.1 A.2 a b
axiom hyperbola_eq' : on_hyperbola P.1 P.2 a b
axiom symm_about_origin : symmetric_about_origin A B
axiom distinct_points : A ≠ P ∧ B ≠ P
axiom slope_eq : slope_product P A B = (5 * c - 4 * a) / (2 * a)

-- Theorem
theorem eccentricity_of_hyperbola : e = 2 :=
sorry

end eccentricity_of_hyperbola_l511_511137


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511232

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let total_words := 6^5
  let words_without_vowels := 4^5
  total_words - words_without_vowels = 6752 :=
by
  let total_words := 6^5
  let words_without_vowels := 4^5
  have h_total_words : total_words = 7776 := by norm_num
  have h_words_without_vowels : words_without_vowels = 1024 := by norm_num
  calc
    7776 - 1024 = 6752 : by norm_num

end number_of_5_letter_words_with_at_least_one_vowel_l511_511232


namespace fraction_simplify_l511_511996

variable (a b : ℝ) (ha : 0 < a) (hb : 0 < b) -- a and b are positive real numbers

theorem fraction_simplify : (real.sqrt (a^3 * b)) / (real.cbrt (a * b)) = a^(7/6) * b^(1/6) :=
sorry

end fraction_simplify_l511_511996


namespace prove_viete_formula_l511_511518

noncomputable theory

def euler_viete_formula : Prop :=
  let phi : ℝ := Real.pi / 2
  let r : ℝ := 1
  (Real.pi = 2 * ∏ n in (Set.Icc 1 (Real.toAnalytic 1)).toFinset, Real.cos (phi / 2^n))

theorem prove_viete_formula : euler_viete_formula :=
  sorry

end prove_viete_formula_l511_511518


namespace Bronquinha_drinks_juice_at_l511_511516

noncomputable def mowing_rate_without_juice := 3
noncomputable def mowing_rate_with_juice := 2
noncomputable def start_time := 10 -- 10 AM represented in hours
noncomputable def end_time := 12.5 -- 12:30 PM in hours

theorem Bronquinha_drinks_juice_at :
  ∃ t : ℝ, start_time + t = 11.5 ∧ 
  (t / mowing_rate_without_juice + (end_time - (start_time + t)) / mowing_rate_with_juice = 1) :=
sorry

end Bronquinha_drinks_juice_at_l511_511516


namespace min_value_abc_l511_511764

open Real

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/10368 :=
sorry

end min_value_abc_l511_511764


namespace bag_cost_is_10_l511_511858

def timothy_initial_money : ℝ := 50
def tshirt_cost : ℝ := 8
def keychain_cost : ℝ := 2
def keychains_per_set : ℝ := 3
def number_of_tshirts : ℝ := 2
def number_of_bags : ℝ := 2
def number_of_keychains : ℝ := 21

noncomputable def cost_of_each_bag : ℝ :=
  let cost_of_tshirts := number_of_tshirts * tshirt_cost
  let remaining_money_after_tshirts := timothy_initial_money - cost_of_tshirts
  let cost_of_keychains := (number_of_keychains / keychains_per_set) * keychain_cost
  let remaining_money_after_keychains := remaining_money_after_tshirts - cost_of_keychains
  remaining_money_after_keychains / number_of_bags

theorem bag_cost_is_10 :
  cost_of_each_bag = 10 := by
  sorry

end bag_cost_is_10_l511_511858


namespace kernel_bag_theorem_l511_511794

noncomputable def kernel_problem_statement : Prop :=
  let k1 := 60
  let k2 := 42
  let k3 := 82
  let t2 := 50
  let t3 := 100
  let total_popped := k1 + k2 + k3
  let average_popped_percentage := 82
  let total_kernels := t2 + t3

  ∃ t1 : ℝ,  (184 = 0.82 * (t1 + total_kernels)) ∧ t1 = 74

theorem kernel_bag_theorem : kernel_problem_statement := 
begin
  sorry
end

end kernel_bag_theorem_l511_511794


namespace lambda_over_m_range_l511_511992

-- Given conditions
variables {λ m α : ℝ}
def a := (λ + 2, λ^2 - cos α * cos α)
def b := (m, m / 2 + sin α)

-- Required proof problem statement
theorem lambda_over_m_range (h : a = (2 * b.1, 2 * b.2)) : 
  -6 ≤ λ / m ∧ λ / m ≤ 1 := 
sorry

end lambda_over_m_range_l511_511992


namespace fenced_area_l511_511012

theorem fenced_area (L W : ℝ) (square_side triangle_leg : ℝ) :
  L = 20 ∧ W = 18 ∧ square_side = 4 ∧ triangle_leg = 3 →
  (L * W - square_side^2 - (1 / 2) * triangle_leg^2 = 339.5) := by
  intros h
  rcases h with ⟨hL, hW, hs, ht⟩
  rw [hL, hW, hs, ht]
  simp
  sorry

end fenced_area_l511_511012


namespace parabola_vertex_l511_511374

theorem parabola_vertex :
  ∀ (x : ℝ), y = 2 * (x + 9)^2 - 3 → 
  (∃ h k, h = -9 ∧ k = -3 ∧ y = 2 * (x - h)^2 + k) :=
by
  sorry

end parabola_vertex_l511_511374


namespace quadratic_rewrite_l511_511530

theorem quadratic_rewrite  (a b c x : ℤ) (h : 25 * x^2 + 30 * x - 35 = 0) (hp : 25 * x^2 + 30 * x + 9 = (5 * x + 3) ^ 2)
(hc : c = 44) : a = 5 → b = 3 → a + b + c = 52 := 
by
  intro ha hb
  sorry

end quadratic_rewrite_l511_511530


namespace value_of_q_at_one_l511_511943

def q(x : ℝ) : ℝ := sorry  -- assume the function q is graphically defined

theorem value_of_q_at_one : q(1) = 3 :=
sorry

end value_of_q_at_one_l511_511943


namespace noah_yearly_bill_l511_511781

-- Define the length of each call in minutes
def call_duration : ℕ := 30

-- Define the cost per minute in dollars
def cost_per_minute : ℝ := 0.05

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Define the cost per call in dollars
def cost_per_call : ℝ := call_duration * cost_per_minute

-- Define the total cost for a year in dollars
def yearly_cost : ℝ := cost_per_call * weeks_in_year

-- State the theorem
theorem noah_yearly_bill : yearly_cost = 78 := by
  -- Proof follows here
  sorry

end noah_yearly_bill_l511_511781


namespace minimize_f_sum_l511_511109

noncomputable def f (x : ℝ) : ℝ := x^2 - 8*x + 10

theorem minimize_f_sum :
  ∃ a₁ : ℝ, (∀ a₂ a₃ : ℝ, a₂ = a₁ + 1 ∧ a₃ = a₁ + 2 →
    f(a₁) + f(a₂) + f(a₃) = 3 * a₁^2 - 18 * a₁ + 30) →
    (∀ b₁ : ℝ, (∀ b₂ b₃ : ℝ, b₂ = b₁ + 1 ∧ b₃ = b₁ + 2 →
      f(b₁) + f(b₂) + f(b₃) ≥ f(a₁) + f(a₂) + f(a₃)) ∧ a₁ = 3) :=
by 
  sorry

end minimize_f_sum_l511_511109


namespace sum_of_differences_l511_511578

/-- 
  For each natural number from 1 to 999, Damir subtracts the last digit from the first digit and 
  writes the resulting differences on a board. We are to prove that the sum of all these differences 
  is 495.
-/
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, (first_digit n - last_digit n)) = 495 :=
sorry

/-- 
  Helper function to get the first digit of a natural number.
  Here, n > 0
-/
def first_digit (n : ℕ) : ℕ :=
  n / 10^(n.digits 10 - 1)

/-- 
  Helper function to get the last digit of a natural number.
  Here, n > 0
-/
def last_digit (n : ℕ) : ℕ :=
  n % 10

end sum_of_differences_l511_511578


namespace option_A_true_option_B_true_option_C_true_l511_511990

section
variables {α β γ λ : ℝ}
variables {A B M : ℝ → ℝ → Prop}

-- Define the points on the unit circle
def on_unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Conditions on the angles and vectors
def pointA := A (cos α) (sin α)
def pointB := B (cos β) (sin β)
def pointM := M (cos γ) (sin γ)
def angles_conditions := 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π ∧ 0 < γ ∧ γ < 2 * π
def vector_condition := ∀ x y, M x y = λ * (A x y + B x y)

-- Theorem statements based on the problem
theorem option_A_true (h : vector_condition) (cond : angles_conditions) (hλ : λ = √2 / 2) : 
  β = α + π / 2 := sorry

theorem option_B_true (h : vector_condition) (cond : angles_conditions) (hλ : λ = 1) : 
  β = α + 2 * π / 3 := sorry

theorem option_C_true (h : vector_condition) (cond : angles_conditions) : 
  |sin γ| = |sin ((α + β) / 2)| := sorry
end

end option_A_true_option_B_true_option_C_true_l511_511990


namespace five_letter_words_with_vowel_l511_511195

theorem five_letter_words_with_vowel : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let n := 5 
  (∃ (w : list(char)), w.length = n ∧ ∀ (i : fin n), w[i] ∈ letters ∧ (∃ (j : fin n), w[j] ∈ vowels)) → 
  (6^5 - 4^5 = 6752) := 
by
  sorry

end five_letter_words_with_vowel_l511_511195


namespace hexagon_coloring_count_l511_511529

-- Definitions of the conditions

/--
We have an arrangement of two columns of hexagons, where each column contains four hexagons.
-/
structure HexagonSetup where
  column1 : List ℕ -- List of colors for column 1
  column2 : List ℕ -- List of colors for column 2
  h_c1_len : column1.length = 4
  h_c2_len : column2.length = 4

/--
The bottom hexagon in the left column is colored blue.
-/
def bottom_left_blue (layout : HexagonSetup) : Prop :=
  layout.column1.head = 4 -- Assuming the color code 4 represents blue

/--
Each hexagon is colored either red, yellow, green, or blue.
-/
def valid_colors (color : ℕ) : Prop :=
  color ∈ [1, 2, 3, 4] -- Assuming color codes 1, 2, 3, 4 represent red, yellow, green, blue respectively

/--
No two hexagons with a common side are colored the same.
-/
def no_common_side_same_color (layout : HexagonSetup) : Prop :=
  ∀ (i j : ℕ), i < 4 → j < 4 → (layout.column1[i] ≠ layout.column1[j] ∨ (i ≠ j + 1 ∧ j ≠ i + 1))
    ∧ (layout.column1[i] ≠ layout.column2[j] ∨ (i ≠ j))
    ∧ (layout.column2[i] ≠ layout.column2[j] ∨ (i ≠ j + 1 ∧ j ≠ i + 1))

-- Statement of the theorem
theorem hexagon_coloring_count :
  ∃ layout : HexagonSetup,
    bottom_left_blue layout ∧
    (∀ color, (∀ i, i < 4 → valid_colors (layout.column1[i])) ∧ (∀ j, j < 4 → valid_colors (layout.column2[j]))) ∧
    no_common_side_same_color layout ∧
    finset.card (finset.filter (λ layout, bottom_left_blue layout ∧ no_common_side_same_color layout)
                  {layout // (∀ color, (∀ i, i < 4 → valid_colors (layout.column1[i]))
                                        ∧ (∀ j, j < 4 → valid_colors (layout.column2[j])))}) = 972 :=
sorry

end hexagon_coloring_count_l511_511529


namespace sum_q_l511_511378

def q : ℝ → ℝ := sorry  -- Assume q is some cubic polynomial

/- Given conditions as hypotheses -/
axiom h1 : q 5 = 2
axiom h2 : q 10 = 22
axiom h3 : q 15 = 12
axiom h4 : q 25 = 32

/- The statement to prove -/
theorem sum_q : (∑ n in (finset.range 23).map (λ n, n + 4), q n) = 391 :=
sorry

end sum_q_l511_511378


namespace remainder_of_sum_mod_l511_511269

theorem remainder_of_sum_mod (n : ℤ) : ((7 + n) + (n + 5)) % 7 = (5 + 2 * n) % 7 :=
by
  sorry

end remainder_of_sum_mod_l511_511269


namespace initial_provision_last_days_l511_511013

-- Define the conditions
def initial_provision (days : ℕ) := 150 * days
def remaining_provision_after_10_days (initial_days : ℕ) := initial_provision(initial_days) - 150 * 10
def food_consumption_125_men_42_days := 125 * 42

-- The statement to be proved
theorem initial_provision_last_days (x : ℕ) (h : remaining_provision_after_10_days x = food_consumption_125_men_42_days) : x = 45 :=
by {
  sorry
}

end initial_provision_last_days_l511_511013


namespace find_k_l511_511864

def point (x y : ℝ) := (x, y)
def line (m k : ℝ) := λ x : ℝ, m * x + k

-- Coordinates of points A, B, C
def A := point 0 10
def B := point 4 0
def C := point 10 0

-- Equations of the lines AB and AC
def lineAB := line (-2.5) 10
def lineAC := line (-1) 10

-- Intersection points T and U of the line y = 2x + k with AB and AC respectively
def T (k : ℝ) : ℝ × ℝ := 
  let x := (10 - k) / 4.5 in 
  (x, 2 * x + k)

def U (k : ℝ) : ℝ × ℝ := 
  let x := (10 - k) / 3 in 
  (x, 2 * x + k)

-- Function to calculate the area of triangle ATU
def area_ATU (k : ℝ) :=
  let T := T k in
  let U := U k in
  let TU := (fst U - fst T) * (snd T - 10) in
  (1 / 2) * abs TU

-- Proof that the area is 20 implies k = 6
theorem find_k : (∃ k : ℝ, area_ATU k = 20) → k = 6 := by
  sorry

end find_k_l511_511864


namespace find_t_find_k_l511_511157

noncomputable def f (x : ℝ) : ℝ := x^2 + 4*x + 2
noncomputable def f' (x : ℝ) : ℝ := 2*x + 4
noncomputable def g (t x : ℝ) : ℝ := t * exp(x) * (f'(x) - 2)
noncomputable def g' (t x : ℝ) : ℝ := t * exp(x) * (2*x + 2)

theorem find_t (t : ℝ) : 
  let A := (-17/8 : ℝ) in
  let B := (0 : ℝ) in
  f' A * g' t B = -1 → t = 1 :=
by intros A B h; sorry

theorem find_k (k : ℝ) (t : ℝ) : 
  (∀ x, x ∈ Icc (2 : ℝ) (real.top) → k * g t x ≥ 2 * f x) → k ∈ Icc (2 : ℝ) (2 * exp(2)) :=
by intros h; sorry

end find_t_find_k_l511_511157


namespace employee_gross_pay_l511_511502

theorem employee_gross_pay
  (pay_rate_regular : ℝ) (pay_rate_overtime : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ)
  (h1 : pay_rate_regular = 11.25)
  (h2 : pay_rate_overtime = 16)
  (h3 : regular_hours = 40)
  (h4 : overtime_hours = 10.75) :
  (pay_rate_regular * regular_hours + pay_rate_overtime * overtime_hours = 622) :=
by
  sorry

end employee_gross_pay_l511_511502


namespace a4_value_l511_511650

-- Definition of the sequence and conditions
def seq (a : ℕ → ℤ) := ∀ n > 1, (a n + 2) / (a (n - 1) + 2) = 3

-- Given a1 = 1
def a1 : ℕ → ℤ := λ n, if n = 1 then 1 else sorry

-- Prove that a_4 = 79 given the conditions
theorem a4_value (a : ℕ → ℤ) (h1 : seq a) (h2 : a 1 = 1) : a 4 = 79 :=
sorry

end a4_value_l511_511650


namespace evaluate_expression_l511_511570

def g (m : ℕ) : ℝ := Real.log m^3 / Real.log 2003

theorem evaluate_expression : 
  g 7 + g 17 + g 29 = 3 * (Real.log (7 * 17 * 29) / Real.log 2003) := 
sorry

end evaluate_expression_l511_511570


namespace positive_integers_satisfy_condition_l511_511255
noncomputable theory

def satisfies_condition (n : ℕ) : Prop :=
  (n + 2000) % 120 = 0 ∧ (n + 2000) / 120 = Nat.floor (Real.cbrt n)

theorem positive_integers_satisfy_condition :
  {n : ℕ | satisfies_condition n}.finite.card = 6 :=
by
  sorry

end positive_integers_satisfy_condition_l511_511255


namespace sum_odd_gt_sum_even_l511_511451

def sum_odd (n : ℕ) : ℕ := ∑ i in range n, 2 * i + 1

def sum_even (n : ℕ) : ℕ := ∑ i in range n, 2 * (i + 1)

theorem sum_odd_gt_sum_even : sum_odd 1010 > sum_even 1009 := by
  sorry

end sum_odd_gt_sum_even_l511_511451


namespace min_value_of_expression_l511_511509

variable (x y : ℝ)
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)

-- Conditions
def A := ℝ × ℝ
def B := ℝ × ℝ
def C := ℝ × ℝ

def AM := x
def AN := y

def D := ((1 : ℝ)/2) * ((1 : ℝ) + (1 : ℝ))
def AD := 1/2 * ((1 : ℝ) + (1 : ℝ))

def AG := (3:ℝ)/5 * AD

-- The given equation from geometric conditions
def geom_eq : Prop := 1/x + 1/y = 10/3

-- Target to minimize
def target_eq : ℝ := 1/(x^2) + 4/(y^2)

theorem min_value_of_expression : 
  geom_eq →
  (∀ x y, target_eq ≥ 85/9) :=
by
  sorry

end min_value_of_expression_l511_511509


namespace find_k_l511_511833

-- Define the conditions
def equation : polynomial ℝ := polynomial.X^2 + 8 * polynomial.X + k
def ratio (r s : ℝ) := r = 3 * s

-- The main theorem
theorem find_k (k r s : ℝ) (h : polynomial.roots equation = {r, s}) (hratio: ratio r s) (hnonzero : r ≠ 0 ∧ s ≠ 0) : 
  k = 12 :=
sorry

end find_k_l511_511833


namespace percentage_of_x_l511_511706

variable {x y : ℝ}
variable {P : ℝ}

theorem percentage_of_x (h1 : (P / 100) * x = (20 / 100) * y) (h2 : x / y = 2) : P = 10 := by
  sorry

end percentage_of_x_l511_511706


namespace sum_of_differences_l511_511617

/-- Proving that the sum of all differences (first digit - last digit) for natural numbers from 
    1 to 999 is 495. -/ 
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, 
    let str := n.toString
    let first := if str.length > 1 then str.head!.toNat - '0'.toNat else 0
    let last := if str.length > 1 then str.getLast.toNat - '0'.toNat else 0
    first - last
  ) = 495 := 
by
  sorry

end sum_of_differences_l511_511617


namespace burn_time_of_rectangle_l511_511517

-- Define the conditions and the main proof statement.
theorem burn_time_of_rectangle 
  (toothpicks : ℕ) 
  (grid_dimension : ℕ × ℕ) 
  (initial_ignitions : ℕ) 
  (burn_time : ℕ) 
  (adjacent_spread : ℕ → ℕ → Prop)
  (fire_continues_to_spread : ∀ t₁ t₂, adjacent_spread t₁ t₂ → adjacent_spread t₂ t₁) :
  toothpicks = 38 →
  grid_dimension = (3, 5) →
  initial_ignitions = 2 →
  burn_time = 10 →
  (∃ t, ∀ p, p ∈ grid_grid_points grid_dimension → fire_reaches p t) →
  t = 65 :=
by sorry

end burn_time_of_rectangle_l511_511517


namespace five_letter_words_with_vowels_l511_511223

/-
How many 5-letter words with at least one vowel can be constructed from the letters 
A, B, C, D, E, and F? (Note that A and E are vowels, any word is valid, not just English language 
words, and letters may be used more than once.)
-/

theorem five_letter_words_with_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := ['A', 'E'],
      consonants := ['B', 'C', 'D', 'F'] in
  let total_words := 6 ^ 5,
      consonant_only_words := 4 ^ 5,
      at_least_one_vowel_words := total_words - consonant_only_words in
  at_least_one_vowel_words = 6752 :=
by
  intro letters vowels consonants total_words consonant_only_words at_least_one_vowel_words
  sorry

end five_letter_words_with_vowels_l511_511223


namespace quartic_poly_roots_l511_511089

noncomputable def quartic_poly : Polynomial ℝ :=
  Polynomial.of_finsupp (finsupp.single 4 1 + finsupp.single 3 (-14) + finsupp.single 2 80 + finsupp.single 1 (-320) + finsupp.single 0 200)

theorem quartic_poly_roots :
  (∀ x ∈ [4+2*complex.I, 4-2*complex.I, 3+1*complex.I, 3-1*complex.I], 
    Polynomial.eval x quartic_poly = 0) ∧
  quartic_poly.coeff 4 = 1 :=
by {
  sorry
}

end quartic_poly_roots_l511_511089


namespace value_of_f_at_1_l511_511671

def f (x : ℝ) : ℝ := x^2 + |x - 2|

theorem value_of_f_at_1 : f 1 = 2 :=
by sorry

end value_of_f_at_1_l511_511671


namespace problem_solution_l511_511746

open Set

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem problem_solution :
  A ∩ B = {1, 2, 3} ∧
  A ∩ C = {3, 4, 5, 6} ∧
  A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6} ∧
  A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8} :=
by
  sorry

end problem_solution_l511_511746


namespace banana_pudding_cost_l511_511053

theorem banana_pudding_cost (trays : ℕ) (cookies_per_tray : ℕ) (cookies_per_box : ℕ) (cost_per_box : ℝ) :
  trays = 3 → cookies_per_tray = 80 → cookies_per_box = 60 → cost_per_box = 3.50 →
  (trays * cookies_per_tray) / cookies_per_box * cost_per_box = 14 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    (3 * 80) / 60 * 3.50 = 240 / 60 * 3.50 : by norm_num
    ... = 4 * 3.50 : by rw div_eq_mul_inv; norm_num
    ... = 14 : by norm_num

end banana_pudding_cost_l511_511053


namespace square_construction_l511_511743

theorem square_construction {A B C D : ℝ×ℝ} 
  (h_convex : convex_quadrilateral A B C D) :
  ∃ (A' B' C' D' : ℝ×ℝ),
    is_square A' B' C' D' ∧ 
    A ≠ A' ∧ B ≠ B' ∧ C ≠ C' ∧ D ≠ D' ∧ 
    concurrent_points A A' B B' C C' D D' := 
by
  sorry


end square_construction_l511_511743


namespace minimum_bailing_rate_l511_511817

theorem minimum_bailing_rate (distance : ℝ) (leak_rate : ℝ) (max_capacity : ℝ) (rowing_speed : ℝ) : 
  distance = 2 ∧ leak_rate = 15 ∧ max_capacity = 50 ∧ rowing_speed = 3 → 
  ∀ r : ℝ, (r ≥ 13.75) → r == 14 :=
begin
  intros,
  sorry
end

end minimum_bailing_rate_l511_511817


namespace hyperbola_no_intersection_l511_511324

theorem hyperbola_no_intersection (a b e : ℝ)
  (ha : 0 < a) (hb : 0 < b)
  (h_e : e = (Real.sqrt (a^2 + b^2)) / a) :
  (√5 ≥ e ∧ 1 < e) → ∀ x y : ℝ, ¬ (y = 2 * x ∧ (x^2 / a^2 - y^2 / b^2 = 1)) :=
begin
  intros h_intersect x y,
  sorry,
end

end hyperbola_no_intersection_l511_511324


namespace trapezoid_fraction_l511_511955

theorem trapezoid_fraction 
  (shorter_base longer_base side_length : ℝ)
  (angle_adjacent : ℝ)
  (h1 : shorter_base = 120)
  (h2 : longer_base = 180)
  (h3 : side_length = 130)
  (h4 : angle_adjacent = 60) :
  ∃ fraction : ℝ, fraction = 1 / 2 :=
by
  sorry

end trapezoid_fraction_l511_511955


namespace k_range_l511_511379

theorem k_range (k : ℝ) : (∀ x y : ℝ, (x^2 / (15 - k) + y^2 / (k - 9) = 1) → (k - 9 > 15 - k → 0)) → 12 < k ∧ k < 15 :=
by
  sorry

end k_range_l511_511379


namespace fraction_cubed_equality_l511_511953

-- Constants for the problem
def A : ℝ := 81000
def B : ℝ := 9000

-- Problem statement
theorem fraction_cubed_equality : (A^3) / (B^3) = 729 :=
by
  sorry

end fraction_cubed_equality_l511_511953


namespace calc_expr_correct_l511_511055

noncomputable def eval_expr : ℚ :=
  57.6 * (8 / 5) + 28.8 * (184 / 5) - 14.4 * 80 + 12.5

theorem calc_expr_correct : eval_expr = 12.5 :=
by
  sorry

end calc_expr_correct_l511_511055


namespace find_angle_B_find_length_BD_l511_511166

-- Define the basics of our triangle and its properties
variables (a b c : ℝ) (A B C : ℝ)
variables (h1 : 2*c + a = 2*b*cos(A)) 

-- Prove that angle B equals 2π/3
theorem find_angle_B (h1 : 2*c + a = 2*b*cos(A)) : B = (2*π) / 3 :=
sorry

-- Given specific values for a, b, and c, and point D as the midpoint of AC
variables (a_val c_val : ℝ) (D : ℝ)
variables (h_a : a_val = 5) (h_c : c_val = 3) (h_mid : D = (5/2 + 3/2)) 

-- Prove the length of BD is sqrt(19)/2
theorem find_length_BD (h_a : a = 5) (h_c : c = 3) (h_mid : D = (a + c) / 2): ℝ :=
BD = (sq_root 19) / 2 :=
sorry

end find_angle_B_find_length_BD_l511_511166


namespace probability_AC_less_than_10_cm_l511_511505

-- Definition of the problem
def distance_between_points (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem probability_AC_less_than_10_cm :
  ∃ (α : ℝ) (h : 0 < α ∧ α < real.pi), 
  let B := (0, 0),
      A := (0, -12 : ℝ),
      C := (8 * real.cos α, 8 * real.sin α : ℝ) in
  ∃ (θ : ℝ), 
  ∃ (prob : ℝ), 
  θ =  real.atan2 (- √ (92 / 9)) (-22/3) - real.atan2 (√ (92 / 9)) (-22 / 3) ∧
  prob = θ / real.pi ∧
  distance_between_points A C < 10 ∧
  prob = 1 / 3 :=
  sorry

end probability_AC_less_than_10_cm_l511_511505


namespace greatest_real_part_of_z5_l511_511788

theorem greatest_real_part_of_z5 :
  let z1 := -3
  let z2 := -2 + 1/2 * Complex.i
  let z3 := -3/2 + 3/2 * Complex.i
  let z4 := -1 + 2 * Complex.i
  let z5 := 3 * Complex.i
  let f := λ (z : ℂ), (z^5).re
  f z1 = -243 ∧
  f z2 = -12.3125 ∧
  f z3 = -168.75 ∧
  f z4 = 39 ∧
  f z5 = 0 →
  (max (max (max (max (f z1) (f z2)) (f z3)) (f z4)) (f z5)) = f z4 :=
by
  intros z1 z2 z3 z4 z5 f h
  sorry

end greatest_real_part_of_z5_l511_511788


namespace sum_diff_1_to_999_l511_511628

def subtract_last_from_first (n : ℕ) : ℤ :=
  let str_n := n.toString
  if str_n.length = 1 then 0
  else
    let first_digit := str_n.toList.head!.digitToInt!
    let last_digit := str_n.toList.reverse.head!.digitToInt!
    first_digit - last_digit

def numbers : List ℕ := List.range 1000.tail

def sum_of_differences : ℤ := (numbers.map subtract_last_from_first).sum

theorem sum_diff_1_to_999 :
  sum_of_differences = 495 := 
sorry

end sum_diff_1_to_999_l511_511628


namespace negative_expression_l511_511388

noncomputable def U : ℝ := -2.5
noncomputable def V : ℝ := -0.8
noncomputable def W : ℝ := 0.4
noncomputable def X : ℝ := 1.0
noncomputable def Y : ℝ := 2.2

theorem negative_expression :
  (U - V < 0) ∧ ¬(U * V < 0) ∧ ¬((X / V) * U < 0) ∧ ¬(W / (U * V) < 0) ∧ ¬((X + Y) / W < 0) :=
by
  sorry

end negative_expression_l511_511388


namespace square_area_proof_l511_511728

theorem square_area_proof (W X Y Z M N O : Type) 
  [square WXYZ] (WZ WX segment : Type) 
  [on_segment M WZ] [on_segment N WX] 
  [intersect_at_right_angle WM YN O] 
  (WO : Real := 8) (MO : Real := 9) : 
  ∃ s : Real, s^2 = 385 :=
by
  sorry

end square_area_proof_l511_511728


namespace order_large_pizzas_sufficient_l511_511049

def pizza_satisfaction (gluten_free_slices_per_large : ℕ) (medium_slices : ℕ) (small_slices : ℕ) 
                       (gluten_free_needed : ℕ) (dairy_free_needed : ℕ) :=
  let slices_gluten_free := small_slices
  let slices_dairy_free := 2 * medium_slices
  (slices_gluten_free < gluten_free_needed) → 
  let additional_slices_gluten_free := gluten_free_needed - slices_gluten_free
  let large_pizzas_gluten_free := (additional_slices_gluten_free + gluten_free_slices_per_large - 1) / gluten_free_slices_per_large
  large_pizzas_gluten_free = 1

theorem order_large_pizzas_sufficient :
  pizza_satisfaction 14 10 8 15 15 :=
by
  unfold pizza_satisfaction
  sorry

end order_large_pizzas_sufficient_l511_511049


namespace sum_of_prime_numbers_in_base_14_form_l511_511091

def is_digit_form (A : ℕ) : Prop :=
  ∃ n : ℕ, A = ∑ i in Finset.range (n+1), 14^(2*i)

def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

theorem sum_of_prime_numbers_in_base_14_form :
  ∀ A, is_digit_form A → is_prime A → A = 197 :=
by
  sorry

end sum_of_prime_numbers_in_base_14_form_l511_511091


namespace locus_of_projection_of_O_l511_511483

theorem locus_of_projection_of_O (p : ℝ) (hp : 0 < p) :
  ∃ M : ℝ × ℝ, (let O := (0,0) in
                 let A := (4*p/(k^2), 4*p/k) in
                 let B := (4*p*k^2, -4*p*k) in
                 let AB := λ x y, (x - fst O)*(y - snd O) + (fst A - x)*(snd A - y) + (fst B - x)*(snd B - y) = 0 in
                 let M := projection_of_O_onto_AB O AB in
                 (M.1 - 2*p)^2 + M.2^2 = 4*p^2) :=
begin
  sorry
end

end locus_of_projection_of_O_l511_511483


namespace rationalize_sqrt_5_div_18_l511_511802

theorem rationalize_sqrt_5_div_18 :
  (Real.sqrt (5 / 18) = Real.sqrt 10 / 6) :=
sorry

end rationalize_sqrt_5_div_18_l511_511802


namespace five_letter_words_with_at_least_one_vowel_l511_511213

open Finset

theorem five_letter_words_with_at_least_one_vowel :
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := ((letters \ vowels).card : ℕ) ^ 5
  total_words - no_vowel_words = 6752 :=
by
  let letters := insert 'A' (insert 'B' (insert 'C' (insert 'D' (insert 'E' (singleton 'F')))))
  let vowels := insert 'A' (singleton 'E')
  let consonants := letters \ vowels
  have : letters.card = 6 := by simp
  have : vowels.card = 2 := by simp
  have : consonants.card = 4 := by simp
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := (consonants.card : ℕ) ^ 5
  calc
    total_words - no_vowel_words
        = 6^5 - 4^5         : by simp
    ... = 7776 - 1024       : by norm_num
    ... = 6752             : by norm_num

end five_letter_words_with_at_least_one_vowel_l511_511213


namespace units_digit_product_l511_511876

theorem units_digit_product (k l : ℕ) (h1 : ∀ n : ℕ, (5^n % 10) = 5) (h2 : ∀ m < 4, (6^m % 10) = 6) :
  ((5^k * 6^l) % 10) = 0 :=
by
  have h5 : (5^k % 10) = 5 := h1 k
  have h6 : (6^4 % 10) = 6 := h2 4 (by sorry)
  have h_product : (5^k * 6^l % 10) = ((5 % 10) * (6 % 10) % 10) := sorry
  norm_num at h_product
  exact h_product

end units_digit_product_l511_511876


namespace find_a_l511_511701

variable (a b c d : ℝ)

def equation (a b c d : ℝ) : Prop :=
  a^2 + b^2 + c^2 + d^2 - ab - bc - cd - d + 2/5 = 0

theorem find_a (h : equation a b c d) : a = 1/5 :=
sorry

end find_a_l511_511701


namespace n_minus_m_l511_511712

theorem n_minus_m (m n : ℤ) (h_m : m - 2 = 3) (h_n : n + 1 = 2) : n - m = -4 := sorry

end n_minus_m_l511_511712


namespace sequence_sum_S21_l511_511843

theorem sequence_sum_S21 :
  (∃ (a : ℕ → ℝ), (∀ n : ℕ, n ≥ 1 → a n + a (n + 1) = 1 / 2) ∧ (a 2 = 1) ∧
  (let S21 := a 1 + (a 2 + a 3) * 10
  in S21 = 9 / 2)) := sorry

end sequence_sum_S21_l511_511843


namespace five_letter_words_with_at_least_one_vowel_l511_511209

open Finset

theorem five_letter_words_with_at_least_one_vowel :
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := ((letters \ vowels).card : ℕ) ^ 5
  total_words - no_vowel_words = 6752 :=
by
  let letters := insert 'A' (insert 'B' (insert 'C' (insert 'D' (insert 'E' (singleton 'F')))))
  let vowels := insert 'A' (singleton 'E')
  let consonants := letters \ vowels
  have : letters.card = 6 := by simp
  have : vowels.card = 2 := by simp
  have : consonants.card = 4 := by simp
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := (consonants.card : ℕ) ^ 5
  calc
    total_words - no_vowel_words
        = 6^5 - 4^5         : by simp
    ... = 7776 - 1024       : by norm_num
    ... = 6752             : by norm_num

end five_letter_words_with_at_least_one_vowel_l511_511209


namespace sum_diff_1_to_999_l511_511632

def subtract_last_from_first (n : ℕ) : ℤ :=
  let str_n := n.toString
  if str_n.length = 1 then 0
  else
    let first_digit := str_n.toList.head!.digitToInt!
    let last_digit := str_n.toList.reverse.head!.digitToInt!
    first_digit - last_digit

def numbers : List ℕ := List.range 1000.tail

def sum_of_differences : ℤ := (numbers.map subtract_last_from_first).sum

theorem sum_diff_1_to_999 :
  sum_of_differences = 495 := 
sorry

end sum_diff_1_to_999_l511_511632


namespace plants_given_away_l511_511773

-- Define the conditions as constants
def initial_plants : ℕ := 3
def final_plants : ℕ := 20
def months : ℕ := 3

-- Function to calculate the number of plants after n months
def plants_after_months (initial: ℕ) (months: ℕ) : ℕ := initial * (2 ^ months)

-- The proof problem statement
theorem plants_given_away : (plants_after_months initial_plants months - final_plants) = 4 :=
by
  sorry

end plants_given_away_l511_511773


namespace length_n_words_no_adjacent_a_l511_511161

theorem length_n_words_no_adjacent_a (a : ℕ → ℝ) (n : ℕ) :
  (a 1 = 3) → (a 2 = 8) →
  (∀ n, n ≥ 3 → a n = 2 * a (n - 1) + 2 * a (n - 2)) →
  (a n = (2 + real.sqrt 3) / (2 * real.sqrt 3) * (1 + real.sqrt 3)^n +
         (-2 + real.sqrt 3) / (2 * real.sqrt 3) * (1 - real.sqrt 3)^n) :=
by
  intros h1 h2 h3
  sorry

end length_n_words_no_adjacent_a_l511_511161


namespace find_k_intersection_l511_511771

theorem find_k_intersection :
  ∃ (k : ℝ), 
  (∀ (x y : ℝ), y = 2 * x + 3 → y = k * x + 1 → (x = 1 ∧ y = 5) → k = 4) :=
sorry

end find_k_intersection_l511_511771


namespace hyperbola_no_common_points_l511_511318

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) : ℝ :=
  real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_no_common_points (a b e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_ecc : e = real.sqrt (1 + (b^2 / a^2)))
  (h_slope : b / a < 2) :
  e = 2 :=
sorry

end hyperbola_no_common_points_l511_511318


namespace five_letter_words_with_vowel_l511_511241

-- Define the set of letters and identify vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels := {'A', 'E'}

-- Define the problem statement
theorem five_letter_words_with_vowel : 
  (number of 5-letter words with at least one vowel) = 6752 := 
sorry

end five_letter_words_with_vowel_l511_511241


namespace parabola_equation_and_max_slope_l511_511684

-- Define the parabola parameters and conditions
def parabola (p : ℝ) (hp : p > 0) : set (ℝ × ℝ) := 
{xy | xy.2 ^ 2 = 2 * p * xy.1}

def focus_distance (d : ℝ) : Prop := 
d = 2

-- Define the points O, P, and Q and the given vector relationship
def point_o : ℝ × ℝ := (0, 0)

def on_parabola (p : ℝ) (hp : p > 0) (P : ℝ × ℝ) : Prop :=
P ∈ parabola p hp

def vector_relationship (P Q F : ℝ × ℝ) : Prop :=
P.1 - Q.1 = 9 * (Q.1 - F.1) ∧ P.2 - Q.2 = 9 * (Q.2 - F.2)

-- Define the conditions and the proof goals
theorem parabola_equation_and_max_slope :
  ∃ p (hp : p > 0) (F : ℝ × ℝ),
  focus_distance (2 * p) →
  (∀ P, on_parabola p hp P → 
       ∃ Q, vector_relationship P Q F →
             (parabola p hp = (λ xy, xy.2^2 = 4 * xy.1) ∧
             (real.slope point_o Q ≤ (1/3))) :=
by 
  -- Proof is omitted
  sorry

end parabola_equation_and_max_slope_l511_511684


namespace perpendicular_line_through_point_l511_511477

-- Declare that we are working in a noncomputable setting
noncomputable theory

-- Definitions based on the conditions
def point : ℝ × ℝ := (4, 1)
def given_line : ℝ → ℝ → Prop := λ x y, 4 * x - 5 * y + 9 = 0
def answer_line : ℝ → ℝ → Prop := λ x y, 5 * x + 4 * y - 24 = 0

-- The Lean theorem statement
theorem perpendicular_line_through_point :
  (∀ x y, given_line x y → answer_line x y) :=
sorry

end perpendicular_line_through_point_l511_511477


namespace zero_point_condition_l511_511382

variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem zero_point_condition :
  (∀ x ∈ set.Icc a b, continuous_at f x) → ((f a * f b < 0) ↔ (∃ x ∈ set.Icc a b, f x = 0 ∧ ∀ y ∈ set.Icc a b, f y = 0 → y = x)) → False :=
by
  sorry

end zero_point_condition_l511_511382


namespace opening_price_l511_511046

noncomputable def percent_increase : ℝ  :=
  33.33 / 100

noncomputable def closing_price : ℝ :=
  8

theorem opening_price (P : ℝ) :
  percent_increase = (closing_price - P) / P ∧ P ≈ 6 :=
by
  sorry

end opening_price_l511_511046


namespace evaluate_expression_l511_511656

theorem evaluate_expression (a b c : ℝ)
  (h : a / (25 - a) + b / (65 - b) + c / (60 - c) = 7) :
  5 / (25 - a) + 13 / (65 - b) + 12 / (60 - c) = 2 := 
sorry

end evaluate_expression_l511_511656


namespace necessary_but_not_sufficient_l511_511898

theorem necessary_but_not_sufficient (x : ℝ) : (x^2 - 3 * x - 4 = 0) -> (x = 4 ∨ x = -1) ∧ ¬(x = 4 ∨ x = -1 -> x = 4) :=
by sorry

end necessary_but_not_sufficient_l511_511898


namespace rectangle_area_comparison_l511_511507

theorem rectangle_area_comparison 
  {A A' B B' C C' D D': ℝ} 
  (h_A: A ≤ A') 
  (h_B: B ≤ B') 
  (h_C: C ≤ C') 
  (h_D: D ≤ B') : 
  A + B + C + D ≤ A' + B' + C' + D' := 
by 
  sorry

end rectangle_area_comparison_l511_511507


namespace scientific_notation_correct_l511_511878

theorem scientific_notation_correct : 1630000 = 1.63 * 10^6 :=
by sorry

end scientific_notation_correct_l511_511878


namespace monotonic_intervals_of_f_g_minus_f_lt_3_l511_511128

noncomputable def f (x : ℝ) : ℝ := -x * Real.log (-x)

noncomputable def g (x : ℝ) : ℝ := Real.exp x - x

-- Theorem 1: Monotonic intervals of f(x)
theorem monotonic_intervals_of_f : 
  (∀ x : ℝ, x ∈ set.Ioo (-∞) (-1 / Real.exp 1) → f' x < 0) ∧ 
  (∀ x : ℝ, x ∈ set.Ioo (-1 / Real.exp 1) 0 → f' x > 0) :=
sorry

-- Theorem 2: Proving g(x) - f(x) < 3
theorem g_minus_f_lt_3 (x : ℝ) : g(x) - f(x) < 3 :=
sorry

end monotonic_intervals_of_f_g_minus_f_lt_3_l511_511128


namespace sum_of_n_values_l511_511445

theorem sum_of_n_values :
  ∑ n in { n : ℕ | lcm n 120 = gcd n 120 + 600 }.to_finset = 2520 :=
by sorry

end sum_of_n_values_l511_511445


namespace parabola_has_correct_equation_l511_511484

noncomputable def parabola_equation := ∃ (x y : ℝ),
  let focus := (2, 5) in
  let directrix (x y : ℝ) := 4 * x + 5 * y - 20 in
  let parabola : ℝ :=
    25 * x^2 - 40 * x * y + 16 * y^2 - 4 * x - 210 * y + 51 in
  parabola = 0

theorem parabola_has_correct_equation : 
  parabola_equation := by
  -- Proof omitted
  sorry

end parabola_has_correct_equation_l511_511484


namespace emmett_situps_eq_20_l511_511971

-- Defining the conditions
def jumping_jacks : ℕ := 12
def pushups : ℕ := 8
def percentage_pushups : ℝ := 0.20

-- Define the total number of exercises
def total_exercises : ℕ := (pushups : ℝ / percentage_pushups).to_nat

-- Total number of situps
def situps : ℕ := total_exercises - (jumping_jacks + pushups)

-- The theorem to prove
theorem emmett_situps_eq_20 :
  situps = 20 :=
by
  sorry

end emmett_situps_eq_20_l511_511971


namespace probability_coprime_l511_511425

open BigOperators

theorem probability_coprime (A : Finset ℕ) (h : A = {1, 2, 3, 4, 5, 6, 7, 8}) :
  let pairs := { (a, b) ∈ (A ×ˢ A) | a < b }
  let coprime_pairs := pairs.filter (λ p, Nat.gcd p.1 p.2 = 1)
  coprime_pairs.card / pairs.card = 5 / 7 := by 
sorry

end probability_coprime_l511_511425


namespace sqrt_meaningful_iff_ge_eight_l511_511264

theorem sqrt_meaningful_iff_ge_eight (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 8)) ↔ x ≥ 8 := by
  sorry

end sqrt_meaningful_iff_ge_eight_l511_511264


namespace degree_f_plus_g_l511_511820

noncomputable def f (z : ℂ) : ℂ := a₃ * z^3 + a₂ * z^2 + a₁ * z + a₀
noncomputable def g (z : ℂ) : ℂ := b₁ * z + b₀

theorem degree_f_plus_g (a₃ a₂ a₁ a₀ b₁ b₀ : ℂ) (h : a₃ ≠ 0) : degree (f + g) = 3 :=
by
  sorry

end degree_f_plus_g_l511_511820


namespace five_letter_words_with_vowel_l511_511238

-- Define the set of letters and identify vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels := {'A', 'E'}

-- Define the problem statement
theorem five_letter_words_with_vowel : 
  (number of 5-letter words with at least one vowel) = 6752 := 
sorry

end five_letter_words_with_vowel_l511_511238


namespace cos_alpha_plus_20_eq_neg_alpha_l511_511658

variable (α : ℝ)

theorem cos_alpha_plus_20_eq_neg_alpha (h : Real.sin (α - 70 * Real.pi / 180) = α) :
    Real.cos (α + 20 * Real.pi / 180) = -α :=
by
  sorry

end cos_alpha_plus_20_eq_neg_alpha_l511_511658


namespace rainfall_on_Monday_l511_511300

theorem rainfall_on_Monday (rain_on_Tuesday : ℝ) (difference : ℝ) (rain_on_Tuesday_eq : rain_on_Tuesday = 0.2) (difference_eq : difference = 0.7) :
  ∃ rain_on_Monday : ℝ, rain_on_Monday = rain_on_Tuesday + difference := 
sorry

end rainfall_on_Monday_l511_511300


namespace euler_lines_intersect_l511_511330

open EuclideanGeometry

theorem euler_lines_intersect
  {A B C P : Point}
  (h_in_triangle : in_triangle P A B C)
  (h_angle_APB : ∠APB = 120)
  (h_angle_BPC : ∠BPC = 120)
  (h_angle_CPA : ∠CPA = 120)
  (h_tri_angles_lt_120 : ∀ α ∈ {∠BAC, ∠ABC, ∠BCA}, α < 120) :
  let Δ_APB := (A, P, B),
      Δ_BPC := (B, P, C),
      Δ_CPA := (C, P, A),
      Euler_APB := euler_line Δ_APB,
      Euler_BPC := euler_line Δ_BPC,
      Euler_CPA := euler_line Δ_CPA
  in intersects_at_one_point Euler_APB Euler_BPC Euler_CPA :=
  sorry

end euler_lines_intersect_l511_511330


namespace double_bed_heavier_than_single_bed_l511_511847

theorem double_bed_heavier_than_single_bed 
  (S D : ℝ) 
  (h1 : 5 * S = 50) 
  (h2 : 2 * S + 4 * D = 100) 
  : D - S = 10 :=
sorry

end double_bed_heavier_than_single_bed_l511_511847


namespace angles_with_same_terminal_side_l511_511977

theorem angles_with_same_terminal_side (k : ℤ) :
  let beta := 45 + k * 360 in
  -720 ≤ beta ∧ beta < 0 ↔ beta = -675 ∨ beta = -315 := 
by 
  sorry

end angles_with_same_terminal_side_l511_511977


namespace det_M_power_4_l511_511261

variables {M : Type*} [Field M] [Matrix M]

theorem det_M_power_4 (A : Matrix M M) (h : det A = -2) : det (A^4) = 16 :=
sorry

end det_M_power_4_l511_511261


namespace distance_from_point_to_line_l511_511703

variable (A B C P : Type)
variable (l : Line)
variable [metric_space P]

-- Definitions to denote points on the line, point outside the line, and distances
def on_line (A : P) (l : Line) : Prop := A ∈ l
def outside_line (P : P) (l : Line) : Prop := ¬ (P ∈ l)
def distance (P Q : P) : ℝ := sorry  -- placeholder for the actual distance metric
def perpendicular (P Q : P) (l : Line) : Prop := sorry  -- placeholder for perpendicular relation
def PA : P := sorry  -- placeholder for the point PA

-- Given Conditions
variables (ha : on_line A l) (hb : on_line B l) (hc : on_line C l)
           (hp : outside_line P l) (perp_PA_l : perpendicular P A l)
           (dist_PA : distance P A = 5) (dist_PB : distance P B = 6)
           (dist_PC : distance P C = 7)

-- Theorem statement
theorem distance_from_point_to_line : distance_from_point_to_line P l = 5 :=
sorry

end distance_from_point_to_line_l511_511703


namespace arven_puppies_on_sale_l511_511044

theorem arven_puppies_on_sale :
  ∃ x : ℕ, (x ≥ 0) ∧ (2 = 5 - x) ∧ (150 * x + 2 * 175 = 800) ∧ (x = 3) :=
by
  use 3
  split; norm_num
  split; refl
  split; norm_num
  sorry

end arven_puppies_on_sale_l511_511044


namespace partition_even_sum_l511_511333

theorem partition_even_sum (m : ℕ) (a : Fin m → ℕ) (h_even : Even m)
  (h_sorted : ∀ i j : Fin m, i ≤ j → a i ≤ a j) (h_sum : ∑ i : Fin m, a i = 2 * m) :
  ∃ (s1 s2 : Finset (Fin m)), s1 ∩ s2 = ∅ ∧ s1 ∪ s2 = Finset.univ ∧ (∑ i in s1, a i) = m ∧ (∑ i in s2, a i) = m :=
by
  sorry

end partition_even_sum_l511_511333


namespace check_rectangle_if_diagonals_equal_l511_511695

structure Quadrilateral (V : Type) :=
  (A B C D : V)
  (side_AB : ℝ)
  (side_BC : ℝ)
  (side_CD : ℝ)
  (side_DA : ℝ)

def are_diagonals_equal {V : Type} [MetricSpace V] (q : Quadrilateral V) : Prop :=
  dist q.A q.C = dist q.B q.D

def is_rectangle {V : Type} [MetricSpace V] (q : Quadrilateral V) : Prop :=
  quadrilateral.A q q.A q.B q.C q.D → q.side_AB = q.side_CD ∧ q.side_BC = q.side_DA

theorem check_rectangle_if_diagonals_equal {V : Type} [MetricSpace V] (q : Quadrilateral V) :
  are_diagonals_equal q → is_rectangle q :=
by
  sorry

end check_rectangle_if_diagonals_equal_l511_511695


namespace james_total_spent_l511_511469

open_locale big_operators

noncomputable theory

def entry_fee : ℕ := 20
def drink_cost : ℕ := 6
def drinks_for_friends : ℕ := 2 * 5
def drinks_for_himself : ℕ := 6
def food_cost : ℕ := 14
def tip_rate : ℚ := 0.3

def total_drinks : ℕ := drinks_for_friends + drinks_for_himself

def drink_total_cost : ℕ := total_drinks * drink_cost

def food_and_drink_total : ℕ := drink_total_cost + food_cost

def tip : ℚ := food_and_drink_total * tip_rate

def total_cost_with_tip : ℚ := food_and_drink_total + tip

def final_cost : ℚ := total_cost_with_tip + entry_fee

theorem james_total_spent : final_cost = 163 := by 
  sorry

end james_total_spent_l511_511469


namespace max_slope_of_OQ_l511_511677

-- Assuming the conditions and problem setup
theorem max_slope_of_OQ :
  let C := {p : ℝ // p = 2}
  let equation_C := ∀ (x y : ℝ), y^2 = 4 * x
  let F : ℝ × ℝ := (1, 0)
  let O : ℝ × ℝ := (0, 0)
  let P := ∀ (m n : ℝ), (10 * m - 9, 10 * n)
  let Q := ∀ (m n : ℝ), (m, n)
  let K : ℝ := ∀ (m n : ℝ), (10 * n) / (25 * n^2 + 9)
  max_slope_of_line_OQ : ∃ (K_max : ℝ), K_max = 1 / 3 :=
  sorry

end max_slope_of_OQ_l511_511677


namespace lucky_set_exists_lucky_set_bound_l511_511120

def isLuckySet (n : ℕ) (a : Fin n → ℕ) : Prop :=
  ∀ (x : Fin n → ℕ), (∑ i, a i = (∑ i, a i)) → (x = fun i => 1 : Fin n → ℕ)

theorem lucky_set_exists (n : ℕ) (h : n > 1) :
  ∃ (a : Fin n → ℕ), isLuckySet n a ∧ (∑ i, a i < n * 2^n) := sorry

theorem lucky_set_bound (n : ℕ) (h : n > 1) (a : Fin n → ℕ) (hlucky : isLuckySet n a) :
  (∑ i, a i > n * 2^(n - 1)) := sorry

end lucky_set_exists_lucky_set_bound_l511_511120


namespace probability_gcd_one_l511_511432

-- Defining the domain of our problem: the set {1, 2, 3, ..., 8}
def S := {1, 2, 3, 4, 5, 6, 7, 8}

-- Defining the selection of two distinct natural numbers from S
def select_two_distinct_from_S (x y : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ x ≠ y

-- Defining the greatest common factor condition
def is_rel_prime (x y : ℕ) : Prop :=
  Nat.gcd x y = 1

-- Defining the probability computation (relatively prime pairs over total pairs)
def probability_rel_prime : ℚ :=
  (21 : ℚ) / 28  -- since 21 pairs are relatively prime out of 28 total pairs

-- The main theorem statement
theorem probability_gcd_one :
  probability_rel_prime = 3 / 4 :=
sorry

end probability_gcd_one_l511_511432


namespace number_of_new_students_l511_511721

-- Definitions and conditions
def average_budget_before : ℝ := 50
def total_students_before : ℕ := 100
def total_expenditure_before : ℝ := total_students_before * average_budget_before

def average_budget_after (x : ℕ) : ℝ := average_budget_before - 10
def total_expenditure_after (x : ℕ) : ℝ := (total_students_before + x) * average_budget_after x
def spending_diff (y : ℝ) : ℝ := y
def room_diff_const (k : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ ∀ (x : ℕ), spending_diff (k * x) = k * x

def sum_individual_spending_diff (z : ℝ) : ℝ := total_students_before + z
def spending_proportional (y z : ℝ) : Prop := y = sum_individual_spending_diff z

def total_expenditure_increase : ℝ := 400
def total_expenditure_after_fixed : ℝ := 5400
def condition_total_expenditure_increase (E : ℝ) : Prop := E + total_expenditure_increase = total_expenditure_after_fixed

-- Proof goal
theorem number_of_new_students (x : ℕ) (B z y k : ℝ) (h₁ : average_budget_before = B)
  (h₂ : average_budget_after x = B - 10)
  (h₃ : spending_diff y = k * x)
  (h₄ : sum_individual_spending_diff z = y)
  (h₅ : total_expenditure_after x = total_expenditure_before + total_expenditure_increase)
  (h₆ : condition_total_expenditure_increase total_expenditure_before := 100 * B)
  (h₇ : total_expenditure_before = 5000)  -- derived from B = 50
  : x = 35 := 
by {
  -- implicit proof
 sorry
}

end number_of_new_students_l511_511721


namespace max_distance_with_optimal_tire_swapping_l511_511103

theorem max_distance_with_optimal_tire_swapping
  (front_tires_last : ℕ)
  (rear_tires_last : ℕ)
  (front_tires_last_eq : front_tires_last = 20000)
  (rear_tires_last_eq : rear_tires_last = 30000) :
  ∃ D : ℕ, D = 30000 :=
by
  sorry

end max_distance_with_optimal_tire_swapping_l511_511103


namespace pat_donut_selection_l511_511795

noncomputable def num_ways_to_buy_donuts : ℕ :=
  let glazed := 0
  let chocolate := 1
  let powdered := 2
  let jelly := 3
  let num_types := 4
  let total_donuts := 5
  let min_jelly := 1
  let remaining_donuts := total_donuts - min_jelly
  nat.choose (remaining_donuts + num_types - 1) (num_types - 1)

theorem pat_donut_selection : num_ways_to_buy_donuts = 35 :=
by simp [num_ways_to_buy_donuts, nat.choose]; sorry

end pat_donut_selection_l511_511795


namespace lollipop_problem_l511_511169

def Henry_lollipops (A : Nat) : Nat := A + 30
def Diane_lollipops (A : Nat) : Nat := 2 * A
def Total_days (H A D : Nat) (daily_rate : Nat) : Nat := (H + A + D) / daily_rate

theorem lollipop_problem
  (A : Nat) (H : Nat) (D : Nat) (daily_rate : Nat)
  (h₁ : A = 60)
  (h₂ : H = Henry_lollipops A)
  (h₃ : D = Diane_lollipops A)
  (h₄ : daily_rate = 45)
  : Total_days H A D daily_rate = 6 := by
  sorry

end lollipop_problem_l511_511169


namespace initial_bottle_caps_l511_511066

theorem initial_bottle_caps (X : ℕ) (h1 : X - 60 + 58 = 67) : X = 69 := by
  sorry

end initial_bottle_caps_l511_511066


namespace tod_driving_time_l511_511413
noncomputable def total_driving_time (distance_north distance_west speed : ℕ) : ℕ :=
  (distance_north + distance_west) / speed

theorem tod_driving_time :
  total_driving_time 55 95 25 = 6 :=
by
  sorry

end tod_driving_time_l511_511413


namespace five_letter_words_with_vowels_l511_511178

noncomputable def num_5_letter_words_with_vowels : Nat := 7776 - 1024

theorem five_letter_words_with_vowels
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'})
  (len : Nat := 5) :
  (letters.card ^ len) - ((letters.filter (λ c, ¬ c ∈ vowels)).card ^ len) = 6752 := by
  sorry

end five_letter_words_with_vowels_l511_511178


namespace small_square_area_two_decimal_places_l511_511403

def side_length_of_square := 20 -- cm

def side_length_of_third := side_length_of_square / 3 -- cm

def area_of_small_square := (side_length_of_third) ^ 2

theorem small_square_area_two_decimal_places :
    Float.modf (Float.of_nat (area_of_small_square * 10^2)) = 44 :=
by
    sorry

end small_square_area_two_decimal_places_l511_511403


namespace maryann_work_time_l511_511774

variables (C A R : ℕ)

theorem maryann_work_time
  (h1 : A = 2 * C)
  (h2 : R = 6 * C)
  (h3 : C + A + R = 1440) :
  C = 160 ∧ A = 320 ∧ R = 960 :=
by
  sorry

end maryann_work_time_l511_511774


namespace matrix_inverse_identity_l511_511510

open Matrix

variable {n : Type*} [DecidableEq n] [Fintype n]
variable {α : Type*} [CommRing α] [Invertible (1 : α)]

theorem matrix_inverse_identity (B : Matrix n n α) 
  (h_inv : Invertible B) 
  (h_eq : (B - 3 * 1) ⬝ (B - 5 * 1) = 0) : 
  B + 10 * ⅟ B = (40 / 3 : α) • 1 :=
begin
  sorry
end

end matrix_inverse_identity_l511_511510


namespace equal_area_centroid_l511_511415

noncomputable def centroid (P Q R : (ℝ × ℝ)) : (ℝ × ℝ) :=
  let (x1, y1) := P
  let (x2, y2) := Q
  let (x3, y3) := R
  ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)

theorem equal_area_centroid : 
  let P := (7, 10)
  let Q := (2, -4)
  let R := (9, 3)
  let S := centroid P Q R
  8 * S.1 + 3 * S.2 = 57 :=
by
  -- Definitions from the conditions
  let P := (7, 10) : ℝ × ℝ
  let Q := (2, -4) : ℝ × ℝ
  let R := (9, 3) : ℝ × ℝ
  let (x1, y1) := P
  let (x2, y2) := Q
  let (x3, y3) := R
  
  -- Proof begins with the calculation of the centroid
  let S := centroid P Q R
  let x := (x1 + x2 + x3) / 3
  let y := (y1 + y2 + y3) / 3

  -- Show that S coordinates match the centroid formula
  have : S = (x, y) := by
    simp [S, centroid]
    
  -- Calculate 8x + 3y
  calc
  8 * x + 3 * y
      = 8 * ((x1 + x2 + x3) / 3) + 3 * ((y1 + y2 + y3) / 3) : by rw [← this]
  ... = 8 * 6 + 3 * 3 : by simp [x1, y1, x2, y2, x3, y3] 
  ... = 48 + 9 : by ring
  ... = 57 : by ring

end equal_area_centroid_l511_511415


namespace S_rational_iff_divides_l511_511305

-- Definition of "divides" for positive integers
def divides (m k : ℕ) : Prop := ∃ j : ℕ, k = m * j

-- Definition of the series S(m, k)
noncomputable def S (m k : ℕ) : ℝ := 
  ∑' n, 1 / (n * (m * n + k))

-- Proof statement
theorem S_rational_iff_divides (m k : ℕ) (hm : 0 < m) (hk : 0 < k) : 
  (∃ r : ℚ, S m k = r) ↔ divides m k :=
sorry

end S_rational_iff_divides_l511_511305


namespace f_bijective_solve_f_l511_511332

variable {R : Type*} [LinearOrderedField R]

def f (x : R) : R := sorry

axiom functional_eqn (f : R → R) (x y : R) : (y+1) * f(x) + f(x * f(y) + f(x + y)) = y

theorem f_bijective (f : R → R) (h : ∀ x y, (y+1) * f(x) + f(x * f(y) + f(x + y)) = y) : function.bijective f := sorry

theorem solve_f (f : R → R) (h : ∀ x y, (y+1) * f(x) + f(x * f(y) + f(x + y)) = y) :
  ∀ x, f(x) = -x := sorry

end f_bijective_solve_f_l511_511332


namespace probability_gcd_one_l511_511433

-- Defining the domain of our problem: the set {1, 2, 3, ..., 8}
def S := {1, 2, 3, 4, 5, 6, 7, 8}

-- Defining the selection of two distinct natural numbers from S
def select_two_distinct_from_S (x y : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ x ≠ y

-- Defining the greatest common factor condition
def is_rel_prime (x y : ℕ) : Prop :=
  Nat.gcd x y = 1

-- Defining the probability computation (relatively prime pairs over total pairs)
def probability_rel_prime : ℚ :=
  (21 : ℚ) / 28  -- since 21 pairs are relatively prime out of 28 total pairs

-- The main theorem statement
theorem probability_gcd_one :
  probability_rel_prime = 3 / 4 :=
sorry

end probability_gcd_one_l511_511433


namespace find_P_coordinates_l511_511654

-- Given points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 2)

-- The area of triangle PAB is 5
def areaPAB (P : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (A.2 - B.2) + A.1 * (B.2 - P.2) + B.1 * (P.2 - A.2))

-- Point P lies on the x-axis
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0

theorem find_P_coordinates (P : ℝ × ℝ) :
  on_x_axis P → areaPAB P = 5 → (P = (-4, 0) ∨ P = (6, 0)) :=
by
  sorry

end find_P_coordinates_l511_511654


namespace ratio_of_areas_l511_511466

-- Definitions based on the conditions
def original_radius (r : ℝ) := r
def new_radius (r : ℝ) := 3 * r
def original_area (r : ℝ) := Real.pi * (original_radius r)^2
def enlarged_area (r : ℝ) := Real.pi * (new_radius r)^2

-- Ratio of original area to enlarged area
theorem ratio_of_areas (r : ℝ) : (original_area r) / (enlarged_area r) = 1 / 9 :=
by
  sorry

end ratio_of_areas_l511_511466


namespace isosceles_triangle_AM_eq_AB_plus_BC_l511_511723

-- Define the isosceles triangle properties and distances
variables {A B C M : Type}
variables (x d h : ℝ) -- BM, AB (and AC), and BC

-- Main theorem statement
theorem isosceles_triangle_AM_eq_AB_plus_BC
  (h_pos : 0 < h) (d_pos : 0 < d)
  (AM_eq_AB_plus_BC : AM = d + h)
  (BM_MC_rel : x + (h - x) = h)
  (isosceles_cond : AB = AC = d) :
  x = (h + (sqrt (2 * d^2 + 3 * d * h))) / 2 := sorry

end isosceles_triangle_AM_eq_AB_plus_BC_l511_511723


namespace five_letter_words_with_vowels_l511_511225

/-
How many 5-letter words with at least one vowel can be constructed from the letters 
A, B, C, D, E, and F? (Note that A and E are vowels, any word is valid, not just English language 
words, and letters may be used more than once.)
-/

theorem five_letter_words_with_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := ['A', 'E'],
      consonants := ['B', 'C', 'D', 'F'] in
  let total_words := 6 ^ 5,
      consonant_only_words := 4 ^ 5,
      at_least_one_vowel_words := total_words - consonant_only_words in
  at_least_one_vowel_words = 6752 :=
by
  intro letters vowels consonants total_words consonant_only_words at_least_one_vowel_words
  sorry

end five_letter_words_with_vowels_l511_511225


namespace balls_cost_price_eq_loss_l511_511790

theorem balls_cost_price_eq_loss :
  ∀ (cp_per_ball sp_for_13_balls : ℕ), 
  cp_per_ball = 90 → sp_for_13_balls = 720 →
  let cp_for_13_balls := 13 * cp_per_ball in
  let total_loss := cp_for_13_balls - sp_for_13_balls in
  total_loss / cp_per_ball = 5 :=
by
  intros cp_per_ball sp_for_13_balls hcp hsp;
  have h_cp13 : cp_for_13_balls = 13 * cp_per_ball := rfl;
  have h_loss : total_loss = cp_for_13_balls - sp_for_13_balls := rfl;
  have h_total_loss : total_loss = 1170 - 720, from by {
    rw [hcp, hsp, h_cp13],
    norm_num
  };
  have h_correct_answer : total_loss / cp_per_ball = 5, from by {
    rw [h_total_loss, hcp],
    norm_num
  };
  exact h_correct_answer

end balls_cost_price_eq_loss_l511_511790


namespace count_5_letter_words_with_at_least_one_vowel_l511_511202

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := (letters.card) ^ 5

def non_vowel_letters : Finset Char := letters \ vowels

def total_non_vowel_5_letter_words : ℕ := (non_vowel_letters.card) ^ 5

theorem count_5_letter_words_with_at_least_one_vowel :
  total_5_letter_words - total_non_vowel_5_letter_words = 6752 :=
by
  sorry

end count_5_letter_words_with_at_least_one_vowel_l511_511202


namespace sum_diff_1_to_999_l511_511631

def subtract_last_from_first (n : ℕ) : ℤ :=
  let str_n := n.toString
  if str_n.length = 1 then 0
  else
    let first_digit := str_n.toList.head!.digitToInt!
    let last_digit := str_n.toList.reverse.head!.digitToInt!
    first_digit - last_digit

def numbers : List ℕ := List.range 1000.tail

def sum_of_differences : ℤ := (numbers.map subtract_last_from_first).sum

theorem sum_diff_1_to_999 :
  sum_of_differences = 495 := 
sorry

end sum_diff_1_to_999_l511_511631


namespace monotonically_increasing_interval_cos_value_given_f_l511_511145
open Real

noncomputable def f (x : ℝ) := 2 * sin (x + π/6) - 2 * cos x

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x, -π/3 + 2 * k * π ≤ x ∧ x ≤ 2 * π/3 + 2 * k * π → is_increasing_on f (set.Icc (-π/3 + 2 * k * π) (2 * π/3 + 2 * k * π)) :=
sorry

theorem cos_value_given_f :
  ∃ x, f x = 6/5 ∧ cos (2 * x - π/3) = 7/25 :=
sorry

end monotonically_increasing_interval_cos_value_given_f_l511_511145


namespace total_students_in_class_l511_511281

theorem total_students_in_class:
  ∃ (n : ℕ), let n_a := (n * 1 / 7), let n_a' := n_a + 1 in
  n ≠ 0 ∧ (n_a' = n * 1 / 6) ∧ n = 42 :=
by
  sorry

end total_students_in_class_l511_511281


namespace final_price_on_monday_l511_511947

-- Definitions based on the conditions
def saturday_price : ℝ := 50
def sunday_increase : ℝ := 1.2
def monday_discount : ℝ := 0.2

-- The statement to prove
theorem final_price_on_monday : 
  let sunday_price := saturday_price * sunday_increase
  let monday_price := sunday_price * (1 - monday_discount)
  monday_price = 48 :=
by
  sorry

end final_price_on_monday_l511_511947


namespace pyramid_cube_volume_l511_511487

def square_based_pyramid (b : ℝ) : Prop :=
  b = 2

def equilateral_triangle_faces (t : ℝ) : Prop :=
  t = 2 * Real.sqrt 2

def cube_within_pyramid (s : ℝ) : Prop :=
  s = Real.sqrt 6 / 2

theorem pyramid_cube_volume :
  ∀ (b t s : ℝ),
  square_based_pyramid b →
  equilateral_triangle_faces t →
  cube_within_pyramid s →
  s^3 = 3 * Real.sqrt 6 / 4 :=
by
  intros b t s hb ht hs
  rw [hb, ht, hs]
  calc (Real.sqrt 6 / 2) ^ 3 = (6 * Real.sqrt 6) / (2^3) : by sorry

end pyramid_cube_volume_l511_511487


namespace f_monotonically_increasing_on_0_to_pi_over_8_f_has_two_zeros_in_interval_0_to_pi_l511_511153

def f (x : Real) : Real := -2 * (Real.sin x) ^ 2 + Real.sin (2 * x) + 1

theorem f_monotonically_increasing_on_0_to_pi_over_8 :
  ∀ x y : Real, 0 < x ∧ x < π / 8 ∧ 0 < y ∧ y < π / 8 ∧ x < y → f x < f y :=
by
  sorry

theorem f_has_two_zeros_in_interval_0_to_pi :
  ∃ a b : Real, 0 ≤ a ∧ a ≤ π ∧ 0 ≤ b ∧ b ≤ π ∧ a ≠ b ∧ f a = 0 ∧ f b = 0 :=
by
  sorry

end f_monotonically_increasing_on_0_to_pi_over_8_f_has_two_zeros_in_interval_0_to_pi_l511_511153


namespace cylinder_volume_correct_l511_511350

noncomputable def cylinder_volume (l α β : ℝ) : ℝ :=
  (π * l^3 * sin (2 * α) * cos(α) ^ 3) / (8 * cos (α + β) * cos (α - β))

theorem cylinder_volume_correct (l α β : ℝ) :
  (∀ A B : ℝ × ℝ × ℝ,
    A.2 = l * sin α ∧  -- A is on the upper base circumference
    B.2 = 0 ∧         -- B is on the lower base circumference
    (l ≠ 0) ∧
    (cos (α + β) ≠ 0) ∧
    (cos (α - β) ≠ 0)) →
  cylinder_volume l α β = (π * l^3 * sin (2 * α) * cos α ^ 3) / (8 * cos (α + β) * cos (α - β)) :=
by
  sorry

end cylinder_volume_correct_l511_511350


namespace sum_digit_differences_l511_511608

def first_digit (n : ℕ) : ℕ := 
  (n / 10 ^ ((Nat.log10 n) : ℕ))

def last_digit (n : ℕ) : ℕ := n % 10

def digit_difference (n : ℕ) : ℤ :=
  (first_digit n : ℤ) - (last_digit n : ℤ)

theorem sum_digit_differences :
  (∑ n in Finset.range 1000, digit_difference n) = 495 := 
sorry

end sum_digit_differences_l511_511608


namespace sum_possible_k_values_l511_511095

-- defining gcd and k_bad conditions
def gcd (a b : ℕ) := ∀ d : ℕ, d ∣ a ∧ d ∣ b → d ≤ a ∧ d ≤ b

def k_bad (k N : ℕ) := ¬ ∃ x y : ℕ, N = 2020 * x + k * y

-- main theorem translation based on conditions
theorem sum_possible_k_values : ∑ k in {k : ℕ | k > 1 ∧ gcd k 2020 = 1 ∧ 
  ∀ m n : ℕ, m > 0 ∧ n > 0 ∧ m + n = 2019 * (k - 1) ∧ m ≥ n ∧ k_bad k m → k_bad k n}, k = 2360 :=
sorry

end sum_possible_k_values_l511_511095


namespace sum_of_differences_l511_511613

/-- Proving that the sum of all differences (first digit - last digit) for natural numbers from 
    1 to 999 is 495. -/ 
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, 
    let str := n.toString
    let first := if str.length > 1 then str.head!.toNat - '0'.toNat else 0
    let last := if str.length > 1 then str.getLast.toNat - '0'.toNat else 0
    first - last
  ) = 495 := 
by
  sorry

end sum_of_differences_l511_511613


namespace count_5_letter_words_with_at_least_one_vowel_l511_511204

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := (letters.card) ^ 5

def non_vowel_letters : Finset Char := letters \ vowels

def total_non_vowel_5_letter_words : ℕ := (non_vowel_letters.card) ^ 5

theorem count_5_letter_words_with_at_least_one_vowel :
  total_5_letter_words - total_non_vowel_5_letter_words = 6752 :=
by
  sorry

end count_5_letter_words_with_at_least_one_vowel_l511_511204


namespace female_with_advanced_degrees_l511_511886

theorem female_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_employees_with_advanced_degrees : ℕ)
  (total_employees_with_college_degree_only : ℕ)
  (total_males_with_college_degree_only : ℕ)
  (h1 : total_employees = 180)
  (h2 : total_females = 110)
  (h3 : total_employees_with_advanced_degrees = 90)
  (h4 : total_employees_with_college_degree_only = 90)
  (h5 : total_males_with_college_degree_only = 35) :
  ∃ (female_with_advanced_degrees : ℕ), female_with_advanced_degrees = 55 :=
by
  -- the proof goes here
  sorry

end female_with_advanced_degrees_l511_511886


namespace vampire_conversion_l511_511434

theorem vampire_conversion (x : ℕ) 
  (h_population : village_population = 300)
  (h_initial_vampires : initial_vampires = 2)
  (h_two_nights_vampires : 2 + 2 * x + x * (2 + 2 * x) = 72) :
  x = 5 :=
by
  -- Proof will be added here
  sorry

end vampire_conversion_l511_511434


namespace I_n_eq_sum_I_n_l511_511744

noncomputable def I (n : ℕ) : ℝ := (1 / (n + 1)) * ∫ (x : ℝ) in 0..Real.pi, x * (Real.sin (n * x) + n * Real.pi * Real.cos (n * x))

theorem I_n_eq (n : ℕ) (hn : n > 0) :
  I n = Real.pi / (n * (n + 1)) := by sorry

theorem sum_I_n :
  ∑' n : ℕ, if h : n > 0 then I n else 0 = Real.pi := by sorry

end I_n_eq_sum_I_n_l511_511744


namespace height_large_cylinder_is_10_l511_511372

noncomputable def height_large_cylinder : ℝ :=
  let V_small := 13.5 * Real.pi
  let factor := 74.07407407407408
  let V_large := 100 * Real.pi
  factor * V_small / V_large

theorem height_large_cylinder_is_10 :
  height_large_cylinder = 10 :=
by
  sorry

end height_large_cylinder_is_10_l511_511372


namespace part1_part2_l511_511113

open Complex

noncomputable def z0 : ℂ := 3 + 4 * Complex.I

theorem part1 (z1 : ℂ) (h : z1 * z0 = 3 * z1 + z0) : z1.im = -3/4 := by
  sorry

theorem part2 (x : ℝ) 
    (z : ℂ := (x^2 - 4 * x) + (x + 2) * Complex.I) 
    (z0_conj : ℂ := 3 - 4 * Complex.I) 
    (h : (z + z0_conj).re < 0 ∧ (z + z0_conj).im > 0) : 
    2 < x ∧ x < 3 :=
  by 
  sorry

end part1_part2_l511_511113


namespace range_of_m_l511_511826

def one_root_condition (m : ℝ) : Prop :=
  (4 - 4 * m) * (2 * m + 4) ≤ 0 ∧ m ≠ 0

theorem range_of_m : {m : ℝ | (4 - 4 * m) * (2 * m + 4) ≤ 0 ∧ m ≠ 0} = {m | m ≤ -2 ∨ m ≥ 1} :=
by
  sorry

end range_of_m_l511_511826


namespace monotonic_intervals_of_f_g_minus_f_less_than_3_l511_511126

noncomputable def f (x : ℝ) : ℝ := -x * Real.log (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x - x

theorem monotonic_intervals_of_f :
  ∀ x : ℝ, x < -1 / Real.exp 1 → f x < f (-1 / Real.exp 1) ∧ x > -1 / Real.exp 1 → f x > f (-1 / Real.exp 1) := sorry

theorem g_minus_f_less_than_3 :
  ∀ x : ℝ, x < 0 → g x - f x < 3 := sorry

end monotonic_intervals_of_f_g_minus_f_less_than_3_l511_511126


namespace f_decreasing_on_positive_reals_f_greater_than_2_div_x_plus_2_l511_511147

noncomputable def f (x : ℝ) : ℝ := if x > 0 then (Real.log (1 + x)) / x else 0

theorem f_decreasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x :=
sorry

theorem f_greater_than_2_div_x_plus_2 :
  ∀ x : ℝ, 0 < x → f x > 2 / (x + 2) :=
sorry

end f_decreasing_on_positive_reals_f_greater_than_2_div_x_plus_2_l511_511147


namespace fiona_frog_probability_l511_511567

/--
  Fiona the frog starts on lily pad 0 in a row of lily pads numbered from 0 to 15. 
  There are predators on lily pads 4, 7, and 11. 
  A morsel of food is placed on lily pad 14. 
  Fiona can either hop to the next pad, jump two pads, or jump three pads forward, each with equal probability of 1/3.
  Prove that the probability that Fiona reaches pad 14 without landing on any of the predator pads is 10/6561.
-/
theorem fiona_frog_probability : 
  let probability_to_reach_14 : ℚ := 10 / 6561
  in fiona_reaches_14_safely (pads : finset ℕ) (start : ℕ) (predators : finset ℕ) (food : ℕ) :=
  start = 0 ∧ pads = (0 : ℕ) ... 15 ∧ 
  predators = {4, 7, 11} ∧ food = 14 ∧ 
  ∀ (k : ℕ), (k > 0 ∧ k < 16) → (hop : ℕ), (hop = 1 ∨ hop = 2 ∨ hop = 3) → (prob (k, hop) = 1 / 3) 
  → (fiona_reaches_food_without_predators = probability_to_reach_14)
sorry

end fiona_frog_probability_l511_511567


namespace V_lt_n_div_b_l511_511900

-- Define the function V
noncomputable def V (n b : ℕ) : ℕ :=
  if b >= n then 0 else
  if b <= 1 then n - 1
  else Sum (fun k : ℕ => if k > b then if n % k = 0 then V (n / k) b else 0 else 0) (range n)

-- The main theorem statement to prove
theorem V_lt_n_div_b (n b : ℕ) (hn : n > 0) (hb : b > 0) : V n b < n / b := sorry

end V_lt_n_div_b_l511_511900


namespace sum_of_differences_l511_511575

/-- 
  For each natural number from 1 to 999, Damir subtracts the last digit from the first digit and 
  writes the resulting differences on a board. We are to prove that the sum of all these differences 
  is 495.
-/
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, (first_digit n - last_digit n)) = 495 :=
sorry

/-- 
  Helper function to get the first digit of a natural number.
  Here, n > 0
-/
def first_digit (n : ℕ) : ℕ :=
  n / 10^(n.digits 10 - 1)

/-- 
  Helper function to get the last digit of a natural number.
  Here, n > 0
-/
def last_digit (n : ℕ) : ℕ :=
  n % 10

end sum_of_differences_l511_511575


namespace equidistant_points_l511_511989

theorem equidistant_points (a : ℝ) : 
  let A := (-2, 0)
      B := (4, a)
      dist := λ (p : ℝ × ℝ), |3 * p.1 - 4 * p.2 + 1| / (Real.sqrt (3^2 + (-4)^2))
  in dist A = 1 → dist B = 1 → (a = 2 ∨ a = 9 / 2) :=
by
  assume A := (-2, 0)
  assume B := (4, a)
  assume dist := λ (p : ℝ × ℝ), |3 * p.1 - 4 * p.2 + 1| / (Real.sqrt (3^2 + (-4)^2))
  intro h1
  intro h2
  sorry

end equidistant_points_l511_511989


namespace no_solution_for_equation_l511_511814

theorem no_solution_for_equation (x : ℝ) (h1 : x ≠ 1) : 
  (⊢ ¬ ∃ x, (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1) :=
by
  sorry

end no_solution_for_equation_l511_511814


namespace calculate_result_l511_511521

theorem calculate_result (x : ℝ) : (-x^3)^3 = -x^9 :=
by {
  sorry  -- Proof not required per instructions
}

end calculate_result_l511_511521


namespace min_value_expression_l511_511087

theorem min_value_expression :
  ∀ (F : Fin 2017 → (ℝ → ℝ)),
  (∀ i, ∀ t ∈ Icc (0 : ℝ) 1, F i t ∈ Icc (0 : ℝ) 1) →
  let expr := λ (x : Fin 2017 → ℝ), abs (∑ i, F i x i - ∏ i, x i)
  in (∃ x, ∀ i, x i ∈ Icc (0 : ℝ) 1 ∧ expr x = real.sqrt ( 6.4^2 ) ) :=
sorry

end min_value_expression_l511_511087


namespace relationship_between_sets_l511_511688

def M : Set ℤ := {x | ∃ k : ℤ, x = 5 * k - 2}
def P : Set ℤ := {x | ∃ n : ℤ, x = 5 * n + 3}
def S : Set ℤ := {x | ∃ m : ℤ, x = 10 * m + 3}

theorem relationship_between_sets : S ⊆ P ∧ P = M := by
  sorry

end relationship_between_sets_l511_511688


namespace square_of_volume_of_rect_box_l511_511664

theorem square_of_volume_of_rect_box (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 18) 
  (h3 : z * x = 10) : (x * y * z) ^ 2 = 2700 :=
sorry

end square_of_volume_of_rect_box_l511_511664


namespace find_m_l511_511976

def polynomial_eq (m x : ℝ) : Prop :=
  m^2 * x^4 + 3 * m * x^3 + 2 * x^2 + x = 1

theorem find_m :
  ∀ (m : ℝ), (\(x : ℝ) → polynomial_eq m x) ↔ m = 1 :=
sorry

end find_m_l511_511976


namespace max_C_trees_l511_511407

theorem max_C_trees 
  (price_A : ℕ) (price_B : ℕ) (price_C : ℕ) (total_price : ℕ) (total_trees : ℕ)
  (h_price_ratio : 2 * price_B = 2 * price_A ∧ 3 * price_A = 2 * price_C)
  (h_price_A : price_A = 200)
  (h_total_price : total_price = 220120)
  (h_total_trees : total_trees = 1000) :
  ∃ (num_C : ℕ), num_C = 201 ∧ ∀ num_C', num_C' > num_C → 
  total_price < price_A * (total_trees - num_C') + price_C * num_C' :=
by
  sorry

end max_C_trees_l511_511407


namespace hyperbola_eccentricity_l511_511312

theorem hyperbola_eccentricity 
  (a b e : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : e = Real.sqrt (1 + (b^2 / a^2))) 
  (h4 : e ≤ Real.sqrt 5) : 
  e = 2 := 
sorry

end hyperbola_eccentricity_l511_511312


namespace five_letter_words_with_vowels_l511_511218

/-
How many 5-letter words with at least one vowel can be constructed from the letters 
A, B, C, D, E, and F? (Note that A and E are vowels, any word is valid, not just English language 
words, and letters may be used more than once.)
-/

theorem five_letter_words_with_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := ['A', 'E'],
      consonants := ['B', 'C', 'D', 'F'] in
  let total_words := 6 ^ 5,
      consonant_only_words := 4 ^ 5,
      at_least_one_vowel_words := total_words - consonant_only_words in
  at_least_one_vowel_words = 6752 :=
by
  intro letters vowels consonants total_words consonant_only_words at_least_one_vowel_words
  sorry

end five_letter_words_with_vowels_l511_511218


namespace range_of_p_l511_511149

def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

def A : Set ℝ := {x | 3*x^2 - 2*x - 10 ≤ 0}

def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

theorem range_of_p (p : ℝ) (h : A ∪ B p = A) : p ≤ 3 :=
by
  sorry

end range_of_p_l511_511149


namespace largest_number_2_3_digits_l511_511441

noncomputable def largest_num_with_digits_sum_13 : ℕ :=
  let digits := [3, 3, 3, 3, 2] in
  digits.foldl (λ acc d, acc * 10 + d) 0

theorem largest_number_2_3_digits (n : ℕ) (digits : List ℕ) (h1 : ∀ d ∈ digits, d = 2 ∨ d = 3) (h2 : digits.sum = 13) :
  largest_num_with_digits_sum_13 = 33332 :=
by
  sorry

end largest_number_2_3_digits_l511_511441


namespace fold_paper_reflection_l511_511093

theorem fold_paper_reflection (m n : ℝ) :
  (∃ m n : ℝ, (0, 2) ↔ (4, 0) ∧ (7, 3) ↔ (m, n)) → (m - n) = -5.6 :=
sorry

end fold_paper_reflection_l511_511093


namespace passengers_off_nc_l511_511040

-- Define the initial number of passengers and the changes during the flight.
def initial_passengers := 124
def passengers_off_texas := 58
def passengers_on_texas := 24
def passengers_on_nc := 14
def total_people_landed_virginia := 67
def crew_members := 10

-- Define the theorem to prove the number of passengers who got off in North Carolina.
theorem passengers_off_nc : 
  let passengers_after_texas := initial_passengers - passengers_off_texas + passengers_on_texas in
  let passengers_after_nc := passengers_after_texas - ?x + passengers_on_nc in
  let passengers_landed_virginia := total_people_landed_virginia - crew_members in
  passengers_after_nc = passengers_landed_virginia → ?x = 47 :=
by
  sorry

end passengers_off_nc_l511_511040


namespace sin_cos_inequality_l511_511797

theorem sin_cos_inequality (n : ℕ) (x : ℝ) (h : 0 < n) : 
  sin^n (2 * x) + (sin^(n) x - cos^(n) x)^2 ≤ 1 :=
by sorry

end sin_cos_inequality_l511_511797


namespace matrix_solution_l511_511713

variable {x : ℝ}

theorem matrix_solution (x: ℝ) :
  let M := (3*x) * (2*x + 1) - (1) * (2*x)
  M = 5 → (x = 5/6) ∨ (x = -1) :=
by
  sorry

end matrix_solution_l511_511713


namespace largest_in_arithmetic_progression_l511_511361

theorem largest_in_arithmetic_progression (a d : ℝ) (n : ℕ) 
  (h1 : n = 7) 
  (seq : Fin 7 → ℝ) 
  (h_seq : ∀ i, seq i = a + (i - 3 : ℕ) * d)
  (h_sum_cubes : ∑ i in Finset.range 7, (seq ⟨i, Nat.lt_succ_self 6⟩) ^ 3 = 0)
  (h_sum_squares : ∑ i in Finset.range 7, (seq ⟨i, Nat.lt_succ_self 6⟩) ^ 2 = 756) :
  max (seq ⟨0, by norm_num⟩) (seq ⟨6, by norm_num⟩) = 9 * real.sqrt 3 := by
  sorry

end largest_in_arithmetic_progression_l511_511361


namespace sum_of_differences_l511_511580

open Nat
open BigOperators

theorem sum_of_differences (n : ℕ) (h : n ≥ 1 ∧ n ≤ 999) : 
  let differences := (fun x => 
                        let first_digit := x / 10;
                        let last_digit := x % 10;
                        first_digit - last_digit) in
  ∑ i in (finset.range (n + 1)).filter (λ x, 1 ≤ x ∧ x ≤ 999), differences i = 495 :=
by
  -- Acknowledge the need for a more refined filtering criteria for numbers between 1 and 999
  sorry

end sum_of_differences_l511_511580


namespace target_avg_weekly_income_l511_511920

-- Define the weekly incomes for the past 5 weeks
def past_incomes : List ℤ := [406, 413, 420, 436, 395]

-- Define the average income over the next 2 weeks
def avg_income_next_two_weeks : ℤ := 365

-- Define the target average weekly income over the 7-week period
theorem target_avg_weekly_income : 
  ((past_incomes.sum + 2 * avg_income_next_two_weeks) / 7 = 400) :=
sorry

end target_avg_weekly_income_l511_511920


namespace ordered_pairs_count_l511_511979

def count_ordered_pairs : ℕ :=
  ∑ a in range 1 51, Int.to_nat (↑((a + 1) / 2))

theorem ordered_pairs_count : count_ordered_pairs = 676 :=
  sorry

end ordered_pairs_count_l511_511979


namespace five_letter_words_with_vowel_l511_511191

theorem five_letter_words_with_vowel : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let n := 5 
  (∃ (w : list(char)), w.length = n ∧ ∀ (i : fin n), w[i] ∈ letters ∧ (∃ (j : fin n), w[j] ∈ vowels)) → 
  (6^5 - 4^5 = 6752) := 
by
  sorry

end five_letter_words_with_vowel_l511_511191


namespace squares_in_50th_ring_l511_511528

-- Define the problem using the given conditions
def centered_square_3x3 : ℕ := 3 -- Represent the 3x3 centered square

-- Define the function that computes the number of unit squares in the nth ring
def unit_squares_in_nth_ring (n : ℕ) : ℕ :=
  if n = 1 then 16
  else 24 + 8 * (n - 2)

-- Define the accumulation of unit squares up to the 50th ring
def total_squares_in_50th_ring : ℕ :=
  33 + 24 * 49

theorem squares_in_50th_ring : unit_squares_in_nth_ring 50 = 1209 :=
by
  -- Ensure that the correct value for the 50th ring can be verified
  sorry

end squares_in_50th_ring_l511_511528


namespace solution_set_l511_511714

-- Definitions based on conditions
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y
def f : ℝ → ℝ := sorry  -- placeholder for f

-- Conditions
axiom h1 : is_even f
axiom h2 : is_increasing_on f (set.Ioi 0)
axiom h3 : f (-3) = 0

-- Statement to prove
theorem solution_set :
  {x : ℝ | x * f x < 0} = set.Iio (-3) ∪ set.Ioo 0 3 :=
sorry

end solution_set_l511_511714


namespace real_z_imaginary_z_pure_imaginary_z_second_quadrant_z_l511_511987

noncomputable def z (m : ℝ) : ℂ :=
  (m^2 - m - 6) / (m + 3) + (m^2 + 5 * m + 6) * Complex.i

theorem real_z (m : ℝ) : z m ∈ ℝ ↔ m = -2 :=
sorry

theorem imaginary_z (m : ℝ) : (∀ x, z m ≠ x) ↔ (m ≠ -2 ∧ m ≠ -3) :=
sorry

theorem pure_imaginary_z (m : ℝ) : (↑(z m).re = 0 ∧ (↑(z m).im ≠ 0)) ↔ m = 3 :=
sorry

theorem second_quadrant_z (m : ℝ) : 
(↑(z m).re < 0 ∧ ↑(z m).im > 0) ↔ 
(m ∈ set.Ioo -∞ (-3) ∪ set.Ioo (-2) 3) :=
sorry

end real_z_imaginary_z_pure_imaginary_z_second_quadrant_z_l511_511987


namespace range_of_k_l511_511279

theorem range_of_k (k : ℝ) (hₖ : 0 < k) :
  (∃ x : ℝ, 1 = x^2 + (k^2 / x^2)) → 0 < k ∧ k ≤ 1 / 2 :=
by
  sorry

end range_of_k_l511_511279


namespace teaching_arrangements_36_l511_511865

theorem teaching_arrangements_36 :
  let classes := 4
  let teachers := 2
  (choose classes 2) * (choose classes 2) = 36 :=
by
  sorry

end teaching_arrangements_36_l511_511865


namespace correct_parametric_form_l511_511792

variable (t : ℝ)

def parametricA : ℝ × ℝ := (t^2, t^4)
def parametricB : ℝ × ℝ := (Real.sin t, Real.sin t ^ 2)
def parametricC : ℝ × ℝ := (Real.sqrt t, t)
def parametricD : ℝ × ℝ := (t, t^2)

theorem correct_parametric_form :
  (∀ t : ℝ, parametricD t = (t, t^2)) ∧
  (¬(∀ t : ℝ, parametricA t.1 = t^2 ∧ parametricA t.2 = (parametricA t.1)^2)) ∧
  (¬(∀ t : ℝ, parametricB t.1 = t ∧ parametricB t.2 = (parametricB t.1)^2)) ∧
  (¬(∀ t : ℝ, parametricC t.1 = t ∧ parametricC t.2 = (parametricC t.1)^2)) :=
by sorry

end correct_parametric_form_l511_511792


namespace solve_integers_l511_511813

theorem solve_integers (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x^(2 * y) + (x + 1)^(2 * y) = (x + 2)^(2 * y) → (x = 3 ∧ y = 1) :=
by
  sorry

end solve_integers_l511_511813


namespace sum_diff_1_to_999_l511_511635

def subtract_last_from_first (n : ℕ) : ℤ :=
  let str_n := n.toString
  if str_n.length = 1 then 0
  else
    let first_digit := str_n.toList.head!.digitToInt!
    let last_digit := str_n.toList.reverse.head!.digitToInt!
    first_digit - last_digit

def numbers : List ℕ := List.range 1000.tail

def sum_of_differences : ℤ := (numbers.map subtract_last_from_first).sum

theorem sum_diff_1_to_999 :
  sum_of_differences = 495 := 
sorry

end sum_diff_1_to_999_l511_511635


namespace calc_fraction_cube_l511_511526

theorem calc_fraction_cube : (88888 ^ 3 / 22222 ^ 3) = 64 := by 
    sorry

end calc_fraction_cube_l511_511526


namespace sum_of_differences_l511_511601

theorem sum_of_differences : 
  let first_digit (n : ℕ) : ℕ := n / 10^(nat.log10 n)
  let last_digit (n : ℕ) : ℕ := n % 10
  (finset.range 1000).sum (λ n, first_digit n - last_digit n) = 495 :=
by
  sorry

end sum_of_differences_l511_511601


namespace number_of_sampled_in_interval_l511_511905

theorem number_of_sampled_in_interval (total_staff : ℕ) (number_sampled : ℕ) (interval_start : ℕ) (interval_end : ℕ) :
  total_staff = 840 ∧ number_sampled = 42 ∧ interval_start = 61 ∧ interval_end = 120 →
  let k := total_staff / number_sampled in
  let sampled_numbers := (list.range (total_staff / k)).map (λ i, k * (i + 1)) in
  (sampled_numbers.filter (λ n, interval_start ≤ n ∧ n ≤ interval_end)).length = 3 := by
  intros h
  rcases h with ⟨rfl, rfl, rfl, rfl⟩
  let k := 840 / 42
  have hk : k = 20 := rfl
  let sampled_numbers := (list.range (840 / k)).map (λ i, k * (i + 1))
  have hs : sampled_numbers = list.map (λ i, 20 * (i + 1)) (list.range 42) := by
    rw hk
    rfl
  have hfilter : (sampled_numbers.filter (λ n, 61 ≤ n ∧ n ≤ 120)).length = 3 := by
    rw hs
    norm_num
    sorry
  exact hfilter

end number_of_sampled_in_interval_l511_511905


namespace five_letter_words_with_at_least_one_vowel_l511_511214

open Finset

theorem five_letter_words_with_at_least_one_vowel :
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := ((letters \ vowels).card : ℕ) ^ 5
  total_words - no_vowel_words = 6752 :=
by
  let letters := insert 'A' (insert 'B' (insert 'C' (insert 'D' (insert 'E' (singleton 'F')))))
  let vowels := insert 'A' (singleton 'E')
  let consonants := letters \ vowels
  have : letters.card = 6 := by simp
  have : vowels.card = 2 := by simp
  have : consonants.card = 4 := by simp
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := (consonants.card : ℕ) ^ 5
  calc
    total_words - no_vowel_words
        = 6^5 - 4^5         : by simp
    ... = 7776 - 1024       : by norm_num
    ... = 6752             : by norm_num

end five_letter_words_with_at_least_one_vowel_l511_511214


namespace trapezoid_concurrent_or_parallel_l511_511334

variables {Point : Type} [EuclideanGeometry Point]

noncomputable def midpoint (A B : Point) : Point := sorry

noncomputable def line_through (P Q : Point) : Set Point := sorry

theorem trapezoid_concurrent_or_parallel
  (A B C D M N I : Point)
  (h_trapezoid : line_through A B ∥ line_through C D)
  (h_M : M = midpoint A B)
  (h_N : N = midpoint C D)
  (h_I : ∃ I, I ∈ (line_through A D ∩ line_through B C)) :
  (line_through M N = line_through A D ∩ line_through B C) ∨ (line_through M N ∥ line_through A D ∧ line_through M N ∥ line_through B C) :=
sorry

end trapezoid_concurrent_or_parallel_l511_511334


namespace power_of_two_l511_511877

theorem power_of_two (n : ℕ) (h : 2^n = 32 * (1 / 2) ^ 2) : n = 3 :=
by {
  sorry
}

end power_of_two_l511_511877


namespace find_phi_l511_511154

def f (x : Real) (φ : Real) : Real := Real.sin (2 * x + φ)

theorem find_phi (φ : Real) (h1 : 0 < φ) (h2 : φ ≤ Real.pi)
  (hsym : ∃ k : ℤ, 2 * (Real.pi / 8) + φ = k * Real.pi + Real.pi / 2) : 
  φ = Real.pi / 4 :=
sorry

end find_phi_l511_511154


namespace total_fare_for_150_km_l511_511941

-- Conditions
def initial_distance : ℝ := 10
def initial_fare : ℝ := 90
def known_distance : ℕ := 100
def known_fare : ℝ := 150
def additional_distance_traveled (distance : ℕ) : ℝ := max 0 (distance - initial_distance)

-- Definitions based on conditions
def excess_distance := additional_distance_traveled known_distance
def excess_fare := known_fare - initial_fare
def cost_per_km := excess_fare / excess_distance

/-- Theorem: The total fare for traveling a given distance based on the given rate. -/
theorem total_fare_for_150_km : 
  let travel_distance : ℕ := 150
      excess_distance := additional_distance_traveled travel_distance
      additional_fare := excess_distance * cost_per_km
      total_fare := initial_fare + additional_fare in
  total_fare = 183.33 :=
by
  sorry

end total_fare_for_150_km_l511_511941


namespace part1_monotonicity_when_a_eq_1_part2_range_of_a_l511_511672

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x - 1) * (Real.log (x - 2)) - a * (x - 3)

theorem part1_monotonicity_when_a_eq_1 :
  ∀ x, 2 < x → ∀ x1, (2 < x1 → f x 1 ≤ f x1 1) := by
  sorry

theorem part2_range_of_a :
  ∀ a, (∀ x, 3 < x → f x a > 0) → a ≤ 2 := by
  sorry

end part1_monotonicity_when_a_eq_1_part2_range_of_a_l511_511672


namespace area_of_triangle_l511_511542

/-
   Prove that the area of a triangle with vertices
   A(-2, 3), B(6, -1), and C(12, 6) is 40.
-/

def point := ℝ × ℝ

def A : point := (-2, 3)
def B : point := (6, -1)
def C : point := (12, 6)

def vector_sub (p1 p2 : point) : point :=
  (p1.1 - p2.1, p1.2 - p2.2)

def parallelogram_area (v w : point) : ℝ :=
  real.abs (v.1 * w.2 - v.2 * w.1)

def triangle_area (A B C : point) : ℝ :=
  parallelogram_area (vector_sub C A) (vector_sub C B) / 2

theorem area_of_triangle : triangle_area A B C = 40 :=
by sorry

end area_of_triangle_l511_511542


namespace sum_of_differences_l511_511624

theorem sum_of_differences : 
  (∑ n in Finset.range 1000, let first_digit := n / 10 ^ (Nat.log10 n) in
                             let last_digit := n % 10 in
                             first_digit - last_digit) = 495 :=
by
  sorry

end sum_of_differences_l511_511624


namespace intersection_points_C1_C2_max_triangle_area_AOB_l511_511734

-- Define the curves and points in Lean
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 2 * x

-- Problem (I) : Intersection points of C1 and C2
theorem intersection_points_C1_C2 : 
  (C1 (1/2) (Real.sqrt 3 / 2) ∧ C2 (1/2) (Real.sqrt 3 / 2)) ∧ 
  (C1 (1/2) (-Real.sqrt 3 / 2) ∧ C2 (1/2) (-Real.sqrt 3 / 2)) :=
sorry

-- Problem (II) : Maximum area of triangle AOB
def A := (4, Real.pi / 3)
def ρ (θ : ℝ) : ℝ := 2 * Real.cos θ

def triangle_area_AOB (θ : ℝ) : ℝ := 
  |4 * ρ(θ) * Real.sin((Real.pi / 3) - θ)|

theorem max_triangle_area_AOB : ∃ θ, triangle_area_AOB θ = 2 + Real.sqrt 3 :=
sorry

end intersection_points_C1_C2_max_triangle_area_AOB_l511_511734


namespace trigonometric_identity_solution_l511_511640

theorem trigonometric_identity_solution 
  (alpha beta : ℝ)
  (h1 : π / 4 < alpha)
  (h2 : alpha < 3 * π / 4)
  (h3 : 0 < beta)
  (h4 : beta < π / 4)
  (h5 : Real.cos (π / 4 + alpha) = -4 / 5)
  (h6 : Real.sin (3 * π / 4 + beta) = 12 / 13) :
  (Real.sin (alpha + beta) = 63 / 65) ∧
  (Real.cos (alpha - beta) = -33 / 65) :=
by
  sorry

end trigonometric_identity_solution_l511_511640


namespace initial_percentage_increase_l511_511494

theorem initial_percentage_increase
  (x : ℝ) 
  (h1 : ∀ P : ℝ, P > 0 → P * (1 + x/100) * 0.9 * 0.85 = P * (1 + 0.002150000000000034))
  : x ≈ 31.02 := 
by 
  sorry

end initial_percentage_increase_l511_511494


namespace angle_equality_l511_511328

open EuclideanGeometry

variables {A B C D E F : Point}
variables {α β γ δ : ℝ}

-- Given conditions:
axiom collinear : Collinear E B C F
axiom angle_BAE_CDF : ∠ B A E = ∠ C D F
axiom angle_EAF_FDE : ∠ E A F = ∠ F D E

-- Goal:
theorem angle_equality : ∠ F A C = ∠ E D B :=
sorry

end angle_equality_l511_511328


namespace total_number_of_balls_l511_511464

theorem total_number_of_balls (white green yellow red purple : Nat) (P : ℚ) 
  (h_white : white = 20)
  (h_green : green = 30)
  (h_yellow : yellow = 10)
  (h_red : red = 37)
  (h_purple : purple = 3)
  (h_P : P = 0.6) :
  let total := white + green + yellow + red + purple,
      red_purple := red + purple,
      prob_red_purple := (red_purple : ℚ) / (total: ℚ) in
  P = 1 - prob_red_purple → total = 100 := 
by
  intros; sorry

end total_number_of_balls_l511_511464


namespace largest_five_digit_number_with_product_120_l511_511556

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def prod_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldr (· * ·) 1

def max_five_digit_prod_120 : ℕ := 85311

theorem largest_five_digit_number_with_product_120 :
  is_five_digit max_five_digit_prod_120 ∧ prod_of_digits max_five_digit_prod_120 = 120 :=
by
  sorry

end largest_five_digit_number_with_product_120_l511_511556


namespace total_cost_kept_l511_511694

def prices_all : List ℕ := [15, 18, 20, 15, 25, 30, 20, 17, 22, 23, 29]
def prices_returned : List ℕ := [20, 25, 30, 22, 23, 29]

def total_cost (prices : List ℕ) : ℕ :=
  prices.foldl (· + ·) 0

theorem total_cost_kept :
  total_cost prices_all - total_cost prices_returned = 85 :=
by
  -- The proof steps go here
  sorry

end total_cost_kept_l511_511694


namespace length_of_platform_l511_511014

noncomputable def speedInMetersPerSecond (v : ℕ) : ℚ :=
  (v : ℚ) * (5 / 18)

theorem length_of_platform (v : ℕ) (t : ℕ) (l_train : ℕ) :
  v = 72 → t = 26 → l_train = 300 →
  let distance := speedInMetersPerSecond v * t in
  let l_platform := distance - l_train in
  l_platform = 220 :=
by
  intros hv ht htrain
  let distance := speedInMetersPerSecond v * t
  let l_platform := distance - l_train
  sorry

end length_of_platform_l511_511014


namespace sweet_numbers_count_l511_511881

def generate_next (n : ℕ) : ℕ :=
  if n <= 20 then 2 * n else n - 10

def generates_sequence (start : ℕ) (seq : List ℕ) : Prop :=
  seq.length > 0 ∧ seq.head = start ∧ 
  ∀ i, i < seq.length - 1 → seq.get! (i + 1) = generate_next (seq.get! i)

def is_sweet_number (F : ℕ) : Prop :=
  ∀ seq, generates_sequence F seq → ¬ (18 ∈ seq)

def count_sweet_numbers_in_range (range : List ℕ) : ℕ :=
  range.countp is_sweet_number

theorem sweet_numbers_count :
  count_sweet_numbers_in_range (List.range' 1 50) = 15 :=
sorry

end sweet_numbers_count_l511_511881


namespace parabola_height_at_5_l511_511485

theorem parabola_height_at_5 (h : ℝ) (span : ℝ) (a k : ℝ) :
  h = 16 ∧ span = 40 ∧ k = 16 ∧ a = -1/25 → 
  (32 - 2) / 2 = 15 := 
by 
  intros,
  delta at *,
  linarith,
sory

end parabola_height_at_5_l511_511485


namespace fair_distribution_l511_511866

theorem fair_distribution (A_wins : ℕ) (B_wins : ℕ) (prize : ℕ) 
  (h1 : 3 ≤ A_wins + B_wins) (h2 : A_wins + B_wins ≤ 7) (h3 : A_wins = 3) (h4 : B_wins = 2) :
  (A_share : ℕ) * (B_share : ℕ) :=
  ∃ A_share B_share,
    A_share = 7500 ∧ B_share = 2500 ∧
    A_share + B_share = prize ∧
    prize = 10000 := sorry

end fair_distribution_l511_511866


namespace relationship_among_values_l511_511674

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

lemma f_is_even : ∀ x, f (-x) = f x := by
  intro x
  change (-x) * Real.sin (-x) = x * Real.sin x
  rw [Real.sin_neg, neg_mul_neg]

lemma f_increasing_in_interval : ∀ x ∈ Ioo 0 (π / 2), deriv f x > 0 := by
  intro x hx
  simp [f]
  apply (Real.add_pos (Real.sin_pos_of_mem_Ioo hx) _)
  apply lt_of_lt_of_le (Real.mul_neg Lé(.sin_pos_of_mem_Algo <_ lemma x hx) 
  apply le_of_lt (hx.2)

theorem relationship_among_values :
  f (-π / 3) > f (-1) ∧ f (-1) > f (π / 11) := by
  have h0 : f (π / 3) = f (-π / 3), from eq.symm (f_is_even _)
  have h1 : f 1 = f (-1),             from eq.symm (f_is_even _)
  have hh := f_increasing_in_interval
  have h3 : f (π / 11) < f 1, from (hh _ ⟨by linarith, by linarith⟩)
  have h4 : f 1 < f (π / 3),  from (hh _ ⟨by linarith, by linarith⟩)
  exact ⟨ by linarith [h0, h1, h4, h3], by linarith [h1, h3]⟩
  sorry

end relationship_among_values_l511_511674


namespace five_letter_words_with_vowel_l511_511250

theorem five_letter_words_with_vowel :
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  total_words - words_no_vowel = 6752 :=
by
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  have h1 : total_words = 7776 := by norm_num
  have h2 : words_no_vowel = 1024 := by norm_num
  show total_words - words_no_vowel = 6752
  calc
    total_words - words_no_vowel
    = 7776 - 1024 : by rw [h1, h2]
    ... = 6752 : by norm_num

end five_letter_words_with_vowel_l511_511250


namespace sum_diff_1_to_999_l511_511633

def subtract_last_from_first (n : ℕ) : ℤ :=
  let str_n := n.toString
  if str_n.length = 1 then 0
  else
    let first_digit := str_n.toList.head!.digitToInt!
    let last_digit := str_n.toList.reverse.head!.digitToInt!
    first_digit - last_digit

def numbers : List ℕ := List.range 1000.tail

def sum_of_differences : ℤ := (numbers.map subtract_last_from_first).sum

theorem sum_diff_1_to_999 :
  sum_of_differences = 495 := 
sorry

end sum_diff_1_to_999_l511_511633


namespace back_seat_tickets_sold_l511_511909

def total_tickets : ℕ := 20000
def main_seat_price : ℕ := 55
def back_seat_price : ℕ := 45
def total_revenue : ℕ := 955000

theorem back_seat_tickets_sold :
  ∃ (M B : ℕ), 
    M + B = total_tickets ∧ 
    main_seat_price * M + back_seat_price * B = total_revenue ∧ 
    B = 14500 :=
by
  sorry

end back_seat_tickets_sold_l511_511909


namespace ham_division_l511_511017

theorem ham_division (v1 v2 v3 : ℕ) (h : v1 + v2 + v3 = 45)
  (store_scale_values : set ℕ) (hs : store_scale_values = {14, 15, 16})
  (third_choice : set ℕ) (tch : third_choice = {16})
  (second_choice : set ℕ) (sch : second_choice = {15})
  (first_choice : set ℕ) (fch : first_choice = {14}) :
  ∃ (a b c : ℕ), (a ∈ third_choice) ∧ (b ∈ second_choice) ∧ (c ∈ first_choice) ∧ (a + b + c = 45) :=
by {
  use [16, 15, 14],
  simp [tch, sch, fch],
  exact h,
  sorry
}

end ham_division_l511_511017


namespace phase_shift_of_given_function_l511_511562

def given_function (x : ℝ) : ℝ := 5 * sin (x - 2 * Real.pi / 3)

theorem phase_shift_of_given_function : ∃ C : ℝ, C = 2 * Real.pi / 3 ∧
  ∀ x : ℝ, given_function x = 5 * sin (x - C) := by
  existsi 2 * Real.pi / 3
  split
  · refl
  · simp [given_function]
  sorry

end phase_shift_of_given_function_l511_511562


namespace exists_monochromatic_triangle_l511_511062

theorem exists_monochromatic_triangle :
  ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i j k : Fin 6, i ≠ j → j ≠ k → i ≠ k → ¬Collinear ℝ
    ![points i, points j, points k]) →
  ∃ (coloring : (Fin 6 × Fin 6) → Prop), 
  (∀ i j : Fin 6, i ≠ j → (coloring (i, j) ∨ ¬coloring (i, j))) →
  ∃ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
  ((coloring (i, j) ∧ coloring (j, k) ∧ coloring (i, k)) ∨ 
  (¬coloring (i, j) ∧ ¬coloring (j, k) ∧ ¬coloring (i, k))) :=
by
  sorry

end exists_monochromatic_triangle_l511_511062


namespace five_letter_words_with_vowels_l511_511224

/-
How many 5-letter words with at least one vowel can be constructed from the letters 
A, B, C, D, E, and F? (Note that A and E are vowels, any word is valid, not just English language 
words, and letters may be used more than once.)
-/

theorem five_letter_words_with_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := ['A', 'E'],
      consonants := ['B', 'C', 'D', 'F'] in
  let total_words := 6 ^ 5,
      consonant_only_words := 4 ^ 5,
      at_least_one_vowel_words := total_words - consonant_only_words in
  at_least_one_vowel_words = 6752 :=
by
  intro letters vowels consonants total_words consonant_only_words at_least_one_vowel_words
  sorry

end five_letter_words_with_vowels_l511_511224


namespace rationalize_sqrt_5_div_18_l511_511803

theorem rationalize_sqrt_5_div_18 :
  (Real.sqrt (5 / 18) = Real.sqrt 10 / 6) :=
sorry

end rationalize_sqrt_5_div_18_l511_511803


namespace balls_into_boxes_l511_511700

theorem balls_into_boxes :
  (number_of_ways_to_distribute 7 4) = 128 := sorry

end balls_into_boxes_l511_511700


namespace noah_yearly_bill_l511_511779

-- Define the length of each call in minutes
def call_duration : ℕ := 30

-- Define the cost per minute in dollars
def cost_per_minute : ℝ := 0.05

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Define the cost per call in dollars
def cost_per_call : ℝ := call_duration * cost_per_minute

-- Define the total cost for a year in dollars
def yearly_cost : ℝ := cost_per_call * weeks_in_year

-- State the theorem
theorem noah_yearly_bill : yearly_cost = 78 := by
  -- Proof follows here
  sorry

end noah_yearly_bill_l511_511779


namespace value_of_k_l511_511139

theorem value_of_k (k : ℤ) (h : (∀ x : ℤ, (x^2 - k * x - 6) = (x - 2) * (x + 3))) : k = -1 := by
  sorry

end value_of_k_l511_511139


namespace lucy_share_l511_511778

theorem lucy_share (total : ℝ) (natalie_share : ℝ) (rick_share : ℝ) (lucy_share : ℝ) (h1 : total = 10000) (h2 : natalie_share = total * 0.5) (h3 : rick_share = (total - natalie_share) * 0.6) (h4 : lucy_share = total - natalie_share - rick_share) : lucy_share = 2000 := by
sorr

end lucy_share_l511_511778


namespace equal_areas_of_shapes_l511_511457

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def semicircle_area (r : ℝ) : ℝ :=
  (Real.pi * r^2) / 2

noncomputable def sector_area (theta : ℝ) (r : ℝ) : ℝ :=
  (theta / (2 * Real.pi)) * Real.pi * r^2

noncomputable def shape1_area (s : ℝ) : ℝ :=
  hexagon_area s + 3 * semicircle_area (s / 4) - 6 * sector_area (Real.pi / 3) (s / 4)

noncomputable def shape2_area (s : ℝ) : ℝ :=
  hexagon_area s + 6 * sector_area (2 * Real.pi / 3) (s / 4) - 3 * semicircle_area (s / 4)

theorem equal_areas_of_shapes (s : ℝ) : shape1_area s = shape2_area s :=
by {
  sorry
}

end equal_areas_of_shapes_l511_511457


namespace conclusions_correct_l511_511667

theorem conclusions_correct : 
  (∀ a b : ℝ, (0 ≤ a ∧ a ≤ 1) ∧ (0 ≤ b ∧ b ≤ 1) → 
    (probability_of_event (λ _, a^2 + b^2 ≤ 1) = π / 4)) ∧
  (∀ ξ : ℝ → ℝ, (is_normal ξ 3 σ^2) ∧ (probability (λ x, ξ x ≤ 5) = m) → 
    (probability (λ x, ξ x ≤ 1) = 1 - m)) ∧
  ((term_of_expansion (sqrt x + 1 / (2 * sqrt x))^8 4 = 35 / 8) →
    (number_of_correct_conclusions = 3)) :=
sorry

end conclusions_correct_l511_511667


namespace multiplication_of_powers_of_10_l511_511443

theorem multiplication_of_powers_of_10 : (10 : ℝ) ^ 65 * (10 : ℝ) ^ 64 = (10 : ℝ) ^ 129 := by
  sorry

end multiplication_of_powers_of_10_l511_511443


namespace rationalize_denominator_l511_511806

-- Lean 4 statement
theorem rationalize_denominator : sqrt (5 / 18) = sqrt 10 / 6 := by
  sorry

end rationalize_denominator_l511_511806


namespace tod_driving_time_l511_511412
noncomputable def total_driving_time (distance_north distance_west speed : ℕ) : ℕ :=
  (distance_north + distance_west) / speed

theorem tod_driving_time :
  total_driving_time 55 95 25 = 6 :=
by
  sorry

end tod_driving_time_l511_511412


namespace find_n_l511_511982

theorem find_n (n : ℕ) (a : ℕ → ℝ) 
  (h1 : ∀ i, a i > 0) 
  (h2 : ∑ i in finset.range n, a i = 17):
  (∃ (n : ℕ), 
  ∃ m : ℤ, 
  m = ∑ k in finset.range(n), real.sqrt ((2*k-1)^2 + (a k)^2) ∧ 
  m - (n^2) = 1 ∧ 
  m + (n^2) = 289) → 
  n = 12 :=
by sorry

end find_n_l511_511982


namespace cylinder_volume_factor_l511_511448

theorem cylinder_volume_factor (h r : ℝ) :
  let V := π * r^2 * h in
  let V_prime := π * (2.5 * r)^2 * (3 * h) in
  V_prime = 18.75 * V :=
by
  sorry

end cylinder_volume_factor_l511_511448


namespace negation_of_no_honors_students_attend_school_l511_511830

-- Definitions (conditions and question)
def honors_student (x : Type) : Prop := sorry -- The condition defining an honors student
def attends_school (x : Type) : Prop := sorry -- The condition defining a student attending the school

-- The theorem statement
theorem negation_of_no_honors_students_attend_school :
  (¬ ∃ x : Type, honors_student x ∧ attends_school x) ↔ (∃ x : Type, honors_student x ∧ attends_school x) :=
sorry

end negation_of_no_honors_students_attend_school_l511_511830


namespace price_per_acre_is_1863_l511_511011

-- Define the conditions
def totalAcres : ℕ := 4
def numLots : ℕ := 9
def pricePerLot : ℤ := 828
def totalRevenue : ℤ := numLots * pricePerLot
def totalCost (P : ℤ) : ℤ := totalAcres * P

-- The proof problem: Prove that the price per acre P is 1863
theorem price_per_acre_is_1863 (P : ℤ) (h : totalCost P = totalRevenue) : P = 1863 :=
by
  sorry

end price_per_acre_is_1863_l511_511011


namespace point_in_first_quadrant_l511_511908

def z : ℂ := 2 + complex.I

theorem point_in_first_quadrant (z : ℂ) (hz : z = 2 + complex.I) : 
  (complex.re z > 0) ∧ (complex.im z > 0) := by
  sorry

end point_in_first_quadrant_l511_511908


namespace new_average_of_modified_consecutive_integers_l511_511368

theorem new_average_of_modified_consecutive_integers (x : ℤ) 
  (h : (x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6) + (x + 7) + (x + 8) + (x + 9)) = 110) : 
  (x - 9 + (x + 1 - 8) + (x + 2 - 7) + (x + 3 - 6) + (x + 4 - 5) + (x + 5 - 4) + (x + 6 - 3) + (x + 7 - 2) + (x + 8 - 1) + (x + 9 - 0)) / 10 = 6.5 :=
by
  sorry

end new_average_of_modified_consecutive_integers_l511_511368


namespace doris_saturday_hours_l511_511074

variable (hours_per_day_weekday : ℕ) 
variable (weekday_days_per_week : ℕ) 
variable (hourly_wage : ℕ) 
variable (monthly_expense : ℕ)
variable (weeks_to_earn_expense : ℕ)
variable (saturdays_in_weeks : ℕ)

theorem doris_saturday_hours 
    (hpdw := 3) 
    (wdpw := 5) 
    (hw := 20)
    (me := 1200) 
    (wtax := 3) 
    (siw := 3) : 
    (me - wtax * hw * (hpdw * wdpw)) / hw / siw = 5 := 
by
  have h1 : 3 * 5 = 15 := by norm_num
  have h2 : 15 * 20 = 300 := by norm_num
  have h3 : 3 * 300 = 900 := by norm_num
  have h4 : 1200 - 900 = 300 := by norm_num
  have h5 : 300 / 20 = 15 := by norm_num
  have h6 : 15 / 3 = 5 := by norm_num
  exact h6


end doris_saturday_hours_l511_511074


namespace min_value_f_at_0_l511_511645

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem min_value_f_at_0 (a : ℝ) : (∀ x : ℝ, f a 0 ≤ f a x) ↔ 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end min_value_f_at_0_l511_511645


namespace five_letter_words_with_vowel_l511_511249

theorem five_letter_words_with_vowel :
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  total_words - words_no_vowel = 6752 :=
by
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  have h1 : total_words = 7776 := by norm_num
  have h2 : words_no_vowel = 1024 := by norm_num
  show total_words - words_no_vowel = 6752
  calc
    total_words - words_no_vowel
    = 7776 - 1024 : by rw [h1, h2]
    ... = 6752 : by norm_num

end five_letter_words_with_vowel_l511_511249


namespace sum_squares_of_six_consecutive_even_eq_1420_l511_511456

theorem sum_squares_of_six_consecutive_even_eq_1420 
  (n : ℤ) 
  (h : n + (n + 2) + (n + 4) + (n + 6) + (n + 8) + (n + 10) = 90) :
  n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 + (n + 8)^2 + (n + 10)^2 = 1420 :=
by
  sorry

end sum_squares_of_six_consecutive_even_eq_1420_l511_511456


namespace spinner_win_sector_area_l511_511467

open Real

theorem spinner_win_sector_area (r : ℝ) (P : ℝ)
  (h_r : r = 8) (h_P : P = 3 / 7) : 
  ∃ A : ℝ, A = 192 * π / 7 :=
by
  sorry

end spinner_win_sector_area_l511_511467


namespace projection_of_a_on_b_l511_511167

namespace ProjectionProof

variables {V : Type*} [inner_product_space ℝ V]

/-- Given vectors a and b with the conditions |a| = 2, |b| = 1 and |a - 4b| = 2√7,
we need to prove that the projection of a on b is -1. -/
theorem projection_of_a_on_b (a b : V) (ha : ‖a‖ = 2) (hb : ‖b‖ = 1) (h : ‖a - 4 • b‖ = 2 * sqrt 7) :
  (inner_product_space.proj b a) = -1 :=
by sorry

end ProjectionProof

end projection_of_a_on_b_l511_511167


namespace convex_pentagon_probability_l511_511970

-- Defining the number of chords and the probability calculation as per the problem's conditions
def number_of_chords (n : ℕ) : ℕ := (n * (n - 1)) / 2
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Problem conditions
def eight_points_on_circle : ℕ := 8
def chords_chosen : ℕ := 5

-- Total number of chords from eight points
def total_chords : ℕ := number_of_chords eight_points_on_circle

-- The probability calculation
def probability_convex_pentagon :=
  binom 8 5 / binom total_chords chords_chosen

-- Statement to be proven
theorem convex_pentagon_probability :
  probability_convex_pentagon = 1 / 1755 := sorry

end convex_pentagon_probability_l511_511970


namespace value_of_f_nine_halves_l511_511659

noncomputable def f : ℝ → ℝ := sorry  -- Define f with noncomputable since it's not explicitly given

axiom even_function (x : ℝ) : f x = f (-x)  -- Define the even function property
axiom not_identically_zero : ∃ x : ℝ, f x ≠ 0 -- Define the property that f is not identically zero
axiom functional_equation (x : ℝ) : x * f (x + 1) = (x + 1) * f x -- Define the given functional equation

theorem value_of_f_nine_halves : f (9 / 2) = 0 := by
  sorry

end value_of_f_nine_halves_l511_511659


namespace min_chips_to_color_all_cells_l511_511033

def min_chips_needed (n : ℕ) : ℕ := n

theorem min_chips_to_color_all_cells (n : ℕ) :
  min_chips_needed n = n :=
sorry

end min_chips_to_color_all_cells_l511_511033


namespace sum_of_differences_l511_511618

/-- Proving that the sum of all differences (first digit - last digit) for natural numbers from 
    1 to 999 is 495. -/ 
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, 
    let str := n.toString
    let first := if str.length > 1 then str.head!.toNat - '0'.toNat else 0
    let last := if str.length > 1 then str.getLast.toNat - '0'.toNat else 0
    first - last
  ) = 495 := 
by
  sorry

end sum_of_differences_l511_511618


namespace noah_yearly_call_cost_l511_511786

structure CallBilling (minutes_per_call : ℕ) (charge_per_minute : ℝ) (calls_per_week : ℕ) (weeks_in_year : ℕ) :=
  (total_minutes : ℕ := weeks_in_year * calls_per_week * minutes_per_call)
  (total_cost : ℝ := total_minutes * charge_per_minute)

theorem noah_yearly_call_cost :
  CallBilling 30 0.05 1 52 .total_cost = 78 := by
  sorry

end noah_yearly_call_cost_l511_511786


namespace parabola_focus_l511_511823

theorem parabola_focus : 
  ∃ p, ∀ x y, (y = 9 * x^2) → (p = (1 : ℝ) / 18) ∧ (y = 2 * p * x^2) ∧ (focus_coordinates = (0, p / 2)) → (focus_coordinates = (0, (1 : ℝ) / 36)) :=
by
  exists (1 / 18 : ℝ)
  intros x y hy hp hfocus
  have h_p : p = 1 / 18 := by rwa [hfocus.1]
  rw [h_p] at hfocus
  exact ⟨hfocus.1, hfocus.2.1, by simp [hfocus.2.2]⟩

end parabola_focus_l511_511823


namespace sum_diff_1_to_999_l511_511630

def subtract_last_from_first (n : ℕ) : ℤ :=
  let str_n := n.toString
  if str_n.length = 1 then 0
  else
    let first_digit := str_n.toList.head!.digitToInt!
    let last_digit := str_n.toList.reverse.head!.digitToInt!
    first_digit - last_digit

def numbers : List ℕ := List.range 1000.tail

def sum_of_differences : ℤ := (numbers.map subtract_last_from_first).sum

theorem sum_diff_1_to_999 :
  sum_of_differences = 495 := 
sorry

end sum_diff_1_to_999_l511_511630


namespace part1_constant_value_l511_511462

open Real

variables (A B O : Point)
hypothesis ellipse_A : (A.x ^ 2 / 9) + (A.y ^ 2 / 4) = 1
hypothesis ellipse_B : (B.x ^ 2 / 9) + (B.y ^ 2 / 4) = 1
hypothesis dot_product_zero : (A x - O x) * (B x - O x) + (A y - O y) * (B y - O y) = 0

theorem part1_constant_value : 
  1 / (A x - O x)^2 + (A y - O y)^2 + 1 / (B x - O x)^2 + (B y - O y)^2 = 13/36 := 
sorry

end part1_constant_value_l511_511462


namespace hyperbola_no_common_points_l511_511321

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) : ℝ :=
  real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_no_common_points (a b e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_ecc : e = real.sqrt (1 + (b^2 / a^2)))
  (h_slope : b / a < 2) :
  e = 2 :=
sorry

end hyperbola_no_common_points_l511_511321


namespace smallest_F_for_beautiful_number_pairs_l511_511986

def F (m n : ℕ) : ℕ :=
(m / 10 + n % 10) * 11 + (m % 10 + n / 10) * 9

def isBeautifulNumberPair (x y : ℕ) : Prop :=
1 ≤ x ∧ x ≤ 4 ∧ 1 ≤ y ∧ y ≤ 5 ∧
let p := 21 * x + y in
let q := 52 + y in
let p' := 10 * y + x in
(16 * y + x) % 13 = 0

theorem smallest_F_for_beautiful_number_pairs :
  ∃ (p q : ℕ), (∃ (x y : ℕ), isBeautifulNumberPair x y ∧ p = 21 * x + y ∧ q = 52 + y) ∧ F p q = 156 :=
begin
  sorry
end

end smallest_F_for_beautiful_number_pairs_l511_511986


namespace cylinder_surface_area_proof_l511_511912

noncomputable def cylinder := 
  { radius : ℝ := 12, 
    height : ℝ := 21, 
    top_face_coverage : ℝ := 0.65, 
    bottom_face_coverage : ℝ := 0.45 }

theorem cylinder_surface_area_proof 
  (r h : ℝ) (top_cover bottom_cover : ℝ) 
  (h_radius : r = 12) 
  (h_height : h = 21) 
  (h_top_cover : top_cover = 0.65)
  (h_bottom_cover : bottom_cover = 0.45) : 
  let LSA := 2 * π * r * h,
      top_area := π * r^2,
      total_top_bottom_area := 2 * top_area,
      TSA := LSA + total_top_bottom_area,
      material_A_area := top_cover * top_area,
      material_B_area := bottom_cover * top_area
  in LSA =  504 * π ∧ 
     TSA = 792 * π ∧ 
     material_A_area = 93.6 * π ∧ 
     material_B_area = 64.8 * π := 
by {
  intros,
  simp [LSA, top_area, total_top_bottom_area, TSA, material_A_area, material_B_area],
  split; norm_num,
  all_goals {
    simp [h_radius, h_height, h_top_cover, h_bottom_cover],
    norm_num,
  },
}

end cylinder_surface_area_proof_l511_511912


namespace length_n_words_no_adjacent_a_l511_511162

theorem length_n_words_no_adjacent_a (a : ℕ → ℝ) (n : ℕ) :
  (a 1 = 3) → (a 2 = 8) →
  (∀ n, n ≥ 3 → a n = 2 * a (n - 1) + 2 * a (n - 2)) →
  (a n = (2 + real.sqrt 3) / (2 * real.sqrt 3) * (1 + real.sqrt 3)^n +
         (-2 + real.sqrt 3) / (2 * real.sqrt 3) * (1 - real.sqrt 3)^n) :=
by
  intros h1 h2 h3
  sorry

end length_n_words_no_adjacent_a_l511_511162


namespace dealer_profit_percent_l511_511883

-- Define the weight used by the dealer and the standard weight
def dishonest_weight : ℕ := 800
def standard_weight : ℕ := 1000

-- Calculate the difference in weights
def weight_difference : ℕ := standard_weight - dishonest_weight

-- Define the profit percentage
def profit_percentage : ℚ := (weight_difference.to_rat / standard_weight.to_rat) * 100

-- The theorem we want to prove
theorem dealer_profit_percent : profit_percentage = 20 := 
by
  sorry

end dealer_profit_percent_l511_511883


namespace sum_of_differences_l511_511576

/-- 
  For each natural number from 1 to 999, Damir subtracts the last digit from the first digit and 
  writes the resulting differences on a board. We are to prove that the sum of all these differences 
  is 495.
-/
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, (first_digit n - last_digit n)) = 495 :=
sorry

/-- 
  Helper function to get the first digit of a natural number.
  Here, n > 0
-/
def first_digit (n : ℕ) : ℕ :=
  n / 10^(n.digits 10 - 1)

/-- 
  Helper function to get the last digit of a natural number.
  Here, n > 0
-/
def last_digit (n : ℕ) : ℕ :=
  n % 10

end sum_of_differences_l511_511576


namespace count_same_family_functions_l511_511707

def is_function_of_same_family (f : ℝ → ℝ) (range : Set ℝ) : Prop :=
  ∀ x : ℝ, f x = x^2 ∧ (f x ∈ range ↔ x = 1 ∨ x = -1 ∨ x = 3 ∨ x = -3)

theorem count_same_family_functions :
  let range : Set ℝ := {1, 9}
  ∃ S : Finset (Set ℝ), (∀ s ∈ S, ∃ f : ℝ → ℝ, is_function_of_same_family f range ∧ S = {1, -1, 3, -3}) ∧ S.card = 9 :=
sorry

end count_same_family_functions_l511_511707


namespace evaluate_f_l511_511335

def f (x : ℝ) : ℝ :=
if x > 6 then x - 1
else if 2 ≤ x ∧ x ≤ 6 then x^2
else 5

theorem evaluate_f : f(f(f(2))) = 15 :=
  by
    sorry

end evaluate_f_l511_511335


namespace parabola_equation_and_max_slope_l511_511686

-- Define the parabola parameters and conditions
def parabola (p : ℝ) (hp : p > 0) : set (ℝ × ℝ) := 
{xy | xy.2 ^ 2 = 2 * p * xy.1}

def focus_distance (d : ℝ) : Prop := 
d = 2

-- Define the points O, P, and Q and the given vector relationship
def point_o : ℝ × ℝ := (0, 0)

def on_parabola (p : ℝ) (hp : p > 0) (P : ℝ × ℝ) : Prop :=
P ∈ parabola p hp

def vector_relationship (P Q F : ℝ × ℝ) : Prop :=
P.1 - Q.1 = 9 * (Q.1 - F.1) ∧ P.2 - Q.2 = 9 * (Q.2 - F.2)

-- Define the conditions and the proof goals
theorem parabola_equation_and_max_slope :
  ∃ p (hp : p > 0) (F : ℝ × ℝ),
  focus_distance (2 * p) →
  (∀ P, on_parabola p hp P → 
       ∃ Q, vector_relationship P Q F →
             (parabola p hp = (λ xy, xy.2^2 = 4 * xy.1) ∧
             (real.slope point_o Q ≤ (1/3))) :=
by 
  -- Proof is omitted
  sorry

end parabola_equation_and_max_slope_l511_511686


namespace five_letter_words_with_vowels_l511_511180

noncomputable def num_5_letter_words_with_vowels : Nat := 7776 - 1024

theorem five_letter_words_with_vowels
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'})
  (len : Nat := 5) :
  (letters.card ^ len) - ((letters.filter (λ c, ¬ c ∈ vowels)).card ^ len) = 6752 := by
  sorry

end five_letter_words_with_vowels_l511_511180


namespace part_I_solution_part_II_solution_l511_511812

def f (x: ℝ) := |2 * x - 1|
def g (x: ℝ) := f x + f (x - 1)

theorem part_I_solution :
  { x : ℝ | f x + |x + 1| < 2 } = set.Ioo 0 (2 / 3) :=
sorry
   
theorem part_II_solution (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 2) :
  (4 / m) + (1 / n) = 9 / 2 :=
sorry

end part_I_solution_part_II_solution_l511_511812


namespace total_crayons_l511_511512
-- Import the whole Mathlib to ensure all necessary components are available

-- Definitions of the number of crayons each person has
def Billy_crayons : ℕ := 62
def Jane_crayons : ℕ := 52
def Mike_crayons : ℕ := 78
def Sue_crayons : ℕ := 97

-- Theorem stating the total number of crayons is 289
theorem total_crayons : (Billy_crayons + Jane_crayons + Mike_crayons + Sue_crayons) = 289 := by
  sorry

end total_crayons_l511_511512


namespace range_f_l511_511156

def g (x : ℝ) : ℝ := x^2 - 2

def f (x : ℝ) : ℝ :=
  if x < g x then g x + x + 4 else g x - x

theorem range_f :
  Set.range f = {y | -9 / 4 ≤ y ∧ y ≤ 0} ∪ {y | 2 < y} :=
sorry

end range_f_l511_511156


namespace problem1_problem2_problem3_l511_511146

noncomputable def f (x : ℝ) : ℝ := (2^(2*x))/(2 + 2^(2*x))

-- Problem 1
theorem problem1 : f (1/2) = 1/2 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : f x + f (1 - x) = 1 :=
by sorry

-- Problem 3
theorem problem3 : ∑ k in Finset.range 99, f ((k + 1) / 100) = 99 / 2 :=
by sorry

end problem1_problem2_problem3_l511_511146


namespace students_recess_time_l511_511398

def initial_recess : ℕ := 20

def extra_minutes_as (as : ℕ) : ℕ := 4 * as
def extra_minutes_bs (bs : ℕ) : ℕ := 3 * bs
def extra_minutes_cs (cs : ℕ) : ℕ := 2 * cs
def extra_minutes_ds (ds : ℕ) : ℕ := ds
def extra_minutes_es (es : ℕ) : ℤ := - es
def extra_minutes_fs (fs : ℕ) : ℤ := -2 * fs

def total_recess (as bs cs ds es fs : ℕ) : ℤ :=
  initial_recess + 
  (extra_minutes_as as + extra_minutes_bs bs +
  extra_minutes_cs cs + extra_minutes_ds ds +
  extra_minutes_es es + extra_minutes_fs fs : ℤ)

theorem students_recess_time :
  total_recess 10 12 14 5 3 2 = 122 := by sorry

end students_recess_time_l511_511398


namespace video_files_initial_l511_511499

theorem video_files_initial (V : ℕ) (h1 : 4 + V - 23 = 2) : V = 21 :=
by 
  sorry

end video_files_initial_l511_511499


namespace goldfish_in_pond_l511_511401

theorem goldfish_in_pond :
  ∃ G : ℕ, (G * 0.20 * 4.5 = 45) ∧ (G = 50) :=
by
  existsi 50
  split
  sorry
  exact eq.refl 50

end goldfish_in_pond_l511_511401


namespace electric_vehicle_sales_l511_511845

noncomputable def sales_volume_Q (initial_Q : ℕ) (rate_Q : ℝ) (months : ℕ) : ℝ :=
  initial_Q * (1 - rate_Q^months) / (1 - rate_Q)

noncomputable def sales_volume_R (initial_R : ℕ) (increment_R : ℕ) (months : ℕ) : ℕ :=
  months * initial_R + (months * (months - 1) / 2) * increment_R

theorem electric_vehicle_sales (reference_11 reference_12 reference_13 : ℝ) 
  (initial_sales_Q initial_sales_R : ℕ) (growth_rate_Q : ℝ) 
  (monthly_increment_R : ℕ) (total_months : ℕ) :
  reference_11 = 2.9 → reference_12 = 3.1 → reference_13 = 3.5 →
  initial_sales_Q = 50 → initial_sales_R = 50 → growth_rate_Q = 1.1 → monthly_increment_R = 20
  → total_months = 12 → 
  sales_volume_Q initial_sales_Q growth_rate_Q total_months ≈ 1050 ∧
  sales_volume_R initial_sales_R monthly_increment_R total_months = 1920 →
  sales_volume_Q initial_sales_Q growth_rate_Q total_months 
    + sales_volume_R initial_sales_R monthly_increment_R total_months ≈ 2970 := 
by
  sorry

end electric_vehicle_sales_l511_511845


namespace dividend_is_5336_l511_511887

theorem dividend_is_5336 (D Q R : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) : 
  D * Q + R = 5336 := 
by sorry

end dividend_is_5336_l511_511887


namespace five_letter_words_with_vowels_l511_511222

/-
How many 5-letter words with at least one vowel can be constructed from the letters 
A, B, C, D, E, and F? (Note that A and E are vowels, any word is valid, not just English language 
words, and letters may be used more than once.)
-/

theorem five_letter_words_with_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := ['A', 'E'],
      consonants := ['B', 'C', 'D', 'F'] in
  let total_words := 6 ^ 5,
      consonant_only_words := 4 ^ 5,
      at_least_one_vowel_words := total_words - consonant_only_words in
  at_least_one_vowel_words = 6752 :=
by
  intro letters vowels consonants total_words consonant_only_words at_least_one_vowel_words
  sorry

end five_letter_words_with_vowels_l511_511222


namespace original_population_is_7021_l511_511928

noncomputable def original_population (final_pop : ℕ) : ℕ :=
  let p := (final_pop : ℝ) / 0.673 in
  round p

theorem original_population_is_7021 :
  original_population 4725 = 7021 :=
by
  -- conditions derived from the problem
  have h1 : 0.9 ≠ 0 := by normNum
  have h2 : 0.75 ≠ 0 := by normNum
  have birth_rate : ℝ := 0.008
  have death_rate : ℝ := 0.01
  have final_population : ℝ := 4725

  -- original population in terms of final population
  let p := final_population / 0.673
  have h3 : original_population final_population = round p := by rfl
  rw [h3]
  norm_cast
  normNum

-- adds sorry to complete the theorem without full proof steps

end original_population_is_7021_l511_511928


namespace concyclic_ABRS_l511_511438

theorem concyclic_ABRS :
    ∀ (w1 w2 Γ : Circle) (l : Line) (K L M N A B R S : Point),
    (Tangent w1 l K) ∧ 
    (Tangent w2 l L) ∧ 
    (Tangent w1 Γ M) ∧ 
    (Tangent w2 Γ N) ∧ 
    (¬ (Intersect w1 w2)) ∧ 
    (¬ (EqualSize w1 w2)) ∧ 
    (CirclePassThrough KL A B) ∧ 
    (Reflection l M R) ∧ 
    (Reflection l N S) →
    Concyclic A B R S :=
by
  sorry

end concyclic_ABRS_l511_511438


namespace sum_digit_differences_l511_511605

def first_digit (n : ℕ) : ℕ := 
  (n / 10 ^ ((Nat.log10 n) : ℕ))

def last_digit (n : ℕ) : ℕ := n % 10

def digit_difference (n : ℕ) : ℤ :=
  (first_digit n : ℤ) - (last_digit n : ℤ)

theorem sum_digit_differences :
  (∑ n in Finset.range 1000, digit_difference n) = 495 := 
sorry

end sum_digit_differences_l511_511605


namespace tenth_integer_is_6785_l511_511836

-- Definitions
def four_digit_numbers (digits : List ℕ) : List ℕ := 
  List.permutations digits
  |> List.map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)
  |> List.filter (λ n, 1000 ≤ n ∧ n < 10000)

def tenth_integer (l : List ℕ) : ℕ :=
  l.nth 9 -- List indexing is zero-based, so the 10th element is the 9th index

-- Conditions
def used_digits := [5, 6, 7, 8]

-- Theorem statement
theorem tenth_integer_is_6785 : 
  ∀ (l : List ℕ), l = four_digit_numbers used_digits → (tenth_integer (l.sorted)) = 6785 :=
by
  sorry

end tenth_integer_is_6785_l511_511836


namespace no_integer_cube_eq_3n_squared_plus_3n_plus_7_l511_511967

theorem no_integer_cube_eq_3n_squared_plus_3n_plus_7 :
  ¬ ∃ x n : ℤ, x^3 = 3 * n^2 + 3 * n + 7 := 
sorry

end no_integer_cube_eq_3n_squared_plus_3n_plus_7_l511_511967


namespace shooters_points_distribution_l511_511855

noncomputable def points_distribution : List ℕ := [50, 25, 25, 20, 20, 20, 10, 10, 10, 5, 5, 3, 3, 2, 2, 1, 1, 1]

def total_points : ℕ := points_distribution.sum -- 213 points

def points_per_shooter (shooters : List (List ℕ)) : Prop :=
  shooters.length = 3 ∧ (∀ shooter ∈ shooters, shooter.length = 6 ∧ shooter.sum = 71)

def anilov_cond (shooter : List ℕ) : Prop := shooter.take 3.sum = 43
def borisov_cond (shooter : List ℕ) : Prop := shooter.head! = 3

theorem shooters_points_distribution (anilov borisov vorobyev : List ℕ)
  (h_valid_anilov : anilov_cond anilov)
  (h_valid_borisov : borisov_cond borisov)
  (h_distinct_elements : List.perm (anilov ++ borisov ++ vorobyev) points_distribution)
  : points_per_shooter [anilov, borisov, vorobyev] :=
by
  sorry

end shooters_points_distribution_l511_511855


namespace south_movement_notation_l511_511709

/-- If moving north 8m is denoted as +8m, then moving south 5m is denoted as -5m. -/
theorem south_movement_notation (north south : ℤ) (h1 : north = 8) (h2 : south = -north) : south = -5 :=
by
  sorry

end south_movement_notation_l511_511709


namespace sqrt_meaningful_iff_ge_eight_l511_511265

theorem sqrt_meaningful_iff_ge_eight (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 8)) ↔ x ≥ 8 := by
  sorry

end sqrt_meaningful_iff_ge_eight_l511_511265


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511227

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let total_words := 6^5
  let words_without_vowels := 4^5
  total_words - words_without_vowels = 6752 :=
by
  let total_words := 6^5
  let words_without_vowels := 4^5
  have h_total_words : total_words = 7776 := by norm_num
  have h_words_without_vowels : words_without_vowels = 1024 := by norm_num
  calc
    7776 - 1024 = 6752 : by norm_num

end number_of_5_letter_words_with_at_least_one_vowel_l511_511227


namespace five_letter_words_with_vowel_l511_511193

theorem five_letter_words_with_vowel : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let n := 5 
  (∃ (w : list(char)), w.length = n ∧ ∀ (i : fin n), w[i] ∈ letters ∧ (∃ (j : fin n), w[j] ∈ vowels)) → 
  (6^5 - 4^5 = 6752) := 
by
  sorry

end five_letter_words_with_vowel_l511_511193


namespace reciprocal_eq_self_l511_511391

theorem reciprocal_eq_self (x : ℝ) : (1 / x = x) ↔ (x = 1 ∨ x = -1) :=
sorry

end reciprocal_eq_self_l511_511391


namespace third_neigh_uses_100_more_l511_511031

def total_water : Nat := 1200
def first_neigh_usage : Nat := 150
def second_neigh_usage : Nat := 2 * first_neigh_usage
def fourth_neigh_remaining : Nat := 350

def third_neigh_usage := total_water - (first_neigh_usage + second_neigh_usage + fourth_neigh_remaining)
def diff_third_second := third_neigh_usage - second_neigh_usage

theorem third_neigh_uses_100_more :
  diff_third_second = 100 := by
  sorry

end third_neigh_uses_100_more_l511_511031


namespace five_letter_words_with_vowels_l511_511219

/-
How many 5-letter words with at least one vowel can be constructed from the letters 
A, B, C, D, E, and F? (Note that A and E are vowels, any word is valid, not just English language 
words, and letters may be used more than once.)
-/

theorem five_letter_words_with_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := ['A', 'E'],
      consonants := ['B', 'C', 'D', 'F'] in
  let total_words := 6 ^ 5,
      consonant_only_words := 4 ^ 5,
      at_least_one_vowel_words := total_words - consonant_only_words in
  at_least_one_vowel_words = 6752 :=
by
  intro letters vowels consonants total_words consonant_only_words at_least_one_vowel_words
  sorry

end five_letter_words_with_vowels_l511_511219


namespace angle_is_pi_over_3_l511_511999

variables {ℝ : Type*} [inner_product_space ℝ ℝ] -- ℝ denotes the real numbers with the Euclidean inner product space structure
variables {a b : ℝ} -- Vectors 'a' and 'b'

noncomputable def angle_between_vectors (a b : ℝ) : ℝ :=
  real.arccos ((inner_product_space.inner a b) / (∥a∥ * ∥b∥))

theorem angle_is_pi_over_3
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 6)
  (hab : inner_product_space.inner a (b - a) = 2) :
  angle_between_vectors a b = real.pi / 3 :=
sorry

end angle_is_pi_over_3_l511_511999


namespace five_letter_words_with_vowels_l511_511179

noncomputable def num_5_letter_words_with_vowels : Nat := 7776 - 1024

theorem five_letter_words_with_vowels
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'})
  (len : Nat := 5) :
  (letters.card ^ len) - ((letters.filter (λ c, ¬ c ∈ vowels)).card ^ len) = 6752 := by
  sorry

end five_letter_words_with_vowels_l511_511179


namespace binomial_p_value_l511_511991

-- Define the conditions X ~ B(n, p)
variables {X : Type*} (n : ℕ) (p : ℝ)

-- Assume the binomial distribution and given condition
noncomputable def is_binomial_and_condition : Prop :=
  ∃ (X : ℝ), (X = n * p ∧ 3 * E(X) = 10 * D(X))

-- Prove that p = 0.7 given the conditions
theorem binomial_p_value : is_binomial_and_condition n p → p = 0.7 :=
by
  sorry

end binomial_p_value_l511_511991


namespace james_spent_163_for_the_night_l511_511470

noncomputable def totalSpent : ℕ :=
  let club_entry_fee := 20
  let rounds_bought_for_friends := 2
  let friends := 5
  let cost_per_drink := 6
  let drinks_for_himself := 6
  let cost_of_chicken := 14
  let tip_percentage := 0.30 in
  let cost_of_drinks_for_friends := rounds_bought_for_friends * friends * cost_per_drink
  let cost_of_drinks_for_himself := drinks_for_himself * cost_per_drink
  let total_cost_of_drinks := cost_of_drinks_for_friends + cost_of_drinks_for_himself
  let subtotal := club_entry_fee + total_cost_of_drinks + cost_of_chicken
  let tip := tip_percentage * (total_cost_of_drinks + cost_of_chicken) in
  subtotal + tip

theorem james_spent_163_for_the_night : totalSpent = 163 :=
sorry

end james_spent_163_for_the_night_l511_511470


namespace find_m_value_l511_511389

theorem find_m_value : 
  ∃ (m : ℝ), 
  (∃ (x y : ℝ), (x - 2)^2 + (y + 1)^2 = 1 ∧ (x - y + m = 0)) → m = -3 :=
by
  sorry

end find_m_value_l511_511389


namespace isosceles_obtuse_triangle_smallest_angle_l511_511504

theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (α β γ : ℝ), α = 1.8 * 90 ∧ β = γ ∧ α + β + γ = 180 → β = 9 :=
by
  intros α β γ h
  sorry

end isosceles_obtuse_triangle_smallest_angle_l511_511504


namespace exists_prime_not_dividing_difference_l511_511068

theorem exists_prime_not_dividing_difference {m : ℕ} (hm : m ≠ 1) : 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, ¬ p ∣ (n^n - m) := 
sorry

end exists_prime_not_dividing_difference_l511_511068


namespace sqrt_equality_l511_511135

theorem sqrt_equality (a b : ℝ) (h1 : sqrt (2 + 2/3) = 2 * sqrt (2/3)) 
    (h2 : sqrt (3 + 3/8) = 3 * sqrt (3/8)) 
    (h3 : sqrt (4 + 4/15) = 4 * sqrt (4/15)) 
    (h4 : sqrt (6 + a / b) = 6 * sqrt (a / b)) : 
    a + b = 41 := by
  sorry

end sqrt_equality_l511_511135


namespace solution_set_of_fraction_inequality_l511_511396

theorem solution_set_of_fraction_inequality (a b x : ℝ) (h1: ∀ x, ax - b > 0 ↔ x ∈ Set.Iio 1) (h2: a < 0) (h3: a - b = 0) :
  ∀ x, (a * x + b) / (x - 2) > 0 ↔ x ∈ Set.Ioo (-1 : ℝ) 2 := 
sorry

end solution_set_of_fraction_inequality_l511_511396


namespace largest_number_with_digits_product_120_is_85311_l511_511558

-- Define the five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the product of digits
def digits_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

-- Define the condition that the product of the digits should be 120
def product_is_120 (n : ℕ) : Prop :=
  digits_product n = 120

-- Define the condition that n is the largest such number
def largest_such_number (n : ℕ) : Prop :=
  ∀ m : ℕ, is_five_digit m → product_is_120 m → m ≤ n

-- The theorem stating that the largest five-digit number whose digits' product equals 120 is 85311
theorem largest_number_with_digits_product_120_is_85311 : ∃ n : ℕ, is_five_digit n ∧ product_is_120 n ∧ largest_such_number n ∧ n = 85311 :=
by
  use 85311
  split
  -- Prove 85311 is a five-digit number
  sorry
  split
  -- Prove the product of the digits of 85311 is 120
  sorry
  split
  -- Prove that 85311 is the largest such number
  sorry
  -- Prove n = 85311
  sorry

end largest_number_with_digits_product_120_is_85311_l511_511558


namespace garden_line_segment_length_l511_511491

theorem garden_line_segment_length :
  ∀ (scale length_in_inches : ℝ), scale = 500 → length_in_inches = 6.5 → length_in_inches * scale = 3250 :=
by
  intros scale length_in_inches hscale hlength_in_inches
  rw [hscale, hlength_in_inches]
  norm_num
  sorry

end garden_line_segment_length_l511_511491


namespace sum_digit_differences_l511_511606

def first_digit (n : ℕ) : ℕ := 
  (n / 10 ^ ((Nat.log10 n) : ℕ))

def last_digit (n : ℕ) : ℕ := n % 10

def digit_difference (n : ℕ) : ℤ :=
  (first_digit n : ℤ) - (last_digit n : ℤ)

theorem sum_digit_differences :
  (∑ n in Finset.range 1000, digit_difference n) = 495 := 
sorry

end sum_digit_differences_l511_511606


namespace find_intersection_pair_l511_511384

def cubic_function (x : ℝ) : ℝ := x^3 - 3*x + 2

def linear_function (x y : ℝ) : Prop := x + 4*y = 4

def intersection_points (x y : ℝ) : Prop := 
  linear_function x y ∧ y = cubic_function x

def sum_x_coord (points : List (ℝ × ℝ)) : ℝ :=
  points.map Prod.fst |>.sum

def sum_y_coord (points : List (ℝ × ℝ)) : ℝ :=
  points.map Prod.snd |>.sum

theorem find_intersection_pair (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1 : intersection_points x1 y1)
  (h2 : intersection_points x2 y2)
  (h3 : intersection_points x3 y3)
  (h_sum_x : sum_x_coord [(x1, y1), (x2, y2), (x3, y3)] = 0) :
  sum_y_coord [(x1, y1), (x2, y2), (x3, y3)] = 3 :=
sorry

end find_intersection_pair_l511_511384


namespace sum_differences_1_to_999_l511_511588

-- Define a utility function to compute the first digit of a number
def first_digit (n : ℕ) : ℕ :=
if n < 10 then n else first_digit (n / 10)

-- Define a utility function to compute the last digit of a number
def last_digit (n : ℕ) : ℕ :=
n % 10

-- Define the operation performed by Damir
def difference (n : ℕ) : ℤ :=
(first_digit n : ℤ) - (last_digit n : ℤ)

-- Define the sum of all differences from 1 to 999
def sum_differences : ℤ :=
(1).to (999).sum (λ n, difference n)

-- State the main theorem to be proved using the previous definitions
theorem sum_differences_1_to_999 : sum_differences = 495 :=
sorry

end sum_differences_1_to_999_l511_511588


namespace vertex_of_parabola_l511_511375

-- Define the parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x + 9)^2 - 3

-- State the theorem to prove
theorem vertex_of_parabola : ∃ h k : ℝ, (h = -9 ∧ k = -3) ∧ (parabola h = k) :=
by sorry

end vertex_of_parabola_l511_511375


namespace pieces_of_fudge_l511_511903

def pan_length : ℝ := 27.5
def pan_width : ℝ := 17.5
def pan_height : ℝ := 2.5
def cube_side : ℝ := 2.3

def volume (l w h : ℝ) : ℝ := l * w * h

def V_pan : ℝ := volume pan_length pan_width pan_height
def V_cube : ℝ := volume cube_side cube_side cube_side

theorem pieces_of_fudge : ⌊V_pan / V_cube⌋ = 98 := by
  -- calculation can be filled in here in the actual proof
  sorry

end pieces_of_fudge_l511_511903


namespace find_p_l511_511692

section vector_problem

variables (a b p : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) (t : ℝ)
 
def vector_a := (2, -2, 3)
def vector_b := (1, 4, -1)
def direction_vector := (vector_b - vector_a)
def p_param := (2 - t, -2 + 6 * t, 3 - 4 * t)

axiom orthogonal_to_direction : (p_param t).1 * direction_vector.1 + (p_param t).2 * direction_vector.2 + (p_param t).3 * direction_vector.3 = 0
axiom collinear_a_b_p : ∃ v, p = v * a ∧ p = v * b

theorem find_p (h₁ : vector_a = (2, -2, 3)) 
  (h₂ : vector_b = (1, 4, -1))
  (h₃ : direction_vector = (vector_b - vector_a)) 
  (h₄ : p = (2 - t, -2 + 6 * t, 3 - 4 * t))
  (h₅ : orthogonal_to_direction)
  : p = (3/2, 1, 1) := 
sorry

end vector_problem

end find_p_l511_511692


namespace find_square_sum_l511_511840

theorem find_square_sum :
  ∃ a b c : ℕ, a = 2494651 ∧ b = 1385287 ∧ c = 9406087 ∧ (a + b + c = 3645^2) :=
by
  have h1 : 2494651 + 1385287 + 9406087 = 13286025 := by norm_num
  have h2 : 3645^2 = 13286025 := by norm_num
  exact ⟨2494651, 1385287, 9406087, rfl, rfl, rfl, h2⟩

end find_square_sum_l511_511840


namespace total_trip_cost_is_1770_l511_511946
noncomputable theory
open BigOperators

def distance_XZ : ℝ := 4000
def distance_XY : ℝ := 4500
def cost_per_km_bus : ℝ := 0.20
def booking_fee_plane : ℝ := 120
def cost_per_km_plane : ℝ := 0.12

def distance_YZ : ℝ := Real.sqrt ((distance_XY ^ 2) - (distance_XZ ^ 2))

def cost (distance : ℝ) (is_bus : Bool) : ℝ :=
  if is_bus then
    cost_per_km_bus * distance
  else
    booking_fee_plane + (cost_per_km_plane * distance)

def cost_XY : ℝ := min (cost distance_XY true) (cost distance_XY false)
def cost_YZ : ℝ := min (cost distance_YZ true) (cost distance_YZ false)
def cost_ZX : ℝ := min (cost distance_XZ true) (cost distance_XZ false)

def total_min_cost : ℝ := cost_XY + cost_YZ + cost_ZX

theorem total_trip_cost_is_1770 : total_min_cost = 1770 := by
  sorry

end total_trip_cost_is_1770_l511_511946


namespace A_is_Z_l511_511819

noncomputable def A : Set ℤ :=
  {x | ∃ a b : ℤ, a ≥ 1 ∧ b ≥ 1 ∧ x = 2^a - 2^b} ∪ {x | ∃ a b : ℤ, a ∈ A ∧ b ∈ A ∧ x = a + b ∧ ∃ k : ℤ, x = 2*k + 1}

lemma exists_at_least_one_odd_in_A : ∃ (u : ℤ), u ∈ A ∧ (∃ k : ℤ, u = 2*k + 1) :=
  sorry

theorem A_is_Z (h : ∃ u : ℤ, u ∈ A ∧ ∃ k : ℤ, u = 2*k + 1 ) : A = Set.univ :=
  sorry

end A_is_Z_l511_511819


namespace sum_of_differences_l511_511584

open Nat
open BigOperators

theorem sum_of_differences (n : ℕ) (h : n ≥ 1 ∧ n ≤ 999) : 
  let differences := (fun x => 
                        let first_digit := x / 10;
                        let last_digit := x % 10;
                        first_digit - last_digit) in
  ∑ i in (finset.range (n + 1)).filter (λ x, 1 ≤ x ∧ x ≤ 999), differences i = 495 :=
by
  -- Acknowledge the need for a more refined filtering criteria for numbers between 1 and 999
  sorry

end sum_of_differences_l511_511584


namespace hyperbola_no_common_points_l511_511319

noncomputable def hyperbola_eccentricity (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) : ℝ :=
  real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_no_common_points (a b e : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_ecc : e = real.sqrt (1 + (b^2 / a^2)))
  (h_slope : b / a < 2) :
  e = 2 :=
sorry

end hyperbola_no_common_points_l511_511319


namespace max_sequence_length_l511_511720

theorem max_sequence_length (n : ℕ) (a : ℕ → ℝ)
  (h7 : ∀ i : ℕ, i + 6 < n → ∑ k in finset.range 7, a (i + k) < 0)
  (h11 : ∀ i : ℕ, i + 10 < n → ∑ k in finset.range 11, a (i + k) > 0) :
  n ≤ 16 :=
sorry

end max_sequence_length_l511_511720


namespace range_of_p_l511_511151

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

-- A = { x | f'(x) ≤ 0 }
def A : Set ℝ := { x | deriv f x ≤ 0 }

-- B = { x | p + 1 ≤ x ≤ 2p - 1 }
def B (p : ℝ) : Set ℝ := { x | p + 1 ≤ x ∧ x ≤ 2 * p - 1 }

-- Given that A ∪ B = A, prove the range of values for p is ≤ 3.
theorem range_of_p (p : ℝ) : (A ∪ B p = A) → p ≤ 3 := sorry

end range_of_p_l511_511151


namespace turbo_never_visits_point_l511_511416

noncomputable def exists_never_visited_point (seq : ℕ → ℝ) (C : ℝ) : Prop :=
  (∀ i, seq i < C) → 
  ∃ X, ∀ n, ∃ P, ¬ ((P ≤ X ∧ P + seq n > X) ∨ (P ≥ X ∧ P - seq n < X))

theorem turbo_never_visits_point :
  exists_never_visited_point (λ n, 1/2) 1/2 :=
by sorry

end turbo_never_visits_point_l511_511416


namespace investment_worth_28_years_l511_511888

theorem investment_worth_28_years (P : ℝ) (x : ℝ) (t : ℝ) (tripling_time : ℝ) :
  x = 0.08 → P = 2500 → t = 28 → tripling_time = 112 / x → 
  ∀ n : ℕ, t / tripling_time = n → 3^n = 9 → n = 2 →
  ((P * 3^n) = 22500) :=
by intro h₁ h₂ h₃ h₄ h₅ h₆ h₇; rw [h₁, h₂, h₃, h₄, h₅, h₆, h₇]; rfl;

end investment_worth_28_years_l511_511888


namespace replace_digits_divisible_by_13_l511_511894

def is_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

theorem replace_digits_divisible_by_13 :
  ∃ (x y : ℕ), is_digit x ∧ is_digit y ∧ 
  (3 * 10^6 + x * 10^4 + y * 10^2 + 3) % 13 = 0 ∧
  (x = 2 ∧ y = 3 ∨ 
   x = 5 ∧ y = 2 ∨ 
   x = 8 ∧ y = 1 ∨ 
   x = 9 ∧ y = 5 ∨ 
   x = 6 ∧ y = 6 ∨ 
   x = 3 ∧ y = 7 ∨ 
   x = 0 ∧ y = 8) :=
by
  sorry

end replace_digits_divisible_by_13_l511_511894


namespace problem1_problem2_problem3_problem4_l511_511899

-- Problem 1
theorem problem1 (a b c : ℝ) :
  (1 / 2 * a^2 * b * c^3) * (-2 * a^3 * b^2 * c)^2 = 2 * a^8 * b^5 * c^5 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) :
  (25 * m^2 + 15 * m^3 * n - 20 * m^4) / (-5 * m^2) = -5 - 3 * m * n + 4 * m^2 :=
by sorry

-- Problem 3
theorem problem3 (x y : ℝ) :
  (2 * x - 1 / 2 * y)^2 = 4 * x^2 - 2 * x * y + 1 / 4 * y^2 :=
by sorry

-- Problem 4
theorem problem4 :
  (-4)^2 - abs (-1 / 2) + 2^(-2) - 2014^0 = (14:ℚ) + 3 / 4 :=
by sorry

end problem1_problem2_problem3_problem4_l511_511899


namespace proposition_3_proposition_4_l511_511038

variables {a b : ℝ} {x : Type} {M N : set x}

theorem proposition_3 (h : log a = log b) : a = b :=
begin
  -- Sufficient but not necessary condition proof
  sorry
end

theorem proposition_4 (h : x ∈ M ∩ N) : x ∈ M ∪ N :=
begin
  -- Sufficient but not necessary condition proof
  sorry
end

end proposition_3_proposition_4_l511_511038


namespace evaluate_fraction_eq_10_pow_10_l511_511972

noncomputable def evaluate_fraction (a b c : ℕ) : ℕ :=
  (a ^ 20) / ((a * b) ^ 10)

theorem evaluate_fraction_eq_10_pow_10 :
  evaluate_fraction 30 3 10 = 10 ^ 10 :=
by
  -- We define what is given and manipulate it directly to form a proof outline.
  sorry

end evaluate_fraction_eq_10_pow_10_l511_511972


namespace scientific_notation_of_42_trillion_l511_511280

theorem scientific_notation_of_42_trillion : (42.1 * 10^12) = 4.21 * 10^13 :=
by
  sorry

end scientific_notation_of_42_trillion_l511_511280


namespace rabbit_catch_up_time_l511_511863

theorem rabbit_catch_up_time
  (rabbit_speed : ℝ)
  (cat_speed : ℝ)
  (head_start_minutes : ℝ)
  (head_start_hours : head_start_minutes / 60)
  (cat_head_start_distance : cat_speed * head_start_hours)
  (speed_difference : rabbit_speed - cat_speed)
  :
  (cat_head_start_distance / speed_difference) = 1 :=
by
  sorry

end rabbit_catch_up_time_l511_511863


namespace product_congruent_three_mod_p_l511_511352

open BigOperators

theorem product_congruent_three_mod_p {p : ℕ} (hp : Nat.Prime p) (hp3 : p > 3) (hp_congruent : p % 3 = 2) :
  (∏ k in Finset.range (p - 1), (k^2 + k + 1) : ZMod p) = 3 := sorry

end product_congruent_three_mod_p_l511_511352


namespace eccentricity_bound_l511_511308

variables {a b c e : ℝ}

-- Definitions of the problem conditions
def hyperbola (x y : ℝ) (a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def line (x : ℝ) := 2 * x
def eccentricity (c a : ℝ) := c / a

-- Proof statement in Lean
theorem eccentricity_bound (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (c : ℝ)
  (h₃ : hyperbola x y a b)
  (h₄ : ∀ x, line x ≠ y) :
  1 < eccentricity c a ∧ eccentricity c a ≤ sqrt 5 :=
sorry

end eccentricity_bound_l511_511308


namespace sum_of_differences_l511_511614

/-- Proving that the sum of all differences (first digit - last digit) for natural numbers from 
    1 to 999 is 495. -/ 
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, 
    let str := n.toString
    let first := if str.length > 1 then str.head!.toNat - '0'.toNat else 0
    let last := if str.length > 1 then str.getLast.toNat - '0'.toNat else 0
    first - last
  ) = 495 := 
by
  sorry

end sum_of_differences_l511_511614


namespace f_increasing_over_ℝ_l511_511327

def f (x : ℝ) : ℝ :=
if x < 0 then x - sin x else x^3 + 1

theorem f_increasing_over_ℝ : ∀ x y : ℝ, x < y → f x < f y := 
sorry

end f_increasing_over_ℝ_l511_511327


namespace five_letter_words_with_vowels_l511_511175

noncomputable def num_5_letter_words_with_vowels : Nat := 7776 - 1024

theorem five_letter_words_with_vowels
  (letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'})
  (vowels : Finset Char := {'A', 'E'})
  (len : Nat := 5) :
  (letters.card ^ len) - ((letters.filter (λ c, ¬ c ∈ vowels)).card ^ len) = 6752 := by
  sorry

end five_letter_words_with_vowels_l511_511175


namespace probability_gcd_one_l511_511430

-- Defining the domain of our problem: the set {1, 2, 3, ..., 8}
def S := {1, 2, 3, 4, 5, 6, 7, 8}

-- Defining the selection of two distinct natural numbers from S
def select_two_distinct_from_S (x y : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ x ≠ y

-- Defining the greatest common factor condition
def is_rel_prime (x y : ℕ) : Prop :=
  Nat.gcd x y = 1

-- Defining the probability computation (relatively prime pairs over total pairs)
def probability_rel_prime : ℚ :=
  (21 : ℚ) / 28  -- since 21 pairs are relatively prime out of 28 total pairs

-- The main theorem statement
theorem probability_gcd_one :
  probability_rel_prime = 3 / 4 :=
sorry

end probability_gcd_one_l511_511430


namespace parabola_intercept_sum_l511_511383

theorem parabola_intercept_sum :
  let a := 6
  let b := 1
  let c := 2
  a + b + c = 9 :=
by
  sorry

end parabola_intercept_sum_l511_511383


namespace ratio_of_third_layer_to_second_l511_511513

theorem ratio_of_third_layer_to_second (s1 s2 s3 : ℕ) (h1 : s1 = 2) (h2 : s2 = 2 * s1) (h3 : s3 = 12) : s3 / s2 = 3 := 
by
  sorry

end ratio_of_third_layer_to_second_l511_511513


namespace probability_coprime_l511_511423

open BigOperators

theorem probability_coprime (A : Finset ℕ) (h : A = {1, 2, 3, 4, 5, 6, 7, 8}) :
  let pairs := { (a, b) ∈ (A ×ˢ A) | a < b }
  let coprime_pairs := pairs.filter (λ p, Nat.gcd p.1 p.2 = 1)
  coprime_pairs.card / pairs.card = 5 / 7 := by 
sorry

end probability_coprime_l511_511423


namespace blue_paint_cans_l511_511346

theorem blue_paint_cans (total_cans : ℕ) (blue_ratio : ℕ) (green_ratio : ℕ) (total_cans = 45) (blue_ratio = 5) (green_ratio = 3) : 
  let blue_fraction := (blue_ratio : ℝ) / (blue_ratio + green_ratio)
  let blue_cans := blue_fraction * total_cans
  Int.floor blue_cans = 28 := by
  sorry

end blue_paint_cans_l511_511346


namespace arrangement_count_l511_511498

/-- Among 7 workers, 5 can do typesetting and 4 can do printing.
    Prove that the number of ways to arrange 2 people for typesetting and 2 for printing is 37. -/
theorem arrangement_count : 
  let T := 5 -- number of workers who can do typesetting
  let P := 4 -- number of workers who can do printing
  let N := 7 -- total number of workers
  ( T + P - N = 2 /\ ∑ i in Finset.range 3, (Nat.choose 3 i) * (Nat.choose 4 (2 - i)) = 37) 
  :=
by
  sorry

end arrangement_count_l511_511498


namespace infinite_squares_of_form_l511_511758

theorem infinite_squares_of_form (k : ℕ) (hk : k > 0) : ∃ᶠ n in at_top, ∃ m : ℕ, n * 2^k - 7 = m^2 := sorry

end infinite_squares_of_form_l511_511758


namespace largest_intersection_value_l511_511958

theorem largest_intersection_value (b c d : ℝ) :
  ∀ x : ℝ, (x^7 - 12*x^6 + 44*x^5 - 24*x^4 + b*x^3 = c*x - d) → x ≤ 6 := sorry

end largest_intersection_value_l511_511958


namespace smaller_square_percentage_l511_511923

theorem smaller_square_percentage
  (side_length_large_square : ℝ)
  (radius_circle : ℝ)
  (side_length_small_square : ℝ)
  (area_small_square : ℝ)
  (area_large_square : ℝ)
  (percent_area : ℝ) :
  side_length_large_square = 4 →  
  radius_circle = 2 * real.sqrt 2 →
  side_length_small_square = 0.8 →
  area_small_square = side_length_small_square ^ 2 →
  area_large_square = side_length_large_square ^ 2 →
  percent_area = 100 * area_small_square / area_large_square →
  percent_area = 4 :=
by
  sorry

end smaller_square_percentage_l511_511923


namespace sum_of_differences_l511_511615

/-- Proving that the sum of all differences (first digit - last digit) for natural numbers from 
    1 to 999 is 495. -/ 
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, 
    let str := n.toString
    let first := if str.length > 1 then str.head!.toNat - '0'.toNat else 0
    let last := if str.length > 1 then str.getLast.toNat - '0'.toNat else 0
    first - last
  ) = 495 := 
by
  sorry

end sum_of_differences_l511_511615


namespace relatively_prime_probability_l511_511428

open Finset

theorem relatively_prime_probability :
  let s := ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)
  in let pairs := s.val.powerset.filter (λ t, t.card = 2)
  in (pairs.count (λ t, (t : Multiset ℕ).gcd = 1)).toRational / pairs.card.toRational = 3 / 4 := 
by
  -- Prove that the probability is 3/4
  sorry

end relatively_prime_probability_l511_511428


namespace infinite_squares_of_form_l511_511759

theorem infinite_squares_of_form (k : ℕ) (hk : k > 0) : ∃ᶠ n in at_top, ∃ m : ℕ, n * 2^k - 7 = m^2 := sorry

end infinite_squares_of_form_l511_511759


namespace percentage_deficit_for_second_side_l511_511726

theorem percentage_deficit_for_second_side
  (L W : ℝ) 
  (measured_first_side : ℝ := 1.12 * L) 
  (error_in_area : ℝ := 1.064) : 
  (∃ x : ℝ, (1.12 * L) * ((1 - 0.01 * x) * W) = 1.064 * (L * W) → x = 5) :=
by
  sorry

end percentage_deficit_for_second_side_l511_511726


namespace problem_statement_l511_511067

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) = f(-x)
  
def is_monotonically_increasing_on_negatives (f : ℝ → ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), x1 < 0 → x2 < 0 → x1 ≠ x2 → (f(x2) - f(x1)) / (x2 - x1) > 0

theorem problem_statement (f : ℝ → ℝ)
  (h1 : is_even_function f)
  (h2 : is_monotonically_increasing_on_negatives f) :
  f(-4) < f(3) ∧ f(3) < f(-2) :=
sorry

end problem_statement_l511_511067


namespace balls_into_boxes_l511_511699

theorem balls_into_boxes :
  (number_of_ways_to_distribute 7 4) = 128 := sorry

end balls_into_boxes_l511_511699


namespace wednesday_earns_1330_50_less_than_sunday_l511_511463

-- Define the amounts generated on Wednesday and Sunday
def WednesdayEarnings := 1832
def SundayEarnings := 3162.5

-- Define the total amount earned from both games
def TotalEarnings := WednesdayEarnings + SundayEarnings

-- The given total amount earned
def GivenTotal := 4994.5

-- Define the proof statement
theorem wednesday_earns_1330_50_less_than_sunday :
  WednesdayEarnings < SundayEarnings ∧ TotalEarnings = GivenTotal → SundayEarnings - WednesdayEarnings = 1330.5 :=
by
  intros
  sorry

end wednesday_earns_1330_50_less_than_sunday_l511_511463


namespace inverse_of_inverse_l511_511752

noncomputable def g (x : ℝ) : ℝ := 6 * x - 3
noncomputable def g_inv (x : ℝ) : ℝ := (x + 3) / 6

theorem inverse_of_inverse (x : ℝ) (h : x = 15) : g_inv (g_inv x) = 1 :=
by
  have h1 : g_inv 15 = 3 := by sorry
  have h2 : g_inv 3 = 1 := by sorry
  rw [h] at h1
  exact h2

end inverse_of_inverse_l511_511752


namespace parabola_equation_and_fixed_point_l511_511730

-- Define the equations and theorems with the given conditions
theorem parabola_equation_and_fixed_point :
  ∃ (C : ℝ → ℝ) (f : ℝ × ℝ → Prop),
    (∀ x y, C(x) = y ↔ x^2 = 4 * y) ∧
    (∀ t, t > 0 → C(t) = 1) ∧
    (∀ x1 y1 x2 y2,
      C(x1) = y1 → C(x2) = y2 →
      x1 ≠ x2 →
      (x1 + 2) * (x2 + 2) = -16 →
      ∃ (px py : ℝ), px = -2 ∧ py = 5 ∧ (x1 * x2 - px * (x1 + x2) + 4 * py = 0)) :=
begin
  sorry
end

end parabola_equation_and_fixed_point_l511_511730


namespace parametric_eq_cartsian_eqn_l511_511835

/-
  Define parametric equations x(t) and y(t)
  as per the correct option D
-/
def x (t : ℝ) : ℝ := Real.tan t
def y (t : ℝ) : ℝ := 1 / (Real.tan t)

/-
  State the theorem claiming that the product
  of x(t) and y(t) equals 1 under the domain
  where tan(t) is defined and non-zero
-/
theorem parametric_eq_cartsian_eqn (t : ℝ) (ht : Real.tan t ≠ 0) : x t * y t = 1 := by
  sorry

end parametric_eq_cartsian_eqn_l511_511835


namespace Ding_reads_Romance_of_the_Three_Kingdoms_l511_511724

inductive Novel where
  | Journey_to_the_West
  | Dream_of_the_Red_Chamber
  | Water_Margin
  | Romance_of_the_Three_Kingdoms

open Novel

variable (Jia Yi Bing Ding : Novel)

variable (A1 : Jia = Journey_to_the_West)
variable (A2 : Yi = Dream_of_the_Red_Chamber)

variable (B1 : Jia = Water_Margin)
variable (B2 : Bing = Romance_of_the_Three_Kingdoms)

variable (C1 : Yi = Water_Margin)
variable (C2 : Bing = Journey_to_the_West)

variable (D1 : Yi = Journey_to_the_West)
variable (D2 : Ding = Romance_of_the_Three_Kingdoms)

variable (one_true_A : (A1 ∨ A2) ∧ ¬ (A1 ∧ A2))
variable (one_true_B : (B1 ∨ B2) ∧ ¬ (B1 ∧ B2))
variable (one_true_C : (C1 ∨ C2) ∧ ¬ (C1 ∧ C2))
variable (one_true_D : (D1 ∨ D2) ∧ ¬ (D1 ∧ D2))

theorem Ding_reads_Romance_of_the_Three_Kingdoms : Ding = Romance_of_the_Three_Kingdoms := by
  sorry

end Ding_reads_Romance_of_the_Three_Kingdoms_l511_511724


namespace possible_values_of_draws_needed_l511_511849

noncomputable theory

-- Definitions based on the conditions
def num_red_balls := 6
def num_white_balls := 5
def total_balls := num_red_balls + num_white_balls

-- Definition of the random variable ξ
def draws_needed_until_white_ball (ξ : ℕ) : Prop :=
ξ ≥ 1 ∧ ξ ≤ num_white_balls + 1

-- Goal: Prove that ξ is in {1, 2, ..., 7} under the given conditions
theorem possible_values_of_draws_needed :
  ∀ (ξ : ℕ), ξ ≥ 1 → ξ ≤ total_balls → draws_needed_until_white_ball ξ :=
begin
  intros ξ h1 h2,
  split,
  exact h1,
  linarith [num_red_balls, num_white_balls],
end

end possible_values_of_draws_needed_l511_511849


namespace grazing_months_l511_511885

theorem grazing_months :
  ∀ (m : ℕ),
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let rent := 210
  let c_share := 54
  let a_ox_months := a_oxen * a_months
  let b_ox_months := b_oxen * b_months
  let c_ox_months := c_oxen * m
  let total_ox_months := a_ox_months + b_ox_months + c_ox_months
  let c_part := (c_ox_months : ℝ) / (total_ox_months : ℝ) * rent
  (c_part = c_share) → m = 3 :=
by { sorry }

end grazing_months_l511_511885


namespace sum_differences_1_to_999_l511_511589

-- Define a utility function to compute the first digit of a number
def first_digit (n : ℕ) : ℕ :=
if n < 10 then n else first_digit (n / 10)

-- Define a utility function to compute the last digit of a number
def last_digit (n : ℕ) : ℕ :=
n % 10

-- Define the operation performed by Damir
def difference (n : ℕ) : ℤ :=
(first_digit n : ℤ) - (last_digit n : ℤ)

-- Define the sum of all differences from 1 to 999
def sum_differences : ℤ :=
(1).to (999).sum (λ n, difference n)

-- State the main theorem to be proved using the previous definitions
theorem sum_differences_1_to_999 : sum_differences = 495 :=
sorry

end sum_differences_1_to_999_l511_511589


namespace warehouse_capacity_l511_511015

theorem warehouse_capacity (total_bins num_20_ton_bins cap_20_ton_bin cap_15_ton_bin : Nat) 
  (h1 : total_bins = 30) 
  (h2 : num_20_ton_bins = 12) 
  (h3 : cap_20_ton_bin = 20) 
  (h4 : cap_15_ton_bin = 15) : 
  total_bins * cap_20_ton_bin + (total_bins - num_20_ton_bins) * cap_15_ton_bin = 510 := 
by
  sorry

end warehouse_capacity_l511_511015


namespace distance_between_points_l511_511550

-- Define the points
def pointA : ℝ × ℝ := (0, 12)
def pointB : ℝ × ℝ := (9, 0)

-- Define the distance formula
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distance_between_points :
  dist pointA pointB = 15 :=
by 
  -- Here we would provide the proof, but we skip it as per the instructions.
  sorry

end distance_between_points_l511_511550


namespace problem1_problem2_l511_511459

-- Problem 1
theorem problem1 :
  (27 / 8) ^ (-2 / 3) + Real.pi ^ (Real.log 1 / Real.log 10) 
  + Real.log 2 (2 / 3) - Real.log 4 (16 / 9) = 4 / 9 :=
by
  sorry

-- Problem 2
theorem problem2 (a : ℝ) (ha : a ^ (1 / 2) + a ^ (-1 / 2) = 3) :
  (a ^ 3 + a ^ (-3)) / (a + a ^ (-1)) = 46 :=
by
  sorry

end problem1_problem2_l511_511459


namespace correct_option_l511_511036

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

def option_A (x : ℝ) : ℝ := -1 / x
def option_B (x : ℝ) : ℝ := x
def option_C (x : ℝ) : ℝ := Real.log (|x - 1|)
def option_D (x : ℝ) : ℝ := -Real.sin x

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  is_odd_function f ∧ is_decreasing_on f 0 1

theorem correct_option :
  satisfies_conditions option_D ∧
  ¬satisfies_conditions option_A ∧
  ¬satisfies_conditions option_B ∧
  ¬satisfies_conditions option_C :=
by
  sorry

end correct_option_l511_511036


namespace mass_of_substance_l511_511829

-- The conditions
def substance_density (mass_cubic_meter_kg : ℝ) (volume_cubic_meter_cm3 : ℝ) : Prop :=
  mass_cubic_meter_kg = 100 ∧ volume_cubic_meter_cm3 = 1*1000000

def specific_amount_volume_cm3 (volume_cm3 : ℝ) : Prop :=
  volume_cm3 = 10

-- The Proof Statement
theorem mass_of_substance (mass_cubic_meter_kg : ℝ) (volume_cubic_meter_cm3 : ℝ) (volume_cm3 : ℝ) (mass_grams : ℝ) :
  substance_density mass_cubic_meter_kg volume_cubic_meter_cm3 →
  specific_amount_volume_cm3 volume_cm3 →
  mass_grams = 10 :=
by
  intros hDensity hVolume
  sorry

end mass_of_substance_l511_511829


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511183

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the vowels
def vowels := {'A', 'E'}

-- Define the number of n-length words constructible from a set of letters
def num_words (n : ℕ) (alphabet : Set Char) : ℕ :=
  (alphabet.size ^ n)

-- The total number of 5-letter words (unrestricted)
def total_words := num_words 5 letters

-- The number of 5-letter words with no vowels
def no_vowel_words := num_words 5 (letters \ vowels)

-- The number of 5-letter words with at least one vowel
def at_least_one_vowel_words := total_words - no_vowel_words

-- The statement to prove that the number of 5-letter words with at least one vowel is 6752
theorem number_of_5_letter_words_with_at_least_one_vowel : 
  at_least_one_vowel_words = 6752 :=
by 
  -- Proof will be provided here
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l511_511183


namespace solve_for_x_l511_511274

theorem solve_for_x (x : ℝ) (h : x ≠ 0) (h1 : 9/x^2 = x/25) : x = real.cbrt 225 :=
by
  sorry

end solve_for_x_l511_511274


namespace five_letter_words_with_vowel_l511_511198

theorem five_letter_words_with_vowel : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let n := 5 
  (∃ (w : list(char)), w.length = n ∧ ∀ (i : fin n), w[i] ∈ letters ∧ (∃ (j : fin n), w[j] ∈ vowels)) → 
  (6^5 - 4^5 = 6752) := 
by
  sorry

end five_letter_words_with_vowel_l511_511198


namespace exists_multiple_of_prime_with_all_nines_digits_l511_511798

theorem exists_multiple_of_prime_with_all_nines_digits (p : ℕ) (hp_prime : Nat.Prime p) (h2 : p ≠ 2) (h5 : p ≠ 5) :
  ∃ n : ℕ, (∀ d ∈ (n.digits 10), d = 9) ∧ p ∣ n :=
by
  sorry

end exists_multiple_of_prime_with_all_nines_digits_l511_511798


namespace first_part_second_part_l511_511303

variable {n : ℕ} {m : ℕ}
variables {d : ℕ → ℕ}

-- Conditions
def is_positive (n : ℕ) : Prop := n > 0
def is_divisors (n : ℕ) (d : ℕ → ℕ) : Prop :=
∀ i j, 1 ≤ i → i < j → j ≤ 2 * m → (d i ∣ n ∧ d j ∣ n ∧ d i < d j)
def is_not_perfect_square (n : ℕ) : Prop :=
∀ (k : ℕ), k * k ≠ n
def determinant (matrix : list (list ℕ)) : ℕ := sorry -- definition needed

-- First part: Prove that n^m divides D
theorem first_part (hn : is_positive n)
                (hd : is_divisors n d)
                (hnps : is_not_perfect_square n)
                (D : ℕ) 
                (hD : D = determinant (list.map (λ i, list.map (λ j, if i = j then n + d i else n) (list.range (2 * m + 1))) (list.range (2 * m + 1)))) :
                n^m ∣ D := sorry

-- Second part: Prove that 1 + d_1 + d_2 + ... + d_{2m} divides D
theorem second_part (hn : is_positive n)
                  (hd : is_divisors n d)
                  (hnps : is_not_perfect_square n)
                  (D : ℕ) 
                  (hD : D = determinant (list.map (λ i, list.map (λ j, if i = j then n + d i else n) (list.range (2 * m + 1))) (list.range (2 * m + 1)))) :
                  (1 + (list.sum (list.map d (list.range (2 * m + 1)))) ∣ D) := sorry

end first_part_second_part_l511_511303


namespace fried_chicken_cost_l511_511942

theorem fried_chicken_cost (entry_fee : ℕ) (num_friends : ℕ) (rounds : ℕ) (drink_cost : ℕ) (num_drinks_self : ℕ) (tip_rate: ℚ) (total_spent : ℚ) : 
  entry_fee = 20 →
  num_friends = 5 →
  rounds = 2 →
  drink_cost = 6 →
  num_drinks_self = 6 →
  tip_rate = 0.3 →
  total_spent = 163 →
  let total_drinks := (num_friends * rounds) + num_drinks_self in
  let total_drink_cost := total_drinks * drink_cost in
  ∃ (F : ℚ), total_spent = entry_fee + total_drink_cost + F + tip_rate * (total_drink_cost + F) ∧ F = 14 :=
by
  intros;
  let total_drinks := (num_friends * rounds) + num_drinks_self;
  let total_drink_cost := total_drinks * drink_cost;
  use 14;
  sorry

end fried_chicken_cost_l511_511942


namespace even_function_periodic_f_neg_5_div_2_equals_half_l511_511751

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
  else if 1 < x ∧ x < 2 then 2 * (x - 1) * (1 - (x - 1))
  else if -1 ≤ x ∧ x < 0 then 2 * (-x) * (1 - (-x))
  else if -2 < x ∧ x < -1 then 2 * ((-x) - 1) * (1 - ((-x) - 1))
  else 0

theorem even_function_periodic_f_neg_5_div_2_equals_half :
  (∀ x : ℝ, f(-x) = f(x)) →
  (∀ x : ℝ, f(x + 2) = f(x)) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = 2 * x * (1 - x)) →
  f(-5/2) = 1/2 :=
by
  -- Proof not required
  sorry

end even_function_periodic_f_neg_5_div_2_equals_half_l511_511751


namespace distance_between_foci_of_ellipse_l511_511501

theorem distance_between_foci_of_ellipse :
  ∃ (a b c : ℝ), 
    (ellipse_tangent_x_axis (6, 0)) ∧ 
    (ellipse_tangent_y_axis (0, 3)) ∧ 
    (ellipse_axes_parallel_to_coordinate_axes) →
    (2 * c = 6 * Real.sqrt 3) :=
by
  sorry

end distance_between_foci_of_ellipse_l511_511501


namespace seating_arrangements_l511_511283

theorem seating_arrangements (n m : ℕ) (h_n : n = 9) (h_m : m = 4) : 
  let arrangements := 24 in
  (number_of_seating_arrangements_given_conditions n m) = arrangements := sorry

end seating_arrangements_l511_511283


namespace sum_of_differences_l511_511626

theorem sum_of_differences : 
  (∑ n in Finset.range 1000, let first_digit := n / 10 ^ (Nat.log10 n) in
                             let last_digit := n % 10 in
                             first_digit - last_digit) = 495 :=
by
  sorry

end sum_of_differences_l511_511626


namespace sum_of_squares_consec_ints_l511_511799

theorem sum_of_squares_consec_ints (N : ℕ) (hN : N = 1984) (a : ℕ) :
  ¬ ∃ k : ℕ, (∑ i in range N, (a + i + 1) ^ 2) = k ^ 2 :=
by sorry

end sum_of_squares_consec_ints_l511_511799


namespace solve_equation_one_solve_equation_two_l511_511366

theorem solve_equation_one (x : ℝ) : 3 * x + 7 = 32 - 2 * x → x = 5 :=
by
  intro h
  sorry

theorem solve_equation_two (x : ℝ) : (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1 → x = -1 :=
by
  intro h
  sorry

end solve_equation_one_solve_equation_two_l511_511366


namespace balls_in_boxes_l511_511697

theorem balls_in_boxes : ∃ (ways : ℕ), ways = 104 ∧
  let balls := 7;
      boxes := 4 in
  -- Here should be the formal definition of the number of ways to distribute the balls into the boxes,
  -- but we state it as an existential statement acknowledging the result.
  ways = (∑ p in (finset.powerset len_le_boxes (univ (finset.range balls + 1))),
               if (∑ x in p, x) = balls then multinomial p else 0) := sorry

end balls_in_boxes_l511_511697


namespace a_120_eq_20_div_41_l511_511922

noncomputable def a : ℕ → ℝ
| 0          := 1
| 1          := 1 / 2
| (n + 2)    := (1 - a (n + 1)) / (2 * a n)

theorem a_120_eq_20_div_41 : a 119 = 20 / 41 := by
  sorry

end a_120_eq_20_div_41_l511_511922


namespace square_side_length_l511_511661

/-- Given triangle ABC with side \( BC \) denoted as \( a \) and altitude from \( A \) to \( BC \) denoted as \( h_a \). Points \( D \) and \( E \) lie on segment \( BC \), and points \( F \) and \( H \) lie on segments \( CA \) and \( AB \) respectively, forming a square \( DEFH \). Prove that the side length \( DE \) of the square \( DEFH \) is \( \frac{a h_a}{a + h_a} \). -/
theorem square_side_length
  (A B C D E F H : Point)
  (a h_a DE : ℝ)
  (h₁ : collinear A B C)
  (h₂ : collinear D E C)
  (h₃ : collinear F A C)
  (h₄ : collinear H A B)
  (h₅ : quadrilateral DEFH)
  (h₆ : is_square DEFH)
  (ha  : BC = a)
  (hha : height A BC = h_a) :
  DE = a * h_a / (a + h_a) := sorry

end square_side_length_l511_511661


namespace find_angle_A_correct_l511_511288

noncomputable def find_angle_A (BC AB angleC : ℝ) : ℝ :=
if BC = 3 ∧ AB = Real.sqrt 6 ∧ angleC = Real.pi / 4 then
  Real.pi / 3
else
  sorry

theorem find_angle_A_correct : find_angle_A 3 (Real.sqrt 6) (Real.pi / 4) = Real.pi / 3 :=
by
  -- proof goes here
  sorry

end find_angle_A_correct_l511_511288


namespace track_length_l511_511515

-- Defining the conditions
variables (x : ℝ) (b s : ℝ → ℝ)
  (h1 : b(0) = 0)  -- Brenda's initial condition
  (h2 : s(0) = x / 2)  -- Sally's initial condition at the diametrically opposite point
  (b_speed : b(1) - b(0) = b(1))  -- Brenda's speed is constant
  (s_speed : s(1) - s(0) = s(1) - x / 2)  -- Sally's speed is also constant

-- Given conditions
def condition1 : Prop := b(1) = 120
def condition2 : Prop := s(1) = x / 2 - 120
def condition3 : Prop := s(2) = x / 2 + 60
def condition4 : Prop := b(2) = x / 2 - 60
def condition5 : Prop := (b(2) - b(1)) / (s(2) - s(1)) = 180 / (x / 2 - 120)

-- The proof problem
theorem track_length (h_cond1 : condition1)
                     (h_cond2 : condition2)
                     (h_cond3 : condition3)
                     (h_cond4 : condition4)
                     (h_cond5 : condition5) : 
  x = 480 := 
sorry

end track_length_l511_511515


namespace regular_polygons_intersections_l511_511359

-- Statement
theorem regular_polygons_intersections :
  let polygons_sides := [4, 7, 9, 10]
  (∀ p q ∈ polygons_sides, p ≠ q → ∃! pts ⊆ interior circle, |pts| = 70) := sorry

end regular_polygons_intersections_l511_511359


namespace number_of_elements_in_P_l511_511132

open Set

def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4, 5}
def P : Set ℕ := {x | ∃ a b, a ∈ M ∧ b ∈ N ∧ x = a + b}

theorem number_of_elements_in_P : Finite P ∧ (Finset.card (toFinset P) = 4) := by
  sorry

end number_of_elements_in_P_l511_511132


namespace range_of_x_l511_511266

-- Problem Statement
theorem range_of_x (x : ℝ) (h : 0 ≤ x - 8) : 8 ≤ x :=
by {
  sorry
}

end range_of_x_l511_511266


namespace largest_b_value_l511_511304

open Real

structure Triangle :=
(side_a side_b side_c : ℝ)
(a_pos : 0 < side_a)
(b_pos : 0 < side_b)
(c_pos : 0 < side_c)
(tri_ineq_a : side_a + side_b > side_c)
(tri_ineq_b : side_b + side_c > side_a)
(tri_ineq_c : side_c + side_a > side_b)

noncomputable def inradius (T : Triangle) : ℝ :=
  let s := (T.side_a + T.side_b + T.side_c) / 2
  let A := sqrt (s * (s - T.side_a) * (s - T.side_b) * (s - T.side_c))
  A / s

noncomputable def circumradius (T : Triangle) : ℝ :=
  let A := sqrt (((T.side_a + T.side_b + T.side_c) / 2) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_a) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_b) * ((T.side_a + T.side_b + T.side_c) / 2 - T.side_c))
  (T.side_a * T.side_b * T.side_c) / (4 * A)

noncomputable def condition_met (T1 T2 : Triangle) : Prop :=
  (inradius T1 / circumradius T1) = (inradius T2 / circumradius T2)

theorem largest_b_value :
  let T1 := Triangle.mk 8 11 11 (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  ∃ b > 0, ∃ T2 : Triangle, T2.side_a = b ∧ T2.side_b = 1 ∧ T2.side_c = 1 ∧ b = 14 / 11 ∧ condition_met T1 T2 :=
  sorry

end largest_b_value_l511_511304


namespace y_coordinates_product_l511_511351

theorem y_coordinates_product : 
  (∀ y, (∀ y, ((4, y) - (7, -3)) = 13) → ((y = -3 + 4 * real.sqrt 10) ∨ (y = -3 - 4 * real.sqrt 10))) 
    → (∏ y, ((-3 + 4 * real.sqrt 10) * (-3 - 4 * real.sqrt 10)) = -151) :=
by
  intros ycond yproduct
  sorry

end y_coordinates_product_l511_511351


namespace minimum_value_expression_l511_511749

theorem minimum_value_expression {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) : 
  a^2 + 4 * a * b + 9 * b^2 + 3 * b * c + c^2 ≥ 18 :=
by
  sorry

end minimum_value_expression_l511_511749


namespace fred_remaining_cards_l511_511098

/-- Given that Fred originally has 40 baseball cards 
and Keith buys 37.5% of them, show that after the purchase, 
Fred has 25 baseball cards remaining. -/
theorem fred_remaining_cards (original_cards : ℕ) (percentage_purchased : ℚ) (cards_remaining : ℕ) 
  (h1 : original_cards = 40) (h2 : percentage_purchased = 37.5) (h3 : cards_remaining = original_cards - (percentage_purchased / 100 * original_cards : ℕ)) :
cards_remaining = 25 :=
by
  rw [h1, h2] at h3 ⊢
  norm_num at h3
  exact h3

end fred_remaining_cards_l511_511098


namespace fish_market_customers_l511_511343

theorem fish_market_customers :
  let num_tuna := 10
  let weight_per_tuna := 200
  let weight_per_customer := 25
  let num_customers_no_fish := 20
  let total_tuna_weight := num_tuna * weight_per_tuna
  let num_customers_served := total_tuna_weight / weight_per_customer
  num_customers_served + num_customers_no_fish = 100 := 
by
  sorry

end fish_market_customers_l511_511343


namespace reflect_point_value_l511_511385

theorem reflect_point_value (mx b : ℝ) 
  (start end_ : ℝ × ℝ)
  (Hstart : start = (2, 3))
  (Hend : end_ = (10, 7))
  (Hreflection : ∃ m b: ℝ, (end_.fst, end_.snd) = 
              (2 * ((5 / 2) - (1 / 2) * 3 * m - b), 2 * ((5 / 2) + (1 / 2) * 3)) ∧ m = -2)
  : m + b = 15 :=
sorry

end reflect_point_value_l511_511385


namespace rayden_spent_more_l511_511807

-- Define the conditions
def lily_ducks := 20
def lily_geese := 10
def lily_chickens := 5
def lily_pigeons := 30

def rayden_ducks := 3 * lily_ducks
def rayden_geese := 4 * lily_geese
def rayden_chickens := 5 * lily_chickens
def rayden_pigeons := lily_pigeons / 2

def duck_price := 15
def geese_price := 20
def chicken_price := 10
def pigeon_price := 5

def lily_total := lily_ducks * duck_price +
                  lily_geese * geese_price +
                  lily_chickens * chicken_price +
                  lily_pigeons * pigeon_price

def rayden_total := rayden_ducks * duck_price +
                    rayden_geese * geese_price +
                    rayden_chickens * chicken_price +
                    rayden_pigeons * pigeon_price

def spending_difference := rayden_total - lily_total

theorem rayden_spent_more : spending_difference = 1325 := 
by 
  unfold spending_difference rayden_total lily_total -- to simplify the definitions
  sorry -- Proof is omitted

end rayden_spent_more_l511_511807


namespace grid_edge_number_assignment_possible_l511_511072

noncomputable def numberAssignmentPossible : Prop :=
  ∃ (f: fin 24 → ℕ), 
    ∀ p1 p2 : List ℕ, length p1 = 6 ∧ length p2 = 6 → 
    ((minimalPath p1) ∧ (minimalPath p2) → sum p1 = sum p2) 

constant minimalPath : List ℕ → Prop
  -- A placeholder definition for now, define later for the actual proof.
  -- It checks if a given path is one of the minimal paths in the 3x3 grid.

theorem grid_edge_number_assignment_possible : numberAssignmentPossible :=
  sorry  -- Proof will be provided later.

end grid_edge_number_assignment_possible_l511_511072


namespace ruler_decree_l511_511893

noncomputable def market_supply (P : ℝ) : ℝ := 6 * P - 312

noncomputable def market_demand (P : ℝ) : ℝ := 688 - 4 * P

def tax_revenue (t Q_d : ℝ) : ℝ := Q_d * t

def max_tax_revenue (t : ℝ) : ℝ := 288 * t - 2.4 * t^2

theorem ruler_decree :
  let t_opt := 60 in
  max_tax_revenue t_opt = 8640 :=
by
  -- Proving the correctness of the maximum tax revenue
  have h1 : deriv (max_tax_revenue t_opt) = 288 - 4.8 * t_opt := by
    sorry
  -- Solving derivative equal to zero for critical point
  have h2 : t_opt = (288 / 4.8) := by
    sorry
  -- Substitute t_opt into max_tax_revenue function to get maximum revenue
  have h3 : max_tax_revenue t_opt = 288 * 60 - 2.4 * 60 ^ 2 := by
    sorry
  -- Confirm the final tax revenue equals 8640
  show max_tax_revenue 60 = 8640 by sorry

end ruler_decree_l511_511893


namespace infinite_squares_form_l511_511760

theorem infinite_squares_form (k : ℕ) (hk : 0 < k) : ∃ f : ℕ → ℕ, ∀ n, ∃ a, a^2 = f n * 2^k - 7 :=
by
  sorry

end infinite_squares_form_l511_511760


namespace relationship_k_theta_minimum_k_l511_511993

-- Conditions
variables {θ k : ℝ}
def e1 := (1, 0)
def e2 := (0, 1)
def a := k * Math.sin θ • e1 + (2 - Math.cos θ) • e2
def b := e1 + e2

-- Parallel vectors condition
def parallel (u v : ℝ × ℝ) := ∃ λ : ℝ, u = λ • v

-- Main theorem
theorem relationship_k_theta (hθ : 0 < θ ∧ θ < π) (h_parallel : parallel a b) :
  k = (2 - Math.cos θ) / Math.sin θ :=
sorry

theorem minimum_k (hθ : 0 < θ ∧ θ < π) (h_parallel : parallel a b) :
  k = (2 - Math.cos θ) / Math.sin θ → k ≥ Real.sqrt 3 :=
sorry

end relationship_k_theta_minimum_k_l511_511993


namespace volume_multiplier_factor_l511_511450

variable (π r h : ℝ)

-- Original volume
def V_original : ℝ := π * r^2 * h

-- New radius and height
def r_new : ℝ := 2.5 * r
def h_new : ℝ := 3 * h

-- New volume
def V_new : ℝ := π * (r_new)^2 * h_new

-- Proof that the volume is multiplied by 18.75
theorem volume_multiplier_factor
  (h_is_positive : 0 < h)
  (r_is_positive : 0 < r) :
  V_new π r h = 18.75 * V_original π r h :=
by
  -- Volume calculation
  have V_original_def : V_original π r h = π * r^2 * h := rfl
  have V_new_def : V_new π r h = π * (2.5 * r)^2 * (3 * h) := rfl
  rw [V_original_def, V_new_def]
  -- Simplify
  calc
    π * (2.5 * r)^2 * (3 * h)
    = π * (2.5^2 * r^2) * (3 * h) : by ring
    ... = π * 6.25 * r^2 * 3 * h : by ring
    ... = 18.75 * π * r^2 * h : by ring
  sorry

end volume_multiplier_factor_l511_511450


namespace jack_additional_sweets_is_correct_l511_511853

/-- Initial number of sweets --/
def initial_sweets : ℕ := 22

/-- Sweets taken by Paul --/
def sweets_taken_by_paul : ℕ := 7

/-- Jack's total sweets taken --/
def jack_total_sweets_taken : ℕ := initial_sweets - sweets_taken_by_paul

/-- Half of initial sweets --/
def half_initial_sweets : ℕ := initial_sweets / 2

/-- Additional sweets taken by Jack --/
def additional_sweets_taken_by_jack : ℕ := jack_total_sweets_taken - half_initial_sweets

theorem jack_additional_sweets_is_correct : additional_sweets_taken_by_jack = 4 := by
  sorry

end jack_additional_sweets_is_correct_l511_511853


namespace sum_differences_1_to_999_l511_511590

-- Define a utility function to compute the first digit of a number
def first_digit (n : ℕ) : ℕ :=
if n < 10 then n else first_digit (n / 10)

-- Define a utility function to compute the last digit of a number
def last_digit (n : ℕ) : ℕ :=
n % 10

-- Define the operation performed by Damir
def difference (n : ℕ) : ℤ :=
(first_digit n : ℤ) - (last_digit n : ℤ)

-- Define the sum of all differences from 1 to 999
def sum_differences : ℤ :=
(1).to (999).sum (λ n, difference n)

-- State the main theorem to be proved using the previous definitions
theorem sum_differences_1_to_999 : sum_differences = 495 :=
sorry

end sum_differences_1_to_999_l511_511590


namespace sin_double_angle_cos_sum_angle_tan_sum_angle_l511_511134

variable (θ : ℝ) (h₁ : cos θ = 4 / 5) (h₂ : 0 < θ ∧ θ < π / 2)

theorem sin_double_angle : sin (2 * θ) = 24 / 25 := by
  sorry

theorem cos_sum_angle : cos (θ + π / 4) = sqrt 2 / 10 := by
  sorry

theorem tan_sum_angle : tan (θ + π / 4) = 7 := by
  sorry

end sin_double_angle_cos_sum_angle_tan_sum_angle_l511_511134


namespace geometric_sequence_third_fourth_terms_l511_511141

theorem geometric_sequence_third_fourth_terms
  (a : ℕ → ℝ)
  (r : ℝ)
  (ha : ∀ n, a (n + 1) = r * a n)
  (hS2 : a 0 + a 1 = 3 * a 1) :
  (a 2 + a 3) / (a 0 + a 1) = 1 / 4 :=
by
  -- proof to be filled in
  sorry

end geometric_sequence_third_fourth_terms_l511_511141


namespace ball_cost_l511_511370

theorem ball_cost (B C : ℝ) (h1 : 7 * B + 6 * C = 3800) (h2 : 3 * B + 5 * C = 1750) (hb : B = 500) : C = 50 :=
by
  sorry

end ball_cost_l511_511370


namespace ratio_peaches_to_oranges_l511_511848

theorem ratio_peaches_to_oranges (total_fruits : ℕ) (oranges : ℕ) (apples : ℕ) (peaches : ℕ)
    (h_total_fruits : total_fruits = 56)
    (h_oranges : oranges = (1/4) * total_fruits)
    (h_apples : apples = 35)
    (h_peaches_apples_ratio : apples = 5 * peaches) :
    (peaches : oranges) = 1 : 2 :=
by
  sorry

end ratio_peaches_to_oranges_l511_511848


namespace polyhedron_height_l511_511387

theorem polyhedron_height :
  let height := (-1 + Real.sqrt 5) / 2
  ∃ (P A B C D : ℝ³), 
  (∀ (X : ℝ³), (X = A ∨ X = B ∨ X = C ∨ X = D ∨ X = P) → 
    (∃ (Δ : ℝ³ × ℝ³ × ℝ³), Δ.fst ≠ Δ.snd ∧ Δ.snd ≠ Δ.third ∧ Δ.third ≠ Δ.fst ∧ 
      let (X₁, X₂, X₃) := Δ; X₁ = P → 
      ((X₂ = A ∧ X₃ = D) ∨ (X₂ = A ∧ X₃ = B) ∨ (X₂ = B ∧ X₃ = C) ∨ (X₂ = D ∧ X₃ = C) →
        (∃ (a : ℝ³), |X₁ - X₂| = 1 ∧ |X₁ - X₃| = 1 ∧ inner_product (X₁ - X₂) (X₁ - X₃) = 0))) :=
  (height = (-1 + Real.sqrt 5) / 2) ∧ height > 0 := 
  sorry

end polyhedron_height_l511_511387


namespace noah_yearly_call_cost_l511_511785

structure CallBilling (minutes_per_call : ℕ) (charge_per_minute : ℝ) (calls_per_week : ℕ) (weeks_in_year : ℕ) :=
  (total_minutes : ℕ := weeks_in_year * calls_per_week * minutes_per_call)
  (total_cost : ℝ := total_minutes * charge_per_minute)

theorem noah_yearly_call_cost :
  CallBilling 30 0.05 1 52 .total_cost = 78 := by
  sorry

end noah_yearly_call_cost_l511_511785


namespace three_digit_odd_numbers_l511_511561

theorem three_digit_odd_numbers: 
  let digits := {1, 3, 5, 7, 9} in
  {x : ℕ | 100 ≤ x ∧ x < 1000 ∧ (∀ i, i ∈ (x.digits 10) → i ∈ digits)}.card = 125 :=
by
  sorry

end three_digit_odd_numbers_l511_511561


namespace Chloe_second_round_points_l511_511525

theorem Chloe_second_round_points (P : ℤ) 
  (first_round_points : 40) 
  (points_lost : -4) 
  (total_points : 86) 
  (h : first_round_points + P + points_lost = total_points) : 
  P = 50 := 
by
  sorry

end Chloe_second_round_points_l511_511525


namespace cost_of_previous_hay_l511_511057

theorem cost_of_previous_hay
    (x : ℤ)
    (previous_hay_bales : ℤ)
    (better_quality_hay_cost : ℤ)
    (additional_amount_needed : ℤ)
    (better_quality_hay_bales : ℤ)
    (new_total_cost : ℤ) :
    previous_hay_bales = 10 ∧ 
    better_quality_hay_cost = 18 ∧ 
    additional_amount_needed = 210 ∧ 
    better_quality_hay_bales = 2 * previous_hay_bales ∧ 
    new_total_cost = better_quality_hay_bales * better_quality_hay_cost ∧ 
    new_total_cost - additional_amount_needed = 10 * x → 
    x = 15 := by
  sorry

end cost_of_previous_hay_l511_511057


namespace probability_of_AB_not_selected_l511_511638

-- The definition for the probability of not selecting both A and B 
def probability_not_selected : ℚ :=
  let total_ways := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial (4 - 2))
  let favorable_ways := 1 -- Only the selection of C and D
  favorable_ways / total_ways

-- The theorem stating the desired probability
theorem probability_of_AB_not_selected : probability_not_selected = 1 / 6 :=
by
  sorry

end probability_of_AB_not_selected_l511_511638


namespace probability_coprime_l511_511424

open BigOperators

theorem probability_coprime (A : Finset ℕ) (h : A = {1, 2, 3, 4, 5, 6, 7, 8}) :
  let pairs := { (a, b) ∈ (A ×ˢ A) | a < b }
  let coprime_pairs := pairs.filter (λ p, Nat.gcd p.1 p.2 = 1)
  coprime_pairs.card / pairs.card = 5 / 7 := by 
sorry

end probability_coprime_l511_511424


namespace five_letter_words_with_vowel_l511_511248

theorem five_letter_words_with_vowel :
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  total_words - words_no_vowel = 6752 :=
by
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  have h1 : total_words = 7776 := by norm_num
  have h2 : words_no_vowel = 1024 := by norm_num
  show total_words - words_no_vowel = 6752
  calc
    total_words - words_no_vowel
    = 7776 - 1024 : by rw [h1, h2]
    ... = 6752 : by norm_num

end five_letter_words_with_vowel_l511_511248


namespace sum_of_differences_l511_511623

theorem sum_of_differences : 
  (∑ n in Finset.range 1000, let first_digit := n / 10 ^ (Nat.log10 n) in
                             let last_digit := n % 10 in
                             first_digit - last_digit) = 495 :=
by
  sorry

end sum_of_differences_l511_511623


namespace trig_expression_simplification_l511_511825

theorem trig_expression_simplification :
  ∃ a b : ℕ, 
  0 < b ∧ b < 90 ∧ 
  (1000 * Real.sin (10 * Real.pi / 180) * Real.cos (20 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) = ↑a * Real.sin (b * Real.pi / 180)) ∧ 
  (100 * a + b = 12560) :=
sorry

end trig_expression_simplification_l511_511825


namespace mark_visits_20_households_each_day_l511_511340

noncomputable def households_visited_each_day (H : ℕ) : Prop :=
  (5 * ((H / 2) * 40) = 2000)

theorem mark_visits_20_households_each_day :
  ∃ H : ℕ, households_visited_each_day H :=
begin
  use 20,
  unfold households_visited_each_day,
  norm_num
end

end mark_visits_20_households_each_day_l511_511340


namespace range_of_k_l511_511662

theorem range_of_k (k : ℝ) : (∃ x ∈ Icc 2 5, log 2 (x - 1) + k - 1 = 0) ↔ -1 ≤ k ∧ k ≤ 1 :=
by
  sorry

end range_of_k_l511_511662


namespace sin_angle_dae_l511_511725

theorem sin_angle_dae (A B C D E : ℝ × ℝ) (h_eq_triangle : equilateral_triangle A B C)
  (h_quarters : ∃ t : ℝ, t = 1/4 ∧ (D = B + t • (C - B)) ∧ (E = D + t • (C - D))) :
  sin (angle D A E) = 1 / 2 :=
by sorry

end sin_angle_dae_l511_511725


namespace five_letter_words_with_vowels_l511_511226

/-
How many 5-letter words with at least one vowel can be constructed from the letters 
A, B, C, D, E, and F? (Note that A and E are vowels, any word is valid, not just English language 
words, and letters may be used more than once.)
-/

theorem five_letter_words_with_vowels :
  let letters := ['A', 'B', 'C', 'D', 'E', 'F'],
      vowels := ['A', 'E'],
      consonants := ['B', 'C', 'D', 'F'] in
  let total_words := 6 ^ 5,
      consonant_only_words := 4 ^ 5,
      at_least_one_vowel_words := total_words - consonant_only_words in
  at_least_one_vowel_words = 6752 :=
by
  intro letters vowels consonants total_words consonant_only_words at_least_one_vowel_words
  sorry

end five_letter_words_with_vowels_l511_511226


namespace five_letter_words_with_vowel_l511_511243

-- Define the set of letters and identify vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels := {'A', 'E'}

-- Define the problem statement
theorem five_letter_words_with_vowel : 
  (number of 5-letter words with at least one vowel) = 6752 := 
sorry

end five_letter_words_with_vowel_l511_511243


namespace sum_first_100_terms_l511_511736

def sequence (a : ℕ → ℝ) := 
  a 1 = 1 ∧
  a 2 = 2 ∧
  ∀ n : ℕ, 1 ≤ n → a (n+2) - a n = 1 + (-1)^n

theorem sum_first_100_terms :
  (∃ a : ℕ → ℝ, sequence a) →
  (∑ i in range 1 101, (λ a, a 1 + a i) = 2600) :=
by 
  sorry

end sum_first_100_terms_l511_511736


namespace algebraic_expression_evaluation_l511_511114

theorem algebraic_expression_evaluation (x : ℝ) (h : x^2 + x - 3 = 0) : x^3 + 2 * x^2 - 2 * x + 2 = 5 :=
by
  sorry

end algebraic_expression_evaluation_l511_511114


namespace min_abs_value_of_expression_l511_511090

theorem min_abs_value_of_expression (m n : ℕ) : ∃ (m = 2014^3 * 2015) (n = 2014^4 * 2015), abs (2015 * m^5 - 2014 * n^4) = 0 :=
by
  -- variables definition
  let m := 2014^3 * 2015,
  let n := 2014^4 * 2015,
  sorry

end min_abs_value_of_expression_l511_511090


namespace maxwell_sister_age_proof_l511_511717

def maxwell_age_now : ℕ := 6
def sister_age_now : ℕ := 2
def in_two_years (age: ℕ) : ℕ := age + 2

theorem maxwell_sister_age_proof (s : ℕ) : 
  sister_age_now = s → 
  in_two_years(maxwell_age_now) = 2 * in_two_years(s) → 
  s = 2 :=
by
  intros h₁ h₂
  rw [in_two_years, in_two_years] at h₂
  simp at h₂
  rw h₁
  assumption

end maxwell_sister_age_proof_l511_511717


namespace total_bedrooms_l511_511041

theorem total_bedrooms : 
  (total_rooms : ℕ) (total_rooms = 50) → 
  (percent_bedrooms : ℕ) (percent_bedrooms = 40) →
  (second_floor_bedrooms : ℕ) (second_floor_bedrooms = 8) →
  (third_floor_bedrooms : ℕ) (third_floor_bedrooms = 6) →
  (fourth_floor_bedrooms : ℕ) (fourth_floor_bedrooms = 12) →
  (fifth_floor_bedrooms : ℕ) (fifth_floor_bedrooms = 2) →
  let total_bedrooms = (0.4 : ℝ) * total_rooms in total_bedrooms = 20 :=
by
  sorry

end total_bedrooms_l511_511041


namespace apex_angle_of_equal_cones_l511_511789

-- Represent the existence of three equal cones with a common vertex
-- and the condition that each cone touches the other two adjacent cones
structure equal_cones_on_plane_with_common_vertex (α : Type) :=
(p : α)
(cone1 cone2 cone3 : α)
(touch : α → α → Prop)
(common_vertex : ∀ (c : α), c = p)
(equal_cones : ∀ (c1 c2 c3 : α), c1 = c2 ∧ c2 = c3)
(touching_adjacency : (touch cone1 cone2) ∧ (touch cone2 cone3) ∧ (touch cone3 cone1))

-- Given the above conditions, state the theorem
theorem apex_angle_of_equal_cones (α : Type) [equal_cones_on_plane_with_common_vertex α] :
  ∃ (θ : ℝ), θ = Real.arccos (1 / 7) := 
sorry

end apex_angle_of_equal_cones_l511_511789


namespace fastest_growth_in_interval_l511_511037

theorem fastest_growth_in_interval :
  (∀ x > 0, (2^x > x^2) ∧ (2^x > log x) ∧ (2^x > 2)) :=
by
  sorry

end fastest_growth_in_interval_l511_511037


namespace dhoni_toys_average_cost_l511_511073

theorem dhoni_toys_average_cost (A : ℝ) (h1 : ∃ x1 x2 x3 x4 x5, (x1 + x2 + x3 + x4 + x5) / 5 = A)
  (h2 : 5 * A = 5 * A)
  (h3 : ∃ x6, x6 = 16)
  (h4 : (5 * A + 16) / 6 = 11) : A = 10 :=
by
  sorry

end dhoni_toys_average_cost_l511_511073


namespace product_remainder_l511_511873

theorem product_remainder (a b c : ℕ) (h1: a = 2456) (h2: b = 7294) (h3: c = 91803) : 
  ((a * b * c) % 10) = 2 :=
by
  rw [h1, h2, h3]
  -- The proof steps are omitted as they are not required
  sorry

end product_remainder_l511_511873


namespace tenth_permutation_is_6785_l511_511838

def digits : List ℕ := [5, 6, 7, 8]

def tenth_permutation : ℕ := 6785

theorem tenth_permutation_is_6785 :
  let permutations := List.permutations digits
  let ordered_permutations := List.sort (≤) permutations
  let tenth := ordered_permutations.get! 9
  List.to_nat tenth = tenth_permutation :=
by
  sorry

end tenth_permutation_is_6785_l511_511838


namespace Reflection_in_y_axis_maps_CD_to_C_D_l511_511348

theorem Reflection_in_y_axis_maps_CD_to_C_D:
  ∀ (C D : ℝ × ℝ) (C' D' : ℝ × ℝ),
  C = (-3, 2) → 
  D = (-4, -2) → 
  C' = (3, 2) → 
  D' = (4, -2) → 
  (∀ (p : ℝ × ℝ), (p.1, p.2) = (-p.1, p.2)) (C.1, C.2) = (C'.1, C'.2) ∧ 
  (p.1, p.2) = (-p.1, p.2) (D.1, D.2) = (D'.1, D'.2)
:= sorry

end Reflection_in_y_axis_maps_CD_to_C_D_l511_511348


namespace meal_cost_l511_511742

theorem meal_cost (total_paid change tip_rate : ℝ)
  (h_total_paid : total_paid = 20 - change)
  (h_change : change = 5)
  (h_tip_rate : tip_rate = 0.2) :
  ∃ x, x + tip_rate * x = total_paid ∧ x = 12.5 := 
by
  sorry

end meal_cost_l511_511742


namespace distribute_students_l511_511722

theorem distribute_students (n : ℕ) (teams : fin 500 → fin 10 → fin n) :
  ∃ (room_assignment : fin n → bool), ∀ t, ∃ s₁ s₂, (s₁ ≠ s₂) ∧ (room_assignment (teams t s₁) ≠ room_assignment (teams t s₂)) :=
sorry

end distribute_students_l511_511722


namespace additional_length_of_track_l511_511919

theorem additional_length_of_track (rise : ℝ) (grade1 grade2 : ℝ) (h_rise : rise = 800) (h_grade1 : grade1 = 0.04) (h_grade2 : grade2 = 0.02) :
  (rise / grade2) - (rise / grade1) = 20000 :=
by
  sorry

end additional_length_of_track_l511_511919


namespace sum_of_differences_l511_511577

/-- 
  For each natural number from 1 to 999, Damir subtracts the last digit from the first digit and 
  writes the resulting differences on a board. We are to prove that the sum of all these differences 
  is 495.
-/
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, (first_digit n - last_digit n)) = 495 :=
sorry

/-- 
  Helper function to get the first digit of a natural number.
  Here, n > 0
-/
def first_digit (n : ℕ) : ℕ :=
  n / 10^(n.digits 10 - 1)

/-- 
  Helper function to get the last digit of a natural number.
  Here, n > 0
-/
def last_digit (n : ℕ) : ℕ :=
  n % 10

end sum_of_differences_l511_511577


namespace sum_of_differences_l511_511596

theorem sum_of_differences : 
  let first_digit (n : ℕ) : ℕ := n / 10^(nat.log10 n)
  let last_digit (n : ℕ) : ℕ := n % 10
  (finset.range 1000).sum (λ n, first_digit n - last_digit n) = 495 :=
by
  sorry

end sum_of_differences_l511_511596


namespace reciprocal_of_fraction_sum_l511_511841

theorem reciprocal_of_fraction_sum : (4⁻¹ + 6⁻¹ - 12⁻¹)⁻¹ = 3 := by
  sorry

end reciprocal_of_fraction_sum_l511_511841


namespace reasoning_is_invalid_l511_511164

-- Definitions based on conditions
variables {Line Plane : Type} (is_parallel_to : Line → Plane → Prop) (is_parallel_to' : Line → Line → Prop) (is_contained_in : Line → Plane → Prop)

-- Conditions
axiom major_premise (b : Line) (α : Plane) : is_parallel_to b α → ∀ (a : Line), is_contained_in a α → is_parallel_to' b a
axiom minor_premise1 (b : Line) (α : Plane) : is_parallel_to b α
axiom minor_premise2 (a : Line) (α : Plane) : is_contained_in a α

-- Conclusion
theorem reasoning_is_invalid : ∃ (a : Line) (b : Line) (α : Plane), ¬ (is_parallel_to b α → ∀ (a : Line), is_contained_in a α → is_parallel_to' b a) :=
sorry

end reasoning_is_invalid_l511_511164


namespace five_letter_words_with_vowel_l511_511237

-- Define the set of letters and identify vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels := {'A', 'E'}

-- Define the problem statement
theorem five_letter_words_with_vowel : 
  (number of 5-letter words with at least one vowel) = 6752 := 
sorry

end five_letter_words_with_vowel_l511_511237


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511229

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let total_words := 6^5
  let words_without_vowels := 4^5
  total_words - words_without_vowels = 6752 :=
by
  let total_words := 6^5
  let words_without_vowels := 4^5
  have h_total_words : total_words = 7776 := by norm_num
  have h_words_without_vowels : words_without_vowels = 1024 := by norm_num
  calc
    7776 - 1024 = 6752 : by norm_num

end number_of_5_letter_words_with_at_least_one_vowel_l511_511229


namespace total_paint_area_l511_511769

def room_width := 20 -- feet
def room_length := 20 -- feet
def room_height := 8 -- feet

def doorway1_width := 3 -- feet
def doorway1_height := 7 -- feet

def window_width := 6 -- feet
def window_height := 4 -- feet

def doorway2_width := 5 -- feet
def doorway2_height := 7 -- feet

theorem total_paint_area :
  let wall_area := 4 * (room_width * room_height),
      doorway1_area := doorway1_width * doorway1_height,
      window_area := window_width * window_height,
      doorway2_area := doorway2_width * doorway2_height in
  wall_area - (doorway1_area + window_area + doorway2_area) = 560 := 
by 
  unfold wall_area
  unfold doorway1_area
  unfold window_area
  unfold doorway2_area
  sorry

end total_paint_area_l511_511769


namespace hyperbola_no_intersection_l511_511325

theorem hyperbola_no_intersection (a b e : ℝ)
  (ha : 0 < a) (hb : 0 < b)
  (h_e : e = (Real.sqrt (a^2 + b^2)) / a) :
  (√5 ≥ e ∧ 1 < e) → ∀ x y : ℝ, ¬ (y = 2 * x ∧ (x^2 / a^2 - y^2 / b^2 = 1)) :=
begin
  intros h_intersect x y,
  sorry,
end

end hyperbola_no_intersection_l511_511325


namespace ways_to_divide_day_l511_511473

theorem ways_to_divide_day (n m : ℕ) (h1 : 86400 = 1440 * 60) (h2 : n > 0) (h3 : m > 0) (h4 : m ≤ 60) :
  (∃ (factors : list ℕ), factors = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 45, 48, 60] ∧ factors.length = 21) :=
sorry

end ways_to_divide_day_l511_511473


namespace smallest_n_terminating_decimal_l511_511874

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (n > 0) ∧
           (∃ (k: ℕ), (n + 150) = 2^k ∧ k < 150) ∨ 
           (∃ (k m: ℕ), (n + 150) = 2^k * 5^m ∧ m < 150) ∧ 
           ∀ m : ℕ, ((m > 0 ∧ (∃ (j: ℕ), (m + 150) = 2^j ∧ j < 150) ∨ 
           (∃ (j l: ℕ), (m + 150) = 2^j * 5^l ∧ l < 150)) → m ≥ n)
:= ⟨10, by {
  sorry
}⟩

end smallest_n_terminating_decimal_l511_511874


namespace largest_in_arithmetic_progression_l511_511362

theorem largest_in_arithmetic_progression (a d : ℝ) (n : ℕ) 
  (h1 : n = 7) 
  (seq : Fin 7 → ℝ) 
  (h_seq : ∀ i, seq i = a + (i - 3 : ℕ) * d)
  (h_sum_cubes : ∑ i in Finset.range 7, (seq ⟨i, Nat.lt_succ_self 6⟩) ^ 3 = 0)
  (h_sum_squares : ∑ i in Finset.range 7, (seq ⟨i, Nat.lt_succ_self 6⟩) ^ 2 = 756) :
  max (seq ⟨0, by norm_num⟩) (seq ⟨6, by norm_num⟩) = 9 * real.sqrt 3 := by
  sorry

end largest_in_arithmetic_progression_l511_511362


namespace angle_BAS_eq_angle_CAM_l511_511857

noncomputable theory -- to handle circumcircle centers and similar non-computable entities

open EuclideanGeometry -- open geometry-specific stuff

variables {A B C O P Q D M S : Point} -- points in the plane
variables [Circumcenter O A B C] -- specify O is the circumcenter of triangle ABC
variables [Altitude D A B C] -- specify AD as altitude of triangle ABC
variables (hA : acute_triangle A B C) (hB : scalene_triangle A B C) -- acute and scalene properties

-- existence of points P and Q which are intersections of lines through O
-- perpendicular to AB and AC with altitude AD
variables (hP : is_perpendicular O P A B) (hQ : is_perpendicular O Q A C)
variables (hP_alt : intersect_at P altitude AD) (hQ_alt : intersect_at Q altitude AD)

-- midpoint M of side BC
variables (hM : is_midpoint M B C)

-- S as the center of the circumcircle of triangle OPQ
variables [Circumcenter S O P Q]

theorem angle_BAS_eq_angle_CAM :
  ∠BAS = ∠CAM :=
sorry

end angle_BAS_eq_angle_CAM_l511_511857


namespace enthalpy_change_is_800_l511_511081

variables (NaH H2O NaOH H2 : Type) (bond_enthalpy : Type → ℤ)

-- Define bond enthalpy constants
def OH_bond_enthalpy := 463
def HH_bond_enthalpy := 432
def NaH_bond_enthalpy := 283
def NaO_bond_enthalpy := 377

-- Balanced chemical reaction
def reaction : Prop := 
  ∀ (NaH_moles H2O_moles NaOH_moles H2_moles : ℕ),
  NaH_moles = 2 ∧ H2O_moles = 2 ∧ NaOH_moles = 2 ∧ H2_moles = 2

-- Enthalpy change calculation
def calculate_enthalpy_change : ℤ :=
  let reactant_enthalpy := (2 * NaH_bond_enthalpy) + (2 * 2 * OH_bond_enthalpy) in
  let product_enthalpy := (2 * NaO_bond_enthalpy) + (2 * HH_bond_enthalpy) in
  reactant_enthalpy - product_enthalpy

theorem enthalpy_change_is_800 :
  reaction NaH H2O NaOH H2 ∧ calculate_enthalpy_change = 800 := by
  sorry

end enthalpy_change_is_800_l511_511081


namespace isosceles_triangle_angles_l511_511369

theorem isosceles_triangle_angles 
  (α r R : ℝ)
  (isosceles : α ∈ {β : ℝ | β = α})
  (circumference_relation : R = 3 * r) :
  (α = Real.arccos (1 / 2 + 1 / (2 * Real.sqrt 3)) ∨ 
   α = Real.arccos (1 / 2 - 1 / (2 * Real.sqrt 3))) ∧ 
  (
    180 = 2 * (Real.arccos (1 / 2 + 1 / (2 * Real.sqrt 3))) + 2 * α ∨
    180 = 2 * (Real.arccos (1 / 2 - 1 / (2 * Real.sqrt 3))) + 2 * α 
  ) :=
by sorry

end isosceles_triangle_angles_l511_511369


namespace largest_square_area_l511_511940

theorem largest_square_area (XY YZ XZ : ℝ)
  (h1 : XZ^2 = XY^2 + YZ^2)
  (h2 : XY^2 + YZ^2 + XZ^2 = 450) :
  XZ^2 = 225 :=
by
  sorry

end largest_square_area_l511_511940


namespace original_decimal_number_l511_511010

theorem original_decimal_number (x : ℝ) (h : 10 * x - x / 10 = 23.76) : x = 2.4 :=
sorry

end original_decimal_number_l511_511010


namespace minimize_sum_of_f_seq_l511_511111

def f (x : ℝ) : ℝ := x^2 - 8 * x + 10

def isArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem minimize_sum_of_f_seq
  (a : ℕ → ℝ)
  (h₀ : isArithmeticSequence a 1)
  (h₁ : a 1 = a₁)
  : f (a 1) + f (a 2) + f (a 3) = 3 * a₁^2 - 18 * a₁ + 30 →

  (∀ x, 3 * x^2 - 18 * x + 30 ≥ 3 * 3^2 - 18 * 3 + 30) →
  a₁ = 3 :=
by
  sorry

end minimize_sum_of_f_seq_l511_511111


namespace quadratic_distinct_roots_l511_511272

theorem quadratic_distinct_roots (p q₁ q₂ : ℝ) 
  (h_eq : p = q₁ + q₂ + 1) :
  q₁ ≥ 1/4 → 
  (∃ x, x^2 + x + q₁ = 0 ∧ ∃ x', x' ≠ x ∧ x'^2 + x' + q₁ = 0) 
  ∨ 
  (∃ y, y^2 + p*y + q₂ = 0 ∧ ∃ y', y' ≠ y ∧ y'^2 + p*y' + q₂ = 0) :=
by 
  sorry

end quadratic_distinct_roots_l511_511272


namespace unit_digit_product_l511_511875

theorem unit_digit_product (n1 n2 n3 : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) :
  (n1 = 68) ∧ (n2 = 59) ∧ (n3 = 71) ∧ (a = 3) ∧ (b = 6) ∧ (c = 7) →
  (a ^ n1 * b ^ n2 * c ^ n3) % 10 = 8 := by
  sorry

end unit_digit_product_l511_511875


namespace fishbowls_count_l511_511400

theorem fishbowls_count (total_fish : ℕ) (fish_per_bowl : ℕ) (h_fish : total_fish = 6003) (h_bowl : fish_per_bowl = 23) : total_fish / fish_per_bowl = 261 :=
by
  rw [h_fish, h_bowl]
  norm_num
  sorry

end fishbowls_count_l511_511400


namespace gcd_probability_is_one_l511_511421

open Set Nat

theorem gcd_probability_is_one :
  let S := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := (finset.powerset_len 2 (finset.image id S.to_finset)).card
  let non_rel_prime_pairs := 6
  (finset.card (finset.filter (λ (p : Finset ℕ), p.gcdₓ = 1) 
                                (finset.powerset_len 2 (finset.image id S.to_finset)))) / 
  total_pairs = 11 / 14 :=
sorry

end gcd_probability_is_one_l511_511421


namespace safe_code_count_l511_511042

theorem safe_code_count :
  let total_codes := 10^4
  let restricted_codes := 2 * 10^3
  let valid_codes := total_codes - restricted_codes
  valid_codes = 9900 :=
by
  let total_codes := 10^4
  let restricted_codes := 2 * 10^3
  let valid_codes := total_codes - restricted_codes
  show valid_codes = 9900, from sorry

end safe_code_count_l511_511042


namespace cylinder_volume_factor_l511_511447

theorem cylinder_volume_factor (h r : ℝ) :
  let V := π * r^2 * h in
  let V_prime := π * (2.5 * r)^2 * (3 * h) in
  V_prime = 18.75 * V :=
by
  sorry

end cylinder_volume_factor_l511_511447


namespace max_slope_of_OQ_l511_511675

-- Assuming the conditions and problem setup
theorem max_slope_of_OQ :
  let C := {p : ℝ // p = 2}
  let equation_C := ∀ (x y : ℝ), y^2 = 4 * x
  let F : ℝ × ℝ := (1, 0)
  let O : ℝ × ℝ := (0, 0)
  let P := ∀ (m n : ℝ), (10 * m - 9, 10 * n)
  let Q := ∀ (m n : ℝ), (m, n)
  let K : ℝ := ∀ (m n : ℝ), (10 * n) / (25 * n^2 + 9)
  max_slope_of_line_OQ : ∃ (K_max : ℝ), K_max = 1 / 3 :=
  sorry

end max_slope_of_OQ_l511_511675


namespace volume_of_tetrahedron_EFGH_l511_511729

-- define the conditions
def edge_length_EF : ℝ := 4
def area_EFG : ℝ := 20
def area_EFH : ℝ := 16
def angle_between_faces : ℝ := 45

-- define the problem statement
theorem volume_of_tetrahedron_EFGH :
  let EF := edge_length_EF
  let [EFG] := area_EFG
  let [EFH] := area_EFH
  let θ := angle_between_faces.to_rad in -- converting degrees to radians
  ∃ V : ℝ, V = (80 * Real.sqrt 2) / 3 :=
sorry

end volume_of_tetrahedron_EFGH_l511_511729


namespace five_letter_words_with_at_least_one_vowel_l511_511212

open Finset

theorem five_letter_words_with_at_least_one_vowel :
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := ((letters \ vowels).card : ℕ) ^ 5
  total_words - no_vowel_words = 6752 :=
by
  let letters := insert 'A' (insert 'B' (insert 'C' (insert 'D' (insert 'E' (singleton 'F')))))
  let vowels := insert 'A' (singleton 'E')
  let consonants := letters \ vowels
  have : letters.card = 6 := by simp
  have : vowels.card = 2 := by simp
  have : consonants.card = 4 := by simp
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := (consonants.card : ℕ) ^ 5
  calc
    total_words - no_vowel_words
        = 6^5 - 4^5         : by simp
    ... = 7776 - 1024       : by norm_num
    ... = 6752             : by norm_num

end five_letter_words_with_at_least_one_vowel_l511_511212


namespace triangle_XPY_area_l511_511298

theorem triangle_XPY_area
  (X Y Z P: Type)
  (M : Type)
  [is_triangle X Y Z M]
  (xm_21 : medians XM = 21)
  (yn_30 : medians YN = 30)
  (xy_30 : XY = 30)
  (yn_circum : YN ∩ circumcircle(X, Y, Z) = P) :
  area (triangle XP Y) = 238.45 := 
sorry

end triangle_XPY_area_l511_511298


namespace range_of_m_l511_511655

theorem range_of_m (a b m : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : (1 / a + 1 / b) * real.sqrt (a^2 + b^2) ≥ 2 * m - 4) : m ≤ 2 + real.sqrt 2 :=
  sorry

end range_of_m_l511_511655


namespace rabbit_catch_up_time_l511_511862

theorem rabbit_catch_up_time
  (rabbit_speed : ℝ)
  (cat_speed : ℝ)
  (head_start_minutes : ℝ)
  (head_start_hours : head_start_minutes / 60)
  (cat_head_start_distance : cat_speed * head_start_hours)
  (speed_difference : rabbit_speed - cat_speed)
  :
  (cat_head_start_distance / speed_difference) = 1 :=
by
  sorry

end rabbit_catch_up_time_l511_511862


namespace sum_of_series_l511_511949

theorem sum_of_series :
  (∑ n in finset.range 100, (2 + (n + 1) * 9) / 8 ^ (100 - n)) = 901 / 7 :=
sorry

end sum_of_series_l511_511949


namespace solution_proof_l511_511461

noncomputable def problem_statement : Prop :=
  let a : ℝ := 0.10
  let b : ℝ := 0.50
  let c : ℝ := 500
  a * (b * c) = 25

theorem solution_proof : problem_statement := by
  sorry

end solution_proof_l511_511461


namespace five_letter_words_with_vowel_l511_511252

theorem five_letter_words_with_vowel :
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  total_words - words_no_vowel = 6752 :=
by
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  have h1 : total_words = 7776 := by norm_num
  have h2 : words_no_vowel = 1024 := by norm_num
  show total_words - words_no_vowel = 6752
  calc
    total_words - words_no_vowel
    = 7776 - 1024 : by rw [h1, h2]
    ... = 6752 : by norm_num

end five_letter_words_with_vowel_l511_511252


namespace speed_of_stream_l511_511915

theorem speed_of_stream
  (v : ℝ)
  (h1 : ∀ t : ℝ, t = 7)
  (h2 : ∀ d : ℝ, d = 72)
  (h3 : ∀ s : ℝ, s = 21)
  : (72 / (21 - v) + 72 / (21 + v) = 7) → v = 3 :=
by
  intro h
  sorry

end speed_of_stream_l511_511915


namespace james_total_spent_l511_511301

noncomputable def total_cost : ℝ :=
  let milk_price := 3.0
  let bananas_price := 2.0
  let bread_price := 1.5
  let cereal_price := 4.0
  let milk_tax := 0.20
  let bananas_tax := 0.15
  let bread_tax := 0.10
  let cereal_tax := 0.25
  let milk_total := milk_price * (1 + milk_tax)
  let bananas_total := bananas_price * (1 + bananas_tax)
  let bread_total := bread_price * (1 + bread_tax)
  let cereal_total := cereal_price * (1 + cereal_tax)
  milk_total + bananas_total + bread_total + cereal_total

theorem james_total_spent : total_cost = 12.55 :=
  sorry

end james_total_spent_l511_511301


namespace sequence_rel_prime_to_m_l511_511306

theorem sequence_rel_prime_to_m (k : ℤ) (hk : k > 1) :
  ∃ a b : ℤ, ∀ n : ℕ, (x : ℤ) = fibonacci2_gen a b n → gcd x (4 * k^2 - 5) = 1 :=
by
  -- Definitions and setup
  let m := 4 * k^2 - 5
  have hm : m > 0 := sorry
  let x_seq := λ (a b : ℤ), λ n, nat.rec_on n a (λ n ih, (λ x1, b + x1) ih)
  -- Existence proof
  use 1, 3
  intro n
  sorry

end sequence_rel_prime_to_m_l511_511306


namespace max_min_z_in_region_l511_511978

open Real

noncomputable def bounding_region (x y : ℝ) : Prop := 
  (0 ≤ x) ∧ (y ≤ 2) ∧ (y ≥ (x^2 / 2))

noncomputable def z (x y : ℝ) : ℝ := 
  2 * x^3 - 6 * x * y + 3 * y^2

theorem max_min_z_in_region :
  ∃ c d : ℝ, 
    (∀ x y, bounding_region x y → z x y ≤ 12) ∧ 
    (∀ x y, bounding_region x y → z x y ≥ -1) ∧ 
    (∃ x y, bounding_region x y ∧ z x y = 12) ∧
    (∃ x y, bounding_region x y ∧ z x y = -1) :=
sorry

end max_min_z_in_region_l511_511978


namespace sum_of_differences_l511_511581

open Nat
open BigOperators

theorem sum_of_differences (n : ℕ) (h : n ≥ 1 ∧ n ≤ 999) : 
  let differences := (fun x => 
                        let first_digit := x / 10;
                        let last_digit := x % 10;
                        first_digit - last_digit) in
  ∑ i in (finset.range (n + 1)).filter (λ x, 1 ≤ x ∧ x ≤ 999), differences i = 495 :=
by
  -- Acknowledge the need for a more refined filtering criteria for numbers between 1 and 999
  sorry

end sum_of_differences_l511_511581


namespace union_sets_l511_511687

-- Define the sets A and B based on their conditions
def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 < x ∧ x < 9 }

-- Statement of the proof problem
theorem union_sets (x : ℝ) : (x ∈ A ∪ B) ↔ (x ∈ { x | -1 ≤ x ∧ x < 9 }) := sorry

end union_sets_l511_511687


namespace perpendicular_lines_b_l511_511535

theorem perpendicular_lines_b (b : ℝ) : 
  (∃ (k m: ℝ), k = 3 ∧ 2 * m + b * k = 14 ∧ (k * m = -1)) ↔ b = 2 / 3 :=
sorry

end perpendicular_lines_b_l511_511535


namespace minimum_value_trig_expression_l511_511086

noncomputable def min_value_trig_expr : ℝ :=
  let expr := λ x: ℝ, (Real.tan x + Real.cot x)^2 + (Real.sec x + Real.csc x)^2
  infi (λ x: ℝ, if (0 < x) ∧ (x < Real.pi / 2) then expr x else ∞)

theorem minimum_value_trig_expression :
  min_value_trig_expr = 12 :=
sorry

end minimum_value_trig_expression_l511_511086


namespace sum_of_differences_l511_511573

/-- 
  For each natural number from 1 to 999, Damir subtracts the last digit from the first digit and 
  writes the resulting differences on a board. We are to prove that the sum of all these differences 
  is 495.
-/
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, (first_digit n - last_digit n)) = 495 :=
sorry

/-- 
  Helper function to get the first digit of a natural number.
  Here, n > 0
-/
def first_digit (n : ℕ) : ℕ :=
  n / 10^(n.digits 10 - 1)

/-- 
  Helper function to get the last digit of a natural number.
  Here, n > 0
-/
def last_digit (n : ℕ) : ℕ :=
  n % 10

end sum_of_differences_l511_511573


namespace finish_lollipops_in_6_days_l511_511172

variables (henry_alison_diff : ℕ) (alison_lollipops : ℕ) (diane_alison_ratio : ℕ) (lollipops_eaten_per_day : ℕ)
variables (days_needed : ℕ) (henry_lollipops : ℕ) (diane_lollipops : ℕ) (total_lollipops : ℕ)

-- Conditions as definitions
def condition_1 : Prop := henry_alison_diff = 30
def condition_2 : Prop := alison_lollipops = 60
def condition_3 : Prop := alison_lollipops * 2 = diane_lollipops
def condition_4 : Prop := lollipops_eaten_per_day = 45

-- Total lollipops calculation
def total_lollipops_calculated : ℕ := alison_lollipops + diane_lollipops + henry_lollipops

-- Days to finish lollipops calculation
def days_needed_calculated : ℕ := total_lollipops / lollipops_eaten_per_day

-- The theorem to prove
theorem finish_lollipops_in_6_days :
  condition_1 →
  condition_2 →
  condition_3 →
  condition_4 →
  henry_lollipops = alison_lollipops + 30 →
  total_lollipops_calculated = 270 →
  days_needed_calculated = 6 :=
by {
  sorry
}

end finish_lollipops_in_6_days_l511_511172


namespace DS_eq_BP_l511_511927

noncomputable def parallelogram (A B C D : ℝ) := 
  (A - B) = (D - C) ∧ (A - D) = (B - C)

variables {A B C D X Y R Q P S : ℝ}

theorem DS_eq_BP 
  (h1 : parallelogram A B C D)
  (h2 : X ∈ segment A B)
  (h3 : Y ∈ segment C D)
  (h4 : R = intersection (line_through A Y) (line_through D X))
  (h5 : Q = intersection (line_through B Y) (line_through C X))
  (h6 : P = intersection (line_through Q R) (line_through B C))
  (h7 : S = intersection (line_through Q R) (line_through A D))
  : distance D S = distance B P :=
by sorry

end DS_eq_BP_l511_511927


namespace bacteria_growth_final_count_l511_511005

theorem bacteria_growth_final_count (initial_count : ℕ) (t : ℕ) 
(h1 : initial_count = 10) 
(h2 : t = 7) 
(h3 : ∀ n : ℕ, (n * 60) = t * 60 → 2 ^ n = 128) : 
(initial_count * 2 ^ t) = 1280 := 
by
  sorry

end bacteria_growth_final_count_l511_511005


namespace remainder_when_divided_by_x_minus_3_l511_511564

noncomputable def polynomial := Polynomial.CoeffRing

def p (x : polynomial) : polynomial := x^4 + x + 2

theorem remainder_when_divided_by_x_minus_3 (x : polynomial) :
  Polynomial.eval 3 (p x) = 86 :=
by
  sorry

end remainder_when_divided_by_x_minus_3_l511_511564


namespace parabola_problem_l511_511681

-- Let C be the parabola
def parabola_C (y x : ℝ) : Prop := y^2 = 4 * x

-- Define point F
def focus_F : (ℝ × ℝ) := (1, 0)

-- Define vector equality condition
def vector_condition (P Q F : ℝ × ℝ) : Prop :=
  let (px, py) := P in let (qx, qy) := Q in let (fx, fy) := F in
  (qx - px = 9 * (fx - qx)) ∧ (qy - py = 9 * (fy - qy))

-- Define line slope
def line_slope (P Q : ℝ × ℝ) : ℝ := 
  let (px, py) := P in let (qx, qy) := Q in
  if qx = 0 then 0 else qy / qx

-- Main theorem
theorem parabola_problem :
  (∀ y x : ℝ, parabola_C y x) ∧ 
  (∀ P Q : ℝ × ℝ, (parabola_C (P.2) (P.1)) ∧ (vector_condition P Q focus_F) → line_slope (0, 0) Q = 1 / 3) :=
by {
  sorry
}

end parabola_problem_l511_511681


namespace digits_in_X_n_occurrences_of_01_in_X_n_l511_511027

-- Definition of the Fibonacci sequence
def fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fib n + fib (n+1)

-- Sequences X_n defined according to the conditions
def X_n : ℕ → list ℕ
| 1     := [1]
| (n+1) := (X_n n).bind (λ d, if d = 0 then [1] else [1, 0])

-- Number of digits in X_n
def a_n (n : ℕ) : ℕ := (X_n n).length

-- Number of occurrences of '01' in X_n
def count_01 (l : list ℕ) : ℕ :=
list.sum (l.zip (list.tail l)).map (λ p, if p = (0, 1) then 1 else 0)

def b_n (n : ℕ) : ℕ := count_01 (X_n n)

-- Proof Problems
theorem digits_in_X_n (n : ℕ) : a_n n = fib (n+1) := sorry

theorem occurrences_of_01_in_X_n (n : ℕ) : 
  b_n n = fib (n-1) + (if n % 2 = 0 then -1 else 0) / 2 := sorry

end digits_in_X_n_occurrences_of_01_in_X_n_l511_511027


namespace admission_given_written_test_passed_l511_511821

variable {Ω : Type*} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}
variables (A B : Set Ω) (hA : P A = 0.2) (hC : P (A ∩ B) = 0.04)

theorem admission_given_written_test_passed :
  P[B | A] = 0.2 :=
by
  have h_cond : P[A ∩ B] = P[A] * P[B | A], from MeasureTheory.probability_def.cond_prob,
  have h_mul : 0.04 = 0.2 * P[B | A], from by rw [hC, hA, h_cond],
  linarith

end admission_given_written_test_passed_l511_511821


namespace intersecting_lines_property_l511_511125

theorem intersecting_lines_property
  {O A1 A2 A3 A4 B : Type}
  (m1 m2 m3 m4 : O → Prop)
  (H1 : ∀ x y, m1 x → m1 y → x = y)
  (H2 : ∀ x y, m2 x → m2 y → x = y)
  (H3 : ∀ x y, m3 x → m3 y → x = y)
  (H4 : ∀ x y, m4 x → m4 y → x = y)
  (A1_on_m1 : m1 A1)
  (A2_on_m2 : ∀ (x : O), m4 x → m2 (A1))
  (A3_on_m3 : ∀ (x : O), m1 x → m3 (A2))
  (A4_on_m4 : ∀ (x : O), m2 x → m4 (A3))
  (B_on_m1 : ∀ (x : O), m3 x → m1 (A4)) :
  ∃ (O B A1 A2 A3 A4 : Type), dist O B < dist O A1 / 2 :=
begin
  sorry
end

end intersecting_lines_property_l511_511125


namespace largest_positive_integer_n_exists_l511_511533

theorem largest_positive_integer_n_exists (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ∃ n, 
    0 < n ∧ 
    (n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 9) ∧ 
    ∀ m, 0 < m → 
      (m^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 9) → 
      m ≤ n :=
  sorry

end largest_positive_integer_n_exists_l511_511533


namespace cos_2x_quadratic_l511_511983

theorem cos_2x_quadratic (x : ℝ) (a b c : ℝ)
  (h : a * (Real.cos x) ^ 2 + b * Real.cos x + c = 0)
  (h_a : a = 4) (h_b : b = 2) (h_c : c = -1) :
  4 * (Real.cos (2 * x)) ^ 2 + 2 * Real.cos (2 * x) - 1 = 0 := sorry

end cos_2x_quadratic_l511_511983


namespace length_CF_area_triangle_ACF_l511_511868

noncomputable def conditions (radius : ℝ) (A B C D F : Point) : Prop :=
  (circle_intersect_at_two_points A B radius) ∧
  (on_circle C radius) ∧
  (on_circle D radius) ∧
  (between B C D) ∧
  (right_angle C A D) ∧
  ((distance B F) = (distance B D))

theorem length_CF (A B C D F : Point) : conditions 5 A B C D F → distance C F = 10 :=
by sorry

noncomputable def area_conditions (radius : ℝ) (A B C D F : Point) : Prop :=
  conditions radius A B C D F ∧
  (distance B C = 6)

theorem area_triangle_ACF (A B C D F : Point) : area_conditions 5 A B C D F → area_triangle A C F = 49 :=
by sorry

end length_CF_area_triangle_ACF_l511_511868


namespace solution_set_of_inequality_l511_511397

theorem solution_set_of_inequality :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} := 
sorry

end solution_set_of_inequality_l511_511397


namespace five_letter_words_with_at_least_one_vowel_l511_511211

open Finset

theorem five_letter_words_with_at_least_one_vowel :
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := ((letters \ vowels).card : ℕ) ^ 5
  total_words - no_vowel_words = 6752 :=
by
  let letters := insert 'A' (insert 'B' (insert 'C' (insert 'D' (insert 'E' (singleton 'F')))))
  let vowels := insert 'A' (singleton 'E')
  let consonants := letters \ vowels
  have : letters.card = 6 := by simp
  have : vowels.card = 2 := by simp
  have : consonants.card = 4 := by simp
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := (consonants.card : ℕ) ^ 5
  calc
    total_words - no_vowel_words
        = 6^5 - 4^5         : by simp
    ... = 7776 - 1024       : by norm_num
    ... = 6752             : by norm_num

end five_letter_words_with_at_least_one_vowel_l511_511211


namespace fishbowl_water_level_rise_l511_511911

noncomputable def volume_cube (side : ℝ) : ℝ :=
  side ^ 3

noncomputable def base_area (side : ℝ) : ℝ :=
  side ^ 2

noncomputable def water_level_rise (V_displaced A_base : ℝ) : ℝ :=
  V_displaced / A_base

theorem fishbowl_water_level_rise :
  let fishbowl_side := 20
  let iron_cube_side := 10
  let V_iron := volume_cube iron_cube_side
  let A_base := base_area fishbowl_side
  V_iron == 1000 →
  A_base == 400 →
  water_level_rise V_iron A_base = 2.5 :=
by
  intros
  rw [volume_cube, base_area]
  sorry

end fishbowl_water_level_rise_l511_511911


namespace rationalize_denominator_l511_511805

-- Lean 4 statement
theorem rationalize_denominator : sqrt (5 / 18) = sqrt 10 / 6 := by
  sorry

end rationalize_denominator_l511_511805


namespace calculate_f_at_2x_l511_511108

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem using the given condition and the desired result
theorem calculate_f_at_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 :=
by
  sorry

end calculate_f_at_2x_l511_511108


namespace five_letter_words_with_vowel_l511_511246

theorem five_letter_words_with_vowel :
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  total_words - words_no_vowel = 6752 :=
by
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  have h1 : total_words = 7776 := by norm_num
  have h2 : words_no_vowel = 1024 := by norm_num
  show total_words - words_no_vowel = 6752
  calc
    total_words - words_no_vowel
    = 7776 - 1024 : by rw [h1, h2]
    ... = 6752 : by norm_num

end five_letter_words_with_vowel_l511_511246


namespace monotonic_intervals_of_f_g_minus_f_lt_3_l511_511129

noncomputable def f (x : ℝ) : ℝ := -x * Real.log (-x)

noncomputable def g (x : ℝ) : ℝ := Real.exp x - x

-- Theorem 1: Monotonic intervals of f(x)
theorem monotonic_intervals_of_f : 
  (∀ x : ℝ, x ∈ set.Ioo (-∞) (-1 / Real.exp 1) → f' x < 0) ∧ 
  (∀ x : ℝ, x ∈ set.Ioo (-1 / Real.exp 1) 0 → f' x > 0) :=
sorry

-- Theorem 2: Proving g(x) - f(x) < 3
theorem g_minus_f_lt_3 (x : ℝ) : g(x) - f(x) < 3 :=
sorry

end monotonic_intervals_of_f_g_minus_f_lt_3_l511_511129


namespace cos_pi_over_4_minus_alpha_l511_511643

theorem cos_pi_over_4_minus_alpha (α : ℝ) (h : Real.sin (Real.pi / 4 + α) = 2 / 3) :
  Real.cos (Real.pi / 4 - α) = 2 / 3 := 
by
  sorry

end cos_pi_over_4_minus_alpha_l511_511643


namespace find_a_minus_b_l511_511997

theorem find_a_minus_b (a b : ℝ) (h1 : a + b = 12) (h2 : a^2 - b^2 = 48) : a - b = 4 :=
by
  sorry

end find_a_minus_b_l511_511997


namespace E5_correct_Sn_difference_lt_1_min_En_correct_l511_511163

-- Define the sequence and the conditions
def seq_A := λ (n : ℕ), list ℚ
def condition (a : ℚ) := a < 1
def sum_S (A : seq_A) (k : ℕ) : ℚ :=
  (list.take k A).sum

-- (I) State specific sequence and define the set E_5
def A5 := [-0.3, 0.7, -0.1, 0.9, 0.1]

def E5 : set ℕ := {k | k ∈ {1, 2, 3, 4, 5} ∧ sum_S A5 k > (list.nth_le ((list.take (k - 1) A5).++[0]) ((k - 1) % (k : ℕ)) sorry : ℚ)}

-- Prove E5 = {2, 4, 5}
theorem E5_correct : E5 = {2, 4, 5} :=
sorry

-- (II) Prove S_{k_{i+1}} - S_{k_i} < 1
theorem Sn_difference_lt_1 (A : seq_A) (k_i k_i1 : ℕ) (h_i1 : k_i < k_i1)
  (h_E : k_i ∈ E5) (h_E1 : k_i1 ∈ E5) :
  sum_S A k_i1 - sum_S A k_i < 1 :=
sorry

-- (III) Minimum number of elements in E_n given S_n > C
noncomputable def min_elements_En (A : seq_A) (n : ℕ) (C : ℚ) (h_Sn : sum_S A n > C) :=
  finset.card {k ∈ finset.range n | sum_S A k > (list.nth_le ((list.take (k - 1) A).++[0]) ((k - 1) % (k : ℕ)) sorry : ℚ)}

theorem min_En_correct (A : seq_A) (n : ℕ) (C : ℚ) (h_Sn : sum_S A n > C) :
  min_elements_En A n C h_Sn ≥ C + 1 :=
sorry

end E5_correct_Sn_difference_lt_1_min_En_correct_l511_511163


namespace sum_of_differences_l511_511619

/-- Proving that the sum of all differences (first digit - last digit) for natural numbers from 
    1 to 999 is 495. -/ 
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, 
    let str := n.toString
    let first := if str.length > 1 then str.head!.toNat - '0'.toNat else 0
    let last := if str.length > 1 then str.getLast.toNat - '0'.toNat else 0
    first - last
  ) = 495 := 
by
  sorry

end sum_of_differences_l511_511619


namespace parabola_line_intersection_l511_511755

noncomputable def exists_parabola_intersection (F : Point) (D L : Line) : Prop :=
∃ (G : Point), G ∈ L ∧ is_parabola_point F D G

axiom is_parabola_point (F : Point) (D : Line) (G : Point) : Prop 

theorem parabola_line_intersection (F : Point) (D L : Line) :
exists_parabola_intersection F D L :=
sorry

end parabola_line_intersection_l511_511755


namespace sum_of_differences_l511_511583

open Nat
open BigOperators

theorem sum_of_differences (n : ℕ) (h : n ≥ 1 ∧ n ≤ 999) : 
  let differences := (fun x => 
                        let first_digit := x / 10;
                        let last_digit := x % 10;
                        first_digit - last_digit) in
  ∑ i in (finset.range (n + 1)).filter (λ x, 1 ≤ x ∧ x ≤ 999), differences i = 495 :=
by
  -- Acknowledge the need for a more refined filtering criteria for numbers between 1 and 999
  sorry

end sum_of_differences_l511_511583


namespace part1_a_b_part2_estimation_part3_factory_choice_l511_511003

-- Definitions based on the given conditions
def factoryA_weights : List ℝ := [74, 74, 74, 75, 73, 77, 78, 72, 76, 77]
def factoryB_weights : List ℝ := [78, 74, 77, 73, 75, 75, 74, 74, 75, 75]

def mean (l: List ℝ) : ℝ := (l.sum) / (l.length.toFloat)
def median (l: List ℝ) : ℝ := 
  let sorted := l.sort
  if (sorted.length % 2 = 0) then
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else
    sorted.get! (sorted.length / 2)

def mode (l: List ℝ) : ℝ := 
  l.groupBy id
  |> List.sortBy (λ g => (-(g.length), g.head!.toFloat))
  |> List.head!.head!

def variance (l: List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / (l.length.toFloat)

-- Formulating the proof problems
theorem part1_a_b : median factoryB_weights = 75 ∧ mode factoryA_weights = 74 := by
  sorry

theorem part2_estimation : 
  let count := (factoryB_weights.filter (λ x => x = 75)).length
  (count / 10 * 100).toInt = 40 := by
  sorry

theorem part3_factory_choice : 
  variance factoryB_weights < variance factoryA_weights := by
  sorry

end part1_a_b_part2_estimation_part3_factory_choice_l511_511003


namespace max_smallest_sum_l511_511392

theorem max_smallest_sum {s : set ℕ} (h : s = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :
  ∃ L : list ℕ, L.perm s.to_list ∧
  ∀ l ∈ L.cyclic_triples, l.sum ≤ 15 ∧
  ∃ l' ∈ L.cyclic_triples, l'.sum = 15 :=
begin
  sorry
end

end max_smallest_sum_l511_511392


namespace exists_sum_of_fractions_for_reciprocal_l511_511299

theorem exists_sum_of_fractions_for_reciprocal (n : ℕ) (h : n > 1) :
  ∃ i j : ℕ, (∑ k in Finset.range (j - i + 1), (1 : ℝ) / ((i + k) * (i + k + 1))) = (1 : ℝ) / n :=
sorry

end exists_sum_of_fractions_for_reciprocal_l511_511299


namespace factory_days_worked_l511_511474

-- Define the number of refrigerators produced per hour
def refrigerators_per_hour : ℕ := 90

-- Define the number of coolers produced per hour
def coolers_per_hour : ℕ := refrigerators_per_hour + 70

-- Define the number of working hours per day
def working_hours_per_day : ℕ := 9

-- Define the total products produced per hour
def products_per_hour : ℕ := refrigerators_per_hour + coolers_per_hour

-- Define the total products produced in a day
def products_per_day : ℕ := products_per_hour * working_hours_per_day

-- Define the total number of products produced in given days
def total_products : ℕ := 11250

-- Define the number of days worked
def days_worked : ℕ := total_products / products_per_day

-- Prove that the number of days worked equals 5
theorem factory_days_worked : days_worked = 5 :=
by
  sorry

end factory_days_worked_l511_511474


namespace proof_l511_511537

noncomputable def total_ways : ℕ := (Finset.range 21).choose 20

noncomputable def A (n : ℕ) : ℕ :=
  5 * 4 * (Finset.range 12).choose 4 * (Finset.range 8).choose 4 * (Finset.range 4).choose 3

noncomputable def B (n : ℕ) : ℕ :=
  (Finset.range 21).choose 20 * (Finset.range 17).choose 20 * (Finset.range 13).choose 4 * (Finset.range 9).choose 4 * (Finset.range 5).choose 4

noncomputable def p (n : ℕ) : ℝ :=
  A n / total_ways

noncomputable def q (n : ℕ) : ℝ :=
  B n / total_ways

theorem proof : (p n) / (q n) = (A n) / (B n) := by
  sorry

end proof_l511_511537


namespace FG_tangent_to_circumcircle_ADG_l511_511347

-- Definitions and conditions
variables (ω : Circle) (A B C D E F G : Point)
variables (h1 : InscribedPentagon ω A B C D E)
variables (h2 : Parallel CD BE)
variables (h3 : TangentThrough ω B F)
variables (h4 : OnLine F AC)
variables (h5 : Between A C F)
variables (h6 : IntersectAt BD AE G)

-- Theorem statement
theorem FG_tangent_to_circumcircle_ADG : Tangent (Line FG) (Circumcircle (Triangle A D G)) :=
by
  sorry

end FG_tangent_to_circumcircle_ADG_l511_511347


namespace count_5_letter_words_with_at_least_one_vowel_l511_511205

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := (letters.card) ^ 5

def non_vowel_letters : Finset Char := letters \ vowels

def total_non_vowel_5_letter_words : ℕ := (non_vowel_letters.card) ^ 5

theorem count_5_letter_words_with_at_least_one_vowel :
  total_5_letter_words - total_non_vowel_5_letter_words = 6752 :=
by
  sorry

end count_5_letter_words_with_at_least_one_vowel_l511_511205


namespace highest_income_highest_expenditure_total_amount_money_after_transactions_l511_511409

section XiaoYingTransactions

def transactions : List ℝ := [+25, -6, +18, +12, -24, -15]

theorem highest_income :
  ∃ (idx : Fin 6), transactions[idx] = 25 ∧ 
                      ∀ (j : Fin 6), transactions[j] ≤ transactions[idx] := by
  sorry

theorem highest_expenditure :
  ∃ (idx : Fin 6), transactions[idx] = -24 ∧
                      ∀ (j : Fin 6), transactions[j] ≥ transactions[idx] := by
  sorry

theorem total_amount :
  List.sum (transactions.map abs) = 100 := by
  sorry

theorem money_after_transactions (initial_money : ℝ) :
  initial_money = 40 → 
  initial_money + List.sum transactions = 50 := by
  sorry

end XiaoYingTransactions

end highest_income_highest_expenditure_total_amount_money_after_transactions_l511_511409


namespace hyperbola_eccentricity_l511_511314

theorem hyperbola_eccentricity 
  (a b e : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : e = Real.sqrt (1 + (b^2 / a^2))) 
  (h4 : e ≤ Real.sqrt 5) : 
  e = 2 := 
sorry

end hyperbola_eccentricity_l511_511314


namespace volume_multiplier_factor_l511_511449

variable (π r h : ℝ)

-- Original volume
def V_original : ℝ := π * r^2 * h

-- New radius and height
def r_new : ℝ := 2.5 * r
def h_new : ℝ := 3 * h

-- New volume
def V_new : ℝ := π * (r_new)^2 * h_new

-- Proof that the volume is multiplied by 18.75
theorem volume_multiplier_factor
  (h_is_positive : 0 < h)
  (r_is_positive : 0 < r) :
  V_new π r h = 18.75 * V_original π r h :=
by
  -- Volume calculation
  have V_original_def : V_original π r h = π * r^2 * h := rfl
  have V_new_def : V_new π r h = π * (2.5 * r)^2 * (3 * h) := rfl
  rw [V_original_def, V_new_def]
  -- Simplify
  calc
    π * (2.5 * r)^2 * (3 * h)
    = π * (2.5^2 * r^2) * (3 * h) : by ring
    ... = π * 6.25 * r^2 * 3 * h : by ring
    ... = 18.75 * π * r^2 * h : by ring
  sorry

end volume_multiplier_factor_l511_511449


namespace pentagon_area_greater_than_square_third_l511_511536

theorem pentagon_area_greater_than_square_third (a b : ℝ) :
  a^2 + (a * b) / 4 + (Real.sqrt 3 / 4) * b^2 > ((a + b)^2) / 3 :=
by
  sorry

end pentagon_area_greater_than_square_third_l511_511536


namespace pet_store_cages_required_l511_511022

theorem pet_store_cages_required {initial_puppies : ℝ} {additional_puppies : ℝ} {puppies_per_cage : ℝ} (h1 : initial_puppies = 18.0) (h2 : additional_puppies = 3.0) (h3 : puppies_per_cage = 5.0) : 
  let total_puppies := initial_puppies + additional_puppies in 
  ⌈total_puppies / puppies_per_cage⌉ = 5 :=
by
  let total_puppies := initial_puppies + additional_puppies
  sorry

end pet_store_cages_required_l511_511022


namespace percentage_increase_l511_511481

-- Definitions and conditions
def final_value : ℕ := 480
def initial_value : ℕ := 400

-- Theorem statement
theorem percentage_increase : ((final_value - initial_value) / initial_value) * 100 = 20 := 
by 
sorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrry

end percentage_increase_l511_511481


namespace sum_of_altitudes_of_triangle_l511_511063

open Real

noncomputable def sum_of_altitudes (a b c : ℝ) : ℝ :=
  let inter_x := -c / a
  let inter_y := -c / b
  let vertex1 := (inter_x, 0)
  let vertex2 := (0, inter_y)
  let vertex3 := (0, 0)
  let area_triangle := (1 / 2) * abs (inter_x * inter_y)
  let altitude_x := abs inter_x
  let altitude_y := abs inter_y
  let altitude_line := abs c / sqrt (a ^ 2 + b ^ 2)
  altitude_x + altitude_y + altitude_line

theorem sum_of_altitudes_of_triangle :
  sum_of_altitudes 15 6 90 = 21 + 10 * sqrt (1 / 29) :=
by
  sorry

end sum_of_altitudes_of_triangle_l511_511063


namespace incorrect_geometric_statements_l511_511660

variables (m n : Type) [line m] [line n]
variables (α β : Type) [plane α] [plane β]

theorem incorrect_geometric_statements 
  (h1 : m ≠ n) 
  (h2 : α ≠ β) :
  (¬ (m ⊥ α ∧ α ⊥ β → m ∥ β)) ∧
  (¬ (n ∥ α ∧ α ∥ β → n ∥ β)) ∧
  (¬ (n ∥ α ∧ α ⊥ β → n ⊥ β)) :=
sorry

end incorrect_geometric_statements_l511_511660


namespace richard_older_than_david_l511_511000

theorem richard_older_than_david
  (R D S : ℕ)   -- ages of Richard, David, Scott
  (x : ℕ)       -- the number of years Richard is older than David
  (h1 : R = D + x)
  (h2 : D = S + 8)
  (h3 : R + 8 = 2 * (S + 8))
  (h4 : D = 14) : 
  x = 6 := sorry

end richard_older_than_david_l511_511000


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511234

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let total_words := 6^5
  let words_without_vowels := 4^5
  total_words - words_without_vowels = 6752 :=
by
  let total_words := 6^5
  let words_without_vowels := 4^5
  have h_total_words : total_words = 7776 := by norm_num
  have h_words_without_vowels : words_without_vowels = 1024 := by norm_num
  calc
    7776 - 1024 = 6752 : by norm_num

end number_of_5_letter_words_with_at_least_one_vowel_l511_511234


namespace find_t_l511_511344

theorem find_t (t : ℝ) :
  let my_hours := t + 2
  let my_rate := 4t - 4
  let bob_hours := 2t - 3
  let bob_rate := t + 3
  let my_earnings := my_hours * my_rate
  let bob_earnings := bob_hours * bob_rate
  (my_earnings = bob_earnings + 3) →
  t = (1 - Real.sqrt 5) / 2 ∨ t = (Real.sqrt 5 - 1) / 2 :=
by
  sorry

end find_t_l511_511344


namespace triangle_and_square_area_proof_l511_511497

noncomputable def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def inscribed_square_area (d : ℕ) : ℚ :=
  (d / Real.sqrt 2)^2

theorem triangle_and_square_area_proof :
  ∃ (a b c : ℕ), a = 15 ∧ b = 36 ∧ c = 39 ∧
  is_right_triangle a b c ∧
  inscribed_square_area c = 760.5 :=
by
  use 15, 36, 39
  split
  { refl }
  split
  { refl }
  split
  { refl }
  split
  { exact sorry }
  { exact sorry }

end triangle_and_square_area_proof_l511_511497


namespace scientific_notation_correct_l511_511969

noncomputable def notation_of_three_million_one_thousand : Prop :=
  scientific_notation 3001000 = 3.001 * 10^6

theorem scientific_notation_correct :
  notation_of_three_million_one_thousand := 
by
  sorry

end scientific_notation_correct_l511_511969


namespace outlet_cost_correct_l511_511910

theorem outlet_cost_correct : 
  let standard_rooms := 45
  let suite_rooms := 15
  let outlets_per_standard := 10
  let outlets_per_suite := 15
  let type_c_cost := 6.0
  let standard_total_outlets := standard_rooms * outlets_per_standard
  let suite_total_outlets := suite_rooms * outlets_per_suite
  let type_c_percentage_standard := 0.30
  let type_c_percentage_suite := 0.40
  let total_type_c_outlets := (type_c_percentage_standard * standard_total_outlets) + 
                             (type_c_percentage_suite * suite_total_outlets)
  let total_cost_type_c := total_type_c_outlets * type_c_cost
  in total_cost_type_c = 1350.0 := 
by
  sorry

end outlet_cost_correct_l511_511910


namespace total_contribution_l511_511936

theorem total_contribution : 
  ∀ (Niraj_contribution : ℕ) (Brittany_contribution Angela_contribution : ℕ),
    (Brittany_contribution = 3 * Niraj_contribution) →
    (Angela_contribution = 3 * Brittany_contribution) →
    (Niraj_contribution = 80) →
    (Niraj_contribution + Brittany_contribution + Angela_contribution = 1040) :=
  by assumption sorry

end total_contribution_l511_511936


namespace y_intercept_of_line_l511_511080

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 6 * y = 24) : y = 4 := by
  sorry

end y_intercept_of_line_l511_511080


namespace AM_dot_BC_eq_one_l511_511731

-- Define the cube and points
structure Cube :=
  (A B C D A₁ B₁ C₁ D₁ : ℝ × ℝ × ℝ)
  (edge_length : ℝ)
  (valid_cube : A.1 = B.1 ∧ A.2 = B.2 ∧ A.3 = B.3 ∧ 
                B.1 ≠ C.1 ∧ B.2 ≠ C.2 ∧ B.3 ≠ C.3 ∧
                A₁.1 = B₁.1 ∧ A₁.2 = B₁.2 ∧ A₁.3 ≠ B₁.3 ∧
                -- Further conditions to ensure it's a valid cube can be added here
                edge_length = 1)

noncomputable def AM_dot_BC (c : Cube) (M : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem AM_dot_BC_eq_one (c : Cube) (M : ℝ × ℝ × ℝ) 
  (hM : M = (c.C₁.1, c.C₁.2, M.3)) : AM_dot_BC c M = 1 :=
by
  sorry

end AM_dot_BC_eq_one_l511_511731


namespace largest_divisor_of_five_consecutive_integers_product_correct_l511_511440

noncomputable def largest_divisor_of_five_consecutive_integers_product : ℕ :=
  120

theorem largest_divisor_of_five_consecutive_integers_product_correct :
  ∀ (n : ℕ), (∃ k : ℕ, k = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ∧ 120 ∣ k) :=
sorry

end largest_divisor_of_five_consecutive_integers_product_correct_l511_511440


namespace quadratic_has_distinct_real_roots_l511_511844

-- Definitions for the quadratic equation coefficients
def a : ℝ := 3
def b : ℝ := -4
def c : ℝ := 1

-- Definition of the discriminant
def Δ : ℝ := b^2 - 4 * a * c

-- Statement of the problem: Prove that the quadratic equation has two distinct real roots
theorem quadratic_has_distinct_real_roots (hΔ : Δ = 4) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) :=
by
  sorry

end quadratic_has_distinct_real_roots_l511_511844


namespace part1_l511_511995

variable (α : ℝ)

theorem part1 (h : Real.tan α = 2) : (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 8 :=
by
  sorry

end part1_l511_511995


namespace initial_salty_cookies_l511_511793

theorem initial_salty_cookies (sweet_init sweet_eaten sweet_left salty_eaten : ℕ) 
  (h1 : sweet_init = 34)
  (h2 : sweet_eaten = 15)
  (h3 : sweet_left = 19)
  (h4 : salty_eaten = 56) :
  sweet_left + sweet_eaten = sweet_init → 
  sweet_init - sweet_eaten = sweet_left →
  ∃ salty_init, salty_init = salty_eaten :=
by
  sorry

end initial_salty_cookies_l511_511793


namespace rabbit_catch_up_time_l511_511860

-- Define the conditions
def rabbit_speed : ℝ := 25 -- Rabbit's speed in miles per hour
def cat_speed : ℝ := 20 -- Cat's speed in miles per hour
def head_start_time : ℝ := 0.25 -- Head start time in hours

-- Define the distance the cat covers in the head start
def distance_cat : ℝ := cat_speed * head_start_time

-- Define the relative speed between the rabbit and the cat
def relative_speed : ℝ := rabbit_speed - cat_speed

-- Define the time for the rabbit to catch up
def time_to_catch_up : ℝ := distance_cat / relative_speed

-- Prove that the time to catch up is 1 hour
theorem rabbit_catch_up_time : time_to_catch_up = 1 :=
by
  -- proof will go here
  -- but this is a properly structured Lean statement based on given conditions 
  sorry

end rabbit_catch_up_time_l511_511860


namespace integer_square_root_35_consecutive_l511_511294

theorem integer_square_root_35_consecutive : 
  ∃ n : ℕ, ∀ k : ℕ, n^2 ≤ k ∧ k < (n+1)^2 ∧ ((n + 1)^2 - n^2 = 35) ∧ (n = 17) := by 
  sorry

end integer_square_root_35_consecutive_l511_511294


namespace incorrect_prob_l511_511719

-- Define the number of cards we are working with
def num_total_cards : ℕ := 52 -- not part of the given, implicitly understood

-- Conditions
def num_jokers : ℕ := 2
def num_3_hearts : ℕ := 1
def num_ace_spades : ℕ := 1
def num_aces : ℕ := 4
def num_kings : ℕ := 4
def num_little_jokers : ℕ := 1

-- Probabilities
def prob_joker : ℝ := num_jokers / num_total_cards
def prob_3_hearts : ℝ := num_3_hearts / num_total_cards
def prob_ace_spades : ℝ := num_ace_spades / num_total_cards
def prob_ace : ℝ := num_aces / num_total_cards
def prob_king : ℝ := num_kings / num_total_cards
def prob_little_joker : ℝ := num_little_jokers / num_total_cards

-- Theorem statement that needs to be proven incorrect:
theorem incorrect_prob :
  prob_ace_spades = prob_joker → False :=
by
-- This is where the proof will go.
sorry

end incorrect_prob_l511_511719


namespace non_trivial_solution_exists_l511_511763

theorem non_trivial_solution_exists (a b c : ℤ) (p : ℕ) [Fact (Nat.Prime p)] :
  ∃ x y z : ℤ, (a * x^2 + b * y^2 + c * z^2) % p = 0 ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :=
sorry

end non_trivial_solution_exists_l511_511763


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511182

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the vowels
def vowels := {'A', 'E'}

-- Define the number of n-length words constructible from a set of letters
def num_words (n : ℕ) (alphabet : Set Char) : ℕ :=
  (alphabet.size ^ n)

-- The total number of 5-letter words (unrestricted)
def total_words := num_words 5 letters

-- The number of 5-letter words with no vowels
def no_vowel_words := num_words 5 (letters \ vowels)

-- The number of 5-letter words with at least one vowel
def at_least_one_vowel_words := total_words - no_vowel_words

-- The statement to prove that the number of 5-letter words with at least one vowel is 6752
theorem number_of_5_letter_words_with_at_least_one_vowel : 
  at_least_one_vowel_words = 6752 :=
by 
  -- Proof will be provided here
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l511_511182


namespace probability_of_diff_grade_equals_two_thirds_l511_511465

-- Define the necessary conditions
variables {students : Finset (Fin 4)} (g1 g2 : Finset (Fin 4))
  (g1_size : g1.card = 2) (g2_size : g2.card = 2)
  (h : g1 ∪ g2 = students) (student_count : students.card = 4)

-- Define the event of selecting 2 students
def select_two_students : Finset (Finset (Fin 4)) := students.powerset.filter (λ s, s.card = 2)

-- Define the event of selecting 2 students from different grades
def diff_grade (s : Finset (Fin 4)) : Prop := ∃ (a ∈ g1) (b ∈ g2), s = {a, b}

-- Calculate the probability
def probability_diff_grade (students : Finset (Fin 4)) (g1 g2 : Finset (Fin 4))
  (g1_size : g1.card = 2) (g2_size : g2.card = 2) (h : g1 ∪ g2 = students) 
  (student_count : students.card = 4) : ℚ :=
let m := (select_two_students.filter diff_grade).card
in m / select_two_students.card

theorem probability_of_diff_grade_equals_two_thirds :
  probability_diff_grade students g1 g2 g1_size g2_size h student_count = 2 / 3 := by
  sorry

end probability_of_diff_grade_equals_two_thirds_l511_511465


namespace sum_diff_1_to_999_l511_511634

def subtract_last_from_first (n : ℕ) : ℤ :=
  let str_n := n.toString
  if str_n.length = 1 then 0
  else
    let first_digit := str_n.toList.head!.digitToInt!
    let last_digit := str_n.toList.reverse.head!.digitToInt!
    first_digit - last_digit

def numbers : List ℕ := List.range 1000.tail

def sum_of_differences : ℤ := (numbers.map subtract_last_from_first).sum

theorem sum_diff_1_to_999 :
  sum_of_differences = 495 := 
sorry

end sum_diff_1_to_999_l511_511634


namespace least_distinct_b_numbers_l511_511458

theorem least_distinct_b_numbers {a : Fin 100 → ℕ} 
  (h_distinct : Function.Injective a) :
  let b (i : Fin 100) := a i + Nat.gcd (Finset.sum (Finset.univ.erase i) a) 1 in
  ∃ (s : Finset ℕ), (∀ i, b i ∈ s) ∧ s.card = 100 :=
by sorry

end least_distinct_b_numbers_l511_511458


namespace sum_of_differences_l511_511586

open Nat
open BigOperators

theorem sum_of_differences (n : ℕ) (h : n ≥ 1 ∧ n ≤ 999) : 
  let differences := (fun x => 
                        let first_digit := x / 10;
                        let last_digit := x % 10;
                        first_digit - last_digit) in
  ∑ i in (finset.range (n + 1)).filter (λ x, 1 ≤ x ∧ x ≤ 999), differences i = 495 :=
by
  -- Acknowledge the need for a more refined filtering criteria for numbers between 1 and 999
  sorry

end sum_of_differences_l511_511586


namespace sum_of_differences_l511_511598

theorem sum_of_differences : 
  let first_digit (n : ℕ) : ℕ := n / 10^(nat.log10 n)
  let last_digit (n : ℕ) : ℕ := n % 10
  (finset.range 1000).sum (λ n, first_digit n - last_digit n) = 495 :=
by
  sorry

end sum_of_differences_l511_511598


namespace find_S40_l511_511331

variables {a : ℕ → ℝ} {r : ℝ}

-- Definition of being a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n * r

-- Sum of first n terms of a sequence
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, a i

-- Given conditions as hypotheses
variables (h_geometric : is_geometric_sequence a r)
          (h_S10 : S a 10 = 10)
          (h_S30 : S a 30 = 70)

-- Goal statement
theorem find_S40 : S a 40 = 150 :=
sorry

end find_S40_l511_511331


namespace trig_identity_example_l511_511071

theorem trig_identity_example :
  (Real.cos (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
   Real.sin (160 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) = Real.sqrt 3 / 2) :=
by {
  -- Placeholder for identifying \(\sin 160^{\circ} = \sin 20^{\circ}\)
  have h1 : Real.sin (160 * Real.pi / 180) = Real.sin (20 * Real.pi / 180),
  { sorry },

  -- Using the given trigonometric identity
  calc
    Real.cos (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180)
      - Real.sin (160 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)
    = Real.cos (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180)
      - Real.sin (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180) : by rw h1
    ... = Real.cos ((20 + 10) * Real.pi / 180) : by rw Real.cos_sub
    ... = Real.cos (30 * Real.pi / 180) : by norm_num
    ... = Real.sqrt 3 / 2 : by norm_num
}

end trig_identity_example_l511_511071


namespace scientific_notation_2150000_l511_511538

theorem scientific_notation_2150000 : 2150000 = 2.15 * 10^6 :=
  by
  sorry

end scientific_notation_2150000_l511_511538


namespace inequality_holds_for_all_x_l511_511277

theorem inequality_holds_for_all_x (m : ℝ) : (∀ x : ℝ, (m^2 + 4*m - 5)*x^2 - 4*(m - 1)*x + 3 > 0) ↔ (1 ≤ m ∧ m < 19) :=
by {
  sorry
}

end inequality_holds_for_all_x_l511_511277


namespace sum_digit_differences_l511_511604

def first_digit (n : ℕ) : ℕ := 
  (n / 10 ^ ((Nat.log10 n) : ℕ))

def last_digit (n : ℕ) : ℕ := n % 10

def digit_difference (n : ℕ) : ℤ :=
  (first_digit n : ℤ) - (last_digit n : ℤ)

theorem sum_digit_differences :
  (∑ n in Finset.range 1000, digit_difference n) = 495 := 
sorry

end sum_digit_differences_l511_511604


namespace hyperbola_eccentricity_l511_511315

theorem hyperbola_eccentricity 
  (a b e : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : e = Real.sqrt (1 + (b^2 / a^2))) 
  (h4 : e ≤ Real.sqrt 5) : 
  e = 2 := 
sorry

end hyperbola_eccentricity_l511_511315


namespace maximize_distance_l511_511105

theorem maximize_distance (front_tires_lifetime: ℕ) (rear_tires_lifetime: ℕ):
  front_tires_lifetime = 20000 → rear_tires_lifetime = 30000 → 
  ∃ D, D = 30000 :=
by
  sorry

end maximize_distance_l511_511105


namespace trigonometric_identity_l511_511994

theorem trigonometric_identity (α : ℝ) (h : Real.tan (π + α) = 2) :
  (Real.sin (α - π) + Real.cos (π - α)) / (Real.sin (π + α) - Real.cos (π + α)) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l511_511994


namespace TruckloadsOfSand_l511_511025

theorem TruckloadsOfSand (S : ℝ) (totalMat dirt cement : ℝ) 
  (h1 : totalMat = 0.67) 
  (h2 : dirt = 0.33) 
  (h3 : cement = 0.17) 
  (h4 : totalMat = S + dirt + cement) : 
  S = 0.17 := 
  by 
    sorry

end TruckloadsOfSand_l511_511025


namespace no_inverse_3_mod_33_l511_511540

theorem no_inverse_3_mod_33 : ¬ ∃ x : ℤ, 3 * x ≡ 1 [MOD 33] := 
by 
  intro h
  have g : Int.gcd 3 33 = 3 := by sorry
  have co_prime : (Int.gcd 3 33 = 1) → False := by {
    intro h,
    have gcd_fact := g,
    contradiction
  }
  have inverse_property : ∀ x : ℤ, 3 * x ≡ 1 [MOD 33] → (Int.gcd 3 33 = 1) := by
    intro x h,
    sorry
  exact co_prime (inverse_property x h)

end no_inverse_3_mod_33_l511_511540


namespace solve_eq1_solve_eq2_l511_511815

theorem solve_eq1 (x : ℤ) : x - 2 * (5 + x) = -4 → x = -6 := by
  sorry

theorem solve_eq2 (x : ℤ) : (2 * x - 1) / 2 = 1 - (3 - x) / 4 → x = 1 := by
  sorry

end solve_eq1_solve_eq2_l511_511815


namespace andrew_beth_heads_probability_l511_511935

theorem andrew_beth_heads_probability :
  let X := binomial 5 (1/2 : ℝ)
  let Y := binomial 6 (1/2 : ℝ)
  P(X >= Y) = 0.5 :=
sorry

end andrew_beth_heads_probability_l511_511935


namespace five_letter_words_with_vowel_l511_511247

theorem five_letter_words_with_vowel :
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  total_words - words_no_vowel = 6752 :=
by
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  have h1 : total_words = 7776 := by norm_num
  have h2 : words_no_vowel = 1024 := by norm_num
  show total_words - words_no_vowel = 6752
  calc
    total_words - words_no_vowel
    = 7776 - 1024 : by rw [h1, h2]
    ... = 6752 : by norm_num

end five_letter_words_with_vowel_l511_511247


namespace multiple_of_weight_lifted_l511_511539

-- Define the weights of Felix and his brother
variables (F B : ℝ)

-- Felix's lifting capacity condition
def felix_lift_condition := 1.5 * F = 150

-- Felix's brother weight condition
def brother_weight_condition := B = 2 * F

-- Brother's lifting capacity
def brother_lift_capacity := 600

-- Proof problem: Prove that Felix's brother can lift 3 times his weight
theorem multiple_of_weight_lifted (h1 : felix_lift_condition) (h2 : brother_weight_condition) :
  600 / B = 3 :=
by
sorry

end multiple_of_weight_lifted_l511_511539


namespace fractions_lcm_l511_511520

noncomputable def lcm_of_fractions_lcm (numerators : List ℕ) (denominators : List ℕ) : ℕ :=
  let lcm_nums := numerators.foldr Nat.lcm 1
  let gcd_denom := denominators.foldr Nat.gcd (denominators.headD 1)
  lcm_nums / gcd_denom

theorem fractions_lcm (hnum : List ℕ := [4, 5, 7, 9, 13, 16, 19])
                      (hdenom : List ℕ := [9, 7, 15, 13, 21, 35, 45]) :
  lcm_of_fractions_lcm hnum hdenom = 1244880 :=
by
  sorry

end fractions_lcm_l511_511520


namespace number_of_fish_caught_in_second_catch_l511_511282

-- Define the conditions as given in the problem
variables (N : ℕ) (x : ℕ)
variable (approx_N : N ≈ 1000)
variable (tagged_first : ℕ := 40)
variable (tagged_second : ℕ := 2)
variable (prop : (tagged_second : ℚ) / (x : ℚ) = (tagged_first : ℚ) / (N : ℚ))

-- The theorem that needs to be proven
theorem number_of_fish_caught_in_second_catch (hN : approx_N N) (hx : N = 1000) : x = 50 :=
by
  -- Translate the proportion to an equation and solve
  have h : (2 : ℚ) / (x : ℚ) = (40 : ℚ) / (1000 : ℚ) := by sorry
  have hx : (2 : ℚ) * 1000 = 40 * x := by sorry
  norm_num at hx
  exact hx

end number_of_fish_caught_in_second_catch_l511_511282


namespace roots_condition_l511_511275

noncomputable def equation_one_has_one_real_root (m : ℝ) : Prop :=
(m - 2) * x ^ 2 - 2 * (m - 1) * x + m = 0 ∧ (m - 2 = 0)

noncomputable def equation_two_has_two_identical_real_roots (m : ℝ) : Prop :=
m * x ^ 2 - (m + 2) * x + (4 - m) = 0 ∧ (m = 2)

theorem roots_condition (m : ℝ) :
  equation_one_has_one_real_root m → equation_two_has_two_identical_real_roots m :=
sorry

end roots_condition_l511_511275


namespace min_distance_sqrt2_l511_511077

noncomputable def C1_parametric (α : ℝ) : ℝ × ℝ :=
  (cos α, sqrt 3 * sin α)

def C1_cartesian (x y : ℝ) : Prop :=
  x^2 + (y^2 / 3) = 1

noncomputable def C2_polar (ρ θ : ℝ) : Prop :=
  ρ * sin (θ + π / 4) = 2 * sqrt 2

def C2_cartesian (x y : ℝ) : Prop :=
  x + y = 4

def distance (P Q : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

noncomputable def min_distance (α : ℝ) : ℝ :=
  abs ((cos α + sqrt 3 * sin α - 4) / sqrt 2)

theorem min_distance_sqrt2 (α : ℝ) : 
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), 
  (P = C1_parametric α) ∧
  (Q.1 + Q.2 = 4) ∧
  (Q.1 = cos α) ∧ 
  (Q.2 = sqrt 3 * sin α) ∧ 
  distance P Q = sqrt 2 :=
sorry

end min_distance_sqrt2_l511_511077


namespace matchsticks_20th_stage_l511_511060

theorem matchsticks_20th_stage :
  let a : ℕ → ℕ := λ n, 5 + (n - 1) * 3 in
  a 20 = 62 :=
by
  sorry

end matchsticks_20th_stage_l511_511060


namespace five_letter_words_with_vowel_l511_511196

theorem five_letter_words_with_vowel : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let n := 5 
  (∃ (w : list(char)), w.length = n ∧ ∀ (i : fin n), w[i] ∈ letters ∧ (∃ (j : fin n), w[j] ∈ vowels)) → 
  (6^5 - 4^5 = 6752) := 
by
  sorry

end five_letter_words_with_vowel_l511_511196


namespace sum_of_digits_1_to_1000_l511_511945

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

def range_sum_of_digits (start : ℕ) (end : ℕ) : ℕ :=
  (List.range' start (end - start + 1)).sum sum_of_digits

theorem sum_of_digits_1_to_1000 : range_sum_of_digits 1 1000 = 14446 :=
by
  sorry

end sum_of_digits_1_to_1000_l511_511945


namespace fraction_cubed_equality_l511_511952

-- Constants for the problem
def A : ℝ := 81000
def B : ℝ := 9000

-- Problem statement
theorem fraction_cubed_equality : (A^3) / (B^3) = 729 :=
by
  sorry

end fraction_cubed_equality_l511_511952


namespace eating_fifth_of_nuts_l511_511904

namespace CrowAndNuts

-- Conditions
def quarter_nuts_eating_time : ℝ := 7.5
def whole_nuts_eating_time : ℝ := 4 * quarter_nuts_eating_time
def fifth_nuts_eating_time : ℝ := whole_nuts_eating_time / 5

-- Theorem to prove
theorem eating_fifth_of_nuts :
  fifth_nuts_eating_time = 6 :=
by
  simp [fifth_nuts_eating_time, whole_nuts_eating_time, quarter_nuts_eating_time]
  sorry

end CrowAndNuts

end eating_fifth_of_nuts_l511_511904


namespace similar_triangles_length_FG_l511_511897

theorem similar_triangles_length_FG (GH KM KL : ℝ) (h1 : GH = 25) (h2 : KM = 15) (h3 : KL = 18) :
  ∃ FG : ℝ, FG = 30 :=
by
  -- As given in the problem, triangles FGH and KLM are similar.
  -- Therefore their corresponding sides are proportional:
  -- FG / KL = GH / KM
  let ratio := 25 / 15
  have h4 : ratio = 5 / 3 := by norm_num
  use 30
  sorry

end similar_triangles_length_FG_l511_511897


namespace initial_pencils_count_l511_511850

theorem initial_pencils_count (pencils_taken : ℕ) (pencils_left : ℕ) (h1 : pencils_taken = 4) (h2 : pencils_left = 75) : 
  pencils_left + pencils_taken = 79 :=
by
  sorry

end initial_pencils_count_l511_511850


namespace initial_cards_l511_511097

variable (x : ℕ)
variable (h1 : x - 3 = 2)

theorem initial_cards (x : ℕ) (h1 : x - 3 = 2) : x = 5 := by
  sorry

end initial_cards_l511_511097


namespace num_eight_sums_to_8000_l511_511954

theorem num_eight_sums_to_8000 : 
  let calc_values := { n | ∃ (a b c : ℕ), 8 * a + 88 * b + 888 * c = 8000 ∧ n = a + 2 * b + 3 * c } in
  calc_values.to_finset.card = 108 := by
  sorry

end num_eight_sums_to_8000_l511_511954


namespace largest_five_digit_number_with_product_120_l511_511555

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def prod_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldr (· * ·) 1

def max_five_digit_prod_120 : ℕ := 85311

theorem largest_five_digit_number_with_product_120 :
  is_five_digit max_five_digit_prod_120 ∧ prod_of_digits max_five_digit_prod_120 = 120 :=
by
  sorry

end largest_five_digit_number_with_product_120_l511_511555


namespace largest_five_digit_product_120_l511_511553

theorem largest_five_digit_product_120 : 
  ∃ n : ℕ, n = 85311 ∧ (nat.digits 10 n).product = 120 ∧ 10000 ≤ n ∧ n < 100000 :=
by
  sorry

end largest_five_digit_product_120_l511_511553


namespace arc_angle_of_circle_with_equilateral_triangle_l511_511035

theorem arc_angle_of_circle_with_equilateral_triangle (a : ℝ) (r : ℝ) 
  (h_eq : h = (√3 / 2) * a) 
  (r_eq_h : r = h) : 
  ∃ φ : ℝ, φ = 60 :=
by
  sorry

end arc_angle_of_circle_with_equilateral_triangle_l511_511035


namespace base7_sum_of_diffs_l511_511522

def base7_sub (a b : ℕ) : ℕ :=
  let dec_a := nat.of_digits 7 a.digits
  let dec_b := nat.of_digits 7 b.digits
  nat.to_digits 7 (dec_a - dec_b)

def base7_add (a b : ℕ) : ℕ :=
  let dec_a := nat.of_digits 7 a.digits
  let dec_b := nat.of_digits 7 b.digits
  nat.to_digits 7 (dec_a + dec_b)

theorem base7_sum_of_diffs :
  let d1 := base7_sub 5243 3105
  let d2 := base7_sub 6665 4312
  base7_add d1 d2 = 4452 := by sorry

end base7_sum_of_diffs_l511_511522


namespace valid_words_no_adjacent_a_l511_511159

def a_n (n : ℕ) : ℝ :=
  let A := (2 + Real.sqrt 3) / (2 * Real.sqrt 3)
  let B := (-2 + Real.sqrt 3) / (2 * Real.sqrt 3)
  in A * ((1 + Real.sqrt 3) ^ n) + B * ((1 - Real.sqrt 3) ^ n)

theorem valid_words_no_adjacent_a (n : ℕ) (h : n ≥ 1) :
  a_n n = (2 + Real.sqrt 3) / (2 * Real.sqrt 3) * ((1 + Real.sqrt 3) ^ n) + 
          (-2 + Real.sqrt 3) / (2 * Real.sqrt 3) * ((1 - Real.sqrt 3) ^ n) :=
by sorry

end valid_words_no_adjacent_a_l511_511159


namespace circulation_ratio_l511_511455

-- Define the conditions
variables (A : ℝ) -- average yearly circulation for the years 1962-1970
def circulation_1961 : ℝ := 4 * A
def total_circulation_1962_1970 : ℝ := 9 * A
def total_circulation_1961_1970 : ℝ := circulation_1961 A + total_circulation_1962_1970 A

-- The proof statement
theorem circulation_ratio (A : ℝ) (h1 : circulation_1961 A = 4 * A) (h2 : total_circulation_1962_1970 A = 9 * A) : 
  circulation_1961 A / total_circulation_1961_1970 A = 4 / 13 := 
by 
  calc
    circulation_1961 A / total_circulation_1961_1970 A
       = (4 * A) / (4 * A + 9 * A) : by rw [h1, h2]
    ... = 4 / 13 : by ring

end circulation_ratio_l511_511455


namespace number_of_solutions_l511_511962

theorem number_of_solutions :
  ∃ (n : ℕ), (n = 4) ∧
  (∃ (θ : ℝ → Prop),
    (∀ θ, θ > 0 ∧ θ ≤ real.pi → 4 - 2 * real.sin θ + 3 * real.cos (2 * θ) = 0) ∧
    set.finite { θ | 0 < θ ∧ θ ≤ real.pi ∧ 4 - 2 * real.sin θ + 3 * real.cos (2 * θ) = 0 } ∧
    { θ | 0 < θ ∧ θ ≤ real.pi ∧ 4 - 2 * real.sin θ + 3 * real.cos (2 * θ) = 0 }.to_finset.card = 4) :=
begin
  sorry
end

end number_of_solutions_l511_511962


namespace projection_of_b_onto_a_l511_511168

variables {E : Type*} [inner_product_space ℝ E]

noncomputable def norm_square (v : E) : ℝ := (∥v∥)^2

theorem projection_of_b_onto_a (a b : E) (h1 : ∥a∥ = 3) (h2 : ∥b∥ = 2 * real.sqrt 3) 
 (h3 : inner_product_space.is_orthogonal a (a + b)) :
  inner_product_space.proj a b = -3 :=
by sorry

end projection_of_b_onto_a_l511_511168


namespace bob_selling_price_per_muffin_l511_511052

variable (dozen_muffins_per_day : ℕ := 12)
variable (cost_per_muffin : ℝ := 0.75)
variable (weekly_profit : ℝ := 63)
variable (days_per_week : ℕ := 7)

theorem bob_selling_price_per_muffin : 
  let daily_cost := dozen_muffins_per_day * cost_per_muffin
  let weekly_cost := daily_cost * days_per_week
  let weekly_revenue := weekly_profit + weekly_cost
  let muffins_per_week := dozen_muffins_per_day * days_per_week
  let selling_price_per_muffin := weekly_revenue / muffins_per_week
  selling_price_per_muffin = 1.50 := 
by
  sorry

end bob_selling_price_per_muffin_l511_511052


namespace root_of_power_eight_l511_511974

theorem root_of_power_eight (h : 3 < π) : (real.sqrt (3 - π) ^ 8) = π - 3 :=
by
  sorry

end root_of_power_eight_l511_511974


namespace sum_of_differences_l511_511597

theorem sum_of_differences : 
  let first_digit (n : ℕ) : ℕ := n / 10^(nat.log10 n)
  let last_digit (n : ℕ) : ℕ := n % 10
  (finset.range 1000).sum (λ n, first_digit n - last_digit n) = 495 :=
by
  sorry

end sum_of_differences_l511_511597


namespace volume_invariant_l511_511357

noncomputable def volume_of_common_region (a b c : ℝ) : ℝ := (5/6) * a * b * c

theorem volume_invariant (a b c : ℝ) (P : ℝ × ℝ × ℝ) (hP : ∀ (x y z : ℝ), 0 ≤ x ∧ x ≤ a ∧ 0 ≤ y ∧ y ≤ b ∧ 0 ≤ z ∧ z ≤ c) :
  volume_of_common_region a b c = (5/6) * a * b * c :=
by sorry

end volume_invariant_l511_511357


namespace decreasing_on_interval_l511_511354

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 + 3 * x^2 - 12 * x + 1

theorem decreasing_on_interval : ∀ (x : ℝ), x ∈ set.Ioo (-2 : ℝ) 1 → (derivative f x) < 0 := 
by 
  -- Here we use notation to indicate that the proof is omitted
  sorry

end decreasing_on_interval_l511_511354


namespace number_of_elements_in_M_l511_511070

def M : Set ℕ := { y | ∃ x : ℕ, y = 8 / (x + 3) ∧ y ∈ ℕ }

theorem number_of_elements_in_M : (M.to_finset.card = 2) :=
sorry

end number_of_elements_in_M_l511_511070


namespace ratio_AP_over_PC_eq_2_AB_over_BC_l511_511914

noncomputable theory

-- Define point type and circle
variable {Point : Type} [Inhabited Point]
structure Circle (Point) :=
(center : Point)
(radius : ℝ)

-- Define line intersection with circle and other essential properties
variables (A B C S T P : Point) (ℂ : Circle Point)

-- Define axioms based on the conditions given
axiom line_through_A_intersects_Circle_at_B_and_C : ∃ (ℂ : Circle Point), ℂ.intersects A B C ∧ B.is_between A C
axiom tangents_from_A_touch_Circle_at_S_and_T : is_tangent S ℂ ∧ is_tangent T ℂ
axiom P_intersection_of_ST_and_AC : is_intersection P S T A C

-- Prove the target statement
theorem ratio_AP_over_PC_eq_2_AB_over_BC :
  ∀ (A B C P : Point), (is_between B A C)  → 
  (is_intersection P (line_through S T) (line_through A C)) → 
  (sphere_contains_point ℂ A) ∧ (sphere_contains_point ℂ B) ∧ (sphere_contains_point ℂ C) →
  (is_tangent S ℂ) ∧ (is_tangent T ℂ) →
  ((distance A P) / (distance P C) = 2 * (distance A B) / (distance B C)) :=
sorry

end ratio_AP_over_PC_eq_2_AB_over_BC_l511_511914


namespace perpendicular_lines_m_value_l511_511278

def is_perpendicular (m : ℝ) : Prop :=
    let slope1 := 1 / 2
    let slope2 := -2 / m
    slope1 * slope2 = -1

theorem perpendicular_lines_m_value (m : ℝ) (h : is_perpendicular m) : m = 1 := by
    sorry

end perpendicular_lines_m_value_l511_511278


namespace international_call_cost_per_minute_l511_511099

theorem international_call_cost_per_minute 
  (local_call_minutes : Nat)
  (international_call_minutes : Nat)
  (local_rate : Nat)
  (total_cost_cents : Nat) 
  (spent_dollars : Nat) 
  (spent_cents : Nat)
  (local_call_cost : Nat)
  (international_call_total_cost : Nat) : 
  local_call_minutes = 45 → 
  international_call_minutes = 31 → 
  local_rate = 5 → 
  total_cost_cents = spent_dollars * 100 → 
  spent_dollars = 10 → 
  local_call_cost = local_call_minutes * local_rate → 
  spent_cents = spent_dollars * 100 → 
  total_cost_cents = spent_cents →  
  international_call_total_cost = total_cost_cents - local_call_cost → 
  international_call_total_cost / international_call_minutes = 25 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end international_call_cost_per_minute_l511_511099


namespace maria_correct_answers_l511_511287

theorem maria_correct_answers (x : ℕ) (n c d s : ℕ) (h1 : n = 30) (h2 : c = 20) (h3 : d = 5) (h4 : s = 325)
  (h5 : n = x + (n - x)) : 20 * x - 5 * (30 - x) = 325 → x = 19 :=
by 
  intros h_eq
  sorry

end maria_correct_answers_l511_511287


namespace factorize1_factorize2_factorize3_factorize4_l511_511975

-- Statement for the first equation
theorem factorize1 (a x : ℝ) : 
  a * x^2 - 7 * a * x + 6 * a = a * (x - 6) * (x - 1) :=
sorry

-- Statement for the second equation
theorem factorize2 (x y : ℝ) : 
  x * y^2 - 9 * x = x * (y + 3) * (y - 3) :=
sorry

-- Statement for the third equation
theorem factorize3 (x y : ℝ) : 
  1 - x^2 + 2 * x * y - y^2 = (1 + x - y) * (1 - x + y) :=
sorry

-- Statement for the fourth equation
theorem factorize4 (x y : ℝ) : 
  8 * (x^2 - 2 * y^2) - x * (7 * x + y) + x * y = (x + 4 * y) * (x - 4 * y) :=
sorry

end factorize1_factorize2_factorize3_factorize4_l511_511975


namespace predict_value_of_x_l511_511399

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def regression_coefficients (xs ys : List ℝ) : ℝ × ℝ :=
  let n := xs.length
  let x_bar := mean xs
  let y_bar := mean ys
  let xys_sum := (List.zipWith (· * ·) xs ys).sum
  let xs_sq_sum := (xs.map (· ^ 2)).sum
  let b := (xys_sum - n * x_bar * y_bar) / (xs_sq_sum - n * x_bar ^ 2)
  let a := y_bar - b * x_bar
  (a, b)

theorem predict_value_of_x (xs ys : List ℝ) (y_target : ℝ) :
  (xs = [16, 14, 12, 8]) →
  (ys = [11, 9, 8, 5]) →
  let (a, b) := regression_coefficients xs ys
  let x_pred := (y_target - a) / b
  x_pred ≈ 15 :=
by
  intros hxs hys hab
  sorry

end predict_value_of_x_l511_511399


namespace grasshoppers_cannot_return_l511_511405

-- Define the initial positions of the grasshoppers
def initial_order : List Char := ['A', 'B', 'C']

-- Define a function that checks if the grasshoppers are in the initial position
def is_initial_order (positions : List Char) : Bool :=
  positions = initial_order

-- Define the jumping mechanism
def jump (positions : List Char) : List (List Char) :=
  match positions with
  | [a, b, c] => [[b, a, c], [a, c, b], [c, a, b]]
  | _ => []

-- Prove that after 1985 jumps the grasshoppers cannot return to the initial order
theorem grasshoppers_cannot_return (n : Nat) : ¬ ∃ (positions : List Char),
  (positions = initial_order) ∧ 
  ((List.iterate jump n initial_order) = [initial_order]) :=
by
  sorry

end grasshoppers_cannot_return_l511_511405


namespace solution_x_y_l511_511092

noncomputable def eq_values (x y : ℝ) := (
  x ≠ 0 ∧ x ≠ 1 ∧ y ≠ 0 ∧ y ≠ 3 ∧ (3/x + 2/y = 1/3)
)

theorem solution_x_y (x y : ℝ) (h : eq_values x y) : x = 9 * y / (y - 6) :=
sorry

end solution_x_y_l511_511092


namespace f_10_value_l511_511144

def f : ℕ → ℕ
| 1 := 1
| 2 := 3
| n := if n ≥ 3 then f (n - 2) + f (n - 1) else 0

theorem f_10_value : f 10 = 123 :=
by
  sorry

end f_10_value_l511_511144


namespace rectangle_perimeter_l511_511489

theorem rectangle_perimeter {w l : ℝ} 
  (h_area : l * w = 450)
  (h_length : l = 2 * w) :
  2 * (l + w) = 90 :=
by sorry

end rectangle_perimeter_l511_511489


namespace find_median_l511_511493

-- Define conditions and elements of the sequence
variables seq : List ℤ
variables (mode : ℤ) (mean : ℤ) (smallest : ℤ) (n : ℕ) (m : ℤ)
(h_mode : mode = 26) (h_mean : mean = 20) (h_smallest : smallest = 8) (h_n : n = 6) 
(h_median_replacement1 : (List.take (n / 2) seq).sum / (n / 2) = mean ∧ 
                        (List.modifyNth' seq (n/2) (λ _ => m + 12) 
                        |> List.slice ((n-1)/2) ((n + 1)/2) = [m + 12]))
(h_median_replacement2 : (List.modifyNth' seq (n/2) (λ _ => m - 10) 
                        |> List.slice ((n-1)/2) ((n + 1)/2) = [m - 5]))
(h_seq : List.sort List.Less seq = [8, 15, m, 26, 26, 26])

-- Prove that the median m is 20
theorem find_median : m = 20 := 
by
  sorry

end find_median_l511_511493


namespace parabola_equation_and_max_slope_l511_511683

-- Define the parabola parameters and conditions
def parabola (p : ℝ) (hp : p > 0) : set (ℝ × ℝ) := 
{xy | xy.2 ^ 2 = 2 * p * xy.1}

def focus_distance (d : ℝ) : Prop := 
d = 2

-- Define the points O, P, and Q and the given vector relationship
def point_o : ℝ × ℝ := (0, 0)

def on_parabola (p : ℝ) (hp : p > 0) (P : ℝ × ℝ) : Prop :=
P ∈ parabola p hp

def vector_relationship (P Q F : ℝ × ℝ) : Prop :=
P.1 - Q.1 = 9 * (Q.1 - F.1) ∧ P.2 - Q.2 = 9 * (Q.2 - F.2)

-- Define the conditions and the proof goals
theorem parabola_equation_and_max_slope :
  ∃ p (hp : p > 0) (F : ℝ × ℝ),
  focus_distance (2 * p) →
  (∀ P, on_parabola p hp P → 
       ∃ Q, vector_relationship P Q F →
             (parabola p hp = (λ xy, xy.2^2 = 4 * xy.1) ∧
             (real.slope point_o Q ≤ (1/3))) :=
by 
  -- Proof is omitted
  sorry

end parabola_equation_and_max_slope_l511_511683


namespace domain_of_sqrt_cos_half_range_of_fraction_sin_l511_511460

theorem domain_of_sqrt_cos_half (x : ℝ) : 
  ∃ y, y = sqrt (1 - cos (x / 2)) ↔ x ∈ ℝ := sorry

theorem range_of_fraction_sin (y : ℝ) : 
  (∃ x, y = (3 * sin x + 1) / (sin x - 2)) ↔ y ∈ set.Icc (-4 : ℝ) (2 / 3) := sorry

end domain_of_sqrt_cos_half_range_of_fraction_sin_l511_511460


namespace divides_iff_l511_511503

def a : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 2 * a (n + 1) + a n

theorem divides_iff (n k : ℕ) : (2^k ∣ a n) ↔ (2^k ∣ n) :=
by sorry

end divides_iff_l511_511503


namespace largest_number_with_digits_product_120_is_85311_l511_511557

-- Define the five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the product of digits
def digits_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

-- Define the condition that the product of the digits should be 120
def product_is_120 (n : ℕ) : Prop :=
  digits_product n = 120

-- Define the condition that n is the largest such number
def largest_such_number (n : ℕ) : Prop :=
  ∀ m : ℕ, is_five_digit m → product_is_120 m → m ≤ n

-- The theorem stating that the largest five-digit number whose digits' product equals 120 is 85311
theorem largest_number_with_digits_product_120_is_85311 : ∃ n : ℕ, is_five_digit n ∧ product_is_120 n ∧ largest_such_number n ∧ n = 85311 :=
by
  use 85311
  split
  -- Prove 85311 is a five-digit number
  sorry
  split
  -- Prove the product of the digits of 85311 is 120
  sorry
  split
  -- Prove that 85311 is the largest such number
  sorry
  -- Prove n = 85311
  sorry

end largest_number_with_digits_product_120_is_85311_l511_511557


namespace correct_propositions_l511_511851

-- Definitions of the conditions
def parallelepiped_with_two_rectangular_faces_is_right_prism : Prop :=
  ∃ (P : Type), is_parallelepiped P ∧ has_two_rectangular_faces P → is_right_prism P

def solid_of_revolution_about_leg_of_right_angled_triangle_is_cone : Prop :=
  ∀ (R : Type) (T : Type), is_right_angled_triangle T ∧ 
  solid_of_revolution R T ∧ rotates_about_leg R T → is_cone R

def line_parallel_within_plane_is_parallel_to_plane : Prop :=
  ∀ (β : Type) (m : Type) (b : Type), 
  line_in_plane m β ∧ line_parallel_to b m → line_parallel_to b β

def existence_of_two_skew_lines_with_properties : Prop :=
  ∃ (a : Type) (b : Type) (α : Type) (β : Type),
  line_in_plane a α ∧ line_in_plane b β ∧ 
  line_parallel_to a β ∧ line_parallel_to b α ∧ skew_lines a b

-- The proof statement
theorem correct_propositions : 
  parallelepiped_with_two_rectangular_faces_is_right_prism = false ∧
  solid_of_revolution_about_leg_of_right_angled_triangle_is_cone = true ∧
  line_parallel_within_plane_is_parallel_to_plane = false ∧
  existence_of_two_skew_lines_with_properties = true :=
sorry

end correct_propositions_l511_511851


namespace arithmetic_sequence_unique_a_l511_511690

theorem arithmetic_sequence_unique_a (a : ℝ) (b : ℕ → ℝ) (a_seq : ℕ → ℝ)
  (h1 : a_seq 1 = a) (h2 : a > 0)
  (h3 : b 1 - a_seq 1 = 1) (h4 : b 2 - a_seq 2 = 2)
  (h5 : b 3 - a_seq 3 = 3)
  (unique_a : ∀ (a' : ℝ), (a_seq 1 = a' ∧ a' > 0 ∧ b 1 - a' = 1 ∧ b 2 - a_seq 2 = 2 ∧ b 3 - a_seq 3 = 3) → a' = a) :
  a = 1 / 3 :=
by
  sorry

end arithmetic_sequence_unique_a_l511_511690


namespace total_contribution_is_1040_l511_511938

-- Definitions of contributions based on conditions.
def Niraj_contribution : ℕ := 80
def Brittany_contribution : ℕ := 3 * Niraj_contribution
def Angela_contribution : ℕ := 3 * Brittany_contribution

-- Statement to prove that total contribution is $1040.
theorem total_contribution_is_1040 : Niraj_contribution + Brittany_contribution + Angela_contribution = 1040 := by
  sorry

end total_contribution_is_1040_l511_511938


namespace total_fish_l511_511360

theorem total_fish (goldfish bluefish : ℕ) (h1 : goldfish = 15) (h2 : bluefish = 7) : goldfish + bluefish = 22 := 
by
  sorry

end total_fish_l511_511360


namespace find_t_closest_to_vec_a_l511_511565

open Real

def vec_v (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 5 * t, -3 + 7 * t, -3 - 2 * t)

def vec_a : ℝ × ℝ × ℝ :=
  (4, 4, 5)

def direction_vec : ℝ × ℝ × ℝ :=
  (5, 7, -2)

theorem find_t_closest_to_vec_a : ∃ t : ℝ, t = 43 / 78 ∧ 
  let dv := vec_v t in
  let diff := (dv.1 - vec_a.1, dv.2 - vec_a.2, dv.3 - vec_a.3) in
  let dot_product := diff.1 * direction_vec.1 + diff.2 * direction_vec.2 + diff.3 * direction_vec.3 in
  dot_product = 0 :=
by
  sorry

end find_t_closest_to_vec_a_l511_511565


namespace solve_given_triangle_l511_511716

noncomputable def solve_triangle (b c A : ℝ) (h_c : c = 2 * Real.sqrt 3) (h_A : A = Real.pi / 6) (h_b : b = 3) : 
  Prop := 
  let a := Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A) in
  let B := Real.asin (b * Real.sin A / a) in
  let C := Real.pi - A - B in
  a = Real.sqrt 3 ∧ B = Real.pi / 3 ∧ C = Real.pi / 2

theorem solve_given_triangle : 
  solve_triangle 3 (2 * Real.sqrt 3) (Real.pi / 6) (by rfl) (by rfl) (by rfl) := 
  sorry

end solve_given_triangle_l511_511716


namespace education_expenses_l511_511032

theorem education_expenses (rent milk groceries petrol miscellaneous savings total_salary education : ℝ) 
  (h_rent : rent = 5000)
  (h_milk : milk = 1500)
  (h_groceries : groceries = 4500)
  (h_petrol : petrol = 2000)
  (h_miscellaneous : miscellaneous = 6100)
  (h_savings : savings = 2400)
  (h_saving_percentage : savings = 0.10 * total_salary)
  (h_total_salary : total_salary = savings / 0.10)
  (h_total_expenses : total_salary - savings = rent + milk + groceries + petrol + miscellaneous + education) :
  education = 2500 :=
by
  sorry

end education_expenses_l511_511032


namespace part1_part2_l511_511152

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem part1 (h : ∀ x : ℝ, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) : a = 2 :=
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, |x - 1| + |x + 4| ≥ m) ↔ m ∈ set.Iic 5 :=
  sorry

end part1_part2_l511_511152


namespace function_passes_through_fixed_point_l511_511669

noncomputable def fixed_point (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) : Prop :=
  ∃ x y : ℝ, (x = 1 ∧ y = 4) ∧ (y = a^(x-1) + 3)

theorem function_passes_through_fixed_point (a : ℝ) (h_pos : 0 < a) (h_ne_one : a ≠ 1) : 
  fixed_point a h_pos h_ne_one :=
by
  use 1
  use 4
  split
  { exact ⟨rfl, rfl⟩ }
  { sorry }

end function_passes_through_fixed_point_l511_511669


namespace exists_set_A_l511_511966

noncomputable def f(n : ℕ) : ℕ :=
  ν 2 n + 4 * ν 3 n + 9 * ν 5 n + 11 * ν 7 n + 7 * ν 11 n + 14 * ν 13 n

theorem exists_set_A :
  (∃ A ⊆ ℕ+, ∀ n : ℕ+, (A ∩ {k : ℕ+ | ∃ m ∈ {1, 2, 3, ..., 15}, k = m * n}) = 1) ∧
  (∃ᶠ m in at_top, {m, m + 2018} ⊆ A) :=
sorry

end exists_set_A_l511_511966


namespace majority_votes_l511_511289

theorem majority_votes (total_votes : ℕ) (percent_won : ℝ) : total_votes = 450 ∧ percent_won = 0.7 → (total_votes * percent_won - total_votes * (1 - percent_won) = 180) :=
by {
  assume h,
  cases h with h_total h_percent,
  sorry
}

end majority_votes_l511_511289


namespace hyperbola_no_intersection_l511_511326

theorem hyperbola_no_intersection (a b e : ℝ)
  (ha : 0 < a) (hb : 0 < b)
  (h_e : e = (Real.sqrt (a^2 + b^2)) / a) :
  (√5 ≥ e ∧ 1 < e) → ∀ x y : ℝ, ¬ (y = 2 * x ∧ (x^2 / a^2 - y^2 / b^2 = 1)) :=
begin
  intros h_intersect x y,
  sorry,
end

end hyperbola_no_intersection_l511_511326


namespace expression_parity_l511_511762

theorem expression_parity (p m : ℤ) (hp : Odd p) : (Odd (p^3 + m * p)) ↔ Even m := by
  sorry

end expression_parity_l511_511762


namespace one_vertical_asymptote_l511_511637

theorem one_vertical_asymptote (k : ℝ) : 
  (∀ x, x ≠ 3 → x ≠ 1 → g x ≠ 0) ↔ (k = 0 ∨ k = 2) :=
by
  let g := λ x, (x^2 - 3 * x + k) / (x^2 - 4 * x + 3)
  sorry

end one_vertical_asymptote_l511_511637


namespace classify_vowels_correctly_l511_511386

-- Define categories based on the classification
def Category := ℕ

def C1 : Category := 1 -- Two axes of symmetry
def C2 : Category := 2 -- Central symmetric figures
def C3 : Category := 3 -- One horizontal axis of symmetry
def C4 : Category := 4 -- One vertical axis of symmetry
def C5 : Category := 5 -- Neither axisymmetric nor centrally symmetric

-- Define the vowel letters
def A : String := "A"
def E : String := "E"
def I : String := "I"
def O : String := "O"
def U : String := "U"

-- Define the expected classifications
def classify_letter (letter : String) : Category :=
  if letter = A then C4
  else if letter = E then C3
  else if letter = I then C1
  else if letter = O then C1
  else if letter = U then C4
  else 0 -- Default case for non-vowel letters

theorem classify_vowels_correctly :
  classify_letter A = C4 ∧
  classify_letter E = C3 ∧
  classify_letter I = C1 ∧
  classify_letter O = C1 ∧
  classify_letter U = C4 :=
by
  sorry

end classify_vowels_correctly_l511_511386


namespace parabola_intersection_subtraction_l511_511393

theorem parabola_intersection_subtraction :
  (∃ a b c d : ℝ, (y = 3 * a ^ 2 - 6 * a + 6) ∧ (y = -c ^ 2 - 4 * c + 6) ∧ (c ≥ a) ∧ (c - a = 1/2)) × (h1 : ∀ x : ℝ, (3 * x ^ 2 - 6 * x + 6 = -x ^ 2 - 4 * x + 6) → (x = 0 ∨ x = 1/2)) →
  (c - a = 1/2) :=
begin
  sorry
end

end parabola_intersection_subtraction_l511_511393


namespace limit_max_xy_l511_511532

noncomputable def ellipse (n : ℕ) : set (ℝ × ℝ) :=
  { p : ℝ × ℝ | let (x, y) := p in (x^2 / 4) + (n * y^2 / (4 * n + 1)) = 1 }

noncomputable def max_xy (n : ℕ) : ℝ :=
  let paramx (θ : ℝ) := 2 * Real.cos θ in
  let paramy (θ : ℝ) := Real.sqrt (4 + 1 / n) * Real.sin θ in
  Real.sqrt(8 + 1 / n)

theorem limit_max_xy : 
  filter.tendsto (max_xy) filter.at_top (nhds (2 * Real.sqrt 2)) :=
sorry

end limit_max_xy_l511_511532


namespace andrew_expected_distinct_colors_l511_511043

noncomputable def expected_distinct_colors_picks (balls: ℕ) (picks: ℕ) : ℚ :=
  let prob_not_picked_once := (balls - 1) / balls
  let prob_not_picked := prob_not_picked_once ^ picks
  let prob_picked := 1 - prob_not_picked
  balls * prob_picked

theorem andrew_expected_distinct_colors :
  (expected_distinct_colors_picks 10 4) = (3439 / 1000) :=
by sorry

end andrew_expected_distinct_colors_l511_511043


namespace sum_of_xs_l511_511338

theorem sum_of_xs (x y z : ℂ) : (x + y * z = 8) ∧ (y + x * z = 12) ∧ (z + x * y = 11) → 
    ∃ S, ∀ (xi yi zi : ℂ), (xi + yi * zi = 8) ∧ (yi + xi * zi = 12) ∧ (zi + xi * yi = 11) →
        xi + yi + zi = S :=
by
  sorry

end sum_of_xs_l511_511338


namespace quadrilateral_area_l511_511822

theorem quadrilateral_area (ABCD_area : ℝ) (AB BE BC CF CD DG DA AH : ℝ)
  (h₁ : ABCD_area = 1)
  (h₂ : AB = BE)
  (h₃ : BC = CF)
  (h₄ : CD = DG)
  (h₅ : DA = AH) :
  let EFGH_area := 5 in
  EFGH_area = 5 :=
by
  sorry

end quadrilateral_area_l511_511822


namespace hyperbola_line_intersects_l511_511531

theorem hyperbola_line_intersects
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (k : ℝ) (C : set (ℝ × ℝ))
  (hC : ∀ x y : ℝ, (x, y) ∈ C ↔ (x^2 / a^2 - y^2 / b^2 = 1)) :
  (-b / a < k ∧ k < b / a) ↔
  ∃ x1 x2 y1 y2 : ℝ, (x1, y1) ∈ C ∧ (x2, y2) ∈ C ∧
  ∃ F : ℝ × ℝ, (F.1, F.2) = ((a * cosh(0)), 0) ∧
    (line_through_focus_slope k F).intersects_both_branches C :=
sorry

end hyperbola_line_intersects_l511_511531


namespace total_customers_l511_511926

theorem total_customers (tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ)
  (h_tables : tables = 7) (h_women_per_table : women_per_table = 7)
  (h_men_per_table : men_per_table = 2) :
  tables * (women_per_table + men_per_table) = 63 := by
begin
  sorry
end

end total_customers_l511_511926


namespace circles_have_common_point_l511_511297

-- Definitions and conditions
variable (A B C M M_A M_B M_C K_A K_B K_C: Type)
variable [Triangle A B C]
variable [Median A M_A B M_B C M_C]
variable [Intersection (Median A M_A) (Median B M_B) M]
variable [Midpoint K_A A M]
variable [Midpoint K_B B M]
variable [Midpoint K_C C M]
variable [Circle Omega_A M_A K_A tangent_to_segment B C]
variable [Circle Omega_B M_B K_B tangent_to_segment A C]
variable [Circle Omega_C M_C K_C tangent_to_segment A B]

-- The theorem to prove
theorem circles_have_common_point :
  ∃ X, (Omega_A.pass_through X) ∧ (Omega_B.pass_through X) ∧ (Omega_C.pass_through X) := 
sorry

end circles_have_common_point_l511_511297


namespace find_angle_F_l511_511107

variable {α : Type} [ordered_semiring α]

-- Define the angles and congruence of triangles
variables (A B C D E F : α)
variables (triangleABC_congruent_triangleDEF : triangle A B C ≅ triangle D E F)

-- Given conditions
def angle_A_eq_35 : A = 35 := sorry
def angle_B_eq_75 : B = 75 := sorry

-- The theorem to prove
theorem find_angle_F : F = 70 :=
by
  have h : A + B + C = 180 := sorry  -- sum of angles in triangle ABC
  have congruent_angles : C = F := sorry  -- congruence property: ∠C = ∠F
  -- Calculate angle C
  have angle_C_eq : C = 180 - A - B := sorry
  rw [angle_A_eq_35, angle_B_eq_75] at angle_C_eq
  simp only [sub_eq_add_neg, add_neg_cancel_right] at angle_C_eq
  rw [congruent_angles, angle_C_eq]
  norm_num  -- finalize the proof for numerical calculation

end find_angle_F_l511_511107


namespace sample_size_is_10_l511_511100

def product := Type

noncomputable def number_of_products : ℕ := 80
noncomputable def selected_products_for_quality_inspection : ℕ := 10

theorem sample_size_is_10 
  (N : ℕ) (sample_size : ℕ) 
  (hN : N = 80) 
  (h_sample_size : sample_size = 10) : 
  sample_size = 10 :=
by 
  sorry

end sample_size_is_10_l511_511100


namespace solve_for_z_l511_511365

variable {z : ℂ}
def complex_i := Complex.I

theorem solve_for_z (h : 1 - complex_i * z = -1 + complex_i * z) : z = -complex_i := by
  sorry

end solve_for_z_l511_511365


namespace five_letter_words_with_vowel_l511_511194

theorem five_letter_words_with_vowel : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let n := 5 
  (∃ (w : list(char)), w.length = n ∧ ∀ (i : fin n), w[i] ∈ letters ∧ (∃ (j : fin n), w[j] ∈ vowels)) → 
  (6^5 - 4^5 = 6752) := 
by
  sorry

end five_letter_words_with_vowel_l511_511194


namespace wolstenholme_theorem_l511_511571

def is_odd_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ p % 2 = 1

def S (p : ℕ) : ℕ :=
  (Finset.range (p / 2)).sum (λ i, ((1 : ℚ) / ((2 * i + 1) * 2 * i)))

theorem wolstenholme_theorem (p : ℕ) [fact : Nat.Prime p] (h_odd_prime : is_odd_prime p) :
  ((2^p - 2) % (p^2) = 0 ↔ (S p) % p = 0) :=
sorry

end wolstenholme_theorem_l511_511571


namespace best_fitting_model_is_model3_l511_511285

-- Definitions of the coefficients of determination for the models
def R2_model1 : ℝ := 0.60
def R2_model2 : ℝ := 0.90
def R2_model3 : ℝ := 0.98
def R2_model4 : ℝ := 0.25

-- The best fitting effect corresponds to the highest R^2 value
theorem best_fitting_model_is_model3 :
  R2_model3 = max (max R2_model1 R2_model2) (max R2_model3 R2_model4) :=
by {
  -- Proofblock is skipped, using sorry
  sorry
}

end best_fitting_model_is_model3_l511_511285


namespace minimum_triangle_perimeter_l511_511737

theorem minimum_triangle_perimeter :
  ∃ (a b c : ℕ), 
    (∃ A B C : ℝ, A = 2 * B ∧ C = 180 - 3 * B ∧ C > 90) ∧ 
    (sin B ≠ 0 ∧ sin A / sin B = a / b ∧ sin (180 - 3 * B) / sin (180 - 3 * B) = c / b) ∧
    a + b + c = 77 :=
sorry

end minimum_triangle_perimeter_l511_511737


namespace sqrt_product_simplified_l511_511519

theorem sqrt_product_simplified (q : ℝ) (hq : 0 < q) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by
  sorry

end sqrt_product_simplified_l511_511519


namespace sum_of_differences_l511_511587

open Nat
open BigOperators

theorem sum_of_differences (n : ℕ) (h : n ≥ 1 ∧ n ≤ 999) : 
  let differences := (fun x => 
                        let first_digit := x / 10;
                        let last_digit := x % 10;
                        first_digit - last_digit) in
  ∑ i in (finset.range (n + 1)).filter (λ x, 1 ≤ x ∧ x ≤ 999), differences i = 495 :=
by
  -- Acknowledge the need for a more refined filtering criteria for numbers between 1 and 999
  sorry

end sum_of_differences_l511_511587


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511189

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the vowels
def vowels := {'A', 'E'}

-- Define the number of n-length words constructible from a set of letters
def num_words (n : ℕ) (alphabet : Set Char) : ℕ :=
  (alphabet.size ^ n)

-- The total number of 5-letter words (unrestricted)
def total_words := num_words 5 letters

-- The number of 5-letter words with no vowels
def no_vowel_words := num_words 5 (letters \ vowels)

-- The number of 5-letter words with at least one vowel
def at_least_one_vowel_words := total_words - no_vowel_words

-- The statement to prove that the number of 5-letter words with at least one vowel is 6752
theorem number_of_5_letter_words_with_at_least_one_vowel : 
  at_least_one_vowel_words = 6752 :=
by 
  -- Proof will be provided here
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l511_511189


namespace probability_green_is_five_ninths_l511_511064

def Container := ℕ × ℕ -- Define a container as a pair of red and green balls

def containerI : Container := (8, 4)
def containerII : Container := (2, 4)
def containerIII : Container := (2, 4)

def containers : List Container := [containerI, containerII, containerIII]

-- Function to calculate the probability of drawing a green ball from a container
noncomputable def probability_green (c : Container) : ℚ :=
  c.snd / (c.fst + c.snd : ℕ)

-- Function to calculate the total probability 
noncomputable def total_probability_green (cs : List Container) : ℚ :=
  cs.map (λ c => probability_green c / cs.length).sum

theorem probability_green_is_five_ninths :
  total_probability_green containers = 5 / 9 := by
  sorry

end probability_green_is_five_ninths_l511_511064


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511190

-- Define the set of letters
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}

-- Define the vowels
def vowels := {'A', 'E'}

-- Define the number of n-length words constructible from a set of letters
def num_words (n : ℕ) (alphabet : Set Char) : ℕ :=
  (alphabet.size ^ n)

-- The total number of 5-letter words (unrestricted)
def total_words := num_words 5 letters

-- The number of 5-letter words with no vowels
def no_vowel_words := num_words 5 (letters \ vowels)

-- The number of 5-letter words with at least one vowel
def at_least_one_vowel_words := total_words - no_vowel_words

-- The statement to prove that the number of 5-letter words with at least one vowel is 6752
theorem number_of_5_letter_words_with_at_least_one_vowel : 
  at_least_one_vowel_words = 6752 :=
by 
  -- Proof will be provided here
  sorry

end number_of_5_letter_words_with_at_least_one_vowel_l511_511190


namespace eighteenth_entry_is_26_l511_511569

-- Define r_6
def r_6 (n : ℕ) : ℕ := n % 6

-- Define the condition for the sequence
def satisfies_condition (n : ℕ) : Prop := r_6 (4 * n) ≤ 3

-- Define the sequence of numbers that satisfy the condition
def sequence : List ℕ := (List.range 1000).filter satisfies_condition -- range large enough to cover first 18 items

-- Extract the 18th element (index 17 since Lean indices start at 0)
def eighteenth_entry : ℕ := sequence.get? 17 |>.get_or_else 0

-- The proof statement
theorem eighteenth_entry_is_26 : eighteenth_entry = 26 := by
  -- Sorry is used to skip the proof.
  sorry

end eighteenth_entry_is_26_l511_511569


namespace gcd_probability_is_one_l511_511420

open Set Nat

theorem gcd_probability_is_one :
  let S := {1, 2, 3, 4, 5, 6, 7, 8}
  let total_pairs := (finset.powerset_len 2 (finset.image id S.to_finset)).card
  let non_rel_prime_pairs := 6
  (finset.card (finset.filter (λ (p : Finset ℕ), p.gcdₓ = 1) 
                                (finset.powerset_len 2 (finset.image id S.to_finset)))) / 
  total_pairs = 11 / 14 :=
sorry

end gcd_probability_is_one_l511_511420


namespace find_a2_l511_511142

variable (S a : ℕ → ℕ)

-- Define the condition S_n = 2a_n - 2 for all n
axiom sum_first_n_terms (n : ℕ) : S n = 2 * a n - 2

-- Define the specific lemma for n = 1 to find a_1
axiom a1 : a 1 = 2

-- State the proof problem for a_2
theorem find_a2 : a 2 = 4 := 
by 
  sorry

end find_a2_l511_511142


namespace sum_digit_differences_l511_511611

def first_digit (n : ℕ) : ℕ := 
  (n / 10 ^ ((Nat.log10 n) : ℕ))

def last_digit (n : ℕ) : ℕ := n % 10

def digit_difference (n : ℕ) : ℤ :=
  (first_digit n : ℤ) - (last_digit n : ℤ)

theorem sum_digit_differences :
  (∑ n in Finset.range 1000, digit_difference n) = 495 := 
sorry

end sum_digit_differences_l511_511611


namespace Tod_drove_time_l511_511411

section
variable (distance_north: ℕ) (distance_west: ℕ) (speed: ℕ)

theorem Tod_drove_time :
  distance_north = 55 → distance_west = 95 → speed = 25 → 
  (distance_north + distance_west) / speed = 6 :=
by
  intros
  sorry
end

end Tod_drove_time_l511_511411


namespace sum_digit_differences_l511_511609

def first_digit (n : ℕ) : ℕ := 
  (n / 10 ^ ((Nat.log10 n) : ℕ))

def last_digit (n : ℕ) : ℕ := n % 10

def digit_difference (n : ℕ) : ℤ :=
  (first_digit n : ℤ) - (last_digit n : ℤ)

theorem sum_digit_differences :
  (∑ n in Finset.range 1000, digit_difference n) = 495 := 
sorry

end sum_digit_differences_l511_511609


namespace chicken_leg_analysis_l511_511001

-- Define the weights from both factories
def weights_A : List ℝ := [74, 74, 74, 75, 73, 77, 78, 72, 76, 77]
def weights_B : List ℝ := [78, 74, 77, 73, 75, 75, 74, 74, 75, 75]

-- Define the conditions given in the problem
def mean_A : ℝ := 75
def median_A : ℝ := 74.5
def variance_A : ℝ := 3.4

def mean_B : ℝ := 75
def variance_B : ℝ := 2

-- Define the correct answers found in the solution.
def median_B : ℝ := 75
def mode_A : ℝ := 74
def estimation_B : ℝ := 40
def preferred_factory := "Factory B"

-- Lean statement to prove the calculated values and preference for Factory B
theorem chicken_leg_analysis :
  (∀ (weights : List ℝ), (weights = weights_B) ∧ (median weights = median_B) → (mode weights_A = mode_A) → 
  ((weights_B.filter (λ x, x = 75)).length / weights_B.length * 100 = estimation_B) →
  (variance_B < variance_A) → preferred_factory = "Factory B") := 
by
  intro weights h1 h2 h3 h4
  simp [median, mode, estimation_B, preferred_factory, h1, h2, h3, h4]
  sorry

end chicken_leg_analysis_l511_511001


namespace number_of_paths_l511_511981

theorem number_of_paths (path_A_to_B : ℕ) (path_B_to_C : ℕ) (direct_path_A_to_C : ℕ) :
  path_A_to_B = 2 → path_B_to_C = 2 → direct_path_A_to_C = 1 → ∃ n, n = 5 :=
by
  intros h1 h2 h3
  use 4 + 1
  rw [h1, h2, h3]
  sorry

end number_of_paths_l511_511981


namespace sum_of_differences_l511_511602

theorem sum_of_differences : 
  let first_digit (n : ℕ) : ℕ := n / 10^(nat.log10 n)
  let last_digit (n : ℕ) : ℕ := n % 10
  (finset.range 1000).sum (λ n, first_digit n - last_digit n) = 495 :=
by
  sorry

end sum_of_differences_l511_511602


namespace divisibility_sum_of_zw_l511_511534

theorem divisibility_sum_of_zw :
  let valid_zw (z w : ℕ) := ((22 + z + w) % 3 = 0) ∧ (10 * z + w < 100)
  in (∑ z in (finset.range 10), ∑ w in (finset.range 10), if valid_zw z w then (10 * z + w) else 0) = 42 :=
by
  let valid_zw (z w : ℕ) := ((22 + z + w) % 3 = 0) ∧ (10 * z + w < 100)
  have valid_sums := finset.sum_bij
    (λ ⟨z, hz⟩ ⟨w, hw⟩ => 10 * z + w)
    (by intros; simp [valid_zw])
    (by intros; dec_trivial)
    (λ _ => finset.mem_filter.2 ∘ and.intro)
    (by intros; exact finset.mem_univ _)
  exact finset.sum_eq_zero.mpr valid_sums sorry

end divisibility_sum_of_zw_l511_511534


namespace problem_inequality_solution_problem_prove_inequality_l511_511155

-- Function definition for f(x)
def f (x : ℝ) := |2 * x - 3| + |2 * x + 3|

-- Problem 1: Prove the solution set for the inequality f(x) ≤ 8
theorem problem_inequality_solution (x : ℝ) : f x ≤ 8 ↔ -2 ≤ x ∧ x ≤ 2 :=
sorry

-- Problem 2: Prove a + 2b + 3c ≥ 9 given conditions
theorem problem_prove_inequality (a b c : ℝ) (M : ℝ) (h1 : M = 6)
  (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 1 / a + 1 / (2 * b) + 1 / (3 * c) = M / 6) :
  a + 2 * b + 3 * c ≥ 9 :=
sorry

end problem_inequality_solution_problem_prove_inequality_l511_511155


namespace noah_yearly_call_cost_l511_511787

structure CallBilling (minutes_per_call : ℕ) (charge_per_minute : ℝ) (calls_per_week : ℕ) (weeks_in_year : ℕ) :=
  (total_minutes : ℕ := weeks_in_year * calls_per_week * minutes_per_call)
  (total_cost : ℝ := total_minutes * charge_per_minute)

theorem noah_yearly_call_cost :
  CallBilling 30 0.05 1 52 .total_cost = 78 := by
  sorry

end noah_yearly_call_cost_l511_511787


namespace rationalize_denominator_l511_511804

-- Lean 4 statement
theorem rationalize_denominator : sqrt (5 / 18) = sqrt 10 / 6 := by
  sorry

end rationalize_denominator_l511_511804


namespace set_intersection_l511_511766

theorem set_intersection (S T : set ℝ) :
  (S = {x | x < -5 ∨ x > 5}) →
  (T = {x | -7 < x ∧ x < 3}) →
  (S ∩ T = {x | -7 < x ∧ x < -5}) :=
by { intros hS hT, sorry }

end set_intersection_l511_511766


namespace sum_of_differences_l511_511620

theorem sum_of_differences : 
  (∑ n in Finset.range 1000, let first_digit := n / 10 ^ (Nat.log10 n) in
                             let last_digit := n % 10 in
                             first_digit - last_digit) = 495 :=
by
  sorry

end sum_of_differences_l511_511620


namespace arithmetic_sequence_a5_l511_511143

noncomputable def a (n : ℕ) (a₁ d : ℝ) : ℝ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_a5 (a₁ d : ℝ) (h1 : a 2 a₁ d = 2 * a 3 a₁ d + 1) (h2 : a 4 a₁ d = 2 * a 3 a₁ d + 7) :
  a 5 a₁ d = 2 :=
by
  sorry

end arithmetic_sequence_a5_l511_511143


namespace no_viable_schedule_l511_511492

theorem no_viable_schedule :
  ∀ (studentsA studentsB : ℕ), 
    studentsA = 29 → 
    studentsB = 32 → 
    ¬ ∃ (a b : ℕ),
      (a = 29 ∧ b = 32 ∧
      (a * b = studentsA * studentsB) ∧
      (∀ (x : ℕ), x < studentsA * studentsB →
        ∃ (iA iB : ℕ), 
          iA < studentsA ∧ 
          iB < studentsB ∧ 
          -- The condition that each pair is unique within this period
          ((iA + iB) % (studentsA * studentsB) = x))) := by
  sorry

end no_viable_schedule_l511_511492


namespace sin_double_angle_neg_l511_511647

theorem sin_double_angle_neg {α : ℝ} (h : tan α < 0) : sin (2 * α) < 0 := 
  sorry

end sin_double_angle_neg_l511_511647


namespace distance_between_points_l511_511549

-- Define the points
def pointA : ℝ × ℝ := (0, 12)
def pointB : ℝ × ℝ := (9, 0)

-- Define the distance formula
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem distance_between_points :
  dist pointA pointB = 15 :=
by 
  -- Here we would provide the proof, but we skip it as per the instructions.
  sorry

end distance_between_points_l511_511549


namespace valid_solution_l511_511816

theorem valid_solution (x : ℝ) (h1 : x = 3) :
  sqrt (3 * x - 7) - sqrt (3 * x^2 - 13 * x + 13) ≥ 3 * x^2 - 16 * x + 20 :=
sorry

end valid_solution_l511_511816


namespace that_three_digit_multiples_of_5_and_7_l511_511257

/-- 
Define the count_three_digit_multiples function, 
which counts the number of three-digit integers that are multiples of both 5 and 7.
-/
def count_three_digit_multiples : ℕ :=
  let lcm := Nat.lcm 5 7
  let first := (100 + lcm - 1) / lcm * lcm
  let last := 999 / lcm * lcm
  (last - first) / lcm + 1

/-- 
Theorem that states the number of positive three-digit integers that are multiples of both 5 and 7 is 26. 
-/
theorem three_digit_multiples_of_5_and_7 : count_three_digit_multiples = 26 := by
  sorry

end that_three_digit_multiples_of_5_and_7_l511_511257


namespace alyssa_ate_limes_l511_511342

def mikes_limes : ℝ := 32.0
def limes_left : ℝ := 7.0

theorem alyssa_ate_limes : mikes_limes - limes_left = 25.0 := by
  sorry

end alyssa_ate_limes_l511_511342


namespace count_5_letter_words_with_at_least_one_vowel_l511_511203

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := (letters.card) ^ 5

def non_vowel_letters : Finset Char := letters \ vowels

def total_non_vowel_5_letter_words : ℕ := (non_vowel_letters.card) ^ 5

theorem count_5_letter_words_with_at_least_one_vowel :
  total_5_letter_words - total_non_vowel_5_letter_words = 6752 :=
by
  sorry

end count_5_letter_words_with_at_least_one_vowel_l511_511203


namespace number_of_girls_l511_511286

-- Define the number of boys and girls as natural numbers
variable (B G : ℕ)

-- First condition: The number of girls is 458 more than the number of boys
axiom h1 : G = B + 458

-- Second condition: The total number of pupils is 926
axiom h2 : G + B = 926

-- The theorem to be proved: The number of girls is 692
theorem number_of_girls : G = 692 := by
  sorry

end number_of_girls_l511_511286


namespace find_xyz_sum_l511_511165

theorem find_xyz_sum
  (x y z : ℝ)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x^2 + x * y + y^2 = 108)
  (h2 : y^2 + y * z + z^2 = 16)
  (h3 : z^2 + z * x + x^2 = 124) :
  x * y + y * z + z * x = 48 := 
  sorry

end find_xyz_sum_l511_511165


namespace divide_triangle_into_similar_l511_511353

theorem divide_triangle_into_similar (n : ℕ) : ∃ T : Triangle, ∃ n : ℕ, (4^n = 1981) → (∃ T' : Triangle, is_similar T T' ∧ cardinality T' = 1981) := by
  sorry

end divide_triangle_into_similar_l511_511353


namespace maximize_distance_l511_511104

theorem maximize_distance (front_tires_lifetime: ℕ) (rear_tires_lifetime: ℕ):
  front_tires_lifetime = 20000 → rear_tires_lifetime = 30000 → 
  ∃ D, D = 30000 :=
by
  sorry

end maximize_distance_l511_511104


namespace largest_five_digit_product_120_l511_511552

theorem largest_five_digit_product_120 : 
  ∃ n : ℕ, n = 85311 ∧ (nat.digits 10 n).product = 120 ∧ 10000 ≤ n ∧ n < 100000 :=
by
  sorry

end largest_five_digit_product_120_l511_511552


namespace max_sub_min_value_l511_511765

variable {x y : ℝ}

noncomputable def expression (x y : ℝ) : ℝ :=
  (abs (x + y))^2 / ((abs x)^2 + (abs y)^2)

theorem max_sub_min_value :
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 
  (expression x y ≤ 2 ∧ 0 ≤ expression x y) → 
  (∃ m M, m = 0 ∧ M = 2 ∧ M - m = 2) :=
by
  sorry

end max_sub_min_value_l511_511765


namespace sum_of_differences_l511_511574

/-- 
  For each natural number from 1 to 999, Damir subtracts the last digit from the first digit and 
  writes the resulting differences on a board. We are to prove that the sum of all these differences 
  is 495.
-/
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, (first_digit n - last_digit n)) = 495 :=
sorry

/-- 
  Helper function to get the first digit of a natural number.
  Here, n > 0
-/
def first_digit (n : ℕ) : ℕ :=
  n / 10^(n.digits 10 - 1)

/-- 
  Helper function to get the last digit of a natural number.
  Here, n > 0
-/
def last_digit (n : ℕ) : ℕ :=
  n % 10

end sum_of_differences_l511_511574


namespace five_letter_words_with_at_least_one_vowel_l511_511216

open Finset

theorem five_letter_words_with_at_least_one_vowel :
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'}
  let vowels := {'A', 'E'}
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := ((letters \ vowels).card : ℕ) ^ 5
  total_words - no_vowel_words = 6752 :=
by
  let letters := insert 'A' (insert 'B' (insert 'C' (insert 'D' (insert 'E' (singleton 'F')))))
  let vowels := insert 'A' (singleton 'E')
  let consonants := letters \ vowels
  have : letters.card = 6 := by simp
  have : vowels.card = 2 := by simp
  have : consonants.card = 4 := by simp
  let total_words := (letters.card : ℕ) ^ 5
  let no_vowel_words := (consonants.card : ℕ) ^ 5
  calc
    total_words - no_vowel_words
        = 6^5 - 4^5         : by simp
    ... = 7776 - 1024       : by norm_num
    ... = 6752             : by norm_num

end five_letter_words_with_at_least_one_vowel_l511_511216


namespace find_height_large_cuboid_l511_511009

noncomputable theory

variables (L_large W_large : ℝ)
variables (n : ℕ) (L_small W_small H_small V_total : ℝ)

-- Given Conditions
def length_large := 12
def width_large := 14
def num_smaller_cuboids := 56
def length_small := 5
def width_small := 3
def height_small := 2

-- Volume calculation
def volume_small := length_small * width_small * height_small
def total_volume := num_smaller_cuboids * volume_small

-- Height of the larger cuboid is to be proved as 10 meters
theorem find_height_large_cuboid :
  ∀ (L_large W_large : ℝ),
  (L_large = 12) → (W_large = 14) →
  (total_volume = 1680) → 
  (V_total = L_large * W_large * 10) := 
sorry

end find_height_large_cuboid_l511_511009


namespace radius_of_circle_l511_511918

theorem radius_of_circle (P : ℝ) (PQ QR : ℝ) (distance_center_P : ℝ) (r : ℝ) :
  P = 17 ∧ PQ = 12 ∧ QR = 8 ∧ (PQ * (PQ + QR) = (distance_center_P - r) * (distance_center_P + r)) → r = 7 :=
by
  sorry

end radius_of_circle_l511_511918


namespace demand_function_is_correct_tax_revenue_collected_at_30_optimal_tax_rate_is_60_max_tax_revenue_is_8640_l511_511890

variables {P t : ℝ}

noncomputable def demandFunction : ℝ → ℝ :=
λ P, 688 - 4 * P

noncomputable def taxRevenue (t : ℝ) : ℝ :=
let Pd := 118 in
let Qd := demandFunction Pd in
Qd * t

theorem demand_function_is_correct :
  ∀ P, demandFunction P = 688 - 4 * P := sorry

theorem tax_revenue_collected_at_30 :
  taxRevenue 30 = 6480 := sorry

noncomputable def optimalTaxRate : ℝ :=
288 / 4.8

theorem optimal_tax_rate_is_60 :
  optimalTaxRate = 60 := sorry

noncomputable def maxTaxRevenue : ℝ :=
let t := optimalTaxRate in
288 * t - 2.4 * t^2

theorem max_tax_revenue_is_8640 :
  maxTaxRevenue = 8640 := sorry

end demand_function_is_correct_tax_revenue_collected_at_30_optimal_tax_rate_is_60_max_tax_revenue_is_8640_l511_511890


namespace sum_of_differences_l511_511572

/-- 
  For each natural number from 1 to 999, Damir subtracts the last digit from the first digit and 
  writes the resulting differences on a board. We are to prove that the sum of all these differences 
  is 495.
-/
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, (first_digit n - last_digit n)) = 495 :=
sorry

/-- 
  Helper function to get the first digit of a natural number.
  Here, n > 0
-/
def first_digit (n : ℕ) : ℕ :=
  n / 10^(n.digits 10 - 1)

/-- 
  Helper function to get the last digit of a natural number.
  Here, n > 0
-/
def last_digit (n : ℕ) : ℕ :=
  n % 10

end sum_of_differences_l511_511572


namespace eccentricity_bound_l511_511311

variables {a b c e : ℝ}

-- Definitions of the problem conditions
def hyperbola (x y : ℝ) (a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def line (x : ℝ) := 2 * x
def eccentricity (c a : ℝ) := c / a

-- Proof statement in Lean
theorem eccentricity_bound (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (c : ℝ)
  (h₃ : hyperbola x y a b)
  (h₄ : ∀ x, line x ≠ y) :
  1 < eccentricity c a ∧ eccentricity c a ≤ sqrt 5 :=
sorry

end eccentricity_bound_l511_511311


namespace five_letter_words_with_vowel_l511_511240

-- Define the set of letters and identify vowels
def letters := {'A', 'B', 'C', 'D', 'E', 'F'}
def vowels := {'A', 'E'}

-- Define the problem statement
theorem five_letter_words_with_vowel : 
  (number of 5-letter words with at least one vowel) = 6752 := 
sorry

end five_letter_words_with_vowel_l511_511240


namespace fibonacci_product_l511_511329

noncomputable def fibonacci (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  if n = 1 then 1 else
  fibonacci (n - 1) + fibonacci (n - 2)

theorem fibonacci_product (F : ℕ → ℕ)
  (h1 : F 1 = 1) (h2 : F 2 = 1) (hFn : ∀ n, F (n + 1) = F n + F (n - 1)) :
  (∏ k in Finset.range (50 - 2 + 1) + 2, (F k) / (F (k - 1)) - (F k) / (F (k + 1))) = (F 50) / (F 51) :=
by
  sorry

end fibonacci_product_l511_511329


namespace number_of_5_letter_words_with_at_least_one_vowel_l511_511231

theorem number_of_5_letter_words_with_at_least_one_vowel :
  let total_words := 6^5
  let words_without_vowels := 4^5
  total_words - words_without_vowels = 6752 :=
by
  let total_words := 6^5
  let words_without_vowels := 4^5
  have h_total_words : total_words = 7776 := by norm_num
  have h_words_without_vowels : words_without_vowels = 1024 := by norm_num
  calc
    7776 - 1024 = 6752 : by norm_num

end number_of_5_letter_words_with_at_least_one_vowel_l511_511231


namespace expression_equality_l511_511380

theorem expression_equality (y : ℝ) : (1^(4*y - 1)) / (5^(-1) + 3^(-1)) = 15 / 8 :=
by
  -- The proof steps would go here
  sorry

end expression_equality_l511_511380


namespace fries_remaining_time_l511_511051

def recommendedTime : ℕ := 5 * 60
def timeInOven : ℕ := 45
def remainingTime : ℕ := recommendedTime - timeInOven

theorem fries_remaining_time : remainingTime = 255 :=
by
  sorry

end fries_remaining_time_l511_511051


namespace seating_exists_l511_511933

/-- Assume we have a set of attendees each associated with a graduation year, 
and each attendee can only informally address those who graduated within 2 years before or after them. -/
def attendee := ℕ

def familiar (a1 a2 : attendee) : Prop := abs (a1 - a2) <= 2

def valid_seating (group : List attendee) : Prop :=
  (∀ i, 0 < i ∧ i < group.length → familiar group[i] group[(i + 1) % group.length]) ∧
  (∀ i, (0 = i ∨ i = group.length - 1) → familiar group[i] group[(i + 1) % group.length])

theorem seating_exists (group : List attendee) (h : ∀ a, a ∈ group → ∃ b, b ∈ group ∧ familiar a b) :
  ∃ seating, valid_seating seating :=
sorry

end seating_exists_l511_511933


namespace prob_yellow_higher_than_blue_l511_511417

def prob_yellow_lands_in_bin (k : ℕ) : ℝ := (1 / (2 : ℝ))^k
def prob_blue_lands_in_bin (k : ℕ) : ℝ := (1 / (3 : ℝ))^k

def prob_joint_same_bin (k : ℕ) : ℝ := prob_yellow_lands_in_bin k * prob_blue_lands_in_bin k

def prob_sum_joint_same_bin : ℝ := ∑' k, prob_joint_same_bin k

theorem prob_yellow_higher_than_blue :
  1 - prob_sum_joint_same_bin = 2 / 5 :=
by
  sorry

end prob_yellow_higher_than_blue_l511_511417


namespace count_5_letter_words_with_at_least_one_vowel_l511_511200

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := (letters.card) ^ 5

def non_vowel_letters : Finset Char := letters \ vowels

def total_non_vowel_5_letter_words : ℕ := (non_vowel_letters.card) ^ 5

theorem count_5_letter_words_with_at_least_one_vowel :
  total_5_letter_words - total_non_vowel_5_letter_words = 6752 :=
by
  sorry

end count_5_letter_words_with_at_least_one_vowel_l511_511200


namespace ratio_area_triangle_AMN_square_ABCD_l511_511727

/-- Given a square ABCD and points M and N such that M is one-third of the way from A to B and N is one-third of the way from B to C, 
prove that the ratio of the area of triangle AMN to the area of square ABCD is 1/9. -/
theorem ratio_area_triangle_AMN_square_ABCD (s : ℝ) (h_square : 0 < s) :
  let A := (0, 0) in
  let B := (s, 0) in
  let C := (s, s) in
  ∃ (M N : ℝ × ℝ), M = (s / 3, 0) ∧ N = (s, s / 3) ∧
  (∃ AMN : ℝ, AMN = 1/2 * (s/3) * (2*s/3) ∧ ∃ ABCD : ℝ, ABCD = s^2 ∧ AMN / ABCD = 1 / 9) :=
by {
  let A := (0, 0),
  let B := (s, 0),
  let C := (s, s),
  use [(s / 3, 0), (s, s / 3)],
  split; norm_num,
  split; norm_num,
  use [ (1/2 * (s / 3) * (2 * s / 3)), s^2],
  split; norm_num,
  field_simp,
  exact (le_of_lt h_square),
  sorry
}

end ratio_area_triangle_AMN_square_ABCD_l511_511727


namespace largest_three_digit_solution_l511_511442

theorem largest_three_digit_solution :
  ∃ m : ℤ, 100 ≤ m ∧ m < 1000 ∧ 40 * m ≡ 120 [MOD 200] ∧ (∀ n : ℤ, 100 ≤ n ∧ n < 1000 ∧ 40 * n ≡ 120 [MOD 200] → n ≤ m) :=
begin
  use 998,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num,
    norm_num1,
  },
  { intros n hn1 hn2 hn3,
    -- Further steps require proving that 998 is indeed the largest such m
    sorry, 
  }
end

end largest_three_digit_solution_l511_511442


namespace largest_prime_divisor_to_test_l511_511408

theorem largest_prime_divisor_to_test (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) : 
  ∃ p, Prime p ∧ p ≤ 33 ∧ ∀ q, Prime q ∧ q ≤ 33 → q ≤ p := 
begin
  use 31,
  split,
  { apply nat.prime_of_prime_up_to_37, exact dec_trivial },
  split,
  { norm_num },
  { intro q,
    intro H,
    exact le_of_lt_succ (nat.prime.not_succ_le_prime_of_prime_lt q_prime),
    solve_by_elim [prime_31], }
Qed

end largest_prime_divisor_to_test_l511_511408


namespace total_windows_l511_511916

theorem total_windows (installed_windows : ℕ) (hours_per_window : ℕ) (total_hours : ℕ)
  (h1 : installed_windows = 5) (h2 : hours_per_window = 4) (h3 : total_hours = 36) :
  installed_windows + total_hours / hours_per_window = 14 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end total_windows_l511_511916


namespace angle_A_in_triangle_range_of_f_in_interval_l511_511715

variable {a b c : ℝ}
variable {B : ℝ}

-- Part I
theorem angle_A_in_triangle (h : a * real.sin B + real.sqrt 3 * a * real.cos B = real.sqrt 3 * c) 
(h_triangle : a > 0 ∧ b > 0 ∧ c > 0) : 
  let A := real.arctan (real.sqrt 3) in A = real.pi / 3 :=
by
  -- proof content here
  sorry

-- Part II
variable {λ ω : ℝ}
variable {A : ℝ := real.pi / 3}

theorem range_of_f_in_interval (h_max : λ * real.cos (A / 2) ^ 2 - 3 = 2) 
(h_positive₁ : λ > 0) (h_positive₂ : ω > 0) 
(h_period : (2 * real.pi / (3 * ω)) = real.pi) : 
  let f := λ x, λ * real.cos (ω * x + A / 2) ^ 2 - 3 in 
  (∀ x ∈ Icc 0 (real.pi / 2), f x ∈ Icc (-3) ((5 * real.sqrt 3 - 2) / 4)) :=
by 
  -- proof content here 
  sorry

end angle_A_in_triangle_range_of_f_in_interval_l511_511715


namespace sixth_root_of_7528758090625_l511_511961

theorem sixth_root_of_7528758090625 :
  (1 * 50^6 + 6 * 50^5 + 15 * 50^4 + 20 * 50^3 + 15 * 50^2 + 6 * 50 + 1)^(1 / 6) = 51 :=
by
  sorry

end sixth_root_of_7528758090625_l511_511961


namespace five_letter_words_with_vowel_l511_511245

theorem five_letter_words_with_vowel :
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  total_words - words_no_vowel = 6752 :=
by
  let total_words := 6 ^ 5
  let words_no_vowel := 4 ^ 5
  have h1 : total_words = 7776 := by norm_num
  have h2 : words_no_vowel = 1024 := by norm_num
  show total_words - words_no_vowel = 6752
  calc
    total_words - words_no_vowel
    = 7776 - 1024 : by rw [h1, h2]
    ... = 6752 : by norm_num

end five_letter_words_with_vowel_l511_511245


namespace tencent_technological_innovation_basis_tencent_innovative_development_analysis_l511_511934

-- Define the dialectical materialist basis conditions
variable (dialectical_negation essence_innovation development_perspective unity_of_opposites : Prop)

-- Define Tencent's emphasis on technological innovation
variable (tencent_innovation : Prop)

-- Define the relationship between Tencent's development and materialist view of development
variable (unity_of_things_developmental progressiveness_tortuosity quantitative_qualitative_changes : Prop)
variable (tencent_development : Prop)

-- Prove that Tencent's emphasis on technological innovation aligns with dialectical materialism
theorem tencent_technological_innovation_basis :
  dialectical_negation ∧ essence_innovation ∧ development_perspective ∧ unity_of_opposites → tencent_innovation :=
by sorry

-- Prove that Tencent's innovative development aligns with dialectical materialist view of development
theorem tencent_innovative_development_analysis :
  unity_of_things_developmental ∧ progressiveness_tortuosity ∧ quantitative_qualitative_changes → tencent_development :=
by sorry

end tencent_technological_innovation_basis_tencent_innovative_development_analysis_l511_511934


namespace sum_of_differences_l511_511600

theorem sum_of_differences : 
  let first_digit (n : ℕ) : ℕ := n / 10^(nat.log10 n)
  let last_digit (n : ℕ) : ℕ := n % 10
  (finset.range 1000).sum (λ n, first_digit n - last_digit n) = 495 :=
by
  sorry

end sum_of_differences_l511_511600


namespace mean_home_runs_correct_l511_511827

-- Define the total home runs in April
def total_home_runs_April : ℕ := 5 * 4 + 6 * 4 + 8 * 2 + 10

-- Define the total home runs in May
def total_home_runs_May : ℕ := 5 * 2 + 6 * 2 + 8 * 3 + 10 * 2 + 11

-- Define the total number of top hitters/players
def total_players : ℕ := 12

-- Define the total home runs over two months
def total_home_runs : ℕ := total_home_runs_April + total_home_runs_May

-- Calculate the mean number of home runs
def mean_home_runs : ℚ := total_home_runs / total_players

-- Prove that the calculated mean is equal to the expected result
theorem mean_home_runs_correct : mean_home_runs = 12.08 := by
  sorry

end mean_home_runs_correct_l511_511827


namespace f_f_one_third_eq_two_l511_511668

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then (1 / 2) ^ x
  else Real.log x / Real.log 3

theorem f_f_one_third_eq_two : f (f (1 / 3)) = 2 := 
by 
  sorry

end f_f_one_third_eq_two_l511_511668


namespace binom_mod_prime_l511_511882

theorem binom_mod_prime (p m n : ℕ) (hp : p.prime) (hmn : m ≥ n) :
  binom (p * m) (p * n) % p = binom m n % p :=
sorry

end binom_mod_prime_l511_511882


namespace sum_of_differences_l511_511579

/-- 
  For each natural number from 1 to 999, Damir subtracts the last digit from the first digit and 
  writes the resulting differences on a board. We are to prove that the sum of all these differences 
  is 495.
-/
theorem sum_of_differences : 
  (∑ n in Finset.range 1000, (first_digit n - last_digit n)) = 495 :=
sorry

/-- 
  Helper function to get the first digit of a natural number.
  Here, n > 0
-/
def first_digit (n : ℕ) : ℕ :=
  n / 10^(n.digits 10 - 1)

/-- 
  Helper function to get the last digit of a natural number.
  Here, n > 0
-/
def last_digit (n : ℕ) : ℕ :=
  n % 10

end sum_of_differences_l511_511579


namespace largest_number_with_digits_product_120_is_85311_l511_511559

-- Define the five-digit number
def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

-- Define the product of digits
def digits_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

-- Define the condition that the product of the digits should be 120
def product_is_120 (n : ℕ) : Prop :=
  digits_product n = 120

-- Define the condition that n is the largest such number
def largest_such_number (n : ℕ) : Prop :=
  ∀ m : ℕ, is_five_digit m → product_is_120 m → m ≤ n

-- The theorem stating that the largest five-digit number whose digits' product equals 120 is 85311
theorem largest_number_with_digits_product_120_is_85311 : ∃ n : ℕ, is_five_digit n ∧ product_is_120 n ∧ largest_such_number n ∧ n = 85311 :=
by
  use 85311
  split
  -- Prove 85311 is a five-digit number
  sorry
  split
  -- Prove the product of the digits of 85311 is 120
  sorry
  split
  -- Prove that 85311 is the largest such number
  sorry
  -- Prove n = 85311
  sorry

end largest_number_with_digits_product_120_is_85311_l511_511559


namespace probability_coprime_l511_511422

open BigOperators

theorem probability_coprime (A : Finset ℕ) (h : A = {1, 2, 3, 4, 5, 6, 7, 8}) :
  let pairs := { (a, b) ∈ (A ×ˢ A) | a < b }
  let coprime_pairs := pairs.filter (λ p, Nat.gcd p.1 p.2 = 1)
  coprime_pairs.card / pairs.card = 5 / 7 := by 
sorry

end probability_coprime_l511_511422


namespace relatively_prime_probability_l511_511429

open Finset

theorem relatively_prime_probability :
  let s := ({1, 2, 3, 4, 5, 6, 7, 8} : Finset ℕ)
  in let pairs := s.val.powerset.filter (λ t, t.card = 2)
  in (pairs.count (λ t, (t : Multiset ℕ).gcd = 1)).toRational / pairs.card.toRational = 3 / 4 := 
by
  -- Prove that the probability is 3/4
  sorry

end relatively_prime_probability_l511_511429


namespace magnitude_proj_v_on_w_l511_511753

variables (v w : ℝ^n) -- Assuming n-dimensional real vectors
variables (dot_product : v ⬝ w = 7) (norm_w : ‖w‖ = 4)

theorem magnitude_proj_v_on_w : ∥v - ((v ⬝ w)/(‖w‖^2)) • w∥ = 7 := by
  sorry

end magnitude_proj_v_on_w_l511_511753


namespace locus_of_points_M_l511_511117
-- Importing the full Mathlib library to access all necessary definitions and theorems.

-- Defining the given conditions and the statement to be proved.
theorem locus_of_points_M (k : Circle) (P : Point) :
  let M := {m | ∃ K ∈ k, equilateralTriangle P K m} in
  if P = k.center then M = {k} else
    M = {circle_of_rotation k P (60), circle_of_rotation k P (-60)} :=
sorry

end locus_of_points_M_l511_511117


namespace class_average_is_correct_l511_511271

noncomputable def classAverages := [25, 35, 20, 10, 10]
noncomputable def scores := [80, 65, 90, 75, 85]

theorem class_average_is_correct :
  let weighted_average := (25 * 80 + 35 * 65 + 20 * 90 + 10 * 75 + 10 * 85) / 100 in
  weighted_average = 76.75 :=
by
  let weighted_average := (25 * 80 + 35 * 65 + 20 * 90 + 10 * 75 + 10 * 85) / 100;
  sorry

end class_average_is_correct_l511_511271
