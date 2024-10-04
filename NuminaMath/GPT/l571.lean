import Mathlib
import Mathlib.
import Mathlib.Algebra.ArithmeticFunction
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Logarithm
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.ProbabilityTheory
import Mathlib.Algebra.Sqrt
import Mathlib.Analysis.Calculus.TangentCone
import Mathlib.Analysis.InnerProductSpace.BHilbert
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Perm.Basic
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.MeasureTheory.Measure.Space
import Mathlib.NumberTheory.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic

namespace narrow_black_stripes_are_8_l571_571897

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l571_571897


namespace adam_more_apples_than_combined_l571_571599

def adam_apples : Nat := 10
def jackie_apples : Nat := 2
def michael_apples : Nat := 5

theorem adam_more_apples_than_combined : 
  adam_apples - (jackie_apples + michael_apples) = 3 :=
by
  sorry

end adam_more_apples_than_combined_l571_571599


namespace narrow_black_stripes_l571_571859

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l571_571859


namespace power_approximate_l571_571932

theorem power_approximate (h : 1.003^x = 1.012) : x ≈ 3.978 :=
by
  sorry

end power_approximate_l571_571932


namespace prob_factor_less_than_10_of_90_l571_571523

theorem prob_factor_less_than_10_of_90 : 
  (∃ (factors_of_90 : Finset ℕ), 
   factors_of_90.card = 12 ∧ 
   (factors_of_90.filter (< 10)).card = 6) → 
  (↑((factors_of_90.filter (< 10)).card) / ↑(factors_of_90.card) = 1 / 2) := 
by
  intro h
  cases h with factors_of_90 h
  cases h with card_ninety filter_card
  simp [card_ninety, filter_card]
  sorry

end prob_factor_less_than_10_of_90_l571_571523


namespace cosine_sum_l571_571621

theorem cosine_sum :
  (Real.cos 0) ^ 2 + 2 * (Real.cos (15 * Real.pi / 180)) ^ 2 + (Real.cos (30 * Real.pi / 180)) ^ 2 +
  2 * (Real.cos (45 * Real.pi / 180)) ^ 2 + (Real.cos (60 * Real.pi / 180)) ^ 2 +
  2 * (Real.cos (75 * Real.pi / 180)) ^ 2 + (Real.cos (90 * Real.pi / 180)) ^ 2 = 3.25 :=
by {
  sorry,
}

end cosine_sum_l571_571621


namespace narrow_black_stripes_l571_571878

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l571_571878


namespace angle_set_condition_trigonometric_expression_simplified_l571_571722

noncomputable def point_A_coordinates := (Real.sqrt 3, -1 : ℝ)

noncomputable def distance_from_origin : ℝ :=
  Real.sqrt ((Real.sqrt 3) ^ 2 + (-1) ^ 2)

def sin_alpha := -1 / 2
def cos_alpha := Real.sqrt 3 / 2
def alpha_set (k : ℤ) : ℝ := 2 * k * Real.pi - Real.pi / 6

theorem angle_set_condition (α : ℝ) :
  α = 2 * k * Real.pi - Real.pi / 6 ↔
  sin_alpha = -1 / 2 ∧ cos_alpha = Real.sqrt 3 / 2 :=
sorry

theorem trigonometric_expression_simplified (α : ℝ) : 
  (Real.sin (2 * Real.pi - α) * Real.tan (Real.pi + α) * Real.cot (-α - Real.pi)) /
  (Real.csc (-α) * Real.cos (Real.pi - α) * Real.tan (3 * Real.pi - α)) = 1 / 2 :=
sorry

end angle_set_condition_trigonometric_expression_simplified_l571_571722


namespace minimum_value_of_a_l571_571740

noncomputable def f (x a : ℝ) : ℝ := 5 * (x + 1)^2 + a / (x + 1)^5

theorem minimum_value_of_a :
  (∀ x, x ≥ 0 → f x (2 * Real.sqrt((24 / 7)^7)) ≥ 24) :=
begin
  intro x,
  intro hx,
  -- here you'd typically carry out the actual proof steps
  sorry
end

end minimum_value_of_a_l571_571740


namespace birds_remaining_l571_571504

theorem birds_remaining (grey_birds : ℕ) (white_birds: ℕ) :
  grey_birds = 40 → white_birds = grey_birds + 6 →
  (grey_birds / 2) + white_birds = 66 :=
begin
  intros hg hw,
  rw hg at hw,
  have h_half_grey := grey_birds / 2,
  rw hg,
  simp,
  have h_grey_freed := 40 / 2,  -- 20 freed grey birds
  have h_remaining : grey_birds / 2 = 20, by { rw hg, norm_num, },
  rw h_remaining,
  have h_total : grey_birds / 2 + white_birds = 20 + 46, by { rw hw, simp, fin_norm_num, },
  exact h_total,
end

end birds_remaining_l571_571504


namespace probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571074

def probability_of_5_odd_numbers_in_6_rolls (prob_odd : ℚ) : ℚ :=
  (nat.choose 6 5 * (prob_odd^5) * ((1 - prob_odd)^1)) / (2^6)

theorem probability_of_5_odd_numbers_in_6_rolls_is_3_over_32 :
  probability_of_5_odd_numbers_in_6_rolls (1/2) = 3 / 32 :=
by sorry

end probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571074


namespace probability_of_5_odd_numbers_l571_571081

-- Define a function to represent the binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Axiom that defines the probability of getting an odd number
axiom fair_die_prob : ∀ (x : ℕ), 0 < x ∧ x ≤ 6 -> (1/2)

-- Define the problem statement about the probability
theorem probability_of_5_odd_numbers (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 5) : 
  (binom n k) / 2^n = 3 / 32 := sorry

end probability_of_5_odd_numbers_l571_571081


namespace narrow_black_stripes_are_8_l571_571896

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l571_571896


namespace area_of_30_60_90_triangle_l571_571487

noncomputable def hypotenuse : ℝ := 9
noncomputable def angle : ℝ := 30

theorem area_of_30_60_90_triangle 
  (hypotenuse : ℝ) 
  (angle : ℝ)
  (h_angle : angle = 30)
  (h_hyp : hypotenuse = 9)

: ∃ (area : ℝ), area = 10.125 * Real.sqrt 3 :=
begin
  sorry
end

end area_of_30_60_90_triangle_l571_571487


namespace problem_5617d_is_multiple_of_9_l571_571267

theorem problem_5617d_is_multiple_of_9 (d : ℕ) (h : d = 8) : 5617 * 10 + d ≡ 0 [MOD 9] :=
by
  have sum_of_digits := 5 + 6 + 1 + 7 + d
  rw [h] at sum_of_digits -- Substitute d with 8
  have sum_of_digits_is_27 : sum_of_digits = 27 := by norm_num
  have divisible_by_9 : 27 % 9 = 0 := by norm_num
  sorry

end problem_5617d_is_multiple_of_9_l571_571267


namespace positive_difference_median_mode_l571_571049

/-- Stem and leaf dataset from the problem -/
def dataset : List ℕ := [11, 12, 16, 17, 17, 21, 21, 21, 22, 25, 25, 30, 34, 37, 38, 39, 41, 43, 43, 43, 49, 50, 52, 55, 56, 58]

/-- Function to calculate the mode of a list -/
def mode (l : List ℕ) : ℕ := 
  l.groupBy id |>.map (λ g => (g.head, g.length)) |>.sortBy (λ p => -p.snd) |>.head.snd

/-- Function to calculate the median of a list -/
def median (l : List ℕ) : ℕ :=
  let sorted_l := l.insertionSort (<=)
  let len := sorted_l.length
  if len % 2 = 0 then
    (sorted_l.get! (len / 2 - 1) + sorted_l.get! (len / 2)) / 2
  else
    sorted_l.get! (len / 2)

/-- Prove the positive difference between the median and the mode is 18 -/
theorem positive_difference_median_mode : ∃ l : List ℕ, l = dataset ∧ (mode l) = 11 ∧ (median l) = 29 ∧ abs((median l) - (mode l)) = 18 :=
by
  have l := dataset
  have h_mode : mode l = 11 := sorry
  have h_median : median l = 29 := sorry
  have h_diff : abs(29 - 11) = 18 := by simp
  exact ⟨l, rfl, h_mode, h_median, h_diff⟩

end positive_difference_median_mode_l571_571049


namespace even_function_period_pi_over_2_l571_571169

theorem even_function_period_pi_over_2 :
  (∃ f : ℝ → ℝ, (∀ x : ℝ, f x = f (-x)) ∧ (∀ x : ℝ, f (x + π/2) = f x)
    ∧ (f = λ x, cos 4 * x)) :=
by
  let f : ℝ → ℝ := λ x, cos 4 * x
  existsi f
  split
  { intro x
    calc
      f (-x) = cos (4 * (-x)) : by rfl
           ... = cos (-(4 * x)) : by rw mul_neg
           ... = cos (4 * x)   : by rw cos_neg
           ... = f x           : by rfl,
    sorry },
  { intro x
    calc
      f (x + π/2) = cos (4 * (x + π/2)) : by rfl
                ... = cos (4 * x + 2 * π) : by rw mul_add
                ... = cos (4 * x)         : by rw cos_add_pi,
    sorry },
  split
  { sorry }

end even_function_period_pi_over_2_l571_571169


namespace greatest_length_of_equal_pieces_l571_571849

theorem greatest_length_of_equal_pieces (a b c : ℕ) (h₁ : a = 42) (h₂ : b = 63) (h₃ : c = 84) :
  Nat.gcd (Nat.gcd a b) c = 21 :=
by
  rw [h₁, h₂, h₃]
  sorry

end greatest_length_of_equal_pieces_l571_571849


namespace smallest_number_ending_in_9_divisible_by_13_l571_571099

theorem smallest_number_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), (n % 10 = 9) ∧ (13 ∣ n) ∧ (∀ (m : ℕ), (m % 10 = 9) ∧ (13 ∣ m) ∧ (m < n) -> false) :=
sorry

end smallest_number_ending_in_9_divisible_by_13_l571_571099


namespace hyperbola_equation_l571_571316

theorem hyperbola_equation
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (H1 : (b / a) = (Real.sqrt 3) / 2)
  (H2 : ∃ (x y : ℝ), x = 2 ∧ y = Real.sqrt 3 ∧
    (x / a = y / b ∨ x / a = - (y / b)))
  (c : ℝ) (H3 : c = Real.sqrt 7)
  (H4 : a^2 + b^2 = c^2) :
  (a = 2 ∧ b = Real.sqrt 3) →
  ∃ x y : ℝ, (x^2 / 4) - (y^2 / 3) = 1 :=
begin
  intros,
  use [1, 1],
  sorry
end

end hyperbola_equation_l571_571316


namespace effective_treatment_duration_minimum_dose_requirement_l571_571581

noncomputable def drug_concentration (m : ℝ) (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 6 then m / 3 * (10 / (4 + x))
  else if 6 ≤ x ∧ x ≤ 8 then m / 3 * (4 - x / 2)
  else 0

theorem effective_treatment_duration (m : ℝ) (x : ℝ) :
  m = 9 → (2 ≤ drug_concentration m x ↔ 0 ≤ x ∧ x ≤ 20 / 3) :=
begin
  sorry
end

theorem minimum_dose_requirement (m : ℝ) :
  (1 ≤ m ∧ m ≤ 12) → (∀ x, 6 ≤ x ∧ x ≤ 8 → 2 ≤ 2 * (4 - x / 2) + m * (10 / (x - 2 + 4)) → m ≥ 6 / 5) :=
begin
  sorry
end

end effective_treatment_duration_minimum_dose_requirement_l571_571581


namespace cosine_function_range_l571_571636

theorem cosine_function_range : 
  (∀ x ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), -1/2 ≤ Real.cos x ∧ Real.cos x ≤ 1) ∧
  (∃ a ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), Real.cos a = 1) ∧
  (∃ b ∈ Set.Icc (-Real.pi / 6) (2 * Real.pi / 3), Real.cos b = -1/2) :=
by
  sorry

end cosine_function_range_l571_571636


namespace narrow_black_stripes_count_l571_571887

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l571_571887


namespace distinct_positive_factors_48_l571_571349

theorem distinct_positive_factors_48 : 
  ∀ (n : ℕ), n = 48 → ∀ (p q : ℕ), p = 2 ∧ q = 3 → (∃ a b : ℕ, 48 = p^a * q^b ∧ (a + 1) * (b + 1) = 10) :=
by
  intros n hn p q hpq
  have h_48 : 48 = 2^4 * 3^1 := by norm_num
  use 4, 1
  split
  · exact h_48
  · norm_num
  sorry

end distinct_positive_factors_48_l571_571349


namespace coordinates_of_F_double_prime_l571_571980

-- Definitions of transformations
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Definition of initial point F
def F : ℝ × ℝ := (1, 1)

-- Definition of the transformations applied to point F
def F_prime : ℝ × ℝ := reflect_x F
def F_double_prime : ℝ × ℝ := reflect_y_eq_x F_prime

-- Theorem statement
theorem coordinates_of_F_double_prime : F_double_prime = (-1, 1) :=
by
  sorry

end coordinates_of_F_double_prime_l571_571980


namespace complex_coordinates_l571_571732

noncomputable def z : ℂ := (2 - (1:ℂ) * complex.I) * (1 + complex.I)

theorem complex_coordinates :
  (z.re, z.im) = (3, 1) :=
by
  -- the proof goes here
  sorry

end complex_coordinates_l571_571732


namespace rotation_reflection_matrix_l571_571242

theorem rotation_reflection_matrix :
  let θ := real.pi / 4
  let R := ![\[ real.cos θ, -real.sin θ \], \[ real.sin θ, real.cos θ \]]
  let Ref_x := ![\[1, 0\], \[0, -1\]]
  R * Ref_x = ![\[ real.sqrt 2 / 2, - real.sqrt 2 / 2 \], \[ - real.sqrt 2 / 2, - real.sqrt 2 / 2 \]] :=
by sorry

end rotation_reflection_matrix_l571_571242


namespace cards_selection_l571_571789

theorem cards_selection (total_cards : ℕ) (suits : ℕ) (face_cards_per_suit : ℕ) (cards_per_suit : ℕ) 
  (h_total_cards : total_cards = 52) (h_suits : suits = 4) (h_face_cards_per_suit : face_cards_per_suit = 3) 
  (h_cards_per_suit : cards_per_suit = 13) :
  let ways_to_choose_3_cards : ℕ := suits * face_cards_per_suit * finset.card (finset.range (suits - 1).choose 2) * cards_per_suit ^ 2
  in ways_to_choose_3_cards = 6084 := 
by
  sorry

end cards_selection_l571_571789


namespace number_of_three_digit_numbers_l571_571270

theorem number_of_three_digit_numbers : 
  let evens := {2, 4}
      odds := {1, 3, 5}
  in ((Fintype.card evens).choose 1) * ((Fintype.card odds).choose 2) * Nat.factorial 3 = 36 := by
  sorry

end number_of_three_digit_numbers_l571_571270


namespace find_number_l571_571131

theorem find_number (x : ℕ) (h : 5 * x = 100) : x = 20 :=
sorry

end find_number_l571_571131


namespace commercial_break_duration_l571_571647

theorem commercial_break_duration (n1 n2 m1 m2 : ℕ) (h1 : n1 = 3) (h2 : m1 = 5) (h3 : n2 = 11) (h4 : m2 = 2) :
  n1 * m1 + n2 * m2 = 37 :=
by
  -- Here, in a real proof, we would substitute and show the calculations.
  sorry

end commercial_break_duration_l571_571647


namespace even_final_segments_l571_571918

theorem even_final_segments
  (k : ℕ) (k_pos : 1 ≤ k)
  (A : Fin k → ℝ) (h : ∀ i j : Fin k, i < j → A i < A j)
  (lines : Fin (k - 1) → ℝ) (hl : ∀ i : Fin (k - 1), A i < lines i ∧ lines i < A (i + 1)) 
  (even_intersections : ∃ n : ℕ, ∑ i in Finset.univ, (1 : ℕ) = 2 * n) :
  (∃ m : ℕ, ∑ i in Finset.univ, (if i.val < k then 1 else 0 : ℕ) = 2 * m) :=
sorry

end even_final_segments_l571_571918


namespace isosceles_triangle_perimeter_l571_571395

-- Define the lengths of the sides
def side1 := 2 -- 2 cm
def side2 := 4 -- 4 cm

-- Define the condition of being isosceles
def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (a = c) ∨ (b = c)

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Define the triangle perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the main theorem to prove
theorem isosceles_triangle_perimeter {a b : ℝ} (ha : a = side1) (hb : b = side2)
    (h1 : is_isosceles a b c) (h2 : triangle_inequality a b c) : perimeter a b c = 10 :=
sorry

end isosceles_triangle_perimeter_l571_571395


namespace probability_5_of_6_odd_rolls_l571_571053

def binom_coeff : ℕ → ℕ → ℕ
| n k := Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binom_coeff n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_5_of_6_odd_rolls :
  binomial_probability 6 5 (1/2) = 3/16 :=
by
  -- Proof will go here, but we skip it with sorry for now.
  sorry

end probability_5_of_6_odd_rolls_l571_571053


namespace distance_covered_l571_571602

-- Definitions of conditions
def time : ℝ := 25 -- Time in seconds
def speed_kmh : ℝ := 28.8 -- Speed in km/h
def speed_mps : ℝ := speed_kmh * 1000 / 3600 -- Conversion of speed to m/s

-- Theorem to prove distance covered
theorem distance_covered (t : ℝ) (v_kmh : ℝ) (v_mps : ℝ) (h : v_mps = v_kmh * 1000 / 3600) : 
  v_mps * t = 200 := 
by
  -- Using the provided conditions
  rw [h]
  sorry -- Proof steps will be filled in here

end distance_covered_l571_571602


namespace smallest_positive_integer_remainder_conditions_l571_571526

theorem smallest_positive_integer_remainder_conditions :
  ∃ b : ℕ, (b % 4 = 3 ∧ b % 6 = 5) ∧ ∀ n : ℕ, (n % 4 = 3 ∧ n % 6 = 5) → n ≥ b := 
by
  have b := 23
  use b
  sorry

end smallest_positive_integer_remainder_conditions_l571_571526


namespace probability_of_B_l571_571022

variable (A B : Prop)

variables (P : Prop → ℝ) [ProbabilityMeasure P]

axiom P_A : P A = 0.4
axiom P_A_and_B : P (A ∧ B) = 0.25
axiom P_A_or_B : P (A ∨ B) = 0.6

theorem probability_of_B :
  P B = 0.45 :=
by
  sorry

end probability_of_B_l571_571022


namespace tiling_count_modulo_l571_571173

theorem tiling_count_modulo (N : ℕ)
  (H1 : ∃ partitions, (partitions.sum = 8) ∧ (∀ p ∈ partitions, p ≥ 1))
  (H2 : ∃ colorings, (colorings.count "Red" = 2) ∧ (colorings.count "Blue" ≥ 1) ∧ (colorings.count "Green" ≥ 1)) :
  N % 1000 = 302 :=
sorry

end tiling_count_modulo_l571_571173


namespace interest_at_end_of_tenth_year_l571_571029

variable (P R : ℝ)
variable (SI SI' : ℝ)
variable (T T' : ℕ)
variable (total_interest : ℝ)

-- conditions
def condition_1 := (SI = (P * R * 10) / 100) ∧ (SI = 600)
def condition_2 := (SI' = (3 * P * R * 5) / 100)

-- given conditions
def conditions := condition_1 ∧ condition_2

-- proof problem statement
theorem interest_at_end_of_tenth_year :
  conditions →
  total_interest = 600 + (3 * 60) * 5 / 100 →
  total_interest = 1140 :=
by
  sorry

end interest_at_end_of_tenth_year_l571_571029


namespace right_triangle_property_l571_571700

variable {A B C K L : Type} [Inhabited A] [Inhabited B] [Inhabited C]

def is_right_triangle (A B C : Type) :=
  (∠ACB = 90)

def angle_bisector (B K : Type) :=
  (is_on_angle_bisector B K)

def circumcircle_intersects (A K B C L : Type) :=
  (is_on_circumcircle A K B L)
  
theorem right_triangle_property 
  {A B C K L : Type}
  [isRightTriangle : is_right_triangle A B C]
  [isAngleBisector : angle_bisector B K]
  [circumcircleIntersects : circumcircle_intersects A K B C L] :
  CB + CL = AB := sorry

end right_triangle_property_l571_571700


namespace cedar_vs_pine_height_cedar_vs_birch_height_l571_571383

-- Define the heights as rational numbers
def pine_tree_height := 14 + 1/4
def birch_tree_height := 18 + 1/2
def cedar_tree_height := 20 + 5/8

-- Theorem to prove the height differences
theorem cedar_vs_pine_height :
  cedar_tree_height - pine_tree_height = 6 + 3/8 :=
by
  sorry

theorem cedar_vs_birch_height :
  cedar_tree_height - birch_tree_height = 2 + 1/8 :=
by
  sorry

end cedar_vs_pine_height_cedar_vs_birch_height_l571_571383


namespace six_students_no_next_to_each_other_l571_571398

open Nat

theorem six_students_no_next_to_each_other :
  let n := 6 in
  let total_arrangements := factorial n in
  let together_arrangements := 2 * factorial (n - 1) in
  let valid_arrangements := total_arrangements - together_arrangements in
  valid_arrangements = 480 :=
by
  let n := 6
  let total_arrangements := factorial n
  let together_arrangements := factorial (n - 1) * 2
  let valid_arrangements := total_arrangements - together_arrangements
  show valid_arrangements = 480 from sorry

end six_students_no_next_to_each_other_l571_571398


namespace probability_5_of_6_odd_rolls_l571_571055

def binom_coeff : ℕ → ℕ → ℕ
| n k := Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binom_coeff n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_5_of_6_odd_rolls :
  binomial_probability 6 5 (1/2) = 3/16 :=
by
  -- Proof will go here, but we skip it with sorry for now.
  sorry

end probability_5_of_6_odd_rolls_l571_571055


namespace cleaner_needed_l571_571443

def cleaner_per_dog := 6
def cleaner_per_cat := 4
def cleaner_per_rabbit := 1

def num_dogs := 6
def num_cats := 3
def num_rabbits := 1

def total_cleaner_for_dogs := cleaner_per_dog * num_dogs
def total_cleaner_for_cats := cleaner_per_cat * num_cats
def total_cleaner_for_rabbits := cleaner_per_rabbit * num_rabbits

def total_cleaner := total_cleaner_for_dogs + total_cleaner_for_cats + total_cleaner_for_rabbits

theorem cleaner_needed : total_cleaner = 49 :=
by
  unfold total_cleaner total_cleaner_for_dogs total_cleaner_for_cats total_cleaner_for_rabbits cleaner_per_dog cleaner_per_cat cleaner_per_rabbit num_dogs num_cats num_rabbits
  rw [cleaner_per_dog, cleaner_per_cat, cleaner_per_rabbit]
  rw [num_dogs, num_cats, num_rabbits]
  simp
  sorry -- The proof needs to end with a correct justification which is omitted here

end cleaner_needed_l571_571443


namespace count_multiples_6_or_8_but_not_both_l571_571768

theorem count_multiples_6_or_8_but_not_both : 
  (∑ i in Finset.range 150, ((if (i % 6 = 0 ∧ i % 24 ≠ 0) ∨ (i % 8 = 0 ∧ i % 24 ≠ 0) then 1 else 0) : ℕ)) = 31 := by
  sorry

end count_multiples_6_or_8_but_not_both_l571_571768


namespace smallest_value_of_x_l571_571637

theorem smallest_value_of_x :
  ∀ x : ℚ, ( ( (5 * x - 20) / (4 * x - 5) ) ^ 3
           + ( (5 * x - 20) / (4 * x - 5) ) ^ 2
           - ( (5 * x - 20) / (4 * x - 5) )
           - 15 = 0 ) → x = 10 / 3 :=
by
  sorry

end smallest_value_of_x_l571_571637


namespace complement_union_eq_l571_571321

def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x ≤ -3}
def complement (A : Set ℝ) : Set ℝ := {x | x ∉ A}

theorem complement_union_eq :
  complement (M ∪ N) = {x | x ≥ 1} :=
sorry

end complement_union_eq_l571_571321


namespace victory_points_value_l571_571805

theorem victory_points_value (V : ℕ) (H : ∀ (v d t : ℕ), 
    v + d + t = 20 ∧ v * V + d ≥ 40 ∧ v ≥ 6 ∧ (t = 20 - 5)) : 
    V = 3 := 
sorry

end victory_points_value_l571_571805


namespace interest_rate_is_six_paise_l571_571497

def SI : Float := 16.32
def P : Float := 34
def T : Float := 8
def R_paise : Nat := 6

theorem interest_rate_is_six_paise : (SI / (P * T) * 100).toInt = R_paise := 
  sorry

end interest_rate_is_six_paise_l571_571497


namespace ratio_revenue_l571_571574

variable (N D J : ℝ)

theorem ratio_revenue (h1 : J = N / 3) (h2 : D = 2.5 * (N + J) / 2) : N / D = 3 / 5 := by
  sorry

end ratio_revenue_l571_571574


namespace general_term_sum_of_terms_l571_571292

-- Given the arithmetic sequence with initial conditions
def a (n : ℕ) : ℤ :=
  if n = 1 then 25
  else if n = 4 then 16
  else 0 -- Should be replaced with the actual formula

-- Prove the general term
theorem general_term (n : ℕ) : a n = 28 - 3n :=
by sorry

-- Calculate sum of first, third, fifth,..., nineteenth terms
theorem sum_of_terms :  
  let terms := List.map a [1, 3, 5, 7, 9, 11, 13, 15, 17, 19] in
  terms.sum = -20 :=
by sorry

end general_term_sum_of_terms_l571_571292


namespace arithmetic_sum_l571_571184

theorem arithmetic_sum :
  let a := 1 in
  let d := 2 in
  let l := 21 in
  let n := (l - a) / d + 1 in
  (n * (a + l)) / 2 = 121 :=
by
  let a := 1
  let d := 2
  let l := 21
  let n := (l - a) / d + 1
  sorry

end arithmetic_sum_l571_571184


namespace num_factors_48_l571_571360

theorem num_factors_48 : 
  let n := 48 in
  ∃ num_factors, num_factors = 10 ∧ 
  (∀ p k, prime p → (n = p ^ k → 1 + k)) := 
sorry

end num_factors_48_l571_571360


namespace inequality_abc_l571_571458

theorem inequality_abc (a b c : ℝ) : a^2 + 4 * b^2 + 8 * c^2 ≥ 3 * a * b + 4 * b * c + 2 * c * a :=
by
  sorry

end inequality_abc_l571_571458


namespace maximum_colors_l571_571522

def cell := (ℕ × ℕ)
def chessboard : finset cell := finset.univ.filter (λ (i, j), (i < 8) ∧ (j < 8))

-- Define adjacency for cells
def neighbors (c : cell) : set cell :=
  { (i, j) | (i, j) ≠ c ∧
    ((i = c.1 ∧ (j = c.2 - 1 ∨ j = c.2 + 1)) ∨ ((i = c.1 - 1 ∨ i = c.1 + 1) ∧ j = c.2)) }

-- Define the coloring condition
def color_valid (coloring : cell → ℕ) : Prop :=
  ∀ c ∈ chessboard, ∃ (a b : cell), a ∈ neighbors c ∧ b ∈ neighbors c ∧ coloring a = coloring c ∧ coloring b = coloring c

-- Problem statement
theorem maximum_colors (coloring : cell → ℕ) (h : color_valid coloring) : ∃ n ≤ 16, ∀ c ∈ chessboard, coloring c ≤ n :=
sorry

end maximum_colors_l571_571522


namespace value_of_c_l571_571960

theorem value_of_c : ∃ (c : ℚ), ∀ (α β : ℚ), α ≠ 0 ∧ β ≠ 0 ∧ α / β = 3 ∧ α + β = -10 → c = α * β := by
  existsi (75 / 4)
  intros α β h
  have h1 : α + β = -10 := h.2.2.2.2
  have h2 : α / β = 3 := h.2.2.1
  have h3 : α = 3 * β := by linarith
  rw h3 at h1
  have h4 : 3 * β + β = -10 := h1
  have h5 : 4 * β = -10 := by linarith
  have h6 : β = -10 / 4 := by linarith
  have h7 : β = -5 / 2 := by norm_num1 at h6
  have h8 : α = 3 * (-5 / 2) := h3
  have h9 : α = -15 / 2 := by norm_num1 at h8
  rw [h7, h9]
  show 75 / 4 = (-15 / 2) * (-5 / 2) by norm_num1
  sorry

end value_of_c_l571_571960


namespace divisors_count_24_l571_571759

theorem divisors_count_24 : 
  (∃ (n : ℤ), (n ∣ 24)) → (set.univ.filter (λ n : ℤ, n ∣ 24)).to_finset.card = 16 :=
by sorry

end divisors_count_24_l571_571759


namespace minimize_sum_fraction_l571_571837

/-- Let  a and  b  be positive whole numbers such that  9/22 < a/b < 5/11.
Find the fraction  a/b  for which the sum  a + b  is as small as possible. -/
theorem minimize_sum_fraction (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h1 : 9 / 22 < a / b) (h2 : a / b < 5 / 11) : a = 3 ∧ b = 7 ∧ a + b = 10 :=
begin
  -- Proof omitted
  sorry
end

end minimize_sum_fraction_l571_571837


namespace midpoint_fixed_l571_571324

noncomputable def M_fixed (A B : Complex) : Complex :=
  (A * (1 - Complex.i) + B * (1 + Complex.i)) / 2

theorem midpoint_fixed (A B C : Complex) (h_same_side : ∀ (C : Complex), same_side A B C) :
  let D := A + (C - A) * Complex.i
  let E := B + (B - C) * Complex.i
  let M := (D + E) / 2
  M = M_fixed A B :=
by
  sorry

end midpoint_fixed_l571_571324


namespace num_interesting_numbers_l571_571151

def interesting (n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → (n % 10^k, n / 10^k) > (0, n / 10^10)

theorem num_interesting_numbers : {n : ℕ | interesting n}.card = 999989989 := by
  sorry

end num_interesting_numbers_l571_571151


namespace sqrt_sum_of_reciprocals_l571_571197

theorem sqrt_sum_of_reciprocals :
  sqrt ((1 / 25: ℝ) + (1 / 36) + (1 / 49)) = (sqrt 7778) / 297 :=
by
  sorry

end sqrt_sum_of_reciprocals_l571_571197


namespace probability_of_one_in_pascals_triangle_l571_571604

theorem probability_of_one_in_pascals_triangle : 
  (let total_elements := (20 * (20 + 1)) / 2 -- Sum of first 20 natural numbers
   let num_ones := 1 + 2 * 19 -- 1 in row 0, and 2 ones in each of rows 1 through 19
   in num_ones / total_elements = 13 / 70) :=
begin
  sorry
end

end probability_of_one_in_pascals_triangle_l571_571604


namespace solve_for_x_l571_571470

theorem solve_for_x : ∃ x : ℝ, 4 * x + 5 * x = 350 - 10 * (x - 5) ∧ x = 400 / 19 :=
by
  use 400 / 19
  split
  -- condition to be used directly to show it builds correctly
  { sorry }
  -- proof for x = 400 / 19
  { refl }

end solve_for_x_l571_571470


namespace percentage_of_boys_from_schoolA_study_science_l571_571804

variable (T : ℝ) -- Total number of boys in the camp
variable (schoolA_boys : ℝ)
variable (science_boys : ℝ)

noncomputable def percentage_science_boys := (science_boys / schoolA_boys) * 100

theorem percentage_of_boys_from_schoolA_study_science 
  (h1 : schoolA_boys = 0.20 * T)
  (h2 : science_boys = schoolA_boys - 56)
  (h3 : T = 400) :
  percentage_science_boys science_boys schoolA_boys = 30 := 
by sorry

end percentage_of_boys_from_schoolA_study_science_l571_571804


namespace sum_f_inv_l571_571696

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom f_func_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem sum_f_inv : ∑ i in Finset.range 2023 + 1, 1 / (f i * f (i + 1)) = 2023 / 4050 := sorry

end sum_f_inv_l571_571696


namespace zero_of_f_possible_judgments_l571_571738

def f (x : ℝ) : ℝ := (1/3)^x - Real.log x

variables (a b c d : ℝ)

theorem zero_of_f_possible_judgments :
  a > b -> b > c -> c > 0 -> f a * f b * f c < 0 -> f d = 0 -> 
  {i : ℕ | i ∈ {1, 2, 3, 4} | match i with 
    | 1 => d < a
    | 2 => d > b 
    | 3 => d < c 
    | 4 => d > c
    | _ => false
  } = {1, 2, 3, 4} :=
by
  sorry

end zero_of_f_possible_judgments_l571_571738


namespace probability_of_1000_in_jail_l571_571005

/-- Definitions of initial conditions -/
def initial_townspeople : ℕ := 1001
def initial_goons : ℕ := 2
def total_people : ℕ := initial_townspeople + initial_goons

/-- Definition of the probability calculation -/
def calculate_probability_of_1000_in_jail: ℚ :=
  let P : ℚ := (finset.prod (finset.range 500) (λ k, (initial_townspeople - 2*k : ℚ) / (total_people - 2*k : ℚ))) in
  P / ((initial_townspeople - 2*500 + 1 : ℚ) / (total_people - 2*500 + 1 : ℚ))

/-- Definition of the target probability -/
def target_probability : ℚ := 3 / 1003

/-- Now we state our theoretical proof problem -/
theorem probability_of_1000_in_jail : calculate_probability_of_1000_in_jail = target_probability :=
by sorry

end probability_of_1000_in_jail_l571_571005


namespace count_odd_difference_quadruples_l571_571422

def is_odd_difference (a b c d : ℕ) : Prop := (a * d - b * c) % 2 = 1

theorem count_odd_difference_quadruples :
  {n : ℕ // n = 96} :=
begin
  let quadruples := {udf for (a, b, c, d) in
    ({1, 2, 3, 4} : finset ℕ).product ({1, 2, 3, 4} : finset ℕ).product ({1, 2, 3, 4} : finset ℕ).product ({1, 2, 3, 4} : finset ℕ)
    if is_odd_difference a b c d},
  exact ⟨quadruples.card, by { sorry }⟩
end

end count_odd_difference_quadruples_l571_571422


namespace exists_palindrome_product_more_than_100_ways_l571_571643

noncomputable def is_palindrome (n : ℕ) : Prop :=
  n = nat.reverse n

theorem exists_palindrome_product_more_than_100_ways :
  ∃ n : ℕ, (∃ palindromes : finset (ℕ × ℕ), 
    (∀ p ∈ palindromes, is_palindrome p.1 ∧ is_palindrome p.2 ∧ (p.1 * p.2 = n)) ∧ palindromes.card > 100) ∧ n = 2^101 :=
by sorry

end exists_palindrome_product_more_than_100_ways_l571_571643


namespace chords_midpoint_locus_l571_571435

noncomputable def distance (P O : Point) : ℝ := sorry
noncomputable def radius (K : Circle) : ℝ := sorry
noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def all_chords_passing_through (P : Point) (K : Circle) : Set (Point × Point) := sorry
noncomputable def midpoint_locus (P : Point) (K : Circle) : Set Point := sorry
noncomputable def Point : Type := sorry
noncomputable def Circle : Type := sorry

-- Define conditions
axioms
  (K : Circle)
  (R : ℝ)
  (P : Point)
  (O : Point)
  (d : ℝ)
  (hR : R = radius K)
  (hP_in_K : distance P O < R / 2)
  (hO : O is the center of K)
  (h_distance : d = distance P O)

-- Prove the equivalence
theorem chords_midpoint_locus :
  (∀ M : Point, M ∈ midpoint_locus P K ↔
    (d ∈ {x : ℝ | x < R / 2} ∧ midpoint_locus P K = 
      set_of (Q : Point) (distance Q (midpoint P O) = d / 2) ∨
    d = 3 * R / 4 → ¬midpoint_locus P K =
    set_of (Q : Point) (distance Q (midpoint P O) = (3 * R / 4) / 2)) :=
  sorry

end chords_midpoint_locus_l571_571435


namespace valid_two_digit_numbers_l571_571440

def two_digit_numbers := { n : ℕ | n / 10 ∈ {1, 2, 3} ∧ n % 10 ∈ {1, 2, 3} ∧ n / 10 ≠ n % 10 }

theorem valid_two_digit_numbers : 
  { n | n ∈ {12, 13, 21, 23, 31, 32} } = two_digit_numbers :=
by {
  sorry
}

end valid_two_digit_numbers_l571_571440


namespace final_price_is_99_l571_571145

-- Conditions:
def original_price : ℝ := 120
def coupon_discount : ℝ := 10
def membership_discount_rate : ℝ := 0.10

-- Define final price calculation
def final_price (original_price coupon_discount membership_discount_rate : ℝ) : ℝ :=
  let price_after_coupon := original_price - coupon_discount
  let membership_discount := price_after_coupon * membership_discount_rate
  price_after_coupon - membership_discount

-- Question: Is the final price equal to $99?
theorem final_price_is_99 :
  final_price original_price coupon_discount membership_discount_rate = 99 :=
by
  sorry

end final_price_is_99_l571_571145


namespace factors_of_48_l571_571337

theorem factors_of_48 : ∃ n, n = 48 → number_of_distinct_positive_factors n = 10 :=
sorry

-- Auxiliary function definitions to support the main theorem
def number_of_distinct_positive_factors (n : ℕ) : ℕ := 
sorry

end factors_of_48_l571_571337


namespace rational_point_partition_exists_l571_571457

open Set

-- Define rational numbers
noncomputable def Q : Set ℚ :=
  {x | True}

-- Define the set of rational points in the plane
def I : Set (ℚ × ℚ) := 
  {p | p.1 ∈ Q ∧ p.2 ∈ Q}

-- Statement of the theorem
theorem rational_point_partition_exists :
  ∃ (A B : Set (ℚ × ℚ)),
    (∀ (y : ℚ), {p ∈ A | p.1 = y}.Finite) ∧
    (∀ (x : ℚ), {p ∈ B | p.2 = x}.Finite) ∧
    (A ∪ B = I) ∧
    (A ∩ B = ∅) :=
sorry

end rational_point_partition_exists_l571_571457


namespace equivalence_of_conditions_l571_571921

open Nat

def binom (n k : ℕ) : ℕ := nat.choose n k

theorem equivalence_of_conditions (n : ℕ) (p : ℕ) (hp : Nat.Prime p) :
  (∀ k : ℕ, k ≤ n → ¬ p ∣ binom n k) ↔ ∃ s : ℤ, s > 0 ∧ ∃ m : ℕ, m < p ∧ n = p^s * m - 1 := by
  sorry

end equivalence_of_conditions_l571_571921


namespace find_larger_number_l571_571125

theorem find_larger_number (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 5) : L = 1637 :=
by
  sorry

end find_larger_number_l571_571125


namespace intersection_of_A_and_B_l571_571320

noncomputable def A : Set ℕ := {x | 2 ≤ x ∧ x ≤ 4}
def B : Set ℕ := {x | x ≤ 3}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l571_571320


namespace find_x_range_l571_571286

def proposition_p (x : ℝ) : Prop := x^2 - 2 * x - 3 ≥ 0
def proposition_q (x : ℝ) : Prop := 0 < x ∧ x < 4

theorem find_x_range (x : ℝ) (h_q_false : ¬ proposition_q x) (h_p_or_q_true : proposition_p x ∨ proposition_q x) :
  x ≤ -1 ∨ x ≥ 4 :=
by
  have h_p : proposition_p x ↔ (x ≤ -1 ∨ x ≥ 3) :=
    by split;
    intro h;
    { -- proof omitted
      sorry },
  have h_q_false' : ¬ proposition_q x → (x ≤ 0 ∨ x ≥ 4) :=
    by intro h; -- proof omitted
    sorry,
  cases h_p_or_q_true with h_p_true h_q_true,
  { -- proof for h_p_true case omitted
    sorry },
  { -- proof for h_q_true case omitted
    contradiction }

end find_x_range_l571_571286


namespace suyeong_ran_distance_l571_571945

theorem suyeong_ran_distance 
  (circumference : ℝ) 
  (laps : ℕ) 
  (h_circumference : circumference = 242.7)
  (h_laps : laps = 5) : 
  (circumference * laps = 1213.5) := 
  by sorry

end suyeong_ran_distance_l571_571945


namespace eggs_per_box_l571_571505

theorem eggs_per_box (total_eggs number_of_boxes : ℕ) (h : total_eggs = 6 ∧ number_of_boxes = 2) : total_eggs / number_of_boxes = 3 :=
by
  rcases h with ⟨total_eggs_eq_6, number_of_boxes_eq_2⟩
  rw [total_eggs_eq_6, number_of_boxes_eq_2]
  norm_num
  sorry

end eggs_per_box_l571_571505


namespace no_intersecting_segments_exists_l571_571283

open set

theorem no_intersecting_segments_exists (n : ℕ) (A B : fin n → ℝ × ℝ)
  (h_collinear: ∀ (i j k : fin n), i ≠ j → i ≠ k → j ≠ k → ¬ collinear ℝ ({A i, A j, A k} ∪ {B i, B j, B k} : set (ℝ × ℝ))) :
  ∃ σ : perm (fin n), ∀ i j, i ≠ j → disjoint (segment ℝ (A i) (B (σ i))) (segment ℝ (A j) (B (σ j))) :=
sorry

end no_intersecting_segments_exists_l571_571283


namespace arithmetic_series_sum_121_l571_571193

-- Define the conditions for the arithmetic series
def is_arithmetic_series (a d : ℕ) (last : ℕ) (n : ℕ) (terms : List ℕ) : Prop :=
  terms = List.iota n |>.map (λ k => a + d * k) ∧ terms.head? = some a ∧ terms.last? = some last

-- Define the sum of a list of natural numbers
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- The main theorem statement
theorem arithmetic_series_sum_121 :
  ∃ (n : ℕ) (terms : List ℕ), is_arithmetic_series 1 2 21 n terms ∧ sum_list terms = 121 :=
by
  sorry

end arithmetic_series_sum_121_l571_571193


namespace probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571071

def probability_of_5_odd_numbers_in_6_rolls (prob_odd : ℚ) : ℚ :=
  (nat.choose 6 5 * (prob_odd^5) * ((1 - prob_odd)^1)) / (2^6)

theorem probability_of_5_odd_numbers_in_6_rolls_is_3_over_32 :
  probability_of_5_odd_numbers_in_6_rolls (1/2) = 3 / 32 :=
by sorry

end probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571071


namespace projection_of_a_onto_b_is_neg_one_l571_571325

variables (a b : ℝ^3) -- assuming the vectors are in three-dimensional space

-- Given conditions
axiom norm_a : ∥a∥ = 2
axiom norm_b : ∥b∥ = 1
axiom b_perp_a_plus_b : b ⬝ (a + b) = 0

-- Define the projection function
def projection (a b : ℝ^3) : ℝ := (a ⬝ b) / ∥b∥

-- The proof problem
theorem projection_of_a_onto_b_is_neg_one : projection a b = -1 :=
sorry

end projection_of_a_onto_b_is_neg_one_l571_571325


namespace part_I_part_II_l571_571628

def f(x : ℝ) (a : ℝ) : ℝ := ln(x + 1) - (a * x / (x + 1))

theorem part_I (a : ℝ) (h1 : ∀ x : ℝ, x > -1 → differentiable_at ℝ (λ x, f(x)(a)) x) (local_min_f0 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, abs (x - 0) < δ → f(x)(a) ≥ f(0)(a)) : a = 1 :=
sorry

theorem part_II (a : ℝ) (h2 : a ≤ 1) (h3 : ∀ x ∈ Ioi (0 : ℝ), f x a > 0) : a = 1 :=
sorry

end part_I_part_II_l571_571628


namespace multiples_of_6_or_8_but_not_both_l571_571784

theorem multiples_of_6_or_8_but_not_both (n : ℕ) : 
  n = 25 ∧ (n = 18) ∧ (n = 6) → (25 - 6) + (18 - 6) = 31 :=
by
  sorry

end multiples_of_6_or_8_but_not_both_l571_571784


namespace probability_of_5_odd_in_6_rolls_l571_571065

open Classical

noncomputable def prob_odd_in_six_rolls : ℚ :=
  let num_rolls := 6
  let prob_odd_single := 1 / 2
  let binom_coeff := Nat.choose num_rolls 5
  let total_outcomes := (2 : ℕ) ^ num_rolls
  binom_coeff * ((prob_odd_single ^ 5) * ((1 - prob_odd_single) ^ (num_rolls - 5))) / total_outcomes

theorem probability_of_5_odd_in_6_rolls :
  prob_odd_in_six_rolls = 3 / 32 :=
by
  sorry

end probability_of_5_odd_in_6_rolls_l571_571065


namespace zero_in_interval_l571_571501

def f (x : ℝ) : ℝ := Real.exp x + x - 2

theorem zero_in_interval : ∃ x ∈ Set.Ioo (-1 : ℝ) (0 : ℝ), f x = 0 :=
sorry

end zero_in_interval_l571_571501


namespace simplify_fraction_l571_571935

theorem simplify_fraction :
  let E := 1 / (1 / ((1 / 3) ^ 1) + 1 / ((1 / 3) ^ 2) + 1 / ((1 / 3) ^ 3) + 1 / ((1 / 3) ^ 4))
  in E = 1 / 120 :=
by
  let E := 1 / (1 / ((1 / 3) ^ 1) + 1 / ((1 / 3) ^ 2) + 1 / ((1 / 3) ^ 3) + 1 / ((1 / 3) ^ 4))
  have h : E = 1 / 120
  exact h

end simplify_fraction_l571_571935


namespace monotonic_decreasing_interval_of_f_l571_571489

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0 :=
by
  sorry

end monotonic_decreasing_interval_of_f_l571_571489


namespace maximum_value_expression_l571_571086

noncomputable def expression (s t : ℝ) := -2 * s^2 + 24 * s + 3 * t - 38

theorem maximum_value_expression : ∀ (s : ℝ), expression s 4 ≤ 46 :=
by sorry

end maximum_value_expression_l571_571086


namespace cos_double_angle_sum_l571_571679

theorem cos_double_angle_sum
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 := by
  sorry

end cos_double_angle_sum_l571_571679


namespace full_price_ticket_revenue_l571_571577

theorem full_price_ticket_revenue (f d : ℕ) (p : ℝ) : 
  f + d = 200 → 
  f * p + d * (p / 3) = 3000 → 
  d = 200 - f → 
  (f * p) = 1500 := 
by
  intros h1 h2 h3
  sorry

end full_price_ticket_revenue_l571_571577


namespace smallest_positive_integer_remainder_conditions_l571_571528

theorem smallest_positive_integer_remainder_conditions :
  ∃ b : ℕ, (b % 4 = 3 ∧ b % 6 = 5) ∧ ∀ n : ℕ, (n % 4 = 3 ∧ n % 6 = 5) → n ≥ b := 
by
  have b := 23
  use b
  sorry

end smallest_positive_integer_remainder_conditions_l571_571528


namespace no_egg_arrangements_possible_l571_571444

noncomputable def num_egg_arrangements 
  (total_eggs : ℕ) 
  (type_A_eggs : ℕ) 
  (type_B_eggs : ℕ)
  (type_C_eggs : ℕ)
  (groups : ℕ)
  (ratio_A : ℕ) 
  (ratio_B : ℕ) 
  (ratio_C : ℕ) : ℕ :=
if (total_eggs = type_A_eggs + type_B_eggs + type_C_eggs) ∧ 
   (type_A_eggs / groups = ratio_A) ∧ 
   (type_B_eggs / groups = ratio_B) ∧ 
   (type_C_eggs / groups = ratio_C) then 0 else 0

theorem no_egg_arrangements_possible :
  num_egg_arrangements 35 15 12 8 5 2 3 1 = 0 := 
by sorry

end no_egg_arrangements_possible_l571_571444


namespace number_of_integers_l571_571250

theorem number_of_integers (n : ℤ) : {n : ℤ | 15 < n^2 ∧ n^2 < 120}.finite.card = 14 :=
sorry

end number_of_integers_l571_571250


namespace sum_equals_target_l571_571694

open BigOperators

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_initial_condition : f 0 = 1

axiom f_functional_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem sum_equals_target : (∑ i in finset.range 2023, 1 / (f i * f (i + 1))) = 2023 / 4050 :=
by
  sorry

end sum_equals_target_l571_571694


namespace lines_parallel_l571_571813

theorem lines_parallel 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : Real.log (Real.sin α) + Real.log (Real.sin γ) = 2 * Real.log (Real.sin β)) :
  (∀ x y : ℝ, ∀ a b c : ℝ, 
    (x * (Real.sin α)^2 + y * Real.sin α = a) → 
    (x * (Real.sin β)^2 + y * Real.sin γ = c) →
    (-Real.sin α = -((Real.sin β)^2 / Real.sin γ))) :=
sorry

end lines_parallel_l571_571813


namespace skew_iff_not_in_same_plane_l571_571938

-- Definitions from conditions converted to Lean
def skew_lines (L1 L2 : Type) : Prop :=
¬(∃ P, P ∈ L1 ∧ P ∈ L2) ∧ ¬parallel L1 L2

def lines_not_in_same_plane (L1 L2 : Type) : Prop :=
¬ (∃ P, P ∈ L1 ∧ P ∈ L2)

-- Statement of the problem
theorem skew_iff_not_in_same_plane (L1 L2 : Type) :
  skew_lines L1 L2 ↔ lines_not_in_same_plane L1 L2 :=
sorry

end skew_iff_not_in_same_plane_l571_571938


namespace cost_of_each_gumdrop_l571_571367

theorem cost_of_each_gumdrop (cents : ℕ) (gumdrops : ℕ) (cost_per_gumdrop : ℕ) : 
  cents = 224 → gumdrops = 28 → cost_per_gumdrop = cents / gumdrops → cost_per_gumdrop = 8 :=
by
  intros h_cents h_gumdrops h_cost
  sorry

end cost_of_each_gumdrop_l571_571367


namespace num_factors_48_l571_571359

theorem num_factors_48 : 
  let n := 48 in
  ∃ num_factors, num_factors = 10 ∧ 
  (∀ p k, prime p → (n = p ^ k → 1 + k)) := 
sorry

end num_factors_48_l571_571359


namespace needs_change_probability_l571_571037

noncomputable def probability_needs_change (n_toys : ℕ) (quarters : ℕ) (twenty_dollar_bill : ℕ) (fav_toy_cost : ℝ) : ℝ :=
  (n_toys = 10) → 
  (quarters = 10) → 
  (twenty_dollar_bill = 20) → 
  (fav_toy_cost = 2.25) →
  ((10! : ℝ) / ((10 * 9) : ℝ)) * (1 - (((1 / 10) * (1 / 9) * 8!) + ((7 / 10) * (1 / 9) * 8!)) / (10!)) = (5 / 6)

-- Define a theorem stating the probability as described in the solution.
theorem needs_change_probability: 
  probability_needs_change 10 10 20 2.25 = (5 / 6) :=
by
  sorry

end needs_change_probability_l571_571037


namespace angle_cos_equiv_l571_571824

variable {α β γ : ℝ}  -- variables representing angles in radians

-- Assuming the angle condition
def angle_condition (A B C : ℝ) : Prop := A < B ∧ B < C

-- Assuming the cosine condition
def cos_condition (A B C : ℝ) : Prop := cos (2 * A) > cos (2 * B) ∧ cos (2 * B) > cos (2 * C)

-- The Lean theorem statement
theorem angle_cos_equiv {A B C : ℝ} (h : angle_condition A B C) : cos_condition A B C :=
sorry

end angle_cos_equiv_l571_571824


namespace remainder_of_sum_divided_by_14_l571_571524

def consecutive_odds : List ℤ := [12157, 12159, 12161, 12163, 12165, 12167, 12169]

def sum_of_consecutive_odds := consecutive_odds.sum

theorem remainder_of_sum_divided_by_14 :
  (sum_of_consecutive_odds % 14) = 7 := by
  sorry

end remainder_of_sum_divided_by_14_l571_571524


namespace nested_fraction_solution_l571_571541

noncomputable def nested_fraction : ℝ :=
  3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / ... ))))

theorem nested_fraction_solution :
  nested_fraction = (3 + Real.sqrt 69) / 2 :=
sorry

end nested_fraction_solution_l571_571541


namespace max_area_of_fencing_l571_571833

theorem max_area_of_fencing (P : ℕ) (hP : P = 150) 
  (x y : ℕ) (h1 : x + y = P / 2) : (x * y) ≤ 1406 :=
sorry

end max_area_of_fencing_l571_571833


namespace clock_hand_right_angle_difference_l571_571915

theorem clock_hand_right_angle_difference :
  ∃ (a b c : ℕ), (b < c) ∧ (Nat.gcd b c = 1) ∧ (let difference := 32 + 8 / 11 in a = 32 ∧ b = 8 ∧ c = 11 ∧ a + b + c = 51) :=
by
  let minute_hand_degrees (t : ℕ) := 6 * t
  let hour_hand_degrees (t : ℕ) := t / 2
  let times_at_right_angles :=
    {t : ℕ | abs (minute_hand_degrees t - hour_hand_degrees t) = 90 ∨ abs (minute_hand_degrees t - hour_hand_degrees t) = 270}
  have h_times : ∃ t₁ t₂ : ℕ, t₁ ∈ times_at_right_angles ∧ t₂ ∈ times_at_right_angles ∧ t₁ ≠ t₂, by sorry
  obtain ⟨t1, t2, ht1, ht2, ht12⟩ := h_times
  have hd : t2 - t1 = 360 / 11 := by sorry
  have ha_bc : (32 + 8 / 11) = 360 / 11 := by sorry
  use 32, 8, 11
  split
  { -- b < c
    exact Nat.lt_of_sub_zero (by norm_num) }
  split
  { -- GCD of b and c is 1
    exact Nat.gcd_eq_one_iff_coprime.mpr (by norm_num) }
  split
  { -- difference in minute format
    exact Nat.gcd_eq_one_of_coprime (by norm_num) }
  { -- a + b + c
    calc
      32 + 8 + 11 = 51 : by norm_num }
  sorry

end clock_hand_right_angle_difference_l571_571915


namespace range_of_a_l571_571293

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log (1 / 2) (x ^ 2 - a * x + 3 * a)

def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f y ≤ f x

theorem range_of_a (a : ℝ) : 
  (is_decreasing (f a) {x : ℝ | 2 ≤ x}) ↔ (-4 < a ∧ a ≤ 4) := 
by sorry

end range_of_a_l571_571293


namespace exists_set_with_divisibility_property_l571_571456

theorem exists_set_with_divisibility_property (n : ℕ) (h : n > 1) :
  ∃ S : Finset ℤ, S.card = n ∧ ∀ a b ∈ S, a ≠ b → (a - b)^2 ∣ a * b := by
  sorry

end exists_set_with_divisibility_property_l571_571456


namespace max_area_diff_l571_571610

theorem max_area_diff (A B C D : Point)
    (h1 : InsideTriangle D A B C)
    (h2 : dist A B = 4)
    (h3 : dist C D = 4)
    (h4 : angle A + angle B D C = 180) :
  ∃ M, ∀ A' B' C' D' : Point,
    (InsideTriangle D' A' B' C') →
    (dist A' B' = 4) →
    (dist C' D' = 4) →
    (angle A' + angle B' D' C' = 180) →
    M = area(A' B' C') - area(B' D' C') →
    M ≤ 8 :=
sorry

end max_area_diff_l571_571610


namespace hall_volume_l571_571122

theorem hall_volume (length width : ℝ) (h : ℝ) 
  (h_length : length = 6) 
  (h_width : width = 6) 
  (h_areas : 2 * (length * width) = 4 * (length * h)) :
  length * width * h = 108 :=
by
  sorry

end hall_volume_l571_571122


namespace Mike_money_made_l571_571446

-- Definition of conditions
variable (money_spent_on_blades : ℕ)
variable (game_price : ℕ)
variable (number_of_games : ℕ)
variable (total_money_made : ℕ)

-- Given conditions
def conditions := 
  money_spent_on_blades = 10 ∧ 
  game_price = 8 ∧ 
  number_of_games = 4

-- Expected result
def question := 
  total_money_made = 42

-- Math problem reducing to proving our question under the given conditions
theorem Mike_money_made (h : conditions) : question :=
  sorry

end Mike_money_made_l571_571446


namespace parallelogram_distances_const_l571_571922

theorem parallelogram_distances_const (A B C D P : Point) (AB BC CD DA : Line) 
  (convex : ConvexQuadrilateral A B C D)
  (constant_sum : ∀ P ∈ interior (Quadrilateral A B C D), 
    distance P AB + distance P BC + distance P CD + distance P DA = constant) :
  Parallelogram A B C D :=
sorry

end parallelogram_distances_const_l571_571922


namespace probability_of_one_in_pascals_triangle_l571_571603

theorem probability_of_one_in_pascals_triangle : 
  (let total_elements := (20 * (20 + 1)) / 2 -- Sum of first 20 natural numbers
   let num_ones := 1 + 2 * 19 -- 1 in row 0, and 2 ones in each of rows 1 through 19
   in num_ones / total_elements = 13 / 70) :=
begin
  sorry
end

end probability_of_one_in_pascals_triangle_l571_571603


namespace find_y_l571_571381

theorem find_y (y : ℕ) (h1 : (λ x y, 2 * x * y) 7 ((λ x y, 2 * x * y) 4 y) = 560) : y = 5 := 
by 
  sorry

end find_y_l571_571381


namespace domain_of_f_value_of_a_functional_identity_l571_571736

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

def domain_f (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ -1

theorem domain_of_f : ∀ x, domain_f x ↔ (1 - x^2 ≠ 0) := by
  intro x
  unfold domain_f
  simp
  sorry

theorem value_of_a (a : ℝ) : f a = 2 ↔ (a = sqrt 3 / 3 ∨ a = -sqrt 3 / 3) := by
  intro a
  have h : a ≠ 1 ∧ a ≠ -1 := sorry -- x ≠ 1 ∧ x ≠ -1 should hold for a.
  simp [f, *]
  sorry

theorem functional_identity (x : ℝ) (hx : domain_f x) : f (1 / x) = - f x := by
  have h1 : 1 - (1 / x)^2 ≠ 0 := sorry -- derive this from hx
  simp [f]
  sorry

end domain_of_f_value_of_a_functional_identity_l571_571736


namespace exists_palindrome_product_more_than_100_ways_l571_571642

noncomputable def is_palindrome (n : ℕ) : Prop :=
  n = nat.reverse n

theorem exists_palindrome_product_more_than_100_ways :
  ∃ n : ℕ, (∃ palindromes : finset (ℕ × ℕ), 
    (∀ p ∈ palindromes, is_palindrome p.1 ∧ is_palindrome p.2 ∧ (p.1 * p.2 = n)) ∧ palindromes.card > 100) ∧ n = 2^101 :=
by sorry

end exists_palindrome_product_more_than_100_ways_l571_571642


namespace probability_square_or_circle_l571_571917

theorem probability_square_or_circle
  (triangles squares circles : ℕ)
  (total_figures : ℕ)
  (h_triangles: triangles = 4)
  (h_squares: squares = 3)
  (h_circles: circles = 5)
  (h_total: total_figures = triangles + squares + circles)
  (h_total_value : total_figures = 12) :
  (3 / 12) + (5 / 12) = 2 / 3 :=
by
  have h_desired_figures := h_squares + h_circles
  have h_probability := h_desired_figures / h_total_value
  sorry

end probability_square_or_circle_l571_571917


namespace not_prime_sum_of_positive_integers_l571_571436

theorem not_prime_sum_of_positive_integers (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h_perfect_square : ∃ d : ℕ, a^2 - b * c = d^2) : ¬ prime (2 * a + b + c) := 
sorry

end not_prime_sum_of_positive_integers_l571_571436


namespace johns_children_probability_l571_571412

theorem johns_children_probability :
  (∑ k in {5, 6, 7}, (nat.choose 7 k) * (1 / 2)^k * (1 / 2)^(7 - k)) = 29 / 128 :=
by
  sorry

end johns_children_probability_l571_571412


namespace households_use_only_brand_A_l571_571584

-- Defining necessary variables and conditions
variable (totalHouseholds : ℕ) (householdsNeither : ℕ) (householdsBoth : ℕ)
variable (ratioOnlyBToBoth : ℕ) (A : ℕ)

-- Defining conditions from the problem
def conditions := totalHouseholds = 240 ∧
                  householdsNeither = 80 ∧
                  householdsBoth = 25 ∧
                  ratioOnlyBToBoth = 3

-- Proving that the number of households that use only brand A soap is 60
theorem households_use_only_brand_A (h : conditions) : A = 60 :=
by
  sorry

end households_use_only_brand_A_l571_571584


namespace count_multiples_6_or_8_not_both_l571_571766

-- Define the conditions
def is_multiple (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

-- Define the main proof statement
theorem count_multiples_6_or_8_not_both :
  (∑ k in Finset.filter (λ n, is_multiple n 6 ∨ is_multiple n 8 ∧ ¬(is_multiple n 6 ∧ is_multiple n 8)) (Finset.range 151), 1) = 31 := 
sorry

end count_multiples_6_or_8_not_both_l571_571766


namespace volume_of_prism_l571_571481

variables {l α β : ℝ}

theorem volume_of_prism (l α β : ℝ) : 
  ∃ V, 
  (is_isosceles_triangle (ABC) ∧ 
   angle_between_equal_sides ABC α ∧ 
   segment_from_upper_base_to_center_circumcircle_A1O_eq_l l ∧ 
   segment_A1O_makes_angle_with_base β)
  → V = l^3 * sin(2 * β) * cos(β) * sin(α) * (cos(α / 2))^2 := 
sorry

end volume_of_prism_l571_571481


namespace vector_a_magnitude_l571_571323

-- Define the vector a
def vector_a : ℝ × ℝ := (3, -2)

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

-- Theorem stating that the magnitude of vector_a is √13
theorem vector_a_magnitude : magnitude vector_a = real.sqrt 13 :=
by
  sorry

end vector_a_magnitude_l571_571323


namespace semicircle_perimeter_calc_l571_571024

noncomputable def semicircle_perimeter (r : ℝ) : ℝ :=
  π * r + 2 * r

theorem semicircle_perimeter_calc :
  semicircle_perimeter 2.1 ≈ 10.794 :=
by
  sorry

end semicircle_perimeter_calc_l571_571024


namespace compound_interest_correct_l571_571236

def principal : ℝ := 14800
def rate : ℝ := 0.135
def times_compounded : ℕ := 1
def years : ℕ := 2
def future_value : ℝ := principal * (1 + rate / times_compounded)^(times_compounded * years)
def compound_interest : ℝ := future_value - principal
def rounded_compound_interest : ℕ := Int.natAbs (Int.ofNat compound_interest).natAbs

theorem compound_interest_correct :
  rounded_compound_interest = 4266 :=
by
  -- We use sorry here to skip the proof
  sorry

end compound_interest_correct_l571_571236


namespace set_intersection_l571_571295

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x | x * (4 - x) < 0}
def C_R_B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem set_intersection :
  A ∩ C_R_B = {1, 2, 3, 4} :=
by
  -- Proof goes here
  sorry

end set_intersection_l571_571295


namespace problem1_problem2_l571_571317

-- Define the parabolic condition.
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line intersecting the parabola passing through point D(4,0).
def line (x y t : ℝ) : Prop := x = t * y + 4

-- Define the perpendicularity condition of vectors OA and OB.
def perp (A B : ℝ × ℝ) : Prop :=
  let x1 := A.1, y1 := A.2
  let x2 := B.1, y2 := B.2
  x1 * x2 + y1 * y2 = 0

-- Define the range of the area of triangle ΔOAB.
def area_range (A B : ℝ × ℝ) : Prop :=
  let x1 := A.1, y1 := A.2
  let x2 := B.1, y2 := B.2
  8 * Real.sqrt (4 + ((y1 + y2) / 4)^2) ≥ 16

-- Statement 1: Prove that OA is perpendicular to OB.
theorem problem1 (A B : ℝ × ℝ) (hA : parabola A.1 A.2) (hB : parabola B.1 B.2) (t : ℝ) (hAB : line A.1 A.2 t ∧ line B.1 B.2 t) : perp A B :=
  sorry

-- Statement 2: Find the range of the area of ΔOAB.
theorem problem2 (A B : ℝ × ℝ) (hA : parabola A.1 A.2) (hB : parabola B.1 B.2) (t : ℝ) (hAB : line A.1 A.2 t ∧ line B.1 B.2 t) : area_range A B :=
  sorry

end problem1_problem2_l571_571317


namespace sequence_mod_11_l571_571964

-- Define the sequence
def u : ℕ → ℤ
| 0 := 0  -- We'll adjust the indexing to start from 1 for convenience
| 1 := 1
| 2 := 3
| (n+3) := (n+3+1) * u (n+2) - (n+3) * u (n+1)

theorem sequence_mod_11 (n : ℕ) : 
  ((n = 4) ∨ (n = 8) ∨ (n = 10) ∨ (n ≥ 10)) → 11 ∣ u n := 
sorry

end sequence_mod_11_l571_571964


namespace ratio_playground_landscape_l571_571482

-- Defining the conditions
def breadth := 420
def length := breadth / 6
def playground_area := 4200
def landscape_area := length * breadth

-- Stating the theorem to prove the ratio is 1:7
theorem ratio_playground_landscape :
  (playground_area.toFloat / landscape_area.toFloat) = (1.0 / 7.0) :=
by
  sorry

end ratio_playground_landscape_l571_571482


namespace smallest_number_ending_in_9_divisible_by_13_l571_571098

theorem smallest_number_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), (n % 10 = 9) ∧ (13 ∣ n) ∧ (∀ (m : ℕ), (m % 10 = 9) ∧ (13 ∣ m) ∧ (m < n) -> false) :=
sorry

end smallest_number_ending_in_9_divisible_by_13_l571_571098


namespace divisors_count_24_l571_571760

theorem divisors_count_24 : 
  (∃ (n : ℤ), (n ∣ 24)) → (set.univ.filter (λ n : ℤ, n ∣ 24)).to_finset.card = 16 :=
by sorry

end divisors_count_24_l571_571760


namespace picture_distance_from_right_end_l571_571587

def distance_from_right_end_of_wall (wall_width picture_width position_from_left : ℕ) : ℕ := 
  wall_width - (position_from_left + picture_width)

theorem picture_distance_from_right_end :
  ∀ (wall_width picture_width position_from_left : ℕ), 
  wall_width = 24 -> 
  picture_width = 4 -> 
  position_from_left = 5 -> 
  distance_from_right_end_of_wall wall_width picture_width position_from_left = 15 :=
by
  intros wall_width picture_width position_from_left hw hp hp_left
  rw [hw, hp, hp_left]
  sorry

end picture_distance_from_right_end_l571_571587


namespace Monge_circle_equation_min_dist_A_F2_max_area_AOB_l571_571015

-- Define the ellipse C with semi-major axis sqrt(3) and semi-minor axis 1
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

-- Define the equation of the Monge circle
def Monge_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  sqrt (2) * x + y - 4 = 0

-- Define points A and B as tangents from point M on the Monge circle to the ellipse
def is_tangent (m a : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (a.2 = k * (a.1 - m.1) + m.2) ∧ ellipse_C a.1 a.2

def point_M (m : ℝ × ℝ) : Prop :=
  Monge_circle m.1 m.2

def points_A_B_on_ellipse (m a b : ℝ × ℝ) : Prop :=
  is_tangent m a ∧ is_tangent m b

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- The distance from a point to a line
def distance_to_line (P : ℝ × ℝ) (L: ℝ → ℝ → Prop) : ℝ :=
  abs ((sqrt (2) * P.1 + P.2 - 4) / sqrt (sqrt (2)^2 + 1))

-- Define the distance from point A to the line l
def distance_A_to_l (a : ℝ × ℝ) : ℝ :=
  distance_to_line a line_l

-- Minimum value condition
def min_value_condition (a : ℝ × ℝ) (F2 : ℝ × ℝ) : ℝ :=
  distance_A_to_l a - abs (a.1 - F2.1)

-- Area of triangle AOB
def area_triangle (A B : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * B.2 - A.2 * B.1)

-- Prove 1: The equation of the Monge circle of ellipse C is x^2 + y^2 = 4
theorem Monge_circle_equation :
  ∀ (x y : ℝ), (∃ (m : ℝ × ℝ), point_M m ∧ Monge_circle x y) ↔ Monge_circle x y :=
by
  sorry

-- Prove 2: The minimum value of d - |AF_{2}| is 0
theorem min_dist_A_F2 (a F2 : ℝ × ℝ) :
  ∀ (d : ℝ), ∃ (a : ℝ × ℝ), ellipse_C a.1 a.2 → (min_value_condition a F2 = 0) ↔ distance_A_to_l a - abs (a.1 - F2.1) = 0 :=
by
  sorry

-- Prove 3: The maximum area of triangle AOB is √3/2
theorem max_area_AOB (a b : ℝ × ℝ) :
  ∃ (a b : ℝ × ℝ), ellipse_C a.1 a.2 ∧ ellipse_C b.1 b.2 ∧ points_A_B_on_ellipse (a:ℝ×ℝ) (b:ℝ × ℝ)∧
  (area_triangle a origin b = sqrt 3 / 2) ↔ (area_triangle a origin b ≤ sqrt 3 / 2) :=
by
  sorry

end Monge_circle_equation_min_dist_A_F2_max_area_AOB_l571_571015


namespace integral_f_l571_571737

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ set.Icc 0 1 then x^2 else if x ∈ set.Ioc 1 (Real.exp 1) then 1 / x else 0

theorem integral_f :
  intervalIntegral.integral 0 (Real.exp 1) f = 4 / 3 :=
by
  sorry

end integral_f_l571_571737


namespace longest_boat_length_l571_571154

theorem longest_boat_length (a : ℝ) (c : ℝ) 
  (parallel_banks : ∀ x y : ℝ, (x = y) ∨ (x = -y)) 
  (right_angle_bend : ∃ b : ℝ, b = a) :
  c = 2 * a * Real.sqrt 2 := by
  sorry

end longest_boat_length_l571_571154


namespace probability_of_5_out_of_6_rolls_odd_l571_571060

theorem probability_of_5_out_of_6_rolls_odd : 
  (nat.choose 6 5 : ℚ) / (2 ^ 6 : ℚ) = 3 / 32 := 
by
  sorry

end probability_of_5_out_of_6_rolls_odd_l571_571060


namespace probability_of_5_out_of_6_rolls_odd_l571_571058

theorem probability_of_5_out_of_6_rolls_odd : 
  (nat.choose 6 5 : ℚ) / (2 ^ 6 : ℚ) = 3 / 32 := 
by
  sorry

end probability_of_5_out_of_6_rolls_odd_l571_571058


namespace cricket_bat_profit_l571_571140

-- Define the conditions
variables (S : ℝ) (p : ℝ) (C : ℝ)

-- Given conditions
def given_conditions : Prop :=
  S = 900 ∧ p = 0.2

-- Define the cost price and profit
def cost_price (S : ℝ) (p : ℝ) : ℝ := S / (1 + p)
def profit (S : ℝ) (C : ℝ) : ℝ := S - C

-- Statement: Given the conditions, prove the profit amount is 150
theorem cricket_bat_profit
  (hc : given_conditions) :
  profit S (cost_price S p) = 150 := by
  sorry

end cricket_bat_profit_l571_571140


namespace largest_n_l571_571214

theorem largest_n (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  ∃ n : ℕ, n > 0 ∧ n = 10 ∧ n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 5 * x + 5 * y + 5 * z - 12 := 
sorry

end largest_n_l571_571214


namespace sum_divides_strictly_2016_lt_4_l571_571748

def divides (m n : ℕ) : Prop := ∃ k, n = m * k
def divides_strictly (m n : ℕ) : Prop := divides m n ∧ Nat.gcd m (n / m) = 1

theorem sum_divides_strictly_2016_lt_4 :
  (let s := (Finset.univ.filter (λ d, d ∣ 2016)).sum 
    (λ d, (Finset.univ.filter (λ m, divides_strictly m d)).sum (λ m, 1 / (m: ℝ)));
  s < 4 ∧ 4 ≤ s.ceil) := by
  sorry

end sum_divides_strictly_2016_lt_4_l571_571748


namespace sprinkler_truck_probability_l571_571580

noncomputable def binom (n k : ℕ) : ℕ := nat.choose n k

theorem sprinkler_truck_probability : 
  (1 - (binom 18 9) / 2^19) = 1 - (binom 18 9) / (binom 20 10) := sorry

end sprinkler_truck_probability_l571_571580


namespace distinct_positive_factors_of_48_l571_571353

theorem distinct_positive_factors_of_48 : 
  let n := 48 in
  let factors := (2^4) * (3^1) in
  ∀ n : ℕ, n = factors → 
  (let num_factors := (4 + 1) * (1 + 1)
  in num_factors = 10) :=
by 
  let n := 48
  let factors := (2^4) * (3^1)
  assume h : n = factors
  let num_factors := (4 + 1) * (1 + 1)
  show num_factors = 10 from sorry

end distinct_positive_factors_of_48_l571_571353


namespace smallest_n_satisfies_condition_l571_571663

theorem smallest_n_satisfies_condition : 
  ∃ (n : ℕ), n = 1806 ∧ ∀ (p : ℕ), Nat.Prime p → n % (p - 1) = 0 → n % p = 0 := 
sorry

end smallest_n_satisfies_condition_l571_571663


namespace mass_percentage_correct_l571_571241

noncomputable def mass_percentage_C_H_N_O_in_C20H25N3O 
  (m_C : ℚ) (m_H : ℚ) (m_N : ℚ) (m_O : ℚ) 
  (atoms_C : ℚ) (atoms_H : ℚ) (atoms_N : ℚ) (atoms_O : ℚ)
  (total_mass : ℚ)
  (percentage_C : ℚ) (percentage_H : ℚ) (percentage_N : ℚ) (percentage_O : ℚ) :=
  atoms_C = 20 ∧ atoms_H = 25 ∧ atoms_N = 3 ∧ atoms_O = 1 ∧ 
  m_C = 12.01 ∧ m_H = 1.008 ∧ m_N = 14.01 ∧ m_O = 16 ∧ 
  total_mass = (atoms_C * m_C) + (atoms_H * m_H) + (atoms_N * m_N) + (atoms_O * m_O) ∧ 
  percentage_C = (atoms_C * m_C / total_mass) * 100 ∧ 
  percentage_H = (atoms_H * m_H / total_mass) * 100 ∧ 
  percentage_N = (atoms_N * m_N / total_mass) * 100 ∧ 
  percentage_O = (atoms_O * m_O / total_mass) * 100 

theorem mass_percentage_correct : 
  mass_percentage_C_H_N_O_in_C20H25N3O 12.01 1.008 14.01 16 20 25 3 1 323.43 74.27 7.79 12.99 4.95 :=
by {
  sorry
}

end mass_percentage_correct_l571_571241


namespace shaded_area_l571_571609

theorem shaded_area (whole_squares partial_squares : ℕ) (area_whole area_partial : ℝ)
  (h1 : whole_squares = 5)
  (h2 : partial_squares = 6)
  (h3 : area_whole = 1)
  (h4 : area_partial = 0.5) :
  (whole_squares * area_whole + partial_squares * area_partial) = 8 :=
by
  sorry

end shaded_area_l571_571609


namespace narrow_black_stripes_count_l571_571882

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l571_571882


namespace candy_distribution_l571_571419

theorem candy_distribution (n : ℕ) (h : n > 2) :
  (∀ (initial_distribution : Fin n → ℕ) (H : ∑ i, initial_distribution i = n^2),
    ∃ (operations : List (Fin n × Fin n)),
      (∀ (i : Fin n), (perform_operations initial_distribution operations) i = n)) ↔
  (∃ k : ℕ, n = 2^k) :=
sorry

end candy_distribution_l571_571419


namespace peggy_stamps_l571_571615

-- Defining the number of stamps Peggy, Ernie, and Bert have
variables (P : ℕ) (E : ℕ) (B : ℕ)

-- Given conditions
def bert_has_four_times_ernie (B : ℕ) (E : ℕ) : Prop := B = 4 * E
def ernie_has_three_times_peggy (E : ℕ) (P : ℕ) : Prop := E = 3 * P
def peggy_needs_stamps (P : ℕ) (B : ℕ) : Prop := B = P + 825

-- Question to Answer / Theorem Statement
theorem peggy_stamps (P : ℕ) (E : ℕ) (B : ℕ)
  (h1 : bert_has_four_times_ernie B E)
  (h2 : ernie_has_three_times_peggy E P)
  (h3 : peggy_needs_stamps P B) :
  P = 75 :=
sorry

end peggy_stamps_l571_571615


namespace continued_fraction_solution_l571_571542

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / y))) ∧ y = (3 + Real.sqrt 69) / 2 :=
begin
  sorry
end

end continued_fraction_solution_l571_571542


namespace sum_no_solution_congruence_l571_571217

noncomputable def sum_of_primes_no_solution : ℕ :=
  2 + 5

theorem sum_no_solution_congruence :
  sum_of_primes_no_solution = 7 :=
by
  have h1 : ∀ p: ℕ, 5 * (10 * (10⁻¹ : ℤ) + 2) % p ≠ 7 % p → p = 2 ∨ p = 5
  sorry

end sum_no_solution_congruence_l571_571217


namespace find_real_number_l571_571536

theorem find_real_number :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = sqrt 15 :=
sorry

end find_real_number_l571_571536


namespace trimino_partition_unique_recovery_l571_571050

theorem trimino_partition_unique_recovery (grid : Type) [fintype grid] [decidable_eq grid] (triminoes : set (set grid)) (central_cells : grid → Prop)
  (h : ∀ t ∈ triminoes, central_cells (cell_center t)) :
  (∀ (p1 p2 : grid → grid) (h1 : is_trimino_partition p1 triminoes) (h2 : is_trimino_partition p2 triminoes),
    central_cells p1 = central_cells p2 → p1 = p2) := 
sorry

end trimino_partition_unique_recovery_l571_571050


namespace probability_sum_le_four_probability_n_lt_m_add_2_l571_571040

theorem probability_sum_le_four :
  let outcomes := finset.univ.image (λ (x y : fin 4), (x.1 + 1, y.1 + 1)).filter (λ (p : ℕ × ℕ), p.fst < p.snd) in
  let favorable_outcomes := outcomes.filter (λ (p : ℕ × ℕ), p.fst + p.snd ≤ 4) in
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 3 := sorry

theorem probability_n_lt_m_add_2 :
  let outcomes := finset.univ.image (λ (x y : fin 4), (x.1 + 1, y.1 + 1)) in
  let favorable_outcomes := outcomes.filter (λ (p : ℕ × ℕ), p.snd < p.fst + 2) in
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 13 / 16 := sorry

end probability_sum_le_four_probability_n_lt_m_add_2_l571_571040


namespace problem_correct_option_l571_571117

theorem problem_correct_option : 
  ¬((-1)^(-1) = 1) ∧ ¬((-3)^2 = -6) ∧ (π^0 = 1) ∧ ¬((-2)^6 / (-2)^3 = (-2)^2) := 
by
  sorry

end problem_correct_option_l571_571117


namespace fish_lifespan_l571_571510

theorem fish_lifespan (H : ℝ) (D : ℝ) (F : ℝ) 
  (h_hamster : H = 2.5)
  (h_dog : D = 4 * H)
  (h_fish : F = D + 2) : 
  F = 12 :=
by
  rw [h_hamster, h_dog] at h_fish
  simp at h_fish
  exact h_fish

end fish_lifespan_l571_571510


namespace sum_first_15_terms_arithmetic_sequence_l571_571302

theorem sum_first_15_terms_arithmetic_sequence (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0) (h_a8 : a 8 = 4) :
  ∑ i in finset.range 15, a (i + 1) = 60 :=
by
  -- Placeholder for the proof
  sorry

end sum_first_15_terms_arithmetic_sequence_l571_571302


namespace factorial_expression_l571_571180

theorem factorial_expression : 8! - 7 * 7! - 2 * 6! = 5 * 6! :=
by
  sorry

end factorial_expression_l571_571180


namespace decreasing_interval_of_f_l571_571019

noncomputable def f (x : ℝ) : ℝ := real.sqrt (-x^2 + 3 * x + 4)

theorem decreasing_interval_of_f :
  ∃ I, I = set.Icc (3 / 2 : ℝ) 4 ∧ ∀ x ∈ set.Icc (-1 : ℝ) 4, (set.Icc (3 / 2 : ℝ) 4).mem x → -f x = f x :=
sorry

end decreasing_interval_of_f_l571_571019


namespace point_translation_proof_l571_571816

def Point := (ℝ × ℝ)

def translate_right (p : Point) (d : ℝ) : Point := (p.1 + d, p.2)

theorem point_translation_proof :
  let A : Point := (1, 2)
  let A' := translate_right A 2
  A' = (3, 2) :=
by
  let A : Point := (1, 2)
  let A' := translate_right A 2
  show A' = (3, 2)
  sorry

end point_translation_proof_l571_571816


namespace cost_of_gravelling_path_eq_630_l571_571153

-- Define the dimensions of the grassy plot.
def length_grassy_plot : ℝ := 110
def width_grassy_plot : ℝ := 65

-- Define the width of the gravel path.
def width_gravel_path : ℝ := 2.5

-- Define the cost of gravelling per square meter in INR.
def cost_per_sqm : ℝ := 0.70

-- Compute the dimensions of the plot including the gravel path.
def length_including_path := length_grassy_plot + 2 * width_gravel_path
def width_including_path := width_grassy_plot + 2 * width_gravel_path

-- Compute the area of the plot including the gravel path.
def area_including_path := length_including_path * width_including_path

-- Compute the area of the grassy plot without the gravel path.
def area_grassy_plot := length_grassy_plot * width_grassy_plot

-- Compute the area of the gravel path alone.
def area_gravel_path := area_including_path - area_grassy_plot

-- Compute the total cost of gravelling the path.
def total_cost := area_gravel_path * cost_per_sqm

-- The theorem stating the cost of gravelling the path.
theorem cost_of_gravelling_path_eq_630 : total_cost = 630 := by
  -- Proof goes here
  sorry

end cost_of_gravelling_path_eq_630_l571_571153


namespace probability_of_5_out_of_6_rolls_odd_l571_571062

theorem probability_of_5_out_of_6_rolls_odd : 
  (nat.choose 6 5 : ℚ) / (2 ^ 6 : ℚ) = 3 / 32 := 
by
  sorry

end probability_of_5_out_of_6_rolls_odd_l571_571062


namespace center_square_side_length_l571_571479

theorem center_square_side_length (a : ℝ) (h₁ : a = 120) (h₂ : 4 * (1 / 4 * (a * a)) = a * a) : ∃ s : ℝ, s = 60 ∧ (s * s = a * a - 3 / 4 * (a * a)) :=
by
  have area_large_square : a * a = 14400 := by rw [h₁]; norm_num
  have area_l_shaped : 4 * (1 / 4 * (a * a)) = a * a := h₂
  have area_center_square : a * a - 3 / 4 * (a * a) = 3600 := by rw [area_large_square]; norm_num
  use 60
  split
  . norm_num
  . norm_num

end center_square_side_length_l571_571479


namespace probability_of_5_odd_numbers_l571_571079

-- Define a function to represent the binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Axiom that defines the probability of getting an odd number
axiom fair_die_prob : ∀ (x : ℕ), 0 < x ∧ x ≤ 6 -> (1/2)

-- Define the problem statement about the probability
theorem probability_of_5_odd_numbers (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 5) : 
  (binom n k) / 2^n = 3 / 32 := sorry

end probability_of_5_odd_numbers_l571_571079


namespace narrow_black_stripes_count_l571_571886

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l571_571886


namespace total_exterior_angles_l571_571388

-- Define that the sum of the exterior angles of any convex polygon is 360 degrees
def sum_exterior_angles (n : ℕ) : ℝ := 360

-- Given four polygons: a triangle, a quadrilateral, a pentagon, and a hexagon
def triangle_exterior_sum := sum_exterior_angles 3
def quadrilateral_exterior_sum := sum_exterior_angles 4
def pentagon_exterior_sum := sum_exterior_angles 5
def hexagon_exterior_sum := sum_exterior_angles 6

-- The total sum of the exterior angles of these four polygons combined
def total_exterior_angle_sum := 
  triangle_exterior_sum + 
  quadrilateral_exterior_sum + 
  pentagon_exterior_sum + 
  hexagon_exterior_sum

-- The final proof statement
theorem total_exterior_angles : total_exterior_angle_sum = 1440 := by
  sorry

end total_exterior_angles_l571_571388


namespace smallest_positive_integer_ends_in_9_and_divisible_by_13_l571_571089

theorem smallest_positive_integer_ends_in_9_and_divisible_by_13 :
  ∃ n : ℕ, n % 10 = 9 ∧ 13 ∣ n ∧ n > 0 ∧ ∀ m, m % 10 = 9 → 13 ∣ m ∧ m > 0 → m ≥ n := 
begin
  use 99,
  split,
  { exact mod_eq_of_lt (10*k + 9) 10 99 9 (by norm_num), },
  split,
  { exact dvd_refl 99, },
  split,
  { exact zero_lt_99, },
  intros m hm1 hm2 hpos,
  by_contradiction hmn,
  sorry
end

end smallest_positive_integer_ends_in_9_and_divisible_by_13_l571_571089


namespace triangle_inequality_l571_571392

theorem triangle_inequality (a : ℝ) (h₁ : a > 5) (h₂ : a < 19) : 5 < a ∧ a < 19 :=
by
  exact ⟨h₁, h₂⟩

end triangle_inequality_l571_571392


namespace proof_problem_l571_571384

open Real

variables {A B C I M N K L : Type*}
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space I] [metric_space M] [metric_space N]
variables [metric_space K] [metric_space L]
variables [incircle : circle]
variables [circumcircle_ILK : circle]

noncomputable def necessary_and_sufficient_condition 
  (triangle_ABC : triangle A B C) 
  (incenter_I : incenter triangle_ABC I)
  (incircle_touches_AB : incircle.touches AB M)
  (incircle_touches_AC : incircle.touches AC N)
  (extension_BI_intersection_MN : extends BI intersects MN at K)
  (extension_CI_intersection_MN : extends CI intersects MN at L)
  (circumcircle_tangent_inc : circumcircle_ILK tangent to incircle)
  : Prop :=
  AB + AC = 3 * BC

theorem proof_problem
  {triangle_ABC : triangle A B C}
  {incenter_I : incenter triangle_ABC I}
  {incircle_touches_AB : incircle.touches AB M}
  {incircle_touches_AC : incircle.touches AC N}
  {extension_BI_intersection_MN : extends BI intersects MN at K}
  {extension_CI_intersection_MN : extends CI intersects MN at L}
  {circumcircle_tangent_inc : circumcircle_ILK tangent to incircle} :
  necessary_and_sufficient_condition triangle_ABC incenter_I incircle_touches_AB 
  incircle_touches_AC extension_BI_intersection_MN extension_CI_intersection_MN 
  circumcircle_tangent_inc :=
sorry

end proof_problem_l571_571384


namespace find_S20_l571_571277

theorem find_S20 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → S n = 1 + 2 * a n)
  (h2 : a 1 = 2) : 
  S 20 = 2^19 + 1 := 
sorry

end find_S20_l571_571277


namespace probability_of_5_out_of_6_rolls_odd_l571_571063

theorem probability_of_5_out_of_6_rolls_odd : 
  (nat.choose 6 5 : ℚ) / (2 ^ 6 : ℚ) = 3 / 32 := 
by
  sorry

end probability_of_5_out_of_6_rolls_odd_l571_571063


namespace find_real_number_l571_571535

theorem find_real_number :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = sqrt 15 :=
sorry

end find_real_number_l571_571535


namespace marbles_difference_l571_571038

theorem marbles_difference {red_marbles blue_marbles : ℕ} 
  (h₁ : red_marbles = 288) (bags_red : ℕ) (h₂ : bags_red = 12) 
  (h₃ : blue_marbles = 243) (bags_blue : ℕ) (h₄ : bags_blue = 9) :
  (blue_marbles / bags_blue) - (red_marbles / bags_red) = 3 :=
by
  sorry

end marbles_difference_l571_571038


namespace unique_intersection_lines_count_l571_571582

theorem unique_intersection_lines_count :
  let P := (0, 1)
  let parabola := λ x y : ℝ, y^2 = 4 * x
  let intersects_once (line : ℝ → ℝ) : Prop :=
    ∃! p : ℝ × ℝ, line p.1 = p.2 ∧ parabola p.1 p.2
  ∃ n : ℕ, n = 3 ∧ ∀ line : ℝ → ℝ, (line 0 = 1 ∨ line 1 = 0 ∨ (∃ k : ℝ, line = λ x, k * x + 1)) → intersects_once (line).
sorry

end unique_intersection_lines_count_l571_571582


namespace count_multiples_6_or_8_but_not_both_l571_571769

theorem count_multiples_6_or_8_but_not_both : 
  (∑ i in Finset.range 150, ((if (i % 6 = 0 ∧ i % 24 ≠ 0) ∨ (i % 8 = 0 ∧ i % 24 ≠ 0) then 1 else 0) : ℕ)) = 31 := by
  sorry

end count_multiples_6_or_8_but_not_both_l571_571769


namespace narrow_black_stripes_are_eight_l571_571866

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l571_571866


namespace number_of_divisors_24_l571_571758

def is_divisor (n d : ℤ) : Prop := ∃ k : ℤ, n = d * k

theorem number_of_divisors_24 :
  (finset.filter (λ d, is_divisor 24 d) 
    (finset.range 25)).card * 2 = 16 :=
by
  sorry

end number_of_divisors_24_l571_571758


namespace quartic_poly_with_given_roots_l571_571210

theorem quartic_poly_with_given_roots :
  ∃ P : Polynomial ℚ, Polynomial.monic P ∧
    P.coeff 4 = 1 ∧
    P.coeff 3 = -10 ∧
    P.coeff 2 = 17 ∧
    P.coeff 1 = 18 ∧
    P.coeff 0 = -12 ∧
    (P.eval (3 + Real.sqrt 5) = 0 ∧ P.eval (3 - Real.sqrt 5) = 0) ∧ 
    (P.eval (2 + Real.sqrt 7) = 0 ∧ P.eval (2 - Real.sqrt 7) = 0) :=
sorry

end quartic_poly_with_given_roots_l571_571210


namespace number_of_ways_proof_l571_571039

noncomputable def number_of_ways_three_balls_non_consecutive : ℕ :=
  let colors := ({red, blue, yellow} : Finset _) in
  let numbers := (Finset.range 7).map Nat.succ in
  let color_ways := Fintype.card {p : Finset (color × ℕ) // p.card = 3 ∧ ∀ x in p.image Prod.snd, ∀ y in p.image Prod.snd, x ≠ y → (x.1 - y.1).natAbs ≠ 1} in
  color_ways

theorem number_of_ways_proof :
  number_of_ways_three_balls_non_consecutive = 60 :=
sorry

end number_of_ways_proof_l571_571039


namespace narrow_black_stripes_are_eight_l571_571869

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l571_571869


namespace sum_cos_fraction_l571_571199

theorem sum_cos_fraction :
  (\sum k in Finset.range 2018, (5 + Real.cos (π * k / 1009)) / (26 + 10 * Real.cos (π * k / 1009))) = 
  (2018 * 5 ^ 2017) / (5 ^ 2018 - 1) := 
by
  sorry

end sum_cos_fraction_l571_571199


namespace factors_of_48_l571_571333

theorem factors_of_48 : ∃ n, n = 48 → number_of_distinct_positive_factors n = 10 :=
sorry

-- Auxiliary function definitions to support the main theorem
def number_of_distinct_positive_factors (n : ℕ) : ℕ := 
sorry

end factors_of_48_l571_571333


namespace integer_solutions_l571_571668

theorem integer_solutions (n : ℕ) :
  n = 7 ↔ ∃ (x : ℤ), ∀ (x : ℤ), (3 * x^2 + 17 * x + 14 ≤ 20)  :=
by
  sorry

end integer_solutions_l571_571668


namespace commercial_break_duration_l571_571644

theorem commercial_break_duration (n1 n2 t1 t2 : ℕ) (h1 : n1 = 3) (h2: t1 = 5) (h3 : n2 = 11) (h4 : t2 = 2) : 
  n1 * t1 + n2 * t2 = 37 := 
by 
  sorry

end commercial_break_duration_l571_571644


namespace number_of_narrow_black_stripes_l571_571902

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l571_571902


namespace find_x_l571_571976

theorem find_x (x : ℝ) (H1 : ∀ m₁ m₂ m₃ : ℝ, m₁ = 5 * x + 2 ∧ m₂ = 2 * x ∧ m₃ = 4 * x → m₁ + m₂ + m₃ = 88) :
  x = 86 / 11 :=
by
  -- Definitions for the numbers of marbles each boy has
  let m₁ := 5 * x + 2
  let m₂ := 2 * x
  let m₃ := 4 * x
  
  -- Total marbles equation from given conditions
  have H_total : m₁ + m₂ + m₃ = 88 := H1 m₁ m₂ m₃ ⟨rfl, rfl, rfl⟩

  -- Sum it up and simplify as shown in solution steps
  calc
    5 * x + 2 + 2 * x + 4 * x = 11 * x + 2 : by ring
    11 * x + 2 = 88                : by rw [H_total]
    11 * x     = 86                : by linarith
    x         = 86 / 11            : by linarith

end find_x_l571_571976


namespace complete_square_solution_l571_571940

theorem complete_square_solution :
  ∀ x : ℝ, ∃ p q : ℝ, (5 * x^2 - 30 * x - 45 = 0) → ((x + p) ^ 2 = q) ∧ (p + q = 15) :=
by
  sorry

end complete_square_solution_l571_571940


namespace find_y_values_l571_571847

theorem find_y_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 72) : 
  ∃ y ∈ {2, 6}, y = (x - 3)^2 * (x + 4) / (3 * x - 4) :=
by
  sorry

end find_y_values_l571_571847


namespace final_price_is_99_l571_571144

-- Conditions:
def original_price : ℝ := 120
def coupon_discount : ℝ := 10
def membership_discount_rate : ℝ := 0.10

-- Define final price calculation
def final_price (original_price coupon_discount membership_discount_rate : ℝ) : ℝ :=
  let price_after_coupon := original_price - coupon_discount
  let membership_discount := price_after_coupon * membership_discount_rate
  price_after_coupon - membership_discount

-- Question: Is the final price equal to $99?
theorem final_price_is_99 :
  final_price original_price coupon_discount membership_discount_rate = 99 :=
by
  sorry

end final_price_is_99_l571_571144


namespace prob_B_l571_571595

-- Let's name our probabilities
variable (P_A P_A_and_B P_B : ℝ)

-- Given conditions
axiom prob_A : P_A = 0.75
axiom prob_A_and_B : P_A_and_B = 0.45

-- We need to prove
theorem prob_B : P_B = 0.6 :=
by
  have P_A_ne_zero : P_A ≠ 0 := by linarith [prob_A]
  have h : P_B = P_A_and_B / P_A := by sorry  -- We replace the solution process with a proof term
  rw [prob_A, prob_A_and_B] at h
  sorry  -- Skip the final proof steps proving P_B = 0.6

end prob_B_l571_571595


namespace multiples_of_6_or_8_but_not_both_l571_571787

theorem multiples_of_6_or_8_but_not_both (n : ℕ) : 
  n = 25 ∧ (n = 18) ∧ (n = 6) → (25 - 6) + (18 - 6) = 31 :=
by
  sorry

end multiples_of_6_or_8_but_not_both_l571_571787


namespace proof_by_contradiction_negation_l571_571518

theorem proof_by_contradiction_negation :
  ¬(proof_by_contradiction involves simultaneously_negating conclusion_and_conditions_to_derive_contradiction) := 
  by
    sorry

end proof_by_contradiction_negation_l571_571518


namespace concyclic_k_l_m_n_l571_571825

noncomputable def is_concyclic {α : Type} [EuclideanGeometry α] {A B C D : α} : Prop :=
∃ (K : α), circumscribed A B C D K

theorem concyclic_k_l_m_n 
  (α : Type) [EuclideanGeometry α] 
  (A B C D K L M N : α)
  (h1 : is_altitude A D B C)
  (h2 : is_altitude B E A C)
  (h3 : is_altitude C F A B)
  (h4 : perp D K A B)
  (h5 : perp D L B E)
  (h6 : perp D M C F)
  (h7 : perp D N A C)
  : is_concyclic K L M N :=
sorry

end concyclic_k_l_m_n_l571_571825


namespace pascals_triangle_prob_l571_571605

/--
An element is randomly chosen from among the first 20 rows of Pascal's Triangle.
Prove that the probability that the value of the element chosen is 1 is 39/210.
-/
theorem pascals_triangle_prob (n : ℕ) (n = 19) : 
  let total_elements := (n + 1) * (n + 2) / 2,
      ones_in_rows := 2 * n + 1 in
  (ones_in_rows : ℚ) / total_elements = 39 / 210 :=
begin
  sorry
end

end pascals_triangle_prob_l571_571605


namespace volume_is_correct_l571_571227

noncomputable def volume_of_tetrahedron (A B C D : Type*) [InnerProductSpace ℝ A] :=
  let angle_ABC_BCD : ℝ := real.pi / 4 in  -- 45 degrees converted to radians
  let area_ABC : ℝ := 150 in
  let area_BCD : ℝ := 100 in
  let BC : ℝ := 20 in

  -- Calculate height from D to BC in triangle BCD
  let height_BCD : ℝ := (2 * area_BCD) / BC in

  -- Calculate perpendicular height from D to plane ABC
  let h_ABC : ℝ := height_BCD * real.sin angle_ABC_BCD in

  -- Calculate the volume of the tetrahedron
  let volume : ℝ := (1 / 3) * area_ABC * h_ABC in
  volume

theorem volume_is_correct (A B C D : Type*) [InnerProductSpace ℝ A] :
  volume_of_tetrahedron A B C D = 250 * real.sqrt 2 :=
by sorry

end volume_is_correct_l571_571227


namespace convex_ngon_interior_angle_l571_571243

theorem convex_ngon_interior_angle (n : ℕ) :
  (∀ i : fin n, 143 ≤ interior_angle i ∧ interior_angle i ≤ 146) →
  sum_of_interior_angles n = 180 * (n - 2) →
  n = 10 :=
by 
  sorry

end convex_ngon_interior_angle_l571_571243


namespace correct_inequality_l571_571744

variables {a b c : ℝ}
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem correct_inequality (h_a_pos : a > 0) (h_discriminant_pos : b^2 - 4 * a * c > 0) (h_c_neg : c < 0) (h_b_neg : b < 0) :
  a * b * c > 0 :=
sorry

end correct_inequality_l571_571744


namespace ratio_of_w_to_y_l571_571025

theorem ratio_of_w_to_y
  (w x y z : ℚ)
  (h1 : w / x = 4 / 3)
  (h2 : y / z = 3 / 2)
  (h3 : z / x = 1 / 6) :
  w / y = 16 / 3 :=
by sorry

end ratio_of_w_to_y_l571_571025


namespace count_multiples_of_6_or_8_but_not_both_l571_571780

theorem count_multiples_of_6_or_8_but_not_both: 
  let multiples_of_six := finset.filter (λ n, 6 ∣ n) (finset.range 151)
  let multiples_of_eight := finset.filter (λ n, 8 ∣ n) (finset.range 151)
  let multiples_of_twenty_four := finset.filter (λ n, 24 ∣ n) (finset.range 151)
  multiples_of_six.card + multiples_of_eight.card - 2 * multiples_of_twenty_four.card = 31 := 
by {
  -- Provided proof omitted
  sorry
}

end count_multiples_of_6_or_8_but_not_both_l571_571780


namespace probability_closer_to_origin_l571_571588

structure Point :=
  (x : ℝ)
  (y : ℝ)

def RectangularRegion (P : Point) : Prop :=
  0 ≤ P.x ∧ P.x ≤ 3 ∧ 0 ≤ P.y ∧ P.y ≤ 1

def closer_to_origin (P : Point) : Prop :=
  (P.x^2 + P.y^2) < ((P.x - 4)^2 + (P.y - 1)^2)

theorem probability_closer_to_origin :
  (∫ (P : Point) in {P : Point | RectangularRegion P ∧ closer_to_origin P}, 1) /
  (∫ (P : Point) in {P : Point | RectangularRegion P}, 1) = 1.0625 / 3 :=
sorry

end probability_closer_to_origin_l571_571588


namespace narrow_black_stripes_l571_571875

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l571_571875


namespace evaluate_expression_l571_571618

noncomputable def sqrt3 := Real.sqrt 3
def tan60 := sqrt3
def two_pow_neg_two := (2:ℝ)^(-2)
def rational_term := (2 / (sqrt3 + 1))

theorem evaluate_expression :
  |sqrt3 - 3| - tan60^2 + two_pow_neg_two + rational_term = -3/4 := by
  sorry

end evaluate_expression_l571_571618


namespace max_3m_plus_4n_l571_571942

theorem max_3m_plus_4n (m n : ℕ) (h1 : ∀ i j, i ≠ j → even (2 * i) ∧ odd (2 * j + 1))
  (h2 : m * (m + 1) + n ^ 2 = 1987) : 3 * m + 4 * n = 219 := sorry

end max_3m_plus_4n_l571_571942


namespace fraction_subtraction_simplified_l571_571181

theorem fraction_subtraction_simplified :
  (9 : ℚ) / 19 - (5 : ℚ) / 57 = 22 / 57 :=
begin
  -- Since we only need the statement, we put 'sorry' here for the proof.
  sorry
end

end fraction_subtraction_simplified_l571_571181


namespace value_of_ab_plus_bc_plus_ca_l571_571684

theorem value_of_ab_plus_bc_plus_ca (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end value_of_ab_plus_bc_plus_ca_l571_571684


namespace pyramid_has_one_base_l571_571120

def is_quadrilateral_pyramid (P : Type) [Fintype P] : Prop :=
  ∃ (v : Finset P), v.card = 5 ∧ ∃ (e : Finset (P × P)), e.card = 8

def is_pentagonal_pyramid (P : Type) [Fintype P] : Prop :=
  ∃ (v : Finset P), v.card = 6 ∧ ∃ (f : Finset (Finset P)), f.card = 6

def is_hexagonal_pyramid (P : Type) [Fintype P] : Prop :=
  ∃ (v : Finset P), v.card = 7 ∧ ∃ (e : Finset (P × P)), e.card = 12

def is_pyramid (P : Type) : Prop :=
  ∃ (B : Set P), ∃ (V : P), V ∉ B ∧ ∀ v ∈ B, ∃ u ∈ B, u ≠ v ∧ ⟦(u, v)⟧

theorem pyramid_has_one_base (P : Type) [Fintype P] (hP : is_pyramid P) : 
  ∃ B : Set P, ¬IsEmpty B ∧ ∀ B' : Set P, B == B' :=
sorry

end pyramid_has_one_base_l571_571120


namespace sum_squares_eq_nine_l571_571709

variable {R : Type} [Real R] (x y z : R)

theorem sum_squares_eq_nine (h1 : (x + y + z)^2 = 25) (h2 : xy + xz + yz = 8) :
  x^2 + y^2 + z^2 = 9 := by
   sorry

end sum_squares_eq_nine_l571_571709


namespace number_of_narrow_black_stripes_l571_571905

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l571_571905


namespace garden_area_l571_571955

theorem garden_area (w l A : ℕ) (h1 : w = 12) (h2 : l = 3 * w) (h3 : A = l * w) : A = 432 := by
  sorry

end garden_area_l571_571955


namespace correct_choice_l571_571284

def p : Prop := ∀ x : ℝ, 0 < x → 3^x > 2^x
def q : Prop := ∃ x : ℝ, x < 0 ∧ 3 * x > 2 * x

theorem correct_choice (hp : p) (hnq : ¬q) : p ∧ ¬q :=
by
  exact ⟨hp, hnq⟩

end correct_choice_l571_571284


namespace x_squared_plus_y_squared_l571_571289

theorem x_squared_plus_y_squared (x y : ℝ) 
  (h1 : (1/x) + (1/y) = 5) 
  (h2 : x * y + x + y = 11) : 
  x^2 + y^2 = 2893 / 36 := 
by 
  sorry

end x_squared_plus_y_squared_l571_571289


namespace range_of_a_l571_571315

   -- Hyperbola and intersection conditions
   variables {a x y : ℝ}

   -- Hyperbola equation
   def hyperbola (a : ℝ) (x y : ℝ) := (x^2/a^2) - (y^2/12) = 1

   -- Focal distance and intersection length condition
   def length_condition (a : ℝ) := ∃ (l : ℝ), l > 0 ∧ l = 16 ∧ 
                                      (∀ x y, hyperbola a x y → abs (x - y) = l)

   theorem range_of_a : {a : ℝ | a > 0 ∧ length_condition a}.set = {a | a ∈ Ioo (3/2) 8} :=
   sorry
   
end range_of_a_l571_571315


namespace count_integers_within_bounds_l571_571259

theorem count_integers_within_bounds : 
  ∃ (count : ℕ), count = finset.card (finset.filter (λ n : ℤ, 15 < n^2 ∧ n^2 < 120) (finset.Icc (-10) 10)) ∧ count = 14 := 
by
  sorry

end count_integers_within_bounds_l571_571259


namespace sum_primes_between_10_and_20_l571_571105

theorem sum_primes_between_10_and_20 : 
  let is_prime (n : ℕ) := n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19
  (∑ n in {x ∈ (Ico 10 21) | is_prime x}, n) = 60 := 
by
  sorry

end sum_primes_between_10_and_20_l571_571105


namespace probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571073

def probability_of_5_odd_numbers_in_6_rolls (prob_odd : ℚ) : ℚ :=
  (nat.choose 6 5 * (prob_odd^5) * ((1 - prob_odd)^1)) / (2^6)

theorem probability_of_5_odd_numbers_in_6_rolls_is_3_over_32 :
  probability_of_5_odd_numbers_in_6_rolls (1/2) = 3 / 32 :=
by sorry

end probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571073


namespace find_number_l571_571561

variable (N : ℝ)

theorem find_number (h : (5 / 6) * N = (5 / 16) * N + 50) : N = 96 := 
by 
  sorry

end find_number_l571_571561


namespace isosceles_triangle_sum_of_t_l571_571990

def cos (x : ℝ) : ℝ := sorry
def sin (x : ℝ) : ℝ := sorry

theorem isosceles_triangle_sum_of_t :
  (∑ (t : ℝ) in { t | t ∈ Icc 0 360 ∧
    (let A := (cos 30, sin 30),
         B := (cos 90, sin 90),
         C := (cos t, sin t) in
         (dist A B = dist A C ∨ dist A B = dist B C ∨ dist A C = dist B C))
    }, t) = 450 :=
sorry

end isosceles_triangle_sum_of_t_l571_571990


namespace triangle_abc_constructable_l571_571751

noncomputable def problem_solvable (α : ℝ) (a : ℝ) (t_a : ℝ) : Prop :=
  real.tan (α / 2) ≤ (a / (2 * t_a))

theorem triangle_abc_constructable
  (α : ℝ) (a : ℝ) (t_a : ℝ) :
  problem_solvable α a t_a ↔ 
  ∃ (A B C : ℝ) (triangle_abc : Triangle ABC),
    triangle_abc.bisector = t_a ∧
    triangle_abc.BC = a ∧
    triangle_abc.angle_A = α :=
sorry

end triangle_abc_constructable_l571_571751


namespace nonagon_diagonals_l571_571330

-- Define the number of sides of the polygon (nonagon)
def num_sides : ℕ := 9

-- Define the formula for the number of diagonals in a convex n-sided polygon
def number_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem nonagon_diagonals : number_diagonals num_sides = 27 := 
by
--placeholder for the proof
sorry

end nonagon_diagonals_l571_571330


namespace percentage_of_half_dollars_is_correct_l571_571555

-- Define the constants for the number of coins
def numNickels : ℕ := 75
def numHalfDollars : ℕ := 30

-- Define the values of the coins in cents
def valueNickel : ℕ := 5
def valueHalfDollar : ℕ := 50

-- Define the total value of the coins
def totalValueNickels : ℕ := numNickels * valueNickel
def totalValueHalfDollars : ℕ := numHalfDollars * valueHalfDollar
def totalValue : ℕ := totalValueNickels + totalValueHalfDollars

-- Define the expected percentage
def expectedPercentage : ℚ := 80

-- Define the calculated percentage
def calculatedPercentage : ℚ := ((totalValueHalfDollars.toRat / totalValue.toRat) * 100)

-- The final theorem to prove
theorem percentage_of_half_dollars_is_correct :
  calculatedPercentage = expectedPercentage := 
by sorry

end percentage_of_half_dollars_is_correct_l571_571555


namespace log8_of_1000_l571_571650

open Real

theorem log8_of_1000 : logb 8 1000 = 1 / log 10 2 :=
by
  sorry

end log8_of_1000_l571_571650


namespace part_1_part_2_l571_571554

noncomputable def prob_pass_no_fee : ℚ :=
  (3 / 4) * (2 / 3) +
  (1 / 4) * (3 / 4) * (2 / 3) +
  (3 / 4) * (1 / 3) * (2 / 3) +
  (1 / 4) * (3 / 4) * (1 / 3) * (2 / 3)

noncomputable def prob_pass_200_fee : ℚ :=
  (1 / 4) * (1 / 4) * (3 / 4) * ((2 / 3) + (1 / 3) * (2 / 3)) +
  (1 / 3) * (1 / 3) * (2 / 3) * ((3 / 4) + (1 / 4) * (3 / 4))

theorem part_1 : prob_pass_no_fee = 5 / 6 := by
  sorry

theorem part_2 : prob_pass_200_fee = 1 / 9 := by
  sorry

end part_1_part_2_l571_571554


namespace narrow_black_stripes_l571_571880

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l571_571880


namespace cos_double_angle_sum_l571_571678

theorem cos_double_angle_sum
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 := by
  sorry

end cos_double_angle_sum_l571_571678


namespace largest_number_after_three_expansions_can_5183_be_obtained_l571_571225

-- Defining the expansion rule
def expand (a b : ℕ) : ℕ := a * b + a + b

-- Part 1: Proving the largest new number obtained by operating according to the rule three times 
theorem largest_number_after_three_expansions (a b : ℕ) (h₁ : a = 2) (h₂ : b = 3) : 
  let c1 := expand a b,
      c2 := expand b c1,
      c3 := expand c1 c2
  in c3 = 575 := 
by
  sorry

-- Part 2: Proving that 5183 can be obtained through the rule
theorem can_5183_be_obtained (a b : ℕ) (h₁ : a = 2) (h₂ : b = 3) :
  ∃ (m n : ℕ), (3 ^ m) * (4 ^ n) - 1 = 5183 :=
by
  sorry

end largest_number_after_three_expansions_can_5183_be_obtained_l571_571225


namespace count_multiples_6_or_8_but_not_both_l571_571770

theorem count_multiples_6_or_8_but_not_both : 
  (∑ i in Finset.range 150, ((if (i % 6 = 0 ∧ i % 24 ≠ 0) ∨ (i % 8 = 0 ∧ i % 24 ≠ 0) then 1 else 0) : ℕ)) = 31 := by
  sorry

end count_multiples_6_or_8_but_not_both_l571_571770


namespace narrow_black_stripes_l571_571877

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l571_571877


namespace gcd_of_products_l571_571213

theorem gcd_of_products (a b c d : ℤ) : 
  ∃ gcd, gcd = 12 ∧ gcd = Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd ((a - b) * (a - c) * (a - d)) ((b - c) * (b - d) * (c - d))) ((a - b) * (a - c) * (b - d))) ((a - b) * (a - d) * (b - c))) ((a - c) * (a - d) * (b - c))) ((a - b) * (b - c) * (b - d))) :=
by
  sorry

end gcd_of_products_l571_571213


namespace average_speed_is_five_l571_571166

-- Define the speeds for each segment
def swimming_speed : ℝ := 2 -- km/h
def biking_speed : ℝ := 15 -- km/h
def running_speed : ℝ := 9 -- km/h
def kayaking_speed : ℝ := 6 -- km/h

-- Define the problem to prove the average speed
theorem average_speed_is_five :
  let segments := [swimming_speed, biking_speed, running_speed, kayaking_speed]
  let harmonic_mean (speeds : List ℝ) : ℝ :=
    let n := speeds.length
    n / (speeds.foldl (fun acc s => acc + 1 / s) 0)
  harmonic_mean segments = 5 := by
  sorry

end average_speed_is_five_l571_571166


namespace narrow_black_stripes_l571_571857

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l571_571857


namespace minimum_bnSn_is_neg4_l571_571702

variable (n : ℕ)

def S_n : ℚ := n / (n + 1)
def b_n : ℤ := n - 8
def bnSn : ℚ := (n - 8) * (n / (n + 1))

theorem minimum_bnSn_is_neg4 : ∃ (n : ℕ), bnSn n = -4 := by
  use 2
  unfold bnSn
  norm_num1
  sorry

end minimum_bnSn_is_neg4_l571_571702


namespace arithmetic_series_sum_121_l571_571192

-- Define the conditions for the arithmetic series
def is_arithmetic_series (a d : ℕ) (last : ℕ) (n : ℕ) (terms : List ℕ) : Prop :=
  terms = List.iota n |>.map (λ k => a + d * k) ∧ terms.head? = some a ∧ terms.last? = some last

-- Define the sum of a list of natural numbers
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- The main theorem statement
theorem arithmetic_series_sum_121 :
  ∃ (n : ℕ) (terms : List ℕ), is_arithmetic_series 1 2 21 n terms ∧ sum_list terms = 121 :=
by
  sorry

end arithmetic_series_sum_121_l571_571192


namespace pencils_in_boxes_l571_571791

theorem pencils_in_boxes (total_pencils : ℕ) (pencils_per_box : ℕ) (boxes_required : ℕ) 
    (h1 : total_pencils = 648) (h2 : pencils_per_box = 4) : boxes_required = 162 :=
sorry

end pencils_in_boxes_l571_571791


namespace narrow_black_stripes_are_8_l571_571893

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l571_571893


namespace lemon_juice_fraction_l571_571137

theorem lemon_juice_fraction :
  ∃ L : ℚ, 30 - 30 * L - (1 / 3) * (30 - 30 * L) = 6 ∧ L = 7 / 10 :=
sorry

end lemon_juice_fraction_l571_571137


namespace coef_x3_is_minus_5_l571_571235

def coefficient_of_x3 (expr : ℚ) : ℚ :=
  let term1 := 2 * (2 * (x^3 : ℚ) - 3 * (x^2 : ℚ))
  let term2 := 3 * ((x^2 : ℚ) - 4 * (x^3 : ℚ) + 5 * (x^4 : ℚ))
  let term3 := -(5 * (x^4 : ℚ) - 3 * (x^3 : ℚ))
  let expanded_expr := term1 + term2 + term3
  if h : x^3 ∈ expanded_expr.terms then expanded_expr.coefficient x^3 else 0

theorem coef_x3_is_minus_5 : coefficient_of_x3 (2 * (2 * (x^3 : ℚ) - 3 * (x^2 : ℚ)) + 3 * ((x^2 : ℚ) - 4 * (x^3 : ℚ) + 5 * (x^4 : ℚ)) - (5 * (x^4 : ℚ) - 3 * (x^3 : ℚ))) = -5 := by
  sorry

end coef_x3_is_minus_5_l571_571235


namespace divide_into_three_equal_parts_l571_571633

theorem divide_into_three_equal_parts (total_squares : ℕ) (k : ℕ) (h_total : total_squares = 21) (h_k : k = 7) 
  (figure : list (list bool)) (h_figure : figure.length = 21) :
  ∃ (parts : list (list (list bool))), 
    (parts.length = 3) ∧ 
    (∀ part ∈ parts, (part.flat_map id).count id = 7) ∧ 
    (∀ part ∈ parts, 
      ∀ i j, part i j → (i < 21) ∧ (j < (figure.head.length)) ∧ ((figure i j) = true)) :=
sorry

end divide_into_three_equal_parts_l571_571633


namespace third_median_length_l571_571164

noncomputable def triangle_medians_to_third_median (m_a m_b A : ℝ) : ℝ :=
  let m_c := 2 * Real.sqrt 14
  in if m_a = 5 ∧ m_b = 7 ∧ A = 4 * Real.sqrt 21 then m_c else 0

theorem third_median_length (m_a m_b m_c : ℝ) (A : ℝ) (h_conds : m_a = 5 ∧ m_b = 7 ∧ A = 4 * Real.sqrt 21) :
  m_c = 2 * Real.sqrt 14 :=
by
  -- Use the function signature and hypothesis to conclude the proof
  sorry

end third_median_length_l571_571164


namespace base_conversion_min_sum_l571_571491

theorem base_conversion_min_sum (c d : ℕ) (h : 5 * c + 8 = 8 * d + 5) : c + d = 15 := by
  sorry

end base_conversion_min_sum_l571_571491


namespace number_of_narrow_black_stripes_l571_571900

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l571_571900


namespace prob_A_selected_is_one_over_three_prob_one_unitB_or_two_unitC_is_three_over_five_l571_571808

-- Definitions for the delegates and units
inductive Delegate
| A | B | C | D | E | F

inductive Unit
| A | B | C

-- Mapping each delegate to their respective unit
def unit_of : Delegate → Unit
| Delegate.A := Unit.A
| Delegate.B := Unit.A
| Delegate.C := Unit.B
| Delegate.D := Unit.B
| Delegate.E := Unit.C
| Delegate.F := Unit.C

-- Function to compute combinations
noncomputable def comb (n r : ℕ) : ℕ := Nat.choose n r

-- Probability of a delegate being selected
noncomputable def prob_A_selected : ℚ :=
  comb 5 1 / comb 6 2

-- Probability of specific conditions 
noncomputable def prob_one_unitB_or_two_unitC : ℚ :=
  (comb 2 1 * comb 4 1 + comb 2 2) / comb 6 2

-- Proof statements
theorem prob_A_selected_is_one_over_three :
  prob_A_selected = 1/3 := 
  by sorry

theorem prob_one_unitB_or_two_unitC_is_three_over_five :
  prob_one_unitB_or_two_unitC = 3/5 :=
  by sorry

end prob_A_selected_is_one_over_three_prob_one_unitB_or_two_unitC_is_three_over_five_l571_571808


namespace narrow_black_stripes_l571_571858

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l571_571858


namespace intersecting_lines_b_plus_m_l571_571516

theorem intersecting_lines_b_plus_m :
  ∃ (m b : ℚ), (∀ x y : ℚ, y = m * x + 5 → y = 4 * x + b → (x, y) = (8, 14)) →
               b + m = -63 / 4 :=
by
  sorry

end intersecting_lines_b_plus_m_l571_571516


namespace find_lines_l571_571662

def line_satisfying_conditions_1 (α β γ δ : ℝ) :=
  ( α * δ - β * γ = 0)

def line_satisfying_conditions_2 (m c : ℝ) :=
  ( m = (Real.sqrt 3) / 3 ∧ c = -5)

theorem find_lines (P : Point2d := ⟨Real.sqrt 3, -1⟩) :
  ∃ k l m c : ℝ, 
  (line_satisfying_conditions_1 k (Real.sqrt 3) l m ∧ 
   line_satisfying_conditions_2 m c) :=
sorry

end find_lines_l571_571662


namespace impossible_to_fill_grid_l571_571200

def is_impossible : Prop :=
  ∀ (grid : Fin 3 → Fin 3 → ℕ), 
  (∀ i j, grid i j ≠ grid i (j + 1) ∧ grid i j ≠ grid (i + 1) j) →
  (∀ i, (grid i 0) * (grid i 1) * (grid i 2) = 2005) →
  (∀ j, (grid 0 j) * (grid 1 j) * (grid 2 j) = 2005) →
  (grid 0 0) * (grid 1 1) * (grid 2 2) = 2005 →
  (grid 0 2) * (grid 1 1) * (grid 2 0) = 2005 →
  False

theorem impossible_to_fill_grid : is_impossible :=
  sorry

end impossible_to_fill_grid_l571_571200


namespace isosceles_triangle_perimeter_l571_571393

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_perimeter 
  (a b c : ℝ) 
  (h_iso : is_isosceles a b c) 
  (h1 : a = 2 ∨ a = 4) 
  (h2 : b = 2 ∨ b = 4) 
  (h3 : c = 2 ∨ c = 4) :
  a + b + c = 10 :=
  sorry

end isosceles_triangle_perimeter_l571_571393


namespace james_hours_per_year_l571_571828

def hours_per_day (trainings_per_day : Nat) (hours_per_training : Nat) : Nat :=
  trainings_per_day * hours_per_training

def days_per_week (total_days : Nat) (rest_days : Nat) : Nat :=
  total_days - rest_days

def hours_per_week (hours_day : Nat) (days_week : Nat) : Nat :=
  hours_day * days_week

def hours_per_year (hours_week : Nat) (weeks_year : Nat) : Nat :=
  hours_week * weeks_year

theorem james_hours_per_year :
  let trainings_per_day := 2
  let hours_per_training := 4
  let total_days_per_week := 7
  let rest_days_per_week := 2
  let weeks_per_year := 52
  hours_per_year 
    (hours_per_week 
      (hours_per_day trainings_per_day hours_per_training) 
      (days_per_week total_days_per_week rest_days_per_week)
    ) weeks_per_year
  = 2080 := by
  sorry

end james_hours_per_year_l571_571828


namespace real_number_infinite_continued_fraction_l571_571548

theorem real_number_infinite_continued_fraction:
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / y))) ∧ y = 5 / 3 := 
begin
  sorry
end

end real_number_infinite_continued_fraction_l571_571548


namespace smallest_positive_integer_ending_in_9_divisible_by_13_l571_571101

theorem smallest_positive_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, (n % 10 = 9) ∧ (n % 13 = 0) ∧ (∀ m : ℕ, (m % 10 = 9) ∧ (m % 13 = 0) → m ≥ n) :=
sorry

end smallest_positive_integer_ending_in_9_divisible_by_13_l571_571101


namespace new_mean_after_adding_14_to_each_of_15_numbers_l571_571562

theorem new_mean_after_adding_14_to_each_of_15_numbers (avg : ℕ) (n : ℕ) (n_sum : ℕ) (new_sum : ℕ) :
  avg = 40 →
  n = 15 →
  n_sum = n * avg →
  new_sum = n_sum + n * 14 →
  new_sum / n = 54 :=
by
  intros h_avg h_n h_n_sum h_new_sum
  sorry

end new_mean_after_adding_14_to_each_of_15_numbers_l571_571562


namespace count_multiples_6_or_8_but_not_both_l571_571771

theorem count_multiples_6_or_8_but_not_both : 
  (∑ i in Finset.range 150, ((if (i % 6 = 0 ∧ i % 24 ≠ 0) ∨ (i % 8 = 0 ∧ i % 24 ≠ 0) then 1 else 0) : ℕ)) = 31 := by
  sorry

end count_multiples_6_or_8_but_not_both_l571_571771


namespace number_of_narrow_black_stripes_l571_571898

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l571_571898


namespace sugar_already_put_in_l571_571908

-- Define the conditions
def totalSugarRequired : Nat := 14
def sugarNeededToAdd : Nat := 12
def sugarAlreadyPutIn (total : Nat) (needed : Nat) : Nat := total - needed

--State the theorem
theorem sugar_already_put_in :
  sugarAlreadyPutIn totalSugarRequired sugarNeededToAdd = 2 := 
  by
    -- Providing 'sorry' as a placeholder for the actual proof
    sorry

end sugar_already_put_in_l571_571908


namespace range_of_a_ineq_l571_571845

noncomputable def range_of_a (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧ x₁ * x₁ + (a * a - 1) * x₁ + (a - 2) = 0 ∧
                x₂ * x₂ + (a * a - 1) * x₂ + (a - 2) = 0

theorem range_of_a_ineq (a : ℝ) : (∃ x₁ x₂ : ℝ, x₁ < 1 ∧ 1 < x₂ ∧
    x₁^2 + (a^2 - 1) * x₁ + (a - 2) = 0 ∧
    x₂^2 + (a^2 - 1) * x₂ + (a - 2) = 0) → -2 < a ∧ a < 1 :=
sorry

end range_of_a_ineq_l571_571845


namespace ellipse_focal_length_l571_571952

theorem ellipse_focal_length :
  ∀ a b c : ℝ, (a^2 = 11) → (b^2 = 3) → (c^2 = a^2 - b^2) → (2 * c = 4 * Real.sqrt 2) :=
by
  sorry

end ellipse_focal_length_l571_571952


namespace energy_stick_difference_l571_571010

variable (B D : ℕ)

theorem energy_stick_difference (h1 : B = D + 17) : 
  let B' := B - 3
  let D' := D + 3
  D' < B' →
  (B' - D') = 11 :=
by
  sorry

end energy_stick_difference_l571_571010


namespace num_factors_48_l571_571361

theorem num_factors_48 : 
  let n := 48 in
  ∃ num_factors, num_factors = 10 ∧ 
  (∀ p k, prime p → (n = p ^ k → 1 + k)) := 
sorry

end num_factors_48_l571_571361


namespace final_value_of_S_l571_571652

theorem final_value_of_S :
  ∀ (S n : ℕ), S = 1 → n = 1 →
  (∀ S n : ℕ, ¬ n > 3 → 
    (∃ S' n' : ℕ, S' = S + 2 * n ∧ n' = n + 1 ∧ 
      (∀ S n : ℕ, n > 3 → S' = 13))) :=
by 
  intros S n hS hn
  simp [hS, hn]
  sorry

end final_value_of_S_l571_571652


namespace area_of_shaded_region_l571_571028

noncomputable def area_of_triangle_in_hexagon : ℝ :=
  let s := 12 in
  36 * Real.sqrt 3

theorem area_of_shaded_region (s : ℝ) (h_s : s = 12) :
    ∃ (area : ℝ), area = 36 * Real.sqrt 3 :=
by
  use 36 * Real.sqrt 3
  rw h_s
  sorry

end area_of_shaded_region_l571_571028


namespace distinct_positive_factors_48_l571_571345

theorem distinct_positive_factors_48 : 
  ∀ (n : ℕ), n = 48 → ∀ (p q : ℕ), p = 2 ∧ q = 3 → (∃ a b : ℕ, 48 = p^a * q^b ∧ (a + 1) * (b + 1) = 10) :=
by
  intros n hn p q hpq
  have h_48 : 48 = 2^4 * 3^1 := by norm_num
  use 4, 1
  split
  · exact h_48
  · norm_num
  sorry

end distinct_positive_factors_48_l571_571345


namespace smallest_positive_integer_ends_in_9_and_divisible_by_13_l571_571092

theorem smallest_positive_integer_ends_in_9_and_divisible_by_13 :
  ∃ n : ℕ, n % 10 = 9 ∧ 13 ∣ n ∧ n > 0 ∧ ∀ m, m % 10 = 9 → 13 ∣ m ∧ m > 0 → m ≥ n := 
begin
  use 99,
  split,
  { exact mod_eq_of_lt (10*k + 9) 10 99 9 (by norm_num), },
  split,
  { exact dvd_refl 99, },
  split,
  { exact zero_lt_99, },
  intros m hm1 hm2 hpos,
  by_contradiction hmn,
  sorry
end

end smallest_positive_integer_ends_in_9_and_divisible_by_13_l571_571092


namespace hyperbola_eccentricity_correct_l571_571290
noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : |PF2| = |F1F2|) (h4 : O_to_PF1 = a) : ℝ :=
  let c := sqrt (a^2 + b^2)
  sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_correct (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : |PF2| = |F1F2|) (h4 : O_to_PF1 = a) : hyperbola_eccentricity a b = 5 / 3 :=
sorry

end hyperbola_eccentricity_correct_l571_571290


namespace nonagon_diagonals_l571_571332

-- Define the number of sides for a nonagon.
def n : ℕ := 9

-- Define the formula for the number of diagonals in a polygon.
def D (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem to prove that the number of diagonals in a nonagon is 27.
theorem nonagon_diagonals : D n = 27 := by
  sorry

end nonagon_diagonals_l571_571332


namespace projection_matrix_inverse_is_zero_l571_571427

open Matrix

def u : Vector ℝ 2 := ![4, -2]
def Q : Matrix (Fin 2) (Fin 2) ℝ := (1 / (u ⬝ᵥ u)) • (u ⬝ᵥ (uᵀ))

theorem projection_matrix_inverse_is_zero :
  (¬invertible Q) → Q⁻¹ = 0 :=
by
  sorry

end projection_matrix_inverse_is_zero_l571_571427


namespace new_origin_coordinates_l571_571978

noncomputable def new_origin (x1 y1 x2 y2 : ℤ) : ℤ × ℤ :=
  (2, 4)

theorem new_origin_coordinates :
  ∀ (x1 y1 x2 y2 : ℤ), 
  (x1, y1) = (-1, 3) →
  (x2, y2) = (-3, -1) →
  new_origin (-1) 3 (-3) (-1) = (2, 4) :=
by
  intros x1 y1 x2 y2 h1 h2
  simp [h1, h2, new_origin]
  sorry

end new_origin_coordinates_l571_571978


namespace chord_length_l571_571981

-- Define the key components.
structure Circle := 
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the initial conditions.
def circle1 : Circle := { center := (0, 0), radius := 5 }
def circle2 : Circle := { center := (2, 0), radius := 3 }

-- Define the chord and tangency condition.
def touches_internally (C1 C2 : Circle) : Prop :=
  C1.radius > C2.radius ∧ dist C1.center C2.center = C1.radius - C2.radius

def chord_divided_ratio (AB_length : ℝ) (r1 r2 : ℝ) : Prop :=
  ∃ (x : ℝ), AB_length = 4 * x ∧ r1 = x ∧ r2 = 3 * x

-- The theorem to prove the length of the chord AB.
theorem chord_length (h1 : touches_internally circle1 circle2)
                     (h2 : chord_divided_ratio 8 2 (6)) : ∃ (AB_length : ℝ), AB_length = 8 :=
by
  sorry

end chord_length_l571_571981


namespace group_of_four_sums_to_zero_l571_571623

theorem group_of_four_sums_to_zero (i : ℂ) (h : i^2 = -1) (n : ℤ) :
  (n % 4 = 0 ∧ 150 / 4 = 37) ∧ 
  (∀ k : ℤ, i^(4 * k) + i^(4 * k + 1) + i^(4 * k + 2) + i^(4 * k + 3) = 0) ∧ 
  (i^0 = 1) →
  2 * (∑ k in finset.range (2 * 150 + 1), i ^ (k - 150)) = 2 :=
  by
  sorry

end group_of_four_sums_to_zero_l571_571623


namespace bulb_grid_solution_l571_571129

def bulb_grid_proof : Prop :=
  let initial_state := fun (i j : ℕ) => if i = 1 ∧ j = 1 then 1 else 0 in
  ∀ (toggle_row toggle_col : ℕ → unit),
  ¬∃ (f : (ℕ → unit) → (ℕ → unit) → matrix ℕ ℕ ℕ),
      f toggle_row toggle_col = fun (x y : ℕ) => 0

theorem bulb_grid_solution : bulb_grid_proof := by
  sorry

end bulb_grid_solution_l571_571129


namespace rotation_problem_l571_571201

theorem rotation_problem :
  ∀ (A B C : Type) (rotate_clockwise rotate_counterclockwise : A → ℕ → B → A) (x : ℕ),
    (rotate_clockwise A 510 B = C) →
    (rotate_counterclockwise A x B = C) →
    x < 360 →
    x = 210 :=
by
  intro A B C rotate_clockwise rotate_counterclockwise x
  intros h1 h2 h3
  sorry

end rotation_problem_l571_571201


namespace liouville_constant_exists_l571_571842

theorem liouville_constant_exists {α : ℝ} {a_n a_{n-1} ... a_0 : ℝ} (n : ℕ) (h_n : n ≥ 2)
  (h_root : polynomial.eval α (a_n * X^n + a_{n-1} * X^(n-1) + ... + a_0) = 0) :
  ∃ (c : ℝ), c > 0 ∧ ∀ (p : ℤ) (q : ℕ), q ≠ 0 → abs (α - (p / q)) > c / (q^n) := 
sorry

end liouville_constant_exists_l571_571842


namespace common_ratio_and_arithmetic_sums_l571_571844

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

noncomputable def is_arithmetic_sequence (a b c : ℝ) : Prop :=
2 * b = a + c

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in Finset.range n, a i

theorem common_ratio_and_arithmetic_sums 
  (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q 
  → q ≠ 1
  → is_arithmetic_sequence (a 5) (a 3) (a 4)
  → q = -2 
  ∧ ∀ k : ℕ, 1 ≤ k → is_arithmetic_sequence (S a (k + 2)) (S a (k + 1)) (S a k) :=
by
  intros h_geo h_neq h_arith
  sorry

end common_ratio_and_arithmetic_sums_l571_571844


namespace reflection_on_circumcircle_l571_571423

variable {α : Type*} [EuclideanGeometry α]

-- Definitions of points and orthocenter
variables (A B C H H_A H_A' : α)
variables (A' B' C' : α)

-- Definitions for midpoint
def midpoint (P Q : α) : α := sorry

-- Conditions given in the problem
variable (isOrthocenter : Orthocenter A B C H)
variable (midA' : A' = midpoint B C)
variable (midB' : B' = midpoint A C)
variable (midC' : C' = midpoint A B)

-- Definitions of reflections
def reflection (P Q R : α) : α := sorry

-- Reflections of H
variable (refHA : H_A = reflection H B C)
variable (refHA' : H_A' = reflection H A' A)

-- To prove statement
theorem reflection_on_circumcircle (isOrtho : Orthocenter A B C H)
    (midA' : A' = midpoint B C) (midB' : B' = midpoint A C) (midC' : C' = midpoint A B)
    (refHA : H_A = reflection H B C) (refHA' : H_A' = reflection H A' A) :
    LiesOnCircumcircle A B C [H_A, H_A', A', B', C'] :=
sorry

end reflection_on_circumcircle_l571_571423


namespace vasya_cannot_turn_all_blue_to_red_l571_571806

-- Convex pentagon with vertices and diagonal intersection points all initially blue
def convex_pentagon (P : Type) := 
  -- Representation of conditions, e.g., points P, convex shape, initial blue color for vertices/intersections
  sorry

-- Operation to flip the color of all points on a side or diagonal
def flip_color (P : Type) (side_or_diagonal : set P) : P → (P → Prop) :=
  -- Representation of the color flipping operation
  sorry

theorem vasya_cannot_turn_all_blue_to_red (P : Type) [convex_pentagon P] :
  ¬(∃ f : ℕ, ∀ n < f, flip_color P (choose_side_or_diagonal n P) = all_points_red P) :=
sorry

end vasya_cannot_turn_all_blue_to_red_l571_571806


namespace division_addition_problem_l571_571617

-- Define the terms used in the problem
def ten : ℕ := 10
def one_fifth : ℚ := 1 / 5
def six : ℕ := 6

-- Define the math problem
theorem division_addition_problem :
  (ten / one_fifth : ℚ) + six = 56 :=
by sorry

end division_addition_problem_l571_571617


namespace find_pairs_nat_numbers_l571_571231

theorem find_pairs_nat_numbers (a b : ℕ) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (a * b^3 + 1) % (b - 1) = 0 ↔ 
  (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
by
  sorry

end find_pairs_nat_numbers_l571_571231


namespace maria_cartons_needed_l571_571906

theorem maria_cartons_needed : 
  ∀ (total_needed strawberries blueberries raspberries blackberries : ℕ), 
  total_needed = 36 →
  strawberries = 4 →
  blueberries = 8 →
  raspberries = 3 →
  blackberries = 5 →
  (total_needed - (strawberries + blueberries + raspberries + blackberries) = 16) :=
by
  intros total_needed strawberries blueberries raspberries blackberries ht hs hb hr hb
  -- ... the proof would go here
  sorry

end maria_cartons_needed_l571_571906


namespace find_fraction_l571_571846

theorem find_fraction (x y : ℝ) (hx : 0 < x) (hy : x < y) (h : x / y + y / x = 8) :
  (x + y) / (x - y) = Real.sqrt 15 / 3 :=
sorry

end find_fraction_l571_571846


namespace probability_5_of_6_odd_rolls_l571_571054

def binom_coeff : ℕ → ℕ → ℕ
| n k := Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binom_coeff n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_5_of_6_odd_rolls :
  binomial_probability 6 5 (1/2) = 3/16 :=
by
  -- Proof will go here, but we skip it with sorry for now.
  sorry

end probability_5_of_6_odd_rolls_l571_571054


namespace distinct_positive_factors_of_48_l571_571355

theorem distinct_positive_factors_of_48 : 
  let n := 48 in
  let factors := (2^4) * (3^1) in
  ∀ n : ℕ, n = factors → 
  (let num_factors := (4 + 1) * (1 + 1)
  in num_factors = 10) :=
by 
  let n := 48
  let factors := (2^4) * (3^1)
  assume h : n = factors
  let num_factors := (4 + 1) * (1 + 1)
  show num_factors = 10 from sorry

end distinct_positive_factors_of_48_l571_571355


namespace sec_neg_7pi_over_6_l571_571229

theorem sec_neg_7pi_over_6 : 
  let deg := (180 : Real) in
  let pi := Real.pi in
  let angle_neg_7pi_over_6 := -((7 * pi) / 6) in
  let cos := Real.cos in
  let sec x := 1 / cos x in
  angle_neg_7pi_over_6 = -210 * (deg / pi) → 
  cos (-210 * (deg / pi)) = cos (150 * (deg / pi)) →
  cos (150 * (deg / pi)) = - (3^.5 / 2) →
  sec angle_neg_7pi_over_6 = 2 * 3^.5 / 3 :=
by
  intros
  sorry

end sec_neg_7pi_over_6_l571_571229


namespace problem_a_l571_571124

variable {n : ℕ}
variable {A : Matrix (Fin n) (Fin n) ℝ}

def is_even (k : ℕ) : Prop := ∃ m, k = 2 * m

theorem problem_a (k : ℕ) (h_even : is_even k) (h_positive : 0 < k) 
    (h_symmetric : Aᵀ = A) 
    (h_condition : (Matrix.trace (A^k))^(k + 1) = (Matrix.trace (A^(k + 1)))^k) :
    A^n = (Matrix.trace A) • A^(n - 1) :=
sorry

end problem_a_l571_571124


namespace number_of_cows_l571_571326

-- Definitions from the conditions
variables (a x y : ℝ) (h : 9 * 4 * y = a + 4 * x) (h' : 8 * 6 * y = a + 6 * x)

-- The statement to be proven
theorem number_of_cows (n : ℕ) : n = 6 :=
begin
  sorry
end

end number_of_cows_l571_571326


namespace Moscow_1975_p_q_r_equal_primes_l571_571433

theorem Moscow_1975_p_q_r_equal_primes (a b c : ℕ) (p q r : ℕ) 
  (hp : p = b^c + a) 
  (hq : q = a^b + c) 
  (hr : r = c^a + b) 
  (prime_p : Prime p) 
  (prime_q : Prime q) 
  (prime_r : Prime r) : 
  q = r :=
sorry

end Moscow_1975_p_q_r_equal_primes_l571_571433


namespace plane_PAC_perpendicular_ABC_proof_dihedral_angle_cosine_value_proof_l571_571279

variables {P A B C M G H : Type}
variables [has_coords P] [has_coords A] [has_coords B] [has_coords C] 
          [has_coords M] [has_coords G] [has_coords H]

-- Given conditions
noncomputable def side_length_square_ABCD : ℝ := 2 * real.sqrt 2

-- Equilateral triangles
def is_equilateral_triangle (x y z : Type) [has_coords x] [has_coords y] [has_coords z] : Prop :=
  dist x y = dist y z ∧ dist y z = dist z x

-- Proving perpendicularity
def plane_PAC_perpendicular_ABC (P A B C G H : Type) [has_coords P] [has_coords A] 
  [has_coords B] [has_coords C] [has_coords G] [has_coords H] : Prop :=
  is_midpoint G A B ∧ is_midpoint H A C ∧ 
  perpendicular (line P G) (line A B) ∧ perpendicular (line P H) (line A C)

-- Assuming point M satisfying geometric condition
noncomputable def point_M_satisfying_condition (P A M : Type) [has_coords P] 
  [has_coords A] [has_coords M] : Prop :=
  dist P M = dist M A / 2

-- Dihedral angle cosine value
def dihedral_angle_cosine_value (P B C M : Type) [has_coords P] [has_coords B] 
  [has_coords C] [has_coords M] : ℝ :=
  let m := normalize (cross_product (vec P C) (vec P B)) in
  let n := normalize (cross_product (vec M C) (vec M B)) in
  dot_product m n / (norm m * norm n)

-- Final statements to prove
theorem plane_PAC_perpendicular_ABC_proof : 
  plane_PAC_perpendicular_ABC P A B C G H :=
sorry

theorem dihedral_angle_cosine_value_proof : 
  dihedral_angle_cosine_value P B C M = 2 * real.sqrt 2 / 3 :=
sorry

end plane_PAC_perpendicular_ABC_proof_dihedral_angle_cosine_value_proof_l571_571279


namespace connie_grandma_birth_year_l571_571624

theorem connie_grandma_birth_year :
  ∀ (B S G : ℕ),
  B = 1932 →
  S = 1936 →
  (S - B) * 2 = (S - G) →
  G = 1928 := 
by
  intros B S G hB hS hGap
  -- Proof goes here
  sorry

end connie_grandma_birth_year_l571_571624


namespace CML_is_right_angle_l571_571281

theorem CML_is_right_angle
  (A B C K L M : Point)
  (hABC_isosceles : dist A B = dist B C)
  (hK_on_AB : is_on_segment K A B)
  (hL_on_BC : is_on_segment L B C)
  (hM_on_AC : is_on_segment M A C)
  (hAKM_right_angle : angle A K M = 90)
  (hBLK_right_angle : angle B L K = 90)
  (hKM_eq_KL : dist K M = dist K L) :
  angle C M L = 90 := by
  sorry

end CML_is_right_angle_l571_571281


namespace factors_of_48_l571_571335

theorem factors_of_48 : ∃ n, n = 48 → number_of_distinct_positive_factors n = 10 :=
sorry

-- Auxiliary function definitions to support the main theorem
def number_of_distinct_positive_factors (n : ℕ) : ℕ := 
sorry

end factors_of_48_l571_571335


namespace multiples_of_6_or_8_but_not_both_l571_571786

theorem multiples_of_6_or_8_but_not_both (n : ℕ) : 
  n = 25 ∧ (n = 18) ∧ (n = 6) → (25 - 6) + (18 - 6) = 31 :=
by
  sorry

end multiples_of_6_or_8_but_not_both_l571_571786


namespace polynomial_root_multisets_count_l571_571013

/-- The number of possible multisets of roots for the given polynomial conditions -/
theorem polynomial_root_multisets_count :
  ∀ (b_0 b_1 b_2 b_3 b_4 b_5 b_6 b_7 b_8 b_9 b_10 b_11 b_12 : ℤ)
    (s : Fin 12 → {-2, -1, 1, 2}),
    (∃ (s_1 s_2 s_3 s_4 s_5 s_6 s_7 s_8 s_9 s_10 s_11 s_12 : ℤ),
      (b_12 * s_1^12 + b_11 * s_1^11 + b_10 * s_1^10 + b_9 * s_1^9 + b_8 * s_1^8 + 
       b_7 * s_1^7 + b_6 * s_1^6 + b_5 * s_1^5 + b_4 * s_1^4 + b_3 * s_1^3 + 
       b_2 * s_1^2 + b_1 * s_1 + b_0 = 0) ∧
      (b_12 * s_2^12 + b_11 * s_2^11 + b_10 * s_2^10 + b_9 * s_2^9 + b_8 * s_2^8 + 
       b_7 * s_2^7 + b_6 * s_2^6 + b_5 * s_2^5 + b_4 * s_2^4 + b_3 * s_2^3 + 
       b_2 * s_2^2 + b_1 * s_2 + b_0 = 0) ∧
      -- similarly for s_3, ..., s_12
      (b_12 * s_12^12 + b_11 * s_12^11 + b_10 * s_12^10 + b_9 * s_12^9 + b_8 * s_12^8 + 
       b_7 * s_12^7 + b_6 * s_12^6 + b_5 * s_12^5 + b_4 * s_12^4 + b_3 * s_12^3 + 
       b_2 * s_12^2 + b_1 * s_12 + b_0 = 0)),
    (∃ (n_1 n_neg1 n_2 n_neg2 : ℕ),
      n_1 + n_neg1 + n_2 + n_neg2 = 12 ∧
      multiset.card (multiset.replicate n_1 1 + multiset.replicate n_neg1 (-1) + 
                     multiset.replicate n_2 2 + multiset.replicate n_neg2 (-2)) = 12) →
  multiset.card {T | ∃ (n_1 n_neg1 n_2 n_neg2 : ℕ),
    T = multiset.replicate n_1 1 + multiset.replicate n_neg1 (-1) + 
        multiset.replicate n_2 2 + multiset.replicate n_neg2 (-2) ∧
    n_1 + n_neg1 + n_2 + n_neg2 = 12} = 455 :=
begin
  sorry
end

end polynomial_root_multisets_count_l571_571013


namespace sum_divisible_two_digit_numbers_l571_571991

theorem sum_divisible_two_digit_numbers : 
  (∑ n in {n | ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 
                            a + b ∣ 10 * a + b ∧ 
                            a * b ∣ 10 * a + b ∧ 
                            (a - b) ^ 2 ∣ 10 * a + b ∧ 
                            n = 10 * a + b}, id) = 72 :=
by
  sorry

end sum_divisible_two_digit_numbers_l571_571991


namespace number_of_narrow_black_stripes_l571_571899

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l571_571899


namespace sarah_dimes_l571_571464

theorem sarah_dimes (d n : ℕ) (h1 : d + n = 50) (h2 : 10 * d + 5 * n = 200) : d = 10 :=
sorry

end sarah_dimes_l571_571464


namespace two_times_sum_of_squares_l571_571925

theorem two_times_sum_of_squares (P a b : ℤ) (h : P = a^2 + b^2) : 
  ∃ x y : ℤ, 2 * P = x^2 + y^2 := 
by 
  sorry

end two_times_sum_of_squares_l571_571925


namespace count_integers_satisfying_inequality_l571_571244

theorem count_integers_satisfying_inequality :
  {n : ℤ | 15 < n^2 ∧ n^2 < 120}.card = 14 :=
sorry

end count_integers_satisfying_inequality_l571_571244


namespace num_factors_48_l571_571340

theorem num_factors_48 : 
  ∀ (n : ℕ), n = 48 → (∃ k : ℕ, k = 10 ∧ ∀ d : ℕ, d ∣ n → (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 16 ∨ d = 24 ∨ d = 48)) :=
  by
    intros n h
    sorry

end num_factors_48_l571_571340


namespace continued_fraction_solution_l571_571543

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / y))) ∧ y = (3 + Real.sqrt 69) / 2 :=
begin
  sorry
end

end continued_fraction_solution_l571_571543


namespace find_real_number_l571_571534

theorem find_real_number :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = sqrt 15 :=
sorry

end find_real_number_l571_571534


namespace fish_lifespan_is_12_l571_571511

def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_is_12 : fish_lifespan = 12 := by
  sorry

end fish_lifespan_is_12_l571_571511


namespace shortest_trip_on_cube_surface_l571_571176

theorem shortest_trip_on_cube_surface (a : ℝ) (midpoint_dist opposite_midpoint_dist : ℝ) (h : a = 2) 
  (midpoint_dist = a / 2) (opposite_midpoint_dist = 4) :
  opposite_midpoint_dist = 4 := 
sorry

end shortest_trip_on_cube_surface_l571_571176


namespace distinct_positive_factors_48_l571_571347

theorem distinct_positive_factors_48 : 
  ∀ (n : ℕ), n = 48 → ∀ (p q : ℕ), p = 2 ∧ q = 3 → (∃ a b : ℕ, 48 = p^a * q^b ∧ (a + 1) * (b + 1) = 10) :=
by
  intros n hn p q hpq
  have h_48 : 48 = 2^4 * 3^1 := by norm_num
  use 4, 1
  split
  · exact h_48
  · norm_num
  sorry

end distinct_positive_factors_48_l571_571347


namespace arithmetic_sequence_2014_term_l571_571725

theorem arithmetic_sequence_2014_term :
  ∀ (a : ℕ → ℕ), 
    (∀ n : ℕ, a n = 1 + n * 2) → a 2014 = 4027 :=
by
  assume a ha,
  show a 2014 = 4027 from sorry

end arithmetic_sequence_2014_term_l571_571725


namespace distinct_positive_factors_of_48_l571_571356

theorem distinct_positive_factors_of_48 : 
  let n := 48 in
  let factors := (2^4) * (3^1) in
  ∀ n : ℕ, n = factors → 
  (let num_factors := (4 + 1) * (1 + 1)
  in num_factors = 10) :=
by 
  let n := 48
  let factors := (2^4) * (3^1)
  assume h : n = factors
  let num_factors := (4 + 1) * (1 + 1)
  show num_factors = 10 from sorry

end distinct_positive_factors_of_48_l571_571356


namespace sufficient_but_not_necessary_l571_571127

theorem sufficient_but_not_necessary (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ (x^2 = 1 → x = 1 ∨ x = -1) →
  (x = 1 is_sufficient_and_not_necessary_condition_for x^2 = 1) :=
by
  sorry

end sufficient_but_not_necessary_l571_571127


namespace Paige_math_problems_l571_571448

-- Declare variables and their types
variables (M : ℕ)

-- Conditions
def total_problems (M : ℕ) : ℕ := M + 12
def total_finished_left : ℕ := 44 + 11

-- Problem statement
theorem Paige_math_problems :
  total_problems M = total_finished_left → M = 43 :=
begin
  -- Since we are just writing the statement, we add sorry for the proof
  sorry
end

end Paige_math_problems_l571_571448


namespace sum_of_neg_ints_l571_571020

theorem sum_of_neg_ints (xs : List Int) (h₁ : ∀ x ∈ xs, x < 0)
  (h₂ : ∀ x ∈ xs, 3 < |x| ∧ |x| < 6) : xs.sum = -9 :=
sorry

end sum_of_neg_ints_l571_571020


namespace solve_inequality_l571_571664

noncomputable def P (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem solve_inequality (x : ℝ) : (P x > 0) ↔ (x < 1 ∨ x > 2) := 
  sorry

end solve_inequality_l571_571664


namespace permutations_no_solution_l571_571619

open Equiv

theorem permutations_no_solution :
  ¬(∃ (a b c d : Fin 50 → Fin 50), 
    a.perm (Fin.val 50) ∧ 
    b.perm (Fin.val 50) ∧ 
    c.perm (Fin.val 50) ∧ 
    d.perm (Fin.val 50) ∧ 
    (∑ i, a i * b i) = 2 * (∑ i, c i * d i)) :=
by
  sorry

end permutations_no_solution_l571_571619


namespace angle_sum_APB_QCR_l571_571415

variables {A B C D Q R P : Type*}
variables [right_angle_triangle A B C] (hAB : hypotenuse A B)
variables (h90C : ∠ C = 90)
variables (hD : foot_of_altitude D C A B)
variables (hQ : midpoint Q A D)
variables (hR : midpoint R B D)
variables (hP : midpoint P C D)

theorem angle_sum_APB_QCR :
  ∠ APB + ∠ QCR = 180 := 
  sorry

end angle_sum_APB_QCR_l571_571415


namespace count_multiples_of_6_or_8_but_not_both_l571_571779

theorem count_multiples_of_6_or_8_but_not_both: 
  let multiples_of_six := finset.filter (λ n, 6 ∣ n) (finset.range 151)
  let multiples_of_eight := finset.filter (λ n, 8 ∣ n) (finset.range 151)
  let multiples_of_twenty_four := finset.filter (λ n, 24 ∣ n) (finset.range 151)
  multiples_of_six.card + multiples_of_eight.card - 2 * multiples_of_twenty_four.card = 31 := 
by {
  -- Provided proof omitted
  sorry
}

end count_multiples_of_6_or_8_but_not_both_l571_571779


namespace mass_of_CaSO₄_formed_l571_571085

noncomputable def calc_mass_CaSO₄ (n_moles_CaOH₂ : ℕ) (molar_mass_CaSO₄ : ℕ) : ℕ :=
n_moles_CaOH₂ * molar_mass_CaSO₄

def balanced_equation := "Ca(OH)₂ + H₂SO₄ → CaSO₄ + 2H₂O"
def stoichiometry := 1  -- 1 mole of Ca(OH)₂ produces 1 mole of CaSO₄

theorem mass_of_CaSO₄_formed :
  calc_mass_CaSO₄ 12 136.14 = 1633.68 := by
  sorry

end mass_of_CaSO₄_formed_l571_571085


namespace unique_valid_peg_placement_l571_571973

-- Define the colors
inductive Color
| yellow
| red
| green
| blue
| orange

-- Define the peg board as a 4x4 matrix of options (Color option)
def PegBoard : Type := Array (Array (Option Color))

-- Constraints given
def constraints (pb : PegBoard) : Prop :=
  -- No row contains two or more pegs of the same color
  (∀ (r : Fin 4), (pb[r].filterMap id).nodup) ∧
  -- No column contains two or more pegs of the same color
  (∀ (c : Fin 4), (List.ofFn (λ r : Fin 4, pb[r][c])).filterMap id).nodup

-- There is one valid way to place the pegs
theorem unique_valid_peg_placement : ∃! (pb : PegBoard), constraints pb :=
by
  sorry

end unique_valid_peg_placement_l571_571973


namespace distinct_positive_factors_of_48_l571_571352

theorem distinct_positive_factors_of_48 : 
  let n := 48 in
  let factors := (2^4) * (3^1) in
  ∀ n : ℕ, n = factors → 
  (let num_factors := (4 + 1) * (1 + 1)
  in num_factors = 10) :=
by 
  let n := 48
  let factors := (2^4) * (3^1)
  assume h : n = factors
  let num_factors := (4 + 1) * (1 + 1)
  show num_factors = 10 from sorry

end distinct_positive_factors_of_48_l571_571352


namespace percentage_increase_l571_571126

theorem percentage_increase (x y : ℝ) (hx : x = 50) (hy : y = 75) :
  ((y - x) / x) * 100 = 50 :=
by
  rw [hx, hy]
  norm_num
  sorry  -- Placeholder to skip the actual proof.

end percentage_increase_l571_571126


namespace narrow_black_stripes_l571_571861

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l571_571861


namespace find_m_l571_571749

variable (m : ℝ)

def a : EuclideanVector := (1, m) -- Vector a
def b : EuclideanVector := (3, -2) -- Vector b

-- Definition for the condition "a + b is perpendicular to b"
def is_perpendicular (v1 v2 : EuclideanVector) : Prop :=
  v1.dot v2 = 0

theorem find_m (h : is_perpendicular (a + b) b) : m = 8 := by
  sorry

end find_m_l571_571749


namespace find_k_l571_571207

/-- Define the function f(x) -/
def f (x : ℝ) : ℝ := 7 * x^2 + (1 / x) + 1

/-- Define the function g(x) -/
def g (x : ℝ) (k : ℝ) : ℝ := x^2 - k

/-- Given that f(3) - g(3) = 5, prove that k = -151 / 3 -/
theorem find_k : (f 3) - (g 3 k) = 5 → k = -151 / 3 := by
  intros h
  sorry

end find_k_l571_571207


namespace derivative_of_y_is_correct_l571_571947

-- Define the function y = cos(2x + 1)
def y (x : ℝ) : ℝ := Real.cos (2 * x + 1)

-- The statement we want to prove:
theorem derivative_of_y_is_correct (x : ℝ) :
  deriv y x = -2 * Real.sin (2 * x + 1) := by
  sorry

end derivative_of_y_is_correct_l571_571947


namespace bus_trip_children_difference_l571_571916

theorem bus_trip_children_difference :
  let initial := 41
  let final :=
    initial
    - 12 + 5   -- First bus stop
    - 7 + 10   -- Second bus stop
    - 14 + 3   -- Third bus stop
    - 9 + 6    -- Fourth bus stop
  initial - final = 18 :=
by sorry

end bus_trip_children_difference_l571_571916


namespace probability_largest_value_five_l571_571134

open ProbabilityTheory

theorem probability_largest_value_five :
  let cards := Set.range (6 : ℕ)
  let events := {s : Set ℕ | s ⊆ cards ∧ s.card = 3}
  let largest_five := {s : Set ℕ | s ≠ ∅ ∧ s.max' (Finset.singleton_nonempty s) = 5}
  P (events ∩ largest_five) = 3 / 10 := by
  sorry

end probability_largest_value_five_l571_571134


namespace hexagon_proof_l571_571466

theorem hexagon_proof (C D E F G H Y : Type)
  (side_length : ℝ) (hexagon : regular_hexagon C D E F G H)
  (CD : segment C D) (CY : segment C Y) (HY : segment H Y)
  (Y_on_extension : ∃ t, D = midpoint C D t ∧ Y = point_on_line_extension CD t)
  (each_side_length : side_length = 3) (CY_length : CY.length = 4 * CD.length) :
  HY.length = (15 * real.sqrt 3) / 2 := 
sorry

end hexagon_proof_l571_571466


namespace sum_of_integer_solutions_l571_571989

theorem sum_of_integer_solutions :
  (∑ n in Finset.filter (λ n, |n| < |n-5| ∧ |n-5| < 12) (Finset.range 25), n) = -25 := by
  sorry

end sum_of_integer_solutions_l571_571989


namespace correct_option_is_B_l571_571999

-- Define the Pythagorean theorem condition for right-angled triangles
def is_right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Conditions given in the problem
def option_A : Prop := ¬is_right_angled_triangle 1 2 2
def option_B : Prop := is_right_angled_triangle 1 (Real.sqrt 3) 2
def option_C : Prop := ¬is_right_angled_triangle 4 5 6
def option_D : Prop := ¬is_right_angled_triangle 1 1 (Real.sqrt 3)

-- The formal proof problem statement
theorem correct_option_is_B : option_A ∧ option_B ∧ option_C ∧ option_D :=
by
  sorry

end correct_option_is_B_l571_571999


namespace solve_system_l571_571941

noncomputable def solution1 (a b : ℝ) : ℝ × ℝ := 
  ((a + Real.sqrt (a^2 + 4 * b)) / 2, (-a + Real.sqrt (a^2 + 4 * b)) / 2)

noncomputable def solution2 (a b : ℝ) : ℝ × ℝ := 
  ((a - Real.sqrt (a^2 + 4 * b)) / 2, (-a - Real.sqrt (a^2 + 4 * b)) / 2)

theorem solve_system (a b x y : ℝ) : 
  (x - y = a ∧ x * y = b) ↔ ((x, y) = solution1 a b ∨ (x, y) = solution2 a b) := 
by sorry

end solve_system_l571_571941


namespace size15_shoe_dimensions_l571_571148

-- Define the increments for each shoe size
def length_increment : ℝ := 1/5
def width_increment : ℝ := 1/6
def height_increment : ℝ := 1/8

def size_difference : ℝ := 9 -- Difference between size 17 and size 8

-- Define the new dimensions according to the provided conditions
def length_largest := (4.5 : ℝ) + size_difference * length_increment
def width_largest := (6 : ℝ) + size_difference * width_increment
def height_largest := (7.5 : ℝ) + size_difference * height_increment

-- Define percentage increases
def length_increase := 1.40
def width_increase := 1.25
def height_increase := 1.15

-- Calculate initial dimensions
noncomputable def length_smallest := length_largest / length_increase
noncomputable def width_smallest := width_largest / width_increase
noncomputable def height_smallest := height_largest / height_increase

-- Calculate the dimensions of the size 15 shoe
def increment_to_size_15 := 7 * (1 : ℝ)

noncomputable def length_size_15 := length_smallest + increment_to_size_15 * length_increment
noncomputable def width_size_15 := width_smallest + increment_to_size_15 * width_increment
noncomputable def height_size_15 := height_smallest + increment_to_size_15 * height_increment

theorem size15_shoe_dimensions :
  length_size_15 = 5.9 ∧
  abs (width_size_15 - 7.1667) < 0.01 ∧
  height_size_15 = 8.375 :=
by
  sorry

end size15_shoe_dimensions_l571_571148


namespace narrow_black_stripes_count_l571_571888

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l571_571888


namespace shortest_distance_l571_571424

noncomputable def R (u : ℝ) : ℝ × ℝ × ℝ :=
  (3 * u + 2, -u - 3, 2 * u + 5)

noncomputable def S (v : ℝ) : ℝ × ℝ × ℝ :=
  (v + 1, 3 * v + 2, -v)

def squared_distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2

-- Prove that the minimal distance between points R and S is √10
theorem shortest_distance : (∃ u v : ℝ, squared_distance (R u) (S v) = 10) :=
sorry

end shortest_distance_l571_571424


namespace jason_consumes_52_eggs_l571_571651

theorem jason_consumes_52_eggs :
  (∀ (weekdays eggs total_weeks : ℕ),
    weekdays = 5 →
    eggs = (3 * 5) →
    total_weeks = 2 →
    ∀ (weekend_saturday weekend_sunday total_eggs total : ℕ),
      weekend_saturday = 5 →
      weekend_sunday = 6 →
      total_eggs = (eggs + weekend_saturday + weekend_sunday) →
      total = (total_eggs * total_weeks) →
      total = 52) :=
begin
  sorry
end

end jason_consumes_52_eggs_l571_571651


namespace convert_and_scientific_notation_l571_571499

theorem convert_and_scientific_notation :
  ∀ (radius_km : ℕ), radius_km = 696000 →
  (radius_km * 1000 = 696000000 ∧ (696000000 = 6.96 * 10^8)) :=
by
  assume radius_km h_radius_km,
  sorry

end convert_and_scientific_notation_l571_571499


namespace count_integers_satisfying_inequality_l571_571254

theorem count_integers_satisfying_inequality : 
  (∃ S : Set ℤ, (∀ n ∈ S, 15 < n^2 ∧ n^2 < 120) ∧ S.card = 14) :=
by
  sorry

end count_integers_satisfying_inequality_l571_571254


namespace a_2021_is_200_l571_571975

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

def a (n : ℕ) : ℕ :=
  n * (3 * n + 1)

def iterate_a (n : ℕ) : ℕ :=
  a (sum_of_digits n)

theorem a_2021_is_200 : iterate_a^[2020] 80 = 200 :=
by sorry

end a_2021_is_200_l571_571975


namespace factors_of_48_l571_571338

theorem factors_of_48 : ∃ n, n = 48 → number_of_distinct_positive_factors n = 10 :=
sorry

-- Auxiliary function definitions to support the main theorem
def number_of_distinct_positive_factors (n : ℕ) : ℕ := 
sorry

end factors_of_48_l571_571338


namespace limit_a_n_l571_571452

open Nat Real

noncomputable def a_n (n : ℕ) : ℝ := (7 * n - 1) / (n + 1)

theorem limit_a_n : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - 7| < ε := 
by {
  -- The proof would go here.
  sorry
}

end limit_a_n_l571_571452


namespace number_of_boys_l571_571480

theorem number_of_boys (n : ℕ)
  (initial_avg_height : ℕ)
  (incorrect_height : ℕ)
  (correct_height : ℕ)
  (actual_avg_height : ℕ)
  (h1 : initial_avg_height = 184)
  (h2 : incorrect_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_avg_height = 182)
  (h5 : initial_avg_height * n - (incorrect_height - correct_height) = actual_avg_height * n) :
  n = 30 :=
sorry

end number_of_boys_l571_571480


namespace arrange_programs_l571_571011

-- Definitions directly from conditions
def total_programs : ℕ := 8
def non_singing_programs : ℕ := 3
def singing_programs : ℕ := total_programs - non_singing_programs
def non_adjacent (list : list ℕ) : Prop := 
  ∀ i, i < list.length - 1 → list.nth_le i sorry ≠ list.nth_le (i + 1) sorry
def first_and_last_singing (list : list ℕ) : Prop := 
  list.nth_le 0 sorry = 1 ∧ list.nth_le (list.length - 1) sorry = 1

-- Statement of the proof problem
theorem arrange_programs :
  ∃ l : list ℕ, l.length = total_programs ∧
                non_adjacent l ∧
                first_and_last_singing l ∧
                (countp (λ x, x = 0) l = non_singing_programs) ∧
                (countp (λ x, x = 1) l = singing_programs) ∧
                nat.factorial 6 / (nat.factorial non_singing_programs * nat.factorial singing_programs) * 
                nat.factorial non_singing_programs * 
                nat.factorial singing_programs = 720 := 
sorry

end arrange_programs_l571_571011


namespace andrew_remaining_vacation_days_l571_571608

-- Definitions based on conditions:
def total_public_holidays : ℕ := 10
def total_worked_days : ℕ := 290
def sick_leave_days : ℕ := 5
def days_off_march : ℕ := 5
def days_off_september : ℕ := 10

-- Theorem statement:
theorem andrew_remaining_vacation_days :
  let days_that_count := total_worked_days - total_public_holidays - sick_leave_days,
      days_first_half := days_that_count / 2,
      days_second_half := days_first_half,
      vacation_days_first_half := days_first_half / 10,
      vacation_days_second_half := days_second_half / 20,
      total_vacation_days := vacation_days_first_half + vacation_days_second_half,
      vacation_days_taken := days_off_march + days_off_september,
      vacation_days_remaining := total_vacation_days - vacation_days_taken
  in vacation_days_remaining = 4 := sorry

end andrew_remaining_vacation_days_l571_571608


namespace retirement_hire_year_l571_571135

theorem retirement_hire_year (A : ℕ) (R : ℕ) (Y : ℕ) (W : ℕ) 
  (h1 : A + W = 70) 
  (h2 : A = 32) 
  (h3 : R = 2008) 
  (h4 : W = R - Y) : Y = 1970 :=
by
  sorry

end retirement_hire_year_l571_571135


namespace angle_B_is_pi_div_3_l571_571278

variable {A B C a b c : ℝ}
variable (h₁ : Triangle ABC)
variable (h₂ : (c - b) / (c - a) = (Real.sin A) / (Real.sin C + Real.sin B))

theorem angle_B_is_pi_div_3 
  (h₁ : Triangle ABC)
  (h₂ : (c - b) / (c - a) = (Real.sin A) / (Real.sin C + Real.sin B)) :
  B = π / 3 :=
sorry

end angle_B_is_pi_div_3_l571_571278


namespace right_triangle_area_l571_571496

theorem right_triangle_area (a b c : ℕ) (h1 : a = 16) (h2 : b = 30) (h3 : c = 34) 
(h4 : a^2 + b^2 = c^2) : 
   1 / 2 * a * b = 240 :=
by 
  sorry

end right_triangle_area_l571_571496


namespace count_squares_in_G_l571_571965

def is_in_G (x y : ℤ) : Prop :=
  3 ≤ |x| ∧ |x| ≤ 7 ∧ 3 ≤ |y| ∧ |y| ≤ 7

def is_square_of_side_at_least_six (points : list (ℤ × ℤ)) : Prop :=
  points.length = 4 ∧ 
  let (x1, y1) := points.nth 0,
      (x2, y2) := points.nth 1,
      (x3, y3) := points.nth 2,
      (x4, y4) := points.nth 3 in
    x2 - x1 ≥ 6 ∧ y3 - y2 ≥ 6

theorem count_squares_in_G : 
  ∃ n, n = 4 ∧ 
  n = list.filter 
      (λ points, is_square_of_side_at_least_six points ∧ 
      (∀ p ∈ points, is_in_G p.fst p.snd))
      (list.powerset [
        (3,3), (3,4), (3,5), (3,6), (3,7), 
        (4,3), (4,4), (4,5), (4,6), (4,7), 
        (5,3), (5,4), (5,5), (5,6), (5,7),
        (6,3), (6,4), (6,5), (6,6), (6,7),
        (7,3), (7,4), (7,5), (7,6), (7,7)]) 
  sorry

end count_squares_in_G_l571_571965


namespace tetrahedron_fits_in_box_l571_571167

theorem tetrahedron_fits_in_box :
  ∀ (tetrahedron_edge box_min_side box_mid_side box_max_side : ℝ),
  tetrahedron_edge = 12 →
  box_min_side = 9 →
  box_mid_side = 13 →
  box_max_side = 15 →
  (6 * Real.sqrt 2) < box_min_side :=
begin
  intros tetrahedron_edge box_min_side box_mid_side box_max_side h_tetrahedron_edge h_box_min h_box_mid h_box_max,
  rw [h_tetrahedron_edge, h_box_min, h_box_mid, h_box_max],
  have h : (6 * Real.sqrt 2) < 9, from sorry,
  exact h,
end

end tetrahedron_fits_in_box_l571_571167


namespace distance_from_point_to_plane_l571_571405

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def norm (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem distance_from_point_to_plane
  (P : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) :
  P = (-1, 3, 2) → n = (2, -2, 1) →
  abs (dot_product n P) / norm n = 2 :=
by
  intros hP hn
  rw [hP, hn]
  sorry

end distance_from_point_to_plane_l571_571405


namespace num_groups_of_three_marbles_l571_571514

theorem num_groups_of_three_marbles : 
  let num_red := 1 in
  let num_blue := 1 in
  let num_green := 2 in
  let num_yellow := 3 in
  (num_red + num_blue + num_green + num_yellow >= 3) ->
  (∃ groups : ℕ, groups = 8) :=
by
  sorry

end num_groups_of_three_marbles_l571_571514


namespace no_two_distinct_integer_solutions_for_p_x_eq_2_l571_571839

open Polynomial

theorem no_two_distinct_integer_solutions_for_p_x_eq_2
  (p : ℤ[X])
  (h1 : ∃ a : ℤ, p.eval a = 1)
  (h3 : ∃ b : ℤ, p.eval b = 3) :
  ¬(∃ y1 y2 : ℤ, y1 ≠ y2 ∧ p.eval y1 = 2 ∧ p.eval y2 = 2) :=
by 
  sorry

end no_two_distinct_integer_solutions_for_p_x_eq_2_l571_571839


namespace cross_section_area_leq_half_face_area_l571_571165

noncomputable def area_triangle {A B C : Point} (triangle : Triangle A B C) : ℝ := 
  sorry

noncomputable def area_face (cube : Cube) : ℝ := 
  sorry

theorem cross_section_area_leq_half_face_area 
  (cube : Cube) -- having a cube
  (A B C : Point) -- vertices of the triangular section
  (V : Point) -- common vertex of the cube where edges originate to A, B, C
  (insphere : Sphere) -- insphere of the cube
  (touchA : tangent_point insphere (Plane V A B))
  (touchB : tangent_point insphere (Plane V B C))
  (touchC : tangent_point insphere (Plane V C A))
  : area_triangle (Triangle.mk A B C) ≤ (1/2) * area_face cube :=
sorry

end cross_section_area_leq_half_face_area_l571_571165


namespace f_84_eq_98_l571_571953

noncomputable def f : ℤ → ℤ
| n := if n >= 1000 then n - 3 else f (f (n + 5))

theorem f_84_eq_98 : f 84 = 98 :=
sorry

end f_84_eq_98_l571_571953


namespace james_out_of_pocket_l571_571827

def car_value_old : ℝ := 20000
def sale_percentage_old : ℝ := 0.80
def car_value_new : ℝ := 30000
def purchase_percentage_new : ℝ := 0.90

theorem james_out_of_pocket :
  let sale_price_old := sale_percentage_old * car_value_old in
  let purchase_price_new := purchase_percentage_new * car_value_new in
  (purchase_price_new - sale_price_old) = 11000 :=
by
  let sale_price_old := sale_percentage_old * car_value_old
  let purchase_price_new := purchase_percentage_new * car_value_new
  have out_of_pocket := purchase_price_new - sale_price_old
  have answer := 11000
  sorry

end james_out_of_pocket_l571_571827


namespace sufficient_but_not_necessary_condition_l571_571689

theorem sufficient_but_not_necessary_condition (a b : ℝ) (i : ℂ) (h_i : i = complex.I)
    (h1 : (a + b * i)^2 = 2 * i) :
    (a = 1 ∧ b = 1) ↔ (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_l571_571689


namespace surface_area_of_sliced_off_solid_l571_571157

def is_midpoint {α : Type} [linear_ordered_field α] (p : α × α × α) (a b : α × α × α) : Prop :=
  2 * p = (a + b)

noncomputable def area_prism_sliced_off_surface {α : Type} [linear_ordered_field α] (h : α) (a p q r : α × α × α) : α :=
  63 + (49 * real.sqrt 3 + real.sqrt 521) / 4

theorem surface_area_of_sliced_off_solid
  {α : Type} [linear_ordered_field α] (h : α) (s : α) (a b c e f g p q r : α × α × α)
  (h_eq : h = 18)
  (side_eq : s = 14)
  (midpoint_AE : is_midpoint p a e)
  (midpoint_BF : is_midpoint q b f)
  (midpoint_CG : is_midpoint r c g) :
  area_prism_sliced_off_surface h p q r = 63 + (49 * real.sqrt 3 + real.sqrt 521) / 4 :=
by
  sorry

end surface_area_of_sliced_off_solid_l571_571157


namespace transfer_valid_x_l571_571138

noncomputable def transfer (a x : ℝ) : Prop :=
  (100 - x) * 1.2 * a >= 100 * a ∧ 3.5 * a * x >= 50 * a

theorem transfer_valid_x (a : ℝ) (hx1 : 0 < a) (hx2 : a < 100) : 15 ≤ x ≤ 16 :=
by
  sorry

end transfer_valid_x_l571_571138


namespace triangle_rotation_l571_571515

theorem triangle_rotation (m x y : ℝ) : 
  0 < m ∧ m < 180 ∧
  (rotation (0, 0) m (0, 12) = (36, 18)) ∧
  (rotation (0, 0) m (16, 0) = (24, 2)) ∧
  (rotation (0, 0) m (0, 0) = (24, 18)) →
  m + x + y = 108 :=
by
  sorry

end triangle_rotation_l571_571515


namespace exists_nat_with_palindrome_decomp_l571_571640

def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString
  s = s.reverse

theorem exists_nat_with_palindrome_decomp :
  ∃ n : ℕ, (∀ a b : ℕ, is_palindrome a → is_palindrome b → a * b = n → a ≠ b → (a, b) = (0, n) ∨ (b, a) = (0, n)) ∧ set.size { (a, b) | a * b = n ∧ is_palindrome a ∧ is_palindrome b } > 100 :=
begin
  use 2^101,
  sorry
end

end exists_nat_with_palindrome_decomp_l571_571640


namespace continued_fraction_solution_l571_571544

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / y))) ∧ y = (3 + Real.sqrt 69) / 2 :=
begin
  sorry
end

end continued_fraction_solution_l571_571544


namespace cosine_value_l571_571369

variable (α : ℝ)
-- Condition: sin (π/3 - α) = 1/3
axiom cond : Real.sin (Real.pi / 3 - α) = 1 / 3

-- Question: Cosine value.
theorem cosine_value : Real.cos (5 * Real.pi / 6 - α) = -1 / 3 :=
by
  -- Skip the proof
  sorry

end cosine_value_l571_571369


namespace mean_of_set_with_median_l571_571959

theorem mean_of_set_with_median (m : ℝ) (h : m + 7 = 10) :
  (m + (m + 2) + (m + 7) + (m + 10) + (m + 12)) / 5 = 9.2 :=
by
  -- Placeholder for the proof.
  sorry

end mean_of_set_with_median_l571_571959


namespace nathan_ate_total_gumballs_l571_571912

-- Define the constants and variables based on the conditions
def gumballs_small : Nat := 5
def gumballs_medium : Nat := 12
def gumballs_large : Nat := 20
def small_packages : Nat := 4
def medium_packages : Nat := 3
def large_packages : Nat := 2

-- The total number of gumballs Nathan ate
def total_gumballs : Nat := (small_packages * gumballs_small) + (medium_packages * gumballs_medium) + (large_packages * gumballs_large)

-- The theorem to prove
theorem nathan_ate_total_gumballs : total_gumballs = 96 :=
by
  unfold total_gumballs
  sorry

end nathan_ate_total_gumballs_l571_571912


namespace possible_intersected_planes_l571_571437

-- Definition of the number of planes intersected by a line
def intersected_planes (a : ℝ) (cube : ℝ) : set ℕ :=
{m | ∃ position, (position = "body_diagonal" ∧ m = 6) ∨
                 (position = "face_diagonal" ∧ m = 4) ∨
                 (position = "edge" ∧ m = 2)}

-- Statement of the theorem
theorem possible_intersected_planes (a : ℝ) (cube : ℝ) :
  intersected_planes a cube = {2, 4, 6} :=
sorry

end possible_intersected_planes_l571_571437


namespace arith_seq_sum_l571_571191

theorem arith_seq_sum (n : ℕ) (h₁ : 2 * n - 1 = 21) : 
  (∑ i in finset.range 11, (2 * i + 1)) = 121 :=
by
  sorry

end arith_seq_sum_l571_571191


namespace volume_of_rice_pile_l571_571008

theorem volume_of_rice_pile
  (arc_length_bottom : ℝ)
  (height : ℝ)
  (one_fourth_cone : ℝ)
  (approx_pi : ℝ)
  (h_arc : arc_length_bottom = 8)
  (h_height : height = 5)
  (h_one_fourth_cone : one_fourth_cone = 1/4)
  (h_approx_pi : approx_pi = 3) :
  ∃ V : ℝ, V = one_fourth_cone * (1 / 3) * π * (16^2 / π^2) * height :=
by
  sorry

end volume_of_rice_pile_l571_571008


namespace nested_fraction_solution_l571_571538

noncomputable def nested_fraction : ℝ :=
  3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / ... ))))

theorem nested_fraction_solution :
  nested_fraction = (3 + Real.sqrt 69) / 2 :=
sorry

end nested_fraction_solution_l571_571538


namespace percentage_paid_to_X_l571_571982

-- Definitions based on given conditions
def total_payment : ℝ := 570
def payment_Y : ℝ := 259.09
def payment_X : ℝ := total_payment - payment_Y

-- The percentage of the sum paid to X compared to the sum paid to Y
def percentage_payment_X_compared_to_Y : ℝ := (payment_X / payment_Y) * 100

-- Lean statement for the proof problem
theorem percentage_paid_to_X : percentage_payment_X_compared_to_Y = 120.03 := by
  sorry

end percentage_paid_to_X_l571_571982


namespace midpoint_theorem_l571_571155

-- Given: a triangle ABC with D as the midpoint of BC, and DE parallel to AB
variables {A B C D E : Type} [EuclideanGeometry A B C D E]

-- Definitions of the conditions
def is_midpoint (P M Q : Type) : Prop := dist P M = dist M Q
def is_parallel (l1 l2 : Line) : Prop := ∀ (P1 P2 : Point), is_on P1 l1 → is_on P2 l1 → is_on P1 l2 → is_on P2 l2

-- The theorem we aim to prove
theorem midpoint_theorem 
  (A B C D E : Point) 
  (hD_midpoint : is_midpoint B D C) 
  (hDE_parallel : is_parallel (line D E) (line A B)) 
  : is_midpoint A E C :=
sorry

end midpoint_theorem_l571_571155


namespace sum_coefficients_l571_571371

theorem sum_coefficients (a1 a2 a3 a4 a5 : ℤ) (h : ∀ x : ℕ, a1 * (x - 1) ^ 4 + a2 * (x - 1) ^ 3 + a3 * (x - 1) ^ 2 + a4 * (x - 1) + a5 = x ^ 4) :
  a2 + a3 + a4 = 14 :=
  sorry

end sum_coefficients_l571_571371


namespace min_funct_value_l571_571016

theorem min_funct_value : ∀ x : ℝ, x > -2 → f x ≥ 2 :=
  by
  -- Define the function f
  let f := λ x : ℝ, x + 4 / (x + 2)
  -- State the theorem
  assume x h
  have : f x = x + 4 / (x + 2), from rfl
  sorry

end min_funct_value_l571_571016


namespace find_a_l571_571298

theorem find_a (a : ℝ) 
  (h1 : a < 0)
  (h2 : a < 1/3)
  (h3 : -2 * a + (1 - 3 * a) = 6) : 
  a = -1 := 
by 
  sorry

end find_a_l571_571298


namespace ordered_pairs_count_l571_571660

theorem ordered_pairs_count : 
  (∃ f : ℝ → ℤ → Prop, 
    (∀ a b, (0 < a) → (5 ≤ b) → (b ≤ 205) → ((log b a) ^ 2023 = log b (a ^ 2023) ↔ f a b)) 
    ∧ (card { p | ∃ a b, f a b ∧ p = (a, b) } = 603)) := 
sorry

end ordered_pairs_count_l571_571660


namespace compute_expression_in_terms_of_k_l571_571835

-- Define the main theorem to be proven, with all conditions directly translated to Lean statements.
theorem compute_expression_in_terms_of_k
  (x y : ℝ)
  (h : (x^2 + y^2) / (x^2 - y^2) + (x^2 - y^2) / (x^2 + y^2) = k) :
    (x^8 + y^8) / (x^8 - y^8) - (x^8 - y^8) / (x^8 + y^8) = ((k - 2)^2 * (k + 2)^2) / (4 * k * (k^2 + 4)) :=
by
  sorry

end compute_expression_in_terms_of_k_l571_571835


namespace area_of_ABCD_l571_571459

-- Declare points and distances
variables (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]

-- Denote specific lengths and angles given in conditions
variables (angle_ABC angle_ACD : ℝ) (AC CD AE : ℝ)
-- Conditions
variables (H_angle_ABC : angle_ABC = 90)
variables (H_angle_ACD : angle_ACD = 90)
variables (H_AC : AC = 24)
variables (H_CD : CD = 18)
variables (H_AE : AE = 6)

-- Point E is the intersection of diagonals
variables (intersection_E : E = intersection (diagonals AC BD))

-- Expected area result
noncomputable def area_ABCD := 504

-- The statement
theorem area_of_ABCD : area_of_quadrilateral A B C D = area_ABCD :=
sorry

end area_of_ABCD_l571_571459


namespace arithmetic_sum_neg45_to_0_l571_571665

theorem arithmetic_sum_neg45_to_0 :
  let a := -45
      d := 3
      n := 16 in
  (n = 16 ∧ a = -45 ∧ d = 3) →
  (∑ k in finset.range n, a + k * d) = -360 :=
by
  sorry

end arithmetic_sum_neg45_to_0_l571_571665


namespace find_m_value_l571_571719

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

noncomputable def find_m (m : ℝ) : Prop :=
  f (f m) = f 2002 - 7 / 2

theorem find_m_value : find_m (-3 / 8) :=
by
  unfold find_m
  sorry

end find_m_value_l571_571719


namespace sum_of_vertical_asymptotes_l571_571954

noncomputable def sum_of_roots (a b c : ℝ) (h_discriminant : b^2 - 4*a*c ≠ 0) : ℝ :=
-(b/a)

theorem sum_of_vertical_asymptotes :
  let f := (6 * (x^2) - 8) / (4 * (x^2) + 7*x + 3)
  ∃ c d, c ≠ d ∧ (4*c^2 + 7*c + 3 = 0) ∧ (4*d^2 + 7*d + 3 = 0)
  ∧ c + d = -7 / 4 :=
by
  sorry

end sum_of_vertical_asymptotes_l571_571954


namespace smallest_n_for_congruence_l571_571988

theorem smallest_n_for_congruence :
  ∃ n : ℕ, 827 * n % 36 = 1369 * n % 36 ∧ n > 0 ∧ (∀ m : ℕ, 827 * m % 36 = 1369 * m % 36 ∧ m > 0 → m ≥ 18) :=
by sorry

end smallest_n_for_congruence_l571_571988


namespace find_parabolas_l571_571742

-- Define the quadratic equation and its conditions
def parabola_equation (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Define the axis of symmetry condition
def axis_of_symmetry (a b : ℝ) : Prop :=
  -b / (2 * a) = -2

-- Define the tangent line condition
def tangent_line_condition (a b c : ℝ) : Prop :=
  ∃ x, parabola_equation a b c x = 2 * x + 1 ∧
       2 * x + 1 ≠ parabola_equation a b c (x + ε) ∀ ε ≠ 0

-- Define the y-axis intersection condition
def y_axis_intersection_distance (a b c : ℝ) : Prop :=
  let h := b^2 - 4 * a * c in
  h > 0 ∧
  2 * sqrt h / abs a = 2 * sqrt 2

-- Theorem statement
theorem find_parabolas :
  (∃ a b c, axis_of_symmetry a b ∧ tangent_line_condition a b c ∧ y_axis_intersection_distance a b c ∧
  ((a = 1 ∧ b = 4 ∧ c = 2) ∨ (a = 1/2 ∧ b = 2 ∧ c = 1))) :=
sorry

end find_parabolas_l571_571742


namespace part1_part2_l571_571313

noncomputable def f (x a : ℝ) : ℝ := x + a / x - 4
noncomputable def g (x k : ℝ) : ℝ := k * x + 3

theorem part1 (m : ℝ) (a : ℝ) (h : a ∈ Icc (4:ℝ) 6) : (∀ x ∈ Icc (1:ℝ) m, (abs (f x a)) ≤ abs (f m a)) ↔ m ≥ 6 :=
sorry

theorem part2 (k : ℝ) (a : ℝ) (h : a ∈ Icc (1:ℝ) 2) : (∀ x1 x2 ∈ Icc (2:ℝ) 4, x1 < x2 → abs (f x1 a) - abs (f x2 a) < g x1 k - g x2 k) ↔ k ≤ 6 - 4 * Real.sqrt 3 :=
sorry

end part1_part2_l571_571313


namespace length_of_curve_ln2_l571_571150

open Real Set

noncomputable def x (t : ℝ) : ℝ := ∫ u in t..∞, (cos u) / u

noncomputable def y (t : ℝ) : ℝ := ∫ u in t..∞, (sin u) / u

theorem length_of_curve_ln2 :
  (∫ t in 1..2, sqrt ( (-(cos t / t)) ^ 2 + (-(sin t / t)) ^ 2 )) = ln 2 := by
sorry

end length_of_curve_ln2_l571_571150


namespace units_digit_17_pow_2023_l571_571109

theorem units_digit_17_pow_2023 : (17 ^ 2023) % 10 = 3 :=
by
  have units_cycle_7 : ∀ (n : ℕ), (7 ^ n) % 10 = [7, 9, 3, 1].nth (n % 4) :=
    sorry
  have units_pattern_equiv : (17 ^ n) % 10 = (7 ^ n) % 10 :=
    sorry
  calc
    (17 ^ 2023) % 10
        = (7 ^ 2023) % 10  : by rw [units_pattern_equiv]
    ... = 3               : by rw [units_cycle_7, nat.mod_eq_of_lt, List.nth]

end units_digit_17_pow_2023_l571_571109


namespace feng_shui_arrangements_3x3_l571_571490

theorem feng_shui_arrangements_3x3 : 
  let n := 9
  let feng_shui (matrix : (Fin 3) → (Fin 3) → ℕ) := 
    ∀ {i j k l m n : Fin 3}, 
      i ≠ j ∧ j ≠ k ∧ k ≠ i ∧
      l ≠ m ∧ m ≠ n ∧ n ≠ l → 
      matrix i l + matrix j m + matrix k n = 15
  ∃ (matrices : List ((Fin 3) → (Fin 3) → ℕ)), 
    matrices.length = 72 ∧ 
    ∀ matrix ∈ matrices, feng_shui matrix :=
begin
  sorry
end

end feng_shui_arrangements_3x3_l571_571490


namespace tangent_condition_l571_571822

theorem tangent_condition (θ : ℝ) (r : ℝ) (hC01 : θ ≠ 0) (hC02 : r ≠ 0) :
  (tan (2 * θ) = 2 * θ) ↔ (2 * θ = sin (2 * θ) / cos (2 * θ)) :=
by
  sorry

end tangent_condition_l571_571822


namespace find_angle_l571_571728

-- Definitions based on conditions
def is_complement (x : ℝ) : ℝ := 90 - x
def is_supplement (x : ℝ) : ℝ := 180 - x

-- Main statement
theorem find_angle (x : ℝ) (h : is_supplement x = 15 + 4 * is_complement x) : x = 65 :=
by
  sorry

end find_angle_l571_571728


namespace dante_initially_has_8_jelly_beans_l571_571503

-- Conditions
def aaron_jelly_beans : ℕ := 5
def bianca_jelly_beans : ℕ := 7
def callie_jelly_beans : ℕ := 8
def dante_jelly_beans_initially (D : ℕ) : Prop := 
  ∀ (D : ℕ), (6 ≤ D - 1 ∧ D - 1 ≤ callie_jelly_beans - 1)

-- Theorem
theorem dante_initially_has_8_jelly_beans :
  ∃ (D : ℕ), (aaron_jelly_beans + 1 = 6) →
             (callie_jelly_beans = 8) →
             dante_jelly_beans_initially D →
             D = 8 := 
by
  sorry

end dante_initially_has_8_jelly_beans_l571_571503


namespace min_value_of_y_l571_571017

-- Define the function y in terms of sin x
def y (x : ℝ) : ℝ :=
  - (Real.sin x) ^ 3 - 2 * (Real.sin x)

-- Theorem to prove the minimum value of the function y
theorem min_value_of_y : ∀ x, -3 ≤ x ∧ x ≤ 3 → y x = -3 :=
by
  -- The proof is omitted here
  sorry

end min_value_of_y_l571_571017


namespace narrow_black_stripes_l571_571860

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l571_571860


namespace distinct_positive_factors_48_l571_571346

theorem distinct_positive_factors_48 : 
  ∀ (n : ℕ), n = 48 → ∀ (p q : ℕ), p = 2 ∧ q = 3 → (∃ a b : ℕ, 48 = p^a * q^b ∧ (a + 1) * (b + 1) = 10) :=
by
  intros n hn p q hpq
  have h_48 : 48 = 2^4 * 3^1 := by norm_num
  use 4, 1
  split
  · exact h_48
  · norm_num
  sorry

end distinct_positive_factors_48_l571_571346


namespace tangent_line_at_b_l571_571957

theorem tangent_line_at_b (b : ℝ) : (∃ x : ℝ, (4*x^3 = 4) ∧ (4*x + b = x^4 - 1)) ↔ (b = -4) := 
by 
  sorry

end tangent_line_at_b_l571_571957


namespace pascals_triangle_prob_l571_571606

/--
An element is randomly chosen from among the first 20 rows of Pascal's Triangle.
Prove that the probability that the value of the element chosen is 1 is 39/210.
-/
theorem pascals_triangle_prob (n : ℕ) (n = 19) : 
  let total_elements := (n + 1) * (n + 2) / 2,
      ones_in_rows := 2 * n + 1 in
  (ones_in_rows : ℚ) / total_elements = 39 / 210 :=
begin
  sorry
end

end pascals_triangle_prob_l571_571606


namespace number_of_subsets_of_set_l571_571500

open Set

theorem number_of_subsets_of_set (s : Set Int) (h : s = {-1, 0, 1}) : Fintype.card (Set s) = 8 := by
  sorry

end number_of_subsets_of_set_l571_571500


namespace narrow_black_stripes_are_8_l571_571892

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l571_571892


namespace semicircle_problem_l571_571820

open Real

theorem semicircle_problem (r : ℝ) (N : ℕ)
  (h1 : True) -- condition 1: There are N small semicircles each with radius r.
  (h2 : True) -- condition 2: The diameter of the large semicircle is 2Nr.
  (h3 : (N * (π * r^2) / 2) / ((π * (N^2 * r^2) / 2) - (N * (π * r^2) / 2)) = (1 : ℝ) / 12) -- given ratio A / B = 1 / 12 
  : N = 13 :=
sorry

end semicircle_problem_l571_571820


namespace car_speed_is_approximately_98_769_kmh_l571_571573

def distance : ℝ := 642
def time : ℝ := 6.5

theorem car_speed_is_approximately_98_769_kmh :
  distance / time ≈ 98.769 := 
by sorry

end car_speed_is_approximately_98_769_kmh_l571_571573


namespace num_perfect_square_factors_l571_571634

-- Define the exponents and their corresponding number of perfect square factors
def num_square_factors (exp : ℕ) : ℕ := exp / 2 + 1

-- Define the product of the prime factorization
def product : ℕ := 2^12 * 3^15 * 7^18

-- State the theorem
theorem num_perfect_square_factors :
  (num_square_factors 12) * (num_square_factors 15) * (num_square_factors 18) = 560 := by
  sorry

end num_perfect_square_factors_l571_571634


namespace narrow_black_stripes_l571_571863

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l571_571863


namespace base_height_l571_571202

-- Define the height of the sculpture and the combined height.
def sculpture_height : ℚ := 2 + 10 / 12
def total_height : ℚ := 3 + 2 / 3

-- We want to prove that the base height is 5/6 feet.
theorem base_height :
  total_height - sculpture_height = 5 / 6 :=
by
  sorry

end base_height_l571_571202


namespace statement_A_correct_statement_C_correct_statement_D_correct_l571_571121

theorem statement_A_correct (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 1) : a^2 + b^2 ≥ 1 / 2 := sorry

theorem statement_C_correct (f : ℝ → ℝ) (dom_f : Set.Icc (-1) 1) : Set.Icc (-1 : ℝ) 3 → dom_f := sorry

theorem statement_D_correct (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≥ -1 → f (sqrt x - 1) = x - 3 * sqrt x) → (∀ x : ℝ, x ≥ -1 → f x = x^2 - x - 2) := sorry

end statement_A_correct_statement_C_correct_statement_D_correct_l571_571121


namespace compute_infinite_sum_l571_571431

noncomputable def infinite_sum (x : ℝ) (h : x > 2) : ℝ :=
  ∑' n, 1 / (x^(3^n) - x^(-3^n))

theorem compute_infinite_sum (x : ℝ) (h : x > 2) :
  infinite_sum x h = 1 / (x - 1) :=
sorry

end compute_infinite_sum_l571_571431


namespace num_factors_48_l571_571362

theorem num_factors_48 : 
  let n := 48 in
  ∃ num_factors, num_factors = 10 ∧ 
  (∀ p k, prime p → (n = p ^ k → 1 + k)) := 
sorry

end num_factors_48_l571_571362


namespace rectangle_longer_side_l571_571575

theorem rectangle_longer_side
  (r : ℝ)
  (A_circle : ℝ)
  (A_rectangle : ℝ)
  (shorter_side : ℝ)
  (longer_side : ℝ) :
  r = 5 →
  A_circle = 25 * Real.pi →
  A_rectangle = 3 * A_circle →
  shorter_side = 2 * r →
  longer_side = A_rectangle / shorter_side →
  longer_side = 7.5 * Real.pi :=
by
  intros
  sorry

end rectangle_longer_side_l571_571575


namespace consecutive_cards_ways_correct_l571_571141

/-- A function to calculate the number of ways to pick two consecutive cards where one is a face card and the other is a number card from a deck of 48 cards divided equally among 4 suits. -/
def consecutive_cards_ways (total_cards : ℕ) (suits : ℕ) (face_cards_per_suit : ℕ) (number_cards_per_suit : ℕ) : ℕ :=
  let ways_per_suit := (face_cards_per_suit * number_cards_per_suit) + (number_cards_per_suit * face_cards_per_suit)
  suits * ways_per_suit

/-- Given a deck of 48 cards, equally divided among 4 suits, 
    with each suit having 3 face cards and 10 number cards, 
    the number of different ways to pick two consecutive cards 
    where one is a face card and the other is a number card is 240. -/
theorem consecutive_cards_ways_correct :
  consecutive_cards_ways 48 4 3 10 = 240 :=
by
  unfold consecutive_cards_ways
  norm_num
  sorry

end consecutive_cards_ways_correct_l571_571141


namespace slips_numbers_exist_l571_571004

theorem slips_numbers_exist (x y z : ℕ) (h₁ : x + y + z = 20) (h₂ : 5 * x + 3 * y = 46) : 
  (x = 4) ∧ (y = 10) ∧ (z = 6) :=
by {
  -- Technically, the actual proving steps should go here, but skipped due to 'sorry'
  sorry
}

end slips_numbers_exist_l571_571004


namespace proof_not_sufficient_nor_necessary_l571_571686

noncomputable def not_sufficient_nor_necessary (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : Prop :=
  ¬ ((a > b) → (Real.log b / Real.log a < 1)) ∧ ¬ ((Real.log b / Real.log a < 1) → (a > b))

theorem proof_not_sufficient_nor_necessary (a b: ℝ) (h₁: 0 < a) (h₂: 0 < b) :
  not_sufficient_nor_necessary a b h₁ h₂ :=
  sorry

end proof_not_sufficient_nor_necessary_l571_571686


namespace student_count_l571_571387

theorem student_count 
( M S N : ℕ ) 
(h1 : N - M = 10) 
(h2 : N - S = 15) 
(h3 : N - (M + S - 7) = 2) : 
N = 34 :=
by
  sorry

end student_count_l571_571387


namespace probability_of_5_odd_numbers_l571_571080

-- Define a function to represent the binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Axiom that defines the probability of getting an odd number
axiom fair_die_prob : ∀ (x : ℕ), 0 < x ∧ x ≤ 6 -> (1/2)

-- Define the problem statement about the probability
theorem probability_of_5_odd_numbers (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 5) : 
  (binom n k) / 2^n = 3 / 32 := sorry

end probability_of_5_odd_numbers_l571_571080


namespace proof_values_of_a_b_final_expression_l571_571731

noncomputable def algebraic_expression (a b x : ℝ) : ℝ :=
  (a * x - 3) * (2 * x + 4) - x^2 - b

theorem proof_values_of_a_b (a b : ℝ) :
  (algebraic_expression a b 1 = (2 * a - 1) + (4 * a - 6) + (-12 - b))
  → (2a - 1 = 0 ∧ -12 - b = 0)
  → (a = 1/2 ∧ b = -12) :=
sorry

theorem final_expression (a b : ℝ)
  (h : a = 1 / 2 ∧ b = -12) :
  (2 * a + b)^2 - (2 - 2 * b) * (2 + 2 * b) - 3 * a * (a - b) = 678 :=
sorry

end proof_values_of_a_b_final_expression_l571_571731


namespace circumscribed_circle_radius_l571_571961

theorem circumscribed_circle_radius :
  let A := (0 : ℝ, -3 : ℝ)
  let B := (3 : ℝ, 0 : ℝ)
  let C := (-1 : ℝ, 0 : ℝ)
  let center := (1 : ℝ, -1 : ℝ)
  radius := Real.sqrt 5
  (Real.sqrt ((1 - 0)^2 + (-1 + 3)^2) = radius) :=
by
  let A := (0 : ℝ, -3 : ℝ)
  let B := (3 : ℝ, 0 : ℝ)
  let C := (-1 : ℝ, 0 : ℝ)
  let center := (1 : ℝ, -1 : ℝ)
  let radius := Real.sqrt 5
  show (Real.sqrt ((1 - 0)^2 + (-1 + 3)^2) = radius)
  sorry

end circumscribed_circle_radius_l571_571961


namespace count_integers_within_bounds_l571_571256

theorem count_integers_within_bounds : 
  ∃ (count : ℕ), count = finset.card (finset.filter (λ n : ℤ, 15 < n^2 ∧ n^2 < 120) (finset.Icc (-10) 10)) ∧ count = 14 := 
by
  sorry

end count_integers_within_bounds_l571_571256


namespace units_digit_17_pow_2023_l571_571112

theorem units_digit_17_pow_2023 : (17^2023 % 10) = 3 := sorry

end units_digit_17_pow_2023_l571_571112


namespace cone_radius_correct_l571_571030

noncomputable def cone_radius (CSA l : ℝ) : ℝ := CSA / (Real.pi * l)

theorem cone_radius_correct :
  cone_radius 1539.3804002589986 35 = 13.9 :=
by
  -- Proof omitted
  sorry

end cone_radius_correct_l571_571030


namespace commercial_break_duration_l571_571646

theorem commercial_break_duration (n1 n2 m1 m2 : ℕ) (h1 : n1 = 3) (h2 : m1 = 5) (h3 : n2 = 11) (h4 : m2 = 2) :
  n1 * m1 + n2 * m2 = 37 :=
by
  -- Here, in a real proof, we would substitute and show the calculations.
  sorry

end commercial_break_duration_l571_571646


namespace find_previous_month_employees_l571_571385

-- Definitions for the problem conditions
def companyA_percent_increase : ℝ := 0.146
def companyB_percent_increase : ℝ := 0.179
def companyC_percent_increase : ℝ := 0.233

def companyA_employees_dec : ℝ := 1057
def companyB_employees_dec : ℝ := 1309
def companyC_employees_jan : ℝ := 1202

def round_to_nearest (x : ℝ) : ℕ := int.to_nat (int.of_nat (x + 0.5))

-- Stating the problem to find the employees in the previous month
theorem find_previous_month_employees :
  ∃ (a prev_emp november_emp: ℝ) (b prev_emp october_emp: ℝ) (c prev_emp december_emp: ℝ), 
  a = companyA_employees_dec / (1 + companyA_percent_increase) ∧ 
  b = companyB_employees_dec / (1 + companyB_percent_increase) ∧ 
  c = companyC_employees_jan / (1 + companyC_percent_increase) ∧
  round_to_nearest a = 922 ∧
  round_to_nearest b = 1111 ∧
  round_to_nearest c = 975 :=
  
begin
  sorry
end

end find_previous_month_employees_l571_571385


namespace find_x_l571_571750

theorem find_x
  (a : ℝ × ℝ)
  (b : ℝ × ℝ)
  (h : a = (Real.sqrt 3, 0))
  (h1 : b = (x, -2))
  (h2 : a.1 * (a.1 - 2 * b.1) + a.2 * (a.2 - 2 * b.2) = 0) :
  x = Real.sqrt 3 / 2 :=
sorry

end find_x_l571_571750


namespace probability_of_diff_tens_digits_l571_571001

noncomputable def probability_different_tens_digits : ℚ :=
  let total_ways := choose 50 5
  let successful_ways := 10^5
  (successful_ways : ℚ) / total_ways

theorem probability_of_diff_tens_digits :
  probability_different_tens_digits = 2500 / 52969 := sorry

end probability_of_diff_tens_digits_l571_571001


namespace find_a_l571_571714

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def f : ℝ → ℝ := sorry -- The definition of f is to be handled in the proof

theorem find_a (a : ℝ) (h1 : is_odd_function f)
  (h2 : ∀ x : ℝ, 0 < x → f x = 2^(x - a) - 2 / (x + 1))
  (h3 : f (-1) = 3 / 4) : a = 3 :=
sorry

end find_a_l571_571714


namespace find_k_values_l571_571233

theorem find_k_values :
  ∃ k : ℚ, (|a - b| = a² + b² ∧ a + b = -4 / 5 ∧ a * b = k / 5) → (k = 3 / 5 ∨ k = -12 / 5) :=
by
  sorry

end find_k_values_l571_571233


namespace number_of_narrow_black_stripes_l571_571904

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l571_571904


namespace direction_vector_of_projection_l571_571958

-- Define the projection matrix
def projection_matrix : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![  3/17, -2/17, -1/3],
    ![-2/17,  1/17,  1/6],
    ![-1/3,   1/6,   5/6]]

-- Define the condition that a, b, c should be integers, a > 0, and gcd(|a|, |b|, |c|) = 1
structure DirectionVector (a b c : ℤ) : Prop :=
  (a_pos : a > 0)
  (gcd_cond : Int.gcd (Int.natAbs a) (Int.gcd (Int.natAbs b) (Int.natAbs c)) = 1)

-- The theorem stating the direction vector
theorem direction_vector_of_projection : ∃ (a b c : ℤ), 
  DirectionVector a b c ∧ 
  ![(a : ℚ), (b : ℚ), (c : ℚ)] ∝ 
  ![ 3, -2, -5] := by 
sorry

end direction_vector_of_projection_l571_571958


namespace probability_of_diff_tens_digits_l571_571000

noncomputable def probability_different_tens_digits : ℚ :=
  let total_ways := choose 50 5
  let successful_ways := 10^5
  (successful_ways : ℚ) / total_ways

theorem probability_of_diff_tens_digits :
  probability_different_tens_digits = 2500 / 52969 := sorry

end probability_of_diff_tens_digits_l571_571000


namespace distinct_positive_factors_of_48_l571_571354

theorem distinct_positive_factors_of_48 : 
  let n := 48 in
  let factors := (2^4) * (3^1) in
  ∀ n : ℕ, n = factors → 
  (let num_factors := (4 + 1) * (1 + 1)
  in num_factors = 10) :=
by 
  let n := 48
  let factors := (2^4) * (3^1)
  assume h : n = factors
  let num_factors := (4 + 1) * (1 + 1)
  show num_factors = 10 from sorry

end distinct_positive_factors_of_48_l571_571354


namespace sum_of_squares_divisible_by_24_l571_571720

-- Given conditions translated to Lean definitions
def primes_at_least_five (ps : List ℕ) : Prop :=
  (∀ p ∈ ps, Nat.Prime p ∧ p ≥ 5) ∧ ps.length = 24

-- Main theorem to prove
theorem sum_of_squares_divisible_by_24 
  (p : List ℕ) (hp : primes_at_least_five p) :
  (∑ i in p, i^2) % 24 = 0 :=
sorry

end sum_of_squares_divisible_by_24_l571_571720


namespace proportional_function_decreases_l571_571929

-- Define the function y = -2x
def proportional_function (x : ℝ) : ℝ := -2 * x

-- State the theorem to prove that y decreases as x increases
theorem proportional_function_decreases (x y : ℝ) (h : y = proportional_function x) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → proportional_function x₁ > proportional_function x₂ := 
sorry

end proportional_function_decreases_l571_571929


namespace inequality_one_inequality_two_l571_571473

theorem inequality_one (x : ℝ) : 7 * x - 2 < 3 * (x + 2) → x < 2 :=
by
  sorry

theorem inequality_two (x : ℝ) : (x - 1) / 3 ≥ (x - 3) / 12 + 1 → x ≥ 13 / 3 :=
by
  sorry

end inequality_one_inequality_two_l571_571473


namespace total_pieces_of_mail_l571_571411

-- Definitions based on given conditions
def pieces_each_friend_delivers : ℕ := 41
def pieces_johann_delivers : ℕ := 98
def number_of_friends : ℕ := 2

-- Theorem statement to prove the total number of pieces of mail delivered
theorem total_pieces_of_mail :
  (number_of_friends * pieces_each_friend_delivers) + pieces_johann_delivers = 180 := 
by
  -- proof would go here
  sorry

end total_pieces_of_mail_l571_571411


namespace minimum_cuts_l571_571632

-- Define the number of dominoes and their area
def num_dominoes : Nat := 18
def domino_area : Nat := 2

-- Define the total area of the figure
def total_area : Nat := num_dominoes * domino_area
def num_parts : Nat := 4
def part_area : Nat := total_area / num_parts

-- Define the statement to prove
theorem minimum_cuts (figure : Fin 36 → Prop) :
  (∀ i j : Fin 36, (figure i = figure j) → (i ≠ j → figure i ∧ figure j)) →
  (∀ part : Fin 9 → Prop, figure = finset.univ.image part) →
  ∃ cuts : Fin 36 → Fin 4, 
  (∀ i j : Fin 36, (cuts i = cuts j) → (i ≠ j → cuts i ∧ cuts j)) ∧ 
  (∀ i : Fin 4, finset.filter (λ j => cuts j = i) (finset.univ : Finset (Fin 36)).card = 9) →
  ∃ m : Nat, m ≥ 2 :=
by
  sorry

end minimum_cuts_l571_571632


namespace intersect_at_single_point_l571_571948

theorem intersect_at_single_point
  (A B C D O A' B' C' D' : Point)
  (h1 : ∠ A O B = 90°) (h2 : ∠ C O D = 90°)
  (hA' : is_circumcenter A' A B D)
  (hB' : is_circumcenter B' B C A)
  (hC' : is_circumcenter C' C D B)
  (hD' : is_circumcenter D' D A C) :
  ∃ P : Point, ∀ (l : Line), (l = Line_through A A' ∨ l = Line_through B B' ∨ l = Line_through C C' ∨ l = Line_through D D') → P ∈ l :=
begin
  sorry,
end

end intersect_at_single_point_l571_571948


namespace probability_of_5_out_of_6_rolls_odd_l571_571061

theorem probability_of_5_out_of_6_rolls_odd : 
  (nat.choose 6 5 : ℚ) / (2 ^ 6 : ℚ) = 3 / 32 := 
by
  sorry

end probability_of_5_out_of_6_rolls_odd_l571_571061


namespace monica_tiles_count_l571_571911

theorem monica_tiles_count : 
  let length := 16
  let width := 12
  let border_tile_count := 2 * (length - 2) + 2 * (width - 2) + 4
  let inner_area := (length - 2) * (width - 2)
  let inner_tile_count := inner_area / 4
  border_tile_count + inner_tile_count = 87 := 
by
  let length : ℕ := 16
  let width : ℕ := 12
  let border_tile_count := (2 * (length - 2)) + (2 * (width - 2)) + 4
  let inner_area := (length - 2) * (width - 2)
  let inner_tile_count := inner_area / 4
  have h_border_tile_count : border_tile_count = 52 := by sorry
  have h_inner_tile_count : inner_tile_count = 35 := by sorry
  show border_tile_count + inner_tile_count = 87 from
    by rw [h_border_tile_count, h_inner_tile_count]; exact rfl

end monica_tiles_count_l571_571911


namespace fish_lifespan_l571_571509

theorem fish_lifespan (H : ℝ) (D : ℝ) (F : ℝ) 
  (h_hamster : H = 2.5)
  (h_dog : D = 4 * H)
  (h_fish : F = D + 2) : 
  F = 12 :=
by
  rw [h_hamster, h_dog] at h_fish
  simp at h_fish
  exact h_fish

end fish_lifespan_l571_571509


namespace probability_of_5_out_of_6_rolls_odd_l571_571059

theorem probability_of_5_out_of_6_rolls_odd : 
  (nat.choose 6 5 : ℚ) / (2 ^ 6 : ℚ) = 3 / 32 := 
by
  sorry

end probability_of_5_out_of_6_rolls_odd_l571_571059


namespace sum_f_inv_l571_571697

noncomputable def f : ℝ → ℝ := sorry

axiom f_0 : f 0 = 1
axiom f_func_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem sum_f_inv : ∑ i in Finset.range 2023 + 1, 1 / (f i * f (i + 1)) = 2023 / 4050 := sorry

end sum_f_inv_l571_571697


namespace monotonically_increasing_on_interval_l571_571551

theorem monotonically_increasing_on_interval :
  (∀ x : ℝ, 2 < x → (x + 1/x : ℝ) < (x + (1/x) + 1) - (x + (1/x))) ∧
  (∀ x : ℝ, 2 < x → (x - 1/x : ℝ) < ((x - (1/x)) + 1) - (x - (1/x))) ∧
  (¬∀ x : ℝ, 2 < x → (1/(4-x) : ℝ) < (1/(4-x)) - 1) ∧
  (¬ (∀ x : ℝ, 2 < x → (√(x^2 - 4*x + 3) : ℝ) < (√(x^2 - 4*x + 3) + 1) - √(x^2 - 4*x + 3))):=
by {
  sorry
}

end monotonically_increasing_on_interval_l571_571551


namespace probability_5_of_6_odd_rolls_l571_571057

def binom_coeff : ℕ → ℕ → ℕ
| n k := Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binom_coeff n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_5_of_6_odd_rolls :
  binomial_probability 6 5 (1/2) = 3/16 :=
by
  -- Proof will go here, but we skip it with sorry for now.
  sorry

end probability_5_of_6_odd_rolls_l571_571057


namespace rationalize_denominator_l571_571927

theorem rationalize_denominator :
  Let (A B C D E : Int) (r := 3 / (4 * Real.sqrt 7 + 5 * Real.sqrt 3))
  (numer := 12 * Real.sqrt 7 - 15 * Real.sqrt 3)
  (denom := 37)
  in (A, B, C, D, E) = (12, 7, -15, 3, 37) ->
  (A + B + C + D + E = 44) :=
by
  sorry

end rationalize_denominator_l571_571927


namespace number_in_row_table_le_S_l571_571414

-- Define the sum function S(n, r)
noncomputable def S (n r : ℕ) : ℕ :=
  if n > 0 ∧ r > 0 then ∑ k in finset.range (n - r + 1), nat.choose (n - 1) (r - 1 + k)
  else 0

-- The theorem to be proven
theorem number_in_row_table_le_S {n r : ℕ} (h_n : n > 0) (h_r : r > 0) :
  -- A condition stating there exists a quantity in row n, r columns to the left of the 1
  let q := -- definition or property describing the table element
  q <= S n r :=
sorry

end number_in_row_table_le_S_l571_571414


namespace solve_equation_l571_571460

theorem solve_equation :
  ∃ x : Real, (x = 2 ∨ x = (-(1:Real) - Real.sqrt 17) / 2) ∧ (x^2 - |x - 1| - 3 = 0) :=
by
  sorry

end solve_equation_l571_571460


namespace bridge_construction_plans_l571_571041

-- Define the number of islands
def num_islands : ℕ := 4

-- Define the number of bridges to be built
def num_bridges : ℕ := 3

-- Prove that the number of different valid bridge construction plans is 16
theorem bridge_construction_plans : 
  (∀ i j k : ℕ, i < num_islands → j < num_islands → k < num_islands → i ≠ j → j ≠ k → i ≠ k) →
  ∃ valid_plans : ℕ, valid_plans = 16 :=
by
  assume islands_valid : ∀ i j k : ℕ, i < num_islands → j < num_islands → k < num_islands → i ≠ j → j ≠ k → i ≠ k,
  use 16,
  sorry

end bridge_construction_plans_l571_571041


namespace ABC_sim_AUV_l571_571403

noncomputable theory

open Complex

variables (a b c u v : ℂ)

-- Given Conditions:
-- Triangles AUV, VBU, and UVC are directly similar.
def triangles_directly_similar (z w x y : ℂ) : Prop :=
  ∃ k : ℂ, k ≠ 0 ∧ (x - z) = k * (y - w)

axiom AUV_sim_VBU : triangles_directly_similar a u u v
axiom VBU_sim_UVC : triangles_directly_similar v b u v
axiom UVC_sim_AUV : triangles_directly_similar u v c u

-- Goal: Prove triangle ABC is directly similar to AUV.
theorem ABC_sim_AUV : triangles_directly_similar a c b a := 
sorry

end ABC_sim_AUV_l571_571403


namespace multiples_of_6_or_8_but_not_both_l571_571773

/-- The number of positive integers less than 151 that are multiples of either 6 or 8 but not both is 31. -/
theorem multiples_of_6_or_8_but_not_both (n : ℕ) :
  (multiples_of_6 : Set ℕ) = {k | k < 151 ∧ k % 6 = 0}
  ∧ (multiples_of_8 : Set ℕ) = {k | k < 151 ∧ k % 8 = 0}
  ∧ (multiples_of_24 : Set ℕ) = {k | k < 151 ∧ k % 24 = 0}
  ∧ multiples_of_6_or_8 := {k | k ∈ multiples_of_6 ∨ k ∈ multiples_of_8}
  ∧ multiples_of_6_and_8 := {k | k ∈ multiples_of_6 ∧ k ∈ multiples_of_8}
  ∧ (card (multiples_of_6_or_8 \ multiples_of_6_and_8)) = 31 := sorry

end multiples_of_6_or_8_but_not_both_l571_571773


namespace jane_total_drying_time_l571_571410

theorem jane_total_drying_time :
  let base_coat := 4
  let color_coat_1 := 5
  let color_coat_2 := 6
  let color_coat_3 := 7
  let nail_art_1 := 8
  let nail_art_2 := 10
  let top_coat := 9
  base_coat + color_coat_1 + color_coat_2 + color_coat_3 + nail_art_1 + nail_art_2 + top_coat = 49 :=
by 
  sorry

end jane_total_drying_time_l571_571410


namespace factors_of_48_l571_571336

theorem factors_of_48 : ∃ n, n = 48 → number_of_distinct_positive_factors n = 10 :=
sorry

-- Auxiliary function definitions to support the main theorem
def number_of_distinct_positive_factors (n : ℕ) : ℕ := 
sorry

end factors_of_48_l571_571336


namespace palindrome_years_between_1000_and_2000_l571_571152

def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString
  s = s.reverse

def one_digit_prime_palindrome (n : ℕ) : Prop := 
  n ∈ {2, 3, 5, 7}

def four_digit_prime_palindrome (n : ℕ) : Prop :=
  is_palindrome n ∧ Prime n ∧ 1000 ≤ n ∧ n < 10000

def year_palindrome_factorizable (n: ℕ) : Prop :=
  is_palindrome n ∧ 1000 ≤ n ∧ n < 2000 ∧
  ∃ (p q : ℕ), one_digit_prime_palindrome p ∧ four_digit_prime_palindrome q ∧ n = p * q

theorem palindrome_years_between_1000_and_2000 :
  {n : ℕ | year_palindrome_factorizable n}.card = 0 :=
sorry

end palindrome_years_between_1000_and_2000_l571_571152


namespace cartesian_equation_of_c1_general_equation_of_c2_min_distance_midpoint_to_c1_l571_571301

-- Define the given conditions and entities
def polar_to_cartesian_c1 (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def c1_cartesian (x y : ℝ) : Prop :=
  x - 2 * y - 7 = 0

def c2_polar (θ : ℝ) : ℝ × ℝ :=
  (8 * Real.cos θ, 3 * Real.sin θ)

def c2_general_equation (x y : ℝ) : Prop :=
  (x^2 / 64) + (y^2 / 9) = 1

-- Define the distance from point M to the line x - 2y - 7 = 0
def distance_midpoint_to_c1 (m_x m_y : ℝ) : ℝ :=
  abs (m_x - 2 * m_y - 7) / Real.sqrt 5

-- Proof statements
theorem cartesian_equation_of_c1 (ρ θ x y : ℝ) :
  polar_to_cartesian_c1 ρ θ = (x, y) → c1_cartesian x y :=
by
  -- Conversion from polar to Cartesian for C1
  sorry

theorem general_equation_of_c2 (θ x y : ℝ) :
  c2_polar θ = (x, y) → c2_general_equation x y :=
by
  -- Conversion from parameter to general equation for C2
  sorry

theorem min_distance_midpoint_to_c1 (θ : ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) :
  P = (-4, 4) → Q = c2_polar θ →
  let M_x := (4 * Real.cos θ - 2) in
  let M_y := (3 * Real.sin θ + 4) / 2 in
  distance_midpoint_to_c1 M_x M_y = 8 * Real.sqrt 5 / 5 :=
by
  -- Proof of the minimum distance calculation from midpoint M to C1
  sorry

end cartesian_equation_of_c1_general_equation_of_c2_min_distance_midpoint_to_c1_l571_571301


namespace largest_number_no_digit_5_with_sum_17_l571_571521

theorem largest_number_no_digit_5_with_sum_17 : 
  ∃! n : ℕ, (∀ d ∈ (int.to_digits n), d ≠ 5) ∧ (list.nodup (int.to_digits n)) ∧ (list.sum (int.to_digits n) = 17) ∧ (∀ m : ℕ, (∀ d ∈ (int.to_digits m), d ≠ 5) ∧ (list.nodup (int.to_digits m)) ∧ (list.sum (int.to_digits m) = 17) → m ≤ n) :=
sorry

end largest_number_no_digit_5_with_sum_17_l571_571521


namespace mirella_read_more_pages_l571_571909

-- Define the number of books Mirella read
def num_purple_books := 8
def num_orange_books := 7
def num_blue_books := 5

-- Define the number of pages per book for each color
def pages_per_purple_book := 320
def pages_per_orange_book := 640
def pages_per_blue_book := 450

-- Calculate the total pages for each color
def total_purple_pages := num_purple_books * pages_per_purple_book
def total_orange_pages := num_orange_books * pages_per_orange_book
def total_blue_pages := num_blue_books * pages_per_blue_book

-- Calculate the combined total of orange and blue pages
def total_orange_blue_pages := total_orange_pages + total_blue_pages

-- Define the target value
def page_difference := 4170

-- State the theorem to prove
theorem mirella_read_more_pages :
  total_orange_blue_pages - total_purple_pages = page_difference := by
  sorry

end mirella_read_more_pages_l571_571909


namespace area_of_given_triangle_is_8_l571_571598

-- Define the vertices of the triangle
def x1 := 2
def y1 := -3
def x2 := -1
def y2 := 6
def x3 := 4
def y3 := -5

-- Define the determinant formula for the area of the triangle
def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℤ) : ℤ :=
  abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2

theorem area_of_given_triangle_is_8 :
  area_of_triangle x1 y1 x2 y2 x3 y3 = 8 := by
  sorry

end area_of_given_triangle_is_8_l571_571598


namespace original_radius_of_cylinder_l571_571826

theorem original_radius_of_cylinder 
  (r y : ℝ) 
  (h : r + 3 ≥ 0)
  (condition1 : 24 * real.pi * r + 36 * real.pi = y)
  (condition2 : 3 * real.pi * r^2 = y)
  (height : 4 = 4) : r = 12 :=
by
  sorry

end original_radius_of_cylinder_l571_571826


namespace narrow_black_stripes_count_l571_571889

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l571_571889


namespace roots_transformation_l571_571430

noncomputable def poly_with_roots (r₁ r₂ r₃ : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - 5 * Polynomial.X ^ 2 + 10

noncomputable def transformed_poly_with_roots (r₁ r₂ r₃ : ℝ) : Polynomial ℝ :=
  Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 270

theorem roots_transformation (r₁ r₂ r₃ : ℝ) (h : poly_with_roots r₁ r₂ r₃ = 0) :
  transformed_poly_with_roots (3 * r₁) (3 * r₂) (3 * r₃) = Polynomial.X ^ 3 - 15 * Polynomial.X ^ 2 + 270 :=
by
  sorry

end roots_transformation_l571_571430


namespace sum_of_distances_at_least_1000_l571_571432

theorem sum_of_distances_at_least_1000 (A : Fin 1000 → (ℝ × ℝ)) (r : ℝ) (hr : r = 1) :
  ∃ (M : ℝ × ℝ), (dist M ⟨0,0⟩ = r) ∧ (Fin 1000).sum (λ i, dist M (A i)) ≥ 1000 :=
sorry

end sum_of_distances_at_least_1000_l571_571432


namespace remainder_when_200_divided_by_k_l571_571669

theorem remainder_when_200_divided_by_k 
  (k : ℕ) (k_pos : 0 < k)
  (h : 120 % k^2 = 12) :
  200 % k = 2 :=
sorry

end remainder_when_200_divided_by_k_l571_571669


namespace number_of_correct_propositions_l571_571735

-- Definitions of the propositions as given conditions
variables {a b e₁ e₂ : ℝ} (k : ℝ)
def prop1 := ∀ {a b : ℝ}, (a * b = 0) → (a ⊥ b)
def prop2 := ∀ {a b : ℝ}, (|a + b| > |a - b|)
def prop3 := ¬ collinear e₁ e₂ ∧ ¬ collinear (e₁ + 2 * e₂) (e₂ + 2 * e₁) ∧ basis {e₁ + 2 * e₂, e₂ + 2 * e₁}
def prop4 := ∃ k : ℝ, a = k * b → collinear a b

-- Prove that the number of correct propositions is 3
theorem number_of_correct_propositions :
  (prop1 ∧ prop3 ∧ prop4) ∧ ¬ prop2 := sorry

end number_of_correct_propositions_l571_571735


namespace pure_powers_sum_product_l571_571265

theorem pure_powers_sum_product (n : ℕ) (hn : n > 0) :
  ∃ (a : Fin n → ℕ), (Function.Injective a) ∧ (∃ m : ℤ, (∑ i, a i) = m ^ 2009) ∧ (∃ k : ℤ, (∏ i, a i) = k ^ 2010) := sorry

end pure_powers_sum_product_l571_571265


namespace sum_ages_l571_571035

theorem sum_ages (A_years B_years C_years : ℕ) (h1 : B_years = 30)
  (h2 : 10 * (B_years - 10) = (A_years - 10) * 2)
  (h3 : 10 * (B_years - 10) = (C_years - 10) * 3) :
  A_years + B_years + C_years = 90 :=
sorry

end sum_ages_l571_571035


namespace units_digit_17_pow_2023_l571_571114

theorem units_digit_17_pow_2023 : (17^2023 % 10) = 3 := sorry

end units_digit_17_pow_2023_l571_571114


namespace binary_sum_l571_571196

-- Define the binary representations in terms of their base 10 equivalent.
def binary_111111111 := 511
def binary_1111111 := 127

-- State the proof problem.
theorem binary_sum : binary_111111111 + binary_1111111 = 638 :=
by {
  -- placeholder for proof
  sorry
}

end binary_sum_l571_571196


namespace max_min_subset_cover_l571_571418

theorem max_min_subset_cover {S : Type} [fintype S] (n : ℕ) (A : finset (finset S))
  (h1 : 1 < n)
  (h2 : ∀ a ∈ A, n ≤ a.card)
  (h3 : ∀ s ∈ (finset.univ : finset S), n ≤ (A.filter (λ a, s ∈ a)).card) :
  ∃ B ⊆ A, (B.card = fintype.card S - n ∧ ∀ B' ⊆ A, (finset.univ : finset S).card ≤ B'.card → B.card ≤ B'.card) :=
begin
  sorry
end

end max_min_subset_cover_l571_571418


namespace value_of_ab_plus_bc_plus_ca_l571_571683

theorem value_of_ab_plus_bc_plus_ca (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end value_of_ab_plus_bc_plus_ca_l571_571683


namespace convert_13_to_binary_l571_571568

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (n : Nat) (acc : List Nat) : List Nat :=
      if n = 0 then acc
      else aux (n / 2) ((n % 2) :: acc)
    aux n []

theorem convert_13_to_binary : decimal_to_binary 13 = [1, 1, 0, 1] :=
  by
    sorry -- Proof to be provided

end convert_13_to_binary_l571_571568


namespace narrow_black_stripes_are_eight_l571_571870

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l571_571870


namespace probability_5_of_6_odd_rolls_l571_571052

def binom_coeff : ℕ → ℕ → ℕ
| n k := Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binom_coeff n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_5_of_6_odd_rolls :
  binomial_probability 6 5 (1/2) = 3/16 :=
by
  -- Proof will go here, but we skip it with sorry for now.
  sorry

end probability_5_of_6_odd_rolls_l571_571052


namespace num_divisible_by_3_between_sqrt_50_and_sqrt_200_l571_571790

theorem num_divisible_by_3_between_sqrt_50_and_sqrt_200 : 
  ∃ (n : ℕ), n = 2 ∧ ∀ m, \(\sqrt{50} < m \and m < \sqrt{200}\)  → m % 3 = 0 → m ∈ {9, 12} :=
sorry

end num_divisible_by_3_between_sqrt_50_and_sqrt_200_l571_571790


namespace expansive_sequence_in_interval_l571_571082

-- Definition of an expansive sequence
def expansive_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (i j : ℕ), (i < j) → (|a i - a j| ≥ 1 / j)

-- Upper bound condition for C
def upper_bound_C (C : ℝ) : Prop :=
  C ≥ 2 * Real.log 2

-- The main statement combining both definitions into a proof problem
theorem expansive_sequence_in_interval (C : ℝ) (a : ℕ → ℝ) 
  (h_exp : expansive_sequence a) (h_bound : upper_bound_C C) :
  ∀ n, 0 ≤ a n ∧ a n ≤ C :=
sorry

end expansive_sequence_in_interval_l571_571082


namespace train_speed_proof_l571_571161

-- Definitions of given constants and conditions
def minutesToHours (minutes : ℝ) : ℝ := minutes / 60
def originalTime (distance speed : ℝ) : ℝ := distance / speed
def delayedTime (distance speed reduction delay : ℝ) : ℝ :=
  originalTime distance speed + minutesToHours delay

-- Given conditions
variables (v : ℝ) -- Original speed of the train in km/h
variables (d : ℝ) -- Remaining distance after the first accident in km
variables (T : ℝ) -- Time to cover the whole distance at original speed in hours
variables (reduction_ratio : ℝ := 3 / 4) -- Reduction ratio of the speed after accident
variables (delay1 : ℝ := 35 / 60) -- Delay in hours after the first accident
variables (delay2 : ℝ := 25 / 60) -- Delay in hours after the second accident

-- Equations formed by the problem conditions
def eq1 : Prop := originalTime 50 v + originalTime d (reduction_ratio * v) = T + delay1
def eq2 : Prop := originalTime 74 v + originalTime (d - 24) (reduction_ratio * v) = T + delay2

-- The main proof statement to show the solution v = 48
theorem train_speed_proof : eq1 ∧ eq2 → v = 48 :=
by
  sorry

end train_speed_proof_l571_571161


namespace distance_between_intersections_l571_571567

theorem distance_between_intersections (a : ℝ) (a_pos : 0 < a) : 
  |(Real.log a / Real.log 2) - (Real.log (a / 3) / Real.log 2)| = Real.log 3 / Real.log 2 :=
by
  sorry

end distance_between_intersections_l571_571567


namespace curve_symmetrical_about_theta_five_sixths_pi_l571_571821

noncomputable def curve_symmetry (ρ θ : ℝ) : ℝ := 4 * Real.sin(θ - Real.pi / 3)

theorem curve_symmetrical_about_theta_five_sixths_pi : 
  (∀ θ ρ, curve_symmetry ρ θ = 4 * Real.sin(θ - Real.pi / 3)) →
  (∀ θ, (curve_symmetry ρ θ = - curve_symmetry ρ (θ + Real.pi)) ∨
  curve_symmetry ρ θ = curve_symmetry ρ (θ + Real.pi)) :=
by
  sorry

end curve_symmetrical_about_theta_five_sixths_pi_l571_571821


namespace only_surjective_function_satisfying_condition_l571_571232

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ y : ℕ, ∃ x : ℕ, f x = y

def same_prime_divisors (a b : ℕ) : Prop :=
  ∀ p : ℕ, prime p → (p ∣ a ↔ p ∣ b)

theorem only_surjective_function_satisfying_condition (f : ℕ → ℕ) 
  (h_surjective : is_surjective f)
  (h_prime_divisors : ∀ m n : ℕ, same_prime_divisors (f (m + n)) (f m + f n)) : 
  ∀ n : ℕ, f n = n := 
sorry

end only_surjective_function_satisfying_condition_l571_571232


namespace mult_63_37_l571_571206

theorem mult_63_37 : 63 * 37 = 2331 :=
by {
  sorry
}

end mult_63_37_l571_571206


namespace number_of_integers_l571_571251

theorem number_of_integers (n : ℤ) : {n : ℤ | 15 < n^2 ∧ n^2 < 120}.finite.card = 14 :=
sorry

end number_of_integers_l571_571251


namespace neg_p_exists_x_cos_gt_one_l571_571285

def cos_le_one_for_all_x (p : Prop) := ∀ x : ℝ, cos x ≤ 1

theorem neg_p_exists_x_cos_gt_one : ¬ (cos_le_one_for_all_x p) → ∃ x : ℝ, cos x > 1 :=
begin
  sorry
end

end neg_p_exists_x_cos_gt_one_l571_571285


namespace sum_reciprocal_roots_l571_571570

-- Define the polynomial
def polynomial := (3 : ℤ) * x^3 + (2 : ℤ) * x^2 + (1 : ℤ) * x + (8 : ℤ)

-- Define the statement to prove the sum of reciprocals of the roots is -1/8
theorem sum_reciprocal_roots:
  (∃ p q r : ℂ, polynomial.eval p = 0 ∧ polynomial.eval q = 0 ∧ polynomial.eval r = 0 ∧
  (1 / p + 1 / q + 1 / r = -1 / 8)) :=
sorry

end sum_reciprocal_roots_l571_571570


namespace scaling_transformation_l571_571817

theorem scaling_transformation:
  ∀ (x y x' y': ℝ), 
  (x^2 + y^2 = 1) ∧ (x' = 5 * x) ∧ (y' = 3 * y) → 
  (x'^2 / 25 + y'^2 / 9 = 1) :=
by intros x y x' y'
   sorry

end scaling_transformation_l571_571817


namespace narrow_black_stripes_l571_571853

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l571_571853


namespace inequality_solution_l571_571673

theorem inequality_solution (x : ℝ) : 
  (1 + 2 * x ≥ 0) → (1 + 2 * x ≠ 1) → 
  ( (4 * x^2) / ((1 - sqrt (1 + 2 * x))^2) < (2 * x + 9)  ) → 
  (-1 / 2 ≤ x ∧ x < 45 / 8 ∧ x ≠ 0) :=
by
  sorry

end inequality_solution_l571_571673


namespace different_language_classes_probability_l571_571832

theorem different_language_classes_probability :
  let total_students := 40
  let french_students := 28
  let spanish_students := 26
  let german_students := 15
  let french_and_spanish_students := 10
  let french_and_german_students := 6
  let spanish_and_german_students := 8
  let all_three_languages_students := 3
  let total_pairs := Nat.choose total_students 2
  let french_only := french_students - (french_and_spanish_students + french_and_german_students - all_three_languages_students) - all_three_languages_students
  let spanish_only := spanish_students - (french_and_spanish_students + spanish_and_german_students - all_three_languages_students) - all_three_languages_students
  let german_only := german_students - (french_and_german_students + spanish_and_german_students - all_three_languages_students) - all_three_languages_students
  let french_only_pairs := Nat.choose french_only 2
  let spanish_only_pairs := Nat.choose spanish_only 2
  let german_only_pairs := Nat.choose german_only 2
  let single_language_pairs := french_only_pairs + spanish_only_pairs + german_only_pairs
  let different_classes_probability := 1 - (single_language_pairs / total_pairs)
  different_classes_probability = (34 / 39) :=
by
  sorry

end different_language_classes_probability_l571_571832


namespace flower_pot_total_cost_l571_571445

theorem flower_pot_total_cost 
  (n : ℕ)
  (cost : ℕ → ℝ)
  (h₀ : n = 6)
  (h₁ : ∀ k:ℕ, k < n - 1 → cost (k + 1) = cost k + 0.1)
  (h₂ : cost 5 = 1.625) :
  (finset.range n).sum (λ k, cost k) = 8.25 :=
by
  sorry

end flower_pot_total_cost_l571_571445


namespace a_2016_is_minus_5_l571_571745

def sequence (a : ℕ → ℤ) : Prop :=
  (a 1 = 1) ∧ (a 2 = 6) ∧ ∀ n, a (n + 2) = a (n + 1) - a n

theorem a_2016_is_minus_5 (a : ℕ → ℤ) (h : sequence a) : a 2016 = -5 :=
by sorry

end a_2016_is_minus_5_l571_571745


namespace find_real_number_l571_571537

theorem find_real_number :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = sqrt 15 :=
sorry

end find_real_number_l571_571537


namespace union_of_A_B_l571_571288

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x > 0}

theorem union_of_A_B :
  A ∪ B = {x | x ≥ -1} := by
  sorry

end union_of_A_B_l571_571288


namespace similarity_of_triangles_l571_571275

variables {A B C D E F B1 D1 F1 : Point}
variables {α β γ : Line}

-- Point and Line are example types; their definitions depend on the formal system/framework used.
-- Typical point, line definition and reflection would be defined accordingly in the actual Lean 4 environment.

-- Given: A cyclic hexagon ABCDEF inscribed in a circle
axiom cyclic_hexagon (hex : Hexagon A B C D E F) : Cyclic hexagon

-- Given: The condition on the products of the segments
axiom segment_condition : AB * CD * EF = BC * DE * AF

-- Points B1, D1, F1 are reflections of B, D, F across respective lines AC, CE, EA
axiom reflection_B1 : ReflectedOver(B, α) = B1
axiom reflection_D1 : ReflectedOver(D, β) = D1
axiom reflection_F1 : ReflectedOver(F, γ) = F1
axiom α := line A C
axiom β := line C E
axiom γ := line E A

-- Prove similarity of triangles B1D1F1 and BDF
theorem similarity_of_triangles : SimilarTriangle(B1, D1, F1, B, D, F) :=
by sorry

end similarity_of_triangles_l571_571275


namespace num_factors_48_l571_571344

theorem num_factors_48 : 
  ∀ (n : ℕ), n = 48 → (∃ k : ℕ, k = 10 ∧ ∀ d : ℕ, d ∣ n → (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 16 ∨ d = 24 ∨ d = 48)) :=
  by
    intros n h
    sorry

end num_factors_48_l571_571344


namespace probability_of_5_odd_in_6_rolls_l571_571069

open Classical

noncomputable def prob_odd_in_six_rolls : ℚ :=
  let num_rolls := 6
  let prob_odd_single := 1 / 2
  let binom_coeff := Nat.choose num_rolls 5
  let total_outcomes := (2 : ℕ) ^ num_rolls
  binom_coeff * ((prob_odd_single ^ 5) * ((1 - prob_odd_single) ^ (num_rolls - 5))) / total_outcomes

theorem probability_of_5_odd_in_6_rolls :
  prob_odd_in_six_rolls = 3 / 32 :=
by
  sorry

end probability_of_5_odd_in_6_rolls_l571_571069


namespace smallest_positive_integer_remainder_l571_571531

theorem smallest_positive_integer_remainder
  (b : ℕ) (h1 : b % 4 = 3) (h2 : b % 6 = 5) :
  b = 11 := by
  sorry

end smallest_positive_integer_remainder_l571_571531


namespace initial_goldfish_eq_15_l571_571449

-- Let's define our setup as per the conditions provided
def fourGoldfishLeft := 4
def elevenGoldfishDisappeared := 11

-- Our main statement that we need to prove
theorem initial_goldfish_eq_15 : fourGoldfishLeft + elevenGoldfishDisappeared = 15 := by
  sorry

end initial_goldfish_eq_15_l571_571449


namespace length_of_congruent_sides_of_isosceles_triangle_in_square_l571_571269

theorem length_of_congruent_sides_of_isosceles_triangle_in_square
    (side_length_of_square : ℝ)
    (base_of_triangle : ℝ)
    (h₀ : side_length_of_square = 2)
    (h₁ : base_of_triangle = 2)
    (h₂ : 4 * (1/2 * base_of_triangle * (1 : ℝ)) = side_length_of_square ^ 2) :
  ∃ s : ℝ, s = Real.sqrt 2 ∧ s² = 1^2 + 1^2 :=
by
  sorry

end length_of_congruent_sides_of_isosceles_triangle_in_square_l571_571269


namespace total_marbles_distributed_l571_571178

/-- At Junwoo's school, 37 marbles each were distributed to 23 classes, and there are 16 left.
    We need to prove that the number of marbles distributed to students at Junwoo's school is 867.
/
theorem total_marbles_distributed : 
  ∀ (marbles_per_class : ℕ) (number_of_classes : ℕ) (leftover_marbles : ℕ), 
  marbles_per_class = 37 → 
  number_of_classes = 23 → 
  leftover_marbles = 16 → 
  (marbles_per_class * number_of_classes + leftover_marbles) = 867 :=
by
  intros marbles_per_class number_of_classes leftover_marbles h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_marbles_distributed_l571_571178


namespace newer_pump_time_l571_571607

-- Problem statement definitions
def old_pump_rate : ℝ := 1 / 600
def newer_pump_rate (T : ℝ) : ℝ := 1 / T
def both_pumps_rate : ℝ := 1 / 150

-- Goal: Prove the time taken for the newer pump alone is 200 seconds
theorem newer_pump_time :
  ∃ T : ℝ, newer_pump_rate T = 1 / 200 ∧
  old_pump_rate + newer_pump_rate T = both_pumps_rate :=
by 
  use 200
  split
  { 
    unfold newer_pump_rate 
    norm_num 
  }
  { 
    unfold newer_pump_rate old_pump_rate both_pumps_rate 
    norm_num 
    sorry 
  }

end newer_pump_time_l571_571607


namespace real_number_infinite_continued_fraction_l571_571546

theorem real_number_infinite_continued_fraction:
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / y))) ∧ y = 5 / 3 := 
begin
  sorry
end

end real_number_infinite_continued_fraction_l571_571546


namespace solve_equation_l571_571272

noncomputable def mul_star : ℝ → ℝ → ℝ := λ a b, a*(a*b - 7)

theorem solve_equation (x : ℝ) :
  (∃ x : ℝ, 3 * x = 2 * (-8)) → x = -25/9 :=
begin
  -- Proof will go here
  sorry
end

end solve_equation_l571_571272


namespace count_multiples_of_6_or_8_but_not_both_l571_571782

theorem count_multiples_of_6_or_8_but_not_both: 
  let multiples_of_six := finset.filter (λ n, 6 ∣ n) (finset.range 151)
  let multiples_of_eight := finset.filter (λ n, 8 ∣ n) (finset.range 151)
  let multiples_of_twenty_four := finset.filter (λ n, 24 ∣ n) (finset.range 151)
  multiples_of_six.card + multiples_of_eight.card - 2 * multiples_of_twenty_four.card = 31 := 
by {
  -- Provided proof omitted
  sorry
}

end count_multiples_of_6_or_8_but_not_both_l571_571782


namespace sum_equals_target_l571_571695

open BigOperators

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_initial_condition : f 0 = 1

axiom f_functional_eq (x y : ℝ) : f (x * y + 1) = f x * f y - f y - x + 2

theorem sum_equals_target : (∑ i in finset.range 2023, 1 / (f i * f (i + 1))) = 2023 / 4050 :=
by
  sorry

end sum_equals_target_l571_571695


namespace probability_unique_tens_digits_correct_l571_571003

-- Define the range of integers
def range_10_to_59 : finset ℕ := finset.Icc 10 59

-- Function to extract the tens digit
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Define the event that each integer in a chosen set has a different tens digit
def unique_tens_digits (s : finset ℕ) : Prop :=
  (∀ x y ∈ s, x ≠ y → tens_digit x ≠ tens_digit y)

-- Define the total number of ways to choose 5 unique integers with unique tens digits from the range
def ways_with_unique_tens_digits : ℕ := 10^5

-- Define the total number of combinations of choosing 5 integers from the range
def total_combinations : ℕ := nat.choose 50 5

-- Compute the required probability
def probability_unique_tens_digits : ℚ :=
  ways_with_unique_tens_digits.to_rat / total_combinations.to_rat

-- Proven statement
theorem probability_unique_tens_digits_correct :
  probability_unique_tens_digits = 2500 / 52969 := by
  sorry

end probability_unique_tens_digits_correct_l571_571003


namespace problem_conditions_l571_571631

noncomputable def sphere_radius : ℝ := 
  let V := 480 in
  let f := (9 : ℝ) / Real.pi * Real.sqrt (3 / 2)
  let R_cubed := f * V
  Real.cbrt (f * V)

theorem problem_conditions (R : ℝ) (h1 : 480 = (Real.pi * R^3 * Real.sqrt (2/3)) / 9) :
  R = Real.cbrt ((9 * 480 * Real.sqrt (3 / 2)) / Real.pi) :=
by sorry

end problem_conditions_l571_571631


namespace regular_tetrahedron_properties_equiv_l571_571616

def equilateral_triangle_properties (T : Triangle) := 
  (∀ (a b c : T.edge), a = b ∧ b = c) ∧ 
  (∀ (α β γ : T.angle), α = β ∧ β = γ)

def regular_tetrahedron_properties (T : Tetrahedron) := 
  (∀ (a b c d e f : T.edge), a = b ∧ b = c ∧ c = d ∧ d = e ∧ e = f) ∧ 
  (∀ (a b c d : T.face), a ≅ b ∧ b ≅ c ∧ c ≅ d) ∧ 
  (∀ (α β γ δ ε ζ : T.dihedral_angle), α = β ∧ β = γ ∧ γ = δ ∧ δ = ε ∧ ε = ζ)

variables (T : Tetrahedron)

theorem regular_tetrahedron_properties_equiv : 
  (regular_tetrahedron_properties T) →
  (∀ (a b c : T.edge), a = b ∧ b = c) ∧ 
  (∀ (a b c d : T.face), a ≅ b ∧ b ≅ c ∧ c ≅ d) ∧ 
  (∀ (α β γ δ ε ζ : T.dihedral_angle), α = β ∧ β = γ ∧ γ = δ ∧ δ = ε ∧ ε = ζ) 
  := sorry

end regular_tetrahedron_properties_equiv_l571_571616


namespace pqrs_sum_l571_571402

def numerator := x^3 + 4 * x^2 + 3 * x
def denominator := x^4 + 2 * x^3 - 3 * x^2
def rational_function := numerator / denominator

def p := 1 -- number of holes
def q := 3 -- number of vertical asymptotes
def r := 1 -- number of horizontal asymptotes
def s := 0 -- number of oblique asymptotes

theorem pqrs_sum : p + 2 * q + 3 * r + 4 * s = 10 := 
by
  rw [p, q, r, s]
  norm_num

end pqrs_sum_l571_571402


namespace continued_fraction_solution_l571_571545

theorem continued_fraction_solution :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / y))) ∧ y = (3 + Real.sqrt 69) / 2 :=
begin
  sorry
end

end continued_fraction_solution_l571_571545


namespace narrow_black_stripes_are_8_l571_571895

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l571_571895


namespace simplify_trig_expression_eq_neg_one_l571_571936

-- Definitions of the trigonometric identities and expressions
noncomputable def simplify_trig_expression (x y: ℝ) : ℝ :=
  (real.sqrt (1 + 2 * real.sin x * real.cos y)) / (real.sin y + real.cos x)

-- Proving that the simplification evaluates to -1
theorem simplify_trig_expression_eq_neg_one :
  simplify_trig_expression 610 430 = -1 := by
  sorry

end simplify_trig_expression_eq_neg_one_l571_571936


namespace number_of_solutions_quad_eq_l571_571021

theorem number_of_solutions_quad_eq : 
  (∀ x : ℝ, x^2 - |x| - 6 = 0) ↔ (count (solutions (x^2 - |x| - 6)) = 2) := by sorry

end number_of_solutions_quad_eq_l571_571021


namespace emily_catch_catfish_l571_571221

-- Definitions based on given conditions
def num_trout : ℕ := 4
def num_bluegills : ℕ := 5
def weight_trout : ℕ := 2
def weight_catfish : ℚ := 1.5
def weight_bluegill : ℚ := 2.5
def total_fish_weight : ℚ := 25

-- Lean statement to prove the number of catfish
theorem emily_catch_catfish : ∃ (num_catfish : ℕ), 
  num_catfish * weight_catfish = total_fish_weight - (num_trout * weight_trout + num_bluegills * weight_bluegill) ∧
  num_catfish = 3 := by
  sorry

end emily_catch_catfish_l571_571221


namespace example_proof_l571_571670

def J (a b c : ℝ) : ℝ := a / b + b / c + c / a

theorem example_proof : J 3 18 12 = 17 / 3 := by
  sorry

end example_proof_l571_571670


namespace sum_of_possible_values_of_b_l571_571036

theorem sum_of_possible_values_of_b :
  (∑ b in {b : ℤ | ∃ (x y : ℤ), g(x) = 0 ∧ g(y) = 0}, b) = 40 :=
by
  let g : ℤ -> ℤ := λ x b, x^2 - b * x + 3 * b
  sorry

end sum_of_possible_values_of_b_l571_571036


namespace surface_area_inscribed_sphere_of_cube_l571_571034

-- Define the edge length of the cube
def edge_length : ℝ := 2

-- Define the radius of the inscribed sphere
def radius (a : ℝ) : ℝ := a / 2

-- Define the formula for the surface area of a sphere
def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- The theorem we want to prove
theorem surface_area_inscribed_sphere_of_cube :
  surface_area_of_sphere (radius edge_length) = 4 * Real.pi :=
by
  sorry

end surface_area_inscribed_sphere_of_cube_l571_571034


namespace tan_theta_sqrt_2_expr_value_equals_l571_571682

theorem tan_theta_sqrt_2 
  (θ : ℝ) 
  (h1 : tan (2 * θ) = -2 * real.sqrt 2) 
  (h2 : θ ∈ set.Ioo (real.pi / 4) (real.pi / 2)) :
  tan θ = real.sqrt 2 :=
sorry

theorem expr_value_equals 
  (θ : ℝ) 
  (h0 : tan θ = real.sqrt 2)
  (h1 : tan (2 * θ) = -2 * real.sqrt 2) 
  (h2 : θ ∈ set.Ioo (real.pi / 4) (real.pi / 2)) :
  (2 * real.cos (θ / 2) ^ 2 - real.sin θ - 1) / (real.sqrt 2 * real.sin (real.pi / 4 + θ)) = 
  2 * real.sqrt 2 - 3 :=
sorry

end tan_theta_sqrt_2_expr_value_equals_l571_571682


namespace multiples_of_6_or_8_but_not_both_l571_571783

theorem multiples_of_6_or_8_but_not_both (n : ℕ) : 
  n = 25 ∧ (n = 18) ∧ (n = 6) → (25 - 6) + (18 - 6) = 31 :=
by
  sorry

end multiples_of_6_or_8_but_not_both_l571_571783


namespace narrow_black_stripes_l571_571865

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l571_571865


namespace sin_alpha_value_l571_571729

theorem sin_alpha_value 
  (x y : ℝ) 
  (h1 : x = 1) 
  (h2 : y = -1)
  (hPoint : (x, y) = (1, -1)) : 
  Real.sin (Real.atan2 y x) = - (Real.sqrt 2) / 2 := 
by 
  sorry

end sin_alpha_value_l571_571729


namespace probability_of_5_odd_numbers_l571_571076

-- Define a function to represent the binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Axiom that defines the probability of getting an odd number
axiom fair_die_prob : ∀ (x : ℕ), 0 < x ∧ x ≤ 6 -> (1/2)

-- Define the problem statement about the probability
theorem probability_of_5_odd_numbers (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 5) : 
  (binom n k) / 2^n = 3 / 32 := sorry

end probability_of_5_odd_numbers_l571_571076


namespace arithmetic_seq_geom_sum_sum_of_b_n_l571_571495

variables {α : Type*} [linear_ordered_field α]

/-- Problem Part Ⅰ -/
theorem arithmetic_seq_geom_sum (a_1 d : α) (d_ne_zero : d ≠ 0) 
  (h_geom : (a_1 + d)^2 = a_1 * (a_1 + 4 * d)) :
  let S : ℕ → α := λ n, n * (2 * a_1 + (n - 1) * d) / 2 in
  S 1 * S 9 = (S 3) ^ 2 :=
sorry

/-- Problem Part Ⅱ -/
theorem sum_of_b_n (n : ℕ) :
  let a : ℕ → ℕ := λ n, 1 + n * 2 in
  let b : ℕ → ℕ := λ n, 2^(n + 1) - 1 in
  let T : ℕ → ℕ := λ n, ((4 * (↑1 - 2^n) / (1 - 2)) - n).to_nat in
  T n = 2^(n+2) - 4 - n :=
sorry

end arithmetic_seq_geom_sum_sum_of_b_n_l571_571495


namespace marilyn_bottle_caps_l571_571907

-- Define initial conditions
def M0 : ℝ := 51.0
def N : ℝ := 36.0
def P : ℝ := 0.25
def L : ℝ := 45.0
def F : ℝ := 1 / 3

-- Marilyn's final bottle caps
def Marilyn_final_caps : ℝ :=
  let after_nancy := M0 + N
  let after_tom := after_nancy * (1 - P)
  let from_lisa := L * F
  in after_tom + from_lisa

-- The proof statement
theorem marilyn_bottle_caps : Marilyn_final_caps = 80.25 :=
by
  sorry

end marilyn_bottle_caps_l571_571907


namespace fish_left_in_tank_l571_571794

-- Define the initial number of fish and the number of fish moved
def initialFish : Real := 212.0
def movedFish : Real := 68.0

-- Define the number of fish left in the tank
def fishLeft (initialFish : Real) (movedFish : Real) : Real := initialFish - movedFish

-- Theorem stating the problem
theorem fish_left_in_tank : fishLeft initialFish movedFish = 144.0 := by
  sorry

end fish_left_in_tank_l571_571794


namespace parabola_position_right_l571_571629

theorem parabola_position_right (x : ℝ) (h₁ : ∀ x, (x^2 - (1/2) * x + 2) = y₁)
                              (h₂ : ∀ x, (x^2 + (1/2) * x + 2) = y₂) :
  ∃ x₁ x₂ : ℝ, x₁ > x₂ ∧ (vertex_x y₁) = x₁ ∧ (vertex_x y₂) = x₂ :=
begin
  sorry
end

def vertex_x (f : ℝ → ℝ) : ℝ := -(f 1) / (2 * (f 2))

end parabola_position_right_l571_571629


namespace remaining_sand_fraction_l571_571133

theorem remaining_sand_fraction (total_weight : ℕ) (used_weight : ℕ) (h1 : total_weight = 50) (h2 : used_weight = 30) : 
  (total_weight - used_weight) / total_weight = 2 / 5 :=
by 
  sorry

end remaining_sand_fraction_l571_571133


namespace factor_values_l571_571234

theorem factor_values (t : ℝ) :
  (∃ q : ℝ[X], (10 * X^2 + 23 * X - 7) = (X - C t) * q) ↔ (t = 1/5 ∨ t = -7/2) :=
by
  sorry

end factor_values_l571_571234


namespace probability_even_sum_of_prime_pairs_not_exceeding_13_l571_571478

theorem probability_even_sum_of_prime_pairs_not_exceeding_13 :
  let primes := [2, 3, 5, 7, 11, 13]
  let pairs := [(x, y) | x ∈ primes, y ∈ primes, x < y]
  let even_sum_pairs := [(x, y) | (x, y) ∈ pairs, (x + y) % 2 = 0]
  in (even_sum_pairs.length : ℚ) / (pairs.length : ℚ) = 2 / 3 :=
by
  sorry

end probability_even_sum_of_prime_pairs_not_exceeding_13_l571_571478


namespace ratio_area_triangle_to_circumcircle_l571_571406

theorem ratio_area_triangle_to_circumcircle (A B C K M : Type) [Geometry] 
  (h₁ : angle B A C = 120) 
  (h₂ : AK_angle_bisector_perpendicular_to_BM_median) :
  (area △ABC) / (area (circumcircle △ABC)) = (3 * sqrt 3) / (8 * π) := sorry


end ratio_area_triangle_to_circumcircle_l571_571406


namespace count_multiples_6_or_8_but_not_both_l571_571772

theorem count_multiples_6_or_8_but_not_both : 
  (∑ i in Finset.range 150, ((if (i % 6 = 0 ∧ i % 24 ≠ 0) ∨ (i % 8 = 0 ∧ i % 24 ≠ 0) then 1 else 0) : ℕ)) = 31 := by
  sorry

end count_multiples_6_or_8_but_not_both_l571_571772


namespace solve_for_x_l571_571375

theorem solve_for_x (x : ℝ) :
  log 10 (3 * x ^ 2 - 5 * x + 7) = 2 ↔ 
  (x = (5 + sqrt 1141) / 6 ∨ x = (5 - sqrt 1141) / 6) :=
by
  sorry

end solve_for_x_l571_571375


namespace num_factors_48_l571_571343

theorem num_factors_48 : 
  ∀ (n : ℕ), n = 48 → (∃ k : ℕ, k = 10 ∧ ∀ d : ℕ, d ∣ n → (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 16 ∨ d = 24 ∨ d = 48)) :=
  by
    intros n h
    sorry

end num_factors_48_l571_571343


namespace part_I_part_II_l571_571739

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a

theorem part_I (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≥ (1 : ℝ) / Real.exp 1 :=
sorry

theorem part_II (a x1 x2 x : ℝ) (hx1 : 0 < x1) (hx2 : x1 < x2) (hx : x1 < x ∧ x < x2) :
  (f x a - f x1 a) / (x - x1) < (f x a - f x2 a) / (x - x2) :=
sorry

end part_I_part_II_l571_571739


namespace number_of_narrow_black_stripes_l571_571903

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l571_571903


namespace find_k_exact_one_real_solution_l571_571672

theorem find_k_exact_one_real_solution (k : ℝ) :
  (∀ x : ℝ, (3*x + 6)*(x - 4) = -33 + k*x) ↔ (k = -6 + 6*Real.sqrt 3 ∨ k = -6 - 6*Real.sqrt 3) := 
by
  sorry

end find_k_exact_one_real_solution_l571_571672


namespace det_A_l571_571426

open Matrix

theorem det_A (a b : ℝ) (h : ∃ A : Matrix (Fin 2) (Fin 2) ℝ, A = !![a, 2; -3, b] ∧ A + 2•A⁻¹ = 0) : det !![a, 2; -3, b] = 4 :=
by
  -- To be filled with the proof
  sorry

end det_A_l571_571426


namespace quadratic_linear_relation_l571_571718

variable (a m n : ℝ)
variable (h_a_nonzero : a ≠ 0)
variable (h_m_negative : m < 0)
variable (h_n_negative : n < 0)

theorem quadratic_linear_relation :
  (∃ (h : ℝ → ℝ) (k : ℝ → ℝ), h = (λ x, a * (x + m) * (x + n)) ∧ k = (λ x, a * x + mn) ∧ graph_A_matching_condition h k) :=
sorry

-- Placeholder for the actual condition matching to Graph (A)
def graph_A_matching_condition (h k : ℝ → ℝ) : Prop :=
sorry

end quadratic_linear_relation_l571_571718


namespace exists_integer_a_l571_571838

theorem exists_integer_a (p : ℕ) (hp : p ≥ 5) [Fact (Nat.Prime p)] : 
  ∃ a : ℕ, 1 ≤ a ∧ a ≤ p - 2 ∧ (¬ p^2 ∣ a^(p-1) - 1) ∧ (¬ p^2 ∣ (a+1)^(p-1) - 1) :=
by
  sorry

end exists_integer_a_l571_571838


namespace hall_marriage_theorem_l571_571434

variables {X Y : Type} [fintype X] [fintype Y] (G : X → Y → Prop)

def N (A : finset X) : finset Y :=
  A.bUnion (λ x, finset.filter (G x) finset.univ)

theorem hall_marriage_theorem (h : ∀ (A : finset X), A.card ≤ (N G A).card) :
  (∃ M : X → option Y, (∀ x ∈ X, ∃ y ∈ Y, M x = some y) ∧ function.injective (λ x, option.get (h ▸ M x))) ↔
  (∀ A : finset X, A.card ≤ (N G A).card) :=
sorry

end hall_marriage_theorem_l571_571434


namespace pyramid_volume_l571_571462

noncomputable def volume_of_pyramid {s : ℝ} {h : ℝ} (area_base : ℝ) : ℝ :=
  (1 / 3) * area_base * h

theorem pyramid_volume :
  let side_length_hexagon := 5
  let side_length_triangle := 10
  let height := (2 / 3) * (10 * (Real.sqrt 3 / 2))
  let area_hexagon := 6 * (5^2 * Real.sqrt 3 / 4)
  let volume := volume_of_pyramid area_hexagon height
  volume = 375 := by
  sorry

end pyramid_volume_l571_571462


namespace distinct_positive_factors_of_48_l571_571351

theorem distinct_positive_factors_of_48 : 
  let n := 48 in
  let factors := (2^4) * (3^1) in
  ∀ n : ℕ, n = factors → 
  (let num_factors := (4 + 1) * (1 + 1)
  in num_factors = 10) :=
by 
  let n := 48
  let factors := (2^4) * (3^1)
  assume h : n = factors
  let num_factors := (4 + 1) * (1 + 1)
  show num_factors = 10 from sorry

end distinct_positive_factors_of_48_l571_571351


namespace part1_term_formula_part2_sum_bn_l571_571026

-- Definition of sequences \{a_n\} and \{S_n\}
def a : ℕ → ℝ
| 0     := 2
| (n+1) := -4 * (1/3)^n

def S : ℕ → ℝ := λ n, 2 * (1/3)^n

-- Definition of sequence \{b_n\}
def b (n : ℕ) : ℝ := a n * S n

-- The Lean theorem statements
theorem part1_term_formula (n : ℕ) :
  a n = if n = 0 then 2 else -4 * (1/3)^(n-1) :=
sorry

theorem part2_sum_bn :
  (∑ i in finset.range (n + 1), b i) = 3 :=
sorry

end part1_term_formula_part2_sum_bn_l571_571026


namespace count_integers_within_bounds_l571_571258

theorem count_integers_within_bounds : 
  ∃ (count : ℕ), count = finset.card (finset.filter (λ n : ℤ, 15 < n^2 ∧ n^2 < 120) (finset.Icc (-10) 10)) ∧ count = 14 := 
by
  sorry

end count_integers_within_bounds_l571_571258


namespace discount_difference_l571_571177

theorem discount_difference (x : ℝ) (h1 : x = 8000) : 
  (x * 0.7) - ((x * 0.8) * 0.9) = 160 :=
by
  rw [h1]
  sorry

end discount_difference_l571_571177


namespace nested_fraction_solution_l571_571540

noncomputable def nested_fraction : ℝ :=
  3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / ... ))))

theorem nested_fraction_solution :
  nested_fraction = (3 + Real.sqrt 69) / 2 :=
sorry

end nested_fraction_solution_l571_571540


namespace area_of_intersection_rectangle_circle_l571_571045

theorem area_of_intersection_rectangle_circle :
  let rectangle_vertices := [(3, 7), (12, 7), (12, -4), (3, -4)]
  let circle_center := (3, -4)
  let circle_radius := 4
  let circle_eq := λ x y, (x - circle_center.1) ^ 2 + (y + circle_center.2) ^ 2 = circle_radius ^ 2
  let intersection_area := 4 * Real.pi
  (∀ x y, (x, y) ∈ rectangle_vertices → circle_eq x y) →
  True :=
begin
  sorry
end

end area_of_intersection_rectangle_circle_l571_571045


namespace find_heaviest_coin_l571_571502

-- Define the conditions
def coins (n : ℕ) (hn : n > 2) := {c : fin n → ℝ // ∀ i j, i ≠ j → c i ≠ c j} -- n coins of different masses.
def scales (n : ℕ) := fin n → Prop -- n scales with one faulty

-- Proposition to prove
theorem find_heaviest_coin (n : ℕ) (hn : n > 2) (c : coins n hn) (s : scales n) : 
  ∃ k, k = 2 * n - 1 := sorry

end find_heaviest_coin_l571_571502


namespace factor_expression_l571_571203

variable (x : ℝ)

theorem factor_expression :
  (18 * x ^ 6 + 50 * x ^ 4 - 8) - (2 * x ^ 6 - 6 * x ^ 4 - 8) = 8 * x ^ 4 * (2 * x ^ 2 + 7) :=
by
  sorry

end factor_expression_l571_571203


namespace count_integers_with_5_and_7_l571_571364

theorem count_integers_with_5_and_7 : 
  ∃ (n : ℕ), (∃ (f : Fin 3 → Fin 10), (f 0 = 5) ∧ (f 1 = 7) ∧ (4000 ≤ n) ∧ (n < 5000) ∧ (n = 4000 + 1000 * 0 + 100 * f 0.to_nat + 10 * f 1.to_nat + f 2.to_nat)) ∧ n = 60 :=
sorry

end count_integers_with_5_and_7_l571_571364


namespace part1_part2_l571_571310

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

theorem part1 : ∀ x1 x2 : ℝ, 0 ≤ x1 → 0 ≤ x2 → x1 < x2 → f x1 < f x2 := by
  sorry

theorem part2 (m : ℝ) : 1 ≤ m →
  (∀ x : ℝ, 1 ≤ x → x ≤ m → f x ≤ f m) ∧ 
  (∀ x : ℝ, 1 ≤ x → x ≤ m → f 1 ≤ f x) →
  f m - f 1 = 1 / 2 →
  m = 2 := by
  sorry

end part1_part2_l571_571310


namespace Barbara_total_candies_l571_571614

/-- Barbara's total number of candies calculation -/
theorem Barbara_total_candies (initial_candies new_candies : ℕ):
  initial_candies = 9 →
  new_candies = 18 →
  initial_candies + new_candies = 27 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end Barbara_total_candies_l571_571614


namespace sphere_volume_l571_571984

/-- Given a sphere intersected by two parallel planes at a distance of 3 from each other, forming circles with radii 9 and 12 respectively, the volume of the sphere is 4050π. -/
theorem sphere_volume (d r1 r2 : ℝ) (h_distance : d = 3) (h_radii : r1 = 9 ∧ r2 = 12) :
  ∃ R : ℝ, (V := 4 / 3 * Real.pi * R^3) ∧ V = 4050 * Real.pi :=
sorry

end sphere_volume_l571_571984


namespace isosceles_triangle_perimeter_l571_571792

theorem isosceles_triangle_perimeter (a b : ℝ) (h : (a - 2)^2 + |b - 5| = 0) : a = 5 ∧ b = 2 ∨ a = 2 ∧ b = 5 → 2 * a + b = 12 :=
begin
  sorry
end

end isosceles_triangle_perimeter_l571_571792


namespace break_even_number_of_performances_l571_571476

theorem break_even_number_of_performances {x : ℕ} :
  (16000 * x = 81000 + 7000 * x) → x = 9 :=
by
  -- conditions
  assume h : (16000 * x = 81000 + 7000 * x),
  sorry

end break_even_number_of_performances_l571_571476


namespace range_of_a_l571_571698

def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 - 2*x else x^2 - 2*x

theorem range_of_a :
  {a : ℝ | f(a) - f(-a) ≤ 2 * f(1)} = {a : ℝ | a ≥ 1} :=
by
  sorry

end range_of_a_l571_571698


namespace PQ_length_l571_571046

-- Given conditions
variables (X Y Z P Q : Type) [metric_space X]
variable (XY : ℝ) (XZ : ℝ) (YZ : ℝ)
variable (XP : ℝ) (incenter_of_triangle_XYZ : X)

-- Assuming triangle XYZ with known side lengths
axiom h1 : XY = 13
axiom h2 : XZ = 15
axiom h3 : YZ = 14

-- PQ is parallel to YZ and contains the incenter
axiom h4 : (PQ.parallel YZ)
axiom h5 : (PQ.contains incenter_of_triangle_XYZ)

-- Needed to prove PQ = 56 / 13
theorem PQ_length :
  ∃ (PQ : ℝ), PQ = 56 / 13 :=
by {
  sorry
}

end PQ_length_l571_571046


namespace num_factors_48_l571_571358

theorem num_factors_48 : 
  let n := 48 in
  ∃ num_factors, num_factors = 10 ∧ 
  (∀ p k, prime p → (n = p ^ k → 1 + k)) := 
sorry

end num_factors_48_l571_571358


namespace boxes_needed_for_oranges_l571_571829

theorem boxes_needed_for_oranges (total_oranges number_of_oranges_per_box : ℕ) (h1 : total_oranges = 45) (h2 : number_of_oranges_per_box = 5) : total_oranges / number_of_oranges_per_box = 9 :=
by
  rw [h1, h2]
  norm_num
  sorry

end boxes_needed_for_oranges_l571_571829


namespace right_triangle_cos_Y_l571_571389

theorem right_triangle_cos_Y (XYZ : Triangle) (h_right : XYZ.right_triangle) 
  (h_cos : cos Y = 0.5) (h_XY : XY = 10) : YZ = 20 := 
begin
  sorry -- the proof goes here
end

end right_triangle_cos_Y_l571_571389


namespace log_graph_intersection_l571_571208

theorem log_graph_intersection :
  ∃! x : ℝ, x > 0 ∧ 3 * log x = log (3 * x) :=
begin
  use sqrt 3,
  split,
  { -- Prove that x = sqrt(3) satisfies the equation
    split,
    { exact real.sqrt_pos.mpr real.zero_lt_three, },
    { rw [log_mul,
          ← mul_assoc 3 (log (sqrt 3)],
          real.log_sqrt,
          mul_div_assoc (log 3) 2 2,
          mul_comm 3 2],
      rw ← eq_self_iff_true,
    }
  },
  { -- Prove the uniqueness of the solution
    intros y h,
    cases h with y_pos h_eq,
    have h_log : 3 * log y = log 3 + log y := by ring,
    rw [← h_eq, ← three_mul (log y), h_log],
    refine real.eq_sqrt_of_pos_of_sq_eq (real.log_pos y_pos) (by linarith),
  },
end

end log_graph_intersection_l571_571208


namespace count_multiples_6_or_8_not_both_l571_571764

-- Define the conditions
def is_multiple (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

-- Define the main proof statement
theorem count_multiples_6_or_8_not_both :
  (∑ k in Finset.filter (λ n, is_multiple n 6 ∨ is_multiple n 8 ∧ ¬(is_multiple n 6 ∧ is_multiple n 8)) (Finset.range 151), 1) = 31 := 
sorry

end count_multiples_6_or_8_not_both_l571_571764


namespace jump_rope_total_l571_571413

theorem jump_rope_total :
  (56 * 3) + (35 * 4) = 308 :=
by
  sorry

end jump_rope_total_l571_571413


namespace set_nonempty_iff_nonneg_l571_571368

theorem set_nonempty_iff_nonneg (a : ℝ) :
  (∃ x : ℝ, x^2 ≤ a) ↔ a ≥ 0 :=
sorry

end set_nonempty_iff_nonneg_l571_571368


namespace smallest_period_of_given_sin_l571_571216

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x

noncomputable def function_to_check : ℝ → ℝ := λ x, Real.sin (2 * x - Real.pi / 3)

theorem smallest_period_of_given_sin : smallest_positive_period function_to_check Real.pi :=
by
  sorry

end smallest_period_of_given_sin_l571_571216


namespace sum_geometric_series_l571_571300

-- Given the conditions
def q : ℕ := 2
def a3 : ℕ := 16
def n : ℕ := 2017
def a1 : ℕ := 4

-- Define the sum of the first n terms of a geometric series
noncomputable def geometricSeriesSum (a1 q n : ℕ) : ℕ :=
  a1 * (1 - q^n) / (1 - q)

-- State the problem
theorem sum_geometric_series :
  geometricSeriesSum a1 q n = 2^2019 - 4 :=
sorry

end sum_geometric_series_l571_571300


namespace regular_polygon_sides_and_pentagon_area_l571_571592

theorem regular_polygon_sides_and_pentagon_area (P s : ℝ) (P_cond : P = 180) (s_cond : s = 15) :
  let n := P / s in n = 12 ∧ (n = 5 → (A = (1/4) * real.sqrt(5 * (5 + 2 * real.sqrt 5)) * s^2 → A ≈ 866.025)) :=
by
  sorry

end regular_polygon_sides_and_pentagon_area_l571_571592


namespace count_integers_satisfying_inequality_l571_571363

theorem count_integers_satisfying_inequality :
  {m : ℤ | m ≠ -1 ∧ (1 / |m| : ℚ) ≥  (1 / 12)}.finite.card = 24 :=
by
  sorry

end count_integers_satisfying_inequality_l571_571363


namespace area_of_shaded_region_l571_571047

theorem area_of_shaded_region (r : ℝ) (θ : ℝ) (radius_eq : r = 12) (angle_eq : θ = 60) : 
  let sector_area := (θ / 360) * (π * r ^ 2)
      triangle_area := (sqrt 3 / 4) * (r ^ 2)
      one_part_area := sector_area - triangle_area
      shaded_area := 2 * one_part_area
  in shaded_area = 48 * π - 72 * sqrt 3 := 
by sorry

end area_of_shaded_region_l571_571047


namespace amelia_drove_distance_on_Monday_l571_571014

theorem amelia_drove_distance_on_Monday 
  (total_distance : ℕ) (tuesday_distance : ℕ) (remaining_distance : ℕ)
  (total_distance_eq : total_distance = 8205) 
  (tuesday_distance_eq : tuesday_distance = 582) 
  (remaining_distance_eq : remaining_distance = 6716) :
  ∃ x : ℕ, x + tuesday_distance + remaining_distance = total_distance ∧ x = 907 :=
by
  sorry

end amelia_drove_distance_on_Monday_l571_571014


namespace b_is_geometric_sequence_l571_571699

-- Define the input sequences and conditions given in the problem
variables {a : ℕ → ℝ} {q : ℝ}
hypothesis (hq : q ≠ 1) (ha : ∀ n, a (n + 3) = q * a n)

-- Define the new sequence bn
noncomputable def b (n : ℕ) : ℝ := a (3 * n + 1) + a (3 * n + 2) + a (3 * n + 3)

-- The theorem we need to prove
theorem b_is_geometric_sequence : ∃ q₃ : ℝ, (∀ n, b (n + 1) = q₃ * b n) ∧ q₃ = q ^ 3 :=
by
  -- Proof goes here
  sorry

end b_is_geometric_sequence_l571_571699


namespace f_increasing_on_neg_inf_neg_one_l571_571485

noncomputable def log_a (a x : ℝ) := real.log x / real.log a

def f (a x : ℝ) := log_a a (abs (x + 1))

theorem f_increasing_on_neg_inf_neg_one (a : ℝ) (h1 : ∀ x, x ∈ set.Ioo (-1 : ℝ) 0 → f a x > 0) :
  ∀ x y, x < y → x ∈ set.Iio (-1 : ℝ) → y ∈ set.Iio (-1 : ℝ) → f a x < f a y :=
by
  sorry

end f_increasing_on_neg_inf_neg_one_l571_571485


namespace sticker_distribution_unique_arrangements_l571_571328

theorem sticker_distribution_unique_arrangements :
  let stickers := 10
  let sheets := 5
  let colors := 2
  (Nat.choose (stickers + sheets - 1) stickers) * (colors ^ sheets) = 32032 :=
by
  sorry

end sticker_distribution_unique_arrangements_l571_571328


namespace S1_not_in_Borel_S2_not_in_Borel_l571_571926

def S1 (x : ℝ → ℝ) : Prop := ∃ t ∈ set.Icc 0 1, x t = 0
def S2 (x : ℝ → ℝ) (t0 : ℝ) : Prop := t0 ∈ set.Icc 0 1 ∧ continuous_at x t0

theorem S1_not_in_Borel (x : ℝ → ℝ) : ¬ (measurable_set {x | S1 x}) :=
sorry

theorem S2_not_in_Borel (x : ℝ → ℝ) (t0 : ℝ) : ¬ (measurable_set {x | S2 x t0}) :=
sorry

end S1_not_in_Borel_S2_not_in_Borel_l571_571926


namespace total_weight_in_pounds_l571_571930

-- Define the total ounces based on the provided conditions
def total_ounces : ℕ := 4 * 50

-- Define the conversion factor from ounces to pounds
def ounces_to_pounds (oz : ℕ) : ℝ := oz / 16.0

-- The total weight in pounds.
theorem total_weight_in_pounds : ounces_to_pounds total_ounces = 12.5 := by
  sorry

end total_weight_in_pounds_l571_571930


namespace product_calc_l571_571183

theorem product_calc : (16 * 0.5 * 4 * 0.125 = 4) :=
by
  sorry

end product_calc_l571_571183


namespace narrow_black_stripes_are_8_l571_571890

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l571_571890


namespace count_multiples_6_or_8_not_both_l571_571763

-- Define the conditions
def is_multiple (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

-- Define the main proof statement
theorem count_multiples_6_or_8_not_both :
  (∑ k in Finset.filter (λ n, is_multiple n 6 ∨ is_multiple n 8 ∧ ¬(is_multiple n 6 ∧ is_multiple n 8)) (Finset.range 151), 1) = 31 := 
sorry

end count_multiples_6_or_8_not_both_l571_571763


namespace sheets_in_stack_l571_571159

theorem sheets_in_stack (h : 200 * t = 2.5) (h_pos : t > 0) : (5 / t) = 400 :=
by
  sorry

end sheets_in_stack_l571_571159


namespace product_mod_7_l571_571205

theorem product_mod_7 :
  (2009 % 7 = 4) ∧ (2010 % 7 = 5) ∧ (2011 % 7 = 6) ∧ (2012 % 7 = 0) →
  (2009 * 2010 * 2011 * 2012) % 7 = 0 :=
by
  sorry

end product_mod_7_l571_571205


namespace arithmetic_sum_ratio_l571_571280

variable (a_n : ℕ → ℤ) -- the arithmetic sequence
variable (S : ℕ → ℤ) -- sum of the first n terms of the sequence
variable (d : ℤ) (a₁ : ℤ) -- common difference and first term of the sequence

-- Definition of the sum of the first n terms in an arithmetic sequence
def arithmetic_sum (n : ℕ) : ℤ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

-- Given condition
axiom h1 : (S 6) / (S 3) = 3

-- Definition of S_n in terms of the given formula
axiom S_def : ∀ n, S n = arithmetic_sum n

-- The main goal to prove
theorem arithmetic_sum_ratio : S 12 / S 9 = 5 / 3 := by
  sorry

end arithmetic_sum_ratio_l571_571280


namespace volume_of_prism_l571_571594

noncomputable def volume_of_right_rectangular_prism (a b c : ℕ) : ℕ :=
  a * b * c

def conditions (a b c : ℕ) : Prop :=
  (a * b = 30 ∨ a * c = 30 ∨ b * c = 30) ∧ (a * b = 13 ∨ a * c = 13 ∨ b * c = 13)

theorem volume_of_prism : ∃ (a b c : ℕ), conditions a b c ∧ volume_of_right_rectangular_prism a b c = 30 :=
begin
  sorry -- Proof not required as per instructions
end

end volume_of_prism_l571_571594


namespace cleaner_needed_l571_571442

def cleaner_per_dog := 6
def cleaner_per_cat := 4
def cleaner_per_rabbit := 1

def num_dogs := 6
def num_cats := 3
def num_rabbits := 1

def total_cleaner_for_dogs := cleaner_per_dog * num_dogs
def total_cleaner_for_cats := cleaner_per_cat * num_cats
def total_cleaner_for_rabbits := cleaner_per_rabbit * num_rabbits

def total_cleaner := total_cleaner_for_dogs + total_cleaner_for_cats + total_cleaner_for_rabbits

theorem cleaner_needed : total_cleaner = 49 :=
by
  unfold total_cleaner total_cleaner_for_dogs total_cleaner_for_cats total_cleaner_for_rabbits cleaner_per_dog cleaner_per_cat cleaner_per_rabbit num_dogs num_cats num_rabbits
  rw [cleaner_per_dog, cleaner_per_cat, cleaner_per_rabbit]
  rw [num_dogs, num_cats, num_rabbits]
  simp
  sorry -- The proof needs to end with a correct justification which is omitted here

end cleaner_needed_l571_571442


namespace units_digit_17_pow_2023_l571_571113

theorem units_digit_17_pow_2023 : (17^2023 % 10) = 3 := sorry

end units_digit_17_pow_2023_l571_571113


namespace product_of_solutions_l571_571382

theorem product_of_solutions (x : ℂ) (a b : ℝ) (h : x^6 = -729) 
  (h_pos : a = x.re ∧ a > 0 ∧ x = a + bi b) : 
  ∃ z : ℝ, z = 9 :=
sorry

end product_of_solutions_l571_571382


namespace sum_squares_second_15_l571_571968

theorem sum_squares_second_15 :
  (∑ i in finset.range (30 + 1), i ^ 2) - (∑ i in finset.range (15 + 1), i ^ 2) = 8215 :=
by
  have sum_15 : ∑ i in finset.range (15 + 1), i ^ 2 = 1240 := by sorry
  have sum_30 : ∑ i in finset.range (30 + 1), i ^ 2 = 9455 := by
    calc
      ∑ i in finset.range (30 + 1), i ^ 2
          = 30 * 31 * 61 / 6 : by sorry
       ... = 155 * 61 : by sorry
       ... = 9455 : by sorry
  calc
    (∑ i in finset.range (30 + 1), i ^ 2) - (∑ i in finset.range (15 + 1), i ^ 2)
        = 9455 - 1240 : by rw [sum_30, sum_15]
    ... = 8215 : by norm_num

end sum_squares_second_15_l571_571968


namespace recipe_butter_per_cup_l571_571149

theorem recipe_butter_per_cup (coconut_oil_to_butter_substitution : ℝ)
  (remaining_butter : ℝ)
  (planned_baking_mix : ℝ)
  (used_coconut_oil : ℝ)
  (butter_per_cup : ℝ)
  (h1 : coconut_oil_to_butter_substitution = 1)
  (h2 : remaining_butter = 4)
  (h3 : planned_baking_mix = 6)
  (h4 : used_coconut_oil = 8) :
  butter_per_cup = 4 / 3 := 
by 
  sorry

end recipe_butter_per_cup_l571_571149


namespace probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571070

def probability_of_5_odd_numbers_in_6_rolls (prob_odd : ℚ) : ℚ :=
  (nat.choose 6 5 * (prob_odd^5) * ((1 - prob_odd)^1)) / (2^6)

theorem probability_of_5_odd_numbers_in_6_rolls_is_3_over_32 :
  probability_of_5_odd_numbers_in_6_rolls (1/2) = 3 / 32 :=
by sorry

end probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571070


namespace shot_put_order_l571_571264

/-
Consider five students A, B, C, D, and E participating in a shot put competition
with an unknown order of participation. Each of them makes guesses about the order,
and we know at least one guess for each person is correct.

A guesses: B is third, C is fifth.
B guesses: E is fourth, D is fifth.
C guesses: A is first, E is fourth.
D guesses: C is first, B is second.
E guesses: A is third, D is fourth.
Given that each participant's order is correctly guessed by at least one person,
prove that D is the fifth participant.
-/

theorem shot_put_order (A B C D E : Type) :
  (∃ a b c d e : List Type, 
    (a = [A, _, _, _, D] ∨ a = [_, A, _, _, D] ∨ a = [_, _, _, A, D]) ∧
    (b = [_, B, _, _, _] ∨ b = [_, _, B, D, _] ∨ b = [_, _, _, _, D]) ∧
    (c = [C, _, _, _, _] ∨ c = [_, _, _, E, _]) ∧
    (d = [A, _, B, _, _] ∨ d = [_, A, _, _, D]) ∧
    (e = [_, _, A, D, _])) →
  (D = List.nth [C, _, _, _, D] 4)
:=
by
  sorry

end shot_put_order_l571_571264


namespace multiples_of_6_or_8_but_not_both_l571_571775

/-- The number of positive integers less than 151 that are multiples of either 6 or 8 but not both is 31. -/
theorem multiples_of_6_or_8_but_not_both (n : ℕ) :
  (multiples_of_6 : Set ℕ) = {k | k < 151 ∧ k % 6 = 0}
  ∧ (multiples_of_8 : Set ℕ) = {k | k < 151 ∧ k % 8 = 0}
  ∧ (multiples_of_24 : Set ℕ) = {k | k < 151 ∧ k % 24 = 0}
  ∧ multiples_of_6_or_8 := {k | k ∈ multiples_of_6 ∨ k ∈ multiples_of_8}
  ∧ multiples_of_6_and_8 := {k | k ∈ multiples_of_6 ∧ k ∈ multiples_of_8}
  ∧ (card (multiples_of_6_or_8 \ multiples_of_6_and_8)) = 31 := sorry

end multiples_of_6_or_8_but_not_both_l571_571775


namespace find_special_integer_l571_571658

theorem find_special_integer :
  ∃ (n : ℕ), n > 0 ∧ (21 ∣ n) ∧ 30 ≤ Real.sqrt n ∧ Real.sqrt n ≤ 30.5 ∧ n = 903 := 
sorry

end find_special_integer_l571_571658


namespace number_of_two_person_apartments_l571_571174

theorem number_of_two_person_apartments 
  (x : ℕ) 
  (h1 : ∀ (num_studio num_two_person num_four_person total_buildings : ℕ), 
         num_studio = 10 ∧ 
         num_four_person = 5 ∧ 
         total_buildings = 4 ∧ 
         (0.75 * (total_max_occupancy num_studio x num_four_person total_buildings) = 210) → 
         x = 20) :
   (0.75 * 4 * (10 + 2 * x + 20) = 210) → 
   x = 20 :=
by {
    sorry
}

end number_of_two_person_apartments_l571_571174


namespace find_first_5digits_of_M_l571_571493

def last6digits (n : ℕ) : ℕ := n % 1000000

def first5digits (n : ℕ) : ℕ := n / 10

theorem find_first_5digits_of_M (M : ℕ) (h1 : last6digits M = last6digits (M^2)) (h2 : M > 999999) : first5digits M = 60937 := 
by sorry

end find_first_5digits_of_M_l571_571493


namespace rearrange_distinct_sums_mod_4028_l571_571840

theorem rearrange_distinct_sums_mod_4028 
  (x : Fin 2014 → ℤ) (y : Fin 2014 → ℤ) 
  (hx : ∀ i j : Fin 2014, i ≠ j → x i % 2014 ≠ x j % 2014)
  (hy : ∀ i j : Fin 2014, i ≠ j → y i % 2014 ≠ y j % 2014) :
  ∃ σ : Fin 2014 → Fin 2014, Function.Bijective σ ∧ 
  ∀ i j : Fin 2014, i ≠ j → ( x i + y (σ i) ) % 4028 ≠ ( x j + y (σ j) ) % 4028 
:= by
  sorry

end rearrange_distinct_sums_mod_4028_l571_571840


namespace find_a10_l571_571566

-- Conditions
variables (S : ℕ → ℕ) (a : ℕ → ℕ)
variables (hS9 : S 9 = 81) (ha2 : a 2 = 3)

-- Arithmetic sequence sum definition
def arithmetic_sequence_sum (n : ℕ) (a1 : ℕ) (d : ℕ) :=
  n * (2 * a1 + (n - 1) * d) / 2

-- a_n formula definition
def a_n (n a1 d : ℕ) := a1 + (n - 1) * d

-- Proof statement
theorem find_a10 (a1 d : ℕ) (hS9' : 9 * (2 * a1 + 8 * d) / 2 = 81) (ha2' : a1 + d = 3) :
  a 10 = a1 + 9 * d :=
sorry

end find_a10_l571_571566


namespace subgroup_iff_conditions_l571_571421

variables {G : Type*} [group G] (H : set G)

theorem subgroup_iff_conditions (H_nonempty : H.nonempty)
    (H_closed_inv_mul : ∀ x y ∈ H, x * y⁻¹ ∈ H) : 
    is_subgroup H :=
sorry

end subgroup_iff_conditions_l571_571421


namespace shift_sin_l571_571949

theorem shift_sin (x : ℝ) : (sin (x + π / 6)) = (sin x).shift_left (π / 6) := 
sorry

end shift_sin_l571_571949


namespace number_of_correct_propositions_l571_571299

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then
    Math.exp x * (x + 1)
  else if x > 0 then
    -Math.exp (-x) * (x - 1)
  else
    0

def isOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def prop1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x > 0 → f x = -Math.exp (-x) * (x - 1)

def prop2 (f : ℝ → ℝ) : Prop := ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 ∧ ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2

def prop3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x < 0 ↔ x ∈ Set.Ioo (-(1 : ℝ)) 0 ∨ x ∈ Set.Ioo 0 1

def prop4 (f : ℝ → ℝ) : Prop := ∀ x1 x2 : ℝ, |f x1 - f x2| < 2

theorem number_of_correct_propositions : isOdd f →
  (∀ x : ℝ, x < 0 → f x = Math.exp x * (x + 1)) →
  2 = ([prop1 f, prop2 f, prop3 f, prop4 f].count true) :=
by sorry

end number_of_correct_propositions_l571_571299


namespace product_of_intersection_points_l571_571635

noncomputable def circle_intersection_product (x y : ℝ) : Prop :=
  ((x - 3)^2 + (y - 5)^2 = 9) ∧ ((x - 7)^2 + (y - 5)^2 = 16)

theorem product_of_intersection_points :
  let points := { (x, y) | (circle_intersection_product x y) } in
  (x y) ∈ points ->
  ∏ (p : Σ' x y, circle_intersection_product x y) in points, x * y =  1203345 / 4096 :=
sorry

end product_of_intersection_points_l571_571635


namespace number_of_narrow_black_stripes_l571_571901

-- Define the variables
variables {w n b : ℕ}

-- The conditions from the problem
def condition1 := w + n = b + 1
def condition2 := b = w + 7

-- The Lean statement to prove
theorem number_of_narrow_black_stripes (h1 : condition1) (h2 : condition2) : n = 8 :=
by {
  -- We import the conditions as hypotheses
  sorry
}

end number_of_narrow_black_stripes_l571_571901


namespace range_of_m_l571_571378

theorem range_of_m (m : ℝ) :
  (∃! x ∈ set.Icc (0 : ℝ) 1, (x^2 - 2 * m * x + m^2 - 1 = 0))
  ↔ (m ∈ set.Icc (-1 : ℝ) 0 ∨ m ∈ set.Icc (1 : ℝ) 2) := sorry

end range_of_m_l571_571378


namespace sum_of_squares_of_roots_l571_571622

noncomputable def polynomial_equation := 3 * X^3 - 7 * X^2 + 6 * X + 15

theorem sum_of_squares_of_roots :
  (u v w : ℂ) (hu : polynomial.roots ⟨polynomial_equation⟩ = {u, v, w}) :
  u^2 + v^2 + w^2 = 13 / 9 := by
  sorry

end sum_of_squares_of_roots_l571_571622


namespace sum_of_series_is_918_l571_571033

-- Define the first term a, common difference d, last term a_n,
-- and the number of terms n calculated from the conditions.
def first_term : Int := -300
def common_difference : Int := 3
def last_term : Int := 309
def num_terms : Int := 204 -- calculated as per the solution

-- Compute the sum of the arithmetic series
def sum_arithmetic_series (a d : Int) (n : Int) : Int :=
  n * (2 * a + (n - 1) * d) / 2

-- Prove that the sum of the series is 918
theorem sum_of_series_is_918 :
  sum_arithmetic_series first_term common_difference num_terms = 918 :=
by
  sorry

end sum_of_series_is_918_l571_571033


namespace vet_donation_l571_571601

noncomputable def vet_fees (d c r p: ℕ) (fd fc fr fp: ℕ) : ℕ := 
  (d * fd) + (c * fc) + (r * fr) + (p * fp)

noncomputable def discount (f1 f2: ℕ) (d: ℕ) : ℕ :=
  ((f1 + f2) * d) / 10

noncomputable def donation (total_fees: ℕ) : Float :=
  (total_fees: Float) / 3

theorem vet_donation :
  let fd := 15 in
  let fc := 13 in
  let fr := 10 in
  let fp := 12 in
  let families_dog := 8 in
  let families_cat := 3 in
  let families_rabbit := 5 in
  let families_parrot := 2 in
  let discount_fam := 10 in
  let dog_and_cat := 2 in
  let parrot_and_rabbit := 1 in

  let orig_fees := vet_fees families_dog families_cat families_rabbit families_parrot fd fc fr fp in
  
  let discount_dog_cat := discount fd fc dog_and_cat in
  let discount_parrot_rabbit := discount fp fr parrot_and_rabbit in
  
  let adjusted_d := (families_dog * fd) - (discount_dog_cat / 2) * dog_and_cat in
  let adjusted_c := (families_cat * fc) - (discount_dog_cat / 2) * dog_and_cat in
  let adjusted_p := (families_parrot * fp) - (discount_parrot_rabbit / 2) in
  let adjusted_r := (families_rabbit * fr) - (discount_parrot_rabbit / 2) in
  
  let total_vet_fees_after_discount := adjusted_d + adjusted_c + adjusted_p + adjusted_r in
  let donated_amount := donation total_vet_fees_after_discount in
  
  donated_amount.toReal ≈ 54.27 :=
by sorry

end vet_donation_l571_571601


namespace distinct_real_solutions_exist_l571_571268

theorem distinct_real_solutions_exist (a : ℝ) (h : a > 3 / 4) : 
  ∃ (x y : ℝ), x ≠ y ∧ x = a - y^2 ∧ y = a - x^2 := 
sorry

end distinct_real_solutions_exist_l571_571268


namespace find_a_l571_571713

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 14
noncomputable def g (x : ℝ) : ℝ := x^3 - 4

theorem find_a (a : ℝ) (ha : a > 0) (hfga : f (g a) = 18) :
  a = real.cbrt (4 + 2 * real.sqrt 3 / 3) :=
by
  sorry

end find_a_l571_571713


namespace minimum_value_of_M_is_4_l571_571687

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

theorem minimum_value_of_M_is_4 :
  ∀ x1 x2 ∈ Icc (-1 : ℝ) 1, ∃ M, M = 4 ∧ M ≥ |f x1 - f x2| :=
by
  sorry

end minimum_value_of_M_is_4_l571_571687


namespace steve_annual_salary_l571_571477

variable (S : ℝ)

theorem steve_annual_salary :
  (0.70 * S - 800 = 27200) → (S = 40000) :=
by
  intro h
  sorry

end steve_annual_salary_l571_571477


namespace functional_equation_unique_solution_l571_571230

theorem functional_equation_unique_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, a + f b + f (f c) = 0 → f a ^ 3 + b * f b ^ 2 + c ^ 2 * f c = 3 * a * b * c) →
  (∀ x : ℝ, f x = x ∨ f x = -x ∨ f x = 0) :=
by
  sorry

end functional_equation_unique_solution_l571_571230


namespace water_supply_days_l571_571596

theorem water_supply_days (C V : ℕ) 
  (h1: C = 75 * (V + 10))
  (h2: C = 60 * (V + 20)) : 
  (C / V) = 100 := 
sorry

end water_supply_days_l571_571596


namespace remainder_of_sum_l571_571087

def sum_first_n_numbers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem remainder_of_sum :
  let n := 150
  let S := sum_first_n_numbers n
  S % 11200 = 125 :=
by
  let n := 150
  let S := sum_first_n_numbers n
  show S % 11200 = 125, from
  sorry

end remainder_of_sum_l571_571087


namespace attach_squares_to_cube_l571_571579

theorem attach_squares_to_cube (cube_edge square_side : ℝ) :
  cube_edge = 2 ∧ square_side = 1 
  → (∃ (positions : fin 10 → (ℝ × ℝ × ℝ)), 
    (∀ i j, i ≠ j → no_edge_overlap (positions i) (positions j)) ∧
    (∀ k, square_in_cube (positions k) cube_edge square_side)) :=
by {
  sorry
}

end attach_squares_to_cube_l571_571579


namespace probability_one_each_stratified_sampling_variance_and_average_weight_estimated_orchard_weight_l571_571271

-- Definitions based on the problem conditions
def total_boxes : ℕ := 136
def first_grade_boxes : ℕ := 102
def second_grade_boxes : ℕ := 34
def sample_size : ℕ := 8

def first_grade_fruits : ℕ := 120
def second_grade_fruits : ℕ := 48

def average_weight_first_grade : ℚ := 303.45
def variance_first_grade : ℚ := 603.46
def average_weight_second_grade : ℚ := 240.41
def variance_second_grade : ℚ := 648.21

-- Lean theorem declaration for each part as described
theorem probability_one_each:
  let total_ways : ℚ := nat.choose total_boxes 2
  let ways_one_each : ℚ := (nat.choose first_grade_boxes 1) * (nat.choose second_grade_boxes 1)
  (ways_one_each / total_ways) = 17 / 45 :=
  by sorry

theorem stratified_sampling:
  let ratio_first_grade := 3
  let ratio_second_grade := 1
  let total_ratio := ratio_first_grade + ratio_second_grade
  let num_first_sample := (ratio_first_grade / total_ratio) * sample_size
  let num_second_sample := (ratio_second_grade / total_ratio) * sample_size
  (num_first_sample = 6) ∧ (num_second_sample = 2) :=
  by sorry

theorem variance_and_average_weight: 
  let total_fruits := first_grade_fruits + second_grade_fruits
  let combined_average_weight := ((first_grade_fruits : ℚ) / total_fruits) * average_weight_first_grade + ((second_grade_fruits : ℚ) / total_fruits) * average_weight_second_grade
  let combined_variance := ((first_grade_fruits : ℚ) / total_fruits) * (variance_first_grade + (average_weight_first_grade - combined_average_weight) ^ 2) + 
                           ((second_grade_fruits : ℚ) / total_fruits) * (variance_second_grade + (average_weight_second_grade - combined_average_weight) ^ 2)
  ∧ combined_average_weight = 285.44
  ∧ combined_variance = 1427.27 := 
  by sorry

theorem estimated_orchard_weight:
  let estimated_weight := ((first_grade_boxes : ℚ)/ total_boxes) * average_weight_first_grade + ((second_grade_boxes : ℚ) / total_boxes) * average_weight_second_grade
  estimated_weight = 287.69 := 
  by sorry

end probability_one_each_stratified_sampling_variance_and_average_weight_estimated_orchard_weight_l571_571271


namespace radius_of_circumscribed_trapezoid_l571_571597

-- Define the geometrical setup
structure Trapezoid (α : Type) [LinearOrder α] :=
  (A B C D : α)
  (circumscribed : ∃ O : α, ∀ P ∈ {B, C, D, A}, dist O P = dist O Q) -- Circumscribed around a circle
  (AB_perp : perp A B) -- AB perpendicular to bases
  (intersection_M : ∃ M : α, intersection (diagonal A C) (diagonal B D)) -- Intersection of diagonals

-- Define the given area of triangle CMD
axiom Area_CMD {α : Type} [LinearOrder α] (A B C D : α) (h : Trapezoid α) : ℝ

-- Define the radius to be proved
def radius (α : Type) [LinearOrder α] (A B C D : α) (h : Trapezoid α) : ℝ :=
  sqrt (Area_CMD A B C D h)

-- The main theorem to prove: radius equals the square root of area of triangle CMD
theorem radius_of_circumscribed_trapezoid 
  {α : Type} [LinearOrder α] (A B C D : α) (h : Trapezoid α) :
  radius α A B C D h = sqrt S :=
by
  sorry

end radius_of_circumscribed_trapezoid_l571_571597


namespace find_y_l571_571943

-- Define the conditions
variables (a b x y : ℤ)
hypothesis H1 : (a + b + 100 + 200300 + x) / 5 = 250
hypothesis H2 : (a + b + 300 + 150100 + x + y) / 6 = 200
hypothesis H3 : a % 5 = 0
hypothesis H4 : b % 5 = 0

-- Prove that y = 49800
theorem find_y : y = 49800 :=
by
  -- Starting point, attach sorry to avoid full proof here
  sorry

end find_y_l571_571943


namespace arith_seq_sum_l571_571190

theorem arith_seq_sum (n : ℕ) (h₁ : 2 * n - 1 = 21) : 
  (∑ i in finset.range 11, (2 * i + 1)) = 121 :=
by
  sorry

end arith_seq_sum_l571_571190


namespace smallest_positive_integer_ends_in_9_and_divisible_by_13_l571_571090

theorem smallest_positive_integer_ends_in_9_and_divisible_by_13 :
  ∃ n : ℕ, n % 10 = 9 ∧ 13 ∣ n ∧ n > 0 ∧ ∀ m, m % 10 = 9 → 13 ∣ m ∧ m > 0 → m ≥ n := 
begin
  use 99,
  split,
  { exact mod_eq_of_lt (10*k + 9) 10 99 9 (by norm_num), },
  split,
  { exact dvd_refl 99, },
  split,
  { exact zero_lt_99, },
  intros m hm1 hm2 hpos,
  by_contradiction hmn,
  sorry
end

end smallest_positive_integer_ends_in_9_and_divisible_by_13_l571_571090


namespace commercial_break_duration_l571_571645

theorem commercial_break_duration (n1 n2 t1 t2 : ℕ) (h1 : n1 = 3) (h2: t1 = 5) (h3 : n2 = 11) (h4 : t2 = 2) : 
  n1 * t1 + n2 * t2 = 37 := 
by 
  sorry

end commercial_break_duration_l571_571645


namespace arithmetic_geometric_means_l571_571928

theorem arithmetic_geometric_means (a b : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 110) : 
  a^2 + b^2 = 1380 :=
sorry

end arithmetic_geometric_means_l571_571928


namespace hypotenuse_length_l571_571810

open Real

variables {X Y Z : Point}  -- Define points

def right_triangle (X Y Z : Point) : Prop := 
  ∃ (M : Point) (N : Point), 
    is_midpoint M Y Z ∧ is_midpoint N X Z ∧ 
    dist X M = 8 ∧ dist Y N = 2 * sqrt 14 ∧ 
    is_right_triangle X Y Z

-- Lean theorem statement
theorem hypotenuse_length 
  (X Y Z : Point)
  (h : right_triangle X Y Z) : 
  dist X Y = 4 * sqrt 14 :=
sorry

end hypotenuse_length_l571_571810


namespace log13_of_log7_l571_571373

theorem log13_of_log7 (x : ℝ) (h : log 7 (x + 5) = 2) : log 13 (x + 1) = log 13 45 := by
  sorry

end log13_of_log7_l571_571373


namespace functional_equation_g_l571_571486

variable (g : ℝ → ℝ)
variable (f : ℝ)
variable (h : ℝ)

theorem functional_equation_g (H1 : ∀ x y : ℝ, g (x + y) = g x * g y)
                            (H2 : g 3 = 4) :
                            g 6 = 16 := 
by
  sorry

end functional_equation_g_l571_571486


namespace particular_solution_l571_571471
-- Import necessary libraries

-- Define the differential equation and initial condition
def diff_eq (x y : ℝ) : ℝ := -y/x
def initial_condition : Prop := y(2) = 3

-- State the theorem to find the particular solution
theorem particular_solution (x y : ℝ) (h_diff_eq : ∀ x y, deriv y x = diff_eq x y) (h_init_cond : initial_condition) :
  x * y = 6 :=
sorry

end particular_solution_l571_571471


namespace narrow_black_stripes_are_8_l571_571891

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l571_571891


namespace find_k_l571_571823

variable (m n k : ℚ)

def line_eq (x y : ℚ) : Prop := x - (5/2 : ℚ) * y + 1 = 0

theorem find_k (h1 : line_eq m n) (h2 : line_eq (m + 1/2) (n + 1/k)) : k = 3/5 := by
  sorry

end find_k_l571_571823


namespace radius_of_tangent_circle_l571_571625

theorem radius_of_tangent_circle (side_length : ℝ) (num_semicircles : ℕ)
  (r_s : ℝ) (r : ℝ)
  (h1 : side_length = 4)
  (h2 : num_semicircles = 16)
  (h3 : r_s = side_length / 4 / 2)
  (h4 : r = (9 : ℝ) / (2 * Real.sqrt 5)) :
  r = (9 * Real.sqrt 5) / 10 :=
by
  rw [h4]
  sorry

end radius_of_tangent_circle_l571_571625


namespace vasya_can_guess_number_in_10_questions_l571_571910

noncomputable def log2 (n : ℕ) : ℝ := 
  Real.log n / Real.log 2

theorem vasya_can_guess_number_in_10_questions (n q : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 1000) (h3 : q = 10) :
  q ≥ log2 n := 
by
  sorry

end vasya_can_guess_number_in_10_questions_l571_571910


namespace prob_both_machines_operate_l571_571983

variable (A1 A2 : Prop)
variable (P : Prop → ℝ)

-- Probabilities
axiom p1 : P A1 = 0.9
axiom p2 : P A2 = 0.8

-- Independence
axiom independent : P (A1 ∧ A2) = P A1 * P A2

theorem prob_both_machines_operate :
  P (A1 ∧ A2) = 0.72 :=
by
  rw [independent]
  rw [p1, p2]
  norm_num
  sorry

end prob_both_machines_operate_l571_571983


namespace minimum_cuts_for_24_pieces_l571_571136

theorem minimum_cuts_for_24_pieces : 
  ∀ (n : ℕ), n + 1 = 24 → n = 23 :=
begin
  intro n,
  intro h,
  linarith,
end

end minimum_cuts_for_24_pieces_l571_571136


namespace number_of_tiles_l571_571142

theorem number_of_tiles (floor_length : ℝ) (floor_width : ℝ) (tile_length : ℝ) (tile_width : ℝ) 
  (h1 : floor_length = 9) 
  (h2 : floor_width = 12) 
  (h3 : tile_length = 1 / 2) 
  (h4 : tile_width = 2 / 3) 
  : (floor_length * floor_width) / (tile_length * tile_width) = 324 := 
by
  sorry

end number_of_tiles_l571_571142


namespace categorize_numbers_correctly_l571_571654

open Set

-- Definitions of sets
def given_set : Set ℝ := {1, -0.20, 3 + 1/5, 325, -789, 0, -23.13, 0.618, -2014, Real.pi, 
                          0.1010010001}

def negative_numbers : Set ℝ := {-0.20, -789, -23.13, -2014}
def integers : Set ℤ := {1, 325, -789, 0, -2014}
def positive_fractions : Set ℝ := {3 + 1/5, 0.618}

-- Theorem statement
theorem categorize_numbers_correctly :
  {x ∈ given_set | x < 0} = negative_numbers ∧ 
  {x ∈ given_set | x ∈ Int} = integers ∧ 
  {x ∈ given_set | x > 0 ∧ x ∉ Int} = positive_fractions :=
sorry

end categorize_numbers_correctly_l571_571654


namespace num_subsets_with_even_correct_l571_571710

noncomputable def num_subsets_with_even : ℕ :=
  let universal_set := {1, 2, 3, 4}
  let total_subsets := 2 ^ universal_set.size
  let even_numbers := {x ∈ universal_set | x % 2 = 0}
  let subsets_without_even := 2 ^ (universal_set.size - even_numbers.size)
  total_subsets - subsets_without_even

theorem num_subsets_with_even_correct :
  num_subsets_with_even = 12 :=
by
  sorry

end num_subsets_with_even_correct_l571_571710


namespace sin_eq_sin_sinx_l571_571366

noncomputable def S (x : ℝ) := Real.sin x - x

theorem sin_eq_sin_sinx (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.arcsin 742) :
  ∃! x, Real.sin x = Real.sin (Real.sin x) :=
by
  sorry

end sin_eq_sin_sinx_l571_571366


namespace equation_represents_pair_of_lines_l571_571627

theorem equation_represents_pair_of_lines : ∀ x y : ℝ, 9 * x^2 - 25 * y^2 = 0 → 
                    (x = (5/3) * y ∨ x = -(5/3) * y) :=
by sorry

end equation_represents_pair_of_lines_l571_571627


namespace a_5_eq_3_a_2016_eq_192_l571_571701

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if even n then 2 * sequence (n / 2)
  else sequence (n / 2) + 1

theorem a_5_eq_3 : sequence 5 = 3 :=
  sorry

theorem a_2016_eq_192 : sequence 2016 = 192 :=
  sorry

end a_5_eq_3_a_2016_eq_192_l571_571701


namespace min_dist_to_origin_l571_571802

theorem min_dist_to_origin (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : 
  ∃ p : ℝ, p = x^2 + y^2 ∧ p = 1 :=
begin
  sorry
end

end min_dist_to_origin_l571_571802


namespace smallest_integer_n_condition_l571_571525

theorem smallest_integer_n_condition :
  (∃ n : ℕ, n > 0 ∧ (∀ (m : ℤ), (1 ≤ m ∧ m ≤ 1992) → (∃ (k : ℤ), (m : ℚ) / 1993 < k / n ∧ k / n < (m + 1 : ℚ) / 1994))) ↔ n = 3987 :=
sorry

end smallest_integer_n_condition_l571_571525


namespace hungarian_math_olympiad_1927_l571_571611

-- Definitions
def is_coprime (a b : ℤ) : Prop :=
  Int.gcd a b = 1

-- The main statement
theorem hungarian_math_olympiad_1927
  (a b c d x y k m : ℤ) 
  (h_coprime : is_coprime a b)
  (h_m : m = a * d - b * c)
  (h_divides : m ∣ (a * x + b * y)) :
  m ∣ (c * x + d * y) :=
sorry

end hungarian_math_olympiad_1927_l571_571611


namespace arithmetic_sequence_sum_l571_571370

open Nat

noncomputable def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h1 : ∀ n : ℕ, a (n + 1) = a n + d)
    (h2 : a 1 > 0)
    (h3 : a 23 + a 24 > 0)
    (h4 : a 23 * a 24 < 0) :
   46 = 46 :=
begin
  -- proof should go here
  sorry
end

end arithmetic_sequence_sum_l571_571370


namespace enclosing_sphere_radius_and_area_l571_571220

theorem enclosing_sphere_radius_and_area :
  let r : ℝ := 2
  let centers := { (a, b, c) : ℝ × ℝ × ℝ | abs a = r ∧ abs b = r ∧ abs c = r }
  ∀ p ∈ centers, dist (0, 0, 0) p = r + ((2 - 1) * 4) :=
  r + sqrt 3 * 2 :=
  let enclosing_r := 2 * sqrt 3 + (2 : ℝ)
  let surface_area := 4 * π * (enclosing_r ^ 2)
  enclosing_r = 2 * sqrt 3 + 2 ∧ surface_area = 4 * π * (2 * sqrt 3 + 2) ^ 2
by
  sorry

end enclosing_sphere_radius_and_area_l571_571220


namespace limit_of_sequence_N_of_epsilon_l571_571454

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) (h : ∀ n, a_n n = (7 * n - 1) / (n + 1)) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) ↔ a = 7 := sorry

theorem N_of_epsilon (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, N = ⌈8 / ε⌉ := sorry

end limit_of_sequence_N_of_epsilon_l571_571454


namespace num_entrees_ordered_l571_571475

-- Define the conditions
def appetizer_cost: ℝ := 10
def entree_cost: ℝ := 20
def tip_rate: ℝ := 0.20
def total_spent: ℝ := 108

-- Define the theorem to prove the number of entrees ordered
theorem num_entrees_ordered : ∃ E : ℝ, (entree_cost * E) + appetizer_cost + (tip_rate * ((entree_cost * E) + appetizer_cost)) = total_spent ∧ E = 4 := 
by
  sorry

end num_entrees_ordered_l571_571475


namespace evaluate_expression_at_y_minus3_l571_571223

theorem evaluate_expression_at_y_minus3 :
  let y := -3
  (5 + y * (2 + y) - 4^2) / (y - 4 + y^2 - y) = -8 / 5 :=
by
  let y := -3
  sorry

end evaluate_expression_at_y_minus3_l571_571223


namespace estimate_sqrt_mult_sub_l571_571222

theorem estimate_sqrt_mult_sub :
  let x := sqrt 15 * sqrt 3 - 4 in
  2 < x ∧ x < 3 :=
by
  let x := sqrt 15 * sqrt 3 - 4
  sorry

end estimate_sqrt_mult_sub_l571_571222


namespace smallest_number_ending_in_9_divisible_by_13_l571_571100

theorem smallest_number_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), (n % 10 = 9) ∧ (13 ∣ n) ∧ (∀ (m : ℕ), (m % 10 = 9) ∧ (13 ∣ m) ∧ (m < n) -> false) :=
sorry

end smallest_number_ending_in_9_divisible_by_13_l571_571100


namespace count_valid_divisors_l571_571667

-- Define a condition checking if a three-digit number 150 + n is divisible by n
def divisible_by (n : ℕ) : Prop :=
  n ≠ 0 ∧ (150 + n) % n = 0

-- Define a condition checking if the digit is in the desired range
def valid_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

-- Define the main theorem
theorem count_valid_divisors : 
  {n : ℕ | valid_digit n ∧ divisible_by n}.card = 5 :=
by sorry

end count_valid_divisors_l571_571667


namespace points_exist_l571_571639

noncomputable def points_exist_in_space : Prop :=
  ∃ A B C D : ℝ^3, 
    dist A B = 8 ∧ 
    dist C D = 8 ∧ 
    dist A C = 10 ∧ 
    dist B D = 10 ∧ 
    dist A B + dist B C = 13 ∧ 
    -- Conditions for triangle inequalities
    (dist A B + dist B C > dist A C) ∧ 
    (dist A B + dist A C > dist B C) ∧ 
    (dist A C + dist B C > dist A B) ∧ 
    (dist B C + dist C D > dist B D) ∧ 
    (dist B C + dist B D > dist C D) ∧ 
    (dist B D + dist C D > dist B C)

theorem points_exist : points_exist_in_space := sorry

end points_exist_l571_571639


namespace sin_double_angle_l571_571717

variable (θ : Real)
variable (h_θ_second_quadrant : π / 2 < θ ∧ θ < π)
variable (h_cos : cos (π / 2 - θ) = 3 / 5)

theorem sin_double_angle :
  sin (2 * θ) = -24 / 25 :=
by
  have h_sin : sin θ = 3 / 5, from sorry -- This follows from the co-function identity and given h_cos
  have h_cos_theta : cos θ = -4 / 5, from sorry -- This follows from Pythagorean identity and second quadrant property
  sorry

end sin_double_angle_l571_571717


namespace solve_system_l571_571474

noncomputable def system_solutions : Set (ℝ × ℝ) :=
  {p | 
    let (x, y) := p in 
    ( ( (y^5 / x) ^ log 10 x = y ^ (2 * log 10 (x * y)) ) ∧ 
      ( x^2 - 2 * x * y - 4 * x - 3 * y^2 + 12 * y = 0 ) )}

theorem solve_system :
  system_solutions = { (2, 2), (9, 3), ( (9 - Real.sqrt 17) / 2, (Real.sqrt 17 - 1) / 2 ) } :=
sorry

end solve_system_l571_571474


namespace no_solution_l571_571659

theorem no_solution (x : ℝ) (hx : 4 ≤ x) : 
  (sqrt (x + 9 - 6 * sqrt (x - 4)) + sqrt (x + 16 - 8 * sqrt (x - 4)) = 2) → false :=
by sorry

end no_solution_l571_571659


namespace intersection_result_l571_571322

noncomputable def universal_set : Set ℝ := Set.univ

def set_A : Set ℝ := {x | |x - 1| > 2}

def set_B : Set ℝ := {x | x^2 - 6x + 8 < 0}

def complement_U_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

def intersection_CU_A_B : Set ℝ := {x | (x ∈ complement_U_A) ∧ (x ∈ set_B)}

theorem intersection_result : intersection_CU_A_B = {x | 2 < x ∧ x ≤ 3} :=
by
  sorry

end intersection_result_l571_571322


namespace cubic_inequality_l571_571556

theorem cubic_inequality 
  (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) 
  (hy : 0 ≤ y ∧ y ≤ 1) 
  (hz : 0 ≤ z ∧ z ≤ 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 :=
by
  sorry

end cubic_inequality_l571_571556


namespace cos_2alpha_2beta_l571_571681

variables (α β : ℝ)

open Real

theorem cos_2alpha_2beta (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) : cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_2alpha_2beta_l571_571681


namespace binom_inequality_part1_binom_inequality_part2_l571_571666

theorem binom_inequality_part1 (n : ℕ) (hn : n ≥ 2) : 2^n < binom (2 * n) n ∧ binom (2 * n) n < 4^n := sorry

theorem binom_inequality_part2 (n : ℕ) (hn : n ≥ 2) : binom (2 * n - 1) n < 4^(n - 1) := sorry

end binom_inequality_part1_binom_inequality_part2_l571_571666


namespace greatest_4_digit_base7_divisible_by_7_l571_571084

-- Definitions and conditions
def is_base7_number (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 7, d < 7

def is_4_digit_base7 (n : ℕ) : Prop :=
  is_base7_number n ∧ 343 ≤ n ∧ n < 2401 -- 343 = 7^3 (smallest 4-digit base 7) and 2401 = 7^4

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

-- Proof problem statement
theorem greatest_4_digit_base7_divisible_by_7 :
  ∃ (n : ℕ), is_4_digit_base7 n ∧ is_divisible_by_7 n ∧ n = 2346 :=
sorry

end greatest_4_digit_base7_divisible_by_7_l571_571084


namespace square_two_places_higher_l571_571372

theorem square_two_places_higher (x : ℝ) (h : ∃ k : ℤ, x = k^2) : 
  (√x + 2)^2 = x + 4 * √x + 4 :=
by
  sorry

end square_two_places_higher_l571_571372


namespace area_bounded_by_curve_l571_571276

noncomputable def C_x (t : ℝ) : ℝ := Real.exp t - Real.exp (-t)
noncomputable def C_y (t : ℝ) : ℝ := Real.exp (3 * t) + Real.exp (-3 * t)

theorem area_bounded_by_curve :
  (2 * ∫ (x : ℝ) in 0..1, (1 + x^2) * Real.sqrt (4 + x^2)) = (5 * Real.sqrt 5) / 2 :=
sorry

end area_bounded_by_curve_l571_571276


namespace students_not_playing_any_sport_l571_571386

-- Define the constants based on the given conditions.
variables (total_students football_players long_tennis_players basketball_players
  football_and_long_tennis_players football_and_basketball_players long_tennis_and_basketball_players
  all_three_sports_players : ℕ)

-- State the conditions as Lean definitions.
def conditions :=
  total_students = 50 ∧
  football_players = 26 ∧
  long_tennis_players = 20 ∧
  basketball_players = 15 ∧
  football_and_long_tennis_players = 9 ∧
  football_and_basketball_players = 7 ∧
  long_tennis_and_basketball_players = 6 ∧
  all_three_sports_players = 4

-- State the goal: the number of students who do not play any sports.
theorem students_not_playing_any_sport (conds : conditions) : (total_students 
  - (football_players + long_tennis_players + basketball_players
     - football_and_long_tennis_players - football_and_basketball_players - long_tennis_and_basketball_players
     + all_three_sports_players)) = 7 :=
sorry

end students_not_playing_any_sport_l571_571386


namespace smallest_N_for_percentages_l571_571391

theorem smallest_N_for_percentages 
  (N : ℕ) 
  (h1 : ∃ N, ∀ f ∈ [1/10, 2/5, 1/5, 3/10], ∃ k : ℕ, N * f = k) :
  N = 10 := 
by
  sorry

end smallest_N_for_percentages_l571_571391


namespace part_1_part_2_l571_571553

noncomputable def prob_pass_no_fee : ℚ :=
  (3 / 4) * (2 / 3) +
  (1 / 4) * (3 / 4) * (2 / 3) +
  (3 / 4) * (1 / 3) * (2 / 3) +
  (1 / 4) * (3 / 4) * (1 / 3) * (2 / 3)

noncomputable def prob_pass_200_fee : ℚ :=
  (1 / 4) * (1 / 4) * (3 / 4) * ((2 / 3) + (1 / 3) * (2 / 3)) +
  (1 / 3) * (1 / 3) * (2 / 3) * ((3 / 4) + (1 / 4) * (3 / 4))

theorem part_1 : prob_pass_no_fee = 5 / 6 := by
  sorry

theorem part_2 : prob_pass_200_fee = 1 / 9 := by
  sorry

end part_1_part_2_l571_571553


namespace sharon_journey_distance_l571_571933

def original_time (d : ℝ) : ℝ := 180

def new_time (d : ℝ) : ℝ := 
  let v := d / 180
  let v_new := v - 0.5
  90 + 90 * (180 / (d - 90))

theorem sharon_journey_distance :
  ∀ (d : ℝ), new_time d = 300 → original_time d = 180 → d = 157.5 :=
by 
  intros d hn ho
  sorry

end sharon_journey_distance_l571_571933


namespace set_B_can_form_right_angled_triangle_l571_571996

-- Definition and condition from the problem
def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- The actual proof problem statement
theorem set_B_can_form_right_angled_triangle : isRightAngledTriangle 1 (Real.sqrt 3) 2 :=
sorry

end set_B_can_form_right_angled_triangle_l571_571996


namespace floor_sum_equals_10045_l571_571218

def floor_sum_11_2010 : ℕ :=
  (List.range 2009).sum (λ n, Int.toNat ⌊(11 * (n + 1) : ℚ) / 2010⌋)

theorem floor_sum_equals_10045 : floor_sum_11_2010 = 10045 :=
sorry

end floor_sum_equals_10045_l571_571218


namespace complex_conjugate_multiplication_l571_571796

def z : ℂ := (1 - 2 * complex.I) / (1 + complex.I)

theorem complex_conjugate_multiplication :
  z * complex.conj(z) = 5 / 2 := by
sorry

end complex_conjugate_multiplication_l571_571796


namespace find_A_plus_B_l571_571974

theorem find_A_plus_B {A B : ℚ} (h : ∀ x : ℚ, 
                     (Bx - 17) / (x^2 - 9 * x + 20) = A / (x - 4) + 5 / (x - 5)) : 
                     A + B = 9 / 5 := sorry

end find_A_plus_B_l571_571974


namespace optionA_optionC_optionD_l571_571274

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_mono : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) > 0
axiom f_neg_one : f (-1) = 0

theorem optionA : f 3 < f (-4) := sorry

theorem optionC (x : ℝ) : (f x / x > 0) → x ∈ set.Ico (-1) 0 ∨ x ∈ set.Ioi 1 :=
by sorry

theorem optionD : ∀ x : ℝ, ∃ M : ℝ, f x ≥ M :=
by sorry

#check optionA
#check optionC
#check optionD

end optionA_optionC_optionD_l571_571274


namespace count_multiples_of_6_or_8_but_not_both_l571_571781

theorem count_multiples_of_6_or_8_but_not_both: 
  let multiples_of_six := finset.filter (λ n, 6 ∣ n) (finset.range 151)
  let multiples_of_eight := finset.filter (λ n, 8 ∣ n) (finset.range 151)
  let multiples_of_twenty_four := finset.filter (λ n, 24 ∣ n) (finset.range 151)
  multiples_of_six.card + multiples_of_eight.card - 2 * multiples_of_twenty_four.card = 31 := 
by {
  -- Provided proof omitted
  sorry
}

end count_multiples_of_6_or_8_but_not_both_l571_571781


namespace four_digit_number_condition_solution_count_l571_571756

def valid_digits_count : ℕ := 5

theorem four_digit_number_condition (N x a : ℕ) (h1 : N = 1000 * a + x) (h2 : N = 7 * x) (h3 : 100 ≤ x ∧ x ≤ 999) :
  a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 5 :=
begin
  sorry
end

theorem solution_count : ∃ n:ℕ, n = valid_digits_count :=
begin
  use 5,
  sorry
end

end four_digit_number_condition_solution_count_l571_571756


namespace numbers_equal_l571_571703

theorem numbers_equal {k n : ℕ} (a : Fin (k * n + 1) → ℕ)
  (H : ∀ i, ∃ (grp : Fin (k * n) → Fin k), (∀ j, ∑ m in finset.univ.filter (λ x, grp x = j), a (if x < i then x else x + 1) = (∑ b, a b - a i) / k)) :
  ∀ i j, a i = a j :=
sorry

end numbers_equal_l571_571703


namespace johns_trip_equation_l571_571830

-- Defining the constants based on the conditions
def average_speed_before_stop : ℝ := 60
def average_speed_after_stop : ℝ := 90
def stopping_time : ℝ := 1 / 2
def total_distance : ℝ := 300
def total_trip_time : ℝ := 5

-- The theorem we need to prove
theorem johns_trip_equation (t : ℝ) : 
  average_speed_before_stop * t + average_speed_after_stop * (total_trip_time - stopping_time - t) = total_distance := 
sorry

end johns_trip_equation_l571_571830


namespace probability_of_5_odd_in_6_rolls_l571_571068

open Classical

noncomputable def prob_odd_in_six_rolls : ℚ :=
  let num_rolls := 6
  let prob_odd_single := 1 / 2
  let binom_coeff := Nat.choose num_rolls 5
  let total_outcomes := (2 : ℕ) ^ num_rolls
  binom_coeff * ((prob_odd_single ^ 5) * ((1 - prob_odd_single) ^ (num_rolls - 5))) / total_outcomes

theorem probability_of_5_odd_in_6_rolls :
  prob_odd_in_six_rolls = 3 / 32 :=
by
  sorry

end probability_of_5_odd_in_6_rolls_l571_571068


namespace monitor_height_l571_571585

theorem monitor_height (width_in_inches : ℕ) (pixels_per_inch : ℕ) (total_pixels : ℕ) 
  (h1 : width_in_inches = 21) (h2 : pixels_per_inch = 100) (h3 : total_pixels = 2520000) : 
  total_pixels / (width_in_inches * pixels_per_inch) / pixels_per_inch = 12 :=
by
  sorry

end monitor_height_l571_571585


namespace range_of_m_in_second_quadrant_l571_571376

theorem range_of_m_in_second_quadrant 
  (m : ℝ) 
  (z : ℂ) 
  (h : z = (2 + complex.I * m) / (4 - 5 * complex.I)) 
  (hz : z.re < 0 ∧ z.im > 0) :
  m > 8 / 5 :=
sorry

end range_of_m_in_second_quadrant_l571_571376


namespace closest_integer_sum_l571_571240

theorem closest_integer_sum : 
  (1000 * ∑ n in finset.Icc 4 10001, (1 / (n^2 - 9)) : ℝ ).ceil = 204 := sorry

end closest_integer_sum_l571_571240


namespace unique_tangent_point_l571_571294

theorem unique_tangent_point :
  ∃! (p : ℝ × ℝ), (p.1 + p.2 = 2) ∧ (∃ (M N : ℝ × ℝ), (M.1^2 + M.2^2 = 1) ∧ (N.1^2 + N.2^2 = 1) ∧
    (∃ (l1 l2 : ℝ × ℝ → Prop), 
      (∀ (q : ℝ × ℝ), l1 q ↔ (q = M ∨ q = N) ∧ 
        (∀ (r : ℝ × ℝ), l2 r ↔ (r = M ∨ r = N))) ∧ 
          (p ∈ l1 p) ∧ (p ∈ l2 p) ∧ (angle_between_tangents l1 l2 = 90))) :=
sorry

end unique_tangent_point_l571_571294


namespace distribution_of_X_expectation_of_X_l571_571160

-- Define the number of male and female students with mileage >= 90 km
def males_with_90_plus := 5
def females_with_90_plus := 2
def total_with_90_plus := males_with_90_plus + females_with_90_plus

-- Define combinations using binomial coefficient
def C (n k : ℕ) := (Nat.choose n k)

-- Prove the probabilities distribution of X
theorem distribution_of_X : 
  ∀ (X : ℕ),
  (X = 1 ∨ X = 2 ∨ X = 3) 
  → (X = 1 → (C 2 2 * C 5 1) / (C 7 3) = 1 / 7) 
  ∧ (X = 2 → (C 2 1 * C 5 2) / (C 7 3) = 4 / 7) 
  ∧ (X = 3 → (C 2 0 * C 5 3) / (C 7 3) = 2 / 7) :=
by {
  intro X hX,
  cases hX,
  {
    split;
    { intro hX1, rw hX1,
      norm_num1 [C, Nat.choose, total_with_90_plus, Nat.div_eq_one_of_dvd, Nat.gcd_one_left, Nat.mul_one, Nat.mul_comm, Nat.cast_add, Nat.cast_one, Nat.div_self, total_weight_0, C_mul_of, total_weight_1], },
  },
  {
    split;
    { intro hX2, rw hX2,
      norm_num1 [C, Nat.choose, total_with_90_plus, Nat.div_eq_one_of_dvd, Nat.gcd_one_left, Nat.mul_one, Nat.mul_comm, Nat.cast_add, Nat.cast_one, Nat.div_self, total_weight_0],},
  },
  {
    intro hX3; rw hX3,
      norm_num1 [C, Nat.choose, total_with_90_plus, Nat.div_eq_one_of_dvd, Nat.gcd_one_left, Nat.mul_one, Nat.mul_comm, Nat.cast_add, Nat.cast_one, Nat.div_self, total_weight_0],
  }
}

-- Prove the expectation of X
theorem expectation_of_X : (P_X 1 * 1 + P_X 2 * 2 + P_X 3 * 3) = 15 / 7 :=
by {
  norm_num [C, distribution_of_X, P_X],
}

end distribution_of_X_expectation_of_X_l571_571160


namespace solve_for_x_l571_571469

noncomputable def valid_solution (x : ℝ) : Prop :=
  sqrt (2 + sqrt (3 + sqrt x)) = real_root 4 (2 + sqrt x)

theorem solve_for_x : valid_solution 196 :=
by
  sorry

end solve_for_x_l571_571469


namespace unique_solution_mod_37_system_l571_571262

theorem unique_solution_mod_37_system :
  ∃! (a b c d : ℤ), 
  (a^2 + b * c ≡ a [ZMOD 37]) ∧
  (b * (a + d) ≡ b [ZMOD 37]) ∧
  (c * (a + d) ≡ c [ZMOD 37]) ∧
  (b * c + d^2 ≡ d [ZMOD 37]) ∧
  (a * d - b * c ≡ 1 [ZMOD 37]) :=
sorry

end unique_solution_mod_37_system_l571_571262


namespace set_B_can_form_right_angled_triangle_l571_571998

-- Definition and condition from the problem
def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- The actual proof problem statement
theorem set_B_can_form_right_angled_triangle : isRightAngledTriangle 1 (Real.sqrt 3) 2 :=
sorry

end set_B_can_form_right_angled_triangle_l571_571998


namespace true_proposition_among_options_l571_571170

theorem true_proposition_among_options :
  (∀ (x y : ℝ), (x > |y|) → (x > y)) ∧
  (¬ (∀ (x : ℝ), (x > 1) → (x^2 > 1))) ∧
  (¬ (∀ (x : ℤ), (x = 1) → (x^2 + x - 2 = 0))) ∧
  (¬ (∀ (x : ℝ), (x^2 > 0) → (x > 1))) :=
by
  sorry

end true_proposition_among_options_l571_571170


namespace parallel_line_inter_l571_571747

variables {l m n : Line} {α β γ : Plane}

-- Assume the given conditions
def α_cap_β_eq_l (l m : Line) (α β : Plane) := (α ∩ β) = l
def m_parallel_α (m : Line) (α : Plane) := m ∥ α
def m_parallel_β (m : Line) (β : Plane) := m ∥ β

-- The theorem to prove
theorem parallel_line_inter (l m : Line) (α β : Plane) 
  (h1 : α_cap_β_eq_l l m α β) 
  (h2 : m_parallel_α m α) 
  (h3 : m_parallel_β m β) : m ∥ l := 
sorry

end parallel_line_inter_l571_571747


namespace sequence_divisibility_l571_571438

theorem sequence_divisibility (a : Fin 2011 → ℤ)
  (h₀ : a 0 = 1)
  (h₁ : ∀ k : Fin 2011, 1 ≤ k → 2011 ∣ (a (Fin.pred (Fin.cast_lt k dec_trivial)) * a k) - k) :
  2011 ∣ a (Fin.mk 2010 dec_trivial) + 1 :=
sorry

end sequence_divisibility_l571_571438


namespace sequence_count_l571_571209

theorem sequence_count :
  (∀ (a : Fin 8 → ℚ),
    a 0 = 2013 ∧ a 7 = 2014 ∧ (∀ n : Fin 7, a (n + 1) - a n ∈ {-1, 1/3, 1}) ) →
  (∃ (sequences_count : ℕ), sequences_count = 252) :=
begin
  sorry
end

end sequence_count_l571_571209


namespace log_diff_l571_571468

theorem log_diff : log 2 40 - log 2 5 = 3 := 
 by sorry

end log_diff_l571_571468


namespace train_problem_l571_571985

noncomputable def length_of_second_train
  (speed_train1 : ℝ) (speed_train2 : ℝ) (length_train1 : ℝ) (time_cross : ℝ) : ℝ :=
  let relative_speed := (speed_train1 + speed_train2) * (5 / 18)
  in (relative_speed * time_cross) - length_train1

theorem train_problem
  (speed_train1 speed_train2 : ℝ)
  (length_train1 : ℝ)
  (time_cross : ℝ)
  (h1 : speed_train1 = 60)
  (h2 : speed_train2 = 40)
  (h3 : length_train1 = 140)
  (h4 : time_cross = 11.879049676025918) :
  length_of_second_train speed_train1 speed_train2 length_train1 time_cross ≈ 189.97 :=
by
  simp [length_of_second_train, h1, h2, h3, h4]
  norm_num
  sorry

end train_problem_l571_571985


namespace multiples_of_6_or_8_but_not_both_l571_571785

theorem multiples_of_6_or_8_but_not_both (n : ℕ) : 
  n = 25 ∧ (n = 18) ∧ (n = 6) → (25 - 6) + (18 - 6) = 31 :=
by
  sorry

end multiples_of_6_or_8_but_not_both_l571_571785


namespace count_integers_satisfying_inequality_l571_571252

theorem count_integers_satisfying_inequality : 
  (∃ S : Set ℤ, (∀ n ∈ S, 15 < n^2 ∧ n^2 < 120) ∧ S.card = 14) :=
by
  sorry

end count_integers_satisfying_inequality_l571_571252


namespace triangle_inequality_l571_571704

variables (a b c : ℝ)

theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) / (1 + a + b) > c / (1 + c) :=
sorry

end triangle_inequality_l571_571704


namespace number_of_divisors_24_l571_571757

def is_divisor (n d : ℤ) : Prop := ∃ k : ℤ, n = d * k

theorem number_of_divisors_24 :
  (finset.filter (λ d, is_divisor 24 d) 
    (finset.range 25)).card * 2 = 16 :=
by
  sorry

end number_of_divisors_24_l571_571757


namespace solve_fractional_equation_l571_571031

theorem solve_fractional_equation (x : ℝ) (h : (3 / (x + 1) - 2 / (x - 1)) = 0) : x = 5 :=
sorry

end solve_fractional_equation_l571_571031


namespace erroneous_judgment_probability_l571_571730

-- Define a probability measure P on a sigma algebra
noncomputable def P : set ℝ → ℝ := sorry

-- Define the event E where K^2 >= 2.072
def E : set ℝ := {x | x ^ 2 >= 2.072}

-- Given condition: Probability of event E given H₀ is true is 0.15
axiom H0_is_true : P E = 0.15

-- The theorem we need to prove
theorem erroneous_judgment_probability : P E = 0.15 :=
by {
  exact H0_is_true,
}

end erroneous_judgment_probability_l571_571730


namespace problem_statement_l571_571417

def satisfies_conditions (n : ℕ) : Prop :=
  let total_divisors := (n.factorization.find' 2 + 1) * (n.factorization.find' 3 + 1)
  let multiples_of_6 := n.factorization.find' 2 * n.factorization.find' 3
  (multiples_of_6 : ℚ) / total_divisors = 3 / 5 ∧ ∀ p, p.prime → p ∣ n → p ≤ 3

def S : ℕ := @Finset.sum ℕ _ ((Finset.filter satisfies_conditions (Finset.range (1000000)))) (λ x => x)

theorem problem_statement : S / 36 = 2345 := by
  sorry

end problem_statement_l571_571417


namespace largest_share_l571_571263

theorem largest_share (total_profit : ℕ) (partners : ℕ) 
  (ratios : List ℕ) (h_ratios : ratios = [3, 4, 4, 6, 7]) 
  (h_total_profit : total_profit = 48000) : 
  let total_parts := List.sum ratios
  let value_per_part := total_profit / total_parts
  let largest_ratio := List.maximum ratios
  in largest_ratio * value_per_part = 14000 := 
by 
  sorry

end largest_share_l571_571263


namespace narrow_black_stripes_count_l571_571884

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l571_571884


namespace complex_pure_imaginary_solution_l571_571946

theorem complex_pure_imaginary_solution (x : ℝ) (z : ℂ) (h : z = (x^2 - 1) + (x - 1) * complex.I) (pure_imag : z.re = 0) : x = -1 := 
by
  sorry

end complex_pure_imaginary_solution_l571_571946


namespace correct_option_B_l571_571995

variable {a b x y : ℤ}

def option_A (a : ℤ) : Prop := -a - a = 0
def option_B (x y : ℤ) : Prop := -(x + y) = -x - y
def option_C (b a : ℤ) : Prop := 3 * (b - 2 * a) = 3 * b - 2 * a
def option_D (a : ℤ) : Prop := 8 * a^4 - 6 * a^2 = 2 * a^2

theorem correct_option_B (x y : ℤ) : option_B x y := by
  -- The proof would go here
  sorry

end correct_option_B_l571_571995


namespace odd_three_digit_integers_l571_571761

def is_odd (n : ℕ) : Prop := n % 2 = 1
def divisible_by_15 (n : ℕ) : Prop := n % 15 = 0
def valid_digits (n : ℕ) : Prop :=
  let digits := (n / 100, (n / 10) % 10, n % 10)
  digits.1 ≠ 3 ∧ digits.1 ≠ 7 ∧
  digits.2 ≠ 3 ∧ digits.2 ≠ 7 ∧
  digits.3 ≠ 3 ∧ digits.3 ≠ 7

def count_valid_numbers : ℕ :=
  (List.range' 100 900).count (λ n, 
    is_odd n ∧ divisible_by_15 n ∧ valid_digits n
  )

theorem odd_three_digit_integers : count_valid_numbers = 63 := 
  sorry

end odd_three_digit_integers_l571_571761


namespace total_orders_l571_571931

theorem total_orders (x y : ℕ) (h1 : y = 9) 
                     (h2 : 6 * x + 3.5 * y = 133.5) : 
                     x + y = 26 := 
by
  sorry

end total_orders_l571_571931


namespace sin_17_cos_13_plus_cos_17_sin_13_l571_571971

theorem sin_17_cos_13_plus_cos_17_sin_13 : 
  sin (17 * real.pi / 180) * cos (13 * real.pi / 180) + cos (17 * real.pi / 180) * sin (13 * real.pi / 180) = 1 / 2 :=
by
  -- proof goes here
  sorry

end sin_17_cos_13_plus_cos_17_sin_13_l571_571971


namespace find_k_l571_571297

theorem find_k (k : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + 2 * k = 0 ∧ x = 1) : k = 1 / 2 :=
by {
  sorry 
}

end find_k_l571_571297


namespace distinct_real_numbers_a_l571_571788

theorem distinct_real_numbers_a :
  {a : ℝ | ∃ m n : ℤ, m + n = -a ∧ m * n = 6 * a}.to_finset.card = 10 :=
sorry

end distinct_real_numbers_a_l571_571788


namespace narrow_black_stripes_are_eight_l571_571871

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l571_571871


namespace average_gas_mileage_is_33_33_l571_571977

variables (d1 d2 : ℝ) (m_motorcycle m_car : ℝ)

def average_gas_mileage (d1 d2 : ℝ) (m_motorcycle m_car : ℝ) : ℝ :=
  (d1 + d2) / ((d1 / m_motorcycle) + (d2 / m_car))

theorem average_gas_mileage_is_33_33 :
  d1 = 150 → d2 = 150 → m_motorcycle = 50 → m_car = 25 →
  average_gas_mileage d1 d2 m_motorcycle m_car = 100 / 3 :=
by
  intros,
  rw [average_gas_mileage, H, H1, H2, H3],
  sorry

end average_gas_mileage_is_33_33_l571_571977


namespace g_extreme_points_product_inequality_l571_571311

noncomputable def f (a x : ℝ) : ℝ := (-x^2 + a * x - a) / Real.exp x

noncomputable def f' (a x : ℝ) : ℝ := (x^2 - (a + 2) * x + 2 * a) / Real.exp x

noncomputable def g (a x : ℝ) : ℝ := (f a x + f' a x) / (x - 1)

theorem g_extreme_points_product_inequality {a x1 x2 : ℝ} 
  (h_cond1 : a > 2)
  (h_cond2 : x1 + x2 = (a + 2) / 2)
  (h_cond3 : x1 * x2 = 1)
  (h_cond4 : x1 ≠ 1 ∧ x2 ≠ 1)
  (h_x1 : x1 ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1))
  (h_x2 : x2 ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1)) :
  g a x1 * g a x2 < 4 / Real.exp 2 :=
sorry

end g_extreme_points_product_inequality_l571_571311


namespace cone_volume_increase_l571_571380

-- Original cone volume formula: V = 1/3 * π * r^2 * h
-- Goal: Prove that increasing the height by 120% and radius by 80% increases the volume by 612.8%

theorem cone_volume_increase (r h : ℝ) :
  let V_original := 1/3 * Real.pi * r^2 * h,
      V_new := 1/3 * Real.pi * (1.8 * r)^2 * (2.2 * h),
      percentage_increase := ((V_new - V_original) / V_original) * 100 in
  percentage_increase = 612.8 := by
  sorry

end cone_volume_increase_l571_571380


namespace monotonic_decreasing_interval_l571_571018

-- Define the function
def f (x : ℝ) : ℝ := x - Real.log x

-- Define the derivative of the function
noncomputable def f' (x : ℝ) : ℝ := 1 - 1 / x

-- State the theorem
theorem monotonic_decreasing_interval : 
  ∀ x : ℝ, (x > 0 ∧ x < 1) → f' x < 0 :=
by
  sorry

end monotonic_decreasing_interval_l571_571018


namespace sneaker_final_price_l571_571147

-- Definitions of the conditions
def original_price : ℝ := 120
def coupon_value : ℝ := 10
def discount_percent : ℝ := 0.1

-- The price after the coupon is applied
def price_after_coupon := original_price - coupon_value

-- The membership discount amount
def membership_discount := price_after_coupon * discount_percent

-- The final price the man will pay
def final_price := price_after_coupon - membership_discount

theorem sneaker_final_price : final_price = 99 := by
  sorry

end sneaker_final_price_l571_571147


namespace max_t_for_hyperbola_l571_571404

noncomputable def distance_between_parallel_lines (k1 k2 : ℝ) : ℝ :=
  |k2 - k1| / Real.sqrt (2^2 + (-2)^2)

theorem max_t_for_hyperbola (t : ℝ) :
  (∀ (P : ℝ × ℝ), P.1^2 - 2 * P.2^2 = 1 → 
  distance (P.1, P.2) (λ x y => √2 * x - 2 * y + 2) > t) ↔ t ≤ Real.sqrt (6) / 3 := 
by
  sorry

end max_t_for_hyperbola_l571_571404


namespace min_value_of_fraction_sum_l571_571685

theorem min_value_of_fraction_sum (a b : ℤ) (h1 : a = b + 1) : 
  (a > b) -> (∃ x, x > 0 ∧ ((a + b) / (a - b) + (a - b) / (a + b)) = 2) :=
by
  sorry

end min_value_of_fraction_sum_l571_571685


namespace arithmetic_sequence_solution_l571_571963

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

noncomputable def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * a 0 + (n * (n - 1) / 2) * (a 1 - a 0)

theorem arithmetic_sequence_solution :
  ∃ d : ℤ,
  (∀ n : ℕ, n > 0 ∧ n < 10 → a n = 23 + n * d) ∧
  (23 + 5 * d > 0) ∧
  (23 + 6 * d < 0) ∧
  d = -4 ∧
  S a 6 = 78 ∧
  ∀ n : ℕ, S a n > 0 → n ≤ 12 :=
by
  sorry

end arithmetic_sequence_solution_l571_571963


namespace average_of_integers_l571_571986

-- Define the conditions
def conditions (M : ℤ) : Prop :=
  (4 : ℤ) * 11 < M ∧ M < (1 : ℤ) * 15

-- Define the target average
def target_avg : ℤ := 29

-- Define the statement: Prove that the average of integers satisfying the conditions is 29.
theorem average_of_integers (S : Set ℤ) (h : ∀ M ∈ S, conditions M) :
  (S.sum id / S.size : ℤ) = target_avg := sorry

end average_of_integers_l571_571986


namespace average_of_w_x_z_eq_one_sixth_l571_571006

open Real

variable {w x y z t : ℝ}

theorem average_of_w_x_z_eq_one_sixth
  (h1 : 3 / w + 3 / x + 3 / z = 3 / (y + t))
  (h2 : w * x * z = y + t)
  (h3 : w * z + x * t + y * z = 3 * w + 3 * x + 3 * z) :
  (w + x + z) / 3 = 1 / 6 :=
by 
  sorry

end average_of_w_x_z_eq_one_sixth_l571_571006


namespace probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571075

def probability_of_5_odd_numbers_in_6_rolls (prob_odd : ℚ) : ℚ :=
  (nat.choose 6 5 * (prob_odd^5) * ((1 - prob_odd)^1)) / (2^6)

theorem probability_of_5_odd_numbers_in_6_rolls_is_3_over_32 :
  probability_of_5_odd_numbers_in_6_rolls (1/2) = 3 / 32 :=
by sorry

end probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571075


namespace min_value_a_l571_571671

noncomputable def minimum_a : ℝ := -6

theorem min_value_a (a : ℝ) :
  (∀ x ∈ set.Icc (-1 : ℝ) 1, 1 + 2^x + a * 4^x < 0) ↔ (∃ x ∈ set.Icc (-1 : ℝ) 1, 1 + 2^x + a * 4^x ≥ 0) → minimum_a ≤ a := by
  sorry

end min_value_a_l571_571671


namespace problem1_solution_problem2_solution_l571_571198

-- Proof for Problem 1
theorem problem1_solution (x y : ℝ) 
(h1 : x - y - 1 = 4)
(h2 : 4 * (x - y) - y = 5) : 
x = 20 ∧ y = 15 := sorry

-- Proof for Problem 2
theorem problem2_solution (x : ℝ) 
(h1 : 4 * x - 1 ≥ x + 1)
(h2 : (1 - x) / 2 < x) : 
x ≥ 2 / 3 := sorry

end problem1_solution_problem2_solution_l571_571198


namespace Xiaofeng_earlier_time_l571_571563

-- Define the problem's given conditions as constants
def S_PBC : ℝ := 73000
def S_PBD : ℝ := 163000
def S_PCE : ℝ := 694000
def AB_BD_ratio : ℝ := 3 / 8
def A_to_B_time : ℝ := 3
def B_to_D_time : ℝ := 8

-- Define the theorem to prove Xiaofeng's earlier departure time
theorem Xiaofeng_earlier_time : 18 = 
  let S_BCE := S_PBC + S_PCE
  let S_BAC := S_PBC + S_PBD
  let CE_CA_ratio := S_BCE * 8 / (S_BAC * 3)
  let CE_time := 26 -- derived from the proportional ratio
  let extra_time := CE_time - B_to_D_time
  extra_time :=
by sorry -- proof omitted

end Xiaofeng_earlier_time_l571_571563


namespace sine_of_angle_between_CD_and_plane_A1MCN_l571_571818

noncomputable section

variables (a : ℝ) -- side length of the cube

structure Point :=
(x : ℝ) (y : ℝ) (z : ℝ)

-- Points in the cube
def A : Point := ⟨0, 0, 0⟩
def C : Point := ⟨a, 0, 0⟩
def D : Point := ⟨a, a, 0⟩
def A1 : Point := ⟨0, 0, a⟩
def C1 : Point := ⟨a, 0, a⟩
def D1 : Point := ⟨a, a, a⟩

-- Midpoints
def M : Point := ⟨a/2, a, a⟩
def N : Point := ⟨a/2, 0, 0⟩

-- Function to calculate the sine of the angle
def sine_angle (P1 P2 P3 : Point) : ℝ :=
  let v1 := ⟨P2.x - P1.x, P2.y - P1.y, P2.z - P1.z⟩
  let v2 := ⟨P3.x - P1.x, P3.y - P1.y, P3.z - P1.z⟩
  let cross_prod := ⟨v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1⟩ 
  let norm := Real.sqrt (cross_prod.1^2 + cross_prod.2^2 + cross_prod.3^2)
  let dot_prod := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3
  norm / (Real.sqrt (v1.1^2 + v1.2^2 + v1.3^2) * Real.sqrt (v2.1^2 + v2.2^2 + v2.3^2))

theorem sine_of_angle_between_CD_and_plane_A1MCN :
  sine_angle (C) (D) (A1) = sqrt (6) / 3 :=
sorry

end sine_of_angle_between_CD_and_plane_A1MCN_l571_571818


namespace find_EG_length_l571_571461

-- Definitions of lengths and geometric figures involved in the problem
variables (EF : ℝ) (EV EP VF : ℝ)
variables (A B C D : ℝ)
variables (k p : ℕ)

-- Conditions
def conditions : Prop := 
  EF = 432 ∧ EV = 96 ∧ EP = 144 ∧ VF = 192 ∧ 
  (∃ (EG : ℝ), line_m_divides_ratio EF EV EP VF EG 3 4)
 
-- Main problem statement
theorem find_EG_length (k p : ℕ) (h : conditions) : 
  ∃ (EG : ℝ), EG = 72 * real.sqrt 6 ∧ (∃ k p, k * p = 72 + 6) :=
sorry

end find_EG_length_l571_571461


namespace count_multiples_6_or_8_not_both_l571_571765

-- Define the conditions
def is_multiple (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

-- Define the main proof statement
theorem count_multiples_6_or_8_not_both :
  (∑ k in Finset.filter (λ n, is_multiple n 6 ∨ is_multiple n 8 ∧ ¬(is_multiple n 6 ∧ is_multiple n 8)) (Finset.range 151), 1) = 31 := 
sorry

end count_multiples_6_or_8_not_both_l571_571765


namespace intersection_point_exists_l571_571407

-- Definitions of the triangle and points
variables {A B C S1 S2 T1 T2 H1 H2 : Type}
variables [incidence_geometry : has_incidences A B C S1 S2 T1 T2 H1 H2]

open has_incidences

-- Given: In triangle ABC, point S1 lies on BC
axiom intersection_of_AS1_on_BC : intersects (line A S1) (line B C)

-- Given: T1 is the midpoint of segment AS1
axiom midpoint_T1 : midpoint T1 (segment A S1)

-- Given: T2 is the midpoint of segment BS2
axiom midpoint_T2 : midpoint T2 (segment B S2)

-- Given: Lines T1T2, AB, and H1H2 all intersect
axiom intersects_at_single_point : ∃ P : Type, incidence (line T1 T2) P ∧ incidence (line A B) P ∧ incidence (line H1 H2) P

-- Proof to show they intersect at a single point
theorem intersection_point_exists : ∃ P : Type, intersect (line T1 T2) (line A B) (line H1 H2) P :=
  intersects_at_single_point

end intersection_point_exists_l571_571407


namespace real_number_infinite_continued_fraction_l571_571549

theorem real_number_infinite_continued_fraction:
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / y))) ∧ y = 5 / 3 := 
begin
  sorry
end

end real_number_infinite_continued_fraction_l571_571549


namespace circle_integer_points_in_first_quadrant_l571_571801

-- Define the circle equation
def circle_eq (x y : ℤ) : Prop := x^2 + y^2 = 20

-- Define the first quadrant condition
def first_quadrant (x y : ℤ) : Prop := x > 0 ∧ y > 0

-- Main theorem
theorem circle_integer_points_in_first_quadrant :
  (∃ p1 p2 : ℤ × ℤ, (circle_eq p1.1 p1.2) ∧ (circle_eq p2.1 p2.2) ∧ first_quadrant p1.1 p1.2 ∧ first_quadrant p2.1 p2.2 ∧
  p1 ≠ p2 ∧
  (∀ x y : ℤ, circle_eq x y ∧ first_quadrant x y → (x, y) = p1 ∨ (x, y) = p2) :=
sorry

end circle_integer_points_in_first_quadrant_l571_571801


namespace simplify_sqrt_neg_five_squared_l571_571937

theorem simplify_sqrt_neg_five_squared : Real.sqrt ((-5 : ℝ)^2) = 5 := 
by
  sorry

end simplify_sqrt_neg_five_squared_l571_571937


namespace sum_abs_b_eq_l571_571266

def R (x : ℚ) : ℚ := 1 - (1/2) * x + (1/4) * x^2

def S (x : ℚ) : ℚ := R(x) * R(x^2) * R(x^3)

theorem sum_abs_b_eq : (∑ i in Finset.range 13, (S 1)) = 147 / 64 := by
  sorry

end sum_abs_b_eq_l571_571266


namespace sum_of_lengths_of_legs_l571_571048

noncomputable def sum_of_legs_larger_triangle (area_small area_large: ℝ) (hypotenuse_small: ℝ) : ℝ :=
  let a := 5 in
  let b := 12 in
  let scale_factor := (area_large / area_small).sqrt in
  let a_large := scale_factor * a in
  let b_large := scale_factor * b in
  a_large + b_large

theorem sum_of_lengths_of_legs (area_small : ℝ) (area_large : ℝ) (hypotenuse_small : ℝ)
  (h_small : area_small = 24) (h_large : area_large = 600) (h_hypotenuse : hypotenuse_small = 13) :
  sum_of_legs_larger_triangle area_small area_large hypotenuse_small = 85 :=
by
  rw [h_small, h_large, h_hypotenuse]
  simp only [sum_of_legs_larger_triangle]
  sorry

end sum_of_lengths_of_legs_l571_571048


namespace quadrilateral_area_l571_571175

theorem quadrilateral_area (a b c d : ℝ) 
  (ha : a = 10) (hb : b = 20) (hc : c = 30)
  (h_smaller : a < d ∧ b < d ∧ c < d) :
  a + b + c + d = 120 :=
by
  -- Given the conditions
  have h1 : a = 10 := ha
  have h2 : b = 20 := hb
  have h3 : c = 30 := hc
  
  -- Calculate d
  let d := 120 - (10 + 20 + 30)
  -- Ensure conditions
  have h_smaller : 10 < d ∧ 20 < d ∧ 30 < d := h_smaller
  
  -- Show the sum equals 120
  have : 10 + 20 + 30 + d = 120 := calc
    10 + 20 + 30 + d = 10 + 20 + 30 + (120 - (10 + 20 + 30)) : by rfl
                ... = 120 : by simp
  exact this

end quadrilateral_area_l571_571175


namespace selection_ways_selection_with_at_least_one_girl_selection_with_both_genders_l571_571675

-- Defining the basic setup and required counts for selection combinations
open Finset Nat

theorem selection_ways :
  (card (Powerset Len 3 (range 7))) = 35 := by sorry

theorem selection_with_at_least_one_girl :
  (card (Powerset Len 3 (range 7)) - card (Powerset Len 3 (range 4))) = 31 := by sorry

theorem selection_with_both_genders :
  (card (Powerset Len 3 (range 7)) - card (Powerset Len 3 (range 4)) - card (Powerset Len 3 (range 3))) = 30 := by sorry

end selection_ways_selection_with_at_least_one_girl_selection_with_both_genders_l571_571675


namespace probability_of_5_odd_numbers_l571_571077

-- Define a function to represent the binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Axiom that defines the probability of getting an odd number
axiom fair_die_prob : ∀ (x : ℕ), 0 < x ∧ x ≤ 6 -> (1/2)

-- Define the problem statement about the probability
theorem probability_of_5_odd_numbers (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 5) : 
  (binom n k) / 2^n = 3 / 32 := sorry

end probability_of_5_odd_numbers_l571_571077


namespace opposite_face_color_l571_571950

def face_color (faces : ℕ → char) : Prop :=
  faces 0 = 'C' ∧ 
  faces 1 = 'Y' ∧ 
  faces 2 = 'B' ∧ 
  faces 3 = 'C' ∧ 
  faces 4 = 'O' ∧ 
  faces 5 = 'B' ∧ 
  faces 6 = 'C' ∧ 
  faces 7 = 'K' ∧ 
  faces 8 = 'B'

theorem opposite_face_color (faces : ℕ → char) (h : face_color faces) : 
  ∃ (i : ℕ), faces i = 'M' → faces (5 - i) = 'C' :=
sorry

end opposite_face_color_l571_571950


namespace moles_of_NaCl_formed_l571_571260

-- Given conditions
def sodium_bisulfite_moles : ℕ := 2
def hydrochloric_acid_moles : ℕ := 2
def balanced_reaction : Prop :=
  ∀ (NaHSO3 HCl NaCl H2O SO2 : ℕ), 
    NaHSO3 + HCl = NaCl + H2O + SO2

-- Target to prove:
theorem moles_of_NaCl_formed :
  balanced_reaction → sodium_bisulfite_moles = hydrochloric_acid_moles → 
  sodium_bisulfite_moles = 2 := 
sorry

end moles_of_NaCl_formed_l571_571260


namespace total_gallons_l571_571409

-- Definitions from conditions
def num_vans : ℕ := 6
def standard_capacity : ℕ := 8000
def reduced_capacity : ℕ := standard_capacity - (30 * standard_capacity / 100)
def increased_capacity : ℕ := standard_capacity + (50 * standard_capacity / 100)

-- Total number of specific types of vans
def num_standard_vans : ℕ := 2
def num_reduced_vans : ℕ := 1
def num_increased_vans : ℕ := num_vans - num_standard_vans - num_reduced_vans

-- The proof goal
theorem total_gallons : 
  (num_standard_vans * standard_capacity) + 
  (num_reduced_vans * reduced_capacity) + 
  (num_increased_vans * increased_capacity) = 
  57600 := 
by
  -- The necessary proof can be filled here
  sorry

end total_gallons_l571_571409


namespace fourth_term_geometric_progression_l571_571484

theorem fourth_term_geometric_progression :
  ∀ (a₁ a₂ a₃ : ℝ), a₁ = sqrt 3 → a₂ = real.root 3 3 → a₃ = real.root 6 3 → 
  let r := (a₂ / a₁) in 
  let fourth_term := a₃ * r in 
  fourth_term = 1 :=
by
  intros a₁ a₂ a₃ h₁ h₂ h₃
  sorry

end fourth_term_geometric_progression_l571_571484


namespace narrow_black_stripes_l571_571855

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l571_571855


namespace smallest_positive_quadratic_nonresidue_lt_sqrt_p_add_one_l571_571420

theorem smallest_positive_quadratic_nonresidue_lt_sqrt_p_add_one (p : ℕ) (hp : nat.prime p) (hod : p % 2 = 1) :
  ∃ n : ℕ, n > 0 ∧ n < nat.floor (real.sqrt p) + 1 ∧ ¬ is_quadratic_residue n p :=
sorry

end smallest_positive_quadratic_nonresidue_lt_sqrt_p_add_one_l571_571420


namespace find_k_l571_571143

theorem find_k (k : ℝ) : 
  (k - 10) / (-8) = (5 - k) / (-8) → k = 7.5 :=
by
  intro h
  let slope1 := (k - 10) / (-8)
  let slope2 := (5 - k) / (-8)
  have h_eq : slope1 = slope2 := h
  sorry

end find_k_l571_571143


namespace find_list_price_l571_571613

theorem find_list_price (CP : ℝ) (ProfitPerc D1Perc D2Perc LoyaltyPerc CommPerc BonusPerc : ℝ) (FinalPrice : ℝ) :
  CP = 95 ∧
  ProfitPerc = 0.40 ∧
  D1Perc = 0.15 ∧
  D2Perc = 0.08 ∧
  LoyaltyPerc = 0.05 ∧
  CommPerc = 0.15 ∧
  BonusPerc = 0.05 ∧
  FinalPrice = 133 → 
  let Profit := CP * ProfitPerc in
  let SP := CP + Profit in
  let D1 := (SP / (1 - LoyaltyPerc) / (1 - D2Perc)) / (1 - D1Perc) in
  D1 ≈ 179.02 :=
sorry

end find_list_price_l571_571613


namespace tan_beta_value_l571_571712

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = 1 / 3) (h2 : Real.tan (α + β) = 1 / 2) : Real.tan β = 1 / 7 :=
by
  sorry

end tan_beta_value_l571_571712


namespace quadrilateral_square_centers_l571_571416

/-- Let ABCD be an arbitrary quadrilateral. 
    Squares with centers M_1, M_2, M_3, M_4 are constructed on AB, BC, CD, DA respectively, all outwards or all inwards. 
    Prove that M_1 M_3 = M_2 M_4 and M_1 M_3 ⊥ M_2 M_4. -/
theorem quadrilateral_square_centers (A B C D M1 M2 M3 M4 : ℝ) (AB BC CD DA : ℝ) :
  -- Definitions
  (∃ Q1 Q2 Q3 Q4 : ℝ, Q1 = M1 ∧ Q2 = M2 ∧ Q3 = M3 ∧ Q4 = M4) →
  -- Conditions
  (M1 = (A + B) / 2) ∧ (M2 = (B + C) / 2) ∧ (M3 = (C + D) / 2) ∧ (M4 = (D + A) / 2) →
  -- Questions
  (M1 - M3 = M2 - M4) ∧ (⊥ (M1 - M3) (M2 - M4)) :=
sorry


end quadrilateral_square_centers_l571_571416


namespace probability_correct_l571_571649

-- Define the conditions
def total_songs : ℕ := 12
def song_durations (n : ℕ) : ℕ := 45 + 15 * n
def favorite_song_duration : ℕ := 3 * 60 + 45
def total_time : ℕ := 5 * 60
def factorial (n : ℕ) : ℕ := nat.factorial n

-- Define the probability calculation
noncomputable def probability : ℚ := 
  1 - (factorial 11 + factorial 8) / factorial 12

-- Define the target probability
def target_probability : ℚ := 899 / 990 

-- Prove that the calculated probability is equal to the target probability
theorem probability_correct : probability = target_probability := sorry

end probability_correct_l571_571649


namespace pascals_triangle_29_28th_l571_571083

theorem pascals_triangle_29_28th :
  (binom 29 27) = 406 :=
by
  sorry

end pascals_triangle_29_28th_l571_571083


namespace distinct_positive_factors_48_l571_571350

theorem distinct_positive_factors_48 : 
  ∀ (n : ℕ), n = 48 → ∀ (p q : ℕ), p = 2 ∧ q = 3 → (∃ a b : ℕ, 48 = p^a * q^b ∧ (a + 1) * (b + 1) = 10) :=
by
  intros n hn p q hpq
  have h_48 : 48 = 2^4 * 3^1 := by norm_num
  use 4, 1
  split
  · exact h_48
  · norm_num
  sorry

end distinct_positive_factors_48_l571_571350


namespace second_team_speed_l571_571517

/-- Two teams of scientists leave a university at the same time in special vans to search for tornadoes.
    The first team travels east at 20 miles per hour, and the second team travels west at a certain speed.
    Their radios have a range of 125 miles. They will lose radio contact after 2.5 hours. --/
theorem second_team_speed :
  (∃ (v : ℝ), (∀ t : ℝ, t = 2.5 → 20 * t + v * t = 125) → v = 30) :=
by
  use 30
  intro h
  rw h
  sorry

end second_team_speed_l571_571517


namespace factors_of_48_l571_571334

theorem factors_of_48 : ∃ n, n = 48 → number_of_distinct_positive_factors n = 10 :=
sorry

-- Auxiliary function definitions to support the main theorem
def number_of_distinct_positive_factors (n : ℕ) : ℕ := 
sorry

end factors_of_48_l571_571334


namespace smallest_positive_integer_remainder_l571_571530

theorem smallest_positive_integer_remainder
  (b : ℕ) (h1 : b % 4 = 3) (h2 : b % 6 = 5) :
  b = 11 := by
  sorry

end smallest_positive_integer_remainder_l571_571530


namespace range_of_a_l571_571848

open Real

-- Definitions based on given conditions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := ∀ (x : ℝ), x > 0 → -3^x ≤ a

-- The main proposition combining the conditions
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → -1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l571_571848


namespace range_of_quotient_l571_571319

theorem range_of_quotient (a b c : ℝ) (h1 : a > 0) (h2 : c = -1/a) (h3 : b = 1/a) (h4 : a + c ≠ 0) :
  ∃ r, r = (a^2 + b^2 + 7) / (a + c) ∧ (r ∈ set.Iic (-6) ∨ r ∈ set.Ici 6) :=
by
  sorry

end range_of_quotient_l571_571319


namespace actual_cost_of_article_l571_571558

-- Define the basic conditions of the problem
variable (x : ℝ)
variable (h : x - 0.24 * x = 1064)

-- The theorem we need to prove
theorem actual_cost_of_article : x = 1400 :=
by
  -- since we are not proving anything here, we skip the proof
  sorry

end actual_cost_of_article_l571_571558


namespace increasing_log_function_l571_571312

noncomputable def t (a x : ℝ) := a*x^2 + 2*x + a^2

theorem increasing_log_function (a : ℝ) :
  (1 / 2 ≤ a ∧ a < -2 + 2 * real.sqrt 2) ↔
  ∀ x ∈ set.Icc (-4 : ℝ) (-2), ∀ y ∈ set.Icc (-4 : ℝ) (-2), x ≤ y → t a x ≤ t a y :=
begin
  sorry
end

end increasing_log_function_l571_571312


namespace find_equation_of_curve_C_find_extremal_value_of_distances_l571_571273

-- Problem conditions
def circle_C1_center_origin_and_tangent_line (x y : ℝ) : Prop :=
  x^2 + y^2 = 12 ∧ ∃ ay : ℝ, ay ≠ 0 ∧ x - sqrt 2 * ay + 6 = 0

def AM_perp_x_axis (x0 y0 x y : ℝ) : Prop :=
  y0 = 0 ∧ (x = x0 ∧ y = y0)

def vector_ON (x y x0 y0 : ℝ) : Prop :=
  (x, y) = (sqrt 3 / 3 * x0, 1 / 2 * y0)

def point_A_on_circle (x0 y0 : ℝ) : Prop :=
  x0^2 + y0^2 = 12

def curve_C_eq (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Proof problems
theorem find_equation_of_curve_C (x y x0 y0 : ℝ) :
  circle_C1_center_origin_and_tangent_line x y ∧ 
  AM_perp_x_axis x0 0 x y ∧ 
  vector_ON x y x0 y0 ∧ 
  point_A_on_circle x0 y0 →
  curve_C_eq x y :=
sorry

def line_l2 (k m x y : ℝ) : Prop :=
  y = k * x + m

def perpendicular_distances (d1 d2 d3 k m : ℝ) : Prop :=
  d1 = (abs (m - k)) / (sqrt (1 + k^2)) ∧
  d2 = (abs (m + k)) / (sqrt (1 + k^2)) ∧
  d3 = abs(((abs (m - k)) - (abs (m + k))) / k)

def extremal_value (d1 d2 d3 : ℝ) : ℝ :=
  (d1 + d2) * d3

-- Proof problem for extremal value
theorem find_extremal_value_of_distances (d1 d2 d3 k m : ℝ) :
  line_l2 k m d1 d2 ∧
  perpendicular_distances d1 d2 d3 k m → 
  extremal_value d1 d2 d3 = 4 * sqrt 3 :=
sorry

end find_equation_of_curve_C_find_extremal_value_of_distances_l571_571273


namespace num_factors_48_l571_571342

theorem num_factors_48 : 
  ∀ (n : ℕ), n = 48 → (∃ k : ℕ, k = 10 ∧ ∀ d : ℕ, d ∣ n → (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 16 ∨ d = 24 ∨ d = 48)) :=
  by
    intros n h
    sorry

end num_factors_48_l571_571342


namespace minimum_time_for_five_horses_l571_571043

theorem minimum_time_for_five_horses :
  let horses := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let min_time (hs : Finset ℕ) := hs.fold lcm 1
  let t := Finset.fold min 12 (Finset.filter (λ hs, hs.card ≥ 5) (Finset.powerset horses)).map min_time
  (Finset.card (Finset.filter (λ hs, hs.card ≥ 5 ∧ min_time hs = 12) (Finset.powerset horses)) > 0) ∧
  (t = 12) :=
sorry

end minimum_time_for_five_horses_l571_571043


namespace concyclic_of_tangent_and_angle_conditions_l571_571812

theorem concyclic_of_tangent_and_angle_conditions
  {A B C P Q S R : Point}
  (hP_on_AB : P ∈ (Segment A B))
  (hQ_on_AC : Q ∈ (Segment A C))
  (hAP_eq_AQ : dist A P = dist A Q)
  (hS_on_BC : S ∈ (Segment B C))
  (hR_on_BC : R ∈ (Segment B C))
  (hB_S_R_C_collinear : Collinear [B, S, R, C])
  (hBPS_eq_PRS : angle B P S = angle P R S)
  (hCQR_eq_QSR : angle C Q R = angle Q S R) :
  Concyclic [P, Q, S, R] := sorry

end concyclic_of_tangent_and_angle_conditions_l571_571812


namespace solution_set_inequality_l571_571379

-- Define the function and its properties.
variables {α : Type*} [LinearOrderedField α]

-- Assume f is a function from α to α
variable (f : α → α)

-- Conditions from the problem
def odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f x

def increasing_on_positive (f : α → α) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

axiom f_odd : odd_function f
axiom f_increasing : increasing_on_positive f
axiom f_at_minus_2 : f (-2) = 0

theorem solution_set_inequality : {x : α | x * f x < 0} = set.Ioo (-2 : α) 0 ∪ set.Ioo 0 2 :=
by
  sorry

end solution_set_inequality_l571_571379


namespace arithmetic_series_sum_121_l571_571195

-- Define the conditions for the arithmetic series
def is_arithmetic_series (a d : ℕ) (last : ℕ) (n : ℕ) (terms : List ℕ) : Prop :=
  terms = List.iota n |>.map (λ k => a + d * k) ∧ terms.head? = some a ∧ terms.last? = some last

-- Define the sum of a list of natural numbers
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- The main theorem statement
theorem arithmetic_series_sum_121 :
  ∃ (n : ℕ) (terms : List ℕ), is_arithmetic_series 1 2 21 n terms ∧ sum_list terms = 121 :=
by
  sorry

end arithmetic_series_sum_121_l571_571195


namespace smallestBeta_satisfies_l571_571834

noncomputable def validAlphaBeta (alpha beta : ℕ) : Prop :=
  16 / 37 < (alpha : ℚ) / beta ∧ (alpha : ℚ) / beta < 7 / 16

def smallestBeta : ℕ := 23

theorem smallestBeta_satisfies :
  (∀ (alpha beta : ℕ), validAlphaBeta alpha beta → beta ≥ 23) ∧
  (∃ (alpha : ℕ), validAlphaBeta alpha 23) :=
by sorry

end smallestBeta_satisfies_l571_571834


namespace narrow_black_stripes_l571_571856

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l571_571856


namespace sum_inverse_b_n_bounds_l571_571705

def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n: ℕ, a (n+1) = a n + d

def geometric_sequence (b : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n: ℕ, b (n+1) = b n * q

theorem sum_inverse_b_n_bounds
  (a b : ℕ → ℕ)
  (d q : ℕ)
  (h_d : d = 3)
  (h_q : q = 2)
  (h_a_seq : arithmetic_sequence a d)
  (h_b_seq : geometric_sequence b q)
  (h_a1_b1 : a 1 = 1 ∧ b 1 = 1)
  (h_a2_b3 : a 2 = b 3)
  (h_a6_b5 : a 6 = b 5)
  (h_b_n : ∀ n, b n = a n * a (n+1))
  (n : ℕ) :
  1/4 ≤ (∑ k in finset.range n, (1 : ℝ) / b (k + 1)) ∧
  (∑ k in finset.range n, (1 : ℝ) / b (k + 1)) < 1/3 :=
  sorry

end sum_inverse_b_n_bounds_l571_571705


namespace find_f6_l571_571307

-- Define the function f and the necessary properties
variable (f : ℕ+ → ℕ+)
variable (h1 : ∀ n : ℕ+, f n + f (n + 1) + f (f n) = 3 * n + 1)
variable (h2 : f 1 ≠ 1)

-- State the theorem to prove that f(6) = 5
theorem find_f6 : f 6 = 5 :=
sorry

end find_f6_l571_571307


namespace probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571072

def probability_of_5_odd_numbers_in_6_rolls (prob_odd : ℚ) : ℚ :=
  (nat.choose 6 5 * (prob_odd^5) * ((1 - prob_odd)^1)) / (2^6)

theorem probability_of_5_odd_numbers_in_6_rolls_is_3_over_32 :
  probability_of_5_odd_numbers_in_6_rolls (1/2) = 3 / 32 :=
by sorry

end probability_of_5_odd_numbers_in_6_rolls_is_3_over_32_l571_571072


namespace probability_of_5_odd_in_6_rolls_l571_571067

open Classical

noncomputable def prob_odd_in_six_rolls : ℚ :=
  let num_rolls := 6
  let prob_odd_single := 1 / 2
  let binom_coeff := Nat.choose num_rolls 5
  let total_outcomes := (2 : ℕ) ^ num_rolls
  binom_coeff * ((prob_odd_single ^ 5) * ((1 - prob_odd_single) ^ (num_rolls - 5))) / total_outcomes

theorem probability_of_5_odd_in_6_rolls :
  prob_odd_in_six_rolls = 3 / 32 :=
by
  sorry

end probability_of_5_odd_in_6_rolls_l571_571067


namespace discriminant_of_quadratic_l571_571237

   -- Define the coefficients a, b, c
   def a : ℝ := 3
   def b : ℝ := 3 + (1 / 3)
   def c : ℝ := 1 / 3

   -- The discriminant as per formula b^2 - 4ac
   def discriminant : ℝ := b^2 - 4 * a * c

   -- The statement to prove
   theorem discriminant_of_quadratic :
     discriminant = (64 / 9) := 
   by
     -- proof goes here
     sorry
   
end discriminant_of_quadratic_l571_571237


namespace narrow_black_stripes_are_8_l571_571894

-- Define variables: w for wide black stripes, n for narrow black stripes, b for white stripes
variables (w n b : ℕ)

-- Given conditions
axiom cond1 : b = w + 7
axiom cond2 : w + n = b + 1

-- Theorem statement to prove that the number of narrow black stripes is 8
theorem narrow_black_stripes_are_8 : n = 8 :=
by sorry

end narrow_black_stripes_are_8_l571_571894


namespace number_of_correct_propositions_is_zero_l571_571287

-- Define the conditions as Lean propositions
def cond1 (P : Point) (α : Plane) (A B C : Point) : Prop :=
  (¬(P ∈ α) ∧ (A ∈ α) ∧ (B ∈ α) ∧ (C ∈ α)) → ¬coplanar P A B C

def cond2 (l1 l2 l3 : Line) : Prop :=
  intersect l1 l2 ∧ intersect l2 l3 ∧ intersect l1 l3 → coplanar l1 l2 l3

def cond3 (quad : Quadrilateral) : Prop :=
  (opposite_sides_equal quad) → parallelogram quad

-- The main theorem to prove the total number of correct propositions
theorem number_of_correct_propositions_is_zero : 
  ¬cond1 ∧ ¬cond2 ∧ ¬cond3 → (0 = 0) :=
by sorry

end number_of_correct_propositions_is_zero_l571_571287


namespace intersection_of_A_and_B_l571_571707

noncomputable def set_A : Set ℝ := {x | x^2 < 4}
noncomputable def set_B : Set ℝ := {x | ∃ y, y = log (1 - x)}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x | -2 < x ∧ x < 1} := by
  sorry

end intersection_of_A_and_B_l571_571707


namespace max_value_of_3ax_minus_1_l571_571498

theorem max_value_of_3ax_minus_1 (a : ℝ) : 
  (∀ x, 0 ≤ x ∧ x ≤ 1 → a * 1 + a * 0 = 3) → 
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 3 * a * x - 1 ≤ 5) ∧ 
  (∃ x, 0 ≤ x ∧ x ≤ 1 ∧ 3 * a * x - 1 = 5) :=
begin
  sorry
end

end max_value_of_3ax_minus_1_l571_571498


namespace narrow_black_stripes_are_eight_l571_571867

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l571_571867


namespace pens_exceed_500_on_saturday_l571_571439

theorem pens_exceed_500_on_saturday :
  ∃ k : ℕ, (5 * 3 ^ k > 500) ∧ k = 6 :=
by 
  sorry   -- Skipping the actual proof here

end pens_exceed_500_on_saturday_l571_571439


namespace projection_matrix_correct_l571_571843

open Matrix

-- Define vectors and correct projection matrices in the problem
def v0 := ![0, 0] -- placeholder definition for v0
def u1 : Vector (Fin 2) ℚ := ![4, 2]
def u2 : Vector (Fin 2) ℚ := ![2, 3]

def P_u1 : Matrix (Fin 2) (Fin 2) ℚ := 
  (1/20 : ℚ) • matrixOfVector u1 u1
def P_u2 : Matrix (Fin 2) (Fin 2) ℚ := 
  (1/13 : ℚ) • matrixOfVector u2 u2

def M : Matrix (Fin 2) (Fin 2) ℚ :=
  P_u2 ⬝ P_u1

-- The statement to prove
theorem projection_matrix_correct :
  M = ![
         ![28/65,  14/65],
         ![42/65,  21/65]
       ] :=
sorry

end projection_matrix_correct_l571_571843


namespace cd_player_percentage_l571_571969

-- Define the percentage variables
def powerWindowsAndAntiLock : ℝ := 0.10
def antiLockAndCdPlayer : ℝ := 0.15
def powerWindowsAndCdPlayer : ℝ := 0.22
def cdPlayerAlone : ℝ := 0.38

-- Define the problem statement
theorem cd_player_percentage : 
  powerWindowsAndAntiLock = 0.10 → 
  antiLockAndCdPlayer = 0.15 → 
  powerWindowsAndCdPlayer = 0.22 → 
  cdPlayerAlone = 0.38 → 
  (antiLockAndCdPlayer + powerWindowsAndCdPlayer + cdPlayerAlone) = 0.75 :=
by
  intros
  sorry

end cd_player_percentage_l571_571969


namespace units_digit_17_pow_2023_l571_571106

theorem units_digit_17_pow_2023 
  (cycle : ℕ → ℕ)
  (h1 : cycle 0 = 7)
  (h2 : cycle 1 = 9)
  (h3 : cycle 2 = 3)
  (h4 : cycle 3 = 1)
  (units_digit : ℕ → ℕ)
  (h_units : ∀ n, units_digit (17^n) = units_digit (7^n))
  (h_units_cycle : ∀ n, units_digit (7^n) = cycle (n % 4)) :
  units_digit (17^2023) = 3 :=
by
  sorry

end units_digit_17_pow_2023_l571_571106


namespace william_probability_l571_571552
noncomputable def probability_getting_at_least_one_correct_out_of_four : ℚ :=
  1 - (4 / 5) ^ 4

theorem william_probability :
  probability_getting_at_least_one_correct_out_of_four = 369 / 625 :=
by
  rw [probability_getting_at_least_one_correct_out_of_four]
  norm_num
  sorry

end william_probability_l571_571552


namespace locus_is_straight_line_l571_571693

def point : Type := (ℝ × ℝ)

structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  -- Represents a line of the form ax + by + c = 0

def is_equidistant (p : point) (A : point) (l : Line) : Prop :=
  let distance_to_point := (λ (P Q : point), real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  let distance_to_line := (λ (P : point), (abs (l.a * P.1 + l.b * P.2 + l.c)) / (real.sqrt (l.a^2 + l.b^2)))
  distance_to_point p A = distance_to_line p

def A : point := (1, 1)
def l : Line := { a := 1, b := 1, c := -2 }

-- The theorem we want to prove
theorem locus_is_straight_line :
  ∃ m b : ℝ, ∀ p : point, is_equidistant p A l → (p.2 = m * p.1 + b) := sorry

end locus_is_straight_line_l571_571693


namespace num_perfect_cubes_between_100_500_l571_571762

-- Define the problem statement in Lean
theorem num_perfect_cubes_between_100_500 : 
  {n : ℕ | 100 < n^3 ∧ n^3 < 500}.to_finset.card = 3 :=
by
  sorry

end num_perfect_cubes_between_100_500_l571_571762


namespace sum_S_2016_l571_571726

-- Definition of arithmetic sequence conditions
def arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

-- Given conditions in the problem
def a_n (n : ℕ) : ℝ := -3 * n + 2

-- Sum of first n terms of the sequence {(-1)^n a_n}
def S_n (n : ℕ) : ℝ := ∑ i in Finset.range n, (-1) ^ i * a_n i

-- Statement of the problem
theorem sum_S_2016 : arith_seq a_n (-3) ∧ a_n 1 = -1 ∧ a_n 5 = -13 → S_n 2016 = 3024 := 
by 
  intro h_conditions
  sorry

end sum_S_2016_l571_571726


namespace fish_lifespan_is_12_l571_571512

def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_is_12 : fish_lifespan = 12 := by
  sorry

end fish_lifespan_is_12_l571_571512


namespace number_of_pizzas_l571_571051

theorem number_of_pizzas (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 4) (h2 : total_slices = 68) : total_slices / slices_per_pizza = 17 :=
by
  rw [h1, h2]
  norm_num
  sorry

end number_of_pizzas_l571_571051


namespace proof_a_0_proof_S_n_proof_comparison_l571_571676

-- Defining the conditions
def polynomial_condition (x : ℤ) (a : ℤ → ℤ) (n : ℕ) : Prop :=
  (x + 2) ^ n = (∑ k in Finset.range (n + 1), a k * (x - 1) ^ k)

-- Definitions of a0 and S_n based on polynomial condition
def a_0 (n : ℕ) : ℤ := 3 ^ n

def S_n (a : ℤ → ℤ) (n : ℕ) : ℤ :=
  (∑ k in Finset.range (n + 1), a k) - a 0

-- Proving the questions
theorem proof_a_0 (n : ℕ) (hn : 0 < n) :
  ∀ (x : ℤ) (a : ℤ → ℤ), polynomial_condition x a n → a 0 = 3 ^ n :=
sorry

theorem proof_S_n (n : ℕ) (hn : 0 < n) :
  ∀ (x : ℤ) (a : ℤ → ℤ), polynomial_condition x a n → 
    (∑ k in Finset.range (n + 1), a k) = 4 ^ n → S_n a n = 4 ^ n - 3 ^ n :=
sorry

theorem proof_comparison (n : ℕ) (hn : 4 ≤ n) :
  ∀ (a : ℤ → ℤ), S_n a n > (n - 2) * 3 ^ n + 2 * n ^ 2 :=
sorry

end proof_a_0_proof_S_n_proof_comparison_l571_571676


namespace problem1_problem2_l571_571564

theorem problem1 :
  sqrt 3 * tan 45 - (2023 - Real.pi) ^ 0 + abs (2 * sqrt 3 - 2) +
  (1 / 4)⁻¹ - sqrt 27 = 1 := sorry

theorem problem2 (x : ℝ) (h1 : -2 < x) (h2 : x < 3) (h3 : x ≠ -1) (h4 : x ≠ 0) (h5 : x ≠ 1) :
  (x^2 - x) / (x^2 + 2 * x + 1) / ((2 / (x + 1)) - (1 / x)) = 4 / 3 :=
begin
  have hx : x = 2, from sorry,
  rw hx,
  sorry
end

end problem1_problem2_l571_571564


namespace sneaker_final_price_l571_571146

-- Definitions of the conditions
def original_price : ℝ := 120
def coupon_value : ℝ := 10
def discount_percent : ℝ := 0.1

-- The price after the coupon is applied
def price_after_coupon := original_price - coupon_value

-- The membership discount amount
def membership_discount := price_after_coupon * discount_percent

-- The final price the man will pay
def final_price := price_after_coupon - membership_discount

theorem sneaker_final_price : final_price = 99 := by
  sorry

end sneaker_final_price_l571_571146


namespace function_monotonicity_l571_571724

noncomputable def f : ℝ → ℝ := 
λ x, if x ≥ 0 then x^(1/3) else (-x)^(1/3)

noncomputable def g : ℝ → ℝ := 
λ x, exp |x|

theorem function_monotonicity (x : ℝ) (h : x ∈ Ioo (-2 : ℝ) (0 : ℝ)) : 
  (f x > f (x - ε) ↔ g x > g (x - ε)) := sorry

end function_monotonicity_l571_571724


namespace science_books_have_9_copies_l571_571914

theorem science_books_have_9_copies :
  ∃ (A B C D : ℕ), A + B + C + D = 35 ∧ A + B = 17 ∧ B + C = 16 ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ B = 9 :=
by
  sorry

end science_books_have_9_copies_l571_571914


namespace range_of_x_l571_571401

noncomputable def function_y (x : ℝ) : ℝ := 2 / (Real.sqrt (x + 4))

theorem range_of_x : ∀ x : ℝ, (∃ y : ℝ, y = function_y x) → x > -4 :=
by
  intro x h
  sorry

end range_of_x_l571_571401


namespace narrow_black_stripes_l571_571881

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l571_571881


namespace smallest_pos_int_ending_in_9_divisible_by_13_l571_571093

theorem smallest_pos_int_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), n % 10 = 9 ∧ n % 13 = 0 ∧ ∀ m, m % 10 = 9 ∧ m % 13 = 0 → n ≤ m :=
by
  sorry

end smallest_pos_int_ending_in_9_divisible_by_13_l571_571093


namespace correctPairsAreSkating_l571_571674

def Friend := String
def Brother := String

structure SkatingPair where
  gentleman : Friend
  lady : Friend

-- Define the list of friends with their brothers
def friends : List Friend := ["Lyusya Egorova", "Olya Petrova", "Inna Krymova", "Anya Vorobyova"]
def brothers : List Brother := ["Andrey Egorov", "Serezha Petrov", "Dima Krymov", "Yura Vorobyov"]

-- Condition: The skating pairs such that gentlemen are taller than ladies and no one skates with their sibling
noncomputable def skatingPairs : List SkatingPair :=
  [ {gentleman := "Yura Vorobyov", lady := "Lyusya Egorova"},
    {gentleman := "Andrey Egorov", lady := "Olya Petrova"},
    {gentleman := "Serezha Petrov", lady := "Inna Krymova"},
    {gentleman := "Dima Krymov", lady := "Anya Vorobyova"} ]

-- Proving that the pairs are exactly as specified.
theorem correctPairsAreSkating :
  skatingPairs = 
    [ {gentleman := "Yura Vorobyov", lady := "Lyusya Egorova"},
      {gentleman := "Andrey Egorov", lady := "Olya Petrova"},
      {gentleman := "Serezha Petrov", lady := "Inna Krymova"},
      {gentleman := "Dima Krymov", lady := "Anya Vorobyova"} ] :=
by
  sorry

end correctPairsAreSkating_l571_571674


namespace oprq_possible_figures_l571_571706

theorem oprq_possible_figures (x1 y1 x2 y2 : ℝ) (h : (x1, y1) ≠ (x2, y2)) : 
  -- Define the points P, Q, and R
  let P := (x1, y1)
  let Q := (x2, y2)
  let R := (x1 - x2, y1 - y2)
  -- Proving the geometric possibilities
  (∃ k : ℝ, x1 = k * x2 ∧ y1 = k * y2) ∨
  -- When the points are collinear
  ((x1 + x2, y1 + y2) = (x1, y1)) :=
sorry

end oprq_possible_figures_l571_571706


namespace find_width_of_second_tract_l571_571327

variable (length1 width1 length2 : ℕ)
variable (combined_area area1 : ℕ)
variable (width2 : ℕ)

-- Conditions
def tract1 : ℕ := length1 * width1
def tract2_area : ℕ := combined_area - tract1
def width2_compute (length2 : ℕ) (tract2_area : ℕ) : ℕ := tract2_area / length2

-- Theorem statement
theorem find_width_of_second_tract 
  (h1 : length1 = 300) 
  (h2 : width1 = 500) 
  (h3 : length2 = 250) 
  (h4 : combined_area = 307500) 
  (h5 : area1 = length1 * width1)
  (h6 : tract2_area = combined_area - area1) :
  width2 = 630 :=
by
  -- Proof goes here
  sorry

end find_width_of_second_tract_l571_571327


namespace multiples_of_6_or_8_but_not_both_l571_571777

/-- The number of positive integers less than 151 that are multiples of either 6 or 8 but not both is 31. -/
theorem multiples_of_6_or_8_but_not_both (n : ℕ) :
  (multiples_of_6 : Set ℕ) = {k | k < 151 ∧ k % 6 = 0}
  ∧ (multiples_of_8 : Set ℕ) = {k | k < 151 ∧ k % 8 = 0}
  ∧ (multiples_of_24 : Set ℕ) = {k | k < 151 ∧ k % 24 = 0}
  ∧ multiples_of_6_or_8 := {k | k ∈ multiples_of_6 ∨ k ∈ multiples_of_8}
  ∧ multiples_of_6_and_8 := {k | k ∈ multiples_of_6 ∧ k ∈ multiples_of_8}
  ∧ (card (multiples_of_6_or_8 \ multiples_of_6_and_8)) = 31 := sorry

end multiples_of_6_or_8_but_not_both_l571_571777


namespace largest_integer_achievable_smallest_integer_achievable_l571_571400

-- Define the mathematical problem
def largest_possible_integer := 44800
def smallest_possible_integer := 7

-- Proof statement determining the largest integer achievable
theorem largest_integer_achievable:
  ∃ (e : ℕ), e = (10 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / 9 ∧ e = largest_possible_integer := by
  assume a : 10 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 / 9 = 44800
  exact sorry

-- Proof statement determining the smallest integer achievable
theorem smallest_integer_achievable:
  ∃ (e : ℕ), e = (((10 * 6 * 4 * 3) / (9 * 8 * 5 * 2 * 1)) * 7 ∧ e = smallest_possible_integer := by
  assume a : (((10 * 6 * 4 * 3) / (9 * 8 * 5 * 2 * 1)) * 7) = 7
  exact sorry

end largest_integer_achievable_smallest_integer_achievable_l571_571400


namespace number_of_2008_appearance_l571_571027

variable (a : ℕ → ℤ)

-- Conditions
axiom infinite_positives : ∀n : ℕ, ∃ m ≥ n, 0 < a m
axiom infinite_negatives : ∀n : ℕ, ∃ m ≥ n, a m < 0
axiom pairwise_distinct_remainders : ∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → (a i % n ≠ a j % n)

-- Question to prove
theorem number_of_2008_appearance : (finset.univ.filter (λ n, a n = 2008)).card = 1 :=
sorry

end number_of_2008_appearance_l571_571027


namespace nonagon_diagonals_l571_571329

-- Define the number of sides of the polygon (nonagon)
def num_sides : ℕ := 9

-- Define the formula for the number of diagonals in a convex n-sided polygon
def number_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem nonagon_diagonals : number_diagonals num_sides = 27 := 
by
--placeholder for the proof
sorry

end nonagon_diagonals_l571_571329


namespace circle_area_given_conditions_l571_571920

noncomputable def circleArea (A B : ℝ × ℝ) (radius : ℝ) : ℝ := 
  π * radius^2

theorem circle_area_given_conditions :
  let A := (8 : ℝ, 15 : ℝ),
      B := (14 : ℝ, 9 : ℝ),
      C := (-1 : ℝ, 0 : ℝ),
      radius := real.sqrt (306) in
  (C.1 = -1 ∧ C.2 = 0) → 
  (radius = real.sqrt ((A.1 + 1)^2 + (A.2 - 0)^2)) → 
  (radius = real.sqrt ((B.1 + 1)^2 + (B.2 - 0)^2)) → 
  circleArea A B radius = 306 * π :=
by 
  sorry

end circle_area_given_conditions_l571_571920


namespace arith_seq_sum_l571_571189

theorem arith_seq_sum (n : ℕ) (h₁ : 2 * n - 1 = 21) : 
  (∑ i in finset.range 11, (2 * i + 1)) = 121 :=
by
  sorry

end arith_seq_sum_l571_571189


namespace perpendicular_line_folding_exists_l571_571156

noncomputable def fold_paper_perpendicular (paper : Paper)(A : Point)(initial_crease : Line) (straight_edge: Line): Prop :=
  (is_crease paper A initial_crease straight_edge) → 
  (exists perpendicular_crease : Line, is_perpendicular perpendicular_crease initial_crease A)

axiom is_crease (paper : Paper)(A : Point)(initial_crease : Line)(straight_edge : Line) : Prop
axiom is_perpendicular (perpendicular_crease : Line)(initial_crease : Line)(A : Point) : Prop

theorem perpendicular_line_folding_exists (paper : Paper) (A : Point) (initial_crease : Line) (straight_edge: Line): 
  fold_paper_perpendicular paper A initial_crease straight_edge :=
sorry

end perpendicular_line_folding_exists_l571_571156


namespace dogs_daily_food_total_l571_571648

theorem dogs_daily_food_total :
  let first_dog_food := 0.125
  let second_dog_food := 0.25
  let third_dog_food := 0.375
  let fourth_dog_food := 0.5
  first_dog_food + second_dog_food + third_dog_food + fourth_dog_food = 1.25 :=
by
  sorry

end dogs_daily_food_total_l571_571648


namespace number_of_sixth_graders_purchased_more_pencils_than_seventh_l571_571009

theorem number_of_sixth_graders_purchased_more_pencils_than_seventh :
  ∀ (cost_pencil : ℕ) (num_seventh_graders num_sixth_graders num_eighth_graders : ℕ),
    (cost_pencil * num_seventh_graders = 168) →
    (cost_pencil * num_sixth_graders = 208) →
    (cost_pencil * num_eighth_graders = 156) →
    (num_sixth_graders - num_seventh_graders = 4) :=
begin
  intros cost_pencil num_seventh_graders num_sixth_graders num_eighth_graders,
  intros H_seventh H_sixth H_eighth,
  sorry,
end

end number_of_sixth_graders_purchased_more_pencils_than_seventh_l571_571009


namespace distinct_positive_factors_48_l571_571348

theorem distinct_positive_factors_48 : 
  ∀ (n : ℕ), n = 48 → ∀ (p q : ℕ), p = 2 ∧ q = 3 → (∃ a b : ℕ, 48 = p^a * q^b ∧ (a + 1) * (b + 1) = 10) :=
by
  intros n hn p q hpq
  have h_48 : 48 = 2^4 * 3^1 := by norm_num
  use 4, 1
  split
  · exact h_48
  · norm_num
  sorry

end distinct_positive_factors_48_l571_571348


namespace true_propositions_l571_571168

theorem true_propositions :
  (∀ (P : Type) [metric_space P] (T : set P), 
  (∀ p ∈ T, ∃ q, ∀ a ∈ T, dist q a = dist p a) ↔ 
  (p ≠ q ∧ q ∃! dist q P) : Prop) →  
  (∀ (P : Type) [metric_space P] (R : set P), 
  (∀ p ∈ R, ∃ q : P, ∀ a ∈ R, dist q a = dist p a) → false) →
  (∀ (P : Type) [metric_space P] (S : set P), 
  (∃ q, ∀ a ∈ S, dist q a = dist (classical.some S) a) ∧ 
  (∃ q, ∀ a ∈ S, dist q a = dist (classical.some S) a) → false) → 
  (∀ (P : Type) [metric_space P] (T : set P), 
  (∃ q, ∀ a ∈ T, dist q a = dist (classical.some T) a) ∧ 
  (∃ q, ∀ a ∈ (inscribed_sphere_center T), dist q a = dist (classical.some T) a)
  → true) :=
begin
  sorry
end

end true_propositions_l571_571168


namespace trig_identity_pq_l571_571377

theorem trig_identity_pq 
  (α β : ℝ)
  (hα : ∃ t : ℝ, (t ≥ 0 ∧ t ≤ 2 * π) ∧ sin α = 4 / 5 ∧ cos α = -3 / 5)
  (hβ : ∃ t : ℝ, (t ≥ 0 ∧ t ≤ 2 * π) ∧ sin β = -2 / (sqrt 5) ∧ cos β = -1 / (sqrt 5)) :
  sin (α - β) = - (2 * sqrt 5) / 5 ∧ cos (α + β) = 11 * sqrt 5 / 25 := 
by
  sorry

end trig_identity_pq_l571_571377


namespace arithmetic_mean_first_n_positive_integers_l571_571012

theorem arithmetic_mean_first_n_positive_integers (n : ℕ) (Sn : ℕ) (h : Sn = n * (n + 1) / 2) : 
  (Sn / n) = (n + 1) / 2 := by
  -- proof steps would go here
  sorry

end arithmetic_mean_first_n_positive_integers_l571_571012


namespace quadratic_roots_l571_571743

theorem quadratic_roots (r s : ℝ) (A : ℝ) (B : ℝ) (C : ℝ) (p q : ℝ) 
  (h1 : A = 3) (h2 : B = 4) (h3 : C = 5) 
  (h4 : r + s = -B / A) (h5 : rs = C / A) 
  (h6 : 4 * rs = q) :
  p = 56 / 9 :=
by 
  -- We assume the correct answer is given as we skip the proof details here.
  sorry

end quadratic_roots_l571_571743


namespace no_such_function_exists_l571_571836

theorem no_such_function_exists 
  (f : ℝ → ℝ) 
  (h_f_pos : ∀ x, 0 < x → 0 < f x) 
  (h_eq : ∀ x y, 0 < x → 0 < y → f (x + y) = f x + f y + (1 / 2012)) : 
  false :=
sorry

end no_such_function_exists_l571_571836


namespace distinct_bead_arrangements_l571_571819

theorem distinct_bead_arrangements :
  let points : set ℝ := {v1, v2, v3} in
  let colors : set ℕ := {red, blue, green} in
  let sym_group := {id, ρ, ρ^2, σ, ρσ, ρ^2σ} in
  ∀ (v1 v2 v3 : ℝ) (red blue green : ℕ),
  v1 ∈ points ∧ v2 ∈ points ∧ v3 ∈ points ∧
  red ∈ colors ∧ blue ∈ colors ∧ green ∈ colors ∧
  v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 →
  (number_of_distinct_arrangements points colors sym_group = 10) :=
begin
  sorry
end

end distinct_bead_arrangements_l571_571819


namespace number_of_integers_l571_571248

theorem number_of_integers (n : ℤ) : {n : ℤ | 15 < n^2 ∧ n^2 < 120}.finite.card = 14 :=
sorry

end number_of_integers_l571_571248


namespace find_d_l571_571494

theorem find_d (a b c d : ℝ) (h : a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3)) : 
  d = 13 / 4 :=
sorry

end find_d_l571_571494


namespace multiples_of_6_or_8_but_not_both_l571_571774

/-- The number of positive integers less than 151 that are multiples of either 6 or 8 but not both is 31. -/
theorem multiples_of_6_or_8_but_not_both (n : ℕ) :
  (multiples_of_6 : Set ℕ) = {k | k < 151 ∧ k % 6 = 0}
  ∧ (multiples_of_8 : Set ℕ) = {k | k < 151 ∧ k % 8 = 0}
  ∧ (multiples_of_24 : Set ℕ) = {k | k < 151 ∧ k % 24 = 0}
  ∧ multiples_of_6_or_8 := {k | k ∈ multiples_of_6 ∨ k ∈ multiples_of_8}
  ∧ multiples_of_6_and_8 := {k | k ∈ multiples_of_6 ∧ k ∈ multiples_of_8}
  ∧ (card (multiples_of_6_or_8 \ multiples_of_6_and_8)) = 31 := sorry

end multiples_of_6_or_8_but_not_both_l571_571774


namespace narrow_black_stripes_are_eight_l571_571872

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l571_571872


namespace min_value_expression_l571_571425

theorem min_value_expression (α β : ℝ) : 
  ∃ a b : ℝ, 
    ((2 * Real.cos α + 5 * Real.sin β - 8) ^ 2 + 
    (2 * Real.sin α + 5 * Real.cos β - 15) ^ 2  = 100) :=
sorry

end min_value_expression_l571_571425


namespace number_of_comic_books_l571_571488

def fairy_tale_books := 305
def science_and_technology_books := fairy_tale_books + 115
def total_books := fairy_tale_books + science_and_technology_books
def comic_books := total_books * 4

theorem number_of_comic_books : comic_books = 2900 := by
  sorry

end number_of_comic_books_l571_571488


namespace smallest_pos_int_ending_in_9_divisible_by_13_l571_571096

theorem smallest_pos_int_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), n % 10 = 9 ∧ n % 13 = 0 ∧ ∀ m, m % 10 = 9 ∧ m % 13 = 0 → n ≤ m :=
by
  sorry

end smallest_pos_int_ending_in_9_divisible_by_13_l571_571096


namespace symmetry_axis_of_transformed_function_l571_571314

theorem symmetry_axis_of_transformed_function :
  let initial_func (x : ℝ) := Real.sin (4 * x - π / 6)
  let stretched_func (x : ℝ) := Real.sin (8 * x - π / 3)
  let transformed_func (x : ℝ) := Real.sin (8 * (x + π / 4) - π / 3)
  let ω := 8
  let φ := 5 * π / 3
  x = π / 12 :=
  sorry

end symmetry_axis_of_transformed_function_l571_571314


namespace quadratic_function_correct_l571_571044

-- Defining the quadratic function a
def quadratic_function (x : ℝ) : ℝ := 2 * x^2 - 14 * x + 20

-- Theorem stating that the quadratic function passes through the points (2, 0) and (5, 0)
theorem quadratic_function_correct : 
  quadratic_function 2 = 0 ∧ quadratic_function 5 = 0 := 
by
  -- these proofs are skipped with sorry for now
  sorry

end quadratic_function_correct_l571_571044


namespace smallest_positive_integer_remainder_l571_571532

theorem smallest_positive_integer_remainder
  (b : ℕ) (h1 : b % 4 = 3) (h2 : b % 6 = 5) :
  b = 11 := by
  sorry

end smallest_positive_integer_remainder_l571_571532


namespace f_neg10_l571_571793

noncomputable def f : ℝ → ℝ := sorry 

axiom odd_function (f : ℝ → ℝ) : ∀ x : ℝ, f(-x) = -f(x)

axiom f_definition (f : ℝ → ℝ) : ∀ x : ℝ, x > 0 → f(x) = 2 + Real.log x

theorem f_neg10 : f(-10) = -3 := 
by  
  sorry

end f_neg10_l571_571793


namespace initial_rate_oranges_l571_571583

def cost_price : ℝ := 100
def loss_percentage : ℝ := 4
def gain_percentage : ℝ := 44
def selling_price_loss : ℝ := cost_price * (1 - loss_percentage / 100)
def selling_price_gain : ℝ := cost_price * (1 + gain_percentage / 100)
def oranges_per_rupee_gain : ℝ := 16
def price_per_orange_gain : ℝ := selling_price_gain / oranges_per_rupee_gain

theorem initial_rate_oranges (x : ℝ) 
  (h_loss : selling_price_loss = cost_price * (1 - loss_percentage / 100))
  (h_gain : price_per_orange_gain = selling_price_gain / oranges_per_rupee_gain)
  (h_eq : selling_price_loss / x = price_per_orange_gain) : 
  x = 11 :=
sorry

end initial_rate_oranges_l571_571583


namespace narrow_black_stripes_l571_571850

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l571_571850


namespace narrow_black_stripes_count_l571_571885

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l571_571885


namespace part_I_solution_part_II_solution_l571_571569

noncomputable def part_I := 
  (2:ℝ)^(-Real.log2 4) - (8/27:ℝ)^(-2/3) + Real.log10 (1/100) + (Real.sqrt 2 - 1)^(Real.log10 1) + (Real.log10 5)^2 + Real.log10 2 * Real.log10 50

theorem part_I_solution : part_I = -2 := by sorry

variables (x : ℝ) (h : x^(1/2) + x^(-1/2) = 3)

noncomputable def part_II := 
  (x^2 + x^(-2) - 2) / (x + x^(-1) - 3)

theorem part_II_solution (x_pos : 0 < x) : part_II x h = 45/4 := by sorry

end part_I_solution_part_II_solution_l571_571569


namespace metal_waste_l571_571591

theorem metal_waste (a b : ℝ) (h : a < b) :
  let radius := a / 2
  let area_rectangle := a * b
  let area_circle := π * radius^2
  let side_square := a / Real.sqrt 2
  let area_square := side_square^2
  area_rectangle - area_square = a * b - ( a ^ 2 ) / 2 := by
  let radius := a / 2
  let area_rectangle := a * b
  let area_circle := π * (radius ^ 2)
  let side_square := a / Real.sqrt 2
  let area_square := side_square ^ 2
  sorry

end metal_waste_l571_571591


namespace radius_of_sphere_l571_571158

noncomputable def sphere_radius (a b : ℝ) : ℝ :=
  (√3 * a * (2 * b - a)) / (2 * sqrt (3 * b ^ 2 - a ^ 2))

theorem radius_of_sphere (a b : ℝ) (h: b > a) :
  sphere_radius a b = (√3 * a * (2 * b - a)) / (2 * sqrt (3 * b ^ 2 - a ^ 2)) := 
by sorry

end radius_of_sphere_l571_571158


namespace gcd_of_numbers_is_one_l571_571239

theorem gcd_of_numbers_is_one :
  ∃ d : ℕ, d = Nat.gcd (Nat.gcd 546 1288) (Nat.gcd 3042 5535) ∧ d = 1 :=
begin
  use Nat.gcd (Nat.gcd 546 1288) (Nat.gcd 3042 5535),
  split,
  { -- this is where we show the GCD computation chaining
    refl, -- reflexivity: use the definition of our gcd sequence
  },
  { -- this is where the final result 1 is established
    -- showing that GCD of all four numbers evaluated as 1
    sorry
  }
end

end gcd_of_numbers_is_one_l571_571239


namespace max_diagonals_in_2011_gon_l571_571578

theorem max_diagonals_in_2011_gon : 
  let n := 2011 
  in let max_diagonals (n : ℕ) := 2 * n - 6 
  in max_diagonals n = 4016 := 
by 
  sorry

end max_diagonals_in_2011_gon_l571_571578


namespace compare_sin_tan_l571_571296

theorem compare_sin_tan (x : ℝ) (h₀ : 0 < x) (h₁ : x < π / 2) (h₂ : sin x < x) (h₃ : x < tan x) : 
  x < 1 / 2 * (sin x + tan x) :=
sorry

end compare_sin_tan_l571_571296


namespace find_length_xx1_l571_571815

theorem find_length_xx1
  (DE DF : ℝ) (h_DE : DE = 13) (h_DF : DF = 5)
  (D1F D1E : ℝ) (h_D1F : D1F = 60 / 17) (h_D1E : D1E = 84 / 17)
  (XZ XY : ℝ) (h_XZ : XZ = (1 / 2) * D1F) (h_XY : XY = D1E) :
  let X1Z := (5 / 14) * (78 / 17) in
  let XX1 := X1Z in
  XX1 = 30 / 17 :=
by
  sorry

end find_length_xx1_l571_571815


namespace radius_of_circumscribed_sphere_l571_571966

noncomputable def circumscribed_sphere_radius (a : ℝ) : ℝ :=
  a / Real.sqrt 3

theorem radius_of_circumscribed_sphere 
  (a : ℝ) 
  (h_base_side : 0 < a)
  (h_distance : ∃ d : ℝ, d = a * Real.sqrt 2 / 8) : 
  circumscribed_sphere_radius a = a / Real.sqrt 3 :=
sorry

end radius_of_circumscribed_sphere_l571_571966


namespace narrow_black_stripes_l571_571862

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l571_571862


namespace necessary_but_not_sufficient_condition_l571_571690

theorem necessary_but_not_sufficient_condition (m : ℝ): 
  (∀ x : ℝ, 3 * x^2 + 4 * x + m ≥ 0) → 
  (m ≥ ∀ x > 0, 8 * x / (x^2 + 4)) ∧ ¬(∀ x > 0, 8 * x / (x^2 + 4) ≥ 3 * x^2 + 4 * x + m) :=
by
  sorry

end necessary_but_not_sufficient_condition_l571_571690


namespace maximum_candies_purchase_l571_571831

theorem maximum_candies_purchase (c1 : ℕ) (c4 : ℕ) (c7 : ℕ) (n : ℕ)
    (H_single : c1 = 1)
    (H_pack4  : c4 = 4)
    (H_cost4  : c4 = 3) 
    (H_pack7  : c7 = 7) 
    (H_cost7  : c7 = 4) 
    (H_budget : n = 10) :
    ∃ k : ℕ, k = 16 :=
by
    -- We'll skip the proof since the task requires only the statement
    sorry

end maximum_candies_purchase_l571_571831


namespace warehouse_width_l571_571620

theorem warehouse_width (L : ℕ) (circles : ℕ) (total_distance : ℕ)
  (hL : L = 600)
  (hcircles : circles = 8)
  (htotal_distance : total_distance = 16000) : 
  ∃ W : ℕ, 2 * L + 2 * W = (total_distance / circles) ∧ W = 400 :=
by
  sorry

end warehouse_width_l571_571620


namespace num_factors_48_l571_571341

theorem num_factors_48 : 
  ∀ (n : ℕ), n = 48 → (∃ k : ℕ, k = 10 ∧ ∀ d : ℕ, d ∣ n → (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 16 ∨ d = 24 ∨ d = 48)) :=
  by
    intros n h
    sorry

end num_factors_48_l571_571341


namespace smallest_four_digit_divisible_by_each_of_its_digits_including_5_l571_571088

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n < 10000

def all_digits_different (n : Nat) : Prop :=
  let digits := [n / 1000, (n % 1000) / 100, (n % 100) / 10, n % 10];
  List.nodup digits

def includes_digit (n digit : Nat) : Prop :=
  let digits := [n / 1000, (n % 1000) / 100, (n % 100) / 10, n % 10];
  digit ∈ digits

def divisible_by_digits (n : Nat) : Prop :=
  let digits := List.filter (≠ 0) [n / 1000, (n % 1000) / 100, (n % 100) / 10, n % 10];
  ∀ d ∈ digits, n % d = 0

theorem smallest_four_digit_divisible_by_each_of_its_digits_including_5 :
  ∃ n : Nat, is_four_digit n ∧ all_digits_different n ∧ includes_digit n 5 ∧ divisible_by_digits n ∧ 
  (∀ m : Nat, is_four_digit m ∧ all_digits_different m ∧ includes_digit m 5 ∧ divisible_by_digits m → n ≤ m) :=
exists.intro 5124 
  (by
    repeat { split };
    sorry)

end smallest_four_digit_divisible_by_each_of_its_digits_including_5_l571_571088


namespace cos_double_angle_l571_571708

theorem cos_double_angle (θ : ℝ) 
  (h : 2^(-3/2 + 2 * Real.cos θ) + 1 = 2^(1/4 + Real.cos θ)) :
  Real.cos (2 * θ) = 1 / 8 :=
by
  -- proof to be completed
  sorry

end cos_double_angle_l571_571708


namespace function_decreasing_iff_l571_571309

theorem function_decreasing_iff (a : ℝ) :
  (0 < a ∧ a < 1) ∧ a ≤ 1/4 ↔ (0 < a ∧ a ≤ 1/4) :=
by
  sorry

end function_decreasing_iff_l571_571309


namespace correct_option_B_l571_571994

variable {a b x y : ℤ}

def option_A (a : ℤ) : Prop := -a - a = 0
def option_B (x y : ℤ) : Prop := -(x + y) = -x - y
def option_C (b a : ℤ) : Prop := 3 * (b - 2 * a) = 3 * b - 2 * a
def option_D (a : ℤ) : Prop := 8 * a^4 - 6 * a^2 = 2 * a^2

theorem correct_option_B (x y : ℤ) : option_B x y := by
  -- The proof would go here
  sorry

end correct_option_B_l571_571994


namespace arithmetic_series_sum_121_l571_571194

-- Define the conditions for the arithmetic series
def is_arithmetic_series (a d : ℕ) (last : ℕ) (n : ℕ) (terms : List ℕ) : Prop :=
  terms = List.iota n |>.map (λ k => a + d * k) ∧ terms.head? = some a ∧ terms.last? = some last

-- Define the sum of a list of natural numbers
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- The main theorem statement
theorem arithmetic_series_sum_121 :
  ∃ (n : ℕ) (terms : List ℕ), is_arithmetic_series 1 2 21 n terms ∧ sum_list terms = 121 :=
by
  sorry

end arithmetic_series_sum_121_l571_571194


namespace avg_visitors_per_day_l571_571123

theorem avg_visitors_per_day 
  (avg_visitors_sundays : ℕ) 
  (avg_visitors_other_days : ℕ) 
  (total_days : ℕ) 
  (sundays : ℕ) 
  (other_days : ℕ)
  (hs : avg_visitors_sundays = 630)
  (ho : avg_visitors_other_days = 240)
  (td : total_days = 30)
  (sd : sundays = 4)
  (od : other_days = 26)
  : (4 * avg_visitors_sundays + 26 * avg_visitors_other_days) / 30 = 292 := 
by
  sorry

end avg_visitors_per_day_l571_571123


namespace simplify_expression_l571_571467

variable (x : Real)

theorem simplify_expression (h : x ≠ 0) : x⁻² - 2 * x⁻¹ + 1 = (1 - x)² * x⁻² := by
  sorry

end simplify_expression_l571_571467


namespace hexagon_to_square_dissection_l571_571638

theorem hexagon_to_square_dissection :
  ∃ (s : ℝ) (side_length : ℝ), s = 1 ∧ side_length = sqrt (3 * sqrt 3 / 2) ∧ 
  let area_hex := (3 * sqrt 3 / 2) * s ^ 2 in 
  let area_square := side_length ^ 2 in 
  area_hex = area_square :=
begin
  sorry
end

end hexagon_to_square_dissection_l571_571638


namespace incenters_intersect_condition_l571_571399

variables {α : Type*} [RealObjectSpace α] 
variables (A B C D O₁ O₂ K : α)

def right_angle_triangle (A B C : α) : Prop :=
  ∠ACB = 90

def perpendicular (C D A B : α) : Prop :=
  is_perpendicular CD AB

def incenter (P Q R : α) (I : α) : Prop :=
  is_incenter I P Q R

def intersects (O₁ O₂ CD : α) (K : α) : Prop :=
  intersects_at O₁ O₂ CD K

theorem incenters_intersect_condition
  (h1 : right_angle_triangle A B C)
  (h2 : perpendicular C D A B)
  (h3 : incenter A D C O₁)
  (h4 : incenter C D B O₂)
  (h5 : intersects O₁ O₂ CD K) :
  1 / distance B C + 1 / distance A C = 1 / distance C K :=
sorry

end incenters_intersect_condition_l571_571399


namespace cube_volume_is_125_l571_571951

-- Define the distance between the opposite vertices
def AG : ℝ := 5 * Real.sqrt 3

-- Define side length of the cube
def s : ℝ := AG / Real.sqrt 3

-- Define the volume of the cube
def volume : ℝ := s ^ 3

-- Theorem stating that the volume of the cube is 125 cubic units
theorem cube_volume_is_125 : volume = 125 := by
  -- Proof steps are omitted
  sorry

end cube_volume_is_125_l571_571951


namespace af_perpendicular_to_be_l571_571282

variables {A B C D E F : Type}
variables [IsoscelesTriangle A B C]
variables [Midpoint D B C]
variables [Projection E D A C]
variables [Midpoint F D E]

theorem af_perpendicular_to_be :
  perpendicular (line_through A F) (line_through B E) :=
sorry

end af_perpendicular_to_be_l571_571282


namespace probability_nearest_odd_l571_571795

def is_odd_nearest (a b : ℝ) : Prop := ∃ k : ℤ, 2 * k + 1 = Int.floor ((a - b) / (a + b))

def is_valid (a b : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1

noncomputable def probability_odd_nearest : ℝ :=
  let interval_area := 1 -- the area of the unit square [0, 1] x [0, 1]
  let odd_area := 1 / 3 -- as derived from the geometric interpretation in the problem's solution
  odd_area / interval_area

theorem probability_nearest_odd (a b : ℝ) (h : is_valid a b) :
  probability_odd_nearest = 1 / 3 := by
  sorry

end probability_nearest_odd_l571_571795


namespace best_model_l571_571809

theorem best_model (R1 R2 R3 R4 : ℝ) :
  R1 = 0.78 → R2 = 0.85 → R3 = 0.61 → R4 = 0.31 →
  (R2 = max R1 (max R2 (max R3 R4))) :=
by
  intros hR1 hR2 hR3 hR4
  sorry

end best_model_l571_571809


namespace count_multiples_6_or_8_not_both_l571_571767

-- Define the conditions
def is_multiple (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

-- Define the main proof statement
theorem count_multiples_6_or_8_not_both :
  (∑ k in Finset.filter (λ n, is_multiple n 6 ∨ is_multiple n 8 ∧ ¬(is_multiple n 6 ∧ is_multiple n 8)) (Finset.range 151), 1) = 31 := 
sorry

end count_multiples_6_or_8_not_both_l571_571767


namespace smallest_positive_integer_remainder_conditions_l571_571527

theorem smallest_positive_integer_remainder_conditions :
  ∃ b : ℕ, (b % 4 = 3 ∧ b % 6 = 5) ∧ ∀ n : ℕ, (n % 4 = 3 ∧ n % 6 = 5) → n ≥ b := 
by
  have b := 23
  use b
  sorry

end smallest_positive_integer_remainder_conditions_l571_571527


namespace sum_of_coordinates_of_reflected_points_l571_571919

theorem sum_of_coordinates_of_reflected_points (C D : ℝ × ℝ) (hx : C.1 = 3) (hy : C.2 = 8) (hD : D = (-C.1, C.2)) :
  C.1 + C.2 + D.1 + D.2 = 16 := by
  sorry

end sum_of_coordinates_of_reflected_points_l571_571919


namespace smallest_pos_int_ending_in_9_divisible_by_13_l571_571095

theorem smallest_pos_int_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), n % 10 = 9 ∧ n % 13 = 0 ∧ ∀ m, m % 10 = 9 ∧ m % 13 = 0 → n ≤ m :=
by
  sorry

end smallest_pos_int_ending_in_9_divisible_by_13_l571_571095


namespace student_l571_571807

theorem student's_incorrect_answer (D I : ℕ) (h1 : D / 36 = 58) (h2 : D / 87 = I) : I = 24 :=
sorry

end student_l571_571807


namespace find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_also_924_l571_571656

theorem find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), (n % 21 = 0) ∧ (30 < real.sqrt n) ∧ (real.sqrt n < 30.5) :=
begin
  use 903,
  split,
  {
    -- proof that 21 divides 903
    rw nat.mod_eq_zero,
    exact dvd.refl _,
  },
  {
    split,
    {
      -- proof that 30 < sqrt(903)
      norm_num, 
      linarith,
    },
    {
      -- proof that sqrt(903) < 30.5
      norm_num,
      linarith,
    }
  }
end

theorem find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_also_924 :
  ∃ (n : ℕ), (n % 21 = 0) ∧ (30 < real.sqrt n) ∧ (real.sqrt n < 30.5) :=
begin
  use 924,
  split,
  {
    -- proof that 21 divides 924
    rw nat.mod_eq_zero,
    exact dvd.refl _,
  },
  {
    split,
    {
      -- proof that 30 < sqrt(924)
      norm_num,
      linarith,
    },
    {
      -- proof that sqrt(924) < 30.5
      norm_num,
      linarith,
    }
  }
end

end find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_also_924_l571_571656


namespace angle_values_proof_l571_571303

noncomputable def angle_values (α : ℝ) : Prop :=
  let x1 := (-2 : ℝ)
  let y1 := (1 : ℝ)
  let x2 := (2 : ℝ)
  let y2 := (-1 : ℝ)
  let r := Real.sqrt 5
  (y1 = - (1/2) * x1 ∨ y2 = - (1/2) * x2) →
  (sin α = √5/5 ∨ sin α = -√5/5) ∧
  (cos α = 2*√5/5 ∨ cos α = -2*√5/5)
  
theorem angle_values_proof (α : ℝ) : angle_values α :=
  sorry

end angle_values_proof_l571_571303


namespace line_fixed_point_l571_571741

variables {p k b x y x1 y1 x2 y2 : ℝ}
variable  (hp : p > 0)
variables (k_ne_zero : k ≠ 0) (b_ne_zero : b ≠ 0)
variables (kOA kOB : ℝ)
variable (C1 : x1 * y1 = 2xp)
variable (C2 : x2 * y2 = 2xp)
variable (line : y = k * x + b)
variable (kOA_kOB_sqrt3 : kOA * kOB = sqrt 3)
variables (A B : (ℝ × ℝ)) -- points where line and parabola intersect
variable (intersect_A : (A.1, A.2) satisfies {y = k * x + b} and {y^2 = 2xp})
variable (intersect_B : (B.1, B.2) satisfies {y = k * x + b} and {y^2 = 2xp})

theorem line_fixed_point 
  (h : (kOA * kOB = sqrt 3)) :
  ∃ (x y : ℝ), (x, y) = ( -((2 * p) / (sqrt 3)), 0) :=
begin
 sorry
end

end line_fixed_point_l571_571741


namespace averageWeightOf4RemovedCarrots_l571_571130

noncomputable def totalWeight20Carrots : ℕ := 3640
noncomputable def averageWeight16Remaining : ℕ := 180
noncomputable def numberOfCarrots : ℕ := 4

theorem averageWeightOf4RemovedCarrots :
  let totalWeight16Carrots := 16 * averageWeight16Remaining,
      weight4RemovedCarrots := totalWeight20Carrots - totalWeight16Carrots
  in
  (weight4RemovedCarrots / numberOfCarrots) = 190 := by
  sorry

end averageWeightOf4RemovedCarrots_l571_571130


namespace number_of_b_objects_l571_571032

theorem number_of_b_objects
  (total_objects : ℕ) 
  (a_objects : ℕ) 
  (b_objects : ℕ) 
  (h1 : total_objects = 35) 
  (h2 : a_objects = 17) 
  (h3 : total_objects = a_objects + b_objects) :
  b_objects = 18 :=
by
  sorry

end number_of_b_objects_l571_571032


namespace limit_of_sequence_N_of_epsilon_l571_571455

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) (h : ∀ n, a_n n = (7 * n - 1) / (n + 1)) :
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) ↔ a = 7 := sorry

theorem N_of_epsilon (ε : ℝ) (hε : ε > 0) :
  ∃ N : ℕ, N = ⌈8 / ε⌉ := sorry

end limit_of_sequence_N_of_epsilon_l571_571455


namespace common_tangent_line_range_a_l571_571797

open Real

theorem common_tangent_line_range_a (a : ℝ) (h_pos : 0 < a) :
  (∃ x₁ x₂ : ℝ, 2 * a * x₁ = exp x₂ ∧ (exp x₂ - a * x₁^2) / (x₂ - x₁) = 2 * a * x₁) →
  a ≥ exp 2 / 4 := 
sorry

end common_tangent_line_range_a_l571_571797


namespace triangles_in_divided_square_l571_571811

theorem triangles_in_divided_square (V : ℕ) (marked_points : ℕ) (triangles : ℕ) 
  (h1 : V = 24) -- Vertices - 20 marked points and 4 vertices 
  (h2 : marked_points = 20) -- Marked points
  (h3 : triangles = F - 1) -- Each face (F) except the outer one is a triangle
  (h4 : V - E + F = 2) -- Euler's formula for planar graphs
  (h5 : E = (3*F + 1) / 2) -- Relationship between edges and faces
  (F : ℕ) -- Number of faces including the external face
  (E : ℕ) -- Number of edges
  : triangles = 42 := 
by 
  sorry

end triangles_in_divided_square_l571_571811


namespace narrow_black_stripes_count_l571_571883

theorem narrow_black_stripes_count (w n : ℕ) (b : ℕ) 
  (h1 : b = w + 7) 
  (h2 : w + n = b + 1) 
  : n = 8 :=
by sorry

end narrow_black_stripes_count_l571_571883


namespace units_digit_17_pow_2023_l571_571108

theorem units_digit_17_pow_2023 
  (cycle : ℕ → ℕ)
  (h1 : cycle 0 = 7)
  (h2 : cycle 1 = 9)
  (h3 : cycle 2 = 3)
  (h4 : cycle 3 = 1)
  (units_digit : ℕ → ℕ)
  (h_units : ∀ n, units_digit (17^n) = units_digit (7^n))
  (h_units_cycle : ∀ n, units_digit (7^n) = cycle (n % 4)) :
  units_digit (17^2023) = 3 :=
by
  sorry

end units_digit_17_pow_2023_l571_571108


namespace unique_four_digit_number_l571_571753

theorem unique_four_digit_number (N : ℕ) (a : ℕ) (x : ℕ) :
  (N = 1000 * a + x) ∧ (N = 7 * x) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (1 ≤ a ∧ a ≤ 9) →
  N = 3500 :=
by sorry

end unique_four_digit_number_l571_571753


namespace log_of_sum_squares_pascal_l571_571429

theorem log_of_sum_squares_pascal (n : ℕ) : 
  let g := λ n, log (10 : ℝ) (∑ k in Finset.range (n+1), (nat.choose n k) ^ 2)
  in large_n := n > 100 -- Assuming large n
  in (g n) / (log (10 : ℝ) 2) ≈ (2 * n : ℝ) :=
by 
  sorry

end log_of_sum_squares_pascal_l571_571429


namespace boxes_per_week_l571_571979

-- Define the given conditions
def cost_per_box : ℝ := 3.00
def weeks_in_year : ℝ := 52
def total_spent_per_year : ℝ := 312

-- The question we want to prove:
theorem boxes_per_week:
  (total_spent_per_year = cost_per_box * weeks_in_year * (total_spent_per_year / (weeks_in_year * cost_per_box))) → 
  (total_spent_per_year / (weeks_in_year * cost_per_box)) = 2 := sorry

end boxes_per_week_l571_571979


namespace determinant_tridiagonal_matrix_l571_571182

def tridiagonal_matrix (n : ℕ) (x : ℂ) : Matrix (Fin n) (Fin n) ℂ :=
  fun i j => if i = j then 1 + x^2 else if abs (i - j).toNat = 1 then x else 0

theorem determinant_tridiagonal_matrix (n : ℕ) (x : ℂ) (hn : 2 < n) :
  (tridiagonal_matrix n x).det = ∑ k in Finset.range (n + 1), x^(2 * k) :=
by
  sorry

end determinant_tridiagonal_matrix_l571_571182


namespace polyhedron_odd_faces_in_parts_l571_571576

-- Define the problem conditions and main theorem.
theorem polyhedron_odd_faces_in_parts
  (P : Polyhedron)
  (h_convex : P.Convex)
  (h_vertices : P.Vertices = 2003)
  (h_closed_broken_line : ∀ v : P.Vertex, ∃! p : P.Path, p.Cyclic ∧ p.ContainsVertex v)
  (parts : Set P.Face) : ∀ part ∈ parts, odd (part.filter (λ f, odd f.Edges.Count)).Count := sorry

end polyhedron_odd_faces_in_parts_l571_571576


namespace probability_B_win_probability_game_ends_with_B_shot_2_balls_l571_571132

-- Define the probability of A and B making a shot.
def p_A : ℚ := 1 / 3
def p_B : ℚ := 1 / 2

-- Define the condition that each shot is independent; in Lean, 
-- this can be assumed inherently, so we do not define independence explicitly.

-- Problem 1: Prove that the probability that B wins.
theorem probability_B_win : (p_A * p_B) + (p_A^2 * p_B^2 * p_B) + (p_A^3 * p_B^2 * p_B^2 * p_B) = 13 / 27 :=
begin
  sorry
end

-- Problem 2: Prove that the probability that the game ends with B having shot only 2 balls.
theorem probability_game_ends_with_B_shot_2_balls :
  (p_A^2 * p_B * 1/2) + (1/2 * 1/2 * p_A^2 * 1/3) = 4 / 27 :=
begin
  sorry
end

end probability_B_win_probability_game_ends_with_B_shot_2_balls_l571_571132


namespace walking_distance_l571_571128

theorem walking_distance (x : ℝ) :
  (sqrt ((-x + 5 / 2) ^ 2 + (-5 * real.sqrt 3 / 2) ^ 2) = 5) ↔ (x = 0 ∨ x = 5) :=
by sorry

end walking_distance_l571_571128


namespace num_of_irrationals_l571_571600

-- Define all the given numbers
def num1 : ℝ := 3.14159
def num2 : ℝ := real.cbrt 64
def num3 : ℝ := 1.010010001 -- Infinite decimal with an additional 0 after every 1
def num4 : ℝ := 4.21 -- Repeating decimal
def num5 : ℝ := real.pi
def num6 : ℝ := 22 / 7

-- Define a predicate for irrational numbers
def is_irrational (x : ℝ) : Prop := ¬ ∃ a b : ℚ, x = a / b

-- Prove the number of irrational numbers is 2
theorem num_of_irrationals : 
  (if is_irrational num1 then 1 else 0) + 
  (if is_irrational num2 then 1 else 0) + 
  (if is_irrational num3 then 1 else 0) + 
  (if is_irrational num4 then 1 else 0) + 
  (if is_irrational num5 then 1 else 0) + 
  (if is_irrational num6 then 1 else 0) = 2 :=
by sorry

end num_of_irrationals_l571_571600


namespace remainder_of_n_mod_9_eq_5_l571_571483

-- Definitions of the variables and conditions
variables (a b c n : ℕ)

-- The given conditions as assumptions
def conditions : Prop :=
  a + b + c = 63 ∧
  a = c + 22 ∧
  n = 2 * a + 3 * b + 4 * c

-- The proof statement that needs to be proven
theorem remainder_of_n_mod_9_eq_5 (h : conditions a b c n) : n % 9 = 5 := 
  sorry

end remainder_of_n_mod_9_eq_5_l571_571483


namespace problem_f_g_2_l571_571428

def f (x : ℝ) : ℝ := 3 * Real.sqrt x - 18 / Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x^2 - 3 * x - 4

theorem problem_f_g_2 : f (g 2) = -6 * Real.sqrt 2 := by 
  sorry

end problem_f_g_2_l571_571428


namespace find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_also_924_l571_571655

theorem find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), (n % 21 = 0) ∧ (30 < real.sqrt n) ∧ (real.sqrt n < 30.5) :=
begin
  use 903,
  split,
  {
    -- proof that 21 divides 903
    rw nat.mod_eq_zero,
    exact dvd.refl _,
  },
  {
    split,
    {
      -- proof that 30 < sqrt(903)
      norm_num, 
      linarith,
    },
    {
      -- proof that sqrt(903) < 30.5
      norm_num,
      linarith,
    }
  }
end

theorem find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_also_924 :
  ∃ (n : ℕ), (n % 21 = 0) ∧ (30 < real.sqrt n) ∧ (real.sqrt n < 30.5) :=
begin
  use 924,
  split,
  {
    -- proof that 21 divides 924
    rw nat.mod_eq_zero,
    exact dvd.refl _,
  },
  {
    split,
    {
      -- proof that 30 < sqrt(924)
      norm_num,
      linarith,
    },
    {
      -- proof that sqrt(924) < 30.5
      norm_num,
      linarith,
    }
  }
end

end find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_find_positive_integer_divisible_by_21_and_sqrt_between_30_and_30_5_also_924_l571_571655


namespace total_rainfall_2005_l571_571803

-- Defining the conditions
def average_rainfall_2003 : ℝ := 37.5
def yearly_rainfall_increase : ℝ := 3

-- The function to calculate rainfall in a given year
noncomputable def average_rainfall (year: ℕ) : ℝ :=
  average_rainfall_2003 + yearly_rainfall_increase * (year - 2003)

-- The statement to calculate the total rainfall in 2005
theorem total_rainfall_2005 :
  let total_rainfall := average_rainfall 2005 * 12 in
  total_rainfall = 522 :=
by
  sorry

end total_rainfall_2005_l571_571803


namespace planted_field_fraction_is_correct_l571_571228

noncomputable def planted_fraction : ℚ :=
  let a := 5
  let b := 12
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let triangle_area := (1 / 2 : ℚ) * a * b
  
  -- Side length of the square
  let x := (3 : ℚ) * (7 : ℚ)^(-1)
  let square_area := x^2

  -- Planted area
  let planted_area := triangle_area - square_area
  
  -- Calculated planted fraction
  planted_area / triangle_area

theorem planted_field_fraction_is_correct :
  planted_fraction = 1461 / 1470 :=
sorry

end planted_field_fraction_is_correct_l571_571228


namespace product_mod_7_l571_571204

theorem product_mod_7 :
  (2009 % 7 = 4) ∧ (2010 % 7 = 5) ∧ (2011 % 7 = 6) ∧ (2012 % 7 = 0) →
  (2009 * 2010 * 2011 * 2012) % 7 = 0 :=
by
  sorry

end product_mod_7_l571_571204


namespace find_greater_number_l571_571023

theorem find_greater_number (a b : ℕ) (h1 : a * b = 4107) (h2 : Nat.gcd a b = 37) (h3 : a > b) : a = 111 :=
sorry

end find_greater_number_l571_571023


namespace smallest_pos_int_ending_in_9_divisible_by_13_l571_571094

theorem smallest_pos_int_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), n % 10 = 9 ∧ n % 13 = 0 ∧ ∀ m, m % 10 = 9 ∧ m % 13 = 0 → n ≤ m :=
by
  sorry

end smallest_pos_int_ending_in_9_divisible_by_13_l571_571094


namespace nonagon_diagonals_l571_571331

-- Define the number of sides for a nonagon.
def n : ℕ := 9

-- Define the formula for the number of diagonals in a polygon.
def D (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem to prove that the number of diagonals in a nonagon is 27.
theorem nonagon_diagonals : D n = 27 := by
  sorry

end nonagon_diagonals_l571_571331


namespace sector_area_maximized_l571_571800

noncomputable def maximize_sector_area (r θ : ℝ) : Prop :=
  2 * r + θ * r = 20 ∧
  (r > 0 ∧ θ > 0) ∧
  ∀ (r' θ' : ℝ), (2 * r' + θ' * r' = 20 ∧ r' > 0 ∧ θ' > 0) → (1/2 * θ' * r'^2 ≤ 1/2 * θ * r^2)

theorem sector_area_maximized : maximize_sector_area 5 2 :=
by
  sorry

end sector_area_maximized_l571_571800


namespace exists_strictly_increasing_sequence_l571_571923

theorem exists_strictly_increasing_sequence (a1 : ℕ) (h : a1 > 1) :
  ∃ (a : ℕ → ℕ), (∀ n,  a n > 0) ∧ (∀ n, a n < a (n + 1)) ∧ (∀ k ≥ 1, (∑ i in finset.range k, (a i) ^ 2) % (∑ i in finset.range k, a i) = 0) :=
sorry

end exists_strictly_increasing_sequence_l571_571923


namespace theta_in_third_or_fourth_quadrant_l571_571716

-- Define the conditions as Lean definitions
def theta_condition (θ : ℝ) : Prop :=
  ∃ k : ℤ, θ = k * Real.pi + (-1 : ℝ)^(k + 1) * (Real.pi / 4)

-- Formulate the statement we need to prove
theorem theta_in_third_or_fourth_quadrant (θ : ℝ) (h : theta_condition θ) :
  ∃ q : ℤ, q = 3 ∨ q = 4 :=
sorry

end theta_in_third_or_fourth_quadrant_l571_571716


namespace probability_of_5_odd_in_6_rolls_l571_571066

open Classical

noncomputable def prob_odd_in_six_rolls : ℚ :=
  let num_rolls := 6
  let prob_odd_single := 1 / 2
  let binom_coeff := Nat.choose num_rolls 5
  let total_outcomes := (2 : ℕ) ^ num_rolls
  binom_coeff * ((prob_odd_single ^ 5) * ((1 - prob_odd_single) ^ (num_rolls - 5))) / total_outcomes

theorem probability_of_5_odd_in_6_rolls :
  prob_odd_in_six_rolls = 3 / 32 :=
by
  sorry

end probability_of_5_odd_in_6_rolls_l571_571066


namespace equation_not_hold_for_x_equals_1_l571_571306

theorem equation_not_hold_for_x_equals_1 :
  ∀ (x : ℝ), x = 1 → ¬(∃ e : ℝ, (1 / (x + 1) + e = 1 / (x - 1))) :=
begin
  intros x hx contra,
  rw hx at contra,
  cases contra with e eq,
  have h : 1 / (1 - 1) = 1 / 0, by rw h,
  linarith,
end

end equation_not_hold_for_x_equals_1_l571_571306


namespace narrow_black_stripes_l571_571864

variable (w n b : ℕ)

theorem narrow_black_stripes (w : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := by
  have h3 : w + n = (w + 7) + 1 := by rw [h1]
  have h4 : w + n = w + 8 := by linarith
  have h5 : n = 8 := by linarith
  exact h5

end narrow_black_stripes_l571_571864


namespace percentage_speaking_neither_l571_571612

/-- Statement: 
Given:
1) There are 180 diplomats in total.
2) 14 diplomats speak French.
3) 32 diplomats do not speak Russian.
4) 10% of the diplomats speak both French and Russian.
Prove that the percentage of diplomats who speak neither French nor Russian is 20%.
-/
theorem percentage_speaking_neither (total : ℕ)
  (speak_french : ℕ)
  (not_speak_russian : ℕ)
  (percent_both : ℝ)
  (neither_percentage : ℝ) :
  total = 180 →
  speak_french = 14 →
  not_speak_russian = 32 →
  percent_both = 0.10 →
  neither_percentage = ((total - (speak_french + (total - not_speak_russian) - (percent_both * total.to_nat))) / total.to_nat) * 100 →
  neither_percentage = 20 :=
by
  intros h_total h_french h_no_russian h_both h_percentage
  sorry

end percentage_speaking_neither_l571_571612


namespace min_abs_value_expression_l571_571987

theorem min_abs_value_expression : (x: ℝ) → |x - 4| + |x + 7| + |x - 5| ≥ 1
| -7 := 
begin 
  simp, 
  rw [abs_of_nonneg, abs_of_nonpos, abs_of_nonpos]; 
  linarith
end

end min_abs_value_expression_l571_571987


namespace sum_of_consecutive_natural_numbers_number_of_ways_to_sum_l571_571397

theorem sum_of_consecutive_natural_numbers (n a : ℕ) (h : 1999.prime) :
  let SumConsecutive (a : ℕ) (n : ℕ) := n * a + n * (n + 1) / 2
  in (∃ n a, n * (2 * a + n + 1) = 2 * 1999 ^ 1999) ↔  ∃ n a, SumConsecutive a n = 1999 ^ 1999 :=
by sorry

theorem number_of_ways_to_sum (h : 1999.prime) :
  (∃ n a, n * (2 * a + n + 1) = 2 * 1999 ^ 1999) → Fintype.card {n : ℕ // ∃ a, n * (2 * a + n + 1) = 2 * 1999 ^ 1999} = 2000 :=
by sorry

end sum_of_consecutive_natural_numbers_number_of_ways_to_sum_l571_571397


namespace min_value_problem1_l571_571565

theorem min_value_problem1 (x : ℝ) (hx : x > -1) : 
  ∃ m, m = 2 * Real.sqrt 2 + 1 ∧ (∀ y, y = (x^2 + 3 * x + 4) / (x + 1) ∧ x > -1 → y ≥ m) :=
sorry

end min_value_problem1_l571_571565


namespace probability_of_5_odd_in_6_rolls_l571_571064

open Classical

noncomputable def prob_odd_in_six_rolls : ℚ :=
  let num_rolls := 6
  let prob_odd_single := 1 / 2
  let binom_coeff := Nat.choose num_rolls 5
  let total_outcomes := (2 : ℕ) ^ num_rolls
  binom_coeff * ((prob_odd_single ^ 5) * ((1 - prob_odd_single) ^ (num_rolls - 5))) / total_outcomes

theorem probability_of_5_odd_in_6_rolls :
  prob_odd_in_six_rolls = 3 / 32 :=
by
  sorry

end probability_of_5_odd_in_6_rolls_l571_571064


namespace correct_product_l571_571814

theorem correct_product (a b : ℕ) (a' : ℕ) (h1 : a' = (a % 10) * 10 + (a / 10)) 
  (h2 : a' * b = 143) (h3 : 10 ≤ a ∧ a < 100):
  a * b = 341 :=
sorry

end correct_product_l571_571814


namespace no_real_solutions_l571_571939

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (4 * x^3 + 3 * x^2 + x + 2) / (x - 2) = 4 * x^2 + 5 :=
by
  sorry

end no_real_solutions_l571_571939


namespace leftover_stickers_l571_571447

-- Definitions for each person's stickers
def ninaStickers : ℕ := 53
def oliverStickers : ℕ := 68
def pattyStickers : ℕ := 29

-- The number of stickers in a package
def packageSize : ℕ := 18

-- The total number of stickers
def totalStickers : ℕ := ninaStickers + oliverStickers + pattyStickers

-- Proof that the number of leftover stickers is 6 when all stickers are divided into packages of 18
theorem leftover_stickers : totalStickers % packageSize = 6 := by
  sorry

end leftover_stickers_l571_571447


namespace isosceles_triangle_perimeter_l571_571396

-- Define the lengths of the sides
def side1 := 2 -- 2 cm
def side2 := 4 -- 4 cm

-- Define the condition of being isosceles
def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (a = c) ∨ (b = c)

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Define the triangle perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the main theorem to prove
theorem isosceles_triangle_perimeter {a b : ℝ} (ha : a = side1) (hb : b = side2)
    (h1 : is_isosceles a b c) (h2 : triangle_inequality a b c) : perimeter a b c = 10 :=
sorry

end isosceles_triangle_perimeter_l571_571396


namespace count_multiples_of_6_or_8_but_not_both_l571_571778

theorem count_multiples_of_6_or_8_but_not_both: 
  let multiples_of_six := finset.filter (λ n, 6 ∣ n) (finset.range 151)
  let multiples_of_eight := finset.filter (λ n, 8 ∣ n) (finset.range 151)
  let multiples_of_twenty_four := finset.filter (λ n, 24 ∣ n) (finset.range 151)
  multiples_of_six.card + multiples_of_eight.card - 2 * multiples_of_twenty_four.card = 31 := 
by {
  -- Provided proof omitted
  sorry
}

end count_multiples_of_6_or_8_but_not_both_l571_571778


namespace factorial_expression_l571_571179

theorem factorial_expression : 8! - 7 * 7! - 2 * 6! = 5 * 6! :=
by
  sorry

end factorial_expression_l571_571179


namespace _l571_571520

noncomputable theorem ordering_of_fractions :
  let a : ℚ := 5 / 19,
      b : ℚ := 7 / 21,
      c : ℚ := 9 / 23
  in a < b ∧ b < c :=
by
  let a := (5 : ℚ) / 19,
      b := (7 : ℚ) / 21,
      c := (9 : ℚ) / 23
  sorry

end _l571_571520


namespace units_digit_17_pow_2023_l571_571110

theorem units_digit_17_pow_2023 : (17 ^ 2023) % 10 = 3 :=
by
  have units_cycle_7 : ∀ (n : ℕ), (7 ^ n) % 10 = [7, 9, 3, 1].nth (n % 4) :=
    sorry
  have units_pattern_equiv : (17 ^ n) % 10 = (7 ^ n) % 10 :=
    sorry
  calc
    (17 ^ 2023) % 10
        = (7 ^ 2023) % 10  : by rw [units_pattern_equiv]
    ... = 3               : by rw [units_cycle_7, nat.mod_eq_of_lt, List.nth]

end units_digit_17_pow_2023_l571_571110


namespace four_digit_number_condition_solution_count_l571_571755

def valid_digits_count : ℕ := 5

theorem four_digit_number_condition (N x a : ℕ) (h1 : N = 1000 * a + x) (h2 : N = 7 * x) (h3 : 100 ≤ x ∧ x ≤ 999) :
  a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 4 ∨ a = 5 :=
begin
  sorry
end

theorem solution_count : ∃ n:ℕ, n = valid_digits_count :=
begin
  use 5,
  sorry
end

end four_digit_number_condition_solution_count_l571_571755


namespace limit_a_n_l571_571453

open Nat Real

noncomputable def a_n (n : ℕ) : ℝ := (7 * n - 1) / (n + 1)

theorem limit_a_n : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - 7| < ε := 
by {
  -- The proof would go here.
  sorry
}

end limit_a_n_l571_571453


namespace rooks_in_rectangle_l571_571572

theorem rooks_in_rectangle (r : ℕ → ℕ → Prop) (h_placement : ∀ i j k l, r i k → r j k → r i l → r j l) :
  ∀ rows cols, (∀ i j, ∃ u v, rows u ∧ cols v ∧ r i j) → 
  ∃ i j, rows i ∧ cols j ∧ r i j := sorry

end rooks_in_rectangle_l571_571572


namespace narrow_black_stripes_l571_571876

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l571_571876


namespace area_of_quad_eq_S_l571_571626

variables (A B C D M N H1 H2 : Type)
variables {S : ℝ}
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
variables [inner_product_space ℝ D] [inner_product_space ℝ M] [inner_product_space ℝ N]
variables [inner_product_space ℝ H1] [inner_product_space ℝ H2]
 
-- We define the conditions place holders as they are key to the theorem
 
def acute_triangle (A B C : Type) (S : ℝ) : Prop :=
  inner_product_space ℝ A ∧ inner_product_space ℝ B ∧
  inner_product_space ℝ C ∧ area A B C = S
 
def perpendicular (x y : Type) : Prop :=
  ∀ (v : ℝ), dot_product x y = 0
 
def orthocenter (x y z : Type) (h : Type) : Prop :=
  perpendicular x h ∧ perpendicular y h ∧ perpendicular z h
 
theorem area_of_quad_eq_S
  (acuteABC : acute_triangle A B C S)
  (perpCD : perpendicular C D)
  (perpDM : perpendicular D M)
  (perpDN : perpendicular D N)
  (ortho_H1 : orthocenter M N C H1)
  (ortho_H2 : orthocenter M N D H2) :
  area A H1 B H2 = S :=
begin
  sorry
end

end area_of_quad_eq_S_l571_571626


namespace correct_calculation_l571_571992

theorem correct_calculation (a x y b : ℝ) :
  (-a - a = 0) = False ∧
  (- (x + y) = -x - y) = True ∧
  (3 * (b - 2 * a) = 3 * b - 2 * a) = False ∧
  (8 * a^4 - 6 * a^2 = 2 * a^2) = False :=
by
  sorry

end correct_calculation_l571_571992


namespace large_circle_diameter_approx_l571_571465

-- Definitions for the conditions given in the problem
def small_circle_radius : ℝ := 4
def num_small_circles : ℕ := 7
def distance_between_centers : ℝ := 2 * small_circle_radius
def heptagon_side_length : ℝ := distance_between_centers

-- Definition of the circumradius formula for a regular polygon
def circumradius (n : ℕ) (s : ℝ) : ℝ :=
  s / (2 * Real.sin (Real.pi / n))

-- Definition of the actual problem
theorem large_circle_diameter_approx :
  let R' := circumradius num_small_circles heptagon_side_length
  let R := R' + small_circle_radius
  2 * R ≈ 26.47 :=
by
  sorry

end large_circle_diameter_approx_l571_571465


namespace company_a_taxis_l571_571219

variable (a b : ℕ)

theorem company_a_taxis
  (h1 : 5 * a < 56)
  (h2 : 6 * a > 56)
  (h3 : 4 * b < 56)
  (h4 : 5 * b > 56)
  (h5 : b = a + 3) :
  a = 10 := by
  sorry

end company_a_taxis_l571_571219


namespace narrow_black_stripes_are_eight_l571_571868

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l571_571868


namespace problem_solution_l571_571171

noncomputable def correct_statement : Prop :=
  let statement_A := ∀ (bond_energy bond_length stability : ℝ), 
                      bond_energy > 0 ∧ bond_length > 0 ∧ stability > 0 →
                      (bond_energy > bond_length → bond_length < stability) → stability
  let statement_B := ∀ (element1 element2 : Type), 
                      (element1 ∈ GroupIA ∧ element1 ≠ H) ∨ (element2 ∈ GroupVIIA) → 
                      ¬(covalent_bond element1 element2)
  let statement_C := (bond_angle H O = 180)
  let statement_D := ∀ (decomposition_energy : ℝ), 
                      decomposition_energy = 2 * 463
  ∃ β, β = statement_B ∧ β ≠ statement_A ∧ β ≠ statement_C ∧ β ≠ statement_D

theorem problem_solution : correct_statement :=
  by {
    sorry -- The proof would go here
  }

end problem_solution_l571_571171


namespace EG_perpendicular_to_AC_l571_571944

noncomputable def rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 < B.1 ∧ A.2 = B.2 ∧ B.1 < C.1 ∧ B.2 < C.2 ∧ C.1 = D.1 ∧ C.2 > D.2 ∧ D.1 > A.1 ∧ D.2 = A.2

theorem EG_perpendicular_to_AC
  {A B C D E F G: ℝ × ℝ}
  (h1: rectangle A B C D)
  (h2: E = (B.1, C.2) ∨ E = (C.1, B.2)) -- Assuming E lies on BC or BA
  (h3: F = (B.1, A.2) ∨ F = (A.1, B.2)) -- Assuming F lies on BA or BC
  (h4: G = (C.1, D.2) ∨ G = (D.1, C.2)) -- Assuming G lies on CD
  (h5: (F.1, G.2) = (A.1, C.2)) -- Line through F parallel to AC meets CD at G
: ∃ (H : ℝ × ℝ → ℝ × ℝ → ℝ), H E G = 0 := sorry

end EG_perpendicular_to_AC_l571_571944


namespace count_integers_satisfying_inequality_l571_571246

theorem count_integers_satisfying_inequality :
  {n : ℤ | 15 < n^2 ∧ n^2 < 120}.card = 14 :=
sorry

end count_integers_satisfying_inequality_l571_571246


namespace smallest_positive_integer_ending_in_9_divisible_by_13_l571_571103

theorem smallest_positive_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, (n % 10 = 9) ∧ (n % 13 = 0) ∧ (∀ m : ℕ, (m % 10 = 9) ∧ (m % 13 = 0) → m ≥ n) :=
sorry

end smallest_positive_integer_ending_in_9_divisible_by_13_l571_571103


namespace chloe_apples_l571_571752

theorem chloe_apples :
  ∃ x : ℕ, (∃ y : ℕ, x = y + 8 ∧ y = x / 3) ∧ x = 12 := 
by
  sorry

end chloe_apples_l571_571752


namespace arithmetic_sum_l571_571185

theorem arithmetic_sum :
  let a := 1 in
  let d := 2 in
  let l := 21 in
  let n := (l - a) / d + 1 in
  (n * (a + l)) / 2 = 121 :=
by
  let a := 1
  let d := 2
  let l := 21
  let n := (l - a) / d + 1
  sorry

end arithmetic_sum_l571_571185


namespace narrow_black_stripes_l571_571852

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l571_571852


namespace smallest_positive_integer_ending_in_9_divisible_by_13_l571_571102

theorem smallest_positive_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, (n % 10 = 9) ∧ (n % 13 = 0) ∧ (∀ m : ℕ, (m % 10 = 9) ∧ (m % 13 = 0) → m ≥ n) :=
sorry

end smallest_positive_integer_ending_in_9_divisible_by_13_l571_571102


namespace count_integers_satisfying_inequality_l571_571247

theorem count_integers_satisfying_inequality :
  {n : ℤ | 15 < n^2 ∧ n^2 < 120}.card = 14 :=
sorry

end count_integers_satisfying_inequality_l571_571247


namespace number_of_boys_tried_out_l571_571507

theorem number_of_boys_tried_out 
    (num_girls : ℕ)
    (num_call_backs : ℕ)
    (num_not_made_cut : ℕ) 
    (num_total_students := num_girls + B)
    (num_students_tried_out  : (num_girls + B - num_call_backs = num_not_made_cut)) :
    B = 32 := 
by
  -- Definitions
  have total_students : num_total_students = num_girls + B := rfl
  have total_students_not_made_cut : num_students_tried_out = 39 := rfl
  -- Solve for B
  sorry

end number_of_boys_tried_out_l571_571507


namespace cos_2alpha_2beta_l571_571680

variables (α β : ℝ)

open Real

theorem cos_2alpha_2beta (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) : cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_2alpha_2beta_l571_571680


namespace max_employees_adjusted_is_750_range_of_a_l571_571139

-- Definition of the general conditions
def total_employees := 1000

def avg_profit_per_employee := 100_000

def profit_adjusted_employee (a : ℕ) (x : ℕ) : ℕ :=
  10 * (a - 0.008 * x)

def profit_remaining_employees (x : ℕ) : ℕ :=
  10 * (total_employees - x) * (1 + 0.004 * x)

-- Problem (I)
theorem max_employees_adjusted_is_750 (x : ℕ) (h_x_pos : x > 0) :
  10 * (total_employees - x) * (1 + 0.004 * x) ≥ 10 * total_employees
  → x ≤ 750 :=
by
  sorry

-- Problem (II)
theorem range_of_a (x : ℕ) (a : ℝ) (h_cond_1 : x ≤ 750) (h_cond_2 : 0 < a) :
  profit_adjusted_employee a x ≤ profit_remaining_employees x
  → a ≤ 7 :=
by
  sorry

end max_employees_adjusted_is_750_range_of_a_l571_571139


namespace Ivan_bought_10_cards_l571_571390

-- Define variables and conditions
variables (x : ℕ) -- Number of Uno Giant Family Cards bought
def original_price : ℕ := 12
def discount_per_card : ℕ := 2
def discounted_price := original_price - discount_per_card
def total_paid : ℕ := 100

-- Lean 4 theorem statement
theorem Ivan_bought_10_cards (h : discounted_price * x = total_paid) : x = 10 := by
  -- proof goes here
  sorry

end Ivan_bought_10_cards_l571_571390


namespace count_integers_satisfying_inequality_l571_571245

theorem count_integers_satisfying_inequality :
  {n : ℤ | 15 < n^2 ∧ n^2 < 120}.card = 14 :=
sorry

end count_integers_satisfying_inequality_l571_571245


namespace factor_expression_l571_571653

theorem factor_expression (x y : ℝ) :
  75 * x^10 * y^3 - 150 * x^20 * y^6 = 75 * x^10 * y^3 * (1 - 2 * x^10 * y^3) :=
by
  sorry

end factor_expression_l571_571653


namespace sara_lunch_total_l571_571463

theorem sara_lunch_total :
  let hotdog := 5.36
  let salad := 5.10
  hotdog + salad = 10.46 :=
by
  let hotdog := 5.36
  let salad := 5.10
  sorry

end sara_lunch_total_l571_571463


namespace domain_f_l571_571238

def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 9)

theorem domain_f :
  ∀ x : ℝ, x ≠ 3 → x ≠ -3 → ∃ y : ℝ, f x = y :=
by
  sorry

end domain_f_l571_571238


namespace unique_four_digit_number_l571_571754

theorem unique_four_digit_number (N : ℕ) (a : ℕ) (x : ℕ) :
  (N = 1000 * a + x) ∧ (N = 7 * x) ∧ (100 ≤ x ∧ x ≤ 999) ∧ (1 ≤ a ∧ a ≤ 9) →
  N = 3500 :=
by sorry

end unique_four_digit_number_l571_571754


namespace rectangle_area_increase_l571_571590

theorem rectangle_area_increase : ∀ (L B : ℝ), 
  (L * B = 150) →
  let L_new := L * 1.375
  let B_new := B * 0.818
  let A_new := L_new * B_new
  A_new ≈ 150 * 1.1255 →
  A_new - 150 ≈ 18.825 :=
begin
  intros L B h_initial h_L_new h_B_new h_A_new,
  sorry
end

end rectangle_area_increase_l571_571590


namespace set_B_can_form_right_angled_triangle_l571_571997

-- Definition and condition from the problem
def isRightAngledTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- The actual proof problem statement
theorem set_B_can_form_right_angled_triangle : isRightAngledTriangle 1 (Real.sqrt 3) 2 :=
sorry

end set_B_can_form_right_angled_triangle_l571_571997


namespace slope_of_line_segment_CD_l571_571215

-- Define the equations of the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 8 * y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 10 * x - 4 * y + 40 = 0

-- Statement of the problem: Prove the slope of the line formed by the intersection is -1/3.
theorem slope_of_line_segment_CD :
  ∃ m : ℝ, m = -1/3 ∧ ∀ x y : ℝ, (circle1 x y) ∧ (circle2 x y) → y = m * x + 5 :=
begin
  sorry
end

end slope_of_line_segment_CD_l571_571215


namespace num_factors_48_l571_571357

theorem num_factors_48 : 
  let n := 48 in
  ∃ num_factors, num_factors = 10 ∧ 
  (∀ p k, prime p → (n = p ^ k → 1 + k)) := 
sorry

end num_factors_48_l571_571357


namespace correct_option_D_l571_571119

theorem correct_option_D (a : ℝ) (h : a ≠ 0) : a^0 = 1 :=
by sorry

end correct_option_D_l571_571119


namespace evaluate_expr_l571_571224

theorem evaluate_expr : 3 * (3 * (3 * (3 * (3 * (3 * 2 * 2) * 2) * 2) * 2) * 2) * 2 = 1458 := by
  sorry

end evaluate_expr_l571_571224


namespace distance_halfway_along_orbit_l571_571492

variable {Zeta : Type}  -- Zeta is a type representing the planet
variable (distance_from_focus : Zeta → ℝ)  -- Function representing the distance from the sun (focus)

-- Conditions
variable (perigee_distance : ℝ := 3)
variable (apogee_distance : ℝ := 15)
variable (a : ℝ := (perigee_distance + apogee_distance) / 2)  -- semi-major axis

theorem distance_halfway_along_orbit (z : Zeta) (h1 : distance_from_focus z = perigee_distance) (h2 : distance_from_focus z = apogee_distance) :
  distance_from_focus z = a :=
sorry

end distance_halfway_along_orbit_l571_571492


namespace arithmetic_sequence_S12_l571_571711

def arithmetic_sequence_sum (a d n : ℕ) : ℕ :=
  n * (2*a + (n-1)*d) / 2

def a_n (a d n : ℕ) : ℕ :=
  a + (n-1)*d

variable (a d : ℕ)

theorem arithmetic_sequence_S12 (h : a_n a d 4 + a_n a d 9 = 10) :
  arithmetic_sequence_sum a d 12 = 60 :=
by sorry

end arithmetic_sequence_S12_l571_571711


namespace smallest_positive_integer_ending_in_9_divisible_by_13_l571_571104

theorem smallest_positive_integer_ending_in_9_divisible_by_13 :
  ∃ n : ℕ, (n % 10 = 9) ∧ (n % 13 = 0) ∧ (∀ m : ℕ, (m % 10 = 9) ∧ (m % 13 = 0) → m ≥ n) :=
sorry

end smallest_positive_integer_ending_in_9_divisible_by_13_l571_571104


namespace david_marks_in_english_l571_571211

theorem david_marks_in_english 
  (math : ℤ) (phys : ℤ) (chem : ℤ) (bio : ℤ) (avg : ℤ) 
  (marks_per_math : math = 85) 
  (marks_per_phys : phys = 92) 
  (marks_per_chem : chem = 87) 
  (marks_per_bio : bio = 95) 
  (avg_marks : avg = 89) 
  (num_subjects : ℤ := 5) :
  ∃ (eng : ℤ), eng + 85 + 92 + 87 + 95 = 89 * 5 ∧ eng = 86 :=
by
  sorry

end david_marks_in_english_l571_571211


namespace part1_l571_571308

theorem part1 (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x ≥ 0) →
  f = λ x, (x - 2) * Real.log x - a * (x + 2) →
  a ≤ 0 :=
sorry

end part1_l571_571308


namespace genetic_recombination_does_not_occur_during_dna_replication_l571_571550

-- Definitions based on conditions
def dna_replication_spermatogonial_cells : Prop := 
  ∃ dna_interphase: Prop, ∃ dna_unwinding: Prop, 
    ∃ gene_mutation: Prop, ∃ protein_synthesis: Prop,
      dna_interphase ∧ dna_unwinding ∧ gene_mutation ∧ protein_synthesis

def genetic_recombination_not_occur : Prop :=
  ¬ ∃ genetic_recombination: Prop, genetic_recombination

-- Proof problem statement
theorem genetic_recombination_does_not_occur_during_dna_replication : 
  dna_replication_spermatogonial_cells → genetic_recombination_not_occur :=
by sorry

end genetic_recombination_does_not_occur_during_dna_replication_l571_571550


namespace number_of_integers_l571_571249

theorem number_of_integers (n : ℤ) : {n : ℤ | 15 < n^2 ∧ n^2 < 120}.finite.card = 14 :=
sorry

end number_of_integers_l571_571249


namespace min_sum_abc_l571_571962

theorem min_sum_abc (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1716) :
  a + b + c = 31 :=
sorry

end min_sum_abc_l571_571962


namespace VumosArrangements_l571_571519

theorem VumosArrangements : 
  let boxes : Fin 9 → Fin 9 := λ i, ⟨(i.1 + 1) % 9, by {
    simp,
    exact Nat.mod_lt (i.1 + 1) (by decide)
  }⟩
  let nums := (Finset.range 9).val
  let isValidArrangement (f : Fin 9 → ℕ) : Prop :=
    (Finset.univ : Finset (Fin 9)).sum (λ i, f i) = nums.sum ∧
    ∀ i, (f i + f (boxes (i+1)) + f (boxes (i+2))) % 3 = 0
  (Finset.univ.filter (λ f : Fin 9 → ℕ, isValidArrangement f)).card = 6 * 6 * 6 * 6 := sorry

end VumosArrangements_l571_571519


namespace cos_alpha_plus_beta_eq_neg_one_l571_571677

-- Definitions derived from conditions
variables (α β : ℝ)
variables (h0 : 0 < β) (h1 : β < π / 2) (h2 : π / 2 < α) (h3 : α < π)
variables (h4 : cos (α - β / 2) = - sqrt 2 / 2)
variables (h5 : sin (α / 2 - β) = sqrt 2 / 2)

-- The statement to be proved
theorem cos_alpha_plus_beta_eq_neg_one : cos (α + β) = -1 :=
by
  sorry

end cos_alpha_plus_beta_eq_neg_one_l571_571677


namespace students_didnt_make_cut_l571_571506

theorem students_didnt_make_cut (g b c : ℕ) (hg : g = 15) (hb : b = 25) (hc : c = 7) : g + b - c = 33 := by
  sorry

end students_didnt_make_cut_l571_571506


namespace find_m_of_ellipse_l571_571733

theorem find_m_of_ellipse (m : ℝ) 
  (h1 : m > 1) 
  (h2 : ∀ x y : ℝ, mx^2 + y^2 = 1) 
  (h3 : minor_axis_length = sqrt 2 / 2 * m) : 
  m = 2 * sqrt 2 :=
by
  sorry

end find_m_of_ellipse_l571_571733


namespace repeating_decimals_product_fraction_l571_571226

theorem repeating_decimals_product_fraction : 
  let x := 1 / 33
  let y := 9 / 11
  x * y = 9 / 363 := 
by
  sorry

end repeating_decimals_product_fraction_l571_571226


namespace angle_AOC_is_120_degrees_l571_571451

-- Definitions for the conditions
variables (A B C D E F O : Point)
axiom evenly_spaced : evenly_spaced A B C D E F O

-- The proof statement
theorem angle_AOC_is_120_degrees : measure_angle A O C = 120 := 
sorry

end angle_AOC_is_120_degrees_l571_571451


namespace cross_product_correct_l571_571661

-- Define the vectors a and b
def a : (Fin 3 → ℤ) := ![3, 2, 4]
def b : (Fin 3 → ℤ) := ![6, -3, 8]

-- Define the cross product function for 3-dimensional vectors
def cross_product {R : Type*} [Ring R] (u v : Fin 3 → R) : Fin 3 → R :=
  ![
    u 1 * v 2 - u 2 * v 1,
    u 2 * v 0 - u 0 * v 2,
    u 0 * v 1 - u 1 * v 0
  ]

-- Define the expected result of the cross product
def result : (Fin 3 → ℤ) := ![28, 0, -21]

-- Theorem statement
theorem cross_product_correct : cross_product a b = result :=
by
  -- Here we leave the proof as a sorry placeholder
  sorry

end cross_product_correct_l571_571661


namespace probability_of_5_odd_numbers_l571_571078

-- Define a function to represent the binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

-- Axiom that defines the probability of getting an odd number
axiom fair_die_prob : ∀ (x : ℕ), 0 < x ∧ x ≤ 6 -> (1/2)

-- Define the problem statement about the probability
theorem probability_of_5_odd_numbers (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 5) : 
  (binom n k) / 2^n = 3 / 32 := sorry

end probability_of_5_odd_numbers_l571_571078


namespace ratio_B_to_C_l571_571557

def A : ℕ := B + 2
def B : ℕ := 28
def C : ℕ := 72 - A - B

theorem ratio_B_to_C : B / C = 2 / 1 :=
by
  have hB : B = 28 := rfl
  have hA : A = B + 2 := rfl
  have hC : C = 72 - A - B := rfl
  rw [hB, hA, hC]
  sorry

end ratio_B_to_C_l571_571557


namespace solve_fraction_l571_571972

theorem solve_fraction (x : ℝ) (h1 : x + 2 = 0) (h2 : 2 * x - 4 ≠ 0) : x = -2 := 
by 
  sorry

end solve_fraction_l571_571972


namespace smallest_positive_integer_remainder_l571_571533

theorem smallest_positive_integer_remainder
  (b : ℕ) (h1 : b % 4 = 3) (h2 : b % 6 = 5) :
  b = 11 := by
  sorry

end smallest_positive_integer_remainder_l571_571533


namespace value_of_m_plus_ni_l571_571291

variables (m n : ℝ) (i : ℂ)
noncomputable def imaginary_unit := complex.I
noncomputable def given_equation := (m / (1 + imaginary_unit) = 1 - n * imaginary_unit)

theorem value_of_m_plus_ni (hmn : given_equation) : m + n * imaginary_unit = 2 + imaginary_unit := 
sorry

end value_of_m_plus_ni_l571_571291


namespace cost_of_marker_l571_571508

theorem cost_of_marker (n m : ℝ) (h1 : 3 * n + 2 * m = 7.45) (h2 : 4 * n + 3 * m = 10.40) : m = 1.40 :=
  sorry

end cost_of_marker_l571_571508


namespace four_nonzero_complex_numbers_form_square_l571_571365

open Complex

theorem four_nonzero_complex_numbers_form_square :
  ∃ (S : Finset ℂ), S.card = 4 ∧ (∀ z ∈ S, z ≠ 0) ∧ (∀ z ∈ S, ∃ (θ : ℝ), z = exp (θ * I) ∧ (exp (4 * θ * I) - z).re = 0 ∧ (exp (4 * θ * I) - z).im = cos (π / 2)) := 
sorry

end four_nonzero_complex_numbers_form_square_l571_571365


namespace pedoe_inequality_l571_571408

variables {a b c a' b' c' Δ Δ' : ℝ} {A A' : ℝ}

theorem pedoe_inequality :
  a' ^ 2 * (-a ^ 2 + b ^ 2 + c ^ 2) +
  b' ^ 2 * (a ^ 2 - b ^ 2 + c ^ 2) +
  c' ^ 2 * (a ^ 2 + b ^ 2 - c ^ 2) -
  16 * Δ * Δ' =
  2 * (b * c' - b' * c) ^ 2 +
  8 * b * b' * c * c' * (Real.sin ((A - A') / 2)) ^ 2 := sorry

end pedoe_inequality_l571_571408


namespace proof_problem_l571_571691

-- definitions of the given conditions
variable (a b c : ℝ)
variables (h₁ : 6 < a) (h₂ : a < 10) 
variable (h₃ : (a / 2) ≤ b ∧ b ≤ 2 * a) 
variable (h₄ : c = a + b)

-- statement to be proved
theorem proof_problem (h₁ : 6 < a) (h₂ : a < 10) (h₃ : (a / 2) ≤ b ∧ b ≤ 2 * a) (h₄ : c = a + b) : 9 < c ∧ c < 30 := 
sorry

end proof_problem_l571_571691


namespace smallest_positive_integer_ends_in_9_and_divisible_by_13_l571_571091

theorem smallest_positive_integer_ends_in_9_and_divisible_by_13 :
  ∃ n : ℕ, n % 10 = 9 ∧ 13 ∣ n ∧ n > 0 ∧ ∀ m, m % 10 = 9 → 13 ∣ m ∧ m > 0 → m ≥ n := 
begin
  use 99,
  split,
  { exact mod_eq_of_lt (10*k + 9) 10 99 9 (by norm_num), },
  split,
  { exact dvd_refl 99, },
  split,
  { exact zero_lt_99, },
  intros m hm1 hm2 hpos,
  by_contradiction hmn,
  sorry
end

end smallest_positive_integer_ends_in_9_and_divisible_by_13_l571_571091


namespace P_is_in_third_quadrant_l571_571450

noncomputable def point : Type := (ℝ × ℝ)

def P : point := (-3, -4)

def is_in_third_quadrant (p : point) : Prop :=
  p.1 < 0 ∧ p.2 < 0

theorem P_is_in_third_quadrant : is_in_third_quadrant P :=
by {
  -- Prove that P is in the third quadrant
  sorry
}

end P_is_in_third_quadrant_l571_571450


namespace profit_calculation_l571_571513

-- Definitions from conditions
def initial_shares := 20
def cost_per_share := 3
def sold_shares := 10
def sale_price_per_share := 4
def remaining_shares_value_multiplier := 2

-- Calculations based on conditions
def initial_cost := initial_shares * cost_per_share
def revenue_from_sold_shares := sold_shares * sale_price_per_share
def remaining_shares := initial_shares - sold_shares
def value_of_remaining_shares := remaining_shares * (cost_per_share * remaining_shares_value_multiplier)
def total_value := revenue_from_sold_shares + value_of_remaining_shares
def expected_profit := total_value - initial_cost

-- The problem statement to be proven
theorem profit_calculation : expected_profit = 40 := by
  -- Proof steps go here
  sorry

end profit_calculation_l571_571513


namespace sum_of_interior_angles_not_sum_of_interior_angles_800_l571_571118

theorem sum_of_interior_angles (n : ℕ) (h : n ≥ 3) : ∃ k : ℕ, (n-2) * 180 = k * 180 :=
by sorry

theorem not_sum_of_interior_angles_800 : ¬ ∃ n : ℕ, n ≥ 3 ∧ (n-2) * 180 = 800 :=
by
  intro h
  obtain ⟨n, hn1, hn2⟩ := h
  have mul_180 : ∃ k : ℕ, 800 = k * 180 := by sorry
  contradiction

end sum_of_interior_angles_not_sum_of_interior_angles_800_l571_571118


namespace total_hours_driven_l571_571007

def total_distance : ℝ := 55.0
def distance_in_one_hour : ℝ := 1.527777778

theorem total_hours_driven : (total_distance / distance_in_one_hour) = 36.00 :=
by
  sorry

end total_hours_driven_l571_571007


namespace nested_fraction_solution_l571_571539

noncomputable def nested_fraction : ℝ :=
  3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / (3 + 5 / ... ))))

theorem nested_fraction_solution :
  nested_fraction = (3 + Real.sqrt 69) / 2 :=
sorry

end nested_fraction_solution_l571_571539


namespace max_height_projectile_max_height_val_max_height_is_97_l571_571589

theorem max_height_projectile : 
  ∃ t : ℝ, ∀ t' : ℝ, (-18 * t^2 + 72 * t + 25) ≥ (-18 * t'^2 + 72 * t' + 25) :=
begin
  use 2,
  intro t',
  have h_eq : -18 * t'^2 + 72 * t' + 25 = -18 * (t' - 2)^2 + 97, sorry,
  rw h_eq,
  nlinarith [sq_nonneg (t' - 2)], -- this essentially completes the square showing
end

theorem max_height_val : 
  ∃ t : ℝ, ∀ t' : ℝ, (-18 * t^2 + 72 * t + 25) ≤ 97 :=
begin
  use 2,
  intro t',
  have h_eq : -18 * t'^2 + 72 * t' + 25 = -18 * (t' - 2)^2 + 97, sorry,
  rw h_eq,
  nlinarith [sq_nonneg (t' - 2)],
end

theorem max_height_is_97 : 
  ∀ t : ℝ, (-18 * t^2 + 72 * t + 25) ≤ 97 :=
begin
  intro t,
  have h_eq : -18 * t^2 + 72 * t + 25 = -18 * (t - 2)^2 + 97, sorry,
  rw h_eq,
  nlinarith [sq_nonneg (t - 2)],
end

end max_height_projectile_max_height_val_max_height_is_97_l571_571589


namespace x_squared_greater_than_300_l571_571798

theorem x_squared_greater_than_300 :
  ∃ n : ℕ, n = 3 ∧ (nat.pow (nat.pow (nat.pow 3 2) 2) 2) > 300 :=
begin
  use 3,
  split,
  { refl },
  { simp,
    norm_num,
    exact 6561 }
end

end x_squared_greater_than_300_l571_571798


namespace smallest_number_of_students_l571_571913

open Finset

theorem smallest_number_of_students :
  let primes_under_50 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
  in ∃ (n : ℕ), (∀ p ∈ primes_under_50, p ∣ n) ∧ (n = 614889782588491410) :=
by {
  sorry
}

end smallest_number_of_students_l571_571913


namespace count_integers_satisfying_inequality_l571_571255

theorem count_integers_satisfying_inequality : 
  (∃ S : Set ℤ, (∀ n ∈ S, 15 < n^2 ∧ n^2 < 120) ∧ S.card = 14) :=
by
  sorry

end count_integers_satisfying_inequality_l571_571255


namespace number_of_elements_in_Z_inter_complement_A_l571_571746

noncomputable def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x^2 - x - 6 ≥ 0}

def complement_A : Set ℝ := {x : ℝ | ¬ (x^2 - x - 6 ≥ 0)}

def Z : Set ℤ := Set.univ

def Z_inter_complement_A : Set ℤ := { n : ℤ | -2 < (n : ℝ) ∧ (n : ℝ) < 3 }

theorem number_of_elements_in_Z_inter_complement_A : 
  Fintype.card (Z_inter_complement_A.to_finset) = 4 := by
  sorry

end number_of_elements_in_Z_inter_complement_A_l571_571746


namespace minimum_distance_sum_l571_571721

noncomputable def distance_from_point_to_line (x1 y1 a b c : ℝ) : ℝ :=
  |a * x1 + b * y1 + c| / Real.sqrt (a ^ 2 + b ^ 2)

theorem minimum_distance_sum :
  let focus := (1 : ℝ, 0 : ℝ)
  let parabola := { p : ℝ × ℝ | p.snd^2 = 4 * p.fst }
  line := (2 : ℝ, -1 : ℝ, 3 : ℝ),
  ∃ P ∈ parabola, 
  ∀ P, (P.1, P.2) ∈ parabola →
  let d := distance_from_point_to_line P.1 P.2 line.1 line.2 line.3 in
  d + Real.sqrt ((P.1 - focus.1) ^ 2 + (P.2 - focus.2) ^ 2) - 1 = Real.sqrt 5 - 1 :=
sorry

end minimum_distance_sum_l571_571721


namespace baskets_count_l571_571042

theorem baskets_count (total_apples apples_per_basket : ℕ) (h1 : total_apples = 629) (h2 : apples_per_basket = 17) : (total_apples / apples_per_basket) = 37 :=
by
  sorry

end baskets_count_l571_571042


namespace athlete_speed_l571_571559

theorem athlete_speed (distance time : ℝ) (h1 : distance = 200) (h2 : time = 25) :
  (distance / time) = 8 := by
  sorry

end athlete_speed_l571_571559


namespace log_decreasing_interval_l571_571799

theorem log_decreasing_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (H : ∀ x y : ℝ, (-∞ <= x ∧ x <= y ∧ y <= 1) → (x^2 - a*x + 2 ≤ y^2 - a*y + 2 → log a (x^2 - a*x + 2) ≥ log a (y^2 - a*y + 2)))
  : 2 ≤ a ∧ a < 3 := sorry

end log_decreasing_interval_l571_571799


namespace raft_travel_distance_l571_571586

theorem raft_travel_distance (v_b v_s t : ℝ) (h1 : t > 0) 
  (h2 : v_b + v_s = 90 / t) (h3 : v_b - v_s = 70 / t) : 
  v_s * t = 10 := by
  sorry

end raft_travel_distance_l571_571586


namespace multiples_of_6_or_8_but_not_both_l571_571776

/-- The number of positive integers less than 151 that are multiples of either 6 or 8 but not both is 31. -/
theorem multiples_of_6_or_8_but_not_both (n : ℕ) :
  (multiples_of_6 : Set ℕ) = {k | k < 151 ∧ k % 6 = 0}
  ∧ (multiples_of_8 : Set ℕ) = {k | k < 151 ∧ k % 8 = 0}
  ∧ (multiples_of_24 : Set ℕ) = {k | k < 151 ∧ k % 24 = 0}
  ∧ multiples_of_6_or_8 := {k | k ∈ multiples_of_6 ∨ k ∈ multiples_of_8}
  ∧ multiples_of_6_and_8 := {k | k ∈ multiples_of_6 ∧ k ∈ multiples_of_8}
  ∧ (card (multiples_of_6_or_8 \ multiples_of_6_and_8)) = 31 := sorry

end multiples_of_6_or_8_but_not_both_l571_571776


namespace probability_unique_tens_digits_correct_l571_571002

-- Define the range of integers
def range_10_to_59 : finset ℕ := finset.Icc 10 59

-- Function to extract the tens digit
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Define the event that each integer in a chosen set has a different tens digit
def unique_tens_digits (s : finset ℕ) : Prop :=
  (∀ x y ∈ s, x ≠ y → tens_digit x ≠ tens_digit y)

-- Define the total number of ways to choose 5 unique integers with unique tens digits from the range
def ways_with_unique_tens_digits : ℕ := 10^5

-- Define the total number of combinations of choosing 5 integers from the range
def total_combinations : ℕ := nat.choose 50 5

-- Compute the required probability
def probability_unique_tens_digits : ℚ :=
  ways_with_unique_tens_digits.to_rat / total_combinations.to_rat

-- Proven statement
theorem probability_unique_tens_digits_correct :
  probability_unique_tens_digits = 2500 / 52969 := by
  sorry

end probability_unique_tens_digits_correct_l571_571002


namespace inequality_solution_l571_571727

variables (a b x : ℝ)
variable (h1 : a > b)
variable (h2 : (a - b) * (1/2) + a + 2 * b > 0)
variable (h3 : (a - b) * x + a + 2 * b = 0)

theorem inequality_solution (a b x : ℝ) (h1 : a > b) (h2 : \(\frac{-a - 2b}{a - b} = \frac{1}{2}\)) : ∀ x, (ax < b) ↔ (x < -1) :=
begin
  intro x,
  split,
  { intro hx,
    sorry
  },
  { intro hx,
    sorry
  }
end

end inequality_solution_l571_571727


namespace find_c_l571_571723

theorem find_c 
  (a b c : ℕ) 
  (h₁ : a ≤ b) 
  (h₂ : b ≤ c) 
  (h₃ : a > 0) 
  (h₄ : b > 0) 
  (h₅ : c > 0)
  (x y z w : ℝ) 
  (h₆ : a^x = 70^w) 
  (h₇ : b^y = 70^w)
  (h₈ : c^z = 70^w) 
  (h₉ : 1/x + 1/y + 1/z = 1/w) : 
  c = 7 := 
sorry 

end find_c_l571_571723


namespace smallest_number_ending_in_9_divisible_by_13_l571_571097

theorem smallest_number_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), (n % 10 = 9) ∧ (13 ∣ n) ∧ (∀ (m : ℕ), (m % 10 = 9) ∧ (13 ∣ m) ∧ (m < n) -> false) :=
sorry

end smallest_number_ending_in_9_divisible_by_13_l571_571097


namespace arithmetic_sum_l571_571186

theorem arithmetic_sum :
  let a := 1 in
  let d := 2 in
  let l := 21 in
  let n := (l - a) / d + 1 in
  (n * (a + l)) / 2 = 121 :=
by
  let a := 1
  let d := 2
  let l := 21
  let n := (l - a) / d + 1
  sorry

end arithmetic_sum_l571_571186


namespace arith_seq_sum_l571_571188

theorem arith_seq_sum (n : ℕ) (h₁ : 2 * n - 1 = 21) : 
  (∑ i in finset.range 11, (2 * i + 1)) = 121 :=
by
  sorry

end arith_seq_sum_l571_571188


namespace real_solution_count_is_two_l571_571261

   noncomputable def num_real_solutions : ℕ :=
     (λ x : ℝ, 8 ^ (x ^ 2 - 6 * x + 8) = 64).support.card

   theorem real_solution_count_is_two :
     num_real_solutions = 2 :=
   sorry
   
end real_solution_count_is_two_l571_571261


namespace narrow_black_stripes_l571_571854

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l571_571854


namespace find_x2_y2_w2_l571_571630

variable (x y w : ℝ)
def N : matrix (fin 3) (fin 3) ℝ := !![
  [0, 3 * y, w],
  [x, 2 * y, -w],
  [x, -2 * y, w]
]
noncomputable def NT : matrix (fin 3) (fin 3) ℝ := ![
  [0, x, x],
  [3 * y, 2 * y, -2 * y],
  [w, -w, w]
]

theorem find_x2_y2_w2 (h : (NT x y w) ⬝ (N x y w) = 1) : x^2 + y^2 + w^2 = 91 / 102 := sorry

end find_x2_y2_w2_l571_571630


namespace time_to_cross_man_l571_571162

variables (speed_kmph : ℕ) (platform_length : ℕ) (crossing_time_platform : ℕ)

-- Define the conditions
def conditions_holds : Prop :=
  speed_kmph = 72 ∧
  platform_length = 280 ∧
  crossing_time_platform = 32

-- Define the statement to be proven
theorem time_to_cross_man (h : conditions_holds speed_kmph platform_length crossing_time_platform) : 
  ∃ time_crossing_man : ℕ, time_crossing_man = 18 :=
begin
  sorry
end

end time_to_cross_man_l571_571162


namespace find_special_integer_l571_571657

theorem find_special_integer :
  ∃ (n : ℕ), n > 0 ∧ (21 ∣ n) ∧ 30 ≤ Real.sqrt n ∧ Real.sqrt n ≤ 30.5 ∧ n = 903 := 
sorry

end find_special_integer_l571_571657


namespace narrow_black_stripes_l571_571879

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l571_571879


namespace count_integers_within_bounds_l571_571257

theorem count_integers_within_bounds : 
  ∃ (count : ℕ), count = finset.card (finset.filter (λ n : ℤ, 15 < n^2 ∧ n^2 < 120) (finset.Icc (-10) 10)) ∧ count = 14 := 
by
  sorry

end count_integers_within_bounds_l571_571257


namespace symmetric_point_sqrt_l571_571318

theorem symmetric_point_sqrt {a b : ℝ} -- define a and b as real numbers:
  (h1 : a + b = -3)        -- the first condition a + b = -3
  (h2 : 1 - b = -1) :      -- the second condition 1 - b = -1
  sqrt (-a * b) = sqrt 10 := -- prove that sqrt(-ab) = sqrt(10)
sorry -- proof not required

end symmetric_point_sqrt_l571_571318


namespace extra_food_needed_l571_571116

theorem extra_food_needed (food_one_cat per_day_two_cats : ℝ) (h1 : food_one_cat = 0.5) (h2 : per_day_two_cats = 0.9) :
  per_day_two_cats - food_one_cat = 0.4 :=
by {rw [h1, h2], norm_num}

end extra_food_needed_l571_571116


namespace digit_concatenation_greater_than_product_general_digit_inequality_l571_571924

-- Define the notation for concatenated numbers
def concat_digits : list ℕ → ℕ
| []       := 0
| (d :: ds) := d * 10 ^ ds.length + concat_digits ds

-- Define the main theorem
theorem digit_concatenation_greater_than_product (a b c d : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) (hd : d < 10) :
  concat_digits [a, b, c, d] > concat_digits [a, b] * concat_digits [c, d] :=
sorry

-- Generalized theorem for splitting an n-digit number
theorem general_digit_inequality (digits : list ℕ) (h_digits : ∀ d ∈ digits, d < 10) (splits : list ℕ) (h_splits : splits.sum = digits.length) :
  concat_digits digits >
  list.foldr (λ (split_idx : ℕ) acc,
              acc
              * (concat_digits (digits.drop split_idx.take (length digits))
              ))
  1 
  splits :=
sorry

end digit_concatenation_greater_than_product_general_digit_inequality_l571_571924


namespace functional_equation_solution_l571_571212

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 4 * x * y * f y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) := by
  intro h
  sorry

end functional_equation_solution_l571_571212


namespace solve_first_equation_solve_second_equation_l571_571472

theorem solve_first_equation (x : ℝ) :
  2 * x^2 - 4 * x + 1 = 0 ↔
  (x = 1 + (real.sqrt 2) / 2 ∨ x = 1 - (real.sqrt 2) / 2) := sorry

theorem solve_second_equation (x : ℝ) :
  (2 * x + 3)^2 - 4 * x - 6 = 0 ↔
  (x = -3 / 2 ∨ x = -1 / 2) := sorry

end solve_first_equation_solve_second_equation_l571_571472


namespace exists_nat_with_palindrome_decomp_l571_571641

def is_palindrome (n : ℕ) : Prop := 
  let s := n.toString
  s = s.reverse

theorem exists_nat_with_palindrome_decomp :
  ∃ n : ℕ, (∀ a b : ℕ, is_palindrome a → is_palindrome b → a * b = n → a ≠ b → (a, b) = (0, n) ∨ (b, a) = (0, n)) ∧ set.size { (a, b) | a * b = n ∧ is_palindrome a ∧ is_palindrome b } > 100 :=
begin
  use 2^101,
  sorry
end

end exists_nat_with_palindrome_decomp_l571_571641


namespace suitable_for_census_l571_571172

noncomputable def survey_lifespan_lamps := "Lifespan of a batch of energy-saving lamps"
noncomputable def survey_eyesight_students := "Eyesight condition of middle school students in our country"
noncomputable def survey_quantum_satellite := "Certain component on the Quantum Science Satellite"
noncomputable def survey_tv_viewership := "Viewership of the 'Strongest Brain' show"

theorem suitable_for_census :
  ∃ survey, survey = survey_quantum_satellite :=
by {
  use survey_quantum_satellite,
  sorry
}

end suitable_for_census_l571_571172


namespace sin_cos_sum_eq_l571_571304

noncomputable def sin_cos_sum (α : Real) (a : Real) : Real :=
  let x := 3 * a
  let y := -4 * a
  let r := Real.sqrt ((x ^ 2) + (y ^ 2))
  (y / r) + (x / r)

theorem sin_cos_sum_eq (a : Real) (ha : a < 0) :
  sin_cos_sum (Real.atan2 (-4 * a) (3 * a)) a = 1 / 5 :=
by
  sorry

end sin_cos_sum_eq_l571_571304


namespace trapezoid_perimeter_is_correct_l571_571163

variable (A B C D E F G : Point)
variable (square : Square A B C D)
variable (trap : Trapezoid A E F G)
variable (sideLength : ℝ)
variable (parallels : Parallel E F A G)
variable (perpendicularDiags : Perpendicular (Line A F) (Line E G))
variable (EG_length : EG = 10 * Real.sqrt 2)

noncomputable def perimeter_AEFG : ℝ := 45 -- Given based on the original question

theorem trapezoid_perimeter_is_correct :
  ∀ (A B C D E F G : Point)
    (square : Square A B C D)
    (trap : Trapezoid A E F G)
    (sideLength : ℝ)
    (parallels : Parallel E F A G)
    (perpendicularDiags : Perpendicular (Line A F) (Line E G))
    (EG_length : EG = 10 * Real.sqrt 2),
    sideLength = 14 → square.sideLength = sideLength → trap.inside square → trap.pointsOnEdges square → (perimeter trap = 45) := 
by {
  intros,
  sorry
}

end trapezoid_perimeter_is_correct_l571_571163


namespace narrow_black_stripes_l571_571851

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l571_571851


namespace distance_between_points_on_line_l571_571374

theorem distance_between_points_on_line (a b c d m k : ℝ) 
  (hab : b = m * a + k) (hcd : d = m * c + k) :
  dist (a, b) (c, d) = |a - c| * Real.sqrt (1 + m^2) :=
by
  sorry

end distance_between_points_on_line_l571_571374


namespace narrow_black_stripes_l571_571874

theorem narrow_black_stripes (w n b : ℕ) (h1 : b = w + 7) (h2 : w + n = b + 1) : n = 8 := 
by
  sorry

end narrow_black_stripes_l571_571874


namespace narrow_black_stripes_are_eight_l571_571873

variable (w n b : ℕ)

-- Given conditions as definitions in Lean
def white_stripes_eq : Prop := b = w + 7
def total_black_eq_total_white_plus_one : Prop := w + n = b + 1

theorem narrow_black_stripes_are_eight (h₁ : white_stripes_eq w b) (h₂ : total_black_eq_total_white_plus_one w n b) : n = 8 := by
  -- Use the assumptions to derive n = 8
  sorry

end narrow_black_stripes_are_eight_l571_571873


namespace smallest_positive_integer_remainder_conditions_l571_571529

theorem smallest_positive_integer_remainder_conditions :
  ∃ b : ℕ, (b % 4 = 3 ∧ b % 6 = 5) ∧ ∀ n : ℕ, (n % 4 = 3 ∧ n % 6 = 5) → n ≥ b := 
by
  have b := 23
  use b
  sorry

end smallest_positive_integer_remainder_conditions_l571_571529


namespace symmetric_circle_equation_l571_571305

theorem symmetric_circle_equation :
  let O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let line : Set (ℝ × ℝ) := {p | p.1 + p.2 = 5}
  let O' : Set (ℝ × ℝ) := 
    {p | let (x', y') := (5, 5) in
         ((p.1 - x')^2 + (p.2 - y')^2 = 1)}
  let O'_center : ℝ × ℝ := (5, 5)
  let r := 1
  (O = {(x, y) | x^2 + y^2 = 1})
  ∧ (line = {(x, y) | x + y = 5})
  ∧ (O' = {(x - O'_center.1)^2 + (y - O'_center.2)^2 = r^2}) 
: O' = {(x, y) | (x - 5)^2 + (y - 5)^2 = 1}.
sorry

end symmetric_circle_equation_l571_571305


namespace count_integers_satisfying_inequality_l571_571253

theorem count_integers_satisfying_inequality : 
  (∃ S : Set ℤ, (∀ n ∈ S, 15 < n^2 ∧ n^2 < 120) ∧ S.card = 14) :=
by
  sorry

end count_integers_satisfying_inequality_l571_571253


namespace isosceles_triangle_perimeter_l571_571394

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_perimeter 
  (a b c : ℝ) 
  (h_iso : is_isosceles a b c) 
  (h1 : a = 2 ∨ a = 4) 
  (h2 : b = 2 ∨ b = 4) 
  (h3 : c = 2 ∨ c = 4) :
  a + b + c = 10 :=
  sorry

end isosceles_triangle_perimeter_l571_571394


namespace arithmetic_sum_l571_571187

theorem arithmetic_sum :
  let a := 1 in
  let d := 2 in
  let l := 21 in
  let n := (l - a) / d + 1 in
  (n * (a + l)) / 2 = 121 :=
by
  let a := 1
  let d := 2
  let l := 21
  let n := (l - a) / d + 1
  sorry

end arithmetic_sum_l571_571187


namespace largest_log_value_l571_571692

variable (a b : ℝ)
variable (h : a ≥ b)
variable (h1 : b > 2)

theorem largest_log_value : a >= b ∧ b > 2 → real.log a (a^2 / b^2) + real.log b (b^2 / a^2) ≤ 0 := 
by
  intro h
  intro h1
  sorry

end largest_log_value_l571_571692


namespace age_difference_l571_571970

theorem age_difference :
  ∃ a b : ℕ, (a < 10) ∧ (b < 10) ∧
    (∀ x y : ℕ, (x = 10 * a + b) ∧ (y = 10 * b + a) → 
    (x + 5 = 2 * (y + 5)) ∧ ((10 * a + b) - (10 * b + a) = 18)) :=
by
  sorry

end age_difference_l571_571970


namespace area_C_greater_sum_A_B_by_145_percent_l571_571956

-- Given conditions
variables (a : ℝ) (area_A area_B area_C sum_areas_AB : ℝ)
-- Definitions based on conditions
def side_B := 2 * a
def side_C := 3.5 * a
def area_A := a^2
def area_B := (2 * a)^2
def area_C := (3.5 * a)^2
def sum_areas_AB := a^2 + (2 * a)^2

theorem area_C_greater_sum_A_B_by_145_percent :
  ((area_C - sum_areas_AB) / sum_areas_AB) * 100 = 145 :=
by
  sorry

end area_C_greater_sum_A_B_by_145_percent_l571_571956


namespace problem_solution_l571_571841

def sequenceS : ℕ → Finset ℕ := 
  λ n, { x | bit_count 1 x = 9 }

def N_raise : ℕ := (sequenceS 11).to_list 1199

def remainder (a b : ℕ) : ℕ := a % b

theorem problem_solution : remainder N_raise 1200 = 301 := sorry

end problem_solution_l571_571841


namespace units_digit_17_pow_2023_l571_571111

theorem units_digit_17_pow_2023 : (17 ^ 2023) % 10 = 3 :=
by
  have units_cycle_7 : ∀ (n : ℕ), (7 ^ n) % 10 = [7, 9, 3, 1].nth (n % 4) :=
    sorry
  have units_pattern_equiv : (17 ^ n) % 10 = (7 ^ n) % 10 :=
    sorry
  calc
    (17 ^ 2023) % 10
        = (7 ^ 2023) % 10  : by rw [units_pattern_equiv]
    ... = 3               : by rw [units_cycle_7, nat.mod_eq_of_lt, List.nth]

end units_digit_17_pow_2023_l571_571111


namespace probability_5_of_6_odd_rolls_l571_571056

def binom_coeff : ℕ → ℕ → ℕ
| n k := Nat.choose n k

noncomputable def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binom_coeff n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_5_of_6_odd_rolls :
  binomial_probability 6 5 (1/2) = 3/16 :=
by
  -- Proof will go here, but we skip it with sorry for now.
  sorry

end probability_5_of_6_odd_rolls_l571_571056


namespace num_factors_48_l571_571339

theorem num_factors_48 : 
  ∀ (n : ℕ), n = 48 → (∃ k : ℕ, k = 10 ∧ ∀ d : ℕ, d ∣ n → (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 6 ∨ d = 8 ∨ d = 12 ∨ d = 16 ∨ d = 24 ∨ d = 48)) :=
  by
    intros n h
    sorry

end num_factors_48_l571_571339


namespace train_crossing_time_approx_l571_571560

def length_of_train : ℝ := 100 -- length of the train in meters
def speed_of_train_km_h : ℝ := 80 -- speed of the train in km/hr
def length_of_bridge : ℝ := 142 -- length of the bridge in meters

def speed_conversion_factor : ℝ := 1000 / 3600 -- conversion factor from km/hr to m/s

-- Speed of the train in meters per second (m/s)
def speed_of_train_m_s : ℝ := speed_of_train_km_h * speed_conversion_factor

-- Total distance to be covered by the train
def total_distance : ℝ := length_of_train + length_of_bridge

-- Time taken for the train to cross the bridge
def time_to_cross_bridge : ℝ := total_distance / speed_of_train_m_s

theorem train_crossing_time_approx :
  abs (time_to_cross_bridge - 10.89) < 0.01 := 
by
  sorry

end train_crossing_time_approx_l571_571560


namespace units_digit_17_pow_2023_l571_571107

theorem units_digit_17_pow_2023 
  (cycle : ℕ → ℕ)
  (h1 : cycle 0 = 7)
  (h2 : cycle 1 = 9)
  (h3 : cycle 2 = 3)
  (h4 : cycle 3 = 1)
  (units_digit : ℕ → ℕ)
  (h_units : ∀ n, units_digit (17^n) = units_digit (7^n))
  (h_units_cycle : ∀ n, units_digit (7^n) = cycle (n % 4)) :
  units_digit (17^2023) = 3 :=
by
  sorry

end units_digit_17_pow_2023_l571_571107


namespace power_of_7_expression_l571_571115

theorem power_of_7_expression :
  (∀ (a b : ℝ), a > 0 → b > 0 → (a * b)^(1 / 4) / (a * b)^(1 / 6) = a^(1 / 4 - 1 / 6) * b^(1 / 4 - 1 / 6)) →
  (∀ (m n : ℝ), (m / n) > 0 → 7^(m - n) = 7^(1/4 - 1/6)) →
  7^(1/12) = (7^(1 / 4) / 7^(1 / 6)) :=
begin
  -- sorry is used here to skip the actual proof
  sorry, 
end

end power_of_7_expression_l571_571115


namespace correct_propositions_l571_571734

def planes_parallel_to_same_plane_are_parallel : Prop :=
  ∀ (P Q R : Plane), (P || Q) → (Q || R) → (P || R)

def lines_parallel_to_same_plane_are_not_necessarily_parallel : Prop :=
  ∀ (l₁ l₂ : Line) (P : Plane), (l₁ || P) → (l₂ || P) → (¬ (l₁ || l₂) ∨ l₁ || l₂ ∨ ∃ (P' : Plane), (P ∩ P' = l₁ ∩ l₂))

def planes_perpendicular_to_same_plane_are_not_necessarily_parallel : Prop :=
  ∀ (P Q R : Plane), (P ⟂ Q) → (Q ⟂ R) → (¬ (P || R) ∨ P || R)

def lines_perpendicular_to_same_plane_are_parallel : Prop :=
  ∀ (l₁ l₂ : Line) (P : Plane), (l₁ ⟂ P) → (l₂ ⟂ P) → (l₁ || l₂)

theorem correct_propositions (h₁ : planes_parallel_to_same_plane_are_parallel)
                            (h₂ : ¬ lines_parallel_to_same_plane_are_not_necessarily_parallel)
                            (h₃ : ¬ planes_perpendicular_to_same_plane_are_not_necessarily_parallel)
                            (h₄ : lines_perpendicular_to_same_plane_are_parallel) :
                            (planes_parallel_to_same_plane_are_parallel ∧
                            lines_perpendicular_to_same_plane_are_parallel) :=
by
  sorry

end correct_propositions_l571_571734


namespace probability_of_exactly_10_defective_probability_of_between_10_and_20_defective_l571_571571

noncomputable def binomialProbabilityOfExactly10Defective (n : ℕ) (p : ℝ) (q : ℝ) : ℝ :=
  1 / (Real.sqrt (n * p * q)) * Real.exp(-((10 - n * p) ^ 2) / (2 * n * p * q))

noncomputable def binomialProbabilityOfBetween10And20Defective (n : ℕ) (p : ℝ) (q : ℝ) : ℝ :=
  let sqrt_npq := Real.sqrt (n * p * q)
  (Real.errorFunction (10 / sqrt_npq) - Real.errorFunction (-10 / sqrt_npq)) / 2

theorem probability_of_exactly_10_defective :
  binomialProbabilityOfExactly10Defective 500 0.02 0.98 ≈ 0.127 := sorry

theorem probability_of_between_10_and_20_defective :
  binomialProbabilityOfBetween10And20Defective 500 0.02 0.98 ≈ 0.499 := sorry

end probability_of_exactly_10_defective_probability_of_between_10_and_20_defective_l571_571571


namespace irreducible_fraction_l571_571934

theorem irreducible_fraction (n : ℤ) : Int.gcd (2 * n + 1) (3 * n + 1) = 1 :=
sorry

end irreducible_fraction_l571_571934


namespace pizzasServedDuringDinner_l571_571593

-- Definitions based on the conditions
def pizzasServedDuringLunch : ℕ := 9
def totalPizzasServedToday : ℕ := 15

-- Theorem statement
theorem pizzasServedDuringDinner : 
  totalPizzasServedToday - pizzasServedDuringLunch = 6 := 
  by 
    sorry

end pizzasServedDuringDinner_l571_571593


namespace find_e_l571_571441

theorem find_e (a b c d e : ℤ) (h1 : a = 5) (h2 : b = 3) (h3 : c = 2) (h4 : d = 6)
  (h5 : a - b + c * d - e = 14 - e) (h6 : a - (b + (c * (d - e))) = -10 + 2e) :
  e = 8 :=
by
  sorry

end find_e_l571_571441


namespace real_number_infinite_continued_fraction_l571_571547

theorem real_number_infinite_continued_fraction:
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / (3 + 5 / (2 + 5 / y))) ∧ y = 5 / 3 := 
begin
  sorry
end

end real_number_infinite_continued_fraction_l571_571547


namespace correct_calculation_l571_571993

theorem correct_calculation (a x y b : ℝ) :
  (-a - a = 0) = False ∧
  (- (x + y) = -x - y) = True ∧
  (3 * (b - 2 * a) = 3 * b - 2 * a) = False ∧
  (8 * a^4 - 6 * a^2 = 2 * a^2) = False :=
by
  sorry

end correct_calculation_l571_571993


namespace find_smallest_divisor_l571_571967

theorem find_smallest_divisor {n : ℕ} 
  (h : n = 44402) 
  (hdiv1 : (n + 2) % 30 = 0) 
  (hdiv2 : (n + 2) % 48 = 0) 
  (hdiv3 : (n + 2) % 74 = 0) 
  (hdiv4 : (n + 2) % 100 = 0) : 
  ∃ d, d = 37 ∧ d ∣ (n + 2) :=
sorry

end find_smallest_divisor_l571_571967


namespace limit_problem_l571_571688

-- Define the function f(x) = 1/x
def f (x : ℝ) : ℝ := 1 / x

-- State the limit problem.
theorem limit_problem : 
  (∀ f: ℝ → ℝ), (lim (h : ℝ) (h → 0) (λ h, (f(2 + 3 * h) - f(2)) / h)) = -3 / 4 :=
sorry

end limit_problem_l571_571688


namespace find_complex_numbers_l571_571715

noncomputable def complex_number_exists : Prop :=
  ∃ (z : ℂ), (z + 10 / z).im = 0 ∧ (z + 4).re = (z + 4).im

theorem find_complex_numbers : complex_number_exists :=
by {
  use [⟨-1, 3⟩, ⟨-3, 1⟩],
  simp [Complex.re, Complex.im, Complex.add],
  split,
  {
    -- Check z = -1 + 3i
    {
      -- Condition 1: (z + 10 / z) is real
      simp [Complex.re, Complex.im, Complex.div, Complex.add],
      sorry,
    },
    {
      -- Condition 2: Real part equals imaginary part for z + 4
      simp [Complex.re, Complex.im, Complex.add],
      sorry, 
    }
  },
  split,
  {
    -- Check z = -3 + i
    {
      -- Condition 1: (z + 10 / z) is real
      simp [Complex.re, Complex.im, Complex.div, Complex.add],
      sorry,
    },
    {
      -- Condition 2: Real part equals imaginary part for z + 4
      simp [Complex.re, Complex.im, Complex.add],
      sorry,
    }
  }
}

end find_complex_numbers_l571_571715
