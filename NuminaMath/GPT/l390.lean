import Mathlib

namespace consecutive_even_integer_bases_l390_390224

/-- Given \(X\) and \(Y\) are consecutive even positive integers and the equation
\[ 241_X + 36_Y = 94_{X+Y} \]
this theorem proves that \(X + Y = 22\). -/
theorem consecutive_even_integer_bases (X Y : ℕ) (h1 : X > 0) (h2 : Y = X + 2)
    (h3 : 2 * X^2 + 4 * X + 1 + 3 * Y + 6 = 9 * (X + Y) + 4) : 
    X + Y = 22 :=
by sorry

end consecutive_even_integer_bases_l390_390224


namespace calculate_expression_l390_390073

theorem calculate_expression :
  ((12 ^ 12 / 12 ^ 11) ^ 2 * 4 ^ 2) / 2 ^ 4 = 144 :=
by
  sorry

end calculate_expression_l390_390073


namespace eccentricity_range_l390_390265

noncomputable def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  { z : ℝ × ℝ // (z.1^2 / a^2) + (z.2^2 / b^2) = 1 }

theorem eccentricity_range (a b : ℝ) (h : a > b) (hb : b > 0):
  ∀ (P : ellipse a b h hb),
    |(P.1, P.2 - b).norm ≤ 2 * b → 
    eccentricity (a b) ∈ set.Ici (real.sqrt 2 / 2) :=
sorry

end eccentricity_range_l390_390265


namespace count_numbers_without_2_or_3_between_1_and_1500_l390_390187

/-- There are 1023 whole numbers between 1 and 1500 that do not contain the digits 2 or 3. -/
theorem count_numbers_without_2_or_3_between_1_and_1500 : 
  (finset.range 1500).filter (λ n, ∀ d ∈ n.digits 10, d ≠ 2 ∧ d ≠ 3)).card = 1023 :=
sorry

end count_numbers_without_2_or_3_between_1_and_1500_l390_390187


namespace range_of_b_minimum_value_of_M_l390_390108

-- Prove (1)
theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, 0 < x → log x + x - 3 + b / x ≥ 0) → b ≥ 2 := 
  by 
    sorry

-- Prove (2)
theorem minimum_value_of_M (a : ℝ) (M : ℝ) :
  (∀ x : ℝ, 0 < x → log x + x - a + b / x ≥ 0) →
  (0 < a → (∃ b : ℝ, 0 < b ∧ a / (b + 1) ≤ M)) → M ≥ 1 := 
  by 
    sorry

end range_of_b_minimum_value_of_M_l390_390108


namespace sum_distances_tetrahedron_base_l390_390378

noncomputable def sum_distances_equal (A B C D M P : Point) : Prop :=
  is_regular_tetrahedron_base_square A B C D M ∧ 
  P ∈ square_or_perimeter A B C D → 
  dist_point_to_plane P (plane M A B) + 
  dist_point_to_plane P (plane M C D) = 
  dist_point_to_plane P (plane M B C) + 
  dist_point_to_plane P (plane M D A)

theorem sum_distances_tetrahedron_base 
  {A B C D M P : Point} 
  (h1 : is_regular_tetrahedron_base_square A B C D M)
  (h2 : P ∈ square_or_perimeter A B C D) : 
  dist_point_to_plane P (plane M A B) + 
  dist_point_to_plane P (plane M C D) = 
  dist_point_to_plane P (plane M B C) + 
  dist_point_to_plane P (plane M D A) :=
sorry

end sum_distances_tetrahedron_base_l390_390378


namespace range_of_eccentricity_l390_390248

noncomputable def upperVertex (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : ℝ × ℝ := (0, b)

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem range_of_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h: a > b) :
  (∀ (x y : ℝ), ellipse a b x y → distance x y 0 b ≤ 2 * b) →
  ∃ (e : ℝ), e ∈ Set.Ioc 0 (Real.sqrt 2 / 2) ∧ a^2 = b^2 / (1 - e^2) := sorry

end range_of_eccentricity_l390_390248


namespace matchstick_game_winner_a_matchstick_game_winner_b_l390_390402

def is_winning_position (pile1 pile2 : Nat) : Bool :=
  (pile1 % 2 = 1) && (pile2 % 2 = 1)

theorem matchstick_game_winner_a : is_winning_position 101 201 = true := 
by
  -- Theorem statement for (101 matches, 201 matches)
  -- The second player wins
  sorry

theorem matchstick_game_winner_b : is_winning_position 100 201 = false := 
by
  -- Theorem statement for (100 matches, 201 matches)
  -- The first player wins
  sorry

end matchstick_game_winner_a_matchstick_game_winner_b_l390_390402


namespace sum_of_alternating_sums_l390_390552

def alternating_sum (S : Finset ℕ) : ℤ :=
  if S = ∅ then 0
  else let L := (S.sort (· ≥ ·)).toList in
       List.alternating_sum L

theorem sum_of_alternating_sums (n : ℕ) (s : Finset ℕ) : 
  n = 8 → s = {1, 2, 3, 4, 5, 6, 7, 8} →
  Finset.sum (s.powerset.filter (λ t, t ≠ ∅)) alternating_sum = 1024 :=
by
  intros n_val s_val
  rw [n_val, s_val]
  sorry

end sum_of_alternating_sums_l390_390552


namespace range_of_a_l390_390200

theorem range_of_a (a : ℝ) (h : (2 - a)^3 > (a - 1)^3) : a < 3/2 :=
sorry

end range_of_a_l390_390200


namespace calculation_correct_l390_390509

theorem calculation_correct : 
  sqrt 12 + 2 * Real.sin (60 * (Real.pi / 180)) - abs (1 - sqrt 3) - (2023 - Real.pi)^0 = 2 * sqrt 3 :=
by
  sorry

end calculation_correct_l390_390509


namespace possible_values_of_ratio_l390_390534

theorem possible_values_of_ratio (a d : ℝ) (h : a ≠ 0) (h_eq : a^2 - 6 * a * d + 8 * d^2 = 0) : 
  ∃ x : ℝ, (x = 1/2 ∨ x = 1/4) ∧ x = d/a :=
by
  sorry

end possible_values_of_ratio_l390_390534


namespace ab_bc_ca_value_a4_b4_c4_value_l390_390897

theorem ab_bc_ca_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  ab + bc + ca = -1/2 :=
sorry

theorem a4_b4_c4_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a^4 + b^4 + c^4 = 1/2 :=
sorry

end ab_bc_ca_value_a4_b4_c4_value_l390_390897


namespace complex_magnitude_of_3_minus_4i_l390_390530

open Complex

theorem complex_magnitude_of_3_minus_4i : Complex.abs ⟨3, -4⟩ = 5 := sorry

end complex_magnitude_of_3_minus_4i_l390_390530


namespace number_of_integers_in_sequence_6561_l390_390112

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

def count_integers_in_sequence (a b : ℝ) : ℕ :=
  Nat.card {n : ℕ | n > 0 ∧ is_integer (a ^ (1 / n.toReal))}

theorem number_of_integers_in_sequence_6561 :
  count_integers_in_sequence 6561 8 = 3 :=
sorry

end number_of_integers_in_sequence_6561_l390_390112


namespace dodecagon_area_l390_390975

/-
Given a convex 12-sided polygon (dodecagon) where:
1. All angles are equal.
2. Ten sides have length 1.
3. One side has length 2.

Prove that the area of the polygon is 8 + 4sqrt(3).
-/
theorem dodecagon_area : 
  ∀ (polygon : Polygon),
  polygon.sides = 12 ∧ 
  polygon.convex ∧ 
  (∀ i < 12, polygon.side_length i = 1 ∨ polygon.side_length i = 2) →
  (polygon.side_length_count 1 = 10 ∧ polygon.side_length_count 2 = 1) →
  polygon.equal_angles →
  polygon.area = 8 + 4 * Real.sqrt 3 := 
by 
  sorry

end dodecagon_area_l390_390975


namespace salary_increase_l390_390754

theorem salary_increase (x : ℕ) (hB_C_sum : 2*x + 3*x = 6000) : 
  ((3 * x - 1 * x) / (1 * x) ) * 100 = 200 :=
by
  -- Placeholder for the proof
  sorry

end salary_increase_l390_390754


namespace word_count_proof_l390_390613

def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def consonants : List Char := ['B', 'C', 'D', 'F']
def vowels : List Char := ['A', 'E']

def count_unrestricted_words : ℕ := 6 ^ 5
def count_all_vowel_words : ℕ := 2 ^ 5
def count_one_consonant_words : ℕ := 5 * 4 * (2 ^ 4)
def count_fewer_than_two_consonant_words : ℕ := count_all_vowel_words + count_one_consonant_words

def count_words_with_at_least_two_consonants : ℕ :=
  count_unrestricted_words - count_fewer_than_two_consonant_words

theorem word_count_proof :
  count_words_with_at_least_two_consonants = 7424 := 
by
  -- Proof will be provided here. For now we skip it.
  sorry

end word_count_proof_l390_390613


namespace find_incenter_minimizes_l390_390080

-- Defining the vertices of triangle ABC
variable (A B C P : Point)

-- Point P is inside triangle ABC
variable (h1 : InsideTriangle P A B C)

-- Defining the feet of perpendiculars from P to the lines BC, CA, AB
variable (D E F : Point)
variable (hD : FootOfPerpendicular D P B C)
variable (hE : FootOfPerpendicular E P C A)
variable (hF : FootOfPerpendicular F P A B)

-- The function that we want to minimize
def sum_perpendiculars (BC CA AB : Real) (PD PE PF : Real) : Real := 
  (BC / PD) + (CA / PE) + (AB / PF)

-- The statement that P being the incenter minimizes the sum
theorem find_incenter_minimizes
  (h1 : InsideTriangle P A B C)
  (hD : FootOfPerpendicular D P B C)
  (hE : FootOfPerpendicular E P C A)
  (hF : FootOfPerpendicular F P A B)
  : IsIncenter P A B C → 
    (∀ Q : Point, InsideTriangle Q A B C → 
      sum_perpendiculars (distance B C) (distance C A) (distance A B) (distance P D) (distance P E) (distance P F) ≤
      sum_perpendiculars (distance B C) (distance C A) (distance A B) (distance Q (foot_of_perpendicular Q B C)) (distance Q (foot_of_perpendicular Q C A)) (distance Q (foot_of_perpendicular Q A B))) :=
sorry

end find_incenter_minimizes_l390_390080


namespace num_5_letter_words_with_at_least_two_consonants_l390_390611

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := 6^5
def words_with_0_consonants : ℕ := 2^5
def words_with_1_consonant : ℕ := 5 * 4 * 2^4

theorem num_5_letter_words_with_at_least_two_consonants : 
  total_5_letter_words - (words_with_0_consonants + words_with_1_consonant) = 7424 := by
  sorry

end num_5_letter_words_with_at_least_two_consonants_l390_390611


namespace golden_section_proportionality_l390_390189

theorem golden_section_proportionality (A B C : ℝ) (hC : C = (A + B) / 2) (hAC_GT_BC : A > B) :
  AB / AC = AC / BC :=
sorry

end golden_section_proportionality_l390_390189


namespace smallest_m_value_l390_390930

noncomputable def smallest_m : ℝ :=
  let k := -1 in
  -(k * Real.pi / 2) - (Real.pi / 12)

theorem smallest_m_value (m : ℝ) :
  (∀ (x : ℝ), sin (2 * (x - m) + Real.pi / 3) = sin (2 * (-x - m) + Real.pi / 3)) → 0 < m → m = smallest_m :=
by
  sorry

end smallest_m_value_l390_390930


namespace find_m_l390_390114

theorem find_m (x y m : ℤ) (h1 : x = -2) (h2 : y = 1) (h3 : m * x + 5 * y = -1) : m = 3 :=
begin
  sorry
end

end find_m_l390_390114


namespace smallest_value_expression_l390_390583

theorem smallest_value_expression (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  ∃ m, m = y ∧ m = 3 :=
by
  sorry

end smallest_value_expression_l390_390583


namespace relationship_between_y_l390_390576

theorem relationship_between_y
  (m y₁ y₂ y₃ : ℝ)
  (hA : y₁ = -(-1)^2 + 2 * -1 + m)
  (hB : y₂ = -(1)^2 + 2 * 1 + m)
  (hC : y₃ = -(2)^2 + 2 * 2 + m) :
  y₁ < y₃ ∧ y₃ < y₂ :=
sorry

end relationship_between_y_l390_390576


namespace battery_will_last_more_hours_l390_390350

noncomputable def battery_lifetime (total_hours_on : ℕ) (hours_used : ℕ) : ℕ :=
let deplete_rate_not_in_use := (1 : ℚ) / 36
let deplete_rate_in_use := (1 : ℚ) / 4
let battery_used_not_in_use := (total_hours_on - hours_used) * deplete_rate_not_in_use
let battery_used_in_use := hours_used * deplete_rate_in_use
let total_battery_used := battery_used_not_in_use + battery_used_in_use
let remaining_battery := 1 - total_battery_used
remaining_battery / deplete_rate_not_in_use

theorem battery_will_last_more_hours :
  battery_lifetime 12 2 = 8 := by
sorry

end battery_will_last_more_hours_l390_390350


namespace eccentricity_range_l390_390293

-- Ellipse definition
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Upper vertex B of the ellipse
def B (b : ℝ) : ℝ × ℝ := (0, b)

-- Distance PB condition
def distance_le_2b (a b x0 y0 : ℝ) (h : ellipse a b (and.intro (sorry) (sorry)) x0 y0) : Prop :=
  (x0 - 0)^2 + (y0 - b)^2 ≤ (2 * b)^2

-- Range of eccentricity
def eccentricity (a b c : ℝ) : ℝ :=
  c / a

-- Main theorem
theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  (forall (x0 y0 : ℝ), ellipse a b h x0 y0 → distance_le_2b a b x0 y0 h) →
  eccentricity a b (sqrt (a^2 - b^2)) ∈ set.Icc 0 (sqrt 2 / 2) :=
by
  sorry

end eccentricity_range_l390_390293


namespace quantity_4th_month_is_935_bikes_exceed_capacity_at_max_l390_390485

/-- Definitions for a_n and b_n -/
def a_n (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 3 then 5 * n^4 + 15 else if n ≥ 4 then -10 * n + 470 else 0

def b_n (n : ℕ) : ℕ := n + 5

/-- Definition for the total quantity of shared bicycles at the end of the 4th month -/
def quantity_4th_month : ℕ := ∑ i in Finset.range 4, a_n i - ∑ i in Finset.range 4, b_n i

/-- Definition for the parking capacity at the end of the 42nd month -/
def S_n (n : ℕ) : ℕ := -4 * (n - 46)^2 + 8800

/-- Proof that the quantity of shared bicycles at the end of the 4th month is 935 -/
theorem quantity_4th_month_is_935 : quantity_4th_month = 935 := by sorry

/-- Proof that the quantity of shared bicycles exceeds the parking capacity at its maximum -/
theorem bikes_exceed_capacity_at_max : ∑ i in Finset.range 42, a_n i - ∑ i in Finset.range 42, b_n i > S_n 42 := by sorry

end quantity_4th_month_is_935_bikes_exceed_capacity_at_max_l390_390485


namespace sphere_surface_area_l390_390355

noncomputable def surface_area_of_sphere (A B C : Point) (O : Point) (AB : ℝ) (angleACB : ℝ) (OC_distance : ℝ) : ℝ :=
  let r := AB / (2 * real.sin (angleACB / 180 * real.pi)) in
  let R := real.sqrt (OC_distance ^ 2 + r ^ 2) in
  4 * real.pi * R ^ 2

theorem sphere_surface_area (A B C O : Point)
  (hAB : dist A B = 4 * real.sqrt 3)
  (hAngleACB : angle A C B = 60)
  (hOC_distance : OdistPlane O A B C = 3) :
  surface_area_of_sphere A B C O (4 * real.sqrt 3) 60 3 = 100 * real.pi := by
  sorry

end sphere_surface_area_l390_390355


namespace series_sum_correct_l390_390457

-- Definition of the series sum
def series_sum (x : ℕ) (y : ℕ) : ℕ :=
  ∑ n in finset.range (y + 1), (2 * n) ^ 3 * n.factorial

-- The main theorem statement in Lean 4
theorem series_sum_correct (x : ℕ) (hx : even x) (y : ℕ) (hy : y = x / 2) : 
  series_sum x y = ∑ n in finset.range (y + 1), (2 * n) ^ 3 * n.factorial := 
by 
  sorry

end series_sum_correct_l390_390457


namespace length_JK_l390_390408

-- Given conditions as definitions
variables (GHI JKL : Triangle)
variable [Similar GHI JKL]
variable (GH HI JK KL : ℝ)
variable (hGH : GH = 8)
variable (hHI : HI = 16)
variable (hKL : KL = 24)

-- Using the conditions to prove the question == answer
theorem length_JK : JK = 12 := by
  sorry

end length_JK_l390_390408


namespace max_min_f_cos_value_of_x0_monotonic_omega_l390_390944

namespace ProofProblems

def a := (λ x : ℝ, 2 * Real.cos x, 1)
def b := (λ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x, -1)
def f (x : ℝ) := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem max_min_f : 
  1 ≤ f x ∧ 2 ≤ f x 
  (0 ≤ x ∧ x ≤ Real.pi / 4) :=
sorry

theorem cos_value_of_x0 (x0 : ℝ) 
  (hx0 : Real.pi / 4 ≤ x0 ∧ x0 ≤ Real.pi / 2)
  (hf : f x0 = 6 / 5) :
  Real.cos (2 * x0) = (3 - 4 * Real.sqrt 3) / 10 :=
sorry

theorem monotonic_omega (ω : ℝ) 
  (ω_pos : 0 < ω) 
  (h_interval : (Real.pi / 3 < x ∧ x < 2 * Real.pi / 3))
  (h_monotonic : ∀ x y, x < y → f (ω * x) < f (ω * y)) :
  0 < ω ∧ ω ≤ 1 / 4 :=
sorry

end ProofProblems

end max_min_f_cos_value_of_x0_monotonic_omega_l390_390944


namespace dart_throw_probability_l390_390234

theorem dart_throw_probability (p : ℝ) (threshold : ℝ) (lg2_approx : ℝ) :
  p = 0.6 →
  threshold = 0.9 →
  lg2_approx = 0.3 →
  ∃ n : ℕ, 1 - (1 - p)^n > threshold ∧ n ≥ 3 :=
by
  intros hp hthreshold hlg2
  use 3
  split
  { 
    -- Here we would provide the proof that 1 - (1 - 0.6) ^ 3 > 0.9, but for now just assume it.
    sorry 
  }
  {
    exact Nat.le_refl 3
  }

end dart_throw_probability_l390_390234


namespace four_identical_absolute_differences_l390_390121

-- Condition: 11 different natural numbers not greater than 20
def condition (S : set ℕ) : Prop :=
  S.card = 11 ∧ ∀ x ∈ S, x ≤ 20

-- The problem is to prove that there are at least 4 identical absolute differences among the pairwise differences of the set.
theorem four_identical_absolute_differences (S : set ℕ) (hS : condition S) :
  ∃ (d : ℕ), (∃ x y z w ∈ S, x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧ abs (x - y) = d ∧ abs (x - z) = d ∧ abs (x - w) = d ∧ abs (y - z) = d) :=
sorry

end four_identical_absolute_differences_l390_390121


namespace proof_f_z_l390_390564

noncomputable def z : ℂ := (i + 1) / (i - 1)

def f (x : ℂ) : ℂ := x^2 - x + 1

theorem proof_f_z : f(z) = i := 
by
  sorry

end proof_f_z_l390_390564


namespace find_f_neg_2_l390_390139

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 3^x - 1 else sorry -- we'll define this not for non-negative x properly later

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem find_f_neg_2 (hodd : is_odd_function f) (hpos : ∀ x : ℝ, 0 ≤ x → f x = 3^x - 1) :
  f (-2) = -8 :=
by
  -- Proof omitted
  sorry

end find_f_neg_2_l390_390139


namespace sum_abs_roots_polynomial_l390_390879

theorem sum_abs_roots_polynomial :
  let poly := Polynomial.C 1 * Polynomial.X^4 - Polynomial.C 6 * Polynomial.X^3 + Polynomial.C 9 * Polynomial.X^2 + Polynomial.C 24 * Polynomial.X - Polynomial.C 36 in
  let roots := [3 + Real.sqrt 3, 3 - Real.sqrt 3, Real.sqrt 6, -Real.sqrt 6] in
  (roots.map Real.abs).sum = 6 + 2 * Real.sqrt 6 :=
by
  let poly := Polynomial.C 1 * Polynomial.X^4 - Polynomial.C 6 * Polynomial.X^3 + Polynomial.C 9 * Polynomial.X^2 + Polynomial.C 24 * Polynomial.X - Polynomial.C 36
  let roots := [3 + Real.sqrt 3, 3 - Real.sqrt 3, Real.sqrt 6, -Real.sqrt 6]
  have h : (roots.map Real.abs).sum = 6 + 2 * Real.sqrt 6 := sorry
  exact h

end sum_abs_roots_polynomial_l390_390879


namespace num_nat_numbers_satisfy_sqrt_condition_l390_390190

theorem num_nat_numbers_satisfy_sqrt_condition :
  { n : ℕ // (∃ k : ℕ, k * k = 12 - n) }.to_finset.card = 4 :=
sorry

end num_nat_numbers_satisfy_sqrt_condition_l390_390190


namespace tangent_line_equation_range_of_a_l390_390931

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - (3/2) * x^2 + 1

theorem tangent_line_equation (a : ℝ) (h : a = 1) :
  ∀ x, f a x = a * x ^ 3 - (3 / 2) * x ^ 2 + 1 →
  ∀ x, deriv (f a) x = 3 * a * x ^ 2 - 3 * x →
  deriv (f a) 2 = 6 ∧ f a 2 = 3 ∧ ∃ m b, m = 6 ∧ b = -9 ∧ (∀ x, f a x = a * x ^ 3 - (3 / 2) * x ^ 2 + 1 → tangent (λ x, f a x) 2 m b) := sorry

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ∈ Icc (-1) (1/2) → f a x < a^2) : 1 < a := sorry

end tangent_line_equation_range_of_a_l390_390931


namespace inequality_not_true_l390_390954

theorem inequality_not_true (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬ (a > 0) :=
sorry

end inequality_not_true_l390_390954


namespace range_of_eccentricity_of_ellipse_l390_390259

-- Definitions for the problem.
def is_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1) 

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

def on_upper_vertex (b : ℝ) : (ℝ × ℝ) := (0, b)

-- The statement to prove the range of eccentricity.
theorem range_of_eccentricity_of_ellipse (a b : ℝ) (x y : ℝ) (e : ℝ) :
  is_on_ellipse a b x y →
  (distance x y 0 b ≤ 2 * b) →
  ∃ e, (e = (1 - (b^2 / a^2))^(1/2) ∧ (0 < e) ∧ (e ≤ (sqrt 2)/2)) :=
by
  sorry

end range_of_eccentricity_of_ellipse_l390_390259


namespace rhombus_diagonals_l390_390042

variable {m n : ℝ}

theorem rhombus_diagonals (hm : m ≥ 0) (hn : n ≥ 0) : 
  let s := m + n,
      BK := real.sqrt (2 * m * n + n^2),
      BD := real.sqrt (2 * n * (m + n)),
      AC := real.sqrt (4 * m^2 + 6 * m * n + 2 * n^2)
  in BD = real.sqrt (2 * n * (m + n)) ∧ AC = real.sqrt (4 * m^2 + 6 * m * n + 2 * n^2) := 
sorry

end rhombus_diagonals_l390_390042


namespace proportional_segments_l390_390444

-- Define the property of being proportional
def isProportional (a b c d : Nat) : Prop :=
  a * d = b * c

-- Define the sets
def SetA := (1, 2, 4, 6)
def SetB := (3, 4, 7, 8)
def SetC := (2, 4, 8, 16)
def SetD := (1, 3, 5, 7)

-- Theorem to prove
theorem proportional_segments :
  (¬ isProportional SetA.1 SetA.2 SetA.3 SetA.4) ∧
  (¬ isProportional SetB.1 SetB.2 SetB.3 SetB.4) ∧
  (isProportional SetC.1 SetC.2 SetC.3 SetC.4) ∧
  (¬ isProportional SetD.1 SetD.2 SetD.3 SetD.4) :=
by
  sorry

end proportional_segments_l390_390444


namespace min_red_hair_students_l390_390017

variable (B N R : ℕ)
variable (total_students : ℕ := 50)

theorem min_red_hair_students :
  B + N + R = total_students →
  (∀ i, B > i → N > 0) →
  (∀ i, N > i → R > 0) →
  R ≥ 17 :=
by {
  -- The specifics of the proof are omitted as per the instruction
  sorry
}

end min_red_hair_students_l390_390017


namespace expression_one_simplifies_expression_two_simplifies_l390_390851

theorem expression_one_simplifies : 
  ((1 : ℝ) * (1 + (1 / 2))^0 - (1 - (0.5)⁻²) / ((27 / 8)^(2 / 3)) = 7 / 3) :=
by
  sorry

theorem expression_two_simplifies : 
  (sqrt (2 * sqrt (2 * sqrt (2)))) = (2^(7 / 8 : ℝ)) :=
by
  sorry

end expression_one_simplifies_expression_two_simplifies_l390_390851


namespace polygon_labeling_l390_390008

theorem polygon_labeling (n : ℕ) (h1 : 2 * n + 1 = M) :
  ∃ (labeling : finset ℕ → ℕ), (∀ s ∈ M, (∑ x in s, labeling x) = 8 * n + 4) :=
by
  sorry

end polygon_labeling_l390_390008


namespace max_value_of_distances_MP_and_MQ_l390_390566

noncomputable def max_value_of_MP_add_MQ : ℝ :=
  4 * real.sqrt 2 

theorem max_value_of_distances_MP_and_MQ 
  (k : ℝ)
  (P Q M : ℝ × ℝ)
  (hP : P = (0, 0))
  (hQ : Q = (2, 2))
  (hM : dist_sq P M + dist_sq Q M = 8) : 
  |dist P M + |dist Q M| ≤ max_value_of_MP_add_MQ := sorry

end max_value_of_distances_MP_and_MQ_l390_390566


namespace highest_growth_total_employees_jan_l390_390512

-- Define the conditions
variable (P_Dec : ℕ) (Q_Dec : ℕ) (R_Dec : ℕ) (P_Jan : ℕ) (Q_Jan : ℕ) (R_Jan : ℕ)

def companyData :=
  P_Dec = 515 ∧ Q_Dec = 558 ∧ R_Dec = 611 ∧
  P_Dec = 1.15 * P_Jan ∧
  Q_Dec = 1.105 * Q_Jan ∧
  R_Dec = 1.195 * R_Jan

-- Proving the highest growth rate
theorem highest_growth (h : companyData P_Dec Q_Dec R_Dec P_Jan Q_Jan R_Jan) :
  let P_gain := P_Dec - P_Jan
  let Q_gain := Q_Dec - Q_Jan
  let R_gain := R_Dec - R_Jan
  R_gain = max P_gain (max Q_gain R_gain) :=
sorry

-- Proving the total combined employees in January
theorem total_employees_jan (h : companyData P_Dec Q_Dec R_Dec P_Jan Q_Jan R_Jan) :
  P_Jan + Q_Jan + R_Jan = 1464 :=
sorry

end highest_growth_total_employees_jan_l390_390512


namespace no_solution_inequalities_l390_390204

theorem no_solution_inequalities (m : Real) : 
  (¬ ∃ (x : Real), (x - 2 < 3x - 6) ∧ (x < m)) → m ≤ 2 :=
sorry

end no_solution_inequalities_l390_390204


namespace find_n_l390_390549

theorem find_n (x y : ℝ) (h1 : (7 * x + 2 * y) / (x - n * y) = 23) (h2 : x / (2 * y) = 3 / 2) :
  ∃ n : ℝ, n = 2 := by
  sorry

end find_n_l390_390549


namespace eccentricity_range_l390_390287

noncomputable def ellipse_eccentricity (a b : ℝ) (h_ab : a > b) : ℝ := (Real.sqrt (a ^ 2 - b ^ 2)) / a

theorem eccentricity_range (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ) :
  (∃ x y : ℝ, (x, y) = P ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  (Real.sqrt (x^2 + (y - b)^2) ≤ 2 * b)) → 
  0 < ellipse_eccentricity a b h_ab ∧ ellipse_eccentricity a b h_ab ≤ Real.sqrt 2 / 2 :=
begin
  sorry
end

end eccentricity_range_l390_390287


namespace sean_more_whistles_than_charles_l390_390718

theorem sean_more_whistles_than_charles :
  ∀ (sean_whistles charles_whistles difference : ℕ), 
  sean_whistles = 45 → 
  charles_whistles = 13 → 
  difference = sean_whistles - charles_whistles → 
  difference = 32 :=
by
  intros sean_whistles charles_whistles difference h_sean h_charles h_difference
  rw [h_sean, h_charles, h_difference]
  sorry

end sean_more_whistles_than_charles_l390_390718


namespace max_val_4ab_sqrt3_12bc_l390_390684

theorem max_val_4ab_sqrt3_12bc {a b c : ℝ} (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 3) :
  4 * a * b * Real.sqrt 3 + 12 * b * c ≤ Real.sqrt 39 :=
sorry

end max_val_4ab_sqrt3_12bc_l390_390684


namespace taco_truck_earnings_l390_390478

/-
Question: How many dollars did the taco truck make during the lunch rush?
Conditions:
1. Soft tacos are $2 each.
2. Hard shell tacos are $5 each.
3. The family buys 4 hard shell tacos and 3 soft tacos.
4. There are ten other customers.
5. Each of the ten other customers buys 2 soft tacos.
Answer: The taco truck made $66 during the lunch rush.
-/

theorem taco_truck_earnings :
  let soft_taco_price := 2
  let hard_taco_price := 5
  let family_hard_tacos := 4
  let family_soft_tacos := 3
  let other_customers := 10
  let other_customers_soft_tacos := 2
  (family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price +
   other_customers * other_customers_soft_tacos * soft_taco_price) = 66 := by
  sorry

end taco_truck_earnings_l390_390478


namespace taco_truck_revenue_l390_390480

-- Conditions
def price_of_soft_taco : ℕ := 2
def price_of_hard_taco : ℕ := 5
def family_soft_tacos : ℕ := 3
def family_hard_tacos : ℕ := 4
def other_customers : ℕ := 10
def soft_tacos_per_other_customer : ℕ := 2

-- Calculation
def total_soft_tacos : ℕ := family_soft_tacos + other_customers * soft_tacos_per_other_customer
def revenue_from_soft_tacos : ℕ := total_soft_tacos * price_of_soft_taco
def revenue_from_hard_tacos : ℕ := family_hard_tacos * price_of_hard_taco
def total_revenue : ℕ := revenue_from_soft_tacos + revenue_from_hard_tacos

-- The proof problem
theorem taco_truck_revenue : total_revenue = 66 := 
by 
-- The proof should go here
sorry

end taco_truck_revenue_l390_390480


namespace find_smaller_number_l390_390703

def one_number_is_11_more_than_3times_another (x y : ℕ) : Prop :=
  y = 3 * x + 11

def their_sum_is_55 (x y : ℕ) : Prop :=
  x + y = 55

theorem find_smaller_number (x y : ℕ) (h1 : one_number_is_11_more_than_3times_another x y) (h2 : their_sum_is_55 x y) :
  x = 11 :=
by
  -- The proof will be inserted here
  sorry

end find_smaller_number_l390_390703


namespace word_count_proof_l390_390615

def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def consonants : List Char := ['B', 'C', 'D', 'F']
def vowels : List Char := ['A', 'E']

def count_unrestricted_words : ℕ := 6 ^ 5
def count_all_vowel_words : ℕ := 2 ^ 5
def count_one_consonant_words : ℕ := 5 * 4 * (2 ^ 4)
def count_fewer_than_two_consonant_words : ℕ := count_all_vowel_words + count_one_consonant_words

def count_words_with_at_least_two_consonants : ℕ :=
  count_unrestricted_words - count_fewer_than_two_consonant_words

theorem word_count_proof :
  count_words_with_at_least_two_consonants = 7424 := 
by
  -- Proof will be provided here. For now we skip it.
  sorry

end word_count_proof_l390_390615


namespace smallest_prime_8_less_than_square_l390_390426

theorem smallest_prime_8_less_than_square :
  ∃ p : ℕ, (∃ n : ℤ, p = n^2 - 8) ∧ Nat.Prime p ∧ p > 0 ∧ (∀ q : ℕ, (∃ m : ℤ, q = m^2 - 8) ∧ Nat.Prime q → q ≥ p) :=
sorry

end smallest_prime_8_less_than_square_l390_390426


namespace imaginary_part_of_z_l390_390590

-- Define the given complex number z
def z : ℂ := (1 + 3 * complex.I) / (3 - complex.I)

-- State the theorem to prove the imaginary part of z
theorem imaginary_part_of_z : complex.im z = 2 / 5 :=
sorry

end imaginary_part_of_z_l390_390590


namespace final_position_relative_total_fuel_needed_l390_390771

noncomputable def navigation_records : List ℤ := [-7, 11, -6, 10, -5]

noncomputable def fuel_consumption_rate : ℝ := 0.5

theorem final_position_relative (records : List ℤ) : 
  (records.sum = 3) := by 
  sorry

theorem total_fuel_needed (records : List ℤ) (rate : ℝ) : 
  (rate * (records.map Int.natAbs).sum = 19.5) := by 
  sorry

#check final_position_relative navigation_records
#check total_fuel_needed navigation_records fuel_consumption_rate

end final_position_relative_total_fuel_needed_l390_390771


namespace smallest_multiple_of_6_and_15_l390_390547

theorem smallest_multiple_of_6_and_15 : 
  ∃ b : ℕ, b > 0 ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ ∀ c : ℕ, (c > 0 ∧ (6 ∣ c) ∧ (15 ∣ c)) → b ≤ c :=
begin
  use 30,
  split,
  { norm_num, },
  split,
  { use 5,
    norm_num, },
  split,
  { use 2,
    norm_num, },
  intros c hc,
  cases hc with hc_pos hc_conditions,
  cases hc_conditions with hc_6 hc_15,
  rw [←nat.dvd_add_self_left (nat.dvd_trans hc_15 ⟨2, rfl⟩) c 6, ←mul_one 15],
  exact nat.mul_le_mul_right 15 (nat.div_le_self 90 (gcd_pos_of_pos_left 30 (by norm_num; exact hc_pos))),
end

end smallest_multiple_of_6_and_15_l390_390547


namespace TotalNumberOfStudents_l390_390715

-- Define the initial values according to the conditions.
def RongRongHeight : ℕ := 140
def LeiLeiHeight : ℕ := 158
def classOneAvgIncrease : Nat := 2
def classTwoAvgDecrease : Nat := 3

-- Definitions for the number of students in each class.
noncomputable def n1 : ℕ :=
  let avg1 := H1 + classOneAvgIncrease
  let eq1 := avg1 * n1 = H1 * n1 + LeiLeiHeight - RongRongHeight
  eq1.solve

noncomputable def n2 : ℕ :=
  let avg2 := H2 - classTwoAvgDecrease
  let eq2 := avg2 * n2 = H2 * n2 - LeiLeiHeight + RongRongHeight
  eq2.solve

-- Definition of the total number of students
def totalStudents : ℕ :=
  n1 + n2

-- Statement to prove that the total number of students is 15
theorem TotalNumberOfStudents : totalStudents = 15 := by
  have h1 : n1 = 9 := by sorry
  have h2 : n2 = 6 := by sorry
  show 9 + 6 = 15 from rfl

end TotalNumberOfStudents_l390_390715


namespace largest_square_in_space_inside_square_outside_triangles_l390_390992

def square_side_length : ℝ := 15
def base_angle : ℝ := real.pi / 4  -- 45 degrees in radians

noncomputable def largest_inscribed_square_side_length : ℝ :=
  square_side_length * (2 - real.sqrt 2) / 4

theorem largest_square_in_space_inside_square_outside_triangles :
  ∃ (s : ℝ), s = largest_inscribed_square_side_length :=
by 
  sorry

end largest_square_in_space_inside_square_outside_triangles_l390_390992


namespace right_triangle_area_l390_390776

theorem right_triangle_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  (1 / 2 : ℝ) * a * b = 24 := by
  sorry

end right_triangle_area_l390_390776


namespace solve_quadratic_eq_solve_inequalities_l390_390458

-- Proof Problem 1
theorem solve_quadratic_eq (x : ℝ) : x^2 - 2 * x - 4 = 0 ↔ (x = 1 + sqrt 5 ∨ x = 1 - sqrt 5) :=
by
  sorry

-- Proof Problem 2
theorem solve_inequalities (x : ℝ) :
  (2 * (x - 1) ≥ -4) ∧ ((3 * x - 6) / 2 < x - 1) ↔ (-1 ≤ x ∧ x < 4) :=
by
  sorry

end solve_quadratic_eq_solve_inequalities_l390_390458


namespace geometric_log_sum_l390_390934

theorem geometric_log_sum :
  (∑ i in Finset.range 11, Real.logBase 2 (1 * (2 ^ i))) = 55 := sorry

end geometric_log_sum_l390_390934


namespace eccentricity_range_l390_390289

-- Ellipse definition
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Upper vertex B of the ellipse
def B (b : ℝ) : ℝ × ℝ := (0, b)

-- Distance PB condition
def distance_le_2b (a b x0 y0 : ℝ) (h : ellipse a b (and.intro (sorry) (sorry)) x0 y0) : Prop :=
  (x0 - 0)^2 + (y0 - b)^2 ≤ (2 * b)^2

-- Range of eccentricity
def eccentricity (a b c : ℝ) : ℝ :=
  c / a

-- Main theorem
theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  (forall (x0 y0 : ℝ), ellipse a b h x0 y0 → distance_le_2b a b x0 y0 h) →
  eccentricity a b (sqrt (a^2 - b^2)) ∈ set.Icc 0 (sqrt 2 / 2) :=
by
  sorry

end eccentricity_range_l390_390289


namespace math_class_girls_ratio_l390_390391

theorem math_class_girls_ratio
  (total_students : ℕ)
  (boys_ratio girls_ratio math_ratio science_ratio : ℕ)
  (students_in_classes : ℕ)
  (h1 : boys_ratio = 5)
  (h2 : girls_ratio = 8)
  (h3 : math_ratio = 7)
  (h4 : science_ratio = 4)
  (h5 : students_in_classes = 458) :
  let x := students_in_classes / (math_ratio + science_ratio),
      y := total_students / (boys_ratio + girls_ratio)
  in girls_ratio * y = 184 :=
by
  sorry

end math_class_girls_ratio_l390_390391


namespace quadratic_function_has_minimum_l390_390909

-- Defining the conditions and the function f(x)
def quadratic_function (a b x : ℝ) : ℝ := a * x^2 + b * x + (-b^2 / (4 * a))

-- Theorem statement to prove that the quadratic_function has a minimum
theorem quadratic_function_has_minimum (a b : ℝ) (h : a > 0) (h_c : ∀ x, quadratic_function a b x = a * x^2 + b * x + (-b^2 / (4 * a))) :
    ∃ x₀ : ℝ, ∀ x : ℝ, quadratic_function a b x ≥ quadratic_function a b x₀ :=
by
  sorry

end quadratic_function_has_minimum_l390_390909


namespace smallest_harmonic_sum_number_exists_largest_harmonic_sum_number_exists_harmonic_number_digits_sum_even_possible_values_of_m_l390_390910

def is_sum_number (x y z : ℕ) : Prop := x = y + z
def is_harmonic_number (x y z : ℕ) : Prop := x = y^2 - z^2
def is_harmonic_sum_number (x y z : ℕ) : Prop := is_sum_number x y z ∧ is_harmonic_number x y z

theorem smallest_harmonic_sum_number_exists : 
  ∃ x y z, 100 ≤ x * 100 + y * 10 + z ∧ 
           x * 100 + y * 10 + z = 110 ∧ 
           is_harmonic_sum_number x y z :=
sorry

theorem largest_harmonic_sum_number_exists : 
  ∃ x y z, 100 ≤ x * 100 + y * 10 + z ∧ 
           x * 100 + y * 10 + z = 954 ∧ 
           is_harmonic_sum_number x y z :=
sorry

theorem harmonic_number_digits_sum_even (x y z : ℕ) 
  (h : is_harmonic_number x y z) : (x + y + z) % 2 = 0 :=
sorry

theorem possible_values_of_m (b c : ℕ) (hb : 0 ≤ b ∧ b ≤ 7) (hc : 1 ≤ c ∧ c ≤ 4) :
  is_sum_number 8 (b + 2) (3*c - 3) → 
  10 * b + 3 * c + 817 ∈ {880, 853, 826} :=
sorry

end smallest_harmonic_sum_number_exists_largest_harmonic_sum_number_exists_harmonic_number_digits_sum_even_possible_values_of_m_l390_390910


namespace range_of_eccentricity_l390_390252

noncomputable def upperVertex (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : ℝ × ℝ := (0, b)

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem range_of_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h: a > b) :
  (∀ (x y : ℝ), ellipse a b x y → distance x y 0 b ≤ 2 * b) →
  ∃ (e : ℝ), e ∈ Set.Ioc 0 (Real.sqrt 2 / 2) ∧ a^2 = b^2 / (1 - e^2) := sorry

end range_of_eccentricity_l390_390252


namespace omega_2n_eq_omega_n_not_omega_2n_plus_3_eq_omega_n_plus_1_omega_8n_plus_5_eq_omega_4n_plus_3_omega_pow2_n_minus_1_eq_n_l390_390330

def bin_weight (n : ℕ) : ℕ := n.binary_digits.count 1

theorem omega_2n_eq_omega_n (n : ℕ) : bin_weight (2 * n) = bin_weight n := by
  sorry

theorem not_omega_2n_plus_3_eq_omega_n_plus_1 (n : ℕ) : bin_weight (2 * n + 3) ≠ bin_weight n + 1 := by
  sorry

theorem omega_8n_plus_5_eq_omega_4n_plus_3 (n : ℕ) : bin_weight (8 * n + 5) = bin_weight (4 * n + 3) := by
  sorry

theorem omega_pow2_n_minus_1_eq_n (n : ℕ) : bin_weight (2^n - 1) = n := by
  sorry

end omega_2n_eq_omega_n_not_omega_2n_plus_3_eq_omega_n_plus_1_omega_8n_plus_5_eq_omega_4n_plus_3_omega_pow2_n_minus_1_eq_n_l390_390330


namespace f_seven_neg_two_l390_390918

noncomputable def f : ℝ → ℝ :=
  λ x, if x ∈ (Ioo 0 2) then 2 * x^2 else sorry

theorem f_seven_neg_two 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (x + 4) = f x)
  (h3 : ∀ x, x ∈ Ioo 0 2 → f x = 2 * x^2) :
  f 7 = -2 :=
by
  sorry

end f_seven_neg_two_l390_390918


namespace eccentricity_range_l390_390264

noncomputable def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  { z : ℝ × ℝ // (z.1^2 / a^2) + (z.2^2 / b^2) = 1 }

theorem eccentricity_range (a b : ℝ) (h : a > b) (hb : b > 0):
  ∀ (P : ellipse a b h hb),
    |(P.1, P.2 - b).norm ≤ 2 * b → 
    eccentricity (a b) ∈ set.Ici (real.sqrt 2 / 2) :=
sorry

end eccentricity_range_l390_390264


namespace segments_in_proportion_l390_390788

theorem segments_in_proportion : ∀ (a b c d : ℕ), a = 4 → b = 8 → c = 5 → d = 10 → a * d = b * c :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  exact Eq.refl 40

#eval segments_in_proportion 4 8 5 10 rfl rfl rfl rfl  -- Should return true

end segments_in_proportion_l390_390788


namespace kerosene_cost_l390_390214

theorem kerosene_cost :
  (∀ (x : ℝ),
    (∀ (dozenEggCost riceCost : ℝ), dozenEggCost = 0.33 → riceCost = 0.33 →
      12 * (dozenEggCost / 12) = riceCost → 
      x * (dozenEggCost / 12) = riceCost → 
      x = 12) ∧ 
      (2 * x * 0.0275 = 0.66)) :=
by
  intro x
  split
  sorry
  sorry

end kerosene_cost_l390_390214


namespace range_of_a_l390_390962

noncomputable def f (x a : ℝ) : ℝ :=
  (1 / 2) * x^2 - 2 * x + a * Real.log x

theorem range_of_a (a : ℝ) :
  (0 < a ∧ 4 - 4 * a > 0) ↔ (0 < a ∧ a < 1) :=
by
  sorry

end range_of_a_l390_390962


namespace angle_in_plane_l390_390699

theorem angle_in_plane (lines : Set (Set (Real × Real))) (h1 : 2 ≤ lines.card)
  (h2 : ∀ l1 l2 ∈ lines, l1 ≠ l2 → ¬ (l1 ∩ l2).card = 2) 
  (h3 : ∀ l1 l2 l3 ∈ lines, l1 ≠ l2 → l2 ≠ l3 → l1 ≠ l3 → (l1 ∩ l2 ∩ l3).card = 1) : 
  ∃ angle : Set (Real × Real), angle ⊆ ⋃₀ lines := sorry

end angle_in_plane_l390_390699


namespace average_price_correct_l390_390790

theorem average_price_correct (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : ∀ s : ℝ, s / a + s / b = s * (a + b) / (ab)) :
  let p := (2 * a * b) / (a + b)
  in p = (2 * a * b) / (a + b) ∧ a < p ∧ p < real.sqrt (a * b) :=
by
  let p := (2 * a * b) / (a + b)
  have h4: p = (2 * a * b) / (a + b) := rfl
  have h5 : a < p := sorry
  have h6 : p < real.sqrt (a * b) := sorry
  exact ⟨h4, ⟨h5, h6⟩⟩

end average_price_correct_l390_390790


namespace troy_piglets_l390_390768

theorem troy_piglets (total_straws : ℕ) (fraction_adult_pigs : ℚ) 
(straw_per_piglet : ℕ) (h1 : total_straws = 300) 
(h2 : fraction_adult_pigs = 3 / 5) 
(h3 : straw_per_piglet = 6) : 
  ∃ (piglets : ℕ), piglets = 30 :=
by 
  let adult_pigs_straws := fraction_adult_pigs * total_straws
  have h_adult_straws : adult_pigs_straws = 180
  { rw [h1, h2], norm_num, }
  have h_piglet_straws : adult_pigs_straws = 180 := h_adult_straws
  let piglets := adult_pigs_straws / straw_per_piglet
  have h_piglets : piglets = 30
  { rw [h3, h_adult_straws],
    norm_num, }
  exact ⟨piglets, h_piglets⟩

end troy_piglets_l390_390768


namespace monotonic_increasing_interval_num_zeros_g_l390_390148

-- Definitions for the functions f and g
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * (sin x)^2 + sin x * cos x - sqrt 3 / 2
noncomputable def g (x : ℝ) : ℝ := sin (2 * (x + π / 6) - π / 3) - 1

-- Assertion 1: Monotonicity of f(x) on [0, π/2]
theorem monotonic_increasing_interval :
  ∀ x ∈ Icc (0 : ℝ) (π / 2), f x = sin (2 * x - π / 3) :=
by sorry

-- Assertion 2: Number of zeros of g(x) on [0, 20π]
theorem num_zeros_g :
  ∃ k : ℕ, k = 40 ∧ ∀ x ∈ Icc (0 : ℝ) (20 * π), g x = 0 :=
by sorry

end monotonic_increasing_interval_num_zeros_g_l390_390148


namespace probability_of_selecting_r_standard_items_l390_390638

variables (N k l r : ℕ)

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_selecting_r_standard_items :
  let total_outcomes := binomial N l in
  let favorable_outcomes := binomial k r * binomial (N - k) (l - r) in
  total_outcomes ≠ 0 →
  (favorable_outcomes : ℚ) / total_outcomes = (binomial k r * binomial (N - k) (l - r) : ℚ) / binomial N l := 
by 
  sorry

end probability_of_selecting_r_standard_items_l390_390638


namespace solve_particle_path_count_l390_390818

-- Define a structure to represent a path
structure Path (start finish : (ℕ × ℕ)) :=
  (steps : list (ℕ × ℕ))
  (valid : ∀ i < steps.length, let (x, y) := steps.nth_le i sorry in
          (x, y) = (start.1 + 1, start.2) ∨ (x, y) = (start.1, start.2 + 1) ∨
          (x, y) = (start.1 + 1, start.2 + 1) ∨ (x, y) = (start.1 + 2, start.2))
  (no_right_angle : ∀ i < steps.length - 1, let (x1, y1) := steps.nth_le i sorry, (x2, y2) := steps.nth_le (i + 1) sorry in
            ¬((x1 = x2 ∧ (y2 = y1 + 1 ∨ y2 = y1 - 1)) ∨ (y1 = y2 ∧ (x2 = x1 + 1 ∨ x2 = x1 - 1))))
  (end_is_finish : (steps.last sorry) = finish)

-- Define the problem in terms of paths from (0,0) to (7,5)
def count_paths_0_0_to_7_5 : ℕ :=
by
  let start := (0, 0)
  let finish := (7, 5)
  exact count (Paths start finish)

theorem solve_particle_path_count : count_paths_0_0_to_7_5 = N :=
sorry

end solve_particle_path_count_l390_390818


namespace three_person_subcommittees_l390_390172

theorem three_person_subcommittees (n k : ℕ) (h_n : n = 8) (h_k : k = 3) : nat.choose n k = 56 := by
  rw [h_n, h_k]
  norm_num
  sorry

end three_person_subcommittees_l390_390172


namespace ω_2n_eq_ω_n_ω_8n_5_eq_ω_4n_3_ω_2n_minus_1_eq_n_l390_390334

def ω (n : ℕ) : ℕ :=
  n.bits.count (λ b => b)

theorem ω_2n_eq_ω_n (n : ℕ) : ω (2 * n) = ω n := by
  sorry

theorem ω_8n_5_eq_ω_4n_3 (n : ℕ) : ω (8 * n + 5) = ω (4 * n + 3) := by
  sorry

theorem ω_2n_minus_1_eq_n (n : ℕ) : ω (2^n - 1) = n := by
  sorry

end ω_2n_eq_ω_n_ω_8n_5_eq_ω_4n_3_ω_2n_minus_1_eq_n_l390_390334


namespace find_a_minus_b_l390_390191

variable {a b : ℤ}

theorem find_a_minus_b (h1 : a^2 = 9) (h2 : |b| = 4) (h3 : a > b) : a - b = 7 :=
  sorry

end find_a_minus_b_l390_390191


namespace four_digit_number_l390_390868

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000)

theorem four_digit_number (n : ℕ) (hn1 : 1000 ≤ n) (hn2 : n < 10000) (condition : n = 9 * (reverse_digits n)) :
  n = 9801 :=
by
  sorry

end four_digit_number_l390_390868


namespace colbert_materials_needed_l390_390077

def wooden_planks_needed (total_needed quarter_in_stock : ℕ) : ℕ :=
  let total_purchased := total_needed - quarter_in_stock / 4
  (total_purchased + 7) / 8 -- ceil division by 8

def iron_nails_needed (total_needed thirty_percent_provided : ℕ) : ℕ :=
  let total_purchased := total_needed - total_needed * thirty_percent_provided / 100
  (total_purchased + 24) / 25 -- ceil division by 25

def fabric_needed (total_needed third_provided : ℚ) : ℚ :=
  total_needed - total_needed / third_provided

def metal_brackets_needed (total_needed in_stock multiple : ℕ) : ℕ :=
  let total_purchased := total_needed - in_stock
  (total_purchased + multiple - 1) / multiple * multiple -- ceil to next multiple of 5

theorem colbert_materials_needed :
  wooden_planks_needed 250 62 = 24 ∧
  iron_nails_needed 500 30 = 14 ∧
  fabric_needed 10 3 = 6.67 ∧
  metal_brackets_needed 40 10 5 = 30 :=
by sorry

end colbert_materials_needed_l390_390077


namespace range_of_eccentricity_l390_390305

-- Definition of the ellipse and its properties
namespace EllipseProof

variables {a b : ℝ} (h : a > b ∧ b > 0)

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def upper_vertex : ℝ × ℝ := (0, b)

def is_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def condition_on_point (P : ℝ × ℝ) : Prop := dist P upper_vertex ≤ 2 * b

-- The eccentricity of the ellipse
def eccentricity : ℝ := real.sqrt (1 - (b^2 / a^2))

-- The required proof statement
theorem range_of_eccentricity :
  (∀ P, is_on_ellipse P → condition_on_point P) →
  eccentricity ≤ real.sqrt 2 / 2 := sorry

end EllipseProof

end range_of_eccentricity_l390_390305


namespace closed_curve_in_circle_l390_390238

noncomputable def lies_in_circle (K : Set Point) : Prop :=
  ∃ (C : Circle), ∀ (p ∈ K), p ∈ C

theorem closed_curve_in_circle (K : Set Point) (hK1 : ∀ x y ∈ K, dist x y < 1) (hK2 : closed_curve K) :
  lies_in_circle K :=
sorry

end closed_curve_in_circle_l390_390238


namespace mean_height_l390_390840

def heights : List ℝ := [58, 59, 60, 61, 62, 63, 65, 65, 68, 70, 71, 74, 76, 78, 79, 81, 83, 85, 86]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum / l.length)

theorem mean_height : mean heights = 70.74 := 
by
  sorry

end mean_height_l390_390840


namespace find_angle_C_find_triangle_area_l390_390636

theorem find_angle_C (A B C : ℝ) (a b c : ℝ) 
  (h1 : B = Real.pi / 4) 
  (h2 : Real.cos A - Real.cos (2 * A) = 0) 
  (h3 : B + C + A = Real.pi) :
  C = Real.pi / 12 :=
by
  sorry

theorem find_triangle_area (A B C : ℝ) (a b c : ℝ)
  (h1 : B = Real.pi / 4) 
  (h2 : Real.cos A - Real.cos (2 * A) = 0) 
  (h3 : b^2 + c^2 = a - b * c + 2) 
  (h4 : B + C + A = Real.pi) 
  (h5 : a^2 = b^2 + c^2 + b * c) :
  (1/2) * a * b * Real.sin C = 1 - Real.sqrt 3 / 3 :=
by
  sorry

end find_angle_C_find_triangle_area_l390_390636


namespace sum_first_5n_integers_eq_630_l390_390632

theorem sum_first_5n_integers_eq_630 {n : ℕ}
  (h : 3 * n * (3 * n + 1) / 2 = n * (n + 1) / 2 + 210) :
  5 * n * (5 * n + 1) / 2 = 630 :=
begin
  -- Proof is omitted. Use 'sorry' as a placeholder.
  sorry
end

end sum_first_5n_integers_eq_630_l390_390632


namespace eccentricity_range_l390_390313

variables {a b : ℝ}
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

theorem eccentricity_range (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ (x y : ℝ), ellipse x y → distance x y 0 b ≤ 2 * b) :
  0 < (sqrt (a^2 - b^2) / a) ∧ (sqrt (a^2 - b^2) / a) ≤ sqrt 2 / 2 :=
sorry

end eccentricity_range_l390_390313


namespace largest_real_part_of_z_plus_w_l390_390726

noncomputable def max_re_z_plus_w (z w : ℂ) (hz : |z| = 2) (hw : |w| = 2) (hzw : z * conj w + conj z * w = -1) : ℝ :=
  real.sqrt 7

theorem largest_real_part_of_z_plus_w (z w : ℂ) (hz : |z| = 2) (hw : |w| = 2) (hzw : z * conj w + conj z * w = -1) :
  re (z + w) ≤ max_re_z_plus_w z w hz hw hzw :=
sorry

end largest_real_part_of_z_plus_w_l390_390726


namespace probability_grunters_win_at_least_4_of_5_games_l390_390733

theorem probability_grunters_win_at_least_4_of_5_games :
  let p := 3/5 in
  let q := 2/5 in
  let n := 5 in
  let k := 4 in
  (nat.choose n k) * (p ^ k) * (q ^ (n - k)) + (p ^ n) = 1053 / 3125 := by
  -- We state the problem and leave the proof as an exercise
  sorry

end probability_grunters_win_at_least_4_of_5_games_l390_390733


namespace complement_of_A_in_U_l390_390555

noncomputable def U : Set ℝ := {x | (x - 2) / x ≤ 1}

noncomputable def A : Set ℝ := {x | 2 - x ≤ 1}

theorem complement_of_A_in_U :
  (U \ A) = {x | 0 < x ∧ x < 1} :=
by
  sorry

end complement_of_A_in_U_l390_390555


namespace exist_three_lines_intersecting_one_point_l390_390864

-- Define the conditions of the problem
def square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

def line_divides_square (l : ℝ × ℝ → Prop) (area_ratio : ℝ) : Prop :=
  ∀ (A B C D : ℝ × ℝ), square A B C D → 
    ∃ (A1 A2 : ℝ × ℝ), l A1 ∧ l A2 ∧ 
    let (left_area, right_area) := (2 * area_ratio / (1 + area_ratio), 1 / (1 + area_ratio)) in 
    (A1 = (0,0) ∧ A2 = (a, 0) ∧ left_area = 3/5 ∧ right_area = 2/5) ∨
    (A1 = (a,0) ∧ A2 = (b,1) ∧ right_area = 2/5 ∧ left_area = 3/5) 

-- The theorem to be proven
theorem exist_three_lines_intersecting_one_point 
  (lines : list (ℝ × ℝ → Prop)) (area_ratio : ℝ) :
  list.length lines = 9 →
  (∀ l, l ∈ lines → line_divides_square l area_ratio) →
  ∃ (p : ℝ × ℝ), ∃ (l1 l2 l3 : ℝ × ℝ → Prop), 
    l1 ∈ lines ∧ l2 ∈ lines ∧ l3 ∈ lines ∧ 
    p ∈ l1 ∧ p ∈ l2 ∧ p ∈ l3 :=
sorry

end exist_three_lines_intersecting_one_point_l390_390864


namespace tenth_term_arithmetic_sequence_l390_390432

theorem tenth_term_arithmetic_sequence :
  let a_1 := (1 : ℝ) / 2
  let a_2 := (5 : ℝ) / 6
  let d := a_2 - a_1
  (a_1 + 9 * d) = 7 / 2 := 
by
  sorry

end tenth_term_arithmetic_sequence_l390_390432


namespace shark_sightings_relationship_l390_390847

theorem shark_sightings_relationship (C D R : ℕ) (h₁ : C + D = 40) (h₂ : C = R - 8) (h₃ : C = 24) :
  R = 32 :=
by
  sorry

end shark_sightings_relationship_l390_390847


namespace range_of_eccentricity_l390_390301

-- Definition of the ellipse and its properties
namespace EllipseProof

variables {a b : ℝ} (h : a > b ∧ b > 0)

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def upper_vertex : ℝ × ℝ := (0, b)

def is_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def condition_on_point (P : ℝ × ℝ) : Prop := dist P upper_vertex ≤ 2 * b

-- The eccentricity of the ellipse
def eccentricity : ℝ := real.sqrt (1 - (b^2 / a^2))

-- The required proof statement
theorem range_of_eccentricity :
  (∀ P, is_on_ellipse P → condition_on_point P) →
  eccentricity ≤ real.sqrt 2 / 2 := sorry

end EllipseProof

end range_of_eccentricity_l390_390301


namespace common_ratio_of_geometric_sequence_l390_390387

noncomputable def a_n (a1 d : ℝ) (n : ℕ) : ℝ := a1 + (n - 1) * d

theorem common_ratio_of_geometric_sequence
  (a1 d : ℝ) (h1 : d ≠ 0)
  (h2 : (a_n a1 d 5) * (a_n a1 d 20) = (a_n a1 d 10) ^ 2) :
  (a_n a1 d 10) / (a_n a1 d 5) = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l390_390387


namespace problem_statement_l390_390322

noncomputable def f : ℝ → ℝ
| x :=
  if x < 2 then 2 * Real.exp (x - 1)
  else Real.log 3 (x ^ 2 - 1)

theorem problem_statement : f (f 2) = 2 := by
  sorry

end problem_statement_l390_390322


namespace maximize_angle_l390_390705

-- Assume the points A, B, and C lie on a line l
def points_on_line (A B C: ℝ × ℝ) : Prop :=
  A.2 = 0 ∧ B.2 = 0 ∧ C.2 = 0

-- Assume the distances AB = 3 and BC = 2
def distances (A B C: ℝ × ℝ) : Prop :=
  dist A B = 3 ∧ dist B C = 2

-- Assume point H is such that CH is perpendicular to line l
def perpendicular (C H: ℝ × ℝ) : Prop :=
  C.1 = H.1 ∧ C.2 = 0 ∧ H.2 ≠ 0

-- Main theorem statement
theorem maximize_angle {A B C H : ℝ × ℝ} 
  (hl: points_on_line A B C)
  (hd: distances A B C)
  (hp: perpendicular C H) :
  dist C H = real.sqrt 10 :=
by
  sorry

end maximize_angle_l390_390705


namespace final_price_correct_l390_390031

def calculate_final_price (original_price : ℝ) 
    (discount_rate : ℝ) 
    (tax_rate : ℝ) 
    (commission_rate : ℝ) : ℝ :=
let discounted_price := original_price * (1 - discount_rate),
    taxed_price := discounted_price * (1 + tax_rate),
    final_price := taxed_price * (1 + commission_rate)
in final_price

theorem final_price_correct :
  calculate_final_price 1200 0.15 0.12 0.05 = 1199.52 :=
by
  unfold calculate_final_price
  sorry

end final_price_correct_l390_390031


namespace number_of_students_in_each_language_l390_390210

/-- Definitions for the proportions of language speakers in each school and the overall district. -/
def school_A := (french english spanish mandarin german russian: ℕ) := 
  (french = 30, english = 45, spanish = 20, mandarin = 10, german = 25, russian = 5)
def school_B := (french english spanish mandarin german russian: ℕ) := 
  (french = 20, english = 40, spanish = 30, mandarin = 15, german = 10, russian = 10)
def school_C := (french english spanish mandarin german russian: ℕ) := 
  (french = 10, english = 50, spanish = 25, mandarin = 20, german = 15, russian = 5)
def district := (french english spanish mandarin german russian: ℕ) := 
  (french = 25, english = 45, spanish = 30, mandarin = 15, german = 20, russian = 8)

/-- Given the proportions in the district and the total number of students,
    compute the number of students speaking each language. -/
def total_students : ℕ := 1200

theorem number_of_students_in_each_language :
  let french_students := 0.25 * total_students,
      english_students := 0.45 * total_students,
      spanish_students := 0.30 * total_students,
      mandarin_students := 0.15 * total_students,
      german_students := 0.20 * total_students,
      russian_students := 0.08 * total_students
  in french_students = 300 ∧
     english_students = 540 ∧
     spanish_students = 360 ∧
     mandarin_students = 180 ∧
     german_students = 240 ∧
     russian_students = 96 :=
by sorry

end number_of_students_in_each_language_l390_390210


namespace range_of_eccentricity_l390_390302

-- Definition of the ellipse and its properties
namespace EllipseProof

variables {a b : ℝ} (h : a > b ∧ b > 0)

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def upper_vertex : ℝ × ℝ := (0, b)

def is_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def condition_on_point (P : ℝ × ℝ) : Prop := dist P upper_vertex ≤ 2 * b

-- The eccentricity of the ellipse
def eccentricity : ℝ := real.sqrt (1 - (b^2 / a^2))

-- The required proof statement
theorem range_of_eccentricity :
  (∀ P, is_on_ellipse P → condition_on_point P) →
  eccentricity ≤ real.sqrt 2 / 2 := sorry

end EllipseProof

end range_of_eccentricity_l390_390302


namespace first_division_percentage_l390_390979

theorem first_division_percentage (total_students : ℕ) (second_division_percentage just_passed_students : ℕ) 
  (h1 : total_students = 300) (h2 : second_division_percentage = 54) (h3 : just_passed_students = 60) : 
  (100 - second_division_percentage - ((just_passed_students * 100) / total_students)) = 26 :=
by
  sorry

end first_division_percentage_l390_390979


namespace sequence_not_geom_progression_l390_390518

def sequence (n : ℕ) : ℕ
| 0       => 0
| 1       => 2
| 2       => 3
| (n+3)   => sequence n + (sequence (n+1))^2

theorem sequence_not_geom_progression :
  ¬ ∃ r : ℝ, ∀ n : ℕ, sequence (n+1) = r * sequence n :=
by sorry

end sequence_not_geom_progression_l390_390518


namespace medicine_price_after_discount_l390_390819

theorem medicine_price_after_discount :
  ∀ (price : ℝ) (discount : ℝ), price = 120 → discount = 0.3 → 
  (price - price * discount) = 84 :=
by
  intros price discount h1 h2
  rw [h1, h2]
  sorry

end medicine_price_after_discount_l390_390819


namespace tan_alpha_implies_fraction_l390_390118

theorem tan_alpha_implies_fraction (α : ℝ) (h : Real.tan α = -3/2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.cos α - Real.sin α) = 1 / 5 := 
sorry

end tan_alpha_implies_fraction_l390_390118


namespace eccentricity_range_l390_390292

-- Ellipse definition
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Upper vertex B of the ellipse
def B (b : ℝ) : ℝ × ℝ := (0, b)

-- Distance PB condition
def distance_le_2b (a b x0 y0 : ℝ) (h : ellipse a b (and.intro (sorry) (sorry)) x0 y0) : Prop :=
  (x0 - 0)^2 + (y0 - b)^2 ≤ (2 * b)^2

-- Range of eccentricity
def eccentricity (a b c : ℝ) : ℝ :=
  c / a

-- Main theorem
theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  (forall (x0 y0 : ℝ), ellipse a b h x0 y0 → distance_le_2b a b x0 y0 h) →
  eccentricity a b (sqrt (a^2 - b^2)) ∈ set.Icc 0 (sqrt 2 / 2) :=
by
  sorry

end eccentricity_range_l390_390292


namespace sample_size_is_15_l390_390809

-- Conditions from the problem
def total_employees : ℕ := 750
def young_employees : ℕ := 350
def middle_aged_employees : ℕ := 250
def elderly_employees : ℕ := 150
def sample_young : ℕ := 7

-- Question: Prove that the sample size is 15 under the stratified sampling method
theorem sample_size_is_15 :
  ∃ x : ℕ, (young_employees / total_employees) = (sample_young / x) → x = 15 :=
begin
  sorry
end

end sample_size_is_15_l390_390809


namespace tank_capacity_l390_390483

/-- A water tank is 40% full and it contains 36 gallons less than when it is 10% empty. 
Prove that the full capacity of the tank is 72 gallons. -/
theorem tank_capacity (C : ℝ) : 0.4 * C + 36 = 0.9 * C → C = 72 :=
by
suffices h : 0.5 * C = 36
  {
    -- We can solve for C using this equality.
    rw [← div_eq_iff_mul_eq _ _ (by norm_num : (0.5 : ℝ) ≠ 0)],
    norm_num at h,
    exact h,
  }
  sorry

end tank_capacity_l390_390483


namespace polygon_sides_l390_390969

-- Define the given conditions
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def sum_exterior_angles : ℕ := 360

-- Define the theorem
theorem polygon_sides (n : ℕ) (h : sum_interior_angles n = 3 * sum_exterior_angles + 180) : n = 9 :=
sorry

end polygon_sides_l390_390969


namespace taco_truck_revenue_l390_390481

-- Conditions
def price_of_soft_taco : ℕ := 2
def price_of_hard_taco : ℕ := 5
def family_soft_tacos : ℕ := 3
def family_hard_tacos : ℕ := 4
def other_customers : ℕ := 10
def soft_tacos_per_other_customer : ℕ := 2

-- Calculation
def total_soft_tacos : ℕ := family_soft_tacos + other_customers * soft_tacos_per_other_customer
def revenue_from_soft_tacos : ℕ := total_soft_tacos * price_of_soft_taco
def revenue_from_hard_tacos : ℕ := family_hard_tacos * price_of_hard_taco
def total_revenue : ℕ := revenue_from_soft_tacos + revenue_from_hard_tacos

-- The proof problem
theorem taco_truck_revenue : total_revenue = 66 := 
by 
-- The proof should go here
sorry

end taco_truck_revenue_l390_390481


namespace parametric_equation_conversion_l390_390682

theorem parametric_equation_conversion (θ : ℝ) :
  ∃ x y : ℝ, (y = 2 * cos θ) ∧ ((x^2 + y^2 - 4 * x = 0) ↔ (x = 2 + 2 * sin θ ∧ y = 2 * cos θ)) :=
by
  sorry

end parametric_equation_conversion_l390_390682


namespace solve_quadratic_eq_l390_390722

theorem solve_quadratic_eq (x : ℝ) : x^2 + 2 * x - 1 = 0 ↔ (x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) :=
by
  sorry

end solve_quadratic_eq_l390_390722


namespace yolanda_walking_rate_l390_390791

-- Define the problem conditions
def distance_XY : ℝ := 10           -- Distance from X to Y is 10 miles
def bob_walked_distance : ℝ := 4    -- Bob walked 4 miles
def bob_rate : ℝ := 4               -- Bob's walking rate is 4 miles per hour
def bob_time : ℝ := bob_walked_distance / bob_rate -- Bob's time to walk 4 miles
def yolanda_time : ℝ := bob_time + 1 -- Yolanda's walking time when they met 
def yolanda_distance : ℝ := distance_XY - bob_walked_distance -- Yolanda's distance covered

-- Define the Yolanda's walking rate to be proved
def yolanda_rate := yolanda_distance / yolanda_time

-- Create the theorem 
theorem yolanda_walking_rate : yolanda_rate = 3 := by
  -- Proof steps go here
  sorry

end yolanda_walking_rate_l390_390791


namespace no_distributive_laws_hold_l390_390857

def operation (a b : ℝ) := (3 * (a + b)) / 2

theorem no_distributive_laws_hold (x y z : ℝ) :
  ¬ (operation x (y + z) = operation x y + operation x z) ∧
  ¬ (x + operation y z = operation (x + y) (x + z)) ∧
  ¬ (operation x (operation y z) = operation (operation x y) (operation x z)) := by
  sorry

end no_distributive_laws_hold_l390_390857


namespace sum_of_integers_m_satisfying_conditions_l390_390195

theorem sum_of_integers_m_satisfying_conditions :
  let neg_int (n : Int) := ∃ y : Int, y < 0 ∧ y = (n - 6) / 4
  let inequality_system (m : Int) (x : Int) := (5 * (x - m) ≤ 0) ∧ ((x + 2) / 3 - x / 2 > 1)
  (∑ m in Finset.filter (λ m, (∀ x : Int, inequality_system m x → x < -2) ∧ neg_int m) (Finset.range 10), m) = 0 := sorry

end sum_of_integers_m_satisfying_conditions_l390_390195


namespace quadratic_integer_roots_l390_390379

noncomputable def initial_equation : Polynomial ℤ := Polynomial.C 2 + Polynomial.X * Polynomial.C 3 + Polynomial.X^2

def increment (poly : Polynomial ℤ) := 
  poly.coeff 1 + 1 * Polynomial.X + 
  poly.coeff 0 + 1 * Polynomial.C 1 +
  Polynomial.X^2

def has_integral_roots (poly : Polynomial ℤ) : Prop := 
  ∃ (x y : ℤ), poly = Polynomial.C y + Polynomial.X * Polynomial.C x + Polynomial.X^2 ∧ 
  (∃ z w : ℤ, poly.eval z = 0 ∧ poly.eval w = 0)

theorem quadratic_integer_roots :
  let p := iterate 4 increment initial_equation
  has_integral_roots p := 
sorry

end quadratic_integer_roots_l390_390379


namespace AC_not_14_l390_390989

variable {A B C : Type}
variable (AB AC BC : ℝ)
variable [AB_eq_5 : AB = 5]
variable [BC_eq_8 : BC = 8]

theorem AC_not_14 (H1 : 8 - 5 < AC) (H2 : AC < 5 + 8) : AC ≠ 14 := by
  have h1 : 3 < AC := by
    exact H1
  have h2 : AC < 13 := by
    exact H2
  intro H
  rw H at *
  linarith

end AC_not_14_l390_390989


namespace probability_yellow_ball_l390_390980

-- Definitions of the conditions
def white_balls : ℕ := 2
def yellow_balls : ℕ := 3
def total_balls : ℕ := white_balls + yellow_balls

-- Theorem statement
theorem probability_yellow_ball : (yellow_balls : ℚ) / total_balls = 3 / 5 :=
by
  -- Using tactics to facilitate the proof
  simp [yellow_balls, total_balls]
  sorry

end probability_yellow_ball_l390_390980


namespace solve_diamondsuit_eq_l390_390803

axiom diamondsuit_property_1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : a ◇ (b ◇ c) = (a ◇ b) * c
axiom diamondsuit_property_2 (a : ℝ) (ha : a ≠ 0) : a ◇ a = 1

theorem solve_diamondsuit_eq (x : ℝ) (hx : x ≠ 0) : (2016 : ℝ) ◇ ((6 : ℝ) ◇ x) = (100 : ℝ) → x = 25 / 84 :=
by
  intros h
  sorry

end solve_diamondsuit_eq_l390_390803


namespace alpha_minus_beta_l390_390588

theorem alpha_minus_beta (
  α β : ℝ
) (h₀ : 0 < α ∧ α < π / 2)
  (h₁ : π / 2 < β ∧ β < π ∧ β ≠ 3 * π / 4)
  (h₂ : ∃ P : ℝ × ℝ, P = ⟨cos (2 * β), 1 + sin (3 * β) * cos β - cos (3 * β) * sin β⟩
         ∧ (∃ θ : ℝ, θ = α ∧ P ∈ terminal_side_of θ)) :
  α - β = -3 * π / 4 := 
sorry

end alpha_minus_beta_l390_390588


namespace no_integer_k_Pk_eq_8_l390_390678

theorem no_integer_k_Pk_eq_8 {P : ℤ[X]} (a b c d : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_values : P.eval a = 5 ∧ P.eval b = 5 ∧ P.eval c = 5 ∧ P.eval d = 5) :
  ∀ k : ℤ, P.eval k ≠ 8 := 
sorry

end no_integer_k_Pk_eq_8_l390_390678


namespace original_cost_of_tshirt_l390_390849

theorem original_cost_of_tshirt
  (backpack_cost : ℕ := 10)
  (cap_cost : ℕ := 5)
  (total_spent_after_discount : ℕ := 43)
  (discount : ℕ := 2)
  (tshirt_cost_before_discount : ℕ) :
  total_spent_after_discount + discount - (backpack_cost + cap_cost) = tshirt_cost_before_discount :=
by
  sorry

end original_cost_of_tshirt_l390_390849


namespace P_subset_Q_l390_390516

def P : Set ℝ := {m | -1 < m ∧ m < 0}

def Q : Set ℝ := {m | ∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0}

theorem P_subset_Q : P ⊂ Q :=
by
  sorry

end P_subset_Q_l390_390516


namespace original_number_is_84_l390_390757

theorem original_number_is_84 (x : ℕ) (h1 : x < 10) (h2 : 12 - x < 10) 
  (h3 : 10 * (12 - x) + x - (10 * x + (12 - x)) = 36) : 
  10 * (12 - x) + x = 84 :=
by {
  -- Proof setup
  have h4 : 0 ≤ x := Nat.zero_le x,
  have h5 : x < 10 := h1,
  have h6 : 0 ≤ 12 - x := Nat.zero_le (12 - x),
  have h7 : 12 - x < 10 := h2,
  -- Simplification of equation to find x
  have h8 : 120 - 12 - 18 * x = 36 := by linarith,
  have h9 : 108 - 18 * x = 36 := by assumption,
  have h10 : 108 - 36 = 18 * x := by linarith,
  have h11 : 72 = 18 * x := by linarith,
  have h12 : x = 4 := Nat.div_eq_of_eq_mul_right (by norm_num) (by linarith),
  -- Original number after finding x
  have original_number := 10 * (12 - x) + x,
  have h13 : 12 - x = 8 := by linarith,
  exact calc
    10 * 8 + 4 : by linarith
    ... = 84 : by norm_num
}

end original_number_is_84_l390_390757


namespace surface_area_of_inscribed_sphere_l390_390902

def Vec3 := (ℝ × ℝ × ℝ)

def dist (p q : Vec3) : ℝ :=
  real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

def edge_length (p q : Vec3) : ℝ :=
  (dist p q) / real.sqrt 3

def radius_inscribed_sphere (a : ℝ) : ℝ :=
  a / 2

def surface_area_sphere (r : ℝ) : ℝ :=
  4 * real.pi * r^2

noncomputable def surface_area_inscribed_sphere (M N : Vec3) : ℝ :=
  let a := edge_length M N
  let r := radius_inscribed_sphere a
  surface_area_sphere r

theorem surface_area_of_inscribed_sphere (M N: Vec3) (hM : M = (-1, 2, -1)) (hN : N = (3, -2, 3)) : 
  surface_area_inscribed_sphere M N = 16 * real.pi := 
  sorry

end surface_area_of_inscribed_sphere_l390_390902


namespace three_person_subcommittees_l390_390171

theorem three_person_subcommittees (n k : ℕ) (h_n : n = 8) (h_k : k = 3) : nat.choose n k = 56 := by
  rw [h_n, h_k]
  norm_num
  sorry

end three_person_subcommittees_l390_390171


namespace range_of_2a_plus_3b_l390_390894

theorem range_of_2a_plus_3b (a b : ℝ)
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1)
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l390_390894


namespace pa_pb_sum_eq_four_l390_390223

open Real

-- Define the parametric equations of the line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (1 - 1/2 * t, sqrt 3 / 2 * t)

-- Define the Cartesian equation of the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + (y - sqrt 3)^2 = 3

-- Define point P
def P : ℝ × ℝ := (1, 0)

-- Define the distances PA and PB
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- The theorem we want to prove
theorem pa_pb_sum_eq_four (A B : ℝ × ℝ)
  (hA : ∃ t : ℝ, line_l t = A ∧ circle_C A.1 A.2)
  (hB : ∃ t : ℝ, line_l t = B ∧ circle_C B.1 B.2) :
  distance P A + distance P B = 4 :=
sorry

end pa_pb_sum_eq_four_l390_390223


namespace quadratic_has_sqrt3_minus2_root_quadratic_has_conjugate_root_l390_390096

noncomputable def quadratic_with_rational_coeffs : (Polynomial ℚ) :=
  Polynomial.C 1 + Polynomial.C 4 * X + Polynomial.X^2

theorem quadratic_has_sqrt3_minus2_root :
  let α := sqrt 3 - 2
  in quadratic_with_rational_coeffs.eval α = 0 :=
sorry

theorem quadratic_has_conjugate_root :
  let β := -sqrt 3 - 2
  in quadratic_with_rational_coeffs.eval β = 0 :=
sorry

end quadratic_has_sqrt3_minus2_root_quadratic_has_conjugate_root_l390_390096


namespace number_of_subsets_of_B_l390_390936

def A := {0, 2, 3}

def B := {x : Nat | ∃ a b : Nat, a ≠ b ∧ a ∈ A ∧ b ∈ A ∧ x = a * b}

theorem number_of_subsets_of_B : B = {0, 6} ∧ (2 ^ B.toFinset.card = 4) := by
  sorry

end number_of_subsets_of_B_l390_390936


namespace product_of_sums_of_squares_l390_390708

-- Given conditions as definitions
def sum_of_squares (a b : ℤ) : ℤ := a^2 + b^2

-- Prove that the product of two sums of squares is also a sum of squares
theorem product_of_sums_of_squares (a b n k : ℤ) (K P : ℤ) (hK : K = sum_of_squares a b) (hP : P = sum_of_squares n k) :
    K * P = (a * n + b * k)^2 + (a * k - b * n)^2 := 
by
  sorry

end product_of_sums_of_squares_l390_390708


namespace BMC_eq_CND_iff_opposite_l390_390243

variables (A B C D M N : Type) [Point A B C D] [Intersection AB CD M] [Intersection AD BC N]

theorem BMC_eq_CND_iff_opposite :
  (angle B M C = angle C N D) ↔ 
  ((diametrically_opposite A C) ∨ (diametrically_opposite B D)) :=
sorry

end BMC_eq_CND_iff_opposite_l390_390243


namespace max_value_of_f_l390_390778

-- Define the function
def f (t : ℝ) : ℝ := ((3^t - 4 * t) * t) / (9^t)

-- State that the maximum value of the function is 1/16
theorem max_value_of_f : ∃ (t : ℝ), ∀ x : ℝ, f x ≤ f t ∧ f t = 1/16 :=
by
  sorry

end max_value_of_f_l390_390778


namespace function_expression_interval_of_increase_range_of_a_l390_390928

noncomputable def f (ω varphi : ℝ) (x : ℝ) : ℝ := (√2) * sin (2 * ω * x + varphi)

axiom ω_pos : ω > 0
axiom varphi_bound : |varphi| < π/2
constant T : ℝ
axiom T_val : T = 2 * π / ω
axiom distance_between_axes : ∀ x : ℝ, f ω varphi x = f ω varphi (x + π / ω)
axiom f_at_zero : f ω varphi 0 = 1

theorem function_expression :
  ω = 1 → varphi = π/4 → (∀ x, f ω varphi x = √2 * sin(2 * x + π / 4)) := sorry

theorem interval_of_increase :
  ω = 1 → varphi = π/4 → 
  (∀ x ∈ [0, π], f ω varphi x is increasing_in [0, π/8] ∪ [5 * π / 8, π]) := sorry

theorem range_of_a (a : ℝ) :
  ω = 1 → varphi = π/4 → 
  (∃ x1 x2 ∈ [0, 5 * π / 8], f ω varphi x1 + a = 0 ∧ f ω varphi x2 + a = 0 ∧ x1 ≠ x2) →
  (-√2 < a ∧ a ≤ 1) := sorry

end function_expression_interval_of_increase_range_of_a_l390_390928


namespace unique_x_value_l390_390517

theorem unique_x_value 
  (P1 P2 : Type)     -- Types representing polygons P1 and P2
  (n1 n2 : ℕ)        -- Number of sides of polygons P1 and P2
  (hx1 : ℝ)          -- Each angle of P1
  (hx2 : ℝ)          -- Each angle of P2
  (hP1 : hx1 = 180 - (360 / n1)) -- Angle condition for P1
  (hP2 : hx2 = 180 - (360 / n2)) -- Angle condition for P2
  (h_ratio : hx2 = 1.5 * hx1)    -- Ratio condition for angles
  : ∃! x : ℝ, hx1 = x := 
begin
  sorry
end

end unique_x_value_l390_390517


namespace ellipse_standard_eq_max_area_of_quadrilateral_function_values_monotonic_sum_x1_x2_gt_2e_l390_390059

section ellipse
-- Given conditions for ellipse
def ellipse_eq (x y : ℝ) : Prop := (x ^ 2) / 4 + (y ^ 2) / 3 = 1
def point_on_ellipse (P : ℝ × ℝ) : Prop := P = (1, 3/2)
def ellipse_condition (F1 F2 P : ℝ × ℝ) : Prop := 
  |F1 - P| + |F2 - P| = 4

-- Prove that the standard equation of the ellipse is as given
theorem ellipse_standard_eq : ellipse_eq = 
  λ x y, (x ^ 2) / 4 + (y ^ 2) / 3 = 1 :=
sorry

-- Condition for the quadrilateral ABCD formed by lines parallel through foci
def quadrilateral_max_area (F1 F2 : ℝ × ℝ) : ℝ := 6

-- Prove the maximum area of the quadrilateral ABCD is 6
theorem max_area_of_quadrilateral 
  (F1 F2 : ℝ × ℝ) (A B C D : ℝ × ℝ) 
  (area : ℝ): 
  quadrilateral_max_area F1 F2 = 6 :=
sorry
end ellipse

section function_and_tangent
-- Given conditions for the function f(x)
def f (x : ℝ) (a b : ℝ) := (a / x) * (Real.log x) + b
def tangent_at_one (a b : ℝ) : Prop := 
  f 1 a b = 0 ∧ 1 = a

-- Prove the values of a, b and the monotonic intervals
theorem function_values_monotonic : 
  ∃ a b, a = 1 ∧ b = 0 ∧ 
  ((∀ x : ℝ, 0 < x < Real.exp 1 → (derivative (f x a b)) > 0) ∧ 
   (∀ x : ℝ, x > Real.exp 1 → (derivative (f x a b)) < 0)) :=
sorry

-- Prove the sum x_1 + x_2 > 2e when f(x1) = f(x2) and x1 ≠ x2
theorem sum_x1_x2_gt_2e
  (x1 x2 : ℝ) (h : f x1 1 0 = f x2 1 0) (hne : x1 ≠ x2) : 
  x1 + x2 > 2 * Real.exp 1 :=
sorry
end function_and_tangent

end ellipse_standard_eq_max_area_of_quadrilateral_function_values_monotonic_sum_x1_x2_gt_2e_l390_390059


namespace hare_and_tortoise_meet_60m_l390_390745

/-- Problem definition for where the hare and the tortoise meet in a 100-meter race. -/
def hare_and_tortoise_meeting_point (v_h v_t : ℝ) (d_tortoise_behind hare_time tortoise_time : ℝ) : ℝ :=
    if hare_time = 100 / v_h ∧ tortoise_time = 25 / v_t ∧ d_tortoise_behind = 75 then (100 / v_h) * (60 / v_t) else 0

theorem hare_and_tortoise_meet_60m (v_h v_t : ℝ) :
    v_h / v_t = 4 → d_tortoise_behind = 75 → (hare_time = 100 / v_h) ∧ (tortoise_time = 25 / v_t) →
    hare_and_tortoise_meeting_point v_h v_t d_tortoise_behind hare_time tortoise_time = 60 :=
by
    sorry

end hare_and_tortoise_meet_60m_l390_390745


namespace problem1_problem2_l390_390608

universe u
variables {α : Type u}

def A : set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def B (m : ℝ) : set ℝ := {x | (x - m + 2) * (x - m - 2) ≤ 0}

theorem problem1 (m : ℝ) :
  (A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3}) ↔ m = 2 :=
by sorry

theorem problem2 (m : ℝ) :
  (A ⊆ {y | ¬(y ∈ B m)}) ↔ m ∈ Iio (-3) ∪ Ioi 5 :=
by sorry

end problem1_problem2_l390_390608


namespace smallest_prime_perf_sqr_minus_eight_l390_390427

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_perf_sqr_minus_eight (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 8

theorem smallest_prime_perf_sqr_minus_eight :
  ∃ (n : ℕ), is_prime n ∧ is_perf_sqr_minus_eight n ∧ (∀ m : ℕ, is_prime m ∧ is_perf_sqr_minus_eight m → n ≤ m) :=
sorry

end smallest_prime_perf_sqr_minus_eight_l390_390427


namespace smallest_multiple_of_2019_of_form_abcabcabc_l390_390553

def is_digit (n : ℕ) : Prop := n < 10

theorem smallest_multiple_of_2019_of_form_abcabcabc
    (a b c : ℕ)
    (h_a : is_digit a)
    (h_b : is_digit b)
    (h_c : is_digit c)
    (k : ℕ)
    (form : Nat)
    (rep: ℕ) : 
  (form = (a * 100 + b * 10 + c) * rep) →
  (∃ n : ℕ, form = 2019 * n) →
  form >= 673673673 :=
sorry

end smallest_multiple_of_2019_of_form_abcabcabc_l390_390553


namespace word_count_proof_l390_390614

def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def consonants : List Char := ['B', 'C', 'D', 'F']
def vowels : List Char := ['A', 'E']

def count_unrestricted_words : ℕ := 6 ^ 5
def count_all_vowel_words : ℕ := 2 ^ 5
def count_one_consonant_words : ℕ := 5 * 4 * (2 ^ 4)
def count_fewer_than_two_consonant_words : ℕ := count_all_vowel_words + count_one_consonant_words

def count_words_with_at_least_two_consonants : ℕ :=
  count_unrestricted_words - count_fewer_than_two_consonant_words

theorem word_count_proof :
  count_words_with_at_least_two_consonants = 7424 := 
by
  -- Proof will be provided here. For now we skip it.
  sorry

end word_count_proof_l390_390614


namespace merchant_mark_price_l390_390034

theorem merchant_mark_price (L : ℝ) (C : ℝ) (M : ℝ) (S : ℝ)
  (hC: C = 0.7 * L)
  (hC_profit: C = 0.7 * S)
  (hS: S = 0.8 * M) :
  M = 1.25 * L :=
by
  have h1 : 0.7 * L = 0.7 * S, from hC.trans hC_profit.symm
  have h2 : L = S, from (mul_right_inj' (by norm_num : (0.7:ℝ) ≠ 0)).mp h1
  have h3 : S = 0.8 * M, from hS
  have h4 : L = 0.8 * M, from h2.trans h3
  have h5 : M = (L / 0.8), from eq_div_of_mul_eq (by norm_num : 0.8 ≠ 0) h4.symm
  show M = 1.25 * L, by rw [←div_eq_inv_mul, ←mul_assoc, inv_eq_one_div, div_div, ←div_eq_mul_inv, div_self 0.8, one_mul, mul_eq_mul_right_iff, or_iff_left (by norm_num)], from h5

end merchant_mark_price_l390_390034


namespace period_one_l390_390368

noncomputable def floor_function (x : ℝ) : ℤ := Int.floor x

def f (x : ℝ) : ℝ := x - floor_function x

theorem period_one (x : ℝ) : f (x + 1) = f x := 
  sorry

end period_one_l390_390368


namespace isosceles_right_triangles_exist_l390_390156

noncomputable def hyperbola (x y m : ℝ) : Prop := 
  x^2 + 2*x - m*y^2 = 0

noncomputable def is_isosceles_right_triangle (A B : ℝ × ℝ) (O : ℝ × ℝ) : Prop :=
  let ox := O.1
  let oy := O.2
  let ax := A.1
  let ay := A.2
  let bx := B.1
  let by := B.2
  (ox = 0 ∧ oy = 0) ∧
  ((ax^2 + ay^2) = (bx^2 + by^2)) ∧
   ((ax - ox) * (bx - ox) + (ay - oy) * (by - oy) = 0)

theorem isosceles_right_triangles_exist 
  (m : ℝ) (hm : 0 < m) 
  (O : ℝ × ℝ) (hO : O = (0, 0)) :
  (0 < m ∧ m < 1 → ∃ A B : ℝ × ℝ, 
    hyperbola A.1 A.2 m ∧ hyperbola B.1 B.2 m ∧ is_isosceles_right_triangle A B O) ∧
  (m = 1 → ¬ (∃ A B : ℝ × ℝ, 
    hyperbola A.1 A.2 m ∧ hyperbola B.1 B.2 m ∧ is_isosceles_right_triangle A B O)) ∧
  (m > 1 → ∃ A B : ℝ × ℝ, 
    hyperbola A.1 A.2 m ∧ hyperbola B.1 B.2 m ∧ is_isosceles_right_triangle A B O) := 
sorry

end isosceles_right_triangles_exist_l390_390156


namespace smallest_c_minus_a_l390_390764

-- Definitions for the problem conditions.
variables {a b c : ℕ}
def is_positive (n : ℕ) : Prop := n > 0

def factorial (n : ℕ) : ℕ :=
match n with
| 0     => 1
| (n+1) => (n+1) * factorial n
end

def fact_9 := factorial 9

-- The Lean theorem statement for the proof problem.
theorem smallest_c_minus_a (a b c : ℕ) (h_pos_a : is_positive a) (h_pos_b : is_positive b) (h_pos_c : is_positive c)
  (h_abc : a < b ∧ b < c) (h_prod : a * b * c = fact_9) :
  c - a = 262 :=
sorry

end smallest_c_minus_a_l390_390764


namespace necessary_but_not_sufficient_sufficient_condition_necessary_condition_x_square_necessary_not_sufficient_l390_390901

theorem necessary_but_not_sufficient (x : ℝ) (h₁ : x^2 = x + 2) : (x = 2) ∨ (x = -1) :=
by {
  -- Assume x satisfies the equation x^2 = x + 2
  have : x^2 - x - 2 = 0,
  rw [h₁],
  by {
    rw [add_sub_cancel', eq_sub_related],
    linarith,
  }
  -- Solve the quadratic equation
  have h₁ : x = 2 ∨ x = -1,
  { exact this }
}

theorem sufficient_condition (x : ℝ) : (x = 2) → (x = sqrt(x + 2)) :=
by {
  intro h,
  rw [h],
  exact sqrt_eq_of_sq (by linarith)
}

-- To state that x^2 = x + 2 is a necessary condition for x = sqrt(x + 2)
theorem necessary_condition (x : ℝ) (h : x = sqrt(x + 2)) : x^2 = x + 2 :=
by {
  rw [h],
  exact pow_two_eq_add_eq_two
}

-- Combining them, we assert the final relationship:
theorem x_square_necessary_not_sufficient (x : ℝ) : 
  (x^2 = x + 2) → (x = sqrt(x + 2)) ∧ ¬((x = sqrt(x + 2)) → (x^2 = x + 2)) :=
begin
  split,
  { -- Prove the necessary condition direction
    intro h,
    rw necessary_condition,
    exact h,
  },
  { -- Prove not sufficient condition
    intro h,
    exact λ h, h.not_implies_self (by rw [h, pow_two_eq_add_eq_two])
  }
  sorry
end

end necessary_but_not_sufficient_sufficient_condition_necessary_condition_x_square_necessary_not_sufficient_l390_390901


namespace problem1_problem2_l390_390929

open Real

noncomputable def f (ω x : ℝ) : ℝ := sin (ω * x + π/4)

theorem problem1 (ω : ℝ) (hω : ω > 0) : f ω 0 = sqrt 2 / 2 :=
by
  sorry

theorem problem2 (ω : ℝ) (hω : ω > 0) (h_period : (2 * π / ω) = π) :
  ∀ x, 0 ≤ x ∧ x ≤ π / 2 → f ω x ≤ 1 :=
by
  have hω_eq_2 : ω = 2 := by
    have := congr_arg (λ x, ω * x) h_period
    simp [this] at h_period
    exact h_period

  intro x hx
  simp [f]
  sorry

end problem1_problem2_l390_390929


namespace circle_area_through_AOC_l390_390862

-- Define the angles named in the problem
def α : Real := 60
def β : Real := 120

-- Define the length of the side of the diamond
def s : Real := 8

-- Define the coordinates of O (the incenter)
def O : (Real × Real) := (0, 0)

-- Function to convert degrees to radians
def toRadians (deg : Real) : Real :=
  deg * π / 180

-- Define the diagonals of the diamond based on the given conditions
def d1 : Real := 2 * s * Real.sin (toRadians 30)  -- α = 60°, β = 120°
def d2 : Real := 2 * s * Real.sin (toRadians 60)

-- Distance from O to a vertex (used as radius of the circle through A, O, and C)
def radius : Real := d2 / 2

-- Area of the circle
def area_of_circle : Real := π * (radius ^ 2)

-- Statement to prove
theorem circle_area_through_AOC :
  area_of_circle = 48 * π :=
sorry

end circle_area_through_AOC_l390_390862


namespace acute_angle_vector_lambda_range_l390_390162

theorem acute_angle_vector_lambda_range (λ : ℝ) 
  (a := (λ, 2 * λ))
  (b := (3 * λ, 2))
  (acute_angle : (λ * 3 * λ + 2 * λ * 2) > 0) : 
  (λ < -4 / 3 ∨ λ > 0 ∧ λ ≠ 1 / 3) :=
by sorry

end acute_angle_vector_lambda_range_l390_390162


namespace three_person_subcommittees_l390_390179

theorem three_person_subcommittees (n k : ℕ) (h1 : n = 8) (h2 : k = 3) : nat.choose n k = 56 := 
by
  rw [h1, h2]
  norm_num
  sorry

end three_person_subcommittees_l390_390179


namespace sum_of_primes_between_1_and_100_l390_390878

open Nat

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

def satisfies_conditions (p : ℕ) : Prop :=
  p % 6 = 1 ∧ p % 7 = 6 ∧ p % 5 = 2

theorem sum_of_primes_between_1_and_100 :
  ∑ p in (List.range' 1 100).filter (λ p => is_prime p ∧ satisfies_conditions p), id p = 97 := 
by
  sorry

end sum_of_primes_between_1_and_100_l390_390878


namespace cos_pi_minus_alpha_proof_l390_390648

-- Define the initial conditions: angle α and point P
def α : ℝ := arbitrary ℝ
def P : ℝ × ℝ := (-1, 2)

-- Noncomputable to define functions involving real number calculations
noncomputable def hypotenuse : ℝ := real.sqrt ((P.1)^2 + (P.2)^2)
noncomputable def cos_alpha : ℝ := P.1 / hypotenuse
noncomputable def cos_pi_minus_alpha : ℝ := -cos_alpha

-- The theorem to prove
theorem cos_pi_minus_alpha_proof :
  cos_pi_minus_alpha = real.sqrt 5 / 5 :=
by sorry

end cos_pi_minus_alpha_proof_l390_390648


namespace audit_options_correct_l390_390835

-- Define the initial number of ORs and GTUs
def initial_ORs : ℕ := 13
def initial_GTUs : ℕ := 15

-- Define the number of ORs and GTUs visited in the first week
def visited_ORs : ℕ := 2
def visited_GTUs : ℕ := 3

-- Calculate the remaining ORs and GTUs
def remaining_ORs : ℕ := initial_ORs - visited_ORs
def remaining_GTUs : ℕ := initial_GTUs - visited_GTUs

-- Calculate the number of ways to choose 2 ORs from remaining ORs
def choose_ORs : ℕ := Nat.choose remaining_ORs 2

-- Calculate the number of ways to choose 3 GTUs from remaining GTUs
def choose_GTUs : ℕ := Nat.choose remaining_GTUs 3

-- The final function to calculate the number of options
def number_of_options : ℕ := choose_ORs * choose_GTUs

-- The proof statement asserting the number of options is 12100
theorem audit_options_correct : number_of_options = 12100 := by
    sorry -- Proof will be filled in here

end audit_options_correct_l390_390835


namespace max_sum_pyramid_on_hexagonal_face_l390_390356

structure hexagonal_prism :=
(faces_initial : ℕ)
(vertices_initial : ℕ)
(edges_initial : ℕ)

structure pyramid_added :=
(faces_total : ℕ)
(vertices_total : ℕ)
(edges_total : ℕ)
(total_sum : ℕ)

theorem max_sum_pyramid_on_hexagonal_face (h : hexagonal_prism) :
  (h = ⟨8, 12, 18⟩) →
  ∃ p : pyramid_added, 
    p = ⟨13, 13, 24, 50⟩ :=
by
  sorry

end max_sum_pyramid_on_hexagonal_face_l390_390356


namespace chef_made_10_cakes_l390_390736

-- Definitions based on the conditions
def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

-- Calculated values based on the definitions
def eggs_for_cakes : ℕ := total_eggs - eggs_in_fridge
def number_of_cakes : ℕ := eggs_for_cakes / eggs_per_cake

-- Theorem to prove
theorem chef_made_10_cakes : number_of_cakes = 10 := by
  sorry

end chef_made_10_cakes_l390_390736


namespace polynomial_determination_sum_of_polynomial_l390_390580

def f (n : ℕ) : ℕ := 2*n^3 - 5*n^2 + 3*n + 1

theorem polynomial_determination : 
  (∀ n, n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 → f n = 1 ∧ f 1 = 1 ∧ f 2 = 3 ∧ f 3 = 19) := sorry

theorem sum_of_polynomial (n : ℕ) : 
  ∑ k in Finset.range (n + 1), f k = (n + 1) * (3 * n^3 - 7 * n^2 + 4 * n + 6) / 6 := sorry

end polynomial_determination_sum_of_polynomial_l390_390580


namespace cos_pi_minus_alpha_correct_l390_390650

noncomputable def cos_pi_minus_alpha (α : ℝ) (P : ℝ × ℝ) : ℝ :=
  let x := P.1
  let y := P.2
  let h := Real.sqrt (x^2 + y^2)
  let cos_alpha := x / h
  let cos_pi_minus_alpha := -cos_alpha
  cos_pi_minus_alpha

theorem cos_pi_minus_alpha_correct :
  cos_pi_minus_alpha α (-1, 2) = Real.sqrt 5 / 5 :=
by
  sorry

end cos_pi_minus_alpha_correct_l390_390650


namespace find_greatest_natural_number_l390_390541

-- Definitions for terms used in the conditions

def sum_of_squares (m : ℕ) : ℕ :=
  (m * (m + 1) * (2 * m + 1)) / 6

def is_perfect_square (a : ℕ) : Prop :=
  ∃ b : ℕ, b * b = a

-- Conditions defined in Lean terms
def condition1 (n : ℕ) : Prop := n ≤ 2010

def condition2 (n : ℕ) : Prop := 
  let sum1 := sum_of_squares n
  let sum2 := sum_of_squares (2 * n) - sum_of_squares n
  is_perfect_square (sum1 * sum2)

-- Main theorem statement
theorem find_greatest_natural_number : ∃ n, n ≤ 2010 ∧ condition2 n ∧ ∀ m, m ≤ 2010 ∧ condition2 m → m ≤ n := 
by 
  sorry

end find_greatest_natural_number_l390_390541


namespace cos_pi_minus_alpha_correct_l390_390649

noncomputable def cos_pi_minus_alpha (α : ℝ) (P : ℝ × ℝ) : ℝ :=
  let x := P.1
  let y := P.2
  let h := Real.sqrt (x^2 + y^2)
  let cos_alpha := x / h
  let cos_pi_minus_alpha := -cos_alpha
  cos_pi_minus_alpha

theorem cos_pi_minus_alpha_correct :
  cos_pi_minus_alpha α (-1, 2) = Real.sqrt 5 / 5 :=
by
  sorry

end cos_pi_minus_alpha_correct_l390_390649


namespace probability_red_card_top_l390_390827

def num_red_cards : ℕ := 26
def total_cards : ℕ := 52
def prob_red_card_top : ℚ := num_red_cards / total_cards

theorem probability_red_card_top : prob_red_card_top = (1 / 2) := by
  sorry

end probability_red_card_top_l390_390827


namespace sum_of_abs_roots_l390_390881

-- Define the polynomial
def poly := polynomial(C: ℂ)([a=1, b=-6, c=9, d=24, e=-36])

-- The main theorem to prove
theorem sum_of_abs_roots {α : Type*} [field α] (hx : ℝ) (hijk : polynomial(a: 1, b:-6, c:9, d:24, e:-36) (has_roots: list(√(3), -√(3), i√(3), -i√(3)))) :
  (∑ root in polynomial.roots(poly.to_finsupp), abs root) = 4 * √3 :=
sorry

end sum_of_abs_roots_l390_390881


namespace smallest_prime_perf_sqr_minus_eight_l390_390428

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_perf_sqr_minus_eight (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 8

theorem smallest_prime_perf_sqr_minus_eight :
  ∃ (n : ℕ), is_prime n ∧ is_perf_sqr_minus_eight n ∧ (∀ m : ℕ, is_prime m ∧ is_perf_sqr_minus_eight m → n ≤ m) :=
sorry

end smallest_prime_perf_sqr_minus_eight_l390_390428


namespace ratio_of_distances_l390_390450

-- defining the speeds and times of car A and car B
constant speed_A : ℝ := 50
constant time_A : ℝ := 8
constant speed_B : ℝ := 25
constant time_B : ℝ := 4

-- defining distances traveled by car A and car B
def distance_A : ℝ := speed_A * time_A
def distance_B : ℝ := speed_B * time_B

-- stating the theorem about the ratio of distances
theorem ratio_of_distances : (distance_A / distance_B) = 4 :=
by
  sorry

end ratio_of_distances_l390_390450


namespace voting_participation_l390_390216

noncomputable def total_voters (x : ℝ) : ℝ := x + 80

theorem voting_participation:
  let x := 180 in
  let additional_votes := 80 in
  let final_total := total_voters x in
  let votes_for_oct_29 := 0.45 * final_total in
  x + additional_votes = 260 :=
by
  -- Conditions taken from the problem
  have h1 : votes_for_oct_29 = 0.65 * x := by
    sorry
  -- We know the final total must account for the additional votes
  have h2 : final_total = x + additional_votes := by
    sorry
  -- Solving for x given the conditions
  have h3 : 0.65 * x = 0.45 * (x + additional_votes) := by
    sorry
  -- Concluding the total number of participants is 260
  sorry

end voting_participation_l390_390216


namespace lim_n_inf_sum_n_inf_l390_390001
noncomputable def zeta (s : ℕ) : ℚ := 1 + (1/2^s : ℚ) + (1/3^s : ℚ) + (1/4^s : ℚ) + (1/5^s : ℚ) + (1/6^s : ℚ) + (1/7^s : ℚ) + (1/8^s : ℚ) + (1/9^s : ℚ) + (1/10^s : ℚ)

theorem lim_n_inf (n : ℕ) : 
  tendsto (λ n, (n + 1/4 - ∑ k in range n, zeta (2*k+1))) at_top (nhds (0 : ℚ)) :=
sorry

theorem sum_n_inf : 
  ∑' n, (n + 1/4 - ∑ k in range n, zeta (2*k+1)) = 0 :=
sorry

end lim_n_inf_sum_n_inf_l390_390001


namespace range_of_2a_plus_3b_l390_390892

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b ∧ a + b ≤ 1) (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l390_390892


namespace total_servings_l390_390411

/-- The first jar contains 24 2/3 tablespoons of peanut butter. -/
def first_jar_pb : ℚ := 74 / 3

/-- The second jar contains 19 1/2 tablespoons of peanut butter. -/
def second_jar_pb : ℚ := 39 / 2

/-- One serving size is 3 tablespoons. -/
def serving_size : ℚ := 3

/-- The total servings of peanut butter in both jars is 14 13/18 servings. -/
theorem total_servings : (first_jar_pb + second_jar_pb) / serving_size = 14 + 13 / 18 :=
by
  sorry

end total_servings_l390_390411


namespace least_sum_of_exponents_l390_390094

theorem least_sum_of_exponents :
  ∃ s : Finset ℕ, (1000 = ∑ i in s, 2 ^ i) ∧ (s.card ≥ 2) ∧ (s.sum id = 38) := 
sorry

end least_sum_of_exponents_l390_390094


namespace section_areas_l390_390765

variable (S A B C : Point) (ABC : Triangle S A B) (α β : Plane)

-- Conditions
axiom reg_triangular_pyramid (SABC : regular_triangular_pyramid S A B C)
axiom plane_alpha_perpendicular_SA (h1 : plane α forms 30∘ with plane ABC ∧ plane α is perpendicular to edge SA)
axiom planes_intersect_with_common_side (h2: common_side_length α β ABC = 1)

-- Question
theorem section_areas (h1 h2) : 
  let area_α_section := area_of_section SABC α
  let area_β_section := area_of_section SABC β
  area_α_section = 3/8 ∧ area_β_section = 54/49 :=
sorry

end section_areas_l390_390765


namespace comparing_exponents_l390_390680

theorem comparing_exponents {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b :=
sorry

end comparing_exponents_l390_390680


namespace area_of_triangle_ABC_l390_390635

def a := 2
def b := 2 * Real.sqrt 3
def B := Real.pi / 3

theorem area_of_triangle_ABC :
  let A := Real.arcsin (a * Real.sin B / b)
  let C := Real.pi - A - B
  b * Real.sin A = a * Real.sin B ∧
  A + B + C = Real.pi ∧
  ∠C = Real.pi / 2 ∧
  A < B ∧
  ∃ area, area = (1 / 2) * a * b ∧ area = 2 * Real.sqrt 3 :=
sorry

end area_of_triangle_ABC_l390_390635


namespace rectangular_garden_width_l390_390007

-- Define the problem conditions as Lean definitions
def rectangular_garden_length (w : ℝ) : ℝ := 3 * w
def rectangular_garden_area (w : ℝ) : ℝ := rectangular_garden_length w * w

-- This is the theorem we want to prove
theorem rectangular_garden_width : ∃ w : ℝ, rectangular_garden_area w = 432 ∧ w = 12 :=
by
  sorry

end rectangular_garden_width_l390_390007


namespace average_series_eq_l390_390506

theorem average_series_eq (z : ℝ) :
  let series := [4 * z, 6 * z, 9 * z, 13.5 * z, 20.25 * z]
  let avg := list.sum series / series.length
  avg = 10.55 * z :=
by
  sorry

end average_series_eq_l390_390506


namespace largest_n_consecutive_sum_l390_390545

theorem largest_n_consecutive_sum (a n : ℕ) (h : n > 0) : 
  (∃ a : ℕ, ∑ k in finset.range n, (a + k) = 2010) ↔ n = 60 := 
by {
  sorry
}

end largest_n_consecutive_sum_l390_390545


namespace snowman_volume_l390_390529

-- Define the radius of the spheres
def r₁ : ℝ := 4
def r₂ : ℝ := 6
def r₃ : ℝ := 8

-- Define the volume formula for a sphere
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the volumes of the individual snowballs
def V₁ : ℝ := volume_sphere r₁
def V₂ : ℝ := volume_sphere r₂
def V₃ : ℝ := volume_sphere r₃

-- State the total volume of the snowman
def V_total : ℝ := V₁ + V₂ + V₃

-- Prove that the total volume equals 1056π cubic inches
theorem snowman_volume : V_total = 1056 * Real.pi := by
  -- Placeholder for the actual proof
  sorry

end snowman_volume_l390_390529


namespace smallest_prime_less_than_perf_square_l390_390422

-- Define a predicate for being a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

-- The main goal
theorem smallest_prime_less_than_perf_square : ∃ n : ℕ, is_prime n ∧ ∃ m : ℕ, n = m^2 - 8 ∧ (∀ k : ℕ, is_prime k ∧ ∃ l : ℕ, k = l^2 - 8 → k ≥ n) :=
begin
  use 17,
  split,
  -- Proof that 17 is a prime number
  {
    unfold is_prime,
    split,
    { exact dec_trivial },
    { intros d hd,
      have h_d : d = 1 ∨ d = 17,
      { cases d,
        { exfalso, linarith, },
        { cases d,
          { left, refl, },
          { right, linarith [Nat.Prime.not_dvd_one 17 hd], }, }, },
      exact h_d, },
  },
  -- Proof that 17 is 8 less than a perfect square and the smallest such prime
  {
    use 5,
    split,
    { refl, },
    { intros k hk,
      cases hk with hk_prime hk_cond,
      cases hk_cond with l hl,
      rw hl,
      have : l ≥ 5,
      { intros,
        linarith, },
      exact this, },
  }
end

end smallest_prime_less_than_perf_square_l390_390422


namespace susan_average_speed_l390_390730

noncomputable def average_speed_trip (d1 d2 : ℝ) (v1 v2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let time1 := d1 / v1
  let time2 := d2 / v2
  let total_time := time1 + time2
  total_distance / total_time

theorem susan_average_speed :
  average_speed_trip 60 30 30 60 = 36 := 
by
  -- The proof can be filled in here
  sorry

end susan_average_speed_l390_390730


namespace PQ_passes_through_fixed_point_independent_of_X_l390_390122

-- Definitions representing the given conditions
variables {A B C X P Q : Point}
variables {triangle_ABC : Triangle}
variables [IsMovingPoint X (Line.extended (BC triangle_ABC))]

def incenter_A (Δ : Triangle) : Point := sorry
def incircle (Δ : Triangle) : Circle := sorry

-- PQ_passes_through_fixed_point is the fixed point which PQ passes through independent of X
def PQ_passes_through_fixed_point (Δ : Triangle) : Point := sorry

-- The main theorem statement
theorem PQ_passes_through_fixed_point_independent_of_X
  (h₁ : Incircle (triangle_ABC.ABX triangle_ABC X).intersects (triangle_ABC.ACX triangle_ABC X) = {P, Q})
  : ∀ X, PQ (Point P) (Point Q) (Line X) passes through (PQ_passes_through_fixed_point (triangle_ABC)) :=
begin
  sorry,
end

end PQ_passes_through_fixed_point_independent_of_X_l390_390122


namespace tan_X_in_triangle_XYZ_l390_390990

theorem tan_X_in_triangle_XYZ :
  ∀ {X Y Z : Type}
    [linear_ordered_field X] [linear_ordered_field Y] [linear_ordered_field Z]
    (XY YZ XZ : X)
    (h1 : XY = 5)
    (h2 : XZ = 2 * real.sqrt 7)
    (h3 : XY^2 + YZ^2 = XZ^2),
  real.tan (real.arctan (YZ / XY)) = real.sqrt 3 / 5 :=
by {
    intro X Y Z,
    intro XY YZ XZ,
    intro h1 h2 h3,
    sorry
}

end tan_X_in_triangle_XYZ_l390_390990


namespace findMathematicsMarks_l390_390083

-- Define the constants for the marks in each subject
def English : ℕ := 76
def Physics : ℕ := 82
def Chemistry : ℕ := 67
def Biology : ℕ := 85

-- Define the total number of subjects
def numSubjects : ℕ := 5

-- Define the average marks
def averageMarks : ℕ := 75

-- Calculate Mathematics marks
theorem findMathematicsMarks :
  let Mathematics := 65 in
  (averageMarks * numSubjects = English + Mathematics + Physics + Chemistry + Biology) :=
by
  sorry

end findMathematicsMarks_l390_390083


namespace ratio_of_carrie_to_barney_l390_390072

-- Defining the conditions
variables (C B J : ℕ)
variable (situps_performed : ℕ)

-- Given conditions
def condition1 := B = 45
def condition2 := J = C + 5
def condition3 := situps_performed = 45 * 1 + C * 2 + J * 3
def condition4 := situps_performed = 510

-- The main statement
theorem ratio_of_carrie_to_barney
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4)
  : C = 90 → C / B = 2 :=
by
  intro hC
  rw [h1] at h3
  rw [hC, h2] at h3
  linarith

end ratio_of_carrie_to_barney_l390_390072


namespace cloth_sales_value_l390_390793

theorem cloth_sales_value (commission_rate : ℝ) (commission : ℝ) (total_sales : ℝ) 
  (h1: commission_rate = 2.5)
  (h2: commission = 18)
  (h3: total_sales = commission / (commission_rate / 100)):
  total_sales = 720 := by
  sorry

end cloth_sales_value_l390_390793


namespace smallest_prime_perf_sqr_minus_eight_l390_390429

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_perf_sqr_minus_eight (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 8

theorem smallest_prime_perf_sqr_minus_eight :
  ∃ (n : ℕ), is_prime n ∧ is_perf_sqr_minus_eight n ∧ (∀ m : ℕ, is_prime m ∧ is_perf_sqr_minus_eight m → n ≤ m) :=
sorry

end smallest_prime_perf_sqr_minus_eight_l390_390429


namespace geometric_seq_a4_l390_390646

variable {a : ℕ → ℝ}

theorem geometric_seq_a4 (h : ∀ n, a (n + 2) / a n = a 2 / a 0)
  (root_condition1 : a 2 * a 6 = 64)
  (root_condition2 : a 2 + a 6 = 34) :
  a 4 = 8 :=
by
  sorry

end geometric_seq_a4_l390_390646


namespace green_square_probability_l390_390865

theorem green_square_probability: 
  ∃ (m n : ℕ), m.gcd n = 1 ∧ (∑ (x, y : Fin 4), (1 / 2))^16 - "\Relevant inclusion exclusion results here" = 81205 :=
sorry

end green_square_probability_l390_390865


namespace find_other_discount_l390_390750

theorem find_other_discount :
  ∃ (x: ℝ), 70 * 0.9 * (1 - x / 100) = 61.74 → x = 2 :=
begin
  sorry
end

end find_other_discount_l390_390750


namespace three_person_subcommittees_count_l390_390175

theorem three_person_subcommittees_count : ∃ n k, n = 8 ∧ k = 3 ∧ nat.choose n k = 56 :=
begin
  use [8, 3],
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end three_person_subcommittees_count_l390_390175


namespace original_price_l390_390021

example (SP : ℝ := 78) (Profit_Percentage : ℝ := 0.30) : ℝ :=
  let CP := SP / (1 + Profit_Percentage)
  CP

theorem original_price (SP : ℝ := 78) (Profit_Percentage : ℝ := 0.30) : SP / (1 + Profit_Percentage) = 60 := by
  sorry

end original_price_l390_390021


namespace whole_numbers_between_sqrts_l390_390185

theorem whole_numbers_between_sqrts :
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let start := Nat.ceil lower_bound
  let end_ := Nat.floor upper_bound
  ∃ n, n = end_ - start + 1 ∧ n = 7 := by
  sorry

end whole_numbers_between_sqrts_l390_390185


namespace calc_expression_l390_390844

theorem calc_expression :
  real.sqrt 12 + (3 - real.pi)^0 + abs (1 - real.sqrt 3) = 3 * real.sqrt 3 :=
by
  sorry

end calc_expression_l390_390844


namespace omega_2n_eq_omega_n_not_omega_2n_plus_3_eq_omega_n_plus_1_omega_8n_plus_5_eq_omega_4n_plus_3_omega_pow2_n_minus_1_eq_n_l390_390329

def bin_weight (n : ℕ) : ℕ := n.binary_digits.count 1

theorem omega_2n_eq_omega_n (n : ℕ) : bin_weight (2 * n) = bin_weight n := by
  sorry

theorem not_omega_2n_plus_3_eq_omega_n_plus_1 (n : ℕ) : bin_weight (2 * n + 3) ≠ bin_weight n + 1 := by
  sorry

theorem omega_8n_plus_5_eq_omega_4n_plus_3 (n : ℕ) : bin_weight (8 * n + 5) = bin_weight (4 * n + 3) := by
  sorry

theorem omega_pow2_n_minus_1_eq_n (n : ℕ) : bin_weight (2^n - 1) = n := by
  sorry

end omega_2n_eq_omega_n_not_omega_2n_plus_3_eq_omega_n_plus_1_omega_8n_plus_5_eq_omega_4n_plus_3_omega_pow2_n_minus_1_eq_n_l390_390329


namespace omega_2n_eq_omega_n_omega_8n_plus_5_eq_omega_4n_plus_3_omega_2_pow_n_minus_1_eq_n_l390_390323

def bit_weight (n : ℕ) : ℕ :=
  (n.bits.map (λ b, if b then 1 else 0)).sum

theorem omega_2n_eq_omega_n (n : ℕ) : 
  bit_weight (2 * n) = bit_weight n := 
sorry

theorem omega_8n_plus_5_eq_omega_4n_plus_3 (n : ℕ) : 
  bit_weight (8 * n + 5) = bit_weight (4 * n + 3) := 
sorry

theorem omega_2_pow_n_minus_1_eq_n (n : ℕ) : 
  bit_weight (2 ^ n - 1) = n := 
sorry

end omega_2n_eq_omega_n_omega_8n_plus_5_eq_omega_4n_plus_3_omega_2_pow_n_minus_1_eq_n_l390_390323


namespace area_of_rectangle_l390_390985

-- Define the given conditions
def side_length_of_square (s : ℝ) (ABCD : ℝ) : Prop :=
  ABCD = 4 * s^2

def perimeter_of_rectangle (s : ℝ) (perimeter : ℝ): Prop :=
  perimeter = 8 * s

-- Statement of the proof problem
theorem area_of_rectangle (s perimeter_area : ℝ) (h_perimeter : perimeter_of_rectangle s 160) :
  side_length_of_square s 1600 :=
by
  sorry

end area_of_rectangle_l390_390985


namespace argument_of_sum_is_correct_l390_390839

noncomputable def calculate_argument : ℂ :=
  complex.exp (complex.I * (11 * real.pi / 60)) +
  complex.exp (complex.I * (23 * real.pi / 60)) +
  complex.exp (complex.I * (35 * real.pi / 60)) +
  complex.exp (complex.I * (47 * real.pi / 60)) +
  complex.exp (complex.I * (59 * real.pi / 60))

theorem argument_of_sum_is_correct : ∀ r θ,
  calculate_argument = r * complex.exp (complex.I * θ) →
  0 ≤ θ ∧ θ < 2 * real.pi →
  θ = 7 * real.pi / 12 :=
by
  -- This part is left as an exercise/proof.
  sorry

end argument_of_sum_is_correct_l390_390839


namespace find_ellipse_equation_l390_390912

structure Ellipse (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0
  major_axis_greater : a > b
  equation : ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

theorem find_ellipse_equation :
  ∃ a b : ℝ, 
    Ellipse a b ∧
    (a^2 - b^2 = 4) ∧
    ((sqrt 2)^2 / a^2 + (sqrt 3)^2 / b^2 = 1) ∧
    (a = 2 * sqrt 2) ∧
    (b = 2) :=
by
  use (2 * sqrt 2), 2
  have a_pos := by norm_num; exact sqrt_pos.2 zero_lt_two
  have b_pos := by norm_num
  have major_axis_greater : 2 * sqrt 2 > 2 := by linarith [sqrt_pos.2 zero_lt_two, sqrt_two_lt_two]
  have equation : ∀ (x y : ℝ), (x^2 / (2 * sqrt 2)^2) + (y^2 / (2)^2) = 1 := sorry
  refine ⟨Ellipse.mk a_pos b_pos major_axis_greater equation, _, _, _, _⟩
  sorry

end find_ellipse_equation_l390_390912


namespace length_of_bridge_l390_390052

def length_of_train := 110
def speed_of_train_km_per_hr := 90
def time_taken_to_cross := 9.679225661947045

theorem length_of_bridge 
: let speed_of_train_m_per_s := speed_of_train_km_per_hr * (1000 / 3600)
    ∧ let total_distance := speed_of_train_m_per_s * time_taken_to_cross
    ∧ let length_of_bridge := total_distance - length_of_train
    in length_of_bridge = 131.9806415486761 :=
by
  let speed_of_train_m_per_s := speed_of_train_km_per_hr * (1000 / 3600)
  let total_distance := speed_of_train_m_per_s * time_taken_to_cross
  let length_of_bridge := total_distance - length_of_train
  have h1 : speed_of_train_m_per_s = 25 := by sorry
  have h2 : total_distance = 241.9806415486761 := by sorry
  have h3 : length_of_bridge = 131.9806415486761 := by sorry
  exact h3

end length_of_bridge_l390_390052


namespace area_of_right_triangle_l390_390981

theorem area_of_right_triangle (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (angle_ABC : ∠ B A C = π / 2) 
  (AB_len : dist A B = 8) 
  (AC_len : dist A C = 10) : 
  ∃ BC_len S_ABC, dist B C = BC_len ∧ BC_len = sqrt (AC_len^2 - AB_len^2) ∧ S_ABC = 1/2 * AB_len * BC_len ∧ S_ABC = 24 :=
by 
  sorry

end area_of_right_triangle_l390_390981


namespace fourth_student_number_is_17_l390_390221

-- Define the condition of ninth-grade students and systematic sampling parameters
def total_population : ℕ := 48
def sample_size : ℕ := 4
def sampling_interval : ℕ := total_population / sample_size
def lowest_student_in_sample : ℕ := 5
def students_in_sample : list ℕ := [5, 29, 41]

-- Theorem stating the number of the fourth student in the sample is 17
theorem fourth_student_number_is_17 :
  (students_in_sample.length = 3 ∧ 
   students_in_sample.contains lowest_student_in_sample ∧ 
   ∀ s ∈ students_in_sample, s + sampling_interval ∈ list.finRange total_population.succ) →
  (lowest_student_in_sample + sampling_interval = 17) :=
by
  sorry

end fourth_student_number_is_17_l390_390221


namespace count_x0_eq_x10_l390_390113

noncomputable def x_sequence (x0 : ℝ) : ℕ → ℝ
| 0     := x0
| (n+1) := if 2 * x_sequence n < 1 then 2 * x_sequence n else 2 * x_sequence n - 1

theorem count_x0_eq_x10 : 
  let valid_x0 := {x0 : ℝ | 0 ≤ x0 ∧ x0 < 1 ∧ x0 = x_sequence x0 10}
  in set.finite valid_x0 ∧ set.card valid_x0 = 1023 := 
by
  sorry

end count_x0_eq_x10_l390_390113


namespace a_4_value_l390_390987

-- Definitions and Theorem
variable {α : Type*} [LinearOrderedField α]

noncomputable def geometric_seq (a₀ : α) (q : α) (n : ℕ) : α := a₀ * q ^ n

theorem a_4_value (a₁ : α) (q : α) (h : geometric_seq a₁ q 1 * geometric_seq a₁ q 2 * geometric_seq a₁ q 6 = 8) : 
  geometric_seq a₁ q 3 = 2 :=
sorry

end a_4_value_l390_390987


namespace rangeOfA_sumOfFunctionValues_l390_390152

noncomputable def f (x a : ℝ) : ℝ := exp x - (1/2) * x^2 - a * x

theorem rangeOfA (a : ℝ) : (1 < a) ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (1 < a) :=
sorry

theorem sumOfFunctionValues {a : ℝ} (h : 1 < a) (x1 x2 : ℝ) (hx1x2 : x1 ≠ x2) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) :
  f x1 a + f x2 a > 2 :=
sorry

end rangeOfA_sumOfFunctionValues_l390_390152


namespace jordan_oreos_l390_390994

def oreos (james jordan total : ℕ) : Prop :=
  james = 2 * jordan + 3 ∧
  jordan + james = total

theorem jordan_oreos (J : ℕ) (h : oreos (2 * J + 3) J 36) : J = 11 :=
by
  sorry

end jordan_oreos_l390_390994


namespace arrange_order_l390_390558

-- Definitions of given conditions
def a : Real := Real.log 8 / Real.log 4
def b : Real := Real.log 8 / Real.log 0.4
def c : Real := 2 ^ 0.4

-- The proof goal, rearranging in increasing order
theorem arrange_order : b < c ∧ c < a := by
  sorry

end arrange_order_l390_390558


namespace work_fraction_left_l390_390460

theorem work_fraction_left (A_efficiency : Rational := 0.9) (B_efficiency : Rational := 0.8) (C_efficiency : Rational := 0.75)
  (A_days : Rational := 15) (B_days : Rational := 20) (C_days : Rational := 30) :
  (A_efficiency * (1 / A_days) + B_efficiency * (1 / B_days) + C_efficiency * (1 / C_days)) * 4 = 1/2 :=
by
  sorry

end work_fraction_left_l390_390460


namespace money_left_l390_390660

-- Conditions
def initial_savings : ℤ := 6000
def spent_on_flight : ℤ := 1200
def spent_on_hotel : ℤ := 800
def spent_on_food : ℤ := 3000

-- Total spent
def total_spent : ℤ := spent_on_flight + spent_on_hotel + spent_on_food

-- Prove that the money left is $1,000
theorem money_left (h1 : initial_savings = 6000)
                   (h2 : spent_on_flight = 1200)
                   (h3 : spent_on_hotel = 800)
                   (h4 : spent_on_food = 3000) :
                   initial_savings - total_spent = 1000 :=
by
  -- Insert proof steps here
  sorry

end money_left_l390_390660


namespace angle_e_l390_390064

noncomputable def complexTrianglePoints : Type := sorry

variables (a b c : complexTrianglePoints)
variables (d e f e' f' : complexTrianglePoints)
variables (h: complexTrianglePoints)

-- acute non-isosceles triangle condition
axiom acute_non_isosceles_triangle : ∀ (a b c : complexTrianglePoints), {ABC : complexTrianglePoints // acuteTriangle a b c ∧ ¬isosceles a b c}

-- altitude conditions
axiom altitudes_intersect_orthocenter : ∀ (a b c d e f : complexTrianglePoints), 
    altitudes_intersect_orthocenter a b c d e f

-- symmetric points condition
axiom symmetric_points : ∀ (e f e' f' a b: complexTrianglePoints), 
    symmetric e a e' ∧ symmetric f b f'

-- point C1 condition
axiom point_c1_condition : ∀ (c1 c d a b: complexTrianglePoints), 
    lies_on_ray c1 c d ∧ distance_condition dc1_3cd c1 c d 3 

theorem angle_e'_c1_f'_equals_angle_acb : 
    ∀ (a b c d e f e' f' c1 : complexTrianglePoints), 
    acute_non_isosceles_triangle a b c →
    altitudes_intersect_orthocenter a b c d e f →
    symmetric_points e f e' f' a b →
    point_c1_condition c1 c d →
    angle E' c1 F' = angle ACB :=
sorry

end angle_e_l390_390064


namespace cannot_be_written_as_sum_of_consecutive_odd_integers_l390_390786

theorem cannot_be_written_as_sum_of_consecutive_odd_integers (n : ℕ) :
  ∀ x ∈ {28, 52, 84, 112, 220}, x = 112 → ¬ (∃ k : ℤ, x = 4 * k + 12 ∧ (4 * k + 4) % 10 = 0) :=
by
  sorry

end cannot_be_written_as_sum_of_consecutive_odd_integers_l390_390786


namespace square_area_l390_390416

theorem square_area (s : ℕ) (h : s = 13) : s * s = 169 := by
  sorry

end square_area_l390_390416


namespace initial_sum_invested_l390_390046

theorem initial_sum_invested (r : ℝ) (P A1 A2 : ℝ) (n t : ℕ) (h : n = 1) (h2 : t = 15)
  (hA1 : A1 = P * (1 + r / n)^(n * t)) (hA2 : A2 = P * (1 + (r / n) + 0.05)^(n * t))
  (hDifference : A2 = A1 + 2500) :
  P ≈ 2317.97 :=
by
  -- Proof is omitted
  sorry

end initial_sum_invested_l390_390046


namespace parabola_region_vertices_count_l390_390702

open Set

variable (n : ℕ) (parabolas : Fin n → ℝ → ℝ)
hypothesis (quadratic : ∀ i, ∃ a b c, parabolas i = fun x => a * x^2 + b * x + c)
hypothesis (no_tangent : ∀ i j, i ≠ j → Disjoint {x | parabolas i (x) = parabolas j (x)} {t | (parabolas i) t > 0})

/--
On the coordinate plane, there are n parabolas, each being the graph of a quadratic polynomial. No two of them are tangent to each other. 
These parabolas divide the plane into several regions. One of these regions is located above all the parabolas. 
Prove that the boundary of this region has at most 2(n-1) vertices (i.e., points of intersection of pairs of parabolas).
-/
theorem parabola_region_vertices_count :
  ∃! region, is_region_above_all_parabolas region n parabolas ∧ region_boundary_vertices region n parabolas ≤ 2 * (n - 1) :=
sorry

end parabola_region_vertices_count_l390_390702


namespace eighty_percent_of_number_l390_390796

theorem eighty_percent_of_number (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 := 
by sorry

end eighty_percent_of_number_l390_390796


namespace car_speed_l390_390500

theorem car_speed (uses_one_gallon_per_30_miles : ∀ d : ℝ, d = 30 → d / 30 ≥ 1)
    (full_tank : ℝ := 10)
    (travel_time : ℝ := 5)
    (fraction_of_tank_used : ℝ := 0.8333333333333334)
    (speed : ℝ := 50) :
  let amount_of_gasoline_used := fraction_of_tank_used * full_tank
  let distance_traveled := amount_of_gasoline_used * 30
  speed = distance_traveled / travel_time :=
by
  sorry

end car_speed_l390_390500


namespace average_daily_production_l390_390449

theorem average_daily_production :
  let production_january := 2000 in
  let increment := 100 in
  let days_in_year := 365 in
  let monthly_production := λ n, production_january + n * increment in
  let total_production := production_january + (List.sum (List.range 11).map monthly_production) in
  (total_production / days_in_year : ℝ) = 83.84 :=
by
  let production_january := 2000
  let increment := 100
  let days_in_year := 365
  let monthly_production := λ n, production_january + n * increment
  let total_production_january := 2000
  let total_production_feb_to_dec := List.sum (List.range 11).map monthly_production
  let total_production := total_production_january + total_production_feb_to_dec
  have : (total_production_january + total_production_feb_to_dec) / days_in_year = 83.84 := sorry
  exact this

end average_daily_production_l390_390449


namespace a_2003_eq_4005_l390_390344

noncomputable def f : ℝ → ℝ := sorry
def a (n : ℕ) : ℕ := if n = 1 then f 0 else a (n - 1) + 2

theorem a_2003_eq_4005 :
  (∀ x : ℝ, f x > 1) ∧
  (∀ x y : ℝ, f (x + y) = f x * f y) ∧
  (∀ n : ℕ, f (a n + 1) = (1 / f (-2 - a n))) → 
  a 2003 = 4005 :=
sorry

end a_2003_eq_4005_l390_390344


namespace pipes_needed_l390_390038

theorem pipes_needed (h : ℝ) : 
  let d_large := 10
  let r_large := d_large / 2
  let d_small := 2
  let r_small := d_small / 2
  let V_large := π * r_large^2 * h
  let V_small := π * r_small^2 * h
  let n := V_large / V_small
  in n = 25 := 
by
  sorry

end pipes_needed_l390_390038


namespace sleepy_hollow_headless_horseman_l390_390227

theorem sleepy_hollow_headless_horseman (n : ℕ) :
  (∃ G : Graph, G.degree_all 3 ∧ G.vertex_count = 2 * n ∧ G.is_traversable_by_headless_horseman) ↔ (odd n ∧ n ≥ 3) :=
sorry

end sleepy_hollow_headless_horseman_l390_390227


namespace relationship_xyz_l390_390119

theorem relationship_xyz (x y z : ℝ) (h1 : x = Real.log x) (h2 : y = Real.logb 5 2) (h3 : z = Real.exp (-0.5)) : x > z ∧ z > y :=
by
  sorry

end relationship_xyz_l390_390119


namespace value_sum_l390_390958

theorem value_sum (x : ℝ) (h : x^5 + x^4 + x = -1) : x^{1997} + x^{1998} + x^{1999} + x^{2000} + x^{2001} + x^{2002} + x^{2003} + x^{2004} + x^{2005} + x^{2006} + x^{2007} = -1 :=
by
  sorry

end value_sum_l390_390958


namespace no_solution_iff_m_leq_2_l390_390202

theorem no_solution_iff_m_leq_2 (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2 < 3x - 6 ∧ x < m)) ↔ m ≤ 2 :=
by
  sorry

end no_solution_iff_m_leq_2_l390_390202


namespace remainder_twice_sum_first_150_mod_10000_eq_2650_l390_390780

theorem remainder_twice_sum_first_150_mod_10000_eq_2650 :
  let n := 150
  let S := n * (n + 1) / 2  -- Sum of first 150 numbers
  let result := 2 * S
  result % 10000 = 2650 :=
by
  sorry -- proof not required

end remainder_twice_sum_first_150_mod_10000_eq_2650_l390_390780


namespace largest_fraction_is_frac23_l390_390927

theorem largest_fraction_is_frac23 : 
    let frac35 := (3 : ℚ) / 5
    let frac23 := (2 : ℚ) / 3
    let frac49 := (4 : ℚ) / 9
    let frac515 := (5 : ℚ) / 15
    let frac845 := (8 : ℚ) / 45
  in frac23 > frac35 ∧ frac23 > frac49 ∧ frac23 > frac515 ∧ frac23 > frac845 :=
by
  let frac35 := (3 : ℚ) / 5
  let frac23 := (2 : ℚ) / 3
  let frac49 := (4 : ℚ) / 9
  let frac515 := (5 : ℚ) / 15
  let frac845 := (8 : ℚ) / 45
  sorry

end largest_fraction_is_frac23_l390_390927


namespace exists_nat_nums_sum_of_cubes_l390_390089

-- Let's define the theorem we want to prove
theorem exists_nat_nums_sum_of_cubes :
  ∃ (a b c d : ℕ), a^3 + b^3 + c^3 + d^3 = (100 : ℕ)^100 :=
begin
  sorry
end

end exists_nat_nums_sum_of_cubes_l390_390089


namespace three_person_subcommittees_count_l390_390177

theorem three_person_subcommittees_count : ∃ n k, n = 8 ∧ k = 3 ∧ nat.choose n k = 56 :=
begin
  use [8, 3],
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end three_person_subcommittees_count_l390_390177


namespace intersection_of_domains_l390_390596

def M (x : ℝ) : Prop := x < 1
def N (x : ℝ) : Prop := x > -1
def P (x : ℝ) : Prop := -1 < x ∧ x < 1

theorem intersection_of_domains : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | P x} :=
by
  sorry

end intersection_of_domains_l390_390596


namespace moe_has_least_money_l390_390504

variables (Bo Coe Flo Jo Moe Zo : ℝ)

-- Conditions
axiom H1 : Bo > Moe ∧ Bo < Coe
axiom H2 : Zo > Moe ∧ Zo < Coe
axiom H3 : Flo > Jo ∧ Flo < Coe
axiom H4 : Jo > Moe ∧ Jo < Bo

theorem moe_has_least_money :
  (∀ x : ℝ, x = Bo ∨ x = Coe ∨ x = Flo ∨ x = Jo ∨ x = Moe ∨ x = Zo → Moe ≤ x) :=
begin
  sorry
end

end moe_has_least_money_l390_390504


namespace salt_solution_proof_l390_390948

theorem salt_solution_proof (x : ℝ) (P : ℝ) (hx : x = 28.571428571428573) :
  ((P / 100) * 100 + x) = 0.30 * (100 + x) → P = 10 :=
by
  sorry

end salt_solution_proof_l390_390948


namespace minimum_red_chips_l390_390022

theorem minimum_red_chips (w b r : ℕ) (h1 : b ≥ w / 4) (h2 : b ≤ r / 6) (h3 : w + b ≥ 75) : r ≥ 90 :=
sorry

end minimum_red_chips_l390_390022


namespace original_number_is_fraction_l390_390357

theorem original_number_is_fraction (x : ℚ) (h : 1 + 1/x = 7/3) : x = 3/4 :=
sorry

end original_number_is_fraction_l390_390357


namespace extremum_at_three_l390_390963

def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a) / exp x

theorem extremum_at_three (a : ℝ) : 
  (∃ x, x = 3 ∧ (deriv (λ x, f x a) x = 0)) → a = -3 := 
by
  sorry

end extremum_at_three_l390_390963


namespace sample_distribution_correct_l390_390807

theorem sample_distribution_correct :
  ∃ (e m y : ℕ),
    e = 6 ∧
    m = 12 ∧
    y = 18 ∧
    (e + m + y = 36) ∧
    (e ≤ 27) ∧
    (m ≤ 54) ∧
    (y ≤ 81) :=
by {
  use [6, 12, 18],
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  split, exact rfl,
  split, linarith,
  split, linarith,
  linarith,
}

end sample_distribution_correct_l390_390807


namespace sum_of_four_terms_l390_390641

theorem sum_of_four_terms (a d : ℕ) (h1 : a + d > a) (h2 : a + 2 * d > a + d)
  (h3 : (a + 2 * d) * (a + 2 * d) = (a + d) * (a + 3 * d)) (h4 : (a + 3 * d) - a = 30) :
  a + (a + d) + (a + 2 * d) + (a + 3 * d) = 129 :=
sorry

end sum_of_four_terms_l390_390641


namespace final_number_on_board_l390_390696

theorem final_number_on_board :
  let nums := list.range 100 |>.map (λ n, 1 / (n + 1))
  (nums.foldl (λ acc x, (acc + 1) * (x + 1) - 1) 0 = 100) :=
by
  let nums := list.range 100 |>.map (λ n, 1 / (n + 1))
  have h : (nums.foldl (λ acc x, (acc + 1) * (x + 1) - 1) 0 = 100) := sorry
  exact h

end final_number_on_board_l390_390696


namespace weighted_average_of_angles_l390_390228

def triangle_inequality (a b c α β γ : ℝ) : Prop :=
  (a - b) * (α - β) ≥ 0 ∧ (b - c) * (β - γ) ≥ 0 ∧ (a - c) * (α - γ) ≥ 0

noncomputable def angle_sum (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi

theorem weighted_average_of_angles (a b c α β γ : ℝ)
  (h1 : triangle_inequality a b c α β γ)
  (h2 : angle_sum α β γ) :
  Real.pi / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < Real.pi / 2 :=
by
  sorry

end weighted_average_of_angles_l390_390228


namespace integral_sin_squared_cos_cubed_l390_390543

theorem integral_sin_squared_cos_cubed :
  ∀ (C : ℝ), 
  ∫ (x : ℝ) in real.interval_integrable -real.pi real.pi, (sin x)^2 * (cos x)^3 = 
  λ (x : ℝ), (sin x)^3 / 3 - (sin x)^5 / 5 + C := 
by
  sorry

end integral_sin_squared_cos_cubed_l390_390543


namespace g_at_3_eq_38_div_5_l390_390371

def f (x : ℝ) : ℝ := 4 / (3 - x)
def f_inv (x : ℝ) : ℝ := 3 - 4 / x
def g (x : ℝ) : ℝ := 1 / f_inv(x) + 7

theorem g_at_3_eq_38_div_5 : g 3 = 38 / 5 :=
by
  sorry

end g_at_3_eq_38_div_5_l390_390371


namespace ashok_total_subjects_l390_390499

variable (n : ℕ) (T : ℕ)

theorem ashok_total_subjects (h_ave_all : 75 * n = T + 80)
                       (h_ave_first : T = 74 * (n - 1)) :
  n = 6 := sorry

end ashok_total_subjects_l390_390499


namespace num_5_letter_words_with_at_least_two_consonants_l390_390612

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := 6^5
def words_with_0_consonants : ℕ := 2^5
def words_with_1_consonant : ℕ := 5 * 4 * 2^4

theorem num_5_letter_words_with_at_least_two_consonants : 
  total_5_letter_words - (words_with_0_consonants + words_with_1_consonant) = 7424 := by
  sorry

end num_5_letter_words_with_at_least_two_consonants_l390_390612


namespace eccentricity_range_l390_390267

noncomputable def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  { z : ℝ × ℝ // (z.1^2 / a^2) + (z.2^2 / b^2) = 1 }

theorem eccentricity_range (a b : ℝ) (h : a > b) (hb : b > 0):
  ∀ (P : ellipse a b h hb),
    |(P.1, P.2 - b).norm ≤ 2 * b → 
    eccentricity (a b) ∈ set.Ici (real.sqrt 2 / 2) :=
sorry

end eccentricity_range_l390_390267


namespace concyclicity_or_collinearity_concyclicity_condition_l390_390833

variables {P : Type} [TopologicalSpace P] [MetricSpace P] [NormedGroup P] [NormedSpace ℝ P]
variables {O1 O2 O : ℝ} -- circles are represented with their radii and centers within P
variables {A C1 C2 B1 B2 D1 D2 : P}
variables {l : Point → Prop} -- l as a predicate that points must satisfy to be on the line
variables {n k : Circle ℝ} -- n and k are Circles defined by radius and center

def tangent_at (O1 O2 : Circle ℝ) (A : Point) : Prop := 
  -- Placeholder definition for circles being tangent at A
  sorry

-- condition definitions
axiom Circle_tangents : ∀ (O1 O2 : Circle ℝ) (A : Point), tangent_at O1 O2 A

axiom Line_intersects : ∀ (l : Point → Prop) (O : Circle ℝ) (A C : Point), l A → l C → OnCircle O C

axiom Circle_passes : ∀ (O : Circle ℝ) (C1 C2 : Point), OnCircle O C1 → OnCircle O C2

axiom Circle_intersection : ∀ (O1 O2 : Circle ℝ) (B1 B2 : Point), OnCircle O1 B1 → OnCircle O2 B2 

axiom Circumcircle : ∀ (A B1 B2 : Point), ∃ (n : Circle ℝ), OnCircle n A ∧ OnCircle n B1 ∧ OnCircle n B2

axiom Tangent_circles : ∀ (n k : Circle ℝ) (A : Point), tangent_at n k A

-- theorem statements
theorem concyclicity_or_collinearity
  (tang: tangent_at O1 O2 A)
  (h_line1: Line_intersects l O1 A C1)
  (h_line2: Line_intersects l O2 A C2) 
  (h_circle1: Circle_passes O C1 C2)
  (h_intersect1: Circle_intersection O1 O B1)
  (h_intersect2: Circle_intersection O2 O B2)
  (h_circum: Circumcircle A B1 B2)
  (h_tangent: Tangent_circles n k A)
  (h_intersect3: Circle_intersection O1 k D1)
  (h_intersect4: Circle_intersection O2 k D2)
  : collinear C1 C2 D1 D2 ∨ concyclic C1 C2 D1 D2 :=
sorry

theorem concyclicity_condition 
  (tang: tangent_at O1 O2 A)
  (h_line1: Line_intersects l O1 A C1)
  (h_line2: Line_intersects l O2 A C2) 
  (h_circle1: Circle_passes O C1 C2)
  (h_intersect1: Circle_intersection O1 O B1)
  (h_intersect2: Circle_intersection O2 O B2)
  (h_circum: Circumcircle A B1 B2)
  (h_tangent: Tangent_circles n k A)
  (h_intersect3: Circle_intersection O1 k D1)
  (h_intersect4: Circle_intersection O2 k D2)
  : concyclic B1 B2 D1 D2 ↔ (diag (A C1) (O1) ∧ diag (A C2) (O2)) :=
sorry

end concyclicity_or_collinearity_concyclicity_condition_l390_390833


namespace least_possible_value_of_quadratic_l390_390626

theorem least_possible_value_of_quadratic (p q : ℝ) (hq : ∀ x : ℝ, x^2 + p * x + q ≥ 0) : q = (p^2) / 4 :=
sorry

end least_possible_value_of_quadratic_l390_390626


namespace max_value_complex_expression_l390_390676

open Complex

theorem max_value_complex_expression (α γ : ℂ) (h1 : |γ| = 2) (h2 : (conj α) * γ ≠ 2) : 
  (∃ α γ : ℂ, |γ| = 2 ∧ (conj α) * γ ≠ 2 ∧ (abs ((γ - α) / (2 - (conj α) * γ)) ≤ 2)) := sorry

end max_value_complex_expression_l390_390676


namespace good_number_sum_l390_390469

def is_good (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem good_number_sum (a : ℕ) (h1 : a > 6) (h2 : is_good a) :
  ∃ x y : ℕ, is_good x ∧ is_good y ∧ a * (a + 1) = x * (x + 1) + 3 * y * (y + 1) :=
sorry

end good_number_sum_l390_390469


namespace a2b_sub_ab2_eq_neg16sqrt5_l390_390898

noncomputable def a : ℝ := 4 + 2 * Real.sqrt 5
noncomputable def b : ℝ := 4 - 2 * Real.sqrt 5

theorem a2b_sub_ab2_eq_neg16sqrt5 : a^2 * b - a * b^2 = -16 * Real.sqrt 5 :=
by
  sorry

end a2b_sub_ab2_eq_neg16sqrt5_l390_390898


namespace focal_length_proof_l390_390593

noncomputable def focal_length_of_ellipse (a : ℝ) (h1 : a > 4) 
  (e : ℝ) (h2 : e = (real.sqrt 3) / 3) : ℝ :=
2 * real.sqrt (a^2 - 16)

theorem focal_length_proof : ∀ (a : ℝ) (h1 : a > 4) (h2 : (real.sqrt 3) / 3 = (real.sqrt 3) / 3),
  focal_length_of_ellipse a h1 h2 = 4 * real.sqrt 2 :=
by
  intros
  unfold focal_length_of_ellipse
  sorry

end focal_length_proof_l390_390593


namespace coffee_containers_used_l390_390725

theorem coffee_containers_used :
  let Suki_coffee := 6.5 * 22
  let Jimmy_coffee := 4.5 * 18
  let combined_coffee := Suki_coffee + Jimmy_coffee
  let containers := combined_coffee / 8
  containers = 28 := 
by
  sorry

end coffee_containers_used_l390_390725


namespace sons_age_l390_390466

theorem sons_age (S M : ℕ) (h1 : M = 3 * S) (h2 : M + 12 = 2 * (S + 12)) : S = 12 :=
by 
  sorry

end sons_age_l390_390466


namespace pipe_fills_tank_without_leak_l390_390037

theorem pipe_fills_tank_without_leak (T : ℝ) (h1 : 1 / 6 = 1 / T - 1 / 12) : T = 4 :=
by
  sorry

end pipe_fills_tank_without_leak_l390_390037


namespace least_value_fX_l390_390372

noncomputable def tetrahedron (A B C D : Point)
  (AD BC : dist A C = 30)
  (AC BD : dist A D = 40)
  (AB CD : dist A B = 50) : Prop := 
∃ (X : Point), ∀ (X : Point), ∃ (m n : ℕ), 
  f(X) = m * (n : ℝ).sqrt ∧ n.is_squarefree ∧ (m + n = 2703)

theorem least_value_fX (A B C D : Point)
  (h1 : dist A D = 30)
  (h2 : dist B C = 30)
  (h3 : dist A C = 40)
  (h4 : dist B D = 40)
  (h5 : dist A B = 50)
  (h6 : dist C D = 50) :
  tetrahedron A B C D AD BC AC BD AB CD := sorry

end least_value_fX_l390_390372


namespace problem_l390_390619

theorem problem (p q r s : ℝ) :
  let B := Matrix.vecCons (Matrix.vecCons (2 * p) (Matrix.vecCons (2 * q) Matrix.vecEmpty))
                         (Matrix.vecCons (2 * r) (Matrix.vecCons (2 * s) Matrix.vecEmpty))
  let B_T := Matrix.vecCons (Matrix.vecCons (2 * p) (Matrix.vecCons (2 * r) Matrix.vecEmpty))
                           (Matrix.vecCons (2 * q) (Matrix.vecCons (2 * s) Matrix.vecEmpty))
  B_T = 4 • B⁻¹ →
  p^2 + q^2 + r^2 + s^2 = 2 :=
begin
  sorry
end

end problem_l390_390619


namespace sum_abs_roots_polynomial_l390_390880

theorem sum_abs_roots_polynomial :
  let poly := Polynomial.C 1 * Polynomial.X^4 - Polynomial.C 6 * Polynomial.X^3 + Polynomial.C 9 * Polynomial.X^2 + Polynomial.C 24 * Polynomial.X - Polynomial.C 36 in
  let roots := [3 + Real.sqrt 3, 3 - Real.sqrt 3, Real.sqrt 6, -Real.sqrt 6] in
  (roots.map Real.abs).sum = 6 + 2 * Real.sqrt 6 :=
by
  let poly := Polynomial.C 1 * Polynomial.X^4 - Polynomial.C 6 * Polynomial.X^3 + Polynomial.C 9 * Polynomial.X^2 + Polynomial.C 24 * Polynomial.X - Polynomial.C 36
  let roots := [3 + Real.sqrt 3, 3 - Real.sqrt 3, Real.sqrt 6, -Real.sqrt 6]
  have h : (roots.map Real.abs).sum = 6 + 2 * Real.sqrt 6 := sorry
  exact h

end sum_abs_roots_polynomial_l390_390880


namespace minimize_transportation_cost_l390_390388

-- Define the given variables and conditions
variables (b a n : ℝ) (h₁ : n > 1) (h₂ : b > 0) (h₃ : a > 0)

-- The theorem stating the proof problem
theorem minimize_transportation_cost 
  (h₄ : b > 0) (h₅ : n > 1) : 
  let min_DC := b / (real.sqrt (n^2 - 1)) in 
  true :=
sorry

end minimize_transportation_cost_l390_390388


namespace eccentricity_range_l390_390291

-- Ellipse definition
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Upper vertex B of the ellipse
def B (b : ℝ) : ℝ × ℝ := (0, b)

-- Distance PB condition
def distance_le_2b (a b x0 y0 : ℝ) (h : ellipse a b (and.intro (sorry) (sorry)) x0 y0) : Prop :=
  (x0 - 0)^2 + (y0 - b)^2 ≤ (2 * b)^2

-- Range of eccentricity
def eccentricity (a b c : ℝ) : ℝ :=
  c / a

-- Main theorem
theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  (forall (x0 y0 : ℝ), ellipse a b h x0 y0 → distance_le_2b a b x0 y0 h) →
  eccentricity a b (sqrt (a^2 - b^2)) ∈ set.Icc 0 (sqrt 2 / 2) :=
by
  sorry

end eccentricity_range_l390_390291


namespace smallest_prime_perf_sqr_minus_eight_l390_390430

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_perf_sqr_minus_eight (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 8

theorem smallest_prime_perf_sqr_minus_eight :
  ∃ (n : ℕ), is_prime n ∧ is_perf_sqr_minus_eight n ∧ (∀ m : ℕ, is_prime m ∧ is_perf_sqr_minus_eight m → n ≤ m) :=
sorry

end smallest_prime_perf_sqr_minus_eight_l390_390430


namespace magnitude_of_sum_l390_390943

open_locale real_inner_product_space

variables (a b : ℝ^3)
variables (ha : ‖a‖ = 1)
variables (hb : ‖b‖ = 1)
variables (angle_ab : real.angle a b = real.pi / 3)

def vec_magnitude_sum : Prop :=
  ‖a + b‖ = real.sqrt 3

theorem magnitude_of_sum (a b : ℝ^3) (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (angle_ab : real.angle a b = real.pi / 3) : vec_magnitude_sum a b :=
by {
  sorry
}

end magnitude_of_sum_l390_390943


namespace units_digit_sum_l390_390783

theorem units_digit_sum :
  let units_digit_35_pow_7 := 5 in
  let pattern_3 := [3, 9, 7, 1] in
  let units_digit_93_pow_45 := pattern_3[(45 % 4)] in
  units_digit_35_pow_7 + units_digit_93_pow_45 % 10 = 8 :=
by
  let pattern_3 := [3, 9, 7, 1]
  have units_digit_35_pow_7 : Nat := 5
  have units_digit_93_pow_45 : Nat := pattern_3[(45 % 4)]
  show (units_digit_35_pow_7 + units_digit_93_pow_45) % 10 = 8, by sorry

end units_digit_sum_l390_390783


namespace shift_and_symmetry_l390_390831

theorem shift_and_symmetry (φ : ℝ) (h : (∀ x : ℝ, cos (2*x + 2*φ + π/3) = cos (-2*x + 2*φ + π/3))) : φ = π/3 :=
sorry

end shift_and_symmetry_l390_390831


namespace economy_class_seats_l390_390474

-- Definitions based on the conditions
def first_class_people : ℕ := 3
def business_class_people : ℕ := 22
def economy_class_fullness (E : ℕ) : ℕ := E / 2

-- Problem statement: Proving E == 50 given the conditions
theorem economy_class_seats :
  ∃ E : ℕ,  economy_class_fullness E = first_class_people + business_class_people → E = 50 :=
by
  sorry

end economy_class_seats_l390_390474


namespace max_value_for_three_n_plus_m_l390_390823

theorem max_value_for_three_n_plus_m :
  ∃ (n m : ℕ), (∀ (S : Finset ℕ), (S.sum id = 1987 → 3 * S.card + (S.filter (λ x, x % 2 = 1)).card ≤ 221)) ∧
               (S.sum id = 1987 → 3 * S.card + (S.filter (λ x, x % 2 = 1)).card = 221) :=
sorry

end max_value_for_three_n_plus_m_l390_390823


namespace seq_periodic_2015_l390_390755

noncomputable def seq : ℕ → ℚ
| 0       := 3
| (n + 1) := (seq n - 1) / seq n

theorem seq_periodic_2015 : seq 2015 = 2 / 3 :=
by
  -- (Proof would be here)
  sorry

end seq_periodic_2015_l390_390755


namespace probability_A_inside_B_l390_390141

noncomputable def region_A : Set (ℝ × ℝ) :=
  {p | 1 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 / p.1}

def region_B : Set (ℝ × ℝ) :=
  {p | 1 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

open Real

theorem probability_A_inside_B : 
  let area_A := ∫ (1:ℝ) in 1..3, 1 / x
  let area_B := (3 - 1) * (1 - 0)
  area_A / area_B = log 3 / 2 := 
by
  sad sorry  

end probability_A_inside_B_l390_390141


namespace number_of_cakes_l390_390738

theorem number_of_cakes (total_eggs eggs_in_fridge eggs_per_cake : ℕ) (h1 : total_eggs = 60) (h2 : eggs_in_fridge = 10) (h3 : eggs_per_cake = 5) :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 :=
by
  sorry

end number_of_cakes_l390_390738


namespace price_of_basic_computer_l390_390396

-- Conditions
variables (C P : ℝ)
axiom cond1 : C + P = 2500
axiom cond2 : 3 * P = C + 500

-- Prove that the price of the basic computer is $1750
theorem price_of_basic_computer : C = 1750 :=
by 
  sorry

end price_of_basic_computer_l390_390396


namespace trajectory_of_midpoint_l390_390906

theorem trajectory_of_midpoint (m : ℝ)
  (hyp : ∃ x1 x2 y1 y2 : ℝ, 
    (y1 = 3 * x1 + m ∧ y2 = 3 * x2 + m) ∧ 
    (x1*x1 / 4 - y1*y1 = 1 ∧ x2*x2 / 4 - y2*y2 = 1) ∧ 
    |m| > sqrt 35) :
  ∃ (x y : ℝ), 
    (x = -12 * m / 35 ∧ y = -m / 35) ∧ x - 12 * y = 0 ∧ |x| > 12 * sqrt 35 / 35 :=
by sorry

end trajectory_of_midpoint_l390_390906


namespace gcd_sum_diff_ge_two_l390_390239

-- Definitions based on the given conditions
variables {n : ℕ}
variables {a : Fin n → ℤ}
variables {b : Fin n → ℤ}
variables {c : Fin n → ℤ}

-- Assumptions from the problem
variable (h1 : ∀ (i j : Fin n), i ≠ j → a i ≠ a j)
variable (h2 : Perm (Fin n) b)
variable (h3 : Perm (Fin n) c)
variable (h4 : ∃ i, a i ≠ b i)
variable (h5 : ∃ i, a i ≠ c i)

-- Definition of sums
def sum_diff (f g : Fin n → ℤ) : ℤ := ∑ i, |f i - g i|

-- Main statement
theorem gcd_sum_diff_ge_two (h1 : ∀ (i j : Fin n), i ≠ j → a i ≠ a j)
                           (h2 : Perm (Fin n) b)
                           (h3 : Perm (Fin n) c)
                           (h4 : ∃ i, a i ≠ b i)
                           (h5 : ∃ i, a i ≠ c i) :
  Int.gcd (sum_diff a b) (sum_diff a c) ≥ 2 :=
by
  sorry

end gcd_sum_diff_ge_two_l390_390239


namespace part_I_part_II_l390_390573

-- Definitions for given conditions
def z1 (m : ℝ) : ℂ := 4 - m^2 + (m - 2) * I
def z2 (λ θ : ℝ) : ℂ := λ + sin θ + (cos θ - 2) * I

-- Part (I) statement
theorem part_I (m : ℝ) (H : z1 m.im = 0) : m = -2 :=
  sorry

-- Part (II) statement
theorem part_II (λ θ : ℝ) (H : z1 (cos θ) = z2 λ θ) : λ ∈ Set.Icc (11/4) 5 :=
  sorry

end part_I_part_II_l390_390573


namespace sum_of_squares_of_four_consecutive_even_numbers_l390_390394

open Int

theorem sum_of_squares_of_four_consecutive_even_numbers (x y z w : ℤ) 
    (hx : x % 2 = 0) (hy : y = x + 2) (hz : z = x + 4) (hw : w = x + 6)
    : x + y + z + w = 36 → x^2 + y^2 + z^2 + w^2 = 344 := by
  sorry

end sum_of_squares_of_four_consecutive_even_numbers_l390_390394


namespace solve_for_x_l390_390594

theorem solve_for_x (x y : ℝ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 6) : x = 3 := 
by 
  sorry

end solve_for_x_l390_390594


namespace problem_1_problem_2_l390_390151

noncomputable def f (a x : ℝ) : ℝ := |x + a| + |x + 1/a|

theorem problem_1 (x : ℝ) : f 2 x > 3 ↔ x < -(11 / 4) ∨ x > 1 / 4 := sorry

theorem problem_2 (a m : ℝ) (ha : a > 0) : f a m + f a (-1 / m) ≥ 4 := sorry

end problem_1_problem_2_l390_390151


namespace min_pairs_to_avoid_loss_l390_390235

noncomputable def cost (n : ℕ) : ℝ := 4000 + 50 * n

noncomputable def selling_price_per_pair : ℝ := 90

noncomputable def revenue (n : ℕ) : ℝ := selling_price_per_pair * n

noncomputable def profit (n : ℕ) : ℝ := revenue(n) - cost(n)

theorem min_pairs_to_avoid_loss (n : ℕ) (h : profit(n) ≥ 0) : n ≥ 100 :=
by
  -- Definitions inserted
  have hc : cost n = 4000 + 50 * n := rfl
  have hr : revenue n = selling_price_per_pair * n := rfl
  have hp : profit n = 40 * n - 4000 :=
    by simp [profit, revenue, cost, selling_price_per_pair, mul_sub]

  -- Loss avoidance inequality simplification
  calc
    profit(n) ≥ 0 : h
    40 * n - 4000 ≥ 0 : by simp [hp]
    n ≥ 100 : by linarith

-- sorry is unnecessary as we provided a complete structure for the theorem

end min_pairs_to_avoid_loss_l390_390235


namespace allocation_schemes_l390_390024

theorem allocation_schemes
  (employees : Finset ℕ) (places : Finset ℕ) (females : Finset ℕ)
  (h_employees_size : employees.card = 5) (h_females_size : females.card = 2)
  (h_places_size : places.card = 3) (h_subset_females : females ⊆ employees)
  (h_all_diff_places : ∀ (p : ℕ), p ∈ places → ∃ e, e ∈ employees ∧ e ∈ places) :
  ∃ allocation : ℕ → ℕ,
  (∀ e ∈ employees, 1 ≤ allocation e ∧ allocation e ≤ 3) ∧ 
  (∀ p ∈ places, ∃ e ∈ employees, allocation e = p) ∧
  (∀ f ∈ females, allocation (classical.some (set.exists_mem_of_ne_empty (females.nonempty_of_ssubset h_employees_size))) = allocation f) ∧
  (fintype.card (finset.image allocation employees) = 3) ∧
  finset.card (finset.filter (λ e, allocation e = classical.some (finset.nonempty_of_ssubset (females.nonempty_of_ssubset h_employees_size.fst)) employees) = 36 :=
  sorry

end allocation_schemes_l390_390024


namespace penny_nickel_dime_heads_probability_l390_390729

def num_successful_outcomes : Nat :=
1 * 1 * 1 * 2

def total_possible_outcomes : Nat :=
2 ^ 4

def probability_event : ℚ :=
num_successful_outcomes / total_possible_outcomes

theorem penny_nickel_dime_heads_probability :
  probability_event = 1 / 8 := 
by
  sorry

end penny_nickel_dime_heads_probability_l390_390729


namespace range_of_f_l390_390442

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) + 2

theorem range_of_f (h : ∀ x : ℝ, x ≤ 1) : (f '' {x : ℝ | x ≤ 1}) = {y : ℝ | 1 ≤ y ∧ y ≤ 2} :=
by
  sorry

end range_of_f_l390_390442


namespace expected_balls_in_original_positions_l390_390090

theorem expected_balls_in_original_positions :
  let balls := (Finset.range 8).Val -- Consider the set of 8 balls, indexed from 0 to 7
    
  (exists (Chris_trans: {c // c ∈ balls}).to_list (Silva_trans: {s // s ∈ balls}).to_list 
            (Alex_trans: {a // a ∈ balls}).to_list, 
  by
    let chosen_pairs := [Chris_trans, Silva_trans, Alex_trans]
    sorry) 
  -- Skip proof details and just assert the expected value result directly
    
  => 
  let expected_value : ℝ := 4.5 -- Correct answer from solution
  in expected_value = 4.5 := 
begin 
  -- Outline the specific steps or just assert the goal directly
  sorry -- Steps can be filled in correspondence to the proof steps
end

end expected_balls_in_original_positions_l390_390090


namespace geo_sequence_necessity_l390_390917

theorem geo_sequence_necessity (a1 a2 a3 a4 : ℝ) (h_non_zero: a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0 ∧ a4 ≠ 0) :
  (a1 * a4 = a2 * a3) → (∀ r : ℝ, (a2 = a1 * r) ∧ (a3 = a2 * r) ∧ (a4 = a3 * r)) → False :=
sorry

end geo_sequence_necessity_l390_390917


namespace find_c_value_l390_390389

def projection_condition (v u : ℝ × ℝ) (c : ℝ) : Prop :=
  let v := (5, c)
  let u := (3, 2)
  let dot_product := (v.fst * u.fst + v.snd * u.snd)
  let norm_u_sq := (u.fst^2 + u.snd^2)
  (dot_product / norm_u_sq) * u.fst = -28 / 13 * u.fst

theorem find_c_value : ∃ c : ℝ, projection_condition (5, c) (3, 2) c :=
by
  use -43 / 2
  unfold projection_condition
  sorry

end find_c_value_l390_390389


namespace Lawrence_walking_speed_l390_390665

def speed (distance time : ℝ) : ℝ := distance / time

theorem Lawrence_walking_speed :
  speed 4 1.33 ≈ 3.01 :=
by
  sorry

end Lawrence_walking_speed_l390_390665


namespace distance_between_points_l390_390538

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := 
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_between_points : distance 0 6 4 0 = 2 * Real.sqrt 13 := by
  sorry

end distance_between_points_l390_390538


namespace three_person_subcommittees_l390_390178

theorem three_person_subcommittees (n k : ℕ) (h1 : n = 8) (h2 : k = 3) : nat.choose n k = 56 := 
by
  rw [h1, h2]
  norm_num
  sorry

end three_person_subcommittees_l390_390178


namespace area_ratio_l390_390136

-- Defining the points A, B, C and P
variables (A B C P : Type) [add_group P] [vector_space ℝ P]

-- Given condition from the problem
def condition1 := 
    ∃ (α β : ℝ), α = (1/3) ∧ β = (1/4) ∧ (0 : P) = alpha • (B - A) + beta • (C - A) + (1 - alpha - beta) • (P - A)

-- Proving the ratio of the areas S1 : S2 : S3
theorem area_ratio 
  (h : condition1 A B C P) : S1 / S2 / S3 = 5 / 4 / 3 :=
sorry

end area_ratio_l390_390136


namespace explicit_form_correct_l390_390101

-- Define the original function form
def f (a b x : ℝ) := 4*x^3 + a*x^2 + b*x + 5

-- Given tangent line slope condition at x = 1
axiom tangent_slope : ∀ (a b : ℝ), (12 * 1^2 + 2 * a * 1 + b = -12)

-- Given the point (1, f(1)) lies on the tangent line y = -12x
axiom tangent_point : ∀ (a b : ℝ), (4 * 1^3 + a * 1^2 + b * 1 + 5 = -12)

-- Definition for the specific f(x) found in solution
def f_explicit (x : ℝ) := 4*x^3 - 3*x^2 - 18*x + 5

-- Finding maximum and minimum values on interval [-3, 1]
def max_value : ℝ := -76
def min_value : ℝ := 16

theorem explicit_form_correct : 
  ∃ a b : ℝ, 
  (∀ x, f a b x = f_explicit x) ∧ 
  (max_value = 16) ∧ 
  (min_value = -76) := 
by
  sorry

end explicit_form_correct_l390_390101


namespace solve_system_integers_l390_390721

noncomputable def log_base_3 (x : ℝ) : ℝ := log x / log 3

-- Define conditions
def condition1 (x y : ℤ) : Prop := (2^x + log_base_3 x = y^2)
def condition2 (x y : ℤ) : Prop := (2^y + log_base_3 y = x^2)

-- Define theorem to prove
theorem solve_system_integers (x y : ℤ) (hx1 : condition1 x y) (hx2 : condition2 x y) : x = 3 ∧ y = 3 := 
sorry

end solve_system_integers_l390_390721


namespace matrix_computation_l390_390317

variable (N : Matrix (Fin 2) (Fin 2) ℝ)
variable (v₁ v₂ v₃ v₄ : Vector (Fin 2) ℝ)

theorem matrix_computation :
    (N ⬝ (Vector.ofList [7, 0]) = Vector.ofList [17.5, 0]) :=
by
    have h₁: N ⬝ (Vector.ofList [3, -2]) = Vector.ofList [5, 1] := sorry
    have h₂: N ⬝ (Vector.ofList [-2, 4]) = Vector.ofList [0, -2] := sorry
    sorry

end matrix_computation_l390_390317


namespace percentage_of_girls_toast_marshmallows_l390_390069

theorem percentage_of_girls_toast_marshmallows :
  ∃ (percentage_of_girls : ℝ), 
    (96 : ℝ) > 0 → 
    (2/3) * 96 = 64 → 
    (1/3) * 96 = 32 → 
    (0.5) * 64 = 32 →
    56 - 32 = 24 → 
    percentage_of_girls = (24 / 32) * 100 →
    percentage_of_girls = 75 :=
by intros percentage_of_girls;
   assume h1 h2 h3 h4 h5;
   exact sorry

end percentage_of_girls_toast_marshmallows_l390_390069


namespace smallest_base_to_represent_124_with_three_digits_l390_390431

theorem smallest_base_to_represent_124_with_three_digits : 
  ∃ (b : ℕ), b^2 ≤ 124 ∧ 124 < b^3 ∧ ∀ c, (c^2 ≤ 124 ∧ 124 < c^3) → (5 ≤ c) :=
by
  sorry

end smallest_base_to_represent_124_with_three_digits_l390_390431


namespace find_expression_l390_390867

theorem find_expression : 1^567 + 3^5 / 3^3 - 2 = 8 :=
by
  sorry

end find_expression_l390_390867


namespace count_magic_numbers_less_than_130_l390_390907

theorem count_magic_numbers_less_than_130 : 
  ∃ (N : ℕ), (∀ (n : ℕ), N = n → (n < 130 ∧ (∃ m : ℕ, n ∣ 10^m))) ∧ N.card = 9 :=
by
  sorry

end count_magic_numbers_less_than_130_l390_390907


namespace percentage_sikhs_l390_390978

/- Define the total number of boys -/
def total_boys : Nat := 700

/- Define the percentage of boys that are Muslims -/
def percentage_muslims : Float := 44.0

/- Define the percentage of boys that are Hindus -/
def percentage_hindus : Float := 28.0

/- Define the number of boys from other communities -/
def other_communities : Nat := 126

/- Prove that the percentage of Sikh boys is 10% -/
theorem percentage_sikhs :
  (((total_boys - (percentage_muslims / 100 * total_boys).toNat - (percentage_hindus / 100 * total_boys).toNat - other_communities).toFloat / total_boys.toFloat) * 100).toInt = 10 := 
by
  sorry

end percentage_sikhs_l390_390978


namespace three_person_subcommittees_from_eight_l390_390166

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem three_person_subcommittees_from_eight (n k : ℕ) (h_n : n = 8) (h_k : k = 3) :
  combination n k = 56 := by
  sorry

end three_person_subcommittees_from_eight_l390_390166


namespace eccentricity_range_l390_390307

variables {a b : ℝ}
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

theorem eccentricity_range (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ (x y : ℝ), ellipse x y → distance x y 0 b ≤ 2 * b) :
  0 < (sqrt (a^2 - b^2) / a) ∧ (sqrt (a^2 - b^2) / a) ≤ sqrt 2 / 2 :=
sorry

end eccentricity_range_l390_390307


namespace line_equation_isosceles_triangle_l390_390905

theorem line_equation_isosceles_triangle 
  (x y : ℝ)
  (l : ℝ → ℝ → Prop)
  (h1 : l 3 2)
  (h2 : ∀ x y, l x y → (x = y ∨ x + y = 2 * intercept))
  (intercept : ℝ) :
  l x y ↔ (x - y = 1 ∨ x + y = 5) :=
by
  sorry

end line_equation_isosceles_triangle_l390_390905


namespace linear_function_no_fourth_quadrant_l390_390551

theorem linear_function_no_fourth_quadrant (k : ℝ) (hk : k > 2) : 
  ∀ x (hx : x > 0), (k-2) * x + k ≥ 0 :=
by
  sorry

end linear_function_no_fourth_quadrant_l390_390551


namespace average_time_other_classes_l390_390995

theorem average_time_other_classes (total_hours_per_day : ℕ) (lunch_break_minutes : ℕ) (total_classes : ℕ) 
  (history_chemistry_minutes : ℕ) (math_minutes : ℕ) (history_chemistry_hours : ℕ) (math_hours : ℕ) 
  (other_classes : ℕ) : total_hours_per_day = 9 → 
                        lunch_break_minutes = 45 → 
                        total_classes = 10 → 
                        history_chemistry_minutes = history_chemistry_hours * 60 → 
                        math_minutes = math_hours * 60 → 
                        history_chemistry_hours = 2 → 
                        math_hours = 1.5 → 
                        other_classes = total_classes - 3 → 
                        (total_hours_per_day * 60 - lunch_break_minutes - history_chemistry_minutes - math_minutes) / other_classes = 40.71 :=
by
  sorry

end average_time_other_classes_l390_390995


namespace good_set_midpoint_condition_l390_390339

structure GoodSet (S : Set ℝ × ℝ) :=
(midpoint_in_S : ∀ a b, a ∈ S ∧ b ∈ S → (a + b) / 2 ∈ S)
(rotation_invariant : ∀ θ, ∀ (p : ℝ × ℝ), p ∈ S → (λ q, (cos θ • (fst q - fst p) - sin θ • (snd q - snd p) + fst p, sin θ • (fst q - fst p) + cos θ • (snd q - snd p) + snd p)) '' S = S)

theorem good_set_midpoint_condition (r : ℚ) (h_r_range: r ∈ set.Icc (-1:ℚ) 1) :
  (∃ n : ℤ, r = (4 * n - 1) / (4 * n)) ↔
  ∀ S : Set (ℝ × ℝ), GoodSet S → (∀ a b ∈ S, (a + b) / 2 ∈ S) :=
sorry

end good_set_midpoint_condition_l390_390339


namespace eccentricity_range_l390_390310

variables {a b : ℝ}
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

theorem eccentricity_range (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ (x y : ℝ), ellipse x y → distance x y 0 b ≤ 2 * b) :
  0 < (sqrt (a^2 - b^2) / a) ∧ (sqrt (a^2 - b^2) / a) ≤ sqrt 2 / 2 :=
sorry

end eccentricity_range_l390_390310


namespace range_of_eccentricity_l390_390276

variables {a b c e : ℝ}

def ellipse (x y : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)
def upper_vertex := b > 0
def distance_condition (x_0 y_0 : ℝ) := (x_0^2 + (y_0 - b)^2 ≤ (2 * b) ^ 2)
def eccentricity := e = c / a
def ellipse_condition := a = real.sqrt (b^2 + c^2)

theorem range_of_eccentricity (h1 : a > b) (h2 : upper_vertex) 
  (h3 : ∀ x_0 y_0, ellipse x_0 y_0 → distance_condition x_0 y_0)
  (h4 : ellipse_condition) :
  0 < e ∧ e ≤ real.sqrt 2 / 2 := 
by
  sorry

end range_of_eccentricity_l390_390276


namespace least_positive_integer_factorial_divisor_l390_390417

theorem least_positive_integer_factorial_divisor : 
  ∃ (n : ℕ), 0 < n ∧ (5005 ∣ factorial n) ∧ ∀ m : ℕ, 0 < m ∧ (5005 ∣ factorial m) → n ≤ m :=
by
  sorry

end least_positive_integer_factorial_divisor_l390_390417


namespace commercials_per_hour_l390_390373

theorem commercials_per_hour (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) : ∃ x : ℝ, x = (1 - p) * 60 := 
sorry

end commercials_per_hour_l390_390373


namespace alternating_sum_of_squares_l390_390874

theorem alternating_sum_of_squares (n : ℕ) : 
  ∑ k in finset.range n, ((2 * k + 1)^2 - (2 * (k + 1))^2) = -n * (2 * n + 1) := 
sorry

end alternating_sum_of_squares_l390_390874


namespace find_alpha_l390_390591

theorem find_alpha (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) (3 * Real.pi / 2))
  (h2 : ∃ k : ℝ, (Real.cos α, Real.sin α) = k • (-3, -3)) :
  α = 3 * Real.pi / 4 :=
by
  sorry

end find_alpha_l390_390591


namespace determine_abs_d_l390_390852

noncomputable def Q (d : ℂ) : Polynomial ℂ :=
  (X^2 - C 3 * X + C 3) * (X^2 - C d * X + C 9) * (X^2 - C 6 * X + C 18)

theorem determine_abs_d (d : ℂ) (h : (Q d).rootSet ℂ).card = 4 : abs d = 9 :=
sorry

end determine_abs_d_l390_390852


namespace basis_exists_l390_390953

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

def is_basis (s : set V) : Prop := ∀ v ∈ s, v ≠ 0 ∧ (∀ u ∈ s, u ≠ v → ¬ (∃ (c₁ c₂ : ℝ), u = c₁ • v + c₂ • v))

noncomputable def basis_abc : Set V := {a, b, c}

theorem basis_exists :
  is_basis basis_abc →
  is_basis ({c, a + b, a - b} : set V) := 
by
  sorry

end basis_exists_l390_390953


namespace factorize_problem1_factorize_problem2_factorize_problem3_factorize_problem4_l390_390095

-- Problem 1: Prove equivalence for factorizing -2a^2 + 4a.
theorem factorize_problem1 (a : ℝ) : -2 * a^2 + 4 * a = -2 * a * (a - 2) := 
by sorry

-- Problem 2: Prove equivalence for factorizing 4x^3 y - 9xy^3.
theorem factorize_problem2 (x y : ℝ) : 4 * x^3 * y - 9 * x * y^3 = x * y * (2 * x + 3 * y) * (2 * x - 3 * y) := 
by sorry

-- Problem 3: Prove equivalence for factorizing 4x^2 - 12x + 9.
theorem factorize_problem3 (x : ℝ) : 4 * x^2 - 12 * x + 9 = (2 * x - 3)^2 := 
by sorry

-- Problem 4: Prove equivalence for factorizing (a+b)^2 - 6(a+b) + 9.
theorem factorize_problem4 (a b : ℝ) : (a + b)^2 - 6 * (a + b) + 9 = (a + b - 3)^2 := 
by sorry

end factorize_problem1_factorize_problem2_factorize_problem3_factorize_problem4_l390_390095


namespace root_expression_value_l390_390623

theorem root_expression_value (m : ℝ) (h : 2 * m^2 + 3 * m - 1 = 0) : 4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end root_expression_value_l390_390623


namespace possible_ticket_prices_l390_390048

-- Definitions and conditions
def ticket_price (x : ℕ) : Prop :=
  ∃ (n m : ℕ), 48 = n * x ∧ 64 = m * x

-- Theorem statement
theorem possible_ticket_prices : 
  (finset.univ.filter (λ x, ticket_price x)).card = 5 :=
sorry

end possible_ticket_prices_l390_390048


namespace max_coins_value_3_l390_390013

/-- A statement for the maximum number of coins valued at 3 satisfying given conditions. -/
theorem max_coins_value_3 (n : ℕ) (coins : Fin n → ℕ) (h₀ : n = 2001)
  (h₁ : ∀ i, coins i = 1 ∨ coins i = 2 ∨ coins i = 3)
  (h₂ : ∀ i j, coins i = 1 → coins j = 1 → i ≠ j → (i < j → (j - i > 1)) ∧ (j < i → (i - j > 1)))
  (h₃ : ∀ i j, coins i = 2 → coins j = 2 → i ≠ j → (i < j → (j - i > 2)) ∧ (j < i → (i - j > 2)))
  (h₄ : ∀ i j, coins i = 3 → coins j = 3 → i ≠ j → (i < j → (j - i > 3)) ∧ (j < i → (i - j > 3)))
  : ∃ k, (∀ i, coins i = 3 → ∃ j < k, i = j) ∧ k = 501 :=
by
  sorry

end max_coins_value_3_l390_390013


namespace possible_ticket_prices_l390_390049

-- Definitions and conditions
def ticket_price (x : ℕ) : Prop :=
  ∃ (n m : ℕ), 48 = n * x ∧ 64 = m * x

-- Theorem statement
theorem possible_ticket_prices : 
  (finset.univ.filter (λ x, ticket_price x)).card = 5 :=
sorry

end possible_ticket_prices_l390_390049


namespace number_of_ordered_pairs_l390_390040

noncomputable def count_valid_pairs : ℕ :=
  let a := 3
  let valid_pairs := ([(b, c) |
    b ← [1..300],
    c ← [1..300],
    1 ≤ b ∧ b < c ∧
    a * b * c + 2 * (a * b + b * c + c * a) = 300
    ])
  valid_pairs.length

theorem number_of_ordered_pairs : count_valid_pairs = 2 := sorry

end number_of_ordered_pairs_l390_390040


namespace calc_expression_l390_390508

theorem calc_expression : (0.125^-((1:ℚ)/3) - (64/27:ℚ)^0 - (Real.log 25 / Real.log 2) * (Real.log 4 / Real.log 3) * (Real.log 9 / Real.log 5)) = -7 := 
begin
  sorry
end

end calc_expression_l390_390508


namespace find_x_for_sin_cos_l390_390535

theorem find_x_for_sin_cos (x : ℝ) (h1 : 0 ≤ x) (h2 : x < 2 * Real.pi) (h3 : Real.sin x + Real.cos x = Real.sqrt 2) : x = Real.pi / 4 :=
sorry

end find_x_for_sin_cos_l390_390535


namespace range_of_eccentricity_of_ellipse_l390_390257

-- Definitions for the problem.
def is_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1) 

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

def on_upper_vertex (b : ℝ) : (ℝ × ℝ) := (0, b)

-- The statement to prove the range of eccentricity.
theorem range_of_eccentricity_of_ellipse (a b : ℝ) (x y : ℝ) (e : ℝ) :
  is_on_ellipse a b x y →
  (distance x y 0 b ≤ 2 * b) →
  ∃ e, (e = (1 - (b^2 / a^2))^(1/2) ∧ (0 < e) ∧ (e ≤ (sqrt 2)/2)) :=
by
  sorry

end range_of_eccentricity_of_ellipse_l390_390257


namespace curve_is_line_l390_390741

noncomputable def curve_representation (x y : ℝ) : Prop :=
  (2 * x + 3 * y - 1) * (-1) = 0

theorem curve_is_line (x y : ℝ) (h : curve_representation x y) : 2 * x + 3 * y - 1 = 0 :=
by
  sorry

end curve_is_line_l390_390741


namespace expression_value_l390_390340

theorem expression_value (a b c : ℝ) (h : a + b + c = 1) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let y1 := a^2 + bc
  let y2 := b^2 + ac
  let y3 := c^2 + ab
  in (a^2 * b^2 / (y1 * y2)) + (a^2 * c^2 / (y1 * y3)) + (b^2 * c^2 / (y2 * y3)) = 1 :=
by
  sorry

end expression_value_l390_390340


namespace foci_of_ellipse_l390_390098

-- Definitions as per the problem conditions
def a : ℝ := Real.sqrt 2
def b : ℝ := 1
def ellipse_eq (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Main theorem statement
theorem foci_of_ellipse :
  (∃ c : ℝ, c = Real.sqrt (a^2 - b^2) ∧
            (∀ x y : ℝ, ellipse_eq x y → (x, y) = (c, 0) ∨ (x, y) = (-c, 0))) :=
  sorry

end foci_of_ellipse_l390_390098


namespace hexagon_triangle_count_l390_390472

-- Definitions based on problem conditions
def numPoints : ℕ := 7
def totalTriangles := Nat.choose numPoints 3
def collinearCases : ℕ := 3

-- Proof problem
theorem hexagon_triangle_count : totalTriangles - collinearCases = 32 :=
by
  -- Calculation is expected here
  sorry

end hexagon_triangle_count_l390_390472


namespace correct_operation_l390_390787

theorem correct_operation (x y : ℝ) : (-x - y) ^ 2 = x ^ 2 + 2 * x * y + y ^ 2 :=
sorry

end correct_operation_l390_390787


namespace highest_water_level_in_A_l390_390527

variables (Vase : Type) (height : Vase → ℝ) (volume : Vase → ℝ) (water_pour : Vase → ℝ)

-- Conditions: Each vase has the same height and a volume of 1 liter
constant same_height : ∀ (v1 v2 : Vase), height v1 = height v2
constant volume_one : ∀ (v : Vase), volume v = 1

-- Condition: Half a liter of water is poured into each vase
constant water_half : ∀ (v : Vase), water_pour v = 0.5

-- Define the Vases
constant vase_A vase_B vase_C vase_D vase_E : Vase

-- Conclusion: The water level is highest in Vase A
theorem highest_water_level_in_A : ∀ (lvl : Vase → ℝ),
  (lvl vase_A > lvl vase_B) ∧ 
  (lvl vase_A > lvl vase_C) ∧ 
  (lvl vase_A > lvl vase_D) ∧ 
  (lvl vase_A > lvl vase_E) := sorry

end highest_water_level_in_A_l390_390527


namespace base_k_conversion_l390_390586

theorem base_k_conversion (k : ℕ) (hk : 4 * k + 4 = 36) : 6 * 8 + 7 = 55 :=
by
  -- Proof skipped
  sorry

end base_k_conversion_l390_390586


namespace total_number_of_games_l390_390045

theorem total_number_of_games (teams : ℕ) (games_per_pair : ℕ) (h1 : teams = 10) (h2 : games_per_pair = 2) :
  (teams * (teams - 1) / 2) * games_per_pair = 90 := by
  -- conditions are represented by h1, h2
  rw [h1, h2]
  sorry

end total_number_of_games_l390_390045


namespace sum_of_evens_from_10_to_31_l390_390104

-- Define the predicate for even numbers
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Sum of all even numbers in the range [10, 31]
def sum_of_evens_in_range : ℕ :=
  Finset.sum (Finset.filter is_even (Finset.Icc 10 31)) id

theorem sum_of_evens_from_10_to_31 : sum_of_evens_in_range = 220 := 
by
  sorry

end sum_of_evens_from_10_to_31_l390_390104


namespace inequality_abc_equality_condition_l390_390341

theorem inequality_abc (a b c : ℝ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (ab : ℝ) / (c - 1) + (bc : ℝ) / (a - 1) + (ca : ℝ) / (b - 1) ≥ 12 :=
sorry

theorem equality_condition (a b c : ℝ) (h_a : a > 1) (h_b : b > 1) (h_c : c > 1) :
  (ab : ℝ) / (c - 1) + (bc : ℝ) / (a - 1) + (ca : ℝ) / (b - 1) = 12 ↔ a = 2 ∧ b = 2 ∧ c = 2 :=
sorry

end inequality_abc_equality_condition_l390_390341


namespace unique_valid_sequence_exists_l390_390673

def even_numbers_up_to_20 := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
def immediate_predecessor : Nat → Nat
| 2  := 0
| n  := if n % 2 = 0 then n - 2 else 0

def largest_odd_predecessor : Nat → Nat
| n := if n % 2 = 0 && n % 4 = 0 then n - 1 else 0

def valid_sequence (seq : List Nat) : Prop :=
  seq = [15, 13, 11, 9, 7, 5, 3, 1, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]

theorem unique_valid_sequence_exists :
  ∃ seq : List Nat, valid_sequence seq ∧ seq = even_numbers_up_to_20 := 
by
  sorry

end unique_valid_sequence_exists_l390_390673


namespace base7_difference_l390_390099

theorem base7_difference (a b : ℕ) (h₁ : a = 12100) (h₂ : b = 3666) :
  ∃ c, c = 1111 ∧ (a - b = c) := by
sorry

end base7_difference_l390_390099


namespace first_three_decimal_digits_of_power_l390_390415

theorem first_three_decimal_digits_of_power (x : ℕ) (y : ℕ) (z : ℝ) (h : x > 0) :
  let n := (x^1001 + 1 : ℝ)
  in (n^(5/3)) - real.floor (n^(5/3)) = 0.333 :=
by
  sorry

end first_three_decimal_digits_of_power_l390_390415


namespace scientific_notation_l390_390645

theorem scientific_notation (n : ℝ) (h : n = 40.9 * 10^9) : n = 4.09 * 10^10 :=
by sorry

end scientific_notation_l390_390645


namespace parabola_equation_l390_390923

theorem parabola_equation {F : ℝ × ℝ} (A B C : ℝ × ℝ) (p : ℝ) 
    (hF : F = (p / 2, 0)) 
    (hVertex : (0, 0)) 
    (hSumZero : (fst A + fst B + fst C = 0)) 
    (hDistanceSum : (dist F A + dist F B + dist F C = 6)) :
    (∀ x y, y^2 = 8 * x ↔ 
    ∃ a : ℝ, a = p / 2 ∧ (x, y) ∈ parabola (0, 0) a) :=
sorry

end parabola_equation_l390_390923


namespace new_person_weight_l390_390377

theorem new_person_weight (W : ℝ) (x : ℝ) : 
  (8 : ℝ) * 2.5 = x - 70 → x = 90 :=
by
  intro h
  have h1 : 8 * 2.5 = 20 := by norm_num
  have h2 : x - 70 = 20 := by rw [←h, h1]
  have h3 : x = 90 := by linarith
  exact h3

end new_person_weight_l390_390377


namespace even_odd_matching_diff_theorem_l390_390799

-- Defining a natural number N.
variable (N : ℕ)

-- Defining what it means to have an even or odd matching on a circle with 2N points
/--
 Given a circle with 2N marked points, where no more than two chords with endpoints at the marked
 points pass through any point inside the circle, a matching is defined as a set of N chords with 
 endpoints at the marked points such that each marked point is an endpoint of exactly one chord. 
 A matching is called even if the number of intersection points of its chords is even, and odd otherwise.
--/
def even_odd_matching_diff := ∀ N : ℕ, 
  let even_matchings : set (matching 2N) := {m | is_even (num_intersections m)} in
  let odd_matchings : set (matching 2N) := {m | ¬ is_even (num_intersections m)} in
  (even_matchings.card - odd_matchings.card) = 1

-- The main theorem to prove.
theorem even_odd_matching_diff_theorem : even_odd_matching_diff N := sorry

end even_odd_matching_diff_theorem_l390_390799


namespace smallest_prime_8_less_than_square_l390_390423

theorem smallest_prime_8_less_than_square :
  ∃ p : ℕ, (∃ n : ℤ, p = n^2 - 8) ∧ Nat.Prime p ∧ p > 0 ∧ (∀ q : ℕ, (∃ m : ℤ, q = m^2 - 8) ∧ Nat.Prime q → q ≥ p) :=
sorry

end smallest_prime_8_less_than_square_l390_390423


namespace eccentricity_hyperbola_l390_390033

variables {a b c e : ℝ}
variables (a_pos : 0 < a) (b_pos : 0 < b)
def h_eq := ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1
def right_focus := c = real.sqrt (a^2 + b^2)
def A := (c, -b^2 / a)
def B := (c, b^2 / a)
def D := (0, b)
def right_angle_triangle := ∀ (a b c : ℝ × ℝ), ∠ a b c = 90 * (π / 180)

theorem eccentricity_hyperbola :
  (h_eq a b)
  → (right_focus a b c) 
  → (A a b c)
  → (B a b c)
  → (D b)
  → ((right_angle_triangle D B A) ∨ (right_angle_triangle A D B))
  → (e = real.sqrt 2 ∨ e = real.sqrt (2 + real.sqrt 2)) :=
sorry

end eccentricity_hyperbola_l390_390033


namespace daily_avg_for_entire_month_is_correct_l390_390637

-- conditions
def avg_first_25_days := 63
def days_first_25 := 25
def avg_last_5_days := 33
def days_last_5 := 5
def total_days := days_first_25 + days_last_5

-- question: What is the daily average for the entire month?
theorem daily_avg_for_entire_month_is_correct : 
  (avg_first_25_days * days_first_25 + avg_last_5_days * days_last_5) / total_days = 58 := by
  sorry

end daily_avg_for_entire_month_is_correct_l390_390637


namespace rhombus_area_in_circle_l390_390747

theorem rhombus_area_in_circle (a : ℝ) (h : a = 16) :
  let d1 := a
  let d2 := a
  let area := (d1 * d2) / 2
  area = 128 :=
by
  let d1 := a
  let d2 := a
  let area := (d1 * d2) / 2
  have ha : area = 128 := by {
    calc
      area = (16 * 16) / 2 : by rw [h]
      ... = 256 / 2      : by rw [mul_self]
      ... = 128          : by norm_num
  }
  exact ha

end rhombus_area_in_circle_l390_390747


namespace three_person_subcommittees_l390_390181

theorem three_person_subcommittees (n k : ℕ) (h1 : n = 8) (h2 : k = 3) : nat.choose n k = 56 := 
by
  rw [h1, h2]
  norm_num
  sorry

end three_person_subcommittees_l390_390181


namespace goldfish_cost_graph_l390_390165

theorem goldfish_cost_graph :
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 15) →
    (if n ≤ 10 then (cost n = 20 * n) else cost n = 18 * n)
  → (graph ∈ {(1, 20), (10, 200)} ∪ {(11, 198), (15, 270)}) 
  → (graph.description = "two connected line segments") := 
sorry

end goldfish_cost_graph_l390_390165


namespace sequence_bounds_l390_390158

noncomputable def a_sequence (θ : ℝ) (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 - 2 * sin θ ^ 2 * cos θ ^ 2 ∧
  ∀ n : ℕ, a (n + 2) = a (n + 1) - a n * sin θ ^ 2 * cos θ ^ 2

theorem sequence_bounds (θ : ℝ) (a : ℕ → ℝ) (n : ℕ) (hθ : θ ∈ Ioo 0 (π / 2))
  (ha : a_sequence θ a) :
  (1 / 2 ^ (n - 1) ≤ a n ∧ a n ≤ 1 - sin (2 * θ) ^ n * (1 - (1 / 2 ^ (n - 1)))) :=
sorry

end sequence_bounds_l390_390158


namespace palindrome_count_on_clock_l390_390018

def is_palindrome (n : ℕ) : Prop :=
  let digits := toDigits 10 n
  digits = digits.reverse

def valid_hour (h : ℕ) : Prop :=
  h < 24

def valid_minute (m : ℕ) : Prop :=
  m < 60

theorem palindrome_count_on_clock : ∑ h in finset.range 24, ∑ m in finset.range 60, ite (is_palindrome (h * 100 + m)) 1 0 = 78 :=
by
  sorry

end palindrome_count_on_clock_l390_390018


namespace remainder_when_divided_by_l390_390418

def P (x : ℤ) : ℤ := 5 * x^8 - 2 * x^7 - 8 * x^6 + 3 * x^4 + 5 * x^3 - 13
def D (x : ℤ) : ℤ := 3 * (x - 3)

theorem remainder_when_divided_by (x : ℤ) : P 3 = 23364 :=
by {
  -- This is where the calculation steps would go, but we're omitting them.
  sorry
}

end remainder_when_divided_by_l390_390418


namespace chord_length_proof_l390_390742

variable (r : ℝ)
variable (O1 O2 M N K : Point)
variables (ω1 ω2 : Circle)

def distance (A B : Point) : ℝ := sorry
def radius (c : Circle) : ℝ := sorry
def tangent (c : Circle) (p : Point) : Prop := sorry
def intersects (c : Circle) (A B : Point) : Prop := sorry
def midpoint (A B : Point) : Point := sorry
def len (A B : Point) : ℝ := sorry

axiom circle_distance : distance O1 O2 = 10 * r
axiom circle_radius_1 : radius ω1 = 5 * r
axiom circle_radius_2 : radius ω2 = 6 * r
axiom line_intersects_circle_1 : intersects ω1 M N
axiom line_tangent_circle_2 : tangent ω2 K
axiom chord_proportion : 2 * (len M N / 2) = len M N

def chord_length_equals_condition : Prop :=
  len M N = 2 * r * Real.sqrt 21

theorem chord_length_proof : chord_length_equals_condition :=
by
  sorry

end chord_length_proof_l390_390742


namespace inequality_proof_l390_390584

theorem inequality_proof (a b c : ℝ) 
    (ha : a > 1) (hb : b > 1) (hc : c > 1) :
    (a^2 / (b - 1)) + (b^2 / (c - 1)) + (c^2 / (a - 1)) ≥ 12 :=
by {
    sorry
}

end inequality_proof_l390_390584


namespace problem_l390_390147

noncomputable def f (x a b : ℝ) : ℝ :=
  (1/3) * x^3 + a * x^2 + b * x

theorem problem (a b : ℝ) :
  (∀ x : ℝ, f x a b = (1/3) * x^3 + a * x^2 + b * x) ∧
  (maximizes (f (-3) a b) (f (-3) a b)) ∧
  (∀ x ∈ set.Icc (-3 : ℝ) 3, f x a b ≤ 9 ∧ f x a b ≥ -5/3) →
  a = 1 ∧ b = -3 ∧ (∀ x ∈ set.Icc (-3 : ℝ) 3, f x a b ≤ 9 ∧ f x a b ≥ -5/3) :=
begin
  sorry
end

end problem_l390_390147


namespace fraction_of_p_amount_l390_390797

-- Define the constant amounts and the variables
def p : ℝ := 56
def diff : ℝ := 42
def f : ℝ

-- State the conditions as hypotheses
theorem fraction_of_p_amount :
  ∃ (f : ℝ), p = 2 * f * p + diff → f = 1 / 8 :=
by
  -- Introduce the given relationships
  assume h : p = 2 * f * p + diff,

  -- Rewrite the necessary parts
  sorry

end fraction_of_p_amount_l390_390797


namespace sum_of_proper_divisors_180_l390_390883

theorem sum_of_proper_divisors_180 : 
  let n := 180 in 
  let sum_of_divisors := (1 + 2 + 4) * (1 + 3 + 9) * (1 + 5) in
  sum_of_divisors - n = 366 :=
by
  sorry

end sum_of_proper_divisors_180_l390_390883


namespace eccentricity_range_l390_390314

variables {a b : ℝ}
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

theorem eccentricity_range (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ (x y : ℝ), ellipse x y → distance x y 0 b ≤ 2 * b) :
  0 < (sqrt (a^2 - b^2) / a) ∧ (sqrt (a^2 - b^2) / a) ≤ sqrt 2 / 2 :=
sorry

end eccentricity_range_l390_390314


namespace area_triang_ABC_l390_390693

variables (A B C F D G : Type*)
variables [EuclideanGeometry Triangle Point]
variables (AF BD CE : medians)
variables (AF_perp_BD : Perpendicular AF BD) (AF_inter_CE : IntersectAt AF CE G)
variables (BD_inter_CE : IntersectAt BD CE G)
variables (lengthAF : Real) (lengthBD : Real)
variables (medAF : AssignedLength AF 12) (medBD : AssignedLength BD 16)

theorem area_triang_ABC :
  area (Triangle.mk A B C) = 128 :=
sorry

end area_triang_ABC_l390_390693


namespace integer_fahrenheit_temps_count_l390_390744

theorem integer_fahrenheit_temps_count :
  let C (F : ℤ) : ℤ := Int.round ((5 / 9:ℚ) * (F - 32))
  let F' (F : ℤ) : ℤ := Int.round ((9 / 5:ℚ) * (C F)) + 32
  ∃ count : ℕ, count = 539 ∧ 
  (count = ∑ F in Finset.Icc (32 : ℤ) 1000, if F = F' F then 1 else 0) :=
by
  sorry

end integer_fahrenheit_temps_count_l390_390744


namespace min_red_hair_students_l390_390015

theorem min_red_hair_students (B N R : ℕ) 
  (h1 : B + N + R = 50)
  (h2 : N ≥ B - 1)
  (h3 : R ≥ N - 1) :
  R = 17 := sorry

end min_red_hair_students_l390_390015


namespace ellipse_equation_slope_range_l390_390592

theorem ellipse_equation (a b : ℝ) (h1: a > b) (h2 : b > 0) (h3 : a = 2) (h4 : b = sqrt 3) :
  ∀ x y, (x, y) ∈ set_of (λ ⟨x, y⟩, x^2 / a^2 + y^2 / b^2 = 1) ↔ (x, y) ∈ set_of (λ ⟨x, y⟩, x^2 / 4 + y^2 / 3 = 1) := sorry

theorem slope_range (k : ℝ) (h : (∃ m : ℝ, m^2 = (3 + 4*k^2)^2 / (8 * abs k) ∧ 4*k^2 - (3 + 4*k^2)^2 / (8 * abs k) + 3 > 0)) :
  - (3 / 2) < k ∧ k < - (1 / 2) ∨ (1 / 2) < k ∧ k < (3 / 2) := sorry

end ellipse_equation_slope_range_l390_390592


namespace back_seat_people_l390_390639

-- Define the problem conditions

def leftSideSeats : ℕ := 15
def seatDifference : ℕ := 3
def peoplePerSeat : ℕ := 3
def totalBusCapacity : ℕ := 88

-- Define the formula for calculating the people at the back seat
def peopleAtBackSeat := 
  totalBusCapacity - ((leftSideSeats * peoplePerSeat) + ((leftSideSeats - seatDifference) * peoplePerSeat))

-- The statement we need to prove
theorem back_seat_people : peopleAtBackSeat = 7 :=
by
  sorry

end back_seat_people_l390_390639


namespace distance_skew_lines_D1B_A1E_distance_B1_to_plane_A1BE_distance_D1C_to_plane_A1BE_distance_planes_A1DB_D1CB1_l390_390569

-- Define the cube with specified points and side length
def Point := EuclideanSpace ℝ (_ : Fin 3)  -- 3-dimensional point

def A1 : Point := ![0, 0, 0]
def B : Point := ![1, 1, -1]
def C1 : Point := ![1, 0, 1]
def D1 : Point := ![0, 1, 1]
def E : Point := ![-1, 0.5, 0]

-- Part 1: Distance between skew lines D1B and A1E
theorem distance_skew_lines_D1B_A1E :
  Euclidean.distance_skew_lines (D1, B) (A1, E) = (Real.sqrt 14) / 14 :=
sorry

-- Part 2: Distance from B1 to the plane A1BE
def B1 : Point := ![0,1,0]
def A1B : Point := ![1,1,-1]
def A1E : Point := ![-1,0.5,0]

theorem distance_B1_to_plane_A1BE :
  Euclidean.distance_point_to_plane B1 A1 A1B A1E = 2 / 3 :=
sorry

-- Part 3: Distance from D1C to the plane A1BE
def D1C : Point := ![1, 0, 1] -- assuming D1C ends at C

theorem distance_D1C_to_plane_A1BE :
  Euclidean.distance_point_to_plane D1C A1 A1B A1E = 1 / 3 :=
sorry

-- Part 4: Distance between planes A1DB and D1CB1
def A1D : Point := ![-1,-1,0]
def D1CB1 : Point := ![0.5,0.5,1]

theorem distance_planes_A1DB_D1CB1 :
  Euclidean.distance_between_planes A1D B D1CB1 B1 = (Real.sqrt 3) / 3 :=
sorry

end distance_skew_lines_D1B_A1E_distance_B1_to_plane_A1BE_distance_D1C_to_plane_A1BE_distance_planes_A1DB_D1CB1_l390_390569


namespace wednesday_more_than_half_millet_l390_390354

namespace BirdFeeder

-- Define the initial conditions
def initial_amount_millet (total_seeds : ℚ) : ℚ := 0.4 * total_seeds
def initial_amount_other (total_seeds : ℚ) : ℚ := 0.6 * total_seeds

-- Define the daily consumption
def eaten_millet (millet : ℚ) : ℚ := 0.2 * millet
def eaten_other (other : ℚ) : ℚ := other

-- Define the seed addition every other day
def add_seeds (day : ℕ) (seeds : ℚ) : Prop :=
  day % 2 = 1 → seeds = 1

-- Define the daily update of the millet and other seeds in the feeder
def daily_update (day : ℕ) (millet : ℚ) (other : ℚ) : ℚ × ℚ :=
  let remaining_millet := (1 - 0.2) * millet
  let remaining_other := 0
  if day % 2 = 1 then
    (remaining_millet + initial_amount_millet 1, initial_amount_other 1)
  else
    (remaining_millet, remaining_other)

-- Define the main property to prove
def more_than_half_millet (day : ℕ) (millet : ℚ) (other : ℚ) : Prop :=
  millet > 0.5 * (millet + other)

-- Define the theorem statement
theorem wednesday_more_than_half_millet
  (millet : ℚ := initial_amount_millet 1)
  (other : ℚ := initial_amount_other 1) :
  ∃ day, day = 3 ∧ more_than_half_millet day millet other :=
  by
  sorry

end BirdFeeder

end wednesday_more_than_half_millet_l390_390354


namespace three_person_subcommittees_from_eight_l390_390169

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem three_person_subcommittees_from_eight (n k : ℕ) (h_n : n = 8) (h_k : k = 3) :
  combination n k = 56 := by
  sorry

end three_person_subcommittees_from_eight_l390_390169


namespace figure_can_form_square_l390_390567

noncomputable def can_cut_into_square (figure : Type) : Prop :=
  ∃ (part1 part2 : Type), (figure = part1 ∪ part2) ∧ (disjoint part1 part2) ∧ 
  ∃ (rotation1 rotation2 flip1 flip2 : (part1 ∪ part2) → (part1 ∪ part2)),
  is_square (rotation1 part1 ∪ rotation2 part2 ∪ flip1 part1 ∪ flip2 part2)

theorem figure_can_form_square (figure : Type) (hole : Type) :
  (has_hole figure hole) → can_cut_into_square figure :=
sorry

end figure_can_form_square_l390_390567


namespace sequence_a_n_correctness_l390_390602

theorem sequence_a_n_correctness (a : ℕ → ℚ) (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, n > 0 → 2 * a (n + 1) = 2 * a n + 1) : a 2 = 1.5 := by
  sorry

end sequence_a_n_correctness_l390_390602


namespace smallest_positive_angle_l390_390617

theorem smallest_positive_angle (α : ℝ) (h : α = 2012) : 
  ∃ θ : ℝ, 0 < θ ∧ θ < 360 ∧ (∀ n : ℤ, θ = α - n * 360) ∧ θ = 212 :=
by
  use 212
  split
  { exact 212 }
  split
  { exact 148 }
  split
  { intro n
    exact 0 }
  { exact α }


end smallest_positive_angle_l390_390617


namespace devin_teaching_years_l390_390767

theorem devin_teaching_years (total_years : ℕ) (tom_years : ℕ) (devin_years : ℕ) 
  (half_tom_years : ℕ)
  (h1 : total_years = 70) 
  (h2 : tom_years = 50)
  (h3 : total_years = tom_years + devin_years) 
  (h4 : half_tom_years = tom_years / 2) : 
  half_tom_years - devin_years = 5 :=
by
  sorry

end devin_teaching_years_l390_390767


namespace part_a_part_b_l390_390701

theorem part_a (r : ℝ) (x1 y1 x2 y2 : ℚ) :
  let M₁ := (x1, y1)
  let M₂ := (x2, y2)
  let M := (real.sqrt 2, real.sqrt 3)
  (M₁ = M₂ ∨ ¬(x1 - real.sqrt 2) ^ 2 + (y1 - real.sqrt 3) ^ 2 = r ^ 2 ∨ ¬(x2 - real.sqrt 2) ^ 2 + (y2 - real.sqrt 3) ^ 2 = r ^ 2) :=
begin
  sorry  -- Proof goes here
end

theorem part_b : ∃ r : ℝ, let M := (real.sqrt 2, real.sqrt 3) in
  (set_of_interior_points : set (ℤ × ℤ)) :=
begin
  sorry  -- Proof goes here
end

end part_a_part_b_l390_390701


namespace length_of_first_train_l390_390770

noncomputable def length_first_train
  (speed_train1_kmh : ℝ)
  (speed_train2_kmh : ℝ)
  (time_sec : ℝ)
  (length_train2_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed_train1_kmh + speed_train2_kmh) * (1000 / 3600)
  let total_distance_m := relative_speed_mps * time_sec
  total_distance_m - length_train2_m

theorem length_of_first_train :
  length_first_train 80 65 7.82006405004841 165 = 150.106201 :=
  by
  -- Proof steps would go here.
  sorry

end length_of_first_train_l390_390770


namespace product_of_divisors_of_a_l390_390345

theorem product_of_divisors_of_a {a b : ℕ} (h: ∀ n, n = 2005 → 
  a = (∏ d in (finset.filter (λ x, n % x = 0) (finset.range (n+1))), d) ∧ 
  b = (∏ d in (finset.filter (λ x, a % x = 0) (finset.range (a+1))), d)) : b = 2005^9 :=
by
  sorry

end product_of_divisors_of_a_l390_390345


namespace find_integer_x_l390_390494

theorem find_integer_x (x : ℤ) :
  1 < x ∧ x < 9 ∧
  2 < x ∧ x < 15 ∧
  -1 < x ∧ x < 7 ∧
  0 < x ∧ x < 4 ∧
  x + 1 < 5 → 
  x = 3 :=
by
  sorry

end find_integer_x_l390_390494


namespace sinx_intersect_axes_and_lines_l390_390515

def y_sin_x := λ x : ℝ, sin x

theorem sinx_intersect_axes_and_lines :
  (∃ p : ℝ × ℝ, p = (0, sin 0)) ∧                               -- intersects the y-axis at (0, 0)
  (∀ c : ℝ, ∃ y : ℝ, y = sin c) ∧                               -- intersects all vertical lines x = c
  (∀ k : ℤ, ∃ p : ℝ × ℝ, p = (k * π, 0))                        -- intersects the x-axis at multiples of π
  := by
  split
  case a : _ =>
    apply Exists.intro (0, 0)
    exact rfl
  case b : _ =>
    exact fun c, Exists.intro (sin c) rfl
  case c : _ =>
    exact fun k, Exists.intro (k * π, 0) rfl
  done

end sinx_intersect_axes_and_lines_l390_390515


namespace solve_for_b_l390_390520

noncomputable def P (x a b d c : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + d * x + c

theorem solve_for_b (a b d c : ℝ) (h1 : -a = d) (h2 : d = 1 + a + b + d + c) (h3 : c = 8) :
    b = -17 :=
by
  sorry

end solve_for_b_l390_390520


namespace false_proposition_l390_390707

-- Definitions based on conditions
def opposite_angles (α β : ℝ) : Prop := α = β
def perpendicular (l m : ℝ → ℝ) : Prop := ∀ x, l x * m x = -1
def parallel (l m : ℝ → ℝ) : Prop := ∃ c, ∀ x, l x = m x + c
def corresponding_angles (α β : ℝ) : Prop := α = β

-- Propositions from the problem
def proposition1 : Prop := ∀ α β, opposite_angles α β → α = β
def proposition2 : Prop := ∀ l m n, perpendicular l n → perpendicular m n → parallel l m
def proposition3 : Prop := ∀ α β, α = β → opposite_angles α β
def proposition4 : Prop := ∀ α β, corresponding_angles α β → α = β

-- Statement to prove proposition 3 is false under given conditions
theorem false_proposition : ¬ proposition3 := by
  -- By our analysis, if proposition 3 is false, then it means the given definition for proposition 3 holds under all circumstances.
  sorry

end false_proposition_l390_390707


namespace point_A_in_third_quadrant_l390_390983

-- Defining the point A with its coordinates
structure Point :=
  (x : Int)
  (y : Int)

def A : Point := ⟨-1, -3⟩

-- The definition of quadrants in Cartesian coordinate system
def quadrant (p : Point) : String :=
  if p.x > 0 ∧ p.y > 0 then "first"
  else if p.x < 0 ∧ p.y > 0 then "second"
  else if p.x < 0 ∧ p.y < 0 then "third"
  else if p.x > 0 ∧ p.y < 0 then "fourth"
  else "boundary"

-- The theorem we want to prove
theorem point_A_in_third_quadrant : quadrant A = "third" :=
by 
  sorry

end point_A_in_third_quadrant_l390_390983


namespace range_of_eccentricity_of_ellipse_l390_390253

-- Definitions for the problem.
def is_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1) 

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

def on_upper_vertex (b : ℝ) : (ℝ × ℝ) := (0, b)

-- The statement to prove the range of eccentricity.
theorem range_of_eccentricity_of_ellipse (a b : ℝ) (x y : ℝ) (e : ℝ) :
  is_on_ellipse a b x y →
  (distance x y 0 b ≤ 2 * b) →
  ∃ e, (e = (1 - (b^2 / a^2))^(1/2) ∧ (0 < e) ∧ (e ≤ (sqrt 2)/2)) :=
by
  sorry

end range_of_eccentricity_of_ellipse_l390_390253


namespace right_triangle_area_l390_390079

theorem right_triangle_area :
  ∀ (a : ℝ), 
  let A := (0, 0) in
  let B := (a, a * Real.sqrt 3) in
  let C := (2 * a, 0) in
  (a * Real.sqrt 3 - (a * Real.sqrt 3 / 2) = 4) →
    area_of_triangle A B C = 16 * Real.sqrt 3 :=
by {
  sorry -- proof placeholder
}

end right_triangle_area_l390_390079


namespace total_possible_situations_l390_390651

theorem total_possible_situations : 
  let floors := {2, 3, 4, 5},
      persons := {'A', 'B', 'C', 'D'}
  in 
  ∃ (f : persons → floors) (on_fifth : {p : persons // f p = 5}),
  (f 'A' ≠ 2) ∧ 
  (∃! p : persons, f p = 5) → 
  (finset.card (finset.filter (λ p, f p = 2 ∨ f p = 3 ∨ f p = 4, persons)) = 3) ∧ 
  (finset.card (finset.filter (λ p, f p = 5, persons)) = 1) :=
sorry

end total_possible_situations_l390_390651


namespace range_of_eccentricity_l390_390244

noncomputable def upperVertex (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : ℝ × ℝ := (0, b)

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem range_of_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h: a > b) :
  (∀ (x y : ℝ), ellipse a b x y → distance x y 0 b ≤ 2 * b) →
  ∃ (e : ℝ), e ∈ Set.Ioc 0 (Real.sqrt 2 / 2) ∧ a^2 = b^2 / (1 - e^2) := sorry

end range_of_eccentricity_l390_390244


namespace initial_population_of_second_village_l390_390412

theorem initial_population_of_second_village :
  ∃ P : ℕ,
  let population_x_initial := 70000
  let population_x_decrease_rate := 1200
  let years := 14
  let population_second_village_increase_rate := 800
  in 
  (population_x_initial - population_x_decrease_rate * years) = (P + population_second_village_increase_rate * years) :=
sorry

end initial_population_of_second_village_l390_390412


namespace fruit_problem_l390_390455

def number_of_pears (A : ℤ) : ℤ := (3 * A) / 5
def number_of_apples (B : ℤ) : ℤ := (3 * B) / 7

theorem fruit_problem
  (A B : ℤ)
  (h1 : A + B = 82)
  (h2 : abs (A - B) < 10)
  (x : ℤ := (2 * A) / 5)
  (y : ℤ := (4 * B) / 7) :
  number_of_pears A = 24 ∧ number_of_apples B = 18 :=
by
  sorry

end fruit_problem_l390_390455


namespace rational_roots_count_l390_390860

theorem rational_roots_count (b₃ b₂ b₁ : ℚ) :
  number_of_different_possible_rational_roots (4 * x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 18) = 12 :=
sorry

end rational_roots_count_l390_390860


namespace miles_flown_on_thursday_l390_390973
-- Importing the necessary library

-- Defining the problem conditions and the proof goal
theorem miles_flown_on_thursday (x : ℕ) : 
  (∀ y, (3 * (1134 + y) = 7827) → y = x) → x = 1475 :=
by
  intro h
  specialize h 1475
  sorry

end miles_flown_on_thursday_l390_390973


namespace cube_side_length_l390_390815

theorem cube_side_length (n : ℕ) (h : (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 :=
sorry

end cube_side_length_l390_390815


namespace arc_BC_length_l390_390640

-- Given a circle with center O and radius OA, and an angle BAC subtending arc BC.
variables (O A B C : Point) (r : ℝ) (angle_BAC : ℝ) (arc_BC : ℝ)

-- Conditions: O is the center of the circle, OA is the radius, angle BAC = 45 degrees, and radius OA = 15 cm
-- Angle BAC subtends arc BC, which does not include point A.
axiom O_is_center : is_center_of_circle O
axiom radius_OA : radius O A = 15
axiom angle_BAC_45 : angle_at_point B A C = π / 4
axiom subtends_arc_BC : subtends_angle O B A C

-- The goal is to prove the length of arc BC.
theorem arc_BC_length : arc_length O B C = 22.5 * π :=
sorry

end arc_BC_length_l390_390640


namespace three_person_subcommittees_l390_390180

theorem three_person_subcommittees (n k : ℕ) (h1 : n = 8) (h2 : k = 3) : nat.choose n k = 56 := 
by
  rw [h1, h2]
  norm_num
  sorry

end three_person_subcommittees_l390_390180


namespace additional_people_needed_l390_390091

-- Define the initial number of people and time they take to mow the lawn 
def initial_people : ℕ := 8
def initial_time : ℕ := 3

-- Define total person-hours required to mow the lawn
def total_person_hours : ℕ := initial_people * initial_time

-- Define the time in which we want to find out how many people can mow the lawn
def desired_time : ℕ := 2

-- Define the number of people needed in desired_time to mow the lawn
def required_people : ℕ := total_person_hours / desired_time

-- Define the additional people required to mow the lawn in desired_time
def additional_people : ℕ := required_people - initial_people

-- Statement to be proved
theorem additional_people_needed : additional_people = 4 := by
  -- Proof to be filled in
  sorry

end additional_people_needed_l390_390091


namespace angle_between_vectors_l390_390965

variables {α : Type*} [inner_product_space ℝ α]

theorem angle_between_vectors (α β : α) (h : ∥α + β∥ = ∥α - β∥) (hα : α ≠ 0) (hβ : β ≠ 0) : 
  ⟪α, β⟫ = 0 :=
sorry

end angle_between_vectors_l390_390965


namespace Dmitriev_cleaned_10th_l390_390723

/-
Students Alekseev, Vasiliev, Sergeev, and Dmitriev were cleaning the classrooms of the 7th, 8th, 9th, and 10th grades. 
During an inspection, it was found that the 10th grade was cleaned poorly. They started to find out who cleaned this classroom. 
Alekseev said, "I cleaned 7th grade, and Dmitriev cleaned 8th grade." 
Vasiliev: "I cleaned 9th grade, and Alekseev cleaned 8th grade." 
Sergeev: "I cleaned 8th grade, and Vasiliev cleaned 10th grade." 
Dmitriev left for home before the inspection.You are to prove that Dmitriev cleaned the 10th grade if exactly one part of each of these statements is true.
-/

def cleaned_classroom : Type := sorry  -- Representation of the classroom cleaning assignment, can be extended later.

-- Statements made by each student:
axiom Alekseev_statement : cleaned_classroom → Prop
axiom Alekseev_statement_7th : cleaned_classroom → Prop
axiom Alekseev_statement_8th : cleaned_classroom → Prop

axiom Vasiliev_statement : cleaned_classroom → Prop
axiom Vasiliev_statement_9th : cleaned_classroom → Prop
axiom Vasiliev_statement_Alekseev_8th : cleaned_classroom → Prop

axiom Sergeev_statement : cleaned_classroom → Prop
axiom Sergeev_statement_8th : cleaned_classroom → Prop
axiom Sergeev_statement_Vasiliev_10th : cleaned_classroom → Prop

-- Assumptions about each statement having exactly one true part
axiom exactly_one_true_Alekseev : (Alekseev_statement_7th ∨ Alekseev_statement_8th) ∧ ¬ (Alekseev_statement_7th ∧ Alekseev_statement_8th)
axiom exactly_one_true_Vasiliev : (Vasiliev_statement_9th ∨ Vasiliev_statement_Alekseev_8th) ∧ ¬ (Vasiliev_statement_9th ∧ Vasiliev_statement_Alekseev_8th)
axiom exactly_one_true_Sergeev : (Sergeev_statement_8th ∨ Sergeev_statement_Vasiliev_10th) ∧ ¬ (Sergeev_statement_8th ∧ Sergeev_statement_Vasiliev_10th)

-- The proposition to prove
theorem Dmitriev_cleaned_10th : cleaned_classroom → Prop := sorry

end Dmitriev_cleaned_10th_l390_390723


namespace eccentricity_range_l390_390309

variables {a b : ℝ}
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

theorem eccentricity_range (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ (x y : ℝ), ellipse x y → distance x y 0 b ≤ 2 * b) :
  0 < (sqrt (a^2 - b^2) / a) ∧ (sqrt (a^2 - b^2) / a) ≤ sqrt 2 / 2 :=
sorry

end eccentricity_range_l390_390309


namespace sean_more_whistles_than_charles_l390_390716

theorem sean_more_whistles_than_charles :
  let S := 45 in
  let C := 13 in
  S - C = 32 :=
by
  let S := 45
  let C := 13
  show S - C = 32
  sorry

end sean_more_whistles_than_charles_l390_390716


namespace number_of_non_empty_proper_subsets_l390_390939

noncomputable def M : Set ℚ :=
  { x | 5 - abs (2 * x - 3) ∈ Set.Icc 1 5 }

theorem number_of_non_empty_proper_subsets :
  ∃ n, n = 510 ∧
    ∀ M ∈ ({ x : ℚ | 5 - abs (2 * x - 3) ∈ Set.Icc 1 5 }) ->
      2 ^ (Set.card M) - 2 = 510 := 
begin
  -- Subscription in the theorem header implies M = {x | 5 - abs (2 * x - 3) ∈ Set.Icc 1 5}.
  sorry
end

end number_of_non_empty_proper_subsets_l390_390939


namespace grassy_plot_width_l390_390041

/-- A rectangular grassy plot has a length of 100 m and a certain width. 
It has a gravel path 2.5 m wide all round it on the inside. The cost of gravelling 
the path at 0.90 rupees per square meter is 742.5 rupees. 
Prove that the width of the grassy plot is 60 meters. -/
theorem grassy_plot_width 
  (length : ℝ)
  (path_width : ℝ)
  (cost_per_sq_meter : ℝ)
  (total_cost : ℝ)
  (width : ℝ) : 
  length = 100 ∧ 
  path_width = 2.5 ∧ 
  cost_per_sq_meter = 0.9 ∧ 
  total_cost = 742.5 → 
  width = 60 := 
by sorry

end grassy_plot_width_l390_390041


namespace stratified_sampling_l390_390473

Variables (students_10 students_11 students_12 total_students selected_students : ℕ)
(h1 : students_10 = 300) (h2 : students_11 = 200) (h3 : students_12 = 400) (h4 : selected_students = 18)
(h5 : total_students = students_10 + students_11 + students_12)

theorem stratified_sampling :
  let students_10_selected := selected_students * students_10 / total_students,
      students_11_selected := selected_students * students_11 / total_students,
      students_12_selected := selected_students * students_12 / total_students in
  students_10_selected = 6 ∧ students_11_selected = 4 ∧ students_12_selected = 8 :=
by
  have total_students_eq : total_students = 300 + 200 + 400, from sorry,
  have students_selector_10_eq : selected_students * 300 / total_students = 6, from sorry,
  have students_selector_11_eq : selected_students * 200 / total_students = 4, from sorry,
  have students_selector_12_eq : selected_students * 400 / total_students = 8, from sorry,
  exact ⟨students_selector_10_eq, students_selector_11_eq, students_selector_12_eq⟩

end stratified_sampling_l390_390473


namespace fixed_point_DE_l390_390129

-- Definitions of points and the given condition
structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨-1, 0⟩
def N : Point := ⟨1, 0⟩

-- Distance function for points on the plane
def distance (A B : Point) : ℝ := 
  real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

-- Condition given in the problem
def condition (P : Point) : Prop :=
  distance P N * distance M N = (distance P M * distance N M)

-- Definition of the trajectory C
def on_trajectory_C (P : Point) : Prop := 
  P.y^2 = 4 * P.x

-- Given point A
def A (m : ℝ) : Point := ⟨m, 2⟩

-- Definitions for slope and line passing
def slope (P1 P2 : Point) : ℝ :=
  if P1.x = P2.x then 0 else (P2.y - P1.y) / (P2.x - P1.x)

-- Main theorem: proving fixed point
theorem fixed_point_DE (D E : Point) (P : Point) (hD : on_trajectory_C D) (hE : on_trajectory_C E) (hA : A 1 = ⟨1, 2⟩) (h1 : slope ⟨1, 2⟩ D * slope ⟨1, 2⟩ E = 2) : 
  ∃ (F : Point), ∀ (t : ℝ), (t = D.x) → (t = E.x) → F = ⟨-1, -2⟩ :=
sorry

end fixed_point_DE_l390_390129


namespace min_trihedral_angles_l390_390230

theorem min_trihedral_angles (A B C D O : Point) (T : Tetrahedron A B C D) (H: O ∈ interior T) :
    ∃ (angles : Finset (TrihedralAngle O)), angles.card = 4 ∧ (∀ (a b ∈ angles), a ≠ b → ¬intersects a b) :=
sorry

end min_trihedral_angles_l390_390230


namespace omega_2n_eq_omega_n_omega_8n_plus_5_eq_omega_4n_plus_3_omega_2_pow_n_minus_1_eq_n_l390_390324

def bit_weight (n : ℕ) : ℕ :=
  (n.bits.map (λ b, if b then 1 else 0)).sum

theorem omega_2n_eq_omega_n (n : ℕ) : 
  bit_weight (2 * n) = bit_weight n := 
sorry

theorem omega_8n_plus_5_eq_omega_4n_plus_3 (n : ℕ) : 
  bit_weight (8 * n + 5) = bit_weight (4 * n + 3) := 
sorry

theorem omega_2_pow_n_minus_1_eq_n (n : ℕ) : 
  bit_weight (2 ^ n - 1) = n := 
sorry

end omega_2n_eq_omega_n_omega_8n_plus_5_eq_omega_4n_plus_3_omega_2_pow_n_minus_1_eq_n_l390_390324


namespace amount_per_person_l390_390452

theorem amount_per_person (total_amount : ℕ) (num_persons : ℕ) (amount_each : ℕ)
  (h1 : total_amount = 42900) (h2 : num_persons = 22) (h3 : amount_each = 1950) :
  total_amount / num_persons = amount_each :=
by
  -- Proof to be filled
  sorry

end amount_per_person_l390_390452


namespace geometric_sequence_sum_l390_390977

variable {α : Type*} 
variable [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ q : α, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → α) (h : is_geometric_sequence a) 
  (h1 : a 0 + a 1 = 20) 
  (h2 : a 2 + a 3 = 40) : 
  a 4 + a 5 = 80 :=
sorry

end geometric_sequence_sum_l390_390977


namespace percent_gain_is_7_14_l390_390812

open Real

noncomputable def cost_per_sheep (c : ℝ) := c
noncomputable def total_cost (c : ℝ) := 900 * c
noncomputable def price_per_sheep_sold (c : ℝ) := (900 * c) / 840
noncomputable def revenue_from_840_sheep (c : ℝ) := 900 * c

theorem percent_gain_is_7_14 (c : ℝ) (h₁ : 0 < c) :
  let price_per_sheep_840 := (900 * c) / 840 in
  let revenue_remaining_sheep := 60 * price_per_sheep_840 in
  let total_revenue := revenue_from_840_sheep c + revenue_remaining_sheep in
  let profit := total_revenue - total_cost c in
  (profit / total_cost c) * 100 ≈ 7.14 :=
by
  let price_per_sheep_840 := (900 * c) / 840
  let revenue_remaining_sheep := 60 * price_per_sheep_840
  let total_revenue := revenue_from_840_sheep c + revenue_remaining_sheep
  let profit := total_revenue - total_cost c
  have profit_correct : profit = 64.2857142857 * c := sorry
  have percentage_gain : (profit / total_cost c) * 100 ≈ 7.14 := sorry
  exact percentage_gain

end percent_gain_is_7_14_l390_390812


namespace consecutive_numbers_l390_390409

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).foldr (· + ·) 0

-- Define the conditions
def condition_1 (n : ℕ) : Prop :=
  sum_of_digits n = 8

def condition_2 (n : ℕ) : Prop :=
  (n + 1) % 8 = 0

-- Main proof problem statement
theorem consecutive_numbers (71: ℕ) (72 : ℕ) : condition_1 71 ∧ condition_2 71 := by
  sorry

end consecutive_numbers_l390_390409


namespace blue_painted_area_ratio_l390_390769

-- Define the conditions
def circle_radius : ℝ := 2
def semi_major_axis : ℝ := 4
def semi_minor_axis : ℝ := 3

-- Define the areas
def area_circle : ℝ := Real.pi * circle_radius^2
def area_ellipse : ℝ := Real.pi * semi_major_axis * semi_minor_axis
def area_blue : ℝ := area_ellipse - area_circle

-- Define the desired ratio
def desired_ratio : ℝ := area_blue / area_circle

-- The theorem to prove
theorem blue_painted_area_ratio :
  desired_ratio = 2 := 
by
  sorry

end blue_painted_area_ratio_l390_390769


namespace hyperbola_s_eq_l390_390463

theorem hyperbola_s_eq (s : ℝ) 
  (hyp1 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (5, -3) → x^2 / 9 - y^2 / b^2 = 1) 
  (hyp2 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (3, 0) → x^2 / 9 - y^2 / b^2 = 1) 
  (hyp3 : ∃ b > 0, ∀ x y : ℝ, (x, y) = (s, -1) → x^2 / 9 - y^2 / b^2 = 1) :
  s^2 = 873 / 81 :=
sorry

end hyperbola_s_eq_l390_390463


namespace inverse_f_486_l390_390192

-- Define the function f with given properties.
def f : ℝ → ℝ := sorry

-- Condition 1: f(5) = 2
axiom f_at_5 : f 5 = 2

-- Condition 2: f(3x) = 3f(x) for all x
axiom f_scale : ∀ x, f (3 * x) = 3 * f x

-- Proposition: f⁻¹(486) = 1215
theorem inverse_f_486 : (∃ x, f x = 486) → ∀ x, f x = 486 → x = 1215 :=
by sorry

end inverse_f_486_l390_390192


namespace tan_double_angle_l390_390916

-- Define the condition
def condition (α : ℝ) := (sin α + cos α) / (sin α - cos α) = 1 / 2

-- Define the theorem
theorem tan_double_angle (α : ℝ) (h : condition α) : tan (2 * α) = 3 / 4 := 
sorry

end tan_double_angle_l390_390916


namespace min_value_of_f_l390_390103

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f : ∃ x : ℝ, (f x = -(1 / Real.exp 1)) ∧ (∀ y : ℝ, f y ≥ f x) := by
  sorry

end min_value_of_f_l390_390103


namespace calculate_oranges_l390_390804

theorem calculate_oranges 
    (cost_price_per_orange : ℕ := 3) 
    (selling_price_per_orange : ℕ := 4) 
    (desired_profit : ℕ := 200ᶜᵉⁿᵗₛ)
    (profit_per_orange : ℕ := selling_price_per_orange - cost_price_per_orange) : 
  profit_per_orange * 200 = desired_profit := 
sorry

end calculate_oranges_l390_390804


namespace angle_of_inclination_l390_390071

-- Definitions and conditions from step a)
def f (x : ℝ) : ℝ := (20 * x^2 - 16 * x + 1) / (5 * x - 2)

-- The theorem to prove that the angle of inclination is arctan(8)
theorem angle_of_inclination (x : ℝ) (hx : f x = 0) :
  real.angle_of_inclination (deriv f x) = real.arctan 8 :=
sorry

end angle_of_inclination_l390_390071


namespace minimum_value_MP_MF_l390_390132

noncomputable def min_value (M P : ℝ × ℝ) (F : ℝ × ℝ) : ℝ := |dist M P + dist M F|

theorem minimum_value_MP_MF :
  ∀ (M : ℝ × ℝ), (M.2 ^ 2 = 4 * M.1) →
  ∀ (F : ℝ × ℝ), (F = (1, 0)) →
  ∀ (P : ℝ × ℝ), (P = (3, 1)) →
  min_value M P F = 4 :=
by
  intros M h_para F h_focus P h_fixed
  rw [min_value]
  sorry

end minimum_value_MP_MF_l390_390132


namespace kerosene_cost_l390_390213

theorem kerosene_cost :
  (∀ (x : ℝ),
    (∀ (dozenEggCost riceCost : ℝ), dozenEggCost = 0.33 → riceCost = 0.33 →
      12 * (dozenEggCost / 12) = riceCost → 
      x * (dozenEggCost / 12) = riceCost → 
      x = 12) ∧ 
      (2 * x * 0.0275 = 0.66)) :=
by
  intro x
  split
  sorry
  sorry

end kerosene_cost_l390_390213


namespace contradiction_method_example_l390_390710

theorem contradiction_method_example (a b : ℝ) : a^2 + b^2 ≥ 2 * (a - b - 1) :=
begin
  by_contra h,
  have : a^2 + b^2 < 2 * (a - b - 1), from h,
  sorry
end

end contradiction_method_example_l390_390710


namespace intersect_point_l390_390997

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 9*x + 15

theorem intersect_point : (∃ a b : ℝ, f(a) = b ∧ f(b) = a ∧ a = b) → (a = -1 ∧ b = -1) := 
by {
  sorry
}

end intersect_point_l390_390997


namespace three_person_subcommittees_from_eight_l390_390168

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem three_person_subcommittees_from_eight (n k : ℕ) (h_n : n = 8) (h_k : k = 3) :
  combination n k = 56 := by
  sorry

end three_person_subcommittees_from_eight_l390_390168


namespace part_I_part_II_part_III_l390_390932

open Real

section Problem

variable (a : ℝ)

def f (x : ℝ) : ℝ := a * x
def g (x : ℝ) : ℝ := log x
def F (x : ℝ) : ℝ := f a x - g x
def G (x : ℝ) : ℝ := a * sin (1 - x) + log x

theorem part_I : (∀ x, diff F x = a - 1 / x) → diff F 1 = 0 → a = 1 :=
by
  -- Proof omitted
  sorry

theorem part_II : (∀ x ∈ Ioo 0 1, diff G x ≥ 0) → a ≤ 1 :=
by
  -- Proof omitted
  sorry

theorem part_III : (∀ n : ℕ, ∑ k in range n, sin (1 / (k + 1)^2) < log 2) :=
by
  -- Proof omitted
  sorry

end Problem

end part_I_part_II_part_III_l390_390932


namespace muriatic_numbers_count_l390_390669

theorem muriatic_numbers_count (a : ℕ → ℕ) (n : ℕ) (h : ∀ i, 1 ≤ i → i ≤ n → 0 ≤ a i) (α : ℝ) (hα : α > 0) :
  (∑ i in finset.range n, a (i + 1)) / α > finset.card ((finset.range n).filter (λ k, 
  ∀ l, 1 ≤ l → l ≤ k + 1 → (∑ j in finset.range l, a (k + 1 - j)) / l > α)) := by
  sorry

end muriatic_numbers_count_l390_390669


namespace solve_eq_l390_390557

theorem solve_eq (x y : ℝ) (H : sqrt (2 * x + 3 * y) + abs (x + 3) = 0) : x = -3 ∧ y = 2 := 
sorry

end solve_eq_l390_390557


namespace problem_l390_390157

-- Conditions from the problem
def is_quadratic_polynomial (a b : ℝ) (p : Polynomial ℝ) : Prop :=
  ∀ x : ℝ, p = Polynomial.C (2021) + Polynomial.C (2) * x +
           Polynomial.C (5) * a * x^2 + Polynomial.C (-13) * x^3 +
           Polynomial.C (-1) * x^4 + Polynomial.C a * x^3 +
           Polynomial.C (-b) * x^4 + Polynomial.C (-13) * x^3

-- Translate the specific polynomial conditions to Lean definitions
theorem problem (a b : ℝ) (h : is_quadratic_polynomial a b
  (Polynomial.C (2021) + Polynomial.C (2) * Polynomial.X +
   Polynomial.C (5) * a * Polynomial.X^2 - 
   Polynomial.C (13) * Polynomial.X^3 - Polynomial.C (1) * Polynomial.X^4 + 
   Polynomial.C a * Polynomial.X^3 - Polynomial.C b * Polynomial.X^4 - 
   Polynomial.C (13)* Polynomial.X^3)) :
  a^2 + b^2 = 677 := by
  sorry

end problem_l390_390157


namespace distinct_nonconsecutive_digits_between_200_and_250_l390_390947

theorem distinct_nonconsecutive_digits_between_200_and_250 :
  {n : ℕ // 200 ≤ n ∧ n < 250 ∧ (∀ i j, n.digit i ≠ n.digit j ∧ abs (n.digit i - n.digit j) ≠ 1)}.card = 29 :=
by
  sorry

end distinct_nonconsecutive_digits_between_200_and_250_l390_390947


namespace participant_problem_partition_l390_390468

theorem participant_problem_partition (participants problems : ℕ) 
    (partI_partII : ℕ × ℕ) (solves : ℕ) (pairs_solved : ℕ):
    (problems = 28) →                                    -- Condition 1: Total problems
    (solves = 7) →                                       -- Condition 2: Each participant solves 7 problems
    (pairs_solved = 2) →                                 -- Condition 3: Each pair of problems is solved by 2 participants
    (partI_partII.1 + partI_partII.2 = problems) →       -- Division into two parts
    ∃ (participant : ℕ),                                 -- Existence of participant
      (participant_solveI = 0 ∨                           -- Participant solves no problems in Part I
       participant_solveI ≥ 4) :=                         -- Participant solves at least 4 problems in Part I
begin
  sorry
end

end participant_problem_partition_l390_390468


namespace find_number_l390_390633

theorem find_number (x n : ℝ) (h1 : x > 0) (h2 : x / 50 + x / n = 0.06 * x) : n = 25 :=
by
  sorry

end find_number_l390_390633


namespace reversal_impossible_l390_390229

theorem reversal_impossible (s : string) : 
  (∀ segment, (∀ char ∈ segment, char = 'A' ∨ char = 'B') →
  (segment.count 'A' = segment.count 'B') → 
  let reversed_segment := segment.reverse.map (λ c, if c = 'A' then 'B' else 'A') in
  s.replace segment reversed_segment) →
  (s = "A".repeat 125 ++ "B".repeat 125) →
  ¬(∃ ops, (∃ reversed_s, reverse_string(ops, s) = reversed_s) ∧ reversed_s = s.reverse)
:= by
  sorry

end reversal_impossible_l390_390229


namespace tangent_line_fx_inequality_k_max_value_l390_390153

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / (Real.exp x + 1)

-- Statement 1: The equation of the tangent line
theorem tangent_line (x : ℝ) :
  let tangent := (1 : ℝ) / 2 * x
  in ∀ y : ℝ, y = f 0 → tangent = (1 : ℝ) / 2 * x :=
by
  intro x
  sorry

-- Statement 2: The inequality for x > 0
theorem fx_inequality (x : ℝ) (h : 0 < x) :
  f x > x / (x + 2) :=
by
  intro x h
  sorry

-- Statement 3: The maximum value of k
theorem k_max_value (k : ℝ) :
  (∀ x : ℝ, 0 < x → f x > k * x / (Real.exp x + 1)) → k ≤ 1 :=
by
  intro k 
  sorry

end tangent_line_fx_inequality_k_max_value_l390_390153


namespace distance_traveled_l390_390814

def speed : ℝ := 4 -- in meters per second
def time : ℝ := 32 -- in seconds

theorem distance_traveled : speed * time = 128 := by
  rw [speed, time]
  norm_num
  -- 4 * 32 = 128
  sorry

end distance_traveled_l390_390814


namespace remainder_of_binary_division_l390_390440

theorem remainder_of_binary_division : 
  (110110111101₂ % 4 = 1) :=
by sorry

end remainder_of_binary_division_l390_390440


namespace largest_natural_number_has_sum_of_digits_property_l390_390544

noncomputable def largest_nat_num_digital_sum : ℕ :=
  let a : ℕ := 1
  let b : ℕ := 0
  let d3 := a + b
  let d4 := 2 * a + 2 * b
  let d5 := 4 * a + 4 * b
  let d6 := 8 * a + 8 * b
  100000 * a + 10000 * b + 1000 * d3 + 100 * d4 + 10 * d5 + d6

theorem largest_natural_number_has_sum_of_digits_property :
  largest_nat_num_digital_sum = 101248 :=
by
  sorry

end largest_natural_number_has_sum_of_digits_property_l390_390544


namespace smallest_sector_angle_l390_390692

-- Definitions and conditions identified in step a.

def a1 (d : ℕ) : ℕ := (48 - 14 * d) / 2

-- Proof statement
theorem smallest_sector_angle : ∀ d : ℕ, d ≥ 0 → d ≤ 3 → 15 * (a1 d + (a1 d + 14 * d)) = 720 → (a1 d = 3) :=
by
  sorry

end smallest_sector_angle_l390_390692


namespace inequality_interval_l390_390085

theorem inequality_interval : ∀ x : ℝ, (x^2 - 3 * x - 4 < 0) ↔ (-1 < x ∧ x < 4) :=
by
  intro x
  sorry

end inequality_interval_l390_390085


namespace no_solution_iff_m_leq_2_l390_390203

theorem no_solution_iff_m_leq_2 (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2 < 3x - 6 ∧ x < m)) ↔ m ≤ 2 :=
by
  sorry

end no_solution_iff_m_leq_2_l390_390203


namespace winning_percentage_l390_390475

/-- A soccer team played 158 games and won 63.2 games. 
    Prove that the winning percentage of the team is 40%. --/
theorem winning_percentage (total_games : ℕ) (won_games : ℝ) (h1 : total_games = 158) (h2 : won_games = 63.2) :
  (won_games / total_games) * 100 = 40 :=
sorry

end winning_percentage_l390_390475


namespace maximum_OC_value_l390_390575

noncomputable def isosceles_right_triangle (A B C : Point) : Prop :=
  dist A B = dist A C ∧ angle A B C = π/2

theorem maximum_OC_value (A B C O : Point) (k : ℝ) (h_A : A = (2, 0))
  (h_B : ∃ x, B = (x, sqrt(1 - x^2)) ∧ x ≤ 1)
  (h_triangle : isosceles_right_triangle A B C) :
  ∃ C, dist O C = 2 * sqrt 2 + 1 :=
by
  sorry

end maximum_OC_value_l390_390575


namespace eq_3n_2m_1_solutions_l390_390084

open Nat

-- Define the theorem
theorem eq_3n_2m_1_solutions (m n : ℕ) (h : 3^n - 2^m = 1) :
  (m = 1 ∧ n = 1) ∨ (m = 3 ∧ n = 2) :=
by
  -- The proof is left as an exercise (we use sorry to skip the proof)
  sorry

-- Example instances to check the implementation
example : eq_3n_2m_1_solutions 1 1 (by norm_num) := Or.inl ⟨rfl, rfl⟩
example : eq_3n_2m_1_solutions 3 2 (by norm_num) := Or.inr ⟨rfl, rfl⟩

end eq_3n_2m_1_solutions_l390_390084


namespace symmetry_of_g_l390_390562

def f (x : ℝ) : ℝ := √3 * sin x * cos x - sin x^2

def g (x : ℝ) : ℝ := sin (2 * x) + 3 / 2

theorem symmetry_of_g :
  ∀ x α : ℝ, g (α - x) = g (α + x) →
  α = k * π / 2 + π / 4 → 
  g (α + π / 4) + g (π / 4) = 4 :=
by
  simp [g, sin, cos, π]
  sorry

end symmetry_of_g_l390_390562


namespace probability_B_in_A_l390_390687

noncomputable def A : Set (ℝ × ℝ) := {p | |p.1| + |p.2| ≤ 2}
noncomputable def B : Set (ℝ × ℝ) := {p | p ∈ A ∧ p.2 ≤ p.1 ^ 2}

-- Given a point P ∈ A, we want to prove the probability P ∈ B is 17/24.
theorem probability_B_in_A : (μ(B) / μ(A) = 17 / 24) :=
sorry

end probability_B_in_A_l390_390687


namespace sum_first_n_terms_geometric_sequence_l390_390395

def geometric_sequence_sum (n : ℕ) (k : ℝ) : ℝ :=
  if n = 0 then 0 else (3 * 2^n + k)

theorem sum_first_n_terms_geometric_sequence (k : ℝ) :
  (geometric_sequence_sum 1 k = 6 + k) ∧ 
  (∀ n > 1, geometric_sequence_sum n k - geometric_sequence_sum (n - 1) k = 3 * 2^(n-1))
  → k = -3 :=
by
  sorry

end sum_first_n_terms_geometric_sequence_l390_390395


namespace negation_of_no_vegetarian_students_eat_at_cafeteria_l390_390393

variable (Student : Type) 
variable (isVegetarian : Student → Prop)
variable (eatsAtCafeteria : Student → Prop)

theorem negation_of_no_vegetarian_students_eat_at_cafeteria :
  (∀ x, isVegetarian x → ¬ eatsAtCafeteria x) →
  (∃ x, isVegetarian x ∧ eatsAtCafeteria x) :=
by
  sorry

end negation_of_no_vegetarian_students_eat_at_cafeteria_l390_390393


namespace quadrilateral_area_is_3_l390_390775

open Real

def point := (ℝ × ℝ)

def area_of_quadrilateral (p₁ p₂ p₃ p₄ : point) : ℝ :=
  (1/2) * abs
    ((p₁.1 * p₂.2 + p₂.1 * p₃.2 + p₃.1 * p₄.2 + p₄.1 * p₁.2)
     - (p₁.2 * p₂.1 + p₂.2 * p₃.1 + p₃.2 * p₄.1 + p₄.2 * p₁.1))

theorem quadrilateral_area_is_3 :
  area_of_quadrilateral (2, 0) (0, 3) (5, 3) (8, 9) = 3 :=
by
  simp [area_of_quadrilateral]
  norm_num
  sorry

end quadrilateral_area_is_3_l390_390775


namespace marcos_salary_end_april_l390_390690

-- Definitions for the conditions
def initial_salary_january : ℝ := 2500
def raise_february : ℝ := 0.15
def raise_march : ℝ := 0.10
def pay_cut_april : ℝ := 0.15

-- Statement to prove
theorem marcos_salary_end_april : 
  let february_salary := initial_salary_january * (1 + raise_february),
      march_salary := february_salary * (1 + raise_march),
      april_salary := march_salary * (1 - pay_cut_april)
  in april_salary = 2688.125 :=
by sorry

end marcos_salary_end_april_l390_390690


namespace sum_A_odot_B_l390_390519

def A : Set ℕ := {0, 1}
def B : Set ℕ := {2, 3}

def A_odot_B : Set ℕ :=
  {z | ∃ x ∈ A, ∃ y ∈ B, z = x * y * (x + y)}

theorem sum_A_odot_B : (Finset.sum (A_odot_B.toFinset) id) = 18 :=
by
  sorry

end sum_A_odot_B_l390_390519


namespace f3_is_quadratic_l390_390445

-- Define the given functions
def f1 (x : ℝ) : ℝ := Real.sqrt (x^2 + 2)
def f2 (x : ℝ) : ℝ := 3 * x - 1
def f3 (x : ℝ) : ℝ := x^2
def f4 (x : ℝ) : ℝ := 1 / x^2 + 3

-- Define what it means to be a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- State the proof problem
theorem f3_is_quadratic : is_quadratic f3 :=
sorry

end f3_is_quadratic_l390_390445


namespace p_inverse_iff_b_eq_negative_three_l390_390503

noncomputable def h (x : ℝ) : ℝ := (x - 5) / (x - 4)

def p (x b : ℝ) : ℝ := h (x + b)

theorem p_inverse_iff_b_eq_negative_three (b : ℝ) :
  (∀ x, p x b = (p x b) → p (p x b) b = x) ↔ b = -3 := sorry

end p_inverse_iff_b_eq_negative_three_l390_390503


namespace y_at_x_eq_120_l390_390413

@[simp] def custom_op (a b : ℕ) : ℕ := List.prod (List.map (λ i => a + i) (List.range b))

theorem y_at_x_eq_120 {x y : ℕ}
  (h1 : custom_op x (custom_op y 2) = 420)
  (h2 : x = 4)
  (h3 : y = 2) :
  custom_op y x = 120 := by
  sorry

end y_at_x_eq_120_l390_390413


namespace part1_part2_minimum_value_expression_part3_maximum_value_l390_390560

noncomputable def f (a x : ℝ) : ℝ := log a x
noncomputable def g (a x : ℝ) : ℝ := a ^ x
noncomputable def F (a m x : ℝ) : ℝ := (2 * m - 1) * g a x + (1 / m - 1 / 2) * g a (-x)
noncomputable def h (m : ℝ) : ℝ := 2 * sqrt ((2 * m - 1) * (1 / m - 1 / 2))

-- (1) Proving x = (7 ± sqrt(29 - 4a) / 2)
theorem part1 (a x : ℝ) (h₁ : f a (x-1) = f a (a-x) - f a (5-x)) :
  x = (7 + sqrt(29 - 4 * a)) / 2 ∨ x = (7 - sqrt(29 - 4 * a)) / 2 := by
  sorry

-- (2) Proving h(m) = 2 * sqrt ((2m-1) * (1/m - 1/2))
theorem part2_minimum_value_expression (m : ℝ) :
  h m = 2 * sqrt ((2 * m - 1) * (1 / m - 1 / 2)) := by
  sorry

-- (3) Proving maximum value of h(m) = sqrt(2)
theorem part3_maximum_value : ∃ m, 1 / 2 < m ∧ m < 2 ∧ h m = sqrt 2 := by
  sorry

end part1_part2_minimum_value_expression_part3_maximum_value_l390_390560


namespace smallest_angle_solution_l390_390514

noncomputable def smallest_positive_angle (x : ℝ) : ℝ :=
  if h : 9 * sin x * (cos x)^3 - 9 * (sin x)^3 * cos x = 3 / 2 
  then x else 0

theorem smallest_angle_solution :
  smallest_positive_angle (10.45 * (2 * Real.pi / 360)) ≈ 10.45 * (2 * Real.pi / 360) :=
by
  sorry

end smallest_angle_solution_l390_390514


namespace range_of_eccentricity_l390_390300

-- Definition of the ellipse and its properties
namespace EllipseProof

variables {a b : ℝ} (h : a > b ∧ b > 0)

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def upper_vertex : ℝ × ℝ := (0, b)

def is_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def condition_on_point (P : ℝ × ℝ) : Prop := dist P upper_vertex ≤ 2 * b

-- The eccentricity of the ellipse
def eccentricity : ℝ := real.sqrt (1 - (b^2 / a^2))

-- The required proof statement
theorem range_of_eccentricity :
  (∀ P, is_on_ellipse P → condition_on_point P) →
  eccentricity ≤ real.sqrt 2 / 2 := sorry

end EllipseProof

end range_of_eccentricity_l390_390300


namespace measure_obtuse_angle_in_right_triangle_l390_390982

open Real

theorem measure_obtuse_angle_in_right_triangle 
  (A B C P : Point)
  (α β γ : Angle)
  (h₁ : α = 45)
  (h₂ : β = 45)
  (h₃ : ∠ABC = 90)
  (h₄ : AngleBisector β ∩ AngleBisector γ = P) :
  ∠BPC = 270 :=
by
  sorry

end measure_obtuse_angle_in_right_triangle_l390_390982


namespace inequality_proof_l390_390406

variables {a b : ℝ}

theorem inequality_proof :
  a^2 + b^2 - 1 - a^2 * b^2 <= 0 ↔ (a^2 - 1) * (b^2 - 1) >= 0 :=
by sorry

end inequality_proof_l390_390406


namespace games_within_division_l390_390459

variables (N M : ℕ)
  (h1 : N > 2 * M)
  (h2 : M > 4)
  (h3 : 3 * N + 4 * M = 76)

theorem games_within_division :
  3 * N = 48 :=
sorry

end games_within_division_l390_390459


namespace typing_time_l390_390236

theorem typing_time :
  let Jonathan_rate := 1 / 4
      Susan_rate := 1 / 3
      Jack_rate := 5 / 12
      combined_rate := Jonathan_rate + Susan_rate + Jack_rate
  in combined_rate = 1 -> (10 / combined_rate) = 10 := 
by
  intro Jonathan_rate Susan_rate Jack_rate combined_rate
  intros h_combined_rate
  rw h_combined_rate
  simp
  rfl

end typing_time_l390_390236


namespace hexagon_rectangle_ratio_l390_390820

theorem hexagon_rectangle_ratio:
  ∀ (h w : ℕ), 
  (6 * h = 24) → (2 * (2 * w + w) = 24) → 
  (h / w = 1) := by
  intros h w
  intro hex_condition
  intro rect_condition
  sorry

end hexagon_rectangle_ratio_l390_390820


namespace lcm_72_108_2100_l390_390102

theorem lcm_72_108_2100 : Nat.lcm (Nat.lcm 72 108) 2100 = 37800 := by
  sorry

end lcm_72_108_2100_l390_390102


namespace initial_cows_count_l390_390816

theorem initial_cows_count (C : ℕ) : 
  (C - 31 + 75 = 83) → C = 39 :=
by
  intro h
  have h1 : C + 44 = 83 := by linarith
  have h2 : C = 83 - 44 := by linarith
  have h3 : C = 39 := by linarith
  exact h3

end initial_cows_count_l390_390816


namespace intersecting_lines_l390_390629

theorem intersecting_lines (m n : ℝ) : 
  (∀ x y : ℝ, y = x / 2 + n → y = mx - 1 → (x = 1 ∧ y = -2)) → 
  m = -1 ∧ n = -5 / 2 :=
by
  sorry

end intersecting_lines_l390_390629


namespace faucets_fill_time_l390_390105

theorem faucets_fill_time (rate_per_faucet : ℕ → ℕ → ℚ) 
  (fill_time : ℚ → ℚ → ℚ) 
  (faucet_rate : rate_per_faucet 5 150 = 15) 
  (ten_faucet_rate : 10 * rate_per_faucet 5 150 / 5 = 30)
  (time_conversion : ∀ (time : ℚ), fill_time time 60 = time * 60) : 
  fill_time (50 / (10 * rate_per_faucet 5 150 / 5)) 60 = 100 :=
by 
  have five_faucet_rate := rate_per_faucet 5 150 / 5
  have rate_10_faucets := 10 * five_faucet_rate
  have fill_time_50_gallon := 50 / rate_10_faucets 
  rw [faucet_rate, ten_faucet_rate] at fill_time_50_gallon
  rw [time_conversion, fill_time_50_gallon]
  norm_num
  rfl

end faucets_fill_time_l390_390105


namespace bond_interest_percentage_approx_l390_390664

noncomputable def bond_interest_percentage (FaceValue : ℝ) (InterestRate : ℝ) (SellingPrice : ℝ) : ℝ :=
  (FaceValue * InterestRate) / SellingPrice * 100

theorem bond_interest_percentage_approx :
  bond_interest_percentage 5000 0.08 6153.846153846153 ≈ 6.5 :=
by
  sorry

end bond_interest_percentage_approx_l390_390664


namespace number_of_elephants_l390_390209

theorem number_of_elephants (giraffes penguins total_animals elephants : ℕ)
  (h1 : giraffes = 5)
  (h2 : penguins = 2 * giraffes)
  (h3 : penguins = total_animals / 5)
  (h4 : elephants = total_animals * 4 / 100) :
  elephants = 2 := by
  -- The proof is omitted
  sorry

end number_of_elephants_l390_390209


namespace concentrated_numbers_count_correct_l390_390630

def is_digit (d : Nat) : Prop := d ≤ 3

def is_concentrated (n : Nat) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n / 100 ≠ 0 ∧ |a - b| ≤ 1 ∧ |b - c| ≤ 1 ∧ is_digit a ∧ is_digit b ∧ is_digit c

def count_concentrated_numbers : Nat :=
  Nat.fold 100 1000 (λ n count, if is_concentrated n then count + 1 else count) 0

theorem concentrated_numbers_count_correct : count_concentrated_numbers = 21 := 
sorry

end concentrated_numbers_count_correct_l390_390630


namespace molecular_weight_one_mole_l390_390779

theorem molecular_weight_one_mole (total_weight : ℝ) (moles : ℝ) (H : total_weight = 854) (M : moles = 7) : 
  total_weight / moles = 122 :=
by
  rw [H, M]
  norm_num
  sorry

end molecular_weight_one_mole_l390_390779


namespace rectangle_section_properties_l390_390447

structure Tetrahedron where
  edge_length : ℝ

structure RectangleSection where
  perimeter : ℝ
  area : ℝ

def regular_tetrahedron : Tetrahedron :=
  { edge_length := 1 }

theorem rectangle_section_properties :
  ∀ (rect : RectangleSection), 
  (∃ tetra : Tetrahedron, tetra = regular_tetrahedron) →
  (rect.perimeter = 2) ∧ (0 ≤ rect.area) ∧ (rect.area ≤ 1/4) :=
by
  -- Provide the hypothesis of the existence of such a tetrahedron and rectangular section
  sorry

end rectangle_section_properties_l390_390447


namespace range_of_eccentricity_of_ellipse_l390_390256

-- Definitions for the problem.
def is_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1) 

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

def on_upper_vertex (b : ℝ) : (ℝ × ℝ) := (0, b)

-- The statement to prove the range of eccentricity.
theorem range_of_eccentricity_of_ellipse (a b : ℝ) (x y : ℝ) (e : ℝ) :
  is_on_ellipse a b x y →
  (distance x y 0 b ≤ 2 * b) →
  ∃ e, (e = (1 - (b^2 / a^2))^(1/2) ∧ (0 < e) ∧ (e ≤ (sqrt 2)/2)) :=
by
  sorry

end range_of_eccentricity_of_ellipse_l390_390256


namespace girls_more_than_boys_l390_390390

variable (total_students ratio_b ratio_g : ℕ)
variable [fact (total_students = 42)]
variable [fact (ratio_b = 3)]
variable [fact (ratio_g = 4)]

theorem girls_more_than_boys : 
  ∀ (x : ℕ), 
    total_students = ratio_b * x + ratio_g * x → 
    ratio_b + ratio_g = 7 → 
    total_students = 7 * x → 
    ratio_g * x - ratio_b * x = 6 :=
by 
  sorry

end girls_more_than_boys_l390_390390


namespace sum_of_indices_l390_390861

theorem sum_of_indices (r : ℕ) (n : fin r → ℕ) (a : fin r → ℤ) 
  (h1 : ∀ i j, i < j → n i > n j)
  (h2 : ∀ k, a k ∈ ({0, 1, -1} : set ℤ)) :
  (∑ i, a i * 3 ^ n i = 2009) → (∑ i, n i = 34) :=
begin
  sorry
end

end sum_of_indices_l390_390861


namespace emily_expenditure_l390_390498

-- Define the conditions
def price_per_flower : ℕ := 3
def roses_bought : ℕ := 2
def daisies_bought : ℕ := 2

-- Total flowers bought
def total_flowers (roses daisies : ℕ) : ℕ :=
  roses + daisies

-- Define the cost function
def cost (flowers price_per_flower : ℕ) : ℕ :=
  flowers * price_per_flower

-- Theorem to prove the total expenditure
theorem emily_expenditure : 
  cost (total_flowers roses_bought daisies_bought) price_per_flower = 12 :=
by
  sorry

end emily_expenditure_l390_390498


namespace shaded_area_computation_l390_390414

noncomputable def side_length_square : ℝ := 24

noncomputable def number_of_circles : ℕ := 6

noncomputable def radius_circle : ℝ := side_length_square / (number_of_circles / 2)

noncomputable def area_square : ℝ := side_length_square ^ 2

noncomputable def area_one_circle : ℝ := real.pi * radius_circle ^ 2

noncomputable def total_area_circles : ℝ := number_of_circles * area_one_circle

noncomputable def shaded_area : ℝ := area_square - total_area_circles

theorem shaded_area_computation : shaded_area = 576 - 96 * real.pi := by
  sorry

end shaded_area_computation_l390_390414


namespace count_valid_numbers_l390_390348

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 10 in
  let b := n % 10 in
  10 ≤ n ∧ n < 100 ∧
  9 * a % 10 == 2 ∧
  (10 * a + b) % 3 == 0

theorem count_valid_numbers : Finset.card (Finset.filter is_valid_number (Finset.range 100)) = 3 :=
by
  sorry

end count_valid_numbers_l390_390348


namespace polynomial_solution_l390_390869

theorem polynomial_solution (P : Polynomial ℝ) (h_0 : P.eval 0 = 0) (h_func : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) :
  P = Polynomial.X :=
sorry

end polynomial_solution_l390_390869


namespace airplane_altitude_l390_390488

noncomputable def altitude_of_airplane 
	(A B C P : ℝ × ℝ) 
	(a_dist b_dist c_dist : ℝ)
	(angle_A angle_B angle_C : ℝ) : ℝ :=
	let x := a_dist * math.sqrt(2)
	let y := b_dist / math.sqrt(3)
	let z := c_dist / math.sqrt(2)
	in if x = y ∧ y = z then x else 0

theorem airplane_altitude :
	let A := (0, 0)
	let B := (15, 0)
	let C := (-10, 0)
	let P := (0, 56.5) 
	let angle_A := 45
	let angle_B := 30
	let angle_C := 60
	let alt := altitude_of_airplane A B C P 15 10 25
	in alt = 56.5 := by
	sorry

end airplane_altitude_l390_390488


namespace cos_inequality_l390_390361

theorem cos_inequality (x y : ℝ) (h : x^2 + y^2 ≤ π / 2) : 
  cos x + cos y ≤ 1 + cos (x * y) :=
sorry

end cos_inequality_l390_390361


namespace lcm_100_40_is_200_l390_390711

theorem lcm_100_40_is_200 : Nat.lcm 100 40 = 200 := by
  sorry

end lcm_100_40_is_200_l390_390711


namespace area_EYH_trapezoid_l390_390407

theorem area_EYH_trapezoid (EF GH : ℕ) (EF_len : EF = 15) (GH_len : GH = 35) 
(Area_trapezoid : (EF + GH) * 16 / 2 = 400) : 
∃ (EYH_area : ℕ), EYH_area = 84 := by
  sorry

end area_EYH_trapezoid_l390_390407


namespace zoe_correct_percentage_l390_390511

noncomputable def t : ℝ := sorry  -- total number of problems
noncomputable def chloe_alone_correct : ℝ := 0.70 * (1/3 * t)  -- Chloe's correct answers alone
noncomputable def chloe_total_correct : ℝ := 0.85 * t  -- Chloe's overall correct answers
noncomputable def together_correct : ℝ := chloe_total_correct - chloe_alone_correct  -- Problems solved correctly together
noncomputable def zoe_alone_correct : ℝ := 0.85 * (1/3 * t)  -- Zoe's correct answers alone
noncomputable def zoe_total_correct : ℝ := zoe_alone_correct + together_correct  -- Zoe's total correct answers
noncomputable def zoe_percentage_correct : ℝ := (zoe_total_correct / t) * 100  -- Convert to percentage

theorem zoe_correct_percentage : zoe_percentage_correct = 90 := 
by
  sorry

end zoe_correct_percentage_l390_390511


namespace student_between_two_girls_l390_390025

theorem student_between_two_girls (students : Fin 50 → Prop)
  (boy girl : Prop)
  (h_boys : ∀ i : Fin 50, students i → (i < 25 → students i = boy))
  (h_girls : ∀ i : Fin 50, students i → (i ≥ 25 → students i = girl))
  (h_random : ∃! circ : ℤ, circ ∈ {0, ..., 49}) :
  ∃ i : Fin 50, students (i + 1) mod 50 = girl ∧ students (i + 2) mod 50 = girl := 
by
  sorry

end student_between_two_girls_l390_390025


namespace find_solution_set_l390_390335

-- Define the problem
def absolute_value_equation_solution_set (x : ℝ) : Prop :=
  |x - 2| + |2 * x - 3| = |3 * x - 5|

-- Define the expected solution set
def solution_set (x : ℝ) : Prop :=
  x ≤ 3 / 2 ∨ 2 ≤ x

-- The proof problem statement
theorem find_solution_set :
  ∀ x : ℝ, absolute_value_equation_solution_set x ↔ solution_set x :=
sorry -- No proof required, so we use 'sorry' to skip the proof

end find_solution_set_l390_390335


namespace max_language_words_l390_390746
-- Import the entirety of the math library

theorem max_language_words (n : ℕ) : 
  ∀ (N : ℕ), 
  (∀ (w1 w2 : vector (bool) n), w1 ≠ w2 → (hdistance w1 w2 >= 3)) → 
  N ≤ 2^n / (n + 1) :=
by
  sorry

end max_language_words_l390_390746


namespace maximum_distance_travel_l390_390889

theorem maximum_distance_travel (front_tire_lifespan rear_tire_lifespan : ℕ) 
  (h1 : front_tire_lifespan = 20000) 
  (h2 : rear_tire_lifespan = 30000) : 
  ∃ max_distance, max_distance = 24000 :=
by {
  existsi 24000,
  rw [h1, h2],
  sorry
}

end maximum_distance_travel_l390_390889


namespace second_derivative_at_e_l390_390154

noncomputable def f (x : ℝ) : ℝ :=
  (1 - x) * Real.log x

def f'' (x : ℝ) : ℝ :=
  -1 / x - 1 / (x^2)

theorem second_derivative_at_e : f'' Real.exp 1 = 1 / Real.exp 1 - 2 :=
by
  sorry

end second_derivative_at_e_l390_390154


namespace eccentricity_range_l390_390262

noncomputable def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  { z : ℝ × ℝ // (z.1^2 / a^2) + (z.2^2 / b^2) = 1 }

theorem eccentricity_range (a b : ℝ) (h : a > b) (hb : b > 0):
  ∀ (P : ellipse a b h hb),
    |(P.1, P.2 - b).norm ≤ 2 * b → 
    eccentricity (a b) ∈ set.Ici (real.sqrt 2 / 2) :=
sorry

end eccentricity_range_l390_390262


namespace min_black_cells_needed_l390_390403

/-- 
Given a 7x7 grid, prove that the minimum number of cells that need to be painted black
to ensure there is no white rectangle containing at least 10 white cells is 4.
-/
theorem min_black_cells_needed : ∃ (n : ℕ), 
  n = 4 ∧ 
  (∀ (grid : matrix (fin 7) (fin 7) bool), 
    (∀ (r₁ r₂ c₁ c₂ : ℕ), 
      r₁ ≤ r₂ → c₁ ≤ c₂ → 
      (r₂-r₁+1) * (c₂-c₁+1) ≥ 10 → 
      (∃ (i j : fin 7), grid i j = tt)) →
    (∃ (black_cells : fin 7 × fin 7 → bool), 
      (∀ (i j : fin 7), black_cells i j = tt → grid i j = tt) ∧ 
      ∑ i j, (if black_cells i j = tt then 1 else 0) = n
  )) :=  
sorry

end min_black_cells_needed_l390_390403


namespace tenth_term_arithmetic_sequence_l390_390433

theorem tenth_term_arithmetic_sequence :
  let a_1 := (1 : ℝ) / 2
  let a_2 := (5 : ℝ) / 6
  let d := a_2 - a_1
  (a_1 + 9 * d) = 7 / 2 := 
by
  sorry

end tenth_term_arithmetic_sequence_l390_390433


namespace range_of_eccentricity_l390_390299

-- Definition of the ellipse and its properties
namespace EllipseProof

variables {a b : ℝ} (h : a > b ∧ b > 0)

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def upper_vertex : ℝ × ℝ := (0, b)

def is_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def condition_on_point (P : ℝ × ℝ) : Prop := dist P upper_vertex ≤ 2 * b

-- The eccentricity of the ellipse
def eccentricity : ℝ := real.sqrt (1 - (b^2 / a^2))

-- The required proof statement
theorem range_of_eccentricity :
  (∀ P, is_on_ellipse P → condition_on_point P) →
  eccentricity ≤ real.sqrt 2 / 2 := sorry

end EllipseProof

end range_of_eccentricity_l390_390299


namespace correct_proposition_l390_390681

variables {Point Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Conditions
axiom perpendicular (m : Line) (α : Plane) : Prop
axiom parallel (n : Line) (α : Plane) : Prop

-- Specific conditions given
axiom m_perp_α : perpendicular m α
axiom n_par_α : parallel n α

-- Statement to prove
theorem correct_proposition : perpendicular m n := sorry

end correct_proposition_l390_390681


namespace real_root_range_of_a_l390_390115

theorem real_root_range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + x + |a - 1/4| + |a| = 0) ↔ (0 ≤ a ∧ a ≤ 1/4) :=
by
  sorry

end real_root_range_of_a_l390_390115


namespace average_age_of_X_union_Y_union_Z_l390_390012

variable (X Y Z : Type) [Fintype X] [Fintype Y] [Fintype Z]
variable (x y z : ℕ)
variable (X_total_age Y_total_age Z_total_age : ℕ)

-- Conditions
axiom H1 : Disjoint X Y
axiom H2 : Disjoint X Z
axiom H3 : Disjoint Y Z
axiom H4 : X_total_age / Fintype.card X = 40
axiom H5 : Y_total_age / Fintype.card Y = 25
axiom H6 : Z_total_age / Fintype.card Z = 45
axiom H7 : (X_total_age + Y_total_age) / (Fintype.card X + Fintype.card Y) = 32
axiom H8 : (X_total_age + Z_total_age) / (Fintype.card X + Fintype.card Z) = 42
axiom H9 : (Y_total_age + Z_total_age) / (Fintype.card Y + Fintype.card Z) = 35

theorem average_age_of_X_union_Y_union_Z : 
  (X_total_age + Y_total_age + Z_total_age) / (Fintype.card X + Fintype.card Y + Fintype.card Z) = 34.26 :=
sorry

end average_age_of_X_union_Y_union_Z_l390_390012


namespace total_red_and_green_peaches_l390_390399

-- Define the number of red peaches and green peaches.
def red_peaches : ℕ := 6
def green_peaches : ℕ := 16

-- Theorem stating the sum of red and green peaches is 22.
theorem total_red_and_green_peaches : red_peaches + green_peaches = 22 := 
by
  -- Proof would go here but is not required
  sorry

end total_red_and_green_peaches_l390_390399


namespace difference_between_max_and_min_coins_l390_390358

/-- Prove that given Paul owes Patricia 80 cents and has 10-cent, 20-cent,
    and 50-cent coins, and given that he must use at least one of each type,
    the difference between the largest and the smallest number of coins he can use
    to pay her is 0. -/
theorem difference_between_max_and_min_coins
  (owes : ℕ) (coins : list ℕ) (has_each : ∀ c ∈ coins, c ∈ [10, 20, 50]) :
  owes = 80 ∧ coins = [10, 20, 50] →
  ∃ min_coins max_coins : ℕ,
    (min_coins = 3 ∧ max_coins = 3 ∧ max_coins - min_coins = 0) :=
by
  sorry

end difference_between_max_and_min_coins_l390_390358


namespace range_of_eccentricity_l390_390251

noncomputable def upperVertex (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : ℝ × ℝ := (0, b)

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem range_of_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h: a > b) :
  (∀ (x y : ℝ), ellipse a b x y → distance x y 0 b ≤ 2 * b) →
  ∃ (e : ℝ), e ∈ Set.Ioc 0 (Real.sqrt 2 / 2) ∧ a^2 = b^2 / (1 - e^2) := sorry

end range_of_eccentricity_l390_390251


namespace max_g_l390_390667

def g (x : ℝ) : ℝ := Real.sqrt (x * (80 - x)) + Real.sqrt (x * (5 - x))

theorem max_g (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 5) : 
  ∃ x1 N, g x1 = N ∧ N = 20 ∧ x1 = (80 : ℝ) / 17 :=
sorry

end max_g_l390_390667


namespace number_of_correct_propositions_l390_390926

theorem number_of_correct_propositions :
  let P1 := ∀ a b: ℝ, b < a ∧ a < 0 → |a| ≤ |b|
  let P2 := ∀ a b: ℝ, b < a ∧ a < 0 → a + b < ab
  let P3 := ∀ a b: ℝ, b < a ∧ a < 0 → (b / a) + (a / b) > 2
  let P4 := ∀ a b: ℝ, b < a ∧ a < 0 → (a^2) / b < 2 * a - b
  let P5 := ∀ a b: ℝ, b < a ∧ a < 0 → (2 * a + b) / (a + 2 * b) > a / b
  let P6 := ∀ a b: ℝ, a + b = 1 → a^2 + b^2 ≥ 1 / 2
  count (λ P, P = true) [P1, P2, P3, P4, P5, P6] = 5 :=
sorry

end number_of_correct_propositions_l390_390926


namespace first_row_number_l390_390866

-- Definitions of the constraints
def filled_with_1_to_5 (lst : List ℕ) : Prop :=
  ∀ n ∈ lst, n ∈ (List.range 1 (5+1)) ∧ lst.nodup

def valid_grid (grid : List (List ℕ)) : Prop :=
  (∀ row ∈ grid, filled_with_1_to_5 row) ∧
  (∀ i j < 5, grid.map (fun row => row.nth_le j sorry) = (List.range 1 (5+1)).erase_dup)

-- The given grid conditions
def grid_conditions (grid : List (List ℕ)) : Prop :=
  (True) -- placeholder for specific sum or product conditions between adjacent cells

-- The problem statement
theorem first_row_number (grid : List (List ℕ)) (h_valid : valid_grid grid) (h_conditions : grid_conditions grid) :
  grid.nth 0 = some [2, 5, 1, 3, 4] →
  [2, 5, 1, 3, 4].foldl (λ acc digit, acc * 10 + digit) 0 = 25134 :=
  sorry

end first_row_number_l390_390866


namespace intersection_two_elements_l390_390938

open Real Set

-- Definitions
def M (k : ℝ) : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ y = k * (x - 1) + 1}
def N : Set (ℝ × ℝ) := {p | ∃ x y : ℝ, p = (x, y) ∧ x^2 + y^2 - 2 * y = 0}

-- Statement of the problem
theorem intersection_two_elements (k : ℝ) (hk : k ≠ 0) :
  ∃ x1 y1 x2 y2 : ℝ,
    (x1, y1) ∈ M k ∧ (x1, y1) ∈ N ∧ 
    (x2, y2) ∈ M k ∧ (x2, y2) ∈ N ∧ 
    (x1, y1) ≠ (x2, y2) := sorry

end intersection_two_elements_l390_390938


namespace trapezium_area_l390_390536

theorem trapezium_area (a b h : ℝ) (h₁ : a = 30) (h₂ : b = 12) (h₃ : h = 16) : 
  (1/2 * (a + b) * h = 336) :=
by
  rw [h₁, h₂, h₃]
  norm_num
  -- left hand side becomes (1 / 2) * (30 + 12) * 16 = (1 / 2) * 42 * 16
  -- simplify to get 336
  sorry

end trapezium_area_l390_390536


namespace triangle_y_values_l390_390054

theorem triangle_y_values (y : ℕ) :
  (8 + 11 > y^2) ∧ (y^2 + 8 > 11) ∧ (y^2 + 11 > 8) ↔ y = 2 ∨ y = 3 ∨ y = 4 :=
by
  sorry

end triangle_y_values_l390_390054


namespace rectangle_side_ratio_square_l390_390196

noncomputable def ratio_square (a b : ℝ) : ℝ :=
(a / b) ^ 2

theorem rectangle_side_ratio_square (a b : ℝ) (h : (a - b) / (a + b) = 1 / 3) : 
  ratio_square a b = 4 := by
  sorry

end rectangle_side_ratio_square_l390_390196


namespace constant_term_expansion_l390_390537

theorem constant_term_expansion :
  let expansion := (1 + x + (1 / x^2)) ^ 10
  in (expansion.constant_term) = 4351 :=
by
  sorry

end constant_term_expansion_l390_390537


namespace omega_2n_eq_omega_n_omega_8n_plus_5_eq_omega_4n_plus_3_omega_2_pow_n_minus_1_eq_n_l390_390325

def bit_weight (n : ℕ) : ℕ :=
  (n.bits.map (λ b, if b then 1 else 0)).sum

theorem omega_2n_eq_omega_n (n : ℕ) : 
  bit_weight (2 * n) = bit_weight n := 
sorry

theorem omega_8n_plus_5_eq_omega_4n_plus_3 (n : ℕ) : 
  bit_weight (8 * n + 5) = bit_weight (4 * n + 3) := 
sorry

theorem omega_2_pow_n_minus_1_eq_n (n : ℕ) : 
  bit_weight (2 ^ n - 1) = n := 
sorry

end omega_2n_eq_omega_n_omega_8n_plus_5_eq_omega_4n_plus_3_omega_2_pow_n_minus_1_eq_n_l390_390325


namespace necessary_but_not_sufficient_l390_390751

theorem necessary_but_not_sufficient (x : ℝ) : (1 - x) * (1 + |x|) > 0 -> x < 2 :=
by
  sorry

end necessary_but_not_sufficient_l390_390751


namespace a_10_eq_1024_l390_390968

noncomputable def a : ℕ → ℝ
| n := sorry

axiom a_property : ∀ m n : ℕ, m > 0 → n > 0 → a (m + n) = a m * a n

axiom a_3 : a 3 = 8

theorem a_10_eq_1024 : a 10 = 1024 :=
sorry

end a_10_eq_1024_l390_390968


namespace find_investment_duration_l390_390100

theorem find_investment_duration :
  ∀ (A P R I : ℝ) (T : ℝ),
    A = 1344 →
    P = 1200 →
    R = 5 →
    I = A - P →
    I = (P * R * T) / 100 →
    T = 2.4 :=
by
  intros A P R I T hA hP hR hI1 hI2
  sorry

end find_investment_duration_l390_390100


namespace average_mpg_is_20_77_l390_390805

variables (x : ℝ) -- Distance from Town B to Town C in miles
variables (d_AB : ℝ) (d_BC : ℝ) -- Distance variables
variables (r_AB : ℝ) (r_BC : ℝ) -- Fuel consumption rates
variables (F_total : ℝ) (mpi : ℝ) -- Total Fuel Consumption and Miles per gallon

-- Define distances based on condition
def distance_AB := 2 * x
def distance_BC := x

-- Define fuel consumption rates
def rate_AB := 20 -- miles per gallon
def rate_BC := 22.5 -- miles per gallon

-- Calculate fuel consumed for each segment
def fuel_AB := distance_AB x / rate_AB
def fuel_BC := distance_BC x / rate_BC

-- Calculate total fuel consumed
def fuel_total := fuel_AB x + fuel_BC x

-- Calculate total distance traveled
def distance_total := distance_AB x + distance_BC x

-- Calculate average miles per gallon
def average_mpg := distance_total x / fuel_total x

-- Now our main theorem: Average miles per gallon is 20.77
theorem average_mpg_is_20_77 (x : ℝ) : (distance_total x / fuel_total x) = 20.77 := by
  sorry

end average_mpg_is_20_77_l390_390805


namespace odd_sum_probability_l390_390490

theorem odd_sum_probability : 
  let nums := {1, 2, 3, 4, 5} in
  let total_ways := (@Finset.card _ _ (Finset.filter (λ s, Finset.card s = 3) (Finset.powerset nums))) in
  let favorable_ways := (@Finset.card _ _ (Finset.filter (λ s, (∃ (a b c : ℕ), 
    (s = {a, b, c} ∧ ∃ e1 e2 : Finset ℕ, e1 ∪ e2 = nums \ s ∧ 
  Finset.card e1 = 1 ∧ Finset.card e2 = 1 ∧ 
  Finset.sum e1 % 2 = 1 ∧ Finset.sum e2 % 2 = 0 )) ) (Finset.filter (λ s,Finset.card s = 3) (Finset.powerset nums)))) in
  (favorable_ways : ℝ) / (total_ways : ℝ) = 0.6 :=
by {
  sorry
}

end odd_sum_probability_l390_390490


namespace professors_chairs_l390_390732

theorem professors_chairs :
  let chairs := 10
  let professors := 3
  let students := 6
  let feasible_chairs := {2, 4, 6, 8}
  -- The set of positions such that no two professors are adjacent and every professor is surrounded by students
  (∑ p in (finset.powerset_len 3 feasible_chairs), p.attach.card_perm) = 24 :=
by
  sorry

end professors_chairs_l390_390732


namespace three_person_subcommittees_from_eight_l390_390167

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem three_person_subcommittees_from_eight (n k : ℕ) (h_n : n = 8) (h_k : k = 3) :
  combination n k = 56 := by
  sorry

end three_person_subcommittees_from_eight_l390_390167


namespace z2_in_first_quadrant_iff_min_value_norm_conjugate_z1_add_z2_l390_390320

noncomputable def z1 : ℂ := 1 + 2 * complex.I
noncomputable def z2 (a : ℝ) : ℂ := (a + complex.I) * (1 - complex.I) * complex.I

theorem z2_in_first_quadrant_iff {a : ℝ} : (0 < (z2 a).re ∧ 0 < (z2 a).im) ↔ (-1 < a ∧ a < 1) :=
sorry

theorem min_value_norm_conjugate_z1_add_z2 {a : ℝ} : ∀ a, ∃ a_min : ℝ, 
  a_min = -3 / 2 ∧ 
  |complex.norm ((complex.conj z1) + (z2 a))| = (sqrt 2) / 2  :=
sorry

end z2_in_first_quadrant_iff_min_value_norm_conjugate_z1_add_z2_l390_390320


namespace erika_pie_balls_covered_l390_390070

theorem erika_pie_balls_covered (rate : ℝ) :
  let radius_erika := 7
  let radius_laura := 9
  let radius_carlos := 14
  let surface_area (r : ℝ) := 4 * real.pi * r^2
  let sa_erika := surface_area radius_erika
  let sa_laura := surface_area radius_laura
  let sa_carlos := surface_area radius_carlos
  let lcm_surface_areas := int.lcm (int.lcm (196 * π) (324 * π)) (784 * π)
  sa_erika / lcm_surface_areas = 84 :=
by
  sorry

end erika_pie_balls_covered_l390_390070


namespace ω_2n_eq_ω_n_ω_8n_5_eq_ω_4n_3_ω_2n_minus_1_eq_n_l390_390332

def ω (n : ℕ) : ℕ :=
  n.bits.count (λ b => b)

theorem ω_2n_eq_ω_n (n : ℕ) : ω (2 * n) = ω n := by
  sorry

theorem ω_8n_5_eq_ω_4n_3 (n : ℕ) : ω (8 * n + 5) = ω (4 * n + 3) := by
  sorry

theorem ω_2n_minus_1_eq_n (n : ℕ) : ω (2^n - 1) = n := by
  sorry

end ω_2n_eq_ω_n_ω_8n_5_eq_ω_4n_3_ω_2n_minus_1_eq_n_l390_390332


namespace pirate_coins_l390_390704

theorem pirate_coins (x : ℕ) (hx : x = 9) :
  (∑ i in finset.range (x + 1), i) = 5 * x → 6 * x = 54 :=
by
  sorry

end pirate_coins_l390_390704


namespace circle_area_from_points_l390_390706

theorem circle_area_from_points (C D : ℝ × ℝ) (hC : C = (2, 3)) (hD : D = (8, 9)) : 
  ∃ A : ℝ, A = 18 * Real.pi :=
by
  sorry

end circle_area_from_points_l390_390706


namespace not_always_two_circles_tangent_not_always_log_inequality_not_subset_sets_l390_390525

-- 1st proof problem rewritten
theorem not_always_two_circles_tangent 
  (P Q : ℝ × ℝ) 
  (l : set (ℝ × ℝ)) (h1 : P ≠ Q) (h2 : same_side l P Q) :
  ¬ (∃ c1 c2, c1 ≠ c2 ∧ passes_through P c1 ∧ passes_through Q c1 ∧ is_tangent l c1 ∧ passes_through P c2 ∧ passes_through Q c2 ∧ is_tangent l c2) :=
begin
  sorry
end

-- 2nd proof problem rewritten
theorem not_always_log_inequality 
  {a b : ℝ} (ha : a > 0) (hb : b > 0) (h1 : a ≠ 1) (h2 : b = 1) :
  ¬ (log a b + log b a ≥ 2) :=
begin
  sorry
end

-- 3rd proof problem rewritten
theorem not_subset_sets 
  (A B : set (ℝ × ℝ)) 
  (h : ∀ r : ℝ, r ≥ 0 → C r ∪ A ⊆ C r ∪ B) :
  ¬ (A ⊆ B) :=
begin
  sorry
end

end not_always_two_circles_tangent_not_always_log_inequality_not_subset_sets_l390_390525


namespace k_ge_n_l390_390671

theorem k_ge_n (n : ℕ) (h1 : 2 ≤ n) (k : ℕ) (n_i : Fin k → ℕ)
  (h2 : (2^n - 1) ∣ (Finset.univ.sum (λ i, 2 ^ (n_i i)))) : k ≥ n :=
sorry

end k_ge_n_l390_390671


namespace num_zeros_right_of_decimal_before_nonzero_l390_390951

theorem num_zeros_right_of_decimal_before_nonzero : 
  ∃ (a b : ℕ), (a = 4) ∧ (b = 9) ∧ (decimal_expansion (1 / (2^7 * 5^9)) = (a, b)) → a = 4 ∧ b = 9 ∧ num_zeros 1 (2^7 * 5^9) = 8 := 
by
  sorry

end num_zeros_right_of_decimal_before_nonzero_l390_390951


namespace extreme_values_tangent_line_l390_390150

noncomputable def f (x : ℝ) : ℝ := x^3 + x^2 - x + 2

theorem extreme_values :
  (∀ x, deriv f x = 0 → (x = -1 ∨ x = 1/3)) ∧
  f (-1) = 3 ∧ f (1/3) = 49/27 := 
sorry

theorem tangent_line (P : ℝ × ℝ) :
  P = (1, 3) →
  ( ∃ m b1 b2 : ℝ, (m = 4 ∧ b1 = 1 ∧ 4 * 1 - 3 - b1 = 0 ∧ 
  ( (λ x, 4 * x - 3 - b1 = 0) ∨ (λ x, 3 - 3 = 0) ) ) ∨ 
  (λ x, (x-1)^2 * (x+1) = 0 → x = 1 ∨ x = -1) ∨ 
  (λ x, f x = 3 * x^2 + 2 * x - 1) // to confirm tangent for m = 0)) :=
sorry

end extreme_values_tangent_line_l390_390150


namespace price_reduction_for_target_profit_l390_390806
-- Import the necessary libraries

-- Define the conditions
def average_sales_per_day := 70
def initial_profit_per_item := 50
def sales_increase_per_dollar_decrease := 2

-- Define the functions for sales volume increase and profit per item
def sales_volume_increase (x : ℝ) : ℝ := 2 * x
def profit_per_item (x : ℝ) : ℝ := initial_profit_per_item - x

-- Define the function for daily profit
def daily_profit (x : ℝ) : ℝ := (profit_per_item x) * (average_sales_per_day + sales_volume_increase x)

-- State the main theorem
theorem price_reduction_for_target_profit : 
  ∃ x : ℝ, daily_profit x = 3572 ∧ x = 12 :=
sorry

end price_reduction_for_target_profit_l390_390806


namespace coefficient_of_friction_l390_390410

/-- Assume m, Pi and ΔL are positive real numbers, and g is the acceleration due to gravity. 
We need to prove that the coefficient of friction μ is given by Pi / (m * g * ΔL). --/
theorem coefficient_of_friction (m Pi ΔL g : ℝ) (h_m : 0 < m) (h_Pi : 0 < Pi) (h_ΔL : 0 < ΔL) (h_g : 0 < g) :
  ∃ μ : ℝ, μ = Pi / (m * g * ΔL) :=
sorry

end coefficient_of_friction_l390_390410


namespace snake_can_turn_exists_l390_390802

noncomputable def exists_snake_that_can_turn (n : ℕ) : Prop :=
∃ k (S : Fin n → Fin n → List (Fin (n * n))),
  0.9 * n^2 ≤ k ∧ 
  (∀ i < k - 1, (S i i.succ) ∈ neighbor_cells (S i)) ∧
  turns_around S

theorem snake_can_turn_exists : ∃ n > 1, exists_snake_that_can_turn n :=
sorry

end snake_can_turn_exists_l390_390802


namespace problem_solution_l390_390130

theorem problem_solution :
  ∃ x y z : ℕ,
    0 < x ∧ 0 < y ∧ 0 < z ∧
    x^2 + y^2 + z^2 = 2 * (y * z + 1) ∧
    x + y + z = 4032 ∧
    x^2 * y + z = 4031 :=
by
  sorry

end problem_solution_l390_390130


namespace problem_statement_l390_390899

-- Conditions
def p (a : ℝ) : Prop := a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a ^ x > 0
def q (x : ℝ) : Prop := x > 0 ∧ x ≠ 1 ∧ (Real.log 2 / Real.log x + Real.log x / Real.log 2 ≥ 2)

-- Theorem statement
theorem problem_statement (a x : ℝ) : ¬p a ∨ ¬q x :=
by sorry

end problem_statement_l390_390899


namespace is_isosceles_trapezoid_l390_390063

-- Definitions
variables {A B C D E : Type}

-- Assume polygons are regular implies certain properties
variables (h_eq_sides : ∀ (x y : A), x = y) (h_eq_angles : ∀ (z : A), z = 60)

-- Prove that ABCD is an isosceles trapezoid
theorem is_isosceles_trapezoid (A B C D : A) : 
  -- Assume sides AD and BC are equal
  AD = BC ∧
  -- Assuming constructed equilateral triangle properties
  (AB = BE ∧ BE = AE ∧ ∠ABE = 60 ∧ ∠BAE = 60 ∧ ∠EAB = 60) →
  -- Prove that ABCD is an isosceles trapezoid
  is_isosceles_trapezoid ABCD :=
begin
  sorry
end

end is_isosceles_trapezoid_l390_390063


namespace number_of_people_in_range_l390_390627

-- Define the conditions
def total_people : ℕ := 420
def selected_people : ℕ := 21
def range_min : ℕ := 241
def range_max : ℕ := 360
def sampling_interval : ℕ := total_people / selected_people

-- State the proof problem
theorem number_of_people_in_range (total_people = 420) (selected_people = 21)
  (range_min = 241) (range_max = 360) (sampling_interval = total_people / selected_people) :
  (range_max - range_min + 1) / sampling_interval = 6 :=
sorry

end number_of_people_in_range_l390_390627


namespace complex_multiplication_l390_390838

theorem complex_multiplication (a b c d : ℤ) (i : ℂ) (h : i^2 = -1) :
  ((a + b * i) * (c + d * i)) = (-6 + 33 * i) :=
by
  have a := 3
  have b := -4
  have c := -6
  have d := 3
  sorry

end complex_multiplication_l390_390838


namespace trigonometric_identity_l390_390524

/-- 
Theorem: Prove that the value of the given trigonometric expression 
         is equal to -1/2.
-/
theorem trigonometric_identity : 
  sin 7 * cos 37 - sin 83 * cos 53 = -1 / 2 := 
sorry

end trigonometric_identity_l390_390524


namespace ranking_order_l390_390659

variable (Person : Type)
variable (Wu Bing Jia Ding Yi : Person)

axiom faster_than : Person → Person → Prop

variables (H1 : faster_than Ding Yi) 
variables (H2 : faster_than Wu Bing) 
variables (H3 : faster_than Bing Jia) (H3' : faster_than Jia Ding)

theorem ranking_order :
  (faster_than Wu Bing) ∧ (faster_than Bing Jia) ∧ (faster_than Jia Ding) ∧ (faster_than Ding Yi) →
  faster_than Wu Bing ∧ faster_than Bing Jia ∧ faster_than Jia Ding ∧ faster_than Ding Yi 
∧ faster_than Wu Jia ∧ faster_than Wu Ding ∧ faster_than Wu Yi 
∧ faster_than Bing Ding ∧ faster_than Bing Yi 
∧ faster_than Jia Yi :=
by {
  intro h,
  sorry,
}

end ranking_order_l390_390659


namespace unique_function_solution_l390_390871

theorem unique_function_solution :
  (∃! (f : ℝ → ℝ), ∀ x y : ℝ, f (x + y) * f (x - y) = (f x ^ 3 + f y ^ 3 + 3 * x * y) - 3 * x ^ 2 * y ^ 2 * f x) ↔ 
  (∃! f : ℝ → ℝ, f = λ x, 0) := 
by
  sorry

end unique_function_solution_l390_390871


namespace bob_current_time_l390_390836

noncomputable def sister_time_secs := 5 * 60 + 20
noncomputable def improvement_factor := 1.5

theorem bob_current_time :
  let bob_current_time_secs := sister_time_secs * improvement_factor in
  bob_current_time_secs = 8 * 60 :=
by
  let bob_current_time_secs := sister_time_secs * improvement_factor
  have : bob_current_time_secs = 480 := by
    sorry -- This will be filled in with the actual proof steps.
  show bob_current_time_secs = 480
  exact this

end bob_current_time_l390_390836


namespace parallel_line_distance_equation_l390_390539

theorem parallel_line_distance_equation :
  ∃ m : ℝ, (m = -20 ∨ m = 32) ∧
  ∀ x y : ℝ, (5 * x - 12 * y + 6 = 0) → 
            (5 * x - 12 * y + m = 0) :=
by
  sorry

end parallel_line_distance_equation_l390_390539


namespace eccentricity_range_l390_390281

noncomputable def ellipse_eccentricity (a b : ℝ) (h_ab : a > b) : ℝ := (Real.sqrt (a ^ 2 - b ^ 2)) / a

theorem eccentricity_range (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ) :
  (∃ x y : ℝ, (x, y) = P ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  (Real.sqrt (x^2 + (y - b)^2) ≤ 2 * b)) → 
  0 < ellipse_eccentricity a b h_ab ∧ ellipse_eccentricity a b h_ab ≤ Real.sqrt 2 / 2 :=
begin
  sorry
end

end eccentricity_range_l390_390281


namespace area_of_triangle_PQR_l390_390127

noncomputable def point := ℝ × ℝ

def P : point := (1, 1)
def Q : point := (4, 1)
def R : point := (3, 4)

def triangle_area (A B C : point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)

theorem area_of_triangle_PQR :
  triangle_area P Q R = 9 / 2 :=
by
  sorry

end area_of_triangle_PQR_l390_390127


namespace percentage_loss_l390_390824

noncomputable def costPrice : ℝ := 200
noncomputable def sellingPriceProfit : ℝ := 240
noncomputable def sellingPriceLoss : ℝ := 170

theorem percentage_loss : (costPrice - sellingPriceLoss) / costPrice * 100 = 15 := by
  have h₁ : costPrice = 240 / 1.20 := rfl
  have h₂ : costPrice - sellingPriceLoss = 30 := by
    rw [h₁]
    norm_num
  have h₃ : (costPrice - sellingPriceLoss) / costPrice * 100 = 15 := by
    rw [h₂]
    norm_num
  exact h₃

#check percentage_loss -- This checks if the theorem is well-formed

end percentage_loss_l390_390824


namespace days_not_worked_correct_l390_390487

def total_days : ℕ := 20
def earnings_for_work (days_worked : ℕ) : ℤ := 80 * days_worked
def penalty_for_no_work (days_not_worked : ℕ) : ℤ := -40 * days_not_worked
def final_earnings (days_worked days_not_worked : ℕ) : ℤ := 
  (earnings_for_work days_worked) + (penalty_for_no_work days_not_worked)
def received_amount : ℤ := 880

theorem days_not_worked_correct {y x : ℕ} 
  (h1 : x + y = total_days) 
  (h2 : final_earnings x y = received_amount) :
  y = 6 :=
sorry

end days_not_worked_correct_l390_390487


namespace value_of_a8_l390_390830

variable (a : ℕ → ℝ) (a_1 : a 1 = 2) (common_sum : ℝ) (h_sum : common_sum = 5)
variable (equal_sum_sequence : ∀ n, a (n + 1) + a n = common_sum)

theorem value_of_a8 : a 8 = 3 :=
sorry

end value_of_a8_l390_390830


namespace number_of_cakes_l390_390739

theorem number_of_cakes (total_eggs eggs_in_fridge eggs_per_cake : ℕ) (h1 : total_eggs = 60) (h2 : eggs_in_fridge = 10) (h3 : eggs_per_cake = 5) :
  (total_eggs - eggs_in_fridge) / eggs_per_cake = 10 :=
by
  sorry

end number_of_cakes_l390_390739


namespace smallest_positive_period_of_f_is_pi_minimum_value_of_f_is_neg2_f_is_strictly_increasing_on_intervals_l390_390875

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - (Real.cos x) ^ 4

theorem smallest_positive_period_of_f_is_pi : (∀ x : ℝ, f (x + π) = f x) where π > 0 := 
  sorry

theorem minimum_value_of_f_is_neg2 : (∀ x : ℝ, f x ≥ -2) ∧ (∃ x : ℝ, f x = -2) := 
  sorry

theorem f_is_strictly_increasing_on_intervals : (∀ x : ℝ, 0 ≤ x ∧ x ≤ π → ((x ∈ [0, π/3]) ∨ (x ∈ [5 * π / 6, π])) → ∀ y, x < y ∧ ((y ∈ [0, π/3]) ∨ (y ∈ [5 * π / 6, π])) → f x < f y) :=
  sorry

end smallest_positive_period_of_f_is_pi_minimum_value_of_f_is_neg2_f_is_strictly_increasing_on_intervals_l390_390875


namespace parallelogram_area_l390_390858

def vec1 : ℝ × ℝ × ℝ := (2, 4, -1)
def vec2 : ℝ × ℝ × ℝ := (1, -3, 5)

theorem parallelogram_area : (λ (v1 v2 : ℝ × ℝ × ℝ), real.sqrt ((v1.2 - v1.1 * v1.3) ^ 2 + (v2.1 - v2.2 * v2.3) ^ 2 + (v2.3 - v2.2 * v2.1) ^ 2)) vec1 vec2 = real.sqrt 510 :=
sorry

end parallelogram_area_l390_390858


namespace eccentricity_range_l390_390280

noncomputable def ellipse_eccentricity (a b : ℝ) (h_ab : a > b) : ℝ := (Real.sqrt (a ^ 2 - b ^ 2)) / a

theorem eccentricity_range (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ) :
  (∃ x y : ℝ, (x, y) = P ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  (Real.sqrt (x^2 + (y - b)^2) ≤ 2 * b)) → 
  0 < ellipse_eccentricity a b h_ab ∧ ellipse_eccentricity a b h_ab ≤ Real.sqrt 2 / 2 :=
begin
  sorry
end

end eccentricity_range_l390_390280


namespace student_correct_answers_l390_390792

variable (C I : ℕ) -- Define C and I as natural numbers
variable (score totalQuestions : ℕ) -- Define score and totalQuestions as natural numbers

-- Define the conditions
def grading_system (C I score : ℕ) : Prop := C - 2 * I = score
def total_questions (C I totalQuestions : ℕ) : Prop := C + I = totalQuestions

-- The theorem statement to prove
theorem student_correct_answers :
  (grading_system C I 76) ∧ (total_questions C I 100) → C = 92 := by
  sorry -- Proof to be filled in

end student_correct_answers_l390_390792


namespace number_of_integers_satisfying_property_l390_390945

theorem number_of_integers_satisfying_property :
  (finset.filter (λ n : ℤ, 100 < n^2 ∧ n^2 < 400)
    (finset.Icc 11 19)).card = 9 := 
by
  sorry

end number_of_integers_satisfying_property_l390_390945


namespace solve_for_x_l390_390056

theorem solve_for_x (a c : ℝ) : 
  let x := (c^2 - a^3) / (3 * a^2 - 1)
  in x^2 + c^2 = (a - x)^3 :=
by
  -- Proof is omitted
  sorry

end solve_for_x_l390_390056


namespace range_of_eccentricity_l390_390298

-- Definition of the ellipse and its properties
namespace EllipseProof

variables {a b : ℝ} (h : a > b ∧ b > 0)

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def upper_vertex : ℝ × ℝ := (0, b)

def is_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def condition_on_point (P : ℝ × ℝ) : Prop := dist P upper_vertex ≤ 2 * b

-- The eccentricity of the ellipse
def eccentricity : ℝ := real.sqrt (1 - (b^2 / a^2))

-- The required proof statement
theorem range_of_eccentricity :
  (∀ P, is_on_ellipse P → condition_on_point P) →
  eccentricity ≤ real.sqrt 2 / 2 := sorry

end EllipseProof

end range_of_eccentricity_l390_390298


namespace taco_truck_earnings_l390_390479

/-
Question: How many dollars did the taco truck make during the lunch rush?
Conditions:
1. Soft tacos are $2 each.
2. Hard shell tacos are $5 each.
3. The family buys 4 hard shell tacos and 3 soft tacos.
4. There are ten other customers.
5. Each of the ten other customers buys 2 soft tacos.
Answer: The taco truck made $66 during the lunch rush.
-/

theorem taco_truck_earnings :
  let soft_taco_price := 2
  let hard_taco_price := 5
  let family_hard_tacos := 4
  let family_soft_tacos := 3
  let other_customers := 10
  let other_customers_soft_tacos := 2
  (family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price +
   other_customers * other_customers_soft_tacos * soft_taco_price) = 66 := by
  sorry

end taco_truck_earnings_l390_390479


namespace exists_two_elements_l390_390813

variable (F : Finset (Finset ℕ))
variable (h1 : ∀ (A B : Finset ℕ), A ∈ F → B ∈ F → (A ∪ B) ∈ F)
variable (h2 : ∀ (A : Finset ℕ), A ∈ F → ¬ (3 ∣ A.card))

theorem exists_two_elements : ∃ (x y : ℕ), ∀ (A : Finset ℕ), A ∈ F → x ∈ A ∨ y ∈ A :=
by
  sorry

end exists_two_elements_l390_390813


namespace whole_numbers_between_sqrts_l390_390186

theorem whole_numbers_between_sqrts :
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let start := Nat.ceil lower_bound
  let end_ := Nat.floor upper_bound
  ∃ n, n = end_ - start + 1 ∧ n = 7 := by
  sorry

end whole_numbers_between_sqrts_l390_390186


namespace derivative_at_zero_is_zero_l390_390456

def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 2 * x^2 + x^2 * cos (1 / (9 * x))
  else 0

theorem derivative_at_zero_is_zero :
  (filter.tendsto (λ Δx : ℝ, (f Δx) / Δx) (nhds 0) (nhds 0)) :=
by
  sorry

end derivative_at_zero_is_zero_l390_390456


namespace problem_statement_l390_390565

noncomputable def trajectory_of_t (t : ℂ) (z : ℂ) : Prop :=
  abs t = 3 ∧ t.re ≠ 3 ∧ t.re ≠ -3

noncomputable def max_min_values_of_z (z : ℂ) : Prop :=
  ∃ (z₀ : ℂ), z = z₀ + 3 + 3 * complex.I * real.sqrt 3 ∧ 
  ∀ w : ℂ, (w = 9 ∨ w = 3) → (abs z = w)

theorem problem_statement (t z : ℂ) (H1 : z = t + 3 + 3 * real.sqrt 3 * complex.I)
  (H2 : ∃ (k : ℂ), t = complex.i * k ∧ (t + 3) / (t - 3) = complex.i * k) :
  trajectory_of_t t z ∧ max_min_values_of_z z := 
sorry

end problem_statement_l390_390565


namespace sqrt_expression_l390_390521

theorem sqrt_expression (a b c : ℤ) (h_c : squarefree c) :
    (sqrt (89 + 24 * sqrt 11) = a + b * sqrt c) →
    a + b + c = 20 :=
by
  -- Squarefree definition placeholder
  sorry

end sqrt_expression_l390_390521


namespace find_quartic_polynomial_l390_390546

noncomputable def p (x : ℝ) : ℝ := -(1 / 9) * x^4 + (40 / 9) * x^3 - 8 * x^2 + 10 * x + 2

theorem find_quartic_polynomial :
  p 1 = -3 ∧
  p 2 = -1 ∧
  p 3 = 1 ∧
  p 4 = -7 ∧
  p 0 = 2 :=
by
  sorry

end find_quartic_polynomial_l390_390546


namespace taxi_fare_distance_l390_390197

-- Definitions for the problem conditions
def first_fare (d : ℝ) : ℝ := 8.0
def subsequent_fare (d : ℝ) : ℝ := 0.8
def total_fare (d : ℝ) (total_distance : ℝ) : ℝ :=
  first_fare d + subsequent_fare d * ((total_distance - d) / d)

-- The main theorem to be proved
theorem taxi_fare_distance (d : ℝ) (total_distance : ℝ) (total_cost : ℝ) :
  total_fare d total_distance = total_cost → d = 1 / 5 :=
by
  sorry

end taxi_fare_distance_l390_390197


namespace problem1_problem2_l390_390896

variable {α : ℝ} (h : Real.tan α = 2)

theorem problem1 : (sin α + 2 * cos α) / (4 * cos α - sin α) = 2 :=
sorry

theorem problem2 : sin α * cos α + cos α ^ 2 = 3 / 5 :=
sorry

end problem1_problem2_l390_390896


namespace chef_made_10_cakes_l390_390737

-- Definitions based on the conditions
def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

-- Calculated values based on the definitions
def eggs_for_cakes : ℕ := total_eggs - eggs_in_fridge
def number_of_cakes : ℕ := eggs_for_cakes / eggs_per_cake

-- Theorem to prove
theorem chef_made_10_cakes : number_of_cakes = 10 := by
  sorry

end chef_made_10_cakes_l390_390737


namespace isosceles_to_equilateral_l390_390642

noncomputable def sum_possible_values (angle : ℕ) : ℕ :=
  if angle = 50 then 145 else 0

theorem isosceles_to_equilateral (y : ℕ) (y1 y2 y3 : ℕ)
  (h1 : ∀ y, y = 65 ∨ y = 80)
  (h2 : 65 + 80 = 145) :
  sum_possible_values 50 = 145 :=
by {
  unfold sum_possible_values,
  simp [h2]
}

end isosceles_to_equilateral_l390_390642


namespace percent_of_x_is_v_l390_390957

variables (x y z v : ℝ)

-- Define the conditions
def cond1 : Prop := 0.45 * z = 0.39 * y
def cond2 : Prop := y = 0.75 * x
def cond3 : Prop := v = 0.8 * z

-- Define the statement to be proven
theorem percent_of_x_is_v (h1 : cond1 x y z) (h2 : cond2 x y) (h3 : cond3 z v) : v = 0.52 * x :=
sorry

end percent_of_x_is_v_l390_390957


namespace tony_lego_sets_l390_390237

theorem tony_lego_sets
  (price_lego price_sword price_dough : ℕ)
  (num_sword num_dough total_cost : ℕ)
  (L : ℕ)
  (h1 : price_lego = 250)
  (h2 : price_sword = 120)
  (h3 : price_dough = 35)
  (h4 : num_sword = 7)
  (h5 : num_dough = 10)
  (h6 : total_cost = 1940)
  (h7 : total_cost = price_lego * L + price_sword * num_sword + price_dough * num_dough) :
  L = 3 := 
by
  sorry

end tony_lego_sets_l390_390237


namespace total_marbles_correct_l390_390082

def first_jar : ℕ := 80
def second_jar (fj : ℕ) : ℕ := 2 * fj
def third_jar (fj : ℕ) : ℕ := fj / 4

theorem total_marbles_correct (fj : ℕ) (sj : ℕ) (tj : ℕ) : 
  sj = second_jar fj → tj = third_jar fj → fj + sj + tj = 260 :=
by {
  intros h1 h2,
  rw [h1, h2],
  exact calc
    fj + second_jar fj + third_jar fj
        = 80 + 2 * 80 + 80 / 4 : by rw [(show fj = 80, from rfl)]
    ... = 80 + 160 + 20       : by norm_num
    ... = 260                 : by norm_num,
}

end total_marbles_correct_l390_390082


namespace original_wattage_l390_390032

theorem original_wattage (W : ℝ) (h1 : 143 = 1.30 * W) : W = 110 := 
by
  sorry

end original_wattage_l390_390032


namespace omega_2n_eq_omega_n_not_omega_2n_plus_3_eq_omega_n_plus_1_omega_8n_plus_5_eq_omega_4n_plus_3_omega_pow2_n_minus_1_eq_n_l390_390327

def bin_weight (n : ℕ) : ℕ := n.binary_digits.count 1

theorem omega_2n_eq_omega_n (n : ℕ) : bin_weight (2 * n) = bin_weight n := by
  sorry

theorem not_omega_2n_plus_3_eq_omega_n_plus_1 (n : ℕ) : bin_weight (2 * n + 3) ≠ bin_weight n + 1 := by
  sorry

theorem omega_8n_plus_5_eq_omega_4n_plus_3 (n : ℕ) : bin_weight (8 * n + 5) = bin_weight (4 * n + 3) := by
  sorry

theorem omega_pow2_n_minus_1_eq_n (n : ℕ) : bin_weight (2^n - 1) = n := by
  sorry

end omega_2n_eq_omega_n_not_omega_2n_plus_3_eq_omega_n_plus_1_omega_8n_plus_5_eq_omega_4n_plus_3_omega_pow2_n_minus_1_eq_n_l390_390327


namespace largest_alpha_l390_390574

theorem largest_alpha (a b : ℕ) (h1 : a < b) (h2 : b < 2 * a) (N : ℕ) :
  ∃ (α : ℝ), α = 1 / (2 * a^2 - 2 * a * b + b^2) ∧
  (∃ marked_cells : ℕ, marked_cells ≥ α * (N:ℝ)^2) :=
by
  sorry

end largest_alpha_l390_390574


namespace razors_blades_equation_l390_390477

/-- Given the number of razors sold x,
each razor sold brings a profit of 30 yuan,
each blade sold incurs a loss of 0.5 yuan,
the number of blades sold is twice the number of razors sold,
and the total profit from these two products is 5800 yuan,
prove that the linear equation is -0.5 * 2 * x + 30 * x = 5800 -/
theorem razors_blades_equation (x : ℝ) :
  -0.5 * 2 * x + 30 * x = 5800 := 
sorry

end razors_blades_equation_l390_390477


namespace modulus_of_z_conjugate_of_z_root_value_of_m_l390_390123

def complex_modulus (z : ℂ) : ℝ := complex.abs z
def complex_conjugate (z : ℂ) : ℂ := complex.conj z

theorem modulus_of_z (z : ℂ) (h : z = 3 + 4 * complex.I) : complex_modulus z = 5 :=
by
  sorry

theorem conjugate_of_z (z : ℂ) (h : z = 3 + 4 * complex.I) : complex_conjugate z = 3 - 4 * complex.I :=
by
  sorry

theorem root_value_of_m (z : ℂ) (h : z = 3 + 4 * complex.I) (h_root : z^2 - 6*z + (25 : ℂ) = 0) : (25 : ℝ) = 25 :=
by
  sorry

end modulus_of_z_conjugate_of_z_root_value_of_m_l390_390123


namespace chord_length_of_intersection_l390_390493

def ellipse (x y : ℝ) := x^2 + 4 * y^2 = 16
def line (x y : ℝ) := y = (1/2) * x + 1

theorem chord_length_of_intersection :
  ∃ A B : ℝ × ℝ, ellipse A.fst A.snd ∧ ellipse B.fst B.snd ∧ line A.fst A.snd ∧ line B.fst B.snd ∧
  dist A B = Real.sqrt 35 :=
sorry

end chord_length_of_intersection_l390_390493


namespace minimum_stool_height_l390_390062

def ceiling_height : ℤ := 280
def alice_height : ℤ := 150
def reach : ℤ := alice_height + 30
def light_bulb_height : ℤ := ceiling_height - 15

theorem minimum_stool_height : 
  ∃ h : ℤ, reach + h = light_bulb_height ∧ h = 85 :=
by
  sorry

end minimum_stool_height_l390_390062


namespace constant_term_in_expansion_is_neg252_l390_390380

theorem constant_term_in_expansion_is_neg252 :
  ∃ const_term : ℤ, 
    (∀ (x : ℂ), (2 * x - (1 / (2 * x)))^10 = ∑ k in (finset.range 11), (nat.choose 10 k) * (2 * x)^(10 - k) * ((- (1 / (2 * x)))^k)) ∧ 
    (const_term = -252) := 
begin
  sorry
end

end constant_term_in_expansion_is_neg252_l390_390380


namespace range_of_eccentricity_l390_390273

variables {a b c e : ℝ}

def ellipse (x y : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)
def upper_vertex := b > 0
def distance_condition (x_0 y_0 : ℝ) := (x_0^2 + (y_0 - b)^2 ≤ (2 * b) ^ 2)
def eccentricity := e = c / a
def ellipse_condition := a = real.sqrt (b^2 + c^2)

theorem range_of_eccentricity (h1 : a > b) (h2 : upper_vertex) 
  (h3 : ∀ x_0 y_0, ellipse x_0 y_0 → distance_condition x_0 y_0)
  (h4 : ellipse_condition) :
  0 < e ∧ e ≤ real.sqrt 2 / 2 := 
by
  sorry

end range_of_eccentricity_l390_390273


namespace knight_placement_count_l390_390616

-- Definitions of the problem conditions
def knight_attacks (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 ≠ x2 ∧ y1 ≠ y2) ∧ (abs (x1 - x2) = 2 ∧ abs (y1 - y2) = 1 ∨ abs (x1 - x2) = 1 ∧ abs (y1 - y2) = 2)

def valid_knight_placement (board : list (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), i < j → (i < board.length ∧ j < board.length) → ¬ knight_attacks (board.nth i).fst (board.nth i).snd (board.nth j).fst (board.nth j).snd

-- The statement of the proof problem
theorem knight_placement_count :
  ∃ (placements : list (list (ℕ × ℕ))), (∀ p ∈ placements, valid_knight_placement p ∧ p.length = 31) ∧ placements.length = 68 :=
sorry

end knight_placement_count_l390_390616


namespace symmetry_about_y_axis_l390_390383

def f (x : ℝ) : ℝ := (2^x - 2^(-x)) / x

theorem symmetry_about_y_axis : 
  ∀ x : ℝ, x ≠ 0 → f x = f (-x) :=
by
  assume x hx,
  sorry

end symmetry_about_y_axis_l390_390383


namespace function_passes_through_fixed_point_l390_390384

noncomputable def passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : Prop :=
  ∃ y : ℝ, y = a^(1-1) + 1 ∧ y = 2

theorem function_passes_through_fixed_point (a : ℝ) (h : a > 0 ∧ a ≠ 1) : passes_through_fixed_point a h :=
by
  sorry

end function_passes_through_fixed_point_l390_390384


namespace molecular_weight_of_al2o3_correct_reaction_enthalpy_of_formation_correct_l390_390074

def atomic_weight_al : ℝ := 26.98
def atomic_weight_o : ℝ := 16.00
def molecular_weight_al2o3 : ℝ := (2 * atomic_weight_al) + (3 * atomic_weight_o)
def standard_enthalpy_of_formation_al2o3 : ℝ := -1675.7
def balanced_formation_equation : Prop := true -- Placeholder for the chemical equation

theorem molecular_weight_of_al2o3_correct :
  molecular_weight_al2o3 = 101.96 :=
by
  sorry

theorem reaction_enthalpy_of_formation_correct :
  let ΔH_reaction := 2 * standard_enthalpy_of_formation_al2o3
  in ΔH_reaction = -3351.4 :=
by
  sorry

end molecular_weight_of_al2o3_correct_reaction_enthalpy_of_formation_correct_l390_390074


namespace solution_m_value_l390_390914

theorem solution_m_value (m : ℝ) : 
  (m^2 - 5*m + 4 > 0) ∧ (m^2 - 2*m = 0) ↔ m = 0 :=
by
  sorry

end solution_m_value_l390_390914


namespace number_of_valid_sextuples_l390_390386

noncomputable def initial_sextuple : List ℤ := [-1, 2, -3, 4, -5, 6]

-- Function to calculate the sum of a list
def sum_list (l : List ℤ) : ℤ := l.sum

-- Given sextuples
noncomputable def sextuples : List (List ℤ) :=
  [ [0, 0, 0, 3, -9, 9],
    [0, 1, 1, 3, 6, -6],
    [0, 0, 0, 3, -6, 9],
    [0, 1, 1, -3, 6, -9],
    [0, 0, 2, 5, 5, 6] ]

-- Checking the sum parity
def is_odd_sum (lst : List ℤ) : Bool :=
  (sum_list lst) % 2 ≠ 0

-- Function to verify if a configuration can be obtained via sum parity rules
def valid_sextuples : List (List ℤ) :=
  sextuples.filter is_odd_sum

-- Problem-related definitions
def final_answer : ℕ := valid_sextuples.length

theorem number_of_valid_sextuples : final_answer = 1 :=
by {
  sorry
}

end number_of_valid_sextuples_l390_390386


namespace find_original_number_l390_390976

-- Definitions of the conditions
def isFiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem find_original_number (n x y : ℕ) 
  (h1 : isFiveDigitNumber n) 
  (h2 : n = 10 * x + y) 
  (h3 : n - x = 54321) : 
  n = 60356 := 
sorry

end find_original_number_l390_390976


namespace find_angle_A_max_perimeter_l390_390655

-- Definitions for the given conditions
def sides_and_angles (a b c A B C : ℝ) (triangle : Triangle ℝ) :=
  triangle.sides = (a, b, c) ∧ 
  triangle.angles = (A, B, C) ∧ 
  a = b * (sin A) / (sin B) ∧ 
  a = c * (sin A) / (sin C)

def vector_m (A B : ℝ) : ℝ × ℝ := (cos (A / 2) ^ 2, cos B)
def vector_n (a b c : ℝ) : ℝ × ℝ := (-a, 4 * c + 2 * b)
def vector_p : ℝ × ℝ := (1, 0)

-- Given vector condition
def vector_condition (A B a b c : ℝ) :=
  let m := vector_m A B
  let n := vector_n a b c
  let p := vector_p
  ∃ k : ℝ, m.1 - 1/2 * p.1 = k * n.1 ∧ m.2 - 1/2 * p.2 = k * n.2

-- Problem part (I)
theorem find_angle_A (a b c A B C : ℝ) (triangle : Triangle ℝ) (h_triangle : sides_and_angles a b c A B C triangle) 
  (h_vector_condition : vector_condition A B a b c) : 
  A = 2 * π / 3 ∨ A = 120 * (π / 180) :=
sorry

-- Problem part (II)
theorem max_perimeter (a b c A B C : ℝ) (triangle : Triangle ℝ) (h_triangle : sides_and_angles a b c A B C triangle) 
  (h_vector_condition : vector_condition A B a b c) (h_a : a = sqrt 3) : 
  a + b + c ≤ sqrt 3 + 2 :=
sorry

end find_angle_A_max_perimeter_l390_390655


namespace find_x_perpendicular_l390_390942

theorem find_x_perpendicular (x : ℝ) : 
    let a := (1 : ℝ, 2 : ℝ)
    let b := (x, -2 : ℝ)
    (a.1 * b.1 + a.2 * b.2 = 0) → x = 4 :=
by
  sorry

end find_x_perpendicular_l390_390942


namespace product_digit_sum_l390_390752

theorem product_digit_sum (n : ℕ) (h : String.replicate n '5').toNat * 5 = 500 :
  n = 72 := 
sorry

end product_digit_sum_l390_390752


namespace volume_of_one_wedge_l390_390476

theorem volume_of_one_wedge (c : ℝ) (r : ℝ) (V : ℝ) : 
  c = 18 * real.pi → 
  2 * real.pi * r = c → 
  V = (4/3) * real.pi * r^3 → 
  V / 6 = 162 * real.pi := 
by 
  intros h1 h2 h3
  sorry

end volume_of_one_wedge_l390_390476


namespace binary_remainder_l390_390438

theorem binary_remainder (n : ℕ) (h : n = 0b110110111101) : n % 4 = 1 :=
begin
  sorry
end

end binary_remainder_l390_390438


namespace selling_price_type_A_maximizing_profit_l390_390352

/-
Problem 1:
- Given conditions:
  - The total sales of type A bikes last year were 50000.
  - This year, the selling price per bike is 400 less than last year.
  - If the quantity sold remains the same, the total sales will decrease by 20%.
- Prove that the selling price per type A bike this year is 1600 yuan.
-/
theorem selling_price_type_A (sales_last_year : ℕ) (price_decrement : ℕ) (sales_decrease : ℕ) :
  sales_last_year = 50000 →
  price_decrement = 400 →
  sales_decrease = 80 / 100 →
  (∃ x : ℕ, 50000 / (x + price_decrement) = 40000 / x ∧ x = 1600) :=
by
  sorry

/-
Problem 2:
- Given conditions:
  - Purchase and selling prices for type A bikes: purchase price is 1100 per bike, selling price is 1600 per bike.
  - Purchase and selling prices for type B bikes: purchase price is 1400 per bike, selling price is 2000 per bike.
  - Total of 60 new bikes must be purchased.
  - The quantity of type B bikes should not exceed twice the quantity of type A bikes.
- Prove the optimal purchase quantities to maximize profit are 20 type A bikes and 40 type B bikes.
-/
theorem maximizing_profit (total_bikes : ℕ) (purchase_price_A : ℕ) (selling_price_A : ℕ) 
(purchase_price_B : ℕ) (selling_price_B : ℕ) :
  total_bikes = 60 →
  purchase_price_A = 1100 →
  selling_price_A = 1600 →
  purchase_price_B = 1400 →
  selling_price_B = 2000 →
  (∀ m : ℕ, 60 - m ≤ 2 * m ∧ m ≥ 20 → 
  let profit := (selling_price_A - purchase_price_A) * m + (selling_price_B - purchase_price_B) * (60 - m) in 
  profit ≤ (selling_price_A - purchase_price_A) * 20 + (selling_price_B - purchase_price_B) * 40) :=
by
  sorry

end selling_price_type_A_maximizing_profit_l390_390352


namespace three_person_subcommittees_l390_390173

theorem three_person_subcommittees (n k : ℕ) (h_n : n = 8) (h_k : k = 3) : nat.choose n k = 56 := by
  rw [h_n, h_k]
  norm_num
  sorry

end three_person_subcommittees_l390_390173


namespace equation_of_line_l_l390_390135

def circle_center (x y : ℝ) : Prop := (x^2 + y^2 - 6 * y + 5) = 0

def axis_of_symmetry (ax by : ℝ) : Prop := ax + by = -1

def perpendicular (a b : ℝ) : Prop := a + b = -2

noncomputable def line_l (a b : ℝ) : (a = 1/3) → (b = -1/3) → Prop := 
    a * x + b * y + 1 = 0

theorem equation_of_line_l : 
    ∀ a b : ℝ,
    axis_of_symmetry a b →
    perpendicular 1 1 →
    line_l (1 / 3) (-1 / 3) :=
by
    sorry

end equation_of_line_l_l390_390135


namespace volume_ratio_pyramids_l390_390854

-- Definitions of vertices
def vertex_A := (1, 0, 0, 0)
def vertex_B := (0, 1, 0, 0)
def vertex_C := (0, 0, 1, 0)
def vertex_D := (0, 0, 0, 1)
def apex := (0, 0, 0, 0)
def center_base := (1/4, 1/4, 1/4, 1/4)
def midpoint_1 := (1/2, 0, 0, 0)
def midpoint_2 := (0, 1/2, 0, 0)
def midpoint_3 := (0, 0, 1/2, 0)
def midpoint_4 := (0, 0, 0, 1/2)

-- Proposition to prove 
theorem volume_ratio_pyramids : 
  let ratio := (1/4 : ℝ)^2 * (3/4 : ℝ) in
  let m := 3 in
  let n := 64 in
  ratio = (m : ℝ) / (n : ℝ) ∧ m.gcd n = 1 ∧ (m + n) = 67 := 
by {
  -- Proof goes here
  sorry
}

end volume_ratio_pyramids_l390_390854


namespace lottery_ticket_count_l390_390218

open Nat

/-- Define the reflection of a number between 1 and 90 -/
def reflection (a : ℕ) : ℕ :=
  91 - a

/-- The set of valid numbers for the lottery -/
def valid_numbers : Finset ℕ :=
  Finset.range 91 \ {0}

/-- Define the set S with 5 elements from valid_numbers -/
def is_valid (s : Finset ℕ) : Prop :=
  s.card = 5

/-- The main theorem statement -/
theorem lottery_ticket_count :
  (Finset.card {s ∈ valid_numbers.powerset.filter is_valid | 228 ≤ s.sum id }) = 21974634 :=
sorry

/-- Helper function to calculate the sum of elements in a set -/
def set_sum (s : Finset ℕ) : ℕ :=
  s.sum id

/-- Reflected sum of a set -/
def reflected_sum (s : Finset ℕ) : ℕ :=
  s.sum reflection

/-- Proof that reflections and original sums always equal to 455 -/
lemma sum_reflection_455 {s : Finset ℕ} (hs : is_valid s) :
  set_sum s + reflected_sum s = 455 :=
sorry

end lottery_ticket_count_l390_390218


namespace sequence_sum_correct_l390_390507

noncomputable def sequence_sum : ℕ :=
  let seq := list.range (198 // 2) in
  seq.foldl (λ acc n, acc + (1980 - 20 * n) - (1970 - 20 * n)) 0

theorem sequence_sum_correct : sequence_sum = 990 := by
  sorry

end sequence_sum_correct_l390_390507


namespace husband_and_wife_age_l390_390467

theorem husband_and_wife_age (x y : ℕ) (h1 : 11 * x = 2 * (22 * y - 11 * x)) (h2 : 11 * x ≠ 0) (h3 : 11 * y ≠ 0) (h4 : 11 * (x + y) ≤ 99) : 
  x = 4 ∧ y = 3 :=
by
  sorry

end husband_and_wife_age_l390_390467


namespace garrett_granola_bars_l390_390887

theorem garrett_granola_bars :
  ∀ (oatmeal_raisin peanut total : ℕ),
  peanut = 8 →
  total = 14 →
  oatmeal_raisin + peanut = total →
  oatmeal_raisin = 6 :=
by
  intros oatmeal_raisin peanut total h_peanut h_total h_sum
  sorry

end garrett_granola_bars_l390_390887


namespace prob_sum_eq_4_l390_390374

theorem prob_sum_eq_4 :
  let x := choose (λ x : ℝ, 0 ≤ x ∧ x ≤ 3.5) in
  let interval1 := set.Ico 0.5 1.5 in
  let interval2 := set.Icc 2.5 3.5 in
  let length_interval1 := interval1.2 - interval1.1 in
  let length_interval2 := interval2.2 - interval2.1 in
  let total_interval := 3.5 in
  length_interval1 + length_interval2 / total_interval = 4 / 7 :=
by
  sorry

end prob_sum_eq_4_l390_390374


namespace eccentricity_range_l390_390285

noncomputable def ellipse_eccentricity (a b : ℝ) (h_ab : a > b) : ℝ := (Real.sqrt (a ^ 2 - b ^ 2)) / a

theorem eccentricity_range (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ) :
  (∃ x y : ℝ, (x, y) = P ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  (Real.sqrt (x^2 + (y - b)^2) ≤ 2 * b)) → 
  0 < ellipse_eccentricity a b h_ab ∧ ellipse_eccentricity a b h_ab ≤ Real.sqrt 2 / 2 :=
begin
  sorry
end

end eccentricity_range_l390_390285


namespace find_power_function_equation_l390_390964

-- Definition of the power function passing through the given point.
def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Condition: The function passes through P(3, √3).
def passes_through_point : Prop :=
  power_function (1/2) 3 = Real.sqrt 3

-- The main statement: If the power function passes through P(3, √3),
-- then the equation of the function is y = √x.
theorem find_power_function_equation (α : ℝ) :
  (power_function α 3 = Real.sqrt 3) → (∀ x, power_function α x = Real.sqrt x) :=
by
  sorry

end find_power_function_equation_l390_390964


namespace subset_proper_l390_390159

def M : Set ℝ := {x | x^2 - x ≤ 0}

def N : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem subset_proper : N ⊂ M := by
  sorry

end subset_proper_l390_390159


namespace algebraic_expression_value_l390_390955

theorem algebraic_expression_value (a b : ℝ) (h : a = b + 1) : a^2 - 2 * a * b + b^2 + 2 = 3 :=
by
  sorry

end algebraic_expression_value_l390_390955


namespace distance_AB_is_sqrt_6_l390_390577

/-- Define points A and B in 3D space -/
def A : ℝ × ℝ × ℝ := (2, 3, 5)
def B : ℝ × ℝ × ℝ := (3, 1, 4)

/-- Define the distance formula for two points in 3D space -/
def distance_3d (P Q : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := P;
  let (x2, y2, z2) := Q;
  ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2).sqrt

/-- Prove that the distance between points A and B is sqrt(6) -/
theorem distance_AB_is_sqrt_6 : distance_3d A B = Real.sqrt 6 :=
  by
  sorry

end distance_AB_is_sqrt_6_l390_390577


namespace maximize_profit_l390_390028

noncomputable def c (x: ℝ) : ℝ := 1200 + (2 / 75) * x^3

def k : ℝ := 250000 -- derived from the condition when p = 50 and x = 100

def p (x: ℝ) : ℝ := Real.sqrt (k / x)

def L (x: ℝ) : ℝ := - (2 / 75) * x^3 + 500 * Real.sqrt x - 1200

theorem maximize_profit : 
  ∃ x : ℝ, x > 0 ∧ 
  (∀ y : ℝ, y > 0 → L(x) ≥ L(y)) ∧ 
  abs (L(x) - 883) < 1 :=
by
  use 25
  sorry

end maximize_profit_l390_390028


namespace motorcyclist_speed_before_delay_l390_390035

/-- Given conditions and question:
1. The motorcyclist was delayed by 0.4 hours.
2. After the delay, the motorcyclist increased his speed by 10 km/h.
3. The motorcyclist made up for the lost time over a stretch of 80 km.
-/
theorem motorcyclist_speed_before_delay :
  ∃ x : ℝ, (80 / x - 0.4 = 80 / (x + 10)) ∧ x = 40 :=
sorry

end motorcyclist_speed_before_delay_l390_390035


namespace parabola_locus_property_l390_390405

-- Given a parabola y^2 = 2px where p is a prime number greater than 2,
-- and a line l passing through the focus F intersecting the parabola at points P and Q.
-- M is the midpoint of segment PQ and N is the point where the perpendicular bisector of PQ intersects the x-axis.
-- Prove the following properties:
-- 1. The locus of the midpoint R of segment MN is y^2 = (1/4) * p * (x - p).
-- 2. There are infinitely many integer points on this locus.
-- 3. The distance from any such integer point to the origin is not an integer.

theorem parabola_locus_property (p : ℕ) (hp : prime p ∧ p > 2):
  (∀ R : ℝ × ℝ, (∃ M N : ℝ × ℝ, (M ≠ N ∧ 
  (M.1 - (P.1 + Q.1) / 2 = 0) ∧ 
  (M.2 - (P.2 + Q.2) / 2 = 0) ∧ 
  (N.1 = some_point_on_perpendicular_bisector PQ) ∧
  (N.2 = 0) ∧
  (R.1 = (M.1 + N.1) / 2) ∧ 
  (R.2 = (M.2 + N.2) / 2))) → 
  (R.2)^2 = (1/4) * p * (R.1 - p)) ∧ 
  (∀ t ∈ ℤ, ∃ R : ℤ × ℤ, (R.2^2 = (1/4) * p * (R.1 - p)) ∧ 
  ∃ d : ℕ, d = R.1^2 + R.2^2 ∧ d ∉ ℤ)) :=
sorry

end parabola_locus_property_l390_390405


namespace distance_to_big_rock_l390_390448

variables (D : ℝ) (stillWaterSpeed : ℝ) (currentSpeed : ℝ) (totalTime : ℝ)

-- Define the conditions as constraints
def conditions := 
  stillWaterSpeed = 6 ∧
  currentSpeed = 1 ∧
  totalTime = 1 ∧
  (D / (stillWaterSpeed - currentSpeed) + D / (stillWaterSpeed + currentSpeed) = totalTime)

-- The theorem to prove the distance to Big Rock
theorem distance_to_big_rock (h : conditions D 6 1 1) : D = 35 / 12 :=
sorry

end distance_to_big_rock_l390_390448


namespace triangle_area_l390_390972

theorem triangle_area (A B C : ℝ) (a b c : ℝ) 
  (hA : A = 60) (ha : a = sqrt 3) (hb_c_sum : b + c = 3) :
  let area := (1 / 2) * b * c * real.sin (A * real.pi / 180)
  in area = sqrt 3 / 2 := by
  sorry

end triangle_area_l390_390972


namespace points_on_line_l390_390526

-- Define the points
def P1 : (ℝ × ℝ) := (8, 16)
def P2 : (ℝ × ℝ) := (2, 4)

-- Define the line equation as a predicate
def on_line (m b : ℝ) (p : ℝ × ℝ) : Prop := p.2 = m * p.1 + b

-- Define the given points to be checked
def P3 : (ℝ × ℝ) := (5, 10)
def P4 : (ℝ × ℝ) := (7, 14)
def P5 : (ℝ × ℝ) := (4, 7)
def P6 : (ℝ × ℝ) := (10, 20)
def P7 : (ℝ × ℝ) := (3, 6)

theorem points_on_line :
  let m := 2
  let b := 0
  on_line m b P3 ∧
  on_line m b P4 ∧
  ¬ on_line m b P5 ∧
  on_line m b P6 ∧
  on_line m b P7 :=
by
  sorry

end points_on_line_l390_390526


namespace abs_f_minus_g_gt_2_l390_390240

noncomputable def f (x : ℝ) : ℝ :=
  ∑ i in (Finset.range 1009), 1 / (x - 2 * i)

noncomputable def g (x : ℝ) : ℝ :=
  ∑ i in (Finset.range 1009), 1 / (x - (2 * i + 1))

theorem abs_f_minus_g_gt_2 (x : ℝ) (hx1 : 0 < x) (hx2 : x < 2018) (hx3 : ∀ n : ℤ, x ≠ n) : 
  |f x - g x| > 2 :=
begin 
  sorry
end

end abs_f_minus_g_gt_2_l390_390240


namespace find_function_l390_390343

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ x y : ℝ, f(x * y) = f(x) * f(y)
axiom cond2 : ∀ x : ℝ, f(x + Real.sqrt 2) = f(x) + f(Real.sqrt 2)
axiom cond3 : ∃ x : ℝ, f(x) ≠ 0

theorem find_function : ∀ x : ℝ, f(x) = x :=
by
  sorry

end find_function_l390_390343


namespace required_moles_Cl2_l390_390949

-- Define the reaction condition and the balanced chemical equation as predicates.
def reaction_condition (n_CH4 n_Cl2 n_CCl4 n_HCl : ℕ) : Prop :=
  n_CH4 = 1 ∧ n_Cl2 = 4 ∧ n_CCl4 = 1 ∧ n_HCl = 4

-- Define the main theorem that needs to be proved.
theorem required_moles_Cl2 (n_CH4 n_Cl2 : ℕ) :
  reaction_condition 1 4 1 4 →
  n_CH4 = 3 →
  n_Cl2 = 12 :=
by {
  intros,
  sorry
}

end required_moles_Cl2_l390_390949


namespace eccentricity_range_l390_390296

-- Ellipse definition
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Upper vertex B of the ellipse
def B (b : ℝ) : ℝ × ℝ := (0, b)

-- Distance PB condition
def distance_le_2b (a b x0 y0 : ℝ) (h : ellipse a b (and.intro (sorry) (sorry)) x0 y0) : Prop :=
  (x0 - 0)^2 + (y0 - b)^2 ≤ (2 * b)^2

-- Range of eccentricity
def eccentricity (a b c : ℝ) : ℝ :=
  c / a

-- Main theorem
theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  (forall (x0 y0 : ℝ), ellipse a b h x0 y0 → distance_le_2b a b x0 y0 h) →
  eccentricity a b (sqrt (a^2 - b^2)) ∈ set.Icc 0 (sqrt 2 / 2) :=
by
  sorry

end eccentricity_range_l390_390296


namespace diana_prime_larger_apollo_prob_l390_390863

noncomputable def probability_Diana_prime_larger_Apollo : ℚ :=
  let primes := {2, 3, 5, 7}
  let favorable_outcomes := 
    set.sum (set.map (λ apollo, (set.filter (λ diana, diana ∈ primes ∧ diana > apollo) (finset.range 1 9))).card) (finset.range 1 7)
  in favorable_outcomes.to_rat / (8 * 6)

theorem diana_prime_larger_apollo_prob : probability_Diana_prime_larger_Apollo = 1 / 4 :=
sorry

end diana_prime_larger_apollo_prob_l390_390863


namespace half_day_division_l390_390030

theorem half_day_division : 
  ∃ (n m : ℕ), n * m = 43200 ∧ (∃! (k : ℕ), k = 60) := sorry

end half_day_division_l390_390030


namespace absolute_difference_l390_390241

theorem absolute_difference (x y : ℝ) (hx : ⌊x⌋ + frac y = 3.7) (hy : frac x + ⌊y⌋ = 8.2) : abs (x - y) = 5.5 :=
by
  sorry

end absolute_difference_l390_390241


namespace path_length_for_right_triangle_l390_390821

def right_triangle_path_length (a b : ℝ) (V : ℝ) (D_sum : ℝ) : ℝ :=
  if h : a^2 * b = 300 ∧ (2 * (a * b / (a + b + sqrt (a^2 + b^2)) + sqrt (a^2 + b^2) / 2) = 17) then
    let AB := sqrt (a^2 + b^2) in
    a + b + AB
  else
    0

theorem path_length_for_right_triangle :
  right_triangle_path_length 5 12 (100 * Real.pi) 17 = 30 :=
by
  sorry

end path_length_for_right_triangle_l390_390821


namespace find_side_b_l390_390991

open Real

variables {A B C : ℝ}
variables {a b c : ℝ}
variable (△ABC : Triangle A B C)

theorem find_side_b (h_a : a = sqrt 5) (h_c : c = 2) (h_cosA : cos A = 2 / 3) :
  b = 3 :=
sorry

end find_side_b_l390_390991


namespace sum_of_abs_roots_l390_390882

-- Define the polynomial
def poly := polynomial(C: ℂ)([a=1, b=-6, c=9, d=24, e=-36])

-- The main theorem to prove
theorem sum_of_abs_roots {α : Type*} [field α] (hx : ℝ) (hijk : polynomial(a: 1, b:-6, c:9, d:24, e:-36) (has_roots: list(√(3), -√(3), i√(3), -i√(3)))) :
  (∑ root in polynomial.roots(poly.to_finsupp), abs root) = 4 * √3 :=
sorry

end sum_of_abs_roots_l390_390882


namespace measure_of_angle_WUV_l390_390984

-- Define points and angles corresponding to the diagram conditions
def point (ℝ) : Type := ℝ
def angle (ℝ) : Type := ℝ

def T : point ℝ := 0
def U : point ℝ := 1.5
def V : point ℝ := 2.5
def W : point ℝ := 1

def angle_TWV : angle ℝ := 55
def angle_WVT : angle ℝ := 43
def angle_WUT : angle ℝ := 72

-- Define the given conditions
axiom TUV_is_straight : T ≠ U ∧ U ≠ V ∧ T ≠ V  
axiom angle_sum_in_triangle (a b c : angle ℝ) : a + b + c = 180

-- Statement to prove
theorem measure_of_angle_WUV : 
  let angle_WTV := 180 - angle_TWV - angle_WVT in
  let angle_WUV := angle_WTV - angle_WUT in
  angle_WUV = 10 :=
by
  sorry

end measure_of_angle_WUV_l390_390984


namespace determine_m_l390_390109

theorem determine_m (x m : ℝ) (h₁ : 2 * x + m = 6) (h₂ : x = 2) : m = 2 := by
  sorry

end determine_m_l390_390109


namespace triangular_prism_no_body_diagonal_l390_390446

-- Define the solids as prisms based on their bases.
inductive BaseShape
| triangle : BaseShape
| quadrilateral : BaseShape
| pentagon : BaseShape
| hexagon : BaseShape

def has_diagonal : BaseShape → Prop
| BaseShape.triangle := false
| _ := true

structure Prism where
  base : BaseShape

def body_diagonal (p : Prism) : Prop :=
  has_diagonal p.base

-- Given that a triangular prism has a triangle as a base which does not have diagonals,
-- prove that a triangular prism does not have a body diagonal
theorem triangular_prism_no_body_diagonal 
  (p : Prism) 
  (h_triangle_base : p.base = BaseShape.triangle) : 
  ¬ body_diagonal p :=
by 
  unfold body_diagonal
  rw h_triangle_base
  simp
  sorry

end triangular_prism_no_body_diagonal_l390_390446


namespace points_exist_with_conditions_l390_390925

noncomputable def find_points (k : ℝ) : Prop :=
  ∃ (s t m n : ℕ), 
    s > m ∧
    t > n ∧
    (∀ x y : ℝ, x^2 + y^2 = 4 → (sqrt ((s - x)^2 + (t - y)^2) / sqrt ((m - x)^2 + (n - y)^2)) = k) ∧
    s = 2 ∧
    t = 2 ∧
    m = 1 ∧
    n = 1

theorem points_exist_with_conditions (k : ℝ) (h : k^2 = 2) : find_points k :=
sorry

end points_exist_with_conditions_l390_390925


namespace range_of_eccentricity_l390_390249

noncomputable def upperVertex (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : ℝ × ℝ := (0, b)

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem range_of_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h: a > b) :
  (∀ (x y : ℝ), ellipse a b x y → distance x y 0 b ≤ 2 * b) →
  ∃ (e : ℝ), e ∈ Set.Ioc 0 (Real.sqrt 2 / 2) ∧ a^2 = b^2 / (1 - e^2) := sorry

end range_of_eccentricity_l390_390249


namespace value_of_expression_l390_390621

variable (m : ℝ)

theorem value_of_expression (h : 2 * m^2 + 3 * m - 1 = 0) : 
  4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end value_of_expression_l390_390621


namespace question_1_question_2_l390_390822

noncomputable def p1_correctness : ℝ := 2 / 3

def problem_conditions (P1 : ℝ) : Prop :=
  (1 - P1) ^ 2 = 1 / 9

def answer_question_1 : ℝ := p1_correctness

theorem question_1 :
  ∃ P1 : ℝ, problem_conditions P1 ∧ P1 = answer_question_1 :=
by
  sorry

noncomputable def advancing_probability : ℝ := 496 / 729

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def probability_of_advancement (P1 pi : ℝ) : ℝ :=
  P1^4 + binomial 4 3 * P1^3 * pi * P1 + binomial 5 3 * P1^3 * pi^2 * P1

theorem question_2 :
  probability_of_advancement p1_correctness (1 / 3) = advancing_probability :=
by
  sorry

end question_1_question_2_l390_390822


namespace maximize_distance_l390_390891

def front_tire_lifespan : ℕ := 20000
def rear_tire_lifespan : ℕ := 30000
def max_distance : ℕ := 24000

theorem maximize_distance : max_distance = 24000 := sorry

end maximize_distance_l390_390891


namespace solve_B_share_l390_390484

def ratio_shares (A B C : ℚ) : Prop :=
  A = 1/2 ∧ B = 1/3 ∧ C = 1/4

def initial_capitals (total_capital : ℚ) (A_s B_s C_s : ℚ) : Prop :=
  A_s = 1/2 * total_capital ∧ B_s = 1/3 * total_capital ∧ C_s = 1/4 * total_capital

def total_capital_contribution (A_contrib B_contrib C_contrib : ℚ) : Prop :=
  A_contrib = 42 ∧ B_contrib = 48 ∧ C_contrib = 36

def B_share (B_contrib total_contrib profit : ℚ) : ℚ := 
  (B_contrib / total_contrib) * profit

theorem solve_B_share : 
  ∀ (A_s B_s C_s total_capital profit A_contrib B_contrib C_contrib total_contrib : ℚ),
  ratio_shares (1/2) (1/3) (1/4) →
  initial_capitals total_capital A_s B_s C_s →
  total_capital_contribution A_contrib B_contrib C_contrib →
  total_contrib = A_contrib + B_contrib + C_contrib →
  profit = 378 →
  B_s = (1/3) * total_capital →
  B_contrib = 48 →
  B_share B_contrib total_contrib profit = 108 := by 
    sorry

end solve_B_share_l390_390484


namespace solve_for_x_l390_390437

theorem solve_for_x : ∃ x : ℚ, x + 5/6 = 7/18 + 1/2 ∧ x = -7/18 := by
  sorry

end solve_for_x_l390_390437


namespace josanna_minimum_test_score_l390_390662

theorem josanna_minimum_test_score 
  (scores : List ℕ) (target_increase : ℕ) (new_score : ℕ)
  (h_scores : scores = [92, 78, 84, 76, 88]) 
  (h_target_increase : target_increase = 5):
  (List.sum scores + new_score) / (List.length scores + 1) ≥ (List.sum scores / List.length scores + target_increase) →
  new_score = 114 :=
by
  sorry

end josanna_minimum_test_score_l390_390662


namespace interest_less_than_sum_lent_l390_390837

def principal : ℝ := 9200
def rate : ℝ := 12
def time : ℕ := 3
def simple_interest (P R : ℝ) (T : ℕ) : ℝ := (P * R * T) / 100

theorem interest_less_than_sum_lent :
    principal - simple_interest principal rate time = 5888 := 
by
    sorry

end interest_less_than_sum_lent_l390_390837


namespace volume_of_revolution_l390_390652

variable (α β l : ℝ)

theorem volume_of_revolution (α β l : ℝ) :
  (let V := (π * l^3 * (Real.cos ((α - β) / 2))^2 * (Real.cos ((α + β) / 2))^2) / 
             (3 * (Real.sin α)^2)
   in V) = (π * l^3 * (Real.cos ((α - β) / 2))^2 * (Real.cos ((α + β) / 2))^2) / 
             (3 * (Real.sin α)^2) :=
by {
  let V := (π * l^3 * (Real.cos ((α - β) / 2))^2 * (Real.cos ((α + β) / 2))^2) / 
           (3 * (Real.sin α)^2),
  exact eq.refl V
}

end volume_of_revolution_l390_390652


namespace no_15_students_with_unique_colors_l390_390761

-- Conditions as definitions
def num_students : Nat := 30
def num_colors : Nat := 15

-- The main statement
theorem no_15_students_with_unique_colors
  (students : Fin num_students → (Fin num_colors × Fin num_colors)) :
  ¬ ∃ (subset : Fin 15 → Fin num_students),
    ∀ i j (hi : i ≠ j), (students (subset i)).1 ≠ (students (subset j)).1 ∧
                         (students (subset i)).2 ≠ (students (subset j)).2 :=
by sorry

end no_15_students_with_unique_colors_l390_390761


namespace circle_radius_c_l390_390110

theorem circle_radius_c (c : ℝ) : (∃ (x y : ℝ), ((x^2 + 6 * x + y^2 - 4 * y + c = 0)) ∧ (sqrt (13 - c) = 4)) ↔ (c = -3) :=
by {
  sorry
}

end circle_radius_c_l390_390110


namespace range_of_eccentricity_l390_390306

-- Definition of the ellipse and its properties
namespace EllipseProof

variables {a b : ℝ} (h : a > b ∧ b > 0)

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def upper_vertex : ℝ × ℝ := (0, b)

def is_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def condition_on_point (P : ℝ × ℝ) : Prop := dist P upper_vertex ≤ 2 * b

-- The eccentricity of the ellipse
def eccentricity : ℝ := real.sqrt (1 - (b^2 / a^2))

-- The required proof statement
theorem range_of_eccentricity :
  (∀ P, is_on_ellipse P → condition_on_point P) →
  eccentricity ≤ real.sqrt 2 / 2 := sorry

end EllipseProof

end range_of_eccentricity_l390_390306


namespace present_age_of_son_l390_390465

variable (S M : ℝ)

-- Conditions
def condition1 : Prop := M = S + 35
def condition2 : Prop := M + 5 = 3 * (S + 5)

-- Proof Problem
theorem present_age_of_son
  (h1 : condition1 S M)
  (h2 : condition2 S M) :
  S = 12.5 :=
sorry

end present_age_of_son_l390_390465


namespace subtraction_problem_solution_l390_390784

theorem subtraction_problem_solution :
  ∃ x : ℝ, (8 - x) / (9 - x) = 4 / 5 :=
by
  use 4
  sorry

end subtraction_problem_solution_l390_390784


namespace range_of_k_l390_390908

theorem range_of_k (k : ℝ) : (∀ x : ℝ, x > 0 → (k+4) * x < 0) → k < -4 :=
by
  sorry

end range_of_k_l390_390908


namespace range_of_eccentricity_l390_390304

-- Definition of the ellipse and its properties
namespace EllipseProof

variables {a b : ℝ} (h : a > b ∧ b > 0)

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def upper_vertex : ℝ × ℝ := (0, b)

def is_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def condition_on_point (P : ℝ × ℝ) : Prop := dist P upper_vertex ≤ 2 * b

-- The eccentricity of the ellipse
def eccentricity : ℝ := real.sqrt (1 - (b^2 / a^2))

-- The required proof statement
theorem range_of_eccentricity :
  (∀ P, is_on_ellipse P → condition_on_point P) →
  eccentricity ≤ real.sqrt 2 / 2 := sorry

end EllipseProof

end range_of_eccentricity_l390_390304


namespace range_of_eccentricity_of_ellipse_l390_390254

-- Definitions for the problem.
def is_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1) 

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

def on_upper_vertex (b : ℝ) : (ℝ × ℝ) := (0, b)

-- The statement to prove the range of eccentricity.
theorem range_of_eccentricity_of_ellipse (a b : ℝ) (x y : ℝ) (e : ℝ) :
  is_on_ellipse a b x y →
  (distance x y 0 b ≤ 2 * b) →
  ∃ e, (e = (1 - (b^2 / a^2))^(1/2) ∧ (0 < e) ∧ (e ≤ (sqrt 2)/2)) :=
by
  sorry

end range_of_eccentricity_of_ellipse_l390_390254


namespace map_distance_equiv_proof_l390_390698

-- Define the conditions given in the problem
def actual_distance_between_mountains : ℝ := 136
def map_distance_Ram : ℝ := 28
def actual_distance_Ram : ℝ := 12.205128205128204

-- Define the scale based on Ram's actual distance and map distance
def scale : ℝ := actual_distance_Ram / map_distance_Ram

-- Define the map distance between the mountains by using the scale
def map_distance_mountains : ℝ := actual_distance_between_mountains / scale

-- The theorem we need to prove
theorem map_distance_equiv_proof :
  map_distance_mountains = 312 := by
    sorry

end map_distance_equiv_proof_l390_390698


namespace parallel_vectors_perpendicular_vectors_l390_390582

section vector_algebra

variables {α : Type*} [normed_field α] [normed_space α α]
variables (a b : α) (k : ℝ)
variables (h1 : ∥a∥ = 2) (h2 : ∥b∥ = 3)
variables (θ : ℝ) (hθ : θ = real.pi / 3) -- 60 degrees in radians
variables (c : α) (d : α)
variables (h3 : c = 5 • a + 3 • b) (h4 : d = 3 • a + k • b)

/-- Case 1: Prove that c is parallel to d if and only if k = 9 / 5 -/
theorem parallel_vectors : (∃ t : ℝ, 5 • a + 3 • b = t • (3 • a + k • b)) ↔ k = 9 / 5 :=
sorry

/-- Case 2: Prove that c is perpendicular to d if and only if k = -29 / 14 -/
theorem perpendicular_vectors : (inner ((5 : ℝ) • a + 3 • b) (3 • a + k • b) = 0) ↔ k = -29 / 14 :=
sorry

end vector_algebra

end parallel_vectors_perpendicular_vectors_l390_390582


namespace kerosene_price_and_egg_equivalence_l390_390212

-- Definitions based on the conditions
def cost_of_rice_per_pound : ℝ := 0.33
def dozen_eggs := 12
def cost_of_egg := cost_of_rice_per_pound / dozen_eggs
def cost_of_half_liter_kerosene := 12 * cost_of_egg
def cost_of_liter_kerosene := 2 * cost_of_half_liter_kerosene

-- Proof problem statement
theorem kerosene_price_and_egg_equivalence :
  (∃ x : ℝ, x = 12 * cost_of_egg ∧ 2 * x = 0.66) :=
by
  use cost_of_half_liter_kerosene
  split
  · rfl
  · have h : 2 * cost_of_half_liter_kerosene = 2 * (12 * cost_of_egg)
    simp [cost_of_half_liter_kerosene, cost_of_egg]
    norm_num

end kerosene_price_and_egg_equivalence_l390_390212


namespace elliptic_eccentricity_range_l390_390585

noncomputable def eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) : Set ℝ :=
  {e : ℝ | e = real.sqrt (1 - (b/a)^2) ∧ 1/3 ≤ e ∧ e < 1 }

theorem elliptic_eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  let c := real.sqrt (a^2 - b^2) in
  let F := (-c, 0) in
  (∃ P Q : (ℝ × ℝ), 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧ 
    (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) ∧ 
    (P.1 + c, P.2) = (2 * (Q.1 + c), 2 * Q.2)) →
  eccentricity_range a b h =  {e : ℝ | e = real.sqrt (1 - (b/a)^2) ∧ 1/3 ≤ e ∧ e < 1 } :=
by
  sorry

end elliptic_eccentricity_range_l390_390585


namespace balls_in_jar_l390_390758

/-
There are 7 balls in a jar, numbered from 1 to 7, inclusive.
Richard takes 'a' balls from the jar at once, where 'a' is an integer between 1 and 6, inclusive.
Janelle takes 'b' of the remaining balls from the jar at once, where 'b' is an integer between 1 and the number of balls left, inclusive.
Tai takes all of the remaining balls from the jar at once, if any are left.
Find the remainder when the number of possible ways for this to occur is divided by 1000.
-/

theorem balls_in_jar (totalBalls : ℕ) (a b : ℕ) (remainingBalls tai : ℕ) :
  totalBalls = 7 ∧ 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ remainingBalls ∧ remainingBalls = totalBalls - a ∧ 
  (remainingBalls = 0 → tai = 0 ∨ tai = remainingBalls) →
  let number_of_ways := 
    (finset.range 7).sum (λ k, nat.choose 7 k * (2 ^ (7 - k) - 2)) in
  number_of_ways % 1000 = 932 := by
  sorry

end balls_in_jar_l390_390758


namespace range_of_f_l390_390146

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (Real.sin x)

theorem range_of_f : 
  ∀ x, x ∈ set.Icc (0 : ℝ) (5 * Real.pi / 6) → f x ∈ set.Icc (1/2 : ℝ) 1 :=
by
  intros x hx
  sorry

end range_of_f_l390_390146


namespace percentage_increase_chef_vs_dishwasher_l390_390834

variables 
  (manager_wage chef_wage dishwasher_wage : ℝ)
  (h_manager_wage : manager_wage = 8.50)
  (h_chef_wage : chef_wage = manager_wage - 3.315)
  (h_dishwasher_wage : dishwasher_wage = manager_wage / 2)

theorem percentage_increase_chef_vs_dishwasher :
  ((chef_wage - dishwasher_wage) / dishwasher_wage) * 100 = 22 :=
by
  sorry

end percentage_increase_chef_vs_dishwasher_l390_390834


namespace correct_pronoun_possessive_l390_390364

theorem correct_pronoun_possessive : 
  (∃ (pronoun : String), 
    pronoun = "whose" ∧ 
    pronoun = "whose" ∨ pronoun = "who" ∨ pronoun = "that" ∨ pronoun = "which") := 
by
  -- the proof would go here
  sorry

end correct_pronoun_possessive_l390_390364


namespace find_plane_equation_l390_390540

open Real
noncomputable def equation_of_plane(
  p1 p2 : ℝ × ℝ × ℝ,
  a b c d : ℝ
) : Prop := 
  (∀ (x y z : ℝ), (a * x + b * y + c * z + d) = 0) 
  ⋀ ((x y z : ℝ), a * x + b * y + c * z ≠ 0 -> 
  (∃ (p' : ℝ × ℝ × ℝ), p' ≠ (p1, p2) ⋀ 
  (p' = p1 ⋀ (a * p1.1 + b * p1.2 + c * p1.3 + d = 0)) ⋀ 
  (p' = p2 ⋀ (a * p2.1 + b * p2.2 + c * p2.3 + d = 0)) ⋀ 
  (a, b, c) ≠ (0, 0, 0)))

theorem find_plane_equation : 
    equation_of_plane (2, -3, 4) (-1, 3, -2) 2 5 (-4) 27 :=
by 
  sorry

end find_plane_equation_l390_390540


namespace modulus_of_z_l390_390342

def z : ℂ := (2 - complex.i)^2

theorem modulus_of_z : complex.abs z = 5 := by
  sorry

end modulus_of_z_l390_390342


namespace range_of_eccentricity_l390_390278

variables {a b c e : ℝ}

def ellipse (x y : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)
def upper_vertex := b > 0
def distance_condition (x_0 y_0 : ℝ) := (x_0^2 + (y_0 - b)^2 ≤ (2 * b) ^ 2)
def eccentricity := e = c / a
def ellipse_condition := a = real.sqrt (b^2 + c^2)

theorem range_of_eccentricity (h1 : a > b) (h2 : upper_vertex) 
  (h3 : ∀ x_0 y_0, ellipse x_0 y_0 → distance_condition x_0 y_0)
  (h4 : ellipse_condition) :
  0 < e ∧ e ≤ real.sqrt 2 / 2 := 
by
  sorry

end range_of_eccentricity_l390_390278


namespace sum_b_odd_formula_l390_390570

variable {a : ℕ → ℕ}

-- Conditions for the geometric sequence {a_n}
axiom cond1 : a 2 - a 0 = 3
axiom cond2 : a 0 + a 1 = 3

-- Define the sequence b_n
def b (n : ℕ) : ℕ := a n ^ 2 + 1

-- Define the sum of b_1, b_3, b_5, ..., b_{2n-1}
def sum_b_odd (n : ℕ) : ℕ :=
  let terms := List.range n
  terms.map (fun k => b (2 * k + 1)).sum

-- The goal to prove
theorem sum_b_odd_formula (n : ℕ) :
  sum_b_odd n = (4^(2*n) - 1) / 15 + n :=
sorry

end sum_b_odd_formula_l390_390570


namespace _l390_390683

open Real

statement theorem proof_problem (a x : ℝ) (h1 : 0 < a ∧ a ≠ 1) (h2 : -4 < x ∧ x < 4) (h3 : 0 < a ∧ a < (8:ℝ)^(1/4)):
  x = 4 - a^4 :=
begin
  sorry
end

end _l390_390683


namespace smallest_prime_8_less_than_square_l390_390424

theorem smallest_prime_8_less_than_square :
  ∃ p : ℕ, (∃ n : ℤ, p = n^2 - 8) ∧ Nat.Prime p ∧ p > 0 ∧ (∀ q : ℕ, (∃ m : ℤ, q = m^2 - 8) ∧ Nat.Prime q → q ≥ p) :=
sorry

end smallest_prime_8_less_than_square_l390_390424


namespace value_of_b_l390_390117

theorem value_of_b (a b : ℝ) (h : b + 5 * complex.I = 9 - a + a * complex.I) : b = 4 :=
sorry

end value_of_b_l390_390117


namespace intersection_complement_l390_390605

universe u

def U : set ℕ := {1, 2, 3, 4, 5, 6}
def A : set ℕ := {1, 3, 5}
def B : set ℕ := {1, 4}

open set

theorem intersection_complement (x : ℕ) (hx : x ∈ A ∩ (U \ B)) : x ∈ {3, 5} :=
by
  sorry

end intersection_complement_l390_390605


namespace area_change_l390_390794

variable (L B : ℝ)

def initial_area (L B : ℝ) : ℝ := L * B

def new_length (L : ℝ) : ℝ := 1.20 * L

def new_breadth (B : ℝ) : ℝ := 0.95 * B

def new_area (L B : ℝ) : ℝ := (new_length L) * (new_breadth B)

theorem area_change (L B : ℝ) : new_area L B = 1.14 * (initial_area L B) := by
  -- Proof goes here
  sorry

end area_change_l390_390794


namespace monotonic_increase_range_of_alpha_l390_390597

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * Real.sin (ω * x) - (Real.sqrt 3 / 2) * Real.cos (ω * x)

theorem monotonic_increase_range_of_alpha
  (ω : ℝ) (hω : ω > 0)
  (zeros_form_ap : ∀ k : ℤ, ∃ x₀ : ℝ, f ω x₀ = 0 ∧ ∀ n : ℤ, f ω (x₀ + n * (π / 2)) = 0) :
  ∃ α : ℝ, 0 < α ∧ α < 5 * π / 12 ∧ ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ α → f ω x ≤ f ω y :=
sorry

end monotonic_increase_range_of_alpha_l390_390597


namespace probability_ge_one_l390_390966

noncomputable def normal_distribution (μ σ : ℝ) (x : ℝ) : ℝ := sorry -- Assume this defines the normal distribution PDF

/--
  Given a normal distribution random variable ξ with mean -1 and variance σ^2
  and the probability P(-3 ≤ ξ ≤ -1) = 0.4,
  prove that P(ξ ≥ 1) = 0.1
-/
theorem probability_ge_one {ξ : ℝ → ℝ} (σ : ℝ) (h₁ : ξ ~ ℙ[N(-1, σ^2)])
  (h₂ : P(-3 ≤ ξ ≤ -1) = 0.4) :
  P(ξ ≥ 1) = 0.1 :=
sorry

end probability_ge_one_l390_390966


namespace count_points_with_integer_coords_l390_390182

noncomputable def k : ℕ := 2013
noncomputable def hyperbola (x : ℝ) : ℝ := k / x
noncomputable def tangent_slope (x : ℝ) : ℝ := -k / (x ^ 2)
noncomputable def tangent_line (x₀ y₀ : ℝ) (x : ℝ) : ℝ := tangent_slope x₀ * (x - x₀) + y₀

theorem count_points_with_integer_coords :
  let num_divisors (n : ℕ) := (2 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  in (num_divisors * 2) = 48 :=
by
  -- Proof omitted
  sorry

end count_points_with_integer_coords_l390_390182


namespace at_least_250_grid_points_in_circle_l390_390700

theorem at_least_250_grid_points_in_circle :
  let grid_points := { p : ℤ × ℤ | p.1 ^ 2 + p.2 ^ 2 < 100 } in
  grid_points.card ≥ 250 :=
by
  sorry

end at_least_250_grid_points_in_circle_l390_390700


namespace algebraic_expression_value_l390_390120

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 3 = 0) :
  (x - 1)^2 - x * (x - 3) + (x + 1) * (x - 1) = 3 :=
by
  sorry

end algebraic_expression_value_l390_390120


namespace elizabeth_haircut_l390_390092

theorem elizabeth_haircut (t s f : ℝ) (ht : t = 0.88) (hs : s = 0.5) : f = t - s := by
  sorry

end elizabeth_haircut_l390_390092


namespace binary_remainder_l390_390439

theorem binary_remainder (n : ℕ) (h : n = 0b110110111101) : n % 4 = 1 :=
begin
  sorry
end

end binary_remainder_l390_390439


namespace correct_expression_l390_390785

theorem correct_expression (a b c : ℝ) : a - b + c = a - (b - c) :=
by
  sorry

end correct_expression_l390_390785


namespace pqrs_area_ratio_sum_l390_390363

theorem pqrs_area_ratio_sum (r : ℝ) :
  let a := 4,
      b := 1,
      c := 1 in
  let angle_PQR := 40,
      angle_QPR := 50,
      angle_PSR := 90,
      area_PQRS := 4 * r^2 * sin(angle_PQR * π / 180) * sin(angle_QPR * π / 180),
      area_circle := π * r^2 in
  a + b + c = 6 := by
  sorry

end pqrs_area_ratio_sum_l390_390363


namespace Carl_fraction_used_to_pay_bills_l390_390076

-- Define constants
def weekly_savings : ℕ := 25
def total_weeks_saved : ℕ := 6
def dad_contribution : ℕ := 70
def coat_cost : ℕ := 170
def amount_used : ℕ := 50

-- Define function to calculate total savings
def total_savings (weekly_savings : ℕ) (total_weeks_saved : ℕ) : ℕ := weekly_savings * total_weeks_saved

-- Define the fraction used to pay bills calculation
def fraction_used (total_savings : ℕ) (amount_used : ℕ) : ℚ := (amount_used : ℚ) / (total_savings : ℚ)

-- Goal: Prove that the fraction used is 1/3 given the above conditions
theorem Carl_fraction_used_to_pay_bills :
  let total_savings := total_savings weekly_savings total_weeks_saved
  in fraction_used total_savings amount_used = 1/3 :=
by {
  -- This is where we would prove the statement.
  sorry
}

end Carl_fraction_used_to_pay_bills_l390_390076


namespace bc_value_l390_390318

noncomputable def right_triangle_bc (A B C P Q : ℝ) :=
  ∃ (x y : ℝ), 
  (4 * x^2 + y^2 = 625) ∧ 
  (x^2 + 4 * y^2 = 225) ∧ 
  (BP = 25) ∧ 
  (CQ = 15) ∧ 
  (BC = real.sqrt (x^2 + y^2))

theorem bc_value {A B C : Point} {P Q : Point} 
  (h1 : ∃ (A B C : Point), right_triangle A B C ∧ ∠BAC = 90) 
  (h2 : midpoint P A B) 
  (h3 : midpoint Q A C) 
  (h4 : dist B P = 25) 
  (h5 : dist C Q = 15) :
  dist B C = real.sqrt 370 :=
by sorry

end bc_value_l390_390318


namespace eccentricity_range_l390_390312

variables {a b : ℝ}
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

theorem eccentricity_range (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ (x y : ℝ), ellipse x y → distance x y 0 b ≤ 2 * b) :
  0 < (sqrt (a^2 - b^2) / a) ∧ (sqrt (a^2 - b^2) / a) ≤ sqrt 2 / 2 :=
sorry

end eccentricity_range_l390_390312


namespace Sn_max_at_8_l390_390126

variable {α : Type} [LinearOrderedField α]
variable (a_n : ℕ → α) (S_n : ℕ → α) (d : α)

-- Conditions
def a1 : a_n 1 = 29 := sorry
def a3_7a6 : 3 * a_n 3 = 7 * a_n 6 := sorry

-- Sequence definition
def a_n_def : ∀ n, a_n n = 29 - 4 * (n - 1) := sorry

-- Sum definition
def S_n_def : ∀ n, S_n n = n * (29 - 2 * n + 1) := sorry

-- Proof statement
theorem Sn_max_at_8 : ∀ n, S_n n <= S_n 8 :=
by
  assume n
  -- Proof that the sum S_n reaches its maximum at n = 8
  sorry

end Sn_max_at_8_l390_390126


namespace count_algebraic_exprs_l390_390207

-- Definitions of the given expressions
def expr1 := 2 * (x : ℝ)^2
def expr2 := 1 - 2 * (x : ℝ) = 0
def expr3 := (a : ℝ) * (b : ℝ)
def expr4 := (a : ℝ) > 0
def expr5 := 0
def expr6 := 1 / (a : ℝ)
def expr7 := Real.pi

-- Condition checking if the given expressions are algebraic
def is_algebraic (e : ℝ → Prop) := ¬ e = (λ x, false)

-- Prove that 5 out of the 7 given expressions are algebraic
theorem count_algebraic_exprs :
  (is_algebraic expr1) ∧ (¬ is_algebraic expr2) ∧ (is_algebraic expr3) ∧ (¬ is_algebraic expr4) ∧ (is_algebraic expr5) ∧ (is_algebraic expr6) ∧ (is_algebraic expr7) →
  5 =
  ([expr1, expr2, expr3, expr4, expr5, expr6, expr7].countp is_algebraic) := by
  sorry

end count_algebraic_exprs_l390_390207


namespace smallest_prime_less_than_perf_square_l390_390421

-- Define a predicate for being a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

-- The main goal
theorem smallest_prime_less_than_perf_square : ∃ n : ℕ, is_prime n ∧ ∃ m : ℕ, n = m^2 - 8 ∧ (∀ k : ℕ, is_prime k ∧ ∃ l : ℕ, k = l^2 - 8 → k ≥ n) :=
begin
  use 17,
  split,
  -- Proof that 17 is a prime number
  {
    unfold is_prime,
    split,
    { exact dec_trivial },
    { intros d hd,
      have h_d : d = 1 ∨ d = 17,
      { cases d,
        { exfalso, linarith, },
        { cases d,
          { left, refl, },
          { right, linarith [Nat.Prime.not_dvd_one 17 hd], }, }, },
      exact h_d, },
  },
  -- Proof that 17 is 8 less than a perfect square and the smallest such prime
  {
    use 5,
    split,
    { refl, },
    { intros k hk,
      cases hk with hk_prime hk_cond,
      cases hk_cond with l hl,
      rw hl,
      have : l ≥ 5,
      { intros,
        linarith, },
      exact this, },
  }
end

end smallest_prime_less_than_perf_square_l390_390421


namespace kerosene_price_and_egg_equivalence_l390_390211

-- Definitions based on the conditions
def cost_of_rice_per_pound : ℝ := 0.33
def dozen_eggs := 12
def cost_of_egg := cost_of_rice_per_pound / dozen_eggs
def cost_of_half_liter_kerosene := 12 * cost_of_egg
def cost_of_liter_kerosene := 2 * cost_of_half_liter_kerosene

-- Proof problem statement
theorem kerosene_price_and_egg_equivalence :
  (∃ x : ℝ, x = 12 * cost_of_egg ∧ 2 * x = 0.66) :=
by
  use cost_of_half_liter_kerosene
  split
  · rfl
  · have h : 2 * cost_of_half_liter_kerosene = 2 * (12 * cost_of_egg)
    simp [cost_of_half_liter_kerosene, cost_of_egg]
    norm_num

end kerosene_price_and_egg_equivalence_l390_390211


namespace faster_train_length_l390_390453

/-- Given two trains moving in the same direction, with speeds 162 kmph and 18 kmph respectively, 
and the faster train crosses a man in the slower train in 33 seconds, 
prove that the length of the faster train is 1320 meters. -/
theorem faster_train_length
  (speed_faster : ℕ) (speed_slower : ℕ) (cross_time : ℕ)
  (faster_speed_is_162 : speed_faster = 162)
  (slower_speed_is_18 : speed_slower = 18)
  (cross_time_is_33 : cross_time = 33) :
  ∃ length_faster : ℕ, length_faster = 1320 :=
begin
  sorry
end

end faster_train_length_l390_390453


namespace new_arithmetic_mean_l390_390735

theorem new_arithmetic_mean (mean : ℝ) (n : ℕ) (nums : list ℝ) (removed : list ℝ)
  (h₁ : mean = 50)
  (h₂ : n = 60)
  (h₃ : nums.length = n)
  (h₄ : removed = [40, 50, 55, 65])
  (h₅ : ∀ x ∈ removed, x ∈ nums)
  (h₆ : ∀ x ∈ nums, x ∉ nums.drop (nums.index_of x)) :
  let original_sum := mean * n,
      removed_sum := removed.sum,
      new_sum := original_sum - removed_sum,
      remaining_count := n - removed.length,
      new_mean := new_sum / remaining_count in
  new_mean = 50 :=
by {
  sorry
}

end new_arithmetic_mean_l390_390735


namespace combinatorial_identity_l390_390842

-- Define the combinatorial binomial coefficient
open BigOperators

-- Lean definitions for binomial coefficients
def binom (n k : ℕ) : ℕ := n.choose k

-- Problem statement
theorem combinatorial_identity (C : ℕ → ℕ → ℕ) :
  C 7 4 + C 7 5 + C 8 6 = C 9 6 :=
by
  rw [Nat.choose_succ_succ, Nat.choose_succ_succ, ←Nat.choose_succ_succ 8 6]
  sorry

end combinatorial_identity_l390_390842


namespace number_of_Ohs_invariant_l390_390720

noncomputable def seats := 1000
noncomputable def initial_seats := 100
noncomputable def n_tickets_sold (n : ℕ) := n > initial_seats ∧ n < seats

-- k_i is the cumulative function for tickets sold from seat 1 to seat i.
noncomputable def k_i (i : ℕ) (tickets_sold : ℕ → ℕ) : ℕ := 
  if i = 0 then 0 else tickets_sold i + k_i (i - 1) tickets_sold

theorem number_of_Ohs_invariant (n : ℕ) (tickets_sold : ℕ → ℕ) :
  n_tickets_sold n →
  ∀ order : list ℕ, 
    ∀ i : ℕ, i ≤ initial_seats → 
    tickets_sold i ≥ 
      sum (list.map (λ x, if x ≤ i then 1 else 0) order) + 
      n - initial_seats
        → 
    let ohs := list.foldl (λ acc seat, acc + (tickets_sold seat - seat)) 0 order 
    in ohs = n - initial_seats
:= by
  sorry

end number_of_Ohs_invariant_l390_390720


namespace count_eight_letter_good_words_l390_390856

def is_good_word (word : List Char) : Prop :=
  ∀ (i : ℕ), i < word.length - 1 →
    (word[i] = 'A' → word[i+1] ≠ 'B') ∧
    (word[i] = 'B' → word[i+1] ≠ 'C') ∧
    (word[i] = 'C' → word[i+1] ≠ 'A')

def num_good_words (n : ℕ) : ℕ :=
  if n = 1 then 4 
  else let prev_words := num_good_words (n - 1) in
    -- this part actually should be the logic to count the
    -- number of good words, which involves dynamic programming approach
    sorry

theorem count_eight_letter_good_words :
  num_good_words 8 = a_8 + b_8 + c_8 + d_8 := 
sorry

end count_eight_letter_good_words_l390_390856


namespace no_geom_seq_exists_l390_390232

noncomputable def geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = q * a n

theorem no_geom_seq_exists (a : ℕ → ℝ) (q : ℝ) : 
  geometric_seq a q → 
  (a 1 + a 6 = 11) →
  (a 3 * a 4 = 32 / 9) → 
  (∀ n : ℕ, a (n + 1) > a n) → 
  (∃ m > 4, 
    let a_m := a m in
    let a_m_minus_1 := a (m - 1) in
    let a_m_plus_1 := a (m + 1) in
    (2 / 3 * a_m_minus_1), (a_m ^ 2), (a_m_plus_1 + 4 / 9)).is_arithmetic_sequence_in_order → 
  False :=
by
  intros h_geom h1 h2 h_incr h_arith
  sorry

end no_geom_seq_exists_l390_390232


namespace monomial_sum_exponent_eq_l390_390201

theorem monomial_sum_exponent_eq:
  ∀ (m n : ℝ), 2 * m + 3 = 4 ∧ n = 3 → (4 * m - n) ^ n = -1 :=
by
  intros m n h
  cases h with hm hn
  rw hn
  rw hm
  sorry

end monomial_sum_exponent_eq_l390_390201


namespace distribution_ways_one_two_four_distribution_ways_three_two_two_l390_390398

theorem distribution_ways_one_two_four (books : Finset ℕ) (n : books.card = 7) :
  ∃ (A B C : Finset ℕ), A.card = 1 ∧ B.card = 2 ∧ C.card = 4 ∧ 
  (A ∪ B ∪ C = books) ∧ (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧ 
  (books.card.choose 1 * (books.card - 1).choose 2 * (books.card - 1 - 2).choose 4 * 3.factorial = 630) := 
by
  sorry

theorem distribution_ways_three_two_two (books : Finset ℕ) (n : books.card = 7) :
  ∃ (A B C : Finset ℕ), A.card = 3 ∧ B.card = 2 ∧ C.card = 2 ∧ 
  (A ∪ B ∪ C = books) ∧ (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧ 
  (books.card.choose 3 * ((books.card - 3).choose 2 * ((books.card - 3 - 2).choose 2) / 2) * 3.factorial = 630) := 
by
  sorry

end distribution_ways_one_two_four_distribution_ways_three_two_two_l390_390398


namespace Stan_pays_magician_l390_390370

theorem Stan_pays_magician :
  let hours_per_day := 3
  let days_per_week := 7
  let weeks := 2
  let hourly_rate := 60
  let total_hours := hours_per_day * days_per_week * weeks
  let total_payment := hourly_rate * total_hours
  total_payment = 2520 := 
by 
  sorry

end Stan_pays_magician_l390_390370


namespace ordered_triple_satisfies_curve_equation_l390_390027

def parametric_equations (t : ℝ) : ℝ × ℝ :=
  (3 * Real.sin t + Real.cos t, 3 * Real.cos t)

def curve_equation (a b c : ℝ) (x y : ℝ) : ℝ :=
  a * x^2 + b * x * y + c * y^2

theorem ordered_triple_satisfies_curve_equation (a b c : ℝ) (h₀ : a = 1/3) (h₁ : b = -2/9) (h₂ : c = 0) 
  : ∀ t : ℝ, curve_equation a b c (fst (parametric_equations t)) (snd (parametric_equations t)) = 3 :=
  by
    sorry

end ordered_triple_satisfies_curve_equation_l390_390027


namespace correct_product_l390_390643

theorem correct_product (a b : ℕ)
  (h1 : 10 ≤ a ∧ a < 100)  -- a is a two-digit number
  (h2 : 0 < b)  -- b is a positive integer
  (h3 : (a % 10) * 10 + (a / 10) * b = 161)  -- Reversing the digits of a and multiplying by b yields 161
  : a * b = 224 := 
sorry

end correct_product_l390_390643


namespace range_of_eccentricity_l390_390279

variables {a b c e : ℝ}

def ellipse (x y : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)
def upper_vertex := b > 0
def distance_condition (x_0 y_0 : ℝ) := (x_0^2 + (y_0 - b)^2 ≤ (2 * b) ^ 2)
def eccentricity := e = c / a
def ellipse_condition := a = real.sqrt (b^2 + c^2)

theorem range_of_eccentricity (h1 : a > b) (h2 : upper_vertex) 
  (h3 : ∀ x_0 y_0, ellipse x_0 y_0 → distance_condition x_0 y_0)
  (h4 : ellipse_condition) :
  0 < e ∧ e ≤ real.sqrt 2 / 2 := 
by
  sorry

end range_of_eccentricity_l390_390279


namespace marbles_solution_l390_390019

def marbles_problem : Prop :=
  let total_marbles := 20
  let blue_marbles := 6
  let red_marbles := 9
  let total_prob_red_white := 0.7
  let white_marbles := 5
  total_marbles = blue_marbles + red_marbles + white_marbles ∧
  (white_marbles / total_marbles + red_marbles / total_marbles = total_prob_red_white)

theorem marbles_solution : marbles_problem :=
by {
  sorry
}

end marbles_solution_l390_390019


namespace rangeOfA_l390_390198

theorem rangeOfA (a : ℝ) : 
  (∃ x : ℝ, 9^x + a * 3^x + 4 = 0) → a ≤ -4 :=
by
  sorry

end rangeOfA_l390_390198


namespace eccentricity_range_l390_390268

noncomputable def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  { z : ℝ × ℝ // (z.1^2 / a^2) + (z.2^2 / b^2) = 1 }

theorem eccentricity_range (a b : ℝ) (h : a > b) (hb : b > 0):
  ∀ (P : ellipse a b h hb),
    |(P.1, P.2 - b).norm ≤ 2 * b → 
    eccentricity (a b) ∈ set.Ici (real.sqrt 2 / 2) :=
sorry

end eccentricity_range_l390_390268


namespace variance_of_arithmetic_sequence_common_diff_3_l390_390911

noncomputable def variance (ξ : List ℝ) : ℝ :=
  let n := ξ.length
  let mean := ξ.sum / n
  let var_sum := (ξ.map (fun x => (x - mean) ^ 2)).sum
  var_sum / n

def arithmetic_sequence (a1 : ℝ) (d : ℝ) (n : ℕ) : List ℝ :=
  List.range n |>.map (fun i => a1 + i * d)

theorem variance_of_arithmetic_sequence_common_diff_3 :
  ∀ (a1 : ℝ),
    variance (arithmetic_sequence a1 3 9) = 60 :=
by
  sorry

end variance_of_arithmetic_sequence_common_diff_3_l390_390911


namespace eccentricity_range_l390_390284

noncomputable def ellipse_eccentricity (a b : ℝ) (h_ab : a > b) : ℝ := (Real.sqrt (a ^ 2 - b ^ 2)) / a

theorem eccentricity_range (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ) :
  (∃ x y : ℝ, (x, y) = P ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  (Real.sqrt (x^2 + (y - b)^2) ≤ 2 * b)) → 
  0 < ellipse_eccentricity a b h_ab ∧ ellipse_eccentricity a b h_ab ≤ Real.sqrt 2 / 2 :=
begin
  sorry
end

end eccentricity_range_l390_390284


namespace parabola_line_intersection_length_l390_390922

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y k : ℝ) : Prop := y = k * x - 1
def focus : ℝ × ℝ := (1, 0)

theorem parabola_line_intersection_length (k x1 x2 y1 y2 : ℝ)
  (h_focus : line 1 0 k)
  (h_parabola1 : parabola x1 y1)
  (h_parabola2 : parabola x2 y2)
  (h_line1 : line x1 y1 k)
  (h_line2 : line x2 y2 k) :
  k = 1 ∧ (x1 + x2 + 2) = 8 :=
by
  sorry

end parabola_line_intersection_length_l390_390922


namespace range_of_2a_plus_3b_l390_390893

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 ≤ a + b ∧ a + b ≤ 1) (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l390_390893


namespace sequence_arith_geo_l390_390713

theorem sequence_arith_geo : 
  let seq := [3, 9, 2187] in
  (∃ a r, ∀ n, seq n = a * r^n) ∨ 
  (∃ a d, ∀ n, seq n = a + n * d) := 
by
  sorry

end sequence_arith_geo_l390_390713


namespace ω_2n_eq_ω_n_ω_8n_5_eq_ω_4n_3_ω_2n_minus_1_eq_n_l390_390333

def ω (n : ℕ) : ℕ :=
  n.bits.count (λ b => b)

theorem ω_2n_eq_ω_n (n : ℕ) : ω (2 * n) = ω n := by
  sorry

theorem ω_8n_5_eq_ω_4n_3 (n : ℕ) : ω (8 * n + 5) = ω (4 * n + 3) := by
  sorry

theorem ω_2n_minus_1_eq_n (n : ℕ) : ω (2^n - 1) = n := by
  sorry

end ω_2n_eq_ω_n_ω_8n_5_eq_ω_4n_3_ω_2n_minus_1_eq_n_l390_390333


namespace margaret_time_l390_390691

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def total_permutations (n : Nat) : Nat :=
  factorial n

def total_time_in_minutes (total_permutations : Nat) (rate : Nat) : Nat :=
  total_permutations / rate

def time_in_hours_and_minutes (total_minutes : Nat) : Nat × Nat :=
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  (hours, minutes)

theorem margaret_time :
  let n := 8
  let r := 15
  let permutations := total_permutations n
  let total_minutes := total_time_in_minutes permutations r
  time_in_hours_and_minutes total_minutes = (44, 48) := by
  sorry

end margaret_time_l390_390691


namespace zeta_sum_fifth_power_l390_390319

noncomputable def complex_nums : Type := {z : ℂ // true} -- Define complex numbers
variables (ζ1 ζ2 ζ3 : complex_nums)

axiom ζ1_ζ2_ζ3_conditions :
  (ζ1 + ζ2 + ζ3 = 2) ∧
  (ζ1^2 + ζ2^2 + ζ3^2 = 6) ∧
  (ζ1^3 + ζ2^3 + ζ3^3 = 8)

theorem zeta_sum_fifth_power (ζ1 ζ2 ζ3 : complex_nums)
  (h : (ζ1 + ζ2 + ζ3 = 2) ∧ (ζ1^2 + ζ2^2 + ζ3^2 = 6) ∧ (ζ1^3 + ζ2^3 + ζ3^3 = 8)) :
  ζ1^5 + ζ2^5 + ζ3^5 = 30 :=
by sorry

end zeta_sum_fifth_power_l390_390319


namespace range_of_2a_plus_3b_l390_390895

theorem range_of_2a_plus_3b (a b : ℝ)
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1)
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) :
  -3 ≤ 2 * a + 3 * b ∧ 2 * a + 3 * b ≤ 3 :=
by
  sorry

end range_of_2a_plus_3b_l390_390895


namespace find_parallel_tangent_line_l390_390607

noncomputable def parallel_tangents (O1 O2 P : Point) (r1 r2 : ℝ) (k1 k2 : Circle) : Prop :=
  ∃ l : Line, l ∋ P ∧ tangents_parallel l k1 k2

variables {Point Line Circle : Type}
variables (O1 O2 P : Point) (r1 r2 : ℝ) (k1 k2 : Circle)
variables (tangents_parallel : Line → Circle → Circle → Prop)

theorem find_parallel_tangent_line (h1 : O1 ≠ O2) (h2 : r1 > r2) :
  parallel_tangents O1 O2 P r1 r2 k1 k2 :=
by sorry

end find_parallel_tangent_line_l390_390607


namespace odd_function_at_2_l390_390581

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)

theorem odd_function_at_2 (f g : ℝ → ℝ)
  (hf : is_odd f)
  (hg : ∀ x, g(x) = f(x) + 9)
  (h : g(-2) = 3) :
  f(2) = 6 :=
by
  sorry

end odd_function_at_2_l390_390581


namespace xiaoqiang_matches_l390_390828

def students : Type := {A, B, C, D, E, Xiaoqiang}

def matches_played : {p // p ∈ students} → ℕ
| ⟨A, _⟩ => 5
| ⟨B, _⟩ => 4
| ⟨C, _⟩ => 3
| ⟨D, _⟩ => 2
| ⟨E, _⟩ => 1
| ⟨Xiaoqiang, _⟩ => match_played_by_Xiaoqiang

axiom condition1 (x : {p // p ∈ students}) :
  x ≠ ⟨Xiaoqiang, _⟩ → (matches_played x < 6)

theorem xiaoqiang_matches :
  matches_played ⟨Xiaoqiang, sorry⟩ = 3 :=
sorry

end xiaoqiang_matches_l390_390828


namespace part1_part2_l390_390149

noncomputable def f (a x : ℝ) : ℝ := x + a / x + a^2 - 2

theorem part1 (a : ℝ) (h_odd : ∀ x, f a x = - f a (-x))
  (h_increasing : ∀ x > 0, 1 - (a / x^2) ≥ 0) : a = -Real.sqrt 2 :=
sorry

theorem part2 (a m n : ℝ) (h_diff_roots : m ≠ n)
  (h_interval : m ∈ Ioo -1 1 ∧ n ∈ Ioo -1 1)
  (h_eq : |Real.log 8 (m + 1)| + a^2 - f a 1 = 0 ∧ |Real.log 8 (n + 1)| + a^2 - f a 1 = 0) :
  (2 / 3 < a ∧ a < 1) ∧ (1 / m + 1 / n = -1) :=
sorry

end part1_part2_l390_390149


namespace arctan_identity_for_right_triangle_l390_390653

theorem arctan_identity_for_right_triangle 
  (a b c : ℝ) 
  (h_triangle : a^2 + c^2 = b^2) : 
  arctan (a / (c + b)) + arctan (c / (a + b)) = π / 4 :=
by 
  sorry

end arctan_identity_for_right_triangle_l390_390653


namespace GH_divides_parallelogram_eq_parts_l390_390088

variables {A B C D E F G H : Type*}
variables [Parallelogram ABCD]
variables [Point E] [Point F] [Point G] [Point H]

-- Given that E is on BC and F is on AD
-- Given the definitions of intersections using existing theorems
def E_on_BC (BC: Parallelogram.C.side) (E: Point) : Prop :=
  E ∈ BC

def F_on_AD (AD: Parallelogram.D.side) (F: Point) : Prop :=
  F ∈ AD

-- Given that line AE intersects BF at G, and ED intersects CF at H
def AE_BF_intersect_G (AE: Line) (BF: Line) (G: Point) : Prop :=
  G ∈ AE ∧ G ∈ BF

def ED_CF_intersect_H (ED: Line) (CF: Line) (H: Point) : Prop :=
  H ∈ ED ∧ H ∈ CF

-- Prove that line GH divides the parallelogram into two equal parts
theorem GH_divides_parallelogram_eq_parts (ABCD: Parallelogram)
    (BC: Parallelogram.C.side) (E: Point) (F: Point) (G: Point) (H: Point)
    (h1: E_on_BC BC E) (h2: F_on_AD ABCD.D.side F)
    (h3: AE_BF_intersect_G (line_through ABCD.A E) (line_through ABCD.B F) G)
    (h4: ED_CF_intersect_H (line_through E ABCD.D) (line_through ABCD.C F) H) :
    divides_parallelogram_eq_parts GH ABCD := by
  sorry

end GH_divides_parallelogram_eq_parts_l390_390088


namespace total_chairs_l390_390107

theorem total_chairs (rows : ℕ) (chairs_per_row : ℕ) (h1 : rows = 27) (h2 : chairs_per_row = 16) : rows * chairs_per_row = 432 :=
by
  rw [h1, h2]
  norm_num

end total_chairs_l390_390107


namespace right_triangle_hypotenuse_l390_390128

theorem right_triangle_hypotenuse
  (A B C : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  (a b c : A)
  (h₁ : dist a c = AC)
  (h₂ : dist b c = BC)
  (h₃ : angle a c b = π / 2)
  (h₄ : AC = 6)
  (h₅ : BC = 8)
  : dist a b = 10 := sorry

end right_triangle_hypotenuse_l390_390128


namespace correct_option_l390_390937

-- Definition of the set A
def A : Set ℚ := {x | x > -1}

-- Proof statement
theorem correct_option : ¬ (Real.sqrt 2 ∈ A) :=
by
  -- Provide a brief outline of steps leading to contradiction
  intro h
  have h₁ : Real.sqrt 2 ∈ ℚ, -- Assume Rational
  sorry

end correct_option_l390_390937


namespace irrationals_l390_390999

open Classical

variable (x : ℝ)

theorem irrationals (h : x^3 + 2 * x^2 + 10 * x = 20) : Irrational x ∧ Irrational (x^2) :=
by
  sorry

end irrationals_l390_390999


namespace find_ratio_GF_FC_l390_390654

-- Define the points A, B, C in Real Euclidean space
variables {A B C D E F G : ℝ}

-- Define the given conditions
-- AD:DB = 4:1 -> D = (1/5)A + (4/5)B
def D := (1 / 5) * A + (4 / 5) * B

-- BE:EC = 2:3 -> E = (2/5)B + (3/5)C
def E := (2 / 5) * B + (3 / 5) * C

-- Intersection point F of lines DE and AC
-- Expressing F in terms of vector combination
def F (s t : ℝ) : ℝ := s * A + (1 - s) * C
-- Alternatively, 
def F' (t : ℝ) : ℝ := t * D + (1 - t) * E

-- AG:GC = 3:2 -> G = (3/5)A + (2/5)C
def G := (3 / 5) * A + (2 / 5) * C

-- Ratio of vectors GF to FC
def ratio_GF_FC (GF FC : ℝ) : ℝ := GF / FC

-- The main statement we want to prove, which is equivalent to the ratio of GF/FC being 3/2
theorem find_ratio_GF_FC (s t u : ℝ) :
  let D := (1 / 5) * A + (4 / 5) * B in
  let E := (2 / 5) * B + (3 / 5) * C in
  let G := (3 / 5) * A + (2 / 5) * C in
  let F := s * A + (1 - s) * C in -- or could use F' (t)
  let GF := G - F in
  let FC := F - C in
  ratio_GF_FC GF FC = 3 / 2 := 
sorry

end find_ratio_GF_FC_l390_390654


namespace circle_existence_l390_390106

-- Define a theorem for each n from 3 to 9
theorem circle_existence (n : ℕ) : exists_circle n ↔ (n = 3 ∨ n = 7) := by
  sorry

-- Define what it means to have an existing circle for a given n
def exists_circle (n : ℕ) : Prop :=
  ∃ (circle : ℕ → ℕ), 
    -- Circle with six different three-digit numbers none of whose digits are 0
    (∀ i j, (i ≠ j) → (100 ≤ circle i) ∧ (circle i < 1000) ∧ (no_zero_digits (circle i)) ∧ (circle i ≠ circle j)) ∧ 
    -- First two digits of each number are the last two digits of the previous number
    -- Cyclic property of indexing 0, 1, 2, 3, 4, 5 corresponding to a circle.
    (∀ i, i < 6 → (first_two_digits (circle ((i + 1) % 6)) = last_two_digits (circle i))) ∧ 
    -- All numbers in the circle are divisible by n
    (∀ i, i < 6 → (circle i % n = 0))

-- Helper function defining no zero digits
def no_zero_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ digits n → d ≠ 0

-- Helper function to extract digits of a given number
def digits (n : ℕ) : List ℕ :=
  n.digits

-- Helper function to get the first two digits of a number
def first_two_digits (n : ℕ) : ℕ :=
  (n / 10) % 100

-- Helper function to get the last two digits of a number
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

end circle_existence_l390_390106


namespace num_students_inventing_one_problem_l390_390404

theorem num_students_inventing_one_problem :
  ∃ (n : Fin 7 → ℕ) (a : Fin 7 → ℕ), 
    (∑ i, n i = 39) ∧ 
    (∑ i, n i * a i = 60) ∧ 
    (∀ i, a i ≠ 0) ∧ 
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    ∃ k, a k = 1 ∧ n k = 1 :=
sorry

end num_students_inventing_one_problem_l390_390404


namespace outfits_count_l390_390000

theorem outfits_count (blue_shirts yellow_shirts pants blue_hats yellow_hats : Nat)
  (h_blue_shirts : blue_shirts = 6)
  (h_yellow_shirts : yellow_shirts = 6)
  (h_pants : pants = 7)
  (h_blue_hats : blue_hats = 9)
  (h_yellow_hats : yellow_hats = 9) :
  (blue_shirts * pants * yellow_hats + yellow_shirts * pants * blue_hats) = 756 := by
  rw [h_blue_shirts, h_yellow_shirts, h_pants, h_blue_hats, h_yellow_hats]
  calc
    6 * 7 * 9 + 6 * 7 * 9 = 378 + 378 := by simp
    _ = 756 := by simp

end outfits_count_l390_390000


namespace remainder_of_11_pow_2023_mod_33_l390_390841

theorem remainder_of_11_pow_2023_mod_33 : (11 ^ 2023) % 33 = 11 := 
by
  sorry

end remainder_of_11_pow_2023_mod_33_l390_390841


namespace pyramid_arrangement_possible_l390_390075

noncomputable def eight_pyramids_touching : Prop :=
  ∃ (triangles : fin 8 → set ℝ^2) 
    (A B : ℝ^3), 
    (∀ i, (triangles i).subtype.inhabited) ∧
    (∀ i < 4, ∃ P : convex_hull (triangles i) × set.interior (convex_hull (triangles i)) = A) ∧
    (∀ i ≥ 4, ∃ P : convex_hull (triangles i) × set.interior (convex_hull (triangles i)) = B) ∧
    (∀ i j, i ≠ j → convex_hull (triangles i) ∩ convex_hull (triangles j)).nonempty

theorem pyramid_arrangement_possible : eight_pyramids_touching :=
sorry

end pyramid_arrangement_possible_l390_390075


namespace math_problem_l390_390140

variable {a b : ℕ → ℝ}

-- Condition 1: Sum of the first n terms of the sequence {a_n} is S_n = n^2
def S (n : ℕ) : ℝ :=
  n^2

-- Condition 2: b_n = (-1)^n a_n a_{n+1}
def b (n : ℕ) : ℝ :=
  (-1)^n * (a n) * (a (n + 1))

-- Condition 3: Sum of the first n terms of the sequence {b_n} is T_n,
-- such that T_n > t * n^2 - 2 * n for all n ∈ ℕ*
def T (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), b i

variable (t : ℝ)

theorem math_problem (t : ℝ) :
  (∀ n, S n = n ^ 2) ∧
  (∀ n, b n = (-1)^n * (a n) * (a (n + 1))) ∧
  (∀ n, T n > t * n^2 - 2 * n) →
  (∀ n, a n = 2 * n - 1) ∧
  (¬∀ n, n % 2 = 1 → T n = -3 * n^2 + 2 * n - 2) ∧
  (∀ n, T (2 * n) = 8 * n^2 + 4 * n) ∧
  t ∈ (-∞, -2] → False := 
begin
  sorry
end

end math_problem_l390_390140


namespace part_a_l390_390998

theorem part_a (a b c : Int) (h1 : a + b + c = 0) : 
  ¬(a ^ 1999 + b ^ 1999 + c ^ 1999 = 2) :=
by
  sorry

end part_a_l390_390998


namespace rectangle_width_l390_390734

theorem rectangle_width (length_rect : ℝ) (width_rect : ℝ) (side_square : ℝ)
  (h1 : side_square * side_square = 5 * (length_rect * width_rect))
  (h2 : length_rect = 125)
  (h3 : 4 * side_square = 800) : width_rect = 64 :=
by 
  sorry

end rectangle_width_l390_390734


namespace common_difference_l390_390924

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Conditions
axiom h1 : a 3 + a 7 = 10
axiom h2 : a 8 = 8

-- Statement to prove
theorem common_difference (h : ∀ n, a (n + 1) = a n + d) : d = 1 :=
  sorry

end common_difference_l390_390924


namespace area_of_S_div_area_of_T_l390_390677

def supports (x y z a b c : ℝ) : Prop :=
  ((x ≥ a ∧ y ≥ b) ∨ (x ≥ a ∧ z ≥ c) ∨ (y ≥ b ∧ z ≥ c)) ∧ ¬(x ≥ a ∧ y ≥ b ∧ z ≥ c)

def in_plane (x y z : ℝ) : Prop := x + y + z = 1 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

noncomputable def area_ratio : ℝ :=
  let T := {p : ℝ × ℝ × ℝ | in_plane p.1 p.2 p.3} in
  let S := {p : ℝ × ℝ × ℝ | (supports p.1 p.2 p.3 (1/4) (1/2) (1/4)) ∨ (supports p.1 p.2 p.3 (1/3) (1/4) (2/5))} in
  (measure_theory.measure_of T S) / (measure_theory.measure_of T)

theorem area_of_S_div_area_of_T : area_ratio = 1/16 :=
by
  sorry

end area_of_S_div_area_of_T_l390_390677


namespace polynomial_degree_one_l390_390337

open Polynomial

noncomputable def P : Polynomial ℤ := sorry

def sequence (P : Polynomial ℤ) : ℕ → ℤ
| 0       := 0
| (n+1) := P.eval (sequence n)

theorem polynomial_degree_one (P : Polynomial ℤ) :
  P.leading_coeff = 1 →
  (∀ n, sequence P (n + 1) ≠ 0) →
  ∃! d, P.natDegree = d ∧ d = 1 :=
begin
  intros h leading_coeff_one no_repeat,
  sorry
end

end polynomial_degree_one_l390_390337


namespace area_of_hexagon_l390_390375
-- Import necessary libraries

-- Define the problem setup
variables (T : ℝ) (m n p : ℕ)
variables (hm : m > 2) (hn : n > 2) (hp : p > 2)

-- The theorem statement
theorem area_of_hexagon (T : ℝ) (m n p : ℕ) (hm : m > 2) (hn : n > 2) (hp : p > 2) :
  ∃ t : ℝ, t = T * (1 - (m + n + p) / (m * n * p)) :=
begin
  use T * (1 - (m + n + p) / (m * n * p)),
  sorry
end

end area_of_hexagon_l390_390375


namespace sum_of_areas_l390_390709

variables {R r : ℝ} {a b c p : ℝ}
variables {A B C A₁ B₁ C₁ A₂ B₂ C₂ : ℝ}

/-- Given a triangle ABC with semiperimeter p, points A₁, B₁, C₁ of tangency of the incircle,
    and points A₂, B₂, C₂ of tangency of the excircle,
    prove the sum of the areas of the specified triangles is 3R^2 - 4Rr - r^2. -/
theorem sum_of_areas (h₁ : p = (a + b + c) / 2) : 
  3 * R ^ 2 - 4 * R * r - r ^ 2 = 
    (λ ABC A₁ B₁ C₁ A₂ B₂ C₂, 
      ... /* computation here */ sorry) :=
by sorry

end sum_of_areas_l390_390709


namespace range_of_eccentricity_of_ellipse_l390_390258

-- Definitions for the problem.
def is_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1) 

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

def on_upper_vertex (b : ℝ) : (ℝ × ℝ) := (0, b)

-- The statement to prove the range of eccentricity.
theorem range_of_eccentricity_of_ellipse (a b : ℝ) (x y : ℝ) (e : ℝ) :
  is_on_ellipse a b x y →
  (distance x y 0 b ≤ 2 * b) →
  ∃ e, (e = (1 - (b^2 / a^2))^(1/2) ∧ (0 < e) ∧ (e ≤ (sqrt 2)/2)) :=
by
  sorry

end range_of_eccentricity_of_ellipse_l390_390258


namespace remainder_polynomial_division_l390_390873

theorem remainder_polynomial_division (β : ℂ) (hβ : β^5 + β^4 + β^3 + β^2 + β + 1 = 0) : 
  polynomial.eval β (polynomial.C (1 : ℂ) * polynomial.X^60 + polynomial.C 1 * polynomial.X^45 + polynomial.C 1 * polynomial.X^30 + polynomial.C 1 * polynomial.X^15 + polynomial.C 1) = 5 :=
by
  have hβ6 : β^6 = 1 := sorry
  -- additional steps to use hβ6 to evaluate the given polynomial at β
  sorry

end remainder_polynomial_division_l390_390873


namespace range_of_eccentricity_l390_390246

noncomputable def upperVertex (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : ℝ × ℝ := (0, b)

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem range_of_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h: a > b) :
  (∀ (x y : ℝ), ellipse a b x y → distance x y 0 b ≤ 2 * b) →
  ∃ (e : ℝ), e ∈ Set.Ioc 0 (Real.sqrt 2 / 2) ∧ a^2 = b^2 / (1 - e^2) := sorry

end range_of_eccentricity_l390_390246


namespace eccentricity_range_l390_390315

variables {a b : ℝ}
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

theorem eccentricity_range (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ (x y : ℝ), ellipse x y → distance x y 0 b ≤ 2 * b) :
  0 < (sqrt (a^2 - b^2) / a) ∧ (sqrt (a^2 - b^2) / a) ≤ sqrt 2 / 2 :=
sorry

end eccentricity_range_l390_390315


namespace maximize_distance_l390_390890

def front_tire_lifespan : ℕ := 20000
def rear_tire_lifespan : ℕ := 30000
def max_distance : ℕ := 24000

theorem maximize_distance : max_distance = 24000 := sorry

end maximize_distance_l390_390890


namespace base_n_system_digits_l390_390360

theorem base_n_system_digits (N : ℕ) (h : N ≥ 6) :
  ((N - 1) ^ 4).digits N = [N-4, 5, N-4, 1] :=
by
  sorry

end base_n_system_digits_l390_390360


namespace bob_walks_more_l390_390974

def street_width : ℝ := 30
def length_side1 : ℝ := 500
def length_side2 : ℝ := 300

def perimeter (length width : ℝ) : ℝ := 2 * (length + width)

def alice_perimeter : ℝ := perimeter (length_side1 + 2 * street_width) (length_side2 + 2 * street_width)
def bob_perimeter : ℝ := perimeter (length_side1 + 4 * street_width) (length_side2 + 4 * street_width)

theorem bob_walks_more :
  bob_perimeter - alice_perimeter = 240 :=
by
  sorry

end bob_walks_more_l390_390974


namespace angle_bisector_length_l390_390060

theorem angle_bisector_length {a b : ℝ} {C : ℝ} (hC : 0 < C ∧ C < π) :
  let CD := (2 * a * b * real.cos (C / 2)) / (a + b)
  in CD = (2 * a * b * real.cos (C / 2)) / (a + b) :=
by
  sorry

end angle_bisector_length_l390_390060


namespace find_f_of_5_l390_390903

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x < 2 then x^2 + 3*x - 1 else sorry

-- State the given conditions
axiom functional_eq : ∀ x : ℝ, f(x) = 2 * f(x - 2)
axiom piecewise_eq : ∀ x : ℝ, (0 ≤ x ∧ x < 2) → f(x) = x^2 + 3*x - 1

-- Statement of the theorem to prove
theorem find_f_of_5 : f 5 = 12 :=
by
  sorry

end find_f_of_5_l390_390903


namespace determine_x_l390_390087

theorem determine_x : (∃ x : ℚ, 3 ^ (4 * x^2 - 9 * x + 5) = 3 ^ (4 * x^2 + 3 * x - 7)) → x = 1 :=
by
  sorry

end determine_x_l390_390087


namespace ω_2n_eq_ω_n_ω_8n_5_eq_ω_4n_3_ω_2n_minus_1_eq_n_l390_390331

def ω (n : ℕ) : ℕ :=
  n.bits.count (λ b => b)

theorem ω_2n_eq_ω_n (n : ℕ) : ω (2 * n) = ω n := by
  sorry

theorem ω_8n_5_eq_ω_4n_3 (n : ℕ) : ω (8 * n + 5) = ω (4 * n + 3) := by
  sorry

theorem ω_2n_minus_1_eq_n (n : ℕ) : ω (2^n - 1) = n := by
  sorry

end ω_2n_eq_ω_n_ω_8n_5_eq_ω_4n_3_ω_2n_minus_1_eq_n_l390_390331


namespace num_distinct_constructions_l390_390811

-- Define the problem statement
def cube_construction_ways :=
  (∃ (cubes : Finset (ℕ × ℕ × ℕ)),
      cubes.card = 8 ∧
      ∃ white_cubes blue_cubes,
      white_cubes ∩ blue_cubes = ∅ ∧
      white_cubes ∪ blue_cubes = cubes ∧
      white_cubes.card = 5 ∧
      blue_cubes.card = 3 ∧
      (∀ rotation reflection, 
        cubes ≈ rotation(cubes) ∧
        cubes ≈ reflection(cubes))
  )

theorem num_distinct_constructions : cube_construction_ways → 3 :=
by
  sorry

end num_distinct_constructions_l390_390811


namespace max_value_of_y_over_x_l390_390193

theorem max_value_of_y_over_x {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 2 * y = 3) :
  y / x ≤ 9 / 8 :=
sorry

end max_value_of_y_over_x_l390_390193


namespace probability_fourth_term_integer_l390_390349

def initial_term : ℕ := 8

/-- The function that generates the next term in the sequence based on coin flip -/
def next_term (previous : ℚ) (heads : Bool) : ℚ :=
  if heads then 2 * previous - 2 else previous / 2 - 2

/-- The fourth term in the sequence after a series of coin flips -/
def fourth_term (coin_flips : List Bool) : ℚ :=
  coin_flips.foldl next_term initial_term

def is_integer (x : ℚ) : Prop :=
  ∃ n : ℤ, x = n

theorem probability_fourth_term_integer :
  let outcomes := List.bind (List.bind [true, false]) (λ h1, [true, false].bind (λ h2, [true, false].bind (λ h3, [true, false].map (λ h4, [h1, h2, h3, h4]))))
  let integer_outcomes := outcomes.filter (λ coin_flips, is_integer (fourth_term coin_flips))
  (integer_outcomes.length : ℚ) / outcomes.length = 3 / 4 :=
sorry

end probability_fourth_term_integer_l390_390349


namespace evaluate_expression_l390_390093

theorem evaluate_expression : 68 + (126 / 18) + (35 * 13) - 300 - (420 / 7) = 170 := by
  sorry

end evaluate_expression_l390_390093


namespace min_letters_required_l390_390760

theorem min_letters_required (n : ℕ) (hn : n = 26) : 
  ∃ k, (∀ (collectors : Fin n) (leader : Fin n), k = 2 * (n - 1)) := 
sorry

end min_letters_required_l390_390760


namespace hyperbola_eccentricity_l390_390138

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (e : ℝ) (h3 : e = (Real.sqrt 3) / 2) 
  (h4 : a ^ 2 = b ^ 2 + (Real.sqrt 3) ^ 2) : (Real.sqrt 5) / 2 = 
    (Real.sqrt (a ^ 2 + b ^ 2)) / a :=
by
  sorry

end hyperbola_eccentricity_l390_390138


namespace eccentricity_hyperbola_l390_390155

-- Definitions of the hyperbola equation and related conditions
def hyperbola (x y a b : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

-- Definition of the asymptotic line equation
def asymptotic_line (y x : ℝ) : Prop :=
  (y = 2 * x)

-- Proving the eccentricity
theorem eccentricity_hyperbola (a b c : ℝ) :
  hyperbola x y a b →
  asymptotic_line y x →
  (c^2 = a^2 + b^2) →
  (b^2 = 4 * a^2) →
  (c = sqrt (a^2 + b^2)) →
  (c = sqrt 5 * a) →
  (c / a = sqrt 5) :=
by
  sorry

end eccentricity_hyperbola_l390_390155


namespace root_expression_value_l390_390622

theorem root_expression_value (m : ℝ) (h : 2 * m^2 + 3 * m - 1 = 0) : 4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end root_expression_value_l390_390622


namespace perpendicular_centroid_orthocenter_l390_390853

theorem perpendicular_centroid_orthocenter 
  (A B C D O : Point) 
  (M N P Q : Point)
  (hO : intersect (line A C) (line B D) = O)
  (hM : centroid A O B = M)
  (hN : centroid C O D = N)
  (hP : orthocenter B O C = P)
  (hQ : orthocenter D O A = Q) 
  : is_perpendicular (line M N) (line P Q) :=
sorry

end perpendicular_centroid_orthocenter_l390_390853


namespace tub_capacity_120_l390_390714

noncomputable def tub_capacity (alternate_minutes : ℕ) (leak_rate : ℕ) (flow_rate : ℕ) (total_time : ℕ) : ℕ :=
  let net_gain_per_cycle := (flow_rate - leak_rate) + (-leak_rate)
  let number_of_cycles := total_time / alternate_minutes
  in number_of_cycles * net_gain_per_cycle

theorem tub_capacity_120 : tub_capacity 2 1 12 24 = 120 := by
  sorry

end tub_capacity_120_l390_390714


namespace distance_between_closest_points_l390_390850

noncomputable def distance_closest_points (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : ℝ :=
  (Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2) - r1 - r2)

theorem distance_between_closest_points :
  distance_closest_points (4, 4) (20, 12) 4 12 = 4 * Real.sqrt 20 - 16 :=
by
  sorry

end distance_between_closest_points_l390_390850


namespace larry_likes_numbers_l390_390996

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def valid_pairs : Finset (ℕ × ℕ) :=
  (Finset.range 10).product (Finset.range 10)

def valid_last_two_digits : Finset (ℕ × ℕ) :=
  valid_pairs.filter (λ p, is_divisible_by_4 (10 * p.1 + p.2))

theorem larry_likes_numbers : valid_last_two_digits.card = 25 := by sorry

end larry_likes_numbers_l390_390996


namespace Chad_savings_l390_390510

theorem Chad_savings :
  let earnings_mowing := 600
  let earnings_birthday := 250
  let earnings_video_games := 150
  let earnings_odd_jobs := 150
  let total_earnings := earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs
  let savings_rate := 0.40
  let savings := savings_rate * total_earnings
  savings = 460 :=
by
  -- Definitions
  let earnings_mowing : ℤ := 600
  let earnings_birthday : ℤ := 250
  let earnings_video_games : ℤ := 150
  let earnings_odd_jobs : ℤ := 150
  let total_earnings : ℤ := earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs
  let savings_rate := (40:ℚ) / 100
  let savings : ℚ := savings_rate * total_earnings
  -- Proof (to be completed by sorry)
  exact sorry

end Chad_savings_l390_390510


namespace range_of_eccentricity_l390_390277

variables {a b c e : ℝ}

def ellipse (x y : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)
def upper_vertex := b > 0
def distance_condition (x_0 y_0 : ℝ) := (x_0^2 + (y_0 - b)^2 ≤ (2 * b) ^ 2)
def eccentricity := e = c / a
def ellipse_condition := a = real.sqrt (b^2 + c^2)

theorem range_of_eccentricity (h1 : a > b) (h2 : upper_vertex) 
  (h3 : ∀ x_0 y_0, ellipse x_0 y_0 → distance_condition x_0 y_0)
  (h4 : ellipse_condition) :
  0 < e ∧ e ≤ real.sqrt 2 / 2 := 
by
  sorry

end range_of_eccentricity_l390_390277


namespace equations_have_different_graphs_l390_390443

theorem equations_have_different_graphs :
  ¬(∀ x : ℝ, (2 * (x - 3)) / (x + 3) = 2 * (x - 3) ∧ 
              (x + 3) * ((2 * x^2 - 18) / (x + 3)) = 2 * x^2 - 18 ∧
              (2 * x - 3) = (2 * (x - 3)) ∧ 
              (2 * x - 3) = (2 * x - 3)) :=
by
  sorry

end equations_have_different_graphs_l390_390443


namespace hotdogs_total_l390_390351

theorem hotdogs_total:
  let e := 2.5
  let l := 2 * (e * 2)
  let m := 7
  let h := 1.5 * (e * 2)
  let z := 0.5
  (e * 2 + l + m + h + z) = 30 := 
by
  sorry

end hotdogs_total_l390_390351


namespace eccentricity_range_l390_390290

-- Ellipse definition
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Upper vertex B of the ellipse
def B (b : ℝ) : ℝ × ℝ := (0, b)

-- Distance PB condition
def distance_le_2b (a b x0 y0 : ℝ) (h : ellipse a b (and.intro (sorry) (sorry)) x0 y0) : Prop :=
  (x0 - 0)^2 + (y0 - b)^2 ≤ (2 * b)^2

-- Range of eccentricity
def eccentricity (a b c : ℝ) : ℝ :=
  c / a

-- Main theorem
theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  (forall (x0 y0 : ℝ), ellipse a b h x0 y0 → distance_le_2b a b x0 y0 h) →
  eccentricity a b (sqrt (a^2 - b^2)) ∈ set.Icc 0 (sqrt 2 / 2) :=
by
  sorry

end eccentricity_range_l390_390290


namespace probability_A_C_not_first_day_l390_390206

theorem probability_A_C_not_first_day :
  let people := ["A", "B", "C"],
  let assignments := people.permutations,
  let valid_outcomes := assignments.filter (λ a, a.head = "B"),
  (valid_outcomes.length : ℚ) / (assignments.length : ℚ) = 1 / 3 :=
by
  sorry

end probability_A_C_not_first_day_l390_390206


namespace eccentricity_range_l390_390282

noncomputable def ellipse_eccentricity (a b : ℝ) (h_ab : a > b) : ℝ := (Real.sqrt (a ^ 2 - b ^ 2)) / a

theorem eccentricity_range (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ) :
  (∃ x y : ℝ, (x, y) = P ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  (Real.sqrt (x^2 + (y - b)^2) ≤ 2 * b)) → 
  0 < ellipse_eccentricity a b h_ab ∧ ellipse_eccentricity a b h_ab ≤ Real.sqrt 2 / 2 :=
begin
  sorry
end

end eccentricity_range_l390_390282


namespace count_integers_with_digit_sum_eq_8_l390_390946

-- Define the function to compute the digit sum of a number
def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) + digit_sum (n / 10)

-- Define the main theorem to prove the count of integers with digit sum of 8
theorem count_integers_with_digit_sum_eq_8 : ∃ (k : ℕ), k = 495 ∧ ∀ m : ℕ, 0 ≤ m ∧ m < 100000 → digit_sum m = 8 ↔ m ∈ finset.Ico 0 100000  :=
sorry

end count_integers_with_digit_sum_eq_8_l390_390946


namespace sean_more_whistles_than_charles_l390_390719

theorem sean_more_whistles_than_charles :
  ∀ (sean_whistles charles_whistles difference : ℕ), 
  sean_whistles = 45 → 
  charles_whistles = 13 → 
  difference = sean_whistles - charles_whistles → 
  difference = 32 :=
by
  intros sean_whistles charles_whistles difference h_sean h_charles h_difference
  rw [h_sean, h_charles, h_difference]
  sorry

end sean_more_whistles_than_charles_l390_390719


namespace additional_experts_needed_l390_390801

noncomputable def rate (experts : ℕ) (days : ℕ) (textbooks : ℝ) : ℝ :=
  textbooks / (experts * days)

theorem additional_experts_needed :
  let r := rate 3 24 1 in
  ∀ (x : ℕ), rate (3 + x) 18 1 = r → x = 1 :=
by
  intros r x h
  dsimp [rate] at h
  sorry

end additional_experts_needed_l390_390801


namespace total_surface_area_correct_l390_390550

def cube_volume (n : ℕ) := n * n * n
def cube_side (v : ℕ) := v^(1/3 : ℝ) -- Using real number root
def cube_surface_area (s : ℝ) := 6 * s * s

def total_surface_area_of_tower :=
  let volumes := [1, 8, 27, 64, 125]
  let sides := volumes.map (λ v, cube_side v)
  let surface_areas := sides.map (λ s, cube_surface_area s)
  let overlaps := sides.tail!.map (λ s, s * s)
  surface_areas.sum - overlaps.sum + (sides.head! * sides.head!) -- Adding non-overlapped area for bottom cube

theorem total_surface_area_correct : total_surface_area_of_tower = 276 :=
by
  sorry

end total_surface_area_correct_l390_390550


namespace total_toys_l390_390067

theorem total_toys (toys_kamari : ℕ) (toys_anais : ℕ) (h1 : toys_kamari = 65) (h2 : toys_anais = toys_kamari + 30) :
  toys_kamari + toys_anais = 160 :=
by 
  sorry

end total_toys_l390_390067


namespace sum_of_possible_a_l390_390548

noncomputable def f (a x : ℤ) := x^2 - a * x + 3 * a

theorem sum_of_possible_a : 
  (∀ a, (∃ (r s : ℤ), f a r = 0 ∧ f a s = 0) ↔ (a ∈ [24, 0, 18, -12, 21, -9, 15, -3, 12, -6])) →
  (∀ (a : ℤ), (∃ (r s : ℤ), f a r = 0 ∧ f a s = 0) → 
  (∀ (a ∈ [24, 0, 18, -12, 21, -9, 15, -3, 12, -6]).sum = 60) := by
sorry

end sum_of_possible_a_l390_390548


namespace three_person_subcommittees_count_l390_390176

theorem three_person_subcommittees_count : ∃ n k, n = 8 ∧ k = 3 ∧ nat.choose n k = 56 :=
begin
  use [8, 3],
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end three_person_subcommittees_count_l390_390176


namespace equilateral_triangle_area_l390_390053

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ := (s * s * real.sqrt 3) / 4

theorem equilateral_triangle_area :
  let s1 := 7.5
  let s2 := 9.3
  let s3 := 10.2
  let p := perimeter s1 s2 s3 in
  let side := p / 3 in
  area_of_equilateral_triangle side = (81 * real.sqrt 3) / 4 :=
by
  let s1 : ℝ := 7.5
  let s2 : ℝ := 9.3
  let s3 : ℝ := 10.2
  let p := perimeter s1 s2 s3
  let side := p / 3
  show area_of_equilateral_triangle side = (81 * real.sqrt 3) / 4
  sorry

end

end equilateral_triangle_area_l390_390053


namespace range_of_eccentricity_l390_390245

noncomputable def upperVertex (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : ℝ × ℝ := (0, b)

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem range_of_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h: a > b) :
  (∀ (x y : ℝ), ellipse a b x y → distance x y 0 b ≤ 2 * b) →
  ∃ (e : ℝ), e ∈ Set.Ioc 0 (Real.sqrt 2 / 2) ∧ a^2 = b^2 / (1 - e^2) := sorry

end range_of_eccentricity_l390_390245


namespace car_distance_ratio_l390_390848

theorem car_distance_ratio (t : ℝ) (h₁ : t > 0)
    (speed_A speed_B : ℝ)
    (h₂ : speed_A = 70)
    (h₃ : speed_B = 35)
    (ratio : ℝ)
    (h₄ : ratio = 2)
    (h_time : ∀ a b : ℝ, a * t = b * t → a = b) :
  (speed_A * t) / (speed_B * t) = ratio := by
  sorry

end car_distance_ratio_l390_390848


namespace three_person_subcommittees_l390_390170

theorem three_person_subcommittees (n k : ℕ) (h_n : n = 8) (h_k : k = 3) : nat.choose n k = 56 := by
  rw [h_n, h_k]
  norm_num
  sorry

end three_person_subcommittees_l390_390170


namespace find_x_l390_390589

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (1, 5)
noncomputable def vector_c (x : ℝ) : ℝ × ℝ := (x, 1)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_x :
  ∃ x : ℝ, collinear (2 • vector_a - vector_b) (vector_c x) ∧ x = -1 := by
  sorry

end find_x_l390_390589


namespace sqrt_of_square_neg7_l390_390845

theorem sqrt_of_square_neg7 : Real.sqrt ((-7:ℝ)^2) = 7 := by
  sorry

end sqrt_of_square_neg7_l390_390845


namespace length_second_train_approx_l390_390002

-- Define the known conditions
def length_first_train : ℕ := 270
def speed_first_train : ℕ := 120
def speed_second_train : ℕ := 80
def crossing_time : ℕ := 9

-- Define the conversion factor and relative speed calculation
def kmph_to_mps (speed_kmph : ℕ) : ℝ := (speed_kmph : ℝ) * (5 / 18)
def relative_speed_mps := kmph_to_mps (speed_first_train + speed_second_train)

-- The main statement: prove the length of the second bullet train
theorem length_second_train_approx :
  ∃ L : ℝ, L ≈ 230.04 ∧ (length_first_train + L = relative_speed_mps * crossing_time) := by
  sorry

end length_second_train_approx_l390_390002


namespace find_volume_l390_390462

-- Define points and the plane conditions
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def D : ℝ × ℝ × ℝ := (0, 1, 0)
def E : ℝ × ℝ × ℝ := (0, 0, 1)
def H : ℝ × ℝ × ℝ := (0, 1, 1)
def P : ℝ × ℝ × ℝ := (0, 0.5, 1)

-- Define the volume calculation
def volume_of_cube : ℝ := 1 
def volume_of_pyramid : ℝ := (1 / 3) * (1 / 2) * 1

theorem find_volume : volume_of_cube - volume_of_pyramid = 5 / 6 :=
by
  sorry

end find_volume_l390_390462


namespace problem1_part1_problem1_part2_l390_390144

def f (x : ℝ) : ℝ :=
  if x < -1 then
    2 * x + 3
  else if -1 <= x ∧ x <= 1 then
    x^2 + 1
  else
    1 + (1 / x)

theorem problem1_part1 : f (f (f (-2))) = 3 / 2 :=
by
  sorry

theorem problem1_part2 (m : ℝ) (h : f m = 3 / 2) : 
  m = - (Real.sqrt 2) / 2 ∨ m = 2 :=
by
  sorry

end problem1_part1_problem1_part2_l390_390144


namespace problem_statement_l390_390919

variable {m n : Type} {α β : Type}
variable [linear_ordered_field m] [linear_ordered_field n]
variable [linear_ordered_field α] [linear_ordered_field β]

-- Conditions: Define non-intersecting lines and planes
-- Note: Proper definitions will depend on precise geometric formalisms in Lean library

axiom non_intersecting_lines (m n : Type) : Prop
axiom non_intersecting_planes (α β : Type) : Prop

-- Define perpendicular and parallel relations
axiom perp {a b : Type} : Prop
axiom para {a b : Type} : Prop

-- Given conditions
variable (m_n_non_intersecting : non_intersecting_lines m n)
variable (α_β_non_intersecting : non_intersecting_planes α β)

-- Proposition ②: If m ⊥ α and m ⊥ β, then α // β
theorem problem_statement (hmα : perp m α) (hmβ : perp m β) : para α β :=
sorry

end problem_statement_l390_390919


namespace exists_coloring_function_l390_390231

open Set Function

-- Define the coloring function
noncomputable def coloring_function (r : ℝ) : Fin 10 :=
  sorry

-- Define the condition for differing in exactly one decimal place
def differs_in_one_decimal_place (a b : ℝ) : Prop :=
  ∃ n : ℤ, a ≠ b ∧ (a - b) * 10^n ∈ Finset.singleton (1 / 10) ∪ Finset.singleton (-1 / 10)

theorem exists_coloring_function : 
  ∃ c : ℝ → Fin 10, ∀ a b : ℝ, 
    differs_in_one_decimal_place a b → c a ≠ c b :=
  sorry

end exists_coloring_function_l390_390231


namespace sufficient_but_not_necessary_l390_390011

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1) → (x^2 > 1) ∧ (∃ y : ℝ, y^2 > 1 ∧ ¬ (y > 1)) :=
by
  intro h1
  have h2 : x^2 > 1 := by
    sorry
  use -2
  have h3 : (-2)^2 > 1 := by
    sorry
  have h4 : ¬(-2 > 1) := by
    sorry
  exact ⟨h2, ⟨h3, h4⟩⟩

end sufficient_but_not_necessary_l390_390011


namespace system_of_equations_solution_l390_390800

theorem system_of_equations_solution
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 + 2 * x2 + 2 * x3 + 2 * x4 + 2 * x5 = 1)
  (h2 : x1 + 3 * x2 + 4 * x3 + 4 * x4 + 4 * x5 = 2)
  (h3 : x1 + 3 * x2 + 5 * x3 + 6 * x4 + 6 * x5 = 3)
  (h4 : x1 + 3 * x2 + 5 * x3 + 7 * x4 + 8 * x5 = 4)
  (h5 : x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 = 5) :
  x1 = 1 ∧ x2 = -1 ∧ x3 = 1 ∧ x4 = -1 ∧ x5 = 1 :=
by {
  -- proof steps go here
  sorry
}

end system_of_equations_solution_l390_390800


namespace area_of_triangle_l390_390915

-- Define the ellipse and its foci
def ellipse {x y : ℝ} := (x ^ 2 / 9) + (y ^ 2 / 4) = 1

-- Define points P, F1, F2 and distances
variables {P F1 F2 : ℝ × ℝ}
variables (d1 d2 d3: ℝ)

-- Define conditions provided
def on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2
def ratio_condition (P F1 F2 : ℝ × ℝ) : Prop := (dist P F1) = 2 * (dist P F2)

-- Define foci distance condition which is known for this ellipse
def foci_distance : ℝ := 2 * real.sqrt 5

-- Define the main theorem to prove
theorem area_of_triangle (P F1 F2 : ℝ × ℝ) (h1 : on_ellipse P) (h2 : ratio_condition P F1 F2) (h3 : dist F1 F2 = foci_distance) : 
  ∃ (area : ℝ), area = 4 :=
sorry

end area_of_triangle_l390_390915


namespace projective_transformations_send_polar_to_infinity_l390_390568

def circle := S -- Circle S
def point_inside_circle := O -- Point O inside circle S
def projective_transformations_circle_center (T : Type) (S O : T) :=
  ∀ (T' : T), ∃ (S' : T), (S = S' ∧ T' = center S')

def polar (O : Type) (S : Type) : Type := PQ -- The intersection line PQ

theorem projective_transformations_send_polar_to_infinity (S : circle) (O : point_inside_circle)
  (projective_transformations_circle_center : ∀ T, ∃ S', projective_transformations_circle_center T SO)
  (polar : O → S) : ∀ T, projective_transformations_circle_center T S O → polar PQ = ∞ :=
sorry

end projective_transformations_send_polar_to_infinity_l390_390568


namespace eccentricity_of_ellipse_l390_390142

theorem eccentricity_of_ellipse {b : ℝ} (hb : 0 < b ∧ b < 2) :
  let e := Real.sqrt (4 - b^2) / 2 in
  (∃ (x₀ y₀ : ℝ), (x₀^2 / 4 + y₀^2 / b^2 = 1 ∧ x₀^2 / 2 - y₀^2 = 1 ∧
    (- (x₀ * b) / (4 * y₀)) * (x₀ / (2 * y₀)) = -1)) →
  e = Real.sqrt 3 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l390_390142


namespace most_frequent_day_2014_l390_390689

/-- Given that March 9, 2014, is a Sunday, prove that Wednesday is the day that occurs most frequently in the year 2014. -/
theorem most_frequent_day_2014 
  (h: nat.mod (5 + 68) 7 = 0):
  (most_frequent_day 2014) = "Wednesday" :=
sorry

end most_frequent_day_2014_l390_390689


namespace maximum_distance_travel_l390_390888

theorem maximum_distance_travel (front_tire_lifespan rear_tire_lifespan : ℕ) 
  (h1 : front_tire_lifespan = 20000) 
  (h2 : rear_tire_lifespan = 30000) : 
  ∃ max_distance, max_distance = 24000 :=
by {
  existsi 24000,
  rw [h1, h2],
  sorry
}

end maximum_distance_travel_l390_390888


namespace number_of_whole_numbers_between_sqrts_l390_390184

noncomputable def count_whole_numbers_between_sqrts : ℕ :=
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let min_int := Int.ceil lower_bound
  let max_int := Int.floor upper_bound
  Int.natAbs (max_int - min_int + 1)

theorem number_of_whole_numbers_between_sqrts :
  count_whole_numbers_between_sqrts = 7 :=
by
  sorry

end number_of_whole_numbers_between_sqrts_l390_390184


namespace series_proof_l390_390679

theorem series_proof (a b : ℝ) (h : (∑' n : ℕ, (-1)^n * a / b^(n+1)) = 6) : 
  (∑' n : ℕ, (-1)^n * a / (a - b)^(n+1)) = 6 / 7 := 
sorry

end series_proof_l390_390679


namespace hike_and_cycle_distance_l390_390970

theorem hike_and_cycle_distance : ∀ (hike_time_min cycle_time_min : ℕ) (hike_rate cycle_rate : ℕ), 
  hike_time_min = 60 → 
  hike_rate = 3 → 
  cycle_time_min = 45 → 
  cycle_rate = 12 → 
  (hike_rate * (hike_time_min / 60) + cycle_rate * (cycle_time_min / 60) = 12) :=
begin
  intros hike_time_min cycle_time_min hike_rate cycle_rate,
  assume h1 h2 c1 c2,
  rw [h2, h1],
  rw [c2, c1],
  norm_num,
end

end hike_and_cycle_distance_l390_390970


namespace min_value_proof_l390_390686

noncomputable def min_value_of_expression (a b c d e f g h : ℝ) : ℝ :=
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2

theorem min_value_proof (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16) :
  ∃ (x : ℝ), x = 32 ∧ min_value_of_expression a b c d e f g h = x :=
by
  use 32
  sorry

end min_value_proof_l390_390686


namespace work_completion_days_l390_390461

theorem work_completion_days (A_days : ℕ) (B_days : ℕ) (C_days : ℕ) (AB_work_days : ℕ) : ℕ :=
  let work_rate_A := (1 : ℚ) / A_days
  let work_rate_B := (1 : ℚ) / B_days
  let work_rate_C := (1 : ℚ) / C_days
  let combined_AB_work_rate := work_rate_A + work_rate_B
  let work_done_AB := combined_AB_work_rate * AB_work_days
  let remaining_work := 1 - work_done_AB
  let combined_BC_work_rate := work_rate_B + work_rate_C
  remaining_work / combined_BC_work_rate

example : work_completion_days 20 12 15 3 = 4 := by sorry

end work_completion_days_l390_390461


namespace best_method_to_understand_security_l390_390222

def Troops := {A, B, C, D}

-- Define the property of being the "best method" to understand the security capabilities
def best_method (method : String) : Prop :=
  method = "Stratified sampling method"

-- There are four troops with differing security capabilities
axiom four_troops_with_differences : ∃ (troops : Set Troops), troops = {A, B, C, D} ∧ 
                                        ∀ troop1 troop2 ∈ troops, troop1 ≠ troop2 → capability(troop1) ≠ capability(troop2)

-- The best method to understand the security capabilities is the method that acknowledges these differences
theorem best_method_to_understand_security : best_method "Stratified sampling method" :=
by
  sorry

end best_method_to_understand_security_l390_390222


namespace eccentricity_range_l390_390270

noncomputable def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  { z : ℝ × ℝ // (z.1^2 / a^2) + (z.2^2 / b^2) = 1 }

theorem eccentricity_range (a b : ℝ) (h : a > b) (hb : b > 0):
  ∀ (P : ellipse a b h hb),
    |(P.1, P.2 - b).norm ≤ 2 * b → 
    eccentricity (a b) ∈ set.Ici (real.sqrt 2 / 2) :=
sorry

end eccentricity_range_l390_390270


namespace rectangle_area_ratio_is_three_l390_390220

variables {a b : ℝ}

-- Rectangle ABCD with midpoint F on CD, BC = 3 * BE
def rectangle_midpoint_condition (CD_length : ℝ) (BC_length : ℝ) (BE_length : ℝ) (F_midpoint : Prop) :=
  F_midpoint ∧ BC_length = 3 * BE_length

-- Areas and the ratio
def area_rectangle (CD_length BC_length : ℝ) : ℝ :=
  CD_length * BC_length

def area_shaded (a b : ℝ) : ℝ :=
  2 * a * b

theorem rectangle_area_ratio_is_three (h : rectangle_midpoint_condition (2 * a) (3 * b) b (F_midpoint := True)) :
  area_rectangle (2 * a) (3 * b) = 3 * area_shaded a b :=
by
  unfold rectangle_midpoint_condition at h
  unfold area_rectangle area_shaded
  rw [←mul_assoc, ←mul_assoc]
  sorry

end rectangle_area_ratio_is_three_l390_390220


namespace find_kids_l390_390502

theorem find_kids (A K : ℕ) (h1 : A + K = 12) (h2 : 3 * A = 15) : K = 7 :=
sorry

end find_kids_l390_390502


namespace midpoint_product_l390_390578

structure Point where
  x : ℝ
  y : ℝ

def midpoint (A B : Point) : Point :=
  ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

theorem midpoint_product {A B M : Point} (hA : A = ⟨5, 6⟩) (hM : M = ⟨3, 8⟩) 
  (hMid : M = midpoint A B) : 
  (B.x * B.y) = 10 :=
sorry

end midpoint_product_l390_390578


namespace my_function_is_avg_l390_390913

noncomputable def my_function : list ℝ → ℝ
| []       := 0
| (x :: xs) := sorry

axiom perm_invariant (X : list ℝ) (hX : X ~ X.perm) : 
  my_function X = my_function X.perm

axiom add_alpha (X : list ℝ) (α : ℝ) : 
  my_function (X.map (λ x, x + α)) = my_function X + α

axiom negation (X : list ℝ) : 
  my_function (X.map (λ x, -x)) = -my_function X

axiom yn_eq_xi (X : list ℝ) (y : ℝ) (h : ∀ x ∈ X, y = my_function X) : 
  my_function (list.cons y (list.map y X)) = my_function X

theorem my_function_is_avg {X : list ℝ} (hX : X ≠ []) : 
  my_function X = list.sum X / list.length X := sorry

end my_function_is_avg_l390_390913


namespace range_of_eccentricity_l390_390303

-- Definition of the ellipse and its properties
namespace EllipseProof

variables {a b : ℝ} (h : a > b ∧ b > 0)

def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def upper_vertex : ℝ × ℝ := (0, b)

def is_on_ellipse (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

def condition_on_point (P : ℝ × ℝ) : Prop := dist P upper_vertex ≤ 2 * b

-- The eccentricity of the ellipse
def eccentricity : ℝ := real.sqrt (1 - (b^2 / a^2))

-- The required proof statement
theorem range_of_eccentricity :
  (∀ P, is_on_ellipse P → condition_on_point P) →
  eccentricity ≤ real.sqrt 2 / 2 := sorry

end EllipseProof

end range_of_eccentricity_l390_390303


namespace riding_is_four_times_walking_l390_390486

variable (D : ℝ) -- Total distance of the route
variable (v_r v_w : ℝ) -- Riding speed and walking speed
variable (t_r t_w : ℝ) -- Time spent riding and walking

-- Conditions given in the problem
axiom distance_riding : (2/3) * D = v_r * t_r
axiom distance_walking : (1/3) * D = v_w * t_w
axiom time_relation : t_w = 2 * t_r

-- Desired statement to prove
theorem riding_is_four_times_walking : v_r = 4 * v_w := by
  sorry

end riding_is_four_times_walking_l390_390486


namespace number_of_integer_solutions_l390_390628

theorem number_of_integer_solutions
    (a : ℤ)
    (x : ℤ)
    (h1 : ∃ x : ℤ, (1 - a) / (x - 2) + 2 = 1 / (2 - x))
    (h2 : ∀ x : ℤ, 4 * x ≥ 3 * (x - 1) ∧ x + (2 * x - 1) / 2 < (a - 1) / 2) :
    (a = 4) :=
sorry

end number_of_integer_solutions_l390_390628


namespace height_of_circular_segment_l390_390542

theorem height_of_circular_segment (d a : ℝ) (h : ℝ) :
  (h = (d - Real.sqrt (d^2 - a^2)) / 2) ↔ 
  ((a / 2)^2 + (d / 2 - h)^2 = (d / 2)^2) :=
sorry

end height_of_circular_segment_l390_390542


namespace one_is_average_of_others_l390_390131

theorem one_is_average_of_others (s : Finset ℤ) (h_card_s : s.card = 10) 
  (h_distinct_diffs : (s.powerset.filter (λ t, t.card = 2)).image (λ t, abs (t.val.sum)).card = 44) :
  ∃ x y z ∈ s, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ y = (x + z) / 2 :=
begin
  sorry
end

end one_is_average_of_others_l390_390131


namespace exists_n_geq_k_l390_390392

theorem exists_n_geq_k (a : ℕ → ℕ) (h_distinct : ∀ i j : ℕ, i ≠ j → a i ≠ a j) 
    (h_positive : ∀ i : ℕ, a i > 0) :
    ∀ k : ℕ, ∃ n : ℕ, n > k ∧ a n ≥ n :=
by
  intros k
  sorry

end exists_n_geq_k_l390_390392


namespace length_of_A_l390_390336

-- Define the points A, B, and C
def A : ℝ × ℝ := (0, 7)
def B : ℝ × ℝ := (0, 14)
def C : ℝ × ℝ := (3, 5)

-- Define the condition that points A' and B' lie on y = x
def lies_on_y_eq_x (p : ℝ × ℝ) : Prop := p.1 = p.2

-- Define A' and B' using their specific coordinates found in the solution
def A' : ℝ × ℝ := (21/5, 21/5)
def B' : ℝ × ℝ := (7/2, 7/2)

-- The distance formula for two points in ℝ²
def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- The theorem to prove
theorem length_of_A'B' : distance A' B' = (3 * Real.sqrt 2) / 10 :=
by
  sorry

end length_of_A_l390_390336


namespace eccentricity_range_l390_390286

noncomputable def ellipse_eccentricity (a b : ℝ) (h_ab : a > b) : ℝ := (Real.sqrt (a ^ 2 - b ^ 2)) / a

theorem eccentricity_range (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ) :
  (∃ x y : ℝ, (x, y) = P ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  (Real.sqrt (x^2 + (y - b)^2) ≤ 2 * b)) → 
  0 < ellipse_eccentricity a b h_ab ∧ ellipse_eccentricity a b h_ab ≤ Real.sqrt 2 / 2 :=
begin
  sorry
end

end eccentricity_range_l390_390286


namespace eccentricity_range_l390_390297

-- Ellipse definition
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Upper vertex B of the ellipse
def B (b : ℝ) : ℝ × ℝ := (0, b)

-- Distance PB condition
def distance_le_2b (a b x0 y0 : ℝ) (h : ellipse a b (and.intro (sorry) (sorry)) x0 y0) : Prop :=
  (x0 - 0)^2 + (y0 - b)^2 ≤ (2 * b)^2

-- Range of eccentricity
def eccentricity (a b c : ℝ) : ℝ :=
  c / a

-- Main theorem
theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  (forall (x0 y0 : ℝ), ellipse a b h x0 y0 → distance_le_2b a b x0 y0 h) →
  eccentricity a b (sqrt (a^2 - b^2)) ∈ set.Icc 0 (sqrt 2 / 2) :=
by
  sorry

end eccentricity_range_l390_390297


namespace ratio_giraffe_to_zebra_enclosures_l390_390058

theorem ratio_giraffe_to_zebra_enclosures
  (tiger_enclosures : ℕ)
  (tigers_per_enclosure : ℕ)
  (zebra_enclosures_per_tiger : ℕ)
  (zebras_per_enclosure : ℕ)
  (giraffes_per_enclosure : ℕ)
  (total_animals : ℕ)
  (ht : tiger_enclosures = 4)
  (htpe : tigers_per_enclosure = 4)
  (hz : zebra_enclosures_per_tiger = 2)
  (hzpe : zebras_per_enclosure = 10)
  (hge : giraffes_per_enclosure = 2)
  (hta : total_animals = 144) :
  (24 / 8) = (3 : ℕ) :=
by
  let total_tigers := tiger_enclosures * tigers_per_enclosure
  let tiger_zebra_enclosures := tiger_enclosures * zebra_enclosures_per_tiger
  let total_zebras := tiger_zebra_enclosures * zebras_per_enclosure
  let total_tiger_zebra_animals := total_tigers + total_zebras
  let remaining_animals := total_animals - total_tiger_zebra_animals
  let giraffe_enclosures := remaining_animals / giraffes_per_enclosure
  have h1 : tiger_enclosures = 4, from ht
  have h2 : tiger_zebra_enclosures = 8, from htpe ▸ hz ▸ congr_arg (λ tiger_enclosures, tiger_enclosures * 2) h1
  have h3 : total_zebras = 80, from hzpe ▸ congr_arg (λ tiger_zebra_enclosures, tiger_zebra_enclosures * 10) h2
  have h4 : total_tigers = 16, from htpe ▸ congr_arg (λ tiger_enclosures, tiger_enclosures * 4) h1
  have h5 : total_tiger_zebra_animals = 96, from congr_arg2 Nat.add h4 h3
  have h6 : remaining_animals = 48, from hta ▸ congr_arg (λ total_animals, total_animals - 96) (refl 144)
  have h7 : giraffe_enclosures = 24, from hge ▸ congr_arg (λ remaining_animals, remaining_animals / 2) h6
  have ratio := 24 / 8
  exact eq_of_sub_eq_zero (Nat.sub_self (show (24 / 8) = 3, from sorry))

end ratio_giraffe_to_zebra_enclosures_l390_390058


namespace at_least_three_equal_l390_390111

theorem at_least_three_equal (a b c d : ℕ) (h1 : (a + b) ^ 2 ∣ c * d)
                                (h2 : (a + c) ^ 2 ∣ b * d)
                                (h3 : (a + d) ^ 2 ∣ b * c)
                                (h4 : (b + c) ^ 2 ∣ a * d)
                                (h5 : (b + d) ^ 2 ∣ a * c)
                                (h6 : (c + d) ^ 2 ∣ a * b) :
  ∃ x : ℕ, (x = a ∧ x = b ∧ x = c) ∨ (x = a ∧ x = b ∧ x = d) ∨ (x = a ∧ x = c ∧ x = d) ∨ (x = b ∧ x = c ∧ x = d) :=
sorry

end at_least_three_equal_l390_390111


namespace junior_score_l390_390047

-- Define the base conditions: percentages and average scores.
def percent_juniors := 0.20
def percent_seniors := 0.80
def average_score := 78
def average_score_seniors := 75

-- Define the total score calculation from the average.
def total_score (students : ℕ) := students * average_score
def total_score_seniors (seniors : ℕ) := seniors * average_score_seniors

-- Given these conditions, prove the score of each junior.
theorem junior_score
    (students : ℕ)
    (juniors seniors : ℕ)
    (h1 : juniors = percent_juniors * students)
    (h2 : seniors = percent_seniors * students)
    (h3 : total_score students = total_score_seniors seniors + juniors * (average_score_seniors + 15)) :
    juniors * (average_score_seniors + 15) / juniors = 90 :=
by
  sorry

end junior_score_l390_390047


namespace students_per_group_l390_390762

theorem students_per_group (total_students not_picked groups : ℕ) 
    (h1 : total_students = 64) 
    (h2 : not_picked = 36) 
    (h3 : groups = 4) : (total_students - not_picked) / groups = 7 :=
by
  sorry

end students_per_group_l390_390762


namespace f_induction_l390_390773

noncomputable def f : ℕ → ℝ
| 0       := 0
| (n + 1) := f n + 1 / (2^n - 1 : ℝ)

theorem f_induction (k : ℕ) :
  f (k + 1) = f k + 1 / 2^k + 1 / (2^k + 1) + ... + 1 / (2^(k+1) - 1) :=
sorry

end f_induction_l390_390773


namespace pq_eq_qr_iff_ratios_equal_l390_390685

structure CyclicQuadrilateral (A B C D P Q R K L : Type) :=
  (cyclic : Cyclic A B C D)
  (perpendicular_PD_BC : Perpendicular D P C)
  (perpendicular_PD_CA : Perpendicular D Q A)
  (perpendicular_PD_AB : Perpendicular D R B)
  (bisector_ABC_AC : Bisector (Angle A B C) K A C)
  (bisector_ADC_AC : Bisector (Angle A D C) L A C)
  (ratio1 : (Length A B) / (Length B C) = (Length A K) / (Length K C))
  (ratio2 : (Length A D) / (Length D C) = (Length A L) / (Length L C))

theorem pq_eq_qr_iff_ratios_equal (A B C D P Q R K L : Type) 
  [CyclicQuadrilateral A B C D P Q R K L] :
  (Distance P Q = Distance Q R) ↔ 
  ((Length A B) / (Length B C) = (Length A D) / (Length D C)) :=
by 
  sorry

end pq_eq_qr_iff_ratios_equal_l390_390685


namespace eccentricity_range_l390_390311

variables {a b : ℝ}
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

theorem eccentricity_range (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ (x y : ℝ), ellipse x y → distance x y 0 b ≤ 2 * b) :
  0 < (sqrt (a^2 - b^2) / a) ∧ (sqrt (a^2 - b^2) / a) ≤ sqrt 2 / 2 :=
sorry

end eccentricity_range_l390_390311


namespace two_digit_number_property_l390_390055

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ (k : ℕ), k * k = n

theorem two_digit_number_property :
  ∃ (a b : ℕ),
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    (∃ (k : ℕ), is_perfect_square (11 * (a + b))) ∧
    ((11 * (a + b)) = k * k) ∧
    ((∃ (m : ℕ), m = 1 ∧ 11 * m = a + b) ∧
    10 * a + b ∈ {29, 38, 47, 56, 65, 74, 83, 92}) :=
begin
  sorry
end

end two_digit_number_property_l390_390055


namespace steve_speed_ratio_l390_390743

/-- Define the distance from Steve's house to work. -/
def distance_to_work := 30

/-- Define the total time spent on the road by Steve. -/
def total_time_on_road := 6

/-- Define Steve's speed on the way back from work. -/
def speed_back := 15

/-- Calculate the ratio of Steve's speed on the way back to his speed on the way to work. -/
theorem steve_speed_ratio (v : ℝ) (h_v_pos : v > 0) 
    (h1 : distance_to_work / v + distance_to_work / speed_back = total_time_on_road) :
    speed_back / v = 2 := 
by
  -- We will provide the proof here
  sorry

end steve_speed_ratio_l390_390743


namespace proof_l390_390940

noncomputable def M : Set ℝ := {x | 1 - (2 / x) > 0}
noncomputable def N : Set ℝ := {x | x ≥ 1}

theorem proof : (Mᶜ ∪ N) = {x | x ≥ 0} := sorry

end proof_l390_390940


namespace distinct_selection_plans_l390_390366

/-
Given:
1. There are 6 individuals.
2. We need to select 4 of them to visit 4 different cities: Paris, London, Sydney, and Moscow.
3. Each person can visit only one city.
4. Persons A and B cannot visit Paris.

Prove:
The total number of distinct selection plans is 240.
-/

def numIndividuals : ℕ := 6
def individuals : Finset ℕ := Finset.range numIndividuals
def numCities : ℕ := 4
def cities : Fin₄ := sorry /- Paris, London, Sydney, Moscow -/

def A : ℕ := 0 /- Let's assume individual A is 0th person -/
def B : ℕ := 1 /- Let's assume individual B is 1st person -/
def Paris : Fin₄ := 0 /- Let's assume Paris is 0th city -/

def validPermutationsCount : ℕ := 240

theorem distinct_selection_plans : 
  ∃ (s : Finset (Finset (ℕ × Fin₄))), s.card = validPermutationsCount ∧ 
  (∀ p ∈ s, (p.filter (λ x, x.2 = Paris)).filter (λ x, x.1 = A) = ∅ ∧
            (p.filter (λ x, x.2 = Paris)).filter (λ x, x.1 = B) = ∅) :=
  sorry

end distinct_selection_plans_l390_390366


namespace smallest_prime_less_than_perf_square_l390_390420

-- Define a predicate for being a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

-- The main goal
theorem smallest_prime_less_than_perf_square : ∃ n : ℕ, is_prime n ∧ ∃ m : ℕ, n = m^2 - 8 ∧ (∀ k : ℕ, is_prime k ∧ ∃ l : ℕ, k = l^2 - 8 → k ≥ n) :=
begin
  use 17,
  split,
  -- Proof that 17 is a prime number
  {
    unfold is_prime,
    split,
    { exact dec_trivial },
    { intros d hd,
      have h_d : d = 1 ∨ d = 17,
      { cases d,
        { exfalso, linarith, },
        { cases d,
          { left, refl, },
          { right, linarith [Nat.Prime.not_dvd_one 17 hd], }, }, },
      exact h_d, },
  },
  -- Proof that 17 is 8 less than a perfect square and the smallest such prime
  {
    use 5,
    split,
    { refl, },
    { intros k hk,
      cases hk with hk_prime hk_cond,
      cases hk_cond with l hl,
      rw hl,
      have : l ≥ 5,
      { intros,
        linarith, },
      exact this, },
  }
end

end smallest_prime_less_than_perf_square_l390_390420


namespace num_5_letter_words_with_at_least_two_consonants_l390_390610

open Finset

def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def total_5_letter_words : ℕ := 6^5
def words_with_0_consonants : ℕ := 2^5
def words_with_1_consonant : ℕ := 5 * 4 * 2^4

theorem num_5_letter_words_with_at_least_two_consonants : 
  total_5_letter_words - (words_with_0_consonants + words_with_1_consonant) = 7424 := by
  sorry

end num_5_letter_words_with_at_least_two_consonants_l390_390610


namespace derivative_cycle_l390_390563

-- Define the initial function and its derivatives
def f : ℕ → (ℝ → ℝ)
| 0       := fun x => Real.sin x
| (n + 1) := fun x => (f n)' x

-- The theorem to prove
theorem derivative_cycle :
  (f 2005) = (fun x => Real.cos x) := by
  sorry

end derivative_cycle_l390_390563


namespace sum_of_three_numbers_l390_390756

theorem sum_of_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : ab + bc + ca = 131) : 
  a + b + c = 22 := 
by 
  have h3 : (a + b + c)^2 = 222 + 2 * 131,
  { 
    simp [h1, h2],
    ring,
  },
  have h4 : (a + b + c)^2 = 484 := by simp [h3],
  have h5 : a + b + c = real.sqrt 484 := by { rw ← h4, exact_real_root rfl }, 
  simp [real.sqrt_eq_rfl],
  linarith

end sum_of_three_numbers_l390_390756


namespace irreducible_polynomial_exists_prime_l390_390010

/-- Prove that there must exist a prime number p such that the polynomial 
    f(x) = ±p + a₁x + a₂x² + ... + aₙxⁿ, where aₙ ≠ 0, cannot be factored 
    into a product of integer coefficient polynomials that are not constants. -/
theorem irreducible_polynomial_exists_prime 
  (a : ℕ → ℤ) (n : ℕ) (a_n_nonzero : a n ≠ 0) :
  ∃ p : ℕ, prime p ∧ irreducible (λ x, (if (n % 2 = 0) then p else -p) + (∑ i in finset.range n, a i * x ^ i) + a n * x ^ n) :=
sorry

end irreducible_polynomial_exists_prime_l390_390010


namespace range_of_eccentricity_l390_390272

variables {a b c e : ℝ}

def ellipse (x y : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)
def upper_vertex := b > 0
def distance_condition (x_0 y_0 : ℝ) := (x_0^2 + (y_0 - b)^2 ≤ (2 * b) ^ 2)
def eccentricity := e = c / a
def ellipse_condition := a = real.sqrt (b^2 + c^2)

theorem range_of_eccentricity (h1 : a > b) (h2 : upper_vertex) 
  (h3 : ∀ x_0 y_0, ellipse x_0 y_0 → distance_condition x_0 y_0)
  (h4 : ellipse_condition) :
  0 < e ∧ e ≤ real.sqrt 2 / 2 := 
by
  sorry

end range_of_eccentricity_l390_390272


namespace chessboard_adjacent_diff_at_least_n_l390_390217

theorem chessboard_adjacent_diff_at_least_n (n : ℕ) (h : 2 ≤ n) (b : Finₓ n × Finₓ n → ℕ) (h_unique : ∀ i j, i ≠ j → b i ≠ b j) :
  ∃ (i j : Finₓ n × Finₓ n), (i.1 = j.1 ∧ (i.2.succ = j.2 ∨ j.2.succ = i.2) ∨ i.2 = j.2 ∧ (i.1.succ = j.1 ∨ j.1.succ = i.1)) ∧
  n ≤ abs (b i - b j) :=
sorry

end chessboard_adjacent_diff_at_least_n_l390_390217


namespace range_of_eccentricity_l390_390275

variables {a b c e : ℝ}

def ellipse (x y : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)
def upper_vertex := b > 0
def distance_condition (x_0 y_0 : ℝ) := (x_0^2 + (y_0 - b)^2 ≤ (2 * b) ^ 2)
def eccentricity := e = c / a
def ellipse_condition := a = real.sqrt (b^2 + c^2)

theorem range_of_eccentricity (h1 : a > b) (h2 : upper_vertex) 
  (h3 : ∀ x_0 y_0, ellipse x_0 y_0 → distance_condition x_0 y_0)
  (h4 : ellipse_condition) :
  0 < e ∧ e ≤ real.sqrt 2 / 2 := 
by
  sorry

end range_of_eccentricity_l390_390275


namespace percent_exceeding_speed_limit_l390_390697

theorem percent_exceeding_speed_limit (N : ℕ) (hn : N > 0) (T : ℕ) (E : ℕ)
  (hT : T = (40 * N) / 100)
  (hE : E = (T * 100) / 80):
  (E * 100) / N = 50 :=
by
  have h20percent : (20 * E) / 100 = E - T :=
    sorry
  rw [hT, ←h20percent, hE]
  ring
  sorry

end percent_exceeding_speed_limit_l390_390697


namespace area_of_triangle_BQW_l390_390226

-- Definitions corresponding to the conditions provided
def is_rectangle (A B C D : ℝ) : Prop := -- Assuming the coordinates defined properly
by sorry

def segment_length (P Q : ℝ) (len : ℝ) : Prop := 
by sorry

def trapezoid_area (A B C D Z W : ℝ) (area : ℝ) : Prop := 
by sorry

def midpoint (Q Z W : ℝ) : Prop := 
by sorry

-- Theorem statement
theorem area_of_triangle_BQW 
  (A B C D Z W Q : ℝ)
  (h_rect : is_rectangle A B C D)
  (h_AZ : segment_length A Z 4)
  (h_WC : segment_length W C 4)
  (h_AB : segment_length A B 10)
  (h_trap : trapezoid_area Z W C D 70)
  (h_mid : midpoint Q Z W) :
  true := 
by sorry

end area_of_triangle_BQW_l390_390226


namespace expression_evaluation_l390_390846

theorem expression_evaluation : |1 - Real.sqrt 3| + 2 * Real.cos (Real.pi / 6) - Real.sqrt 12 - 2023 = -2024 := 
by {
    sorry
}

end expression_evaluation_l390_390846


namespace problem_l390_390044

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n:ℕ, a (n+1) = 2 * (∑ k in finset.range n, a k) + 1

noncomputable def bn (a : ℕ → ℕ) (n : ℕ) := 2 * n * a n

noncomputable def Tn (b : ℕ → ℕ) (n : ℕ) := ∑ k in finset.range n, b k

theorem problem
  : ∀ a, sequence a →
    (∀ n, a n = 3^(n-1)) ∧
    (∀ n, Tn (bn a) n = (n - 1/2) * 3^n + 1/2) :=
by
  intros a h_seq
  have h_a : ∀ n, a n = 3^(n-1), sorry
  split
  { exact h_a }
  { intro n
    have h_b : ∀ k, bn a k = 2 * k * (3^(k-1)), sorry
    have h_T : Tn (bn a) n = (n - 1/2) * 3^n + 1/2, sorry
    exact h_T }


end problem_l390_390044


namespace range_of_eccentricity_of_ellipse_l390_390255

-- Definitions for the problem.
def is_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1) 

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

def on_upper_vertex (b : ℝ) : (ℝ × ℝ) := (0, b)

-- The statement to prove the range of eccentricity.
theorem range_of_eccentricity_of_ellipse (a b : ℝ) (x y : ℝ) (e : ℝ) :
  is_on_ellipse a b x y →
  (distance x y 0 b ≤ 2 * b) →
  ∃ e, (e = (1 - (b^2 / a^2))^(1/2) ∧ (0 < e) ∧ (e ≤ (sqrt 2)/2)) :=
by
  sorry

end range_of_eccentricity_of_ellipse_l390_390255


namespace value_of_expression_l390_390620

variable (m : ℝ)

theorem value_of_expression (h : 2 * m^2 + 3 * m - 1 = 0) : 
  4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end value_of_expression_l390_390620


namespace truncated_pyramid_volume_l390_390748

theorem truncated_pyramid_volume (a b : ℝ) (h : a > b) :
  let V := (sqrt 3 / 12) * (a^3 - b^3) in
  V = (sqrt 3 / 12) * (a^3 - b^3) :=
by
  sorry

end truncated_pyramid_volume_l390_390748


namespace jordan_run_time_l390_390661

theorem jordan_run_time (jordan_time_per_mile: ℝ) (steve_time_per_mile: ℝ) (steve_time_6_miles: ℝ):
  steve_time_6_miles = 36 →
  steve_time_per_mile = steve_time_6_miles / 6 →
  jordan_time_per_mile = (steve_time_6_miles / 3) / 4 →
  jordan_time_per_mile * 8 = 24 :=
by
  intros h1 h2 h3
  rw [h3]
  rw [h2]
  rw [h1]
  norm_num
  sorry

end jordan_run_time_l390_390661


namespace triangle_isosceles_or_right_angled_l390_390959

theorem triangle_isosceles_or_right_angled
  (β γ : ℝ)
  (h : Real.tan β * Real.sin γ ^ 2 = Real.tan γ * Real.sin β ^ 2) :
  (β = γ ∨ β + γ = Real.pi / 2) :=
sorry

end triangle_isosceles_or_right_angled_l390_390959


namespace cos_minus_sin_l390_390133

theorem cos_minus_sin (x : ℝ) (h1 : sin (2 * x) = 3 / 4) (h2 : x ∈ Ioo (π / 4) (π / 2)) :
  cos x - sin x = -1 / 2 :=
sorry

end cos_minus_sin_l390_390133


namespace trig_eq_solutions_l390_390369

open Real

theorem trig_eq_solutions (k : ℤ) (x : ℝ) : 
    (sin^4 x + sin^4 (2*x) + sin^4 (3*x) = cos^4 x + cos^4 (2*x) + cos^4 (3*x)) → 
    (∃ n : ℤ, x = (2 * n + 1) * (π / 8) ∨ x = (2 * n + 1) * (π / 4)) :=
by
  intro h
  sorry

end trig_eq_solutions_l390_390369


namespace eccentricity_range_l390_390294

-- Ellipse definition
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Upper vertex B of the ellipse
def B (b : ℝ) : ℝ × ℝ := (0, b)

-- Distance PB condition
def distance_le_2b (a b x0 y0 : ℝ) (h : ellipse a b (and.intro (sorry) (sorry)) x0 y0) : Prop :=
  (x0 - 0)^2 + (y0 - b)^2 ≤ (2 * b)^2

-- Range of eccentricity
def eccentricity (a b c : ℝ) : ℝ :=
  c / a

-- Main theorem
theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  (forall (x0 y0 : ℝ), ellipse a b h x0 y0 → distance_le_2b a b x0 y0 h) →
  eccentricity a b (sqrt (a^2 - b^2)) ∈ set.Icc 0 (sqrt 2 / 2) :=
by
  sorry

end eccentricity_range_l390_390294


namespace smallest_three_digit_palindromic_prime_with_hundreds_and_ones_digits_equal_to_2_l390_390876

def is_palindromic (n : ℕ) : Prop := 
  let s := n.repr
  s = s.reverse

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem smallest_three_digit_palindromic_prime_with_hundreds_and_ones_digits_equal_to_2 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ is_palindromic n ∧
  (n / 100 = 2 ∧ n % 10 = 2) ∧ is_prime n ∧ 
  ∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ is_palindromic m ∧
  (m / 100 = 2 ∧ m % 10 = 2) ∧ is_prime m) → n ≤ m :=
  ∃ n : ℕ, n = 232 := 
begin
  sorry
end

end smallest_three_digit_palindromic_prime_with_hundreds_and_ones_digits_equal_to_2_l390_390876


namespace handout_distribution_count_l390_390531

noncomputable def num_handout_distributions (n : ℕ) (k : ℕ) : ℕ :=
  if h : (15 = n) ∧ (6 = k) then 125 else 0

theorem handout_distribution_count :
  num_handout_distributions 15 6 = 125 :=
by
  unfold num_handout_distributions
  simp
  sorry

end handout_distribution_count_l390_390531


namespace tenth_term_is_correct_l390_390435

-- Define the first term and common difference for the sequence
def a1 : ℚ := 1 / 2
def d : ℚ := 1 / 3

-- The property that defines the n-th term of the arithmetic sequence
def a (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Statement to prove that the tenth term in the arithmetic sequence is 7 / 2
theorem tenth_term_is_correct : a 10 = 7 / 2 := 
by 
  -- To be filled in with the proof later
  sorry

end tenth_term_is_correct_l390_390435


namespace quadratic_inequality_l390_390199

-- Defining the quadratic expression
def quadratic_expr (a x : ℝ) : ℝ :=
  (a + 2) * x^2 + 2 * (a + 2) * x + 4

-- Statement to be proven
theorem quadratic_inequality {a : ℝ} :
  (∀ x : ℝ, quadratic_expr a x > 0) ↔ -2 ≤ a ∧ a < 2 :=
by
  sorry -- Proof omitted

end quadratic_inequality_l390_390199


namespace James_gold_bars_l390_390657

theorem James_gold_bars (P : ℝ) (h_condition1 : 60 - P / 100 * 60 = 54) : P = 10 := 
  sorry

end James_gold_bars_l390_390657


namespace percent_not_red_balls_l390_390496

theorem percent_not_red_balls (percent_cubes percent_red_balls : ℝ) 
  (h1 : percent_cubes = 0.3) (h2 : percent_red_balls = 0.25) : 
  (1 - percent_red_balls) * (1 - percent_cubes) = 0.525 :=
by
  sorry

end percent_not_red_balls_l390_390496


namespace angle_BAD_is_80_l390_390362

variable {P : Type} [MetricSpace P]

/-- a structure for quadrilateral with specified sides and angles -/
structure Quadrilateral (P : Type) [MetricSpace P] :=
(A B C D : P)
(AB_BC_CD_eq : dist A B = dist B C ∧ dist B C = dist C D)
(angle_ABC_eq_80 : ∠ B A C = 80)
(angle_BCD_eq_160 : ∠ B C D = 160)

/-- statement to calculate the angle BAD in the quadrilateral ABCD -/
theorem angle_BAD_is_80 {Q : Quadrilateral P} : ∠ B A D = 80 :=
sorry

end angle_BAD_is_80_l390_390362


namespace no_integer_solution_l390_390522

theorem no_integer_solution : 
  (∀ (a b : ℤ), 3^a + 3^b = 17 → ∀ (c d : ℤ), 4^c + 4^d ≠ 17) :=
sorry

end no_integer_solution_l390_390522


namespace seq_alternates_l390_390668

-- Define the sequence (x_n)
noncomputable def seq (x : ℕ) : ℕ → ℕ
| 0       := x
| (n + 1) := some (nat.min_fac (seq n + 1))

-- Assuming x1 is a six-digit number
def six_digit (x : ℕ) : Prop := 100000 ≤ x ∧ x < 1000000

theorem seq_alternates {x : ℕ} (hx : six_digit x) :
  (x = 2 ∨ x = 3) ∨ (seq x 19 + seq x 20 = 5) := sorry

end seq_alternates_l390_390668


namespace david_telephone_numbers_count_l390_390832

-- Define a function to count the number of telephone numbers
def telephone_numbers : ℕ :=
  Nat.choose 8 6

-- The theorem stating the number of such telephone numbers is 28
theorem david_telephone_numbers_count : telephone_numbers = 28 :=
by
  -- Simplify the combination calculation
  unfold telephone_numbers
  simp
  sorry

end david_telephone_numbers_count_l390_390832


namespace sin_of_sum_of_angles_l390_390956

noncomputable def sin_theta_plus_phi (θ φ : ℝ) : ℝ :=
  real.sin (θ + φ)

theorem sin_of_sum_of_angles 
  (θ φ : ℝ)
  (h1: complex.exp (complex.I * θ) = (4 / 5) + (3 / 5) * complex.I)
  (h2: complex.exp (complex.I * φ) = -(5 / 13) + (12 / 13) * complex.I) : 
  sin_theta_plus_phi θ φ = 33 / 65 :=
sorry

end sin_of_sum_of_angles_l390_390956


namespace find_a_b_find_m_n_l390_390598

-- Define function f(x)
noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (2^x + b) / (2^x + a)

-- Define the condition that f passes through the origin
def passes_through_origin (a b : ℝ) : Prop := f 0 a b = 0

-- Define the condition that f passes through the point A(1, 1/3)
def passes_through_A (a b : ℝ) : Prop := f 1 a b = 1 / 3

-- Statement for Part 1: Finding a and b.
theorem find_a_b : ∃ a b : ℝ, passes_through_origin a b ∧ passes_through_A a b ∧ a = 1 ∧ b = -1 :=
sorry

-- Define function g(x)
noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := f x a b + x

-- Define the condition for the range of g on [m, 1]
def range_of_g (m n a b : ℝ) : Prop :=
∀ x : ℝ, (m ≤ x ∧ x ≤ 1) → (g m a b = m ∧ g 1 a b = n)

-- Statement for Part 2: Finding m and n.
theorem find_m_n : ∃ m n : ℝ, (range_of_g m n 1 -1) ∧ m = 0 ∧ n = 4 / 3 :=
sorry

end find_a_b_find_m_n_l390_390598


namespace find_angle_BAC_l390_390986

-- Define the predicates for congruence and angle measures
def congruent_triangles (T1 T2 : Type) : Prop := sorry
def angle_measure (p1 p2 p3 : Type) (angle : ℝ) : Prop := sorry

-- Define the problem and its conditions
variable (ABC EBD ADE ABD : Type)
variable (A B C D E : Type)
variable [congruent_triangles ABC EBD]
variable [congruent_triangles ABD EBD]
variable [angle_measure D A E 37]
variable [angle_measure D E A 37]

-- The angle in question
def angle_BAC : ℝ := sorry

-- The final proof statement
theorem find_angle_BAC : angle_BAC = 7 :=
sorry

end find_angle_BAC_l390_390986


namespace train_length_proof_l390_390482

def train_length_crosses_bridge (train_speed_kmh : ℕ) (bridge_length_m : ℕ) (crossing_time_s : ℕ) : ℕ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * crossing_time_s
  distance - bridge_length_m

theorem train_length_proof : 
  train_length_crosses_bridge 72 150 20 = 250 :=
by
  let train_speed_kmh := 72
  let bridge_length_m := 150
  let crossing_time_s := 20
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance := train_speed_ms * crossing_time_s
  have h : distance = 400 := by sorry
  have h_eq : distance - bridge_length_m = 250 := by sorry
  exact h_eq

end train_length_proof_l390_390482


namespace omega_2n_eq_omega_n_not_omega_2n_plus_3_eq_omega_n_plus_1_omega_8n_plus_5_eq_omega_4n_plus_3_omega_pow2_n_minus_1_eq_n_l390_390328

def bin_weight (n : ℕ) : ℕ := n.binary_digits.count 1

theorem omega_2n_eq_omega_n (n : ℕ) : bin_weight (2 * n) = bin_weight n := by
  sorry

theorem not_omega_2n_plus_3_eq_omega_n_plus_1 (n : ℕ) : bin_weight (2 * n + 3) ≠ bin_weight n + 1 := by
  sorry

theorem omega_8n_plus_5_eq_omega_4n_plus_3 (n : ℕ) : bin_weight (8 * n + 5) = bin_weight (4 * n + 3) := by
  sorry

theorem omega_pow2_n_minus_1_eq_n (n : ℕ) : bin_weight (2^n - 1) = n := by
  sorry

end omega_2n_eq_omega_n_not_omega_2n_plus_3_eq_omega_n_plus_1_omega_8n_plus_5_eq_omega_4n_plus_3_omega_pow2_n_minus_1_eq_n_l390_390328


namespace intersection_point_exists_l390_390798

theorem intersection_point_exists :
  ∃ (x y z t : ℝ), (x = 1 - 2 * t) ∧ (y = 2 + t) ∧ (z = -1 - t) ∧
                   (x - 2 * y + 5 * z + 17 = 0) ∧ 
                   (x = -1) ∧ (y = 3) ∧ (z = -2) :=
by
  sorry

end intersection_point_exists_l390_390798


namespace angle_A_equal_pi_div_4_side_a_sqrt_2_l390_390634

variable {α : Type*} [LinearOrderedField α]

noncomputable def triangle (A B C a b c : α) : Prop :=
  (b * Real.cos A - a * Real.sin B = 0) ∧ (0 < A) ∧ (A < Real.pi)

noncomputable def area_of_triangle (b c A : α) : α :=
  (1/2) * b * c * Real.sin A

theorem angle_A_equal_pi_div_4 (a b c A B C : α) (h : triangle A B C a b c) :
  A = (Real.pi / 4) := by
  -- sorry to skip the proof
  sorry

theorem side_a_sqrt_2 (a b c A B C : α) (h : triangle A B C a b c) (hb : b = Real.sqrt 2)
  (h_area : area_of_triangle b c A = 1) (hA : A = Real.pi / 4) : a = Real.sqrt 2 := by
  -- sorry to skip the proof
  sorry

end angle_A_equal_pi_div_4_side_a_sqrt_2_l390_390634


namespace number_division_l390_390470

theorem number_division (x : ℚ) (h : x / 6 = 1 / 10) : (x / (3 / 25)) = 5 :=
by {
  sorry
}

end number_division_l390_390470


namespace fish_bird_apple_fraction_l390_390065

theorem fish_bird_apple_fraction (M : ℝ) (hM : 0 < M) :
  let R_fish := 120
  let R_bird := 60
  let R_total := 180
  let T := M / R_total
  let fish_fraction := (R_fish * T) / M
  let bird_fraction := (R_bird * T) / M
  fish_fraction = 2/3 ∧ bird_fraction = 1/3 := by
  sorry

end fish_bird_apple_fraction_l390_390065


namespace eccentricity_range_l390_390295

-- Ellipse definition
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Upper vertex B of the ellipse
def B (b : ℝ) : ℝ × ℝ := (0, b)

-- Distance PB condition
def distance_le_2b (a b x0 y0 : ℝ) (h : ellipse a b (and.intro (sorry) (sorry)) x0 y0) : Prop :=
  (x0 - 0)^2 + (y0 - b)^2 ≤ (2 * b)^2

-- Range of eccentricity
def eccentricity (a b c : ℝ) : ℝ :=
  c / a

-- Main theorem
theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  (forall (x0 y0 : ℝ), ellipse a b h x0 y0 → distance_le_2b a b x0 y0 h) →
  eccentricity a b (sqrt (a^2 - b^2)) ∈ set.Icc 0 (sqrt 2 / 2) :=
by
  sorry

end eccentricity_range_l390_390295


namespace uphill_distance_is_100_l390_390023

def speed_uphill := 30  -- km/hr
def speed_downhill := 60  -- km/hr
def distance_downhill := 50  -- km
def avg_speed := 36  -- km/hr

-- Let d be the distance traveled uphill
variable (d : ℕ)

-- total distance is d + 50 km
def total_distance := d + distance_downhill

-- total time is (time uphill) + (time downhill)
def total_time := (d / speed_uphill) + (distance_downhill / speed_downhill)

theorem uphill_distance_is_100 (d : ℕ) (h : avg_speed = total_distance / total_time) : d = 100 :=
by
  sorry  -- proof is omitted

end uphill_distance_is_100_l390_390023


namespace shuttle_speed_conversion_l390_390826

theorem shuttle_speed_conversion (speed_in_kmph : ℕ) (conversion_factor : ℕ) : speed_in_kmph = 21600 → conversion_factor = 3600 → speed_in_kmph / conversion_factor = 6 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end shuttle_speed_conversion_l390_390826


namespace ticket_prices_l390_390051

theorem ticket_prices (x : ℕ) (h1 : ∃ n, x * n = 48) (h2 : ∃ m, x * m = 64) : 
  {x : ℕ | ∃ n, x * n = 48 ∧ ∃ m, x * m = 64}.card = 5 := 
by 
  sorry

end ticket_prices_l390_390051


namespace max_triangles_intersected_l390_390029

theorem max_triangles_intersected (needle_length side_length : ℝ) (tiles : set (set (ℝ × ℝ))) :
  side_length = 1 ∧ needle_length = 2 ∧ (∀ t ∈ tiles, ∃ (polygon t) (equilateral t), side_length (polygon t) = 1) →
  ∃ max_intersections : ℕ, max_intersections = 8 :=
by
  sorry

end max_triangles_intersected_l390_390029


namespace area_ratio_invariant_l390_390675

variables {A B C D E F P : Type*}
variables [affine_space A] [affine_space B] [affine_space C] [affine_space D]
variables [convex_space AB] [convex_space CD] [convex_space EF]
variables [measurable_space APD] [measurable_space BPC]
variables [cyclic_quadrilateral ABCD]

def is_point_between (P E F : A) : Prop := sorry -- To be provided.
def segment_ratio (X Y Z : Type*) [affine_space Z] : Prop := sorry -- ratio AE/EB
def point_on_segment (P E F : Type*) : Prop := sorry -- PE/PF ratio
def area_ratio (P D B C : Type*) [measurable_space] : Prop := sorry -- S_APD / S_BPC ratio.

theorem area_ratio_invariant 
 (cyclic_quadrilateral ABCD)
 (point_ratio_condition: Π (E F : A), segment_ratio AE EB ∧ segment_ratio CF FD)
 (P_on_EF: Π (E F : A), point_on_segment P E F)
 (S_APD S_BPC: measurable_space)
 : area_ratio APD BPC = AD / BC :=
begin
  sorry
end

end area_ratio_invariant_l390_390675


namespace train_pass_time_approx_5_07_l390_390795

def train_length : ℝ := 110  -- meters
def train_speed_kmph : ℝ := 78  -- km/h
def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600  -- converting km/h to m/s
def time_to_pass_pole : ℝ := train_length / train_speed_mps  -- time = distance / speed

theorem train_pass_time_approx_5_07 :
  abs(time_to_pass_pole - 5.07) < 0.01 := 
by
  sorry

end train_pass_time_approx_5_07_l390_390795


namespace range_of_a_l390_390933

noncomputable def f (x : ℝ) := Real.log x / Real.log 2

noncomputable def g (x a : ℝ) := Real.sqrt x + Real.sqrt (a - x)

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x1 : ℝ, 0 <= x1 ∧ x1 <= a → ∃ x2 : ℝ, 4 ≤ x2 ∧ x2 ≤ 16 ∧ g x1 a = f x2) →
  4 ≤ a ∧ a ≤ 8 :=
sorry 

end range_of_a_l390_390933


namespace probability_wait_10_probability_wait_15_l390_390036

theorem probability_wait_10 (arrival : ℝ) (h1 : 0 ≤ arrival ∧ arrival ≤ 30) : 
  (arrival ≤ 20) → (prob := 20 / 30) :=
by
  sorry

theorem probability_wait_15 (arrival : ℝ) (h1 : 0 ≤ arrival ∧ arrival ≤ 30) : 
  (arrival ≤ 25) → (prob := 25 / 30) :=
by
  sorry

end probability_wait_10_probability_wait_15_l390_390036


namespace range_of_eccentricity_l390_390274

variables {a b c e : ℝ}

def ellipse (x y : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)
def upper_vertex := b > 0
def distance_condition (x_0 y_0 : ℝ) := (x_0^2 + (y_0 - b)^2 ≤ (2 * b) ^ 2)
def eccentricity := e = c / a
def ellipse_condition := a = real.sqrt (b^2 + c^2)

theorem range_of_eccentricity (h1 : a > b) (h2 : upper_vertex) 
  (h3 : ∀ x_0 y_0, ellipse x_0 y_0 → distance_condition x_0 y_0)
  (h4 : ellipse_condition) :
  0 < e ∧ e ≤ real.sqrt 2 / 2 := 
by
  sorry

end range_of_eccentricity_l390_390274


namespace eccentricity_range_l390_390288

noncomputable def ellipse_eccentricity (a b : ℝ) (h_ab : a > b) : ℝ := (Real.sqrt (a ^ 2 - b ^ 2)) / a

theorem eccentricity_range (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ) :
  (∃ x y : ℝ, (x, y) = P ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  (Real.sqrt (x^2 + (y - b)^2) ≤ 2 * b)) → 
  0 < ellipse_eccentricity a b h_ab ∧ ellipse_eccentricity a b h_ab ≤ Real.sqrt 2 / 2 :=
begin
  sorry
end

end eccentricity_range_l390_390288


namespace determinant_equality_l390_390579

-- Given values p, q, r, s such that the determinant of the first matrix is 5
variables {p q r s : ℝ}

-- Define the determinant condition
def det_condition (p q r s : ℝ) : Prop := p * s - q * r = 5

-- State the theorem that we need to prove
theorem determinant_equality (h : det_condition p q r s) :
  p * (5*r + 2*s) - r * (5*p + 2*q) = 10 :=
sorry

end determinant_equality_l390_390579


namespace sequence_sum_S6_l390_390571

theorem sequence_sum_S6 (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (h : ∀ n, S_n n = 2 * a_n n - 3) :
  S_n 6 = 189 :=
by
  sorry

end sequence_sum_S6_l390_390571


namespace sin_cos_identity_l390_390513

theorem sin_cos_identity :
  let sin : ℝ → ℝ := λ θ, Real.sin θ
  let cos : ℝ → ℝ := λ theta, Real.cos theta
  let theta₁ : ℝ := -120 * Real.pi / 180
  let theta₂ : ℝ := 1290 * Real.pi / 180
  sin theta₁ * cos theta₂ = 3 / 4 :=
by
  have coterminal : (1290 * Real.pi / 180) % (2 * Real.pi) = -150 * Real.pi / 180 := sorry
  have sin_neg : sin (-120 * Real.pi / 180) = -sin (120 * Real.pi / 180) := sorry
  have sin_120 := sin (120 * Real.pi / 180) := Real.sqrt 3 / 2
  have cos_neg : cos (-150 * Real.pi / 180) = cos (150 * Real.pi / 180) := sorry
  have cos_150 := cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2
  show sin theta₁ * cos theta₂ = 3 / 4 from sorry

end sin_cos_identity_l390_390513


namespace triangle_solvability_condition_l390_390609

theorem triangle_solvability_condition (a b α β : ℝ) (h₁ : α = 1 * β / 2) (h₂ : a < b) :
    (b / 2 < a) ↔ (a > b / 2 ∧ a < b) :=
begin
  sorry
end

end triangle_solvability_condition_l390_390609


namespace inverse_proportion_k_value_l390_390385

theorem inverse_proportion_k_value (k m : ℝ) 
  (h1 : m = k / 3) 
  (h2 : 6 = k / (m - 1)) 
  : k = 6 :=
by
  sorry

end inverse_proportion_k_value_l390_390385


namespace linear_function_passes_through_another_point_l390_390920

theorem linear_function_passes_through_another_point
  (k : ℝ) (h1 : (∃ k, ∀ x y, y = k * x - k ∧ y (-1) = 4)) :
  ∃ c : ℝ, y = -2 * 1 + 2 := 
sorry

end linear_function_passes_through_another_point_l390_390920


namespace positive_integers_count_l390_390492

theorem positive_integers_count :
  let numbers := [4 / 3, 1, 3.14, 0, 10 / 100, -4, 100]
  in (list.count (λ n, (n > 0) ∧ (n = int.of_nat (n.to_nat))) numbers) = 2 := 
by
  let numbers := [4 / 3, 1, 3.14, 0, 10 / 100, -4, 100]
  exact sorry

end positive_integers_count_l390_390492


namespace solve_for_x_l390_390436

theorem solve_for_x (x y : ℤ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 :=
by
  sorry

end solve_for_x_l390_390436


namespace smallest_prime_less_than_perf_square_l390_390419

-- Define a predicate for being a prime number
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

-- The main goal
theorem smallest_prime_less_than_perf_square : ∃ n : ℕ, is_prime n ∧ ∃ m : ℕ, n = m^2 - 8 ∧ (∀ k : ℕ, is_prime k ∧ ∃ l : ℕ, k = l^2 - 8 → k ≥ n) :=
begin
  use 17,
  split,
  -- Proof that 17 is a prime number
  {
    unfold is_prime,
    split,
    { exact dec_trivial },
    { intros d hd,
      have h_d : d = 1 ∨ d = 17,
      { cases d,
        { exfalso, linarith, },
        { cases d,
          { left, refl, },
          { right, linarith [Nat.Prime.not_dvd_one 17 hd], }, }, },
      exact h_d, },
  },
  -- Proof that 17 is 8 less than a perfect square and the smallest such prime
  {
    use 5,
    split,
    { refl, },
    { intros k hk,
      cases hk with hk_prime hk_cond,
      cases hk_cond with l hl,
      rw hl,
      have : l ≥ 5,
      { intros,
        linarith, },
      exact this, },
  }
end

end smallest_prime_less_than_perf_square_l390_390419


namespace OC_expression_l390_390808

variable (θ : ℝ) -- angle theta
variable (s : ℝ) -- sin 2θ
variable (c : ℝ) -- cos 2θ

axiom sin_double_angle : s = Real.sin (2 * θ)

theorem OC_expression :
  ∃ (OC : ℝ), OC = 2 / (1 + s) :=
by {
  use 2 / (1 + s),
  sorry
}

end OC_expression_l390_390808


namespace three_person_subcommittees_count_l390_390174

theorem three_person_subcommittees_count : ∃ n k, n = 8 ∧ k = 3 ∧ nat.choose n k = 56 :=
begin
  use [8, 3],
  split,
  { refl },
  split,
  { refl },
  { norm_num }
end

end three_person_subcommittees_count_l390_390174


namespace value_of_a_l390_390961

theorem value_of_a (a : ℝ) (h : (a ^ 3) * ((5).choose (2)) = 80) : a = 2 :=
  sorry

end value_of_a_l390_390961


namespace slips_drawn_l390_390528

theorem slips_drawn (P : ℚ) (P_value : P = 24⁻¹) :
  ∃ n : ℕ, (n ≤ 5 ∧ P = (Nat.choose 5 n) / (Nat.choose 10 n) ∧ n = 4) := by
{
  sorry
}

end slips_drawn_l390_390528


namespace larry_wins_probability_eq_l390_390663

-- Define the conditions
def larry_probability_knocks_off : ℚ := 1 / 3
def julius_probability_knocks_off : ℚ := 1 / 4
def larry_throws_first : Prop := True
def independent_events : Prop := True

-- Define the proof that Larry wins the game with probability 2/3
theorem larry_wins_probability_eq :
  larry_throws_first ∧ independent_events →
  larry_probability_knocks_off = 1/3 ∧ julius_probability_knocks_off = 1/4 →
  ∃ p : ℚ, p = 2 / 3 :=
by
  sorry

end larry_wins_probability_eq_l390_390663


namespace book_cost_l390_390993

theorem book_cost (n_5 n_3 : ℕ) (N : ℕ) :
  (N = n_5 + n_3) ∧ (N > 10) ∧ (N < 20) ∧ (5 * n_5 = 3 * n_3) →  5 * n_5 = 30 := 
sorry

end book_cost_l390_390993


namespace conic_section_type_l390_390081

theorem conic_section_type (x y : ℝ) : 
  9 * x^2 - 36 * y^2 = 36 → 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1) :=
by
  sorry

end conic_section_type_l390_390081


namespace probability_john_david_chosen_l390_390005

theorem probability_john_david_chosen (total_workers : ℕ) (choose_workers : ℕ) (john_david_ways : ℕ) (total_ways : ℕ) :
  total_workers = 10 →
  choose_workers = 2 →
  john_david_ways = 1 →
  total_ways = Nat.choose total_workers choose_workers →
  (john_david_ways : ℚ) / total_ways = 1 / 45 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  dsimp [Nat.choose]
  norm_num
  sorry

end probability_john_david_chosen_l390_390005


namespace even_functions_implications_l390_390321

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_deriv : ℝ → ℝ := sorry

lemma f_x_minus_1_even (x : ℝ) : f (-x - 1) = f (x - 1) := sorry
lemma f_x_minus_2_even (x : ℝ) : f (-x - 2) = f (x - 2) := sorry

theorem even_functions_implications :
  (∀ x, f (-x - 1) = f (x - 1)) ∧ (∀ x, f (-x - 2) = f (x - 2)) →
    (∀ x, f (-x) = f (x)) ∧
    (∃ k > 0, ∀ x, f (x + k) = f (x)) ∧
    (∀ x, f_deriv (-x + 2) = f_deriv (x + 2)) :=
begin
  intros h,
  split,
  -- Part A
  sorry, -- Prove f is symmetric about x = -1
  split,
  -- Part B
  sorry, -- Prove 2 is a period of f
  -- Part C
  sorry -- Prove the graph of f' is symmetric about (2, 0)
end

end even_functions_implications_l390_390321


namespace number_of_terms_divisible_by_2_not_4_l390_390900

-- Define the sum of the first n natural numbers
def a (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the series sum Sm
def S (m : ℕ) : ℕ := (Finset.range (m + 1)).sum (λ i => a i)

-- Define the condition that Sm is divisible by 2 but not by 4
def condition (m : ℕ) : Bool := S m % 4 = 2

-- Prove that the number of m in the range 1 to 2017 such that Sm % 4 = 2 is 252
theorem number_of_terms_divisible_by_2_not_4 : (Finset.range 2017).filter condition = 252 := by
  sorry

end number_of_terms_divisible_by_2_not_4_l390_390900


namespace omega_2n_eq_omega_n_omega_8n_plus_5_eq_omega_4n_plus_3_omega_2_pow_n_minus_1_eq_n_l390_390326

def bit_weight (n : ℕ) : ℕ :=
  (n.bits.map (λ b, if b then 1 else 0)).sum

theorem omega_2n_eq_omega_n (n : ℕ) : 
  bit_weight (2 * n) = bit_weight n := 
sorry

theorem omega_8n_plus_5_eq_omega_4n_plus_3 (n : ℕ) : 
  bit_weight (8 * n + 5) = bit_weight (4 * n + 3) := 
sorry

theorem omega_2_pow_n_minus_1_eq_n (n : ℕ) : 
  bit_weight (2 ^ n - 1) = n := 
sorry

end omega_2n_eq_omega_n_omega_8n_plus_5_eq_omega_4n_plus_3_omega_2_pow_n_minus_1_eq_n_l390_390326


namespace triangle_area_l390_390971

theorem triangle_area (a c : ℝ) (B : ℝ) (h_a : a = 7) (h_c : c = 5) (h_B : B = 120 * Real.pi / 180) : 
  (1 / 2 * a * c * Real.sin B) = 35 * Real.sqrt 3 / 4 := by
  sorry

end triangle_area_l390_390971


namespace f_is_odd_f_extrema_l390_390688

-- Given conditions for f(x)
variable (f : ℝ → ℝ)
hypothesis H1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
hypothesis H2 : ∀ x : ℝ, x < 0 → f(x) < 0
hypothesis H3 : f(-1) = -2

-- 1. Prove that f(x) is an odd function
theorem f_is_odd : ∀ x : ℝ, f(-x) = -f(x) := by {
  sorry
}

-- 2. Prove that f(x) has a maximum value 4 and minimum value -4 in [-2, 2]
theorem f_extrema : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f(x) ≤ 4 ∧ f(x) ≥ -4 := by {
  sorry
}

end f_is_odd_f_extrema_l390_390688


namespace fraction_sum_eq_neg_one_l390_390554

theorem fraction_sum_eq_neg_one (p q : ℝ) (hpq : (1 / p) + (1 / q) = (1 / (p + q))) :
  (p / q) + (q / p) = -1 :=
by
  sorry

end fraction_sum_eq_neg_one_l390_390554


namespace min_red_hair_students_l390_390016

variable (B N R : ℕ)
variable (total_students : ℕ := 50)

theorem min_red_hair_students :
  B + N + R = total_students →
  (∀ i, B > i → N > 0) →
  (∀ i, N > i → R > 0) →
  R ≥ 17 :=
by {
  -- The specifics of the proof are omitted as per the instruction
  sorry
}

end min_red_hair_students_l390_390016


namespace sum_of_fourth_powers_l390_390753

theorem sum_of_fourth_powers (n : ℤ) 
  (h : n * (n + 1) * (n + 2) = 12 * (n + (n + 1) + (n + 2))) : 
  (n^4 + (n + 1)^4 + (n + 2)^4) = 7793 := 
by 
  sorry

end sum_of_fourth_powers_l390_390753


namespace friends_mushroom_count_l390_390886

noncomputable def mushroom : Type := ℕ

theorem friends_mushroom_count (x1 x2 x3 x4 : mushroom) 
  (h1 : x1 + x2 = 6) 
  (h2 : x1 + x3 = 7) 
  (h3 : x2 + x3 = 9) 
  (h4 : x1 + x4 ∈ {11, 12}) 
  (h5 : x2 + x4 ∈ {11, 12}) 
  (h6 : x3 + x4 ∈ {11, 12}) :
  (x1 = 2) ∧ (x2 = 4) ∧ (x3 = 5) ∧ (x4 = 7) :=
  sorry

end friends_mushroom_count_l390_390886


namespace females_with_advanced_degrees_l390_390004

theorem females_with_advanced_degrees (total_employees : ℕ) (total_females : ℕ) (total_advanced_degrees : ℕ)
  (total_college_degree_only : ℕ) (males_college_degree_only : ℕ) (total_employees = 148)
  (total_females = 92) (total_advanced_degrees = 78) (total_college_degree_only = 148 - 78)
  (males_college_degree_only = 31) :
  (total_advanced_degrees - (total_employees - total_females - males_college_degree_only)) = 53 :=
by sorry

end females_with_advanced_degrees_l390_390004


namespace value_of_x_l390_390625

theorem value_of_x (x : ℚ) (h : (3 * x + 4) / 7 = 15) : x = 101 / 3 :=
by
  sorry

end value_of_x_l390_390625


namespace solution_set_of_f_lt_exp_l390_390137

noncomputable def f : ℝ → ℝ := sorry -- assume f is a differentiable function

-- Define the conditions
axiom h_deriv : ∀ x : ℝ, deriv f x < f x
axiom h_periodic : ∀ x : ℝ, f (x + 2) = f (x - 2)
axiom h_value_at_4 : f 4 = 1

-- The main statement to be proved
theorem solution_set_of_f_lt_exp :
  ∀ x : ℝ, (f x < Real.exp x ↔ x > 0) :=
by
  intro x
  sorry

end solution_set_of_f_lt_exp_l390_390137


namespace pool_capacity_given_conditions_l390_390454

variable {V1 V2 V3 : ℝ} (r : ℝ)

-- Conditions
def condition1 : Prop := 36 * (V1 + V2 + V3) = r
def condition2 : Prop := 120 * V1 = r
def condition3 : Prop := V2 = V1 + 50
def condition4 : Prop := V3 = V1 - 25

theorem pool_capacity_given_conditions :
  condition1 r V1 V2 V3 →
  condition2 r V1 →
  condition3 V1 V2 →
  condition4 V1 V3 →
  r = 9000 :=
by
  intros
  sorry

end pool_capacity_given_conditions_l390_390454


namespace sequence_periodic_l390_390601

noncomputable def sequence (n : ℕ) : ℝ :=
  Nat.recOn n 2 (λ n a_n, 1 - 1 / a_n)

theorem sequence_periodic : (sequence 2018) = 1 / 2 := by
  sorry

end sequence_periodic_l390_390601


namespace orthogonality_of_vectors_l390_390161

variables {V : Type*} [inner_product_space ℝ V]
variables (a e : V)
variable (t : ℝ)

theorem orthogonality_of_vectors
  (h1 : a ≠ e)
  (h2 : ‖e‖ = 1)
  (h3 : ∀ t : ℝ, ‖a - t • e‖ ≥ ‖a - e‖) : 
  inner e (a - e) = 0 := 
sorry

end orthogonality_of_vectors_l390_390161


namespace part1_part2_l390_390533

theorem part1 (n : Nat) (hn : 0 < n) : 
  (∃ k, -5^4 + 5^5 + 5^n = k^2) -> n = 5 :=
by
  sorry

theorem part2 (n : Nat) (hn : 0 < n) : 
  (∃ m, 2^4 + 2^7 + 2^n = m^2) -> n = 8 :=
by
  sorry

end part1_part2_l390_390533


namespace prime_factor_of_difference_l390_390382

noncomputable def two_digit_difference_prime_factor (A B : ℕ) (h1 : A ≠ B) (h2 : A > B) : Prop :=
  let AB := 10 * A + B
  let BA := 10 * B + A
  let diff := AB - (BA - 5)
  (∀ p : ℕ, prime p → p ∣ diff → p = 3)

-- Statement
theorem prime_factor_of_difference (A B : ℕ) (h1 : A ≠ B) (h2 : A > B) : two_digit_difference_prime_factor A B h1 h2 :=
  sorry

end prime_factor_of_difference_l390_390382


namespace correct_student_mark_l390_390376

theorem correct_student_mark (x : ℕ) : 
  (∀ (n : ℕ), n = 30) →
  (∀ (avg correct_avg wrong_mark correct_mark : ℕ), 
    avg = 100 ∧ 
    correct_avg = 98 ∧ 
    wrong_mark = 70 ∧ 
    (n * avg) - wrong_mark + correct_mark = n * correct_avg) →
  x = 10 := by
  intros
  sorry

end correct_student_mark_l390_390376


namespace number_of_sides_of_polygon_l390_390872

variable (R : ℝ) (n : ℕ)
variables (α : ℝ)
variable (k : ℕ)
variables (A B C D : Type)

-- Assume conditions
axiom regular_polygon (n : ℕ): n ≥ 3
axiom regular_polygon_in_circle (O : Type) (R : ℝ) (A B C D : Type) :
  ∃ (α : ℝ), (0 < α ∧ α < 120 * (π/180)) ∧
  AB ≠ 0 ∧
  (A = B ∨ B = C ∨ C = D ∨ D = A)

-- Length definitions in terms of Radius and α
def arc_length_AB (R : ℝ) (α : ℝ) : ℝ := 2 * R * sin (α / 2)
def arc_length_AC (R : ℝ) (α : ℝ) : ℝ := 2 * R * sin α
def arc_length_AD (R : ℝ) (α : ℝ) : ℝ := 2 * R * sin (3 * α / 2)

-- Given equality
axiom given_equality (R : ℝ) (α : ℝ) (A B C D : Type) :
  1 / arc_length_AB R α = 1 / arc_length_AC R α + 1 / arc_length_AD R α

-- The final theorem
theorem number_of_sides_of_polygon (R : ℝ) (A B C D : Type) :
  ∃ (n : ℕ), n = 7 :=
by
  -- Each step would build upon the Proof from here
  sorry

end number_of_sides_of_polygon_l390_390872


namespace find_f3_l390_390451

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f(2 * x + 1) = 2 * f(x) + 1
axiom initial_condition : f(0) = 2

theorem find_f3 : f(3) = 11 := sorry

end find_f3_l390_390451


namespace eccentricity_range_l390_390269

noncomputable def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  { z : ℝ × ℝ // (z.1^2 / a^2) + (z.2^2 / b^2) = 1 }

theorem eccentricity_range (a b : ℝ) (h : a > b) (hb : b > 0):
  ∀ (P : ellipse a b h hb),
    |(P.1, P.2 - b).norm ≤ 2 * b → 
    eccentricity (a b) ∈ set.Ici (real.sqrt 2 / 2) :=
sorry

end eccentricity_range_l390_390269


namespace vertex_of_quadratic_function_l390_390740

theorem vertex_of_quadratic_function :
  ∀ x: ℝ, (2 - (x + 1)^2) = 2 - (x + 1)^2 → (∃ h k : ℝ, (h, k) = (-1, 2) ∧ ∀ x: ℝ, (2 - (x + 1)^2) = k - (x - h)^2) :=
by
  sorry

end vertex_of_quadratic_function_l390_390740


namespace functional_equation_solution_l390_390097

theorem functional_equation_solution (f : ℚ → ℚ) (H : ∀ x y : ℚ, f (x + y) = f x + f y) :
  ∃ k : ℚ, ∀ x : ℚ, f x = k * x := 
sorry

end functional_equation_solution_l390_390097


namespace expression_equals_4_l390_390843

noncomputable def calculate_expression : ℝ :=
  0.25 * (1 / 2) ^ (-2) + Real.log10 8 + 3 * Real.log10 5

theorem expression_equals_4 : calculate_expression = 4 := 
by
  sorry

end expression_equals_4_l390_390843


namespace probability_f_geq_zero_is_six_elevens_l390_390125
noncomputable def probability_f_geq_zero : ℝ := 
  let a := -Real.pi / 4 in
  let b := 2 * Real.pi / 3 in
  let f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6) in
  let total_length := b - a in
  let c := Real.pi / 12 in
  let d := 7 * Real.pi / 12 in
  let f_nonneg_length := d - c in
  f_nonneg_length / total_length

theorem probability_f_geq_zero_is_six_elevens :
  probability_f_geq_zero = 6 / 11 :=
by
  sorry

end probability_f_geq_zero_is_six_elevens_l390_390125


namespace binomial_probability_eq_l390_390124

noncomputable def binomial_pmf (n : ℕ) (p : ℚ) : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.mixture (n+1) (λ k, Mathbin.binom n k • (p^k • (1 - p)^(n - k)))

theorem binomial_probability_eq
  (n : ℕ) (p : ℚ) (X : ℕ → MeasureTheory.Meas ℕ)
  (h1 : ProbabilityMassFunction.likelihood X = binomial_pmf n p)
  (h2 : (X.map ennreal.to_real).expected_value = 2)
  (h3 : (X.map ennreal.to_real).variance = 4 / 3) :
  ProbabilityMassFunction.probability X 2 = 80 / 243 :=
sorry

end binomial_probability_eq_l390_390124


namespace fraction_of_p_l390_390006

theorem fraction_of_p (p q r f : ℝ) (hp : p = 49) (hqr : p = (2 * f * 49) + 35) : f = 1/7 :=
sorry

end fraction_of_p_l390_390006


namespace average_spring_headcount_average_fall_headcount_l390_390505

namespace AverageHeadcount

def springHeadcounts := [10900, 10500, 10700, 11300]
def fallHeadcounts := [11700, 11500, 11600, 11300]

def averageHeadcount (counts : List ℕ) : ℕ :=
  counts.sum / counts.length

theorem average_spring_headcount :
  averageHeadcount springHeadcounts = 10850 := by
  sorry

theorem average_fall_headcount :
  averageHeadcount fallHeadcounts = 11525 := by
  sorry

end AverageHeadcount

end average_spring_headcount_average_fall_headcount_l390_390505


namespace eccentricity_range_l390_390283

noncomputable def ellipse_eccentricity (a b : ℝ) (h_ab : a > b) : ℝ := (Real.sqrt (a ^ 2 - b ^ 2)) / a

theorem eccentricity_range (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ) :
  (∃ x y : ℝ, (x, y) = P ∧ (x^2 / a^2 + y^2 / b^2 = 1) ∧ 
  (Real.sqrt (x^2 + (y - b)^2) ≤ 2 * b)) → 
  0 < ellipse_eccentricity a b h_ab ∧ ellipse_eccentricity a b h_ab ≤ Real.sqrt 2 / 2 :=
begin
  sorry
end

end eccentricity_range_l390_390283


namespace absent_minded_scientist_two_packages_probability_l390_390208

noncomputable def probability_two_packages : ℝ :=
  let n := 10 in
  (2^n - 1).toReal / ((2^(n - 1)) * n).toReal

theorem absent_minded_scientist_two_packages_probability :
  probability_two_packages = 0.1998 := by
  sorry

end absent_minded_scientist_two_packages_probability_l390_390208


namespace find_prime_m_and_n_l390_390532

def is_perfect_power (x : ℕ) : Prop :=
  ∃ a k : ℕ, k > 1 ∧ x = a^k

theorem find_prime_m_and_n (m n : ℕ) (hm : m > 1) :
  (∀ b : ℕ, b ≤ m → gcd b m ≠ 1 → ∃ a : ℕ → ℕ, (∀ i, gcd (a i) m = 1) ∧ is_perfect_power (m + finset.sum (finset.range n) (λ i, (a i) * b^(i + 1)))) ↔ ∃ p : ℕ, prime p ∧ m = p ∧ n ∈ ℕ :=
sorry

end find_prime_m_and_n_l390_390532


namespace area_ABC_l390_390870

-- Define the points A, B, and C
variables {A B C : Type} [Euclidean_space ℝ V]

-- Define the basic properties and conditions
variables [triangle_ABC : triangle A B C]
variables (h_right_angle : ∠ B = 90°)
variables (h_AC : A.distance C = 40)
variables (h_angle_A : ∠ A = 60°)
variables (h_30_60_90 : is_30_60_90_triangle A B C)

-- State the theorem for the area
theorem area_ABC : area A B C = 200 * sqrt 3 :=
by sorry

end area_ABC_l390_390870


namespace largest_divisor_l390_390960

theorem largest_divisor (n : ℕ) (h1 : 0 < n) (h2 : 450 ∣ n ^ 2) : 30 ∣ n :=
sorry

end largest_divisor_l390_390960


namespace find_m_l390_390855

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x m : ℝ) : ℝ := x^2 - m * x - 8

theorem find_m (m : ℝ) (h : f 5 - g 5 m = 20) : m = -13.6 :=
by sorry

end find_m_l390_390855


namespace exists_triangle_with_no_interior_points_l390_390674

theorem exists_triangle_with_no_interior_points 
  (A B : Finset ℝ × ℝ) 
  (no_common : ∀ (a ∈ A) (b ∈ B), a ≠ b)
  (no_collinear : ∀ {x y z : (ℝ × ℝ)}, x ∈ (A ∪ B) → y ∈ (A ∪ B) → z ∈ (A ∪ B) → ¬collinear {x, y, z})
  (A_or_B_large : A.card ≥ 5 ∨ B.card ≥ 5) : 
  ∃ T ∈ (Finset.powersetLen 3 (A ∪ B)), 
    (T ⊆ A ∨ T ⊆ B) ∧ ∀ p ∈ B, (p ∉ T) ∧ ∀ q ∈ A, (q ∉ T) → 
    (T.convexHull : Set (ℝ × ℝ)).interior ∩ (if T ⊆ A then B else A) = ∅ := 
sorry

end exists_triangle_with_no_interior_points_l390_390674


namespace quadratic_mean_ge_arithmetic_mean_l390_390774

theorem quadratic_mean_ge_arithmetic_mean {n : ℕ} (hn : 0 < n) (a : Fin n → ℝ) :
    sqrt ((∑ i, (a i)^2) / n) >= (∑ i, a i) / n ∧
    (sqrt ((∑ i, (a i)^2) / n) = (∑ i, a i) / n ↔ ∀ i, a i = a 0) :=
by
  sorry

end quadratic_mean_ge_arithmetic_mean_l390_390774


namespace sum_zero_of_product_one_l390_390727

theorem sum_zero_of_product_one (x y : ℝ) 
  (h : (x + sqrt (1 + x^2)) * (y + sqrt (1 + y^2)) = 1) : x + y = 0 :=
sorry

end sum_zero_of_product_one_l390_390727


namespace sequence_formula_sum_formula_l390_390233

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  a ^ x

def S_n (n : ℕ) (a : ℝ) : ℝ :=
  f n a - 1

-- General formula for the nth term of the sequence {a_n}
theorem sequence_formula 
  (a : ℝ) (ha : 0 < a) (ha_ne : a ≠ 1) (h_point : f 1 a = 2) (n : ℕ) :
  ∃ a_n : ℕ → ℝ, a_n n = 2 ^ (n - 1) :=
sorry

-- Sum of the first n terms of the sequence {a_n b_n}
theorem sum_formula (a : ℝ) (ha : 0 < a) (ha_ne : a ≠ 1) (h_point : f 1 a = 2) (n : ℕ) :
  ∃ T_n : ℕ → ℝ, T_n n = (n - 1) * 2 ^ n + 1 :=
sorry

end sequence_formula_sum_formula_l390_390233


namespace circles_externally_tangent_l390_390523

noncomputable def circle1_center : ℝ × ℝ := (-1, -2)
noncomputable def circle1_radius : ℝ := Real.sqrt 2
noncomputable def circle2_center : ℝ × ℝ := (2, 1)
noncomputable def circle2_radius : ℝ := 2 * Real.sqrt 2

def distance_between_centers (c1 c2 : ℝ × ℝ) : ℝ := (Real.sqrt ((c2.1 - c1.1) ^ 2 + (c2.2 - c1.2) ^ 2))

def are_externally_tangent (cir1 c1_rad cir2 c2_rad : ℝ × ℝ) : Prop :=
  distance_between_centers cir1 cir2 = c1_rad + c2_rad

theorem circles_externally_tangent :
  are_externally_tangent circle1_center circle1_radius circle2_center circle2_radius :=
by
  unfold are_externally_tangent distance_between_centers circle1_center circle1_radius circle2_center circle2_radius
  sorry

end circles_externally_tangent_l390_390523


namespace range_of_a_l390_390145

variable (f : ℝ → ℝ)
variable (a : ℝ)
noncomputable def f_def (x : ℝ) : ℝ := x^2 * Real.log x - a * (x^2 - 1)

theorem range_of_a (h : ∀ x ∈ Set.Ioc 0 1, f_def f a x ≥ 0) : a ≥ 1 / 2 :=
by
  sorry

end range_of_a_l390_390145


namespace count_non_congruent_triangles_with_perimeter_18_l390_390950

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def num_non_congruent_triangles_with_perimeter_18 : ℕ :=
  8

theorem count_non_congruent_triangles_with_perimeter_18 :
  (∃ (triangles : Finset (ℕ × ℕ × ℕ)),
    ∀ t ∈ triangles, 
      (let a := t.1 in let b := t.2.1 in let c := t.2.2 in
      3 ≤ a ∧ 3 ≤ b ∧ 3 ≤ c ∧ a + b + c = 18 ∧ is_valid_triangle a b c) ∧
    triangles.card = 8) :=
sorry

end count_non_congruent_triangles_with_perimeter_18_l390_390950


namespace contrapositive_sin_l390_390595

theorem contrapositive_sin (x y : ℝ) : (¬ (sin x = sin y)) → (x ≠ y) :=
by {
  have h1 : (x = y) → (sin x = sin y) := 
    fun hxy : x = y => by { rw hxy },
  exact λ hsin : ¬ (sin x = sin y), λ hxy : x = y, hsin (h1 hxy)
}

end contrapositive_sin_l390_390595


namespace density_of_body_is_1500_l390_390644

noncomputable def density_of_body_in_equilibrium 
  (F : ℝ) (V : ℝ) (rho_liq : ℝ) (g : ℝ) : ℝ :=
  (rho_liq * g * V + F) / (g * V)

theorem density_of_body_is_1500 
  (hF : F = 5)
  (hV : V = 10⁻³)
  (hrho_liq : rho_liq = 1000)
  (hg : g = 10)
  : density_of_body_in_equilibrium 5 10⁻³ 1000 10 = 1500 := by
  sorry

end density_of_body_is_1500_l390_390644


namespace eccentricity_range_l390_390266

noncomputable def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  { z : ℝ × ℝ // (z.1^2 / a^2) + (z.2^2 / b^2) = 1 }

theorem eccentricity_range (a b : ℝ) (h : a > b) (hb : b > 0):
  ∀ (P : ellipse a b h hb),
    |(P.1, P.2 - b).norm ≤ 2 * b → 
    eccentricity (a b) ∈ set.Ici (real.sqrt 2 / 2) :=
sorry

end eccentricity_range_l390_390266


namespace cos_7_theta_l390_390618

variable (θ : Real)

namespace CosineProof

theorem cos_7_theta (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -5669 / 16384 := by
  sorry

end CosineProof

end cos_7_theta_l390_390618


namespace determinant_transformation_l390_390952

theorem determinant_transformation (p q r s : ℝ) (h : p * s - q * r = -3) :
    p * (5 * r + 2 * s) - r * (5 * p + 2 * q) = -6 := by
  sorry

end determinant_transformation_l390_390952


namespace Suzanna_rides_8_miles_in_40_minutes_l390_390731

theorem Suzanna_rides_8_miles_in_40_minutes :
  (∀ n : ℕ, Suzanna_distance_in_n_minutes = (n / 10) * 2) → Suzanna_distance_in_40_minutes = 8 :=
by
  sorry

-- Definitions for Suzanna's distance conditions
def Suzanna_distance_in_n_minutes (n : ℕ) : ℕ := (n / 10) * 2

noncomputable def Suzanna_distance_in_40_minutes := Suzanna_distance_in_n_minutes 40

#check Suzanna_rides_8_miles_in_40_minutes

end Suzanna_rides_8_miles_in_40_minutes_l390_390731


namespace no_solution_inequalities_l390_390205

theorem no_solution_inequalities (m : Real) : 
  (¬ ∃ (x : Real), (x - 2 < 3x - 6) ∧ (x < m)) → m ≤ 2 :=
sorry

end no_solution_inequalities_l390_390205


namespace number_of_whole_numbers_between_sqrts_l390_390183

noncomputable def count_whole_numbers_between_sqrts : ℕ :=
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let min_int := Int.ceil lower_bound
  let max_int := Int.floor upper_bound
  Int.natAbs (max_int - min_int + 1)

theorem number_of_whole_numbers_between_sqrts :
  count_whole_numbers_between_sqrts = 7 :=
by
  sorry

end number_of_whole_numbers_between_sqrts_l390_390183


namespace eccentricity_range_l390_390308

variables {a b : ℝ}
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x2 - x1)^2 + (y2 - y1)^2).sqrt

theorem eccentricity_range (h1 : a > b) (h2 : b > 0)
  (h3 : ∀ (x y : ℝ), ellipse x y → distance x y 0 b ≤ 2 * b) :
  0 < (sqrt (a^2 - b^2) / a) ∧ (sqrt (a^2 - b^2) / a) ≤ sqrt 2 / 2 :=
sorry

end eccentricity_range_l390_390308


namespace car_speed_l390_390003

variable (v : ℝ)
variable (Distance : ℝ := 1)  -- distance in kilometers
variable (Speed_120 : ℝ := 120)  -- speed in kilometers per hour
variable (Time_120 : ℝ := Distance / Speed_120)  -- time in hours to travel 1 km at 120 km/h
variable (Time_120_sec : ℝ := Time_120 * 3600)  -- time in seconds to travel 1 km at 120 km/h
variable (Additional_time : ℝ := 2)  -- additional time in seconds
variable (Time_v_sec : ℝ := Time_120_sec + Additional_time)  -- time in seconds for unknown speed
variable (Time_v : ℝ := Time_v_sec / 3600)  -- time in hours for unknown speed

theorem car_speed (h : v = Distance / Time_v) : v = 112.5 :=
by
  -- The given proof steps will go here
  sorry

end car_speed_l390_390003


namespace number_of_small_triangles_l390_390066

theorem number_of_small_triangles 
  (side_large : ℝ) 
  (side_small : ℝ) 
  (h1 : side_large = 12) 
  (h2 : side_small = 2) : 
  (side_large * side_large * (sqrt 3 / 4)) / (side_small * side_small * (sqrt 3 / 4)) = 36 := by 
    sorry

end number_of_small_triangles_l390_390066


namespace round_3967149_6528_l390_390365

-- Set up the necessary definitions and conditions
def fractional_part (x : ℝ) : ℝ := x - x.floor

def rounds_up (x : ℝ) : Prop :=
  fractional_part x ≥ 0.5

def rounds_to_nearest_integer (x : ℝ) (y : ℤ) : Prop :=
  if rounds_up x then y = x.ceil else y = x.floor

-- State the problem as a theorem
theorem round_3967149_6528 :
  rounds_to_nearest_integer 3967149.6528 3967150 :=
sorry

end round_3967149_6528_l390_390365


namespace arithmetic_sequence_general_formula_l390_390572

noncomputable def a_n (n : ℕ) : ℝ :=
sorry

theorem arithmetic_sequence_general_formula (h1 : (a_n 2 + a_n 6) / 2 = 5)
                                            (h2 : (a_n 3 + a_n 7) / 2 = 7) :
  a_n n = 2 * (n : ℝ) - 3 :=
sorry

end arithmetic_sequence_general_formula_l390_390572


namespace ratio_of_tangent_to_circumference_l390_390381

theorem ratio_of_tangent_to_circumference
  {r x : ℝ}  -- radius of the circle and length of the tangent
  (hT : x = 2 * π * r)  -- given the length of tangent PQ
  (hA : (1 / 2) * x * r = π * r^2)  -- given the area equivalence

  : (x / (2 * π * r)) = 1 :=  -- desired ratio
by
  -- proof omitted, just using sorry to indicate proof
  sorry

end ratio_of_tangent_to_circumference_l390_390381


namespace feet_in_mile_l390_390495

theorem feet_in_mile (d t : ℝ) (speed_mph : ℝ) (speed_fps : ℝ) (miles_to_feet : ℝ) (hours_to_seconds : ℝ) :
  d = 200 → t = 4 → speed_mph = 34.09 → miles_to_feet = 5280 → hours_to_seconds = 3600 → 
  speed_fps = d / t → speed_fps = speed_mph * miles_to_feet / hours_to_seconds → 
  miles_to_feet = 5280 :=
by
  intros hd ht hspeed_mph hmiles_to_feet hhours_to_seconds hspeed_fps_eq hconversion
  -- You can add the proof steps here.
  sorry

end feet_in_mile_l390_390495


namespace zero_sequence_if_p_balanced_l390_390347

def p_balanced (a : ℕ → ℤ) (p : ℕ) : Prop :=
∀ k : ℕ, (∑ i in Finset.range 50 \ Finset.Ico 0 k, a (i*p + k)) = (∑ i in Finset.Ico 0 k, a (i*p + k))

theorem zero_sequence_if_p_balanced :
  (∀ p ∈ ({3, 5, 7, 11, 13, 17} : Finset ℕ), p_balanced (fun k => if k < 50 then a k else 0) p) → 
  (∀ k, k < 50 → a k = 0) :=
begin
  sorry
end

end zero_sequence_if_p_balanced_l390_390347


namespace area_of_square_l390_390695

-- Conditions: Points A (5, -2) and B (5, 3) are adjacent corners of a square.
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (5, 3)

-- The statement to prove that the area of the square formed by these points is 25.
theorem area_of_square : (∃ s : ℝ, s = Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) → s^2 = 25 :=
sorry

end area_of_square_l390_390695


namespace calculate_x15_plus_y15_l390_390194

open Complex

noncomputable def x : ℂ := (-1 + complex.I * real.sqrt 3) / 2
noncomputable def y : ℂ := (-1 - complex.I * real.sqrt 3) / 2

theorem calculate_x15_plus_y15 : x^15 + y^15 = 2 := by
  sorry

end calculate_x15_plus_y15_l390_390194


namespace geom_prog_sum_of_squares_l390_390904

theorem geom_prog_sum_of_squares :
  ∃ (a r : ℕ), 
    (0 < a ∧ a < 100) ∧ 
    (a * r < 100) ∧ 
    (a * r^2 < 100) ∧ 
    (a * r^3 < 100) ∧ 
    (a * r^4 < 100) ∧ 
    (a + a * r + a * r^2 + a * r^3 + a * r^4 = 211) ∧ 
    (let S := (if is_square (a) then a else 0) + 
              (if is_square (a * r) then a * r else 0) + 
              (if is_square (a * r^2) then a * r^2 else 0) + 
              (if is_square (a * r^3) then a * r^3 else 0) + 
              (if is_square (a * r^4) then a * r^4 else 0) in 
    S = 133) :=
by
  -- Specifying that no constructive proof is provided here
  sorry

-- Helper function to check if a number is a perfect square
def is_square (n : ℕ) : Prop := ∃ k : ℕ, k^2 = n

end geom_prog_sum_of_squares_l390_390904


namespace 数学竞赛_is_1962_l390_390225

-- Define the digits and their constraints
variable (数 学 竞 赛 : ℕ)

-- Conditions: each digit is between 0 and 9, and all digits are unique
axiom 数_bounds : 0 ≤ 数 ∧ 数 ≤ 9
axiom 学_bounds : 0 ≤ 学 ∧ 学 ≤ 9
axiom 竞_bounds : 0 ≤ 竞 ∧ 竞 ≤ 9
axiom 赛_bounds : 0 ≤ 赛 ∧ 赛 ≤ 9

axiom 数_neq_学 : 数 ≠ 学
axiom 数_neq_竞 : 数 ≠ 竞
axiom 数_neq_赛 : 数 ≠ 赛
axiom 学_neq_竞 : 学 ≠ 竞
axiom 学_neq_赛 : 学 ≠ 赛
axiom 竞_neq_赛 : 竞 ≠ 赛

-- Defining the four-digit number represented by "数学竞赛"
def 数学竞赛_number := 数 * 1000 + 学 * 100 + 竞 * 10 + 赛

-- The correct four-digit number is 1962
theorem 数学竞赛_is_1962 (h : 数学竞赛_number 数 学 竞 赛 = 1962) : true :=
by
  sorry

end 数学竞赛_is_1962_l390_390225


namespace solution_set_for_absolute_value_inequality_l390_390781

theorem solution_set_for_absolute_value_inequality :
  {x : ℝ | |2 * x - 1| ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by 
  sorry

end solution_set_for_absolute_value_inequality_l390_390781


namespace sean_more_whistles_than_charles_l390_390717

theorem sean_more_whistles_than_charles :
  let S := 45 in
  let C := 13 in
  S - C = 32 :=
by
  let S := 45
  let C := 13
  show S - C = 32
  sorry

end sean_more_whistles_than_charles_l390_390717


namespace cos_identity_l390_390559

noncomputable def f (α : Real) : Real :=
  (Real.tan (π - α) * Real.cos (2 * π - α) * Real.sin (π / 2 + α)) / Real.cos (-α - π)

theorem cos_identity (α : Real) (h1 : f α = 4 / 5) (h2 : π / 2 < α ∧ α < π) : Real.cos (2 * α + π / 4) = (17 * Real.sqrt 2) / 50 := by
    sorry

end cos_identity_l390_390559


namespace min_red_hair_students_l390_390014

theorem min_red_hair_students (B N R : ℕ) 
  (h1 : B + N + R = 50)
  (h2 : N ≥ B - 1)
  (h3 : R ≥ N - 1) :
  R = 17 := sorry

end min_red_hair_students_l390_390014


namespace sin_transformation_identity_l390_390556

theorem sin_transformation_identity 
  (θ : ℝ) 
  (h : Real.cos (π / 12 - θ) = 1 / 3) : 
  Real.sin (2 * θ + π / 3) = -7 / 9 := 
by 
  sorry

end sin_transformation_identity_l390_390556


namespace part_a_part_b_l390_390670

-- Definition of the conditions
def n : ℕ := 2

noncomputable def A (i : ℕ) : set ℕ := sorry
def S (A_i : set ℕ) : ℕ := ∑ x in A_i, x

-- The arrangement problem for part (a)
theorem part_a (n : ℕ) (h : n ≥ 2) :
  ∃ (A : ℕ → set ℕ), 
    (∀ i, (|A i.succ| = |A i| + 1) ∨ (|A i| - 1 = |A i.succ|)) ∧ 
    (A (2^n + 1) = A 1) :=
sorry

-- The sum evaluation problem for part (b)
theorem part_b (n : ℕ) (h : n ≥ 2) (A : ℕ → set ℕ) 
  (arrangement_cond : (∀ i, (|A i.succ| = |A i| + 1) ∨ (|A i| - 1 = |A i.succ|)) ∧ 
                      (A (2^n + 1) = A 1)) :
  ∑ i in finset.range (2^n + 1), (-1)^i * S (A i) = 0 :=
sorry

end part_a_part_b_l390_390670


namespace area_of_circle_with_radius_2_is_4pi_l390_390587

theorem area_of_circle_with_radius_2_is_4pi :
  ∀ (π : ℝ), ∀ (r : ℝ), r = 2 → π > 0 → π * r^2 = 4 * π := 
by
  intros π r hr hπ
  sorry

end area_of_circle_with_radius_2_is_4pi_l390_390587


namespace required_speed_is_85_l390_390464

-- Definitions based on conditions
def speed1 := 60
def time1 := 3
def total_time := 5
def average_speed := 70

-- Derived conditions
def distance1 := speed1 * time1
def total_distance := average_speed * total_time
def remaining_distance := total_distance - distance1
def remaining_time := total_time - time1
def required_speed := remaining_distance / remaining_time

-- Theorem statement
theorem required_speed_is_85 : required_speed = 85 := by
    sorry

end required_speed_is_85_l390_390464


namespace range_of_a_l390_390561

def f (a x : ℝ) : ℝ := 
  if x < 1 then (3 - a) * x - a else Real.log a x

theorem range_of_a (a : ℝ) : 
  (∀ x < 1, (3 - a) > 0) →                          -- Condition 1
  (∀ x ≥ 1, Real.strictMono (Real.log a)) →       -- Condition 2
  Real.log a 1 = 0 →                           -- Condition 3
  (3 - 2 * a ≤ 0) →                            -- Condition 4
  a ∈ Set.Ico (3 / 2) 3 :=   
  sorry

end range_of_a_l390_390561


namespace shopkeeper_profit_percentage_l390_390825

theorem shopkeeper_profit_percentage
  (x : ℝ)
  (theft_loss_percent overall_loss_percent : ℝ)
  (theft_loss_percent_eq : theft_loss_percent = 40)
  (overall_loss_percent_eq : overall_loss_percent = 34)
  (cost_price_eq : x = 100) :
  let theft_loss := theft_loss_percent / 100 * x in
  let overall_loss := overall_loss_percent / 100 * x in
  let profit := theft_loss - overall_loss in
  let remaining_goods := x - theft_loss in
  (profit / remaining_goods * 100) = 10 :=
by
  sorry -- Proof skipped for the problem statement

end shopkeeper_profit_percentage_l390_390825


namespace sum_mean_median_mode_eq_38_over_9_l390_390782

-- Define the list of numbers
def numbers : List ℕ := [4, 2, 5, 4, 0, 4, 1, 0, 0]

-- Define functions to find the mean, median, and mode
def mean (l : List ℕ) : Rat :=
  let s := l.sum 
  let n := l.length
  s / n

def median (l : List ℕ) : ℕ :=
  match l.sort.batches with
  | [] => 0
  | mid :: _ => mid

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => max (l.count x) (l.count acc)) 0

-- Define a function to sum the mean, median, and mode
def sum_mean_median_mode (l : List ℕ) : Rat :=
  mean l + (median l) + (mode l)

-- Statement to prove
theorem sum_mean_median_mode_eq_38_over_9 : 
  sum_mean_median_mode numbers = 38 / 9 :=
by
  -- This is where the proof would go, but for now we use sorry.
  sorry

end sum_mean_median_mode_eq_38_over_9_l390_390782


namespace fall_increase_l390_390057

theorem fall_increase (x : ℝ) :
    (1 + x / 100) * (1 - 19 / 100) = 1 + 12.52 / 100 → x ≈ 38.91 :=
by
  sorry

end fall_increase_l390_390057


namespace A_is_false_l390_390885

variables {a b : ℝ}

-- Condition: Proposition B - The sum of the roots of the equation is 2
axiom sum_of_roots : ∀ (x1 x2 : ℝ), x1 + x2 = -a

-- Condition: Proposition C - x = 3 is a root of the equation
axiom root3 : ∃ (x1 x2 : ℝ), (x1 = 3 ∨ x2 = 3)

-- Condition: Proposition D - The two roots have opposite signs
axiom opposite_sign_roots : ∀ (x1 x2 : ℝ), x1 * x2 < 0

-- Prove: Proposition A is false
theorem A_is_false : ¬ (∃ x1 x2 : ℝ, x1 = 1 ∨ x2 = 1) :=
by
  sorry

end A_is_false_l390_390885


namespace rectangle_aspect_ratio_l390_390772

theorem rectangle_aspect_ratio (x y : ℝ) (h : x > 0 ∧ y > 0 ∧ x / y = 2 * y / x) : x / y = Real.sqrt 2 :=
by
  sorry

end rectangle_aspect_ratio_l390_390772


namespace conference_handshakes_l390_390068

theorem conference_handshakes (total_people : ℕ) (group1_people : ℕ) (group2_people : ℕ)
  (group1_knows_each_other : group1_people = 25)
  (group2_knows_no_one_in_group1 : group2_people = 15)
  (total_group : total_people = group1_people + group2_people)
  (total_handshakes : ℕ := group2_people * (group1_people + group2_people - 1) - group2_people * (group2_people - 1) / 2) :
  total_handshakes = 480 := by
  -- Placeholder for proof
  sorry

end conference_handshakes_l390_390068


namespace find_a_l390_390160

-- Define the condition that the sets are equal
theorem find_a (a : ℝ) (h : {2, -1} = {2, a^2 - 2a}) : a = 1 :=
sorry

end find_a_l390_390160


namespace center_of_array_is_seven_l390_390489

theorem center_of_array_is_seven (cell : Fin 3 × Fin 3 → ℕ)
  (h_all_unique : ∀ i j : Fin 3 × Fin 3, i ≠ j → cell i ≠ cell j)
  (h_consecutive_adjacency : ∀ i j : Fin 3 × Fin 3, abs (cell i - cell j) = 1 → (abs (i.1 - j.1) + abs (i.2 - j.2) = 1))
  (h_corners_sum : cell (0, 0) + cell (0, 2) + cell (2, 0) + cell (2, 2) = 20) :
  cell (1, 1) = 7 :=
by
  sorry

end center_of_array_is_seven_l390_390489


namespace projection_of_vector_l390_390921

variable (e₁ e₂ : ℝ)

-- Conditions
def unit_vectors (e₁ e₂ : ℝ) : Prop := e₁ * e₂ = 1 / 2

def angle_60_degrees (e₁ e₂ : ℝ) : Prop := unit_vectors e₁ e₂

-- Theorem statement
theorem projection_of_vector
  (h₁ : angle_60_degrees e₁ e₂) :
  let v := e₂ - 2 * e₁
  let u := e₁ + e₂
  let proj := (v * u) / real.norm u
  proj = - real.sqrt 3 / 2 := sorry

end projection_of_vector_l390_390921


namespace residue_neg_998_mod_28_l390_390086

theorem residue_neg_998_mod_28 : ∃ r : ℤ, r = -998 % 28 ∧ 0 ≤ r ∧ r < 28 ∧ r = 10 := 
by sorry

end residue_neg_998_mod_28_l390_390086


namespace least_possible_value_l390_390988

theorem least_possible_value (x : ℚ) (h1 : x > 5 / 3) (h2 : x < 9 / 2) : 
  (9 / 2 - 5 / 3 : ℚ) = 17 / 6 :=
by sorry

end least_possible_value_l390_390988


namespace probability_of_purple_marble_l390_390020

theorem probability_of_purple_marble (p_blue p_green p_purple : ℝ) 
  (h_blue : p_blue = 0.3) 
  (h_green : p_green = 0.4) 
  (h_sum : p_blue + p_green + p_purple = 1) : 
  p_purple = 0.3 := 
by 
  -- proof goes here
  sorry

end probability_of_purple_marble_l390_390020


namespace tenth_term_is_correct_l390_390434

-- Define the first term and common difference for the sequence
def a1 : ℚ := 1 / 2
def d : ℚ := 1 / 3

-- The property that defines the n-th term of the arithmetic sequence
def a (n : ℕ) : ℚ := a1 + (n - 1) * d

-- Statement to prove that the tenth term in the arithmetic sequence is 7 / 2
theorem tenth_term_is_correct : a 10 = 7 / 2 := 
by 
  -- To be filled in with the proof later
  sorry

end tenth_term_is_correct_l390_390434


namespace coffee_containers_used_l390_390724

theorem coffee_containers_used :
  let Suki_coffee := 6.5 * 22
  let Jimmy_coffee := 4.5 * 18
  let combined_coffee := Suki_coffee + Jimmy_coffee
  let containers := combined_coffee / 8
  containers = 28 := 
by
  sorry

end coffee_containers_used_l390_390724


namespace last_a_replacement_is_d_l390_390810

def message : String := "Alaska has areas as harsh as Sahara"

def count_occurrences (letter : Char) (msg : String) : Nat :=
  msg.toList.count (λ c => c = letter)

def sum_natural_numbers_up_to (n : Nat) : Nat :=
  n * (n + 1) / 2

def modulo_shift (shift : Nat) : Nat :=
  shift % 26

def shift_char (c : Char) (shift : Nat) : Char :=
  if c.is_lower then char.of_nat (((c.to_nat - 'a'.to_nat) + shift) % 26 + 'a'.to_nat)
  else if c.is_upper then char.of_nat (((c.to_nat - 'A'.to_nat) + shift) % 26 + 'A'.to_nat)
  else c

theorem last_a_replacement_is_d :
  let num_a := count_occurrences 'a' message
  let total_shift := sum_natural_numbers_up_to num_a
  let net_shift := modulo_shift total_shift
  let last_a_position := List.foldl
    (λ (pos, idx) c => if c = 'a' then (pos + 1, idx) else (pos, idx)) (0, 0) message.toList |> snd
  letter := shift_char 'a' net_shift in
  letter = 'd' := by
    sorry

end last_a_replacement_is_d_l390_390810


namespace households_using_only_brand_A_l390_390817

section

variables (T N AB : ℕ) 

-- Given conditions
def households_total := 160
def households_neither := 80
def households_both_brands := 5

-- Axiom for relationship and calculation
axiom ratio_b_condition : ∀ (B : ℕ), B = 3 * households_both_brands

-- Definition of variables aligned to the conditions
def B : ℕ := 3 * households_both_brands
def households_using_brands (A : ℕ) := households_total = households_neither + A + B + households_both_brands

-- Theorem to prove
theorem households_using_only_brand_A : (households_total = households_neither + A + B + households_both_brands) → A = 60 :=
begin
  -- By the axiom, we use the B value.
  have B_val : B = 3 * households_both_brands := ratio_b_condition B,
  -- Households_total = 80 + A + 15 + 5 => A = 60
  sorry
end

end

end households_using_only_brand_A_l390_390817


namespace max_intersections_circle_quadrilateral_l390_390777

/--
Given a circle and a quadrilateral, each side of the quadrilateral can intersect the circle at most twice.
A quadrilateral has four sides.
Prove the maximum number of points of intersection between the circle and the quadrilateral is 8.
-/
theorem max_intersections_circle_quadrilateral
  (circle : Type)
  (quadrilateral : Type)
  (intersect : circle → quadrilateral → ℕ) 
  (h1 : ∀ (c : circle) (s : quadrilateral), intersect c s ≤ 2)
  (h2 : ∀ (q : quadrilateral), (∃ (sides : fin 4 → quadrilateral), true))
  : ∃ n, n = 8 := 
sorry

end max_intersections_circle_quadrilateral_l390_390777


namespace range_of_PA_l390_390215

-- Define points, distances, and line segment
noncomputable def A : ℝ×ℝ := (0, 0)
noncomputable def B : ℝ×ℝ := (4, 0)

-- Define a function to calculate the distance between two points
def dist (P Q : ℝ×ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Condition: The sum of distances PA and PB is 6
def condition_on_P (P : ℝ × ℝ) : Prop :=
  dist P A + dist P B = 6

-- The range of values for |PA| is [1,5]
def range_PA (p : ℝ) : Prop :=
  1 ≤ p ∧ p ≤ 5

-- The main theorem to be proven
theorem range_of_PA (P : ℝ × ℝ) (h : condition_on_P P) : range_PA (dist P A) :=
  sorry

end range_of_PA_l390_390215


namespace volume_regions_correct_l390_390401

-- Definitions based on given conditions
def radius_small : ℝ := 4
def radius_medium : ℝ := 7
def radius_large : ℝ := 10

def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define volumes
def volume_small : ℝ := volume_sphere radius_small
def volume_medium : ℝ := volume_sphere radius_medium
def volume_large : ℝ := volume_sphere radius_large

-- Theorem statement without the proof
theorem volume_regions_correct :
  (volume_large - volume_medium = (2628 / 3) * Real.pi) ∧
  (volume_medium - volume_small = (1116 / 3) * Real.pi) :=
by
  -- Proof would go here
  sorry

end volume_regions_correct_l390_390401


namespace partI_partII_l390_390599

-- Define the function and the inequality condition
def h (x : ℝ) : ℝ := |x - 1| - |x + 2|

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

-- Define the solution set M
def M : set ℝ := { x : ℝ | -1/2 < x ∧ x < 1/2 }

-- Lean statement for the first question (proof not provided)
theorem partI (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  |(1/3) * a + (1/6) * b| < 1/4 := 
sorry

-- Lean statement for the second question (proof not provided)
theorem partII (a : ℝ) : 
  (∀ x : ℝ, f x - real.log (a^2 - 3 * a) / real.log 2 > 2) ↔
  (-1 < a ∧ a < 0) ∨ (3 < a ∧ a < 4) := 
sorry

end partI_partII_l390_390599


namespace right_triangle_tangent_circle_l390_390728

theorem right_triangle_tangent_circle :
  ∀ (D E F : Type) [InnerProductSpace ℝ E] [hd : MetricSpace D]
  (h_triangle : InnerProductSpace.OrthonormalBasis ι ℝ E)
  (h_right : inner E -ᵢ E = 0)
  (h_df : ∥D - F∥ = Real.sqrt 85)
  (h_de : ∥D - E∥ = 7)
  (circle_center : E = ∥t - t∥ → Metric.closedBall 0 t)
  (h_tangent_df : Metric.closedBall 0 t ↑Fd)
  (h_tangent_ef : Metric.closedBall 0 Fd) :
  ∥F - Q∥ = 6 :=
by
  sorry

end right_triangle_tangent_circle_l390_390728


namespace intersection_of_sets_l390_390935

-- Definition of the sets A and B
def setA : Set ℤ := {-1, 0, 1, 2}
def setB : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

-- Proof statement that A ∩ B = {0, 1}
theorem intersection_of_sets : (setA : Set ℝ) ∩ setB = {0, 1} :=
  by sorry

end intersection_of_sets_l390_390935


namespace otimes_main_l390_390884

-- Definitions for the proof statements
noncomputable def otimes (x y z : ℝ) (h : y ≠ z) : ℝ := x / (y - z)^2

theorem otimes_main :
  otimes (otimes 2 4 6 (by norm_num)) 
         (otimes 3 6 2 (by norm_num)) 
         (otimes 5 1 3 (by norm_num)) 
         (by {
            let y := otimes 3 6 2 (by norm_num);
            let z := otimes 5 1 3 (by norm_num);
            exact y ≠ z;
         }) 
  = 128 / 289 := sorry

end otimes_main_l390_390884


namespace ending_number_condition_l390_390759

theorem ending_number_condition (h : ∃ k : ℕ, k < 21 ∧ 100 < 19 * k) : ∃ n, 21.05263157894737 * 19 = n → n = 399 :=
by
  sorry  -- this is where the proof would go

end ending_number_condition_l390_390759


namespace AE_BF_intersect_circumcircle_of_square_l390_390829

theorem AE_BF_intersect_circumcircle_of_square
  (a : ℝ)
  (A B C D E F : (ℝ × ℝ))
  (h_square: (A = (0, 0)) ∧ (B = (a, 0)) ∧ (C = (a, a)) ∧ (D = (0, a)))
  (h_E_on_BC: E = (a, a / 3))
  (h_F_on_DC: F = (a / 2, a))
  (circumcenter : (ℝ × ℝ)) (circumradius : ℝ)
  (h_circumcenter: circumcenter = (a / 2, a / 2))
  (h_circumradius: circumradius = a * (Real.sqrt 2) / 2) :
  ∃ (X : (ℝ × ℝ)), 
    (X = (6 * a / 7, 2 * a / 7)) ∧ 
    (Real.dist X circumcenter = circumradius) := by
  sorry

end AE_BF_intersect_circumcircle_of_square_l390_390829


namespace problem1_problem2_problem3_l390_390242

variables {α β : ℝ} {a b c d : ℝ × ℝ}

-- Conditions
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sin α)
def vector_c (β : ℝ) : ℝ × ℝ := (Real.sin β, 2 * Real.cos β)
def vector_d (β : ℝ) : ℝ × ℝ := (Real.cos β, -2 * Real.sin β)

-- Problem 1: If a ⊥ b, find α
theorem problem1 (h : vector_a.1 * (2 * Real.cos α) + vector_a.2 * (Real.sin α) = 0) 
  (hα : 0 < α ∧ α < π) : α = π / 4 := by 
  sorry 

-- Problem 2: If |c + d| = √3, find the value of sin β + cos β
theorem problem2 (h : Real.sqrt ((Real.sin β + Real.cos β) ^ 2 + (2 * Real.cos β - 2 * Real.sin β) ^ 2) = Real.sqrt 3) 
  (hβ : π < β ∧ β < 2 * π) : Real.sin β + Real.cos β = -Real.sqrt 15 / 3 := by 
  sorry 

-- Problem 3: If tan α tan β = 4, prove that b ∥ c
theorem problem3 (h : Real.tan α * Real.tan β = 4) 
  (hα : 0 < α ∧ α < π) (hβ : π < β ∧ β < 2 * π) : ∃ k : ℝ, vector_b α = k • vector_c β := by 
  sorry 

end problem1_problem2_problem3_l390_390242


namespace tiles_needed_and_remaining_area_l390_390043

noncomputable theory

def tile_dimensions_in_feet : ℝ × ℝ := (1/3, 1/2)  -- Convert 4 inches to 1/3 feet and 6 inches to 1/2 feet
def room_dimensions_in_feet : ℝ × ℝ := (10, 15)   -- Room dimensions in feet
def tile_area : ℝ := tile_dimensions_in_feet.1 * tile_dimensions_in_feet.2  -- Calculate the area of each tile
def room_area : ℝ := room_dimensions_in_feet.1 * room_dimensions_in_feet.2  -- Calculate the total area of the room

def number_of_tiles_along_length : ℕ := int.floor (room_dimensions_in_feet.1 / tile_dimensions_in_feet.1)  -- Number of tiles along the length of the room
def number_of_tiles_along_breadth : ℕ := int.floor (room_dimensions_in_feet.2 / tile_dimensions_in_feet.2)  -- Number of tiles along the breadth of the room
def total_number_of_tiles : ℕ := number_of_tiles_along_length * number_of_tiles_along_breadth  -- Total number of tiles needed

def total_covered_area : ℝ := total_number_of_tiles * tile_area  -- Total area covered by tiles

theorem tiles_needed_and_remaining_area : (total_number_of_tiles = 900) ∧ (total_covered_area = room_area) :=
by
  -- Proof goes here
  sorry

end tiles_needed_and_remaining_area_l390_390043


namespace conical_pendulum_height_l390_390026

theorem conical_pendulum_height
  (L : ℝ) (m : ℝ) (T : ℝ) (g : ℝ)
  (h : ℝ)
  (h_def : h = L * (g / ((2 * Real.pi / T)^2 * L)))
  (L_val : L = 0.50) (m_val : m = 0.003) (T_val : T = 1.0) (g_val : g = 9.8) :
  h ≈ 0.062 :=
by
  sorry

end conical_pendulum_height_l390_390026


namespace translate_sin_to_cos_l390_390766

theorem translate_sin_to_cos :
  ∀ x : ℝ, 3 * sin (2 * (x - π / 8)) = 3 * cos (2 * x - π / 2) :=
by
  sorry

end translate_sin_to_cos_l390_390766


namespace cos_pi_minus_alpha_proof_l390_390647

-- Define the initial conditions: angle α and point P
def α : ℝ := arbitrary ℝ
def P : ℝ × ℝ := (-1, 2)

-- Noncomputable to define functions involving real number calculations
noncomputable def hypotenuse : ℝ := real.sqrt ((P.1)^2 + (P.2)^2)
noncomputable def cos_alpha : ℝ := P.1 / hypotenuse
noncomputable def cos_pi_minus_alpha : ℝ := -cos_alpha

-- The theorem to prove
theorem cos_pi_minus_alpha_proof :
  cos_pi_minus_alpha = real.sqrt 5 / 5 :=
by sorry

end cos_pi_minus_alpha_proof_l390_390647


namespace carla_paints_120_square_feet_l390_390061

def totalWork : ℕ := 360
def ratioAlex : ℕ := 3
def ratioBen : ℕ := 5
def ratioCarla : ℕ := 4
def ratioTotal : ℕ := ratioAlex + ratioBen + ratioCarla
def workPerPart : ℕ := totalWork / ratioTotal
def carlasWork : ℕ := ratioCarla * workPerPart

theorem carla_paints_120_square_feet : carlasWork = 120 := by
  sorry

end carla_paints_120_square_feet_l390_390061


namespace range_of_eccentricity_l390_390271

variables {a b c e : ℝ}

def ellipse (x y : ℝ) := (x^2 / a^2 + y^2 / b^2 = 1)
def upper_vertex := b > 0
def distance_condition (x_0 y_0 : ℝ) := (x_0^2 + (y_0 - b)^2 ≤ (2 * b) ^ 2)
def eccentricity := e = c / a
def ellipse_condition := a = real.sqrt (b^2 + c^2)

theorem range_of_eccentricity (h1 : a > b) (h2 : upper_vertex) 
  (h3 : ∀ x_0 y_0, ellipse x_0 y_0 → distance_condition x_0 y_0)
  (h4 : ellipse_condition) :
  0 < e ∧ e ≤ real.sqrt 2 / 2 := 
by
  sorry

end range_of_eccentricity_l390_390271


namespace minimize_length_AC_l390_390219

/-- Given a rectangle ABCD with AB = 3, AD = 4, and point P moving on sides AD and DC.
    Let θ be the angle ∠ABP. 
    Triangle ABP is folded along BP so that the dihedral angle A-BP-C is a right dihedral angle.
    Prove this angle θ is 45 degrees when the length of AC is minimized. -/
theorem minimize_length_AC (AB AD: ℝ) (θ: ℝ) (h_AB: AB = 3) (h_AD: AD = 4) 
  (P_on_AD_or_DC : P ∈ (AD ∪ DC)) 
  (right_dihedral_angle : ∠ABC + ∠DABP = π) :
  θ = 45 :=
sorry

end minimize_length_AC_l390_390219


namespace find_x_interval_l390_390624

noncomputable def log2 := real.logBase 2

theorem find_x_interval (x : ℝ) (h : log2 (x^2 - 5) < 0) : -real.sqrt 6 < x ∧ x < real.sqrt 6 :=
by
  sorry

end find_x_interval_l390_390624


namespace eccentricity_range_l390_390263

noncomputable def ellipse (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  { z : ℝ × ℝ // (z.1^2 / a^2) + (z.2^2 / b^2) = 1 }

theorem eccentricity_range (a b : ℝ) (h : a > b) (hb : b > 0):
  ∀ (P : ellipse a b h hb),
    |(P.1, P.2 - b).norm ≤ 2 * b → 
    eccentricity (a b) ∈ set.Ici (real.sqrt 2 / 2) :=
sorry

end eccentricity_range_l390_390263


namespace isosceles_triangle_third_side_l390_390397

theorem isosceles_triangle_third_side (a b c : ℝ) (h1 : triangle.is_isosceles a b c) (h2 : a = 2 ∨ a = 5) (h3 : b = 2 ∨ b = 5) (h4 : c ∈ {a, b}) : c = 5 :=
by
  sorry

end isosceles_triangle_third_side_l390_390397


namespace integer_part_sum_l390_390316

def integer_part (x : ℝ) : ℤ := floor x

theorem integer_part_sum :
  let x := 9.42 in
  integer_part x + integer_part (2 * x) + integer_part (3 * x) = 55 :=
by
  let x := 9.42
  have h1 : 2 * x = 18.84 := by norm_num
  have h2 : 3 * x = 28.26 := by norm_num
  have hx : integer_part x = 9 := by norm_num
  have h2x : integer_part (2 * x) = 18 := by norm_num
  have h3x : integer_part (3 * x) = 28 := by norm_num
  calc
    integer_part x + integer_part (2 * x) + integer_part (3 * x)
    = 9 + 18 + 28 : by rw [hx, h2x, h3x]
    ... = 55 : by norm_num

end integer_part_sum_l390_390316


namespace smallest_disks_required_l390_390694

-- Definitions for the problem conditions
def files_0_9_MB : Nat := 5
def files_0_6_MB : Nat := 15
def files_0_5_MB : Nat := 10
def disk_capacity_MB : Real := 1.44

-- Statement for the proof
theorem smallest_disks_required : 
  let total_files := files_0_9_MB + files_0_6_MB + files_0_5_MB in
  total_files > 0 -> 
  ∑ (i : Nat) in range files_0_9_MB, 0.9 + ∑ (i: Nat) in range files_0_6_MB, 0.6 + ∑ (i : Nat) in range files_0_5_MB, 0.5 <= 0.9 * files_0_9_MB + 0.6 * files_0_6_MB + 0.5 * files_0_5_MB ->
  exists (n : Nat), n = 13 := 
by
  sorry

end smallest_disks_required_l390_390694


namespace angle_between_vectors_l390_390163

noncomputable theory

variables {a b : EuclideanSpace ℝ (Fin 2)} -- Defining vectors in ℝ^2 for simplicity
variables (ha : ‖a‖ = sqrt 3) (hb : ‖b‖ = 2)
variables (h_perp : inner (a - b) a = 0)

theorem angle_between_vectors (a b : EuclideanSpace ℝ (Fin 2)) (ha : ‖a‖ = sqrt 3) (hb : ‖b‖ = 2) (h_perp : inner (a - b) a = 0) : 
  ∠ between a and b = real.pi / 6 :=
sorry

end angle_between_vectors_l390_390163


namespace lada_vs_elevator_l390_390666

def Lada_speed_ratio (V U : ℝ) (S : ℝ) : Prop :=
  (∃ t_wait t_wait' : ℝ,
  ((t_wait = 3*S/U - 3*S/V) ∧ (t_wait' = 7*S/(2*U) - 7*S/V)) ∧
   (t_wait' = 3 * t_wait)) →
  U = 11/4 * V

theorem lada_vs_elevator (V U : ℝ) (S : ℝ) : Lada_speed_ratio V U S :=
sorry

end lada_vs_elevator_l390_390666


namespace range_of_eccentricity_l390_390247

noncomputable def upperVertex (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : ℝ × ℝ := (0, b)

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem range_of_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h: a > b) :
  (∀ (x y : ℝ), ellipse a b x y → distance x y 0 b ≤ 2 * b) →
  ∃ (e : ℝ), e ∈ Set.Ioc 0 (Real.sqrt 2 / 2) ∧ a^2 = b^2 / (1 - e^2) := sorry

end range_of_eccentricity_l390_390247


namespace dot_product_of_vectors_is_3_l390_390967

namespace complex_to_vector

open Complex

def Z1 : ℂ := (1 - 2 * I) * I
def Z2 : ℂ := (1 - 3 * I) / (1 - I)

def vector_of_complex (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def a := vector_of_complex Z1
def b := vector_of_complex Z2

def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

theorem dot_product_of_vectors_is_3 : dot_product a b = 3 := by
  sorry

end complex_to_vector

end dot_product_of_vectors_is_3_l390_390967


namespace midpoint_distance_to_y_axis_l390_390600

theorem midpoint_distance_to_y_axis 
  (x1 x2 y1 y2 : ℝ)
  (parabola_eq : y1^2 = 4 * x1 ∧ y2^2 = 4 * x2)
  (focus_line : ∃ l, l.contains (1, 0) ∧ l.contains (x1, y1) ∧ l.contains (x2, y2))
  (sum_x : x1 + x2 = 3) : 
  (x1 + x2) / 2 = 3 / 2 :=
by
  sorry

end midpoint_distance_to_y_axis_l390_390600


namespace probability_of_region_l390_390039

theorem probability_of_region (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 6 ∧ 0 ≤ y ∧ y ≤ 2) :
  x < y ∧ x + y < 2 → (x, y) ∈ set.probability_space (set.rectangle 0 6 0 2) := 
-- Here, we ensure that the Lean theorem matches the conditions and the equivalent question-answer pair.
begin
  sorry
end

end probability_of_region_l390_390039


namespace function_domain_l390_390859

noncomputable def f (x : ℝ) := real.sqrt (x - 1) + real.cbrt (8 - x)

theorem function_domain : ∀ x, f x ≠ 0 ↔ 1 ≤ x :=
by sorry

end function_domain_l390_390859


namespace ticket_prices_l390_390050

theorem ticket_prices (x : ℕ) (h1 : ∃ n, x * n = 48) (h2 : ∃ m, x * m = 64) : 
  {x : ℕ | ∃ n, x * n = 48 ∧ ∃ m, x * m = 64}.card = 5 := 
by 
  sorry

end ticket_prices_l390_390050


namespace find_line_equation_l390_390749

theorem find_line_equation :
  ∃ (m : ℝ), ∃ (b : ℝ), (∀ x y : ℝ,
  (x + 3 * y - 2 = 0 → y = -1/3 * x + 2/3) ∧
  (x = 3 → y = 0) →
  y = m * x + b) ∧
  (m = 3 ∧ b = -9) :=
  sorry

end find_line_equation_l390_390749


namespace exists_equilateral_triangle_of_6_points_8_unit_distances_l390_390078

noncomputable theory

open_locale classical

def point (α : Type) := (α × α)

def distance {α : Type} [metric_space α] (a b : α) : ℝ := dist a b

def equilateral_triangle {α : Type} [metric_space α] (a b c : α) : Prop :=
  distance a b = distance b c ∧ distance b c = distance c a ∧ distance c a = distance a b

theorem exists_equilateral_triangle_of_6_points_8_unit_distances {α : Type} [metric_space α] 
  (points : fin 6 → point α) 
  (h_dist : ∃ p c : finset (fin 6), |p| = 6 ∧ 8 ≤ finset.card c ∧ ∀ i j ∈ p, i ≠ j → i ∈ c → distance (points i) (points j) = 1) : 
  ∃ (a b c : fin 6), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ equilateral_triangle (points a) (points b) (points c) :=
sorry

end exists_equilateral_triangle_of_6_points_8_unit_distances_l390_390078


namespace find_a_l390_390631

theorem find_a (x y a : ℝ) (h1 : x + 2 * y = 2) (h2 : 2 * x + y = a) (h3 : x + y = 5) : a = 13 := by
  sorry

end find_a_l390_390631


namespace a_6_value_l390_390603

noncomputable def a_n (n : ℕ) : ℚ :=
  if h : n > 0 then (3 * n - 2) / (2 ^ (n - 1))
  else 0

theorem a_6_value : a_n 6 = 1 / 2 :=
by
  -- placeholder for the proof
  sorry

end a_6_value_l390_390603


namespace smallest_prime_l390_390877

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ , m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n 

theorem smallest_prime :
  ∃ n : ℕ, n = 29 ∧ 
  n >= 10 ∧ n < 100 ∧
  is_prime n ∧
  ((n / 10) = 3) ∧ 
  is_composite (n % 10 * 10 + n / 10) ∧
  (n % 10 * 10 + n / 10) % 5 = 0 :=
by {
  sorry
}

end smallest_prime_l390_390877


namespace average_speed_l390_390658

theorem average_speed (d1 d2 d3 v1 v2 v3 total_distance total_time avg_speed : ℝ)
    (h1 : d1 = 40) (h2 : d2 = 20) (h3 : d3 = 10) 
    (h4 : v1 = 8) (h5 : v2 = 40) (h6 : v3 = 20) 
    (h7 : total_distance = d1 + d2 + d3)
    (h8 : total_time = d1 / v1 + d2 / v2 + d3 / v3) 
    (h9 : avg_speed = total_distance / total_time) : avg_speed = 11.67 :=
by 
  sorry

end average_speed_l390_390658


namespace larger_number_of_two_with_conditions_l390_390188

theorem larger_number_of_two_with_conditions (x y : ℕ) (h1 : x * y = 30) (h2 : x + y = 13) : max x y = 10 :=
by
  sorry

end larger_number_of_two_with_conditions_l390_390188


namespace remainder_of_binary_division_l390_390441

theorem remainder_of_binary_division : 
  (110110111101₂ % 4 = 1) :=
by sorry

end remainder_of_binary_division_l390_390441


namespace sum_tenth_powers_l390_390353

theorem sum_tenth_powers (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) (h3 : a^3 + b^3 = 4) (h4 : a^4 + b^4 = 7) (h5 : a^5 + b^5 = 11) : a^10 + b^10 = 123 :=
  sorry

end sum_tenth_powers_l390_390353


namespace range_of_eccentricity_l390_390250

noncomputable def upperVertex (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) : ℝ × ℝ := (0, b)

def ellipse (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2 + y^2 / b^2 = 1)

def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem range_of_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h: a > b) :
  (∀ (x y : ℝ), ellipse a b x y → distance x y 0 b ≤ 2 * b) →
  ∃ (e : ℝ), e ∈ Set.Ioc 0 (Real.sqrt 2 / 2) ∧ a^2 = b^2 / (1 - e^2) := sorry

end range_of_eccentricity_l390_390250


namespace number_of_correct_propositions_l390_390606

def plane := Type
def line := Type
def P : Type

variables (α β γ : plane) (a b c : line)

-- assumptions given in the problem
axiom H1 : α ≠ β ∧ α ≠ γ ∧ β ≠ γ
axiom H2 : α ∩ β = a
axiom H3 : α ∩ γ = b
axiom H4 : β ∩ γ = c

-- propositions to validate
def proposition1 := ∀ (a b c : line), a ⊥ b ∧ a ⊥ c → ¬b ⊥ c
def proposition2 := ∀ (a b : line), a ∩ b = P → a ∩ c = P
def proposition3 := ∀ (a b c : line), a ⊥ b ∧ a ⊥ c → α ⊥ γ
def proposition4 := ∀ (a b : line), a ∥ b → a ∥ c

-- statement to prove the number of correct propositions
theorem number_of_correct_propositions : 
  (proposition2 a b) ∧ (proposition3 a b c) ∧ (proposition4 a b) :=
sorry

end number_of_correct_propositions_l390_390606


namespace set_has_n_plus_one_elements_l390_390672

theorem set_has_n_plus_one_elements
  (p n : ℕ) (a : ℝ) 
  (hp_pos : 2 ≤ p) 
  (hn_pos : 0 < n) 
  (ha_bounds : 1 ≤ a ∧ a + n ≤ p) :
  ∃ S : finset ℕ, 
    S = finset.image (λ x, 
      (⌊real.log x / real.log 2⌋ + ⌊real.log x / real.log 3⌋ + ... + ⌊real.log x / real.log p⌋) (set.Icc a (a+n))) 
    ∧ S.card = n + 1 :=
sorry

end set_has_n_plus_one_elements_l390_390672


namespace exists_prime_q_and_positive_n_l390_390338

open Nat

theorem exists_prime_q_and_positive_n (p : ℕ) (hp_prime : Prime p) (hp_gt_5 : p > 5) :
  ∃ (q : ℕ) (hq_prime : Prime q) (hq_lt_p : q < p), ∃ (n : ℕ) (hn_pos : 0 < n), p ∣ (n ^ 2 - q) :=
sorry

end exists_prime_q_and_positive_n_l390_390338


namespace cover_disk_with_smaller_disks_l390_390367

-- Define the main problem
theorem cover_disk_with_smaller_disks 
  (D : set (ℝ × ℝ))
  (r : ℝ)
  (D_condition : D = { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≤ r^2 })
  (r : 2) 
  (r_small : 1) :
  ∃ (disk_centers : fin 7 → (ℝ × ℝ)), 
    ∀ (p : ℝ × ℝ), (p.1 ^ 2 + p.2 ^ 2 ≤ 4) →
    ∃ i : fin 7, ((p.1 - (disk_centers i).1) ^ 2 + (p.2 - (disk_centers i).2) ^ 2 ≤ 1) :=
begin
  sorry
end

end cover_disk_with_smaller_disks_l390_390367


namespace AndrasCanWin_l390_390497

-- Definitions: Dice and assignments
def Die := List ℕ

-- Specific dice assignments
def Die1 : Die := [18, 17, 6, 5, 4, 3]
def Die2 : Die := [16, 15, 14, 13, 2, 1]
def Die3 : Die := [12, 11, 10, 9, 8, 7]

-- Function to calculate András winning probability if he chooses Die3 and Béla Die1
def AndrasWinsProbabilityDie3AgainstDie1 : Prop :=
  (24 / 36 : ℚ) > 1 / 2

-- Function to calculate András winning probability if he chooses Die1 and Béla Die2
def AndrasWinsProbabilityDie1AgainstDie2 : Prop :=
  (20 / 36 : ℚ) > 1 / 2

-- Function to calculate András winning probability if he chooses Die2 and Béla Die3
def AndrasWinsProbabilityDie2AgainstDie3 : Prop :=
  (24 / 36 : ℚ) > 1 / 2

-- Proving the main problem, using the above probability conditions
theorem AndrasCanWin : Prop :=
  AndrasWinsProbabilityDie3AgainstDie1 ∧ AndrasWinsProbabilityDie1AgainstDie2 ∧ AndrasWinsProbabilityDie2AgainstDie3

#eval AndrasCanWin -- this evaluates to true if the theorem is correct

-- Initial stub for proof
example : AndrasCanWin := by sorry

end AndrasCanWin_l390_390497


namespace sum_simplification_l390_390789

/-- Given the sum:
  S(n) = sum (k=1 to n) (k+2) / (k! + (k+1)! + (k+2)!),
  we aim to prove:
  S(n) = 1/2 - 1/(n+2)!
-/
theorem sum_simplification (n : ℕ) : 
  (\sum k in Finset.range (n+1), (k + 2) / (Nat.factorial k + Nat.factorial (k + 1) + Nat.factorial (k + 2))) 
  = 1/2 - 1/Nat.factorial (n + 2) := 
sorry

end sum_simplification_l390_390789


namespace range_of_eccentricity_of_ellipse_l390_390260

-- Definitions for the problem.
def is_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1) 

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

def on_upper_vertex (b : ℝ) : (ℝ × ℝ) := (0, b)

-- The statement to prove the range of eccentricity.
theorem range_of_eccentricity_of_ellipse (a b : ℝ) (x y : ℝ) (e : ℝ) :
  is_on_ellipse a b x y →
  (distance x y 0 b ≤ 2 * b) →
  ∃ e, (e = (1 - (b^2 / a^2))^(1/2) ∧ (0 < e) ∧ (e ≤ (sqrt 2)/2)) :=
by
  sorry

end range_of_eccentricity_of_ellipse_l390_390260


namespace crows_and_trees_l390_390359

theorem crows_and_trees : ∃ (x y : ℕ), 3 * y + 5 = x ∧ 5 * (y - 1) = x ∧ x = 20 ∧ y = 5 :=
by
  sorry

end crows_and_trees_l390_390359


namespace smallest_prime_8_less_than_square_l390_390425

theorem smallest_prime_8_less_than_square :
  ∃ p : ℕ, (∃ n : ℤ, p = n^2 - 8) ∧ Nat.Prime p ∧ p > 0 ∧ (∀ q : ℕ, (∃ m : ℤ, q = m^2 - 8) ∧ Nat.Prime q → q ≥ p) :=
sorry

end smallest_prime_8_less_than_square_l390_390425


namespace boat_reaches_first_l390_390763

-- Defining the speeds in still water
def speed_A : ℝ := 42
def speed_B : ℝ := 36
def speed_C : ℝ := 48

-- Defining the rates of current
def current_A : ℝ := 4
def current_B : ℝ := 5
def current_C : ℝ := 6

-- Defining the distance downstream
def distance : ℝ := 60

-- Effective speeds of each boat downstream
def effective_speed_A : ℝ := speed_A + current_A
def effective_speed_B : ℝ := speed_B + current_B
def effective_speed_C : ℝ := speed_C + current_C

-- Times taken to reach 60 km downstream
def time_A : ℝ := distance / effective_speed_A
def time_B : ℝ := distance / effective_speed_B
def time_C : ℝ := distance / effective_speed_C

-- Main theorem stating the derived times and confirming which boat reaches first
theorem boat_reaches_first : time_A ≈ 1.3043 ∧ time_B ≈ 1.4634 ∧ time_C ≈ 1.1111 ∧ time_C < time_A ∧ time_C < time_B := 
by
  sorry -- Skipping the proof

end boat_reaches_first_l390_390763


namespace line_perpendicular_to_BC_passing_through_A_line_through_B_equidistant_l390_390604

theorem line_perpendicular_to_BC_passing_through_A (A B C : ℝ × ℝ)
  (hA : A = (4, 0)) (hB : B = (8, 10)) (hC : C = (0, 6)) :
  ∃ (m : ℝ), ∃ (b : ℝ), (-2 = m) ∧ (b = 8) ∧ (line_eq : 2 * A.1 + A.2 - 8 = 0):=
  sorry

theorem line_through_B_equidistant (A B C : ℝ × ℝ)
  (hA : A = (4, 0)) (hB : B = (8, 10)) (hC : C = (0, 6)) :
  ∃ (k1 k2 : ℝ), ((7 / 6 = k1 ∧ 7 * A.1 - 6 * A.2 + 4 = 0) ∨ (-3 / 2 = k2 ∧ 3 * A.1 + 2 * A.2 - 44 = 0)) :=
  sorry

end line_perpendicular_to_BC_passing_through_A_line_through_B_equidistant_l390_390604


namespace at_least_one_big_bear_or_little_bear_chosen_group_leader_l390_390400

theorem at_least_one_big_bear_or_little_bear_chosen_group_leader :
  let animals := ["Big Bear", "Little Bear", "Gigi", "Momo", "Bouncy", "Radish Head", "Tootoo"];
  let total_combinations := Nat.choose 7 2;
  let excluding_combinations := Nat.choose 5 2;
  let probability := 1 - (excluding_combinations.to_rat / total_combinations.to_rat)
  in probability = (11 : ℚ) / 21 :=
by
  sorry

end at_least_one_big_bear_or_little_bear_chosen_group_leader_l390_390400


namespace compute_P_at_9_and_neg5_l390_390009

noncomputable def P (x : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

axiom a_real : a ∈ ℝ
axiom b_real : b ∈ ℝ
axiom c_real : c ∈ ℝ
axiom d_real : d ∈ ℝ

axiom P_one : P 1 = 7
axiom P_two : P 2 = 52
axiom P_three : P 3 = 97

theorem compute_P_at_9_and_neg5 :
  (P 9 + P (-5)) / 4 = 1183 :=
sorry

end compute_P_at_9_and_neg5_l390_390009


namespace largest_square_side_length_l390_390656

theorem largest_square_side_length (side_length : ℝ) (triangle_vertex : ℝ) :
  side_length = 12 →
  triangle_vertex = 6 - real.sqrt 6 →
  largest_square_inscribed_side_length side_length triangle_vertex = 6 - real.sqrt 6 :=
by
  intros h_square_side h_triangle_vertex
  rw [h_square_side, h_triangle_vertex]
  sorry

def largest_square_inscribed_side_length (side_length : ℝ) (triangle_vertex : ℝ) : ℝ :=
  sorry

end largest_square_side_length_l390_390656


namespace part1_part2_part3_l390_390346

-- Define sequence with conditions
def sequence (a_n : ℕ → ℕ) : Prop :=
  a_n 1 = 1 ∧ ∀ n : ℕ, n > 0 → 
  2 * (∑ i in Finset.range n, a_n (i + 1)) / n = a_n (n + 1) - (1/3 : ℚ) * n^2 - n - (2/3 : ℚ)

-- Part 1: Prove that a2 = 4
theorem part1 (a_n : ℕ → ℕ) (h : sequence a_n) : 
  a_n 2 = 4 :=
sorry

-- Part 2: Prove general formula for the sequence
theorem part2 (a_n : ℕ → ℕ) (h : sequence a_n) : 
  ∀ n : ℕ, n > 0 → a_n n = n^2 :=
sorry

-- Part 3: Prove that the sum of reciprocals is less than 7/4
theorem part3 (a_n : ℕ → ℕ) (h : sequence a_n) : 
  ∀ n : ℕ, n > 0 → (∑ i in Finset.range n, (1 / (a_n (i + 1) : ℚ))) < (7 / 4 : ℚ) :=
sorry

end part1_part2_part3_l390_390346


namespace projection_of_a_on_b_l390_390941

theorem projection_of_a_on_b (
  a b : ℝ × ℝ
  (dot_product : ℝ)
  (h_dot_product : a.1 * b.1 + a.2 * b.2 = dot_product)
  (h_b : b = (3, 4))
  (h_dot : dot_product = 10)
) : (dot_product / (b.1 ^ 2 + b.2 ^ 2)) * b = (6 / 5, 8 / 5) :=
by {
  sorry
}

end projection_of_a_on_b_l390_390941


namespace ramu_profit_percent_l390_390712

-- Definitions of the given conditions
def usd_to_inr (usd : ℤ) : ℤ := usd * 45 / 10
def eur_to_inr (eur : ℤ) : ℤ := eur * 567 / 100
def jpy_to_inr (jpy : ℤ) : ℤ := jpy * 1667 / 10000

def cost_of_car_in_inr := usd_to_inr 10000
def engine_repair_cost_in_inr := eur_to_inr 3000
def bodywork_repair_cost_in_inr := jpy_to_inr 150000
def total_cost_in_inr := cost_of_car_in_inr + engine_repair_cost_in_inr + bodywork_repair_cost_in_inr

def selling_price_in_inr : ℤ := 80000
def profit_or_loss_in_inr : ℤ := selling_price_in_inr - total_cost_in_inr

-- Profit percent calculation
def profit_percent (profit_or_loss total_cost : ℤ) : ℚ := (profit_or_loss : ℚ) / (total_cost : ℚ) * 100

-- The theorem stating the mathematically equivalent problem
theorem ramu_profit_percent :
  profit_percent profit_or_loss_in_inr total_cost_in_inr = -8.06 := by
  sorry

end ramu_profit_percent_l390_390712


namespace closest_fraction_l390_390501

-- Define the given conditions
def medals_won : ℕ := 24
def total_medals : ℕ := 120
def possible_fractions : List ℚ := [1/4, 1/5, 1/6, 1/7, 1/8, 1/9]

-- State the theorem to be proved
theorem closest_fraction : abs ((24 / 120 : ℚ) - 1/5) ≤ abs ((24 / 120) - f) 
  for all f in possible_fractions :=
sorry

end closest_fraction_l390_390501


namespace positive_integers_count_l390_390491

theorem positive_integers_count :
  let numbers := [4 / 3, 1, 3.14, 0, 10 / 100, -4, 100]
  in (list.count (λ n, (n > 0) ∧ (n = int.of_nat (n.to_nat))) numbers) = 2 := 
by
  let numbers := [4 / 3, 1, 3.14, 0, 10 / 100, -4, 100]
  exact sorry

end positive_integers_count_l390_390491


namespace range_of_eccentricity_of_ellipse_l390_390261

-- Definitions for the problem.
def is_on_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1) 

def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2)

def on_upper_vertex (b : ℝ) : (ℝ × ℝ) := (0, b)

-- The statement to prove the range of eccentricity.
theorem range_of_eccentricity_of_ellipse (a b : ℝ) (x y : ℝ) (e : ℝ) :
  is_on_ellipse a b x y →
  (distance x y 0 b ≤ 2 * b) →
  ∃ e, (e = (1 - (b^2 / a^2))^(1/2) ∧ (0 < e) ∧ (e ≤ (sqrt 2)/2)) :=
by
  sorry

end range_of_eccentricity_of_ellipse_l390_390261


namespace sum_lambda_ellipse_l390_390143

theorem sum_lambda_ellipse (a b : ℝ) (x y : ℝ) (λ₁ λ₂ : ℝ) 
  (h₁ : x^2 / a^2 + y^2 / b^2 = 1)
  (h₂ : ∃ P : ℝ × ℝ, P = (0, y) ∧ (∃ A B : ℝ × ℝ, 
    A ≠ B ∧ A = (λ₁, y) ∧ B = (λ₂, y) ∧ (x, y) = (a, 0))) :
  λ₁ + λ₂ = 2 * a := 
sorry

end sum_lambda_ellipse_l390_390143


namespace monotonic_increasing_a_1_minimum_value_g_l390_390116

noncomputable def f (a x : ℝ) : ℝ := 2 * x ^ 3 - 3 * (a + 1) * x ^ 2 + 6 * a * x

theorem monotonic_increasing_a_1 (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → (f a x) ≤ (f a y)) ↔ a = 1 :=
begin
  sorry
end

theorem minimum_value_g (a : ℝ) (h : |a| > 1) :
  ∃ g : ℝ, (∀ x ∈ Icc 0 (2 * |a|), f a x ≥ g) ∧ 
    g = if a > 3 then a ^ 2 * (3 - a) 
        else if 1 < a ∧ a ≤ 3 then 0 
        else (3 * a - 1) :=
begin
  sorry
end

end monotonic_increasing_a_1_minimum_value_g_l390_390116


namespace probability_of_sum_greater_than_3_l390_390471

def is_palindromic (n : ℕ) : Prop :=
  let s := n.digits 10 in s = s.reverse

def three_digit_palindromic (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 150 ∧ is_palindromic n

def valid_palindromics : List ℕ :=
  List.filter (λ n => three_digit_palindromic n) (List.range 150)

theorem probability_of_sum_greater_than_3 :
  let palindromics := valid_palindromics
  let num_palindromics := palindromics.length
  let pairs := (palindromics.product palindromics).filter (λ p => p.fst < p.snd)
  let favorable_pairs := pairs.filter (λ p => (p.fst.digits 10).sum + (p.snd.digits 10).sum > 3)
  (favorable_pairs.length * 1.0 / pairs.length = 3.0 / 10.0) :=
by
  sorry

end probability_of_sum_greater_than_3_l390_390471


namespace vectors_are_coplanar_l390_390164

-- Definitions of the vectors a, b, and c.
def a (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -2)
def b : ℝ × ℝ × ℝ := (0, 1, 2)
def c : ℝ × ℝ × ℝ := (1, 0, 0)

-- The proof statement 
theorem vectors_are_coplanar (x : ℝ) 
  (h : ∃ m n : ℝ, a x = (n, m, 2 * m)) : 
  x = -1 :=
sorry

end vectors_are_coplanar_l390_390164


namespace part1_part2_part3_l390_390134

variable (a : ℝ)

def f (x : ℝ) : ℝ := (x^2 - 4) * (x - a)

theorem part1 : 
  deriv f x = 3 * x^2 - 2 * a * x - 4 := 
  sorry

theorem part2 (h : deriv f -1 = 0) : 
  a = 1 / 2 ∧ 
  (∀ x ∈ Icc (-2 : ℝ) 2, f x ≤ 9 / 2) ∧ 
  (∀ x ∈ Icc (-2 : ℝ) 2, f x ≥ -50 / 27) := 
  sorry

theorem part3 (h1 : ∀ x ∈ Icc (-∞ : ℝ) -2, 0 ≤ deriv f x) 
  (h2 : ∀ x ∈ Icc 2 (∞ : ℝ), 0 ≤ deriv f x) :
  -2 ≤ a ∧ a ≤ 2 := 
  sorry

end part1_part2_part3_l390_390134
