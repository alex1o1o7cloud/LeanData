import Mathlib
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Module.LinearMap
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.SimpleGraph.Connectivity
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Default
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.ProbabilityTheory.Basic
import Mathlib.Tactic
import Mathlib.Topology.InfiniteSum

namespace four_digit_number_count_l49_49204

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49204


namespace smaller_rectangle_area_l49_49323

theorem smaller_rectangle_area (L_h S_h : ℝ) (L_v S_v : ℝ) 
  (ratio_h : L_h = (8 / 7) * S_h) 
  (ratio_v : L_v = (9 / 4) * S_v) 
  (area_large : L_h * L_v = 108) :
  S_h * S_v = 42 :=
sorry

end smaller_rectangle_area_l49_49323


namespace pencil_distribution_l49_49475

theorem pencil_distribution (x : ℕ) 
  (Alice Bob Charles : ℕ)
  (h1 : Alice = 2 * Bob)
  (h2 : Charles = Bob + 3)
  (h3 : Bob = x)
  (total_pencils : 53 = Alice + Bob + Charles) : 
  Bob = 13 ∧ Alice = 26 ∧ Charles = 16 :=
by
  sorry

end pencil_distribution_l49_49475


namespace greatest_product_digit_count_l49_49678

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49678


namespace cube_volume_surface_area_l49_49013

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s^3 = 8 * x ∧ 6 * s^2 = 2 * x) → x = 0 :=
by
  sorry

end cube_volume_surface_area_l49_49013


namespace paths_from_A_to_D_l49_49134

theorem paths_from_A_to_D :
  let paths_A_to_B := 2,
      paths_B_to_C := 2,
      direct_paths_A_to_C := 1,
      paths_C_to_D := 2 in
  paths_A_to_B * paths_B_to_C * paths_C_to_D + direct_paths_A_to_C * paths_C_to_D = 10 :=
by
  sorry

end paths_from_A_to_D_l49_49134


namespace altitudes_bisect_angles_points_are_feet_of_altitudes_l49_49033

variable {A B C A1 B1 C1 : Type}
variables [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty A1] [Nonempty B1] [Nonempty C1]

-- Given conditions
axiom acute_angle_triangle (ABC: triangle A B C) : triangle.is_acute ABC
axiom points_on_sides (AB BC CA: Type) (C1 A1 B1: Type) 
          (on_AB: C1 ∈ AB) (on_BC: A1 ∈ BC) (on_CA: B1 ∈ CA) : True

axiom angle_conditions
    (B1_A1_C : Type) (B_A1_C1 : Type) (A1_B1_C : Type) (A_B1_C1 : Type)
    (A1_C1_B : Type) (A_C1_B1 : Type)
    (angle_eq1: angle B1_A1_C = angle B_A1_C1)
    (angle_eq2: angle A1_B1_C = angle A_B1_C1)
    (angle_eq3: angle A1_C1_B = angle A_C1_B1) : True

-- Prove part (a)
theorem altitudes_bisect_angles {ABC : Type}
    (AA1_perpendicular_BC : (AA1 : line A1 A) -> AA1 ⊥ BC)
    (BB1_perpendicular_CA : (BB1 : line B1 B) -> BB1 ⊥ CA)
    (CC1_perpendicular_AB : (CC1 : line C1 C) -> CC1 ⊥ AB)
    : ∀ (A1 B1 C1 : Type) (triangle A1 B1 C1 : Type),
    ∃ (b1 b2 b3: Type), 
    is_bisector (line AA1) (angle C1 A1 B) ∧
    is_bisector (line BB1) (angle A1 B1 A) ∧ 
    is_bisector (line CC1) (angle A1 C1 B)
:= sorry

-- Prove part (b)
theorem points_are_feet_of_altitudes {ABC : Type}
    (points_on_sides : points_on_sides (AB BC CA) (C1 A1 B1) True)
    (angle_conditions : angle_conditions 
        (B1_A1_C B_A1_C1 A1_B1_C A_B1_C1 A1_C1_B A_C1_B1)
        True True True)
    : are_feet_of_altitudes A1 B1 C1 ABC 
:= sorry

end altitudes_bisect_angles_points_are_feet_of_altitudes_l49_49033


namespace three_four_five_six_solution_l49_49365

-- State that the equation 3^x + 4^x = 5^x is true when x=2
axiom three_four_five_solution : 3^2 + 4^2 = 5^2

-- We need to prove the following theorem
theorem three_four_five_six_solution : 3^3 + 4^3 + 5^3 = 6^3 :=
by sorry

end three_four_five_six_solution_l49_49365


namespace largest_cos_a_l49_49383

variable {a b c d : Real}

def sin_eq_cot (x y : Real) : Prop := Real.sin x = Real.cot y

theorem largest_cos_a
  (h1 : sin_eq_cot a b)
  (h2 : sin_eq_cot b c)
  (h3 : sin_eq_cot c d)
  (h4 : sin_eq_cot d a) :
  Real.cos a = (Real.sqrt (Real.sqrt 5 - 1)) / 2 := sorry

end largest_cos_a_l49_49383


namespace four_digit_number_count_l49_49239

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49239


namespace greatest_possible_digits_in_product_l49_49506

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49506


namespace cos_of_three_pi_div_two_l49_49115

theorem cos_of_three_pi_div_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  sorry

end cos_of_three_pi_div_two_l49_49115


namespace four_digit_numbers_count_l49_49218

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49218


namespace max_oranges_to_teachers_l49_49052

theorem max_oranges_to_teachers {n r : ℕ} (h1 : n % 8 = r) (h2 : r < 8) : r = 7 :=
sorry

end max_oranges_to_teachers_l49_49052


namespace arctan_sum_pi_div_two_l49_49821

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l49_49821


namespace mo_hot_chocolate_l49_49416

noncomputable def cups_of_hot_chocolate (total_drinks: ℕ) (extra_tea: ℕ) (non_rainy_days: ℕ) (tea_per_day: ℕ) : ℕ :=
  let tea_drinks := non_rainy_days * tea_per_day 
  let chocolate_drinks := total_drinks - tea_drinks 
  (extra_tea - chocolate_drinks)

theorem mo_hot_chocolate :
  cups_of_hot_chocolate 36 14 5 5 = 11 :=
by
  sorry

end mo_hot_chocolate_l49_49416


namespace eccentricity_of_ellipse_fixed_points_of_sum_of_slopes_is_constant_l49_49169

variable (a b c x y t : ℝ)
variable (F M P : ℝ × ℝ)

def ellipse (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def is_symmetric (M F : (ℝ × ℝ)) (a : ℝ) : Prop :=
  (M.2 - F.2) = F.1 - M.1 + a

def lies_on_line (M : (ℝ × ℝ)) (line : ℝ × ℝ) : Prop :=
  line.1 * M.fst + line.2 * M.snd = 0

def eccentricity (a c : ℝ) : ℝ :=
  c / a

def intersection_chord_length (a b x y : ℝ) : Prop :=
  exists (x1 x2 : ℝ), (x1 - x2)^2 + y^2 = 9

def sum_of_slopes_is_constant (A B P : (ℝ × ℝ)) : Prop :=
  let k_PA := (A.snd - P.snd) / (A.fst - P.fst)
  let k_PB := (B.snd - P.snd) / (B.fst - P.fst)
  k_PA + k_PB = 0

theorem eccentricity_of_ellipse
  (h1 : a > b) (h2 : b > 0) (h3 : a = 2 * c)
  (h4 : F = (c, 0)) (h5 : M = (-a, a + c))
  (h6 : lies_on_line M (3, 2)) :
  eccentricity a c = 1 / 2 := 
sorry

theorem fixed_points_of_sum_of_slopes_is_constant
  (h1 : a = 2) (h2 : b = sqrt 3)
  (h3 : ellipse 2 (sqrt 3) x y)
  (h4 : intersection_chord_length a b x y)
  (h5 : lies_on_line M (3, 2)) :
  exists P : (ℝ × ℝ), 
  sum_of_slopes_is_constant (1, 3 / 2) (-1, -3 / 2) P := 
sorry

end eccentricity_of_ellipse_fixed_points_of_sum_of_slopes_is_constant_l49_49169


namespace intersect_or_parallel_l49_49139

-- Define the conditions as Lean terms
variables {A B C P A1 A2 A3 B1 B2 C1 C2 : Point}

-- Define that PA1 and PA2 are perpendicular to BC, and similarly define the other points
axiom PA1_perp_BC : is_perpendicular (line_through P A1) (line_through B C)
axiom PA2_perp_BC : is_perpendicular (line_through P A2) (line_through B C)
axiom PA3_perp_AA3 : is_perpendicular (line_through P A3) (line_through A A3)

-- Similarly for B and C points
axiom PB1_perp_AC : is_perpendicular (line_through P B1) (line_through A C)
axiom PB2_perp_AC : is_perpendicular (line_through P B2) (line_through A C)
axiom PC1_perp_AB : is_perpendicular (line_through P C1) (line_through A B)
axiom PC2_perp_AB : is_perpendicular (line_through P C2) (line_through A B)

-- State the final theorem based on the problem and solution analysis
theorem intersect_or_parallel (A1 A2 B1 B2 C1 C2 : Point) :
  lines_intersect_or_parallel A1 A2 B1 B2 C1 C2 := sorry

end intersect_or_parallel_l49_49139


namespace variance_sleep_duration_l49_49760

theorem variance_sleep_duration : 
  let durations := [6, 6, 7, 6, 7, 8, 9]
  let avg := 7
  let n := list.length durations
  let variance := (list.sum (list.map (fun x => (x - avg) ^ 2) durations : ℝ)) / n
  variance = 8 / 7 := by
  sorry

end variance_sleep_duration_l49_49760


namespace minimal_P_l49_49900

variables {P Q M P' Q' l : Type*} [Nonempty P] [Nonempty Q]
variables {line : P → Q → l}
variables {points_on_same_side : ∀ P Q l, P ≠ Q → ¬ ∃ M ∈ l, M ∈ segment PQ}
variables {altitudes : ∀ M l P Q, ∃ P' Q', altitude P' P M ∧ altitude Q' Q M}
variables {circle_diameter_touch_intersect_line : ∀ PQ l, ∃ M, circle_with_diameter PQ ∩ l = {M}}

theorem minimal_P'Q'_length (h1 : point_on_same_side P Q l) (h2 : altitude P' P M) (h3 : altitude Q' Q M) :
  same_length P' Q' :=
sorry

end minimal_P_l49_49900


namespace greatest_product_digits_l49_49657

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49657


namespace solve_for_x_l49_49341

-- Define the necessary condition
def problem_statement (x : ℚ) : Prop :=
  x / 4 - x - 3 / 6 = 1

-- Prove that if the condition holds, then x = -14/9
theorem solve_for_x (x : ℚ) (h : problem_statement x) : x = -14 / 9 :=
by
  sorry

end solve_for_x_l49_49341


namespace axis_of_symmetry_of_parabola_l49_49167

theorem axis_of_symmetry_of_parabola (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a * 1 ^ 2 + b * 1 + c = 0) (h₂ : a * 5 ^ 2 + b * 5 + c = 0) :
  axis_of_symmetry (λ x : ℝ, a * x ^ 2 + b * x + c) = 3 :=
sorry

def axis_of_symmetry (f : ℝ → ℝ) : ℝ :=
(sorry : ℝ)  -- Placeholder; you would normally define the axis of symmetry here.

end axis_of_symmetry_of_parabola_l49_49167


namespace number_of_four_digit_numbers_l49_49298

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49298


namespace greatest_digits_product_l49_49581

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49581


namespace max_digits_in_product_l49_49491

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49491


namespace four_digit_number_count_l49_49237

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49237


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49634

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49634


namespace part1_geometric_progression_part2_range_of_t_l49_49095

-- Definitions from the problem conditions
def sequence_an : ℕ → ℝ
| 0     := 3
| (n+1) := 2 / (sequence_an n + 1)

def sequence_Qn (n : ℕ) : ℝ :=
  (sequence_an n - 1) / (sequence_an n + 2)

-- Part (1): Sequence is a geometric progression with ratio -1/2
theorem part1_geometric_progression :
  (∀ n, ∃ r, sequence_Qn (n + 1) = r * sequence_Qn n) :=
sorry

-- Part (2): Range of t given the inequality holds
theorem part2_range_of_t (t : ℝ) :
  (∀ n : ℕ, ∀ m ∈ set.Icc (-1 : ℝ) 1, sequence_an n - t^2 - m * t ≥ 0) →
  t ∈ set.Icc ((1 - real.sqrt 3) / 2) ((real.sqrt 3 - 1) / 2) :=
sorry

end part1_geometric_progression_part2_range_of_t_l49_49095


namespace find_ellipse_equation_find_line_slopes_sum_slopes_constant_l49_49925

open real

noncomputable def ellipse (a b : ℝ) := ∃ x y : ℝ, (x, y) (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def intersect_line_ellipse (m : ℝ) := ∃ x₁ y₁ x₂ y₂ : ℝ,
  ((y₁ = (1/2) * x₁ + m) ∧ (y₂ = (1/2) * x₂ + m) ∧
  (x₁^2 / 8 + y₁^2 / 2 = 1) ∧ (x₂^2 / 8 + y₂^2 / 2 = 1))

noncomputable def line_slope_through_point (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ) := (y₁ - y₀) / (x₁ - x₀)

theorem find_ellipse_equation (a b : ℝ) (h₁ : ellipse a b)
  (h₂ : a + b = 3 * sqrt 2) (h₃ : b = sqrt 2) : a = 2 * sqrt 2 ∧ a^2 / 8 + b^2 / 2 = 1 :=
begin
  sorry
end

theorem find_line_slopes
  ((x₁ y₁) (x₂ y₂) : ℝ) 
  (h₁ : intersect_line_ellipse sqrt 2) 
  (h₂ : (x₁, y₁) = (0, sqrt 2)) 
  (h₃ : (x₂, y₂) = (-2 * sqrt 2, 0)) 
  : line_slope_through_point x₁ y₁ 2 1 = - ((sqrt 2 - 1) / 2) ∧
    line_slope_through_point x₂ y₂ 2 1 = ((sqrt 2 - 1) / 2) :=
begin
  sorry
end

theorem sum_slopes_constant (m : ℝ) (h₁ : intersect_line_ellipse m)
  : ∀ x₁ y₁ x₂ y₂ : ℝ, 
  (x₁ + x₂ = -2 * m) ∧ (x₁ * x₂ = 2 * m^2 - 4) →
  (k₁ + k₂ = 0) :=
begin
  sorry
end

end find_ellipse_equation_find_line_slopes_sum_slopes_constant_l49_49925


namespace ash_cloud_radius_l49_49771

theorem ash_cloud_radius
  (diam_ratio : ℝ)
  (height : ℝ)
  (diam_ratio_eq : diam_ratio = 18)
  (height_eq : height = 300) :
  let diameter := diam_ratio * height in
  let radius := diameter / 2 in
  radius = 2700 :=
by sorry

end ash_cloud_radius_l49_49771


namespace supplement_of_angle_l49_49966

theorem supplement_of_angle (complement_of_angle : ℝ) (h1 : complement_of_angle = 30) :
  ∃ (angle supplement_angle : ℝ), angle + complement_of_angle = 90 ∧ angle + supplement_angle = 180 ∧ supplement_angle = 120 :=
by
  sorry

end supplement_of_angle_l49_49966


namespace expression_evaluation_l49_49954

theorem expression_evaluation (m n : ℤ) (h : m * n = m + 3) : 2 * m * n + 3 * m - 5 * m * n - 10 = -19 := 
by 
  sorry

end expression_evaluation_l49_49954


namespace four_digit_numbers_count_l49_49230

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49230


namespace incorrect_statement_l49_49329

theorem incorrect_statement (a x y : ℝ) (hx1 : 0 < x) (hx2 : x < 1) (ha : 1 < a)
  (hxy : y = log a x) : 
  ¬ (∀ s, s = "Only some of the above statements are correct" ∨ 
          (s = "If x = 1, y = 0" ∧ y = log a 1) ∨ 
          (s = "If x = a, y = 1" ∧ y = log a a) ∨ 
          (s = "If x = -1, y is imaginary (complex)" ∧ complex.log (a : ℂ) (-1) = complex.log (a : ℂ) (-1)) ∨ 
          (s = "If 0 < x < 1, y is always less than 0 and decreases without limit as x approaches zero" ∧ 
           x > 0 ∧ x < 1 ∧ log a x < 0 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < x ∧ x < δ → log a x < -ε)) :=
begin
  sorry
end

end incorrect_statement_l49_49329


namespace seq_a_formula_sum_seq_b_l49_49901

-- Definitions based on conditions
def seq_a (n : ℕ) : ℕ := 2^n
def seq_b (n : ℕ) : ℕ := seq_a n * log (1/2) (seq_a n)

-- Question (Ⅰ)
theorem seq_a_formula : ∀ n : ℕ, seq_a n = 2^n := 
sorry

-- Question (Ⅱ)
theorem sum_seq_b (n : ℕ) : 
  (∑ i in finset.range n, seq_b (i + 1)) = (n - 1) * 2 ^ (n + 1) + 2 :=
sorry

end seq_a_formula_sum_seq_b_l49_49901


namespace four_digit_numbers_count_l49_49216

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49216


namespace max_digits_of_product_l49_49550

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49550


namespace greatest_possible_number_of_digits_in_product_l49_49526

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49526


namespace tangent_identity_proof_l49_49374

open Real

variable {x y z : ℝ}

theorem tangent_identity_proof (hx: |x| ≠ 1/√3) (hy: |y| ≠ 1/√3) (hz: |z| ≠ 1/√3) (h: x + y + z = x * y * z) :
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2) =
  ((3 * x - x^3) / (1 - 3 * x^2)) * ((3 * y - y^3) / (1 - 3 * y^2)) * ((3 * z - z^3) / (1 - 3 * z^2)) :=
sorry

end tangent_identity_proof_l49_49374


namespace abs_five_minus_sqrt_pi_l49_49802

theorem abs_five_minus_sqrt_pi : |5 - Real.sqrt Real.pi| = 3.22755 := by
  sorry

end abs_five_minus_sqrt_pi_l49_49802


namespace greatest_number_of_digits_in_product_l49_49689

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49689


namespace arctan_sum_pi_div_two_l49_49810

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l49_49810


namespace circle_eq_l49_49056

theorem circle_eq 
    (center : ℝ × ℝ) (h_center : center = (1, 1)) 
    (line_eq : ∀ x y : ℝ, x + y = 4 → Prop)
    (chord_length : ℝ) (h_chord_length : chord_length = 2 * real.sqrt 3) :
    ∃ (r : ℝ), (λ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = r^2) 5 :=
begin
    sorry
end

end circle_eq_l49_49056


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49711

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49711


namespace greatest_possible_number_of_digits_in_product_l49_49530

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49530


namespace four_digit_numbers_count_eq_l49_49260

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49260


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49640

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49640


namespace triangle_perimeter_l49_49070

theorem triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) 
  (h1 : area = 150)
  (h2 : leg1 = 30)
  (h3 : 0 < leg2)
  (h4 : hypotenuse = (leg1^2 + leg2^2).sqrt)
  (hArea : area = 0.5 * leg1 * leg2)
  : hypotenuse = 10 * Real.sqrt 10 ∧ leg2 = 10 ∧ (leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10) := 
by
  sorry

end triangle_perimeter_l49_49070


namespace common_fraction_equiv_l49_49489

noncomputable def decimal_equivalent_frac : Prop :=
  ∃ (x : ℚ), x = 413 / 990 ∧ x = 0.4 + (7/10^2 + 1/10^3) / (1 - 1/10^2)

theorem common_fraction_equiv : decimal_equivalent_frac :=
by
  sorry

end common_fraction_equiv_l49_49489


namespace prob_event_A_given_B_l49_49726

def EventA (visits : Fin 4 → Fin 4) : Prop :=
  Function.Injective visits

def EventB (visits : Fin 4 → Fin 4) : Prop :=
  visits 0 = 0

theorem prob_event_A_given_B :
  ∀ (visits : Fin 4 → Fin 4),
  (∃ f : (Fin 4 → Fin 4) → Prop, f visits → (EventA visits ∧ EventB visits)) →
  (∃ P : ℚ, P = 2 / 9) :=
by
  intros visits h
  -- Proof omitted
  sorry

end prob_event_A_given_B_l49_49726


namespace greatest_product_digit_count_l49_49685

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49685


namespace four_digit_numbers_count_l49_49250

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49250


namespace midpoint_of_segment_l49_49001

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem midpoint_of_segment :
  midpoint (-5, 8) (11, -3) = (3, 2.5) :=
by
  sorry

end midpoint_of_segment_l49_49001


namespace smallest_d_for_inverse_l49_49386

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 4

theorem smallest_d_for_inverse :
  ∃ d : ℝ, (∀ x1 x2 : ℝ, d ≤ x1 → d ≤ x2 → g x1 = g x2 → x1 = x2) ∧ d = 3 :=
by {
  -- skip to avoid proof steps and assure lean file build success
  use 3,
  sorry,
}

end smallest_d_for_inverse_l49_49386


namespace train_speed_is_117_l49_49031

def speed_of_man_kmph : ℝ := 3
def speed_of_man_mps : ℝ := (speed_of_man_kmph * 1000) / 3600
def train_length : ℝ := 300
def time_to_cross : ℝ := 9
def relative_speed : ℝ := train_length / time_to_cross
def train_speed_mps : ℝ := relative_speed - speed_of_man_mps
def train_speed_kmph : ℝ := train_speed_mps * (3600 / 1000)

theorem train_speed_is_117 :
  train_speed_kmph = 117 := by
  sorry

end train_speed_is_117_l49_49031


namespace largest_d_dividing_factorial_l49_49027

theorem largest_d_dividing_factorial (p : ℕ) (h_prime : Nat.Prime p) : 
  ∃ d, (∏ i in Finset.range (p^4 + 1), i).factorization p = d ∧ d = (p^4 - 1) / (p - 1) :=
by
  sorry

end largest_d_dividing_factorial_l49_49027


namespace exists_even_a_l49_49337

def f (a x : ℝ) : ℝ := x^2 + a * x

theorem exists_even_a : ∃ a : ℝ, ∀ x : ℝ, f a x = f a (-x) := by
  use 0
  intros x
  simp [f]
  sorry

end exists_even_a_l49_49337


namespace find_a8_l49_49981

-- Definitions for the problem
variable {a : ℕ → ℝ}  -- an arithmetic sequence

-- Conditions as given in the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_first_n (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n / 2) * (a 0 + a (n - 1))

-- Condition: The sum of the first 15 terms is 90
def S15_sum_condition : Prop :=
  sum_first_n 15 a = 90

-- Theorem: Prove a_8 = 6 given the conditions
theorem find_a8 (h_arith : is_arithmetic_sequence a) (h_sum : S15_sum_condition) : a 7 = 6 := sorry

end find_a8_l49_49981


namespace arithmetic_sequences_ratio_l49_49144

theorem arithmetic_sequences_ratio (x y a1 a2 a3 b1 b2 b3 b4 : Real) (hxy : x ≠ y) 
  (h_arith1 : a1 = x + (y - x) / 4 ∧ a2 = x + 2 * (y - x) / 4 ∧ a3 = x + 3 * (y - x) / 4 ∧ y = x + 4 * (y - x) / 4)
  (h_arith2 : b1 = x - (y - x) / 2 ∧ b2 = x + (y - x) / 2 ∧ b3 = x + 2 * (y - x) / 2 ∧ y = x + 2 * (y - x) / 2 ∧ b4 = y + (y - x) / 2):
  (b4 - b3) / (a2 - a1) = 8 / 3 := 
sorry

end arithmetic_sequences_ratio_l49_49144


namespace max_digits_in_product_l49_49498

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49498


namespace four_digit_number_count_l49_49240

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49240


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49837

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49837


namespace greatest_number_of_digits_in_product_l49_49615

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49615


namespace area_of_45_45_90_triangle_l49_49458

theorem area_of_45_45_90_triangle (h : hypotenuse = 10 * sqrt 2) (θ : angle = 45) :
  area = 50 := 
sorry

end area_of_45_45_90_triangle_l49_49458


namespace residue_neg_811_mod_24_l49_49103

theorem residue_neg_811_mod_24 :
  ∃ r : ℤ, 0 ≤ r ∧ r < 24 ∧ (-811) % 24 = r ∧ r = 5 :=
by
  use 5
  simp
  sorry

end residue_neg_811_mod_24_l49_49103


namespace digits_of_product_l49_49944

theorem digits_of_product (a : ℕ) (b : ℕ) (ha : a = 3^6) (hb : b = 2^{15}) :
  (⌊ log 10 (a * b) ⌋ + 1) = 8 :=
by
-- Initial assumptions from the problem
  have h3 : a = 729, from ha,
  have h2 : b = 32768, from hb,
  -- Sorry to skip the proof
  sorry

end digits_of_product_l49_49944


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49836

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49836


namespace distances_equal_l49_49967

noncomputable def distance_from_point_to_line (x y m : ℝ) : ℝ :=
  |m * x + y + 3| / Real.sqrt (m^2 + 1)

theorem distances_equal (m : ℝ) :
  distance_from_point_to_line 3 2 m = distance_from_point_to_line (-1) 4 m ↔
  (m = 1 / 2 ∨ m = -6) := 
sorry

end distances_equal_l49_49967


namespace triangle_angle_C_l49_49998

theorem triangle_angle_C (A B C : ℝ) (h1 : A = 86) (h2 : B = 3 * C + 22) (h3 : A + B + C = 180) : C = 18 :=
by
  sorry

end triangle_angle_C_l49_49998


namespace cube_x_value_l49_49006

noncomputable def cube_side_len (x : ℝ) : ℝ := (8 * x) ^ (1 / 3)

lemma cube_volume (x : ℝ) : (cube_side_len x) ^ 3 = 8 * x :=
  by sorry

lemma cube_surface_area (x : ℝ) : 6 * (cube_side_len x) ^ 2 = 2 * x :=
  by sorry

theorem cube_x_value (x : ℝ) (hV : (cube_side_len x) ^ 3 = 8 * x) (hS : 6 * (cube_side_len x) ^ 2 = 2 * x) : x = sqrt 3 / 72 :=
  by sorry

end cube_x_value_l49_49006


namespace circle_zero_rational_points_circle_one_rational_point_circle_two_rational_points_circle_three_implies_infinitely_many_rational_l49_49425

-- Definition of rational point
def is_rational_point (p : ℚ × ℚ) : Prop :=
  ∃ (x y : ℚ), p = (x, y)

-- Circle equation with no rational points
theorem circle_zero_rational_points :
  ∀ p : ℚ × ℚ, p.1^2 + p.2^2 ≠ real.sqrt 2 := 
begin
  intro p,
  sorry
end

-- Circle equation with exactly one rational point (0, 0)
theorem circle_one_rational_point : 
  ∀ p : ℚ × ℚ, (p.1 - real.sqrt 2)^2 + p.2^2 = 2 ↔ p = (0, 0) := 
begin
  intro p,
  sorry
end

-- Circle equation with exactly two rational points (1, 0) and (-1, 0)
theorem circle_two_rational_points : 
  ∀ p : ℚ × ℚ, (p.1^2 + (p.2 - real.sqrt 2)^2 = 3) ↔ (p = (1, 0) ∨ p = (-1, 0)) := 
begin
  intro p,
  sorry
end

-- Circle with exactly 3 rational points implies infinitely many rational points
theorem circle_three_implies_infinitely_many_rational :
  ∀ (p1 p2 p3 : ℚ × ℚ), 
  (∃ r : ℝ, ∀ q : ℚ × ℚ, (q.1^2 + q.2^2 = r)) → 
  ((p1.1^2 + p1.2^2 = r) ∧ (p2.1^2 + p2.2^2 = r) ∧ (p3.1^2 + p3.2^2 = r)) → 
  ∃ s : ℚ × ℚ, s ≠ p1 ∧ s ≠ p2 ∧ s ≠ p3 ∧ is_rational_point s := 
begin
  intros p1 p2 p3 hr h,
  sorry
end

end circle_zero_rational_points_circle_one_rational_point_circle_two_rational_points_circle_three_implies_infinitely_many_rational_l49_49425


namespace number_of_triangles_in_figure_l49_49946

-- Define the conditions and problem
def main_rectangle := "A figure consisting of a main rectangle partitioned into smaller rectangles and triangles by vertical, horizontal, and diagonal lines."

-- The proof problem
theorem number_of_triangles_in_figure : 
  ∃ n : ℕ, n = 54 ∧ 
      figure_partitioned_by_lines_and_diagonals = main_rectangle →
      count_triangles_in_figure main_rectangle = n := 
by 
  sorry

end number_of_triangles_in_figure_l49_49946


namespace number_of_four_digit_numbers_l49_49273

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49273


namespace volcano_ash_height_l49_49772

theorem volcano_ash_height (r d : ℝ) (h : r = 2700) (h₁ : 2 * r = 18 * d) : d = 300 :=
by
  sorry

end volcano_ash_height_l49_49772


namespace greatest_number_of_digits_in_product_l49_49699

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49699


namespace calculate_rent_is_correct_l49_49773

noncomputable def requiredMonthlyRent 
  (purchase_cost : ℝ) 
  (monthly_set_aside_percent : ℝ)
  (annual_property_tax : ℝ)
  (annual_insurance : ℝ)
  (annual_return_percent : ℝ) : ℝ :=
  let annual_return := annual_return_percent * purchase_cost
  let total_yearly_expenses := annual_return + annual_property_tax + annual_insurance
  let monthly_expenses := total_yearly_expenses / 12
  let retention_rate := 1 - monthly_set_aside_percent
  monthly_expenses / retention_rate

theorem calculate_rent_is_correct 
  (purchase_cost : ℝ := 200000)
  (monthly_set_aside_percent : ℝ := 0.2)
  (annual_property_tax : ℝ := 5000)
  (annual_insurance : ℝ := 2400)
  (annual_return_percent : ℝ := 0.08) :
  requiredMonthlyRent purchase_cost monthly_set_aside_percent annual_property_tax annual_insurance annual_return_percent = 2437.50 :=
by
  sorry

end calculate_rent_is_correct_l49_49773


namespace solve_y_l49_49388

-- Given conditions
variables (c d y : ℝ)
variables (hc : 0 < c) (hd : 0 < d) (hy : 0 < y)

-- We assume the expression for s
def s := (4 * c) ^ (4 * d)

-- The problem states that s equals the square of the product of c^d and y^d
theorem solve_y (hc : 0 < c) (hd : 0 < d) (hy : 0 < y) (h : s = (c ^ d * y ^ d) ^ 2) : y = 16 * c :=
by sorry

end solve_y_l49_49388


namespace count_four_digit_numbers_l49_49309

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49309


namespace find_values_of_A_and_B_l49_49131

-- Definitions of the conditions.
def cubic_eq (x : ℝ) : ℝ := x^3 - 3 * x + 2
def linear_eq (x : ℝ) : ℝ := -2 / 3 * x + 2

-- Intersection points
def intersection_points : set (ℝ × ℝ) :=
  {p | ∃ x,
    let y := linear_eq x in
    ⋆  p = (x, y) ∧
    cubic_eq x = y}

-- Proof statement
theorem find_values_of_A_and_B :
  ∃ A B,
  (∀ (x1 xm x3 y1 y2 y3 : ℝ),
     (x1, y1) ∈ intersection_points ∧
     (x2, y2) ∈ intersection_points ∧
     (x3, y3) ∈ intersection_points →
     x1 + x2 + x3 = A ∧
     y1 + y2 + y3 = B) →
   (A = 0 ∧ B = 6) :=
begin
  -- Placeholder for proof.
  sorry
end

end find_values_of_A_and_B_l49_49131


namespace soccer_team_goals_l49_49133

theorem soccer_team_goals (p : ℕ) (h₁ : p > 0) (h₂ : p % 15 = 0) :
  ∃ z : ℕ, z = 20 ∧ (1/3 : ℚ) * p + (1/5 : ℚ) * p + 8 + z = p :=
begin
  -- We'll provide the steps here, essentially reproducing the calculations:
  have h3 : (1/3 : ℚ) * p = (5/15 : ℚ) * p, { norm_num },
  have h4 : (1/5 : ℚ) * p = (3/15 : ℚ) * p, { norm_num },
  have h5 : (5/15 : ℚ) * p + (3/15 : ℚ) * p = (8/15 : ℚ) * p, { linarith },
  have h6 : (8/15 : ℚ) * p + 8 + z = p,
  {
    sorry
  },
end

end soccer_team_goals_l49_49133


namespace find_length_PQ_l49_49352

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 - x - y = 0

def line_l (t : ℝ) : ℝ × ℝ :=
  (1 / 2 - (real.sqrt 2) / 2 * t, (real.sqrt 2) / 2 * t)

def line_intersects_curve (t : ℝ) : Prop :=
  let ⟨x, y⟩ := line_l t in curve_C x y

theorem find_length_PQ :
  ∃ t1 t2 : ℝ, line_intersects_curve t1 ∧ line_intersects_curve t2 ∧
  abs (t1 - t2) = real.sqrt 3 / real.sqrt 2 :=
sorry

end find_length_PQ_l49_49352


namespace max_digits_in_product_l49_49495

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49495


namespace greatest_product_digit_count_l49_49671

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49671


namespace circles_touch_each_other_l49_49446

-- Define the radii of the two circles and the distance between their centers.
variables (R r d : ℝ)

-- Hypotheses: the condition and the relationships derived from the solution.
variables (x y t : ℝ)

-- The core relationships as conditions based on the problem and the solution.
axiom h1 : x + y = t
axiom h2 : x / y = R / r
axiom h3 : t / d = x / R

-- The proof statement
theorem circles_touch_each_other 
  (h1 : x + y = t) 
  (h2 : x / y = R / r) 
  (h3 : t / d = x / R) : 
  d = R + r := 
by 
  sorry

end circles_touch_each_other_l49_49446


namespace bridge_length_l49_49731

theorem bridge_length (train_length : ℕ) (train_speed_kmh : ℕ) (cross_time : ℕ) : 
  train_length = 140 → train_speed_kmh = 45 → cross_time = 30 → 
  let train_speed_ms := train_speed_kmh * 1000 / 3600 in
  let total_distance := train_speed_ms * cross_time in
  let bridge_length := total_distance - train_length in 
  bridge_length = 235 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  let train_speed_ms := 45 * 1000 / 3600
  let total_distance := train_speed_ms * 30
  let bridge_length := total_distance - 140
  have h : bridge_length = 235 := sorry
  exact h

end bridge_length_l49_49731


namespace greatest_product_digits_l49_49560

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49560


namespace problem1_problem2_l49_49909

noncomputable def cos_alpha (α : ℝ) : ℝ := (Real.sqrt 2 + 4) / 6
noncomputable def cos_alpha_plus_half_beta (α β : ℝ) : ℝ := 5 * Real.sqrt 3 / 9

theorem problem1 {α : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
                 (h1 : Real.cos (Real.pi / 4 + α) = 1 / 3) :
  Real.cos α = cos_alpha α :=
sorry

theorem problem2 {α β : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
                 (hβ1 : -Real.pi / 2 < β) (hβ2 : β < 0) 
                 (h1 : Real.cos (Real.pi / 4 + α) = 1 / 3) 
                 (h2 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α + β / 2) = cos_alpha_plus_half_beta α β :=
sorry

end problem1_problem2_l49_49909


namespace f_symmetric_a_eq_one_alpha_condition_l49_49175

noncomputable theory

def f (a : ℝ) (x : ℝ) : ℝ := cos (2 * x) + a * sin x

theorem f_symmetric (a : ℝ) : f a (π - x) = f a x :=
by simp [f, cos_sub, sin_sub, cos_add, sin_add]

theorem a_eq_one_alpha_condition (α : ℝ) : 
  (∀ x ∈ Set.Ioc (-π / 6) α, (∀ t ∈ | -1 <= t <= 1 , f 1 t)) → (π / 2 < α ∧ α ≤ (7 * π) / 6) :=
sorry

end f_symmetric_a_eq_one_alpha_condition_l49_49175


namespace tangent_line_inequality_holds_l49_49179

noncomputable def f (x : ℝ) : ℝ := 2 * x - (x + 1) * Real.log (x + 1) + 1

theorem tangent_line (x y : ℝ) (h : (x, y) = (0, f 0)) : x - y + 1 = 0 := 
by {
  subst h,
  simp [f],
  sorry
}

theorem inequality_holds (k : ℝ) : (∀ x : ℝ, 0 ≤ x → f x ≥ k * x^2 + x + 1) → k ≤ -1/2 := 
by {
  intro h,
  sorry
}

end tangent_line_inequality_holds_l49_49179


namespace unique_4_digit_number_l49_49048

theorem unique_4_digit_number (P E R U : ℕ) 
  (hP : 0 ≤ P ∧ P < 10)
  (hE : 0 ≤ E ∧ E < 10)
  (hR : 0 ≤ R ∧ R < 10)
  (hU : 0 ≤ U ∧ U < 10)
  (hPERU : 1000 ≤ (P * 1000 + E * 100 + R * 10 + U) ∧ (P * 1000 + E * 100 + R * 10 + U) < 10000) 
  (h_eq : (P * 1000 + E * 100 + R * 10 + U) = (P + E + R + U) ^ U) : 
  (P = 4) ∧ (E = 9) ∧ (R = 1) ∧ (U = 3) ∧ (P * 1000 + E * 100 + R * 10 + U = 4913) :=
sorry

end unique_4_digit_number_l49_49048


namespace sum_first_100_natural_numbers_l49_49090

theorem sum_first_100_natural_numbers : (∑ i in Finset.range 101, i) = 5050 := by
  sorry

end sum_first_100_natural_numbers_l49_49090


namespace daily_profit_35_selling_price_for_600_profit_no_900_profit_possible_l49_49752

-- Definitions based on conditions
def purchase_price : ℝ := 30
def max_selling_price : ℝ := 55
def linear_relationship (x : ℝ) : ℝ := -2 * x + 140
def profit (x : ℝ) : ℝ := (x - purchase_price) * linear_relationship x

-- Part 1: Daily profit when selling price is 35 yuan
theorem daily_profit_35 : profit 35 = 350 :=
  sorry

-- Part 2: Selling price for a daily profit of 600 yuan
theorem selling_price_for_600_profit (x : ℝ) (h1 : 30 ≤ x) (h2 : x ≤ 55) : profit x = 600 → x = 40 :=
  sorry

-- Part 3: Possibility of daily profit of 900 yuan
theorem no_900_profit_possible (h1 : ∀ x, 30 ≤ x ∧ x ≤ 55 → profit x ≠ 900) : ¬ ∃ x, 30 ≤ x ∧ x ≤ 55 ∧ profit x = 900 :=
  sorry

end daily_profit_35_selling_price_for_600_profit_no_900_profit_possible_l49_49752


namespace man_older_than_son_l49_49062

theorem man_older_than_son (S M : ℕ) (h1 : S = 23) (h2 : M + 2 = 2 * (S + 2)) : M - S = 25 :=
by
  sorry

end man_older_than_son_l49_49062


namespace reciprocal_repeating_decimal_l49_49720

theorem reciprocal_repeating_decimal :
  (let x := 0.353535... in 1 / x) = 99 / 35 :=
by
  sorry

end reciprocal_repeating_decimal_l49_49720


namespace greatest_number_of_digits_in_product_l49_49692

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49692


namespace smaller_group_men_l49_49958

theorem smaller_group_men (M : ℕ) (h1 : 36 * 25 = M * 90) : M = 10 :=
by
  -- Here we would provide the proof. Unfortunately, proving this in Lean 4 requires knowledge of algebra.
  sorry

end smaller_group_men_l49_49958


namespace table_sum_square_l49_49769

theorem table_sum_square (n : ℕ) : 
  let s := ∑ i in finset.range n, (i + 1) * 2^i in 
  (s + s) * (s + s) = (2^(n + 1) - 1) * (2^(n + 1) - 1) :=
by
  sorry

end table_sum_square_l49_49769


namespace greatest_number_of_digits_in_product_l49_49619

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49619


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49715

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49715


namespace winning_strategy_l49_49096

/-
 Given Sergio and Mia's game where each move can remove one brick, two adjacent bricks, or three adjacent bricks,
 the starting configuration (7, 3, 3) has a guaranteed losing position for the player who moves first
 if both play optimally.
-/

def nim_value (bricks : Nat) : Nat :=
  match bricks with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 0
  | 4 => 1
  | 5 => 1
  | 6 => 2
  | 7 => 0
  | _ => sorry -- Assume nim-values for bricks > 7 can be calculated

def nim_sum (a b c : Nat) : Nat :=
  nat.xor (nat.xor a b) c

theorem winning_strategy : nim_sum (nim_value 7) (nim_value 3) (nim_value 3) = 0 :=
by
  -- Since we don't need to provide the proof steps, we'll use sorry here.
  sorry

end winning_strategy_l49_49096


namespace max_value_of_g_eq_f_l49_49931

noncomputable def g (t x : ℝ) : ℝ := (t - 1) * x - 4 / x

-- Define the maximum value function f(t)
noncomputable def f (t : ℝ) : ℝ :=
  if t ≤ -3 then t - 5
  else if t < 0 then -4 * Real.sqrt (1 - t)
  else 2 * t - 4

-- State the theorem regarding maximum value
theorem max_value_of_g_eq_f (t : ℝ) :
  ∃ x ∈ Set.Icc (1 : ℝ) 2, g t x = f t :=
begin
  sorry
end

end max_value_of_g_eq_f_l49_49931


namespace four_digit_numbers_count_l49_49251

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49251


namespace type_of_x_l49_49330

theorem type_of_x (x : ℝ) (hx1 : (-3)^(2*x) = 3^(12 - x)) (hx2 : x = 4) : x ∈ ℤ :=
by {
    sorry -- proof to be completed
}

end type_of_x_l49_49330


namespace max_digits_of_product_l49_49544

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49544


namespace impossible_sum_of_two_smaller_angles_l49_49980

theorem impossible_sum_of_two_smaller_angles
  {α β γ : ℝ}
  (h1 : α + β + γ = 180)
  (h2 : 0 < α + β ∧ α + β < 180) :
  α + β ≠ 130 :=
sorry

end impossible_sum_of_two_smaller_angles_l49_49980


namespace four_digit_number_count_l49_49242

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49242


namespace perp_vectors_dot_product_eq_zero_l49_49940

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perp_vectors_dot_product_eq_zero (x : ℝ) (h : dot_product vector_a (vector_b x) = 0) : x = -8 :=
  by sorry

end perp_vectors_dot_product_eq_zero_l49_49940


namespace factorial_vs_power_l49_49854

theorem factorial_vs_power : fact 200 < (100:ℕ)^200 := 
by sorry

end factorial_vs_power_l49_49854


namespace four_digit_numbers_count_eq_l49_49263

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49263


namespace function_C_is_odd_and_decreasing_l49_49023

-- Conditions
def f (x : ℝ) : ℝ := -x^3 - x

-- Odd function condition
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Strictly decreasing condition
def is_strictly_decreasing (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

-- The theorem we want to prove
theorem function_C_is_odd_and_decreasing : 
  is_odd f ∧ is_strictly_decreasing f :=
by
  sorry

end function_C_is_odd_and_decreasing_l49_49023


namespace cube_ratio_squared_l49_49984

-- Definitions for midpoints J and I, areas, and ratio R^2
variables {a : ℝ} (cube : ∀ (X : Type) (Y : Type), Y) (J I : (ℝ × ℝ × ℝ)) 
[hJ : ∃ F B : ℝ × ℝ × ℝ, J = ((F.1 + B.1)/2, (F.2 + B.2)/2, (F.3 + B.3)/2)]
[hI : ∃ H D : ℝ × ℝ × ℝ, I = ((H.1 + D.1)/2, (H.2 + D.2)/2, (H.3 + D.3)/2)]
(A : ℝ)
(R : ℝ)

noncomputable
def area_face := a ^ 2

noncomputable
def area_EJCI := (5 / 4) * (a ^ 2) -- Computed from geometric considerations

noncomputable
def R := area_EJCI / area_face

theorem cube_ratio_squared : R ^ 2 = 25 / 16 :=
by sorry

end cube_ratio_squared_l49_49984


namespace cubes_configuration_l49_49066

-- Define the problem's conditions
structure Cube where
  has_snap : Bool :=
  has_hole : Fin 3 → Bool :=
  has_plain : Fin 2 → Bool :=

-- Define the key property to be proved
def min_cubes_required : Nat :=
4

theorem cubes_configuration :
  ∀ (cubes : List Cube),
  (∀ cube ∈ cubes, cube.has_snap = true ∧ 
   (∀ i, cube.has_hole i = true) ∧
   (∀ i, cube.has_plain i = true)) →
  (∃ t, t ≤ 4 ∧ 
   (∀ i < t, cubes.nthLe i sorry = some ({
     has_snap := false,
     has_hole := fun _ => true,
     has_plain := fun _ => true
   }: Cube))) →
  t = min_cubes_required :=
by
  -- Proof goes here
  sorry

end cubes_configuration_l49_49066


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49712

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49712


namespace cost_price_represents_articles_l49_49334

theorem cost_price_represents_articles (C S : ℝ) (N : ℕ)
  (h1 : N * C = 16 * S)
  (h2 : S = C * 1.125) :
  N = 18 :=
by
  sorry

end cost_price_represents_articles_l49_49334


namespace students_failed_l49_49469

theorem students_failed (total_students : ℕ) (percent_A : ℚ) (fraction_BC : ℚ) (students_A : ℕ)
  (students_remaining : ℕ) (students_BC : ℕ) (students_failed : ℕ)
  (h1 : total_students = 32) (h2 : percent_A = 0.25) (h3 : fraction_BC = 0.25)
  (h4 : students_A = total_students * percent_A)
  (h5 : students_remaining = total_students - students_A)
  (h6 : students_BC = students_remaining * fraction_BC)
  (h7 : students_failed = total_students - students_A - students_BC) :
  students_failed = 18 :=
sorry

end students_failed_l49_49469


namespace fraction_of_full_tank_used_l49_49785

def speed : ℝ := 50 -- miles per hour
def fuel_efficiency : ℝ := 1 / 30 -- gallons per mile
def full_tank : ℝ := 10 -- gallons
def time_traveled : ℝ := 5 -- hours

def distance_traveled : ℝ := speed * time_traveled
def gallons_used : ℝ := distance_traveled * fuel_efficiency

theorem fraction_of_full_tank_used : gallons_used / full_tank = 5 / 6 := 
by
  -- The proof is omitted and replaced with sorry.
  sorry

end fraction_of_full_tank_used_l49_49785


namespace cards_different_suits_l49_49075

theorem cards_different_suits :
  let deck_size := 52 in
  let cards_per_suit := 13 in
  let different_suit_ways := deck_size * (deck_size - cards_per_suit) in
  different_suit_ways = 2028 :=
by
  let deck_size := 52
  let cards_per_suit := 13
  let different_suit_ways := deck_size * (deck_size - cards_per_suit)
  exact rfl

end cards_different_suits_l49_49075


namespace g_diff_l49_49841

def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 2 * x - 1

theorem g_diff (x h : ℝ) : g (x + h) - g x = h * (6 * x^2 + 6 * x * h + 2 * h^2 + 10 * x + 5 * h - 2) := 
by
  sorry

end g_diff_l49_49841


namespace greatest_digits_product_l49_49588

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49588


namespace balls_in_boxes_l49_49318

theorem balls_in_boxes : 
  ∃ (x1 x2 x3 : ℕ), x1 + x2 + x3 = 7 ∧ 
  0 ≤ x1 ∧ x1 ≤ 4 ∧ 
  0 ≤ x2 ∧ x2 ≤ 4 ∧ 
  0 ≤ x3 ∧ x3 ≤ 4 ∧ 
  (∀ (y1 y2 y3 : ℕ), y1 + y2 + y3 = 7 ∧ 0 ≤ y1 ∧ y1 ≤ 4 ∧ 0 ≤ y2 ∧ y2 ≤ 4 ∧ 0 ≤ y3 ∧ y3 ≤ 4 → 
    {x1, x2, x3} = {y1, y2, y3}) ∧ 
  card {x1, x2, x3} = 6 :=
sorry

end balls_in_boxes_l49_49318


namespace max_digits_in_product_l49_49597

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49597


namespace range_of_a_l49_49186

-- Define the propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 0 ≤ x → x ≤ 1 → a ≥ Real.exp x
def q (a : ℝ) := ∃ x : ℝ, x^2 + 4 * x + a = 0

-- The proof statement
theorem range_of_a (a : ℝ) : (p a ∧ q a) → a ∈ Set.Icc (Real.exp 1) 4 := by
  intro h
  sorry

end range_of_a_l49_49186


namespace four_digit_number_count_l49_49207

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49207


namespace greatest_number_of_digits_in_product_l49_49698

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49698


namespace four_digit_numbers_count_l49_49209

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49209


namespace repeating_decimals_to_fraction_l49_49866

theorem repeating_decimals_to_fraction :
  (0.\overline{6} - 0.\overline{4} + 0.\overline{8}) = (10 / 9) :=
by
  -- Define the repeating decimals as fractions
  let x := 6 / 9
  let y := 4 / 9
  let z := 8 / 9
  -- Express the statement to be proved
  have h1 : 0.\overline{6} = x := sorry
  have h2 : 0.\overline{4} = y := sorry
  have h3 : 0.\overline{8} = z := sorry
  calc
    (0.\overline{6} - 0.\overline{4} + 0.\overline{8}) = (x - y + z) : by rw [h1, h2, h3]
    ... = (6 / 9 - 4 / 9 + 8 / 9) : by congr -- Use congruence for fractions
    ... = (10 / 9) : by ring

end repeating_decimals_to_fraction_l49_49866


namespace cos_of_three_pi_div_two_l49_49116

theorem cos_of_three_pi_div_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  sorry

end cos_of_three_pi_div_two_l49_49116


namespace largest_in_set_l49_49077

theorem largest_in_set : max_set {1, -1, 0, Real.sqrt 2} = Real.sqrt 2 := by
  sorry

end largest_in_set_l49_49077


namespace cube_dimension_l49_49019

theorem cube_dimension (x s : ℝ) (hx1 : s^3 = 8 * x) (hx2 : 6 * s^2 = 2 * x) : x = 1728 := 
by {
  sorry
}

end cube_dimension_l49_49019


namespace four_digit_number_count_l49_49197

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49197


namespace probability_factor_of_120_l49_49764

open Finset

theorem probability_factor_of_120 : 
  let s := range 37
  let five_factorial := 5!
  let factors_of_120 := {n ∈ s | five_factorial % n = 0}
  (finsupp.card factors_of_120 : ℚ) / (finsupp.card s : ℚ) = 4 / 9 :=
by sorry

end probability_factor_of_120_l49_49764


namespace greatest_digits_product_l49_49592

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49592


namespace ceil_sqrt_180_eq_14_l49_49860

theorem ceil_sqrt_180_eq_14
  (h : 13 < Real.sqrt 180 ∧ Real.sqrt 180 < 14) :
  Int.ceil (Real.sqrt 180) = 14 :=
  sorry

end ceil_sqrt_180_eq_14_l49_49860


namespace greatest_possible_digits_in_product_l49_49514

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49514


namespace tan_alpha_eq_neg_one_l49_49914

theorem tan_alpha_eq_neg_one (α : ℝ) (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : Real.tan α = -1 :=
  sorry

end tan_alpha_eq_neg_one_l49_49914


namespace seating_arrangements_l49_49985

-- Define the number of people
def n : ℕ := 7

-- Define the number of unique seating arrangements around a round table
def unique_seating_arrangements (n : ℕ) : ℕ := factorial n / n

-- State the theorem 
theorem seating_arrangements : unique_seating_arrangements n = factorial 6 := 
by 
  sorry

end seating_arrangements_l49_49985


namespace four_digit_numbers_count_l49_49224

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49224


namespace tank_volume_ratio_l49_49373

variable {V1 V2 : ℝ}

theorem tank_volume_ratio
  (h1 : 3 / 4 * V1 = 5 / 8 * V2) :
  V1 / V2 = 5 / 6 :=
sorry

end tank_volume_ratio_l49_49373


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49573

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49573


namespace meat_pie_cost_l49_49140

variable (total_farthings : ℕ) (farthings_per_pfennig : ℕ) (remaining_pfennigs : ℕ)

def total_pfennigs (total_farthings farthings_per_pfennig : ℕ) : ℕ :=
  total_farthings / farthings_per_pfennig

def pie_cost (total_farthings farthings_per_pfennig remaining_pfennigs : ℕ) : ℕ :=
  total_pfennigs total_farthings farthings_per_pfennig - remaining_pfennigs

theorem meat_pie_cost
  (h1 : total_farthings = 54)
  (h2 : farthings_per_pfennig = 6)
  (h3 : remaining_pfennigs = 7) :
  pie_cost total_farthings farthings_per_pfennig remaining_pfennigs = 2 :=
by
  sorry

end meat_pie_cost_l49_49140


namespace identify_jars_l49_49949

namespace JarIdentification

/-- Definitions of Jar labels -/
inductive JarLabel
| Nickels
| Dimes
| Nickels_and_Dimes

open JarLabel

/-- Mislabeling conditions for each jar -/
def mislabeled (jarA : JarLabel) (jarB : JarLabel) (jarC : JarLabel) : Prop :=
  ((jarA ≠ Nickels) ∧ (jarB ≠ Dimes) ∧ (jarC ≠ Nickels_and_Dimes)) ∧
  ((jarC = Nickels ∨ jarC = Dimes))

/-- Given the result of a coin draw from the jar labeled "Nickels and Dimes" -/
def jarIdentity (jarA jarB jarC : JarLabel) (drawFromC : String) : Prop :=
  if drawFromC = "Nickel" then
    jarC = Nickels ∧ jarA = Nickels_and_Dimes ∧ jarB = Dimes
  else if drawFromC = "Dime" then
    jarC = Dimes ∧ jarB = Nickels_and_Dimes ∧ jarA = Nickels
  else 
    false

/-- Main theorem to prove the identification of jars -/
theorem identify_jars (jarA jarB jarC : JarLabel) (draw : String)
  (h1 : mislabeled jarA jarB jarC) :
  jarIdentity jarA jarB jarC draw :=
by
  sorry

end JarIdentification

end identify_jars_l49_49949


namespace expansion_correct_l49_49863

variable (x y : ℝ)

theorem expansion_correct : 
  (3 * x - 15) * (4 * y + 20) = 12 * x * y + 60 * x - 60 * y - 300 :=
by
  sorry

end expansion_correct_l49_49863


namespace sum_of_remainders_mod_500_l49_49130

theorem sum_of_remainders_mod_500 : 
  (5 ^ (5 ^ (5 ^ 5)) + 2 ^ (2 ^ (2 ^ 2))) % 500 = 49 := by
  sorry

end sum_of_remainders_mod_500_l49_49130


namespace cube_x_value_l49_49008

noncomputable def cube_side_len (x : ℝ) : ℝ := (8 * x) ^ (1 / 3)

lemma cube_volume (x : ℝ) : (cube_side_len x) ^ 3 = 8 * x :=
  by sorry

lemma cube_surface_area (x : ℝ) : 6 * (cube_side_len x) ^ 2 = 2 * x :=
  by sorry

theorem cube_x_value (x : ℝ) (hV : (cube_side_len x) ^ 3 = 8 * x) (hS : 6 * (cube_side_len x) ^ 2 = 2 * x) : x = sqrt 3 / 72 :=
  by sorry

end cube_x_value_l49_49008


namespace triangle_probability_from_nine_sticks_l49_49412

theorem triangle_probability_from_nine_sticks :
  let sticks := [3, 4, 5, 6, 8, 9, 10, 12, 15] in
  let combinations := (Finset.powersetLen 4 (Finset.ofList sticks)).toList in
  let possible_triangles := combinations.filter (λ l, ∀ c ∈ (Finset.powersetLen 3 (Finset.ofList l)), 
    match c.val with
    | a :: b :: c :: [] => a + b > c ∧ a + c > b ∧ b + c > a
    | _ => false
    end) in
  (possible_triangles.length = combinations.length) :=
sorry

end triangle_probability_from_nine_sticks_l49_49412


namespace find_second_group_of_men_l49_49051

noncomputable def work_rate_of_man := ℝ
noncomputable def work_rate_of_woman := ℝ

variables (m w : ℝ)

-- Condition 1: 3 men and 8 women complete the task in the same time as x men and 2 women.
axiom condition1 (x : ℝ) : 3 * m + 8 * w = x * m + 2 * w

-- Condition 2: 2 men and 3 women complete half the task in the same time as 3 men and 8 women completing the whole task.
axiom condition2 : 2 * m + 3 * w = 0.5 * (3 * m + 8 * w)

theorem find_second_group_of_men (x : ℝ) (m w : ℝ) (h1 : 0.5 * m = w)
  (h2 : 3 * m + 8 * w = x * m + 2 * w) : x = 6 :=
by {
  sorry
}

end find_second_group_of_men_l49_49051


namespace trapezoid_median_l49_49076

theorem trapezoid_median (h : ℝ) (b : ℝ) (A_triangle A_trapezoid : ℝ) : 
  A_triangle = 12 * h → 
  A_trapezoid = (3 * b / 2) * h → 
  A_triangle = A_trapezoid → 
  24 = b + b → 
  m = 12 :=
begin
  sorry
end

end trapezoid_median_l49_49076


namespace expand_product_l49_49864

-- Define the expressions (x + 3)(x + 8) and x^2 + 11x + 24
def expr1 (x : ℝ) : ℝ := (x + 3) * (x + 8)
def expr2 (x : ℝ) : ℝ := x^2 + 11 * x + 24

-- Prove that the two expressions are equal
theorem expand_product (x : ℝ) : expr1 x = expr2 x := by
  sorry

end expand_product_l49_49864


namespace greatest_possible_digits_in_product_l49_49513

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49513


namespace ratio_BQ_QF_l49_49999

noncomputable def triangle : Type := sorry

variables (A B C D F Q : triangle)
variables (AB AC BC AD BD DC BF FC BQ QF : ℝ)

-- Conditions
axiom AB_eq_8 : AB = 8
axiom AC_eq_4 : AC = 4
axiom BC_eq_6 : BC = 6
axiom angle_bisector_AD : BD / DC = AB / AC
axiom angle_bisector_BF : BF / FC = AB / BC
axiom Q_intersection : ∃ (Q : triangle), angle_bisector_AD ∧ angle_bisector_BF

-- Goal
theorem ratio_BQ_QF :
  BQ / QF = 4 / 3 := sorry

end ratio_BQ_QF_l49_49999


namespace arctan_sum_pi_div_two_l49_49823

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l49_49823


namespace four_digit_number_count_l49_49199

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49199


namespace extreme_value_a1_monotonicity_f_l49_49176

/-- Part (1): Extreme value of f(x) when a = 1 -/
theorem extreme_value_a1 (x : ℝ) : 
  f x = (1 / 2) * x ^ 2 - log x →
  (∀ x > 0, (x < 1 → f' x < 0) ∧ (x > 1 → f' x > 0)) →
  f 1 = 1 / 2 :=
sorry

/-- Part (2): Monotonicity of the function f(x) -/
theorem monotonicity_f (a x : ℝ) (hx : x > 0) (ha : a ≠ 0) :
  f x = (a ^ 3 / 2) * x ^ 2 - a * log x →
  (a < 0 → ((0 < x ∧ x < -1/a) → f' x > 0) ∧ ((x > -1/a) → f' x < 0)) ∧
  (a > 0 → ((0 < x ∧ x < 1/a) → f' x < 0) ∧ ((x > 1/a) → f' x > 0)) :=
sorry

end extreme_value_a1_monotonicity_f_l49_49176


namespace four_digit_numbers_count_eq_l49_49264

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49264


namespace positive_integer_expression_l49_49886

theorem positive_integer_expression (x : ℝ) (k : ℤ) (h : x ≠ 0) : ∃ n : ℤ, n > 0 ∧ (| x - |k * x| | / x) = n :=
by sorry

end positive_integer_expression_l49_49886


namespace greatest_possible_digits_in_product_l49_49516

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49516


namespace bucket_full_weight_l49_49750

variable (p q : ℝ)

theorem bucket_full_weight (p q : ℝ) (x y: ℝ) (h1 : x + 3/4 * y = p) (h2 : x + 1/3 * y = q) :
  x + y = (8 * p - 7 * q) / 5 :=
by
  sorry

end bucket_full_weight_l49_49750


namespace probability_of_not_red_l49_49745

-- Definitions based on conditions
def total_number_of_jelly_beans : ℕ := 7 + 9 + 10 + 12 + 5
def number_of_non_red_jelly_beans : ℕ := 9 + 10 + 12 + 5

-- Proving the probability
theorem probability_of_not_red : 
  (number_of_non_red_jelly_beans : ℚ) / total_number_of_jelly_beans = 36 / 43 :=
by sorry

end probability_of_not_red_l49_49745


namespace fraction_reduction_l49_49963

theorem fraction_reduction (x y : ℝ) : 
  (4 * x - 4 * y) / (4 * x * 4 * y) = (1 / 4) * ((x - y) / (x * y)) := 
by 
  sorry

end fraction_reduction_l49_49963


namespace greatest_possible_number_of_digits_in_product_l49_49535

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49535


namespace bank_balance_after_two_years_l49_49435

theorem bank_balance_after_two_years :
  let P := 100 -- initial deposit
  let r := 0.1 -- annual interest rate
  let t := 2   -- time in years
  in P * (1 + r) ^ t = 121 :=
by
  sorry

end bank_balance_after_two_years_l49_49435


namespace arctan_sum_eq_pi_div_two_l49_49826

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l49_49826


namespace four_digit_number_count_l49_49244

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49244


namespace combination_property_l49_49956

theorem combination_property (x : ℕ) (hx : 2 * x - 1 ≤ 11 ∧ x ≤ 11) :
  (Nat.choose 11 (2 * x - 1) = Nat.choose 11 x) → (x = 1 ∨ x = 4) :=
by
  sorry

end combination_property_l49_49956


namespace a_can_finish_in_60_days_l49_49029

open_locale classical

theorem a_can_finish_in_60_days (W : ℝ) (a b : ℝ) : 
  (1 / a + 1 / b = 1 / 30) →
  (20 * (W / 30) + 20 * (W / a) = W) →
  a = 60 :=
by
  intros h_combined_rate h_worked_20_each
  sorry

end a_can_finish_in_60_days_l49_49029


namespace volume_of_tetrahedron_OEFG_l49_49148

-- Define the vectors representing the coordinates of points O, E, F, and G
def O := (1 / 2 : ℝ) ∶ ℝ × ℝ × ℝ
def E := (0 : ℝ, 1 / 2, 1 / 2) ∶ ℝ × ℝ × ℝ
def F := (1 / 2 : ℝ, 1 : ℝ, 1 / 2) ∶ ℝ × ℝ × ℝ
def G := (1 : ℝ, 1 / 2, 1 / 2) ∶ ℝ × ℝ × ℝ

-- Volume of tetrahedron given vertices O, E, F, G
def volume_tetrahedron (A B C D : ℝ × ℝ × ℝ) :=
  1 / 6 * abs ((B.1 - A.1) * (C.2 - A.2) * (D.3 - A.3) + (B.2 - A.2) * (C.3 - A.3) * (D.1 - A.1) + 
              (B.3 - A.3) * (C.1 - A.1) * (D.2 - A.2) - (B.1 - A.1) * (C.3 - A.3) * (D.2 - A.2) - 
              (B.2 - A.2) * (C.1 - A.1) * (D.3 - A.3) - (B.3 - A.3) * (C.2 - A.2) * (D.1 - A.1))

-- Define the theorem we want to prove
theorem volume_of_tetrahedron_OEFG : volume_tetrahedron O E F G = 5 / 48 :=
by
  -- Use sorry to indicate the proof is omitted
  sorry

end volume_of_tetrahedron_OEFG_l49_49148


namespace angle_between_vectors_l49_49964

variable {V : Type*} [InnerProductSpace ℝ V]

theorem angle_between_vectors (α β : V) (hα : α ≠ 0) (hβ : β ≠ 0) 
  (h : ‖α + β‖ = ‖α - β‖) : ∡ α β = 90 :=
by
  -- Introduction of the variables and assumptions
  intros α β hα hβ h,
  
  -- Conversion of the given condition to an equality involving the inner product
  let h' : ⟪α + β, α + β⟫ = ⟪α - β, α - β⟫ := by 
    rw [norm_sq_eq_inner, norm_sq_eq_inner, heq, ←inner_mul_left, smul_eq_minus, add_eq_zero_if_inner_eq_zero],
    ring_nf;
    intros α β hα hβ,
    let ⟪α + β, α + β⟫ = ⟪AB⟫,
    rw [⟪ibitα + β, ahβ⟩, ⟪ah, ⟩, α, β_neq],

  -- Simplification of the inner product equality to show ⟪α, β⟫ = 0
  have : 2 * ⟪α, β⟫ = -2 * ⟪α, β⟫ := by
    calc 2 * ⟪α, β⟫ + 2 * ⟪α, β⟫ = 0,
  
  -- Conversion of the zero inner product to show the angle is 90 degrees
  have : ⟪α, β⟫ = 0 := by
    exact (nonzero α, nonzero β hxy),

  -- Establish the final result that the angle is 90 degrees
  sorry

end angle_between_vectors_l49_49964


namespace paving_length_of_road_l49_49057

noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h
noncomputable def road_length (V w t : ℝ) : ℝ := V / (w * t)

theorem paving_length_of_road : 
  let r := 1
  let h := 7.5
  let w := 5
  let t := 0.02
  let V := volume_of_cone r h
  road_length V w t = 78.5 :=
by
  sorry

end paving_length_of_road_l49_49057


namespace valid_triangles_pentadecagon_l49_49947

-- Definitions of the problem
def vertices : ℕ := 15

def total_triangles (n : ℕ) : ℕ := (finset.range (n).choose 3).card

def invalid_triangles (n : ℕ) : ℕ := n

def valid_triangles (n : ℕ) : ℕ := total_triangles n - invalid_triangles n

-- The theorem stating the number of valid triangles
theorem valid_triangles_pentadecagon : valid_triangles vertices = 440 :=
by sorry

end valid_triangles_pentadecagon_l49_49947


namespace adeline_speed_l49_49775

def distance : ℝ := 2304
def time : ℝ := 36
def speed (d : ℝ) (t : ℝ) : ℝ := d / t

theorem adeline_speed : speed distance time = 64 := by
  sorry

end adeline_speed_l49_49775


namespace curve_C1_and_segment_PQ_l49_49988

-- Defining the parametric equations of the curve C1
def x_param (θ : ℝ) : ℝ := 1 + 2 * Real.cos θ
def y_param (θ : ℝ) : ℝ := 2 * Real.sin θ

-- Defining the polar equation of the line l
def polar_line_eq (ρ θ : ℝ) : Prop := 2 * ρ * Real.sin (θ + π / 3) = 3 * Real.sqrt 3

-- Converting polar_line_eq to Cartesian equation y + √3 * x = 3 √3
def cartesian_line_eq (x y : ℝ) : Prop := y + Real.sqrt 3 * x = 3 * Real.sqrt 3

--General equation of curve C1
def general_eq_C1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

--Proving curve C1 and length segment PQ
theorem curve_C1_and_segment_PQ (θ ρ x y : ℝ) :
    (x = x_param θ) ∧ (y = y_param θ) ∧ polar_line_eq ρ θ →
    (general_eq_C1 x y) ∧ ∃ P Q : ℝ × ℝ, cartesian_line_eq P.1 P.2 ∧ 
    cartesian_line_eq Q.1 Q.2 ∧ 
    dist (1, 0) (1 + 2, 2) = 2 := sorry

end curve_C1_and_segment_PQ_l49_49988


namespace greatest_possible_number_of_digits_in_product_l49_49528

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49528


namespace max_digits_in_product_l49_49492

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49492


namespace evaluate_expression_l49_49884

-- Definitions of M and m
def M (x y : ℝ) := max x y
def m (x y : ℝ) := min x y

theorem evaluate_expression {p q r s t : ℝ} (h₀ : p ≠ q) (h₁ : p ≠ r) (h₂ : p ≠ s) (h₃ : p ≠ t) 
  (h₄ : q ≠ r) (h₅ : q ≠ s) (h₆ : q ≠ t) (h₇ : r ≠ s) (h₈ : r ≠ t) (h₉ : s ≠ t)
  (h₁₀ : p < r) (h₁₁ : r < s) (h₁₂ : s < q) (h₁₃ : q < t) : 
  M (m (M(p, q), s), M (r, m (s, t))) = s :=
sorry

end evaluate_expression_l49_49884


namespace base3_to_base5_conversion_l49_49845

-- Define the conversion from base 3 to decimal
def base3_to_decimal (n : ℕ) : ℕ :=
  n % 10 * 1 + (n / 10 % 10) * 3 + (n / 100 % 10) * 9 + (n / 1000 % 10) * 27 + (n / 10000 % 10) * 81

-- Define the conversion from decimal to base 5
def decimal_to_base5 (n : ℕ) : ℕ :=
  n % 5 + (n / 5 % 5) * 10 + (n / 25 % 5) * 100

-- The initial number in base 3
def initial_number_base3 : ℕ := 10121

-- The final number in base 5
def final_number_base5 : ℕ := 342

-- The theorem that states the conversion result
theorem base3_to_base5_conversion :
  decimal_to_base5 (base3_to_decimal initial_number_base3) = final_number_base5 :=
by
  sorry

end base3_to_base5_conversion_l49_49845


namespace tangent_intersects_parabola_find_monotonic_intervals_find_max_k_l49_49177

noncomputable def f (x : ℝ) := Real.log (x + 1) - x
def tangent_at_0 (x : ℝ) := (1 - (1 : ℝ)) * x -- since a = 1 from the solution
def intersection := ∀ x : ℝ, tangent_at_0 x = (1/2) * x^2

def g (x : ℝ) := (x * f (x - 1) + x^2) / (x - 1)
def h (x : ℝ) := x - Real.log x - 2

theorem tangent_intersects_parabola :
    (intersect_tangent_with_y : ∀ x : ℝ, tangent_at_0 x = (1/2) * x^2 ↔ a = 1) :=
sorry

theorem find_monotonic_intervals :
    (montonic_intervals : (∀ x ∈ (-1, 0), deriv f x > 0) ∧ (∀ x ∈ (0, +inf), deriv f x < 0)) :=
sorry

theorem find_max_k (k : ℤ) :
    (∀ x > 1, (k + 1) * (x - 1) < x * f (x - 1) + x^2 → k < 4 - Real.log 4 - 2) :=
sorry

end tangent_intersects_parabola_find_monotonic_intervals_find_max_k_l49_49177


namespace number_of_four_digit_numbers_l49_49269

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49269


namespace arctan_triangle_complementary_l49_49815

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l49_49815


namespace four_digit_numbers_count_l49_49219

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49219


namespace city_mpg_l49_49030

noncomputable def highway_mpg : ℝ := 20 -- from the solution
noncomputable def highway_distance_per_tankful : ℝ := 480
noncomputable def city_distance_per_tankful : ℝ := 336
noncomputable def mpg_difference : ℝ := 6

theorem city_mpg (H : ℝ) (T : ℝ) (h_dist: H * T = highway_distance_per_tankful) 
    (c_dist : (H - mpg_difference) * T = city_distance_per_tankful) :
    H = highway_mpg → H - mpg_difference = 14 :=
by
  intro h_eq
  rw h_eq
  calc (highway_mpg - mpg_difference) = 14
      sorry

end city_mpg_l49_49030


namespace max_digits_of_product_l49_49536

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49536


namespace harvey_sold_steaks_after_12_left_l49_49196

theorem harvey_sold_steaks_after_12_left :
  ∀ (initial_steaks sold_steaks_left total_steaks_sold : ℕ),
    initial_steaks = 25 →
    sold_steaks_left = 12 →
    total_steaks_sold = 17 →
    (total_steaks_sold - (initial_steaks - sold_steaks_left)) = 4 :=
by
  intros initial_steaks sold_steaks_left total_steaks_sold
  intros initial_h sold_h total_h
  rw [initial_h, sold_h, total_h]
  exact rfl

end harvey_sold_steaks_after_12_left_l49_49196


namespace can_vasya_obtain_400_mercedes_l49_49085

-- Define the types for the cars
inductive Car : Type
| Zh : Car
| V : Car
| M : Car

-- Define the initial conditions as exchange constraints
def exchange1 (Zh V M : ℕ) : Prop :=
  3 * Zh = V + M

def exchange2 (V Zh M : ℕ) : Prop :=
  3 * V = 2 * Zh + M

-- Define the initial number of Zhiguli cars Vasya has.
def initial_Zh : ℕ := 700

-- Define the target number of Mercedes cars Vasya wants.
def target_M : ℕ := 400

-- The proof goal: Vasya cannot exchange to get exactly 400 Mercedes cars.
theorem can_vasya_obtain_400_mercedes (Zh V M : ℕ) (h1 : exchange1 Zh V M) (h2 : exchange2 V Zh M) :
  initial_Zh = 700 → target_M = 400 → (Zh ≠ 0 ∨ V ≠ 0 ∨ M ≠ 400) := sorry

end can_vasya_obtain_400_mercedes_l49_49085


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49703

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49703


namespace tan_alpha_eq_neg1_l49_49912

theorem tan_alpha_eq_neg1 (α : ℝ) (h₁ : sin α - cos α = real.sqrt 2) (h₂ : 0 < α ∧ α < real.pi) : tan α = -1 := by
  sorry

end tan_alpha_eq_neg1_l49_49912


namespace four_digit_numbers_count_eq_l49_49267

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49267


namespace max_digits_in_product_l49_49601

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49601


namespace unique_integer_solution_l49_49850

theorem unique_integer_solution (n : ℤ) :
  (⌊n^2 / 4 + n⌋ - ⌊n / 2⌋^2 = 5) ↔ (n = 10) :=
by sorry

end unique_integer_solution_l49_49850


namespace minimum_value_of_reciprocal_sum_of_ab_l49_49157

theorem minimum_value_of_reciprocal_sum_of_ab :
  ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ (real.sqrt 2 = real.geom_mean (8^a) (2^b)) → (∃ (min_val : ℝ), min_val = 5 + 2 * real.sqrt 3 ∧ min_val = min (λ x : ℝ, x = (1 / a + 2 / b))) :=
by
  intros a b h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2
  -- Skipping the proof
  sorry

end minimum_value_of_reciprocal_sum_of_ab_l49_49157


namespace count_four_digit_numbers_l49_49283

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49283


namespace symmetric_point_x_axis_l49_49359

theorem symmetric_point_x_axis (A : ℝ × ℝ × ℝ) : 
  (A = (1, 1, 2)) → (sym_point_z : A.1 = 1 ∧ A.2 = -1 ∧ A.3 = -2) :=
by 
assume h : A = (1, 1, 2)
show A.1 = 1 ∧ A.2 = -1 ∧ A.3 = -2
sorry

end symmetric_point_x_axis_l49_49359


namespace area_MNDC_eq_5_l49_49047

variable (A B C D M N : Point)
variable (parallelogram_lean : Parallelogram A B C D)
variable (area_ABCD : area parallelogram_lean = 12)
variable (midpoint_M : Midpoint B C M)
variable (line_AM : LineThrough A M)
variable (intersection_N : Intersection line_AM (Line B D) N)

theorem area_MNDC_eq_5 :
  area (Quadrilateral M N D C) = 5 := 
  sorry

end area_MNDC_eq_5_l49_49047


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49833

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49833


namespace max_digits_of_product_l49_49650

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49650


namespace arctan_sum_pi_div_two_l49_49808

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l49_49808


namespace greatest_product_digits_l49_49557

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49557


namespace total_cows_l49_49410

-- Define the variables and assumptions
variable {C : ℕ} -- total number of cows

-- Given conditions
axiom h1 : 0.40 * C = 40 * C / 100 -- 40% have a red spot
axiom h2 : 0.60 * C * 0.75 = 45 * C / 100 -- 45% of total cows have no spot
axiom h3 : 63 = 45 * C / 100 -- 63 cows have no spot

-- To prove
theorem total_cows (C : ℕ) (h1 : 0.40 * C = 40 * C / 100) (h2 : 0.60 * C * 0.75 = 45 * C / 100) (h3 : 63 = 45 * C / 100) : C = 140 :=
by
  sorry

end total_cows_l49_49410


namespace three_digit_even_numbers_with_5_and_1_condition_l49_49035

/--
Theorem:
The number of 3-digit even numbers such that if one of the digits is 5,
the next digit must be 1, is 14.
-/
theorem three_digit_even_numbers_with_5_and_1_condition : 
  ∃ n : ℕ, n = 14 ∧ 
    (∀ x : ℕ, 100 ≤ x ∧ x < 1000 ∧ x % 2 = 0 → 
      (∃ d1 d2 d3 : ℕ, d1 * 100 + d2 * 10 + d3 = x ∧ 
       ((d1 = 5 ∧ d2 = 1) ∨ (d2 = 5 ∧ d3 = 1) ∨ d1 ≠ 5 ∧ d2 ≠ 5 ∧ d3 ≠ 5) 
       ∧ d1 ≠ 0 ∧ d3 % 2 = 0)) :=
begin
  use 14,
  split,
  sorry,
  intros x hx,
  cases hx with h₁ h₂,
  cases h₂ with h₃ h₄,
  cases h₄ with d1 hd,
  cases hd with d2 hd',
  cases hd' with d3 hxyz,
  cases (nat.exists_eq_succ_of_ne_zero (ne_of_gt (lt_of_le_of_lt h₁ (lt_trans h₃ (lt_add_one 1000))) )),
  sorry
end

end three_digit_even_numbers_with_5_and_1_condition_l49_49035


namespace number_of_four_digit_numbers_l49_49270

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49270


namespace cheese_cut_indefinite_l49_49064

theorem cheese_cut_indefinite (w : ℝ) (R : ℝ) (h : ℝ) :
  R = 0.5 →
  (∀ a b c d : ℝ, a > b → b > c → c > d →
    (∃ h, h < min (a - d) (d - c) ∧
     (d + h < a ∧ d - h > c))) →
  ∃ l1 l2 : ℕ → ℝ, (∀ n, l1 (n + 1) > l2 (n) ∧ l1 n > R * l2 (n)) :=
sorry

end cheese_cut_indefinite_l49_49064


namespace mice_count_after_two_breedings_l49_49793

theorem mice_count_after_two_breedings :
  let initial_mice := 8
  let pups_per_mouse := 6
  let first_gen_pups := initial_mice * pups_per_mouse
  let total_after_first_gen := initial_mice + first_gen_pups
  let second_gen_pups := total_after_first_gen * pups_per_mouse
  let total_after_second_gen := total_after_first_gen + second_gen_pups
  let eaten_pups := initial_mice * 2
  in total_after_second_gen - eaten_pups = 280 :=
by
  sorry

end mice_count_after_two_breedings_l49_49793


namespace max_a4_l49_49906

theorem max_a4 (a1 d a4 : ℝ) 
  (h1 : 2 * a1 + 3 * d ≥ 5) 
  (h2 : a1 + 2 * d ≤ 3) 
  (ha4 : a4 = a1 + 3 * d) : 
  a4 ≤ 4 := 
by 
  sorry

end max_a4_l49_49906


namespace solve_a_plus_b_l49_49953

theorem solve_a_plus_b (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : 143 * a + 500 * b = 2001) : a + b = 9 :=
by
  -- Add proof here
  sorry

end solve_a_plus_b_l49_49953


namespace find_a_plus_b_l49_49842

theorem find_a_plus_b (a b : ℝ) 
  (h1 : 2 = a - b / 2) 
  (h2 : 6 = a - b / 3) : 
  a + b = 38 := by
  sorry

end find_a_plus_b_l49_49842


namespace range_of_a_l49_49156

variable (a x : ℝ)

-- Condition p: ∀ x ∈ [1, 2], x^2 - a ≥ 0
def p : Prop := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

-- Condition q: ∃ x ∈ ℝ, x^2 + 2 * a * x + 2 - a = 0
def q : Prop := ∃ x, x^2 + 2 * a * x + 2 - a = 0

-- The proof goal given p ∧ q: a ≤ -2 or a = 1
theorem range_of_a (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := sorry

end range_of_a_l49_49156


namespace min_value_of_inverse_distances_l49_49986

variables {A B C P : Type} [Inhabited P]
variables [IsRightTriangle A B C] (AC BC: ℝ) (HAC: AC = 4) (HBC: BC = 1)
          (d_1 d_2: ℝ) (Hd1: ∀ P, distance_to_leg P AC = d_1) (Hd2: ∀ P, distance_to_leg P BC = d_2)

theorem min_value_of_inverse_distances : (∃ P : P, P ∈ hypotenuse A B \ {A, B} → 
  (1 / d_1 + 1 / d_2) = 9 / 4) := 
by 
  sorry

end min_value_of_inverse_distances_l49_49986


namespace min_value_sqrt_sum_l49_49143

open Real

theorem min_value_sqrt_sum (x : ℝ) : 
    ∃ c : ℝ, (∀ x : ℝ, c ≤ sqrt (x^2 - 4 * x + 13) + sqrt (x^2 - 10 * x + 26)) ∧ 
             (sqrt ((17/4)^2 - 4 * (17/4) + 13) + sqrt ((17/4)^2 - 10 * (17/4) + 26) = 5 ∧ c = 5) := 
by
  sorry

end min_value_sqrt_sum_l49_49143


namespace smallest_x_l49_49129

theorem smallest_x (x: ℕ) (hx: x > 0) (h: 11^2021 ∣ 5^(3*x) - 3^(4*x)) : 
  x = 11^2020 := sorry

end smallest_x_l49_49129


namespace four_digit_number_count_l49_49205

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49205


namespace max_digits_of_product_l49_49646

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49646


namespace equation_of_line_m_l49_49856

-- Given conditions
def point (α : Type*) := α × α

def l_eq (p : point ℝ) : Prop := p.1 + 3 * p.2 = 7 -- Equation of line l
def m_intercept : point ℝ := (1, 2) -- Intersection point of l and m
def q : point ℝ := (2, 5) -- Point Q
def q'' : point ℝ := (5, 0) -- Point Q''

-- Proving the equation of line m
theorem equation_of_line_m (m_eq : point ℝ → Prop) :
  (∀ P : point ℝ, m_eq P ↔ P.2 = 2 * P.1 - 2) ↔
  (∃ P : point ℝ, m_eq P ∧ P = (5, 0)) :=
sorry

end equation_of_line_m_l49_49856


namespace cosine_of_3pi_over_2_l49_49112

theorem cosine_of_3pi_over_2 : Real.cos (3 * Real.pi / 2) = 0 := by
  sorry

end cosine_of_3pi_over_2_l49_49112


namespace center_of_circle_l49_49122

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := 4 * x^2 - 8 * x + 4 * y^2 - 24 * y - 36 = 0

-- Define what it means to be the center of the circle, which is (h, k)
def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_eq x y ↔ (x - h)^2 + (y - k)^2 = 1

-- The statement that we need to prove
theorem center_of_circle : is_center 1 3 :=
sorry

end center_of_circle_l49_49122


namespace find_A_l49_49738

def set_A (a b c : ℝ) (h : a * b * c ≠ 0) : ℝ :=
  (a / |a|) + (|b| / b) + (|c| / c) + ((a * b * c) / |a * b * c|)

theorem find_A :
  (A = {-4, 0, 4}) :=
by {
  sorry
}

end find_A_l49_49738


namespace fraction_of_money_left_l49_49796

theorem fraction_of_money_left (m c : ℝ) 
   (h1 : (1/5) * m = (1/3) * c) :
   (m - ((3/5) * m) = (2/5) * m) := by
  sorry

end fraction_of_money_left_l49_49796


namespace crease_length_squared_l49_49767

noncomputable def semicircle_crease_squared : ℝ := 
  let r := 4
  let ratio := 7:1
  36

theorem crease_length_squared (r : ℝ) (ratio : ℤ) (h_ratio : ratio = 7:1) :
  let ℓ := semicircle_crease_squared in
  ℓ ^ 2 = 36 :=
by 
  -- Assumptions: 
  -- radius r = 4
  -- point of tangency ratio = 7:1
  -- proof of ℓ^2 = 36 to be filled
  sorry

end crease_length_squared_l49_49767


namespace g_is_odd_l49_49106

def g (x : ℝ) : ℝ := (1 / (3^x - 1)) + (1 / 3)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  have h1 : g (-x) = (1 / (3^(-x) - 1)) + (1 / 3) := by sorry
  have h2 : g (-x) = ((1/3^x - 1)) + (1 / 3) := by sorry
  have h3 : g (-x) = -g x := by sorry
  exact h3

end g_is_odd_l49_49106


namespace num_people_present_l49_49034

-- Given conditions
def associatePencilCount (A : ℕ) : ℕ := 2 * A
def assistantPencilCount (B : ℕ) : ℕ := B
def associateChartCount (A : ℕ) : ℕ := A
def assistantChartCount (B : ℕ) : ℕ := 2 * B

def totalPencils (A B : ℕ) : ℕ := associatePencilCount A + assistantPencilCount B
def totalCharts (A B : ℕ) : ℕ := associateChartCount A + assistantChartCount B

-- Prove the total number of people present
theorem num_people_present (A B : ℕ) (h1 : totalPencils A B = 11) (h2 : totalCharts A B = 16) : A + B = 9 :=
by
  sorry

end num_people_present_l49_49034


namespace greatest_product_digits_l49_49670

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49670


namespace counterfeit_weight_reduction_l49_49754

/-- Given that:
    1. A dealer sells at cost price using a counterfeit weight.
    2. The counterfeit weight is some percentage less than the real weight.
    3. The dealer adds 35% impurities.
    4. The net profit percentage is 68.75%.
    
Prove that the percentage by which the counterfeit weight is less than the real weight is 
68.75%. -/
theorem counterfeit_weight_reduction :
  ∀ (real_weight_cost dealer_selling_price profit_percentage impurity_percentage : ℝ),
  dealer_selling_price = real_weight_cost →
  impurity_percentage = 35 →
  profit_percentage = 68.75 →
  ∃ (x : ℝ), (dealer_selling_price - real_weight_cost * (1 - x / 100)) / (real_weight_cost * (1 - x / 100)) = profit_percentage / 100
  ∧ x = 68.75 := by
  intros real_weight_cost dealer_selling_price profit_percentage impurity_percentage
  assume h1 h2 h3
  use 68.75
  sorry

end counterfeit_weight_reduction_l49_49754


namespace limit_cos_sin_squared_identity_l49_49800

theorem limit_cos_sin_squared_identity :
  tendsto (fun x => (1 + cos (3 * x)) / (sin (7 * x))^2) (𝓝 π) (𝓝 (9 / 98)) :=
by
  sorry

end limit_cos_sin_squared_identity_l49_49800


namespace nonzero_digits_right_of_decimal_l49_49099

noncomputable def fraction := 120 / (2^4 * 5^9)

theorem nonzero_digits_right_of_decimal :
  let decimal_rep := (fraction : ℚ).approx (10^8)
  (decimal_rep.fract).non_zero_digits_count = 3 := 
sorry

end nonzero_digits_right_of_decimal_l49_49099


namespace count_four_digit_numbers_l49_49306

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49306


namespace trip_to_office_duration_l49_49403

noncomputable def distance (D : ℝ) : Prop :=
  let T1 := D / 58
  let T2 := D / 62
  T1 + T2 = 3

theorem trip_to_office_duration (D : ℝ) (h : distance D) : D / 58 = 1.55 :=
by sorry

end trip_to_office_duration_l49_49403


namespace sequence_value_2023_l49_49849

noncomputable def a : ℕ → ℚ
| 1     := 2
| 2     := 5/11
| (n+3) := (3 * a (n+1) * a (n+2)) / (4 * a (n+1) - a (n+2))

theorem sequence_value_2023 :
  let p := 2 in
  let q := 8115 in
  p.coprime q →
  p + q = 8117 := by
    sorry

end sequence_value_2023_l49_49849


namespace four_digit_number_count_l49_49200

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49200


namespace solution_a_2011_l49_49918

def f : ℝ → ℝ
def a_n : ℕ → ℝ
axiom even_f : ∀ x, f x = f (-x)
axiom functional_equation : ∀ x, f (1 + x) = f (3 - x)
axiom definition_for_negatives : ∀ x, -2 ≤ x ∧ x ≤ 0 → f x = 3^x
axiom definition_a_n : ∀ n : ℕ, 0 < n → a_n n = f n

theorem solution_a_2011 : a_n 2011 = 1 / 3 := 
by
  sorry

end solution_a_2011_l49_49918


namespace part1_part2_l49_49894

-- Given function
def f (m x : ℝ) := Real.exp x - m * x

-- Condition for part 1
def condition1 (m x : ℝ) := (x - 2) * f m x + m * x^2 + 2 > 0

-- Solution for part 1
theorem part1 (m : ℝ) : (∀ x > 0, condition1 m x) ↔ m ∈ set.Ici 0.5 := sorry

-- Zeros of the function
def is_zero_of_f (m x : ℝ) := f m x = 0

-- Condition for part 2
def condition2 (m x1 x2 : ℝ) := is_zero_of_f m x1 ∧ is_zero_of_f m x2

-- Solution for part 2
theorem part2 (m x1 x2 : ℝ) (h : condition2 m x1 x2) : x1 + x2 = 2 := sorry

end part1_part2_l49_49894


namespace prime_sol_is_7_l49_49119

theorem prime_sol_is_7 (p : ℕ) (x y : ℕ) (hp : Nat.Prime p) 
  (hx : p + 1 = 2 * x^2) (hy : p^2 + 1 = 2 * y^2) : 
  p = 7 := 
  sorry

end prime_sol_is_7_l49_49119


namespace quadratic_root_q_value_l49_49332

theorem quadratic_root_q_value
  (p q : ℝ)
  (h1 : ∃ r : ℝ, r = -3 ∧ 3 * r^2 + p * r + q = 0)
  (h2 : ∃ s : ℝ, -3 + s = -2) :
  q = -9 :=
sorry

end quadratic_root_q_value_l49_49332


namespace average_marks_all_students_l49_49730

theorem average_marks_all_students (A₁ A₂ : ℕ) (n₁ n₂ : ℕ) (H1 : n₁ = 22) (H2 : n₂ = 28) (H3 : A₁ = 40) (H4 : A₂ = 60) :
  (n₁ * A₁ + n₂ * A₂) / (n₁ + n₂) = 51.2 := by
  -- Proceed with the proof here
  sorry

end average_marks_all_students_l49_49730


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49570

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49570


namespace tan_cot_sum_relation_l49_49145

open Real

theorem tan_cot_sum_relation (n : ℕ) (θ : Fin n → ℝ) 
  (hθ : ∀ i, 0 < θ i ∧ θ i < π / 2) :
  (∑ i, tan (θ i)) * (∑ i, cot (θ i)) ≥ (∑ i, sin (θ i)) * (∑ i, csc (θ i)) :=
  sorry

end tan_cot_sum_relation_l49_49145


namespace election_total_votes_l49_49982

noncomputable def total_votes (V : ℝ) :=
  let valid_votes := 0.85 * V in
  let votes_for_A := 0.85 * valid_votes in
  votes_for_A = 404600

theorem election_total_votes : ∃ V : ℝ, total_votes V ∧ V = 560000 := by
  sorry

end election_total_votes_l49_49982


namespace P_intersection_complement_Q_l49_49189

-- Define sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }
def Q : Set ℝ := { x | x^2 ≥ 4 }

-- Prove the required intersection
theorem P_intersection_complement_Q : P ∩ (Set.univ \ Q) = { x | 0 ≤ x ∧ x < 2 } :=
by
  -- Proof will be inserted here
  sorry

end P_intersection_complement_Q_l49_49189


namespace calculate_A_l49_49028

noncomputable def A (x : ℝ) (h : 0 < x) : ℝ :=
  (sqrt 3 * x^(3/2) - 5 * x^(1/3) + 5 * x^(4/3) - sqrt 3 * sqrt x) /
  ((sqrt (3 * x + 10 * sqrt (3 * x^(5/6)) + 25 * x^(2/3))) * sqrt (1 - 2 / x + 1 / x^2))

theorem calculate_A (x : ℝ) (h : 0 < x) :
  (0 < x ∧ x < 1 → 2.331 * A x h = - x) ∧ (x > 1 → 2.331 * A x h = x) :=
by
  sorry

end calculate_A_l49_49028


namespace greatest_digits_product_l49_49590

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49590


namespace greatest_product_digits_l49_49664

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49664


namespace greatest_digits_product_l49_49582

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49582


namespace prime_constraint_unique_solution_l49_49120

theorem prime_constraint_unique_solution (p x y : ℕ) (h_prime : Prime p)
  (h1 : p + 1 = 2 * x^2)
  (h2 : p^2 + 1 = 2 * y^2) :
  p = 7 :=
by
  sorry

end prime_constraint_unique_solution_l49_49120


namespace sin_range_l49_49155

theorem sin_range (p : Prop) (q : Prop) :
  (¬ ∃ x : ℝ, Real.sin x = 3/2) → (∀ x : ℝ, x^2 - 4 * x + 5 > 0) → (¬p ∧ q) :=
by
  sorry

end sin_range_l49_49155


namespace area_of_45_45_90_right_triangle_l49_49456

theorem area_of_45_45_90_right_triangle (h : ℝ) (hypotenuse : h = 10 * real.sqrt 2)
    (angle : ∃ θ, θ = real.pi / 4) :
    ∃ A, A = 50 := sorry

end area_of_45_45_90_right_triangle_l49_49456


namespace greatest_digits_product_l49_49593

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49593


namespace range_of_f_l49_49398

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1 - x) else 1 - Real.logb 2 x

theorem range_of_f (x : ℝ) : f x ≤ 2 → x ∈ set.Ici 0 :=
by
  sorry

end range_of_f_l49_49398


namespace max_digits_of_product_l49_49645

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49645


namespace no_a_solution_for_f_extreme_points_and_zeros_l49_49928

noncomputable def f (x a : ℝ) : ℝ := log x - a * (x - (1 / x))

theorem no_a_solution_for_f_extreme_points_and_zeros 
  {a x1 x2 t1 t2 t3 : ℝ} 
  (hx1_lt_x2 : x1 < x2) 
  (ht1_lt_t2 : t1 < t2) 
  (ht2_lt_t3 : t2 < t3) 
  (hfx1 : f x1 a = 0) 
  (hfx2 : f x2 a = 0) 
  (hft1 : f t1 a = 0) 
  (hft2 : f t2 a = 0) 
  (hft3 : f t3 a = 0)
  (hnum_extreme: ¬ (0 < x1) ∨ ¬ (x2 < 0)) 
  (hnum_zeros: ¬ (0 < t1) ∨ ¬ (t3 < 0)) :
  ¬ ∃ a : ℝ, (f x2 a - f x1 a) / 2 = (2 * (t3 - t1)) / (log t3 - log t1) - 2 := 
  sorry

end no_a_solution_for_f_extreme_points_and_zeros_l49_49928


namespace multiple_of_marriage_game_l49_49855

theorem multiple_of_marriage_game :
  ∀ (age_sarah letters_sarah age_marriage multiple : ℕ),
    age_sarah = 9 →
    letters_sarah = 5 →
    age_marriage = 23 →
    letters_sarah + (age_sarah * multiple) = age_marriage →
    multiple = 2 :=
by {
  intros age_sarah letters_sarah age_marriage multiple,
  intros h1 h2 h3 h4,
  sorry
}

end multiple_of_marriage_game_l49_49855


namespace range_MN_l49_49432

variables {a x MN : ℝ}

-- Definitions of points M and N such that AM = FN = x
def A := (0,0)
def B := (a,0)
def C := (a,a)
def D := (0,a)
def E := (a,a)
def F := (a,-a)

-- Conditions for the problem
def is_square (s a : ℝ) := s = a -- Simplified representation for square side length
def are_squares_inclined (θ : ℝ) := θ = 120.0 -- Simplified representation for angle between the planes

-- Condition for M on diagonal AC and N on BF with AM = FN = x
def is_on_diagonal (p q : ℝ × ℝ) := p.1 = q.1 ∧ p.2 = q.2
def AM_FN_eq_x (p q : ℝ × ℝ) (x : ℝ) := dist A p = x ∧ dist F q = x

-- Main theorem to prove
theorem range_MN (a : ℝ) (x : ℝ) :
  is_square (dist B C) a ∧ is_square (dist E F) a
  ∧ are_squares_inclined 120.0
  ∧ ∃ M N : ℝ × ℝ, is_on_diagonal M C ∧ is_on_diagonal N F ∧ AM_FN_eq_x M N x
  → ( √(x^2 - √2 * a * x + a^2) ) ∈ Icc (√3 / 2 * a) a :=
sorry

end range_MN_l49_49432


namespace sin_alpha_plus_beta_l49_49889

theorem sin_alpha_plus_beta (α β : ℝ) (h1 : real.cos α = 4 / 5) (h2 : real.cos β = 3 / 5) 
    (h3 : β ∈ set.Ioo (3 * real.pi / 2) (2 * real.pi)) (h4 : 0 < α ∧ α < β) 
    : real.sin (α + β) = -7 / 25 :=
sorry

end sin_alpha_plus_beta_l49_49889


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49834

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49834


namespace count_four_digit_numbers_l49_49288

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49288


namespace original_cube_volume_l49_49418

theorem original_cube_volume (a : ℕ) (h : (a + 2) * (a + 1) * (a - 1) + 6 = a^3) : a = 2 :=
by sorry

example : 2^3 = 8 := by norm_num

end original_cube_volume_l49_49418


namespace four_digit_numbers_count_l49_49211

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49211


namespace external_ratio_division_l49_49055

open EuclideanGeometry

variables {A B C D : Point} {a b c : ℝ}

-- Triangle ABC is inscribed in a circle
axiom circumscribed_triangle : ¬Collinear A B C ∧ ∃ O : Point, O ∈ circle(A, B, C)

-- The tangent at C intersects the opposite side AB at point D
axiom tangent_intersection_D : IsTangent (circle(A, B, C)) C D ∧ D ∈ Line(A, B)

-- Define the sides of the triangle
axiom BC_eq_a : dist B C = a
axiom CA_eq_b : dist C A = b
axiom AB_eq_c : dist A B = c

-- Prove the required ratio
theorem external_ratio_division : 
  ∃ (x y : ℝ), (x + y = c) ∧ 
  (AD = x) ∧ (DB = y) ∧ 
  (x / y = (b^2 : ℝ) / (a^2 : ℝ)) := sorry

end external_ratio_division_l49_49055


namespace shanghai_expo_visitors_l49_49413

theorem shanghai_expo_visitors :
  505000 = 5.05 * 10^5 :=
by
  sorry

end shanghai_expo_visitors_l49_49413


namespace complex_series_sum_l49_49377

theorem complex_series_sum (ω : ℂ) (h₁ : ω^7 = 1) (h₂ : ω ≠ 1) :
  (ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 + ω^32 + 
   ω^34 + ω^36 + ω^38 + ω^40 + ω^42 + ω^44 + ω^46 + ω^48 + ω^50 + 
   ω^52 + ω^54) = -1 :=
by
  sorry

end complex_series_sum_l49_49377


namespace parallelogram_area_eq_sqrt2_l49_49376

noncomputable def area_of_parallelogram (r s : ℝ^3) (hr : ‖r‖ = 1) (hs : ‖s‖ = 1) (angle_rs : real.angle_between r s = real.pi / 4) : ℝ :=
  let c := -r + s
  let d := 2 * r + 2 * s
  ‖c × d‖

theorem parallelogram_area_eq_sqrt2 (r s : ℝ^3) (hr : ‖r‖ = 1) (hs : ‖s‖ = 1) (angle_rs : real.angle_between r s = real.pi / 4) :
  area_of_parallelogram r s hr hs angle_rs = real.sqrt 2 :=
sorry

end parallelogram_area_eq_sqrt2_l49_49376


namespace four_digit_numbers_count_l49_49229

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49229


namespace nested_expression_equals_4094_l49_49089

-- Define the nested expression
def nested_expression : ℕ :=
  2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2 * (1 + 2)))))))))))

-- The theorem to prove
theorem nested_expression_equals_4094 : nested_expression = 4094 :=
by
  sorry

end nested_expression_equals_4094_l49_49089


namespace max_digits_of_product_l49_49644

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49644


namespace area_of_region_S_l49_49431

-- Define the square WXYZ with given side length and angle at W
def square_side_length : ℝ := 4
def angle_W : ℝ := 90

-- Define region S as the set of points inside the square closer to W than any other vertex
def region_S (p : ℝ × ℝ) : Prop :=
  let (x, y) := p in
  (x ≤ 2) ∧ (y ≤ 2)

-- The proof problem: Prove that the area of region S is 2
theorem area_of_region_S : 
  let square_area := square_side_length * square_side_length in
  let region_S_area := square_area / 2 in
  region_S_area = 2 :=
by
  sorry

end area_of_region_S_l49_49431


namespace volume_of_removed_tetrahedra_correct_l49_49846

noncomputable def prism_dimensions : ℝ × ℝ × ℝ := (2, 3, 4)

def triangle_side_length (x : ℝ) : ℝ := x * sqrt 2

def calculate_x (prism_min_dimension : ℝ) : ℝ :=
  let num := prism_min_dimension
  in num / (1 + sqrt 2)

def tetrahedron_height (x : ℝ) : ℝ := 2 - x

def tetrahedron_base_area (x : ℝ) : ℝ :=
  let t_side := triangle_side_length x
  in (sqrt 3 / 4) * t_side^2

def tetrahedron_volume (x : ℝ) : ℝ :=
  (1 / 3) * (tetrahedron_base_area x) * (tetrahedron_height x)

def total_volume_of_removed_tetrahedra : ℝ :=
  let x := calculate_x 2 -- using the smallest dimension of the prism
  in 8 * tetrahedron_volume x

theorem volume_of_removed_tetrahedra_correct :
  total_volume_of_removed_tetrahedra = (16 * sqrt 3 * (2 - sqrt 2) * (4 - 2 * sqrt 2)) / 3 :=
by sorry

end volume_of_removed_tetrahedra_correct_l49_49846


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49637

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49637


namespace fewer_red_pens_than_pencils_l49_49973

theorem fewer_red_pens_than_pencils :
  (∃ B K P R X : ℕ,
    P = 8 ∧
    K = B + 10 ∧
    B = 2 * P ∧
    R = P - X ∧
    B + K + R = 48 ∧
    X = 2) :=
begin
  sorry
end

end fewer_red_pens_than_pencils_l49_49973


namespace expression_is_negative_seven_l49_49803

def simplify_expression : ℝ := 
  (Real.sqrt 15)^2 / Real.sqrt 3 * (1 / Real.sqrt 3) - Real.sqrt 6 * Real.sqrt 24

theorem expression_is_negative_seven : simplify_expression = -7 := 
by
  sorry

end expression_is_negative_seven_l49_49803


namespace equal_share_each_get_l49_49420

-- Definitions of initial conditions
def Paityn_red_hats := 20
def Paityn_blue_hats := 24
def Zola_red_hats := (4 / 5) * Paityn_red_hats
def Zola_blue_hats := 2 * Paityn_blue_hats
def total_hats := Paityn_red_hats + Paityn_blue_hats + Zola_red_hats + Zola_blue_hats

-- Theorem statement
theorem equal_share_each_get (Paityn_red_hats = 20) (Paityn_blue_hats = 24) :
  (total_hats / 2) = 54 := 
  by 
    simp [total_hats, Paityn_red_hats, Paityn_blue_hats, Zola_red_hats, Zola_blue_hats]
    sorry

end equal_share_each_get_l49_49420


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49566

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49566


namespace find_x_for_perpendicular_and_parallel_l49_49195

noncomputable def a : ℝ × ℝ × ℝ := (2, -1, 3)
noncomputable def b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def parallel (u v : ℝ × ℝ × ℝ) : Prop := (u.1 / v.1 = u.2 / v.2) ∧ (u.2 / v.2 = u.3 / v.3)

theorem find_x_for_perpendicular_and_parallel :
  (dot_product a (b (10/3)) = 0) ∧ (parallel a (b (-6))) :=
by
  sorry

end find_x_for_perpendicular_and_parallel_l49_49195


namespace max_digits_in_product_l49_49500

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49500


namespace max_digits_of_product_l49_49547

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49547


namespace count_four_digit_numbers_l49_49311

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49311


namespace greatest_product_digits_l49_49656

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49656


namespace exp_gt_f_n_y_between_0_and_x_l49_49172

open Real

noncomputable def f_n (x : ℝ) (n : ℕ) : ℝ :=
  (Finset.range (n + 1)).sum (λ k => x^k / k.factorial)

theorem exp_gt_f_n (x : ℝ) (n : ℕ) (h1 : 0 < x) :
  exp x > f_n x n :=
sorry

theorem y_between_0_and_x (x : ℝ) (n : ℕ) (y : ℝ)
  (h1 : 0 < x)
  (h2 : exp x = f_n x n + x^(n+1) / (n + 1).factorial * exp y) :
  0 < y ∧ y < x :=
sorry

end exp_gt_f_n_y_between_0_and_x_l49_49172


namespace greatest_product_digits_l49_49669

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49669


namespace intersection_of_A_and_B_l49_49908

open Set

-- Definitions of sets A and B as per conditions in the problem
def A := {x : ℝ | -1 < x ∧ x < 2}
def B := {x : ℝ | -3 < x ∧ x ≤ 1}

-- The proof statement that A ∩ B = {x | -1 < x ∧ x ≤ 1}
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l49_49908


namespace intersecting_lines_ratio_l49_49193

theorem intersecting_lines_ratio (k1 k2 a : ℝ) (h1 : k1 * a + 4 = 0) (h2 : k2 * a - 2 = 0) : k1 / k2 = -2 :=
by
    sorry

end intersecting_lines_ratio_l49_49193


namespace two_cities_connect_all_others_l49_49347

theorem two_cities_connect_all_others (n : ℕ) (h_n : n = 100)
    (has_road : ∀ (u v : fin n), Prop)
    (condition_1 : ∀ (s : finset (fin n)), s.card = 4 → (∃ (u v ∈ s), has_road u v))
    (condition_2 : ¬ ∃ (p : list (fin n)), p.nodup ∧ (list.length p = n) ∧ (∀ (i : ℕ) (hi : i < p.length - 1), has_road (p.nth_le i hi) (p.nth_le (i + 1) (nat.lt_succ_self _)))) :
    ∃ (a b : fin n), ∀ (c : fin n), c ≠ a → c ≠ b → (has_road a c ∨ has_road b c) :=
sorry

end two_cities_connect_all_others_l49_49347


namespace GE_eq_GH_l49_49735

variables (A B C D E F G H : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
          [Inhabited E] [Inhabited F] [Inhabited G] [Inhabited H]
          
variables (AC : Line A C) (AB : Line A B) (BE : Line B E) (DE : Line D E)
          (BG : Line B G) (AF : Line A F) (DE' : Line D E') (angleC : Angle C = 90)

variables (circB : Circle B BC) (tangentDE : Tangent DE circB E) (perpAB : Perpendicular AC AB)
          (intersectionF : Intersect (PerpendicularLine C AB) BE F)
          (intersectionG : Intersect AF DE G) (intersectionH : Intersect (ParallelLine A BG) DE H)

theorem GE_eq_GH : GE = GH := sorry

end GE_eq_GH_l49_49735


namespace prime_constraint_unique_solution_l49_49121

theorem prime_constraint_unique_solution (p x y : ℕ) (h_prime : Prime p)
  (h1 : p + 1 = 2 * x^2)
  (h2 : p^2 + 1 = 2 * y^2) :
  p = 7 :=
by
  sorry

end prime_constraint_unique_solution_l49_49121


namespace hyperbola_k_range_l49_49926

theorem hyperbola_k_range {k : ℝ} 
  (h : ∀ x y : ℝ, x^2 + (k-1)*y^2 = k+1 → (k > -1 ∧ k < 1)) : 
  -1 < k ∧ k < 1 :=
by 
  sorry

end hyperbola_k_range_l49_49926


namespace triangle_area_equal_eight_l49_49483

-- Definition of the lines
def L1 : ℝ → ℝ := λ x, -x + 4
def L2 : ℝ → ℝ := λ x, 3 * x
def L3 (x y : ℝ) : Prop := x - y = 2

-- Points of intersection
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (-1, -3)
def C : ℝ × ℝ := (3, 1)

-- Area calculation using determinant based formula
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

-- Main theorem statement to prove
theorem triangle_area_equal_eight : triangle_area A B C = 8 := by
  sorry

end triangle_area_equal_eight_l49_49483


namespace color_cartridge_cost_l49_49477

theorem color_cartridge_cost :
  ∃ C : ℝ, 
  (1 * 27) + (3 * C) = 123 ∧ C = 32 :=
by
  sorry

end color_cartridge_cost_l49_49477


namespace men_work_days_l49_49053

theorem men_work_days (M : ℕ) (W : ℕ) (h : W / (M * 40) = W / ((M - 5) * 50)) : M = 25 :=
by
  -- Will add the proof later
  sorry

end men_work_days_l49_49053


namespace arctan_triangle_complementary_l49_49814

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l49_49814


namespace Mitzel_has_26_left_l49_49408

variable (A : ℝ) (spent : ℝ)

-- Given conditions
def spent_percent := 0.35
def spent_amount := 14
def total_allowance := spent_amount / spent_percent
#check total_allowance -- to verify the computation

-- The final amount left after spending
def amount_left := total_allowance - spent_amount

theorem Mitzel_has_26_left : amount_left = 26 := by
  -- This is where the proof would go
  sorry

end Mitzel_has_26_left_l49_49408


namespace hindi_probability_l49_49049

noncomputable def probability_hindi (N T E B : ℕ) : ℚ :=
  let hindi_speakers := N - (T + E - B)
  hindi_speakers / N

theorem hindi_probability : 
  let N := 1024 in
  let T := 720 in
  let E := 562 in
  let B := 346 in
  probability_hindi N T E B ≈ 0.0859375 :=
by
  sorry

end hindi_probability_l49_49049


namespace option_a_matches_sqrt6_type_l49_49780

-- Define the various square roots
def sqrt6 := Real.sqrt 6
def sqrt2_div_3 := Real.sqrt (2 / 3)
def sqrt12 := Real.sqrt 12
def sqrt18 := Real.sqrt 18
def sqrt30 := Real.sqrt 30

-- State that we need to prove sqrt2_div_3 is of the same type as sqrt6
theorem option_a_matches_sqrt6_type :
  sqrt2_div_3 = Real.sqrt 2 / Real.sqrt 3 :=
by sorry

end option_a_matches_sqrt6_type_l49_49780


namespace greatest_possible_digits_in_product_l49_49520

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49520


namespace find_x_condition_l49_49879

theorem find_x_condition (x : ℚ) :
  (∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) → x = 3 / 2 :=
begin
  sorry
end

end find_x_condition_l49_49879


namespace permutation_four_members_l49_49975

theorem permutation_four_members :
  let members := ["Alice", "Bob", "Carol", "Dave"]
  let roles := ["president", "vice-president", "secretary", "treasurer"]
  (members.permutations.size = 24) :=
by
  sorry

end permutation_four_members_l49_49975


namespace y3_gt_y1_and_y1_gt_y2_l49_49888

noncomputable def y1 : ℝ := 0.9 ^ 0.2 
noncomputable def y2 : ℝ := 0.9 ^ 0.4 
noncomputable def y3 : ℝ := 1.2 ^ 0.1

theorem y3_gt_y1_and_y1_gt_y2 : y3 > y1 ∧ y1 > y2 :=
by
  sorry

end y3_gt_y1_and_y1_gt_y2_l49_49888


namespace mice_count_after_two_breedings_l49_49794

theorem mice_count_after_two_breedings :
  let initial_mice := 8
  let pups_per_mouse := 6
  let first_gen_pups := initial_mice * pups_per_mouse
  let total_after_first_gen := initial_mice + first_gen_pups
  let second_gen_pups := total_after_first_gen * pups_per_mouse
  let total_after_second_gen := total_after_first_gen + second_gen_pups
  let eaten_pups := initial_mice * 2
  in total_after_second_gen - eaten_pups = 280 :=
by
  sorry

end mice_count_after_two_breedings_l49_49794


namespace greatest_product_digits_l49_49552

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49552


namespace max_digits_of_product_l49_49539

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49539


namespace factor_2210_two_digit_l49_49319

theorem factor_2210_two_digit :
  (∃ (a b : ℕ), a * b = 2210 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99) ∧
  (∃ (c d : ℕ), c * d = 2210 ∧ 10 ≤ c ∧ c ≤ 99 ∧ 10 ≤ d ∧ d ≤ 99) ∧
  (∀ (x y : ℕ), x * y = 2210 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 → 
   ((x = c ∧ y = d) ∨ (x = d ∧ y = c) ∨ (x = a ∧ y = b) ∨ (x = b ∧ y = a))) :=
sorry

end factor_2210_two_digit_l49_49319


namespace min_questions_l49_49083

theorem min_questions (num_people : ℕ) (q : ℕ)
  (h1 : ∀ (S : set (fin num_people)), S.card = 5 → ∀ (i : fin q), ∃ (j ∈ S), j answers i)
  (h2 : ∀ (T : set (fin num_people)), T.card = 4 → ∃ (i : fin q), ∀ (j ∈ T), ¬(j answers i)) :
  q = 210 := 
sorry

end min_questions_l49_49083


namespace polynomial_positive_values_l49_49870

noncomputable def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

theorem polynomial_positive_values :
  ∀ (z : ℝ), (∃ (x y : ℝ), P x y = z) ↔ z > 0 :=
by
  sorry

end polynomial_positive_values_l49_49870


namespace projection_areas_are_correct_l49_49997

noncomputable def S1 := 1/2 * 2 * 2
noncomputable def S2 := 1/2 * 2 * Real.sqrt 2
noncomputable def S3 := 1/2 * 2 * Real.sqrt 2

theorem projection_areas_are_correct :
  S3 = S2 ∧ S3 ≠ S1 :=
by
  sorry

end projection_areas_are_correct_l49_49997


namespace g_product_of_roots_l49_49389

def f (x : ℂ) : ℂ := x^6 + x^3 + 1
def g (x : ℂ) : ℂ := x^2 + 1

theorem g_product_of_roots (x_1 x_2 x_3 x_4 x_5 x_6 : ℂ) 
    (h1 : ∀ x, (x - x_1) * (x - x_2) * (x - x_3) * (x - x_4) * (x - x_5) * (x - x_6) = f x) :
    g x_1 * g x_2 * g x_3 * g x_4 * g x_5 * g x_6 = 1 :=
by 
    sorry

end g_product_of_roots_l49_49389


namespace infinite_product_value_l49_49861

noncomputable def infinite_product : ℝ :=
  ∏' (n : ℕ), 3^(n/(2^n * n))

theorem infinite_product_value :
  infinite_product = 15.5884572681 :=
sorry

#eval infinite_product -- should evaluate to 15.5884572681 (approximately)

end infinite_product_value_l49_49861


namespace complex_sum_series_l49_49380

theorem complex_sum_series (ω : ℂ) (h1 : ω ^ 7 = 1) (h2 : ω ≠ 1) :
  ω ^ 16 + ω ^ 18 + ω ^ 20 + ω ^ 22 + ω ^ 24 + ω ^ 26 + ω ^ 28 + ω ^ 30 + 
  ω ^ 32 + ω ^ 34 + ω ^ 36 + ω ^ 38 + ω ^ 40 + ω ^ 42 + ω ^ 44 + ω ^ 46 +
  ω ^ 48 + ω ^ 50 + ω ^ 52 + ω ^ 54 = -1 :=
sorry

end complex_sum_series_l49_49380


namespace greatest_digits_product_l49_49594

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49594


namespace count_four_digit_numbers_l49_49289

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49289


namespace height_of_intersection_l49_49340

theorem height_of_intersection (h1 h2 d : ℝ) (h1_eq : h1 = 30) (h2_eq : h2 = 70) (d_eq : d = 150) : ∃ y, y = 21 :=
by
  let eq1 := λ x : ℝ, -1 / 5 * x + 30
  let eq2 := λ x : ℝ, 7 / 15 * x
  have xy_eq : ∃ x : ℝ, eq1 x = eq2 x := sorry
  cases xy_eq with x hx
  use eq2 x
  sorry

end height_of_intersection_l49_49340


namespace greatest_possible_digits_in_product_l49_49507

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49507


namespace additional_days_use_l49_49755

variable (m a : ℝ)

theorem additional_days_use (hm : m > 0) (ha : a > 1) : 
  (m / (a - 1) - m / a) = m / (a * (a - 1)) :=
sorry

end additional_days_use_l49_49755


namespace greatest_number_of_digits_in_product_l49_49695

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49695


namespace right_triangle_perimeter_l49_49067

theorem right_triangle_perimeter
  (a b : ℝ)
  (h_area : 0.5 * 30 * b = 150)
  (h_leg : a = 30) :
  a + b + Real.sqrt (a^2 + b^2) = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l49_49067


namespace find_x_intervals_l49_49872

theorem find_x_intervals (x : ℝ) :
  (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ≤ 2 ↔
  ((-∞ : ℝ), -1/3) ∪ (-(1/3 : ℝ), 0) ∪ (0, 1) ∪ (1, (2 : ℝ)] :=
sorry

end find_x_intervals_l49_49872


namespace BC_half_AD_l49_49995

-- Definitions for the conditions given in the problem
structure Quadrilateral (A B C D : Type) :=
  (BC_parallel_AD : ∀ BC AD, BC ∥ AD)
  (E_on_AD : ∀ E, E ∈ segment AD)
  (perimeters_equal : ∀ A B C E, (ABE.perimeter = BCE.perimeter) ∧ (BCE.perimeter = CDE.perimeter))

-- Statement of the problem
theorem BC_half_AD {A B C D : Type} 
  (q : Quadrilateral A B C D) : 
  ∀ BC AD E, BC = AD / 2 :=
by
  sorry

end BC_half_AD_l49_49995


namespace greatest_possible_digits_in_product_l49_49510

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49510


namespace common_point_graphs_l49_49939

theorem common_point_graphs 
  (a b c d : ℝ)
  (h1 : ∃ x : ℝ, 2*a + (1 / (x - b)) = 2*c + (1 / (x - d))) :
  ∃ x : ℝ, 2*b + (1 / (x - a)) = 2*d + (1 / (x - c)) :=
by
  sorry

end common_point_graphs_l49_49939


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49638

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49638


namespace symmetry_graph_l49_49929

theorem symmetry_graph (θ:ℝ) (hθ: θ > 0):
  (∀ k: ℤ, 2 * (3 * Real.pi / 4) + (Real.pi / 3) - 2 * θ = k * Real.pi + Real.pi / 2) 
  → θ = Real.pi / 6 :=
by 
  sorry

end symmetry_graph_l49_49929


namespace number_of_four_digit_numbers_l49_49271

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49271


namespace number_of_four_digit_numbers_l49_49297

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49297


namespace determine_mass_CaSO4_formed_l49_49102

-- Definitions for molar masses in g/mol
def molar_mass_CaCO3 : Float := 100.09
def molar_mass_H2SO4 : Float := 98.09
def molar_mass_CaSO4 : Float := 136.15

-- Given amounts in grams
def mass_CaCO3 : Float := 50
def mass_H2SO4 : Float := 100

-- Moles of reactants
def moles_CaCO3 : Float := mass_CaCO3 / molar_mass_CaCO3
def moles_H2SO4 : Float := mass_H2SO4 / molar_mass_H2SO4

-- Balanced reaction information
def balanced_reaction (moles_CaCO3 moles_H2SO4 : Float) : Bool :=
  moles_CaCO3 = moles_H2SO4

-- Limiting reactant is CaCO3, compute the mass of CaSO4 formed
def mass_CaSO4_formed : Float := moles_CaCO3 * molar_mass_CaSO4

-- Theorem to determine the mass of CaSO4 formed
theorem determine_mass_CaSO4_formed : mass_CaSO4_formed = 68.02 := by
  sorry

end determine_mass_CaSO4_formed_l49_49102


namespace greatest_number_of_digits_in_product_l49_49618

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49618


namespace min_diff_proof_l49_49480

noncomputable def triangleMinDiff : ℕ :=
  let PQ := 666
  let QR := 667
  let PR := 2010 - PQ - QR
  if (PQ < QR ∧ QR < PR ∧ PQ + QR > PR ∧ PQ + PR > QR ∧ PR + QR > PQ) then QR - PQ else 0

theorem min_diff_proof :
  ∃ PQ QR PR : ℕ, PQ + QR + PR = 2010 ∧ PQ < QR ∧ QR < PR ∧ (PQ + QR > PR) ∧ (PQ + PR > QR) ∧ (PR + QR > PQ) ∧ (QR - PQ = triangleMinDiff) := sorry

end min_diff_proof_l49_49480


namespace school_athletic_team_profit_l49_49073

-- Define the conditions
def num_bars : ℕ := 1200
def cost_per_three_bars : ℝ := 1.50
def sell_rate : ℝ := 3 / 4
def discount_per_100_bars : ℝ := 2
def cost_per_bar : ℝ := cost_per_three_bars / 3
def total_cost : ℝ := num_bars * cost_per_bar
def revenue_per_bar : ℝ := 3 / 4
def total_revenue_without_discount : ℝ := num_bars * revenue_per_bar
def num_discounts : ℝ := num_bars / 100
def total_discount : ℝ := num_discounts * discount_per_100_bars
def adjusted_total_revenue : ℝ := total_revenue_without_discount - total_discount
def profit : ℝ := adjusted_total_revenue - total_cost

-- Statement to be proved
theorem school_athletic_team_profit : profit = 276 := by
  sorry

end school_athletic_team_profit_l49_49073


namespace selection_problem_l49_49138

def group_size : ℕ := 10
def selected_group_size : ℕ := 3
def total_ways_without_C := Nat.choose 9 3
def ways_without_A_B_C := Nat.choose 7 3
def correct_answer := total_ways_without_C - ways_without_A_B_C

theorem selection_problem:
  (∃ (A B C : ℕ), total_ways_without_C - ways_without_A_B_C = 49) :=
by
  sorry

end selection_problem_l49_49138


namespace quadratic_imaginary_roots_l49_49339

theorem quadratic_imaginary_roots (a b c x1 x2 : ℝ) (h_eq : a * x1^2 + b * x1 + c = 0) (h_conj : x2 = complex.conj x1) (h_real : (x1^3).re = (x1^3)) : ac ÷ b^2 = 1 := 
sorry

end quadratic_imaginary_roots_l49_49339


namespace greatest_digits_product_l49_49587

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49587


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49705

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49705


namespace repeating_decimals_to_fraction_l49_49865

theorem repeating_decimals_to_fraction :
  (0.\overline{6} - 0.\overline{4} + 0.\overline{8}) = (10 / 9) :=
by
  -- Define the repeating decimals as fractions
  let x := 6 / 9
  let y := 4 / 9
  let z := 8 / 9
  -- Express the statement to be proved
  have h1 : 0.\overline{6} = x := sorry
  have h2 : 0.\overline{4} = y := sorry
  have h3 : 0.\overline{8} = z := sorry
  calc
    (0.\overline{6} - 0.\overline{4} + 0.\overline{8}) = (x - y + z) : by rw [h1, h2, h3]
    ... = (6 / 9 - 4 / 9 + 8 / 9) : by congr -- Use congruence for fractions
    ... = (10 / 9) : by ring

end repeating_decimals_to_fraction_l49_49865


namespace length_of_PS_l49_49361

theorem length_of_PS (P Q R S : Type) [Number (P Q R S)] 
    (right_angle : ∠P = 90) 
    (PR : ℝ) (PQ : ℝ) (QR : ℝ) (PS : ℝ) (SR : ℝ)
    (PR_val : PR = 3) (PQ_val : PQ = 4)
    (QR_val : QR = 5) -- from Pythagorean theorem calculation step in b)
    (PQ_PR_ratio : PQ / PR = 4 / 3) -- from angle bisector theorem in b)
    (angle_bisector : PQ / PR = PS / SR) -- from angle bisector theorem
     : PS = 20 / 7 :=
  sorry

end length_of_PS_l49_49361


namespace actual_distance_between_towns_l49_49063

def map_scale : ℕ := 600000
def distance_on_map : ℕ := 2

theorem actual_distance_between_towns :
  (distance_on_map * map_scale) / 100 / 1000 = 12 :=
by
  sorry

end actual_distance_between_towns_l49_49063


namespace triangle_ratio_AN_MB_l49_49042

/-- Let triangle ABC be such that ∠A = 60°. 
 Points M and N lie on sides AB and AC respectively. 
 The circumcenter of triangle ABC bisects segment MN. 
 Prove that the ratio AN:MB = 2:1. -/
theorem triangle_ratio_AN_MB (A B C M N : Point) (O : Point)
  (h_angle_A : ∠BAC = 60) 
  (hM : M ∈ Line A B) (hN : N ∈ Line A C) 
  (hOI : Is_circumcenter O A B C) (hOB : O ∈ midpoint M N) :
  (AN / MB) = 2 :=
sorry

end triangle_ratio_AN_MB_l49_49042


namespace curve_C_cartesian_eq_line_l_rectangular_eq_value_PA_PB_l49_49358

noncomputable def cartesian_eq_curve_C (x y α : ℝ) : Prop :=
  x = 2 + 3 * Real.cos α ∧ y = 3 * Real.sin α 

theorem curve_C_cartesian_eq (x y α : ℝ) :
  (2 + 3 * Real.cos α = x) → 
  (3 * Real.sin α = y) →
  (x - 2)^2 + y^2 = 9 :=
by
  sorry

noncomputable def rectangular_eq_line_l (ρ θ x y : ℝ) : Prop :=
  2 * x - y - 1 = 0 ∧ ρ * (2 * Real.cos θ - Real.sin θ) = 1 

theorem line_l_rectangular_eq (ρ θ x y : ℝ) :
  (ρ * (2 * Real.cos θ - Real.sin θ) = 1) →
  (2 * x - y - 1 = 0) :=
by
  sorry

noncomputable def parametric_eq_line_l (t : ℝ) : ℝ × ℝ :=
  (ℝ.sqrt 5 / 5 * t, -1 + 2 * ℝ.sqrt 5 / 5 * t)

theorem value_PA_PB (t1 t2 : ℝ) :
  (t1 + t2 = 8 * ℝ.sqrt 5 / 5) →
  (t1 * t2 = -4) →
  (abs t1 + abs t2) / abs (t1 * t2) = 3 * ℝ.sqrt 5 / 5 :=
by
  sorry

end curve_C_cartesian_eq_line_l_rectangular_eq_value_PA_PB_l49_49358


namespace max_digits_in_product_l49_49605

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49605


namespace total_mice_after_stress_l49_49791

theorem total_mice_after_stress (initial_mice : ℕ) (pups_per_mouse_first : ℕ) (pups_per_mouse_second : ℕ) (pups_eaten_per_mouse : ℕ) :
    initial_mice = 8 →
    pups_per_mouse_first = 6 →
    pups_per_mouse_second = 6 →
    pups_eaten_per_mouse = 2 →
    let first_generation_pups := initial_mice * pups_per_mouse_first,
        total_mice_first_gen := initial_mice + first_generation_pups,
        surviving_pups_per_mouse := pups_per_mouse_second - pups_eaten_per_mouse,
        second_generation_pups := total_mice_first_gen * surviving_pups_per_mouse,
        total_mice_second_gen := total_mice_first_gen + second_generation_pups
    in
    total_mice_second_gen = 280 :=
by
  intros h0 h1 h2 h3
  sorry

end total_mice_after_stress_l49_49791


namespace four_digit_numbers_count_l49_49213

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49213


namespace line_passes_through_fixed_point_min_value_AC_range_MA_MC_dot_product_l49_49147

noncomputable def Circle := {x : ℝ × ℝ // (x.1 - 4)^2 + (x.2 - 5)^2 = 12}
noncomputable def Line (m : ℝ) := {x : ℝ × ℝ // m * x.1 - x.2 - 2 * m + 3 = 0}

theorem line_passes_through_fixed_point (m : ℝ) : (2, 3) ∈ Line m :=
sorry

theorem min_value_AC (A C : ℝ × ℝ) (hA : A ∈ Circle) (hC : C ∈ Circle) (h : ∃ m, A ∈ Line m ∧ C ∈ Line m) : 
  ∃ l, l = dist A C ∧ l = 4 :=
sorry

theorem range_MA_MC_dot_product (A C : ℝ × ℝ) (hA : A ∈ Circle) (hC : C ∈ Circle) (h : ∃ m, A ∈ Line m ∧ C ∈ Line m) :
  ∃ r, r = ((A.1 - 4) * (C.1 - 4) + (A.2 - 5) * (C.2 - 5)) ∧ -12 ≤ r ∧ r ≤ 4 :=
sorry

end line_passes_through_fixed_point_min_value_AC_range_MA_MC_dot_product_l49_49147


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49572

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49572


namespace count_four_digit_numbers_l49_49290

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49290


namespace exists_infinite_solutions_l49_49857

noncomputable def infinite_solutions_exist (m : ℕ) : Prop := 
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧  (1 / a + 1 / b + 1 / c + 1 / (a * b * c) = m / (a + b + c))

theorem exists_infinite_solutions : infinite_solutions_exist 12 :=
  sorry

end exists_infinite_solutions_l49_49857


namespace min_colors_required_l49_49723

noncomputable def color_scheme (a : ℝ) : ℕ := 
  let f := int.floor (Real.logb 2 a) 
  in f % 5

theorem min_colors_required : 
  (∀ (a b : ℝ), (a > 0 ∧ b > 0) → (a / b = 4 ∨ a / b = 8) → color_scheme a ≠ color_scheme b) → 
  ∃ (n : ℕ), n = 3 := 
by
  sorry

end min_colors_required_l49_49723


namespace exists_moment_distance_condition_l49_49466

theorem exists_moment_distance_condition 
  (O : Point) (M A : Fin 50 → Point) 
  (d_center_to_end : ∀ i, O.distance (A i) ≥ 0) 
  (d_center_to_watch : ∀ i, O.distance (M i) ≥ 0) :
  (∃ t : ℕ, ∑ i in Finset.range 50, O.distance (A i) > ∑ i in Finset.range 50, O.distance (M i)) := 
by 
  sorry

end exists_moment_distance_condition_l49_49466


namespace fill_bathtub_time_l49_49488

noncomputable def water_flow_rate : ℕ := 15 / 3

theorem fill_bathtub_time (capacity : ℕ) (rate : ℕ) : ℕ :=
  capacity / rate

example : fill_bathtub_time 140 water_flow_rate = 28 :=
by 
  have rate := water_flow_rate
  have capacity := 140
  calc
    fill_bathtub_time capacity rate = capacity / rate : rfl
                                ... = 140 / (15 / 3) : by rw [water_flow_rate]
                                ... = 140 / 5 : by norm_num
                                ... = 28 : by norm_num

end fill_bathtub_time_l49_49488


namespace math_problem_f_g_l49_49160

variable {R : Type} [LinearOrderedField R]
variable (f g : R → R)

def P1 : Prop := ∀ x, f x = f (-x) → g x = g (-x)
def P2 : Prop := (∀ (x : R), x ≠ 0 → x * (f' x) > 0) → ∀ x1 x2, (x1 ≤ x2) → (f x1 + g x1) ≤ (f x2 + g x2)

theorem math_problem_f_g
(h1 : ∀ x1 x2, |f x1 - f x2| ≥ |g x1 - g x2|)
(h2 : ∃ x, ∀ x, f x = f (-x)) -- f is even
(h3 : ∀ x1 x2, (f x1 - f x2 = 0))  -- Given in problem statement
: P1 f g ∧ ¬P2 f g :=
begin
  sorry
end

end math_problem_f_g_l49_49160


namespace four_digit_number_count_l49_49233

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49233


namespace count_four_digit_numbers_l49_49281

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49281


namespace find_x_l49_49881

theorem find_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) : x = 3 / 2 :=
sorry

end find_x_l49_49881


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49568

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49568


namespace length_of_AD_l49_49046

-- Definitions
def is_midpoint {α : Type} [linear_order α] (A D M : α) : Prop :=
  M = (A + D) / 2

def quartisect {α : Type} [linear_order α] (A D B C N : α) : Prop :=
  (A < B) ∧ (B < C) ∧ (C < N) ∧ (N < D) ∧ (B - A = C - B) ∧ (C - B = N - C) ∧ (N - C = D - N)

-- The problem statement
theorem length_of_AD {α : Type} [linear_order α] (A D B C N M : α) (MC : α) (h_mid: is_midpoint A D M)
  (h_quart: quartisect A D B C N) (h_MC: MC = C - M) (h_MC_val: MC = 12) :  D - A = 32 := by
  sorry

end length_of_AD_l49_49046


namespace complex_division_example_l49_49168

theorem complex_division_example (z : ℂ) 
  (h : z = (sqrt 3 + complex.i) / (1 - sqrt 3 * complex.i)) :
  z = complex.i :=
by
  sorry

end complex_division_example_l49_49168


namespace definite_integral_example_l49_49859

noncomputable def f (x : ℝ) : ℝ := 1 / x + x

theorem definite_integral_example : ∫ x in 2..4, f x = Real.log 2 + 6 := 
by 
  sorry

end definite_integral_example_l49_49859


namespace greatest_product_digit_count_l49_49676

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49676


namespace speed_of_stream_l49_49465

def speed_of_boat := 14
def distance := 4864
def total_time := 700

theorem speed_of_stream (x : ℝ) :
  4864 / (speed_of_boat - x) + 4864 / (speed_of_boat + x) = total_time ↔ x = 1.2 :=
begin
  sorry
end

end speed_of_stream_l49_49465


namespace dave_walk_probability_l49_49847

/-- Prove that with 15 gates arranged 100 feet apart, if a departure gate is 
    initially assigned at random and changed to another gate at random, then
    the probability that the walking distance is 300 feet or less simplifies 
    to the fraction 37/105. Consequently, m + n = 142. --/
theorem dave_walk_probability : 
  ∃ (m n : ℕ), (m + n = 142) ∧ ((74 : ℚ) / 210).num = m ∧ ((74 : ℚ) / 210).denom = n := 
by
  sorry

end dave_walk_probability_l49_49847


namespace correct_answer_d_l49_49848

noncomputable def f : ℝ → ℝ 
| x := if (x % 2) ∈ [1, 3] then 2 - abs (x - 2) else f (x % 2 + 2)

theorem correct_answer_d :
  f (Real.sin 2) < f (Real.cos 2) := 
sorry

end correct_answer_d_l49_49848


namespace sin_alpha_value_l49_49910

theorem sin_alpha_value (α : ℝ) (h1 : (π / 4) < α) (h2 : α < π) (h3 : cos (α - π / 4) = 3 / 5) : 
  sin α = 7 * sqrt 2 / 10 :=
by
  sorry

end sin_alpha_value_l49_49910


namespace train_speed_l49_49729

theorem train_speed (l t: ℝ) (h1: l = 441) (h2: t = 21) : l / t = 21 := by
  sorry

end train_speed_l49_49729


namespace hexagon_side_sum_l49_49757

theorem hexagon_side_sum (area_hexagon : ℝ) (AB BC FA AD DE EF : ℝ) (h1 : area_hexagon = 98)
  (h2 : AB = 10) (h3 : BC = 12) (h4 : FA = 7) (h5 : AD = 5) (h6 : AD ⊥ AB)
  (h7 : DE = EF) (CD_not_parallel_AB : ¬(CD // AB)) : DE + EF = 16 :=
by
  sorry

end hexagon_side_sum_l49_49757


namespace sqrt_operation_l49_49184

def operation (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt_operation (sqrt5 : ℝ) (h : sqrt5 = Real.sqrt 5) : 
  operation sqrt5 sqrt5 = 20 := by
  sorry

end sqrt_operation_l49_49184


namespace wall_height_l49_49751

noncomputable def brick_length := 21 * 0.01  -- Meter
noncomputable def brick_width := 10 * 0.01   -- Meter
noncomputable def brick_height := 8 * 0.01   -- Meter
noncomputable def wall_length := 9           -- Meter
noncomputable def wall_width := 5            -- Meter
noncomputable def number_of_bricks := 4955.357142857142

noncomputable def brick_volume := brick_length * brick_width * brick_height

theorem wall_height : 
  ∃ H : ℝ, H ≈ 0.185 ∧ 
           wall_length * wall_width * H = number_of_bricks * brick_volume := 
by
  sorry

end wall_height_l49_49751


namespace number_of_trucks_filled_l49_49349

noncomputable def service_cost_per_vehicle := 2.20
noncomputable def fuel_cost_per_liter := 0.70
noncomputable def num_minivans := 4
noncomputable def total_cost := 395.4
noncomputable def minivan_tank_size := 65.0
noncomputable def truck_tank_multiplier := 1.2

def truck_tank_size := minivan_ttank_size * (1 + truck_tank_multiplier)
def minivan_cost := (minivan_ttank_size * fuel_cost_per_liter) + service_cost_per_vehicle
def trucks_cost := total_cost - num_minivans * minivan_cost
def truck_cost := (truck_tank_size * fuel_cost_per_liter) + service_cost_per_vehicle
def num_trucks := trucks_cost / truck_cost

theorem number_of_trucks_filled:
  num_trucks = 2 :=
  by
    sorry

end number_of_trucks_filled_l49_49349


namespace exists_perfect_square_of_the_form_l49_49136

theorem exists_perfect_square_of_the_form (k : ℕ) (h : k > 0) : ∃ n : ℕ, n > 0 ∧ ∃ m : ℕ, m * m = n * 2^k - 7 :=
by sorry

end exists_perfect_square_of_the_form_l49_49136


namespace shoveling_driveways_l49_49372

-- Definitions of the conditions
def cost_of_candy_bars := 2 * 0.75
def cost_of_lollipops := 4 * 0.25
def total_cost := cost_of_candy_bars + cost_of_lollipops
def portion_of_earnings := total_cost * 6
def charge_per_driveway := 1.50
def number_of_driveways := portion_of_earnings / charge_per_driveway

-- The theorem to prove Jimmy shoveled 10 driveways
theorem shoveling_driveways :
  number_of_driveways = 10 := 
by
  sorry

end shoveling_driveways_l49_49372


namespace problem_proof_l49_49353

noncomputable def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem problem_proof (a : ℕ → ℝ) (h_arith : is_arithmetic_seq a) (h_condition: a 4 + a 7 = 2) :
  (2^(a 1) * 2^(a 2) * 2^(a 3) * 2^(a 4) * 2^(a 5) * 2^(a 6) * 2^(a 7) * 2^(a 8) * 2^(a 9) * 2^(a 10)) = 1024 :=
sorry

end problem_proof_l49_49353


namespace equal_segments_l49_49080

-- Definitions of the various points
variables {A B C H D E F P Q R S T U : Type}

-- Assume the conditions given in the problem
variables [is_orthocenter H A B C]
variables [is_midpoint D B C] [is_midpoint E C A] [is_midpoint F A B]
variables [circle_centered_at H P Q]
variables [circle_centered_at H R S]
variables [circle_centered_at H T U]

-- State what we need to prove
theorem equal_segments (h₁ : CP = CQ) (h₂ : AR = AS) (h₃ : BT = BU) :
  CP = CQ ∧ AR = AS ∧ BT = BU :=
by {
  sorry
}

end equal_segments_l49_49080


namespace total_mice_after_stress_l49_49792

theorem total_mice_after_stress (initial_mice : ℕ) (pups_per_mouse_first : ℕ) (pups_per_mouse_second : ℕ) (pups_eaten_per_mouse : ℕ) :
    initial_mice = 8 →
    pups_per_mouse_first = 6 →
    pups_per_mouse_second = 6 →
    pups_eaten_per_mouse = 2 →
    let first_generation_pups := initial_mice * pups_per_mouse_first,
        total_mice_first_gen := initial_mice + first_generation_pups,
        surviving_pups_per_mouse := pups_per_mouse_second - pups_eaten_per_mouse,
        second_generation_pups := total_mice_first_gen * surviving_pups_per_mouse,
        total_mice_second_gen := total_mice_first_gen + second_generation_pups
    in
    total_mice_second_gen = 280 :=
by
  intros h0 h1 h2 h3
  sorry

end total_mice_after_stress_l49_49792


namespace greatest_possible_number_of_digits_in_product_l49_49523

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49523


namespace find_years_l49_49485

variable (p m x : ℕ)

def two_years_ago := p - 2 = 2 * (m - 2)
def four_years_ago := p - 4 = 3 * (m - 4)
def ratio_in_x_years (x : ℕ) := (p + x) * 2 = (m + x) * 3

theorem find_years (h1 : two_years_ago p m) (h2 : four_years_ago p m) : ratio_in_x_years p m 2 :=
by
  sorry

end find_years_l49_49485


namespace count_four_digit_numbers_l49_49285

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49285


namespace apples_for_juice_l49_49442

theorem apples_for_juice (total_apples : ℝ) (exported_pct : ℝ) (juice_pct : ℝ) (spoil_pct : ℝ) :
  total_apples = 7 →
  exported_pct = 0.25 →
  juice_pct = 0.60 →
  spoil_pct = 0.05 →
  (total_apples * (1 - exported_pct) * juice_pct * (1 - spoil_pct) = 3.0) :=
by
  intros h1 h2 h3 h4
  have h5 : total_apples * (1 - exported_pct) = 5.25, by linarith,
  have h6 : 5.25 * juice_pct = 3.15, by linarith,
  have h7 : 3.15 * (1 - spoil_pct) = 2.9925, by linarith,
  have h8 : (2.9925 - 3.0) < 0.05, by linarith,
  have h9 : 3.0 ≤ 2.9925, by linarith,
  sorry

end apples_for_juice_l49_49442


namespace value_of_hash_l49_49399

def hash (a b c d : ℝ) : ℝ := b^2 - 4 * a * c * d

theorem value_of_hash : hash 2 3 2 1 = -7 := by
  sorry

end value_of_hash_l49_49399


namespace greatest_product_digit_count_l49_49684

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49684


namespace greatest_possible_digits_in_product_l49_49511

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49511


namespace semicircle_area_percentage_increase_l49_49065

def area_of_large_semicircles (a b : ℝ) : ℝ :=
  let r_large := a / 2 in
  2 * (1 / 2 * Real.pi * r_large^2)

def area_of_small_semicircles (a b : ℝ) : ℝ :=
  let r_small := b / 2 in
  2 (1 / 2 * Real.pi * r_small^2)

theorem semicircle_area_percentage_increase
  (a b : ℝ) (h_a : a = 14) (h_b : b = 8) :
  ((area_of_large_semicircles a b) / (area_of_small_semicircles a b) - 1) * 100 ≈ 206 := 
by
  sorry

end semicircle_area_percentage_increase_l49_49065


namespace calendar_sum_l49_49346

theorem calendar_sum (n : ℕ) : 
    n + (n + 7) + (n + 14) = 3 * n + 21 :=
by sorry

end calendar_sum_l49_49346


namespace product_of_B_eq_36_l49_49936

-- Definitions based on conditions
def A : Set ℝ := {2, 0, 1, 7}
def B : Set ℝ := { x | (x^2 - 2) ∈ A ∧ (x - 2) ∉ A }

-- The main theorem to prove
theorem product_of_B_eq_36 : (∏ x in B, x) = 36 := sorry

end product_of_B_eq_36_l49_49936


namespace no_expression_yields_perfect_square_l49_49325

theorem no_expression_yields_perfect_square :
  ¬ (∃ n : ℕ, (100! * 101! = n ^ 2) ∨ (101! * 102! = n ^ 2) ∨ (102! * 103! = n ^ 2) ∨ 
               (103! * 104! = n ^ 2) ∨ (104! * 105! = n ^ 2)) :=
by
  -- Solution steps are skipped
  sorry

end no_expression_yields_perfect_square_l49_49325


namespace lcm_1332_888_l49_49717

theorem lcm_1332_888 : Nat.lcm 1332 888 = 2664 :=
by
  have h1 : 1332 = 2^2 * 3^2 * 37 := by sorry
  have h2 : 888 = 2^3 * 3 * 37 := by sorry
  have h3 : Nat.prime 2 := by sorry
  have h4 : Nat.prime 3 := by sorry
  have h5 : Nat.prime 37 := by sorry
  sorry

end lcm_1332_888_l49_49717


namespace three_digit_numbers_valid_count_l49_49317

theorem three_digit_numbers_valid_count : 
  let valid_count := λ (a b c : ℕ), (a > 0 ∧ c ≥ 0 ∧ c < 10 ∧ b = 2 * (a - c).abs ∧ 0 ≤ b ∧ b < 10) in
  (Finset.card (Finset.filter (λ (abc : ℕ × ℕ × ℕ), valid_count (abc.1) (abc.2) (abc.3)) 
    (Finset.pi Finset.univ (λ _, Finset.range 10)))) = 73 :=
by
  sorry

end three_digit_numbers_valid_count_l49_49317


namespace number_of_four_digit_numbers_l49_49280

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49280


namespace area_of_right_triangle_30_60_90_l49_49071

noncomputable def right_triangle_area : ℤ :=
sorry

theorem area_of_right_triangle_30_60_90
  (angle_A : real := 30)
  (angle_B : real := 60)
  (angle_C : real := 90)
  (altitude : ℤ := 5) :
  right_triangle_area = 25 :=
sorry

end area_of_right_triangle_30_60_90_l49_49071


namespace find_a2016_l49_49151

variable {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Given conditions
def cond1 : S 1 = 6 := by sorry
def cond2 : S 2 = 4 := by sorry
def cond3 (n : ℕ) : S n > 0 := by sorry
def cond4 (n : ℕ) : S (2 * n - 1) ^ 2 = S (2 * n) * S (2 * n + 2) := by sorry
def cond5 (n : ℕ) : 2 * S (2 * n + 2) = S (2 * n - 1) + S (2 * n + 1) := by sorry

theorem find_a2016 : a 2016 = -1009 := by
  -- Use the provided conditions to prove the statement
  sorry

end find_a2016_l49_49151


namespace number_of_four_digit_numbers_l49_49275

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49275


namespace perm_prime_count_12345_l49_49079

theorem perm_prime_count_12345 : 
  (∀ x : List ℕ, (x ∈ (List.permutations [1, 2, 3, 4, 5])) → 
    (10^4 * x.head! + 10^3 * x.tail.head! + 10^2 * x.tail.tail.head! + 10 * x.tail.tail.tail.head! + x.tail.tail.tail.tail.head!) % 3 = 0)
  → 
  0 = 0 :=
by
  sorry

end perm_prime_count_12345_l49_49079


namespace tangent_line_eqn_l49_49181

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x - 2

theorem tangent_line_eqn (x0 : ℝ) (hx0 : f'(x0) = 3) : 3*x0 - f(x0) - 1 = 0 := sorry

end tangent_line_eqn_l49_49181


namespace average_age_of_5_l49_49444

theorem average_age_of_5 (h1 : 19 * 15 = 285) (h2 : 9 * 16 = 144) (h3 : 15 = 71) :
    (285 - 144 - 71) / 5 = 14 :=
sorry

end average_age_of_5_l49_49444


namespace necessary_but_not_sufficient_l49_49762

-- Definitions
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c

-- The condition we are given
axiom m : ℝ

-- The quadratic equation specific condition
axiom quadratic_condition : quadratic_eq 1 2 m = 0

-- The necessary but not sufficient condition for real solutions
theorem necessary_but_not_sufficient (h : m < 2) : 
  ∃ x : ℝ, quadratic_eq 1 2 m x = 0 ∧ quadratic_eq 1 2 m x = 0 → m ≤ 1 ∨ m > 1 :=
sorry

end necessary_but_not_sufficient_l49_49762


namespace heather_blocks_l49_49942

theorem heather_blocks (heather_initial_blocks : ℝ) (jose_shared_blocks : ℝ) (heather_final_blocks : ℝ) : 
  heather_initial_blocks = 86.0 → jose_shared_blocks = 41.0 → heather_final_blocks = heather_initial_blocks + jose_shared_blocks → heather_final_blocks = 127.0 :=
by
  intros h_start j_share h_final
  rw [h_start, j_share] at h_final
  exact h_final

#print heather_blocks

end heather_blocks_l49_49942


namespace greatest_possible_number_of_digits_in_product_l49_49529

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49529


namespace weight_difference_l49_49787

-- Definitions based on the conditions
def O : ℝ := 5
def J : ℝ := 2 * O + x
def F : ℝ := (1 / 2) * J - 3

-- Hypotheses: 
axiom total_weight_gain_hyp : O + J + F = 20
axiom jose_twice_orlando_hyp : ∃ x, J = 2 * O + x

-- Goal: Prove the difference between Jose's weight gain and twice Orlando's weight gain is approximately 3.67 pounds
theorem weight_difference : ∃ (epsilon : ℝ), abs (J - 2 * O - 3.67) < epsilon :=
by
  sorry

end weight_difference_l49_49787


namespace right_triangle_perimeter_l49_49068

theorem right_triangle_perimeter
  (a b : ℝ)
  (h_area : 0.5 * 30 * b = 150)
  (h_leg : a = 30) :
  a + b + Real.sqrt (a^2 + b^2) = 40 + 10 * Real.sqrt 10 :=
by
  sorry

end right_triangle_perimeter_l49_49068


namespace smallest_result_l49_49721

open Finset

theorem smallest_result :
  ∀ (a b c : ℕ), {a, b, c} ⊆ {6, 8, 10, 12, 14, 16} → 
  a ≠ b → b ≠ c → a ≠ c →
  min ((a + b) * c - 10) ((a + c) * b - 10) = 98 :=
by
  intros a b c Hsub Hneq1 Hneq2 Hneq3
  -- Proof omitted
  sorry

end smallest_result_l49_49721


namespace angle_and_length_in_circle_tangent_problem_l49_49081

theorem angle_and_length_in_circle_tangent_problem
    (circle : Type)
    (O : circle)
    (PA : ℝ)
    (A B C D E P : circle)
    (hPA_Tangent : PA = 2 * Real.sqrt 3)
    (hAngleAPB : ∠ APB = 30)
    (hMidpoint : D = midpoint O C)
    (hIntersectionAD : segment AD = segment AE ∧ E ∈ circle) :
  ∠ AEC = 60 ∧ length AE = 10 * Real.sqrt 7 / 7 := by
  sorry

end angle_and_length_in_circle_tangent_problem_l49_49081


namespace four_digit_numbers_count_l49_49231

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49231


namespace calculate_drift_l49_49747

def width_of_river : ℕ := 400
def speed_of_boat : ℕ := 10
def time_to_cross : ℕ := 50
def actual_distance_traveled := speed_of_boat * time_to_cross

theorem calculate_drift : actual_distance_traveled - width_of_river = 100 :=
by
  -- width_of_river = 400
  -- speed_of_boat = 10
  -- time_to_cross = 50
  -- actual_distance_traveled = 10 * 50 = 500
  -- expected drift = 500 - 400 = 100
  sorry

end calculate_drift_l49_49747


namespace four_digit_number_count_l49_49235

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49235


namespace angle_A_range_l49_49385

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def strictly_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y : ℝ, x < y ∧ x ∈ I ∧ y ∈ I → f x < f y

theorem angle_A_range (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_strict_inc : strictly_increasing f {x | 0 < x})
  (h_f_half : f (1 / 2) = 0)
  (A : ℝ)
  (h_cos_A : f (Real.cos A) < 0) :
  (π / 3 < A ∧ A < π / 2) ∨ (2 * π / 3 < A ∧ A < π) :=
by
  sorry

end angle_A_range_l49_49385


namespace min_max_x_l49_49002

def is_valid_x (b c : ℝ) (x : ℝ) : Prop :=
  5.025 ≤ b ∧ b ≤ 5.035 ∧ 1.745 ≤ c ∧ c ≤ 1.755 ∧ x = (b * c + 12) / (8 - 3 * c)

theorem min_max_x : 
  ∀ (b c : ℝ), 5.025 ≤ b ∧ b ≤ 5.035 ∧ 1.745 ≤ c ∧ c ≤ 1.755 → 7.512 ≤ (b * c + 12) / (8 - 3 * c) ∧ (b * c + 12) / (8 - 3 * c) ≤ 7.618 :=
begin
  intros b c h,
  sorry
end

end min_max_x_l49_49002


namespace four_digit_numbers_count_l49_49256

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49256


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49636

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49636


namespace brianna_remaining_money_l49_49797

variable (m c n : ℕ)

theorem brianna_remaining_money (h : (1 / 5 : ℝ) * m = (1 / 3 : ℝ) * n * c) : (m - n * c) / m = 2 / 5 :=
by
  have hnc : n * c = (3 / 5 : ℝ) * m := by
    rw ← mul_assoc
    rw ← (div_eq_mul_one_div _ _).symm
    rw h
    ring

  have h1 : (m - n * c) = m - (3 / 5 : ℝ) * m := by
    rw hnc

  have h2 : 1 = 5 / 5 := by norm_num

  have h3 : (5 / 5) * m = m := by rw [h2, mul_one]

  have h4 : (m - (3 / 5) * m) = (2 / 5) * m := by
    rw [← sub_mul, h3]
    norm_num

  rw div_eq_mul_inv
  rw ← h4
  norm_num

  sorry

end brianna_remaining_money_l49_49797


namespace Mitzel_has_26_left_l49_49409

variable (A : ℝ) (spent : ℝ)

-- Given conditions
def spent_percent := 0.35
def spent_amount := 14
def total_allowance := spent_amount / spent_percent
#check total_allowance -- to verify the computation

-- The final amount left after spending
def amount_left := total_allowance - spent_amount

theorem Mitzel_has_26_left : amount_left = 26 := by
  -- This is where the proof would go
  sorry

end Mitzel_has_26_left_l49_49409


namespace max_digits_in_product_l49_49493

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49493


namespace greatest_number_of_digits_in_product_l49_49614

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49614


namespace smallest_x_for_perfect_cube_l49_49128

theorem smallest_x_for_perfect_cube (x : ℕ) (M : ℤ) (hx : x > 0) (hM : ∃ M, 1680 * x = M^3) : x = 44100 :=
sorry

end smallest_x_for_perfect_cube_l49_49128


namespace sum_of_digits_X_eq_201_l49_49146

noncomputable def sum_of_digits (X : ℕ) (digits : ℕ → ℕ) : Prop :=
  (∀ k, digits k < 10) ∧ 
  digits 1 ≠ 0 ∧ 
  (∀ k, digits k > 0 → (∀ j, j ≠ k → (digits k = digits (30 - j))))

theorem sum_of_digits_X_eq_201 (X : ℕ) (digits : ℕ → ℕ) : 
  sum_of_digits X digits →
  (finset.range 29).sum digits = 201 :=
by
  sorry

end sum_of_digits_X_eq_201_l49_49146


namespace greatest_number_of_digits_in_product_l49_49617

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49617


namespace greatest_product_digits_l49_49659

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49659


namespace distance_from_center_to_face_ABC_l49_49903

open Real

/-- The distance from the center of the sphere to the face ABC
of a regular tetrahedron P-ABC with PA, PB, and PC mutually perpendicular,
and with points P, A, B, C lying on the surface of a sphere with surface area 12π,
is √3 / 3. -/
theorem distance_from_center_to_face_ABC (P A B C : ℝ) (r : ℝ) 
  (h1 : P ≠ A) (h2 : P ≠ B) (h3 : P ≠ C)
  (h4 : r = (sqrt 3))
  (h5 : dist P A = 2) (h6 : dist P B = 2) (h7 : dist P C = 2)
  (h8 : PA * PA + PB * PB + PC * PC = (2 ^ 2))
  (h9 : is_regular_tetrahedron P A B C)
  : dist (center_of_sphere P A B C) (face_ABC A B C) = (sqrt 3) / 3 :=
sorry

end distance_from_center_to_face_ABC_l49_49903


namespace four_digit_number_count_l49_49234

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49234


namespace num_valid_complex_numbers_l49_49875

open Complex

-- Definitions and theorem
noncomputable def isValidZ (z : ℂ) : Prop :=
  abs z = 1 ∧ abs (z / (conj z) + (conj z) / z) = 2

theorem num_valid_complex_numbers : 
  {z : ℂ | isValidZ z}.toFinset.card = 4 := by
  sorry

end num_valid_complex_numbers_l49_49875


namespace max_digits_of_product_l49_49549

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49549


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49576

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49576


namespace sum_of_roots_eq_n_squared_l49_49174

open Set Real

noncomputable def f : ℝ → ℝ := λ x, if x < 2 then 2 - x else 2 - (x % 2)

theorem sum_of_roots_eq_n_squared (n : ℕ) (hn : 0 < n) :
  (∑ i in finset.range n, ((2 * n) - (2 * i) / (n + 1))) = n * n :=
sorry

end sum_of_roots_eq_n_squared_l49_49174


namespace count_four_digit_numbers_l49_49291

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49291


namespace bank_balance_after_2_years_l49_49434

noncomputable def compound_interest (P₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P₀ * (1 + r)^n

theorem bank_balance_after_2_years :
  compound_interest 100 0.10 2 = 121 := 
  by
  sorry

end bank_balance_after_2_years_l49_49434


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49701

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49701


namespace circle_equation_polar_coordinates_l49_49994

-- Define the given conditions in a) as Lean 4 definitions
def polar_center : ℝ × ℝ := (1, Real.pi / 4)
def radius : ℝ := 1

-- State the proof problem
theorem circle_equation_polar_coordinates :
  ∃ (rho θ : ℝ), (rho = 2 * Real.cos (θ - Real.pi / 4)) :=
sorry

end circle_equation_polar_coordinates_l49_49994


namespace product_of_factors_l49_49801

theorem product_of_factors : 
  2 * ( ∏ n in Finset.range (12 - 1) + 1, (1 - (1 / (n + 2))) ) = (1 / 6) :=
by 
  sorry

end product_of_factors_l49_49801


namespace max_digits_of_product_l49_49541

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49541


namespace sum_of_valid_a_l49_49838

theorem sum_of_valid_a :
  (∑ a in Finset.filter (λ a, ∃ b c, (a + 23 * b + 15 * c - 2) % 26 = 0 ∧ (2 * a + 5 * b + 14 * c - 8) % 26 = 0) (Finset.range 27), a) = 31 :=
sorry

end sum_of_valid_a_l49_49838


namespace ram_initial_deposit_l49_49486

-- Definitions based on the problem conditions
def initial_deposit := 500
def first_year_interest := 100
def total_first_year := 600
def interest_rate := 0.10
def total_second_year_balance := 660
def balance_increase_percentage := 0.32

-- Theorem stating that given the conditions, the initial deposit must be $500
theorem ram_initial_deposit : 
    ∃ P : ℝ, 
    (P + first_year_interest = total_first_year) ∧ 
    (total_first_year * (1 + interest_rate) = total_second_year_balance) ∧ 
    (total_second_year_balance = initial_deposit * (1 + balance_increase_percentage))
    :=
begin
    use initial_deposit,
    split,
    { 
      -- P + 100 = 600
      exact rfl, 
    },
    split,
    { 
      -- 600 * 1.10 = 660
      exact rfl, 
    },
    { 
      -- 660 = 500 * 1.32
      exact rfl, 
    }
end

end ram_initial_deposit_l49_49486


namespace arctan_sum_pi_div_two_l49_49812

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l49_49812


namespace max_digits_in_product_l49_49501

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49501


namespace arctan_sum_pi_div_two_l49_49811

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l49_49811


namespace determine_a_l49_49952

theorem determine_a (a b x : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 13 * x^3) (h3 : a - b = 2 * x) :
  a = x + (sqrt 66 * x / 6) ∨ a = x - (sqrt 66 * x / 6) :=
  sorry

end determine_a_l49_49952


namespace boys_in_class_l49_49463

theorem boys_in_class (r : ℕ) (g b : ℕ) (h1 : g/b = 4/3) (h2 : g + b = 35) : b = 15 :=
  sorry

end boys_in_class_l49_49463


namespace range_of_p_l49_49397

def A := {x : ℝ | x^2 - x - 2 > 0}
def B := {x : ℝ | (3 / x) - 1 ≥ 0}
def intersection := {x : ℝ | x ∈ A ∧ x ∈ B}
def C (p : ℝ) := {x : ℝ | 2 * x + p ≤ 0}

theorem range_of_p (p : ℝ) : (∀ x : ℝ, x ∈ intersection → x ∈ C p) → p < -6 := by
  sorry

end range_of_p_l49_49397


namespace greatest_digits_product_l49_49589

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49589


namespace greatest_product_digits_l49_49559

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49559


namespace count_four_digit_numbers_l49_49282

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49282


namespace ratio_of_voters_l49_49983

theorem ratio_of_voters (V_X V_Y : ℝ) 
  (h1 : 0.62 * V_X + 0.38 * V_Y = 0.54 * (V_X + V_Y)) : V_X / V_Y = 2 :=
by
  sorry

end ratio_of_voters_l49_49983


namespace number_of_four_digit_numbers_l49_49295

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49295


namespace arithmetic_sequence_sum_l49_49839

theorem arithmetic_sequence_sum :
  (\sum i in Finset.range 10, (2 + 2 * i) / 7) = 110 / 7 :=
by
  sorry

end arithmetic_sequence_sum_l49_49839


namespace greatest_distance_of_vertices_l49_49074

theorem greatest_distance_of_vertices (p1 p2 : ℝ) (h1 : p1 = 36) (h2 : p2 = 64) :
  let s1 := p1 / 4 in
  let s2 := p2 / 4 in
  let d_inner := (s1 * Real.sqrt 2) / 2 in
  let d_outer := (s2 * Real.sqrt 2) / 2 in
  d_inner + d_outer = 17 * Real.sqrt 2 / 2 := 
by
  sorry

end greatest_distance_of_vertices_l49_49074


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49713

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49713


namespace correct_propositions_l49_49895

-- Defining planes and lines
variables {Plane : Type} {Line : Type}

-- Defining relationships
variables (α β : Plane) (m n : Line)

-- Propositions
def Prop2 : Prop :=
  (m ⊥ α) → (n ∥ α) → (m ⊥ n)

def Prop5 : Prop :=
  (m ∥ n) → (α ∥ β) →
  ∀ θ1 θ2 (m_on_α : m.on_plane θ1) (n_on_β : n.on_plane θ2),
    angle θ1 α = angle θ2 β

-- Final theorem combining the propositions
theorem correct_propositions (h1 : Prop2 α m n) (h2 : Prop5 α β m n) :
  Prop2 α m n ∧ Prop5 α β m n :=
by sorry

end correct_propositions_l49_49895


namespace max_digits_of_product_l49_49545

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49545


namespace unique_ordered_pairs_satisfying_equation_l49_49851

theorem unique_ordered_pairs_satisfying_equation :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^6 * y^6 - 19 * x^3 * y^3 + 18 = 0 ↔ (x, y) = (1, 1) ∧
  (∀ x y : ℕ, 0 < x ∧ 0 < y ∧ x^6 * y^6 - 19 * x^3 * y^3 + 18 = 0 → (x, y) = (1, 1)) :=
by
  sorry

end unique_ordered_pairs_satisfying_equation_l49_49851


namespace distance_point_to_line_l49_49448

theorem distance_point_to_line :
  let (x0, y0) := (0, 5)
  let (A, B, C) := (-2, 1, 0)
  let distance := (| A * x0 + B * y0 + C |) / (Real.sqrt (A^2 + B^2))
  in distance = Real.sqrt 5 :=
by {
  let x0 := 0
  let y0 := 5
  let A := -2
  let B := 1
  let C := 0
  let distance := (abs (A * x0 + B * y0 + C)) / (Real.sqrt (A^2 + B^2))
  have h : distance = Real.sqrt 5,
  sorry
}

end distance_point_to_line_l49_49448


namespace advertising_quota_met_and_total_commercial_time_l49_49979

theorem advertising_quota_met_and_total_commercial_time :
  let p1_time := 0.20 * 30
      p2_time := 0.25 * 30
      p3_time := 0.30 * 30
      p4_time := 0.35 * 30
      p5_time := 0.40 * 30
      p6_time := 0.45 * 30
      total_commercial_time := p1_time + p2_time + p3_time + p4_time + p5_time + p6_time
      quota := 50 in
  total_commercial_time = 58.5 ∧ total_commercial_time ≥ quota :=
by
  let p1_time := 0.20 * 30
  let p2_time := 0.25 * 30
  let p3_time := 0.30 * 30
  let p4_time := 0.35 * 30
  let p5_time := 0.40 * 30
  let p6_time := 0.45 * 30
  let total_commercial_time := p1_time + p2_time + p3_time + p4_time + p5_time + p6_time
  let quota := 50
  sorry

end advertising_quota_met_and_total_commercial_time_l49_49979


namespace smallest_a1_l49_49395

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
∀ n ≥ 1, a (n + 1) = 7 * a n + 2 * (n + 1)

theorem smallest_a1 {a : ℕ → ℝ} (h : sequence a) : ∃ a1, a1 = 0 ∧ ∀ n, a n > 0 :=
begin
    use 0,
    split,
    { refl },
    { intro n,
      induction n with k hk,
      { simp [h 1 (by norm_num)], },
      { exact add_pos_of_pos_of_nonneg (add_pos (mul_pos (by norm_num) hk) (mul_nonneg (by norm_num) (by norm_num))) hk, } }
end

end smallest_a1_l49_49395


namespace triangles_from_chords_l49_49411

theorem triangles_from_chords (h1: ∀ x y : Set (Fin 9), (x ∈ y) → True) 
  (h2 : ∀ (a b c d : Fin 9), a ≠ b → b ≠ c → c ≠ d → d ≠ a → True) :
  ∃ t : Nat, t = Nat.choose 9 6 ∧ t = 84 :=
by
  exists Nat.choose 9 6
  constructor
  · rw Nat.choose
  · apply Eq.refl

end triangles_from_chords_l49_49411


namespace geometric_sequence_n_l49_49992

theorem geometric_sequence_n (a1 an q : ℚ) (n : ℕ) (h1 : a1 = 9 / 8) (h2 : an = 1 / 3) (h3 : q = 2 / 3) : n = 4 :=
by
  sorry

end geometric_sequence_n_l49_49992


namespace probability_of_six_on_fourth_roll_l49_49805

noncomputable def biased_die_probability : Rational :=
  let P_Df := 1 / 4
  let P_Db1 := 1 / 4
  let P_Db2 := 1 / 4

  let P_3sixes_Df := (1 / 6)^3
  let P_3sixes_Db1 := (3 / 4)^3
  let P_3sixes_Db2 := (1 / 4)^3

  let P_A := P_Df * P_3sixes_Df + P_Db1 * P_3sixes_Db1 + P_Db2 * P_3sixes_Db2

  let P_Df_A := (P_Df * P_3sixes_Df) / P_A
  let P_Db1_A := (P_Db1 * P_3sixes_Db1) / P_A
  let P_Db2_A := (P_Db2 * P_3sixes_Db2) / P_A

  let P_six_4th_A := P_Df_A * (1 / 6) + P_Db1_A * (3 / 4) + P_Db2_A * (1 / 4)

  P_six_4th_A

theorem probability_of_six_on_fourth_roll :
  biased_die_probability = (3025/3300) := 
sorry

end probability_of_six_on_fourth_roll_l49_49805


namespace max_digits_of_product_l49_49648

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49648


namespace max_digits_in_product_l49_49499

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49499


namespace width_of_channel_at_bottom_l49_49447

theorem width_of_channel_at_bottom
    (top_width : ℝ)
    (area : ℝ)
    (depth : ℝ)
    (b : ℝ)
    (H1 : top_width = 12)
    (H2 : area = 630)
    (H3 : depth = 70)
    (H4 : area = 0.5 * (top_width + b) * depth) :
    b = 6 := 
sorry

end width_of_channel_at_bottom_l49_49447


namespace largest_divisor_of_product_five_faces_l49_49768

theorem largest_divisor_of_product_five_faces (P : ℕ) (hP : ∃ (a b c d e : ℕ),
  {1, 2, 3, 4, 5, 6} = {a, b, c, d, e, P}) : 12 ∣ P :=
sorry

end largest_divisor_of_product_five_faces_l49_49768


namespace greatest_product_digits_l49_49658

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49658


namespace means_properties_l49_49487

noncomputable def arithmetic_mean (a b : ℝ) : ℝ := (a + b) / 2
noncomputable def geometric_mean (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def harmonic_mean (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

theorem means_properties (a b : ℝ) (h : a ≤ b) :
  let m := arithmetic_mean a b,
      g := geometric_mean a b,
      h := harmonic_mean a b
  in h ≤ g ∧ g ≤ m ∧ ((m = g ∧ g = h) ↔ a = b) :=
sorry

end means_properties_l49_49487


namespace greatest_digits_product_l49_49595

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49595


namespace ptolemy_inequality_six_point_inequality_ptolemy_equality_cyclic_six_point_equality_cyclic_l49_49032

-- Define the setup for points on a plane and distances between them
variable {Point : Type} [MetricSpace Point]
variables (A B C D : Point)

variable (dist : Point → Point → ℝ)
local notation "d" x y := dist x y

-- Ptolemy's inequality
theorem ptolemy_inequality (A B C D : Point) : 
  d A B * d C D + d B C * d A D ≥ d A C * d B D := sorry

-- Six-point inequality
variables (A1 A2 A3 A4 A5 A6 : Point)

theorem six_point_inequality :
  d A1 A4 * d A2 A5 * d A3 A6 ≤ d A1 A2 * d A3 A6 * d A4 A5 +
  d A1 A2 * d A3 A4 * d A5 A6 + d A2 A3 * d A1 A4 * d A5 A6 +
  d A2 A3 * d A4 A5 * d A1 A6 + d A3 A4 * d A2 A5 * d A1 A6 := sorry

-- Ptolemy's equality case
theorem ptolemy_equality_cyclic (A B C D : Point) :
  d A B * d C D + d B C * d A D = d A C * d B D ↔ is_convex_cyclic_quadrilateral [A, B, C, D] := sorry

-- Six-point equality case
theorem six_point_equality_cyclic (A1 A2 A3 A4 A5 A6 : Point) :
  d A1 A4 * d A2 A5 * d A3 A6 = 
  d A1 A2 * d A3 A6 * d A4 A5 + d A1 A2 * d A3 A4 * d A5 A6 + 
  d A2 A3 * d A1 A4 * d A5 A6 + d A2 A3 * d A4 A5 * d A1 A6 + 
  d A3 A4 * d A2 A5 * d A1 A6 ↔ is_cyclic_hexagon [A1, A2, A3, A4, A5, A6] := sorry

end ptolemy_inequality_six_point_inequality_ptolemy_equality_cyclic_six_point_equality_cyclic_l49_49032


namespace single_ring_probability_6_single_ring_probability_2n_l49_49043

theorem single_ring_probability_6 : 
  (probability_single_ring 6 = 8 / 15) :=
by sorry

theorem single_ring_probability_2n (n : ℕ) (hn : n > 0) :
   (probability_single_ring (2 * n) = sqrt π / (2 * sqrt n)) :=
by sorry

end single_ring_probability_6_single_ring_probability_2n_l49_49043


namespace projection_coordinates_l49_49164

noncomputable def proj_vector (a b : ℝ × ℝ) (theta : ℝ) : ℝ × ℝ :=
  let a_dot_b := (a.1 * b.1 + a.2 * b.2) in
  let b_magnitude_squared := (b.1^2 + b.2^2) in
  (a_dot_b / b_magnitude_squared) • b

theorem projection_coordinates
  (θ : ℝ) (hθ : θ = 2 * Real.pi / 3)
  (a : ℝ) (h₁ : a = 10)
  (b : ℝ × ℝ) (h₂ : b = (3, 4)) :
  proj_vector (10, 10 * Real.cos θ) b θ = (-3, -4) :=
by
  sorry

end projection_coordinates_l49_49164


namespace k_values_l49_49969

noncomputable def find_k_satisfying_conditions : set ℝ :=
  { k : ℝ | let d := (abs (2 * k + 1)) / (sqrt (1 + k^2)),
                R := sqrt 5,
                chord_length := 4 in
            chord_length = 2 * sqrt (R^2 - d^2) }

theorem k_values (k: ℝ) (h: k ∈ find_k_satisfying_conditions) : k = 0 ∨ k = -(4/3) :=
sorry

end k_values_l49_49969


namespace time_increase_25_percent_l49_49415

theorem time_increase_25_percent (x : ℝ) (hx : x > 0) : 
  let T_Sunday := 64 / x,
      D1 := 32, S1 := 2 * x, T1 := D1 / S1,
      D2 := 32, S2 := x / 2, T2 := D2 / S2,
      T_Monday := T1 + T2,
      Percent_Increase := ((T_Monday - T_Sunday) / T_Sunday) * 100 in
  Percent_Increase = 25 :=
by
  sorry

end time_increase_25_percent_l49_49415


namespace part1_part2_l49_49362

variables {a b c : ℝ} {A B C : ℝ}

-- Given Condition for Part (1)
axiom condition1 : (2 * a + c) * (vector3D.BA).dot (vector3D.BC) = 
                    c * (vector3D.CB).dot (vector3D.AC)

-- Given Condition for Part (2)
axiom side_b : b = real.sqrt 6

-- Necessary Trigonometric Definitions and Theorems
noncomputable def angle_B : ℝ :=
if h : (2 * a + c) * (vector3D.BA).dot (vector3D.BC) = 
         c * (vector3D.CB).dot (vector3D.AC)
then 2 * real.pi / 3
else 0

-- Part (1): Proving B's Value
theorem part1 : angle_B = 2 * real.pi / 3 := 
sorry

-- Part (2): Proving the Range of Area
noncomputable def area_of_triangle (b : ℝ) : set ℝ :=
{S : ℝ | S > 0 ∧ S <= real.sqrt 3 / 2}

theorem part2 : ∃ (S ∈ area_of_triangle b), S = sorry := 
sorry

end part1_part2_l49_49362


namespace smallest_sum_arith_geo_sequence_l49_49461

theorem smallest_sum_arith_geo_sequence :
  ∃ (X Y Z W : ℕ),
    X < Y ∧ Y < Z ∧ Z < W ∧
    (2 * Y = X + Z) ∧
    (Y ^ 2 = Z * X) ∧
    (Z / Y = 7 / 4) ∧
    (X + Y + Z + W = 97) :=
by
  sorry

end smallest_sum_arith_geo_sequence_l49_49461


namespace constant_term_in_expansion_max_terms_in_expansion_l49_49355

noncomputable def binomial_constant_term (n : ℕ) (x : ℝ) : ℝ :=
  (Polynomial.binomial (sqrt x) (1 / (2 * sqrt x)) n).coeff 4

noncomputable def binomial_max_terms (n : ℕ) (x : ℝ) : List (Polynomial ℝ) :=
  let poly := Polynomial.binomial (sqrt x) (1 / (2 * sqrt x)) n
  [poly.coeff (2 + 1) * x^2, poly.coeff (3 + 1) * x^3]

theorem constant_term_in_expansion: 
  binomial_constant_term 8 1 = 35 / 8 :=
by
  sorry

theorem max_terms_in_expansion: 
  binomial_max_terms 8 1 = [7, 7 * x] :=
by
  sorry

end constant_term_in_expansion_max_terms_in_expansion_l49_49355


namespace find_first_offset_l49_49873

variable (d y A x : ℝ)

theorem find_first_offset (h_d : d = 40) (h_y : y = 6) (h_A : A = 300) :
    x = 9 :=
by
  sorry

end find_first_offset_l49_49873


namespace amplitude_ratio_seven_four_l49_49445

noncomputable def richter_magnitude_scale (M A A0 : ℝ) := M = log A - log A0

theorem amplitude_ratio_seven_four:
  (A7 A4 A0 : ℝ) (h7 : richter_magnitude_scale 7 A7 A0) (h4 : richter_magnitude_scale 4 A4 A0) :
  A7 / A4 = 10^3 := 
sorry

end amplitude_ratio_seven_four_l49_49445


namespace prove_func_expr_prove_find_sin_alpha_beta_l49_49173

open Real

def f (x : ℝ) : ℝ := (√3 / 2) * sin (2 * x) - cos (x) ^ 2 - (a / 2)

def func_expr : Prop :=
  ∃ a ∈ ℝ, 
    (∀ x ∈ Icc (-π / 12) (π / 2), 2 * f x = √3 * sin (2 * x) - 2 * cos (x) ^ 2 - a) ∧ 
    (2 ≤ √3 - 0.5) ∧ 
    (∀ x ∈ Icc (-π / 12) (π / 2), f x ∈ [-√3, 2])

def find_sin_alpha_beta : Prop :=
  ∃ (α β : ℝ), 
    α ∈ (0, π / 2) ∧
    β ∈ (0, π / 2) ∧
    f(α / 2 + π / 12) = 10 / 13 ∧
    f(β / 2 + π / 3) = 6 / 5 ∧
    sin (α - β) = -33 / 65

theorem prove_func_expr : func_expr := sorry
theorem prove_find_sin_alpha_beta : find_sin_alpha_beta := sorry

end prove_func_expr_prove_find_sin_alpha_beta_l49_49173


namespace cauchy_schwarz_inequality_cauchy_schwarz_equality_condition_l49_49094

theorem cauchy_schwarz_inequality (a b x y : ℝ) :
  ax + by ≤ Real.sqrt (a^2 + b^2) * Real.sqrt (x^2 + y^2) :=
by
  sorry

theorem cauchy_schwarz_equality_condition (a b x y : ℝ) :
  ax + by = Real.sqrt (a^2 + b^2) * Real.sqrt (x^2 + y^2) ↔ a * y = b * x :=
by
  sorry

end cauchy_schwarz_inequality_cauchy_schwarz_equality_condition_l49_49094


namespace four_digit_numbers_count_l49_49227

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49227


namespace find_min_max_value_of_squares_l49_49394

theorem find_min_max_value_of_squares
  (x1 x2 x3 x4 : ℝ)
  (h : x1 < x2 ∧ x2 < x3 ∧ x3 < x4) :
  let expr := λ (y1 y2 y3 y4 : ℝ), (y1 - y2)^2 + (y2 - y3)^2 + (y3 - y4)^2 + (y4 - y1)^2 in
  ∃ y1 y2 y3 y4 : ℝ, 
    (y1, y2, y3, y4) ∈ [(x1, x2, x3, x4), (x1, x2, x4, x3), (x1, x3, x2, x4)] ∧
    (expr y1 y2 y3 y4 = (x1 - x2)^2 + (x2 - x4)^2 + (x4 - x3)^2 + (x3 - x1)^2 ∨
     expr y1 y2 y3 y4 = (x1 - x3)^2 + (x3 - x2)^2 + (x2 - x4)^2 + (x4 - x1)^2) :=
sorry

end find_min_max_value_of_squares_l49_49394


namespace sum_first_n_terms_l49_49382

-- Define the arithmetic sequence {a_n}
def a_arith_seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ (∃ d, ∀ n, a (n + 1) = a n + d)

-- Define the geometric sequence {b_n}
def b_geom_seq (b : ℕ → ℕ) : Prop :=
  b 1 = 1 ∧ (∃ q, ∀ n, b (n + 1) = b n * q)

-- Define the sequence {c_n}
def c_seq (a b c : ℕ → ℕ) : Prop :=
  ∀ n, c n = a (2 * n - 1) + b (2 * n)

-- Given conditions
variables (a b c : ℕ → ℕ)
variable (d q : ℕ)
axiom a_arith : a_arith_seq a
axiom b_geom : b_geom_seq b
axiom c_s : c_seq a b c
axiom a1 : a 1 = 1
axiom b1 : b 1 = 1
axiom b4 : b 4 = 64
axiom q_eq_2d : q = 2 * d

-- The main theorem to prove
theorem sum_first_n_terms (n : ℕ) :
  (∑ k in Finset.range n, c (k + 1)) = 2 * n^2 - n + (4^(2*n + 1) - 4) / 15 :=
sorry

end sum_first_n_terms_l49_49382


namespace four_digit_numbers_count_l49_49226

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49226


namespace handshake_problem_l49_49467

theorem handshake_problem (R C : ℕ) (hR : R = 3) (hC : C = 5) : 
  let total_people := R * C,
      handshakes_per_person := total_people - 1 - (R - 1)
  in (total_people * handshakes_per_person) / 2 = 90 := 
by
  have : total_people = 15 := by rw [hR, hC]; norm_num,
  have : handshakes_per_person = 12 := by rw [this]; norm_num,
  sorry

end handshake_problem_l49_49467


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49707

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49707


namespace greatest_product_digits_l49_49563

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49563


namespace area_ratio_l49_49877

def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

variable (A B C Y : Type) [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variable {AB BC AC : ℝ}
variable {BY AY : ℝ}
variable {h₁ h₂ : ℝ}

-- Given conditions
def CY_bisects_BCA (BC AC : ℝ) : Prop := BC / AC = 34 / 28

theorem area_ratio (h₁ h₂ : ℝ) (H : BY / AY = 17 / 14) : triangle_area BCAY h₁ / triangle_area ABAY h₂ = 17 / 14 := by
  sorry

end area_ratio_l49_49877


namespace chess_club_girls_l49_49753

theorem chess_club_girls (B G : ℕ) (h1 : B + G = 32) (h2 : (1 / 2 : ℝ) * G + B = 20) : G = 24 :=
by
  -- proof
  sorry

end chess_club_girls_l49_49753


namespace greatest_number_of_digits_in_product_l49_49697

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49697


namespace stamp_collection_l49_49481

theorem stamp_collection (x : ℕ) :
  (5 * x + 3 * (x + 20) = 300) → (x = 30) ∧ (x + 20 = 50) :=
by
  sorry

end stamp_collection_l49_49481


namespace max_digits_of_product_l49_49655

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49655


namespace Euros_Berengere_Needs_l49_49024

noncomputable def eurosNeededByBerengere (USDtoFr, EURtoFr : ℝ) (USD_Emily : ℝ) (price : ℝ) : ℝ :=
  let francs_Emily := USD_Emily * USDtoFr
  let francs_needed := price - francs_Emily
  francs_needed / EURtoFr

theorem Euros_Berengere_Needs
  (cake_cost : ℝ)
  (Emily_USD : ℝ)
  (USD_to_Fr : ℝ)
  (EUR_to_Fr : ℝ) :
  eurosNeededByBerengere USD_to_Fr EUR_to_Fr Emily_USD cake_cost = 5.454545454545454 {
    -- Given conditions
    let cost_of_cake := 15  -- in Swiss francs
    let emily_dollars := 10 -- Emily's dollars

    let US_to_CHF := 0.90
    let EUR_to_CHF := 1.10

    -- Proof goal
    sorry
}

end Euros_Berengere_Needs_l49_49024


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49575

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49575


namespace jimmy_shoveled_10_driveways_l49_49369

theorem jimmy_shoveled_10_driveways :
  ∀ (cost_candy_bar : ℝ) (num_candy_bars : ℕ)
    (cost_lollipop : ℝ) (num_lollipops : ℕ)
    (fraction_spent : ℝ)
    (charge_per_driveway : ℝ),
    cost_candy_bar = 0.75 →
    num_candy_bars = 2 →
    cost_lollipop = 0.25 →
    num_lollipops = 4 →
    fraction_spent = 1/6 →
    charge_per_driveway = 1.5 →
    let total_spent := (num_candy_bars * cost_candy_bar + num_lollipops * cost_lollipop) in
    let total_earned := total_spent / fraction_spent in
    (total_earned / charge_per_driveway) = 10 := sorry

end jimmy_shoveled_10_driveways_l49_49369


namespace four_digit_number_count_l49_49238

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49238


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49835

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49835


namespace thin_film_radius_volume_l49_49061

theorem thin_film_radius_volume :
  ∀ (r : ℝ) (V : ℝ) (t : ℝ), 
    V = 216 → t = 0.1 → π * r^2 * t = V → r = Real.sqrt (2160 / π) :=
by
  sorry

end thin_film_radius_volume_l49_49061


namespace greatest_number_of_digits_in_product_l49_49693

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49693


namespace max_digits_in_product_l49_49608

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49608


namespace cube_volume_surface_area_l49_49016

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s^3 = 8 * x ∧ 6 * s^2 = 2 * x) → x = 0 :=
by
  sorry

end cube_volume_surface_area_l49_49016


namespace andy_distance_to_school_l49_49782

theorem andy_distance_to_school :
  ∃ x : ℕ, (2 * x + 40 = 140) ∧ (x = 50) :=
by
  existsi (50 : ℕ)
  split
  · calc
      2 * 50 + 40 = 100 + 40 : by rfl
               ... = 140 : by rfl
  · rfl
<html>

end andy_distance_to_school_l49_49782


namespace probability_one_not_first_class_given_one_first_class_l49_49441

def total_products : ℕ := 8
def first_class_products : ℕ := 6
def selected_products : ℕ := 2

theorem probability_one_not_first_class_given_one_first_class :
  let A := (1 - (nat.choose 6 2) / (nat.choose 8 2))
  let B := (nat.choose 2 1 * nat.choose 6 1) / (nat.choose 8 2)
  (B / A) = 12 / 13 :=
by
  sorry

end probability_one_not_first_class_given_one_first_class_l49_49441


namespace glass_original_water_l49_49058

theorem glass_original_water 
  (O : ℝ)  -- Ounces of water originally in the glass
  (evap_per_day : ℝ)  -- Ounces of water evaporated per day
  (total_days : ℕ)    -- Total number of days evaporation occurs
  (percent_evaporated : ℝ)  -- Percentage of the original amount that evaporated
  (h1 : evap_per_day = 0.06)  -- 0.06 ounces of water evaporated each day
  (h2 : total_days = 20)  -- Evaporation occurred over a period of 20 days
  (h3 : percent_evaporated = 0.12)  -- 12% of the original amount evaporated during this period
  (h4 : evap_per_day * total_days = 1.2)  -- 0.06 ounces per day for 20 days total gives 1.2 ounces
  (h5 : percent_evaporated * O = evap_per_day * total_days) :  -- 1.2 ounces is 12% of the original amount
  O = 10 :=  -- Prove that the original amount is 10 ounces
sorry

end glass_original_water_l49_49058


namespace greatest_product_digit_count_l49_49674

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49674


namespace count_four_digit_numbers_l49_49312

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49312


namespace function_nonnegative_intervals_l49_49137

noncomputable def function_f (x : ℝ) : ℝ :=
  (x^2 * (1 - 3*x)^2) / ((x^2 - real.sqrt (real.sqrt 10)) * (x^2 + real.sqrt (real.sqrt 10)))

theorem function_nonnegative_intervals :
  {x : ℝ | function_f x ≥ 0} =
    set.Icc (-real.sqrt (real.sqrt 10)) 0 ∪ set.Icc 0 (1/3:ℝ) ∪ set.Icc (1/3:ℝ) (real.sqrt (real.sqrt 10)) :=
begin
  sorry
end

end function_nonnegative_intervals_l49_49137


namespace arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49832

noncomputable def arctan_add (a b : ℝ) : ℝ :=
  Real.arctan a + Real.arctan b

theorem arctan_3_4_add_arctan_4_3_is_pi_div_2 :
  arctan_add (3 / 4) (4 / 3) = Real.pi / 2 :=
sorry

end arctan_3_4_add_arctan_4_3_is_pi_div_2_l49_49832


namespace four_digit_numbers_count_l49_49221

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49221


namespace sin_double_angle_l49_49166

noncomputable def unit_circle_point :=
  (1 / 2, Real.sqrt (1 - (1 / 2) ^ 2))

theorem sin_double_angle 
  (α : Real)
  (h1 : (1 / 2, Real.sqrt (1 - (1 / 2) ^ 2)) ∈ { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 })
  (h2 : α = (Real.arccos (1 / 2)) ∨ α = -(Real.arccos (1 / 2))) :
  Real.sin (π / 2 + 2 * α) = -1 / 2 :=
by
  sorry

end sin_double_angle_l49_49166


namespace circle_problem_l49_49898

def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def line (p1 p2 : point) : ℝ × ℝ × ℝ :=
  let a := p2.2 - p1.2
  let b := p1.1 - p2.1
  let c := p2.1 * p1.2 - p1.1 * p2.2
  (a, b, c)

def circle_eq (center : point) (radius : ℝ) : (ℝ → ℝ → Prop) :=
  λ x y, (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2

theorem circle_problem :
  let A := (2, 4)
  let B := (6, 2)
  let C := midpoint A B
  let D := (C.1 + √10, C.2 + √10)
  let E := (C.1 - √10, C.2 - √10)
  line C D = (2, -1, 5) ∧
  (∃ M1 M2 : point, circle_eq M1 (√10) ∧ M1 ∈ ({A, B, C, D} : set point) ∧
    circle_eq M2 (√10) ∧ M2 ∈ ({A, B, C, E} : set point)) := by
  sorry

end circle_problem_l49_49898


namespace four_digit_numbers_count_l49_49252

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49252


namespace max_digits_of_product_l49_49647

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49647


namespace complex_modulus_eq_one_l49_49911

open Complex

theorem complex_modulus_eq_one (z : ℂ) (hz : (z + 1) * (conj z - 1)).im = 0 : |z| = 1 :=
sorry

end complex_modulus_eq_one_l49_49911


namespace largest_two_digit_n_l49_49387

theorem largest_two_digit_n (x : ℕ) (n : ℕ) (hx : x < 10) (hx_nonzero : 0 < x)
  (hn : n = 12 * x * x) (hn_two_digit : n < 100) : n = 48 :=
by sorry

end largest_two_digit_n_l49_49387


namespace reservoir_shortage_l49_49786

noncomputable def reservoir_information := 
  let current_level := 14 -- million gallons
  let normal_level_due_to_yield := current_level / 2
  let percentage_of_capacity := 0.70
  let evaporation_factor := 0.90
  let total_capacity := current_level / percentage_of_capacity
  let normal_level_after_evaporation := normal_level_due_to_yield * evaporation_factor
  let shortage := total_capacity - normal_level_after_evaporation
  shortage

theorem reservoir_shortage :
  reservoir_information = 13.7 := 
by
  sorry

end reservoir_shortage_l49_49786


namespace interval_has_zero_find_zeros_of_function_l49_49454

noncomputable def f : ℝ → ℝ
| -2 := -4
| 0 := -3
| 1 := 6
| 3 := -5
| 4 := 7
| _ := sorry -- fill in continuous definition appropriately

theorem interval_has_zero (a b : ℝ) (hb : a < b) (hfa : f a < 0 ∨ f a > 0) 
   (hfb : f b < 0 ∨ f b > 0) (hf : f a * f b < 0) : ∃ x : ℝ, a < x ∧ x < b ∧ f x = 0 := 
by
  apply intermediate_value_theorem -- or a similar IVT application
  apply sorry

theorem find_zeros_of_function : 
  (∃ x : ℝ, (0 < x ∧ x < 1) ∧ f x = 0) ∧ 
  (∃ x : ℝ, (1 < x ∧ x < 3) ∧ f x = 0) :=
by
  have h1 : f 0 * f 1 < 0 := by norm_num -- since f(0) = -3 and f(1) = 6
  have h2 : f 1 * f 3 < 0 := by norm_num -- since f(1) = 6 and f(3) = -5
  apply And.intro
  { exact interval_has_zero 0 1 (by norm_num) (by norm_num) (by norm_num) h1 }
  { exact interval_has_zero 1 3 (by norm_num) (by norm_num) (by norm_num) h2 }

end interval_has_zero_find_zeros_of_function_l49_49454


namespace factor_2210_two_digit_l49_49320

theorem factor_2210_two_digit :
  (∃ (a b : ℕ), a * b = 2210 ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99) ∧
  (∃ (c d : ℕ), c * d = 2210 ∧ 10 ≤ c ∧ c ≤ 99 ∧ 10 ≤ d ∧ d ≤ 99) ∧
  (∀ (x y : ℕ), x * y = 2210 ∧ 10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 → 
   ((x = c ∧ y = d) ∨ (x = d ∧ y = c) ∨ (x = a ∧ y = b) ∨ (x = b ∧ y = a))) :=
sorry

end factor_2210_two_digit_l49_49320


namespace operation_star_correct_l49_49843

def op_table (i j : ℕ) : ℕ :=
  if i = 1 then
    if j = 1 then 4 else if j = 2 then 1 else if j = 3 then 2 else if j = 4 then 3 else 0
  else if i = 2 then
    if j = 1 then 1 else if j = 2 then 3 else if j = 3 then 4 else if j = 4 then 2 else 0
  else if i = 3 then
    if j = 1 then 2 else if j = 2 then 4 else if j = 3 then 1 else if j = 4 then 3 else 0
  else if i = 4 then
    if j = 1 then 3 else if j = 2 then 2 else if j = 3 then 3 else if j = 4 then 4 else 0
  else 0

theorem operation_star_correct : op_table (op_table 3 1) (op_table 4 2) = 3 :=
  by sorry

end operation_star_correct_l49_49843


namespace greatest_digits_product_l49_49591

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49591


namespace max_digits_of_product_l49_49654

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49654


namespace initial_population_l49_49038

theorem initial_population (P : ℝ) : 
  (P * 1.2 * 0.8 = 9600) → P = 10000 :=
by
  sorry

end initial_population_l49_49038


namespace max_digits_in_product_l49_49494

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49494


namespace f_evaluation_l49_49183

def f (a b c : ℚ) : ℚ := a^2 + 2 * b * c

theorem f_evaluation :
  f 1 23 76 + f 23 76 1 + f 76 1 23 = 10000 := by
  sorry

end f_evaluation_l49_49183


namespace ratio_initial_to_total_capacity_l49_49743

-- Definitions of the initial problem conditions
def capacity : ℚ := 2  -- in kiloliters
def fill_rate_pipe : ℚ := 1 / 2  -- kiloliters per minute (1 kiloliter every 2 minutes)
def drain_rate_1 : ℚ := 1 / 4  -- kiloliters per minute (1 kiloliter every 4 minutes)
def drain_rate_2 : ℚ := 1 / 6  -- kiloliters per minute (1 kiloliter every 6 minutes)
def total_fill_time : ℚ := 12  -- in minutes

-- Theorem statement
theorem ratio_initial_to_total_capacity : 
  let net_fill_rate := fill_rate_pipe - (drain_rate_1 + drain_rate_2) in
  let total_water_added := net_fill_rate * total_fill_time in
  let initial_water := capacity - total_water_added in
  initial_water / capacity = 1 / 2 :=
by
  sorry

end ratio_initial_to_total_capacity_l49_49743


namespace remaining_tickets_after_all_purchases_l49_49084

-- Define initial conditions
def initial_tickets : ℕ := 25
def beanie_cost : ℕ := 22
def additional_tickets : ℕ := 15
def discount_percentage : ℝ := 0.10

-- Define remaining tickets after buying the beanie
def remaining_tickets_after_beanie (initial_tickets : ℕ) (beanie_cost : ℕ) : ℕ :=
  initial_tickets - beanie_cost

-- Define the cost of the keychain
def keychain_cost (remaining_tickets : ℕ) : ℝ := 
  2 * remaining_tickets

-- Define the discounted price of the keychain
def discounted_keychain_cost (keychain_cost : ℝ) (discount_percentage : ℝ) : ℝ :=
  keychain_cost - (keychain_cost * discount_percentage)

-- Because Dave cannot have a fraction of a ticket, the keychain cost will be rounded up
def rounded_keychain_cost (discounted_keychain_cost : ℝ) : ℕ :=
  ⌈discounted_keychain_cost⌉  -- ceil function

-- Prove the final remaining tickets
theorem remaining_tickets_after_all_purchases :
  let rt_after_beanie := remaining_tickets_after_beanie initial_tickets beanie_cost,
      kc := keychain_cost rt_after_beanie,
      discounted_kc := discounted_keychain_cost kc discount_percentage,
      rounded_kc := rounded_keychain_cost discounted_kc,
      final_tickets := (rt_after_beanie + additional_tickets) - rounded_kc
  in final_tickets = 12 :=
by sorry

end remaining_tickets_after_all_purchases_l49_49084


namespace jim_trees_15th_birthday_l49_49368

theorem jim_trees_15th_birthday : 
  let initial_trees := 2 * 4 in
  let years_planting := 15 - 10 in
  let trees_planted_each_year := 4 in
  let total_new_trees := years_planting * trees_planted_each_year in
  let total_trees_before_doubling := initial_trees + total_new_trees in
  let final_trees := total_trees_before_doubling * 2 in
  final_trees = 56 :=
by
  sorry

end jim_trees_15th_birthday_l49_49368


namespace angle_between_vectors_l49_49185

variables {α : Type*} [inner_product_space α] {a b : α}
variables (ha : ∥a∥ = 2) (hb : ∥b∥ = 1) (hab : inner a b = -1)

theorem angle_between_vectors :
  real.angle_between_vectors a b = 2 * real.pi / 3 := sorry

end angle_between_vectors_l49_49185


namespace allison_greater_probability_l49_49777

-- Definitions and conditions for the problem
def faceRollAllison : Nat := 6
def facesBrian : List Nat := [1, 3, 3, 5, 5, 6]
def facesNoah : List Nat := [4, 4, 4, 4, 5, 5]

-- Function to calculate probability
def probability_less_than (faces : List Nat) (value : Nat) : ℚ :=
  (faces.filter (fun x => x < value)).length / faces.length

-- Main theorem statement
theorem allison_greater_probability :
  probability_less_than facesBrian 6 * probability_less_than facesNoah 6 = 5 / 6 := by
  sorry

end allison_greater_probability_l49_49777


namespace complex_sum_series_l49_49379

theorem complex_sum_series (ω : ℂ) (h1 : ω ^ 7 = 1) (h2 : ω ≠ 1) :
  ω ^ 16 + ω ^ 18 + ω ^ 20 + ω ^ 22 + ω ^ 24 + ω ^ 26 + ω ^ 28 + ω ^ 30 + 
  ω ^ 32 + ω ^ 34 + ω ^ 36 + ω ^ 38 + ω ^ 40 + ω ^ 42 + ω ^ 44 + ω ^ 46 +
  ω ^ 48 + ω ^ 50 + ω ^ 52 + ω ^ 54 = -1 :=
sorry

end complex_sum_series_l49_49379


namespace max_digits_in_product_l49_49598

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49598


namespace max_digits_in_product_l49_49610

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49610


namespace greatest_number_of_digits_in_product_l49_49700

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49700


namespace chord_length_l49_49025

theorem chord_length :
  ∀ (M : ℝ × ℝ), M = (8, 0) → 
  (∀ (B₁ B₂ : ℝ × ℝ), 
     B₁.1 = 6 ∧ B₁.2 = -2 * real.sqrt 3 ∧ B₂.1 = 32 / 3 ∧ B₂.2 = 8 * real.sqrt 3 / 3 → 
     ∃ (θ : ℝ), θ = real.pi / 3 → 
     dist B₁ B₂ = 28 / 3) := 
sorry

end chord_length_l49_49025


namespace greatest_product_digit_count_l49_49675

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49675


namespace cube_volume_surface_area_l49_49015

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s^3 = 8 * x ∧ 6 * s^2 = 2 * x) → x = 0 :=
by
  sorry

end cube_volume_surface_area_l49_49015


namespace who_saws_faster_l49_49744

-- Defining the conditions
def same_length_and_thickness (wood : Type) : Prop := sorry
def saw_into_3_sections (A : Type) (wood : Type) : Prop := sorry
def saw_into_2_sections (B : Type) (wood : Type) : Prop := sorry
def A_saws_24_pieces (A : Type) (wood : Type) : Prop := sorry
def B_saws_28_pieces (B : Type) (wood : Type) : Prop := sorry

-- The main theorem statement
theorem who_saws_faster (A B wood : Type) (h1 : same_length_and_thickness wood)
  (h2 : saw_into_3_sections A wood) (h3 : saw_into_2_sections B wood)
  (h4 : A_saws_24_pieces A wood) (h5 : B_saws_28_pieces B wood) : 
  (A takes less time to saw one piece of wood than B) :=
sorry

end who_saws_faster_l49_49744


namespace max_digits_in_product_l49_49606

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49606


namespace greatest_product_digits_l49_49565

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49565


namespace quadratic_root_form_eq_l49_49853

theorem quadratic_root_form_eq (c : ℚ) : 
  (∀ x : ℚ, x^2 - 7 * x + c = 0 → x = (7 + Real.sqrt (9 * c)) / 2 ∨ x = (7 - Real.sqrt (9 * c)) / 2) →
  c = 49 / 13 := 
by
  sorry

end quadratic_root_form_eq_l49_49853


namespace work_completion_together_l49_49759

theorem work_completion_together (man_days : ℕ) (son_days : ℕ) (together_days : ℕ) 
  (h_man : man_days = 10) (h_son : son_days = 10) : together_days = 5 :=
by sorry

end work_completion_together_l49_49759


namespace seonyeong_class_size_l49_49428

theorem seonyeong_class_size :
  (12 * 4 + 3) - 12 = 39 :=
by
  sorry

end seonyeong_class_size_l49_49428


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49635

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49635


namespace four_digit_numbers_count_eq_l49_49258

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49258


namespace greatest_number_of_digits_in_product_l49_49624

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49624


namespace count_four_digit_numbers_l49_49284

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49284


namespace four_digit_numbers_count_l49_49255

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49255


namespace four_digit_numbers_count_l49_49212

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49212


namespace four_digit_number_count_l49_49203

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49203


namespace max_digits_in_product_l49_49505

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49505


namespace greatest_number_of_digits_in_product_l49_49613

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49613


namespace part1_decreasing_interval_part2_tangent_lines_l49_49396

-- Define the curve C function
def f (x : ℝ) (a b : ℝ) : ℝ := x^3 - a * x + b

-- Define the derivative of curve C function
def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - a

-- Define the function g
def g (x : ℝ) (a : ℝ) : ℝ := Real.log x - (a / 6) * (f' x a + a) - 2 * x

-- Define the derivative of g
def g' (x : ℝ) (a : ℝ) : ℝ := (1 / x) - a * x - 2

-- Problem 1
theorem part1_decreasing_interval (a b : ℝ) :
  (∃ x : ℝ, x > 0 ∧ g' x a < 0) → a > -1 :=
sorry

-- Define the tangent point equation
def tangent_point_eq (c a b : ℝ) : ℝ := -2 * c^3 + 3 * c^2 - a + b

-- Problem 2
theorem part2_tangent_lines (a b : ℝ) :
  (∃ c1 c2 c3 : ℝ, tangent_point_eq c1 a b = 0 ∧
                    tangent_point_eq c2 a b = 0 ∧
                    tangent_point_eq c3 a b = 0 ∧
                    c1 ≠ c2 ∧ c1 ≠ c3 ∧ c2 ≠ c3) →
  0 < a - b ∧ a - b < 1 :=
sorry

end part1_decreasing_interval_part2_tangent_lines_l49_49396


namespace quadratic_satisfies_l49_49381

noncomputable def xi : Complex := sorry -- Definition for xi where xi^5 = 1 and xi ≠ 1

-- Conditions
axiom xi_prop1 : xi^5 = 1
axiom xi_prop2 : xi ≠ 1

def alpha : Complex := xi + xi^2
def beta : Complex := xi^3 + xi^4 

theorem quadratic_satisfies (a b : ℝ) (hαβ : α + β = -1) (hαβ_prod : α * β = xi^2 + xi + 4) :
  (∃ (a b : ℝ), a = 1 ∧ b = 4) :=
begin
  use [1, 4],
  split;
  refl,
end

end quadratic_satisfies_l49_49381


namespace four_digit_numbers_count_l49_49253

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49253


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49632

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49632


namespace greatest_product_digits_l49_49555

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49555


namespace greatest_product_digits_l49_49554

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49554


namespace four_digit_number_count_l49_49236

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49236


namespace sin_double_angle_l49_49913

-- Define the condition stating that sin θ + cos θ = 1/5
variable {θ : Real} (h : sin θ + cos θ = 1 / 5)

-- The goal is to prove that sin 2θ = -24/25 under the given condition
theorem sin_double_angle : sin (2 * θ) = -24 / 25 :=
by
  sorry

end sin_double_angle_l49_49913


namespace shoveling_driveways_l49_49371

-- Definitions of the conditions
def cost_of_candy_bars := 2 * 0.75
def cost_of_lollipops := 4 * 0.25
def total_cost := cost_of_candy_bars + cost_of_lollipops
def portion_of_earnings := total_cost * 6
def charge_per_driveway := 1.50
def number_of_driveways := portion_of_earnings / charge_per_driveway

-- The theorem to prove Jimmy shoveled 10 driveways
theorem shoveling_driveways :
  number_of_driveways = 10 := 
by
  sorry

end shoveling_driveways_l49_49371


namespace greatest_possible_number_of_digits_in_product_l49_49532

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49532


namespace cone_volume_increase_l49_49037

theorem cone_volume_increase (r h : ℝ) (k : ℝ) :
  let V := (1/3) * π * r^2 * h
  let h' := 2.60 * h
  let r' := r * (1 + k / 100)
  let V' := (1/3) * π * (r')^2 * h'
  let percentage_increase := ((V' / V) - 1) * 100
  percentage_increase = ((1 + k / 100)^2 * 2.60 - 1) * 100 :=
by
  sorry

end cone_volume_increase_l49_49037


namespace more_than_half_sunflower_seeds_on_friday_l49_49414

noncomputable def sunflower_seeds_day (d : ℕ) : ℚ :=
  if d = 0 then 0.9
  else 0.9 * (1 - 0.3)^(d - 1) + 0.9 * (1 - 0.2)*(1 - 0.1)*∑ i in finset.range (d - 1), (1 - 0.3)^i

noncomputable def seeds_day (d : ℕ) : ℚ :=
  3 * d

theorem more_than_half_sunflower_seeds_on_friday :
  ∀ d : ℕ, d = 5 → sunflower_seeds_day d > 0.5 * seeds_day d :=
by
  sorry

end more_than_half_sunflower_seeds_on_friday_l49_49414


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49633

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49633


namespace incorrect_conversion_D_l49_49724

-- Definition of base conversions as conditions
def binary_to_decimal (b : String) : ℕ := -- Converts binary string to decimal number
  sorry

def octal_to_decimal (o : String) : ℕ := -- Converts octal string to decimal number
  sorry

def decimal_to_base_n (d : ℕ) (n : ℕ) : String := -- Converts decimal number to base-n string
  sorry

-- Given conditions
axiom cond1 : binary_to_decimal "101" = 5
axiom cond2 : octal_to_decimal "27" = 25 -- Note: "27"_base(8) is 2*8 + 7 = 23 in decimal; there's a typo in question's option.
axiom cond3 : decimal_to_base_n 119 6 = "315"
axiom cond4 : decimal_to_base_n 13 2 = "1101" -- Note: correcting from 62 to "1101"_base(2) which is 13

-- Prove the incorrect conversion between number systems
theorem incorrect_conversion_D : decimal_to_base_n 31 4 ≠ "62" :=
  sorry

end incorrect_conversion_D_l49_49724


namespace cookies_left_l49_49943

/-- Define the number of pans of cookies baked, cookies per pan, cookies eaten, and cookies burnt --/
variable (p : ℕ) (c : ℕ) (e : ℕ) (b : ℕ)

/-- Provide the specific values for each variable --/
def p := 12
def c := 15
def e := 9
def b := 6

/-- Define the total number of cookies baked and the number of cookies unavailable (eaten + burnt) --/
def total_cookies := p * c
def unavailable_cookies := e + b

/-- The proposition that the number of cookies left is 165 --/
theorem cookies_left : total_cookies - unavailable_cookies = 165 := by
  sorry

end cookies_left_l49_49943


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49708

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49708


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49626

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49626


namespace four_digit_numbers_count_l49_49225

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49225


namespace problem_1_problem_2_l49_49916

theorem problem_1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : ∀ x, |x + a| + |x - b| + c ≥ 4) : 
  a + b + c = 4 :=
sorry

theorem problem_2 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 4) : 
  (1/4) * a^2 + (1/9) * b^2 + c^2 = 8 / 7 :=
sorry

end problem_1_problem_2_l49_49916


namespace four_digit_numbers_count_eq_l49_49261

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49261


namespace greatest_number_of_digits_in_product_l49_49690

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49690


namespace greatest_product_digits_l49_49660

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49660


namespace minimum_seedlings_needed_l49_49478

theorem minimum_seedlings_needed (n : ℕ) (h1 : 75 ≤ n) (h2 : n ≤ 80) (H : 1200 * 100 / n = 1500) : n = 80 :=
sorry

end minimum_seedlings_needed_l49_49478


namespace bucket_capacity_l49_49749

theorem bucket_capacity (x : ℝ) (hx1 : x > 8) (hx2 : x ∈ {7, 9, 11, 13}) :
  ((x - 8) - 4 * ((x - 8) / x)) ≤ 0.2 * x → x ∈ {9, 11} :=
by
  sorry

end bucket_capacity_l49_49749


namespace problem1_problem2_l49_49045

-- Problem 1: Prove the solutions to the equation
theorem problem1 (x : ℝ) : 4 * (x + 1)^2 = 49 ↔ (x = 5 / 2 ∨ x = -9 / 2) :=
by sorry

-- Problem 2: Prove the value of the expression
theorem problem2 : 
  sqrt 9 - (-1)^2018 - sqrt 27 + abs (2 - sqrt 5) = sqrt 5 - 3 :=
by sorry

end problem1_problem2_l49_49045


namespace max_digits_in_product_l49_49596

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49596


namespace intersection_is_isosceles_right_angled_l49_49188

def is_isosceles_triangle (x : Type) : Prop := sorry -- Definition of isosceles triangle
def is_right_angled_triangle (x : Type) : Prop := sorry -- Definition of right-angled triangle

def M : Set Type := {x | is_isosceles_triangle x}
def N : Set Type := {x | is_right_angled_triangle x}

theorem intersection_is_isosceles_right_angled :
  (M ∩ N) = {x | is_isosceles_triangle x ∧ is_right_angled_triangle x} := by
  sorry

end intersection_is_isosceles_right_angled_l49_49188


namespace integral_of_piecewise_l49_49891

noncomputable def f : ℝ → ℝ :=
λ x, if -1 ≤ x ∧ x ≤ 0 then x^2 else 
     if 0 < x ∧ x ≤ 1 then 1 else 0

theorem integral_of_piecewise :
  ∫ x in (-1 : ℝ)..(1 : ℝ), f x = 4 / 3 :=
by
  sorry

end integral_of_piecewise_l49_49891


namespace tan_degree_identity_l49_49159

theorem tan_degree_identity (k : ℝ) (hk : Real.cos (Real.pi * -80 / 180) = k) : 
  Real.tan (Real.pi * 100 / 180) = - (Real.sqrt (1 - k^2) / k) := 
by 
  sorry

end tan_degree_identity_l49_49159


namespace last_two_digits_of_sum_l49_49124

-- Condition stating that n! + 1 ends in 01 for n ≥ 12
def factorial_ends_in_01 (n : ℕ) (h : n ≥ 12) : (n! + 1) % 100 = 1 := sorry

-- Calculation that ensures the last two digits of 6! + 1 are 21
def factorial_6_plus_1 : (6! + 1) % 100 = 21 :=
by norm_num [nat.factorial]

-- Number of terms in the arithmetic sequence starting from 12 to 96 with diff 6
def arithmetic_sequence_count : (96 - 12) / 6 + 1 = 15 := 
by norm_num

-- Proof of the last two digits of the sum
theorem last_two_digits_of_sum : 
  (6! + 1 + ∑ i in finset.range 15, ((12 + i * 6)! + 1)) % 100 = 36 := 
by 
  have step1 := factorial_6_plus_1
  have step2 : (∑ i in finset.range 15, 1) % 100 = 15 := 
    by norm_num [finset.sum_const_nat, finset.card_range]
  have step3 : (21 + 15) % 100 = 36 := 
    by norm_num
  show (6! + 1 + ∑ i in finset.range 15, ((12 + i * 6)! + 1)) % 100 = 36 by
    rw [step1, sum_comm, step2, step3]

end last_two_digits_of_sum_l49_49124


namespace greatest_possible_digits_in_product_l49_49508

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49508


namespace geometric_sequence_fraction_l49_49922

noncomputable def a_n : ℕ → ℝ := sorry -- geometric sequence {a_n}
noncomputable def S : ℕ → ℝ := sorry   -- sequence sum S_n
def q : ℝ := sorry                     -- common ratio

theorem geometric_sequence_fraction (h_sequence: ∀ n, 2 * S (n - 1) = S n + S (n + 1))
  (h_q: ∀ n, a_n (n + 1) = q * a_n n)
  (h_q_neg2: q = -2) :
  (a_n 5 + a_n 7) / (a_n 3 + a_n 5) = 4 :=
by 
  sorry

end geometric_sequence_fraction_l49_49922


namespace printing_machine_completion_time_l49_49766

-- Definitions of times in hours
def start_time : ℕ := 9 -- 9:00 AM
def half_job_time : ℕ := 12 -- 12:00 PM
def completion_time : ℕ := 15 -- 3:00 PM

-- Time taken to complete half the job
def half_job_duration : ℕ := half_job_time - start_time

-- Total time to complete the entire job
def total_job_duration : ℕ := 2 * half_job_duration

-- Proof that the machine will complete the job at 3:00 PM
theorem printing_machine_completion_time : 
    start_time + total_job_duration = completion_time :=
sorry

end printing_machine_completion_time_l49_49766


namespace _l49_49163

def A (x : ℝ) : Prop := -1 < x ∧ x < 6
def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a
def p (x : ℝ) : Prop := A x
def q (x : ℝ) (a : ℝ) : Prop := B x a

theorem (a : ℝ) (h : a > 0) : (∀ x, ¬(p x) → ¬(q x a)) → a ≥ 5 := 
sorry

theorem (a : ℝ) (h : a > 0) : (¬(∀ x, ¬(p x) → q x a)) → (0 < a ∧ a ≤ 2) := 
sorry

end _l49_49163


namespace count_four_digit_numbers_l49_49315

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49315


namespace eval_expr_l49_49109

theorem eval_expr (b c : ℕ) (hb : b = 2) (hc : c = 5) : b^3 * b^4 * c^2 = 3200 :=
by {
  -- the proof is omitted
  sorry
}

end eval_expr_l49_49109


namespace greatest_product_digits_l49_49666

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49666


namespace probability_prime_sum_l49_49778

noncomputable def integers_coprime_with_9 : set ℕ := {1, 2, 4, 5, 7, 8}

noncomputable def pairs_with_prime_sum : set (ℕ × ℕ) := 
  {p | p ∈ ({(a, b) | a ≠ b ∧ a ∈ integers_coprime_with_9 ∧ b ∈ integers_coprime_with_9}) ∧ 
  nat.prime (p.1 + p.2)}

theorem probability_prime_sum : (pairs_with_prime_sum.card : ℚ) / (finset.card (finset.univ.pairs ⟨integers_coprime_with_9.finite_to_finset⟩)).card = 1 / 3 :=
by sorry

end probability_prime_sum_l49_49778


namespace XiaoMaHu_correct_calculation_l49_49725

theorem XiaoMaHu_correct_calculation :
  (∃ A B C D : Prop, (A = ((a b : ℝ) → (a - b)^2 = a^2 - b^2)) ∧ 
                   (B = ((a : ℝ) → (-2 * a^3)^2 = 4 * a^6)) ∧ 
                   (C = ((a : ℝ) → a^3 + a^2 = 2 * a^5)) ∧ 
                   (D = ((a : ℝ) → -(a - 1) = -a - 1)) ∧ 
                   (¬A ∧ B ∧ ¬C ∧ ¬D)) :=
sorry

end XiaoMaHu_correct_calculation_l49_49725


namespace max_digits_in_product_l49_49496

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49496


namespace books_per_shelf_l49_49799

theorem books_per_shelf (total_books bookshelves : ℕ) (h_books : total_books = 38) (h_shelves : bookshelves = 19) :
  (total_books / bookshelves) = 2 :=
by {
  rw [h_books, h_shelves],
  norm_num,
  sorry
}

end books_per_shelf_l49_49799


namespace max_digits_in_product_l49_49607

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49607


namespace max_digits_of_product_l49_49641

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49641


namespace arctan_sum_eq_pi_div_two_l49_49831

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l49_49831


namespace students_failed_l49_49471

theorem students_failed (total_students : ℕ) (A_percentage : ℚ) (fraction_remaining_B_or_C : ℚ) :
  total_students = 32 → A_percentage = 0.25 → fraction_remaining_B_or_C = 1/4 →
  let students_A := total_students * A_percentage.to_nat in
  let remaining_students := total_students - students_A in
  let students_B_or_C := remaining_students * fraction_remaining_B_or_C.to_nat in
  let students_failed := remaining_students - students_B_or_C in
  students_failed = 18 :=
by
  intros
  simp [students_A, remaining_students, students_B_or_C]
  sorry

end students_failed_l49_49471


namespace calculate_expression_l49_49091

theorem calculate_expression (a b c d : ℤ) (h1 : 3^0 = 1) (h2 : (-1 / 2 : ℚ)^(-2 : ℤ) = 4) : 
  (202 : ℤ) * 3^0 + (-1 / 2 : ℚ)^(-2 : ℤ) = 206 :=
by
  sorry

end calculate_expression_l49_49091


namespace find_naturals_divisibility_l49_49871

theorem find_naturals_divisibility :
  {n : ℕ | (2^n + n) ∣ (8^n + n)} = {1, 2, 4, 6} :=
by sorry

end find_naturals_divisibility_l49_49871


namespace jill_net_monthly_salary_l49_49107

variable (S : ℝ) -- Jill's net monthly salary

-- Conditions
variable (discretionary_income : ℝ) := S / 5
variable (vacation_fund : ℝ) := 0.3 * discretionary_income
variable (savings : ℝ) := 0.2 * discretionary_income
variable (eating_out : ℝ) := 0.35 * discretionary_income
variable (gifts_charity : ℝ) := 102.0

-- Proof Statement
theorem jill_net_monthly_salary : S = 3400 := by
  sorry

end jill_net_monthly_salary_l49_49107


namespace max_digits_in_product_l49_49599

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49599


namespace solve_for_a_l49_49161

open Complex

theorem solve_for_a (a : ℝ) (h : ∃ x : ℝ, (2 * Complex.I - (a * Complex.I) / (1 - Complex.I) = x)) : a = 4 := 
sorry

end solve_for_a_l49_49161


namespace problem1_problem2_l49_49736

-- Problem 1
theorem problem1 : 
  (2 * Nat.descFactorial 8 5 + 7 * Nat.descFactorial 8 4) / (Nat.descFactorial 8 8 + Nat.descFactorial 9 5) = 5 / 11 := 
by 
  sorry

-- Problem 2
theorem problem2 : 
  binomial 200 192 + binomial 200 196 + 2 * binomial 200 197 = 67331650 :=
by 
  sorry

end problem1_problem2_l49_49736


namespace exists_point_with_at_most_3_nearest_l49_49392

variable {Point : Type} [MetricSpace Point]

def has_at_most_3_nearest (M : Finset Point) : Prop :=
∃ (p ∈ M), (M.filter (λ q, dist p q = inf (M.erase p).image (dist p))).card ≤ 3

theorem exists_point_with_at_most_3_nearest (M : Finset Point) : has_at_most_3_nearest M := 
by 
s

end exists_point_with_at_most_3_nearest_l49_49392


namespace probability_distinct_real_roots_l49_49427

theorem probability_distinct_real_roots :
  let possible_vals := {n | 1 ≤ n ∧ n ≤ 6}
  let valid_pairs := { (a, b) | a ∈ possible_vals ∧ b ∈ possible_vals ∧ a^2 - 4 * b > 0 ∧ a > 2 ∧ b > a - 1 }
  let total_pairs : ℕ := 6 * 6
  let valid_pairs_count : ℕ := Fintype.card { (a, b) : ℕ × ℕ // (a, b) ∈ valid_pairs }
  valid_pairs_count / total_pairs = 1 / 12 := by
  sorry

end probability_distinct_real_roots_l49_49427


namespace count_four_digit_numbers_l49_49310

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49310


namespace number_of_four_digit_numbers_l49_49301

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49301


namespace max_digits_in_product_l49_49604

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49604


namespace hexagon_perimeter_l49_49190

-- Defining the side lengths of the hexagon
def side_lengths : List ℕ := [7, 10, 8, 13, 11, 9]

-- Defining the perimeter calculation
def perimeter (sides : List ℕ) : ℕ := sides.sum

-- The main theorem stating the perimeter of the given hexagon
theorem hexagon_perimeter :
  perimeter side_lengths = 58 := by
  -- Skipping proof here
  sorry

end hexagon_perimeter_l49_49190


namespace cube_volume_surface_area_eq_1728_l49_49012

theorem cube_volume_surface_area_eq_1728 (x : ℝ) (h1 : (side : ℝ) (v : ℝ) hvolume : v = 8 * x ∧ v = side^3) (h2: (side : ℝ) (a : ℝ) hsurface : a = 2 * x ∧ a = 6 * side^2) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_eq_1728_l49_49012


namespace four_digit_numbers_count_eq_l49_49262

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49262


namespace Iain_pennies_problem_l49_49324

theorem Iain_pennies_problem :
  ∀ (P : ℝ), 200 - 30 = 170 →
             170 - (P / 100) * 170 = 136 →
             P = 20 :=
by
  intros P h1 h2
  sorry

end Iain_pennies_problem_l49_49324


namespace max_f_value_l49_49924

def f (x y z : ℝ) : ℝ :=
  real.sqrt (2 * x + 13) + real.cbrt (3 * y + 5) + (8 * z + 12).pow (1 / 4)

theorem max_f_value (x y z : ℝ) (h : x + y + z = 3) : f x y z ≤ 8 :=
sorry

end max_f_value_l49_49924


namespace arctan_sum_pi_div_two_l49_49820

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l49_49820


namespace trajectory_equation_circle_equation_l49_49990

noncomputable def circle_trajectory_eq (a b R : ℝ)
  (intercept_x : R^2 - b^2 = 2)
  (intercept_y : R^2 - a^2 = 3) : Prop :=
b^2 - a^2 = 1

noncomputable def circle_eq (a b R : ℝ)
  (intercept_x : R^2 - b^2 = 2)
  (intercept_y : R^2 - a^2 = 3)
  (distance_condition : |b - a| = sqrt(2) / 2) : Prop :=
a = 0 ∧ (b = 1 ∨ b = -1) ∧ R = sqrt(3)

theorem trajectory_equation (a b R : ℝ)
  (intercept_x : R^2 - b^2 = 2)
  (intercept_y : R^2 - a^2 = 3) : circle_trajectory_eq a b R intercept_x intercept_y :=
sorry

theorem circle_equation (a b R : ℝ)
  (intercept_x : R^2 - b^2 = 2)
  (intercept_y : R^2 - a^2 = 3)
  (distance_condition : |b - a| = sqrt(2) / 2) : circle_eq a b R intercept_x intercept_y distance_condition :=
sorry

end trajectory_equation_circle_equation_l49_49990


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49709

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49709


namespace greatest_number_of_digits_in_product_l49_49620

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49620


namespace minimize_sum_of_distances_at_A5_l49_49154

theorem minimize_sum_of_distances_at_A5
  (A : ℕ → ℝ) (Q : ℝ)
  (h_sorted : ∀ i, A i < A (i + 1))
  (h_length : ∀ i, i < 9) :
  let t := (∑ i in Range 9, |Q - A i|) in
  Q = A 4 → t = (∑ i in Range 9, |A 4 - A i|) :=
by sorry

end minimize_sum_of_distances_at_A5_l49_49154


namespace cube_dimension_l49_49018

theorem cube_dimension (x s : ℝ) (hx1 : s^3 = 8 * x) (hx2 : 6 * s^2 = 2 * x) : x = 1728 := 
by {
  sorry
}

end cube_dimension_l49_49018


namespace count_four_digit_numbers_l49_49292

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49292


namespace expensive_feed_cost_l49_49476

/-- Tim and Judy mix two kinds of feed for pedigreed dogs. They made 35 pounds of feed worth 0.36 dollars per pound by mixing one kind worth 0.18 dollars per pound with another kind. They used 17 pounds of the cheaper kind in the mix. What is the cost per pound of the more expensive kind of feed? --/
theorem expensive_feed_cost 
  (total_feed : ℝ := 35) 
  (avg_cost : ℝ := 0.36) 
  (cheaper_feed : ℝ := 17) 
  (cheaper_cost : ℝ := 0.18) 
  (total_cost : ℝ := total_feed * avg_cost) 
  (cheaper_total_cost : ℝ := cheaper_feed * cheaper_cost) 
  (expensive_feed : ℝ := total_feed - cheaper_feed) : 
  (total_cost - cheaper_total_cost) / expensive_feed = 0.53 :=
by
  sorry

end expensive_feed_cost_l49_49476


namespace students_failed_l49_49472

theorem students_failed (total_students : ℕ) (A_percentage : ℚ) (fraction_remaining_B_or_C : ℚ) :
  total_students = 32 → A_percentage = 0.25 → fraction_remaining_B_or_C = 1/4 →
  let students_A := total_students * A_percentage.to_nat in
  let remaining_students := total_students - students_A in
  let students_B_or_C := remaining_students * fraction_remaining_B_or_C.to_nat in
  let students_failed := remaining_students - students_B_or_C in
  students_failed = 18 :=
by
  intros
  simp [students_A, remaining_students, students_B_or_C]
  sorry

end students_failed_l49_49472


namespace eccentricity_of_hyperbola_l49_49899

variables {a b c : ℝ} (h₀ : a > b) (h₁ : b > 0)
def hyperbola (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

theorem eccentricity_of_hyperbola (h : b = 3 * a) : 
    let c := sqrt (a^2 + b^2) in 
    let e := c / a in 
    e = sqrt 10 :=
by sorry

end eccentricity_of_hyperbola_l49_49899


namespace day_of_week_Feb1_l49_49959

-- Proving that if February 14 is a Saturday, what day of the week is February 1?
theorem day_of_week_Feb1
    (feb14_day : ℕ)
    (h : feb14_day % 7 = 6) : 
    1 = 0 % 7 :=
by
  -- By reducing to known modulo days we compute the answer
  have feb1_day := (feb14_day - 13) % 7
  rw [sub_eq_iff_eq_add, add_comm, add_mod, zero_add, add_comm, mod_mod] at feb1_day
  exact feb1_day.symm

end day_of_week_Feb1_l49_49959


namespace max_sin_cos_ratio_l49_49327

noncomputable def angles := { α β γ : ℝ // (0 < α ∧ α < π/2) ∧ (0 < β ∧ β < π/2) ∧ (0 < γ ∧ γ < π/2) }

theorem max_sin_cos_ratio (α β γ : angles) 
  (h : sin α.val ^ 2 + sin β.val ^ 2 + sin γ.val ^ 2 = 1) :
  ∀ x : ℝ, x = (sin α.val + sin β.val + sin γ.val) / (cos α.val + cos β.val + cos γ.val) → 
  x ≤ sqrt 2 / 2 :=
sorry

end max_sin_cos_ratio_l49_49327


namespace range_of_a_l49_49971

theorem range_of_a {a : ℝ} (h : ∃ b ∈ set.Icc (1 : ℝ) 2, (2 ^ b) * (b + a) ≥ 4) : a ∈ set.Ici (-1) :=
by
  sorry

end range_of_a_l49_49971


namespace greatest_number_of_digits_in_product_l49_49686

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49686


namespace sqrt_pattern_l49_49915

theorem sqrt_pattern (a : ℕ)
  (h_n2 : sqrt(2 + 2 / (2^2 - 1)) = 2 * sqrt(2 / (2^2 - 1)))
  (h_n3 : sqrt(3 + 3 / (3^2 - 1)) = 3 * sqrt(3 / (3^2 - 1)))
  (h_n4 : sqrt(4 + 4 / (4^2 - 1)) = 4 * sqrt(4 / (4^2 - 1)))
  (h_n8 : sqrt(8 + 8 / a) = 8 * sqrt(8 / a)) : a = 63 := by
  sorry

end sqrt_pattern_l49_49915


namespace total_books_proof_l49_49977

def physics_books (P : ℕ) === 3 * chemistry_books (C : ℕ) / 2 := sorry
def chemistry_books (C : ℕ) === 4 * biology_books (B : ℕ) / 3 := sorry
def biology_books (B : ℕ) === 5 * mathematics_books (M : ℕ) / 6 := sorry
def mathematics_books (M : ℕ) := M
def history_books (H : ℕ) === 8 * mathematics_books (M : ℕ) / 7 := sorry

def total_books (P C B M H : ℕ) > 10000 := sorry
def min_mathematics_books (M : ℕ) ≥ 1000 := sorry

def possible_total_books := (65 / 14) * 2155 := 10050

theorem total_books_proof : 
  ∀ P C B M H,
    (physics_books P) ∧ (chemistry_books C) ∧ (biology_books B) ∧
    (mathematics_books M) ∧ (history_books H) ∧ (total_books P C B M H) ∧ (min_mathematics_books M) →
    (possible_total_books = 10050) := sorry

end total_books_proof_l49_49977


namespace brianna_remaining_money_l49_49798

variable (m c n : ℕ)

theorem brianna_remaining_money (h : (1 / 5 : ℝ) * m = (1 / 3 : ℝ) * n * c) : (m - n * c) / m = 2 / 5 :=
by
  have hnc : n * c = (3 / 5 : ℝ) * m := by
    rw ← mul_assoc
    rw ← (div_eq_mul_one_div _ _).symm
    rw h
    ring

  have h1 : (m - n * c) = m - (3 / 5 : ℝ) * m := by
    rw hnc

  have h2 : 1 = 5 / 5 := by norm_num

  have h3 : (5 / 5) * m = m := by rw [h2, mul_one]

  have h4 : (m - (3 / 5) * m) = (2 / 5) * m := by
    rw [← sub_mul, h3]
    norm_num

  rw div_eq_mul_inv
  rw ← h4
  norm_num

  sorry

end brianna_remaining_money_l49_49798


namespace units_digit_17_pow_53_l49_49722

theorem units_digit_17_pow_53 : (17^53) % 10 = 7 := 
by sorry

end units_digit_17_pow_53_l49_49722


namespace problem1_problem2_l49_49044

-- Problem 1
theorem problem1 : ( (1/2 : ℝ)^(-2) + (2023 - Real.sqrt 121)^0 - |(-5)| - 2 * Real.cos (Real.pi / 4) ) = -Real.sqrt 2 :=
by sorry

-- Problem 2
theorem problem2 (a : ℝ) (h : a ≠ 0) (h2: a ≠ 2) :
    ( (a^2 - 4) / a / (a - (4*a - 4) / a) - 2 / (a - 2) ) = a / (a - 2) :=
by sorry

end problem1_problem2_l49_49044


namespace fabian_spent_l49_49111

theorem fabian_spent : ∀ (mouseCost : ℕ) (keyboardCost : ℕ),
  mouseCost = 16 ∧ keyboardCost = 3 * mouseCost → 
  mouseCost + keyboardCost = 64 :=
by
  intros mouseCost keyboardCost h
  have h1 : mouseCost = 16 := h.1
  have h2 : keyboardCost = 3 * mouseCost := h.2
  rw [h1] at h2
  rw [h1, h2]
  sorry

end fabian_spent_l49_49111


namespace largest_circle_covered_l49_49127

-- Definition of the radii R1, R2, and R3
variables {R1 R2 R3 : ℝ}

-- Helper function to check if three lengths can form an acute triangle
def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

-- Definition of the maximum function for three radii
def max_radius (R1 R2 R3 : ℝ) : ℝ :=
  max R1 (max R2 R3)

-- The proof statement in Lean 4
theorem largest_circle_covered
  (R1 R2 R3 : ℝ) :
  (is_acute_triangle (2*R1) (2*R2) (2*R3) → 
    ∃ r : ℝ, r = (2 * max_radius R1 R2 R3) / Real.sqrt 3) ∧
  (¬is_acute_triangle (2*R1) (2*R2) (2*R3) → 
    ∃ r : ℝ, r = max_radius R1 R2 R3) :=
by
  sorry

end largest_circle_covered_l49_49127


namespace different_signs_l49_49968

noncomputable def f : ℝ → ℝ := sorry

theorem different_signs (h0 : 0 < 4)
                       (h1 : 0 < 2)
                       (h2 : 1 < 3/2)
                       (h3 : 5/4 < 3/2)
                       (h_cont : continuous f)
                       (h_zero1 : ∃! c, 0 < c ∧ c < 4 ∧ f c = 0)
                       (h_zero2 : ∃! c, 0 < c ∧ c < 2 ∧ f c = 0)
                       (h_zero3 : ∃! c, 1 < c ∧ c < 3/2 ∧ f c = 0)
                       (h_zero4 : ∃! c, 5/4 < c ∧ c < 3/2 ∧ f c = 0) :
  (f 4 < 0 ∧ f 0 > 0) ∨ (f 0 < 0 ∧ f 4 > 0) ∧
  (f 2 < 0 ∧ f 0 > 0) ∨ (f 0 < 0 ∧ f 2 > 0) ∧
  (f (3/2) < 0 ∧ f 0 > 0) ∨ (f 0 < 0 ∧ f (3/2) > 0) :=
sorry

end different_signs_l49_49968


namespace train_speed_l49_49770

variable (length_train : ℝ)
variable (time : ℝ)
variable (length_platform : ℝ)

theorem train_speed :
  length_train = 250 ∧
  time = 17.998560115190784 ∧
  length_platform = 200 →
  let total_distance := length_train + length_platform in
  let speed_mps := total_distance / time in
  let speed_kmph := speed_mps * 3.6 in
  abs (speed_kmph - 90.01) < 1e-2 :=
by
  intros h
  sorry

end train_speed_l49_49770


namespace four_digit_numbers_count_l49_49248

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49248


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49630

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49630


namespace toby_garage_sale_total_l49_49479

def treadmill_price := 200
def chest_of_drawers_price := treadmill_price / 2
def television_price := 3 * treadmill_price
def bicycle_price := chest_of_drawers_price * 0.75
def antique_vase_price := bicycle_price + 50

def total_sales_5_items := 
  treadmill_price + chest_of_drawers_price + television_price + bicycle_price + antique_vase_price

def total_sales := total_sales_5_items / 0.85

theorem toby_garage_sale_total : total_sales = 1294.12 := 
by
  -- proving the total sales calculation
  sorry

end toby_garage_sale_total_l49_49479


namespace four_digit_numbers_count_eq_l49_49268

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49268


namespace calculate_dollar_value_l49_49098

def dollar (x y : ℤ) : ℤ := x * (y + 2) + x * y - 5

theorem calculate_dollar_value : dollar 3 (-1) = -5 := by
  sorry

end calculate_dollar_value_l49_49098


namespace cosine_of_3pi_over_2_l49_49114

theorem cosine_of_3pi_over_2 : Real.cos (3 * Real.pi / 2) = 0 := by
  sorry

end cosine_of_3pi_over_2_l49_49114


namespace solution_l49_49384

noncomputable def is_monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x ≤ y → f x ≤ f y

noncomputable def is_monotonic_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, x ≤ y → f y ≤ f x

noncomputable def is_monotonic (f : ℝ → ℝ) : Prop :=
  is_monotonic_increasing f ∨ is_monotonic_decreasing f

noncomputable def problem (f g : ℝ → ℝ) (hf : is_monotonic f) (hg : is_monotonic g) : Prop :=
  let p1 := is_monotonic_increasing f ∧ is_monotonic_increasing g ∧ is_monotonic_increasing (λ x, f x - g x)
  let p2 := is_monotonic_increasing f ∧ is_monotonic_decreasing g ∧ is_monotonic_increasing (λ x, f x - g x)
  let p3 := is_monotonic_decreasing f ∧ is_monotonic_increasing g ∧ is_monotonic_decreasing (λ x, f x - g x)
  let p4 := is_monotonic_decreasing f ∧ is_monotonic_decreasing g ∧ is_monotonic_decreasing (λ x, f x - g x)
  p2 ∧ p3

theorem solution (f g : ℝ → ℝ) (hf : is_monotonic f) (hg : is_monotonic g) : problem f g hf hg :=
  sorry

end solution_l49_49384


namespace greatest_product_digit_count_l49_49682

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49682


namespace max_digits_of_product_l49_49548

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49548


namespace commission_percentage_l49_49072

def SaleA (total : ℕ) := 
  if total <= 400 
    then 0.15 * total 
    else 0.15 * 400 + 0.22 * (total - 400)

def SaleB (total : ℕ) := 
  if total <= 500 
    then 0.20 * total 
    else 0.20 * 500 + 0.25 * (total - 500)

def SaleC (total : ℕ) := 
  if total <= 600 
    then 0.18 * total 
    else 0.18 * 600 + 0.30 * (total - 600)

def TotalCommission : ℕ := (SaleA 700) + (SaleB 800) + (SaleC 900)

def CombinedTotal : ℕ := 700 + 800 + 900

theorem commission_percentage :
  let total_commission := TotalCommission in
  let combined_total := CombinedTotal in
  total_commission = 499 ∧ 
  (total_commission / combined_total: ℝ) * 100 ≈ 20.79 :=
by
  sorry

end commission_percentage_l49_49072


namespace smallest_k_l49_49996

theorem smallest_k (s : Finset ℕ) (h₁ : ∀ x ∈ s, 1 ≤ x ∧ x ≤ 99) (h₂ : s.card = 7) :
  ∃ a b ∈ s, a ≠ b ∧ (1 / 2 : ℝ) ≤ (b : ℝ) / (a : ℝ) ∧ (b : ℝ) / (a : ℝ) ≤ 2 := 
sorry

end smallest_k_l49_49996


namespace cosine_of_3pi_over_2_l49_49113

theorem cosine_of_3pi_over_2 : Real.cos (3 * Real.pi / 2) = 0 := by
  sorry

end cosine_of_3pi_over_2_l49_49113


namespace sandy_age_correct_l49_49039

def is_age_ratio (S M : ℕ) : Prop := S * 9 = M * 7
def is_age_difference (S M : ℕ) : Prop := M = S + 12

theorem sandy_age_correct (S M : ℕ) (h1 : is_age_ratio S M) (h2 : is_age_difference S M) : S = 42 := by
  sorry

end sandy_age_correct_l49_49039


namespace line_tangent_to_parabola_l49_49060

theorem line_tangent_to_parabola (d : ℝ) :
  (∀ x y: ℝ, y = 3 * x + d ↔ y^2 = 12 * x) → d = 1 :=
by
  sorry

end line_tangent_to_parabola_l49_49060


namespace find_ratio_at_4_l49_49453

-- Define the problem parameters and conditions
def p (x : ℝ) : ℝ := 9 * (x - 2)^2
def q (x : ℝ) : ℝ := (x - 3) * (x + 3)

-- The theorem statement
theorem find_ratio_at_4 : p 4 / q 4 = 36 / 7 :=
by
  have hq : q 4 = (4 - 3) * (4 + 3), from rfl
  have hp : p 4 = 9 * (4 - 2)^2, from rfl
  rw [hp, hq]
  norm_num
  sorry

end find_ratio_at_4_l49_49453


namespace max_sum_m_n_l49_49170

noncomputable def ellipse_and_hyperbola_max_sum : Prop :=
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ (∃ x y : ℝ, (x^2 / 25 + y^2 / m^2 = 1 ∧ x^2 / 7 - y^2 / n^2 = 1)) ∧
  (25 - m^2 = 7 + n^2) ∧ (m + n = 6)

theorem max_sum_m_n : ellipse_and_hyperbola_max_sum :=
  sorry

end max_sum_m_n_l49_49170


namespace simplify_expression_l49_49022

variable {x y z : ℝ} 
variable (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)

theorem simplify_expression :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (x * y * z)⁻¹ * (x + y + z)⁻¹ :=
sorry

end simplify_expression_l49_49022


namespace greatest_possible_digits_in_product_l49_49517

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49517


namespace greatest_possible_number_of_digits_in_product_l49_49527

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49527


namespace max_digits_in_product_l49_49504

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49504


namespace four_digit_numbers_count_l49_49222

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49222


namespace cos_of_three_pi_div_two_l49_49117

theorem cos_of_three_pi_div_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  sorry

end cos_of_three_pi_div_two_l49_49117


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49702

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49702


namespace number_of_four_digit_numbers_l49_49299

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49299


namespace andrew_balloons_left_l49_49078

-- Declare the conditions as variables.
def blue_balloons : ℕ := 303
def purple_balloons : ℕ := 453
def total_balloons : ℕ := blue_balloons + purple_balloons
def half_balloons (total : ℕ) : ℕ := total / 2

-- The statement to be proved.
theorem andrew_balloons_left : half_balloons total_balloons = 378 :=
by
  have total_eq : total_balloons = 756 := by rfl
  rw [total_eq]
  have half_eq : half_balloons 756 = 378 := by rfl
  rw [half_eq]
  exact sorry -- Proof placeholder

end andrew_balloons_left_l49_49078


namespace greatest_product_digits_l49_49556

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49556


namespace number_property_l49_49869

theorem number_property (n : ℕ) (h : n = 7101449275362318840579) :
  n / 7 = 101449275362318840579 :=
sorry

end number_property_l49_49869


namespace four_digit_number_count_l49_49208

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49208


namespace greatest_number_of_digits_in_product_l49_49696

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49696


namespace petya_vasya_same_sum_l49_49423

theorem petya_vasya_same_sum :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 2^99 * (2^100 - 1) :=
by
  sorry

end petya_vasya_same_sum_l49_49423


namespace charles_richard_difference_in_dimes_l49_49092

variable (q : ℕ)

-- Charles' quarters
def charles_quarters : ℕ := 5 * q + 1

-- Richard's quarters
def richard_quarters : ℕ := q + 5

-- Difference in quarters
def diff_quarters : ℕ := charles_quarters q - richard_quarters q

-- Difference in dimes
def diff_dimes : ℕ := (diff_quarters q) * 5 / 2

theorem charles_richard_difference_in_dimes : diff_dimes q = 10 * (q - 1) := by
  sorry

end charles_richard_difference_in_dimes_l49_49092


namespace number_of_polynomials_in_H_l49_49375

-- Define the set H as polynomials with specific properties
def H := { Q : Polynomial ℤ // ∃ (n : ℕ) (c : Fin n → ℤ), 
            Q = Polynomial.X^n + ∑ i in Finset.range (n-1), c i * Polynomial.X^i - 50 ∧
            ∀ r ∈ Q.roots, isDistinctRoot r ∧ isIntegerRoot r ∧ sumZeroSubset r }

-- Stating the count of polynomials in H that meet the given conditions
theorem number_of_polynomials_in_H : ∃ N : ℕ, N = actual_count
:= sorry

-- Definitions used in the theorem
def isDistinctRoot (r : ℤ) : Prop := -- Definition of distinct root
  sorry

def isIntegerRoot (r : ℤ) : Prop := -- Definition of root being an integer
  sorry

def sumZeroSubset (roots : Set ℤ) : Prop := -- Definition of sum zero subset
  sorry

end

end number_of_polynomials_in_H_l49_49375


namespace combined_gross_profit_correct_l49_49758

def calculate_final_selling_price (initial_price : ℝ) (markup : ℝ) (discounts : List ℝ) : ℝ :=
  let marked_up_price := initial_price * (1 + markup)
  let final_price := List.foldl (λ price discount => price * (1 - discount)) marked_up_price discounts
  final_price

def calculate_gross_profit (initial_price : ℝ) (markup : ℝ) (discounts : List ℝ) : ℝ :=
  calculate_final_selling_price initial_price markup discounts - initial_price

noncomputable def combined_gross_profit : ℝ :=
  let earrings_gross_profit := calculate_gross_profit 240 0.25 [0.15]
  let bracelet_gross_profit := calculate_gross_profit 360 0.30 [0.10, 0.05]
  let necklace_gross_profit := calculate_gross_profit 480 0.40 [0.20, 0.05]
  let ring_gross_profit := calculate_gross_profit 600 0.35 [0.10, 0.05, 0.02]
  let pendant_gross_profit := calculate_gross_profit 720 0.50 [0.20, 0.03, 0.07]
  earrings_gross_profit + bracelet_gross_profit + necklace_gross_profit + ring_gross_profit + pendant_gross_profit

theorem combined_gross_profit_correct : combined_gross_profit = 224.97 :=
  by
  sorry

end combined_gross_profit_correct_l49_49758


namespace f_value_at_pi_over_4_l49_49178

noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π/6)

theorem f_value_at_pi_over_4 :
  f (π / 4) = sqrt 3 / 2 :=
by
  -- We state all the problem conditions here
  have h_omega : ∃ ω, ω = 2, by use 2
  have h_phi : ∃ φ, φ = π / 6, by use π / 6
  have h1 : ∀ x, f x = sin (2 * x + π / 6), from λ x, rfl
  sorry

end f_value_at_pi_over_4_l49_49178


namespace largest_lambda_property_l49_49123

theorem largest_lambda_property :
  ∃ λ : ℝ, (λ = Real.sqrt 3) ∧ (∀ p q r s : ℝ, p > 0 → q > 0 → r > 0 → s > 0 →
  ∃ (z : ℂ), let a := z.re in let b := z.im in
  |b| ≥ λ * |a| ∧ (p * z^3 + 2 * q * z^2 + 2 * r * z + s) * (q * z^3 + 2 * p * z^2 + 2 * s * z + r) = 0) :=
by
  use Real.sqrt 3
  intros p q r s hp hq hr hs
  sorry

end largest_lambda_property_l49_49123


namespace arctan_sum_pi_div_two_l49_49824

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l49_49824


namespace tangent_FE_circumcircle_EGH_l49_49354

-- Define the cyclic quadrilateral and associated points
variables {A B C D E F G H P : Type}

-- Assume conditions given in the problem
variables [cyclic_quadrilateral A B C D]
          [intersect_diagonals AC BD E]
          [intersect_lines AD BC F]
          [midpoints G_A_B G_C_D]

-- The main theorem
theorem tangent_FE_circumcircle_EGH :
  tangent_to_circumcircle FE (triangle E G H) := 
sorry

end tangent_FE_circumcircle_EGH_l49_49354


namespace bank_balance_after_two_years_l49_49436

theorem bank_balance_after_two_years :
  let P := 100 -- initial deposit
  let r := 0.1 -- annual interest rate
  let t := 2   -- time in years
  in P * (1 + r) ^ t = 121 :=
by
  sorry

end bank_balance_after_two_years_l49_49436


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49628

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49628


namespace largest_four_digit_number_divisible_by_7_with_different_digits_l49_49874

theorem largest_four_digit_number_divisible_by_7_with_different_digits : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 7 = 0) ∧ (∀ i j : ℕ, i ≠ j → nth_digit i n ≠ nth_digit j n) ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 7 = 0) ∧ (∀ i j : ℕ, i ≠ j → nth_digit i m ≠ nth_digit j m) → m ≤ n :=
begin
  sorry
end

def nth_digit (i : ℕ) (n : ℕ) : ℕ :=
  (n / 10^i) % 10 -- helper function to extract the i-th digit from the right

end largest_four_digit_number_divisible_by_7_with_different_digits_l49_49874


namespace greatest_product_digits_l49_49667

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49667


namespace max_digits_in_product_l49_49600

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49600


namespace greatest_product_digit_count_l49_49679

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49679


namespace greatest_product_digits_l49_49553

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49553


namespace max_digits_in_product_l49_49502

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49502


namespace cube_volume_surface_area_l49_49014

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s^3 = 8 * x ∧ 6 * s^2 = 2 * x) → x = 0 :=
by
  sorry

end cube_volume_surface_area_l49_49014


namespace A_beats_B_by_53_735_meters_l49_49350

theorem A_beats_B_by_53_735_meters :
  ∀ (time_A : ℝ) (race_distance : ℝ) (time_difference_B : ℝ) (speed_A : ℝ) (distance_B_11s : ℝ),
    time_A = 204.69 → race_distance = 1000 → time_difference_B = 11 →
    speed_A = race_distance / time_A →
    distance_B_11s = speed_A * time_difference_B →
    distance_B_11s = 53.735 :=
by
  intros time_A race_distance time_difference_B speed_A distance_B_11s
  assume h1 : time_A = 204.69
  assume h2 : race_distance = 1000
  assume h3 : time_difference_B = 11
  assume h4 : speed_A = race_distance / time_A
  assume h5 : distance_B_11s = speed_A * time_difference_B
  sorry

end A_beats_B_by_53_735_meters_l49_49350


namespace factorize_expression1_factorize_expression2_l49_49867

section
variable (x y : ℝ)

theorem factorize_expression1 : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
sorry

theorem factorize_expression2 : 3 * x^3 - 12 * x^2 * y + 12 * x * y^2 = 3 * x * (x - 2 * y)^2 :=
sorry
end

end factorize_expression1_factorize_expression2_l49_49867


namespace arctan_sum_eq_pi_div_two_l49_49828

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l49_49828


namespace greatest_digits_product_l49_49585

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49585


namespace distinct_roots_b_value_l49_49450

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - b * x^2 + 1

theorem distinct_roots_b_value :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 b = 0 ∧ f x2 b = 0) →
  b = 3/2 * real.cbrt 2 :=
by
  sorry

end distinct_roots_b_value_l49_49450


namespace cube_volume_surface_area_eq_1728_l49_49009

theorem cube_volume_surface_area_eq_1728 (x : ℝ) (h1 : (side : ℝ) (v : ℝ) hvolume : v = 8 * x ∧ v = side^3) (h2: (side : ℝ) (a : ℝ) hsurface : a = 2 * x ∧ a = 6 * side^2) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_eq_1728_l49_49009


namespace assign_teachers_to_classes_l49_49783

theorem assign_teachers_to_classes :
  ∃ assignments : Finset (Finset (Fin 4)),
    (assignments.card = 3) ∧
    (∀ c ∈ assignments, c.card ≥ 1) ∧
    (assignments.sum (λ c, c.card) = 4) ∧
    (assignments.product Finset.univ).card = 36 :=
by
  sorry

end assign_teachers_to_classes_l49_49783


namespace max_digits_in_product_l49_49497

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49497


namespace max_digits_in_product_l49_49603

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49603


namespace quadratic_has_sum_r_s_l49_49429

/-
  Define the quadratic equation 6x^2 - 24x - 54 = 0
-/
def quadratic_eq (x : ℝ) : Prop :=
  6 * x^2 - 24 * x - 54 = 0

/-
  Define the value 11 which is the sum r + s when completing the square
  for the above quadratic equation  
-/
def result_value := -2 + 13

/-
  State the proof that r + s = 11 given the quadratic equation.
-/
theorem quadratic_has_sum_r_s : ∀ x : ℝ, quadratic_eq x → -2 + 13 = 11 :=
by
  intros
  exact rfl

end quadratic_has_sum_r_s_l49_49429


namespace general_term_formula_range_of_n_l49_49390

section ArithmeticSequence

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {n : ℕ}

-- Conditions
def condition1 : Prop := S 9 = -a 5
def condition2 : Prop := a 3 = 4
def condition3 : Prop := a 1 > 0

-- 1. Proving the general term formula
theorem general_term_formula : condition1 ∧ condition2 → ∀ n, a n = -2 * n + 10 :=
sorry

-- 2. Proving the range of n for which S_n ≥ a_n
theorem range_of_n : condition1 ∧ condition2 ∧ condition3 → ∀ n, (1 ≤ n ∧ n ≤ 10) ↔ S n ≥ a n :=
sorry

end ArithmeticSequence

end general_term_formula_range_of_n_l49_49390


namespace find_initial_number_of_girls_l49_49059

def initial_number_of_girls (p g b : ℕ) : ℕ :=
    -- Initial number of girls is 0.5 * p. However, we use integer arithmetic
    let girls_initial := p / 2 in
    let students_after_first_change := p + 1 in
    let girls_after_first_change := girls_initial - 3 in
    let girls_after_second_change := girls_after_first_change + 1 in
    let students_after_second_change := p in
    -- At this stage, the girls should make up 40% of the student body
    have : 2 * (girls_after_second_change * 5) = students_after_second_change :=
    by linarith [g, b]
    girls_initial

theorem find_initial_number_of_girls (p : ℕ) (h : p / 2 = 10) : initial_number_of_girls p 3 4 = 10 :=
by linarith

-- The correct answer
example : find_initial_number_of_girls 20 10 = 10 := rfl

end find_initial_number_of_girls_l49_49059


namespace complement_M_l49_49191

open Set

-- Define the universal set U as the set of all real numbers
def U := ℝ

-- Define the set M as {x | |x| > 2}
def M : Set ℝ := {x | |x| > 2}

-- State that the complement of M (in the universal set U) is [-2, 2]
theorem complement_M : Mᶜ = {x | -2 ≤ x ∧ x ≤ 2} :=
by
  sorry

end complement_M_l49_49191


namespace ratio_of_sums_l49_49110

noncomputable def numerator_sequence_sum : ℕ := 
  let a := 4
  let d := 4
  let l := 60
  let n := (l - a) / d + 1
  (n * (a + l)) / 2

noncomputable def denominator_sequence_sum : ℕ := 
  let a := 5
  let d := 5
  let l := 75
  let n := (l - a) / d + 1
  (n * (a + l)) / 2

theorem ratio_of_sums :
  (numerator_sequence_sum.toRat / denominator_sequence_sum.toRat) = (4 : ℝ) / 5 :=
sorry

end ratio_of_sums_l49_49110


namespace max_digits_of_product_l49_49540

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49540


namespace min_distance_line_ellipse_chord_through_point_P_l49_49171

-- Problem 1: Proving the minimum distance from a point on line to a point on an ellipse.
theorem min_distance_line_ellipse :
  ∀ (M N : ℝ) (x y : ℝ),
    (x^2 / 4 + y^2 / 2 = 1) →
    (x / 4 + y / 2 = 1) →
    ∃ d : ℝ, d = (4 * sqrt 5 - 2 * sqrt 15) / 5 := by
  sorry

-- Problem 2: Proving the equation of the chord containing AB.
theorem chord_through_point_P :
  ∀ (x y : ℝ),
    (x^2 / 4 + y^2 / 2 = 1) →
    x = sqrt 2 ∨ 3 * sqrt 2 * x + 8 * y - 10 = 0 := by
  sorry

end min_distance_line_ellipse_chord_through_point_P_l49_49171


namespace four_digit_numbers_count_l49_49215

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49215


namespace rectangle_formation_l49_49165

variable (z1 z2 z3 z4 : ℂ)

-- Define the points on the unit circle
axiom unit_circle (z : ℂ) : |z| = 1

-- Assume the given conditions
theorem rectangle_formation (h1 : unit_circle z1) (h2 : unit_circle z2) (h3 : unit_circle z3) (h4 : unit_circle z4) (h5 : z1 + z2 + z3 + z4 = 0) : 
  ∃ (Z1 Z2 Z3 Z4 : ℂ), Z1 * Z2 = Z3 * Z4 ∧ Z1 * Z4 = Z3 * Z2 := 
sorry

end rectangle_formation_l49_49165


namespace tan_value_l49_49960

theorem tan_value (x : ℝ) (h0 : 0 < x) (h1 : x < π / 2) (h2 : sin x ^ 4 / 9 + cos x ^ 4 / 4 = 1 / 13) : tan x = 3 / 2 :=
sorry

end tan_value_l49_49960


namespace parabola_always_intersects_x_axis_find_integer_m_l49_49932

def quadratic_parabola (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + (m - 3) * x - 3

theorem parabola_always_intersects_x_axis (m : ℝ) (h : m > 0) :
  ∃ x : ℝ, quadratic_parabola m x = 0 :=
by
  have discriminant := (m - 3)^2 + 12 * m
  have discriminant_nonneg : discriminant > 0 := by nlinarith
  use (- (m - 3) + real.sqrt discriminant) / (2 * m)
  field_simp [quadratic_parabola, discriminant]
  sorry

theorem find_integer_m (m : ℝ) (h : m > 0) (h_int_roots : ∃ x1 x2 : ℤ, quadratic_parabola m x1 = 0 ∧ quadratic_parabola m x2 = 0) :
  m = 1 ∨ m = 3 :=
by
  have hx := h_int_roots
  rcases hx with ⟨x1, x2, h1, h2⟩
  have h_divisor := (abs (x2 - x1)) % 3 = 0
  have m_pos_divisors := h
  sorry

end parabola_always_intersects_x_axis_find_integer_m_l49_49932


namespace smallest_m_l49_49328

theorem smallest_m (m : ℤ) (h : 2 * m + 1 ≥ 0) : m ≥ 0 :=
sorry

end smallest_m_l49_49328


namespace number_of_four_digit_numbers_l49_49272

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49272


namespace problem_l49_49153

-- Definition of arithmetic sequence and its sum
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

def sum_of_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem problem (a_4 a_1 S_4 : ℤ) (d a_n k : ℤ) :
  a_4 - a_1 = 6 →
  S_4 = -20 →
  d = (a_4 - a_1) / 3 →
  a_1 = -8 →
  (∀ n, a_n = a_1 + (n - 1) * d) →
  a_n = 2 * n - 10 →
  sum_of_arithmetic_sequence a_1 d k = -18 →
  k = 3 ∨ k = 6 :=
sorry

end problem_l49_49153


namespace four_digit_numbers_count_l49_49217

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49217


namespace cube_volume_surface_area_eq_1728_l49_49011

theorem cube_volume_surface_area_eq_1728 (x : ℝ) (h1 : (side : ℝ) (v : ℝ) hvolume : v = 8 * x ∧ v = side^3) (h2: (side : ℝ) (a : ℝ) hsurface : a = 2 * x ∧ a = 6 * side^2) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_eq_1728_l49_49011


namespace find_x_l49_49036

theorem find_x (x y z : ℤ) (h1 : 4 * x + y + z = 80) (h2 : 2 * x - y - z = 40) (h3 : 3 * x + y - z = 20) : x = 20 := by
  sorry

end find_x_l49_49036


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49629

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49629


namespace greatest_product_digits_l49_49663

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49663


namespace students_failed_l49_49468

theorem students_failed (total_students : ℕ) (percent_A : ℚ) (fraction_BC : ℚ) (students_A : ℕ)
  (students_remaining : ℕ) (students_BC : ℕ) (students_failed : ℕ)
  (h1 : total_students = 32) (h2 : percent_A = 0.25) (h3 : fraction_BC = 0.25)
  (h4 : students_A = total_students * percent_A)
  (h5 : students_remaining = total_students - students_A)
  (h6 : students_BC = students_remaining * fraction_BC)
  (h7 : students_failed = total_students - students_A - students_BC) :
  students_failed = 18 :=
sorry

end students_failed_l49_49468


namespace triangle_area_identity_l49_49331

variable (a b c : ℝ)
noncomputable def s := (a + b + c) / 2
noncomputable def t := Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_identity (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (habc : a + b > c ∧ b + c > a ∧ c + a > b) :
  16 * t^2 = 2 * a^2 * b^2 + 2 * a^2 * c^2 + 2 * b^2 * c^2 - a^4 - b^4 - c^4 :=
by
  sorry

end triangle_area_identity_l49_49331


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49578

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49578


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49567

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49567


namespace arctan_triangle_complementary_l49_49818

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l49_49818


namespace factorization_of_2210_l49_49322

theorem factorization_of_2210 : 
  ∃! (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 2210) :=
sorry

end factorization_of_2210_l49_49322


namespace prime_sol_is_7_l49_49118

theorem prime_sol_is_7 (p : ℕ) (x y : ℕ) (hp : Nat.Prime p) 
  (hx : p + 1 = 2 * x^2) (hy : p^2 + 1 = 2 * y^2) : 
  p = 7 := 
  sorry

end prime_sol_is_7_l49_49118


namespace minimize_sum_of_sequence_l49_49935

theorem minimize_sum_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : 
  (∀ n, a n = 2 * n - 37) → 
  (∀ n, S n = ∑ i in range (n + 1), a i) → 
  (∀ n, n ≥ 0 → S n ≥ S 18) :=
by {
  sorry
}

end minimize_sum_of_sequence_l49_49935


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49569

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49569


namespace gcd_2048_2101_eq_1_l49_49490

theorem gcd_2048_2101_eq_1 : Int.gcd 2048 2101 = 1 := sorry

end gcd_2048_2101_eq_1_l49_49490


namespace greatest_product_digits_l49_49661

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49661


namespace greatest_product_digits_l49_49561

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49561


namespace pages_read_over_weekend_l49_49788

-- Define the given conditions
def total_pages : ℕ := 408
def days_left : ℕ := 5
def pages_per_day : ℕ := 59

-- Define the calculated pages to be read over the remaining days
def pages_remaining := days_left * pages_per_day

-- Define the pages read over the weekend
def pages_over_weekend := total_pages - pages_remaining

-- Prove that Bekah read 113 pages over the weekend
theorem pages_read_over_weekend : pages_over_weekend = 113 :=
by {
  -- proof should be here, but we place sorry since proof is not required
  sorry
}

end pages_read_over_weekend_l49_49788


namespace total_birds_on_fence_l49_49474

-- Definitions for the problem conditions
def initial_birds : ℕ := 12
def new_birds : ℕ := 8

-- Theorem to state that the total number of birds on the fence is 20
theorem total_birds_on_fence : initial_birds + new_birds = 20 :=
by
  -- Skip the proof as required
  sorry

end total_birds_on_fence_l49_49474


namespace arctan_triangle_complementary_l49_49819

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l49_49819


namespace coprime_lcm_product_square_implies_perfect_squares_l49_49192

theorem coprime_lcm_product_square_implies_perfect_squares
  (a b c : ℕ)
  (h_coprime : Nat.coprime a b ∧ Nat.coprime b c ∧ Nat.coprime c a)
  (h_lcm_square : ∃ k : ℕ, Nat.lcm a (Nat.lcm b c) = k * k)
  (h_prod_square : ∃ m : ℕ, a * b * c = m * m) :
  ∃ x y z : ℕ, a = x * x ∧ b = y * y ∧ c = z * z := 
sorry

end coprime_lcm_product_square_implies_perfect_squares_l49_49192


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49574

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49574


namespace find_l2_l49_49162

-- Define the curve
def curve (x : ℝ) : ℝ := x^2 + x - 2

-- Define line l_1
def l1 (x : ℝ) : ℝ := x - 2

-- Define that l1 is tangent to the curve at (0, -2)
axiom tangent_l1_at_0 : l1 0 = curve 0 ∧ (∃ (x : ℝ), (l1' x) = (curve' x))

-- Define that l1 is perpendicular to l2
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Define line l2 (we will find its equation)
def l2 (b x : ℝ) : ℝ := (2*b + 1)*x - b^2 - 2

-- Main theorem that we want to prove: l2 has the equation x + y + 3 = 0
theorem find_l2 : (∀ (b : ℝ), perpendicular (l1' 0) (2*b+1) → ∃ (c : ℝ), l2 b c = 0) :=
sorry

end find_l2_l49_49162


namespace expression_simplification_l49_49955

open Real

theorem expression_simplification (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : 3*x + y / 3 ≠ 0) :
  (3*x + y/3)⁻¹ * ((3*x)⁻¹ + (y/3)⁻¹) = 1 / (3 * (x * y)) :=
by
  -- proof steps would go here
  sorry

end expression_simplification_l49_49955


namespace no_discount_on_backpacks_l49_49941

def cost_monogrammed (num_backpacks : ℕ) (cost_per_backpack : ℕ) : ℕ :=
  num_backpacks * cost_per_backpack

def total_cost (num_backpacks : ℕ) (cost_per_backpack : ℕ) (total : ℕ) : ℕ :=
  total - cost_monogrammed(num_backpacks, cost_per_backpack)

theorem no_discount_on_backpacks :
  ∀ (num_backpacks : ℕ) (cost_per_backpack : ℕ) (total : ℕ),
    num_backpacks = 5 →
    cost_per_backpack = 12 →
    total = 140 →
    total_cost num_backpacks cost_per_backpack(total) = total - 60 :=
by
  intros
  sorry

end no_discount_on_backpacks_l49_49941


namespace candy_bar_cost_l49_49790

def cost_soft_drink : ℕ := 2
def num_candy_bars : ℕ := 5
def total_spent : ℕ := 27
def cost_per_candy_bar (C : ℕ) : Prop := cost_soft_drink + num_candy_bars * C = total_spent

-- The theorem we want to prove
theorem candy_bar_cost (C : ℕ) (h : cost_per_candy_bar C) : C = 5 :=
by sorry

end candy_bar_cost_l49_49790


namespace four_digit_numbers_count_l49_49249

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49249


namespace cos_B_value_l49_49363

theorem cos_B_value 
  (a b c : ℝ) 
  (h_b : b = Real.sqrt (a * c))
  (h_sin : 2 * Real.sin (Real.angle.A) = Real.sin (Real.angle.B - Real.angle.A) + Real.sin (Real.angle.C)) :
  Real.cos (Real.angle.B) = (Real.sqrt 5 - 1) / 2 :=
sorry

end cos_B_value_l49_49363


namespace area_of_45_45_90_triangle_l49_49459

theorem area_of_45_45_90_triangle (h : hypotenuse = 10 * sqrt 2) (θ : angle = 45) :
  area = 50 := 
sorry

end area_of_45_45_90_triangle_l49_49459


namespace sum_of_odd_powers_l49_49807

theorem sum_of_odd_powers (ω : ℂ) (h : ω = complex.exp (real.pi * complex.I / 15)) :
  (finset.range 15).sum (λ k, ω ^ (2 * k + 1)) = 0 :=
sorry

end sum_of_odd_powers_l49_49807


namespace greatest_product_digit_count_l49_49673

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49673


namespace count_four_digit_numbers_l49_49308

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49308


namespace greatest_product_digits_l49_49665

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49665


namespace g_invertibility_l49_49440

noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

def g_values : ∀ x : ℝ, (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5) → g x = if x = 1 then 3 else if x = 2 then 5 else if x = 3 then 6 else if x = 4 then 8 else 9 := sorry
def g_inv_values : ∀ y : ℝ, (y = 3 ∨ y = 5 ∨ y = 6 ∨ y = 8 ∨ y = 9) → g_inv y = if y = 3 then 1 else if y = 5 then 2 else if y = 6 then 3 else sorry := sorry

theorem g_invertibility : ∀ x : ℝ, g (g_inv x) = x ∧ g_inv (g x) = x := sorry

example : g(g(2)) + g(g_inv(3)) + g_inv(g_inv(6)) = 13 := by {
  have h1: g(g(2)) = 9, { sorry },
  have h2: g(g_inv(3)) = 3, { sorry },
  have h3: g_inv(g_inv(6)) = 1, { sorry },
  calc
    g(g(2)) + g(g_inv(3)) + g_inv(g_inv(6)) = 9 + 3 + 1 : by rw [h1, h2, h3]
                                ... = 13 : by norm_num
}

end g_invertibility_l49_49440


namespace max_digits_of_product_l49_49649

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49649


namespace trajectory_of_M_l49_49021

-- Define the conditions: P moves on the circle, and Q is fixed
variable (P Q M : ℝ × ℝ)
variable (P_moves_on_circle : P.1^2 + P.2^2 = 1)
variable (Q_fixed : Q = (3, 0))
variable (M_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2))

-- Theorem statement
theorem trajectory_of_M :
  (2 * M.1 - 3)^2 + 4 * M.2^2 = 1 :=
sorry

end trajectory_of_M_l49_49021


namespace greatest_number_of_digits_in_product_l49_49687

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49687


namespace four_digit_numbers_count_eq_l49_49259

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49259


namespace collinear_proj_iff_lies_on_circumcircle_l49_49041

noncomputable def collinear_proj (A B C P: Type) (proj: (Type) -> Type) : Prop :=
  let P1 := proj BC
  let P2 := proj CA
  let P3 := proj AB
  collinear P1 P2 P3 ↔ lies_on_circumcircle P (triangle A B C)

-- Definitions for collinear and lies_on_circumcircle need to be explicitly defined based on mathematical properties and usage in specific conditions of the problem.
-- However, they are abstracted here for the sake of focusing on translating the question.

-- Example definitions (simplified and need concrete mathematical properties in real scenario):

def collinear (P1 P2 P3 : Type) : Prop := sorry -- Placeholder for collinearity definition
def lies_on_circumcircle (P : Type) (T : Type) : Prop := sorry -- Placeholder for circumcircle property definition

-- The final theorem statement:
theorem collinear_proj_iff_lies_on_circumcircle (A B C P: Type) (proj: (Type) -> Type) :
  collinear_proj A B C P proj := sorry

end collinear_proj_iff_lies_on_circumcircle_l49_49041


namespace min_distance_PQ_l49_49840

-- Define the vertices of the regular tetrahedron
noncomputable def A := (1, 1, 1 : ℝ)
noncomputable def B := (1, -1, -1 : ℝ)
noncomputable def C := (-1, 1, -1 : ℝ)
noncomputable def D := (-1, -1, 1 : ℝ)

-- P is the midpoint of edge AB
noncomputable def P := (1, 0, 0 : ℝ)

-- Q is a point on edge CD such that the distance CQ = t, where 0 ≤ t ≤ 2
noncomputable def Q (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 2) := (0, 0, t : ℝ)

-- Calculate the distance between P and Q
noncomputable def distance_PQ (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 2) :=
  real.sqrt ((1 - 0)^2 + (0 - 0)^2 + (0 - t)^2)

-- The proof statement
theorem min_distance_PQ : (∀ t : ℝ, 0 ≤ t ∧ t ≤ 2 → distance_PQ t ⟨0, 2⟩ ≥ 1) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 ∧ distance_PQ t ⟨0, 2⟩ = 1) :=
by
  sorry

end min_distance_PQ_l49_49840


namespace max_digits_in_product_l49_49503

theorem max_digits_in_product : ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) ∧ (1000 ≤ b ∧ b < 10000) → 
  ∃ k, nat.log10 (a * b) + 1 = 9 :=
begin
  intros a b h,
  use 9,
  sorry
end

end max_digits_in_product_l49_49503


namespace four_digit_numbers_count_eq_l49_49257

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49257


namespace sum_first_5_terms_arithmetic_sequence_l49_49989

theorem sum_first_5_terms_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a n = a 0 + n * d)
  (h_a2 : a 2 = 1)
  (h_a4 : a 4 = 7) : 
  let S_5 := (5 / 2) * (a 0 + a 4) in
  S_5 = 20 := 
by 
  intro S_5
  have h := h_arith_seq
  sorry

end sum_first_5_terms_arithmetic_sequence_l49_49989


namespace number_of_four_digit_numbers_l49_49304

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49304


namespace greatest_number_of_digits_in_product_l49_49611

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49611


namespace dave_deleted_17_apps_l49_49097

-- Define the initial and final state of Dave's apps
def initial_apps : Nat := 10
def added_apps : Nat := 11
def apps_left : Nat := 4

-- The total number of apps before deletion
def total_apps : Nat := initial_apps + added_apps

-- The expected number of deleted apps
def deleted_apps : Nat := total_apps - apps_left

-- The proof statement
theorem dave_deleted_17_apps : deleted_apps = 17 := by
  -- detailed steps are not required
  sorry

end dave_deleted_17_apps_l49_49097


namespace plane_through_A_perpendicular_to_BC_l49_49026

-- Define points A, B, and C
def A : ℝ × ℝ × ℝ := (0, -3, 5)
def B : ℝ × ℝ × ℝ := (-7, 0, 6)
def C : ℝ × ℝ × ℝ := (-3, 2, 4)

-- Define the vector BC 
def BC : ℝ × ℝ × ℝ := (C.1 - B.1, C.2 - B.2, C.3 - B.3)

-- Define the equation of the plane
def plane_eq (x y z : ℝ) : ℝ := 2 * x - z + 5

-- Lean theorem statement
theorem plane_through_A_perpendicular_to_BC : 
  ∃ d, plane_eq 0 (-3) 5 = 0 :=
by
  let A := (0, -3, 5)
  let normal_vector := BC
  let d := 10
  exists d
  sorry

end plane_through_A_perpendicular_to_BC_l49_49026


namespace min_time_to_shoe_horses_l49_49739

-- Definitions based on the conditions
def n_blacksmiths : ℕ := 48
def n_horses : ℕ := 60
def t_hoof : ℕ := 5 -- minutes per hoof
def n_hooves : ℕ := n_horses * 4
def total_time : ℕ := n_hooves * t_hoof
def t_min : ℕ := total_time / n_blacksmiths

-- The theorem states that the minimal time required is 25 minutes
theorem min_time_to_shoe_horses : t_min = 25 := by
  sorry

end min_time_to_shoe_horses_l49_49739


namespace greatest_possible_digits_in_product_l49_49512

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49512


namespace count_four_digit_numbers_l49_49314

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49314


namespace both_pumps_drain_lake_l49_49419

theorem both_pumps_drain_lake (T : ℝ) (h₁ : 1 / 9 + 1 / 6 = 5 / 18) : 
  (5 / 18) * T = 1 → T = 18 / 5 := sorry

end both_pumps_drain_lake_l49_49419


namespace four_digit_numbers_count_l49_49245

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49245


namespace bronze_medals_l49_49093

theorem bronze_medals (G S B : ℕ) 
  (h1 : G + S + B = 89) 
  (h2 : G + S = 4 * B - 6) :
  B = 19 :=
sorry

end bronze_medals_l49_49093


namespace sequence_stabilization_l49_49391

open Set

theorem sequence_stabilization (A₀ : List ℤ) : ∃ m : ℕ, 
  let rec successor (A : List ℤ) : List ℤ :=
    match A with
    | [] => []
    | (a :: as) =>
      let l := as.length;
      let predecessors := List.filter (λ x => x < a) as;
      let successors := List.filter (λ x => x > a) as;
      (predecessors.length - successors.length) :: successor as
  in ∃ m : ℕ, (Nat.repeat (successor) m A₀) = ((Nat.repeat (successor) m A₀).tail ++ [(Nat.repeat (successor) m A₀).lastD 0])

end sequence_stabilization_l49_49391


namespace min_transfers_to_uniform_cards_l49_49734

theorem min_transfers_to_uniform_cards (n : ℕ) (h : n = 101) (s : Fin n) :
  ∃ k : ℕ, (∀ s1 s2 : Fin n → ℕ, 
    (∀ i, s1 i = i + 1) ∧ (∀ j, s2 j = 51) → -- Initial and final conditions
    k ≤ 42925) := 
sorry

end min_transfers_to_uniform_cards_l49_49734


namespace focus_with_greatest_y_coordinate_l49_49789

-- Define the conditions as hypotheses
def ellipse_major_axis : (ℝ × ℝ) := (0, 3)
def ellipse_minor_axis : (ℝ × ℝ) := (2, 0)
def ellipse_semi_major_axis : ℝ := 3
def ellipse_semi_minor_axis : ℝ := 2

-- Define the theorem to compute the coordinates of the focus with the greater y-coordinate
theorem focus_with_greatest_y_coordinate :
  let a := ellipse_semi_major_axis
  let b := ellipse_semi_minor_axis
  let c := Real.sqrt (a^2 - b^2)
  (0, c) = (0, (Real.sqrt 5) / 2) :=
by
  -- skipped proof
  sorry

end focus_with_greatest_y_coordinate_l49_49789


namespace number_of_four_digit_numbers_l49_49302

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49302


namespace worst_player_is_niece_l49_49784

structure Player where
  name : String
  sex : String
  generation : Nat

def grandmother := Player.mk "Grandmother" "Female" 1
def niece := Player.mk "Niece" "Female" 2
def grandson := Player.mk "Grandson" "Male" 3
def son_in_law := Player.mk "Son-in-law" "Male" 2

def worst_player : Player := niece
def best_player : Player := grandmother

-- Conditions
def cousin_check : worst_player ≠ best_player ∧
                   worst_player.generation ≠ best_player.generation ∧ 
                   worst_player.sex ≠ best_player.sex := 
  by sorry

-- Prove that the worst player is the niece
theorem worst_player_is_niece : worst_player = niece :=
  by sorry

end worst_player_is_niece_l49_49784


namespace max_digits_in_product_l49_49609

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49609


namespace find_number_l49_49338

noncomputable def op (x w : ℕ) : ℚ := (2^x) / (2^w)

theorem find_number :
  ∃ n : ℕ, op (op 4 2) n = 8 :=
begin
  use 1,
  simp [op],
  linarith,
end

example : find_number := by tidy

end find_number_l49_49338


namespace parabola_solution_l49_49158

noncomputable def parabolaProblem : Prop :=
  ∃ (a : ℝ) (A B C D : (ℝ × ℝ)) (k1 k2 : ℝ),
    (a > 0) ∧
    (let y1 := A.2,
         y2 := B.2,
         y3 := C.2,
         y4 := D.2 in
     (k1 = (A.2 - B.2) / (A.1 - B.1)) ∧
     (k2 = (a / (C.2 + D.2))) ∧
     (k1 = √2 * k2) ∧
     (y1 + y2 = (√2 / 2) * (y3 + y4)) ∧
     (y1 * y3 = -4 * a) ∧
     (y2 * y4 = -4 * a) ∧
     (y1 * y2 = -2 * √2 * a) ∧
     (y1 * y2 = -(a^2) / 4) ∧
     (a = 8 * √2))

theorem parabola_solution : parabolaProblem :=
sorry

end parabola_solution_l49_49158


namespace transformed_data_mean_and_variance_l49_49152

variable (x1 x2 x3 x4 : ℝ)
variable (x_avg : ℝ := (x1 + x2 + x3 + x4) / 4)
variable (s_squared : ℝ := ((x1 - x_avg)^2 + (x2 - x_avg)^2 + (x3 - x_avg)^2 + (x4 - x_avg)^2) / 4)

theorem transformed_data_mean_and_variance 
    (h1 : x_avg = 1)
    (h2 : s_squared = 1) :
    let y1 := 2 * x1 + 1
    let y2 := 2 * x2 + 1
    let y3 := 2 * x3 + 1
    let y4 := 2 * x4 + 1
    let y_avg := (y1 + y2 + y3 + y4) / 4
    let y_variance := ((y1 - y_avg)^2 + (y2 - y_avg)^2 + (y3 - y_avg)^2 + (y4 - y_avg)^2) / 4
    in y_avg = 3 ∧ y_variance = 4 :=
by
  sorry

end transformed_data_mean_and_variance_l49_49152


namespace evaluate_expression_l49_49108

variable (a b : ℝ)

theorem evaluate_expression : a = 5 → b = -3 → 3 / (a + b) = 3 / 2 :=
by
  intro ha hb
  rw [ha, hb]
  simp
  sorry

end evaluate_expression_l49_49108


namespace domain_of_f_l49_49100

-- Define the function for which we are finding the domain
def f (x : ℝ) : ℝ := sqrt (cos x - 1/2)

-- Define the predicate that checks if the function is defined and non-negative
def is_defined (x : ℝ) : Prop := cos x - 1/2 ≥ 0

-- State the theorem that determines the domain given the predicate
theorem domain_of_f :
  { x : ℝ | is_defined x } = ⋃ (k : ℤ), Icc (2 * k * real.pi - real.pi / 3) (2 * k * real.pi + real.pi / 3) :=
sorry

end domain_of_f_l49_49100


namespace sum_of_smallest_and_largest_prime_l49_49367

def primes_between (a b : ℕ) : List ℕ := List.filter Nat.Prime (List.range' a (b - a + 1))

def smallest_prime_in_range (a b : ℕ) : ℕ :=
  match primes_between a b with
  | [] => 0
  | h::t => h

def largest_prime_in_range (a b : ℕ) : ℕ :=
  match List.reverse (primes_between a b) with
  | [] => 0
  | h::t => h

theorem sum_of_smallest_and_largest_prime : smallest_prime_in_range 1 50 + largest_prime_in_range 1 50 = 49 := 
by
  -- Let the Lean prover take over from here
  sorry

end sum_of_smallest_and_largest_prime_l49_49367


namespace distance_between_locations_l49_49972

theorem distance_between_locations (speedA speedB : ℝ) (time : ℝ) (gap : ℝ) (distance : ℝ) :
  speedA = 50 ∧ speedB = (4 / 5) * speedA ∧ time = 2 ∧ gap = 20 ∧ 
  distance = 2 * speedA + 2 * speedB + gap → distance = 200 :=
by {
  intros h,
  cases h with hA h_rest,
  cases h_rest with hB h_rest,
  cases h_rest with hT h_rest,
  cases h_rest with hG hD,
  rw [hA, hB, hT, hG, hD],
  norm_num,
  sorry
}

end distance_between_locations_l49_49972


namespace number_of_four_digit_numbers_l49_49274

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49274


namespace students_failed_l49_49470

theorem students_failed (total_students : ℕ) (percent_A : ℚ) (fraction_BC : ℚ) (students_A : ℕ)
  (students_remaining : ℕ) (students_BC : ℕ) (students_failed : ℕ)
  (h1 : total_students = 32) (h2 : percent_A = 0.25) (h3 : fraction_BC = 0.25)
  (h4 : students_A = total_students * percent_A)
  (h5 : students_remaining = total_students - students_A)
  (h6 : students_BC = students_remaining * fraction_BC)
  (h7 : students_failed = total_students - students_A - students_BC) :
  students_failed = 18 :=
sorry

end students_failed_l49_49470


namespace ratio_of_coefficients_l49_49335

theorem ratio_of_coefficients (a b c : ℝ) (h : ∀ x : ℝ, ax^2 + bx + c = (x-1)*(2x+1)) : a = 2 ∧ b = -1 :=
by
  sorry

example (a b : ℝ) (h : a = 2 ∧ b = -1) : a / b = 2 / -1 :=
by
  cases h
  rw [h_left, h_right]
  exact rfl

end ratio_of_coefficients_l49_49335


namespace greatest_product_digit_count_l49_49672

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49672


namespace find_a_b_c_f_monotonicity_f_min_value_f_l49_49921

variable {R : Type*} [LinearOrderedField R]

def quadratic (a b c : R) (x : R) : R := a * x^2 + b * x + c

theorem find_a_b_c_f {a b c : R} (h_odd : ∀ x : R, quadratic a b c (-x) = -quadratic a b c x)
  (h1 : quadratic a b c 1 = b + c) (h2 : quadratic a b c 2 = 4 * a + 2 * b + c) :
  a = 2 ∧ b = -3 ∧ c = 0 :=
by
  sorry

theorem monotonicity_f (a b c : R) (h_odd : ∀ x : R, quadratic a b c (-x) = -quadratic a b c x)
  (h1 : quadratic a b c 1 = b + c) (h2 : quadratic a b c 2 = 4 * a + 2 * b + c) 
  (hf : ∀ x : R, a = 2 ∧ b = -3 ∧ c = 0) :
  ∀ x : R, 0 < x → fderiv ℝ (quadratic a b c) x < 0 :=
by
  sorry

theorem min_value_f (a b c : R)
  (h_odd : ∀ x : R, quadratic a b c (-x) = -quadratic a b c x)
  (h1 : quadratic a b c 1 = b + c) (h2 : quadratic a b c 2 = 4 * a + 2 * b + c) 
  (hf : ∀ x : R, a = 2 ∧ b = -3 ∧ c = 0) :
  ∃ x : R, 0 < x ∧ quadratic a b c x = 2 :=
by
  sorry

end find_a_b_c_f_monotonicity_f_min_value_f_l49_49921


namespace number_of_four_digit_numbers_l49_49277

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49277


namespace greatest_number_of_digits_in_product_l49_49688

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49688


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49631

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49631


namespace problem_1_problem_2_l49_49930

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := x * (Real.log x - a)

-- Derivative of the function
def f_prime (x a : ℝ) : ℝ := Real.log x + 1 - a

theorem problem_1 {a : ℝ} : 
  (∀ x ∈ Set.Icc 1 4, f_prime x a ≥ 0) → a ≤ 1 := 
sorry

theorem problem_2 {a x : ℝ} (ha : 0 < a) : 
  f x a ≤ x * (x - 2 - Real.log a) := 
sorry

end problem_1_problem_2_l49_49930


namespace linear_eq_represents_plane_l49_49987

theorem linear_eq_represents_plane (A B C : ℝ) (h : ¬ (A = 0 ∧ B = 0 ∧ C = 0)) :
  ∃ (P : ℝ × ℝ × ℝ → Prop), (∀ (x y z : ℝ), P (x, y, z) ↔ A * x + B * y + C * z = 0) ∧ 
  (P (0, 0, 0)) :=
by
  -- To be filled in with the proof steps
  sorry

end linear_eq_represents_plane_l49_49987


namespace emily_dice_probability_l49_49858

theorem emily_dice_probability :
  let prob_first_die := 1 / 2
  let prob_second_die := 3 / 8
  prob_first_die * prob_second_die = 3 / 16 :=
by
  sorry

end emily_dice_probability_l49_49858


namespace four_digit_number_count_l49_49201

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49201


namespace four_digit_number_count_l49_49198

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49198


namespace four_digit_numbers_count_l49_49254

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49254


namespace arctan_sum_pi_div_two_l49_49825

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l49_49825


namespace find_angle_A_l49_49907

theorem find_angle_A (a b : ℝ) (A B : ℝ) 
  (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 3) (hB : B = Real.pi / 3) :
  A = Real.pi / 4 :=
by
  -- This is a placeholder for the proof
  sorry

end find_angle_A_l49_49907


namespace probability_same_two_subjects_l49_49357

theorem probability_same_two_subjects (A B : Finset String) :
  (A.card = 3 ∧ B.card = 3) ∧
  (∃ x1 x2 : String, x1 ∈ A ∧ x2 ∈ A ∧ x1 ∈ B ∧ x2 ∈ B ∧
   ∀ x ∈ A, (x ∈ {'Physics', 'History', 'Chemistry', 'Biology', 'Geography', 'Politics'})) ∧
  (∀ x ∈ B, (x ∈ {'Physics', 'History', 'Chemistry', 'Biology', 'Geography', 'Politics'})) →
  (∃ prob : ℚ, prob = 5 / 12) :=
by 
  sorry

end probability_same_two_subjects_l49_49357


namespace num_positive_integers_prime_polynomial_l49_49885

def is_prime (p : ℤ) : Prop :=
  p > 1 ∧ ∀ n, n > 1 → n < p → (p % n ≠ 0)

theorem num_positive_integers_prime_polynomial : ∃ (n1 n2 n3 : ℕ), 
  (n1 = 2 ∧ n2 = 3 ∧ n3 = 4) ∧ 
  ∀ n, (n = n1 ∨ n = n2 ∨ n = n3) → is_prime (n^3 - 8 * n^2 + 20 * n - 13) 
  ∧ ∀ m, m > 0 → m ≠ n1 ∧ m ≠ n2 ∧ m ≠ n3 → ¬ is_prime (m^3 - 8 * m^2 + 20 * m - 13) :=
begin
  sorry
end

end num_positive_integers_prime_polynomial_l49_49885


namespace greatest_product_digit_count_l49_49680

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49680


namespace yeast_cells_at_10_30_l49_49430

def yeast_population (initial_population : ℕ) (intervals : ℕ) (growth_rate : ℝ) (decay_rate : ℝ) : ℝ :=
  initial_population * (growth_rate * (1 - decay_rate)) ^ intervals

theorem yeast_cells_at_10_30 :
  yeast_population 50 6 3 0.10 = 52493 := by
  sorry

end yeast_cells_at_10_30_l49_49430


namespace greatest_number_of_digits_in_product_l49_49612

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49612


namespace locus_points_triangle_angles_l49_49905

-- Supposing we have a triangle ABC
variables {P A B C : Type*}
variable [metric_space P]

-- Assume three points A, B, C form a triangle
variable (A B C : P)
#check ∠

-- Definition of Thales circle for a side of a triangle
def Thales_circle (X Y : P) : set P := sorry -- Describes the set of points creating a Thales' circle with diameter XY

-- Define the condition under which the sum of angles by the sides subtended equals 180 degrees
def angle_sum_180 (P : P) := 
  ∠P A B + ∠P B C + ∠P C A = 180

-- Lean proof statement to verify the locus of points satisfying the angle condition 
theorem locus_points_triangle_angles(A B C : P) :
  { P : P | angle_sum_180 P A B C } = 
  (Thales_circle A B \ {A, B}) ∪ (Thales_circle B C \ {B, C}) ∪ (Thales_circle C A \ {C, A}) :=
sorry

end locus_points_triangle_angles_l49_49905


namespace jimmy_shoveled_10_driveways_l49_49370

theorem jimmy_shoveled_10_driveways :
  ∀ (cost_candy_bar : ℝ) (num_candy_bars : ℕ)
    (cost_lollipop : ℝ) (num_lollipops : ℕ)
    (fraction_spent : ℝ)
    (charge_per_driveway : ℝ),
    cost_candy_bar = 0.75 →
    num_candy_bars = 2 →
    cost_lollipop = 0.25 →
    num_lollipops = 4 →
    fraction_spent = 1/6 →
    charge_per_driveway = 1.5 →
    let total_spent := (num_candy_bars * cost_candy_bar + num_lollipops * cost_lollipop) in
    let total_earned := total_spent / fraction_spent in
    (total_earned / charge_per_driveway) = 10 := sorry

end jimmy_shoveled_10_driveways_l49_49370


namespace range_of_cos_neg_alpha_l49_49326

theorem range_of_cos_neg_alpha (α : ℝ) (h : 12 * (Real.sin α)^2 + Real.cos α > 11) :
  -1 / 4 < Real.cos (-α) ∧ Real.cos (-α) < 1 / 3 := 
sorry

end range_of_cos_neg_alpha_l49_49326


namespace angle_RQP_is_90_l49_49993

noncomputable def x : ℝ := 18
noncomputable def P := ℝ
noncomputable def Q := ℝ
noncomputable def R := ℝ
noncomputable def S := ℝ

-- Conditions
axiom on_segment_RS (P : ℝ) (R : ℝ) (S : ℝ) : True
axiom perp_bisector_QP_SRQ (P Q R S : ℝ) : angle Q P R = 90
axiom PQ_eq_PR (P Q R : ℝ) : dist P Q = dist P R
axiom angle_RSQ (R S Q : ℝ) (x : ℝ) : angle R S Q = 3 * x
axiom angle_PQS (P Q S : ℝ) (x : ℝ) : angle P Q S = 4 * x

-- Goal
theorem angle_RQP_is_90 (P Q R S : ℝ) (x : ℝ) (h1 : True) (h2 : angle Q P R = 90) 
  (h3 : dist P Q = dist P R) (h4 : angle R S Q = 3 * x) (h5 : angle P Q S = 4 * x) : 
  angle R Q P = 90 :=
by
  -- Proof goes here
  sorry

end angle_RQP_is_90_l49_49993


namespace greatest_possible_digits_in_product_l49_49509

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49509


namespace polar_equation_eccentricity_l49_49449

theorem polar_equation_eccentricity :
  ∃ e : ℝ, e = √2 ∧ (∀ ρ θ : ℝ, (ρ^2 * Real.cos (2 * θ)) = 1 ↔ ((ρ * Real.sin θ)^2 - (ρ * Real.cos θ)^2 = 1)) := 
sorry

end polar_equation_eccentricity_l49_49449


namespace four_digit_numbers_count_l49_49214

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49214


namespace greatest_product_digit_count_l49_49683

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49683


namespace rational_expression_iff_rational_root_l49_49104

theorem rational_expression_iff_rational_root (x : ℝ) : 
  (∃ k : ℚ, 2 * x + real.sqrt (x ^ 2 + 1) - 1 / (2 * x + real.sqrt (x ^ 2 + 1)) = k) 
  ↔ ∃ r : ℚ, real.sqrt (x ^ 2 + 1) = r :=
sorry

end rational_expression_iff_rational_root_l49_49104


namespace max_digits_of_product_l49_49538

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49538


namespace unique_planes_through_P_l49_49082

-- Define the given conditions
variables {P : Point} (l1 l2 l3 : Line)
-- Assume the lines pass through point P
variable (lines_through_P : ∀ l, l ∈ {l1, l2, l3} → passes_through l P)
-- Assume the lines are not coplanar
axiom not_coplanar_lines : ¬Coplanar l1 l2 l3

-- Define the statement to be proved
theorem unique_planes_through_P (h : ∀ l, l ∈ {l1, l2, l3} → passes_through l P) : 
  ∃ (planes : set Plane), 
    (∀ p ∈ planes, passes_through p P ∧ ∀ l ∈ {l1, l2, l3}, forms_same_angle_with l p) ∧ 
    planes_finite_and_unique planes 4 :=
sorry

end unique_planes_through_P_l49_49082


namespace cube_dimension_l49_49017

theorem cube_dimension (x s : ℝ) (hx1 : s^3 = 8 * x) (hx2 : 6 * s^2 = 2 * x) : x = 1728 := 
by {
  sorry
}

end cube_dimension_l49_49017


namespace greatest_product_digit_count_l49_49681

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49681


namespace max_digits_of_product_l49_49542

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49542


namespace min_distance_sum_square_l49_49426

-- Define the distance function between two points in the plane
def dist (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Vertices of the square
def O := (0, 0)
def A := (1, 0)
def B := (1, 1)
def C := (0, 1)

-- Statement of the problem
theorem min_distance_sum_square
  (P : ℝ × ℝ) -- Any point inside the square
  (hP_in_square : 0 ≤ P.1 ∧ P.1 ≤ 1 ∧ 0 ≤ P.2 ∧ P.2 ≤ 1) :
  dist P.1 P.2 0 0 + dist P.1 P.2 1 0 + dist P.1 P.2 1 1 + dist P.1 P.2 0 1 = 2 * Real.sqrt 2 :=
sorry

end min_distance_sum_square_l49_49426


namespace price_reduction_for_2100_yuan_price_reduction_for_max_profit_l49_49054

-- Condition definitions based on the problem statement
def units_sold (x : ℝ) : ℝ := 30 + 2 * x
def profit_per_unit (x : ℝ) : ℝ := 50 - x
def daily_profit (x : ℝ) : ℝ := profit_per_unit x * units_sold x

-- Statement to prove the price reduction for achieving a daily profit of 2100 yuan
theorem price_reduction_for_2100_yuan : ∃ x : ℝ, daily_profit x = 2100 ∧ x = 20 :=
  sorry

-- Statement to prove the price reduction to maximize the daily profit
theorem price_reduction_for_max_profit : ∀ x : ℝ, ∃ y : ℝ, (∀ z : ℝ, daily_profit z ≤ y) ∧ x = 17.5 :=
  sorry

end price_reduction_for_2100_yuan_price_reduction_for_max_profit_l49_49054


namespace bobby_toy_cars_in_7_years_l49_49088

/-- Bobby's toy cars after seven years -/
def bobbyToyCars : ℕ :=
  let initialCars := 30
  let years := 7
  let rec calcCars (cars : ℕ) (year : ℕ) : ℕ :=
    if year = years then cars
    else if year % 2 = 0 then calcCars (cars * 2 * 9 / 10) (year + 1)
    else calcCars (cars * 2) (year + 1)
  calcCars initialCars 0

theorem bobby_toy_cars_in_7_years :
  bobbyToyCars = 2792 := by
  sorry

end bobby_toy_cars_in_7_years_l49_49088


namespace minimum_value_of_m_l49_49180

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x

theorem minimum_value_of_m :
  (∀ x₁ x₂ ∈ Icc (-3 : ℝ) (2 : ℝ), |f x₁ - f x₂| ≤ (81/4 : ℝ)) ↔ (81/4 : ℝ) =
  (18 - (-9/4 : ℝ)) := 
sorry

end minimum_value_of_m_l49_49180


namespace order_of_a_b_c_l49_49917

noncomputable def a : Real := logBase 3 (2 : Real)
noncomputable def b : Real := Real.log 2
noncomputable def c : Real := 5 ^ (-0.5 : Real)

theorem order_of_a_b_c : b > a ∧ a > c :=
by
  -- Declaration of the conditions for a, b, and c
  let a_def : a = logBase 3 (2 : Real) := rfl
  let b_def : b = Real.log 2 := rfl
  let c_def : c = 5 ^ (-0.5 : Real) := rfl
  sorry

end order_of_a_b_c_l49_49917


namespace factorization_of_2210_l49_49321

theorem factorization_of_2210 : 
  ∃! (a b : ℕ), (10 ≤ a ∧ a < 100) ∧ (10 ≤ b ∧ b < 100) ∧ (a * b = 2210) :=
sorry

end factorization_of_2210_l49_49321


namespace probability_smaller_area_l49_49765

structure Triangle (α : Type) [OrderedRing α] :=
(A B C : α × α)
(is_isosceles : A.snd = B.snd ∧ A.snd = C.snd)
(med_AD : ∃ D : α × α, (D = (B + C) / 2) ∧ A.fst = D.fst)
(centroid_O : ∃ O : α × α, (O = (2/A + D) / 3) ∧ A.fst = O.fst)

theorem probability_smaller_area {α : Type} [OrderedRing α] (t : Triangle α) :
  P (point_in_t.has_smaller_area t) = 1 / 6 :=
sorry

end probability_smaller_area_l49_49765


namespace smallest_positive_m_condition_l49_49004

theorem smallest_positive_m_condition
  (p q : ℤ) (m : ℤ) (h_prod : p * q = 42) (h_diff : |p - q| ≤ 10) 
  (h_roots : 15 * (p + q) = m) : m = 195 :=
sorry

end smallest_positive_m_condition_l49_49004


namespace area_triangle_CYP_l49_49991

variables (AX WD AB Area : ℝ)
variables (X Y W C P : ℝ)

-- Defining the conditions
def is_rectangle (AX WD AB : ℝ) (Area : ℝ) : Prop :=
  AX = 4 ∧ WD = 4 ∧ AB = 16 ∧ Area = 136

def is_parallel (XY CD : ℝ) : Prop :=
  XY = CD

def is_midpoint (X W P : ℝ) : Prop :=
  P = (X + W) / 2

axiom area_trapezoid (AX WD AB Area : ℝ) (XY CD : ℝ) : is_rectangle AX WD AB Area → is_parallel XY CD → Area

theorem area_triangle_CYP (AX WD AB Area : ℝ) (XY CD X W C P : ℝ) :
  is_rectangle AX WD AB Area → is_parallel XY CD → is_midpoint X W P → 
  Area = 136 → AX = 4 → WD = 4 → AB = 16 → XY = CD → P = (X + W) / 2 → 
  2 * (Area / 2) / 2 = 68 :=
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9,
  rw h4, rw h9, 
  sorry,
end

end area_triangle_CYP_l49_49991


namespace find_x_perpendicular_l49_49142

noncomputable def a : ℝ × ℝ := (-1, 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, -1)
noncomputable def c (x : ℝ) : ℝ × ℝ := (2 * a.1 + b(x).1, 2 * a.2 + b(x).2)

theorem find_x_perpendicular (x : ℝ) 
  (h : let c := (2 * (-1) + x, 2 * 2 - 1) in  
       (x, -1) ≠ (0:ℝ, 0:ℝ) ∧ 
       (x * (2 * (-1) + x) + (-1) * (2 * 2 - 1) = 0))
  : x = -1 ∨ x = 3 := by
  sorry

end find_x_perpendicular_l49_49142


namespace arctan_triangle_complementary_l49_49817

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l49_49817


namespace same_terminal_side_l49_49893

variable (k : ℤ)

theorem same_terminal_side (k : ℤ) : 
  ((2 * k + 1) * 180 ≡ (4 * k + 1) * 180 [MOD 360]) ∨ ((2 * k + 1) * 180 ≡ (4 * k - 1) * 180 [MOD 360]) := by
  sorry

end same_terminal_side_l49_49893


namespace smallest_k_is_40_l49_49422

def cell := ℕ × ℕ

def is_marked (board : cell → Prop) (c : cell) := board c

def can_determine_position (board : cell → Prop) (rect_positions : set (cell × cell × cell × cell)) : Prop :=
  ∀ pos1 pos2 ∈ rect_positions, (is_marked board pos1.fst ∨ is_marked board pos1.snd ∨ 
                                 is_marked board pos1.fst.snd ∨ is_marked board pos1.snd.snd) → 
                                 (is_marked board pos2.fst ∨ is_marked board pos2.snd ∨ 
                                 is_marked board pos2.fst.snd ∨ is_marked board pos2.snd.snd) → 
                                 pos1 = pos2

def min_marks (k : ℕ) : Prop :=
  ∃ board : cell → Prop, (∀ i j, i < 9 ∧ j < 9 → is_marked board (i, j) ∨ ¬is_marked board (i, j)) ∧
  can_determine_position board {((i, j), (i, j + 3), (i + 3, j), (i + 3, j + 3)) | i < 6 ∧ j < 6} ∧ 
  (∑ i j, if is_marked board (i, j) then 1 else 0 = k)

theorem smallest_k_is_40 : min_marks 40 :=
sorry

end smallest_k_is_40_l49_49422


namespace hypotenuse_of_triangle_PQR_l49_49978

theorem hypotenuse_of_triangle_PQR (PA PB PC QR : ℝ) (h1: PA = 2) (h2: PB = 3) (h3: PC = 2)
  (h4: PA + PB + PC = QR) (h5: QR = PA + 3 + 2 * PA): QR = 5 * Real.sqrt 2 := 
sorry

end hypotenuse_of_triangle_PQR_l49_49978


namespace zero_count_non_decreasing_zero_count_tends_to_2N_l49_49727

noncomputable def f (a : ℕ → ℝ) (x : ℝ) : ℝ :=
  ∑ k in Finset.range (Nat.succ N), a k * Real.sin (2 * k * π * x)

def number_of_zeros (f : ℝ → ℝ) (i : ℕ) (interval : Set.Ici (0 : ℝ)) : ℕ :=
  (Set.Icc (0 : ℝ) 1).count {x | deriv^[i] f x = 0}

theorem zero_count_non_decreasing (a : ℕ → ℝ) (h : ∀ k, k ≥ N → a k = 0) (h_nonzero : a N ≠ 0) 
  (i : ℕ) :
  ∀ j ≥ i, number_of_zeros (f a) j (set.Ici 0) ≥ number_of_zeros (f a) i (set.Ici 0) :=
sorry

theorem zero_count_tends_to_2N (a : ℕ → ℝ) (h : ∀ k, k ≥ N → a k = 0) (h_nonzero : a N ≠ 0) :
  limit (λ i, number_of_zeros (f a) i (set.Ici 0)) at_top = 2 * N :=
sorry

end zero_count_non_decreasing_zero_count_tends_to_2N_l49_49727


namespace greatest_possible_number_of_digits_in_product_l49_49524

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49524


namespace earnings_walking_dogs_each_month_correct_l49_49404

noncomputable def earns_walking_dogs_each_month : ℝ :=
  let earnings_washing_cars := 20
  let savings_in_5_months := 150
  let months := 5
  let savings_per_month := savings_in_5_months / months
  let total_earnings_per_month := savings_per_month * 2
  total_earnings_per_month - earnings_washing_cars

theorem earnings_walking_dogs_each_month_correct :
  earns_walking_dogs_each_month = 40 :=
by
  let earnings_washing_cars := 20
  let savings_in_5_months := 150
  let months := 5
  let savings_per_month := savings_in_5_months / months
  have step1 : savings_per_month = 30 := by sorry
  let total_earnings_per_month := savings_per_month * 2
  have step2 : total_earnings_per_month = 60 := by sorry
  have step3 : 60 - 20 = 40 := by sorry
  show earns_walking_dogs_each_month = 40 from step3

end earnings_walking_dogs_each_month_correct_l49_49404


namespace max_digits_of_product_l49_49651

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49651


namespace surface_area_of_prism_l49_49050

theorem surface_area_of_prism (l w h : ℕ)
  (h_internal_volume : l * w * h = 24)
  (h_external_volume : (l + 2) * (w + 2) * (h + 2) = 120) :
  2 * ((l + 2) * (w + 2) + (w + 2) * (h + 2) + (h + 2) * (l + 2)) = 148 :=
by
  sorry

end surface_area_of_prism_l49_49050


namespace binom_n_n_minus_2_l49_49883

variable (n : ℕ) (h : n > 3)

theorem binom_n_n_minus_2 :
  binomialCoeff n (n - 2) = (n * (n - 1) / 2) := sorry

end binom_n_n_minus_2_l49_49883


namespace area_of_sector_l49_49923

theorem area_of_sector (r : ℝ) (θ : ℝ) (h_r : r = 1) (h_θ : θ = 30) : 
  (θ * real.pi * r^2) / 360 = real.pi / 12 :=
by
  -- Assume the conditions in the theorem
  -- Rest of the proof is omitted with sorry
  sorry

end area_of_sector_l49_49923


namespace exists_tactical_partition_l49_49896

def is_tactical_set (n : ℕ) (p : ℕ) (S : finset (ℕ × ℕ)) : Prop :=
  (S.card = n) ∧
  (∀ (a b c d : ℕ), (a, b) ∈ S → (c, d) ∈ S → 
    a ≠ c ∧ b ≠ d ∧ abs (a - c) ≠ abs (b - d))

def no_main_diagonals (p : ℕ) (S : finset (ℕ × ℕ)) : Prop :=
  ∀ (a b : ℕ), (a, b) ∈ S → a ≠ b ∧ a + b ≠ p + 1

theorem exists_tactical_partition (n : ℕ) (h₁ : 4 ≤ n) (p : ℕ) (h₂ : p = n + 1) (h₃ : nat.prime p) :
  ∃ (S : finset (ℕ × ℕ) → finset (ℕ × ℕ)), 
    (∀ k ∈ (finset.range (p - 3)).map (λ k, k + 2), is_tactical_set n p (S k) ∧ no_main_diagonals p (S k)) ∧ 
    (∀ i j, i ≠ j → i ∈ (finset.range (p - 3)).map (λ k, k + 2) → j ∈ (finset.range (p - 3)).map (λ k, k + 2) → 
      disjoint (S i) (S j)) ∧ 
    (⋃ i ∈ (finset.range (p - 3)).map (λ k, k + 2), S i) = (finset.univ : finset (ℕ × ℕ)) :=
sorry

end exists_tactical_partition_l49_49896


namespace money_left_is_26_l49_49407

-- Define the allowance and the spent amount
def total_allowance : ℝ := sorry
def spent := 14
def spent_percent := 35

-- State that spent amount is 35% of the total allowance
axiom spent_percentage (total_allowance : ℝ) : spent = spent_percent * total_allowance / 100

-- Define the amount left and its correctness
def amount_left : ℝ := total_allowance * (100 - spent_percent) / 100

-- The theorem we need to prove
theorem money_left_is_26 (total_allowance : ℝ) (h : spent_percentage total_allowance) : amount_left = 26 :=
sorry

end money_left_is_26_l49_49407


namespace next_perfect_number_after_six_l49_49962

def is_perfect_number (n : ℕ) : Prop := 
  n = (Nat.divisors n).sum - n

def euclid_perfect_number (n : ℕ) : ℕ := 
  2^(n - 1) * (2^n - 1)

theorem next_perfect_number_after_six : ∃ n : ℕ, 6 < euclid_perfect_number n ∧
  is_perfect_number (euclid_perfect_number n) ∧ 
  ∀ m : ℕ, 6 < euclid_perfect_number m → euclid_perfect_number m = euclid_perfect_number n :=
by 
  exists 3
  sorry

end next_perfect_number_after_six_l49_49962


namespace interval_length_le_one_l49_49101

theorem interval_length_le_one :
  let I := {x : ℝ | x^2 - 5 * x + 6 ≤ 0}
  intervalLength I = 1 :=
by
  sorry

end interval_length_le_one_l49_49101


namespace probability_of_including_A_B_C_l49_49774

theorem probability_of_including_A_B_C :
  let seed_set := {A, B, C, D, E, F, G, H, I, J};
  let included_seeds := {A, B, C};
  let total_combinations := nat.choose 10 3;
  let excluded_combinations := nat.choose 7 3;
  (10.card = 10) →
  (3.card = 3) →
  (total_combinations - excluded_combinations) / total_combinations = 17 / 24 := 
by
  intros seed_set included_seeds total_combinations excluded_combinations h1 h2
  sorry

end probability_of_including_A_B_C_l49_49774


namespace Jack_minimum_cars_per_hour_l49_49421

theorem Jack_minimum_cars_per_hour (J : ℕ) (h1 : 2 * 8 + 8 * J ≥ 40) : J ≥ 3 :=
by {
  -- The statement of the theorem directly follows
  sorry
}

end Jack_minimum_cars_per_hour_l49_49421


namespace max_digits_in_product_l49_49602

theorem max_digits_in_product : 
  ∀ (x y : ℕ), x = 99999 → y = 9999 → (nat.digits 10 (x * y)).length = 9 := 
by
  intros x y hx hy
  rw [hx, hy]
  have h : 99999 * 9999 = 999890001 := by norm_num
  rw h
  norm_num
sorry

end max_digits_in_product_l49_49602


namespace four_digit_numbers_count_l49_49210

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49210


namespace determine_constants_l49_49401

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

variables (A B P : V)

-- Define the Section Formula for the position vector P given ratios
def section_formula (m n : ℝ) (A B : V) : V :=
  (m / (m + n)) • A + (n / (m + n)) • B

-- Express the specific case where AP:PB = 7:2
theorem determine_constants
  (h_ratio : 7 * (B - A) = 2 * (P - B)) :
  ∃ t u : ℝ, P = t • A + u • B ∧ t = 7 / 9 ∧ u = 2 / 9 :=
by
  sorry

end determine_constants_l49_49401


namespace four_digit_number_count_l49_49202

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49202


namespace greatest_possible_number_of_digits_in_product_l49_49533

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49533


namespace total_edge_length_is_least_l49_49402

-- Definitions of the conditions
def height_condition (x : ℝ) : Prop := x + 1 = x + 1
def surface_area_condition (x : ℝ) : Prop := (sqrt 3) * x^2 ≥ 100
def edge_length (x : ℝ) : ℝ := x + 1

-- Main theorem statement
theorem total_edge_length_is_least (x : ℝ) (h₁ : surface_area_condition x) (h₂ : height_condition x) :
  edge_length x = 6.75 :=
by
  sorry

end total_edge_length_is_least_l49_49402


namespace greatest_product_digits_l49_49662

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49662


namespace max_distance_polar_to_line_l49_49970

-- Definitions from conditions of the problem
def polar_eq_line (rho theta : ℝ) : Prop := rho * Real.cos (theta - (Real.pi / 4)) = 3 * Real.sqrt 2

def polar_eq_curve (rho theta : ℝ) : Prop := rho = 1

-- Main theorem for the proof problem
theorem max_distance_polar_to_line :
  let d : ℝ := _ in
  ∀ (theta : ℝ), polar_eq_curve 1 theta → polar_eq_line (1 : ℝ) theta → d = 3 * Real.sqrt 2 + 1 :=
sorry

end max_distance_polar_to_line_l49_49970


namespace find_f_e_l49_49927

noncomputable def f (x : ℝ) := 2 * x * (deriv (λ (x : ℝ), 2 * x * (deriv (λ (x : ℝ), 2 * x * deriv f e + real.log x) x) e + real.log e) e) + real.log x

theorem find_f_e (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x * deriv f e + real.log x) : f e = -1 :=
by sorry

end find_f_e_l49_49927


namespace part_a_part_b_l49_49135

-- Define the polynomial having integer root
def hasIntRoot (m n : ℤ) :=
  ∃ x : ℤ, x*x + m*x + n = 0

-- Define the set F
def F (n : ℤ) : Set ℤ :=
  {m : ℤ | m > 0 ∧ hasIntRoot m n}

-- Define the set S for part (a)
def S : Set ℕ := {n : ℕ | n > 0 ∧ ∃ m : ℤ, m ∈ F n ∧ m + 1 ∈ F n}

-- Part (a): S is infinite and ∑_{n ∈ S} 1/n ≤ 1
theorem part_a : (Set.Infinite S) ∧ (∑' (n : ℕ) in S, (1 / n : ℝ)) ≤ 1 :=
begin
  sorry
end

-- Part (b): Infinitely many positive integers n such that F(n) contains three consecutive integers
theorem part_b : ∃ (f : ℕ → ℕ) (hf : Function.Injective f), 
  ∀ k, ∃ m : ℤ, m ∈ F (f k) ∧ m + 1 ∈ F (f k) ∧ m + 2 ∈ F (f k) :=
begin
  sorry
end

end part_a_part_b_l49_49135


namespace largest_decimal_of_four_digit_binary_l49_49460

theorem largest_decimal_of_four_digit_binary : ∀ n : ℕ, (n < 16) → n ≤ 15 :=
by {
  -- conditions: a four-digit binary number implies \( n \) must be less than \( 2^4 = 16 \)
  sorry
}

end largest_decimal_of_four_digit_binary_l49_49460


namespace fraction_of_money_left_l49_49795

theorem fraction_of_money_left (m c : ℝ) 
   (h1 : (1/5) * m = (1/3) * c) :
   (m - ((3/5) * m) = (2/5) * m) := by
  sorry

end fraction_of_money_left_l49_49795


namespace math_problem_l49_49957

theorem math_problem (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) : (a + b)^9 + a^6 = 2 :=
sorry

end math_problem_l49_49957


namespace distinct_values_50_l49_49844

theorem distinct_values_50 : 
  let S := {p : ℕ | p = 2 ∨ p = 4 ∨ p = 6 ∨ p = 8 ∨ p = 10 ∨ p = 12 ∨ p = 14 ∨ p = 16 ∨ p = 18 ∨ p = 20}
  in (S.product S).image (λ (pq : ℕ × ℕ), pq.1 * pq.2 + pq.1 + pq.2).to_finset.card = 50 :=
by sorry

end distinct_values_50_l49_49844


namespace perimeter_AMR_l49_49484

-- Definitions based on the problem
variables {A B C M R Q : Type}  -- Defining our points as variables
variables (circle : Type)        -- Defining the circle as a type
variables [IsTangent circle A B] -- A is tangent to the circle at B
variables [IsTangent circle A C] -- A is tangent to the circle at C
variables [Between A M B (MidPoint AB)] -- M is the midpoint on tangent AB
variables [Between A R C] -- R is a point on tangent AC
variables [IsTangent circle R Q] -- Third tangent touches the circle at Q

-- Given conditions
axiom AB_eq_24 : AB = 24
axiom AC_eq_24 : AC = 24
axiom AM_eq_12 : AM = 12
axiom MR_eq_12 : MR = 12

-- The perimeter statement to prove
theorem perimeter_AMR : ∀ (A B C M R Q : Type)
    [IsTangent circle A B]
    [IsTangent circle A C]
    [Between A M B (MidPoint AB)]
    [Between A R C]
    [IsTangent circle R Q]
    (AB_eq_24 : AB = 24)
    (AC_eq_24 : AC = 24)
    (AM_eq_12 : AM = 12)
    (MR_eq_12 : MR = 12)
    , AM + MR + AR = 36 := by sorry

end perimeter_AMR_l49_49484


namespace cos_B_fraction_l49_49344

theorem cos_B_fraction (a b c : ℝ) (h1 : a^2 = b^2 + (1 / 4) * c^2) (h_triangle : ∃ A B C, ∀ (x y z : ℝ), x = a ∧ y = b ∧ z = c) :
  (a * (cos (B) / c) = 5 / 8) :=
sorry

end cos_B_fraction_l49_49344


namespace greatest_product_digits_l49_49558

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49558


namespace solution_set_ineq1_solution_set_ineq2_l49_49737

theorem solution_set_ineq1 (x : ℝ) : 
  (-3 * x ^ 2 + x + 1 > 0) ↔ (x ∈ Set.Ioo ((1 - Real.sqrt 13) / 6) ((1 + Real.sqrt 13) / 6)) := 
sorry

theorem solution_set_ineq2 (x : ℝ) : 
  (x ^ 2 - 2 * x + 1 ≤ 0) ↔ (x = 1) := 
sorry

end solution_set_ineq1_solution_set_ineq2_l49_49737


namespace max_balls_in_cylinder_l49_49348

noncomputable def cylinder_diameter : ℝ := Real.sqrt 2 + 1
noncomputable def cylinder_height : ℝ := 8
noncomputable def ball_diameter : ℝ := 1

theorem max_balls_in_cylinder : 
  (cylinder_diameter = Real.sqrt 2 + 1) →
  (cylinder_height = 8) →
  (ball_diameter = 1) →
  ∃ (n : ℕ), n = 36 :=
by
  intros h1 h2 h3
  use 36
  sorry

end max_balls_in_cylinder_l49_49348


namespace sum_first_n_terms_arithmetic_sequence_l49_49149

theorem sum_first_n_terms_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (a 2 + a 4 = 10) ∧ (∀ n : ℕ, a (n + 1) - a n = 2) → 
  (∀ n : ℕ, S n = n^2) := by
  intro h
  sorry

end sum_first_n_terms_arithmetic_sequence_l49_49149


namespace number_of_four_digit_numbers_l49_49300

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49300


namespace find_diminished_value_l49_49464

theorem find_diminished_value :
  ∃ (x : ℕ), 1015 - x = Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 12 16) 18) 21) 28 :=
by
  use 7
  simp
  unfold Nat.lcm
  sorry

end find_diminished_value_l49_49464


namespace midpoints_perpendicular_axis_of_parabola_l49_49742

-- Definitions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2

def circle (b c r : ℝ) (x y : ℝ) : Prop := (x - b)^2 + (y - c)^2 = r^2

def intersects (a b c r : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.snd = parabola a p.fst ∧ circle b c r p.fst p.snd}

-- Core theorem to prove
theorem midpoints_perpendicular_axis_of_parabola
  {a b c r : ℝ} {P Q R S M N : ℝ × ℝ}
  (h : injective (λ p : ℝ × ℝ, p.fst))
  (hP : P ∈ intersects a b c r)
  (hQ : Q ∈ intersects a b c r)
  (hR : R ∈ intersects a b c r)
  (hS : S ∈ intersects a b c r)
  (h_distinct : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S)
  (hM : M = ((P.fst + S.fst) / 2, (P.snd + S.snd) / 2))
  (hN : N = ((Q.fst + R.fst) / 2, (Q.snd + R.snd) / 2)) :
  M.snd = N.snd := 
sorry

end midpoints_perpendicular_axis_of_parabola_l49_49742


namespace four_digit_numbers_count_l49_49232

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49232


namespace max_digits_of_product_l49_49653

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49653


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49627

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49627


namespace sqrt3_op_sqrt3_l49_49343

def custom_op (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt3_op_sqrt3 : custom_op (Real.sqrt 3) (Real.sqrt 3) = 12 :=
  sorry

end sqrt3_op_sqrt3_l49_49343


namespace triangle_perimeter_l49_49069

theorem triangle_perimeter (area : ℝ) (leg1 : ℝ) (leg2 : ℝ) (hypotenuse : ℝ) 
  (h1 : area = 150)
  (h2 : leg1 = 30)
  (h3 : 0 < leg2)
  (h4 : hypotenuse = (leg1^2 + leg2^2).sqrt)
  (hArea : area = 0.5 * leg1 * leg2)
  : hypotenuse = 10 * Real.sqrt 10 ∧ leg2 = 10 ∧ (leg1 + leg2 + hypotenuse = 40 + 10 * Real.sqrt 10) := 
by
  sorry

end triangle_perimeter_l49_49069


namespace angle_DBC_parallelogram_l49_49902

theorem angle_DBC_parallelogram (A B C D : Type) [parallelogram A B C D]
    (circumcenter_ABC : Type) (circumcenter_CDA : Type)
    (H1 : circumcenter_ABC ∈ line(B, D)) (H2 : circumcenter_CDA ∈ line(B, D))
    (angle_ABD : angle A B D = 35) :
  angle D B C = 35 ∨ angle D B C = 55 :=
sorry

end angle_DBC_parallelogram_l49_49902


namespace four_digit_numbers_count_l49_49220

theorem four_digit_numbers_count : ∃ n : ℕ, 
  let smallest := 1000 in
  let largest := 9999 in
  n = largest - smallest + 1 ∧ n = 9000 :=
sorry

end four_digit_numbers_count_l49_49220


namespace greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49639

def num_digits (n : ℕ) : ℕ := n.to_digits.length

theorem greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers :
  let a := 10^5 - 1,
      b := 10^4 - 1 in
  num_digits (a * b) = 10 :=
by
  sorry

end greatest_possible_number_of_digits_in_product_of_5_and_4_digit_numbers_l49_49639


namespace polynomial_a1_a3_a5_a7_a9_l49_49933

theorem polynomial_a1_a3_a5_a7_a9 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℤ)
  (P : ℤ → ℤ)
  (hP : ∀ x, P x = x^10)
  (hP_eq : P x = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + a_9 * (x + 1)^9 + a_{10} * (x + 1)^{10}) :
  a_1 + a_3 + a_5 + a_7 + a_9 = -512 :=
sorry

end polynomial_a1_a3_a5_a7_a9_l49_49933


namespace valid_paths_count_l49_49763

theorem valid_paths_count :
  let grid := 10 × 10
  let alternating_colors := ∀ (i j : ℕ), (i < 10) → (j < 10) → (i + j) % 2 == if (i + j) % 2 = 0 then "black" else "white"
  let R := (8, even)
  let S := (0, odd)
  let is_valid_path (path : list (ℕ × ℕ)) := 
    path.length = 8 ∧ 
    (path.head = R) ∧ 
    (path.last = S) ∧ 
    ∀ (k : ℕ), (k < path.length - 1) → 
    (path[k].1, path[k].2) == (path[k + 1].1 - 1, path[k + 1].2 - 1) ∨
    (path[k].1, path[k].2) == (path[k + 1].1 - 1, path[k + 1].2 + 1)
in
count_paths R S is_valid_path = 70 :=
sorry

end valid_paths_count_l49_49763


namespace number_of_four_digit_numbers_l49_49279

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49279


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49706

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49706


namespace problem_one_problem_two_l49_49806

noncomputable def prob_one_from_each (p : ℝ) (n : ℕ) : ℝ :=
(choose n 1 * p * (1 - p)) * (choose n 1 * p * (1 - p))

noncomputable def prob_at_least_one (p : ℝ) (n : ℕ) : ℝ :=
1 - (1 - p)^n

theorem problem_one:
  prob_one_from_each 0.6 2 = 0.2304 := 
by sorry

theorem problem_two:
  prob_at_least_one 0.6 4 = 0.9744 := 
by sorry

end problem_one_problem_two_l49_49806


namespace starters_choice_l49_49455

/-- There are 18 players including a set of quadruplets: Bob, Bill, Ben, and Bert. -/
def total_players : ℕ := 18

/-- The set of quadruplets: Bob, Bill, Ben, and Bert. -/
def quadruplets : Finset (String) := {"Bob", "Bill", "Ben", "Bert"}

/-- We need to choose 7 starters, exactly 3 of which are from the set of quadruplets. -/
def ways_to_choose_starters : ℕ :=
  let quadruplet_combinations := Nat.choose 4 3
  let remaining_spots := 4
  let remaining_players := total_players - 4
  quadruplet_combinations * Nat.choose remaining_players remaining_spots

theorem starters_choice (h1 : total_players = 18)
                        (h2 : quadruplets.card = 4) :
  ways_to_choose_starters = 4004 :=
by 
  -- conditional setups here
  sorry

end starters_choice_l49_49455


namespace students_failed_l49_49473

theorem students_failed (total_students : ℕ) (A_percentage : ℚ) (fraction_remaining_B_or_C : ℚ) :
  total_students = 32 → A_percentage = 0.25 → fraction_remaining_B_or_C = 1/4 →
  let students_A := total_students * A_percentage.to_nat in
  let remaining_students := total_students - students_A in
  let students_B_or_C := remaining_students * fraction_remaining_B_or_C.to_nat in
  let students_failed := remaining_students - students_B_or_C in
  students_failed = 18 :=
by
  intros
  simp [students_A, remaining_students, students_B_or_C]
  sorry

end students_failed_l49_49473


namespace garden_length_l49_49087

theorem garden_length (L : ℝ)
  (h1 : ∃ L : ℝ, L > 0)   
  (h2 : ∀ L : ℝ, let width := 120 in
    let total_area := L * width in
    let tilled_area := 1 / 2 * total_area in
    let non_tilled_area := total_area - tilled_area in
    let raised_beds_area := 2 / 3 * non_tilled_area in
    raised_beds_area = 8800) :
  L = 220 :=
by sorry

end garden_length_l49_49087


namespace new_ratio_of_crops_l49_49756

theorem new_ratio_of_crops (total_acres : ℕ) (corn_ratio sugar_cane_ratio tobacco_ratio : ℕ) (additional_tobacco_acres : ℕ)
    (h1 : total_acres = 1350)
    (h2 : corn_ratio = 5)
    (h3 : sugar_cane_ratio = 2)
    (h4 : tobacco_ratio = 2)
    (h5 : additional_tobacco_acres = 450) :
    (let original_units := corn_ratio + sugar_cane_ratio + tobacco_ratio,
         acres_per_unit := total_acres / original_units,
         original_corn_acres := corn_ratio * acres_per_unit,
         original_sugar_cane_acres := sugar_cane_ratio * acres_per_unit,
         original_tobacco_acres := tobacco_ratio * acres_per_unit,
         new_tobacco_acres := original_tobacco_acres + additional_tobacco_acres,
         remaining_acres := total_acres - new_tobacco_acres,
         new_corn_ratio := original_corn_acres / acres_per_unit,
         new_sugar_cane_ratio := original_sugar_cane_acres / acres_per_unit,
         new_tobacco_ratio := new_tobacco_acres / acres_per_unit in
     new_corn_ratio : new_sugar_cane_ratio : new_tobacco_ratio = 5 : 2 : 5) :=
    sorry

end new_ratio_of_crops_l49_49756


namespace number_of_four_digit_numbers_l49_49296

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49296


namespace greatest_digits_product_l49_49584

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49584


namespace count_four_digit_numbers_l49_49313

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49313


namespace books_sold_price_l49_49948

-- Given conditions
variable (total_cost cost_lost percent_loss cost_gain percent_gain : ℝ)
variable (total_cost_eq cost_lost_eq percent_loss_eq percent_gain_eq : Prop)

-- Define the given conditions
def book_conditions := 
  total_cost_eq: total_cost = 600 ∧
  cost_lost_eq: cost_lost = 350 ∧
  percent_loss_eq: percent_loss = 0.15 ∧
  percent_gain_eq: percent_gain = 0.19

-- Define the selling prices and total selling price
def selling_price1 (cost_lost percent_loss : ℝ) : ℝ :=
  cost_lost - percent_loss * cost_lost

def cost_gain (total_cost cost_lost : ℝ) : ℝ :=
  total_cost - cost_lost

def selling_price2 (cost_gain percent_gain : ℝ) : ℝ :=
  cost_gain + percent_gain * cost_gain

def total_selling_price (price1 price2 : ℝ) : ℝ :=
  price1 + price2

-- Theorem to prove the final statement
theorem books_sold_price (total_cost cost_lost percent_loss percent_gain : ℝ)
  (h : book_conditions total_cost_eq cost_lost_eq percent_loss_eq percent_gain_eq) :
  total_selling_price 
    (selling_price1 cost_lost percent_loss)
    (selling_price2 (cost_gain total_cost cost_lost) percent_gain) = 595 :=
by 
  -- Placeholder for the proof
  sorry

end books_sold_price_l49_49948


namespace max_digits_of_product_l49_49643

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49643


namespace greatest_possible_number_of_digits_in_product_l49_49522

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49522


namespace four_digit_number_count_l49_49241

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49241


namespace max_digits_of_product_l49_49652

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49652


namespace greatest_number_of_digits_in_product_l49_49691

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49691


namespace magician_solution_l49_49733

theorem magician_solution 
  (K E C : ℕ)
  (УЧУЙ : ℕ)
  (cond1 : УЧУЙ = (10 * K + E) * (10 * K + C))
  (cond2 : ∀ x y : ℕ, x ≠ y → 10 * K + x ≠ 10 * K + y)
  (cond3 : ∀ k e c УЧУЙ : ℕ, 
            k * 11 + (10 * K + e) * (10 * K + c + 1) + 
            (10 * K + e + 1) * (10 * K + c) + 121 = УЧУЙ + 1111)
  : УЧУЙ = 2021 := 
sorry

end magician_solution_l49_49733


namespace greatest_product_digits_l49_49551

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49551


namespace proof_l49_49400

open Set

-- Universal set U
def U : Set ℕ := {x | x ∈ Finset.range 7}

-- Set A
def A : Set ℕ := {1, 3, 5}

-- Set B
def B : Set ℕ := {4, 5, 6}

-- Complement of A in U
def CU (s : Set ℕ) : Set ℕ := U \ s

-- Proof statement
theorem proof : (CU A) ∩ B = {4, 6} :=
by
  sorry

end proof_l49_49400


namespace log_order_l49_49890

theorem log_order (a b c : ℝ) (h_a : a = Real.log 6 / Real.log 2) 
  (h_b : b = Real.log 15 / Real.log 5) (h_c : c = Real.log 21 / Real.log 7) : 
  a > b ∧ b > c := by sorry

end log_order_l49_49890


namespace greatest_power_of_3_l49_49716

theorem greatest_power_of_3 (n : ℕ) : 
  (n = 603) → 
  3^603 ∣ (15^n - 6^n + 3^n) ∧ ¬ (3^(603+1) ∣ (15^n - 6^n + 3^n)) :=
by
  intro hn
  cases hn
  sorry

end greatest_power_of_3_l49_49716


namespace donuts_total_l49_49781
noncomputable theory

def monday_donuts : ℕ := 14

def tuesday_donuts : ℕ := monday_donuts / 2

def wednesday_donuts : ℕ := 4 * monday_donuts

def total_donuts : ℕ := monday_donuts + tuesday_donuts + wednesday_donuts

theorem donuts_total : total_donuts = 77 := by
  -- Proof goes here
  sorry

end donuts_total_l49_49781


namespace don_travel_time_l49_49405

/-!
# Problem Statement

Mary goes into labor at her local grocery store and is rushed to a hospital in an ambulance traveling 60 mph.
Her husband Don drives after the ambulance at an average speed of 30 mph. Mary reaches the hospital
fifteen minutes later. Prove that it takes Don 0.5 hours to get to the hospital from the store.
-/

theorem don_travel_time :
  ∀ (distance speed_amb speed_don : ℕ),
  speed_amb = 60 →
  speed_don = 30 →
  let time_amb := 0.25 in -- 15 minutes converted to hours
  let distance := speed_amb * time_amb in
  distance = 15 →
  (distance / speed_don) = 0.5 :=
by 
  intros distance speed_amb speed_don h_speed_amb h_speed_don time_amb distance_eq,
  have speed_amb_def : speed_amb = 60 := h_speed_amb,
  have speed_don_def : speed_don = 30 := h_speed_don,
  have time_amb_def : time_amb = 0.25,
  have distance_def : distance = 15 := distance_eq,
  sorry

end don_travel_time_l49_49405


namespace complex_identity_l49_49919

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := - (1 / 2) + (sqrt 3 / 2) * i

-- State the theorem to prove z² + z + 1 = 0
theorem complex_identity (z : ℂ) (hz : z = - (1 / 2) + (sqrt 3 / 2) * i) : z^2 + z + 1 = 0 := 
by sorry

end complex_identity_l49_49919


namespace greatest_number_of_digits_in_product_l49_49625

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49625


namespace count_four_digit_numbers_l49_49307

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49307


namespace arctan_sum_eq_pi_div_two_l49_49829

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l49_49829


namespace greatest_product_digits_l49_49668

theorem greatest_product_digits (a b : ℕ) (h1 : 10000 ≤ a) (h2 : a < 100000) (h3 : 1000 ≤ b) (h4 : b < 10000) :
  ∃ x : ℤ, (int.of_nat (a * b)).digits 10 = 9 :=
by
  -- Placeholder for the proof
  sorry

end greatest_product_digits_l49_49668


namespace greatest_possible_digits_in_product_l49_49515

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49515


namespace cube_x_value_l49_49005

noncomputable def cube_side_len (x : ℝ) : ℝ := (8 * x) ^ (1 / 3)

lemma cube_volume (x : ℝ) : (cube_side_len x) ^ 3 = 8 * x :=
  by sorry

lemma cube_surface_area (x : ℝ) : 6 * (cube_side_len x) ^ 2 = 2 * x :=
  by sorry

theorem cube_x_value (x : ℝ) (hV : (cube_side_len x) ^ 3 = 8 * x) (hS : 6 * (cube_side_len x) ^ 2 = 2 * x) : x = sqrt 3 / 72 :=
  by sorry

end cube_x_value_l49_49005


namespace solve_t_l49_49951

theorem solve_t (t : ℝ) (h : sqrt (5 * (sqrt (t - 5))) = (10 - t + t^2) ^ (1/4)) :
  t = 13 + sqrt 34 ∨ t = 13 - sqrt 34 :=
by
  sorry

end solve_t_l49_49951


namespace four_digit_numbers_count_eq_l49_49265

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49265


namespace solve_y_l49_49876

theorem solve_y (y : ℝ) (h1 : y > 0) (h2 : (y - 6) / 16 = 6 / (y - 16)) : y = 22 :=
by
  sorry

end solve_y_l49_49876


namespace max_distance_l49_49182

open Real

/-- Given the functions f(x) = sin x and g(x) = sin (π/2 - x), 
and the line x = m that intersects their graphs at points M and N respectively, 
the maximum value of |MN| is √2. --/
theorem max_distance {m : ℝ} (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = sin x)
  (hg : ∀ x, g x = sin (π / 2 - x)) :
  let M := (m, f m), N := (m, g m) in
  let dist := abs (f m - g m) in
  M = (m, sin m) ∧ N = (m, sin (π / 2 - m)) ∧ dist = abs (sin m - cos m) ∧ 
  dist ≤ sqrt 2 ∧ (∃ x, dist = sqrt 2) :=
by
  sorry

end max_distance_l49_49182


namespace trapezoid_area_l49_49351

variables (A B C D E : Type) [EuclideanTriangle A B C D E]

-- Geometric settings and given conditions
variable (AB_parallel_CD : A B ∥ C D)
variable (diagonals_intersect : ∃ E, is_intersection_point A C B D E)
variable (area_△ABE : area A B E = 75)
variable (area_△ADE : area A D E = 30)

-- Prove that the area of trapezoid ABCD is 147 square units
theorem trapezoid_area (h1 : AB_parallel_CD)
    (h2 : diagonals_intersect)
    (h3 : area_△ABE)
    (h4 : area_△ADE) :
    area_trapezoid A B C D = 147 := sorry

end trapezoid_area_l49_49351


namespace bob_eats_10_apples_l49_49482

variable (B C : ℕ)
variable (h1 : B + C = 30)
variable (h2 : C = 2 * B)

theorem bob_eats_10_apples : B = 10 :=
by sorry

end bob_eats_10_apples_l49_49482


namespace sequence_formula_l49_49150

-- Define the sequence and the sum of the first n terms
def sequence (n : ℕ) : ℝ := 1 / (n * (n + 1))

def partial_sum (n : ℕ) : ℝ := (finset.range n).sum sequence

-- State the main theorem to prove
theorem sequence_formula :
  ∀ (n : ℕ), (partial_sum n - 1) ^ 2 = (sequence n) * (partial_sum n) → sequence n = 1 / (n * (n + 1)) :=
by
  sorry

end sequence_formula_l49_49150


namespace union_of_A_and_B_l49_49187

open Set

def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem union_of_A_and_B : A ∪ B = {2, 3, 5, 6} := sorry

end union_of_A_and_B_l49_49187


namespace count_four_digit_numbers_l49_49316

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49316


namespace greatest_product_digit_count_l49_49677

theorem greatest_product_digit_count :
  let a := 99999
  let b := 9999
  Nat.digits 10 (a * b) = 9 := by
    sorry

end greatest_product_digit_count_l49_49677


namespace find_length_AC_l49_49443

variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [has_dist A B] [has_dist B C] [has_dist C D] [has_dist A C] [has_dist A D] [has_dist B D]

-- Define the quadrilateral with vertices A, B, C, D
variables (AB : ℝ) (BD : ℝ) (DC : ℝ) (AC : ℝ)
-- Condition: The area of the quadrilateral ABCD is 18
def area_quadrilateral_ABCD : ℝ := 18
-- Condition: The sum of the lengths of AB, BD, and DC is 12
def sum_AB_BD_DC : ℝ := 12

theorem find_length_AC : 
  area_quadrilateral_ABCD = 18 →
  AB + BD + DC = 12 →
  |AC| = 6 * real.sqrt 2 :=
by
  sorry

end find_length_AC_l49_49443


namespace min_edges_for_resilient_network_l49_49003

theorem min_edges_for_resilient_network (n : ℕ) (h : n = 10) :
    ∃ G : SimpleGraph (Fin n), G.edgeCount = 15 ∧ G.kConnectivity 2 :=
sorry

end min_edges_for_resilient_network_l49_49003


namespace max_digits_of_product_l49_49537

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49537


namespace arctan_sum_pi_div_two_l49_49809

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l49_49809


namespace area_of_45_45_90_right_triangle_l49_49457

theorem area_of_45_45_90_right_triangle (h : ℝ) (hypotenuse : h = 10 * real.sqrt 2)
    (angle : ∃ θ, θ = real.pi / 4) :
    ∃ A, A = 50 := sorry

end area_of_45_45_90_right_triangle_l49_49457


namespace leak_drains_in_34_hours_l49_49728

-- Define the conditions
def pump_rate := 1 / 2 -- rate at which the pump fills the tank (tanks per hour)
def time_with_leak := 17 / 8 -- time to fill the tank with the pump and the leak (hours)

-- Define the combined rate of pump and leak
def combined_rate := 1 / time_with_leak -- tanks per hour

-- Define the leak rate
def leak_rate := pump_rate - combined_rate -- solve for leak rate

-- Define the proof statement
theorem leak_drains_in_34_hours : (1 / leak_rate) = 34 := by
    sorry

end leak_drains_in_34_hours_l49_49728


namespace number_of_four_digit_numbers_l49_49278

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49278


namespace number_of_four_digit_numbers_l49_49276

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l49_49276


namespace combined_weight_of_daughter_and_child_l49_49761

theorem combined_weight_of_daughter_and_child 
  (G D C : ℝ)
  (h1 : G + D + C = 110)
  (h2 : C = 1/5 * G)
  (h3 : D = 50) :
  D + C = 60 :=
sorry

end combined_weight_of_daughter_and_child_l49_49761


namespace a_eq_b_if_conditions_l49_49437

theorem a_eq_b_if_conditions (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := 
sorry

end a_eq_b_if_conditions_l49_49437


namespace determine_a_l49_49451

-- Given: 
-- - The equation of the hyperbola is x^2 - y^2 = a^2, where a > 0
-- - The distance from one of the foci of the hyperbola to an asymptote is 2
-- To prove: a = 2

noncomputable def hyperbola_foci_distance (a : ℝ) (h : a > 0) : Prop :=
  let foci := (Real.sqrt 2 * a) in
  let asymptote := 2 in
  foci / (Real.sqrt (1 + 1)) = asymptote

theorem determine_a (a : ℝ) (h : a > 0) (d : hyperbola_foci_distance a h) : a = 2 :=
sorry

end determine_a_l49_49451


namespace four_digit_numbers_count_l49_49246

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49246


namespace inclination_angle_range_l49_49938

theorem inclination_angle_range {θ : ℝ} (h : cos θ ≠ 0) :
  ∃ α : ℝ, tan α = -cos θ ∧ (0 < α ∧ α ≤ π/4 ∨ 3 * π / 4 ≤ α ∧ α < π) :=
sorry

end inclination_angle_range_l49_49938


namespace greatest_number_of_digits_in_product_l49_49616

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49616


namespace greatest_number_of_digits_in_product_l49_49622

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49622


namespace expand_binomials_l49_49862

variable (x y : ℝ)

theorem expand_binomials: (2 * x - 5) * (3 * y + 15) = 6 * x * y + 30 * x - 15 * y - 75 :=
by sorry

end expand_binomials_l49_49862


namespace orthocenters_collinear_l49_49887

-- Define the geometric setup with lines and triangles
variables {P : Type} [metric_space P] [normed_group P]
variables (L1 L2 L3 L4 : line P)
variables [distinct L1 L2 L3 L4]

theorem orthocenters_collinear :
  ∀ (Δ1 Δ2 Δ3 Δ4 : triangle P)
  (h1 : formed_by L1 L2 Δ1) (h2 : formed_by L2 L3 Δ2)
  (h3 : formed_by L3 L4 Δ3) (h4 : formed_by L4 L1 Δ4),
  collinear (orthocenter Δ1) (orthocenter Δ2) (orthocenter Δ3) (orthocenter Δ4) := 
sorry

end orthocenters_collinear_l49_49887


namespace greatest_digits_product_l49_49586

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49586


namespace arctan_sum_eq_pi_div_two_l49_49827

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l49_49827


namespace decreasing_function_value_l49_49336

theorem decreasing_function_value (a : ℝ) : 
  (∀ x y : ℝ, x < y → y < 2 → f y ≤ f x) → (a = -1) := by
  let f : ℝ → ℝ := λ x, x^2 + 2 * (a - 1) * x + 1
  intro h
  sorry

end decreasing_function_value_l49_49336


namespace max_cardinality_of_set_proof_l49_49904

noncomputable def max_cardinality_of_set (n : ℕ) : ℕ :=
  ⌊ (3 * (n + 1) ^ 2 + 1) / 4 ⌋

theorem max_cardinality_of_set_proof (n : ℕ) (S : Finset ℕ) 
  (hD : ∀ s ∈ S, s ∈ finset.range (n + 1))
  (hNoDivide : ∀ ⦃a b⦄, a ∈ S → b ∈ S → a ≠ b → ¬ (a ∣ b ∨ b ∣ a)) :
  S.card ≤ max_cardinality_of_set n :=
by
  sorry

end max_cardinality_of_set_proof_l49_49904


namespace find_t_l49_49141

theorem find_t (t : ℝ) : 
  (∀ x ∈ Icc (0 : ℝ) 3, |x^2 - 2*x - t| ≤ 2) →
  (∃ x ∈ Icc (0 : ℝ) 3, |x^2 - 2*x - t| = 2) → 
  (t = 1) :=
by
  sorry

end find_t_l49_49141


namespace cube_dimension_l49_49020

theorem cube_dimension (x s : ℝ) (hx1 : s^3 = 8 * x) (hx2 : 6 * s^2 = 2 * x) : x = 1728 := 
by {
  sorry
}

end cube_dimension_l49_49020


namespace total_value_of_coins_l49_49746

theorem total_value_of_coins (total_count : ℕ) (one_rupee_count : ℕ)
                             (fifty_paise_count : ℕ) (twentyfive_paise_count : ℕ)
                             (one_rupee_value fifty_paise_value twentyfive_paise_value : ℕ) :
  one_rupee_count = 120 →
  fifty_paise_count = 120 →
  twentyfive_paise_count = 120 →
  one_rupee_value = 1 →
  fifty_paise_value = 50 →
  twentyfive_paise_value = 25 →
  total_value_of_coins = (one_rupee_count * one_rupee_value) +
                        (fifty_paise_count * fifty_paise_value) / 100 +
                        (twentyfive_paise_count * twentyfive_paise_value) / 100 →
  total_value_of_coins = 210 :=
by
  intros h_one_rupee h_fifty_paise h_twentyfive_paise
        h_one_value h_fifty_value h_twentyfive_value
        h_total_value
  sorry

end total_value_of_coins_l49_49746


namespace min_value_of_expression_l49_49462

noncomputable def X : Type := sorry  -- Placeholder, as we assume it's defined in the context

-- Define the properties and conditions
def normal_distribution (X : Type) (μ : ℝ) (σ2 : ℝ) : Prop := sorry -- Normal distribution definition placeholder
def prob (X : Type) (event : Set X) : ℝ := sorry  -- Probability measure definition placeholder

variables (σ2 : ℝ)
variables (X : Type) (μ : ℝ) (m n : ℝ)

-- Conditions from the problem
axiom cond1 : normal_distribution X 10 σ2
axiom cond2 : prob X { x | x > 12 } = m
axiom cond3 : prob X { x | 8 ≤ x ∧ x ≤ 10 } = n
axiom cond4 : m + n = 1 / 2

-- The statement to prove
theorem min_value_of_expression : 
  (∀ m n, normal_distribution X 10 σ2 → prob X { x | x > 12 } = m → prob X { x | 8 ≤ x ∧ x ≤ 10 } = n → m + n = 1/2 → (2/m + 1/n) = 6 + 4 * Real.sqrt 2) :=
sorry

end min_value_of_expression_l49_49462


namespace greatest_possible_digits_in_product_l49_49519

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49519


namespace max_digits_of_product_l49_49546

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49546


namespace greatest_possible_number_of_digits_in_product_l49_49531

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49531


namespace max_digits_of_product_l49_49642

theorem max_digits_of_product : 
  ∀ (a b : ℕ), (a = 10^5 - 1) → (b = 10^4 - 1) → Nat.digits 10 (a * b) = 9 := 
by
  intros a b ha hb
  -- Definitions
  have h_a : a = 99999 := ha
  have h_b : b = 9999 := hb
  -- Sorry to skip the proof part
  sorry

end max_digits_of_product_l49_49642


namespace sum_of_distances_l49_49424

noncomputable theory

-- Let A, B, C, and D be the vertices of the isosceles trapezoid
variables {A B C D O : Point}
variables {dOA dOB dOC dOD : ℝ}

-- Assume BC is parallel to AD, and A, B, C, D form an isosceles trapezoid
axiom is_isosceles_trapezoid (h : is_isosceles_trapezoid A B C D)

-- Distance function from point to point
def distance (P Q : Point) : ℝ := sorry

-- Define distances from O to vertices
axiom distance_from_O_to_vertices :
  distance O A = dOA ∧ distance O B = dOB ∧ distance O C = dOC ∧ distance O D = dOD

-- The main statement to be proven
theorem sum_of_distances (h_isosceles : is_isosceles_trapezoid A B C D) (h_dist : distance_from_O_to_vertices) :
  dOA + dOB + dOC > dOD :=
by sorry

end sum_of_distances_l49_49424


namespace greatest_number_of_digits_in_product_l49_49694

theorem greatest_number_of_digits_in_product :
  ∀ (n m : ℕ), 10000 ≤ n ∧ n < 100000 → 1000 ≤ m ∧ m < 10000 → 
  ( ∃ k : ℕ, k = 10 ∧ ∀ p : ℕ, p = n * m → p.digitsCount ≤ k ) :=
begin
  sorry
end

end greatest_number_of_digits_in_product_l49_49694


namespace bananas_to_oranges_equivalence_l49_49439

theorem bananas_to_oranges_equivalence :
  (3 / 4 : ℚ) * 16 = 12 ->
  (2 / 5 : ℚ) * 10 = 4 :=
by
  intros h
  sorry

end bananas_to_oranges_equivalence_l49_49439


namespace prove_product_in_base9_l49_49126

def product_in_base9 (a b : ℕ) (h1 : a = 3*9^2 + 2*9 + 5) (h2 : b = 6) : (c : ℕ) (h3 : c = 2*9^3 + 0*9^2 + 3*9 + 3) := 
  a * b = c

-- We assert and then leave the proof with sorry
theorem prove_product_in_base9 : product_in_base9 3 6 2033_9 := 
  sorry

end prove_product_in_base9_l49_49126


namespace sequence_general_formula_l49_49934

theorem sequence_general_formula (a : ℕ → ℝ) (h₁ : a 1 = 3) 
    (h₂ : ∀ n : ℕ, 1 < n → a n = (n / (n - 1)) * a (n - 1)) : 
    ∀ n : ℕ, 1 ≤ n → a n = 3 * n :=
by
  -- Proof description here
  sorry

end sequence_general_formula_l49_49934


namespace bank_balance_after_2_years_l49_49433

noncomputable def compound_interest (P₀ : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P₀ * (1 + r)^n

theorem bank_balance_after_2_years :
  compound_interest 100 0.10 2 = 121 := 
  by
  sorry

end bank_balance_after_2_years_l49_49433


namespace greatest_possible_number_of_digits_in_product_l49_49525

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49525


namespace greatest_product_digits_l49_49562

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49562


namespace four_digit_number_count_l49_49206

/-- Four-digit numbers start at 1000 and end at 9999. -/
def fourDigitNumbersStart : ℕ := 1000
def fourDigitNumbersEnd : ℕ := 9999

theorem four_digit_number_count : (fourDigitNumbersEnd - fourDigitNumbersStart + 1 = 9000) := 
by 
  sorry

end four_digit_number_count_l49_49206


namespace eccentricity_range_l49_49897

-- Definitions of the problem
def circle := {center : ℝ × ℝ, radius : ℝ}
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := y^2 / a^2 - x^2 / b^2 = 1
def asymptotes (a b : ℝ) (x y : ℝ) : Prop := y = a / b * x ∨ y = -a / b * x

-- Conditions given in the problem
structure problem_conditions (a b : ℝ) : Prop :=
  (radius_pos : a > 0)
  (center_pos : b > 0)
  (disjoint : ∀ x y, ¬ ((a^2 + b^2) ≤ b^2 + a * sqrt(a^2 + b^2)))

-- To prove: given the conditions, the eccentricity e falls within the specified range
theorem eccentricity_range (a b : ℝ) (e : ℝ) (h : problem_conditions a b) :
  (1 < e) ∧ (e < (1 + Real.sqrt 5) / 2) :=
sorry

end eccentricity_range_l49_49897


namespace hexagon_radical_axes_intersect_l49_49976

theorem hexagon_radical_axes_intersect
  (vertices : list Point)
  (H1 : ∀ (A B C D : Point), ¬(∃ c : Circle, ∀ x ∈ [A, B, C, D], x ∈ c))
  (H2 : ∃ P : Point, ∀ (A B C D : Point), are_diagonals_intersecting_in P [A, B, C, D]) :
  ∃ Q : Point, ∀ (A B : Circle), radical_axis A B = Q :=
sorry

end hexagon_radical_axes_intersect_l49_49976


namespace incorrect_analogy_l49_49937

-- Define the multiplication rule of complex numbers and polynomials as analogous.
def mult_rule_analogous (a b : ℂ) (c d : Polynomial ℂ) : Prop :=
  a * b = c.eval a * d.eval b

-- Define the geometric meaning of addition for vectors and complex numbers as analogous.
def geom_add_analogous (a b : ℂ) (ca cb : Vector ℂ) : Prop :=
  a + b = ca + cb

-- The incorrect analogy to be proven based on the given conditions.
theorem incorrect_analogy :
  (mult_rule_analogous (z : ℂ) (w : ℂ) (p : Polynomial ℂ) (q : Polynomial ℂ)) →
  (geom_add_analogous (z : ℂ) (w : ℂ) (va : Vector ℂ) (vb : Vector ℂ)) →
  ∃ (z : ℂ), (|z|^2) ≠ (z^2) :=
by
  intro mult_rule geom_add
  use i
  sorry

end incorrect_analogy_l49_49937


namespace average_race_time_l49_49804

def casey_time : ℝ := 6
def zendaya_time : ℝ := casey_time + (1/3) * casey_time

theorem average_race_time :
  (casey_time + zendaya_time) / 2 = 7 := by
  sorry

end average_race_time_l49_49804


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49704

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49704


namespace expression_for_f_l49_49892

noncomputable def f (x : ℝ) : ℝ := sorry

theorem expression_for_f (x : ℝ) :
  (∀ x, f (x - 1) = x^2) → f x = x^2 + 2 * x + 1 :=
by
  intro h
  sorry

end expression_for_f_l49_49892


namespace min_value_is_one_l49_49000

def min_value_expression (x y : ℝ) : ℝ :=
  (x * y + 1)^2 + (x^2 + y^2)^2

theorem min_value_is_one : ∃ x y : ℝ, min_value_expression x y = 1 :=
by {
  use [0, 0], 
  -- here, we provide the values that achieve the minimum
  sorry
}

end min_value_is_one_l49_49000


namespace geometric_sequence_third_term_l49_49333

theorem geometric_sequence_third_term (a₁ : ℤ) (r : ℤ) (sum_first_two_terms : ℤ) 
    (h1 : r = -3) (h2 : sum_first_two_terms = a₁ + a₁ * r) : 
    let a₃ := a₁ * r^2 in a₃ = -45 :=
by
  -- Define a₁ using the given conditions
  have h3 : a₁ = -5, from by
    calc
      (a₁ + a₁ * r = 10) : by rw [h2]
      (a₁ + a₁ * -3 = 10) : by rw [h1]
      (a₁ - 3 * a₁ = 10) : rfl
      (-2 * a₁ = 10) : by ring
      (a₁ = 10 / -2) : by {ring, apply eq_neg_of_eq_neg, ring}
      (a₁ = -5) : rfl,
  -- Use this to define a₃ and show that it equals -45
  calc
    a₃ = a₁ * r^2 : rfl
    a₃ = -5 * (-3)^2 : by rw [h1, h3]
    a₃ = -5 * 9 : rfl
    a₃ = -45 : rfl,

end geometric_sequence_third_term_l49_49333


namespace complex_series_sum_l49_49378

theorem complex_series_sum (ω : ℂ) (h₁ : ω^7 = 1) (h₂ : ω ≠ 1) :
  (ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 + ω^32 + 
   ω^34 + ω^36 + ω^38 + ω^40 + ω^42 + ω^44 + ω^46 + ω^48 + ω^50 + 
   ω^52 + ω^54) = -1 :=
by
  sorry

end complex_series_sum_l49_49378


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49710

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49710


namespace prob_of_drawing_2_math_books_is_correct_l49_49719

open Finset

def total_books := 12
def chinese_books := 10
def math_books := 2

noncomputable def choose (n k : ℕ) : ℕ := Nat.descFactorial n k / Nat.factorial k

def ways_to_draw_2_total : ℕ := choose total_books 2
def ways_to_draw_2_chinese : ℕ := choose chinese_books 2

def prob_drawing_2_math_books : ℚ :=
  1 - (ways_to_draw_2_chinese / ways_to_draw_2_total)

theorem prob_of_drawing_2_math_books_is_correct :
  prob_drawing_2_math_books = 7 / 22 := by
  sorry

end prob_of_drawing_2_math_books_is_correct_l49_49719


namespace count_four_digit_numbers_l49_49286

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49286


namespace x_varies_nth_power_of_z_l49_49961

noncomputable def x_variation (k j : ℝ) (y z : ℝ) : Prop :=
  let x := k * (y^2)
  let y := j * (z^(1 / 3))
  x = k * j^2 * z^(2 / 3)

theorem x_varies_nth_power_of_z (k j : ℝ) (y z : ℝ) :
  x_variation k j y z → exists m : ℝ, ∀ z : ℝ, x_variation k j y z = m * z^(2 / 3) :=
by
  sorry

end x_varies_nth_power_of_z_l49_49961


namespace four_digit_numbers_count_eq_l49_49266

theorem four_digit_numbers_count_eq :
  let a := 1000
  let b := 9999
  (b - a + 1) = 9000 := by
  sorry

end four_digit_numbers_count_eq_l49_49266


namespace sec_neg_420_equals_2_l49_49868

theorem sec_neg_420_equals_2 : ∀ (x : ℝ), x = -420 * (π / 180) →
  ∀ (y : ℝ), y = 60 * (π / 180) →
  (cos (x + 2 * π) = cos x) →
  (cos (-y) = cos y) →
  cos y = 1 / 2 →
  sec x = 2 :=
by
  intros x hx y hy h1 h2 h3
  rw hx
  rw [cos_add (by simp) (show 2 * π = 2 * real.pi from rfl)] at h1
  rw [←real.pi_div_two_add_pi]
  rw ←hy at h3
  rw h1 at h2
  rw h2 at h3
  sorry
  -- the completed proof can proceed using the above hypotheses

end sec_neg_420_equals_2_l49_49868


namespace greatest_possible_digits_in_product_l49_49518

theorem greatest_possible_digits_in_product :
  ∀ n m : ℕ, (9999 ≤ n ∧ n < 10^5) → (999 ≤ m ∧ m < 10^4) → 
  natDigits (10^9 - 10^5 - 10^4 + 1) 10 = 9 := 
by
  sorry

end greatest_possible_digits_in_product_l49_49518


namespace max_intersections_with_10_streets_l49_49345

def max_intersections (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem max_intersections_with_10_streets :
  max_intersections 10 = 45 :=
by
  rw [max_intersections]
  norm_num
  sorry

end max_intersections_with_10_streets_l49_49345


namespace truncated_pyramid_volume_l49_49105

theorem truncated_pyramid_volume (B1D1 a b S1 S2 : ℝ) (h S1_eq S2_eq : ℝ) :
  B1D1 = 18 ∧ a = 14 ∧ b = 10 ∧ S1 = 196 ∧ S2 = 100 →
  h = sqrt (B1D1 ^ 2 - (12 * sqrt 2) ^ 2) →
  ∀ (V : ℝ), V = (h / 3) * (S1 + S2 + sqrt (S1 * S2)) →
  V = 872 :=
by
  intros h0 h1 h2;
  sorry

end truncated_pyramid_volume_l49_49105


namespace number_of_four_digit_numbers_l49_49294

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49294


namespace functional_eq1_bijective_functional_eq2_neither_functional_eq3_neither_functional_eq4_neither_l49_49779

-- Problem 1
theorem functional_eq1_bijective (f : ℝ → ℝ) 
  (h : ∀ x y, f(x + f(y)) = 2 * f(x) + y) : 
  function.bijective f := 
sorry

-- Problem 2
theorem functional_eq2_neither (f : ℝ → ℝ) 
  (h : ∀ x, f(f(x)) = 0) : 
  ¬ function.injective f ∧ ¬ function.surjective f := 
sorry

-- Problem 3
theorem functional_eq3_neither (f : ℝ → ℝ) 
  (h : ∀ x, f(f(x)) = Real.sin x) : 
  ¬ function.injective f ∧ ¬ function.surjective f := 
sorry

-- Problem 4
theorem functional_eq4_neither (f : ℝ → ℝ) 
  (h : ∀ x y, f(x + y) = f(x) * f(y)) : 
  ¬ function.injective f ∧ ¬ function.surjective f := 
sorry

end functional_eq1_bijective_functional_eq2_neither_functional_eq3_neither_functional_eq4_neither_l49_49779


namespace depth_of_channel_l49_49040

noncomputable def trapezium_area (a b h : ℝ) : ℝ :=
1/2 * (a + b) * h

theorem depth_of_channel :
  ∃ h : ℝ, trapezium_area 12 8 h = 700 ∧ h = 70 :=
by
  use 70
  unfold trapezium_area
  sorry

end depth_of_channel_l49_49040


namespace find_CK_and_angle_ABC_l49_49360

noncomputable def problem_conditions (ABC : Triangle) (O O1 O2 : Point) (K K1 K2 K3 : Point) (r R : ℝ) : Prop :=
  -- Circles inscribed at B and C with equal radii r
  is_incircle O1 (Triangle.mk B O1 C) ∧ 
  is_incircle O2 (Triangle.mk C O2 B) ∧ 
  radius O1 = r ∧ radius O2 = r ∧
  
  -- Incircle of Triangle ABC
  is_incircle O (Triangle.mk A B C) ∧ radius O = R ∧

  -- The circles touch side BC at K1, K2, and K respectively
  touches_at_point O1 B C K1 ∧ touches_at_point O2 C B K2 ∧ touches_at_point O A B C K ∧

  -- Given lengths
  distance B K1 = 4 ∧ distance C K2 = 8 ∧ distance B C = 18
  
noncomputable def part_a_statement (CK : ℝ) : Prop :=
  CK = 12

noncomputable def part_b_statement (angleABC : ℝ) : Prop :=
  angleABC = 60

theorem find_CK_and_angle_ABC (ABC : Triangle) (O O1 O2 : Point) (K K1 K2 K3 : Point) (r R : ℝ) :
  problem_conditions ABC O O1 O2 K K1 K2 K3 r R →
  (∃ CK, part_a_statement CK) ∧ (∃ angleABC, part_b_statement angleABC) :=
by 
  intros h
  split
  -- Proofs for length of segment CK and angle ABC
  sorry
  sorry

end find_CK_and_angle_ABC_l49_49360


namespace arctan_sum_pi_div_two_l49_49822

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end arctan_sum_pi_div_two_l49_49822


namespace arctan_triangle_complementary_l49_49816

theorem arctan_triangle_complementary :
  (Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2) :=
begin
  sorry
end

end arctan_triangle_complementary_l49_49816


namespace circumscribed_circle_diameter_l49_49965

theorem circumscribed_circle_diameter (a : ℝ) (A : ℝ) (h_a : a = 16) (h_A : A = 30) :
    let D := a / Real.sin (A * Real.pi / 180)
    D = 32 := by
  sorry

end circumscribed_circle_diameter_l49_49965


namespace find_a_l49_49438

noncomputable def f : ℝ+ → ℝ := sorry

theorem find_a (f : ℝ+ → ℝ) (h1 : ∀ (x y : ℝ+), f (x * y) = f x + f y)
  (h2 : f 8 = -3) :
  ∃ a : ℝ+, f a = 1 / 2 ∧ a = ⟨√2 / 2, sorry⟩ :=
sorry

end find_a_l49_49438


namespace percentage_relationship_l49_49950

variable (x y : ℕ)

theorem percentage_relationship (h1 : 2 * x = 0.5 * y) (h2 : x = 16) : y = 64 :=
  sorry

end percentage_relationship_l49_49950


namespace arctan_sum_eq_pi_div_two_l49_49830

theorem arctan_sum_eq_pi_div_two : Real.arctan (3 / 4) + Real.arctan (4 / 3) = Real.pi / 2 :=
by
  sorry

end arctan_sum_eq_pi_div_two_l49_49830


namespace angle_between_clock_hands_at_340_l49_49718

theorem angle_between_clock_hands_at_340 (h m : ℕ) (H1 : h = 3) (H2 : m = 40) : 
  let minute_angle := (m : ℝ) * 6,
      hour_angle := (h : ℝ) * 30 + (m : ℝ) * 0.5,
      angle_diff := abs (minute_angle - hour_angle) in
  min angle_diff (360 - angle_diff) = 140 :=
by
  -- Placeholder for actual proof
  have idm : minute_angle = 240 := by sorry,
  have idh : hour_angle = 100 := by sorry,
  have idad : angle_diff = 140 := by sorry,
  sorry

end angle_between_clock_hands_at_340_l49_49718


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49577

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49577


namespace find_x_condition_l49_49878

theorem find_x_condition (x : ℚ) :
  (∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) → x = 3 / 2 :=
begin
  sorry
end

end find_x_condition_l49_49878


namespace smallest_n_f_greater_than_20_l49_49393

def f (n : ℕ) : ℕ := sorry

theorem smallest_n_f_greater_than_20 : ∃ n, (∃ r, n = 20 * r ∧ r = 17) ∧ f(n) > 20 :=
begin
  have h1: f(340) > 20 := sorry,
  use 340,
  split,
  { use 17,
    rw mul_comm,
    simp, },
  { exact h1, },
end

end smallest_n_f_greater_than_20_l49_49393


namespace inverse_proportion_comparison_l49_49920

theorem inverse_proportion_comparison (y1 y2 : ℝ) 
  (h1 : y1 = - 6 / 2)
  (h2 : y2 = - 6 / -1) : 
  y1 < y2 :=
by
  sorry

end inverse_proportion_comparison_l49_49920


namespace number_of_four_digit_numbers_l49_49293

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49293


namespace count_congruent_to_3_mod_8_l49_49945

theorem count_congruent_to_3_mod_8 : 
  ∃ (count : ℕ), count = 31 ∧ ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 250 → x % 8 = 3 → x = 8 * ((x - 3) / 8) + 3 := sorry

end count_congruent_to_3_mod_8_l49_49945


namespace alice_favorite_number_l49_49776

def is_multiple (x y : ℕ) : Prop := ∃ k : ℕ, k * y = x
def digit_sum (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem alice_favorite_number 
  (n : ℕ) 
  (h1 : 90 ≤ n ∧ n ≤ 150) 
  (h2 : is_multiple n 13) 
  (h3 : ¬ is_multiple n 4) 
  (h4 : is_multiple (digit_sum n) 4) : 
  n = 143 := 
by 
  sorry

end alice_favorite_number_l49_49776


namespace solution_l49_49356

variable (a : ℕ → ℝ)

axiom geometric_sequence (a : ℕ → ℝ) (a1 : ℝ) (q : ℝ) :
  a 1 = a1 →
  ∀ n : ℕ, a (n + 1) = a n * q → ∃ q: ℝ, a n = a1 * q^(n-1)

noncomputable def a (n : ℕ) : ℝ := 27 * (1 / 3) ^ (n - 1)

def problem_statement := a 6 = 1 / 9

theorem solution : problem_statement := by
  unfold problem_statement
  sorry

end solution_l49_49356


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49579

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49579


namespace four_digit_numbers_count_l49_49223

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49223


namespace largest_last_digit_of_special_string_l49_49452

theorem largest_last_digit_of_special_string :
  ∃ (s : List ℕ), s.length = 1500 ∧
    s.head = 1 ∧
    (∀ (i : ℕ), i < 1499 → ∃ (d : ℕ), 17 * d < 100 ∨ 23 * d < 100 ∧ 
      (10 * s[i] + s[i + 1]) = 17 * d ∨ (10 * s[i] + s[i + 1]) = 23 * d) ∧
    s.last = some 5 := 
sorry

end largest_last_digit_of_special_string_l49_49452


namespace four_digit_number_count_l49_49243

-- Define the start and end of four-digit numbers
def fourDigitStart : Nat := 1000
def fourDigitEnd : Nat := 9999

-- Main theorem: Number of four-digit numbers
theorem four_digit_number_count : fourDigitEnd - fourDigitStart + 1 = 9000 := by
  sorry  -- Proof here

end four_digit_number_count_l49_49243


namespace johns_age_l49_49132

theorem johns_age :
  ∃ x : ℕ, (∃ n : ℕ, x - 5 = n^2) ∧ (∃ m : ℕ, x + 3 = m^3) ∧ x = 69 :=
by
  sorry

end johns_age_l49_49132


namespace expected_games_is_correct_l49_49732

def prob_A_wins : ℚ := 2 / 3
def prob_B_wins : ℚ := 1 / 3
def max_games : ℕ := 6

noncomputable def expected_games : ℚ :=
  2 * (prob_A_wins^2 + prob_B_wins^2) +
  4 * (prob_A_wins * prob_B_wins * (prob_A_wins^2 + prob_B_wins^2)) +
  6 * (prob_A_wins * prob_B_wins)^2

theorem expected_games_is_correct : expected_games = 266 / 81 := by
  sorry

end expected_games_is_correct_l49_49732


namespace find_x_l49_49880

theorem find_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) : x = 3 / 2 :=
sorry

end find_x_l49_49880


namespace unique_pairs_of_students_l49_49974

/-- In a classroom of 15 students, the number of unique pairs of students is 105. -/
theorem unique_pairs_of_students (n : ℕ) (h : n = 15) : (nat.choose n 2) = 105 :=
by
  rw [h]
  -- sorry to skip the proof steps
  sorry

end unique_pairs_of_students_l49_49974


namespace sum_of_real_roots_l49_49852

theorem sum_of_real_roots (f : ℝ → ℝ) (h : f = λ x, x^4 - 6*x - 2) :
  (∑ x in Finset.filter (λ x, is_zero (f x)) (Multiset.to_finset (roots f ℝ)), x) = 2 :=
sorry

end sum_of_real_roots_l49_49852


namespace greatest_number_of_digits_in_product_l49_49623

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49623


namespace problem_three_draws_prob_l49_49748

noncomputable def probability_three_draws : ℚ :=
  let chips := {1, 2, 3, 4, 5, 6}
  let favorable_pairs := { (1,2), (1,3), (1,4), (1,5), (1,6),
                           (2,1), (2,3), (2,4), (2,5), 
                           (3,1), (3,2), (3,4), 
                           (4,1), (4,2), (4,3), 
                           (5,1), 
                           (6,1) }
  let n_favorable_pairs := 17
  let prob_pair := 1 / 30
  7 / 30 -- This is the known result based on the problem-solving

theorem problem_three_draws_prob (chips : finset ℕ) (favorable_pairs : finset (ℕ × ℕ)):
  (chips = {1, 2, 3, 4, 5, 6}) →
  (favorable_pairs = { (1,2), (1,3), (1,4), (1,5), (1,6),
                        (2,1), (2,3), (2,4), (2,5),
                        (3,1), (3,2), (3,4),
                        (4,1), (4,2), (4,3),
                        (5,1), 
                        (6,1) }) →
  probability_three_draws = 7 / 30 := by
  intros
  sorry

end problem_three_draws_prob_l49_49748


namespace cube_volume_surface_area_eq_1728_l49_49010

theorem cube_volume_surface_area_eq_1728 (x : ℝ) (h1 : (side : ℝ) (v : ℝ) hvolume : v = 8 * x ∧ v = side^3) (h2: (side : ℝ) (a : ℝ) hsurface : a = 2 * x ∧ a = 6 * side^2) : x = 1728 :=
by
  sorry

end cube_volume_surface_area_eq_1728_l49_49010


namespace money_left_is_26_l49_49406

-- Define the allowance and the spent amount
def total_allowance : ℝ := sorry
def spent := 14
def spent_percent := 35

-- State that spent amount is 35% of the total allowance
axiom spent_percentage (total_allowance : ℝ) : spent = spent_percent * total_allowance / 100

-- Define the amount left and its correctness
def amount_left : ℝ := total_allowance * (100 - spent_percent) / 100

-- The theorem we need to prove
theorem money_left_is_26 (total_allowance : ℝ) (h : spent_percentage total_allowance) : amount_left = 26 :=
sorry

end money_left_is_26_l49_49406


namespace angle_APB_is_90_degrees_l49_49417

-- Define the setup for the problem
noncomputable def rectangle (ABCD : Type) := sorry

/-- Define the points M, E, P, and their properties -/
variables (A B C D M E P : Type)

/-- Conditions:
1. ABCD is a rectangle,
2. M on AB,
3. Perpendicular from M to CM intersects AD at E,
4. P is the foot of the perpendicular from M to CE
-/
variables (ABCD_is_rectangle : rectangle (A B C D))
          (M_on_AB: M)
          (Perpendicular_M_to_CM: M → E)
          (E_on_AD: E → P)
          (Perpendicular_M_to_CE: P)

-- Define the angle APB in terms of the angle configuration
def angle_APB := sorry

theorem angle_APB_is_90_degrees :
  ∀ (A B C D M E P : Type) (ABCD_is_rectangle : rectangle (A B C D))
    (M_on_AB: M) (Perpendicular_M_to_CM: M → E) (E_on_AD: E → P) (P_foot_of_perpendicular : Perpendicular_M_to_CE),
    angle_APB = 90 := sorry

end angle_APB_is_90_degrees_l49_49417


namespace four_digit_numbers_count_l49_49247

theorem four_digit_numbers_count : ∃ n : ℕ, n = 9000 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 9999 ↔ x ∈ {1000, ..., 9999} := sorry

end four_digit_numbers_count_l49_49247


namespace number_of_four_digit_numbers_l49_49303

theorem number_of_four_digit_numbers : 
  let start := 1000 
  let end := 9999 
  end - start + 1 = 9000 := 
by 
  sorry

end number_of_four_digit_numbers_l49_49303


namespace symmetric_about_line_l49_49086

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x - 2)
noncomputable def g (x a : ℝ) : ℝ := f (x + a)

theorem symmetric_about_line (a : ℝ) : (∀ x, g x a = x + 1) ↔ a = 0 :=
by sorry

end symmetric_about_line_l49_49086


namespace greatest_digits_product_l49_49583

theorem greatest_digits_product :
  ∀ (n m : ℕ), (10000 ≤ n ∧ n ≤ 99999) → (1000 ≤ m ∧ m ≤ 9999) → 
    nat.digits 10 (n * m) ≤ 9 :=
by 
  sorry

end greatest_digits_product_l49_49583


namespace paint_time_l49_49882

theorem paint_time (n1 t1 n2 : ℕ) (k : ℕ) (h : n1 * t1 = k) (h1 : 5 * 4 = k) (h2 : n2 = 6) : (k / n2) = 10 / 3 :=
by {
  -- Proof would go here
  sorry
}

end paint_time_l49_49882


namespace greatest_number_of_digits_in_product_l49_49621

-- Definitions based on problem conditions
def greatest_5digit_number : ℕ := 99999
def greatest_4digit_number : ℕ := 9999

-- Number of digits in a number
def num_digits (n : ℕ) : ℕ := n.toString.length

-- Defining the product
def product : ℕ := greatest_5digit_number * greatest_4digit_number

-- Statement to prove
theorem greatest_number_of_digits_in_product :
  num_digits product = 9 :=
by
  sorry

end greatest_number_of_digits_in_product_l49_49621


namespace five_term_geometric_sequence_value_of_b_l49_49364

theorem five_term_geometric_sequence_value_of_b (a b c : ℝ) (h₁ : b ^ 2 = 81) (h₂ : a ^ 2 = b) (h₃ : 1 * a = a) (h₄ : c * c = c) :
  b = 9 :=
by 
  sorry

end five_term_geometric_sequence_value_of_b_l49_49364


namespace x_is_percent_w_l49_49342

def x_percent_w (x y w z : ℝ) : ℝ := (x / w) * 100

theorem x_is_percent_w (x y w z : ℝ) (h1 : x = 2.24 * y) (h2 : y = 0.70 * z) (h3 : w = 2.50 * z) : x_percent_w x y w z = 62.72 :=
by
  -- We assume the correctness of the conditions and the proof steps
  sorry

end x_is_percent_w_l49_49342


namespace arctan_sum_pi_div_two_l49_49813

theorem arctan_sum_pi_div_two :
  let α := Real.arctan (3 / 4)
  let β := Real.arctan (4 / 3)
  α + β = Real.pi / 2 := by
  sorry

end arctan_sum_pi_div_two_l49_49813


namespace four_digit_numbers_count_l49_49228

theorem four_digit_numbers_count : 
  let smallest := 1000
  let largest := 9999
  largest - smallest + 1 = 9000 :=
by
  let smallest := 1000
  let largest := 9999
  show largest - smallest + 1 = 9000 from sorry

end four_digit_numbers_count_l49_49228


namespace probability_equal_dice_show_numbers_5_dice_l49_49740

noncomputable def probability_equal_dice_show_numbers (n : ℕ) (k : ℕ) : ℚ :=
  (nat.choose n k) * (1/2)^n

theorem probability_equal_dice_show_numbers_5_dice :
  probability_equal_dice_show_numbers 5 2 + probability_equal_dice_show_numbers 5 3 = 5 / 8 := by
  sorry

end probability_equal_dice_show_numbers_5_dice_l49_49740


namespace greatest_product_digits_l49_49564

/-- Define a function to count the number of digits in a positive integer. -/
def num_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

/-- The greatest possible number of digits in the product of a 5-digit whole number
    and a 4-digit whole number is 9. -/
theorem greatest_product_digits :
  ∀ (a b : ℕ), (10000 ≤ a ∧ a < 100000) → (1000 ≤ b ∧ b < 10000) → num_digits (a * b) = 9 :=
by
  intros a b a_range b_range
  have ha : 99999 = 10^5 - 1 := by norm_num
  have hb : 9999 = 10^4 - 1 := by norm_num
  have hab : a * b ≤ 99999 * 9999 := by
    have ha' : a ≤ 99999 := a_range.2
    have hb' : b ≤ 9999 := b_range.2
    exact mul_le_mul ha' hb' (nat.zero_le b) (nat.zero_le a)
  have h_max_product : 99999 * 9999 = 10^9 - 100000 - 10000 + 1 := by norm_num
  have h_max_digits : num_digits ((10^9 : ℕ) - 100000 - 10000 + 1) = 9 := by
    norm_num [num_digits, nat.log10]
  exact eq.trans_le h_max_digits hab
  sorry

end greatest_product_digits_l49_49564


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49571

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49571


namespace cube_x_value_l49_49007

noncomputable def cube_side_len (x : ℝ) : ℝ := (8 * x) ^ (1 / 3)

lemma cube_volume (x : ℝ) : (cube_side_len x) ^ 3 = 8 * x :=
  by sorry

lemma cube_surface_area (x : ℝ) : 6 * (cube_side_len x) ^ 2 = 2 * x :=
  by sorry

theorem cube_x_value (x : ℝ) (hV : (cube_side_len x) ^ 3 = 8 * x) (hS : 6 * (cube_side_len x) ^ 2 = 2 * x) : x = sqrt 3 / 72 :=
  by sorry

end cube_x_value_l49_49007


namespace count_four_digit_numbers_l49_49287

-- Definition of the smallest four-digit number
def smallest_four_digit_number : ℕ := 1000

-- Definition of the largest four-digit number
def largest_four_digit_number : ℕ := 9999

-- The theorem stating the number of four-digit numbers
theorem count_four_digit_numbers : 
  largest_four_digit_number - smallest_four_digit_number + 1 = 9000 := by
  -- Provide the proof here
  sorry

end count_four_digit_numbers_l49_49287


namespace product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49580

def largest5DigitNumber : ℕ := 99999
def largest4DigitNumber : ℕ := 9999

def numDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

theorem product_of_largest_5_and_4_digit_numbers_has_9_digits : 
  numDigits (largest5DigitNumber * largest4DigitNumber) = 9 := 
by 
  sorry

end product_of_largest_5_and_4_digit_numbers_has_9_digits_l49_49580


namespace max_digits_of_product_l49_49543

theorem max_digits_of_product : nat.digits 10 (99999 * 9999) = 9 := sorry

end max_digits_of_product_l49_49543


namespace greatest_possible_number_of_digits_in_product_l49_49534

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49534


namespace jackson_meat_left_l49_49366

def initial_meat : ℝ := 20
def meat_used_for_meatballs (x : ℝ) : ℝ := x / 4
def meat_left_after_meatballs (x : ℝ) : ℝ := x - meat_used_for_meatballs x
def meat_used_for_spring_rolls (x : ℝ) : ℝ := 0.15 * x
def meat_left_after_spring_rolls (x : ℝ) : ℝ := x - meat_used_for_spring_rolls x
def pounds_to_kg (pounds : ℝ) : ℝ := pounds * 0.453592
def meat_used_for_stew : ℝ := pounds_to_kg 2
def meat_left_after_stew (x : ℝ) : ℝ := x - meat_used_for_stew
def meat_used_for_kebabs (x : ℝ) : ℝ := 0.1 * x
def meat_left_after_kebabs (x : ℝ) : ℝ :=
  let m := meat_left_after_meatballs x
  let n := meat_left_after_spring_rolls m
  let p := meat_left_after_stew n
  p - meat_used_for_kebabs p

theorem jackson_meat_left : meat_left_after_kebabs initial_meat ≈ 10.66 := 
  -- The proof is omitted
  sorry

end jackson_meat_left_l49_49366


namespace count_four_digit_numbers_l49_49305

theorem count_four_digit_numbers 
  (a b : ℕ) 
  (h1 : a = 1000) 
  (h2 : b = 9999) : 
  b - a + 1 = 9000 := 
by
  sorry

end count_four_digit_numbers_l49_49305


namespace greatest_possible_number_of_digits_in_product_l49_49521

noncomputable def maxDigitsInProduct (a b : ℕ) (ha : 10^4 ≤ a ∧ a < 10^5) (hb : 10^3 ≤ b ∧ b < 10^4) : ℕ :=
  let product := a * b
  Nat.digits 10 product |>.length

theorem greatest_possible_number_of_digits_in_product : ∀ (a b : ℕ), (10^4 ≤ a ∧ a < 10^5) → (10^3 ≤ b ∧ b < 10^4) → maxDigitsInProduct a b _ _ = 9 :=
by
  intros a b ha hb
  sorry

end greatest_possible_number_of_digits_in_product_l49_49521


namespace digits_of_product_of_max_5_and_4_digit_numbers_l49_49714

theorem digits_of_product_of_max_5_and_4_digit_numbers :
  ∃ n : ℕ, (n = (99999 * 9999)) ∧ (nat_digits n = 10) :=
by
  sorry

-- definition to calculate the number of digits of a natural number
def nat_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else nat.log10 n + 1

end digits_of_product_of_max_5_and_4_digit_numbers_l49_49714


namespace third_player_wins_l49_49741

theorem third_player_wins {players : Fin₈ → ℝ} 
  (h_scores_unique : Function.Injective players)
  (h_games_total : ∑ i, ∑ j : Fin₈, i ≠ j → 1/2 * (players i + players j) = 28) 
  (h_second_player_score : ∃ i j k l m : Fin₈, players m = max {players 0, players 1, players 2, players 3, players 4, players 5, players 6, players 7} 
    ∧ i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧ l = 1 
    ∧ players l = players i + players j + players k + players m)
    (h_score_range : ∀ i, 0 ≤ players i ∧ players i ≤ 7)
    : players 2 > players 6 :=
by 
  sorry

end third_player_wins_l49_49741


namespace number_of_ten_tuples_l49_49125

theorem number_of_ten_tuples :
  (∃ (x : Fin 10 → ℝ),
    (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + 
    (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + (x 7 - x 8)^2 + (x 8 - x 9)^2 + 
    (x 9)^2 = 5 / 22) ↔
  2 := 
sorry

end number_of_ten_tuples_l49_49125


namespace max_x_y_l49_49194

-- Definitions according to conditions
def length (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

def angle_O_A_O_B : ℝ := real.pi * (2/3)  -- 120 degrees in radians

def in_arc (O A B C : ℝ × ℝ) (theta : ℝ) : Prop :=
  ∃ θ, (θ ∈ set.Icc 0 real.pi) ∧
  C = (real.cos θ, real.sin θ)

-- Main theorem statement
theorem max_x_y (O A B C : ℝ × ℝ) (x y : ℝ) (θ : ℝ) 
  (h1 : length O A = 1) 
  (h2 : length O B = 1) 
  (h3 : angle_O_A_O_B = real.pi * (2/3)) 
  (h4 : in_arc O A B C θ) 
  (h5 : C = (x * A.1 + y * B.1, x * A.2 + y * B.2)) 
  : x + y ≤ 2 :=
  sorry

end max_x_y_l49_49194
