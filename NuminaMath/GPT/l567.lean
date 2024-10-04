import Mathlib
import Mathlib.Algebra.ArithMean
import Mathlib.Algebra.ArithmeticSequence
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Divisors.Factorization
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Periodic
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Limits.Basic
import Mathlib.Analysis.Optics.Elliptic
import Mathlib.Analysis.SpecialFunctions.Polynomials
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.Triangle.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Perm
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.Prime
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Polygon
import Mathlib.NumberTheory.GCD
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring
import algebra.module.basic
import data.nat.prime

namespace constants_sine_identity_l567_567867

theorem constants_sine_identity :
  ∃ (c d : ℝ), (∀ θ : ℝ, sin θ ^ 3 = c * sin (3 * θ) + d * sin θ) 
  ∧ c = -1 / 4 ∧ d = 3 / 4 :=
sorry

end constants_sine_identity_l567_567867


namespace cards_difference_l567_567259

theorem cards_difference (A : Finset ℕ) (hA : A.card = 26) (h_sub : A ⊆ Finset.range 101) :
  ∃ (a b ∈ A), a ≠ b ∧ (|a - b| = 1 ∨ |a - b| = 2 ∨ |a - b| = 3) :=
sorry

end cards_difference_l567_567259


namespace find_track_circumference_l567_567797

noncomputable def track_circumference : ℝ := 720

theorem find_track_circumference
  (A B : ℝ)
  (uA uB : ℝ)
  (h1 : A = 0)
  (h2 : B = track_circumference / 2)
  (h3 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = 150 / uB)
  (h4 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = (track_circumference - 90) / uA)
  (h5 : ∀ t : ℝ, A + uA * t = B + uB * t ↔ t = 1.5 * track_circumference / uA) :
  track_circumference = 720 :=
by sorry

end find_track_circumference_l567_567797


namespace set_intersection_complement_U_M_N_l567_567165

def U := {0, -1, -2, -3, -4}
def M := {0, -1, -2}
def N := {0, -3, -4}
def complement_U_M := U \ M

theorem set_intersection_complement_U_M_N :
  (complement_U_M ∩ N) = \{-3, -4\} :=
by sorry

end set_intersection_complement_U_M_N_l567_567165


namespace problem1_problem2_l567_567069

theorem problem1 (a b c : V) (h_basis : LinearIndependent ℝ ![a, b, c]) :
  let OM := 2•a + b - c
  let OA := 3•a + 3•b
  let OB := 2•a + 4•b + 2•c
  let OC := -a + 2•b + 3•c
  ∃ k l m : ℝ, OM = k • OB + l • OA + m • OC := sorry

theorem problem2 (a b c : V) (h_basis : LinearIndependent ℝ ![a, b, c]) :
  let OA := 3•a + 3•b
  let OB := 2•a + 4•b + 2•c
  let OC := -a + 2•b + 3•c
  ¬LinearIndependent ℝ ![OA, OB, OC] := sorry

end problem1_problem2_l567_567069


namespace side_length_b_l567_567560

theorem side_length_b (A C : ℝ) (a c b : ℝ)
  (h1 : ∠C = 4 * ∠A)
  (h2 : a = 15)
  (h3 : c = 60) :
  b = 15 * sqrt (2 + sqrt 2) :=
sorry

end side_length_b_l567_567560


namespace find_angle_ACB_l567_567128

open EuclideanGeometry

variable (A B C D : Point)
variable (h_parallel : ParallelLine DC AB)
variable (h_angle_DCA : Angle DCA = 55º)
variable (h_angle_ABC : Angle ABC = 60º)

theorem find_angle_ACB :
  Angle ACB = 65º :=
by
  sorry

end find_angle_ACB_l567_567128


namespace polar_coordinates_of_point_l567_567383

open Real

theorem polar_coordinates_of_point :
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ 
  (r = 2) ∧ (θ = 7 * π / 6) ∧ 
  (- sqrt 3, 1) = (r * cos θ, r * sin θ) := by
  use [2, 7 * π / 6]
  have h1 : 2 > 0 := by norm_num
  have h2 : 0 ≤ 7 * π / 6 := by norm_num
  have h3 : 7 * π / 6 < 2 * π := by norm_num
  have h4 : 2 = 2 := by norm_num
  have h5 : 7 * π / 6 = 7 * π / 6 := by norm_num
  have h6 : (- sqrt 3, 1) = (2 * cos (7 * π / 6), 2 * sin (7 * π / 6)) := by sorry
  constructor; assumption; constructor; assumption; constructor; assumption; constructor; assumption; constructor; assumption; assumption

end polar_coordinates_of_point_l567_567383


namespace nonagon_diagonal_intersection_probability_l567_567987

-- Define a regular nonagon
def is_regular_nonagon (P : Type) [PlanarPolygon P] : Prop :=
  P.is_regular ∧ P.vertex_count = 9

-- The main theorem about the probability
theorem nonagon_diagonal_intersection_probability (P : Type) [PlanarPolygon P]
  (h : is_regular_nonagon P) :
  probability_of_intersecting_diagonals P = 14 / 39 := 
sorry

end nonagon_diagonal_intersection_probability_l567_567987


namespace number_of_incorrect_propositions_l567_567914

variables (l m n α β : Type) [LinearOrder l] [LinearOrder m] [LinearOrder n] [LinearOrder α] [LinearOrder β]

-- Define the conditions and propositions
def Prop1 (l m : l) (α : α) : Prop :=
  parallel_to_plane l α → perpendicular_to_plane m α → perpendicular_to l m

def Prop2 (m : m) (α β : α) : Prop :=
  perpendicular_to_plane m β → perpendicular_to_plane α β → not (in_plane m α) → parallel_to_plane m α

def Prop3 (l : l) (α β : α) : Prop :=
  parallel_planes α β → perpendicular_to_plane l α → perpendicular_to_plane l β

def Prop4 (m n : m) (α β : α) : Prop :=
  perpendicular_planes α β → in_plane m α → in_plane n β → perpendicular_to m n

-- The main statement
theorem number_of_incorrect_propositions :
  (∃ l m (α : α), Prop1 l m α) ∧
  (∃ m (α β : α), Prop2 m α β) ∧
  (∃ l (α β : α), Prop3 l α β) ∧
  (∃ m n (α β : α), Prop4 m n α β) →
  1 :=
sorry

end number_of_incorrect_propositions_l567_567914


namespace problem_xyz_l567_567140

noncomputable def distance_from_intersection_to_side_CD (s : ℝ) : ℝ :=
  s * ((8 - Real.sqrt 15) / 8)

theorem problem_xyz
  (s : ℝ)
  (ABCD_is_square : (0 ≤ s))
  (X_is_intersection: ∃ (X : ℝ × ℝ), (X.1^2 + X.2^2 = s^2) ∧ ((X.1 - s)^2 + X.2^2 = (s / 2)^2))
  : distance_from_intersection_to_side_CD s = (s * (8 - Real.sqrt 15) / 8) :=
sorry

end problem_xyz_l567_567140


namespace ratio_of_turtles_l567_567621

noncomputable def initial_turtles_owen : ℕ := 21
noncomputable def initial_turtles_johanna : ℕ := initial_turtles_owen - 5
noncomputable def turtles_johanna_after_month : ℕ := initial_turtles_johanna / 2
noncomputable def turtles_owen_after_month : ℕ := 50 - turtles_johanna_after_month

theorem ratio_of_turtles (a b : ℕ) (h1 : a = 21) (h2 : b = 5) (h3 : initial_turtles_owen = a) (h4 : initial_turtles_johanna = initial_turtles_owen - b) 
(h5 : turtles_johanna_after_month = initial_turtles_johanna / 2) (h6 : turtles_owen_after_month = 50 - turtles_johanna_after_month) : 
turtles_owen_after_month / initial_turtles_owen = 2 := by
  sorry

end ratio_of_turtles_l567_567621


namespace angle_AMC_is_70_l567_567141

theorem angle_AMC_is_70 (A B C M : Type) (angle_MBA angle_MAB angle_ACB : ℝ) (AC BC : ℝ) :
  AC = BC → 
  angle_MBA = 30 → 
  angle_MAB = 10 → 
  angle_ACB = 80 → 
  ∃ angle_AMC : ℝ, angle_AMC = 70 :=
by
  sorry

end angle_AMC_is_70_l567_567141


namespace line_circle_intersect_not_center_l567_567676

def line (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 2)/2 * t, 2 + (Real.sqrt 2)/2 * t)
def circle (θ : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

theorem line_circle_intersect_not_center :
  ((∃ (t θ : ℝ), line t = circle θ) ∧ ¬ line t = (2, 1)) := 
sorry

end line_circle_intersect_not_center_l567_567676


namespace number_of_integer_solutions_l567_567954

open Real

theorem number_of_integer_solutions (π_approx : ℝ) (hπ : π_approx = 3.14) :
  {x : ℤ | abs x < 3 * π_approx}.to_finset.card = 19 := 
sorry

end number_of_integer_solutions_l567_567954


namespace sum_ends_with_1379_l567_567620

theorem sum_ends_with_1379 (S : Set ℕ) (hS : S ⊆ {n | n % 2 = 1 ∧ n % 5 ≠ 0} ∧ S.card = 10000) : 
  ∃ (T : Finset ℕ), T ⊆ S ∧ (T.sum id % 10000 = 1379) :=
begin
  sorry
end

end sum_ends_with_1379_l567_567620


namespace median_length_l567_567549

theorem median_length (DE DF : ℝ) (HDE : DE = 6) (HDF : DF = 8)
  (Hright : ∀ D E F: Type, ∃ DFE : E = F → ∀ D E F: Type → D E F : Prop, DFE → D)
  (Hmidpoint : ∀ E F N: Type, ∃ mn : E = F → ∀ E F:Type, E F: Prop, mn := 1/2):
  (DN : ℝ) = 5 := by
  sorry

end median_length_l567_567549


namespace sum_of_divisors_143_l567_567722

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l567_567722


namespace range_of_a_plus_2014b_l567_567950

theorem range_of_a_plus_2014b (a b : ℝ) (h1 : a < b) (h2 : |(Real.log a) / (Real.log 2)| = |(Real.log b) / (Real.log 2)|) :
  ∃ c : ℝ, c > 2015 ∧ ∀ x : ℝ, a + 2014 * b = x → x > 2015 := by
  sorry

end range_of_a_plus_2014b_l567_567950


namespace seq_diff_l567_567508

theorem seq_diff : 
  (∀ n : ℕ, S n = n^2 - 4 * n) → 
  (∃ (a : ℕ → ℕ), ∀ n, S n = ∑ i in finset.range n, a i) → 
  (a 2 - a 1 = 2) :=
by
  sorry

end seq_diff_l567_567508


namespace find_cosine_of_angle_subtraction_l567_567903

variable (α : ℝ)
variable (h : Real.sin ((Real.pi / 6) - α) = 1 / 3)

theorem find_cosine_of_angle_subtraction :
  Real.cos ((2 * Real.pi / 3) - α) = -1 / 3 :=
by
  exact sorry

end find_cosine_of_angle_subtraction_l567_567903


namespace polynomial_prob_mn_sum_eq_fifteen_l567_567602

theorem polynomial_prob_mn_sum_eq_fifteen :
  let b := set.Icc (-20 : ℝ) 20
  let equation := λ x b, x^4 + 36 * b^2 = (6 * b^2 - 15 * b) * x^2
  let interval_lengths := (0 - (-20)) + (9 - 5) + (20 - 9)
  let total_length := 20 - (-20)
  let prob := interval_lengths / total_length
  let m := 7
  let n := 8
  (m / n) = prob → (m + n) = 15 := 
by 
  sorry

end polynomial_prob_mn_sum_eq_fifteen_l567_567602


namespace range_of_k_l567_567559

noncomputable def sequence (n k: ℕ) : ℕ := n^2 - k * n

theorem range_of_k (k : ℤ) (h : ∀ (n : ℕ), n > 0 → sequence n k ≥ sequence 3 k) : 5 ≤ k ∧ k ≤ 7 :=
by
  sorry

end range_of_k_l567_567559


namespace two_digit_integers_count_l567_567500

def digits : Set ℕ := {3, 5, 7, 8, 9}

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem two_digit_integers_count : 
  ∃ (count : ℕ), count = 16 ∧
  (∀ (t : ℕ), t ∈ digits → 
  ∀ (u : ℕ), u ∈ digits → 
  t ≠ u ∧ is_odd u → 
  (∃ n : ℕ, 10 * t + u = n)) :=
by
  -- The total number of unique two-digit integers is 16
  use 16
  -- Proof skipped
  sorry

end two_digit_integers_count_l567_567500


namespace largest_lambda_inequality_l567_567890

theorem largest_lambda_inequality : 
  ∃ λ : ℝ, λ = 3/2 ∧ ∀ (a : Fin 2019 → ℝ), 
    (∑ i in Finset.range 2019, (a i)^2) 
    ≥ (∑ i in Finset.range 1008, (a i) * (a (i + 1))) 
    + λ * (a 1008) * (a 1009) 
    + λ * (a 1009) * (a 1010) 
    + (∑ i in Finset.range (1008) \ (1009, 1010), (a (i + 2)) * (a (i + 3))) := sorry

end largest_lambda_inequality_l567_567890


namespace total_heads_l567_567339

theorem total_heads (h : ℕ) (c : ℕ) (total_feet : ℕ) 
  (h_count : h = 30)
  (hen_feet : h * 2 + c * 4 = total_feet)
  (total_feet_val : total_feet = 140) 
  : h + c = 50 :=
by
  sorry

end total_heads_l567_567339


namespace least_lcm_possible_l567_567675

theorem least_lcm_possible (a b c : ℕ) (h1 : Nat.lcm a b = 24) (h2 : Nat.lcm b c = 18) : Nat.lcm a c = 12 :=
sorry

end least_lcm_possible_l567_567675


namespace total_popsicle_sticks_l567_567047

def Gino_popsicle_sticks : ℕ := 63
def My_popsicle_sticks : ℕ := 50
def Nick_popsicle_sticks : ℕ := 82

theorem total_popsicle_sticks : Gino_popsicle_sticks + My_popsicle_sticks + Nick_popsicle_sticks = 195 := by
  sorry

end total_popsicle_sticks_l567_567047


namespace gcd_4557_1953_5115_l567_567030

def problem_conditions : Prop := (4557 > 0) ∧ (1953 > 0) ∧ (5115 > 0)

theorem gcd_4557_1953_5115 : Int.gcd (Int.gcd 4557 1953) 5115 = 93 := by
  have h1 : problem_conditions := by
    -- Since 4557, 1953, and 5115 are all greater than 0, we have:
    sorry
  -- Use the Euclidean algorithm to find the GCD of the numbers 4557, 1953, and 5115.
  sorry

end gcd_4557_1953_5115_l567_567030


namespace sum_of_divisors_143_l567_567735

theorem sum_of_divisors_143 : 
  (finset.range (144)).filter (λ d, 143 % d = 0).sum = 168 :=
by
  sorry

end sum_of_divisors_143_l567_567735


namespace fitted_bowling_ball_volume_l567_567804

-- Definitions of the conditions
def radius_ball := 12 -- cm
def volume_sphere (r : ℝ) := (4 / 3) * Math.pi * r^3

def radius_hole1 := 0.75 -- cm
def depth_hole1 := 6 -- cm
def volume_cylinder (r h : ℝ) := Math.pi * r^2 * h

def radius_hole3 := 1.25 -- cm
def depth_hole3 := 6 -- cm

-- Volumes of individual components
def volume_ball := volume_sphere radius_ball
def volume_hole1_and_2 := 2 * (volume_cylinder radius_hole1 depth_hole1)
def volume_hole3 := volume_cylinder radius_hole3 depth_hole3
def volume_holes := volume_hole1_and_2 + volume_hole3

-- Required volume of the fitted bowling ball
def fitted_volume := volume_ball - volume_holes

-- Statement to be proved
theorem fitted_bowling_ball_volume : 
  fitted_volume = 2287.875 * Math.pi := by
  sorry

end fitted_bowling_ball_volume_l567_567804


namespace expected_pairs_of_red_in_circle_deck_l567_567651

noncomputable def expected_pairs_of_adjacent_red_cards (deck_size : ℕ) (red_cards : ℕ) : ℚ :=
  let adjacent_probability := (red_cards - 1 : ℚ) / (deck_size - 1)
  in red_cards * adjacent_probability

theorem expected_pairs_of_red_in_circle_deck :
  expected_pairs_of_adjacent_red_cards 52 26 = 650 / 51 :=
by
  sorry

end expected_pairs_of_red_in_circle_deck_l567_567651


namespace set_intersection_example_l567_567094

open Set

theorem set_intersection_example :
  let A := {2, 4, 6, 8} 
  let B := {x ∈ (Icc 3 6)}
  A ∩ B = {4, 6} := 
  sorry

end set_intersection_example_l567_567094


namespace range_of_a_if_monotonic_l567_567947

theorem range_of_a_if_monotonic :
  (∀ x : ℝ, 1 < x ∧ x < 2 → 3 * a * x^2 - 2 * x + 1 ≥ 0) → a > 1 / 3 :=
by
  sorry

end range_of_a_if_monotonic_l567_567947


namespace price_of_each_cake_is_correct_l567_567467

-- Define the conditions
def total_flour : ℕ := 6
def flour_for_cakes : ℕ := 4
def flour_per_cake : ℚ := 0.5
def remaining_flour := total_flour - flour_for_cakes
def flour_per_cupcake : ℚ := 1 / 5
def total_earnings : ℚ := 30
def cupcake_price : ℚ := 1

-- Number of cakes and cupcakes
def number_of_cakes := flour_for_cakes / flour_per_cake
def number_of_cupcakes := remaining_flour / flour_per_cupcake

-- Earnings from cupcakes
def earnings_from_cupcakes := number_of_cupcakes * cupcake_price

-- Earnings from cakes
def earnings_from_cakes := total_earnings - earnings_from_cupcakes

-- Price per cake
def price_per_cake := earnings_from_cakes / number_of_cakes

-- Final statement to prove
theorem price_of_each_cake_is_correct : price_per_cake = 2.50 := by
  sorry

end price_of_each_cake_is_correct_l567_567467


namespace perp_sum_eq_perimeter_l567_567138

-- Define the triangle and its right angle
variables {A B C D E M : Type} [Metric_Space Type] [GeoSpace Type]

-- Define properties for points and midpoints
def is_right_triangle (A B C : Triangle) := angle C = 90

def is_midpoint (D : Type) (AC : Segment) := AD = DC
def is_midpoint (E : Type) (BC : Segment) := BE = EC
def is_midpoint (M : Type) (AB : Segment) := AM = MB

-- Define angle relationships
def angle_equal (ABC ADC : Angle) := ∠ABC = ∠ADC
def angle_equal (BAC BEC : Angle) := ∠BAC = ∠BEC

-- The property we need to prove
theorem perp_sum_eq_perimeter 
  (h_right_triangle : is_right_triangle A B C)
  (h_midpoint_D : is_midpoint D AC)
  (h_midpoint_E : is_midpoint E BC)
  (h_midpoint_M : is_midpoint M AB)
  (h_angle_ADC : angle_equal ∠ABC ∠ADC)
  (h_angle_BEC : angle_equal ∠BAC ∠BEC) :
  DM + ME = AB + BC + CA :=
sorry

end perp_sum_eq_perimeter_l567_567138


namespace fifth_term_arithmetic_sequence_l567_567255

theorem fifth_term_arithmetic_sequence (a d : ℤ) 
  (h_twentieth : a + 19 * d = 12) 
  (h_twenty_first : a + 20 * d = 16) : 
  a + 4 * d = -48 := 
by sorry

end fifth_term_arithmetic_sequence_l567_567255


namespace p_suff_not_necess_q_l567_567919

def proposition_p (a : ℝ) : Prop := ∀ (x : ℝ), x > 0 → (3*a - 1)^x < 1
def proposition_q (a : ℝ) : Prop := a > (1 / 3)

theorem p_suff_not_necess_q : 
  (∀ (a : ℝ), proposition_p a → proposition_q a) ∧ (¬∀ (a : ℝ), proposition_q a → proposition_p a) :=
  sorry

end p_suff_not_necess_q_l567_567919


namespace quadrilateral_inscribed_circle_tangent_segments_diff_l567_567341

theorem quadrilateral_inscribed_circle_tangent_segments_diff :
  ∀ (m n : ℕ), 
    (∃ (AB BC CD DA : ℕ),
      AB = 60 ∧ 
      BC = 110 ∧ 
      CD = 140 ∧ 
      DA = 90 ∧ 
      ∃ (circle_inscribed : bool),
        circle_inscribed = true ∧
        m + n = 140) →
    |m - n| = 120 :=
by
  intros m n h
  cases h with AB h
  cases h with BC h
  cases h with CD h
  cases h with DA h
  cases h with hAB h
  cases h with hBC h
  cases h with hCD h
  cases h with hDA h
  cases h with circle_inscribed h
  cases h with h_inscribed h_sum
  sorry

end quadrilateral_inscribed_circle_tangent_segments_diff_l567_567341


namespace largest_unreachable_amount_largest_amount_cannot_be_purchased_l567_567845

theorem largest_unreachable_amount :
  ∀ (n : ℕ), n >= 25 → (∃ (x y z : ℕ), n = 5 * x + 8 * y + 12 * z) :=
begin
  sorry
end

theorem largest_amount_cannot_be_purchased (n : ℕ) :
  (¬∃ (x y z : ℕ), 19 = 5 * x + 8 * y + 12 * z) :=
begin
  sorry
end

end largest_unreachable_amount_largest_amount_cannot_be_purchased_l567_567845


namespace ellipse_properties_l567_567940

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a = 2 ∧ b = sqrt 3 ∧ ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

noncomputable def triangle_area : Prop :=
  ∀ (F1 F2 : ℝ×ℝ) (P : ℝ×ℝ), F1 = (-1, 0) ∧ F2 = (1, 0) ∧ 
  P ∈ { (x, y) | x^2 / 4 + y^2 / 3 = 1 } ∧
  ∃ (θ : ℝ), θ = π / 3 ∧
  let PF1 := dist P F1,
      PF2 := dist P F2 in
  ∃ (S : ℝ), S = (1 / 2) * PF1 * PF2 * sin θ ∧ S = sqrt 3

theorem ellipse_properties :
  ellipse_equation ∧ triangle_area :=
by
  -- The proofs are omitted as required by the problem statement
  sorry

end ellipse_properties_l567_567940


namespace sin_lt_alpha_lt_tan_l567_567788

theorem sin_lt_alpha_lt_tan {α : ℝ} (h₁ : 0 < α) (h₂ : α < real.pi / 2) : 
  real.sin α < α ∧ α < real.tan α :=
sorry

end sin_lt_alpha_lt_tan_l567_567788


namespace largest_power_dividing_factorial_l567_567856

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 2015) : ∃ k : ℕ, (2015^k ∣ n!) ∧ k = 67 :=
by
  sorry

end largest_power_dividing_factorial_l567_567856


namespace projection_calculation_l567_567683

-- Define the vectors and the projection condition
def v1 := ⟨2, -4⟩ : ℝ × ℝ
def v2 := ⟨-1, 5⟩ : ℝ × ℝ
def proj_v1 := ⟨3/5, -4/5⟩ : ℝ × ℝ

-- Prove the projection
theorem projection_calculation :
  (∃ w : ℝ × ℝ, (w ≠ ⟨0, 0⟩ ∧ ∀ v : ℝ × ℝ, (v = v1 → v ⋅ w = proj_v1 ⋅ w)) →
   let proj_v2 = (v2 ⋅ w) / (w ⋅ w) • w
   in proj_v2 = ⟨-69/25, 92/25⟩) := by
sorry

end projection_calculation_l567_567683


namespace equal_tangents_l567_567159

noncomputable theory
open_locale classical

-- Definitions based on given problem conditions
variables {A B C H A1 B1 C1 : Type*} [metric_space H]

-- Problem-specific definitions and assumptions
variables (triangle_ABC : triangle A B C)
variables (H_orthocenter : ortho_triangle H triangle_ABC)
variables (A1_on_BC : on_side A1 B C)
variables (B1_on_AC : on_side B1 A C)
variables (C1_on_AB : on_side C1 A B)

-- Statement of the theorem
theorem equal_tangents :
  let circle_AA1 := circle_through_diameter A A1,
      circle_BB1 := circle_through_diameter B B1,
      circle_CC1 := circle_through_diameter C C1 in
  tangent_length_from_point H circle_AA1 = tangent_length_from_point H circle_BB1 ∧
  tangent_length_from_point H circle_BB1 = tangent_length_from_point H circle_CC1 :=
begin
  sorry -- Proof goes here
end

end equal_tangents_l567_567159


namespace birthday_height_l567_567619

def previous_year_height : ℝ := 119.7
def increase_rate : ℝ := 5 / 100

theorem birthday_height : previous_year_height * (1 + increase_rate) = 125.685 :=
by
  sorry

end birthday_height_l567_567619


namespace min_omega_even_function_l567_567628

theorem min_omega_even_function (ω: ℝ) (h1: ω > 0) (g: ℝ → ℝ) 
    (h2: ∀ x: ℝ, g x = sin (ω * x - (π * ω / 3) + (π / 6)))
    (h3: ∀ x: ℝ, g x = g (-x)) : ω = 2 :=
    sorry  -- Proof omitted

end min_omega_even_function_l567_567628


namespace johns_cost_per_sheet_equals_2_75_l567_567831

variable (J : ℝ)

-- Conditions
def johns_total_cost (n : ℕ) : ℝ := n * J + 125
def sams_total_cost (n : ℕ) : ℝ := n * 1.5 + 140

-- Problem statement
theorem johns_cost_per_sheet_equals_2_75 :
  (johns_total_cost 12 = sams_total_cost 12) → J = 2.75 := by
  sorry

end johns_cost_per_sheet_equals_2_75_l567_567831


namespace num_ordered_quadruples_l567_567894

theorem num_ordered_quadruples :
  (∃ (a b c d : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a^2 + b^2 + c^2 + d^2 = 9 ∧ 
    (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81) ↔
  (5 ≤ {t : ℝ × ℝ × ℝ × ℝ | let (a, b, c, d) := t in 
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a^2 + b^2 + c^2 + d^2 = 9 ∧ 
    (a + b + c + d) * (a^3 + b^3 + c^3 + d^3) = 81}.to_finset.card) := sorry

end num_ordered_quadruples_l567_567894


namespace part_1_min_value_part_1_max_value_part_2_range_of_a_l567_567501

noncomputable def f (x : ℝ) (a : ℝ) := 4 * Real.log x - a * x + (a + 3) / x

noncomputable def g (x : ℝ) (a : ℝ) := 2 * Real.exp x - 4 * x + 2 * a

theorem part_1_min_value :
  ∀ x : ℝ, a = 1 / 2 → f x a = 3 → x = 1 :=
by
  intro x a h_eq
  have ha : a = 1 / 2 := by assumption
  sorry

theorem part_1_max_value :
  ∀ x : ℝ, a = 1 / 2 → f x a = 4 * Real.log 7 - 3 → x = 7 :=
by
  intro x a h_eq
  have ha : a = 1 / 2 := by assumption
  sorry

theorem part_2_range_of_a (a : ℝ) :
  (∃ x₁ x₂ ∈ Icc (1 / 2) 2, f x₁ a > g x₂ a) → 1 ≤ a ∧ a < 4 :=
by
  intro h
  have ha : a ≥ 1 := by assumption
  sorry

end part_1_min_value_part_1_max_value_part_2_range_of_a_l567_567501


namespace find_values_l567_567046

noncomputable def equation_satisfaction (x y : ℝ) : Prop :=
  x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3

theorem find_values (x y : ℝ) :
  equation_satisfaction x y → x = 1 / 3 ∧ y = 2 / 3 :=
by
  intro h
  sorry

end find_values_l567_567046


namespace men_in_business_class_l567_567877

def passengers_total : ℕ := 300
def men_percentage : ℝ := 0.8
def business_percentage : ℝ := 0.2

theorem men_in_business_class :
  (passengers_total * men_percentage * business_percentage).toNat = 48 := 
  sorry

end men_in_business_class_l567_567877


namespace average_student_headcount_l567_567001

theorem average_student_headcount :
  let hc_03_04 := 11500
  let hc_04_05 := 11600
  let hc_05_06 := 11300
  (Int.round (((hc_03_04 + hc_04_05 + hc_05_06) / 3 : Float))) = 11467 :=
by
  sorry

end average_student_headcount_l567_567001


namespace fx_is_odd_l567_567674

def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem fx_is_odd : ∀ x : ℝ, f (-x) = - f x := by
  sorry

end fx_is_odd_l567_567674


namespace group_a_mats_in_12_days_group_b_mats_in_12_days_group_c_mats_in_12_days_l567_567265

def mats_weaved (weavers mats days : ℕ) : ℕ :=
  (mats / days) * weavers

theorem group_a_mats_in_12_days (mats_req : ℕ) :
  let weavers := 4
  let mats_per_period := 4
  let period_days := 4
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

theorem group_b_mats_in_12_days (mats_req : ℕ) :
  let weavers := 6
  let mats_per_period := 9
  let period_days := 3
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

theorem group_c_mats_in_12_days (mats_req : ℕ) :
  let weavers := 8
  let mats_per_period := 16
  let period_days := 4
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

end group_a_mats_in_12_days_group_b_mats_in_12_days_group_c_mats_in_12_days_l567_567265


namespace population_of_village_l567_567319

-- Define the given condition
def total_population (P : ℝ) : Prop :=
  0.4 * P = 23040

-- The theorem to prove that the total population is 57600
theorem population_of_village : ∃ P : ℝ, total_population P ∧ P = 57600 :=
by
  sorry

end population_of_village_l567_567319


namespace total_strictly_monotonous_positive_integers_l567_567375

def is_strictly_monotonous (n : ℕ) : Prop :=
  (n > 0) ∧
  (n < 10 ∨
    ((list.of_digits (n.digits 10)).nodup ∧
     (list.of_digits (n.digits 10)).pairwise (<) ∨
     (list.of_digits (n.digits 10)).pairwise (>)))

def number_of_strictly_monotonous_numbers : ℕ :=
  9 + ∑ n in finset.range (10 - 2 + 1), 2 * nat.choose 9 n

theorem total_strictly_monotonous_positive_integers: number_of_strictly_monotonous_numbers = 1013 :=
by
  -- Proof would go here
  sorry

end total_strictly_monotonous_positive_integers_l567_567375


namespace number_of_integral_solutions_l567_567596

theorem number_of_integral_solutions : 
  (∃! (s : Finset (ℤ × ℤ × ℤ × ℤ)), ∀ (x y z w : ℤ), (x, y, z, w) ∈ s 
  ↔ (x^2 + y^2 + z^2 + w^2 = 3 * (x + y + z + w))) → s.card = 208 :=
by
  sorry

end number_of_integral_solutions_l567_567596


namespace massive_crate_chocolate_bars_l567_567815

theorem massive_crate_chocolate_bars :
  (54 * 24 * 37 = 47952) :=
by
  sorry

end massive_crate_chocolate_bars_l567_567815


namespace exp_neg_sum_l567_567531

theorem exp_neg_sum (θ φ : ℝ) (h : complex.exp (complex.I * θ) + complex.exp (complex.I * φ) = (-2/3 : ℂ) + (1/9 : ℂ) * complex.I) :
  complex.exp (-complex.I * θ) + complex.exp (-complex.I * φ) = (-2/3 : ℂ) - (1/9 : ℂ) * complex.I :=
by sorry

end exp_neg_sum_l567_567531


namespace no_exist_k_r_m_l567_567185

open BigOperators

def seq_a : ℕ → ℤ
| 0       := 1
| (n + 1) := 9 * (seq_a n) - 2 * (seq_b n)

def seq_b : ℕ → ℤ
| 0       := 1
| (n + 1) := 2 * (seq_a n) + 4 * (seq_b n)

def seq_c (n : ℕ) : ℤ :=
  seq_a n + seq_b n

theorem no_exist_k_r_m (k r m : ℕ) (hk : 0 < k) (hr : 0 < r) (hm : 0 < m) : ¬(seq_c r ^ 2 = seq_c k * seq_c m) := 
sorry

end no_exist_k_r_m_l567_567185


namespace final_payment_order_450_l567_567846

noncomputable def finalPayment (orderAmount : ℝ) : ℝ :=
  let serviceCharge := if orderAmount < 500 then 0.04 * orderAmount
                      else if orderAmount < 1000 then 0.05 * orderAmount
                      else 0.06 * orderAmount
  let salesTax := if orderAmount < 500 then 0.05 * orderAmount
                  else if orderAmount < 1000 then 0.06 * orderAmount
                  else 0.07 * orderAmount
  let totalBeforeDiscount := orderAmount + serviceCharge + salesTax
  let discount := if totalBeforeDiscount < 600 then 0.05 * totalBeforeDiscount
                  else if totalBeforeDiscount < 800 then 0.10 * totalBeforeDiscount
                  else 0.15 * totalBeforeDiscount
  totalBeforeDiscount - discount

theorem final_payment_order_450 :
  finalPayment 450 = 465.98 := by
  sorry

end final_payment_order_450_l567_567846


namespace roots_of_equation_l567_567462

theorem roots_of_equation : 
  ∀ x : ℝ,
  3 * Real.sqrt x + 3 * x^(-1/2) = 7 ↔ 
  x = ( (7 + Real.sqrt 13) / 6 )^2 ∨ x = ( (7 - Real.sqrt 13) / 6 )^2 :=
by
  sorry

end roots_of_equation_l567_567462


namespace problem_statement_l567_567886

theorem problem_statement (x : ℝ) (h : x > 6) : 
  (sqrt (x - 6 * sqrt (x - 6)) + 3 = sqrt (x + 6 * sqrt (x - 6)) - 3) ↔ (x ∈ Set.Ici 18) :=
sorry

end problem_statement_l567_567886


namespace sum_of_divisors_143_l567_567757

theorem sum_of_divisors_143 : ∑ d in finset.filter (λ d, 143 % d = 0) (finset.range 144), d = 168 :=
by sorry

end sum_of_divisors_143_l567_567757


namespace prob_three_cards_hearts_king_spade_l567_567269

noncomputable def probability_hearts_king_spade : ℚ :=
  let total_cards := 52
  let total_hearts := 13
  let hearts_kings := 1 -- one king in hearts
  let rest_hearts := 12 -- remaining hearts cards
  let others_suit := 39 -- cards that are not hearts
  let total_kings := 4 -- total number of kings
  let spades := 13 -- total number of spades
  
  -- Probability of the first card is a hearts suit
  let P_first_hearts : ℚ := total_hearts / total_cards
  
  -- Probability of the first card is king of hearts
  let P_first_king_hearts : ℚ := hearts_kings / total_cards
  
  -- Probability of the second card is a king (excluding already drawn king if king of hearts is drawn first)
  let P_second_king_given_first_king_hearts : ℚ := (total_kings - hearts_kings) / (total_cards - 1)
  
  -- Probability of third card being a spade if the first king of hearts is drawn
  let P_third_spade_given_first_king_hearts : ℚ := spades / (total_cards - 2)
  
  -- Combine probabilities for the scenario where the first card is the king of hearts
  let case_king_hearts := P_first_king_hearts * P_second_king_given_first_king_hearts * P_third_spade_given_first_king_hearts
  
  -- Probability of the first card being a hearts (excluding king of hearts)
  let P_first_hearts_not_king : ℚ := rest_hearts / total_cards
  
  -- Probability of second card being any king
  let P_second_king_given_first_hearts_not_king : ℚ := total_kings / (total_cards - 1)
  
  -- Probability of third card being a spade
  let P_third_spade_given_first_hearts_not_king : ℚ := spades / (total_cards - 2)
  
  -- Combine probabilities for the scenario where the first card is a hearts but not the king
  let case_hearts_not_king := P_first_hearts_not_king * P_second_king_given_first_hearts_not_king * P_third_spade_given_first_hearts_not_king
  
  -- Total probability
  case_king_hearts + case_hearts_not_king
  
theorem prob_three_cards_hearts_king_spade : 
  probability_hearts_king_spade = 1 / 200 :=
by
  sorry

end prob_three_cards_hearts_king_spade_l567_567269


namespace taxes_ratio_l567_567610

theorem taxes_ratio
  (lottery_amount : ℕ)
  (remaining_amount : ℕ)
  (taxes : ℕ)
  (savings : ℕ)
  (invested : ℕ) :
  lottery_amount = 12006 →
  remaining_amount = 2802 →
  savings = 1000 →
  invested = 200 →
  lottery_amount - taxes - (1/3) * (lottery_amount - taxes) - savings - invested - remaining_amount = remaining_amount →
  taxes = 1800 ∧
  (taxes / 6) : (lottery_amount / 6) = 300 : 2001 :=
by
  intro h₁ h₂ h₃ h₄ h₅
  have h₆ : lottery_amount = 12006 := h₁
  have h₇ : remaining_amount = 2802 := h₂
  have h₈ : savings = 1000 := h₃
  have h₉ : invested = 200 := h₄
  have h₁₀ : taxes = 1800
    sorry
  have h₁₁ : (taxes / 6) : (lottery_amount / 6) = 300 : 2001
    sorry
  exact ⟨h₁₀, h₁₁⟩

end taxes_ratio_l567_567610


namespace function_properties_l567_567948

def f (x : ℝ) : ℝ := 4^x - 4^(-x)

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x y : ℝ, x < y → f (x) < f (y)) :=
by
  sorry

end function_properties_l567_567948


namespace concat_div_1980_l567_567299

def n : ℕ := read "192021…80"  -- pseudo-function to represent the concatenation

theorem concat_div_1980 : 1980 ∣ n :=
  sorry

end concat_div_1980_l567_567299


namespace expected_adjacent_red_pairs_correct_l567_567663

-- The deck conditions
def standard_deck : Type := {c : ℕ // c = 52}
def num_red_cards (d : standard_deck) := 26

-- Probability definition
def prob_red_right_of_red : ℝ := 25 / 51

-- Expected number of adjacent red pairs calculation
def expected_adjacent_red_pairs (n_red : ℕ) (prob_right_red : ℝ) : ℝ :=
  n_red * prob_right_red

-- Main theorem statement
theorem expected_adjacent_red_pairs_correct (d : standard_deck) :
  expected_adjacent_red_pairs (num_red_cards d) prob_red_right_of_red = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_correct_l567_567663


namespace max_abs_z_l567_567472

noncomputable def max_abs_value_within_circle (z : ℂ) (cond : |z - complex.I * 2| ≤ 1) : ℝ :=
  max (complex.abs z) sorry

theorem max_abs_z (z : ℂ) (cond : |z - complex.I * 2| ≤ 1) : complex.abs z ≤ 3 := sorry

end max_abs_z_l567_567472


namespace axis_of_symmetry_l567_567109

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x : ℝ, f x = f (4 - x)) : 
  ∀ x : ℝ, f x = f (4 - x) := 
  by sorry

end axis_of_symmetry_l567_567109


namespace smallest_m_l567_567291

theorem smallest_m (m : ℕ) (h_prime_m : prime 15) :
  (∃ x y : ℤ, 15 * x^2 - m * x + 315 = 0 ∧ x * y = 21 ∧ x + y = 10 ∧ m = 15 * (x + y)) → 
  m = 150 :=
by
  sorry

end smallest_m_l567_567291


namespace problem_solution_l567_567378

noncomputable def original_speed_is_9 (circumference : ℝ) (time_decrease : ℝ) (speed_increase : ℝ) : Prop :=
  let c_miles := circumference / 5280 in
  let t_original := c_miles / (r : ℝ) in
  let r_increased := r + speed_increase in
  let t_decreased := t_original - time_decrease / 3600 in
  r * t_original = (r_increased) * t_decreased ∧ r = 9

theorem problem_solution :
  original_speed_is_9 15 (1 / 3) 6 :=
sorry

end problem_solution_l567_567378


namespace cost_price_per_meter_l567_567148

theorem cost_price_per_meter (total_cost : ℝ) (total_length : ℝ) (h1 : total_cost = 397.75) (h2 : total_length = 9.25) : total_cost / total_length = 43 :=
by
  -- Proof omitted
  sorry

end cost_price_per_meter_l567_567148


namespace percentage_of_initial_salt_solution_l567_567958

theorem percentage_of_initial_salt_solution 
  (x : ℕ)
  (P : ℝ)
  (H1 : x = 40)
  (H2 : 0.60 * x + 0.01 * P * 40 = 0.40 * 80) : 
  P = 20 :=
begin
  -- Proof steps to be completed here
  sorry
end

end percentage_of_initial_salt_solution_l567_567958


namespace triangle_ABC_properties_l567_567539

theorem triangle_ABC_properties
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a ≠ b)
  (h2 : c = sqrt 3)
  (h3 : cos A ^ 2 - cos B ^ 2 = sqrt 3 * sin A * cos A - sqrt 3 * sin B * cos B)
  (h4 : sin A = sqrt 2 / 2)
  (h5 : A + B + C = π) :
  (C = π / 3) ∧ (1 / 2 * a * c * sin B = (3 + sqrt 3) / 4) :=
by
  sorry

end triangle_ABC_properties_l567_567539


namespace problem_statement_l567_567007

noncomputable def f : ℝ → ℝ :=
λ x, if x < 1 then 1 + Real.logb 2 (2 - x) else Real.rpow 2 (x - 1)

theorem problem_statement : f (-2) + f (Real.logb 2 12) = 9 := 
by 
  -- the proof will be placed here
  sorry

end problem_statement_l567_567007


namespace symmetric_graph_phi_l567_567949

theorem symmetric_graph_phi (x : ℝ) (ϕ : ℝ) :
  (∀ x, (sin (2 * x + ϕ) + 1) = (sin (2 * (- (π / 8)) + ϕ) + 1)) →
    ϕ = (3 * π / 4) :=
by
  sorry

end symmetric_graph_phi_l567_567949


namespace ratio_of_rooms_l567_567009

def rooms_in_danielle_apartment : Nat := 6
def rooms_in_heidi_apartment : Nat := 3 * rooms_in_danielle_apartment
def rooms_in_grant_apartment : Nat := 2

theorem ratio_of_rooms :
  (rooms_in_grant_apartment : ℚ) / (rooms_in_heidi_apartment : ℚ) = 1 / 9 := 
by 
  sorry

end ratio_of_rooms_l567_567009


namespace number_of_four_digit_mountain_numbers_l567_567865

-- Define what is a four-digit mountain number
def is_four_digit_mountain_number (n : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  n >= 1000 ∧ n < 10000 ∧ d2 = d3 ∧ d2 > d1 ∧ d2 > d4

-- The theorem to prove
theorem number_of_four_digit_mountain_numbers : 
  (Finset.range 10000).filter is_four_digit_mountain_number .card = 56 :=
by
  sorry

end number_of_four_digit_mountain_numbers_l567_567865


namespace fish_caught_together_l567_567154

theorem fish_caught_together (Blaines_fish Keiths_fish : ℕ) 
  (h1 : Blaines_fish = 5) 
  (h2 : Keiths_fish = 2 * Blaines_fish) : 
  Blaines_fish + Keiths_fish = 15 := 
by 
  sorry

end fish_caught_together_l567_567154


namespace root_implies_quadratic_eq_l567_567967

theorem root_implies_quadratic_eq (m : ℝ) (h : (m + 2) - 2 + m^2 - 2 * m - 6 = 0) : 
  2 * m^2 - m - 6 = 0 :=
sorry

end root_implies_quadratic_eq_l567_567967


namespace number_of_students_in_range_019_056_is_4_l567_567985

def systematic_sample_100_10_contains_003 (s : set ℕ) : Prop :=
  s = {n | ∃ k : ℕ, k < 10 ∧ n = 3 + 10 * k}

theorem number_of_students_in_range_019_056_is_4 :
  ∀ (s : set ℕ), systematic_sample_100_10_contains_003 s →
  let ss := s ∩ {n | 19 ≤ n ∧ n ≤ 56} in
  ss.card = 4 :=
by sorry

end number_of_students_in_range_019_056_is_4_l567_567985


namespace expected_adjacent_red_pairs_correct_l567_567664

-- The deck conditions
def standard_deck : Type := {c : ℕ // c = 52}
def num_red_cards (d : standard_deck) := 26

-- Probability definition
def prob_red_right_of_red : ℝ := 25 / 51

-- Expected number of adjacent red pairs calculation
def expected_adjacent_red_pairs (n_red : ℕ) (prob_right_red : ℝ) : ℝ :=
  n_red * prob_right_red

-- Main theorem statement
theorem expected_adjacent_red_pairs_correct (d : standard_deck) :
  expected_adjacent_red_pairs (num_red_cards d) prob_red_right_of_red = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_correct_l567_567664


namespace solution_set_l567_567507

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 1 else -1

theorem solution_set (x : ℝ) :
  x + (x + 2) * f(x + 2) ≤ 5 → x ≤ (3 : ℝ) / 2 :=
by
  intros h
  have : x + 2 ≥ 0 ∨ x + 2 < 0 := lt_or_ge (x + 2) 0
  cases this with h0 h1
  · have : f (x + 2) = -1 := by simp [f, h0]
    linarith
  · have : f (x + 2) = 1 := by simp [f, h1]
    linarith
  sorry

end solution_set_l567_567507


namespace smallest_n_for_large_area_l567_567859

noncomputable def area_of_triangle (n : ℕ) : ℝ :=
  let z1 := (n : ℂ) + complex.I
  let z2 := (n + 2 * complex.I) ^ 2
  let z3 := (n + 3 * complex.I) ^ 3
  (complex.abs (
    (z1.re * z2.im - z1.im * z2.re) +
    (z2.re * z3.im - z2.im * z3.re) +
    (z3.re * z1.im - z3.im * z1.re)
  ) / 2)

theorem smallest_n_for_large_area : ∃ (n : ℕ), 0 < n ∧ area_of_triangle n > 1000 ∧ n = 3 :=
by 
  sorry

end smallest_n_for_large_area_l567_567859


namespace B_subset_A_implies_m_le_5_l567_567098

variable (A B : Set ℝ)
variable (m : ℝ)

def setA : Set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
def setB (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 2}

theorem B_subset_A_implies_m_le_5 :
  B ⊆ A → (∀ k : ℝ, k ∈ setB m → k ∈ setA) → m ≤ 5 :=
by
  sorry

end B_subset_A_implies_m_le_5_l567_567098


namespace trapezoidRatioSum_l567_567465

open Real

variables (AD AO OB BC CD AB DO OC : ℝ)
variables (X Y : ℝ → ℝ)
variables (p q : ℕ)

# Conditions
noncomputable def isCongruentTriangles : Prop :=
  AD = 15 ∧ AO = 15 ∧ OB = 15 ∧ BC = 15 ∧ CD = 15 ∧
  AB = 16 ∧ DO = 16 ∧ OC = 16

# Point midpoints
def midpoint (a b : ℝ) : ℝ := (a + b) / 2

def conditions : Prop := 
  let O := (0 : ℝ, 0 : ℝ) in
  let A := (0 : ℝ, 15) in
  let B := (16 : ℝ, 15) in
  let C := (16 : ℝ, 0) in
  let D := (0 : ℝ, 0) in
  let X := midpoint 0 0 in
  let Y := midpoint 16 0 in
  true

# Proof statement
theorem trapezoidRatioSum (h₁ : isCongruentTriangles AD AO OB BC CD AB DO OC)
  (h₂ : conditions AD AO OB BC CD AB DO OC X Y) :
  p + q = 2 :=
sorry

end trapezoidRatioSum_l567_567465


namespace sum_of_divisors_143_l567_567717

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l567_567717


namespace terminating_decimals_count_l567_567044

theorem terminating_decimals_count : 
  (Finset.card (Finset.filter (λ n : ℕ, (1 ≤ n ∧ n ≤ 499)) (Finset.range 500))) = 499 :=
by
  sorry

end terminating_decimals_count_l567_567044


namespace value_of_s_l567_567157

theorem value_of_s :
  let S := (1 / (4 - Real.sqrt 15)) - (1 / (Real.sqrt 15 - Real.sqrt 14)) +
           (1 / (Real.sqrt 14 - Real.sqrt 13)) - (1 / (Real.sqrt 13 - Real.sqrt 12)) +
           (1 / (Real.sqrt 12 - 3))
  in S = 7 :=
by
  let S := (1 / (4 - Real.sqrt 15)) - (1 / (Real.sqrt 15 - Real.sqrt 14)) +
           (1 / (Real.sqrt 14 - Real.sqrt 13)) - (1 / (Real.sqrt 13 - Real.sqrt 12)) +
           (1 / (Real.sqrt 12 - 3))
  have h : S = 7, from sorry
  exact h

end value_of_s_l567_567157


namespace sum_of_divisors_143_l567_567760

theorem sum_of_divisors_143 : ∑ d in finset.filter (λ d, 143 % d = 0) (finset.range 144), d = 168 :=
by sorry

end sum_of_divisors_143_l567_567760


namespace remove_pot_37_l567_567004

def initial_pots : List Nat := [81, 71, 41, 37, 35]

def sum_pots (pots : List Nat) : Nat :=
  List.sum pots

theorem remove_pot_37 :
  ∃ (gold silver : List Nat), 
    let remaining_pots := List.erase initial_pots 37 in
    gold ++ silver = remaining_pots ∧
    sum_pots silver = 2 * sum_pots gold :=
sorry

end remove_pot_37_l567_567004


namespace sin2x_sin3x_eq_cos2x_cos3x_l567_567964

theorem sin2x_sin3x_eq_cos2x_cos3x (x : ℝ) (h : sin (2 * x) * sin (3 * x) = cos (2 * x) * cos (3 * x)) : x = (real.pi / 10) ∨ ∃ k : ℤ, x = (real.pi / 10) + k * real.pi :=
by
  sorry

end sin2x_sin3x_eq_cos2x_cos3x_l567_567964


namespace slope_angle_range_l567_567904

open Real

structure Point where
  x : ℝ
  y : ℝ

def slope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

def slope_angle (p1 p2 : Point) : ℝ :=
  arctan (slope p1 p2)

axiom arctan_range {m : ℝ} {α : ℝ} : slope_angle (Point.mk 0 0) (Point.mk m 1) = α → 
  0 ≤ α ∧ α ≤ π

theorem slope_angle_range :
  let A := Point.mk (-3) 4
  let B := Point.mk 3 2
  let P := Point.mk 1 0
  ∀ α : ℝ,
  (∃ m : ℝ, slope (Point.mk 1 m) P = α) →
  α ∈ set.Icc (arctan 1) (arctan (-1)) := sorry

end slope_angle_range_l567_567904


namespace angle_between_v_a_and_a_is_3π_div_4_l567_567916

def vector := (ℝ × ℝ)

def a : vector := (1, -2)
def b : vector := (-4, 3)
def a_plus_b : vector := (a.1 + b.1, a.2 + b.2)
def dot_product (u v : vector) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (v : vector) : ℝ := Real.sqrt (v.1^2 + v.2^2)
def angle_cos (u v : vector) : ℝ := dot_product u v / (magnitude u * magnitude v)
def θ : ℝ := Real.acos (angle_cos a_plus_b a)

theorem angle_between_v_a_and_a_is_3π_div_4 : θ = 3 * Real.pi / 4 :=
by
  -- skip the proof
  sorry

end angle_between_v_a_and_a_is_3π_div_4_l567_567916


namespace boys_on_trip_l567_567201

theorem boys_on_trip (B G : ℕ) 
    (h1 : G = B + (2 / 5 : ℚ) * B) 
    (h2 : 1 + 1 + 1 + B + G = 123) : 
    B = 50 := 
by 
  -- Proof skipped 
  sorry

end boys_on_trip_l567_567201


namespace abs_neg_1_l567_567641

theorem abs_neg_1.5_eq_1.5 : abs (-1.5) = 1.5 := 
by 
  -- Proof is omitted
  sorry

end abs_neg_1_l567_567641


namespace sqrt_equation_solution_l567_567888

theorem sqrt_equation_solution (x : ℝ) (h : x > 6) :
  (sqrt (x - 6 * sqrt (x - 6)) + 3 = sqrt (x + 6 * sqrt (x - 6)) - 3) ↔ x ∈ set.Ici 18 :=
sorry

end sqrt_equation_solution_l567_567888


namespace primes_in_sequence_l567_567216

open Int Nat

def is_prime (p : ℤ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def in_sequence (p : ℤ) : Prop := ∃ n : ℕ, p = Int.sqrt (24 * n + 1)

theorem primes_in_sequence (p : ℕ) (hp : is_prime p) (hp_ne_2 : p ≠ 2) (hp_ne_3 : p ≠ 3) :
  in_sequence p := by
  sorry

end primes_in_sequence_l567_567216


namespace area_of_figure_l567_567381

theorem area_of_figure : 
  let left_rectangle_area := 7 * 7 in
  let middle_rectangle_area := 3 * 2 in
  let right_rectangle_area := 4 * 4 in
  left_rectangle_area + middle_rectangle_area + right_rectangle_area = 71 :=
by
  sorry

end area_of_figure_l567_567381


namespace problem_l567_567504

noncomputable def f (a c x : ℝ) : ℝ := a * x^3 + c * x

theorem problem (x₁ x₂ a c : ℝ) (h1 : a ≠ 0) (h2 : f a c 1 = -2) (h3 : a + c = -2) (h4 : 3 * a + c = 0) 
  (hx1: x₁ ∈ Ioo (-1:ℝ) 1) (hx2: x₂ ∈ Ioo (-1:ℝ) 1) : 
  |f a c x₁ - f a c x₂| < 4 := 
sorry

end problem_l567_567504


namespace mandy_pieces_eq_fifteen_l567_567325

-- Define the initial chocolate pieces
def total_pieces := 60

-- Define Michael's share
def michael_share := total_pieces / 2

-- Define the remainder after Michael's share
def remainder_after_michael := total_pieces - michael_share

-- Define Paige's share
def paige_share := remainder_after_michael / 2

-- Define the remainder after Paige's share
def mandy_share := remainder_after_michael - paige_share

-- Theorem to prove Mandy gets 15 pieces
theorem mandy_pieces_eq_fifteen : mandy_share = 15 :=
by
  sorry

end mandy_pieces_eq_fifteen_l567_567325


namespace abs_x_minus_1_lt_2_necessary_but_not_sufficient_for_x_mul_x_minus_3_lt_0_l567_567313

theorem abs_x_minus_1_lt_2_necessary_but_not_sufficient_for_x_mul_x_minus_3_lt_0 :
  ∀ (x : ℝ), |x - 1| < 2 → (x(x - 3) < 0) :=
by sorry

end abs_x_minus_1_lt_2_necessary_but_not_sufficient_for_x_mul_x_minus_3_lt_0_l567_567313


namespace num_red_balls_is_eight_l567_567992

-- Define the conditions
def num_black_balls : ℕ := 8
def num_white_balls : ℕ := 4
def red_ball_frequency : ℝ := 0.4

-- Prove the number of red balls
theorem num_red_balls_is_eight : 
  let total_balls := (num_black_balls + num_white_balls) / (1 - red_ball_frequency) in
  total_balls - (num_black_balls + num_white_balls) = 8 :=
by
  let total_balls := (num_black_balls + num_white_balls) / (1 - red_ball_frequency)
  have calculation : total_balls - (num_black_balls + num_white_balls) = 8 := sorry
  exact calculation

end num_red_balls_is_eight_l567_567992


namespace expected_adjacent_red_pairs_l567_567658

theorem expected_adjacent_red_pairs :
  ∃ (E : ℚ), E = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_l567_567658


namespace integer_solution_for_equation_l567_567285

theorem integer_solution_for_equation :
  ∃ (M : ℤ), 14^2 * 35^2 = 10^2 * (M - 10)^2 ∧ M = 59 :=
by
  sorry

end integer_solution_for_equation_l567_567285


namespace min_weight_of_lightest_l567_567706

theorem min_weight_of_lightest (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (h1 : 71 * m + m = 72 * m) 
  (h2 : 34 * n + n = 35 * n) 
  (h3 : 72 * m = 35 * n) : m = 35 := 
sorry

end min_weight_of_lightest_l567_567706


namespace part_a_part_b_l567_567142

theorem part_a (cube : set ℝ) (h_cube : ∀ x y z ∈ cube, x^2 + y^2 + z^2 ≤ 1)
  : ∃ parts : finset (set ℝ), (∀ p ∈ parts, ∀ x y ∈ p, dist x y < 4/5) :=
by {
  sorry
}

theorem part_b (cube : set ℝ) (h_cube : ∀ x y z ∈ cube, x^2 + y^2 + z^2 ≤ 1)
  : ¬ ∃ parts : finset (set ℝ), (∀ p ∈ parts, ∀ x y ∈ p, dist x y < 4/7) :=
by {
  sorry
}

end part_a_part_b_l567_567142


namespace example_theorem_l567_567127

-- Definitions of the conditions
def parallel (l1 l2 : Line) : Prop := sorry

def Angle (A B C : Point) : ℝ := sorry

-- Given conditions
def DC_parallel_AB (DC AB : Line) : Prop := parallel DC AB
def DCA_eq_55 (D C A : Point) : Prop := Angle D C A = 55
def ABC_eq_60 (A B C : Point) : Prop := Angle A B C = 60

-- Proof that angle ACB equals 5 degrees given the conditions
theorem example_theorem (D C A B : Point) (DC AB : Line) :
  DC_parallel_AB DC AB →
  DCA_eq_55 D C A →
  ABC_eq_60 A B C →
  Angle A C B = 5 := by
  sorry

end example_theorem_l567_567127


namespace grandmother_age_l567_567198

theorem grandmother_age (minyoung_age_current : ℕ)
                         (minyoung_age_future : ℕ)
                         (grandmother_age_future : ℕ)
                         (h1 : minyoung_age_future = minyoung_age_current + 3)
                         (h2 : grandmother_age_future = 65)
                         (h3 : minyoung_age_future = 10) : grandmother_age_future - (minyoung_age_future -minyoung_age_current) = 62 := by
  sorry

end grandmother_age_l567_567198


namespace possible_last_digits_count_l567_567614

theorem possible_last_digits_count : 
  ∃ s : Finset Nat, s = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ ∀ n ∈ s, ∃ m, (m % 10 = n) ∧ (m % 3 = 0) := 
sorry

end possible_last_digits_count_l567_567614


namespace x_over_y_l567_567971

theorem x_over_y (x y : ℝ) (h : 16 * x = 0.24 * 90 * y) : x / y = 1.35 :=
sorry

end x_over_y_l567_567971


namespace workers_contribution_eq_l567_567298

variable (W C : ℕ)

theorem workers_contribution_eq :
  W * C = 300000 → W * (C + 50) = 320000 → W = 400 :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end workers_contribution_eq_l567_567298


namespace bowen_spending_l567_567355

noncomputable def total_amount_spent (pen_cost pencil_cost : ℕ) (number_of_pens number_of_pencils : ℕ) : ℕ :=
  number_of_pens * pen_cost + number_of_pencils * pencil_cost

theorem bowen_spending :
  let pen_cost := 15 in
  let pencil_cost := 25 in
  let number_of_pens := 40 in
  let number_of_pencils := number_of_pens + (2 * number_of_pens / 5) in
  total_amount_spent pen_cost pencil_cost number_of_pens number_of_pencils = 2000 :=
by
  sorry

end bowen_spending_l567_567355


namespace f_31_eq_neg1_l567_567054

noncomputable def f : ℝ → ℝ := sorry

axiom f_neg : ∀ x : ℝ, f (-x) = -f x
axiom f_period : ∀ x : ℝ, f (x + 1) = f (1 - x)
axiom f_log : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = log (x + 1) / log 2

theorem f_31_eq_neg1 : f 31 = -1 := sorry

end f_31_eq_neg1_l567_567054


namespace find_y_l567_567969

theorem find_y (y : ℚ) (h : 1/3 - 1/4 = 4/y) : y = 48 := sorry

end find_y_l567_567969


namespace daryl_crates_needed_l567_567384

theorem daryl_crates_needed (a b c d e : Nat) 
  (h1 : a = 4 * 5)                    -- weight of 4 bags of nails
  (h2 : b = 12 * 5)                   -- weight of 12 bags of hammers
  (h3 : c = 10 * 30)                  -- weight of 10 bags of wooden planks
  (h4 : d = a + b + c)                -- total weight of items
  (h5 : e = d - 80)                   -- weight to be loaded into crates
  (crate_capacity : Nat := 20)        -- weight capacity of each crate
  : (e / crate_capacity) = 15 :=      -- number of crates needed
by
  have h_total_weight : d = 380 := by
    rw [←h1, ←h2, ←h3]
    calc
      d = a + b + c      : h4
      ... = 20 + 60 + 300 : by rw [h1, h2, h3]
      ... = 380          : by norm_num
  have h_weight_to_load : e = 300 := by
    rw [←h5, h_total_weight]
    calc
      e = d - 80      : h5
      ... = 380 - 80  : by rw [h_total_weight]
      ... = 300       : by norm_num
  show (e / crate_capacity) = 15 from 
    calc
      e / crate_capacity = 300 / 20 : by rw [←h_weight_to_load]
      ... = 15                     : by norm_num

end daryl_crates_needed_l567_567384


namespace largest_integer_less_100_leaves_remainder_4_l567_567453

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l567_567453


namespace sum_of_divisors_143_l567_567738

theorem sum_of_divisors_143 : 
  (finset.range (144)).filter (λ d, 143 % d = 0).sum = 168 :=
by
  sorry

end sum_of_divisors_143_l567_567738


namespace sum_of_divisors_143_l567_567758

theorem sum_of_divisors_143 : ∑ d in finset.filter (λ d, 143 % d = 0) (finset.range 144), d = 168 :=
by sorry

end sum_of_divisors_143_l567_567758


namespace range_of_m_l567_567584

theorem range_of_m (A : Set ℝ) (m : ℝ) (h : ∃ x, x ∈ A ∩ {x | x ≠ 0}) :
  -4 < m ∧ m < 0 :=
by
  have A_def : A = {x | x^2 + (m+2)*x + 1 = 0} := sorry
  have h_non_empty : ∃ x, x ∈ A ∧ x ≠ 0 := sorry
  have discriminant : (m+2)^2 - 4 < 0 := sorry
  exact ⟨sorry, sorry⟩

end range_of_m_l567_567584


namespace X_Y_discrete_distribution_l567_567177
noncomputable theory

variables {Ω : Type*} [ProbabilitySpace Ω]
variables (X Y : Ω → ℝ)

-- Assume X and Y are non-degenerate.
def non_degenerate (X : Ω → ℝ) : Prop := ∀ x, P ^ (X = x) < 1
axiom X_non_degenerate : non_degenerate X
axiom Y_non_degenerate : non_degenerate Y

-- Assume X and Y are independent random variables and XY has a discrete distribution.
axiom X_Y_independent : Indep X Y
axiom XY_discrete : Discrete (λ ω, X ω * Y ω)

-- Prove that X and Y have discrete distributions
theorem X_Y_discrete_distribution : Discrete X ∧ Discrete Y :=
by {
  sorry
}

end X_Y_discrete_distribution_l567_567177


namespace alice_speed_problem_l567_567613

open Real

theorem alice_speed_problem (d t : ℝ) 
  (h1 : d = 50 * (t + 4 / 60))
  (h2 : d = 70 * (t - 2 / 60)) : 
  let r := d / t in 
  r = 57 := 
by
  -- Proof goes here
  sorry

end alice_speed_problem_l567_567613


namespace difference_between_two_smallest_integers_l567_567267

noncomputable def lcm_2_to_12 : ℕ :=
  Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 10 (Nat.lcm 11 12))))))))))

theorem difference_between_two_smallest_integers:
  (∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → ∃ n : ℕ, n > 1 ∧ n % k = 1) →
  (let n1 := 1 + lcm_2_to_12,
       n2 := 1 + 2 * lcm_2_to_12 in
       n2 - n1 = 27720) :=
by
  sorry

end difference_between_two_smallest_integers_l567_567267


namespace largest_integer_with_remainder_l567_567426

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l567_567426


namespace largest_int_less_than_100_mod_6_eq_4_l567_567441

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l567_567441


namespace complex_number_quadrant_l567_567811

def inSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_quadrant : inSecondQuadrant (i / (1 - i)) :=
by
  sorry

end complex_number_quadrant_l567_567811


namespace equal_areas_if_midpoints_coincide_l567_567694
open Classical

-- Define what it means for two polygons to have coinciding midpoints
noncomputable def midpoints_coincide {P Q : Type} [Polygon P] [Polygon Q] (n : ℕ) 
  (coincide : ∀ i : Fin (2 * n), midpoint (side P i) = midpoint (side Q i)) : Prop :=
  sorry

-- Theorem statement: Equal areas if midpoints coincide
theorem equal_areas_if_midpoints_coincide {P Q : Type} [Polygon P] [Polygon Q] 
  (n : ℕ) (coincide : ∀ i : Fin (2 * n), midpoint (side P i) = midpoint (side Q i)) :
  area P = area Q :=
sorry

end equal_areas_if_midpoints_coincide_l567_567694


namespace stratified_sampling_l567_567807

theorem stratified_sampling (total_employees : ℕ) (middle_ratio young_ratio senior_ratio : ℕ) (sample_size : ℕ)
    (h_total : total_employees = 3200)
    (h_ratio : middle_ratio = 5 ∧ young_ratio = 3 ∧ senior_ratio = 2)
    (h_sample : sample_size = 400) :
  let ratio_sum := middle_ratio + young_ratio + senior_ratio
      middle_sample := sample_size * middle_ratio / ratio_sum
      young_sample := sample_size * young_ratio / ratio_sum
      senior_sample := sample_size * senior_ratio / ratio_sum
  in middle_sample = 200 ∧ young_sample = 120 ∧ senior_sample = 80 := by
  sorry

end stratified_sampling_l567_567807


namespace finite_squares_cover_black_cells_l567_567617

-- Define the problem conditions
variable (N : ℕ)
axiom infinite_grid : Prop

-- Define the main statement
theorem finite_squares_cover_black_cells (N : ℕ) (infinite_grid : Prop) :
  ∃ (S : set (set (ℕ × ℕ))), 
    (∀ cell, cell ∈ S → cell coordinates are within a finite bound) ∧
    (∀ black_cell, black_cell coordinates in the original grid → ∃ s ∈ S, black_cell ∈ s) ∧
    (∀ s ∈ S, (1 / 5 : ℝ) ≤ (count of black cells in s / area of s) ∧ (count of black cells in s / area of s) ≤ (4 / 5 : ℝ)) :=
sorry

end finite_squares_cover_black_cells_l567_567617


namespace b_finishes_work_in_14_days_l567_567329

noncomputable def b_work_days : ℝ := 
  let work_rate_a := 1 / 4
  let time_together := 2
  let time_b_alone := 5.000000000000001
  let work_together := time_together * (work_rate_a + 1 / x)
  let work_b_alone := time_b_alone * (1 / x)
  if work_together + work_b_alone = 1 then x else 0

theorem b_finishes_work_in_14_days : b_work_days = 14 := by
  sorry

end b_finishes_work_in_14_days_l567_567329


namespace sum_of_divisors_143_l567_567736

theorem sum_of_divisors_143 : 
  (finset.range (144)).filter (λ d, 143 % d = 0).sum = 168 :=
by
  sorry

end sum_of_divisors_143_l567_567736


namespace chromium_alloy_problem_l567_567996

theorem chromium_alloy_problem :
  ∃ x : ℝ, 0.12 * x + 0.08 * 30 = 0.09 * (x + 30) ∧ x = 10 :=
by
  use 10
  split
  sorry

end chromium_alloy_problem_l567_567996


namespace expected_adjacent_red_pairs_l567_567657

theorem expected_adjacent_red_pairs :
  ∃ (E : ℚ), E = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_l567_567657


namespace smallest_x_l567_567015

theorem smallest_x (x : ℝ) (h1 : ∃ k : ℤ, x = k + (x - k) ∧ 0 ≤ x - k ∧ x - k < 1) :
  (⨅ x, (floor(x) = 3 + 50 * (x - floor(x)))) = 3 :=
by
  sorry

end smallest_x_l567_567015


namespace problem_statement_l567_567283

-- Definitions for the conditions in the problem
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p
def has_three_divisors (k : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ k = p^2

-- Given conditions
def m : ℕ := 3 -- the smallest odd prime
def n : ℕ := 49 -- the largest integer less than 50 with exactly three positive divisors

-- The proof statement
theorem problem_statement : m + n = 52 :=
by sorry

end problem_statement_l567_567283


namespace circle_tangent_parabola_height_difference_l567_567806

theorem circle_tangent_parabola_height_difference
  (a b r : ℝ)
  (parabola_eq : ∀ x, 4 * x^2 = 4 * x^2)
  (circle_tangent_points : ∀ c, (c^2 + (4 * a^2 - b)^2 = r^2) ∧ (c^2 + (4 * (-a)^2 - b)^2 = r^2))
  (center_eq : b = 4 * a^2 + 1 / 8) :
  (b - 4 * a^2) = 1 / 8 :=
begin
  sorry,
end

end circle_tangent_parabola_height_difference_l567_567806


namespace inscribed_sphere_radius_eq_l567_567249

noncomputable def radius_of_inscribed_sphere_in_hexagonal_pyramid
  (a b : ℝ) (h : b^2 > (3 * a^2 / 4)) : ℝ :=
  let PM := real.sqrt (b^2 - (3 * a^2 / 4))
  let KM := (a * real.sqrt 3) / 2
  let KP := real.sqrt (b^2 - (a^2 / 4))
  PM * (KM / (KM + KP))

theorem inscribed_sphere_radius_eq
  (a b : ℝ) (h : b^2 > (3 * a^2 / 4)) :
  radius_of_inscribed_sphere_in_hexagonal_pyramid a b h =
  (a * real.sqrt (3 * (b^2 - (3 * a^2 / 4)))) / 
  (2 * a * real.sqrt 3 + 2 * real.sqrt (b^2 - (a^2 / 4))) :=
sorry

end inscribed_sphere_radius_eq_l567_567249


namespace non_congruent_rectangles_unique_l567_567343

theorem non_congruent_rectangles_unique (P : ℕ) (w : ℕ) (h : ℕ) :
  P = 72 ∧ w = 14 ∧ 2 * (w + h) = P → 
  (∃ h, w = 14 ∧ 2 * (w + h) = 72 ∧ 
  ∀ w' h', w' = w → 2 * (w' + h') = 72 → (h' = h)) :=
by
  sorry

end non_congruent_rectangles_unique_l567_567343


namespace octagon_perimeter_l567_567714

theorem octagon_perimeter (n : ℕ) (side_length : ℝ) (h1 : n = 8) (h2 : side_length = 2) : 
  n * side_length = 16 :=
by
  sorry

end octagon_perimeter_l567_567714


namespace innkeeper_room_assignments_l567_567827

open Finset

theorem innkeeper_room_assignments :
  let scholars := 6
  let rooms := 6
  let max_per_room := 2
  let unoccupied := 1
  ∃ (ways : ℕ), ways = 9720 ∧
    (ways = (choose rooms (rooms - unoccupied)) * (nat.perm scholars (rooms - unoccupied)) +
            (choose rooms (rooms - unoccupied - 1)) * 
            (nat.perm scholars (rooms - unoccupied - 1)) *
            (choose (rooms - unoccupied - 1) ((rooms - unoccupied - 1) / max_per_room))) :=
sorry

end innkeeper_room_assignments_l567_567827


namespace count_routes_on_3x3_grid_l567_567516

theorem count_routes_on_3x3_grid : 
  ∃ (n : ℕ), n = 24 ∧ 
  (∀ (R D X : ℕ), R + D + X = 3 ∧ R + X ≤ 3 ∧ D + X ≤ 3 ∧ 
    (nat.choose (R + D + X) R * nat.choose (D + X) D * nat.choose (X) X = n)) :=
sorry

end count_routes_on_3x3_grid_l567_567516


namespace largest_int_less_than_100_mod_6_eq_4_l567_567442

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l567_567442


namespace most_probable_standard_parts_in_batch_l567_567342

theorem most_probable_standard_parts_in_batch :
  let q := 0.075
  let p := 1 - q
  let n := 39
  ∃ k₀ : ℤ, 36 ≤ k₀ ∧ k₀ ≤ 37 := 
by
  sorry

end most_probable_standard_parts_in_batch_l567_567342


namespace tetrahedron_max_volume_l567_567686

noncomputable def tetrahedron_volume (AC AB BD CD : ℝ) : ℝ :=
  let x := (2 : ℝ) * (Real.sqrt 3) / 3
  let m := Real.sqrt (1 - x^2 / 4)
  let α := Real.pi / 2 -- Maximize with sin α = 1
  x * m^2 * Real.sin α / 6

theorem tetrahedron_max_volume : ∀ (AC AB BD CD : ℝ),
  AC = 1 → AB = 1 → BD = 1 → CD = 1 →
  tetrahedron_volume AC AB BD CD = 2 * Real.sqrt 3 / 27 :=
by
  intros AC AB BD CD hAC hAB hBD hCD
  rw [hAC, hAB, hBD, hCD]
  dsimp [tetrahedron_volume]
  norm_num
  sorry

end tetrahedron_max_volume_l567_567686


namespace julie_initial_savings_l567_567573

def calculate_earnings (lawns newspapers dogs : ℕ) (price_lawn price_newspaper price_dog : ℝ) : ℝ :=
  (lawns * price_lawn) + (newspapers * price_newspaper) + (dogs * price_dog)

def calculate_total_spent_bike (earnings remaining_money : ℝ) : ℝ :=
  earnings + remaining_money

def calculate_initial_savings (cost_bike total_spent : ℝ) : ℝ :=
  cost_bike - total_spent

theorem julie_initial_savings :
  let cost_bike := 2345
  let lawns := 20
  let newspapers := 600
  let dogs := 24
  let price_lawn := 20
  let price_newspaper := 0.40
  let price_dog := 15
  let remaining_money := 155
  let earnings := calculate_earnings lawns newspapers dogs price_lawn price_newspaper price_dog
  let total_spent := calculate_total_spent_bike earnings remaining_money
  calculate_initial_savings cost_bike total_spent = 1190 :=
by
  -- Although the proof is not required, this setup assumes correctness.
  sorry

end julie_initial_savings_l567_567573


namespace evaluate_expression_l567_567293

theorem evaluate_expression : 12^2 + 2 * 12 * 5 + 5^2 = 289 := by
  sorry

end evaluate_expression_l567_567293


namespace banana_price_l567_567330

theorem banana_price (b : ℝ) : 
    (∃ x : ℕ, 0.70 * x + b * (9 - x) = 5.60 ∧ x + (9 - x) = 9) → b = 0.60 :=
by
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  -- equations to work with:
  -- 0.70 * x + b * (9 - x) = 5.60
  -- x + (9 - x) = 9
  sorry

end banana_price_l567_567330


namespace beth_total_crayons_l567_567796

theorem beth_total_crayons :
  let packs := 4
  let crayons_per_pack := 10
  let extra_crayons := 6
  packs * crayons_per_pack + extra_crayons = 46 :=
by
  let packs := 4
  let crayons_per_pack := 10
  let extra_crayons := 6
  show packs * crayons_per_pack + extra_crayons = 46
  sorry

end beth_total_crayons_l567_567796


namespace johns_cost_per_sheet_correct_l567_567829

noncomputable def johns_cost_per_sheet : ℝ := 2.75

-- Define the total cost functions for both photo stores
def johns_photo_world_cost (sheets : ℕ) (cost_per_sheet : ℝ) : ℝ :=
  sheets * cost_per_sheet + 125

def sams_picture_emporium_cost (sheets : ℕ) : ℝ :=
  sheets * 1.50 + 140

-- Prove John's Photo World charges $2.75 per sheet if both stores cost the same for 12 sheets
theorem johns_cost_per_sheet_correct :
  (johns_photo_world_cost 12 johns_cost_per_sheet) = (sams_picture_emporium_cost 12) :=
by
  have h : 12 * johns_cost_per_sheet + 125 = 12 * 1.50 + 140 := by
    have : 12 * 1.50 = 18 := by norm_num
    have : 18 + 140 = 158 := by norm_num
    linarith
  exact h

end johns_cost_per_sheet_correct_l567_567829


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567438

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567438


namespace tax_refund_l567_567310

-- Definitions based on the problem conditions
def monthly_salary : ℕ := 9000
def treatment_cost : ℕ := 100000
def medication_cost : ℕ := 20000
def tax_rate : ℚ := 0.13

-- Annual salary calculation
def annual_salary := monthly_salary * 12

-- Total spending on treatment and medications
def total_spending := treatment_cost + medication_cost

-- Possible tax refund based on total spending
def possible_tax_refund := total_spending * tax_rate

-- Income tax paid on the annual salary
def income_tax_paid := annual_salary * tax_rate

-- Prove statement that the actual tax refund is equal to income tax paid
theorem tax_refund : income_tax_paid = 14040 := by
  sorry

end tax_refund_l567_567310


namespace no_pentagon_section_l567_567821

-- Definitions corresponding to the conditions
def isRegularTetrahedron (tet : Type) : Prop :=
  ∃ v₁ v₂ v₃ v₄ : ℝ × ℝ × ℝ, 
    v1 ≠ v2 ∧ v1 ≠ v3 ∧ v1 ≠ v4 ∧ v2 ≠ v3 ∧ v2 ≠ v4 ∧ v3 ≠ v4 ∧ 
    ∀ (x y : ℝ × ℝ × ℝ), x ∈ {v₁, v₂, v₃, v₄} → y ∈ {v₁, v₂, v₃, v₄} → 
    ‖x - y‖ = ‖v₁ - v₂‖ 

def singlePlaneCut (tet : Type) (cut : ℝ × ℝ × ℝ → ℝ) : Prop :=
  ∀ v₁ v₂ v₃ v₄ : ℝ × ℝ × ℝ, 
    isRegularTetrahedron tet → cut v₁ = 0 ∧ cut v₂ = 0 ∧ cut v₃ = 0 ∧ cut v₄ = 0 → 
    (cut v₁ = 0 ↔ cut v₂ = 0 ↔ cut v₃ = 0 ↔ cut v₄ = 0)

-- Statement of the theorem based on question and conditions
theorem no_pentagon_section (tet : Type) (cut : ℝ × ℝ × ℝ → ℝ) :
  isRegularTetrahedron tet → singlePlaneCut tet cut → ¬ (∃ p : set (ℝ × ℝ × ℝ), p.card = 5 ∧ ∀ x ∈ p, ∃ y z : ℝ × ℝ × ℝ, cut y = 0 ∧ cut z = 0 ∧ ∀ w ∈ p, w = x ∨ w = y ∨ w = z) :=
by
  sorry

end no_pentagon_section_l567_567821


namespace point_in_second_or_third_quadrant_l567_567491

theorem point_in_second_or_third_quadrant (k b : ℝ) (h₁ : k < 0) (h₂ : b ≠ 0) : 
  (k < 0 ∧ b > 0) ∨ (k < 0 ∧ b < 0) :=
by
  sorry

end point_in_second_or_third_quadrant_l567_567491


namespace intersection_of_M_and_N_is_correct_l567_567095

-- Definitions according to conditions
def M : Set ℤ := {-4, -2, 0, 2, 4, 6}
def N : Set ℤ := {x | -3 ≤ x ∧ x ≤ 4}

-- Proof statement
theorem intersection_of_M_and_N_is_correct : (M ∩ N) = {-2, 0, 2, 4} := by
  sorry

end intersection_of_M_and_N_is_correct_l567_567095


namespace sum_of_divisors_143_l567_567754

theorem sum_of_divisors_143 : ∑ d in finset.filter (λ d, 143 % d = 0) (finset.range 144), d = 168 :=
by sorry

end sum_of_divisors_143_l567_567754


namespace distance_A_C_15_l567_567623

noncomputable def distance_from_A_to_C : ℝ := 
  let AB := 6
  let AC := AB + (3 * AB) / 2
  AC

theorem distance_A_C_15 (A B C D : ℝ) (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (h4 : D - A = 24) (h5 : D - B = 3 * (B - A)) 
  (h6 : C = (B + D) / 2) :
  distance_from_A_to_C = 15 :=
by sorry

end distance_A_C_15_l567_567623


namespace machine_b_finishes_in_12_hours_l567_567192

noncomputable def machine_b_time : ℝ :=
  let rA := 1 / 4  -- rate of Machine A
  let rC := 1 / 6  -- rate of Machine C
  let rTotalTogether := 1 / 2  -- rate of all machines working together
  let rB := (rTotalTogether - rA - rC)  -- isolate the rate of Machine B
  1 / rB  -- time for Machine B to finish the job

theorem machine_b_finishes_in_12_hours : machine_b_time = 12 :=
by
  sorry

end machine_b_finishes_in_12_hours_l567_567192


namespace Bowen_total_spent_l567_567357

def pencil_price : ℝ := 0.25
def pen_price : ℝ := 0.15
def num_pens : ℕ := 40

def num_pencils := num_pens + (2 / 5) * num_pens

theorem Bowen_total_spent : num_pencils * pencil_price + num_pens * pen_price = 20 := by
  sorry

end Bowen_total_spent_l567_567357


namespace sum_of_divisors_of_143_l567_567767

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (nat.divisors n).toFinset, k

theorem sum_of_divisors_of_143 : sum_divisors 143 = 168 :=
by sorry

end sum_of_divisors_of_143_l567_567767


namespace sum_of_divisors_143_l567_567741

theorem sum_of_divisors_143 : 
  (finset.range (144)).filter (λ d, 143 % d = 0).sum = 168 :=
by
  sorry

end sum_of_divisors_143_l567_567741


namespace complex_problem_l567_567931

noncomputable theory

open Complex

theorem complex_problem (z1 z2 : ℂ) (a : ℝ) (h : a ≠ 0) 
  (cond1 : abs z1 = abs (z1 + 2 * z2))
  (cond2 : conj z1 * z2 = a * (2 - I)) :
  z2 / z1 = -((4 : ℝ) / 5) + ((2 : ℝ) / 5) * I := by
sorry

end complex_problem_l567_567931


namespace number_of_points_l567_567124

theorem number_of_points (a b : ℤ) : (|a| = 3 ∧ |b| = 2) → ∃! (P : ℤ × ℤ), P = (a, b) :=
by sorry

end number_of_points_l567_567124


namespace days_matt_and_son_eat_only_l567_567611

theorem days_matt_and_son_eat_only (x y : ℕ) 
  (h1 : x + y = 7)
  (h2 : 2 * x + 8 * y = 38) : 
  x = 3 :=
by
  sorry

end days_matt_and_son_eat_only_l567_567611


namespace find_complex_number_l567_567028

noncomputable def complex_solution := 2 - (13 / 6) * complex.I

theorem find_complex_number (z : ℂ) (h1 : |z - 2| = |z + 2|) (h2 : |z - 2| = |z - 3 * complex.I|) :
  z = complex_solution := by
  sorry

end find_complex_number_l567_567028


namespace largest_integer_less_100_leaves_remainder_4_l567_567457

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l567_567457


namespace triangle_max_third_side_l567_567639

theorem triangle_max_third_side (D E F : ℝ) (a b : ℝ) (h1 : a = 8) (h2 : b = 15) 
(h_cos : Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1) 
: ∃ c : ℝ, c = 13 :=
by
  sorry

end triangle_max_third_side_l567_567639


namespace range_of_b_for_triangle_solutions_l567_567118

theorem range_of_b_for_triangle_solutions (a : ℝ) (b : ℝ) (A : ℝ) (h1 : a = 12) (h2 : A = real.pi / 3) :
  (b * real.sin A < a ∧ a < b) ↔ (12 < b ∧ b < 8 * real.sqrt 3) :=
by
  sorry

end range_of_b_for_triangle_solutions_l567_567118


namespace committee_count_l567_567360

theorem committee_count :
  let num_ways_first3_dept := (nat.choose 3 2) * (nat.choose 3 2)
  let num_ways_physics := (nat.choose 1 1) * (nat.choose 1 1)
  let total_ways := num_ways_first3_dept ^ 3 * num_ways_physics
  total_ways = 729 :=
by
  sorry

end committee_count_l567_567360


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567424

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567424


namespace arithmetic_sequence_general_term_sequence_sum_l567_567060

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ) (S : ℕ → ℤ) (a_5 : a 5 = a 1 + 4 * (a 2 - a 1)) 
  (a_6 : a 6 = a 1 + 5 * (a 2 - a 1)) 
  (h₁ : a 5 + a 6 = 24) 
  (h₂ : S 3 = 15) 
  (Sn_formula : ∀ n, S n = (n / 2 : ℤ) * (2 * a 1 + (n - 1) * (a 2 - a 1))) : 
  ∀ n, a n = 2 * n + 1 := 
sorry

theorem sequence_sum 
  (a : ℕ → ℤ) (b : ℕ → ℤ) 
  (h : ∀ n, a n = 2 * n + 1) 
  (bn_def : ∀ n, b n = 1 / (a n ^ 2 - 1)) : 
  ∀ n, (Finset.range n).sum (λ n, b n) = n / (4 * (n + 1)) := 
sorry

end arithmetic_sequence_general_term_sequence_sum_l567_567060


namespace fill_time_with_both_pipes_l567_567282

variable (C : ℝ)

def pipe_a_rate : ℝ := C / 10
def pipe_b_rate : ℝ := C / 20
def combined_rate : ℝ := pipe_a_rate C + pipe_b_rate C
def time_to_fill : ℝ := C / combined_rate C

theorem fill_time_with_both_pipes :
  pipe_a_rate C = C / 10 ∧
  pipe_b_rate C = C / 20 ∧
  combined_rate C = (2 * C + C) / 20 ∧
  time_to_fill C = 20 / 3 :=
by
  sorry

end fill_time_with_both_pipes_l567_567282


namespace no_solution_iff_n_eq_neg2_l567_567528

noncomputable def has_no_solution (n : ℝ) : Prop :=
  ∀ x y z : ℝ, ¬ (n * x + y + z = 2 ∧ 
                  x + n * y + z = 2 ∧ 
                  x + y + n * z = 2)

theorem no_solution_iff_n_eq_neg2 (n : ℝ) : has_no_solution n ↔ n = -2 := by
  sorry

end no_solution_iff_n_eq_neg2_l567_567528


namespace largest_int_mod_6_less_than_100_l567_567452

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l567_567452


namespace find_triplets_l567_567026

theorem find_triplets (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  (a^m + 1 ∣ (a + 1)^n) ↔ ((a = 1) ∨ (a = 2 ∧ m = 3 ∧ n ≥ 2)) :=
by
  sorry

end find_triplets_l567_567026


namespace increase_by_40_percent_l567_567317

theorem increase_by_40_percent (initial_number : ℕ) (increase_rate : ℕ) :
  initial_number = 150 → increase_rate = 40 →
  initial_number + (increase_rate / 100 * initial_number) = 210 := by
  sorry

end increase_by_40_percent_l567_567317


namespace average_minutes_correct_l567_567986

variable (s : ℕ)
def sixth_graders := 3 * s
def seventh_graders := s
def eighth_graders := s / 2

def minutes_sixth_graders := 18 * sixth_graders s
def minutes_seventh_graders := 20 * seventh_graders s
def minutes_eighth_graders := 22 * eighth_graders s

def total_minutes := minutes_sixth_graders s + minutes_seventh_graders s + minutes_eighth_graders s
def total_students := sixth_graders s + seventh_graders s + eighth_graders s

def average_minutes := total_minutes s / total_students s

theorem average_minutes_correct : average_minutes s = 170 / 9 := sorry

end average_minutes_correct_l567_567986


namespace find_conjugate_sum_l567_567526

-- Define the required complex numbers and variables
variable {θ φ : ℝ}
variable (z1 z2 : ℂ) (cond : z1 + z2 = 1/3 + 2/5 * complex.I)

-- Define the problem to be proven
theorem find_conjugate_sum (h1 : z1 = complex.exp (complex.I * θ))
                           (h2 : z2 = complex.exp (complex.I * φ)) :
  complex.conj z1 + complex.conj z2 = 1/3 - 2/5 * complex.I :=
by
  have h_conj1 : complex.conj z1 = complex.exp (-complex.I * θ), 
  by rw [complex.conj_exp]
  have h_conj2 : complex.conj z2 = complex.exp (-complex.I * φ), 
  by rw [complex.conj_exp]
  rw [h_conj1, h_conj2]
  exact cond

end find_conjugate_sum_l567_567526


namespace who_gets_largest_final_answer_l567_567607

noncomputable def Liam_final := (15 - 2) * 3 + 3
noncomputable def Maya_final := (15 * 3 - 4) + 5
noncomputable def Arjun_final := (15 - 3 + 4) * 3

theorem who_gets_largest_final_answer :
  Arjun_final = 48 ∧ Arjun_final > Liam_final ∧ Arjun_final > Maya_final :=
by
  have h_liam : Liam_final = 42 := by norm_num
  have h_maya : Maya_final = 46 := by norm_num
  have h_arjun : Arjun_final = 48 := by norm_num
  rw [h_liam, h_maya, h_arjun]
  exact ⟨rfl, by norm_num, by norm_num⟩

end who_gets_largest_final_answer_l567_567607


namespace h_3_value_l567_567158

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := Real.sqrt (f x) - 3

def h (x : ℝ) : ℝ := f (g x)

theorem h_3_value : h 3 = 3 * Real.sqrt 13 - 5 := by
  sorry

end h_3_value_l567_567158


namespace find_analytical_f_l567_567945

-- Define the function f based on given conditions
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x / (a * x + b)

-- State the conditions
def condition1 (a : ℝ) (b : ℝ) : Prop := (a ≠ 0)
def condition2 (a : ℝ) (b : ℝ) : Prop := f 2 a b = 1
def condition3 (a : ℝ) (b : ℝ) : Prop := ∃! x, f x a b = x

-- Define the analytical form of the function under the provided conditions
def analytical_expression (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = 2 * x / (x + 2)) ∨ (∀ x, f x = 1)

-- Proof problem statement
theorem find_analytical_f (a b : ℝ) (h1 : condition1 a b) (h2 : condition2 a b) (h3 : condition3 a b) :
  analytical_expression (f (·) a b) :=
sorry

end find_analytical_f_l567_567945


namespace readers_of_science_fiction_l567_567544

theorem readers_of_science_fiction (T L B : ℕ) (hT : T = 150) (hL : L = 90) (hB : B = 60) : 
  ∃ S : ℕ, S = 120 :=
by
  let S : ℕ := T + B - L
  have hS : S = 120 := by
    calc
      S = 150 + 60 - 90 : by rw [hT, hB, hL]
      ... = 120 : by norm_num
  exact ⟨S, hS⟩

end readers_of_science_fiction_l567_567544


namespace fish_caught_together_l567_567153

theorem fish_caught_together (Blaines_fish Keiths_fish : ℕ) 
  (h1 : Blaines_fish = 5) 
  (h2 : Keiths_fish = 2 * Blaines_fish) : 
  Blaines_fish + Keiths_fish = 15 := 
by 
  sorry

end fish_caught_together_l567_567153


namespace four_digit_numbers_count_l567_567102

theorem four_digit_numbers_count :
  let valid_first_digits := {4, 5, 6, 7, 8, 9}
  let valid_last_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let valid_middle_pairs := { (2,9), (3,8), (3,9), (4,7), (4,8), (4,9), (5,6), (5,7), (5,8), (5,9), (6,7), (6,8), (6,9), (7,8), (7,9), (8,9) }
  6 * 16 * 10 = 960 := by 
sorry

end four_digit_numbers_count_l567_567102


namespace terminating_decimal_numbers_count_l567_567042

def is_terminating_decimal (n: ℕ) : Prop :=
  ∃ m : ℕ, ∃ k : ℤ, n = k * (2^m) * (5^m)

theorem terminating_decimal_numbers_count :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 499 → is_terminating_decimal (n / 500) = true :=
begin
  sorry
end

end terminating_decimal_numbers_count_l567_567042


namespace triangle_and_circle_l567_567476

theorem triangle_and_circle (A B C P Q K L : Point)
    (triangle_right_ABC: is_right_triangle A B C)
    (AB_eq: dist A B = 42)
    (BC_eq: dist B C = 56)
    (circle_intersects: circle_intersecting_points B P Q K L)
    (PK_eq_KQ: dist P K = dist K Q)
    (QL_PL_ratio: dist Q L / dist P L = 3 / 4) :
    dist P Q ^ 2 = 1250 := 
sorry

end triangle_and_circle_l567_567476


namespace find_c_in_terms_of_a_and_b_l567_567224

theorem find_c_in_terms_of_a_and_b (a b : ℝ) :
  (∃ α β : ℝ, (α + β = -a) ∧ (α * β = b)) →
  (∃ c d : ℝ, (∃ α β : ℝ, (α^3 + β^3 = -c) ∧ (α^3 * β^3 = d))) →
  c = a^3 - 3 * a * b :=
by
  intros h1 h2
  sorry

end find_c_in_terms_of_a_and_b_l567_567224


namespace area_ratio_PQR_QST_l567_567587

-- Define the conditions and triangles
variables (P Q R S T U : Type) [Euclidean_Space P]

variables (PQ_parallel_RT QR_parallel_PS PR_parallel_ST : Prop)
variables (angle_PQR_is_150 : ∠ P Q R = 150)
variables (PQ_length QR_length ST_length: ℝ)

-- Given relationships and lengths
axiom PQ_RT_parallel : PQ_parallel_RT = true
axiom QR_PS_parallel : QR_parallel_PS = true
axiom PR_ST_parallel : PR_parallel_ST = true
axiom PQ_eq_4 : PQ_length = 4
axiom QR_eq_6 : QR_length = 6
axiom ST_eq_18 : ST_length = 18

-- Given similarity and congruence relationships of triangles
axiom triangle_cong_PQR_QRU : congruent_triangles P Q R Q R U
axiom triangle_sim_PQR_QST : similar_triangles P Q R Q S T

-- Main theorem to prove
theorem area_ratio_PQR_QST : 
  let (a, b) := rat_area_tris P Q R Q S T in
  a + b = 65 :=
begin
  sorry
end

end area_ratio_PQR_QST_l567_567587


namespace contrapositive_example_l567_567234

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem contrapositive_example (x y : ℤ) :
  (¬ is_even (x + y) → (¬ (is_even x ∧ is_even y))) :=
begin
  sorry
end

end contrapositive_example_l567_567234


namespace additional_vegetables_can_be_planted_l567_567203

-- Defines the garden's initial conditions.
def tomatoes_kinds := 3
def tomatoes_each := 5
def cucumbers_kinds := 5
def cucumbers_each := 4
def potatoes := 30
def rows := 10
def spaces_per_row := 15

-- The proof statement.
theorem additional_vegetables_can_be_planted (total_tomatoes : ℕ := tomatoes_kinds * tomatoes_each)
                                              (total_cucumbers : ℕ := cucumbers_kinds * cucumbers_each)
                                              (total_potatoes : ℕ := potatoes)
                                              (total_spaces : ℕ := rows * spaces_per_row) :
  total_spaces - (total_tomatoes + total_cucumbers + total_potatoes) = 85 := 
by 
  sorry

end additional_vegetables_can_be_planted_l567_567203


namespace second_player_wins_with_optimal_play_l567_567697

theorem second_player_wins_with_optimal_play :
  (∀ (a : ℕ → ℕ) (f : ℕ → ℕ), (∃ (k : ℕ), ∀ n, k ≤ n ∧ n % 2 = 1 → f n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) →
  (∃ (k : ℕ), ∀ n, k ≤ n ∧ n % 2 = 0 → f n ∈ {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) →
  (∃ n, let S := (alternate_sum_up_to n f) in S % 11 = 0) →
  False) :=
sorry

end second_player_wins_with_optimal_play_l567_567697


namespace b_geometric_a_general_term_sum_ineq_l567_567952

def a_seq (a : ℕ → ℝ) : Prop := 
  a 1 = 1 ∧ 
  a 2 = 2 ∧ 
  ∀ n : ℕ, n ≥ 2 → a (n+1) = 2 * a n + 3 * a (n-1)

def b_seq (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b n = a (n+1) + a n

theorem b_geometric (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h : a_seq a) (h2 : b_seq a b) : 
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 2 → b n = r * b (n-1) :=
sorry

theorem a_general_term (a : ℕ → ℝ) 
  (h : a_seq a) : 
  ∀ n : ℕ, a n = (3^n + (-1)^(n-1)) / 4 :=
sorry

theorem sum_ineq (a : ℕ → ℝ) 
  (h : a_seq a) (h2 : ∀ n, a n = (3^n + (-1)^(n-1)) / 4) : 
  ∀ n : ℕ, n ≥ 1 → ∑ i in (range (2*n)).map succ, (1 / a i) < 7 / 4 :=
sorry

end b_geometric_a_general_term_sum_ineq_l567_567952


namespace best_candidate_proof_l567_567841

-- Define the average scores for each participant.
def average_score : ℕ → ℝ
| 0 := 8.5  -- A
| 1 := 8.8  -- B
| 2 := 9.1  -- C
| 3 := 9.1  -- D
| _ := 0.0

-- Define the variances for each participant.
def variance : ℕ → ℝ
| 0 := 1.7  -- A
| 1 := 2.1  -- B
| 2 := 1.7  -- C
| 3 := 2.5  -- D
| _ := 0.0

-- Define the best candidate function.
def best_candidate (A B C D : ℕ) : ℕ :=
if (average_score C = 9.1) ∧ (variance C = 1.7) then C else D

-- Statement of the proof problem
theorem best_candidate_proof : best_candidate 0 1 2 3 = 2 :=
by sorry

end best_candidate_proof_l567_567841


namespace sqrt_equation_a_plus_b_eq_41_l567_567067

theorem sqrt_equation_a_plus_b_eq_41 :
  ∃ a b : ℝ, a + b = 41 ∧ sqrt (6 + (a / b)) = 6 * sqrt (a / b) :=
by 
  sorry

end sqrt_equation_a_plus_b_eq_41_l567_567067


namespace constant_term_40_l567_567029

noncomputable def binomial_expansion (r : ℕ) := 
  Nat.choose 5 r * 2^(5 - r) * (-1)^r

noncomputable def constant_from_term (r : ℕ) : ℤ :=
  binomial_expansion r * if 5 - 2 * r = 0 then 1 else 0

theorem constant_term_40 :
  let c0 := constant_from_term 2
  let c1 := constant_from_term 3
  (x * (2 * x - (1 / x))^5 * c0 + x * (2 * x - (1 / x))^5 * c1).constant_term = 40 :=
by
  sorry

end constant_term_40_l567_567029


namespace log2_x_eq_neg2_l567_567112

-- definition of the problem
noncomputable def x := (Real.log 4 / Real.log 16) ^ (Real.log 16 / Real.log 4)

-- lean statement of the theorem
theorem log2_x_eq_neg2 : Real.log2 x = -2 :=
sorry

end log2_x_eq_neg2_l567_567112


namespace determine_c_l567_567975

theorem determine_c (c : ℚ) : (∀ x : ℝ, (x + 7) * (x^2 * c * x + 19 * x^2 - c * x - 49) = 0) → c = 21 / 8 :=
by
  sorry

end determine_c_l567_567975


namespace suresh_walking_speed_correct_l567_567200

noncomputable def suresh_walking_speed (circumference : ℕ) (wife_speed : ℕ) (meeting_time_minutes : ℕ) : ℚ :=
  let wife_speed_mpm := (wife_speed : ℚ) * 1000 / 60
  let distance_wife := wife_speed_mpm * (meeting_time_minutes : ℚ)
  let distance_suresh := (circumference : ℚ) - distance_wife
  let suresh_speed_mpm := distance_suresh / (meeting_time_minutes : ℚ)
  (suresh_speed_mpm * 60) / 1000

theorem suresh_walking_speed_correct : 
  suresh_walking_speed 726 3.75 5.28 = 4.51 :=
by 
  sorry

end suresh_walking_speed_correct_l567_567200


namespace find_m_l567_567100

variables {R : Type*} [LinearOrderedField R]

-- Define vectors and their properties
variables (a b : EuclideanSpace R (Fin 2))
variable (m : R)
variable (theta : Real := 60 * Real.pi / 180)  -- 60 degrees in radians
variable (a_len b_len : R := 3, 2)  -- lengths of a and b

-- Assume the given conditions
axiom a_length: ‖a‖ = a_len
axiom b_length: ‖b‖ = b_len
axiom angle_ab: ⟪a, b⟫ = ‖a‖ * ‖b‖ * Real.cos theta
axiom perpendicular: ⟪3 • a + 5 • b, m • a - b⟫ = 0

-- Prove that m = 29 / 42
theorem find_m : m = 29 / 42 :=
by
  sorry

end find_m_l567_567100


namespace integral_is_negative_14_div_3_l567_567368

noncomputable def complex_integral : ℂ := ∫ (φ in set.Icc (0 : ℝ) π), ((complex.exp (complex.I * φ))^2 + 
  2 * (complex.exp (complex.I * φ)) * (complex.exp (-complex.I * φ))) * 
  complex.I * (complex.exp (complex.I * φ))

theorem integral_is_negative_14_div_3 : complex_integral = - (14 / 3 : ℂ) :=
  sorry

end integral_is_negative_14_div_3_l567_567368


namespace at_least_one_not_less_than_two_l567_567070

theorem at_least_one_not_less_than_two
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (i : ℕ), i ∈ {1, 2, 3} ∧ (if i = 1 then a + 1 / b else if i = 2 then b + 1 / c else c + 1 / a) ≥ 2 :=
by
  sorry

end at_least_one_not_less_than_two_l567_567070


namespace rectangle_area_60_l567_567162

theorem rectangle_area_60
  (AB CD : ℝ)
  (AX XC : ℝ)
  (P Q : ℝ)
  (AP'B CQ'D : ℝ)
  (XP PQ QX : ℝ)
  (h : ℝ)
  (h_pos : h > 0)
  (parallel_AB_CD : ABCD.1 AB = ABCD.1 CD)
  (AX_eq : AX = 4)
  (XC_eq : XC = 8)
  (XP_eq : XP = 1)
  (PQ_eq : PQ = 2)
  (QX_eq : QX = 5)
  (APQ_proj : AP'B = 90)
  (CQD_proj : CQ'D = 90)
  : AB * (AX + XC) = 60 
  :=
sorry

end rectangle_area_60_l567_567162


namespace allison_not_lowest_probability_l567_567836

open ProbabilityTheory

-- Definitions based on conditions
def allisonRolls : ℕ := 3

def brianRollDistribution : Pmf ℕ :=
Pmf.ofFinset (finset.range 6) (λ n, if n > 0 ∧ n ≤ 6 then 1 else 0)

def noahRollDistribution : Pmf ℕ :=
Pmf.ofFinset (finset.range 2) (λ n, if n = 0 then 3 else if n = 1 then 3 else 0)

-- The proof problem: proving the probability calculation
theorem allison_not_lowest_probability :
  P (λ (a b n : ℕ), allisonRolls ≥ b ∧ allisonRolls ≥ n) = 5 / 6 :=
by
  sorry

end allison_not_lowest_probability_l567_567836


namespace largest_int_less_than_100_mod_6_eq_4_l567_567445

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l567_567445


namespace streetlight_problem_l567_567262

theorem streetlight_problem : ∃ s : set ℕ, s ⊆ {2, 3, 4, 5, 6, 7, 8, 9} ∧ s.card = 3 ∧
  (∀ i ∈ {2, 3, 4, 5, 6, 7, 8}, (i + 1) ∈ s ∨ (i + 2) ∈ s ∨ (i + 3) ∈ s) ∧
  (s = {2, 5, 8} ∨ s = {3, 5, 8} ∨ s = {3, 6, 9} ∨ s = {3, 6, 8}) :=
by
  -- Proof steps would go here
  sorry

end streetlight_problem_l567_567262


namespace solve_inequality_l567_567913

noncomputable theory

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def periodic_function_3 (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (x + 3)

def inequality_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + deriv (deriv f x) > 0

def exponential_condition (f : ℝ → ℝ) : Prop :=
  exp(3) * f 2018 = 1

theorem solve_inequality (f : ℝ → ℝ)
  (hf_even : even_function f)
  (hf_periodic : ∀ x, f x = -f (x + 3/2))
  (hf_inequality : inequality_condition f)
  (hf_exponential : exponential_condition f)
  : {x : ℝ | f (x - 2) > 1 / exp x} = {x : ℝ | x > 3} :=
sorry

end solve_inequality_l567_567913


namespace sum_of_digits_l567_567271

theorem sum_of_digits (A T M : ℕ) (h1 : T = A + 3) (h2 : M = 3)
    (h3 : (∃ k : ℕ, T = k^2 * M) ∧ (∃ l : ℕ, T = 33)) : 
    ∃ x : ℕ, ∃ dsum : ℕ, (A + x) % (M + x) = 0 ∧ dsum = 12 :=
by
  sorry

end sum_of_digits_l567_567271


namespace isabel_total_distance_l567_567566

theorem isabel_total_distance 
  (d1 m1 a1 d2 m2 a2 d3 r1 d4 r2 : ℝ)
  (hm1 : d1 = 365)
  (hm2 : m1 = 7)
  (ha1 : a1 = 3)
  (ht1 : d2 = 450)
  (ht2 : m2 = 5)
  (ha2 : a2 = 4)
  (hw1 : d3 = 800)
  (hr1 : r1 = 2)
  (hw2 : d4 = 600)
  (hr2 : r2 = 4) :
  let total_m_w_f := 3 * (d1 * m1 + d1 * a1),
      total_t_th := 2 * (d2 * m2 + d2 * a2),
      total_weekends := 2 * (d3 * r1 + d4 * r2),
      total_distance := total_m_w_f + total_t_th + total_weekends in
  total_distance = 27050 :=
by
  sorry

end isabel_total_distance_l567_567566


namespace find_inscribed_circle_radius_l567_567823

-- Define the given conditions as Lean definitions
def central_angle (theta : ℝ) := theta = 60
def arc_length (s : ℝ) (R : ℝ) := s = (theta * R * Math.pi / 180)

-- Define the radius of the inscribed circle
def inscribed_circle_radius (r R : ℝ) := 3 * r = R

-- The theorem statement
theorem find_inscribed_circle_radius (theta R s r : ℝ) 
  (h1 : central_angle theta)
  (h2 : arc_length s R)
  (h3 : 2 * Math.pi = s) :
  inscribed_circle_radius r R → r = 2 :=
by {
  sorry
}

end find_inscribed_circle_radius_l567_567823


namespace sum_to_12_of_7_chosen_l567_567215

theorem sum_to_12_of_7_chosen (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) (T : Finset ℕ) (hT1 : T ⊆ S) (hT2 : T.card = 7) :
  ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ a + b = 12 :=
by
  sorry

end sum_to_12_of_7_chosen_l567_567215


namespace mike_ride_distance_l567_567197

theorem mike_ride_distance (M : ℕ) 
  (cost_Mike : ℝ) 
  (cost_Annie : ℝ) 
  (annies_miles : ℕ := 26) 
  (annies_toll : ℝ := 5) 
  (mile_cost : ℝ := 0.25) 
  (initial_fee : ℝ := 2.5)
  (hc_Mike : cost_Mike = initial_fee + mile_cost * M)
  (hc_Annie : cost_Annie = initial_fee + annies_toll + mile_cost * annies_miles)
  (heq : cost_Mike = cost_Annie) :
  M = 46 := by 
  sorry

end mike_ride_distance_l567_567197


namespace geometric_sequence_inequality_l567_567049

theorem geometric_sequence_inequality 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * q) :
  a 1 ^ 2 + a 3 ^ 2 ≥ 2 * (a 2 ^ 2) :=
begin
  sorry
end

end geometric_sequence_inequality_l567_567049


namespace infinite_product_equals_nine_l567_567849

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, ite (n = 0) 1 (3^(n * (1 / 2^n)))

theorem infinite_product_equals_nine : infinite_product = 9 := sorry

end infinite_product_equals_nine_l567_567849


namespace range_of_t_l567_567088

noncomputable def f (x : ℝ) : ℝ := -1/2 * x ^ 2 + 4 * x - 3 * Real.log x

theorem range_of_t (t : ℝ) :
  (∃ t, ∀ x ∈ Set.Icc t (t + 1), f x = -1 / 2 * x ^ 2 + 4 * x - 3 * Real.log x ∧ x ≥ 0 ∧ ∃ t, x >= t ∧ x <= t + 1 ∧ ¬MonotonicOn f (Set.Icc t (t + 1))) ↔ (∃ t, t ∈ Set.Ioo 0 1 ∨ t ∈ Set.Ioo 2 3) :=
sorry

end range_of_t_l567_567088


namespace number_of_integer_solutions_l567_567957

theorem number_of_integer_solutions (pi : ℝ) (h : pi = Real.pi) : 
  ((Set.Icc (-9 : ℤ) 9).filter (λ x, |(x : ℝ)| < 3 * pi)).card = 19 := 
by
  sorry

end number_of_integer_solutions_l567_567957


namespace at_least_one_false_l567_567918

-- Define propositions p and q
def p (a : ℝ) := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) := ∃ (x : ℝ), x^2 + 2 * a * x + 2 - a = 0

-- The statement of the problem: at least one of p and q is false
theorem at_least_one_false (a : ℝ) : ¬(p a ∧ q a) ↔ a ∈ Ioo (-2 : ℝ) 1 ∨ a > 1 :=
by
  sorry

end at_least_one_false_l567_567918


namespace enhanced_prime_looking_count_below_500_l567_567854

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n

def divisible_by (n k : ℕ) : Prop := k > 0 ∧ n % k = 0 

def enhanced_prime_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬ (divisible_by n 2 ∨ divisible_by n 3 ∨ divisible_by n 5 ∨ divisible_by n 7)

def total_primes_below_500 : ℕ := 95

def exclude_first_few_primes (total_primes : ℕ) : ℕ := total_primes - 4

theorem enhanced_prime_looking_count_below_500 : 
  (E : set ℕ) := {n | n < 500 ∧ enhanced_prime_looking n} ∧
  (finite E) ∧ 
  (finset.card (finset.filter enhanced_prime_looking (finset.range 500))) = 87 :=
begin
  sorry
end

end enhanced_prime_looking_count_below_500_l567_567854


namespace part_I_part_II_l567_567505

def f (x a : ℝ) : ℝ := abs (x + a) + abs (x - 2)

theorem part_I (a : ℝ) : 
  a = -4 → {x : ℝ | f x a ≥ 6} = {x : ℝ | x ≤ 0 ∨ x ≥ 6} :=
by
  intros ha
  rw [ha]
  sorry

theorem part_II (a : ℝ) :
  (∀ x ∈ Icc 0 1, f x a ≤ abs (x - 3)) → -1 ≤ a ∧ a ≤ 0 :=
by
  intros H
  sorry

end part_I_part_II_l567_567505


namespace expected_adjacent_red_pairs_correct_l567_567661

-- The deck conditions
def standard_deck : Type := {c : ℕ // c = 52}
def num_red_cards (d : standard_deck) := 26

-- Probability definition
def prob_red_right_of_red : ℝ := 25 / 51

-- Expected number of adjacent red pairs calculation
def expected_adjacent_red_pairs (n_red : ℕ) (prob_right_red : ℝ) : ℝ :=
  n_red * prob_right_red

-- Main theorem statement
theorem expected_adjacent_red_pairs_correct (d : standard_deck) :
  expected_adjacent_red_pairs (num_red_cards d) prob_red_right_of_red = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_correct_l567_567661


namespace four_comparisons_suffice_l567_567195

-- Define the types of pies
inductive PieType
| apple
| cherry

-- Define the pies
def Pies : Fin 6 → PieType

-- Define the machine function
def machine : List PieType → Bool :=
  λ pies, pies.any (λ pie₁, pies.any (λ pie₂, pie₁ ≠ pie₂))

-- State the main theorem
theorem four_comparisons_suffice (machine : List PieType → Bool) (P1 P2 P3 P4 P5 P6 : PieType) :
  (∃ (group1 group2 : List PieType), machine group1 = false ∧ machine group2 = false ∧
    Perm (group1 ++ group2) [P1, P2, P3, P4, P5, P6] ∧ group1.length = 3 ∧ group2.length = 3) :=
sorry

end four_comparisons_suffice_l567_567195


namespace problem_equivalence_l567_567882

theorem problem_equivalence (x : ℝ) (h1 : x > 6) :
  (sqrt (x - 6 * sqrt (x - 6)) + 3 = sqrt (x + 6 * sqrt (x - 6)) - 3) ↔ (x ≥ 18) :=
by
  sorry

end problem_equivalence_l567_567882


namespace find_x_y_l567_567588

open classical
noncomputable theory

-- Consider three points C, D, and Q in a vector space (ℝ^n for simplicity)
variables (C D Q : ℝ^n)

-- Define the condition that CQ : QD = 4 : 1
def ratio_condition (C D Q : ℝ^n) : Prop :=
  ∥Q - C∥ / ∥D - Q∥ = 4

-- Define the theorem to prove
theorem find_x_y (C D Q : ℝ^n) (h : ratio_condition C D Q) :
  ∃ x y : ℝ, Q = x • C + y • D ∧ (x, y) = (1 / 5, 4 / 5) :=
sorry

end find_x_y_l567_567588


namespace hexagon_area_l567_567347

theorem hexagon_area (A B C K L M N O P : Point) (A₀ B₀ C₀ K₀ L₀ M₀ N₀ O₀ P₀ : Real) 
  (h1 : regular_hexagon K L M N O P)
  (h2 : inscribed_in_equilateral_triangle K L M N O P A B C)
  (h3 : midpoint K₀ A₀ B₀) (h4 : midpoint M₀ B₀ C₀) (h5 : midpoint O₀ A₀ C₀)
  (h6 : area_equilateral_triangle A B C = 60) :
  area_regular_hexagon K L M N O P = 30 :=
  sorry

end hexagon_area_l567_567347


namespace find_numbers_l567_567025

theorem find_numbers 
  (a b c d e : ℝ) 
  (h1 : a + b + c + d + e = 0)
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 = 0)
  (h3 : a^5 + b^5 + c^5 + d^5 + e^5 = 10)
  (h_range : ∀ x ∈ {a, b, c, d, e}, -2 ≤ x ∧ x ≤ 2) :
  {a, b, c, d, e} = {2, (real.sqrt 5 - 1) / 2, (real.sqrt 5 - 1) / 2, -(real.sqrt 5 + 1) / 2, -(real.sqrt 5 + 1) / 2} :=
by
  sorry

end find_numbers_l567_567025


namespace sum_of_repeating_decimals_l567_567021

-- Definitions for periodic decimals
def repeating_five := 5 / 9
def repeating_seven := 7 / 9

-- Theorem statement
theorem sum_of_repeating_decimals : (repeating_five + repeating_seven) = 4 / 3 :=
by
  -- Placeholder for the proof
  sorry

end sum_of_repeating_decimals_l567_567021


namespace g_properties_l567_567592

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_properties :
  (∀ x y : ℝ, g (x * g (y) + 2 * x) = 2 * x * y + g (x)) →
  let n := 2 in
  let t := g(3) + (-g(3)) in
  n * t = 0 :=
begin
  intros h,
  have h1 : n = 2, by { sorry },  -- From the steps, we know n is 2
  have h2 : t = g(3) + (-g(3)), by { sorry },  -- t is the sum of possible values of g(3)
  rw [h1, h2],
  simp,
end

end g_properties_l567_567592


namespace expected_adjacent_red_pairs_l567_567654

open Probability

def num_all_cards : ℕ := 52
def num_red_cards : ℕ := 26

/-- The expected number of pairs of adjacent cards which are both red 
     in a standard 52-card deck dealt out in a circle is 650/51. -/
theorem expected_adjacent_red_pairs : 
  let p_red_right : ℚ := 25 / 51 in
  let expected_pairs := (num_red_cards : ℚ) * p_red_right
  in expected_pairs = 650 / 51 :=
by
  sorry

end expected_adjacent_red_pairs_l567_567654


namespace find_a_l567_567404

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end find_a_l567_567404


namespace evaluate_expression_l567_567391

theorem evaluate_expression :
  let a := (1 : ℚ) / 5
  let b := (1 : ℚ) / 3
  let c := (3 : ℚ) / 7
  let d := (1 : ℚ) / 4
  (a + b) / (c - d) = 224 / 75 := by
sorry

end evaluate_expression_l567_567391


namespace find_maximum_marks_l567_567304

variable (percent_marks : ℝ := 0.92)
variable (obtained_marks : ℝ := 368)
variable (max_marks : ℝ := obtained_marks / percent_marks)

theorem find_maximum_marks : max_marks = 400 := by
  sorry

end find_maximum_marks_l567_567304


namespace sum_of_coordinates_of_D_l567_567204

theorem sum_of_coordinates_of_D (P C D : ℝ × ℝ)
  (hP : P = (4, 9))
  (hC : C = (10, 5))
  (h_mid : P = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 + D.2 = 11 :=
sorry

end sum_of_coordinates_of_D_l567_567204


namespace sequence_term_l567_567238

theorem sequence_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 2 → (∏ i in finset.range (n + 1), a (i + 1)) = n^2) :
  ∀ n : ℕ, n ≥ 2 → a n = n^2 / (n - 1)^2 :=
by
  sorry

end sequence_term_l567_567238


namespace total_increase_by_five_l567_567264

-- Let B be the number of black balls
variable (B : ℕ)
-- Let W be the number of white balls
variable (W : ℕ)
-- Initially the total number of balls
def T := B + W
-- If the number of black balls is increased to 5 times the original, the total becomes twice the original
axiom h1 : 5 * B + W = 2 * (B + W)
-- If the number of white balls is increased to 5 times the original 
def k : ℕ := 5
-- The new total number of balls 
def new_total := B + k * W

-- Prove that the new total is 4 times the original total.
theorem total_increase_by_five : new_total = 4 * T :=
by
sorry

end total_increase_by_five_l567_567264


namespace number_of_integer_solutions_l567_567956

theorem number_of_integer_solutions (pi : ℝ) (h : pi = Real.pi) : 
  ((Set.Icc (-9 : ℤ) 9).filter (λ x, |(x : ℝ)| < 3 * pi)).card = 19 := 
by
  sorry

end number_of_integer_solutions_l567_567956


namespace find_f_zero_find_f_explicit_expression_find_alpha_l567_567184

noncomputable def f (ω x : ℝ) : ℝ := 3 * Real.sin (ω * x + Real.pi / 6)

theorem find_f_zero (ω : ℝ) (hω : 0 < ω) : f ω 0 = 3 / 2 := by
  sorry

theorem find_f_explicit_expression (min_period : ℝ) (h_period : min_period = Real.pi / 2) :
  ∃ ω > 0, ∀ x, f 4 x = 3 * Real.sin (4 * x + Real.pi / 6) := by
  have hω : 4 = 2 * Real.pi / (Real.pi / 2) := by sorry
  exact ⟨4, by linarith, λ x, by sorry⟩

theorem find_alpha (α : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) 
  (hx : f 4 (α / 2) = 3 / 2) : α = Real.pi / 3 := by
  sorry

end find_f_zero_find_f_explicit_expression_find_alpha_l567_567184


namespace expression_for_t_l567_567583

theorem expression_for_t (n : ℕ) : 
  let b : ℕ → ℕ := λ k, nat.coeff (1 - X + X^2)^n X^k
  let t := ∑ i in finset.range (n+1), b (2 * i)
  t = (3^n + 1) / 2 :=
sorry

end expression_for_t_l567_567583


namespace inequality_transitive_l567_567107

theorem inequality_transitive (a b c : ℝ) (h : a < b) (h' : b < c) : a - c < b - c :=
by
  sorry

end inequality_transitive_l567_567107


namespace imaginary_part_of_product_l567_567484

def imaginary_unit : ℂ := Complex.I

def z : ℂ := 2 + imaginary_unit

theorem imaginary_part_of_product : (z * imaginary_unit).im = 2 := by
  sorry

end imaginary_part_of_product_l567_567484


namespace gym_class_students_l567_567338

theorem gym_class_students :
  ∃ n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 6 = 3 ∧ n % 8 = 5 ∧ n % 9 = 2 ∧ (n = 165 ∨ n = 237) :=
by
  sorry

end gym_class_students_l567_567338


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567435

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567435


namespace pairs_of_integers_l567_567408

theorem pairs_of_integers:
  ∃ (a b : ℕ), (11 * a * b ≤ a ^ 3 - b ^ 3 ∧ a ^ 3 - b ^ 3 ≤ 12 * a * b) ∧ ((a = 30 ∧ b = 25) ∨ (a = 8 ∧ b = 4)) := 
by {
  use 30, 25,
  split,
  { split; {
      linarith [by norm_num },
    },
  },
  { left, split; exact by norm_num, use 8, 4, split, split, linarith [by norm_num ], right
},
 end sorry

end pairs_of_integers_l567_567408


namespace conjugate_of_complex_number_l567_567907

open Complex

theorem conjugate_of_complex_number : 
  let z := 2 * Complex.I / (-1 + 2 * Complex.I) in 
  conj z = 4 / 5 + (2 * Complex.I) / 5 :=
by
  sorry

end conjugate_of_complex_number_l567_567907


namespace correct_addition_result_l567_567784

-- Define the particular number x and state the condition.
variable (x : ℕ) (h₁ : x + 21 = 52)

-- Assert that the correct result when adding 40 to x is 71.
theorem correct_addition_result : x + 40 = 71 :=
by
  -- Proof would go here; represented as a placeholder for now.
  sorry

end correct_addition_result_l567_567784


namespace largest_int_less_than_100_mod_6_eq_4_l567_567440

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l567_567440


namespace polar_coordinates_of_point_l567_567226

theorem polar_coordinates_of_point (x y : ℝ) (hx : x = -sqrt 3) (hy : y = -1) : 
  ∃ ρ θ, (ρ = 2) ∧ (θ = 7 * Real.pi / 6) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) := 
by
  sorry

end polar_coordinates_of_point_l567_567226


namespace tray_height_correct_l567_567348

open Real
open Classical

noncomputable def height_of_tray : ℝ :=
  let a := 120  -- side length of the square paper
  let dist := sqrt 13  -- distance from the corner to the start of the cuts
  let angle := π / 4  -- 45 degrees in radians
  let hyp := sqrt 26  -- hypotenuse of the right triangle formed
  (1 / 2) * (sqrt 26 + 2 * sqrt 13)  -- height calculated as per given solution

-- Prove that the height of the tray satisfies the given form
theorem tray_height_correct : height_of_tray = real.root 4 507 :=
  sorry

end tray_height_correct_l567_567348


namespace compute_d_l567_567487

theorem compute_d 
  (c d : ℚ) 
  (h1 : is_root (λ x : ℝ, x^3 + c * x^2 + d * x + 15) (3 + real.sqrt 5))
  (h2 : is_rational c)
  (h3 : is_rational d) :
  d = -18.5 := sorry

end compute_d_l567_567487


namespace students_suggested_tomatoes_l567_567217

theorem students_suggested_tomatoes (students_total mashed_potatoes bacon tomatoes : ℕ) 
  (h_total : students_total = 826)
  (h_mashed_potatoes : mashed_potatoes = 324)
  (h_bacon : bacon = 374)
  (h_tomatoes : students_total = mashed_potatoes + bacon + tomatoes) :
  tomatoes = 128 :=
by {
  sorry
}

end students_suggested_tomatoes_l567_567217


namespace sqrt_equation_solution_l567_567887

theorem sqrt_equation_solution (x : ℝ) (h : x > 6) :
  (sqrt (x - 6 * sqrt (x - 6)) + 3 = sqrt (x + 6 * sqrt (x - 6)) - 3) ↔ x ∈ set.Ici 18 :=
sorry

end sqrt_equation_solution_l567_567887


namespace probability_divisible_by_3_l567_567206

/-- 
Let \( S \) be the set \(\{1, 2, \dots, 2009\} \). 
Let \( a, b, c, \) and \( d \) be elements randomly and independently chosen from \( S \).
We aim to prove that the probability \( P(abcd + abc + ab + a \equiv 0 \text{ (mod 3)}) = \frac{13}{27} \).
-/
theorem probability_divisible_by_3 : 
  let S := set.range (λ n, n + 1) ∩ set.Icc 1 2009
    in ∀ (a b c d : ℕ), 
        (a ∈ S) → (b ∈ S) → (c ∈ S) → (d ∈ S) → 
        ((a * b * c * d + a * b * c + a * b + a) % 3 = 0) → 
        (real.rat_cast ((13 : ℚ) / 27) = real.rat_cast ((1 / 3 : ℚ) * (670 / 2009) + sorry)) :=
sorry

end probability_divisible_by_3_l567_567206


namespace transform_line_equation_l567_567236

def scaling_transformation (T : Matrix (Fin 2) (Fin 2) ℝ) (x y : ℝ) : ℝ × ℝ :=
  (T 0 0 * x + T 0 1 * y, T 1 0 * x + T 1 1 * y)

def transform_line (T : Matrix (Fin 2) (Fin 2) ℝ) (x y : ℝ) : Prop :=
  let (x', y') := scaling_transformation T x y
  3 * x' - 8 * y' + 12 = 0

theorem transform_line_equation :
  (scaling_transformation 
    (Matrix.ofVec
      ![#[4, 0],
        #[0, 3]])
    x y = (4*x, 3*y))
  → transform_line 
       (Matrix.ofVec
        ![#[4, 0],
          #[0, 3]]) 
       x y  :=
sorry

end transform_line_equation_l567_567236


namespace sum_of_divisors_143_l567_567732

theorem sum_of_divisors_143 : sigma 143 = 168 := by
  have h : 143 = 11 * 13 := by norm_num
  rw [sigma_mul_prime_prime 11 13]
  norm_num
  sorry

end sum_of_divisors_143_l567_567732


namespace geometric_locus_of_lines_is_two_skew_cylinders_l567_567266

noncomputable def is_geometric_locus_of_lines (sphere : Type) [Geometry.Sphere sphere]
  (projection_plane1 projection_plane2 : Geometry.Plane) [Geometry.Intersects sphere projection_plane1] [Geometry.Intersects sphere projection_plane2]
  (line : Geometry.Line) : Prop :=
Geometry.Parallel line projection_plane1 ∧
(embed_line_sphere_intersection line sphere) ∧
equal_partitions line projection_plane1 projection_plane2 sphere →
∃ cylinder1 cylinder2 : Geometry.Cylinder,
  Geometry.Skew cylinder1 cylinder2 ∧
  Geometry.PrincipalCircle cylinder1.base sphere ∧
  Geometry.PrincipalCircle cylinder2.base sphere ∧
  Geometry.MutuallyPerpendicular cylinder1.axis cylinder2.axis

theorem geometric_locus_of_lines_is_two_skew_cylinders (sphere : Type) [Geometry.Sphere sphere]
  (projection_plane1 projection_plane2 : Geometry.Plane) [Geometry.Intersects sphere projection_plane1] [Geometry.Intersects sphere projection_plane2]
  (line : Geometry.Line) :
  is_geometric_locus_of_lines sphere projection_plane1 projection_plane2 line :=
by sorry

end geometric_locus_of_lines_is_two_skew_cylinders_l567_567266


namespace range_of_a_has_three_integer_solutions_l567_567982

theorem range_of_a_has_three_integer_solutions (a : ℝ) :
  (∃ (x : ℤ → ℝ), (2 * x - 1 > 3) ∧ (x ≤ 2 * a - 1) ∧ (x = 3 ∨ x = 4 ∨ x = 5)) → (3 ≤ a ∧ a < 3.5) :=
sorry

end range_of_a_has_three_integer_solutions_l567_567982


namespace angle_BDC_is_24_l567_567579

variable {A B C D : Type}
variables (A B C D : Point)

-- Define the given angles as constants
constant angle_BAC : ℝ := 48
constant angle_CAD : ℝ := 66
constant angle_CBD : ℝ := angle_DBA

-- Define the angle BDC
constant angle_BDC : ℝ 

-- The theorem to prove
theorem angle_BDC_is_24 (h1 : angle_BAC = 48) (h2 : angle_CAD = 66) (h3 : angle_CBD = angle_DBA):
  angle_BDC = 24 :=
sorry

end angle_BDC_is_24_l567_567579


namespace solve_inequality_l567_567633

open Real

def inequality (x : ℝ) : Prop := 
  (2 * x + 3) / (x^2 - 2 * x + 4) > (4 * x + 5) / (2 * x^2 + 5 * x + 7)

noncomputable def discrim_positive (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem solve_inequality :
  ∀ x : ℝ, discrim_positive 19 23 1 ∧ (x^2 - 2 * x + 4 ≠ 0) ∧ (2 * x^2 + 5 * x + 7 ≠ 0) →
  inequality x ↔ x ∈ Ioo ((-23 - Real.sqrt 453) / 38) ((-23 + Real.sqrt 453) / 38) := sorry

end solve_inequality_l567_567633


namespace pentagonal_prism_vertices_l567_567895

theorem pentagonal_prism_vertices :
  let bases := 2
  let vertices_per_base := 5
  let connected_vertices := vertices_per_base
  ∀ (b : ℕ) (v : ℕ) (c : ℕ),
    b = bases → 
    v = vertices_per_base → 
    c = connected_vertices → 
    (b * v = 10) → 
    b * v = 10 :=
by { intros b v c hb hv hc h, rw [hb, hv], exact h }

end pentagonal_prism_vertices_l567_567895


namespace positive_difference_l567_567380

def A : ℕ :=
  (list.range' 1 50).sum (λ n, n * (n + 1)) + 51

def B : ℕ :=
  1 + (list.range 1 50).sum (λ n, n * (n + 1)) + 50 * 51

theorem positive_difference :
  |A - B| = 1250 :=
sorry

end positive_difference_l567_567380


namespace part_a_l567_567183

variable (f : ℝ → ℝ)
variable (h_cont : Continuous f)
variable (h_prop : ∀ (x : ℝ), x * f x ≥ ∫ t in 0..x, f t)

theorem part_a (x : ℝ) (h : x ≠ 0) : 
  ((G : ℝ → ℝ) := (λ x, (1/x) * ∫ t in 0..x, f t), (G x) ≥ (G 0)) :=
by sorry

end part_a_l567_567183


namespace find_y_value_l567_567063

noncomputable def valid_y (y : ℚ) : Prop :=
  let P := (-2 : ℚ, 7 : ℚ)
  let Q := (5 : ℚ, y)
  (Q.2 - P.2) / (Q.1 - P.1) = (5 : ℚ) / 3

theorem find_y_value : valid_y (56 / 3) :=
sorry

end find_y_value_l567_567063


namespace FastFoodCost_l567_567359

theorem FastFoodCost :
  let sandwich_cost := 4
  let soda_cost := 1.5
  let fries_cost := 2.5
  let num_sandwiches := 4
  let num_sodas := 6
  let num_fries := 3
  let discount := 5
  let total_cost := (sandwich_cost * num_sandwiches) + (soda_cost * num_sodas) + (fries_cost * num_fries) - discount
  total_cost = 27.5 := 
by
  sorry

end FastFoodCost_l567_567359


namespace tom_invited_siblings_l567_567276

theorem tom_invited_siblings (plates_used : ℕ) (days : ℕ) (meals_per_day : ℕ) (plates_per_meal : ℕ) (num_people : ℕ)
  (Tom_incl : bool) (num_parents : ℕ) : 
  num_people = 6 →
  days = 4 →
  meals_per_day = 3 →
  plates_per_meal = 2 →
  num_people = (plates_used / (plates_per_meal * meals_per_day * days)) →
  Tom_incl = tt →
  num_parents = 2 →
  (plates_used : ℕ) = 144 →
  (num_people - 1) - num_parents = 3 := 
by 
  sorry

end tom_invited_siblings_l567_567276


namespace range_of_a_l567_567085

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (hf : f = λ x => log a (3 * x^2 - 2 * a * x)) :
  (∀ x ∈ (set.Icc (1/2 : ℝ) 1), deriv f x < 0) → 0 < a ∧ a < 3 / 4 := by
  sorry

end range_of_a_l567_567085


namespace no_solution_sin_geometric_sequence_l567_567023

theorem no_solution_sin_geometric_sequence :
  ¬ ∃ a : ℝ, (0 < a ∧ a < 360) ∧
  (∃ r : ℝ, sin (2 * a) = r * sin a ∧ sin (3 * a) = r * sin (2 * a)) :=
by
  sorry

end no_solution_sin_geometric_sequence_l567_567023


namespace expected_pairs_of_red_in_circle_deck_l567_567649

noncomputable def expected_pairs_of_adjacent_red_cards (deck_size : ℕ) (red_cards : ℕ) : ℚ :=
  let adjacent_probability := (red_cards - 1 : ℚ) / (deck_size - 1)
  in red_cards * adjacent_probability

theorem expected_pairs_of_red_in_circle_deck :
  expected_pairs_of_adjacent_red_cards 52 26 = 650 / 51 :=
by
  sorry

end expected_pairs_of_red_in_circle_deck_l567_567649


namespace sum_of_divisors_of_143_l567_567749

theorem sum_of_divisors_of_143 : 
  (∑ d in (finset.filter (λ d, 143 % d = 0) (finset.range (143 + 1))), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l567_567749


namespace quadratic_roots_primes_4_possible_k_l567_567363

theorem quadratic_roots_primes_4_possible_k :
  ∃ k_set: set ℕ, k_set.card = 4 ∧
    (∀ k ∈ k_set, ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧
      p + q = 58 ∧ p * q = k) :=
by sorry

end quadratic_roots_primes_4_possible_k_l567_567363


namespace longest_pencil_l567_567220

/-- Hallway dimensions and the longest pencil problem -/
theorem longest_pencil (L : ℝ) : 
    (∃ P : ℝ, P = 3 * L) :=
sorry

end longest_pencil_l567_567220


namespace find_a_max_value_4a_b_l567_567091

-- Definitions from conditions
def f (x : ℝ) (a : ℝ) := x^2 + 3 * x + a
def g (x : ℝ) (a : ℝ) := (f x a) / (x + 1)
def solution_set_f (a x : ℝ) := f x a < 0

-- Lean 4 Statement: Part 1
theorem find_a (a : ℝ) : (∃ a, solution_set_f a = (λ x, a < x ∧ x < 1)) → a = -4 :=
by
  sorry

-- Lean 4 Statement: Part 2
theorem max_value_4a_b (a b : ℝ) (h : ab < 0) :
  (g b a = b + (1/2) * a) → (4 * a + b) ≤ 9 :=
by
  sorry

end find_a_max_value_4a_b_l567_567091


namespace animal_video_ratio_l567_567367

theorem animal_video_ratio:
  let cat_video := 4 in
  let dog_video := 2 * cat_video in
  let total_time := 36 in
  let combined_cat_dog := cat_video + dog_video in
  let gorilla_video := total_time - combined_cat_dog in
  gorilla_video / combined_cat_dog = 2 :=
by
  sorry

end animal_video_ratio_l567_567367


namespace coloring_ways_3x3_grid_l567_567146

theorem coloring_ways_3x3_grid :
  ∃ ways : ℕ, ways = 1296 ∧
  (∀ (grid : list (list ℕ)), grid.length = 3 → (∀ row, row.length = 3) ∧
  (∀ color, color < 3) →
  (∀ i j, (i < 3 ∧ j < 3) → ∀ k l, (|i - k| + |j - l| = 1) → grid[i]![j] ≠ grid[k]![l]) → ways = 1296) :=
begin
  sorry
end

end coloring_ways_3x3_grid_l567_567146


namespace boy_girl_probability_l567_567466

theorem boy_girl_probability :
  let num_people := 4,
      boys := 2,
      girls := 2,
      total_ways := Nat.choose num_people 2 * (Nat.factorial 2),
      favorable_ways := Nat.choose boys 1 * Nat.choose girls 1
  in
  total_ways = 12 ∧ favorable_ways = 4 →
  (favorable_ways / total_ways : ℚ) = 1 / 3 :=
begin
  sorry
end

end boy_girl_probability_l567_567466


namespace max_min_values_on_interval_l567_567892

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 + 5

theorem max_min_values_on_interval :
  ∃ x_max x_min, x_max ∈ Icc (-2 : ℝ) 2 ∧ x_min ∈ Icc (-2 : ℝ) 2 ∧
  (∀ x ∈ Icc (-2 : ℝ) 2, f(x) ≤ f(x_max)) ∧
  (∀ x ∈ Icc (-2 : ℝ) 2, f(x_min) ≤ f(x)) ∧
  f(x_max) = 5 ∧ f(x_min) = -11 :=
sorry

end max_min_values_on_interval_l567_567892


namespace largest_integer_with_remainder_l567_567430

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l567_567430


namespace collinear_X_Y_Z_l567_567665

-- Define the necessary geometric entities
variables {Point Circle : Type} [Geometry Point Circle]

-- Define the conditions
variables (W1 W2 W3 : Circle)
variable [externally_tangent W1 W2]
variable [externally_tangent W2 W3]
variable [externally_tangent W1 W3]

variable (P1 : Point) [tangent_point P1 W1 W3]
variable (P2 : Point) [tangent_point P2 W2 W3]

variable (A B : Point) [on_circle A W3] [on_circle B W3] [diameter AB W3]

variable (X : Point) [second_intersection (line_through A P1) W1 X]
variable (Y : Point) [second_intersection (line_through B P2) W2 Y]
variable (Z : Point) [intersection (line_through A P2) (line_through B P1) Z]

-- The proof goal
theorem collinear_X_Y_Z : collinear {X, Y, Z} := 
sorry

end collinear_X_Y_Z_l567_567665


namespace question_1_question_2_l567_567509

noncomputable theory
open Set

def A := {x : ℝ | -3 ≤ x ∧ x ≤ 2}
def B (m : ℝ) := {x : ℝ | 1 - m ≤ x ∧ x ≤ 3 * m - 1}

theorem question_1 (m : ℝ) (h_m : m = 3) : 
  (A ∩ B m = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) ∧ 
  (A ∪ B m = {x : ℝ | -3 ≤ x ∧ x ≤ 8}) := 
by {
  rw h_m,
  sorry 
}

theorem question_2 (m : ℝ) (h : A ∩ B m = B m) : 
  m ≤ 1 := 
by {
  have subset_eq := subset.antisymm,
  sorry
}

end question_1_question_2_l567_567509


namespace probability_of_one_triplet_without_any_pairs_l567_567876

noncomputable def probability_one_triplet_no_pairs : ℚ :=
  let total_outcomes := 6^5
  let choices_for_triplet := 6
  let ways_to_choose_triplet_dice := Nat.choose 5 3
  let choices_for_remaining_dice := 5 * 4
  let successful_outcomes := choices_for_triplet * ways_to_choose_triplet_dice * choices_for_remaining_dice
  successful_outcomes / total_outcomes

theorem probability_of_one_triplet_without_any_pairs :
  probability_one_triplet_no_pairs = 25 / 129 := by
  sorry

end probability_of_one_triplet_without_any_pairs_l567_567876


namespace find_a5_l567_567558

def seq (a : ℕ → ℚ) := a 0 = 1 / 3 ∧ ∀ n ≥ 1, a (n+1) = (-1)^(n+1) * 2 * a n

theorem find_a5 : seq a → a 4 = 16 / 3 :=
by
  intro h,
  cases h with h₁ h₂,
  unfold seq at h₁,
  sorry

end find_a5_l567_567558


namespace compute_d_l567_567485

noncomputable def polynomial := Polynomial

theorem compute_d :
  ∀ (c d : ℚ), 
  (polynomial.eval (3 + Real.sqrt 5) (Polynomial.mk [15, d, c, 1]) = 0) →
  (polynomial.eval (3 - Real.sqrt 5) (Polynomial.mk [15, d, c, 1]) = 0) →
  d = -18.5 :=
by
  sorry

end compute_d_l567_567485


namespace tan_sin_cos_ratio_l567_567923

open Real

variable {α β : ℝ}

theorem tan_sin_cos_ratio (h1 : tan (α + β) = 2) (h2 : tan (α - β) = 3) :
  sin (2 * α) / cos (2 * β) = 5 / 7 := sorry

end tan_sin_cos_ratio_l567_567923


namespace expected_heads_after_four_tosses_l567_567878

theorem expected_heads_after_four_tosses :
  ∀ (n : ℕ), n = 50 → (∑ i in Finset.range n, (1/2 + 1/4 + 1/8 + 1/16)) = 50 * (15/16) → round (50 * (15/16)) = 47 :=
by
  intros n hn hsum
  rw hn at hsum
  norm_num at hsum
  rw hsum
  norm_num
  sorry

end expected_heads_after_four_tosses_l567_567878


namespace zero_term_index_l567_567684

noncomputable def sequence (x : ℕ → ℝ) : Prop :=
  ∀ n ≥ 1, x (n + 2) = x n - 1 / x (n + 1)

theorem zero_term_index (x : ℕ → ℝ) (h : sequence x) : 
  ∃ k, x k = 0 ∧ k = 1845 :=
by
  sorry

end zero_term_index_l567_567684


namespace p_divides_mn_minus_1_l567_567909

theorem p_divides_mn_minus_1
  (m n : ℕ)
  (p : ℕ)
  (hp_prime : p.prime)
  (hm_pos : m > 0)
  (hn_pos : n > 0)
  (hmn_bound : m < n ∧ n < p)
  (hp_div_m2_plus_1 : p ∣ (m^2 + 1))
  (hp_div_n2_plus_1 : p ∣ (n^2 + 1)) :
  p ∣ (m * n - 1) :=
  sorry

end p_divides_mn_minus_1_l567_567909


namespace largest_integer_with_remainder_l567_567431

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l567_567431


namespace largest_integer_with_remainder_l567_567427

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l567_567427


namespace sum_squares_alternating_l567_567700

theorem sum_squares_alternating (n : ℕ) (h : n > 0) : 
  ∑ i in finset.range n, (-1) ^ i * (i + 1)^2 = (-1) ^ (n - 1) * (n * (n + 1)) / 2 := 
sorry

end sum_squares_alternating_l567_567700


namespace evaluation_problem_l567_567392

theorem evaluation_problem :
  (8⁻¹ = 1 / 8) → (8⁻³ = 1 / 512) → (3^2 = 9) → 
  (8⁻¹ * 3^2 / 8⁻³ = 576) :=
by
  intros h1 h2 h3
  sorry

end evaluation_problem_l567_567392


namespace point_in_fourth_quadrant_l567_567938

noncomputable def z : ℂ := 2 * (complex.I ^ 3) / (1 - complex.I)

theorem point_in_fourth_quadrant : 
  let coord := complex.re z, complex.im z in
  coord.1 > 0 ∧ coord.2 < 0 :=
by
  sorry

end point_in_fourth_quadrant_l567_567938


namespace find_a_b_and_m_range_l567_567502

-- Definitions and initial conditions
def f (x : ℝ) (a b m : ℝ) : ℝ := 2*x^3 + a*x^2 + b*x + m
def f_prime (x : ℝ) (a b : ℝ) : ℝ := 6*x^2 + 2*a*x + b

-- Problem statement
theorem find_a_b_and_m_range (a b m : ℝ) :
  (∀ x, f_prime x a b = 6 * (x + 0.5)^2 - k) →
  f_prime 1 a b = 0 →
  a = 3 ∧ b = -12 ∧ -20 < m ∧ m < 7 :=
sorry

end find_a_b_and_m_range_l567_567502


namespace smile_area_eq_l567_567824

theorem smile_area_eq :
  let A := (0, 0)
  let B := (4, 0)
  let C := (2, 2)
  let D := (2, 0)
  let radius := 2
  ∃ E F: ℝ × ℝ,
    let lengthBD := 2
    let lengthBE := 3 * lengthBD
    let lengthAF := 3 * lengthBD
    let radiusBE := lengthBE
    let radiusAF := lengthAF
    let radiusDEF := radiusBE - radius
    let sector_area s r := 1 / 2 * r^2 * (real.pi / 2)
    let semicircle_area r := 1 / 2 * real.pi * r^2
    let triangle_area b h := 1 / 2 * b * h
    sector_area B radiusBE + sector_area A radiusAF + sector_area D radiusDEF
    - semicircle_area radius - triangle_area 4 2 = 20 * real.pi - 2 :=
  sorry

end smile_area_eq_l567_567824


namespace johns_cost_per_sheet_correct_l567_567830

noncomputable def johns_cost_per_sheet : ℝ := 2.75

-- Define the total cost functions for both photo stores
def johns_photo_world_cost (sheets : ℕ) (cost_per_sheet : ℝ) : ℝ :=
  sheets * cost_per_sheet + 125

def sams_picture_emporium_cost (sheets : ℕ) : ℝ :=
  sheets * 1.50 + 140

-- Prove John's Photo World charges $2.75 per sheet if both stores cost the same for 12 sheets
theorem johns_cost_per_sheet_correct :
  (johns_photo_world_cost 12 johns_cost_per_sheet) = (sams_picture_emporium_cost 12) :=
by
  have h : 12 * johns_cost_per_sheet + 125 = 12 * 1.50 + 140 := by
    have : 12 * 1.50 = 18 := by norm_num
    have : 18 + 140 = 158 := by norm_num
    linarith
  exact h

end johns_cost_per_sheet_correct_l567_567830


namespace line_KL_passes_through_midpoint_of_AB_l567_567643

open EuclideanGeometry

variable {A B C D K L M O: Point}
variable {α β : ℝ}

def acute_triangle (A B C: Point) : Prop := 
  ∠ACB < π / 2 ∧ ∠BAC < π / 2 ∧ ∠ABC < π / 2

def on_circle (O: Point) (A: Point) (r: ℝ): Prop :=
  dist O A = r

def triangle_inequality (A B C: Point) (AC BC: ℝ) : Prop := 
  dist A C > dist B C

def midpoint (M A B: Point) : Prop := 
  dist M A = dist M B

def diameter (O C D: Point): Prop :=
  collinear O C D ∧ dist O C = dist O D / 2

theorem line_KL_passes_through_midpoint_of_AB:
  (acute_triangle A B C)
  → (∃ r : ℝ, on_circle O A r ∧ on_circle O B r ∧ on_circle O C r ∧ on_circle O D r)
  → diameter O C D
  → triangle_inequality A B C (dist A C) (dist B C)
  → ray DA K
  → on_segment B D L
  → dist D L > dist L B
  → ∠OKD = ∠BAC
  → ∠OLD = ∠ABC
  → ∃ M, midpoint M A B ∧ line_passing_through K L M :=
begin
  sorry
end

end line_KL_passes_through_midpoint_of_AB_l567_567643


namespace sqrt_inequality_l567_567489

theorem sqrt_inequality (x : ℝ) (h₁ : 3 / 2 ≤ x) (h₂ : x ≤ 5) : 
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := 
sorry

end sqrt_inequality_l567_567489


namespace inequality_for_a_ne_1_l567_567629

theorem inequality_for_a_ne_1 (a : ℝ) (h : a ≠ 1) : (1 + a + a^2)^2 < 3 * (1 + a^2 + a^4) :=
sorry

end inequality_for_a_ne_1_l567_567629


namespace largest_area_similar_triangle_l567_567099

variables {Point : Type*} [MetricSpace Point]

structure Triangle (Point : Type*) :=
(A₁ A₂ A₃ : Point)

def is_similar (T₁ T₂ : Triangle Point) : Prop :=
sorry  -- Define similarity between two triangles

def lies_on (P : Point) (l : Line Point) : Prop :=
sorry  -- Define the relation for a point lying on a line

noncomputable def largest_area_triangle (A B : Triangle Point) : Triangle Point :=
sorry  -- Define the construction of the triangle with the largest area

theorem largest_area_similar_triangle
  (A B : Triangle Point)
  (h₁ : is_similar (largest_area_triangle A B) B)
  (h₂ : lies_on A.A₁ (line_through (largest_area_triangle A B).A₂ (largest_area_triangle A B).A₃))
  (h₃ : lies_on A.A₂ (line_through A.A₁ A.A₃))
  (h₄ : lies_on A.A₃ (line_through A.A₁ A.A₂)) :
  (largest_area_triangle A B) = family_S_123 A B :=
sorry

end largest_area_similar_triangle_l567_567099


namespace main_problem_l567_567939

noncomputable def ellipseStandardEquation : Prop :=
∃ a b : ℝ, ∃ e : ℝ, ∃ c : ℝ,
  (a = Real.sqrt 6) ∧ (b = Real.sqrt 5) ∧ (e = Real.sqrt 6 / 6) ∧ (2 * c = 2) ∧ 
  ((c / a = e) ∧ (b = Real.sqrt (a^2 - c^2)) ∧ 
   (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 6 + y^2 / 5 = 1)))

noncomputable def lineIntersectionCondition (m : ℝ) : Prop :=
∀ x1 x2 : ℝ, ∃ a b : ℝ,
  (11 * x1^2 + 12 * m * x1 + 6 * m^2 - 30 = 0) ∧
  (11 * x2^2 + 12 * m * x2 + 6 * m^2 - 30 = 0) ∧
  (a = 2 * x1 * x2 + m * (x1 + x2) + m^2) ∧ (a = 0)

noncomputable def solveForM : Prop :=
∀ m : ℝ, lineIntersectionCondition m → (m = 2 * Real.sqrt 165 / 11) ∨ (m = -2 * Real.sqrt 165 / 11)

theorem main_problem : ellipseStandardEquation ∧ solveForM :=
begin
  sorry
end

end main_problem_l567_567939


namespace dolphins_scored_15_l567_567541

theorem dolphins_scored_15 (s d : ℤ) 
  (h1 : s + d = 48) 
  (h2 : s - d = 18) : 
  d = 15 := 
sorry

end dolphins_scored_15_l567_567541


namespace largest_int_less_than_100_remainder_4_l567_567415

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l567_567415


namespace f_g_of_1_l567_567527

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 5 * x + 6
def g (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

-- The statement we need to prove
theorem f_g_of_1 : f (g 1) = 132 := by
  sorry

end f_g_of_1_l567_567527


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567437

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567437


namespace largest_int_mod_6_less_than_100_l567_567448

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l567_567448


namespace meaningful_sqrt_range_l567_567274

theorem meaningful_sqrt_range (x : ℝ) : (3 * x - 6 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end meaningful_sqrt_range_l567_567274


namespace calculate_total_boundary_length_l567_567349

-- Definitions
def area_square (s : ℝ) : ℝ := s * s
def semi_circle_length (r : ℝ) : ℝ := π * r
def total_boundary_length : ℝ :=
  let s := real.sqrt 256
  let segment_length := s / 4
  let semi_circle_radius := segment_length / 2
  let semi_circle_total_length := 4 * 3 * semi_circle_length semi_circle_radius
  let straight_segment_length := 4 * 3 * segment_length
  semi_circle_total_length + straight_segment_length

-- Main theorem
theorem calculate_total_boundary_length :
  real.round_to_nearest (total_boundary_length) = 123.4 :=
by
  sorry

end calculate_total_boundary_length_l567_567349


namespace mandy_chocolate_l567_567327

theorem mandy_chocolate (total : ℕ) (h1 : total = 60)
  (michael : ℕ) (h2 : michael = total / 2)
  (paige : ℕ) (h3 : paige = (total - michael) / 2) :
  (total - michael - paige = 15) :=
by
  -- By hypothesis: total = 60, michael = 30, paige = 15
  sorry 

end mandy_chocolate_l567_567327


namespace regular_polygon_double_sides_l567_567536

theorem regular_polygon_double_sides (n : ℕ) (h1 : 2 ≤ n) :
  let A := (n - 2) * 180 / n, B := (2 * n - 2) * 180 / (2 * n)
  in B = A + 15 → n = 12 := 
by
  sorry

end regular_polygon_double_sides_l567_567536


namespace city_phone_number_remainder_l567_567207

theorem city_phone_number_remainder :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧
  (312837 % n = 96) ∧ (310650 % n = 96) := sorry

end city_phone_number_remainder_l567_567207


namespace relationship_between_a_b_c_l567_567928

theorem relationship_between_a_b_c :
  (∃ a : ℝ, 5^(20 * a) + 12^(20 * a) = 13^(20 * a)) →
  let b := Real.exp 0.1 - 1 in
  let c := Real.tan 0.1 in
  b > c ∧ c > (2 : ℝ) / 20 :=
by
  intro h
  rcases h with ⟨a, ha⟩
  let b : ℝ := Real.exp 0.1 - 1
  let c : ℝ := Real.tan 0.1
  have ha : a = 0.1 := by sorry
  have hc : c > a := by sorry
  have hb : b > c := by sorry
  exact ⟨hb, hc⟩

end relationship_between_a_b_c_l567_567928


namespace problem_statement_l567_567678

def otimes (a b : ℝ) : ℝ := b^2 / a

theorem problem_statement : 
  (otimes (otimes 3 1) 6) - (otimes (otimes 3 6) 1) = 1295 / 12 :=
by sorry

end problem_statement_l567_567678


namespace number_of_integer_solutions_l567_567955

open Real

theorem number_of_integer_solutions (π_approx : ℝ) (hπ : π_approx = 3.14) :
  {x : ℤ | abs x < 3 * π_approx}.to_finset.card = 19 := 
sorry

end number_of_integer_solutions_l567_567955


namespace find_m_for_split_l567_567901

theorem find_m_for_split (m : ℕ) (h1 : m > 1) (h2 : ∃ k, k < m ∧ 2023 = (m^2 - m + 1) + 2*k) : m = 45 :=
sorry

end find_m_for_split_l567_567901


namespace sum_of_divisors_143_l567_567742

theorem sum_of_divisors_143 : 
  (finset.range (144)).filter (λ d, 143 % d = 0).sum = 168 :=
by
  sorry

end sum_of_divisors_143_l567_567742


namespace find_smallest_in_arithmetic_progression_l567_567038

theorem find_smallest_in_arithmetic_progression (a d : ℝ)
  (h1 : (a-2*d)^3 + (a-d)^3 + a^3 + (a+d)^3 + (a+2*d)^3 = 0)
  (h2 : (a-2*d)^4 + (a-d)^4 + a^4 + (a+d)^4 + (a+2*d)^4 = 136) :
  (a - 2*d) = -2 * Real.sqrt 2 :=
sorry

end find_smallest_in_arithmetic_progression_l567_567038


namespace concyclicity_of_points_l567_567983

theorem concyclicity_of_points
  (A B C O P Q : Type)
  [CircumcenterTriangle O A B C]
  (h1 : P ∈ line_segment(A, B))
  (h2 : Q ∈ line_segment(A, C))
  (h3 : dist(B, P) / dist(A, C) = dist(P, Q) / dist(C, B))
  (h4 : dist(P, Q) / dist(C, B) = dist(Q, C) / dist(B, A)) 
  : Concyclic {A, P, Q, O} :=
sorry

end concyclicity_of_points_l567_567983


namespace cube_volume_l567_567237

theorem cube_volume (d : ℝ) (s : ℝ) (h₁ : d = 6 * real.sqrt 3) (h₂ : d = s * real.sqrt 3) : s^3 = 216 := by
  sorry

end cube_volume_l567_567237


namespace Lizzie_group_number_l567_567186

theorem Lizzie_group_number (x : ℕ) (h1 : x + (x + 17) = 91) : x + 17 = 54 :=
by
  sorry

end Lizzie_group_number_l567_567186


namespace roy_total_pens_l567_567210

theorem roy_total_pens (b : ℕ) (h1 : b = 5) (h2 : ∃ k, k = 3 * b) 
 (h3 : ∃ r, r = 2 * (3 * b) - 4) : 
  ∃ t, t = b + (3 * b) + (2 * (3 * b) - 4) := by
  let b := 5
  let k := 3 * b
  let r := 2 * k - 4
  let t := b + k + r
  have : t = 46 := sorry
  exact ⟨t, this⟩

end roy_total_pens_l567_567210


namespace probability_of_interval_l567_567055

theorem probability_of_interval (a : ℝ) (h : a ∈ Ioc 15 25) : 
  ∃ p : ℚ, p = 3 / 10 ∧ Prob (λ a, 17 < a ∧ a < 20 ∣ 15 < a ∧ a ≤ 25 ) p := 
sorry

end probability_of_interval_l567_567055


namespace max_sector_area_l567_567933

theorem max_sector_area (r θ : ℝ) (h₁ : 2 * r + r * θ = 16) : 
  (∃ A : ℝ, A = 1/2 * r^2 * θ ∧ A ≤ 16) ∧ (∃ r θ, r = 4 ∧ θ = 2 ∧ 1/2 * r^2 * θ = 16) := 
by
  sorry

end max_sector_area_l567_567933


namespace proportional_segments_l567_567108

-- Define the problem
theorem proportional_segments :
  ∀ (a b c d : ℕ), a = 6 → b = 9 → c = 12 → (a * d = b * c) → d = 18 :=
by
  intros a b c d ha hb hc hrat
  rw [ha, hb, hc] at hrat
  exact sorry

end proportional_segments_l567_567108


namespace unique_function_property_l567_567407

def f (n : Nat) : Nat := sorry

theorem unique_function_property :
  (∀ x y : ℕ+, x < y → f x < f y) ∧
  (∀ y x : ℕ+, f (y * f x) = x^2 * f (x * y)) →
  ∀ n : ℕ+, f n = n^2 :=
by
  intros h
  sorry

end unique_function_property_l567_567407


namespace number_of_black_squares_in_58th_row_l567_567321

theorem number_of_black_squares_in_58th_row :
  let pattern := [1, 0, 0] -- pattern where 1 represents a black square
  let n := 58
  let total_squares := 2 * n - 1 -- total squares in the 58th row
  let black_count := total_squares / 3 -- number of black squares in the repeating pattern
  black_count = 38 :=
by
  let pattern := [1, 0, 0]
  let n := 58
  let total_squares := 2 * n - 1
  let black_count := total_squares / 3
  have black_count_eq_38 : 38 = (115 / 3) := by sorry
  exact black_count_eq_38.symm

end number_of_black_squares_in_58th_row_l567_567321


namespace intersection_x_diff_l567_567871

-- Define the line equation
def line_eq (x : ℝ) : ℝ := 3 * x + 1

-- Define the parabola equation
def parabola_eq (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem intersection_x_diff :
  let a := min 3 5 in
  let c := max 3 5 in
  c - a = 2 :=
by
  sorry

end intersection_x_diff_l567_567871


namespace new_concentration_of_mixture_l567_567301

theorem new_concentration_of_mixture
  (v1_cap : ℝ) (v1_alcohol_percent : ℝ)
  (v2_cap : ℝ) (v2_alcohol_percent : ℝ)
  (new_vessel_cap : ℝ) (poured_liquid : ℝ)
  (filled_water : ℝ) :
  v1_cap = 2 →
  v1_alcohol_percent = 0.25 →
  v2_cap = 6 →
  v2_alcohol_percent = 0.50 →
  new_vessel_cap = 10 →
  poured_liquid = 8 →
  filled_water = (new_vessel_cap - poured_liquid) →
  ((v1_cap * v1_alcohol_percent + v2_cap * v2_alcohol_percent) / new_vessel_cap) = 0.35 :=
by
  intros v1_h v1_per_h v2_h v2_per_h v_new_h poured_h filled_h
  sorry

end new_concentration_of_mixture_l567_567301


namespace sum_of_cubes_8001_l567_567254
-- Import the entire Mathlib library

-- Define a property on integers
def approx (x y : ℝ) := abs (x - y) < 0.000000000000004

-- Define the variables a and b
variables (a b : ℤ)

-- State the theorem
theorem sum_of_cubes_8001 (h : approx (a * b : ℝ) 19.999999999999996) : a^3 + b^3 = 8001 := 
sorry

end sum_of_cubes_8001_l567_567254


namespace find_Sn_l567_567911

noncomputable def sequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
a 1 = -1 ∧ (∀ n, a (n + 1) = S (n + 1) * S n)

theorem find_Sn (a : ℕ → ℚ) (S : ℕ → ℚ) (h : sequence a S) : 
  ∀ n, S n = -1 / n :=
by
  sorry

end find_Sn_l567_567911


namespace necessary_and_sufficient_l567_567312

theorem necessary_and_sufficient (a b : ℝ) : 
  a > b ↔ (2^a > 2^b) :=
by 
  sorry

end necessary_and_sufficient_l567_567312


namespace expected_adjacent_red_pairs_l567_567656

open Probability

def num_all_cards : ℕ := 52
def num_red_cards : ℕ := 26

/-- The expected number of pairs of adjacent cards which are both red 
     in a standard 52-card deck dealt out in a circle is 650/51. -/
theorem expected_adjacent_red_pairs : 
  let p_red_right : ℚ := 25 / 51 in
  let expected_pairs := (num_red_cards : ℚ) * p_red_right
  in expected_pairs = 650 / 51 :=
by
  sorry

end expected_adjacent_red_pairs_l567_567656


namespace largest_int_mod_6_less_than_100_l567_567447

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l567_567447


namespace chair_arrays_48_l567_567344

theorem chair_arrays_48 :
  let n := 48
  (∀ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ a * b = n) → (let num_configs := 4 in let unique_arrays := num_configs * 2 in unique_arrays = 8) :=
by
  sorry

end chair_arrays_48_l567_567344


namespace radius_ratio_of_hyperbola_l567_567917

theorem radius_ratio_of_hyperbola {a b : ℝ} (h1 : a > 0) (h2 : b > 0)
  {F₁ F₂ : Point} (h3 : is_foci F₁ F₂ (Hyperbola a b))
  {G : Point} (h4 : lies_on_hyperbola G (Hyperbola a b))
  (h5 : ∥G - F₁∥ = 7 * ∥G - F₂∥) :
  0 < b / a ∧ b / a ≤ sqrt(7) / 3 := 
sorry

end radius_ratio_of_hyperbola_l567_567917


namespace candy_bars_saved_l567_567574

theorem candy_bars_saved
  (candy_bars_per_week : ℕ)
  (weeks : ℕ)
  (candy_bars_eaten_per_4_weeks : ℕ) :
  candy_bars_per_week = 2 →
  weeks = 16 →
  candy_bars_eaten_per_4_weeks = 1 →
  (candy_bars_per_week * weeks) - (weeks / 4 * candy_bars_eaten_per_4_weeks) = 28 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end candy_bars_saved_l567_567574


namespace find_lambda_l567_567513

-- Definitions based on the conditions
def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (0, 1)

-- Dot product definition
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The required proof problem
theorem find_lambda : ∃ λ : ℝ, dot_product b (λ • a + b) = 0 ∧ λ = -1 :=
by
  sorry

end find_lambda_l567_567513


namespace product_of_consecutive_integers_l567_567645

theorem product_of_consecutive_integers
  (a b : ℕ) (n : ℕ)
  (h1 : a = 12)
  (h2 : b = 22)
  (mean_five_numbers : (a + b + n + (n + 1) + (n + 2)) / 5 = 17) :
  (n * (n + 1) * (n + 2)) = 4896 := by
  sorry

end product_of_consecutive_integers_l567_567645


namespace rain_forest_animals_l567_567228

theorem rain_forest_animals (R : ℕ) 
  (h1 : 16 = 3 * R - 5) : R = 7 := 
  by sorry

end rain_forest_animals_l567_567228


namespace length_of_AC_l567_567578

variable {A B C H O K : Type}
variable [triangle : Triangle A B C]
variable [circumradius : Circumradius A B C 1]
variable [orthocenter : Orthocenter A B C H]
variable [circumcenter : Circumcenter A B C O]
variable [circle_center : CircleCenterPassingThrough A C H K]
variable [lies_on : LiesOnCircumcircle K (TriangleCircumcircle A B C)]

theorem length_of_AC : |AC| = sqrt(3) := sorry

end length_of_AC_l567_567578


namespace bottles_last_days_l567_567385

theorem bottles_last_days :
  let total_bottles := 8066
  let bottles_per_day := 109
  total_bottles / bottles_per_day = 74 :=
by
  sorry

end bottles_last_days_l567_567385


namespace Bowen_total_spent_l567_567358

def pencil_price : ℝ := 0.25
def pen_price : ℝ := 0.15
def num_pens : ℕ := 40

def num_pencils := num_pens + (2 / 5) * num_pens

theorem Bowen_total_spent : num_pencils * pencil_price + num_pens * pen_price = 20 := by
  sorry

end Bowen_total_spent_l567_567358


namespace simplify_complex_fraction_l567_567631

section ComplexMath
variable (i : ℂ) (h : i ^ 2 = -1)

theorem simplify_complex_fraction : ((2-i)/(3+4i) = 2/5 - (11/25)*i) :=
by
  have h1 : i ^ 2 = -1 := h
  sorry
end

end simplify_complex_fraction_l567_567631


namespace trig_identity_l567_567786

noncomputable theory

open Real

theorem trig_identity (α : ℝ) : 
  (sin (4 * α) / (1 + cos (4 * α))) * (cos (2 * α) / (1 + cos (2 * α))) = cott (3 / 2 * π - α) :=
by
  -- Proof goes here
  sorry

end trig_identity_l567_567786


namespace quadratic_roots_real_distinct_l567_567863

theorem quadratic_roots_real_distinct (k : ℝ) :
  ∆ = (b^2 - 4 * a * c) ∧ ∆ = 18 ∧ a = 4 ∧ b = -6 * sqrt 3 ∧ c = k → k = 45 / 8 ∧ ∆ > 0 :=
by
  intro h
  have : a = 4 := h.2.2.2.1
  have : b = -6 * sqrt 3 := h.2.2.2.2.1
  have : c = k := h.2.2.2.2.2
  have : ∆ = 18 := h.2.1
  have : ∆ = b^2 - 4 * a * c := h.1
  sorry

end quadratic_roots_real_distinct_l567_567863


namespace tree_shadow_length_l567_567145

theorem tree_shadow_length (jane_shadow : ℝ) (jane_height : ℝ) (tree_height : ℝ) (tree_shadow : ℝ)
  (h₁ : jane_shadow = 0.5)
  (h₂ : jane_height = 1.5)
  (h₃ : tree_height = 30)
  (h₄ : jane_height / jane_shadow = tree_height / tree_shadow)
  : tree_shadow = 10 :=
by
  -- skipping the proof steps
  sorry

end tree_shadow_length_l567_567145


namespace number_of_sides_of_regular_polygon_l567_567076

theorem number_of_sides_of_regular_polygon (n : ℕ) (h : ∀ i, interior_angle (polygon n) i = 140) : 
  n = 9 :=
by sorry

end number_of_sides_of_regular_polygon_l567_567076


namespace cave_depth_l567_567116

theorem cave_depth 
  (total_depth : ℕ) 
  (remaining_depth : ℕ) 
  (h1 : total_depth = 974) 
  (h2 : remaining_depth = 386) : 
  total_depth - remaining_depth = 588 := 
by 
  sorry

end cave_depth_l567_567116


namespace min_triangles_four_points_l567_567263

-- Definition of points not collinear
variables (A B C D : Type) 

-- Hypotheses that no three of the points are collinear
def no_three_collinear (A B C D : Type) : Prop :=
  ∀ (P Q R : Type), P ≠ Q → Q ≠ R → P ≠ R → 
  ¬ collinear [P, Q, R]

-- Proof statement
theorem min_triangles_four_points (A B C D : Type) 
  (h : no_three_collinear A B C D) :
  (num_triangles A B C D = 4) := 
sorry

end min_triangles_four_points_l567_567263


namespace numbers_diff_1_2_3_from_26_out_of_100_l567_567256

theorem numbers_diff_1_2_3_from_26_out_of_100 :
  ∀ (cards : List ℕ), (∀ n ∈ cards, n ∈ (List.range 100).map (λ x => x + 1)) →
  cards.length = 26 →
  ∃ (n m ∈ cards), n ≠ m ∧ (abs (n - m) = 1 ∨ abs (n - m) = 2 ∨ abs (n - m) = 3) :=
by
  intros cards h_range h_len
  sorry

end numbers_diff_1_2_3_from_26_out_of_100_l567_567256


namespace solve_for_x_l567_567673

theorem solve_for_x (x : ℝ) :
  let area_square1 := (2 * x) ^ 2
  let area_square2 := (5 * x) ^ 2
  let area_triangle := 0.5 * (2 * x) * (5 * x)
  (area_square1 + area_square2 + area_triangle = 850) → x = 5 := by
  sorry

end solve_for_x_l567_567673


namespace sum_a_1_to_100_l567_567084

def f (n : ℕ) : ℤ :=
  if even n then n^2 else -n^2

def a (n : ℕ) : ℤ :=
  f n + f (n + 1)

theorem sum_a_1_to_100 : 
  (Finset.sum (Finset.range 100) (λ n, a (n + 1))) = -100 :=
sorry

end sum_a_1_to_100_l567_567084


namespace exists_smallest_positive_period_even_function_l567_567353

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

noncomputable def functions : List (ℝ → ℝ) :=
  [
    (λ x => Real.sin (2 * x + Real.pi / 2)),
    (λ x => Real.cos (2 * x + Real.pi / 2)),
    (λ x => Real.sin (2 * x) + Real.cos (2 * x)),
    (λ x => Real.sin x + Real.cos x)
  ]

def smallest_positive_period_even_function : ℝ → Prop :=
  λ T => ∃ f ∈ functions, is_even_function f ∧ period f T ∧ T > 0

theorem exists_smallest_positive_period_even_function :
  smallest_positive_period_even_function Real.pi :=
sorry

end exists_smallest_positive_period_even_function_l567_567353


namespace solution_set_for_inequality_l567_567034

theorem solution_set_for_inequality :
  {x : ℝ | -x^2 + 4 * x + 5 < 0} = {x : ℝ | x > 5 ∨ x < -1} :=
by
  sorry

end solution_set_for_inequality_l567_567034


namespace remainder_mod_88_l567_567390

noncomputable def binom : ℕ → ℕ → ℕ := λ n k => Nat.choose n k

def expression : ℤ :=
  1 - 90 * binom 10 1 + 90^2 * binom 10 2 - 90^3 * binom 10 3 +
  90^4 * binom 10 4 - 90^5 * binom 10 5 + 90^6 * binom 10 6 - 
  90^7 * binom 10 7 + 90^8 * binom 10 8 - 90^9 * binom 10 9 +
  90^10 * binom 10 10

theorem remainder_mod_88 : expression % 88 = 1 := by
  sorry

end remainder_mod_88_l567_567390


namespace sq_diff_eq_binom_identity_l567_567850

variable (a b : ℝ)

theorem sq_diff_eq_binom_identity : (a - b) ^ 2 = a ^ 2 - 2 * a * b + b ^ 2 :=
by
  sorry

end sq_diff_eq_binom_identity_l567_567850


namespace limit_of_f_at_1_l567_567792

noncomputable def f (x : ℝ) : ℝ := (3 - real.sqrt (10 - x)) / real.sin (3 * real.pi * x)

theorem limit_of_f_at_1 :
  filter.tendsto (λ x : ℝ, f x) (nhds 1) (nhds (-1 / (18 * real.pi))) := sorry

end limit_of_f_at_1_l567_567792


namespace find_coals_per_bag_l567_567332

open Nat

variable (burnRate : ℕ) (timePerSet : ℕ) (totalTime : ℕ) (totalBags : ℕ)

def coal_per_bag (burnRate : ℕ) (timePerSet : ℕ) (totalTime : ℕ) (totalBags : ℕ) : ℕ :=
  (totalTime / timePerSet) * burnRate / totalBags

theorem find_coals_per_bag :
  coal_per_bag 15 20 240 3 = 60 :=
by
  sorry

end find_coals_per_bag_l567_567332


namespace problem_equivalence_l567_567881

theorem problem_equivalence (x : ℝ) (h1 : x > 6) :
  (sqrt (x - 6 * sqrt (x - 6)) + 3 = sqrt (x + 6 * sqrt (x - 6)) - 3) ↔ (x ≥ 18) :=
by
  sorry

end problem_equivalence_l567_567881


namespace cos_value_l567_567048

variable (α : Real)

theorem cos_value :
  (sin ((π / 6) - α) = 1 / 3) →
  cos ((5 * π / 3) + 2 * α) = 7 / 9 :=
by
  intro h
  sorry

end cos_value_l567_567048


namespace probability_grid_white_l567_567322

-- Define the conditions as a structure
structure grid_conditions :=
  (unit_squares : list (bool))
  (grid_size : nat := 3)
  (each_square_white_black_random : bool)
  (rotated_90_clockwise : bool)
  (black_to_white_if_previously_white_after_rotation : bool)
  (other_squares_unchanged : bool)

def all_squares_white_after_operations (conds : grid_conditions) : Prop :=
  let probability_grid_white := 1 / 512
  (conds.grid_size = 3) ∧
  (conds.each_square_white_black_random = true) ∧
  (conds.rotated_90_clockwise = true) ∧
  (conds.black_to_white_if_previously_white_after_rotation = true) ∧
  (conds.other_squares_unchanged = true) →
  sorry -- proof steps

theorem probability_grid_white (conds : grid_conditions) :
  all_squares_white_after_operations conds :=
sorry

end probability_grid_white_l567_567322


namespace cube_vertex_coloring_l567_567105

theorem cube_vertex_coloring (cube_vertices_coloring : (fin 8 → bool)) :
  ∃ (valid_colorings : ℕ), valid_colorings = 8 :=
by
  sorry

end cube_vertex_coloring_l567_567105


namespace minimum_lightest_weight_l567_567703

-- Definitions
def lightest_weight (m : ℕ) : Prop := ∃ n, 72 * m = 35 * n ∧ m % 35 = 0 ∧ m ≥ 35

-- Theorem statement
theorem minimum_lightest_weight : ∃ m, lightest_weight m ∧ m = 35 :=
by
  use 35
  split
  sorry
  exact rfl

end minimum_lightest_weight_l567_567703


namespace smallest_positive_period_minimum_value_of_f_center_of_symmetry_interval_of_increasing_l567_567944

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - Real.pi / 6)

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem minimum_value_of_f :
  ∃ x, f x = -3 :=
sorry

theorem center_of_symmetry (k : ℤ) :
  ∃ p, (∀ x, f (p + x) = f (p - x)) ∧ p = (Real.pi / 12) + (k * Real.pi / 2) :=
sorry

theorem interval_of_increasing (k : ℤ) :
  ∃ a b, a = -(Real.pi / 6) + k * Real.pi ∧ b = (Real.pi / 3) + k * Real.pi ∧
  ∀ x, (a <= x ∧ x <= b) → StrictMonoOn f (Set.Icc a b) :=
sorry

end smallest_positive_period_minimum_value_of_f_center_of_symmetry_interval_of_increasing_l567_567944


namespace sum_of_divisors_143_l567_567727

theorem sum_of_divisors_143 : sigma 143 = 168 := by
  have h : 143 = 11 * 13 := by norm_num
  rw [sigma_mul_prime_prime 11 13]
  norm_num
  sorry

end sum_of_divisors_143_l567_567727


namespace sum_integers_30_to_50_subtract_15_l567_567851

-- Definitions and proof problem based on conditions
def sumIntSeries (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_30_to_50_subtract_15 : sumIntSeries 30 50 - 15 = 825 := by
  -- We are stating that the sum of the integers from 30 to 50 minus 15 is equal to 825
  sorry


end sum_integers_30_to_50_subtract_15_l567_567851


namespace mean_of_temperatures_l567_567241

theorem mean_of_temperatures :
  let temps := [84, 86, 85, 87, 89, 90, 88] in
  (temps.sum : ℝ) / temps.length = 87 :=
by
  sorry

end mean_of_temperatures_l567_567241


namespace probability_product_zero_l567_567270

theorem probability_product_zero :
  let s := (-2 :: -1 :: 0 :: 0 :: 3 :: 4 :: 5 :: []) in
  (∃ l : List ℤ, l.length = 3 ∧ l.allDiff ∧ l ⊆ s ∧ 0 ∈ l) →
  (((List.filter (λ l : List ℤ, l.length = 3 ∧ l.allDiff ∧ l ⊆ s ∧ l.prod = 0) (s.powerset)).length.toReal) /
  ((List.filter (λ l : List ℤ, l.length = 3 ∧ l.allDiff ∧ l ⊆ s) (s.powerset)).length.toReal) = 5 / 7) :=
by
  sorry

end probability_product_zero_l567_567270


namespace problem_statement_l567_567885

theorem problem_statement (x : ℝ) (h : x > 6) : 
  (sqrt (x - 6 * sqrt (x - 6)) + 3 = sqrt (x + 6 * sqrt (x - 6)) - 3) ↔ (x ∈ Set.Ici 18) :=
sorry

end problem_statement_l567_567885


namespace solution_correct_l567_567122

noncomputable def problem_lean : Prop :=
  ∃ (DB EC FA : ℕ),
    (DB + EC + FA = 12) ∧
    (DB + 1 = EC) ∧
    (EC + 1 = FA) ∧
    (AB = 26) ∧
    (AD = 7) ∧
    (DE = 10) ∧
    (EF = 12) ∧
    (DB = 3) ∧
    (EC = 4) ∧
    (FA = 5) ∧
    (let s := (DB + EC + FA) / 2 in
     let area := Math.sqrt (s * (s - DB) * (s - EC) * (s - FA)) in
     area = 20)

theorem solution_correct : problem_lean := 
begin
  -- Definitions
  let DB := 3,
  let EC := 4,
  let FA := 5,
  -- Ensuring DB, EC, and FA are consecutive integers
  have h1 : DB + 1 = EC, by norm_num,
  have h2 : EC + 1 = FA, by norm_num,
  -- Ensuring AB, AD, DE, EF, and the area condition
  have h3 : AB = 26, by norm_num,
  have h4 : AD = 7, by norm_num,
  have h5 : DE = 10, by norm_num,
  have h6 : EF = 12, by norm_num,
  let s := (DB + EC + FA) / 2,
  let area := Math.sqrt (s * (s - DB) * (s - EC) * (s - FA)),
  have h7 : area = 20, by sorry,
  -- Combining all conditions
  use [DB, EC, FA],
  refine ⟨_, h1, h2, h3, h4, h5, h6, rfl, rfl, rfl, h7⟩,
  calc
    DB + EC + FA = 3 + 4 + 5 : by norm_num,
end

end solution_correct_l567_567122


namespace find_smallest_in_arith_prog_l567_567039

theorem find_smallest_in_arith_prog (a d : ℝ) 
    (h1 : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)
    (h2 : (a - 2 * d)^4 + (a - d)^4 + a^4 + (a + d)^4 + (a + 2 * d)^4 = 136) :
    a = -2 * Real.sqrt 2 ∨ a = 2 * Real.sqrt 2 :=
begin
  -- sorry placeholder for proof steps
  sorry
end

end find_smallest_in_arith_prog_l567_567039


namespace root_implies_m_values_l567_567965

theorem root_implies_m_values (m : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (m + 2) * x^2 - 2 * x + m^2 - 2 * m - 6 = 0) →
  (m = 3 ∨ m = -2) :=
by
  sorry

end root_implies_m_values_l567_567965


namespace maximum_value_of_x_l567_567926

theorem maximum_value_of_x (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) :
  let x := min 1 (min a (b / (a^2 + b^2)))
  in x <= (Real.sqrt 2) / 2 :=
sorry

end maximum_value_of_x_l567_567926


namespace pressure_relation_l567_567308

-- Definitions from the problem statement
variables (Q Δu A k x P S ΔV V R T T₀ c_v n P₀ V₀ : ℝ)
noncomputable def first_law := Q = Δu + A
noncomputable def Δu_def := Δu = c_v * (T - T₀)
noncomputable def A_def := A = (k * x^2) / 2
noncomputable def spring_relation := k * x = P * S
noncomputable def volume_change := ΔV = S * x
noncomputable def volume_after_expansion := V = (n / (n - 1)) * (S * x)
noncomputable def ideal_gas_law := P * V = R * T
noncomputable def initial_state := P₀ * V₀ = R * T₀
noncomputable def expanded_state := P * (n * V₀) = R * T

-- Theorem to prove the final relation
theorem pressure_relation
  (h1: first_law Q Δu A)
  (h2: Δu_def Δu c_v T T₀)
  (h3: A_def A k x)
  (h4: spring_relation k x P S)
  (h5: volume_change ΔV S x)
  (h6: volume_after_expansion V S x n)
  (h7: ideal_gas_law P V R T)
  (h8: initial_state P₀ V₀ R T₀)
  (h9: expanded_state P R T n V₀)
  : P / P₀ = 1 / (n * (1 + ((n - 1) * R) / (2 * n * c_v))) :=
  sorry

end pressure_relation_l567_567308


namespace area_of_given_triangle_l567_567710

open BigOperators

structure Point where
  x : ℝ
  y : ℝ

def area_of_triangle (A B C : Point) : ℝ :=
  (1 / 2) * abs ((A.x * (B.y - C.y)) + (B.x * (C.y - A.y)) + (C.x * (A.y - B.y)))

noncomputable def A := ⟨2, 3⟩
noncomputable def B := ⟨8, 7⟩
noncomputable def C := ⟨2, 9⟩

theorem area_of_given_triangle :
  area_of_triangle A B C = 18 := by
  sorry

end area_of_given_triangle_l567_567710


namespace kim_candy_bars_saved_l567_567576

theorem kim_candy_bars_saved
  (n : ℕ)
  (c : ℕ)
  (w : ℕ)
  (total_bought : ℕ := n * c)
  (total_eaten : ℕ := n / w)
  (candy_bars_saved : ℕ := total_bought - total_eaten) :
  candy_bars_saved = 28 :=
by
  sorry

end kim_candy_bars_saved_l567_567576


namespace count_primes_between_2500_and_4500_l567_567960

theorem count_primes_between_2500_and_4500 :
  {p : ℤ | prime p ∧ 50 ≤ p ∧ p ≤ 67}.to_finset.card = 4 :=
by
  sorry

end count_primes_between_2500_and_4500_l567_567960


namespace point_in_first_quadrant_l567_567205

variable (P : ℝ × ℝ)
axiom point_coordinates : P = (5, 4)

def is_first_quadrant : Prop :=
  P.1 > 0 ∧ P.2 > 0

theorem point_in_first_quadrant : is_first_quadrant P :=
by
  unfold is_first_quadrant
  rw [point_coordinates]
  sorry

end point_in_first_quadrant_l567_567205


namespace function_count_l567_567093

open Set

namespace FunctionProblem

def A : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}

def f (x : ℕ) : ℕ := sorry -- placeholder for the function

theorem function_count :
  {f : ℕ → ℕ // (∀ x y ∈ A, x ≤ y → f x ≤ f y) ∧ f 3 = 4 ∧ (∀ x ∉ A, f x = 0)}
  .card = 17160 :=
sorry

end FunctionProblem

end function_count_l567_567093


namespace similar_triangles_perimeter_l567_567699

theorem similar_triangles_perimeter (A1 A2 : ℝ) (P1 : ℝ) (h1 : A1 / A2 = 1 / 9) (h2 : P1 = 15) :
  ∃ P2 : ℝ, P2 = 45 :=
by
  use 45
  sorry

end similar_triangles_perimeter_l567_567699


namespace range_of_b_l567_567942

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 / 2 then (2 * x + 1) / (x ^ 2) else x + 1

def g (x : ℝ) : ℝ := x ^ 2 - 4 * x - 4

-- The main theorem to prove the range of b
theorem range_of_b (a b : ℝ) (h : f a + g b = 0) : b ∈ Set.Icc (-1) 5 := by
  sorry

end range_of_b_l567_567942


namespace sum_of_reciprocals_of_root_products_eq_4_l567_567173

theorem sum_of_reciprocals_of_root_products_eq_4
  (p q r s t : ℂ)
  (h_poly : ∀ x : ℂ, x^5 + 10*x^4 + 20*x^3 + 15*x^2 + 8*x + 5 = 0 ∨ (x - p)*(x - q)*(x - r)*(x - s)*(x - t) = 0)
  (h_vieta_2 : p*q + p*r + p*s + p*t + q*r + q*s + q*t + r*s + r*t + s*t = 20)
  (h_vieta_all : p*q*r*s*t = 5) :
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(p*t) + 1/(q*r) + 1/(q*s) + 1/(q*t) + 1/(r*s) + 1/(r*t) + 1/(s*t) = 4 := 
sorry

end sum_of_reciprocals_of_root_products_eq_4_l567_567173


namespace part_a_part_b_part_c_l567_567910

variable {n : ℕ} (n_pos : n > 0)

def S : set ℕ := {0, 1, ..., 2 * n + 1}

def F (n : ℕ) : set (ℤ × S → ℝ) :=
{ f | (∀ x : ℤ, f (x, 0) = 0 ∧ f (x, 2 * n + 1) = 0) ∧ 
      (∀ x : ℤ, ∀ y ∈ S, 1 ≤ y ∧ y ≤ 2 * n → f (x-1, y) + f (x+1, y) + 
      f (x, y-1) + f (x, y+1) = 1) }

noncomputable def v (f : ℤ × S → ℝ) : set ℝ := {r | ∃ x : ℤ, ∃ y ∈ S, f (x, y) = r}

theorem part_a (n_pos : n > 0) : ∃ (F : set (ℤ × S → ℝ)), F = F n ∧ infinite F := 
by
  sorry

theorem part_b (n_pos : n > 0) (f : ℤ × S → ℝ) (hf : f ∈ F n) : finite (v f) := 
by
  sorry

theorem part_c (n_pos : n > 0) : ∃ f ∈ F n, sup_fin (v f) = 2 * n + 1 :=
by
  sorry

end part_a_part_b_part_c_l567_567910


namespace segment_ratios_MN_length_l567_567622

theorem segment_ratios_MN_length (A B M N : Point)
  (h : segment_length A B = 18)
  (ratio_condition : segment_length A M / segment_length M N / segment_length N B = 1 / 2 / 3) :
  segment_length M N = 6 :=
sorry

end segment_ratios_MN_length_l567_567622


namespace at_least_three_integer_areas_l567_567581

structure Point where
  x : Int
  y : Int

def area_triangle (p1 p2 p3 : Point) : Rat :=
  (1 : Rat) / 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)).natAbs

def no_three_collinear (p1 p2 p3 p4 p5 : Point) : Prop :=
  ∀ p a b c, ¬ (p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 ∨ p = p5) → 
  (area_triangle p a b = 0 → area_triangle p b c = 0 → area_triangle p c a = 0 → False)

theorem at_least_three_integer_areas (P1 P2 P3 P4 P5 : Point)
  (h1 : no_three_collinear P1 P2 P3 P4 P5) : 
  ∃ t1 t2 t3 : (Point × Point × Point), 
  let triangles := [(P1, P2, P3), (P1, P2, P4), (P1, P2, P5), (P1, P3, P4), (P1, P3, P5), (P1, P4, P5),
                     (P2, P3, P4), (P2, P3, P5), (P2, P4, P5), (P3, P4, P5)] in
  list_filter (λ t, area_triangle t.1.1 t.1.2 t.2 ∈ Int) triangles ⋙
  length ⋙ (≥ 3) :=
sorry

end at_least_three_integer_areas_l567_567581


namespace lottery_jackpot_probability_l567_567132

noncomputable def C (n k : ℕ) : ℕ := Fact.factorial n / (Fact.factorial k * Fact.factorial (n - k))

theorem lottery_jackpot_probability :
  (C 45 6 = 8145060) →
  (100: ℚ) / (C 45 6: ℚ) = 0.0000123 :=
by
  sorry

end lottery_jackpot_probability_l567_567132


namespace cards_difference_l567_567258

theorem cards_difference (A : Finset ℕ) (hA : A.card = 26) (h_sub : A ⊆ Finset.range 101) :
  ∃ (a b ∈ A), a ≠ b ∧ (|a - b| = 1 ∨ |a - b| = 2 ∨ |a - b| = 3) :=
sorry

end cards_difference_l567_567258


namespace Tom_worked_8_hours_l567_567542

-- Definition for the number of hours Tom worked on Monday
def worked_hours (h : ℕ) : Prop :=
  (0.20 * (10 * h) = 16)

-- Theorem stating that Tom worked 8 hours on Monday
theorem Tom_worked_8_hours : ∃ h : ℕ, worked_hours h ∧ h = 8 :=
by {
  -- We will provide the proof later
  sorry
}

end Tom_worked_8_hours_l567_567542


namespace sum_of_divisors_143_l567_567770

theorem sum_of_divisors_143 : ∑ d in List.toFinset (List.divisors 143), d = 168 := by
  have prime_factors_143 : 143 = 11 * 13 := by
    calc 143 = 11 * 13 : by norm_num
  sorry

end sum_of_divisors_143_l567_567770


namespace complex_number_in_second_quadrant_l567_567810

-- We must prove that the complex number z = i / (1 - i) lies in the second quadrant
theorem complex_number_in_second_quadrant : let z := (Complex.I / (1 - Complex.I)) in (z.re < 0) ∧ (z.im > 0) :=
by
  let z := (Complex.I / (1 - Complex.I))
  have h1 : (1 - Complex.I).conj = 1 + Complex.I := by sorry
  have h2 : (Complex.I * (1 + Complex.I)) = -1 + Complex.I := by sorry
  have h3 : (1 - Complex.I) * (1 + Complex.I) = 2 := by sorry
  have h4 : z = (-1 + Complex.I) / 2 := by sorry
  have h5 : z.re = -1 / 2 := by sorry
  have h6 : z.im = 1 / 2 := by sorry
  show (z.re < 0) ∧ (z.im > 0), from and.intro (by norm_num) (by norm_num)

end complex_number_in_second_quadrant_l567_567810


namespace number_of_boys_l567_567333

-- Definitions of the conditions
def total_members (B G : ℕ) : Prop := B + G = 26
def meeting_attendance (B G : ℕ) : Prop := (1 / 2 : ℚ) * G + B = 16

-- Theorem statement
theorem number_of_boys (B G : ℕ) (h1 : total_members B G) (h2 : meeting_attendance B G) : B = 6 := by
  sorry

end number_of_boys_l567_567333


namespace people_got_on_at_third_stop_l567_567268

theorem people_got_on_at_third_stop
  (initial : ℕ)
  (got_off_first : ℕ)
  (got_off_second : ℕ)
  (got_on_second : ℕ)
  (got_off_third : ℕ)
  (people_after_third : ℕ) :
  initial = 50 →
  got_off_first = 15 →
  got_off_second = 8 →
  got_on_second = 2 →
  got_off_third = 4 →
  people_after_third = 28 →
  ∃ got_on_third : ℕ, got_on_third = 3 :=
by
  sorry

end people_got_on_at_third_stop_l567_567268


namespace max_distance_origin_perpendicular_bisector_l567_567473

theorem max_distance_origin_perpendicular_bisector :
  ∀ (k m : ℝ), k ≠ 0 → 
  (|m| = Real.sqrt (1 + k^2)) → 
  ∃ (d : ℝ), d = 4 / 3 :=
by
  sorry

end max_distance_origin_perpendicular_bisector_l567_567473


namespace problem_statement_l567_567223

def points_on_circle (n : ℕ) : Type := { S : set ℝ // S.card = n }

def evenly_distributed_on_circle (n : ℕ) [fact (n > 0)] : points_on_circle n := {
  S := set.range (λ k : fin n, (k : ℝ) / n),
  card := sorry
}

def probability_intersection_of_chords 
(p q r s : ℝ) (h : set.pairwise_disjoint {p, q, r, s}) : ℝ := 
match p, q, r, s with
  | ... := -- Cases detailing valid configurations
sorry -- Implement configurations and cases accordingly

theorem problem_statement : 
  ∃ (S : points_on_circle 1988), 
  let p := evenly_distributed_on_circle 1988 in
  ∀ (P Q R S : ℝ) 
    (h1 : P ∈ S.val) 
    (h2 : Q ∈ S.val) 
    (h3 : R ∈ S.val) 
    (h4 : S ∈ S.val) 
    (h_distinct : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S), 
  probability_intersection_of_chords P Q R S h_distinct = 1/3 :=
begin
  sorry -- Proof is omitted as per request
end


end problem_statement_l567_567223


namespace purely_imaginary_implies_m_eq_neg_half_simplify_z_squared_over_z_add_5_plus_2i_l567_567937

def z (m : ℝ) : Complex := Complex.mk (2 * m^2 - 3 * m - 2) (m^2 - 3 * m + 2)

theorem purely_imaginary_implies_m_eq_neg_half (m : ℝ) : 
  (z m).re = 0 ↔ m = -1 / 2 := sorry

theorem simplify_z_squared_over_z_add_5_plus_2i (z_zero : ℂ) :
  z 0 = ⟨-2, 2⟩ →
  (z 0)^2 / (z 0 + Complex.mk 5 2) = ⟨-32 / 25, -24 / 25⟩ := sorry

end purely_imaginary_implies_m_eq_neg_half_simplify_z_squared_over_z_add_5_plus_2i_l567_567937


namespace find_b_l567_567533

noncomputable def n : ℝ := 2 ^ 0.3
noncomputable def b : ℝ := 40 / 3

theorem find_b (h₁ : n = 2 ^ 0.3) (h₂ : n ^ b = 16) : b = 40 / 3 :=
by
  sorry

end find_b_l567_567533


namespace angle_CED_120_l567_567281

noncomputable def circle_centered_at (center : ℝ × ℝ) (r : ℝ) : set (ℝ × ℝ) :=
{ p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = r^2 }

variables (A B C D E : ℝ × ℝ) (rA rB : ℝ)

axiom ratio_radii : rA = rB / 2
axiom center_A_pass_B : B ∈ circle_centered_at A rA
axiom center_B_pass_A : A ∈ circle_centered_at B rB
axiom AB_contains_C : C ∈ (circle_centered_at A rA ∩ circle_centered_at B rB)
axiom AB_contains_D : D ∈ (circle_centered_at A rA ∩ circle_centered_at B rB)
axiom circles_intersect_E : E ∈ (circle_centered_at A rA ∩ circle_centered_at B rB)

theorem angle_CED_120 :
  angle C E D = 120 := sorry

end angle_CED_120_l567_567281


namespace valid_outfit_combinations_l567_567523

theorem valid_outfit_combinations (shirts pants hats shoes : ℕ) (colors : ℕ) 
  (h₁ : shirts = 6) (h₂ : pants = 6) (h₃ : hats = 6) (h₄ : shoes = 6) (h₅ : colors = 6) :
  ∀ (valid_combinations : ℕ),
  (valid_combinations = colors * (colors - 1) * (colors - 2) * (colors - 3)) → valid_combinations = 360 := 
by
  intros valid_combinations h_valid_combinations
  sorry

end valid_outfit_combinations_l567_567523


namespace sum_of_consecutive_integers_l567_567682

theorem sum_of_consecutive_integers (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + 1 = b) (h4 : b + 1 = c) (h5 : a * b * c = 336) : a + b + c = 21 :=
sorry

end sum_of_consecutive_integers_l567_567682


namespace jane_doe_gift_l567_567156

theorem jane_doe_gift (G : ℝ) (h1 : 0.25 * G + 0.1125 * (0.75 * G) = 15000) : G = 41379 := 
sorry

end jane_doe_gift_l567_567156


namespace find_x_l567_567512

variable (x : ℝ)

def vec_a : ℝ × ℝ := (1, x)
def vec_b : ℝ × ℝ := (x, 4)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_x (h : dot_product (vec_a x) (vec_b x) = magnitude (vec_a x) * magnitude (vec_b x)) : x = 2 :=
by
  sorry

end find_x_l567_567512


namespace complex_conjugate_z_l567_567936

/-- Given the complex number z that satisfies (1+i)z = |sqrt(3)-i|, prove that the conjugate of z is 1+i. -/
theorem complex_conjugate_z (z : ℂ) (h : (1 + complex.i) * z = complex.abs (complex.mk (real.sqrt 3) (-1))) :
  complex.conj z = 1 + complex.i :=
by sorry

end complex_conjugate_z_l567_567936


namespace other_train_speed_is_correct_l567_567698

noncomputable def other_train_speed
  (length_train1 length_train2 : ℕ) 
  (speed_faster_train : ℕ) 
  (cross_time_seconds : ℕ) : ℕ := 
  let total_distance_km := (length_train1 + length_train2) / 1000.0
  let cross_time_hours := cross_time_seconds / 3600.0
  let relative_speed := total_distance_km / cross_time_hours
  speed_faster_train - relative_speed

theorem other_train_speed_is_correct 
  (length_train1 length_train2 : ℕ := 200) 
  (speed_faster_train : ℕ := 45) 
  (cross_time_seconds : ℕ := 273.6) :
  other_train_speed length_train1 length_train2 speed_faster_train cross_time_seconds = 40 := by
  sorry

end other_train_speed_is_correct_l567_567698


namespace sqrt_domain_l567_567272

theorem sqrt_domain (x : ℝ) : 3 * x - 6 ≥ 0 ↔ x ≥ 2 := sorry

end sqrt_domain_l567_567272


namespace coefficient_x3_in_expansion_l567_567233

noncomputable def coefficient_of_x3 : ℤ :=
  let T := λ r: ℕ, binomial 5 r * (2*x^2 + x)^(5-r) * (-1)^r in
  T 2 * binomial 5 2 + T 3 * binomial 5 3

theorem coefficient_x3_in_expansion :
  coefficient_of_x3 = -30 := sorry

end coefficient_x3_in_expansion_l567_567233


namespace water_collected_first_day_l567_567834

theorem water_collected_first_day : 
  ∀ (x : ℝ), (tank_volume = 100) → (initial_water = 2/5 * tank_volume) → 
  (day1 = x) → (day2 = x + 5) → (day3 = 25) → 
  (day1 + day2 + initial_water = tank_volume) → 
  (x = 27.5) :=
begin
  intros,
  sorry
end

end water_collected_first_day_l567_567834


namespace positive_solution_approx_l567_567897

noncomputable def find_positive_solution : ℝ :=
  let y := cbrt 3 in
  let x := y^3 in
  x

theorem positive_solution_approx :
  ∃ x : ℝ, (x ≈ 3.1481) ∧
    (sqrt4 (x + sqrt4 (x + sqrt4 (x + ...))) = sqrt4 (x * sqrt4 (x * sqrt4 (x * ...)))) :=
by
  sorry

end positive_solution_approx_l567_567897


namespace prime_divisibility_condition_l567_567580

-- Define the set of primes less than 10000
def P_star : Set ℕ := {p | Nat.Prime p ∧ p < 10000}

-- Define the set of Mersenne primes less than 10000
def Mersenne_primes_less_than_10000 : Set ℕ := {3, 7, 31, 127, 8191}

-- Define the divisibility condition
def divides (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

-- Main statement
theorem prime_divisibility_condition :
  {p ∈ P_star | ∀ S : Finset ℕ, (∀ x ∈ S, x ∈ P_star) ∧ S.card ≥ 2 → 
  (∀ p ∉ S, ∃ q ∈ P_star \ S, divides (q + 1) (S.val.map (λ x, x + 1)).prod)} 
  = Mersenne_primes_less_than_10000 := sorry

end prime_divisibility_condition_l567_567580


namespace find_coeffs_l567_567474

theorem find_coeffs (a b c : ℝ) :
  (1 : ℝ) = a * (1 : ℝ)^2 + b * (1 : ℝ) + c ∧
  (-1 : ℝ) = a * (2 : ℝ)^2 + b * (2 : ℝ) + c ∧
  (1 : ℝ) = 2 * a * (2 : ℝ) + b →
  (a = 3 ∧ b = -11 ∧ c = 9) :=
by
  intros h1 h2 h3
  sorry

end find_coeffs_l567_567474


namespace unique_prime_roots_k_value_l567_567366

theorem unique_prime_roots_k_value :
  ∃! k : ℤ, (∃ p q : ℤ, p.prime ∧ q.prime ∧ p + q = 63 ∧ p * q = k) :=
by
  sorry

end unique_prime_roots_k_value_l567_567366


namespace average_first_last_numbers_l567_567243

theorem average_first_last_numbers
  (largest_not_second : ∀ (l : ℕ), l != 15 ∨ (l != 2 )→ l < 4)
  (smallest_not_third : ∀ (s : ℕ), s != -5 ∨ (s != 3) → s > 1)
  (median_not_last_first : ∀ (m : ℕ), m != 7 ∨ (m != 1 )→ m < 4) : 
  (15 + (-5)) / 2 = 5 :=
by
  sorry

end average_first_last_numbers_l567_567243


namespace length_BD_is_sqrt_13_l567_567999

def complex_point (z : ℂ) : ℝ × ℝ := (z.re, z.im)

def parallelogram_length_BD (A B C D : ℂ) : ℝ :=
  let BA := (complex_point A - complex_point B)
  let BC := (complex_point C - complex_point B)
  let BD := (BA + BC)
  real.sqrt (BD.1 ^ 2 + BD.2 ^ 2)

theorem length_BD_is_sqrt_13 :
  let A := complex_point (complex.I)
  let B := complex_point (1 : ℂ)
  let C := complex_point (4 + 2 * complex.I)
  let D := complex_point (B + (C - A) - B)
  parallelogram_length_BD (complex.I) 1 (4 + 2 * complex.I) D = real.sqrt 13 :=
begin
  sorry
end

end length_BD_is_sqrt_13_l567_567999


namespace area_of_trapezoid_l567_567990

-- Define the problem using the provided conditions and question
theorem area_of_trapezoid
  (ABCD : Type)
  (E : Type)
  (A B C D : ABCD)
  (AB_parallel_CD : AB.parallel CD)
  (AC_BD_intersect_E : AC.intersect BD = E)
  (area_ΔABE : Area (triangle A B E) = 60)
  (area_ΔADE : Area (triangle A D E) = 25) :
  Area (trapezoid A B C D) = 135 :=
by sorry

end area_of_trapezoid_l567_567990


namespace sum_of_divisors_of_143_l567_567763

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (nat.divisors n).toFinset, k

theorem sum_of_divisors_of_143 : sum_divisors 143 = 168 :=
by sorry

end sum_of_divisors_of_143_l567_567763


namespace exists_point_P_equal_distance_squares_l567_567689

-- Define the points in the plane representing the vertices of the triangles
variables {A1 A2 A3 B1 B2 B3 C1 C2 C3 : ℝ × ℝ}
-- Define the function that calculates the square distance between two points
def sq_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

-- Define the proof statement
theorem exists_point_P_equal_distance_squares :
  ∃ P : ℝ × ℝ,
    sq_distance P A1 + sq_distance P A2 + sq_distance P A3 =
    sq_distance P B1 + sq_distance P B2 + sq_distance P B3 ∧
    sq_distance P A1 + sq_distance P A2 + sq_distance P A3 =
    sq_distance P C1 + sq_distance P C2 + sq_distance P C3 := sorry

end exists_point_P_equal_distance_squares_l567_567689


namespace sum_of_divisors_143_l567_567733

theorem sum_of_divisors_143 : sigma 143 = 168 := by
  have h : 143 = 11 * 13 := by norm_num
  rw [sigma_mul_prime_prime 11 13]
  norm_num
  sorry

end sum_of_divisors_143_l567_567733


namespace option_d_correct_l567_567296

theorem option_d_correct (x : ℝ) (h : x > 0) : 
  (∀ x, (1 : ℝ) / x^2 = (1 : ℝ) / (-x)^2) ∧ (∀ x1 x2, 0 < x1 ∧ x1 < x2 → (1 : ℝ) / x1^2 > (1 : ℝ) / x2^2) :=
by
  -- Proof of even function
  have even_function : ∀ x, (1 : ℝ) / x^2 = (1 : ℝ) / (-x)^2, sorry,
  -- Proof of monotonically decreasing
  have monot_dec : ∀ x1 x2, 0 < x1 ∧ x1 < x2 → (1 : ℝ) / x1^2 > (1 : ℝ) / x2^2, sorry,
  exact ⟨even_function, monot_dec⟩

end option_d_correct_l567_567296


namespace matrix_solution_l567_567870

variable {ℝ : Type*} [NoncomputableField ℝ]

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![4, 4], ![2, 4]]

theorem matrix_solution (N : Matrix (Fin 2) (Fin 2) ℝ) 
    (h : N^4 - 3*N^3 + 3*N^2 - N = ![![16, 24], ![8, 12]]): 
    N = A :=
  by
  sorry

end matrix_solution_l567_567870


namespace cos_difference_identity_cos_phi_value_l567_567316

variables (α β θ φ : ℝ)
variables (a b : ℝ × ℝ)

-- Part I
theorem cos_difference_identity (hα : 0 ≤ α ∧ α ≤ 2 * Real.pi) (hβ : 0 ≤ β ∧ β ≤ 2 * Real.pi) : 
  Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β :=
sorry

-- Part II
theorem cos_phi_value (hθ : 0 < θ ∧ θ < Real.pi / 2) (hφ : 0 < φ ∧ φ < Real.pi / 2)
  (ha : a = (Real.sin θ, -2)) (hb : b = (1, Real.cos θ)) (dot_ab_zero : a.1 * b.1 + a.2 * b.2 = 0)
  (h_sin_diff : Real.sin (theta - phi) = Real.sqrt 10 / 10) :
  Real.cos φ = Real.sqrt 2 / 2 :=
sorry

end cos_difference_identity_cos_phi_value_l567_567316


namespace centroid_of_triangle_inaccessible_l567_567059

variable (A B C : Point) (ABC_triangle : Triangle A B C)

theorem centroid_of_triangle_inaccessible 
    (inaccessible_C : InaccessibleVertex C) :
    ∃ G : Point, is_centroid G A B C ∧ is_median_intersection G A B C :=
sorry

end centroid_of_triangle_inaccessible_l567_567059


namespace average_speed_l567_567240

theorem average_speed (D T : ℝ) (hD : D = 200) (hT : T = 6) : D / T = 33.33 := by
  -- Sorry is used to skip the proof, only the statement is provided as per instruction
  sorry

end average_speed_l567_567240


namespace probability_of_region_C_l567_567803

theorem probability_of_region_C :
  let x := (7 : ℚ) / 60 in
  ∀ (pA pB pC pD pE : ℚ),
    pA = 1 / 5 →
    pB = 1 / 3 →
    pC = x →
    pD = x →
    pE = 2 * x →
    pA + pB + pC + pD + pE = 1 :=
begin
  intros x pA pB pC pD pE hA hB hC hD hE,
  sorry
end

end probability_of_region_C_l567_567803


namespace find_angle_ACB_l567_567129

open EuclideanGeometry

variable (A B C D : Point)
variable (h_parallel : ParallelLine DC AB)
variable (h_angle_DCA : Angle DCA = 55º)
variable (h_angle_ABC : Angle ABC = 60º)

theorem find_angle_ACB :
  Angle ACB = 65º :=
by
  sorry

end find_angle_ACB_l567_567129


namespace number_of_pairs_d_n_l567_567144

-- Let's define Jane's current age and Dick's current age calculation
def Jane_current_age : ℕ := 21
def Dick_current_age : ℕ := 2 * Jane_current_age - 3

-- Define conditions as per the problem
axiom n (n : ℕ) (hn : n > 0) -- positive integer
axiom a_digit (a : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) -- a is a digit
axiom b_digit (b : ℕ) (hb : 1 ≤ b ∧ b ≤ 9 ∧ b > a ∧ b - a = 2) -- b is a digit and b > a with b - a = 2

-- Define the future ages based on the conditions
def Jane_future_age (n a b : ℕ) : ℕ := 10 * a + b
def Dick_future_age (n a b : ℕ) : ℕ := 10 * b + a

-- Lean statement to prove the number of pairs (d, n)
theorem number_of_pairs_d_n : ∃ (d : ℕ) (count : ℕ), 
  d = Dick_current_age ∧
  count = 7 ∧
  ∀ (n a b : ℕ), n > 0 → 1 ≤ a ∧ a ≤ 9 → 1 ≤ b ∧ b ≤ 9 → b > a → b - a = 2 →
  Jane_future_age n a b = 21 + n ∧ Dick_future_age n a b = 39 + n :=
sorry

end number_of_pairs_d_n_l567_567144


namespace notebook_cost_l567_567570

-- Define the conditions
def cost_pen := 1
def num_pens := 3
def num_notebooks := 4
def cost_folder := 5
def num_folders := 2
def initial_bill := 50
def change_back := 25

-- Calculate derived values
def total_spent := initial_bill - change_back
def total_cost_pens := num_pens * cost_pen
def total_cost_folders := num_folders * cost_folder
def total_cost_notebooks := total_spent - total_cost_pens - total_cost_folders

-- Calculate the cost per notebook
def cost_per_notebook := total_cost_notebooks / num_notebooks

-- Proof statement
theorem notebook_cost : cost_per_notebook = 3 := by
  sorry

end notebook_cost_l567_567570


namespace find_k_l567_567687

-- Define the operations A and B on a stack.
def operation_A (s : List Nat) : List Nat := 
  s.tail ++ [s.head!]

def operation_B (s : List Nat) : List Nat :=
  s.tail

-- Define the sequence of operations (ABBAABBA... until one card remains)
def seq_operations (s : List Nat) (n : Nat) : List Nat :=
  if s.length = 1 then s
  else if n % 3 == 1 then seq_operations (operation_A s) (n + 1)
  else seq_operations (operation_B s) (n + 1)

-- Define L(n) as the remaining card after the sequence of operations
def L (n : Nat) : Nat :=
  (seq_operations (List.range n).tail++[n] 1).head!

-- Define the main theorem
theorem find_k (k : Nat) : 
  L (3 * k) = k ↔ 
  ∃ j, k = (3 ^ (6 * j + 2) - 2) / 7 ∨ k = (2 * 3 ^ (6 * j) - 2) / 7 :=
by
  sorry

end find_k_l567_567687


namespace sum_of_divisors_143_l567_567740

theorem sum_of_divisors_143 : 
  (finset.range (144)).filter (λ d, 143 % d = 0).sum = 168 :=
by
  sorry

end sum_of_divisors_143_l567_567740


namespace tildeA_independent_prob_union_prob_none_occur_l567_567595

variable (n : ℕ)
variable (A : Fin n → Event)
variable (p : Fin n → ℝ)

-- Condition: P(A_i) = p_i
axiom independent_events : ∀ i : Fin n, P (A i) = p i

-- The events \widetilde{A}_1, \ldots, \widetilde{A}_n are independent.
theorem tildeA_independent (A : Fin n → Event)
  (h_independent: IndependentEvents A) :
  (IndependentEvents (λ i, A i ∨ ¬ A i))
:= sorry

-- Prove that P (⋃ i, A i) = 1 - ∏ i, P (¬ A_i)
theorem prob_union (A : Fin n → Event)
  (h_independent : IndependentEvents A) 
  (h_prob : ∀ i, P (A i) = p i) :
  P (⋃ i, A i) = 1 - ∏ i, P (¬ A i) :=
sorry

-- Prove that the probability P_0 that none of the events A_1, ..., A_n occur is ∏ i, P (¬ A_i)
theorem prob_none_occur (A : Fin n → Event)
  (h_independent : IndependentEvents A)
  (h_prob : ∀ i, P (A i) = p i) :
  P (⋃ i, ¬ A i) = ∏ i, (1 - p i) :=
sorry

end tildeA_independent_prob_union_prob_none_occur_l567_567595


namespace nonnegative_sequence_inequality_l567_567053

theorem nonnegative_sequence_inequality 
  (n : ℕ) 
  (h_pos : 0 < n)
  (a : Fin n → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ a i ∧ a i ≤ 1) :
  (1 - ∏ i, a i) / n ≤ 1 / (1 + ∑ i, a i) := 
by {
  sorry
}

end nonnegative_sequence_inequality_l567_567053


namespace range_of_m_l567_567171

variable {ℝ : Type*}
variable {f : ℝ → ℝ}
variable {m : ℝ}

axiom decreasing (h : ∀ x y : ℝ, x < y → f(x) > f(y))

theorem range_of_m (h1 : decreasing f) (h2 : f (m - 1) > f (2 * m - 1)) : m > 0 := by 
  sorry

end range_of_m_l567_567171


namespace number_of_subsets_divisible_by_p_l567_567172

theorem number_of_subsets_divisible_by_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ (n : ℕ), n = 2 ∧ (∃ (s : Finset (Fin (2 * p))) (hs : s.card = p), (∑ x in s, x.val) % p = 0) :=
by
  sorry

end number_of_subsets_divisible_by_p_l567_567172


namespace largest_integer_less_100_leaves_remainder_4_l567_567454

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l567_567454


namespace integer_solutions_l567_567387

theorem integer_solutions (n : ℤ) : ∃ m : ℤ, n^2 + 15 = m^2 ↔ n = -7 ∨ n = -1 ∨ n = 1 ∨ n = 7 :=
by
  sorry

end integer_solutions_l567_567387


namespace find_a_l567_567405

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end find_a_l567_567405


namespace tan_sum_angles_l567_567125

noncomputable def slope_of_line := sqrt 2
def equation_of_circle := ∀ (x y : ℝ), x^2 + y^2 = 1
def line_intersects_circle := ∀ (x y : ℝ), y = slope_of_line * x → equation_of_circle x y
def initial_sides_non_negative_half_x_axis := true -- Implicit in stating angles start from x-axis
def terminal_sides_on_rays (α β : ℝ) (A B : ℝ → ℝ → Prop) := 
  A (cos α) (sin α) ∧ B (cos β) (sin β)

theorem tan_sum_angles (α β : ℝ) (A B : ℝ → ℝ → Prop) (h1 : line_intersects_circle (cos α) (sin α))
  (h2 : line_intersects_circle (cos β) (sin β)) (h3 : β = α + π) 
  : tan (α + β) = -2 * sqrt 2 :=
by 
  sorry

end tan_sum_angles_l567_567125


namespace find_a_l567_567401

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end find_a_l567_567401


namespace largest_integer_less_100_leaves_remainder_4_l567_567458

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l567_567458


namespace largest_k_for_divisibility_by_2k_l567_567164

theorem largest_k_for_divisibility_by_2k :
  let Q := (List.range' 1 101).map (fun n => 2 * n) in
  let Q_prod := Q.prod in
  ∃ k : ℕ, (Q_prod % 2^k = 0) ∧ (∀ j : ℕ, (Q_prod % 2^j = 0) → j ≤ 197) :=
by
  sorry

end largest_k_for_divisibility_by_2k_l567_567164


namespace largest_integer_is_222_l567_567647

theorem largest_integer_is_222
  (a b c d : ℤ)
  (h_distinct : a < b ∧ b < c ∧ c < d)
  (h_mean : (a + b + c + d) / 4 = 72)
  (h_min_a : a ≥ 21) 
  : d = 222 :=
sorry

end largest_integer_is_222_l567_567647


namespace element_with_36_36_percentage_is_O_l567_567460

-- Define the chemical formula N2O and atomic masses
def chemical_formula : String := "N2O"
def atomic_mass_N : Float := 14.01
def atomic_mass_O : Float := 16.00

-- Define the molar mass of N2O
def molar_mass_N2O : Float := (2 * atomic_mass_N) + (1 * atomic_mass_O)

-- Mass of nitrogen in N2O
def mass_N_in_N2O : Float := 2 * atomic_mass_N

-- Mass of oxygen in N2O
def mass_O_in_N2O : Float := 1 * atomic_mass_O

-- Mass percentages
def mass_percentage_N : Float := (mass_N_in_N2O / molar_mass_N2O) * 100
def mass_percentage_O : Float := (mass_O_in_N2O / molar_mass_N2O) * 100

-- Prove that the element with a mass percentage of 36.36% is oxygen
theorem element_with_36_36_percentage_is_O : mass_percentage_O = 36.36 := sorry

end element_with_36_36_percentage_is_O_l567_567460


namespace snowball_volume_surface_area_l567_567379

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4/3) * real.pi * r^3

noncomputable def surface_area_sphere (r : ℝ) : ℝ :=
  4 * real.pi * r^2

theorem snowball_volume_surface_area :
  let V_total := volume_sphere 4 + volume_sphere 6 + volume_sphere 7;
  let S_largest := surface_area_sphere 7;
  V_total = (2492 / 3) * real.pi ∧ S_largest = 196 * real.pi :=
by
  sorry

end snowball_volume_surface_area_l567_567379


namespace divide_8_friends_among_4_teams_l567_567521

def num_ways_to_divide_friends (n : ℕ) (teams : ℕ) :=
  teams ^ n

theorem divide_8_friends_among_4_teams :
  num_ways_to_divide_friends 8 4 = 65536 :=
by sorry

end divide_8_friends_among_4_teams_l567_567521


namespace greatest_multiple_of_8_remainder_l567_567585

/-- 
Let M be the greatest integer multiple of 8, no two of whose digits are the same, using digits only from 1 to 9.
Prove that the remainder when M is divided by 1000 is 976.
-/
theorem greatest_multiple_of_8_remainder :
  ∃ M : ℕ, (∃ (d : ℕ), (M = d) ∧ (M % 8 = 0) ∧ (∀ i j, i ≠ j → (d.to_digits i) ≠ (d.to_digits j)) ∧ (∀ k, k ∈ d.to_digits → 1 ≤ k ∧ k ≤ 9)) ∧ (M % 1000 = 976) :=
sorry

end greatest_multiple_of_8_remainder_l567_567585


namespace dough_completion_time_l567_567340

def start_time := 7 + 0 / 60 -- 7:00 AM
def one_third_done_time := 10 + 20 / 60 -- 10:20 AM
def lunch_start_time := 12 -- 12:00 PM
def lunch_duration := 1 -- 1 hour
def total_time_to_complete := 10 -- 10 hours
def remaining_time_to_complete := 20 / 3 -- time after resuming work at 1:00 PM
def expected_completion_time := 19 + 40 / 60 -- 7:40 PM (in 24-hour format)

theorem dough_completion_time :
  let time_taken_one_third := one_third_done_time - start_time in
  let total_work_time := time_taken_one_third * 3 in
  let working_time_after_lunch := total_work_time - time_taken_one_third in
  let time_after_resuming_work := lunch_start_time + lunch_duration in
  let completion_time := time_after_resuming_work + (remaining_time_to_complete) in
  total_time_to_complete = 10 ∧
  (completion_time = expected_completion_time) :=
by
  sorry

end dough_completion_time_l567_567340


namespace find_smallest_in_arithmetic_progression_l567_567037

theorem find_smallest_in_arithmetic_progression (a d : ℝ)
  (h1 : (a-2*d)^3 + (a-d)^3 + a^3 + (a+d)^3 + (a+2*d)^3 = 0)
  (h2 : (a-2*d)^4 + (a-d)^4 + a^4 + (a+d)^4 + (a+2*d)^4 = 136) :
  (a - 2*d) = -2 * Real.sqrt 2 :=
sorry

end find_smallest_in_arithmetic_progression_l567_567037


namespace pressure_ratio_l567_567305

-- Define Q, Δu, and A
def Q (Δu A : ℝ) : ℝ := Δu + A

-- Define the relationship for Q = 0
def Q_zero (Δu A : ℝ) : Prop := Q Δu A = 0

-- Define Δu in terms of cv, T, and T0
def Δu (cv T T0 : ℝ) : ℝ := cv * (T - T0)

-- Define A in terms of k, x
def A (k x : ℝ) : ℝ := (k * x^2) / 2

-- Define relationship between k, x, P, and S
def pressure_relation (k x P S : ℝ) : Prop := k * x = P * S

-- Define change in volume
def ΔV (S x : ℝ) : ℝ := S * x

-- Define the expanded volume
def V (n S x : ℝ) : ℝ := (n / (n - 1)) * S * x

-- Ideal gas law relationships
def ideal_gas_law (P V R T : ℝ) : Prop := P * V = R * T

-- Initial and expanded states
def initial_state (P0 V0 R T0 : ℝ) : Prop := P0 * V0 = R * T0
def expanded_state (P n V0 R T : ℝ) : Prop := P * n * V0 = R * T

-- Define target proof statement
theorem pressure_ratio (cv k x P S n R T T0 P0 V0 : ℝ)
  (hQ_zero : Q_zero (Δu cv T T0) (A k x))
  (hPressRel : pressure_relation k x P S)
  (hIdealGasLaw : ideal_gas_law P (V n S x) R T)
  (hInitialState : initial_state P0 V0 R T0)
  (hExpandedState : expanded_state P n V0 R T) :
  P / P0 = 1 / (n * (1 + ((n - 1) * R) / (2 * n * cv))) :=
by sorry

end pressure_ratio_l567_567305


namespace exists_plane_perpendicular_l567_567114

-- Definitions of line, plane and perpendicularity intersection etc.
variables (Point : Type) (Line Plane : Type)
variables (l : Line) (α : Plane) (intersects : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop) (perpendicular_planes : Plane → Plane → Prop)
variables (β : Plane) (subset : Line → Plane → Prop)

-- Conditions
axiom line_intersects_plane (h1 : intersects l α) : Prop
axiom line_not_perpendicular_plane (h2 : ¬perpendicular l α) : Prop

-- The main statement to prove
theorem exists_plane_perpendicular (h1 : intersects l α) (h2 : ¬perpendicular l α) :
  ∃ (β : Plane), (subset l β) ∧ (perpendicular_planes β α) :=
sorry

end exists_plane_perpendicular_l567_567114


namespace probability_two_digit_between_21_and_30_l567_567196

theorem probability_two_digit_between_21_and_30 (dice1 dice2 : ℤ) (h1 : 1 ≤ dice1 ∧ dice1 ≤ 6) (h2 : 1 ≤ dice2 ∧ dice2 ≤ 6) :
∃ (p : ℚ), p = 11 / 36 := 
sorry

end probability_two_digit_between_21_and_30_l567_567196


namespace closest_integer_to_99_times_9_l567_567297

theorem closest_integer_to_99_times_9 :
  let target := 99 * 9
  let dist (n : ℤ) := |target - n|
  let choices := [10000, 100, 100000, 1000, 10]
  1000 ∈ choices ∧ ∀ (n : ℤ), n ∈ choices → dist 1000 ≤ dist n :=
by
  let target := 99 * 9
  let dist (n : ℤ) := |target - n|
  let choices := [10000, 100, 100000, 1000, 10]
  sorry

end closest_integer_to_99_times_9_l567_567297


namespace sum_of_divisors_143_l567_567777

theorem sum_of_divisors_143 : ∑ d in List.toFinset (List.divisors 143), d = 168 := by
  have prime_factors_143 : 143 = 11 * 13 := by
    calc 143 = 11 * 13 : by norm_num
  sorry

end sum_of_divisors_143_l567_567777


namespace milk_price_increase_l567_567556

theorem milk_price_increase
  (P : ℝ) (C : ℝ) (P_new : ℝ)
  (h1 : P * C = P_new * (5 / 6) * C) :
  (P_new - P) / P * 100 = 20 :=
by
  sorry

end milk_price_increase_l567_567556


namespace sum_of_divisors_143_l567_567773

theorem sum_of_divisors_143 : ∑ d in List.toFinset (List.divisors 143), d = 168 := by
  have prime_factors_143 : 143 = 11 * 13 := by
    calc 143 = 11 * 13 : by norm_num
  sorry

end sum_of_divisors_143_l567_567773


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567420

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567420


namespace power_multiplication_l567_567372

variable (p : ℝ)  -- Assuming p is a real number

theorem power_multiplication :
  (-p)^2 * (-p)^3 = -p^5 :=
sorry

end power_multiplication_l567_567372


namespace length_of_football_field_l567_567615

theorem length_of_football_field :
  ∃ x : ℝ, (4 * x + 500 = 1172) ∧ x = 168 :=
by
  use 168
  simp
  sorry

end length_of_football_field_l567_567615


namespace tan_of_given_sinx_l567_567468

theorem tan_of_given_sinx 
  (x : ℝ) 
  (h1 : sin x = 3 / 5) 
  (h2 : π / 2 < x ∧ x < π) : 
  tan x = -3 / 4 := 
by 
  sorry

end tan_of_given_sinx_l567_567468


namespace find_coefficients_of_j_l567_567245

theorem find_coefficients_of_j (x : ℝ):
  let h := Polynomial.Cubic.realPolynomial x - 2 * (Polynomial.degree 2).realPolynomial x + 3 * (Polynomial.degree 1).realPolynomial x - 4 in
  let j := Polynomial.Cubic.realPolynomial x + b * (Polynomial.degree 2).realPolynomial x + c * (Polynomial.degree 1).realPolynomial x + d in
  ∀ x ∈ (Polynomial.roots h), ∃ y ∈ (Polynomial.roots j), y = x^3 → (b = -8) ∧ (c = 36) ∧ (d = -64) :=
sorry

end find_coefficients_of_j_l567_567245


namespace subtraction_property_l567_567800

theorem subtraction_property : (12.56 - (5.56 - 2.63)) = (12.56 - 5.56 + 2.63) := 
by 
  sorry

end subtraction_property_l567_567800


namespace probability_of_all_dice_showing_three_l567_567290

theorem probability_of_all_dice_showing_three : 
  let p := 1 / 6 in
  (p * p * p * p = 1 / 1296) :=
by
  sorry

end probability_of_all_dice_showing_three_l567_567290


namespace solve_inequality_l567_567634

variable (a : ℝ)

-- Define a function that represents the set of x satisfying the inequality
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then
    {-∞, -1}
  else if a > 0 then
    {-∞, -1} ∪ {x : ℝ | 1 / a < x}
  else if a = -1 then
    ∅
  else if a < -1 then
    {x : ℝ | -1 < x ∧ x < 1 / a}
  else
    {x : ℝ | 1 / a < x ∧ x < -1}

-- Define the theorem to state the equivalence
theorem solve_inequality (x : ℝ) :
  (a ≠ -1) → (a ≠ 0) → (a * x - 1) / (x + 1) > 0 ↔ x ∈ solution_set a :=
sorry

end solve_inequality_l567_567634


namespace num_pairs_of_pants_l567_567822

theorem num_pairs_of_pants (S P : ℕ) (total_cost refund : ℕ) (n_shirts n_pants : ℕ)
    (h1 : S = 45)
    (h2 : total_cost = 120)
    (h3 : refund = 0.25 * (n_pants * P))
    (h4 : 2 * S + n_pants * P = total_cost)
    (h5 : refund = 7.50):
    n_pants = 1 := by
  sorry

end num_pairs_of_pants_l567_567822


namespace class_average_l567_567972

theorem class_average (p1 p2 p3 avg1 avg2 avg3 : ℝ)
  (h1 : p1 = 0.20) (h2 : p2 = 0.50) (h3 : p3 = 0.30)
  (h_avg1 : avg1 = 80) (h_avg2 : avg2 = 60) (h_avg3 : avg3 = 40) :
  let A := p1 * avg1 + p2 * avg2 + p3 * avg3 in
  A = 58 := by
  intros
  sorry

end class_average_l567_567972


namespace prob_no_markers_given_no_X_l567_567989
open Classical BigOperators

noncomputable def prob_only_X := 0.15
noncomputable def prob_only_Y := 0.15
noncomputable def prob_only_Z := 0.15
noncomputable def prob_XY := 0.18
noncomputable def prob_XZ := 0.18
noncomputable def prob_YZ := 0.18
noncomputable def prob_XYZ_given_XY := 1 / 4

theorem prob_no_markers_given_no_X 
  (prob_only_X : ℝ := 0.15)
  (prob_only_Y : ℝ := 0.15)
  (prob_only_Z : ℝ := 0.15)
  (prob_XY : ℝ := 0.18)
  (prob_XZ : ℝ := 0.18)
  (prob_YZ : ℝ := 0.18)
  (prob_XYZ_given_XY : ℝ := 1 / 4) : 
  (let x := prob_XY * prob_XYZ_given_XY in
   let no_X_population := 1 - (prob_only_X + prob_XY + prob_XZ) in
   x = 0.06 ∧ 
   no_X_population - x = 0.43 →
   19 / 43 = 19 / 43) := sorry

end prob_no_markers_given_no_X_l567_567989


namespace pressure_ratio_l567_567306

-- Define Q, Δu, and A
def Q (Δu A : ℝ) : ℝ := Δu + A

-- Define the relationship for Q = 0
def Q_zero (Δu A : ℝ) : Prop := Q Δu A = 0

-- Define Δu in terms of cv, T, and T0
def Δu (cv T T0 : ℝ) : ℝ := cv * (T - T0)

-- Define A in terms of k, x
def A (k x : ℝ) : ℝ := (k * x^2) / 2

-- Define relationship between k, x, P, and S
def pressure_relation (k x P S : ℝ) : Prop := k * x = P * S

-- Define change in volume
def ΔV (S x : ℝ) : ℝ := S * x

-- Define the expanded volume
def V (n S x : ℝ) : ℝ := (n / (n - 1)) * S * x

-- Ideal gas law relationships
def ideal_gas_law (P V R T : ℝ) : Prop := P * V = R * T

-- Initial and expanded states
def initial_state (P0 V0 R T0 : ℝ) : Prop := P0 * V0 = R * T0
def expanded_state (P n V0 R T : ℝ) : Prop := P * n * V0 = R * T

-- Define target proof statement
theorem pressure_ratio (cv k x P S n R T T0 P0 V0 : ℝ)
  (hQ_zero : Q_zero (Δu cv T T0) (A k x))
  (hPressRel : pressure_relation k x P S)
  (hIdealGasLaw : ideal_gas_law P (V n S x) R T)
  (hInitialState : initial_state P0 V0 R T0)
  (hExpandedState : expanded_state P n V0 R T) :
  P / P0 = 1 / (n * (1 + ((n - 1) * R) / (2 * n * cv))) :=
by sorry

end pressure_ratio_l567_567306


namespace successful_activation_process_l567_567202

theorem successful_activation_process (n : ℕ) (h : n ≥ 3)
  (points : Fin n → ℝ × ℝ)
  (h_no_collinear : ∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → 
    ¬collinear (points i) (points j) (points k)) :
  ∃ (P : List (Fin n)), directed_loop P ∧ nat.card {Q : List (Fin n) // directed_loop Q} ≤ 2 * n :=
sorry

-- Additional necessary definitions

def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
(p2.snd - p1.snd) * (p3.fst - p2.fst) = (p3.snd - p2.snd) * (p2.fst - p1.fst)

def directed_loop (P : List (Fin n)) : Prop :=
-- This definition should encode the logic of forming a directed loop with the points
sorry

end successful_activation_process_l567_567202


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567423

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567423


namespace prime_factor_of_sum_of_consecutive_integers_l567_567011

theorem prime_factor_of_sum_of_consecutive_integers (n : ℤ) : ∃ p : ℕ, (Nat.Prime p) ∧ (∀ n : ℤ, p ∣ (4 * n + 2)) :=
begin
  use 2,
  split,
  { exact Nat.prime_two },
  { intro n,
    apply dvd.intro (2 * n + 1),
    ring }
end

end prime_factor_of_sum_of_consecutive_integers_l567_567011


namespace abs_eq_neg_self_iff_l567_567111

theorem abs_eq_neg_self_iff (a : ℝ) : |a| = -a ↔ a ≤ 0 :=
by
  -- skipping proof with sorry
  sorry

end abs_eq_neg_self_iff_l567_567111


namespace ratio_of_width_to_length_l567_567553

-- Definitions of length, width, perimeter
def l : ℕ := 10
def P : ℕ := 30

-- Define the condition for the width
def width_from_perimeter (l P : ℕ) : ℕ :=
  (P - 2 * l) / 2

-- Calculate the width using the given length and perimeter
def w : ℕ := width_from_perimeter l P

-- Theorem stating the ratio of width to length
theorem ratio_of_width_to_length : (w : ℚ) / l = 1 / 2 := by
  -- Proof steps will go here
  sorry

end ratio_of_width_to_length_l567_567553


namespace slope_perpendicular_l567_567479

open Real

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_perpendicular (A : ℝ × ℝ) (origin : ℝ × ℝ) (hA : A = (3, 5)) (hO : origin = (0, 0)) :
  slope origin A * slope A (A.1 + 1, A.2 + 1) = -1 :=
by
  sorry

end slope_perpendicular_l567_567479


namespace prime_fraction_divisibility_l567_567582

theorem prime_fraction_divisibility (k m n : ℕ) (p : ℕ) (hp_prime : Prime p) (hp_form : p = 4 * k + 3)
  (h_fraction : (∑ i in Finset.range p, ((1 : ℚ) / ((i : ℚ)^2 + 1)) = (m : ℚ) / (n : ℚ))) :
  p ∣ (2 * m - n) :=
sorry

end prime_fraction_divisibility_l567_567582


namespace num_factors_of_M_l567_567868

theorem num_factors_of_M (M : ℕ) 
  (hM : M = (2^5) * (3^4) * (5^3) * (11^2)) : ∃ n : ℕ, n = 360 ∧ M = (2^5) * (3^4) * (5^3) * (11^2) := 
by
  sorry

end num_factors_of_M_l567_567868


namespace mandy_pieces_eq_fifteen_l567_567324

-- Define the initial chocolate pieces
def total_pieces := 60

-- Define Michael's share
def michael_share := total_pieces / 2

-- Define the remainder after Michael's share
def remainder_after_michael := total_pieces - michael_share

-- Define Paige's share
def paige_share := remainder_after_michael / 2

-- Define the remainder after Paige's share
def mandy_share := remainder_after_michael - paige_share

-- Theorem to prove Mandy gets 15 pieces
theorem mandy_pieces_eq_fifteen : mandy_share = 15 :=
by
  sorry

end mandy_pieces_eq_fifteen_l567_567324


namespace num_good_sets_l567_567605

-- Define the universal set I and the subsets A and B
def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define what it means for subsets A and B to be "good sets"
def good_sets (A B : Set ℕ) : Prop := A ∩ B = {1, 2, 3}

-- Theorem statement: the total number of "good sets" is 3^6
theorem num_good_sets : (finset.powerset I).filter (λ P, ∃ A B, good_sets A B).card = 3^6 :=
sorry

end num_good_sets_l567_567605


namespace intersection_sets_l567_567921

variable (Z : Set ℕ) -- Assume ℕ represents the set of integers here for simplicity.

def setA : Set ℤ := {x | x ∈ Z ∧ x^2 - 3 * x - 4 ≤ 0}
def setB : Set ℤ := {x | x^2 - 2 * x ≤ 0}

theorem intersection_sets (A B : Set ℤ) (hA : A = {x | x ∈ Z ∧ x^2 - 3 * x - 4 ≤ 0})
  (hB : B = {x | x^2 - 2 * x ≤ 0}) : 
  A ∩ B = {0, 1, 2} :=
by
  subst hA
  subst hB
  sorry

end intersection_sets_l567_567921


namespace acute_angle_in_first_quadrant_l567_567783

theorem acute_angle_in_first_quadrant (θ : ℝ) (h : 0 < θ ∧ θ < 90) : θ ∈ set.Ioo 0 90 := 
by {
  intro h,
  exact h,
  sorry
}

end acute_angle_in_first_quadrant_l567_567783


namespace sum_of_divisors_143_l567_567772

theorem sum_of_divisors_143 : ∑ d in List.toFinset (List.divisors 143), d = 168 := by
  have prime_factors_143 : 143 = 11 * 13 := by
    calc 143 = 11 * 13 : by norm_num
  sorry

end sum_of_divisors_143_l567_567772


namespace pqr_value_l567_567638

theorem pqr_value (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h1 : p + q + r = 24)
  (h2 : (1 / p : ℚ) + (1 / q) + (1 / r) + 240 / (p * q * r) = 1): 
  p * q * r = 384 :=
by
  sorry

end pqr_value_l567_567638


namespace product_a_n_l567_567041

theorem product_a_n :
  ∏ (n : ℕ) in Finset.range (101) \ Finset.range (6), 
    (n^2 + 4 * n + 5) / (n^3 - 1) = 79 / 100! := by
  sorry

end product_a_n_l567_567041


namespace S_maximized_at_16_l567_567062

variable (a : ℕ → ℝ)
variable (b : ℕ → ℝ)
variable (S : ℕ → ℝ)
variable (d : ℝ)

-- Conditions
axiom a_sequence : ∀ n : ℕ, a n = a 1 + (n - 1) * d
axiom a_5_condition : 3 * a 5 = 8 * a 12
axiom a_5_positive : 3 * a 5 > 0
axiom b_definition : ∀ n : ℕ, b n = a n * a (n + 1) * a (n + 2)
axiom S_definition : ∀ n : ℕ, S n = ∑ i in range n, b i

-- Goal
theorem S_maximized_at_16 : argmax S = 16 := sorry

end S_maximized_at_16_l567_567062


namespace length_of_BC_l567_567117

variable (A B C X : Type)
variable [OrderedRing A]
variable [OrderedRing B]
variable [OrderedRing C]
variable [OrderedRing X]

-- Definitions for the specific lengths and conditions
def AB : A := 75
def AC : A := 100

-- Definitions of integer segments
variable (BX CX BC : A)

-- Conditions that BX and CX are integers and the circle centered at A with radius AB intersects BC at B and X
axiom circle_intersects : (BX : ℤ) ∧ (CX : ℤ)

-- Power of a point theorem applied and the specific conditions from the problem
axiom power_of_point : (CX * (CX + BX) = 4375)

-- Asking for proof that BC equals 125 given the conditions
theorem length_of_BC : BC = 125 :=
by 
  -- eliding the proof details
  sorry

end length_of_BC_l567_567117


namespace S_shaped_growth_curve_varied_growth_rate_l567_567995

theorem S_shaped_growth_curve_varied_growth_rate :
  ∀ (population_growth : ℝ → ℝ), 
    (∃ t1 t2 : ℝ, t1 < t2 ∧ 
      (∃ r : ℝ, r = population_growth t1 / t1 ∧ r ≠ population_growth t2 / t2)) 
    → 
    ∀ t3 t4 : ℝ, t3 < t4 → (population_growth t3 / t3) ≠ (population_growth t4 / t4) :=
by
  sorry

end S_shaped_growth_curve_varied_growth_rate_l567_567995


namespace asymptotic_lines_of_hyperbola_l567_567410

open Real

-- Given: Hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- To Prove: Asymptotic lines equation
theorem asymptotic_lines_of_hyperbola : 
  ∀ x y : ℝ, hyperbola x y → (y = x ∨ y = -x) :=
by
  intros x y h
  sorry

end asymptotic_lines_of_hyperbola_l567_567410


namespace lines_perpendicular_and_parallel_l567_567799

noncomputable def l1 l2 l3 : Type → Prop := sorry
variables (l1 l2 l3 : Type) [linear_ordered_field 𝔞] [add_comm_group l1] [module 𝔞 l1] 
  [add_comm_group l2] [module 𝔞 l2] [add_comm_group l3] [module 𝔞 l3]

/-- 
  Assume three different lines l1, l2, l3 in space.
  If l1 is perpendicular to l2 and l2 is parallel to l3, 
  then l1 is perpendicular to l3.
-/
theorem lines_perpendicular_and_parallel (h1 : l1 ≠ l2) (h2 : l2 ≠ l3) (h3 : l3 ≠ l1)
  (h4 : l1.perpendicular l2) (h5 : l2.parallel l3) : l1.perpendicular l3 :=
sorry

end lines_perpendicular_and_parallel_l567_567799


namespace sum_of_divisors_143_l567_567771

theorem sum_of_divisors_143 : ∑ d in List.toFinset (List.divisors 143), d = 168 := by
  have prime_factors_143 : 143 = 11 * 13 := by
    calc 143 = 11 * 13 : by norm_num
  sorry

end sum_of_divisors_143_l567_567771


namespace min_button_presses_to_open_lock_l567_567352

-- Definitions based on the problem conditions
def sequence := list char -- Define the sequence as a list of characters (representing button presses)
def valid_chars : list char := ['A', 'B', 'C'] -- The possible characters

-- Function to generate all combinations of sequences of length 3
def all_sequences_of_length_3 : list (list char) :=
(list.replicate 3 valid_chars).product_foldl list.cons list.nil

-- Property to check if a given sequence contains all subsequences of length 3
def contains_all_subsequences_of_length_3 (seq : sequence) : Prop :=
(all_sequences_of_length_3 ⊆ seq.tails.map (list.take 3))

-- Main theorem stating the minimum number of button presses needed
theorem min_button_presses_to_open_lock (s : sequence) (h : contains_all_subsequences_of_length_3 s) : s.length ≥ 29 :=
sorry

end min_button_presses_to_open_lock_l567_567352


namespace jill_speed_up_is_9_l567_567147

-- Conditions as given in the problem
def distance : ℝ := 900
def time_down : ℝ := distance / 12
def total_time : ℝ := 175

-- Jill's speed up the hill
def speed_up (v : ℝ) : ℝ := distance / v

-- Problem Statement: Prove Jill's speed running up the hill is 9 feet/second
theorem jill_speed_up_is_9 (v : ℝ) (h1 : speed_up v + time_down = total_time) : v = 9 :=
sorry

end jill_speed_up_is_9_l567_567147


namespace computer_price_problem_l567_567978

theorem computer_price_problem (x : ℝ) (h : x + 0.30 * x = 351) : x + 351 = 621 :=
by
  sorry

end computer_price_problem_l567_567978


namespace f_2014_eq_one_l567_567386

noncomputable def f : ℝ → ℝ
| x if x ≤ 0 := Real.log (1 - x) / Real.log 2
| x if x > 0 := f (x - 1) - f (x - 2)
| _ := 0  -- By default, to avoid non-exhaustiveness

theorem f_2014_eq_one : f 2014 = 1 :=
sorry

end f_2014_eq_one_l567_567386


namespace sequence_eventually_repeats_l567_567057

-- Define the sequence and the operation that replaces any number with the sum of the numbers to its right.
def sequence_replacement (seq : List ℕ) (i : ℕ) : List ℕ :=
  if i < seq.length then
    let right_sum := (List.drop (i + 1) seq).sum
    seq.take i ++ [right_sum] ++ seq.drop (i + 1)
  else
    seq

-- Define the main theorem to be proved.
theorem sequence_eventually_repeats (seq : List ℕ) :
  ∃ N : ℕ, ∃ n ≥ N, (∀ i < seq.length, sequence_replacement seq i) = (∀ i < seq.length, sequence_replacement seq (i + 1)) :=
by
  sorry

end sequence_eventually_repeats_l567_567057


namespace minimum_lightest_weight_l567_567701

-- Definitions
def lightest_weight (m : ℕ) : Prop := ∃ n, 72 * m = 35 * n ∧ m % 35 = 0 ∧ m ≥ 35

-- Theorem statement
theorem minimum_lightest_weight : ∃ m, lightest_weight m ∧ m = 35 :=
by
  use 35
  split
  sorry
  exact rfl

end minimum_lightest_weight_l567_567701


namespace rectangular_field_area_l567_567250

theorem rectangular_field_area
  (x : ℝ) 
  (length := 3 * x) 
  (breadth := 4 * x) 
  (perimeter := 2 * (length + breadth))
  (cost_per_meter : ℝ := 0.25) 
  (total_cost : ℝ := 87.5) 
  (paise_per_rupee : ℝ := 100)
  (perimeter_eq_cost : 14 * x * cost_per_meter * paise_per_rupee = total_cost * paise_per_rupee) :
  (length * breadth = 7500) := 
by
  -- proof omitted
  sorry

end rectangular_field_area_l567_567250


namespace paths_inequality_l567_567860
open Nat

-- Definitions
def m : ℕ := sorry -- m represents the number of rows.
def n : ℕ := sorry -- n represents the number of columns.
def N : ℕ := sorry -- N is the number of ways to color the grid such that there is a path composed of black cells from the left edge to the right edge.
def M : ℕ := sorry -- M is the number of ways to color the grid such that there are two non-intersecting paths composed of black cells from the left edge to the right edge.

-- Theorem statement
theorem paths_inequality : (N ^ 2) ≥ 2 ^ (m * n) * M := 
by
  sorry

end paths_inequality_l567_567860


namespace minimum_value_fraction_l567_567905

-- Definitions and conditions from the problem statement
variables {A B C D : Type*}
variables (x y : ℝ)
variable  (BC_segment_contains_D : D ∈ line_segment B C) -- D lies on BC excluding endpoints
variable  (vector_relation_AD : vector AD = x * vector AB + y * vector AC)

theorem minimum_value_fraction (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : x + y = 1) : 
  inf {z | z = (1/x) + (2/y)} = 2*sqrt(2) + 3 :=
sorry -- proof goes here

end minimum_value_fraction_l567_567905


namespace find_f_neg_one_l567_567591

variable {f : ℝ → ℝ}
variable {b : ℝ}

-- f is an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define f when x >= 0
def f_def (x : ℝ) (b : ℝ) : ℝ :=
  2^x + 2*x + b

-- Main proof
theorem find_f_neg_one (h_odd : odd_function f) (h_def : ∀ x, 0 ≤ x → f x = f_def x b) (h_f0 : f 0 = 0) :
  f (-1) = -3 := by
  sorry

end find_f_neg_one_l567_567591


namespace parallel_B1D1_AE_l567_567818

open Set

-- Define the pentagon and related points
variables {ω : Circle} {A B C D E: Point}
variable {inscribed : ∀ (A B C D E: Point), InscribedPolygon ω [A, B, C, D, E]}
variables {E1 A1 : Point}
variables {intersection1 : IntersectAt (extend_line A B) (extend_line C D) E1}
variables {intersection2 : IntersectAt (extend_line B C) (extend_line D E) A1}
variables {B1 D1 : Point}
variables {tangent_at_B : TangentCircle B (circumcircle B E1 C) B1}
variables {tangent_at_D : TangentCircle D (circumcircle D A1 E) D1}
variables {obtuse_angles : ∀ (A B C D E : Point), (Angle(A) > 90° ∧ Angle(B) > 90° ∧ Angle(C) > 90° ∧ Angle(D) > 90° ∧ Angle(E) > 90°)}

theorem parallel_B1D1_AE :
  Parallel B1 D1 A E :=
sorry

end parallel_B1D1_AE_l567_567818


namespace a_1_is_3_geometric_seq_l567_567477

-- the condition: sequence S_n is the sum of first n terms of sequence a_n
def is_sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = ∑ i in Finset.range n, a i

-- the condition: sequence {sqrt(S_n + 1)} is a geometric sequence with common ratio 2
def is_geometric_sqrt_seq (S : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r = 2 ∧ ∀ n, (Sqrt (S n + 1)) = (Sqrt (S 1 + 1)) * r^(n - 1) 

-- proving that a_1 = 3 is the necessary and sufficient condition for {a_n} to be a geometric sequence
theorem a_1_is_3_geometric_seq (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (is_sum_of_first_n_terms a S) ∧ (is_geometric_sqrt_seq S) ↔ (a 1 = 3) :=
begin
  sorry  -- proof is not required
end

end a_1_is_3_geometric_seq_l567_567477


namespace eight_pow_negative_x_correct_l567_567961

noncomputable def eight_pow_negative_x (x : ℝ) (h : 8^(3 * x) = 64) : Prop :=
  8^(-x) = 1 / 4

-- Define the theorem to be proved
theorem eight_pow_negative_x_correct (x : ℝ) (h : 8^(3 * x) = 64) : eight_pow_negative_x x h :=
sorry

end eight_pow_negative_x_correct_l567_567961


namespace cevian_concurrency_l567_567604

theorem cevian_concurrency
  (A B C Z X Y : ℝ)
  (a b c s : ℝ)
  (h1 : s = (a + b + c) / 2)
  (h2 : AZ = s - c) (h3 : ZB = s - b)
  (h4 : BX = s - a) (h5 : XC = s - c)
  (h6 : CY = s - b) (h7 : YA = s - a)
  : (AZ / ZB) * (BX / XC) * (CY / YA) = 1 :=
by
  sorry

end cevian_concurrency_l567_567604


namespace sum_of_divisors_of_143_l567_567765

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (nat.divisors n).toFinset, k

theorem sum_of_divisors_of_143 : sum_divisors 143 = 168 :=
by sorry

end sum_of_divisors_of_143_l567_567765


namespace solution_set_l567_567535

-- Define the function and the conditions
variable {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

-- Problem statement
theorem solution_set (hf_even : is_even f)
                     (hf_increasing : increasing_on f (Set.Ioi 0))
                     (hf_value : f (-2013) = 0) :
  {x | x * f x < 0} = {x | x < -2013 ∨ (0 < x ∧ x < 2013)} :=
by
  sorry

end solution_set_l567_567535


namespace area_of_square_l567_567058

noncomputable def square_area (u v : ℝ) (h_u : 0 < u) (h_v : 0 < v) : ℝ :=
  (v * v) / 4

theorem area_of_square (u v : ℝ) (h_u : 0 < u) (h_v : 0 < v) (h_cond : ∀ (A B C : ℝ × ℝ), 
  A = (0, 0) → B = (u, 0) → C = (u, v) → 
  (u - 0) * (u - 0) + (v - 0) * (v - 0) = (u - 0) * (u - 0)) :
  square_area u v h_u h_v = v * v / 4 := 
by 
  sorry

end area_of_square_l567_567058


namespace complex_number_in_second_quadrant_l567_567809

-- We must prove that the complex number z = i / (1 - i) lies in the second quadrant
theorem complex_number_in_second_quadrant : let z := (Complex.I / (1 - Complex.I)) in (z.re < 0) ∧ (z.im > 0) :=
by
  let z := (Complex.I / (1 - Complex.I))
  have h1 : (1 - Complex.I).conj = 1 + Complex.I := by sorry
  have h2 : (Complex.I * (1 + Complex.I)) = -1 + Complex.I := by sorry
  have h3 : (1 - Complex.I) * (1 + Complex.I) = 2 := by sorry
  have h4 : z = (-1 + Complex.I) / 2 := by sorry
  have h5 : z.re = -1 / 2 := by sorry
  have h6 : z.im = 1 / 2 := by sorry
  show (z.re < 0) ∧ (z.im > 0), from and.intro (by norm_num) (by norm_num)

end complex_number_in_second_quadrant_l567_567809


namespace lower_limit_of_range_with_multiples_l567_567261

theorem lower_limit_of_range_with_multiples (n : ℕ) (h : 2000 - n ≥ 198 * 10 ∧ n % 10 = 0 ∧ n + 1980 ≤ 2000) :
  n = 30 :=
by
  sorry

end lower_limit_of_range_with_multiples_l567_567261


namespace range_of_x_l567_567089

open Real

noncomputable def f (x : ℝ) : ℝ := log x - (x / (1 + 2 * x))

theorem range_of_x (x : ℝ) :
  f (x * (3 * x - 2)) < -1 / 3 ↔ (-(1 / 3) < x ∧ x < 0) ∨ ((2 / 3) < x ∧ x < 1) :=
by
  sorry

end range_of_x_l567_567089


namespace min_xy_l567_567115

theorem min_xy (x y : ℝ) (h : 1 + cos (2 * x + 3 * y - 1) ^ 2 = (x ^ 2 + y ^ 2 + 2 * (x + 1) * (1 - y)) / (x - y + 1)) :
  xy = 1 / 25 :=
sorry

end min_xy_l567_567115


namespace problem_statement_l567_567374

noncomputable def sqrt_eight : ℝ := real.sqrt 8
noncomputable def abs_sqrt_two_minus_two : ℝ := abs (real.sqrt 2 - 2)
noncomputable def inv_neg_half : ℝ := (- (1 / 2))⁻¹

theorem problem_statement :
  sqrt_eight + abs_sqrt_two_minus_two + inv_neg_half = real.sqrt 2 :=
by sorry

end problem_statement_l567_567374


namespace digit_properties_l567_567669

theorem digit_properties {x : ℕ} 
  (h1: 12 * x - 21 * x = 36)   -- Condition: difference between numbers is 36
  (h2: ∃ k, 2 * k = x)        -- Condition: ratio of 1:2 between digits
  (hx : x < 10)               -- Condition: digits have to be less than 10
  : ((10 * x + 2 * x) % 10 + ((2 * 10 * x) % 10)) - ((2 * (10 * x) % 10 ) - x) = 6 := 
begin
  sorry
end

end digit_properties_l567_567669


namespace probability_spade_then_king_l567_567278

theorem probability_spade_then_king :
  let spades := 13
  let total_cards := 52
  let non_spade_kings := 3
  let kings := 4
  let prob_case1 := (12 / total_cards) * (kings / (total_cards - 1))
  let prob_case2 := (1 / total_cards) * (non_spade_kings / (total_cards - 1))
  prob_case1 + prob_case2 = (17 / 884) :=
by {
  let spades := 13
  let total_cards := 52
  let non_spade_kings := 3
  let kings := 4
  let prob_case1 := (12 / total_cards) * (kings / (total_cards - 1))
  let prob_case2 := (1 / total_cards) * (non_spade_kings / (total_cards - 1))
  have h1 : prob_case1 = 48 / 2652 := sorry,
  have h2 : prob_case2 = 3 / 2652 := sorry,
  calc
    prob_case1 + prob_case2 = (48 / 2652) + (3 / 2652) : by rw [h1, h2]
    ... = 51 / 2652 : by norm_num
    ... = 17 / 884 : by norm_num
}

end probability_spade_then_king_l567_567278


namespace sum_of_divisors_143_l567_567725

theorem sum_of_divisors_143 : sigma 143 = 168 := by
  have h : 143 = 11 * 13 := by norm_num
  rw [sigma_mul_prime_prime 11 13]
  norm_num
  sorry

end sum_of_divisors_143_l567_567725


namespace gcd_bn_bn1_l567_567590

def b (n : ℕ) : ℤ := (7^n - 1) / 6
def e (n : ℕ) : ℤ := Int.gcd (b n) (b (n + 1))

theorem gcd_bn_bn1 (n : ℕ) : e n = 1 := by
  sorry

end gcd_bn_bn1_l567_567590


namespace amount_paid_l567_567143

-- Defining the conditions as constants
def cost_of_apple : ℝ := 0.75
def change_received : ℝ := 4.25

-- Stating the theorem that needs to be proved
theorem amount_paid (a : ℝ) : a = cost_of_apple + change_received :=
by
  sorry

end amount_paid_l567_567143


namespace N_inequality_l567_567180

variables {n : ℕ} {x w : Finₓ n → ℝ} {p q : ℝ}

def N_p (p : ℝ) (w x : Finₓ n → ℝ) : ℝ :=
if p = 0 then
    ∏ i, x i ^ w i
else if p = +∞ then
    max (Set.range x)
else if p = -∞ then
    min (Set.range x)
else
    (∑ i, w i * (x i) ^ p) ^ (1 / p)

theorem N_inequality (x : Finₓ n → ℝ) (w : Finₓ n → ℝ)
  (h_pos_w : ∀ i, 0 < w i) (h_sum_w : ∑ i, w i = 1)
  (h_pos_x : ∀ i, 0 < x i) (p q : ℝ)
  (hq : q ≠ 0) (hp : p ≠ q) (hp_gt_q : p > q) :
  N_p p w x ≥ N_p q w x ∧ (N_p p w x = N_p q w x ↔ ∀ i j, x i = x j) :=
sorry

end N_inequality_l567_567180


namespace max_ab_bc_2ac_l567_567869

theorem max_ab_bc_2ac (a b c : ℝ) (h_non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) (h_sum : a + b + c = 1) :
  ab + bc + 2ac ≤ 1/2 :=
sorry

end max_ab_bc_2ac_l567_567869


namespace problem_solution_l567_567335

theorem problem_solution (b : ℚ) :
  (∃ q : ℚ × ℚ, q.2 = b ∧
   ∃ p1 p2 : ℚ × ℚ, 
     p1.1 = 0 ∧ p1.2 = 0 ∧ 
     p2.1 ≠ 0 ∧ 
     (p1.2 = 4 / 3 * p1.1^2) ∧
     (p2.2 = 4 / 3 * p2.1^2) ∧ 
     (p2.2 = 4 / 3 * p2.1 + b ∧ 
      (∃ c : ℚ, c = (4/3 * p2.1 + b)))) → 
  b = 25 / 12 :=
begin
  sorry
end

end problem_solution_l567_567335


namespace cards_distribution_l567_567973

theorem cards_distribution (cards : ℕ) (people : ℕ) (h1 : cards = 60) (h2 : people = 7) :
  (let cards_per_person := cards / people in
   let remainder := cards % people in
   let full_hand := cards_per_person + 1 in
   let full_hand_count := remainder in
   let less_than_full_hand_count := people - full_hand_count in
   less_than_full_hand_count) = 3 := by
  have h_div : 60 / 7 = 8 := by norm_num
  have h_mod : 60 % 7 = 4 := by norm_num
  rw [h1, h2, h_div, h_mod]
  norm_num

end cards_distribution_l567_567973


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567436

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567436


namespace largest_int_mod_6_less_than_100_l567_567446

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l567_567446


namespace lottery_jackpot_probability_l567_567131

noncomputable def C (n k : ℕ) : ℕ := Fact.factorial n / (Fact.factorial k * Fact.factorial (n - k))

theorem lottery_jackpot_probability :
  (C 45 6 = 8145060) →
  (100: ℚ) / (C 45 6: ℚ) = 0.0000123 :=
by
  sorry

end lottery_jackpot_probability_l567_567131


namespace isosceles_triangle_sum_l567_567292

theorem isosceles_triangle_sum :
  let A := (Real.cos (30 * Real.pi / 180), Real.sin (30 * Real.pi / 180))
  let B := (Real.cos (90 * Real.pi / 180), Real.sin (90 * Real.pi / 180))
  ∑ t in { t : Real | 0 ≤ t ∧ t ≤ 360 ∧
                     let C := (Real.cos (t * Real.pi / 180), Real.sin (t * Real.pi / 180))
                     (((C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2) ∨
                      ((C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2) ∨
                      ((A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2)) }, t = 1200 :=
sorry

end isosceles_triangle_sum_l567_567292


namespace geometric_sequence_a5_l567_567130

-- Define the initial conditions of the geometric sequence
def a_1 := 1 / 3
def q : ℝ := Real.sqrt 2

-- Given the relationship 2 * a_2 = a_4, verify q^2 = 2
lemma q_squared_is_two : q ^ 2 = 2 :=
by 
  show (Real.sqrt 2) ^ 2 = 2
  exact Real.sq_sqrt (by norm_num) 

-- Define the geometric sequence
noncomputable def a (n : ℕ) : ℝ := a_1 * q ^ (n - 1)

-- Given statement to prove
theorem geometric_sequence_a5 : a 5 = 4 / 3 :=
by 
  have h1 := q_squared_is_two
  have h2 : q ^ 4 = (q ^ 2) ^ 2 := by ring
  have h3 : (q ^ 2) ^ 2 = 4 := by rw [h1]; norm_num
  show a 5 = 4 / 3
  rw [a, h2, h3]
  norm_num

end geometric_sequence_a5_l567_567130


namespace square_side_length_l567_567242

theorem square_side_length (x : ℝ) (h : 4 * x = x^2) : x = 4 := 
by
  sorry

end square_side_length_l567_567242


namespace first_player_winning_strategy_l567_567547

-- Definitions based on conditions
def initial_position (m n : ℕ) : ℕ × ℕ := (m - 1, n - 1)

-- Main theorem statement
theorem first_player_winning_strategy (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  (initial_position m n).fst ≠ (initial_position m n).snd ↔ m ≠ n :=
by
  sorry

end first_player_winning_strategy_l567_567547


namespace solve_for_x_l567_567632

theorem solve_for_x :
  let denom := (7 - 3 / 4 + 1 / 8 : ℚ)
  in (48 / denom : ℚ) = (128 / 17 : ℚ) :=
by
  sorry

end solve_for_x_l567_567632


namespace widow_share_l567_567813

theorem widow_share (w d s : ℝ) (h_sum : w + 5 * s + 4 * d = 8000)
  (h1 : d = 2 * w)
  (h2 : s = 3 * d) :
  w = 8000 / 39 := by
sorry

end widow_share_l567_567813


namespace terminating_decimal_numbers_count_l567_567043

def is_terminating_decimal (n: ℕ) : Prop :=
  ∃ m : ℕ, ∃ k : ℤ, n = k * (2^m) * (5^m)

theorem terminating_decimal_numbers_count :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 499 → is_terminating_decimal (n / 500) = true :=
begin
  sorry
end

end terminating_decimal_numbers_count_l567_567043


namespace total_workers_l567_567866

theorem total_workers (h_beavers : ℕ := 318) (h_spiders : ℕ := 544) :
  h_beavers + h_spiders = 862 :=
by
  sorry

end total_workers_l567_567866


namespace total_pencils_l567_567843

def initial_pencils : ℕ := 9
def additional_pencils : ℕ := 56

theorem total_pencils : initial_pencils + additional_pencils = 65 :=
by
  -- proof steps are not required, so we use sorry
  sorry

end total_pencils_l567_567843


namespace triangle_XYZ_area_l567_567635

-- Definitions and conditions
variables {O A B C D : Type} -- Points
variable (ABCD : set Type) -- Square ABCD
variable (circle : set Type) -- Circle
variable (X Y Z : Type) -- Points of Triangle XYZ
variable (side_length : ℕ)

axiom square_inscribed_in_circle :
  square ABCD → ∃ (O : Type) (r : ℝ), (side_length^2 = 256) ∧ 
  (∀ p ∈ ABCD, dist O p = r ∧ 
  ∀ p ∈ circle, dist O p = r)

-- Given conditions
axiom midpoint_X : midpoint A B X
axiom vertices_on_circle : ∀ p ∈ {Y, Z}, p ∈ circle

-- Proof statement:
theorem triangle_XYZ_area : 
  ∀ (O : Type) (X Y Z : Type) (side_length : ℕ),
  (
    square ABCD → 
    ∃ (O : Type) (r : ℝ), (side_length^2 = 256) ∧ 
    (∀ p ∈ ABCD, dist O p = r ∧ 
    ∀ p ∈ circle, dist O p = r)
  ) → 
  (midpoint A B X) → 
  (∀ p ∈ {Y, Z}, p ∈ circle) → 
  (area_of_triangle X Y Z = 32) :=
sorry

end triangle_XYZ_area_l567_567635


namespace bowen_spending_l567_567356

noncomputable def total_amount_spent (pen_cost pencil_cost : ℕ) (number_of_pens number_of_pencils : ℕ) : ℕ :=
  number_of_pens * pen_cost + number_of_pencils * pencil_cost

theorem bowen_spending :
  let pen_cost := 15 in
  let pencil_cost := 25 in
  let number_of_pens := 40 in
  let number_of_pencils := number_of_pens + (2 * number_of_pens / 5) in
  total_amount_spent pen_cost pencil_cost number_of_pens number_of_pencils = 2000 :=
by
  sorry

end bowen_spending_l567_567356


namespace candy_bars_saved_l567_567575

theorem candy_bars_saved
  (candy_bars_per_week : ℕ)
  (weeks : ℕ)
  (candy_bars_eaten_per_4_weeks : ℕ) :
  candy_bars_per_week = 2 →
  weeks = 16 →
  candy_bars_eaten_per_4_weeks = 1 →
  (candy_bars_per_week * weeks) - (weeks / 4 * candy_bars_eaten_per_4_weeks) = 28 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end candy_bars_saved_l567_567575


namespace triangle_area_ratio_l567_567136

noncomputable theory

open_locale big_operators

variables {P Q R S T : Type*} [linear_ordered_field P Q R S T]

def trapezoid (PQ: P) (RS: Q) : Prop :=
RS = 2 * PQ 

def height_trapezoid : R :=
6

def area_ratio (TPQ PQRS: R) : Prop :=
(TPQ / PQRS) = 1 / 3

theorem triangle_area_ratio (hPQRS: height_trapezoid) (htrapezoid: trapezoid PQ RS) 
  (harea: area_ratio TPQ PQRS) : 
  TPQ / PQRS = 1 / 3 :=
begin
  sorry
end

end triangle_area_ratio_l567_567136


namespace range_of_a_l567_567980

theorem range_of_a (a : ℝ) : 
  (∃ x : ℤ, 2 < (x : ℝ) ∧ (x : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ y : ℤ, 2 < (y : ℝ) ∧ (y : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ z : ℤ, 2 < (z : ℝ) ∧ (z : ℝ) ≤ 2 * a - 1) ∧ 
  (∀ w : ℤ, 2 < (w : ℝ) ∧ (w : ℝ) ≤ 2 * a - 1 → w = 3 ∨ w = 4 ∨ w = 5) :=
  by
    sorry

end range_of_a_l567_567980


namespace find_initial_dogs_l567_567150

def initial_dogs (D : ℕ) (cats : ℕ) (lizards : ℕ) (new_pets: ℕ) (total_pets: ℕ) :=
  0 ≤ D ∧ 0 ≤ cats ∧ 0 ≤ lizards ∧ 0 ≤ new_pets ∧ 0 ≤ total_pets

theorem find_initial_dogs :
  ∃ D : ℕ, initial_dogs D 28 20 13 65 ∧ (0.5 * D + 0.75 * 28 + 0.8 * 20 + 13 = 65) ∧ D = 30 :=
by
  sorry -- Proof omitted

end find_initial_dogs_l567_567150


namespace extra_fee_packages_count_l567_567246

theorem extra_fee_packages_count:
  let length_x := 8
  let width_x := 5
  let length_y := 12
  let width_y := 4
  let length_z := 9
  let width_z := 9
  let length_w := 14
  let width_w := 5

  let ratio_x := (length_x: ℝ) / (width_x: ℝ)
  let ratio_y := (length_y: ℝ) / (width_y: ℝ)
  let ratio_z := (length_z: ℝ) / (width_z: ℝ)
  let ratio_w := (length_w: ℝ) / (width_w: ℝ)

  let is_out_of_range (r: ℝ) : Bool := r < 1.5 ∨ r > 3.0

  (is_out_of_range ratio_x =
    (ratio_x < 1.5 ∨ ratio_x > 3.0)) ∧
  (is_out_of_range ratio_y =
    (ratio_y < 1.5 ∨ ratio_y > 3.0)) ∧
  (is_out_of_range ratio_z =
    (ratio_z < 1.5 ∨ ratio_z > 3.0)) ∧
  (is_out_of_range ratio_w =
    (ratio_w < 1.5 ∨ ratio_w > 3.0)) →
  [is_out_of_range ratio_x, is_out_of_range ratio_y, is_out_of_range ratio_z, is_out_of_range ratio_w].count true = 1 :=
begin
  sorry
end

end extra_fee_packages_count_l567_567246


namespace three_a_plus_two_b_value_l567_567671

theorem three_a_plus_two_b_value : 
  ∀ a b : ℝ, (x : ℝ) (h₁ : x^2 - 6 * x + 13 = 25) 
  (h₂: x = a ∨ x = b) 
  (h₃ : a ≥ b), 
  3 * a + 2 * b = 15 + sqrt 21 :=
by
  sorry

end three_a_plus_two_b_value_l567_567671


namespace solve_jack_water_problem_l567_567568

def jack_water_problem : Prop :=
  ∀ (W V : ℝ), W = 3 ∧ V = 0.5 → (∃ b : ℕ, W / V = b ∧ b = 6)

theorem solve_jack_water_problem : jack_water_problem :=
by
  unfold jack_water_problem
  intros W V h
  cases h with hw hv
  use (W / V).to_nat
  simp [hw, hv]
  sorry

end solve_jack_water_problem_l567_567568


namespace abs_neg_one_over_2023_l567_567642

theorem abs_neg_one_over_2023 : abs (-1 / 2023) = 1 / 2023 :=
by
  sorry

end abs_neg_one_over_2023_l567_567642


namespace correct_equation_l567_567524

namespace MathProblem

def is_two_digit_positive_integer (P : ℤ) : Prop :=
  10 ≤ P ∧ P < 100

def equation_A : Prop :=
  ∀ x : ℤ, x^2 + (-98)*x + 2001 = (x - 29) * (x - 69)

def equation_B : Prop :=
  ∀ x : ℤ, x^2 + (-110)*x + 2001 = (x - 23) * (x - 87)

def equation_C : Prop :=
  ∀ x : ℤ, x^2 + 110*x + 2001 = (x + 23) * (x + 87)

def equation_D : Prop :=
  ∀ x : ℤ, x^2 + 98*x + 2001 = (x + 29) * (x + 69)

theorem correct_equation :
  is_two_digit_positive_integer 98 ∧ equation_D :=
  sorry

end MathProblem

end correct_equation_l567_567524


namespace number_of_valid_mappings_l567_567163

def M : Set ℕ := {a, b, c}
def N : Set ℤ := {-1, 0, 1}

def valid_mapping (f : ℕ → ℤ) : Prop :=
  f a ∈ N ∧ f b ∈ N ∧ f c ∈ N ∧ (f a + f b + f c = 0)

noncomputable def count_valid_mappings : Nat :=
  (Set.toFinset {f | valid_mapping f}).card

theorem number_of_valid_mappings : count_valid_mappings = 7 :=
by
  sorry

end number_of_valid_mappings_l567_567163


namespace bing_location_subject_l567_567688

-- Defining entities
inductive City
| Beijing
| Shanghai
| Chongqing

inductive Subject
| Mathematics
| Chinese
| ForeignLanguage

inductive Teacher
| Jia
| Yi
| Bing

-- Defining the conditions
variables (works_in : Teacher → City) (teaches : Teacher → Subject)

axiom cond1_jia_not_beijing : works_in Teacher.Jia ≠ City.Beijing
axiom cond1_yi_not_shanghai : works_in Teacher.Yi ≠ City.Shanghai
axiom cond2_beijing_not_foreign : ∀ t, works_in t = City.Beijing → teaches t ≠ Subject.ForeignLanguage
axiom cond3_shanghai_math : ∀ t, works_in t = City.Shanghai → teaches t = Subject.Mathematics
axiom cond4_yi_not_chinese : teaches Teacher.Yi ≠ Subject.Chinese

-- The question
theorem bing_location_subject : 
  works_in Teacher.Bing = City.Beijing ∧ teaches Teacher.Bing = Subject.Chinese :=
by
  sorry

end bing_location_subject_l567_567688


namespace exists_unobserved_planet_l567_567618

-- Definitions and assumptions based on given conditions
variables (n : ℕ) [Odd n]
variables (d : Finset (Finset ℕ)) -- to represent pairwise different distances
variables (observes : Fin n → Fin n) -- represents each astronomer's observation

-- Assumption: each planet observes the nearest planet (pairwise distinct distances imply unique observing rules)
axiom observing_nearest (i : Fin n) : observes i ≠ i

-- The main theorem statement
theorem exists_unobserved_planet : ∃ (i : Fin n), ∀ (j : Fin n), observes j ≠ i :=
by
  sorry -- proof goes here

end exists_unobserved_planet_l567_567618


namespace non_zero_digits_in_decimal_l567_567963

theorem non_zero_digits_in_decimal (n d : ℕ) (h : (n = 90) ∧ (d = 2^4 * 5^9)) :
  (let decimal_repr := n / d  in
   (∀ (k : ℕ), decimal_repr = 288 / 10^8 → k = 3) ) := by
  sorry

end non_zero_digits_in_decimal_l567_567963


namespace sqrt_equation_solution_l567_567889

theorem sqrt_equation_solution (x : ℝ) (h : x > 6) :
  (sqrt (x - 6 * sqrt (x - 6)) + 3 = sqrt (x + 6 * sqrt (x - 6)) - 3) ↔ x ∈ set.Ici 18 :=
sorry

end sqrt_equation_solution_l567_567889


namespace sum_of_divisors_143_l567_567739

theorem sum_of_divisors_143 : 
  (finset.range (144)).filter (λ d, 143 % d = 0).sum = 168 :=
by
  sorry

end sum_of_divisors_143_l567_567739


namespace find_a_l567_567397

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end find_a_l567_567397


namespace rich_walks_total_distance_l567_567626

-- Definitions of the distances Rich walks at each step
def dist1 := 20
def dist2 := 200

-- Total distance so far after the first two segments
def total1 := dist1 + dist2

-- Distance walked after making a left
def dist3 := 2 * total1

-- Total distance so far after the third segment
def total2 := total1 + dist3

-- Distance walked after walking half the total distance so far
def dist4 := 0.5 * total2

-- Total distance to the end of the route
def total3 := total2 + dist4

-- The total distance walked including the return trip
def total_distance := total3 + total3

-- The proof statement
theorem rich_walks_total_distance : total_distance = 1980 := by
  sorry

end rich_walks_total_distance_l567_567626


namespace find_V_D_l567_567781

noncomputable def V_A : ℚ := sorry
noncomputable def V_B : ℚ := sorry
noncomputable def V_C : ℚ := sorry
noncomputable def V_D : ℚ := sorry
noncomputable def V_E : ℚ := sorry

axiom condition1 : V_A + V_B + V_C + V_D + V_E = 1 / 7.5
axiom condition2 : V_A + V_C + V_E = 1 / 5
axiom condition3 : V_A + V_C + V_D = 1 / 6
axiom condition4 : V_B + V_D + V_E = 1 / 4

theorem find_V_D : V_D = 1 / 12 := 
  by
    sorry

end find_V_D_l567_567781


namespace every_positive_integer_sum_form_l567_567630

theorem every_positive_integer_sum_form (n : ℕ) (h : n > 0) :
  ∃ (l : List ℕ), (∀ x ∈ l, ∃ r s : ℕ, x = 2^r * 3^s) ∧ (∀ x y ∈ l, x ≠ y → ¬ (x ∣ y) ∧ ¬ (y ∣ x)) ∧ l.sum = n :=
by
  sorry

end every_positive_integer_sum_form_l567_567630


namespace sum_of_divisors_143_l567_567730

theorem sum_of_divisors_143 : sigma 143 = 168 := by
  have h : 143 = 11 * 13 := by norm_num
  rw [sigma_mul_prime_prime 11 13]
  norm_num
  sorry

end sum_of_divisors_143_l567_567730


namespace lattice_points_on_curve_l567_567517

theorem lattice_points_on_curve :
  {p : ℤ × ℤ | p.1^2 - p.2^2 = 65}.to_finset.card = 6 :=
by
  -- proof to be filled in later
  sorry

end lattice_points_on_curve_l567_567517


namespace sin_theta_value_l567_567073

open Real

theorem sin_theta_value
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioo (3 * π / 4) (5 * π / 4))
  (h2 : sin (θ - π / 4) = 5 / 13) :
  sin θ = - (7 * sqrt 2) / 26 :=
  sorry

end sin_theta_value_l567_567073


namespace lollipop_surface_area_is_9pi_l567_567833

-- Constants and conditions
def base_radius : ℝ := 3      -- Base radius of cylindrical container in cm
def height : ℝ := 10          -- Height of cylindrical container in cm
def num_lollipops : ℕ := 20   -- Number of lollipops

-- Volumes
def volume_cylinder := Real.pi * base_radius^2 * height
def volume_lollipop := volume_cylinder / num_lollipops

-- Radius of lollipop and surface area calculation
def lollipop_radius := (volume_lollipop / ((4 / 3) * Real.pi))^(1 / 3)
def surface_area_lollipop := 4 * Real.pi * lollipop_radius^2

-- Theorem to prove the surface area of each lollipop
theorem lollipop_surface_area_is_9pi : surface_area_lollipop = 9 * Real.pi := by
  sorry

end lollipop_surface_area_is_9pi_l567_567833


namespace number_of_combinations_of_two_products_l567_567106

theorem number_of_combinations_of_two_products :
  nat.choose 4 2 = 6 := 
by { 
  -- We skip the proof for now
  -- Add the proof here if needed in later steps
  sorry 
}

end number_of_combinations_of_two_products_l567_567106


namespace vector_equal_norms_l567_567097

variable {V : Type*} [InnerProductSpace ℝ V] (a b : V)

theorem vector_equal_norms
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h_angle : ∠ (a + b) (a - b) = π / 2) :
  ‖a‖ = ‖b‖ :=
by sorry

end vector_equal_norms_l567_567097


namespace arithmetic_sequence_sum_l567_567998

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_sum :
  (a 3 = 5) →
  (a 4 + a 8 = 22) →
  ( ∑ i in range 8, a (i + 1) = 64) :=
by
  intros h1 h2
  sorry

end arithmetic_sequence_sum_l567_567998


namespace ripe_oranges_per_day_l567_567515

theorem ripe_oranges_per_day (R : ℕ) (h : 73 * R = 365) : R = 5 :=
by
  have eq: 73 * 5 = 365 := sorry -- this line is a known fact, or can be calculated directly
  exact eq ▸ h.symm

end ripe_oranges_per_day_l567_567515


namespace problem_statement_l567_567908

open Real

theorem problem_statement (x1 x2 : ℝ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x1 + x2 < exp 1 * x1 * x2) : x1 + x2 > 1 :=
sorry

end problem_statement_l567_567908


namespace smallest_possible_value_of_S_l567_567244

theorem smallest_possible_value_of_S : ∃ (a b c : ℕ × ℕ × ℕ), 
    (∀ (x y z : ℕ × ℕ × ℕ), ((x = (a.1, a.2, a.3) ∨ x = (b.1, b.2, b.3) ∨ x = (c.1, c.2, c.3)) → 
    (x.2 = (x.1 + x.3) / 2 )) ∧ (a.1 ≠ b.1 ∧ a.1 ≠ c.1 ∧ b.1 ≠ c.1 ∧ 
    a.2 ≠ b.2 ∧ a.2 ≠ c.2 ∧ b.2 ≠ c.2 ∧ 
    a.3 ≠ b.3 ∧ a.3 ≠ c.3 ∧ b.3 ≠ c.3) ∧ 
    {a.1, a.2, a.3, b.1, b.2, b.3, c.1, c.2, c.3} = {1, 2, 3, 4, 5, 6, 7, 8, 9})
    → (a.1 * a.2 * a.3 + b.1 * b.2 * b.3 + c.1 * c.2 * c.3 = 270).
Proof
  sorry

end smallest_possible_value_of_S_l567_567244


namespace area_RWP_l567_567221

-- Definitions
variables (X Y Z W P Q R : ℝ × ℝ)
variables (h₁ : (X.1 - Z.1) * (X.1 - Z.1) + (X.2 - Z.2) * (X.2 - Z.2) = 144)
variables (h₂ : P.1 = X.1 - 8 ∧ P.2 = X.2)
variables (h₃ : Q.1 = (Z.1 + P.1) / 2 ∧ Q.2 = (Z.2 + P.2) / 2)
variables (h₄ : R.1 = (Y.1 + P.1) / 2 ∧ R.2 = (Y.2 + P.2) / 2)
variables (h₅ : 1 / 2 * ((Z.1 - X.1) * (W.2 - X.2) - (Z.2 - X.2) * (W.1 - X.1)) = 72)
variables (h₆ : 1 / 2 * abs ((Q.1 - X.1) * (W.2 - X.2) - (Q.2 - X.2) * (W.1 - X.1)) = 20)

-- Theorem statement
theorem area_RWP : 
  1 / 2 * abs ((R.1 - W.1) * (P.2 - W.2) - (R.2 - W.2) * (P.1 - W.1)) = 12 :=
sorry

end area_RWP_l567_567221


namespace ways_to_draw_at_least_two_defective_l567_567837

-- Definitions based on the conditions of the problem
def total_products : ℕ := 100
def defective_products : ℕ := 3
def selected_products : ℕ := 5

-- Binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to prove
theorem ways_to_draw_at_least_two_defective :
  C defective_products 2 * C (total_products - defective_products) 3 + C defective_products 3 * C (total_products - defective_products) 2 =
  (C total_products selected_products - C defective_products 1 * C (total_products - defective_products) 4) :=
sorry

end ways_to_draw_at_least_two_defective_l567_567837


namespace positive_solution_approx_l567_567896

noncomputable def find_positive_solution : ℝ :=
  let y := cbrt 3 in
  let x := y^3 in
  x

theorem positive_solution_approx :
  ∃ x : ℝ, (x ≈ 3.1481) ∧
    (sqrt4 (x + sqrt4 (x + sqrt4 (x + ...))) = sqrt4 (x * sqrt4 (x * sqrt4 (x * ...)))) :=
by
  sorry

end positive_solution_approx_l567_567896


namespace convert_spherical_coords_l567_567997

theorem convert_spherical_coords :
  ∀ (ρ θ φ : ℝ), ρ > 0 → 0 ≤ θ ∧ θ < 2 * π → 0 ≤ φ ∧ φ ≤ π →
    (ρ = 4 ∧ θ = π/4 ∧ φ = 9 * π / 5)  → (ρ = 4 ∧ θ = 5 * π / 4 ∧ φ = π / 5) :=
begin
  intros ρ θ φ hρ hθ hφ hcoords,
  sorry -- proof goes here
end

end convert_spherical_coords_l567_567997


namespace sum_le_n_l567_567794

theorem sum_le_n (n : ℕ) (a : ℕ → ℝ) (h : ∀ (φ : ℝ), (∑ i in finset.range n, a (i + 1) * real.cos ((i + 1) * φ)) ≥ -1) : 
  (∑ i in finset.range n, a (i + 1)) ≤ n :=
by sorry


end sum_le_n_l567_567794


namespace find_a_l567_567943

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x >= 0 then real.sqrt (a * x - 1)
  else -x^2 - 4 * x

theorem find_a (x : ℝ) (a : ℝ) (h : f (f (-2) a) a = 3) : a = 5 / 2 :=
  sorry

end find_a_l567_567943


namespace find_smallest_in_arith_prog_l567_567040

theorem find_smallest_in_arith_prog (a d : ℝ) 
    (h1 : (a - 2 * d)^3 + (a - d)^3 + a^3 + (a + d)^3 + (a + 2 * d)^3 = 0)
    (h2 : (a - 2 * d)^4 + (a - d)^4 + a^4 + (a + d)^4 + (a + 2 * d)^4 = 136) :
    a = -2 * Real.sqrt 2 ∨ a = 2 * Real.sqrt 2 :=
begin
  -- sorry placeholder for proof steps
  sorry
end

end find_smallest_in_arith_prog_l567_567040


namespace sum_of_divisors_143_l567_567734

theorem sum_of_divisors_143 : 
  (finset.range (144)).filter (λ d, 143 % d = 0).sum = 168 :=
by
  sorry

end sum_of_divisors_143_l567_567734


namespace even_number_divisors_reciprocal_sum_l567_567012

theorem even_number_divisors_reciprocal_sum (n : ℕ) (h_even : n % 2 = 0) (h_condition : (∑ d in (finset.filter (λ d, n % d = 0) (finset.range (n+1))) → ℚ) (λ d, (1:ℚ) / d) = 1620 / 1003) : 
  ∃ k : ℕ, n = 2006 * k :=
sorry

end even_number_divisors_reciprocal_sum_l567_567012


namespace dennis_taught_for_34_years_l567_567791

/-- Definitions of years taught by Virginia, Adrienne, and Dennis -/
variable (A V D : ℕ)
variable (h1 : V = A + 9)
variable (h2 : D = V + 9)
variable (h3 : A + V + D = 75)

/-- Proof that Dennis has taught for 34 years -/
theorem dennis_taught_for_34_years : D = 34 := by
  have hv := h1  -- V = A + 9
  have hd := h2  -- D = V + 9 
  rw [hv] at hd  -- D = (A + 9) + 9
  rw [add_assoc] at hd -- D = A + 18
  have h4 : D = A + 18 := hd
  have h5 : A + (A + 9) + (A + 18) = 75 := h3 -- Total years constraint
  rw [add_assoc] at h5 -- A + A + 9 + A + 18 = 75
  rw [add_assoc _ 9] at h5 -- 3A + 27 = 75
  have h6 : 3 * A + 27 = 75 := h5
  have h7 : 3 * A = 48 := by
    sorry
  have h8 : A = 16 := by
    sorry
  rw [h8] at h4 -- D = 16 + 18
  have h9 : D = 34 := by
    sorry
  exact h9

end dennis_taught_for_34_years_l567_567791


namespace sqrt_domain_l567_567273

theorem sqrt_domain (x : ℝ) : 3 * x - 6 ≥ 0 ↔ x ≥ 2 := sorry

end sqrt_domain_l567_567273


namespace largest_integer_with_remainder_l567_567428

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l567_567428


namespace ratio_amyl_alcohol_to_ethanol_l567_567103

noncomputable def mol_amyl_alcohol : ℕ := 3
noncomputable def mol_hcl : ℕ := 3
noncomputable def mol_ethanol : ℕ := 1
noncomputable def mol_h2so4 : ℕ := 1
noncomputable def mol_ch3_cl2_c5_h9 : ℕ := 3
noncomputable def mol_h2o : ℕ := 3
noncomputable def mol_ethyl_dimethylpropyl_sulfate : ℕ := 1

theorem ratio_amyl_alcohol_to_ethanol : 
  (mol_amyl_alcohol / mol_ethanol = 3) :=
by 
  have h1 : mol_amyl_alcohol = 3 := by rfl
  have h2 : mol_ethanol = 1 := by rfl
  sorry

end ratio_amyl_alcohol_to_ethanol_l567_567103


namespace largest_int_less_than_100_remainder_4_l567_567417

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l567_567417


namespace even_function_value_l567_567493

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem even_function_value (h_even : ∀ x, f a b x = f a b (-x))
    (h_domain : a - 1 = -2 * a) :
    f a (0 : ℝ) (1 / 2) = 13 / 12 :=
by
  sorry

end even_function_value_l567_567493


namespace sum_of_divisors_143_l567_567721

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l567_567721


namespace find_polynomial_R_l567_567600

theorem find_polynomial_R (Q R : Polynomial ℂ) :
  (∀ z : ℂ, z ^ 2023 + 1 = (z ^ 2 + z + 1) * Q + R) ∧ R.degree < 2 → R = - Polynomial.X + 1 :=
by
  intro h
  sorry

end find_polynomial_R_l567_567600


namespace probability_divisible_by_5_l567_567817

theorem probability_divisible_by_5 :
  let S := {x | 100 ≤ x ∧ x ≤ 1000}
  let T := {x ∈ S | x % 5 = 0}
  (T.card : ℚ) / S.card = 181 / 901 :=
by
  sorry

end probability_divisible_by_5_l567_567817


namespace find_y_l567_567970

theorem find_y (y : ℚ) (h : 1/3 - 1/4 = 4/y) : y = 48 := sorry

end find_y_l567_567970


namespace sum_of_divisors_of_143_l567_567746

theorem sum_of_divisors_of_143 : 
  (∑ d in (finset.filter (λ d, 143 % d = 0) (finset.range (143 + 1))), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l567_567746


namespace john_tax_rate_l567_567572

theorem john_tax_rate { P: Real → Real → Real → Real → Prop }:
  ∀ (cNikes cBoots totalPaid taxRate: ℝ), 
  cNikes = 150 →
  cBoots = 120 →
  totalPaid = 297 →
  taxRate = ((totalPaid - (cNikes + cBoots)) / (cNikes + cBoots)) * 100 →
  taxRate = 10 :=
by
  intros cNikes cBoots totalPaid taxRate HcNikes HcBoots HtotalPaid HtaxRate
  sorry

end john_tax_rate_l567_567572


namespace mandy_chocolate_l567_567326

theorem mandy_chocolate (total : ℕ) (h1 : total = 60)
  (michael : ℕ) (h2 : michael = total / 2)
  (paige : ℕ) (h3 : paige = (total - michael) / 2) :
  (total - michael - paige = 15) :=
by
  -- By hypothesis: total = 60, michael = 30, paige = 15
  sorry 

end mandy_chocolate_l567_567326


namespace count_ordered_pairs_l567_567518

theorem count_ordered_pairs (M N : ℕ) (hM_pos : M > 0) (hN_pos : N > 0) (h_eq : M * N = 48) : 
  card {p : ℕ × ℕ | (p.1 > 0) ∧ (p.2 > 0) ∧ (p.1 * p.2 = 48)} = 10 :=
sorry

end count_ordered_pairs_l567_567518


namespace robin_camera_pictures_l567_567208

-- Given conditions
def pictures_from_phone : Nat := 35
def num_albums : Nat := 5
def pics_per_album : Nat := 8

-- Calculate total pictures and the number of pictures from the camera
theorem robin_camera_pictures : num_albums * pics_per_album - pictures_from_phone = 5 := by
  sorry

end robin_camera_pictures_l567_567208


namespace kim_candy_bars_saved_l567_567577

theorem kim_candy_bars_saved
  (n : ℕ)
  (c : ℕ)
  (w : ℕ)
  (total_bought : ℕ := n * c)
  (total_eaten : ℕ := n / w)
  (candy_bars_saved : ℕ := total_bought - total_eaten) :
  candy_bars_saved = 28 :=
by
  sorry

end kim_candy_bars_saved_l567_567577


namespace digit_properties_l567_567670

theorem digit_properties {x : ℕ} 
  (h1: 12 * x - 21 * x = 36)   -- Condition: difference between numbers is 36
  (h2: ∃ k, 2 * k = x)        -- Condition: ratio of 1:2 between digits
  (hx : x < 10)               -- Condition: digits have to be less than 10
  : ((10 * x + 2 * x) % 10 + ((2 * 10 * x) % 10)) - ((2 * (10 * x) % 10 ) - x) = 6 := 
begin
  sorry
end

end digit_properties_l567_567670


namespace problem_equivalence_l567_567883

theorem problem_equivalence (x : ℝ) (h1 : x > 6) :
  (sqrt (x - 6 * sqrt (x - 6)) + 3 = sqrt (x + 6 * sqrt (x - 6)) - 3) ↔ (x ≥ 18) :=
by
  sorry

end problem_equivalence_l567_567883


namespace largest_int_less_than_100_remainder_4_l567_567414

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l567_567414


namespace ratio_of_x_to_y_l567_567537

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x^2 - 2 * y^2) / (x^2 + 4 * y^2) = 5 / 7) : 
  x / y = Real.sqrt (17 / 8) :=
by
  sorry

end ratio_of_x_to_y_l567_567537


namespace book_arrangement_count_l567_567522

theorem book_arrangement_count :
  let total_books := 7
  let identical_math_books := 3
  let identical_physics_books := 2
  (Nat.factorial 7) / ((Nat.factorial 3) * (Nat.factorial 2)) = 420 := 
by
  sorry

end book_arrangement_count_l567_567522


namespace telescoping_product_l567_567371

theorem telescoping_product :
  ∏ (n : ℕ) in (finset.range 98).map (λ n, n + 3), (1 - (1 / (n : ℝ))) = (1 / 50) := 
by
  sorry

end telescoping_product_l567_567371


namespace tangent_lines_parallel_l567_567712

-- Definitions and conditions
def curve (x : ℝ) : ℝ := x^3 + x - 2
def line (x : ℝ) : ℝ := 4 * x - 1
def tangent_line_eq (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Proof statement
theorem tangent_lines_parallel (tangent_line : ℝ → ℝ) :
  (∃ x : ℝ, tangent_line_eq 4 (-1) 0 x (curve x)) ∧ 
  (∃ x : ℝ, tangent_line_eq 4 (-1) (-4) x (curve x)) :=
sorry

end tangent_lines_parallel_l567_567712


namespace largest_int_mod_6_less_than_100_l567_567451

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l567_567451


namespace quadrilateral_DFGE_cyclic_l567_567168

variables {ℂ : Type*} [inner_product_space ℝ ℂ]

-- Definitions of the points and the geometric objects involved
structure circle (ℂ : Type*) extends set ℂ 
structure chord (ℂ : Type*) :=
  (circle_point : circle ℂ)

-- Definitions according to given conditions
def midpoint_of_arc (arc : set ℂ) : ℂ := sorry
def intersection_point (c1 c2 : chord ℂ) : ℂ := sorry 

-- Given conditions
variables {C : circle ℂ} {B C : chord ℂ}
variable A : ℂ
variable (AD AE : chord ℂ)
variable (F G : ℂ)

-- Establish intersections and midpoint
hypothesis A_midpoint_bc : A = midpoint_of_arc (B.circle_point ∪ C.circle_point)
hypothesis AD_intersects_BC_at_F : F = intersection_point AD B
hypothesis AE_intersects_BC_at_G : G = intersection_point AE C

-- The quadrilateral in question and the theorem to be proven
def quadrilateral_cyclic (D F G E : ℂ) : Prop := sorry

-- Claim: DFGE is cyclic
theorem quadrilateral_DFGE_cyclic : 
  quadrilateral_cyclic A F G A :=
sorry

end quadrilateral_DFGE_cyclic_l567_567168


namespace tenth_term_of_arithmetic_sequence_l567_567013

-- Define the initial conditions: first term 'a' and the common difference 'd'
def a : ℤ := 2
def d : ℤ := 1 - a

-- Define the n-th term of an arithmetic sequence formula
def nth_term (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

-- Statement to prove
theorem tenth_term_of_arithmetic_sequence :
  nth_term a d 10 = -7 := 
by
  sorry

end tenth_term_of_arithmetic_sequence_l567_567013


namespace inequality_max_k_l567_567032

theorem inequality_max_k (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2 * d)^5) ≥ 174960 * a * b * c * d^3 :=
sorry

end inequality_max_k_l567_567032


namespace number_of_complex_solutions_l567_567014

theorem number_of_complex_solutions:
  {z : ℂ // abs z = 1 ∧ (z ^ (40320) - z ^ (5040)).im = 0}.toFinset.card = 588 := sorry

end number_of_complex_solutions_l567_567014


namespace train_crossing_time_l567_567564

-- Definitions and conditions
def train_length : ℝ := 250  -- length of the train in meters
def train_speed_kmh : ℝ := 200  -- speed of the train in km/hr
def conversion_factor : ℝ := 1000 / 3600  -- conversion factor from km/hr to m/s
def train_speed_ms : ℝ := train_speed_kmh * conversion_factor  -- speed of the train in m/s

-- Theorem statement
theorem train_crossing_time :
  (train_length / train_speed_ms) ≈ 4.5 :=
by
  sorry

end train_crossing_time_l567_567564


namespace sum_of_divisors_of_143_l567_567750

theorem sum_of_divisors_of_143 : 
  (∑ d in (finset.filter (λ d, 143 % d = 0) (finset.range (143 + 1))), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l567_567750


namespace lightest_weight_minimum_l567_567707

theorem lightest_weight_minimum (distinct_masses : ∀ {w : set ℤ}, ∀ (x ∈ w) (y ∈ w), x = y → x = y)
  (lightest_weight_ratio : ∀ {weights : list ℤ} (m : ℤ), m = list.minimum weights →
     sum (list.filter (≠ m) weights) = 71 * m)
  (two_lightest_weights_ratio : ∀ {weights : list ℤ} (n m : ℤ), m ∈ weights → n ∈ weights →
     n + m = list.minimum (m :: list.erase weights m) →
     sum (list.filter (≠ n + m) weights) = 34 * (n + m)) :
  ∃ (m : ℤ), m = 35 := 
sorry

end lightest_weight_minimum_l567_567707


namespace expected_adjacent_red_pairs_l567_567655

open Probability

def num_all_cards : ℕ := 52
def num_red_cards : ℕ := 26

/-- The expected number of pairs of adjacent cards which are both red 
     in a standard 52-card deck dealt out in a circle is 650/51. -/
theorem expected_adjacent_red_pairs : 
  let p_red_right : ℚ := 25 / 51 in
  let expected_pairs := (num_red_cards : ℚ) * p_red_right
  in expected_pairs = 650 / 51 :=
by
  sorry

end expected_adjacent_red_pairs_l567_567655


namespace vector_calculation_l567_567181

section
variables 
  (d : ℝ^3) (e : ℝ^3) (f : ℝ^3)

def d_vector : ℝ^3 := ![2, -3, 4]
def e_vector : ℝ^3 := ![6, 1, -5]
def f_vector : ℝ^3 := ![-1, 7, 2]

theorem vector_calculation :
  (d_vector - e_vector) • ((e_vector - f_vector) × (f_vector - d_vector)) = 0 :=
begin
  sorry
end
end

end vector_calculation_l567_567181


namespace cora_cookies_per_day_l567_567463

theorem cora_cookies_per_day :
  (∀ (day : ℕ), day ∈ (Finset.range 30) →
    ∃ cookies_per_day : ℕ,
    cookies_per_day * 30 = 1620 / 18) →
  cookies_per_day = 3 := by
  sorry

end cora_cookies_per_day_l567_567463


namespace problem_statement_l567_567606

-- Define the "24-pretty" number as given in the problem
def is_24_pretty (n : ℕ) : Prop :=
  Nat.dvd 24 n ∧ (Nat.divisors n).card = 24

-- Sum of all 24-pretty numbers less than 3000
def S : ℕ := ∑ n in (List.finRange 3000).filter is_24_pretty, n

-- The main statement to prove
theorem problem_statement : S / 24 = 219 := by
  sorry

end problem_statement_l567_567606


namespace mean_of_remaining_three_l567_567646

theorem mean_of_remaining_three (a b c : ℝ) (h₁ : (a + b + c + 105) / 4 = 93) : (a + b + c) / 3 = 89 :=
  sorry

end mean_of_remaining_three_l567_567646


namespace three_non_overlapping_lines_intersection_points_l567_567134

open Set

variable {α : Type*} [LinearOrderedField α]

/-- Three non-overlapping lines in the same plane can intersect at 0, 1, 2, or 3 points. -/
theorem three_non_overlapping_lines_intersection_points (L1 L2 L3 : AffineSubspace ℝ ℝ) 
  (h1 : ∀ p1 p2, p1 ∈ L1 ∧ p1 ∈ L2 ∧ p2 ∈ L2 ∧ p2 ∈ L3 → p1 = p2)
  (h2 : ∀ p1 p2, p1 ∈ L2 ∧ p1 ∈ L3 ∧ p2 ∈ L3 ∧ p2 ∈ L1 → p1 = p2)
  (h3 : ∀ p1 p2, p1 ∈ L3 ∧ p1 ∈ L1 ∧ p2 ∈ L1 ∧ p2 ∈ L2 → p1 = p2)
: ∃ n : ℕ, n ∈ ({0, 1, 2, 3}) := 
sorry

end three_non_overlapping_lines_intersection_points_l567_567134


namespace maximum_possible_shortest_piece_length_l567_567320

theorem maximum_possible_shortest_piece_length :
  ∃ (A B C D E : ℝ), A ≤ B ∧ B ≤ C ∧ C ≤ D ∧ D ≤ E ∧ 
  C = 140 ∧ (A + B + C + D + E = 640) ∧ A = 80 :=
by
  sorry

end maximum_possible_shortest_piece_length_l567_567320


namespace sum_of_divisors_143_l567_567753

theorem sum_of_divisors_143 : ∑ d in finset.filter (λ d, 143 % d = 0) (finset.range 144), d = 168 :=
by sorry

end sum_of_divisors_143_l567_567753


namespace problem_statement_l567_567884

theorem problem_statement (x : ℝ) (h : x > 6) : 
  (sqrt (x - 6 * sqrt (x - 6)) + 3 = sqrt (x + 6 * sqrt (x - 6)) - 3) ↔ (x ∈ Set.Ici 18) :=
sorry

end problem_statement_l567_567884


namespace cosine_identity_sum_l567_567900

noncomputable theory

theorem cosine_identity_sum (x : ℝ) :
  (sin x)^2 + (sin (3 * x))^2 + (sin (4 * x))^2 + (sin (5 * x))^2 = 3 →
  (cos x) * (cos (4 * x)) * (cos (7 * x)) = 0 :=
begin
  sorry
end

noncomputable def sum_of_cosine_coefficients (a b c : ℕ) (ha : a = 1) (hb : b = 4) (hc : c = 7) :
  a + b + c = 12 :=
begin
  rw [ha, hb, hc],
  norm_num,
end

end cosine_identity_sum_l567_567900


namespace factorize_difference_of_squares_factorize_cubic_l567_567393

-- Problem 1: Prove that 4x^2 - 36 = 4(x + 3)(x - 3)
theorem factorize_difference_of_squares (x : ℝ) : 4 * x^2 - 36 = 4 * (x + 3) * (x - 3) := 
  sorry

-- Problem 2: Prove that x^3 - 2x^2y + xy^2 = x(x - y)^2
theorem factorize_cubic (x y : ℝ) : x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2 := 
  sorry

end factorize_difference_of_squares_factorize_cubic_l567_567393


namespace root_implies_m_values_l567_567966

theorem root_implies_m_values (m : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (m + 2) * x^2 - 2 * x + m^2 - 2 * m - 6 = 0) →
  (m = 3 ∨ m = -2) :=
by
  sorry

end root_implies_m_values_l567_567966


namespace cos_expression_l567_567934

-- Define the condition for the line l and its relationship
def slope_angle_of_line_l (α : ℝ) : Prop :=
  ∃ l : ℝ, l = 2

-- Given the tangent condition for α
def tan_alpha (α : ℝ) : Prop :=
  Real.tan α = 2

theorem cos_expression (α : ℝ) (h1 : slope_angle_of_line_l α) (h2 : tan_alpha α) :
  Real.cos (2015 * Real.pi / 2 - 2 * α) = -4/5 :=
by sorry

end cos_expression_l567_567934


namespace pressure_relation_l567_567307

-- Definitions from the problem statement
variables (Q Δu A k x P S ΔV V R T T₀ c_v n P₀ V₀ : ℝ)
noncomputable def first_law := Q = Δu + A
noncomputable def Δu_def := Δu = c_v * (T - T₀)
noncomputable def A_def := A = (k * x^2) / 2
noncomputable def spring_relation := k * x = P * S
noncomputable def volume_change := ΔV = S * x
noncomputable def volume_after_expansion := V = (n / (n - 1)) * (S * x)
noncomputable def ideal_gas_law := P * V = R * T
noncomputable def initial_state := P₀ * V₀ = R * T₀
noncomputable def expanded_state := P * (n * V₀) = R * T

-- Theorem to prove the final relation
theorem pressure_relation
  (h1: first_law Q Δu A)
  (h2: Δu_def Δu c_v T T₀)
  (h3: A_def A k x)
  (h4: spring_relation k x P S)
  (h5: volume_change ΔV S x)
  (h6: volume_after_expansion V S x n)
  (h7: ideal_gas_law P V R T)
  (h8: initial_state P₀ V₀ R T₀)
  (h9: expanded_state P R T n V₀)
  : P / P₀ = 1 / (n * (1 + ((n - 1) * R) / (2 * n * c_v))) :=
  sorry

end pressure_relation_l567_567307


namespace minimum_lightest_weight_l567_567702

-- Definitions
def lightest_weight (m : ℕ) : Prop := ∃ n, 72 * m = 35 * n ∧ m % 35 = 0 ∧ m ≥ 35

-- Theorem statement
theorem minimum_lightest_weight : ∃ m, lightest_weight m ∧ m = 35 :=
by
  use 35
  split
  sorry
  exact rfl

end minimum_lightest_weight_l567_567702


namespace max_min_tuples_count_l567_567603

theorem max_min_tuples_count (n : ℕ) : 
  (let A := { i | 1 ≤ i ∧ i ≤ n };
        X Y : finset ℕ 
        in (X ⊆ A ∧ Y ⊆ A) →
           X.nonempty ∧ Y.nonempty →
           X.max' (by sorry) > Y.min' (by sorry) →
           2 ^ 2 * n - (n + 1) * 2^n) := 
by 
  sorry

end max_min_tuples_count_l567_567603


namespace geometric_sequence_ratio_l567_567696

theorem geometric_sequence_ratio (a b : ℕ → ℝ) (A B : ℕ → ℝ)
  (hA9 : A 9 = (a 5) ^ 9)
  (hB9 : B 9 = (b 5) ^ 9)
  (h_ratio : a 5 / b 5 = 2) :
  (A 9 / B 9) = 512 := by
  sorry

end geometric_sequence_ratio_l567_567696


namespace general_formula_for_an_unique_value_of_a_l567_567510

noncomputable def geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
a * q^(n-1)

variables {a b_1 b_2 b_3 q : ℝ}
variables {n : ℕ}

def b_seq (a q : ℝ) : ℕ → ℝ
| 1 => 1 + a
| 2 => 2 + a * q
| 3 => 3 + a * q^2
| (n + 1) => sorry

-- Given conditions
axiom h1 : a > 0
axiom h2 : b_1 - a = 1
axiom h3 : b_2 - a * q = 2
axiom h4 : b_3 - a * q^2 = 3

-- Proving the general formula for the sequence {a_n} if a = 1
theorem general_formula_for_an (a : ℝ) (n : ℕ) : 
  a = 1 → 
  (geometric_seq 1 (2 + sqrt 2) n = (2 + sqrt 2)^(n - 1) ∨ 
   geometric_seq 1 (2 - sqrt 2) n = (2 - sqrt 2)^(n - 1)) :=
sorry

-- Proving the unique value of 'a'
theorem unique_value_of_a : 
  ∃ (a : ℝ), (∀ q, (geometric_seq a q 3 = (b_seq a q 3)) → a = 1/3) :=
sorry

end general_formula_for_an_unique_value_of_a_l567_567510


namespace quadratic_inequality_solution_l567_567529

theorem quadratic_inequality_solution (x : ℝ) (h : x^2 - 9*x + 14 < 0) : 2 < x ∧ x < 7 :=
by
  sorry

end quadratic_inequality_solution_l567_567529


namespace angle_in_quadrant_l567_567922

-- Define the problem statement as a theorem to prove
theorem angle_in_quadrant (α : ℝ) (k : ℤ) 
  (hα : 2 * (k:ℝ) * Real.pi + Real.pi < α ∧ α < 2 * (k:ℝ) * Real.pi + 3 * Real.pi / 2) :
  (k:ℝ) * Real.pi + Real.pi / 2 < α / 2 ∧ α / 2 < (k:ℝ) * Real.pi + 3 * Real.pi / 4 := 
sorry

end angle_in_quadrant_l567_567922


namespace min_value_theorem_l567_567929

noncomputable def min_value_problem (x y : ℝ) (h : 0 < x ∧ 0 < y ∧ exp x = y * log x + y * log y) : ℝ :=
  min (λ z, exp z / z - log y)

theorem min_value_theorem (x y : ℝ) (h : 0 < x ∧ 0 < y ∧ exp x = y * log x + y * log y) :
  min_value_problem x y h = exp 1 - 1 :=
sorry

end min_value_theorem_l567_567929


namespace initial_percent_l567_567779

theorem initial_percent (x : ℝ) :
  (x / 100) * (5 / 100) = 60 / 100 → x = 1200 := 
by 
  sorry

end initial_percent_l567_567779


namespace equal_sum_of_square_distances_from_center_of_chosen_square_l567_567828

noncomputable def choose_square := sorry  -- Placeholder definition for chosen square

def black_cells_on_chessboard (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n / 2) * (n / 2)

def white_cells_on_chessboard (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n / 2) * (n / 2)

theorem equal_sum_of_square_distances_from_center_of_chosen_square :
  ∀ chosen_square, ∑ black_cells_on_chessboard 8 = ∑ white_cells_on_chessboard 8 := 
sorry

end equal_sum_of_square_distances_from_center_of_chosen_square_l567_567828


namespace flower_selection_l567_567337

theorem flower_selection : Nat.choose 10 6 = 210 := 
by
  sorry

end flower_selection_l567_567337


namespace find_a_l567_567398

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end find_a_l567_567398


namespace smallest_positive_period_monotonically_increasing_interval_minimum_m_for_symmetry_l567_567946

noncomputable def f (x : ℝ) (a : ℝ) := sin(2 * x + π / 6) + sin(2 * x - π / 6) - cos(2 * x) + a

theorem smallest_positive_period (a : ℝ) : (∃ T > 0, ∀ x, f x a = f (x + T) a) :=
  sorry

theorem monotonically_increasing_interval (a : ℝ) : 
  ∀ k : ℤ, ∃ (I : set ℝ), (I = set.Icc (↑k * π - π / 6) (↑k * π + π / 3) ∧ 
  ∀ x y ∈ I, x < y → f x a ≤ f y a) :=
  sorry

theorem minimum_m_for_symmetry (a : ℝ) (m > 0) : 
  (∃ k : ℤ, m = ↑k * π / 2 + π / 3) ∧ (2 * m - π / 6 = k * π  + π / 2) → m = π / 3 :=
  sorry

end smallest_positive_period_monotonically_increasing_interval_minimum_m_for_symmetry_l567_567946


namespace bertha_gave_away_balls_l567_567361

def balls_initial := 2
def balls_worn_out := 20 / 10
def balls_lost := 20 / 5
def balls_purchased := (20 / 4) * 3
def balls_after_20_games_without_giveaway := balls_initial - balls_worn_out - balls_lost + balls_purchased
def balls_after_20_games := 10

theorem bertha_gave_away_balls : balls_after_20_games_without_giveaway - balls_after_20_games = 1 := by
  sorry

end bertha_gave_away_balls_l567_567361


namespace three_digit_numbers_without_135_three_digit_numbers_contain_135_at_least_once_l567_567520

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 1 ∧ d ≠ 3 ∧ d ≠ 5

def is_valid_hundreds_digit (d : ℕ) : Prop :=
  d ≠ 0 ∧ is_valid_digit d

def count_valid_numbers : ℕ :=
  let hundreds_choices := {d : ℕ | is_valid_hundreds_digit d}.card
  let tens_and_units_choices := {d : ℕ | is_valid_digit d}.card
  hundreds_choices * tens_and_units_choices * tens_and_units_choices

theorem three_digit_numbers_without_135 :
  count_valid_numbers = 294 :=
by sorry

theorem three_digit_numbers_contain_135_at_least_once :
  900 - count_valid_numbers = 606 :=
by sorry

end three_digit_numbers_without_135_three_digit_numbers_contain_135_at_least_once_l567_567520


namespace max_books_john_can_buy_l567_567571

-- Define the key variables and conditions
def johns_money : ℕ := 3745
def book_cost : ℕ := 285
def sales_tax_rate : ℚ := 0.05

-- Define the total cost per book including tax
def total_cost_per_book : ℝ := book_cost + book_cost * sales_tax_rate

-- Define the inequality problem
theorem max_books_john_can_buy : ∃ (x : ℕ), 300 * x ≤ johns_money ∧ 300 * (x + 1) > johns_money :=
by
  sorry

end max_books_john_can_buy_l567_567571


namespace pow_equation_sum_l567_567974

theorem pow_equation_sum (x y : ℕ) (hx : 2 ^ 11 * 6 ^ 5 = 4 ^ x * 3 ^ y) : x + y = 13 :=
  sorry

end pow_equation_sum_l567_567974


namespace solve_m_l567_567211

theorem solve_m (x y m : ℝ) (h1 : 4 * x + 2 * y = 3 * m) (h2 : 3 * x + y = m + 2) (h3 : y = -x) : m = 1 := 
by {
  sorry
}

end solve_m_l567_567211


namespace a_n_formula_S_n_formula_T_n_formula_l567_567912

noncomputable def a_sequence (n : ℕ) : ℕ := 2 * n
noncomputable def S (n : ℕ) : ℕ := n * (n + 1)
noncomputable def b_sequence (n : ℕ) : ℕ := a_sequence (3 ^ n)
noncomputable def T (n : ℕ) : ℕ := 3^(n + 1) - 3

theorem a_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → a_sequence n = 2 * n :=
sorry

theorem S_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → S n = n * (n + 1) :=
sorry

theorem T_n_formula :
  ∀ {n : ℕ}, a_sequence 5 = 10 ∧ S 15 = 240 → T n = 3^(n + 1) - 3 :=
sorry

end a_n_formula_S_n_formula_T_n_formula_l567_567912


namespace expected_pairs_of_red_in_circle_deck_l567_567650

noncomputable def expected_pairs_of_adjacent_red_cards (deck_size : ℕ) (red_cards : ℕ) : ℚ :=
  let adjacent_probability := (red_cards - 1 : ℚ) / (deck_size - 1)
  in red_cards * adjacent_probability

theorem expected_pairs_of_red_in_circle_deck :
  expected_pairs_of_adjacent_red_cards 52 26 = 650 / 51 :=
by
  sorry

end expected_pairs_of_red_in_circle_deck_l567_567650


namespace quadratic_roots_primes_4_possible_k_l567_567362

theorem quadratic_roots_primes_4_possible_k :
  ∃ k_set: set ℕ, k_set.card = 4 ∧
    (∀ k ∈ k_set, ∃ p q : ℕ, nat.prime p ∧ nat.prime q ∧
      p + q = 58 ∧ p * q = k) :=
by sorry

end quadratic_roots_primes_4_possible_k_l567_567362


namespace count_good_pairs_l567_567862

noncomputable def is_parallel (m1 m2 : ℝ) : Prop :=
  m1 = m2

noncomputable def is_perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1

noncomputable def slope_of_line (a b c : ℝ) (h : b ≠ 0) : ℝ :=
  -a / b

theorem count_good_pairs :
  let line1 := (2, -1, 3) in
  let line2 := (2, -3, 9) in
  let line3 := (4, -1, -2) in
  let line4 := (-8, -2, 6) in
  let line5 := (-16, -4, 12) in
  let lines := [line1, line2, line3, line4, line5] in
  ∃ n : ℕ, n = 3 ∧
    let slopes := lines.map (λ l => slope_of_line l.1 l.2 l.3 sorry) in
    let pairs := (List.product slopes slopes).filter (λ p => p.1 ≠ p.2) in
    let good_pairs := pairs.filter (λ p => is_parallel p.1 p.2 ∨ is_perpendicular p.1 p.2) in
    good_pairs.length = n :=
by
  sorry

end count_good_pairs_l567_567862


namespace sum_of_nonzero_digit_products_l567_567149

theorem sum_of_nonzero_digit_products : 
  (∑ n in Finset.range (10^2009 + 1), (∏ d in (n.digits 10).to_finset.erase 0, d)) = 46^2009 := 
sorry

end sum_of_nonzero_digit_products_l567_567149


namespace sum_of_divisors_of_143_l567_567748

theorem sum_of_divisors_of_143 : 
  (∑ d in (finset.filter (λ d, 143 % d = 0) (finset.range (143 + 1))), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l567_567748


namespace solution_set_l567_567083

open Classical
open Real

def f (x : ℝ) : ℝ :=
if x ≥ 3 then 9 else -x^2 + 6 * x

theorem solution_set (x : ℝ) : (1 < x ∧ x < 3) ↔ f (x^2 - 2 * x) < f (3 * x - 4) :=
by
  sorry

end solution_set_l567_567083


namespace root_implies_quadratic_eq_l567_567968

theorem root_implies_quadratic_eq (m : ℝ) (h : (m + 2) - 2 + m^2 - 2 * m - 6 = 0) : 
  2 * m^2 - m - 6 = 0 :=
sorry

end root_implies_quadratic_eq_l567_567968


namespace largest_int_less_than_100_remainder_4_l567_567413

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l567_567413


namespace angle_CAB_EQ_angle_EAD_l567_567594

variable {A B C D E : Type}

-- Define the angles as variables for the pentagon ABCDE
variable (ABC ADE CEA BDA CAB EAD : ℝ)

-- Given conditions
axiom angle_ABC_EQ_angle_ADE : ABC = ADE
axiom angle_CEA_EQ_angle_BDA : CEA = BDA

-- Prove that angle CAB equals angle EAD
theorem angle_CAB_EQ_angle_EAD : CAB = EAD :=
by
  sorry

end angle_CAB_EQ_angle_EAD_l567_567594


namespace triangle_area_of_line_l567_567409

theorem triangle_area_of_line (x y : ℝ) (h : y = 9 - 3 * x) :
  let A := (1 / 2) * 3 * 9 in
  A = 13.5 :=
by
  sorry

end triangle_area_of_line_l567_567409


namespace area_of_triangle_AOB_l567_567133

-- Define the curve C as ρ = 4sinθ in the polar coordinate system
def polar_curve (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define points A and B with their given polar angles
def A_angle := Real.pi / 6
def B_angle := 5 * Real.pi / 6

-- Define radii from the origin O to points A and B based on the curve C
def OA := polar_curve A_angle
def OB := polar_curve B_angle

-- Define the area of the triangle AOB using the given polar coordinates
theorem area_of_triangle_AOB : 
  (1 / 2) * OA * OB * Real.sin (B_angle - A_angle) = Real.sqrt 3 := 
by
  sorry

end area_of_triangle_AOB_l567_567133


namespace sum_of_divisors_of_143_l567_567764

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (nat.divisors n).toFinset, k

theorem sum_of_divisors_of_143 : sum_divisors 143 = 168 :=
by sorry

end sum_of_divisors_of_143_l567_567764


namespace range_of_a_for_negative_root_l567_567389

theorem range_of_a_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 4^x - 2^(x-1) + a = 0) →
  - (1/2 : ℝ) < a ∧ a ≤ (1/16 : ℝ) :=
by
  sorry

end range_of_a_for_negative_root_l567_567389


namespace stan_water_intake_l567_567636

theorem stan_water_intake :
  let words_per_minute := 85
  let pages := 12
  let words_per_page := 550
  let pages_per_break := 3
  let break_time_minutes := 10
  let water_per_hour_typing := 22
  let water_per_break := 5
  let total_words := pages * words_per_page
  let typing_time_minutes := total_words / words_per_minute
  let breaks := pages / pages_per_break
  let total_break_time_minutes := breaks * break_time_minutes
  let total_time_minutes := typing_time_minutes + total_break_time_minutes
  let total_time_hours := total_time_minutes / 60
  let water_typing := total_time_hours * water_per_hour_typing
  let water_breaks := breaks * water_per_break
  let total_water := water_typing + water_breaks
  in total_water ≈ 63 :=
by
  sorry

end stan_water_intake_l567_567636


namespace sum_of_divisors_of_143_l567_567769

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (nat.divisors n).toFinset, k

theorem sum_of_divisors_of_143 : sum_divisors 143 = 168 :=
by sorry

end sum_of_divisors_of_143_l567_567769


namespace sum_of_divisors_of_143_l567_567747

theorem sum_of_divisors_of_143 : 
  (∑ d in (finset.filter (λ d, 143 % d = 0) (finset.range (143 + 1))), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l567_567747


namespace hexagon_area_correct_l567_567369

open Real

def point := (ℝ × ℝ)

def hexagon_vertices : list point :=
  [(0,0), (2,4), (6,4), (8,0), (6,-4), (2,-4)]

def hexagon_area (vertices : list point) : ℝ :=
  -- Using the Shoelace formula here as a placeholder for an actual formula or method
  -- Implementation is skipped with 'sorry'
  sorry

theorem hexagon_area_correct :
  hexagon_area hexagon_vertices = 32 := 
  sorry

end hexagon_area_correct_l567_567369


namespace probability_classroom_key_l567_567300

theorem probability_classroom_key (total_keys: ℕ) (classroom_keys: ℕ) (other_keys: ℕ) (h1: total_keys = 7) (h2: classroom_keys = 2) (h3: other_keys = 5) (h4: total_keys = classroom_keys + other_keys) :
  (classroom_keys.to_rat / total_keys.to_rat) = (2 / 7 : ℚ) :=
by sorry

end probability_classroom_key_l567_567300


namespace find_a_l567_567395

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end find_a_l567_567395


namespace fish_catch_l567_567151

theorem fish_catch (B : ℕ) (K : ℕ) (hB : B = 5) (hK : K = 2 * B) : B + K = 15 :=
by
  sorry

end fish_catch_l567_567151


namespace largest_int_less_than_100_remainder_4_l567_567412

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l567_567412


namespace find_a_l567_567394

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end find_a_l567_567394


namespace total_amount_paid_l567_567101

def grapes_quantity : ℝ := 8
def grapes_rate : ℝ := 70
def grapes_discount : ℝ := 0.10
def grapes_tax : ℝ := 0.05

def mangoes_quantity : ℝ := 9
def mangoes_rate : ℝ := 50
def mangoes_tax : ℝ := 0.08

theorem total_amount_paid : ℝ :=
  let grapes_cost_before_discount := grapes_quantity * grapes_rate
  let grapes_discount_amount := grapes_cost_before_discount * grapes_discount
  let grapes_cost_after_discount := grapes_cost_before_discount - grapes_discount_amount
  let grapes_tax_amount := grapes_cost_after_discount * grapes_tax
  let total_grapes_cost := grapes_cost_after_discount + grapes_tax_amount

  let mangoes_cost := mangoes_quantity * mangoes_rate
  let mangoes_tax_amount := mangoes_cost * mangoes_tax
  let total_mangoes_cost := mangoes_cost + mangoes_tax_amount

  let total_amount := total_grapes_cost + total_mangoes_cost

  total_amount = 1015.2
  := by sorry

end total_amount_paid_l567_567101


namespace number_of_zeros_in_interval_l567_567086

noncomputable def f (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x < 2 then 1 - abs (2 * x - 3)
else if x ≥ 2 then 1 / 2 * f (1 / 2 * x)
else 0

def y (x : ℝ) : ℝ := 2 * x * f x - 3

theorem number_of_zeros_in_interval : 
  set.countable {x : ℝ | 1 < x ∧ x < 2016 ∧ y x = 0}.to_finset.card = 11 := 
sorry

end number_of_zeros_in_interval_l567_567086


namespace chewbacca_gum_problem_l567_567002

variable (x : ℕ)

theorem chewbacca_gum_problem
  (h1 : ∀ n, n = 25 - 2 * x)
  (h2 : ∀ m, m = 35 + 3 * x)
  (h3 : ∀ p, p = 25 - x)
  (h4 : ∀ q, q = 35 + 5 * x)
  (h5 : h1 25 - 2 * x h2 35 + 3 * x = h3 25 - x h4 35 + 5 * x)
  :
  x = 25 :=
by
  sorry

end chewbacca_gum_problem_l567_567002


namespace evaluate_f_2x_l567_567906

def f (x : ℝ) := x^2 - 1

theorem evaluate_f_2x (x : ℝ) : f(2 * x) = 4 * x^2 - 1 := by
  -- Proof goes here
  sorry

end evaluate_f_2x_l567_567906


namespace common_divisors_count_l567_567519

open Nat

theorem common_divisors_count (h1 : 36 = 2^2 * 3^2) (h2 : 60 = 2^2 * 3 * 5) :
  (finset.card (finset.filter (λ x, 36 % x = 0 ∧ 60 % x = 0) (finset.range 61))) = 6 :=
by
  sorry

end common_divisors_count_l567_567519


namespace linda_broke_51_eggs_l567_567608

noncomputable def number_of_eggs_broken (white brown green remaining evidence: ℕ) : ℕ :=
  let initial := white + brown + green in
  initial - remaining

theorem linda_broke_51_eggs:
  let brown := 20 in
  let white := 3 * brown in
  let green := 6 in
  let remaining := 35 in
  number_of_eggs_broken white brown green remaining = 51 :=
by
  sorry

end linda_broke_51_eggs_l567_567608


namespace division_result_l567_567782

theorem division_result (d q r : ℕ) (h_d : d = 3) (h_q : q = 7) (h_r : r = 2) : d * q + r = 23 :=
by {
  rw [h_d, h_q, h_r],
  norm_num,
}

end division_result_l567_567782


namespace shaded_area_calculation_l567_567861

noncomputable section

-- Definition of the total area of the grid
def total_area (rows columns : ℕ) : ℝ :=
  rows * columns

-- Definition of the area of a right triangle
def triangle_area (base height : ℕ) : ℝ :=
  1 / 2 * base * height

-- Definition of the shaded area in the grid
def shaded_area (total_area triangle_area : ℝ) : ℝ :=
  total_area - triangle_area

-- Theorem stating the shaded area
theorem shaded_area_calculation :
  let rows := 4
  let columns := 13
  let height := 3
  shaded_area (total_area rows columns) (triangle_area columns height) = 32.5 :=
  sorry

end shaded_area_calculation_l567_567861


namespace coloring_problem_l567_567160

theorem coloring_problem (a : ℕ → ℕ) (n t : ℕ) 
  (h1 : ∀ i j, i < j → a i < a j) 
  (h2 : ∀ x : ℤ, ∃ i, 0 < i ∧ i ≤ n ∧ ((x + a (i - 1)) % t) = 0) : 
  n ∣ t :=
by
  sorry

end coloring_problem_l567_567160


namespace sugar_added_l567_567323

theorem sugar_added (initial_volume : ℕ) (water_percent : ℚ) (kola_percent : ℚ)
  (added_water : ℚ) (added_kola : ℚ) (final_sugar_percent : ℚ) :
  initial_volume = 340 ∧ water_percent = 0.8 ∧ kola_percent = 0.06 ∧ 
  added_water = 10 ∧ added_kola = 6.8 ∧ final_sugar_percent = 0.141111111111111112 →
  let initial_water := water_percent * initial_volume,
      initial_kola := kola_percent * initial_volume,
      initial_sugar := initial_volume - initial_water - initial_kola,
      new_water := initial_water + added_water,
      new_kola := initial_kola + added_kola,
      new_total_volume := new_water + new_kola + initial_sugar in
    ∃ x : ℚ, 
      (initial_sugar + x) / (new_total_volume + x) = final_sugar_percent ∧ 
      x ≈ 3.18 :=
by sorry

end sugar_added_l567_567323


namespace odd_function_solution_set_l567_567503

def f (x : ℝ) : ℝ := if x >= 0 then x^2 - 3 * x else -x^2 - 3 * x

theorem odd_function_solution_set {f : ℝ → ℝ}
  (h_odd : ∀ x, f (-x) = - f x) :
  {x : ℝ | f x < 4} = {x : ℝ | x < 4} :=
by sorry

end odd_function_solution_set_l567_567503


namespace five_fourths_of_fifteen_fourths_l567_567027

theorem five_fourths_of_fifteen_fourths :
  (5 / 4) * (15 / 4) = 75 / 16 := by
  sorry

end five_fourths_of_fifteen_fourths_l567_567027


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567433

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567433


namespace sum_of_divisors_of_143_l567_567766

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (nat.divisors n).toFinset, k

theorem sum_of_divisors_of_143 : sum_divisors 143 = 168 :=
by sorry

end sum_of_divisors_of_143_l567_567766


namespace find_m_purely_imaginary_l567_567498

-- Define the complex number z
def complex_z (m : ℝ) : ℂ := (m + complex.i) / (1 + complex.i)

-- Define the condition of z being purely imaginary
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Proof statement
theorem find_m_purely_imaginary : ∀ (m : ℝ), is_purely_imaginary (complex_z m) → m = -1 :=
by
  sorry

end find_m_purely_imaginary_l567_567498


namespace remainder_of_13_pow_13_plus_13_div_14_l567_567295

theorem remainder_of_13_pow_13_plus_13_div_14 : ((13 ^ 13 + 13) % 14) = 12 :=
by
  sorry

end remainder_of_13_pow_13_plus_13_div_14_l567_567295


namespace digit_123_of_16_over_432_l567_567286

-- Define the repeating decimal representation of 16/432
def repeating_decimal : ℤ -> Char :=
  λ n, if n % 3 == 0 then '7' else if n % 3 == 1 then '0' else '3'

-- Main theorem to prove the 123rd digit is 7
theorem digit_123_of_16_over_432 : repeating_decimal 123 = '7' :=
by
  have lemma : repeating_decimal n = ['0', '3', '7'][n % 3],
  sorry

  cases mod_cases : (123 % 3) with
  | zero =>
    simp [repeating_decimal, mod_cases],
    sorry
  | succ zero =>
    contradiction
  | succ (succ zero) =>
    contradiction
  | succ ... =>
    contradiction

end digit_123_of_16_over_432_l567_567286


namespace rachel_picked_2_apples_l567_567625

def apples_picked (initial_apples picked_apples final_apples : ℕ) : Prop :=
  initial_apples - picked_apples = final_apples

theorem rachel_picked_2_apples (initial_apples final_apples : ℕ)
  (h_initial : initial_apples = 9)
  (h_final : final_apples = 7) :
  apples_picked initial_apples 2 final_apples :=
by
  rw [h_initial, h_final]
  sorry

end rachel_picked_2_apples_l567_567625


namespace height_side_relation_l567_567624

-- Definition of a triangle with sides and corresponding heights
variables {α : Type*} [linear_ordered_field α]
variables (A B C H_A H_B : α)

-- Assume heights comparison
axiom heights_comparison : H_A > H_B

-- Define the sides
noncomputable theory
def side_bc : α := (A * B) / H_A
def side_ac : α := (A * C) / H_B

-- The theorem we want to prove
theorem height_side_relation (heights_comparison : H_A > H_B) : side_bc < side_ac :=
sorry

end height_side_relation_l567_567624


namespace factor_is_2_l567_567331

variable (x : ℕ) (f : ℕ)

theorem factor_is_2 (h₁ : x = 36)
                    (h₂ : ((f * (x + 10)) / 2) - 2 = 44) : f = 2 :=
by {
  sorry
}

end factor_is_2_l567_567331


namespace real_part_abs_z_l567_567161

theorem real_part_abs_z (z : Complex)
  (h : ∃ a b c d : ℝ, 
          let p := Polynomial.Coeff 3
          let p := p + Polynomial.Coeff (4 - Complex.i) 2
          let p := p + Polynomial.Coeff (2 + 5 * Complex.i) 1
          let p := p + Polynomial.Coeff z 0
          Polynomial.Root p a
          ∧ Polynomial.Root p (a + b * Complex.i)
          ∧ Polynomial.Root p (a - b * Complex.i)) : 
  abs z.re = 423 :=
by
  sorry

end real_part_abs_z_l567_567161


namespace polynomial_degree_5_value_at_0_l567_567593

theorem polynomial_degree_5_value_at_0 :
  ∃ p : Polynomial ℝ, p.degree = 5 ∧ (∀ n ∈ ({0, 1, 2, 3, 4, 5} : set ℕ), p.eval (2^n) = 1/(2^(n+1))) ∧ p.eval 0 = 1/2 :=
sorry

end polynomial_degree_5_value_at_0_l567_567593


namespace numCounterexamplesCorrect_l567_567893

-- Define a function to calculate the sum of digits of a number
def digitSum (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Predicate to check if a number is prime
def isPrime (n : Nat) : Prop := 
  Nat.Prime n

-- Set definition where the sum of digits must be 5 and all digits are non-zero
def validSet (n : Nat) : Prop :=
  digitSum n = 5 ∧ ∀ d ∈ n.digits 10, d ≠ 0

-- Define the number of counterexamples
def numCounterexamples : Nat := 6

-- The final theorem stating the number of counterexamples
theorem numCounterexamplesCorrect :
  (∃ ns : Finset Nat, 
    (∀ n ∈ ns, validSet n) ∧ 
    (∀ n ∈ ns, ¬ isPrime n) ∧ 
    ns.card = numCounterexamples) :=
sorry

end numCounterexamplesCorrect_l567_567893


namespace small_triangles_count_l567_567814

-- Define the side length of the large triangle
def side_length_large : ℝ := 8

-- Define the side length of the small triangles
def side_length_small : ℝ := 2

-- Define the area of an equilateral triangle given its side length
def equilateral_triangle_area (s : ℝ) : ℝ :=
  (sqrt 3 / 4) * s ^ 2

-- Define the area of the large triangle
def area_large : ℝ := equilateral_triangle_area side_length_large

-- Define the area of a small triangle
def area_small : ℝ := equilateral_triangle_area side_length_small

-- Define the number of small triangles needed to fill the large triangle
def number_of_small_triangles : ℝ := area_large / area_small

-- Prove that the number of small triangles needed is 16
theorem small_triangles_count : number_of_small_triangles = 16 :=
by
  sorry

end small_triangles_count_l567_567814


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567432

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567432


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567419

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567419


namespace sum_of_divisors_143_l567_567719

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l567_567719


namespace abc_value_l567_567525

noncomputable def find_abc (a b c : ℝ) (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) : ℝ :=
  a * b * c

theorem abc_value (a b c : ℝ) (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) :
  a * b * c = 720 :=
by
  -- We skip the proof by providing sorry.
  sorry

end abc_value_l567_567525


namespace largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567434

theorem largest_integer_less_than_100_with_remainder_4_when_divided_by_6 :
  ∃ x : ℤ, x < 100 ∧ (∃ n : ℤ, x = 6 * n + 4) ∧ x = 94 := sorry

end largest_integer_less_than_100_with_remainder_4_when_divided_by_6_l567_567434


namespace mark_age_l567_567194

-- Definitions based on the conditions in the problem
variables (M J P : ℕ)  -- Current ages of Mark, John, and their parents respectively

-- Condition definitions
def condition1 : Prop := J = M - 10
def condition2 : Prop := P = 5 * J
def condition3 : Prop := P - 22 = M

-- The theorem to prove the correct answer
theorem mark_age : condition1 M J ∧ condition2 J P ∧ condition3 P M → M = 18 := by
  sorry

end mark_age_l567_567194


namespace difference_students_guinea_pigs_l567_567875

-- Define the conditions as constants
def students_per_classroom : Nat := 20
def guinea_pigs_per_classroom : Nat := 3
def number_of_classrooms : Nat := 6

-- Calculate the total number of students
def total_students : Nat := students_per_classroom * number_of_classrooms

-- Calculate the total number of guinea pigs
def total_guinea_pigs : Nat := guinea_pigs_per_classroom * number_of_classrooms

-- Define the theorem to prove the equality
theorem difference_students_guinea_pigs :
  total_students - total_guinea_pigs = 102 :=
by
  sorry -- Proof to be filled in

end difference_students_guinea_pigs_l567_567875


namespace proof_problem_l567_567064

open Real

def p : Prop := ∀ x : ℝ, 2^x + 1 / 2^x > 2
def q : Prop := ∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ sin x + cos x = 1 / 2

theorem proof_problem : ¬p ∧ ¬q :=
by
  sorry

end proof_problem_l567_567064


namespace find_coordinates_of_B_l567_567079

-- Definitions for given conditions
def Point := (ℝ × ℝ)
def A : Point := (1, -1)
def B_y1 := -4
def B_y2 := 2

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Main theorem statement
theorem find_coordinates_of_B (B : Point) :
  B.1 = 1 ∧ (B.2 = B_y1 ∨ B.2 = B_y2) ∧ distance A B = 3 ∧ B.1 = A.1 ∧ (B.2 = A.2 + 3 ∨ B.2 = A.2 - 3)
  := sorry

end find_coordinates_of_B_l567_567079


namespace number_of_b_with_log_b_729_is_positive_integer_l567_567959

theorem number_of_b_with_log_b_729_is_positive_integer : 
  {b : ℕ | b > 0 ∧ ∃ n : ℕ, n > 0 ∧ b^n = 729}.card = 4 :=
by sorry

end number_of_b_with_log_b_729_is_positive_integer_l567_567959


namespace probability_spade_then_king_l567_567277

theorem probability_spade_then_king :
  let spades := 13
  let total_cards := 52
  let non_spade_kings := 3
  let kings := 4
  let prob_case1 := (12 / total_cards) * (kings / (total_cards - 1))
  let prob_case2 := (1 / total_cards) * (non_spade_kings / (total_cards - 1))
  prob_case1 + prob_case2 = (17 / 884) :=
by {
  let spades := 13
  let total_cards := 52
  let non_spade_kings := 3
  let kings := 4
  let prob_case1 := (12 / total_cards) * (kings / (total_cards - 1))
  let prob_case2 := (1 / total_cards) * (non_spade_kings / (total_cards - 1))
  have h1 : prob_case1 = 48 / 2652 := sorry,
  have h2 : prob_case2 = 3 / 2652 := sorry,
  calc
    prob_case1 + prob_case2 = (48 / 2652) + (3 / 2652) : by rw [h1, h2]
    ... = 51 / 2652 : by norm_num
    ... = 17 / 884 : by norm_num
}

end probability_spade_then_king_l567_567277


namespace expected_adjacent_red_pairs_l567_567660

theorem expected_adjacent_red_pairs :
  ∃ (E : ℚ), E = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_l567_567660


namespace min_possible_value_M_l567_567022

theorem min_possible_value_M :
  let n := 21
  let max_diff (grid : Fin n × Fin n → ℕ) : ℕ :=
    let row_diffs := Finset.univ.image (λ i, Finset.univ.image (λ j, grid (i, j)).max' - Finset.univ.image (λ j, grid (i, j)).min')
    let col_diffs := Finset.univ.image (λ j, Finset.univ.image (λ i, grid (i, j)).max' - Finset.univ.image (λ i, grid (i, j)).min')
    max (row_diffs ∪ col_diffs)
  ∀ grid : Fin n × Fin n → ℕ,
    (∀ x, grid x ∈ Finset.range (21 * 21 + 1)) →
    M grid = max_diff grid →
    min_val M grid >= 230 := by 
  sorry

end min_possible_value_M_l567_567022


namespace sum_of_divisors_143_l567_567718

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l567_567718


namespace circles_fit_l567_567565

noncomputable def fit_circles_in_rectangle : Prop :=
  ∃ (m n : ℕ) (α : ℝ), (m * n * α * α = 1) ∧ (m * n * α / 2 = 1962)

theorem circles_fit : fit_circles_in_rectangle :=
by sorry

end circles_fit_l567_567565


namespace sum_of_divisors_of_143_l567_567762

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (nat.divisors n).toFinset, k

theorem sum_of_divisors_of_143 : sum_divisors 143 = 168 :=
by sorry

end sum_of_divisors_of_143_l567_567762


namespace find_x_with_cosine_property_l567_567599

theorem find_x_with_cosine_property : 
  ∃ x : ℝ, 0 < x ∧ x < 30 ∧ (Real.cos (3 * x * Real.pi / 180) = Real.cos ((2 * x^2 - x) * Real.pi / 180)) → Int.round x = 1 :=
by
  sorry

end find_x_with_cosine_property_l567_567599


namespace intersection_eq_singleton_l567_567976

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_eq_singleton :
  A ∩ B = {1} :=
sorry

end intersection_eq_singleton_l567_567976


namespace partI_partII_partIII_l567_567534

noncomputable def f (x : ℝ) : ℝ := sorry

variables (x y a : ℝ)

-- Conditions
def domain_condition := ∀ x, x > 0
def increasing_function := ∀ x y, x < y → f(x) < f(y)
def functional_equation := ∀ x y, f(x * y) = f(x) + f(y)

-- Question Part I: Find the value of f(1)
theorem partI (h1 : domain_condition) (h2 : increasing_function) (h3 : functional_equation) : f(1) = 0 := sorry

-- Question Part II: Prove that f(x / y) = f(x) - f(y)
theorem partII (h1 : domain_condition) (h2 : increasing_function) (h3 : functional_equation) : f(x / y) = f(x) - f(y) := sorry

-- Question Part III: Find the range of values for a given f(3) = 1 and f(a) > f(a-1) + 2
theorem partIII (h1 : domain_condition) (h2 : increasing_function) (h3 : functional_equation) (h4 : f(3) = 1) (h5 : f(a) > f(a - 1) + 2) : 1 < a ∧ a < 9 / 8 := sorry

end partI_partII_partIII_l567_567534


namespace simplify_triangle_expression_l567_567050

theorem simplify_triangle_expression (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  |a + b + c| - |a - b - c| - |a + b - c| = a - b + c :=
by
  sorry

end simplify_triangle_expression_l567_567050


namespace sum_of_divisors_of_143_l567_567751

theorem sum_of_divisors_of_143 : 
  (∑ d in (finset.filter (λ d, 143 % d = 0) (finset.range (143 + 1))), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l567_567751


namespace height_of_remaining_cube_l567_567235

def unit_cube := {x : ℝ × ℝ × ℝ // (0 ≤ x.1 ∧ x.1 ≤ 1) ∧
                                (0 ≤ x.2 ∧ x.2 ≤ 1) ∧
                                (0 ≤ x.3 ∧ x.3 ≤ 1)}

def chopped_height (cube : unit_cube) := 1 - (1 / Real.sqrt 3)

theorem height_of_remaining_cube :
  ∀ (c : unit_cube), 
    (chopped_height c) = (2 * Real.sqrt 3 / 3) :=
by 
  intro c 
  sorry

end height_of_remaining_cube_l567_567235


namespace trapezoid_angle_equality_l567_567137

variable {Point : Type}
variable {A B C D K M : Point}
variable {angle : Point → Point → Point → ℝ}

-- Conditions
variable (h1 : ∀ x y z w : Point, x ≠ y → z ≠ w → parallel x y z w → segment_length x y > segment_length z w)
variable (h2 : ∀ x y z w : Point, x ≠ w → point_on_line x y w → ∀ u : Point, point_on_line y z u → equivalent_angles x y z w u)
variable (h3 : angle A D M = angle C B K)

-- Question to prove
theorem trapezoid_angle_equality (h_parallel : AB \parallel CD) (h_ab_gt_cd : AB > CD) (h_angles : angle DAM = angle CBK) : angle DMA = angle CKB :=
sorry

end trapezoid_angle_equality_l567_567137


namespace lightest_weight_minimum_l567_567708

theorem lightest_weight_minimum (distinct_masses : ∀ {w : set ℤ}, ∀ (x ∈ w) (y ∈ w), x = y → x = y)
  (lightest_weight_ratio : ∀ {weights : list ℤ} (m : ℤ), m = list.minimum weights →
     sum (list.filter (≠ m) weights) = 71 * m)
  (two_lightest_weights_ratio : ∀ {weights : list ℤ} (n m : ℤ), m ∈ weights → n ∈ weights →
     n + m = list.minimum (m :: list.erase weights m) →
     sum (list.filter (≠ n + m) weights) = 34 * (n + m)) :
  ∃ (m : ℤ), m = 35 := 
sorry

end lightest_weight_minimum_l567_567708


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567418

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567418


namespace proof_problem_l567_567924

theorem proof_problem :
  ∀ (a b c : ℤ),
    a = 1 →
    b = 0 →
    c = 2 →
    (2 * a + 3 * c) * b = 0 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  simp
  sorry

end proof_problem_l567_567924


namespace expected_adjacent_red_pairs_l567_567653

open Probability

def num_all_cards : ℕ := 52
def num_red_cards : ℕ := 26

/-- The expected number of pairs of adjacent cards which are both red 
     in a standard 52-card deck dealt out in a circle is 650/51. -/
theorem expected_adjacent_red_pairs : 
  let p_red_right : ℚ := 25 / 51 in
  let expected_pairs := (num_red_cards : ℚ) * p_red_right
  in expected_pairs = 650 / 51 :=
by
  sorry

end expected_adjacent_red_pairs_l567_567653


namespace arithmetic_sequence_S_pq_gt_four_l567_567548

theorem arithmetic_sequence_S_pq_gt_four {p q : ℕ} (hpq : p ≠ q) 
    (S_p S_q : ℚ) [hneS_p : S_p = p / q] [hneS_q : S_q = q / p] : 
    ∃ (d : ℚ), let a1 := - (p + q - 1) / 2 * d + (p + q) / (p * q) in
    let S_pq := (p + q) * a1 + (p + q) * (p + q - 1) / 2 * d in
    S_pq > 4 := by
  sorry

end arithmetic_sequence_S_pq_gt_four_l567_567548


namespace lengthOfAB_l567_567470

variable {F₁ F₂ A B : ℝ}
def ellipse : Prop := 
  ∃ x y : ℝ, (x^2 / 25) + (y^2 / 9) = 1

def semiMajorAxis : ℝ := 5

def focusCondition : Prop := 
  F₁^2 + F₂^2 = 25

def passingLine : Prop :=
  ∃ A B : ℝ, ∃ foci: ℝ, foci = F₁ ∧ foci = F₂ ∧
  A ≠ B

def sumOfDistances : Prop := 
  |F₂ - A| + |F₂ - B| = 12

def totalSegment : ℝ := 
  2 * semiMajorAxis

def lengthAB : ℝ := 
  8

theorem lengthOfAB {F₁ F₂ A B : ℝ}
  (h1 : ellipse)
  (h2 : focusCondition)
  (h3 : passingLine)
  (h4 : sumOfDistances)
  : |A - B| = lengthAB := by
  sorry

end lengthOfAB_l567_567470


namespace candies_distribution_l567_567864

/-- 
Daniel has 24 pieces of candy and needs to distribute them equally among his 5 sisters. 
Prove that the least number of pieces he should take away is 4 to achieve this.
-/
theorem candies_distribution (total_candies : ℕ) (num_sisters : ℕ) (candies_per_sister : ℕ) (pieces_to_take_away : ℕ) :
  total_candies = 24 → num_sisters = 5 → candies_per_sister = 4 → pieces_to_take_away = 4 → 
  ((total_candies - pieces_to_take_away) % num_sisters = 0) :=
by {
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
}

end candies_distribution_l567_567864


namespace part1_part2_part3_l567_567499

theorem part1 (x a : ℝ) (h : arctan (x / 2) + arctan (2 - x) = a) (ha : a = π / 4) :
  arccos (x / 2) = 2 * π / 3 ∨ arccos (x / 2) = 0 := sorry

theorem part2 (a : ℝ) :
  (∃ x : ℝ, arctan (x / 2) + arctan (2 - x) = a) ↔ a ∈ Set.Icc (arctan (1 / (- 2 * Real.sqrt 10 - 6))) (arctan (1 / (2 * Real.sqrt 10 - 6))) := sorry

theorem part3 (a α β : ℝ) (h1 : arctan (α / 2) + arctan (2 - α) = a)
  (h2 : arctan (β / 2) + arctan (2 - β) = a)
  (h3 : 5 ≤ α ∧ α ≤ 15)
  (h4 : 5 ≤ β ∧ β ≤ 15)
  (h5 : α ≠ β) :
  α + β ≤ 19 := sorry

end part1_part2_part3_l567_567499


namespace sequence_a_l567_567092

theorem sequence_a (a : ℕ → ℝ) 
  (h1 : a 1 = 2)
  (h2 : a 2 = 1)
  (h3 : ∀ n ≥ 2, a n / a (n + 1) + a n / a (n - 1) = 2) :
  a 12 = 1 / 6 :=
sorry

end sequence_a_l567_567092


namespace tax_refund_l567_567311

-- Definitions based on the problem conditions
def monthly_salary : ℕ := 9000
def treatment_cost : ℕ := 100000
def medication_cost : ℕ := 20000
def tax_rate : ℚ := 0.13

-- Annual salary calculation
def annual_salary := monthly_salary * 12

-- Total spending on treatment and medications
def total_spending := treatment_cost + medication_cost

-- Possible tax refund based on total spending
def possible_tax_refund := total_spending * tax_rate

-- Income tax paid on the annual salary
def income_tax_paid := annual_salary * tax_rate

-- Prove statement that the actual tax refund is equal to income tax paid
theorem tax_refund : income_tax_paid = 14040 := by
  sorry

end tax_refund_l567_567311


namespace minimum_f_value_l567_567780

noncomputable def f (x : ℕ) : ℕ :=
  ∑ i in Finset.range 2017, i.succ * abs (x - i.succ)

theorem minimum_f_value : ∃ x ∈ Finset.range 2017, f x = 801730806 := by
  sorry

end minimum_f_value_l567_567780


namespace geometric_sequence_product_l567_567554

noncomputable def a1 : ℝ := -- first root of 2x^2 + 5x + 1 = 0
noncomputable def a10 : ℝ := -- second root of 2x^2 + 5x + 1 = 0

theorem geometric_sequence_product (a1 a10 : ℝ) (h1 : 2 * a1^2 + 5 * a1 + 1 = 0) (h2 : 2 * a10^2 + 5 * a10 + 1 = 0) :
  let a4 := a1 * (a10 / a1)^(3/9)
      a7 := a1 * (a10 / a1)^(6/9)
  in a4 * a7 = 1/2 :=
by sorry

end geometric_sequence_product_l567_567554


namespace coordinates_of_B_min_BK_l567_567119

-- Define points C, K, A, D, B
def C : ℝ × ℝ := (0, 4)
def K : ℝ × ℝ := (6, 0)

-- Given A = (x, 0)
def A (x : ℝ) : ℝ × ℝ := (x, 0)

-- Midpoint D of AC
def D (x : ℝ) : ℝ × ℝ := (x / 2, 2)

-- Point B after rotating AD 90 degrees clockwise around A
def B (x : ℝ) : ℝ × ℝ := (x + 2, x / 2)

-- Define the distance square between two points
def dist_sq (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Define BK in terms of x
noncomputable def BK (x : ℝ) : ℝ :=
  dist_sq (B x) K

-- Proof statement: Coordinates of B when BK is minimized
theorem coordinates_of_B_min_BK : 
  B (16 / 5) = (26 / 5, 8 / 5) :=
by
  -- We would provide the complete proof here
  sorry

end coordinates_of_B_min_BK_l567_567119


namespace find_ab_l567_567113

theorem find_ab (a b : ℝ) (h1 : a - b = 4) (h2 : a^2 + b^2 = 80) : a * b = 32 :=
sorry

end find_ab_l567_567113


namespace compute_d_l567_567488

theorem compute_d 
  (c d : ℚ) 
  (h1 : is_root (λ x : ℝ, x^3 + c * x^2 + d * x + 15) (3 + real.sqrt 5))
  (h2 : is_rational c)
  (h3 : is_rational d) :
  d = -18.5 := sorry

end compute_d_l567_567488


namespace proof_f_sum_l567_567471

def f : ℝ → ℝ
| x => if x > 0 then 2 * x else f (x + 1)

theorem proof_f_sum :
  f (4 / 3) + f (-4 / 3) = 4 :=
sorry

end proof_f_sum_l567_567471


namespace precise_approximation_K_l567_567984

theorem precise_approximation_K :
  (∀ K : ℝ, |K - 5.72788| ≤ 0.00625 → K.round_to_significant_figures 1 = 5.7) :=
by
  sorry

end precise_approximation_K_l567_567984


namespace hyperbola_area_correct_l567_567550

noncomputable def hyperbola_area_problem : ℝ :=
  let C := {p : ℝ × ℝ | p.1 ^ 2 - (p.2 ^ 2) / 3 = 1}
  let F := (2 : ℝ, 0 : ℝ)
  let l := {p : ℝ × ℝ | p.1 = 2}
  let asymptote1 := {p : ℝ × ℝ | p.2 = sqrt 3 * p.1}
  let asymptote2 := {p : ℝ × ℝ | p.2 = - sqrt 3 * p.1}
  let triangle_vertices := [(2 : ℝ, 2 * sqrt 3), (2 : ℝ, -2 * sqrt 3)]
  let base := 4 * sqrt 3
  let height := 2
  0.5 * base * height

theorem hyperbola_area_correct :
  hyperbola_area_problem = 4 * sqrt 3 :=
by
  sorry

end hyperbola_area_correct_l567_567550


namespace tan_double_angle_l567_567066

-- Given condition
variables {α : Real} 
hypothesis : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3

-- Lean statement to prove
theorem tan_double_angle (α : Real) (h : (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3) : 
  Real.tan (2 * α) = -8 / 15 := 
sorry

end tan_double_angle_l567_567066


namespace areas_equal_l567_567155

-- Let ABCD be a convex quadrilateral
variables {A B C D : Point}

-- Kolya’s choosen points E and F as midpoints of sides AB and CD
def Kolya_E : Point := midpoint A B
def Kolya_F : Point := midpoint C D

-- Vlad’s choosen points E' and F' at 1/3 the length from A along AB and from C along CD
def Vlad_E' : Point := one_third_point A B
def Vlad_F' : Point := one_third_point C D

-- Define midpoints for Kolya
def Kolya_K : Point := midpoint A Kolya_F
def Kolya_L : Point := midpoint D Kolya_E
def Kolya_M : Point := midpoint B Kolya_F
def Kolya_N : Point := midpoint C Kolya_E

-- Define midpoints for Vlad
def Vlad_K' : Point := midpoint A Vlad_F'
def Vlad_L' : Point := midpoint D Vlad_E'
def Vlad_M' : Point := midpoint B Vlad_F'
def Vlad_N' : Point := midpoint C Vlad_E'

-- Assertion that the areas of the quadrilaterals are equal
theorem areas_equal : area (quadrilateral Kolya_K Kolya_L Kolya_M Kolya_N) =
                       area (quadrilateral Vlad_K' Vlad_L' Vlad_M' Vlad_N') :=
sorry

end areas_equal_l567_567155


namespace jerry_age_is_10_l567_567612

-- Define the ages of Mickey and Jerry
def MickeyAge : ℝ := 20
def mickey_eq_jerry (JerryAge : ℝ) : Prop := MickeyAge = 2.5 * JerryAge - 5

theorem jerry_age_is_10 : ∃ JerryAge : ℝ, mickey_eq_jerry JerryAge ∧ JerryAge = 10 :=
by
  -- By solving the equation MickeyAge = 2.5 * JerryAge - 5,
  -- we can find that Jerry's age must be 10.
  use 10
  sorry

end jerry_age_is_10_l567_567612


namespace tangent_line_eq_l567_567672

theorem tangent_line_eq
  (x y : ℝ)
  (h : x^2 + y^2 - 4 * x = 0)
  (P : ℝ × ℝ)
  (hP : P = (1, Real.sqrt 3))
  : x - Real.sqrt 3 * y + 2 = 0 :=
sorry

end tangent_line_eq_l567_567672


namespace solve_triangle_proof_problem_l567_567096

noncomputable def triangle_proof_problem (a b c C : ℝ) :=
  a^2 + b^2 = c^2 + ab ∧
  sqrt 3 * c = 14 * sin C ∧
  a + b = 13 ∧
  C = π / 3 ∧
  c = 7 ∧
  1 / 2 * a * b * sin C = 10 * sqrt 3

theorem solve_triangle_proof_problem :
  ∃ (a b c C : ℝ), triangle_proof_problem a b c C :=
by {
  sorry
}

end solve_triangle_proof_problem_l567_567096


namespace find_a_l567_567399

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end find_a_l567_567399


namespace find_n_mean_l567_567179

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def φ (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ m, Nat.gcd m n = 1).card

def τ (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (λ d, d ∣ n).card

theorem find_n_mean (n : ℕ) 
  (h1 : 0 < n) 
  (h2 : ∃ p : ℕ, is_prime p ∧ (n = p^5 ∨ (∃ q : ℕ, is_prime q ∧ n = p^2 * q))) : 
  (n = (φ n + τ n) / 2 ∨ φ n = (n + τ n) / 2 ∨ τ n = (n + φ n) / 2) :=
sorry

end find_n_mean_l567_567179


namespace all_propositions_incorrect_l567_567082

theorem all_propositions_incorrect :
  ¬ (∀ (P : Prop),
      (P = ∀ (f : ℝ → ℝ), (∀ x, f x = x → f x = inverse f x → x = f x)) ∨
      (P = ∀ (f : ℝ → ℝ), (∀ x, f (1 - x) = f (1 + x) → symmetric f (x, y))) ∨
      (P = ∀ (f : ℝ → ℝ), (odd f → symmetric f (x, a) → period f = 2 * a)) ∨
      (P = ∃ (A B : Set ℕ), (A = {1,2,3}) → (B = {4,5}) → number_of_functions A B = 8)) :=
sorry

end all_propositions_incorrect_l567_567082


namespace range_q_l567_567174

def q (x : ℝ) : ℝ :=
  if (Nat.floor x).prime then x + 3
  else q (Int.gpf (Nat.floor x)) + 2 * (x + 1 - Nat.floor x)

theorem range_q : set.range q = set.Ico 5 18 :=
by
  sorry

end range_q_l567_567174


namespace abs_sum_le_abs_one_plus_mul_l567_567052

theorem abs_sum_le_abs_one_plus_mul {x y : ℝ} (hx : |x| ≤ 1) (hy : |y| ≤ 1) : 
  |x + y| ≤ |1 + x * y| :=
sorry

end abs_sum_le_abs_one_plus_mul_l567_567052


namespace distance_AB_l567_567557

-- Definition of point A in polar coordinates
def A := (3, Real.pi / 3)

-- Definition of point B in polar coordinates
def B := (1, 4 * Real.pi / 3)

-- Definition to convert polar coordinates to Cartesian coordinates
def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

-- Cartesian coordinates of point A
def A_cartesian := polar_to_cartesian A.1 A.2

-- Cartesian coordinates of point B
def B_cartesian := polar_to_cartesian B.1 B.2

-- Definition of the distance formula between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Statement of the theorem
theorem distance_AB : distance A_cartesian B_cartesian = 4 :=
by
  -- this is where the proof would go; we'll mark it as a placeholder
  sorry

end distance_AB_l567_567557


namespace term_10_of_sequence_l567_567935

theorem term_10_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n * (2 * n + 1)) →
  (∀ n, a n = S n - S (n - 1)) →
  a 10 = 39 :=
by
  intros hS ha
  sorry

end term_10_of_sequence_l567_567935


namespace find_alpha_l567_567068

theorem find_alpha (α β : ℝ) (h1 : Real.arctan α = 1/2) (h2 : Real.arctan (α - β) = 1/3)
  (h3 : 0 < α ∧ α < π/2) (h4 : 0 < β ∧ β < π/2) : α = π/4 := by
  sorry

end find_alpha_l567_567068


namespace value_of_v3_is_70_l567_567852

-- Defining the polynomial f(x)
def f (x : ℕ) : ℕ := 2 * x^6 + 5 * x^4 + x^3 + 7 * x^2 + 3 * x + 1

-- Horner's method step calculations using the given conditions
def calculate_v3_using_horners_method (x : ℕ) : ℕ := 
  let v0 := 2 in
  let v1 := v0 * x in
  let v2 := v1 * x + 5 in
  let v3 := v2 * x + 1 in
  v3 

-- Statement that the value of v3 is 70 when x = 3
theorem value_of_v3_is_70 : calculate_v3_using_horners_method 3 = 70 := by
  sorry

end value_of_v3_is_70_l567_567852


namespace sum_of_divisors_143_l567_567728

theorem sum_of_divisors_143 : sigma 143 = 168 := by
  have h : 143 = 11 * 13 := by norm_num
  rw [sigma_mul_prime_prime 11 13]
  norm_num
  sorry

end sum_of_divisors_143_l567_567728


namespace sin_B_l567_567562

variable (a b A : ℝ)
variable (sin : ℝ → ℝ)

axiom sqrt_eq (s : ℝ) : s > 0 → ∃ x, x * x = s

axiom sin_pi_div_four : sin (π / 4) = sqrt 2 / 2

-- Conditions
def a_def : Prop := a = sqrt 6
def b_def : Prop := b = 2
def A_def : Prop := A = π / 4

-- Question to answer
theorem sin_B : a_def ∧ b_def ∧ A_def → ∃ sin_B, sin_B = sqrt 3 / 3 :=
by
  assume h
  sorry

end sin_B_l567_567562


namespace quadratic_solution_interval_l567_567464

noncomputable def quadratic_inequality (z : ℝ) : Prop :=
  z^2 - 56*z + 360 ≤ 0

theorem quadratic_solution_interval :
  {z : ℝ // quadratic_inequality z} = {z : ℝ // 8 ≤ z ∧ z ≤ 45} :=
by
  sorry

end quadratic_solution_interval_l567_567464


namespace terminating_decimals_count_l567_567045

theorem terminating_decimals_count : 
  (Finset.card (Finset.filter (λ n : ℕ, (1 ≤ n ∧ n ≤ 499)) (Finset.range 500))) = 499 :=
by
  sorry

end terminating_decimals_count_l567_567045


namespace symmetry_axis_value_l567_567977

theorem symmetry_axis_value (a : ℝ) :
  (∃ f : ℝ → ℝ, f = (λ x, Real.sin x + a * Real.cos x) ∧
   ∀ x, f (x + π/2) = f (-x + π/2)) → a = 1 :=
by
  sorry

end symmetry_axis_value_l567_567977


namespace matrix_vector_computation_l567_567166

variable (N : Matrix (Fin 2) (Fin 2) ℝ)
variable (a b : Vector (Fin 2) ℝ)

-- Given conditions
axiom h1 : N.mul_vec a = ![2, -3]
axiom h2 : N.mul_vec b = ![4, 5]

-- The proof statement
theorem matrix_vector_computation :
  N.mul_vec (3 • a - 2 • b) = ![-2, -19] :=
  sorry

end matrix_vector_computation_l567_567166


namespace sum_of_digits_of_second_smallest_positive_integer_divisible_by_numbers_less_than_8_l567_567586

theorem sum_of_digits_of_second_smallest_positive_integer_divisible_by_numbers_less_than_8 :
  let N := Nat.lcm 1 2 3 4 5 6 7 in
  Nat.digits 10 (2 * N) = [8, 4, 0] → ([8, 4, 0].sum = 12) :=
by
  assume N
  assume h_dig : Nat.digits 10 (2 * N) = [8, 4, 0]
  show ([8, 4, 0].sum = 12) from sorry

end sum_of_digits_of_second_smallest_positive_integer_divisible_by_numbers_less_than_8_l567_567586


namespace sum_of_divisors_143_l567_567729

theorem sum_of_divisors_143 : sigma 143 = 168 := by
  have h : 143 = 11 * 13 := by norm_num
  rw [sigma_mul_prime_prime 11 13]
  norm_num
  sorry

end sum_of_divisors_143_l567_567729


namespace rain_on_first_day_l567_567690

theorem rain_on_first_day (x : ℝ) (h1 : x >= 0)
  (h2 : (2 * x) + 50 / 100 * (2 * x) = 3 * x) 
  (h3 : 6 * 12 = 72)
  (h4 : 3 * 3 = 9)
  (h5 : x + 2 * x + 3 * x = 6 * x)
  (h6 : 6 * x + 21 - 9 = 72) : x = 10 :=
by 
  -- Proof would go here, but we skip it according to instructions
  sorry

end rain_on_first_day_l567_567690


namespace r_p_q_sum_l567_567927

theorem r_p_q_sum (t p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4)
    (h2 : (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r)
    (h3 : r > 0) (h4 : p > 0) (h5 : q > 0)
    (h6 : Nat.gcd p q = 1) : r + p + q = 5 := 
sorry

end r_p_q_sum_l567_567927


namespace hannah_hours_per_week_l567_567514

theorem hannah_hours_per_week (H : ℕ) :
    let hourly_wage := 30
    let late_deduction := 5
    let times_late := 3
    let total_deduction := times_late * late_deduction
    let actual_pay := 525
    hourly_wage * H - total_deduction = actual_pay -> H = 18 := 
by
  intros
  dsimp only [hourly_wage, late_deduction, times_late, total_deduction, actual_pay]
  rw [mul_comm hourly_wage H, mul_comm times_late late_deduction] at *
  sorry

end hannah_hours_per_week_l567_567514


namespace problem1_problem2_l567_567051

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Assuming given conditions
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 2

-- Problem (1)
theorem problem1 (h : inner a b = cos (real.pi / 3) * (∥a∥ * ∥b∥)) : 
  ∥a + b∥ = real.sqrt 7 :=
sorry

-- Problem (2)
theorem problem2 (h : inner (a + b) a = 0) : 
  real.arccos (inner a b / (∥a∥ * ∥b∥)) = 2 * real.pi / 3 :=
sorry

end problem1_problem2_l567_567051


namespace square_side_length_l567_567175

theorem square_side_length (E A B C D : Point) (s : ℝ) :
  is_square A B C D s →
  E ∈ interior (square A B C D) →
  (∃ E, 
    ∀ P, dist E A + dist E B + dist E C = min (dist P A + dist P B + dist P C)) →
  (min (dist E A + dist E B + dist E C) = sqrt 2 + sqrt 6) →
  s = 2 :=
by sorry

end square_side_length_l567_567175


namespace least_square_difference_l567_567685

theorem least_square_difference (a : Fin 5 → ℝ) (h : ∑ i, a i ^ 2 = 1) :
  ∃ i j : Fin 5, i ≠ j ∧ (a i - a j) ^ 2 ≤ 1 / 10 := 
by
  sorry

end least_square_difference_l567_567685


namespace min_number_of_coins_l567_567798

variable (k : ℕ) 

-- Define the function that calculates the minimum number of coins needed
def min_coins (k : ℕ) : ℕ :=
  k + (k + 3) / 4

theorem min_number_of_coins (h1 : ∀ i : ℕ, i < k → (∃ c : ℕ, c ∈ {10, 15, 20}) ) (h2 : ∀ i : ℕ, i < k → (∃ n : ℕ, 5 * n < 20 ∧ n + ((k + 3) / 4) <= 3 )) : 
  min_coins k = k + (k + 3) / 4 :=
  sorry

end min_number_of_coins_l567_567798


namespace max_min_sum_distance_on_circumcircle_min_sum_distance_anywhere_l567_567074

-- Definitions for the regular n-gon and circumcircle
def regular_ngon (n : ℕ) : ℕ → Complex := λ k, Complex.exp (2 * π * Complex.I * (k / n))

-- The first problem
theorem max_min_sum_distance_on_circumcircle
  (n : ℕ) (P : Complex)
  (h : ∃ θ, P = Complex.exp (θ * Complex.I) ∧ 0 ≤ θ ∧ θ < 2 * π / n ) :
  max (∑ k in range n, Complex.abs (P - (regular_ngon n k))) = 2 / Complex.sin (π / (2 * n))
  ∧ min (∑ k in range n, Complex.abs (P - (regular_ngon n k))) = 2 * Complex.cot (π / (2 * n)) :=
sorry

-- The second problem
theorem min_sum_distance_anywhere (n : ℕ) (P : Complex) :
  (∑ k in range n, Complex.abs (P - (regular_ngon n k))) ≥ n :=
sorry

end max_min_sum_distance_on_circumcircle_min_sum_distance_anywhere_l567_567074


namespace permutation_probability_l567_567589

theorem permutation_probability (S : Finset (List ℕ)) (T : Finset (List ℕ))
  (hS : S = {l | l ∈ permutations [1, 2, 3, 4, 5, 6] })
  (hT : T = S.filter (λ l, l.head? ≠ some 1 ∧ l.head? ≠ some 6)) :
  let prob := (T.filter (λ l, l.nth 2 = some 3)).card / T.card in
  prob = 1 / 5 ∧ 1 + 5 = 6 :=
by
  sorry

end permutation_probability_l567_567589


namespace sum_of_divisors_of_143_l567_567743

theorem sum_of_divisors_of_143 : 
  (∑ d in (finset.filter (λ d, 143 % d = 0) (finset.range (143 + 1))), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l567_567743


namespace example_theorem_l567_567126

-- Definitions of the conditions
def parallel (l1 l2 : Line) : Prop := sorry

def Angle (A B C : Point) : ℝ := sorry

-- Given conditions
def DC_parallel_AB (DC AB : Line) : Prop := parallel DC AB
def DCA_eq_55 (D C A : Point) : Prop := Angle D C A = 55
def ABC_eq_60 (A B C : Point) : Prop := Angle A B C = 60

-- Proof that angle ACB equals 5 degrees given the conditions
theorem example_theorem (D C A B : Point) (DC AB : Line) :
  DC_parallel_AB DC AB →
  DCA_eq_55 D C A →
  ABC_eq_60 A B C →
  Angle A C B = 5 := by
  sorry

end example_theorem_l567_567126


namespace range_of_a_l567_567951

open Real

-- Given condition
def given_condition_false (a : ℝ) : Prop :=
  ¬ ∃ (x0 : ℝ), ∀ (x : ℝ), x + a * x0 + 1 < 0

theorem range_of_a (a : ℝ) : given_condition_false a → -2 ≤ a ∧ a ≤ 2 :=
begin
  sorry
end

end range_of_a_l567_567951


namespace range_of_a_l567_567979

theorem range_of_a (a : ℝ) : 
  (∃ x : ℤ, 2 < (x : ℝ) ∧ (x : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ y : ℤ, 2 < (y : ℝ) ∧ (y : ℝ) ≤ 2 * a - 1) ∧ 
  (∃ z : ℤ, 2 < (z : ℝ) ∧ (z : ℝ) ≤ 2 * a - 1) ∧ 
  (∀ w : ℤ, 2 < (w : ℝ) ∧ (w : ℝ) ≤ 2 * a - 1 → w = 3 ∨ w = 4 ∨ w = 5) :=
  by
    sorry

end range_of_a_l567_567979


namespace tan_alpha_quadrant_fourth_l567_567077

theorem tan_alpha_quadrant_fourth {α : ℝ} (h1 : sin α + cos α = 1 / 5) (h2 : α ∈ Icc (3 * π / 2) (2 * π)) :
  tan α = - 3 / 4 :=
sorry

end tan_alpha_quadrant_fourth_l567_567077


namespace joyce_pencils_given_l567_567020

def original_pencils : ℕ := 51
def total_pencils_after : ℕ := 57

theorem joyce_pencils_given : total_pencils_after - original_pencils = 6 :=
by
  sorry

end joyce_pencils_given_l567_567020


namespace first_spade_second_king_prob_l567_567280

-- Definitions and conditions of the problem
def total_cards := 52
def total_spades := 13
def total_kings := 4
def spades_excluding_king := 12 -- Number of spades excluding the king of spades
def remaining_kings_after_king_spade := 3

-- Calculate probabilities for each case
def first_non_king_spade_prob := spades_excluding_king / total_cards
def second_king_after_non_king_spade_prob := total_kings / (total_cards - 1)
def case1_prob := first_non_king_spade_prob * second_king_after_non_king_spade_prob

def first_king_spade_prob := 1 / total_cards
def second_king_after_king_spade_prob := remaining_kings_after_king_spade / (total_cards - 1)
def case2_prob := first_king_spade_prob * second_king_after_king_spade_prob

def combined_prob := case1_prob + case2_prob

-- The proof statement
theorem first_spade_second_king_prob :
  combined_prob = 1 / total_cards := by
  sorry

end first_spade_second_king_prob_l567_567280


namespace sum_fractions_range_l567_567902

theorem sum_fractions_range (n : ℕ) (h : 0 < n) :
  ∀ m : ℕ, (∃ (a : Fin n → ℕ), (∀ i j, (i < j : Fin n) → a i < a j)
    ∧ m = ∑ i : Fin n, (i + 1) / (a i)) ↔ 1 ≤ m ∧ m ≤ n :=
by sorry

end sum_fractions_range_l567_567902


namespace range_of_a_for_common_tangents_l567_567376

theorem range_of_a_for_common_tangents :
  ∃ (a : ℝ), ∀ (x y : ℝ),
    ((x - 2)^2 + y^2 = 4) ∧ ((x - a)^2 + (y + 3)^2 = 9) →
    (-2 < a) ∧ (a < 6) := by
  sorry

end range_of_a_for_common_tangents_l567_567376


namespace combined_value_after_2_years_l567_567382

def machine_value_after_n_years (present_value : ℝ) (depletion_rate : ℝ) (n : ℕ) : ℝ :=
  present_value * (1 - depletion_rate) ^ n

def machine_A := (800 : ℝ, 0.10)
def machine_B := (1200 : ℝ, 0.08)
def machine_C := (1500 : ℝ, 0.05)
def n := 2

def combined_value := 
  (machine_value_after_n_years (fst machine_A) (snd machine_A) n) +
  (machine_value_after_n_years (fst machine_B) (snd machine_B) n) +
  (machine_value_after_n_years (fst machine_C) (snd machine_C) n)

theorem combined_value_after_2_years :
  combined_value = 3017.43 :=
sorry

end combined_value_after_2_years_l567_567382


namespace determine_N_l567_567248

noncomputable def sequence_t (k : ℕ) (m : ℕ) : ℕ :=
  if k = 1 then 1
  else if k = 2 then m
  else k * (finset.range (k - 1)).sum (λ i, sequence_t (i + 1) m)

noncomputable def sequence_s (k : ℕ) (m : ℕ) : ℕ :=
  (finset.range k).sum (λ i, sequence_t (i + 1) m)

def ends_in_four_9s (n : ℕ) : Prop :=
  (n % 10000) = 9999

theorem determine_N (m : ℕ) (N : ℕ) :
  ends_in_four_9s m → 
  sequence_t 1 m = 1 →
  sequence_t 2 m = m →
  (∀ k ≥ 1, sequence_s k m = (finset.range k).sum (λ i, sequence_t (i + 1) m)) →
  (∀ n ≥ 3, sequence_t n m = n * sequence_s (n - 1) m) →
  sequence_t 30 m = Nat.factorial N →
  N ∈ {50, 51, 52, 53, 54} :=
begin
  intro h1,
  intros, 
  sorry -- place the logic based on the proof steps if providing full proof
end

end determine_N_l567_567248


namespace product_of_five_consecutive_integers_divisible_by_120_l567_567713

theorem product_of_five_consecutive_integers_divisible_by_120 (n : ℤ) : 
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by 
  sorry

end product_of_five_consecutive_integers_divisible_by_120_l567_567713


namespace expected_distinct_abs_values_eq_l567_567597

theorem expected_distinct_abs_values_eq :
  let S := {-100, -99, -98, ..., 99, 100}
      n := 100 in
  ∀ (T : set ℤ), 
    T ⊆ S ∧ T.card = 50 →
    (expected_distinct_absolute_vals T = 8825 / 201) :=
by
  let S := {-100, -99, -98, ..., 99, 100}
  let n := 100
  assume T : set ℤ
  assume hT : T ⊆ S ∧ T.card = 50
  sorry

end expected_distinct_abs_values_eq_l567_567597


namespace original_total_thumbtacks_l567_567609

-- Conditions
def num_cans : ℕ := 3
def num_boards_tested : ℕ := 120
def thumbtacks_per_board : ℕ := 3
def thumbtacks_remaining_per_can : ℕ := 30

-- Question
theorem original_total_thumbtacks :
  (num_cans * num_boards_tested * thumbtacks_per_board) + (num_cans * thumbtacks_remaining_per_can) = 450 :=
sorry

end original_total_thumbtacks_l567_567609


namespace height_of_brick_l567_567328

variable (L W B : ℕ)
variable (Vol_wall Vol_brick : ℕ)

-- Definitions converted from conditions
def length_wall := 26 * 100 -- in cm
def width_wall := 2 * 100 -- in cm
def height_wall := 0.75 * 100 -- in cm
def volume_wall := length_wall * width_wall * height_wall -- in cm^3

def number_of_bricks := 26000

def length_brick := 20 -- in cm
def width_brick := 10 -- in cm
def volume_brick (height_brick : ℕ) := length_brick * width_brick * height_brick -- in cm^3

-- The Lean 4 statement for the proof problem
theorem height_of_brick (H : ℕ) (h_volume_wall : Vol_wall = 39000000) (h_volume_brick : Vol_brick = 200 * H) : 
  ∃ H, number_of_bricks * Vol_brick = Vol_wall -> H = 7.5 :=
by
  sorry

end height_of_brick_l567_567328


namespace abs_value_of_negative_three_l567_567229

noncomputable def abs_value_example : Prop :=
  | -3 | = 3

theorem abs_value_of_negative_three : abs_value_example :=
by
  sorry

end abs_value_of_negative_three_l567_567229


namespace rain_forest_animals_l567_567227

theorem rain_forest_animals (R : ℕ) 
  (h1 : 16 = 3 * R - 5) : R = 7 := 
  by sorry

end rain_forest_animals_l567_567227


namespace functional_equation_solution_l567_567406

theorem functional_equation_solution (f : ℝ+ → ℝ+)
  (H : ∀ (a b c d : ℝ+), a * b * c * d = 1 →
    (f a + f b) * (f c + f d) = (a + b) * (c + d)) :
  (∀ x : ℝ+, f x = x) ∨ (∀ x : ℝ+, f x = 1 / x) :=
sorry

end functional_equation_solution_l567_567406


namespace angle_between_line_and_plane_l567_567511

variable (m n : ℝ^3)
variable (cos_theta : ℝ)
variable (theta : ℝ)

-- Given conditions: m is the direction vector of a line, n is the normal vector of a plane,
-- and cos(theta) = -1/2
axiom condition_1 : cos_theta = -1/2
axiom condition_2 : theta = real.arccos(cos_theta)

-- Prove that the angle between the line and the plane is π/3
theorem angle_between_line_and_plane (m n : ℝ^3)
  (cos_theta_eq : cos_theta = -1/2) :
  θ = real.pi / 3 := 
sorry

end angle_between_line_and_plane_l567_567511


namespace max_value_of_function_l567_567169

theorem max_value_of_function (a : ℝ) (h_a : a > 1) :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 * π ∧
  (∀ y : ℝ, 0 ≤ y ∧ y ≤ 2 * π → cos y ^ 2 + 2 * a * sin y - 1 ≤ cos x ^ 2 + 2 * a * sin x - 1) ∧
  cos x ^ 2 + 2 * a * sin x - 1 = 2 * a - 1 :=
sorry

end max_value_of_function_l567_567169


namespace area_bound_ge_sqrt2_l567_567006

def point := (ℝ × ℝ)

def area_signed (O A B : point) : ℝ := 
  1/2 * abs ((fst A - fst O) * (snd B - snd O) - (fst B - fst O) * (snd A - snd O))

variables {O A1 A2 A3 A4 : point} 

def area_condition (O A1 A2 A3 A4 : point) : Prop :=
  area_signed O A1 A2 ≥ 1 ∧
  area_signed O A1 A3 ≥ 1 ∧
  area_signed O A1 A4 ≥ 1 ∧
  area_signed O A2 A3 ≥ 1 ∧
  area_signed O A2 A4 ≥ 1 ∧
  area_signed O A3 A4 ≥ 1

theorem area_bound_ge_sqrt2 :
  ∀ O A1 A2 A3 A4, area_condition O A1 A2 A3 A4 →
  ∃ (i j ∈ {1, 2, 3, 4}), i ≠ j ∧ area_signed O (list.nth! [A1, A2, A3, A4] (i-1)) (list.nth! [A1, A2, A3, A4] (j-1)) ≥ sqrt 2 :=
begin
  sorry
end

end area_bound_ge_sqrt2_l567_567006


namespace speed_second_hour_l567_567253

variable (v1 v_avg t : ℝ)
variable (v2 : ℝ)

-- Given conditions
def speed_first_hour : v1 = 120 := by sorry
def average_speed : v_avg = 90 := by sorry
def total_duration : t = 2 := by sorry

-- State theorem to prove
theorem speed_second_hour : v2 = 60 :=
by 
  have total_distance : v_avg * t = 180 := by sorry
  have total_distance_eq : v1 + v2 = 180 := by sorry
  calc
    v2 = 180 - v1   := by sorry
    ... = 180 - 120 := by exact sorry
    ... = 60       := by exact sorry

end speed_second_hour_l567_567253


namespace sum_of_divisors_of_143_l567_567761

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (nat.divisors n).toFinset, k

theorem sum_of_divisors_of_143 : sum_divisors 143 = 168 :=
by sorry

end sum_of_divisors_of_143_l567_567761


namespace odd_function_property_find_value_of_function_at_negative_angle_l567_567932

-- Define the function f and its properties
def f (x : ℝ) : ℝ := if x >= 0 then cos x else -cos (-x)

-- Prove that the function f satisfies the given conditions and find f(-π/6)
theorem odd_function_property : 
  ∀ x : ℝ, f (-x) = -f (x) ∧ (∀ x : ℝ, x ≥ 0 → f(x) = cos x) :=
by 
  intros x
  split
  {
    unfold f
    split_ifs
    {
      rw [not_le] at h
      rw [neg_neg, not_not] at h
      rw [Function.comp_apply, cos, neg_mul_eq_neg_mul]
      exact rfl
    }
    {
      unfold f
      rw [not_not] at h
      split_ifs
      all_goals { exact rfl }
    }
  }
  {
    intros x hx
    unfold f
    split_ifs
    exact rfl
  }

theorem find_value_of_function_at_negative_angle : f (-π / 6) = -√3 / 2 :=
by 
  have h := odd_function_property
  specialize h π
  finish

end odd_function_property_find_value_of_function_at_negative_angle_l567_567932


namespace transformation_correct_l567_567692

noncomputable def original_function (x : ℝ) : ℝ := 2^x
noncomputable def transformed_function (x : ℝ) : ℝ := 2^x - 1
noncomputable def log_function (x : ℝ) : ℝ := Real.log x / Real.log 2 + 1

theorem transformation_correct :
  ∀ x : ℝ, transformed_function x = log_function (original_function x) :=
by
  intros x
  rw [transformed_function, log_function, original_function]
  sorry

end transformation_correct_l567_567692


namespace seating_arrangements_l567_567994

open Nat

theorem seating_arrangements (n : ℕ) (J W P : Fin n) : 
  ∀ total_people : n = 10, 
  ∀ (condition_JW_not_adjacent : ∀ i : Fin n, J ≠ i → W ≠ i),
  ∀ (condition_JP_not_adjacent : ∀ i : Fin n, J ≠ i → P ≠ i),
  ∀ (condition_WP_not_adjacent : ∀ i : Fin n, W ≠ i → P ≠ i),
  (∑ s in Finset.univ.filter (λ l => 
    ¬((J + 1 = W) ∨ (J - 1 = W) ∨ 
      (P + 1 = W) ∨ (P - 1 = W) ∨
      (J + 1 = P) ∨ (J - 1 = P))), 1) = 3507840 :=
begin
  sorry
end

end seating_arrangements_l567_567994


namespace range_of_c_l567_567925

noncomputable def p (c : ℝ) : Prop := ∀ x : ℝ, (2 * c - 1) ^ x = (2 * c - 1) ^ x

def q (c : ℝ) : Prop := ∀ x : ℝ, x + |x - 2 * c| > 1

theorem range_of_c (c : ℝ) (h1 : c > 0)
  (h2 : p c ∨ q c) (h3 : ¬ (p c ∧ q c)) : c ≥ 1 :=
sorry

end range_of_c_l567_567925


namespace largest_value_of_c_l567_567891

theorem largest_value_of_c (c : ℝ) (h : ∃ x: ℝ, g x = 2) : c ≤ 33/4 :=
  let g (x : ℝ) := x^2 - 5*x + c
  in sorry

end largest_value_of_c_l567_567891


namespace DK_bisects_BC_l567_567377

noncomputable theory
open BigOperators
open Classical

variables {ABC : Type*} [EuclideanGeometry ABC] (A B C D K : ABC)

theorem DK_bisects_BC 
  (h1 : is_tangent (circumcircle A B C) C D)
  (h2 : is_tangent (circumcircle A C D) A K)
  (h3 : is_tangent (circumcircle A C D) C K)
  : ∃ M : ABC, midpoint M B C ∧ D K M collinear := sorry

end DK_bisects_BC_l567_567377


namespace surface_area_of_equivalent_cube_l567_567819

noncomputable def volume_of_prism (l w h : ℝ) : ℝ := l * w * h

noncomputable def edge_length_of_cube_with_same_volume (v : ℝ) : ℝ := (v)^(1 / 3)

noncomputable def surface_area_of_cube (s : ℝ) : ℝ := 6 * s^2

theorem surface_area_of_equivalent_cube :
    surface_area_of_cube (edge_length_of_cube_with_same_volume (volume_of_prism 10 4 40)) ≈ 832.37 :=
by
  sorry

end surface_area_of_equivalent_cube_l567_567819


namespace hyperbola_equation_l567_567490

variables (x y : ℝ)

def is_hyperbola
  (center focus1 focus2 : Point)
  (p : Point)
  (a² b²: ℝ) :=
  focus1 = (0, -6) ∧ focus2 = (0, 6) ∧
  center = (0, 0) ∧
  p = (2, -5) ∧
  a² + b² = 36 ∧
  a² = 20 ∧ b² = 16

theorem hyperbola_equation :
  is_hyperbola (0, 0) (0, -6) (0, 6) (2, -5) 20 16 →
  (y^2 / 20) - (x^2 / 16) = 1 := by
  sorry

end hyperbola_equation_l567_567490


namespace sum_of_divisors_143_l567_567755

theorem sum_of_divisors_143 : ∑ d in finset.filter (λ d, 143 % d = 0) (finset.range 144), d = 168 :=
by sorry

end sum_of_divisors_143_l567_567755


namespace product_of_roots_l567_567858

theorem product_of_roots (r1 r2 r3 : ℝ) :
  (r1, r2, r3) ∈ {(r1, r2, r3) | 2 * r1^3 - 5 * r1^2 - 10 * r1 + 22 = 0 ∧
                                2 * r2^3 - 5 * r2^2 - 10 * r2 + 22 = 0 ∧
                                2 * r3^3 - 5 * r3^2 - 10 * r3 + 22 = 0} →
  r1 * r2 * r3 = -11 :=
begin
  sorry
end

end product_of_roots_l567_567858


namespace length_ratio_l567_567962

-- Variables and definitions
variable {Q : Type} [metric_space Q] [inner_product_space ℝ Q]
variables (A B C D O P : Q)
variable [circle_centers : circle_center_data Q]

noncomputable def perpendicular_diameters (O : Q) (A B C D : Q) : Prop :=
  circle O (dist O A) ∧ circle O (dist O C) ∧
  dist A O = dist B O ∧ dist C O = dist D O ∧
  dist A B = dist B O + dist O A ∧ dist C D = dist D O + dist O C ∧
  ⟪A - O, B - O⟫ = 0 ∧ ⟪C - O, D - O⟫ = 0

noncomputable def angle_45 (Q P C : Q) [metric_space.angle_center_data Q P C] : Prop :=
  metric_space.angle_center_data.angle Q P C = π / 4

-- The theorem statement needing proof
theorem length_ratio
  (h1 : perpendicular_diameters O A B C D)
  (h2 : P ∈ line_segment_parameter_domain A O)
  (h3 : angle_45 Q P C) :
  dist P Q / dist A Q = √2 / 2 := by sorry
  
end length_ratio_l567_567962


namespace no_possible_values_of_k_l567_567364

theorem no_possible_values_of_k (k : ℤ) :
  (∀ p q : ℤ, p * q = k ∧ p + q = 58 → ¬ (nat.prime p ∧ nat.prime q)) := 
by
  sorry

end no_possible_values_of_k_l567_567364


namespace exist_pair_sum_to_12_l567_567212

theorem exist_pair_sum_to_12 (S : Set ℤ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) (chosen : Set ℤ) (hchosen : chosen ⊆ S) (hsize : chosen.card = 7) :
  ∃x ∈ chosen, ∃y ∈ chosen, x ≠ y ∧ x + y = 12 := 
sorry

end exist_pair_sum_to_12_l567_567212


namespace forty_days_from_tuesday_is_sunday_l567_567284

inductive DayOfWeek
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open DayOfWeek

def addDays (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  let days := [Sunday, Monday, Tuesday, Wednesday, Thursday, Friday, Saturday]
  days[(days.indexOf start + n) % 7]

theorem forty_days_from_tuesday_is_sunday :
  addDays Tuesday 40 = Sunday := 
  sorry

end forty_days_from_tuesday_is_sunday_l567_567284


namespace eq1_solutions_eq2_solutions_eq3_solutions_l567_567219

noncomputable def solution1 := Set { x : ℝ | 2 * (2 * x - 1)^2 = 32 }
noncomputable def solution2 := Set { x : ℝ | -x^2 + 2 * x + 1 = 0 }
noncomputable def solution3 := Set { x : ℝ | (x - 3)^2 + 2 * x * (x - 3) = 0 }

theorem eq1_solutions : solution1 = {2.5, -1.5} := sorry

theorem eq2_solutions : solution2 = {-1 - Real.sqrt 2, -1 + Real.sqrt 2} := sorry

theorem eq3_solutions : solution3 = {3, 1} := sorry

end eq1_solutions_eq2_solutions_eq3_solutions_l567_567219


namespace largest_int_less_than_100_mod_6_eq_4_l567_567444

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l567_567444


namespace find_a_l567_567396

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end find_a_l567_567396


namespace find_GH_to_nearest_tenth_l567_567123

-- Define the conditions
def angle_G : ℝ := 40
def angle_H : ℝ := 90
def side_HI : ℝ := 7

-- Define the mathematical theorem
theorem find_GH_to_nearest_tenth
  (h1 : angle_H = 90)
  (h2 : angle_G = 40)
  (h3 : side_HI = 7) :
  abs ((side_HI / real.tan (angle_G * real.pi / 180)) - 8.3) < 0.1 :=
by
  -- assuming the use of a calculator for trigonometric values and arithmetic computations
  sorry

end find_GH_to_nearest_tenth_l567_567123


namespace evaluate_f_3_minus_f_neg_3_l567_567110

def f (x : ℝ) : ℝ := x^4 + x^2 + 7 * x

theorem evaluate_f_3_minus_f_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end evaluate_f_3_minus_f_neg_3_l567_567110


namespace sum_of_divisors_143_l567_567752

theorem sum_of_divisors_143 : ∑ d in finset.filter (λ d, 143 % d = 0) (finset.range 144), d = 168 :=
by sorry

end sum_of_divisors_143_l567_567752


namespace part1_part2_l567_567080

-- Define the periodic function f
def f (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x ≤ 3 then x^2 + 4 else (x - 3 * real.floor(x / 3))^2 + 4

-- Statement for part (1)
theorem part1 : f 5 + f 7 = 13 :=
sorry

-- Statement for part (2)
theorem part2 (m : ℝ) (h : ∀ x ∈ set.Icc (4 : ℝ) (6 : ℝ), f x = m * x^2) : 4 / 13 ≤ m ∧ m ≤ 13 / 36 :=
sorry

end part1_part2_l567_567080


namespace find_AB_l567_567552

theorem find_AB 
    (AE EC BD : ℝ)
    (h1 : AE = 6)
    (h2 : EC = 12)
    (h3 : BD = 10) :
    ∃ (AB : ℝ), AB = 8 * real.sqrt 3 :=
sorry

end find_AB_l567_567552


namespace smallest_three_digit_solution_l567_567715

noncomputable def smallest_three_digit : ℤ := 103

theorem smallest_three_digit_solution (n : ℤ) (h1 : n ≥ 100) (h2 : n < 1000) (h3 : 60 * n ≡ 180 [MOD 300]) :
  n = smallest_three_digit :=
by
  sorry

end smallest_three_digit_solution_l567_567715


namespace simplify_composite_product_fraction_l567_567017

def first_four_composite_product : ℤ := 4 * 6 * 8 * 9
def next_four_composite_product : ℤ := 10 * 12 * 14 * 15
def expected_fraction_num : ℤ := 12
def expected_fraction_den : ℤ := 175

theorem simplify_composite_product_fraction :
  (first_four_composite_product / next_four_composite_product : ℚ) = (expected_fraction_num / expected_fraction_den) :=
by
  rw [first_four_composite_product, next_four_composite_product]
  norm_num
  sorry

end simplify_composite_product_fraction_l567_567017


namespace sum_of_divisors_143_l567_567759

theorem sum_of_divisors_143 : ∑ d in finset.filter (λ d, 143 % d = 0) (finset.range 144), d = 168 :=
by sorry

end sum_of_divisors_143_l567_567759


namespace maximize_profit_l567_567075

def revenue (x : ℝ) : ℝ := 
if 0 < x ∧ x ≤ 40 then 
  400 - 6 * x 
else 
  8000 / x - 57600 / (x^2)

def profit (x : ℝ) : ℝ := 
if 0 < x ∧ x ≤ 40 then 
  -6 * x^2 + 384 * x - 40 
else 
  -57600 / x - 16 * x + 7960

theorem maximize_profit : 
  profit 60 = 7768 ∧ (∀ x > 0, profit x ≤ profit 60) :=
sorry

end maximize_profit_l567_567075


namespace expected_adjacent_red_pairs_l567_567659

theorem expected_adjacent_red_pairs :
  ∃ (E : ℚ), E = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_l567_567659


namespace six_digit_number_reversed_by_9_l567_567874

-- Hypothetical function to reverse digits of a number
def reverseDigits (n : ℕ) : ℕ := sorry

theorem six_digit_number_reversed_by_9 :
  ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n * 9 = reverseDigits n :=
by
  sorry

end six_digit_number_reversed_by_9_l567_567874


namespace tan_squared_phi_over_2_eq_tan_C_over_2_l567_567991

-- Define the entities and conditions
variables {A B C D : Point}
variables {phi : ℝ} -- phi is the measure of angle ADB

-- Assume the inradii of the triangles ABD and ADC are equal
def has_equal_inradii (ABD ADC : Triangle) : Prop :=
  inradius(ABD) = inradius(ADC)

-- Assume ABC is a right triangle at B
def right_triangle_at_B (ABC : Triangle) : Prop :=
  ∠ B = 90

-- Define the given problem statement as a Lean 4 theorem
theorem tan_squared_phi_over_2_eq_tan_C_over_2 
  (hc : Triangle ABC)
  (h_right : right_triangle_at_B ABC)
  (h_equal_inradii : has_equal_inradii (Triangle ABD) (Triangle ADC)) :
  tan (phi / 2) ^ 2 = tan (angle C / 2) :=
sorry

end tan_squared_phi_over_2_eq_tan_C_over_2_l567_567991


namespace sum_of_divisors_143_l567_567756

theorem sum_of_divisors_143 : ∑ d in finset.filter (λ d, 143 % d = 0) (finset.range 144), d = 168 :=
by sorry

end sum_of_divisors_143_l567_567756


namespace mac_total_loss_l567_567189

-- Definitions based on conditions in part a)
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_quarter : ℝ := 0.25
def dimes_per_quarter : ℕ := 3
def nickels_per_quarter : ℕ := 7
def quarters_traded_dimes : ℕ := 20
def quarters_traded_nickels : ℕ := 20

-- Lean statement for the proof problem
theorem mac_total_loss : (dimes_per_quarter * value_dime * quarters_traded_dimes 
                          + nickels_per_quarter * value_nickel * quarters_traded_nickels
                          - 40 * value_quarter) = 3.00 := 
sorry

end mac_total_loss_l567_567189


namespace numbers_diff_1_2_3_from_26_out_of_100_l567_567257

theorem numbers_diff_1_2_3_from_26_out_of_100 :
  ∀ (cards : List ℕ), (∀ n ∈ cards, n ∈ (List.range 100).map (λ x => x + 1)) →
  cards.length = 26 →
  ∃ (n m ∈ cards), n ≠ m ∧ (abs (n - m) = 1 ∨ abs (n - m) = 2 ∨ abs (n - m) = 3) :=
by
  intros cards h_range h_len
  sorry

end numbers_diff_1_2_3_from_26_out_of_100_l567_567257


namespace problem1_problem2_problem3_l567_567315

-- Definition for Question 1
theorem problem1 (n : ℕ) : (choose n 3) = 7 * (choose n 1) → n = 8 :=
  by sorry

-- Definition for Question 2
theorem problem2 (a : ℝ) (h : a ≠ 0) : (choose 7 5) * a^2 + (choose 7 3) * a^4 = 2 * (choose 7 4) * a^3 → 
  a = 1 + (10).sqrt / 5 ∨ a = 1 - (10).sqrt / 5 :=
  by sorry

-- Definition for Question 3
theorem problem3 (x : ℝ) : (choose 8 4) * (2 * x)^4 * (x^Real.log x)^4 = 1120 → 
  x = 1 ∨ x = 1/10 :=
  by sorry

end problem1_problem2_problem3_l567_567315


namespace determine_values_a_b_l567_567016

theorem determine_values_a_b (a b x : ℝ) (h₁ : x > 1)
  (h₂ : 3 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = (10 * (Real.log x)^2) / (Real.log a + Real.log b)) :
  b = a ^ ((5 + Real.sqrt 10) / 3) ∨ b = a ^ ((5 - Real.sqrt 10) / 3) :=
by sorry

end determine_values_a_b_l567_567016


namespace triangle_eum_similar_to_triangle_efa_l567_567003

open EuclideanGeometry

theorem triangle_eum_similar_to_triangle_efa
  {E F B C M U A : Point}
  (h1 : PerpendicularBisector EF BC)
  (h2 : Intersects EF BC M)
  (h3 : Between B U M)
  (h4 : ExtendedMeetsCircle EU A) :
  Similar (Triangle E U M) (Triangle E F A) :=
by
  sorry

end triangle_eum_similar_to_triangle_efa_l567_567003


namespace trig_identity_l567_567373

noncomputable def verifyTrigExpression : ℝ :=
  let tangentThirty := sqrt 3 / 3
  let cosineSixty := 1 / 2
  let sineFortyFive := sqrt 2 / 2
  sqrt 3 * tangentThirty + 2 * cosineSixty - sqrt 2 * sineFortyFive

theorem trig_identity : verifyTrigExpression = 1 := by
  sorry

end trig_identity_l567_567373


namespace find_n_value_l567_567640

theorem find_n_value : (15 * 25 + 20 * 5) = (10 * 25 + 45 * 5) := 
  sorry

end find_n_value_l567_567640


namespace sum_of_divisors_143_l567_567774

theorem sum_of_divisors_143 : ∑ d in List.toFinset (List.divisors 143), d = 168 := by
  have prime_factors_143 : 143 = 11 * 13 := by
    calc 143 = 11 * 13 : by norm_num
  sorry

end sum_of_divisors_143_l567_567774


namespace polynomial_roots_arithmetic_progression_complex_conjugates_l567_567024

open Complex

theorem polynomial_roots_arithmetic_progression_complex_conjugates (b : ℝ) :
  (∃ (s₁ s₂ s₃ : ℂ), (s₁ + s₂ + s₃ = 9) ∧ (s₁ * s₂ + s₁ * s₃ + s₂ * s₃ = 27) ∧ 
   (s₁ * s₂ * s₃ = -(b)) ∧ (s₂ - s₁ = s₃ - s₂) ∧ (conj s₁ = s₃) ∧ (s₁ ≠ s₂ ∨ s₃ ≠ s₂)) → b = -18 :=
by
  sorry

end polynomial_roots_arithmetic_progression_complex_conjugates_l567_567024


namespace length_of_real_axis_l567_567494

def asymptote_equation_1 (x y : ℝ) : Prop := y = 2 * x
def asymptote_equation_2 (x y : ℝ) : Prop := y = -2 * x
def line_1 (x y : ℝ) : Prop := x + y = 3
def line_2 (x y : ℝ) : Prop := 2 * x - y = -6
def intersection_point (x y : ℝ) : Prop := line_1 x y ∧ line_2 x y

theorem length_of_real_axis :
  (∃ (x y : ℝ), intersection_point x y) →
  ∀ (a : ℝ), (asymptote_equation_1 a _ ∧ asymptote_equation_2 a _) →
  ∃ (length : ℝ), length = 4 * real.sqrt 3 :=
by
  sorry

end length_of_real_axis_l567_567494


namespace jack_keeps_10800_pounds_l567_567567

def number_of_months_in_a_quarter := 12 / 4
def monthly_hunting_trips := 6
def total_hunting_trips := monthly_hunting_trips * number_of_months_in_a_quarter
def deers_per_trip := 2
def total_deers := total_hunting_trips * deers_per_trip
def weight_per_deer := 600
def total_weight := total_deers * weight_per_deer
def kept_weight_fraction := 1 / 2
def kept_weight := total_weight * kept_weight_fraction

theorem jack_keeps_10800_pounds :
  kept_weight = 10800 :=
by
  -- This is a stub for the automated proof
  sorry

end jack_keeps_10800_pounds_l567_567567


namespace first_spade_second_king_prob_l567_567279

-- Definitions and conditions of the problem
def total_cards := 52
def total_spades := 13
def total_kings := 4
def spades_excluding_king := 12 -- Number of spades excluding the king of spades
def remaining_kings_after_king_spade := 3

-- Calculate probabilities for each case
def first_non_king_spade_prob := spades_excluding_king / total_cards
def second_king_after_non_king_spade_prob := total_kings / (total_cards - 1)
def case1_prob := first_non_king_spade_prob * second_king_after_non_king_spade_prob

def first_king_spade_prob := 1 / total_cards
def second_king_after_king_spade_prob := remaining_kings_after_king_spade / (total_cards - 1)
def case2_prob := first_king_spade_prob * second_king_after_king_spade_prob

def combined_prob := case1_prob + case2_prob

-- The proof statement
theorem first_spade_second_king_prob :
  combined_prob = 1 / total_cards := by
  sorry

end first_spade_second_king_prob_l567_567279


namespace digit_problem_l567_567667

variable {x y : ℕ}

theorem digit_problem (h1 : 10 * x + y - (10 * y + x) = 36) (h2 : x * 2 = y) :
  (x + y) - (x - y) = 16 :=
by sorry

end digit_problem_l567_567667


namespace digit_problem_l567_567668

variable {x y : ℕ}

theorem digit_problem (h1 : 10 * x + y - (10 * y + x) = 36) (h2 : x * 2 = y) :
  (x + y) - (x - y) = 16 :=
by sorry

end digit_problem_l567_567668


namespace find_number_of_valid_tuples_l567_567461

-- Define conditions for the problem
def valid_tuple (k1 k2 k3 k4 : ℕ) : Prop :=
  0 ≤ k1 ∧ k1 ≤ 20 ∧
  0 ≤ k2 ∧ k2 ≤ 20 ∧
  0 ≤ k3 ∧ k3 ≤ 20 ∧
  0 ≤ k4 ∧ k4 ≤ 20 ∧
  k1 + k3 = k2 + k4

-- Define the statement to prove
theorem find_number_of_valid_tuples :
  (finset.univ.filter (λ (t : ℕ × ℕ × ℕ × ℕ),
    let (k1, k2, k3, k4) := t in valid_tuple k1 k2 k3 k4)).card = 6181 :=
sorry

end find_number_of_valid_tuples_l567_567461


namespace town_population_growth_l567_567680

noncomputable def populationAfterYears (population : ℝ) (year1Increase : ℝ) (year2Increase : ℝ) : ℝ :=
  let populationAfterFirstYear := population * (1 + year1Increase)
  let populationAfterSecondYear := populationAfterFirstYear * (1 + year2Increase)
  populationAfterSecondYear

theorem town_population_growth :
  ∀ (initialPopulation : ℝ) (year1Increase : ℝ) (year2Increase : ℝ),
    initialPopulation = 1000 → year1Increase = 0.10 → year2Increase = 0.20 →
      populationAfterYears initialPopulation year1Increase year2Increase = 1320 :=
by
  intros initialPopulation year1Increase year2Increase h1 h2 h3
  rw [h1, h2, h3]
  have h4 : populationAfterYears 1000 0.10 0.20 = 1320 := sorry
  exact h4

end town_population_growth_l567_567680


namespace area_PQR_is_3_l567_567693

def point := (ℝ × ℝ)

noncomputable def P : point := (4, 2)
noncomputable def Q : point := (2, 5)
noncomputable def on_line_R (R : point) : Prop := R.1 + R.2 = 10

def area_triangle (A B C : point) : ℝ :=
  0.5 * ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))).abs

theorem area_PQR_is_3 {R : point} (h : on_line_R R) : area_triangle P Q R = 3 := 
  sorry

end area_PQR_is_3_l567_567693


namespace mac_loses_l567_567191

def dime_value := 0.10
def nickel_value := 0.05
def quarter_value := 0.25

def dimes_per_quarter := 3
def nickels_per_quarter := 7

def num_quarters_with_dimes := 20
def num_quarters_with_nickels := 20

def total_quarters := num_quarters_with_dimes + num_quarters_with_nickels

def value_of_quarters_received := total_quarters * quarter_value
def value_of_dimes_traded := num_quarters_with_dimes * dimes_per_quarter * dime_value
def value_of_nickels_traded := num_quarters_with_nickels * nickels_per_quarter * nickel_value

def total_value_traded := value_of_dimes_traded + value_of_nickels_traded

theorem mac_loses
  : total_value_traded - value_of_quarters_received = 3.00 :=
by sorry

end mac_loses_l567_567191


namespace planes_have_common_point_center_circumsphere_implies_regular_l567_567182

variables {A B C D A' B' C' D' P I : Type} -- Points in space
variables [Cirumcenters A B C D A' B' C' D'] -- circumcenters of corresponding triangles

def Plane (p : Type) : Type := {point : Type // point ≠ ∅}

variables (P_A P_B P_C P_D : Type) -- Planes in space

-- Conditions
axiom circumcenter_A' : Circumcenter A B C D A'
axiom circumcenter_B' : Circumcenter A B C D B'
axiom circumcenter_C' : Circumcenter A B C D C'
axiom circumcenter_D' : Circumcenter A B C D D'

axiom plane_P_A : Plane P_A
axiom plane_P_B : Plane P_B
axiom plane_P_C : Plane P_C
axiom plane_P_D : Plane P_D

axiom plane_P_A_through_A : passes_through plane_P_A A
axiom plane_P_B_through_B : passes_through plane_P_B B
axiom plane_P_C_through_C : passes_through plane_P_C C
axiom plane_P_D_through_D : passes_through plane_P_D D

axiom plane_P_A_perpendicular_C'D' : perpendicular plane_P_A (line_from C' D')
axiom plane_P_B_perpendicular_D'A' : perpendicular plane_P_B (line_from D' A')
axiom plane_P_C_perpendicular_A'B' : perpendicular plane_P_C (line_from A' B')
axiom plane_P_D_perpendicular_B'C' : perpendicular plane_P_D (line_from B' C')

-- Proof 1: Prove these planes have a common point I
theorem planes_have_common_point :
  ∃ I : Type, passes_through plane_P_A I ∧ passes_through plane_P_B I ∧ passes_through plane_P_C I ∧ passes_through plane_P_D I :=
sorry

-- Proof 2: If P is the center of the circumsphere, must the tetrahedron be regular?
theorem center_circumsphere_implies_regular (P: Type) (circumcenter : Center P A B C D):
 P = I → RegularTetrahedron A B C D :=
sorry

end planes_have_common_point_center_circumsphere_implies_regular_l567_567182


namespace johns_cost_per_sheet_equals_2_75_l567_567832

variable (J : ℝ)

-- Conditions
def johns_total_cost (n : ℕ) : ℝ := n * J + 125
def sams_total_cost (n : ℕ) : ℝ := n * 1.5 + 140

-- Problem statement
theorem johns_cost_per_sheet_equals_2_75 :
  (johns_total_cost 12 = sams_total_cost 12) → J = 2.75 := by
  sorry

end johns_cost_per_sheet_equals_2_75_l567_567832


namespace dot_product_AB_BC_l567_567563

theorem dot_product_AB_BC 
  (a b c : ℝ) 
  (h1 : b^2 = a * c)
  (h2 : a + c = 3)
  (cosB : ℝ)
  (h3 : cosB = 3 / 4) : 
  (a * c * (-cosB) = -3/2) :=
by 
  -- Given conditions
  sorry

end dot_product_AB_BC_l567_567563


namespace five_letter_words_count_l567_567388

/-- Number of five-letter words where the first and last letters are the same vowel from {a, e, i, o, u}, 
and the remaining three letters can be any letters from the alphabet. -/
theorem five_letter_words_count : 
  ∃ n : ℕ, n = 5 * 26 ^ 3 ∧ n = 87880 :=
by
  use 5 * 26 ^ 3
  split
  . rfl
  . norm_num

end five_letter_words_count_l567_567388


namespace triangle_is_isosceles_l567_567139

theorem triangle_is_isosceles 
  (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) 
  (h_sin_identity : Real.sin A = 2 * Real.sin C * Real.cos B) : 
  (B = C) :=
sorry

end triangle_is_isosceles_l567_567139


namespace sum_of_divisors_143_l567_567737

theorem sum_of_divisors_143 : 
  (finset.range (144)).filter (λ d, 143 % d = 0).sum = 168 :=
by
  sorry

end sum_of_divisors_143_l567_567737


namespace geom_seq_sum_first_four_terms_l567_567555

noncomputable def sum_first_n_terms_geom (a₁ q: ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geom_seq_sum_first_four_terms
  (a₁ : ℕ) (q : ℕ) (h₁ : a₁ = 1) (h₂ : a₁ * q^3 = 27) :
  sum_first_n_terms_geom a₁ q 4 = 40 :=
by
  sorry

end geom_seq_sum_first_four_terms_l567_567555


namespace probability_of_two_even_balls_l567_567801

theorem probability_of_two_even_balls
  (total_balls : ℕ) (even_balls : ℕ)
  (h_total_balls : total_balls = 16) (h_even_balls : even_balls = 8) :
  let P_even_first := (even_balls : ℚ) / (total_balls : ℚ),
      P_even_second := (even_balls - 1 : ℚ) / (total_balls - 1 : ℚ),
      P_both_even := P_even_first * P_even_second
  in P_both_even = 7 / 30 := by
{
  sorry
}

end probability_of_two_even_balls_l567_567801


namespace min_value_expression_l567_567178

theorem min_value_expression (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_sum : a + b + c = 12) : 
    (min (frac (4 * c) (a^3 + b^3) + frac (4 * a) (b^3 + c^3) + frac b (a^3 + c^3)) = (9/32)) :=
by
  sorry

end min_value_expression_l567_567178


namespace find_a_l567_567078

variable {x y a : ℝ}

theorem find_a (h1 : 2 * x - y + a ≥ 0) (h2 : 3 * x + y ≤ 3) (h3 : ∀ (x y : ℝ), 4 * x + 3 * y ≤ 8) : a = 2 := 
sorry

end find_a_l567_567078


namespace mac_total_loss_l567_567188

-- Definitions based on conditions in part a)
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_quarter : ℝ := 0.25
def dimes_per_quarter : ℕ := 3
def nickels_per_quarter : ℕ := 7
def quarters_traded_dimes : ℕ := 20
def quarters_traded_nickels : ℕ := 20

-- Lean statement for the proof problem
theorem mac_total_loss : (dimes_per_quarter * value_dime * quarters_traded_dimes 
                          + nickels_per_quarter * value_nickel * quarters_traded_nickels
                          - 40 * value_quarter) = 3.00 := 
sorry

end mac_total_loss_l567_567188


namespace A_eq_2C_not_a2_minus_c2_eq_2bc_min_value_expr_range_a_div_c_l567_567561

variables {A B C a b c : ℝ} (h_condition : a / c = (1 + cos A) / cos C) 

-- 1. Proving A = 2C
theorem A_eq_2C (h_triangle : A + B + C = π) (h_condition : a / c = (1 + cos A) / cos C) : A = 2 * C :=
sorry

-- 2. Negating a² - c² = 2bc
theorem not_a2_minus_c2_eq_2bc (h_law_of_cosines : a^2 = b^2 + c^2 - 2 * b * c * cos A) (h_condition : a / c = (1 + cos A) / cos C) : a^2 - c^2 ≠ 2 * b * c :=
sorry

-- 3. Minimum value of 1/tan(C) - 1/tan(A) + 2sin(A) is 2√2
theorem min_value_expr (h_condition : a / c = (1 + cos A) / cos C) : 
  ∃ x, (1 / tan C - 1 / tan A + 2 * sin A = 2 * sqrt 2) :=
sorry

-- 4. Proving range of a/c is not (0, 2)
theorem range_a_div_c (h_triangle : A + B + C = π) (h_condition : a / c = (1 + cos A) / cos C) : ¬(a / c > 0 ∧ a / c < 2) :=
sorry

end A_eq_2C_not_a2_minus_c2_eq_2bc_min_value_expr_range_a_div_c_l567_567561


namespace compute_d_l567_567486

noncomputable def polynomial := Polynomial

theorem compute_d :
  ∀ (c d : ℚ), 
  (polynomial.eval (3 + Real.sqrt 5) (Polynomial.mk [15, d, c, 1]) = 0) →
  (polynomial.eval (3 - Real.sqrt 5) (Polynomial.mk [15, d, c, 1]) = 0) →
  d = -18.5 :=
by
  sorry

end compute_d_l567_567486


namespace milk_water_ratio_l567_567120

theorem milk_water_ratio (total_volume : ℕ) (initial_milk_ratio : ℕ) (initial_water_ratio : ℕ) (added_water : ℕ)
  (h₁ : total_volume = 45) (h₂ : initial_milk_ratio = 4) (h₃ : initial_water_ratio = 1) (h₄ : added_water = 9) :
  (36 : ℕ) / (18 : ℕ) = 2 :=
by sorry

end milk_water_ratio_l567_567120


namespace largest_integer_less_100_leaves_remainder_4_l567_567455

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l567_567455


namespace probability_cos_l567_567816

noncomputable def probability_cos_interval : Prop :=
  let interval := set.Icc (-1 : ℝ) (1 : ℝ)
  let b_2 := (real.sqrt 2) / 2
  let range := set.Icc b_2 (1 : ℝ)
  (set.measure (interval ∩ {x | cos (real.pi * x / 2) ∈ range}) /
   set.measure interval) = 1 / 2

theorem probability_cos (x : ℝ) (h : x ∈ set.Icc (-1 : ℝ) (1 : ℝ)) :
  probability_cos_interval :=
sorry

end probability_cos_l567_567816


namespace cyclic_powers_of_i_sum_l567_567370

theorem cyclic_powers_of_i_sum :
  (∑ k in (finset.range 2003), (k+2)*complex.i^(k+1)) + 1000 = -1000 - 1004*complex.i :=
by
  sorry

end cyclic_powers_of_i_sum_l567_567370


namespace find_vertical_shift_l567_567847

theorem find_vertical_shift
  (a b c d : ℝ)
  (h : ∀ x : ℝ, a * sin (b * x + c) + d ≤ 3 ∧ a * sin (b * x + c) + d ≥ -1) :
  d = 1 :=
by
  sorry

end find_vertical_shift_l567_567847


namespace positive_integers_square_less_than_three_times_l567_567289

theorem positive_integers_square_less_than_three_times (n : ℕ) (hn : 0 < n) (ineq : n^2 < 3 * n) : n = 1 ∨ n = 2 :=
by sorry

end positive_integers_square_less_than_three_times_l567_567289


namespace num_factors_of_M_l567_567104

theorem num_factors_of_M (M : ℕ) (hM : M = 2^5 * 3^2 * 7^3 * 11^1) : 
  ∃ n, n = 144 ∧ n = (finset.range 6).card * (finset.range 3).card * (finset.range 4).card * (finset.range 2).card := 
  by
    sorry

end num_factors_of_M_l567_567104


namespace ratio_of_chord_segments_l567_567695

theorem ratio_of_chord_segments (EQ FQ GQ HQ : ℝ) 
  (h1 : EQ = 5) 
  (h2 : GQ = 12) 
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 12 / 5 := 
by 
  rw [h1, h2] at h3 -- Substitute EQ and GQ values into the equation
  -- Algebraically solve for FQ / HQ
  have key : 5 * FQ = 12 * HQ := by exact h3
  have ratio := (eq_div_iff_mul_eq.mpr (key.symm)) -- Rearrange to find the ratio
  exact ratio

end ratio_of_chord_segments_l567_567695


namespace max_value_of_f_on_interval_l567_567482

noncomputable def f (x : ℝ) : ℝ := (3 + 2*x)^3 * (4 - x)^4

theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), (-3/2 < x ∧ x < 4) ∧ x = 6/7 ∧
  (∀ (y : ℝ), (-3/2 < y ∧ y < 4) → f(x) ≥ f(y)) ∧
  f(x) = 432 * (11 / 7)^7 :=
by
  sorry

end max_value_of_f_on_interval_l567_567482


namespace largest_int_less_than_100_mod_6_eq_4_l567_567443

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l567_567443


namespace largest_int_mod_6_less_than_100_l567_567450

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l567_567450


namespace avg_percentage_decrease_l567_567805

theorem avg_percentage_decrease (x : ℝ) 
  (h : 16 * (1 - x)^2 = 9) : x = 0.25 :=
sorry

end avg_percentage_decrease_l567_567805


namespace relationship_among_a_b_c_l567_567506

noncomputable def f : ℝ → ℝ := sorry

theorem relationship_among_a_b_c :
  (∀ x, f (x - 1) = f (1 - (x - 1))) →                -- Symmetry condition
  (∀ x, x < 0 → f x + x * (has_deriv_at f x) < 0) →     -- Derivative condition
  (let a := 3^0.3 * f (3^0.3),
       b := real.log 3 / real.log π * f (real.log 3 / real.log π),
       c := -2 * f (-2) 
  in c > a ∧ a > b) :=
by sorry

end relationship_among_a_b_c_l567_567506


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567422

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567422


namespace no_integer_solution_for_expression_l567_567302

theorem no_integer_solution_for_expression (x y z : ℤ) :
  x^4 + y^4 + z^4 - 2 * x^2 * y^2 - 2 * y^2 * z^2 - 2 * z^2 * x^2 ≠ 2000 :=
by sorry

end no_integer_solution_for_expression_l567_567302


namespace largest_minus_second_largest_l567_567199

-- Define the digits
def digits : List ℕ := [1, 0, 5, 8]

-- Define a helper function to generate two-digit numbers from a list of digits
def two_digit_numbers (ds : List ℕ) : List ℕ :=
  (ds.product ds).filter (λ p => p.fst ≠ p.snd ∧ p.fst ≠ 0).map (λ p => 10 * p.fst + p.snd)

-- Statement of the proof problem: the difference between the largest and the second largest two-digit numbers.
theorem largest_minus_second_largest : 
  let numbers := two_digit_numbers digits in
  (numbers.maximum.get_or_else 0) - (numbers.erase (numbers.maximum.get_or_else 0)).maximum.get_or_else 0 = 4 :=
by
  sorry

end largest_minus_second_largest_l567_567199


namespace sum_of_divisors_143_l567_567724

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l567_567724


namespace cosine_value_m_neg3_right_angled_triangle_m_value_l567_567480

notation "ℝ" => Real

noncomputable def point_A : ℝ × ℝ := (-2, 4)
noncomputable def point_B : ℝ × ℝ := (3, -1)
noncomputable def point_C (m : ℝ) : ℝ × ℝ := (m, -4)

-- Vectors
noncomputable def vector_AB : ℝ × ℝ := (point_B.1 - point_A.1, point_B.2 - point_A.2)
noncomputable def vector_BC (m : ℝ) : ℝ × ℝ := (point_C(m).1 - point_B.1, point_C(m).2 - point_B.2)

-- Magnitudes of the vectors
noncomputable def magnitude_AB : ℝ := Math.sqrt (vector_AB.1^2 + vector_AB.2^2)
noncomputable def magnitude_BC (m : ℝ) : ℝ := Math.sqrt ((vector_BC m).1^2 + (vector_BC m).2^2)

-- Dot product of the vectors
noncomputable def dot_product_AB_BC (m : ℝ) : ℝ := vector_AB.1 * (vector_BC m).1 + vector_AB.2 * (vector_BC m).2

-- Cosine of the angle between vectors
noncomputable def cosine_angle (m : ℝ) : ℝ := dot_product_AB_BC(m) / (magnitude_AB * magnitude_BC m)

-- Condition for the cosine value when m = -3
theorem cosine_value_m_neg3 : cosine_angle (-3) = -Math.sqrt(10)/10 :=
by
  sorry

-- Condition for the value of m such that triangle is right-angled at A
theorem right_angled_triangle_m_value : (dot_product_AB_BC(-10) = 0) ↔ (vector_AB.1 * (point_C(-10).1 - point_A.1) + vector_AB.2 * (point_C(-10).2 - point_A.2) = 0):=
by
  sorry

end cosine_value_m_neg3_right_angled_triangle_m_value_l567_567480


namespace sum_of_first_100_terms_l567_567135

def seq : ℕ → ℤ
| 0       := 1
| 1       := 3
| (n + 2) := seq (n + 1) - seq n

theorem sum_of_first_100_terms : 
  (Finset.range 100).sum seq = 5 := 
sorry

end sum_of_first_100_terms_l567_567135


namespace problem_l567_567492

noncomputable def f (x : ℝ) : ℝ :=
  sorry

theorem problem (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x + y) - f y = x * (x + 2 * y + 1))
                (h2 : f 1 = 0) :
  f 0 = -2 ∧ ∀ x : ℝ, f x = x^2 + x - 2 := by
  sorry

end problem_l567_567492


namespace product_of_integers_l567_567303

theorem product_of_integers (x y : ℕ) (h1 : x + y = 26) (h2 : x^2 - y^2 = 52) : x * y = 168 := by
  sorry

end product_of_integers_l567_567303


namespace max_profit_under_budget_max_profit_no_budget_l567_567808

-- Definitions from conditions
def sales_revenue (x1 x2 : ℝ) : ℝ :=
  -2 * x1^2 - x2^2 + 13 * x1 + 11 * x2 - 28

def profit (x1 x2 : ℝ) : ℝ :=
  sales_revenue x1 x2 - x1 - x2

-- Statements for the conditions
theorem max_profit_under_budget :
  (∀ x1 x2 : ℝ, x1 + x2 = 5 → profit x1 x2 ≤ 9) ∧
  (profit 2 3 = 9) :=
by sorry

theorem max_profit_no_budget :
  (∀ x1 x2 : ℝ, profit x1 x2 ≤ 15) ∧
  (profit 3 5 = 15) :=
by sorry

end max_profit_under_budget_max_profit_no_budget_l567_567808


namespace graph_intersections_l567_567222

noncomputable def g : ℝ → ℝ :=
sorry  -- g is a placeholder for any invertible function

theorem graph_intersections : 
  (∃ a b : ℝ, a ≠ b ∧ g (a^3) = g (a^6) ∧ g (b^3) = g (b^6) ∧ (∀ c : ℝ, g (c^3) = g (c^6) → c = a ∨ c = b)) :=
begin
  sorry
end

end graph_intersections_l567_567222


namespace triangle_area_l567_567287

section
variables (P Q R : ℝ × ℝ)

def PQ : ℝ := (Q.1 - P.1)
def RS : ℝ := (P.2 - R.2)
def area_triangle : ℝ := (1 / 2) * PQ * RS

theorem triangle_area : 
  P = (-4, 2) ∧ Q = (6, 2) ∧ R = (2, -5) → 
  area_triangle P Q R = 35 :=
by 
  sorry
end

end triangle_area_l567_567287


namespace max_value_x_sq_y_l567_567495

theorem max_value_x_sq_y (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 3 * y = 9) :
  x^2 * y ≤ 36 :=
sorry

end max_value_x_sq_y_l567_567495


namespace sum_of_divisors_143_l567_567778

theorem sum_of_divisors_143 : ∑ d in List.toFinset (List.divisors 143), d = 168 := by
  have prime_factors_143 : 143 = 11 * 13 := by
    calc 143 = 11 * 13 : by norm_num
  sorry

end sum_of_divisors_143_l567_567778


namespace largest_integer_with_remainder_l567_567429

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l567_567429


namespace final_price_calculation_l567_567790

variable originalPrice : ℝ := 160
variable increasedPriceRate : ℝ := 0.25
variable discountRate : ℝ := 0.25

theorem final_price_calculation (op : ℝ) (ipr : ℝ) (dr : ℝ) : 
  let increasedPrice := op * (1 + ipr)
  let discount := increasedPrice * dr
  increasedPrice - discount = 150 := 
by
  intro op ipr dr
  let increasedPrice := op * (1 + ipr)
  let discount := increasedPrice * dr
  exact sorry

#eval final_price_calculation originalPrice increasedPriceRate discountRate

end final_price_calculation_l567_567790


namespace expected_pairs_of_red_in_circle_deck_l567_567652

noncomputable def expected_pairs_of_adjacent_red_cards (deck_size : ℕ) (red_cards : ℕ) : ℚ :=
  let adjacent_probability := (red_cards - 1 : ℚ) / (deck_size - 1)
  in red_cards * adjacent_probability

theorem expected_pairs_of_red_in_circle_deck :
  expected_pairs_of_adjacent_red_cards 52 26 = 650 / 51 :=
by
  sorry

end expected_pairs_of_red_in_circle_deck_l567_567652


namespace probability_no_adjacent_stand_l567_567019

theorem probability_no_adjacent_stand (
  n : ℕ := 8
) :
  ∑ (k : ℕ) in (finset.range (n + 1)).filter (λ k, ∀ i j, i ≠ j → (i + 1 = j ∨ i = j + 1) → ¬(i ∈ S ∧ j ∈ S)), 
    Nat.choose n k = 47 / 256 := 
  sorry

end probability_no_adjacent_stand_l567_567019


namespace totalBirdsOnFence_l567_567318

/-
Statement: Given initial birds and additional birds joining, the total number
           of birds sitting on the fence is 10.
Conditions:
1. Initially, there are 4 birds.
2. 6 more birds join them.
3. There are 46 storks on the fence, but they do not affect the number of birds.
-/

def initialBirds : Nat := 4
def additionalBirds : Nat := 6

theorem totalBirdsOnFence : initialBirds + additionalBirds = 10 := by
  sorry

end totalBirdsOnFence_l567_567318


namespace min_weight_of_lightest_l567_567704

theorem min_weight_of_lightest (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (h1 : 71 * m + m = 72 * m) 
  (h2 : 34 * n + n = 35 * n) 
  (h3 : 72 * m = 35 * n) : m = 35 := 
sorry

end min_weight_of_lightest_l567_567704


namespace find_center_and_radius_l567_567496

-- Conditions
variables {m : ℝ}

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 + x - 6 * y + m

def line_eq (x y : ℝ) : ℝ := x + 2 * y - 3

axiom OP_perpendicular_OQ : ∀ (P Q : ℝ × ℝ), 
  OP P ∧ OQ Q → 
  let ⟨x1, y1⟩ := P, ⟨x2, y2⟩ := Q in 
  (x1 * x2 + y1 * y2 = 0)

-- The proof statement
theorem find_center_and_radius (P Q : ℝ × ℝ) (H_inter : circle_eq (P.fst) (P.snd) = 0 ∧ circle_eq (Q.fst) (Q.snd) = 0) 
  (H_line : line_eq (P.fst) (P.snd) = 0 ∧ line_eq (Q.fst) (Q.snd) = 0) 
  (H_perpendicular : OP_perpendicular_OQ P Q) :
  ∃ (h k r : ℝ), h = -1 / 2 ∧ k = 3 ∧ r = 5 / 2 :=
sorry

end find_center_and_radius_l567_567496


namespace derivative_of_f_at_pi_over_2_l567_567469

noncomputable def f (x : Real) := 5 * Real.sin x

theorem derivative_of_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = 0 :=
by
  -- The proof is omitted
  sorry

end derivative_of_f_at_pi_over_2_l567_567469


namespace sum_to_12_of_7_chosen_l567_567214

theorem sum_to_12_of_7_chosen (S : Finset ℕ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) (T : Finset ℕ) (hT1 : T ⊆ S) (hT2 : T.card = 7) :
  ∃ (a b : ℕ), a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ a + b = 12 :=
by
  sorry

end sum_to_12_of_7_chosen_l567_567214


namespace complex_div_eq_half_add_half_i_l567_567071

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem to be proven
theorem complex_div_eq_half_add_half_i :
  (i / (1 + i)) = (1 / 2 + (1 / 2) * i) :=
by
  -- The proof will go here
  sorry

end complex_div_eq_half_add_half_i_l567_567071


namespace min_choir_members_l567_567334

theorem min_choir_members (n : ℕ) : 
  (∀ (m : ℕ), m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → m ≥ n) → 
  n = 990 :=
by
  sorry

end min_choir_members_l567_567334


namespace no_possible_values_of_k_l567_567365

theorem no_possible_values_of_k (k : ℤ) :
  (∀ p q : ℤ, p * q = k ∧ p + q = 58 → ¬ (nat.prime p ∧ nat.prime q)) := 
by
  sorry

end no_possible_values_of_k_l567_567365


namespace floor_sqrt_sum_l567_567005

theorem floor_sqrt_sum : (Finset.range 25).sum (λ n, Int.floor (Real.sqrt (n + 1))) = 71 := sorry

end floor_sqrt_sum_l567_567005


namespace union_of_solution_sets_l567_567252

theorem union_of_solution_sets (p q : ℝ) (A B : set ℝ)
  (hA : A = {x | x^2 - (p - 1) * x + q = 0})
  (hB : B = {x | x^2 + (q - 1) * x + p = 0})
  (h_inter : A ∩ B = {-2}) :
  A ∪ B = {-2, -1, 1} :=
sorry

end union_of_solution_sets_l567_567252


namespace meaningful_sqrt_range_l567_567275

theorem meaningful_sqrt_range (x : ℝ) : (3 * x - 6 ≥ 0) ↔ (x ≥ 2) := by
  sorry

end meaningful_sqrt_range_l567_567275


namespace math_expression_equals_80_476_l567_567855

noncomputable def math_expression : ℝ :=
  let part1 := (2^(1/3) * 3^(1/2))^6
  let part2 := (2^(1/2) * 2^(1/4))^(4/3)
  let part3 := -4 * (16/49)^(-1/2)
  let part4 := -2^(1/4) * 8^(0.25)
  let part5 := -2 * ((1 - (Real.log 3 / Real.log 6))^2 + (Real.log 2 / Real.log 6) * (Real.log 2 / Real.log 18)) / (Real.log 4 / Real.log 6)
  part1 + part2 + part3 + part4 + part5 

theorem math_expression_equals_80_476 : math_expression = 80.476 := by
  sorry

end math_expression_equals_80_476_l567_567855


namespace area_closed_figure_l567_567644

noncomputable def area_between_curves : ℝ :=
  ∫ x in 0..1, x^2 - x^3

theorem area_closed_figure :
  area_between_curves = 1 / 12 := sorry

end area_closed_figure_l567_567644


namespace tetrahedron_dot_product_l567_567988

noncomputable def point_vector (A B C D : ℝ³) := sorry

def midpoint_vector (x y : ℝ³) : ℝ³ := (x + y) / 2

theorem tetrahedron_dot_product
  (A B C D E F : ℝ³)
  (h1 : E = midpoint_vector B C)
  (h2 : F = midpoint_vector A D)
  (tetrahedron_condition : ∀ (i j : ℝ³), (i ≠ j → (i • j = -1/3) ∧ (i • i = 1)))
  : ((midpoint_vector B C) - A) • ((midpoint_vector A D) - C) = -1/2 :=
sorry

end tetrahedron_dot_product_l567_567988


namespace largest_piece_remains_l567_567309

-- Define the initial conditions for the problem
def cake_size : ℕ := 3
def cut_square_size : ℕ := 1

-- Statement of the problem
theorem largest_piece_remains (cake_size = 3) (cut_square_size = 1) :
  let largest_possible_side := 1 / 3
  in largest_possible_side > 0 :=
sorry

end largest_piece_remains_l567_567309


namespace minimum_of_expression_l567_567920

-- Define the conditions
variable {x y : ℝ}
variable condition : (x - 3) ^ 2 + y ^ 2 = 9

-- Define the target statement
theorem minimum_of_expression : (∃ (x y : ℝ), (x - 3) ^ 2 + y ^ 2 = 9) → ∃ (min_val : ℝ), min_val = -3 * Real.sqrt 13 - 9 ∧ (∀ (x y : ℝ), (x - 3) ^ 2 + y ^ 2 = 9 → -2 * y - 3 * x ≥ min_val) := by
  sorry

end minimum_of_expression_l567_567920


namespace number_of_b_l567_567857

theorem number_of_b :
  (∃ b : ℕ, 1 ≤ b ∧ b ≤ 2013 ∧ b ≠ 17 ∧ b ≠ 18 ∧
  ∃ N : ℕ, N > 0 ∧
  (∃ k₁ : ℕ, (17^k₁ ∣ N ∧ (N / 17^k₁) ^ 17 = N / 17 ^ k₁) ∧
  ∃ k₂ : ℕ, (18^k₂ ∣ N ∧ (N / 18^k₂) ^ 18 = N / 18^k₂) ∧
  ∃ k₃ : ℕ, (b^k₃ ∣ N ∧ (N / b^k₃) ^ b = N / b ^ k₃))) ↔ 690) :=
sorry

end number_of_b_l567_567857


namespace find_increase_in_perimeter_l567_567789

variable (L B y : ℕ)

theorem find_increase_in_perimeter (h1 : 2 * (L + y + (B + y)) = 2 * (L + B) + 16) : y = 4 := by
  sorry

end find_increase_in_perimeter_l567_567789


namespace largest_int_mod_6_less_than_100_l567_567449

theorem largest_int_mod_6_less_than_100 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
sorry

end largest_int_mod_6_less_than_100_l567_567449


namespace number_of_zeros_in_square_l567_567483

theorem number_of_zeros_in_square 
(h1 : (9:ℕ)^2 = 81)
(h2 : (99:ℕ)^2 = 9801)
(h3 : (999:ℕ)^2 = 998001) :
  let n := 999999999
  in natZeros ((n:ℕ)^2) = 8 := 
begin
  sorry
end

end number_of_zeros_in_square_l567_567483


namespace regular_price_of_one_tire_l567_567230

theorem regular_price_of_one_tire
  (x : ℝ) -- Define the variable \( x \) as the regular price of one tire
  (h1 : 3 * x + 10 = 250) -- Set up the equation based on the condition

  : x = 80 := 
sorry

end regular_price_of_one_tire_l567_567230


namespace sum_of_divisors_143_l567_567716

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l567_567716


namespace train_length_is_600_l567_567835

noncomputable def length_of_train (v : ℝ) (d_bridge : ℝ) (t : ℝ) : ℝ :=
  let speed := v * (1000 / 3600)
  let distance := speed * t
  distance - d_bridge

theorem train_length_is_600
  (v : ℝ := 36)
  (d_bridge : ℝ := 100)
  (t : ℝ := 70) :
  length_of_train v d_bridge t = 600 :=
by
  unfold length_of_train
  calc
    v * (1000 / 3600) * t - d_bridge = 36 * (1000 / 3600) * 70 - 100 : by rfl
                              ...   = 10 * 70 - 100 : by norm_num
                              ...   = 700 - 100 : by norm_num
                              ...   = 600 : by norm_num

end train_length_is_600_l567_567835


namespace malysh_wins_with_mirroring_strategy_l567_567616

theorem malysh_wins_with_mirroring_strategy :
  ∃ strategy: ℕ → ℕ → ℕ, 
  ∀ n: ℕ, n = 5000 → 
    (∃ turns : ℕ,
      (∀ i < turns, strategy n i ∈ {n // n % 2 = 0} ∨ {n // n % 5 = 0} ∨ {n // n % 10 = 0}) ∧ 
      ∃ k: ℕ, strategy 5000 k → ∀ k' > k, strategy 5000 k' = 0 ∧ 
      (∃ m: ℕ, ∀ m' > m, ¬(strategy 5000 m' ∈ {n // n % 2 = 0} ∨ {n // n % 5 = 0} ∨ {n // n % 10 = 0}))) :=
begin
  sorry
end

end malysh_wins_with_mirroring_strategy_l567_567616


namespace complex_number_quadrant_l567_567812

def inSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_quadrant : inSecondQuadrant (i / (1 - i)) :=
by
  sorry

end complex_number_quadrant_l567_567812


namespace range_of_a_has_three_integer_solutions_l567_567981

theorem range_of_a_has_three_integer_solutions (a : ℝ) :
  (∃ (x : ℤ → ℝ), (2 * x - 1 > 3) ∧ (x ≤ 2 * a - 1) ∧ (x = 3 ∨ x = 4 ∨ x = 5)) → (3 ≤ a ∧ a < 3.5) :=
sorry

end range_of_a_has_three_integer_solutions_l567_567981


namespace sum_of_divisors_143_l567_567726

theorem sum_of_divisors_143 : sigma 143 = 168 := by
  have h : 143 = 11 * 13 := by norm_num
  rw [sigma_mul_prime_prime 11 13]
  norm_num
  sorry

end sum_of_divisors_143_l567_567726


namespace minimum_production_meets_demand_l567_567336

-- Define f(x) as given in the problem
def f (x : ℕ) : ℕ := x * (x + 1) * (35 - 2 * x)

-- Define the condition on x
def valid_x (x : ℕ) : Prop := x ∈ {n : ℕ | n ≥ 1 ∧ n ≤ 12}

-- Define g(x) that corresponds to the monthly demand
def g (x : ℕ) : ℕ := f(x) - f(x - 1)

-- The expression found in the solution for g(x)
def g_expression (x : ℕ) : ℕ := -6 * x^2 + 72 * x

-- Minimum monthly production required to meet the demand
def minimum_a : ℕ := 171

-- Main theorem to be proven
theorem minimum_production_meets_demand (x a : ℕ) (hx : valid_x x) :
  g_expression x = g x ∧ a ≥ minimum_a :=
by
  sorry

end minimum_production_meets_demand_l567_567336


namespace largest_int_less_than_100_remainder_4_l567_567411

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l567_567411


namespace cos_sum_equals_neg_one_l567_567087

def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 + x + 1 else 2 * x + 1

variables {α β r : ℝ}

theorem cos_sum_equals_neg_one
  (H1 : f (sin α + sin β + sin r - 1) = -1)
  (H2 : f (cos α + cos β + cos r + 1) = 3) :
  cos (α - β) + cos (β - r) = -1 :=
by
-- We provide the theorem statement; the proof is omitted.
sorry

end cos_sum_equals_neg_one_l567_567087


namespace base_8_to_10_conversion_l567_567711

theorem base_8_to_10_conversion : (2 * 8^4 + 3 * 8^3 + 4 * 8^2 + 5 * 8^1 + 6 * 8^0) = 10030 := by 
  -- specify the summation directly 
  sorry

end base_8_to_10_conversion_l567_567711


namespace projection_of_sum_on_b_l567_567953

variables (a b : ℝ^3)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1) (angle_ab : ∡ a b = 120)

theorem projection_of_sum_on_b (a b : ℝ^3)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 1)
  (angle_ab : real.angle (a • b / (∥a∥ * ∥b∥))) = real.angle.vector_angle 120):
  (a + b) • b / ∥b∥ = 1 / 2 :=
  sorry

end projection_of_sum_on_b_l567_567953


namespace lightest_weight_minimum_l567_567709

theorem lightest_weight_minimum (distinct_masses : ∀ {w : set ℤ}, ∀ (x ∈ w) (y ∈ w), x = y → x = y)
  (lightest_weight_ratio : ∀ {weights : list ℤ} (m : ℤ), m = list.minimum weights →
     sum (list.filter (≠ m) weights) = 71 * m)
  (two_lightest_weights_ratio : ∀ {weights : list ℤ} (n m : ℤ), m ∈ weights → n ∈ weights →
     n + m = list.minimum (m :: list.erase weights m) →
     sum (list.filter (≠ n + m) weights) = 34 * (n + m)) :
  ∃ (m : ℤ), m = 35 := 
sorry

end lightest_weight_minimum_l567_567709


namespace find_a_l567_567400

def polys_are_integers (a b : Int) : Prop :=
  a ≠ 0 ∧ b ≠ 0

def factor_condition (a b : Int) : Prop :=
  ∀ x : ℝ, (x^2 - x - 1 = 0) → (a * x^19 + b * x^18 + 1 = 0)

theorem find_a (a b : Int) (h : polys_are_integers a b) (h_fac : factor_condition a b) : a = 1597 :=
by
  sorry

end find_a_l567_567400


namespace trig_function_set_count_l567_567170

theorem trig_function_set_count :
  ∃ S : Set (ℝ × ℝ × Icc (0 : ℝ) (2 * Real.pi)), S.card = 4 ∧
  ∀ (a b : ℝ) (c : Icc (0 : ℝ) (2 * Real.pi)), (a, b, c) ∈ S ↔ 
  ∀ x : ℝ, 2 * Real.sin (3 * x - Real.pi / 3) = a * Real.sin (b * x + c) :=
sorry

end trig_function_set_count_l567_567170


namespace complex_number_in_third_quadrant_l567_567679

def complex_plane_quadrant (z : ℂ) : ℕ :=
  if z.im > 0 then
    if z.re > 0 then 1 else 2
  else 
    if z.re < 0 then 3 else 4

theorem complex_number_in_third_quadrant :
  complex_plane_quadrant (i * (-2 + i)) = 3 :=
by
  sorry

end complex_number_in_third_quadrant_l567_567679


namespace at_least_502_friendly_numbers_l567_567354

def friendly (a : ℤ) : Prop :=
  ∃ (m n : ℤ), m > 0 ∧ n > 0 ∧ (m^2 + n) * (n^2 + m) = a * (m - n)^3

theorem at_least_502_friendly_numbers :
  ∃ S : Finset ℤ, (∀ a ∈ S, friendly a) ∧ 502 ≤ S.card ∧ ∀ x ∈ S, 1 ≤ x ∧ x ≤ 2012 :=
by
  sorry

end at_least_502_friendly_numbers_l567_567354


namespace min_weight_of_lightest_l567_567705

theorem min_weight_of_lightest (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (h1 : 71 * m + m = 72 * m) 
  (h2 : 34 * n + n = 35 * n) 
  (h3 : 72 * m = 35 * n) : m = 35 := 
sorry

end min_weight_of_lightest_l567_567705


namespace main_theorem_l567_567056

open EuclideanGeometry

variables (A B C P Q B' C' P' Q' O O' : Point)

def reflection_point (P : Point) (BC : Line) : Point := sorry -- Assuming a definition exists

def is_circumcenter (O : Point) (Δ : Triangle) : Prop := sorry -- Assuming a definition exists

theorem main_theorem (hABC_scalene : ¬ collinear A B C)
  (hB' : B' ∈ ray A B) (hC' : C' ∈ ray A C)
  (hAB' : distance A B' = distance A C) (hAC' : distance A C' = distance A B)
  (hQ : Q = reflection_point P (line_through B C))
  (hIsCircumABC : is_circumcenter O (triangle A B C))
  (hIsCircumAB'C' : is_circumcenter O' (triangle A B' C'))
  (hP' : in_circles_intersection (circle_through B B' P) (circle_through C C' P) P')
  (hQ' : in_circles_intersection (circle_through B B' Q) (circle_through C C' Q) Q') :
  collinear [O', P', Q'] ∧ (distance O' P' * distance O' Q' = distance O A ^ 2) :=
begin
  sorry
end

end main_theorem_l567_567056


namespace fish_catch_l567_567152

theorem fish_catch (B : ℕ) (K : ℕ) (hB : B = 5) (hK : K = 2 * B) : B + K = 15 :=
by
  sorry

end fish_catch_l567_567152


namespace round_475_to_nearest_half_l567_567209

-- Given the rounding rule: if a number is exactly in the middle (.x5), round up.
def round_nearest_half (x : ℝ) : ℝ :=
  if x - x.floor == 0.5 then x.ceil else
  if x - x.floor < 0.5 then x.floor + 0.5 else x.floor + 1

theorem round_475_to_nearest_half : round_nearest_half 4.75 = 5 :=
by
  sorry

end round_475_to_nearest_half_l567_567209


namespace continued_fraction_equality_l567_567842

theorem continued_fraction_equality: 
    let x := 1 + (1 / (2 + (1 / (3 + (1 / 4))))) in 
    x = 43 / 30 :=
by
    sorry

end continued_fraction_equality_l567_567842


namespace modulus_of_z_l567_567497

theorem modulus_of_z (z : ℂ) (h : (1 + complex.I) * z = 1 - complex.I) : complex.abs z = real.sqrt 2 :=
sorry

end modulus_of_z_l567_567497


namespace walkway_area_correct_l567_567993

/-- Definitions as per problem conditions --/
def bed_length : ℕ := 8
def bed_width : ℕ := 3
def walkway_bed_width : ℕ := 2
def walkway_row_width : ℕ := 1
def num_beds_in_row : ℕ := 3
def num_rows : ℕ := 4

/-- Total dimensions including walkways --/
def total_width := num_beds_in_row * bed_length + (num_beds_in_row + 1) * walkway_bed_width
def total_height := num_rows * bed_width + (num_rows + 1) * walkway_row_width

/-- Total areas --/
def total_area := total_width * total_height
def bed_area := bed_length * bed_width
def total_bed_area := num_beds_in_row * num_rows * bed_area
def walkway_area := total_area - total_bed_area

theorem walkway_area_correct : walkway_area = 256 := by
  /- Import necessary libraries and skip the proof -/
  sorry

end walkway_area_correct_l567_567993


namespace ratio_sum_divisors_l567_567176

def sum_of_divisors (n : ℕ) : ℕ :=
  ∑ i in (finset.range (n + 1)).filter (λ d, n % d = 0), d

def sum_of_odd_divisors (n : ℕ) : ℕ :=
  ∑ i in (finset.range (n + 1)).filter (λ d, n % d = 0 ∧ d % 2 = 1), d

def sum_of_even_divisors (n : ℕ) : ℕ :=
  ∑ i in (finset.range (n + 1)).filter (λ d, n % d = 0 ∧ d % 2 = 0), d

theorem ratio_sum_divisors (M : ℕ) (hM : M = 42 * 43 * 75 * 196) :
  (sum_of_odd_divisors M) * 14 = sum_of_even_divisors M :=
by {
  rw hM,
  sorry
}

end ratio_sum_divisors_l567_567176


namespace spheres_tangent_to_lines_l567_567802

-- This defines a spatial quadrilateral where points are not coplanar
structure SpatialQuadrilateral :=
  (A B C D : Point)
  (nonCoplanar : ¬coplanar A B C D)

-- Given conditions
variables (A B C D: Point)
variables (AB BC CD DA: Line)
variables [SpatialQuadrilateral (A B C D)]

-- Main theorem stating the proof problem in Lean 4
theorem spheres_tangent_to_lines
  (AB BC CD DA: Line)
  (AB_eq_a : |AB| = a)
  (BC_eq_b : |BC| = b)
  (CD_eq_c : |CD| = c)
  (DA_eq_d : |DA| = d) :
  (∃ (spheres: Set Sphere), spheres.TangentLines = {AB, BC, CD, DA} ∧ spheres.Count ≥ 8) ∧
  ((a + c = b + d) → (∃ (spheres: Set Sphere), spheres.TangentLines = {AB, BC, CD, DA} ∧ spheres.Count = ∞)) :=
sorry

end spheres_tangent_to_lines_l567_567802


namespace probability_non_expired_bags_l567_567839

theorem probability_non_expired_bags :
  let total_bags := 5
  let expired_bags := 2
  let selected_bags := 2
  let total_combinations := Nat.choose total_bags selected_bags
  let non_expired_bags := total_bags - expired_bags
  let favorable_outcomes := Nat.choose non_expired_bags selected_bags
  (favorable_outcomes : ℚ) / (total_combinations : ℚ) = 3 / 10 := by
  sorry

end probability_non_expired_bags_l567_567839


namespace exist_pair_sum_to_12_l567_567213

theorem exist_pair_sum_to_12 (S : Set ℤ) (hS : S = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}) (chosen : Set ℤ) (hchosen : chosen ⊆ S) (hsize : chosen.card = 7) :
  ∃x ∈ chosen, ∃y ∈ chosen, x ≠ y ∧ x + y = 12 := 
sorry

end exist_pair_sum_to_12_l567_567213


namespace number_of_toothpicks_l567_567691

def num_horizontal_toothpicks(lines width : Nat) : Nat := lines * width
def num_vertical_toothpicks(lines height : Nat) : Nat := lines * height

theorem number_of_toothpicks (high wide : Nat) (missing : Nat) 
  (h_high : high = 15) (h_wide : wide = 15) (h_missing : missing = 1) : 
  num_horizontal_toothpicks (high + 1) wide + num_vertical_toothpicks (wide + 1) high - missing = 479 := by
  sorry

end number_of_toothpicks_l567_567691


namespace susie_investment_l567_567225

theorem susie_investment :
  ∃ x : ℝ, x * (1 + 0.04)^3 + (2000 - x) * (1 + 0.06)^3 = 2436.29 → x = 820 :=
by
  sorry

end susie_investment_l567_567225


namespace largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567421

theorem largest_int_with_remainder (x : ℕ) (h1 : x % 6 = 4) (h2 : x < 100) : x ≤ 94 :=
by 
  sorry

theorem largest_int_with_remainder_achievable : ∃ n : ℕ, n % 6 = 4 ∧ n < 100 ∧ n = 94 := 
by
  exists 94
  simp [←nat.mod_eq_of_lt 100]
  constructor
  exact nat.mod_eq_of_lt (by norm_num : 94 < 6*16)
  constructor
  norm_num
  rfl

end largest_int_with_remainder_largest_int_with_remainder_achievable_l567_567421


namespace max_abs_sum_l567_567530

theorem max_abs_sum (x y : ℝ) (h : x^2 + y^2 = 16) : |x| + |y| ≤ 4 * real.sqrt 2 :=
sorry

end max_abs_sum_l567_567530


namespace pos_solution_approx_l567_567899

noncomputable def find_positive_solution := 
  let y := 1.4656
  let x := y * y * y
  x

theorem pos_solution_approx : abs (find_positive_solution - 3.1412) < 0.0001 := sorry

end pos_solution_approx_l567_567899


namespace reed_smilax_equal_height_in_days_l567_567231

noncomputable def log2 : ℝ := 0.3010
noncomputable def log3 : ℝ := 0.4771

theorem reed_smilax_equal_height_in_days :
  ∃ n : ℝ, 6 * (1 - 1 / (2^n)) = (2^n - 1) ∧ abs(n - (1 + log3 / log2)) < 0.1 :=
by
  sorry

end reed_smilax_equal_height_in_days_l567_567231


namespace sum_of_divisors_of_143_l567_567745

theorem sum_of_divisors_of_143 : 
  (∑ d in (finset.filter (λ d, 143 % d = 0) (finset.range (143 + 1))), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l567_567745


namespace proof_triangle_equilateral_l567_567546

noncomputable def triangle_equilateral (A B C E F M : Type) 
  (h1 : |AB| = |AC|)
  (h2 : M ∈ minor_arc_AC)
  (h3 : M ≠ A ∧ M ≠ C)
  (h4 : ∃ E, BM ∩ AC = {E})
  (h5 : ∃ F, bisector ∠BMC ∩ BC = {F})
  (h6 : ∠AFB = ∠CFE) : Prop :=
triangle ABC is equilateral

theorem proof_triangle_equilateral (A B C E F M : Type) 
  (h1 : |AB| = |AC|)
  (h2 : M ∈ minor_arc_AC)
  (h3 : M ≠ A ∧ M ≠ C)
  (h4 : ∃ E, BM ∩ AC = {E})
  (h5 : ∃ F, bisector ∠BMC ∩ BC = {F})
  (h6 : ∠AFB = ∠CFE) : triangle_equilateral A B C E F M h1 h2 h3 h4 h5 h6 :=
sorry

end proof_triangle_equilateral_l567_567546


namespace equivalence_of_triangle_properties_l567_567538

theorem equivalence_of_triangle_properties
  (a b c A B C R r : ℝ)
  (h1 : A + B + C = π)
  (h2 : a = b = c ↔ (A + C = 2B ∧ a + c = 2b) ↔ (A + C = 2B ∧ b^2 = ac) ↔
        (A + C = 2B ∧ (1/a + 1/c = 2/b)) ↔ (cos A + cos B + cos C = 3/2) ↔
        (cos A * cos B * cos C = 1/8) ↔ (cos A^2 + cos B^2 + cos C^2 = 3/4) ↔
        (sin (A/2)^2 + sin (B/2)^2 + sin (C/2)^2 = 3/4) ↔ (R = 2 * r)) :
  a = b = c :=
sorry

end equivalence_of_triangle_properties_l567_567538


namespace sum_of_divisors_143_l567_567723

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l567_567723


namespace binary_to_base_10_l567_567008

theorem binary_to_base_10 : 
  let binary := [1, 1, 0, 1, 1]
  let base := 2
  ∑ (i : Nat) in finRange binary.length, binary.get?! i * base ^ i = 27 := 
by
  sorry

end binary_to_base_10_l567_567008


namespace smallest_a_exists_l567_567033

theorem smallest_a_exists : ∃ a b c : ℤ, a > 0 ∧ b^2 > 4*a*c ∧ 
  (∀ x : ℝ, x > 0 ∧ x < 1 → (a * x^2 - b * x + c) = 0 → false) 
  ∧ a = 5 :=
by sorry

end smallest_a_exists_l567_567033


namespace mac_loses_l567_567190

def dime_value := 0.10
def nickel_value := 0.05
def quarter_value := 0.25

def dimes_per_quarter := 3
def nickels_per_quarter := 7

def num_quarters_with_dimes := 20
def num_quarters_with_nickels := 20

def total_quarters := num_quarters_with_dimes + num_quarters_with_nickels

def value_of_quarters_received := total_quarters * quarter_value
def value_of_dimes_traded := num_quarters_with_dimes * dimes_per_quarter * dime_value
def value_of_nickels_traded := num_quarters_with_nickels * nickels_per_quarter * nickel_value

def total_value_traded := value_of_dimes_traded + value_of_nickels_traded

theorem mac_loses
  : total_value_traded - value_of_quarters_received = 3.00 :=
by sorry

end mac_loses_l567_567190


namespace find_a_if_f_is_odd_l567_567090

noncomputable def f (a x : ℝ) : ℝ := (Real.logb 2 ((a - x) / (1 + x))) 

theorem find_a_if_f_is_odd (a : ℝ) (h : ∀ x : ℝ, f a (-x) = -f a x) : a = 1 := sorry

end find_a_if_f_is_odd_l567_567090


namespace sum_of_divisors_143_l567_567731

theorem sum_of_divisors_143 : sigma 143 = 168 := by
  have h : 143 = 11 * 13 := by norm_num
  rw [sigma_mul_prime_prime 11 13]
  norm_num
  sorry

end sum_of_divisors_143_l567_567731


namespace reflection_preserves_point_l567_567346

open Matrix

-- Define the initial vectors and the reflection result
def v₁ : Matrix (Fin 2) (Fin 1) ℝ := ![![2], ![-3]]
def v₂ : Matrix (Fin 2) (Fin 1) ℝ := ![![-2], ![7]]
def u : Matrix (Fin 2) (Fin 1) ℝ := ![![3], ![-1]]
def reflected_u : Matrix (Fin 2) (Fin 1) ℝ := ![![-3], ![-3]]

-- Define the reflection function
def reflection (a b : Matrix (Fin 2) (Fin 1) ℝ) : Matrix (Fin 2) (Fin 1) ℝ :=
  2 * b - a

-- Lean proof statement
theorem reflection_preserves_point :
  reflection u (reflection v₁ v₂) = reflected_u := by
  sorry

end reflection_preserves_point_l567_567346


namespace standard_deviation_is_1_5_l567_567232

variable (mean value σ : ℝ)
variable (cond1 : mean = 16.5)
variable (cond2 : value = 13.5)
variable (cond3 : value = mean - 2 * σ)

theorem standard_deviation_is_1_5 : σ = 1.5 :=
by
  rw [cond1, cond2, cond3]
  sorry

end standard_deviation_is_1_5_l567_567232


namespace lottery_probability_prizes_l567_567838

theorem lottery_probability_prizes :
  let total_tickets := 3
  let first_prize_tickets := 1
  let second_prize_tickets := 1
  let non_prize_tickets := 1
  let person_a_wins_first := (2 / 3 : ℝ)
  let person_b_wins_from_remaining := (1 / 2 : ℝ)
  (2 / 3 * 1 / 2) = (1 / 3 : ℝ) := sorry

end lottery_probability_prizes_l567_567838


namespace sharon_trip_distance_l567_567627

noncomputable def usual_speed (x : ℝ) : ℝ := x / 200

noncomputable def reduced_speed (x : ℝ) : ℝ := x / 200 - 30 / 60

theorem sharon_trip_distance (x : ℝ) (h1 : (x / 3) / usual_speed x + (2 * x / 3) / reduced_speed x = 310) : 
x = 220 :=
by
  sorry

end sharon_trip_distance_l567_567627


namespace sequence_contains_every_integer_exactly_once_l567_567010

-- Define the sequence
noncomputable def x_seq : ℕ → ℤ
| 0     := 0
| (n+1) :=
  if ∃ r k, n = 3^(r-1)*(3*k + 1) then
    let r := nat.find (exists_snd (classical.some_spec (nat.exists_eq f'))) in
    let k := nat.find (classical.some_spec (nat.exists_eq g')) in
    x_seq n + (3^r - 1) / 2
  else if ∃ r k, n = 3^(r-1)*(3*k + 2) then
    let r := nat.find (exists_snd (classical.some_spec (nat.exists_eq f'))) in
    let k := nat.find (classical.some_spec (nat.exists_eq g')) in
    x_seq n - (3^r + 1) / 2
  else 0 -- This part will never happen due to the conditions provided

-- The main theorem to prove
theorem sequence_contains_every_integer_exactly_once :
  ∀ m : ℤ, ∃! n : ℕ, x_seq n = m :=
begin
  sorry,
end

end sequence_contains_every_integer_exactly_once_l567_567010


namespace matrix_determinant_zero_implies_sum_of_squares_l567_567065

theorem matrix_determinant_zero_implies_sum_of_squares (a b : ℝ)
  (h : (Matrix.det ![![a - Complex.I, b - 2 * Complex.I],
                       ![1, 1 + Complex.I]]) = 0) :
  a^2 + b^2 = 1 :=
sorry

end matrix_determinant_zero_implies_sum_of_squares_l567_567065


namespace sum_of_divisors_of_143_l567_567744

theorem sum_of_divisors_of_143 : 
  (∑ d in (finset.filter (λ d, 143 % d = 0) (finset.range (143 + 1))), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l567_567744


namespace S_n_formula_l567_567844

def P (n : ℕ) : Type := sorry -- The type representing the nth polygon, not fully defined here.
def S : ℕ → ℝ := sorry -- The sequence S_n defined recursively.

-- Recursive definition of S_n given
axiom S_0 : S 0 = 1

-- This axiom represents the recursive step mentioned in the problem.
axiom S_rec : ∀ (k : ℕ), S (k + 1) = S k + (4^k / 3^(2*k + 2))

-- The main theorem we need to prove
theorem S_n_formula (n : ℕ) : 
  S n = (8 / 5) - (3 / 5) * (4 / 9)^n := sorry

end S_n_formula_l567_567844


namespace sum_of_quadratic_roots_l567_567035

-- Defining the coefficients of the quadratic equation
def a : ℝ := -32
def b : ℝ := 84

-- The equation is -32x^2 + 84x + 135 = 0
def quadratic_eq (x : ℝ) : Prop := -32 * x^2 + 84 * x + 135 = 0

-- The sum of the roots of the quadratic equation
def sum_of_roots : ℝ := -b / a

-- The theorem statement to be proved
theorem sum_of_quadratic_roots : sum_of_roots = 21 / 8 :=
by 
  -- Sum of the roots calculation based on the given quadratic equation
  -- Skipping the actual proof details with 'sorry'
  sorry

end sum_of_quadratic_roots_l567_567035


namespace overall_percentage_loss_is_correct_l567_567825

-- Define original price and percentages
def original_price : ℝ := 100
def increase_percentage : ℝ := 60 / 100
def discount_1_percentage : ℝ := 20 / 100
def discount_2_percentage : ℝ := 15 / 100
def discount_3_percentage : ℝ := 10 / 100
def discount_4_percentage : ℝ := 5 / 100

-- Calculate the final price after successive changes
def increased_price : ℝ := original_price * (1 + increase_percentage)
def price_after_discount_1 : ℝ := increased_price * (1 - discount_1_percentage)
def price_after_discount_2 : ℝ := price_after_discount_1 * (1 - discount_2_percentage)
def price_after_discount_3 : ℝ := price_after_discount_2 * (1 - discount_3_percentage)
def final_price : ℝ := price_after_discount_3 * (1 - discount_4_percentage)

-- Calculate the loss and percentage loss
def loss : ℝ := original_price - final_price
def percentage_loss : ℝ := (loss / original_price) * 100

-- The theorem to be proved
theorem overall_percentage_loss_is_correct : percentage_loss = 6.976 := by
  sorry

end overall_percentage_loss_is_correct_l567_567825


namespace triangle_area_from_arithmetic_sequence_l567_567061

theorem triangle_area_from_arithmetic_sequence:
  (∃ a : ℝ, ∀ n : ℕ, ∑ i in range (n + 1), (a + 1) * (i : ℝ)^2 + a = S_n) →
  let a_1 := 2 * a + 1
      a_2 := 3 * a + 3
      a_3 := a_2 + (a + 2)
      a_4 := a_3 + (a + 2) in
  let P := (a_2 + a_3 + a_4) / 2 in
  (S = (P * (P - a_2) * (P - a_3) * (P - a_4)).sqrt)
    → S = 15 / 4 * real_sqrt 3 :=

begin
  sorry
end

end triangle_area_from_arithmetic_sequence_l567_567061


namespace cost_of_gravelling_path_l567_567787

theorem cost_of_gravelling_path (length width path_width : ℝ) (cost_per_sq_m : ℝ)
  (h1 : length = 110) (h2 : width = 65) (h3 : path_width = 2.5) (h4 : cost_per_sq_m = 0.50) :
  (length * width - (length - 2 * path_width) * (width - 2 * path_width)) * cost_per_sq_m = 425 := by
  sorry

end cost_of_gravelling_path_l567_567787


namespace approx_pi_from_cone_volume_l567_567648

theorem approx_pi_from_cone_volume (L h : ℝ) (r : ℝ) 
  (H1 : L = 2 * real.pi * r) 
  (H2 : V ≈ (2 / 75) * L^2 * h) : 
  real.pi ≈ 25 / 8 :=
by
  sorry

end approx_pi_from_cone_volume_l567_567648


namespace siblings_ate_two_slices_l567_567785

-- Let slices_after_dinner be the number of slices left after eating one-fourth of 16 slices
def slices_after_dinner : ℕ := 16 - 16 / 4

-- Let slices_after_yves be the number of slices left after Yves ate one-fourth of the remaining pizza
def slices_after_yves : ℕ := slices_after_dinner - slices_after_dinner / 4

-- Let slices_left be the number of slices left after Yves's siblings ate some slices
def slices_left : ℕ := 5

-- Let slices_eaten_by_siblings be the number of slices eaten by Yves's siblings
def slices_eaten_by_siblings : ℕ := slices_after_yves - slices_left

-- Since there are two siblings, each ate half of the slices_eaten_by_siblings
def slices_per_sibling : ℕ := slices_eaten_by_siblings / 2

-- The theorem stating that each sibling ate 2 slices
theorem siblings_ate_two_slices : slices_per_sibling = 2 :=
by
  -- Definition of slices_after_dinner
  have h1 : slices_after_dinner = 12 := by sorry
  -- Definition of slices_after_yves
  have h2 : slices_after_yves = 9 := by sorry
  -- Definition of slices_eaten_by_siblings
  have h3 : slices_eaten_by_siblings = 4 := by sorry
  -- Final assertion of slices_per_sibling
  have h4 : slices_per_sibling = 2 := by sorry
  exact h4

end siblings_ate_two_slices_l567_567785


namespace team_division_count_l567_567121

theorem team_division_count (A : Finset ℕ) (hA : A.card = 12) :
  ∃ B C D : Finset ℕ, 
    B.card = 4 ∧ C.card = 4 ∧ D.card = 4 ∧ 
    B ∪ C ∪ D = A ∧
    B ∩ C = ∅ ∧ B ∩ D = ∅ ∧ C ∩ D = ∅ ∧
    (Finset.card ((B ∪ C ∪ D).powerset.filter (λ x, x.card = 4))) / 6 = 5775 := sorry

end team_division_count_l567_567121


namespace expected_adjacent_red_pairs_correct_l567_567662

-- The deck conditions
def standard_deck : Type := {c : ℕ // c = 52}
def num_red_cards (d : standard_deck) := 26

-- Probability definition
def prob_red_right_of_red : ℝ := 25 / 51

-- Expected number of adjacent red pairs calculation
def expected_adjacent_red_pairs (n_red : ℕ) (prob_right_red : ℝ) : ℝ :=
  n_red * prob_right_red

-- Main theorem statement
theorem expected_adjacent_red_pairs_correct (d : standard_deck) :
  expected_adjacent_red_pairs (num_red_cards d) prob_red_right_of_red = 650 / 51 :=
sorry

end expected_adjacent_red_pairs_correct_l567_567662


namespace pos_solution_approx_l567_567898

noncomputable def find_positive_solution := 
  let y := 1.4656
  let x := y * y * y
  x

theorem pos_solution_approx : abs (find_positive_solution - 3.1412) < 0.0001 := sorry

end pos_solution_approx_l567_567898


namespace stormi_cars_washed_l567_567637

-- Definitions based on conditions
def cars_earning := 10
def lawns_number := 2
def lawn_earning := 13
def bicycle_cost := 80
def needed_amount := 24

-- Auxiliary calculations
def lawns_total_earning := lawns_number * lawn_earning
def already_earning := bicycle_cost - needed_amount
def cars_total_earning := already_earning - lawns_total_earning

-- Main problem statement
theorem stormi_cars_washed : (cars_total_earning / cars_earning) = 3 :=
  by sorry

end stormi_cars_washed_l567_567637


namespace green_pens_l567_567677

theorem green_pens (blue_pens green_pens : ℕ) (ratio_blue_to_green : blue_pens / green_pens = 4 / 3) (total_blue : blue_pens = 16) : green_pens = 12 :=
by sorry

end green_pens_l567_567677


namespace sum_of_divisors_143_l567_567720

def sum_divisors (n : ℕ) : ℕ :=
  (1 + 11) * (1 + 13)  -- The sum of the divisors of 143 is interpreted from the given prime factors.

theorem sum_of_divisors_143 : sum_divisors 143 = 168 := by
  sorry

end sum_of_divisors_143_l567_567720


namespace height_of_parallelogram_l567_567031

-- Definition of given conditions
def area : ℝ := 216
def base : ℝ := 12

-- Definition of the height using the given area and base
def height : ℝ := area / base

-- Theorem to prove that the height is 18 cm
theorem height_of_parallelogram : height = 18 := by
  sorry

end height_of_parallelogram_l567_567031


namespace max_area_triangle_PAB_l567_567475

theorem max_area_triangle_PAB (a : ℝ) (h : 0 < a) :
  let A := (-real.sqrt a, 0),
      B := (real.sqrt a, 0) in
  ∀ P : ℝ × ℝ, (P.1 + real.sqrt a) ^ 2 + P.2 ^ 2 = 4 * ((P.1 - real.sqrt a) ^ 2 + P.2 ^ 2) →
  ∃ P_max : ℝ × ℝ, abs P_max.2 = (4 / 3) * real.sqrt a ∧
    let area := real.sqrt a * abs P_max.2 in
  area = (4 / 3) * a :=
by
    -- proof omitted
    sorry


end max_area_triangle_PAB_l567_567475


namespace maximize_profit_l567_567826

noncomputable def profit (p : ℝ) : ℝ :=
  let cost := 20
  let Q := 8300 - 170 * p - p^2
  (8300 - 170 * p - p^2) * (p - cost)

theorem maximize_profit : argmax (λ p : ℝ, profit p) (λ p, p > 20) = 30 :=
by sorry

end maximize_profit_l567_567826


namespace probability_of_winning_five_tickets_l567_567247

def probability_of_winning_one_ticket := 1 / 10000000
def number_of_tickets_bought := 5

theorem probability_of_winning_five_tickets : 
  (number_of_tickets_bought * probability_of_winning_one_ticket) = 5 / 10000000 :=
by
  sorry

end probability_of_winning_five_tickets_l567_567247


namespace sum_of_fractions_l567_567879

theorem sum_of_fractions :
  (2 / 20 : ℝ) + (4 / 40 : ℝ) + (5 / 50 : ℝ) = 0.3 :=
begin
  sorry
end

end sum_of_fractions_l567_567879


namespace percentage_decrease_l567_567681

theorem percentage_decrease (P : ℝ) :
  (let new_price := 1.40 * P in
   ∃ x : ℝ, new_price * (1 - x / 100) = 1.19 * P ∧ x = 15) :=
by
  let new_price := 1.40 * P
  use 15
  split
  { calc new_price * (1 - 15 / 100)
        = 1.40 * P * (1 - 0.15) : by simp
    ... = 1.40 * P * 0.85 : by rw sub_div
    ... = 1.19 * P : by norm_num },
  { sorry }

end percentage_decrease_l567_567681


namespace orthogonal_vectors_l567_567601

open Real

variables (r s : ℝ)

def a : ℝ × ℝ × ℝ := (5, r, -3)
def b : ℝ × ℝ × ℝ := (-1, 2, s)

theorem orthogonal_vectors
  (orthogonality : 5 * (-1) + r * 2 + (-3) * s = 0)
  (magnitude_condition : 34 + r^2 = 4 * (5 + s^2)) :
  ∃ (r s : ℝ), (2 * r - 3 * s = 5) ∧ (r^2 - 4 * s^2 = -14) :=
  sorry

end orthogonal_vectors_l567_567601


namespace four_students_same_acquaintances_l567_567018

theorem four_students_same_acquaintances :
  ∃ (students : Finset ℕ) (knows : ℕ → Finset ℕ), 
  students.card = 102 ∧
  (∀ s ∈ students, (knows s).card ≥ 68) ∧
  (∃ (n : ℕ), (Finset.filter (λ s, (knows s).card = n) students).card ≥ 4) :=
by
  sorry

end four_students_same_acquaintances_l567_567018


namespace exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012_l567_567872

theorem exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012 :
  ∃ (a b c : ℕ), 
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧ 
    a ∣ (a * b * c + 2012) ∧ b ∣ (a * b * c + 2012) ∧ c ∣ (a * b * c + 2012) :=
by
  sorry

end exists_natural_numbers_gt_10pow10_divisible_by_self_plus_2012_l567_567872


namespace marcy_yellow_marbles_l567_567193

theorem marcy_yellow_marbles (n : ℕ) (h : n = 50) :
  let blue_marbles := n / 5
  let green_marbles := 10
  let red_marbles := 2 * green_marbles
  let y := n - (blue_marbles + green_marbles + red_marbles)
  y = 10 :=
by
  rw [h]
  sorry

end marcy_yellow_marbles_l567_567193


namespace steven_acres_of_grassland_l567_567532

def plowing_rate : ℕ := 10
def mowing_rate : ℕ := 12
def total_days : ℕ := 8
def farmland : ℕ := 55

theorem steven_acres_of_grassland (G : ℕ) 
  (H : G = 24) : 
  (total_days = (ceil (farmland / plowing_rate)) + (G / mowing_rate)) := by
  sorry

end steven_acres_of_grassland_l567_567532


namespace odd_cycle_length_le_3_l567_567915

/--
Given a simple graph with 239 vertices, which is not bipartite and each vertex has degree at least 3,
prove that the smallest k such that each odd cycle has length at most k is 3.
-/
theorem odd_cycle_length_le_3 (G : SimpleGraph (Fin 239)) (h_not_bipartite : ¬G.IsBipartite) 
  (h_degree : ∀ v : Fin 239, 3 ≤ G.degree v) : 
  ∃ k, (∀ C ∈ G.cycles, k ≥ C.length ∧ C.length % 2 = 1 → k = 3) :=
by
  sorry

end odd_cycle_length_le_3_l567_567915


namespace decimal_period_lengths_l567_567795

theorem decimal_period_lengths (p : ℕ) (a : ℕ → ℕ) (n : ℕ)
  (hp_prime : Prime p)
  (hp_not_div_2_5 : p ≠ 2 ∧ p ≠ 5)
  (ha_coprime : ∀ n, Nat.coprime (a n) p):
  ∃ k : ℕ, ∀ n : ℕ, decimal_period_length (a n / p^n) = k * p^(n-1) :=
sorry

end decimal_period_lengths_l567_567795


namespace polyline_distance_A_O_min_polyline_distance_origin_line_l567_567551

structure Point :=
  (x : ℝ)
  (y : ℝ)

def polyline_distance (P Q : Point) : ℝ :=
  (abs (P.x - Q.x)) + (abs (P.y - Q.y))

def A : Point := ⟨-1, 3⟩
def O : Point := ⟨0, 0⟩

def on_line (P : Point) : Prop :=
  2 * P.x + P.y = 2 * real.sqrt 5

theorem polyline_distance_A_O :
  polyline_distance A O = 4 := by
  sorry

theorem min_polyline_distance_origin_line :
  ∃ P : Point, on_line P ∧ polyline_distance O P = real.sqrt 5 := by
  sorry

end polyline_distance_A_O_min_polyline_distance_origin_line_l567_567551


namespace simple_interest_rate_in_paise_l567_567251

-- Define the given conditions and the result to prove
theorem simple_interest_rate_in_paise (SI P : ℝ) (T : ℕ) (h1 : SI = 4.8) (h2 : P = 8) (h3 : T = 12) :
    let R := 100 * (SI / (P * T)) in R = 5 :=
by
  sorry

end simple_interest_rate_in_paise_l567_567251


namespace lloyd_work_day_hours_l567_567187

-- Definitions necessary for the problem statement
def work_day_hours : ℝ := 7.5
def regular_rate : ℝ := 4.5
def overtime_multiplier : ℝ := 2.0
def total_earnings : ℝ := 60.75

-- Definitions of intermediate earnings calculations
def regular_earnings := work_day_hours * regular_rate
def overtime_rate := regular_rate * overtime_multiplier
def overtime_earnings := total_earnings - regular_earnings
def overtime_hours := overtime_earnings / overtime_rate
def total_hours := work_day_hours + overtime_hours

-- Statement of the problem proof
theorem lloyd_work_day_hours : total_hours = 10.5 := 
by sorry

end lloyd_work_day_hours_l567_567187


namespace no_solution_frac_eq_l567_567218

theorem no_solution_frac_eq (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) : 
  3 / x + 6 / (x - 1) - (x + 5) / (x * (x - 1)) ≠ 0 :=
by {
  sorry
}

end no_solution_frac_eq_l567_567218


namespace find_a_l567_567403

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end find_a_l567_567403


namespace minimum_value_proof_l567_567072

noncomputable def minimum_value (x : ℝ) (h : x > 1) : ℝ :=
  (x^2 + x + 1) / (x - 1)

theorem minimum_value_proof : ∃ x : ℝ, x > 1 ∧ minimum_value x (by sorry) = 3 + 2*Real.sqrt 3 :=
sorry

end minimum_value_proof_l567_567072


namespace det_of_new_matrix_l567_567167

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def new_det (a b c d : V) (D : ℝ) : ℝ :=
  let D' := (a + d) ⬝ ((b + 2 • d) × (c + 3 • d)) in
  D' = D + 3 * (a ⬝ (b × d)) + 2 * (a ⬝ (d × c)) + d ⬝ (b × c)

-- Statement of the proof problem
theorem det_of_new_matrix (a b c d : V) (D : ℝ) (hD : D = a ⬝ (b × c)) :
  new_det a b c d D = D + 3 * (a ⬝ (b × d)) + 2 * (a ⬝ (d × c)) + d ⬝ (b × c) :=
by
  sorry

end det_of_new_matrix_l567_567167


namespace incorrect_proposition_B_l567_567081

variables (α β : Plane) (m n : Line)
variables (hαβ : α ≠ β) (hmn : m ≠ n)

theorem incorrect_proposition_B :
  ¬ (∀ (m n : Line) (α β : Plane),
     m || n → α ∩ β = m → (n || α ∧ n || β)) :=
sorry

end incorrect_proposition_B_l567_567081


namespace find_a_l567_567402

theorem find_a (a b : ℤ) (h1 : 4181 * a + 2584 * b = 0) (h2 : 2584 * a + 1597 * b = -1) : a = 1597 :=
by
  sorry

end find_a_l567_567402


namespace delta_eq_l567_567260

-- Definitions of E, Delta, and conditions given in the problem
axiom E : ℕ → Operator
axiom Delta : Operator
axiom I : Operator
axiom C : ℕ → ℕ → ℕ

-- Given conditions
axiom E_def : ∀ n, E n = (Δ + I) ^ n = ∑ k in finset.range (n + 1), C n k * (Δ ^ k)
axiom n_natural : ∀ n, n ∈ ℕ

-- The theorem to prove
theorem delta_eq (n : ℕ) : Δ ^ n = ∑ k in finset.range (n + 1), (-1)^(n - k) * C n k * (E k) :=
sorry

end delta_eq_l567_567260


namespace column_shape_correct_base_radii_correct_l567_567294

noncomputable def column_shape_eqn (P : ℝ) (δ : ℝ) (r : ℝ) (y z : ℝ) : ℝ :=
  (2 * P / (π * δ * r^2)) * real.log ((real.sqrt (y^2 + z^2)) / r)

theorem column_shape_correct :
  ∀ (P δ r x y z : ℝ),
  ∀ (hδ_pos : δ > 0) (hP_pos : P > 0) (hr_pos : r > 0),
  x = column_shape_eqn P δ r y z →
  (x = (2 * P / (π * δ * r^2)) * real.log ((real.sqrt (y^2 + z^2)) / r)) :=
begin
  intros,
  unfold column_shape_eqn,
  sorry,
end

def upper_base_radius (P : ℝ) (pressure : ℝ) : ℝ :=
  real.sqrt (P / (π * pressure))

theorem base_radii_correct :
  upper_base_radius 90000 3000 ≈ 3.09 ∧ 
  ∀ (h : ℝ) (δ : ℝ) (P : ℝ) (r : ℝ) (y : ℝ)
  (h_height : h = 12) (h_δ : δ = 2.5) (hP : P = 90000) (hr : r ≈ 3.09),
  column_shape_eqn P δ r y 0 = h →
  r ≈ 3.24 :=
begin
  unfold upper_base_radius,
  split,
  { sorry, },
  { intros,
    unfold column_shape_eqn,
    sorry, },
end

end column_shape_correct_base_radii_correct_l567_567294


namespace series_ln_factorial_diverges_l567_567873

theorem series_ln_factorial_diverges : ¬ (summable (λ n : ℕ, if n > 1 then 1 / log (n.factorial) else 0)) := 
  sorry

end series_ln_factorial_diverges_l567_567873


namespace sum_of_divisors_143_l567_567776

theorem sum_of_divisors_143 : ∑ d in List.toFinset (List.divisors 143), d = 168 := by
  have prime_factors_143 : 143 = 11 * 13 := by
    calc 143 = 11 * 13 : by norm_num
  sorry

end sum_of_divisors_143_l567_567776


namespace irrational_D_l567_567840

def A := Real.sqrt 4
def B := (3.14 : ℝ)
def C := Real.cbrt (-27)
def D := 5 * Real.pi

theorem irrational_D : ¬ Rational 5 * Real.pi :=
by sorry

end irrational_D_l567_567840


namespace multiple_of_savings_l567_567351

theorem multiple_of_savings (P : ℝ) (h : P > 0) :
  let monthly_savings := (1 / 4) * P
  let monthly_non_savings := (3 / 4) * P
  let total_yearly_savings := 12 * monthly_savings
  ∃ M : ℝ, total_yearly_savings = M * monthly_non_savings ∧ M = 4 := 
by
  sorry

end multiple_of_savings_l567_567351


namespace age_difference_l567_567848

-- Define the conditions and the statement
theorem age_difference :
  ∀ (benjie_age : ℝ) (factor : ℝ),
    (factor = 5) →
    (benjie_age = 28) →
    let margo_age := benjie_age / factor in
    let margo_future_age := 3 * margo_age in
    let benjie_future_age := benjie_age + (margo_future_age - margo_age) in
    (benjie_future_age - margo_future_age) = 22.4 :=
by
  intros benjie_age factor h1 h2
  let margo_age := benjie_age / factor
  let margo_future_age := 3 * margo_age
  let benjie_future_age := benjie_age + (margo_future_age - margo_age)
  sorry

end age_difference_l567_567848


namespace smallest_n_l567_567598

-- Conditions and the statement to prove
theorem smallest_n (n : ℕ) (h1 : n > 1) :
  (∃ (a : Fin n → ℤ), (∑ (i : Fin n), a i = 2005) ∧ (∏ (i : Fin n), a i = 2005)) ↔ n = 5 := 
sorry

end smallest_n_l567_567598


namespace octagon_circumference_l567_567820

noncomputable def radius (side_length : ℝ) : ℝ :=
  side_length / (2 * real.sin (real.pi / 8))

noncomputable def circumference (side_length : ℝ) : ℝ :=
  2 * real.pi * radius side_length

theorem octagon_circumference {side_length : ℝ} (h : side_length = 5) :
  circumference side_length = 5 * real.pi / real.sin (real.pi / 8) :=
by
  sorry

end octagon_circumference_l567_567820


namespace g_diff_l567_567239

variable {R : Type} [LinearOrderedField R]

def linear_function (g : R → R) : Prop :=
  ∀ (a b : R), g(a + b) = g(a) + g(b) ∧ g(a * b) = a * g(b)

-- Define g function and its properties
noncomputable def g (d : R) : R := sorry

axiom linear_g : linear_function g
axiom diff_g (d : R) : g(d + 1) - g(d) = 5

theorem g_diff : g(2 : R) - g(7) = -25 :=
by
  sorry

end g_diff_l567_567239


namespace zeros_of_f_eq_one_l567_567481

noncomputable def f (x a b : ℝ) : ℝ := a^x + x - b

theorem zeros_of_f_eq_one (a b : ℝ) 
  (h1 : 2^a = 3) 
  (h2 : 3^b = 2) : 
  ∃! x : ℝ, f x a b = 0 :=
begin
  sorry
end

end zeros_of_f_eq_one_l567_567481


namespace largest_integer_less_100_leaves_remainder_4_l567_567456

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l567_567456


namespace slope_of_symmetric_line_l567_567930

theorem slope_of_symmetric_line (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 - 6 * x + 6 * y + 14 = 0 → ax + 4 * y - 6 = 0) →
  a = 6 →
  ∃ m : ℝ, m = -3/2 :=
by {
  intro h_symmetry,
  intro h_a,
  use -3/2,
  sorry
}

end slope_of_symmetric_line_l567_567930


namespace triangle_ABC_area_l567_567000

open Real

-- Define points A, B, and C
structure Point :=
  (x: ℝ)
  (y: ℝ)

def A : Point := ⟨-1, 2⟩
def B : Point := ⟨8, 2⟩
def C : Point := ⟨6, -1⟩

-- Function to calculate the area of a triangle given vertices A, B, and C
noncomputable def triangle_area (A B C : Point) : ℝ := 
  abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y)) / 2

-- The statement to be proved
theorem triangle_ABC_area : triangle_area A B C = 13.5 :=
by
  sorry

end triangle_ABC_area_l567_567000


namespace carpet_needed_in_sqyards_l567_567345

-- Define the dimensions of the room and the cabinet
def room_length_ft : ℝ := 18
def room_width_ft : ℝ := 9
def cabinet_length_ft : ℝ := 3
def cabinet_width_ft : ℝ := 2

-- Define the areas
def room_area_sqft := room_length_ft * room_width_ft
def cabinet_area_sqft := cabinet_length_ft * cabinet_width_ft
def carpet_area_sqft := room_area_sqft - cabinet_area_sqft
def carpet_area_sqyards := carpet_area_sqft / 9

theorem carpet_needed_in_sqyards :
  carpet_area_sqyards = 17.33 :=
by
  sorry

end carpet_needed_in_sqyards_l567_567345


namespace largest_int_less_than_100_remainder_4_l567_567416

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l567_567416


namespace sum_of_divisors_of_143_l567_567768

noncomputable def sum_divisors (n : ℕ) : ℕ :=
  ∑ k in (nat.divisors n).toFinset, k

theorem sum_of_divisors_of_143 : sum_divisors 143 = 168 :=
by sorry

end sum_of_divisors_of_143_l567_567768


namespace total_cost_of_new_movie_l567_567569

noncomputable def previous_movie_length_hours : ℕ := 2
noncomputable def new_movie_length_increase_percent : ℕ := 60
noncomputable def previous_movie_cost_per_minute : ℕ := 50
noncomputable def new_movie_cost_per_minute_factor : ℕ := 2 

theorem total_cost_of_new_movie : 
  let new_movie_length_hours := previous_movie_length_hours + (previous_movie_length_hours * new_movie_length_increase_percent / 100)
  let new_movie_length_minutes := new_movie_length_hours * 60
  let new_movie_cost_per_minute := previous_movie_cost_per_minute * new_movie_cost_per_minute_factor
  let total_cost := new_movie_length_minutes * new_movie_cost_per_minute
  total_cost = 19200 := 
by
  sorry

end total_cost_of_new_movie_l567_567569


namespace average_rainfall_feb_1983_l567_567540

theorem average_rainfall_feb_1983 (total_rainfall : ℕ) (days_in_february : ℕ) (hours_per_day : ℕ) 
  (H1 : total_rainfall = 789) (H2 : days_in_february = 28) (H3 : hours_per_day = 24) : 
  total_rainfall / (days_in_february * hours_per_day) = 789 / 672 :=
by
  sorry

end average_rainfall_feb_1983_l567_567540


namespace minimum_weights_required_l567_567350

-- Definition of the input conditions
def nuts : List ℝ := [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
def unlimited_weights : Set ℝ := {x | 1 ≤ x ∧ x ≤ 1000}
def number_of_nuts : ℕ := 15

-- Define the theorem to be proven
theorem minimum_weights_required (H1 : ∀ n ∈ nuts, n ∈ set.Icc 50 64)
                                 (H2 : ∀ w ∈ unlimited_weights, w ∈ set.Icc 1 1000)
                                 (unlimited_weighings : ℕ → ℕ)
                                 : ∃ m : ℕ, m = 1 := 
sorry

end minimum_weights_required_l567_567350


namespace problem_solution_l567_567880

theorem problem_solution (n : ℕ) (hn : n = 1 ∨ (n % 2 = 0)) :
  ∀ (R : polynomial ℝ), ∃ (a b : ℝ) (k l : ℕ), (0 ≤ k ∧ k ≤ n) ∧ (0 ≤ l ∧ l ≤ n) ∧
  ∀ x : ℝ, (R.eval x + a * x^k + b * x^l) ≠ 0 :=
sorry

end problem_solution_l567_567880


namespace find_f_g_2_l567_567314

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x^2 - 6

theorem find_f_g_2 : f (g 2) = 1 := 
  by
  -- Proof goes here
  sorry

end find_f_g_2_l567_567314


namespace glide_reflection_translation_vector_length_l567_567478

-- Define the triangle and necessary constructs
structure Triangle :=
  (A B C : ℝ × ℝ)
  (angle_A angle_B angle_C : ℝ) -- Angles at A, B, and C
  (R : ℝ) -- Circumradius

-- Define reflections, symmetries, and compositions
def reflect_across_line (p : ℝ × ℝ) (line : ℝ × ℝ) : ℝ × ℝ := sorry -- Implementation of reflection across a line
def S_AC (triangle : Triangle) (p : ℝ × ℝ) : ℝ × ℝ := reflect_across_line p (triangle.A, triangle.C)
def S_AB (triangle : Triangle) (p : ℝ × ℝ) : ℝ × ℝ := reflect_across_line p (triangle.A, triangle.B)
def S_BC (triangle : Triangle) (p : ℝ × ℝ) : ℝ × ℝ := reflect_across_line p (triangle.B, triangle.C)

def S (triangle : Triangle) : ℝ × ℝ → ℝ × ℝ :=
  S_AC triangle ∘ S_AB triangle ∘ S_BC triangle

-- The mathematical statement to be proved
theorem glide_reflection_translation_vector_length (triangle : Triangle) :
  -- Given conditions
  let α := triangle.angle_A,
      β := triangle.angle_B,
      γ := triangle.angle_C,
      R := triangle.R,
      S := S triangle in
  -- Assert translation vector length
  ∃ (v : ℝ × ℝ), (∥v∥ = 2 * R * sin α * sin β * sin γ) ∧ (∃ (is_glide_reflection : Prop), is_glide_reflection = (∃ (line : ℝ × ℝ), ∀ (p : ℝ × ℝ), S p = p + line)) := 
sorry

end glide_reflection_translation_vector_length_l567_567478


namespace largest_int_less_than_100_mod_6_eq_4_l567_567439

theorem largest_int_less_than_100_mod_6_eq_4 : 
  ∃ x, x < 100 ∧ x % 6 = 4 ∧ ∀ y, y < 100 ∧ y % 6 = 4 → y ≤ x :=
begin
  sorry
end

end largest_int_less_than_100_mod_6_eq_4_l567_567439


namespace largest_integer_with_remainder_l567_567425

theorem largest_integer_with_remainder (n : ℕ) (h : 6 * n + 4 < 100) : 6 * 15 + 4 = 94 :=
by
  have h1 : n ≤ 15 := by sorry
  have h2 : n = 15 := by sorry
  exact calc
    6 * 15 + 4 = 90 + 4 : by rw [mul_comm 6 15]
    ... = 94 : by norm_num

end largest_integer_with_remainder_l567_567425


namespace largest_n_unique_k_l567_567288

theorem largest_n_unique_k : ∃! (n : ℕ), ∃ (k : ℤ),
  (7 / 16 : ℚ) < (n : ℚ) / (n + k : ℚ) ∧ (n : ℚ) / (n + k : ℚ) < (8 / 17 : ℚ) ∧ n = 112 := 
sorry

end largest_n_unique_k_l567_567288


namespace proof_angle_l567_567793

-- Definitions for the hyperbola and its foci
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 45 = 1

def distance (P Q : ℝ × ℝ) := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

variables (F1 F2 P : ℝ × ℝ)
variables (hF1F2 : distance F1 F2 = 14)
variables (d_12 d_21 : ℝ)

-- Conditions of the problem:
def condition1 := hyperbola P.1 P.2
def condition2 := distance P F2 = d_12
def condition3 := distance P F1 = d_21
def condition4 := d_21 - d_12 = 4
def condition5 := 2 * d_21 = d_12 + 14

theorem proof_angle :
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 → 
  ∃ θ, θ = 2 * real.pi / 3 :=
sorry

end proof_angle_l567_567793


namespace students_only_english_l567_567543

theorem students_only_english
  (total_students : ℕ)
  (students_both : ℕ)
  (students_german : ℕ)
  (H : total_students = 50)
  (H1 : students_both = 18)
  (H2 : students_german = 34) :
  ∃ students_english : ℕ, students_english = 16 :=
by
  let students_german_only := students_german - students_both
  have h1 : students_german_only = 16 := by
    rw [H1, H2]
    exact Nat.sub_sub_self H1 H2

  let students_english := total_students - students_both - students_german_only
  have h2 : students_english = 16 := by
    rw [H, h1]
    exact Nat.sub_sub_self H

  exact ⟨students_english, h2⟩

end students_only_english_l567_567543


namespace sum_of_divisors_143_l567_567775

theorem sum_of_divisors_143 : ∑ d in List.toFinset (List.divisors 143), d = 168 := by
  have prime_factors_143 : 143 = 11 * 13 := by
    calc 143 = 11 * 13 : by norm_num
  sorry

end sum_of_divisors_143_l567_567775


namespace find_13_points_within_radius_one_l567_567545

theorem find_13_points_within_radius_one (points : Fin 25 → ℝ × ℝ)
  (h : ∀ i j k : Fin 25, min (dist (points i) (points j)) (min (dist (points i) (points k)) (dist (points j) (points k))) < 1) :
  ∃ (subset : Finset (Fin 25)), subset.card = 13 ∧ ∃ (center : ℝ × ℝ), ∀ i ∈ subset, dist (points i) center < 1 :=
  sorry

end find_13_points_within_radius_one_l567_567545


namespace circumference_in_scientific_notation_l567_567666

noncomputable def circumference_m : ℝ := 4010000

noncomputable def scientific_notation (m: ℝ) : Prop :=
  m = 4.01 * 10^6

theorem circumference_in_scientific_notation : scientific_notation circumference_m :=
by
  sorry

end circumference_in_scientific_notation_l567_567666


namespace find_numbers_l567_567036

theorem find_numbers (a b c : ℝ) (x y z: ℝ) (h1 : x + y = z + a) (h2 : x + z = y + b) (h3 : y + z = x + c) :
    x = (a + b - c) / 2 ∧ y = (a - b + c) / 2 ∧ z = (-a + b + c) / 2 := by
  sorry

end find_numbers_l567_567036


namespace max_f_5_value_l567_567941

noncomputable def f (x : ℝ) : ℝ := x ^ 2 + 2 * x

noncomputable def f_1 (x : ℝ) : ℝ := f x
noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0       => x -- Not used, as n starts from 1
  | (n + 1) => f (f_n n x)

noncomputable def max_f_5 : ℝ := 3 ^ 32 - 1

theorem max_f_5_value : ∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f_n 5 x ≤ max_f_5 :=
by
  intro x hx
  have := hx
  -- The detailed proof would go here,
  -- but for the statement, we end with sorry.
  sorry

end max_f_5_value_l567_567941


namespace problem1_problem2_l567_567853

-- Problem 1: Prove that (a/(a - b)) + (b/(b - a)) = 1
theorem problem1 (a b : ℝ) (h : a ≠ b) : (a / (a - b)) + (b / (b - a)) = 1 := 
sorry

-- Problem 2: Prove that (a^2 / (b^2 * c)) * (- (b * c^2) / (2 * a)) / (a / b) = -c
theorem problem2 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a^2 / (b^2 * c)) * (- (b * c^2) / (2 * a)) / (a / b) = -c :=
sorry

end problem1_problem2_l567_567853


namespace largest_integer_less_100_leaves_remainder_4_l567_567459

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end largest_integer_less_100_leaves_remainder_4_l567_567459
