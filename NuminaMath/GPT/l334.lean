import Mathlib

namespace normal_distribution_prob_l334_334217

open MeasureTheory

noncomputable def normal_cdf : ℝ → ℝ := sorry -- Placeholder for the standard normal CDF

variable {σ : ℝ} (σ_pos : 0 < σ)

theorem normal_distribution_prob (h : normal_cdf (1 / σ) = 2 / 3) :
  normal_cdf (1 * σ⁻¹) = 2 / 3 →
  (Π (σ > 0), normal_cdf (-1 / σ) = 1 / 3) :=
sorry

end normal_distribution_prob_l334_334217


namespace least_five_digit_perfect_square_and_cube_l334_334430

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334430


namespace shortest_distance_to_y_axis_is_3_l334_334403

-- Define the parabola and the fixed length of the line segment
def parabola (x y : ℝ) := y^2 = 8 * x
def fixed_length (A B : ℝ × ℝ) := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 100

-- Define the midpoint of AB
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the shortest distance from P to the y-axis
def mid_distance_to_y_axis (P : ℝ × ℝ) : ℝ := abs P.1

-- The theorem statement
theorem shortest_distance_to_y_axis_is_3 (A B P : ℝ × ℝ) 
  (hA : parabola A.1 A.2) 
  (hB : parabola B.1 B.2) 
  (hAB : fixed_length A B) 
  (hP : P = midpoint A B) :
  mid_distance_to_y_axis P = 3 :=
sorry

end shortest_distance_to_y_axis_is_3_l334_334403


namespace least_five_digit_perfect_square_and_cube_l334_334479

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334479


namespace chess_tournament_scores_l334_334128

theorem chess_tournament_scores :
    ∃ (A D V S : ℝ),
    A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S ∧
    A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 ∧
    A > D ∧ D > V ∧ V > S ∧
    (∃ (wins_A wins_S : ℕ), wins_A = wins_S) :=
begin
    sorry
end

end chess_tournament_scores_l334_334128


namespace trigonometric_identity_l334_334061

theorem trigonometric_identity :
  cos (60 * real.pi / 180) + sin (60 * real.pi / 180) - sqrt (3 / 4) + (tan (45 * real.pi / 180))⁻¹ = 3 / 2 :=
by
  sorry

end trigonometric_identity_l334_334061


namespace median_of_sequence_l334_334068

theorem median_of_sequence : 
  let N := ∑ i in (Finset.range 151), i
  ∃ median : ℕ, median = 106 := 
by
  let N := ∑ i in (Finset.range 151), i
  have : N = 11325 := by
    -- Calculate sum of integers from 1 to 150
    sorry
  let median_pos := (N + 1) / 2 
  have : median_pos = 5663 := by
    -- Calculate the position of the median in the sequence
    rw [this]
    sorry
  have median_val := 
    (Finset.range 151).find (λ n, ∑ i in (Finset.range (n+1)), i ≥ 5663)
  have h_median : median_val = 106 := by
    -- Verify cumulative sum condition with correct value
    sorry
  exact ⟨median_val, h_median⟩

end median_of_sequence_l334_334068


namespace angle_parallel_minimum_positive_period_interval_monotonic_increase_l334_334240

-- Definitions for vectors and conditions
def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x)
def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def parallel (v w : ℝ × ℝ) : Prop := ∃ k : ℝ, v = (k * w.1, k * w.2)

-- Proof for the angle x given vectors are parallel
theorem angle_parallel (x : ℝ) (h_parallel: parallel (m x) (n x)) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  x = Real.pi / 6 ∨ x = 0 := sorry

-- Definition for the dot product
def f (x : ℝ) : ℝ := (Real.sqrt 3 * Real.sin x * Real.cos x) + (Real.sin x)^2

-- Proof for minimum positive period of f(x)
theorem minimum_positive_period (T : ℝ) :
  (∀ x, f (x + T) = f x) ∧ T > 0 → T = Real.pi := sorry

-- Proof for the interval of monotonic increase for f(x)
theorem interval_monotonic_increase (x : ℝ) (k : ℤ) :
  (∀ x₁ x₂, x₁ ∈ [k * Real.pi - Real.pi / 6, k * Real.pi + Real.pi / 3] → 
   x₁ < x₂ → f x₁ < f x₂) ↔
  ∀ x, x ∈ [k * Real.pi - Real.pi / 6, k * Real.pi + Real.pi / 3] := sorry

end angle_parallel_minimum_positive_period_interval_monotonic_increase_l334_334240


namespace remainder_mod_5_l334_334395

theorem remainder_mod_5 :
  let a := 1492
  let b := 1776
  let c := 1812
  let d := 1996
  (a * b * c * d) % 5 = 4 :=
by
  sorry

end remainder_mod_5_l334_334395


namespace line_through_point_conditions_l334_334709

-- Definitions for points and lines
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Definition of a line passing through a given point
def passes_through (l : Line) (p : Point) : Prop := l.a * p.x + l.b * p.y + l.c = 0

-- Definition of distance of a line from the origin
def distance_from_origin (l : Line) : ℝ := 
  (abs l.c) / (sqrt (l.a^2 + l.b^2))

-- Definition of line slope
def slope (l : Line) : Option ℝ :=
  if l.b = 0 then none else some ((-l.a) / l.b)

-- Lean 4 statement for the mathematically equivalent proof problem
theorem line_through_point_conditions (P : Point)
  (hP : P = {x := 2, y := -1}) :
  (∃ l1 : Line, passes_through l1 P ∧ distance_from_origin l1 = 2 ∧ (l1 = {a := 1, b := 0, c := -2} ∨ l1 = {a := 3, b := -4, c := -10})) ∧ 
  (∃ l2 : Line, passes_through l2 P ∧ distance_from_origin l2 = sqrt 5 ∧ l2 = {a := 2, b := -1, c := -5}) := 
sorry

end line_through_point_conditions_l334_334709


namespace total_matches_in_2006_world_cup_l334_334793

-- Define relevant variables and conditions
def teams := 32
def groups := 8
def top2_from_each_group := 16

-- Calculate the number of matches in Group Stage
def matches_in_group_stage :=
  let matches_per_group := 6
  matches_per_group * groups

-- Calculate the number of matches in Knockout Stage
def matches_in_knockout_stage :=
  let first_round_matches := 8
  let quarter_final_matches := 4
  let semi_final_matches := 2
  let final_and_third_place_matches := 2
  first_round_matches + quarter_final_matches + semi_final_matches + final_and_third_place_matches

-- Total number of matches
theorem total_matches_in_2006_world_cup : matches_in_group_stage + matches_in_knockout_stage = 64 := by
  sorry

end total_matches_in_2006_world_cup_l334_334793


namespace geometric_configuration_l334_334978

/-- 
Given four distinct points \(A\), \(B\), \(C\), and \(D\) in space, if every plane or spherical surface containing \(A\) and \(B\) intersects with every distinct plane or sphere that \(C\) and \(D\) lie on, then the points \(A\), \(B\), \(C\), and \(D\) either lie on a circle or a straight line, and \(A\) and \(B\) separate \(C\) and \(D\). 
-/
theorem geometric_configuration (A B C D : Point) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (h_intersect : ∀ (plane1 plane2 : Plane), (A ∈ plane1 ∧ B ∈ plane1) → (C ∈ plane2 ∧ D ∈ plane2) → (plane1 ≠ plane2) → (plane1 ∩ plane2).Nonempty) :
    (Collinear A B C ∨ Collinear A B D ∨ lies_on_circle A B C D) :=
sorry

end geometric_configuration_l334_334978


namespace side_length_octahedron_l334_334069

-- Definitions for unit cube vertices and octahedron vertices placements
def vertex_A := (0, 0, 0)
def vertex_A' := (1, 1, 1)

-- Vertices of the octahedron placed at 1/3 distances from A
def oct_vtx_1_3_1 := (1 / 3 : ℝ, 0, 0)
def oct_vtx_1_3_2 := (0, 1 / 3 : ℝ, 0)
def oct_vtx_1_3_3 := (0, 0, 1 / 3 : ℝ)

-- Vertices of the octahedron placed at 2/3 distances from A'
def oct_vtx_2_3_1 := (1, 1 - 2 / 3 : ℝ, 1)
def oct_vtx_2_3_2 := (1 - 2 / 3 : ℝ, 1, 1)
def oct_vtx_2_3_3 := (1, 1, 1 - 2 / 3 : ℝ)

-- Function to calculate the Euclidean distance between two points in 3D space
def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2).sqrt

-- Side length calculation
def octahedron_side_length : ℝ :=
  distance oct_vtx_1_3_1 oct_vtx_1_3_2

-- Theorem proving the side length of the octahedron
theorem side_length_octahedron : octahedron_side_length = (Real.sqrt 2) / 3 :=
by
  -- Add the necessary detailed proof here
  sorry

end side_length_octahedron_l334_334069


namespace prod_eq_65536_l334_334064

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 9)

theorem prod_eq_65536 : (∏ k in Finset.range 6, ∏ j in Finset.range 8, (omega^(3 * (j + 1)) - omega^(2 * (k + 1)))) = 65536 :=
by
  sorry

end prod_eq_65536_l334_334064


namespace remove_green_balls_l334_334015

theorem remove_green_balls (total_balls green_balls yellow_balls x : ℕ) 
  (h1 : total_balls = 600)
  (h2 : green_balls = 420)
  (h3 : yellow_balls = 180)
  (h4 : green_balls = 70 * total_balls / 100)
  (h5 : yellow_balls = total_balls - green_balls)
  (h6 : (green_balls - x) = 60 * (total_balls - x) / 100) :
  x = 150 := 
by {
  -- sorry placeholder for proof.
  sorry
}

end remove_green_balls_l334_334015


namespace probability_of_region_l334_334592

-- Definition of the bounds
def bounds (x y : ℝ) : Prop := 0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 8

-- Definition of the region where x + y <= 5
def region (x y : ℝ) : Prop := x + y ≤ 5

-- The proof statement
theorem probability_of_region : 
  (∃ (x y : ℝ), bounds x y ∧ region x y) →
  ∃ (p : ℚ), p = 3/8 :=
by sorry

end probability_of_region_l334_334592


namespace total_profit_is_6600_l334_334048

variable (x : ℝ) -- C's investment
variable (P : ℝ) -- Total Profit

-- Conditions
def B_investment : ℝ := (2/3) * x
def A_investment : ℝ := 3 * B_investment
def total_investment : ℝ := A_investment + B_investment + x
def B_share : ℝ := 1200

-- Proof statement
theorem total_profit_is_6600 (h : B_share = ((2/3) * x / total_investment) * P) : P = 6600 := by
  sorry

end total_profit_is_6600_l334_334048


namespace points_on_ellipse_l334_334247

-- Define the properties of our ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition that the line segments are perpendicular
def perpendicular_to_foci (a b : ℝ) (x y c : ℝ) : Prop :=
  let f1 := (c, 0)
  let f2 := (-c, 0)
  (x, y) = f1 ∨ (x, y) = f2 ∧ x^2 / a^2 + y^2 / b^2 = 1

-- Prove the final statement
theorem points_on_ellipse :
  ∃ p1 p2 : ℝ × ℝ, 
    (ellipse 2.sqrt2 2 p1.fst p1.snd ∧ perpendicular_to_foci 2.sqrt2 2 p1.fst p1.snd 2) ∧
    (ellipse 2.sqrt2 2 p2.fst p2.snd ∧ perpendicular_to_foci 2.sqrt2 2 p2.fst p2.snd 2) ∧
    p1 ≠ p2 :=
by
  -- Proof goes here
  sorry

end points_on_ellipse_l334_334247


namespace probability_midpoint_in_T_l334_334824

open Nat

def num_points_in_T : Nat := 4 * 5 * 6
def valid_pairs_x : Nat := 8
def valid_pairs_y : Nat := 11
def valid_pairs_z : Nat := 18
def valid_midpoint_pairs : Nat := 8 * 11 * 18 - num_points_in_T
def total_combinations : Nat := binom 120 2
def gcd_4_183 : Nat := 4.gcd 183 -- Verify GCD
def probability_fraction : Rational := ⟨4, 183⟩ -- Fraction in simplest form
def p_and_q_sum : Nat := 4 + 183

theorem probability_midpoint_in_T :
    8 * 11 * 18 - num_points_in_T = 1560 ∧
    total_combinations = 7140 ∧
    gcd_4_183 = 1 ∧ -- Ensure the fraction is simplified
    probability_fraction.num + probability_fraction.denom = 187 :=
by {
  sorry
}

end probability_midpoint_in_T_l334_334824


namespace jackets_sold_after_noon_l334_334993

theorem jackets_sold_after_noon 
  (x y : ℕ) 
  (h1 : x + y = 214) 
  (h2 : 31.95 * x + 18.95 * y = 5108.30) : 
  y = 133 := 
by 
  -- Proof is skipped
  sorry

end jackets_sold_after_noon_l334_334993


namespace problem_part_1_problem_part_2_problem_part_3_l334_334695

noncomputable def f : ℝ → ℝ := sorry

axiom additivity (a b : ℝ) : f(a + b) = f(a) + f(b)
axiom f_at_4 : f(4) = 1/4
axiom f_positive (x : ℝ) (h : 0 < x) : 0 < f(x)

theorem problem_part_1 (a : ℝ) :
  f(0) = 0 ∧ f(8) = 1/2 :=
sorry

theorem problem_part_2 (x1 x2 : ℝ) (h : x1 < x2) :
  f(x1) < f(x2) :=
sorry

theorem problem_part_3 (x : ℝ) :
  f(x - 3) - f(3 * x - 5) ≤ 1/2 ↔ x ≥ -3 :=
sorry

end problem_part_1_problem_part_2_problem_part_3_l334_334695


namespace greatest_five_consecutive_odd_integers_l334_334959

theorem greatest_five_consecutive_odd_integers (A B C D E : ℤ) (x : ℤ) 
  (h1 : B = x + 2) 
  (h2 : C = x + 4)
  (h3 : D = x + 6)
  (h4 : E = x + 8)
  (h5 : A + B + C + D + E = 148) :
  E = 33 :=
by {
  sorry -- proof not required
}

end greatest_five_consecutive_odd_integers_l334_334959


namespace sum_of_solutions_l334_334533

theorem sum_of_solutions : 
  let integer_solutions := { x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 } in
  ∑ x in integer_solutions, x = 24 := 
sorry

end sum_of_solutions_l334_334533


namespace probability_cos_equation_l334_334806

/-- 
In the set {1, 2, 3, 4, ..., 10}, the probability that a randomly selected element
exactly satisfies the equation cos(30° * x) = 1/2 is 1/5.
-/
theorem probability_cos_equation : 
  let S := { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } in
  (∃ x ∈ S, Real.cos (30 * Real.pi / 180 * x) = 1 / 2) →
  (card { x ∈ S | Real.cos (30 * Real.pi / 180 * x) = 1 / 2 }).toRat / (card S).toRat = 1 / 5 := 
by
  sorry

end probability_cos_equation_l334_334806


namespace arithmetic_seq_problem_l334_334703

-- Conditions and definitions for the arithmetic sequence
def arithmetic_seq (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n+1) = a_n n + d

def sum_seq (a_n S_n : ℕ → ℝ) : Prop :=
  ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2

def T_plus_K_eq_19 (T K : ℕ) : Prop :=
  T + K = 19

-- The given problem to prove
theorem arithmetic_seq_problem (a_n S_n : ℕ → ℝ) (d : ℝ) (h1 : d > 0)
  (h2 : arithmetic_seq a_n d) (h3 : sum_seq a_n S_n)
  (h4 : ∀ T K, T_plus_K_eq_19 T K → S_n T = S_n K) :
  ∃! n, a_n n - S_n n ≥ 0 := sorry

end arithmetic_seq_problem_l334_334703


namespace max_value_of_quadratic_l334_334525

theorem max_value_of_quadratic :
  ∃ r : ℝ, ∀ x : ℝ, -3 * x^2 + 30 * x + 24 ≤ 99 :=
by
  existsi (5 : ℝ)
  intros x
  have h : -3 * (x - 5) ^ 2 ≤ 0 := by sorry -- complete the proof later
  calc
    -3 * x^2 + 30 * x + 24
        = -3 * (x - 5)^2 + 99 : by sorry -- complete the proof later
    ... ≤ 99 : by linarith


end max_value_of_quadratic_l334_334525


namespace least_five_digit_perfect_square_and_cube_l334_334458

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334458


namespace number_of_perfect_squares_l334_334756

theorem number_of_perfect_squares (a b : ℤ) (h1 : a^2 = 100) (h2 : b^2 = 289) : 
  ∃ n : ℕ, ∀ m, (10 ≤ m ∧ m ≤ 17 → m ≠ n) ∧ (count (λ n, 100 ≤ n^2 ∧ n^2 ≤ 300) = 8) := 
sorry

end number_of_perfect_squares_l334_334756


namespace least_five_digit_perfect_square_and_cube_l334_334472

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334472


namespace james_driving_speed_l334_334288

theorem james_driving_speed
  (distance : ℝ)
  (total_time : ℝ)
  (stop_time : ℝ)
  (driving_time : ℝ)
  (speed : ℝ)
  (h1 : distance = 360)
  (h2 : total_time = 7)
  (h3 : stop_time = 1)
  (h4 : driving_time = total_time - stop_time)
  (h5 : speed = distance / driving_time) :
  speed = 60 := by
  -- Here you would put the detailed proof.
  sorry

end james_driving_speed_l334_334288


namespace angle_of_inclination_of_tangent_at_point_l334_334082

theorem angle_of_inclination_of_tangent_at_point :
  ∀ (x y : ℝ), y = (1 / 3) * x^3 - 2 → 
  ((∃ (x0: ℝ), x0 = 1) ∧ (∃ (y0: ℝ), y0 = -5/3)) → 
  ∃ θ : ℝ, θ = 45 := by
  sorry

end angle_of_inclination_of_tangent_at_point_l334_334082


namespace regular_polygon_area_l334_334040

theorem regular_polygon_area (n : ℕ) (R : ℝ)
  (h : 1 / 2 * n * R^2 * real.sin (2 * real.pi / n) = 4 * R^2) : n = 24 :=
sorry

end regular_polygon_area_l334_334040


namespace frustum_slant_height_l334_334032

-- The setup: we are given specific conditions for a frustum resulting from cutting a cone
variable {r : ℝ} -- represents the radius of the upper base of the frustum
variable {h : ℝ} -- represents the slant height of the frustum
variable {h_removed : ℝ} -- represents the slant height of the removed cone

-- The given conditions
def upper_base_radius : ℝ := r
def lower_base_radius : ℝ := 4 * r
def slant_height_removed_cone : ℝ := 3

-- The proportion derived from similar triangles
def proportion (h r : ℝ) := (h / (4 * r)) = ((h + 3) / (5 * r))

-- The main statement: proving the slant height of the frustum is 9 cm
theorem frustum_slant_height (r : ℝ) (h : ℝ) (hr : proportion h r) : h = 9 :=
sorry

end frustum_slant_height_l334_334032


namespace nat_divides_power_difference_l334_334841

theorem nat_divides_power_difference (n : ℕ) : n ∣ 2 ^ (2 * n.factorial) - 2 ^ n.factorial := by
  sorry

end nat_divides_power_difference_l334_334841


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334496

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334496


namespace least_five_digit_perfect_square_and_cube_l334_334453

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334453


namespace symmetric_line_equation_l334_334364

theorem symmetric_line_equation (x y : ℝ) (h₁ : x + y + 1 = 0) : (2 - x) + (4 - y) - 7 = 0 :=
by
  sorry

end symmetric_line_equation_l334_334364


namespace construct_triangle_from_medians_l334_334634

theorem construct_triangle_from_medians
    (s_a s_b s_c : ℝ)
    (h1 : s_a + s_b > s_c)
    (h2 : s_a + s_c > s_b)
    (h3 : s_b + s_c > s_a) :
    ∃ (a b c : ℝ), 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ 
    (∃ (median_a median_b median_c : ℝ), 
        median_a = s_a ∧ 
        median_b = s_b ∧ 
        median_c = s_c) :=
sorry

end construct_triangle_from_medians_l334_334634


namespace sum_of_integer_solutions_l334_334541

theorem sum_of_integer_solutions :
  (∑ x in ({ x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 }.to_finset), x) = 18 :=
by
  sorry

end sum_of_integer_solutions_l334_334541


namespace interest_rate_part2_l334_334885

noncomputable def total_investment : ℝ := 3400
noncomputable def part1 : ℝ := 1300
noncomputable def part2 : ℝ := total_investment - part1
noncomputable def rate1 : ℝ := 0.03
noncomputable def total_interest : ℝ := 144
noncomputable def interest1 : ℝ := part1 * rate1
noncomputable def interest2 : ℝ := total_interest - interest1
noncomputable def rate2 : ℝ := interest2 / part2

theorem interest_rate_part2 : rate2 = 0.05 := sorry

end interest_rate_part2_l334_334885


namespace verify_option_B_l334_334054

def binary_eq_decimal : Prop := 
  nat.of_digits 2 [1, 1, 0, 1] = 13

theorem verify_option_B : binary_eq_decimal :=
  by
    sorry

end verify_option_B_l334_334054


namespace sum_of_integer_solutions_l334_334536

theorem sum_of_integer_solutions :
  (∑ x in { x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 }, x) = 18 :=
by
  sorry

end sum_of_integer_solutions_l334_334536


namespace convert_500_to_base5_l334_334075

def base10_to_base5 (n : ℕ) : ℕ :=
  -- A function to convert base 10 to base 5 would be defined here
  sorry

theorem convert_500_to_base5 : base10_to_base5 500 = 4000 := 
by 
  -- The actual proof would go here
  sorry

end convert_500_to_base5_l334_334075


namespace part1_part2_l334_334564

def f (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2
def g (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
def F (x : ℝ) : ℝ := g x / f x

def h (x : ℝ) : ℝ := 4 / 3 - 2 / (2^(x - 1) + 1)

def H (xs : List ℝ) : ℝ := xs.sum_map (λ x => h x)

theorem part1 (λ : ℝ) (x : ℝ) (cond : 0 < x ∧ x < Real.log 3) 
    (eq_cond : F (f (2 * x)) + F (2 * λ * g x - 5) = 0) :
    λ ∈ set.Ioi (1 / 6) := sorry

theorem part2 (xs : List ℝ) (n : ℤ) (cond : n > 0 ∧ xs.length = 2 * n ∧
    xs.all (λ x => 0 < x ∧ x < 2)) :
    (∃ x : ℝ, x ∈ xs ∧ F (2 * x) / F x ≥ H (xs)) ↔ n ∈ {1, 2, 3} := sorry

end part1_part2_l334_334564


namespace probability_of_winning_second_lawsuit_l334_334057

theorem probability_of_winning_second_lawsuit
  (P_W1 P_L1 P_W2 P_L2 : ℝ)
  (h1 : P_W1 = 0.30)
  (h2 : P_L1 = 0.70)
  (h3 : P_L1 * P_L2 = P_W1 * P_W2 + 0.20)
  (h4 : P_L2 = 1 - P_W2) :
  P_W2 = 0.50 :=
by
  sorry

end probability_of_winning_second_lawsuit_l334_334057


namespace find_a_l334_334215

theorem find_a (a : ℝ) :
  (∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1 + A.2 - a = 0) ∧ (B.1 + B.2 - a = 0) ∧ (A.1^2 + A.2^2 = 2) ∧ (B.1^2 + B.2^2 = 2) ∧ 
  |2 * ⟨A.1,A.2⟩ - 3 * ⟨B.1,B.2⟩| = |2 * ⟨A.1,A.2⟩ + 3 * ⟨B.1,B.2⟩| ) →
  a = √2 ∨ a = -√2 := 
sorry

end find_a_l334_334215


namespace least_five_digit_perfect_square_and_cube_l334_334426

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334426


namespace bowling_ball_weight_l334_334122

theorem bowling_ball_weight (b k : ℝ) (h1 : 5 * b = 3 * k) (h2 : 4 * k = 120) : b = 18 :=
by
  sorry

end bowling_ball_weight_l334_334122


namespace find_pyramid_height_l334_334783

-- We define a regular triangular pyramid and its properties
structure Pyramid :=
  (a : ℝ)  -- side length of the base
  (angle_apothem_face : ℝ := π / 4)  -- angle between slant height and a lateral face

noncomputable def height_of_pyramid (p : Pyramid) : ℝ :=
  p.a * real.sqrt (6) / 6

-- We specify that for our given pyramid, the calculated height equals the desired result
theorem find_pyramid_height (p : Pyramid) : height_of_pyramid p = p.a * real.sqrt (6) / 6 :=
by
  -- Skipping the proof, just asserting the equality
  sorry

end find_pyramid_height_l334_334783


namespace count_divisors_g_1000_l334_334126

def g (n : ℕ) : ℕ :=
  3^n

theorem count_divisors_g_1000 :
  ∀ d : ℕ, d > 0 → d ≤ 3^1000 → (∃ k : ℕ, d = 3^k) → (∑ d in (finset.range (3^1000 + 1)).filter (λ d, d ∣ g 1000), 1) = 1001 :=
by sorry

end count_divisors_g_1000_l334_334126


namespace dance_classes_total_cost_l334_334868

def cost_per_class : Type := ℝ

def classesPerWeek (classCost : cost_per_class) (per_week : ℕ) (weeks : ℕ) : cost_per_class :=
  per_week * weeks * classCost

def classesEveryNWeeks (classCost : cost_per_class) (total_weeks : ℕ) (n_weeks : ℕ) : cost_per_class :=
  (total_weeks / n_weeks) * classCost

def salsaAdditionalDiscount (classCost : cost_per_class) (discountCost : cost_per_class) : cost_per_class :=
  classCost + discountCost

def totalDanceCost : cost_per_class :=
  let hip_hop := classesPerWeek 10.50 3 6
  let ballet := classesPerWeek 12.25 2 6
  let jazz := classesPerWeek 8.75 1 6
  let salsa := classesEveryNWeeks 15 6 2 + salsaAdditionalDiscount 0 12
  let contemporary := classesEveryNWeeks 10 6 3
  hip_hop + ballet + jazz + salsa + contemporary

theorem dance_classes_total_cost :
  totalDanceCost = 465.50 :=
by
  sorry

end dance_classes_total_cost_l334_334868


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334499

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334499


namespace range_of_f_value_of_f_B_l334_334736

-- Define function f
def f (x : ℝ) : ℝ := 2 * sqrt 3 * (sin x) * (cos x) - 3 * (sin x)^2 - (cos x)^2 + 3

-- Condition given for the transformation to 2 * sin (2x + π/6) + 1
theorem range_of_f : set_of (λ y : ℝ, ∃ x ∈ Icc 0 (real.pi / 2), f x = y) = Icc 0 3 :=
sorry

-- Triangles and given conditions
variables {a b c A B C : ℝ}
hypothesis h1 : b = sqrt 3 * a
hypothesis h2 : sin (2 * A + C) = 2 * sin A * (1 + cos(A + C)) 

-- Proving the value of f(B)
theorem value_of_f_B (h_AB_triangle : 0 < A ∧ A < real.pi ∧ 0 < B ∧ B < real.pi ∧ 0 < C ∧ C < real.pi ∧ A + B + C = real.pi) : 
  f B = 2 := 
sorry

end range_of_f_value_of_f_B_l334_334736


namespace number_of_bent_strips_is_odd_l334_334653

/--
Given a cube with dimensions 9x9x9 that is partitioned into unit squares, and 
the surface area of this cube is covered perfectly by 243 strips of dimensions 2x1 
without overlapping. Prove that the number of bent strips (strips that bend around edges) 
is odd.
-/
theorem number_of_bent_strips_is_odd :
  let n := 9
  let strips := 243
  let face_unit_squares := n * n
  let surface_area := 6 * face_unit_squares
  let strip_area := 2
  ∃ (bent_strips : ℕ), 
    strips = (surface_area / strip_area) ∧ 
    (odd bent_strips) := 
sorry

end number_of_bent_strips_is_odd_l334_334653


namespace bounded_region_area_l334_334918

theorem bounded_region_area : 
  let eq := λ (x y : ℝ), y^2 + 4*x*y + 64*|x| + 9*y = 576 in
  ∃ (a : ℝ), 
  (∀ x y : ℝ, eq x y → a = 1536) := 
begin
  sorry
end

end bounded_region_area_l334_334918


namespace parabola_shift_mn_value_l334_334394

def shift_parabola (y : ℝ → ℝ) (up : ℝ) (left : ℝ) : (ℝ → ℝ) :=
  λ x, y (x + left) + up

theorem parabola_shift_mn_value :
  let y := λ x : ℝ, x^2 in
  let shifted_y := shift_parabola y 2 3 in
  let new_parabola := λ x, x^2 + (6 * x) + 11 in
  ∃ m n : ℝ, (shifted_y = new_parabola) ∧ (m * n = 66) :=
by
  sorry

end parabola_shift_mn_value_l334_334394


namespace range_of_a_in_third_quadrant_l334_334271

def pointInThirdQuadrant (x y : ℝ) := x < 0 ∧ y < 0

theorem range_of_a_in_third_quadrant (a : ℝ) (M : ℝ × ℝ) 
  (hM : M = (-1, a-1)) (hThirdQuad : pointInThirdQuadrant M.1 M.2) : 
  a < 1 :=
by
  sorry

end range_of_a_in_third_quadrant_l334_334271


namespace intersection_point_on_line_ac_l334_334203

noncomputable theory

-- Definitions of points and line segments in the context of the problem
variables {A B C D E F G H Q : Type*}

-- Defining the conditions
def is_parallelogram {A B C D : Type*} (AB : A → B → Type*) (BC : B → C → Type*) (CD : C → D → Type*) (DA : D → A → Type*) :=
∀ (A B C D : Type*), AB A B → BC B C → CD C D → DA D A

def intersects_lines_at_points {A B C D E F G H : Type*} (AB : A → B → Type*) (BC : B → C → Type*) (CD : C → D → Type*) (DA : D → A → Type*)
(E : Type*) (F : Type*) (G : Type*) (H : Type*) :=
∀ (l : Type*), ¬ (l = A ∨ l = B ∨ l = C ∨ l = D) → (l intersects AB at E ∧ l intersects BC at F ∧ l intersects CD at G ∧ l intersects DA at H)

def circles_intersect_at_point (E F C G H : Type*) (Q : Type*) :=
circulate Q E F C ∧ circulate Q G H C

-- Main theorem statement in Lean 4
theorem intersection_point_on_line_ac (AB : A → B → Type*) (BC : B → C → Type*) (CD : C → D → Type*) (DA : D → A → Type*)
(EFC GHC : Type*) (Q A C : Type*) [is_parallelogram AB BC CD DA]
(intersects_lines_at_points AB BC CD DA E F G H) (circles_intersect_at_point E F C G H Q) :
  Q lies_on_line AC :=
sorry

end intersection_point_on_line_ac_l334_334203


namespace sum_of_integer_solutions_l334_334540

theorem sum_of_integer_solutions :
  (∑ x in ({ x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 }.to_finset), x) = 18 :=
by
  sorry

end sum_of_integer_solutions_l334_334540


namespace fraction_sum_simplest_form_l334_334966

theorem fraction_sum_simplest_form (n d : ℕ) (h : n = 63) (h_d : d = 117) :
    (let gcd := Nat.gcd n d in (n / gcd) + (d / gcd) = 20) :=
by
  -- Let n = 63 and d = 117
  let n := 63
  let d := 117
  -- Calculate the gcd of n and d
  let gcd := Nat.gcd n d
  -- Express n and d in simplest form and take the sum
  have : (n / gcd) + (d / gcd) = 20 by sorry
  exact this

end fraction_sum_simplest_form_l334_334966


namespace coefficient_of_x5_expansion_l334_334908

noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_of_x5_expansion :
  let T := (λ r : ℕ, binomial_coeff 7 r * (x^3)^(7 - r) * (1/x)^r)
  in T 4 = 35 :=
  sorry

end coefficient_of_x5_expansion_l334_334908


namespace law_of_sines_incorrect_statement_l334_334545

theorem law_of_sines_incorrect_statement (A B C : ℝ) (a b c r : ℝ) (hA : a = 2 * r * sin A) (hB : b = 2 * r * sin B) (hC : c = 2 * r * sin C) :
  ¬(a = b ↔ sin (2 * A) = sin (2 * B)) := 
sorry

end law_of_sines_incorrect_statement_l334_334545


namespace no_7_edges_edges_greater_than_5_l334_334001

-- Define the concept of a convex polyhedron in terms of its edges and faces.
structure ConvexPolyhedron where
  V : ℕ    -- Number of vertices
  E : ℕ    -- Number of edges
  F : ℕ    -- Number of faces
  Euler : V - E + F = 2   -- Euler's characteristic

-- Define properties of convex polyhedron

-- Part (a) statement: A convex polyhedron cannot have exactly 7 edges.
theorem no_7_edges (P : ConvexPolyhedron) : P.E ≠ 7 :=
sorry

-- Part (b) statement: A convex polyhedron can have any number of edges greater than 5 and different from 7.
theorem edges_greater_than_5 (n : ℕ) (h : n > 5) (h2 : n ≠ 7) : ∃ P : ConvexPolyhedron, P.E = n :=
sorry

end no_7_edges_edges_greater_than_5_l334_334001


namespace numLinesTangentToCircles_eq_2_l334_334331

noncomputable def lineTangents (A B : Point) (dAB rA rB : ℝ) : ℕ :=
  if dAB < rA + rB then 2 else 0

theorem numLinesTangentToCircles_eq_2
  (A B : Point) (dAB rA rB : ℝ)
  (hAB : dAB = 4) (hA : rA = 3) (hB : rB = 2) :
  lineTangents A B dAB rA rB = 2 := by
  sorry

end numLinesTangentToCircles_eq_2_l334_334331


namespace common_chord_eq_l334_334237

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 2*y - 40 = 0

-- Define the statement to prove
theorem common_chord_eq (x y : ℝ) : circle1 x y ∧ circle2 x y → 2*x + y - 5 = 0 :=
sorry

end common_chord_eq_l334_334237


namespace sum_of_integer_solutions_l334_334534

theorem sum_of_integer_solutions :
  (∑ x in { x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 }, x) = 18 :=
by
  sorry

end sum_of_integer_solutions_l334_334534


namespace complex_magnitude_power_l334_334111

theorem complex_magnitude_power (z : ℂ) (h : z = 2 + 2 * (√3) * complex.I) (n : ℕ) (h_n : n = 6) :
  complex.abs (z^n) = 4096 :=
by
  sorry

end complex_magnitude_power_l334_334111


namespace professors_seating_l334_334414

/-- There are 13 chairs arranged in a single row. Four professors (Alpha, Beta, Gamma, and Delta) and 
nine students need to sit such that each professor is seated between at least one student. Professors 
cannot occupy the first or last position. Prove that the number of ways in which the four professors 
can choose their chairs is 3024. -/
theorem professors_seating : 
  ∃ c : ℕ, 
  (c = (∑ x in (Finset.range 13).powerset.filter (λ s, 
    s.card = 4 ∧ ∀ x ∈ s, 2 ≤ x ∧ x ≤ 11 ∧ 
    (∀ p q ∈ s, p ≠ q → (p + 1 < q ∨ q + 1 < p)) ), 
    1) * factoral 4)) ∧ c = 3024 := 
by
  sorry

end professors_seating_l334_334414


namespace ratio_girls_to_boys_l334_334555

-- Definitions of the conditions
def numGirls : ℕ := 10
def numBoys : ℕ := 20

-- Statement of the proof problem
theorem ratio_girls_to_boys : (numGirls / Nat.gcd numGirls numBoys) = 1 ∧ (numBoys / Nat.gcd numGirls numBoys) = 2 :=
by
  sorry

end ratio_girls_to_boys_l334_334555


namespace product_of_solutions_abs_eq_l334_334274

theorem product_of_solutions_abs_eq (x : ℝ) (h : |x - 5| + 4 = 7) : x * (if x = 8 then 2 else 8) = 16 :=
by {
  sorry
}

end product_of_solutions_abs_eq_l334_334274


namespace neg_P_4_of_P_implication_and_neg_P_5_l334_334195

variable (P : ℕ → Prop)

theorem neg_P_4_of_P_implication_and_neg_P_5
  (h1 : ∀ k : ℕ, 0 < k → (P k → P (k+1)))
  (h2 : ¬ P 5) :
  ¬ P 4 :=
by
  sorry

end neg_P_4_of_P_implication_and_neg_P_5_l334_334195


namespace cake_piece_volume_l334_334598

theorem cake_piece_volume (h : ℝ) (d : ℝ) (n : ℕ) (V_piece : ℝ) : 
  h = 1/2 ∧ d = 16 ∧ n = 8 → V_piece = 4 * Real.pi :=
by
  sorry

end cake_piece_volume_l334_334598


namespace exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero_l334_334086

theorem exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero :
  ∃ f : ℤ[X], f.degree = 2 ∧ ∀ x : ℝ, f.eval (f.eval x) = 0 → x = (Real.sqrt 3) := sorry

end exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero_l334_334086


namespace expected_value_of_fair_8_sided_die_l334_334424

-- Define the outcomes of the fair 8-sided die
def outcomes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the probability of each outcome for a fair die
def prob (n : ℕ) : ℚ := 1 / 8

-- Calculate the expected value of the outcomes
noncomputable def expected_value : ℚ :=
  (outcomes.map (λ x => prob x * x)).sum

-- State the theorem that the expected value is 4.5
theorem expected_value_of_fair_8_sided_die : expected_value = 4.5 :=
  sorry

end expected_value_of_fair_8_sided_die_l334_334424


namespace brother_growth_is_one_l334_334897

-- Define measurements related to Stacy's height.
def Stacy_previous_height : ℕ := 50
def Stacy_current_height : ℕ := 57

-- Define the condition that Stacy's growth is 6 inches more than her brother's growth.
def Stacy_growth := Stacy_current_height - Stacy_previous_height
def Brother_growth := Stacy_growth - 6

-- Prove that Stacy's brother grew 1 inch.
theorem brother_growth_is_one : Brother_growth = 1 :=
by
  sorry

end brother_growth_is_one_l334_334897


namespace number_of_two_digit_numbers_l334_334421

theorem number_of_two_digit_numbers
  (学 : ℕ)
  (习 : ℕ)
  (h1 : 学 ≠ 习)
  (h2 : 1 ≤ 学 ∧ 学 ≤ 9)
  (h3 : 1 ≤ 习 ∧ 习 ≤ 9)
  (h4 : ∃ (p : ℕ), p = (1111 * 学) * (1111 * 习) ∧ p / 10^6 % 10 = 学 ∧ p % 10 = 学)
  : ∃ n : ℕ, n = 3 :=
by
  sorry

end number_of_two_digit_numbers_l334_334421


namespace domain_log2_x_add_one_l334_334114

-- Define the function y = log2(x+1)
def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2
def f (x : ℝ) : ℝ := log_base_2 (x + 1)

-- State the theorem about the domain of the function
theorem domain_log2_x_add_one {x : ℝ} : x > -1 ↔ ∃ y : ℝ, f x = y := sorry

end domain_log2_x_add_one_l334_334114


namespace conic_sections_are_parabolas_l334_334366

theorem conic_sections_are_parabolas (x y : ℝ) :
  y^6 - 9*x^6 = 3*y^3 - 1 → ∃ k : ℝ, (y^3 - 1 = k * 3 * x^3 ∨ y^3 = -k * 3 * x^3 + 1) := by
  sorry

end conic_sections_are_parabolas_l334_334366


namespace calc_h_one_l334_334819

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 6
noncomputable def g (x : ℝ) : ℝ := Real.exp (f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- the final theorem that we are proving
theorem calc_h_one : h 1 = 3 * Real.exp 26 - 14 * Real.exp 13 + 21 := by
  sorry

end calc_h_one_l334_334819


namespace train_speed_correct_l334_334607

def train_length : ℝ := 250  -- length of the train in meters
def time_to_pass : ℝ := 18  -- time to pass a tree in seconds
def speed_of_train_km_hr : ℝ := 50  -- speed of the train in km/hr

theorem train_speed_correct :
  (train_length / time_to_pass) * (3600 / 1000) = speed_of_train_km_hr :=
by
  sorry

end train_speed_correct_l334_334607


namespace perfect_square_factors_of_ten_thousand_l334_334242

-- Definitions based on conditions
def ten_thousand : ℕ := 10000
def factorization : ten_thousand = 2^4 * 5^4 := by norm_num

-- Theorem with proof problem statement
theorem perfect_square_factors_of_ten_thousand : 
  (∃ n : ℕ, ten_thousand = 2^(2 * n) * 5^(2 * n)) → 
  ∃ count : ℕ, count = 9 := 
sorry

end perfect_square_factors_of_ten_thousand_l334_334242


namespace reflection_matrix_squared_identity_l334_334831

def Q : Matrix (Fin 2) (Fin 2) ℝ := -- reflection matrix over vector (4, -2)
  let a : ℝ := 4
  let b : ℝ := -2
  let n2 := a^2 + b^2
  ![![ (a^2 - b^2) / n2 , 2*a*b / n2],
    ![2*a*b / n2, (b^2 - a^2) / n2]]

theorem reflection_matrix_squared_identity : Q ⬝ Q = (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end reflection_matrix_squared_identity_l334_334831


namespace find_cos_abe_l334_334283

noncomputable def cos_abe : ℚ :=
  -- Conditions of the problem
  let AB : ℚ := 4 in
  let AC : ℚ := 7 in
  let BC : ℚ := 9 in
  let cos_angle_ABC := (AB^2 + BC^2 - AC^2) / (2 * AB * BC) in
  let angle_ABC := Real.arccos cos_angle_ABC in
  let cos_angle_A_B_E_half := Real.sqrt ((1 + cos_angle_ABC) / 2) in
  cos_angle_A_B_E_half

theorem find_cos_abe :
  cos_abe = Real.sqrt (5/6) := sorry

end find_cos_abe_l334_334283


namespace equilateral_is_peculiar_rt_triangle_is_peculiar_peculiar_rt_triangle_ratio_l334_334335

-- Definition of a peculiar triangle.
def is_peculiar_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2

-- Problem 1: Proving an equilateral triangle is a peculiar triangle
theorem equilateral_is_peculiar (a : ℝ) : is_peculiar_triangle a a a :=
sorry

-- Problem 2: Proving the case when b is the hypotenuse in Rt△ABC makes it peculiar
theorem rt_triangle_is_peculiar (a b c : ℝ) (ha : a = 5 * Real.sqrt 2) (hc : c = 10) : 
  is_peculiar_triangle a b c ↔ b = Real.sqrt (c^2 + a^2) :=
sorry

-- Problem 3: Proving the ratio of the sides in a peculiar right triangle is 1 : √2 : √3
theorem peculiar_rt_triangle_ratio (a b c : ℝ) (hc : c^2 = a^2 + b^2) (hpeculiar : is_peculiar_triangle a c b) :
  (b = Real.sqrt 2 * a) ∧ (c = Real.sqrt 3 * a) :=
sorry

end equilateral_is_peculiar_rt_triangle_is_peculiar_peculiar_rt_triangle_ratio_l334_334335


namespace sqrt_evaluation_l334_334650

theorem sqrt_evaluation (a b c : ℤ) (h₁ : c = 11) (h₂ : a * b = 12) (h₃ : a^2 + b^2 * c = 83) : 
  (∀ a b c : ℤ, c = 11 → a * b = 12 → a^2 + b^2 * c = 83 → sqrt (83 + 24 * sqrt 11) = a + b * sqrt c) :=
by
  intros a b c h₁ h₂ h₃
  sorry

end sqrt_evaluation_l334_334650


namespace artist_painting_time_l334_334615

theorem artist_painting_time (hours_per_week : ℕ) (weeks : ℕ) (total_paintings : ℕ) :
  hours_per_week = 30 → weeks = 4 → total_paintings = 40 →
  ((hours_per_week * weeks) / total_paintings) = 3 := by
  intros h_hours h_weeks h_paintings
  sorry

end artist_painting_time_l334_334615


namespace star_calculation_l334_334074

-- Define the operation '*' via the given table
def star_table : Matrix (Fin 5) (Fin 5) (Fin 5) :=
  ![
    ![0, 1, 2, 3, 4],
    ![1, 0, 4, 2, 3],
    ![2, 3, 1, 4, 0],
    ![3, 4, 0, 1, 2],
    ![4, 2, 3, 0, 1]
  ]

def star (a b : Fin 5) : Fin 5 := star_table a b

-- Prove (3 * 5) * (2 * 4) = 3
theorem star_calculation : star (star 2 4) (star 4 1) = 2 := by
  sorry

end star_calculation_l334_334074


namespace integer_with_exactly_12_integers_to_its_left_l334_334815

theorem integer_with_exactly_12_integers_to_its_left :
  let initial_list := List.range' 1 20
  let first_half := initial_list.take 10
  let second_half := initial_list.drop 10
  let new_list := second_half ++ first_half
  new_list.get! 12 = 3 :=
by
  let initial_list := List.range' 1 20
  let first_half := initial_list.take 10
  let second_half := initial_list.drop 10
  let new_list := second_half ++ first_half
  sorry

end integer_with_exactly_12_integers_to_its_left_l334_334815


namespace exists_quadratic_poly_with_integer_coeffs_l334_334095

theorem exists_quadratic_poly_with_integer_coeffs (α : ℝ) :
  (∃ (a b c : ℤ), ∀ x : ℝ, (λ x : ℝ, ↑a * x^2 + ↑b * x + ↑c) (↑(λ x : ℝ, ↑a * x^2 + ↑b * x + ↑c) (α)) = 0) :=
  sorry

end exists_quadratic_poly_with_integer_coeffs_l334_334095


namespace ratio_apples_simplified_l334_334336

variable (n : ℕ) (m : ℕ) (k : ℕ)
variable (a : n = 45) (b : m = 9) (c : k = 27)

theorem ratio_apples_simplified (n m k : ℕ) (a : n = 45) (b : m = 9) (c : k = 27) : 
  (n / n.gcd m / n.gcd k) = 5 ∧ (m / n.gcd m / n.gcd k) = 1 ∧ (k / n.gcd m / n.gcd k) = 3 := 
by
  sorry

end ratio_apples_simplified_l334_334336


namespace number_of_three_digit_numbers_l334_334009

theorem number_of_three_digit_numbers : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let num_three_digit_numbers := 900
  let num_without_repeats := 9 * 9 * 8
  let num_with_repeats := num_three_digit_numbers - num_without_repeats
  num_with_repeats = 252 :=
by
  sorry

end number_of_three_digit_numbers_l334_334009


namespace cosine_angle_AB_AC_l334_334113

noncomputable def vector (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

noncomputable def dot_product (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2 + v₁.3 * v₂.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def cosine_angle (v₁ v₂ : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v₁ v₂ / (magnitude v₁ * magnitude v₂)

def A : ℝ × ℝ × ℝ := (2, -4, 6)
def B : ℝ × ℝ × ℝ := (0, -2, 4)
def C : ℝ × ℝ × ℝ := (6, -8, 10)

noncomputable def AB : ℝ × ℝ × ℝ := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
noncomputable def AC : ℝ × ℝ × ℝ := (C.1 - A.1, C.2 - A.2, C.3 - A.3)

theorem cosine_angle_AB_AC :
  cosine_angle AB AC = -1 := by
  sorry

end cosine_angle_AB_AC_l334_334113


namespace smallest_perimeter_arithmetic_sequence_triangle_l334_334055

noncomputable def heron_area (a b c : ℕ) : ℚ :=
  (1/4 : ℚ) * real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))

theorem smallest_perimeter_arithmetic_sequence_triangle :
  ∃ (a b c : ℕ), (a < b ∧ b < c) ∧ (2 * b = a + c) ∧ 
  (triangle_is_acute a b c) ∧ 
  (∃ (area : ℚ), area = heron_area a b c ∧ (area : ℚ).denom = 1) ∧
  ((a + b + c) = 18) := 
begin
  sorry
end

end smallest_perimeter_arithmetic_sequence_triangle_l334_334055


namespace num_divisors_32481_l334_334243

theorem num_divisors_32481 : 
  (finset.filter (λ n : ℕ, n ∣ 32481) (finset.range 10)).card = 3 :=
by
  -- Use properties of divisors to construct proofs for each step
  sorry

end num_divisors_32481_l334_334243


namespace min_teams_needed_l334_334020

theorem min_teams_needed (total_employees : ℕ) (max_members_per_team : ℕ) (h1 : total_employees = 36)
    (h2 : max_members_per_team = 12) : (total_employees / max_members_per_team) = 3 :=
by
  rw [h1, h2]
  norm_num

end min_teams_needed_l334_334020


namespace unique_correct_proposition_l334_334053

-- Conditions
def exterior_angle_greater (α β γ : ℝ) (hαβγ : α + β + γ = 180) : Prop :=
  ∀ (θ : ℝ), θ = 180 - β ∨ θ = 180 - γ → θ > α

def median_divides_equal_area (T : Type) [HasArea T] (t : T) (m : Medians T) : Prop :=
  ∀ (half_t : T), 2 * area half_t = area t

def congruent_if_sas (T : Type) [HasAngle T] (t₁ t₂ : T) : Prop :=
  ∀ (a₁ a₂ b₁ b₂ : ℝ) (θ₁ θ₂ : T.Angle), 
    (a₁ = a₂) ∧ (b₁ = b₂) ∧ (θ₁ = θ₂) → T.Congruent t₁ t₂

def altitude_inside_triangle (T : Type) [HasVertices T] [HasAltitudes T] (t : T) : Prop :=
  ∀ (v₁ v₂ v₃ : T.Vertex) (alt₁ alt₂ alt₃ : T.Altitude),
    (alt₁ inside t) ∧ (alt₂ inside t) ∧ (alt₃ inside t) → ¬ T.Obtuse t

-- Proof problem
theorem unique_correct_proposition :
  (median_divides_equal_area T t m) ∧ ¬ (exterior_angle_greater α β γ hαβγ) ∧ ¬ (congruent_if_sas T t₁ t₂) ∧ ¬ (altitude_inside_triangle T t) :=
sorry

end unique_correct_proposition_l334_334053


namespace inverse_17_mod_1021_l334_334066

theorem inverse_17_mod_1021 : ∃ x : ℕ, (17 * x) % 1021 = 1 ∧ x = 961 := by
  use 961
  split
  · sorry -- Proof that (17 * 961) % 1021 = 1
  · rfl

end inverse_17_mod_1021_l334_334066


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334492

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334492


namespace solution_set_of_inequality_l334_334665

theorem solution_set_of_inequality (x : ℝ) : x * (1 - x) > 0 ↔ 0 < x ∧ x < 1 :=
sorry

end solution_set_of_inequality_l334_334665


namespace tile_prices_minimize_payment_l334_334017

-- Problem 1: Unit prices of red and blue tiles
theorem tile_prices (a b : ℕ) 
  (h1 : 4000 * a + 5400 * b = 86000) 
  (h2 : 8000 * a + 3500 * b = 99000) :
  a = 8 ∧ b = 10 :=
sorry

-- Problem 2: Minimizing the payment for 12000 tiles
theorem minimize_payment (x : ℕ) 
  (h1 : 4000 ≤ x) 
  (h2 : x ≤ 6000)
  (tile_cost : ℕ → ℕ := λ x, if 4000 ≤ x ∧ x < 5000 then 76800 + 3.6 * x else if 5000 ≤ x ∧ x ≤ 6000 then 76800 + 2.6 * x else 0) :
  ∃ (x : ℕ), 5000 ≤ x ∧ x ≤ 6000 ∧ tile_cost x = 89800 :=
sorry

end tile_prices_minimize_payment_l334_334017


namespace least_five_digit_perfect_square_and_cube_l334_334475

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334475


namespace max_points_to_form_equilateral_triangles_l334_334198

open Set

def equilateral_triangle (A B C : ℝ × ℝ) : Prop :=
 ∥B - A∥ = ∥C - A∥ ∧ ∥B - A∥ = ∥C - B∥

def all_subsets_equilateral (M : Set (ℝ × ℝ)) : Prop :=
  ∀ (A B C : ℝ × ℝ), {A, B, C} ⊆ M → (equilateral_triangle A B C)

theorem max_points_to_form_equilateral_triangles (M : Set (ℝ × ℝ)) (n : ℕ) :
  all_subsets_equilateral M ∧ card M = n → n ≤ 3 :=
sorry

end max_points_to_form_equilateral_triangles_l334_334198


namespace MN_length_correct_l334_334999

open Real

noncomputable def MN_segment_length (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : ℝ :=
  sqrt (a * b)

theorem MN_length_correct (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ (MN : ℝ), MN = MN_segment_length a b h1 h2 :=
by
  use sqrt (a * b)
  exact rfl

end MN_length_correct_l334_334999


namespace area_of_overlap_length_segment_AB_l334_334417

variables {m n : ℝ}

-- Given conditions
axiom h1 : m + n = 5 -- Given m + n = 5
axiom h2 : BQ = AK ∧ AR = BP -- BQ = AK and AR = BP
axiom h3 : AL = BL ∧ BN = AN -- AL = BL and BN = AN
axiom h4 : BQ = AR -- Since they are equivalent to m
axiom h5 : AL = BN -- Since they are equivalent to n

theorem area_of_overlap (h1 : m + n = 5) (m_value : m = 2.4) : 
  let overlap_area := m in overlap_area = 2.4 := by
  sorry

theorem length_segment_AB (h1 : m + n = 5) (m_value : m = 2.4) (n_value : n = 2.6) : 
  let AB := sqrt (2 * n^2 - 2 * m * n) in AB = sqrt(26) / 5 := by
  sorry

end area_of_overlap_length_segment_AB_l334_334417


namespace largest_error_in_circle_area_l334_334295

theorem largest_error_in_circle_area (d : ℝ) (error_percent : ℝ) (A : ℝ) : 
  d = 30 → error_percent = 0.3 → A = (Real.pi * (d / 2)^2) → 
  let d_min := d * (1 - error_percent),
      d_max := d * (1 + error_percent),
      A_min := Real.pi * (d_min / 2)^2,
      A_max := Real.pi * (d_max / 2)^2 in
  max ((A - A_min) / A * 100) ((A_max - A) / A * 100) = 69 :=
sorry

end largest_error_in_circle_area_l334_334295


namespace actual_average_height_l334_334265

theorem actual_average_height (average_height : ℝ) (num_students : ℕ)
  (incorrect_heights actual_heights : Fin 3 → ℝ)
  (h_avg : average_height = 165)
  (h_num : num_students = 50)
  (h_incorrect : incorrect_heights 0 = 150 ∧ incorrect_heights 1 = 175 ∧ incorrect_heights 2 = 190)
  (h_actual : actual_heights 0 = 135 ∧ actual_heights 1 = 170 ∧ actual_heights 2 = 185) :
  (average_height * num_students 
   - (incorrect_heights 0 + incorrect_heights 1 + incorrect_heights 2) 
   + (actual_heights 0 + actual_heights 1 + actual_heights 2))
   / num_students = 164.5 :=
by
  -- proof steps here
  sorry

end actual_average_height_l334_334265


namespace least_five_digit_perfect_square_and_cube_l334_334511

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334511


namespace simplify_and_evaluate_expression_l334_334338

variable (x y : ℝ)
variable (h1 : x = 1)
variable (h2 : y = Real.sqrt 2)

theorem simplify_and_evaluate_expression : 
  (x + 2 * y) ^ 2 - x * (x + 4 * y) + (1 - y) * (1 + y) = 7 := by
  sorry

end simplify_and_evaluate_expression_l334_334338


namespace outfits_count_l334_334546

-- Definitions of various clothing counts
def numRedShirts : ℕ := 7
def numGreenShirts : ℕ := 3
def numPants : ℕ := 8
def numBlueShoes : ℕ := 5
def numRedShoes : ℕ := 5
def numGreenHats : ℕ := 10
def numRedHats : ℕ := 6

-- Statement of the theorem based on the problem description
theorem outfits_count :
  (numRedShirts * numPants * numBlueShoes * numGreenHats) + 
  (numGreenShirts * numPants * (numBlueShoes + numRedShoes) * numRedHats) = 4240 := 
by
  -- No proof required, only the statement is needed
  sorry

end outfits_count_l334_334546


namespace digit_sum_divisible_by_5_count_digit_sum_divisible_by_5_l334_334755

def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem digit_sum_divisible_by_5 {n : ℕ} (h : 1 ≤ n ∧ n ≤ 1997) :
  ∃ k, (k = n ∧ (digit_sum k) % 5 = 0) :=
sorry

theorem count_digit_sum_divisible_by_5 :
  (Finset.filter (λ n, digit_sum n % 5 = 0) (Finset.range 1998)).card = 399 :=
sorry

end digit_sum_divisible_by_5_count_digit_sum_divisible_by_5_l334_334755


namespace pyramid_volume_is_correct_l334_334038

open Real

-- Defining the regular octagon and the conditions of the problem
structure RegularOctagon (V : Type) :=
(vertices : V → Prop)

-- Defining the conditions of the equilateral triangle PAE
def is_equilateral_triangle (A B C : V → ℝ) : Prop :=
  ∀ (p1 p2 p3 : V), (A p1 = 10 ∧ B p2 = 10 ∧ C p3 = 10)

-- Defining the pyramid structure
structure Pyramid (B : RegularOctagon V) (P : V) :=
(vertex : P)
(base : B)

-- Definition to calculate the volume of a pyramid
def pyramid_volume (B : RegularOctagon ℝ) (P : ℝ) : ℝ :=
  1/3 * (8 * (1/2 * (5 * sqrt 2) * (5 * sqrt 2)) * (5 * sqrt 3))

-- Statement of the proof problem
theorem pyramid_volume_is_correct (B : RegularOctagon ℝ) (P : ℝ)
  (h_base_area : 8 * (1/2 * (5 * sqrt 2) * (5 * sqrt 2)) = 400)
  (h_height : (sqrt 3 / 2) * 10 = 5 * sqrt 3)
  (h_triangle : is_equilateral_triangle (λ x, 10) (λ x, 10) (λ x, 10)) :
  pyramid_volume B P = 2000 * sqrt 3 / 3 :=
  by sorry

end pyramid_volume_is_correct_l334_334038


namespace max_sum_value_l334_334675

noncomputable def maxSum (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : ℤ :=
  i + j + k

theorem max_sum_value (i j k : ℤ) (h : i^2 + j^2 + k^2 = 2011) : 
  maxSum i j k h ≤ 77 :=
  sorry

end max_sum_value_l334_334675


namespace check_conditions_l334_334202

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) := a + (n - 1) * d

noncomputable def sum_of_first_n_terms (a d : ℤ) (n : ℕ) := n * a + (n * (n - 1) / 2) * d

theorem check_conditions {a d : ℤ}
  (S6 S7 S5 : ℤ)
  (h1 : S6 = sum_of_first_n_terms a d 6)
  (h2 : S7 = sum_of_first_n_terms a d 7)
  (h3 : S5 = sum_of_first_n_terms a d 5)
  (h : S6 > S7 ∧ S7 > S5) :
  d < 0 ∧
  sum_of_first_n_terms a d 11 > 0 ∧
  sum_of_first_n_terms a d 13 < 0 ∧
  sum_of_first_n_terms a d 9 > sum_of_first_n_terms a d 3 := 
sorry

end check_conditions_l334_334202


namespace math_problem_l334_334791

theorem math_problem (ABC : Triangle) (I I_a D M P Q R : Point)
  (h1 : IsIncenter I ABC) 
  (h2 : IsAExcenter I_a ABC)
  (h3 : IsMidpoint D (arc_not_containing A B C (circumcircle ABC)))
  (h4 : IsMidpoint M B C)
  (h5 : OnRay IM P ∧ Dist IM = Dist MP)
  (h6 : Intersection DP MI_a Q)
  (h7 : Parallel AR DP)
  (h8 : ratio_eq AI_a AI 9) :
  let m := 2, n := 9 in m + n = 11 := 
begin
  sorry
end

end math_problem_l334_334791


namespace equilateral_triangle_intersection_l334_334356

theorem equilateral_triangle_intersection
  (a b c : ℝ) (hc : c^2 = a^2 + b^2) :
  ∃ d : ℝ, d ≈ 5.09 :=
by
  unfold ≈
  sorry

end equilateral_triangle_intersection_l334_334356


namespace n_squared_divisible_by_144_l334_334253

theorem n_squared_divisible_by_144
  (n : ℕ)
  (h1 : 0 < n)
  (h2 : ∀ d : ℕ, d > 1 → d ∣ n → d ≤ 12) :
  144 ∣ n^2 :=
by
  sorry

end n_squared_divisible_by_144_l334_334253


namespace arithmetic_sequence_sum_l334_334835

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

theorem arithmetic_sequence_sum 
  {a1 d : ℕ} (h_pos_d : d > 0) 
  (h_sum : a1 + (a1 + d) + (a1 + 2 * d) = 15) 
  (h_prod : a1 * (a1 + d) * (a1 + 2 * d) = 80) 
  : a_n a1 d 11 + a_n a1 d 12 + a_n a1 d 13 = 105 :=
sorry

end arithmetic_sequence_sum_l334_334835


namespace fitted_bowling_ball_volume_correct_l334_334986

noncomputable def volume_of_fitted_bowling_ball : ℝ :=
  let ball_radius := 12
  let ball_volume := (4/3) * Real.pi * ball_radius^3
  let hole1_radius := 1
  let hole1_volume := Real.pi * hole1_radius^2 * 6
  let hole2_radius := 1.25
  let hole2_volume := Real.pi * hole2_radius^2 * 6
  let hole3_radius := 2
  let hole3_volume := Real.pi * hole3_radius^2 * 6
  ball_volume - (hole1_volume + hole2_volume + hole3_volume)

theorem fitted_bowling_ball_volume_correct :
  volume_of_fitted_bowling_ball = 2264.625 * Real.pi := by
  -- proof would go here
  sorry

end fitted_bowling_ball_volume_correct_l334_334986


namespace decreasing_function_positive_l334_334638

variable {f : ℝ → ℝ}

theorem decreasing_function_positive (h1 : ∀ x, f x > 0)
  (h2 : ∀ x, f'(x) < 0) 
  (h3 : ∀ x, f(x) / f''(x) < 1 - x) :
  ∀ x : ℝ, f x > 0 :=
sorry

end decreasing_function_positive_l334_334638


namespace equation_of_l1_equation_of_l2_l334_334205

structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := -1, y := 3}
def B : Point := {x := 5, y := -7}

def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 20 = 0

theorem equation_of_l1 : 
  ∃ (k : ℝ) (b : ℝ), 
    (k = -3/4) ∧ 
    (l1 : ℝ → ℝ → Prop) := 
  ∃ k b, k = -3/4 ∧ ∀ x y, l1 x y ↔ y = k * x + b ∧ l1 (-1) 3 := sorry

theorem equation_of_l2 : 
  ∃ (k : ℝ) (b : ℝ), 
    (midpoint := {x := (A.x + B.x) / 2, y := (A.y + B.y) / 2}) ∧ 
    (k = 4/3) ∧ 
    (l2 : ℝ → ℝ → Prop) := 
  ∃ k b, k = 4/3 ∧ 
  ∀ x y, l2 x y ↔ y = k * x + b ∧ l2 ((A.x + B.x) / 2) ((A.y + B.y) / 2) := sorry

end equation_of_l1_equation_of_l2_l334_334205


namespace M_intersect_N_equals_M_l334_334235

-- Define the sets M and N
def M := { x : ℝ | x^2 - 3 * x + 2 = 0 }
def N := { x : ℝ | x * (x - 1) * (x - 2) = 0 }

-- The theorem we want to prove
theorem M_intersect_N_equals_M : M ∩ N = M := 
by 
  sorry

end M_intersect_N_equals_M_l334_334235


namespace retail_women_in_LA_l334_334864

/-
Los Angeles has 6 million people living in it. If half the population is women 
and 1/3 of the women work in retail, how many women work in retail in Los Angeles?
-/

theorem retail_women_in_LA 
  (total_population : ℕ)
  (half_population_women : total_population / 2 = women_population)
  (third_women_retail : women_population / 3 = retail_women)
  : total_population = 6000000 → retail_women = 1000000 :=
by
  sorry

end retail_women_in_LA_l334_334864


namespace value_at_pi_six_max_and_min_on_interval_l334_334692

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos (π / 2 - x) * Real.cos x - Real.sqrt 3 * Real.cos (2 * x)

theorem value_at_pi_six : f (π / 6) = 0 :=
  sorry

theorem max_and_min_on_interval : 
  ∃ (max min : ℝ), (max = 2 ∧ min = -Real.sqrt 3) ∧
  ∀ x ∈ Set.Icc 0 (π / 2), f x ≤ max ∧ min ≤ f x :=
  sorry

end value_at_pi_six_max_and_min_on_interval_l334_334692


namespace like_terms_product_l334_334712

theorem like_terms_product :
  ∀ (m n : ℕ),
    (-x^3 * y^n) = (3 * x^m * y^2) → (m = 3 ∧ n = 2) → m * n = 6 :=
by
  intros m n h1 h2
  sorry

end like_terms_product_l334_334712


namespace min_value_inequality_correct_l334_334850

noncomputable def min_value_inequality (x : Fin 50 → ℝ) (k : ℝ) (h1 : 0 < k ∧ k > 1)
  (h2 : ∀ i, 0 < x i) (h3 : k * ∑ i, (x i)^2 = 1) : ℝ := 
  ∑ i, x i / (1 - (x i)^2)

theorem min_value_inequality_correct (x : Fin 50 → ℝ) (k : ℝ) 
  (h1 : 0 < k ∧ k > 1)
  (h2 : ∀ i, 0 < x i)
  (h3 : k * ∑ i, (x i)^2 = 1) :
  min_value_inequality x k h1 h2 h3 = (3 * Real.sqrt 3) / (2 * k) := 
sorry

end min_value_inequality_correct_l334_334850


namespace intersecting_segments_l334_334947

/-- There are 4n points on the circumference of a circle, alternatingly colored blue and red. 
Divide all blue points into pairs and connect each pair with a blue line segment, and do the same for the red points. 
Prove that there are at least n pairs of red and blue line segments that intersect each other. -/
theorem intersecting_segments (n : ℕ) : 
  ∃ (red_pairs blue_pairs : list (ℕ × ℕ)), 
    (∀ (pair : ℕ × ℕ), pair ∈ red_pairs → pair.1 % 2 = 1 ∧ pair.2 % 2 = 1) ∧  -- Red pairs have odd indices
    (∀ (pair : ℕ × ℕ), pair ∈ blue_pairs → pair.1 % 2 = 0 ∧ pair.2 % 2 = 0) ∧ -- Blue pairs have even indices
    (∀ (pair : ℕ × ℕ), pair ∈ red_pairs → pair.1 ≠ pair.2) ∧ -- Red pairs are distinct
    (∀ (pair : ℕ × ℕ), pair ∈ blue_pairs → pair.1 ≠ pair.2) ∧ -- Blue pairs are distinct
    (list.length red_pairs = n ∧ list.length blue_pairs = n) ∧ -- Number of pairs
    (∃ k, 1 ≤ k ∧ k ≤ n ∧ -- there are at least n pairs of intersecting red and blue line segments
        ∀ i, i < k → 
          (∃ red blue, red ∈ red_pairs ∧ blue ∈ blue_pairs ∧ 
             (segment_intersection red blue))) :=
sorry

/-- Helper function to determine if two pairs of points on a circle intersect -/
def segment_intersection (red blue : ℕ × ℕ) : Prop :=
  -- Placeholder for actual segment intersection logic
  sorry

end intersecting_segments_l334_334947


namespace least_five_digit_perfect_square_cube_l334_334514

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334514


namespace intersection_of_translated_polyhedra_l334_334974

-- Defining the Polyhedron type and the necessary conditions for the problem
structure Polyhedron :=
(vertices : Finset ℝ³)
(convex : ∀ (x y : ℝ³) (hx : x ∈ vertices) (hy : y ∈ vertices) (t : ℝ), 0 ≤ t → t ≤ 1 → t • x + (1 - t) • y ∈ vertices)
(card_vertices : vertices.card = 9)

-- Assuming a polyhedron P with one vertex A
variables (P : Polyhedron) (A : ℝ³) (hA : A ∈ P.vertices)

-- Defining translated versions of P by moving A to each of the remaining vertices
def translated_polyhedra (P : Polyhedron) (A : ℝ³) (hA : A ∈ P.vertices) : Finset (Set ℝ³) :=
P.vertices.filter (λ v, v ≠ A) |>.image (λ v, (λ x, x - A + v) '' P.vertices)

-- The Lean 4 statement to prove the main question
theorem intersection_of_translated_polyhedra :
  ∃ (P1 P2 : Set ℝ³), P1 ∈ translated_polyhedra P A hA ∧ P2 ∈ translated_polyhedra P A hA ∧ P1 ≠ P2 ∧
    ∃ (x : ℝ³), x ∈ P1 ∧ x ∈ P2 :=
sorry

end intersection_of_translated_polyhedra_l334_334974


namespace line_parabola_intersection_one_point_l334_334387

theorem line_parabola_intersection_one_point (k : ℝ) :
  (∃ y : ℝ, (-3 * y^2 - 4 * y + 7 = k) ∧ ∀ y1 y2 : ℝ, ( 3 * y1^2 + 4 * y1 + (k - 7) = 0 → 3 * y2^2 + 4 * y2 + (k - 7) = 0 → y1 = y2)) ↔ (k = 25 / 3) :=
by
  sorry

end line_parabola_intersection_one_point_l334_334387


namespace bowling_ball_weight_l334_334875

theorem bowling_ball_weight :
  ∀ (W K : ℝ), (9 * W = 4 * K) ∧ (K = 32) → (W = 128 / 9) :=
by 
  intros W K h
  cases h with h1 h2
  rw h2 at h1
  have h3 : 4 * 32 = 128, by norm_num
  rw h3 at h1
  linarith

end bowling_ball_weight_l334_334875


namespace problem_statement_l334_334655

noncomputable def func {X : Type} [OrderedCommGroupExtend X] : Type := X → X

variable {f : func (ℝ≥0)}

theorem problem_statement :
  (∀ (ω x y z : ℝ≥0), ω * x = y * z →
    (f(ω)^2 + f(x)^2) / (f(y^2) + f(z^2)) = (ω^2 + x^2) / (y^2 + z^2))
  →
  (∀ x : ℝ≥0, f(x) = x ∨ f(x) = 1 / x) :=
by
  sorry

end problem_statement_l334_334655


namespace least_five_digit_is_15625_l334_334437

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334437


namespace integer_solutions_count_l334_334754

theorem integer_solutions_count : 
  {x : ℤ | (x - 3) ^ (36 - x^2) = 1}.finite_card = 4 := 
by 
  sorry

end integer_solutions_count_l334_334754


namespace chess_tournament_points_distribution_l334_334182

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l334_334182


namespace part1_part2_l334_334888

noncomputable def f (m x : ℝ) : ℝ := m - |x - 1| - |x + 1|

theorem part1 (x : ℝ) : -3 / 2 < x ∧ x < 3 / 2 ↔ f 5 x > 2 := by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, ∃ y : ℝ, x^2 + 2 * x + 3 = f m y) ↔ 4 ≤ m := by
  sorry

end part1_part2_l334_334888


namespace jordan_time_for_7_miles_l334_334293

noncomputable def time_for_7_miles (jordan_miles : ℕ) (jordan_time : ℤ) : ℤ :=
  jordan_miles * jordan_time 

theorem jordan_time_for_7_miles :
  ∃ jordan_time : ℤ, (time_for_7_miles 7 (16 / 3)) = 112 / 3 :=
by
  sorry

end jordan_time_for_7_miles_l334_334293


namespace tangent_line_at_1_f_has_minimum_value_l334_334740

def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * Real.log x - a

theorem tangent_line_at_1 (a : ℝ) (h : a = Real.exp 1) :
    let f_x := f 1 a
    let f_prime_x := (fun x => Real.exp x - a / x) 1
    f_x = 0 ∧ f_prime_x = 0 → y = 0 :=
  by
    intro h_f_x h_f_prime_x
    exact sorry

theorem f_has_minimum_value (a : ℝ) (h : 0 < a ∧ a < Real.exp 1) :
    ∃ x₀ ∈ Ioo (a / Real.exp 1) 1, ∀ x ∈ Ioo (a / Real.exp 1) 1, f x₀ a < f x a ∧ f x₀ a > 0 :=
  by
    exact sorry

end tangent_line_at_1_f_has_minimum_value_l334_334740


namespace intersection_points_count_l334_334631

variables {a b c d : ℝ}

theorem intersection_points_count (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) :
  let f := λ x : ℝ, a * x ^ 2 + b * x + c,
      g := λ x : ℝ, a * x ^ 2 + (d - b) * x + c in
  ∃! p : ℝ × ℝ, p = (0, c) ∧ f p.1 = g p.1 :=
by
  sorry

end intersection_points_count_l334_334631


namespace single_straight_cut_l334_334605

theorem single_straight_cut (S : Type) [square S] (cut : S → S × S) :
  ∀ (shape : S), shape ≠ square → (∃ (A B : S), cut square = (A, B) ∧ (shape ≠ A ∧ shape ≠ B)) :=
by
  sorry

end single_straight_cut_l334_334605


namespace population_net_increase_l334_334784

-- Define the birth rate and death rate conditions
def birth_rate := 4 / 2 -- people per second
def death_rate := 2 / 2 -- people per second
def net_increase_per_sec := birth_rate - death_rate -- people per second

-- Define the duration of one day in seconds
def seconds_in_a_day := 24 * 3600 -- seconds

-- Define the problem to prove
theorem population_net_increase :
  net_increase_per_sec * seconds_in_a_day = 86400 :=
by
  sorry

end population_net_increase_l334_334784


namespace least_five_digit_perfect_square_and_cube_l334_334477

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334477


namespace find_k_intersects_parabola_at_one_point_l334_334372

theorem find_k_intersects_parabola_at_one_point :
  ∃ k : ℝ, (∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k ↔ y = (-4 / (2 * 3))) →
    k = 25 / 3 :=
by sorry

end find_k_intersects_parabola_at_one_point_l334_334372


namespace properties_of_f_in_interval_l334_334024

noncomputable def f : ℝ → ℝ := sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := sorry
lemma f_shift (x : ℝ) : f (x + 1) = f (-x) := sorry
lemma f_log (x : ℝ) (h : 0 < x ∧ x ≤ 1/2) : f x = log (x + 1) / log 2 := sorry

theorem properties_of_f_in_interval :
  ∀ x, (1 < x ∧ x < 3/2) → (f x < 0 ∧ ∃ δ > 0, 0 < δ ∧ ∀ y, (x - δ < y ∧ y < x + δ) → f y < f x) := 
by 
  -- Proof comes here
  sorry

end properties_of_f_in_interval_l334_334024


namespace reflection_square_identity_l334_334832

def vector := ℝ × ℝ

def reflection_matrix (v : vector) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := v in
  let norm_sq := x^2 + y^2 in
  ![
    [2 * x^2 / norm_sq - 1, 2 * x * y / norm_sq],
    [2 * x * y / norm_sq, 2 * y^2 / norm_sq - 1]
  ]

theorem reflection_square_identity (v : vector) (Q := reflection_matrix v) : 
  v = (4, -2) → Q * Q = 1 :=
by 
  sorry

end reflection_square_identity_l334_334832


namespace least_five_digit_perfect_square_and_cube_l334_334509

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334509


namespace bernoulli_inequality_l334_334560

theorem bernoulli_inequality (x : ℝ) (n : ℕ) (hx : x > -1) (hn : n > 0) : 
  (1 + x) ^ n ≥ 1 + n * x := 
sorry

end bernoulli_inequality_l334_334560


namespace sales_on_third_day_l334_334569

variable (a m : ℕ)

def first_day_sales : ℕ := a
def second_day_sales : ℕ := 3 * a - 3 * m
def third_day_sales : ℕ := (3 * a - 3 * m) + m

theorem sales_on_third_day 
  (a m : ℕ) : third_day_sales a m = 3 * a - 2 * m :=
by
  -- Assuming the conditions as our definitions:
  let fds := first_day_sales a
  let sds := second_day_sales a m
  let tds := third_day_sales a m

  -- Proof direction:
  show tds = 3 * a - 2 * m
  sorry

end sales_on_third_day_l334_334569


namespace correct_propositions_count_l334_334916

theorem correct_propositions_count :
  let propositions := [
    "Vertical angles are equal",
    "Two lines perpendicular to the same line are parallel",
    "Only one line passing through a point is parallel to a given line",
    "Only one line passing through a point is perpendicular to a given line",
    "The perpendicular line segment is the shortest"
  ] in
  let is_correct (p : String) : Prop :=
    (p = "Vertical angles are equal") ∨
    (p = "Only one line passing through a point is parallel to a given line") ∨
    (p = "Only one line passing through a point is perpendicular to a given line") ∨
    (p = "The perpendicular line segment is the shortest") in
  ∑ p in propositions | is_correct p, 1 = 4 :=
by
  sorry

end correct_propositions_count_l334_334916


namespace max_value_of_abc_expression_l334_334838

noncomputable def max_abc_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) : ℝ :=
  a^3 * b^2 * c^2

theorem max_value_of_abc_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  max_abc_expression a b c h1 h2 h3 h4 ≤ 432 / 7^7 :=
sorry

end max_value_of_abc_expression_l334_334838


namespace remaining_pie_portion_l334_334624

theorem remaining_pie_portion (Carlos_takes: ℝ) (fraction_Maria: ℝ) :
  Carlos_takes = 0.60 →
  fraction_Maria = 0.25 →
  (1 - Carlos_takes) * (1 - fraction_Maria) = 0.30 := by
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end remaining_pie_portion_l334_334624


namespace family_arrangement_l334_334763

-- Definitions for the conditions
def family_members : ℕ := 5
def parents : ℕ := 2
def ways_to_arrange_parents : ℕ := 2
noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)
def ways_to_arrange_others := factorial (family_members - parents) -- 3!

-- The lean statement for the theorem
theorem family_arrangement : ways_to_arrange_parents * ways_to_arrange_others = 12 :=
by
  have h1 : ways_to_arrange_others = 6 := by compute
  have h2 : ways_to_arrange_parents = 2 := by compute
  rw [h1, h2]
  norm_num
  intro
  sorry

end family_arrangement_l334_334763


namespace salt_solution_percentage_l334_334049

theorem salt_solution_percentage (salt water : ℕ) (pct : ℚ) :
  salt = 10 ∧ water = 100 ∧ pct = 9.1 → 
  (salt / (salt + water) * 100 = pct) :=
by {
  intros h,
  sorry
}

end salt_solution_percentage_l334_334049


namespace coloring_equilateral_triangle_l334_334931

theorem coloring_equilateral_triangle :
  ∀ (A B C : Type) (color : A → Type) (d : A → A → ℝ),
  (∀ x y, d x y = 1 → color x = color y) :=
by sorry

end coloring_equilateral_triangle_l334_334931


namespace chess_tournament_points_l334_334147

theorem chess_tournament_points
  (points : String → ℝ)
  (total_points : points "Andrey" + points "Dima" + points "Vanya" + points "Sasha" = 12)
  (distinct_points : 
    points "Andrey" ≠ points "Dima" ∧ 
    points "Andrey" ≠ points "Vanya" ∧ 
    points "Andrey" ≠ points "Sasha" ∧ 
    points "Dima" ≠ points "Vanya" ∧ 
    points "Dima" ≠ points "Sasha" ∧ 
    points "Vanya" ≠ points "Sasha")
  (order : 
    points "Andrey" > points "Dima" ∧ 
    points "Dima" > points "Vanya" ∧ 
    points "Vanya" > points "Sasha")
  (same_wins :
    let games_won (student : String) := (points student - 3) / 0.5 in
    games_won "Andrey" = games_won "Sasha") :
  points "Andrey" = 4 ∧ points "Dima" = 3.5 ∧ points "Vanya" = 2.5 ∧ points "Sasha" = 2 :=
by
  sorry

end chess_tournament_points_l334_334147


namespace second_number_is_four_l334_334412

noncomputable def findSecondNumber : ℕ :=
  let lcm_val := Nat.lcm 2 7
  let approx_count := Nat.ceil (600 / 11.9)
  let estimate_14X := 50.42
  let div_est := estimate_14X / lcm_val
  let X := Nat.round div_est
  X

theorem second_number_is_four (lcm_val : ℕ) (approx_count : ℕ) :
  lcm_val = Nat.lcm 2 7 ∧ approx_count = Nat.ceil (600 / 11.9) → findSecondNumber = 4 :=
by
  intros
  sorry

end second_number_is_four_l334_334412


namespace range_of_a_l334_334717

-- Define f(x) piecewise
def f (a x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else log a x

-- Define the required conditions
def is_increasing (a : ℝ) : Prop :=
  (1 < a ∧ a < 3) ∧ (∀ x y : ℝ, x < y → f a x ≤ f a y)

-- The theorem to prove
theorem range_of_a (a : ℝ) : is_increasing a → a ∈ set.Ico (3/2) 3 :=
  sorry

end range_of_a_l334_334717


namespace derivative_of_trig_function_l334_334361

-- Prove that the derivative of y = 2 * sin x * cos x is 2 * cos (2 * x)
theorem derivative_of_trig_function :
  ∀ (x : ℝ), 
    (deriv (λ x, 2 * (sin x) * (cos x))) x = 2 * cos (2 * x) :=
sorry

end derivative_of_trig_function_l334_334361


namespace lakeside_middle_school_field_trip_fraction_l334_334817

theorem lakeside_middle_school_field_trip_fraction (b : ℕ) (b_gt0 : b > 0) :
  (let girls_on_trip := b in 
     let boys_on_trip := (3 / 4 : ℚ) * b in 
     let total_on_trip := girls_on_trip + boys_on_trip in
     girls_on_trip / total_on_trip = (4 / 7 : ℚ)) :=
by
  sorry

end lakeside_middle_school_field_trip_fraction_l334_334817


namespace sequence_value_lemma_l334_334232

noncomputable def seq (n : ℕ) : ℕ → ℚ
| 0     := 3
| (n+1) := (5 * seq n - 13) / (3 * seq n - 7)

theorem sequence_value_lemma 
  (a : ℕ → ℚ) 
  (h0 : a 0 = 3)
  (h1 : ∀ n, a (n + 1) = (5 * a n - 13) / (3 * a n - 7)) :
  a 2016 = 2 :=
begin
  have h_per_0 : a 1 = 1, 
  { rw [h1 0, h0], simp }, -- calculate a_2 from a_1 = 3

  have h_per_1 : a 2 = 2, 
  { rw [h1 1, h_per_0], norm_num }, -- calculate a_3 from a_2 = 1
  
  have h_per_2 : a 3 = 3, 
  { rw [h1 2, h_per_1], norm_num }, -- calculate a_4 from a_3 = 2

  have h_per : ∀ n, a (n + 3) = a n,
  { intro n, induction n with n ih,
    { simp [h1, h0] },
    { simp [h1, ih] } }, -- use induction to show periodicity

  specialize h_per 2013, -- 2016 mod 3 = 0, use periodicity
  calc 
  a 2016 = a (2013 + 3) : by simp
  ... = a 3 : by rw h_per
  ... = 2   : by { rw [h1 2, h_per_2], norm_num }
end

end sequence_value_lemma_l334_334232


namespace lever_equilibrium_min_force_l334_334025

noncomputable def lever_minimum_force (F L : ℝ) : Prop :=
  (F * L = 49 + 2 * (L^2))

theorem lever_equilibrium_min_force : ∃ F : ℝ, ∃ L : ℝ, L = 7 → lever_minimum_force F L :=
by
  sorry

end lever_equilibrium_min_force_l334_334025


namespace sum_of_integer_solutions_l334_334535

theorem sum_of_integer_solutions :
  (∑ x in { x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 }, x) = 18 :=
by
  sorry

end sum_of_integer_solutions_l334_334535


namespace mint_issue_coins_l334_334368

theorem mint_issue_coins :
  ∃ (d : Fin 12 → ℕ), 
    ∀ x : ℕ, 1 ≤ x ∧ x ≤ 6543 → ∃ (c : Fin 8 → ℕ), ∃ (count : ℕ), count ≤ 8 ∧ ∑ i in Finset.range count, c i = x ∧ ∀ i : Fin 8, c i ∈ Finset.image d Finset.univ :=
sorry

end mint_issue_coins_l334_334368


namespace quadratic_unique_real_root_l334_334257

theorem quadratic_unique_real_root (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ∃! r : ℝ, x = r) → m = 2/9 :=
by
  sorry

end quadratic_unique_real_root_l334_334257


namespace age_of_B_l334_334776

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 9) : B = 39 := by
  sorry

end age_of_B_l334_334776


namespace boat_speed_still_water_l334_334013

def effective_upstream_speed (b c : ℝ) : ℝ := b - c
def effective_downstream_speed (b c : ℝ) : ℝ := b + c

theorem boat_speed_still_water :
  ∃ b c : ℝ, effective_upstream_speed b c = 9 ∧ effective_downstream_speed b c = 15 ∧ b = 12 :=
by {
  sorry
}

end boat_speed_still_water_l334_334013


namespace least_five_digit_perfect_square_and_cube_l334_334508

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334508


namespace lucy_popsicles_l334_334318

theorem lucy_popsicles :
  let lucy_cents := 2540 in
  let popsicle_cost := 175 in
  (lucy_cents / popsicle_cost).floor = 14 :=
sorry

end lucy_popsicles_l334_334318


namespace probability_of_black_ball_l334_334565

theorem probability_of_black_ball (P_red P_white : ℝ) (h_red : P_red = 0.43) (h_white : P_white = 0.27) : 
  (1 - P_red - P_white) = 0.3 := 
by
  sorry

end probability_of_black_ball_l334_334565


namespace max_cards_terminates_l334_334408

def game_terminates (n : ℕ) (cards : Fin n → ℕ) : Prop :=
  ∀ (p : Fin n), ∃ k : ℕ, k < n ∧
  (∀ i : ℕ, i < k → cards ((p + ⟨i, sorry⟩) % n) > 0 ∧ cards ((p + ⟨i + 1, sorry⟩) % n) > 0)

theorem max_cards_terminates (n : ℕ) (hpos : 0 < n) : ∀ (cards : Fin n → ℕ), 
  (game_terminates n cards ↔ ∑ k in Finset.univ, cards k ≤ n - 1) :=
sorry

end max_cards_terminates_l334_334408


namespace intersecting_line_at_one_point_l334_334382

theorem intersecting_line_at_one_point (k : ℝ) :
  (∃ y : ℝ, k = -3 * y^2 - 4 * y + 7 ∧ 
           ∀ z : ℝ, k = -3 * z^2 - 4 * z + 7 → y = z) ↔ 
  k = 25 / 3 :=
by
  sorry

end intersecting_line_at_one_point_l334_334382


namespace exists_quadratic_polynomial_with_property_l334_334093

theorem exists_quadratic_polynomial_with_property :
  ∃ (f : ℝ → ℝ), (∃ (a b c : ℤ), ∀ x, f x = (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)) ∧ f (f (real.sqrt 3)) = 0 :=
by
  sorry

end exists_quadratic_polynomial_with_property_l334_334093


namespace smallest_period_g_shift_l334_334348

noncomputable def f : ℝ → ℝ := sorry
def g (x : ℝ) : ℝ := f (x / 4)

lemma periodic_f {x : ℝ} : f x = f (x - 24) := sorry

theorem smallest_period_g_shift :
  (∃ b : ℝ, b > 0 ∧ ∀ x, g (x - b) = g x) ∧
  (∀ b, b > 0 ∧ (∀ x, g (x - b) = g x) → b ≥ 96) :=
sorry

end smallest_period_g_shift_l334_334348


namespace solve_the_problem_l334_334801

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D A1 B1 C1 D1 M : V)
variables (x y z : ℝ)

-- Given conditions
def condition1 : Prop := A1 -ᵥ M = 2 • (C -ᵥ M)
def condition2 : Prop := A -ᵥ M = x • (B -ᵥ A) + y • (D -ᵥ A) + z • (A1 -ᵥ A)

-- The goal
def goal : Prop := x = 2 / 3 ∧ y = 2 / 3 ∧ z = 1 / 3

theorem solve_the_problem (h1 : condition1) (h2 : condition2) : goal :=
sorry

end solve_the_problem_l334_334801


namespace profit_cents_l334_334567

-- Define the conditions
def buys_price : ℕ := 14 -- 14 cents
def buys_oranges : ℕ := 4 -- 4 oranges

def sells_price : ℕ := 24 -- 24 cents
def sells_oranges : ℕ := 6 -- 6 oranges

-- Define the proof goal
theorem profit_cents (desired_profit : ℕ) : 
  (let cost_per_orange := buys_price / buys_oranges in
   let sell_per_orange := sells_price / sells_oranges in
   let profit_per_orange := sell_per_orange - cost_per_orange in
   desired_profit / profit_per_orange) = 240 :=
by
  -- substitute the value of desired profit in cents 
  have h : profit_per_orange = 0.5 := sorry
  exact sorry

end profit_cents_l334_334567


namespace rocket_coaster_total_capacity_l334_334904

theorem rocket_coaster_total_capacity :
  ∃ (n_fours n_sixes : ℕ),
  n_fours = 9 ∧ n_sixes = 15 - n_fours ∧
  (9 * 4 + n_sixes * 6 = 72) :=
by
  use [9, 15 - 9]
  split
  · refl
  split
  · refl
  · norm_num

end rocket_coaster_total_capacity_l334_334904


namespace least_five_digit_perfect_square_and_cube_l334_334459

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334459


namespace quasi_geometric_progression_probability_l334_334261

theorem quasi_geometric_progression_probability :
  let numbers := {2, 4, 8, 16, 32, 64, 128}
  let combinations := {s | ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
                               (a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ d ∈ numbers) ∧
                               (a * b = c * d)}
  let total_combinations := (fintype.card {s // ∃ a b c d, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
                                                (a ∈ numbers) ∧ (b ∈ numbers) ∧ (c ∈ numbers) ∧ (d ∈ numbers)})
  ∃ (favs : set {a // a ∈ combinations}) (total : ℕ), 
  favs.card = 13 ∧ total_combinations = 35 ∧ 
  (favs.card / total_combinations : ℚ) = 13 / 35 := by { sorry }

end quasi_geometric_progression_probability_l334_334261


namespace sum_of_determinants_l334_334214

def determinant_2x2 (a b c d : ℤ) : ℤ :=
  a * d - b * c

def a_n (n : ℕ) : ℤ :=
  determinant_2x2 (8 * n - 4) (8 * n - 2) (8 * n) (8 * n + 2)

theorem sum_of_determinants : 
  (∑ n in Finset.range 251, a_n (n + 1)) = -2008 :=
by
  sorry

end sum_of_determinants_l334_334214


namespace find_OB_l334_334059

noncomputable def square_side_length : ℝ := 6
noncomputable def AE_length : ℝ := 8
noncomputable def diagonal_length (s : ℝ) : ℝ := s * Real.sqrt 2

theorem find_OB : 
  let s := square_side_length in
  let AE := AE_length in
  let d := diagonal_length s in
  d / 2 = 4.5 :=
by
  sorry

end find_OB_l334_334059


namespace ratio_tangency_l334_334627

theorem ratio_tangency 
  (O : Type) [metric_space O] [smooth_inv_fun O] {A B T X : O}
  (rO : ℝ) (r_small : ℝ)
  (hT : T ∈ sphere O rO)
  (hX : X ∈ line_segment A B)
  (hT_tangent : ∀ (Y : O), dist T Y = rO ↔ dist X Y = 2 * r_small)
  (h_ratio : dist A X = 2 * dist X B)
  : dist A T / dist B T = 2 :=
by sorry

end ratio_tangency_l334_334627


namespace best_fitting_model_l334_334955

def model_1 : ℝ := 0.86
def model_2 : ℝ := 0.96
def model_3 : ℝ := 0.73
def model_4 : ℝ := 0.66

theorem best_fitting_model : 
  model_2 = 0.96 →
  model_1 = 0.86 →
  model_3 = 0.73 →
  model_4 = 0.66 →
  ∀ (m : ℝ), m ∈ {model_1, model_2, model_3, model_4} →
  (|model_2 - 1| ≤ |m - 1|) :=
by 
  intros h2 h1 h3 h4 m hm
  simp [h1, h2, h3, h4, hm]
  sorry

end best_fitting_model_l334_334955


namespace least_five_digit_perfect_square_and_cube_l334_334425

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334425


namespace sum_of_possible_values_l334_334846

theorem sum_of_possible_values (x y : ℝ) (h : x * y - (2 * x) / (y ^ 3) - (2 * y) / (x ^ 3) = 6) :
  ∃ (a b : ℝ), (a - 2) * (b - 2) = 4 ∧ (a - 2) * (b - 2) = 9 ∧ 4 + 9 = 13 :=
sorry

end sum_of_possible_values_l334_334846


namespace cone_lateral_area_l334_334216

/--
Given that the radius of the base of a cone is 3 cm and the slant height is 6 cm,
prove that the lateral area of this cone is 18π cm².
-/
theorem cone_lateral_area {r l : ℝ} (h_radius : r = 3) (h_slant_height : l = 6) :
  (π * r * l) = 18 * π :=
by
  have h1 : r = 3 := h_radius
  have h2 : l = 6 := h_slant_height
  rw [h1, h2]
  norm_num
  sorry

end cone_lateral_area_l334_334216


namespace geo_prog_condition_l334_334080

/-- Define the sequence {a_n} using the recurrence relation -/
noncomputable def seq (a c : ℕ) : ℕ → ℕ
| 0       => a
| 1       => a
| (n + 2) => seq n * seq (n + 1) - c

/-- Prove that the sequence is a geometric progression if and only if a_1 = a_2 and c = 0 -/
theorem geo_prog_condition (a1 a2 c : ℕ) : 
  (∀ n, seq a1 c (n + 2) = seq a1 c n * seq a1 c (n + 1) - c) ↔ 
  (a1 = a2 ∧ c = 0) :=
sorry

end geo_prog_condition_l334_334080


namespace split_coins_l334_334285

theorem split_coins (p n d q : ℕ) (hp : p % 5 = 0) 
  (h_total : p + 5 * n + 10 * d + 25 * q = 10000) :
  ∃ (p1 n1 d1 q1 p2 n2 d2 q2 : ℕ),
    (p1 + 5 * n1 + 10 * d1 + 25 * q1 = 5000) ∧
    (p2 + 5 * n2 + 10 * d2 + 25 * q2 = 5000) ∧
    (p = p1 + p2) ∧ (n = n1 + n2) ∧ (d = d1 + d2) ∧ (q = q1 + q2) :=
sorry

end split_coins_l334_334285


namespace compute_x_plus_y_l334_334206

theorem compute_x_plus_y :
    ∃ (x y : ℕ), 4 * y = 7 * 84 ∧ 4 * 63 = 7 * x ∧ x + y = 183 :=
by
  sorry

end compute_x_plus_y_l334_334206


namespace chess_tournament_scores_l334_334131

theorem chess_tournament_scores :
    ∃ (A D V S : ℝ),
    A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S ∧
    A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 ∧
    A > D ∧ D > V ∧ V > S ∧
    (∃ (wins_A wins_S : ℕ), wins_A = wins_S) :=
begin
    sorry
end

end chess_tournament_scores_l334_334131


namespace polygon_diagonals_div_by_3_l334_334022

theorem polygon_diagonals_div_by_3 :
  let n := 30 in
  let total_diagonals := n * (n - 3) / 2 in
  let divisors_of_3 := [3, 6, 9, 12, 15, 18, 21, 24, 27] in
  total_diagonals = 405 ∧ list.length divisors_of_3 * 2 - 1 = 17 :=
by
  sorry

end polygon_diagonals_div_by_3_l334_334022


namespace min_value_fraction_geq_3_div_2_l334_334782

theorem min_value_fraction_geq_3_div_2 (a : ℕ → ℝ) (m n : ℕ) (q : ℝ) (h1 : q > 0) 
  (h2 : ∀ k, a (k + 2) = q * a (k + 1)) (h3 : a 2016 = a 2015 + 2 * a 2014) 
  (h4 : a m * a n = 16 * (a 1) ^ 2) :
  (∃ q, q = 2 ∧ m + n = 6) → 4 / m + 1 / n ≥ 3 / 2 :=
by sorry

end min_value_fraction_geq_3_div_2_l334_334782


namespace red_socks_l334_334811

variable {R : ℕ}

theorem red_socks (h1 : 2 * R + R + 6 * R = 90) : R = 10 := 
by
  sorry

end red_socks_l334_334811


namespace least_five_digit_perfect_square_and_cube_l334_334452

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334452


namespace part_a_part_b_part_c_l334_334621

-- Part (a)
theorem part_a (x : ℝ) (h : 0 ≤ x ∧ x ≤ 10) : 
  let side1 := x / 4
  let side2 := (10 - x) / 4
  let area1 := (x^2) / 16
  let area2 := ((10 - x)^2) / 16
  side1 = x / 4 ∧ side2 = (10 - x) / 4 ∧ area1 = (x^2) / 16 ∧ area2 = ((10 - x)^2) / 16 :=
by
  intros
  simp only []
  exact ⟨rfl, rfl, rfl, rfl⟩

-- Part (b)
theorem part_b (x : ℝ) (h : 0 ≤ x ∧ x ≤ 10) :
  let S := (x^2) / 16 + ((10 - x)^2) / 16
  ∃ minimum_area : ℝ, minimum_area = 5 :=
by
  intros
  use 5
  have h_min : ∀ x : ℝ, S = (x^2) / 16 + ((10 - x)^2) / 16 :=
    by intros; simp only [S]; rw [← add_assoc]; exact rfl
  suffices : x = 5, by assumption
  sorry

-- Part (c)
theorem part_c :
  let piece_length := 10 / 10
  piece_length = 1 :=
by
  intros piece_length
  simp only []
  exact rfl

end part_a_part_b_part_c_l334_334621


namespace rectangle_inscribed_circle_distances_ratio_l334_334399

-- Defining the conditions of the problem
def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  B.1 - A.1 = 12 ∧ D.2 - A.2 = 5 ∧ ((A.1, D.2) = D ∧ (B.1, D.2) = C)

def diagonals_intersect_at (A B C D E : ℝ × ℝ) : Prop :=
  let M := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) in E = M

def inscribed_circle_center_distance_ratio (A B C D E : ℝ × ℝ) (r : ℝ) : Prop :=
  r = 10 / 3

-- Main theorem to be proved
theorem rectangle_inscribed_circle_distances_ratio
  {A B C D E : ℝ × ℝ}
  (h_rect : is_rectangle A B C D)
  (h_diag : diagonals_intersect_at A B C D E) :
  ∃ r, inscribed_circle_center_distance_ratio A B C D E r :=
begin
  use (10 / 3),
  unfold inscribed_circle_center_distance_ratio,
  sorry
end

end rectangle_inscribed_circle_distances_ratio_l334_334399


namespace solution_N_exists_l334_334117

noncomputable def find_integer_N : Prop :=
  ∃ (N : ℕ) (α β γ θ : ℝ), 
  (0.1 = Real.sin γ * Real.cos θ * Real.sin α) ∧
  (0.2 = Real.sin γ * Real.sin θ * Real.cos α) ∧
  (0.3 = Real.cos γ * Real.cos θ * Real.sin β) ∧
  (0.4 = Real.cos γ * Real.sin θ * Real.cos β) ∧
  (0.5 ≥ Abs.abs (N - 100 * Real.cos (2 * θ))) ∧
  (N = 79)

theorem solution_N_exists : find_integer_N :=
  sorry

end solution_N_exists_l334_334117


namespace find_m_l334_334724

theorem find_m (m : ℝ) : 
  (∀ (x y : ℝ), (y = x + m ∧ x = 0) → y = m) ∧
  (∀ (x y : ℝ), (y = 2 * x - 2 ∧ x = 0) → y = -2) ∧
  (∀ (x : ℝ), (∃ y : ℝ, (y = x + m ∧ x = 0) ∧ (y = 2 * x - 2 ∧ x = 0))) → 
  m = -2 :=
by 
  sorry

end find_m_l334_334724


namespace unanswered_questions_count_l334_334874

-- Define the variables: c (correct), w (wrong), u (unanswered)
variables (c w u : ℕ)

-- Define the conditions based on the problem statement.
def total_questions (c w u : ℕ) : Prop := c + w + u = 35
def new_system_score (c u : ℕ) : Prop := 6 * c + 3 * u = 120
def old_system_score (c w : ℕ) : Prop := 5 * c - 2 * w = 55

-- Prove that the number of unanswered questions, u, equals 10
theorem unanswered_questions_count (c w u : ℕ) 
    (h1 : total_questions c w u)
    (h2 : new_system_score c u)
    (h3 : old_system_score c w) : u = 10 :=
by
  sorry

end unanswered_questions_count_l334_334874


namespace maximum_coins_each_thief_can_secure_l334_334972

theorem maximum_coins_each_thief_can_secure (total_coins : ℕ) (total_divisions : ℕ) : 
  total_coins = 1987 → total_divisions = 40 → 
  ∀ (n : ℕ), (n < total_divisions) → 
  let parts := total_divisions + 1 in
  let max_coins := Nat.ceil (total_coins / parts : ℚ) in
  max_coins = 49 := 
by
  intros h_coins h_divisions n h_n_lt_divisions parts max_coins
  have parts_eq : parts = 41 := by rw [Nat.add_one, h_divisions]
  have max_coins_eq : max_coins = 49
    sorry 
  rw [←max_coins_eq]
  exact max_coins_eq

end maximum_coins_each_thief_can_secure_l334_334972


namespace smallest_m_for_probability_l334_334882

theorem smallest_m_for_probability :
  ∃ (m : ℕ), (∀ (x y z : ℝ), 0 ≤ x ∧ x ≤ ↑m ∧ 0 ≤ y ∧ y ≤ ↑m ∧ 0 ≤ z ∧ z ≤ ↑m ∧ 
               (|x - y| ≥ 2) ∧ (|y - z| ≥ 2) ∧ (|z - x| ≥ 2)) → 
               ((m - 4)^3 : ℝ) / (m^3 : ℝ) > 1/2 ∧ m = 16 :=
begin
  sorry
end

end smallest_m_for_probability_l334_334882


namespace tournament_teams_matches_l334_334903

theorem tournament_teams_matches (teams : Fin 10 → ℕ) 
  (h : ∀ i, teams i ≤ 9) : 
  ∃ i j : Fin 10, i ≠ j ∧ teams i = teams j := 
by 
  sorry

end tournament_teams_matches_l334_334903


namespace find_f_neg3_l334_334900

theorem find_f_neg3 : 
  (∃ f : ℝ → ℝ, (∀ x : ℝ, x ≠ 0 → 5 * f (1 / x) + 3 * f x / x = 2 * x^2) ∧ f (-3) = 14029 / 72) :=
sorry

end find_f_neg3_l334_334900


namespace missy_yells_at_obedient_dog_12_times_l334_334320

theorem missy_yells_at_obedient_dog_12_times (x : ℕ) (h : x + 4 * x = 60) : x = 12 :=
by
  -- Proof steps can be filled in here
  sorry

end missy_yells_at_obedient_dog_12_times_l334_334320


namespace average_white_paper_per_ton_trees_saved_per_ton_l334_334590

-- Define the given conditions
def waste_paper_tons : ℕ := 5
def produced_white_paper_tons : ℕ := 4
def saved_trees : ℕ := 40

-- State the theorems that need to be proved
theorem average_white_paper_per_ton :
  (produced_white_paper_tons : ℚ) / waste_paper_tons = 0.8 := 
sorry

theorem trees_saved_per_ton :
  (saved_trees : ℚ) / waste_paper_tons = 8 := 
sorry

end average_white_paper_per_ton_trees_saved_per_ton_l334_334590


namespace max_red_points_no_right_angle_triangle_l334_334719

variable (m n : ℕ)
variable (h_m : m > 7)
variable (h_n : n > 7)
variable (k : ℕ)

theorem max_red_points_no_right_angle_triangle (h_k : ∀ (i j p q r s : ℕ), 
  (i < m ∧ j < n ∧ p < m ∧ q < n ∧ r < m ∧ s < n) ∧ 
  (i = p ∨ j = q) ∧ (i = r ∨ j = s) ∧ (p = r ∨ q = s) → 
  ¬ (red_point i j ∧ red_point p q ∧ red_point r s)) : k ≤ m + n - 2 := 
sorry

end max_red_points_no_right_angle_triangle_l334_334719


namespace least_five_digit_perfect_square_and_cube_l334_334504

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334504


namespace three_numbers_difference_of_two_primes_l334_334245

def is_prime (n : ℕ) : Prop := Nat.Prime n
def difference_of_two_primes (a : ℕ) : Prop := ∃ p1 p2 : ℕ, is_prime p1 ∧ is_prime p2 ∧ a = p2 - p1
def sequence (n : ℕ) : ℕ := 10 * n + 1

theorem three_numbers_difference_of_two_primes :
  ∃ n1 n2 n3 : ℕ, difference_of_two_primes (sequence n1) ∧ difference_of_two_primes (sequence n2) ∧ difference_of_two_primes (sequence n3) ∧ n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 :=
  sorry

end three_numbers_difference_of_two_primes_l334_334245


namespace height_difference_is_9_l334_334884

-- Definitions of the height of Petronas Towers and Empire State Building.
def height_Petronas : ℕ := 452
def height_EmpireState : ℕ := 443

-- Definition stating the height difference.
def height_difference := height_Petronas - height_EmpireState

-- Proving the height difference is 9 meters.
theorem height_difference_is_9 : height_difference = 9 :=
by
  -- the proof goes here
  sorry

end height_difference_is_9_l334_334884


namespace max_value_l334_334847

theorem max_value (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) : 
  8 * x + 3 * y + 15 * z ≤ Real.sqrt 298 :=
sorry

end max_value_l334_334847


namespace probability_midpoint_in_T_l334_334825

open Nat

def num_points_in_T : Nat := 4 * 5 * 6
def valid_pairs_x : Nat := 8
def valid_pairs_y : Nat := 11
def valid_pairs_z : Nat := 18
def valid_midpoint_pairs : Nat := 8 * 11 * 18 - num_points_in_T
def total_combinations : Nat := binom 120 2
def gcd_4_183 : Nat := 4.gcd 183 -- Verify GCD
def probability_fraction : Rational := ⟨4, 183⟩ -- Fraction in simplest form
def p_and_q_sum : Nat := 4 + 183

theorem probability_midpoint_in_T :
    8 * 11 * 18 - num_points_in_T = 1560 ∧
    total_combinations = 7140 ∧
    gcd_4_183 = 1 ∧ -- Ensure the fraction is simplified
    probability_fraction.num + probability_fraction.denom = 187 :=
by {
  sorry
}

end probability_midpoint_in_T_l334_334825


namespace parallel_vectors_x_eq_neg6_l334_334863

variables (x : ℝ)
def a : ℝ × ℝ := (-1, 3)
def b : ℝ × ℝ := (2, x)

theorem parallel_vectors_x_eq_neg6 (h : ∃ k : ℝ, a = (k * fst b, k * snd b)) : x = -6 :=
by sorry

end parallel_vectors_x_eq_neg6_l334_334863


namespace least_five_digit_is_15625_l334_334439

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334439


namespace value_of_f5_and_f_neg5_l334_334691

noncomputable def f (a b c x : ℝ) : ℝ := a * x^7 - b * x^5 + c * x^3 + 2

theorem value_of_f5_and_f_neg5 (a b c : ℝ) (m : ℝ) (h : f a b c (-5) = m) :
  f a b c 5 + f a b c (-5) = 4 :=
sorry

end value_of_f5_and_f_neg5_l334_334691


namespace least_five_digit_perfect_square_and_cube_l334_334476

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334476


namespace find_a_l334_334314

theorem find_a (a : ℝ) (h : (1 + 2*Complex.I) * (a + Complex.I) = (a - 2) + (2*a + 1) * Complex.I) : a = -3 := sorry

end find_a_l334_334314


namespace interest_rate_A_l334_334998

-- Given conditions
variables (Principal : ℝ := 4000)
variables (interestRate_C : ℝ := 11.5 / 100)
variables (gain_B : ℝ := 180)
variables (time : ℝ := 3)
variables (interest_from_C : ℝ := Principal * interestRate_C * time)
variables (interest_to_A : ℝ := interest_from_C - gain_B)

-- The proof goal
theorem interest_rate_A (R : ℝ) : 
  1200 = Principal * (R / 100) * time → 
  R = 10 :=
by
  sorry

end interest_rate_A_l334_334998


namespace f_f_neg2_eq_l334_334223

def f (x : ℝ) : ℝ :=
  if x ≤ -2 then x + 2
  else if -2 < x ∧ x < 3 then 2^x
  else log x

theorem f_f_neg2_eq : f(f(-2)) = 1 := by
  sorry

end f_f_neg2_eq_l334_334223


namespace product_of_real_roots_l334_334935

theorem product_of_real_roots : 
  (∃ x y : ℝ, (x ^ Real.log x = Real.exp 1) ∧ (y ^ Real.log y = Real.exp 1) ∧ x ≠ y ∧ x * y = 1) :=
by
  sorry

end product_of_real_roots_l334_334935


namespace det_dilation_matrix_l334_334829

def dilationMatrix (n : ℕ) (s : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal fun _ => s

def E : Matrix (Fin 3) (Fin 3) ℝ := dilationMatrix 3 4

theorem det_dilation_matrix : det E = 64 := by
  -- Proof goes here
  sorry

end det_dilation_matrix_l334_334829


namespace white_figures_count_l334_334820

theorem white_figures_count (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) : 
  (Finset.card (Finset.choose n k) * Finset.card (Finset.choose n k)) = (Nat.choose n k) ^ 2 := 
by 
  sorry

end white_figures_count_l334_334820


namespace log_simplification_l334_334010

theorem log_simplification :
  (1:ℝ) * (Real.log 3 / Real.log 5 + Real.log 9 / (1/2 * Real.log 5)) * (Real.log 5 / Real.log 3 - Real.log 5 / (2 * Real.log 3)) +
  (1 - Real.sqrt 2) ^ 6 ^ (1/6 : ℝ) * Real.sqrt 0.25 = 2 + Real.sqrt 2 / 2 :=
by sorry

end log_simplification_l334_334010


namespace all_valid_triples_listed_l334_334316

constant ε : ℝ := 22.5

def angle_triple_valid (α β γ : ℝ) : Prop :=
  ∃ k m n : ℕ, α = k * ε ∧ β = m * ε ∧ γ = n * ε ∧ (k + m + n = 8) ∧ α + β + γ = 180

noncomputable def valid_angle_triples : set (ℝ × ℝ × ℝ) :=
  { (22.5, 22.5, 135),
    (22.5, 45, 112.5),
    (22.5, 67.5, 90),
    (45, 45, 90),
    (45, 67.5, 67.5) }

theorem all_valid_triples_listed :
  ∀ α β γ : ℝ, angle_triple_valid α β γ → (α, β, γ) ∈ valid_angle_triples :=
by
  sorry

end all_valid_triples_listed_l334_334316


namespace least_five_digit_perfect_square_and_cube_l334_334512

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334512


namespace complex_quadrant_third_l334_334220

def i := Complex.i
def z := (2 - 3 * i) / (1 + i)

theorem complex_quadrant_third : 
  let z_coord := Complex.re z, Complex.im z in
  z_coord = (-1 / 2, -5 / 2) →
  (Complex.re z < 0 ∧ Complex.im z < 0) := by
  intros z_coord hz
  have hz_re : z_coord.fst = -1 / 2 := by sorry
  have hz_im : z_coord.snd = -5 / 2 := by sorry
  split
  { rw [hz_re], linarith }
  { rw [hz_im], linarith }

end complex_quadrant_third_l334_334220


namespace exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero_l334_334089

theorem exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero :
  ∃ f : ℤ[X], f.degree = 2 ∧ ∀ x : ℝ, f.eval (f.eval x) = 0 → x = (Real.sqrt 3) := sorry

end exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero_l334_334089


namespace average_water_drunk_l334_334953

theorem average_water_drunk (d1 d2 d3 : ℕ) (h1 : d1 = 215) (h2 : d2 = d1 + 76) (h3 : d3 = d2 - 53) :
  (d1 + d2 + d3) / 3 = 248 :=
by
  -- placeholder for actual proof
  sorry

end average_water_drunk_l334_334953


namespace terrell_total_distance_l334_334353

theorem terrell_total_distance (saturday_distance sunday_distance : ℝ) (h_saturday : saturday_distance = 8.2) (h_sunday : sunday_distance = 1.6) :
  saturday_distance + sunday_distance = 9.8 :=
by
  rw [h_saturday, h_sunday]
  -- sorry
  norm_num

end terrell_total_distance_l334_334353


namespace sum_of_N_values_eq_neg_one_l334_334252

theorem sum_of_N_values_eq_neg_one (R : ℝ) :
  ∀ (N : ℝ), N ≠ 0 ∧ (N + N^2 - 5 / N = R) →
  (∃ N₁ N₂ N₃ : ℝ, N₁ + N₂ + N₃ = -1 ∧ N₁ ≠ 0 ∧ N₂ ≠ 0 ∧ N₃ ≠ 0) :=
by
  sorry

end sum_of_N_values_eq_neg_one_l334_334252


namespace odd_binomial_coeffs_power_of_2_l334_334333

theorem odd_binomial_coeffs_power_of_2 (n : ℕ) (h_pos : n > 0) : 
  ∃ k : ℕ, k = n.binary_repr.count '1' ∧ 
           (∃ N : ℕ, (N = fib k) ∧ (∑ i in finset.range (n+1), (choose n i) % 2) = N) :=
by sorry

end odd_binomial_coeffs_power_of_2_l334_334333


namespace new_rectangle_area_l334_334920

theorem new_rectangle_area (L W : ℝ) (h : L * W = 300) :
  let L_new := 2 * L
  let W_new := 3 * W
  L_new * W_new = 1800 :=
by
  let L_new := 2 * L
  let W_new := 3 * W
  sorry

end new_rectangle_area_l334_334920


namespace compute_floor_expr_l334_334629

theorem compute_floor_expr : 
  (Int.floor ((3005^3 : ℝ) / (3003 * 3004) - (3003^3) / (3004 * 3005)) = 8) :=
by
  have h : n = 3004 := rfl
  sorry

end compute_floor_expr_l334_334629


namespace total_number_of_digits_l334_334003

theorem total_number_of_digits (n : ℕ) (h : n = 356) :
  let digits_1_to_9 := 9,
      digits_10_to_99 := 90 * 2,
      digits_100_to_356 := (356 - 100 + 1) * 3
  in digits_1_to_9 + digits_10_to_99 + digits_100_to_356 = 960 := 
by
  sorry

end total_number_of_digits_l334_334003


namespace chess_tournament_distribution_l334_334173

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l334_334173


namespace original_perimeter_not_necessarily_integer_l334_334037

/--
Given a rectangle that is divided into several smaller rectangles,
where each of the smaller rectangles has a perimeter that is an integer,
prove that the perimeter of the original rectangle is not necessarily an integer.
-/
theorem original_perimeter_not_necessarily_integer
  (n : ℕ) (subrects : fin n → ℝ × ℝ)
  (h : ∀ i, let (l, w) := subrects i in 2 * (l + w) ∈ ℤ) :
  ∃ (L W : ℝ), (2 * (L + W)) ∉ ℤ := sorry

end original_perimeter_not_necessarily_integer_l334_334037


namespace download_time_correct_l334_334422

-- Define Vicky's internet speed.
def internet_speed : ℕ := 30 -- in MB/second

-- Define the program size.
def program_size_gb : ℕ := 720 -- in GB

-- Define the conversion factor from GB to MB.
def gb_to_mb : ℕ := 1000 -- 1 GB = 1000 MB

-- Define the conversion factor from seconds to hours.
def seconds_to_hours : ℕ := 3600 -- 1 hour = 3600 seconds

-- Define the program size in MB.
def program_size_mb : ℕ := program_size_gb * gb_to_mb

-- Define the download time in seconds.
def download_time_seconds : ℕ := program_size_mb / internet_speed

-- Define the download time in hours.
def download_time_hours : ℕ := download_time_seconds / seconds_to_hours

-- The final theorem which states the download time in hours.
theorem download_time_correct : download_time_hours = 720 * 1000 / (30 * 3600) :=
by
  have h1 : program_size_mb = 720 * 1000 := rfl
  have h2 : download_time_seconds = program_size_mb / 30 := rfl
  have h3 : download_time_hours = download_time_seconds / 3600 := rfl
  rw [h1, h2, h3]
  norm_num
  refl

end download_time_correct_l334_334422


namespace find_parallel_lines_l334_334419

noncomputable def Point := (ℝ × ℝ)

def passes_through (L : ℝ → ℝ → Prop) (P : Point) : Prop :=
  L P.1 P.2

def parallel_lines (L1 L2 : ℝ → ℝ → Prop) : Prop :=
  ∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L2 x2 y2 → (y1 - y2) = 5

def line_y_eq_c (c : ℝ) : ℝ → ℝ → Prop :=
  λ x y, y = c

def line_general_eq (a b c : ℝ) : ℝ → ℝ → Prop :=
  λ x y, a * x + b * y + c = 0

theorem find_parallel_lines (A B : Point) (d : ℝ):
  A = (1, 0) →
  B = (0, 5) →
  d = 5 →
  ∃ L1 L2 : (ℝ → ℝ → Prop), 
    (L1 = line_y_eq_c 0 ∧ L2 = line_y_eq_c 5) ∨ 
    (L1 = line_general_eq 5 (-12) (-5) ∧ L2 = line_general_eq 5 (-12) 60) ∧
    passes_through L1 A ∧
    passes_through L2 B ∧
    parallel_lines L1 L2 :=
by
  sorry

end find_parallel_lines_l334_334419


namespace least_five_digit_perfect_square_and_cube_l334_334454

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334454


namespace power_boat_travel_time_l334_334034

theorem power_boat_travel_time {r p t : ℝ} (h1 : r > 0) (h2 : p > 0) 
  (h3 : (p + r) * t + (p - r) * (9 - t) = 9 * r) : t = 4.5 :=
by
  sorry

end power_boat_travel_time_l334_334034


namespace least_five_digit_perfect_square_and_cube_l334_334507

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334507


namespace triangle_area_bound_l334_334330

noncomputable def area (A B C : Point) : ℝ := sorry

theorem triangle_area_bound {A B C M : Point} (hABC_eq : equilateral_triangle A B C) (hM_inside : inside_triangle M A B C) :
  area M A B + area M B C + area M C A ≤ (1/3) * area A B C :=
sorry

end triangle_area_bound_l334_334330


namespace probability_satisfied_estimation_l334_334647

def prob_dissatisfied_leave_angry := 0.80
def prob_satisfied_leave_positive := 0.15
def num_angry_reviews := 60
def num_positive_reviews := 20
def ratio_reviews := (num_positive_reviews : ℝ) / num_angry_reviews

theorem probability_satisfied_estimation 
    (p : ℝ) 
    (h₁ : prob_dissatisfied_leave_angry * (1 - p) = (num_positive_reviews / num_angry_reviews) * prob_satisfied_leave_positive * p) 
    : p = 0.64 := 
sorry

end probability_satisfied_estimation_l334_334647


namespace gerald_pfennigs_left_l334_334685

theorem gerald_pfennigs_left (cost_of_pie : ℕ) (farthings_initial : ℕ) (farthings_per_pfennig : ℕ) :
  cost_of_pie = 2 → farthings_initial = 54 → farthings_per_pfennig = 6 → 
  (farthings_initial / farthings_per_pfennig) - cost_of_pie = 7 :=
by
  intros h1 h2 h3
  sorry

end gerald_pfennigs_left_l334_334685


namespace find_f_3_l334_334749

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f_3 (a b : ℝ) (h : f (-3) a b = 10) : f 3 a b = -26 :=
by sorry

end find_f_3_l334_334749


namespace magnitude_of_a_plus_3b_l334_334728

open Real

-- Definitions and conditions as per the problem statement
variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Given conditions
def is_unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop := ∥v∥ = 1

def angle_between_unit_vectors (u v : EuclideanSpace ℝ (Fin 3)) (θ : ℝ) : Prop :=
  (u.inner v) = ∥u∥ * ∥v∥ * cos θ

-- Statement to be proven
theorem magnitude_of_a_plus_3b (ha : is_unit_vector a) (hb : is_unit_vector b)
    (hab : angle_between_unit_vectors a b (π / 3)) :
  ∥a + 3 • b∥ = sqrt 13 :=
sorry

end magnitude_of_a_plus_3b_l334_334728


namespace least_five_digit_is_15625_l334_334446

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334446


namespace find_b_l334_334770

-- Definitions of the functions f and g
def f (x : ℝ) : ℝ := Real.cos x
def g (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x + 1

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := -Real.sin x
def g' (x : ℝ) (b : ℝ) : ℝ := 2 * x + b

-- The main theorem to prove
theorem find_b (b : ℝ) : 
  (f 0 = g 0 b) ∧ (f' 0 = g' 0 b) → b = 0 :=
by
  sorry

end find_b_l334_334770


namespace inverse_function_point_l334_334765

theorem inverse_function_point
  (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (f : ℝ → ℝ) (h₃ : ∀ x, f x = a^(x + 2))
  (P : ℝ × ℝ) (h₄ : (f⁻¹).graph.contains P) :
  P = (1, -2)
:= sorry

end inverse_function_point_l334_334765


namespace chess_tournament_scores_l334_334158

def points (name : String) := Real

def total_points : Real := 12

variables (A D V S : Real)
variable (total_games : ℕ := 12)

axiom different_scores : A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S

axiom ranking : A > D ∧ D > V ∧ V > S

axiom equal_wins (A S : Real) : (A = 2 * win_points) ∧ (S = 2 * win_points)

axiom total_points_constraint : A + D + V + S = total_points

theorem chess_tournament_scores :
  A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 :=
by 
  sorry

end chess_tournament_scores_l334_334158


namespace least_five_digit_perfect_square_and_cube_l334_334433

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334433


namespace remainder_sum_pow_l334_334528

theorem remainder_sum_pow (n : ℕ) (h_n : n = 2003) : (2^n + n^2) % 7 = 5 := by
  rw h_n
  sorry

end remainder_sum_pow_l334_334528


namespace triangle_area_l334_334656

noncomputable def vector (x y z : ℝ) : ℝ × ℝ × ℝ :=
(x, y, z)

noncomputable def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(v.2.1 * w.2.2 - v.2.2 * w.2.1,
 v.2.2 * w.1 - v.1 * w.2.2,
 v.1 * w.2.1 - v.2.1 * w.1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

theorem triangle_area :
  let A := vector 2 1 (-1)
  let B := vector 3 0 3
  let C := vector 7 3 2
  let AB := (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2)
  let AC := (C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2)
  0.5 * magnitude (cross_product AB AC) = (1 / 2) * Real.sqrt 459 :=
by
  -- All the steps needed to prove the theorem here
  sorry

end triangle_area_l334_334656


namespace min_ab_value_l334_334790

theorem min_ab_value {M : Type} (A B C : M) (a b : ℝ) : 
  (∀ θ : ℝ, C = (sqrt(3) * cos(θ), sin(θ))) ∧
  (A = (0,1) ∧ B = (0,-1)) ∧
  (∀ m n : ℝ, (frac(m^2, 3) + n^2 = 1) ∧ (-sqrt(3) ≤ m ∧ m ≤ sqrt(3)) ∧ (-1 ≤ n ∧ n ≤ 1)) ∧
  (∀ m n : ℝ, M = (m, n) → (a = (1 - m) / n ∧ b = (m + 1) / n)) →
  (∀ n : ℝ, 0 < n → abs(a + b) ≥ 2) :=
by
  sorry

end min_ab_value_l334_334790


namespace minimum_value_of_difference_l334_334848

noncomputable def A' (x y z w : ℝ) := 
  real.sqrt (x + 3) + real.sqrt (y + 6) + real.sqrt (z + 11) + real.sqrt (w + 15)

noncomputable def B' (x y z w : ℝ) := 
  real.sqrt (x + 2) + real.sqrt (y + 3) + real.sqrt (z + 5) + real.sqrt (w + 8)

theorem minimum_value_of_difference (x y z w : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hw : 0 ≤ w) :
  (A' x y z w) ^ 2 - (B' x y z w) ^ 2 ≥ 100 :=
sorry

end minimum_value_of_difference_l334_334848


namespace symmetric_interval_95_l334_334946

variable {X : ℝ}
variable (μ σ : ℝ)

-- Conditions: X is normally distributed, practical values lie within (7, 10.6)
axiom normal_dist_X : ∀ (X : ℝ), (normalized_distribution X μ σ)
axiom X_values : ∀ (X : ℝ), X ∈ set.Icc 7 10.6

-- Prove the interval symmetric with respect to the mathematical expectation (μ)
-- where the wool shear yield is contained with 95% probability
theorem symmetric_interval_95 : 
  μ = 8.8 → σ = 0.6 → (7.624, 9.976) = (μ - 1.176, μ + 1.176) :=
begin
  assume hμ : μ = 8.8,
  assume hσ : σ = 0.6,
  sorry
end

end symmetric_interval_95_l334_334946


namespace find_fn_l334_334851

theorem find_fn (n: ℕ) (hn: n > 0) (a: Fin n → ℝ) (sum_integral: ∑ i, a i ∈ ℤ): 
  let f := if n % 2 = 0 then 0 else (1 / (2 * n)) in
  ∃ i, |a i - 1/2| ≥ f := 
sorry

end find_fn_l334_334851


namespace sleeping_bag_selling_price_l334_334184

variable (wholesale_cost : ℝ) (gross_profit_percentage : ℝ)

def gross_profit (wholesale_cost : ℝ) (gross_profit_percentage : ℝ) : ℝ :=
  (gross_profit_percentage / 100) * wholesale_cost

def selling_price (wholesale_cost : ℝ) (gross_profit : ℝ) : ℝ :=
  wholesale_cost + gross_profit

theorem sleeping_bag_selling_price :
  gross_profit 25 12 = 3 ∧ selling_price 25 3 = 28 :=
by 
  sorry

end sleeping_bag_selling_price_l334_334184


namespace correct_propositions_l334_334193

variables {l m : Type} {α β : Type}
variable [LinearOrderedField l] [LinearOrderedField m]

-- Define the conditions
def line_perpendicular_to_plane (l : Type) (α : Type) : Prop := sorry
def line_contained_in_plane (m : Type) (β : Type) : Prop := sorry

-- Define the propositions
def proposition1 (α β : Type) : Prop := ∀ (l m : Type), (line_perpendicular_to_plane l α ∧ line_contained_in_plane m β) → (α ∥ β → l ⊥ m)
def proposition3 (l m : Type) : Prop := ∀ (α β : Type), (line_perpendicular_to_plane l α ∧ line_contained_in_plane m β) → (l ∥ m → α ⊥ β)

-- Combine the proofs
theorem correct_propositions (l m α β : Type) [LinearOrderedField l] [LinearOrderedField m] :
  (line_perpendicular_to_plane l α) ∧ (line_contained_in_plane m β) → 
  (proposition1 α β) ∧ (proposition3 l m) :=
by
  sorry

end correct_propositions_l334_334193


namespace perfect_days_in_2018_l334_334623

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_perfect_day (year month day : ℕ) : Prop :=
  sum_of_digits year = sum_of_digits month + sum_of_digits day

theorem perfect_days_in_2018 : (Finset.range (12 + 1)).sum (λ m, (Finset.range (nat.days_in_month 2018 (fin.val m))).count (λ d, is_perfect_day 2018 m d)) = 36 :=
sorry

end perfect_days_in_2018_l334_334623


namespace domain_of_log_function_l334_334363

theorem domain_of_log_function :
  {x : ℝ | (3 - x) * (x + 1) > 0} = set.Ioo (-1 : ℝ) 3 :=
sorry

end domain_of_log_function_l334_334363


namespace least_five_digit_is_15625_l334_334443

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334443


namespace determine_a_l334_334073

noncomputable def f (a b c x : ℝ) := real.sqrt (a * x^2 + b * x + c)

theorem determine_a (a b c : ℝ) (D : set ℝ) :
  (∀ (t : ℝ), t ∈ D → ∃ s ∈ D, (a, f a b c t) = (s, f a b c s)) →
  a < 0 →
  (∃ V: ℝ × ℝ, (\<forall s t ∈ D, (s, f a b c s) ∈ V ∧ (t, f a b c t) ∈ V) →
  a = -8 :=
by
  sorry

end determine_a_l334_334073


namespace volume_of_original_cube_l334_334325

theorem volume_of_original_cube (s : ℕ) (h : (s - 2) * s * (s + 2) = s^3 - 12) : s^3 = 27 :=
begin
  have h_eq : s^3 - ((s - 2) * s * (s + 2)) = 12 := by rw [h, sub_eq_add_neg, ← sub_add, add_neg_eq_sub],
  have h_s : s * (s^2 - 4) = s^3 - 4 * s := by ring,
  rw [← h_s] at h_eq,
  rw sub_eq_iff_eq_add at h_eq,
  have h_s_simplified : 4 * s = 12 := by linarith,
  have s_eq : s = 3 := by linarith,
  rw [s_eq],
  norm_num,
end

end volume_of_original_cube_l334_334325


namespace coloring_equilateral_triangle_l334_334929

theorem coloring_equilateral_triangle :
  ∀ (A B C : Type) (color : A → Type) (d : A → A → ℝ),
  (∀ x y, d x y = 1 → color x = color y) :=
by sorry

end coloring_equilateral_triangle_l334_334929


namespace find_missing_number_l334_334106

theorem find_missing_number (x : ℤ) : x + 64 = 16 → x = -48 := by
  intro h
  linarith

end find_missing_number_l334_334106


namespace multiple_of_sales_total_l334_334297

theorem multiple_of_sales_total
  (A : ℝ)
  (M : ℝ)
  (h : M * A = 0.3125 * (11 * A + M * A)) :
  M = 5 :=
by
  sorry

end multiple_of_sales_total_l334_334297


namespace triangle_angle_range_l334_334774

theorem triangle_angle_range 
  (a b c : ℝ) (A B C : ℝ) 
  (h_cos : cos(2 * C) - cos(2 * A) = 2 * sin(π / 3 + C) * sin(π / 3 - C))
  (h_a : a = sqrt 3)
  (h_b_ge_a : b ≥ a)
  (h_sine_law : a / sin A = b / sin B)
  (h_A_acute : 0 < A ∧ A < π / 2)
  : A = π / 3 ∧ ∃ (B : ℝ), π / 3 ≤ B ∧ B < 2 * π / 3 ∧ 2 * b - c ∈ set.Ico (sqrt 3) (2 * sqrt 3) := 
by
  sorry

end triangle_angle_range_l334_334774


namespace least_five_digit_perfect_square_and_cube_l334_334460

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334460


namespace orvin_max_balloons_l334_334879

theorem orvin_max_balloons (r : ℝ) (h : r > 0) : 
  let regular_price := r in
  let sale_price := r / 2 in 
  let total_money := 40 * regular_price in
  let pair_cost := regular_price + sale_price in
  let max_pairs := (total_money / pair_cost).to_nat in
  max_pairs * 2 = 52 :=
sorry

end orvin_max_balloons_l334_334879


namespace trigonometric_identity_l334_334062

noncomputable def cos190 := Real.cos (190 * Real.pi / 180)
noncomputable def sin290 := Real.sin (290 * Real.pi / 180)
noncomputable def cos40 := Real.cos (40 * Real.pi / 180)
noncomputable def tan10 := Real.tan (10 * Real.pi / 180)

theorem trigonometric_identity :
  (cos190 * (1 + Real.sqrt 3 * tan10)) / (sin290 * Real.sqrt (1 - cos40)) = 2 * Real.sqrt 2 :=
by
  sorry

end trigonometric_identity_l334_334062


namespace least_five_digit_perfect_square_and_cube_l334_334449

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334449


namespace problem_mod_1000_l334_334304

noncomputable def M : ℕ := Nat.choose 18 9

theorem problem_mod_1000 : M % 1000 = 620 := by
  sorry

end problem_mod_1000_l334_334304


namespace b_is_zero_if_f_is_even_l334_334771

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f (x)

theorem b_is_zero_if_f_is_even (b : ℝ) :
  is_even_function (λ x, x * (x + b)) → b = 0 :=
by
  intros h
  sorry

end b_is_zero_if_f_is_even_l334_334771


namespace student_marks_in_geography_l334_334606

theorem student_marks_in_geography
  (marks_history_and_government : ℕ := 60)
  (marks_art : ℕ := 72)
  (marks_computer_science : ℕ := 85)
  (marks_modern_literature : ℕ := 80)
  (average_marks : ℝ := 70.6)
  (number_of_subjects : ℕ := 5) :
  (353 - (marks_history_and_government + marks_art + marks_computer_science + marks_modern_literature)) = 56 := 
begin
  sorry,
end

end student_marks_in_geography_l334_334606


namespace chess_tournament_distribution_l334_334172

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l334_334172


namespace part1_extreme_value_part2_range_of_a_l334_334746

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem part1_extreme_value :
  ∃ x : ℝ, f x = -1 :=
  sorry

theorem part2_range_of_a :
  ∀ x > 0, ∃ a : ℝ, f x ≥ x + Real.log x + a + 1 → a ≤ 1 :=
  sorry

end part1_extreme_value_part2_range_of_a_l334_334746


namespace hyperbola_equation_correct_l334_334120

-- Define the parameters for the hyperbola
def a : ℕ := 4
def b : ℕ := 5

-- Standard equation of the hyperbola with foci on the y-axis
def hyperbola_in_standard_form : string := "y² / 16 - x² / 25 = 1"

theorem hyperbola_equation_correct (a : ℕ) (b : ℕ) (ha : a = 4) (hb : b = 5) :
  hyperbola_in_standard_form = "y² / 16 - x² / 25 = 1" :=
by
  rw [←ha, ←hb]
  sorry

end hyperbola_equation_correct_l334_334120


namespace least_five_digit_perfect_square_and_cube_l334_334457

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334457


namespace min_value_of_quadratic_l334_334924

theorem min_value_of_quadratic (x : ℝ) : ∃ x_min, x^2 + 4 * x + 5 = 1 :=
by {
  let f := λ x, x^2 + 4 * x + 5,
  have h : ∃ x_min, ∀ x, f x_min ≤ f x,
  { use -2,
    sorry },
  cases h with x_min h_min,
  use f x_min,
  exact h_min x,
}

end min_value_of_quadratic_l334_334924


namespace identify_negative_values_l334_334221

-- Define the trigonometric expressions as functions
def expression1 (x : ℝ) : ℝ := Real.sin x
def expression2 (x : ℝ) : ℝ := Real.tan x * Real.sin x
def expression3 (x : ℝ) : ℝ := Real.sin x / Real.tan x
def expression4 (x : ℝ) : ℝ := Real.sin (Real.abs x)

-- Define the problem in Lean 4 that the expressions must fulfill
theorem identify_negative_values :
  expression1 (1125 * Real.pi / 180) > 0 ∧
  expression2 (37 / 12 * Real.pi) < 0 ∧
  expression3 4 < 0 ∧
  expression4 (-1) > 0 
:=
sorry

end identify_negative_values_l334_334221


namespace least_five_digit_perfect_square_and_cube_l334_334483

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334483


namespace like_terms_product_l334_334711

theorem like_terms_product :
  ∀ (m n : ℕ),
    (-x^3 * y^n) = (3 * x^m * y^2) → (m = 3 ∧ n = 2) → m * n = 6 :=
by
  intros m n h1 h2
  sorry

end like_terms_product_l334_334711


namespace half_circle_perimeter_l334_334796

noncomputable def radius := 7
noncomputable def circumference := 2 * (Real.pi) * radius

theorem half_circle_perimeter :
  let OP := radius,
      OQ := radius,
      arc_PQ := (circumference / 2)
  in OP + OQ + arc_PQ = 14 + 7 * Real.pi :=
by
  let OP := radius
  let OQ := radius
  let arc_PQ := (circumference / 2)
  exact sorry

end half_circle_perimeter_l334_334796


namespace least_five_digit_perfect_square_cube_l334_334523

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334523


namespace chess_tournament_scores_l334_334152

def points (name : String) := Real

def total_points : Real := 12

variables (A D V S : Real)
variable (total_games : ℕ := 12)

axiom different_scores : A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S

axiom ranking : A > D ∧ D > V ∧ V > S

axiom equal_wins (A S : Real) : (A = 2 * win_points) ∧ (S = 2 * win_points)

axiom total_points_constraint : A + D + V + S = total_points

theorem chess_tournament_scores :
  A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 :=
by 
  sorry

end chess_tournament_scores_l334_334152


namespace alex_tree_expected_value_m_plus_n_l334_334050

-- Definition of the expected size of the subtree of a randomly selected vertex after adding each vertex
noncomputable def expected_size (n : ℕ) : ℚ :=
  ∑ i in (Finset.range n).map Nat.succ, 1/i

-- Final statement where we want to find m + n for E[10]
theorem alex_tree_expected_value_m_plus_n :
  let frac := expected_size 10 in
  let m := frac.num in
  let n := frac.denom in
  Nat.gcd m n = 1 → m + n = 9901 :=
by
  sorry

end alex_tree_expected_value_m_plus_n_l334_334050


namespace extreme_value_of_f_range_of_a_l334_334748

noncomputable def f (x : ℝ) : ℝ := x * real.exp (x + 1)

theorem extreme_value_of_f : f (-1) = -1 := by
  -- The proof would go here
  sorry

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → f x ≥ x + real.log x + a + 1) → a ≤ 1 := by
  -- The proof would go here
  sorry

end extreme_value_of_f_range_of_a_l334_334748


namespace shortest_distance_to_y_axis_l334_334405

noncomputable def parabola : set (ℝ × ℝ) := {p | ∃ x, p = (x, sqrt (8 * x)) ∨ p = (x, -sqrt (8 * x))}

theorem shortest_distance_to_y_axis :
  ∀ (A B : ℝ × ℝ), 
  (A ∈ parabola) → (B ∈ parabola) → 
  (let d := dist A B in d = 10) →
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  abs P.1 = 3 :=
by
  intros A B hA hB hAB P
  sorry

end shortest_distance_to_y_axis_l334_334405


namespace k_is_perfect_square_l334_334976

theorem k_is_perfect_square (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (k : ℕ)
  (h_k : k = (m + n)^2 / (4 * m * (m - n)^2 + 4)) 
  (h_int_k : k * (4 * m * (m - n)^2 + 4) = (m + n)^2) :
  ∃ x : ℕ, k = x^2 := 
sorry

end k_is_perfect_square_l334_334976


namespace sin_4θ_eq_neg_4_over_9_l334_334759

noncomputable def geometric_series_sincos (θ : ℝ) : Prop :=
  ∑' n, (sin θ)^(2*n) = 3

theorem sin_4θ_eq_neg_4_over_9 (θ : ℝ) (h : geometric_series_sincos θ) : sin (4 * θ) = -4/9 :=
  sorry

end sin_4θ_eq_neg_4_over_9_l334_334759


namespace tax_free_amount_l334_334046

theorem tax_free_amount (X : ℝ) (total_value : ℝ) (tax_paid : ℝ) (tax_rate : ℝ) 
(h1 : total_value = 1720) 
(h2 : tax_paid = 134.4) 
(h3 : tax_rate = 0.12) 
(h4 : tax_paid = tax_rate * (total_value - X)) 
: X = 600 := 
sorry

end tax_free_amount_l334_334046


namespace park_area_in_square_miles_l334_334581

theorem park_area_in_square_miles 
  (scale_ratio_miles_per_inch : ℝ)
  (diagonal_map_inches : ℝ) 
  (diagonal_actual_miles : ℝ)
  (area_actual_square_miles : ℝ) :
  scale_ratio_miles_per_inch = 250 →
  diagonal_map_inches = 10 →
  diagonal_actual_miles = diagonal_map_inches * scale_ratio_miles_per_inch →
  area_actual_square_miles = (real.sqrt 3) / 2 * (diagonal_actual_miles ^ 2) →
  area_actual_square_miles = 3125000 * (real.sqrt 3) :=
begin
  intros h1 h2 h3 h4,
  rw h1 at h3,
  rw h2 at h3,
  rw h3 at h4,
  rw mul_pow at h4,
  rw real.mul_div_cancel (_root_.pow 2500 2) (le_of_lt real.sqrt_ne_zero'.1),
  exact h4,
  sorry
end

end park_area_in_square_miles_l334_334581


namespace profit_per_piece_max_profit_personnel_distribution_l334_334570

-- Profit definitions and conditions
def profit_A : ℝ := 15
def profit_B : ℝ := 120

-- Number of people conditions
def total_people : ℕ := 65
def min_production_B : ℕ := 5

-- Function to calculate total profit
def total_profit (m : ℕ) : ℝ :=
  let profit_per_B := 130 - 2 * m in
  15 * 2 * (65 - m) + profit_per_B * m

-- Results to be proved
theorem profit_per_piece :
  profit_A = 15 ∧ profit_B = 120 := by
  sorry

theorem max_profit :
  ∃ (m : ℕ), m = 25 ∧ total_profit 25 = 3200 := by
  sorry

theorem personnel_distribution :
  ∃ (m : ℕ), m = 25 ∧ (total_people - m = 40 ∧ m = 25) := by
  sorry

end profit_per_piece_max_profit_personnel_distribution_l334_334570


namespace at_least_two_sums_equal_l334_334604

noncomputable def generateGrid (n : ℕ) : list (list ℕ) := sorry

noncomputable def sumRow (row : list ℕ) : ℕ := row.sum

noncomputable def sumColumn (grid : list (list ℕ)) (j : ℕ) : ℕ := 
  (list.map (λ row, row.get! j) grid).sum

noncomputable def sumMainDiagonal (grid : list (list ℕ)) : ℕ := 
  (list.map_with_index (λ i row, row.get! i) grid).sum

noncomputable def sumAntiDiagonal (grid : list (list ℕ)) : ℕ := 
  (list.map_with_index (λ i row, row.get! (grid.length - 1 - i)) grid).sum

theorem at_least_two_sums_equal 
  (grid : list (list ℕ)) 
  (hgrid_size : grid.length = 10 ∧ ∀ row, row ∈ grid → row.length = 10)
  (hgrid_values : ∀ row, row ∈ grid → ∀ (value : ℕ), value ∈ row → value = 1 ∨ value = 2 ∨ value = 3) :
  let sums := 
    (list.map sumRow grid) ++ 
    (list.map (λ j, sumColumn grid j) (list.range 10)) ++ 
    [sumMainDiagonal grid, sumAntiDiagonal grid] in 
    ∃ x y, x ∈ sums ∧ y ∈ sums ∧ x = y ∧ x ≠ y := sorry

end at_least_two_sums_equal_l334_334604


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334493

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334493


namespace cost_price_percentage_l334_334911

theorem cost_price_percentage (MP CP SP : ℝ) (h1 : SP = 0.88 * MP) (h2 : SP = 1.375 * CP) :
  (CP / MP) * 100 = 64 :=
by
  sorry

end cost_price_percentage_l334_334911


namespace ratio_incorrect_to_true_l334_334609

noncomputable def true_average (scores : List ℝ) (h : scores.length = 50) : ℝ :=
  (scores.sum) / 50

noncomputable def incorrect_average (scores : List ℝ) (h : scores.length = 50) : ℝ :=
  let A := (scores.sum) / 50 in
  (scores.sum + A) / 51

theorem ratio_incorrect_to_true (scores : List ℝ) (h : scores.length = 50) :
  (incorrect_average scores h) / (true_average scores h) = 1 :=
by
  sorry

end ratio_incorrect_to_true_l334_334609


namespace proof_problem_l334_334226

-- Define the parametric equations of line l
def parametric_eq_x (t : ℝ) : ℝ := (sqrt 2 / 2) * t + 1
def parametric_eq_y (t : ℝ) : ℝ := -(sqrt 2 / 2) * t

-- Define the polar coordinate equation of circle C
def polar_eq (θ : ℝ) : ℝ := 2 * sqrt 2 * cos (θ + π / 4)

-- Define the final theorem statement
theorem proof_problem :
  (∀ t : ℝ, parametric_eq_x t + parametric_eq_y t - 1 = 0) ∧
  (∀ ρ θ : ℝ, (ρ = polar_eq θ) → (ρ^2 = 2 * ρ * cos θ - 2 * ρ * sin θ) →
    ∃ x y : ℝ, (x^2 + y^2 - 2 * x + 2 * y = 0)) ∧
  (∃ t1 t2 : ℝ, (t1 + t2 = sqrt 2) ∧ (t1 * t2 = -1) ∧ (|1 - t1| + |1 - t2| = sqrt 6)) :=
sorry  -- Proof omitted

end proof_problem_l334_334226


namespace subtraction_of_decimals_l334_334423

theorem subtraction_of_decimals : 58.3 - 0.45 = 57.85 := by
  sorry

end subtraction_of_decimals_l334_334423


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334498

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334498


namespace ratio_AB_to_w_l334_334267

theorem ratio_AB_to_w (a : ℝ) (h_pos : 0 < a) :
  let ABC_area := (2 * a * a) / 2,
      painted_area := ABC_area / 3,
      w := painted_area / a
  in (2 * a) / w = 6 :=
by
  sorry

end ratio_AB_to_w_l334_334267


namespace intersection_M_N_l334_334234

noncomputable def M (x : ℝ) : Prop := 2^(x - 1) < 1
noncomputable def N (x : ℝ) : Prop := Real.log x / Real.log (1 / 2) < 1

theorem intersection_M_N :
  { x : ℝ | M x } ∩ { x : ℝ | N x } = set.Ioo (1 / 2) 1 := by
  sorry

end intersection_M_N_l334_334234


namespace evaluate_floor_ceil_l334_334649

def floor (x : ℝ) : ℤ := ⌊x⌋
def ceil (x : ℝ) : ℤ := ⌈x⌉

theorem evaluate_floor_ceil : floor (-3.72) + ceil (34.1) = 31 :=
by
  sorry

end evaluate_floor_ceil_l334_334649


namespace exists_nice_game_for_valid_n_exists_tiresome_game_for_valid_n_l334_334576

-- Definitions based on the problem conditions
def is_nice_game (n : Nat) (P : List (Nat × Nat) → List (Nat × Nat)) : Prop :=
  (∀ i < n, i ∉ P)  -- No player has their own ball at the end

def is_tiresome_game (n : Nat) (P : List (Nat × Nat) → List (Nat × Nat)) : Prop :=
  (∀ i < n, i ∈ P)  -- Every player ends up with her original ball

-- Theorem statements
theorem exists_nice_game_for_valid_n : ∀ n, n ≠ 3 → ∃ P, is_nice_game n P :=
by sorry

theorem exists_tiresome_game_for_valid_n : ∀ n, (n % 4 = 0 ∨ n % 4 = 1) → ∃ P, is_tiresome_game n P :=
by sorry

end exists_nice_game_for_valid_n_exists_tiresome_game_for_valid_n_l334_334576


namespace solve_inequality_l334_334895

def inequality_solution (x : ℝ) : Prop := |2 * x - 1| - x ≥ 2 

theorem solve_inequality (x : ℝ) : 
  inequality_solution x ↔ (x ≥ 3 ∨ x ≤ -1/3) :=
by sorry

end solve_inequality_l334_334895


namespace second_perimeter_equals_28_l334_334291

-- Definitions and conditions
def l (w : ℝ) := 2 * w
def first_perimeter (w : ℝ) := 2 * (3 * l(w) + 2 * w)
def second_perimeter (w : ℝ) := 2 * (6 * l(w) + w)

-- The given condition for the first perimeter
axiom first_perimeter_eq_20 (w : ℝ) : first_perimeter(w) = 20

-- The goal is to show that the perimeter of the second arrangement equals 28 cm
theorem second_perimeter_equals_28 : ∀ w : ℝ, first_perimeter(w) = 20 → second_perimeter(w) = 28 := by
  intros
  sorry

end second_perimeter_equals_28_l334_334291


namespace platform_length_l334_334012

theorem platform_length
  (train_length : ℕ)
  (time_pole : ℕ)
  (time_platform : ℕ)
  (h_train_length : train_length = 300)
  (h_time_pole : time_pole = 18)
  (h_time_platform : time_platform = 39) :
  ∃ (platform_length : ℕ), platform_length = 350 :=
by
  sorry

end platform_length_l334_334012


namespace scores_are_correct_l334_334143

variable
  Andrey_score : ℝ
  Dima_score : ℝ
  Vanya_score : ℝ
  Sasha_score : ℝ

-- Conditions
axiom andrey_first : Andrey_score > Dima_score ∧ Andrey_score > Vanya_score ∧ Andrey_score > Sasha_score
axiom dima_second : Dima_score > Vanya_score ∧ Dima_score > Sasha_score
axiom vanya_third : Vanya_score > Sasha_score
axiom unique_scores : Andrey_score ≠ Dima_score ∧ Andrey_score ≠ Vanya_score ∧ Andrey_score ≠ Sasha_score ∧ Dima_score ≠ Vanya_score ∧ Dima_score ≠ Sasha_score ∧ Vanya_score ≠ Sasha_score
axiom total_points : Andrey_score + Dima_score + Vanya_score + Sasha_score = 12
axiom andrey_sasha_wins : Andrey_score = 4 ∧ Sasha_score = 2

-- Conclusion
theorem scores_are_correct :
  Andrey_score = 4 ∧ Dima_score = 3.5 ∧ Vanya_score = 2.5 ∧ Sasha_score = 2 :=
  sorry

end scores_are_correct_l334_334143


namespace height_of_cone_correct_l334_334127

-- Defining the conditions
def sphere_radius : ℝ := 1
def edge_length : ℝ := 2 
def tetrahedron_height : ℝ := 2 * real.sqrt (2 / 3)

noncomputable def sin_alpha : ℝ := (real.sqrt 3) / 3
noncomputable def distance_A_O4 : ℝ := 2 / sin_alpha
def height_of_cone : ℝ := sphere_radius + tetrahedron_height + distance_A_O4

-- The formal statement that needs to be proven
theorem height_of_cone_correct :
  height_of_cone = 1 + 2 * real.sqrt (2 / 3) + real.sqrt 3 :=
sorry

end height_of_cone_correct_l334_334127


namespace find_coordinates_of_N_l334_334191

-- Definitions based on conditions provided
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4
def point_M (x y : ℝ) : Prop := x = 1 ∧ y = 1
def constant_ratio_property (m n : ℝ) : Prop :=
  ∀ (x0 y0 : ℝ), circle x0 y0 →
  ∃ λ : ℝ, λ > 0 ∧
  λ^2 = ((x0 - m)^2 + (y0 - n)^2) / ((x0 - 1)^2 + (y0 - 1)^2)

-- The statement to prove the coordinates of N
theorem find_coordinates_of_N :
  (m n : ℝ) (hN : m ≠ 1 ∨ n ≠ 1),
  constant_ratio_property m n → 
  m = 2 ∧ n = 2 :=
sorry

end find_coordinates_of_N_l334_334191


namespace shape_is_cone_given_phi_eq_d_l334_334671

-- Define the spherical coordinate system and necessary variables
variables {ρ θ d : ℝ}

-- Define the shape condition in spherical coordinates
def shape_described_by (d : ℝ) : Prop :=
  ∀ (ρ θ : ℝ), (0 ≤ ρ) → (0 ≤ θ) → (θ < 2 * π) → ((0 ≤ d) → (d ≤ π) → φ = d)

-- Define the conclusion that this shape is a cone
def is_cone (d : ℝ) : Prop :=
  shape_described_by d

-- Main theorem statement
theorem shape_is_cone_given_phi_eq_d (d : ℝ) : is_cone d :=
  sorry

end shape_is_cone_given_phi_eq_d_l334_334671


namespace unique_intersection_point_l334_334379

theorem unique_intersection_point (k : ℝ) :
x = k ->
∃ x : ℝ, x = -3*y^2 - 4*y + 7 -> ∃ k : ℝ, k = 25/3 -> y = 0 -> x = k

end unique_intersection_point_l334_334379


namespace seq_abs_sum_l334_334805

noncomputable def seq (n : ℕ) : ℤ := sorry

theorem seq_abs_sum :
  (∀ n : ℕ, 2 * seq (n + 1) - seq (n + 2) = seq n) →
  seq 1 = 10 →
  seq 2 = 8 →
  seq 4 = 2 →
  (|seq 1| + |seq 2| + |seq 3| + |seq 4| + |seq 5| + |seq 6| + |seq 7| + |seq 8| + |seq 9| + |seq 10| = 50) :=
begin
  sorry,
end

end seq_abs_sum_l334_334805


namespace sum_of_integer_solutions_l334_334537

theorem sum_of_integer_solutions :
  (∑ x in { x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 }, x) = 18 :=
by
  sorry

end sum_of_integer_solutions_l334_334537


namespace sequence_a2019_l334_334603

theorem sequence_a2019 (a : ℕ → ℚ) (h1 : a 1 = 1) (h2 : a 2 = 3/7)
  (h_rec : ∀ n ≥ 3, a n = (a (n-2) * a (n-1)) / (2 * a (n-2) - a (n-1)))
  : ∃ p q : ℕ, nat.coprime p q ∧ a 2019 = p / q ∧ p + q = 8078 := 
sorry

end sequence_a2019_l334_334603


namespace sequence_a2019_l334_334602

theorem sequence_a2019 (a : ℕ → ℚ) (h1 : a 1 = 1) (h2 : a 2 = 3/7)
  (h_rec : ∀ n ≥ 3, a n = (a (n-2) * a (n-1)) / (2 * a (n-2) - a (n-1)))
  : ∃ p q : ℕ, nat.coprime p q ∧ a 2019 = p / q ∧ p + q = 8078 := 
sorry

end sequence_a2019_l334_334602


namespace intersection_set_eq_l334_334233

noncomputable def M : set ℝ := {x | x * (x - 5) ≤ 6}
noncomputable def N : set ℝ := {x | 0 ≤ x}

theorem intersection_set_eq : 
  M ∩ N = {x | 0 ≤ x ∧ x ≤ 6} := 
by {
    sorry
}

end intersection_set_eq_l334_334233


namespace avoid_vertices_l334_334706

theorem avoid_vertices (A_f A_c : ℝ) (h : A_f < A_c) :
  ∃ (x y : ℝ), ∀ (i j : ℤ), 
  ¬ (x ∈ set.Icc (i * sqrt(A_c)) ((i + 1) * sqrt(A_c)) ∧ 
     y ∈ set.Icc (j * sqrt(A_c)) ((j + 1) * sqrt(A_c)) ∧ 
     F (x, y)) := 
by
  sorry

end avoid_vertices_l334_334706


namespace total_oil_leak_l334_334616

theorem total_oil_leak (oil_leak_before : ℕ) (oil_leak_during : ℕ) (total_leak : ℕ) :
  oil_leak_before = 6522 → 
  oil_leak_during = 5165 → 
  total_leak = (oil_leak_before + oil_leak_during) → 
  total_leak = 11687 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_oil_leak_l334_334616


namespace scores_are_correct_l334_334136

variable
  Andrey_score : ℝ
  Dima_score : ℝ
  Vanya_score : ℝ
  Sasha_score : ℝ

-- Conditions
axiom andrey_first : Andrey_score > Dima_score ∧ Andrey_score > Vanya_score ∧ Andrey_score > Sasha_score
axiom dima_second : Dima_score > Vanya_score ∧ Dima_score > Sasha_score
axiom vanya_third : Vanya_score > Sasha_score
axiom unique_scores : Andrey_score ≠ Dima_score ∧ Andrey_score ≠ Vanya_score ∧ Andrey_score ≠ Sasha_score ∧ Dima_score ≠ Vanya_score ∧ Dima_score ≠ Sasha_score ∧ Vanya_score ≠ Sasha_score
axiom total_points : Andrey_score + Dima_score + Vanya_score + Sasha_score = 12
axiom andrey_sasha_wins : Andrey_score = 4 ∧ Sasha_score = 2

-- Conclusion
theorem scores_are_correct :
  Andrey_score = 4 ∧ Dima_score = 3.5 ∧ Vanya_score = 2.5 ∧ Sasha_score = 2 :=
  sorry

end scores_are_correct_l334_334136


namespace coloring_equilateral_triangle_l334_334930

theorem coloring_equilateral_triangle :
  ∀ (A B C : Type) (color : A → Type) (d : A → A → ℝ),
  (∀ x y, d x y = 1 → color x = color y) :=
by sorry

end coloring_equilateral_triangle_l334_334930


namespace range_m_inequality_l334_334919

theorem range_m_inequality (m : ℝ) :
    (∀ x : ℝ, x^2 - m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
begin
  sorry
end

end range_m_inequality_l334_334919


namespace least_five_digit_perfect_square_cube_l334_334518

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334518


namespace line_parabola_intersection_one_point_l334_334386

theorem line_parabola_intersection_one_point (k : ℝ) :
  (∃ y : ℝ, (-3 * y^2 - 4 * y + 7 = k) ∧ ∀ y1 y2 : ℝ, ( 3 * y1^2 + 4 * y1 + (k - 7) = 0 → 3 * y2^2 + 4 * y2 + (k - 7) = 0 → y1 = y2)) ↔ (k = 25 / 3) :=
by
  sorry

end line_parabola_intersection_one_point_l334_334386


namespace trigonometric_identity_l334_334688

theorem trigonometric_identity (x : ℝ) (h : Real.sin (x + Real.pi / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 :=
by
  sorry

end trigonometric_identity_l334_334688


namespace sum_of_numerator_and_denominator_prob_divisible_by_10_7_l334_334889

theorem sum_of_numerator_and_denominator_prob_divisible_by_10_7 :
  let S := {x : ℕ // ∃ m n : ℕ, 0 ≤ m ∧ m ≤ 5 ∧ 0 ≤ n ∧ n ≤ 5 ∧ x = 2^m * 5^n} in
  (∃ a b : ℕ, a.gcd b = 1 ∧
    (prob (λ ⦃x y : S⦄, x ≠ y ∧ 2^7 ∣ (x.val * y.val) ∧ 5^7 ∣ (x.val * y.val)) = a / b) ∧
    a + b = 349)
:=
by
  -- definitions and actual proof will go here
  sorry

end sum_of_numerator_and_denominator_prob_divisible_by_10_7_l334_334889


namespace exists_same_color_points_one_meter_apart_l334_334934

-- Declare the colors as an enumeration
inductive Color
| red : Color
| black : Color

-- Define the function that assigns a color to each point in the plane
def color (point : ℝ × ℝ) : Color := sorry

-- The theorem to be proven
theorem exists_same_color_points_one_meter_apart :
  ∃ x y : ℝ × ℝ, x ≠ y ∧ dist x y = 1 ∧ color x = color y :=
sorry

end exists_same_color_points_one_meter_apart_l334_334934


namespace min_x9_minus_x1_l334_334855

theorem min_x9_minus_x1
  (x : Fin 9 → ℕ)
  (h_pos : ∀ i, x i > 0)
  (h_sorted : ∀ i j, i < j → x i < x j)
  (h_sum : (Finset.univ.sum x) = 220) :
    ∃ x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℕ,
    x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5 ∧ x5 < x6 ∧ x6 < x7 ∧ x7 < x8 ∧ x8 < x9 ∧
    (x1 + x2 + x3 + x4 + x5 = 110) ∧
    x1 = x 0 ∧ x2 = x 1 ∧ x3 = x 2 ∧ x4 = x 3 ∧ x5 = x 4 ∧ x6 = x 5 ∧ x7 = x 6 ∧ x8 = x 7 ∧ x9 = x 8
    ∧ (x9 - x1 = 9) :=
sorry

end min_x9_minus_x1_l334_334855


namespace quadratic_single_root_pos_value_l334_334259

theorem quadratic_single_root_pos_value (m : ℝ) (h1 : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
sorry

end quadratic_single_root_pos_value_l334_334259


namespace cristina_nicky_head_start_l334_334873

theorem cristina_nicky_head_start (s_c s_n : ℕ) (t d : ℕ) 
  (h1 : s_c = 5) 
  (h2 : s_n = 3) 
  (h3 : t = 30)
  (h4 : d = s_n * t):
  d = 90 := 
by
  sorry

end cristina_nicky_head_start_l334_334873


namespace chess_tournament_points_l334_334149

theorem chess_tournament_points
  (points : String → ℝ)
  (total_points : points "Andrey" + points "Dima" + points "Vanya" + points "Sasha" = 12)
  (distinct_points : 
    points "Andrey" ≠ points "Dima" ∧ 
    points "Andrey" ≠ points "Vanya" ∧ 
    points "Andrey" ≠ points "Sasha" ∧ 
    points "Dima" ≠ points "Vanya" ∧ 
    points "Dima" ≠ points "Sasha" ∧ 
    points "Vanya" ≠ points "Sasha")
  (order : 
    points "Andrey" > points "Dima" ∧ 
    points "Dima" > points "Vanya" ∧ 
    points "Vanya" > points "Sasha")
  (same_wins :
    let games_won (student : String) := (points student - 3) / 0.5 in
    games_won "Andrey" = games_won "Sasha") :
  points "Andrey" = 4 ∧ points "Dima" = 3.5 ∧ points "Vanya" = 2.5 ∧ points "Sasha" = 2 :=
by
  sorry

end chess_tournament_points_l334_334149


namespace chess_tournament_solution_l334_334166

def chess_tournament_points (points : List ℝ) : Prop :=
  let andrey := points[0]
  let dima := points[1]
  let vanya := points[2]
  let sasha := points[3]
  andrey = 4 ∧ dima = 3.5 ∧ vanya = 2.5 ∧ sasha = 2

axiom chess_tournament_conditions (points : List ℝ) :
  -- Andrey secured first place, Dima secured second, Vanya secured third, and Sasha secured fourth.
  List.Nodup points ∧
  points.length = 4 ∧
  (∀ p, p ∈ points → p = 4 ∨ p = 3.5 ∨ p = 2.5 ∨ p = 2) ∧
  -- Andrey and Sasha won the same number of games.
  (points[0] ≠ points[1] ∧ points[0] ≠ points[2] ∧ points[0] ≠ points[3] ∧
   points[1] ≠ points[2] ∧ points[1] ≠ points[3] ∧
   points[2] ≠ points[3])

theorem chess_tournament_solution (points : List ℝ) :
  chess_tournament_conditions points → chess_tournament_points points :=
by
  sorry

end chess_tournament_solution_l334_334166


namespace find_x_l334_334108

theorem find_x (x : ℝ) (h : log 6 (5 * x) = 3) : x = 216 / 5 :=
by 
  sorry

end find_x_l334_334108


namespace sum_of_roots_l334_334085

theorem sum_of_roots (a b c : ℝ) (ha : a = -4) (hb : b = -18) (hc : c = 81) :
  let r₁ r₂ := let D := b^2 - 4*a*c in
               if D ≥ 0 then 
                 ((-b + Real.sqrt D) / (2*a), (-b - Real.sqrt D) / (2*a))
               else (0, 0)
  in r₁ + r₂ = 4.5 :=
by
  have h : (r₁, r₂) = (4.5, 0) ∨ (r₁, r₂) = (0, 4.5), from sorry
  cases h; simp [h]

end sum_of_roots_l334_334085


namespace tangent_lines_line_intersects_circle_l334_334731

noncomputable theory

open Real 

def circle (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 4

def point_M : ℝ × ℝ := (3, 1)

def line (a : ℝ) (x y : ℝ) : Prop := a * x - y + 3 = 0

theorem tangent_lines (M : ℝ × ℝ) : 
  (∃ m, (circle (fst M) (m * (fst M - 3) + 1)) ∧ (m = 3/4 ∨ m = 0) ∧ (x = 3 ∨ 3*x - 4*m*x + 1 - 3 * m = 0 )) :=
sorry  -- Tangent lines are x = 3 and 3x - 4y - 5 = 0

theorem line_intersects_circle (a : ℝ) : ∀ x y : ℝ, line a x y → circle x y → 
  ((0 - 1)^2 + (3 - 2)^2 < 4) → True :=
sorry  -- Line ax - y + 3 = 0 intersects the circle

end tangent_lines_line_intersects_circle_l334_334731


namespace find_v2_using_horner_method_l334_334230

noncomputable def horner (a b : ℕ) : ℕ := a * 5 + b

-- Definitions from conditions
def poly_coeffs : List ℕ := [2, -5, -4, 3, -6, 7]
def x := 5 

-- Using Horner's method
def v0 := poly_coeffs.head
def v1 := horner v0 poly_coeffs[1]
def v2 := horner v1 poly_coeffs[2]

theorem find_v2_using_horner_method : v2 = 21 :=
by
  simp [v0, v1, v2, horner]
  sorry

end find_v2_using_horner_method_l334_334230


namespace quadratic_unique_real_root_l334_334258

theorem quadratic_unique_real_root (m : ℝ) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ∃! r : ℝ, x = r) → m = 2/9 :=
by
  sorry

end quadratic_unique_real_root_l334_334258


namespace beverage_distribution_l334_334645

theorem beverage_distribution (total_cans : ℕ) (number_of_children : ℕ) (hcans : total_cans = 5) (hchildren : number_of_children = 8) :
  (total_cans / number_of_children : ℚ) = 5 / 8 :=
by
  -- Given the conditions
  have htotal_cans : total_cans = 5 := hcans
  have hnumber_of_children : number_of_children = 8 := hchildren
  
  -- we need to show the beverage distribution
  rw [htotal_cans, hnumber_of_children]
  exact by norm_num

end beverage_distribution_l334_334645


namespace reflection_square_identity_l334_334833

def vector := ℝ × ℝ

def reflection_matrix (v : vector) : matrix (fin 2) (fin 2) ℝ :=
  let ⟨x, y⟩ := v in
  let norm_sq := x^2 + y^2 in
  ![
    [2 * x^2 / norm_sq - 1, 2 * x * y / norm_sq],
    [2 * x * y / norm_sq, 2 * y^2 / norm_sq - 1]
  ]

theorem reflection_square_identity (v : vector) (Q := reflection_matrix v) : 
  v = (4, -2) → Q * Q = 1 :=
by 
  sorry

end reflection_square_identity_l334_334833


namespace slope_of_line_l334_334401

theorem slope_of_line (A B C : ℝ) (h : A = 1 ∧ B = 3 ∧ C = 3) : 
  (- A / B) = -1 / 3 :=
by
  have hA : A = 1 := h.left,
  have hB : B = 3 := h.right.left,
  rw [hA, hB],
  norm_num,
  sorry

end slope_of_line_l334_334401


namespace find_x_l334_334407

theorem find_x (x : ℚ) (h : (3 - x) / (2 - x) - 1 / (x - 2) = 3) : x = 1 := 
  sorry

end find_x_l334_334407


namespace coeff_a10_of_poly_l334_334229

theorem coeff_a10_of_poly:
  ∀ (x : ℝ), ∃ (a : ℕ → ℝ), 
    (x^2 + x^11 = ∑ i in range 12, (a i) * (x + 1)^i) → 
    a 10 = -11 :=
by
  sorry

end coeff_a10_of_poly_l334_334229


namespace least_five_digit_perfect_square_and_cube_l334_334427

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334427


namespace value_of_power_of_a_l334_334351

variable (a : ℝ)

theorem value_of_power_of_a (h : 6 = a + a⁻¹) : a^4 + a⁻⁴ = 1154 :=
sorry

end value_of_power_of_a_l334_334351


namespace option_A_option_B_option_C_option_D_l334_334809

noncomputable def triangle (a b c : ℝ) (A B C : ℝ) := 
  A + B + C = π ∧ a = b ∧ b = c

theorem option_A (A B : ℝ): (A > B) → (sin A > sin B) :=
by
  sorry

theorem option_B (A B : ℝ) : (sin (2 * A) = sin (2 * B)) → ¬(A = B) :=
by
  sorry

theorem option_C (a b c A B : ℝ): (a * cos B) - (b * cos A) = c → A = π / 2 :=
by
  sorry

theorem option_D (a b c : ℝ) : (a / b = 3 / 5) ∧ (b / c = 5 / 7) → 
  let C := angle_opposite_longest_side a b c in C > π / 2 :=
by
  sorry

end option_A_option_B_option_C_option_D_l334_334809


namespace cone_radius_height_ratio_l334_334595

theorem cone_radius_height_ratio 
  (V : ℝ) (π : ℝ) (r h : ℝ)
  (circumference : ℝ) 
  (original_height : ℝ)
  (new_volume : ℝ)
  (volume_formula : V = (1/3) * π * r^2 * h)
  (radius_from_circumference : 2 * π * r = circumference)
  (base_circumference : circumference = 28 * π)
  (original_height_eq : original_height = 45)
  (new_volume_eq : new_volume = 441 * π) :
  (r / h) = 14 / 9 :=
by
  sorry

end cone_radius_height_ratio_l334_334595


namespace ratio_of_selling_to_buying_l334_334872

noncomputable def natasha_has_3_times_carla (N C : ℕ) : Prop :=
  N = 3 * C

noncomputable def carla_has_2_times_cosima (C S : ℕ) : Prop :=
  C = 2 * S

noncomputable def total_buying_price (N C S : ℕ) : ℕ :=
  N + C + S

noncomputable def total_selling_price (buying_price profit : ℕ) : ℕ :=
  buying_price + profit

theorem ratio_of_selling_to_buying (N C S buying_price selling_price ratio : ℕ) 
  (h1 : natasha_has_3_times_carla N C)
  (h2 : carla_has_2_times_cosima C S)
  (h3 : N = 60)
  (h4 : buying_price = total_buying_price N C S)
  (h5 : total_selling_price buying_price 36 = selling_price)
  (h6 : 18 * ratio = selling_price * 5): ratio = 7 :=
by
  sorry

end ratio_of_selling_to_buying_l334_334872


namespace min_value_288_l334_334307

noncomputable def minValueForProduct (a b c : ℝ) (h : a * b * c = 8) : ℝ := (2 * a + 3 * b) * (2 * b + 3 * c) * (2 * c + 3 * a)

theorem min_value_288 (a b c : ℝ) (h : a * b * c = 8) : (forall x > 0, (forall y > 0, (forall z > 0, minValueForProduct x y z h >= 288))) :=
sorry

end min_value_288_l334_334307


namespace quadratic_equation_has_real_root_l334_334337

theorem quadratic_equation_has_real_root
  (a c m n : ℝ) :
  ∃ x : ℝ, c * x^2 + m * x - a = 0 ∨ ∃ y : ℝ, a * y^2 + n * y + c = 0 :=
by
  -- Proof omitted
  sorry

end quadratic_equation_has_real_root_l334_334337


namespace polynomial_zero_of_multiplicity_two_l334_334033

theorem polynomial_zero_of_multiplicity_two (P : Polynomial ℤ)
  (h_degree : P.degree = 4)
  (h_leading_coeff : P.leadingCoeff = 1)
  (h_integer_coeffs : ∀ n, P.coeff n ∈ ℤ)
  (r : ℤ)
  (h_zero : (P.derivative.eval r) = 0 ∧ (P.eval r) = 0 ∧ (P.eval (r + 1)) ≠ 0) :
  ((∃ α β : ℤ, P = (X - C r)^2 * (X^2 + C α * X + C β) ∧ 
    α^2 - 4 * β = -2) → 
    P.eval (1 + I * real.sqrt 2) = 0) :=
by 
  sorry

end polynomial_zero_of_multiplicity_two_l334_334033


namespace probability_midpoint_in_T_l334_334826

def point_in_T (x y z : ℕ) : Prop := 
  0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 0 ≤ z ∧ z ≤ 5

def midpoint_in_T (p1 p2 : ℕ × ℕ × ℕ) : Prop :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  let mz := (z1 + z2) / 2
  point_in_T mx my mz

theorem probability_midpoint_in_T :
  ∃ (p q : ℕ), p + q = 454 ∧ (num_valid_combinations * q = 97 * total_combinations)
  where
    num_valid_combinations : ℕ := 1940
    total_combinations : ℕ := 7140 := sorry

end probability_midpoint_in_T_l334_334826


namespace original_triangle_area_l334_334327

theorem original_triangle_area (A B C : Point) (triangle_ABC : EquilateralTriangle A B C)
  (fold1 : Fold A C triangle_ABC)
  (fold2 : Fold B C fold1.resulting_shape)
  (final_shape_area : ShapeArea fold2.resulting_shape = 12) :
  TriangleArea triangle_ABC = 36 := 
sorry

end original_triangle_area_l334_334327


namespace willy_crayons_eq_l334_334967

def lucy_crayons : ℕ := 3971
def more_crayons : ℕ := 1121

theorem willy_crayons_eq : 
  ∀ willy_crayons : ℕ, willy_crayons = lucy_crayons + more_crayons → willy_crayons = 5092 :=
by
  sorry

end willy_crayons_eq_l334_334967


namespace degrees_at_7_20_l334_334619

noncomputable def degrees_of_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  let minute_angle := (minute / 60.0) * 360
  let hour_angle := (hour % 12 + minute / 60.0) * 30
  let angle_diff := Math.abs (hour_angle - minute_angle)
  min angle_diff (360 - angle_diff)

theorem degrees_at_7_20 : degrees_of_angle 7 20 = 100 :=
sorry

end degrees_at_7_20_l334_334619


namespace chess_tournament_points_distribution_l334_334176

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l334_334176


namespace inequality_proof_l334_334209

theorem inequality_proof (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a^2 - b^2) / (a^2 + b^2) > (a - b) / (a + b) :=
by 
  sorry

end inequality_proof_l334_334209


namespace triangle_sides_l334_334957
-- Import the entire library mainly used for geometry and algebraic proofs.

-- Define the main problem statement as a theorem.
theorem triangle_sides (a b c : ℕ) (r_incircle : ℕ)
  (r_excircle_a r_excircle_b r_excircle_c : ℕ) (s : ℕ)
  (area : ℕ) : 
  r_incircle = 1 → 
  area = s →
  r_excircle_a * r_excircle_b * r_excircle_c = (s * s * s) →
  s = (a + b + c) / 2 →
  r_excircle_a = s / (s - a) →
  r_excircle_b = s / (s - b) →
  r_excircle_c = s / (s - c) →
  a * b = 12 → 
  a = 3 ∧ b = 4 ∧ c = 5 :=
by {
  -- Placeholder for the proof.
  sorry
}

end triangle_sides_l334_334957


namespace scores_are_correct_l334_334139

variable
  Andrey_score : ℝ
  Dima_score : ℝ
  Vanya_score : ℝ
  Sasha_score : ℝ

-- Conditions
axiom andrey_first : Andrey_score > Dima_score ∧ Andrey_score > Vanya_score ∧ Andrey_score > Sasha_score
axiom dima_second : Dima_score > Vanya_score ∧ Dima_score > Sasha_score
axiom vanya_third : Vanya_score > Sasha_score
axiom unique_scores : Andrey_score ≠ Dima_score ∧ Andrey_score ≠ Vanya_score ∧ Andrey_score ≠ Sasha_score ∧ Dima_score ≠ Vanya_score ∧ Dima_score ≠ Sasha_score ∧ Vanya_score ≠ Sasha_score
axiom total_points : Andrey_score + Dima_score + Vanya_score + Sasha_score = 12
axiom andrey_sasha_wins : Andrey_score = 4 ∧ Sasha_score = 2

-- Conclusion
theorem scores_are_correct :
  Andrey_score = 4 ∧ Dima_score = 3.5 ∧ Vanya_score = 2.5 ∧ Sasha_score = 2 :=
  sorry

end scores_are_correct_l334_334139


namespace unique_intersection_point_l334_334376

theorem unique_intersection_point (k : ℝ) :
x = k ->
∃ x : ℝ, x = -3*y^2 - 4*y + 7 -> ∃ k : ℝ, k = 25/3 -> y = 0 -> x = k

end unique_intersection_point_l334_334376


namespace prove_AF_eq_l334_334266

-- Definitions
variables {A B C E F : Type*}
variables [Field A] [Field B] [Field C] [Field E] [Field F]

-- Conditions
def triangle_ABC (AB AC : ℝ) (h : AB > AC) : Prop := true

def external_bisector (angleA : ℝ) (circumcircle_meets : ℝ) : Prop := true

def foot_perpendicular (E AB : ℝ) : Prop := true

-- Theorem statement
theorem prove_AF_eq (AB AC AF : ℝ) (h_triangle : triangle_ABC AB AC (by sorry))
  (h_external_bisector : external_bisector (by sorry) (by sorry))
  (h_foot_perpendicular : foot_perpendicular (by sorry) AB) :
  2 * AF = AB - AC := by
  sorry

end prove_AF_eq_l334_334266


namespace extreme_value_at_one_symmetric_points_range_l334_334734

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then
  x^2 + 3 * a * x
else
  2 * Real.exp x - x^2 + 2 * a * x

theorem extreme_value_at_one (a : ℝ) :
  (∀ x > 0, f x a = 2 * Real.exp x - x^2 + 2 * a * x) →
  (∀ x < 0, f x a = x^2 + 3 * a * x) →
  (∀ x > 0, deriv (fun x => f x a) x = 2 * Real.exp x - 2 * x + 2 * a) →
  deriv (fun x => f x a) 1 = 0 →
  a = 1 - Real.exp 1 :=
  sorry

theorem symmetric_points_range (a : ℝ) :
  (∃ x0 > 0, (∃ y0 : ℝ, 
  (f x0 a = y0 ∧ f (-x0) a = -y0))) →
  a ≥ 2 * Real.exp 1 :=
  sorry

end extreme_value_at_one_symmetric_points_range_l334_334734


namespace construction_blocks_needed_l334_334315

/- Define the dimensions of the storage -/
def length : ℕ := 20
def width : ℕ := 15
def height : ℕ := 10

/- Define the thickness of the walls and floor -/
def thickness : ℕ := 2

/- Calculate the total volume of the storage -/
def total_volume (l w h : ℕ) : ℕ := l * w * h
def V_total := total_volume length width height

/- Calculate the interior dimensions -/
def interior_length := length - 2 * thickness
def interior_width := width - 2 * thickness
def interior_height := height - thickness

/- Calculate the volume of the interior space -/
def interior_volume (l w h : ℕ) : ℕ := l * w * h
def V_interior := interior_volume interior_length interior_width interior_height

/- Calculate the number of blocks needed -/
def number_of_blocks_needed : ℕ := V_total - V_interior

/- Prove that the number of blocks required is 1592 -/
theorem construction_blocks_needed : number_of_blocks_needed = 1592 :=
by
  sorry

end construction_blocks_needed_l334_334315


namespace simplify_expression_l334_334891

theorem simplify_expression :
  (3 + 4 + 5 + 7) / 3 + (3 * 6 + 9) / 4 = 157 / 12 :=
by
  sorry

end simplify_expression_l334_334891


namespace line_intersects_x_axis_at_l334_334583

theorem line_intersects_x_axis_at (a b : ℝ) (h1 : a = 12) (h2 : b = 2)
  (c d : ℝ) (h3 : c = 6) (h4 : d = 6) : 
  ∃ x : ℝ, (x, 0) = (15, 0) := 
by
  -- proof needed here
  sorry

end line_intersects_x_axis_at_l334_334583


namespace count_dot_not_line_l334_334262

variable (T DL L_D D_L : ℕ)

-- Given conditions
def condition_total_letters : T = 56 := by sorry
def condition_both : DL = 18 := by sorry
def condition_line_not_dot : L_D = 30 := by sorry
def condition_all_letters : T = D_L + L_D + DL := by sorry

-- Prove the number of letters that contain a dot but not a straight line is 8
theorem count_dot_not_line (h1 : condition_total_letters T) 
                            (h2 : condition_both DL) 
                            (h3 : condition_line_not_dot L_D) 
                            (h4 : condition_all_letters T DL L_D D_L) :
  D_L = 8 :=
by
  -- Proof would go here
  sorry

end count_dot_not_line_l334_334262


namespace sum_reciprocals_eq_three_l334_334943

-- Define nonzero real numbers x and y with their given condition
variables (x y : ℝ) (hx : x ≠ 0) (hy: y ≠ 0) (h : x + y = 3 * x * y)

-- State the theorem to prove the sum of reciprocals of x and y is 3
theorem sum_reciprocals_eq_three (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) : (1 / x) + (1 / y) = 3 :=
sorry

end sum_reciprocals_eq_three_l334_334943


namespace count_of_b_l334_334673

lemma number_of_valid_b (b : ℕ) (hb : b ≤ 100) : 
  (∃ (x : ℤ), x^2 + (b + 3) * x + ((b + 3) / 2)^2 = 0) ↔ b % 2 = 1 :=
sorry

theorem count_of_b : 
  (finset.card (finset.filter (λ b, ∃ (x : ℤ), x^2 + (b + 3) * x + ((b + 3) / 2)^2 = 0) (finset.range 101))) = 50 :=
begin
  sorry
end

end count_of_b_l334_334673


namespace largest_gcd_sum_eq_1023_l334_334944

theorem largest_gcd_sum_eq_1023 (c d : ℕ) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_sum : c + d = 1023) : 
  ∃ k : ℕ, k = 341 ∧ k = Nat.gcd c d :=
begin
  sorry
end

end largest_gcd_sum_eq_1023_l334_334944


namespace height_to_hypotenuse_l334_334597

theorem height_to_hypotenuse (a b c : ℝ) (h : a^2 + b^2 = c^2) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c = 10) :
  ∃ x : ℝ, 10 * x / 2 = a * b / 2 ∧ x = 4.8 :=
by
  -- Begin the proof
  use 4.8
  split
  calc
    10 * 4.8 / 2 = a * b / 2 : sorry
  exact sorry

end height_to_hypotenuse_l334_334597


namespace smallest_circle_covering_region_l334_334228

/-- 
Given the conditions describing the plane region:
1. x ≥ 0
2. y ≥ 0
3. x + 2y - 4 ≤ 0

Prove that the equation of the smallest circle covering this region is (x - 2)² + (y - 1)² = 5.
-/
theorem smallest_circle_covering_region :
  (∀ (x y : ℝ), (x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y - 4 ≤ 0) → (x - 2)^2 + (y - 1)^2 ≤ 5) :=
sorry

end smallest_circle_covering_region_l334_334228


namespace fraction_of_girls_correct_l334_334060

-- Define the total number of students in each school
def total_greenwood : ℕ := 300
def total_maplewood : ℕ := 240

-- Define the ratios of boys to girls
def ratio_boys_girls_greenwood := (3, 2)
def ratio_boys_girls_maplewood := (3, 4)

-- Define the number of boys and girls at Greenwood Middle School
def boys_greenwood (x : ℕ) : ℕ := 3 * x
def girls_greenwood (x : ℕ) : ℕ := 2 * x

-- Define the number of boys and girls at Maplewood Middle School
def boys_maplewood (y : ℕ) : ℕ := 3 * y
def girls_maplewood (y : ℕ) : ℕ := 4 * y

-- Define the total fractions
def total_girls (x y : ℕ) : ℚ := (girls_greenwood x + girls_maplewood y)
def total_students : ℚ := (total_greenwood + total_maplewood)

-- Main theorem to prove the fraction of girls at the event
theorem fraction_of_girls_correct (x y : ℕ)
  (h1 : 5 * x = total_greenwood)
  (h2 : 7 * y = total_maplewood) :
  (total_girls x y) / total_students = 5 / 7 :=
by
  sorry

end fraction_of_girls_correct_l334_334060


namespace find_a_b_l334_334753

def A (x : ℝ) : Prop := 12 - 5 * x - 2 * x^2 > 0
def B (x : ℝ) (a b : ℝ) : Prop := x^2 - a * x + b ≤ 0

theorem find_a_b : (∀ x, A x → ¬B x (19/2) 12) ∧ 
                   (∀ x, x ∈ set.Ioc (-4 : ℝ) (3/2) → A x) ∧ 
                   (∀ x, x ∈ set.Icc (3/2) 8 → B x (19/2) 12) ∧ 
                   (∀ x, (A x ∨ B x (19/2) 12) ↔ x ∈ set.Ioc (-4) 8) := 
sorry

end find_a_b_l334_334753


namespace chess_tournament_scores_l334_334157

def points (name : String) := Real

def total_points : Real := 12

variables (A D V S : Real)
variable (total_games : ℕ := 12)

axiom different_scores : A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S

axiom ranking : A > D ∧ D > V ∧ V > S

axiom equal_wins (A S : Real) : (A = 2 * win_points) ∧ (S = 2 * win_points)

axiom total_points_constraint : A + D + V + S = total_points

theorem chess_tournament_scores :
  A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 :=
by 
  sorry

end chess_tournament_scores_l334_334157


namespace chess_tournament_points_distribution_l334_334183

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l334_334183


namespace least_five_digit_is_15625_l334_334444

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334444


namespace standard_curve_equation_pa_times_pb_range_l334_334227

noncomputable def curve_c(parametric_eq : ℝ → ℝ × ℝ) (x : ℝ) (y : ℝ) : Prop :=
parametric_eq ∈ { (a, b) | a = x ∧ b = y }

theorem standard_curve_equation
  (parametric_eq : ℝ → ℝ × ℝ)
  (α : ℝ)
  (hx : parametric_eq α = (1, (√2)/2))
  : ∀ x y, (∃ a b, parametric_eq = λ α, (a * Real.cos α, b * Real.sin α) ∧ a = √2 ∧ b = 1) → (x = √2 * Real.cos α ∧ y = Real.sin α → x^2 / 2 + y^2 = 1) := 
by
  intros _ _ ⟨a, b, hparam_eq, ha, hb⟩ _ _
  sorry

theorem pa_times_pb_range
  (parametric_eq : ℝ → ℝ × ℝ)
  (θ : ℝ)
  (hx : parametric_eq θ = (0, √2)): 
  ∃ t_1 t_2, (1 + Real.sin θ ^ 2) * t_1 ^ 2 + 4 * √2 * Real.sin θ * t_1 + 2 = 0 ∧ (t_1 * t_2 = -2 / (1 + Real.sin θ ^ 2)) → 
  (1 ≤ -t_1 * t_2 ∧ -t_1 * t_2 ≤ 2) := 
by
  intros 
  sorry

end standard_curve_equation_pa_times_pb_range_l334_334227


namespace game_ends_after_27_rounds_l334_334781

theorem game_ends_after_27_rounds :
  ∃ n : ℕ,
    let X : ℕ := 16,
    let Y : ℕ := 15,
    let Z : ℕ := 14,
    let Rounds : ℕ := 27,
    ∀ i < Rounds, -- iterate for 27 rounds
      (i % 3 = 0 → X - 6*(i/3) + 2*((i+1)/3) + 2*((i+2)/3) = 0 ∨ -- X gives tokens
       i % 3 = 1 → Y - 6*(i/3) + 2*((i+1)/3) + 2*((i+2)/3) = 0 ∨ -- Y gives tokens
       i % 3 = 2 → Z - 6*(i/3) + 2*((i+1)/3) + 2*((i+2)/3) = 0) -- Z gives tokens
      → n = Rounds := 
by 
  intros,
  sorry

end game_ends_after_27_rounds_l334_334781


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334500

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334500


namespace gcd_99_36_l334_334956

def successive_subtraction_gcd : ℕ → ℕ → ℕ
| a, 0 => a
| a, b => if a ≥ b then successive_subtraction_gcd (a - b) b else successive_subtraction_gcd b (a - b)

theorem gcd_99_36 : successive_subtraction_gcd 99 36 = 9 :=
by
sorry

end gcd_99_36_l334_334956


namespace competition_scores_l334_334021

theorem competition_scores (n d : ℕ) (h_n : 1 < n)
  (h_total_score : d * (n * (n + 1)) / 2 = 26 * n) :
  (n, d) = (3, 13) ∨ (n, d) = (12, 4) ∨ (n, d) = (25, 2) :=
by
  sorry

end competition_scores_l334_334021


namespace least_five_digit_perfect_square_and_cube_l334_334434

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334434


namespace simplify_polynomial_l334_334340

theorem simplify_polynomial : 
  ∀ (x : ℝ), 
    (2 * x + 1) ^ 5 - 5 * (2 * x + 1) ^ 4 + 10 * (2 * x + 1) ^ 3 - 10 * (2 * x + 1) ^ 2 + 5 * (2 * x + 1) - 1 
    = 32 * x ^ 5 := 
by sorry

end simplify_polynomial_l334_334340


namespace proof_problem_1_proof_problem_2_l334_334000

-- Define the given conditions
variables {S1 S2 : Circle} 
variables {A B : Point}
variable (l : Line)
variables {P Q R : Point}
variables {ABC : Triangle}
variables {a : ℝ}

-- Define circles intersecting at points A and B
axiom circles_intersect (hS1 : Circle S1) (hS2 : Circle S2) (hA : PointInCircle A S1) (hA' : PointInCircle A S2) (hB : PointInCircle B S1) (hB' : PointInCircle B S2) : Prop

-- Define a line l through point A such that the segment within S₁ and S₂ has specified length a
axiom line_through_A_segment_length (hS1 : Circle S1) (hS2 : Circle S2) (hA : PointInCircle A S1) (hA' : PointInCircle A S2) : LineContainsPoint l A → ∃ (P Q : Point), (P ∈ S1) ∧ (Q ∈ S2) ∧ length_segment P Q = a

-- Define the triangle congruence condition
axiom inscribe_triangle_congruent (hABC : Triangle ABC) (hPQR : Triangle PQR) : ∃ (A' B' C' : Point), Triangle A' B' C' ∧ congruent_triangles ABC PQR

-- Define the two main statements to prove
theorem proof_problem_1 (hS1 : Circle S1) (hS2 : Circle S2) (hA : PointInCircle A S1) (hA' : PointInCircle A S2) : ∃ (l : Line), LineContainsPoint l A ∧ segment_length_within_circles l S1 S2 = a :=
by
  sorry

theorem proof_problem_2 (hABC : Triangle ABC) (hPQR : Triangle PQR) : ∃ (inscribed : Triangle), congruent_triangles inscribed PQR :=
by
  sorry

end proof_problem_1_proof_problem_2_l334_334000


namespace percentage_decrease_after_raise_l334_334396

theorem percentage_decrease_after_raise
  (original_salary : ℝ) (final_salary : ℝ) (initial_raise_percent : ℝ)
  (initial_salary_raised : original_salary * (1 + initial_raise_percent / 100) = 5500): 
  original_salary = 5000 -> final_salary = 5225 -> initial_raise_percent = 10 ->
  ∃ (percentage_decrease : ℝ),
    final_salary = original_salary * (1 + initial_raise_percent / 100) * (1 - percentage_decrease / 100)
    ∧ percentage_decrease = 5 := by
  intros h1 h2 h3
  use 5
  rw [h1, h2, h3]
  simp
  sorry

end percentage_decrease_after_raise_l334_334396


namespace pages_per_chapter_l334_334678

-- Definitions based on conditions
def chapters_in_book : ℕ := 2
def days_to_finish : ℕ := 664
def chapters_per_day : ℕ := 332
def total_chapters_read : ℕ := chapters_per_day * days_to_finish

-- Theorem that states the problem
theorem pages_per_chapter : total_chapters_read / chapters_in_book = 110224 :=
by
  -- Proof is omitted
  sorry

end pages_per_chapter_l334_334678


namespace min_range_of_three_test_takers_l334_334004

-- Proposition: The minimum possible range in scores of the 3 test-takers
-- where the ranges of their scores in the 5 practice tests are 18, 26, and 32, is 76.
theorem min_range_of_three_test_takers (r1 r2 r3: ℕ) 
  (h1 : r1 = 18) (h2 : r2 = 26) (h3 : r3 = 32) : 
  (r1 + r2 + r3) = 76 := by
  sorry

end min_range_of_three_test_takers_l334_334004


namespace Cherie_boxes_l334_334296

theorem Cherie_boxes (x : ℕ) :
  (2 * 8 + x * (8 + 9) = 33) → x = 1 :=
by
  intros h
  have h_eq : 16 + 17 * x = 33 := by simp [mul_add, mul_comm, h]
  linarith

end Cherie_boxes_l334_334296


namespace concyclic_points_l334_334312

-- Definitions for cyclic quadrilateral, intersections, and reflection
variable {P : Type} [EuclideanGeometry P]

def cyclic_quadrilateral (A B C D : P) : Prop := 
  ∃ (O : P), inscribed_in_circle O A B C D

def meet (X Y Z W : P) (E : P) : Prop := 
  collinear X Z E ∧ collinear Y W E ∧ ¬ collinear X Y E

def parallelogram (E C G D : P) : Prop :=
  midpoint E C = midpoint G D

def reflection (E A D : P) : P :=
  point_reflection A D E

-- Main theorem statement for Lean
theorem concyclic_points
  (A B C D E F G H : P) 
  (h_cyclic : cyclic_quadrilateral A B C D)
  (h_meet : meet A C B D E)
  (h_ext : meet A D B C F)
  (h_parallelogram : parallelogram E C G D)
  (h_reflection : H = reflection E A D) :
  cyclic_quadrilateral D H F G := 
sorry

end concyclic_points_l334_334312


namespace arithmetic_sequence_a5_eq_6_l334_334207

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_a5_eq_6 (h_arith : is_arithmetic_sequence a_n) (h_sum : a_n 2 + a_n 8 = 12) : a_n 5 = 6 :=
by
  sorry

end arithmetic_sequence_a5_eq_6_l334_334207


namespace circumcircle_radius_is_one_l334_334051

noncomputable def circumcircle_radius_of_triangle_ABC
  (P A B C : Point)
  (r : ℝ)
  (circle1 : Circle := mkCircle P r)
  (circle2 : Circle := mkCircle P r)
  (circle3 : Circle := mkCircle P r)
  (h1 : circles_intersect_at_two_points circle1 circle2 P A)
  (h2 : circles_intersect_at_two_points circle2 circle3 P B)
  (h3 : circles_intersect_at_two_points circle3 circle1 P C) : ℝ :=
begin
  sorry
end

theorem circumcircle_radius_is_one
  (P A B C : Point)
  (r : ℝ := 3)
  (circle1 : Circle := mkCircle P r)
  (circle2 : Circle := mkCircle P r)
  (circle3 : Circle := mkCircle P r)
  (h1 : circles_intersect_at_two_points circle1 circle2 P A)
  (h2 : circles_intersect_at_two_points circle2 circle3 P B)
  (h3 : circles_intersect_at_two_points circle3 circle1 P C) :
  circumcircle_radius_of_triangle_ABC P A B C r circle1 circle2 circle3 h1 h2 h3 = 1 :=
begin
  sorry
end

end circumcircle_radius_is_one_l334_334051


namespace area_ratio_of_rectangle_and_triangle_l334_334963

theorem area_ratio_of_rectangle_and_triangle
  (L W : ℝ)
  (hL : 0 < L)
  (hW : 0 < W) :
  let A_rectangle := L * W,
      A_triangle := (1 / 2) * L * W in
  A_rectangle / A_triangle = 2 :=
by
  sorry

end area_ratio_of_rectangle_and_triangle_l334_334963


namespace find_k_intersects_parabola_at_one_point_l334_334375

theorem find_k_intersects_parabola_at_one_point :
  ∃ k : ℝ, (∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k ↔ y = (-4 / (2 * 3))) →
    k = 25 / 3 :=
by sorry

end find_k_intersects_parabola_at_one_point_l334_334375


namespace team_hierarchy_exists_l334_334646

open Function

def number_of_teams : ℕ := 8

def plays_match (A B : ℕ) : Bool := A ≠ B

def wins (A B : ℕ) : Prop := plays_match A B

theorem team_hierarchy_exists :
  ∃ (A B C D : ℕ), 
    wins A B ∧ wins A C ∧ wins A D ∧
    wins B C ∧ wins B D ∧
    wins C D 
:= 
begin
  sorry
end

end team_hierarchy_exists_l334_334646


namespace algebraic_expression_value_l334_334187

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -4) :
  (2 * y - x) ^ 2 - 2 * x + 4 * y - 1 = 23 :=
by
  sorry

end algebraic_expression_value_l334_334187


namespace mean_cat_weights_l334_334941

-- Define a list representing the weights of the cats from the stem-and-leaf plot
def cat_weights : List ℕ := [12, 13, 14, 20, 21, 21, 25, 25, 28, 30, 31, 32, 32, 36, 38, 39, 39]

-- Function to calculate the sum of elements in a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Function to calculate the mean of a list of natural numbers
def mean_list (l : List ℕ) : ℚ := (sum_list l : ℚ) / l.length

-- The theorem we need to prove
theorem mean_cat_weights : mean_list cat_weights = 27 := by 
  sorry

end mean_cat_weights_l334_334941


namespace total_heart_cookies_l334_334321

-- Mrs. Snyder made some cookies.
-- Define the number of red cookies and pink cookies.
def numRedCookies : ℕ := 36
def numPinkCookies : ℕ := 50

-- Prove the total number of heart cookies is the sum of red and pink cookies.
theorem total_heart_cookies : numRedCookies + numPinkCookies = 86 := by
  rfl
sorry

end total_heart_cookies_l334_334321


namespace avg_frac_position_l334_334058

def fracs := [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]

def avg (lst : List ℚ) : ℚ :=
  (lst.sum) / lst.length

theorem avg_frac_position :
  let fractions := [1/2, 1/3, 1/4, 1/5, 1/6, 1/7]
  let average := avg fractions
  let sorted_fracs := (average :: fractions).qsort (≤)
  sorted_fracs.indexOf average = 4 := -- index 4 corresponds to the 5th position (0-based index)
by
  sorry

end avg_frac_position_l334_334058


namespace competition_participants_l334_334392

theorem competition_participants (n : ℕ) :
    (100 < n ∧ n < 200) ∧
    (n % 4 = 2) ∧
    (n % 5 = 2) ∧
    (n % 6 = 2)
    → (n = 122 ∨ n = 182) :=
by
  intro h
  sorry

end competition_participants_l334_334392


namespace sum_of_solutions_l334_334121

-- We need to define the equation and show the sum of the solutions
def equation (x : ℝ) : Prop := (4 * x) / (x^2 - 4) = (-3 * x) / (x - 2) + 2 / (x + 2)

theorem sum_of_solutions : ∑ x in {x : ℝ | equation x}, x = -8 / 3 :=
by
  sorry

end sum_of_solutions_l334_334121


namespace distance_between_A_and_B_l334_334971

-- Definitions according to the problem's conditions
def speed_train_A : ℕ := 50
def speed_train_B : ℕ := 60
def distance_difference : ℕ := 100

-- The main theorem statement to prove
theorem distance_between_A_and_B
  (x : ℕ) -- x is the distance traveled by the first train
  (distance_train_A := x)
  (distance_train_B := x + distance_difference)
  (total_distance := distance_train_A + distance_train_B)
  (meet_condition : distance_train_A / speed_train_A = distance_train_B / speed_train_B) :
  total_distance = 1100 := 
sorry

end distance_between_A_and_B_l334_334971


namespace first_quadrant_sin_cos_inequality_l334_334758

def is_first_quadrant_angle (α : ℝ) : Prop :=
  0 < Real.sin α ∧ 0 < Real.cos α

theorem first_quadrant_sin_cos_inequality (α : ℝ) :
  (is_first_quadrant_angle α ↔ Real.sin α + Real.cos α > 1) :=
by
  sorry

end first_quadrant_sin_cos_inequality_l334_334758


namespace sequence_a_eq_b_l334_334938

-- Definition of sequence aₙ 
def sequence_a (p : ℕ) (hp : Nat.Prime p) (n : ℕ) : ℕ :=
  if n < p - 1 then n else find (λ m, ∀ (k : Fin p) (d : ℕ), ∃ (r : Fin (n + 1)), n! - m! = p * k + d)

-- Definition of sequence bₙ 
def sequence_b (p : ℕ) (hp : Nat.Prime p)  (n : ℕ) : ℕ := 
  let n_in_base_p_minus_1 := (Nat.digits (p - 1) n)
  Nat.ofDigits p n_in_base_p_minus_1

-- The theorem to prove
theorem sequence_a_eq_b (p : ℕ) (hp : Nat.Prime p) (n : ℕ) (odd_p : p % 2 = 1) : sequence_a p hp n = sequence_b p hp n :=
  sorry

end sequence_a_eq_b_l334_334938


namespace chess_tournament_scores_l334_334156

def points (name : String) := Real

def total_points : Real := 12

variables (A D V S : Real)
variable (total_games : ℕ := 12)

axiom different_scores : A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S

axiom ranking : A > D ∧ D > V ∧ V > S

axiom equal_wins (A S : Real) : (A = 2 * win_points) ∧ (S = 2 * win_points)

axiom total_points_constraint : A + D + V + S = total_points

theorem chess_tournament_scores :
  A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 :=
by 
  sorry

end chess_tournament_scores_l334_334156


namespace chess_tournament_scores_l334_334153

def points (name : String) := Real

def total_points : Real := 12

variables (A D V S : Real)
variable (total_games : ℕ := 12)

axiom different_scores : A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S

axiom ranking : A > D ∧ D > V ∧ V > S

axiom equal_wins (A S : Real) : (A = 2 * win_points) ∧ (S = 2 * win_points)

axiom total_points_constraint : A + D + V + S = total_points

theorem chess_tournament_scores :
  A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 :=
by 
  sorry

end chess_tournament_scores_l334_334153


namespace cream_ratio_max_maya_l334_334869

def initial_volume_coffee : ℝ := 16
def max_drank_coffee : ℝ := 3
def cream_added : ℝ := 4
def maya_drank_volume : ℝ := 4

def final_cream_ratio (max_cream maya_cream : ℝ) : ℝ :=
  max_cream / maya_cream

theorem cream_ratio_max_maya :
  let max_final_cream := cream_added in
  let max_final_volume := initial_volume_coffee - max_drank_coffee + cream_added in
  let maya_initial_volume := initial_volume_coffee + cream_added in
  let maya_cream_taken := (cream_added / maya_initial_volume) * maya_drank_volume in
  let maya_final_cream := cream_added - maya_cream_taken in
  final_cream_ratio max_final_cream maya_final_cream = 5 / 4 :=
by
  sorry

end cream_ratio_max_maya_l334_334869


namespace bowls_apple_pear_l334_334948

theorem bowls_apple_pear (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) :
  (∃ f : Fin (a + b) → Fin (a + b) → ℕ, -- Represent legal moves as function
    (∀ i j, (i - j) % 2 = 0 → -- Condition on index difference
      (if i < a then f i (i+1) else 0) + (if j >= a then f j (j-1) else 0) ≥ 1) -- Move apples and pears
    ∧ (∀ i, i < a → f i (i+1) > 0) -- Ensure apples and pears must be moved 
    ∧ (∀ j, j >= a → f j (j-1) > 0) -- Ensure all bowls are considered
   ) ↔ (a * b).even := 
by
  sorry

end bowls_apple_pear_l334_334948


namespace probability_reaches_edge_in_three_hops_l334_334317

def Position := (Nat, Nat)
def GridSize := 4

def isEdgeCell (p : Position) : Prop :=
  p.1 = 0 ∨ p.1 = GridSize - 1 ∨ p.2 = 0 ∨ p.2 = GridSize - 1

def initialPosition : Position := (2, 2)

def nextPositions (p : Position) : List Position :=
  let moves := [(0, 1), (1, 0), (0, -1), (-1, 0)]
  moves.map (λ (dx, dy) => 
    let nextPos := (p.1 + dx, p.2 + dy)
    let clamp (x : Int) : Nat := Nat.clamp 0 (GridSize - 1) x
    (clamp nextPos.1, clamp nextPos.2))

def probabilityToEdgeInNHops (n : Nat) : Prop :=
  ∀ p : Position, 
    p = initialPosition → 
    ∃ edgeP : Position, isEdgeCell edgeP ∧ reachableInNHops p edgeP n

constant reachableInNHops : Position → Position → Nat → Prop

theorem probability_reaches_edge_in_three_hops : probabilityToEdgeInNHops 3 := 
by 
  intro p pos_def
  rw [pos_def]
  use nextPositions
  sorry

end probability_reaches_edge_in_three_hops_l334_334317


namespace exists_x_with_odd_even_ν₂_l334_334123

-- Define ν₂ function
def ν₂ (k : ℤ) : ℕ :=
if hk : k = 0 then 0 else
 ⟨λ t, if 2 ^ t ∣ k then t else 0⟩

-- Prove the main theorem
theorem exists_x_with_odd_even_ν₂ (n : ℕ) (H : 2 ≤ n) (a : fin n → ℤ)
  (h_distinct : function.injective a) :
  ∃ x : ℤ, (¬ ∃ i : fin n, x = a i) ∧
    (∃ m₁ m₂ : fin n →ℕ, (∀ i, ν₂ (x - a i) = m₁ i ∨ ν₂ (x - a i) = m₂ i) ∧
    (n / 4 ≤ ∑ i in finset.filter (λ i, ν₂ (x - a i) % 2 = 1) (finset.univ : finset (fin n))) ∧
    (n / 4 ≤ ∑ i in finset.filter (λ i, ν₂ (x - a i) % 2 = 0) (finset.univ : finset (fin n)))) :=
by sorry

end exists_x_with_odd_even_ν₂_l334_334123


namespace chess_tournament_points_l334_334150

theorem chess_tournament_points
  (points : String → ℝ)
  (total_points : points "Andrey" + points "Dima" + points "Vanya" + points "Sasha" = 12)
  (distinct_points : 
    points "Andrey" ≠ points "Dima" ∧ 
    points "Andrey" ≠ points "Vanya" ∧ 
    points "Andrey" ≠ points "Sasha" ∧ 
    points "Dima" ≠ points "Vanya" ∧ 
    points "Dima" ≠ points "Sasha" ∧ 
    points "Vanya" ≠ points "Sasha")
  (order : 
    points "Andrey" > points "Dima" ∧ 
    points "Dima" > points "Vanya" ∧ 
    points "Vanya" > points "Sasha")
  (same_wins :
    let games_won (student : String) := (points student - 3) / 0.5 in
    games_won "Andrey" = games_won "Sasha") :
  points "Andrey" = 4 ∧ points "Dima" = 3.5 ∧ points "Vanya" = 2.5 ∧ points "Sasha" = 2 :=
by
  sorry

end chess_tournament_points_l334_334150


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334495

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334495


namespace second_sunday_date_l334_334789

theorem second_sunday_date {month_days : List ℕ} (h : ∀Wed_even_days, 
    (Wed_even_days = [2, 16, 30]) ∧ 
     ∃ n, 
       (forall i < Wed_even_days.length, month_days.index (23 + 7 * i) = 6) ↔
       second_sunday = 13 :=
begin
  sorry
end

variables {month_length : ℕ} (weds_on_even_days : List ℕ)

end second_sunday_date_l334_334789


namespace probability_midpoint_in_T_l334_334827

def point_in_T (x y z : ℕ) : Prop := 
  0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 0 ≤ z ∧ z ≤ 5

def midpoint_in_T (p1 p2 : ℕ × ℕ × ℕ) : Prop :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  let mz := (z1 + z2) / 2
  point_in_T mx my mz

theorem probability_midpoint_in_T :
  ∃ (p q : ℕ), p + q = 454 ∧ (num_valid_combinations * q = 97 * total_combinations)
  where
    num_valid_combinations : ℕ := 1940
    total_combinations : ℕ := 7140 := sorry

end probability_midpoint_in_T_l334_334827


namespace trigonometric_identity_l334_334977

theorem trigonometric_identity :
  4 * real.cos (10 * real.pi / 180) - real.tan (80 * real.pi / 180) = -real.sqrt 3 :=
by 
  sorry

end trigonometric_identity_l334_334977


namespace least_five_digit_perfect_square_and_cube_l334_334486

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334486


namespace domino_tilings_2_by_10_l334_334575

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

-- Define what it means to tile a 2-by-n rectangle with dominoes
def domino_tilings (n : ℕ) : ℕ := fibonacci (n + 1)

-- State that the number of tilings of a 2-by-10 rectangle is 144
theorem domino_tilings_2_by_10 : domino_tilings 10 = 144 :=
  by sorry

end domino_tilings_2_by_10_l334_334575


namespace invalid_sequence_general_term_l334_334721

theorem invalid_sequence_general_term :
  ¬ ∀ (n : ℕ), (if n = 1 then 2 else if n = 2 then 0 else if n = 3 then 2 else 0) = 2 * sin (n * Real.pi / 2) := 
sorry

end invalid_sequence_general_term_l334_334721


namespace b_general_formula_a_arithmetic_sequence_T_sum_l334_334727

noncomputable def b_sequence (n : ℕ) : ℕ := 2^(n - 1)

noncomputable def a_sequence (n : ℕ) : ℕ := n + 2

noncomputable def c_sequence (n : ℕ) : ℕ := (n + 2) * 2^(n - 1)

-- Sum of first n terms of c_sequence
noncomputable def T (n : ℕ) : ℕ := Finset.sum (Finset.range n) (λ k, c_sequence (k + 1))

theorem b_general_formula (n : ℕ) : 
  b_sequence n = 2^(n - 1) := 
by 
  sorry

theorem a_arithmetic_sequence : 
  ∀ n, a_sequence n = n + 2 := 
by
  sorry

theorem T_sum (n : ℕ) : 
  T n = (n + 1) * 2^n - 1 := 
by
  sorry

end b_general_formula_a_arithmetic_sequence_T_sum_l334_334727


namespace dropped_student_quiz_score_l334_334906

theorem dropped_student_quiz_score
(a b : ℕ) (avg_initial avg_remaining : ℚ)
    (h1 : a = 16)
    (h2 : b = 15)
    (h3 : avg_initial = 62.5)
    (h4 : avg_remaining = 62.0) :
  let total_initial := a * avg_initial,
      total_remaining := b * avg_remaining,
      dropout_score := total_initial - total_remaining in
  dropout_score = 70 :=
by
  sorry

end dropped_student_quiz_score_l334_334906


namespace parabola_vector_sum_distance_l334_334303

noncomputable def parabola_focus (x y : ℝ) : Prop := x^2 = 8 * y

noncomputable def on_parabola (x y : ℝ) : Prop := parabola_focus x y

theorem parabola_vector_sum_distance :
  ∀ (A B C : ℝ × ℝ) (F : ℝ × ℝ),
  on_parabola A.1 A.2 ∧ on_parabola B.1 B.2 ∧ on_parabola C.1 C.2 ∧
  F = (0, 2) ∧
  ((A.1 - F.1)^2 + (A.2 - F.2)^2) + ((B.1 - F.1)^2 + (B.2 - F.2)^2) + ((C.1 - F.1)^2 + (C.2 - F.2)^2) = 0
  → (abs ((A.2 + F.2)) + abs ((B.2 + F.2)) + abs ((C.2 + F.2))) = 12 :=
by sorry

end parabola_vector_sum_distance_l334_334303


namespace remainder_6_pow_23_mod_5_l334_334119

theorem remainder_6_pow_23_mod_5 : (6 ^ 23) % 5 = 1 := 
by {
  sorry
}

end remainder_6_pow_23_mod_5_l334_334119


namespace max_odd_integers_l334_334611

theorem max_odd_integers (set : Finset ℕ) (h_len : set.card = 6) (h_pos : ∀ x ∈ set, 0 < x) (h_div4 : (∏ x in set, x) % 4 = 0) : 
  ∃ n, n ≤ 5 ∧ ∃ odd_set : Finset ℕ, odd_set = set.filter (λ x, x % 2 = 1) ∧ odd_set.card = n :=
sorry

end max_odd_integers_l334_334611


namespace sum_first_five_terms_geometric_sequence_l334_334668

noncomputable def sum_first_five_geometric (a0 : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a0 * (1 - r^n) / (1 - r)

theorem sum_first_five_terms_geometric_sequence : 
  sum_first_five_geometric (1/3) (1/3) 5 = 121 / 243 := 
by 
  sorry

end sum_first_five_terms_geometric_sequence_l334_334668


namespace trigonometric_identity_l334_334807

variable {α β γ : Real} -- Variables representing angles A, B, C respectively
variable {a b c h : Real} -- Variables representing sides a, b, c and height h respectively

theorem trigonometric_identity
  (h_triangle : Triangle α β γ)
  (c_minus_a_eq_h : c - a = h) :
  Real.sin ((γ - α) / 2) + Real.cos ((γ + α) / 2) = 1 :=
by
  sorry

end trigonometric_identity_l334_334807


namespace f_decreasing_sufficient_not_necessary_for_g_increasing_l334_334715

-- Present the conditions as definitions
def a_pos (a : ℝ) : Prop := a > 0
def a_not_one (a : ℝ) : Prop := a ≠ 1

-- Define the respective properties of the functions
def f_decreasing (a : ℝ) (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y
def g_increasing (a : ℝ) (g : ℝ → ℝ) : Prop := ∀ x y, x < y → g x < g y

-- State the mathematically equivalent proof problem in Lean 4 statement
theorem f_decreasing_sufficient_not_necessary_for_g_increasing (a : ℝ) :
  a_pos a → a_not_one a →
  (f_decreasing a (fun x => a ^ x) →
   g_increasing a (fun x => (2 - a) * x ^ 3)) ∧
  ¬ (g_increasing a (fun x => (2 - a) * x ^ 3) →
     f_decreasing a (fun x => a ^ x)) :=
by 
  intros h1 h2 h3
  split
  all_goals { sorry }

end f_decreasing_sufficient_not_necessary_for_g_increasing_l334_334715


namespace triangle_BXC_right_triangle_AXD_right_l334_334418

variables {A B C D X : Type}
variables [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry X]

-- Define segments and their properties
variables (segment_AB segment_CD : LineSegment A B) (segment_CD : LineSegment C D)

-- Define the conditions
def equal_segments : Prop :=
  segment_AB.length = segment_CD.length

def perpendicular_segments : Prop :=
  segment_AB.is_perpendicular_to segment_CD

def C_on_AB : Prop :=
  C ∈ segment_AB

def X_properties_BX_XC_AX_XD : Prop :=
  distance B X = distance X C ∧ distance A X = distance X D

-- Prove triangle BXC is a right triangle
theorem triangle_BXC_right
  (H_segments_eq : equal_segments segment_AB segment_CD)
  (H_segments_perpendicular : perpendicular_segments segment_AB segment_CD)
  (H_C_on_AB : C_on_AB C segment_AB)
  (H_X_properties : X_properties_BX_XC_AX_XD B X C A D) :
  is_right_triangle B X C := sorry

-- Prove triangle AXD is a right triangle
theorem triangle_AXD_right
  (H_segments_eq : equal_segments segment_AB segment_CD)
  (H_segments_perpendicular : perpendicular_segments segment_AB segment_CD)
  (H_C_on_AB : C_on_AB C segment_AB)
  (H_X_properties : X_properties_BX_XC_AX_XD B X C A D) :
  is_right_triangle A X D := sorry

end triangle_BXC_right_triangle_AXD_right_l334_334418


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334491

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334491


namespace least_five_digit_perfect_square_and_cube_l334_334506

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334506


namespace women_in_retail_l334_334867

theorem women_in_retail (total_population : ℕ) (half_population : total_population / 2 = women_count) 
  (third_of_women_work_in_retail : women_count / 3 = women_retail_count) :
  women_retail_count = 1000000 :=
by
  let total_population := 6000000
  let women_count := total_population / 2
  let women_retail_count := women_count / 3
  have h1 : women_count = 3000000 := rfl
  have h2 : women_retail_count = 1000000 := by
     rw [h1]
     exact rfl
  exact h2

end women_in_retail_l334_334867


namespace cone_same_volume_as_spherical_sector_l334_334681

theorem cone_same_volume_as_spherical_sector (R h : ℝ) (hR : 0 < R) (hh : 0 < h) :
  let V := (2 * π * R^2 * h) / 3 in
  (∃ (r hc : ℝ), r = R ∧ hc = 2 * h ∧ (1 / 3) * π * r^2 * hc = V) ∨
  (∃ (r' h' : ℝ), r' = R * Real.sqrt 2 ∧ h' = h ∧ (1 / 3) * π * r'^2 * h' = V) :=
by
  sorry

end cone_same_volume_as_spherical_sector_l334_334681


namespace eval_expression_l334_334651

theorem eval_expression : 4 * (8 - 3) - 6 = 14 :=
by
  sorry

end eval_expression_l334_334651


namespace chess_tournament_solution_l334_334160

def chess_tournament_points (points : List ℝ) : Prop :=
  let andrey := points[0]
  let dima := points[1]
  let vanya := points[2]
  let sasha := points[3]
  andrey = 4 ∧ dima = 3.5 ∧ vanya = 2.5 ∧ sasha = 2

axiom chess_tournament_conditions (points : List ℝ) :
  -- Andrey secured first place, Dima secured second, Vanya secured third, and Sasha secured fourth.
  List.Nodup points ∧
  points.length = 4 ∧
  (∀ p, p ∈ points → p = 4 ∨ p = 3.5 ∨ p = 2.5 ∨ p = 2) ∧
  -- Andrey and Sasha won the same number of games.
  (points[0] ≠ points[1] ∧ points[0] ≠ points[2] ∧ points[0] ≠ points[3] ∧
   points[1] ≠ points[2] ∧ points[1] ≠ points[3] ∧
   points[2] ≠ points[3])

theorem chess_tournament_solution (points : List ℝ) :
  chess_tournament_conditions points → chess_tournament_points points :=
by
  sorry

end chess_tournament_solution_l334_334160


namespace students_per_group_l334_334008

theorem students_per_group (total_students not_picked groups : ℕ) (h_total : total_students = 58) (h_not_picked : not_picked = 10) (h_groups: groups = 8) :
  (total_students - not_picked) / groups = 6 :=
by
  rw [h_total, h_not_picked, h_groups]
  norm_num
  sorry

end students_per_group_l334_334008


namespace mass_percentage_C_eq_40_91_l334_334661

def molar_mass (C_atoms H_atoms O_atoms : ℕ) : ℝ :=
  C_atoms * 12.01 + H_atoms * 1.008 + O_atoms * 16.00

def mass_percentage_C (C_atoms H_atoms O_atoms : ℕ) : ℝ :=
  (C_atoms * 12.01) / (molar_mass C_atoms H_atoms O_atoms) * 100

theorem mass_percentage_C_eq_40_91 (x : ℕ) :
  (mass_percentage_C x 8 6 = 40.91) ↔ (x = 6) :=
by
  sorry

end mass_percentage_C_eq_40_91_l334_334661


namespace solve_for_x_l334_334893

theorem solve_for_x : ∃ x : ℤ, 64^(3 * x + 1) = 16^(4 * x - 5) ∧ x = -13 := by
  use -13
  have h1 : 64 = 2^6 := by norm_num
  have h2 : 16 = 2^4 := by norm_num
  calc
    64^(3 * (-13) + 1) = (2^6)^(3 * (-13) + 1)    : by rw [h1]
    ...               = 2^(6 * (3 * (-13) + 1))   : by rw [pow_mul]
    ...               = 2^(6 * (-39 + 1))         : by rw [mul_add]
    ...               = 2^(6 * (-38))             : by rw [show 3 * (-13) + 1 = -38 by norm_num]
    ...               = 2^(-228)                  : by rw [mul_neg, show 6 * (-38) = -228 by norm_num]

    16^(4 * (-13) - 5) = (2^4)^(4 * (-13) - 5)    : by rw [h2]
    ...               = 2^(4 * (4 * (-13) - 5))   : by rw [pow_mul]
    ...               = 2^(4 * (-52 - 5))         : by rw [mul_sub]
    ...               = 2^(4 * (-57))             : by rw [show 4 * (-13) - 5 = -57 by norm_num]
    ...               = 2^(-228)                  : by rw [mul_neg, show 4 * (-57) = -228 by norm_num]

  exact ⟨rfl, rfl⟩

end solve_for_x_l334_334893


namespace limo_cost_is_correct_l334_334287

def prom_tickets_cost : ℕ := 2 * 100
def dinner_cost : ℕ := 120
def dinner_tip : ℕ := (30 * dinner_cost) / 100
def total_cost_before_limo : ℕ := prom_tickets_cost + dinner_cost + dinner_tip
def total_cost : ℕ := 836
def limo_hours : ℕ := 6
def limo_total_cost : ℕ := total_cost - total_cost_before_limo
def limo_cost_per_hour : ℕ := limo_total_cost / limo_hours

theorem limo_cost_is_correct : limo_cost_per_hour = 80 := 
by
  sorry

end limo_cost_is_correct_l334_334287


namespace composite_sum_of_four_integers_l334_334856

theorem composite_sum_of_four_integers 
  (a b c d : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) 
  (h_eq : a^2 + b^2 + a * b = c^2 + d^2 + c * d) : 
  ∃ n m : ℕ, 1 < a + b + c + d ∧ a + b + c + d = n * m ∧ 1 < n ∧ 1 < m := 
sorry

end composite_sum_of_four_integers_l334_334856


namespace fraction_replacement_result_l334_334275

theorem fraction_replacement_result (x : ℝ) (h : x = 2) : 
  let f := (λ x : ℝ, (x + 2) / (x - 2)) in 
  f (f x) = 1 := 
by 
  -- Definition and condition
  let f := (λ x : ℝ, (x + 2) / (x - 2))
  -- Assume x = 2
  have h : x = 2 := sorry
  -- Show that evaluating the nested function at x = 2 results in 1
  have : f (f x) = 1 := sorry
  exact this

end fraction_replacement_result_l334_334275


namespace sequence_a_n_sum_b_n_l334_334823

theorem sequence_a_n (a_n S_n : ℕ → ℕ) (h_positive : ∀ n, a_n n > 0) (h_relation : ∀ n, (a_n n)^2 + 2 * (a_n n) = 4 * (S_n n) + 3) :
  ∀ n, a_n n = 2 * n + 1 :=
sorry

theorem sum_b_n (b_n : ℕ → ℚ) (a_n S_n : ℕ → ℕ) (h_positive : ∀ n, a_n n > 0) (h_relation : ∀ n, (a_n n)^2 + 2 * (a_n n) = 4 * (S_n n) + 3)
  (b_relation : ∀ n, b_n n = (2^{a_n n} : ℚ) / ((2^{a_n n} + 1) * (2^{a_n (n + 1)} + 1))) :
  ∀ n, (finset.sum (finset.range n) b_n) = (2^(2 * n + 3) - 8) / (27 * (2^(2 * n + 3) + 1)) :=
sorry

end sequence_a_n_sum_b_n_l334_334823


namespace exists_quadratic_polynomial_with_given_property_l334_334098

theorem exists_quadratic_polynomial_with_given_property :
  ∃ f : ℚ[X], degree f = 2 ∧ (∀ n : ℤ, (n : ℚ) ∈ f.coeff) ∧ f.eval (f.eval (real.sqrt 3)) = 0 :=
by
  sorry

end exists_quadratic_polynomial_with_given_property_l334_334098


namespace least_five_digit_perfect_square_and_cube_l334_334431

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334431


namespace exists_quadratic_polynomial_with_property_l334_334090

theorem exists_quadratic_polynomial_with_property :
  ∃ (f : ℝ → ℝ), (∃ (a b c : ℤ), ∀ x, f x = (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)) ∧ f (f (real.sqrt 3)) = 0 :=
by
  sorry

end exists_quadratic_polynomial_with_property_l334_334090


namespace general_term_sum_alternate_terms_l334_334197

section

-- Defining the sequence {a_n} and the sum S_n where S_n = 2a_n - n
variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}

-- First condition: S(n) = 2 * a(n) - n
axiom seq_def (n : ℕ) : S n = 2 * a n - n

-- First part of the proof: finding the general formula for a_n
theorem general_term (n : ℕ) : a n = 2^n - 1 :=
sorry

-- Second part of the proof: finding the sum of a_1 + a_3 + a_5 + ... + a_{2n+1}
theorem sum_alternate_terms (n : ℕ) : ∑ i in finset.range (n + 1), a (2 * i + 1) = (2^(2*n + 3) - 3 * n - 5) / 3 :=
sorry

end

end general_term_sum_alternate_terms_l334_334197


namespace probability_of_two_rain_days_l334_334420

def is_rain_day (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

def count_rain_days (nums : List ℕ) : ℕ :=
  nums.countp is_rain_day

def is_target_group (nums : List ℕ) : Prop :=
  count_rain_days nums = 2

def groups : List (List ℕ) :=
  [[9,0,7], [9,6,6], [1,9,1], [9,2,5], [2,7,1],
   [9,3,2], [8,1,2], [4,5,8], [5,6,9], [6,8,3],
   [4,3,1], [2,5,7], [3,9,3], [0,2,7], [5,5,6],
   [4,8,8], [7,3,0], [1,1,3], [5,3,7], [9,8,9]]

def target_probability (groups : List (List ℕ)) : ℚ :=
  (groups.filter is_target_group).length / groups.length

theorem probability_of_two_rain_days : target_probability groups = 1/4 :=
by 
  -- Proof will be placed here
  sorry

end probability_of_two_rain_days_l334_334420


namespace range_f_in_interval_l334_334743

-- Define the function f and the interval
def f (x : ℝ) (f_deriv_neg1 : ℝ) := x^3 + 2 * x * f_deriv_neg1
def interval := Set.Icc (-2 : ℝ) (3 : ℝ)

-- State the theorem
theorem range_f_in_interval :
  ∃ (f_deriv_neg1 : ℝ),
  (∀ x ∈ interval, f x f_deriv_neg1 ∈ Set.Icc (-4 * Real.sqrt 2) 9) :=
sorry

end range_f_in_interval_l334_334743


namespace domain_of_f_l334_334225

def f (x : ℝ) : ℝ := x + 1 + Real.sqrt x

theorem domain_of_f : ∀ x, (0 ≤ x) ↔ (∃ y, f y = x) := sorry

end domain_of_f_l334_334225


namespace sum_of_solutions_l334_334531

theorem sum_of_solutions : 
  let integer_solutions := { x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 } in
  ∑ x in integer_solutions, x = 24 := 
sorry

end sum_of_solutions_l334_334531


namespace max_c_magnitude_l334_334714

variables {a b c : ℝ × ℝ}

-- Definitions of the given conditions
def unit_vector (v : ℝ × ℝ) : Prop := ‖v‖ = 1
def orthogonal (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def satisfied_c (c a b : ℝ × ℝ) : Prop := ‖c - (a + b)‖ = 2

-- Main theorem to prove
theorem max_c_magnitude (ha : unit_vector a) (hb : unit_vector b) (hab : orthogonal a b) (hc : satisfied_c c a b) : ‖c‖ ≤ 2 + Real.sqrt 2 := 
sorry

end max_c_magnitude_l334_334714


namespace minimum_choir_size_l334_334599

theorem minimum_choir_size : ∃ (choir_size : ℕ), 
  (choir_size % 9 = 0) ∧ 
  (choir_size % 11 = 0) ∧ 
  (choir_size % 13 = 0) ∧ 
  (choir_size % 10 = 0) ∧ 
  (choir_size = 12870) :=
by
  sorry

end minimum_choir_size_l334_334599


namespace equation_of_line_l334_334582

-- Definitions based on conditions from part a)
def point_P : ℝ × ℝ := (2, 3)
def slope_120_degrees : ℝ := -real.sqrt 3
def intercept_sum_zero : (ℝ × ℝ) → Prop := λ a b, a + b = 0

-- Mathematical property to be proved
theorem equation_of_line (l : ℝ → ℝ) :
  (∀ x, l x = -real.sqrt 3 * x + 2 * real.sqrt 3 + 3) ∨
  (∀ x, l x = 1.5 * x) ∨
  (∀ x, l x = x + 1) :=
sorry

end equation_of_line_l334_334582


namespace total_cost_correct_l334_334548

-- Conditions given in the problem.
def net_profit : ℝ := 44
def gross_revenue : ℝ := 47
def lemonades_sold : ℝ := 50
def babysitting_income : ℝ := 31

def cost_per_lemon : ℝ := 0.20
def cost_per_sugar : ℝ := 0.15
def cost_per_ice : ℝ := 0.05

def one_time_cost_sunhat : ℝ := 10

-- Definition of variable cost per lemonade.
def variable_cost_per_lemonade : ℝ := cost_per_lemon + cost_per_sugar + cost_per_ice

-- Definition of total variable cost for all lemonades sold.
def total_variable_cost : ℝ := lemonades_sold * variable_cost_per_lemonade

-- Final total cost to operate the lemonade stand.
def total_cost : ℝ := total_variable_cost + one_time_cost_sunhat

-- The proof statement that total cost is equal to $30.
theorem total_cost_correct : total_cost = 30 := by
  sorry

end total_cost_correct_l334_334548


namespace linear_function_decreasing_iff_l334_334084

-- Define the conditions
def linear_function (m b x : ℝ) : ℝ := m * x + b

-- Define the condition for decreasing function
def is_decreasing (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 ≥ f x2

-- The theorem to prove
theorem linear_function_decreasing_iff (m b : ℝ) :
  (is_decreasing (linear_function m b)) ↔ (m < 0) :=
by
  sorry

end linear_function_decreasing_iff_l334_334084


namespace sofia_total_time_l334_334892

-- Definitions for the conditions
def laps : ℕ := 5
def track_length : ℕ := 400  -- in meters
def speed_first_100 : ℕ := 4  -- meters per second
def speed_remaining_300 : ℕ := 5  -- meters per second

-- Times taken for respective distances
def time_first_100 (distance speed : ℕ) : ℕ := distance / speed
def time_remaining_300 (distance speed : ℕ) : ℕ := distance / speed

def time_one_lap : ℕ := time_first_100 100 speed_first_100 + time_remaining_300 300 speed_remaining_300
def total_time_seconds : ℕ := laps * time_one_lap
def total_time_minutes : ℕ := 7
def total_time_extra_seconds : ℕ := 5

-- Problem statement
theorem sofia_total_time :
  total_time_seconds = total_time_minutes * 60 + total_time_extra_seconds :=
by
  sorry

end sofia_total_time_l334_334892


namespace total_area_calculations_l334_334102

noncomputable def total_area_in_hectares : ℝ :=
  let sections := 5
  let area_per_section := 60
  let conversion_factor_acre_to_hectare := 0.404686
  sections * area_per_section * conversion_factor_acre_to_hectare

noncomputable def total_area_in_square_meters : ℝ :=
  let conversion_factor_hectare_to_square_meter := 10000
  total_area_in_hectares * conversion_factor_hectare_to_square_meter

theorem total_area_calculations :
  total_area_in_hectares = 121.4058 ∧ total_area_in_square_meters = 1214058 := by
  sorry

end total_area_calculations_l334_334102


namespace least_five_digit_perfect_square_cube_l334_334515

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334515


namespace intersecting_line_at_one_point_l334_334383

theorem intersecting_line_at_one_point (k : ℝ) :
  (∃ y : ℝ, k = -3 * y^2 - 4 * y + 7 ∧ 
           ∀ z : ℝ, k = -3 * z^2 - 4 * z + 7 → y = z) ↔ 
  k = 25 / 3 :=
by
  sorry

end intersecting_line_at_one_point_l334_334383


namespace content_paths_l334_334559

theorem content_paths : 
  ∃ n, n = 129 ∧
    let grid := [ 
    -- Row 1
    [none, none, none, none, none, none, none, some 'C', none, none, none, none, none, none, none], 
    -- Row 2
    [none, none, none, none, none, none, some 'C', some 'O', some 'C', none, none, none, none, none, none], 
    -- Row 3
    [none, none, none, none, none, some 'C', some 'O', some 'N', some 'O', some 'C', none, none, none, none, none],
    -- Row 4
    [none, none, none, none, some 'C', some 'O', some 'N', some 'T', some 'N', some 'O', some 'C', none, none, none, none],
    -- Row 5
    [none, none, none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'T', some 'N', some 'O', some 'C', none, none, none],
    -- Row 6
    [none, none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C', none, none],
    -- Row 7
    [none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'T', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C', none],
    -- Row 8
    [some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'T', none, none, some 'T', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C'] 
    ] in 
    ∀ p : ℕ → ℕ → list char, 
    (∀ i j, (i, j) ∈ grid → p i j = ['C', 'O', 'N', 'T', 'E', 'N', 'T'] 
    → (i, j) are adjacent → p i j = 129). 
sorry

end content_paths_l334_334559


namespace sum_2001_terms_is_986_l334_334369

-- Given definitions to set up the problem
def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, n ≥ 3 → a n = a (n - 1) - a (n - 2)

def sum_first_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

-- The theorem which needs to be proven
theorem sum_2001_terms_is_986 (a : ℕ → ℝ) (h_seq : sequence a)
    (h_sum_1492 : sum_first_n a 1492 = 1985)
    (h_sum_1985 : sum_first_n a 1985 = 1492) :
    sum_first_n a 2001 = 986 :=
sorry

end sum_2001_terms_is_986_l334_334369


namespace heptagon_angle_arithmetic_progression_l334_334905

theorem heptagon_angle_arithmetic_progression : ∃ k : ℕ, k ∈ {126, 128, 129, 130, 132} ∧ 
  (∃ a n : ℝ, a + 3 * n = k ∧ 7 * a + 21 * n = 900) :=
sorry

end heptagon_angle_arithmetic_progression_l334_334905


namespace trig_identity_l334_334689

theorem trig_identity (x : ℝ) (h : Real.tan x = 1 / 3) : 
  sin x * cos x + 1 = 13 / 10 :=
  sorry

end trig_identity_l334_334689


namespace exists_quadratic_polynomial_with_property_l334_334091

theorem exists_quadratic_polynomial_with_property :
  ∃ (f : ℝ → ℝ), (∃ (a b c : ℤ), ∀ x, f x = (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)) ∧ f (f (real.sqrt 3)) = 0 :=
by
  sorry

end exists_quadratic_polynomial_with_property_l334_334091


namespace find_m_and_z2_abs_equal_sqrt_10_l334_334729

-- Define z1 and z2
def z1 (m : ℝ) : ℂ := complex.mk (m^2 + 2*m - 3) (m - 1)
def z2 (z1 : ℂ) : ℂ := complex.div (complex.mk 4 (-2)) ((complex.add (complex.of_real 1) (complex.div (complex.of_real 1) (complex.of_real 4 * z1))) * complex.I)

-- Conditions
axiom z1_is_pure_imaginary : ∀ (m : ℝ), z1 m = complex.I * complex.of_real (m - 1) → m = -3

-- Main theorem statement
theorem find_m_and_z2_abs_equal_sqrt_10 :
  let m := -3 in
  z1_is_pure_imaginary m →
  |z2 (-4 * complex.I)| = complex.abs (complex.mk 1 (-3)) := by
  sorry

end find_m_and_z2_abs_equal_sqrt_10_l334_334729


namespace least_five_digit_perfect_square_and_cube_l334_334470

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334470


namespace QR_value_l334_334898

-- Given conditions for the problem
def QP : ℝ := 15
def sinQ : ℝ := 0.4

-- Define QR based on the given conditions
noncomputable def QR : ℝ := QP / sinQ

-- The theorem to prove that QR = 37.5
theorem QR_value : QR = 37.5 := 
by
  unfold QR QP sinQ
  sorry

end QR_value_l334_334898


namespace chess_tournament_scores_l334_334135

theorem chess_tournament_scores :
    ∃ (A D V S : ℝ),
    A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S ∧
    A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 ∧
    A > D ∧ D > V ∧ V > S ∧
    (∃ (wins_A wins_S : ℕ), wins_A = wins_S) :=
begin
    sorry
end

end chess_tournament_scores_l334_334135


namespace angle_between_a_c_l334_334708

variables {V : Type*} [InnerProductSpace ℝ V]

theorem angle_between_a_c 
  (a b c : V) 
  (hab : a + b + c = 0)
  (angle_ab : real.angle a b = real.angle.deg 150)
  (norm_b : ∥b∥ = (2 * real.sqrt 3 / 3) * ∥a∥) :
  real.angle a c = real.angle.deg 90 :=
by
  sorry

end angle_between_a_c_l334_334708


namespace determine_a_range_of_b_l334_334861

section
  /-- Definition of the piecewise function f --/
  def f : ℝ → ℝ
  | x => if x < 0 then -4 * x^2 else x^2 - x

  /-- The first part of the proof problem: Determining a such that f(a) = -1/4 --/
  theorem determine_a (a : ℝ) (h : f a = -1/4) : a = -1/4 ∨ a = 1/2 := sorry

  /-- The second part of the proof problem: Determining the range of b for which f(x) - b = 0 has three distinct real roots --/
  theorem range_of_b (b : ℝ) : (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = b ∧ f x₂ = b ∧ f x₃ = b) ↔ b ∈ Ioo (-1/4) 0 := sorry
end

end determine_a_range_of_b_l334_334861


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334501

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334501


namespace walking_rate_ratio_l334_334568

-- Define the variables and conditions.
variables (R R' : ℝ) (D : ℝ)

-- Define the usual time, early arrival time and distance.
def usual_time := 49
def early_arrival := 7
def distance_usual := R * usual_time
def distance_new := R' * (usual_time - early_arrival)

-- Statement to prove the required ratio.
theorem walking_rate_ratio : 
  (distance_usual = D) → (distance_new = D) → (R' / R = 7 / 6) := 
by
  intros h1 h2
  exact sorry

end walking_rate_ratio_l334_334568


namespace unique_isosceles_triangle_y_coordinate_sum_l334_334301

theorem unique_isosceles_triangle_y_coordinate_sum :
  (u v w q : ℤ) 
  (yQ : ℤ -> ℤ) (yA : ℤ -> ℤ) :
  -- Conditions for triangle PQO
  (∃ q : ℤ, in_parabola₁ yQ q) ∧
  -- yQ is the function describing the y-coordinate for point Q
  (yQ q = 12 * q ^ 2) →
  -- Conditions for triangle ABV
  (yA : ℤ -> ℤ = fun b => b^2 / 5 + 1) ∧
  -- yA is the function describing the y-coordinate for point A
  (∃ b : ℤ, in_parabola₂ yA b) →
  -- Relationship for y-coordinate of A in terms of q
  (yA (k * q) = u * q^2 + v * q + w) →
  -- Correct answer condition
  u + v + w = 781 := sorry

-- Auxiliary definitions
def in_parabola₁ (yQ : ℤ -> ℤ) (q : ℤ) : Prop :=
  ∃ x, yQ x = 12 * x ^ 2

def in_parabola₂ (yA : ℤ -> ℤ) (b : ℤ) : Prop :=
  ∃ y, yA y = y ^ 2 / 5 + 1

end unique_isosceles_triangle_y_coordinate_sum_l334_334301


namespace extreme_value_of_f_range_of_a_l334_334747

noncomputable def f (x : ℝ) : ℝ := x * real.exp (x + 1)

theorem extreme_value_of_f : f (-1) = -1 := by
  -- The proof would go here
  sorry

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → f x ≥ x + real.log x + a + 1) → a ≤ 1 := by
  -- The proof would go here
  sorry

end extreme_value_of_f_range_of_a_l334_334747


namespace sum_of_squares_divisible_by_a1_l334_334858

open Nat

def pairwise_coprime (l : List ℕ) : Prop :=
  ∀ i j : Fin l.length, i ≠ j → gcd (l.get i) (l.get j) = 1

def is_prime (p : ℕ) : Prop := Nat.Prime p

def create_intervals (a : List ℕ) : ℕ → List (ℕ × ℕ)
| 0     => []
| (n+1) => let Ai := a.take (n+1).prod in
           let marked_points := Finset.image (λ k, k * a ! 0) (Finset.range ((Ai / a ! 0) + 1)) in
           let intervals := Finset.sort lt ((Finset.range (Ai + 1)).sdiff marked_points).toList in
           if intervals = list.nil then [] else (List.zip intervals (List.tail intervals)) ++ create_intervals a n

theorem sum_of_squares_divisible_by_a1 (a : List ℕ) (h1 : a.Nth 0 > 0) (h2 : pairwise_coprime a) (h3 : a.get 0 ≥ a.length + 2) (h4 : is_prime (a.get 0)) : 
  let intervals := create_intervals a (a.length - 1) in
  (Finset.sum (Finset.image (λ i, (i.2 - i.1) ^ 2) (Finset.of_list intervals)) % a.get 0) = 0 :=
sorry

end sum_of_squares_divisible_by_a1_l334_334858


namespace chess_tournament_points_l334_334146

theorem chess_tournament_points
  (points : String → ℝ)
  (total_points : points "Andrey" + points "Dima" + points "Vanya" + points "Sasha" = 12)
  (distinct_points : 
    points "Andrey" ≠ points "Dima" ∧ 
    points "Andrey" ≠ points "Vanya" ∧ 
    points "Andrey" ≠ points "Sasha" ∧ 
    points "Dima" ≠ points "Vanya" ∧ 
    points "Dima" ≠ points "Sasha" ∧ 
    points "Vanya" ≠ points "Sasha")
  (order : 
    points "Andrey" > points "Dima" ∧ 
    points "Dima" > points "Vanya" ∧ 
    points "Vanya" > points "Sasha")
  (same_wins :
    let games_won (student : String) := (points student - 3) / 0.5 in
    games_won "Andrey" = games_won "Sasha") :
  points "Andrey" = 4 ∧ points "Dima" = 3.5 ∧ points "Vanya" = 2.5 ∧ points "Sasha" = 2 :=
by
  sorry

end chess_tournament_points_l334_334146


namespace sum_mod_13_l334_334664

theorem sum_mod_13 :
  (9023 % 13 = 5) → 
  (9024 % 13 = 6) → 
  (9025 % 13 = 7) → 
  (9026 % 13 = 8) → 
  ((9023 + 9024 + 9025 + 9026) % 13 = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end sum_mod_13_l334_334664


namespace sufficient_but_not_necessary_condition_l334_334750

theorem sufficient_but_not_necessary_condition
  (a : ℝ)
  (l1 : ℝ → ℝ → ℝ := λ x y, x + a * y - 2)
  (l2 : ℝ → ℝ → ℝ := λ x y, (a + 1) * x - a * y + 1)
  (parallel : (ℝ → ℝ → ℝ) → (ℝ → ℝ → ℝ) → Prop := λ l1 l2, ∀ x y x' y', l1 x y = 0 ∧ l2 x' y' = 0 → x/x' = a/(a+1) ∧ y/y' = a/a):
  (parallel l1 l2 ↔ a = -2) ∨ (parallel l1 l2 ↔ a = 0) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l334_334750


namespace ratio_of_area_to_sum_of_perimeter_and_bisector_l334_334962

noncomputable def area_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (side_length^2 * real.sqrt 3) / 4

noncomputable def perimeter_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  3 * side_length

noncomputable def bisector_length_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (side_length * real.sqrt 3) / 2

noncomputable def ratio_equilateral_triangle (side_length : ℝ) : ℝ :=
  let A := area_of_equilateral_triangle side_length
  let P := perimeter_of_equilateral_triangle side_length
  let B := bisector_length_of_equilateral_triangle side_length
  A / (P + B)

theorem ratio_of_area_to_sum_of_perimeter_and_bisector :
  ratio_equilateral_triangle 10 = (10 * real.sqrt 3 - 5) / 11 := by
    sorry

end ratio_of_area_to_sum_of_perimeter_and_bisector_l334_334962


namespace swappable_propositions_l334_334768

def proposition1 (line plane : Type) : Prop := 
∀ (l₁ l₂ : line) (p : plane), 
l₁ ⟂ p ∧ l₂ ⟂ p → l₁ ∥ l₂

def proposition2 (plane : Type) : Prop := 
∀ (p₁ p₂ p₃ : plane), 
p₁ ⟂ p₃ ∧ p₂ ⟂ p₃ → p₁ ∥ p₂

def proposition3 (line : Type) : Prop := 
∀ (l₁ l₂ l₃ : line), 
l₁ ∥ l₃ ∧ l₂ ∥ l₃ → l₁ ∥ l₂

def proposition4 (line plane : Type) : Prop :=
∀ (l₁ l₂ : line) (p : plane), 
l₁ ∥ p ∧ l₂ ∥ p → l₁ ∥ l₂

def swappable_prop (line plane : Type) (prop : Prop) : Prop :=
let swapped_prop := ∀ (p₁ p₂ : plane) (l : line), prop in
prop ∧ swapped_prop

theorem swappable_propositions (line plane : Type) :
  swappable_prop line plane (proposition1 line plane) ∧
  swappable_prop line plane (proposition3 line)  := 
by 
  sorry

end swappable_propositions_l334_334768


namespace least_five_digit_perfect_square_and_cube_l334_334450

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334450


namespace parallel_planes_sufficient_condition_l334_334052

theorem parallel_planes_sufficient_condition
  (α β γ : Plane)
  (m n : Line)
  (H1 : n ∥ m)
  (H2 : n ⊥ α)
  (H3 : m ⊥ β) :
  α ∥ β :=
sorry

end parallel_planes_sufficient_condition_l334_334052


namespace number_of_three_digit_cubes_divisible_by_9_l334_334248

theorem number_of_three_digit_cubes_divisible_by_9 :
  ∃ n : ℕ, (999 / 27 : ℕ) = 37 ∧ n ∈ { k | 100 ≤ k * (27 : ℕ) * k^2 ∧ k * (27 : ℕ) * k^2 ≤ 999 } → set.size { k ∈ finset.range(4) | 100 ≤ (27 * k^3 : ℕ) ∧ (27 * k^3 : ℕ) ≤ 999 } = 2 := 
by
  sorry

end number_of_three_digit_cubes_divisible_by_9_l334_334248


namespace probability_of_event_l334_334362

noncomputable theory

def domain (x : ℝ) : Prop := (0 < x ∧ x < 1) ∨ (1 < x)

def die_values : set ℝ := {1, 2, 3, 4, 5, 6}

def favorable_outcomes : set ℝ := { t ∈ die_values | domain t }

def probability : ℝ := (favorable_outcomes.to_finset.card : ℝ) / (die_values.to_finset.card : ℝ)

theorem probability_of_event : probability = 5 / 6 := 
by 
  sorry

end probability_of_event_l334_334362


namespace solve_for_x_l334_334894

theorem solve_for_x (x : ℝ) : (3^x * 9^x = 27^(x - 20)) → x = 20 :=
by
  sorry

end solve_for_x_l334_334894


namespace sum_of_geometric_sequence_l334_334043

noncomputable def geometric_sequence (n : ℕ) : ℝ :=
if odd n then 2 else 5 / 2

def sum_first_n_terms (n : ℕ) : ℝ :=
∑ i in finset.range n, geometric_sequence (i + 1)

theorem sum_of_geometric_sequence (n : ℕ) : 
  sum_first_n_terms n = 
  if even n then 9*n / 4 
  else (9*n - 1) / 4 := 
sorry

end sum_of_geometric_sequence_l334_334043


namespace sum_of_solutions_l334_334529

theorem sum_of_solutions : 
  let integer_solutions := { x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 } in
  ∑ x in integer_solutions, x = 24 := 
sorry

end sum_of_solutions_l334_334529


namespace competition_ratings_l334_334778

theorem competition_ratings (a b k : ℕ) (h_b_ge_3 : b ≥ 3) (h_b_odd : b % 2 = 1)
  (h : ∀ (J1 J2 : ℕ), J1 ≠ J2 → ∀ (C : ℕ), J1 ≤ b ∧ J2 ≤ b ∧ C ≤ a ∧ (J1, J2, C) is consistent ≤ k) :
  (k:ℝ) / (a:ℝ) ≥ (b - 1 : ℝ) / (2 * b : ℝ) :=
by
  sorry

end competition_ratings_l334_334778


namespace ratio_chloe_to_max_l334_334626

/-- Chloe’s wins and Max’s wins -/
def chloe_wins : ℕ := 24
def max_wins : ℕ := 9

/-- The ratio of Chloe's wins to Max's wins is 8:3 -/
theorem ratio_chloe_to_max : (chloe_wins / Nat.gcd chloe_wins max_wins) = 8 ∧ (max_wins / Nat.gcd chloe_wins max_wins) = 3 := by
  sorry

end ratio_chloe_to_max_l334_334626


namespace quadratic_single_root_pos_value_l334_334260

theorem quadratic_single_root_pos_value (m : ℝ) (h1 : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
sorry

end quadratic_single_root_pos_value_l334_334260


namespace trajectory_of_M_l334_334027

open Real

-- Define the endpoints A and B
variable {A B M : Real × Real}

-- Given conditions
def segment_length (A B : Real × Real) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 25

def on_axes (A B : Real × Real) : Prop :=
  A.2 = 0 ∧ B.1 = 0

def point_m_relationship (A B M : Real × Real) : Prop :=
  let AM := (M.1 - A.1, M.2 - A.2)
  let MB := (M.1 - B.1, M.2 - B.2)
  AM.1 = (2 / 3) * MB.1 ∧ AM.2 = (2 / 3) * MB.2 ∧
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = 4

theorem trajectory_of_M (A B M : Real × Real)
  (h1 : segment_length A B)
  (h2 : on_axes A B)
  (h3 : point_m_relationship A B M) :
  (M.1^2 / 9) + (M.2^2 / 4) = 1 :=
sorry

end trajectory_of_M_l334_334027


namespace find_a_l334_334311

theorem find_a (a : ℝ) (h : log 2 (4^a + 4) = a + log 2 (2^(a + 1) - 3)) : a = 2 :=
sorry

end find_a_l334_334311


namespace shortest_distance_to_y_axis_l334_334406

noncomputable def parabola : set (ℝ × ℝ) := {p | ∃ x, p = (x, sqrt (8 * x)) ∨ p = (x, -sqrt (8 * x))}

theorem shortest_distance_to_y_axis :
  ∀ (A B : ℝ × ℝ), 
  (A ∈ parabola) → (B ∈ parabola) → 
  (let d := dist A B in d = 10) →
  let P := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  abs P.1 = 3 :=
by
  intros A B hA hB hAB P
  sorry

end shortest_distance_to_y_axis_l334_334406


namespace hall_length_width_difference_l334_334554

theorem hall_length_width_difference (L W : ℝ) 
(h1 : W = 1 / 2 * L) 
(h2 : L * W = 200) : L - W = 10 := 
by 
  sorry

end hall_length_width_difference_l334_334554


namespace sum_first_five_terms_geometric_sequence_l334_334667

noncomputable def sum_first_five_geometric (a0 : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a0 * (1 - r^n) / (1 - r)

theorem sum_first_five_terms_geometric_sequence : 
  sum_first_five_geometric (1/3) (1/3) 5 = 121 / 243 := 
by 
  sorry

end sum_first_five_terms_geometric_sequence_l334_334667


namespace banana_bread_pieces_l334_334290

def area_pan : ℕ := 24 * 20

def area_piece : ℕ := 3 * 4

def number_of_pieces (area_pan area_piece : ℕ) : ℕ :=
  area_pan / area_piece

theorem banana_bread_pieces : number_of_pieces area_pan area_piece = 40 := by
  -- conditions
  let pan_dims := (24, 20)
  let piece_dims := (3, 4)
  have area_pan_eq : area_pan = 24 * 20 := rfl
  have area_piece_eq : area_piece = 3 * 4 := rfl
  have div_eq : number_of_pieces (24 * 20) (3 * 4) = 40 := by
    rw [number_of_pieces, area_pan_eq, area_piece_eq]
    exact Nat.div_eq_of_eq_mul (Eq.symm (Nat.mul_div_cancel_left 480 12)).symm
  rw [area_pan_eq, area_piece_eq]
  exact div_eq

end banana_bread_pieces_l334_334290


namespace find_x_l334_334349

theorem find_x (x y z : ℝ) (h1 : x^2 / y = 4) (h2 : y^2 / z = 9) (h3 : z^2 / x = 16) : x = 4 :=
sorry

end find_x_l334_334349


namespace closest_integer_to_sum_l334_334116

theorem closest_integer_to_sum : 
  ∃ (k : ℤ), abs (500 * ∑ n in Finset.range (20000 - 2 + 1) + 2, 1 / (n^2 - 1) - (375 : ℤ)) < 1 := 
by
  -- The proof will use the telescoping series property and demonstrate that 375 is the closest integer.
  sorry

end closest_integer_to_sum_l334_334116


namespace f_continuous_l334_334860

noncomputable def f : ℝ → ℝ := sorry

axiom surjective_f : Function.Surjective f

axiom seq_property (x_n : ℕ → ℝ) :
  Filter.Tendsto (fun n => f (x_n n)) Filter.atTop (Filter.principal {a : ℝ | ∃ l, Filter.Tendsto (x_n n) Filter.atTop (Filter.principal {l})}) →

theorem f_continuous : Continuous f := 
sorry

end f_continuous_l334_334860


namespace distance_between_foci_of_ellipse_l334_334659

theorem distance_between_foci_of_ellipse : 
  let eq := (9 * x^2 + 36 * x + 4 * y^2 - 8 * y + 1 = 0) in
  distance_between_foci (ellipse eq) = 2 * (sqrt 195) / 3 :=
sorry

end distance_between_foci_of_ellipse_l334_334659


namespace sum_f_eq_28743_l334_334109

def f (n : ℕ) : ℕ := 4 * n ^ 3 - 6 * n ^ 2 + 4 * n + 13

theorem sum_f_eq_28743 : (Finset.range 13).sum (λ n => f (n + 1)) = 28743 :=
by
  -- Placeholder for actual proof
  sorry

end sum_f_eq_28743_l334_334109


namespace probability_two_distinct_real_roots_l334_334277

theorem probability_two_distinct_real_roots : 
  (∀ m : ℝ, m ∈ set.Ioo 0 1 → (4 * m^2 - 2 > 0)) 
  → (measure_theory.volume (set.Ioo (real.sqrt 2 / 2) 1) / measure_theory.volume (set.Ioo 0 1) = (2 - real.sqrt 2) / 2) := 
  sorry

end probability_two_distinct_real_roots_l334_334277


namespace trajectory_is_parabola_l334_334254

def distance_to_line (p : ℝ × ℝ) (a : ℝ) : ℝ :=
|p.1 - a|

noncomputable def distance_to_point (p q : ℝ × ℝ) : ℝ :=
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def parabola_condition (P : ℝ × ℝ) : Prop :=
distance_to_line P (-1) + 1 = distance_to_point P (2, 0)

theorem trajectory_is_parabola : ∀ (P : ℝ × ℝ), parabola_condition P ↔
(P.1 + 1)^2 = (Real.sqrt ((P.1 - 2)^2 + P.2^2))^2 := 
by 
  sorry

end trajectory_is_parabola_l334_334254


namespace negation_of_proposition_l334_334390

theorem negation_of_proposition :
  (¬ (∀ a b : ℤ, a = 0 → a * b = 0)) ↔ (∃ a b : ℤ, a = 0 ∧ a * b ≠ 0) :=
by
  sorry

end negation_of_proposition_l334_334390


namespace weight_of_new_person_l334_334355

theorem weight_of_new_person:
  (∀ (W : ℝ) (avg_increase : ℝ) (original_weight : ℝ) (new_person_weight : ℝ),
    avg_increase = 2.5 → 
    original_weight = 20 → 
    (W + 8 * avg_increase = (W - original_weight) + new_person_weight) → 
    new_person_weight = 40) :=
by
  intros W avg_increase original_weight new_person_weight h1 h2 h3
  rw [h1, h2] at h3
  linarith

end weight_of_new_person_l334_334355


namespace pqrs_product_l334_334213

noncomputable def P : ℝ := Real.sqrt 2012 + Real.sqrt 2013
noncomputable def Q : ℝ := -Real.sqrt 2012 - Real.sqrt 2013
noncomputable def R : ℝ := Real.sqrt 2012 - Real.sqrt 2013
noncomputable def S : ℝ := Real.sqrt 2013 - Real.sqrt 2012

theorem pqrs_product : P * Q * R * S = 1 := 
by 
  sorry

end pqrs_product_l334_334213


namespace trapezoid_height_l334_334371

theorem trapezoid_height (BC AD AB CD h : ℝ) (hBC : BC = 4) (hAD : AD = 25) (hAB : AB = 20) (hCD : CD = 13) :
  h = 12 :=
by
  sorry

end trapezoid_height_l334_334371


namespace exists_quadratic_poly_with_integer_coeffs_l334_334094

theorem exists_quadratic_poly_with_integer_coeffs (α : ℝ) :
  (∃ (a b c : ℤ), ∀ x : ℝ, (λ x : ℝ, ↑a * x^2 + ↑b * x + ↑c) (↑(λ x : ℝ, ↑a * x^2 + ↑b * x + ↑c) (α)) = 0) :=
  sorry

end exists_quadratic_poly_with_integer_coeffs_l334_334094


namespace compare_numbers_l334_334083

theorem compare_numbers : (0.7^6 < 1) → (6^0.7 > 1) → (log 0.7 6 < 0) → (log 0.7 6 < 0.7^6 ∧ 0.7^6 < 6^0.7) := 
by
  intros h1 h2 h3
  exact ⟨h3, h1.trans h2⟩

end compare_numbers_l334_334083


namespace linear_function_properties_l334_334722

theorem linear_function_properties :
  ∃ k b, (∀ x y, (y = k * x + b) ↔ ((x = 1 ∧ y = -1) ∨ (x = -2 ∧ y = 8)))
  ∧ (k = -3) ∧ (b = 2)
  ∧ (∀ y, y = (-3 * (-10) + 2) ↔ y = 32) := 
by
  use [-3, 2]
  sorry

end linear_function_properties_l334_334722


namespace part1_extreme_value_part2_range_of_a_l334_334745

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (x + 1)

theorem part1_extreme_value :
  ∃ x : ℝ, f x = -1 :=
  sorry

theorem part2_range_of_a :
  ∀ x > 0, ∃ a : ℝ, f x ≥ x + Real.log x + a + 1 → a ≤ 1 :=
  sorry

end part1_extreme_value_part2_range_of_a_l334_334745


namespace chess_tournament_solution_l334_334163

def chess_tournament_points (points : List ℝ) : Prop :=
  let andrey := points[0]
  let dima := points[1]
  let vanya := points[2]
  let sasha := points[3]
  andrey = 4 ∧ dima = 3.5 ∧ vanya = 2.5 ∧ sasha = 2

axiom chess_tournament_conditions (points : List ℝ) :
  -- Andrey secured first place, Dima secured second, Vanya secured third, and Sasha secured fourth.
  List.Nodup points ∧
  points.length = 4 ∧
  (∀ p, p ∈ points → p = 4 ∨ p = 3.5 ∨ p = 2.5 ∨ p = 2) ∧
  -- Andrey and Sasha won the same number of games.
  (points[0] ≠ points[1] ∧ points[0] ≠ points[2] ∧ points[0] ≠ points[3] ∧
   points[1] ≠ points[2] ∧ points[1] ≠ points[3] ∧
   points[2] ≠ points[3])

theorem chess_tournament_solution (points : List ℝ) :
  chess_tournament_conditions points → chess_tournament_points points :=
by
  sorry

end chess_tournament_solution_l334_334163


namespace minimum_value_f_within_domain_l334_334693

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem minimum_value_f_within_domain :
  ∀ x : ℝ, x ≥ 1 → ∃ y : ℝ, y = f 1 ∧ ∀ z : ℝ, z ∈ set.Ici 1 → f z ≥ y :=
sorry

end minimum_value_f_within_domain_l334_334693


namespace maddox_more_profit_than_theo_l334_334319

-- Definitions (conditions)
def cost_per_camera : ℕ := 20
def num_cameras : ℕ := 3
def total_cost : ℕ := num_cameras * cost_per_camera

def maddox_selling_price_per_camera : ℕ := 28
def theo_selling_price_per_camera : ℕ := 23

-- Total selling price
def maddox_total_selling_price : ℕ := num_cameras * maddox_selling_price_per_camera
def theo_total_selling_price : ℕ := num_cameras * theo_selling_price_per_camera

-- Profits
def maddox_profit : ℕ := maddox_total_selling_price - total_cost
def theo_profit : ℕ := theo_total_selling_price - total_cost

-- Proof Statement
theorem maddox_more_profit_than_theo : maddox_profit - theo_profit = 15 := by
  sorry

end maddox_more_profit_than_theo_l334_334319


namespace chess_tournament_scores_l334_334132

theorem chess_tournament_scores :
    ∃ (A D V S : ℝ),
    A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S ∧
    A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 ∧
    A > D ∧ D > V ∧ V > S ∧
    (∃ (wins_A wins_S : ℕ), wins_A = wins_S) :=
begin
    sorry
end

end chess_tournament_scores_l334_334132


namespace mutuallyOrthogonalSet_is_M4_l334_334700

def mutuallyOrthogonalSet (M : set (ℝ × ℝ)) : Prop :=
  ∀ (x1 y1 : ℝ), (x1, y1) ∈ M → ∃ (x2 y2 : ℝ), (x2, y2) ∈ M ∧ x1 * x2 + y1 * y2 = 0

def M1 : set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 ^ 2 + 1}
def M2 : set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.log p.1}
def M3 : set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.exp p.1}
def M4 : set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = Real.sin p.1 + 1}

theorem mutuallyOrthogonalSet_is_M4 :
  ¬ mutuallyOrthogonalSet M1 ∧ ¬ mutuallyOrthogonalSet M2 ∧ ¬ mutuallyOrthogonalSet M3 ∧ mutuallyOrthogonalSet M4 :=
by
  sorry

end mutuallyOrthogonalSet_is_M4_l334_334700


namespace price_of_tea_mixture_l334_334902

theorem price_of_tea_mixture 
  (p1 p2 p3 : ℝ) 
  (q1 q2 q3 : ℝ) 
  (h_p1 : p1 = 126) 
  (h_p2 : p2 = 135) 
  (h_p3 : p3 = 173.5) 
  (h_q1 : q1 = 1) 
  (h_q2 : q2 = 1) 
  (h_q3 : q3 = 2) : 
  (p1 * q1 + p2 * q2 + p3 * q3) / (q1 + q2 + q3) = 152 := 
by 
  sorry

end price_of_tea_mixture_l334_334902


namespace cos_angle_AB_AC_l334_334973

def point := (ℝ × ℝ × ℝ)

def vector (p1 p2: point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def dot_product (v1 v2: point) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def magnitude (v: point) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

def cosine_angle (v1 v2: point) : ℝ :=
  dot_product v1 v2 / (magnitude v1 * magnitude v2)

noncomputable def A : point := (6, 2, -3)
noncomputable def B : point := (6, 3, -2)
noncomputable def C : point := (7, 3, -3)

noncomputable def AB : point := vector A B
noncomputable def AC : point := vector A C

theorem cos_angle_AB_AC : cosine_angle AB AC = 1 / 2 :=
by sorry

end cos_angle_AB_AC_l334_334973


namespace find_m_l334_334725

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m
noncomputable def g (x : ℝ) : ℝ := 2 * x - 2

theorem find_m : 
  ∃ m : ℝ, ∀ x : ℝ, f m x = g x → m = -2 := by
  sorry

end find_m_l334_334725


namespace inequality_sine_half_angles_l334_334284

variable {A B C r R : ℝ}

theorem inequality_sine_half_angles (h1: ∠A / 2 > 0) (h2: ∠B / 2 > 0) (h3: ∠C / 2 > 0)
  (circumradius : R = 4 * real.sin (A / 2) * real.sin (B / 2) * real.sin (C / 2) * (1 / r)):
  3 * r / (2 * R) ≤ real.sin (A / 2) + real.sin (B / 2) + real.sin (C / 2) - (real.sin (A / 2))^2 - (real.sin (B / 2))^2 - (real.sin (C / 2))^2 ∧
  real.sin (A / 2) + real.sin (B / 2) + real.sin (C / 2) - (real.sin (A / 2))^2 - (real.sin (B / 2))^2 - (real.sin (C / 2))^2 ≤ 3 / 4 :=
sorry

end inequality_sine_half_angles_l334_334284


namespace greatest_m_eq_377_l334_334125

def h : ℕ → ℕ 
| x := if x = 0 then 0 else (3 ^ (Nat.find_greatest (λ j, 3^j ∣ x) x))

def T_m (m : ℕ) : ℕ := (Finset.range (3^(m-1))).sum (λ k, h (3 * (k + 1)))

theorem greatest_m_eq_377 : 
  ∃ m, m < 500 ∧ (∃ c, c ^ 3 = T_m m) ∧ ∀ n, n < 500 → (∃ c, c ^ 3 = T_m n) → n ≤ m := 
begin 
  use 377,
  split,
  { norm_num, },
  split,
  { use_nat c,
    sorry },
  { intros n hn hc,
    sorry }
end

end greatest_m_eq_377_l334_334125


namespace intersecting_line_at_one_point_l334_334381

theorem intersecting_line_at_one_point (k : ℝ) :
  (∃ y : ℝ, k = -3 * y^2 - 4 * y + 7 ∧ 
           ∀ z : ℝ, k = -3 * z^2 - 4 * z + 7 → y = z) ↔ 
  k = 25 / 3 :=
by
  sorry

end intersecting_line_at_one_point_l334_334381


namespace gabrielle_total_crates_l334_334684

theorem gabrielle_total_crates (monday tuesday wednesday thursday : ℕ)
  (h_monday : monday = 5)
  (h_tuesday : tuesday = 2 * monday)
  (h_wednesday : wednesday = tuesday - 2)
  (h_thursday : thursday = tuesday / 2) :
  monday + tuesday + wednesday + thursday = 28 :=
by
  sorry

end gabrielle_total_crates_l334_334684


namespace friend_balloon_count_l334_334547

theorem friend_balloon_count (you_balloons friend_balloons : ℕ) (h1 : you_balloons = 7) (h2 : you_balloons = friend_balloons + 2) : friend_balloons = 5 :=
by
  sorry

end friend_balloon_count_l334_334547


namespace triangle_sum_proof_l334_334329

def triangle_numbers := {1, 2, 3, 4, 5, 6, 7, 8}

noncomputable def total_possible_sum_S :=
  let S_set := { S | ∃ (a b c d e f g h : ℕ), 
    a ∈ triangle_numbers ∧ b ∈ triangle_numbers ∧ c ∈ triangle_numbers ∧ d ∈ triangle_numbers ∧ 
    e ∈ triangle_numbers ∧ f ∈ triangle_numbers ∧ g ∈ triangle_numbers ∧ h ∈ triangle_numbers ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    a + b + e + h = S ∧
    a + c + f + h = S ∧
    b + d + g + h = S
  } in S_set.sum sorry

theorem triangle_sum_proof : total_possible_sum_S = 67 := sorry

example : total_possible_sum_S = 67 := by
  apply triangle_sum_proof

end triangle_sum_proof_l334_334329


namespace exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero_l334_334088

theorem exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero :
  ∃ f : ℤ[X], f.degree = 2 ∧ ∀ x : ℝ, f.eval (f.eval x) = 0 → x = (Real.sqrt 3) := sorry

end exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero_l334_334088


namespace least_five_digit_perfect_square_and_cube_l334_334455

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334455


namespace least_five_digit_perfect_square_cube_l334_334519

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334519


namespace incenter_divides_angle_bisector_2_1_l334_334701

def is_incenter_divide_angle_bisector (AB BC AC : ℝ) (O : ℝ) : Prop :=
  AB = 15 ∧ BC = 12 ∧ AC = 18 → O = 2 / 1

theorem incenter_divides_angle_bisector_2_1 :
  is_incenter_divide_angle_bisector 15 12 18 (2 / 1) :=
by
  sorry

end incenter_divides_angle_bisector_2_1_l334_334701


namespace bicycle_frame_stability_l334_334357

-- Definitions from the conditions in a)
def bicycle_frame_shape (bicycle: Type) : Prop := ∃ triangle: Type, is_triangle_shape bicycle triangle
def utilizes_stability (design: Type) : Prop := ∃ triangle: Type, is_stable triangle

-- The theorem we need to prove based on the conditions and correct answer
theorem bicycle_frame_stability
  (bicycle : Type)
  (design : Type)
  (h1 : bicycle_frame_shape bicycle)
  (h2 : utilizes_stability design) :
  ∀ b : bicycle, ∀ d : design, reason_bicycle_frame_triangle b d = stability := 
sorry

end bicycle_frame_stability_l334_334357


namespace shape_is_line_l334_334792

def spherical_coords := {ρ : ℝ, θ : ℝ, φ : ℝ}

variables {c d : ℝ}

theorem shape_is_line (c d : ℝ) : 
  (∃ (ρ : ℝ), spherical_coords = {ρ, θ : ℝ := d, φ := c}) := sorry

end shape_is_line_l334_334792


namespace sum_of_solutions_l334_334530

theorem sum_of_solutions : 
  let integer_solutions := { x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 } in
  ∑ x in integer_solutions, x = 24 := 
sorry

end sum_of_solutions_l334_334530


namespace least_five_digit_is_15625_l334_334442

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334442


namespace find_t_range_l334_334733

def f (x : ℝ) (t : ℝ): ℝ :=
  if x < t then -6 + Real.exp (x - 1) else x^2 - 4 * x

theorem find_t_range :
  {t : ℝ | 1 < t ∧ t ≤ 2} → ∃ (x1 x2 x3 : ℝ), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
    f x1 t = x1 - 6 ∧ f x2 t = x2 - 6 ∧ f x3 t = x3 - 6 := sorry

end find_t_range_l334_334733


namespace digit_inequality_l334_334308

theorem digit_inequality : ∀ d: ℕ, d ≤ 9 → ((3.014 + 0.0001 * d) ≤ 3.015) :=
begin
  sorry
end

end digit_inequality_l334_334308


namespace find_pairs_l334_334641

theorem find_pairs (m n : ℕ) : 
  (20^m - 10 * m^2 + 1 = 19^n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2)) :=
by
  sorry

end find_pairs_l334_334641


namespace shaded_fraction_in_square_quilt_l334_334786

theorem shaded_fraction_in_square_quilt :
  let quilt := λ (i j : ℕ), if i = j ∧ i < 4 ∧ j < 4 then 1 else 0
  let shaded := λ (i j : ℕ), if quilt i j = 1 then 0.5 else 0
  let total_shaded := ∑ i in finset.range 3, ∑ j in finset.range 3, shaded i j
  let total_area := 3 * 3
  fraction_shaded = total_shaded / total_area
  fraction_shaded = 1 / 6 :=
by
  let quilt := λ (i j : ℕ), if i = j ∧ i < 3 ∧ j < 3 then 1 else 0
  let shaded := λ (i j : ℕ), if quilt i j = 1 then 0.5 else 0
  have total_shaded : ℝ := (∑ i in finset.range 3, ∑ j in finset.range 3, shaded i j)
  have total_area : ℝ := 3 * 3
  have fraction_shaded : ℝ := total_shaded / total_area
  calc
    fraction_shaded = (3 * 0.5) / 9
                ... = 1.5 / 9
                ... = 1 / 6
  sorry

end shaded_fraction_in_square_quilt_l334_334786


namespace measure_angleA_l334_334070

-- Define the kite ABCD with given conditions
variables {A B C D : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Interrelationships between points
noncomputable def lengths_equal (a b c d: Type) [MetricSpace a] [MetricSpace b] [MetricSpace c] [MetricSpace d]: Prop :=
  dist a b = dist a d ∧ dist c b = dist c d

-- Angle conditions
variables {angleA angleB angleD : ℝ}
variables (angle_condition1 : angleB = 2 * angleD)
variables (angle_condition2 : angleD = 2 * angleA)

-- To prove:
theorem measure_angleA (h1 : lengths_equal A B C D) (hc1 : angle_condition1) (hc2 : angle_condition2) : angleA = 40 :=
  sorry

end measure_angleA_l334_334070


namespace find_side_a_l334_334896

noncomputable def hypotenuse := 17.04 -- hypotenuse c
noncomputable def angle_bisector := 12.96 -- angle bisector t

-- Lean assertion that the side 'a' given the conditions is approximately 11.95.
theorem find_side_a (c : ℝ) (t : ℝ) (a : ℝ) (Hc : c = hypotenuse) (Ht : t = angle_bisector) : 
    a = 11.95 :=
by
  -- replace sorry with actual proof
  sorry

end find_side_a_l334_334896


namespace chess_tournament_scores_l334_334130

theorem chess_tournament_scores :
    ∃ (A D V S : ℝ),
    A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S ∧
    A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 ∧
    A > D ∧ D > V ∧ V > S ∧
    (∃ (wins_A wins_S : ℕ), wins_A = wins_S) :=
begin
    sorry
end

end chess_tournament_scores_l334_334130


namespace right_triangle_area_l334_334785

/-- Given a right triangle with one leg of length 3 and the hypotenuse of length 5,
    the area of the triangle is 6. -/
theorem right_triangle_area (a b c : ℝ) (h₁ : a = 3) (h₂ : c = 5) (h₃ : c^2 = a^2 + b^2) :
  (1 / 2) * a * b = 6 := 
sorry

end right_triangle_area_l334_334785


namespace sum_of_possible_M_l334_334666

theorem sum_of_possible_M (M : ℝ) (h : M * (M - 8) = -8) : M = 4 ∨ M = 4 := 
by sorry

end sum_of_possible_M_l334_334666


namespace sector_angle_l334_334679

noncomputable def cone_height (r : ℝ) (V : ℝ) : ℝ :=
  3 * V / (real.pi * r^2)

noncomputable def slant_height (r : ℝ) (h : ℝ) : ℝ :=
  real.sqrt (r^2 + h^2)

theorem sector_angle (r_paper r_cone V : ℝ) (h : ℝ)
  (cone_radius_eq : r_cone = 14)
  (cone_volume_eq : V = 1232 * real.pi) 
  (paper_radius_eq : r_paper = 18)
  (h_eq : h = cone_height r_cone V):
  let l := slant_height r_cone h,
      circumference_cone := 2 * real.pi * r_cone,
      circumference_paper := 2 * real.pi * r_paper,
      theta := 360 * (circumference_cone / circumference_paper) in
  theta = 280 ∧ (360 - theta = 80) :=
sorry

end sector_angle_l334_334679


namespace number_of_BMWs_sold_l334_334023

theorem number_of_BMWs_sold (total_cars_sold : ℕ)
  (percent_Ford percent_Nissan percent_Chevrolet : ℕ)
  (h_total : total_cars_sold = 300)
  (h_percent_Ford : percent_Ford = 18)
  (h_percent_Nissan : percent_Nissan = 25)
  (h_percent_Chevrolet : percent_Chevrolet = 20) :
  (300 * (100 - (percent_Ford + percent_Nissan + percent_Chevrolet)) / 100) = 111 :=
by
  -- We assert that the calculated number of BMWs is 111
  sorry

end number_of_BMWs_sold_l334_334023


namespace necessary_but_not_sufficient_condition_l334_334358

-- Definitions
def represents_ellipse (m n : ℝ) (x y : ℝ) : Prop := 
  (x^2 / m + y^2 / n = 1)

-- Main theorem statement
theorem necessary_but_not_sufficient_condition 
    (m n x y : ℝ) (h_mn_pos : m * n > 0) :
    (represents_ellipse m n x y) → 
    (m ≠ n ∧ m > 0 ∧ n > 0 ∧ represents_ellipse m n x y) → 
    (m * n > 0) ∧ ¬(
    ∀ m n : ℝ, (m ≠ n ∧ m > 0 ∧ n > 0) →
    represents_ellipse m n x y
    ) :=
by
  sorry

end necessary_but_not_sufficient_condition_l334_334358


namespace purple_tile_cost_correct_l334_334289

-- Definitions of given conditions
def turquoise_cost_per_tile : ℕ := 13
def wall1_area : ℕ := 5 * 8
def wall2_area : ℕ := 7 * 8
def total_area : ℕ := wall1_area + wall2_area
def tiles_per_square_foot : ℕ := 4
def total_tiles_needed : ℕ := total_area * tiles_per_square_foot
def turquoise_total_cost : ℕ := total_tiles_needed * turquoise_cost_per_tile
def savings : ℕ := 768
def purple_total_cost : ℕ := turquoise_total_cost - savings
def purple_cost_per_tile : ℕ := 11

-- Theorem stating the problem
theorem purple_tile_cost_correct :
  purple_total_cost / total_tiles_needed = purple_cost_per_tile :=
sorry

end purple_tile_cost_correct_l334_334289


namespace find_expression_value_l334_334208

variable (a b : ℝ)

theorem find_expression_value (h : a - 2 * b = 7) : 6 - 2 * a + 4 * b = -8 := by
  sorry

end find_expression_value_l334_334208


namespace number_of_ways_to_choose_a_pair_of_socks_l334_334779

-- Define the number of socks of each color
def white_socks := 5
def brown_socks := 5
def blue_socks := 5
def green_socks := 5

-- Define the total number of socks
def total_socks := white_socks + brown_socks + blue_socks + green_socks

-- Define the number of ways to choose 2 blue socks from 5 blue socks
def num_ways_choose_two_blue_socks : ℕ := Nat.choose blue_socks 2

-- The proof statement
theorem number_of_ways_to_choose_a_pair_of_socks :
  num_ways_choose_two_blue_socks = 10 :=
sorry

end number_of_ways_to_choose_a_pair_of_socks_l334_334779


namespace least_five_digit_perfect_square_and_cube_l334_334482

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334482


namespace least_five_digit_perfect_square_and_cube_l334_334468

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334468


namespace least_five_digit_perfect_square_cube_l334_334516

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334516


namespace least_five_digit_perfect_square_and_cube_l334_334467

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334467


namespace chess_tournament_points_distribution_l334_334177

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l334_334177


namespace length_of_side_AB_correct_l334_334282

open Real

noncomputable def length_of_side_AB (A B C : Point) (h : ∠ A B C = π / 2) (BC : ℝ) (tan_cond : tan (angle B C) = 3 * sin (angle B C)) : ℝ :=
  sqrt (BC^2 - (BC/3)^2)

theorem length_of_side_AB_correct (A B C : Point) (h : ∠ A B C = π / 2) (BC : ℝ) (tan_cond : tan (angle B C) = 3 * sin (angle B C)) :
  length_of_side_AB A B C h BC tan_cond = 20 * sqrt 2 / 3 :=
by
  sorry

end length_of_side_AB_correct_l334_334282


namespace chess_tournament_solution_l334_334165

def chess_tournament_points (points : List ℝ) : Prop :=
  let andrey := points[0]
  let dima := points[1]
  let vanya := points[2]
  let sasha := points[3]
  andrey = 4 ∧ dima = 3.5 ∧ vanya = 2.5 ∧ sasha = 2

axiom chess_tournament_conditions (points : List ℝ) :
  -- Andrey secured first place, Dima secured second, Vanya secured third, and Sasha secured fourth.
  List.Nodup points ∧
  points.length = 4 ∧
  (∀ p, p ∈ points → p = 4 ∨ p = 3.5 ∨ p = 2.5 ∨ p = 2) ∧
  -- Andrey and Sasha won the same number of games.
  (points[0] ≠ points[1] ∧ points[0] ≠ points[2] ∧ points[0] ≠ points[3] ∧
   points[1] ≠ points[2] ∧ points[1] ≠ points[3] ∧
   points[2] ≠ points[3])

theorem chess_tournament_solution (points : List ℝ) :
  chess_tournament_conditions points → chess_tournament_points points :=
by
  sorry

end chess_tournament_solution_l334_334165


namespace coco_hours_used_l334_334914

noncomputable def electricity_price : ℝ := 0.10
noncomputable def consumption_rate : ℝ := 2.4
noncomputable def total_cost : ℝ := 6.0

theorem coco_hours_used (hours_used : ℝ) : hours_used = total_cost / (consumption_rate * electricity_price) :=
by
  sorry

end coco_hours_used_l334_334914


namespace find_equation_line_AC_max_area_triangle_OPQ_l334_334707

-- Definition of the circle B
def circle_B := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 2 }

-- Definitions of the two lines l1 and l2 passing through origin O(0,0)
def line_through_origin (m : ℝ) := { p : ℝ × ℝ | p.2 = m * p.1 } -- y = mx form

-- Perpendicular lines from point B to l1 and l2 intersecting at points A and C
def perpendicular_foot (B A : ℝ × ℝ) (l : ℝ × ℝ → Prop) := l (A.1, A.2) ∧ (A.1 - B.1) * (2 * B.2 - A.2) = (A.2 - B.2) * (2 * B.1 - A.1)

-- Proof statement 1: Finding the equation of line AC
theorem find_equation_line_AC (A C B : ℝ × ℝ) (l1 l2 : ℝ× ℝ → Prop) :
  (∀ p, circle_B p → l1 p ∨ l1 p) →
  perpendicular_foot B A l1 ∧
  perpendicular_foot B C l2 ∧
  (B.1 - A.1) * (B.2 - C.2) + (B.1 - C.1) * (B.2 - A.2) = 0 ∧
  dist B A = dist B C →
  ∃ k : ℝ, ∀ p : ℝ × ℝ, (p.1 + p.2 - 1 = 0) := sorry

-- Proof statement 2: Finding maximum area of triangle OPQ
theorem max_area_triangle_OPQ (P Q O B : ℝ × ℝ) (l1 l2 : ℝ → Prop) :
  O = (0, 0) ∧ circle_B B →
  (P.2 = 0 ∨ Q.2 = 0) →
  l1 ⟨P.1, P.2⟩ ∧ l2 ⟨Q.1, Q.2⟩ →
  l1 P ∧ l2 Q ∧
  (P.1 - O.1) * (Q.2 - O.2) - (Q.1 - O.1) * (P.2 - O.2) = sqrt(2) →
  ∃ area_max : ℝ, area_max = 2 := 
  sorry

end find_equation_line_AC_max_area_triangle_OPQ_l334_334707


namespace vitya_knows_answers_29_attempts_vitya_knows_answers_24_attempts_l334_334007

/-- The test consists of 30 questions, each with two possible answers (one correct and one incorrect). 
    Vitya can proceed in such a way that he can guarantee to know all the correct answers no later than:
    (a) after the 29th attempt (and answer all questions correctly on the 30th attempt)
    (b) after the 24th attempt (and answer all questions correctly on the 25th attempt)
    - Vitya initially does not know any of the answers.
    - The test is always the same.
-/
def vitya_test (k : Nat) : Prop :=
  k = 30 ∧ (∀ (attempts : Fin 30 → Bool), attempts 30 = attempts 29 ∧ attempts 30)

theorem vitya_knows_answers_29_attempts :
  vitya_test 30 :=
by 
  sorry

theorem vitya_knows_answers_24_attempts :
  vitya_test 25 :=
by 
  sorry

end vitya_knows_answers_29_attempts_vitya_knows_answers_24_attempts_l334_334007


namespace least_five_digit_perfect_square_and_cube_l334_334464

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334464


namespace binary_111011_to_decimal_is_59_l334_334360

-- Define the binary number as a sequence of digits
def binaryNumber : List Nat := [1, 1, 1, 0, 1, 1]

-- Define the function that converts a binary number (represented as a list of digits) to a decimal number
def binaryToDecimal (bin : List Nat) : Nat :=
  bin.foldr (λ (digit power: Nat), digit * 2^power + power) 0 (List.length bin - 1)

-- State the theorem
theorem binary_111011_to_decimal_is_59 : binaryToDecimal binaryNumber = 59 :=
by
  sorry

end binary_111011_to_decimal_is_59_l334_334360


namespace hyperbola_eccentricity_l334_334912

theorem hyperbola_eccentricity (a b : ℝ) (h1 : b ≠ 0) (h_condition : (a*b) / (Real.sqrt (a^2 + b^2)) = (1/4) * (2*b)) :
  let c := Real.sqrt (a^2 + b^2),
  let e := c / a,
  e = 2 :=
by 
  sorry

end hyperbola_eccentricity_l334_334912


namespace exists_quadratic_poly_with_integer_coeffs_l334_334097

theorem exists_quadratic_poly_with_integer_coeffs (α : ℝ) :
  (∃ (a b c : ℤ), ∀ x : ℝ, (λ x : ℝ, ↑a * x^2 + ↑b * x + ↑c) (↑(λ x : ℝ, ↑a * x^2 + ↑b * x + ↑c) (α)) = 0) :=
  sorry

end exists_quadratic_poly_with_integer_coeffs_l334_334097


namespace age_difference_l334_334945

variable (A B C : ℕ)

-- Conditions
def condition1 : Prop := A + B > B + C
def condition2 : Prop := C = A - 18

theorem age_difference (h1 : condition1 A B C) (h2 : condition2 A B C) : (A + B) - (B + C) = 18 :=
by
  sorry

end age_difference_l334_334945


namespace unique_intersection_point_l334_334378

theorem unique_intersection_point (k : ℝ) :
x = k ->
∃ x : ℝ, x = -3*y^2 - 4*y + 7 -> ∃ k : ℝ, k = 25/3 -> y = 0 -> x = k

end unique_intersection_point_l334_334378


namespace least_five_digit_perfect_square_and_cube_l334_334481

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334481


namespace polar_to_cartesian_eq_traj_l334_334105

theorem polar_to_cartesian_eq_traj
  (center_polar : ℝ × ℝ := (2, π / 3))
  (r : ℝ := 1)
  (p : ℝ × ℝ)
  (midpoint_Q : ℝ × ℝ) :
  (p.2 = center_polar.2 + θ →
  p.1 ^ 2 - 4 * p.1 * cos (θ - center_polar.2) + 3 = 0 →
  ∃ x y, midpoint_Q = (x, y) ∧ (x - 1 / 2) ^ 2 + (y - sqrt 3 / 2) ^ 2 = 1 / 4) :=
by
  sorry

end polar_to_cartesian_eq_traj_l334_334105


namespace num_congruent_mod_7_count_mod_7_eq_22_l334_334246

theorem num_congruent_mod_7 (n : ℕ) :
  (1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 1) → ∃ k, 0 ≤ k ∧ k ≤ 21 ∧ n = 7 * k + 1 :=
sorry

theorem count_mod_7_eq_22 : 
  (∃ n_set : Finset ℕ, 
    (∀ n ∈ n_set, 1 ≤ n ∧ n ≤ 150 ∧ n % 7 = 1) ∧ 
    Finset.card n_set = 22) :=
sorry

end num_congruent_mod_7_count_mod_7_eq_22_l334_334246


namespace sum_of_first_five_terms_geo_seq_l334_334669

theorem sum_of_first_five_terms_geo_seq 
  (a : ℚ) (r : ℚ) (n : ℕ) 
  (h_a : a = 1 / 3)
  (h_r : r = 1 / 3)
  (h_n : n = 5) :
  (∑ i in Finset.range n, a * r^i) = 121 / 243 :=
by
  sorry

end sum_of_first_five_terms_geo_seq_l334_334669


namespace chess_tournament_solution_l334_334167

def chess_tournament_points (points : List ℝ) : Prop :=
  let andrey := points[0]
  let dima := points[1]
  let vanya := points[2]
  let sasha := points[3]
  andrey = 4 ∧ dima = 3.5 ∧ vanya = 2.5 ∧ sasha = 2

axiom chess_tournament_conditions (points : List ℝ) :
  -- Andrey secured first place, Dima secured second, Vanya secured third, and Sasha secured fourth.
  List.Nodup points ∧
  points.length = 4 ∧
  (∀ p, p ∈ points → p = 4 ∨ p = 3.5 ∨ p = 2.5 ∨ p = 2) ∧
  -- Andrey and Sasha won the same number of games.
  (points[0] ≠ points[1] ∧ points[0] ≠ points[2] ∧ points[0] ≠ points[3] ∧
   points[1] ≠ points[2] ∧ points[1] ≠ points[3] ∧
   points[2] ≠ points[3])

theorem chess_tournament_solution (points : List ℝ) :
  chess_tournament_conditions points → chess_tournament_points points :=
by
  sorry

end chess_tournament_solution_l334_334167


namespace find_k_intersects_parabola_at_one_point_l334_334374

theorem find_k_intersects_parabola_at_one_point :
  ∃ k : ℝ, (∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k ↔ y = (-4 / (2 * 3))) →
    k = 25 / 3 :=
by sorry

end find_k_intersects_parabola_at_one_point_l334_334374


namespace stratified_sampling_males_l334_334579

theorem stratified_sampling_males
  (total_male_students : ℕ)
  (total_female_students : ℕ)
  (sample_size : ℕ)
  (sampling_ratio : ℚ)
  (male_sample : ℕ) :
  total_male_students = 400 ∧ total_female_students = 300 ∧ sample_size = 35 →
  sampling_ratio = (sample_size : ℚ) / (total_male_students + total_female_students) →
  male_sample = total_male_students * sampling_ratio →
  male_sample = 20 :=
by {
  intros h ht,
  obtain ⟨h1, h2, h3⟩ := h,
  rw h1 at ht,
  rw h2 at ht,
  rw h3 at ht,
  norm_num,
  exact ht,
  sorry
}

end stratified_sampling_males_l334_334579


namespace chess_tournament_points_l334_334144

theorem chess_tournament_points
  (points : String → ℝ)
  (total_points : points "Andrey" + points "Dima" + points "Vanya" + points "Sasha" = 12)
  (distinct_points : 
    points "Andrey" ≠ points "Dima" ∧ 
    points "Andrey" ≠ points "Vanya" ∧ 
    points "Andrey" ≠ points "Sasha" ∧ 
    points "Dima" ≠ points "Vanya" ∧ 
    points "Dima" ≠ points "Sasha" ∧ 
    points "Vanya" ≠ points "Sasha")
  (order : 
    points "Andrey" > points "Dima" ∧ 
    points "Dima" > points "Vanya" ∧ 
    points "Vanya" > points "Sasha")
  (same_wins :
    let games_won (student : String) := (points student - 3) / 0.5 in
    games_won "Andrey" = games_won "Sasha") :
  points "Andrey" = 4 ∧ points "Dima" = 3.5 ∧ points "Vanya" = 2.5 ∧ points "Sasha" = 2 :=
by
  sorry

end chess_tournament_points_l334_334144


namespace exists_quadratic_polynomial_with_given_property_l334_334100

theorem exists_quadratic_polynomial_with_given_property :
  ∃ f : ℚ[X], degree f = 2 ∧ (∀ n : ℤ, (n : ℚ) ∈ f.coeff) ∧ f.eval (f.eval (real.sqrt 3)) = 0 :=
by
  sorry

end exists_quadratic_polynomial_with_given_property_l334_334100


namespace david_presents_l334_334637

variables (C B E : ℕ)

def total_presents (C B E : ℕ) : ℕ := C + B + E

theorem david_presents :
  C = 60 →
  B = 3 * E →
  E = (C / 2) - 10 →
  total_presents C B E = 140 :=
by
  intros hC hB hE
  sorry

end david_presents_l334_334637


namespace functional_relationship_and_range_maximum_monthly_profit_range_of_additional_costs_l334_334359

variable (x : ℕ) (a : ℝ)

/-- The functional relationship between y and x -/
def y (x : ℕ) : ℕ := 210 - 10 * x

/-- The cost price of a unit -/
def cost_price : ℕ := 40

/-- The initial selling price -/
def initial_selling_price : ℕ := 50

/-- The maximum selling price -/
def max_selling_price : ℕ := 65

/-- The profit function given an increase in selling price by x and additional cost a -/
def profit (x : ℕ) (a : ℝ) : ℝ :=
  (y x) * (initial_selling_price + x - cost_price - a)

/-- The functional relationship and range of x -/
theorem functional_relationship_and_range :
  (y x = 210 - 10 * x) ∧ (0 < x ∧ x ≤ 15) := by
  sorry

/-- The selling price and maximum monthly profit -/
theorem maximum_monthly_profit :
  ∃ (p : ℕ), p ∈ {55, 56} ∧ max_profit = 2400 where max_profit := profit p 0 := by
  sorry

/-- The range of additional costs a -/
theorem range_of_additional_costs (x : ℕ) :
  (x ≥ 8 ∧ profit x a < profit x 0) → (0 < a ∧ a < 6) := by
  sorry

end functional_relationship_and_range_maximum_monthly_profit_range_of_additional_costs_l334_334359


namespace least_five_digit_perfect_square_and_cube_l334_334451

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334451


namespace evaluate_expression_at_two_l334_334798

theorem evaluate_expression_at_two :
  (let x := ((2 - 1) / (2 + 1)) in
   (x + 1) / (x - 1)) = -2 :=
by
  -- Substitute x with (2 - 1) / (2 + 1)
  let x := (2 - 1) / (2 + 1)
  -- Evaluate (x + 1) / (x - 1)
  have h1 : x = 1 / 3 := by sorry
  have h2 : (x + 1) = 1/3 + 1 := by sorry
  have h3 : (x - 1) = 1/3 - 1 := by sorry
  conclude (1/3 + 1) / (1/3 - 1) = -2 := by sorry
  sorry

end evaluate_expression_at_two_l334_334798


namespace paint_16_seats_l334_334269

def is_odd (n : ℕ) : Prop := ¬ even n

def valid_painting : list ℕ → Prop
| []          := true
| [x]         := true
| (x :: y :: rest) := (is_odd (length (take_while (eq x) (y :: rest)))) 
                      ∧ valid_painting (drop_while (eq x) (y :: rest))

def seat_paint_count : ℕ → ℤ
| 0 := 1
| 1 := 1
| n := (seat_paint_count (n - 1)) + (seat_paint_count (n - 2))

theorem paint_16_seats : ∑ (valid_painting (list.range 16)) = 1974 :=
by {
  sorry
}

end paint_16_seats_l334_334269


namespace chess_tournament_scores_l334_334154

def points (name : String) := Real

def total_points : Real := 12

variables (A D V S : Real)
variable (total_games : ℕ := 12)

axiom different_scores : A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S

axiom ranking : A > D ∧ D > V ∧ V > S

axiom equal_wins (A S : Real) : (A = 2 * win_points) ∧ (S = 2 * win_points)

axiom total_points_constraint : A + D + V + S = total_points

theorem chess_tournament_scores :
  A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 :=
by 
  sorry

end chess_tournament_scores_l334_334154


namespace tire_usage_l334_334365

theorem tire_usage (miles_traveled : ℕ) (num_tires : ℕ) (tire_miles : ℕ) (tire_spare : ℕ) :
  miles_traveled = 30000 ∧ num_tires = 4 ∧ (miles_traveled * num_tires = tire_miles) ∧ tire_spare = 5 -> 
  (tire_miles / tire_spare = 24000) :=
by
  intro h
  cases h with h_miles_traveled h_rest
  cases h_rest with h_num_tires h_rest'
  cases h_rest' with h_tire_miles h_tire_spare
  sorry

end tire_usage_l334_334365


namespace solve_arccos_eq_l334_334344

theorem solve_arccos_eq (x : ℝ) : 
    (arccos (3 * x) - arccos (2 * x) = π / 6) ↔ 
    (x = 1 / (2 * sqrt (12 - 6 * sqrt 3)) ∨ x = -1 / (2 * sqrt (12 - 6 * sqrt 3))) := 
by 
  sorry

end solve_arccos_eq_l334_334344


namespace probability_even_sum_of_six_primes_l334_334343

open Nat

/-- The first 12 prime numbers are listed here for reference -/
def first_twelve_primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

/-- Define the main problem -/
theorem probability_even_sum_of_six_primes :
  let primes_six_combo : Finset (Finset ℕ) := (Finset.powerset_len 6 first_twelve_primes.to_finset),
  let even_sum_count := primes_six_combo.filter (λ s, s.sum % 2 = 0),
  even_sum_count.card / primes_six_combo.card = (1 : ℚ) / 2 :=
by
  sorry

end probability_even_sum_of_six_primes_l334_334343


namespace partition_exists_l334_334852

def partition_colored_sets (n : ℕ) (h : n > 3) : Prop :=
  let k := ⌊(1/6 : ℚ) * n * (n+1)⌋ in
  let total_elements := (1/2 : ℚ) * n * (n+1) in
  let blue := k in
  let red := k in
  let white := total_elements - (blue + red) in
  ∃ (A : fin n → set ℕ), (∀ m : fin n, #A m = m ∧ (∀ x y ∈ A m, x ≠ y → (x - y) % 3 = 0))

theorem partition_exists (n : ℕ) (h : n > 3) : partition_colored_sets n h :=
by sorry

end partition_exists_l334_334852


namespace three_pumps_drain_time_l334_334878

-- Definitions of the rates of each pump
def rate1 := 1 / 9
def rate2 := 1 / 6
def rate3 := 1 / 12

-- Combined rate of all three pumps working together
def combined_rate := rate1 + rate2 + rate3

-- Time to drain the lake with all three pumps working together
def time_to_drain := 1 / combined_rate

-- Theorem: The time it takes for three pumps working together to drain the lake is 36/13 hours
theorem three_pumps_drain_time : time_to_drain = 36 / 13 := by
  sorry

end three_pumps_drain_time_l334_334878


namespace sum_of_integer_solutions_l334_334542

theorem sum_of_integer_solutions :
  (∑ x in ({ x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 }.to_finset), x) = 18 :=
by
  sorry

end sum_of_integer_solutions_l334_334542


namespace geometric_to_arithmetic_l334_334204

theorem geometric_to_arithmetic {a1 a2 a3 a4 q : ℝ}
  (hq : q ≠ 1)
  (geom_seq : a2 = a1 * q ∧ a3 = a1 * q^2 ∧ a4 = a1 * q^3)
  (arith_seq : (2 * a3 = a1 + a4 ∨ 2 * a2 = a1 + a4)) :
  q = (1 + Real.sqrt 5) / 2 ∨ q = (-1 + Real.sqrt 5) / 2 :=
by
  sorry

end geometric_to_arithmetic_l334_334204


namespace function_machine_output_l334_334278

def modified_function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3 in
  let step2 := if step1 ≤ 38 then step1 * 2 else step1 - 10 in
  step2

theorem function_machine_output :
  modified_function_machine 12 = 72 :=
by
  sorry

end function_machine_output_l334_334278


namespace least_number_remainder_seven_exists_l334_334524

theorem least_number_remainder_seven_exists :
  ∃ x : ℕ, x ≡ 7 [MOD 11] ∧ x ≡ 7 [MOD 17] ∧ x ≡ 7 [MOD 21] ∧ x ≡ 7 [MOD 29] ∧ x ≡ 7 [MOD 35] ∧ 
           x ≡ 1547 [MOD Nat.lcm 11 (Nat.lcm 17 (Nat.lcm 21 (Nat.lcm 29 35)))] :=
  sorry

end least_number_remainder_seven_exists_l334_334524


namespace red_ball_higher_than_others_l334_334415

noncomputable def probability_red_higher_than_others : ℚ :=
  (1 / 3) * (1 - ∑' k : ℕ, (2 : ℚ)^(-(k + 1) * 3))

theorem red_ball_higher_than_others :
  probability_red_higher_than_others = 2 / 7 :=
by
  sorry

end red_ball_higher_than_others_l334_334415


namespace least_five_digit_perfect_square_and_cube_l334_334461

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334461


namespace arithmetic_sequence_formula_l334_334795

theorem arithmetic_sequence_formula :
  ∀ (a : ℕ → ℕ), (a 1 = 2) → (∀ n, a (n + 1) = a n + 2) → ∀ n, a n = 2 * n :=
by
  intro a
  intro h1
  intro hdiff
  sorry

end arithmetic_sequence_formula_l334_334795


namespace least_clock_equivalent_hour_l334_334876

theorem least_clock_equivalent_hour (h : ℕ) (h_gt_9 : h > 9) (clock_equiv : (h^2 - h) % 12 = 0) : h = 13 :=
sorry

end least_clock_equivalent_hour_l334_334876


namespace sum_incorrect_correct_l334_334982

theorem sum_incorrect_correct (x : ℕ) (h : x + 9 = 39) :
  ((x - 5 + 14) + (x * 5 + 14)) = 203 :=
sorry

end sum_incorrect_correct_l334_334982


namespace shorter_side_length_l334_334002

theorem shorter_side_length (L W : ℝ) (h₁ : L * W = 120) (h₂ : 2 * L + 2 * W = 46) : L = 8 ∨ W = 8 := 
by 
  sorry

end shorter_side_length_l334_334002


namespace line_points_k_l334_334026

noncomputable def k : ℝ := 8

theorem line_points_k (k : ℝ) : 
  (∀ k : ℝ, ∃ b : ℝ, b = (10 - k) / (5 - 5) ∧
  ∀ b, b = (-k) / (20 - 5) → k = 8) :=
  by
  sorry

end line_points_k_l334_334026


namespace max_candy_received_l334_334018

theorem max_candy_received (students : ℕ) (candies : ℕ) (min_candy_per_student : ℕ) 
    (h_students : students = 40) (h_candies : candies = 200) (h_min_candy : min_candy_per_student = 2) :
    ∃ max_candy : ℕ, max_candy = 122 := by
  sorry

end max_candy_received_l334_334018


namespace seats_shortage_l334_334984

-- Definitions of the conditions
def children := 52
def adults := 29
def seniors := 15
def pets := 3
def total_seats := 95

-- Theorem statement to prove the number of people and pets without seats
theorem seats_shortage : children + adults + seniors + pets - total_seats = 4 :=
by
  sorry

end seats_shortage_l334_334984


namespace problem_statement_l334_334845

def t (x : ℝ) : ℝ := 4 * x - 9
def s (y : ℝ) : ℝ := (Real.sqrtSq' (1 / 4 * (y + 9)))^2 + 4 * (Real.sqrtSq' (1 / 4 * (y + 9))) - 5

theorem problem_statement : s 1 = 11.25 := by
  sorry

end problem_statement_l334_334845


namespace student_queue_length_l334_334601

-- Theorem statement equivalent to the problem.
theorem student_queue_length (queue_speed student_speed : ℕ) (time_minutes : ℝ)
  (h1 : queue_speed = 8) (h2 : student_speed = 12) 
  (h3 : time_minutes = 7.2) : 
  let x := (12 - 8 : ℝ) in
  let y := (12 + 8 : ℝ) in
  let time_hours := time_minutes / 60 in
  (x / queue_speed + x / student_speed) * x = 400 :=
by
  sorry

end student_queue_length_l334_334601


namespace division_of_decimals_l334_334065

theorem division_of_decimals : 0.18 / 0.003 = 60 :=
by
  sorry

end division_of_decimals_l334_334065


namespace tangents_intersect_on_AC_l334_334196

-- Let A, B, C be points forming a scalene triangle ABC
variables (A B C O I B' : Point)
variable [metric_space Point]

-- Definitions and conditions
def is_scalene_triangle := ¬collinear A B C ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A
def is_circumcenter_of_O := is_circumcenter O A B C
def is_incenter_of_I := is_incenter I A B C
def is_reflection_of_B_across_OI := B'.reflect O I = B

-- The reflection of B should be inside angle ABI
def lies_inside_angle_ABI := angle_inside B A I B'

-- Prove that the intersections of the tangents to the circumcircle of triangle BIB' passing through I and B' lie on line AC
theorem tangents_intersect_on_AC
  (h_scalene : is_scalene_triangle A B C)
  (h_circumcenter : is_circumcenter_of_O A B C O)
  (h_incenter : is_incenter_of_I A B C I)
  (h_reflection : is_reflection_of_B_across_OI B O I B')
  (h_inside_angle : lies_inside_angle_ABI B A I B') :
  ∃ P, tangent_point (circumcircle B I B') I P ∧ tangent_point (circumcircle B I B') B' P ∧ lies_on_line P A C :=
sorry

end tangents_intersect_on_AC_l334_334196


namespace nth_equation_l334_334324

theorem nth_equation (n : ℕ) (hn : n ≠ 0) : 
  (↑n + 2) / ↑n - 2 / (↑n + 2) = ((↑n + 2)^2 + ↑n^2) / (↑n * (↑n + 2)) - 1 :=
by
  sorry

end nth_equation_l334_334324


namespace least_five_digit_perfect_square_and_cube_l334_334510

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334510


namespace ratio_surface_side_area_l334_334574

-- Conditions
variables (r h : ℝ) (π : ℝ := Real.pi)
def is_square_unfolding (r h : ℝ) : Prop := 2 * π * r = h

-- Theorem to prove the ratio of surface area to side area given the above condition
theorem ratio_surface_side_area (r h : ℝ) (h1 : is_square_unfolding r h) :
  (2 * π * r^2 + 2 * π * r * h) / (2 * π * r * h) = (1 + 2 * π) / (2 * π) :=
by
  sorry

end ratio_surface_side_area_l334_334574


namespace max_value_of_expression_l334_334837

-- We have three nonnegative real numbers a, b, and c,
-- such that a + b + c = 3.
def nonnegative (x : ℝ) := x ≥ 0

theorem max_value_of_expression (a b c : ℝ) (h1 : nonnegative a) (h2 : nonnegative b) (h3 : nonnegative c) (h4 : a + b + c = 3) :
  a + b^2 + c^4 ≤ 3 :=
  sorry

end max_value_of_expression_l334_334837


namespace intersections_in_parallelogram_l334_334696

noncomputable def locus_of_intersections
  (ABCD : Type)
  [quadrilateral ABCD]
  (K L M N : ABCD → ABCD)
  (a b : ℝ) : Set (ABCD × ABCD) :=
  {p | ∃ (K L M N : ABCD), K ∈ AB ∧ L ∈ BC ∧ M ∈ CD ∧ N ∈ AD ∧
          (dist K B = a) ∧ (dist L B = a) ∧ (dist M D = b) ∧ (dist N D = b) ∧
          (∃ x ∈ interior (parallelogram ABCD), x = intersect KL MN)}

theorem intersections_in_parallelogram
  (ABCD : Type)
  [quadrilateral ABCD]
  (K L M N : ABCD → ABCD) 
  (a b : ℝ) :
  locus_of_intersections ABCD K L M N a b = 
  {p | ∃ x ∈ interior (parallelogram (abcd K L M N a b)), 
        x = intersect (line_through K L) (line_through M N)} :=
sorry

end intersections_in_parallelogram_l334_334696


namespace three_combinations_without_equilateral_triangle_l334_334342

noncomputable def total_combinations (n k : ℕ) : ℕ := Nat.choose n k
def six_points_combinations : ℕ := total_combinations 6 3

def equilateral_combinations : ℕ := 2

theorem three_combinations_without_equilateral_triangle : 
  six_points_combinations - equilateral_combinations = 18 := 
by 
  sorry

end three_combinations_without_equilateral_triangle_l334_334342


namespace intersecting_line_at_one_point_l334_334380

theorem intersecting_line_at_one_point (k : ℝ) :
  (∃ y : ℝ, k = -3 * y^2 - 4 * y + 7 ∧ 
           ∀ z : ℝ, k = -3 * z^2 - 4 * z + 7 → y = z) ↔ 
  k = 25 / 3 :=
by
  sorry

end intersecting_line_at_one_point_l334_334380


namespace linear_function_properties_l334_334644

def monotonicity_and_parity (a b : ℝ) : Prop :=
  (if a > 0 then ∀ x1 x2 : ℝ, x1 < x2 → ax1 + b < a * x2 + b
   else if a < 0 then ∀ x1 x2 : ℝ, x1 < x2 → ax1 + b > a * x2 + b
   else ∀ x1 x2 : ℝ, ax1 + b = ax2 + b)
  ∧ (¬(∀ x : ℝ, y (-x) = y x) ∨ ¬(∀ x : ℝ, y (-x) = -y x))

theorem linear_function_properties (a b : ℝ) : monotonicity_and_parity a b :=
sorry

end linear_function_properties_l334_334644


namespace sum_of_products_eq_neg_one_l334_334398

-- Given the set M
def M := {4, 3, -1, 0, 1}

-- Let Mi be the non-empty subsets of M (lean will implicitly understand this within the context of the proof)
def nonEmptySubsets (s : Set ℤ) : Set (Set ℤ) :=
  {x | x ⊆ s ∧ x ≠ ∅}

-- Let mi be the product of elements in Mi
def product (s : Set ℤ) : ℤ :=
  s.toList.prod

-- Define the sum of products over all non-empty subsets
def sumOfProducts (s : Set ℤ) : ℤ :=
  ∑ x in nonEmptySubsets s, product x

theorem sum_of_products_eq_neg_one (s : Set ℤ) (h : s = {4, 3, -1, 0, 1}) : sumOfProducts s = -1 :=
by
  subst h
  sorry

end sum_of_products_eq_neg_one_l334_334398


namespace numerator_multiple_of_prime_l334_334391

theorem numerator_multiple_of_prime (n : ℕ) (hp : Nat.Prime (3 * n + 1)) :
  (2 * n - 1) % (3 * n + 1) = 0 :=
sorry

end numerator_multiple_of_prime_l334_334391


namespace curve_is_circle_l334_334657

theorem curve_is_circle (θ : ℝ) :
  let r := 2 / (Real.sin θ + Real.cos θ) in
  ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ x^2 + y^2 = 2 :=
by
  sorry

end curve_is_circle_l334_334657


namespace minimize_J_l334_334250

noncomputable def H (p q : ℝ) : ℝ :=
  -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ :=
  real.Sup (set.image (H p) (set.Icc (0: ℝ) (1: ℝ)))

theorem minimize_J : ∃ p, 0 ≤ p ∧ p ≤ 1 ∧ (∀ x, 0 ≤ x ∧ x ≤ 1 → J(p) ≤ J(x)) ∧ p = 9 / 16 := by
  sorry

end minimize_J_l334_334250


namespace exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero_l334_334087

theorem exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero :
  ∃ f : ℤ[X], f.degree = 2 ∧ ∀ x : ℝ, f.eval (f.eval x) = 0 → x = (Real.sqrt 3) := sorry

end exists_quadratic_poly_f_has_integer_coeffs_and_f_of_f_sqrt3_eq_zero_l334_334087


namespace least_five_digit_perfect_square_and_cube_l334_334478

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334478


namespace jordan_rectangle_length_l334_334625

theorem jordan_rectangle_length (lengthC widthC widthJ : ℝ)
  (hC : lengthC = 15) (wC : widthC = 24) (wJ : widthJ = 45) 
  (equal_area : lengthC * widthC = widthJ * 8) : widthJ = 45 → 8 = 360 / 45 :=
by
  -- provided conditions
  have hC_mul_wC : 15 * 24 = 360, by norm_num
  have wJ_eq : 45 = 45, by norm_num
  -- calculation 
  calc
    8 = 360 / 45 : by norm_num -- sorry

end jordan_rectangle_length_l334_334625


namespace stones_in_10th_image_l334_334617

-- Conditions
def stones_in_image (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 5
  | 3 => 12
  | 4 => 22
  | k => stones_in_image (k - 1) + (3 * k - 2)

-- Proof problem
theorem stones_in_10th_image : stones_in_image 10 = 145 := 
by 
  sorry

end stones_in_10th_image_l334_334617


namespace product_of_last_two_digits_l334_334527

theorem product_of_last_two_digits (A B : ℕ) (h1 : B = 0 ∨ B = 5) (h2 : A + B = 12) : A * B = 35 :=
by {
  -- proof omitted
  sorry
}

end product_of_last_two_digits_l334_334527


namespace cost_per_metre_is_26_point_5_l334_334921

-- Define the conditions
def length_is_b_plus_20 (b l : ℕ) : Prop := l = b + 20
def length_is_60 (l : ℕ) : Prop := l = 60
def total_cost_is_5300 (C : ℕ) : Prop := C = 5300

-- Define the problem statement
theorem cost_per_metre_is_26_point_5 (b l C : ℕ) 
  (h1 : length_is_b_plus_20 b l) 
  (h2 : length_is_60 l) 
  (h3 : total_cost_is_5300 C) :
  C / (2 * l + 2 * (l - 20)) = 26.5 := 
by
  sorry

end cost_per_metre_is_26_point_5_l334_334921


namespace exists_primes_sum_2024_with_one_gt_1000_l334_334071

open Nat

-- Definition of primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Conditions given in the problem
def sum_primes_eq_2024 (p q : ℕ) : Prop :=
  p + q = 2024 ∧ is_prime p ∧ is_prime q

def at_least_one_gt_1000 (p q : ℕ) : Prop :=
  p > 1000 ∨ q > 1000

-- The theorem to be proved
theorem exists_primes_sum_2024_with_one_gt_1000 :
  ∃ (p q : ℕ), sum_primes_eq_2024 p q ∧ at_least_one_gt_1000 p q :=
sorry

end exists_primes_sum_2024_with_one_gt_1000_l334_334071


namespace least_five_digit_perfect_square_and_cube_l334_334466

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334466


namespace statement_a_statement_b_statement_c_l334_334767

theorem statement_a (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  0 ≤ a ∧ a ≤ 4 := sorry

theorem statement_b (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  -1 ≤ b ∧ b ≤ 3 := sorry

theorem statement_c (a b : ℝ) (h1 : 1 ≤ a + b ∧ a + b ≤ 5) (h2 : -1 ≤ a - b ∧ a - b ≤ 3) :
  -2 ≤ 3 * a - 2 * b ∧ 3 * a - 2 * b ≤ 10 := sorry

end statement_a_statement_b_statement_c_l334_334767


namespace find_m_l334_334726

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m
noncomputable def g (x : ℝ) : ℝ := 2 * x - 2

theorem find_m : 
  ∃ m : ℝ, ∀ x : ℝ, f m x = g x → m = -2 := by
  sorry

end find_m_l334_334726


namespace union_of_sets_l334_334238

variable {α : Type*} 

-- Defining the sets A and B with the given constraints
def A (a : α) : Set α := {-2, a}
def B (a b : α) : Set α := {2015^a, b}

-- Given the conditions in the problem
variable {a b l : α}
variable (h1 : A a ∩ B a b = {l})

-- We need to prove that the union of A and B is { -2, 1, 2015 }
theorem union_of_sets (h_a1 : a = 1) (h_b1 : b = 1) :
  (A a ∪ B a b) = { -2, (1 : α), (2015 : α) } :=
by
  sorry

end union_of_sets_l334_334238


namespace Maggie_age_l334_334816

theorem Maggie_age (Kate Maggie Sue : ℕ) (h1 : Kate + Maggie + Sue = 48) (h2 : Kate = 19) (h3 : Sue = 12) : Maggie = 17 := by
  sorry

end Maggie_age_l334_334816


namespace graph_of_equation_is_two_lines_l334_334081

theorem graph_of_equation_is_two_lines :
  ∀ x y : ℝ, x^2 - 16*y^2 - 8*x + 16 = 0 ↔ (x = 4 + 4*y ∨ x = 4 - 4*y) :=
by
  sorry

end graph_of_equation_is_two_lines_l334_334081


namespace range_of_a_l334_334764

theorem range_of_a (a : ℝ) (h : ∀ θ : ℝ, (a + real.cos θ)^2 + (2 * a - real.sin θ)^2 ≤ 4) :
  -real.sqrt(5) / 5 ≤ a ∧ a ≤ real.sqrt(5) / 5 :=
sorry

end range_of_a_l334_334764


namespace least_five_digit_perfect_square_and_cube_l334_334469

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334469


namespace find_unknown_number_l334_334553

theorem find_unknown_number (x : ℤ) (h : (20 + 40 + 60) / 3 = 9 + (10 + 70 + x) / 3) : x = 13 :=
by
  sorry

end find_unknown_number_l334_334553


namespace product_of_integer_a_that_satisfy_conditions_l334_334674

theorem product_of_integer_a_that_satisfy_conditions :
  (∀ y : ℤ, (2 * y - 6 ≤ 3 * (y - 1)) → (∃ a : ℤ, (1/2 : ℚ) * a - 3 * y > 0)) ∧
  (∀ a : ℤ, (set_of (λ y:ℤ, 2 * y - 6 ≤ 3 * (y - 1) ∧ (1/2 : ℚ) * a - 3 * y > 0)).to_finset.card = 4) →
  ∏ (a : ℤ) in {a | (set_of (λ y:ℤ, 2 * y - 6 ≤ 3 * (y - 1) ∧ (1/2 : ℚ) * a - 3 * y > 0)).to_finset.card = 4}.to_finset, a = 720 :=
sorry

end product_of_integer_a_that_satisfy_conditions_l334_334674


namespace differentiable_at_sqrt_k_l334_334975

def is_lowest_terms (p q : ℤ) : Prop := Int.gcd p q = 1

def f (x : ℝ) : ℝ :=
  if irrational x then 0
  else if ∃ (p : ℤ) (q : ℕ), is_lowest_terms p q ∧ x = p / q then 
    (let ⟨p, q, h⟩ := exists_rat_of_real x in 1 / q^3)
  else 0

theorem differentiable_at_sqrt_k (k : ℕ) (hk : ∀ n : ℕ, n^2 ≠ k) : 
  ∃ f' : ℝ → ℝ, f' (sqrt k) = 0 ∧ differentiable_at ℝ f (sqrt k) :=
sorry

end differentiable_at_sqrt_k_l334_334975


namespace num_ints_between_sqrt2_and_sqrt32_l334_334244

theorem num_ints_between_sqrt2_and_sqrt32 : 
  ∃ n : ℕ, n = 4 ∧ 
  (∀ k : ℤ, (2 ≤ k) ∧ (k ≤ 5)) :=
by
  sorry

end num_ints_between_sqrt2_and_sqrt32_l334_334244


namespace part_I_part_II_l334_334741

noncomputable def f (x : ℝ) : ℝ := x^2 + 1 / Real.sqrt (1 + x)

theorem part_I (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f(x) ≥ x^2 - (1 / 2) * x + 1 := by
  sorry

theorem part_II (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : 15 / 16 < f(x) ∧ f(x) ≤ (2 + Real.sqrt 2) / 2 := by
  sorry

end part_I_part_II_l334_334741


namespace root_equation_l334_334212

variables (m : ℝ)

theorem root_equation {m : ℝ} (h : m^2 - 2 * m - 3 = 0) : m^2 - 2 * m + 2023 = 2026 :=
by {
  sorry 
}

end root_equation_l334_334212


namespace ratio_of_areas_l334_334056

-- Define the terms and parameters given in the problem.
def equilateral_side_length : ℝ := 12
def isosceles_side_length : ℝ := 5

-- Function to calculate the height of an equilateral triangle.
def equilateral_height (a : ℝ) : ℝ :=
  (real.sqrt 3 / 2) * a

-- Function to calculate the area of an equilateral triangle.
def equilateral_area (a : ℝ) : ℝ :=
  (real.sqrt 3 / 4) * (a^2)

-- Function to calculate the height of an isosceles triangle given the side lengths.
def isosceles_height (a b : ℝ) : ℝ :=
  real.sqrt (a^2 - (b / 2)^2)

-- Function to calculate the area of an isosceles triangle.
def isosceles_area (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

-- Function to calculate the remaining area of the central hexagon.
def remaining_hexagon_area (equilateral_area removed_area : ℝ) : ℝ :=
  equilateral_area - 3 * removed_area

-- Define the main problem statement.
theorem ratio_of_areas :
  let base := equilateral_side_length / 2 in
  let equilateral_ht := equilateral_height equilateral_side_length in
  let equilateral_ar := equilateral_area equilateral_side_length in
  let isosceles_ht := isosceles_height isosceles_side_length base in
  let isosceles_ar := isosceles_area base isosceles_ht in
  let remaining_ar := remaining_hexagon_area equilateral_ar isosceles_ar in
  isosceles_ar / remaining_ar = 12 / (36 * real.sqrt 3 - 36) :=
sorry

end ratio_of_areas_l334_334056


namespace least_five_digit_perfect_square_and_cube_l334_334490

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334490


namespace scores_are_correct_l334_334137

variable
  Andrey_score : ℝ
  Dima_score : ℝ
  Vanya_score : ℝ
  Sasha_score : ℝ

-- Conditions
axiom andrey_first : Andrey_score > Dima_score ∧ Andrey_score > Vanya_score ∧ Andrey_score > Sasha_score
axiom dima_second : Dima_score > Vanya_score ∧ Dima_score > Sasha_score
axiom vanya_third : Vanya_score > Sasha_score
axiom unique_scores : Andrey_score ≠ Dima_score ∧ Andrey_score ≠ Vanya_score ∧ Andrey_score ≠ Sasha_score ∧ Dima_score ≠ Vanya_score ∧ Dima_score ≠ Sasha_score ∧ Vanya_score ≠ Sasha_score
axiom total_points : Andrey_score + Dima_score + Vanya_score + Sasha_score = 12
axiom andrey_sasha_wins : Andrey_score = 4 ∧ Sasha_score = 2

-- Conclusion
theorem scores_are_correct :
  Andrey_score = 4 ∧ Dima_score = 3.5 ∧ Vanya_score = 2.5 ∧ Sasha_score = 2 :=
  sorry

end scores_are_correct_l334_334137


namespace interior_angle_of_regular_polygon_l334_334039

theorem interior_angle_of_regular_polygon (n : ℕ) (h_diagonals : n * (n - 3) / 2 = n) :
    n = 5 ∧ (5 - 2) * 180 / 5 = 108 := by
  sorry

end interior_angle_of_regular_polygon_l334_334039


namespace team_X_played_24_games_l334_334901

def games_played_X (x : ℕ) : ℕ := x
def games_played_Y (x : ℕ) : ℕ := x + 9
def games_won_X (x : ℕ) : ℚ := 3 / 4 * x
def games_won_Y (x : ℕ) : ℚ := 2 / 3 * (x + 9)

theorem team_X_played_24_games (x : ℕ) 
  (h1 : games_won_Y x = games_won_X x + 4) : games_played_X x = 24 :=
by
  sorry

end team_X_played_24_games_l334_334901


namespace find_ratio_of_radii_l334_334558

variables (R r x : ℝ)
variables (S1 : {s1 // s1 = r})
variables (S2 : {s2 // s2 = x})
variables (AOB : true)
variables (OC : true)
variables (tangent_AB : true)
variables (tangent_OC : true)
variables (tangent_OA : true)

theorem find_ratio_of_radii 
  (h1 : S1.1 = r)
  (h2 : S2.1 = x)
  (h3 : ∃S1 S2, tangent_AB ∧ tangent_OC ∧ tangent_OA) 
  (h4 : AOB ∧ OC) 
  : S1.1 / S2.1 = 4 * (2 + real.sqrt 3) / 3 ∨ S1.1 / S2.1 = 4 * (2 - real.sqrt 3) / 3 := 
sorry

end find_ratio_of_radii_l334_334558


namespace avg_remaining_wires_length_l334_334410

theorem avg_remaining_wires_length :
    ∀ (n : ℕ) (total_wires : ℕ) (avg_total_length : ℤ) 
      (q1 q2 q3 : ℕ) (avg_q1_length avg_q2_length : ℤ),
        total_wires = 12 ∧ 
        avg_total_length = 95 ∧ 
        q1 = total_wires / 4 ∧
        avg_q1_length = 120 ∧ 
        q2 = total_wires / 3 ∧ 
        avg_q2_length = 75 ∧ 
        n = total_wires - (q1 + q2) → 
        (total_wires * avg_total_length - (q1 * avg_q1_length + q2 * avg_q2_length)) / n = 96 := 
by 
  intros n total_wires avg_total_length q1 q2 q3 avg_q1_length avg_q2_length
  rintros ⟨total_wires_eq, avg_total_length_eq, q1_def, avg_q1_length_eq, 
           q2_def, avg_q2_length_eq, n_def⟩
  sorry

end avg_remaining_wires_length_l334_334410


namespace least_five_digit_perfect_square_and_cube_l334_334474

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334474


namespace probability_both_cards_are_diamonds_l334_334045

-- Conditions definitions
def total_cards : ℕ := 52
def diamonds_in_deck : ℕ := 13
def two_draws : ℕ := 2

-- Calculation definitions
def total_possible_outcomes : ℕ := (total_cards * (total_cards - 1)) / two_draws
def favorable_outcomes : ℕ := (diamonds_in_deck * (diamonds_in_deck - 1)) / two_draws

-- Definition of the probability asked in the question
def probability_both_diamonds : ℚ := favorable_outcomes / total_possible_outcomes

theorem probability_both_cards_are_diamonds :
  probability_both_diamonds = 1 / 17 := 
sorry

end probability_both_cards_are_diamonds_l334_334045


namespace like_terms_product_l334_334713

theorem like_terms_product :
  ∀ (m n : ℕ),
    (-x^3 * y^n) = (3 * x^m * y^2) → (m = 3 ∧ n = 2) → m * n = 6 :=
by
  intros m n h1 h2
  sorry

end like_terms_product_l334_334713


namespace max_mass_grain_l334_334586

open Real

def length : Real := 8
def width : Real := 5
def angle_degrees : Real := 45
def density : Real := 1200 -- kg/m³
def height : Real := width / 2

noncomputable def volume_prism : Real := (width / sqrt 2 / 2) * (width / sqrt 2 / 2) * length

noncomputable def volume_pyramid : Real := (1 / 3) * (width * height) * height

noncomputable def volume_total : Real := volume_prism + 2 * volume_pyramid

noncomputable def max_mass : Real := volume_total * density

theorem max_mass_grain (h_angle : angle_degrees ≤ 45)
  : max_mass = 85000 :=
by
  sorry

end max_mass_grain_l334_334586


namespace solve_trig_eq_l334_334549

theorem solve_trig_eq (k : ℤ) :
  ∀ x : ℝ, 
  (8.413 * cos x * sqrt (tan x ^ 2 - sin x ^ 2) + sin x * sqrt (cot x ^ 2 - cos x ^ 2) = 2 * sin x) ↔ 
  (x = (π / 6) + π * k ∨ x = arcsin ((1 - sqrt 3) / 2) + π * k) :=
by
  sorry

end solve_trig_eq_l334_334549


namespace common_points_circle_curve_l334_334927

def circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * y = 0

def curve (x y : ℝ) : Prop := y = |x| - 1

theorem common_points_circle_curve : ∀ (x y : ℝ),
  circle x y ∧ curve x y → false := by
  sorry

end common_points_circle_curve_l334_334927


namespace tan_a8_of_arithmetic_seq_l334_334704

theorem tan_a8_of_arithmetic_seq (a : ℕ → ℝ) (S : ℝ → ℝ) (h1 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h2 : S 15 = 25 * real.pi) : real.tan (a 8) = -real.sqrt 3 :=
by
  sorry

end tan_a8_of_arithmetic_seq_l334_334704


namespace not_shape_E_l334_334968

-- Definitions
def shape_A : Prop :=
  ∃ (s1 s2 s3 : ℤ × ℤ), s1 = (1, 3) ∨ s2 = (1, 1) ∧ s3 = (1, 1) ∧ 
  ∀ (x y : ℤ), (1, 1) x ⊢ y ⟹ x = 2 * id ∧ y = 3 * id

def shape_B : Prop :=
  ∃ (s1 s2 s3 : ℤ × ℤ), s1 = (1, 3) ∧ s2 = (1, 1) x (3, 3) - (unit - unit^2)

def shape_C : Prop :=
  ∃ (s1 s2 s3 : ℤ × ℤ), s1 = (3, 1) ∧ s2 = (1, 2) ∧ s3 = (1, 1) ^ add

def shape_D : Prop :=
  ∃ (s1 s2 s3 : ℤ × ℤ), s1 ≠ (1, 6) ∘ s3 = (3, 3)

def shape_E : Prop :=
  ∃ (s1 s2 s3 : ℤ × ℤ), s1 = (z, p) a ∧ y ≠ (3, 1) ∘ s3 = (x, y)

-- Proposition: Figure E cannot be formed with the given pieces.
theorem not_shape_E : ¬ shape_E 
by
sortry

end not_shape_E_l334_334968


namespace square_sum_l334_334760

theorem square_sum (a b : ℝ) (h1 : a + b = 8) (h2 : a * b = -2) : a^2 + b^2 = 68 := 
by 
  sorry

end square_sum_l334_334760


namespace polar_coordinates_to_rectangular_l334_334635

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_coordinates_to_rectangular :
  polar_to_rectangular 10 (11 * Real.pi / 6) = (5 * Real.sqrt 3, -5) :=
by
  sorry

end polar_coordinates_to_rectangular_l334_334635


namespace least_five_digit_is_15625_l334_334440

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334440


namespace find_m_l334_334723

theorem find_m (m : ℝ) : 
  (∀ (x y : ℝ), (y = x + m ∧ x = 0) → y = m) ∧
  (∀ (x y : ℝ), (y = 2 * x - 2 ∧ x = 0) → y = -2) ∧
  (∀ (x : ℝ), (∃ y : ℝ, (y = x + m ∧ x = 0) ∧ (y = 2 * x - 2 ∧ x = 0))) → 
  m = -2 :=
by 
  sorry

end find_m_l334_334723


namespace problem_1_problem_2_problem_3_l334_334917

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (2^x + 1)
noncomputable def f_inv (x : ℝ) : ℝ := Real.logb 2 (2^x - 1)

theorem problem_1 : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_inv x = m + f x) ↔ 
  m ∈ (Set.Icc (Real.logb 2 (1/3)) (Real.logb 2 (3/5))) :=
sorry

theorem problem_2 : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f_inv x > m + f x) ↔ 
  m ∈ (Set.Iio (Real.logb 2 (3/5))) :=
sorry

theorem problem_3 : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f_inv x > m + f x) ↔ 
  m ∈ (Set.Iio (Real.logb 2 (1/3))) :=
sorry

end problem_1_problem_2_problem_3_l334_334917


namespace chess_tournament_distribution_l334_334171

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l334_334171


namespace correct_statements_l334_334950
-- Import the necessary Lean library

-- Define the conditions as predicates
def statement1 (r : ℝ) : Prop := r > 0 → ∀ x y, x < y → y < x
def statement2 (r : ℝ) : Prop := r < 0 → ∀ x y, x < y → y < x
def statement3 (r : ℝ) : Prop := (r = 1 ∨ r = -1) → ∀ x y, x = y → ∃ f, ∀ a b, y = f a

-- Define the theorem that shows the set of correct statements is exactly {1, 3}
theorem correct_statements (r : ℝ) : {statement1 r, statement3 r} = {true, true} :=
by
  -- Skip the proof
  sorry

end correct_statements_l334_334950


namespace scores_are_correct_l334_334140

variable
  Andrey_score : ℝ
  Dima_score : ℝ
  Vanya_score : ℝ
  Sasha_score : ℝ

-- Conditions
axiom andrey_first : Andrey_score > Dima_score ∧ Andrey_score > Vanya_score ∧ Andrey_score > Sasha_score
axiom dima_second : Dima_score > Vanya_score ∧ Dima_score > Sasha_score
axiom vanya_third : Vanya_score > Sasha_score
axiom unique_scores : Andrey_score ≠ Dima_score ∧ Andrey_score ≠ Vanya_score ∧ Andrey_score ≠ Sasha_score ∧ Dima_score ≠ Vanya_score ∧ Dima_score ≠ Sasha_score ∧ Vanya_score ≠ Sasha_score
axiom total_points : Andrey_score + Dima_score + Vanya_score + Sasha_score = 12
axiom andrey_sasha_wins : Andrey_score = 4 ∧ Sasha_score = 2

-- Conclusion
theorem scores_are_correct :
  Andrey_score = 4 ∧ Dima_score = 3.5 ∧ Vanya_score = 2.5 ∧ Sasha_score = 2 :=
  sorry

end scores_are_correct_l334_334140


namespace least_five_digit_perfect_square_and_cube_l334_334489

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334489


namespace angle_ZAX_pentagon_triangle_common_vertex_l334_334594

theorem angle_ZAX_pentagon_triangle_common_vertex :
  let n_pentagon := 5
  let n_triangle := 3
  let internal_angle_pentagon := (n_pentagon - 2) * 180 / n_pentagon
  let internal_angle_triangle := 60
  let common_angle_A := 360 - (internal_angle_pentagon + internal_angle_pentagon + internal_angle_triangle + internal_angle_triangle) / 2
  common_angle_A = 192 := by
  let n_pentagon := 5
  let n_triangle := 3
  let internal_angle_pentagon := (n_pentagon - 2) * 180 / n_pentagon
  let internal_angle_triangle := 60
  let common_angle_A := 360 - (internal_angle_pentagon + internal_angle_pentagon + internal_angle_triangle + internal_angle_triangle) / 2
  sorry

end angle_ZAX_pentagon_triangle_common_vertex_l334_334594


namespace surface_area_of_bowling_ball_volume_of_bowling_ball_l334_334014

-- Define the diameter and radius
def diameter : ℝ := 9
def radius : ℝ := diameter / 2

-- Theorem for the surface area of the sphere
theorem surface_area_of_bowling_ball : 4 * Real.pi * radius^2 = 81 * Real.pi :=
by
  let r := radius
  have h1 : r = 9 / 2 := rfl
  sorry

-- Theorem for the volume of the sphere
theorem volume_of_bowling_ball : (4 / 3) * Real.pi * radius^3 = 162 * Real.pi :=
by
  let r := radius
  have h2 : r = 9 / 2 := rfl
  sorry

end surface_area_of_bowling_ball_volume_of_bowling_ball_l334_334014


namespace solution_interval_l334_334859

theorem solution_interval (X₀ : ℝ) (h₀ : Real.log (X₀ + 1) = 2 / X₀) : 1 < X₀ ∧ X₀ < 2 :=
by
  admit -- to be proved

end solution_interval_l334_334859


namespace least_five_digit_perfect_square_and_cube_l334_334429

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334429


namespace cynthia_total_cans_l334_334076

theorem cynthia_total_cans (c1 c2 : ℕ) (h : 4 * c1 = 2 * c2):
  c2 = 8 → c1 = 4 → 8 + 4 = 12 := by
  intros h1 h2
  rw [h1, h2]
  rfl

end cynthia_total_cans_l334_334076


namespace xyz_zero_unique_solution_l334_334112

theorem xyz_zero_unique_solution {x y z : ℝ} (h1 : x^2 * y + y^2 * z + z^2 = 0)
                                 (h2 : z^3 + z^2 * y + z * y^3 + x^2 * y = 1 / 4 * (x^4 + y^4)) :
  x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end xyz_zero_unique_solution_l334_334112


namespace josh_gave_pencils_l334_334294

theorem josh_gave_pencils (original_pencils : ℕ) (remaining_pencils : ℕ) :
  original_pencils = 142 → remaining_pencils = 111 → original_pencils - remaining_pencils = 31 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end josh_gave_pencils_l334_334294


namespace smallest_value_in_Q_eval_l334_334231

def Q (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

theorem smallest_value_in_Q_eval :
  let Q2 := Q 2,
      product_of_zeros := 5, 
      product_of_non_real_zeros := 5,
      sum_of_coefficients := Q 1,
      sum_of_real_zeros := 0 in
  min Q2 (min product_of_zeros (min product_of_non_real_zeros (min sum_of_coefficients sum_of_real_zeros))) = 0 :=
by {
  let Q2 := Q 2,
  let product_of_zeros := 5,
  let product_of_non_real_zeros := 5,
  let sum_of_coefficients := Q 1,
  let sum_of_real_zeros := 0,
  sorry
}

end smallest_value_in_Q_eval_l334_334231


namespace least_five_digit_perfect_square_and_cube_l334_334484

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334484


namespace function_defined_on_reals_l334_334762

theorem function_defined_on_reals (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f(x + 3) ≥ f(x) + 3)
  (h2 : ∀ x : ℝ, f(x + 2) ≤ f(x) + 2)
  (h3 : f(1) = 1) :
  f(2017) = 2017 :=
sorry

end function_defined_on_reals_l334_334762


namespace min_sin6_cos6_l334_334662

theorem min_sin6_cos6 (x : ℝ) : ∃ y : ℝ, (∀ x : ℝ, sin x ^ 6 + 2 * cos x ^ 6 ≥ y) ∧ y = 2 / 3 :=
by
  sorry

end min_sin6_cos6_l334_334662


namespace solve_x_l334_334346

noncomputable def solveEquation (a b c d : ℝ) (x : ℝ) : Prop :=
  x = 3 * a * b + 33 * b^2 + 333 * c^3 + 3.33 * (Real.sin d)^4

theorem solve_x :
  solveEquation 2 (-1) 0.5 (Real.pi / 6) 68.833125 :=
by
  sorry

end solve_x_l334_334346


namespace poly_not_divisible_l334_334256

theorem poly_not_divisible (k : ℕ) (h : ¬(x^(2*k) + 1 + (x + 1)^(2*k)).divisible_by (x^2 + x + 1)) :
  ∃ l : ℕ, k = 3 * l := by
  sorry

end poly_not_divisible_l334_334256


namespace area_midpt_quad_l334_334969

variables {A B C D E F G H : Type*}
variables [convex_quadrilateral A B C D]
variables (E F G H : midpoint_sides A B C D)

theorem area_midpt_quad (S_ABCD : ℝ) (S_EFGH : ℝ)
  (h_midpt : E = midpoint AB ∧ F = midpoint BC ∧ G = midpoint CD ∧ H = midpoint DA) :
  S_EFGH = S_ABCD / 2 :=
sorry

end area_midpt_quad_l334_334969


namespace chess_tournament_points_l334_334148

theorem chess_tournament_points
  (points : String → ℝ)
  (total_points : points "Andrey" + points "Dima" + points "Vanya" + points "Sasha" = 12)
  (distinct_points : 
    points "Andrey" ≠ points "Dima" ∧ 
    points "Andrey" ≠ points "Vanya" ∧ 
    points "Andrey" ≠ points "Sasha" ∧ 
    points "Dima" ≠ points "Vanya" ∧ 
    points "Dima" ≠ points "Sasha" ∧ 
    points "Vanya" ≠ points "Sasha")
  (order : 
    points "Andrey" > points "Dima" ∧ 
    points "Dima" > points "Vanya" ∧ 
    points "Vanya" > points "Sasha")
  (same_wins :
    let games_won (student : String) := (points student - 3) / 0.5 in
    games_won "Andrey" = games_won "Sasha") :
  points "Andrey" = 4 ∧ points "Dima" = 3.5 ∧ points "Vanya" = 2.5 ∧ points "Sasha" = 2 :=
by
  sorry

end chess_tournament_points_l334_334148


namespace max_mass_grain_l334_334587

open Real

def length : Real := 8
def width : Real := 5
def angle_degrees : Real := 45
def density : Real := 1200 -- kg/m³
def height : Real := width / 2

noncomputable def volume_prism : Real := (width / sqrt 2 / 2) * (width / sqrt 2 / 2) * length

noncomputable def volume_pyramid : Real := (1 / 3) * (width * height) * height

noncomputable def volume_total : Real := volume_prism + 2 * volume_pyramid

noncomputable def max_mass : Real := volume_total * density

theorem max_mass_grain (h_angle : angle_degrees ≤ 45)
  : max_mass = 85000 :=
by
  sorry

end max_mass_grain_l334_334587


namespace find_triangle_angles_l334_334810

theorem find_triangle_angles (A B C BC AC : ℝ)
  (h1 : A - B = 60) 
  (h2 : CH = BC - AC)
  (H_ch : CH_height_to_AB CH A B C AB): 
   A = 90 ∧ B = 30 ∧ C = 60 :=
by
  sorry

end find_triangle_angles_l334_334810


namespace concurrency_of_ceva_lines_l334_334313

theorem concurrency_of_ceva_lines 
  (A B C O D E F D' E' F' : Point) 
  (incircle: Triangle E)
  (tangency_D: Tangency (triangle_vertices A B C).sideBC (center O) (circle O))
  (tangency_E: Tangency tA (center O) (circle O))
  (tangency_F: Tangency tA (center O) (circle O))
  (diameter_D: Diameter (line_segment D D' (circle O)))
  (diameter_E: Diameter (line_segment E E' (circle O)))
  (diameter_F: Diameter (line_segment F F' (circle O))) : 
  concurrency (point A D') (point B E') (point C F') := 
sorry

end concurrency_of_ceva_lines_l334_334313


namespace chess_tournament_scores_l334_334159

def points (name : String) := Real

def total_points : Real := 12

variables (A D V S : Real)
variable (total_games : ℕ := 12)

axiom different_scores : A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S

axiom ranking : A > D ∧ D > V ∧ V > S

axiom equal_wins (A S : Real) : (A = 2 * win_points) ∧ (S = 2 * win_points)

axiom total_points_constraint : A + D + V + S = total_points

theorem chess_tournament_scores :
  A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 :=
by 
  sorry

end chess_tournament_scores_l334_334159


namespace purchasing_plans_and_optimal_plan_l334_334016

def company_time := 10
def model_A_cost := 60000
def model_B_cost := 40000
def model_A_production := 15
def model_B_production := 10
def budget := 440000
def production_capacity := 102

theorem purchasing_plans_and_optimal_plan (x y : ℕ) (h1 : x + y = company_time) (h2 : model_A_cost * x + model_B_cost * y ≤ budget) :
  (x = 0 ∧ y = 10) ∨ (x = 1 ∧ y = 9) ∨ (x = 2 ∧ y = 8) ∧ (x = 1 ∧ y = 9) :=
by 
  sorry

end purchasing_plans_and_optimal_plan_l334_334016


namespace loom_weaving_time_l334_334989

/-
  Define the time it takes to weave a given amount of cloth 
  given the conditions and prove the relationship.
-/

theorem loom_weaving_time
  (weaving_rate : ℝ)         -- Rate at which loom weaves (in meters per second)
  (cloth_amount_1 : ℝ)       -- Amount of cloth in meters for the given time
  (time_taken_1 : ℝ)         -- Time taken to weave cloth_amount_1 meters
  (cloth_amount_2 : ℝ)       -- Amount of cloth we need to weave
  (h_weaving_rate : weaving_rate = 0.128)  -- Given rate is 0.128 meters/second
  (h_25_meters_in_195_3125_sec : cloth_amount_1 = 25 ∧ time_taken_1 = 195.3125)
  : (cloth_amount_2 / weaving_rate) = (cloth_amount_2 / 0.128) :=
begin
  sorry
end

end loom_weaving_time_l334_334989


namespace least_five_digit_perfect_square_cube_l334_334513

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334513


namespace least_five_digit_perfect_square_and_cube_l334_334485

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334485


namespace restock_quantities_correct_l334_334577

-- Definition for the quantities of cans required
def cans_peas : ℕ := 810
def cans_carrots : ℕ := 954
def cans_corn : ℕ := 675

-- Definition for the number of cans per box, pack, and case.
def cans_per_box_peas : ℕ := 4
def cans_per_pack_carrots : ℕ := 6
def cans_per_case_corn : ℕ := 5

-- Define the expected order quantities.
def order_boxes_peas : ℕ := 203
def order_packs_carrots : ℕ := 159
def order_cases_corn : ℕ := 135

-- Proof statement for the quantities required to restock exactly.
theorem restock_quantities_correct :
  (order_boxes_peas = Nat.ceil (cans_peas / cans_per_box_peas))
  ∧ (order_packs_carrots = cans_carrots / cans_per_pack_carrots)
  ∧ (order_cases_corn = cans_corn / cans_per_case_corn) :=
by
  sorry

end restock_quantities_correct_l334_334577


namespace triangle_ratio_l334_334752

theorem triangle_ratio
  (A B C O H D E D' E' K : Type)
  [Point A] [Point B] [Point C] [Point O] [Point H] [Point D] [Point E] [Point D'] [Point E'] [Point K]
  (h1 : Circumcenter O A B C)
  (h2 : Orthocenter H A B C)
  (h3 : O ≠ H)
  (h4 : Midpoint D B C)
  (h5 : Midpoint E C A)
  (h6 : Reflection D' D H)
  (h7 : Reflection E' E H)
  (h8 : Intersect AD' BE' K) :
  |KO| / |KH| = 3 / 2 := 
sorry

end triangle_ratio_l334_334752


namespace total_expenditure_of_7_people_l334_334985

theorem total_expenditure_of_7_people :
  ∃ A : ℝ, 
    (6 * 11 + (A + 6) = 7 * A) ∧
    (6 * 11 = 66) ∧
    (∃ total : ℝ, total = 6 * 11 + (A + 6) ∧ total = 84) :=
by 
  sorry

end total_expenditure_of_7_people_l334_334985


namespace parabola_equation_l334_334115

theorem parabola_equation :
  ∃ a b c : ℝ, (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y = -2 * x^2 + 12 * x - 13) ∧
  (∀ x y : ℝ, y = a * (x - 3)^2 + 5 ↔ y = -2 * (x - 3)^2 + 5) ∧
  (∀ x y : ℝ, y = a * (x - 3)^2 + 5 ∧ (4, 3 : ℝ) ∈ set_of (λ p, p.snd = a * ((p.fst - 3) ^ 2) + 5)) := 
sorry

end parabola_equation_l334_334115


namespace exists_same_color_points_one_meter_apart_l334_334933

-- Declare the colors as an enumeration
inductive Color
| red : Color
| black : Color

-- Define the function that assigns a color to each point in the plane
def color (point : ℝ × ℝ) : Color := sorry

-- The theorem to be proven
theorem exists_same_color_points_one_meter_apart :
  ∃ x y : ℝ × ℝ, x ≠ y ∧ dist x y = 1 ∧ color x = color y :=
sorry

end exists_same_color_points_one_meter_apart_l334_334933


namespace chess_tournament_scores_l334_334129

theorem chess_tournament_scores :
    ∃ (A D V S : ℝ),
    A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S ∧
    A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 ∧
    A > D ∧ D > V ∧ V > S ∧
    (∃ (wins_A wins_S : ℕ), wins_A = wins_S) :=
begin
    sorry
end

end chess_tournament_scores_l334_334129


namespace analytical_expression_of_f_range_of_m_l334_334739

open Real

noncomputable def f (x : ℝ) (ω : ℝ) := sqrt 3 * sin (2 * ω * x) - cos (2 * ω * x)

theorem analytical_expression_of_f (ω : ℝ) (hω : ω > 0) :
    ∀ x, f x ω = 2 * sin (2 * x - π / 6) :=
sorry

theorem range_of_m (m : ℝ) :
    (∀ x ∈ Icc (0 : ℝ) (π / 2), f x 1 ≤ m) ↔ m ∈ Icc (2 : ℝ) ∞ :=
sorry

end analytical_expression_of_f_range_of_m_l334_334739


namespace least_five_digit_is_15625_l334_334438

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334438


namespace change_color_while_preserving_friendship_l334_334561

-- Definitions
def children := Fin 10000
def colors := Fin 7
def friends (a b : children) : Prop := sorry -- mutual and exactly 11 friends per child
def refuses_to_change (c : children) : Prop := sorry -- only 100 specified children refuse to change color

theorem change_color_while_preserving_friendship :
  ∃ c : children, ¬refuses_to_change c ∧
    ∃ new_color : colors, 
      (∀ friend : children, friends c friend → 
      (∃ current_color current_friend_color : colors, current_color ≠ current_friend_color)) :=
sorry

end change_color_while_preserving_friendship_l334_334561


namespace correct_statements_are_l334_334572

-- Definitions for conditions
def total_members := 50
def male_members := 30
def female_members := 20
def sample_size := 5
def male_sample := 2
def female_sample := 3

-- Boolean evaluations of statements
def is_systematic_sampling_possible : Bool :=
  -- given the explanation proving systematic sampling can be done
  true

def is_random_sampling_possible : Bool :=
  -- given the explanation proving random sampling is possible
  true

def is_stratified_sampling_appropriate : Bool :=
  -- given the explanation proving stratified sampling is not appropriate
  false

def male_probability : ℚ := male_sample / male_members
def female_probability : ℚ := female_sample / female_members
def is_male_probability_greater : Bool := male_probability > female_probability

-- Proof that the correct choice among statements is B
theorem correct_statements_are (h: is_systematic_sampling_possible = true)
                               (h1: is_random_sampling_possible = true)
                               (h2: is_stratified_sampling_appropriate = false)
                               (h3: is_male_probability_greater = false) :
    (h1 = true) ∧ (h2 = true) ∧ (h = false) ∧ (h3 = false) :=
begin
  -- Sorry used to skip the actual proof
  sorry
end

end correct_statements_are_l334_334572


namespace least_five_digit_is_15625_l334_334445

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334445


namespace ratio_of_B_to_C_l334_334580

variables (A B C : ℕ)

def A_condition := A = B + 2
def B_condition := B = 8
def total_condition := A + B + C = 22

theorem ratio_of_B_to_C : (A = B + 2) → (B = 8) → (A + B + C = 22) → (B / C = 2) :=
by
  intro h1 h2 h3
  rw [h2, h1] at h3
  have hA : A = 10 := by linarith[h1, h2]
  have hC : C = 4 := by linarith[h3]
  have ratio : B / C = 2 := by linarith [h2, hC]
  exact ratio

end ratio_of_B_to_C_l334_334580


namespace least_five_digit_perfect_square_cube_l334_334517

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334517


namespace key_sequence_correct_l334_334952

def key_mapping : Char → Char :=
  λ c, match c with
  | 'J' => 'H'
  | 'K' => 'O'
  | 'L' => 'P'
  | 'R' => 'E'
  | 'N' => 'M'
  | 'Q' => 'A'
  | 'Y' => 'T'
  | _ => c  -- Default case (although, out of scope for given problem)

theorem key_sequence_correct :
  (key_mapping 'J' = 'H') ∧
  (key_mapping 'K' = 'O') ∧
  (key_mapping 'L' = 'P') ∧
  (key_mapping 'R' = 'E') ∧
  (key_mapping 'N' = 'M') ∧
  (key_mapping 'Q' = 'A') ∧
  (key_mapping 'Y' = 'T') ∧
  (key_mapping 'J' = 'H') →
  "JKLRNQYJ".map key_mapping = "HOPEMATH" :=
by
  intros
  sorry

end key_sequence_correct_l334_334952


namespace xy_sum_of_squares_l334_334188

theorem xy_sum_of_squares (x y : ℝ) (h1 : x - y = 5) (h2 : -x * y = 4) : x^2 + y^2 = 17 := 
sorry

end xy_sum_of_squares_l334_334188


namespace maximize_area_of_intersection_l334_334006

variables {A B C D K M : Type*}
variables [parallelogram A B C D] [point_on_segment K A B] [point_on_segment M C D]

theorem maximize_area_of_intersection (h1 : parallelogram A B C D) (h2 : point_on_segment K A B) (h3 : point_on_segment M C D) :
  ∃ (K M : Type*), (segment_parallel K . M A . D) ∧ (segment_parallel K . M B . C) :=
begin
  sorry
end

end maximize_area_of_intersection_l334_334006


namespace constant_term_in_expansion_is_10_l334_334909

-- Definition of the binomial expansion terms
def binomial_expansion_general_term (n r : ℕ) (a b : ℤ) :=
  (nat.choose n r) * (a ^ (n - r)) * (b ^ r)

-- Specific parameters for this problem
def x := 2 -- exponent for the first term x^2
def y := -3 -- exponent for the second term 1/x^3 (represented as x^{-3})

-- Function that computes the term in the expansion where exponents sum to 0
def constant_term_of_expansion : ℤ :=
  let r := 2 in -- solve for r such that 10 - 5r = 0
  binomial_expansion_general_term 5 r 1 1

-- Theorem: Proving the constant term is 10
theorem constant_term_in_expansion_is_10 : constant_term_of_expansion = 10 := by
  -- The proof steps are omitted. We'll directly assert the correct answer.
  sorry

end constant_term_in_expansion_is_10_l334_334909


namespace least_five_digit_perfect_square_and_cube_l334_334432

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334432


namespace compute_u_dot_v_cross_w_l334_334834

open Real EuclideanSpace

variables (u v w : ℝ^3) 

-- Assumptions
h1 : ∥u∥ = 1  -- u is a unit vector
h2 : ∥v∥ = 1  -- v is a unit vector
h3 : w = u × v + 2 • u
h4 : w × u = 2 • v

theorem compute_u_dot_v_cross_w : u ⋅ (v × w) = 1 :=
by
  sorry

end compute_u_dot_v_cross_w_l334_334834


namespace smallest_positive_period_of_f_range_of_f_in_interval_l334_334737

noncomputable def f (x: ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 + 2 * (Real.sqrt 3) * (Real.sin x) * (Real.sin (x + Real.pi / 2))

#eval (Real.sin (x + Real.pi / 2)) -- should simplify to Real.cos x

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f(x + T) = f x) ∧ (∀ τ > 0, (∀ x, f(x + τ) = f x) → τ ≥ T) := sorry

theorem range_of_f_in_interval :
  ∀ x, 0 ≤ x ∧ x ≤ (2 * Real.pi / 3) → 0 ≤ f x ∧ f x ≤ 3 := sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l334_334737


namespace remainder_mod_7_l334_334622

theorem remainder_mod_7 :
  let seq := list.range' 3 91 10
  let squares := seq.map (fun x => x^2)
  ∏ i in squares, i % 7 = 2 := by
  sorry

end remainder_mod_7_l334_334622


namespace original_decimal_l334_334871

theorem original_decimal (x : ℝ) : (10 * x = x + 2.7) → x = 0.3 := 
by
    intro h
    sorry

end original_decimal_l334_334871


namespace find_pdf_l334_334118

noncomputable def F (x y : ℝ) : ℝ :=
  (1 / Real.pi * Real.arctan x + 1 / 2) * (1 / Real.pi * Real.arctan y + 1 / 2)

theorem find_pdf 
  (x y : ℝ) : 
  let F := F x y
  let p (x y : ℝ) := (1 / Real.pi^2) * (1 / ((1 + x^2) * (1 + y^2)))
  ∂∂ (F x y) = p x y := 
sorry

end find_pdf_l334_334118


namespace total_wheels_in_neighborhood_l334_334292

def cars_in_Jordan_driveway := 2
def wheels_per_car := 4
def spare_wheel := 1
def bikes_with_2_wheels := 3
def wheels_per_bike := 2
def bike_missing_rear_wheel := 1
def bike_with_training_wheel := 2 + 1
def trash_can_wheels := 2
def tricycle_wheels := 3
def wheelchair_main_wheels := 2
def wheelchair_small_wheels := 2
def wagon_wheels := 4
def roller_skates_total_wheels := 4
def roller_skates_missing_wheel := 1

def pickup_truck_wheels := 4
def boat_trailer_wheels := 2
def motorcycle_wheels := 2
def atv_wheels := 4

theorem total_wheels_in_neighborhood :
  (cars_in_Jordan_driveway * wheels_per_car + spare_wheel + bikes_with_2_wheels * wheels_per_bike + bike_missing_rear_wheel + bike_with_training_wheel + trash_can_wheels + tricycle_wheels + wheelchair_main_wheels + wheelchair_small_wheels + wagon_wheels + (roller_skates_total_wheels - roller_skates_missing_wheel)) +
  (pickup_truck_wheels + boat_trailer_wheels + motorcycle_wheels + atv_wheels) = 47 := by
  sorry

end total_wheels_in_neighborhood_l334_334292


namespace two_zeroes_condition_l334_334222

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 0 then 2^x - a else 3 * x - a

theorem two_zeroes_condition (a : ℝ) (h : ∃ x₁ x₂ : ℝ, f x₁ a = 0 ∧ f x₂ a = 0 ∧ x₁ ≠ x₂) :
  a ∈ set.Ioc 0 1 :=
begin
  sorry
end

end two_zeroes_condition_l334_334222


namespace determine_f_l334_334192

noncomputable def transformed_function (f : ℝ → ℝ) :=
  λ x : ℝ, (f ((x + (Real.pi / 2)) / 2)) * 4

theorem determine_f (f : ℝ → ℝ) :
  ∀ x : ℝ, transformed_function f x = 2 * Real.sin x → 
  f x = - (1 / 2) * Real.cos (2 * x) :=
begin
  sorry
end

end determine_f_l334_334192


namespace sum_of_coefficients_of_rational_terms_in_binomial_expansion_l334_334942

theorem sum_of_coefficients_of_rational_terms_in_binomial_expansion :
  let T (r : ℕ) := 2^(6-r) * (-1)^r * (Nat.choose 6 r : ℂ) * x^(3 - 3*r/2 : ℤ)
  let S := {(T r) | r ∈ {0, 2, 4, 6}}
  sum (coefficients where S) = 365 := sorry

end sum_of_coefficients_of_rational_terms_in_binomial_expansion_l334_334942


namespace final_lights_on_l334_334949

def lights_on_by_children : ℕ :=
  let total_lights := 200
  let flips_x := total_lights / 7
  let flips_y := total_lights / 11
  let lcm_xy := 77  -- since lcm(7, 11) = 7 * 11 = 77
  let flips_both := total_lights / lcm_xy
  flips_x + flips_y - flips_both

theorem final_lights_on : lights_on_by_children = 44 :=
by
  sorry

end final_lights_on_l334_334949


namespace vasya_wins_l334_334328

/-
  Petya and Vasya are playing a game where initially there are 2022 boxes, 
  each containing exactly one matchstick. In one move, a player can transfer 
  all matchsticks from one non-empty box to another non-empty box. They take turns, 
  with Petya starting first. The winner is the one who, after their move, has 
  at least half of all the matchsticks in one box for the first time. 

  We want to prove that Vasya will win the game with the optimal strategy.
-/

theorem vasya_wins : true :=
  sorry -- placeholder for the actual proof

end vasya_wins_l334_334328


namespace probability_division_integer_l334_334370

-- Definitions of sets and conditions
def R : Finset ℤ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def K : Finset ℤ := {3, 4, 5, 6, 7, 8, 9}

-- Lean definition of the proof problem
theorem probability_division_integer :
  let valid_pairs := Finset.filter (λ (rk : ℤ × ℤ), rk.2 ∣ rk.1) (R.product K) in
  (valid_pairs.card : ℚ) / (R.card * K.card) = 1 / 7 := by
{
  -- Sorry placeholder for the proof
  sorry
}

end probability_division_integer_l334_334370


namespace tangent_line_at_P_l334_334732

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x
noncomputable def f_prime (x : ℝ) : ℝ := 3 * x^2 - 3
def P : ℝ × ℝ := (2, -6)

theorem tangent_line_at_P :
  ∃ (m b : ℝ), (∀ (x : ℝ), f_prime x = m) ∧ (∀ (x : ℝ), f x - f 2 = m * (x - 2) + b) ∧ (2 : ℝ) = 2 → b = 0 ∧ m = -3 :=
by
  sorry

end tangent_line_at_P_l334_334732


namespace combination_indices_l334_334757

theorem combination_indices (x : ℕ) (h : nat.choose 20 (2*x - 1) = nat.choose 20 (x+3)) : x = 4 ∨ x = 6 :=
by
  sorry

end combination_indices_l334_334757


namespace problem_solution_l334_334124

variable {V : Type} [InnerProductSpace ℝ V] [CompleteSpace V]

noncomputable def statement_B (a b : V) : Prop :=
  ∥a - b∥ ≤ abs (∥a∥ - ∥b∥)

theorem problem_solution : ∃ (a b : V), ¬statement_B a b :=
sorry

end problem_solution_l334_334124


namespace base6_digit_divisible_by_13_l334_334676

noncomputable def base6_to_base10 (d : ℕ) : ℕ := 3 * 6^3 + d * 6^2 + d * 6 + 4

theorem base6_digit_divisible_by_13 :
  ∃ d : ℕ, 0 ≤ d ∧ d ≤ 5 ∧ (652 + 42 * d) % 13 = 0 := by
  use 4
  split
  { exact nat.zero_le 4 }
  split
  { exact nat.le_of_lt_succ (nat.lt_succ_self 4) }
  { norm_num }
  sorry

end base6_digit_divisible_by_13_l334_334676


namespace total_cost_is_135_25_l334_334951

-- defining costs and quantities
def cost_A : ℕ := 9
def num_A : ℕ := 4
def cost_B := cost_A + 5
def num_B : ℕ := 2
def cost_clay_pot := cost_A + 20
def cost_bag_soil := cost_A - 2
def cost_fertilizer := cost_A + (cost_A / 2)
def cost_gardening_tools := cost_clay_pot - (cost_clay_pot / 4)

-- total cost calculation
def total_cost : ℚ :=
  (num_A * cost_A) + 
  (num_B * cost_B) + 
  cost_clay_pot + 
  cost_bag_soil + 
  cost_fertilizer + 
  cost_gardening_tools

theorem total_cost_is_135_25 : total_cost = 135.25 := by
  sorry

end total_cost_is_135_25_l334_334951


namespace chess_tournament_solution_l334_334162

def chess_tournament_points (points : List ℝ) : Prop :=
  let andrey := points[0]
  let dima := points[1]
  let vanya := points[2]
  let sasha := points[3]
  andrey = 4 ∧ dima = 3.5 ∧ vanya = 2.5 ∧ sasha = 2

axiom chess_tournament_conditions (points : List ℝ) :
  -- Andrey secured first place, Dima secured second, Vanya secured third, and Sasha secured fourth.
  List.Nodup points ∧
  points.length = 4 ∧
  (∀ p, p ∈ points → p = 4 ∨ p = 3.5 ∨ p = 2.5 ∨ p = 2) ∧
  -- Andrey and Sasha won the same number of games.
  (points[0] ≠ points[1] ∧ points[0] ≠ points[2] ∧ points[0] ≠ points[3] ∧
   points[1] ≠ points[2] ∧ points[1] ≠ points[3] ∧
   points[2] ≠ points[3])

theorem chess_tournament_solution (points : List ℝ) :
  chess_tournament_conditions points → chess_tournament_points points :=
by
  sorry

end chess_tournament_solution_l334_334162


namespace polygon_number_of_sides_l334_334923

theorem polygon_number_of_sides
  (n : ℕ)
  (d : ℝ)
  (largest_angle : ℝ)
  (h₁ : d = 10)
  (h₂ : largest_angle = 175)
  (angles_are_in_ap : ∀ i ∈ (range n), (((i + 1) * d + (largest_angle - d)) = largest_angle)) :
  n = 39 := by
  sorry

end polygon_number_of_sides_l334_334923


namespace rectangle_area_error_l334_334270

theorem rectangle_area_error
  (L W : ℝ) :
  let L' := 1.15 * L in
  let W' := 0.89 * W in
  let A := L * W in
  let A' := L' * W' in
  (A' - A) / A * 100 = 2.35 :=
by
  -- Actual definition of L' and W'
  let L' := 1.15 * L
  let W' := 0.89 * W
  -- Definition of actual area A and computed area A'
  let A := L * W
  let A' := L' * W'
  -- Proof statement that needs to be shown
  show (A' - A) / A * 100 = 2.35 from
    sorry

end rectangle_area_error_l334_334270


namespace least_five_digit_perfect_square_and_cube_l334_334447

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334447


namespace inequality_p_l334_334672

-- Define p(n) as a function that counts the number of non-decreasing sequences of positive integers summing to n.
def p (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition.

theorem inequality_p (n : ℕ) (hn : 0 < n) :
  (1 + ∑ i in Finset.range n, p i) / p n ≤ Real.sqrt (2 * n) :=
sorry

end inequality_p_l334_334672


namespace expression_rewrite_l334_334883

theorem expression_rewrite :
  ∃ (d r s : ℚ), (∀ k : ℚ, 8*k^2 - 6*k + 16 = d*(k + r)^2 + s) ∧ s / r = -118 / 3 :=
by sorry

end expression_rewrite_l334_334883


namespace shortest_distance_to_y_axis_is_3_l334_334404

-- Define the parabola and the fixed length of the line segment
def parabola (x y : ℝ) := y^2 = 8 * x
def fixed_length (A B : ℝ × ℝ) := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 100

-- Define the midpoint of AB
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the shortest distance from P to the y-axis
def mid_distance_to_y_axis (P : ℝ × ℝ) : ℝ := abs P.1

-- The theorem statement
theorem shortest_distance_to_y_axis_is_3 (A B P : ℝ × ℝ) 
  (hA : parabola A.1 A.2) 
  (hB : parabola B.1 B.2) 
  (hAB : fixed_length A B) 
  (hP : P = midpoint A B) :
  mid_distance_to_y_axis P = 3 :=
sorry

end shortest_distance_to_y_axis_is_3_l334_334404


namespace order_of_numbers_l334_334393

theorem order_of_numbers (a b c : ℝ) (ha : a = 3^0.7) (hb : b = 0.7^3) (hc : c = log 3 0.7) : c < b ∧ b < a :=
by
  sorry

end order_of_numbers_l334_334393


namespace exists_polynomial_with_properties_l334_334890

theorem exists_polynomial_with_properties :
  ∃ (P : ℤ[X]) (Q : ℚ[X]),
    P * 2016 = Q * polynomial.X^(Q.natDegree + 1)
    ∧ ∀ x : ℤ, P.eval x ∈ ℤ ∧ (Q.leadingCoeff = 1 / 2016) := sorry

end exists_polynomial_with_properties_l334_334890


namespace find_z_from_cubes_l334_334797

theorem find_z_from_cubes :
  (∃ (w x y z : ℕ), w^3 + x^3 + y^3 = z^3 ∧ w = x - 2 ∧ x = y - 1 ∧ y = z - 3 ∧ w, x, y, z > 0) →
  (∃ z : ℕ, z = 9) :=
by
  sorry

end find_z_from_cubes_l334_334797


namespace chess_tournament_points_distribution_l334_334178

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l334_334178


namespace decreasing_interval_of_even_quadratic_l334_334255

theorem decreasing_interval_of_even_quadratic {k : ℝ} 
  (h : ∀ x : ℝ, f x = k * x^2 + (k - 1) * x + 2 ∧ f x = f (-x)) : 
  ∀ x : ℝ, x ∈ set.Ioo (neg_infinity) (0) → f x > f 0 := 
sorry

where
noncomputable def f (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

end decreasing_interval_of_even_quadratic_l334_334255


namespace chess_tournament_distribution_l334_334170

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l334_334170


namespace intersection_A_B_l334_334302

def A : Set ℝ := { x | x + 1 > 0 }
def B : Set ℝ := { x | x < 0 }

theorem intersection_A_B :
  A ∩ B = { x | -1 < x ∧ x < 0 } :=
sorry

end intersection_A_B_l334_334302


namespace sum_of_zeros_is_neg4_l334_334367

noncomputable def sum_of_zeros_of_transformed_parabola : ℝ :=
let f1 := λ x : ℝ, (x - 2)^2 + 3
let f2 := λ x : ℝ, -((x - 2)^2) + 3
let f3 := λ x : ℝ, -(x + 2 - 4)^2 + 3
let f4 := λ x : ℝ, -(x + 2)^2
let transformed_parabola := f4
let zeros := { x : ℝ | transformed_parabola x = 0}
in
zeros.sum

theorem sum_of_zeros_is_neg4 : sum_of_zeros_of_transformed_parabola = -4 :=
sorry

end sum_of_zeros_is_neg4_l334_334367


namespace house_cats_initial_l334_334031

def initial_house_cats (S A T H : ℝ) : Prop :=
  S + H + A = T

theorem house_cats_initial (S A T H : ℝ) (h1 : S = 13.0) (h2 : A = 10.0) (h3 : T = 28) :
  initial_house_cats S A T H ↔ H = 5 := by
sorry

end house_cats_initial_l334_334031


namespace BETA_length_sum_l334_334642

def length_B := 2 * 2 + 1 + 1 + Real.sqrt (1^2 + 2^2)
def length_E := 1 + 1 + 1 + 2
def length_T := 2 + 1
def length_A := 1 + 2 * Real.sqrt (1^2 + 1^2)
def total_length := length_B + length_E + length_T + length_A

theorem BETA_length_sum : total_length = 14 + 2 * Real.sqrt 2 + Real.sqrt 5 := 
by 
  -- Insert the proof here
  sorry

end BETA_length_sum_l334_334642


namespace find_aquarium_breadth_l334_334980

/-- This problem verifies the breadth of an aquarium based on its dimensions and the volume of water added. -/
theorem find_aquarium_breadth
  (length height volume : ℝ)
  (water_rise : ℝ)
  (h1 : length = 50)
  (h2 : height = 40)
  (h3 : volume = 10000)
  (h4 : water_rise = 10) :
  ∃ breadth : ℝ, breadth = 20 :=
begin
  sorry
end

end find_aquarium_breadth_l334_334980


namespace sine_alpha_eqn_l334_334687

-- Setting up the hypotheses and goal
theorem sine_alpha_eqn (α : ℝ) :
  (sqrt 2 / 2) * (sin (α / 2) - cos (α / 2)) = sqrt 6 / 3 → sin α = -1 / 3 :=
by
  sorry

end sine_alpha_eqn_l334_334687


namespace simplify_polynomial_simplify_expression_l334_334341

-- Problem 1:
theorem simplify_polynomial (x : ℝ) : 
  2 * x^3 - 4 * x^2 - 3 * x - 2 * x^2 - x^3 + 5 * x - 7 = x^3 - 6 * x^2 + 2 * x - 7 := 
by
  sorry

-- Problem 2:
theorem simplify_expression (m n : ℝ) (A B : ℝ) (hA : A = 2 * m^2 - m * n) (hB : B = m^2 + 2 * m * n - 5) : 
  4 * A - 2 * B = 6 * m^2 - 8 * m * n + 10 := 
by
  sorry

end simplify_polynomial_simplify_expression_l334_334341


namespace triangle_perimeter_l334_334036

theorem triangle_perimeter (A B C D C' E : EuclideanGeometry.Point)
  (h1 : EuclideanGeometry.dist A B = 1)
  (h2 : EuclideanGeometry.dist B C = 2)
  (h3 : EuclideanGeometry.dist B D = 1)
  (h4 : EuclideanGeometry.dist A D = 2)
  (h5 : EuclideanGeometry.dist C' D = 1 / 4)
  (h6 : EuclideanGeometry.collinear B A E)
  (h7 : EuclideanGeometry.collinear B C E)
  (h8 : EuclideanGeometry.collinear A D C')
  : EuclideanGeometry.perimeter (EuclideanGeometry.triangle A E C') = (41 + Real.sqrt 73) / 12 := 
sorry

end triangle_perimeter_l334_334036


namespace gabrielle_total_crates_l334_334683

theorem gabrielle_total_crates (monday tuesday wednesday thursday : ℕ)
  (h_monday : monday = 5)
  (h_tuesday : tuesday = 2 * monday)
  (h_wednesday : wednesday = tuesday - 2)
  (h_thursday : thursday = tuesday / 2) :
  monday + tuesday + wednesday + thursday = 28 :=
by
  sorry

end gabrielle_total_crates_l334_334683


namespace sum_of_solutions_l334_334532

theorem sum_of_solutions : 
  let integer_solutions := { x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 } in
  ∑ x in integer_solutions, x = 24 := 
sorry

end sum_of_solutions_l334_334532


namespace find_s2_midpoint_l334_334887

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

noncomputable def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

def s1_midpoint : ℝ × ℝ :=
  midpoint (3, 2) (-7, 8)

def s2_midpoint : ℝ × ℝ :=
  translate s1_midpoint 4 2

theorem find_s2_midpoint : s2_midpoint = (2, 7) :=
by
  sorry

end find_s2_midpoint_l334_334887


namespace chess_tournament_distribution_l334_334168

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l334_334168


namespace find_A_and_area_l334_334686

variable (A B C : ℝ) -- Angles of triangle
variable (a b c : ℝ) -- Opposite sides

-- Conditions
variables (h1 : cos B * cos C - sin B * sin C = 1 / 2)
variables (h2 : a = 2 * sqrt 3)
variables (h3 : b + c = 4)

theorem find_A_and_area (h1 : cos B * cos C - sin B * sin C = 1 / 2)
                        (h2 : a = 2 * sqrt 3)
                        (h3 : b + c = 4) :
    A = 2 * π / 3 ∧
    (1 / 2) * b * c * sin A = sqrt 3 := by
    sorry

end find_A_and_area_l334_334686


namespace quadrilateral_area_proof_l334_334035

def point := (ℝ × ℝ)

def quadrilateral_area (A B C D : point) : ℝ :=
  let triangle_area ((x1, y1) : point) ((x2, y2) : point) ((x3, y3) : point) : ℝ :=
    0.5 * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))
  in triangle_area A B C + triangle_area A C D

def A : point := (1, 3)
def B : point := (1, 1)
def C : point := (3, 1)
def D : point := (2007, 2008)

theorem quadrilateral_area_proof : quadrilateral_area A B C D = 2830.42 :=
  sorry

end quadrilateral_area_proof_l334_334035


namespace percentage_decrease_l334_334937

variable (S : ℝ) (x : ℝ)

-- The conditions
def initial_salary := S
def increased_salary := 1.10 * S
def net_salary := 1.01 * S

-- The equation based on the problem statement
def equation := 1.10 * S * (1 - x / 100) = 1.01 * S

-- The proof problem
theorem percentage_decrease : equation → x = 9 / 1.10 :=
by
  intro h
  sorry

end percentage_decrease_l334_334937


namespace least_five_digit_perfect_square_and_cube_l334_334471

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334471


namespace compare_values_l334_334190

noncomputable def a : ℝ := 2^(-1/3)
noncomputable def b : ℝ := Real.logb 2 (1/3)
noncomputable def c : ℝ := Real.logb (1/2) (1/3)

theorem compare_values : c > a ∧ a > b := 
by
  sorry

end compare_values_l334_334190


namespace determine_g_l334_334079

noncomputable def g : ℝ → ℝ :=
  λ x, 5^x - 3^x

theorem determine_g :
  g 1 = 1 ∧ ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y :=
by
  split
  sorry

end determine_g_l334_334079


namespace contains_all_numbers_from_1_to_100_l334_334347

-- Definitions and Hypotheses
variables {α : Type} [LinearOrder α] 
def blue_numbers (a : ℕ → ℕ) : Set ℕ := {n | a n < k}
def red_numbers (a : ℕ → ℕ) : Set ℕ := {n | a n >= k}

-- Conditions from the problem
variables (a : ℕ → ℕ)
variables (h1 : ∀ i, 1 ≤ i ∧ i ≤ 100 → (a i ∈ blue_numbers a) ∨ (a i ∈ red_numbers a))
variables (h2 : ∀ i, i ∈ red_numbers a → a i ∈ Icc 1 100)
variables (h3 : ∀ i, i ∈ blue_numbers a → a i ∈ Icc 1 100)
variables (k : ℕ)

-- Statement of the proof problem
theorem contains_all_numbers_from_1_to_100 :
  ∀ n ∈ Icc 1 100, ∃ i, 1 ≤ i ∧ i ≤ 100 ∧ a i = n := sorry

end contains_all_numbers_from_1_to_100_l334_334347


namespace equation_proof_l334_334979

theorem equation_proof :
  (40 + 5 * 12) / (180 / 3^2) + Real.sqrt 49 = 12 := 
by 
  sorry

end equation_proof_l334_334979


namespace pints_in_3_liters_is_6_point_8_l334_334718

noncomputable def pints_in_liters (liters : ℝ) : ℝ := (liters / 0.5) * 1.13

theorem pints_in_3_liters_is_6_point_8 :
  (pints_in_liters 3).round = 6.8 :=
by
  sorry

end pints_in_3_liters_is_6_point_8_l334_334718


namespace forty_percent_of_number_l334_334766

variables {N : ℝ}

theorem forty_percent_of_number (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 10) : 0.40 * N = 120 :=
by sorry

end forty_percent_of_number_l334_334766


namespace sum_of_squares_l334_334648

theorem sum_of_squares (ABC : Type) [equilateral_triangle ABC]
  (s : ℝ) (h_s : s = sqrt 75)
  (D1 D2 : Points ABC) (E1 E2 E3 E4 : Points ABC)
  (h_D1 : distance B D1 = sqrt 15) (h_D2 : distance B D2 = sqrt 15)
  (h_congruent1 : congruent (triangle A D1 E1) (triangle A B C))
  (h_congruent2 : congruent (triangle A D1 E2) (triangle A B C))
  (h_congruent3 : congruent (triangle A D2 E3) (triangle A B C))
  (h_congruent4 : congruent (triangle A D2 E4) (triangle A B C)) :
  ∑ k in [CE1, CE2, CE3, CE4], (distance C k)^2 = 465 := 
sorry

end sum_of_squares_l334_334648


namespace staff_price_l334_334550

theorem staff_price (d : ℝ) : (d - 0.55 * d) / 2 = 0.225 * d := by
  sorry

end staff_price_l334_334550


namespace circle_sum_l334_334799

theorem circle_sum :
  ∃ (a b c d e f : ℕ),
    {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6} ∧
    a + b + c = 14 ∧ d + e + f = 14 ∧ a + d + e = 14 :=
begin
  sorry
end

end circle_sum_l334_334799


namespace problem1_problem2_l334_334738

noncomputable def Px (f : ℝ → ℝ) : Prop :=
  ∃ (A ω φ : ℝ), (A > 0) ∧ (ω > 0) ∧ (|φ| ≤ π/2) ∧
  (∀ x, f x = A * Real.cos (ω * x + φ)) ∧
  (∀ x, f x ≤ 2) ∧ (∀ x, f x - f (x + π / ω) = 0) ∧
  (∀ x, f (x + π/6) = f (π/6 - x))

theorem problem1 (f : ℝ → ℝ) : Px f → 
  (∀ x, f x = 2 * Real.cos (2 * x + π / 3)) :=
sorry

noncomputable def Qx (f : ℝ → ℝ) (x : ℝ) : ℝ :=
(2 * Real.cos (2 * (x - π/6)) + 1) * (1/2 - 2 * Real.cos x) + 1 / 2

theorem problem2 (x : ℝ) :
  ∀ (x ∈ Icc (-3*π/4) (π/2)), 
  -1 ≤ Qx (λ x, 2 * Real.cos (2 * x + π / 3)) x ∧ Qx (λ x, 2 * Real.cos (2 * x + π / 3)) x ≤ (2*Real.sqrt 2 + 1)/2 :=
sorry

end problem1_problem2_l334_334738


namespace integral_sin5_over_cos5_eq_l334_334110

noncomputable def integral_of_sin5_over_cos5 (x : ℝ) : ℝ :=
  ∫ (y : ℝ), (sin(3 * y)^5 / cos(3 * y)^5)

theorem integral_sin5_over_cos5_eq (C : ℝ) : 
  ∀ (x : ℝ), integral_of_sin5_over_cos5 x = 
    (tan(3 * x)^4 / 12) - (tan(3 * x)^2 / 6) - (1 / 3) * log (abs (cos (3 * x))) + C :=
by
  sorry

end integral_sin5_over_cos5_eq_l334_334110


namespace rationalize_denominator_l334_334334

theorem rationalize_denominator (cbrt : ℝ → ℝ) (h₁ : cbrt 81 = 3 * cbrt 3) :
  1 / (cbrt 3 + cbrt 81) = cbrt 9 / 12 :=
sorry

end rationalize_denominator_l334_334334


namespace sum_of_products_formula_l334_334397

-- Define the sequence A_n
def A (n : ℕ) : List ℕ := List.range' 1 n |>.map (λ k, 2^k - 1)

-- Define T_k as the sum of products of all k-element subsets of A_n
noncomputable def T (k n : ℕ) : ℕ :=
(List.subsetsOfLen k (A n)).foldl (λ acc l, acc + (l.foldl (*) 1)) 0

-- Define S_n as the sum of T_1, T_2, ..., T_n
noncomputable def S (n : ℕ) : ℕ := (List.range' 1 (n + 1)).foldl (λ acc k, acc + T k n) 0

-- The theorem we need to prove
theorem sum_of_products_formula (n : ℕ) : S n = 2^(n * (n + 1) / 2) - 1 :=
by
  sorry

end sum_of_products_formula_l334_334397


namespace find_s_at_1_l334_334842

variable (t s : ℝ → ℝ)
variable (x : ℝ)

-- Define conditions
def t_def : t x = 4 * x - 9 := by sorry

def s_def : s (t x) = x^2 + 4 * x - 5 := by sorry

-- Prove the question
theorem find_s_at_1 : s 1 = 11.25 := by
  -- Proof goes here
  sorry

end find_s_at_1_l334_334842


namespace molecular_weight_of_4_moles_AlCl3_is_correct_l334_334960

/-- The atomic weight of aluminum (Al) is 26.98 g/mol. -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of chlorine (Cl) is 35.45 g/mol. -/
def atomic_weight_Cl : ℝ := 35.45

/-- A molecule of AlCl3 consists of 1 atom of Al and 3 atoms of Cl. -/
def molecular_weight_AlCl3 := (1 * atomic_weight_Al) + (3 * atomic_weight_Cl)

/-- The total weight of 4 moles of AlCl3. -/
def total_weight_4_moles_AlCl3 := 4 * molecular_weight_AlCl3

/-- We prove that the total weight of 4 moles of AlCl3 is 533.32 g. -/
theorem molecular_weight_of_4_moles_AlCl3_is_correct :
  total_weight_4_moles_AlCl3 = 533.32 :=
sorry

end molecular_weight_of_4_moles_AlCl3_is_correct_l334_334960


namespace length_BC_fraction_AD_l334_334332

theorem length_BC_fraction_AD {A B C D : Type} {AB BD AC CD AD BC : ℕ} 
  (h1 : AB = 4 * BD) (h2 : AC = 9 * CD) (h3 : AD = AB + BD) (h4 : AD = AC + CD)
  (h5 : B ≠ A) (h6 : C ≠ A) (h7 : A ≠ D) : BC = AD / 10 :=
by
  sorry

end length_BC_fraction_AD_l334_334332


namespace james_success_rate_decrease_l334_334812

theorem james_success_rate_decrease 
  (initial_success : ℕ) (initial_attempts : ℕ) (next_attempts : ℕ) (next_success_fraction : ℚ) :
  initial_success = 8 →
  initial_attempts = 15 →
  next_attempts = 16 →
  next_success_fraction = 1 / 2 →
  let new_success := initial_success + (next_success_fraction * ↑next_attempts).natAbs in
  let total_attempts := initial_attempts + next_attempts in
  let initial_rate := (initial_success : ℚ) / initial_attempts in
  let new_rate := (new_success : ℚ) / total_attempts in
  let rate_difference := new_rate - initial_rate in
  let percentage_difference := rate_difference * 100 in
  percentage_difference ≈ -2 := 
by
  sorry

end james_success_rate_decrease_l334_334812


namespace syllogism_correct_l334_334788

-- Hypotheses for each condition
def OptionA := "The first section, the second section, the third section"
def OptionB := "Major premise, minor premise, conclusion"
def OptionC := "Induction, conjecture, proof"
def OptionD := "Dividing the discussion into three sections"

-- Definition of a syllogism in deductive reasoning
def syllogism_def := "A logical argument that applies deductive reasoning to arrive at a conclusion based on two propositions assumed to be true"

-- Theorem stating that a syllogism corresponds to Option B
theorem syllogism_correct :
  syllogism_def = OptionB :=
by
  sorry

end syllogism_correct_l334_334788


namespace original_price_l334_334028

theorem original_price (P : ℝ) (h : P * 0.80 = 960) : P = 1200 :=
sorry

end original_price_l334_334028


namespace scores_are_correct_l334_334141

variable
  Andrey_score : ℝ
  Dima_score : ℝ
  Vanya_score : ℝ
  Sasha_score : ℝ

-- Conditions
axiom andrey_first : Andrey_score > Dima_score ∧ Andrey_score > Vanya_score ∧ Andrey_score > Sasha_score
axiom dima_second : Dima_score > Vanya_score ∧ Dima_score > Sasha_score
axiom vanya_third : Vanya_score > Sasha_score
axiom unique_scores : Andrey_score ≠ Dima_score ∧ Andrey_score ≠ Vanya_score ∧ Andrey_score ≠ Sasha_score ∧ Dima_score ≠ Vanya_score ∧ Dima_score ≠ Sasha_score ∧ Vanya_score ≠ Sasha_score
axiom total_points : Andrey_score + Dima_score + Vanya_score + Sasha_score = 12
axiom andrey_sasha_wins : Andrey_score = 4 ∧ Sasha_score = 2

-- Conclusion
theorem scores_are_correct :
  Andrey_score = 4 ∧ Dima_score = 3.5 ∧ Vanya_score = 2.5 ∧ Sasha_score = 2 :=
  sorry

end scores_are_correct_l334_334141


namespace sqrt_9_8_lt_pi_l334_334610

theorem sqrt_9_8_lt_pi : real.sqrt 9.8 < real.pi := by
  have h1 : 3.13 < real.sqrt 9.8,
  { sorry },
  have h2 : real.sqrt 9.8 < 3.14,
  { sorry },
  have h3 : 3.14 < real.pi,
  { sorry },
  linarith

end sqrt_9_8_lt_pi_l334_334610


namespace f_expression_l334_334915

def f : ℝ → ℝ := sorry    -- Define f as a function from real numbers to real numbers

axiom f_even : ∀ x : ℝ, f(-x) = f(x)  -- f is an even function
axiom f_property : ∀ x y : ℝ, f(x + y) = f(x) + f(y) - 2 * x * y - 1  -- Given functional equation

theorem f_expression : ∀ x : ℝ, f(x) = -x^2 + 1 := sorry

end f_expression_l334_334915


namespace line_parabola_intersection_one_point_l334_334385

theorem line_parabola_intersection_one_point (k : ℝ) :
  (∃ y : ℝ, (-3 * y^2 - 4 * y + 7 = k) ∧ ∀ y1 y2 : ℝ, ( 3 * y1^2 + 4 * y1 + (k - 7) = 0 → 3 * y2^2 + 4 * y2 + (k - 7) = 0 → y1 = y2)) ↔ (k = 25 / 3) :=
by
  sorry

end line_parabola_intersection_one_point_l334_334385


namespace line_parabola_intersection_one_point_l334_334384

theorem line_parabola_intersection_one_point (k : ℝ) :
  (∃ y : ℝ, (-3 * y^2 - 4 * y + 7 = k) ∧ ∀ y1 y2 : ℝ, ( 3 * y1^2 + 4 * y1 + (k - 7) = 0 → 3 * y2^2 + 4 * y2 + (k - 7) = 0 → y1 = y2)) ↔ (k = 25 / 3) :=
by
  sorry

end line_parabola_intersection_one_point_l334_334384


namespace least_five_digit_perfect_square_and_cube_l334_334502

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334502


namespace chess_tournament_distribution_l334_334174

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l334_334174


namespace value_of_x_in_equation_l334_334643

theorem value_of_x_in_equation :
  ∃ x : ℤ, (2010 + x)^2 = x^2 → x = -1005 :=
begin
  sorry
end

end value_of_x_in_equation_l334_334643


namespace inscribed_circle_radius_l334_334822

noncomputable def ellipse : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x^2) / 64 + (y^2) / 36 = 1 }

theorem inscribed_circle_radius 
    (M : ℝ × ℝ) 
    (hM : M ∈ ellipse) 
    (N : ℝ × ℝ) 
    (hN : M.1 = N.1 ∧ N.2 = 0) 
    (O : ℝ × ℝ) 
    (hO : O = (0, 0)) : 
    ∃ r : ℝ, r = sqrt 2 :=
sorry

end inscribed_circle_radius_l334_334822


namespace exists_quadratic_polynomial_with_property_l334_334092

theorem exists_quadratic_polynomial_with_property :
  ∃ (f : ℝ → ℝ), (∃ (a b c : ℤ), ∀ x, f x = (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)) ∧ f (f (real.sqrt 3)) = 0 :=
by
  sorry

end exists_quadratic_polynomial_with_property_l334_334092


namespace rectangle_diagonals_equal_l334_334613

-- Define the properties of a rectangle
def is_rectangle (AB CD AD BC : ℝ) (diagonal1 diagonal2 : ℝ) : Prop :=
  AB = CD ∧ AD = BC ∧ diagonal1 = diagonal2

-- State the theorem to prove that the diagonals of a rectangle are equal
theorem rectangle_diagonals_equal (AB CD AD BC diagonal1 diagonal2 : ℝ) (h : is_rectangle AB CD AD BC diagonal1 diagonal2) :
  diagonal1 = diagonal2 :=
by
  sorry

end rectangle_diagonals_equal_l334_334613


namespace other_juice_cost_l334_334578

theorem other_juice_cost (total_spent : ℕ := 94)
    (mango_cost_per_glass : ℕ := 5)
    (other_total_spent : ℕ := 54)
    (total_people : ℕ := 17) : 
  other_total_spent / (total_people - (total_spent - other_total_spent) / mango_cost_per_glass) = 6 := 
sorry

end other_juice_cost_l334_334578


namespace at_least_one_boy_selected_l334_334680

-- Define the number of boys and girls
def boys : ℕ := 6
def girls : ℕ := 2

-- Define the total group and the total selected
def total_people : ℕ := boys + girls
def selected_people : ℕ := 3

-- Statement: In any selection of 3 people from the group, the selection contains at least one boy
theorem at_least_one_boy_selected :
  ∀ (selection : Finset ℕ), selection.card = selected_people → selection.card > girls :=
sorry

end at_least_one_boy_selected_l334_334680


namespace exists_quadratic_polynomial_with_given_property_l334_334101

theorem exists_quadratic_polynomial_with_given_property :
  ∃ f : ℚ[X], degree f = 2 ∧ (∀ n : ℤ, (n : ℚ) ∈ f.coeff) ∧ f.eval (f.eval (real.sqrt 3)) = 0 :=
by
  sorry

end exists_quadratic_polynomial_with_given_property_l334_334101


namespace retail_women_in_LA_l334_334865

/-
Los Angeles has 6 million people living in it. If half the population is women 
and 1/3 of the women work in retail, how many women work in retail in Los Angeles?
-/

theorem retail_women_in_LA 
  (total_population : ℕ)
  (half_population_women : total_population / 2 = women_population)
  (third_women_retail : women_population / 3 = retail_women)
  : total_population = 6000000 → retail_women = 1000000 :=
by
  sorry

end retail_women_in_LA_l334_334865


namespace scores_are_correct_l334_334142

variable
  Andrey_score : ℝ
  Dima_score : ℝ
  Vanya_score : ℝ
  Sasha_score : ℝ

-- Conditions
axiom andrey_first : Andrey_score > Dima_score ∧ Andrey_score > Vanya_score ∧ Andrey_score > Sasha_score
axiom dima_second : Dima_score > Vanya_score ∧ Dima_score > Sasha_score
axiom vanya_third : Vanya_score > Sasha_score
axiom unique_scores : Andrey_score ≠ Dima_score ∧ Andrey_score ≠ Vanya_score ∧ Andrey_score ≠ Sasha_score ∧ Dima_score ≠ Vanya_score ∧ Dima_score ≠ Sasha_score ∧ Vanya_score ≠ Sasha_score
axiom total_points : Andrey_score + Dima_score + Vanya_score + Sasha_score = 12
axiom andrey_sasha_wins : Andrey_score = 4 ∧ Sasha_score = 2

-- Conclusion
theorem scores_are_correct :
  Andrey_score = 4 ∧ Dima_score = 3.5 ∧ Vanya_score = 2.5 ∧ Sasha_score = 2 :=
  sorry

end scores_are_correct_l334_334142


namespace fold_points_area_calculation_l334_334632

-- Definition of the isosceles right triangle with given conditions
def isosceles_right_triangle (A B C : Point) : Prop :=
  let AB := dist A B in
  let AC := dist A C in
  let BC := dist B C in
  is_right_angle C ∧
  is_isosceles_right_triangle A B C 36

-- Definition of fold points P where folding does not result in overlapping creases
def fold_points_area (A B C P : Point) (S : Set Point) : Prop :=
  fold_point P ∧
  no_overlapping_creases A B C P ∧
  fold_points_set A B C S = {P}

-- The area result expressed in the form q π - r √s
def area_expr (q r s : ℕ) : ℝ :=
  q * π - r * real.sqrt s

-- Main proof statement that calculates q+r+s for the given conditions
theorem fold_points_area_calculation (A B C : Point) (q r s : ℕ) :
  isosceles_right_triangle A B C →
  fold_points_area A B C P S →
  q * π - r * real.sqrt s = 81 * π - 162 * real.sqrt 2 →
  q + r + s = 245 :=
by
  sorry

end fold_points_area_calculation_l334_334632


namespace vertices_form_equilateral_triangle_l334_334103

noncomputable theory

-- Define the hexagon vertices and properties
structure Hexagon (α : Type*) [LinearOrderedField α] :=
  (A B C D E F : α)
  (angle_A : ℝ := 120)
  (angle_C : ℝ := 120)
  (angle_E : ℝ := 120)
  (sideAB : α)
  (sideAF : α)
  (sideBC : α)
  (sideCD : α)
  (sideDE : α)
  (sideEF : α)
  (sideAB_eq_sideAF : sideAB = sideAF)
  (sideBC_eq_sideCD : sideBC = sideCD)
  (sideDE_eq_sideEF : sideDE = sideEF)

-- The theorem to prove
theorem vertices_form_equilateral_triangle (α : Type*) [LinearOrderedField α] 
  (h : Hexagon α) : 
  let S := h.angle_A, T := h.angle_C, U := h.angle_E in
  S = 120 ∧ T = 120 ∧ U = 120 ∧ h.sideAB = h.sideAF ∧ h.sideBC = h.sideCD ∧ h.sideDE = h.sideEF → 
  true := 
by
  sorry

end vertices_form_equilateral_triangle_l334_334103


namespace arithmetic_sequence_value_l334_334794

theorem arithmetic_sequence_value (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0)) -- definition of arithmetic sequence
  (h2 : a 2 + a 10 = -12) -- given that a_2 + a_{10} = -12
  (h3 : a_2 = -6) -- given that a_6 is the average of a_2 and a_{10}
  : a 6 = -6 :=
sorry

end arithmetic_sequence_value_l334_334794


namespace eccentricity_ellipse_l334_334913

theorem eccentricity_ellipse : 
  let a := 5
  let b := 4
  let c := 3
  let e := c / a
  in e = 3 / 5 :=
by
  sorry

end eccentricity_ellipse_l334_334913


namespace win_sector_area_l334_334571

theorem win_sector_area (r : ℝ) (p : ℝ) (h_r : r = 6) (h_p : p = 1 / 3) : 
  let area_circle := π * r^2 in
  let area_win := p * area_circle in
  area_win = 12 * π :=
by 
  simp [h_r, h_p]
  -- Proof is omitted
  sorry

end win_sector_area_l334_334571


namespace least_five_digit_perfect_square_cube_l334_334520

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334520


namespace area_of_BDF_half_area_hexagon_l334_334067

open EuclideanGeometry

-- Definitions based on the problem conditions
structure HexagonInCircle (O A B C D E F : Point) (a b c : ℝ) :=
  (circ : Circle O)
  (hex : hexagon O A B C D E F circ)
  (AB_eq_BC : AB = BC)
  (CD_eq_DE : CD = DE)
  (EF_eq_FA : EF = FA)

def area_BDF_is_half_area_hexagon {O A B C D E F : Point} {a b c : ℝ}
  (h : HexagonInCircle O A B C D E F a b c) : Prop :=
  2 * (area (Triangle.mk B D F)) = area (hexagon O A B C D E F)

theorem area_of_BDF_half_area_hexagon {O A B C D E F : Point} {a b c : ℝ}
  (h : HexagonInCircle O A B C D E F a b c) :
  area_BDF_is_half_area_hexagon h :=
sorry

end area_of_BDF_half_area_hexagon_l334_334067


namespace polynomial_division_remainder_l334_334663

-- Define the polynomials f and g
def f : ℚ[X] := 3 * X^7 - 2 * X^5 + 5 * X^3 - 8
def g : ℚ[X] := X^2 + 3 * X + 2

-- Given conditions: g is factored as (X+1)(X+2)
theorem polynomial_division_remainder :
  ∃ (a b : ℚ), (∀ x : ℚ, f.eval x = ((X + 1) * (X + 2) * (lagrange_quadratic f g)) + a * x + b) := by
  sorry

end polynomial_division_remainder_l334_334663


namespace polar_coordinates_intersection_minimum_value_ratio_l334_334802

noncomputable def solvePolarCoordinates (ρ θ: ℝ): Prop :=
  let C1 := ρ * cos (θ + π / 4) = sqrt 2 / 2
  let C2 := ρ = 1 
  (0 ≤ θ ∧ θ ≤ π) 
  let M_polar := (ρ, θ)
  
  ρ = 1 ∧ θ = 0

noncomputable def minValueDistRatio (α: ℝ): Prop :=
  let C3 := ∀ ρ θ, 1 / (ρ ^ 2) = (cos(θ) ^ 2 / 3) + sin(θ) ^ 2
  let l := (α ≥ 0 ∧ α ≤ π)
  let t1 t2 := (2 * cos(α)) / (3 * sin(α)^2 + cos(α)^2)
                (2) / (3 * sin(α)^2 + cos(α)^2)
  let MA_MB := (2) / (3 * sin(α)^2 + cos(α)^2)
  let AB := sqrt ((2 * cos(α))^2 - 4)

  0 ≤ α ∧ α ≤ π →
  sqrt(1 + sin^2(α)) = (sqrt(6) / 6)

theorem polar_coordinates_intersection (ρ θ: ℕ) : solvePolarCoordinates ρ θ := by
  sorry

theorem minimum_value_ratio (α: ℕ) : minValueDistRatio α := by
  sorry

end polar_coordinates_intersection_minimum_value_ratio_l334_334802


namespace convert_and_find_chord_length_l334_334803

theorem convert_and_find_chord_length :
  (∀ ρ θ, ρ * sin (θ - π / 4) = sqrt 2 → y = ρ * sin θ ∧ x = ρ * cos θ → y = x + 2) ∧
  (∀ t, t ∈ Icc (-π / 2) (π / 2) → 
    (x = 2 * cos t) ∧ (y = 2 + 2 * sin t) → (x^2 + (y - 2)^2 = 4)) ∧
  (∃ ρ θ t, ρ * sin (θ - π / 4) = sqrt 2 ∧ t ∈ Icc (-π / 2) (π / 2) ∧ 
    (x = 2 * cos t) ∧ (y = 2 + 2 * sin t) →
    let d := 4 in d = 4) :=
sorry

end convert_and_find_chord_length_l334_334803


namespace beetle_movement_l334_334620

-- Define the checkerboard and the initial configuration
def checkerboard (n : ℕ) := Fin n × Fin n

-- Define the movement (from each node to the center of a cell)
def movement (n : ℕ) (initial final : checkerboard n → checkerboard n) :=
  ∀ (p q : checkerboard n), p ≠ q → 
  let d_initial := abs (initial p.1 - initial q.1) + abs (initial p.2 - initial q.2) in
  let d_final := abs (final p.1 - final q.1) + abs (final p.2 - final q.2) in
  d_final ≤ d_initial

-- State the theorem
theorem beetle_movement (n : ℕ)
  (initial final : checkerboard n → checkerboard n)
  (h_move : movement n initial final) :
  ∃ p : checkerboard n, final p = p := 
sorry

end beetle_movement_l334_334620


namespace triangle_problem_l334_334200

variables {A B C : ℝ} {a b c : ℝ}
variables {b_range : Set ℝ} [DecidablePred (λ x, x ∈ b_range)]

-- Part (I) definitions
def B_is_acute (B : ℝ) : Prop := B > 0 ∧ B < π / 2
def vectors_parallel (sinB cos2B cos_halfB : ℝ) : Prop :=
  2 * sinB * (2 * cos_halfB^2 - 1) + sqrt 3 * cos2B = 0

-- Part (II) definitions
def angle_condition (cosB : ℝ) : Prop := cosB = 1 / 2
def triangle_area (a c b B : ℝ) : ℝ := 1 / 2 * a * c * sin B

-- Proof statement
theorem triangle_problem 
  (B : ℝ) (sinB cos2B cos_halfB : ℝ) (b_val : ℝ) (a c R S : ℝ)
  (h_acwt : a^2 + c^2 ∈ b_range → a^2 + c^2 ≥ 2 * a * c)
  (h_Bacute : B_is_acute B)
  (h_vectors_parallel : vectors_parallel sinB cos2B cos_halfB)
  (h_b_range: b ∈ [sqrt 3, 2 * sqrt 3])
  (h_cosB: angle_condition (cos B))
  (h_b_val: b = 2) :
   (B = π / 3 ∧ R ∈ [1, 2] ∧ triangle_area a c 2 B ≤ sqrt 3) :=
  sorry

end triangle_problem_l334_334200


namespace car_speeds_l334_334636

-- Declare the speeds of each car as variables
variables (orange_speed green_speed red_speed grey_speed blue_speed pink_speed : ℝ)

-- Given conditions as hypotheses
namespace Dale_Car_Speeds

axiom H1 : orange_speed = 300
axiom H2 : green_speed = orange_speed * 0.5
axiom H3 : red_speed = green_speed * 1.25
axiom H4 : grey_speed = (green_speed + orange_speed) / 2
axiom H5 : blue_speed = grey_speed
axiom H6 : pink_speed = blue_speed - 50

-- Theorem stating the speeds
theorem car_speeds :
  orange_speed = 300 ∧
  green_speed = 150 ∧
  red_speed = 187.5 ∧
  grey_speed = 225 ∧
  blue_speed = 225 ∧
  pink_speed = 175 :=
by
  split;
  [exact H1,
  split;
  [calc green_speed = orange_speed * 0.5 : H2
                ... = 300 * 0.5 : by rw [H1]
                ... = 150 : by norm_num,
  split;
  [calc red_speed = green_speed * 1.25 : H3
              ... = (orange_speed * 0.5) * 1.25 : by rw [H2]
              ... = 150 * 1.25 : by norm_num
              ... = 187.5 : by norm_num,
  split;
  [calc grey_speed = (green_speed + orange_speed) / 2 : H4
                ... = (150 + 300) / 2 : by rw [H2, H1]
                ... = 450 / 2 : by norm_num
                ... = 225 : by norm_num,
  split;
  [exact H5, exact H6]]]] -- Ends the proofs for blue and pink cars

end Dale_Car_Speeds

end car_speeds_l334_334636


namespace distance_inequality_l334_334298

variables {A B C A₁ B₁ C₁ I H : Type} [MetricSpace Type]

-- Definitions for points being on sides and being bisectors
variables (A : Point) (B : Point) (C : Point)
variables (A₁ : Point) (B₁ : Point) (C₁ : Point)
variables (ABC : Triangle A B C)
variables (incenter : Incenter ABC = I)
variables (orthocenter : Orthocenter (Triangle A₁ B₁ C₁) = H)

axiom sides_A₁ : A₁ ∈ ← Segment BC
axiom sides_B₁ : B₁ ∈ ← Segment CA
axiom sides_C₁ : C₁ ∈ ← Segment AB

axiom bisector_A₁ : AA₁ ∈ AngleBisector ∠ A
axiom bisector_B₁ : BB₁ ∈ AngleBisector ∠ B
axiom bisector_C₁ : CC₁ ∈ AngleBisector ∠ C

-- The theorem to prove
theorem distance_inequality : dist A H + dist B H + dist C H ≥ dist A I + dist B I + dist C I := sorry

end distance_inequality_l334_334298


namespace share_per_person_in_dollars_l334_334352

-- Definitions based on conditions
def total_cost_euros : ℝ := 25 * 10^9  -- 25 billion Euros
def number_of_people : ℝ := 300 * 10^6  -- 300 million people
def exchange_rate : ℝ := 1.2  -- 1 Euro = 1.2 dollars

-- To prove
theorem share_per_person_in_dollars : (total_cost_euros * exchange_rate) / number_of_people = 100 := 
by 
  sorry

end share_per_person_in_dollars_l334_334352


namespace age_of_B_l334_334775

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 9) : B = 39 := by
  sorry

end age_of_B_l334_334775


namespace laticia_knitted_fewer_pairs_l334_334818

variable (first_week : ℕ) (second_week : ℕ) (third_week : ℕ) (fourth_week : ℕ) (total : ℕ)
variable (diff : ℤ)

-- Conditions
def conditions : Prop :=
  first_week = 12 ∧
  second_week = 12 + 4 ∧
  third_week = (first_week + second_week) / 2 ∧
  fourth_week = total - (first_week + second_week + third_week) ∧
  total = 57 ∧
  diff = abs (third_week - fourth_week)

-- The theorem statement
theorem laticia_knitted_fewer_pairs {first_week second_week third_week fourth_week total diff} :
  conditions first_week second_week third_week fourth_week total diff → 
  diff = 1 := 
by
  sorry

end laticia_knitted_fewer_pairs_l334_334818


namespace two_digit_number_is_54_l334_334047

theorem two_digit_number_is_54 
    (n : ℕ) 
    (h1 : 10 ≤ n ∧ n < 100) 
    (h2 : n % 2 = 0) 
    (h3 : ∃ (a b : ℕ), a * b = 20 ∧ 10 * a + b = n) : 
    n = 54 := 
by
  sorry

end two_digit_number_is_54_l334_334047


namespace total_money_collected_in_rupees_l334_334996

variable (n : ℕ) (p : ℕ)

def total_collection (n p : ℕ) : ℕ :=
  n * p

def total_collection_rupees (n p : ℕ) : ℚ :=
  (total_collection n p) / 100

theorem total_money_collected_in_rupees
  (h1 : n = 76)
  (h2 : p = 76) :
  total_collection_rupees n p = 5776 / 100 :=
by
  rw [h1, h2, total_collection_rupees, total_collection]
  norm_cast
  sorry

end total_money_collected_in_rupees_l334_334996


namespace rhombus_perimeter_is_52_l334_334041

def rhombus_perimeter (d1 d2 : ℝ) (half_diag1 half_diag2 : ℝ) : ℝ := 
  4 * real.sqrt (half_diag1^2 + half_diag2^2)

theorem rhombus_perimeter_is_52 (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  rhombus_perimeter d1 d2 (d1 / 2) (d2 / 2) = 52 :=
by
  rw [h1, h2]
  sorry

end rhombus_perimeter_is_52_l334_334041


namespace find_dot_product_l334_334239

noncomputable def dot_product_vector (a b : ℝ × ℝ) : ℝ :=
a.1 * b.1 + a.2 * b.2

theorem find_dot_product
  (a : ℝ × ℝ)
  (ha : ∥a∥ = 1)
  (b : ℝ × ℝ := (-2, 1))
  (hab : ∥a - b∥ = 2) :
  dot_product_vector a b = 1 :=
sorry

end find_dot_product_l334_334239


namespace factorize_poly1_factorize_poly2_factorize_poly3_factorize_poly4_l334_334107

-- Statements corresponding to the given problems

-- Theorem for 1)
theorem factorize_poly1 (a : ℤ) : 
  (a^7 + a^5 + 1) = (a^2 + a + 1) * (a^5 - a^4 + a^3 - a + 1) := 
by sorry

-- Theorem for 2)
theorem factorize_poly2 (a b : ℤ) : 
  (a^5 + a*b^4 + b^5) = (a + b) * (a^4 - a^3*b + a^2*b^2 - a*b^3 + b^4) := 
by sorry

-- Theorem for 3)
theorem factorize_poly3 (a : ℤ) : 
  (a^7 - 1) = (a - 1) * (a^6 + a^5 + a^4 + a^3 + a^2 + a + 1) := 
by sorry

-- Theorem for 4)
theorem factorize_poly4 (a x : ℤ) : 
  (2 * a^3 - a * x^2 - x^3) = (a - x) * (2 * a^2 + 2 * a * x + x^2) := 
by sorry

end factorize_poly1_factorize_poly2_factorize_poly3_factorize_poly4_l334_334107


namespace natalie_bushes_needed_l334_334104

theorem natalie_bushes_needed (b c p : ℕ) 
  (h1 : ∀ b, b * 10 = c) 
  (h2 : ∀ c, c * 2 = p)
  (target_p : p = 36) :
  ∃ b, b * 10 ≥ 72 :=
by
  sorry

end natalie_bushes_needed_l334_334104


namespace exists_star_with_more_stars_in_row_than_column_l334_334264

open Matrix

variable (m n : ℕ) (A : Matrix (Fin m) (Fin n) Bool)

def conditions := 
  m < n ∧ 
  (∀ j : Fin n, ∃ i : Fin m, A i j = true)

theorem exists_star_with_more_stars_in_row_than_column :
  conditions m n A →
  ∃ i j : Fin m, A i j = true ∧ (∑ j', (A i j') = true.count) > (∑ i', (A i' j) = true.count) :=
sorry

end exists_star_with_more_stars_in_row_than_column_l334_334264


namespace find_s_at_1_l334_334843

variable (t s : ℝ → ℝ)
variable (x : ℝ)

-- Define conditions
def t_def : t x = 4 * x - 9 := by sorry

def s_def : s (t x) = x^2 + 4 * x - 5 := by sorry

-- Prove the question
theorem find_s_at_1 : s 1 = 11.25 := by
  -- Proof goes here
  sorry

end find_s_at_1_l334_334843


namespace exists_quadratic_polynomial_with_given_property_l334_334099

theorem exists_quadratic_polynomial_with_given_property :
  ∃ f : ℚ[X], degree f = 2 ∧ (∀ n : ℤ, (n : ℚ) ∈ f.coeff) ∧ f.eval (f.eval (real.sqrt 3)) = 0 :=
by
  sorry

end exists_quadratic_polynomial_with_given_property_l334_334099


namespace min_value_inverse_sum_l334_334716

theorem min_value_inverse_sum (a b : ℝ) (h : a > 0) (k : b > 0) (hab : a + 2 * b = 1) : 
  ∃ (y : ℝ), y = 3 + 2 * Real.sqrt 2 ∧ (∀ x, x = (1 / a) + (1 / b) → y ≤ x) :=
sorry

end min_value_inverse_sum_l334_334716


namespace exists_quadratic_poly_with_integer_coeffs_l334_334096

theorem exists_quadratic_poly_with_integer_coeffs (α : ℝ) :
  (∃ (a b c : ℤ), ∀ x : ℝ, (λ x : ℝ, ↑a * x^2 + ↑b * x + ↑c) (↑(λ x : ℝ, ↑a * x^2 + ↑b * x + ↑c) (α)) = 0) :=
  sorry

end exists_quadratic_poly_with_integer_coeffs_l334_334096


namespace separate_commissions_l334_334310

open Finset

-- Define the problem setting
variables (Deputy : Type) [Fintype Deputy] [DecidableEq Deputy]

-- Define the enmity relationship (if a is an enemy of b, then b is an enemy of a)
variable (enmity : Deputy → Deputy → Prop)
variable [decidable_rel enmity]

-- Additional conditions
variables (n : ℕ)
variable (Hn : 1 ≤ n)
variable (Hsym : ∀ {a b : Deputy}, enmity a b → enmity b a)
variable (Hthree : ∀ a : Deputy, (univ.filter (enmity a)).card = 3)

-- The theorem we need to prove
theorem separate_commissions :
  ∃ (committee1 committee2 : Finset Deputy), 
  (committee1 ∪ committee2 = univ) ∧
  committee1.disjoint committee2 ∧
  (∀ a : Deputy, ((committee1 ∩ univ.filter (enmity a)).card ≤ 1) ∧
                 ((committee2 ∩ univ.filter (enmity a)).card ≤ 1)) :=
sorry

end separate_commissions_l334_334310


namespace count_divisors_between_6_and_15_l334_334566

theorem count_divisors_between_6_and_15 :
  let number_of_divisors : ℕ := (Finset.filter (λ x => x ≥ 6 ∧ x ≤ 15) (Finset.divisors 90)).card
  number_of_divisors = 4 :=
by
  sorry

end count_divisors_between_6_and_15_l334_334566


namespace Lin_peels_15_potatoes_l334_334241

-- Define the conditions
def total_potatoes : Nat := 60
def homer_rate : Nat := 2 -- potatoes per minute
def christen_rate : Nat := 3 -- potatoes per minute
def lin_rate : Nat := 4 -- potatoes per minute
def christen_join_time : Nat := 6 -- minutes
def lin_join_time : Nat := 9 -- minutes

-- Prove that Lin peels 15 potatoes
theorem Lin_peels_15_potatoes :
  ∃ (lin_potatoes : Nat), lin_potatoes = 15 :=
by
  sorry

end Lin_peels_15_potatoes_l334_334241


namespace maximize_profit_at_36_l334_334593

noncomputable def cost : ℝ → ℝ :=
  λ x, 300 + (1/12) * x^3 - 5 * x^2 + 170 * x

def price_per_unit : ℝ := 134

noncomputable def revenue (x : ℝ) : ℝ :=
  price_per_unit * x

noncomputable def profit (x : ℝ) : ℝ :=
  revenue x - cost x

theorem maximize_profit_at_36 :
  ∃ x : ℝ, (x = 36) ∧ (∀ y : ℝ, profit y ≤ profit 36) :=
sorry

end maximize_profit_at_36_l334_334593


namespace sum_of_squares_divided_by_one_plus_l334_334849

theorem sum_of_squares_divided_by_one_plus :
  ∀ (x : Fin 50 → ℝ),
    (∑ i, x i) = 0 →
    (∑ i, x i / (1 + x i)) = 1 →
    (∑ i, x i^2 / (1 + x i)) = 1 :=
by
  intros x h1 h2
  sorry

end sum_of_squares_divided_by_one_plus_l334_334849


namespace sequence_contains_integer_term_l334_334276

noncomputable def sequence (x : ℚ) : ℕ → ℚ
| 0     := x
| (n+1) := sequence n + 1 / ⌊sequence n⌋

theorem sequence_contains_integer_term (x0 : ℚ) (h : x0 > 1) : ∃ n : ℕ, ∃ m : ℤ, sequence x0 n = m :=
by
  sorry

end sequence_contains_integer_term_l334_334276


namespace boys_down_slide_l334_334011

theorem boys_down_slide (boys_1 boys_2 : ℕ) (h : boys_1 = 22) (h' : boys_2 = 13) : boys_1 + boys_2 = 35 := by
  sorry

end boys_down_slide_l334_334011


namespace function_equality_l334_334761

theorem function_equality (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, f (2 * x + 1) = 2 * x^2 + 1) :
  ∀ x : ℝ, f x = (1/2) * x^2 - x + (3/2) :=
by
  sorry

end function_equality_l334_334761


namespace least_five_digit_perfect_square_and_cube_l334_334480

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334480


namespace smallest_k_divisible_by_10_l334_334854

def largest_prime_with_n_digits (n : ℕ) : ℕ :=
  sorry  -- Assume we have a function that gives us the largest prime with n digits

noncomputable def p : ℕ := largest_prime_with_n_digits 2005

theorem smallest_k_divisible_by_10 :
  ∃ k : ℕ, k > 0 ∧ (p^2 - k) % 10 = 0 ∧ k = 5 :=
by 
  use 5
  split
  { sorry } -- proof that 5 > 0
  split
  { sorry } -- proof that (p^2 - 5) % 10 = 0
  { refl }  -- proof that k = 5

end smallest_k_divisible_by_10_l334_334854


namespace binkie_gemstones_l334_334077

noncomputable def gemstones_solution : ℕ :=
sorry

theorem binkie_gemstones : ∀ (Binkie Frankie Spaatz Whiskers Snowball : ℕ),
  Spaatz = 1 ∧
  Whiskers = Spaatz + 3 ∧
  Snowball = 2 * Whiskers ∧ 
  Snowball % 2 = 0 ∧
  Whiskers % 2 = 0 ∧
  Spaatz = (1 / 2 * Frankie) - 2 ∧
  Binkie = 4 * Frankie ∧
  Binkie + Frankie + Spaatz + Whiskers + Snowball <= 50 →
  Binkie = 24 :=
sorry

end binkie_gemstones_l334_334077


namespace seq_a3_l334_334697

noncomputable def seq : ℕ → ℕ
| 0     := 0 -- This placeholder is not used
| 1     := 3
| (n+1) := 2 * seq n + 1

theorem seq_a3 : seq 3 = 15 :=
by {
  sorry
}

end seq_a3_l334_334697


namespace part1_part2_l334_334808

open Real

def m : ℝ × ℝ := (sqrt 2 / 2, -sqrt 2 / 2)
def n (x : ℝ) : ℝ × ℝ := (cos x, sin x)

-- Part 1: Prove tan x = 1 if m ⊥ n and x ∈ (0, π/2)
theorem part1 (x : ℝ) (h1 : 0 < x ∧ x < π / 2) (h2 : (m.1 * (cos x) + m.2 * (sin x) = 0)) : tan x = 1 :=
by sorry

-- Part 2: Prove x = π/12 if the angle between m and n is π/3 and x ∈ (0, π/2)
theorem part2 (x : ℝ) (h1 : 0 < x ∧ x < π / 2) (h2 : (m.1 * (cos x) + m.2 * (sin x)) = cos (π / 3)) : x = π / 12 :=
by sorry

end part1_part2_l334_334808


namespace remainder_modulo_l334_334772

theorem remainder_modulo (n : ℤ) (h : n % 50 = 23) : (3 * n - 5) % 15 = 4 := 
by 
  sorry

end remainder_modulo_l334_334772


namespace max_value_sum_seq_l334_334281

theorem max_value_sum_seq : 
  ∃ a1 a2 a3 a4 : ℝ, 
    a1 = 0 ∧ 
    |a2| = |a1 - 1| ∧ 
    |a3| = |a2 - 1| ∧ 
    |a4| = |a3 - 1| ∧ 
    a1 + a2 + a3 + a4 = 2 := 
by 
  sorry

end max_value_sum_seq_l334_334281


namespace find_n_l334_334185

theorem find_n (n : ℕ) (h : nat.choose 12 n = nat.choose 12 (2 * n - 3)) : (n = 3) ∨ (n = 5) := by
  sorry

end find_n_l334_334185


namespace cos_B_find_b_l334_334306

theorem cos_B (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 7 * a = 3 * c) :
  Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 11 / 14 := by
  sorry

theorem find_b (a b c : ℝ) (h1 : 2 * b = a + c) (h2 : 7 * a = 3 * c)
  (area : ℝ := 15 * Real.sqrt 3 / 4)
  (h3 : (1/2) * a * c * Real.sin (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = area) :
  b = 5 := by
  sorry

end cos_B_find_b_l334_334306


namespace Paula_initial_cans_l334_334880

theorem Paula_initial_cans :
  ∀ (cans rooms_lost : ℕ), rooms_lost = 10 → 
  (40 / (rooms_lost / 5) = cans + 5 → cans = 20) :=
by
  intros cans rooms_lost h_rooms_lost h_calculation
  sorry

end Paula_initial_cans_l334_334880


namespace hiring_probabilities_l334_334019

def num_applicants : ℕ := 10
def applicants : fin num_applicants := fin.mk 0 sorry
def k_rank (k : ℕ) (n : ℕ) : Prop := k ∈ applicants
def hired_prob (k : ℕ) : ℝ := 
  if k = 10 then 1 / num_applicants! 
  else if k < num_applicants then sorry else 0

theorem hiring_probabilities :
  let A_k (k : ℕ) : ℕ := sorry in 
  (A_1 > A_2 > A_3 > A_4 > A_5 > A_6 > A_7 > A_8 = A_9 = A_10) ∧
  (hired_prob 1 + hired_prob 2 + hired_prob 3 > 0.7) ∧
  (hired_prob 8 + hired_prob 9 + hired_prob 10 < 0.1) :=
by
  sorry -- proof of the theorem

end hiring_probabilities_l334_334019


namespace graph_t_intersects_x_axis_exists_integer_a_with_integer_points_on_x_axis_intersection_l334_334323

open Real

def function_y (a x : ℝ) : ℝ := (4 * a + 2) * x^2 + (9 - 6 * a) * x - 4 * a + 4

theorem graph_t_intersects_x_axis (a : ℝ) : ∃ x : ℝ, function_y a x = 0 :=
by sorry

theorem exists_integer_a_with_integer_points_on_x_axis_intersection :
  ∃ (a : ℤ), 
  (∀ x : ℝ, (function_y a x = 0) → ∃ (x_int : ℤ), x = x_int) ∧ 
  (a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1) :=
by sorry

end graph_t_intersects_x_axis_exists_integer_a_with_integer_points_on_x_axis_intersection_l334_334323


namespace angle_in_fourth_quadrant_l334_334249
-- Import the necessary library

-- Definition of the conditions
def pos_cos (α : ℝ) := cos α > 0
def neg_tan (α : ℝ) := tan α < 0

-- Theorem statement
theorem angle_in_fourth_quadrant (α : ℝ) 
  (hcos : pos_cos α) 
  (htan : neg_tan α) : 
  ∃ n : ℤ, (π / 2 + n * π < α) ∧ (α < π + n * π) :=
sorry

end angle_in_fourth_quadrant_l334_334249


namespace minimum_third_highest_score_l334_334413

theorem minimum_third_highest_score (scores : Fin 6 → ℕ) (h_uniq : Function.Injective scores)
  (h_avg : (∑ i, scores i) = 555) (h_max : ∃ i, scores i = 99) 
  (h_min : ∃ i, scores i = 76) : 
  ∃ s, s = 95 ∧ 
    ∃ (i : Fin 6), scores i = s ∧ 
    ∃ (j : Fin 6), (i ≠ j) ∧ (scores j < scores i) ∧ 
    ∃ (k : Fin 6), (i ≠ k) ∧ (j ≠ k) ∧ (scores k < scores j) :=
  sorry

end minimum_third_highest_score_l334_334413


namespace find_value_of_a_l334_334899

theorem find_value_of_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_eq : 7 * a^2 + 14 * a * b = a^3 + 2 * a^2 * b) : a = 7 := 
sorry

end find_value_of_a_l334_334899


namespace gabrielle_total_crates_l334_334682

theorem gabrielle_total_crates (monday tuesday wednesday thursday : ℕ)
  (h_monday : monday = 5)
  (h_tuesday : tuesday = 2 * monday)
  (h_wednesday : wednesday = tuesday - 2)
  (h_thursday : thursday = tuesday / 2) :
  monday + tuesday + wednesday + thursday = 28 :=
by
  sorry

end gabrielle_total_crates_l334_334682


namespace unique_intersection_point_l334_334377

theorem unique_intersection_point (k : ℝ) :
x = k ->
∃ x : ℝ, x = -3*y^2 - 4*y + 7 -> ∃ k : ℝ, k = 25/3 -> y = 0 -> x = k

end unique_intersection_point_l334_334377


namespace symmetric_point_l334_334910

theorem symmetric_point (x y : ℝ) : 
  (x - 2 * y + 1 = 0) ∧ (y / x * 1 / 2 = -1) → (x = -2/5 ∧ y = 4/5) :=
by 
  sorry

end symmetric_point_l334_334910


namespace circle_intersection_length_l334_334991

theorem circle_intersection_length 
  (α : ℝ) (r R : ℝ) : 
  R > r → 
  let AB := 4 * real.cos (α / 2) * real.sqrt ((R - r) * (R * real.sin (α / 2)^2 + r * real.cos (α / 2)^2)) in 
  true := 
sorry

end circle_intersection_length_l334_334991


namespace least_five_digit_perfect_square_and_cube_l334_334462

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334462


namespace monotonically_decreasing_interval_l334_334926

theorem monotonically_decreasing_interval :
  (∀ x : ℝ, 3 < x → (log (0.2:ℝ) ((x^2 - 2*x) - 3) < 0.0)) := 
sorry

end monotonically_decreasing_interval_l334_334926


namespace simplest_form_eq_a_l334_334400

theorem simplest_form_eq_a (a : ℝ) (h : a ≠ 1) : 1 - (1 / (1 + (a / (1 - a)))) = a :=
by sorry

end simplest_form_eq_a_l334_334400


namespace product_of_abs_diff_of_squares_l334_334961

theorem product_of_abs_diff_of_squares (a b : ℕ) (h1 : a = 105) (h2 : b = 95) : 3 * abs (a^2 - b^2) = 6000 :=
by
  sorry

end product_of_abs_diff_of_squares_l334_334961


namespace parametric_eq_C1_and_rectangular_eq_C2_min_distance_C1_to_C2_l334_334280

-- Given conditions and questions rewritten into Lean 4 statements

def curve_C := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }
def transformation_ϕ : (ℝ × ℝ) → (ℝ × ℝ)
| (x, y) := (√3 * x, y)
def curve_C1 := transformation_ϕ '' curve_C

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * θ.cos, ρ * θ.sin)
def curve_C2_pol : ℝ × ℝ := { p : ℝ × ℝ | ∃ (ρ θ : ℝ), ρ * (θ + π/4).sin = 4√2 ∧ polar_to_cartesian ρ θ = p }
def curve_C2 := { p : ℝ × ℝ | p.1 + p.2 = 8 }

def min_distance (P : ℝ × ℝ) (L : ℝ × ℝ → Prop) : ℝ :=
Abs (L.1 * P.1 + L.2 * P.2 + L.3) / sqrt(L.1^2 + L.2^2)

theorem parametric_eq_C1_and_rectangular_eq_C2 : 
  (∀ (x y : ℝ), (x, y) ∈ curve_C1 ↔ ∃ α : ℝ, x = √3 * α.cos ∧ y = α.sin) ∧
  (∀ (x y : ℝ), (x, y) ∈ curve_C2_pol ↔ (x + y = 8)) := by
  sorry

theorem min_distance_C1_to_C2 :
  ∀ P : ℝ × ℝ, P ∈ curve_C1 → min_distance P curve_C2 = 3 * √2 := by
  sorry

end parametric_eq_C1_and_rectangular_eq_C2_min_distance_C1_to_C2_l334_334280


namespace chess_tournament_distribution_l334_334175

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l334_334175


namespace least_five_digit_perfect_square_and_cube_l334_334503

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334503


namespace solution_proof_l334_334345

noncomputable def proof_problem : Prop :=
  ∀ (x : ℝ), x ≠ 1 → (1 - 1 / (x - 1) = 2 * x / (1 - x)) → x = 2 / 3

theorem solution_proof : proof_problem := 
by
  sorry

end solution_proof_l334_334345


namespace chess_tournament_points_l334_334151

theorem chess_tournament_points
  (points : String → ℝ)
  (total_points : points "Andrey" + points "Dima" + points "Vanya" + points "Sasha" = 12)
  (distinct_points : 
    points "Andrey" ≠ points "Dima" ∧ 
    points "Andrey" ≠ points "Vanya" ∧ 
    points "Andrey" ≠ points "Sasha" ∧ 
    points "Dima" ≠ points "Vanya" ∧ 
    points "Dima" ≠ points "Sasha" ∧ 
    points "Vanya" ≠ points "Sasha")
  (order : 
    points "Andrey" > points "Dima" ∧ 
    points "Dima" > points "Vanya" ∧ 
    points "Vanya" > points "Sasha")
  (same_wins :
    let games_won (student : String) := (points student - 3) / 0.5 in
    games_won "Andrey" = games_won "Sasha") :
  points "Andrey" = 4 ∧ points "Dima" = 3.5 ∧ points "Vanya" = 2.5 ∧ points "Sasha" = 2 :=
by
  sorry

end chess_tournament_points_l334_334151


namespace chess_tournament_scores_l334_334134

theorem chess_tournament_scores :
    ∃ (A D V S : ℝ),
    A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S ∧
    A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 ∧
    A > D ∧ D > V ∧ V > S ∧
    (∃ (wins_A wins_S : ℕ), wins_A = wins_S) :=
begin
    sorry
end

end chess_tournament_scores_l334_334134


namespace cold_brew_cost_l334_334870

theorem cold_brew_cost :
  let drip_coffee_cost := 2.25
  let espresso_cost := 3.50
  let latte_cost := 4.00
  let vanilla_syrup_cost := 0.50
  let cappuccino_cost := 3.50
  let total_order_cost := 25.00
  let drip_coffee_total := 2 * drip_coffee_cost
  let lattes_total := 2 * latte_cost
  let known_costs := drip_coffee_total + espresso_cost + lattes_total + vanilla_syrup_cost + cappuccino_cost
  total_order_cost - known_costs = 5.00 →
  5.00 / 2 = 2.50 := by sorry

end cold_brew_cost_l334_334870


namespace population_never_dies_l334_334005

def x (p : ℝ) (q : ℝ) : ℝ :=
  if 0 ≤ p ∧ p ≤ 0.5 then 1
  else if 0.5 < p ∧ p ≤ 1 then q / p
  else 0

theorem population_never_dies (p q : ℝ) (h : p = 0.6) (hq : q = 0.4) : x p q = 2 / 3 := by
  have hp : 0.5 < p ∧ p ≤ 1 := by
    rw [h]
    norm_num
  have hq' : q = 0.4 := hq
  rw [x, if_neg, if_pos hp, hq']
  norm_num
  sorry -- Proof steps are not required in this task

end population_never_dies_l334_334005


namespace range_of_exponential_l334_334072

theorem range_of_exponential :
  ∀ x : ℝ, (0 ≤ x → 1 ≤ 3^x) ∧ (∃ y : ℝ, 3^y = x) :=
by
  sorry

end range_of_exponential_l334_334072


namespace least_five_digit_perfect_square_and_cube_l334_334505

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ ∃ k : ℕ, k^6 = n ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334505


namespace ab_value_l334_334350

theorem ab_value (a b : ℤ) (r s : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0)
  (h3 : (∃ r r' s : ℤ, r = r' ∧ r ≠ s) → has_roots x^3 + a * x^2 + b * x + 18 * b [r, r, s])
  (h4 : ∀ x, 0 = x^3 + a * x^2 + b * x + 18 * b → x ∈ ℤ) :
  |a * b| = 1440 := by
  sorry

end ab_value_l334_334350


namespace larger_K2_more_likely_related_l334_334800

-- Definition of our problem conditions
variable (x y : Type)

-- Assume K2 as a correlation index from x to y
variable (K2 : ℝ) 
variable (probability_related : ℝ → Prop) 
-- This phrase represents the likelihood that x is related to y based on K2's value

axiom correlation_index_K2 :
  ∀ k : ℝ, increases_likelihood (probability_related k) k

-- The theorem to be proved
theorem larger_K2_more_likely_related (k : ℝ) : 
  (probability_related k) :=
by
  sorry

end larger_K2_more_likely_related_l334_334800


namespace line_passing_through_intersection_parallel_l334_334660

theorem line_passing_through_intersection_parallel :
  (∃ x y : ℝ, 3 * x - 2 * y + 3 = 0 ∧ x + y - 4 = 0 ∧ 2 * x + y - 5 = 0) :=
begin
  sorry,
end

end line_passing_through_intersection_parallel_l334_334660


namespace find_ending_number_of_range_l334_334907

theorem find_ending_number_of_range :
  ∃ n : ℕ, (∀ avg_200_400 avg_100_n : ℕ,
    avg_200_400 = (200 + 400) / 2 ∧
    avg_100_n = (100 + n) / 2 ∧
    avg_100_n + 150 = avg_200_400) ∧
    n = 200 :=
sorry

end find_ending_number_of_range_l334_334907


namespace seating_circular_table_l334_334573

variable (V : Type) [Fintype V] [DecidableEq V]
variable (G : SimpleGraph V)
variable [Fintype (EdgeSet G)]
variable [DecidableRel G.Adj]

theorem seating_circular_table (P : Fintype (Fin 5))
  (h : ∀ (X : Finset (Fin 5)), X.card = 3 → ∃ (x y z : V), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (G.Adj x y ∨ G.Adj y z) ∧ (¬ G.Adj x y ∨ ¬ G.Adj y z)) :
  ∃ (C : Cycle G), ∀ (v : Fin 5), v ∈ C.support → ∃ (u : Fin 5), G.Adj v u :=
begin
  sorry
end

end seating_circular_table_l334_334573


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334494

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334494


namespace distance_between_closest_points_l334_334628

-- Definitions of the centers of the circles
def c₁ : ℝ × ℝ × ℝ := (3, 3, 3)
def c₂ : ℝ × ℝ × ℝ := (15, 12, 3)

-- Define the radius of both circles
def radius : ℝ := 3

-- Define the distance function between two points in 3D space
def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2 + (p.3 - q.3)^2)

-- Prove that the distance between the closest points of the two circles is 9
theorem distance_between_closest_points : distance c₁ c₂ - radius - radius = 9 :=
by
  sorry

end distance_between_closest_points_l334_334628


namespace count_paths_from_A_to_B_l334_334987

-- Define the types and basic structure of the path and the lattice.
noncomputable def hexagonal_lattice : Type := sorry

-- Define the conditions of the problem.
-- Each arrow in the lattice can be traveled only in the direction of the arrow.
-- The bug never travels the same segment more than once.
-- Path segments are marked with arrow colors.

structure PathSegment where
  start : hexagonal_lattice 
  end : hexagonal_lattice 
  direction : Prop  -- Represents travel direction constraint.
  color : string  -- Represents the color of the arrow.

-- Define the lattice structure and path conditions
def modified_hexagonal_lattice (s : PathSegment) : Prop := 
  s.direction → ¬(s = s) -- Never travel the same segment
  
-- Main theorem to prove the number of paths from A to B under the given conditions.
theorem count_paths_from_A_to_B : 
  ∃ (paths : ℕ), paths = 600 := by
  sorry

end count_paths_from_A_to_B_l334_334987


namespace expr_equals_1_203_l334_334652

def expr (b : ℝ) : ℝ :=
  (1 / 8) * (b ^ 0) + ((1 / (8 * b)) ^ 0) - (128 ^ (-1 / 3)) - ((-16) ^ (-2 / 3))

theorem expr_equals_1_203 {b : ℝ} (hb : b ≠ 0) : expr b = 1.203 :=
  by
    sorry

end expr_equals_1_203_l334_334652


namespace sequence_b_sum_l334_334699

noncomputable section

def sequence_a (n : ℕ) : ℕ := 2 * n + 1

def sequence_b (n : ℕ) : ℚ := (2 * n + 1) / (n ^ 2 * (2 * (n + 1) + 1 - 1) ^ 2)

def sum_first_n_terms_b (T : ℕ → ℚ) (S₂ : ℕ → ℚ) : Prop :=
  ∀ (n : ℕ), T n = (1 / 4) - (1 / (4 * (n + 1) ^ 2))

theorem sequence_b_sum (T : ℕ → ℚ) (n : ℕ) : 
  sum_first_n_terms_b T (λ n, (2 * n + 1)) := sorry

end sequence_b_sum_l334_334699


namespace problem_l334_334710

theorem problem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 + a*b*c = 4) : 
  a + b + c ≤ 3 := 
sorry

end problem_l334_334710


namespace inequality_abc_l334_334881

theorem inequality_abc 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a * b * c ≥ (b + c - a) * (a + c - b) * (a + b - c) := 
by 
  sorry

end inequality_abc_l334_334881


namespace least_five_digit_whole_number_is_perfect_square_and_cube_l334_334497

theorem least_five_digit_whole_number_is_perfect_square_and_cube :
  ∃ (n : ℕ), (10000 ≤ n ∧ n < 100000) ∧ (∃ (a : ℕ), n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_whole_number_is_perfect_square_and_cube_l334_334497


namespace hyperbola_focus_eq_m_l334_334720

theorem hyperbola_focus_eq_m (m : ℝ) (h : ∀ x y : ℝ, x ^ 2 / 9 - y ^ 2 / m = 1) (focus : (5, 0)) : m = 16 :=
sorry

end hyperbola_focus_eq_m_l334_334720


namespace jenna_stamp_division_l334_334813

theorem jenna_stamp_division (a b c : ℕ) (h₁ : a = 945) (h₂ : b = 1260) (h₃ : c = 630) :
  Nat.gcd (Nat.gcd a b) c = 105 :=
by
  rw [h₁, h₂, h₃]
  -- Now we need to prove Nat.gcd (Nat.gcd 945 1260) 630 = 105
  sorry

end jenna_stamp_division_l334_334813


namespace exists_attainable_graph_with_triangles_l334_334299

noncomputable def transformation (G : Type*) (x : G) : Type* := sorry

def is_G_attainable (G H : Type*) : Prop := sorry

theorem exists_attainable_graph_with_triangles (G : Type*) (n : ℕ) (h1 : ∃ (𝓝 : set G), fintype.card 𝓝 = 4 * n)
  (h2 : ∃ (E : set (G × G)), fintype.card E = n) (h3 : 4 ∣ n) :
  ∃ H : Type*, is_G_attainable G H ∧ (∃ triangles : set (set G), fintype.card triangles ≥ 9 * n^2 / 4) :=
sorry

end exists_attainable_graph_with_triangles_l334_334299


namespace sum_of_integer_solutions_l334_334538

theorem sum_of_integer_solutions :
  (∑ x in { x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 }, x) = 18 :=
by
  sorry

end sum_of_integer_solutions_l334_334538


namespace most_reasonable_sample_l334_334994

-- Define what it means to be a reasonable sample
def is_reasonable_sample (sample : String) : Prop :=
  sample = "D"

-- Define the conditions for each sample
def sample_A := "A"
def sample_B := "B"
def sample_C := "C"
def sample_D := "D"

-- Define the problem statement
theorem most_reasonable_sample :
  is_reasonable_sample sample_D :=
sorry

end most_reasonable_sample_l334_334994


namespace find_total_price_l334_334886

noncomputable def total_price (p : ℝ) : Prop := 0.20 * p = 240

theorem find_total_price (p : ℝ) (h : total_price p) : p = 1200 :=
by sorry

end find_total_price_l334_334886


namespace average_height_is_63_l334_334326

-- Define the heights of Parker, Daisy, Reese, and Giselle
def Reese_height : ℝ := 60
def Daisy_height : ℝ := Reese_height + 8
def Parker_height : ℝ := Daisy_height - 4
def Giselle_height : ℝ := Parker_height - 2

-- Total height and number of people
def total_height : ℝ := Reese_height + Daisy_height + Parker_height + Giselle_height
def number_of_people : ℝ := 4

-- Average height
def average_height : ℝ := total_height / number_of_people

-- Proof statement
theorem average_height_is_63.5 : average_height = 63.5 := by
  sorry

end average_height_is_63_l334_334326


namespace rooted_set_contains_diff_of_powers_of_two_is_all_integers_l334_334044

def is_rooted (S : Set ℤ) :=
  ∀ (n : ℕ) (a : Fin n → ℤ), (∀ x : ℤ, (∑ i in Finset.range n, a i * x^i) = 0 → x ∈ S)

def contains_diff_of_powers_of_two (S : Set ℤ) :=
  ∀ a b : ℕ, a > 0 → b > 0 → (2^a - 2^b) ∈ S

theorem rooted_set_contains_diff_of_powers_of_two_is_all_integers
    (S : Set ℤ) (h_rooted : is_rooted S) (h_contains : contains_diff_of_powers_of_two S) :
  S = Set.univ := 
sorry

end rooted_set_contains_diff_of_powers_of_two_is_all_integers_l334_334044


namespace complex_multiplication_l334_334630

theorem complex_multiplication :
  (1 - 2 * complex.i) * (3 + 4 * complex.i) * (-1 + complex.i) = -9 + 13 * complex.i :=
by sorry

end complex_multiplication_l334_334630


namespace g_n1_minus_g_nm1_l334_334640

def g (n : ℕ) : ℝ := (7 + 4 * Real.sqrt 7) / 14 * ((2 + Real.sqrt 7) / 3) ^ n + 
                      (7 - 4 * Real.sqrt 7) / 14 * ((2 - Real.sqrt 7) / 3) ^ n + 3

theorem g_n1_minus_g_nm1 (n : ℕ) : g (n + 1) - g (n - 1) = g n := 
by 
  -- Proof skipped for this problem
  sorry

end g_n1_minus_g_nm1_l334_334640


namespace chess_tournament_distribution_l334_334169

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l334_334169


namespace sum_of_series_l334_334857

theorem sum_of_series (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_gt : a > b) :
  ∑' n, 1 / ( ((n - 1) * a + (n - 2) * b) * (n * a + (n - 1) * b)) = 1 / ((a + b) * b) :=
by
  sorry

end sum_of_series_l334_334857


namespace solve_inequality_l334_334654

theorem solve_inequality (x : ℝ) : (x^2 + 7 * x < 8) ↔ x ∈ (Set.Ioo (-8 : ℝ) 1) := by
  sorry

end solve_inequality_l334_334654


namespace minimum_divisions_unique_l334_334992

-- Definition and conditions
def deck := fin 54
def division (n : ℕ) (d : list (list deck)) : Prop :=
  d.sum (λ l, l.length) = 54 ∧ (∀ l ∈ d, l.length ≤ n)

-- The statement to prove
theorem minimum_divisions_unique (n : ℕ) (d1 d2 d3 : list (list deck)) :
  (division n d1) → (division n d2) → (division n d3) →
  (∀ i j : deck, i ≠ j → (∃ k, (i ∈ d1[k] ∨ i ∈ d2[k] ∨ i ∈ d3[k]) ∧ (j ∈ d1[k] ∨ j ∈ d2[k] ∨ j ∈ d3[k])) → false) → 3 := 
sorry

end minimum_divisions_unique_l334_334992


namespace chess_tournament_solution_l334_334164

def chess_tournament_points (points : List ℝ) : Prop :=
  let andrey := points[0]
  let dima := points[1]
  let vanya := points[2]
  let sasha := points[3]
  andrey = 4 ∧ dima = 3.5 ∧ vanya = 2.5 ∧ sasha = 2

axiom chess_tournament_conditions (points : List ℝ) :
  -- Andrey secured first place, Dima secured second, Vanya secured third, and Sasha secured fourth.
  List.Nodup points ∧
  points.length = 4 ∧
  (∀ p, p ∈ points → p = 4 ∨ p = 3.5 ∨ p = 2.5 ∨ p = 2) ∧
  -- Andrey and Sasha won the same number of games.
  (points[0] ≠ points[1] ∧ points[0] ≠ points[2] ∧ points[0] ≠ points[3] ∧
   points[1] ≠ points[2] ∧ points[1] ≠ points[3] ∧
   points[2] ≠ points[3])

theorem chess_tournament_solution (points : List ℝ) :
  chess_tournament_conditions points → chess_tournament_points points :=
by
  sorry

end chess_tournament_solution_l334_334164


namespace exists_pentagonal_pyramid_intersects_hexagon_l334_334286

noncomputable def pentagonal_pyramid_intersects_hexagon : Prop :=
  ∃ (ABCDEF : set ℝ^3) (P : ℝ^3) (O : ℝ^3) (S S' : set ℝ^2),
    is_regular_hexagon ABCDEF ∧
    O = center_of ABCDEF ∧
    P ∉ S ∧
    P ∈ line_perpendicular_through O S ∧
    S' ≠ S ∧
    S'.contains_segment (A, B) ∧
    ∃ C' D' E' F' : ℝ^3,
      S'.intersects_line (line P C) C' ∧
      S'.intersects_line (line P D) D' ∧
      S'.intersects_line (line P E) E' ∧
      S'.intersects_line (line P F) F' ∧
      forms_pentagonal_pyramid P {G, C', D', E', F'}.intersects_hexagon S ABCDEF

theorem exists_pentagonal_pyramid_intersects_hexagon :
  pentagonal_pyramid_intersects_hexagon :=
sorry

end exists_pentagonal_pyramid_intersects_hexagon_l334_334286


namespace arithmetic_sequence_general_term_l334_334862

theorem arithmetic_sequence_general_term (a_n S_n : ℕ → ℕ) (d : ℕ) (a1 S1 S5 S7 : ℕ)
  (h1: a_n 3 = 5)
  (h2: ∀ n, S_n n = (n * (a1 * 2 + (n - 1) * d)) / 2)
  (h3: S1 = S_n 1)
  (h4: S5 = S_n 5)
  (h5: S7 = S_n 7)
  (h6: S1 + S7 = 2 * S5):
  ∀ n, a_n n = 2 * n - 1 :=
by
  sorry

end arithmetic_sequence_general_term_l334_334862


namespace least_five_digit_perfect_square_and_cube_l334_334487

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334487


namespace part_1_part_2_l334_334224

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem part_1 : f' 2 = 9 :=
by sorry

theorem part_2 :
  (∀ x, -∞ < x ∧ x < -1 → f' x > 0) ∧ 
  (∀ x, 1 < x ∧ x < ∞ → f' x > 0) ∧
  (∀ x, -1 < x ∧ x < 1 → f' x < 0) ∧
  f(1) = -2 ∧
  f(-1) = 2 :=
by sorry

end part_1_part_2_l334_334224


namespace triangle_obtuse_l334_334702

theorem triangle_obtuse (k : ℝ) (h : 0 < k) : 
  let a := 4 * k,
      b := 5 * k,
      c := 7 * k
  in a^2 + b^2 < c^2 :=
by
  let a := 4 * k
  let b := 5 * k
  let c := 7 * k
  calc
    a^2 + b^2 = (4 * k)^2 + (5 * k)^2 : by sorry
            ... = 16 * k^2 + 25 * k^2 : by sorry
            ... = 41 * k^2           : by sorry
    c^2 = (7 * k)^2                  : by sorry
       ... = 49 * k^2                : by sorry
    show 41 * k^2 < 49 * k^2         : by sorry

end triangle_obtuse_l334_334702


namespace women_in_retail_l334_334866

theorem women_in_retail (total_population : ℕ) (half_population : total_population / 2 = women_count) 
  (third_of_women_work_in_retail : women_count / 3 = women_retail_count) :
  women_retail_count = 1000000 :=
by
  let total_population := 6000000
  let women_count := total_population / 2
  let women_retail_count := women_count / 3
  have h1 : women_count = 3000000 := rfl
  have h2 : women_retail_count = 1000000 := by
     rw [h1]
     exact rfl
  exact h2

end women_in_retail_l334_334866


namespace quadratic_inequality_solution_l334_334939

theorem quadratic_inequality_solution :
  {x : ℝ | 2 * x ^ 2 - x - 3 > 0} = {x : ℝ | x > 3 / 2 ∨ x < -1} :=
sorry

end quadratic_inequality_solution_l334_334939


namespace max_value_sequence_l334_334263

noncomputable def a_n (n : ℕ) : ℝ := 16 * (1/2)^(n-1)
def b_n (n : ℕ) : ℝ := Real.log (a_n n) / Real.log 2
def S_n (n : ℕ) : ℝ := (∑ i in Finset.range n, b_n (i + 1))

theorem max_value_sequence (n : ℕ) : 
  (0 < n) → (n <= 9) → 
  (∀ k : ℕ, 1 ≤ k → k ≤ n → (S_n k / k ≤ S_n n / n)) := 
sorry

end max_value_sequence_l334_334263


namespace remaining_gas_volume_and_type_l334_334562

def given_hydrogen_volume := 30 -- in cm³
def given_oxygen_mass := 0.03 -- in grams
def standard_molar_volume := 22400 -- in cm³ per mole
def molar_mass_oxygen := 32 -- in g/mol

def moles_of_hydrogen := given_hydrogen_volume / standard_molar_volume
def moles_of_oxygen := given_oxygen_mass / molar_mass_oxygen

def hydrogen_to_oxygen_ratio := 2.0 / 1.0

def required_oxygen_for_hydrogen := moles_of_hydrogen / hydrogen_to_oxygen_ratio

def remaining_oxygen := moles_of_oxygen - required_oxygen_for_hydrogen

def remaining_oxygen_volume := remaining_oxygen * standard_molar_volume

-- The theorem we want to prove:
theorem remaining_gas_volume_and_type : 
  remaining_oxygen_volume = 6 :=
by 
  sorry

end remaining_gas_volume_and_type_l334_334562


namespace exists_same_color_points_one_meter_apart_l334_334932

-- Declare the colors as an enumeration
inductive Color
| red : Color
| black : Color

-- Define the function that assigns a color to each point in the plane
def color (point : ℝ × ℝ) : Color := sorry

-- The theorem to be proven
theorem exists_same_color_points_one_meter_apart :
  ∃ x y : ℝ × ℝ, x ≠ y ∧ dist x y = 1 ∧ color x = color y :=
sorry

end exists_same_color_points_one_meter_apart_l334_334932


namespace difference_local_values_l334_334958

theorem difference_local_values (n : ℕ) (h : n = 58408) :
  ∃ (d : ℕ), (d = 8) ∧ (8000 - 8 = 7992) :=
begin
  use 8,
  split,
  { refl, },
  { refl, },
end

end difference_local_values_l334_334958


namespace seq_nat_eq_n_l334_334199

theorem seq_nat_eq_n (a : ℕ → ℕ) (h_inc : ∀ n, a n < a (n + 1))
  (h_le : ∀ n, a n ≤ n + 2020)
  (h_div : ∀ n, a (n + 1) ∣ (n^3 * a n - 1)) :
  ∀ n, a n = n :=
by
  sorry

end seq_nat_eq_n_l334_334199


namespace overall_profit_percentage_correct_l334_334589

-- Define the initial cost prices and profit percentages
def CP_TV : ℕ := 16000
def CP_DVD : ℕ := 6250
def CP_HomeTheater : ℕ := 11500
def CP_GamingConsole : ℕ := 18500
def Profit_TV_Percent : ℚ := 30 / 100
def Profit_DVD_Percent : ℚ := 20 / 100
def Profit_HomeTheater_Percent : ℚ := 25 / 100
def Profit_GamingConsole_Percent : ℚ := 15 / 100

-- Calculate selling prices using the given profit percentages
def SP_TV : ℚ := CP_TV + (CP_TV * Profit_TV_Percent)
def SP_DVD : ℚ := CP_DVD + (CP_DVD * Profit_DVD_Percent)
def SP_HomeTheater : ℚ := CP_HomeTheater + (CP_HomeTheater * Profit_HomeTheater_Percent)
def SP_GamingConsole : ℚ := CP_GamingConsole + (CP_GamingConsole * Profit_GamingConsole_Percent)

-- Calculate total cost price and total selling price
def Total_CP : ℚ := CP_TV + CP_DVD + CP_HomeTheater + CP_GamingConsole
def Total_SP : ℚ := SP_TV + SP_DVD + SP_HomeTheater + SP_GamingConsole

-- Calculate overall profit and overall profit percentage
def Overall_Profit : ℚ := Total_SP - Total_CP
def Overall_Profit_Percent : ℚ := (Overall_Profit / Total_CP) * 100

-- The theorem statement
theorem overall_profit_percentage_correct : Overall_Profit_Percent ≈ 22.39 := by
  sorry

end overall_profit_percentage_correct_l334_334589


namespace least_five_digit_perfect_square_and_cube_l334_334456

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334456


namespace sum_ge_n_l334_334194

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ (k : ℕ), k ≥ 1 → a (k + 1) ≥ (k * a k) / (a k ^ 2 + k - 1)

theorem sum_ge_n (a : ℕ → ℝ) (h : seq a) (n : ℕ) (hn : n ≥ 2) : 
  ∑ i in Finset.range n, a (i + 1) ≥ n :=
sorry

end sum_ge_n_l334_334194


namespace solve_triangle_l334_334268

variables {A B C : ℝ}
variables {a b c : ℝ}
variables m n : ℝ × ℝ

noncomputable def angles (A B C : ℝ) : Prop :=
  (0 < A ∧ A < π / 2) ∧ (0 < B ∧ B < π / 2) ∧ (0 < C ∧ C < π / 2)

theorem solve_triangle
  (h1 : a^2 - a * b = c^2 - b^2)
  (h2 : sqrt 3 * (tan A - tan B) = 1 + tan A * tan B)
  (h3 : angles A B C)
  (m_def : m = (sin A, cos A))
  (n_def : n = (cos B, sin B))
  : A = 5 * π / 12 ∧ B = π / 4 ∧ C = π / 3 ∧
    1 ≤ |3 * m - 2 * n| ∧ |3 * m - 2 * n| < sqrt 7 :=
sorry

end solve_triangle_l334_334268


namespace find_AB_length_l334_334618

noncomputable theory

def is_inradius (r : ℝ) (A B C : ℝ × ℝ) (D : ℝ × ℝ) : Prop :=
  let a := (A.1 - D.1)^2 + (A.2 - D.2)^2 in
  let b := (B.1 - D.1)^2 + (B.2 - D.2)^2 in
  let c := (C.1 - D.1)^2 + (C.2 - D.2)^2 in
  let s := (Math.sqrt a + Math.sqrt b + Math.sqrt c) / 2 in
  r = (s - Math.sqrt a) / s

def is_exradius (r : ℝ) (A B C : ℝ × ℝ) (D : ℝ × ℝ) : Prop :=
  let a := (A.1 - D.1)^2 + (A.2 - D.2)^2 in
  let b := (B.1 - D.1)^2 + (B.2 - D.2)^2 in
  let c := (C.1 - D.1)^2 + (C.2 - D.2)^2 in
  let s := (Math.sqrt a + Math.sqrt b + Math.sqrt c) / 2 in
  r = (Math.sqrt a * Math.sqrt b * Math.sqrt c) / (s * (s - Math.sqrt a))

theorem find_AB_length (A B C D : ℝ × ℝ) (h_acute : true) 
  (AC_eq : (A.1 - C.1)^2 + (A.2 - C.2)^2 = 100) 
  (BC_eq : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 100)
  (D_on_AB : ∃ t : ℝ, t ∈ Icc 0 1 ∧ (D = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)))
  (inradius_eq : is_inradius 2 A C D)
  (exradius_eq : is_exradius 2 B C D) 
  : ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 80 :=
by sorry

end find_AB_length_l334_334618


namespace least_five_digit_perfect_square_cube_l334_334522

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334522


namespace cab_drivers_income_on_third_day_l334_334988

theorem cab_drivers_income_on_third_day
  (day1 day2 day4 day5 avg_income n_days : ℝ)
  (h_day1 : day1 = 600)
  (h_day2 : day2 = 250)
  (h_day4 : day4 = 400)
  (h_day5 : day5 = 800)
  (h_avg_income : avg_income = 500)
  (h_n_days : n_days = 5) :
  ∃ day3 : ℝ, (day1 + day2 + day3 + day4 + day5) / n_days = avg_income ∧ day3 = 450 :=
by
  sorry

end cab_drivers_income_on_third_day_l334_334988


namespace both_boys_and_girls_selected_probability_l334_334600

theorem both_boys_and_girls_selected_probability :
  let total_students := 5
  let boys := 2
  let girls := 3
  let selected_students := 3
  let total_ways := Nat.choose total_students selected_students
  let only_girls_ways := Nat.choose girls selected_students
  (1 - (only_girls_ways / total_ways : ℚ)) = (9 / 10 : ℚ) :=
by
  let total_students := 5
  let boys := 2
  let girls := 3
  let selected_students := 3
  let total_ways := Nat.choose total_students selected_students
  let only_girls_ways := Nat.choose girls selected_students
  have h : (only_girls_ways / total_ways : ℚ) = (1 / 10 : ℚ) := sorry
  have h1 : (1 - (only_girls_ways / total_ways : ℚ)) = (9 / 10 : ℚ) := by rw [h]; norm_num
  exact h1

end both_boys_and_girls_selected_probability_l334_334600


namespace hiker_total_distance_l334_334997

theorem hiker_total_distance :
  (∃ (d1 d2 d3 : ℕ),
    d1 = 18 ∧
    d2 = (let rate2 := 3 + 1 in let hours2 := (18 / 3) - 1 in rate2 * hours2) ∧
    d3 = 5 * 6 ∧
    d1 + d2 + d3 = 68) :=
by
  have d1 : ℕ := 18
  have rate2 : ℕ := 3 + 1
  have hours2 : ℕ := (18 / 3) - 1
  have d2 : ℕ := rate2 * hours2
  have d3 : ℕ := 5 * 6
  use [d1, d2, d3]
  split
  · exact rfl
  split
  · exact rfl
  split
  · exact rfl
  sorry

end hiker_total_distance_l334_334997


namespace probability_of_x_lt_y_l334_334591

-- Define the vertices of the rectangle
def vertices : list (ℝ × ℝ) := [(0, 0), (3, 0), (3, 2), (0, 2)]

-- Definition of a point being inside the rectangle
def in_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 2

-- Definition of the condition x < y
def condition (x y : ℝ) : Prop :=
  x < y

-- Calculate the probability
theorem probability_of_x_lt_y : 
  (∃ (x y : ℝ), in_rectangle x y ∧ condition x y) →
  (∃ (p : ℚ), p = 1/3) :=
sorry

end probability_of_x_lt_y_l334_334591


namespace sum_of_integer_solutions_l334_334539

theorem sum_of_integer_solutions :
  (∑ x in ({ x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 }.to_finset), x) = 18 :=
by
  sorry

end sum_of_integer_solutions_l334_334539


namespace monotonic_increasing_interval_of_sine_variant_l334_334388

theorem monotonic_increasing_interval_of_sine_variant :
  ∀ k : ℤ, ∀ x : ℝ,
  (k * (real.pi : ℝ) + 5 * (real.pi : ℝ)/ 12 ≤ x ∧ x ≤ k * (real.pi : ℝ) + 11 * (real.pi : ℝ) / 12) ↔
  (∃ n : ℤ,
  ((real.pi / 2 + 2 * n * real.pi) ≤ (real.pi / 3 - 2 * x) ∧
  (real.pi / 3 - 2 * x) ≤ (3 * real.pi / 2 + 2 * n * real.pi))) := sorry

end monotonic_increasing_interval_of_sine_variant_l334_334388


namespace min_sum_S12_l334_334273

noncomputable def minimum_sum_of_arithmetic_sequence (a : ℕ → ℝ) : ℝ :=
  if h : (∀ n, a n > 0) ∧ (a 3 * a 8 = 36)  -- remember Lean uses zero-indexing
  then 72 else 0

-- The theorem statement
theorem min_sum_S12 (a : ℕ → ℝ) 
  (h_pos: ∀ n, a n > 0)
  (h_prod: a 3 * a 8 = 36): 
  minimum_sum_of_arithmetic_sequence a = 72 :=
by
  unfold minimum_sum_of_arithmetic_sequence
  simp [h_pos, h_prod]
  sorry

end min_sum_S12_l334_334273


namespace find_α_l334_334828

noncomputable def problem_α_β (α β : ℂ) :=
  α - β ∈ ℝ ∧ α - β > 0 ∧ 2 * complex.I * (α + β) ∈ ℝ ∧ 2 * complex.I * (α + β) > 0 ∧ β = 4 + complex.I

theorem find_α (α β : ℂ) : problem_α_β α β → α = -4 + complex.I :=
by
  intro h
  sorry

end find_α_l334_334828


namespace least_five_digit_perfect_square_cube_l334_334521

theorem least_five_digit_perfect_square_cube :
  ∃ n : Nat, (10000 ≤ n ∧ n < 100000) ∧ (∃ m1 m2 : Nat, n = m1^2 ∧ n = m2^3 ∧ n = 15625) :=
by
  sorry

end least_five_digit_perfect_square_cube_l334_334521


namespace no_integer_solutions_for_equation_l334_334526

theorem no_integer_solutions_for_equation (x y : ℤ) : ¬ (2 ^ (2 * x) - 3 ^ (2 * y) = 35) :=
by 
  sorry

end no_integer_solutions_for_equation_l334_334526


namespace perpendicular_groups_l334_334614

def dot_product (u v : (ℝ × ℝ × ℝ)) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem perpendicular_groups :
  dot_product (3, 4, 0) (0, 0, 5) = 0 ∧
  dot_product (-2, 1, 2) (4, -6, 7) = 0 ∧
  dot_product (3, 1, 3) (1, 0, -1) = 0 →
  dot_product (6, 0, 12) (6, -5, 7) ≠ 0 :=
by
  sorry

end perpendicular_groups_l334_334614


namespace pq_parallel_bc_l334_334821

-- Definitions of the geometric elements
variable {α : Type*}
variables (A B C : α) (Γ : circle α)
variables (M : α)
variables (A1 B1 C1 P Q : α)

-- Conditions
variables [PointOnBisector (angle A)] -- Point M lies on the bisector of angle A
variables [SecondIntersection A1 (AM) Γ]
variables [SecondIntersection B1 (BM) Γ]
variables [SecondIntersection C1 (CM) Γ]
variables [IntersectionPoint P (segment AB) (segment A1C1)]
variables [IntersectionPoint Q (segment AC) (segment A1B1)]

-- Statement of the theorem
theorem pq_parallel_bc :
  is_parallel (segment PQ) (segment BC) :=
sorry

end pq_parallel_bc_l334_334821


namespace mate_time_to_run_down_escalator_l334_334322

-- Define the time it takes Máté to stand on the moving escalator
def t1 : ℝ := 1.5 -- minutes

-- Define the time it takes Máté to run down the stationary stairs
def t2 : ℝ := 1.0 -- minutes

-- Goal: Prove that the time Máté takes to descend the moving escalator while running is 36 seconds
theorem mate_time_to_run_down_escalator : Máté_time_when_running_down_moving_escalator t1 t2 = 36 :=
sorry

end mate_time_to_run_down_escalator_l334_334322


namespace chess_tournament_scores_l334_334133

theorem chess_tournament_scores :
    ∃ (A D V S : ℝ),
    A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S ∧
    A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 ∧
    A > D ∧ D > V ∧ V > S ∧
    (∃ (wins_A wins_S : ℕ), wins_A = wins_S) :=
begin
    sorry
end

end chess_tournament_scores_l334_334133


namespace annual_growth_rate_is_25_percent_l334_334954

-- Defining the initial and final number of trees
def T₀ : ℕ := 64000
def T₃ : ℕ := 125000

-- Defining the growth rate variable
variable a : ℝ

-- Stating the theorem
theorem annual_growth_rate_is_25_percent :
  (T₀ : ℝ) * (1 + a)^3 = T₃ → a = 1 / 4 :=
by
  intros h
  -- proof is skipped
  sorry

end annual_growth_rate_is_25_percent_l334_334954


namespace polar_coordinate_circle_center_l334_334804

theorem polar_coordinate_circle_center (θ : ℝ) : 
    (∃ ρ : ℝ, ρ = -2 * Real.cos θ) → (1, Real.pi) = ((-1 : ℝ), 0).to_polar :=
by
  sorry

end polar_coordinate_circle_center_l334_334804


namespace reflection_matrix_squared_identity_l334_334830

def Q : Matrix (Fin 2) (Fin 2) ℝ := -- reflection matrix over vector (4, -2)
  let a : ℝ := 4
  let b : ℝ := -2
  let n2 := a^2 + b^2
  ![![ (a^2 - b^2) / n2 , 2*a*b / n2],
    ![2*a*b / n2, (b^2 - a^2) / n2]]

theorem reflection_matrix_squared_identity : Q ⬝ Q = (1 : Matrix (Fin 2) (Fin 2) ℝ) :=
by
  sorry

end reflection_matrix_squared_identity_l334_334830


namespace allocation_methods_count_l334_334030

def number_of_allocation_methods (doctors nurses : ℕ) (hospitals : ℕ) (nurseA nurseB : ℕ) :=
  if (doctors = 3) ∧ (nurses = 6) ∧ (hospitals = 3) ∧ (nurseA = 1) ∧ (nurseB = 1) then 684 else 0

theorem allocation_methods_count :
  number_of_allocation_methods 3 6 3 2 2 = 684 :=
by
  sorry

end allocation_methods_count_l334_334030


namespace chess_tournament_points_distribution_l334_334181

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l334_334181


namespace chess_tournament_points_distribution_l334_334179

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l334_334179


namespace max_mass_of_grain_pile_l334_334584

constant length : ℝ := 8
constant width : ℝ := 5
constant max_angle : ℝ := 45
constant density : ℝ := 1200

theorem max_mass_of_grain_pile : 
  let height := width / 2 in
  let prism_base_area := (width / 2 * real.sqrt(2)) / 2 * height in
  let prism_volume := prism_base_area * length in
  let pyramid_base_area := width * height in
  let pyramid_volume := pyramid_base_area * height / 3 in
  let total_volume := prism_volume + 2 * pyramid_volume in
  let max_mass := total_volume * density in
  max_mass = 47500 :=
sorry

end max_mass_of_grain_pile_l334_334584


namespace find_x_squared_plus_y_squared_l334_334694

variable (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : y > 0) 
(h2 : {x^2 + x + 1, -x, -x - 1} = {-y, -y / 2, y + 1}) : 
  x^2 + y^2 = 5 :=
sorry

end find_x_squared_plus_y_squared_l334_334694


namespace max_mass_of_grain_pile_l334_334585

constant length : ℝ := 8
constant width : ℝ := 5
constant max_angle : ℝ := 45
constant density : ℝ := 1200

theorem max_mass_of_grain_pile : 
  let height := width / 2 in
  let prism_base_area := (width / 2 * real.sqrt(2)) / 2 * height in
  let prism_volume := prism_base_area * length in
  let pyramid_base_area := width * height in
  let pyramid_volume := pyramid_base_area * height / 3 in
  let total_volume := prism_volume + 2 * pyramid_volume in
  let max_mass := total_volume * density in
  max_mass = 47500 :=
sorry

end max_mass_of_grain_pile_l334_334585


namespace total_weight_of_candy_in_14_bags_l334_334981

noncomputable def avg_weight := 90.2
noncomputable def num_bags := 14
noncomputable def weight_upper_bound := 90.25
noncomputable def weight_lower_bound := 90.15

theorem total_weight_of_candy_in_14_bags :
  (num_bags * weight_lower_bound <= W) ∧ (W < num_bags * weight_upper_bound) ∧ (∃ (total_weight : ℕ), total_weight = W) → W = 1263 :=
begin
  sorry
end

end total_weight_of_candy_in_14_bags_l334_334981


namespace problem1_problem2_l334_334744

-- Define the function f(x) for the conditions stated.
def f1 (x : ℝ) : ℝ := |x + 1| + |2 * x - 1|
def f2 (x m : ℝ) : ℝ := |x + m| + |2 * x - 1|

-- Define the first proof problem
theorem problem1 (x : ℝ) : f1 x ≥ 3 ↔ x ≤ -1 ∨ x ≥ 1 :=
by
  sorry

-- Define the range of values for m in the second proof problem
theorem problem2 (m x : ℝ) (h1 : 0 < m) (h2 : m < 1 / 4) (h3 : x ∈ set.Icc m (2 * m)) :
  (1 / 2) * f2 x m ≤ |x + 1| :=
by
  sorry

end problem1_problem2_l334_334744


namespace fruits_left_after_dog_eats_l334_334677

theorem fruits_left_after_dog_eats :
  (let apples_on_tree := 5 in
   let oranges_on_tree := 7 in
   let apples_on_ground := 8 in
   let oranges_on_ground := 10 in
   let apples_eaten_by_dog := 3 in
   let oranges_eaten_by_dog := 2 in
   let total_apples := apples_on_tree + (apples_on_ground - apples_eaten_by_dog) in
   let total_oranges := oranges_on_tree + (oranges_on_ground - oranges_eaten_by_dog) in
   total_apples + total_oranges = 25) := 
  sorry

end fruits_left_after_dog_eats_l334_334677


namespace least_five_digit_is_15625_l334_334441

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334441


namespace diameter_exists_l334_334990

theorem diameter_exists :
  ∀ (k : ℕ), 
  (λ (circle : Type) (div_points : finset point) (arc_length : point → point → ℕ),
    finset.card div_points = 3 * k ∧
    (∀ (p p' : point), p ≠ p' → arc_length p p' ∈ {1, 2, 3}) ∧
    (finset.filter (λ (arc : ℕ), arc = 1) (finset.image (λ (p p' : point), arc_length p p') div_points)).card = k ∧
    (finset.filter (λ (arc : ℕ), arc = 2) (finset.image (λ (p p' : point), arc_length p p') div_points)).card = k ∧
    (finset.filter (λ (arc : ℕ), arc = 3) (finset.image (λ (p p' : point), arc_length p p') div_points)).card = k)
  → ∃ (p1 p2 : point), p1 ≠ p2 ∧ dist p1 p2 = diameter circle := 
begin
  sorry
end

end diameter_exists_l334_334990


namespace books_more_than_movies_l334_334409

theorem books_more_than_movies (books_count movies_count read_books watched_movies : ℕ) 
  (h_books : books_count = 10)
  (h_movies : movies_count = 6)
  (h_read_books : read_books = 10) 
  (h_watched_movies : watched_movies = 6) : 
  read_books - watched_movies = 4 := by
  sorry

end books_more_than_movies_l334_334409


namespace hcf_of_numbers_l334_334354

def lcm_factors (a b l : ℕ) : Prop := lcm a b = l

theorem hcf_of_numbers : 
  ∃ N1 N2,
    max N1 N2 = 600 ∧
    lcm_factors N1 N2 (11 * 12) ∧
    Nat.gcd N1 N2 = 12 :=
by
  sorry

end hcf_of_numbers_l334_334354


namespace cos_pi_over_6_minus_2alpha_l334_334305

open Real

noncomputable def tan_plus_pi_over_6 (α : ℝ) := tan (α + π / 6) = 2

theorem cos_pi_over_6_minus_2alpha (α : ℝ) 
  (h1 : π < α ∧ α < 2 * π) 
  (h2 : tan_plus_pi_over_6 α) : 
  cos (π / 6 - 2 * α) = 4 / 5 :=
sorry

end cos_pi_over_6_minus_2alpha_l334_334305


namespace find_cosine_l334_334189
open Real

noncomputable def alpha (α : ℝ) : Prop := 0 < α ∧ α < π / 2 ∧ sin α = 3 / 5

theorem find_cosine (α : ℝ) (h : alpha α) :
  cos (π - α / 2) = - (3 * sqrt 10) / 10 :=
by sorry

end find_cosine_l334_334189


namespace integer_part_divisible_by_112_l334_334309

def is_odd (n : ℕ) : Prop := n % 2 = 1
def not_divisible_by_3 (n : ℕ) : Prop := n % 3 ≠ 0

theorem integer_part_divisible_by_112
  (m : ℕ) (hm_pos : 0 < m) (hm_odd : is_odd m) (hm_not_div3 : not_divisible_by_3 m) :
  ∃ n : ℤ, 112 * n = 4^m - (2 + Real.sqrt 2)^m - (2 - Real.sqrt 2)^m :=
by
  sorry

end integer_part_divisible_by_112_l334_334309


namespace solution_of_inequality_l334_334940

theorem solution_of_inequality (x : ℝ) : -2 * x - 1 < -1 → x > 0 :=
by
  sorry

end solution_of_inequality_l334_334940


namespace least_five_digit_perfect_square_and_cube_l334_334473

theorem least_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, n = 5^6 ∧ (n >= 10000 ∧ n < 100000) ∧ (∃ k : ℕ, n = k^2) ∧ (∃ m : ℕ, n = m^3) :=
by
  use 15625
  split
  · exact pow_succ 5 6
  split
  · have h1: 10000 ≤ 5^6 := nat.le_of_eq (calc 5^6 = 15625 : rfl)
    have h2: 5^6 < 100000 := nat.lt_of_eq_of_lt (calc 5^6 = 15625 : rfl) (by decide)
    exact ⟨h1, h2⟩
  · use 125
    exact pow_two 125
  · use 25
    exact pow_three 25
  sorry

end least_five_digit_perfect_square_and_cube_l334_334473


namespace sum_modified_series_l334_334633

theorem sum_modified_series : 
  let T := ∑' n, (-1) ^ (n + 1) * (1 / 3 ^ (n * 2 - 2)) in 
  T = 5 / 78 :=
by
  sorry

end sum_modified_series_l334_334633


namespace z_hourly_rate_correct_l334_334556

noncomputable def z_hourly_rate : ℚ :=
let x_rate := (1 : ℚ) / 15 in  -- x's work per day
let y_rate := (1 : ℚ) / 10 in  -- y's work per day
let total_work_x_y_5_days := 5 * x_rate + 5 * y_rate in  -- Total work done by x and y in 5 days
let remaining_work := 1 - total_work_x_y_5_days in  -- Remaining work for z
let z_total_hours := 4 * 5 in  -- Total hours worked by z
let total_earnings := 450 + 600 in  -- Total earnings for the work
let proportion_z_work := remaining_work / 1 in  -- Proportion of the work done by z
let z_earnings := total_earnings * proportion_z_work in  -- Earnings for z
z_earnings / z_total_hours -- Z's hourly rate

theorem z_hourly_rate_correct : z_hourly_rate = 8.75 := by
  sorry

end z_hourly_rate_correct_l334_334556


namespace least_five_digit_is_15625_l334_334436

noncomputable def least_five_digit_perfect_square_and_cube : ℕ :=
  15625

theorem least_five_digit_is_15625 :
  (10000 ≤ least_five_digit_perfect_square_and_cube) ∧ (least_five_digit_perfect_square_and_cube < 100000) ∧
  (∃ a : ℕ, least_five_digit_perfect_square_and_cube = a ^ 6) :=
by
  sorry

end least_five_digit_is_15625_l334_334436


namespace least_five_digit_perfect_square_and_cube_l334_334435

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334435


namespace tessellation_triangles_and_squares_no_tessellation_square_and_pentagon_no_tessellation_pentagon_and_hexagon_no_tessellation_hexagon_and_octagon_l334_334612

def interior_angle (n : ℕ) : ℝ := (n - 2) * 180 / n

def can_tessellate (angles : List ℝ) : Prop :=
  angles.sum = 360

theorem tessellation_triangles_and_squares :
  can_tessellate [interior_angle 3, interior_angle 3, interior_angle 3, interior_angle 4, interior_angle 4] :=
by
  sorry

theorem no_tessellation_square_and_pentagon :
  ¬ can_tessellate [interior_angle 4, interior_angle 5] :=
by
  sorry

theorem no_tessellation_pentagon_and_hexagon :
  ¬ can_tessellate [interior_angle 5, interior_angle 6] :=
by
  sorry

theorem no_tessellation_hexagon_and_octagon :
  ¬ can_tessellate [interior_angle 6, interior_angle 8] :=
by
  sorry

end tessellation_triangles_and_squares_no_tessellation_square_and_pentagon_no_tessellation_pentagon_and_hexagon_no_tessellation_hexagon_and_octagon_l334_334612


namespace common_difference_l334_334787

variable {a : ℕ → ℤ} -- Define the arithmetic sequence

theorem common_difference (h : a 2015 = a 2013 + 6) : 
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 3 := 
by
  use 3
  sorry

end common_difference_l334_334787


namespace problem1_problem2_l334_334063

theorem problem1 : (-1: ℝ)^4 - (2 - 3)^2 * (-2)^3 = 7 := 
by 
  -- sorry to skip the proof
  sorry
  
theorem problem2 : |( Real.sqrt 2 - 2 : ℝ) | + Real.sqrt (4/9) - (8: ℝ) ^ (1/3) = (2/3: ℝ) - Real.sqrt 2 :=
by 
  -- sorry to skip the proof
  sorry

end problem1_problem2_l334_334063


namespace prism_volume_is_25_l334_334552

noncomputable def triangle_area (a b : ℝ) : ℝ := (1 / 2) * a * b

noncomputable def prism_volume (base_area height : ℝ) : ℝ := base_area * height

theorem prism_volume_is_25 :
  let a := Real.sqrt 5
  let base_area := triangle_area a a
  let volume := prism_volume base_area 10
  volume = 25 :=
by
  intros
  sorry

end prism_volume_is_25_l334_334552


namespace values_of_x_l334_334840

def f (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x (x : ℝ) : f (f x) = f x → x = 0 ∨ x = -2 ∨ x = 5 ∨ x = 6 :=
by {
  sorry
}

end values_of_x_l334_334840


namespace sum_of_first_five_terms_geo_seq_l334_334670

theorem sum_of_first_five_terms_geo_seq 
  (a : ℚ) (r : ℚ) (n : ℕ) 
  (h_a : a = 1 / 3)
  (h_r : r = 1 / 3)
  (h_n : n = 5) :
  (∑ i in Finset.range n, a * r^i) = 121 / 243 :=
by
  sorry

end sum_of_first_five_terms_geo_seq_l334_334670


namespace least_five_digit_perfect_square_and_cube_l334_334448

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334448


namespace circles_equal_or_tangent_l334_334836

theorem circles_equal_or_tangent (a b c : ℝ) 
  (h : (2 * a)^2 - 4 * (b^2 - c * (b - a)) = 0) : 
  a = b ∨ c = a + b :=
by
  -- Will fill the proof later
  sorry

end circles_equal_or_tangent_l334_334836


namespace diameter_of_phi_l334_334201

theorem diameter_of_phi (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ Φ : ℝ, Φ = (a + b + c) / 2 :=
by
  use (a + b + c) / 2
  sorry

end diameter_of_phi_l334_334201


namespace lottery_systematic_l334_334780

variable (number_set : Finset ℕ)
variable (n : ℕ)
variable (ends_in_23 : ℕ → Prop)
variable (systematic_sampling : Finset ℕ → Prop)

-- Conditions
def ends_in_23 (x : ℕ) := x % 100 = 23
def number_set := { n | n ≤ 99999 }
def systematic_sampling (s : Finset ℕ) := ∃ k, ∀ m, m ∈ s → m % k = 23

theorem lottery_systematic :
  (∀ x ∈ number_set, ends_in_23 x) → systematic_sampling number_set :=
by
  sorry

end lottery_systematic_l334_334780


namespace percentage_increase_in_lines_l334_334970

/-- Given an increase of 110 lines resulting in a total of 240 lines,
    the percentage increase in the number of lines is approximately 84.62%. -/
theorem percentage_increase_in_lines (L : ℕ) 
  (h1 : L + 110 = 240) : 
  (110 : ℤ) / L * 100 ≈ 84.62 :=
by 
  sorry

end percentage_increase_in_lines_l334_334970


namespace sum_of_integer_solutions_l334_334543

theorem sum_of_integer_solutions :
  (∑ x in ({ x : ℤ | 4 < (x - 3)^2 ∧ (x - 3)^2 < 36 }.to_finset), x) = 18 :=
by
  sorry

end sum_of_integer_solutions_l334_334543


namespace proper_subsets_of_A_l334_334751

def A : Set ℕ := {1, 2, 3} -- We can use some integers to represent a, b, c

def properSubsetsCount (s : Set ℕ) : ℕ :=
  2 ^ s.toFinset.card - 1

theorem proper_subsets_of_A : properSubsetsCount A = 7 :=
by
  sorry

end proper_subsets_of_A_l334_334751


namespace converse_inverse_l334_334236

-- Define the properties
def is_parallelogram (polygon : Type) : Prop := sorry -- needs definitions about polygons
def has_two_pairs_of_parallel_sides (polygon : Type) : Prop := sorry -- needs definitions about polygons

-- The given condition
axiom parallelogram_implies_parallel_sides (polygon : Type) :
  is_parallelogram polygon → has_two_pairs_of_parallel_sides polygon

-- Proof of the converse:
theorem converse (polygon : Type) :
  has_two_pairs_of_parallel_sides polygon → is_parallelogram polygon := sorry

-- Proof of the inverse:
theorem inverse (polygon : Type) :
  ¬is_parallelogram polygon → ¬has_two_pairs_of_parallel_sides polygon := sorry

end converse_inverse_l334_334236


namespace increasing_interval_triangle_cos_c_l334_334735

def f (x : ℝ) := sqrt 3 * sin x * cos x + sin x ^ 2

theorem increasing_interval (k : ℤ) :
  ∀ x, k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3 →
  ∃ ε > 0, ∀ y, x < y → y < x + ε → f x ≤ f y := 
sorry

theorem triangle_cos_c (A B C a b c AD BD : ℝ) (hAD : AD = sqrt 2 * BD) (hADLength : AD = 2)
  (fA : f A = 3 / 2) (hAngleCond : 0 < A ∧ A < π / 2) (hA_def : A = π / 3) :
  cos C = (sqrt 6 - sqrt 2) / 4 :=
sorry

end increasing_interval_triangle_cos_c_l334_334735


namespace sum_slope_intercept_of_median_l334_334416

def midpoint (A B : Point) : Point :=
  ⟨(A.1 + B.1) / 2, (A.2 + B.2) / 2⟩

noncomputable def line_slope (A B : Point) : ℝ :=
  (B.2 - A.2) / (B.1 - A.1)

noncomputable def line_intercept (A : Point) (slope : ℝ) : ℝ :=
  A.2 - slope * A.1

noncomputable def calculate_sum_slope_intercept (A B : Point) : ℝ :=
  let slope := line_slope A B
  let intercept := line_intercept A slope
  slope + intercept

theorem sum_slope_intercept_of_median 
  (P Q R : Point)
  (hP : P = ⟨0, 6⟩) (hQ : Q = ⟨3, 0⟩) (hR : R = ⟨9, 0⟩) :
  calculate_sum_slope_intercept Q (midpoint P R) = -4 :=
by
  sorry

end sum_slope_intercept_of_median_l334_334416


namespace at_least_one_non_zero_l334_334389

theorem at_least_one_non_zero (a b : ℝ) : a^2 + b^2 > 0 ↔ (a ≠ 0 ∨ b ≠ 0) :=
by sorry

end at_least_one_non_zero_l334_334389


namespace derivative_y_by_x_l334_334658

def x (t : ℝ) : ℝ :=
  (sqrt (t - t ^ 2)) - arctan (sqrt ((1 - t) / t))

def y (t : ℝ) : ℝ :=
  (sqrt t) - (sqrt (1 - t)) * (arcsin (sqrt t))

theorem derivative_y_by_x (t : ℝ) (ht : 0 < t ∧ t < 1) :
  derivative (fun t => y t) t / derivative (fun t => x t) t = (sqrt t * (arcsin (sqrt t))) / (2 * (1 - t)) :=
by
  sorry

end derivative_y_by_x_l334_334658


namespace find_point_P_coordinates_l334_334272

noncomputable def coordinates_of_point (x y : ℝ) : Prop :=
  y > 0 ∧ x < 0 ∧ abs x = 4 ∧ abs y = 4

theorem find_point_P_coordinates : ∃ (x y : ℝ), coordinates_of_point x y ∧ (x, y) = (-4, 4) :=
by
  sorry

end find_point_P_coordinates_l334_334272


namespace complex_conjugate_of_z_l334_334769

theorem complex_conjugate_of_z (z : ℂ) (i : ℂ) (hi : i * i = -1) 
  (hz : (z - 3) * (2 - i) = 5) : conj z = 5 - i :=
sorry

end complex_conjugate_of_z_l334_334769


namespace more_girls_than_boys_l334_334411

theorem more_girls_than_boys (total_kids girls boys : ℕ) (h1 : total_kids = 34) (h2 : girls = 28) (h3 : total_kids = girls + boys) : girls - boys = 22 :=
by
  -- Proof placeholder
  sorry

end more_girls_than_boys_l334_334411


namespace find_ellipse_eq_a_b_find_AM_dot_AN_eq_zero_find_line_mn_isosceles_l334_334705

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) (e : ℝ) (pointA : ℝ × ℝ) : Prop :=
  pointA = (0, -1) ∧ e = (Real.sqrt (a^2 - b^2)) / a ∧ pointA ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

theorem find_ellipse_eq_a_b (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (condition : ellipse_equation a b h1 h2 (Real.sqrt 3 / 2) (0, -1)) :
  a = 2 ∧ b = 1 ∧ (∀ x y, (x^2 / 4 + y^2 = 1) ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}) :=
by
  sorry

theorem find_AM_dot_AN_eq_zero :
  ∀ (M N : ℝ × ℝ),
  (M ≠ N) ∧ (M.2 = fractional.line_through (0, 3/5) M.1) ∧ (N ≠ (0, -1)) ∧ (M ≠ (0, -1)) →
  (M.1 * N.1 + (M.2 + 1) * (N.2 + 1)) = 0 :=
by
  sorry

theorem find_line_mn_isosceles :
  ∀ (MN : ℝ × ℝ → ℝ × ℝ → Prop),
  (∀ (k : ℝ), MN (λ x, (x, k * x + 3/5))) ∧ isosceles_right_triangle (M, N, (0, -1)) →
  (MN = (λ x, (x, 3/5)) ∨ MN = (λ x, (√5 * x - 5 * y + 3)) ∨ MN = (λ x, (√5 * x + 5 * y - 3))) :=
by
  sorry

end find_ellipse_eq_a_b_find_AM_dot_AN_eq_zero_find_line_mn_isosceles_l334_334705


namespace range_of_k_l334_334219

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ :=
  sqrt 3 * sin (π * x / k)

-- Define the circle equation
def circle (k x y : ℝ) : Prop :=
  x^2 + y^2 = k^2

-- The Main Statement to Prove
theorem range_of_k (k : ℝ) :
  (∃ x, f k x = sqrt 3 ∧ circle k (x/2) (sqrt 3)) ∨
  (∃ x, f k x = -sqrt 3 ∧ circle k ((x+π)/(π/2/k)) (-sqrt 3)) →
  k ∈ (Iio (-2) ∪ Ici 2) :=
begin
  sorry
end

end range_of_k_l334_334219


namespace chess_tournament_scores_l334_334155

def points (name : String) := Real

def total_points : Real := 12

variables (A D V S : Real)
variable (total_games : ℕ := 12)

axiom different_scores : A ≠ D ∧ A ≠ V ∧ A ≠ S ∧ D ≠ V ∧ D ≠ S ∧ V ≠ S

axiom ranking : A > D ∧ D > V ∧ V > S

axiom equal_wins (A S : Real) : (A = 2 * win_points) ∧ (S = 2 * win_points)

axiom total_points_constraint : A + D + V + S = total_points

theorem chess_tournament_scores :
  A = 4 ∧ D = 3.5 ∧ V = 2.5 ∧ S = 2 :=
by 
  sorry

end chess_tournament_scores_l334_334155


namespace triangle_largest_angle_l334_334186

theorem triangle_largest_angle (A B C : ℝ) (h : sin A = sin B ∧ sin C = sqrt 2 * sin A) :
  (max A (max B C)) = π / 2 :=
by
  sorry

end triangle_largest_angle_l334_334186


namespace chess_tournament_points_l334_334145

theorem chess_tournament_points
  (points : String → ℝ)
  (total_points : points "Andrey" + points "Dima" + points "Vanya" + points "Sasha" = 12)
  (distinct_points : 
    points "Andrey" ≠ points "Dima" ∧ 
    points "Andrey" ≠ points "Vanya" ∧ 
    points "Andrey" ≠ points "Sasha" ∧ 
    points "Dima" ≠ points "Vanya" ∧ 
    points "Dima" ≠ points "Sasha" ∧ 
    points "Vanya" ≠ points "Sasha")
  (order : 
    points "Andrey" > points "Dima" ∧ 
    points "Dima" > points "Vanya" ∧ 
    points "Vanya" > points "Sasha")
  (same_wins :
    let games_won (student : String) := (points student - 3) / 0.5 in
    games_won "Andrey" = games_won "Sasha") :
  points "Andrey" = 4 ∧ points "Dima" = 3.5 ∧ points "Vanya" = 2.5 ∧ points "Sasha" = 2 :=
by
  sorry

end chess_tournament_points_l334_334145


namespace percentage_increase_l334_334983

theorem percentage_increase (i f : ℝ) (h_i : i = 500) (h_f : f = 650) :
    ((f - i) / i) * 100 = 30 := 
by 
  rw [h_i, h_f]
  rw [sub_self]
  rw [zero_div]
  linarith

end percentage_increase_l334_334983


namespace find_special_numbers_l334_334300

/-- Define the sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

/-- Define the main statement to be proved -/
theorem find_special_numbers :
  { n : ℕ | sum_of_digits n * (sum_of_digits n - 1) = n - 1 } = {1, 13, 43, 91, 157} :=
by
  sorry

end find_special_numbers_l334_334300


namespace least_five_digit_perfect_square_and_cube_l334_334465

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334465


namespace compute_expression_l334_334078

-- Define the operation a Δ b
def Delta (a b : ℝ) : ℝ := a^2 - 2 * b

theorem compute_expression :
  let x := 3 ^ (Delta 4 10)
  let y := 4 ^ (Delta 2 3)
  Delta x y = ( -819.125 / 6561) :=
by 
  sorry

end compute_expression_l334_334078


namespace range_of_m_l334_334730

/-- Given the curve \( C: y = \sqrt{-x^2 - 2x} \) and the line \( l: x + y - m = 0 \) intersect at two points,
find the range of values for \( m \). Prove that the range for \( m \) is \( 0 < m < \sqrt{2} - 1 \). -/
theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, y = real.sqrt (-x^2 - 2 * x) ∧ x + y = m ∧ y ≥ 0) ↔ (0 < m ∧ m < real.sqrt 2 - 1) := 
sorry

end range_of_m_l334_334730


namespace tangent_parallel_l334_334402

-- Define the curve function
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve function
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the slope of the line 4x - y - 1 = 0, which is 4
def line_slope : ℝ := 4

-- The main theorem statement
theorem tangent_parallel (a b : ℝ) (h1 : f a = b) (h2 : f' a = line_slope) :
  (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = -4) :=
sorry

end tangent_parallel_l334_334402


namespace solve_for_t_l334_334588

variables (V0 V g a t S : ℝ)

-- Given conditions
def velocity_eq : Prop := V = (g + a) * t + V0
def displacement_eq : Prop := S = (1/2) * (g + a) * t^2 + V0 * t

-- The theorem to prove
theorem solve_for_t (h1 : velocity_eq V0 V g a t)
                    (h2 : displacement_eq V0 g a t S) :
  t = 2 * S / (V + V0) :=
sorry

end solve_for_t_l334_334588


namespace number_base_5_representation_l334_334279

/-- In the base 5 numeral system, the number 53 in decimal is represented as 203, 
   which has three digits that are non-consecutive. -/
theorem number_base_5_representation : 
  ∃ (digits : List ℕ), 
    to_digits 5 53 = digits ∧ 
    List.length digits = 3 ∧ 
    (∀ i j, i ≠ j → (digits.nth i ≠ digits.nth j)) :=
sorry

end number_base_5_representation_l334_334279


namespace sin_plus_π_over_2_plus_2a_l334_334218

theorem sin_plus_π_over_2_plus_2a (a : ℝ) (y : ℝ) (hyp : (1 / 2) ^ 2 + y ^ 2 = 1) :
    sin (π / 2 + 2 * a) = -1 / 2 :=
sorry

end sin_plus_π_over_2_plus_2a_l334_334218


namespace find_k_intersects_parabola_at_one_point_l334_334373

theorem find_k_intersects_parabola_at_one_point :
  ∃ k : ℝ, (∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k ↔ y = (-4 / (2 * 3))) →
    k = 25 / 3 :=
by sorry

end find_k_intersects_parabola_at_one_point_l334_334373


namespace friends_payment_l334_334563

theorem friends_payment
  (num_friends : ℕ) (num_bread : ℕ) (cost_bread : ℕ) 
  (num_hotteok : ℕ) (cost_hotteok : ℕ) (total_cost : ℕ)
  (cost_per_person : ℕ)
  (h1 : num_friends = 4)
  (h2 : num_bread = 5)
  (h3 : cost_bread = 200)
  (h4 : num_hotteok = 7)
  (h5 : cost_hotteok = 800)
  (h6 : total_cost = num_bread * cost_bread + num_hotteok * cost_hotteok)
  (h7 : cost_per_person = total_cost / num_friends) :
  cost_per_person = 1650 := by
  sorry

end friends_payment_l334_334563


namespace problem_l334_334211

def log_function (x : ℝ) : ℝ := if 0 < x ∧ x < 1 then Real.log x else 0

def f (x : ℝ) : ℝ :=
if 0 < x && x < 1 then log_function x
else if x ≤ 0 then - f (- x)
else f (x - 3)

noncomputable def a : ℝ := f 1
noncomputable def b : ℝ := f 2
noncomputable def c : ℝ := f 3

theorem problem :
  c < a ∧ a < b :=
by
  sorry

end problem_l334_334211


namespace eq_3_solutions_l334_334853

theorem eq_3_solutions (p : ℕ) (hp : Nat.Prime p) :
  ∃! (x y : ℕ), (0 < x) ∧ (0 < y) ∧ ((1 / x) + (1 / y) = (1 / p)) ∧
  ((x = p + 1 ∧ y = p^2 + p) ∨ (x = p + p ∧ y = p + p) ∨ (x = p^2 + p ∧ y = p + 1)) :=
sorry

end eq_3_solutions_l334_334853


namespace fA_satisfies_inverse_negative_fB_satisfies_inverse_negative_fC_satisfies_inverse_negative_fD_satisfies_inverse_negative_l334_334995

def inverse_negative_transformation (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = -f x

def fA (x : ℝ) : ℝ := x - 1 / x
def fB (x : ℝ) : ℝ := x + 1 / x
def fC (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 1) then
    x
  else if (x = 1) then
    0
  else 
    -1 / x
def fD (x : ℝ) : ℝ := -x^3 + 1 / x^3

theorem fA_satisfies_inverse_negative : inverse_negative_transformation fA := sorry
theorem fB_satisfies_inverse_negative : ¬ inverse_negative_transformation fB := sorry
theorem fC_satisfies_inverse_negative : inverse_negative_transformation fC := sorry
theorem fD_satisfies_inverse_negative : inverse_negative_transformation fD := sorry

end fA_satisfies_inverse_negative_fB_satisfies_inverse_negative_fC_satisfies_inverse_negative_fD_satisfies_inverse_negative_l334_334995


namespace inequality_not_always_true_l334_334210

-- Declare the variables and conditions
variables {a b c : ℝ}

-- Given conditions
axiom h1 : a < b 
axiom h2 : b < c 
axiom h3 : a * c < 0

-- Statement of the problem
theorem inequality_not_always_true : ¬ (∀ a b c, (a < b ∧ b < c ∧ a * c < 0) → (c^2 / a < b^2 / a)) :=
by { sorry }

end inequality_not_always_true_l334_334210


namespace find_p_l334_334551

theorem find_p (m n p : ℝ) 
  (h1 : m = 3 * n + 5) 
  (h2 : m + 2 = 3 * (n + p) + 5) : p = 2 / 3 :=
by
  sorry

end find_p_l334_334551


namespace correct_option_l334_334544

theorem correct_option (a b c d : Prop)
  (hA : ∛(-6) = -∛6)
  (hB : ¬ (∃ x, ± (sqrt 16) = 4))
  (hC : ¬ (sqrt 25 = ± 5))
  (hD : ¬ (sqrt ((-3) ^ 2) = -3)) : a ↔ hA :=
by {
  sorry
}

end correct_option_l334_334544


namespace scores_are_correct_l334_334138

variable
  Andrey_score : ℝ
  Dima_score : ℝ
  Vanya_score : ℝ
  Sasha_score : ℝ

-- Conditions
axiom andrey_first : Andrey_score > Dima_score ∧ Andrey_score > Vanya_score ∧ Andrey_score > Sasha_score
axiom dima_second : Dima_score > Vanya_score ∧ Dima_score > Sasha_score
axiom vanya_third : Vanya_score > Sasha_score
axiom unique_scores : Andrey_score ≠ Dima_score ∧ Andrey_score ≠ Vanya_score ∧ Andrey_score ≠ Sasha_score ∧ Dima_score ≠ Vanya_score ∧ Dima_score ≠ Sasha_score ∧ Vanya_score ≠ Sasha_score
axiom total_points : Andrey_score + Dima_score + Vanya_score + Sasha_score = 12
axiom andrey_sasha_wins : Andrey_score = 4 ∧ Sasha_score = 2

-- Conclusion
theorem scores_are_correct :
  Andrey_score = 4 ∧ Dima_score = 3.5 ∧ Vanya_score = 2.5 ∧ Sasha_score = 2 :=
  sorry

end scores_are_correct_l334_334138


namespace electronics_weight_l334_334936

variable (B C E : ℝ)
variable (h1 : B / (B * (4 / 7) - 8) = 2 * (B / (B * (4 / 7))))
variable (h2 : C = B * (4 / 7))
variable (h3 : E = B * (3 / 7))

theorem electronics_weight : E = 12 := by
  sorry

end electronics_weight_l334_334936


namespace sum_of_special_right_triangle_areas_l334_334596

noncomputable def is_special_right_triangle (a b : ℕ) : Prop :=
  let area := (a * b) / 2
  area = 3 * (a + b)

noncomputable def special_right_triangle_areas : List ℕ :=
  [(18, 9), (9, 18), (15, 10), (10, 15), (12, 12)].map (λ p => (p.1 * p.2) / 2)

theorem sum_of_special_right_triangle_areas : 
  special_right_triangle_areas.eraseDups.sum = 228 := by
  sorry

end sum_of_special_right_triangle_areas_l334_334596


namespace exists_beautiful_connected_subset_l334_334777

-- Definitions reflecting conditions
structure City := (name : String)

structure TransportationSystem :=
  (cities : Set City)
  (direct_bus : City → City → Bool)
  (beautiful : Set City → Bool :=
    fun S =>
      (S.card ≥ 3) ∧ 
      (∀ A ∈ S, ∃ B C ∈ S, (B ≠ A ∧ C ≠ A) ∧ (direct_bus A B ∧ direct_bus A C)))

-- Main theorem statement
theorem exists_beautiful_connected_subset
  (S : Set City)
  (system : TransportationSystem)
  (hS : system.beautiful S) :
  ∃ T ⊆ S, system.beautiful T ∧ (∀ A B ∈ T, ∃ path : List City, path.head = A ∧ path.head? ≠ none ∧ path.last? = some B ∧ ∀ i < path.length - 1, system.direct_bus (path.get i) (path.get (i+1))) :=
sorry

end exists_beautiful_connected_subset_l334_334777


namespace rectangle_width_l334_334928

-- Definitions and Conditions
variables (L W : ℕ)

-- Condition 1: The perimeter of the rectangle is 16 cm
def perimeter_eq : Prop := 2 * (L + W) = 16

-- Condition 2: The width is 2 cm longer than the length
def width_eq : Prop := W = L + 2

-- Proof Statement: Given the above conditions, the width of the rectangle is 5 cm
theorem rectangle_width (h1 : perimeter_eq L W) (h2 : width_eq L W) : W = 5 := 
by
  sorry

end rectangle_width_l334_334928


namespace min_value_of_f_in_interval_l334_334925

noncomputable def f (x : Real) : Real := x + 2 * Real.cos x

theorem min_value_of_f_in_interval : 
  Real.Inf (Set.image f (Set.Icc (-Real.pi / 2) 0)) = -Real.pi / 2 :=
by
  sorry

end min_value_of_f_in_interval_l334_334925


namespace least_five_digit_perfect_square_and_cube_l334_334488

theorem least_five_digit_perfect_square_and_cube : 
  ∃ (n : ℕ), 10000 ≤ n ∧ n < 100000 ∧ (∃ a : ℕ, n = a ^ 6) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334488


namespace max_profit_l334_334608

noncomputable def y (x : ℕ) : ℕ :=
if h : 1 ≤ x ∧ x ≤ 35 then 800
else if h : 35 < x ∧ x ≤ 60 then -10 * x + 1150
else 0

noncomputable def Q (x : ℕ) : ℤ :=
if h : 1 ≤ x ∧ x ≤ 35 then 800 * x - 18000
else if h : 35 < x ∧ x ≤ 60 then -10 * (x : ℤ)^2 + 1150 * x - 18000
else 0

theorem max_profit :
  ∃ x, (35 < x ∧ x ≤ 60) ∧ (Q x = 15060) ∧ (∀ y, 1 ≤ y ∧ y ≤ 60 → Q y ≤ 15060) :=
begin
  sorry
end

end max_profit_l334_334608


namespace find_min_value_of_function_l334_334964

def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem find_min_value_of_function :
  ∀ x : ℝ, x > 1 → f x ≥ 3 := 
by
  intro x hx
  sorry

end find_min_value_of_function_l334_334964


namespace solution_set_f_lt_zero_a_two_solution_set_f_gt_zero_l334_334742

-- Given function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - (a - 1) * x - a

-- Problem 1: for a = 2, solution to f(x) < 0
theorem solution_set_f_lt_zero_a_two :
  { x : ℝ | f x 2 < 0 } = { x : ℝ | -1 < x ∧ x < 2 } :=
sorry

-- Problem 2: for any a in ℝ, solution to f(x) > 0
theorem solution_set_f_gt_zero (a : ℝ) :
  { x : ℝ | f x a > 0 } =
  if a > -1 then
    {x : ℝ | x < -1} ∪ {x : ℝ | x > a}
  else if a = -1 then
    {x : ℝ | x ≠ -1}
  else
    {x : ℝ | x < a} ∪ {x : ℝ | x > -1} :=
sorry

end solution_set_f_lt_zero_a_two_solution_set_f_gt_zero_l334_334742


namespace final_monotonicity_l334_334639

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ [3, 5] then
  1 - (x - 4)^2
else
  have ∀ n : ℤ, x + 2 * n ∈ [3, 5] → f (x + 2 * n) = 1 - (x + 2 * n - 4)^2 := sorry,
  f (x - 2 * (int.floor (x / 2)) + 2 * (int.floor (x / 2)))

lemma f_periodic : ∀ x : ℝ, f(x) = f(x + 2) := sorry

lemma interval_3_4_increasing : ∀ x y : ℝ, 3 ≤ x ∧ x < y ∧ y ≤ 4 → f(x) < f(y) := sorry

lemma interval_4_5_decreasing : ∀ x y : ℝ, 4 < x ∧ x < y ∧ y ≤ 5 → f(x) > f(y) := sorry

lemma interval_minus2_minus1_decreasing : ∀ x y : ℝ, -2 ≤ x ∧ x < y ∧ y ≤ -1 → f(x) > f(y) := sorry

lemma interval_5_6_increasing : ∀ x y : ℝ, 5 ≤ x ∧ x < y ∧ y ≤ 6 → f(x) < f(y) := sorry

-- Final statement to prove
theorem final_monotonicity : 
  (∀ x y : ℝ, -2 ≤ x ∧ x < y ∧ y ≤ -1 → f(x) > f(y)) ∧ 
  (∀ x y : ℝ, 5 ≤ x ∧ x < y ∧ y ≤ 6 → f(x) < f(y)) :=
begin
  split,
  exact interval_minus2_minus1_decreasing,
  exact interval_5_6_increasing,
end

end final_monotonicity_l334_334639


namespace sum_weights_greater_than_2p_l334_334251

variables (p x y l l' : ℝ)

-- Conditions
axiom balance1 : x * l = p * l'
axiom balance2 : y * l' = p * l

-- The statement to prove
theorem sum_weights_greater_than_2p : x + y > 2 * p :=
by
  sorry

end sum_weights_greater_than_2p_l334_334251


namespace maximum_sum_of_integers_l334_334965

theorem maximum_sum_of_integers 
  (a b c d e : ℤ) 
  (h_sorted : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h_median : c = 6)
  (h_mode : ∃ k, k = 8 ∧ ((a = k ∨ b = k ∨ c = k ∨ d = k ∨ e = k) ∧
            ((aa = bb ∨ b = c ∨ c = d ∨ d = e))) to List α) → c(i = k) ∧ ( ¬ 3 = RH )
  (mode :∅ ≤ 2 ∧  ¬ 2 = 0 ) :=  2 ) 
   ( max_blored : ∑to r = list.zip .max(.[rep]i = nk)
∧block  x) ) and 

     (a,

-- to: rest)
   31 .
sorry

end maximum_sum_of_integers_l334_334965


namespace hypotenuse_length_l334_334042

theorem hypotenuse_length 
  (A B C D E F : Point) 
  (x : ℝ)
  (hA : A ∈ triangle)
  (hB : B ∈ triangle)
  (hC : C ∈ triangle)
  (h hypotenuse (B C) : hypotenuse)
  (hD_quad : quadrisects B C D)
  (hE_quad : quadrisects B C E)
  (hF_quad : quadrisects B C F)
  (hAD : dist A D = sin x)
  (hAE : dist A E = cos x)
  (hAF : dist A F = tan x)
  (hx_range : 0 < x ∧ x < π / 2) :
  dist B C = (4 * sqrt 2) / 3 := 
sorry

end hypotenuse_length_l334_334042


namespace least_five_digit_perfect_square_and_cube_l334_334428

/-- 
Prove that the least five-digit whole number that is both a perfect square 
and a perfect cube is 15625.
-/
theorem least_five_digit_perfect_square_and_cube : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k : ℕ, n = k^6) ∧ (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ (∃ l : ℕ, m = l^6) → n ≤ m) :=
begin
  use 15625,
  split,
  { split,
    { norm_num },
    { norm_num } },
  { use 5,
    norm_num,
    intros m hm,
    rcases hm with ⟨⟨hm1, hm2⟩, ⟨l, rfl⟩⟩,
    cases le_or_lt l 5 with hl hl,
    { by_contradiction h,
      have : m < 10000 := by norm_num [nat.pow_succ, nat.pow_succ, hl],
      exact not_le_of_gt this hm1 },
    { have : l ≥ 6 := nat.lt_succ_iff.mp hl,
      have : m ≥ 6^6 := nat.pow_le_pow_of_le_right nat.zero_lt_succ_iff.1 this,
      norm_num at this,
      exact not_lt_of_ge this hm2 } }
end

end least_five_digit_perfect_square_and_cube_l334_334428


namespace range_of_k_eccentricity_l334_334839

theorem range_of_k_eccentricity (k : ℝ) (h_e : ∃ e : ℝ, e ∈ set.Ioo (1/2 : ℝ) 1 ∧ e^2 = if k > 4 then (k - 4) / k else (4 - k) / 4) :
  k ∈ set.Ioo 0 3 ∪ set.Ioi (16/3) :=
sorry

end range_of_k_eccentricity_l334_334839


namespace quadrilateral_is_rhombus_l334_334922

theorem quadrilateral_is_rhombus (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = ab + bc + cd + ad) : a = b ∧ b = c ∧ c = d :=
by
  sorry

end quadrilateral_is_rhombus_l334_334922


namespace simplify_and_evaluate_expression_l334_334339

variable (x y : ℝ)
variable (h1 : x = 1)
variable (h2 : y = Real.sqrt 2)

theorem simplify_and_evaluate_expression : 
  (x + 2 * y) ^ 2 - x * (x + 4 * y) + (1 - y) * (1 + y) = 7 := by
  sorry

end simplify_and_evaluate_expression_l334_334339


namespace problem_statement_l334_334844

def t (x : ℝ) : ℝ := 4 * x - 9
def s (y : ℝ) : ℝ := (Real.sqrtSq' (1 / 4 * (y + 9)))^2 + 4 * (Real.sqrtSq' (1 / 4 * (y + 9))) - 5

theorem problem_statement : s 1 = 11.25 := by
  sorry

end problem_statement_l334_334844


namespace find_distance_walker_l334_334029

noncomputable def distance_walked (x t d : ℝ) : Prop :=
  (d = x * t) ∧
  (d = (x + 1) * (3 / 4) * t) ∧
  (d = (x - 1) * (t + 3))

theorem find_distance_walker (x t d : ℝ) (h : distance_walked x t d) : d = 18 := 
sorry

end find_distance_walker_l334_334029


namespace geometric_sequence_b_n_formula_max_n_l334_334698

noncomputable theory

-- Definition of the sequence {a_n} and conditions
def sequence_a (n : ℕ) : ℕ := if n = 0 then 0 else 3 * 2^n - 1

-- Conditions on the sequence sums
def sequence_S (n : ℕ) : ℕ := (if n = 0 then 0 else (3 * (n - 1) * 2^(n + 1) - n * (n + 1) / 2 + 6))

-- Problem (1): Prove that the sequence {a_{n+1} + 1} is geometric
theorem geometric_sequence (n : ℕ) (hn : 0 < n) :
  sequence_a (n + 1) + 1 = 2 * (sequence_a n + 1) := sorry

-- Definition of the function f(x)
def f (x : ℕ → ℕ) (n : ℕ) : ℕ → ℕ := λ x, ∑ i in finset.range n, sequence_a i * (x ^ i)

-- Definition of b_n
def b_n (n : ℕ) : ℕ := f(λ x, 1) n

-- Problem (2): Find the general formula for {b_n}
theorem b_n_formula (n : ℕ) :
  b_n n = 3 * (n - 1) * 2^(n + 1) - n * (n + 1) / 2 + 6 := sorry

-- Problem (3): If b_n < 30, find the maximum value of n
theorem max_n (n : ℕ) (h : b_n n < 30) : n ≤ 2 := sorry

end geometric_sequence_b_n_formula_max_n_l334_334698


namespace chess_tournament_solution_l334_334161

def chess_tournament_points (points : List ℝ) : Prop :=
  let andrey := points[0]
  let dima := points[1]
  let vanya := points[2]
  let sasha := points[3]
  andrey = 4 ∧ dima = 3.5 ∧ vanya = 2.5 ∧ sasha = 2

axiom chess_tournament_conditions (points : List ℝ) :
  -- Andrey secured first place, Dima secured second, Vanya secured third, and Sasha secured fourth.
  List.Nodup points ∧
  points.length = 4 ∧
  (∀ p, p ∈ points → p = 4 ∨ p = 3.5 ∨ p = 2.5 ∨ p = 2) ∧
  -- Andrey and Sasha won the same number of games.
  (points[0] ≠ points[1] ∧ points[0] ≠ points[2] ∧ points[0] ≠ points[3] ∧
   points[1] ≠ points[2] ∧ points[1] ≠ points[3] ∧
   points[2] ≠ points[3])

theorem chess_tournament_solution (points : List ℝ) :
  chess_tournament_conditions points → chess_tournament_points points :=
by
  sorry

end chess_tournament_solution_l334_334161


namespace least_five_digit_perfect_square_and_cube_l334_334463

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ k1 : ℕ, n = k1 ^ 2) ∧ (∃ k2 : ℕ, n = k2 ^ 3) ∧ n = 15625 :=
by
  sorry

end least_five_digit_perfect_square_and_cube_l334_334463


namespace theta_range_l334_334690

open Real

theorem theta_range (θ : ℝ) (hx : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * x^2 * sin θ - 4 * x * (1 - x) * cos θ + 3 * (1 - x)^2 > 0) :
  θ ∈ (0, 2 * π) → θ ∈ (π / 6, π) :=
by
  sorry

end theta_range_l334_334690


namespace compound_interest_calculation_l334_334773

noncomputable def simple_interest (P R T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def compound_amount (P R : ℝ) : ℝ :=
  P * (1 + R / 100)

noncomputable def compound_interest_three_years (P R1 R2 R3 : ℝ) : ℝ :=
  let A1 := compound_amount P R1
  let A2 := compound_amount A1 R2
  let A3 := compound_amount A2 R3
  A3 - P

theorem compound_interest_calculation :
  ∃ P : ℝ, simple_interest P 5 2 = 58 →
  compound_interest_three_years P 4 6 5 = 91.2524 :=
by
  use 580  -- Given simple interest conditions, the calculated principal is 580.
  intro h
  rw [simple_interest] at h
  simp at h
  rw [compound_interest_three_years]
  have hA1 : compound_amount 580 4 = 602.72 := by simp [compound_amount]
  have hA2 : compound_amount 602.72 6 = 639.288 := by simp [compound_amount]
  have hA3 : compound_amount 639.288 5 = 671.2524 := by simp [compound_amount]
  simp [hA3]
  sorry  -- Skipping the proof step as requested.

end compound_interest_calculation_l334_334773


namespace balance_blue_balls_l334_334877

variable (G Y W B : ℝ)

-- Define the conditions
def condition1 : 4 * G = 8 * B := sorry
def condition2 : 3 * Y = 8 * B := sorry
def condition3 : 4 * B = 3 * W := sorry

-- Prove the required balance of 3G + 4Y + 3W
theorem balance_blue_balls (h1 : 4 * G = 8 * B) (h2 : 3 * Y = 8 * B) (h3 : 4 * B = 3 * W) :
  3 * (2 * B) + 4 * (8 / 3 * B) + 3 * (4 / 3 * B) = 62 / 3 * B := by
  sorry

end balance_blue_balls_l334_334877


namespace work_last_duration_l334_334557

theorem work_last_duration
  (work_rate_x : ℚ := 1 / 20)
  (work_rate_y : ℚ := 1 / 12)
  (days_x_worked_alone : ℚ := 4)
  (combined_work_rate : ℚ := work_rate_x + work_rate_y)
  (remaining_work : ℚ := 1 - days_x_worked_alone * work_rate_x) :
  (remaining_work / combined_work_rate + days_x_worked_alone = 10) :=
by
  sorry

end work_last_duration_l334_334557


namespace chess_tournament_points_distribution_l334_334180

noncomputable def points_distribution (Andrey Dima Vanya Sasha : ℝ) : Prop :=
  ∃ (p_a p_d p_v p_s : ℝ), 
    p_a ≠ p_d ∧ p_d ≠ p_v ∧ p_v ≠ p_s ∧ p_a ≠ p_v ∧ p_a ≠ p_s ∧ p_d ≠ p_s ∧
    p_a + p_d + p_v + p_s = 12 ∧ -- Total points sum
    p_a > p_d ∧ p_d > p_v ∧ p_v > p_s ∧ -- Order of points
    Andrey = p_a ∧ Dima = p_d ∧ Vanya = p_v ∧ Sasha = p_s ∧
    Andrey - (Sasha - 2) = 2 -- Andrey and Sasha won the same number of games

theorem chess_tournament_points_distribution :
  points_distribution 4 3.5 2.5 2 :=
sorry

end chess_tournament_points_distribution_l334_334180


namespace john_max_misses_l334_334814

-- Define the conditions given in the problem
def total_rounds : ℕ := 60
def required_percentage : ℝ := 0.90
def hits_after_40_rounds : ℕ := 30
def completed_rounds : ℕ := 40
def remaining_rounds : ℕ := total_rounds - completed_rounds

-- Calculate the total hits needed to reach the goal
def total_hits_needed (total_rounds : ℕ) (required_percentage : ℝ) : ℕ :=
  (required_percentage * total_rounds).ceil

-- Calculate the additional hits needed
def additional_hits_needed (total_hits_needed : ℕ) (hits_after_40_rounds : ℕ) : ℕ :=
  total_hits_needed - hits_after_40_rounds

-- The proof statement
theorem john_max_misses (total_rounds : ℕ) (required_percentage : ℝ) (hits_after_40_rounds : ℕ) 
  (completed_rounds : ℕ) (remaining_rounds : ℕ) :
  let total_hits := total_hits_needed total_rounds required_percentage in
  let additional_hits := additional_hits_needed total_hits hits_after_40_rounds in
  additional_hits > remaining_rounds → 0 = 0 := sorry

end john_max_misses_l334_334814
