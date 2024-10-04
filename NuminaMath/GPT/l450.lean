import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Divisibility
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics
import Mathlib.Combinatorics.Combinations
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.MassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Angle
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Logic.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NonlinearArith
import Mathlib.Topology.Algebra.Parabola
import Mathlib.Topology.Basic
import Mathlib.Trigonometry.Cosine
import Real

namespace spinner_even_product_probability_l450_450452

-- Define the outcomes for the two spinners
def spinner1 := {0, 2}
def spinner2 := {1, 3, 5}

-- Define the event that the product is even
def is_product_even (a b : ℕ) : Prop := (a * b) % 2 = 0

-- Theorem statement
theorem spinner_even_product_probability : 
  ∀ (a ∈ spinner1) (b ∈ spinner2), is_product_even a b :=
by
  sorry

end spinner_even_product_probability_l450_450452


namespace total_triangles_in_geometric_figure_l450_450084

noncomputable def numberOfTriangles : ℕ :=
  let smallest_triangles := 3 + 2 + 1
  let medium_triangles := 2
  let large_triangle := 1
  smallest_triangles + medium_triangles + large_triangle

theorem total_triangles_in_geometric_figure : numberOfTriangles = 9 := by
  unfold numberOfTriangles
  sorry

end total_triangles_in_geometric_figure_l450_450084


namespace max_sum_of_circle_eq_eight_l450_450870

noncomputable def max_sum_of_integer_solutions (r : ℕ) : ℕ :=
  if r = 6 then 8 else 0

theorem max_sum_of_circle_eq_eight 
  (h1 : ∃ (x y : ℤ), (x - 1)^2 + (y - 1)^2 = 36 ∧ (r : ℕ) = 6) :
  max_sum_of_integer_solutions r = 8 := 
by
  sorry

end max_sum_of_circle_eq_eight_l450_450870


namespace pyramid_intersection_area_l450_450553

-- Definitions for points A, B, C, D, and E
def A := (0 : ℝ, 0 : ℝ, 0 : ℝ)
def B := (6 : ℝ, 0 : ℝ, 0 : ℝ)
def C := (6 : ℝ, 6 : ℝ, 0 : ℝ)
def D := (0 : ℝ, 6 : ℝ, 0 : ℝ)
def E := (3 : ℝ, 3 : ℝ, 3 * Real.sqrt 2)

-- Definitions for midpoints M, N, and P
def M := ((1.5 : ℝ), (1.5 : ℝ), 1.5 * Real.sqrt 2)
def N := (3 : ℝ, 3 : ℝ, 0 : ℝ)
def P := (3 : ℝ, 6 : ℝ, 0 : ℝ)

-- The equation of the plane passing through M, N, P
def plane (x y z : ℝ) : Prop := 
  x + y + z * Real.sqrt 2 = 3 * Real.sqrt 2

-- Intersection points derived from solution (these will be points on AE, BD, CD)
def intersection1 := (3.5 : ℝ, 2.5 : ℝ, 2.5 * Real.sqrt 2)
def intersection2 := (3 : ℝ, 3 : ℝ, 0 : ℝ)
def intersection3 := (3 : ℝ, 6 : ℝ, 0 : ℝ)

-- Area of the triangle formed by these intersection points
noncomputable def triangle_area := 
  0.5 * (Real.sqrt (Real.pow (2.5 * Real.sqrt 2 - 0) 2 + Real.pow (2.5 - 3) 2 + Real.pow (3.5 - 3) 2)) * 
           (Real.sqrt (Real.pow (2.5 * Real.sqrt 2 - 0) 2 + Real.pow (6 - 3) 2 + Real.pow (3 - 3) 2))

theorem pyramid_intersection_area : triangle_area = 4.5 * Real.sqrt 2 :=
  sorry

end pyramid_intersection_area_l450_450553


namespace shorter_leg_length_l450_450798

theorem shorter_leg_length (m h x : ℝ) (H1 : m = 15) (H2 : h = 3 * x) (H3 : m = 0.5 * h) : x = 10 :=
by
  sorry

end shorter_leg_length_l450_450798


namespace burattino_suspects_cheating_after_seventh_draw_l450_450076

theorem burattino_suspects_cheating_after_seventh_draw 
  (total_balls : ℕ := 45) (drawn_balls : ℕ := 6) (a : ℝ := ((nat.choose 39 6 : ℕ) : ℝ) / ((nat.choose 45 6 : ℕ) : ℝ))
  (threshold : ℝ := 0.01) (probability : ℝ := 0.4) :
  (∃ n, a^n < threshold) → (∃ n > 5, a^n < threshold) :=
begin
  -- Definitions from conditions
  have fact_prob : a = ((nat.choose 39 6 : ℕ) : ℝ) / ((nat.choose 45 6 : ℕ) : ℝ), by refl,
  have fact_approx : a ≈ probability, by simp,

  -- Statement to prove
  intros h,
  use 6,
  split,
  { linarith, },
  { sorry }
end

end burattino_suspects_cheating_after_seventh_draw_l450_450076


namespace volume_inside_cube_outside_sphere_closest_to_A_l450_450040

noncomputable def volume_of_region_closest_to_vertex_A (a : ℝ) (pi : ℝ) : ℝ :=
  let V_cube := a^3
  let r := a / 2
  let V_sphere := (pi * a^3) / 6
  let V_outside_sphere := V_cube - V_sphere
  let V_closest_to_vertex_A := (a^3 / 8) * (1 - (pi / 6))
  V_closest_to_vertex_A

theorem volume_inside_cube_outside_sphere_closest_to_A (pi : ℝ) :
  volume_of_region_closest_to_vertex_A 4 pi ≈ 3.8112 := 
sorry

end volume_inside_cube_outside_sphere_closest_to_A_l450_450040


namespace original_savings_l450_450848

-- Define Linda's original savings
variable (S : ℕ)

-- Conditions
variable (furnitureSpent : 3/4 * S)
variable (tvSpent : 250)

theorem original_savings (h : 1/4 * S = 250) : S = 1000 :=
by
  sorry

end original_savings_l450_450848


namespace num_functions_with_period_pi_l450_450897

def f1 (x : ℝ) : ℝ := Real.sin (|x|)
def f2 (x : ℝ) : ℝ := |Real.sin x|
def f3 (x : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3)
def f4 (x : ℝ) : ℝ := Real.cos (x / 2 + 2 * Real.pi / 3)
def f5 (x : ℝ) : ℝ := Real.cos x + |Real.cos x|
def f6 (x : ℝ) : ℝ := Real.tan (x / 2) + 1

theorem num_functions_with_period_pi : 
  ([(f1, false), 
    (f2, true), 
    (f3, true), 
    (f4, false), 
    (f5, false), 
    (f6, false)].count (λ f, f.2)) = 2 := 
by sorry

end num_functions_with_period_pi_l450_450897


namespace tank_filling_time_l450_450391

noncomputable def netWaterPerCycle (rateA rateB rateC : ℕ) : ℕ := rateA + rateB - rateC

noncomputable def totalTimeToFill (tankCapacity rateA rateB rateC cycleDuration : ℕ) : ℕ :=
  let netWater := netWaterPerCycle rateA rateB rateC
  let cyclesNeeded := tankCapacity / netWater
  cyclesNeeded * cycleDuration

theorem tank_filling_time :
  totalTimeToFill 750 40 30 20 3 = 45 :=
by
  -- replace "sorry" with the actual proof if required
  sorry

end tank_filling_time_l450_450391


namespace find_n_value_l450_450591

theorem find_n_value (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 9) : n = 210 := sorry

end find_n_value_l450_450591


namespace math_problem_proof_l450_450149

-- Define the equations of the given circles
def circle1 (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define the locus L of the center of circle C
def locus_C (y : ℝ) : Prop := y = -1

-- Conditions and properties
def distances (M : ℝ × ℝ) (F : ℝ × ℝ) (L : ℝ → Prop) (m n : ℝ) : Prop :=
  let d1 := abs (M.snd + 1)
  let d2 := real.sqrt(M.fst^2 + (M.snd - 1)^2)
  L M.snd ∧ d1 = m ∧ d2 = n ∧ m = n

-- Define the locus Q of point M such that m = n
def locus_Q (x y : ℝ) := x^2 = 4 * y

-- Define the existence of point B (x1, y1) on the locus Q
def point_on_locus_Q (x1 y1 : ℝ) := locus_Q x1 y1

-- Verify the area of the triangle formed by the tangent line at point B and the two coordinate axes
def triangle_area (x1 y1 : ℝ) := abs (x1^2 * abs x1 / 16) = 1 / 2

theorem math_problem_proof :
  (∀ C, C circle1 ∧ C circle2 → locus_C C.snd) ∧
  (∀ M F m n, distances M F locus_C m n → locus_Q M.fst M.snd) ∧
  (∃ (B : ℝ × ℝ), point_on_locus_Q B.fst B.snd ∧ triangle_area B.fst B.snd ∧ (B = (2, 1) ∨ B = (-2, 1))) :=
by sorry

end math_problem_proof_l450_450149


namespace compute_expected_value_l450_450758

def expected_diff_one (n : ℕ) : ℚ := 
  let total_pairs := (n - 1)
  let prob_diff_one := 1 / (n * (n - 1) / 2)
  total_pairs * prob_diff_one

theorem compute_expected_value : 
  ∀ (n : ℕ), n = 50 →
  let expected_value := expected_diff_one n in
  let (m, d) := expected_value.num_denom in
  let (m, n) := (m, d.gcd) in  -- Ensure they are in simplest form
  100 * m + n = 4925 :=
by
  intro n h
  rw [expected_diff_one, h]
  dsimp [expected_diff_one]
  -- placeholder sorry
  sorry

end compute_expected_value_l450_450758


namespace true_propositions_count_l450_450209

-- Defining the conditions as per the given propositions
def proposition1 (a b c : ℝ) (h : a ≠ 0) : Prop :=
  (b^2 - 4 * a * c >= 0) -> ∃ x : ℝ, a * x^2 + b * x + c = 0

def proposition2 (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (h : ∀ x y z : A, dist x y = dist y z ∧ dist y z = dist z x ∧ dist z x = dist x y → equilateral_triangle A) :
  Prop :=
  ∀ (x y z : A), equilateral_triangle A → dist x y = dist y z ∧ dist y z = dist z x

def proposition3 (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∀ x y : ℝ, (∛x > ∛y → x > y)  -- Cube root properties

def proposition4 (m : ℝ) : Prop :=
  (∀ x : ℝ, m * x^2 - 2 * (m + 1) * x + (m + 3) > 0) → m ≥ 1

-- The theorem stating the number of true propositions
theorem true_propositions_count : (proposition1 a b c h₁)
  ∧ (proposition2 ℝ ℝ ℝ h₂)
  ∧ (proposition3 a b h₃)
  ∧ ¬ (proposition4 m)
    → true := sorry

end true_propositions_count_l450_450209


namespace ellipse_standard_form_l450_450989

theorem ellipse_standard_form (F : ℝ × ℝ) (e : ℝ) (h1 : F = (0, 1)) (h2 : e = 1 / 2) :
  (∃ a b : ℝ, (a = 2 ∧ b = sqrt 3) ∧ (∀ x y : ℝ, (x^2) / (4) + (y^2) / (3) = 1)) :=
by {
  sorry
}

end ellipse_standard_form_l450_450989


namespace max_x_for_management_fee_l450_450874

theorem max_x_for_management_fee (x : ℝ) (h₀ : 0 < x) (h₁ : x ≤ 10)
  (h₂ : x ≥ 2) :
  ∀ f : ℝ, f = (70 + 70 * x / (100 - x)) * (11.8 - x) * x / 100 →
  f ≥ 14 :=
begin
  intros, sorry
end

end max_x_for_management_fee_l450_450874


namespace solve_inequality_l450_450369

theorem solve_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_diff : ∀ x ∈ set.Ioi 0, has_deriv_at f (f' x) x)
  (h_cond : ∀ x ∈ set.Ioi 0, x * (f' x) + 2 * (f x) > 0) :
  set_of (λ x:ℝ, (x + 2016) / 5 * f (x + 2016) < 5 * f 5 / (x + 2016)) = set.Ioo (-2016) (-2011) :=
by
  sorry

end solve_inequality_l450_450369


namespace total_marbles_l450_450379

def mary_marbles := 9
def joan_marbles := 3
def john_marbles := 7

theorem total_marbles :
  mary_marbles + joan_marbles + john_marbles = 19 :=
by
  sorry

end total_marbles_l450_450379


namespace midpoint_AB_on_fixed_line_l450_450726

-- Definition of the geometric entities and the conditions
variables (a b : Line) (O M : Point)
variables (C : Circle) (A B : Point)

-- Conditions from the problem
axiom lines_intersect_at_O : a ∩ b = {O}
axiom M_not_on_a : M ∉ a
axiom M_not_on_b : M ∉ b
axiom circle_passing_through_O_M : O ∈ C ∧ M ∈ C 
axiom circle_intersects_a_b : A ∈ a ∧ B ∈ b ∧ A ≠ O ∧ B ≠ O

-- The goal is to prove that the midpoint of segment AB lies on a fixed line
theorem midpoint_AB_on_fixed_line :
  ∃ l : Line, midpoint A B ∈ l :=
sorry

end midpoint_AB_on_fixed_line_l450_450726


namespace count_integers_without_1248_l450_450668

noncomputable def usable_digits : Finset ℕ := {0, 3, 5, 6, 7, 9}

noncomputable def count_valid_numbers : ℕ := 
  let single_digits := (usable_digits.filter (λ x, x ≠ 0)).card
  let two_digit_numbers := (usable_digits.filter (λ x, x ≠ 0)).card * usable_digits.card
  let three_digit_numbers := (usable_digits.filter (λ x, x ≠ 0)).card * usable_digits.card ^ 2
  single_digits + two_digit_numbers + three_digit_numbers

theorem count_integers_without_1248 : count_valid_numbers = 215 :=
  by sorry

end count_integers_without_1248_l450_450668


namespace value_of_f_at_point_l450_450642

noncomputable def f : ℝ → ℝ 
| x => if x ≥ 4 then (1/2)^x else f (x + 1)

theorem value_of_f_at_point : f (2 + log 2 3) = 1 / 24 := 
by 
  sorry

end value_of_f_at_point_l450_450642


namespace tangency_proof_l450_450786

-- Definitions (Conditions)
variables (A B C : Point) -- vertices of triangle ABC
variables (I : Point) -- incenter of triangle ABC
variables (circumcircle : Circle) -- circumcircle of triangle ABC
variables (A₀ C₀ P : Point) -- Points A₀, C₀, and P as defined in conditions

-- Assume that A₀ and C₀ are where the angle bisectors of A and C intersect the circumcircle
def angle_bisector_A (A B C : Point) : Line := sorry
def angle_bisector_C (A B C : Point) : Line := sorry

-- A₀ and C₀ are the intersection points of these bisectors with the circumcircle
axiom angle_bisector_A_circumcircle_intersection : angle_bisector_A A B C ∩ circumcircle = {A₀}
axiom angle_bisector_C_circumcircle_intersection : angle_bisector_C A B C ∩ circumcircle = {C₀}

-- Define the line through I which is parallel to AC
def parallel_through_I (A C I : Point) : Line := sorry

-- Assume P is the intersection of this parallel line with A₀C₀
axiom parallel_intersection : parallel_through_I A C I ∩ (Line.mk A₀ C₀) = {P}

-- Statement to be proven
theorem tangency_proof : Tangent (circumcircle) (Line.mk P B) :=
sorry

end tangency_proof_l450_450786


namespace median_of_100_numbers_l450_450305

theorem median_of_100_numbers (x : Fin 100 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i = 78 → x j = 66 → i = 51 ∧ j = 50 ∨ i = 50 ∧ j = 51)
  (h2 : ∀ i, i ≠ 51 → x 51 = 78)
  (h3 : ∀ i, i ≠ 50 → x 50 = 66) :
  (x 50 + x 51) / 2 = 72 :=
by sorry

end median_of_100_numbers_l450_450305


namespace smallest_n_divisibility_l450_450955

theorem smallest_n_divisibility :
  ∃ (n : ℕ), n ≥ 4 ∧ (∀ (s : Finset ℤ), s.card = n → 
    ∃ (a b c d : ℤ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ d ∈ s ∧ 
    (a + b - c - d) % 20 = 0) := 
begin
  use 9,
  split,
  exact le_of_eq rfl,
  intros s hs,
  -- Proof is skipped using sorry
  sorry,
end

end smallest_n_divisibility_l450_450955


namespace triangle_third_side_length_l450_450195

theorem triangle_third_side_length (a b : ℝ) (x : ℝ) (h₁ : a = 5) (h₂ : b = 8) (hx : x ∈ {2, 3, 6, 13}) :
  3 < x ∧ x < 13 → x = 6 :=
by sorry

end triangle_third_side_length_l450_450195


namespace paul_homework_average_l450_450115

def hoursOnWeeknights : ℕ := 2 * 5
def hoursOnWeekend : ℕ := 5
def totalHomework : ℕ := hoursOnWeeknights + hoursOnWeekend
def practiceNights : ℕ := 2
def daysAvailable : ℕ := 7 - practiceNights
def averageHomeworkPerNight : ℕ := totalHomework / daysAvailable

theorem paul_homework_average :
  averageHomeworkPerNight = 3 := 
by
  -- sorry because we skip the proof
  sorry

end paul_homework_average_l450_450115


namespace possible_third_side_l450_450188

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end possible_third_side_l450_450188


namespace geometric_sequence_sum_l450_450205

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_sum1 : a 0 + a 1 + a 2 = 3)
  (h_sum2 : a 3 + a 4 + a 5 = 6) :
  (∑ i in finset.range 12, a i) = 45 :=
sorry

end geometric_sequence_sum_l450_450205


namespace rectangular_solid_surface_area_l450_450581

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (h_volume : a * b * c = 1001) :
  2 * (a * b + b * c + c * a) = 622 :=
by
  sorry

end rectangular_solid_surface_area_l450_450581


namespace no_real_solution_intersection_l450_450112

theorem no_real_solution_intersection :
  ¬ ∃ x y : ℝ, (y = 8 / (x^3 + 4 * x + 3)) ∧ (x + y = 5) :=
by
  sorry

end no_real_solution_intersection_l450_450112


namespace triangle_inscribed_in_semicircle_l450_450012

variables {R : ℝ} (P Q R' : ℝ) (PR QR : ℝ)
variables (hR : 0 < R) (h_pq_diameter: P = -R ∧ Q = R)
variables (h_pr_square_qr_square : PR^2 + QR^2 = 4 * R^2)
variables (t := PR + QR)

theorem triangle_inscribed_in_semicircle (h_pos_pr : 0 < PR) (h_pos_qr : 0 < QR) : 
  t^2 ≤ 8 * R^2 :=
sorry

end triangle_inscribed_in_semicircle_l450_450012


namespace burattino_suspects_cheating_after_seventh_draw_l450_450063

theorem burattino_suspects_cheating_after_seventh_draw
  (balls : ℕ)
  (draws : ℕ)
  (a : ℝ)
  (p_limit : ℝ)
  (h_balls : balls = 45)
  (h_draws : draws = 6)
  (h_a : a = (39.choose 6 : ℝ) / (45.choose 6 : ℝ))
  (h_p_limit : p_limit = 0.01) :
  ∃ (n : ℕ), n > 5 ∧ a^n < p_limit := by
  sorry

end burattino_suspects_cheating_after_seventh_draw_l450_450063


namespace sin_cos_product_positive_in_third_quadrant_l450_450927

theorem sin_cos_product_positive_in_third_quadrant (θ : ℝ) (h1 : π < θ) (h2 : θ < 3*π/2) : sin θ * cos θ > 0 :=
sorry

end sin_cos_product_positive_in_third_quadrant_l450_450927


namespace computation_l450_450912

theorem computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  have h₁ : 27 = 3^3 := by rfl
  have h₂ : (3 : ℕ) ^ 4 = 81 := by norm_num
  have h₃ : 27^63 / 27^61 = (3^3)^63 / (3^3)^61 := by rw [h₁]
  rwa [← pow_sub, nat.sub_eq_iff_eq_add] at h₃
  have h4: 3 * 3^4 = 3^5 := by norm_num
  have h5: -486 = 3^5 - 3^6 := by norm_num
  exact h5
  sorry

end computation_l450_450912


namespace triangle_inequality_third_side_l450_450170

theorem triangle_inequality_third_side (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) :
  3 < x ∧ x < 13 ↔ (5 + 8 > x) ∧ (5 + x > 8) ∧ (8 + x > 5) :=
by 
  -- Placeholder for proof
  sorry

end triangle_inequality_third_side_l450_450170


namespace fibonacci_product_l450_450367

-- Define the Fibonacci sequence
noncomputable def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Statement of the problem
theorem fibonacci_product :
  (∏ k in finset.range 99 \ finset.range 2, 
    ((fibonacci (k+2) / fibonacci k) - (fibonacci (k+2) / fibonacci (k+4)))) = 
  (fibonacci 100 / fibonacci 102) :=
sorry

end fibonacci_product_l450_450367


namespace repeating_decimal_as_fraction_l450_450585

theorem repeating_decimal_as_fraction :
  (3 + 45 / 99) = 38 / 11 :=
by
  -- Here you would perform the necessary steps and computations to show the equivalency.
  sorry

end repeating_decimal_as_fraction_l450_450585


namespace min_val_vector_sum_is_sqrt_3_over_2_l450_450664

noncomputable def min_value_of_vector_sum (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  if h1 : ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ real.angle a b = 2 * real.pi / 3 ∧ collinear ℝ ({a + b}) then
    ∥a + λ * (a + b)∥
  else
    0

theorem min_val_vector_sum_is_sqrt_3_over_2 
  (a b : EuclideanSpace ℝ (Fin 3))
  (h1 : ∥a∥ = 1)
  (h2 : ∥b∥ = 1)
  (h3 : real.angle a b = 2 * real.pi / 3)
  (h4 : collinear ℝ ({a + b})) :
  infimum (min_value_of_vector_sum a b) = √3 / 2 :=
sorry

end min_val_vector_sum_is_sqrt_3_over_2_l450_450664


namespace wall_height_l450_450864

theorem wall_height (brick_length brick_width brick_height : ℝ) (wall_length wall_width : ℝ) (num_bricks : ℕ) : 
  brick_length = 0.2 → 
  brick_width = 0.1 → 
  brick_height = 0.075 → 
  wall_length = 26 → 
  wall_width = 0.75 → 
  num_bricks = 26000 → 
  let brick_volume := brick_length * brick_width * brick_height in 
  let total_brick_volume := num_bricks * brick_volume in 
  let wall_volume := wall_length * wall_width in 
  let wall_height := total_brick_volume / wall_volume in
  wall_height = 2.67 := 
by
  -- The proof should go here
  sorry

end wall_height_l450_450864


namespace choco_candies_cost_l450_450524

-- The definitions based on the conditions specified in the problem
def box_candies := 20
def box_cost := 8
def discount_threshold := 400
def discount_rate := 0.10

-- Definition of the problem statement
theorem choco_candies_cost (total_candies : ℕ) (H : total_candies > 400) : 
  total_candies = 500 → 
  ((total_candies / box_candies * box_cost) - (discount_rate * (total_candies / box_candies * box_cost))) = 180 :=
by
  -- sorry is used to skip the proof part since only the statement is required
  sorry

end choco_candies_cost_l450_450524


namespace player1_wins_11th_round_l450_450485

noncomputable def egg_strength_probability (n : ℕ) : ℚ :=
  (n - 1) / n

theorem player1_wins_11th_round :
  let player1_wins_first_10_rounds := true,
      total_rounds := 11,
      new_egg := 12 in
  player1_wins_first_10_rounds → egg_strength_probability total_rounds = 11 / 12 :=
by
  intros
  exact sorry

end player1_wins_11th_round_l450_450485


namespace burattino_suspects_cheating_after_seventh_draw_l450_450070

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def binomial (n k : ℕ) : ℚ :=
(factorial n) / ((factorial k) * factorial (n - k))

noncomputable def probability_no_repeats : ℚ :=
(binomial 39 6) / (binomial 45 6)

noncomputable def estimate_draws_needed : ℕ :=
let a : ℚ := probability_no_repeats in
nat.find (λ n, a^n < 0.01)

theorem burattino_suspects_cheating_after_seventh_draw :
  estimate_draws_needed + 1 = 7 := sorry

end burattino_suspects_cheating_after_seventh_draw_l450_450070


namespace curve_cartesian_line_cartesian_min_distance_l450_450343

noncomputable def curve_parametric (theta : ℝ) : ℝ × ℝ :=
  (Real.cos theta, Real.sin theta)

noncomputable def curve_cartesian_eqn : Prop :=
  ∀ theta : ℝ, (Real.cos theta)^2 + (Real.sin theta)^2 = 1

noncomputable def line_polar_eqn (rho theta : ℝ) : ℝ :=
  rho * (2 * Real.cos theta - Real.sin theta) - 6

noncomputable def line_cartesian_eqn (x y : ℝ) : Prop :=
  2 * x - y = 6

noncomputable def min_distance_point : ℝ × ℝ :=
  (2 * Real.sqrt 5 / 5, -Real.sqrt 5 / 5)

noncomputable def min_distance_to_line : ℝ :=
  (6 * Real.sqrt 5 / 5) - 1

theorem curve_cartesian :
  curve_cartesian_eqn := sorry

theorem line_cartesian:
  ∀ rho theta: ℝ,  line_polar_eqn rho theta = 0 -> (line_cartesian_eqn (rho * Real.cos theta) (rho * Real.sin theta)) := sorry

theorem min_distance:
  ∀ θ: ℝ, (curve_parametric θ = min_distance_point) -> 
  let (x, y) := curve_parametric θ in 
  abs (2 * x - y - 6 ) / Real.sqrt (2^2 + 1^2) = min_distance_to_line := sorry

end curve_cartesian_line_cartesian_min_distance_l450_450343


namespace find_height_l450_450815

noncomputable def height_from_A (a b c : ℝ) (angle_B : ℝ) (r : ℝ) : ℝ :=
  (2 * c * (Real.sin (angle_B / 2))) / (sqrt(3) * a)

theorem find_height
  (a b c : ℝ)
  (h_sum : a + c = 11)
  (angle_B : ℝ)
  (h_angle_B : angle_B = Real.pi / 3)
  (r : ℝ)
  (h_r : r = 2 / sqrt 3)
  (h_ab_greater_bc : a > c)
  : height_from_A a b c angle_B r = 4 * sqrt 3 := by
  sorry

end find_height_l450_450815


namespace trajectory_of_M_lines_perpendicular_l450_450354

-- Define the given conditions
def parabola (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 = P.2

def midpoint_condition (P M : ℝ × ℝ) : Prop :=
  P.1 = 1/2 * M.1 ∧ P.2 = M.2

def trajectory_condition (M : ℝ × ℝ) : Prop :=
  M.1 ^ 2 = 4 * M.2

theorem trajectory_of_M (P M : ℝ × ℝ) (H1 : parabola P) (H2 : midpoint_condition P M) : 
  trajectory_condition M :=
sorry

-- Define the conditions for the second part
def line_through_F (A B : ℝ × ℝ) (F : ℝ × ℝ): Prop :=
  ∃ k : ℝ, A.2 = k * A.1 + F.2 ∧ B.2 = k * B.1 + F.2

def perpendicular_feet (A B A1 B1 : ℝ × ℝ) : Prop :=
  A1 = (A.1, -1) ∧ B1 = (B.1, -1)

def perpendicular_lines (A1 B1 F : ℝ × ℝ) : Prop :=
  let v1 := (-A1.1, F.2 - A1.2)
  let v2 := (-B1.1, F.2 - B1.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem lines_perpendicular (A B A1 B1 F : ℝ × ℝ) (H1 : trajectory_condition A) (H2 : trajectory_condition B) 
(H3 : line_through_F A B F) (H4 : perpendicular_feet A B A1 B1) :
  perpendicular_lines A1 B1 F :=
sorry

end trajectory_of_M_lines_perpendicular_l450_450354


namespace giants_win_probability_l450_450784

noncomputable def championship_probability : ℝ :=
  ∑ k in (finset.range 5), (nat.choose (4 + k) k) * ((4/7) ^ 5) * ((3/7) ^ k)

theorem giants_win_probability : championship_probability ≈ 0.63 :=
by
  have p : ℝ := championship_probability
  sorry  -- Proof omitted

end giants_win_probability_l450_450784


namespace quadratic_inequality_solution_l450_450610

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 4*x - 21 ≤ 0 ↔ -3 ≤ x ∧ x ≤ 7 :=
sorry

end quadratic_inequality_solution_l450_450610


namespace students_taking_neither_l450_450865

theorem students_taking_neither (total_students music art science music_and_art music_and_science art_and_science three_subjects : ℕ)
  (h1 : total_students = 800)
  (h2 : music = 80)
  (h3 : art = 60)
  (h4 : science = 50)
  (h5 : music_and_art = 30)
  (h6 : music_and_science = 25)
  (h7 : art_and_science = 20)
  (h8 : three_subjects = 15) :
  total_students - (music + art + science - music_and_art - music_and_science - art_and_science + three_subjects) = 670 :=
by sorry

end students_taking_neither_l450_450865


namespace fraction_product_eq_one_l450_450571

theorem fraction_product_eq_one :
  (7 / 4 : ℚ) * (8 / 14) * (21 / 12) * (16 / 28) * (49 / 28) * (24 / 42) * (63 / 36) * (32 / 56) = 1 := by
  sorry

end fraction_product_eq_one_l450_450571


namespace bedbug_infested_mattress_l450_450540

example (initial_bugs : ℕ) (days : ℕ) (tripling_rate : ℕ) : Prop :=
  initial_bugs = 30 ∧ days = 4 ∧ tripling_rate = 3 → 
  let final_bugs := initial_bugs * tripling_rate ^ days in
  final_bugs = 2430

-- Proof not required, thus using sorry.
theorem bedbug_infested_mattress : example 30 4 3 := by
  sorry

end bedbug_infested_mattress_l450_450540


namespace triangle_third_side_length_l450_450196

theorem triangle_third_side_length (a b : ℝ) (x : ℝ) (h₁ : a = 5) (h₂ : b = 8) (hx : x ∈ {2, 3, 6, 13}) :
  3 < x ∧ x < 13 → x = 6 :=
by sorry

end triangle_third_side_length_l450_450196


namespace chromatic_number_bound_l450_450609

theorem chromatic_number_bound (p : ℝ) (ε : ℝ) (G : Type) [graph G] [probability_space (graph G)] (n : ℕ) :
  p ∈ set.Ioo 0 1 →
  ε > 0 →
  (almost_every_graph G p n → 
    χ(G) > (log (1 / q) / (2 + ε)) * (n / log n)) :=
by sorry

end chromatic_number_bound_l450_450609


namespace hexagon_side_length_l450_450463

theorem hexagon_side_length (h : ℝ) (s : ℝ):
  h = 24 → h = (2 / real.sqrt 3) * s → s = 16 * real.sqrt 3 := by
  intro h_eq
  intro h_calc
  sorry

end hexagon_side_length_l450_450463


namespace smallest_N_l450_450272

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450272


namespace find_first_number_l450_450867

theorem find_first_number (ã : ℤ) (h : ã - ã + 93 * (ã - 93) = 19898) : ã = 307 := 
sorry

end find_first_number_l450_450867


namespace diagonal_length_of_convex_quadrilateral_l450_450742

theorem diagonal_length_of_convex_quadrilateral (AB BC AD CD BD : ℝ)
  (H1 : AB + BC = 2021)
  (H2 : AD = CD)
  (H3 : angle ABC = 90)
  (H4 : angle CDA = 90)
  : BD = 2021 / Real.sqrt 2 :=
  sorry

end diagonal_length_of_convex_quadrilateral_l450_450742


namespace angle_ABD_eq_CBD_l450_450716

variables {A B C D M B' : Type*} [trapezoid ABCD]
variables (AC AB CD AM B B' M : Point)
variables (ACeq: AC = AB + CD)
variables (MidM: midpoint M B C)
variables (SymB: sym_point B' B AM)

theorem angle_ABD_eq_CBD' :
  ∠ ABD = ∠ CB'D :=
by
  sorry

end angle_ABD_eq_CBD_l450_450716


namespace power_function_even_decreasing_l450_450203

theorem power_function_even_decreasing (m : ℝ) :
  (∀ x : ℝ, x^m = (-x)^m) ∧ (∀ x y : ℝ, 0 < x ∧ x < y → y^m < x^m)
  → m = -2 :=
by
  intros h,
  sorry

end power_function_even_decreasing_l450_450203


namespace Paul_average_homework_l450_450117

theorem Paul_average_homework :
  let weeknights := 5,
      weekend_homework := 5,
      night_homework := 2,
      nights_no_homework := 2,
      days_in_week := 7,
      total_homework := weekend_homework + night_homework * weeknights,
      available_nights := days_in_week - nights_no_homework,
      average_homework_per_night := total_homework / available_nights
  in average_homework_per_night = 3 := 
by
  sorry

end Paul_average_homework_l450_450117


namespace average_speed_of_entire_journey_l450_450777

-- Definitions of conditions
def distance_leg1 : ℕ := 150
def distance_leg2 : ℕ := 180
def time_leg1 : ℝ := 2 + 40/60
def time_leg2 : ℝ := 3 + 20/60

-- Definition of the total distance and total time
def total_distance : ℕ := distance_leg1 + distance_leg2
def total_time : ℝ := time_leg1 + time_leg2

-- Prove the average speed
theorem average_speed_of_entire_journey : total_distance / total_time = 55 := by
  sorry

end average_speed_of_entire_journey_l450_450777


namespace ordered_triples_count_l450_450886

theorem ordered_triples_count (d f: ℕ) (h: d ≤ 1980 ∧ 1980 ≤ f):
  (∃ u v w : ℕ, u ≤ v ∧ v ≤ w ∧ 
     (u * w = d * f) ∧ 
     v = 1980 ∧ 
     (d, 1980, f) ≠ (1980, 1980, 1980)) → 27 :=
by
  sorry

end ordered_triples_count_l450_450886


namespace centers_of_faces_form_regular_icosahedron_l450_450499

-- Define what it means to be a regular pentagon
def is_regular_pentagon (s : Type) [metric_space s] := sorry

-- Define what it means to be a regular dodecahedron
def is_regular_dodecahedron (s : Type) [metric_space s] := sorry

-- Define what it means to be a regular icosahedron
def is_regular_icosahedron (s : Type) [metric_space s] := sorry

theorem centers_of_faces_form_regular_icosahedron (s : Type) [metric_space s] 
  (h : is_regular_dodecahedron s) : 
  is_regular_icosahedron (centers_of_faces s) :=
sorry

end centers_of_faces_form_regular_icosahedron_l450_450499


namespace simplify_expression_l450_450400

-- Definition of the vectors
variables (a b : Vector)

-- The proof problem statement
theorem simplify_expression (a b : Vector) : 
  (1 / 3 : ℝ) • ((1 / 2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b)) = 2 • b - a :=
sorry

end simplify_expression_l450_450400


namespace exists_infinitely_many_repunits_div_by_m_l450_450395

open Nat

theorem exists_infinitely_many_repunits_div_by_m (m : ℕ) (h_coprime : Nat.coprime m 10) :
  ∃ E_n : ℕ → ℕ, (∀ k : ℕ, m ∣ E_n (k * Nat.totient m)) ∧ (∀ k : ℕ, m ∣ E_n k) :=
sorry

end exists_infinitely_many_repunits_div_by_m_l450_450395


namespace find_a_l450_450285

theorem find_a (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h1 : a^b = b^a) (h2 : b = 27 * a) : a = Real.root 26 27 :=
by
  sorry

end find_a_l450_450285


namespace count_numbers_in_set_S_l450_450672
open Nat

-- Define the sequence set
def S : Set ℕ := {n | ∃ k : ℕ, n = 7 + 10 * k}

-- Define a function to check if n can be expressed as the difference of two primes
def can_be_written_as_difference_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p > q ∧ n = p - q

-- Define a list of the numbers in the set S that can be written as such
def number_of_elements_in_S_that_can_be_written_as_difference_of_two_primes : ℕ :=
  (Finset.filter can_be_written_as_difference_of_two_primes (Finset.range 1000)).card

-- The main theorem
theorem count_numbers_in_set_S : number_of_elements_in_S_that_can_be_written_as_difference_of_two_primes = 2 := 
sorry

end count_numbers_in_set_S_l450_450672


namespace KE_parallel_AB_l450_450697

noncomputable def semicircle {ι : Type*} (O : ι) (A B : ι) : Set ι := sorry
noncomputable def on_diameter {ι : Type*} (A B C : ι) : Prop := sorry
noncomputable def equal_angles {ι : Type*} (A B C D E : ι) : Prop := sorry
noncomputable def perpendicular {ι : Type*} (D C : ι) : ι := sorry
noncomputable def intersects {ι : Type*} (K : ι) (circle : Set ι) : Prop := sorry
noncomputable def is_parallel {ι : Type*} (KE AB : ι) : Prop := sorry

theorem KE_parallel_AB {ι : Type*} 
  (O A B C D E K : ι) 
  (h1 : semicircle O A B) 
  (h2 : on_diameter A B C) 
  (h3 : equal_angles A B C D E) 
  (h4 : intersects K (semicircle O A B))
  (h5 : perpendicular D C = K)
  (h6 : D ≠ E) :
  is_parallel (perpendicular K E) (on_diameter A B) :=
sorry

end KE_parallel_AB_l450_450697


namespace cj_more_stamps_than_twice_kj_l450_450081

variable (C K A : ℕ) (x : ℕ)

theorem cj_more_stamps_than_twice_kj :
  (C = 2 * K + x) →
  (K = A / 2) →
  (C + K + A = 930) →
  (A = 370) →
  (x = 25) →
  (C - 2 * K = 5) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cj_more_stamps_than_twice_kj_l450_450081


namespace triangle_inequality_third_side_l450_450169

theorem triangle_inequality_third_side (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) :
  3 < x ∧ x < 13 ↔ (5 + 8 > x) ∧ (5 + x > 8) ∧ (8 + x > 5) :=
by 
  -- Placeholder for proof
  sorry

end triangle_inequality_third_side_l450_450169


namespace fibonacci_identity_l450_450086

theorem fibonacci_identity : 
  let F : ℕ → ℕ := λ n, (matrix.of (λ (i j : fin 2), i + j) ^ n) 0 1 in
  F 1094 * F 1096 - (F 1095 * F 1095) = -1 := sorry

end fibonacci_identity_l450_450086


namespace artist_painting_l450_450898

variable (total_pictures june_pictures august_pictures july_pictures : ℕ)

theorem artist_painting :
  (june_pictures = 2) →
  (august_pictures = 9) →
  (total_pictures = 13) →
  (total_pictures = june_pictures + july_pictures + august_pictures) →
  (july_pictures = june_pictures) :=
begin
  intros h1 h2 h3 h4,
  sorry, -- Proof is omitted
end

end artist_painting_l450_450898


namespace proposition_p_is_necessary_and_sufficient_condition_of_proposition_q_l450_450392

variable (a : ℝ)

def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x + y + 1 = 0
def line2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, x + a * y + 2 * a - 1 = 0

theorem proposition_p_is_necessary_and_sufficient_condition_of_proposition_q :
  (a = -1) ↔ (∀ x y, line1 a x y → ∃ x' y', line2 a x' y') :=
sorry

end proposition_p_is_necessary_and_sufficient_condition_of_proposition_q_l450_450392


namespace weight_of_new_person_l450_450787

-- Define the conditions
def avg_weight_increase : ℝ := 6.2
def replaced_person_weight : ℝ := 76
def num_persons : ℕ := 7

-- Define total weight increase
def total_weight_increase : ℝ := num_persons * avg_weight_increase

-- Define the weight of the new person
def new_person_weight : ℝ := replaced_person_weight + total_weight_increase

-- The proof statement
theorem weight_of_new_person : new_person_weight = 119.4 :=
by
  unfold new_person_weight total_weight_increase avg_weight_increase replaced_person_weight num_persons
  have : total_weight_increase = 43.4 := rfl
  show 76 + 43.4 = 119.4
  sorry

end weight_of_new_person_l450_450787


namespace son_time_to_complete_work_l450_450881

noncomputable def man_work_rate : ℚ := 1 / 6
noncomputable def combined_work_rate : ℚ := 1 / 3

theorem son_time_to_complete_work :
  (1 / (combined_work_rate - man_work_rate)) = 6 := by
  sorry

end son_time_to_complete_work_l450_450881


namespace probability_decreasing_on_neg_infinity_to_neg_one_l450_450646

noncomputable def f (a b x : ℝ) : ℝ := (1/2) * a * x^2 + b * x + 1

theorem probability_decreasing_on_neg_infinity_to_neg_one :
  let a_values := {2, 4}
      b_values := {1, 3}
      pairs := [(2, 1), (2, 3), (4, 1), (4, 3)]
      event_A := [(2, 1), (4, 1), (4, 3)] 
  in (event_A.length / pairs.length : ℝ) = 3 / 4 :=
by sorry

end probability_decreasing_on_neg_infinity_to_neg_one_l450_450646


namespace problem1_problem2_l450_450981

variable {α : ℝ}

-- Given condition
def tan_alpha (α : ℝ) : Prop := Real.tan α = 3

-- Proof statements to be shown
theorem problem1 (h : tan_alpha α) : (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 :=
by sorry

theorem problem2 (h : tan_alpha α) : Real.sin α ^ 2 + Real.sin α * Real.cos α + 3 * Real.cos α ^ 2 = 6 :=
by sorry

end problem1_problem2_l450_450981


namespace difference_of_squares_l450_450097

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_l450_450097


namespace difference_of_squares_l450_450096

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_l450_450096


namespace evaluate_fraction_l450_450122

theorem evaluate_fraction :
  (20: ℝ)⁻¹ * (3: ℝ)^0 / (5: ℝ)⁻² = (5: ℝ) / 4 :=
by 
  sorry

end evaluate_fraction_l450_450122


namespace no_brave_children_l450_450512

/-- A definition of a boy and a girl sitting on a bench initially -/
def initial_children_on_bench : list bool := [true, false]  -- Let's assume true is boy, false is girl

/-- A function to insert a child between every pair, resulting in alternating sequence -/
def add_children (lst : list bool) (n : ℕ) : list bool :=
list.bind lst (λ x, [x, !x])

/-- Initial setup with initial children and new children inserted -/
def final_arrangement : list bool :=
add_children initial_children_on_bench 20

/-- Check if a child is brave (which in this case, should always be false) -/
def is_brave (lst : list bool) (i : ℕ) : bool :=
(i > 0) && (i + 1 < lst.length) && ((lst[i - 1] = lst[i + 1]) && (lst[i - 1] != lst[i]))

/-- Count of brave children which should be zero -/
def count_brave_children (lst : list bool) : ℕ :=
lst.enum.filter (λ ⟨i, _⟩, is_brave lst i).length

theorem no_brave_children :
  final_arrangement.length = 22 ∧ count_brave_children final_arrangement = 0 :=
begin
  sorry
end

end no_brave_children_l450_450512


namespace cube_surface_area_l450_450094

theorem cube_surface_area (pi_pos : 0 < π) (sphere_surface_area : 4 * π * (1 / 2)^2 = π) :
  let r := 1 / 2,
      d := 2 * r,
      s := d / sqrt 3
  in 6 * s^2 = 2 :=
by
  -- This is the statement, the proof is omitted.
  sorry

end cube_surface_area_l450_450094


namespace triangle_BC_length_l450_450690

theorem triangle_BC_length (A B C X : Type) 
  (AB AC : ℕ) (BX CX BC : ℕ)
  (h1 : AB = 100)
  (h2 : AC = 121)
  (h3 : ∃ x y : ℕ, x = BX ∧ y = CX ∧ AB = 100 ∧ x + y = BC)
  (h4 : x * y = 31 * 149 ∧ x + y = 149) :
  BC = 149 := 
by
  sorry

end triangle_BC_length_l450_450690


namespace ratio_problem_l450_450655

/-
  Given the ratio A : B : C = 3 : 2 : 5, we need to prove that 
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19.
-/

theorem ratio_problem
  (A B C : ℚ)
  (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (2 * A + 3 * B) / (5 * C - 2 * A) = 12 / 19 :=
by sorry

end ratio_problem_l450_450655


namespace proof_problem_l450_450280

noncomputable def log2 := Real.log 3 / Real.log 2
noncomputable def log5 := Real.log 3 / Real.log 5

theorem proof_problem (x y : ℝ) 
  (h : log2 ^ x - log5 ^ x ≥ log2 ^ -y - log5 ^ -y) : 
  x + y ≥ 0 := 
sorry

end proof_problem_l450_450280


namespace longest_segment_l450_450101

theorem longest_segment (α β γ δ : ℝ) 
  (hα : α = 30) (hβ : β = 45) (hγ : γ = 65) (hδ : δ = 50) :
  let BAD := 180 - α - β,
      BCD := 180 - γ - δ in
  BAD > β ∧ β > α ∧ BCD > δ ∧ δ > γ ∧ 
  (segments := [AB, AD, BD, BC, CD]) ∧
  (order := List.sort (segments)) = [AD, BD, AB, CD, BC] :=
sorry

end longest_segment_l450_450101


namespace smallest_N_l450_450268

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450268


namespace median_of_100_numbers_l450_450324

theorem median_of_100_numbers (numbers : List ℝ) (h_length : numbers.length = 100)
  (h_sorted : numbers.sorted (≤))
  (h_51 : numbers.nth_le 50 h_51_nat = 78) -- Note: nth_le is zero-indexed, so 51st element is 50th index.
  (h_50 : numbers.nth_le 49 h_50_nat = 66) : 
  (numbers.nth_le 49 h_50_nat + numbers.nth_le 50 h_51_nat) / 2 = 72 :=
sorry

end median_of_100_numbers_l450_450324


namespace quadratic_equation_count_l450_450140

theorem quadratic_equation_count :
  let coeffs : Finset ℕ := {0, 1, 3, 5, 7}
  let a_set : Finset ℕ := {1, 3, 5, 7}
  let num_quadratic := ∏ _ in a_set, (coeffs.erase _).choose 2 * 2
  let real_root_count := 
    let c_zero_count := a_set.choose 2
    let non_c_zero_count := ∑ b in {5, 7}, if b = 5 
                                               then a_set.erase b .choose 2
                                               else 2 * a_set.erase b .choose 2
    c_zero_count + non_c_zero_count
  num_quadratic = 48 ∧ real_root_count = 18 := 
  ∏ sorry ∧ sorry -- expression verifying the combined conditions

end quadratic_equation_count_l450_450140


namespace problem1_solution_l450_450856

def f (x : ℝ) : ℝ := abs (x + 1) + abs (2 * x - 4)

theorem problem1_solution (x : ℝ) : f(x) ≥ 6 → (x ≤ -1 ∨ x ≥ 3) :=
by
  sorry

end problem1_solution_l450_450856


namespace tan_double_angle_l450_450635

-- Define the conditions and the angles in Lean
variables {θ : ℝ} (h_acute : 0 < θ ∧ θ < π / 2)
variables (h_sine : sin (θ - π / 4) = √2 / 10)

-- The statement to prove
theorem tan_double_angle :
  tan 2θ = -24 / 7 :=
sorry

end tan_double_angle_l450_450635


namespace latest_time_for_temperature_at_60_l450_450295

theorem latest_time_for_temperature_at_60
  (t : ℝ) (h : -t^2 + 10 * t + 40 = 60) : t = 12 :=
sorry

end latest_time_for_temperature_at_60_l450_450295


namespace intersection_distance_l450_450799

theorem intersection_distance : 
  let intersect_points := (x : ℝ) → x^2 + 2 * x - 4 = 0 in
  let C := -1 + Real.sqrt 5 in
  let D := -1 - Real.sqrt 5 in
  let distance := Real.abs ((-1 + Real.sqrt 5) - (-1 - Real.sqrt 5)) in
  let p := 20 in
  let q := 1 in
  distance = 2 * Real.sqrt 5 → p - q = 19 :=
by
  let intersect_points := (x : ℝ) → x^2 + 2 * x - 4 = 0
  let C := -1 + Real.sqrt 5
  let D := -1 - Real.sqrt 5
  let distance := Real.abs ((-1 + Real.sqrt 5) - (-1 - Real.sqrt 5))
  let p := 20
  let q := 1
  have sqrt5 : Real.sqrt 20 = Real.sqrt 4 * Real.sqrt 5 := sorry -- Skipping this proof
  have dist_correct : distance = 2 * Real.sqrt 5 := sorry -- Skipping this proof
  have p_minus_q_correct : p - q = 19 := sorry -- Skipping this proof
  sorry

end intersection_distance_l450_450799


namespace average_leaves_correct_l450_450769

def leaves_fell_first_tree := [7, 12, 9]
def leaves_fell_second_tree := [4, 4, 6]
def leaves_fell_third_tree := [10, 20, 15]

-- Calculate the total leaves for each tree
def total_leaves (leaves_list : List Int) : Int :=
  leaves_list.sum

def total_leaves_first_tree := total_leaves leaves_fell_first_tree
def total_leaves_second_tree := total_leaves leaves_fell_second_tree
def total_leaves_third_tree := total_leaves leaves_fell_third_tree

-- All leaves from all trees
def total_leaves_all_trees := 
  total_leaves_first_tree + total_leaves_second_tree + total_leaves_third_tree

-- Total number of hours for counting by Rylee
def total_hours := 3 * 3 -- 3 hours per tree, 3 trees

-- Calculate the average number of leaves per hour
def average_leaves_per_hour :=
  total_leaves_all_trees.toFloat / total_hours.toFloat

-- Prove the average number of leaves per hour is 9.67
theorem average_leaves_correct :
  average_leaves_per_hour = 9.67 :=
by
  unfold average_leaves_per_hour
  unfold total_leaves_all_trees
  unfold total_leaves_first_tree
  unfold total_leaves_second_tree
  unfold total_leaves_third_tree
  unfold total_leaves
  -- Calculation follows:
  have h1 : total_leaves (leaves_fell_first_tree) = 28 :=
    by simp [leaves_fell_first_tree, total_leaves]; rfl
  have h2 : total_leaves (leaves_fell_second_tree) = 14 :=
    by simp [leaves_fell_second_tree, total_leaves]; rfl
  have h3 : total_leaves (leaves_fell_third_tree) = 45 :=
    by simp [leaves_fell_third_tree, total_leaves]; rfl
  have : total_leaves_all_trees = 87 :=
    by simp [total_leaves_all_trees, h1, h2, h3]
  have : total_hours = 9 := by simp [total_hours]
  show 87.toFloat / 9.toFloat = 9.67
  norm_num


end average_leaves_correct_l450_450769


namespace triangle_BC_length_l450_450293

theorem triangle_BC_length (A B C X : Type) (AB AC BC BX CX : ℕ)
  (h1 : AB = 75)
  (h2 : AC = 85)
  (h3 : BC = BX + CX)
  (h4 : BX * (BX + CX) = 1600)
  (h5 : BX + CX = 80) :
  BC = 80 :=
by
  sorry

end triangle_BC_length_l450_450293


namespace buoy_radius_l450_450525

-- Define the conditions based on the given problem
def is_buoy_hole (width : ℝ) (depth : ℝ) : Prop :=
  width = 30 ∧ depth = 10

-- Define the statement to prove the radius of the buoy
theorem buoy_radius : ∀ r x : ℝ, is_buoy_hole 30 10 → (x^2 + 225 = (x + 10)^2) → r = x + 10 → r = 16.25 := by
  intros r x h_cond h_eq h_add
  sorry

end buoy_radius_l450_450525


namespace burattino_suspects_cheating_after_seventh_draw_l450_450078

theorem burattino_suspects_cheating_after_seventh_draw 
  (total_balls : ℕ := 45) (drawn_balls : ℕ := 6) (a : ℝ := ((nat.choose 39 6 : ℕ) : ℝ) / ((nat.choose 45 6 : ℕ) : ℝ))
  (threshold : ℝ := 0.01) (probability : ℝ := 0.4) :
  (∃ n, a^n < threshold) → (∃ n > 5, a^n < threshold) :=
begin
  -- Definitions from conditions
  have fact_prob : a = ((nat.choose 39 6 : ℕ) : ℝ) / ((nat.choose 45 6 : ℕ) : ℝ), by refl,
  have fact_approx : a ≈ probability, by simp,

  -- Statement to prove
  intros h,
  use 6,
  split,
  { linarith, },
  { sorry }
end

end burattino_suspects_cheating_after_seventh_draw_l450_450078


namespace Kit_time_to_reach_ice_cream_stand_l450_450724

theorem Kit_time_to_reach_ice_cream_stand :
  ∀ (rate distance remaining_distance_in_yards conversion_factor : ℕ),
    (rate = 2) →
    (remaining_distance_in_yards = 100) →
    (conversion_factor = 3) →
    (distance = remaining_distance_in_yards * conversion_factor) →
    (distance / rate = 150) :=
by
  intros rate distance remaining_distance_in_yards conversion_factor
  assume h_rate h_remaining_distance h_conversion_factor h_distance
  sorry

end Kit_time_to_reach_ice_cream_stand_l450_450724


namespace compute_expression_l450_450915

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end compute_expression_l450_450915


namespace equal_diagonals_convex_polygon_l450_450277

theorem equal_diagonals_convex_polygon (n : ℕ) (hconvex : convex_polygon n) (hdiagonals : ∀ (i j : ℕ), i ≠ j → diagonal_length n i j = diagonal_length n (i+1) (j+1)) :
  n = 4 ∨ n = 5 :=
sorry

end equal_diagonals_convex_polygon_l450_450277


namespace arithmetic_sequence_general_formula_sum_of_new_sequence_l450_450344

noncomputable def arithmeticSequence : ℕ → ℕ :=
  sorry  -- The proof is omitted.

theorem arithmetic_sequence_general_formula :
  ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
    (∀ n, S n = n * (a n + a 1) / 2) →  -- Sum of the first n terms of the arithmetic sequence
    (a 5 = 5) →  -- Given condition
    (S 8 = 36) →  -- Given condition
    (∀ n, a n = n) :=  -- Statement we want to prove
  sorry  -- The proof is omitted

theorem sum_of_new_sequence :
  ∀ (a : ℕ → ℕ) (b T : ℕ → ℕ),
    (∀ n, S n = n * (a n + a 1) / 2) →  -- Sum of the first n terms of the arithmetic sequence
    (a 5 = 5) →  -- Given condition
    (S 8 = 36) →  -- Given condition
    (∀ n, a n = n) →  -- General formula for the sequence
    (∀ n, b n = 2^n) →  -- New sequence
    (∀ n, T n = (n-1) * 2^(n+1) + 2) :=  -- Statement we want to prove
  sorry  -- The proof is omitted

end arithmetic_sequence_general_formula_sum_of_new_sequence_l450_450344


namespace monotonically_increasing_interval_l450_450144

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x) / Real.log (1/2)

theorem monotonically_increasing_interval :
  ∀ x, x ∈ Set.Iio (0 : ℝ) → StrictMono f :=
by
  sorry

end monotonically_increasing_interval_l450_450144


namespace AI_squared_eq_AD_mul_AE_l450_450544

-- Let us declare the necessary objects first
variables {A B C D E I : Point}
variables {l : Line}

-- Given conditions
axiom parallel_l_BC : parallel l (line_through B C)
axiom incircle_touches_l : l.touches (incircle ABC)
axiom circumcircle_intersection_points : l.meets_at_circumcircle_points D E (circumcircle ABC)
axiom I_is_incenter : incenter I ABC

-- To prove
theorem AI_squared_eq_AD_mul_AE : (distance A I) ^ 2 = (distance A D) * (distance A E) :=
sorry

end AI_squared_eq_AD_mul_AE_l450_450544


namespace total_canoes_built_by_End_of_May_l450_450055

noncomputable def total_canoes_built (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem total_canoes_built_by_End_of_May :
  total_canoes_built 7 2 5 = 217 :=
by
  -- The proof would go here.
  sorry

end total_canoes_built_by_End_of_May_l450_450055


namespace burattino_suspects_cheating_after_seventh_draw_l450_450067

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def binomial (n k : ℕ) : ℚ :=
(factorial n) / ((factorial k) * factorial (n - k))

noncomputable def probability_no_repeats : ℚ :=
(binomial 39 6) / (binomial 45 6)

noncomputable def estimate_draws_needed : ℕ :=
let a : ℚ := probability_no_repeats in
nat.find (λ n, a^n < 0.01)

theorem burattino_suspects_cheating_after_seventh_draw :
  estimate_draws_needed + 1 = 7 := sorry

end burattino_suspects_cheating_after_seventh_draw_l450_450067


namespace sum_inequality_l450_450986

noncomputable def a_seq : ℕ → ℕ 
| 1 => 1
| (n+1) => a_seq n + 2 * n

noncomputable def b_seq : ℕ → ℚ
| 1 => 1
| (n+1) => b_seq n + (b_seq n) ^ 2 / n

noncomputable def I (n : ℕ) : ℚ :=
  ∑ k in finset.range n, 1 / (real.sqrt (a_seq (k + 1) * b_seq k + k * a_seq (k + 1) - b_seq k - k))

theorem sum_inequality (n : ℕ) : 
  1 / 2 ≤ I n ∧ I n < 1 := 
  sorry

end sum_inequality_l450_450986


namespace function_even_l450_450650

variable {ℝ : Type} [NontriviallyNormedField ℝ]

/-- Given a function f : ℝ → ℝ where ℝ is the set of non-zero real numbers,
    and f satisfies f(x₁ * x₂) = f(x₁) + f(x₂) for all non-zero x₁ and x₂ in ℝ,
    prove that f is an even function, i.e., f(x) = f(-x). -/
theorem function_even (f : ℝ → ℝ) (h : ∀ x₁ x₂ : ℝ, x₁ ≠ 0 ∧ x₂ ≠ 0 → f (x₁ * x₂) = f x₁ + f x₂) :
  ∀ x : ℝ, x ≠ 0 → f x = f (-x) :=
by
  sorry

end function_even_l450_450650


namespace binom_ns_eq_binom_nrk_eq_binom_2n_eq_binom_ns_s_eq_binom_sum_even_odd_eq_binom_sum_ni_eq_l450_450766

-- Statement for (1) 
theorem binom_ns_eq (n s : ℕ) : 
  binom n s = (n / s) * binom (n - 1) (s - 1) := 
sorry

-- Statement for (2)
theorem binom_nrk_eq (n r k : ℕ) :
  binom n r * binom r k = binom n k * binom (n - k) (r - k) :=
sorry

-- Statement for (3)
theorem binom_2n_eq (n : ℕ) :
  ∑ i in finset.range (n + 1), binom n i * binom n (n - i) = binom (2 * n) n :=
sorry

-- Statement for (4)
theorem binom_ns_s_eq (n s : ℕ) : 
  binom n s = (n / (n - s)) * binom (n - 1) s := 
sorry

-- Statement for (5)
theorem binom_sum_even_odd_eq (n : ℕ) :
  ∑ i in finset.range (n / 2 + 1), binom n (2 * i) = ∑ i in finset.range (n / 2 + 1), binom n (2 * i + 1) :=
sorry

-- Statement for (6)
theorem binom_sum_ni_eq (n r : ℕ) :
  ∑ i in finset.range (r + 1), binom (n + i) i = binom (n + r + 1) r :=
sorry

end binom_ns_eq_binom_nrk_eq_binom_2n_eq_binom_ns_s_eq_binom_sum_even_odd_eq_binom_sum_ni_eq_l450_450766


namespace ratio_is_one_l450_450341

noncomputable def ratio_of_areas (A B C : Point) (PQ QR: ℝ) (H : right_triangle A B C) (PQ_10 : PQ = 10) (QR_15 : QR = 15) (S T : Point) (HS : midpoint S A B) (HT : midpoint T A C) (RS QT : LineSegment) (HRS : connects RS C S) (HQT : connects QT B T) (Y : Point) (HY : intersection Y RS QT) : ℝ :=
  let P := A
  let Q := B
  let R := C
  let PTYQ := quadrilateral P T Y Q
  let QYR := triangle Q Y R
  ratio PTYQ QYR

theorem ratio_is_one (A B C : Point) (PQ QR: ℝ) (H : right_triangle A B C) (PQ_10 : PQ = 10) (QR_15 : QR = 15) (S T : Point) (HS : midpoint S A B) (HT : midpoint T A C) (RS QT : LineSegment) (HRS : connects RS C S) (HQT : connects QT B T) (Y : Point) (HY : intersection Y RS QT) :
  ratio_of_areas A B C PQ QR H PQ_10 QR_15 S T HS HT RS QT HRS HQT Y HY = 1 := sorry

end ratio_is_one_l450_450341


namespace ratio_of_guests_l450_450909

theorem ratio_of_guests (x y : ℕ) (h1 : x + y = 30) (h2 : 2 * x + y = 45) : (x : ℚ) / (x + y : ℚ) = 1 / 2 :=
by
  have hx : x = 15, from sorry,
  have hy : y = 15, from sorry,
  rw [hx, hy],
  norm_num

end ratio_of_guests_l450_450909


namespace subset_M_N_l450_450925

theorem subset_M_N (k : ℤ) : 
  let M := {x | ∃ k: ℤ, x = (k * real.pi) / 2 + real.pi / 4 } in
  let N := {x | ∃ k: ℤ, x = (k * real.pi) / 4 + real.pi / 2 } in
  M ⊆ N :=
sorry

end subset_M_N_l450_450925


namespace solve_inequality_part1_solve_inequality_part1_neg_find_a_range_l450_450644

noncomputable def f (a x : ℝ) : ℝ := -1/a + 2/x

theorem solve_inequality_part1 (a x : ℝ) (h_pos : a > 0) : f a x > 0 ↔ 0 < x ∧ x < 2 * a := sorry

theorem solve_inequality_part1_neg (a x : ℝ) (h_neg : a < 0) : f a x > 0 ↔ x > 0 := sorry

theorem find_a_range (a : ℝ) : (∀ x > 0, f a x + 2 * x ≥ 0) ↔ a ∈ Iic 0 ∪ Ici (1/4) := sorry

end solve_inequality_part1_solve_inequality_part1_neg_find_a_range_l450_450644


namespace problem_solution_l450_450645

def f (x : ℝ) : ℝ :=
  if x > 0 then - (1 / x)
  else x^2

theorem problem_solution : f 2 + f (-2) = 7 / 2 := by 
  sorry

end problem_solution_l450_450645


namespace smallest_N_value_proof_l450_450250

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l450_450250


namespace consecutive_draws_probability_l450_450521

theorem consecutive_draws_probability :
  let total_chips := 14
      tan_chips := 5
      pink_chips := 3
      violet_chips := 6
      factorial := Nat.factorial in
  (factorial tan_chips * factorial pink_chips * factorial violet_chips * factorial 3)
  / (factorial total_chips) = 1 / 28080 :=
by
  let total_chips := 14
  let tan_chips := 5
  let pink_chips := 3
  let violet_chips := 6
  let factorial := Nat.factorial
  
  have h : (factorial tan_chips * factorial pink_chips * factorial violet_chips * factorial 3) / (factorial total_chips) = (120 * 6 * 720 * 6) / 87178291200 := by sorry
  
  exact h ▸ (3110400 / 87178291200 = 1 / 28080) sorry

end consecutive_draws_probability_l450_450521


namespace smallest_N_value_proof_l450_450248

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l450_450248


namespace complex_modulus_l450_450966

variable {x y : ℝ}
def z : ℂ := x + y * complex.I

theorem complex_modulus :
  (1 / 2 * x - y = 0) →
  (x + y = 3) →
  complex.abs z = real.sqrt 5 :=
by
  intros h1 h2
  sorry

end complex_modulus_l450_450966


namespace classify_c_and_d_l450_450822

theorem classify_c_and_d (c d : ℂ) (h1 : c ≠ 0) (h2 : c + d^2 ≠ 0) : 
  ∃ x y : ℂ, (x ≠ y ∨ x.im ≠ 0 ∨ y.im ≠ 0) :=
by 
  have eq1 : (c + d^2) / c^2 = 2 * d / (c + d^2), from sorry,
  sorry

end classify_c_and_d_l450_450822


namespace plane_through_point_perpendicular_to_vector_l450_450839

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

noncomputable def vector_from_points (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

noncomputable def plane_equation (point : Point3D) (normal : Point3D) (p : Point3D) : ℝ :=
  normal.x * (p.x - point.x) + normal.y * (p.y - point.y) + normal.z * (p.z - point.z)

theorem plane_through_point_perpendicular_to_vector (A B C : Point3D) :
  A = ⟨-10, 0, 9⟩ →
  B = ⟨12, 4, 11⟩ →
  C = ⟨8, 5, 15⟩ →
  ∃ d, ∀ p, plane_equation A (vector_from_points B C) p = 0 ↔ -4 * p.x + 1 * p.y + 4 * p.z - d = 0 :=
by {
  intros hA hB hC,
  use 76,
  intros p,
  split,
  { intro h,
    have : vector_from_points B C = ⟨-4, 1, 4⟩ := by {
      rw [hB, hC],
      simp [vector_from_points, Point3D.mk],
    },
    simp [plane_equation, this, Point3D.mk] at h,
    linarith },
  { intro h,
    have : vector_from_points B C = ⟨-4, 1, 4⟩ := by {
      rw [hB, hC],
      simp [vector_from_points, Point3D.mk],
    },
    simp [plane_equation, this, Point3D.mk],
    linarith },
  }
sorry
}

end plane_through_point_perpendicular_to_vector_l450_450839


namespace burattino_suspects_cheating_after_seventh_draw_l450_450061

theorem burattino_suspects_cheating_after_seventh_draw
  (balls : ℕ)
  (draws : ℕ)
  (a : ℝ)
  (p_limit : ℝ)
  (h_balls : balls = 45)
  (h_draws : draws = 6)
  (h_a : a = (39.choose 6 : ℝ) / (45.choose 6 : ℝ))
  (h_p_limit : p_limit = 0.01) :
  ∃ (n : ℕ), n > 5 ∧ a^n < p_limit := by
  sorry

end burattino_suspects_cheating_after_seventh_draw_l450_450061


namespace ratio_15_20_rounded_l450_450390

def ratio_rounded (a b : ℕ) : ℝ := (a : ℝ) / (b : ℝ)

theorem ratio_15_20_rounded :
  (ratio_rounded 15 20).round = 0.8 :=
by
  -- sorry is a placeholder for the actual proof.
  sorry

end ratio_15_20_rounded_l450_450390


namespace evaluate_expression_l450_450147

theorem evaluate_expression :
  let a := 1
  let b := 10
  let c := 100
  let d := 1000
  (a + b + c - d) + (a + b - c + d) + (a - b + c + d) + (-a + b + c + d) = 2222 :=
by
  let a := 1
  let b := 10
  let c := 100
  let d := 1000
  sorry

end evaluate_expression_l450_450147


namespace locus_of_midpoints_circle_l450_450730

theorem locus_of_midpoints_circle (O Q : Point) (r d : ℝ) (C : Circle) 
  (hO : center_of C O) (hr : radius_of C r) 
  (hd : 0 < d ∧ d < r) (hQ : distance O Q = d) :
    ∃ (M : Point), locus M (λ AB, midpoint_of_chord_passing_through Q AB) ∧ radius_of (locus_circle M) (d / 2) := 
sorry

end locus_of_midpoints_circle_l450_450730


namespace abs_inequality_solution_set_l450_450133

-- Define the main problem as a Lean theorem statement
theorem abs_inequality_solution_set (x : ℝ) : 
  (|x - 5| + |x + 3| ≥ 10 ↔ (x ≤ -4 ∨ x ≥ 6)) :=
by {
  sorry
}

end abs_inequality_solution_set_l450_450133


namespace third_side_length_l450_450179

theorem third_side_length (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) (h3 : 3 < x < 13) : (5 < x + 8) ∧ (x < 5 + 8) ∧ (5 < x + 3) ∧ (x < 8 + 5) := 
by
  sorry

end third_side_length_l450_450179


namespace problem_statement_l450_450215

open Real

noncomputable def f (x : ℝ) (ϕ : ℝ) := sin (2 * x + ϕ)

theorem problem_statement
  (ϕ : ℝ)
  (hϕ : -π/2 < ϕ ∧ ϕ < π/2)
  (h_sym : ∀ x, f x ϕ = f (3 * π / 8 - x) ϕ) :
  (f (x + π / 8) ϕ).odd ∧
  (∀ x1 x2, |f x1 ϕ - f x2 ϕ| = 2 → |x1 - x2| = π / 2) ∧
  (let g := λ x, f (x - 3 * π / 8) ϕ in 
    ∃ x, g x * (-cos x) = 4 / 9 * sqrt 3) := sorry

end problem_statement_l450_450215


namespace maximum_G_p_div_G_q_l450_450289

noncomputable def is_difference_2_multiple (p : ℕ) : Prop :=
  let u := p % 10
  let t := (p / 10) % 10
  let h := (p / 100) % 10
  let th := (p / 1000) % 10
  (th - h = 2) ∧ (t - u = 4) ∧ (u = 3)

noncomputable def is_difference_3_multiple (q : ℕ) : Prop :=
  let u := q % 10
  let t := (q / 10) % 10
  let h := (q / 100) % 10
  let th := (q / 1000) % 10
  (th - h = 3) ∧ (t - u = 6) ∧ (u = 3)

noncomputable def G (n : ℕ) : ℕ :=
  let u := n % 10
  let t := (n / 10) % 10
  let h := (n / 100) % 10
  let th := (n / 1000) % 10
  th + h + t + u

noncomputable def F (p q : ℕ) : ℤ :=
  (p - q) / 10

theorem maximum_G_p_div_G_q (p q : ℕ) (hp : is_difference_2_multiple p) (hq : is_difference_3_multiple q) 
(hint : ∃ k : ℤ, k = F(p, q) / (G(p) - G(q) + 3)) : 
  ∃ Max : ℚ, Max = 6/5 ∧ Max = G(p) / G(q) :=
by 
  sorry

end maximum_G_p_div_G_q_l450_450289


namespace molecular_weight_is_correct_l450_450833

/-- Assume the molecular weight of the compound is 1176 -/ 
constant molecular_weight : ℕ := 1176

/-- Assume the weight of 7 moles of the compound is 8232 -/ 
constant weight_of_7_moles : ℕ := 8232

/-- Prove that the molecular weight of the compound is 1176 given these conditions -/
theorem molecular_weight_is_correct : molecular_weight * 7 = weight_of_7_moles := by
  sorry

end molecular_weight_is_correct_l450_450833


namespace tiffany_math_homework_pages_l450_450455

theorem tiffany_math_homework_pages (x : ℕ) (h1 : ∀ y, y = 4) (h2 : ∀ z, z = 3) (h3 : ∀ w, w = 30) :
  x = 6 :=
by 
  have hreading : 4 * 3 = 12 := by norm_num,
  have htotal : 30 - 12 = 18 := by norm_num,
  have hmath : 18 / 3 = 6 := by norm_num,
  exact hmath

end tiffany_math_homework_pages_l450_450455


namespace triangle_third_side_length_l450_450197

theorem triangle_third_side_length (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 8) :
  (a + b > c) → (a + c > b) → (b + c > a) → c = 6 :=
by
  intros h₃ h₄ h₅
  rw [h₁, h₂] at *
  -- Simplified inequalities from the solution step
  have h₆ : 5 + 8 > c := by rw [h₁, h₂]; exact h₃
  have h₇ : 8 + c > 5 := by rw h₂; exact h₄
  have h₈ : 5 + c > 8 := by rw h₁; exact h₅
  sorry

end triangle_third_side_length_l450_450197


namespace burattino_suspects_cheating_after_seventh_draw_l450_450079

theorem burattino_suspects_cheating_after_seventh_draw 
  (total_balls : ℕ := 45) (drawn_balls : ℕ := 6) (a : ℝ := ((nat.choose 39 6 : ℕ) : ℝ) / ((nat.choose 45 6 : ℕ) : ℝ))
  (threshold : ℝ := 0.01) (probability : ℝ := 0.4) :
  (∃ n, a^n < threshold) → (∃ n > 5, a^n < threshold) :=
begin
  -- Definitions from conditions
  have fact_prob : a = ((nat.choose 39 6 : ℕ) : ℝ) / ((nat.choose 45 6 : ℕ) : ℝ), by refl,
  have fact_approx : a ≈ probability, by simp,

  -- Statement to prove
  intros h,
  use 6,
  split,
  { linarith, },
  { sorry }
end

end burattino_suspects_cheating_after_seventh_draw_l450_450079


namespace union_of_M_N_l450_450632

-- Define the sets M and N
def M : Set ℕ := {0, 2, 3}
def N : Set ℕ := {1, 3}

-- State the theorem to prove that M ∪ N = {0, 1, 2, 3}
theorem union_of_M_N : M ∪ N = {0, 1, 2, 3} :=
by
  sorry -- Proof goes here

end union_of_M_N_l450_450632


namespace sum_S7_l450_450372

-- Define the arithmetic sequence and sum conditions
def a1 : ℤ := 4
def a (n : ℕ) : ℤ := a1 + (n - 1) * (1 / 3)
def S (n : ℕ) : ℤ := n * a1 + (n * (n - 1) / 2) * (1 / 3)

-- Define the specific conditions in Lean 4
def condition1 : a1 = 4 := by rfl
def condition2 : a 6 + a 8 = 12 := by
  calc
    a 6 + a 8 = (a1 + 5 * (1 / 3)) + (a1 + 7 * (1 / 3)) : by simp [a]
           ... = 4 + 5 * (1 / 3) + (4 + 7 * (1 / 3)) : by congr; simp [a1]
           ... = 12                        : by linarith

-- The statement to be proven
theorem sum_S7 : S 7 = 35 :=
  sorry

end sum_S7_l450_450372


namespace length_of_parametric_curve_of_circle_l450_450599

noncomputable def parametric_curve_length : ℝ :=
  ∫ t in 0..2 * Real.pi, Real.sqrt ((Real.derivative (λ t, 3 * Real.sin t))^2 + (Real.derivative (λ t, 3 * Real.cos t))^2)

theorem length_of_parametric_curve_of_circle :
  parametric_curve_length = 6 * Real.pi :=
by
  sorry

end length_of_parametric_curve_of_circle_l450_450599


namespace length_BI_incenter_l450_450818

theorem length_BI_incenter:
  ∀ (A B C I : Type) [geometry A B C I],
  is_right_triangle A B C → 
  angle A B C = 90 → 
  length B C > length A B → 
  length A B = 6 → 
  angle B A C = 30 →
  incenter_of_triangle A B C I → 
  length B I = sqrt(6) - 3 * sqrt(2) :=
by sorry

end length_BI_incenter_l450_450818


namespace dragon_rope_problem_l450_450532

noncomputable def calculate_p_q_r_sum 
    (total_rope_length : ℕ := 25)
    (castle_radius : ℕ := 5)
    (dragon_height : ℕ := 3)
    (tangent_length : ℕ := 3)
    (p q r : ℕ := by exact 75, exact 450, exact 3) 
    (h_prime_r : Nat.Prime r) : ℕ :=
  let rope_touch_length := (p - Nat.sqrt q) / r
  in (p + q + r)

theorem dragon_rope_problem :
  calculate_p_q_r_sum 25 5 3 3 75 450 3 (by norm_num) = 528 :=
by 
  dsimp [calculate_p_q_r_sum]
  norm_num
  unfold Nat.sqrt
  norm_num
  sorry

end dragon_rope_problem_l450_450532


namespace monotonic_increasing_of_derivative_positive_l450_450560

variable {X : Type} [Real]

theorem monotonic_increasing_of_derivative_positive 
  {f : X → X} 
  (h_diff : ∀ x : X, DifferentiableAt ℝ f x)
  (h_pos : ∀ x : X, (derivative f) x > 0) : 
  MonotonicIncreasing f :=
sorry

end monotonic_increasing_of_derivative_positive_l450_450560


namespace exists_all_white_2x2_grid_l450_450976

theorem exists_all_white_2x2_grid (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) 
    (grid : Fin m × Fin n → Bool) 
    (h : ∃ k, k ≥ m * n / 3 ∧ ∀ i j, (i < m - 1) ∧ (j < n - 1) → (grid (⟨i, h_m⟩, ⟨j, h_n⟩) = tt ∧ grid (⟨i+1, h_m⟩, ⟨j, h_n⟩) = tt ∧ grid (⟨i, h_m⟩, ⟨j+1, h_n⟩) = tt ∧ grid (⟨i+1, h_m⟩, ⟨j+1, h_n⟩) = ff)) :
  ∃ i j, (i < m - 1) ∧ (j < n - 1) ∧ (grid (⟨i, h_m⟩, ⟨j, h_n⟩) = tt ∧ grid (⟨i+1, h_m⟩, ⟨j, h_n⟩) = tt ∧ grid (⟨i, h_m⟩, ⟨j+1, h_n⟩) = tt ∧ grid (⟨i+1, h_m⟩, ⟨j+1, h_n⟩) = tt) :=
sorry

end exists_all_white_2x2_grid_l450_450976


namespace smallest_possible_value_of_N_l450_450229

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l450_450229


namespace max_students_with_equal_distribution_l450_450552

theorem max_students_with_equal_distribution (pens pencils : ℕ) (h_pens : pens = 3540) (h_pencils : pencils = 2860) :
  gcd pens pencils = 40 :=
by
  rw [h_pens, h_pencils]
  -- Proof steps will go here
  sorry

end max_students_with_equal_distribution_l450_450552


namespace product_of_two_numbers_l450_450816

-- State the conditions and the proof problem
theorem product_of_two_numbers (x y : ℤ) (h_sum : x + y = 30) (h_diff : x - y = 6) :
  x * y = 216 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end product_of_two_numbers_l450_450816


namespace smallest_possible_value_of_N_l450_450228

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l450_450228


namespace cylindrical_tank_volume_increase_l450_450873

theorem cylindrical_tank_volume_increase (k : ℝ) (H R : ℝ) 
  (hR : R = 10) (hH : H = 5)
  (condition : (π * (10 * k)^2 * 5 - π * 10^2 * 5) = (π * 10^2 * (5 + k) - π * 10^2 * 5)) :
  k = (1 + Real.sqrt 101) / 10 :=
by
  sorry

end cylindrical_tank_volume_increase_l450_450873


namespace a_and_b_together_complete_work_in_6_days_l450_450843

-- Definitions used from the condition
def days_to_complete_alone (days : ℝ) : ℝ := 1 / days

-- Constants based on the given conditions
def a_days : ℝ := 12
def b_days : ℝ := 12

-- Work rates for A and B based on the conditions
def work_rate_A : ℝ := days_to_complete_alone a_days
def work_rate_B : ℝ := days_to_complete_alone b_days

-- Target proof statement
theorem a_and_b_together_complete_work_in_6_days :
  1 / (work_rate_A + work_rate_B) = 6 := 
by
  -- Here lies the place for proof steps, which are skipped.
  sorry

end a_and_b_together_complete_work_in_6_days_l450_450843


namespace right_triangle_sides_unique_l450_450428

theorem right_triangle_sides_unique (a b c : ℕ) 
  (relatively_prime : Int.gcd (Int.gcd a b) c = 1) 
  (right_triangle : a ^ 2 + b ^ 2 = c ^ 2) 
  (increased_right_triangle : (a + 100) ^ 2 + (b + 100) ^ 2 = (c + 140) ^ 2) : 
  (a = 56 ∧ b = 33 ∧ c = 65) :=
by
  sorry 

end right_triangle_sides_unique_l450_450428


namespace picnic_midpoint_l450_450748

theorem picnic_midpoint :
  let Mark := (0, 8) in
  let Sandy := (-6, -2) in
  let midpoint := ((Mark.1 + Sandy.1) / 2, (Mark.2 + Sandy.2) / 2) in
  midpoint = (-3, 3) :=
by
  sorry

end picnic_midpoint_l450_450748


namespace find_median_of_100_l450_450302

noncomputable def median_of_set (s : Finset ℝ) : ℝ :=
if h : ∃ median, is_median s median then classical.some h else 0

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
∃ (f : ℤ), (f : ℝ) = (card s : ℝ) / 2 ∧
    ∃ (low : Finset ℝ) (high : Finset ℝ),
        low ⊆ s ∧ high ⊆ s ∧
        card low = floor ((card s : ℝ) / 2) ∧
        card high = ceil ((card s : ℝ) / 2) ∧
        (∀ x ∈ low, x ≤ m) ∧ (∀ x ∈ high, x ≥ m)

theorem find_median_of_100 (s : Finset ℝ) (h_size : s.card = 100)
(h1 : ∃ x ∈ s, median_of_set (s.erase x) = 78)
(h2 : ∃ y ∈ s, median_of_set (s.erase y) = 66) :
  median_of_set s = 72 := by
sorry

end find_median_of_100_l450_450302


namespace sufficient_not_necessary_condition_l450_450979

variable (m : ℤ)

def set_A : Set ℤ := {1, m^2}
def set_B : Set ℤ := {2, 4}

theorem sufficient_not_necessary_condition :
  (A ∩ B = {4} → m^2 = 4) ∧ (m = -2 → (m = 2 ∨ m = -2)) :=
by
  sorry

end sufficient_not_necessary_condition_l450_450979


namespace median_of_100_numbers_l450_450333

theorem median_of_100_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 100)
  (h_median_99_1 : ∀ num ∈ numbers, median (numbers.erase num) = 78 → num ∈ numbers)
  (h_median_99_2 : ∀ num ∈ numbers, median (numbers.erase num) = 66 → num ∈ numbers) :
  median numbers = 72 :=
sorry

end median_of_100_numbers_l450_450333


namespace problem1_problem2_l450_450087

noncomputable def expression1 : ℝ :=
  (1.5) ^ (-2) + (-9.6) ^ 0 - (3 + 3 / 8) ^ (-2 / 3) + real.sqrt ((real.pi - 4) ^ 2) + real.cbrt ((real.pi - 2) ^ 3)

theorem problem1 : expression1 = 3 :=
by sorry

noncomputable def log3 (x : ℝ) : ℝ := real.log x / real.log 3

noncomputable def expression2 : ℝ :=
  2 * log3 2 - log3 (32 / 9) + log3 8

theorem problem2 : expression2 = 2 :=
by sorry

end problem1_problem2_l450_450087


namespace inclination_angle_range_l450_450811

theorem inclination_angle_range (α : ℝ) (θ : ℝ) :
  (∃ (x y : ℝ), x*cos α + sqrt(3)*y + 2 = 0) →
  (0 ≤ θ) ∧ (θ < π) →
  (-√3 / 3 ≤ tan θ) ∧ (tan θ ≤ √3 / 3) →
  θ ∈ ([0, π/6] ∪ [5*π/6, π)) :=
sorry

end inclination_angle_range_l450_450811


namespace sam_age_l450_450568

-- Definitions
variables (B J S : ℕ)
axiom H1 : B = 2 * J
axiom H2 : B + J = 60
axiom H3 : S = (B + J) / 2

-- Problem statement
theorem sam_age : S = 30 :=
sorry

end sam_age_l450_450568


namespace angle_ADB_is_right_l450_450035

variables {A B C D : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables (ABC : A) (BCA : B) (CAB : C) (ABD : D)

-- Define points and distances
variable (AB_CD : ℝ)
variable (BC : ℝ)
variable (CA : ℝ)
variable (circumference_C_AB : ℝ)

-- Given Conditions
def is_right_triangle (BCA : A) := (∠BCA = 90)
def circle_with_center_C (C : A) (A B D : B) := 
  dist C A = dist C B ∧ dist C B = dist C D
def extend_AB_intersect_circle (AB : B) (D : B) (circle : circle_with_center_C) :=
  line_through A B ∩ circle D

-- Proof Statement
theorem angle_ADB_is_right (ABC : A) (BCA : B) (CAB : C) (ABD : D) 
  (AB_CD : ℝ := 13) (BC : ℝ := 12) (CA : ℝ := (sqrt (13^2 - 12^2))) -- Pythagorean theorem
  (circumference_C_AB : circle_with_center_C C A B D) : 
    is_right_triangle ABC → angle (ABD) = 90 :=
by sorry

end angle_ADB_is_right_l450_450035


namespace incorrect_conclusion_c_l450_450541

-- Definitions from the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = f(x)
def two_periodic (f : ℝ → ℝ) : Prop := ∀ x, f(x + 2) = f(x) + f(1)
def increasing_on_01 (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → y ≤ 1 → f(x) ≤ f(y)

-- Problem statement
theorem incorrect_conclusion_c (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_periodic : two_periodic f) 
  (h_increasing : increasing_on_01 f)
  (h_false : ¬(∀ x, -3 ≤ x → x ≤ -2 → f'(x) ≥ 0)) : 
  ∃ k : ℤ, f(x + 2 * k) = f(x) :=
sorry

end incorrect_conclusion_c_l450_450541


namespace lines_not_parallel_not_perpendicular_same_plane_l450_450985

variables {L : Type*} [linear_ordered_field L] 
variables (m n : affine.line L) (α β : affine.plane L)

theorem lines_not_parallel_not_perpendicular_same_plane 
  (h1 : m ≠ n) (h2 : ¬parallel m n) : 
  ¬ (∃ π : affine.plane L, m ⊥ π ∧ n ⊥ π) :=
sorry

end lines_not_parallel_not_perpendicular_same_plane_l450_450985


namespace some_number_is_five_l450_450605

theorem some_number_is_five (x : ℕ) (some_number : ℕ) (h1 : x = 5) (h2 : x / some_number + 3 = 4) : some_number = 5 := by
  sorry

end some_number_is_five_l450_450605


namespace paint_snake_2016_cubes_l450_450355

/-- The amount of paint needed to paint a snake composed of 2016 cubes, given that it takes 60 grams of paint to paint a cube on all sides, is 80660 grams. -/
theorem paint_snake_2016_cubes :
  let paint_per_cube := 60
  let total_cubes := 2016
  let segments := total_cubes / 6
  let paint_per_segment := 240
  let additional_paint := 20
  total_cubes * paint_per_cube - (4 * segments / 3) * paint_per_cube + additional_paint = 80660 := 
begin 
   sorry
end

end paint_snake_2016_cubes_l450_450355


namespace lambda_in_triangle_l450_450693

open Real

variables {V : Type*} [inner_product_space ℝ V]

theorem lambda_in_triangle {a b : V} (D_altitude : ∃ D : V, collinear ({a, b, D} : set V) ∧ ⟪D - a, a⟫ = 0) :
  ∃ λ : ℝ, (λ = (⟪a, a - b⟫ / ∥a - b∥^2)) :=
begin
  sorry
end

end lambda_in_triangle_l450_450693


namespace find_angle_between_vectors_find_vector_sum_magnitude_l450_450162

variables (a b : EuclideanSpace ℝ 3) (h1 : ‖a‖ = 4) (h2 : ‖b‖ = 3) (h3 : 2 • a - 3 • b ⋅ 2 • a + b = 61)

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ 3) :=
  real.arccos ((a ⋅ b) / (‖a‖ * ‖b‖))

theorem find_angle_between_vectors : 
  angle_between_vectors a b = real.arccos (- 5 / 6) := 
sorry

theorem find_vector_sum_magnitude : 
  ‖a + b‖ = 7 := 
sorry

end find_angle_between_vectors_find_vector_sum_magnitude_l450_450162


namespace box_height_is_6_l450_450861

-- Defining the problem setup
variables (h : ℝ) (r_large r_small : ℝ)
variables (box_size : ℝ) (n_spheres : ℕ)

-- The conditions of the problem
def rectangular_box :=
  box_size = 5 ∧ r_large = 3 ∧ r_small = 1.5 ∧ n_spheres = 4 ∧
  (∀ k : ℕ, k < n_spheres → 
   ∃ C : ℝ, 
     (C = r_small) ∧ 
     -- Each smaller sphere is tangent to three sides of the box condition
     (C ≤ box_size))

def sphere_tangency (h r_large r_small : ℝ) :=
  h = 2 * r_large ∧ r_large + r_small = 4.5

def height_of_box (h : ℝ) := 2 * 3 = h

-- The mathematically equivalent proof problem
theorem box_height_is_6 (h : ℝ) (r_large : ℝ) (r_small : ℝ) (box_size : ℝ) (n_spheres : ℕ) 
  (conditions : rectangular_box box_size r_large r_small n_spheres) 
  (tangency : sphere_tangency h r_large r_small) :
  height_of_box h :=
by {
  -- Proof is omitted
  sorry
}

end box_height_is_6_l450_450861


namespace Paul_average_homework_l450_450118

theorem Paul_average_homework :
  let weeknights := 5,
      weekend_homework := 5,
      night_homework := 2,
      nights_no_homework := 2,
      days_in_week := 7,
      total_homework := weekend_homework + night_homework * weeknights,
      available_nights := days_in_week - nights_no_homework,
      average_homework_per_night := total_homework / available_nights
  in average_homework_per_night = 3 := 
by
  sorry

end Paul_average_homework_l450_450118


namespace value_of_expr_l450_450961

theorem value_of_expr (a : Int) (h : a = -2) : a + 1 = -1 := by
  -- Placeholder for the proof, assuming it's correct
  sorry

end value_of_expr_l450_450961


namespace quadrilateral_circumcenter_l450_450895

-- Definitions for each type of quadrilateral.
def is_square (Q : Type) [quadrilateral Q] : Prop := 
  regular Q ∧ equilateral Q ∧ equal_angles Q

def is_rectangle (Q : Type) [quadrilateral Q] : Prop := 
  opposite_sides_equal Q ∧ right_angles Q ∧ ¬(is_square Q)

def is_rhombus (Q : Type) [quadrilateral Q] : Prop := 
  equilateral Q ∧ ¬(equal_angles Q ∧ right_angles Q)

def is_parallelogram (Q : Type) [quadrilateral Q] : Prop := 
  opposite_sides_equal Q ∧ opposite_angles_equal Q ∧ ¬(is_rectangle Q ∨ is_rhombus Q)

def is_isosceles_trapezoid (Q : Type) [quadrilateral Q] : Prop := 
  one_pair_of_parallel_sides Q ∧ non_parallel_sides_equal Q ∧ equal_base_angles

-- The circumcenter existence property
def has_circumcenter (Q : Type) [quadrilateral Q] : Prop :=
  ∃ (O : Point), ∀ (V ∈ vertices Q), dist O V = r ∧ r > 0 -- Circumcenter condition

-- The theorem to be stated in Lean
theorem quadrilateral_circumcenter (Q : Type) [quadrilateral Q] : 
  let squares_have it := 
    has_circumcenter Q ↔ (is_square Q ∨ is_rectangle Q ∨ is_isosceles_trapezoid Q) :=
sorry

end quadrilateral_circumcenter_l450_450895


namespace rational_m_abs_nonneg_l450_450283

theorem rational_m_abs_nonneg (m : ℚ) : m + |m| ≥ 0 :=
by sorry

end rational_m_abs_nonneg_l450_450283


namespace total_surface_area_of_rectangular_solid_is_422_l450_450935

noncomputable def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_prime_edge_length (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c

def volume (a b c : ℕ) : ℕ := a * b * c

def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

theorem total_surface_area_of_rectangular_solid_is_422 :
  ∃ (a b c : ℕ), is_prime_edge_length a b c ∧ volume a b c = 399 ∧ surface_area a b c = 422 :=
begin
  sorry
end

end total_surface_area_of_rectangular_solid_is_422_l450_450935


namespace least_five_digit_congruent_to_7_mod_17_l450_450831

theorem least_five_digit_congruent_to_7_mod_17 : ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 17 = 7 ∧ (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 17 = 7 → n ≤ m) :=
sorry

end least_five_digit_congruent_to_7_mod_17_l450_450831


namespace quadratic_intersects_xaxis_once_l450_450687

theorem quadratic_intersects_xaxis_once (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0) ↔ k = 1 :=
by
  sorry

end quadratic_intersects_xaxis_once_l450_450687


namespace ellipse_standard_equation_proof_l450_450974

variable {ℝ : Type}
noncomputable def ellipse_center_origin : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (c^2 = a^2 - b^2 ∧  sqrt 3 = c ∧ a = 2 ∧ 
  equation_of_ellipse x y -> (x^2)/(a^2) + (y^2)/(b^2) = 1)

noncomputable def midpoint_intersection (x y : ℝ) : Prop :=
  ∃ (A : ℝ × ℝ) (k : ℝ), A = (1, 1/2) ∧
  equation_of_line x y A k -> (x - 1)/(2 * k) = y - 1 /2

noncomputable def standard_equation_question : Prop :=
  equation (x^2)/(4:ℝ) + y^2 - ℝ = 0
  
noncomputable def line_intersects_chord_question (x y : ℝ) : Prop :=
  equation x + 2*y - 2 = 0

theorem ellipse_standard_equation_proof :
  ellipse_center_origin ∧ midpoint_intersection (1, 1/2) -> 
  standard_equation_question ∧ line_intersects_chord_question (1, 1/2) :=
  sorry

end ellipse_standard_equation_proof_l450_450974


namespace value_of_f_l450_450653

def f (x z : ℕ) (y : ℕ) : ℕ := 2 * x^2 + y - z

theorem value_of_f (y : ℕ) (h1 : f 2 3 y = 100) : f 5 7 y = 138 := by
  sorry

end value_of_f_l450_450653


namespace find_a1_l450_450368

theorem find_a1 (f : ℝ → ℝ) (a : ℕ → ℝ) (h₀ : ∀ x, f x = (x - 1)^3 + x + 2)
(h₁ : ∀ n, a (n + 1) = a n + 1/2)
(h₂ : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 18) :
a 1 = -1 / 4 :=
by
  sorry

end find_a1_l450_450368


namespace team_dig_time_l450_450384

def father's_depth : ℝ := 4 * 400
def mother's_depth : ℝ := 5 * 350

def michael_desired_depth : ℝ := 3 * father's_depth - 500
def emma_desired_depth : ℝ := 2 * mother's_depth + 250
def lucas_desired_depth : ℝ := (father's_depth + mother's_depth) - 150

def combined_rate : ℝ := 3 + 5 + 4
def required_depth : ℝ := max michael_desired_depth (max emma_desired_depth lucas_desired_depth)

def time_to_dig : ℝ := required_depth / combined_rate

theorem team_dig_time : ceil(time_to_dig) = 359 :=
by
  sorry -- Proof to be provided

end team_dig_time_l450_450384


namespace range_of_m_l450_450662

def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B (m : ℝ) : Set ℝ := {x | abs (x - 3) ≤ m}
def p (x : ℝ) : Prop := x ∈ A
def q (x : ℝ) (m : ℝ) : Prop := x ∈ B m

theorem range_of_m (m : ℝ) (hm : m > 0):
  (∀ x, p x → q x m) ↔ (6 ≤ m) := by
  sorry

end range_of_m_l450_450662


namespace firstVisitExceptTigger_l450_450504

   /- Define the number of honey pots each of Winnie-the-Pooh's friends has -/
   def tiggerHoney : ℕ := 1
   def pigletHoney : ℕ := 2
   def owlHoney : ℕ := 3
   def eeyoreHoney : ℕ := 4
   def rabbitHoney : ℕ := 5

   /- Define a function that computes the number of honey pots Winnie-the-Pooh has after visiting all friends except the first one -/
   def totalHoneyAfterFirstVisit (firstVisit : ℕ) : ℕ :=
     let totalHoney := tiggerHoney + pigletHoney + owlHoney + eeyoreHoney + rabbitHoney
     totalHoney - firstVisit - 4

   /- Define the main theorem to be proven -/
   theorem firstVisitExceptTigger (firstVisit : ℕ) :
     firstVisit ∈ {pigletHoney, owlHoney, eeyoreHoney, rabbitHoney} → totalHoneyAfterFirstVisit firstVisit = 10 :=
   by
     intros h
     sorry
   
end firstVisitExceptTigger_l450_450504


namespace balloon_permutations_l450_450666

theorem balloon_permutations : 
  (Nat.factorial 7 / 
  ((Nat.factorial 1) * 
  (Nat.factorial 1) * 
  (Nat.factorial 2) * 
  (Nat.factorial 2) * 
  (Nat.factorial 1))) = 1260 := by
  sorry

end balloon_permutations_l450_450666


namespace median_of_100_numbers_l450_450326

theorem median_of_100_numbers (numbers : List ℝ) (h_length : numbers.length = 100)
  (h_sorted : numbers.sorted (≤))
  (h_51 : numbers.nth_le 50 h_51_nat = 78) -- Note: nth_le is zero-indexed, so 51st element is 50th index.
  (h_50 : numbers.nth_le 49 h_50_nat = 66) : 
  (numbers.nth_le 49 h_50_nat + numbers.nth_le 50 h_51_nat) / 2 = 72 :=
sorry

end median_of_100_numbers_l450_450326


namespace smallest_N_possible_l450_450025

theorem smallest_N_possible : ∃ N : ℕ, (∀ (students : ℕ) (candies : ℕ), 
  students = 25 → 
  ∃ k : ℕ, candies = k * students ∧ 
  (∀ (solve_eq_tasks : ℕ → ℕ) (i : ℕ), 
    i < students → solve_eq_tasks i ∈ {0, 1, ..., solve_eq_tasks students - 1}) → 
  N = 25 * 24 ∧ ∀ k : ℕ, (N = 25 * k) → k ≥ 24 ) :=
begin
  sorry
end

end smallest_N_possible_l450_450025


namespace part1_part2_part3_l450_450287

variables (a b c : ℤ)
-- Condition: For all integer values of x, (ax^2 + bx + c) is a square number 
def quadratic_is_square_for_any_x (a b c : ℤ) : Prop :=
  ∀ x : ℤ, ∃ k : ℤ, a * x^2 + b * x + c = k^2

-- Question (1): Prove that 2a, 2b, c are all integers
theorem part1 (h : quadratic_is_square_for_any_x a b c) : 
  ∃ m n : ℤ, 2 * a = m ∧ 2 * b = n ∧ ∃ k₁ : ℤ, c = k₁ :=
sorry

-- Question (2): Prove that a, b, c are all integers, and c is a square number
theorem part2 (h : quadratic_is_square_for_any_x a b c) : 
  ∃ k₁ k₂ m n : ℤ, a = k₁ ∧ b = k₂ ∧ c = m^2 :=
sorry

-- Question (3): Prove that if (2) holds, it does not necessarily mean that 
-- for all integer values of x, (ax^2 + bx + c) is always a square number.
theorem part3 (a b c : ℤ) (h : ∃ k₁ k₂ m n : ℤ, a = k₁ ∧ b = k₂ ∧ c = m^2) : 
  ¬ quadratic_is_square_for_any_x a b c :=
sorry

end part1_part2_part3_l450_450287


namespace largest_multiples_of_3_is_9999_l450_450502

theorem largest_multiples_of_3_is_9999 :
  ∃ n : ℕ, (n = 9999 ∧ n < 10000 ∧ 1000 ≤ n ∧ 3 ∣ n) ∧ 
  (∀ k : ℕ, (k < 10000 ∧ 1000 ≤ k ∧ 3 ∣ k) → k ≤ n) :=
by
  sorry

end largest_multiples_of_3_is_9999_l450_450502


namespace apple_weight_susan_l450_450053

theorem apple_weight_susan 
  (P : ℚ) (B : ℚ) (T : ℚ) (S : ℚ) (sliced_weight : ℚ) (w : ℚ)
  (hP : P = 38.25)
  (hB : B = P + 8.5)
  (hT : T = (3 / 8) * B)
  (hsliced : sliced_weight = (T / 2) * 75)
  (hS : S = (1 / 2) * T + 7)
  (h90 : w = (S * 0.9) * 150) 
  :
  w = 2128.359375 := 
  sorry

end apple_weight_susan_l450_450053


namespace exponential_function_strictly_increasing_l450_450161

-- Given conditions
variable {m n : ℝ}
variable (h : 2^m > 2^n)

-- Prove that m > n
theorem exponential_function_strictly_increasing (h : 2^m > 2^n) : m > n :=
sorry

end exponential_function_strictly_increasing_l450_450161


namespace inequality_solution_l450_450775

theorem inequality_solution (x : ℝ) : 
  3 - (1 / (4 * x + 6)) ≤ 5 ↔ x ∈ set.Ioo (-∞, -3/2) ∪ set.Ioo (-1/8, ∞) :=
by
  sorry

end inequality_solution_l450_450775


namespace total_respondents_l450_450298

theorem total_respondents (h1 : ∀ n : ℕ, ∃ x : ℝ, 0.382 * x = 29)
  (h2 : ∀ x : ℝ, ∃ y : ℝ, 0.753 * y = x)
  (h3 : ∀ n : ℕ, ∃ x : ℝ, x ≈ 75.92)
  (h4 : ∀ x : ℝ, ∃ y : ℝ, y ≈ 100.93) 
  : ∃ y : ℕ, y = 101 :=
by
  sorry

end total_respondents_l450_450298


namespace ratio_XY_7_l450_450572

variable (Z : ℕ)
variable (population_Z : ℕ := Z)
variable (population_Y : ℕ := 2 * Z)
variable (population_X : ℕ := 14 * Z)

theorem ratio_XY_7 :
  population_X / population_Y = 7 := by
  sorry

end ratio_XY_7_l450_450572


namespace complex_computation_l450_450919

theorem complex_computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end complex_computation_l450_450919


namespace smallest_sector_angle_divided_circle_l450_450363

theorem smallest_sector_angle_divided_circle : ∃ a d : ℕ, 
  (2 * a + 7 * d = 90) ∧ 
  (8 * (a + (a + 7 * d)) / 2 = 360) ∧ 
  a = 38 := 
by
  sorry

end smallest_sector_angle_divided_circle_l450_450363


namespace books_configuration_count_l450_450029

theorem books_configuration_count :
  let books := 10 in
  let conditions (library: ℕ) (checked_out: ℕ) := 
    (2 ≤ library) ∧ (2 ≤ checked_out) ∧ (library + checked_out = books) in
  ∃ (count: ℕ), 
  count = 7 ∧ 
  ∀ library checked_out, conditions library checked_out → ∃ k, (2 + k = library ∨ 8 - k = library) :=
by
  sorry

end books_configuration_count_l450_450029


namespace joe_toy_cars_l450_450361

theorem joe_toy_cars (initial_count : ℕ) (growth_factor : ℕ) (tripling_factor : ℕ) (new_count : ℕ) (final_count : ℕ) :
  initial_count = 50 →
  growth_factor = 2.5 →
  new_count = initial_count * growth_factor →
  tripling_factor = 3 →
  final_count = new_count * tripling_factor →
  final_count = 375 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h4] at h3
  have : (50 : ℕ) * (2.5 : ℕ) = 125 := sorry
  rw this at h3
  rw h3 at h5
  have : (125 : ℕ) * (3 : ℕ) = 375 := sorry
  rw this at h5
  exact h5

end joe_toy_cars_l450_450361


namespace geometric_sequence_sum_first_n_terms_S_n_l450_450995

noncomputable def point_in_region (x y : ℝ) (n : ℕ) : Prop :=
  (x + 2 * y ≤ 2 * n) ∧ (0 ≤ x) ∧ (0 ≤ y)

noncomputable def maximum_value_z_n (n : ℕ) : ℝ := 2 * n

noncomputable def S_n (n a_n : ℕ) : ℝ := 2 * n - a_n

noncomputable def a_n (a_n_minus_1 S_n S_n_minus_1 : ℝ) : ℝ := S_n - S_n_minus_1 + a_n_minus_1 - 2

theorem geometric_sequence (n : ℕ) : 
  ∀ (a_n : ℕ), a_n - 2 = -1 * (1 / 2)^(n - 1) :=
sorry

theorem sum_first_n_terms_S_n (n : ℕ) : 
  ∃ (T_n : ℕ), T_n = n^2 - n + 2 - (1 / 2)^(n - 1) :=
sorry

end geometric_sequence_sum_first_n_terms_S_n_l450_450995


namespace number_of_good_committees_l450_450549

def good_committee 
(n : ℕ) 
(has_enemies : ℕ → ℕ)
(friends_and_enemies : Type) 
(pairwise_relation : friends_and_enemies → bool) : Prop :=
∃ (total_members : ℕ) (memb : friends_and_enemies → bool), 
  total_members = 30 ∧
  (∀ (x : friends_and_enemies), memb x → (has_enemies x = 6)) ∧
  (∀ (x y : friends_and_enemies), pairwise_relation (x, y) ∨ pairwise_relation (y, x)) ∧
  ∃ (comm : list friends_and_enemies), ∑ c in comm, 
  ((∀ (u v w: friends_and_enemies), (u, v, w) ∈ comm → 
    (pairwise_relation (u, v) ∧ pairwise_relation (v, w) ∧ pairwise_relation (u, w)) ∨
    (¬pairwise_relation (u, v)) ∧ ¬pairwise_relation (v, w) ∧ ¬pairwise_relation (u, w))) ∧ 
    (comm.length = 1990))

theorem number_of_good_committees : 
  ∃ (members : bool) (n : ℕ), 
  (good_committee n (λ x, if x < 7 then 6 else 0) friends_and_enemies (λ x, tt)) :=
begin
  sorry
end

end number_of_good_committees_l450_450549


namespace valid_third_side_length_l450_450177

theorem valid_third_side_length {x : ℝ} (h1 : 5 + 8 > x) (h2 : 5 + x > 8) (h3 : 8 + x > 5) : x = 6 :=
by
  -- Given 5 + 8 > x, 5 + x > 8, 8 + x > 5
  have range1 : 13 > x := h1,
  have range2 : x > 3 := (by linarith [h2]),
  have _ : 3 < 6 ∧ 6 < 13 := by norm_num,
  linarith

#check valid_third_side_length

end valid_third_side_length_l450_450177


namespace find_angle_OKE_l450_450759

structure Triangle :=
(A B C : Point)
(right_angle : Line.isPerpendicular (Line.mk C A) (Line.mk C B))

def point_on_segment (A B P : Point) : Prop := 
  collinear A B P ∧ segment_contains A B P

def foot_of_perpendicular (A : Point) (l : Line) : Point := 
  some (exists_foot_of_perpendicular A l)

structure GeometrySetup :=
(TriangleABC : Triangle)
(P : Point)
(P_on_AC : point_on_segment TriangleABC.A TriangleABC.C P)
(D : Point)
(D_perpendicular : foot_of_perpendicular TriangleABC.A (Line.mk TriangleABC.B P) = D)
(E : Point)
(E_perpendicular : foot_of_perpendicular P (Line.mk TriangleABC.A TriangleABC.B) = E)
(T : Point)
(circumcircle_PA : Circle)
(tangent_A : tangent_to_circle T TriangleABC.A circumcircle_PA)
(tangent_P : tangent_to_circle T P circumcircle_PA)
(center_O : Point)
(center_O_def : Circle.center circumcircle_PA = center_O)
(Q : Point)
(perpendicular_from_T : foot_of_perpendicular T (Line.mk D E) = Q ∧ collinear Q TriangleABC.B TriangleABC.C)
(K : Point)
(parallel_C_OQ : parallel (Line.mk TriangleABC.C K) (Line.mk center_O Q) ∧ collinear K (Line.mk TriangleABC.B center_O))

theorem find_angle_OKE (setup : GeometrySetup) : 
  ∠ (setup.center_O) setup.K setup.E = 90 :=
sorry

end find_angle_OKE_l450_450759


namespace player1_wins_11th_round_probability_l450_450481

-- Definitions based on the conditions
def egg_shell_strength (n : ℕ) : ℝ := sorry
def player1_won_first_10_rounds : Prop := sorry

-- Main theorem
theorem player1_wins_11th_round_probability
  (h : player1_won_first_10_rounds) :
  Prob (egg_shell_strength 11 > egg_shell_strength 12) = 11 / 12 := sorry

end player1_wins_11th_round_probability_l450_450481


namespace range_of_a_l450_450150

-- Define the continuous function g(x) with the given properties.
variable (g : ℝ → ℝ)
variable (hg_continuous : Continuous g)
variable (hg_monotonic : ∀ x > 0, Deriv g x > 0)
variable (hg_even : ∀ x, g x = g (-x))

-- Define the function f(x) with the given properties.
variable (f : ℝ → ℝ)
variable (hf_periodic : ∀ x, f (sqrt 3 + x) = -f x)
variable (hf_def : ∀ x ∈ Set.Icc 0 (sqrt 3), f x = x^3 - 3 * x)

-- Given condition: inequality holds for x in [-3, 3]
variable (hg_inequality : ∀ x ∈ Set.Icc (-3) 3, g (f x) ≤ g (a^2 - a + 2))

-- Goal: Prove the range for a
theorem range_of_a (a : ℝ) : a ≥ 1 ∨ a ≤ 0 :=
sorry

end range_of_a_l450_450150


namespace log_base_16_of_256_l450_450121

open Real

theorem log_base_16_of_256 : logBase 16 256 = 32 := by
  have h1 : 16 = 2^4 := by
    norm_num
  have h2 : 256 = 2^8 := by
    norm_num
  rw [h1, h2, logBase_pow, logBase_pow]
  norm_num

end log_base_16_of_256_l450_450121


namespace burattino_suspects_cheating_l450_450075

theorem burattino_suspects_cheating :
  ∃ n : ℕ, (0.4 ^ n) < 0.01 ∧ n + 1 = 7 :=
by
  sorry

end burattino_suspects_cheating_l450_450075


namespace smallest_possible_N_l450_450261

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l450_450261


namespace negation_of_proposition_l450_450432

theorem negation_of_proposition : (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 := by
  sorry

end negation_of_proposition_l450_450432


namespace ball_hits_ground_in_2_5_seconds_l450_450791

theorem ball_hits_ground_in_2_5_seconds :
  ∃ t : ℝ, y t = 0 ∧ t = 2.5 :=
  let y (t : ℝ) : ℝ := -8 * t^2 - 12 * t + 80 in
by
  sorry

end ball_hits_ground_in_2_5_seconds_l450_450791


namespace original_function_properties_l450_450823

noncomputable theory

def transformation_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x, f (x - a)
def transformation_stretch (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x, f (x / b)

def g : ℝ → ℝ := λ x, Real.cos x

theorem original_function_properties :
  ∃ f : ℝ → ℝ, 
  (∀ x, f x = Real.cos (2 * x - π/5)) ∧
  (periodic f π) ∧ 
  (¬(∀ x, f (-x) = f x) ∧ ¬(∀ x, f (-x) = -f x)) :=
begin
  sorry
end

end original_function_properties_l450_450823


namespace common_ratio_of_geometric_sequence_is_4_l450_450970

theorem common_ratio_of_geometric_sequence_is_4 
  (a_n : ℕ → ℝ) 
  (b_n : ℕ → ℝ) 
  (d : ℝ) 
  (h₁ : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h₂ : d ≠ 0)
  (h₃ : (a_n 3)^2 = (a_n 2) * (a_n 7)) :
  b_n 2 / b_n 1 = 4 :=
sorry

end common_ratio_of_geometric_sequence_is_4_l450_450970


namespace median_of_100_set_l450_450321

theorem median_of_100_set 
  (S : Finset ℝ) (h_card : S.card = 100)
  (h_remove1 : ∃ x ∈ S, median (S.erase x) = 78)
  (h_remove2 : ∃ y ∈ S, median (S.erase y) = 66) : 
  median S = 72 :=
by
  sorry

end median_of_100_set_l450_450321


namespace total_students_l450_450052

theorem total_students (n x : ℕ) (h1 : 3 * n + 48 = 6 * n) (h2 : 4 * n + x = 2 * n + 2 * x) : n = 16 :=
by
  sorry

end total_students_l450_450052


namespace cricket_target_runs_l450_450712

def target_runs (first_10_overs_run_rate remaining_40_overs_run_rate : ℝ) : ℝ :=
  10 * first_10_overs_run_rate + 40 * remaining_40_overs_run_rate

theorem cricket_target_runs : target_runs 4.2 6 = 282 := by
  sorry

end cricket_target_runs_l450_450712


namespace CE_length_l450_450031

theorem CE_length (AF ED AE area : ℝ) (hAF : AF = 30) (hED : ED = 50) (hAE : AE = 120) (h_area : area = 7200) : 
  ∃ CE : ℝ, CE = 138 :=
by
  -- omitted proof steps
  sorry

end CE_length_l450_450031


namespace units_digit_of_R12345_is_9_l450_450676

noncomputable def a := 3 + 2 * Real.sqrt 2
noncomputable def b := 3 - 2 * Real.sqrt 2
noncomputable def R (n : Nat) : ℝ := (1/2) * (a^n + b^n)

theorem units_digit_of_R12345_is_9 :
  ((R 12345).toInt % 10) = 9 :=
by sorry

end units_digit_of_R12345_is_9_l450_450676


namespace area_of_triangle_ABC_l450_450491

variable (A B C K : Type) -- Points A, B, C, and K
variable [GeometryPoint A B C K] -- Assumed type for dealing with geometric points
variable (AC BK BC AK : ℝ) -- Lengths
variable (triangle_ABC : Triangle A B C) (point_K_on_BC : A -> B -> C -> Prop) (AK_altitude : A -> K -> Prop)

-- Given conditions
axiom AC_value : AC = 15
axiom BK_value : BK = 9
axiom BC_value : BC = 17
axiom AK_is_altitude : AK_altitude A K
axiom K_on_BC : point_K_on_BC B C K

-- Translate into the problem that needs to be proved
theorem area_of_triangle_ABC : 
  ∃ AK^2, (AK^2 + 8^2 = 15^2) -> (triangle_area triangle_ABC = 17 * sqrt 161 / 2) :=
by
   sorry

end area_of_triangle_ABC_l450_450491


namespace cos_seven_pi_over_six_l450_450589

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
by
  sorry

end cos_seven_pi_over_six_l450_450589


namespace evelyn_wins_l450_450385

theorem evelyn_wins (k : ℕ) (h : k > 0) : 
  ∃ strategy : (nat → nat), 
  -- Define the game state and the rules
  ∀ (gameState : nat → nat), -- (mapping from box index to the number written in the box)
    -- Game ends when exactly k boxes are 0s, Evelyn wins if remaining are 1s
    (∃ count : ℕ, count = k ∧ 
      ∀ i, (gameState i = 0) → 
        ((∀ j, j ≠ i → gameState j = 1) → 
          ∃ e_strategy: (nat → nat), e_strategy i = 0 ∨ e_strategy i < gameState i)) ∨  
    -- Game ends when a player cannot move
    ¬(∃ x, gameState x ≠ 0 → 
      ((∃ o_move : nat, odd o_move ∧ o_move < gameState x) ∨ 
      (∃ e_move : nat, even e_move ∧ e_move < gameState x))) :=
begin
  sorry
end

end evelyn_wins_l450_450385


namespace gcd_108_45_eq_9_l450_450825

noncomputable def gcd (a b : ℕ) : ℕ := gcd a b 

theorem gcd_108_45_eq_9 : gcd 108 45 = 9 := by
  sorry

end gcd_108_45_eq_9_l450_450825


namespace probability_of_at_least_19_l450_450554

-- Define a standard die with six faces each numbered 1 to 6
def standard_die : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the event that the total number of dots on the five faces that are not lying on the table is at least 19
def event_at_least_19 (x : ℕ) : Prop := 21 - x ≥ 19

-- Define the probabilities
def favorable_outcomes := {1, 2}
def possible_outcomes := standard_die

-- Calculate the probability that the total number of dots on the five faces that are not lying on the table is at least 19
theorem probability_of_at_least_19 : (favorable_outcomes.card : ℚ) / possible_outcomes.card = 1 / 3 := 
by
  sorry

end probability_of_at_least_19_l450_450554


namespace isosceles_triangle_perimeter_l450_450701

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 3 ∨ a = 7) (h2 : b = 3 ∨ b = 7) (h3 : a ≠ b) :
  a + b + b = 17 :=
by
  cases h1; cases h2;
  simp;
  sorry

end isosceles_triangle_perimeter_l450_450701


namespace sufficient_condition_for_parallel_l450_450975

-- Definitions for lines and planes
variables {Line Plane : Type}

-- Definitions of parallelism and perpendicularity
variable {Parallel Perpendicular : Line → Plane → Prop}
variable {ParallelLines : Line → Line → Prop}

-- Definition of subset relation
variable {Subset : Line → Plane → Prop}

-- Theorems or conditions
variables (a b : Line) (α β : Plane)

-- Assertion of the theorem
theorem sufficient_condition_for_parallel (h1 : ParallelLines a b) (h2 : Parallel b α) (h3 : ¬ Subset a α) : Parallel a α :=
sorry

end sufficient_condition_for_parallel_l450_450975


namespace arccos_sin_three_l450_450573

noncomputable def arccos_sin_three_eq : Prop :=
  ∃ (x : ℝ), x = 3 - π / 2 ∧ real.arccos (real.sin 3) = x

theorem arccos_sin_three :
  arccos_sin_three_eq :=
sorry

end arccos_sin_three_l450_450573


namespace abs_p_sub_q_lt_one_l450_450013

-- Define the lengths of the sides of the triangle
variables (a b c : ℝ)
-- Ensure a, b, c are positive
-- (because they are lengths of the sides of a triangle)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)

-- Define p and q based on given formulas
def p := a / b + b / c + c / a
def q := a / c + c / b + b / a

-- The theorem to prove
theorem abs_p_sub_q_lt_one : abs (p a b c - q a b c) < 1 := by
  sorry

end abs_p_sub_q_lt_one_l450_450013


namespace magnitude_a_minus_2b_l450_450665

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (θ : Real.Angle)
#align 
theorem magnitude_a_minus_2b :
  θ.radians = Real.pi / 3 ∧ (a.inner b) = 1 → ‖a - 2 • b‖ = Real.sqrt 13 :=
begin
  sorry
end

end magnitude_a_minus_2b_l450_450665


namespace median_of_set_l450_450312

open List

def is_median (l : List ℝ) (m : ℝ) : Prop :=
  l.length % 2 = 1 ∧ (sorted l) ∧ (l.nth (l.length / 2)).iget = m

theorem median_of_set (s : List ℝ) (h_len : s.length = 100)
  (h1 : ∃ n, is_median (s.erase n) 78)
  (h2 : ∃ n, is_median (s.erase n) 66) :
  is_median s 72 :=
sorry

end median_of_set_l450_450312


namespace median_of_100_numbers_l450_450328

theorem median_of_100_numbers (numbers : List ℝ) (h_length : numbers.length = 100)
  (h_sorted : numbers.sorted (≤))
  (h_51 : numbers.nth_le 50 h_51_nat = 78) -- Note: nth_le is zero-indexed, so 51st element is 50th index.
  (h_50 : numbers.nth_le 49 h_50_nat = 66) : 
  (numbers.nth_le 49 h_50_nat + numbers.nth_le 50 h_51_nat) / 2 = 72 :=
sorry

end median_of_100_numbers_l450_450328


namespace smallest_possible_value_of_N_l450_450243

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l450_450243


namespace area_inequality_of_intersecting_diagonals_l450_450788

theorem area_inequality_of_intersecting_diagonals
  {A B C D P : Type*}
  [trapezoid ABCD] [bases AB CD] (h_intersect : diag_intersect_at_P ABCD P) :
  area P A B + area P C D > area P B C + area P D A :=
sorry

end area_inequality_of_intersecting_diagonals_l450_450788


namespace surface_area_of_solid_l450_450933

-- Definitions about the problem
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_rectangular_solid (a b c : ℕ) : Prop := is_prime a ∧ is_prime b ∧ is_prime c ∧ (a * b * c = 399)

-- Main statement of the problem
theorem surface_area_of_solid (a b c : ℕ) (h : is_rectangular_solid a b c) : 
  2 * (a * b + b * c + c * a) = 422 := sorry

end surface_area_of_solid_l450_450933


namespace computation_l450_450911

theorem computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  have h₁ : 27 = 3^3 := by rfl
  have h₂ : (3 : ℕ) ^ 4 = 81 := by norm_num
  have h₃ : 27^63 / 27^61 = (3^3)^63 / (3^3)^61 := by rw [h₁]
  rwa [← pow_sub, nat.sub_eq_iff_eq_add] at h₃
  have h4: 3 * 3^4 = 3^5 := by norm_num
  have h5: -486 = 3^5 - 3^6 := by norm_num
  exact h5
  sorry

end computation_l450_450911


namespace triangle_third_side_length_l450_450200

theorem triangle_third_side_length (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 8) :
  (a + b > c) → (a + c > b) → (b + c > a) → c = 6 :=
by
  intros h₃ h₄ h₅
  rw [h₁, h₂] at *
  -- Simplified inequalities from the solution step
  have h₆ : 5 + 8 > c := by rw [h₁, h₂]; exact h₃
  have h₇ : 8 + c > 5 := by rw h₂; exact h₄
  have h₈ : 5 + c > 8 := by rw h₁; exact h₅
  sorry

end triangle_third_side_length_l450_450200


namespace sum_first_11_terms_l450_450708

-- Define the arithmetic sequence and condition
def arithmetic_sequence (a : Nat → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + (a 2 - a 1)

def condition (a : Nat → ℝ) : Prop := 
  (a 4 + a 8 = 16)

-- Theorem statement
theorem sum_first_11_terms (a : Nat → ℝ) (h_seq : arithmetic_sequence a) (h_cond : condition a) :
  (∑ i in Finset.range 11, a i) = 88 :=
sorry

end sum_first_11_terms_l450_450708


namespace conformal_map_unit_circle_l450_450129

noncomputable def f (z : ℂ) : ℂ := 
  i * (2 * z + 1 - i) / (2 + z * (1 + i))

theorem conformal_map_unit_circle :
  f (\frac{i-1}{2}) = 0 ∧ arg (derivative f (\frac{i-1}{2})) = \frac{π}{2} :=
by 
  sorry

end conformal_map_unit_circle_l450_450129


namespace easter_egg_battle_probability_l450_450469

theorem easter_egg_battle_probability (players : Type) [fintype players] [decidable_eq players]
  (egg_strength : players → ℕ) (p1 : players) (p2 : players) (n : ℕ) [decidable (p1 ≠ p2)] :
  (∀ i in finset.range n, egg_strength p1 > egg_strength p2) →
  let prob11thWin := 11 / 12 in
  11 / 12 = prob11thWin :=
by sorry

end easter_egg_battle_probability_l450_450469


namespace third_side_length_l450_450182

theorem third_side_length (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) (h3 : 3 < x < 13) : (5 < x + 8) ∧ (x < 5 + 8) ∧ (5 < x + 3) ∧ (x < 8 + 5) := 
by
  sorry

end third_side_length_l450_450182


namespace all_cards_same_number_l450_450755

theorem all_cards_same_number 
  (cards : Fin 99 → ℕ) 
  (h_range : ∀ i, 1 ≤ cards i ∧ cards i ≤ 99) 
  (h_sum_not_divisible : ∀ (s : Finset (Fin 99)), (∑ i in s, cards i) % 100 ≠ 0) :
  ∃ n, ∀ i, cards i = n := 
sorry

end all_cards_same_number_l450_450755


namespace compound_interest_semiannual_l450_450286

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_semiannual :
  compound_interest 900 0.10 2 1 = 992.25 :=
by
  rw [compound_interest]
  norm_num
  sorry

end compound_interest_semiannual_l450_450286


namespace sum_y_coordinates_distance_to_origin_p1_distance_to_origin_p2_l450_450910

noncomputable theory

def circle_eq (x y : ℝ) := (x + 8)^2 + (y - 4)^2 = 144
def on_y_axis (x : ℝ) := x = 0
def point_1_y := 4 + 4 * Real.sqrt 5
def point_2_y := 4 - 4 * Real.sqrt 5

theorem sum_y_coordinates : (point_1_y + point_2_y) = 8 :=
by
  rw [point_1_y, point_2_y]
  simp [Real.sqrt_eq_rpow]

theorem distance_to_origin_p1 : Real.sqrt ((0 - 0)^2 + ((point_1_y) - 0)^2) = 4 * Real.sqrt 5 :=
by
  rw [point_1_y]
  simp [Real.sqrt_eq_rpow]

theorem distance_to_origin_p2 : Real.sqrt ((0 - 0)^2 + ((point_2_y) - 0)^2) = 4 * Real.sqrt 5 :=
by
  rw [point_2_y]
  simp [Real.sqrt_eq_rpow]

end sum_y_coordinates_distance_to_origin_p1_distance_to_origin_p2_l450_450910


namespace smallest_possible_value_of_N_l450_450242

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l450_450242


namespace compute_diff_of_squares_l450_450092

theorem compute_diff_of_squares : (65^2 - 35^2 = 3000) :=
by
  sorry

end compute_diff_of_squares_l450_450092


namespace smallest_N_value_proof_l450_450246

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l450_450246


namespace determine_k_l450_450736

theorem determine_k (k : ℕ) (f : ℝ → ℝ := fun x => sin(((2 * k + 1) / 3) * real.pi * x)) :
  (∀ a : ℝ, ∃ I : finset ℝ, (∀ x ∈ I, f x = 1/3) ∧ 2 ≤ I.card ∧ I.card ≤ 6 ∧ I.sum (λ x, f x) ≤ 6 ) →
  (k = 1 ∨ k = 2) :=
by
  sorry

end determine_k_l450_450736


namespace population_change_l450_450551

theorem population_change :
  let s1 := (1 - 0.10) in
  let s2 := (1 + 0.15) in
  let s3 := (1 + 0.20) in
  let overall_scale := s1 * s2 * s3 in
  let percentage_change := (overall_scale - 1) * 100 in
  percentage_change = 24.2 :=
sorry

end population_change_l450_450551


namespace max_pasture_area_l450_450034

/-- A rectangular sheep pasture is enclosed on three sides by a fence, while the fourth side uses the 
side of a barn that is 500 feet long. The fence costs $10 per foot, and the total budget for the 
fence is $2000. Determine the length of the side parallel to the barn that will maximize the pasture area. -/
theorem max_pasture_area (length_barn : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  length_barn = 500 ∧ cost_per_foot = 10 ∧ budget = 2000 → 
  ∃ x : ℝ, x = 100 ∧ (∀ y : ℝ, y ≥ 0 → 
    (budget / cost_per_foot) ≥ 2*y + x → 
    (y * x ≤ y * 100)) :=
by
  sorry

end max_pasture_area_l450_450034


namespace polynomial_has_real_root_l450_450883

variables {R : Type*} [Field R] [LinearOrderedField R]

theorem polynomial_has_real_root
  (P : R → R) 
  (a1 a2 a3 b1 b2 b3 : R) 
  (h_ne_zero : a1 * a2 * a3 ≠ 0)
  (h_functional : ∀ x : R, P(a1 * x + b1) + P(a2 * x + b2) = P(a3 * x + b3))
  : ∃ x : R, P(x) = 0 := 
sorry

end polynomial_has_real_root_l450_450883


namespace find_interest_rate_per_annum_l450_450770

noncomputable def principal := 12000
noncomputable def amount := 13230
noncomputable def compounding_periods := 2
noncomputable def time := 1

theorem find_interest_rate_per_annum (P: ℝ) (A: ℝ) (n: ℝ) (t: ℝ) (r: ℝ):
  P = principal →
  A = amount →
  n = compounding_periods →
  t = time →
  A = P * (1 + (r / (2 * 100)))^(2 * t) →
  r ≈ 10.24 := sorry

end find_interest_rate_per_annum_l450_450770


namespace median_of_set_l450_450313

open List

def is_median (l : List ℝ) (m : ℝ) : Prop :=
  l.length % 2 = 1 ∧ (sorted l) ∧ (l.nth (l.length / 2)).iget = m

theorem median_of_set (s : List ℝ) (h_len : s.length = 100)
  (h1 : ∃ n, is_median (s.erase n) 78)
  (h2 : ∃ n, is_median (s.erase n) 66) :
  is_median s 72 :=
sorry

end median_of_set_l450_450313


namespace probability_10_sided_die_less_than_or_equal_6_sided_die_l450_450534

theorem probability_10_sided_die_less_than_or_equal_6_sided_die : 
  let total_outcomes := 10 * 6 in
  let favorable_outcomes := (1 + 2 + 3 + 4 + 5 + 6) in
  (favorable_outcomes / total_outcomes : ℚ) = 7 / 20 :=
by
  sorry

end probability_10_sided_die_less_than_or_equal_6_sided_die_l450_450534


namespace smallest_possible_value_of_N_l450_450237

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l450_450237


namespace solve_for_x_l450_450932

theorem solve_for_x (x : ℤ) (h : 5 * (x - 9) = 6 * (3 - 3 * x) + 9) : x = 72 / 23 :=
by
  sorry

end solve_for_x_l450_450932


namespace power_n_value_l450_450497

theorem power_n_value :
  ∃ n : ℕ, 1024 * (1/4 : ℚ)^n = 4^3 ∧ n = 2 :=
by
  use 2
  split
  · calc
    1024 * (1/4 : ℚ)^2 = _ := sorry
  · rfl

end power_n_value_l450_450497


namespace valid_third_side_length_l450_450176

theorem valid_third_side_length {x : ℝ} (h1 : 5 + 8 > x) (h2 : 5 + x > 8) (h3 : 8 + x > 5) : x = 6 :=
by
  -- Given 5 + 8 > x, 5 + x > 8, 8 + x > 5
  have range1 : 13 > x := h1,
  have range2 : x > 3 := (by linarith [h2]),
  have _ : 3 < 6 ∧ 6 < 13 := by norm_num,
  linarith

#check valid_third_side_length

end valid_third_side_length_l450_450176


namespace pen_count_l450_450751

theorem pen_count (red_pens : ℕ) (blue_pens : ℕ) (black_pens : ℕ) (h1 : red_pens = 65) (h2 : blue_pens = 45) (h3 : black_pens = 58) : red_pens + blue_pens + black_pens = 168 :=
by
  rw [h1, h2, h3]
  exact Nat.add_assoc 65 45 58 ▸ Nat.add_comm 110 58 ▸ rfl

end pen_count_l450_450751


namespace Lisa_earns_15_more_than_Tommy_l450_450376

variables (total_earnings Lisa_earnings Tommy_earnings : ℝ)

-- Conditions
def condition1 := total_earnings = 60
def condition2 := Lisa_earnings = total_earnings / 2
def condition3 := Tommy_earnings = Lisa_earnings / 2

-- Theorem to prove
theorem Lisa_earns_15_more_than_Tommy (h1: condition1) (h2: condition2) (h3: condition3) : 
  Lisa_earnings - Tommy_earnings = 15 :=
sorry

end Lisa_earns_15_more_than_Tommy_l450_450376


namespace men_joined_l450_450015

noncomputable def number_of_men_joined (initial_men : ℕ) (initial_days : ℕ) (provision_days_after_joining : ℝ) : ℝ :=
  let total_provisions := (initial_men : ℝ) * (initial_days : ℝ)
  let new_men := λ x : ℝ, initial_men + x
  let provisions_with_x_joined := λ x : ℝ, (new_men x) * provision_days_after_joining
  let equation := total_provisions = provisions_with_x_joined x
  (total_provisions - (initial_men * provision_days_after_joining)) / provision_days_after_joining

theorem men_joined : number_of_men_joined 1200 15 12.857 ≈ 200 := sorry

end men_joined_l450_450015


namespace find_length_of_df_l450_450704

theorem find_length_of_df : 
  ∀ (DEF : Type) [right_triangle DEF] (F : DEF) 
    (cos_F : Real := 5 * Real.sqrt 221 / 221)
    (DE : Real := Real.sqrt 221),
    (∃ DF : Real, cos_F = DF / DE) → DF = 5 := 
sorry

end find_length_of_df_l450_450704


namespace measure_of_y_l450_450347

-- Define variables and conditions
variables {p q : Line} {a b y : ℝ}

-- State the given conditions
axiom parallel_lines : parallel p q
axiom angle_on_q : angle q a = 55
axiom supplementary : y + 55 = 180

-- State the theorem we want to prove
theorem measure_of_y : y = 125 :=
by
  -- Proof omitted
  sorry

end measure_of_y_l450_450347


namespace conjugate_z_min_modulus_z_l450_450958

open Complex Real

noncomputable def z (m : ℝ) := (m + 2) + (3 - 2m) * Complex.I

theorem conjugate_z (m : ℝ) :
  z m = conj (12 + 17 * Complex.I) ↔ m = 10 := by
  sorry

theorem min_modulus_z :
  let m_min := (2 / 5 : ℝ)
  let min_val := (7 * sqrt 5) / 5
  |z m_min| = min_val := by
  sorry

end conjugate_z_min_modulus_z_l450_450958


namespace sum_of_perimeters_of_triangle_ACD_l450_450763

theorem sum_of_perimeters_of_triangle_ACD :
  ∃ (s : ℕ),
    (∀ (AD CD BD : ℕ), AD = CD 
    ∧ ((AD^2 - (12^2) = BD^2) 
    ∧ (CD^2 - (18^2) = BD^2)) 
    → ∃ (perimeters : List ℕ), perimeters.sum = s 
    ∧ s = 168) :=
begin
  sorry
end

end sum_of_perimeters_of_triangle_ACD_l450_450763


namespace focus_of_parabola_l450_450597

theorem focus_of_parabola : 
  ∀ x y : ℝ, y = -2 * x ^ 2 → ∃ focus: ℝ × ℝ, focus = (0, -1/8) :=
by
  intro x y h
  -- added temporarily prove
  use (0, -1/8)
  sorry

end focus_of_parabola_l450_450597


namespace smallest_a_coeff_x4_l450_450959

theorem smallest_a_coeff_x4 (a : ℤ) :
  (∃ a : ℤ, (coeff (expand (1 - 2 * X + a * X^2)^8) X^4 = -1540) ∧ ∀ b : ℤ, (coeff (expand (1 - 2 * X + b * X^2)^8) X^4 = -1540 → a ≤ b)) :=
begin
  use -21,
  -- The proof will be added here.
sorry

end smallest_a_coeff_x4_l450_450959


namespace minimize_S_n_l450_450426

def general_term (n : ℕ) : ℕ := 3 * n - 23

def seq_min_at (n : ℕ) : ℕ :=
if general_term n ≤ 0 then n else 0

theorem minimize_S_n : {n : ℕ // n = 7} :=
begin
  have h1 : general_term 5 = 3 * 5 - 23 := rfl,
  have h2 : general_term 6 = 3 * 6 - 23 := rfl,
  have h3 : general_term 7 = 3 * 7 - 23 := rfl,
  have h4 : general_term 8 = 3 * 8 - 23 := rfl,
  have h_min : general_term 7 ≤ 0 := by {
    calc
      general_term 7 = -2  : by rw h3
    ...         ≤ 0     : by norm_num, },
  exact ⟨7, rfl⟩,
end

end minimize_S_n_l450_450426


namespace burattino_suspects_cheating_after_seventh_draw_l450_450069

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def binomial (n k : ℕ) : ℚ :=
(factorial n) / ((factorial k) * factorial (n - k))

noncomputable def probability_no_repeats : ℚ :=
(binomial 39 6) / (binomial 45 6)

noncomputable def estimate_draws_needed : ℕ :=
let a : ℚ := probability_no_repeats in
nat.find (λ n, a^n < 0.01)

theorem burattino_suspects_cheating_after_seventh_draw :
  estimate_draws_needed + 1 = 7 := sorry

end burattino_suspects_cheating_after_seventh_draw_l450_450069


namespace profit_percent_correct_l450_450845

noncomputable def purchase_price : ℕ := 225
noncomputable def overhead_expenses : ℕ := 15
noncomputable def selling_price : ℕ := 350

def total_cost_price (purchase_price : ℕ) (overhead_expenses : ℕ) : ℕ :=
  purchase_price + overhead_expenses

def profit (selling_price total_cost_price : ℕ) : ℕ :=
  selling_price - total_cost_price

noncomputable def profit_percent (profit total_cost_price : ℕ) : ℝ :=
  (profit.to_nat : ℝ) / (total_cost_price.to_nat : ℝ) * 100

theorem profit_percent_correct :
  profit_percent (profit selling_price (total_cost_price purchase_price overhead_expenses))
                 (total_cost_price purchase_price overhead_expenses) = 45.83 :=
by
  sorry

end profit_percent_correct_l450_450845


namespace easter_egg_battle_probability_l450_450471

theorem easter_egg_battle_probability (players : Type) [fintype players] [decidable_eq players]
  (egg_strength : players → ℕ) (p1 : players) (p2 : players) (n : ℕ) [decidable (p1 ≠ p2)] :
  (∀ i in finset.range n, egg_strength p1 > egg_strength p2) →
  let prob11thWin := 11 / 12 in
  11 / 12 = prob11thWin :=
by sorry

end easter_egg_battle_probability_l450_450471


namespace median_of_100_set_l450_450322

theorem median_of_100_set 
  (S : Finset ℝ) (h_card : S.card = 100)
  (h_remove1 : ∃ x ∈ S, median (S.erase x) = 78)
  (h_remove2 : ∃ y ∈ S, median (S.erase y) = 66) : 
  median S = 72 :=
by
  sorry

end median_of_100_set_l450_450322


namespace arithmetic_seq_intersection_quadrant_l450_450972

theorem arithmetic_seq_intersection_quadrant
  (a₁ : ℝ := 1)
  (d : ℝ := -1 / 2)
  (n : ℕ)
  (h₁ : ∃ n : ℕ, a₁ + n * d = aₙ ∧ aₙ > -1 ∧ aₙ < 1/8)
  (h₂ : n ∈ {n : ℕ | n = 3 ∨ n = 4}) :
  (aₙ = 0 ∧ n = 3) ∨ (aₙ = 1/2 ∧ n = 4) :=
by
  sorry

end arithmetic_seq_intersection_quadrant_l450_450972


namespace box_height_calculation_l450_450517

open Real

theorem box_height_calculation :
  ∃ h : ℝ, 
  (∀ (r₁ r₂ : ℝ) (dim : ℝ),
    dim = 5 ∧ r₁ = 2.5 ∧ r₂ = 1 →
    let CA := sqrt ((dim / 2 - r₂) ^ 2 + (dim / 2 - r₂) ^ 2 + (h / 2 - r₂) ^ 2) in
    CA = r₁ + r₂) →
  h = 2 + 2 * sqrt 7.75 :=
by
  sorry

end box_height_calculation_l450_450517


namespace z_2015_value_l450_450207

noncomputable def z : ℕ → ℂ
| 1 := 1
| (n + 1) := complex.conj (z n) + 1 + (n : ℂ) * complex.I

theorem z_2015_value : z 2015 = 2015 + 1007 * complex.I := 
sorry

end z_2015_value_l450_450207


namespace sequence_sum_l450_450813

open Nat

-- Define the sequence
def a : ℕ → ℕ
| 0     => 1
| (n+1) => a n + (n + 1)

-- Define the sum of reciprocals up to the 2016 term
def sum_reciprocals : ℕ → ℚ
| 0     => 1 / (a 0)
| (n+1) => sum_reciprocals n + 1 / (a (n+1))

-- Define the property we wish to prove
theorem sequence_sum :
  sum_reciprocals 2015 = 4032 / 2017 :=
sorry

end sequence_sum_l450_450813


namespace triangle_inequality_third_side_l450_450167

theorem triangle_inequality_third_side (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) :
  3 < x ∧ x < 13 ↔ (5 + 8 > x) ∧ (5 + x > 8) ∧ (8 + x > 5) :=
by 
  -- Placeholder for proof
  sorry

end triangle_inequality_third_side_l450_450167


namespace part_I_part_II_l450_450692

noncomputable def triangle_conditions (a b c : ℝ) (B C : ℝ) : Prop :=
  (a > c) ∧
  (cos B = 1/3) ∧
  (a * c = 6) ∧
  (b = 3)

theorem part_I (a b c : ℝ) (B C : ℝ) (h : triangle_conditions a b c B C) :
  a = 3 ∧ cos C = 7/9 :=
sorry

theorem part_II (a b c : ℝ) (B C : ℝ) (h : triangle_conditions a b c B C) :
  cos (2 * C + real.pi / 3) = (17 - 56 * real.sqrt 6) / 162 :=
sorry

end part_I_part_II_l450_450692


namespace sin_average_le_average_sin_l450_450284

theorem sin_average_le_average_sin (n : ℕ) (x : ℕ → ℝ) 
  (h : ∀ i, i < n → 0 ≤ x i ∧ x i ≤ real.pi) : 
  (∑ i in finset.range n, real.sin (x i)) / n ≤ real.sin ((∑ i in finset.range n, x i) / n) :=
by sorry

end sin_average_le_average_sin_l450_450284


namespace Jim_catches_Bob_in_20_minutes_l450_450904

theorem Jim_catches_Bob_in_20_minutes
  (Bob_Speed : ℕ := 6)
  (Jim_Speed : ℕ := 9)
  (Head_Start : ℕ := 1) :
  (Head_Start / (Jim_Speed - Bob_Speed) * 60 = 20) :=
by
  sorry

end Jim_catches_Bob_in_20_minutes_l450_450904


namespace greatest_possible_value_of_x_l450_450405

theorem greatest_possible_value_of_x
    (x : ℕ)
    (h1 : x > 0)
    (h2 : x % 4 = 0)
    (h3 : x^3 < 8000) :
    x ≤ 16 :=
    sorry

end greatest_possible_value_of_x_l450_450405


namespace complex_computation_l450_450917

theorem complex_computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end complex_computation_l450_450917


namespace smallest_N_l450_450254

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450254


namespace triangle_third_side_length_l450_450194

theorem triangle_third_side_length (a b : ℝ) (x : ℝ) (h₁ : a = 5) (h₂ : b = 8) (hx : x ∈ {2, 3, 6, 13}) :
  3 < x ∧ x < 13 → x = 6 :=
by sorry

end triangle_third_side_length_l450_450194


namespace conditional_probability_l450_450454

open Finset

namespace Probability

variables {Ω : Type*} [Fintype Ω] [DecidableEq Ω]

def classical_probability (s : Finset Ω) : ℝ :=
  (s.card : ℝ) / (univ.card : ℝ)

variables (S A B : Finset Ω)

def P (s : Finset Ω) : ℝ := classical_probability S s

theorem conditional_probability (hS : S = {1, 2, 3, 4, 5, 6})
  (hA : A = {2, 3, 5})
  (hB : B = {1, 2, 4, 5, 6}) :
  P S (A ∩ B) / P S B = 2 / 5 := by
  -- using the fact S, A, and B are defined as per the conditions
  sorry

end Probability

end conditional_probability_l450_450454


namespace equilateral_triangle_BJ_l450_450700

-- Define points G, F, H, J and their respective lengths on sides AB and BC
def equilateral_triangle_AG_GF_HJ_FC (AG GF HJ FC BJ : ℕ) : Prop :=
  AG = 3 ∧ GF = 11 ∧ HJ = 5 ∧ FC = 4 ∧ 
    (∀ (side_length : ℕ), side_length = AG + GF + HJ + FC → 
    (∀ (length_J : ℕ), length_J = side_length - (AG + HJ) → BJ = length_J))

-- Example usage statement
theorem equilateral_triangle_BJ : 
  ∃ BJ, equilateral_triangle_AG_GF_HJ_FC 3 11 5 4 BJ ∧ BJ = 15 :=
by
  use 15
  sorry

end equilateral_triangle_BJ_l450_450700


namespace number_of_divisors_2013_l450_450667

theorem number_of_divisors_2013^13 :
  let n := 2013
  let prime_factors := [3, 11, 61]
  let exp := 13
  (n = prime_factors.product) →
  (∏ p in prime_factors, p ^ exp = n ^ exp) →
  (∏ p in prime_factors, exp + 1)^prime_factors.length = 2744 :=
by
  intros n prime_factors exp h1 h2
  sorry

end number_of_divisors_2013_l450_450667


namespace xiaoming_age_is_two_l450_450638

-- Defining the problem conditions in Lean 4
open Nat

-- Xiaoming's current age
def current_age_x (x : ℕ) : Prop := x ≥ 1

-- Father's age and Mother's age definitions
def fathers_age (x k : ℕ) : ℕ := k * x
def mothers_age (x l : ℕ) : ℕ := l * x

-- Age difference constraints within 10 years and k != l
def valid_age_difference (x k l : ℕ) : Prop :=
  k ≠ l ∧ abs (k * x - l * x) ≤ 10

-- Last year and next year's constraints being multiples of Xiaoming's age
def last_year_multiple (x k : ℕ) : Prop := (k * x - 1) % (x - 1) = 0
def next_year_multiple (x k : ℕ) : Prop := (k * x + 1) % (x + 1) = 0
def all_multiples (x k l : ℕ) : Prop :=
  last_year_multiple x k ∧ last_year_multiple x l ∧
  next_year_multiple x k ∧ next_year_multiple x l

-- Using the proof structure
theorem xiaoming_age_is_two : ∃ x, (current_age_x x) ∧ (∃ k l : ℕ, valid_age_difference x k l ∧ all_multiples x k l) ∧ x = 2 :=
by
  exists 2
  split
  { show current_age_x 2, from by norm_num }
  exists 1, 3
  split
  { show valid_age_difference 2 1 3, 
    split
    { norm_num }
    { norm_num } }
  split
  { show last_year_multiple 2 1, by { norm_num } }
  split
  { show last_year_multiple 2 3, by { norm_num } }
  split
  { show next_year_multiple 2 1, by { norm_num } }
  { show next_year_multiple 2 3, by { norm_num } }
  sorry

end xiaoming_age_is_two_l450_450638


namespace loss_percentage_is_10_l450_450892

-- Define the given conditions
def cost_price : ℝ := 1200
def additional_gain := 168
def gain_percentage : ℝ := 0.04

-- Define the entities involved
def selling_price_with_gain := cost_price * (1 + gain_percentage)
def actual_selling_price := selling_price_with_gain - additional_gain
def loss := cost_price - actual_selling_price
def loss_percentage : ℝ := (loss / cost_price) * 100

-- Requirement to prove
theorem loss_percentage_is_10 :
  loss_percentage = 10 := 
sorry

end loss_percentage_is_10_l450_450892


namespace smallest_possible_N_l450_450265

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l450_450265


namespace combined_apples_sold_l450_450032

theorem combined_apples_sold (red_apples green_apples total_apples : ℕ) 
    (h1 : red_apples = 32) 
    (h2 : green_apples = (3 * (32 / 8))) 
    (h3 : total_apples = red_apples + green_apples) : 
    total_apples = 44 :=
by
  sorry

end combined_apples_sold_l450_450032


namespace alpha_in_fourth_quadrant_l450_450790

-- Define the given conditions
def hyperbola_eq (x y : ℝ) (α : ℝ) : Prop :=
  x^2 * sin α + y^2 * cos α = 1

def represents_hyperbola_with_foci_on_y_axis (α : ℝ) : Prop :=
  cos α > 0 ∧ sin α < 0

-- The theorem stating the mathematically equivalent proof problem
theorem alpha_in_fourth_quadrant (α : ℝ) (x y : ℝ) :
  hyperbola_eq x y α → represents_hyperbola_with_foci_on_y_axis α → (0 < α ∧ α < 2 * π) ∧ (-π < α ∧ α < 0) :=
sorry

end alpha_in_fourth_quadrant_l450_450790


namespace smallest_possible_value_of_N_l450_450240

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l450_450240


namespace median_of_100_numbers_l450_450330

theorem median_of_100_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 100)
  (h_median_99_1 : ∀ num ∈ numbers, median (numbers.erase num) = 78 → num ∈ numbers)
  (h_median_99_2 : ∀ num ∈ numbers, median (numbers.erase num) = 66 → num ∈ numbers) :
  median numbers = 72 :=
sorry

end median_of_100_numbers_l450_450330


namespace product_fraction_simplification_l450_450806

theorem product_fraction_simplification :
  (1 - (1 / 3)) * (1 - (1 / 4)) * (1 - (1 / 5)) = 2 / 5 :=
by
  sorry

end product_fraction_simplification_l450_450806


namespace triangle_inequality_third_side_l450_450168

theorem triangle_inequality_third_side (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) :
  3 < x ∧ x < 13 ↔ (5 + 8 > x) ∧ (5 + x > 8) ∧ (8 + x > 5) :=
by 
  -- Placeholder for proof
  sorry

end triangle_inequality_third_side_l450_450168


namespace valid_third_side_length_l450_450173

theorem valid_third_side_length {x : ℝ} (h1 : 5 + 8 > x) (h2 : 5 + x > 8) (h3 : 8 + x > 5) : x = 6 :=
by
  -- Given 5 + 8 > x, 5 + x > 8, 8 + x > 5
  have range1 : 13 > x := h1,
  have range2 : x > 3 := (by linarith [h2]),
  have _ : 3 < 6 ∧ 6 < 13 := by norm_num,
  linarith

#check valid_third_side_length

end valid_third_side_length_l450_450173


namespace geometric_sequence_common_ratio_l450_450715

theorem geometric_sequence_common_ratio
  (a_n : ℕ → ℝ)
  (q : ℝ)
  (h1 : a_n 3 = 7)
  (h2 : a_n 1 + a_n 2 + a_n 3 = 21) :
  q = 1 ∨ q = -1 / 2 :=
sorry

end geometric_sequence_common_ratio_l450_450715


namespace find_y_l450_450006

theorem find_y (x y : ℕ) (h1 : x > 0 ∧ y > 0) (h2 : x % y = 9) (h3 : (x:ℝ) / (y:ℝ) = 96.45) : y = 20 :=
by
  sorry

end find_y_l450_450006


namespace compute_fraction_power_eq_l450_450922

theorem compute_fraction_power_eq :
  (77_777 ^ 6 / 11_111 ^ 6) = 117649 := 
by
  sorry

end compute_fraction_power_eq_l450_450922


namespace even_and_decreasing_function_l450_450561

def is_even_function (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ x ∈ s, f x = f (-x)

def is_decreasing (f : ℝ → ℝ) (s : set ℝ) : Prop :=
  ∀ x y ∈ s, x < y → f x > f y

theorem even_and_decreasing_function :
  (is_even_function (λ x : ℝ, x ^ (-2)) set.univ) ∧ 
  (is_decreasing (λ x : ℝ, x ^ (-2)) (set.Ioi 0)) ∧
  ¬ (is_even_function (λ x : ℝ, x ^ (-1)) set.univ) ∧ 
  ¬ (is_even_function (λ x : ℝ, x ^ 2) set.univ ∧ is_decreasing (λ x : ℝ, x ^ 2) (set.Ioi 0)) ∧
  ¬ (is_even_function (λ x : ℝ, real.sqrt x) (set.Ici 0)) :=
begin
  sorry
end

end even_and_decreasing_function_l450_450561


namespace john_increased_playtime_l450_450723

def daily_play_old := 4
def weeks := 2
def days_per_week := 7
def first_period_days := weeks * days_per_week
def first_period_hours := first_period_days * daily_play_old
def completion_first_period := 0.4
def total_game_hours := first_period_hours / completion_first_period
def remaining_game_hours := total_game_hours - first_period_hours
def second_period_days := 12
def daily_play_new := remaining_game_hours / second_period_days
def playtime_increase := daily_play_new - daily_play_old

theorem john_increased_playtime :
  playtime_increase = 3 := sorry

end john_increased_playtime_l450_450723


namespace correct_answer_is_C_l450_450837

def is_quadratic_radical_expr (f : ℝ → ℝ) :=
  ∀ x : ℝ, ∃ y : ℝ, y * y = f x

def expr_A (x : ℝ) := -x - 2
def expr_B (x : ℝ) := x
def expr_C (x : ℝ) := x^2 + 2
def expr_D (x : ℝ) := x^2 - 2

theorem correct_answer_is_C : is_quadratic_radical_expr (λ x, expr_C x) ∧
  ¬is_quadratic_radical_expr (λ x, expr_A x) ∧
  ¬is_quadratic_radical_expr (λ x, expr_B x) ∧
  ¬is_quadratic_radical_expr (λ x, expr_D x) :=
by sorry

end correct_answer_is_C_l450_450837


namespace burattino_suspects_cheating_after_seventh_draw_l450_450062

theorem burattino_suspects_cheating_after_seventh_draw
  (balls : ℕ)
  (draws : ℕ)
  (a : ℝ)
  (p_limit : ℝ)
  (h_balls : balls = 45)
  (h_draws : draws = 6)
  (h_a : a = (39.choose 6 : ℝ) / (45.choose 6 : ℝ))
  (h_p_limit : p_limit = 0.01) :
  ∃ (n : ℕ), n > 5 ∧ a^n < p_limit := by
  sorry

end burattino_suspects_cheating_after_seventh_draw_l450_450062


namespace sample_volume_l450_450577

theorem sample_volume (k : ℕ) (f1 f2 f3 f4 f5 f6 : ℕ)
  (h1 : f1 = 2 * k)
  (h2 : f2 = 3 * k)
  (h3 : f3 = 4 * k)
  (h4 : f4 = 6 * k)
  (h5 : f5 = 4 * k)
  (h6 : f6 = k)
  (h_sum : f1 + f2 + f3 = 27) :
  let n := f1 + f2 + f3 + f4 + f5 + f6 in
  n = 60 :=
by
  sorry

end sample_volume_l450_450577


namespace area_of_tangent_triangle_l450_450906

theorem area_of_tangent_triangle 
  (x y : ℝ)
  (h1 : y = x * x * x) 
  (h2 : (x, y) = (3, 3 * 3 * 3)) : 
  ∃ (A : ℝ), A = 54 :=
by
  -- Definitions and conditions
  let m := 3 * 3 * 3
  let y_tangent := (27: ℝ) * x - 54
  let (x_intercept, _) := (2, 0)
  let (_, y_intercept) := (0, -54)
  have h3 : 0 = 27 * x_intercept - 54 := rfl
  have h4 : -54 = y_tangent := rfl

  -- Area calculation using the determined intercepts
  let base := (x_intercept : ℝ) - 0
  let height := 0 - y_intercept
  let area := (1/2) * base * height

  use area
  have h5 : base = 2 := rfl
  have h6 : height = 54 := rfl
  show area = 54
  rw h5
  rw h6
  norm_num
  sorry -- completeness of proof

end area_of_tangent_triangle_l450_450906


namespace loom_weaving_rate_l450_450564

theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) 
    (h1 : total_cloth = 25) (h2 : total_time = 195.3125) : 
    total_cloth / total_time = 0.128 :=
sorry

end loom_weaving_rate_l450_450564


namespace circle_C_equation_l450_450706

/-- Definitions of circles C1 and C2 -/
def circle_C1 := ∀ (x y : ℝ), (x - 4) ^ 2 + (y - 8) ^ 2 = 1
def circle_C2 := ∀ (x y : ℝ), (x - 6) ^ 2 + (y + 6) ^ 2 = 9

/-- Condition that the center of circle C is on the x-axis -/
def center_on_x_axis (x : ℝ) : Prop := ∃ y : ℝ, y = 0

/-- Bisection condition circle C bisects circumferences of circles C1 and C2 -/
def bisects_circumferences (x : ℝ) : Prop := 
  (∀ (y1 y2 : ℝ), ((x - 4) ^ 2 + (y1 - 8) ^ 2 + 1 = (x - 6) ^ 2 + (y2 + 6) ^ 2 + 9)) ∧ 
  center_on_x_axis x

/-- Statement to prove -/
theorem circle_C_equation : ∃ x y : ℝ, bisects_circumferences x ∧ (x^2 + y^2 = 81) := 
sorry

end circle_C_equation_l450_450706


namespace f_is_odd_and_increasing_l450_450212

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem f_is_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f (x)) ∧ (∀ x : ℝ, ∀ y : ℝ, x < y → f (x) < f (y)) :=
by
  -- Proving the function is odd: f(-x) = -f(x)
  have h1 : ∀ x : ℝ, f (-x) = -f (x), from
    sorry,
  -- Proving the function is increasing: x < y → f(x) < f(y)
  have h2 : ∀ x y : ℝ, x < y → f (x) < f (y), from
    sorry,
  exact ⟨h1, h2⟩

end f_is_odd_and_increasing_l450_450212


namespace smallest_possible_value_of_N_l450_450234

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l450_450234


namespace probability_correct_l450_450804

-- Define the range from 1 to 200
def range_1_to_200 : List ℕ := List.range' 1 200

-- Define perfect squares within the range
def perfect_squares := (range_1_to_200.filter (λ n, ∃ m, m * m = n)).length

-- Define perfect cubes within the range
def perfect_cubes := (range_1_to_200.filter (λ n, ∃ m, m * m * m = n)).length

-- Define sixth powers within the range
def sixth_powers := (range_1_to_200.filter (λ n, ∃ m, m * m * m * m * m * m = n)).length

-- Define numbers that are either perfect squares or perfect cubes
def squares_or_cubes := perfect_squares + perfect_cubes - sixth_powers

-- Define numbers that are neither perfect squares nor perfect cubes
def neither_squares_nor_cubes := 200 - squares_or_cubes

-- Define the probability as a fraction
def probability_neither_squares_nor_cubes : ℚ := neither_squares_nor_cubes / 200

-- The statement to prove
theorem probability_correct : probability_neither_squares_nor_cubes = 183 / 200 := by
  -- Skip the proof
  sorry

end probability_correct_l450_450804


namespace lines_parallel_with_conditions_l450_450143

variables {Plane Line : Type} [HasParallel Line] [HasPerp Line Plane]
variables (a b : Line) (alpha beta : Plane)

open HasParallel

theorem lines_parallel_with_conditions :
  (a ⊥ alpha) ∧ (b ⊥ beta) ∧ (alpha ∥ beta) → (a ∥ b) :=
by 
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  sorry

end lines_parallel_with_conditions_l450_450143


namespace smallest_possible_value_of_N_l450_450232

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l450_450232


namespace rod_length_l450_450689

theorem rod_length (L : ℝ) (weight : ℝ → ℝ) (weight_6m : weight 6 = 14.04) (weight_L : weight L = 23.4) :
  L = 10 :=
by 
  sorry

end rod_length_l450_450689


namespace number_of_palindromic_numbers_with_zero_in_hundreds_place_l450_450276

-- Definition of a five-digit palindromic number with zero in the hundred's place
def isPalindromicNumberWithZeroInHundredsPlace (n : ℕ) : Prop :=
  let digits := n.digits 10
  (digits.length = 5 ∧ digits.nth 2 = some 0 ∧ digits = digits.reverse)

-- The main statement to prove
theorem number_of_palindromic_numbers_with_zero_in_hundreds_place : 
  {n : ℕ | isPalindromicNumberWithZeroInHundredsPlace n}.card = 90 := 
by
  sorry

end number_of_palindromic_numbers_with_zero_in_hundreds_place_l450_450276


namespace func1_odd_func2_even_and_odd_func3_even_l450_450930

-- Definitions and theorems to check the parity of functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f(x) = -f(-x)

-- 1. Prove that \( f(x) = \lg (\sqrt{x^2 + 1} + x) \) is an odd function
noncomputable def func1 (x : ℝ) : ℝ := Real.log10 ((Real.sqrt (x^2 + 1)) + x)
theorem func1_odd : is_odd func1 := 
by sorry

-- 2. Prove that \( f(x) = \ln x^2 + \ln \frac{1}{x^2} \) is both an even and odd function
noncomputable def func2 (x : ℝ) : ℝ := Real.log x^2 + Real.log (1 / x^2)
theorem func2_even_and_odd : is_even func2 ∧ is_odd func2 := 
by sorry

-- 3. Prove that the piecewise function is an even function
noncomputable def func3 (x : ℝ) : ℝ :=
if h : x ≥ 0 then x * (1 - x) else -x * (1 + x)
theorem func3_even : is_even func3 := 
by sorry

end func1_odd_func2_even_and_odd_func3_even_l450_450930


namespace area_of_pentagon_l450_450412

-- Define the areas of the triangles
variables (α β γ θ δ S : ℝ)

-- The condition is that S should satisfy the quadratic equation
theorem area_of_pentagon :
  S^2 - (α + β + γ + θ + δ)*S + (α*β + β*γ + γ*θ + θ*δ + δ*α) = 0 :=
begin
  sorry
end

end area_of_pentagon_l450_450412


namespace platform_length_l450_450555

theorem platform_length
  (train_length : ℕ) (train_speed_kmph : ℕ) (time_seconds : ℕ) (train_speed_mps := (train_speed_kmph : ℚ) * 1000 / 3600) :
  train_length = 360 → train_speed_kmph = 45 → time_seconds = 44 →  
  let total_distance := train_speed_mps * time_seconds in 
  let platform_length := total_distance - train_length in 
  platform_length = 190 :=
by intros train_length_eq train_speed_kmph_eq time_seconds_eq
   simp [train_length_eq, train_speed_kmph_eq, time_seconds_eq]
   sorry

end platform_length_l450_450555


namespace condition_on_p_q_l450_450794

-- (1) The function f(x)
def f (x : ℝ) (q p : ℝ) : ℝ :=
  if x < 0 then - 3^(q * x) / (3^(q * x) + p - 1)
  else if x = 0 then 0
  else 1 / ((p - 1) * 3^(q * x) + 1)

-- (2) Sequence {a_n} where a_n = 3n - 2
def a_n (n : ℕ) : ℝ := 3 * n - 2

-- (3) Condition for p + q s.t. lim_{n -> ∞} f(a_n) = 0
theorem condition_on_p_q (p q : ℝ) (h1 : p >= 1) : p + q > 1 :=
begin
  sorry
end

end condition_on_p_q_l450_450794


namespace x_expression_l450_450734

noncomputable def f (t : ℝ) : ℝ := t / (1 - t)

theorem x_expression {x y : ℝ} (hx : x ≠ 1) (hy : y = f x) : x = y / (1 + y) :=
by {
  sorry
}

end x_expression_l450_450734


namespace triangle_properties_l450_450983

theorem triangle_properties (a b c : ℝ)
  (h1 : (b - 5) ^ 2 + (c - 7) ^ 2 = 0)
  (h2 : |a - 3| = 2) :
  a = 5 ∧ b = 5 ∧ c = 7 ∧ (a + b + c = 17 ∧ (a = b ∨ a = c ∨ b = c)) :=
by
  -- from (b-5)^2 + (c-7)^2 = 0, we can conclude b=5 and c=7
  have h_b : b = 5, from eq_of_sq_eq_zero (by linarith [h1]),
  have h_c : c = 7, from eq_of_sq_eq_zero (by linarith [h1]),
  rw [h_b, h_c] at h1,
  -- from |a-3|=2, we find possible values for a: a=5 or a=1
  have h_a : a = 5 ∨ a = 1, from abs_eq (by linarith),
  -- if a=1, it does not satisfy the triangle inequality a + b > c, a + c > b, b + c > a
  cases h_a,
  -- case a = 1
  -- not possible because 1+5=6 which is not greater than c=7
  linarith,
  -- case a = 5
  use [h_a, h_b, h_c],
  split,
  -- perimeter calculation
  linarith,
  -- check isosceles triangle condition
  linarith
  sorry

end triangle_properties_l450_450983


namespace player1_wins_11th_round_l450_450486

noncomputable def egg_strength_probability (n : ℕ) : ℚ :=
  (n - 1) / n

theorem player1_wins_11th_round :
  let player1_wins_first_10_rounds := true,
      total_rounds := 11,
      new_egg := 12 in
  player1_wins_first_10_rounds → egg_strength_probability total_rounds = 11 / 12 :=
by
  intros
  exact sorry

end player1_wins_11th_round_l450_450486


namespace temperature_representation_l450_450089

theorem temperature_representation (a : ℤ) (b : ℤ) (h1 : a = 8) (h2 : b = -5) :
    b < 0 → b = -5 :=
by
  sorry

end temperature_representation_l450_450089


namespace arithmetic_geometric_sequence_l450_450048

variable (a₁ : ℝ)
def q : ℝ := 3
def S_n (n : ℕ) : ℝ := a₁ * (1 - q^n) / (1 - q)
def a_n (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem arithmetic_geometric_sequence :
  S_n 4 / a_n 4 = 40 / 27 := by
  sorry

end arithmetic_geometric_sequence_l450_450048


namespace a_can_finish_work_in_5_days_l450_450020

variable (x : ℝ) -- x represents the number of days A takes to finish the work alone.

-- Work rates
def a_work_rate : ℝ := 1 / x
def b_work_rate : ℝ := 1 / 15

-- Work done in 2 days when A and B work together
def work_done_together : ℝ := 2 * (a_work_rate + b_work_rate)

-- Work done by B alone in 7 days
def work_done_by_b : ℝ := 7 * b_work_rate

-- Total work done should be equal to 1 (the whole work)
theorem a_can_finish_work_in_5_days : 
  work_done_together + work_done_by_b = 1 → x = 5 := by
  sorry

end a_can_finish_work_in_5_days_l450_450020


namespace complement_A_inter_B_l450_450160

def A : set ℝ := {x | (x - 1) / (x - 3) ≤ 0}
def B : set ℕ := {x | 0 ≤ x ∧ x ≤ 4}

theorem complement_A_inter_B :
  (compl A) ∩ (B : set ℝ) = ({0, 3, 4} : set ℝ) :=
by
  sorry

end complement_A_inter_B_l450_450160


namespace limit_series_pi_over_4_l450_450952

theorem limit_series_pi_over_4 :
  (∀ N, ∃ n, N = n^2) →
  tendsto (λ n, ∑ i in finset.range n^2, (n : ℝ) / (n^2 + i ^ 2)) at_top (𝓝 (π / 4)) :=
by sorry

end limit_series_pi_over_4_l450_450952


namespace player1_wins_11th_round_l450_450468

theorem player1_wins_11th_round (player1_wins_first_10 : ∀ (round : ℕ), round < 10 → player1_wins round) : 
  prob_winning_11th_round player1 = 11 / 12 :=
sorry

end player1_wins_11th_round_l450_450468


namespace whisky_replacement_l450_450543

variable (V x : ℝ)

/-- The initial whisky in the jar contains 40% alcohol -/
def initial_volume_of_alcohol (V : ℝ) : ℝ := 0.4 * V

/-- A part (x liters) of this whisky is replaced by another containing 19% alcohol -/
def volume_replaced_whisky (x : ℝ) : ℝ := x
def remaining_whisky (V x : ℝ) : ℝ := V - x

/-- The percentage of alcohol in the jar after replacement is 24% -/
def final_volume_of_alcohol (V x : ℝ) : ℝ := 0.4 * (remaining_whisky V x) + 0.19 * (volume_replaced_whisky x)

/- Prove that the quantity of whisky replaced is 0.16/0.21 times the total volume -/
theorem whisky_replacement :
  final_volume_of_alcohol V x = 0.24 * V → x = (0.16 / 0.21) * V :=
by sorry

end whisky_replacement_l450_450543


namespace circular_divisible_l450_450139

theorem circular_divisible (n : ℕ) (h : n ≥ 3) : 
  -- If it is possible to find such an arrangement, then it must be n = 3.
  (∀ σ : Fin n → ℕ, 
    (∀ i : Fin n, 1 ≤ σ i ∧ σ i ≤ n) ∧ 
    (Function.injective σ) ∧ 
    (∀ k : Fin n, σ k ∣ (σ (Fin.mk ((k + 1) % n) sorry) + σ (Fin.mk ((k + 2) % n) sorry)))) ↔ n = 3 := 
  sorry

end circular_divisible_l450_450139


namespace find_valid_pairs_l450_450948

-- Definitions and conditions:
def satisfies_equation (a b : ℤ) : Prop := a^2 + a * b - b = 2018

-- Correct answers:
def valid_pairs : List (ℤ × ℤ) :=
  [(2, 2014), (0, -2018), (2018, -2018), (-2016, 2014)]

-- Statement to prove:
theorem find_valid_pairs :
  ∀ (a b : ℤ), satisfies_equation a b ↔ (a, b) ∈ valid_pairs.toFinset := by
  sorry

end find_valid_pairs_l450_450948


namespace exists_sphere_centers_l450_450926

-- Definitions related to the problem
variables {A B C D A0 B0 C0 O1 O2 : Type*}

def tetrahedron (A B C D : Type*) := true
def plane (ABC : Type*) := true
def quarter_points (A0 B0 C0 : Type*) := 
  -- Specific positions of the quarter points
  DA_0 = (3 / 4) * DA ∧ DB_0 = (1 / 2) * DB ∧ DC_0 = (1 / 4) * DC

-- Condition enforcing that the tetrahedron is regular
axiom regular_tetrahedron : tetrahedron A B C D

-- Conditions enforcing the particular positions of the quarter points
axiom specific_quarter_points : quarter_points A0 B0 C0

-- Proof problem statement
theorem exists_sphere_centers :
  ∃ O1 O2, (tetrahedron A B C D) ∧ (quarter_points A0 B0 C0) ∧ 
  -- centers O1 and O2 are on the line perpendicular to the determining plane
  -- the spheres centered at O1 and O2 pass through A0, B0, and C0, and touch plane ABC
  (plane ABC) :=
sorry

end exists_sphere_centers_l450_450926


namespace solve_for_x_l450_450113

theorem solve_for_x :
  (∀ y : ℝ, 10 * x * y - 15 * y + 4 * x - 6 = 0) ↔ x = 3 / 2 :=
by
  sorry

end solve_for_x_l450_450113


namespace doughnut_machine_completion_time_l450_450522

/-- The doughnut machine completes the job at 7:10 PM given it starts at 8:30 AM and has completed one-fourth of the job by 11:10 AM. -/
theorem doughnut_machine_completion_time :
  ∀ (start_time one_fourth_completion_time : ℚ),
  start_time = 510 / 1440 ∧ one_fourth_completion_time = 670 / 1440 ∧ (one_fourth_completion_time - start_time) = 8 / 3 / 24 
  → start_time + (4 * (one_fourth_completion_time - start_time)) = 1180 / 1440 := 
by
  intros start_time one_fourth_completion_time h,
  sorry

end doughnut_machine_completion_time_l450_450522


namespace determine_n_l450_450651

def f (x : ℝ) : ℝ := -0.5 * x ^ 2 + x

theorem determine_n (m n k : ℝ) (h1 : m < n) (h2 : 1 < k) (h3 : ∀ x ∈ set.Icc m n, f x ∈ set.Icc (k * m) (k * n)) : 
  n = 0 :=
by
  sorry

end determine_n_l450_450651


namespace symmetric_angle_l450_450503
def symmetric_about_x_axis (theta: ℝ) := ∃ k : ℤ, theta = -60 + 360 * k

theorem symmetric_angle : symmetric_about_x_axis 660 := by
  use 2
  norm_num
  sorry

end symmetric_angle_l450_450503


namespace smallest_N_l450_450255

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450255


namespace triangle_side_length_uniqueness_l450_450699

-- Define the conditions as axioms
variable (n : ℕ)
variable (h : n > 0)
variable (A1 : 3 * n + 9 > 5 * n - 4)
variable (A2 : 5 * n - 4 > 4 * n + 6)

-- The theorem stating the constraints and expected result
theorem triangle_side_length_uniqueness :
  (4 * n + 6) + (3 * n + 9) > (5 * n - 4) ∧
  (3 * n + 9) + (5 * n - 4) > (4 * n + 6) ∧
  (5 * n - 4) + (4 * n + 6) > (3 * n + 9) ∧
  3 * n + 9 > 5 * n - 4 ∧
  5 * n - 4 > 4 * n + 6 → 
  n = 11 :=
by {
  -- Proof steps can be filled here
  sorry
}

end triangle_side_length_uniqueness_l450_450699


namespace max_red_socks_l450_450533

theorem max_red_socks (r b : ℕ) (h1 : r + b ≤ 2000) 
  (h2 : ((r * (r - 1) + b * (b - 1)) : ℚ) / ((r + b) * (r + b - 1)) = 5 / 12) :
  r ≤ 109 :=
begin
  sorry
end

end max_red_socks_l450_450533


namespace irrational_numbers_count_l450_450894

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_numbers_count : 
  let nums := [Real.sqrt 16, 0, Real.sqrt 5, 3.14159, (Real.pi - 1)^2, 2 / 5, 0.20220222022220...] in
  (nums.filter is_irrational).length = 3 :=
by
  sorry

end irrational_numbers_count_l450_450894


namespace player1_wins_11th_round_l450_450484

noncomputable def egg_strength_probability (n : ℕ) : ℚ :=
  (n - 1) / n

theorem player1_wins_11th_round :
  let player1_wins_first_10_rounds := true,
      total_rounds := 11,
      new_egg := 12 in
  player1_wins_first_10_rounds → egg_strength_probability total_rounds = 11 / 12 :=
by
  intros
  exact sorry

end player1_wins_11th_round_l450_450484


namespace emily_speed_l450_450943

theorem emily_speed (distance time : ℝ) (h1 : distance = 10) (h2 : time = 2) : (distance / time) = 5 := 
by sorry

end emily_speed_l450_450943


namespace sqrt_sum_fractions_eq_l450_450085

theorem sqrt_sum_fractions_eq :
  (Real.sqrt ((1 / 25) + (1 / 36)) = (Real.sqrt 61) / 30) :=
by
  sorry

end sqrt_sum_fractions_eq_l450_450085


namespace emily_wrong_questions_l450_450120

variable (E F G H : ℕ)

theorem emily_wrong_questions (h1 : E + F + 4 = G + H) 
                             (h2 : E + H = F + G + 8) 
                             (h3 : G = 6) : 
                             E = 8 :=
sorry

end emily_wrong_questions_l450_450120


namespace solution_set_f_gt_zero_l450_450370

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom f_neg2_zero : f (-2) = 0
axiom f_prime_pos : ∀ x > 0, 3 * f x + x * f' x > 0

theorem solution_set_f_gt_zero : {x : ℝ | f x > 0} = {x : ℝ | (-2 < x ∧ x < 0) ∨ (2 < x ∧ x < ∞)} :=
sorry

end solution_set_f_gt_zero_l450_450370


namespace circumcircle_radius_of_right_triangle_l450_450024

theorem circumcircle_radius_of_right_triangle (a b c : ℝ) (h1: a = 8) (h2: b = 6) (h3: c = 10) (h4: a^2 + b^2 = c^2) : (c / 2) = 5 := 
by
  sorry

end circumcircle_radius_of_right_triangle_l450_450024


namespace possible_third_side_l450_450190

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end possible_third_side_l450_450190


namespace smallest_possible_N_l450_450267

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l450_450267


namespace least_number_of_elements_of_set_A_l450_450951

-- Definitions and conditions
def has_property (f : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, 0 < i → 0 < j → nat.prime (i - j) → f i ≠ f j

-- Main statement
theorem least_number_of_elements_of_set_A :
  ∃ (A : finset ℕ), (∃ (f : ℕ → A), has_property f) ∧ A.card = 4 :=
begin
  sorry
end

end least_number_of_elements_of_set_A_l450_450951


namespace median_of_100_numbers_l450_450323

theorem median_of_100_numbers (numbers : List ℝ) (h_length : numbers.length = 100)
  (h_sorted : numbers.sorted (≤))
  (h_51 : numbers.nth_le 50 h_51_nat = 78) -- Note: nth_le is zero-indexed, so 51st element is 50th index.
  (h_50 : numbers.nth_le 49 h_50_nat = 66) : 
  (numbers.nth_le 49 h_50_nat + numbers.nth_le 50 h_51_nat) / 2 = 72 :=
sorry

end median_of_100_numbers_l450_450323


namespace smallest_N_value_proof_l450_450244

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l450_450244


namespace radii_of_circles_l450_450720

theorem radii_of_circles
  (r s : ℝ)
  (h_ratio : r / s = 9 / 4)
  (h_right_triangle : ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2)
  (h_tangent : (r + s)^2 = (r - s)^2 + 12^2) :
   r = 20 / 47 ∧ s = 45 / 47 :=
by
  sorry

end radii_of_circles_l450_450720


namespace smallest_possible_N_l450_450263

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l450_450263


namespace expression_evaluation_l450_450082

theorem expression_evaluation : 2 + 3 * 4 - 5 + 6 * (2 - 1) = 15 := 
by sorry

end expression_evaluation_l450_450082


namespace player1_wins_11th_round_probability_l450_450482

-- Definitions based on the conditions
def egg_shell_strength (n : ℕ) : ℝ := sorry
def player1_won_first_10_rounds : Prop := sorry

-- Main theorem
theorem player1_wins_11th_round_probability
  (h : player1_won_first_10_rounds) :
  Prob (egg_shell_strength 11 > egg_shell_strength 12) = 11 / 12 := sorry

end player1_wins_11th_round_probability_l450_450482


namespace rectangle_area_given_diagonal_l450_450440

noncomputable def area_of_rectangle (x : ℝ) : ℝ :=
  1250 - x^2 / 2

theorem rectangle_area_given_diagonal (P : ℝ) (x : ℝ) (A : ℝ) :
  P = 100 → x^2 = (P / 2)^2 - 2 * A → A = area_of_rectangle x :=
by
  intros hP hx
  sorry

end rectangle_area_given_diagonal_l450_450440


namespace sequence_inequality_l450_450445

theorem sequence_inequality (a : ℕ → ℕ) (strictly_increasing : ∀ n, a n < a (n + 1))
  (sum_condition : ∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) :
  ∀ n, a n ≤ n^2 :=
by sorry

end sequence_inequality_l450_450445


namespace angle_of_inclination_of_line_is_135_degrees_l450_450493

theorem angle_of_inclination_of_line_is_135_degrees :
  ∃ θ : ℝ, (x + y - 1 = 0) ∧ (θ = 135.0 * real.pi / 180.0) :=
sorry

end angle_of_inclination_of_line_is_135_degrees_l450_450493


namespace simplify_expression_l450_450507

theorem simplify_expression (y : ℝ) (h : y ≠ -((√2)/3)^5) : 
    ( ((2^(3/2) + 27 * y^(3/5)) / (√2 + 3 * y^(1/5)) + 3 * (32 * y^2)^(1/10) - 2) * 3^(-2) )^5 = y^2 :=
by
  sorry

end simplify_expression_l450_450507


namespace count_numbers_in_set_S_l450_450673
open Nat

-- Define the sequence set
def S : Set ℕ := {n | ∃ k : ℕ, n = 7 + 10 * k}

-- Define a function to check if n can be expressed as the difference of two primes
def can_be_written_as_difference_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p > q ∧ n = p - q

-- Define a list of the numbers in the set S that can be written as such
def number_of_elements_in_S_that_can_be_written_as_difference_of_two_primes : ℕ :=
  (Finset.filter can_be_written_as_difference_of_two_primes (Finset.range 1000)).card

-- The main theorem
theorem count_numbers_in_set_S : number_of_elements_in_S_that_can_be_written_as_difference_of_two_primes = 2 := 
sorry

end count_numbers_in_set_S_l450_450673


namespace poly_degree_exists_l450_450030

noncomputable def poly_degree :=
  ∃ (p : Polynomial ℚ),
    (∀ n, 2 ≤ n ∧ n ≤ 1500 → Polynomial.is_root p (n + Real.sqrt (n + 1)))
      ∧ (∀ n, 2 ≤ n ∧ n ≤ 1500 → Polynomial.is_root p (n - Real.sqrt (n + 1)))
      ∧ (∀ n, 1 ≤ n ∧ n ≤ 1500 → Polynomial.is_root p (n + Real.cos (n * Real.pi)))
      ∧ Polynomial.degree p = 4424

theorem poly_degree_exists : poly_degree := sorry

end poly_degree_exists_l450_450030


namespace almost_square_as_quotient_l450_450827

-- Defining what almost squares are
def isAlmostSquare (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

-- Statement of the theorem
theorem almost_square_as_quotient (n : ℕ) (hn : n > 0) :
  ∃ a b : ℕ, isAlmostSquare a ∧ isAlmostSquare b ∧ n * (n + 1) = a / b := by
  sorry

end almost_square_as_quotient_l450_450827


namespace integral_root_of_equation_l450_450924

theorem integral_root_of_equation : 
  ∀ x : ℤ, (x - 8 / (x - 4)) = 2 - 8 / (x - 4) ↔ x = 2 := 
sorry

end integral_root_of_equation_l450_450924


namespace largest_digit_to_correct_sum_l450_450785

theorem largest_digit_to_correct_sum :
  (725 + 864 + 991 = 2570) → (∃ (d : ℕ), d = 9 ∧ 
  (∃ (n1 : ℕ), n1 ∈ [702, 710, 711, 721, 715] ∧ 
  ∃ (n2 : ℕ), n2 ∈ [806, 805, 814, 854, 864] ∧ 
  ∃ (n3 : ℕ), n3 ∈ [918, 921, 931, 941, 981, 991] ∧ 
  n1 + n2 + n3 = n1 + n2 + n3 - 10))
    → d = 9 :=
by
  sorry

end largest_digit_to_correct_sum_l450_450785


namespace smallest_possible_value_of_N_l450_450241

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l450_450241


namespace triangle_inequality_third_side_l450_450172

theorem triangle_inequality_third_side (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) :
  3 < x ∧ x < 13 ↔ (5 + 8 > x) ∧ (5 + x > 8) ∧ (8 + x > 5) :=
by 
  -- Placeholder for proof
  sorry

end triangle_inequality_third_side_l450_450172


namespace bisect_CD_by_BH_l450_450346

open EuclideanGeometry

-- Definitions representing the geometric entities and properties
variables {A B C D H : Point}
variables {AB CD AC AD DH : Segment}
variables {angle_DAB : Angle}

-- Conditions
axiom quadrilateral_convex : is_convex A B C D
axiom angle_B_right : angle_eq (∠ B) (right_angle)
axiom AC_bisector_DAB : is_angle_bisector AC angle_DAB
axiom AC_equals_AD : eq_length AC AD
axiom DH_altitude_ADC : is_altitude DH (triangle A D C)

-- Prove that BH bisects CD
theorem bisect_CD_by_BH 
  (is_midpoint_BH : midpoint B H)
  (is_midpoint_CD : midpoint H C D) : bisects B H C D := 
sorry

end bisect_CD_by_BH_l450_450346


namespace digit_712th_is_1_l450_450501

-- Conditions
def seven_over_twenty_nine : ℚ := 7 / 29

def decimal_rep_of_seven_over_twenty_nine : String :=
  "0.241379310344827586206896551724137931".cycle(29)

def mod_712_29 : ℕ := 712 % 29

-- Theorem statement
theorem digit_712th_is_1 : (decimal_rep_of_seven_over_twenty_nine.toList.nth (mod_712_29 - 1)) = some '1' :=
by
  sorry

end digit_712th_is_1_l450_450501


namespace total_ducks_and_ducklings_l450_450381

theorem total_ducks_and_ducklings :
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6
  (ducks1 + ducks2 + ducks3) + (ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3) = 99 :=
by
  sorry

end total_ducks_and_ducklings_l450_450381


namespace percent_decrease_area_square_l450_450694

/-- 
In a configuration, two figures, an equilateral triangle and a square, are initially given. 
The equilateral triangle has an area of 27√3 square inches, and the square has an area of 27 square inches.
If the side length of the square is decreased by 10%, prove that the percent decrease in the area of the square is 19%.
-/
theorem percent_decrease_area_square 
  (triangle_area : ℝ := 27 * Real.sqrt 3)
  (square_area : ℝ := 27)
  (percentage_decrease : ℝ := 0.10) : 
  let new_square_side := Real.sqrt square_area * (1 - percentage_decrease)
  let new_square_area := new_square_side ^ 2
  let area_decrease := square_area - new_square_area
  let percent_decrease := (area_decrease / square_area) * 100
  percent_decrease = 19 := 
by
  sorry

end percent_decrease_area_square_l450_450694


namespace burattino_suspects_cheating_after_seventh_draw_l450_450077

theorem burattino_suspects_cheating_after_seventh_draw 
  (total_balls : ℕ := 45) (drawn_balls : ℕ := 6) (a : ℝ := ((nat.choose 39 6 : ℕ) : ℝ) / ((nat.choose 45 6 : ℕ) : ℝ))
  (threshold : ℝ := 0.01) (probability : ℝ := 0.4) :
  (∃ n, a^n < threshold) → (∃ n > 5, a^n < threshold) :=
begin
  -- Definitions from conditions
  have fact_prob : a = ((nat.choose 39 6 : ℕ) : ℝ) / ((nat.choose 45 6 : ℕ) : ℝ), by refl,
  have fact_approx : a ≈ probability, by simp,

  -- Statement to prove
  intros h,
  use 6,
  split,
  { linarith, },
  { sorry }
end

end burattino_suspects_cheating_after_seventh_draw_l450_450077


namespace production_units_l450_450678

-- Define the production function U
def U (women hours days : ℕ) : ℕ := women * hours * days

-- State the theorem
theorem production_units (x z : ℕ) (hx : ¬ x = 0) :
  U z z z = (z^3 / x) :=
  sorry

end production_units_l450_450678


namespace log_probability_is_2_7_l450_450461

-- Define the set of numbers
def numbers : Finset ℕ := (Finset.range 16).filter (fun n => n > 0) |>.image (fun n => 3^n)

-- Define the condition for log_a b to be an integer
def log_condition (a b : ℕ) : Prop := ∃ z : ℤ, a^z = b

-- Calculate the probability
def probability_log_integer : ℚ :=
  let pairs := numbers.ssubsets_of_card 2
  let valid_pairs := pairs.filter (λ s => match s.toList with
    | [a, b] => log_condition a b ∨ log_condition b a
    | _ => false)
  (valid_pairs.card : ℚ) / pairs.card

-- The target theorem
theorem log_probability_is_2_7 : probability_log_integer = 2 / 7 := by
  sorry

end log_probability_is_2_7_l450_450461


namespace harry_hours_last_week_l450_450846

noncomputable def harry_hours_worked (x : ℝ) : ℕ :=
let james_pay := 41.5 * x in
let harry_pay := 30 * x + 2 * x * (36 - 30) in
if harry_pay = james_pay then 36 else 0

theorem harry_hours_last_week (x : ℝ) (james_worked_hours : ℕ) (james_pay_per_first_40_hours : ℝ)
    (james_pay_per_additional_hour : ℝ) (harry_pay_per_first_30_hours : ℝ) 
    (harry_pay_per_additional_hour : ℝ) 
    (harry_and_james_same_pay : harry_pay_per_first_30_hours * 30 + harry_pay_per_additional_hour * (36 - 30) = 
    james_pay_per_first_40_hours * 40 + james_pay_per_additional_hour * (james_worked_hours - 40)) :
    harry_hours_worked x = 36 :=
begin
    sorry
end

end harry_hours_last_week_l450_450846


namespace find_median_of_100_l450_450300

noncomputable def median_of_set (s : Finset ℝ) : ℝ :=
if h : ∃ median, is_median s median then classical.some h else 0

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
∃ (f : ℤ), (f : ℝ) = (card s : ℝ) / 2 ∧
    ∃ (low : Finset ℝ) (high : Finset ℝ),
        low ⊆ s ∧ high ⊆ s ∧
        card low = floor ((card s : ℝ) / 2) ∧
        card high = ceil ((card s : ℝ) / 2) ∧
        (∀ x ∈ low, x ≤ m) ∧ (∀ x ∈ high, x ≥ m)

theorem find_median_of_100 (s : Finset ℝ) (h_size : s.card = 100)
(h1 : ∃ x ∈ s, median_of_set (s.erase x) = 78)
(h2 : ∃ y ∈ s, median_of_set (s.erase y) = 66) :
  median_of_set s = 72 := by
sorry

end find_median_of_100_l450_450300


namespace cube_edge_length_bounds_l450_450492

variable (a h x : ℝ)

def cube_edge_length_of_pyramid (a h : ℝ) : Prop :=
  ∀ x, 
    (1) (0 < a ∧ 0 < h) → 
    (2) (∃ x, (a * h) / (a + h * Real.sqrt 2) ≤ x ∧ x ≤ (a * h) / (a + h)).

theorem cube_edge_length_bounds :
  cube_edge_length_of_pyramid a h :=
begin
  sorry
end

end cube_edge_length_bounds_l450_450492


namespace solve_for_a_l450_450997

theorem solve_for_a (a x : ℝ) (h1 : 3 * x - 5 = x + a) (h2 : x = 2) : a = -1 :=
by
  sorry

end solve_for_a_l450_450997


namespace average_percentage_reduction_l450_450528

/-- A proof problem to show that the average percentage reduction
    in price for each reduction results in a specific value given conditions. -/
theorem average_percentage_reduction (initial_price final_price : ℝ)
  (h_initial : initial_price = 2000)
  (h_final : final_price = 1620) : 
  let x := 1 - real.sqrt (final_price / initial_price) in x = 0.1 :=
by
  sorry

end average_percentage_reduction_l450_450528


namespace median_of_set_l450_450315

open List

def is_median (l : List ℝ) (m : ℝ) : Prop :=
  l.length % 2 = 1 ∧ (sorted l) ∧ (l.nth (l.length / 2)).iget = m

theorem median_of_set (s : List ℝ) (h_len : s.length = 100)
  (h1 : ∃ n, is_median (s.erase n) 78)
  (h2 : ∃ n, is_median (s.erase n) 66) :
  is_median s 72 :=
sorry

end median_of_set_l450_450315


namespace points_on_circle_harmonic_quad_A1_B1_C1_D1_l450_450969

variables {A B C D A1 B1 C1 D1 : Point}

-- Given conditions
axiom harmonic_quad_A_B_C_D {A B C D : Point} :
  ∏ (A B) (C D : Point), AB * CD = AC * BD ∧ AC * BD = AD * BC

axiom harmonic_quad_A1_B_C_D {A1 B C D : Point} (h : A1 ≠ A) :
  harmonic_quad_A_B_C_D A1 B C D

axiom harmonic_quad_B1_A_C_D {B1 A C D : Point} (h : B1 ≠ B) :
  harmonic_quad_A_B_C_D B1 A C D

axiom harmonic_quad_C1_A_B_D {C1 A B D : Point} (h : C1 ≠ C) :
  harmonic_quad_A_B_C_D C1 A B D

axiom harmonic_quad_D1_A_B_C {D1 A B C : Point} (h : D1 ≠ D) :
  harmonic_quad_A_B_C_D D1 A B C

-- Problem (a): Prove A, B, C1, and D1 lie on a single circle
theorem points_on_circle {A B C1 D1 : Point} :
  harmonic_quad_A_B_C_D A B C1 D1 →
  lies_on_circle A B C1 D1 :=
  sorry

-- Problem (b): Prove A1, B1, C1, and D1 form a harmonic quadrilateral
theorem harmonic_quad_A1_B1_C1_D1 {A1 B1 C1 D1 : Point} :
  (A1 ≠ A) →
  (B1 ≠ B) →
  (C1 ≠ C) →
  (D1 ≠ D) →
  harmonic_quad_A_B_C_D A1 B1 C1 D1 :=
  sorry

end points_on_circle_harmonic_quad_A1_B1_C1_D1_l450_450969


namespace smallest_x_abs_eq_9_smallest_x_abs_eq_9_l450_450134

theorem smallest_x_abs_eq_9 (x : ℝ) (h : |x - 8| = 9) : x ≥ -1 := sorry

theorem smallest_x_abs_eq_9 (x : ℝ) (h : |x - 8| = 9) : ∃ y, y = -1 ∧ y ≤ x := sorry

end smallest_x_abs_eq_9_smallest_x_abs_eq_9_l450_450134


namespace probability_of_three_heads_in_eight_tosses_l450_450535

noncomputable def coin_toss_probability : ℚ :=
  let total_outcomes := 2^8
  let favorable_outcomes := Nat.choose 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_three_heads_in_eight_tosses : coin_toss_probability = 7 / 32 :=
  by
  sorry

end probability_of_three_heads_in_eight_tosses_l450_450535


namespace monotonicity_of_f_range_of_x1_x2_l450_450652

-- Define the function f(x)
def f (k : ℝ) (x : ℝ) : ℝ := (k + 4 / k) * Real.log x + (4 - x^2) / x

-- Condition: k > 0
axiom k_pos (k : ℝ) : k > 0

-- Monotonicity on interval (0, 2) based on the given conditions
theorem monotonicity_of_f (k : ℝ) (x : ℝ) (h₁ : 0 < x) (h₂ : x < 2) :
  (if (0 < k ∧ k < 2) then
    f kderiv x < 0 ∧ f kderiv x > 0
    else if (k = 2) then
    f kderiv x < 0
    else if (k > 2) then
    f kderiv x < 0 ∧ f kderiv x > 0) := sorry


-- Range of x₁ + x₂
theorem range_of_x1_x2 (k : ℝ) (x1 x2 : ℝ) (h₁ : 4 ≤ k) (h₂ : x1 ≠ x2) (h₃ : x1 > 0) (h₄ : x2 > 0)
  (h₅ : deriv (f k) x1 = deriv (f k) x2) :
  x1 + x2 > 16 / (k + 4 / k) := sorry

end monotonicity_of_f_range_of_x1_x2_l450_450652


namespace find_xyz_l450_450679

theorem find_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * (y + z) = 198) (h5 : y * (z + x) = 216) (h6 : z * (x + y) = 234) :
  x * y * z = 1080 :=
sorry

end find_xyz_l450_450679


namespace sum_of_fractions_l450_450123

theorem sum_of_fractions :
  (1 / (1^2 * 2^2) + 1 / (2^2 * 3^2) + 1 / (3^2 * 4^2) + 1 / (4^2 * 5^2)
  + 1 / (5^2 * 6^2) + 1 / (6^2 * 7^2)) = 48 / 49 := 
by
  sorry

end sum_of_fractions_l450_450123


namespace number_of_pumpkin_pies_l450_450869

-- Definitions for the conditions
def apple_pies : ℕ := 2
def pecan_pies : ℕ := 4
def total_pies : ℕ := 13

-- The proof statement
theorem number_of_pumpkin_pies
  (h_apple : apple_pies = 2)
  (h_pecan : pecan_pies = 4)
  (h_total : total_pies = 13) : 
  total_pies - (apple_pies + pecan_pies) = 7 :=
by 
  sorry

end number_of_pumpkin_pies_l450_450869


namespace sum_of_solutions_l450_450739

noncomputable def f (x : ℝ) := 9 * x + 2

noncomputable def f_inv (x : ℝ) := (x - 2) / 9

theorem sum_of_solutions : 
  (∑ x in { x : ℝ | f_inv x = f ((3 * x)⁻¹) }, x) = 20 :=
by
  sorry

end sum_of_solutions_l450_450739


namespace number_of_cars_parked_l450_450358

-- Definitions for the given conditions
def total_area (length width : ℕ) : ℕ := length * width
def usable_area (total : ℕ) : ℕ := (8 * total) / 10
def cars_parked (usable : ℕ) (area_per_car : ℕ) : ℕ := usable / area_per_car

-- Given conditions
def length : ℕ := 400
def width : ℕ := 500
def area_per_car : ℕ := 10
def expected_cars : ℕ := 16000 -- correct answer from solution

-- Define a proof statement
theorem number_of_cars_parked : cars_parked (usable_area (total_area length width)) area_per_car = expected_cars := by
  sorry

end number_of_cars_parked_l450_450358


namespace find_s30_l450_450625

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n m : ℕ, a m - a n = a (m+1) - a (n+1)

def sum_first_n_terms (a : ℕ → ℕ) (s : ℕ → ℕ) : Prop :=
∀ n : ℕ, s n = ∑ i in range n, a i

-- Conditions
variables (a : ℕ → ℕ) (s : ℕ → ℕ)

axiom arithmetic_sequence : is_arithmetic_sequence a
axiom sum_def : sum_first_n_terms a s
axiom s10_eq_s20 : s 10 = s 20

-- Proof Statement
theorem find_s30 : s 30 = 0 :=
sorry

end find_s30_l450_450625


namespace count_prime_diff_numbers_l450_450671

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_prime_diff (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p - q

def in_target_set (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 + 10 * k

def prime_diff_count : ℕ :=
  (Finset.filter (λ n => is_prime_diff n) (Finset.range (100))).card  -- assuming we consider first 100 numbers for simplicity

theorem count_prime_diff_numbers :
  ∃ k : ℕ, prime_diff_count = k :=
begin
  sorry
end

end count_prime_diff_numbers_l450_450671


namespace min_value_f_l450_450600

def f (x : ℝ) : ℝ := (x^2 + 8) / Real.sqrt (x^2 + 4)

theorem min_value_f : ∃ x : ℝ, f x = 4 ∧ (∀ y : ℝ, f y ≥ 4) :=
by
  sorry

end min_value_f_l450_450600


namespace smallest_N_l450_450253

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450253


namespace player1_wins_11th_round_l450_450467

theorem player1_wins_11th_round (player1_wins_first_10 : ∀ (round : ℕ), round < 10 → player1_wins round) : 
  prob_winning_11th_round player1 = 11 / 12 :=
sorry

end player1_wins_11th_round_l450_450467


namespace burattino_suspects_cheating_l450_450073

theorem burattino_suspects_cheating :
  ∃ n : ℕ, (0.4 ^ n) < 0.01 ∧ n + 1 = 7 :=
by
  sorry

end burattino_suspects_cheating_l450_450073


namespace circumscribed_quadrilateral_angle_sum_l450_450808

theorem circumscribed_quadrilateral_angle_sum (A B C D O : Type)
  (h : ∀ (a b : Type), (a ∈ [A, B]) ∧ (b ∈ [C, D]) → 
       ∃ (X : Type), (X ∈ [A, B, C, D]) ∧ (X = O)) :
  ∠(A, O, B) + ∠(C, O, D) = 180 :=
sorry

end circumscribed_quadrilateral_angle_sum_l450_450808


namespace cannot_reach_all_pluses_l450_450829

open Matrix

-- Define an 8x8 grid with values representing "+" and "-" as boolean
def Grid := Matrix (Fin 8) (Fin 8) Bool

-- Define the operation to flip signs within a sub-grid
def invertSubGrid (grid : Grid) (r c : Fin 6) (size : Fin 2) : Grid :=
  let n := if size = 0 then 3 else 4
  fun i j =>
    if r ≤ i < r + n ∧ c ≤ j < c + n then
      not (grid i j)
    else
      grid i j

-- Theorem to state that we cannot always achieve a grid with all '+' signs
theorem cannot_reach_all_pluses (init_grid : Grid) : 
  (∃ g : Grid, 
  (∀ r : Fin 6, ∀ c : Fin 6, ∀ size : Fin 2, invertSubGrid g r c size = init_grid) → ∀ i j, g i j = true) → 
  False :=
sorry

end cannot_reach_all_pluses_l450_450829


namespace circumradius_of_triangle_is_12_5_l450_450353

-- Define the conditions given in the problem
variables (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
variables (AB AC : ℝ)
variables (O P : Type) [metric_space O] [metric_space P]

-- Assume distances given in the problem
def AB_length : ℝ := 20
def AC_length : ℝ := 24

-- State the geometric condition with the circle and center on AC
def geometric_condition : Prop :=
  sorry -- Detailed geometric definitions would go here

-- Main theorem to prove
theorem circumradius_of_triangle_is_12_5 
  (AB_len : AB_length = 20) 
  (AC_len : AC_length = 24)
  (geo_cond : geometric_condition) : 
  ∃ (R : ℝ), R = 12.5 := 
begin
  -- The proof would be here
  sorry
end

end circumradius_of_triangle_is_12_5_l450_450353


namespace factor_27x6_minus_512y6_sum_coeffs_is_152_l450_450792

variable {x y : ℤ}

theorem factor_27x6_minus_512y6_sum_coeffs_is_152 :
  ∃ a b c d e f g h j k : ℤ, 
    (27 * x^6 - 512 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) ∧ 
    (a + b + c + d + e + f + g + h + j + k = 152) := 
sorry

end factor_27x6_minus_512y6_sum_coeffs_is_152_l450_450792


namespace find_n_l450_450779

theorem find_n (n : ℕ) (d : ℕ) (h_pos : n > 0) (h_digit : d < 10) (h_equiv : n * 999 = 810 * (100 * d + 25)) : n = 750 :=
  sorry

end find_n_l450_450779


namespace martin_toy_cars_sum_l450_450749

theorem martin_toy_cars_sum :
  let possible_M := {M | ∃ a b, M = 6 * a + 4 ∧ M = 8 * b + 5 ∧ M < 100}
  M ∈ possible_M → set.sum possible_M id = 244 :=
by
  sorry

end martin_toy_cars_sum_l450_450749


namespace incorrect_statement_A_l450_450614

theorem incorrect_statement_A (x_1 x_2 y_1 y_2 : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 - 2*x - 4*y - 4 = 0) ∧
  x_1 = 1 - Real.sqrt 5 ∧
  x_2 = 1 + Real.sqrt 5 ∧
  y_1 = 2 - 2 * Real.sqrt 2 ∧
  y_2 = 2 + 2 * Real.sqrt 2 →
  x_1 + x_2 ≠ -2 := by
  intro h
  sorry

end incorrect_statement_A_l450_450614


namespace geometric_sequence_first_term_l450_450136

theorem geometric_sequence_first_term (a r : ℝ) (h1 : a * r^2 = 18) (h2 : a * r^4 = 162) : a = 2 := by
  sorry

end geometric_sequence_first_term_l450_450136


namespace winning_strategy_l450_450752

def S : finset (fin 100 → fin 6) := sorry

noncomputable def f (x : fin 100 → fin 6) : fin 100 :=
if h : ∃ k, x k = 5 then classical.some h else 99

theorem winning_strategy : 
  ∑ a in S, ∑ b in S, (if a (f b) = 5 ∧ b (f a) = 5 then (1 : ℝ) else 0) > (6 ^ 200 / 36 : ℝ) :=
sorry

end winning_strategy_l450_450752


namespace min_value_f_l450_450130

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^2 - Real.sqrt 3 * Real.abs x + 1) + Real.sqrt (x^2 + Real.sqrt 3 * Real.abs x + 3)

theorem min_value_f : ∃ (x : ℝ), f x = √7 ∧ (x = √3/4 ∨ x = -√3/4) :=
by
  sorry

end min_value_f_l450_450130


namespace career_preference_angle_l450_450004

theorem career_preference_angle (x : ℝ) :
  let males := 2 * x,
      females := 3 * x,
      pref_males := (1/4) * males,
      pref_females := (3/4) * females,
      total_students := males + females,
      pref_students := pref_males + pref_females,
      pref_proportion := pref_students / total_students,
      degrees := 360 * pref_proportion
  in degrees = 198 := 
by 
  -- Assumptions given in the problem
  have males_eq2x : males = 2 * x := rfl,
  have females_eq3x : females = 3 * x := rfl,
  have total_students_eq5x : total_students = 5 * x := by simp [males_eq2x, females_eq3x],
  -- Calculating the number of students preferring the career
  have pref_males_eq : pref_males = (1/4) * (2 * x) := rfl,
  have pref_females_eq : pref_females = (3/4) * (3 * x) := rfl,
  have pref_students_eq : pref_students = (1/4) * (2 * x) + (3/4) * (3 * x) := by simp [pref_males_eq, pref_females_eq],
  -- Simplifying the fraction of students preferring the career
  have pref_students_eq_simplified : pref_students = (11/4) * x := by linarith,
  have pref_proportion_eq : pref_proportion = (11/4) * x / (5 * x) := by simp [total_students_eq5x, pref_students_eq_simplified],
  have pref_proportion_simplified : pref_proportion = 11/20 := by field_simp [pref_proportion_eq],
  -- Calculating the degrees of the circle representing the career preference
  have degrees_eq : degrees = 360 * (11/20) := rfl,
  have degrees_simplified : degrees = 198 := by norm_num [degrees_eq],
  exact degrees_simplified

end career_preference_angle_l450_450004


namespace find_p_l450_450709

variables (Q A B O C : ℝ × ℝ) (p : ℝ)

-- Definition of points based on the given coordinates
def Q : ℝ × ℝ := (0, 15)
def A : ℝ × ℝ := (3, 15)
def B : ℝ × ℝ := (15, 0)
def O : ℝ × ℝ := (0, 0)
def C (p : ℝ) : ℝ × ℝ := (0, p)

-- Condition that the area of triangle ABC is 36
def area_ABC : ℝ := 36

-- The target is to prove that p = 12.75
theorem find_p : area_ABC = 36 → C 12.75 = (0,12.75) :=
by
  intro h
  sorry

end find_p_l450_450709


namespace point_M_quadrant_l450_450639

theorem point_M_quadrant (θ : ℝ) (h1 : π / 2 < θ) (h2 : θ < π) :
  (0 < Real.sin θ) ∧ (Real.cos θ < 0) :=
by
  sorry

end point_M_quadrant_l450_450639


namespace num_samples_meet_requirements_l450_450523

/-- 
A batch of nut products has a requirement for the inner diameter to have an error of ±0.03mm.
5 samples are randomly selected for inspection and their results are recorded as follows:
+0.031, +0.017, +0.023, -0.021, -0.015.
Prove that the number of samples that meet the requirement (having deviations within the range -0.03mm to +0.03mm) is 4.
-/
theorem num_samples_meet_requirements : (∃ (samples : list ℝ), samples = [+0.031, +0.017, +0.023, -0.021, -0.015]
                                         ∧ (count (λ x : ℝ, x ≥ -0.03 ∧ x ≤ +0.03) samples = 4)) :=
begin
  sorry
end

end num_samples_meet_requirements_l450_450523


namespace orangeade_price_second_day_l450_450849

theorem orangeade_price_second_day :
  ∀ (X O : ℝ), (2 * X * 0.60 = 3 * X * E) → (E = 2 * 0.60 / 3) →
  E = 0.40 := by
  intros X O h₁ h₂
  sorry

end orangeade_price_second_day_l450_450849


namespace median_of_set_l450_450311

open List

def is_median (l : List ℝ) (m : ℝ) : Prop :=
  l.length % 2 = 1 ∧ (sorted l) ∧ (l.nth (l.length / 2)).iget = m

theorem median_of_set (s : List ℝ) (h_len : s.length = 100)
  (h1 : ∃ n, is_median (s.erase n) 78)
  (h2 : ∃ n, is_median (s.erase n) 66) :
  is_median s 72 :=
sorry

end median_of_set_l450_450311


namespace negation_of_exists_statement_l450_450436

theorem negation_of_exists_statement :
  ¬ (∃ x0 : ℝ, x0 > 0 ∧ x0^2 - 5 * x0 + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 :=
by
  sorry

end negation_of_exists_statement_l450_450436


namespace find_n_value_l450_450980

noncomputable theory

open real

theorem find_n_value (x n : ℝ)
  (h1 : log 10 (sin x) + log 10 (cos x) = -2)
  (h2 : log 10 (sin x + cos x) = 1 / 2 * (log 10 n + 1)) :
  n = 0.102 :=
sorry

end find_n_value_l450_450980


namespace necessary_and_sufficient_condition_l450_450374

def point := (ℝ × ℝ)

def U : set point := {p | True}
def A (m : ℝ) : set point := {p | 2 * p.1 - p.2 + m > 0}
def B (n : ℝ) : set point := {p | p.1 + p.2 - n ≤ 0}

def complement (s : set point) : set point := {p | ¬ s p}

def P : point := (2, 3)

theorem necessary_and_sufficient_condition (m n : ℝ) :
  P ∈ A m ∧ P ∈ complement (B n) ↔ m > -1 ∧ n < 5 :=
by
  sorry

end necessary_and_sufficient_condition_l450_450374


namespace modular_arithmetic_example_l450_450281

theorem modular_arithmetic_example {x : ℤ} (h : 5 * x + 3 ≡ 4 [MOD 16]) : 4 * x + 5 ≡ 9 [MOD 16] :=
sorry

end modular_arithmetic_example_l450_450281


namespace probability_of_winning_11th_round_l450_450474

-- Definitions of the conditions
def player1_wins_ten_rounds (eggs : List ℕ) : Prop :=
  ∀ i, i < 10 → eggs.indexOf (eggs.nthLe 0 (i+1)) < eggs.indexOf (eggs.nthLe 1 (i+1))

def is_strongest (egg : ℕ) (eggs : List ℕ) : Prop :=
  egg = List.maximum (0 :: eggs)

-- The proof to show the probability of winning the 11th round
theorem probability_of_winning_11th_round
  (eggs : List ℕ) : player1_wins_ten_rounds eggs →
  (1 - 1 / (length eggs + 1) = 11 / 12) :=
by
  sorry

end probability_of_winning_11th_round_l450_450474


namespace smallest_possible_value_of_N_l450_450230

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l450_450230


namespace find_point_P_l450_450789

def point_on_line (A B C x1 y1 d : ℝ) : Prop :=
  abs (A * x1 + B * y1 + C) / (real.sqrt (A^2 + B^2)) = d

def in_region (x y : ℝ) : Prop := 
  2 * x + y < 4

def point_P (x y : ℝ) : Prop := 
  point_on_line 4 (-3) 1 x y 4 ∧ in_region x y

theorem find_point_P :
  ∃ (a : ℝ), point_P a 3 ∧ a = -3 :=
begin
  use -3,
  split,
  {
    unfold point_P point_on_line in_region,
    norm_num,
    rw abs_of_nonneg,
    norm_num,
    exact abs_nonneg (16:ℝ)
  },
  {
    refl,
  },
end

end find_point_P_l450_450789


namespace odd_function_express_f12_increasing_function_l450_450049

variable (f : ℝ → ℝ)
variable (a : ℝ)

lemma functional_equation (x y : ℝ) : f(x + y) = f(x) + f(y) := sorry

theorem odd_function : ∀ x : ℝ, f(-x) = -f(x) := sorry

theorem express_f12 : f(-3) = a → f(12) = -4 * a := sorry

theorem increasing_function (h : ∀ x > 0, f(x) > 0) : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f(x₁) < f(x₂) := sorry

end odd_function_express_f12_increasing_function_l450_450049


namespace cube_edge_length_l450_450028

theorem cube_edge_length
  (length_base : ℝ) (width_base : ℝ) (rise_level : ℝ) (volume_displaced : ℝ) (volume_cube : ℝ) (edge_length : ℝ)
  (h_base : length_base = 20) (h_width : width_base = 15) (h_rise : rise_level = 3.3333333333333335)
  (h_volume_displaced : volume_displaced = length_base * width_base * rise_level)
  (h_volume_cube : volume_cube = volume_displaced)
  (h_edge_length_eq : volume_cube = edge_length ^ 3)
  : edge_length = 10 :=
by
  sorry

end cube_edge_length_l450_450028


namespace paul_homework_average_l450_450116

def hoursOnWeeknights : ℕ := 2 * 5
def hoursOnWeekend : ℕ := 5
def totalHomework : ℕ := hoursOnWeeknights + hoursOnWeekend
def practiceNights : ℕ := 2
def daysAvailable : ℕ := 7 - practiceNights
def averageHomeworkPerNight : ℕ := totalHomework / daysAvailable

theorem paul_homework_average :
  averageHomeworkPerNight = 3 := 
by
  -- sorry because we skip the proof
  sorry

end paul_homework_average_l450_450116


namespace find_C_l450_450105

theorem find_C (A B C : ℕ) (k m : ℤ) (S1 S2 : ℕ) (h1 : S1 = A + B) (h2 : S2 = A + B + C) 
  (hS1 : S1 = 15 * k.nat_abs + 11) (hS2 : S2 % 15 = 0) : C = 4 :=
sorry

end find_C_l450_450105


namespace polynomial_coeffs_identity_l450_450596

theorem polynomial_coeffs_identity : 
  (∀ a b c : ℝ, (2 * x^4 + x^3 - 41 * x^2 + 83 * x - 45 = 
                (a * x^2 + b * x + c) * (x^2 + 4 * x + 9))
                  → a = 2 ∧ b = -7 ∧ c = -5) :=
by
  intros a b c h
  have h₁ : a = 2 := 
    sorry-- prove that a = 2
  have h₂ : b = -7 := 
    sorry-- prove that b = -7
  have h₃ : c = -5 := 
    sorry-- prove that c = -5
  exact ⟨h₁, h₂, h₃⟩

end polynomial_coeffs_identity_l450_450596


namespace black_white_ratio_l450_450606

-- Given conditions
def radius1 := 2
def radius2 := 4
def radius3 := 6
def radius4 := 8
def radius5 := 10

-- Areas of circles
def area (r : ℕ) : ℝ := π * r ^ 2

-- White areas
def white1 := area radius1
def white2 := area radius3 - area radius2
def white3 := area radius5 - area radius4

-- Black areas
def black1 := area radius2 - area radius1
def black2 := area radius4 - area radius3

-- Total areas
def totalWhite := white1 + white2 + white3
def totalBlack := black1 + black2

-- Ratio of black area to white area
def black_to_white_ratio := totalBlack / totalWhite

theorem black_white_ratio : black_to_white_ratio = 2 / 3 := 
  by
    sorry

end black_white_ratio_l450_450606


namespace wickets_before_last_match_l450_450546

theorem wickets_before_last_match
  (W : ℝ)  -- Number of wickets before last match
  (R : ℝ)  -- Total runs before last match
  (h1 : R = 12.4 * W)
  (h2 : (R + 26) / (W + 8) = 12.0)
  : W = 175 :=
sorry

end wickets_before_last_match_l450_450546


namespace cost_of_camel_l450_450859

-- Define the cost of each animal as variables
variables (C H O E : ℝ)

-- Assume the given relationships as hypotheses
def ten_camels_eq_twentyfour_horses := (10 * C = 24 * H)
def sixteens_horses_eq_four_oxen := (16 * H = 4 * O)
def six_oxen_eq_four_elephants := (6 * O = 4 * E)
def ten_elephants_eq_140000 := (10 * E = 140000)

-- The theorem that we want to prove
theorem cost_of_camel (h1 : ten_camels_eq_twentyfour_horses C H)
                      (h2 : sixteens_horses_eq_four_oxen H O)
                      (h3 : six_oxen_eq_four_elephants O E)
                      (h4 : ten_elephants_eq_140000 E) :
  C = 5600 := sorry

end cost_of_camel_l450_450859


namespace burattino_suspects_cheating_l450_450071

theorem burattino_suspects_cheating :
  ∃ n : ℕ, (0.4 ^ n) < 0.01 ∧ n + 1 = 7 :=
by
  sorry

end burattino_suspects_cheating_l450_450071


namespace find_ending_number_l450_450083

theorem find_ending_number (n : ℕ) 
  (h1 : ∃ numbers, numbers = (filter (λ x, x % 2 = 0) (list.range' 12 (n - 12 + 1)))) 
  (h2 : (list.sum (filter (λ x, x % 2 = 0) (list.range' 12 (n - 12 + 1)))) / (list.length (filter (λ x, x % 2 = 0) (list.range' 12 (n - 12 + 1)))) = 19) :
  n = 26 := 
sorry

end find_ending_number_l450_450083


namespace median_of_100_set_l450_450319

theorem median_of_100_set 
  (S : Finset ℝ) (h_card : S.card = 100)
  (h_remove1 : ∃ x ∈ S, median (S.erase x) = 78)
  (h_remove2 : ∃ y ∈ S, median (S.erase y) = 66) : 
  median S = 72 :=
by
  sorry

end median_of_100_set_l450_450319


namespace distance_to_face_ABC_l450_450764
open Real

/-- Define the coordinates of the points A, B, C, and T in space./
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

/-- Given points in space -/
variables (A B C T : Point)

/-- Define the distances between points -/
noncomputable def distance (P Q : Point) : ℝ :=
  Real.sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2 + (Q.z - P.z)^2)

/-- Conditions as per the problem statement -/
axiom TA_perpendicular_TB : (T.x - A.x) * (T.x - B.x) + (T.y - A.y) * (T.y - B.y) + (T.z - A.z) * (T.z - B.z) = 0
axiom TB_perpendicular_TC : (T.x - B.x) * (T.x - C.x) + (T.y - B.y) * (T.y - C.y) + (T.z - B.z) * (T.z - C.z) = 0
axiom TC_not_perpendicular_TA : (T.x - C.x) * (T.x - A.x) + (T.y - C.y) * (T.y - A.y) + (T.z - C.z) * (T.z - A.z) ≠ 0
axiom TA_length : distance T A = 10
axiom TB_length : distance T B = 15
axiom TC_length : distance T C = 8

/-- The goal to prove distance from T to face ABC. -/
noncomputable def distance_from_T_to_ABC (A B C T : Point) : ℝ :=
  let area_TAB := 0.5 * 10 * 15 in
  let volume_TABC := (1 / 3) * area_TAB * 8 in
  let AB := Real.sqrt (10^2 + 15^2) in
  let AC := Real.sqrt (10^2 + 8^2) in
  let BC := 17 in
  let s := (AB + AC + BC) / 2 in
  let area_ABC := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC)) in
  (3 * volume_TABC) / area_ABC

/-- Required statement to prove -/
theorem distance_to_face_ABC :
  distance_from_T_to_ABC A B C T = 600 / Real.sqrt ((distance (A B C).x - A.x)^2 + (distance (A B C).y - A.y)^2 + (distance (A B C).z - A.z)^2) := by 
  sorry

end distance_to_face_ABC_l450_450764


namespace lines_concurrent_l450_450565

-- Define the given triangle and point P
variables (A B C P D E F X Z Y : Type)
variables [geometry_instance : geometry]

-- Define perpendicular relations and the points they intersect on the sides.
axiom perp_PD_BC (P D : point) : is_perpendicular P D  
axiom perp_PE_CA (P E : point) : is_perpendicular P E
axiom perp_PF_AB (P F : point) : is_perpendicular P F

-- Define points X, Z, and Y and their perpendicular relations.
axiom on_PD (X : point) : lies_on_line X P D
axiom perp_X_PB (X Z : point) : is_perpendicular X Z
axiom perp_X_PC (X Y : point) : is_perpendicular X Y

-- Prove lines AX, BY, and CZ are concurrent
theorem lines_concurrent :
  concurrent_lines P A B C X Z Y :=
sorry

end lines_concurrent_l450_450565


namespace number_of_possible_scenarios_l450_450296

-- Definitions based on conditions
def num_companies : Nat := 5
def reps_company_A : Nat := 2
def reps_other_companies : Nat := 1
def total_speakers : Nat := 3

-- Problem statement
theorem number_of_possible_scenarios : 
  ∃ (scenarios : Nat), scenarios = 16 ∧ 
  (scenarios = 
    (Nat.choose reps_company_A 1 * Nat.choose 4 2) + 
    Nat.choose 4 3) :=
by
  sorry

end number_of_possible_scenarios_l450_450296


namespace cartesian_curve_length_segment_AB_l450_450994

def parametric_line (t : ℝ) : ℝ × ℝ :=
  (sqrt 3 + (1 / 2) * t, 2 + (sqrt 3 / 2) * t)

def parametric_curve (θ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos θ, 4 * Real.sin θ)

theorem cartesian_curve (x y : ℝ) (θ : ℝ) :
  (x = 4 * Real.cos θ) ∧ (y = 4 * Real.sin θ) → (x^2 + y^2 = 16) :=
  by
  intros h
  cases h with h₁ h₂
  rw [h₁, h₂]
  calc
    (4 * Real.cos θ)^2 + (4 * Real.sin θ)^2 
      = 16 * (Real.cos θ)^2 + 16 * (Real.sin θ)^2 : by ring
  ... = 16 * ((Real.cos θ)^2 + (Real.sin θ)^2) : by rw [←mul_add]
  ... = 16 * 1                             : by rw [Real.cos_square_add_sin_square θ]
  ... = 16                                 : by ring

theorem length_segment_AB (t₁ t₂ : ℝ) :
  ( (sqrt 3 + (1 / 2) * t₁)^2 + (2 + (sqrt 3 / 2) * t₁)^2 = 16 ) ∧
  ( (sqrt 3 + (1 / 2) * t₂)^2 + (2 + (sqrt 3 / 2) * t₂)^2 = 16 ) →
  abs (3 * sqrt 7) = 3 * sqrt 7 :=
  by
  intros h
  sorry  -- Prove the quadratic relationship and calculate the length

end cartesian_curve_length_segment_AB_l450_450994


namespace H_subgroup_Q_union_H_union_subgroups_eq_Q_l450_450608

def H (n : ℕ) : Set ℚ := {x | ∃ k : ℤ, x = k / n!}

theorem H_subgroup (n : ℕ) (hn : 0 < n) : is_subgroup (H n) := sorry

theorem Q_union_H : (⋃ n in {n : ℕ | 0 < n}, H n) = Set.univ := sorry

theorem union_subgroups_eq_Q {m : ℕ} (G : Fin m → Set ℚ)
  (h_subgroups : ∀ i, is_subgroup (G i))
  (h_neq_Q : ∀ i, G i ≠ Set.univ) :
  (⋃ i, G i) ≠ Set.univ := sorry

end H_subgroup_Q_union_H_union_subgroups_eq_Q_l450_450608


namespace password_problem_l450_450340

theorem password_problem (n : ℕ) :
  (n^4 - n * (n - 1) * (n - 2) * (n - 3) = 936) → n = 6 :=
by
  sorry

end password_problem_l450_450340


namespace lisa_earns_more_than_tommy_l450_450378

theorem lisa_earns_more_than_tommy {total_earnings : ℤ} (h1 : total_earnings = 60) :
  let lisa_earnings := total_earnings / 2
  let tommy_earnings := lisa_earnings / 2
  lisa_earnings - tommy_earnings = 15 :=
by
  sorry

end lisa_earns_more_than_tommy_l450_450378


namespace sum_a_geq_two_div_1009_l450_450852

open Real

theorem sum_a_geq_two_div_1009 (a : Fin 2018 → ℝ) (x : Fin 2019 → ℝ) 
  (hpos : ∀ i, 0 < a i) 
  (hnot_zero : ∃ i, x i ≠ 0) 
  (hbound : x 0 = 0 ∧ x 2018 = 0)
  (heq : ∀ k, 1 ≤ k → k ≤ 2017 → x (k - 1) - 2 * x k + x (k + 1) + a k * x k = 0) :
    (∑ i in range 1 2018, a i) ≥ 2 / 1009 := 
by
  sorry

end sum_a_geq_two_div_1009_l450_450852


namespace num_proper_subsets_M_inter_N_l450_450659

universes u

-- Define the sets M and N
def M : Set ℤ := {-1, 0, 1}
def N : Set ℝ := {y | ∃ (x : ℤ) (hx : x ∈ M), y = 1 - Real.cos (Real.pi / 2 * x)}

-- Define the intersection of sets M and N
def M_inter_N : Set ℝ := M ∩ N

-- Define the number of proper subsets of a set
def num_proper_subsets (s : Set α) [Fintype {A : Set α // A ⊂ s}] : ℕ :=
  Fintype.card {A : Set α // A ⊂ s}

-- State the theorem
theorem num_proper_subsets_M_inter_N : num_proper_subsets M_inter_N = 3 :=
sorry

end num_proper_subsets_M_inter_N_l450_450659


namespace median_of_100_numbers_l450_450331

theorem median_of_100_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 100)
  (h_median_99_1 : ∀ num ∈ numbers, median (numbers.erase num) = 78 → num ∈ numbers)
  (h_median_99_2 : ∀ num ∈ numbers, median (numbers.erase num) = 66 → num ∈ numbers) :
  median numbers = 72 :=
sorry

end median_of_100_numbers_l450_450331


namespace player1_wins_11th_round_l450_450466

theorem player1_wins_11th_round (player1_wins_first_10 : ∀ (round : ℕ), round < 10 → player1_wins round) : 
  prob_winning_11th_round player1 = 11 / 12 :=
sorry

end player1_wins_11th_round_l450_450466


namespace range_of_f_l450_450132

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem range_of_f : 
  ∃ (L U : ℝ), L = 2 ∧ U = 11 ∧ set.range (λ x : {x : ℝ // 1 ≤ x ∧ x < 5}, f x) = set.Ico L U :=
by
  sorry

end range_of_f_l450_450132


namespace simplify_and_combine_sqrt_12_with_sqrt_3_l450_450563

theorem simplify_and_combine_sqrt_12_with_sqrt_3 : simplify_and_combine (√12) (√3) :=
by
  unfold simplify_and_combine
  have h : √12 = 2 * √3, by
    calc
      √12 = √(4 * 3) : by rw [← mul_assoc, mul_comm, mul_assoc]
      ... = √4 * √3 : by rw [Real.sqrt_mul']
      ... = 2 * √3 : by rw [Real.sqrt_two]
  sorry

end simplify_and_combine_sqrt_12_with_sqrt_3_l450_450563


namespace burattino_suspects_after_seventh_draw_l450_450056

noncomputable def probability (total : ℕ) (choose : ℕ) : ℚ := 
  (nat.factorial total / (nat.factorial choose * nat.factorial (total - choose))) 

noncomputable def suspicion_threshold (threshold : ℚ) (probability_per_draw : ℚ) : ℕ :=
  nat.find (λ n, probability_per_draw^n < threshold)

theorem burattino_suspects_after_seventh_draw :
  let a := (probability 39 6) / (probability 45 6) in
  let threshold := (1 : ℚ) / 100 in
  suspicion_threshold threshold a = 6 :=
by
  sorry

end burattino_suspects_after_seventh_draw_l450_450056


namespace limit_arcsin_expr_l450_450850

noncomputable def limit_expr : ℝ :=
  lim (λ x : ℝ, (arcsin (2 * x) / (2^(-3 * x) - 1)) * log 2) (nhds_within 0 (set.univ))

theorem limit_arcsin_expr : limit_expr = - (2 / 3) :=
by
  sorry

end limit_arcsin_expr_l450_450850


namespace triangle_max_area_BQC_l450_450717

noncomputable def inTriangle {α : Type*} [LinearOrder α] (a b c : α) : Set α :=
{ x | a < x ∧ x < b ∨ b < x ∧ x < c ∨ c < x ∧ x < a }

noncomputable def maximum_area_BQC (p q r : ℕ) : ℝ :=
112.5 - 56.25 * Real.sqrt 3

theorem triangle_max_area_BQC (AB BC CA : ℝ) (D : ℝ) (J_B J_C Q : ℝ) (p q r : ℕ) (hsides: AB = 13 ∧ BC = 15 ∧ CA = 14) (hint: inTriangle BC 0 1 D) :
  p + q + r = 171.75 :=
by
  have h : maximum_area_BQC p q r = 112.5 - 56.25 * Real.sqrt 3 := rfl
  sorry

end triangle_max_area_BQC_l450_450717


namespace general_formula_bn_find_S3_l450_450973

-- Define the arithmetic and geometric sequences and their initial conditions
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) : Prop :=
  a 1 = a1 ∧ ∀ n, a (n + 1) = a n + d

def geometric_seq (b : ℕ → ℤ) (q : ℤ) (b1 : ℤ) : Prop :=
  b 1 = b1 ∧ ∀ n, b (n + 1) = b n * q

-- Conditions from the problem
def conditions (a : ℕ → ℤ) (b : ℕ → ℤ) (d q : ℤ) : Prop :=
  arithmetic_seq a d (-1) ∧ geometric_seq b q 1 ∧
  a 2 + b 2 = 2 ∧ a 3 + b 3 = 5

-- Problem 1: Prove the general formula for b_n
theorem general_formula_bn (b : ℕ → ℤ) (q : ℤ) (n : ℕ) (h : ∀ n, n > 0 → b n = 2 ^ (n - 1)) :
  ∀ n, n > 0 → b n = 2^(n-1) :=
by
  sorry

-- Problem 2: Prove S₃ given T₃ = 21 and q = 4
theorem find_S3 (a b : ℕ → ℤ) (d q : ℤ):
  conditions a b d q → q = 4 → (1 + q + q^2 = 21) → 
  let S_3 := a 1 + a 2 + a 3 in S_3 = -6 :=
by
  sorry

end general_formula_bn_find_S3_l450_450973


namespace matrix_inverse_problem_l450_450686

variables {α : Type*} [field α] [decidable_eq α] {n : ℕ}
variables (B : matrix (fin n) (fin n) α)
variables (I : matrix (fin n) (fin n) α) [invertible B]

-- Conditions 
def matrix_condition1 : Prop := (B - 3 • I) ⬝ (B - 5 • I) = 0

-- Theorem we aim to prove
theorem matrix_inverse_problem (h_inv : invertible B) (h_cond : matrix_condition1 B I):
  B + 15 • (⅟B) = 8 • I :=
sorry

end matrix_inverse_problem_l450_450686


namespace smallest_positive_period_of_f_sum_max_min_values_of_f_l450_450210

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x) - 1

theorem smallest_positive_period_of_f :
  ∀ T > 0, (∀ x, f (x + T) = f x) ↔ T = Real.pi :=
sorry

theorem sum_max_min_values_of_f :
  (∀ (x ∈ Set.Icc (-Real.pi / 6) (-Real.pi / 12)), 
    f x ≤ f (-Real.pi / 12) ∧ f x ≥ f (-Real.pi / 6)) →
  (f (-Real.pi / 6) + f (-Real.pi / 12)) = 0 :=
sorry

end smallest_positive_period_of_f_sum_max_min_values_of_f_l450_450210


namespace tan_ratio_max_tan_A_l450_450988

variable {a b c A B C : ℝ}

-- Assuming that we are dealing with a triangle and the given condition
def triangle_condition (a b c : ℝ) : Prop := b^2 + 3 * a^2 = c^2

-- Result for the ratio of tangents of angles C and B
theorem tan_ratio (h : triangle_condition a b c) (h_triangle : ∃ A B C : ℝ, a + b + c = π ∧ A + B + C = π) :
  (Real.tan C) / (Real.tan B) = -2 := 
sorry

-- Maximum value of tan A
theorem max_tan_A (h : triangle_condition a b c) (h_triangle : ∃ A B C : ℝ, a + b + c = π ∧ A + B + C = π) :
  ∃ tan_max : ℝ, tan_max = sqrt 2 / 4 ∧ ∀ x : ℝ, x = Real.tan A → x ≤ sqrt 2 / 4 :=
sorry

end tan_ratio_max_tan_A_l450_450988


namespace set_intersection_example_l450_450219

theorem set_intersection_example :
  let A := { y | ∃ x, y = Real.log x / Real.log 2 ∧ x ≥ 3 }
  let B := { x | x^2 - 4 * x + 3 = 0 }
  A ∩ B = {3} :=
by
  sorry

end set_intersection_example_l450_450219


namespace B_can_complete_work_in_6_days_l450_450841

theorem B_can_complete_work_in_6_days (A B : ℝ) (h1 : (A + B) = 1 / 4) (h2 : A = 1 / 12) : B = 1 / 6 := 
by
  sorry

end B_can_complete_work_in_6_days_l450_450841


namespace suitable_for_sampling_survey_l450_450046

def survey_A : Prop :=
  ∀ (safety_bars: Fin 40 → Bool), ∃ (s : Fin 40), ¬ safety_bars s = true

def survey_B : Prop :=
  ∀ (students: List Bool), Perm (students, List.repeat true students.length)

def survey_C : Prop :=
  ∃ (sample: List Bool) (population: List Bool), sample ⊂ population

def survey_D : Prop :=
  ∀ (parts: List Bool), ∃ (p : List Bool), p.length = parts.length ∧ p = parts

theorem suitable_for_sampling_survey : survey_C := sorry

end suitable_for_sampling_survey_l450_450046


namespace union_of_sets_l450_450220

open Set

variable (a : ℤ)

def setA : Set ℤ := {1, 3}
def setB (a : ℤ) : Set ℤ := {a + 2, 5}

theorem union_of_sets (h : {3} = setA ∩ setB a) : setA ∪ setB a = {1, 3, 5} :=
by
  sorry

end union_of_sets_l450_450220


namespace tangent_line_eqn_l450_450950

theorem tangent_line_eqn 
  (x y : ℝ)
  (H_curve : y = x^3 + 3 * x^2 - 5)
  (H_point : (x, y) = (-1, -3)) :
  (3 * x + y + 6 = 0) := 
sorry

end tangent_line_eqn_l450_450950


namespace team_average_correct_l450_450890

theorem team_average_correct (v w x y : ℝ) (h : v < w ∧ w < x ∧ x < y) :
  (B = A) :=
by
  let A := (v + w + x + y) / 4
  let B := ((v + w) / 2 + (x + y) / 2) / 2
  have hA : A = (v + w + x + y) / 4 := by refl
  have hB : B = (v + w + x + y) / 4 := by
    calc
      B = ((v + w) / 2 + (x + y) / 2) / 2 : by refl
      ... = (v + w + x + y) / 4 : by
        rw [add_div, add_div, div_add_div_same, add_assoc, add_comm (w + x), add_assoc]
  show B = A from by rw [hA, hB]
  sorry

end team_average_correct_l450_450890


namespace smallest_N_l450_450271

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450271


namespace player1_wins_11th_round_probability_l450_450480

-- Definitions based on the conditions
def egg_shell_strength (n : ℕ) : ℝ := sorry
def player1_won_first_10_rounds : Prop := sorry

-- Main theorem
theorem player1_wins_11th_round_probability
  (h : player1_won_first_10_rounds) :
  Prob (egg_shell_strength 11 > egg_shell_strength 12) = 11 / 12 := sorry

end player1_wins_11th_round_probability_l450_450480


namespace calculate_number_of_boys_l450_450413

theorem calculate_number_of_boys (old_average new_average misread correct_weight : ℝ) (number_of_boys : ℕ)
  (h1 : old_average = 58.4)
  (h2 : misread = 56)
  (h3 : correct_weight = 61)
  (h4 : new_average = 58.65)
  (h5 : (number_of_boys : ℝ) * old_average + (correct_weight - misread) = (number_of_boys : ℝ) * new_average) :
  number_of_boys = 20 :=
by
  sorry

end calculate_number_of_boys_l450_450413


namespace given_conditions_implies_correct_answer_l450_450987

noncomputable def is_binomial_coefficient_equal (n : ℕ) : Prop := 
  Nat.choose n 2 = Nat.choose n 6

noncomputable def sum_of_odd_terms (n : ℕ) : ℕ :=
  2 ^ (n - 1)

theorem given_conditions_implies_correct_answer (n : ℕ) (h : is_binomial_coefficient_equal n) : 
  n = 8 ∧ sum_of_odd_terms n = 128 := by 
  sorry

end given_conditions_implies_correct_answer_l450_450987


namespace motorboat_speed_half_distance_covered_l450_450547

theorem motorboat_speed_half_distance_covered (m : ℝ) :
  let v₀ := 5  -- Initial speed in m/s
  let k := m / 50  -- Proportionality coefficient
  let t_half := 10  -- Time in seconds to half the speed
  let v_half := v₀ / 2  -- Half of initial speed
  let distance := 50 * Real.log 2
  ∃ t s, t = t_half ∧ 
           s = distance ∧ 
           (v₀ / (t + 10) = v_half)  -- The condition when speed is half
           sorry -- prove distance based on the integrated equation

end motorboat_speed_half_distance_covered_l450_450547


namespace expression_evaluation_l450_450570

theorem expression_evaluation:
  ( (1/3)^2000 * 27^669 + Real.sin (60 * Real.pi / 180) * Real.tan (60 * Real.pi / 180) + (2009 + Real.sin (25 * Real.pi / 180))^0 ) = 
  (2 + 29/54) := by
  sorry

end expression_evaluation_l450_450570


namespace smallest_n_modulo_l450_450834

theorem smallest_n_modulo : ∃ n : ℕ, 5 * n ≡ 2023 [MOD 26] ∧ n = 51 :=
by {
  use 51,
  split,
  { 
    norm_num,
    exact (by norm_num : 5 * 51 % 26 = 21),
    exact (by norm_num : 2023 % 26 = 21),
    exact (by norm_num : 21 = 21)
  },
  {
    reflexivity
  }
}

end smallest_n_modulo_l450_450834


namespace minimum_dot_product_l450_450223

variables (a b : EuclideanSpace ℝ (Fin 2))

theorem minimum_dot_product (a b : EuclideanSpace ℝ (Fin 2)) (h : ‖a - 2 • b‖ ≤ 2) : 
  a ⬝ b ≥ -1 / 2 := 
sorry

end minimum_dot_product_l450_450223


namespace dice_product_not_odd_probability_l450_450460

theorem dice_product_not_odd_probability :
  let odd_faces := {1, 3, 5}
  let even_faces := {2, 4, 6}
  let total_outcomes := 6 * 6
  let odd_product_outcomes := 3 * 3
  let even_product_outcomes := total_outcomes - odd_product_outcomes
  let probability : ℚ := even_product_outcomes / total_outcomes
  probability = 3 / 4 :=
by
  sorry

end dice_product_not_odd_probability_l450_450460


namespace burattino_suspects_cheating_after_seventh_draw_l450_450068

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def binomial (n k : ℕ) : ℚ :=
(factorial n) / ((factorial k) * factorial (n - k))

noncomputable def probability_no_repeats : ℚ :=
(binomial 39 6) / (binomial 45 6)

noncomputable def estimate_draws_needed : ℕ :=
let a : ℚ := probability_no_repeats in
nat.find (λ n, a^n < 0.01)

theorem burattino_suspects_cheating_after_seventh_draw :
  estimate_draws_needed + 1 = 7 := sorry

end burattino_suspects_cheating_after_seventh_draw_l450_450068


namespace savings_from_discount_l450_450364

-- Define the initial price
def initial_price : ℝ := 475.00

-- Define the discounted price
def discounted_price : ℝ := 199.00

-- The theorem to prove the savings amount
theorem savings_from_discount : initial_price - discounted_price = 276.00 :=
by 
  -- This is where the actual proof would go
  sorry

end savings_from_discount_l450_450364


namespace num_mappings_l450_450658

def A : set char := {'a', 'b'}
def B : set ℕ := {0, 1}

-- Define what it means to be a function from A to B
def is_function {α β : Type} (f : α → β) (A : set α) (B: set β) : Prop :=
  ∀ x ∈ A, f x ∈ B

-- Define the set of all functions from A to B
noncomputable def all_functions (A : set char) (B : set ℕ) : set (char → ℕ) :=
  {f | is_function f A B}

-- Theorem statement
theorem num_mappings : fintype.card (all_functions A B) = 4 :=
sorry

end num_mappings_l450_450658


namespace player1_wins_11th_round_l450_450488

noncomputable def egg_strength_probability (n : ℕ) : ℚ :=
  (n - 1) / n

theorem player1_wins_11th_round :
  let player1_wins_first_10_rounds := true,
      total_rounds := 11,
      new_egg := 12 in
  player1_wins_first_10_rounds → egg_strength_probability total_rounds = 11 / 12 :=
by
  intros
  exact sorry

end player1_wins_11th_round_l450_450488


namespace part1_part2_part3_l450_450649

-- Definitions for the given function and its derivative
def f (a x : ℝ) : ℝ := Real.exp x * (a * x^2 + x - 1)
def f' (a x : ℝ) : ℝ := Real.exp x * (2 * a * x + 1) + Real.exp x * (a * x^2 + x - 1)

-- Part 1: Prove that a = 2 if the function has an extremum at x = -5/2
theorem part1 (a : ℝ) (h : f' a (-5/2) = 0) : a = 2 := by
  sorry

-- Part 2: Under the condition a = 2, prove that f(x) has a local maximum at x = -5/2
theorem part2 (h : a = 2) : ¬ ∃ x, f' a x < 0 := by
  sorry

-- Part 3: For a = 1, prove that there are exactly 3 tangent lines passing through (3/4, 0)
noncomputable def g (a x : ℝ) : ℝ := 4 * x^3 + 5 * x^2 - 13 * x + 4

theorem part3 {x : ℝ} (a : ℝ) (h : a = 1) : ∃! l : ℝ, ¬ g a l < 0 := by
  sorry

end part1_part2_part3_l450_450649


namespace third_side_length_l450_450183

theorem third_side_length (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) (h3 : 3 < x < 13) : (5 < x + 8) ∧ (x < 5 + 8) ∧ (5 < x + 3) ∧ (x < 8 + 5) := 
by
  sorry

end third_side_length_l450_450183


namespace boys_number_l450_450866

variable (M W B : ℕ)

-- Conditions
axiom h1 : M = W
axiom h2 : W = B
axiom h3 : M * 8 = 120

theorem boys_number :
  B = 15 := by
  sorry

end boys_number_l450_450866


namespace empty_one_box_l450_450009
   
   theorem empty_one_box (a b c : ℕ) (h : a ≤ b ∧ b ≤ c) :
     ∃ (steps : ℕ → (ℕ × ℕ × ℕ)), steps 0 = (a, b, c) ∧ 
      (∃ n : ℕ, steps (n + 1) = (steps n).1 ∧ 
        (steps n).2 = 0 ∨ (steps n).3 = 0 ∨ (steps n).4 = 0) :=
   sorry
   
end empty_one_box_l450_450009


namespace minimize_perimeter_l450_450630

-- Definitions
variables {Point : Type} [metric_space Point]
variables (A B C M : Point)

-- Condition: M lies inside acute ∠BAC
def inside_acute_angle (M A B C : Point) : Prop :=
  let ∠BAC := ∠ B A C in
  ∠BAC < 90 ∧ (M inside angle ∠BAC)

-- Reflection definitions
def reflect_across (p l : Point) : Point :=
  sorry  -- Reflection implementation

def M1 := reflect_across M (segment AB)
def M2 := reflect_across M (segment AC)

-- The main theorem statement
theorem minimize_perimeter
  (X Y : Point)
  (h_inside : inside_acute_angle M A B C)
  (h_X_on_AB : X ∈ segment AB)
  (h_Y_on_AC : Y ∈ segment AC)
  (h_intersection_X : X = (line_segment M1 M2 ∩ line segment AB))
  (h_intersection_Y : Y = (line_segment M1 M2 ∩ line segment AC)) :
  ∀ (X_other Y_other : Point), X_other ∈ segment AB → 
  Y_other ∈ segment AC → perimeter (triangle M X_other Y_other) ≥ perimeter (triangle M X Y) :=
sorry

end minimize_perimeter_l450_450630


namespace divisors_of_expression_l450_450404

theorem divisors_of_expression (a b : ℤ) (h : 4 * b = 9 - 3 * a) :
  ({1, 2, 3, 4, 5, 6, 7, 8} ∩ {d | d ∣ (3 * b + 18)}).card = 4 := by
  sorry

end divisors_of_expression_l450_450404


namespace find_interest_rate_l450_450901

-- Definitions based on conditions
def P : ℝ := sorry    -- Principal amount
def R : ℝ := sorry    -- Rate of interest
def T : ℝ := 10       -- Time period in years
def SI : ℝ := (2/5) * P  -- Simple interest is 2/5 of principal amount

-- Simple interest formula
def simple_interest (P R T : ℝ) : ℝ :=
  (P * R * T) / 100

-- The required theorem statement
theorem find_interest_rate (P : ℝ) (H : simple_interest P R T = SI) : R = 4 :=
by simp [simple_interest, SI, T] at H
sorry

end find_interest_rate_l450_450901


namespace solve_equation_and_evaluate_l450_450423

theorem solve_equation_and_evaluate (c d : ℝ) (h1 : c = 3 + sqrt 19) (h2 : d = 3 - sqrt 19) (h3 : c ≥ d) :
  3 * c + 2 * d = 15 + sqrt 19 :=
by
  -- The proof goes here.
  sorry

end solve_equation_and_evaluate_l450_450423


namespace probability_of_winning_11th_round_l450_450477

-- Definitions of the conditions
def player1_wins_ten_rounds (eggs : List ℕ) : Prop :=
  ∀ i, i < 10 → eggs.indexOf (eggs.nthLe 0 (i+1)) < eggs.indexOf (eggs.nthLe 1 (i+1))

def is_strongest (egg : ℕ) (eggs : List ℕ) : Prop :=
  egg = List.maximum (0 :: eggs)

-- The proof to show the probability of winning the 11th round
theorem probability_of_winning_11th_round
  (eggs : List ℕ) : player1_wins_ten_rounds eggs →
  (1 - 1 / (length eggs + 1) = 11 / 12) :=
by
  sorry

end probability_of_winning_11th_round_l450_450477


namespace pyramid_base_length_l450_450411

theorem pyramid_base_length (A s h : ℝ): A = 120 ∧ h = 40 ∧ (A = 1/2 * s * h) → s = 6 := 
by
  sorry

end pyramid_base_length_l450_450411


namespace crayons_selection_l450_450819

theorem crayons_selection (X : ℕ) :
  let total_crayons := 15 in
  let total_colors := 4 in
  let red_crayons := 4 in
  let blue_crayons := 5 in
  let green_crayons := 3 in
  let yellow_crayons := 3 in 
  total_crayons = red_crayons + blue_crayons + green_crayons + yellow_crayons → 
  (∑ i in (finset.range total_crayons).choose 5, 1)  = 
  X := sorry

end crayons_selection_l450_450819


namespace smallest_possible_value_of_N_l450_450233

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l450_450233


namespace find_BC_l450_450545

noncomputable def BC_length (A B C O1 O2 : ℝ) (h_line : A ∈ segment B C) (h_proj_dist : dist O1 O2 = 12) : ℝ :=
BC

theorem find_BC (A B C O1 O2 : ℝ) (h_line : A ∈ segment B C) (h_proj_dist : dist (projection O1) (projection O2) = 12) :
  BC_length A B C O1 O2 h_line h_proj_dist = 24 :=
sorry

end find_BC_l450_450545


namespace value_of_b_l450_450803

noncomputable def k := 675

theorem value_of_b (a b : ℝ) (h1 : a * b = k) (h2 : a + b = 60) (h3 : a = 3 * b) (h4 : a = -12) :
  b = -56.25 := by
  sorry

end value_of_b_l450_450803


namespace find_p_q_l450_450688

theorem find_p_q (p q : ℝ) (M N : set ℝ) 
  (h1: M = {x | x^2 - p * x + 8 = 0})
  (h2: N = {x | x^2 - q * x + p = 0})
  (h3: M ∩ N = {1}) :
  p + q = 19 :=
sorry

end find_p_q_l450_450688


namespace factor_sum_l450_450124

theorem factor_sum :
  ∃ d e f : ℤ, (∀ x : ℤ, x^2 + 11 * x + 24 = (x + d) * (x + e)) ∧
              (∀ x : ℤ, x^2 + 9 * x - 36 = (x + e) * (x - f)) ∧
              d + e + f = 14 := by
  sorry

end factor_sum_l450_450124


namespace train_distance_in_45_minutes_l450_450877

theorem train_distance_in_45_minutes (d t : ℝ) (intervals : ℕ) (h1 : d = 1) (h2 : t = 2) (h3 : intervals = 22) :
  ∀ (total_time : ℝ), total_time = 45 → (intervals * d) = 22 :=
by
  intros total_time
  assume h_total_time
  have h_intervals_time : total_time / t = intervals + 0.5 := by sorry  -- This helps us ensure interval count.
  have h_correct_intervals : intervals = 22 := by sorry  -- Based on total_time == 45
  rw [h_correct_intervals, mul_comm]
  exact rfl

end train_distance_in_45_minutes_l450_450877


namespace median_of_updated_set_l450_450801

theorem median_of_updated_set :
  let S := [92, 88, 94, 90, x] in
  (∑ i in S, i) / 5 = 91 →
  let T := insert 93 S in
  multiset.median T = 91.5 :=
by
  sorry

end median_of_updated_set_l450_450801


namespace percentage_calculation_l450_450127

theorem percentage_calculation : ∀ (p a : ℝ), p = 0.25 → a = 800 → p * a = 200 :=
by
  intros p a hp ha
  rw [hp, ha]
  norm_num
  sorry

end percentage_calculation_l450_450127


namespace infinite_decimal_rational_l450_450399

noncomputable def T : ℚ := 0 + 1/10^1 + 4/10^2 + 9/10^3 + 6/10^4 + 5/10^5 + 6/10^6 + 9/10^7 + 4/10^8 + 1/10^9 + 
                   0/10^(10+1) + 1/10^(10+2) + 4/10^(10+3) + 9/10^(10+4) + 6/10^(10+5) + 5/10^(10+6) + 
                   6/10^(10+7) + 9/10^(10+8) + 4/10^(10+9) + 1/10^(10+10) + -- and so on

theorem infinite_decimal_rational : T = (166285490 / 1111111111) :=
sorry

end infinite_decimal_rational_l450_450399


namespace area_of_trapezoid_JKLM_l450_450351

-- Define trapezoid and conditions
variables {J K L M O : Type} [inhabited J] [inhabited K] [inhabited L] [inhabited M] [inhabited O]

-- Define the geometrical conditions and areas
variables (area_JKO area_JMO : ℝ)
  (h_parallel : ∀ {a b : ℝ}, K = a ∧ L = b → ∃ m n : ℝ, J = m ∧ M = n ∧ m ⬝ a = n ⬝ b)
  (h_JKO : area_JKO = 75) -- Area of triangle JKO
  (h_JMO : area_JMO = 45) -- Area of triangle JMO

-- Define the proof statement
theorem area_of_trapezoid_JKLM (area_JMOL : ℝ) :
  let area_OKM := area_JMO in
  let area_MOL := ((9 : ℝ) / 25) * area_JKO in
  let total_area := area_JKO + area_JMO + area_OKM + area_MOL in
  total_area = 192 :=
by
  -- Provide the necessary steps or skip proof assuming the steps lead to the result
  sorry

end area_of_trapezoid_JKLM_l450_450351


namespace coordinates_with_respect_to_origin_l450_450418

theorem coordinates_with_respect_to_origin (x y : ℤ) (hx : x = 3) (hy : y = -2) : (x, y) = (3, -2) :=
by
  sorry

end coordinates_with_respect_to_origin_l450_450418


namespace median_of_set_l450_450314

open List

def is_median (l : List ℝ) (m : ℝ) : Prop :=
  l.length % 2 = 1 ∧ (sorted l) ∧ (l.nth (l.length / 2)).iget = m

theorem median_of_set (s : List ℝ) (h_len : s.length = 100)
  (h1 : ∃ n, is_median (s.erase n) 78)
  (h2 : ∃ n, is_median (s.erase n) 66) :
  is_median s 72 :=
sorry

end median_of_set_l450_450314


namespace complex_computation_l450_450918

theorem complex_computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end complex_computation_l450_450918


namespace smallest_N_l450_450257

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450257


namespace lila_substituted_value_l450_450747

theorem lila_substituted_value:
  let a := 2
  let b := 3
  let c := 4
  let d := 5
  let f := 6
  ∃ e : ℚ, 20 * e = 2 * (3 - 4 * (5 - (e / 6))) ∧ e = -51 / 28 := sorry

end lila_substituted_value_l450_450747


namespace trajectory_of_M_l450_450159

theorem trajectory_of_M
  (A : ℝ × ℝ := (3, 0))
  (P_circle : ∀ (P : ℝ × ℝ), P.1^2 + P.2^2 = 1)
  (M_midpoint : ∀ (P M : ℝ × ℝ), M = ((P.1 + 3) / 2, P.2 / 2) → M.1 = x ∧ M.2 = y) :
  (∀ (x y : ℝ), (x - 3/2)^2 + y^2 = 1/4) := 
sorry

end trajectory_of_M_l450_450159


namespace sin_cos_solutions_l450_450675

noncomputable def sin_cos_solutions_count : ℝ :=
  let f1 (x : ℝ) := Math.sin (2 * x)
  let f2 (x : ℝ) := Math.cos (x / 2)
  -- Define bounds of the interval
  let a := (0 : ℝ)
  let b := (2 * Real.pi : ℝ)
  -- We're interested in the number of solutions where f1 meets f2 over the interval [a, b]
  number_of_solutions := 4

theorem sin_cos_solutions : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi → Math.sin (2 * x) = Math.cos (x / 2)) ↔ number_of_solutions = 4 :=
sorry

end sin_cos_solutions_l450_450675


namespace number_of_triples_l450_450607

theorem number_of_triples (n : ℕ) : 
  (Finset.card {p : ℕ × ℕ × ℕ | p.1 + p.2.1 + p.2.2 = 6 * n}) = 3 * n * n := 
by
  sorry

end number_of_triples_l450_450607


namespace sum_geometric_sequence_l450_450152

theorem sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * a 1)
  (h_a1 : a 1 = 1)
  (h_arithmetic : 4 * a 2 + a 4 = 2 * a 3) : 
  a 2 + a 3 + a 4 = 14 :=
sorry

end sum_geometric_sequence_l450_450152


namespace tiles_needed_l450_450876

def floor9ₓ12_ft : Type := {l : ℕ × ℕ // l = (9, 12)}
def tile4ₓ6_inch : Type := {l : ℕ × ℕ // l = (4, 6)}

theorem tiles_needed (floor : floor9ₓ12_ft) (tile : tile4ₓ6_inch) : 
  ∃ tiles : ℕ, tiles = 648 :=
sorry

end tiles_needed_l450_450876


namespace probability_sum_prime_or_multiple_of_4_l450_450458

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def sum_is_prime_or_multiple_of_4 (sum : ℕ) : Prop :=
  is_prime sum ∨ sum % 4 = 0

theorem probability_sum_prime_or_multiple_of_4 :
  let outcomes := { (i, j) | i ∈ Finset.range 1 9, j ∈ Finset.range 1 9 }
  let valid_outcomes := { (i, j) ∈ outcomes | sum_is_prime_or_multiple_of_4 (i + j) }
  (valid_outcomes.card : ℚ) / (outcomes.card : ℚ) = 39 / 64 :=
by {
  sorry
}

end probability_sum_prime_or_multiple_of_4_l450_450458


namespace internal_common_tangent_bisects_arc_l450_450414

-- Define the centers and point of tangencies
variables {A B C : Point} -- Centers of circles

-- Circles defined with respective centers and radii
variables {r1 r2 r3 : ℝ} -- Radii of the circles
variables {K1 : Circle} (hK1 : K1.center = A ∧ K1.radius = r1)
variables {K2 : Circle} (hK2 : K2.center = B ∧ K2.radius = r2)
variables {K3 : Circle} (hK3 : K3.center = C ∧ K3.radius = r3)

-- Points of tangencies and the external common tangent
variables {T P Q : Point} -- Points of tangency and intersection points
variables (HT : tangent_point K1 K2 = T)
variables (HPQ : external_common_tangent K1 K2 K3 = (P, Q))

-- The proof goal: The internal common tangent bisects the arc PQ closer to T
theorem internal_common_tangent_bisects_arc 
  (hT : externally_tangent K1 K2)
  (hPQRST : common_tangent_meeting_points K1 K2 K3 = (P, Q)) 
  (h_arc_condition : closer_arc_bisected_by_internal_tangent K1 K2 K3 PQ T) :
  bisects_internal_common_tangent K1 K2 K3 PQ T :=
sorry

end internal_common_tangent_bisects_arc_l450_450414


namespace sin_C_eq_sin_A_minus_B_eq_l450_450691

open Real

-- Problem 1
theorem sin_C_eq (A B C : ℝ) (a b c : ℝ)
  (hB : B = π / 3) 
  (h3a2b : 3 * a = 2 * b) 
  (hA_sum_B_C : A + B + C = π) 
  (h_sin_law_a : sin A / a = sin B / b) 
  (h_sin_law_b : sin B / b = sin C / c) :
  sin C = (sqrt 3 + 3 * sqrt 2) / 6 :=
sorry

-- Problem 2
theorem sin_A_minus_B_eq (A B C : ℝ) (a b c : ℝ)
  (h_cosC : cos C = 2 / 3) 
  (h3a2b : 3 * a = 2 * b) 
  (hA_sum_B_C : A + B + C = π) 
  (h_sin_law_a : sin A / a = sin B / b) 
  (h_sin_law_b : sin B / b = sin C / c) 
  (hA_acute : 0 < A ∧ A < π / 2)
  (hB_acute : 0 < B ∧ B < π / 2) :
  sin (A - B) = -sqrt 5 / 3 :=
sorry

end sin_C_eq_sin_A_minus_B_eq_l450_450691


namespace divisibility_condition_l450_450594

noncomputable def phi (x : ℝ) : ℝ := x^2 - x + 1

noncomputable def f (x : ℝ) (n : ℕ) (a : ℝ) : ℝ := (x - 1)^n + (x - 2)^(2 * n + 1) + (1 - x^2)^(2 * n + 1) + a

theorem divisibility_condition (n : ℕ) (a : ℝ) :
  (∃ c : ℕ, n = 3 * c ∧ a = -1) ↔ (∀ x : ℝ, φ x ∣ f x n a) :=
by sorry

end divisibility_condition_l450_450594


namespace triangle_third_side_length_l450_450193

theorem triangle_third_side_length (a b : ℝ) (x : ℝ) (h₁ : a = 5) (h₂ : b = 8) (hx : x ∈ {2, 3, 6, 13}) :
  3 < x ∧ x < 13 → x = 6 :=
by sorry

end triangle_third_side_length_l450_450193


namespace number_of_mappings_A_to_B_number_of_mappings_B_to_A_l450_450446

theorem number_of_mappings_A_to_B (A B : Finset ℕ) (hA : A.card = 5) (hB : B.card = 4) :
  (B.card ^ A.card) = 4^5 :=
by sorry

theorem number_of_mappings_B_to_A (A B : Finset ℕ) (hA : A.card = 5) (hB : B.card = 4) :
  (A.card ^ B.card) = 5^4 :=
by sorry

end number_of_mappings_A_to_B_number_of_mappings_B_to_A_l450_450446


namespace common_roots_product_l450_450425

theorem common_roots_product (C D u v w t : ℝ) 
  (h1: u + v + w = 0)
  (h2: u * v * w = -20)
  (h3: u * v + u * t + v * t = 0)
  (h4: u * v * t = -80)
  (h5: u * v = w * t) :
  ∃ (a b c : ℕ), a + b + c = 17 ∧ (u * v = a * ℝ.sqrt b c) :=
sorry

end common_roots_product_l450_450425


namespace steve_biking_time_l450_450402

theorem steve_biking_time (time_jordan : ℝ) (dist_jordan : ℝ) (dist_steve : ℝ) (dist_steve_target : ℝ) :
  dist_jordan = 3 ∧ time_jordan = 18 ∧ dist_steve = 5 ∧ dist_steve_target = 7 →
  let rate_steve := dist_steve / time_jordan in
  let time_steve := dist_steve_target / rate_steve in
  time_steve = 126 / 5 := 
by 
  intros h
  have h1 : rate_steve = dist_steve / time_jordan, by assumption
  have h2 : time_steve = dist_steve_target / rate_steve, by assumption
  simp [dist_jordan, time_jordan, dist_steve, dist_steve_target, rate_steve, time_steve] at h
  sorry

end steve_biking_time_l450_450402


namespace A_arrives_first_l450_450519

theorem A_arrives_first (s a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (a_ne_b : a ≠ b) :
  let x := (2 * s) / (a + b)
  let y := (s * (a + b)) / (2 * a * b)
  x < y :=
by
  let x := (2 * s) / (a + b)
  let y := (s * (a + b)) / (2 * a * b)
  have H : 4 * a * b < (a + b)^2,
    calc
      4 * a * b = (a + b)^2 - (a^2 - 2 * a * b + b^2) : by ring
      ... < (a + b)^2 : by linarith [a^2 - 2 * a * b + b^2 > 0, a_pos, b_pos]
  have x_y_eq : x / y = 4 * a * b / (a + b)^2, by sorry -- Calculation step
  linarith

end A_arrives_first_l450_450519


namespace smallest_N_value_proof_l450_450249

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l450_450249


namespace sum_of_p_squared_array_is_zero_mod_2011_l450_450018

theorem sum_of_p_squared_array_is_zero_mod_2011 :
  let p := 2011 in
  let sum_term := (2 * p^4) / ((2 * p^2 - 1) * (p^2 - 1)) in
  ∃ m n, sum_term = m / n ∧ m + n ≡ 0 [MOD p] :=
by -- begin
  sorry -- Proof goes here
-- end

end sum_of_p_squared_array_is_zero_mod_2011_l450_450018


namespace boys_belong_to_other_communities_l450_450696

-- Definitions for the given problem
def total_boys : ℕ := 850
def percent_muslims : ℝ := 0.34
def percent_hindus : ℝ := 0.28
def percent_sikhs : ℝ := 0.10
def percent_other : ℝ := 1 - (percent_muslims + percent_hindus + percent_sikhs)

-- Statement to prove that the number of boys belonging to other communities is 238
theorem boys_belong_to_other_communities : 
  (percent_other * total_boys) = 238 := by 
  sorry

end boys_belong_to_other_communities_l450_450696


namespace mimi_spent_on_clothes_l450_450750

theorem mimi_spent_on_clothes : 
  let A := 800
  let N := 2 * A
  let S := 4 * A
  let P := 1 / 2 * N
  let total_spending := 10000
  let total_sneaker_spending := A + N + S + P
  let amount_spent_on_clothes := total_spending - total_sneaker_spending
  amount_spent_on_clothes = 3600 := 
by
  sorry

end mimi_spent_on_clothes_l450_450750


namespace grocer_can_package_in_9_weighings_l450_450007

-- Definitions for the conditions
def pounds_of_tea : Nat := 20
def bag_weight : Nat := 2
def five_pound_weight : Nat := 5
def nine_pound_weight : Nat := 9

-- Theorem statement
theorem grocer_can_package_in_9_weighings : 
  ∃ (number_of_weighings : Nat), 
    number_of_weighings = 9 ∧ 
    (∀ remaining_tea : Nat, remaining_tea = pounds_of_tea - (number_of_weighings * bag_weight) -> remaining_tea = 0) :=
begin
  sorry
end

end grocer_can_package_in_9_weighings_l450_450007


namespace easter_egg_battle_probability_l450_450472

theorem easter_egg_battle_probability (players : Type) [fintype players] [decidable_eq players]
  (egg_strength : players → ℕ) (p1 : players) (p2 : players) (n : ℕ) [decidable (p1 ≠ p2)] :
  (∀ i in finset.range n, egg_strength p1 > egg_strength p2) →
  let prob11thWin := 11 / 12 in
  11 / 12 = prob11thWin :=
by sorry

end easter_egg_battle_probability_l450_450472


namespace rational_roots_of_polynomial_l450_450498

-- Given polynomial equation
def polynomial (x p : ℚ) : Prop :=
  4 * x^4 + 4 * p * x^3 = (p - 4) * x^2 - 4 * p * x + p

-- The proof statement
theorem rational_roots_of_polynomial (p : ℤ) :
  (∀ x : ℚ, polynomial x p → x ∈ ℚ) ↔ (p = 0 ∨ p = -1) :=
by sorry

end rational_roots_of_polynomial_l450_450498


namespace gcd_lcm_product_l450_450602

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 108) (h2 : b = 250) :
  Nat.gcd a b * Nat.lcm a b = a * b := by
  rw [h1, h2]
  have h_gcd : Nat.gcd 108 250 = 2 := by
    sorry
  have h_lcm : Nat.lcm 108 250 = 13500 := by
    sorry
  rw [h_gcd, h_lcm]
  norm_num
  sorry

end gcd_lcm_product_l450_450602


namespace possible_third_side_l450_450186

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end possible_third_side_l450_450186


namespace ab_value_l450_450001

theorem ab_value (a b : ℝ) (h : 6 * a = 20 ∧ 7 * b = 20) : 84 * (a * b) = 800 :=
by sorry

end ab_value_l450_450001


namespace black_king_check_problems_l450_450017

/-- 
In a game of chess played on a 1000 x 1000 board, with 499 white rooks and one black king, 
we prove the following:
1. The black king can always get into check after some finite number of moves.
2. The black king cannot move so that apart from some initial moves, it is always in check after its move.
3. The black king cannot move so that apart from some initial moves, it is always in check (even just after white has moved).
-/
theorem black_king_check_problems 
  (board_size : ℕ)
  (num_rooks : ℕ)
  (black_king_moves : set (fin board_size × fin board_size))
  (white_rook_moves : set (fin board_size × fin board_size))
  (taking_not_allowed : Prop)
  (king_allowed_in_check : Prop)
  (finite_moves : ∃ (n : ℕ), ∀ (pos_king : fin board_size × fin board_size), pos_king ∈ black_king_moves)
  (always_in_check_after_move : ∀ (pos_king : fin board_size × fin board_size), pos_king ∈ black_king_moves → ∀ (pos_rooks : Π i, fin num_rooks × fin num_rooks), pos_king ∈ white_rook_moves → pos_king ∈ black_king_moves):
  (finite_moves)
  ∧ ¬(always_in_check_after_move)
  ∧ ¬(always_in_check_after_move ∨ finite_moves) := 
sorry

end black_king_check_problems_l450_450017


namespace mr_klinker_twice_as_old_l450_450753

theorem mr_klinker_twice_as_old (x : ℕ) (current_age_klinker : ℕ) (current_age_daughter : ℕ)
  (h1 : current_age_klinker = 35) (h2 : current_age_daughter = 10) 
  (h3 : current_age_klinker + x = 2 * (current_age_daughter + x)) : 
  x = 15 :=
by 
  -- We include sorry to indicate where the proof should be
  sorry

end mr_klinker_twice_as_old_l450_450753


namespace man_speed_in_still_water_l450_450882

theorem man_speed_in_still_water
  (speed_of_current_kmph : ℝ)
  (time_seconds : ℝ)
  (distance_meters : ℝ)
  (speed_of_current_ms : ℝ := speed_of_current_kmph * (1000 / 3600))
  (speed_downstream : ℝ := distance_meters / time_seconds) :
  speed_of_current_kmph = 3 →
  time_seconds = 13.998880089592832 →
  distance_meters = 70 →
  (speed_downstream = (25 / 6)) →
  (speed_downstream - speed_of_current_ms) * (3600 / 1000) = 15 :=
by
  intros h_speed_current h_time h_distance h_downstream
  sorry

end man_speed_in_still_water_l450_450882


namespace expected_percentage_of_rain_l450_450297

/-- In a particular state, 60% of the counties have a 70% chance of receiving some rain on Monday.
    55% of the counties have an 80% chance of receiving some on Tuesday.
    40% of the counties have a 60% chance of receiving rain on Wednesday.
    No rain fell on any of these three days in 20% of the counties in the state. -/
theorem expected_percentage_of_rain :
  let m := 0.60 * 0.70, t := 0.55 * 0.80, w := 0.40 * 0.60, no_rain := 0.20 
  in (m * t * w * (1 - no_rain)) = 0.0354816 :=
by
  let m := 0.60 * 0.70
  let t := 0.55 * 0.80
  let w := 0.40 * 0.60
  let no_rain := 0.20
  have expected_percentage := m * t * w * (1 - no_rain)
  calc
    m * t * w * (1 - no_rain) = 0.60 * 0.70 * 0.55 * 0.80 * 0.40 * 0.60 * 0.80 : by 
      simp [m, t, w, no_rain]
    ... = 0.0354816 : sorry  -- numerical calculation to be verified

end expected_percentage_of_rain_l450_450297


namespace valid_triplet_exists_l450_450946

theorem valid_triplet_exists :
  ∃ (a b c : ℤ), 2 ≤ a ∧ a < b ∧ b < c ∧ (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ∧ (a = 4) ∧ (b = 5) ∧ (c = 6) :=
begin
  sorry
end

end valid_triplet_exists_l450_450946


namespace trigonometric_identity_solution_l450_450840

-- Define the conditions
def sin_ne_zero (x : ℝ) : Prop := Real.sin x ≠ 0
def cos_ne_zero (x : ℝ) : Prop := Real.cos x ≠ 0
def sin_cos_positive (x : ℝ) : Prop := Real.sin x * Real.cos x > 0

-- Define the theorem to prove
theorem trigonometric_identity_solution (x : ℝ) (k : ℤ) :
  sin_ne_zero x → cos_ne_zero x → sin_cos_positive x →
  (Real.sin x ^ 3 * (1 + Real.cos x / Real.sin x) + Real.cos x ^ 3 * (1 + Real.sin x / Real.cos x) = 
   2 * Real.sqrt (Real.sin x * Real.cos x)) →
  ∃ k : ℤ, x = π / 4 + 2 * k * π :=
begin
  intro h1,
  intro h2,
  intro h3,
  intro h4,
  -- Here would be the proof, but we are using sorry here as instructed.
  sorry
end

end trigonometric_identity_solution_l450_450840


namespace find_borrowed_amount_l450_450550

noncomputable def borrowed_amount (P : ℝ) : Prop :=
  let interest_paid := P * (4 / 100) * 2
  let interest_earned := P * (6 / 100) * 2
  let total_gain := 120 * 2
  interest_earned - interest_paid = total_gain

theorem find_borrowed_amount : ∃ P : ℝ, borrowed_amount P ∧ P = 3000 :=
by
  use 3000
  unfold borrowed_amount
  simp
  sorry

end find_borrowed_amount_l450_450550


namespace sum_possible_n_values_l450_450042

/-- Define the conditions using Triangle Inequality Theorem -/
def TriangleInequality (n : ℕ) : Prop := 
  5 < n ∧ n < 19

/-- Theorem to prove the sum of all possible values of n forming a valid triangle -/
theorem sum_possible_n_values : 
  (∑ n in Finset.filter (λ x, TriangleInequality x) (Finset.Icc 1 20), n) = 156 := 
by
  sorry

end sum_possible_n_values_l450_450042


namespace circle_tangent_y_axis_intersects_chord_l450_450615

open Real

noncomputable def circle_eqn (x y : ℝ) (hx1 : 3 * x = y) (hx2 : x ≠ 0) : Prop :=
  (∃ (h : x > 0), (x - 6 * sqrt 2)^2 + (y - 2 * sqrt 2)^2 = 72) ∨ 
  (∃ (h : x < 0), (x + 6 * sqrt 2)^2 + (y + 2 * sqrt 2)^2 = 72)

theorem circle_tangent_y_axis_intersects_chord (hx : ℝ) (hy : ℝ) 
  (h1 : ∃ t : ℝ, hx = 3 * t ∧ hy = t)
  (h2 : ∃ r : ℝ, r = abs (3 * (classical.some h1)) ∧ 
        (hx - hy = 0 ∨ (hx - 6 * sqrt 2 = 0 ∧ hy - 2 * sqrt 2 = 0))) :
  (circle_eqn hx hy (classical.some_spec h1).left (by linarith)) := sorry

end circle_tangent_y_axis_intersects_chord_l450_450615


namespace Gabriel_always_wins_l450_450960

/-- Representation of the game where Gabriel and Nora must choose numbers more than
    two units apart, Gabriel always starts first. -/
theorem Gabriel_always_wins (n : ℤ) (hn : n ≥ 7) : ∀ x ∈ set.Icc 0 n,
  (∀ y ∈ set.Icc 0 n, |x - y| > 2 → ∃ y' ∈ set.Icc 0 n, |y - y'| > 2) → ∃ x ∈ set.Icc 0 n, false :=
sorry

end Gabriel_always_wins_l450_450960


namespace A_minus_B_l450_450729

def A : ℕ := 3^7 + Nat.choose 7 2 * 3^5 + Nat.choose 7 4 * 3^3 + Nat.choose 7 6 * 3
def B : ℕ := Nat.choose 7 1 * 3^6 + Nat.choose 7 3 * 3^4 + Nat.choose 7 5 * 3^2 + 1

theorem A_minus_B : A - B = 128 := by
  sorry

end A_minus_B_l450_450729


namespace function_inequality_m_l450_450643

theorem function_inequality_m (m : ℝ) : (∀ x : ℝ, (1 / 2) * x^4 - 2 * x^3 + 3 * m + 9 ≥ 0) ↔ m ≥ (3 / 2) := sorry

end function_inequality_m_l450_450643


namespace cost_price_is_correct_l450_450444

-- Initialize the variables according to the problem.
variables (SP C : ℝ)
variable (tax_rate : ℝ := 0.10)
variable (profit_rate : ℝ := 0.16)
variable (sale_price_incl_tax : ℝ := 616)

-- Define the relationship between the variables.
def selling_price_before_tax (C : ℝ) : ℝ := (1 + profit_rate) * C
def sales_tax (SP : ℝ) : ℝ := tax_rate * SP
def total_sale_price (SP : ℝ) (ST : ℝ) : ℝ := SP + ST

-- The theorem stating that the cost price is Rs. 482.76
theorem cost_price_is_correct :
  ∀ C : ℝ, total_sale_price (selling_price_before_tax C) (sales_tax (selling_price_before_tax C)) = sale_price_incl_tax → 
  C = 482.76 :=
by
  intros,
  sorry

end cost_price_is_correct_l450_450444


namespace f_f_neg1_eq_3_l450_450744

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x + 2 else 1

theorem f_f_neg1_eq_3 : f (f (-1)) = 3 :=
by {
  -- We can leave the proof blank with "sorry" because the steps are not required.
  sorry
}

end f_f_neg1_eq_3_l450_450744


namespace find_a_range_l450_450641

noncomputable def f (a x : ℝ) : ℝ := log (a * x^2 - x + a)

theorem find_a_range (a : ℝ) (h : ∀ x : ℝ, a * x^2 - x + a > 0) : a > 1/2 :=
sorry

end find_a_range_l450_450641


namespace ellipse_focus_value_l450_450421

theorem ellipse_focus_value (k : ℝ) (hk : 5 * (0:ℝ)^2 - k * (2:ℝ)^2 = 5) : k = -1 :=
by
  sorry

end ellipse_focus_value_l450_450421


namespace smallest_possible_N_l450_450266

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l450_450266


namespace range_sin_squared_plus_sin_l450_450443

theorem range_sin_squared_plus_sin (x : ℝ) :
  ∃ y, y = (λ z, sin (3 * π / 2 - z) ^ 2 + sin (z + π)) x ∧ y ∈ set.Icc (-1 : ℝ) (5/4 : ℝ) :=
sorry

end range_sin_squared_plus_sin_l450_450443


namespace hyperbola_min_focal_asymptote_eq_l450_450836

theorem hyperbola_min_focal_asymptote_eq {x y m : ℝ}
  (h1 : -2 ≤ m)
  (h2 : m < 0)
  (h_eq : x^2 / m^2 - y^2 / (2 * m + 6) = 1)
  (h_min_focal : m = -1) :
  y = 2 * x ∨ y = -2 * x :=
by
  sorry

end hyperbola_min_focal_asymptote_eq_l450_450836


namespace probability_of_three_heads_in_eight_tosses_l450_450536

noncomputable def coin_toss_probability : ℚ :=
  let total_outcomes := 2^8
  let favorable_outcomes := Nat.choose 8 3
  favorable_outcomes / total_outcomes

theorem probability_of_three_heads_in_eight_tosses : coin_toss_probability = 7 / 32 :=
  by
  sorry

end probability_of_three_heads_in_eight_tosses_l450_450536


namespace conical_wheat_pile_weight_l450_450872

-- Define the given conditions
variables 
  (h : ℝ) (base_circumference : ℝ) (density : ℝ)

-- Define the calculated radius and volume
def radius : ℝ := base_circumference / (2 * Real.pi)
def volume : ℝ := (1/3) * Real.pi * (radius ^ 2) * h

-- Define the total weight by multiplying volume by density
def total_weight : ℝ := volume * density

-- Lean theorem statement ensuring the total weight is as calculated
theorem conical_wheat_pile_weight
  (h : ℝ := 1.5)
  (base_circumference : ℝ := 18.84)
  (density : ℝ := 750) :
  total_weight h base_circumference density = 10597.5 :=
by {
  -- skipping the proof steps
  sorry
}

end conical_wheat_pile_weight_l450_450872


namespace exists_quadratic_trinomial_has_2n_distinct_roots_l450_450578

theorem exists_quadratic_trinomial_has_2n_distinct_roots :
  ∃ (f : ℝ → ℝ) (p : polynomial ℝ), p.degree = 2 →
  (∀ n : ℕ, ∃ (D : set ℝ), (∀ x ∈ D, x ∈ (-1, 1)) ∧ (D.card = 2 * n) ∧ (∀ x ∈ D, p.eval x = 0)) :=
sorry

end exists_quadratic_trinomial_has_2n_distinct_roots_l450_450578


namespace choose_student_B_l450_450793

-- Define the scores for students A and B
def scores_A : List ℕ := [72, 85, 86, 90, 92]
def scores_B : List ℕ := [76, 83, 85, 87, 94]

-- Function to calculate the average of scores
def average (scores : List ℕ) : ℚ :=
  scores.sum / scores.length

-- Function to calculate the variance of scores
def variance (scores : List ℕ) : ℚ :=
  let mean := average scores
  (scores.map (λ x => (x - mean) * (x - mean))).sum / scores.length

-- Calculate the average scores for A and B
def avg_A : ℚ := average scores_A
def avg_B : ℚ := average scores_B

-- Calculate the variances for A and B
def var_A : ℚ := variance scores_A
def var_B : ℚ := variance scores_B

-- The theorem to be proved
theorem choose_student_B : var_B < var_A :=
  by sorry

end choose_student_B_l450_450793


namespace range_of_p_l450_450741

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def greatestPrimeFactor (n : ℕ) : ℕ := sorry -- Placeholder for greatest prime factor function

noncomputable def p (x : ℕ) : ℕ :=
if 2 ≤ x ∧ x ≤ 15 then
  if isPrime (nat.floor x) then x + 2
  else if isPerfectSquare (nat.floor x) then x * x
  else p (greatestPrimeFactor (nat.floor x)) + (x + 1 - nat.floor x)
else 0

theorem range_of_p : set.range p = { n | (4 ≤ n ∧ n ≤ 17) ∨ n = 81 } :=
sorry

end range_of_p_l450_450741


namespace isosceles_triangle_in_15gon_l450_450393

theorem isosceles_triangle_in_15gon (polygon : fin 15 → Prop) 
  (h_reg : ∀ i j, polygon i → polygon j → (dist i j < dist (i+1) j + dist (i+1) (j+1))) :
  ∀ (vertices : fin 15 → Prop), 
  (∀ v, vertices v → polygon v) → ∃ (v1 v2 v3 : fin 15), vertices v1 ∧ vertices v2 ∧ vertices v3 ∧ (dist v1 v2 = dist v1 v3) :=
sorry

end isosceles_triangle_in_15gon_l450_450393


namespace exists_APredicting_function_l450_450781

def is_APredicting (f : Set ℕ → ℕ) (A : Set ℕ) : Prop :=
  { x ∈ ℕ | x ∉ A ∧ f (A ∪ {x}) ≠ x }.finite

theorem exists_APredicting_function :
  ∃ f : Set ℕ → ℕ, ∀ A : Set ℕ, is_APredicting f A :=
sorry

end exists_APredicting_function_l450_450781


namespace sum_of_possible_k_l450_450373

theorem sum_of_possible_k (a b c k : ℂ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h4 : a / (2 - b) = k) (h5 : b / (3 - c) = k) (h6 : c / (4 - a) = k) :
  k = 1 ∨ k = -1 ∨ k = -2 → k = 1 + (-1) + (-2) :=
by
  sorry

end sum_of_possible_k_l450_450373


namespace angle_between_a_b_is_120_degrees_l450_450634

variable (e1 e2 : ℝ^3) -- Representing vectors in 3D space for generality
variable (a : ℝ^3) := 2 • e1 + e2
variable (b : ℝ^3) := -3 • e1 + 2 • e2

-- Conditions: e1 and e2 are unit vectors and their dot product is 1/2
axiom unit_vector_e1 : ∥e1∥ = 1
axiom unit_vector_e2 : ∥e2∥ = 1
axiom angle_e1_e2 : (e1 ⬝ e2) = 1/2

-- The angle θ between a and b should be 120 degrees
theorem angle_between_a_b_is_120_degrees : 
  let θ := real.arccos ((a ⬝ b) / (∥a∥ * ∥b∥)) in 
  θ = real.pi * (2/3) := 
sorry

end angle_between_a_b_is_120_degrees_l450_450634


namespace intervals_of_increase_l450_450111

def f (x : ℝ) : ℝ := 2*x^3 - 6*x^2 + 7

theorem intervals_of_increase : 
  ∀ x : ℝ, (x < 0 ∨ x > 2) → (6*x^2 - 12*x > 0) :=
by
  -- Placeholder for proof
  sorry

end intervals_of_increase_l450_450111


namespace gold_exceeds_payment_l450_450338

/-- In a graph G with n vertices and e edges where no cycle exists of length 4 or more,
    prove that the total number of pieces of gold collected (3n) exceeds the number of pieces 
    paid out for each handshake (2e) by at least 3 units. -/
theorem gold_exceeds_payment (n e : ℕ) (h_graph : ∀ (G : SimpleGraph (Fin n)), 
  (∀ C : Finset (Fin n), C.card ≥ 4 → ¬ C.IsCycle) → e ≤ (3 * n - 3) / 2) : 
  3 * n - 2 * e ≥ 3 :=
by
  sorry

end gold_exceeds_payment_l450_450338


namespace distinct_arrangements_of_mouse_l450_450674

theorem distinct_arrangements_of_mouse : 
  let n := 5 in
  fact n = 120 :=
by
  sorry

end distinct_arrangements_of_mouse_l450_450674


namespace weight_computation_requires_initial_weight_l450_450279

-- Let's define the conditions
variable (initial_weight : ℕ) -- The initial weight of the pet; needs to be provided
def yearly_gain := 11  -- The pet gains 11 pounds each year
def age := 8  -- The pet is 8 years old

-- Define the goal to be proved
def current_weight_computable : Prop :=
  initial_weight ≠ 0 → initial_weight + (yearly_gain * age) ≠ 0

-- State the theorem
theorem weight_computation_requires_initial_weight : ¬ ∃ current_weight, initial_weight + (yearly_gain * age) = current_weight :=
by {
  sorry
}

end weight_computation_requires_initial_weight_l450_450279


namespace power_of_two_has_half_nines_l450_450394

theorem power_of_two_has_half_nines (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, (∃ m : ℕ, (k / 2 < m) ∧ 
            (10^k ∣ (2^n + m + 1)) ∧ 
            (2^n % (10^k) = 10^k - 1)) :=
sorry

end power_of_two_has_half_nines_l450_450394


namespace negation_of_proposition_l450_450431

theorem negation_of_proposition : (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 := by
  sorry

end negation_of_proposition_l450_450431


namespace truncated_cone_radius_l450_450291

theorem truncated_cone_radius (R: ℝ) (l: ℝ) (h: 0 < l)
  (h1 : ∃ (r: ℝ), r = (R + 5) / 2 ∧ (5 + r) = (1 / 2) * (R + r))
  : R = 25 :=
sorry

end truncated_cone_radius_l450_450291


namespace sum_circle_areas_l450_450216

def parabola (a b c : ℝ) := (x : ℝ) → a * x^2 + b * x + c 

def circle_area (r : ℝ) := π * r^2

def inscribed_circle_radius (a : ℝ) := 1 / |a|

def sequence_circle_area_sum (a : ℝ) (n : ℕ) := 
  ∑ k in List.range (n+1), circle_area (k/a)

theorem sum_circle_areas (a : ℝ) (b c : ℝ) (n : ℕ) (h : a ≠ 0) : 
  sequence_circle_area_sum a n = 
  π * n * (n + 1) * (2 * n + 1) / (6 * a^2) :=
by
  sorry

end sum_circle_areas_l450_450216


namespace sum_of_squares_gt_twice_area_l450_450137

-- Define the regular n-gon inscribed in a circle
variables {n : ℕ} (R : ℝ) (vertices : Fin n → Complex)

noncomputable def length_of_segments (k j : Fin n) : ℝ :=
  2 * R * (Complex.abs (Complex.sin ((π * (k.val - j.val : ℕ : ℝ)) / n)))

-- Sum of the squares of the lengths of the segments joining vertices
noncomputable def sum_of_squares (vertices : Fin n → Complex) : ℝ :=
  ∑ k j in Finset.range n, (length_of_segments R vertices k j) ^ 2

-- Area of the regular n-gon
noncomputable def area_of_polygon : ℝ :=
  (n : ℝ) * (R ^ 2) * Real.sin (2 * π / (n : ℝ)) / 2

-- Main theorem statement
theorem sum_of_squares_gt_twice_area
  (n : ℕ) (R : ℝ) (vertices : Fin n → Complex) :
  sum_of_squares R vertices > 2 * area_of_polygon R :=
by
  sorry

end sum_of_squares_gt_twice_area_l450_450137


namespace problem_solved_by_three_l450_450016

/--  There are 21 girls and 21 boys, each solving at most 6 problems,
and for every pair of girl and boy, there is at least one problem that both have solved.
The goal is to show that there exists at least one problem that at least 3 girls and at least 3 boys have solved. -/
theorem problem_solved_by_three : 
  ∃ (P : Type) (girls boys : Finset P) (solved_by : P → Finset P) (girl_solved : Finset (Finset P)) (boy_solved : Finset (Finset P)),
  (Finset.card girls = 21) ∧
  (Finset.card boys = 21) ∧
  (∀ p ∈ (girls ∪ boys), Finset.card (solved_by p) ≤ 6) ∧
  (∀ g ∈ girls, ∀ b ∈ boys, ∃ p ∈ (solved_by g ∩ solved_by b), true) ∧
  ∃ p, 2 < (Finset.card (solved_by p ∩ girls)) ∧ 2 < (Finset.card (solved_by p ∩ boys)) :=
begin
  sorry
end

end problem_solved_by_three_l450_450016


namespace julia_drove_214_miles_l450_450527

def daily_rate : ℝ := 29
def cost_per_mile : ℝ := 0.08
def total_cost : ℝ := 46.12

theorem julia_drove_214_miles :
  (total_cost - daily_rate) / cost_per_mile = 214 :=
by
  sorry

end julia_drove_214_miles_l450_450527


namespace median_of_100_numbers_l450_450309

theorem median_of_100_numbers (x : Fin 100 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i = 78 → x j = 66 → i = 51 ∧ j = 50 ∨ i = 50 ∧ j = 51)
  (h2 : ∀ i, i ≠ 51 → x 51 = 78)
  (h3 : ∀ i, i ≠ 50 → x 50 = 66) :
  (x 50 + x 51) / 2 = 72 :=
by sorry

end median_of_100_numbers_l450_450309


namespace tangent_circles_radii_l450_450853

noncomputable def radii_of_tangent_circles (R r : ℝ) (h : R > r) : Set ℝ :=
  { x | x = (R * r) / ((Real.sqrt R + Real.sqrt r)^2) ∨ x = (R * r) / ((Real.sqrt R - Real.sqrt r)^2) }

theorem tangent_circles_radii (R r : ℝ) (h : R > r) :
  ∃ x, x ∈ radii_of_tangent_circles R r h := sorry

end tangent_circles_radii_l450_450853


namespace sequence_bound_l450_450039

def sequence (a : ℕ → ℝ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 1, a (n + 1) = a n + 1 / ⌊a n⌋

theorem sequence_bound (a : ℕ → ℝ) (n : ℕ) 
  (h_seq : sequence a) :
  a n > 20 ↔ n > 191 :=
sorry

end sequence_bound_l450_450039


namespace inequality_ineq_l450_450148

variable (x y z : Real)

theorem inequality_ineq {x y z : Real} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = 3) :
  (1 / (x^5 - x^2 + 3)) + (1 / (y^5 - y^2 + 3)) + (1 / (z^5 - z^2 + 3)) ≤ 1 :=
by 
  sorry

end inequality_ineq_l450_450148


namespace proof_problem_l450_450366

variables {n : ℕ} {a : Fin n → ℝ}

-- Condition: a sequence of real numbers in non-increasing order
def non_increasing_seq (a : Fin n → ℝ) := ∀ i j, i < j → a i ≥ a j

-- Condition: the sum of powers is non-negative for all k > 0
def sum_of_powers_nonnegative (a : Fin n → ℝ) := ∀ k > 0, (∑ i, (a i) ^ k) ≥ 0

-- Definition for the maximum absolute value
def max_abs_value (a : Fin n → ℝ) : ℝ := Finset.sup Finset.univ (λ i, |a i|)

-- The proof statement
theorem proof_problem
  (h1 : non_increasing_seq a)
  (h2 : sum_of_powers_nonnegative a) :
  max_abs_value a = a 0 ∧ ∀ x > a 0, (∏ i, (x - a i)) ≤ x ^ n - (a 0) ^ n := 
sorry

end proof_problem_l450_450366


namespace perfect_shuffle_cycle_l450_450575

theorem perfect_shuffle_cycle (n : ℕ) (h : 0 < n) : 
  ∃ k : ℕ, cycle_n_shuffles 2n k = 1 :=
sorry

-- Additional definitions that may be required for clarity:
def perfect_shuffle (lst : List ℕ) : List ℕ :=
  let n := lst.length / 2
  List.zipWith (· + ·) (lst.drop n) (lst.take n)

def cycle_n_shuffles (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then 1
  else (2^k) % (2 * n + 1)

end perfect_shuffle_cycle_l450_450575


namespace number_of_bad_arrangements_l450_450439

def is_bad_arrangement (arrangement : List ℕ) : Prop :=
  ∃ n ∈ (Finset.range 17).erase 0, ¬ ∃ (l : List ℕ), l ≠ [] ∧
  l.sum = n ∧ (@List.cyclicPermutationOf arrangement l)

def distinct_bad_arrangements (arrangements : Finset (List ℕ)) : Finset (List ℕ) :=
  arrangements.filter is_bad_arrangement

theorem number_of_bad_arrangements : distinct_bad_arrangements (Finset.of_list [ [1, 2, 3, 4, 6],
                                                      -- Add all distinct permutations here
                                                      -- after considering rotations and reflections
                                                     ]) = 3 :=
sorry

end number_of_bad_arrangements_l450_450439


namespace tangential_circles_internal_tangent_equality_l450_450459

theorem tangential_circles_internal_tangent_equality
  (ω₁ ω₂ : Circle)
  (P A B : Point)
  (C D : Point)
  (h_tangent : ω₁.isTangentInternal ω₂ P)
  (h_common_tangent : collinear [A, P, B])
  (h_P_between : P ∈ lineSegment A B)
  (h_intersections : 
    ∃ a₁ b₁ a₂ b₂ : Line, 
      isTangent A a₁ ω₁ ∧ isTangent B b₁ ω₁ ∧ 
      isTangent A a₂ ω₂ ∧ isTangent B b₂ ω₂ ∧ 
      intersection a₁ b₂ = C ∧ intersection a₂ b₁ = D):
  dist C A + dist C B = dist D A + dist D B :=
by sorry

end tangential_circles_internal_tangent_equality_l450_450459


namespace max_amount_paul_received_l450_450761

theorem max_amount_paul_received :
  ∃ (numBplus numA numAplus : ℕ),
  (numBplus + numA + numAplus = 10) ∧ 
  (numAplus ≥ 2 → 
    let BplusReward := 5;
    let AReward := 2 * BplusReward;
    let AplusReward := 15;
    let Total := numAplus * AplusReward + numA * (2 * AReward) + numBplus * (2 * BplusReward);
    Total = 190
  ) :=
sorry

end max_amount_paul_received_l450_450761


namespace part1_part2_l450_450214

noncomputable def f (g : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ :=
  log (g x) / log 2 + (k - 1) * x

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem part1 (g : ℝ → ℝ) (k : ℝ)
  (h1 : ∀ x : ℝ, g (log x / log 2) = x + 1)
  (h2 : is_even_function (f g k)) :
  k = 1 / 2 :=
sorry

theorem part2 (a : ℝ)
  (h1 : ∀ x : ℝ, f (λ x, a * x^2 + (a + 1) * x + a) 1 x ∈ set.univ) :
  0 ≤ a ∧ a ≤ 1 :=
sorry

end part1_part2_l450_450214


namespace jim_catches_bob_in_20_minutes_l450_450902

theorem jim_catches_bob_in_20_minutes
  (bob_speed : ℝ)
  (jim_speed : ℝ)
  (bob_head_start : ℝ)
  (bob_speed_mph : bob_speed = 6)
  (jim_speed_mph : jim_speed = 9)
  (bob_headstart_miles : bob_head_start = 1) :
  ∃ (m : ℝ), m = 20 := 
by
  sorry

end jim_catches_bob_in_20_minutes_l450_450902


namespace sum_of_integers_is_18_l450_450406

theorem sum_of_integers_is_18 (a b c d : ℕ) 
  (h1 : a * b + c * d = 38)
  (h2 : a * c + b * d = 34)
  (h3 : a * d + b * c = 43) : 
  a + b + c + d = 18 := 
  sorry

end sum_of_integers_is_18_l450_450406


namespace choose_club_officers_l450_450389

theorem choose_club_officers :
  ∃ (members boys girls: ℕ), 
    members = 25 ∧ boys = 12 ∧ girls = 13 ∧
    (∃ (president_ways vice_president_ways treasurer_ways : ℕ),
      president_ways = girls ∧ 
      vice_president_ways = boys ∧ 
      treasurer_ways = boys - 1 ∧ 
      president_ways * vice_president_ways * treasurer_ways = 1716) :=
by
    use 25, 12, 13
    existsi 13, 12, 11
    simp
    sorry

end choose_club_officers_l450_450389


namespace volume_of_given_pyramid_is_one_l450_450967

-- Define a pyramid with a given height and square base
structure Pyramid :=
  (height : ℝ)
  (side_length : ℝ)

-- Define the conditions given in the problem
def given_pyramid : Pyramid :=
  { height := 3, side_length := 1 }

-- Define the base area of the pyramid
def base_area (p : Pyramid) : ℝ :=
  p.side_length ^ 2

-- Define the volume formula for a pyramid
def volume (p : Pyramid) : ℝ :=
  1 / 3 * base_area p * p.height

-- State the theorem to prove
theorem volume_of_given_pyramid_is_one : volume given_pyramid = 1 :=
by
  sorry

end volume_of_given_pyramid_is_one_l450_450967


namespace era_slices_burger_l450_450582

theorem era_slices_burger (slices_per_burger : ℕ) (h : 5 * slices_per_burger = 10) : slices_per_burger = 2 :=
by 
  sorry

end era_slices_burger_l450_450582


namespace proposition_d_l450_450962

-- Defining lines as sets of points and planes as sets of lines
variables {Line Plane : Type}
variables {m n : Line} {α β γ : Plane}

-- Assumptions
variable (L1 : α ≠ β)
variable (L2 : m ≠ n)
variable (P1 : α ≠ γ)
variable (P2 : β ≠ γ)

-- Propositions
def parallel_lines (l1 l2 : Line) : Prop := ∀ (p1 p2 ∈ l1) (p3 p4 ∈ l2), p1 ≠ p2 → p3 ≠ p4 → ∃ (v : Vector), v ∈ l1 ∧ v ∈ l2
def parallel_planes (π1 π2 : Plane) : Prop := ∃ (v : Vector), ∀ (l ∈ π1) (m ∈ π2), parallel_lines l m
def line_plane_intersection (l : Line) (π : Plane) (p q : Point) : Prop := ∀ (l1 ∈ π) (p ∈ l), l1 ∩ l = l

-- The correct proposition D
theorem proposition_d (h1 : parallel_planes α β) (h2 : line_plane_intersection m α γ) (h3 : line_plane_intersection n β γ) : 
  parallel_lines m n :=
sorry

end proposition_d_l450_450962


namespace number_of_correct_statements_l450_450923

noncomputable def f (x: ℝ) : ℝ := sorry

axiom f_def {m n : ℝ} (hm : m > 0) (hn : n > 0) : f (m * n) = f m + f n
axiom f_pos {x : ℝ} (hx : x > 1) : f x > 0

theorem number_of_correct_statements :
  ( (f 1 = 0) ∧ 
    (∀ (m n : ℝ), m > 0 → n > 0 → f (m / n) = f m - f n) ∧ 
    (f 2 = 1 → ∀ x, 0 < x ∧ x < 2/7 → f(x+2) - f(2*x) > 2) ∧ 
    ¬(∀ x, x > 0 → f x < f (x + 1)) ∧ 
    (∀ m n, m > 0 → n > 0 → f ((m + n) / 2) ≥ (f m + f n) / 2) 
  ) 
→ 4 := 
sorry

end number_of_correct_statements_l450_450923


namespace shaded_area_is_correct_l450_450710

-- We define constants for the radii and positions
constant r_big : ℝ := 10
constant r_small : ℝ := 5
constant area_big : ℝ := real.pi * r_big^2
constant area_small : ℝ := real.pi * r_small^2

-- Definition of the area of the shaded region
def area_shaded : ℝ := area_big - 2 * area_small

-- The theorem stating the area of the shaded region
theorem shaded_area_is_correct : area_shaded = 50 * real.pi := 
by
  -- Skipping the proof steps with sorry
  sorry

end shaded_area_is_correct_l450_450710


namespace coordinates_with_respect_to_origin_l450_450416

theorem coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (3, -2)) : (x, y) = (3, -2) :=
by
  -- Given that the point (x, y) is (3, -2)
  rw h

end coordinates_with_respect_to_origin_l450_450416


namespace smallest_positive_integer_mod_l450_450835

theorem smallest_positive_integer_mod (a : ℕ) (h1 : a ≡ 4 [MOD 5]) (h2 : a ≡ 6 [MOD 7]) : a = 34 :=
by
  sorry

end smallest_positive_integer_mod_l450_450835


namespace unique_nonzero_b_l450_450680

variable (a b m n : ℝ)
variable (h_ne : m ≠ n)
variable (h_m_nonzero : m ≠ 0)
variable (h_n_nonzero : n ≠ 0)

theorem unique_nonzero_b (h : (a * m + b * n + m)^2 - (a * m + b * n + n)^2 = (m - n)^2) : 
  a = 0 ∧ b = -1 :=
sorry

end unique_nonzero_b_l450_450680


namespace ray_steps_problem_l450_450767

theorem ray_steps_problem : ∃ n, n > 15 ∧ n % 3 = 2 ∧ n % 7 = 1 ∧ n % 4 = 3 ∧ n = 71 :=
by
  sorry

end ray_steps_problem_l450_450767


namespace fourth_root_eq_solution_l450_450949

theorem fourth_root_eq_solution (x : ℝ) (h : Real.sqrt (Real.sqrt x) = 16 / (8 - Real.sqrt (Real.sqrt x))) : x = 256 := by
  sorry

end fourth_root_eq_solution_l450_450949


namespace smallest_N_l450_450270

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450270


namespace complete_the_square_l450_450490

theorem complete_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) :=
by
  intro h
  sorry

end complete_the_square_l450_450490


namespace min_ratio_l450_450743

theorem min_ratio (x y : ℕ) (hx1 : 10 ≤ x) (hx2 : x ≤ 99) (hy1 : 10 ≤ y) (hy2 : y ≤ 99)
  (hxy : x + y = 150) : 
  (let r := (x : ℚ) / (3 * y + 4 : ℚ) in r) = 70 / 17 :=
by
  sorry

end min_ratio_l450_450743


namespace smallest_N_l450_450269

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450269


namespace median_of_100_numbers_l450_450325

theorem median_of_100_numbers (numbers : List ℝ) (h_length : numbers.length = 100)
  (h_sorted : numbers.sorted (≤))
  (h_51 : numbers.nth_le 50 h_51_nat = 78) -- Note: nth_le is zero-indexed, so 51st element is 50th index.
  (h_50 : numbers.nth_le 49 h_50_nat = 66) : 
  (numbers.nth_le 49 h_50_nat + numbers.nth_le 50 h_51_nat) / 2 = 72 :=
sorry

end median_of_100_numbers_l450_450325


namespace price_of_turban_l450_450511

theorem price_of_turban (T : ℝ) (h1 : ∀ (T : ℝ), 3 / 4 * (90 + T) = 40 + T) : T = 110 :=
by
  sorry

end price_of_turban_l450_450511


namespace value_of_7_l450_450003

theorem value_of_7'_prime :
  (let q_prime : ℕ → ℕ := λ q, 3 * q - 3 in q_prime(q_prime 7) = 51) :=
by
  sorry

end value_of_7_l450_450003


namespace quartic_polynomial_value_l450_450809

theorem quartic_polynomial_value
  (P : ℝ → ℝ)
  (h_poly : ∀ x, polynomial.degree P = 4)
  (h1 : P 1 = 0)
  (h_max_2 : ∀ x, x = 2 → P x = 3)
  (h_max_3 : ∀ x, x = 3 → P x = 3) :
  P 5 = -24 :=
sorry

end quartic_polynomial_value_l450_450809


namespace solve_y_l450_450401

theorem solve_y (y : ℝ) (h : y > 0) : is_arithmetic_sequence (2^2) (y^2) (4^2) → y = Real.sqrt 10 :=
by
  intros h_arith
  sorry

end solve_y_l450_450401


namespace unique_two_digit_number_l450_450451

theorem unique_two_digit_number (x y : ℕ) (h1 : 10 ≤ 10 * x + y ∧ 10 * x + y < 100) (h2 : 3 * y = 2 * x) (h3 : y + 3 = x) : 10 * x + y = 63 :=
by
  sorry

end unique_two_digit_number_l450_450451


namespace problem1_problem2_l450_450566

-- Definitions of the geometric constructs and assumptions
variables {O1 O2 : Type} [circle O1] [circle O2]
variables (A B C D E F : point)
variables (BC : chord O1 B C) (BD : chord O2 B D)

-- The conditions from the problem
def conditions1 := 
  ∃ (BC_chord : chord O1 B C) (BD_chord : chord O2 B D) 
  (intersect1 : intersects BC BD = B) (intersect2 : intersects BC O2 = E) 
  (intersect3 : intersects BD O1 = F),
  angle (B D A) = angle (B C A)

-- The statement to be proven for Problem 1
theorem problem1 
  (h : conditions1 BC BD A B C D E F) :
  (dist D F) = (dist C E) :=
  sorry

-- The conditions for Problem 2
def conditions2 :=
  ∃ (DF_eq_CE : dist D F = dist C E), true

-- The statement to be proven for Problem 2
theorem problem2 
  (h : conditions2 C D F E) :
  (angle (B D A) = angle (B C A)) :=
  sorry

end problem1_problem2_l450_450566


namespace minor_premise_wrong_l450_450010

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = f x

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := x^2 + x

theorem minor_premise_wrong : ¬ is_even_function f ∧ ¬ is_odd_function f := 
by
  sorry

end minor_premise_wrong_l450_450010


namespace smallest_possible_N_l450_450262

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l450_450262


namespace sphere_radius_l450_450457

-- Given conditions
variables (x r : ℝ)
variable spheres_packed_in_box : Prop
variable length_ratio : 2
variable width_ratio : 2
variable height_ratio : 1
variable sphere_tangent_to_others_and_box : Prop
variable center_sphere_centered : C.real_of_rat 2[box.Center]

-- The goal is to prove that the radius of each sphere is x/4
theorem sphere_radius (x : ℝ) (ratio_condition : 2 * x = 2 * x ∨ 2 * x = 1) 
  (spheres_packed_in_box : Prop) : 
  (∃ r : ℝ, r = x / 4) :=
sorry

end sphere_radius_l450_450457


namespace net_wealth_increase_l450_450019

variable (S R : ℝ)

-- Define the initial investments
def initial_stock := S
def initial_real_estate := R

-- Year 1 changes
def stock_after_year1 := 1.5 * initial_stock
def realestate_after_year1 := 1.2 * initial_real_estate

-- Year 2 changes
def stock_after_year2 := (0.7 * stock_after_year1)
def realestate_after_year2 := (1.1 * realestate_after_year1)

-- Year 3 adjustments
def additional_stock_investment := 0.5 * initial_stock
def withdrawal_real_estate := 0.2 * initial_real_estate

def stock_after_adjustments := stock_after_year2 + additional_stock_investment
def realestate_after_adjustments := realestate_after_year2 - withdrawal_real_estate

-- Year 3 percentage changes
def stock_final := 1.25 * stock_after_adjustments
def realestate_final := 0.95 * realestate_after_adjustments

-- Final total wealth
def final_wealth := stock_final + realestate_final

-- Initial total wealth
def initial_wealth := initial_stock + initial_real_estate

-- Net change in wealth
def net_change := final_wealth - initial_wealth

theorem net_wealth_increase :
  net_change = 0.9375 * S + 0.064 * R :=
by
  sorry

end net_wealth_increase_l450_450019


namespace area_of_centroid_quadrilateral_l450_450776

noncomputable def square_side_length : ℝ := 40
noncomputable def point_Q_E_distance : ℝ := 15
noncomputable def point_Q_F_distance : ℝ := 35
noncomputable def quad_area : ℝ := 3200 / 9

-- Definitions for vertices, points, and centroids in a square configuration
structure Point :=
(x : ℝ)
(y : ℝ)

def E : Point := { x := 0, y := 0 }
def F : Point := { x := square_side_length, y := 0 }
def G : Point := { x := square_side_length, y := square_side_length }
def H : Point := { x := 0, y := square_side_length }
def Q : Point := { x := _, y := _ } -- Coordinates of Q need to be determined based on distance constraints

-- TODO: Proper positioning for Q based on given distances (complex in Euclidean space)

def centroid (A B C : Point) : Point := 
  { x := (A.x + B.x + C.x) / 3,
    y := (A.y + B.y + C.y) / 3 }

def centroid_quad_area : ℝ := 
  let cent_EFQ := centroid E F Q
  let cent_FGQ := centroid F G Q
  let cent_GHQ := centroid G H Q
  let cent_HEQ := centroid H E Q
  -- Assuming the positions of centroids form the vertices of a quadrilateral
  -- Calculate the area of this quadrilateral
  sorry -- Complex geometric calculations skipped

theorem area_of_centroid_quadrilateral : 
  centroid_quad_area = quad_area :=
by
  sorry

end area_of_centroid_quadrilateral_l450_450776


namespace circular_garden_area_l450_450529

open Real

theorem circular_garden_area (r : ℝ) (h₁ : r = 8)
      (h₂ : 2 * π * r = (1 / 4) * π * r ^ 2) :
  π * r ^ 2 = 64 * π :=
by
  -- The proof will go here
  sorry

end circular_garden_area_l450_450529


namespace find_triples_l450_450126

theorem find_triples (a b c : ℝ) :
  a^2 + 2*b^2 - 2*b*c = 16 ∧ 2*a*b - c^2 = 16 →
  (a = 4 ∧ b = 4 ∧ c = 4) ∨ (a = -4 ∧ b = -4 ∧ c = -4) :=
sory

end find_triples_l450_450126


namespace third_factor_of_product_l450_450868

theorem third_factor_of_product (w : ℕ) (h_w_pos : w > 0) (h_w_168 : w = 168)
  (w_factors : (936 * w) = 2^5 * 3^3 * x)
  (h36_factors : 2^5 ∣ (936 * w)) (h33_factors : 3^3 ∣ (936 * w)) : 
  (936 * w) / (2^5 * 3^3) = 182 :=
by {
  -- This is a placeholder. The actual proof is omitted.
  sorry
}

end third_factor_of_product_l450_450868


namespace sequence_is_arithmetic_progression_l450_450449

noncomputable def base60_to_decimal (b60 : ℕ × ℕ) : ℕ :=
  b60.1 * 60 + b60.2

def sequence_base60 : List (ℕ × ℕ) := [
  (1, 36), (1, 52), (2, 8), (2, 24), (2, 40), (2, 56), (3, 12), (3, 28), (3, 44), (4, 0)
]

def sequence_decimal : List ℕ := sequence_base60.map base60_to_decimal

def common_difference (seq : List ℕ) : Option ℕ :=
  if seq.length < 2 then none 
  else some (seq[1] - seq.head!)

theorem sequence_is_arithmetic_progression :
  sequence_decimal.erase 4 = [96, 112, 128, 144, 160, 176, 192, 208, 224] ∧ 
  common_difference (sequence_decimal.erase 4) = some 16 :=
by
  sorry

end sequence_is_arithmetic_progression_l450_450449


namespace find_k_l450_450878

-- Define the function f as described in the problem statement
def f (n : ℕ) : ℕ := 
  if n % 2 = 1 then 
    n + 3 
  else 
    n / 2

theorem find_k (k : ℕ) (h_odd : k % 2 = 1) : f (f (f k)) = k → k = 1 :=
by {
  sorry
}

end find_k_l450_450878


namespace sum_first_7_terms_eq_105_l450_450626

variable {a : ℕ → ℤ}

-- Definitions from conditions.
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a)

def a_4_eq_15 : a 4 = 15 := sorry

-- Sum definition specific for 7 terms of an arithmetic sequence.
def sum_first_7_terms (a : ℕ → ℤ) : ℤ := (7 / 2 : ℤ) * (a 1 + a 7)

-- The theorem to prove.
theorem sum_first_7_terms_eq_105 
    (arith_seq : is_arithmetic_sequence a) 
    (a4 : a 4 = 15) : 
  sum_first_7_terms a = 105 := 
sorry

end sum_first_7_terms_eq_105_l450_450626


namespace find_point_M_l450_450619

def parabola (x y : ℝ) := x^2 = 4 * y
def focus_dist (M : ℝ × ℝ) := dist M (0, 1) = 2
def point_on_parabola (M : ℝ × ℝ) := parabola M.1 M.2

theorem find_point_M (M : ℝ × ℝ) (h1 : point_on_parabola M) (h2 : focus_dist M) :
  M = (2, 1) ∨ M = (-2, 1) := by
  sorry

end find_point_M_l450_450619


namespace winner_vote_percentage_l450_450336

theorem winner_vote_percentage (total_votes winner_votes other_votes : ℕ)
    (h1 : winner_votes = 1054)
    (h2 : winner_votes = other_votes + 408)
    (h3 : total_votes = winner_votes + other_votes) :
    (winner_votes : ℝ) / (total_votes : ℝ) * 100 ≈ 61.94 := 
sorry

end winner_vote_percentage_l450_450336


namespace compute_expression_l450_450916

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end compute_expression_l450_450916


namespace totalCostOfFencing_l450_450682

def numberOfSides : ℕ := 4
def costPerSide : ℕ := 79

theorem totalCostOfFencing (n : ℕ) (c : ℕ) (hn : n = numberOfSides) (hc : c = costPerSide) : n * c = 316 :=
by 
  rw [hn, hc]
  exact rfl

end totalCostOfFencing_l450_450682


namespace median_of_100_numbers_l450_450306

theorem median_of_100_numbers (x : Fin 100 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i = 78 → x j = 66 → i = 51 ∧ j = 50 ∨ i = 50 ∧ j = 51)
  (h2 : ∀ i, i ≠ 51 → x 51 = 78)
  (h3 : ∀ i, i ≠ 50 → x 50 = 66) :
  (x 50 + x 51) / 2 = 72 :=
by sorry

end median_of_100_numbers_l450_450306


namespace Lisa_earns_15_more_than_Tommy_l450_450375

variables (total_earnings Lisa_earnings Tommy_earnings : ℝ)

-- Conditions
def condition1 := total_earnings = 60
def condition2 := Lisa_earnings = total_earnings / 2
def condition3 := Tommy_earnings = Lisa_earnings / 2

-- Theorem to prove
theorem Lisa_earns_15_more_than_Tommy (h1: condition1) (h2: condition2) (h3: condition3) : 
  Lisa_earnings - Tommy_earnings = 15 :=
sorry

end Lisa_earns_15_more_than_Tommy_l450_450375


namespace profit_percentage_A_l450_450038

def CP_A : ℝ := 150
def SP_B : ℝ := 225
def profit_B : ℝ := 0.25

-- Let CP_B be the cost price of the bicycle for B, which is also SP_A (selling price of A to B).
def CP_B : ℝ := SP_B / (1 + profit_B)
def SP_A : ℝ := CP_B

def P_A : ℝ := ((SP_A - CP_A) / CP_A) * 100

theorem profit_percentage_A :
  CP_A = 150 ∧ SP_B = 225 ∧ profit_B = 0.25 → P_A = 20 := by
  intro h
  have h1 : CP_B = 180 := by sorry
  have h2 : SP_A = 180 := h1
  have h3 : P_A = 20 := by sorry
  exact h3

end profit_percentage_A_l450_450038


namespace triangle_third_side_length_l450_450199

theorem triangle_third_side_length (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 8) :
  (a + b > c) → (a + c > b) → (b + c > a) → c = 6 :=
by
  intros h₃ h₄ h₅
  rw [h₁, h₂] at *
  -- Simplified inequalities from the solution step
  have h₆ : 5 + 8 > c := by rw [h₁, h₂]; exact h₃
  have h₇ : 8 + c > 5 := by rw h₂; exact h₄
  have h₈ : 5 + c > 8 := by rw h₁; exact h₅
  sorry

end triangle_third_side_length_l450_450199


namespace max_AB_CD_value_l450_450746

def is_digit (x : ℕ) : Prop := x ≥ 1 ∧ x ≤ 9

noncomputable def max_AB_CD : ℕ :=
  let A := 9
  let B := 8
  let C := 7
  let D := 6
  (A + B) + (C + D)

theorem max_AB_CD_value :
  ∀ (A B C D : ℕ), 
    is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧ 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
    (A + B) + (C + D) ≤ max_AB_CD :=
by
  sorry

end max_AB_CD_value_l450_450746


namespace centrally_symmetric_implies_congruent_l450_450824

-- Definitions according to the problem conditions
def is_centrally_symmetric (shape1 shape2 : Type) (center : Type) : Prop :=
  ∃ p : center, (rotate shape1 p 180 = shape2 ∨ rotate shape2 p 180 = shape1)

-- Lean theorem statement
theorem centrally_symmetric_implies_congruent (shape1 shape2 : Type) (center : Type) :
  is_centrally_symmetric shape1 shape2 center → congruent shape1 shape2 :=
sorry

end centrally_symmetric_implies_congruent_l450_450824


namespace negation_of_p_l450_450631

variable {f : ℝ → ℝ}

def proposition_p : Prop :=
  ∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) ≥ 0

theorem negation_of_p :
  ¬proposition_p ↔ ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
by
  sorry

end negation_of_p_l450_450631


namespace problem1_problem2_l450_450993

-- Definitions for first problem
def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Theorem for first problem
theorem problem1 (f : ℝ → ℝ) (h1 : increasing_function f) (h2 : ∀ x, -3 ≤ x → x ≤ 3) (h : f (m + 1) > f (2 * m - 1)) :
  -1 ≤ m ∧ m < 2 :=
sorry

-- Definitions for second problem
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem for second problem
theorem problem2 (f : ℝ → ℝ) (h1 : increasing_function f) (h2 : odd_function f) (h3 : f 2 = 1) (h4 : ∀ x, -3 ≤ x → x ≤ 3) :
  ∀ x, f (x + 1) + 1 > 0 ↔ -3 < x ∧ x ≤ 2 :=
sorry

end problem1_problem2_l450_450993


namespace sum_geometric_sequence_l450_450151

theorem sum_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * a 1)
  (h_a1 : a 1 = 1)
  (h_arithmetic : 4 * a 2 + a 4 = 2 * a 3) : 
  a 2 + a 3 + a 4 = 14 :=
sorry

end sum_geometric_sequence_l450_450151


namespace area_enclosed_by_equation_l450_450830

theorem area_enclosed_by_equation :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 10 * y = -20) → (∃ r : ℝ, r^2 = 9 ∧ ∃ c : ℝ × ℝ, (∃ a b, (x - a)^2 + (y - b)^2 = r^2)) :=
by
  sorry

end area_enclosed_by_equation_l450_450830


namespace problem_statement_l450_450677

theorem problem_statement (a : ℝ) (h : (a + 1/a)^2 = 12) : a^3 + 1/a^3 = 18 * Real.sqrt 3 :=
by
  -- We'll skip the proof as per instruction
  sorry

end problem_statement_l450_450677


namespace dartboard_partition_count_l450_450558

-- number of darts and dartboards
def num_darts := 5
def num_boards := 5

-- definition of partition problem
def dart_partitions (n k : ℕ) : Set (List ℕ) :=
  {l | l.sum = n ∧ l.length ≤ k ∧ l.all (λ x, x ≥ 0)}

-- theorem statement
theorem dartboard_partition_count : 
  (Set.card (dart_partitions num_darts num_boards)) = 7 :=
sorry

end dartboard_partition_count_l450_450558


namespace B_can_complete_work_in_6_days_l450_450842

theorem B_can_complete_work_in_6_days (A B : ℝ) (h1 : (A + B) = 1 / 4) (h2 : A = 1 / 12) : B = 1 / 6 := 
by
  sorry

end B_can_complete_work_in_6_days_l450_450842


namespace no_natural_n_for_perfect_square_l450_450947

theorem no_natural_n_for_perfect_square :
  ¬ ∃ n : ℕ, ∃ k : ℕ, 2007 + 4^n = k^2 :=
by {
  sorry  -- Proof omitted
}

end no_natural_n_for_perfect_square_l450_450947


namespace rational_cos_ext_l450_450727

noncomputable theory
open Real

def rational_cos_seq (x : ℝ) (k : ℕ) : Prop :=
  (cos ((k - 1) * x) ∈ ℚ) ∧ (cos (k * x) ∈ ℚ)

theorem rational_cos_ext (k : ℕ) (x : ℝ) (hk : k ≥ 3) :
  rational_cos_seq x k → ∃ n : ℕ, n > k ∧ rational_cos_seq x n := 
sorry

end rational_cos_ext_l450_450727


namespace degree_of_polynomial_l450_450494

-- Define the polynomial
noncomputable def P (x : ℝ) : ℝ := 7 + 3 * x^5 + 150 + 5 * real.pi * x^3 + 2 * real.sqrt 3 * x^4 + 12 * x^(1/2) + 3

-- Statement to prove degree of the polynomial
theorem degree_of_polynomial : polynomial.degree (P x) = 5 := 
sorry

end degree_of_polynomial_l450_450494


namespace log_identity_l450_450569

theorem log_identity : log 5 0.25 + 2 * log 5 10 = 2 := by
  sorry

end log_identity_l450_450569


namespace value_of_a2_l450_450971

theorem value_of_a2 (a : ℕ → ℤ) (h1 : ∀ n : ℕ, a (n + 1) = a n + 2)
  (h2 : ∃ r : ℤ, a 3 = r * a 1 ∧ a 4 = r * a 3) :
  a 2 = -6 :=
by
  sorry

end value_of_a2_l450_450971


namespace remainder_when_s_div_6_is_5_l450_450805

theorem remainder_when_s_div_6_is_5 (s t : ℕ) (h1 : s > t) (Rs Rt : ℕ) (h2 : s % 6 = Rs) (h3 : t % 6 = Rt) (h4 : (s - t) % 6 = 5) : Rs = 5 := 
by
  sorry

end remainder_when_s_div_6_is_5_l450_450805


namespace geometric_sequence_sum_l450_450154

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℕ) (q : ℕ), 
    a 1 = 1 ∧ 
    (∀ n, a n = q^(n-1)) ∧ 
    (4 * a 2, 2 * a 3, a 4 form arithmetic_sequence) ∧
    (a 2 + a 3 + a 4 = 14) := 
begin
  sorry
end

end geometric_sequence_sum_l450_450154


namespace star_running_back_yardage_l450_450005

-- Definitions
def total_yardage : ℕ := 150
def catching_passes_yardage : ℕ := 60
def running_yardage (total_yardage catching_passes_yardage : ℕ) : ℕ :=
  total_yardage - catching_passes_yardage

-- Statement to prove
theorem star_running_back_yardage :
  running_yardage total_yardage catching_passes_yardage = 90 := 
sorry

end star_running_back_yardage_l450_450005


namespace volume_of_truncated_pyramid_l450_450885

-- Define the necessary geometrical and trigonometric properties

def volume_of_truncated_pyramid_in_sphere (R : ℝ) : ℝ :=
  let lower_base_area := (3 * R^2 * real.sqrt 3) / 2
  let upper_base_area := lower_base_area / 4
  let height := (R * real.sqrt 3) / 2
  (1 / 3) * height * (lower_base_area + upper_base_area + real.sqrt (lower_base_area * upper_base_area))

-- Lean statement
theorem volume_of_truncated_pyramid (R : ℝ) (hR : 0 < R) :
  volume_of_truncated_pyramid_in_sphere R = (21 * R^3) / 16 :=
sorry

end volume_of_truncated_pyramid_l450_450885


namespace geometric_sequence_sum_l450_450153

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℕ) (q : ℕ), 
    a 1 = 1 ∧ 
    (∀ n, a n = q^(n-1)) ∧ 
    (4 * a 2, 2 * a 3, a 4 form arithmetic_sequence) ∧
    (a 2 + a 3 + a 4 = 14) := 
begin
  sorry
end

end geometric_sequence_sum_l450_450153


namespace isosceles_triangle_rearrangement_l450_450899

-- Define the properties of the original isosceles triangle
def base : ℝ := 10
def side : ℝ := 13
def height : ℝ := Real.sqrt (side^2 - (base/2)^2)

-- Define the area function for a triangle given its base and height
def area (b h : ℝ) : ℝ := (1/2) * b * h

theorem isosceles_triangle_rearrangement :
  ∃ (h : ℝ), h = Real.sqrt (side^2 - (base/2)^2) ∧
  ∃ (b' : ℝ), b' = 2 * h ∧
  ∃ (h' : ℝ), h' = base / 2 ∧
  area b' h' = area base height :=
by
  sorry

end isosceles_triangle_rearrangement_l450_450899


namespace player1_wins_11th_round_probability_l450_450479

-- Definitions based on the conditions
def egg_shell_strength (n : ℕ) : ℝ := sorry
def player1_won_first_10_rounds : Prop := sorry

-- Main theorem
theorem player1_wins_11th_round_probability
  (h : player1_won_first_10_rounds) :
  Prob (egg_shell_strength 11 > egg_shell_strength 12) = 11 / 12 := sorry

end player1_wins_11th_round_probability_l450_450479


namespace ln_x_le_x_minus_1_l450_450956

noncomputable def tangent_line_ln_x := λ x : ℝ, x - 1

theorem ln_x_le_x_minus_1 (x : ℝ) (hx : x > 0) : log x ≤ x - 1 :=
by sorry

end ln_x_le_x_minus_1_l450_450956


namespace max_possible_k_l450_450516

-- Define basic notions for knights and liars
inductive Person
| knight : Person
| liar : Person

-- Define the round table setting and the unique cards
structure RoundTable :=
  (people : Fin 2015 → Person)
  (cards : Fin 2015 → Nat)
  (unique_cards : ∀ i j : Fin 2015, i ≠ j → cards i ≠ cards j)

-- Everyone says, "My number is greater than both of my neighbors' numbers."
def initial_statement (table : RoundTable) (i : Fin 2015) : Prop :=
  let l := if h : i = 0 then ⟨2014, sorry⟩ else i.pred sorry
  let r := if h : i.val = 2014 then ⟨0, sorry⟩ else i.succ sorry
  if table.people i = Person.knight then
    table.cards i > table.cards l ∧ table.cards i > table.cards r
  else
    ¬(table.cards i > table.cards l ∧ table.cards i > table.cards r)

-- k people then say, "My number is less than both of my neighbors' numbers."
def final_statement (table : RoundTable) (i : Fin 2015) : Prop :=
  let l := if h : i = 0 then ⟨2014, sorry⟩ else i.pred sorry
  let r := if h : i.val = 2014 then ⟨0, sorry⟩ else i.succ sorry
  ¬(table.people i = Person.knight ∧ table.cards i > table.cards l ∧ table.cards i > table.cards r) 
  ∧ table.cards i < table.cards l ∧ table.cards i < table.cards r

-- Prove the maximum possible value of k satisfying the above conditions is 2013
theorem max_possible_k : ∃ (table : RoundTable) (k : ℕ), (∀ i, final_statement table i → (k := k + 1)) ∧ k = 2013 :=
sorry

end max_possible_k_l450_450516


namespace f1_assoc_S1_f1_not_assoc_S2_f2_solution_inequality_f3_assoc_S4_S5_iff_S6_l450_450612

-- Definitions of the sets and conditions
def R := ℝ
def S1 := set.Ici 0  -- [0,+∞)
def S2 := set.Icc 0 1  -- [0,1]
def S3 := {3}  -- {3}
def S4 := {1}  -- {1}
def S5 := set.Ici 0  -- [0,+∞)
def S6 := set.Icc 1 2  -- [1,2]

-- Definition of the functions
def f1 (x : ℝ) := 2 * x - 1
def f2 (x : ℝ) := x^2 - 2 * x

-- Statements to be proved
theorem f1_assoc_S1 : ∀ x1 x2 : ℝ, (x2 - x1) ∈ S1 → (f1 x2 - f1 x1) ∈ S1 := 
sorry

theorem f1_not_assoc_S2 : ∃ x1 x2 : ℝ, (x2 - x1) ∈ S2 ∧ ¬ (f1 x2 - f1 x1) ∈ S2 := 
sorry

theorem f2_solution_inequality : {x : ℝ | 0 ≤ x ∧ x < 3 ∧ 2 ≤ x^2 - 2 * x ∧ x^2 - 2 * x ≤ 3} = set.Icc (real.sqrt 3 + 1) 5 :=
sorry

theorem f3_assoc_S4_S5_iff_S6 : (∀ x1 x2 : ℝ, (x2 - x1) ∈ S4 ∨ (x2 - x1) ∈ S5 ↔ (f1 x2 - f1 x1) ∈ S4 ∨ (f1 x2 - f1 x1) ∈ S5) ↔ 
                                 (∀ x1 x2 : ℝ, (x2 - x1) ∈ S6 → (f1 x2 - f1 x1) ∈ S6) := 
sorry

end f1_assoc_S1_f1_not_assoc_S2_f2_solution_inequality_f3_assoc_S4_S5_iff_S6_l450_450612


namespace cost_of_ice_cream_l450_450088

theorem cost_of_ice_cream (x : ℝ) (h1 : 10 * x = 40) : x = 4 :=
by sorry

end cost_of_ice_cream_l450_450088


namespace describe_cylinder_l450_450339

noncomputable def cylinder_geometric_shape (c : ℝ) (r θ z : ℝ) : Prop :=
  r = c

theorem describe_cylinder (c : ℝ) (hc : 0 < c) :
  ∀ r θ z : ℝ, cylinder_geometric_shape c r θ z ↔ (r = c) :=
by
  sorry

end describe_cylinder_l450_450339


namespace maximum_special_points_l450_450782

theorem maximum_special_points (n : ℕ) (h : n = 11) : 
  ∃ p : ℕ, p = 91 := 
sorry

end maximum_special_points_l450_450782


namespace norris_savings_l450_450756

theorem norris_savings:
  ∀ (N : ℕ), 
  (29 + 25 + N = 85) → N = 31 :=
by
  intros N h
  sorry

end norris_savings_l450_450756


namespace distance_relation_point_Q_triangle_similarity_l450_450387

variables (a b p q x1 y1 x2 y2 x1' y1' x2' y2' : ℝ)
variable (b_nonzero : b ≠ 0)

-- Given the transformation equations
def transform (x y : ℝ) : ℝ × ℝ :=
  (a * x - b * y + p, b * x + a * y + q)

-- Given the points and their transformations
def P1 := (x1, y1)
def P2 := (x2, y2)
def P1' := transform a b p q x1 y1
def P2' := transform a b p q x2 y2

-- 1. Prove the distance relationship
theorem distance_relation :
  ∥P1'.1 - P2'.1∥^2 + ∥P1'.2 - P2'.2∥^2 = (a^2 + b^2) * (∥x1 - x2∥^2 + ∥y1 - y2∥^2) :=
sorry

-- 2. Prove the coordinates of point Q
theorem point_Q :
  let Q := ( (1 - a) * p - b * q ) / ((1 - a)^2 + b^2),
         ( q * ( 1 - a ) + b * p ) / ((1 - a)^2 + b^2) in
  transform a b p q Q.1 Q.2 = Q :=
sorry

-- 3. Prove similarity and area ratio of triangles
theorem triangle_similarity :
  ∃ k : ℝ, k = 1 / (a^2 + b^2) ∧
  (similar (triangle.mk Q P1 P2) (triangle.mk Q P1' P2') ∧
  (area (triangle.mk Q P1 P2)) / (area (triangle.mk Q P1' P2')) = k) :=
sorry

end distance_relation_point_Q_triangle_similarity_l450_450387


namespace octahedron_side_length_correct_l450_450891

noncomputable def octahedron_side_length : ℝ :=
let Q1 := (0 : ℝ, 0 : ℝ, 0 : ℝ),
    Q1' := (1 : ℝ, 1 : ℝ, 1 : ℝ),
    Q2_on_octa := (1/3 : ℝ, 0 : ℝ, 0 : ℝ),
    Q3_on_octa := (0 : ℝ, 1/3 : ℝ, 0 : ℝ),
    Q4_on_octa := (0 : ℝ, 0 : ℝ, 1/3 : ℝ) in
real.sqrt (((1/3 : ℝ) - (0 : ℝ))^(2:ℝ) + ((0 : ℝ) - (1/3 : ℝ))^2)

theorem octahedron_side_length_correct : octahedron_side_length = real.sqrt 2 / 3 :=
by
  sorry

end octahedron_side_length_correct_l450_450891


namespace path_traversed_by_P_is_correct_l450_450037

-- Definitions of the conditions
def right_angled_isosceles_triangle (A B P : Type) (AB BP : ℝ) : Prop :=
  AB = BP ∧ ∃ (CP: ℝ), AB^2 + BP^2 = CP^2

def square (A X Y Z : Type) (side_length : ℝ) : Prop :=
  side_length = 8

-- Midpoints M_AB, M_BP, and M_AP are implicitly from the geometry
-- Total path length calculation
noncomputable def path_length_traversed_by_P (A B P X Y Z : Type)
  (AB BP : ℝ) (side_length: ℝ) [right_angled_isosceles_triangle A B P AB BP] [square A X Y Z side_length] : ℝ :=
  4 * 3 * (2 * Real.sqrt 2 * (Real.pi / 2))

-- The statement to prove
theorem path_traversed_by_P_is_correct (A B P X Y Z : Type) (AB BP : ℝ) (side_length: ℝ) 
  [right_angled_isosceles_triangle A B P AB BP] [square A X Y Z side_length] :
  path_length_traversed_by_P A B P X Y Z AB BP side_length = 12 * Real.pi * Real.sqrt 2 :=
sorry

end path_traversed_by_P_is_correct_l450_450037


namespace johan_painted_green_fraction_l450_450722

theorem johan_painted_green_fraction :
  let total_rooms := 10
  let walls_per_room := 8
  let purple_walls := 32
  let purple_rooms := purple_walls / walls_per_room
  let green_rooms := total_rooms - purple_rooms
  (green_rooms : ℚ) / total_rooms = 3 / 5 := by
  sorry

end johan_painted_green_fraction_l450_450722


namespace jim_catches_bob_in_20_minutes_l450_450903

theorem jim_catches_bob_in_20_minutes
  (bob_speed : ℝ)
  (jim_speed : ℝ)
  (bob_head_start : ℝ)
  (bob_speed_mph : bob_speed = 6)
  (jim_speed_mph : jim_speed = 9)
  (bob_headstart_miles : bob_head_start = 1) :
  ∃ (m : ℝ), m = 20 := 
by
  sorry

end jim_catches_bob_in_20_minutes_l450_450903


namespace arithmetic_seq_value_l450_450345

open_locale classical

noncomputable def a (n : ℕ) : ℕ := sorry

variables (a_4 a_10 a_16 a_18 a_14 : ℕ) (d : ℕ)
  (h_arithmetic : ∀ n m : ℕ, a (n + m + 1) - a n = d * (m + 1))
  (h_condition : a 4 + a 10 + a 16 = 30)
  -- Assuming some properties and general forms of arithmetic sequences.

theorem arithmetic_seq_value :
  (a 18 - 2 * a 14 = -10) :=
by sorry

end arithmetic_seq_value_l450_450345


namespace solve_for_x_l450_450033

theorem solve_for_x :
  ∃ x : ℝ, (64^(x-1) / 4^(x-1) = 256^(2*x)) ∧ x = -1/3 :=
by
  sorry

end solve_for_x_l450_450033


namespace smallest_N_l450_450275

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450275


namespace player1_wins_11th_round_l450_450464

theorem player1_wins_11th_round (player1_wins_first_10 : ∀ (round : ℕ), round < 10 → player1_wins round) : 
  prob_winning_11th_round player1 = 11 / 12 :=
sorry

end player1_wins_11th_round_l450_450464


namespace median_of_100_set_l450_450318

theorem median_of_100_set 
  (S : Finset ℝ) (h_card : S.card = 100)
  (h_remove1 : ∃ x ∈ S, median (S.erase x) = 78)
  (h_remove2 : ∃ y ∈ S, median (S.erase y) = 66) : 
  median S = 72 :=
by
  sorry

end median_of_100_set_l450_450318


namespace tire_circumference_constant_l450_450526

/--
Given the following conditions:
1. Car speed v = 120 km/h
2. Tire rotation rate n = 400 rpm
3. Tire pressure P = 32 psi
4. Tire radius changes according to the formula R = R_0(1 + kP)
5. R_0 is the initial tire radius
6. k is a constant relating to the tire's elasticity
7. Change in tire pressure due to the incline is negligible

Prove that the circumference C of the tire is 5 meters.
-/
theorem tire_circumference_constant (v : ℝ) (n : ℝ) (P : ℝ) (R_0 : ℝ) (k : ℝ) 
  (h1 : v = 120 * 1000 / 3600) -- Car speed in m/s
  (h2 : n = 400 / 60)           -- Tire rotation rate in rps
  (h3 : P = 32)                 -- Tire pressure in psi
  (h4 : ∀ R P, R = R_0 * (1 + k * P)) -- Tire radius formula
  (h5 : ∀ P, P = 0)             -- Negligible change in tire pressure
  : C = 5 :=
  sorry

end tire_circumference_constant_l450_450526


namespace pallets_total_l450_450889

theorem pallets_total (P : ℕ) (h1 : P / 2 + P / 4 + P / 5 + 1 = P) : P = 20 :=
by
  sorry

end pallets_total_l450_450889


namespace Alfred_repair_cost_l450_450045

noncomputable def scooter_price : ℕ := 4700
noncomputable def sale_price : ℕ := 5800
noncomputable def gain_percent : ℚ := 9.433962264150944
noncomputable def gain_value (repair_cost : ℚ) : ℚ := sale_price - (scooter_price + repair_cost)

theorem Alfred_repair_cost : ∃ R : ℚ, gain_percent = (gain_value R / (scooter_price + R)) * 100 ∧ R = 600 :=
by
  sorry

end Alfred_repair_cost_l450_450045


namespace robbers_can_divide_loot_equally_l450_450489

theorem robbers_can_divide_loot_equally (coins : List ℕ) (h1 : (coins.sum % 2 = 0)) 
    (h2 : ∀ k, (k % 2 = 1 ∧ 1 ≤ k ∧ k ≤ 2017) → k ∈ coins) :
  ∃ (subset1 subset2 : List ℕ), subset1 ∪ subset2 = coins ∧ subset1.sum = subset2.sum :=
by
  sorry

end robbers_can_divide_loot_equally_l450_450489


namespace median_of_100_numbers_l450_450329

theorem median_of_100_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 100)
  (h_median_99_1 : ∀ num ∈ numbers, median (numbers.erase num) = 78 → num ∈ numbers)
  (h_median_99_2 : ∀ num ∈ numbers, median (numbers.erase num) = 66 → num ∈ numbers) :
  median numbers = 72 :=
sorry

end median_of_100_numbers_l450_450329


namespace determinant_of_given_matrix_l450_450450

variable (A : Matrix (Fin 2) (Fin 2) ℤ)
variable hA : A = ![![1, 2], ![4, 7]]

theorem determinant_of_given_matrix : Matrix.det A = -1 := 
by
  sorry

end determinant_of_given_matrix_l450_450450


namespace cos_seven_pi_six_l450_450588

theorem cos_seven_pi_six : (Real.cos (7 * Real.pi / 6) = - Real.sqrt 3 / 2) :=
sorry

end cos_seven_pi_six_l450_450588


namespace parallelogram_solution_l450_450703

theorem parallelogram_solution {x y : ℝ} (EF FG GH HE : ℝ) 
  (h1 : EF = 45) (h2 : FG = 4*y^2) (h3 : GH = 3*x + 6) (h4 : HE = 32) :
  x = 13 ∧ y = 2*sqrt(2) :=
by
  have hx : x = 13, from eq_of_sub_eq_zero (by linarith)
  have hy : y = 2*sqrt(2), from eq_of_sub_eq_zero (by linarith)
  exact ⟨hx, hy⟩

end parallelogram_solution_l450_450703


namespace min_sqrt_combo_eq_three_l450_450165

noncomputable def min_value_of_sqrt_sum (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : ℝ :=
  inf { c | c = sqrt (a^2 + 1 / a) + sqrt (b^2 + 1 / b) }

theorem min_sqrt_combo_eq_three (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : sqrt (a^2 + 1 / a) + sqrt (b^2 + 1 / b) >= 3 :=
  by
  sorry

end min_sqrt_combo_eq_three_l450_450165


namespace triangle_third_side_length_l450_450201

theorem triangle_third_side_length (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 8) :
  (a + b > c) → (a + c > b) → (b + c > a) → c = 6 :=
by
  intros h₃ h₄ h₅
  rw [h₁, h₂] at *
  -- Simplified inequalities from the solution step
  have h₆ : 5 + 8 > c := by rw [h₁, h₂]; exact h₃
  have h₇ : 8 + c > 5 := by rw h₂; exact h₄
  have h₈ : 5 + c > 8 := by rw h₁; exact h₅
  sorry

end triangle_third_side_length_l450_450201


namespace compute_expression_l450_450738

theorem compute_expression (p q : ℚ) (hp : p = 4 / 7) (hq : q = 5 / 6) : 
  2 * (p ^ (-2)) * (q ^ 3) = 2041.67 / 576 := by 
  sorry

end compute_expression_l450_450738


namespace find_radius_Q2_l450_450620

noncomputable def radius_Q2_of_pyramid (side_length_base : ℕ) (length_lateral_edge : ℕ) : ℝ :=
  if side_length_base = 12 ∧ length_lateral_edge = 10 then
    (6 * Real.sqrt 7) / 49
  else 0

theorem find_radius_Q2 :
  ∀ (Q1 Q2 : Sphere) (pyramid : Pyramid), 
    pyramid.base_side_length = 12 →
    pyramid.lateral_edge_length = 10 →
    Q1.is_inscribed_in_pyramid pyramid →
    Q2.touches_all_lateral_faces_and_Q1 pyramid Q1 →
    Q2.radius = (6 * Real.sqrt 7) / 49 := 
sorry

end find_radius_Q2_l450_450620


namespace three_digit_permutation_difference_l450_450420

theorem three_digit_permutation_difference (a b c d e f : ℕ) 
  (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) (h5 : 0 ≤ c) (h6 : c ≤ 9)
  (h7 : 1 ≤ d) (h8 : d ≤ 9) (h9 : 0 ≤ e) (h10 : e ≤ 9) (h11 : 0 ≤ f) (h12 : f ≤ 9) 
  (h13 : (100 * a + 10 * b + c) ≠ (100 * d + 10 * e + f))
  (h14 : set.univ.perm (a :: b :: c :: []) = (d :: e :: f :: [])) 
  : abs (100 * a + 10 * b + c - (100 * d + 10 * e + f)) = 36 ∨
    abs (100 * a + 10 * b + c - (100 * d + 10 * e + f)) = 81 :=
sorry

end three_digit_permutation_difference_l450_450420


namespace circumcircle_equivalency_l450_450157

theorem circumcircle_equivalency {A B C P : Point} (h_equilateral : equilateral_triangle A B C)
  (h_distances : distance P A ≤ distance P B ∧ distance P C ≤ distance P B) :
  (on_circumcircle P A B C ↔ distance P B = distance P A + distance P C) :=
sorry

end circumcircle_equivalency_l450_450157


namespace sandy_bought_fish_l450_450771

theorem sandy_bought_fish (original_fish : ℕ) (current_fish : ℕ) (h1 : original_fish = 26) (h2 : current_fish = 32) : current_fish - original_fish = 6 := 
by
  rw [h1, h2]
  simp
  sorry

end sandy_bought_fish_l450_450771


namespace negation_proposition_l450_450437

theorem negation_proposition (h : ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) :
  ¬(∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) = ∃ x : ℝ, x^2 - 2*x + 4 > 0 :=
by
  sorry

end negation_proposition_l450_450437


namespace third_side_length_l450_450184

theorem third_side_length (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) (h3 : 3 < x < 13) : (5 < x + 8) ∧ (x < 5 + 8) ∧ (5 < x + 3) ∧ (x < 8 + 5) := 
by
  sorry

end third_side_length_l450_450184


namespace probability_divisible_by_5_l450_450044

-- Definitions based on the given conditions:
def wheel_sections := {2, 4, 6, 8}

-- Defining the function to form a three-digit number from three spins
def is_valid_number (a b c : ℕ) : bool :=
  a ∈ wheel_sections ∧ b ∈ wheel_sections ∧ c ∈ wheel_sections

-- Defining the criterion for being divisible by 5
def divisible_by_5 (n : ℕ) : bool :=
  n % 10 = 0 ∨ n % 10 = 5

-- Given that no three-digit number formed from the wheel spins ends in a digit allowing divisibility by 5
theorem probability_divisible_by_5 : 
  ∀ a b c, is_valid_number a b c → ¬ (divisible_by_5 (a * 100 + b * 10 + c)) :=
by sorry

end probability_divisible_by_5_l450_450044


namespace symmetric_of_A_l450_450335

def point : Type := ℝ × ℝ × ℝ

def symmetric_point_x_axis (p : point) : point :=
  (p.1, -p.2, -p.3)

theorem symmetric_of_A : symmetric_point_x_axis (3, 4, -5) = (3, -4, 5) :=
  by 
    -- declaring each coordinate explicitly
    let x := 3
    let y := 4
    let z := -5
    -- applying symmetry
    show (x, -y, -z) = (3, -4, 5)
    sorry

end symmetric_of_A_l450_450335


namespace decryption_easier_with_repair_l450_450422

theorem decryption_easier_with_repair :
  let unique_letters_thermometer := {'т', 'е', 'р', 'м', 'о'} 
  let unique_letters_repair := {'р', 'е', 'м', 'о', 'н', 'т'}
  unique_letters_thermometer.card < unique_letters_repair.card :=
by
  sorry

end decryption_easier_with_repair_l450_450422


namespace probability_of_same_color_balls_l450_450520

theorem probability_of_same_color_balls :
  let num_balls := 5
  let num_white := 3
  let num_black := 2
  let total_drawn := 2
  let total_outcomes := nat.choose num_balls total_drawn
  let white_outcomes := nat.choose num_white total_drawn
  let black_outcomes := nat.choose num_black total_drawn
  let favorable_outcomes := white_outcomes + black_outcomes
  let probability := favorable_outcomes / total_outcomes
  probability = (2 : ℚ) / 5 :=
by
  sorry

end probability_of_same_color_balls_l450_450520


namespace triangle_third_side_length_l450_450202

theorem triangle_third_side_length (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 8) :
  (a + b > c) → (a + c > b) → (b + c > a) → c = 6 :=
by
  intros h₃ h₄ h₅
  rw [h₁, h₂] at *
  -- Simplified inequalities from the solution step
  have h₆ : 5 + 8 > c := by rw [h₁, h₂]; exact h₃
  have h₇ : 8 + c > 5 := by rw h₂; exact h₄
  have h₈ : 5 + c > 8 := by rw h₁; exact h₅
  sorry

end triangle_third_side_length_l450_450202


namespace trig_cos_square_identity_solution_l450_450506

theorem trig_cos_square_identity_solution (x : ℝ) :
  (cos x)^2 + (cos (2 * x))^2 - (cos (3 * x))^2 - (cos (4 * x))^2 = 0 →
  (∃ k : ℤ, x = π / 2 + k * π) ∨
  (∃ n : ℤ, x = n * π / 5) ∨
  (∃ n : ℤ, x = - (n * π / 2)) :=
by
  sorry

end trig_cos_square_identity_solution_l450_450506


namespace find_k_l450_450731

open Nat

def sequence (p q r : ℕ) : ℕ := 3^p + 3^q + 3^r

def a_n := { n : ℕ | ∃ p q r : ℕ, 0 ≤ p ∧ p < q ∧ q < r ∧ n = sequence p q r }

theorem find_k : ∃ k, a_n k = 2511 ∧ k = 50 := sorry

end find_k_l450_450731


namespace monotonic_decreasing_interval_l450_450429

noncomputable def f (x : ℝ) : ℝ := log 3 (x^2 - 2 * x - 8)

theorem monotonic_decreasing_interval : 
  ∀ x y : ℝ, x < y → x < -2 ∧ y < -2 → f(x) > f(y) := 
by
  intros x y hxy hx_neg2 hy_neg2
  sorry

end monotonic_decreasing_interval_l450_450429


namespace sum_of_acute_angles_l450_450163

theorem sum_of_acute_angles (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h : sin α ^ 2 + sin β ^ 2 = sin (α + β)) : α + β = π / 2 :=
sorry

end sum_of_acute_angles_l450_450163


namespace proof_problem_l450_450797

def label_sum_of_domains_specified (labels: List Nat) (domains: List Nat) : Nat :=
  let relevant_labels := labels.filter (fun l => domains.contains l)
  relevant_labels.foldl (· + ·) 0

def label_product_of_continuous_and_invertible (labels: List Nat) (properties: List Bool) : Nat :=
  let relevant_labels := labels.zip properties |>.filter (fun (_, p) => p) |>.map (·.fst)
  relevant_labels.foldl (· * ·) 1

theorem proof_problem :
  label_sum_of_domains_specified [1, 2, 3, 4] [4] = 4 ∧ label_product_of_continuous_and_invertible [1, 2, 3, 4] [true, false, true, false] = 3 :=
by
  sorry

end proof_problem_l450_450797


namespace tammy_total_distance_l450_450408

-- Define the times and speeds for each segment and breaks
def initial_speed : ℝ := 55   -- miles per hour
def initial_time : ℝ := 2     -- hours
def road_speed : ℝ := 40      -- miles per hour
def road_time : ℝ := 5        -- hours
def first_break : ℝ := 1      -- hour
def drive_after_break_speed : ℝ := 50  -- miles per hour
def drive_after_break_time : ℝ := 15   -- hours
def hilly_speed : ℝ := 35     -- miles per hour
def hilly_time : ℝ := 3       -- hours
def second_break : ℝ := 0.5   -- hours
def finish_speed : ℝ := 60    -- miles per hour
def total_journey_time : ℝ := 36 -- hours

-- Define a function to calculate the segment distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Define the total distance calculation
def total_distance : ℝ :=
  distance initial_speed initial_time +
  distance road_speed road_time +
  distance drive_after_break_speed drive_after_break_time +
  distance hilly_speed hilly_time +
  distance finish_speed (total_journey_time - (initial_time + road_time + drive_after_break_time + hilly_time + first_break + second_break))

-- The final proof statement
theorem tammy_total_distance : total_distance = 1735 :=
  sorry

end tammy_total_distance_l450_450408


namespace cosine_angle_l450_450629

variable (a b : ℝ → ℝ) -- vectors a and b, represented as functions of ℝ

-- Assumptions and conditions
axiom a_non_zero : a ≠ 0
axiom b_non_zero : b ≠ 0
axiom a_magnitude : ‖a‖ = 1
axiom b_magnitude : ‖b‖ = 2
axiom orthogonal_condition : a + b ⬝ (3 * a - b) = 0 -- dot product being zero means orthogonality

-- Definition of cosine of the angle between two vectors
def cos_theta (a b : ℝ → ℝ) : ℝ := (a ⬝ b) / (‖a‖ * ‖b‖)

-- Main theorem to prove the cosine value
theorem cosine_angle : cos_theta a b = 1 / 4 :=
by
  sorry

end cosine_angle_l450_450629


namespace g_at_10_l450_450104

-- Define the function condition as a predicate
def func_condition (g: ℝ → ℝ) :=
  ∀ x y : ℝ, g(x) + g(3 * x + y) + 7 * x * y = g(4 * x - y) + 3 * x^2 + 2

-- Theorem statement to prove g(10) = -48 under the given condition
theorem g_at_10 (g: ℝ → ℝ) (h: func_condition g) : g(10) = -48 :=
sorry

end g_at_10_l450_450104


namespace problem1_solution_l450_450857

def f (x : ℝ) : ℝ := abs (x + 1) + abs (2 * x - 4)

theorem problem1_solution (x : ℝ) : f(x) ≥ 6 → (x ≤ -1 ∨ x ≥ 3) :=
by
  sorry

end problem1_solution_l450_450857


namespace percent_of_x_is_z_l450_450000

theorem percent_of_x_is_z (x y z : ℝ) (h1 : 0.45 * z = 1.2 * y) (h2 : y = 0.75 * x) : z = 2 * x :=
by
  sorry

end percent_of_x_is_z_l450_450000


namespace solution_set_l450_450640

open Set

variable {f : ℝ → ℝ}

-- Definitions of conditions
def even_function_on_ℝ (g : ℝ → ℝ) : Prop :=
  ∀ x, g(x) = g(-x)

variable (h1 : even_function_on_ℝ (λ x, f(x + 1)))
variable (h2 : ∀ x, x > 1 → f'(x) < 0)
variable (h3 : f 4 = 0)

-- The main theorem statement representing the math proof problem
theorem solution_set (x : ℝ) : (x + 3) * f (x + 4) < 0 ↔ (x ∈ Ioo (-6) (-3)) ∨ (x ∈ Ioi 0) :=
by
  sorry

end solution_set_l450_450640


namespace part_i_part_ii_part_iii_part_iv_l450_450603

-- Part (i): 3-adic valuation of A = 2^27 + 1
theorem part_i : padicVal 3 (2^27 + 1) = 4 := sorry

-- Part (ii): 7-adic valuation of B = 161^14 - 112^14
theorem part_ii : padicVal 7 (161^14 - 112^14) = 16 := sorry

-- Part (iii): 2-adic valuation of C = 7^20 + 1
theorem part_iii : padicVal 2 (7^20 + 1) = 1 := sorry

-- Part (iv): 2-adic valuation of D = 17^48 - 5^48
theorem part_iv : padicVal 2 (17^48 - 5^48) = 6 := sorry

end part_i_part_ii_part_iii_part_iv_l450_450603


namespace exists_similar_quadrilateral_l450_450108

theorem exists_similar_quadrilateral 
  (Q : Type) [quadrilateral Q]
  (A B C D : Q) 
  (l1 l2 l3 l4 : line)
  (M1 M2 M3 M4 : point)
  (S : circle)
  (condition1 : A ∈ l1 ∧ B ∈ l2 ∧ C ∈ l3 ∧ D ∈ l4)
  (condition2 : A ∈ M1 ∧ B ∈ M2 ∧ C ∈ M3 ∧ D ∈ M4)
  (condition3 : BC ∈ M ∧ CD ∈ N ∧ BD ∈ P ∧ A ∈ S)
  : ∃ (Q' : quadrilateral) (A' B' C' D' : Q'),
      similar_quadrilateral Q' Q ∧
      A' ∈ l1 ∧ B' ∈ l2 ∧ C' ∈ l3 ∧ D' ∈ l4 ∧
      A' ∈ M1 ∧ B' ∈ M2 ∧ C' ∈ M3 ∧ D' ∈ M4 ∧
      BC ∈ M ∧ CD ∈ N ∧ BD ∈ P ∧ A' ∈ S :=
sorry -- Proof omitted

end exists_similar_quadrilateral_l450_450108


namespace non_similar_triangles_l450_450669

theorem non_similar_triangles (d : ℕ) (h1 : even d) (h2 : 0 < d ∧ d < 60) :
  ∃ t : ℕ, t = 29 :=
by
  sorry

end non_similar_triangles_l450_450669


namespace tan_alpha_eq_two_l450_450225

-- Definitions based on conditions
def a (α : ℝ) : ℝ × ℝ := (Real.sin α, 2)
def b (α : ℝ) : ℝ × ℝ := (1, -Real.cos α)
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

-- Theorem statement
theorem tan_alpha_eq_two (α : ℝ) (h : perpendicular (a α) (b α)) : Real.tan α = 2 :=
by
  sorry

end tan_alpha_eq_two_l450_450225


namespace total_ducks_and_ducklings_l450_450380

theorem total_ducks_and_ducklings :
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6
  (ducks1 + ducks2 + ducks3) + (ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3) = 99 :=
by
  sorry

end total_ducks_and_ducklings_l450_450380


namespace computation_l450_450913

theorem computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  have h₁ : 27 = 3^3 := by rfl
  have h₂ : (3 : ℕ) ^ 4 = 81 := by norm_num
  have h₃ : 27^63 / 27^61 = (3^3)^63 / (3^3)^61 := by rw [h₁]
  rwa [← pow_sub, nat.sub_eq_iff_eq_add] at h₃
  have h4: 3 * 3^4 = 3^5 := by norm_num
  have h5: -486 = 3^5 - 3^6 := by norm_num
  exact h5
  sorry

end computation_l450_450913


namespace andrew_brian_ratio_l450_450403

-- Definitions based on conditions extracted from the problem
variables (A S B : ℕ)

-- Conditions
def steven_shirts : Prop := S = 72
def brian_shirts : Prop := B = 3
def steven_andrew_relation : Prop := S = 4 * A

-- The goal is to prove the ratio of Andrew's shirts to Brian's shirts is 6
theorem andrew_brian_ratio (A S B : ℕ) 
  (h1 : steven_shirts S) 
  (h2 : brian_shirts B)
  (h3 : steven_andrew_relation A S) :
  A / B = 6 := by
  sorry

end andrew_brian_ratio_l450_450403


namespace player1_wins_11th_round_probability_l450_450483

-- Definitions based on the conditions
def egg_shell_strength (n : ℕ) : ℝ := sorry
def player1_won_first_10_rounds : Prop := sorry

-- Main theorem
theorem player1_wins_11th_round_probability
  (h : player1_won_first_10_rounds) :
  Prob (egg_shell_strength 11 > egg_shell_strength 12) = 11 / 12 := sorry

end player1_wins_11th_round_probability_l450_450483


namespace trisection_of_angle_l450_450158

-- Define the basic geometrical elements: points, angles, and perpendicularity.
structure Point := (x : ℝ) (y : ℝ)

structure Triangle (A B C : Point) := (angle_AOB : ℝ) (right_angle_E : Point → Point)

structure Line (A B : Point) := (parallel : B → Point)

-- Define the properties for the specific problem
def isPerpendicular (A E O B : Point) : Prop := 
  A.y - E.y = 0 ∧ E.x - B.x = 0

def isParallel (A B : Point) : Prop :=
  A.y = B.y

def segmentLength (K T : Point) (AO_length : ℝ) : Prop :=
  (K.x - T.x)^2 + (K.y - T.y)^2 = 4 * AO_length^2

def angleTrisected (A O B : Point) (α : ℝ) : Prop :=
  (3 * α = A.angle_AOB)

-- The main theorem
theorem trisection_of_angle
  (A O B E : Point) 
  (h1 : isPerpendicular A E O B)
  (line_through_A_parallel_OB : Line A B)
  (K T : Point) 
  (AO_length : ℝ)
  (h2 : segmentLength K T AO_length) :
  ∃ α, angleTrisected A O B α := 
by 
  sorry -- Proof omitted as per the instructions.

end trisection_of_angle_l450_450158


namespace game_ends_after_21_rounds_l450_450539

def tokensPerPlayerInitial : Type := ℕ -- Type for representing tokens per player initially
def rounds : Type := ℕ -- Type for representing number of rounds in the game

-- Initial token counts as conditions
def playerA_initial_tokens : tokensPerPlayerInitial := 24
def playerB_initial_tokens : tokensPerPlayerInitial := 21
def playerC_initial_tokens : tokensPerPlayerInitial := 20

-- The main theorem to be proven
theorem game_ends_after_21_rounds :
  ∀ (A B C R : tokensPerPlayerInitial),
  A = 24 → B = 21 → C = 20 →
  (∃ n : rounds, n = 21 ∧ 
  (let A_final := A - 6*n/3 in
   let B_final := B - 6*n/3 in 
   let C_final := C - 6*n/3 in
   B_final = 0 ∨ A_final = 0 ∨ C_final = 0)) :=
begin
  intros A B C R,
  assume h_Ae,h_Be,h_Ce,
  use 21,
  split,
  { refl },
  { sorry }
end

end game_ends_after_21_rounds_l450_450539


namespace charlie_cleaning_time_l450_450557

theorem charlie_cleaning_time :
  ∀ (Alice_time : ℝ) (Bob_time : ℝ), 
  Alice_time = 30 →
  Bob_time = (3 / 4) * Alice_time →
  ∃ Charlie_time : ℝ, 
  Charlie_time = (1 / 3) * Bob_time ∧ Charlie_time = 7.5 :=
by
  intros Alice_time Bob_time hAlice_time hBob_time
  use (1 / 3) * Bob_time
  split
  · calc (1 / 3) * Bob_time = (1 / 3) * ((3 / 4) * 30) : by rw [hBob_time]
                   ... = 7.5 : by norm_num
  · sorry

end charlie_cleaning_time_l450_450557


namespace suma_work_rate_l450_450398

theorem suma_work_rate (x : ℝ) (h1 : 1 / 6 + 1 / x = 1 / 4) : x = 6 :=
by
  nontrivial
  -- The next line simply states that 6 is the solution to the equation.
  sorry

end suma_work_rate_l450_450398


namespace matrix_pow_101_l450_450725

noncomputable def matrixA : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]
  ]

theorem matrix_pow_101 :
  matrixA ^ 101 =
  ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ] :=
sorry

end matrix_pow_101_l450_450725


namespace cannot_reach_all_pluses_l450_450828

open Matrix

-- Define an 8x8 grid with values representing "+" and "-" as boolean
def Grid := Matrix (Fin 8) (Fin 8) Bool

-- Define the operation to flip signs within a sub-grid
def invertSubGrid (grid : Grid) (r c : Fin 6) (size : Fin 2) : Grid :=
  let n := if size = 0 then 3 else 4
  fun i j =>
    if r ≤ i < r + n ∧ c ≤ j < c + n then
      not (grid i j)
    else
      grid i j

-- Theorem to state that we cannot always achieve a grid with all '+' signs
theorem cannot_reach_all_pluses (init_grid : Grid) : 
  (∃ g : Grid, 
  (∀ r : Fin 6, ∀ c : Fin 6, ∀ size : Fin 2, invertSubGrid g r c size = init_grid) → ∀ i j, g i j = true) → 
  False :=
sorry

end cannot_reach_all_pluses_l450_450828


namespace arithmetic_sequence_general_term_geometric_sequence_general_term_l450_450636

-- Definitions of the arithmetic and geometric sequences given the conditions.
def arithmetic_seq (a_1 d : ℤ) (n : ℕ+) : ℤ := a_1 + (n - 1) * d
def geometric_seq (b_1 q : ℝ) (n : ℕ+) : ℝ := b_1 * (q^(n - 1))

theorem arithmetic_sequence_general_term (d : ℤ) (a_1 : ℤ) (n : ℕ+) 
  (h1 : (a_1 + 5 * d = 16)) 
  (h2 : (3 * d - a_1 = 8)) : 
  (arithmetic_seq a_1 d n) = 3 * n - 2 :=
by
  sorry

theorem geometric_sequence_general_term (b_1 : ℝ) (q : ℝ) (n : ℕ+) 
  (h3 : b_1 = 2)
  (h4 : 0 < q)
  (h5 : b_1 * (q + q^2) = 12) 
  (h6 : q = 2): 
  (geometric_seq b_1 q n) = 2^n :=
by
  sorry

def product_sequence_sum (T_n : ℝ) (n : ℕ+) 
  (h7 : arithmetic_seq 1 3 4 - 2 * arithmetic_seq 1 3 1 = 8) 
  (h8 : arithmetic_seq 1 3 11 = 11 * geometric_seq 2 2 4) : 
  T_n = ((3 * n - 2) / 3) * 4^(n + 1) + 8 / 3 :=
by
  sorry

end arithmetic_sequence_general_term_geometric_sequence_general_term_l450_450636


namespace negation_of_exists_statement_l450_450435

theorem negation_of_exists_statement :
  ¬ (∃ x0 : ℝ, x0 > 0 ∧ x0^2 - 5 * x0 + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 :=
by
  sorry

end negation_of_exists_statement_l450_450435


namespace coordinates_with_respect_to_origin_l450_450419

theorem coordinates_with_respect_to_origin (x y : ℤ) (hx : x = 3) (hy : y = -2) : (x, y) = (3, -2) :=
by
  sorry

end coordinates_with_respect_to_origin_l450_450419


namespace find_x_pow_y_l450_450613

theorem find_x_pow_y (x y : ℝ) : |x + 2| + (y - 3)^2 = 0 → x ^ y = -8 :=
by
  sorry

end find_x_pow_y_l450_450613


namespace unit_vector_exists_in_xy_plane_l450_450135

theorem unit_vector_exists_in_xy_plane :
  ∃ (x y : ℝ), (x^2 + y^2 = 1 ∧ 
  ((x + 2 * y) / (Real.sqrt 5) = Real.sqrt (3 / 4)) ∧
  ((-x + y) / (Real.sqrt 2) = Real.sqrt (2) / 2)) :=
by {
  let u := (x, y, 0),
  have hu : x^2 + y^2 = 1, sorry,
  have h1 : (x + 2 * y) / (Real.sqrt 5) = Real.sqrt 3 / 2, sorry,
  have h2 : (-x + y) / (Real.sqrt 2) = 1 / Real.sqrt 2, sorry,
  existsi x, existsi y,
  exact ⟨hu, h1, h2⟩,
  sorry
}

end unit_vector_exists_in_xy_plane_l450_450135


namespace greatest_divisor_sum_digits_four_l450_450745

theorem greatest_divisor_sum_digits_four :
  ∃ n : ℕ, (∀ a b : ℕ, a = 4665 → b = 6905 → n ∣ (a - b) ∧ sum_digits n = 4) ∧ n = 4 :=
by sorry

-- Define sum_digits function
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

end greatest_divisor_sum_digits_four_l450_450745


namespace Edmund_can_wrap_15_boxes_every_3_days_l450_450938

-- We define the conditions as Lean definitions
def inches_per_gift_box : ℕ := 18
def inches_per_day : ℕ := 90

-- We state the theorem to prove the question (15 gift boxes every 3 days)
theorem Edmund_can_wrap_15_boxes_every_3_days :
  (inches_per_day / inches_per_gift_box) * 3 = 15 :=
by
  sorry

end Edmund_can_wrap_15_boxes_every_3_days_l450_450938


namespace functions_equal_l450_450838
noncomputable theory

def f (t : ℝ) : ℝ := abs t
def g (x : ℝ) : ℝ := real.sqrt (x^2)

theorem functions_equal : ∀ (x : ℝ), f x = g x := by
  intro x
  -- proof would go here
  sorry

end functions_equal_l450_450838


namespace day_9_tuesday_7280_students_l450_450360

def initial_count : ℕ :=
  1 -- Jessica

def day1_count : ℕ := 
  initial_count + 3 -- Jessica tells three friends

def day2_count : ℕ := 
  day1_count + 2 * 3 -- Two friends tell three others each

def day3_count : ℕ :=
  day2_count + 3 * 3 + 3 -- Each new person from Tuesday + third friend from Monday

def count_on_nth_day (n : ℕ) (prev2 prev1 : ℕ) : ℕ :=
  prev1 + 3 * (prev1 - prev2)

-- Function to compute the count of students by day
noncomputable def compute_total_students (day : ℕ) : ℕ :=
  if day = 1 then day1_count
  else if day = 2 then day2_count
  else if day = 3 then day3_count
  else compute_total_students (day - 1) + 3 * (compute_total_students (day - 1) - compute_total_students (day - 2))

theorem day_9_tuesday_7280_students : compute_total_students 9 ≥ 7280 :=
by {
  sorry -- Proof not required
}

end day_9_tuesday_7280_students_l450_450360


namespace percentage_of_professors_who_are_women_l450_450050

theorem percentage_of_professors_who_are_women
(u {professors : Type}) [fintype u] [decidable_eq u] (W T : ℝ) (women tenured : finset u) :
    T = 0.70 → 
    (∑ x in women ∪ tenured, 1 : ℝ) / (fintype.card u) = 0.90 → 
    (∑ x in tenured \ women, 1 : ℝ) / (∑ x in tenured, 1 : ℝ) = 0.52 → 
    W = 0.7917 := by
  sorry

end percentage_of_professors_who_are_women_l450_450050


namespace temperature_on_last_day_l450_450438

noncomputable def last_day_temperature (T1 T2 T3 T4 T5 T6 T7 : ℕ) (mean : ℕ) : ℕ :=
  8 * mean - (T1 + T2 + T3 + T4 + T5 + T6 + T7)

theorem temperature_on_last_day 
  (T1 T2 T3 T4 T5 T6 T7 mean x : ℕ)
  (hT1 : T1 = 82) (hT2 : T2 = 80) (hT3 : T3 = 84) 
  (hT4 : T4 = 86) (hT5 : T5 = 88) (hT6 : T6 = 90) 
  (hT7 : T7 = 88) (hmean : mean = 86) 
  (hx : x = last_day_temperature T1 T2 T3 T4 T5 T6 T7 mean) :
  x = 90 := by
  sorry

end temperature_on_last_day_l450_450438


namespace find_fraction_value_l450_450728

variable {x y : ℂ}

theorem find_fraction_value
    (h1 : (x^2 + y^2) / (x + y) = 4)
    (h2 : (x^4 + y^4) / (x^3 + y^3) = 2) :
    (x^6 + y^6) / (x^5 + y^5) = 4 := by
  sorry

end find_fraction_value_l450_450728


namespace burattino_suspects_cheating_after_seventh_draw_l450_450066

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

noncomputable def binomial (n k : ℕ) : ℚ :=
(factorial n) / ((factorial k) * factorial (n - k))

noncomputable def probability_no_repeats : ℚ :=
(binomial 39 6) / (binomial 45 6)

noncomputable def estimate_draws_needed : ℕ :=
let a : ℚ := probability_no_repeats in
nat.find (λ n, a^n < 0.01)

theorem burattino_suspects_cheating_after_seventh_draw :
  estimate_draws_needed + 1 = 7 := sorry

end burattino_suspects_cheating_after_seventh_draw_l450_450066


namespace range_of_a_l450_450656

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ)
  (h1 : ∀ (n : ℕ), a_seq n = if n < 6 then (1 / 2 - a) * n + 1 else a ^ (n - 5))
  (h2 : ∀ (n : ℕ), n > 0 → a_seq n > a_seq (n + 1)) :
  (1 / 2 : ℝ) < a ∧ a < (7 / 12 : ℝ) :=
sorry

end range_of_a_l450_450656


namespace part1_part2_l450_450211

noncomputable def f (e a b : ℝ) (x : ℝ) := exp (-x) * (a * x^2 + b * x + 1)
noncomputable def f' (e a b : ℝ) (x : ℝ) := -exp (-x) * (a * x^2 + b * x + 1) + exp (-x) * (2 * a * x + b)

theorem part1 (e : ℝ) (b : ℝ) 
(h1 : f e 1 b (-1) = 0) : 
  f' e 1 b 0 = 1 :=
sorry

theorem part2 (e : ℝ) (a b : ℝ) 
(h2 : a > 1 / 5) 
(h3 : f e a b (-1) = 0) 
(h4 : \(\max_{x \in [-1, 1]} f e a b x = 4 * e)) : 
  a = (8 * e^2 - 3) / 5 ∧ b = (12 * e^2 - 2) / 5 :=
sorry

end part1_part2_l450_450211


namespace midpoint_arc_DE_on_line_MN_l450_450023

variables {A B C D E O M N K : Point}

-- Conditions
variables (abc : triangle A B C)
  (circle_through_BC : circle (triangle.vertex B abc) (triangle.vertex C abc))
  (intersect_AB_AC : ∃ D E, circle_through_BC (triangle.side AB abc) D ∧ circle_through_BC (triangle.side AC abc) E)
  (CD_BE_intersect_O : intersects (line_segment (triangle.vertex C abc) D) (line_segment (triangle.vertex B abc) E) O)
  (M_center_ADE : incenter A D E M)
  (N_center_ODE : incenter O D E N)
  (K_midpoint_arc_DE : midpoint_arc D E circle_through_BC K)
  
-- Question to prove 
theorem midpoint_arc_DE_on_line_MN
  (H : same_line K M N) : 
  midpoint_arc D E circle_through_BC K := 
  sorry

end midpoint_arc_DE_on_line_MN_l450_450023


namespace intersection_A_B_l450_450657

noncomputable def straightLineInA (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (m+3)*x + (m-2)*y - 1 - 2*m = 0

def isTangentLineOfCircle : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 = 2

theorem intersection_A_B :
  (∀ l, (∃ m, straightLineInA m l) → isTangentLineOfCircle l) → 
  (∃ l, isTangentLineOfCircle l) ↔ 
  ((x + y - 2 = 0)) :=
sorry

end intersection_A_B_l450_450657


namespace conjugate_complex_point_l450_450999

-- Define the given complex number
def z : ℂ := 1/2 + complex.i

-- State the theorem to prove the corresponding point of the conjugate of z in the complex plane
theorem conjugate_complex_point (z : ℂ) (hz : z = (1/2 + complex.i)) : complex.conj z = (1/2 - complex.i) :=
by
  rw [hz]
  rw [complex.conj_mk]
  norm_num
  sorry

end conjugate_complex_point_l450_450999


namespace circles_intersect_l450_450218

theorem circles_intersect (R r d: ℝ) (hR: R = 7) (hr: r = 4) (hd: d = 8) : (R - r < d) ∧ (d < R + r) :=
by
  rw [hR, hr, hd]
  exact ⟨by linarith, by linarith⟩

end circles_intersect_l450_450218


namespace angle_of_a_b_k_parallel_k_perpendicular_l450_450663

-- Definitions of the vectors provided in the problem conditions
def vec_a : ℝ × ℝ := (3, 0)
def vec_b : ℝ × ℝ := (-5, 5)
def vec_c (k : ℝ) : ℝ × ℝ := (2, k)

-- Prove the angle between vector a and vector b
theorem angle_of_a_b : 
  let θ := real.arccos ((vec_a.fst * vec_b.fst + vec_a.snd * vec_b.snd) / 
                       (real.sqrt (vec_a.fst^2 + vec_a.snd^2) * real.sqrt (vec_b.fst^2 + vec_b.snd^2)))
  θ = 3 * real.pi / 4 := sorry

-- Prove the value of k if vector b is parallel to vector c
theorem k_parallel :
  (∃ k : ℝ, ∀ k, (-5) / 2 = 5 / k) ↔ k = -2 := sorry

-- Prove the value of k if vector b is perpendicular to (a + c)
theorem k_perpendicular :
  (∃ k : ℝ, let vec_ac := (vec_a.fst + vec_c k .fst, vec_a.snd + vec_c k .snd)
            in (vec_b.fst * vec_ac.fst + vec_b.snd * vec_ac.snd = 0)) ↔ k = 5 := sorry

end angle_of_a_b_k_parallel_k_perpendicular_l450_450663


namespace math_problem_l450_450407

-- Define the function g and its inverse
variable {g : ℝ → ℝ}
variable {g_inv : ℝ → ℝ}

-- Conditions of the problem
axiom g_invertible : Function.injective g ∧ Function.surjective g
@[simp] axiom g_inverts : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x
axiom g_table : (g 1 = 3) ∧ (g 2 = 4) ∧ (g 3 = 6) ∧ (g 4 = 8) ∧ (g 5 = 9)

-- Question rewritten as a theorem statement
theorem math_problem :
  g (g 2) + g (g_inv 3) + g_inv (g_inv 6) = 12 :=
by
  -- Insert proof here
  sorry

end math_problem_l450_450407


namespace smallest_N_l450_450274

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450274


namespace part1_solution_part2_solution_l450_450855

-- Part (1)
def f (x : ℝ) : ℝ := |x + 1| + |2 * x - 4|

theorem part1_solution (x : ℝ) : f(x) >= 6 ↔ x ≤ -1 ∨ x ≥ 3 := 
by 
  intros
  sorry

-- Part (2)
theorem part2_solution (a b c : ℝ) (h : a + 2 * b + 4 * c = 8) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) : 
  ∀x, ∃m, m = min (1/a + 1/b + 1/c) (x) := 
by
  sorry

end part1_solution_part2_solution_l450_450855


namespace missed_both_shots_l450_450580

variables (p q : Prop)

theorem missed_both_shots : (¬p ∧ ¬q) ↔ ¬(p ∨ q) :=
by sorry

end missed_both_shots_l450_450580


namespace area_of_triangle_DEF_l450_450711

theorem area_of_triangle_DEF (PQRS_area : ℕ) (smaller_square_side : ℕ) (DE_eq_DF : ∀ x, x = x ∨ ¬ x = x {x := ℕ})
(point_D_coincides_T : ∀ T, D = T):
  area DEF = 39 / 4 :=
by
  let side_PQRS := real.sqrt PQRS_area
  let DN := side_PQRS / 2 + smaller_square_side + smaller_square_side
  let EF := side_PQRS - smaller_square_side - smaller_square_side
  let area_DEF := 1 / 2 * EF * DN
  exact area_DEF
  sorry

end area_of_triangle_DEF_l450_450711


namespace valid_third_side_length_l450_450175

theorem valid_third_side_length {x : ℝ} (h1 : 5 + 8 > x) (h2 : 5 + x > 8) (h3 : 8 + x > 5) : x = 6 :=
by
  -- Given 5 + 8 > x, 5 + x > 8, 8 + x > 5
  have range1 : 13 > x := h1,
  have range2 : x > 3 := (by linarith [h2]),
  have _ : 3 < 6 ∧ 6 < 13 := by norm_num,
  linarith

#check valid_third_side_length

end valid_third_side_length_l450_450175


namespace crown_cost_before_tip_l450_450880

theorem crown_cost_before_tip (total_paid : ℝ) (tip_percentage : ℝ) (crown_cost : ℝ) :
  total_paid = 22000 → tip_percentage = 0.10 → total_paid = crown_cost * (1 + tip_percentage) → crown_cost = 20000 :=
by
  sorry

end crown_cost_before_tip_l450_450880


namespace solution_pairs_count_l450_450131

theorem solution_pairs_count :
  ∃ n : ℕ, n = 8 ∧
  ∃ f : fin n → ℕ × ℕ,
    (∀ i, 4 * (f i).1 + 7 * (f i).2 = 600 ∧ (f i).1 ≤ (f i).2 ∧
        (f i).1 > 0 ∧ (f i).2 > 0) ∧
    (∀ i j, i ≠ j → f i ≠ f j) :=
by
  sorry

end solution_pairs_count_l450_450131


namespace median_of_100_numbers_l450_450310

theorem median_of_100_numbers (x : Fin 100 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i = 78 → x j = 66 → i = 51 ∧ j = 50 ∨ i = 50 ∧ j = 51)
  (h2 : ∀ i, i ≠ 51 → x 51 = 78)
  (h3 : ∀ i, i ≠ 50 → x 50 = 66) :
  (x 50 + x 51) / 2 = 72 :=
by sorry

end median_of_100_numbers_l450_450310


namespace work_problem_l450_450508

theorem work_problem (W : ℝ) (hW : W > 0) : 
  let A_rate := W / 20 
  let B_rate := W / 15 
  let C_rate := W / 10 
  let combined_rate := A_rate + B_rate + C_rate
  combined_rate = (13 * W) / 60 →
  60 / 13 = 60 / 13 :=
by
  intro h_combined_rate
  rw h_combined_rate
  sorry

end work_problem_l450_450508


namespace smallest_N_l450_450256

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450256


namespace find_B_l450_450227

variables {a b c A B C : ℝ}

-- Conditions
axiom given_condition_1 : (c - b) / (c - a) = (Real.sin A) / (Real.sin C + Real.sin B)

-- Law of Sines
axiom law_of_sines_1 : (c - b) / (c - a) = a / (c + b)

-- Law of Cosines
axiom law_of_cosines_1 : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c)

-- Target
theorem find_B : B = Real.pi / 3 := 
sorry

end find_B_l450_450227


namespace easter_egg_battle_probability_l450_450470

theorem easter_egg_battle_probability (players : Type) [fintype players] [decidable_eq players]
  (egg_strength : players → ℕ) (p1 : players) (p2 : players) (n : ℕ) [decidable (p1 ≠ p2)] :
  (∀ i in finset.range n, egg_strength p1 > egg_strength p2) →
  let prob11thWin := 11 / 12 in
  11 / 12 = prob11thWin :=
by sorry

end easter_egg_battle_probability_l450_450470


namespace value_of_one_plus_i_to_the_x_plus_y_l450_450964

-- Define the variables and given conditions
variable (x y : ℝ)
def i := Complex.I
variable (h1 : (x - 2) * i - y = -1 + i)

-- The theorem we aim to prove
theorem value_of_one_plus_i_to_the_x_plus_y : (1 + i) ^ (x + y) = -4 :=
by
  -- Skip the proof
  sorry

end value_of_one_plus_i_to_the_x_plus_y_l450_450964


namespace f_of_g_l450_450282

def g (x : ℝ) : ℝ := 1 - x^2

def f (x : ℝ) : ℝ := if x ≠ 0 then (2 - x) / (x^4) else 0

theorem f_of_g (y : ℝ) (h : g(y) = 1/4) : f(g(y)) = 20 / 9 :=
by sorry

end f_of_g_l450_450282


namespace fleas_after_treatment_l450_450531

theorem fleas_after_treatment
  (F : ℕ)  -- F is the number of fleas the dog has left after the treatments
  (half_fleas : ℕ → ℕ)  -- Function representing halving fleas
  (initial_fleas := F + 210)  -- Initial number of fleas before treatment
  (half_fleas_def : ∀ n, half_fleas n = n / 2)  -- Definition of half_fleas function
  (condition : F = (half_fleas (half_fleas (half_fleas (half_fleas initial_fleas)))))  -- Condition given in the problem
  :
  F = 14 := 
  sorry

end fleas_after_treatment_l450_450531


namespace quadrilateral_angle_bad_l450_450397

noncomputable theory
open_locale classical

theorem quadrilateral_angle_bad 
  (AB BC CD DA : ℝ) (angle_ABC angle_BCD : ℝ) 
  (h1 : AB = BC)
  (h2 : CD = DA)
  (h3 : angle_ABC = 75)
  (h4 : angle_BCD = 165) :
  ∃ BAD : ℝ, BAD = 82.5 :=
by
  use 82.5
  sorry

end quadrilateral_angle_bad_l450_450397


namespace larger_segment_of_triangle_l450_450814

theorem larger_segment_of_triangle (a b c : ℝ) (h : ℝ) (hc : c = 100) (ha : a = 40) (hb : b = 90) 
  (h_triangle : a^2 + h^2 = x^2)
  (h_triangle2 : b^2 + h^2 = (100 - x)^2) :
  100 - x = 82.5 :=
sorry

end larger_segment_of_triangle_l450_450814


namespace find_c_l450_450996

-- Define that X follows normal distribution N(3, 1)
variable {X : ℝ → ℝ}
def normal_dist (μ σ : ℝ) (x : ℝ) : ℝ := 
  1 / (σ * real.sqrt (2 * real.pi)) * real.exp (-(x - μ)^2 / (2 * σ^2))

-- Given: X follows N(3,1)
axiom normal_X : ∀ x, X x = normal_dist 3 1 x

-- Prove: If P(X < 2 * c + 1) = P(X > c + 5), then c = 0.
theorem find_c (c : ℝ) : (∫ x in set.Iic (2 * c + 1), X x) = (∫ x in set.Ioi (c + 5), X x) → c = 0 := 
by
  sorry

end find_c_l450_450996


namespace weird_fraction_implies_weird_power_fraction_l450_450396

theorem weird_fraction_implies_weird_power_fraction 
  (a b c : ℝ) (k : ℕ) 
  (h1 : (1/a) + (1/b) + (1/c) = (1/(a + b + c))) 
  (h2 : Odd k) : 
  (1 / (a^k) + 1 / (b^k) + 1 / (c^k) = 1 / (a^k + b^k + c^k)) := 
by 
  sorry

end weird_fraction_implies_weird_power_fraction_l450_450396


namespace only_pairs_satisfying_conditions_l450_450125

theorem only_pairs_satisfying_conditions (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (b^2 + b + 1) % a = 0 ∧ (a^2 + a + 1) % b = 0 → a = 1 ∧ b = 1 :=
by
  sorry

end only_pairs_satisfying_conditions_l450_450125


namespace error_percent_area_l450_450702

-- Given conditions
variables (L W : ℝ) (hL : L > 0) (hW : W > 0)

-- Definitions derived from conditions
def measured_length := 1.09 * L
def measured_width := 0.92 * W
def actual_area := L * W
def measured_area := measured_length * measured_width

-- The correct answer we want to prove
theorem error_percent_area :
  let error := measured_area - actual_area in
  let error_percent := (error / actual_area) * 100 in
  error_percent = 0.28 := by
  sorry

end error_percent_area_l450_450702


namespace repeated_digits_count_l450_450896

theorem repeated_digits_count : 
  let digits := {0, 1, 2, 3} in
  let total_numbers := 3 * 4 * 4 * 4 in  -- first digit: 3 choices (1, 2, 3), each subsequent digit: 4 choices
  let unique_numbers := 3 * 3 * 2 * 1 in -- first digit: 3 choices (1, 2, 3), next: 3, next: 2, next: 1 
  total_numbers - unique_numbers = 174 :=
by {
  let digits := {0, 1, 2, 3},
  let total_numbers := 3 * 4 * 4 * 4,  -- total valid four-digit numbers
  let unique_numbers := 3 * 3 * 2 * 1, -- valid four-digit numbers with unique digits
  show total_numbers - unique_numbers = 174,
  sorry -- proof to be filled in
}

end repeated_digits_count_l450_450896


namespace circle_through_points_tangent_to_line_l450_450110

theorem circle_through_points_tangent_to_line :
  (∃ a r : ℝ, ((x - 3)^2 + (y - 6)^2 = 20) ∨ ((x + 7)^2 + (y - 6)^2 = 80) ∧
  (1 - a)^2 + (10 - 6)^2 = r^2 ∧
  (a - 1)^2 + 16 = (a - 12)^2 / 5) :=
begin
  sorry
end

end circle_through_points_tangent_to_line_l450_450110


namespace inequality_proof_l450_450732

variables (a b c : ℝ)

theorem inequality_proof
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (cond : a^2 + b^2 + c^2 + ab + bc + ca ≤ 2) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 := 
sorry

end inequality_proof_l450_450732


namespace Joeys_age_is_14_l450_450362

theorem Joeys_age_is_14 (ages : set ℕ) 
  (h1 : ages = {2, 4, 6, 8, 10, 12, 14})
  (h2 : ∃ a b ∈ ages, a + b = 18)
  (h3 : ∃ c d ∈ ages, c < 12 ∧ d < 12 ∧ c ≠ 4 ∧ d ≠ 4)
  (h4 : ∀ j ∈ ages, j = 4 → ¬(j = 2 ∨ j = 6 ∨ j = 8 ∨ j = 10 ∨ j = 14)) :
  (∃ j ∈ ages, j = 14 ∧ (j ≠ 4 ∧ j ≠ 6 ∧ j ≠ 8 ∧ j ≠ 10 ∧ j ≠ 2)) :=
by {
    sorry
}

end Joeys_age_is_14_l450_450362


namespace center_and_radius_of_circle_l450_450415

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 6 * y + 6 = 0

-- State the theorem
theorem center_and_radius_of_circle :
  (∃ x₀ y₀ r, (∀ x y, circle_eq x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
  x₀ = 1 ∧ y₀ = -3 ∧ r = 2) :=
by
  -- Proof is omitted
  sorry

end center_and_radius_of_circle_l450_450415


namespace find_k_b_find_x_when_y_neg_8_l450_450349

theorem find_k_b (k b : ℤ) (h1 : -20 = 4 * k + b) (h2 : 16 = -2 * k + b) : k = -6 ∧ b = 4 := 
sorry

theorem find_x_when_y_neg_8 (x : ℤ) (k b : ℤ) (h_k : k = -6) (h_b : b = 4) (h_target : -8 = k * x + b) : x = 2 := 
sorry

end find_k_b_find_x_when_y_neg_8_l450_450349


namespace least_positive_t_l450_450093

theorem least_positive_t
  (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (ht : ∃ t, 0 < t ∧ (∃ r, (Real.arcsin (Real.sin α) * r = Real.arcsin (Real.sin (3 * α)) ∧ 
                            Real.arcsin (Real.sin (3 * α)) * r = Real.arcsin (Real.sin (5 * α)) ∧
                            Real.arcsin (Real.sin (5 * α)) * r = Real.arcsin (Real.sin (t * α))))) :
  t = 6 :=
sorry

end least_positive_t_l450_450093


namespace tangent_line_circle_l450_450128

theorem tangent_line_circle : 
  ∃ (k : ℚ), (∀ x y : ℚ, ((x - 3) ^ 2 + (y - 4) ^ 2 = 25) 
               → (3 * x + 4 * y - 25 = 0)) :=
sorry

end tangent_line_circle_l450_450128


namespace compute_difference_of_squares_l450_450099

theorem compute_difference_of_squares :
  (23 + 15) ^ 2 - (23 - 15) ^ 2 = 1380 := by
  sorry

end compute_difference_of_squares_l450_450099


namespace coordinates_with_respect_to_origin_l450_450417

theorem coordinates_with_respect_to_origin (x y : ℤ) (h : (x, y) = (3, -2)) : (x, y) = (3, -2) :=
by
  -- Given that the point (x, y) is (3, -2)
  rw h

end coordinates_with_respect_to_origin_l450_450417


namespace least_positive_integer_modular_conditions_l450_450832

theorem least_positive_integer_modular_conditions :
  ∃ n : ℤ, n > 0 ∧ (n % 4 = 1) ∧ (n % 3 = 2) ∧ (n % 5 = 3) ∧ n = 13 :=
by
  use 13
  split
  . norm_num
  split
  . norm_num
  split
  . norm_num
  split
  . norm_num
  norm_num

end least_positive_integer_modular_conditions_l450_450832


namespace fraction_vcr_units_l450_450350

theorem fraction_vcr_units (H : ℕ) (V : ℚ) :
  -- Condition 1: 1/5 of the housing units are equipped with cable television
  let C := (1 : ℚ) / 5 * H in
  -- Condition 2: 1/3 of those that are equipped with cable television are also equipped with videocassette recorders
  let VC := (1 : ℚ) / 3 * C in
  -- Condition 3: 0.7666666666666667 of the housing units have neither cable television nor videocassette recorders
  let N := (0.7666666666666667 : ℚ) * H in
  -- Remaining fraction of housing units must have either cable, videocassette recorders, or both
  (H - N) = (C + V * H - VC) →
  V = 1 / 10 :=
sorry

end fraction_vcr_units_l450_450350


namespace tangent_line_y_intercept_l450_450022

theorem tangent_line_y_intercept (y : ℝ) : 
    let center1 := (3 : ℝ, 0 : ℝ)
    let center2 := (8 : ℝ, 0 : ℝ)
    let radius1 := 3
    let radius2 := 2
    (∀ y, let A := (3, y), B := (8, y), dist A center1 = 3 ∧ dist B center2 = 2 → y = 3)
    → y = 3 := 
by
  intros h
  specialize h 3
  simp [center1, center2, radius1, radius2, dist] at h
  assumption

end tangent_line_y_intercept_l450_450022


namespace kanul_total_amount_l450_450002

theorem kanul_total_amount (T : ℝ) (h1 : 35000 + 40000 + 0.2 * T = T) : T = 93750 := 
by
  sorry

end kanul_total_amount_l450_450002


namespace smallest_N_l450_450258

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450258


namespace minimum_value_of_expression_l450_450685

theorem minimum_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0)
    (hline : ∀ x y : ℝ, (x^2 + y^2 + 8*x + 2*y + 1 = 0) → (a*x + b*y + 1 = 0)) :
    4 * a + b = 1 → ∀ a b > 0, ∀ x y : ℝ, (x^2 + y^2 + 8*x + 2*y + 1 = 0) → (a*x + b*y + 1 = 0) → 
    (x + 4)^2 + (y + 1)^2 = 16 := sorry

end minimum_value_of_expression_l450_450685


namespace limit_tan_exp_div_limit_problem_l450_450908

theorem limit_tan_exp_div (x : ℝ) (h : 0 ≤ x ∧ x < 1):
  (Real.exp x - 1) / x = 1 :=
begin
  sorry
end

theorem limit_problem : 
  (∀ x, 0 ≤ x ∧ x < 1 → 
    (Real.tan (Real.pi / 4 - x))^((Real.exp x - 1) / x) = 1) :=
begin
  intro x,
  intro hx,
  have H1 := limit_tan_exp_div x hx,
  have H2 : Real.tan (Real.pi / 4 - x) = 1,
  {
    rw [Real.tan_sub, Real.tan_pi_div_four],
    simp,
  },
  rw H2,
  rw H1,
  norm_num,
end

end limit_tan_exp_div_limit_problem_l450_450908


namespace inequality_proof_l450_450733

variables (a b c : ℝ)

theorem inequality_proof
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (cond : a^2 + b^2 + c^2 + ab + bc + ca ≤ 2) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 := 
sorry

end inequality_proof_l450_450733


namespace b_geometric_inequality_minimum_m_l450_450622

-- Define the sequences and conditions
def a : ℕ → ℚ
| 1       := 1
| (n + 1) := 1 + 2 / (a n)

def b (n : ℕ) : ℚ := (a n - 2) / (a n + 1)

theorem b_geometric {n : ℕ} : b n = (-1/2)^n :=
sorry

def c (n : ℕ) : ℚ := n * b n

def S (n : ℕ) : ℚ := ∑ i in range (1, n + 1), c i

theorem inequality_minimum_m (m : ℕ) : 
  (∀ n : ℕ, n > 0 → (m : ℚ)/32 + 3/2 * S n + n * (-1/2)^(n + 1) - 1/3 * (-1/2)^n > 0) ↔ m ≥ 11 :=
sorry

end b_geometric_inequality_minimum_m_l450_450622


namespace burattino_suspects_cheating_after_seventh_draw_l450_450080

theorem burattino_suspects_cheating_after_seventh_draw 
  (total_balls : ℕ := 45) (drawn_balls : ℕ := 6) (a : ℝ := ((nat.choose 39 6 : ℕ) : ℝ) / ((nat.choose 45 6 : ℕ) : ℝ))
  (threshold : ℝ := 0.01) (probability : ℝ := 0.4) :
  (∃ n, a^n < threshold) → (∃ n > 5, a^n < threshold) :=
begin
  -- Definitions from conditions
  have fact_prob : a = ((nat.choose 39 6 : ℕ) : ℝ) / ((nat.choose 45 6 : ℕ) : ℝ), by refl,
  have fact_approx : a ≈ probability, by simp,

  -- Statement to prove
  intros h,
  use 6,
  split,
  { linarith, },
  { sorry }
end

end burattino_suspects_cheating_after_seventh_draw_l450_450080


namespace find_n_value_l450_450592

theorem find_n_value (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 9) : n = 210 := sorry

end find_n_value_l450_450592


namespace comb_sum_l450_450091

-- Define the combination function C(n, k)
def comb (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Define the statement to prove C(99, 2) + C(99, 3) = 161700
theorem comb_sum : comb 99 2 + comb 99 3 = 161700 := by
  sorry

end comb_sum_l450_450091


namespace number_of_initial_values_l450_450106

noncomputable def f (x : ℝ) : ℝ := 6 * x - x ^ 2

def seq (x0 : ℝ) : ℕ → ℝ
| 0       => x0
| (n + 1) => f (seq n)

def finiteValues (x0 : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n m : ℕ, n ≥ N → m ≥ N → seq x0 n = seq x0 m

theorem number_of_initial_values (h : {x0 : ℝ // finiteValues x0 }) : 3 :=
  sorry

end number_of_initial_values_l450_450106


namespace minimize_CP_l450_450348

variables (a b c : ℝ)
variables (hab : 0 < a) (hbc : 0 < b) (hc : 0 < c)

def AB := 2 * a
def AD := 2 * b
def AF := 2 * c

theorem minimize_CP :
  let CP_min := (4 * c * Real.sqrt (a^2 + b^2)) / (Real.sqrt (a^2 + b^2 + 4 * c^2)) in
    ∃ CP : ℝ, CP = CP_min
  :=
sorry

end minimize_CP_l450_450348


namespace probability_of_winning_11th_round_l450_450478

-- Definitions of the conditions
def player1_wins_ten_rounds (eggs : List ℕ) : Prop :=
  ∀ i, i < 10 → eggs.indexOf (eggs.nthLe 0 (i+1)) < eggs.indexOf (eggs.nthLe 1 (i+1))

def is_strongest (egg : ℕ) (eggs : List ℕ) : Prop :=
  egg = List.maximum (0 :: eggs)

-- The proof to show the probability of winning the 11th round
theorem probability_of_winning_11th_round
  (eggs : List ℕ) : player1_wins_ten_rounds eggs →
  (1 - 1 / (length eggs + 1) = 11 / 12) :=
by
  sorry

end probability_of_winning_11th_round_l450_450478


namespace valid_third_side_length_l450_450174

theorem valid_third_side_length {x : ℝ} (h1 : 5 + 8 > x) (h2 : 5 + x > 8) (h3 : 8 + x > 5) : x = 6 :=
by
  -- Given 5 + 8 > x, 5 + x > 8, 8 + x > 5
  have range1 : 13 > x := h1,
  have range2 : x > 3 := (by linarith [h2]),
  have _ : 3 < 6 ∧ 6 < 13 := by norm_num,
  linarith

#check valid_third_side_length

end valid_third_side_length_l450_450174


namespace find_m_invariant_interval_l450_450617

-- Definitions
def g (x : ℝ) (m : ℝ) : ℝ := x + m + Real.log x
def invariant_interval (g : ℝ → ℝ) (interval : Set ℝ) : Prop :=
  ∀ x ∈ interval, g x ∈ interval

-- Theorem statement
theorem find_m_invariant_interval (m : ℝ) :
  invariant_interval (λ x, g x m) (Set.Ici (Real.exp 1)) → m = -1 :=
sorry

end find_m_invariant_interval_l450_450617


namespace largest_x_l450_450598

theorem largest_x (x : ℝ) (h : ⌊x⌋ / x = 7 / 8) : x = 48 / 7 := 
sorry

end largest_x_l450_450598


namespace possible_third_side_l450_450189

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end possible_third_side_l450_450189


namespace gift_boxes_in_3_days_l450_450941
-- Conditions:
def inchesPerBox := 18
def dailyWrapper := 90
-- "how many gift boxes will he be able to wrap every 3 days?"
theorem gift_boxes_in_3_days : 3 * (dailyWrapper / inchesPerBox) = 15 :=
by
  sorry

end gift_boxes_in_3_days_l450_450941


namespace count_prime_diff_numbers_l450_450670

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_prime_diff (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p - q

def in_target_set (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 + 10 * k

def prime_diff_count : ℕ :=
  (Finset.filter (λ n => is_prime_diff n) (Finset.range (100))).card  -- assuming we consider first 100 numbers for simplicity

theorem count_prime_diff_numbers :
  ∃ k : ℕ, prime_diff_count = k :=
begin
  sorry
end

end count_prime_diff_numbers_l450_450670


namespace cannot_determine_b_l450_450991

theorem cannot_determine_b 
  (a b c d : ℝ) 
  (h_avg : (a + b + c + d) / 4 = 12.345) 
  (h_ineq : a > b ∧ b > c ∧ c > d) : 
  ¬((b = 12.345) ∨ (b > 12.345) ∨ (b < 12.345)) :=
sorry

end cannot_determine_b_l450_450991


namespace compute_difference_of_squares_l450_450100

theorem compute_difference_of_squares :
  (23 + 15) ^ 2 - (23 - 15) ^ 2 = 1380 := by
  sorry

end compute_difference_of_squares_l450_450100


namespace smallest_possible_N_l450_450264

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l450_450264


namespace triangle_third_side_length_l450_450191

theorem triangle_third_side_length (a b : ℝ) (x : ℝ) (h₁ : a = 5) (h₂ : b = 8) (hx : x ∈ {2, 3, 6, 13}) :
  3 < x ∧ x < 13 → x = 6 :=
by sorry

end triangle_third_side_length_l450_450191


namespace median_of_100_numbers_l450_450307

theorem median_of_100_numbers (x : Fin 100 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i = 78 → x j = 66 → i = 51 ∧ j = 50 ∨ i = 50 ∧ j = 51)
  (h2 : ∀ i, i ≠ 51 → x 51 = 78)
  (h3 : ∀ i, i ≠ 50 → x 50 = 66) :
  (x 50 + x 51) / 2 = 72 :=
by sorry

end median_of_100_numbers_l450_450307


namespace B_pow_97_l450_450365

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem B_pow_97 : B ^ 97 = B := by
  sorry

end B_pow_97_l450_450365


namespace smallest_N_value_proof_l450_450247

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l450_450247


namespace median_of_100_set_l450_450317

theorem median_of_100_set 
  (S : Finset ℝ) (h_card : S.card = 100)
  (h_remove1 : ∃ x ∈ S, median (S.erase x) = 78)
  (h_remove2 : ∃ y ∈ S, median (S.erase y) = 66) : 
  median S = 72 :=
by
  sorry

end median_of_100_set_l450_450317


namespace price_of_each_brownie_l450_450356

variable (B : ℝ)

theorem price_of_each_brownie (h : 4 * B + 10 + 28 = 50) : B = 3 := by
  -- proof steps would go here
  sorry

end price_of_each_brownie_l450_450356


namespace possible_third_side_l450_450185

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end possible_third_side_l450_450185


namespace burattino_suspects_cheating_after_seventh_draw_l450_450065

theorem burattino_suspects_cheating_after_seventh_draw
  (balls : ℕ)
  (draws : ℕ)
  (a : ℝ)
  (p_limit : ℝ)
  (h_balls : balls = 45)
  (h_draws : draws = 6)
  (h_a : a = (39.choose 6 : ℝ) / (45.choose 6 : ℝ))
  (h_p_limit : p_limit = 0.01) :
  ∃ (n : ℕ), n > 5 ∧ a^n < p_limit := by
  sorry

end burattino_suspects_cheating_after_seventh_draw_l450_450065


namespace satisfies_equation_l450_450114

theorem satisfies_equation (a b c : ℤ) (h₁ : a = b) (h₂ : b = c + 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 3 := 
by 
  sorry

end satisfies_equation_l450_450114


namespace valid_third_side_length_l450_450178

theorem valid_third_side_length {x : ℝ} (h1 : 5 + 8 > x) (h2 : 5 + x > 8) (h3 : 8 + x > 5) : x = 6 :=
by
  -- Given 5 + 8 > x, 5 + x > 8, 8 + x > 5
  have range1 : 13 > x := h1,
  have range2 : x > 3 := (by linarith [h2]),
  have _ : 3 < 6 ∧ 6 < 13 := by norm_num,
  linarith

#check valid_third_side_length

end valid_third_side_length_l450_450178


namespace expected_value_is_7_l450_450875

def win (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2 * (10 - n) else 10 - n

def fair_die_values := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def expected_value (values : List ℕ) (win : ℕ → ℕ) : ℚ :=
  (values.map (λ n => win n)).sum / values.length

theorem expected_value_is_7 :
  expected_value fair_die_values win = 7 := 
sorry

end expected_value_is_7_l450_450875


namespace find_QT_l450_450718

noncomputable def QT_value : ℚ := 3825 / 481

structure Triangle :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (R : ℝ × ℝ)
  (PQ : ℝ := (dist P Q))
  (QR : ℝ := (dist Q R))
  (PR : ℝ := (dist P R))

axiom given_triangle : Triangle

axiom PQ_eq : given_triangle.PQ = 15
axiom QR_eq : given_triangle.QR = 17
axiom PR_eq : given_triangle.PR = 16

structure PointsOnQR :=
  (S : ℝ × ℝ)
  (T : ℝ × ℝ)
  (RS : ℝ := (dist given_triangle.R S))
  (QS : ℝ := (dist given_triangle.Q S))
  (RT : ℝ := (dist given_triangle.R T))
  (QT : ℝ := (dist given_triangle.Q T))
  (angle_PQT_EQ_PSR : ∠PQT = ∠PSR)

axiom S_on_QR : PointsOnQR.S
axiom T_on_QR : PointsOnQR.T
axiom RS_eq : PointsOnQR.RS = 7
axiom angle_equality : PointsOnQR.angle_PQT_EQ_PSR

theorem find_QT : PointsOnQR.QT = QT_value := 
sorry

end find_QT_l450_450718


namespace sum_of_square_areas_l450_450719

theorem sum_of_square_areas (XY YZ : ℝ)
  (hXY : XY = 5)
  (hYZ : YZ = 12)
  (right_triangle : ∃ XZ, XZ^2 = XY^2 + YZ^2) :
  (XY^2 + YZ^2 + (classical.some right_triangle)^2) = 338 := by
suppress
  -- Sorry, add the proof here later.

end sum_of_square_areas_l450_450719


namespace find_k_l450_450604

noncomputable def series_sum (k : ℝ) : ℝ :=
  5 + (5 + k) / 5 + (5 + 2 * k) / 5^2 + (5 + 3 * k) / 5^3 + ∑' n, (5 + (n + 4) * k) / 5^(n + 4)

theorem find_k : ∃ k : ℝ, series_sum k = 12 ∧ (k.round = 18) :=
sorry

end find_k_l450_450604


namespace find_pairs_l450_450593

theorem find_pairs (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n)
  (h3 : (m^2 - n) ∣ (m + n^2)) (h4 : (n^2 - m) ∣ (n + m^2)) :
  (m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 3) := by
  sorry

end find_pairs_l450_450593


namespace compute_expression_l450_450914

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end compute_expression_l450_450914


namespace monotonic_intervals_l450_450430

def f (x : ℝ) : ℝ := |x^2 - 2 * x - 3|

theorem monotonic_intervals :
  (∀ x y : ℝ, -1 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y) ∧
  (∀ x y : ℝ, 3 ≤ x ∧ x ≤ y → f x ≤ f y) :=
begin
  sorry
end

end monotonic_intervals_l450_450430


namespace inclination_angle_l450_450447

theorem inclination_angle : 
  ∀ (l : Line) (origin : Point) (p : Point),
  l.passes_through origin ∧ l.passes_through p ∧ origin = (0, 0) ∧ p = (1, -1) → 
  l.angle_of_inclination = 135 := 
by 
  sorry

end inclination_angle_l450_450447


namespace smallest_possible_value_of_N_l450_450239

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l450_450239


namespace probability_of_winning_11th_round_l450_450476

-- Definitions of the conditions
def player1_wins_ten_rounds (eggs : List ℕ) : Prop :=
  ∀ i, i < 10 → eggs.indexOf (eggs.nthLe 0 (i+1)) < eggs.indexOf (eggs.nthLe 1 (i+1))

def is_strongest (egg : ℕ) (eggs : List ℕ) : Prop :=
  egg = List.maximum (0 :: eggs)

-- The proof to show the probability of winning the 11th round
theorem probability_of_winning_11th_round
  (eggs : List ℕ) : player1_wins_ten_rounds eggs →
  (1 - 1 / (length eggs + 1) = 11 / 12) :=
by
  sorry

end probability_of_winning_11th_round_l450_450476


namespace find_angle_between_l450_450146

variables (a b : EuclideanSpace ℝ (Fin 3)) -- Assuming 3D space for vectors a and b
variables (angle_between : EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) → ℝ)

noncomputable def angle_between (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  real.arccos ((inner_product_space.toInner ℝ (Fin 3)).inner a b / (norm a * norm b))

theorem find_angle_between 
  (ha : ∥a∥ = 5) 
  (hb : ∥b∥ = 4) 
  (hab : inner_product_space.toInner ℝ (Fin 3) a b = -10) : 
  angle_between a b = (2 * Real.pi / 3) :=
by
  sorry

end find_angle_between_l450_450146


namespace valid_years_count_l450_450107

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string
  s = s.reverse

noncomputable def is_prime (n : ℕ) : Prop :=
  Prime n

noncomputable def is_valid_year (n : ℕ) : Prop :=
  is_palindrome n ∧ ∃ p q : ℕ, p < 100 ∧ p ≥ 10 ∧ q < 1000 ∧ q ≥ 100 ∧ is_prime p ∧ is_prime q ∧ n = p * q

theorem valid_years_count : (Finset.filter is_valid_year (Finset.range' 1900 101)).card = 4 := by
  sorry

end valid_years_count_l450_450107


namespace smallest_N_value_proof_l450_450245

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l450_450245


namespace parallel_vectors_implies_value_of_x_l450_450224

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the condition for parallel vectors
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (u.1 = k * v.1) ∧ (u.2 = k * v.2)

-- The proof statement
theorem parallel_vectors_implies_value_of_x : ∀ (x : ℝ), parallel a (b x) → x = 6 :=
by
  intro x
  intro h
  sorry

end parallel_vectors_implies_value_of_x_l450_450224


namespace determine_odd_and_monotonically_increasing_l450_450929

/-- Define the four functions and their respective domains --/
def f_A (x : ℝ) : ℝ := -1 / x
def f_B (x : ℝ) : ℝ := -Real.log x / Real.log 2
def f_C (x : ℝ) : ℝ := 3 ^ x
def f_D (x : ℝ) : ℝ := x ^ 3

/-- Define the domains of the functions --/
def domain_A (x : ℝ) : Prop := x ≠ 0
def domain_B (x : ℝ) : Prop := 0 < x
def domain_C (x : ℝ) : Prop := True
def domain_D (x : ℝ) : Prop := True

/-- Define when a function is odd --/
def is_odd (f : ℝ → ℝ) (domain : ℝ → Prop) : Prop :=
∀ x : ℝ, domain x → f (-x) = -f x

/-- Define when a function is monotonically increasing --/
def is_monotonically_increasing (f : ℝ → ℝ) (domain : ℝ → Prop) : Prop :=
∀ x y : ℝ, domain x → domain y → x < y → f x < f y

/-- Main theorem --/
theorem determine_odd_and_monotonically_increasing :
  (is_odd f_A domain_A ∧ is_monotonically_increasing f_A domain_A) ∨
  (is_odd f_B domain_B ∧ is_monotonically_increasing f_B domain_B) ∨
  (is_odd f_C domain_C ∧ is_monotonically_increasing f_C domain_C) ∨
  (is_odd f_D domain_D ∧ is_monotonically_increasing f_D domain_D) ↔
  (is_odd f_D domain_D ∧ is_monotonically_increasing f_D domain_D) :=
by
  sorry

end determine_odd_and_monotonically_increasing_l450_450929


namespace find_angle_DFE_l450_450011

theorem find_angle_DFE
  (A B C D E F : Point)
  (hD : D ∈ Line.segment A B)
  (hE : E ∈ Line.segment A C)
  (hF_bisector_BDE : Angle.bisector F D E)
  (hF_bisector_CED : Angle.bisector F E D)
  (hBAC : Angle A B C = 38) :
  Angle D F E = 71 :=
sorry

end find_angle_DFE_l450_450011


namespace maximum_right_angles_convex_polygon_l450_450800

theorem maximum_right_angles_convex_polygon (P : Type) [Polygon P] (h1 : sum_exterior_angles P = 360) :
  right_angles_in_polygon P ≤ 4 :=
sorry

end maximum_right_angles_convex_polygon_l450_450800


namespace find_alpha_l450_450217

def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

def parametric_line (t α : ℝ) : ℝ × ℝ :=
  (1 + t * cos α, t * sin α)

def circle_eq (x y : ℝ) :=
  (x - 2) ^ 2 + y ^ 2 = 4

def distance (A B : ℝ × ℝ) : ℝ :=
  let dx := A.1 - B.1
  let dy := A.2 - B.2
  sqrt (dx^2 + dy^2)

theorem find_alpha {α : ℝ} (t₁ t₂ : ℝ) : 
  (parametric_line t₁ α).1 = 1 + t₁ * cos α → 
  (parametric_line t₁ α).2 = t₁ * sin α → 
  (parametric_line t₂ α).1 = 1 + t₂ * cos α → 
  (parametric_line t₂ α).2 = t₂ * sin α → 
  distance (parametric_line t₁ α) (parametric_line t₂ α) = sqrt 14 →
  (α = π / 4 ∨ α = 3 * π / 4) :=
by
  sorry

end find_alpha_l450_450217


namespace simplify_expression_l450_450774

theorem simplify_expression : 8 * (15 / 4) * (-56 / 45) = -112 / 3 :=
by sorry

end simplify_expression_l450_450774


namespace burattino_suspects_after_seventh_draw_l450_450059

noncomputable def probability (total : ℕ) (choose : ℕ) : ℚ := 
  (nat.factorial total / (nat.factorial choose * nat.factorial (total - choose))) 

noncomputable def suspicion_threshold (threshold : ℚ) (probability_per_draw : ℚ) : ℕ :=
  nat.find (λ n, probability_per_draw^n < threshold)

theorem burattino_suspects_after_seventh_draw :
  let a := (probability 39 6) / (probability 45 6) in
  let threshold := (1 : ℚ) / 100 in
  suspicion_threshold threshold a = 6 :=
by
  sorry

end burattino_suspects_after_seventh_draw_l450_450059


namespace player1_wins_11th_round_l450_450465

theorem player1_wins_11th_round (player1_wins_first_10 : ∀ (round : ℕ), round < 10 → player1_wins round) : 
  prob_winning_11th_round player1 = 11 / 12 :=
sorry

end player1_wins_11th_round_l450_450465


namespace smallest_N_value_proof_l450_450251

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end smallest_N_value_proof_l450_450251


namespace final_cash_amounts_l450_450820

-- Initial conditions
variables (A_initial_cash : ℕ) (B_initial_cash : ℕ) (C_initial_value_of_house : ℕ)
variables (value_of_car : ℕ) (value_of_house : ℕ)
variables (transaction1_amount : ℕ) (transaction2_amount : ℕ)
variables (transaction3_amount : ℕ) (transaction4_amount : ℕ)

-- Proof that verifies the final cash amounts
theorem final_cash_amounts :
  let A_final_cash := A_initial_cash + transaction1_amount - transaction3_amount - transaction4_amount,
      B_final_cash := B_initial_cash - transaction1_amount + transaction2_amount,
      C_final_cash := transaction3_amount + transaction4_amount
  in A_initial_cash = 10000 ∧ B_initial_cash = 15000 ∧ C_initial_value_of_house = 10000 ∧
     value_of_car = 5000 ∧ value_of_house = 10000 ∧
     transaction1_amount = 6000 ∧ transaction2_amount = 11000 ∧
     transaction3_amount = 4500 ∧ transaction4_amount = 9000 →
  A_final_cash = 2500 ∧ B_final_cash = 20000 ∧ C_final_cash = 13500 :=
by
  intros A_initial_cash_eq B_initial_cash_eq C_initial_value_of_house_eq
         value_of_car_eq value_of_house_eq
         transaction1_amount_eq transaction2_amount_eq
         transaction3_amount_eq transaction4_amount_eq,
  sorry

end final_cash_amounts_l450_450820


namespace square_circle_area_ratio_l450_450041

theorem square_circle_area_ratio {r : ℝ} (h_pos : r > 0) 
  (side_has_chord : ∀ (s : ℝ), (∃ x, x = s / 2) → (∃ y, y = r / 2)) :
  let A_square := (√15 * r / 2) ^ 2 in
  let A_circle := π * r ^ 2 in
  A_square / A_circle = 15 / (4 * π) :=
by
  have h1 : A_square = 15 * r^2 / 4 := sorry
  have h2 : A_circle = π * r^2 := sorry
  have h3 : A_square / A_circle = (15*r^2 / 4) / (π * r^2) := by sorry
  have h4 : (15*r^2 / 4) / (π * r^2) = 15 / (4 * π) := by sorry
  exact h4

end square_circle_area_ratio_l450_450041


namespace find_angles_and_k_l450_450812

noncomputable def angle_1 (k : ℝ) : ℝ :=
  real.arccos ((k + real.sqrt (k^2 + 4)) / 4)

noncomputable def angle_2 (k : ℝ) : ℝ :=
  real.arccos (real.sqrt (k^2 + 4) / 30)

theorem find_angles_and_k (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 = angle_1 k ∧ x2 = angle_2 k) ∧
  (2 / (2 * real.sqrt 3) < k) ∧ (k < 3) :=
sorry

end find_angles_and_k_l450_450812


namespace Jim_catches_Bob_in_20_minutes_l450_450905

theorem Jim_catches_Bob_in_20_minutes
  (Bob_Speed : ℕ := 6)
  (Jim_Speed : ℕ := 9)
  (Head_Start : ℕ := 1) :
  (Head_Start / (Jim_Speed - Bob_Speed) * 60 = 20) :=
by
  sorry

end Jim_catches_Bob_in_20_minutes_l450_450905


namespace tenth_term_is_correct_l450_450410

def sequence (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)

theorem tenth_term_is_correct : sequence 10 = 20 / 21 := by
  sorry

end tenth_term_is_correct_l450_450410


namespace smallest_N_l450_450259

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450259


namespace compute_one_minus_i_pow_six_l450_450920

theorem compute_one_minus_i_pow_six : (1 - (complex.I)) ^ 6 = 8 * (complex.I) := 
by
  sorry

end compute_one_minus_i_pow_six_l450_450920


namespace polynomial_bound_l450_450740

theorem polynomial_bound (n : ℕ) (a : Fin n.succ → ℝ) :
  ∃ k : Fin n.succ, ∀ x : ℝ, 0 ≤ x → x ≤ 1 → 
    ∑ i in Finset.range n.succ, a i * x ^ (i : ℕ) ≤ ∑ i in Finset.range (k + 1), a i :=
by sorry

end polynomial_bound_l450_450740


namespace new_profit_percentage_l450_450054

theorem new_profit_percentage (
  original_price : ℝ,
  original_selling_price : ℝ := 1100,
  original_profit_rate : ℝ := 0.10,
  extra_amount : ℝ := 70
) : 
∃ new_profit_rate : ℝ, 
  original_selling_price = (1 + original_profit_rate) * original_price ∧
  original_price = 1000 ∧
  new_profit_rate = 0.30 := 
by
  let original_price := 1000
  have h1: original_selling_price = 1100 
    := rfl
  have h2: new_profit_rate = 0.30 
    := rfl
  exact ⟨new_profit_rate, h1, rfl, h2⟩

end new_profit_percentage_l450_450054


namespace trajectory_equation_l450_450968

variable {x y λ : ℝ} -- variables x, y, and λ are real numbers

-- Definitions for the conditions
def isEllipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1
def slopeProductCondition (x y λ : ℝ) : Prop := (λ ≠ 0) ∧ (y / (x + 1) * y / (x - 1) = λ)

-- The theorem we need to prove
theorem trajectory_equation (h_ellipse : isEllipse x y) (h_slopeProduct : slopeProductCondition x y λ) :
  x^2 - y^2 / λ = 1 :=
sorry

end trajectory_equation_l450_450968


namespace solutions_of_quadratic_l450_450737

theorem solutions_of_quadratic 
  (p q : ℚ) 
  (h₁ : 2 * p * p + 11 * p - 21 = 0) 
  (h₂ : 2 * q * q + 11 * q - 21 = 0) : 
  (p - q) * (p - q) = 289 / 4 := 
sorry

end solutions_of_quadratic_l450_450737


namespace sum_mnp_l450_450103

noncomputable def volume_of_parallelepiped := 2 * 3 * 4
noncomputable def volume_of_extended_parallelepipeds := 
  2 * (1 * 2 * 3 + 1 * 2 * 4 + 1 * 3 * 4)
noncomputable def volume_of_quarter_cylinders := 
  4 * (1 / 4 * Real.pi * 1^2 * (2 + 3 + 4))
noncomputable def volume_of_spherical_octants := 
  8 * (1 / 8 * (4 / 3) * Real.pi * 1^3)

noncomputable def total_volume := 
  volume_of_parallelepiped + volume_of_extended_parallelepipeds + 
  volume_of_quarter_cylinders + volume_of_spherical_octants

theorem sum_mnp : 228 + 85 + 3 = 316 := by
  sorry

end sum_mnp_l450_450103


namespace semicircle_area_correct_l450_450860

noncomputable def semicircle_area (l w : ℝ) : ℝ :=
  let d := real.sqrt(l^2 + (2*w)^2) / 2
  (real.pi * d^2) / 2

theorem semicircle_area_correct :
  semicircle_area 3 1 = (13 * real.pi) / 8 :=
begin
  unfold semicircle_area,
  norm_num,
  simp [real.sqrt_eq_rpow],
  ring,
  simp [real.rpow_one_half],
  sorry -- You can replace this with an actual proof
end

end semicircle_area_correct_l450_450860


namespace find_side_length_l450_450427

noncomputable def problem_statement: Prop :=
  let α := real.arctan (3 / 4)
  let a := 16
  let h := a * real.sqrt 3 / 8
  let S₁ := a^2 * real.sqrt 3 / 16
  let S₂ := a^2 * real.sqrt 3 / 64
  let S₃ := 3 * ((3 * a * h) / 16)
  let S₄ := 3 * (a^2 * real.sqrt 3 / 128)
  let total_surface_area := S₁ + S₂ + S₃ + S₄
  total_surface_area = 53 * real.sqrt 3

theorem find_side_length 
  (α := real.arctan (3 / 4))
  (a := 16)
  (M N K F P R : ℝ) -- Representing points for the context of their midpoints
  (h := a * real.sqrt 3 / 8) :
  problem_statement :=
by 
  let S_MNK := a^2 * real.sqrt 3 / 16
  let S_FPR := a^2 * real.sqrt 3 / 64
  let S_MPN := 3 * (3 * a * h / 16)
  let S_FPM := 3 * (a^2 * real.sqrt 3 / 128)
  let total_surface_area := S_MNK + S_FPR + S_MPN + S_FPM
  have h_eq : h = a * real.sqrt 3 / 8 := rfl
  -- Given constraints total surface area
  have surface_area_eq : total_surface_area = 53 * real.sqrt 3 := by 
    sorry
  -- Side length confirmation
  have a_eq : a = 16 := rfl
  surface_area_eq


end find_side_length_l450_450427


namespace find_a_monotonic_intervals_find_b_range_l450_450637

-- Define the function and its derivative
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * real.log (1 + x) + x^2 - 10 * x

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a / (1 + x) + 2 * x - 10

theorem find_a (a : ℝ) : x = 3 → f' a 3 = 0 → a = 16 :=
by sorry

theorem monotonic_intervals (a : ℝ) :
  a = 16 → 
  (∀ x, (-1 < x ∧ x < 1) ∨ (3 < x ∧ x < ∞) → f' 16 x > 0) ∧ 
  (∀ x, (1 < x ∧ x < 3) → f' 16 x < 0) :=
by sorry

theorem find_b_range (b : ℝ) : 
  (32 * real.log 2 - 21 < b ∧ b < 16 * real.log 2 - 9) :=
by sorry

end find_a_monotonic_intervals_find_b_range_l450_450637


namespace triangle_side_lengths_condition_l450_450145

noncomputable def f (x k : ℝ) : ℝ := (x^2 + k*x + 1) / (x^2 + x + 1)

theorem triangle_side_lengths_condition (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, x1 > 0 → x2 > 0 → x3 > 0 →
    (f x1 k) + (f x2 k) > (f x3 k) ∧ (f x2 k) + (f x3 k) > (f x1 k) ∧ (f x3 k) + (f x1 k) > (f x2 k))
  ↔ (-1/2 ≤ k ∧ k ≤ 4) :=
by
  sorry

end triangle_side_lengths_condition_l450_450145


namespace ratio_female_to_male_l450_450900

variables (f m : ℕ)
variables (age_female age_male age_total : ℚ)

-- Conditions
def average_age_female := 35
def average_age_male := 30
def average_age_total := 32

-- Sum of ages
def sum_ages_female := f * average_age_female
def sum_ages_male := m * average_age_male

-- Sum of all ages
def sum_ages := sum_ages_female + sum_ages_male

-- Total number of members
def total_members := f + m

-- Average age equation
def avg_age_equation := sum_ages = average_age_total * total_members

-- Theorem: Ratio of female to male members
theorem ratio_female_to_male : (sum_ages_female = 35 * f ∧ 
                               sum_ages_male = 30 * m ∧ 
                               avg_age_equation = 32 * total_members) → 
                               (f : ℚ) / (m : ℚ) = 2 / 3 :=
by
  sorry

end ratio_female_to_male_l450_450900


namespace magnitude_product_l450_450583

def z1 : ℂ := (3 * Real.sqrt 5) - 3 * Complex.I
def z2 : ℂ := Real.sqrt 7 + 7 * Complex.I

theorem magnitude_product :
  abs (z1 * z2) = 12 * Real.sqrt 21 :=
by
  -- we leave the proof sketch here
  sorry

end magnitude_product_l450_450583


namespace exist_two_same_remainder_l450_450559

theorem exist_two_same_remainder (n : ℕ) (h_pos : 0 < n) :
  ∃ i j : ℕ, 1 ≤ i ∧ i ≤ 2 * n ∧ 1 ≤ j ∧ j ≤ 2 * n ∧ i ≠ j ∧
  (i + f i) % (2 * n) = (j + f j) % (2 * n) :=
sorry

end exist_two_same_remainder_l450_450559


namespace triangle_equilateral_l450_450288

def is_triangle (a b c : ℂ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ c ≠ a

noncomputable def omega : ℂ := -1/2 + (real.sqrt 3)/2 * complex.I

theorem triangle_equilateral (a b c : ℂ) (h_distinct: is_triangle a b c) 
  (h_omega: omega = -1/2 + (real.sqrt 3)/2 * complex.I) 
  (h_relation: a + omega * b + omega^2 * c = 0) :
  (abs (a - b) = abs (b - c)) ∧ (abs (b - c) = abs (c - a)) := 
sorry

end triangle_equilateral_l450_450288


namespace james_spends_252_per_week_l450_450721

noncomputable def cost_pistachios_per_ounce := 10 / 5
noncomputable def cost_almonds_per_ounce := 8 / 4
noncomputable def cost_walnuts_per_ounce := 12 / 6

noncomputable def daily_consumption_pistachios := 30 / 5
noncomputable def daily_consumption_almonds := 24 / 4
noncomputable def daily_consumption_walnuts := 18 / 3

noncomputable def weekly_consumption_pistachios := daily_consumption_pistachios * 7
noncomputable def weekly_consumption_almonds := daily_consumption_almonds * 7
noncomputable def weekly_consumption_walnuts := daily_consumption_walnuts * 7

noncomputable def weekly_cost_pistachios := weekly_consumption_pistachios * cost_pistachios_per_ounce
noncomputable def weekly_cost_almonds := weekly_consumption_almonds * cost_almonds_per_ounce
noncomputable def weekly_cost_walnuts := weekly_consumption_walnuts * cost_walnuts_per_ounce

noncomputable def total_weekly_cost := weekly_cost_pistachios + weekly_cost_almonds + weekly_cost_walnuts

theorem james_spends_252_per_week :
  total_weekly_cost = 252 := by
  sorry

end james_spends_252_per_week_l450_450721


namespace larger_number_l450_450462

theorem larger_number (x y : ℕ) (h1 : x + y = 47) (h2 : x - y = 3) : max x y = 25 :=
sorry

end larger_number_l450_450462


namespace find_f_prime_3_plus_f_3_l450_450735

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≤ 2 then 
    Real.exp x 
  else 
    f (4 - x)

theorem find_f_prime_3_plus_f_3 : (deriv f 3) + f 3 = 0 := 
sorry

end find_f_prime_3_plus_f_3_l450_450735


namespace smallest_possible_value_of_N_l450_450235

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l450_450235


namespace infinite_rational_points_in_region_l450_450953

theorem infinite_rational_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), 
  (∀ p ∈ S, (p.1 ^ 2 + p.2 ^ 2 ≤ 16) ∧ (p.1 ≤ 3) ∧ (p.2 ≤ 3) ∧ (p.1 > 0) ∧ (p.2 > 0)) ∧
  Set.Infinite S :=
sorry

end infinite_rational_points_in_region_l450_450953


namespace Taimour_painting_time_l450_450359

theorem Taimour_painting_time (T : ℝ) 
  (h1 : ∀ (T : ℝ), Jamshid_time = 0.5 * T) 
  (h2 : (1 / T + 2 / T) * 7 = 1) : 
    T = 21 :=
by
  sorry

end Taimour_painting_time_l450_450359


namespace x_greater_than_y_l450_450661

theorem x_greater_than_y (x y z : ℝ) (h1 : x + y + z = 28) (h2 : 2 * x - y = 32) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) : 
  x > y :=
by 
  sorry

end x_greater_than_y_l450_450661


namespace polygon_sides_l450_450027

theorem polygon_sides (x : ℕ) (h₁ : convex) (h₂ : ∀ i j, (interior_angle i = 150 ∨ interior_angle j = 120) → ((interior_angle i = 150 ∧ interior_angle j = 150) ∨ interior_angle j ≠ 150)): 
  2 * 150 + (x - 2) * 120 = 180 * (x - 2) → x = 7 :=
sorry

end polygon_sides_l450_450027


namespace player1_wins_11th_round_l450_450487

noncomputable def egg_strength_probability (n : ℕ) : ℚ :=
  (n - 1) / n

theorem player1_wins_11th_round :
  let player1_wins_first_10_rounds := true,
      total_rounds := 11,
      new_egg := 12 in
  player1_wins_first_10_rounds → egg_strength_probability total_rounds = 11 / 12 :=
by
  intros
  exact sorry

end player1_wins_11th_round_l450_450487


namespace sum_of_squares_five_consecutive_ints_not_perfect_square_l450_450509

theorem sum_of_squares_five_consecutive_ints_not_perfect_square (n : ℤ) :
  ∀ k : ℤ, k^2 ≠ 5 * (n^2 + 2) := 
sorry

end sum_of_squares_five_consecutive_ints_not_perfect_square_l450_450509


namespace range_of_slope_angle_l450_450647

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x + 5

theorem range_of_slope_angle :
  ∀ (x : ℝ), let α := real.arctan (3 * x^2 - 6 * x + 2) in
  0 ≤ α ∧ α < (real.pi / 2) ∨ (3 * real.pi / 4) ≤ α ∧ α < real.pi :=
begin
  sorry
end

end range_of_slope_angle_l450_450647


namespace find_number_l450_450500

theorem find_number :
  ∃ x : ℕ, (8 * x + 5400) / 12 = 530 ∧ x = 120 :=
by
  sorry

end find_number_l450_450500


namespace amount_y_gets_each_rupee_x_gets_l450_450043

-- Given conditions
variables (x y z a : ℝ)
variables (h_y_share : y = 36) (h_total : x + y + z = 156) (h_z : z = 0.50 * x)

-- Proof problem
theorem amount_y_gets_each_rupee_x_gets (h : 36 / x = a) : a = 9 / 20 :=
by {
  -- The proof is omitted and replaced with 'sorry'.
  sorry
}

end amount_y_gets_each_rupee_x_gets_l450_450043


namespace prob_digit_three_in_repeating_block_of_18_div_23_l450_450388

theorem prob_digit_three_in_repeating_block_of_18_div_23 :
  let repeating_block := "78260869565217391304347826086"
  in (repeating_block.toList.filter (λ ch, ch = '3')).length / repeating_block.length = 3 / 26 := sorry

end prob_digit_three_in_repeating_block_of_18_div_23_l450_450388


namespace cos_seven_pi_six_l450_450587

theorem cos_seven_pi_six : (Real.cos (7 * Real.pi / 6) = - Real.sqrt 3 / 2) :=
sorry

end cos_seven_pi_six_l450_450587


namespace area_of_right_triangle_l450_450036

theorem area_of_right_triangle (h : ℝ) : 
  ∃ (k : ℝ), (∀ (a b : ℝ), a / b = 3 / 4 → a^2 + b^2 = h^2 → 1 / 2 * a * b = k * h^2) := 
by {
  use (6/25),
  intros a b hab_eq h_eq,
  simp,
  sorry
}

end area_of_right_triangle_l450_450036


namespace parallel_vectors_tan_x_zero_of_f_l450_450222

open Real

variable {x : ℝ}

def a : ℝ × ℝ := (sin x, 3 / 2)
def b : ℝ × ℝ := (cos x, -1)

-- Part I
theorem parallel_vectors_tan_x :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → tan x = -3 / 2 :=
by
  sorry

-- Part II
noncomputable def f (x : ℝ) : ℝ := (sin x + cos x) * cos x + (3 / 2 - 1) * (-1)

theorem zero_of_f :
  ∃ x ∈ Icc (-pi / 2) 0, f x = 0 :=
by
  use -pi / 8
  split
  { norm_num, linarith },
  { sorry }

end parallel_vectors_tan_x_zero_of_f_l450_450222


namespace smallest_perimeter_triangle_l450_450624

-- Definitions of given conditions
variable {Point : Type}
variable (O X Y A : Point)
variable (OX OY : Point → Prop)
variable [Geometry OX OY] -- Assume appropriate geometric context and assumptions

-- The main theorem statement 
theorem smallest_perimeter_triangle :
  ∃ (B' C' : Point), 
  (OX B') ∧ (OY C') ∧ (is_perpendicular A O A B') ∧
  (is_perpendicular A O A C') ∧
  (∀ (B C : Point), OX B → OY C → (perimeter_triangle A B' C') ≤ (perimeter_triangle A B C)) := 
sorry

end smallest_perimeter_triangle_l450_450624


namespace csc_of_angle_in_fourth_quadrant_l450_450984

theorem csc_of_angle_in_fourth_quadrant (α : ℝ) 
  (h_cos : cos α = 4 / 5) 
  (h_quad : ∃ k : ℤ, α = 2 * π * k + (2 * π - arccos (4 / 5))) : 
  csc α = -5 / 3 :=
by sorry

end csc_of_angle_in_fourth_quadrant_l450_450984


namespace largest_quotient_is_3_l450_450495

noncomputable def largest_quotient : ℚ :=
  let S : set ℚ := { -32, -4, -1, 3, 6, 9 }
  have H : ∃ (a b : ℚ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∀ (x y : ℚ), x ∈ S ∧ y ∈ S ∧ x ≠ y → x / y ≤ a / b,
    from sorry,
  H.some / H.some_spec.some

theorem largest_quotient_is_3 :
  largest_quotient = 3 :=
begin
  unfold largest_quotient,
  have H : ∃ (a b : ℚ), a ∈ { -32, -4, -1, 3, 6, 9 } ∧ b ∈ { -32, -4, -1, 3, 6, 9 } ∧ a ≠ b ∧ ∀ (x y : ℚ), x ∈ { -32, -4, -1, 3, 6, 9 } ∧ y ∈ { -32, -4, -1, 3, 6, 9 } ∧ x ≠ y → x / y ≤ a / b,
    from sorry,
  rw H.some_spec.some_spec.some_spec,
  sorry
end

end largest_quotient_is_3_l450_450495


namespace burattino_suspects_after_seventh_draw_l450_450060

noncomputable def probability (total : ℕ) (choose : ℕ) : ℚ := 
  (nat.factorial total / (nat.factorial choose * nat.factorial (total - choose))) 

noncomputable def suspicion_threshold (threshold : ℚ) (probability_per_draw : ℚ) : ℕ :=
  nat.find (λ n, probability_per_draw^n < threshold)

theorem burattino_suspects_after_seventh_draw :
  let a := (probability 39 6) / (probability 45 6) in
  let threshold := (1 : ℚ) / 100 in
  suspicion_threshold threshold a = 6 :=
by
  sorry

end burattino_suspects_after_seventh_draw_l450_450060


namespace min_questions_to_determine_P_in_square_l450_450008

-- Definitions and conditions
def Point : Type := ℝ × ℝ
def Square (A B C D : Point) : Prop := 
  A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ -- non-degenerate quadrilateral
  A.1 ≠ C.1 ∧ A.2 ≠ C.2 ∧ -- diagonal AC should be non-vertical and non-horizontal

-- Main theorem statement
theorem min_questions_to_determine_P_in_square (A B C D P : Point) (sq : Square A B C D) :
  ∃ (q : ℕ), q = 3 ∧ ∀ (answer_AC answer_AB answer_BC : bool), 
    (determine_side_P answer_AC P A C) ∧ 
    (determine_side_P answer_AB P A B) ∧ 
    (determine_side_P answer_BC P B C) 
    → inside_square P A B C D := 
  sorry

noncomputable def determine_side_P (answer : bool) (P A B : Point) : Prop :=
  if answer then (P.1 - A.1) * (B.2 - A.2) - (P.2 - A.2) * (B.1 - A.1) > 0 
  else (P.1 - A.1) * (B.2 - A.2) - (P.2 - A.2) * (B.1 - A.1) ≤ 0 

def inside_square (P A B C D : Point) : Prop :=
  -- Placeholder definition to express the property
  sorry

end min_questions_to_determine_P_in_square_l450_450008


namespace total_surface_area_of_rectangular_solid_is_422_l450_450936

noncomputable def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_prime_edge_length (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c

def volume (a b c : ℕ) : ℕ := a * b * c

def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

theorem total_surface_area_of_rectangular_solid_is_422 :
  ∃ (a b c : ℕ), is_prime_edge_length a b c ∧ volume a b c = 399 ∧ surface_area a b c = 422 :=
begin
  sorry
end

end total_surface_area_of_rectangular_solid_is_422_l450_450936


namespace median_of_100_numbers_l450_450332

theorem median_of_100_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 100)
  (h_median_99_1 : ∀ num ∈ numbers, median (numbers.erase num) = 78 → num ∈ numbers)
  (h_median_99_2 : ∀ num ∈ numbers, median (numbers.erase num) = 66 → num ∈ numbers) :
  median numbers = 72 :=
sorry

end median_of_100_numbers_l450_450332


namespace sides_inequality_l450_450780

-- Define the values representing the sides of the triangle and the area
variables {a b c T : ℝ}

-- Specify the conditions: a, b, c are sides of a triangle and T is the area
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom area_def : T = (√(s * (s - a) * (s - b) * (s - c)))
  where s := (a + b + c) / 2

-- Prove the inequality for the sides of the triangle
theorem sides_inequality (hT : area_def) (htriangle : triangle_sides) :
  a^2 + b^2 + c^2 ≥ 4 * √3 * T :=
by
  sorry

end sides_inequality_l450_450780


namespace proof_problem_l450_450990

noncomputable def R (x : ℝ) : ℝ :=
if h : ∃ (p q : ℕ), x = p / q ∧ Nat.gcd p q = 1 then
  let ⟨p, q, hx, hq⟩ := h in 1 / q
else 0

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ set.Icc 0 1 then R x else
if x ∈ set.Icc (-1) 0 then -R (-x) else
if x ∈ set.Icc 1 2 then
  let z := 2 - x in
  if z ∈ set.Icc 0 1 then -R z else 0
else 0

theorem proof_problem : f (-7/5) - f (sqrt 2 / 3) = 1/5 :=
sorry

end proof_problem_l450_450990


namespace median_eq_mean_sum_l450_450931

theorem median_eq_mean_sum (y : ℝ) :
  let nums := [3, 7, 9, y, 20] in
  (median nums = mean nums) → y = -4 :=
by {
  sorry
}

end median_eq_mean_sum_l450_450931


namespace burattino_suspects_after_seventh_draw_l450_450058

noncomputable def probability (total : ℕ) (choose : ℕ) : ℚ := 
  (nat.factorial total / (nat.factorial choose * nat.factorial (total - choose))) 

noncomputable def suspicion_threshold (threshold : ℚ) (probability_per_draw : ℚ) : ℕ :=
  nat.find (λ n, probability_per_draw^n < threshold)

theorem burattino_suspects_after_seventh_draw :
  let a := (probability 39 6) / (probability 45 6) in
  let threshold := (1 : ℚ) / 100 in
  suspicion_threshold threshold a = 6 :=
by
  sorry

end burattino_suspects_after_seventh_draw_l450_450058


namespace smallest_possible_value_of_N_l450_450231

theorem smallest_possible_value_of_N :
  ∃ N : ℕ, N > 70 ∧ (21 * N) % 70 = 0 ∧ N = 80 :=
by
  existsi 80
  split
  · norm_num
  split
  · norm_num
  sorry

end smallest_possible_value_of_N_l450_450231


namespace smallest_N_l450_450273

theorem smallest_N (N : ℕ) (hN : N > 70) (hdv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450273


namespace smallest_possible_N_l450_450260

theorem smallest_possible_N (N : ℕ) (h1 : N > 70) (h2 : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_possible_N_l450_450260


namespace problem_l450_450807

theorem problem (x : ℝ) (h1 : x^2 = 1 → x = 1) :
  let converse := ∀ x, x = 1 → x^2 = 1,
      inverse := ∀ x, x^2 ≠ 1 → x ≠ 1,
      contrapositive := ∀ x, x ≠ 1 → x^2 ≠ 1 in
  (converse ∨ inverse ∨ contrapositive → (converse ∧ inverse ∧ ¬contrapositive)) :=
by sorry

end problem_l450_450807


namespace lisa_earns_more_than_tommy_l450_450377

theorem lisa_earns_more_than_tommy {total_earnings : ℤ} (h1 : total_earnings = 60) :
  let lisa_earnings := total_earnings / 2
  let tommy_earnings := lisa_earnings / 2
  lisa_earnings - tommy_earnings = 15 :=
by
  sorry

end lisa_earns_more_than_tommy_l450_450377


namespace one_third_of_recipe_l450_450884

noncomputable def recipe_flour_required : ℚ := 7 + 3 / 4

theorem one_third_of_recipe : (1 / 3) * recipe_flour_required = (2 : ℚ) + 7 / 12 :=
by
  sorry

end one_third_of_recipe_l450_450884


namespace triangle_median_identity_l450_450773

-- Define the setup for the triangle and median
variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
variable (m : ℝ)  -- median to side c
variable (a b c : ℝ)

-- Define the triangle conditions
-- The specific definitions for the sides and median are abstracted for simplicity in this example

theorem triangle_median_identity 
  (h : ∀ a b c m, -- for any given \( a, b, c \) and median \( m \)
    a^{2} + b^{2} = 2 \left( \frac{c^{2}}{4} + m^{2} \right)) : 
  a^{2} + b^{2} = 2 \left( \frac{c^{2}}{4} + m^{2} \right) :=
sorry

end triangle_median_identity_l450_450773


namespace find_median_of_100_l450_450301

noncomputable def median_of_set (s : Finset ℝ) : ℝ :=
if h : ∃ median, is_median s median then classical.some h else 0

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
∃ (f : ℤ), (f : ℝ) = (card s : ℝ) / 2 ∧
    ∃ (low : Finset ℝ) (high : Finset ℝ),
        low ⊆ s ∧ high ⊆ s ∧
        card low = floor ((card s : ℝ) / 2) ∧
        card high = ceil ((card s : ℝ) / 2) ∧
        (∀ x ∈ low, x ≤ m) ∧ (∀ x ∈ high, x ≥ m)

theorem find_median_of_100 (s : Finset ℝ) (h_size : s.card = 100)
(h1 : ∃ x ∈ s, median_of_set (s.erase x) = 78)
(h2 : ∃ y ∈ s, median_of_set (s.erase y) = 66) :
  median_of_set s = 72 := by
sorry

end find_median_of_100_l450_450301


namespace median_room_number_l450_450567

theorem median_room_number (rooms : List ℕ) (h_distinct : rooms.Nodup) (h_sorted : rooms = (List.range' 1 30).filter (λ n, n ≠ 15 ∧ n ≠ 20 ∧ n ≠ 25)) : 
  rooms.nth (rooms.length / 2) = some 18 :=
by
  sorry

end median_room_number_l450_450567


namespace steps_to_school_l450_450505

-- Define the conditions as assumptions
def distance : Float := 900
def step_length : Float := 0.45

-- Define the statement to be proven
theorem steps_to_school (x : Float) : step_length * x = distance → x = 2000 := by
  intro h
  sorry

end steps_to_school_l450_450505


namespace number_of_valid_S_l450_450660

def M : Set ℕ := { x | ∃ n, x = 3 * n ∧ n ∈ {1, 2, 3, 4} }
def P : Set ℕ := { x | ∃ k, x = 3 ^ k ∧ k ∈ {1, 2, 3} }
def M_inter_P : Set ℕ := {3, 9}
def M_union_P : Set ℕ := {3, 6, 9, 12, 27}
def valid_S (S : Set ℕ) : Prop := M_inter_P ⊆ S ∧ S ⊆ M_union_P

theorem number_of_valid_S : ∃ n, n = 8 ∧ (Card (Σ' (S : Set ℕ), valid_S S) = n) :=
by
  -- Proof to be filled in
  sorry

end number_of_valid_S_l450_450660


namespace carrots_remaining_l450_450515

theorem carrots_remaining 
  (total_carrots : ℕ)
  (weight_20_carrots : ℕ)
  (removed_carrots : ℕ)
  (avg_weight_remaining : ℕ)
  (avg_weight_removed : ℕ)
  (h1 : total_carrots = 20)
  (h2 : weight_20_carrots = 3640)
  (h3 : removed_carrots = 4)
  (h4 : avg_weight_remaining = 180)
  (h5 : avg_weight_removed = 190) :
  total_carrots - removed_carrots = 16 :=
by 
  -- h1 : 20 carrots in total
  -- h2 : total weight of 20 carrots is 3640 grams
  -- h3 : 4 carrots are removed
  -- h4 : average weight of remaining carrots is 180 grams
  -- h5 : average weight of removed carrots is 190 grams
  sorry

end carrots_remaining_l450_450515


namespace increasing_function_cond_l450_450616

theorem increasing_function_cond (f : ℝ → ℝ)
  (h : ∀ a b : ℝ, a ≠ b → (f a - f b) / (a - b) > 0) :
  ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end increasing_function_cond_l450_450616


namespace water_added_to_mixture_is_11_l450_450695

noncomputable def initial_mixture_volume : ℕ := 45
noncomputable def initial_milk_ratio : ℚ := 4
noncomputable def initial_water_ratio : ℚ := 1
noncomputable def final_milk_ratio : ℚ := 9
noncomputable def final_water_ratio : ℚ := 5

theorem water_added_to_mixture_is_11 :
  ∃ x : ℚ, (initial_milk_ratio * initial_mixture_volume / 
            (initial_water_ratio * initial_mixture_volume + x)) = (final_milk_ratio / final_water_ratio)
  ∧ x = 11 :=
by
  -- Proof here
  sorry

end water_added_to_mixture_is_11_l450_450695


namespace edward_spent_on_toys_l450_450119

theorem edward_spent_on_toys 
    (board_game_cost : ℝ := 2)
    (action_figure_count : ℕ := 4)
    (action_figure_cost : ℝ := 7)
    (puzzle_cost : ℝ := 6)
    (deck_cost : ℝ := 3.5)
    (discount_rate : ℝ := 0.10) :
    let total_action_figure_cost := action_figure_count * action_figure_cost,
        discount := discount_rate * total_action_figure_cost,
        discounted_action_figure_cost := total_action_figure_cost - discount,
        total_cost := board_game_cost + discounted_action_figure_cost + puzzle_cost + deck_cost
    in total_cost = 36.70 :=
by
  let total_action_figure_cost := action_figure_count * action_figure_cost
  let discount := discount_rate * total_action_figure_cost
  let discounted_action_figure_cost := total_action_figure_cost - discount
  let total_cost := board_game_cost + discounted_action_figure_cost + puzzle_cost + deck_cost
  sorry

end edward_spent_on_toys_l450_450119


namespace perimeter_of_resulting_figure_l450_450888

def side_length := 100
def original_square_perimeter := 4 * side_length
def rectangle_width := side_length
def rectangle_height := side_length / 2
def number_of_longer_sides_of_rectangles_touching := 4

theorem perimeter_of_resulting_figure :
  let new_perimeter := 3 * side_length + number_of_longer_sides_of_rectangles_touching * rectangle_height
  new_perimeter = 500 :=
by
  sorry

end perimeter_of_resulting_figure_l450_450888


namespace triangle_third_side_length_l450_450192

theorem triangle_third_side_length (a b : ℝ) (x : ℝ) (h₁ : a = 5) (h₂ : b = 8) (hx : x ∈ {2, 3, 6, 13}) :
  3 < x ∧ x < 13 → x = 6 :=
by sorry

end triangle_third_side_length_l450_450192


namespace both_not_in_area_l450_450579

variables (p q: Prop)

theorem both_not_in_area (p q: Prop) : ¬p ∧ ¬q ↔ "Both trainees did not land within the designated area" :=
by sorry

end both_not_in_area_l450_450579


namespace limit_tan_exp_div_limit_problem_l450_450907

theorem limit_tan_exp_div (x : ℝ) (h : 0 ≤ x ∧ x < 1):
  (Real.exp x - 1) / x = 1 :=
begin
  sorry
end

theorem limit_problem : 
  (∀ x, 0 ≤ x ∧ x < 1 → 
    (Real.tan (Real.pi / 4 - x))^((Real.exp x - 1) / x) = 1) :=
begin
  intro x,
  intro hx,
  have H1 := limit_tan_exp_div x hx,
  have H2 : Real.tan (Real.pi / 4 - x) = 1,
  {
    rw [Real.tan_sub, Real.tan_pi_div_four],
    simp,
  },
  rw H2,
  rw H1,
  norm_num,
end

end limit_tan_exp_div_limit_problem_l450_450907


namespace binom_coeff_divisible_l450_450014

theorem binom_coeff_divisible (n k : ℕ) (h_coprime : Nat.coprime n k) (h_lt : k < n) : 
  (Nat.choose n k) % n = 0 := 
sorry

end binom_coeff_divisible_l450_450014


namespace option_A_option_B_option_C_option_D_l450_450977

variable (a b : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a + b = 1)

theorem option_A : ab ≤ 1 / 8 := by 
  sorry

theorem option_B : ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ 2 * a + b = 1 ∧ a^2 + b^2 = 1 / 5 := by 
  sorry

theorem option_C : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ 2 * a + b = 1 → (frac 1 a + frac 1 b ≠ 6) := by 
  sorry

theorem option_D : 0 < (b - 1) / (a - 1) ∧ (b - 1) / (a - 1) < 2 := by 
  sorry

end option_A_option_B_option_C_option_D_l450_450977


namespace median_of_100_set_l450_450320

theorem median_of_100_set 
  (S : Finset ℝ) (h_card : S.card = 100)
  (h_remove1 : ∃ x ∈ S, median (S.erase x) = 78)
  (h_remove2 : ∃ y ∈ S, median (S.erase y) = 66) : 
  median S = 72 :=
by
  sorry

end median_of_100_set_l450_450320


namespace find_median_of_100_l450_450299

noncomputable def median_of_set (s : Finset ℝ) : ℝ :=
if h : ∃ median, is_median s median then classical.some h else 0

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
∃ (f : ℤ), (f : ℝ) = (card s : ℝ) / 2 ∧
    ∃ (low : Finset ℝ) (high : Finset ℝ),
        low ⊆ s ∧ high ⊆ s ∧
        card low = floor ((card s : ℝ) / 2) ∧
        card high = ceil ((card s : ℝ) / 2) ∧
        (∀ x ∈ low, x ≤ m) ∧ (∀ x ∈ high, x ≥ m)

theorem find_median_of_100 (s : Finset ℝ) (h_size : s.card = 100)
(h1 : ∃ x ∈ s, median_of_set (s.erase x) = 78)
(h2 : ∃ y ∈ s, median_of_set (s.erase y) = 66) :
  median_of_set s = 72 := by
sorry

end find_median_of_100_l450_450299


namespace exists_increasing_or_decreasing_subsequence_l450_450623

theorem exists_increasing_or_decreasing_subsequence (n : ℕ) (a : Fin (n^2 + 1) → ℝ) :
  ∃ (b : Fin (n + 1) → ℝ), (StrictMono b ∨ StrictAnti b) :=
sorry

end exists_increasing_or_decreasing_subsequence_l450_450623


namespace roots_real_if_k_pure_imaginary_l450_450574

-- Definitions based on given problem conditions
def quadratic_roots (a b c : ℂ) : Prop :=
  ∃ (r1 r2 : ℂ), r1 + r2 = b/a ∧ r1 * r2 = c/a

-- Main theorem to prove
theorem roots_real_if_k_pure_imaginary (k : ℂ) (z : ℂ) 
  (h: 5 * z^2 + 7 * complex.I * z - k = 0)
  (hk: k.im ≠ 0 ∧ k.re = 0): 
  quadratic_roots 5 (7 * complex.I) (-k) ∧
  ∀ r1 r2 : ℂ, 5 * (r1^2) + 7 * complex.I * r1 - k = 0 ∧ 5 * (r2^2) + 7 * complex.I * r2 - k = 0 → 
    (r1.im = 0 ∧ r2.im = 0) := 
sorry

end roots_real_if_k_pure_imaginary_l450_450574


namespace total_ducks_and_ducklings_l450_450383

theorem total_ducks_and_ducklings : 
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6 
  let total_ducklings := ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3
  let total_ducks := ducks1 + ducks2 + ducks3
  in total_ducks + total_ducklings = 99 := 
by {
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6
  let total_ducklings := ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3
  let total_ducks := ducks1 + ducks2 + ducks3
  show total_ducks + total_ducklings = 99
  sorry
}

end total_ducks_and_ducklings_l450_450383


namespace triangle_GVN_sim_triangle_GND_l450_450090

variable {G H K L N V D : Point}
variable [CircleGeometry G H K L N V D]

-- Given conditions
axiom GH_perp_bis_KL : PerpendicularBisector GH KL ∧ Intersects GH KL N
axiom V_between_KN : Between V K N
axiom GV_extends_to_circle_at_D : ExtendsToCircle GV D

-- To prove similarity of triangles
theorem triangle_GVN_sim_triangle_GND :
  Similar (Triangle G V N) (Triangle G N D) :=
sorry

end triangle_GVN_sim_triangle_GND_l450_450090


namespace find_f3_l450_450510

def f : ℝ → ℝ := sorry  -- Placeholder for the function definition

-- Conditions from the problem
axiom h1 : ∀ x : ℝ, f(2 * x + 1) = 2 * f(x) + 1
axiom h2 : f(0) = 2

-- Statement to be proved
theorem find_f3 : f(3) = 11 :=
by
  sorry

end find_f3_l450_450510


namespace Edmund_can_wrap_15_boxes_every_3_days_l450_450939

-- We define the conditions as Lean definitions
def inches_per_gift_box : ℕ := 18
def inches_per_day : ℕ := 90

-- We state the theorem to prove the question (15 gift boxes every 3 days)
theorem Edmund_can_wrap_15_boxes_every_3_days :
  (inches_per_day / inches_per_gift_box) * 3 = 15 :=
by
  sorry

end Edmund_can_wrap_15_boxes_every_3_days_l450_450939


namespace tangent_line_range_l450_450684

theorem tangent_line_range (a : ℝ) :
  (∀ (x y : ℝ), (x - 1)^2 + y^2 = 1 → x * a + y - 2 = 0) ∧
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = 1/4 → x * a + y - 2 = 0) ↔
  -real.sqrt 3 < a ∧ a < 3/4 :=
sorry

end tangent_line_range_l450_450684


namespace range_of_fx2_l450_450290
open real

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := (x - 1)^2 + a * ln x

-- Define the conditions
variable (a : ℝ)
axiom a_range : 0 < a ∧ a < 1 / 2
axiom extreme_points_exist : ∃ (x1 x2 : ℝ), x1 < x2 ∧ x1 + x2 = 1

-- Statement of the theorem
theorem range_of_fx2 : ∃ (y_range : ℝ), 
  (∃ x2, (1 / 2 < x2 ∧ x2 < 1) ∧ y_range = f x2 a) 
  ∧ y_range = \frac{1 - 2 * ln 2}{4} ∨ 0 < y_range := sorry

end range_of_fx2_l450_450290


namespace length_of_cloth_l450_450844

theorem length_of_cloth (L : ℝ) (h : 35 = (L + 4) * (35 / L - 1)) : L = 10 :=
sorry

end length_of_cloth_l450_450844


namespace unique_value_of_n_l450_450337

theorem unique_value_of_n
  (n t : ℕ) (h1 : t ≠ 0)
  (h2 : 15 * t + (n - 20) * t / 3 = (n * t) / 2) :
  n = 50 :=
by sorry

end unique_value_of_n_l450_450337


namespace initial_ratio_l450_450538

theorem initial_ratio (partners associates associates_after_hiring : ℕ)
  (h_partners : partners = 20)
  (h_associates_after_hiring : associates_after_hiring = 20 * 34)
  (h_assoc_equation : associates + 50 = associates_after_hiring) :
  (partners : ℚ) / associates = 2 / 63 :=
by
  sorry

end initial_ratio_l450_450538


namespace jade_monthly_earnings_l450_450357

theorem jade_monthly_earnings :
  ∃ E : ℝ, (0.75 * E) + (0.2 * E) + 80 = E ∧ E = 1600 :=
begin
  use 1600,
  split,
  { sorry },
  { refl }
end

end jade_monthly_earnings_l450_450357


namespace value_of_e_over_f_l450_450164

theorem value_of_e_over_f 
    (a b c d e f : ℝ) 
    (h1 : a * b * c = 1.875 * d * e * f)
    (h2 : a / b = 5 / 2)
    (h3 : b / c = 1 / 2)
    (h4 : c / d = 1)
    (h5 : d / e = 3 / 2) : 
    e / f = 1 / 3 :=
by
  sorry

end value_of_e_over_f_l450_450164


namespace count_rainbow_four_digit_numbers_l450_450681

def is_rainbow_four_digit_number (a b c d : ℕ) : Prop :=
  (a ≠ 0) ∧ (1000 * a + 100 * b + 10 * c + d ≥ 1000) ∧ ((a - b) * (c - d) < 0)

theorem count_rainbow_four_digit_numbers : 
  let count := (∑ a in finset.range 9 \{0}, ∑ b in finset.range 10, ∑ c in finset.range 10, ∑ d in finset.range 10, 
    if is_rainbow_four_digit_number a b c d then 1 else 0)
  in count = 3645 :=
by
  -- Proof is omitted
  sorry

end count_rainbow_four_digit_numbers_l450_450681


namespace smallest_possible_value_of_N_l450_450236

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l450_450236


namespace negation_of_universal_proposition_l450_450802

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ ∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l450_450802


namespace arrangements_count_l450_450051

-- Definitions of students and grades
inductive Student : Type
| A | B | C | D | E | F
deriving DecidableEq

inductive Grade : Type
| first | second | third
deriving DecidableEq

-- A function to count valid arrangements
def valid_arrangements (assignments : Student → Grade) : Bool :=
  assignments Student.A = Grade.first ∧
  assignments Student.B ≠ Grade.third ∧
  assignments Student.C ≠ Grade.third ∧
  (assignments Student.A = Grade.first) ∧
  ((assignments Student.B = Grade.second ∧ assignments Student.C = Grade.second ∧ 
    (assignments Student.D ≠ Grade.first ∨ assignments Student.E ≠ Grade.first ∨ assignments Student.F ≠ Grade.first)) ∨
   ((assignments Student.B ≠ Grade.second ∨ assignments Student.C ≠ Grade.second) ∧ 
    (assignments Student.B ≠ Grade.first ∨ assignments Student.C ≠ Grade.first)))

theorem arrangements_count : 
  ∃ (count : ℕ), count = 9 ∧
  count = (Nat.card { assign : Student → Grade // valid_arrangements assign } : ℕ) := sorry

end arrangements_count_l450_450051


namespace identical_answers_l450_450141
-- Import necessary libraries

-- Define the entities and conditions
structure Person :=
  (name : String)
  (always_tells_truth : Bool)

def Fyodor : Person := { name := "Fyodor", always_tells_truth := true }
def Sasha : Person := { name := "Sasha", always_tells_truth := false }

def answer (p : Person) : String :=
  if p.always_tells_truth then "Yes" else "No"

-- The theorem statement
theorem identical_answers :
  answer Fyodor = answer Sasha :=
by
  -- Proof steps will be filled in later
  sorry

end identical_answers_l450_450141


namespace product_of_undefined_values_l450_450954

noncomputable def quadratic := λ (a b c x : ℝ), a * x^2 + b * x + c

theorem product_of_undefined_values : 
  ∃ x1 x2 : ℝ, quadratic 1 -1 -6 x1 = 0 ∧ quadratic 1 -1 -6 x2 = 0 ∧ x1 * x2 = -6 :=
by
  have h : x^2 - x - 6 = 0, from sorry,
  use [3, -2],
  split; 
  sorry

end product_of_undefined_values_l450_450954


namespace boat_travel_distance_upstream_l450_450862

noncomputable def upstream_distance (v : ℝ) : ℝ :=
  let d := 2.5191640969412834 * (v + 3)
  d

theorem boat_travel_distance_upstream :
  ∀ v : ℝ, 
  (∀ D : ℝ, D / (v + 3) = 2.5191640969412834 → D / (v - 3) = D / (v + 3) + 0.5) → 
  upstream_distance 33.2299691632954 = 91.25 :=
by
  sorry

end boat_travel_distance_upstream_l450_450862


namespace simplify_expression_l450_450584

theorem simplify_expression (x : ℝ) :
  ( ( ((x + 1) ^ 3 * (x ^ 2 - x + 1) ^ 3) / (x ^ 3 + 1) ^ 3 ) ^ 2 *
    ( ((x - 1) ^ 3 * (x ^ 2 + x + 1) ^ 3) / (x ^ 3 - 1) ^ 3 ) ^ 2 ) = 1 :=
by
  sorry

end simplify_expression_l450_450584


namespace negation_of_exists_statement_l450_450434

theorem negation_of_exists_statement :
  ¬ (∃ x0 : ℝ, x0 > 0 ∧ x0^2 - 5 * x0 + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 :=
by
  sorry

end negation_of_exists_statement_l450_450434


namespace prove_angles_greater_than_120_l450_450221

variables {n : ℕ} -- Introduce the number of points
variables {A : fin n → Point} -- Introduce the n points as a mapping from fin n to points

-- Introduce the function angle which will take three points and return the measure of the angle in degrees
noncomputable def angle (A_i A_j A_k : Point) : ℝ := sorry

-- State the problem as a Lean theorem
theorem prove_angles_greater_than_120 (h : ∀ (i j k : fin n), 1 ≤ i.val → i.val < j.val → j.val < k.val → k.val ≤ n → angle (A i) (A j) (A k) > 120) : 
  ∀ (i j k : fin n), 1 ≤ i.val → i.val < j.val → j.val < k.val → k.val ≤ n → angle (A i) (A j) (A k) > 120 :=
begin
  sorry,
end

end prove_angles_greater_than_120_l450_450221


namespace percentage_increase_new_energy_vehicles_l450_450026

variable {a : ℝ} (h1 : 0.9 * a * (1 - 0.1) + 0.1 * a * (1 + x / 100) = a)

theorem percentage_increase_new_energy_vehicles (x : ℝ) :
  h1 → x = 90 :=
by
  sorry

end percentage_increase_new_energy_vehicles_l450_450026


namespace largest_square_test_plots_l450_450537

theorem largest_square_test_plots (field_length : ℕ) (field_width : ℕ) (available_fencing : ℕ) (squares_parallel : ∀ (s : ℕ), divides s field_length ∧ divides s field_width) : 
  field_length = 60 ∧ field_width = 20 ∧ available_fencing = 2200 ∧ (∃ s : ℕ, s > 0 ∧ divides s 60 ∧ divides s 20 ∧ ((20 * ((60 / s) - 1)) + (60 * ((20 / s) - 1)) ≤ available_fencing) ∧ ((field_length / s) * (field_width / s) = 75)) :=
by
  sorry

end largest_square_test_plots_l450_450537


namespace distinguishable_colorings_of_cube_l450_450937

theorem distinguishable_colorings_of_cube : 
  let colors := 4 in
  let faces := 6 in
  let colorings := (colors * (faces-0) -- All faces the same color
                  + colors * (colors-1) * (faces-5) -- Five faces one color, one face another color
                  + (colors * (colors-1) * ((faces choose 2) / 2)) -- Four faces one color, two faces another color
                  + (colors * (colors-1) * ((faces choose 3) / 2)) -- Three faces one color, three faces another color
                  ) in
  colorings = 40 := sorry

end distinguishable_colorings_of_cube_l450_450937


namespace imaginary_part_of_z_l450_450683

theorem imaginary_part_of_z (z : ℂ) (h : (3 + 4 * complex.i) * z = 5) : complex.im z = -4 / 5 := sorry

end imaginary_part_of_z_l450_450683


namespace tunnel_height_at_10_feet_l450_450548

theorem tunnel_height_at_10_feet :
  ∃ (a k : ℝ), k = 20 ∧
  (∀ x y, y = a * x^2 + k ∧ ((x = 25 ∨ x = -25) → y = 0)) ∧
  (∀ x, x = 10 → ( y ≈ 16.8)) :=
sorry

end tunnel_height_at_10_feet_l450_450548


namespace Gary_final_amount_l450_450142

theorem Gary_final_amount
(initial_amount dollars_snake dollars_hamster dollars_supplies : ℝ)
(h1 : initial_amount = 73.25)
(h2 : dollars_snake = 55.50)
(h3 : dollars_hamster = 25.75)
(h4 : dollars_supplies = 12.40) :
  initial_amount + dollars_snake - dollars_hamster - dollars_supplies = 90.60 :=
by
  sorry

end Gary_final_amount_l450_450142


namespace planes_parallel_l450_450292

variables (P P1 : Point) (a b c a1 b1 : Line) (α α1 : Plane)

-- Conditions
axiom intersecting_lines_in_planes (P : Point) (a b : Line) (α : Plane) : a ∈ α ∧ b ∈ α ∧ a ∩ b = {P}
axiom intersecting_lines_in_planes1 (P1 : Point) (a1 b1 : Line) (α1 : Plane) : a1 ∈ α1 ∧ b1 ∈ α1 ∧ a1 ∩ b1 = {P1}
axiom parallel_lines_a (a a1 : Line) : a ∥ a1
axiom parallel_lines_b (b b1 : Line) : b ∥ b1

theorem planes_parallel (P P1 : Point) (a b c a1 b1 : Line) (α α1 : Plane)
    (h1 : intersecting_lines_in_planes P a b α)
    (h2 : intersecting_lines_in_planes1 P1 a1 b1 α1)
    (h3 : parallel_lines_a a a1)
    (h4 : parallel_lines_b b b1) :
    α ∥ α1 := 
sorry

end planes_parallel_l450_450292


namespace little_ming_problem_solution_l450_450409

theorem little_ming_problem_solution :
  let number_of_ways := ∑ i in (Finset.range 10), Nat.choose 9 i
  number_of_ways = 512 :=
by
  -- The proof steps are not required according to the problem statement
  sorry

end little_ming_problem_solution_l450_450409


namespace parallelogram_below_line_l450_450760

open Real

def A := (4, 2)
def B := (-2, -4)
def C := (-8, -4)
def D := (0, 4)

def isBelowLine (p : ℝ × ℝ) (y : ℝ) : Prop := p.2 < y

theorem parallelogram_below_line :
  (∀ p ∈ {p : ℝ × ℝ | ((p.1, p.2) = A) ∨ ((p.1, p.2) = B) ∨ ((p.1, p.2) = C) ∨ ((p.1, p.2) = D)},
  isBelowLine p (-2)) :=
sorry

end parallelogram_below_line_l450_450760


namespace negation_of_proposition_l450_450433

theorem negation_of_proposition : (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x + 6 > 0) ↔ ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0 := by
  sorry

end negation_of_proposition_l450_450433


namespace sum_f_eq_24112_l450_450213

noncomputable def f (x : ℝ) : ℝ := 
  let g := x^2 - 2013 * x + 6030
  g + |g|

theorem sum_f_eq_24112 : (Finset.range 2014).sum f = 24112 := by
  sorry

end sum_f_eq_24112_l450_450213


namespace paths_count_l450_450945

/-- A dodecahedron with the given properties and constraints --/
inductive Corner
| A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T

def dodecahedron_edges : List (Corner × Corner) := [
  (Corner.A, Corner.B), (Corner.A, Corner.C), (Corner.A, Corner.D),
  (Corner.B, Corner.E), (Corner.B, Corner.F), (Corner.C, Corner.G),
  -- Assume all edges are listed according to the dodecahedron structure
  -- ...
  -- Last few edges for illustration:
  (Corner.R, Corner.S), (Corner.R, Corner.T), (Corner.S, Corner.T)
]

def valid_path (path : List Corner) : Prop :=
  (path.head = some Corner.A) ∧ -- Starts at corner (0,0,0)
  (path.last = some Corner.B) ∧ -- Finishes at (1,1,0)
  (path.length = 20) ∧ -- 20 corners so 19 edges
  (path.nodup) ∧ -- Visits every corner exactly once
  (∀ (c1 c2 : Corner), (c1, c2) ∈ dodecahedron_edges → c2 ∈ path.dropWhile (≠ c1)) -- Follows valid edges

theorem paths_count : (paths : List (List Corner) := 90 :=
begin
  sorry, -- Proof to show there are exactly 90 such paths
end

end paths_count_l450_450945


namespace smallest_N_l450_450252

theorem smallest_N (N : ℕ) (hN : N > 70) (hDiv : 70 ∣ 21 * N) : N = 80 :=
sorry

end smallest_N_l450_450252


namespace max_amount_paul_received_l450_450762

theorem max_amount_paul_received :
  ∃ (numBplus numA numAplus : ℕ),
  (numBplus + numA + numAplus = 10) ∧ 
  (numAplus ≥ 2 → 
    let BplusReward := 5;
    let AReward := 2 * BplusReward;
    let AplusReward := 15;
    let Total := numAplus * AplusReward + numA * (2 * AReward) + numBplus * (2 * BplusReward);
    Total = 190
  ) :=
sorry

end max_amount_paul_received_l450_450762


namespace coefficient_of_quadratic_polynomial_l450_450957

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem coefficient_of_quadratic_polynomial (a b c : ℝ) (h : a > 0) :
  |f a b c 1| = 2 ∧ |f a b c 2| = 2 ∧ |f a b c 3| = 2 →
  (a = 4 ∧ b = -16 ∧ c = 14) ∨ (a = 2 ∧ b = -6 ∧ c = 2) ∨ (a = 2 ∧ b = -10 ∧ c = 10) :=
by
  sorry

end coefficient_of_quadratic_polynomial_l450_450957


namespace triangle_inequality_third_side_l450_450171

theorem triangle_inequality_third_side (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) :
  3 < x ∧ x < 13 ↔ (5 + 8 > x) ∧ (5 + x > 8) ∧ (8 + x > 5) :=
by 
  -- Placeholder for proof
  sorry

end triangle_inequality_third_side_l450_450171


namespace market_value_of_stock_stock_market_value_is_correct_l450_450518

theorem market_value_of_stock (dividend_percent: ℝ) (yield_percent: ℝ) (face_value: ℝ) : ℝ :=
by
  let dividend_per_share := (dividend_percent / 100) * face_value
  let market_value := (dividend_per_share / (yield_percent / 100)) * 100
  exact market_value

--- Using the given problem conditions:
def stock_yield := 8.0
def stock_dividend := 11.0
def face_value := 100.0

theorem stock_market_value_is_correct : market_value_of_stock stock_dividend stock_yield face_value = 137.50 :=
by
  sorry

end market_value_of_stock_stock_market_value_is_correct_l450_450518


namespace emily_patches_difference_l450_450386

theorem emily_patches_difference (h p : ℕ) (h_eq : p = 3 * h) :
  (p * h) - ((p + 5) * (h - 3)) = (4 * h + 15) :=
by
  sorry

end emily_patches_difference_l450_450386


namespace burattino_suspects_after_seventh_draw_l450_450057

noncomputable def probability (total : ℕ) (choose : ℕ) : ℚ := 
  (nat.factorial total / (nat.factorial choose * nat.factorial (total - choose))) 

noncomputable def suspicion_threshold (threshold : ℚ) (probability_per_draw : ℚ) : ℕ :=
  nat.find (λ n, probability_per_draw^n < threshold)

theorem burattino_suspects_after_seventh_draw :
  let a := (probability 39 6) / (probability 45 6) in
  let threshold := (1 : ℚ) / 100 in
  suspicion_threshold threshold a = 6 :=
by
  sorry

end burattino_suspects_after_seventh_draw_l450_450057


namespace f_increasing_neg_pi_div_3_to_0_l450_450226

-- Define the given vectors and function
def m (x : ℝ) : ℝ × ℝ := (2 * (Real.cos x)^2, Real.sqrt 3)
def n (x : ℝ) : ℝ × ℝ := (1, Real.sin (2 * x))
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- State the main theorem
theorem f_increasing_neg_pi_div_3_to_0 : 
  ∀ x, -Real.pi / 3 < x ∧ x < 0 → f(x) < f(x + ε) where ε: ℝ := sorry :=
sorry

end f_increasing_neg_pi_div_3_to_0_l450_450226


namespace ellipse_equation_and_max_AB_distance_l450_450992

-- Definitions and Conditions
def eccentricity : ℝ := sqrt 3 / 2
def major_axis_length : ℝ := 4
def foci_on_x_axis : Prop := True   -- This is just a place-holder to maintain structure
def slope_of_line : ℝ := 1

-- Equation of ellipse proof problem
theorem ellipse_equation_and_max_AB_distance :
  (∃ (a b : ℝ), 
     a > b ∧ b > 0 ∧ 
     -- Conditions on the ellipse
     (eccentricity = c / a) ∧
     (2 * a = major_axis_length) ∧ 
     (a^2 = b^2 + c^2) ∧ 
     -- Result 1: Standard equation of the ellipse
     (∀ (x y : ℝ), (y^2 - 1 = -(x^2 / 4)) = (x / a)^2 + (y / b)^2 = 1) ∧
     -- Result 2: Maximum |AB| value
     (∀ (l : ℝ), (l = slope_of_line) → (|AB| ≤ 4 * sqrt 10 / 5))) :=
sorry

end ellipse_equation_and_max_AB_distance_l450_450992


namespace median_of_100_numbers_l450_450327

theorem median_of_100_numbers (numbers : List ℝ) (h_length : numbers.length = 100)
  (h_sorted : numbers.sorted (≤))
  (h_51 : numbers.nth_le 50 h_51_nat = 78) -- Note: nth_le is zero-indexed, so 51st element is 50th index.
  (h_50 : numbers.nth_le 49 h_50_nat = 66) : 
  (numbers.nth_le 49 h_50_nat + numbers.nth_le 50 h_51_nat) / 2 = 72 :=
sorry

end median_of_100_numbers_l450_450327


namespace find_median_of_100_l450_450304

noncomputable def median_of_set (s : Finset ℝ) : ℝ :=
if h : ∃ median, is_median s median then classical.some h else 0

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
∃ (f : ℤ), (f : ℝ) = (card s : ℝ) / 2 ∧
    ∃ (low : Finset ℝ) (high : Finset ℝ),
        low ⊆ s ∧ high ⊆ s ∧
        card low = floor ((card s : ℝ) / 2) ∧
        card high = ceil ((card s : ℝ) / 2) ∧
        (∀ x ∈ low, x ≤ m) ∧ (∀ x ∈ high, x ≥ m)

theorem find_median_of_100 (s : Finset ℝ) (h_size : s.card = 100)
(h1 : ∃ x ∈ s, median_of_set (s.erase x) = 78)
(h2 : ∃ y ∈ s, median_of_set (s.erase y) = 66) :
  median_of_set s = 72 := by
sorry

end find_median_of_100_l450_450304


namespace parallel_line_and_plane_l450_450963

variables {Line Plane : Type}
variables (m n a : Line) (α β : Plane)
  
-- Definitions for the conditions
def parallel (x y : Line) : Prop := 
  -- Assume some definition for parallelism between lines
  sorry 

def plane_parallel (π₁ π₂ : Plane) : Prop := 
  -- Assume some definition for parallelism between planes
  sorry

def in_plane (l : Line) (π : Plane) : Prop := 
  -- Assume some definition for a line being contained in a plane
  sorry

-- The theorem we need to prove
theorem parallel_line_and_plane (α_parallel_β : plane_parallel α β) 
                               (a_in_α : in_plane a α) :
  parallel a β :=
sorry

end parallel_line_and_plane_l450_450963


namespace unique_flavors_l450_450783

theorem unique_flavors (x y : ℕ) (h₀ : x = 5) (h₁ : y = 4) : 
  (∃ f : ℕ, f = 17) :=
sorry

end unique_flavors_l450_450783


namespace difference_of_squares_l450_450095

theorem difference_of_squares : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end difference_of_squares_l450_450095


namespace children_absent_l450_450757

theorem children_absent (A : ℕ) (total_children : ℕ) (bananas_per_child : ℕ) (extra_bananas_per_child : ℕ) :
  total_children = 660 →
  bananas_per_child = 2 →
  extra_bananas_per_child = 2 →
  (total_children * bananas_per_child) = 1320 →
  ((total_children - A) * (bananas_per_child + extra_bananas_per_child)) = 1320 →
  A = 330 :=
by
  intros
  sorry

end children_absent_l450_450757


namespace value_of_star_l450_450847

theorem value_of_star :
  ∀ x : ℕ, 45 - (28 - (37 - (15 - x))) = 55 → x = 16 :=
by
  intro x
  intro h
  sorry

end value_of_star_l450_450847


namespace problem_solution_l450_450705

theorem problem_solution
  (x y θ : ℝ)
  (h₁ : x = 2 * Real.cos θ)
  (h₂ : y = Real.sin θ)
  (C₂_center_polar : (3, Real.pi / 2))
  (C₂_radius : 1)
  (C₂_center_cartesian : (0, 3)) :
  (∃ (x y : ℝ), (x = 2 * Real.cos θ ∧ y = Real.sin θ) → x^2 / 4 + y^2 = 1) ∧
  (∃ (x y : ℝ), x^2 + (y - 3)^2 = 1) ∧
  (∀ (θ : ℝ), 1 ≤ Real.sqrt (4 * (Real.cos θ) ^ 2 + (Real.sin θ - 3) ^ 2) + 1 ∧ Real.sqrt (4 * (Real.cos θ) ^ 2 + (Real.sin θ - 3) ^ 2) + 1 ≤ 5) :=
by
  sorry

end problem_solution_l450_450705


namespace quadrilateral_is_rhombus_l450_450102

-- Definitions of points E, F, G, H
def E : (ℝ × ℝ) := (0, 0)
def F : (ℝ × ℝ) := (0, 4)
def G : (ℝ × ℝ) := (6, 4)
def H : (ℝ × ℝ) := (6, 0)

-- Definitions of lines
def line_45 (x : ℝ) : ℝ := x
def line_neg_45 (x : ℝ) : ℝ := 4 - x
def tan75 : ℝ := 2 + Real.sqrt 3
def line_75 (x : ℝ) : ℝ := tan75 * x
def line_neg_75 (x : ℝ) : ℝ := 4 - tan75 * x

-- Definitions of intersection points
def intersection_45_neg_45 : (ℝ × ℝ) := (2, 2)
def intersection_75_neg_75 : (ℝ × ℝ) := (4 / (2 * tan75), tan75 * (4 / (2 * tan75)))

-- The mathematical statement to prove
theorem quadrilateral_is_rhombus :
  quadrilateral (intersection_45_neg_45) (intersection_75_neg_75) (intersection_75_neg_75_other) (intersection_45_neg_45_other) :=
sorry

end quadrilateral_is_rhombus_l450_450102


namespace continuity_at_x0_l450_450514

noncomputable def f (x : ℝ) : ℝ := -4 * x^2 - 7

theorem continuity_at_x0 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 1| < δ → |f x - f 1| < ε :=
by
  sorry

end continuity_at_x0_l450_450514


namespace obtuse_triangle_side_range_l450_450654

theorem obtuse_triangle_side_range (x : ℝ) : 
  (2 < 5) ∧ (x > 0) ∧ (x < 5) ∧
  ((3 < x ∧ x < 5 ∧ x^2 > 2^2 + 3^2) 
   ∨ (x < 3 ∧ 3^2 > x^2 + 2^2)) -> 
  (1 < x ∧ x < real.sqrt 5) 
  ∨ (real.sqrt 13 < x ∧ x < 5) :=
begin
  sorry,
end

end obtuse_triangle_side_range_l450_450654


namespace cone_volume_ratio_l450_450871

theorem cone_volume_ratio (h : ℝ) (V : ℝ → ℝ) (V₁ V₂ V₃ : ℝ) :
  (h > 0) →
  (∀ x : ℝ, x > 0 → V x = (π / 3) * x^3) →
  let c₁ := V (h / 3),
      c₂ := V (2 * h / 3) - V (h / 3),
      c₃ := V h - V (2 * h / 3)
  in c₁ / c₁ = 1 / 1 ∧ c₂ / c₁ = 7 / 1 ∧ c₃ / c₁ = 19 / 1 :=
by
  intros h_pos V_def
  let c₁ := V (h / 3)
  let c₂ := V (2 * h / 3) - c₁
  let c₃ := V h - V (2 * h / 3)
  sorry

end cone_volume_ratio_l450_450871


namespace treasure_coins_problem_l450_450371

theorem treasure_coins_problem (N m n t k s u : ℤ) 
  (h1 : N = (2/3) * (2/3) * (2/3) * (m - 1) - (2/3) - (2^2 / 3^2))
  (h2 : N = 3 * n)
  (h3 : 8 * (m - 1) - 30 = 81 * k)
  (h4 : m - 1 = 3 * t)
  (h5 : 8 * t - 27 * k = 10)
  (h6 : m = 3 * t + 1)
  (h7 : k = 2 * s)
  (h8 : 4 * t - 27 * s = 5)
  (h9 : t = 8 + 27 * u)
  (h10 : s = 1 + 4 * u)
  (h11 : 110 ≤ 81 * u + 25)
  (h12 : 81 * u + 25 ≤ 200) :
  m = 187 :=
sorry

end treasure_coins_problem_l450_450371


namespace ratio_of_volumes_l450_450810

noncomputable def volumeSphere (p : ℝ) : ℝ := (4/3) * Real.pi * (p^3)

noncomputable def volumeHemisphere (p : ℝ) : ℝ := (1/2) * (4/3) * Real.pi * (3*p)^3

theorem ratio_of_volumes (p : ℝ) (hp : p > 0) : volumeSphere p / volumeHemisphere p = 2 / 27 :=
by
  sorry

end ratio_of_volumes_l450_450810


namespace range_f_equals_0_to_1_l450_450442

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(-abs x)

theorem range_f_equals_0_to_1 (a : ℝ) (h : 1 < a) : 
  set.range (f a) = set.Ioc 0 1 := 
sorry

end range_f_equals_0_to_1_l450_450442


namespace range_of_a_l450_450628

variable (a x : ℝ)
def A (a : ℝ) := {x : ℝ | 2 * a ≤ x ∧ x ≤ a ^ 2 + 1}
def B (a : ℝ) := {x : ℝ | (x - 2) * (x - (3 * a + 1)) ≤ 0}

theorem range_of_a (a : ℝ) : (∀ x, x ∈ A a → x ∈ B a) ↔ (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) := by sorry

end range_of_a_l450_450628


namespace sum_projections_eq_semi_perimeter_l450_450627

variables {α : Type*}

-- Defined an equilateral triangle ABC
structure EquilateralTriangle (α : Type*) :=
(A B C : α)

-- Projections of point P onto the sides AB, BC, and AC
variables (ABC : EquilateralTriangle α) (P C1 A1 B1 : α)

-- Side length of triangle ABC
variable (a : ℝ)

-- Projections of P onto sides AB, BC, and AC are C1, A1, and B1 respectively
def projection_condition (P C1 A1 B1 : α) (ABC : EquilateralTriangle α) := 
  -- Assume some way to define projections here, abstractly
  true

-- Semi-perimeter of the triangle
def semi_perimeter (a : ℝ) : ℝ := 3 * a / 2

theorem sum_projections_eq_semi_perimeter
  (ABC : EquilateralTriangle α)
  (P C1 A1 B1 : α)
  (a : ℝ)
  (h : projection_condition P C1 A1 B1 ABC) :
  let AC1 := dist A C1,
      BA1 := dist B A1,
      CB1 := dist C B1 in
  AC1 + BA1 + CB1 = semi_perimeter a := by
  sorry

end sum_projections_eq_semi_perimeter_l450_450627


namespace equal_perimeters_and_areas_l450_450858

theorem equal_perimeters_and_areas (n : ℕ) (a b c : ℝ) (k l m : ℕ)
  (h1 : 2 * n = k + l + m)
  (h2 : k + l - m ≥ 0)
  (h3 : k + m - l ≥ 0)
  (h4 : l + m - k ≥ 0)
  (h5 : ∀ i j, i ≠ j → [a, b, c][i] ≠ [a, b, c][j]) : 
  (∑ i in finset.range n, (if i % 2 = 0 then [a,b,c][i % 3] else [a,b,c][i % 3])) = 
  (∑ i in finset.range n, (if i % 2 = 1 then [a,b,c][i % 3] else [a,b,c][i % 3])) 
  ∧ polygon_area n [a,b,c] [k,l,m] = polygon_area n [a,b,c] [k,l,m] :=
by
  sorry

end equal_perimeters_and_areas_l450_450858


namespace probability_within_half_mile_l450_450530

-- Defining the problem in Lean 4
theorem probability_within_half_mile (track_length : ℝ) (travel_distance : ℝ) (target_distance : ℝ) : 
  track_length = 3 ∧ travel_distance = 0.5 ∧ target_distance = 2.5 →
  ∀ (start_loc : ℝ), (0 ≤ start_loc ∧ start_loc < track_length) →
  let end_loc := (start_loc + travel_distance) % track_length in
  (end_loc ≤ target_distance + 0.5 ∨ end_loc ≥ target_distance - 0.5) →
  1 / 3 := 
sorry

end probability_within_half_mile_l450_450530


namespace opposite_difference_five_times_l450_450342

variable (a b : ℤ) -- Using integers for this example

theorem opposite_difference_five_times (a b : ℤ) : (-a - 5 * b) = -(a) - (5 * b) := 
by
  -- The proof details would be filled in here
  sorry

end opposite_difference_five_times_l450_450342


namespace cosine_shift_correctness_l450_450821

noncomputable def shift_cosine_graph (x : ℝ) : Prop :=
  (∀ x : ℝ, cos(2 * x - 2 * (π / 6)) = cos(2 * x + π / 3))

theorem cosine_shift_correctness : shift_cosine_graph :=
by
  -- Add the proof here.
  sorry

end cosine_shift_correctness_l450_450821


namespace cos_seven_pi_over_six_l450_450590

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 :=
by
  sorry

end cos_seven_pi_over_six_l450_450590


namespace perpendicular_to_both_if_parallel_l450_450562

-- Definitions of lines and their relations.
def is_perpendicular (l m : ℝ) : Prop := ∃ (θ : ℝ) (hθ: 0 < θ ∧ θ < π), l = m + θ
def is_parallel (l₁ l₂ : ℝ) : Prop := ∀ (x y: ℝ), l₁ = l₂ + x + y ∧ x = 0 ∧ y = 0

-- Problem Statement
theorem perpendicular_to_both_if_parallel (l₁ l₂ m : ℝ) (h1 : is_parallel l₁ l₂) (h2 : is_perpendicular m l₁) : is_perpendicular m l₂ :=
begin
  sorry
end

end perpendicular_to_both_if_parallel_l450_450562


namespace tangent_line_eqn_l450_450424

theorem tangent_line_eqn :
  let y := λ x : ℝ, x - Real.cos x 
  let point := (Real.pi / 2, Real.pi / 2)
  point ∈ set_of (λ p, p.snd = y p.fst)
  → ∃ a b c : ℝ, a = 2 ∧ b = -1 ∧ c = -Real.pi / 2 ∧ 
    ∀ x : ℝ, (a * x + b * (y x - Real.pi / 2) = c) := 
by sorry

end tangent_line_eqn_l450_450424


namespace cost_of_3200_pencils_l450_450863

theorem cost_of_3200_pencils :
  (let unit_price := 45 / 150
   in unit_price * 3200) = 960 :=
by
  let unit_price := (45 : ℝ) / 150
  have h1 : unit_price = 0.3, by norm_num
  have h2 : unit_price * 3200 = 960, by norm_num
  exact (congr_arg (λ x, x * 3200) h1).trans h2

end cost_of_3200_pencils_l450_450863


namespace tan_sum_diff_l450_450204

theorem tan_sum_diff (m n : ℝ) :
  (∃ φ ψ : ℝ, (tan φ + tan ψ = m) ∧ (tan φ * tan ψ = n)) →
  tan (φ + ψ) = m / (1 - n) ∧ tan (φ - ψ) = sqrt (m^2 - 4*n) / (1 + n) :=
by
  sorry

end tan_sum_diff_l450_450204


namespace combined_probability_correct_l450_450754

-- Definitions of rolling a fair six-sided die
def roll_first_die : ℕ := sorry  -- Assume this returns a value between 1 and 6
def roll_second_die : ℕ := sorry -- Assume this returns a value between 1 and 6

-- Predicate for a number being less than three
def less_than_three (n : ℕ) : Prop := n < 3

-- Predicate for a number being greater than three
def greater_than_three (n : ℕ) : Prop := n > 3

-- Calculate the probability as a common fraction
def fraction (numerator denominator : ℕ) : ℚ := numerator / denominator

-- The probability of the desired outcome can be expressed as
noncomputable def desired_probability : ℚ := fraction 1 6

-- Proving that the combined probability of rolling less than three on the first die and greater than three on the second die is 1/6
theorem combined_probability_correct :
    ∃ (p : ℚ), p = fraction 1 6 ∧
    (∀ (die1 die2 : ℕ), 
        (less_than_three die1) ∧ (greater_than_three die2) → 
        (p = fraction 1 6)) :=
by
    use desired_probability
    split
    sorry
    intros die1 die2
    sorry

end combined_probability_correct_l450_450754


namespace easter_egg_battle_probability_l450_450473

theorem easter_egg_battle_probability (players : Type) [fintype players] [decidable_eq players]
  (egg_strength : players → ℕ) (p1 : players) (p2 : players) (n : ℕ) [decidable (p1 ≠ p2)] :
  (∀ i in finset.range n, egg_strength p1 > egg_strength p2) →
  let prob11thWin := 11 / 12 in
  11 / 12 = prob11thWin :=
by sorry

end easter_egg_battle_probability_l450_450473


namespace postal_service_revenue_l450_450441

theorem postal_service_revenue 
  (price_colored : ℝ := 0.50)
  (price_bw : ℝ := 0.35)
  (price_golden : ℝ := 2.00)
  (sold_colored : ℕ := 578833)
  (sold_bw : ℕ := 523776)
  (sold_golden : ℕ := 120456) : 
  (price_colored * (sold_colored : ℝ) + 
  price_bw * (sold_bw : ℝ) + 
  price_golden * (sold_golden : ℝ) = 713650.10) :=
by
  sorry

end postal_service_revenue_l450_450441


namespace exists_equidistant_point_l450_450826

-- Define three points A, B, and C in 2D space
variables {A B C P: ℝ × ℝ}

-- Assume the points A, B, and C are not collinear
def not_collinear (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) ≠ (C.1 - A.1) * (B.2 - A.2)

-- Define the concept of a point being equidistant from three given points
def equidistant (P A B C : ℝ × ℝ) : Prop :=
  dist P A = dist P B ∧ dist P B = dist P C

-- Define the intersection of the perpendicular bisectors of the sides of the triangle formed by A, B, and C
def circumcenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- placeholder for the actual construction

-- The main theorem statement: If A, B, and C are not collinear, then there exists a unique point P that is equidistant from A, B, and C
theorem exists_equidistant_point (h: not_collinear A B C) :
  ∃! P, equidistant P A B C := 
sorry

end exists_equidistant_point_l450_450826


namespace range_of_a_l450_450138

def difference_set (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2*x - 1/2 ∧ 1 - sqrt 2 / 2 < x ∧ x < 3 / 2}
def Q (a : ℝ) : Set ℝ := {x : ℝ | x^2 + (a - 1) * x - a < 0}

theorem range_of_a (a : ℝ) : difference_set P (Q a) = ∅ → 0 ≤ a := sorry

end range_of_a_l450_450138


namespace arrangement_count_l450_450879

theorem arrangement_count :
  ∃ (A_8_8 A_9_3 A_4_4 : Nat), 
  (A_8_8N * A_9_3M * A_4_4F = 14!) → 
  8! * 9! * 4! ∧
  (∀ i, (i = 1 ∨ i = 2 ∨ i = 3)) ∧
  (∀ j, j ≠ A ∧ j ≠ B ∧ j ≠ C ∧ j ≠ D) :=
sorry

end arrangement_count_l450_450879


namespace remainder_of_sum_of_residues_eq_7_l450_450496

-- Define the sequence of integers
def sequence := [13001, 13003, 13005, 13007, 13009, 13011, 13013, 13015, 13017, 13019, 13021]

-- Calculate their residues modulo 18
def residues := sequence.map (λ x, x % 18)

-- Sum these residues
def sum_residues := residues.sum

-- Define the remainder we expect when sum_residues is divided by 18
def expected_remainder := 7

-- The theorem we need to prove
theorem remainder_of_sum_of_residues_eq_7 : sum_residues % 18 = expected_remainder := by
  -- Skipping proof steps for now
  sorry

end remainder_of_sum_of_residues_eq_7_l450_450496


namespace find_median_of_100_l450_450303

noncomputable def median_of_set (s : Finset ℝ) : ℝ :=
if h : ∃ median, is_median s median then classical.some h else 0

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
∃ (f : ℤ), (f : ℝ) = (card s : ℝ) / 2 ∧
    ∃ (low : Finset ℝ) (high : Finset ℝ),
        low ⊆ s ∧ high ⊆ s ∧
        card low = floor ((card s : ℝ) / 2) ∧
        card high = ceil ((card s : ℝ) / 2) ∧
        (∀ x ∈ low, x ≤ m) ∧ (∀ x ∈ high, x ≥ m)

theorem find_median_of_100 (s : Finset ℝ) (h_size : s.card = 100)
(h1 : ∃ x ∈ s, median_of_set (s.erase x) = 78)
(h2 : ∃ y ∈ s, median_of_set (s.erase y) = 66) :
  median_of_set s = 72 := by
sorry

end find_median_of_100_l450_450303


namespace remainder_of_3x_minus_2y_mod_30_l450_450942

theorem remainder_of_3x_minus_2y_mod_30
  (p q : ℤ) (x y : ℤ)
  (hx : x = 60 * p + 53)
  (hy : y = 45 * q + 28) :
  (3 * x - 2 * y) % 30 = 13 :=
by 
  sorry

end remainder_of_3x_minus_2y_mod_30_l450_450942


namespace domain_of_f_l450_450928

noncomputable def f (x : ℝ) : ℝ := sqrt (2 - 2^x) + 1 / (Real.log x)

theorem domain_of_f : {x : ℝ | 2 - 2^x ≥ 0 ∧ Real.log x ≠ 0 ∧ x > 0} = {x : ℝ | 0 < x ∧ x < 1} :=
by {
  sorry
}

end domain_of_f_l450_450928


namespace range_of_slope_angle_l450_450453

theorem range_of_slope_angle (P A B : ℝ × ℝ) (hP : P = (0, -1)) (hA : A = (1, -2)) (hB : B = (2, 1)) :
  ∀ l, (∃ m : ℝ, l = λ x, m * (x - P.1) + P.2 ∧ (-1 ≤ m ∧ m ≤ 1) ∧ ∃ t ∈ Icc 0 1, (1 - t) • A + t • B = l x) →
  ∃ θ : ℝ, θ ∈ Icc 0 (π / 4) ∨ θ ∈ Icc (3 * π / 4) π ∧ real.tan θ = m :=
by
  sorry

end range_of_slope_angle_l450_450453


namespace unique_solution_exists_l450_450586

theorem unique_solution_exists (a x y z : ℝ) 
  (h1 : z = a * (x + 2 * y + 5 / 2)) 
  (h2 : x^2 + y^2 + 2 * x - y + a * (x + 2 * y + 5 / 2) = 0) :
  a = 1 → x = -3 / 2 ∧ y = -1 / 2 ∧ z = 0 := 
by
  sorry

end unique_solution_exists_l450_450586


namespace total_ducks_and_ducklings_l450_450382

theorem total_ducks_and_ducklings : 
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6 
  let total_ducklings := ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3
  let total_ducks := ducks1 + ducks2 + ducks3
  in total_ducks + total_ducklings = 99 := 
by {
  let ducks1 := 2
  let ducklings1 := 5
  let ducks2 := 6
  let ducklings2 := 3
  let ducks3 := 9
  let ducklings3 := 6
  let total_ducklings := ducks1 * ducklings1 + ducks2 * ducklings2 + ducks3 * ducklings3
  let total_ducks := ducks1 + ducks2 + ducks3
  show total_ducks + total_ducklings = 99
  sorry
}

end total_ducks_and_ducklings_l450_450382


namespace max_g_eq_25_l450_450576

-- Define the function g on positive integers.
def g : ℕ → ℤ
| n => if n < 12 then n + 14 else g (n - 7)

-- Prove that the maximum value of g is 25.
theorem max_g_eq_25 : ∀ n : ℕ, 1 ≤ n → g n ≤ 25 ∧ (∃ n : ℕ, 1 ≤ n ∧ g n = 25) := by
  sorry

end max_g_eq_25_l450_450576


namespace pentagon_parallels_l450_450765

variables {A B C D E : Type}

-- Defining the vertices and their respective sides as sets
variables (AB CE BC DA CD BE DE CA EA DB : Set (A × A))

-- Hypotheses (conditions)
def condition_1 : AB ∥ CE := sorry
def condition_2 : BC ∥ DA := sorry
def condition_3 : CD ∥ BE := sorry
def condition_4 : DE ∥ CA := sorry

-- The theorem statement to be proved
theorem pentagon_parallels (h1 : AB ∥ CE) (h2 : BC ∥ DA) (h3 : CD ∥ BE) (h4 : DE ∥ CA) : EA ∥ DB :=
sorry

end pentagon_parallels_l450_450765


namespace smallest_possible_value_of_N_l450_450238

-- Declares the context and required constraints
theorem smallest_possible_value_of_N (N : ℕ) (h1 : 70 < N) (h2 : 70 ∣ 21 * N) : N = 80 :=
by
  sorry

end smallest_possible_value_of_N_l450_450238


namespace compute_difference_of_squares_l450_450098

theorem compute_difference_of_squares :
  (23 + 15) ^ 2 - (23 - 15) ^ 2 = 1380 := by
  sorry

end compute_difference_of_squares_l450_450098


namespace decreasing_interval_of_g_l450_450796

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem decreasing_interval_of_g :
  ∀ k : ℤ, ∀ x : ℝ,
  (k * Real.pi + Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 6) →
  ∃ I : set ℝ, I = set.Icc (k * Real.pi + Real.pi / 3) (k * Real.pi + 5 * Real.pi / 6) ∧ 
    ∀ y ∈ I, g y ≥ g (y + (I.end - I.start) / 2) :=
by
  sorry

end decreasing_interval_of_g_l450_450796


namespace find_monthly_income_l450_450556

-- Define the percentages spent on various categories
def household_items_percentage : ℝ := 0.35
def clothing_percentage : ℝ := 0.18
def medicines_percentage : ℝ := 0.06
def entertainment_percentage : ℝ := 0.11
def transportation_percentage : ℝ := 0.12
def mutual_fund_percentage : ℝ := 0.05
def taxes_percentage : ℝ := 0.07

-- Define the savings amount
def savings_amount : ℝ := 12500

-- Total spent percentage
def total_spent_percentage := household_items_percentage + clothing_percentage + medicines_percentage + entertainment_percentage + transportation_percentage + mutual_fund_percentage + taxes_percentage

-- Percentage saved
def savings_percentage := 1 - total_spent_percentage

-- Prove that Ajay's monthly income is Rs. 208,333.33
theorem find_monthly_income (I : ℝ) (h : I * savings_percentage = savings_amount) : I = 208333.33 := by
  sorry

end find_monthly_income_l450_450556


namespace median_of_100_numbers_l450_450334

theorem median_of_100_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 100)
  (h_median_99_1 : ∀ num ∈ numbers, median (numbers.erase num) = 78 → num ∈ numbers)
  (h_median_99_2 : ∀ num ∈ numbers, median (numbers.erase num) = 66 → num ∈ numbers) :
  median numbers = 72 :=
sorry

end median_of_100_numbers_l450_450334


namespace integer_values_of_a_l450_450109

variable (a b c x : ℤ)

theorem integer_values_of_a (h : (x - a) * (x - 12) + 4 = (x + b) * (x + c)) : a = 7 ∨ a = 17 := by
  sorry

end integer_values_of_a_l450_450109


namespace burattino_suspects_cheating_l450_450072

theorem burattino_suspects_cheating :
  ∃ n : ℕ, (0.4 ^ n) < 0.01 ∧ n + 1 = 7 :=
by
  sorry

end burattino_suspects_cheating_l450_450072


namespace range_of_x_plus_y_l450_450978

open Real

theorem range_of_x_plus_y (x y : ℝ) (h : x - sqrt (x + 1) = sqrt (y + 1) - y) :
  -sqrt 5 + 1 ≤ x + y ∧ x + y ≤ sqrt 5 + 1 :=
by sorry

end range_of_x_plus_y_l450_450978


namespace arithmetic_seq_a5_l450_450156

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem arithmetic_seq_a5 (h1 : is_arithmetic_sequence a) (h2 : a 2 + a 8 = 12) :
  a 5 = 6 :=
by
  sorry

end arithmetic_seq_a5_l450_450156


namespace problem_proof_l450_450166

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def f (x : ℝ) : ℝ := 
if x > 0 then Real.log x 
else if x < 0 then -Real.log (-x)
else 0 -- This part is not needed in the conditions but included for completion.

-- Proof statement
theorem problem_proof : is_odd_function f ∧ (∀ x : ℝ, x > 0 → f x = Real.log x) → f (-Real.exp 1) = -1 := 
by 
  sorry

end problem_proof_l450_450166


namespace limit_arcsin_expr_l450_450851

noncomputable def limit_expr : ℝ :=
  lim (λ x : ℝ, (arcsin (2 * x) / (2^(-3 * x) - 1)) * log 2) (nhds_within 0 (set.univ))

theorem limit_arcsin_expr : limit_expr = - (2 / 3) :=
by
  sorry

end limit_arcsin_expr_l450_450851


namespace gift_boxes_in_3_days_l450_450940
-- Conditions:
def inchesPerBox := 18
def dailyWrapper := 90
-- "how many gift boxes will he be able to wrap every 3 days?"
theorem gift_boxes_in_3_days : 3 * (dailyWrapper / inchesPerBox) = 15 :=
by
  sorry

end gift_boxes_in_3_days_l450_450940


namespace probability_of_winning_11th_round_l450_450475

-- Definitions of the conditions
def player1_wins_ten_rounds (eggs : List ℕ) : Prop :=
  ∀ i, i < 10 → eggs.indexOf (eggs.nthLe 0 (i+1)) < eggs.indexOf (eggs.nthLe 1 (i+1))

def is_strongest (egg : ℕ) (eggs : List ℕ) : Prop :=
  egg = List.maximum (0 :: eggs)

-- The proof to show the probability of winning the 11th round
theorem probability_of_winning_11th_round
  (eggs : List ℕ) : player1_wins_ten_rounds eggs →
  (1 - 1 / (length eggs + 1) = 11 / 12) :=
by
  sorry

end probability_of_winning_11th_round_l450_450475


namespace solution_set_of_inequality_l450_450448

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 0) : 
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio (0 : ℝ) ∪ Set.Ici (1 / 2) :=
by sorry

end solution_set_of_inequality_l450_450448


namespace base6_problem_l450_450778

theorem base6_problem
  (x y : ℕ)
  (h1 : 453 = 2 * x * 10 + y) -- Constraint from base-6 to base-10 conversion
  (h2 : 0 ≤ x ∧ x ≤ 9) -- x is a base-10 digit
  (h3 : 0 ≤ y ∧ y ≤ 9) -- y is a base-10 digit
  (h4 : 4 * 6^2 + 5 * 6 + 3 = 177) -- Conversion result for 453_6
  (h5 : 2 * x * 10 + y = 177) -- Conversion from condition
  (hx : x = 7) -- x value from solution
  (hy : y = 7) -- y value from solution
  : (x * y) / 10 = 49 / 10 := 
by 
  sorry

end base6_problem_l450_450778


namespace geometric_series_sum_l450_450921

theorem geometric_series_sum :
  let a := -2
  let r := 3
  let n := 10
  ∑ i in finset.range n, a * r^i = -59048 :=
by
  sorry

end geometric_series_sum_l450_450921


namespace Dan_balloons_l450_450611

def Fred_balloons : ℕ := 10
def Sam_balloons : ℕ := 46
def total_balloons : ℕ := 72

theorem Dan_balloons : ∃ (Dan_balloons: ℕ), Fred_balloons + Sam_balloons + Dan_balloons = total_balloons :=
by
  use 16
  simp [Fred_balloons, Sam_balloons, total_balloons]
  sorry

end Dan_balloons_l450_450611


namespace quadrant_of_i_times_conjugate_specifiedComplex_l450_450206

def complexConjugate (z : ℂ) : ℂ := ⟨z.re, -z.im⟩

def specifiedComplex : ℂ := (2 : ℂ) / (-1 - I : ℂ)

theorem quadrant_of_i_times_conjugate_specifiedComplex : 
  let z := specifiedComplex
  let conj_z := complexConjugate z
  let result := I * conj_z
  Real (result).re > 0 ∧ Real (result).im < 0 := by
  sorry

end quadrant_of_i_times_conjugate_specifiedComplex_l450_450206


namespace Shekar_social_studies_marks_l450_450772

theorem Shekar_social_studies_marks :
  (∃ (m_sci m_eng m_bio soc : ℕ), m_sci = 65 ∧ m_eng = 67 ∧ m_bio = 75 
  ∧ avg : ℚ, avg = 73 
  ∧ let sum_marks : ℕ := 76 + m_sci + m_eng + m_bio in
    let total_marks : ℕ := 5 * avg in
    total_marks - sum_marks = 82) :=
sorry

end Shekar_social_studies_marks_l450_450772


namespace possible_third_side_l450_450187

theorem possible_third_side {x : ℕ} (h_option_A : x = 2) (h_option_B : x = 3) (h_option_C : x = 6) (h_option_D : x = 13) : 3 < x ∧ x < 13 ↔ x = 6 :=
by
  sorry

end possible_third_side_l450_450187


namespace age_difference_l450_450817

variable {A B C : ℕ}

-- Definition of conditions
def condition1 (A B C : ℕ) : Prop := A + B > B + C
def condition2 (A C : ℕ) : Prop := C = A - 16

-- The theorem stating the math problem
theorem age_difference (h1 : condition1 A B C) (h2 : condition2 A C) :
  (A + B) - (B + C) = 16 := by
  sorry

end age_difference_l450_450817


namespace trees_planted_l450_450456

theorem trees_planted (interval trail_length : ℕ) (h1 : interval = 30) (h2 : trail_length = 1200) : 
  trail_length / interval = 40 :=
by
  sorry

end trees_planted_l450_450456


namespace find_a_find_b_find_T_l450_450621

open Real

def S (n : ℕ) : ℝ := 2 * n^2 + n

def a (n : ℕ) : ℝ := if n = 1 then 3 else S n - S (n - 1)

def b (n : ℕ) : ℝ := 2^(n - 1)

def T (n : ℕ) : ℝ := (4 * n - 5) * 2^n + 5

theorem find_a (n : ℕ) (hn : n > 0) : a n = 4 * n - 1 :=
by sorry

theorem find_b (n : ℕ) (hn : n > 0) : b n = 2^(n-1) :=
by sorry

theorem find_T (n : ℕ) (hn : n > 0) (a_def : ∀ n, a n = 4 * n - 1) (b_def : ∀ n, b n = 2^(n-1)) : T n = (4 * n - 5) * 2^n + 5 :=
by sorry

end find_a_find_b_find_T_l450_450621


namespace arithmetic_problem_l450_450982

variable {a : ℕ → ℤ}
variable {d : ℤ}

# Given conditions: Arithmetic sequence with common difference -2 and sum of first 7 terms is 14.
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (finset.range n).sum a

theorem arithmetic_problem 
  (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a (-2)) 
  (h2 : sum_of_first_n_terms a 7 = 14) :
  a 1 + a 3 + a 5 = 6 :=
sorry

end arithmetic_problem_l450_450982


namespace find_all_possible_values_l450_450893

theorem find_all_possible_values (N : ℕ) (h1 : N > 1) 
  (divisors : list ℕ) (h2 : divisors.sorted (<)) 
  (h3 : divisors.head = 1) (h4 : divisors.last = N) 
  (h5 : ∑ i in finset.range (divisors.length - 1), Nat.gcd (divisors.nth_le i (nat.lt_sub_left_of_add_le i.succ_le_sub i h5)) (divisors.nth_le i.succ (nat.lt_trans i h5)) = N - 2) : 
  N = 3 :=
by
  sorry

end find_all_possible_values_l450_450893


namespace surface_area_of_solid_l450_450934

-- Definitions about the problem
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_rectangular_solid (a b c : ℕ) : Prop := is_prime a ∧ is_prime b ∧ is_prime c ∧ (a * b * c = 399)

-- Main statement of the problem
theorem surface_area_of_solid (a b c : ℕ) (h : is_rectangular_solid a b c) : 
  2 * (a * b + b * c + c * a) = 422 := sorry

end surface_area_of_solid_l450_450934


namespace find_CN_l450_450513

-- Define the isosceles trapezoid and the conditions
structure IsoscelesTrapezoid (A B C D M N : Point) where
  AD_eq_4 : dist A D = 4
  BC_eq_3 : dist B C = 3
  M_on_BD : M ∈ line BD
  N_on_BD : N ∈ line BD
  AM_perp_BD : angle A M = 90
  CN_perp_BD : angle C N = 90
  BM_DN_ratio : dist B M / dist D N = 2 / 3

-- Define the proof statement
theorem find_CN (A B C D M N : Point) (trapezoid : IsoscelesTrapezoid A B C D M N) :
  dist C N = sqrt 15 / 2 := 
sorry

end find_CN_l450_450513


namespace choose_captains_l450_450278

open Locale.Nat.Combinatorics

theorem choose_captains (n r : ℕ) (hn : n = 15) (hr : r = 4) : Nat.choose n r = 1365 := by
  rw [hn, hr]
  norm_num
  sorry

end choose_captains_l450_450278


namespace statement_B_statement_C_statement_D_l450_450998

variables {α : Type*} [inner_product_space ℝ α]

def vector (α : Type*) [inner_product_space ℝ α] := α

variables (a b c : vector α)

-- Condition: non-zero vectors
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)

-- Statement B: if a is parallel to b and b is parallel to c, then a is parallel to c
def parallel (u v : vector α) : Prop := ∃ k : ℝ, u = k • v

theorem statement_B (h1 : parallel a b) (h2 : parallel b c) : parallel a c := sorry

-- Statement C: (a / |a|) represents a unit vector in the direction of a
theorem statement_C (ha : a ≠ 0) : ∥a / ∥a∥∥ = 1 := 
begin
  rw [norm_smul_inv_norm ha],
end

-- Statement D: if a is parallel to b, the magnitude of the projection of a onto b is |a|
def projection (u v : vector α) : vector α := ((inner_product u v) / (∥v∥^2)) • v

theorem statement_D (h1 : parallel a b) : ∥projection a b∥ = ∥a∥ := sorry

end statement_B_statement_C_statement_D_l450_450998


namespace part1_solution_part2_solution_l450_450854

-- Part (1)
def f (x : ℝ) : ℝ := |x + 1| + |2 * x - 4|

theorem part1_solution (x : ℝ) : f(x) >= 6 ↔ x ≤ -1 ∨ x ≥ 3 := 
by 
  intros
  sorry

-- Part (2)
theorem part2_solution (a b c : ℝ) (h : a + 2 * b + 4 * c = 8) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) : 
  ∀x, ∃m, m = min (1/a + 1/b + 1/c) (x) := 
by
  sorry

end part1_solution_part2_solution_l450_450854


namespace median_of_set_l450_450316

open List

def is_median (l : List ℝ) (m : ℝ) : Prop :=
  l.length % 2 = 1 ∧ (sorted l) ∧ (l.nth (l.length / 2)).iget = m

theorem median_of_set (s : List ℝ) (h_len : s.length = 100)
  (h1 : ∃ n, is_median (s.erase n) 78)
  (h2 : ∃ n, is_median (s.erase n) 66) :
  is_median s 72 :=
sorry

end median_of_set_l450_450316


namespace third_side_length_l450_450180

theorem third_side_length (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) (h3 : 3 < x < 13) : (5 < x + 8) ∧ (x < 5 + 8) ∧ (5 < x + 3) ∧ (x < 8 + 5) := 
by
  sorry

end third_side_length_l450_450180


namespace median_of_100_numbers_l450_450308

theorem median_of_100_numbers (x : Fin 100 → ℝ)
  (h1 : ∀ i j, i ≠ j → x i = 78 → x j = 66 → i = 51 ∧ j = 50 ∨ i = 50 ∧ j = 51)
  (h2 : ∀ i, i ≠ 51 → x 51 = 78)
  (h3 : ∀ i, i ≠ 50 → x 50 = 66) :
  (x 50 + x 51) / 2 = 72 :=
by sorry

end median_of_100_numbers_l450_450308


namespace triangle_third_side_length_l450_450198

theorem triangle_third_side_length (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 8) :
  (a + b > c) → (a + c > b) → (b + c > a) → c = 6 :=
by
  intros h₃ h₄ h₅
  rw [h₁, h₂] at *
  -- Simplified inequalities from the solution step
  have h₆ : 5 + 8 > c := by rw [h₁, h₂]; exact h₃
  have h₇ : 8 + c > 5 := by rw h₂; exact h₄
  have h₈ : 5 + c > 8 := by rw h₁; exact h₅
  sorry

end triangle_third_side_length_l450_450198


namespace quadratic_symmetry_range_of_quadratic_quadratic_transformation_l450_450155

def quadratic_func_f (a x : ℝ) : ℝ := x^2 - a * x + 3

theorem quadratic_symmetry (a : ℝ) :
  (∀ x : ℝ, quadratic_func_f a (4 - x) = quadratic_func_f a x) → a = 4 :=
by
  sorry

theorem range_of_quadratic (a : ℝ) :
  (a = 4) → (∀ x ∈ set.Icc (0:ℝ) 3, -1 ≤ quadratic_func_f a x ∧ quadratic_func_f a x ≤ 3) :=
by
  sorry

theorem quadratic_transformation (a : ℝ) :
  (a = 4) → (∀ x : ℝ, quadratic_func_f a x = (x - 2)^2 - 1) :=
by
  sorry

end quadratic_symmetry_range_of_quadratic_quadratic_transformation_l450_450155


namespace single_colony_habitat_limit_reach_time_l450_450021

noncomputable def doubling_time (n : ℕ) : ℕ := 2^n

theorem single_colony_habitat_limit_reach_time :
  ∀ (S : ℕ), ∀ (n : ℕ), doubling_time (n + 1) = S → doubling_time (2 * (n - 1)) = S → n + 1 = 16 :=
by
  intros S n H1 H2
  sorry

end single_colony_habitat_limit_reach_time_l450_450021


namespace average_speed_of_horse_l450_450542

/-- Definitions of the conditions given in the problem. --/
def pony_speed : ℕ := 20
def pony_head_start_hours : ℕ := 3
def horse_chase_hours : ℕ := 4

-- Define a proof problem for the average speed of the horse.
theorem average_speed_of_horse : (pony_head_start_hours * pony_speed + horse_chase_hours * pony_speed) / horse_chase_hours = 35 := by
  -- Setting up the necessary distances
  let pony_head_start_distance := pony_head_start_hours * pony_speed
  let pony_additional_distance := horse_chase_hours * pony_speed
  let total_pony_distance := pony_head_start_distance + pony_additional_distance
  -- Asserting the average speed of the horse
  let horse_average_speed := total_pony_distance / horse_chase_hours
  show horse_average_speed = 35
  sorry

end average_speed_of_horse_l450_450542


namespace airplane_total_seats_l450_450047

theorem airplane_total_seats (s : ℕ) 
  (h1 : 30 = 30) 
  (h2 : 0.20 * s = 20 * s / 100) 
  (h3 : 0.70 * s = 70 * s / 100) 
  : s = 300 
  :=
sorry

end airplane_total_seats_l450_450047


namespace solution_inequality_1_range_of_a_l450_450648

noncomputable def f (x : ℝ) : ℝ := abs x + abs (x - 2)

theorem solution_inequality_1 :
  {x : ℝ | f x < 3} = {x : ℝ | - (1/2) < x ∧ x < (5/2)} :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x < a) → a > 2 :=
by
  sorry

end solution_inequality_1_range_of_a_l450_450648


namespace surface_area_volume_ratio_equal_l450_450887

variable (l r R : ℝ)
variable (h : ℝ) (hl2 : h = Real.sqrt (l ^ 2 - r ^ 2))

def cone_surface_area (r l : ℝ) : ℝ :=
  π * r ^ 2 + π * r * l

def sphere_surface_area (R : ℝ) : ℝ :=
  4 * π * R ^ 2

def cone_volume (r l h : ℝ) : ℝ :=
  1 / 3 * π * r ^ 2 * h

def sphere_volume (R : ℝ) : ℝ :=
  4 / 3 * π * R ^ 3

theorem surface_area_volume_ratio_equal
  (hl2 : h = Real.sqrt (l ^ 2 - r ^ 2))
  (inscribed : r * Real.sqrt (l^2 - r^2) = R * (l + r)) :
  cone_surface_area r l / sphere_surface_area R = cone_volume r l (Real.sqrt (l^2 - r^2)) / sphere_volume R :=
by
  sorry

end surface_area_volume_ratio_equal_l450_450887


namespace zero_point_condition_sufficient_not_necessary_l450_450633

theorem zero_point_condition_sufficient_not_necessary (a : ℝ) : a < -3 → 
  (∃ x₀, x₀ ∈ set.Ioo (-1 : ℝ) 2 ∧ (a * x₀ + 3 = 0)) ∧ (¬(∀ x₀, x₀ ∈ set.Ioo (-1 : ℝ) 2 → (a * x₀ + 3 = 0))) := sorry

end zero_point_condition_sufficient_not_necessary_l450_450633


namespace inradius_of_triangle_l450_450698

theorem inradius_of_triangle (A p s r : ℝ) 
  (h1 : A = (1/2) * p) 
  (h2 : p = 2 * s) 
  (h3 : A = r * s) : 
  r = 1 :=
by
  sorry

end inradius_of_triangle_l450_450698


namespace burattino_suspects_cheating_after_seventh_draw_l450_450064

theorem burattino_suspects_cheating_after_seventh_draw
  (balls : ℕ)
  (draws : ℕ)
  (a : ℝ)
  (p_limit : ℝ)
  (h_balls : balls = 45)
  (h_draws : draws = 6)
  (h_a : a = (39.choose 6 : ℝ) / (45.choose 6 : ℝ))
  (h_p_limit : p_limit = 0.01) :
  ∃ (n : ℕ), n > 5 ∧ a^n < p_limit := by
  sorry

end burattino_suspects_cheating_after_seventh_draw_l450_450064


namespace briana_investment_l450_450944

theorem briana_investment :
  ∃ (B : ℝ), 
    let emma_investment := 300
    let emma_return := 2 * (0.15 * emma_investment)
    let briana_return := 2 * (0.10 * B)
    in (emma_return - briana_return = 10) ∧ (B = 400) :=
begin
  let emma_investment := 300,
  let emma_return := 2 * (0.15 * emma_investment),
  let B := 400,
  let briana_return := 2 * (0.10 * B),
  have h_return_diff: emma_return - briana_return = 10,
  { sorry },
  use B,
  refine ⟨h_return_diff, rfl⟩,
end

end briana_investment_l450_450944


namespace geom_seq_sum_first_10_terms_l450_450714

variable (a : ℕ → ℝ) (a₁ : ℝ) (q : ℝ)
variable (h₀ : a₁ = 1/4)
variable (h₁ : ∀ n, a (n + 1) = a₁ * q ^ n)
variable (S : ℕ → ℝ)
variable (h₂ : S n = a₁ * (1 - q ^ n) / (1 - q))

theorem geom_seq_sum_first_10_terms :
  a 1 = 1 / 4 →
  (a 3) * (a 5) = 4 * ((a 4) - 1) →
  S 10 = 1023 / 4 :=
by
  sorry

end geom_seq_sum_first_10_terms_l450_450714


namespace parallel_lines_ac_xy_l450_450352

theorem parallel_lines_ac_xy 
  (A B C M N X Y : Type)
  [angle_bisectors A B C M N]
  [min_distance_point B X A M]
  [min_distance_point B Y C N] :
  parallel AC XY :=
  sorry

end parallel_lines_ac_xy_l450_450352


namespace burattino_suspects_cheating_l450_450074

theorem burattino_suspects_cheating :
  ∃ n : ℕ, (0.4 ^ n) < 0.01 ∧ n + 1 = 7 :=
by
  sorry

end burattino_suspects_cheating_l450_450074


namespace smallest_t_for_sin_theta_circle_l450_450795

theorem smallest_t_for_sin_theta_circle :
  ∃ t > 0, ∀ θ, 0 ≤ θ ∧ θ ≤ t → sin θ = r ↔ (r, θ) completes_circle ∧ t = π :=
by
  sorry

end smallest_t_for_sin_theta_circle_l450_450795


namespace general_term_formula_sum_bn_l450_450713

section
variable {a : ℕ → ℝ} (b : ℕ → ℝ)

-- Conditions of the problem
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 2 ∧ a 4 = 16 ∧ (∀ n, a (n + 1) = a n * q)

def bn (a : ℕ → ℝ) (n : ℕ) := (1 : ℝ) / ((Real.log2 (a n)) * (Real.log2 (a (n + 1))))

-- Given conditions
axiom h : is_geometric_sequence a 2

-- Formulation of problems as proofs
theorem general_term_formula : ∀ n, a n = 2 ^ n :=
sorry

theorem sum_bn (n : ℕ) : ∑ k in Finset.range n, bn a k = (n : ℝ) / (n + 1) :=
sorry

end

end general_term_formula_sum_bn_l450_450713


namespace area_of_rectangle_l450_450768

-- Define the conditions
variables (DA FD AE : ℝ) (h1 : DA = 20) (h2 : FD = 12) (h3 : AE = 12)

-- Definition of the diameter of the semicircle
def diameter (DA FD AE : ℝ) : ℝ := FD + DA + AE

-- Radius of the semicircle
def radius (DA FD AE : ℝ) : ℝ := (diameter DA FD AE) / 2

-- Calculation using Pythagorean theorem for AB
def AB (DA radius : ℝ) : ℝ := 2 * real.sqrt (radius ^ 2 - DA ^ 2)

-- Area of the rectangle
def area_rectangle (DA AB : ℝ) : ℝ := AB * DA

-- The proof problem statement
theorem area_of_rectangle (DA FD AE : ℝ) (h1 : DA = 20) (h2 : FD = 12) (h3 : AE = 12) :
  area_rectangle DA (AB DA (radius DA FD AE)) = 80 * real.sqrt 21 := by
  sorry

end area_of_rectangle_l450_450768


namespace triangle_perimeter_l450_450208

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem triangle_perimeter :
  let C := {xy : ℝ × ℝ | (xy.1^2 / 6) + (xy.2^2 / 2) = 1}
  let F1 : ℝ × ℝ := (-2, 0)
  let F2 : ℝ × ℝ := (2, 0)
  ∃ A B : ℝ × ℝ,
    A ∈ C ∧ B ∈ C ∧ (∃ m : ℝ, A.2 = m * (A.1 + 2) ∧ B.2 = m * (B.1 + 2)) →
      distance A F2 + distance B F2 + distance A B = 4 * Real.sqrt 6 := 
by
  intros
  let C := {xy : ℝ × ℝ | (xy.1^2 / 6) + (xy.2^2 / 2) = 1}
  let F1 : ℝ × ℝ := (-2, 0)
  let F2 : ℝ × ℝ := (2, 0)
  have h1: ∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ (∃ m : ℝ, A.2 = m * (A.1 + 2) ∧ B.2 = m * (B.1 + 2)) := sorry
  use h1
  sorry

end triangle_perimeter_l450_450208


namespace third_side_length_l450_450181

theorem third_side_length (x : ℝ) (h1 : 5 > 0) (h2 : 8 > 0) (h3 : 3 < x < 13) : (5 < x + 8) ∧ (x < 5 + 8) ∧ (5 < x + 3) ∧ (x < 8 + 5) := 
by
  sorry

end third_side_length_l450_450181


namespace find_real_solutions_l450_450601

noncomputable def f (x : ℝ) : ℝ :=
  ∑ i in Finset.range 100, (i + 3 : ℕ) / (x - (i + 1 : ℕ))

theorem find_real_solutions :
  ∃ n : ℕ, n = 101 ∧ ∀ x : ℝ, f x = x → true := by
  sorry

end find_real_solutions_l450_450601


namespace max_min_value_l450_450965

noncomputable def f (A B x a b : ℝ) : ℝ :=
  A * Real.sqrt (x - a) + B * Real.sqrt (b - x)

theorem max_min_value (A B a b : ℝ) (hA : A > 0) (hB : B > 0) (ha_lt_b : a < b) :
  (∀ x, a ≤ x ∧ x ≤ b → f A B x a b ≤ Real.sqrt ((A^2 + B^2) * (b - a))) ∧
  min (f A B a a b) (f A B b a b) ≤ f A B x a b :=
  sorry

end max_min_value_l450_450965


namespace partI_partII_l450_450618

-- Part I: Verify the equation of line given midpoint condition
def parabola (x y : ℝ) := y^2 = -x
def line1 (x y : ℝ) := x + 2 * y + 2 = 0
def midpoint (x1 y1 x2 y2 : ℝ) := ((x1 + x2) / 2, (y1 + y2) / 2)

-- Part II: Verify the area of trapezoid
def line2 (m x y : ℝ) := x = m * y - 1
def trapezoidArea (m : ℝ) := (2 * m^2 + 5) / 4 * sqrt (m^2 + 4)

theorem partI (x1 y1 x2 y2 : ℝ) (h1 : parabola x1 y1) (h2 : parabola x2 y2) (h3 : midpoint x1 y1 x2 y2 = (-4, 1)) :
  line1 x1 y1 ∧ line1 x2 y2 := 
sorry

theorem partII (m : ℝ) (x1 y1 x2 y2 A1x A1y B1x B1y : ℝ) 
(h1 : line2 m x1 y1) (h2 : line2 m x2 y2) (h3 : parabola x1 y1) (h4 : parabola x2 y2) 
(h5 : proj_initial_conditions) :
  area_of_trapezoid AA_1 B_1 BB A = trapezoidArea m :=
sorry

end partI_partII_l450_450618


namespace sin_B_gt_sin_A_plus_B_l450_450294

theorem sin_B_gt_sin_A_plus_B 
  (A B : ℝ) 
  (hAacute : 0 < A ∧ A < π / 2) 
  (hBobtuse : π / 2 < B ∧ B < π) : 
  sin B > sin (A + B) := 
by
  sorry

end sin_B_gt_sin_A_plus_B_l450_450294


namespace find_a_l450_450707

theorem find_a (a t : ℝ) 
    (h1 : (a + t) / 2 = 2020) 
    (h2 : t / 2 = 11) : 
    a = 4018 := 
by 
    sorry

end find_a_l450_450707


namespace coeff_x_82_l450_450595

noncomputable def poly : Polynomial ℤ :=
  (∏ i in Finset.range 15, (Polynomial.X ^ (i + 1) - (i + 1)))

theorem coeff_x_82 : (Polynomial.coeff poly 82) = 1426 := by
  sorry

end coeff_x_82_l450_450595
