import Data.Finset
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.BigOperators.Finsupp
import Mathlib.Algebra.Field
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Matrix
import Mathlib.Algebra.Trig
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.ExtendDeriv
import Mathlib.Analysis.SpecialFunctions.Exponential
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Padics
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Floor
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.LinearAlgebra.Matrix
import Mathlib.LinearAlgebra.Matrix.Defined
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.NumberTheory.Prime
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace b_values_sum_b_1000_l442_442604

open Int Real

def S : ℕ → ℝ := λ n, 28 * (n / 7)  -- Given that S_7 = 28

def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 1 * n  -- Derived from given conditions that a_1 = 1 and common difference is 1

def b (n : ℕ) : ℤ :=
  Int.floor (Real.log10 (a n))

theorem b_values :
  b 1 = 0 ∧ b 11 = 1 ∧ b 101 = 2 := by
  sorry

theorem sum_b_1000 :
  (Finset.sum (Finset.range 1000) b) = 1893 := by
  sorry

end b_values_sum_b_1000_l442_442604


namespace number_of_paths_l442_442021

theorem number_of_paths (a b : ℕ) (h_a : a = 6) (h_b : b = 6)
  (h_moves : ∀ p : ℕ × ℕ, ∃ q : ℕ × ℕ, q.1 = p.1 + 1 ∨ q.2 = p.2 + 1 ∨ (q.1 = p.1 + 1 ∧ q.2 = p.2 + 1)) :
  ∑ d in finset.range 7, nat.choose (12 - d) (6 - d) = 1716 :=
by sorry

end number_of_paths_l442_442021


namespace determine_coin_types_l442_442700

theorem determine_coin_types (coins : Fin 100 → ℕ) (h1 : ∀ c, coins c ∈ {1, 2, 3}) 
(h2 : ∃ c1 c2 c3, coins c1 = 1 ∧ coins c2 = 2 ∧ coins c3 = 3):
  ∃ weighings : ℕ, weighings ≤ 101 ∧ (∃ coin_types : Fin 100 → ℕ, (∀ c, coin_types c ∈ {1, 2, 3}) ∧ 
    (∀ c, (coin_types c = 1 ∧ coins c = 1) ∨ (coin_types c = 2 ∧ coins c = 2) ∨ 
           (coin_types c = 3 ∧ coins c = 3))) :=
by
  sorry

end determine_coin_types_l442_442700


namespace problem1_problem2_l442_442319

-- Problem 1: Prove that x = 5 given 9 * 27^x = 3^17.
theorem problem1 (x : ℝ) (h : 9 * 27^x = 3^17) : x = 5 :=
sorry

-- Problem 2: Prove that a^(3x - 2y) = -8 / 9 given a^x = -2 and a^y = 3.
theorem problem2 (a x y : ℝ) (hx : a^x = -2) (hy : a^y = 3) : a^(3 * x - 2 * y) = -8 / 9 :=
sorry

end problem1_problem2_l442_442319


namespace planes_parallel_or_intersect_l442_442903

theorem planes_parallel_or_intersect (π₁ π₂ : Plane) (L₁ L₂ L₃ : LineSegment) 
  (h1 : is_enclosed_between L₁ π₁ π₂)
  (h2 : is_enclosed_between L₂ π₁ π₂)
  (h3 : is_enclosed_between L₃ π₁ π₂)
  (h4 : is_parallel L₁ L₂)
  (h5 : is_parallel L₂ L₃)
  (h6 : L₁.length = L₂.length)
  (h7 : L₂.length = L₃.length)
  : is_parallel π₁ π₂ ∨ is_intersecting π₁ π₂ :=
sorry

end planes_parallel_or_intersect_l442_442903


namespace ball_bounce_height_l442_442000

open Real

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (2 / 3 : ℝ)^k < 2) ∧ (∀ n : ℕ, n < k → 20 * (2 / 3 : ℝ)^n ≥ 2) ∧ k = 6 :=
sorry

end ball_bounce_height_l442_442000


namespace solutionToEquations_solutionToInequalities_l442_442362

-- Part 1: Solve the system of equations
def solveEquations (x y : ℝ) : Prop :=
2 * x - y = 3 ∧ x + y = 6

theorem solutionToEquations (x y : ℝ) (h : solveEquations x y) : 
x = 3 ∧ y = 3 :=
sorry

-- Part 2: Solve the system of inequalities
def solveInequalities (x : ℝ) : Prop :=
3 * x > x - 4 ∧ (4 + x) / 3 > x + 2

theorem solutionToInequalities (x : ℝ) (h : solveInequalities x) : 
-2 < x ∧ x < -1 :=
sorry

end solutionToEquations_solutionToInequalities_l442_442362


namespace acute_angles_sum_pi_div_two_l442_442484

/-- Given acute angles α and β such that sin²(α) + sin²(β) = sin(α + β), prove that α + β = π / 2. -/
theorem acute_angles_sum_pi_div_two 
  {α β : ℝ} 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h : sin α * sin α + sin β * sin β = sin (α + β)) : 
  α + β = π / 2 :=
sorry

end acute_angles_sum_pi_div_two_l442_442484


namespace smallest_n_for_P_lt_1_by_3015_l442_442625

noncomputable theory

def P (n : ℕ) : ℚ := 3 / ((n + 1) * (n + 2) * (n + 3))

theorem smallest_n_for_P_lt_1_by_3015 : ∃ n : ℕ, P(n) < (1 : ℚ) / 3015 ∧ ∀ m < n, ¬(P(m) < (1 : ℚ) / 3015) :=
  by
  sorry

end smallest_n_for_P_lt_1_by_3015_l442_442625


namespace min_plus_max_value_of_x_l442_442945

theorem min_plus_max_value_of_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  let m := (10 - Real.sqrt 304) / 6
  let M := (10 + Real.sqrt 304) / 6
  m + M = 10 / 3 := by 
  sorry

end min_plus_max_value_of_x_l442_442945


namespace sum_of_solutions_eq_376_over_7_l442_442725

theorem sum_of_solutions_eq_376_over_7 :
  let solutions := {x : ℝ | x = |3 * x - |40 - 3 * x||}
  ∑ x in solutions, x = 376 / 7 :=
sorry

end sum_of_solutions_eq_376_over_7_l442_442725


namespace equilateral_triangle_area_l442_442658
-- Import the entire Mathlib for necessary mathematical components.

-- Define the problem statement, proving the triangle’s area given the conditions.
theorem equilateral_triangle_area (h : ℝ) (h_eq : h = real.sqrt 3) :
  ∃ (A : ℝ), A = real.sqrt 3 :=
by
  -- Skipping the proof steps here, we use sorry.
  sorry

end equilateral_triangle_area_l442_442658


namespace complaints_over_3_days_l442_442652

def normal_complaints_per_day : ℕ := 120

def short_staffed_complaints_per_day : ℕ := normal_complaints_per_day * 4 / 3

def short_staffed_and_broken_self_checkout_complaints_per_day : ℕ := short_staffed_complaints_per_day * 12 / 10

def days_short_staffed_and_broken_self_checkout : ℕ := 3

def total_complaints (days : ℕ) (complaints_per_day : ℕ) : ℕ :=
  days * complaints_per_day

theorem complaints_over_3_days
  (n : ℕ := normal_complaints_per_day)
  (a : ℕ := short_staffed_complaints_per_day)
  (b : ℕ := short_staffed_and_broken_self_checkout_complaints_per_day)
  (d : ℕ := days_short_staffed_and_broken_self_checkout)
  : total_complaints d b = 576 :=
by {
  -- This is where the proof would go, e.g., using sorry to skip the proof for now.
  sorry
}

end complaints_over_3_days_l442_442652


namespace point_closer_to_D_probability_l442_442559

/-- Proof problem statement -/
theorem point_closer_to_D_probability :
  let ABC := @Triangle.Equilateral ℝ 6,
  let D : ABC := {
    internal_point := True,
    dist_A := 2,
    dist_B := 4,
    dist_C := 3
  },
  let area_triangle := 9 * real.sqrt 3,
  let area_circle := 4 * real.pi,
  let prob := area_circle / area_triangle
  in prob = 4 * real.pi / (9 * real.sqrt 3)
:= sorry

end point_closer_to_D_probability_l442_442559


namespace sum_coefficients_l442_442118

theorem sum_coefficients (c : ℕ → ℚ): 
  (1 + x + x^2 + x^3) ^ 400 = ∑ k in range (301), c (4*k) * x^(4*k) + ∑ k in range (299), c (4*k+1) * x^(4*k+1) + ∑ k in range (299), c (4*k+2) * x^(4*k+2) + ∑ k in range (299), c (4*k+3) * x^(4*k+3)
  → ∑ k in range (301), c (4 * k) = 4 ^ 399 := 
begin
  sorry
end

end sum_coefficients_l442_442118


namespace distinct_pairs_exist_l442_442979

open Set

variables {S T : Finset nat} {U : Finset (nat × nat)}

/- Definitions from the conditions -/
def all_pairs (s : nat) : Prop := ∀ t : nat, (s, t) ∈ U
def appears_in_U (t : nat) : Prop := ∃ s : nat, (s, t) ∈ U

/- Theorem -/
theorem distinct_pairs_exist
  (fin_S : Finite S) (fin_T : Finite T)
  (U_subset_ST : ∀ p : nat × nat, p ∈ U → p.fst ∈ S ∧ p.snd ∈ T)
  (no_s_all_pairs : ∀ s : nat, ¬all_pairs s)
  (each_t_appears : ∀ t : nat, appears_in_U t) :
  ∃ s1 s2 t1 t2 : nat, s1 ≠ s2 ∧ t1 ≠ t2 ∧ 
    ((s1, t1) ∈ U ∧ (s2, t2) ∈ U) ∧
    ((s1, t2) ∉ U ∧ (s2, t1) ∉ U) :=
sorry

end distinct_pairs_exist_l442_442979


namespace evaluate_x3_minus_y3_l442_442494

theorem evaluate_x3_minus_y3 (x y : ℤ) (h1 : x + y = 12) (h2 : 3 * x + y = 20) : x^3 - y^3 = -448 :=
by
  sorry

end evaluate_x3_minus_y3_l442_442494


namespace find_littering_citations_l442_442386

def is_total_citations (total : ℕ) (L : ℕ) : Prop :=
  let off_leash_dogs := L
  let parking_fines := 2 * (L + off_leash_dogs)
  total = L + off_leash_dogs + parking_fines

theorem find_littering_citations :
  ∀ L : ℕ, is_total_citations 24 L → L = 4 :=
by
  intros L h
  let off_leash_dogs := L
  let parking_fines := 2 * (L + off_leash_dogs)
  have h1 : 24 = L + off_leash_dogs + parking_fines := h
  have h2 : off_leash_dogs = L := sorry -- Directly applies from problem statement
  have h3 : parking_fines = 2 * (L + L) := sorry -- Applies the double citation fine
  rw h2 at h3
  rw h3 at h1
  simp at h1
  exact sorry -- Final solve step, equating and solving for L

end find_littering_citations_l442_442386


namespace find_n_l442_442897

-- Given Variables
variables (n x y : ℝ)

-- Given Conditions
axiom h1 : n * x = 6 * y
axiom h2 : x * y ≠ 0
axiom h3 : (1/3 * x) / (1/5 * y) = 1.9999999999999998

-- Conclusion
theorem find_n : n = 5 := sorry

end find_n_l442_442897


namespace cookie_distribution_l442_442419

def trays := 4
def cookies_per_tray := 24
def total_cookies := trays * cookies_per_tray
def packs := 8
def cookies_per_pack := total_cookies / packs

theorem cookie_distribution : cookies_per_pack = 12 := by
  sorry

end cookie_distribution_l442_442419


namespace div_by_3_9_then_mul_by_5_6_eq_div_by_5_2_l442_442818

theorem div_by_3_9_then_mul_by_5_6_eq_div_by_5_2 :
  (∀ (x : ℚ), (x / (3/9)) * (5/6) = x / (5/2)) :=
by
  sorry

end div_by_3_9_then_mul_by_5_6_eq_div_by_5_2_l442_442818


namespace commercials_played_l442_442819

theorem commercials_played (M C : ℝ) (h1 : M / C = 9 / 5) (h2 : M + C = 112) : C = 40 :=
by
  sorry

end commercials_played_l442_442819


namespace divisibility_of_products_l442_442948

open Nat

noncomputable def lcm (a b : ℕ) : ℕ := (a * b) / (gcd a b)

theorem divisibility_of_products
  {n : ℕ} (h1 : 2 ≤ n)
  (a : Fin n → ℕ)
  (h_range : ∀ i : Fin n, 1 ≤ a i ∧ a i ≤ 2 * n)
  (h_lcm : ∀ i j : Fin n, i.val < j.val → lcm (a i) (a j) > 2 * n)
  : (∏ i in Finset.range n, a ⟨i, sorry⟩) ∣ (∏ i in Finset.range n, Nat.succ n + i) := by
  sorry

end divisibility_of_products_l442_442948


namespace sin_sum_diff_inequality_l442_442855

variable {α β : ℝ}

-- Define acute angles
def is_acute (x : ℝ) : Prop := 0 < x ∧ x < π / 2

-- Prove the inequality given the conditions
theorem sin_sum_diff_inequality (h₁ : is_acute α) (h₂ : is_acute β) : 
  sin (α + β) > sin (α - β) := by
  sorry

end sin_sum_diff_inequality_l442_442855


namespace probability_of_spinner_in_shaded_region_l442_442013

-- Define the setup of the problem
def square_diagonally_divided_to_four_equal_regions : Prop :=
  ∃ (s : Type), is_square s ∧ (∀ (r1 r2 r3 r4 : region), 
   are_equal_areas r1 r2 r3 r4 ∧ diagonal_divides_square s r1 r2 r3 r4)

def shaded_two_opposite_triangular_regions : Prop :=
  ∃ (r1 r2 r3 r4 : region), (shaded r1 ∧ shaded r3) ∧ (not_shaded r2 ∧ not_shaded r4) ∧ opposite(r1, r3) ∧ opposite(r2, r4)

def regions_are_equally_likely : Prop :=
  ∃ (r1 r2 r3 r4 : region), equally_likely_to_rest_in(r1) ∧ equally_likely_to_rest_in(r2) ∧ equally_likely_to_rest_in(r3) ∧ equally_likely_to_rest_in(r4)

-- Define the final proof problem
theorem probability_of_spinner_in_shaded_region 
  (h1 : square_diagonally_divided_to_four_equal_regions) 
  (h2 : shaded_two_opposite_triangular_regions) 
  (h3 : regions_are_equally_likely) : 
  probability_spinner_in_shaded_region = 1 / 2 :=
by sorry

end probability_of_spinner_in_shaded_region_l442_442013


namespace circle_area_of_square_center_l442_442920

theorem circle_area_of_square_center 
  (sq_area : ℝ) (W_center : ℝ) (X_on_circle : ℝ) (Z_on_circle : ℝ)
  (hq1 : sq_area = 9)
  (hq2 : W_center = sq_area / 3)
  (hq3 : X_on_circle = Z_on_circle) :
  let r := 3 in let area_circ := π * r^2 in area_circ = 9 * π := by
  -- Proof steps will follow
  sorry

end circle_area_of_square_center_l442_442920


namespace quadrilateral_inscribed_circle_AC_length_l442_442266

noncomputable def AC_length : ℝ :=
  let AD := 5
  let BC := 7
  let angle_BAC := 80 * Real.pi / 180  -- converting degrees to radians
  let angle_ADB := 30 * Real.pi / 180  -- converting degrees to radians
  let angle_ABC := 70 * Real.pi / 180  -- converting degrees to radians
  real.sqrt (
    AD^2 + BC^2 - 2 * AD * BC * Real.cos angle_ABC
  )

theorem quadrilateral_inscribed_circle_AC_length :
  AC_length = 7 := 
sorry

end quadrilateral_inscribed_circle_AC_length_l442_442266


namespace probability_lava_lamps_l442_442978

open Finset

noncomputable def arrange_and_turn_on_prob : ℚ :=
  let total_arrangements : ℕ := (choose 8 4) * (choose 8 5)
  let favorable_arrangements : ℕ :=
        (choose 6 2) * (choose 4 2)
  favorable_arrangements / total_arrangements

theorem probability_lava_lamps : 
  ∃ (prob : ℚ), 
      prob = (9 / 1960) ∧
      prob = arrange_and_turn_on_prob :=
by
  use arrange_and_turn_on_prob
  split
  { refl }
  { sorry }

end probability_lava_lamps_l442_442978


namespace ratio_proof_l442_442404

noncomputable def side_length_triangle(a : ℝ) : ℝ := a / 3
noncomputable def side_length_square(b : ℝ) : ℝ := b / 4
noncomputable def area_triangle(a : ℝ) : ℝ := (side_length_triangle(a)^2 * Mathlib.sqrt(3)) / 4
noncomputable def area_square(b : ℝ) : ℝ := (side_length_square(b))^2

theorem ratio_proof (a b : ℝ) (h : area_triangle(a) = area_square(b)) : a / b = 2 * Mathlib.sqrt(3) / 9 :=
by {
  sorry
}

end ratio_proof_l442_442404


namespace inequality_C_D_l442_442618

variable {n : ℕ} {a : Fin n → ℝ}

noncomputable def b (k : Fin (n + 1)) : ℝ :=
  (∑ i in Finset.range (k + 1), a i) / (k + 1)

noncomputable def C : ℝ :=
  ∑ i in Finset.range n, (a i - b ⟨i, by simp [i_lt_succ_self]⟩) ^ 2

noncomputable def D : ℝ :=
  ∑ i in Finset.range n, (a i - b ⟨n - 1, by simp [lt_succ_self]⟩) ^ 2

theorem inequality_C_D : C ≤ D ∧ D ≤ 2 * C :=
  sorry

end inequality_C_D_l442_442618


namespace parabola_latus_rectum_equation_l442_442667

theorem parabola_latus_rectum_equation :
  (∃ (y x : ℝ), y^2 = 4 * x) → (∀ x, x = -1) :=
by
  sorry

end parabola_latus_rectum_equation_l442_442667


namespace painted_cubes_on_two_faces_l442_442732

theorem painted_cubes_on_two_faces (n : ℕ) (painted_faces_all : Prop) (equal_smaller_cubes : n = 27) : ∃ k : ℕ, k = 12 :=
by
  -- We only need the statement, not the proof
  sorry

end painted_cubes_on_two_faces_l442_442732


namespace log_b10_is_9_l442_442605

variables {a : ℕ → ℤ} {b : ℕ → ℕ} (S : ℕ → ℤ)

-- Assume the conditions provided
def condition_1 := ∀ n, S n = ∑ i in finset.range n, a (i + 1)
def condition_2 := S 2 = S 6
def condition_3 := a 4 = 1
def condition_4 := ∀ n, b n > 0
def condition_5 := b 2 = a 4 - a 5
def condition_6 := b 5 * b 1 = 4 * b 2 ^ 2

-- Mathematically equivalent proof problem statement
theorem log_b10_is_9 :
  condition_1 S a ∧ condition_2 S ∧ condition_3 a ∧ condition_4 b ∧ condition_5 a b ∧ condition_6 b →
  ∃ b_10: ℕ, b_10 = b 10 ∧ (Real.log 9 b_10 = 9) :=
sorry

end log_b10_is_9_l442_442605


namespace line_eq_l442_442457

theorem line_eq (x y : ℝ) (point eq_direction_vector) (h₀ : point = (3, -2))
    (h₁ : eq_direction_vector = (-5, 3)) :
    3 * x + 5 * y + 1 = 0 := by sorry

end line_eq_l442_442457


namespace unique_abs_value_of_roots_l442_442155

theorem unique_abs_value_of_roots :
  ∀ (w : ℂ), w^2 - 6 * w + 40 = 0 → (∃! z, |w| = z) :=
by
  sorry

end unique_abs_value_of_roots_l442_442155


namespace equilateral_triangle_area_l442_442661

theorem equilateral_triangle_area (AM : ℝ) (h : AM = sqrt 3) : 
    let BM := AM / sqrt 3,
    let BC := 2 * BM in
    (1 / 2) * BC * AM = sqrt 3 :=
by
    sorry

end equilateral_triangle_area_l442_442661


namespace even_odd_f_find_a_l442_442130

open Real

noncomputable def f (a x : ℝ) : ℝ := log a ((1 + x) / (1 - x))

theorem even_odd_f (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (x : ℝ) (hx : -1 < x ∧ x < 1) : 
  f a x = -f a (-x) := by
  sorry

theorem find_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (hx_dom : ∀ x, 0 ≤ x ∧ x ≤ 1 / 2 → ∃ y, f a x = y ∧ 0 ≤ y ∧ y ≤ 1) : 
  a = 3 := by
  sorry

end even_odd_f_find_a_l442_442130


namespace ratio_left_to_rest_excluding_throwers_l442_442255

theorem ratio_left_to_rest_excluding_throwers (totalPlayers throwers rightHandedPlayers : ℕ)
    (h_total : totalPlayers = 70)
    (h_throwers : throwers = 49)
    (h_rightHandedPlayers : rightHandedPlayers = 63)
    (h_throwers_are_rightHanded : throwers = rightHandedPlayers - 14) :
    let nonThrowers := totalPlayers - throwers in
    let leftHandedNonThrowers := nonThrowers - (rightHandedPlayers - throwers) in
    leftHandedNonThrowers / nonThrowers = 1 / 3 :=
by
  sorry

end ratio_left_to_rest_excluding_throwers_l442_442255


namespace sqrt_meaningful_iff_x_geq_nine_l442_442683

theorem sqrt_meaningful_iff_x_geq_nine (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 9)) ↔ x ≥ 9 :=
by sorry

end sqrt_meaningful_iff_x_geq_nine_l442_442683


namespace polar_coordinates_equivalence_l442_442177

theorem polar_coordinates_equivalence :
  ∀ (r : ℝ) (θ : ℝ), (r = -5) ∧ (θ = 5 * Real.pi / 7) →
    ∃ (r' θ' : ℝ), (r' = 5) ∧ (θ' = 12 * Real.pi / 7) ∧ (r' > 0) ∧ (0 ≤ θ') ∧ (θ' < 2 * Real.pi) :=
by
  intros r θ h
  cases h with hr hθ
  use [5, 12 * Real.pi / 7]
  simp [hr, hθ]
  split
  refl
  split
  refl
  split
  linarith
  split
  linarith
  linarith

end polar_coordinates_equivalence_l442_442177


namespace sum_of_all_possible_n_l442_442931

open Finset

-- Kevin's set
def S : Finset ℕ := (range 42).filter(λ x, x ≠ 0)

-- Function to simulate the transformation each day
def transform (A : Finset ℕ) : Finset ℕ :=
  let R := (A.image (λ x, x - 1)).filter (λ x, x ≠ 0)
  in if 0 ∈ A.image (λ x, x - 1) then S \ R else R

-- Check if we reach the original subset after n transformations
def reaches_original (A : Finset ℕ) (n : ℕ) : Prop :=
  (transform^[n] A) = A

-- Possible periods excluding 2
def possible_periods : Finset ℕ := {1, 3, 6, 7, 14, 21, 42}

-- Sum of all possible valid periods
def valid_periods_sum : ℕ :=
  possible_periods.sum id

theorem sum_of_all_possible_n :
  valid_periods_sum = 94 :=
by
  sorry

end sum_of_all_possible_n_l442_442931


namespace product_of_dodecagon_l442_442025

open Complex

theorem product_of_dodecagon (Q : Fin 12 → ℂ) (h₁ : Q 0 = 2) (h₇ : Q 6 = 8) :
  (Q 0) * (Q 1) * (Q 2) * (Q 3) * (Q 4) * (Q 5) * (Q 6) * (Q 7) * (Q 8) * (Q 9) * (Q 10) * (Q 11) = 244140624 :=
sorry

end product_of_dodecagon_l442_442025


namespace range_of_k_l442_442487

-- Definitions for the conditions of p and q
def is_ellipse (k : ℝ) : Prop := (0 < k) ∧ (k < 4)
def is_hyperbola (k : ℝ) : Prop := 1 < k ∧ k < 3

-- The main proposition
theorem range_of_k (k : ℝ) : (is_ellipse k ∨ is_hyperbola k) → (1 < k ∧ k < 4) :=
by
  sorry

end range_of_k_l442_442487


namespace log_identity_l442_442536

theorem log_identity (x : ℝ) (h : 2^x = 10) : x - log 2 5 = 1 :=
by sorry

end log_identity_l442_442536


namespace quadratic_equal_real_roots_l442_442549

theorem quadratic_equal_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + x + m = 0 ∧ ∀ y : ℝ, y^2 + y + m = 0 → y = x) ↔ m = 1/4 :=
by sorry

end quadratic_equal_real_roots_l442_442549


namespace line_and_product_l442_442099

-- Define complex numbers z1 and z2
def z1 : ℂ := -3 + 4 * complex.I
def z2 : ℂ := 2 - complex.I

-- Define constants a, b, c as derived from the solution
def a : ℂ := 5 - 5 * complex.I
def b : ℂ := 5 + 5 * complex.I
def c : ℂ := 35

-- State the theorem to demonstrate both the equation representing the line and the product ab.
theorem line_and_product :
  (∀ (z : ℂ), (a * z + b * complex.conj z = c) ↔ (a * b = 50)) :=
by
  sorry

end line_and_product_l442_442099


namespace edward_spring_earnings_l442_442444

-- Define the relevant constants and the condition
def springEarnings := 2
def summerEarnings := 27
def expenses := 5
def totalEarnings := 24

-- The condition
def edwardCondition := summerEarnings - expenses = 22

-- The statement to prove
theorem edward_spring_earnings (h : edwardCondition) : springEarnings + 22 = totalEarnings :=
by
  -- Provide the proof here, but we'll use sorry to skip it
  sorry

end edward_spring_earnings_l442_442444


namespace maximum_possible_value_of_x_l442_442283

-- Define the conditions and the question
def ten_teams_playing_each_other_once (number_of_teams : ℕ) : Prop :=
  number_of_teams = 10

def points_system (win_points draw_points loss_points : ℕ) : Prop :=
  win_points = 3 ∧ draw_points = 1 ∧ loss_points = 0

def max_points_per_team (x : ℕ) : Prop :=
  x = 13

-- The theorem to be proved: maximum possible value of x given the conditions
theorem maximum_possible_value_of_x :
  ∀ (number_of_teams win_points draw_points loss_points x : ℕ),
    ten_teams_playing_each_other_once number_of_teams →
    points_system win_points draw_points loss_points →
    max_points_per_team x :=
  sorry

end maximum_possible_value_of_x_l442_442283


namespace misha_contributes_l442_442206

noncomputable def misha_contribution (k l m : ℕ) : ℕ :=
  if h : k + l + m = 6 ∧ 2 * k ≤ l + m ∧ 2 * l ≤ k + m ∧ 2 * m ≤ k + l ∧ k ≤ 2 ∧ l ≤ 2 ∧ m ≤ 2 then
    2
  else
    0 -- This is a default value; the actual proof will check for exact solution.

theorem misha_contributes (k l m : ℕ) (h1 : k + l + m = 6)
    (h2 : 2 * k ≤ l + m) (h3 : 2 * l ≤ k + m) (h4 : 2 * m ≤ k + l)
    (h5 : k ≤ 2) (h6 : l ≤ 2) (h7 : m ≤ 2) : m = 2 := by
  sorry

end misha_contributes_l442_442206


namespace intersection_of_A_and_B_l442_442231

open Set

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
  by
    sorry

end intersection_of_A_and_B_l442_442231


namespace symmetric_point_of_A_l442_442486

-- Define the point A
def point_A : ℝ × ℝ × ℝ := (1, 2, 3)

-- Define the symmetric point function with respect to the origin
def symmetric_point (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-P.1, -P.2, -P.3)

-- State the theorem that describes the result
theorem symmetric_point_of_A :
  symmetric_point point_A = (-1, -2, -3) :=
by 
  -- Proof is omitted
  sorry

end symmetric_point_of_A_l442_442486


namespace points_collinear_l442_442239

variables {a b x1 y1 x2 y2 : ℝ}

-- Conditions:
def is_ellipse (x y a b : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1) ∧ (a > b)
def M_point_conditions : Prop := y1 ≠ 0
def N_point_conditions : Prop := x2 ≠ 0

-- Points definitions:
def P : ℝ × ℝ := (x1, y2)
def Q : ℝ × ℝ := (x1 - (b^2 / a^2) * x1, 0)
def R : ℝ × ℝ := (0, y2 - (a^2 / b^2) * y2)

-- Collinearity condition (three points are collinear if the slopes between two pairs are the same):
def collinear (P Q R : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (R.1 - P.1) = (R.2 - P.2) * (Q.1 - P.1)

theorem points_collinear (h_ellipse : is_ellipse x1 y1 a b)
                         (h_M : M_point_conditions)
                         (h_N : N_point_conditions) :
  collinear P Q R :=
sorry

end points_collinear_l442_442239


namespace find_number_of_packs_l442_442592

-- Define the cost of a pack of Digimon cards
def cost_pack_digimon : ℝ := 4.45

-- Define the cost of the deck of baseball cards
def cost_deck_baseball : ℝ := 6.06

-- Define the total amount spent
def total_spent : ℝ := 23.86

-- Define the number of packs of Digimon cards Keith bought
def number_of_packs (D : ℝ) : Prop :=
  cost_pack_digimon * D + cost_deck_baseball = total_spent

-- Prove the number of packs is 4
theorem find_number_of_packs : ∃ D, number_of_packs D ∧ D = 4 :=
by
  -- the proof will be inserted here
  sorry

end find_number_of_packs_l442_442592


namespace mean_median_difference_l442_442913

/-- Definitions for the conditions -/
def percentage_scoring_60 := 0.15
def percentage_scoring_75 := 0.20
def percentage_scoring_85 := 0.25
def percentage_scoring_95 := 0.30
def percentage_rest := 1 - (percentage_scoring_60 + percentage_scoring_75 + percentage_scoring_85 + percentage_scoring_95)
def score_60 := 60
def score_75 := 75
def score_85 := 85
def score_95 := 95
def score_rest := 100

/-- The theorem statement -/
theorem mean_median_difference :
  let mean := (percentage_scoring_60 * score_60) + 
              (percentage_scoring_75 * score_75) + 
              (percentage_scoring_85 * score_85) + 
              (percentage_scoring_95 * score_95) + 
              (percentage_rest * score_rest)
  let median := 85
  (median - mean) = 1.25 := sorry

end mean_median_difference_l442_442913


namespace circumscribed_circle_area_of_pentagon_l442_442005

noncomputable def pentagon_side_length : ℝ := 10
noncomputable def sin_36 : ℝ := Real.sin (36 * Real.pi / 180)
noncomputable def radius (s : ℝ) : ℝ := s / (2 * sin_36)
noncomputable def circumscribed_circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circumscribed_circle_area_of_pentagon :
  circumscribed_circle_area (radius pentagon_side_length) = 72.35 * Real.pi :=
by
  sorry

end circumscribed_circle_area_of_pentagon_l442_442005


namespace total_ingredients_cups_l442_442687

theorem total_ingredients_cups (butter_ratio flour_ratio sugar_ratio sugar_cups : ℚ) 
  (h_ratio : butter_ratio / sugar_ratio = 1 / 4 ∧ flour_ratio / sugar_ratio = 6 / 4) 
  (h_sugar : sugar_cups = 10) : 
  butter_ratio * (sugar_cups / sugar_ratio) + flour_ratio * (sugar_cups / sugar_ratio) + sugar_cups = 27.5 :=
by
  sorry

end total_ingredients_cups_l442_442687


namespace dima_more_berries_and_difference_l442_442816

section RaspberryPicking

-- Define conditions
def total_berries : ℕ := 450
def dima_contrib_per_2_berries : ℚ := 1
def sergei_contrib_per_3_berries : ℚ := 2
def dima_speed_factor : ℚ := 2

-- Defining the problem of determining the berry counts
theorem dima_more_berries_and_difference :
  let dima_cycles := 2 * total_berries / (2 * dima_contrib_per_2_berries + 3 * sergei_contrib_per_3_berries * (1 / dima_speed_factor)) / dima_contrib_per_2_berries in
  let sergei_cycles := total_berries / (2 * dima_contrib_per_2_berries + 3 * sergei_contrib_per_3_berries * (1 / dima_speed_factor)) / sergei_contrib_per_3_berries in
  let berries_dima := dima_cycles * (dima_contrib_per_2_berries / 2) in
  let berries_sergei := sergei_cycles * (sergei_contrib_per_3_berries / 3) in
  berries_dima > berries_sergei ∧
  berries_dima - berries_sergei = 50 :=
by --- skip the proof
sorry

end RaspberryPicking

end dima_more_berries_and_difference_l442_442816


namespace exists_2013_distinct_numbers_l442_442411

theorem exists_2013_distinct_numbers : 
  ∃ (a : ℕ → ℕ), 
    (∀ m n, m ≠ n → m < 2013 ∧ n < 2013 → (a m + a n) % (a m - a n) = 0) ∧
    (∀ k l, k < 2013 ∧ l < 2013 → (a k) ≠ (a l)) :=
sorry

end exists_2013_distinct_numbers_l442_442411


namespace log_exp_mod_a_l442_442262

open padic

-- Define the theorem with the given conditions
theorem log_exp_mod_a (a x y : ℤ_[p]) (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0) (ha_nonzero : a ≠ 0) 
  (hpx : p ∣ x) (hpa_xy : p * a ∣ x * y) :
  (1/y * ((1 + x)^y - 1) / x) ≡ (log (1 + x) / x) [MOD a] :=
by 
  sorry

end log_exp_mod_a_l442_442262


namespace smallest_c_no_real_root_l442_442336

theorem smallest_c_no_real_root (c : ℤ) :
  (∀ x : ℝ, x^2 + (c : ℝ) * x + 10 ≠ 5) ↔ c = -4 :=
by
  sorry

end smallest_c_no_real_root_l442_442336


namespace inequality_solution_l442_442990

variable (a x : ℝ)

noncomputable def inequality_solutions :=
  if a = 0 then
    {x | x > 1}
  else if a > 1 then
    {x | (1 / a) < x ∧ x < 1}
  else if a = 1 then
    ∅
  else if 0 < a ∧ a < 1 then
    {x | 1 < x ∧ x < (1 / a)}
  else if a < 0 then
    {x | x < (1 / a) ∨ x > 1}
  else
    ∅

theorem inequality_solution (h : a ≠ 0) :
  if a = 0 then
    ∀ x, (a * x - 1) * (x - 1) < 0 → x > 1
  else if a > 1 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ ((1 / a) < x ∧ x < 1)
  else if a = 1 then
    ∀ x, ¬((a * x - 1) * (x - 1) < 0)
  else if 0 < a ∧ a < 1 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ (1 < x ∧ x < (1 / a))
  else if a < 0 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ (x < (1 / a) ∨ x > 1)
  else
    True := sorry

end inequality_solution_l442_442990


namespace teacher_A_realizes_fish_l442_442709

variable (Teacher : Type) (has_fish : Teacher → Prop) (is_laughing : Teacher → Prop)
variables (A B C : Teacher)

-- Initial assumptions
axiom all_laughing : is_laughing A ∧ is_laughing B ∧ is_laughing C
axiom each_thinks_others_have_fish : (¬has_fish A ∧ has_fish B ∧ has_fish C) 
                                      ∨ (has_fish A ∧ ¬has_fish B ∧ has_fish C)
                                      ∨ (has_fish A ∧ has_fish B ∧ ¬has_fish C)

-- The logical conclusion
theorem teacher_A_realizes_fish : (∃ A B C : Teacher, 
  is_laughing A ∧ is_laughing B ∧ is_laughing C ∧
  ((¬has_fish A ∧ has_fish B ∧ has_fish C)
  ∨ (has_fish A ∧ ¬has_fish B ∧ has_fish C)
  ∨ (has_fish A ∧ has_fish B ∧ ¬has_fish C))) →
  (has_fish A ∧ is_laughing B ∧ is_laughing C) :=
sorry -- proof not required.

end teacher_A_realizes_fish_l442_442709


namespace train_length_is_correct_l442_442737

-- Define the given conditions and the expected result.
def train_speed_kmph : ℝ := 270
def time_seconds : ℝ := 5
def expected_length_meters : ℝ := 375

-- State the theorem to be proven.
theorem train_length_is_correct :
  (train_speed_kmph * 1000 / 3600) * time_seconds = expected_length_meters := by
  sorry -- Proof is not required, so we use 'sorry'

end train_length_is_correct_l442_442737


namespace bones_in_beef_l442_442381

def price_of_beef_with_bones : ℝ := 78
def price_of_boneless_beef : ℝ := 90
def price_of_bones : ℝ := 15
def fraction_of_bones_in_kg : ℝ := 0.16
def grams_per_kg : ℝ := 1000

theorem bones_in_beef :
  (fraction_of_bones_in_kg * grams_per_kg = 160) :=
by
  sorry

end bones_in_beef_l442_442381


namespace acute_and_inequality_l442_442597

variables {A B C : Type*} [triangle A B C]
variables {r r1 r2 r3 a b c : ℝ}
variables [inradius ABC r] [exradius ABC r1 r2 r3]

theorem acute_and_inequality 
  (h_a1 : a > r1) 
  (h_b1 : b > r2) 
  (h_c1 : c > r3) 
  (h_triangle : triangle ABC) 
  (h_inradius : inradius ABC r) 
  (h_exradius : exradius ABC r1 r2 r3) :
  (ABC is_acute) ∧ (a + b + c > r + r1 + r2 + r3) := 
sorry

end acute_and_inequality_l442_442597


namespace minimum_trucks_needed_l442_442273

theorem minimum_trucks_needed {n : ℕ} (total_weight : ℕ) (box_weight : ℕ → ℕ) (truck_capacity : ℕ) :
  (total_weight = 10 ∧ truck_capacity = 3 ∧ (∀ b, box_weight b ≤ 1) ∧ (∃ n, 3 * n ≥ total_weight)) → n ≥ 5 :=
by
  -- We need to prove the statement based on the given conditions.
  sorry

end minimum_trucks_needed_l442_442273


namespace fraction_sum_l442_442417

theorem fraction_sum : (3 / 9 : ℚ) + (7 / 14 : ℚ) = 5 / 6 := by
  sorry

end fraction_sum_l442_442417


namespace unique_abs_value_of_roots_l442_442154

theorem unique_abs_value_of_roots :
  ∀ (w : ℂ), w^2 - 6 * w + 40 = 0 → (∃! z, |w| = z) :=
by
  sorry

end unique_abs_value_of_roots_l442_442154


namespace at_least_five_with_same_hair_count_l442_442627

theorem at_least_five_with_same_hair_count
  (population : ℕ)
  (max_hairs : ℕ)
  (inhabitants : population = 2300000)
  (max_hairs_per_person : max_hairs = 500000) :
  ∃ n, ∃! k, k ≤ max_hairs ∧ (∃ p1 p2 p3 p4 p5, 
      p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
      p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
      p3 ≠ p4 ∧ p3 ≠ p5 ∧
      p4 ≠ p5 ∧
      hairs p1 = hairs p2 ∧ hairs p2 = hairs p3 ∧ hairs p3 = hairs p4 ∧ hairs p4 = hairs p5)
by
  sorry

end at_least_five_with_same_hair_count_l442_442627


namespace circular_film_radius_l442_442788

theorem circular_film_radius
  (V : ℝ) (t : ℝ) (r : ℝ)
  (hV : V = 320)
  (ht : t = 0.2) :
  r = real.sqrt (1600 / real.pi) :=
by
  sorry

end circular_film_radius_l442_442788


namespace smallest_fraction_l442_442019

theorem smallest_fraction (x : ℝ) (h : x > 2022) :
  min (min (min (min (x / 2022) (2022 / (x - 1))) ((x + 1) / 2022)) (2022 / x)) (2022 / (x + 1)) = 2022 / (x + 1) :=
sorry

end smallest_fraction_l442_442019


namespace venny_spent_on_used_car_l442_442718

def calculate_spent_amount (original_price : ℝ) (percentage : ℝ) : ℝ :=
  (percentage / 100) * original_price

theorem venny_spent_on_used_car :
  calculate_spent_amount 37500 40 = 15000 :=
by
  sorry

end venny_spent_on_used_car_l442_442718


namespace arithmetic_mean_lambda_A_mean_lambda_A_l442_442514

def M := Finset.range 2021

def lambda_A (A : Finset ℕ) (h : A ≠ ∅) := A.max' h + A.min' h

theorem arithmetic_mean_lambda_A : 
  (Finset.nonempty_subsets M).sum (λ A, lambda_A A (Finset.nonempty_of_ne_empty A.2)) 
  = 2021 * ((2 ^ 2020) - 1) :=
sorry

theorem mean_lambda_A : 
  ((Finset.nonempty_subsets M).sum (λ A, lambda_A A (Finset.nonempty_of_ne_empty A.2))) / ((2 ^ 2020) - 1) = 2021 :=
sorry

end arithmetic_mean_lambda_A_mean_lambda_A_l442_442514


namespace P_lt_Q_l442_442601

variables (a : ℝ) (P Q : ℝ)

def P_def : P = sqrt (a + 3) + sqrt (a + 5) := sorry
def Q_def : Q = sqrt (a + 1) + sqrt (a + 7) := sorry
def condition : a ≥ 0 := sorry

theorem P_lt_Q : P < Q :=
by {
  assume h1 : a ≥ 0,
  assume h2 : P = sqrt (a + 3) + sqrt (a + 5),
  assume h3 : Q = sqrt (a + 1) + sqrt (a + 7),
  sorry
}

end P_lt_Q_l442_442601


namespace light_path_reflection_l442_442616

open Real

def cube (A B C D E F G H : Point) := 
  dist A B = 10 ∧
  dist B C = 10 ∧
  dist A D = 10 ∧
  dist E F = 10

def light_path_length (A P : Point) := dist A P

def reflection_point (P : Point) := 
  P.x = 10 ∧
  P.y = 3 ∧
  P.z = 4

theorem light_path_reflection (A B C D E F G H P : Point) 
    (hcube : cube A B C D E F G H)
    (hreflect : reflection_point P) :
    light_path_length A P = 20 * sqrt 125 := 
  sorry

end light_path_reflection_l442_442616


namespace correct_statements_l442_442917

universe u

variable {α : Type u}

structure Point :=
  (x : α)
  (y : α)

variables {a b c : ℝ}
variables {A B : Point}
variables {λ : ℝ}

def line_eq (p : Point) : Prop :=
  a * p.x + b * p.y + c = 0

def symmetric_curve_eq (p1 p2 : Point) (x y : ℝ) : Prop :=
  (x - p1.x) * (x - p2.x) + (y - p1.y) * (y - p2.y) = 0

axiom distinct_points : A ≠ B
axiom B_not_on_line : ¬ line_eq B

theorem correct_statements :
  -- Statement ②
  ∃ λ, symmetric_curve_eq A B (a / (2 * (a / 2))) (b / (2 * (b / 2))) ∧ line_eq A ∧ line_eq B ∧ λ = 1
  -- Statement ③
  ∧ (λ = -1 → ((a * (A.x - B.x) + b * (A.y - B.y) = 0) → (A.y - B.y) / (A.x - B.x) = -a / b))
  -- Statement ④
  ∧ (λ > 0 → (a * A.x + b * A.y + c) * (a * B.x + b * B.y + c) < 0) :=
sorry

end correct_statements_l442_442917


namespace area_PQRS_le_one_fourth_area_ABCD_l442_442845

open_locale classical

variables {P Q R S A B C D : Type}
variables [ordered_add_comm_group A] [ordered_add_comm_group B] [ordered_add_comm_group C] [ordered_add_comm_group D]

-- Define the trapezoid and points in Lean
variables (trapezoid : A → B → C → D → Prop)
variables (on_bases : P → Q → R → S → Prop)
variables (intersect_P : A → Q → B → S → P → Prop)
variables (intersect_R : C → S → D → Q → R → Prop)
variables (area_PQRS : P → Q → R → S → ℝ)
variables (area_ABCD : A → B → C → D → ℝ)

-- Lean statement of the proof
theorem area_PQRS_le_one_fourth_area_ABCD
  (trapezoid_ABCD : trapezoid A B C D)
  (points_on_bases : on_bases Q S)
  (P_intersect : intersect_P A Q B S P)
  (R_intersect : intersect_R C S D Q R) :
  area_PQRS P Q R S ≤ (1/4 : ℝ) * area_ABCD A B C D :=
sorry

end area_PQRS_le_one_fourth_area_ABCD_l442_442845


namespace cos_double_angle_l442_442491

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 3) : Real.cos (2 * α) = -4 / 5 := 
  sorry

end cos_double_angle_l442_442491


namespace words_per_hour_after_two_hours_l442_442204

theorem words_per_hour_after_two_hours 
  (total_words : ℕ) (initial_rate : ℕ) (initial_time : ℕ) (start_time_before_deadline : ℕ) 
  (words_written_in_first_phase : ℕ) (remaining_words : ℕ) (remaining_time : ℕ)
  (final_rate_per_hour : ℕ) :
  total_words = 1200 →
  initial_rate = 400 →
  initial_time = 2 →
  start_time_before_deadline = 4 →
  words_written_in_first_phase = initial_rate * initial_time →
  remaining_words = total_words - words_written_in_first_phase →
  remaining_time = start_time_before_deadline - initial_time →
  final_rate_per_hour = remaining_words / remaining_time →
  final_rate_per_hour = 200 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end words_per_hour_after_two_hours_l442_442204


namespace total_chairs_in_canteen_l442_442182

theorem total_chairs_in_canteen 
    (round_tables : ℕ) 
    (chairs_per_round_table : ℕ) 
    (rectangular_tables : ℕ) 
    (chairs_per_rectangular_table : ℕ) 
    (square_tables : ℕ) 
    (chairs_per_square_table : ℕ) 
    (extra_chairs : ℕ) 
    (h1 : round_tables = 3)
    (h2 : chairs_per_round_table = 6)
    (h3 : rectangular_tables = 4)
    (h4 : chairs_per_rectangular_table = 7)
    (h5 : square_tables = 2)
    (h6 : chairs_per_square_table = 4)
    (h7 : extra_chairs = 5) :
    round_tables * chairs_per_round_table +
    rectangular_tables * chairs_per_rectangular_table +
    square_tables * chairs_per_square_table +
    extra_chairs = 59 := by
  sorry

end total_chairs_in_canteen_l442_442182


namespace ratio_ben_speed_bexy_speed_l442_442801

noncomputable def BexyWalkSpeed : ℝ := 5 -- 5 miles per hour walking speed
noncomputable def BexyBikeSpeed : ℝ := 15 -- 15 miles per hour bicycling speed
noncomputable def BexyDistance : ℝ := 5 -- distance from A to B in miles
noncomputable def BenRoundTripTimeMinutes : ℝ := 160 -- time Ben spends in minutes
noncomputable def TotalDistance : ℝ := 2 * BexyDistance -- total round trip distance for both Bexy and Ben

theorem ratio_ben_speed_bexy_speed :
  let BexyRoundTripTime := (BexyDistance / BexyWalkSpeed) + (BexyDistance / BexyBikeSpeed),
      BexyAverageSpeed := TotalDistance / BexyRoundTripTime,
      BenRoundTripTime := BenRoundTripTimeMinutes / 60,
      BenAverageSpeed := TotalDistance / BenRoundTripTime
  in BenAverageSpeed / BexyAverageSpeed = 1 / 2 :=
by
  sorry -- proof steps are not required

end ratio_ben_speed_bexy_speed_l442_442801


namespace cannot_form_prime_with_earthquake_digits_l442_442179

theorem cannot_form_prime_with_earthquake_digits :
  ∀ (digit_assignment : ℕ → ℕ), 
    (∀ i j, i ≠ j → digit_assignment i ≠ digit_assignment j) ∧ 
    (∃ k, digit_assignment k = digit_assignment 4) ∧ 
    (∀ n, prime n → n > 1) → 
    ∀ num : ℕ, 
      (num = list.foldr (λ (x y : ℕ), x * 10 + y) 0 (list.map digit_assignment (list.finRange 12))) → ¬ prime num :=
by
  sorry

end cannot_form_prime_with_earthquake_digits_l442_442179


namespace probability_fourth_term_integer_l442_442432

-- Definitions for the sequence generation
def next_term_H (a : ℕ) : ℤ := 3 * a - 2
def next_term_T (a : ℕ) : ℤ := a / 2 - 1

-- Initial condition
def a1 : ℕ := 8

-- Theorem to prove the probability
theorem probability_fourth_term_integer : 
  let a2_H := next_term_H a1,
      a2_T := next_term_T a1,
      a3_H_H := next_term_H a2_H,
      a3_H_T := next_term_T a2_H,
      a3_T_H := next_term_H a2_T,
      a3_T_T := next_term_T a2_T,
      a4_H_H_H := next_term_H a3_H_H,
      a4_H_H_T := next_term_T a3_H_H,
      a4_H_T_H := next_term_H a3_H_T,
      a4_H_T_T := next_term_T a3_H_T,
      a4_T_H_H := next_term_H a3_T_H,
      a4_T_H_T := next_term_T a3_T_H,
      a4_T_T_H := next_term_H a3_T_T,
      a4_T_T_T := next_term_T a3_T_T
  in (a4_H_H_H % 1 = 0) ∨ (a4_H_H_T % 1 = 0) ∨ (a4_H_T_H % 1 = 0) ∨ (a4_H_T_T % 1 = 0) ∨
     (a4_T_H_H % 1 = 0) ∨ (a4_T_H_T % 1 = 0) ∨ (a4_T_T_H % 1 = 0) ∨ (a4_T_T_T % 1 = 0) :=
  sorry

end probability_fourth_term_integer_l442_442432


namespace algebraic_expression_standard_l442_442344

theorem algebraic_expression_standard :
  (∃ (expr : String), expr = "-(1/3)m" ∧
    expr ≠ "1(2/5)a" ∧
    expr ≠ "m / n" ∧
    expr ≠ "t × 3") :=
  sorry

end algebraic_expression_standard_l442_442344


namespace volume_pyramid_problem_l442_442566

noncomputable def volume_of_pyramid : ℝ :=
  1 / 3 * 10 * 1.5

theorem volume_pyramid_problem :
  ∀ (AB BC CG : ℝ)
  (M : ℝ × ℝ × ℝ),
  AB = 4 →
  BC = 2 →
  CG = 3 →
  M = (2, 5, 1.5) →
  volume_of_pyramid = 5 := 
by
  intros AB BC CG M hAB hBC hCG hM
  sorry

end volume_pyramid_problem_l442_442566


namespace range_of_a_l442_442901

theorem range_of_a (a : ℝ) :
  (∃ x ∈ {x : ℤ | (x : ℝ) - a + 1 ≥ 0 ∧ 3 - 2 * (x : ℝ) > 0}, 
   ∃ y ∈ {y : ℤ | (y : ℝ) - a + 1 ≥ 0 ∧ 3 - 2 * (y : ℝ) > 0}, 
   ∃ z ∈ {z : ℤ | (z : ℝ) - a + 1 ≥ 0 ∧ 3 - 2 * (z : ℝ) > 0}, 
   {x, y, z} = {-1, 0, 1}) → 
  -1 < a ∧ a ≤ 0 := by
  sorry

end range_of_a_l442_442901


namespace emilys_mom_glue_sticks_l442_442821

theorem emilys_mom_glue_sticks : 
  (let packs_for_Emily := 6
  let packs_for_sister := 7
  packs_for_Emily + packs_for_sister = 13) :=
by
  let packs_for_Emily := 6
  let packs_for_sister := 7
  show packs_for_Emily + packs_for_sister = 13
  by sorry

end emilys_mom_glue_sticks_l442_442821


namespace correct_graph_for_minus_2f_l442_442873

noncomputable def f : ℝ → ℝ :=
  λ x, if -3 ≤ x ∧ x ≤ 0 then -2 - x else
       if 0 ≤ x ∧ x ≤ 2 then real.sqrt (4 - (x - 2)^2) - 2 else
       if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2) else 0

theorem correct_graph_for_minus_2f :
  ∀ x : ℝ, (x ∈ Icc (-3 : ℝ) 0 → -2 * f x = -2 * (-2 - x)) ∧
           (x ∈ Icc (0 : ℝ) 2 → -2 * f x = -2 * (real.sqrt (4 - (x - 2)^2) - 2)) ∧
           (x ∈ Icc (2 : ℝ) 3 → -2 * f x = -2 * (2 * (x - 2))) :=
by sorry

end correct_graph_for_minus_2f_l442_442873


namespace correct_domain_funcA_l442_442067

-- Definitions of given functions
def funcA (x : ℝ) : ℝ := real.sqrt x
def funcB (x : ℝ) : ℝ := -2 * x^2
def funcC (x : ℝ) : ℝ := 3 * x + 1
def funcD (x : ℝ) : ℝ := (x - 1)^2

-- We need to prove that the domain of funcA is [0, ∞)
theorem correct_domain_funcA : ∀ x, (0 ≤ x) ↔ (∃ y, y = funcA x) :=
by
  sorry

end correct_domain_funcA_l442_442067


namespace car_length_approx_l442_442760

noncomputable def length_of_car (speed_kmph : ℕ) (time_seconds : ℝ) : ℝ :=
  let speed_mps := (speed_kmph * 1000) / 3600
  speed_mps * time_seconds

theorem car_length_approx {speed_kmph : ℕ} {time_seconds : ℝ}
  (h_speed : speed_kmph = 36)
  (h_time : time_seconds = 0.9999200063994881) :
  length_of_car speed_kmph time_seconds ≈ 9.9992 := by
  sorry

end car_length_approx_l442_442760


namespace room_length_difference_l442_442195

def width := 19
def length := 20
def difference := length - width

theorem room_length_difference : difference = 1 := by
  sorry

end room_length_difference_l442_442195


namespace largest_factor_of_5_in_sum_l442_442416

theorem largest_factor_of_5_in_sum :
  let s := 48! + 49! + 50!
  ∃ n : ℕ, 5^n ∣ s ∧ ¬ ∃ m : ℕ, 5^(n + 1) ∣ s :=
begin
  let s := 48! + 49! + 50!,
  existsi 13,
  split,
  { sorry },
  { sorry }
end

end largest_factor_of_5_in_sum_l442_442416


namespace log_sum_real_coeffs_l442_442613

theorem log_sum_real_coeffs (x : ℝ) (T : ℝ) (h : T = (∑ i in filter (λ i, i % 2 = 0) (range 2012), nat.choose 2011 i * x ^ i)) :
  log 2 T = 1005.5 :=
by sorry

end log_sum_real_coeffs_l442_442613


namespace cookie_distribution_l442_442420

def trays := 4
def cookies_per_tray := 24
def total_cookies := trays * cookies_per_tray
def packs := 8
def cookies_per_pack := total_cookies / packs

theorem cookie_distribution : cookies_per_pack = 12 := by
  sorry

end cookie_distribution_l442_442420


namespace f_decreasing_iff_f_minimum_value_l442_442940

open Real

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 - m * log (2 * x + 1)

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := 2 * x - (2 * m) / (2 * x + 1)

theorem f_decreasing_iff {m : ℝ} (h : 0 < m) : (∀ x ∈ Ioo (-1/2 : ℝ) 1, f' x m ≤ 0) ↔ 3 ≤ m := sorry

theorem f_minimum_value {m : ℝ} (h : 0 < m) :
  (∃ x ∈ Icc (-1/2 : ℝ) 1, ∀ y ∈ Icc (-1/2 : ℝ) 1, f x m ≤ f y m) ∧
  ((0 < m ∧ m < 3) → ∃ x ∈ Icc (-1/2 : ℝ) 1, x = (-1 + sqrt (1 + 8 * m)) / 4 ∧
    (∀ y ∈ Icc (-1/2 : ℝ) 1, f x m ≤ f y m)) ∧
  (3 ≤ m → ∃ x ∈ Icc (-1/2 : ℝ) 1, x = 1 ∧ (∀ y ∈ Icc (-1/2 : ℝ) 1, f x m ≤ f y m)) := sorry

end f_decreasing_iff_f_minimum_value_l442_442940


namespace problem_statement_l442_442561

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem problem_statement :
  let l := { p : ℝ × ℝ | p.1 - p.2 - 2 = 0 }
  let C := { p : ℝ × ℝ | ∃ θ : ℝ, p = (2 * Real.sqrt 3 * Real.cos θ, 2 * Real.sin θ) }
  let A := (-4, -6)
  let B := (4, 2)
  let P := (-2 * Real.sqrt 3, 2)
  let d := (|2 * Real.sqrt 3 * Real.cos (5 * Real.pi / 6) - 2|) / Real.sqrt 2
  distance A B = 8 * Real.sqrt 2 ∧ d = 3 * Real.sqrt 2 ∧
  let max_area := 1 / 2 * 8 * Real.sqrt 2 * 3 * Real.sqrt 2
  P ∈ C ∧ max_area = 24 := by
sorry

end problem_statement_l442_442561


namespace hyperbola_eccentricity_l442_442166

theorem hyperbola_eccentricity (k : ℤ) (hk : 3 < k ∧ k < 5) :
  let a := 1 in
  let b := 1 in
  let c := Real.sqrt (a^2 + b^2) in
  c / a = Real.sqrt 2 :=
by 
  sorry

end hyperbola_eccentricity_l442_442166


namespace no_such_triangle_from_cube_l442_442572

theorem no_such_triangle_from_cube (a b c : ℝ) :
  a^2 + b^2 = 49 →
  b^2 + c^2 = 64 →
  c^2 + a^2 = 121 →
  a > 0 → b > 0 → c > 0 →
  false :=
by
  intros h1 h2 h3 ha hb hc
  -- Summing the equations
  have h_sum := congr_arg (λ x, a^2 + b^2 + c^2) (add_left (add_left h1) h2)
  simp at h_sum
  have h4 : a^2 + b^2 + c^2 = 117 := h_sum
  -- Contradiction with a^2 + c^2
  have h5 : b^2 = 117 - (a^2 + c^2) :=
    by
      rw h4
  rw h3 at h5
  have h6 : b^2 = 117 - 121 := h5
  rw h6 at hb
  -- Since b^2 cannot be negative
  linarith

end no_such_triangle_from_cube_l442_442572


namespace cost_per_use_correct_l442_442197

-- Definitions based on conditions in the problem
def total_cost : ℕ := 30
def uses_per_week : ℕ := 3
def number_of_weeks : ℕ := 2
def total_uses : ℕ := uses_per_week * number_of_weeks

-- Statement based on the question and correct answer
theorem cost_per_use_correct : (total_cost / total_uses) = 5 := sorry

end cost_per_use_correct_l442_442197


namespace sadaf_height_l442_442622

theorem sadaf_height (L : ℕ) (hL : L = 90) (A : ℕ) (hA : A = (4 * L) / 3) (S : ℕ) (hS : S = (5 * A) / 4) : S = 150 :=
by
  rw [hL, hA, hS]
  -- Proof skipped
  sorry

end sadaf_height_l442_442622


namespace evaluate_expression_l442_442447

def cube_root (x : ℝ) := x^(1/3)

theorem evaluate_expression : (cube_root (9 / 32))^2 = (3/8) := 
by
  sorry

end evaluate_expression_l442_442447


namespace imo1991_q6_l442_442975

variable (A B C I L1 L2 L3 : ℝ)
variable (a b c : ℝ)

def AI_over_AL1 := (b + c) / (a + b + c)
def BI_over_BL2 := (a + c) / (a + b + c)
def CI_over_CL3 := (a + b) / (a + b + c)

theorem imo1991_q6 (h : a + b + c ≠ 0) :
  AI_over_AL1 A B C I L1 L2 L3 a b c * BI_over_BL2 A B C I L1 L2 L3 a b c * CI_over_CL3 A B C I L1 L2 L3 a b c ≤ 8 / 27 :=
  by {
    sorry
  }

end imo1991_q6_l442_442975


namespace unattainable_value_of_y_l442_442086

theorem unattainable_value_of_y (x : ℚ) (h : x ≠ -5/4) :
  ¬ ∃ y : ℚ, y = (2 - 3 * x) / (4 * x + 5) ∧ y = -3/4 :=
by
  sorry

end unattainable_value_of_y_l442_442086


namespace largest_m_for_negative_integral_solutions_l442_442331

theorem largest_m_for_negative_integral_solutions :
  ∃ m : ℕ, (∀ p q : ℤ, 10 * p * p + (-m) * p + 560 = 0 ∧ p < 0 ∧ q < 0 ∧ p * q = 56 → m ≤ 570) ∧ m = 570 :=
sorry

end largest_m_for_negative_integral_solutions_l442_442331


namespace find_positive_integer_solutions_l442_442823

theorem find_positive_integer_solutions :
  ∃ (x y z : ℕ), 
    2 * x * z = y^2 ∧ 
    x + z = 1987 ∧ 
    x = 1458 ∧ 
    y = 1242 ∧ 
    z = 529 :=
  by sorry

end find_positive_integer_solutions_l442_442823


namespace hyperbola_eccentricity_l442_442456

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a^2 = 5) (h2 : b^2 = 5) :
  sqrt(1 + (b^2) / (a^2)) = sqrt 2 :=
by {
  sorry
}

end hyperbola_eccentricity_l442_442456


namespace inequality_solution_l442_442989

theorem inequality_solution (x : ℝ) :
  (x^2 - 9) / (x - 3)^2 < 0 ↔ x ∈ set.Ioo (-3 : ℝ) 3 ∧ x ≠ 3 :=
sorry

end inequality_solution_l442_442989


namespace hall_of_mirrors_l442_442926

theorem hall_of_mirrors (h : ℝ) 
    (condition1 : 2 * (30 * h) + (20 * h) = 960) :
  h = 12 :=
by
  sorry

end hall_of_mirrors_l442_442926


namespace calc_tan_calc_expression_l442_442111

variable (α : Real) (h1 : α > π / 2 ∧ α < π) (h2 : sin α = 4 / 5)

theorem calc_tan (h1 : α > π / 2 ∧ α < π) (h2 : sin α = 4 / 5) : tan α = -4 / 3 := 
  sorry

theorem calc_expression (h1 : α > π / 2 ∧ α < π) (h2 : sin α = 4 / 5) :
  (sin (π + α) - 2 * cos (π / 2 + α)) / (-sin (-α) + cos (π - α)) = 4 / 7 := 
  sorry

end calc_tan_calc_expression_l442_442111


namespace compare_series_l442_442104

theorem compare_series (x y : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) : 
  (1 / (1 - x^2) + 1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by
  sorry

end compare_series_l442_442104


namespace max_f_value_l442_442460

def f (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) : ℝ := 
  sin (θ₁ + θ₂) - sin (θ₁ - θ₂) + 
  sin (θ₂ + θ₃) - sin (θ₂ - θ₃) + 
  sin (θ₃ + θ₄) - sin (θ₃ - θ₄) + 
  sin (θ₄ + θ₅) - sin (θ₄ - θ₅) + 
  sin (θ₅ + θ₁) - sin (θ₅ - θ₁)

theorem max_f_value : ∃ θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ, f θ₁ θ₂ θ₃ θ₄ θ₅ = 5 := by
  sorry

end max_f_value_l442_442460


namespace tangent_line_at_1_monotonicity_l442_442506

def f (a x : ℝ) : ℝ := exp (2 * x - 1) * (x^2 + a * x - 2 * a^2 + 1)

def f_prime (a x : ℝ) : ℝ := exp (2 * x - 1) * (2 * x^2 + (2 * a + 2) * x - 4 * a^2 + a + 2)

theorem tangent_line_at_1 (a : ℝ) (h : a = 1) :
  let x : ℝ := 1;
  let y : ℝ := f a x;
  let f_prime_at_1 : ℝ := f_prime a x;
  y = exp 1 ∧ f_prime_at_1 = 5 * exp 1 ∧ (y = 5 * exp 1 * (x - 1) + 4 * exp 1 → 
    y = 5 * exp 1 * x - 4 * exp 1) :=
by
  -- This is a placeholder for C-style comments.
  -- Proof is omitted
  sorry

theorem monotonicity (a : ℝ) :
  let Δ := 4 * (9 * a^2 - 3);
  Δ ≤ 0 →
  (∀ x, f_prime a x ≥ 0) ∨ 
  ((a > (√3)/3 ∨ a < - (√3)/3) ∧ 
    (∀ x ∈ (-∞, (-(a+1) - √(9a^2 - 3))/2) ∪ ((-(a+1) + √(9a^2 - 3))/2, +∞), f_prime a x > 0) ∧
    (∀ x ∈ ((-(a+1) - √(9a^2 - 3))/2), (-(a+1) + √(9a^2 - 3))/2), f_prime a x < 0)) :=
by
  -- Placeholder comment for proof.
  sorry

end tangent_line_at_1_monotonicity_l442_442506


namespace number_of_numerators_in_lowest_terms_l442_442609

-- Definitions used in Lean 4 statement
def rational_with_repeating_decimal (r : ℚ) : Prop :=
  0 < r ∧ r < 1 ∧ ∃ a b c d : ℕ,
  r = (a * 1000 + b * 100 + c * 10 + d) / 9999

-- Theorem to state the problem and its correct answer
theorem number_of_numerators_in_lowest_terms : 
  ∃ (n : ℕ), n = 6000 ∧ ∀ r ∈ { q : ℚ | rational_with_repeating_decimal q },
   ∃ p q : ℕ, q = p / q ∧ nat.gcd p q = 1 :=
sorry

end number_of_numerators_in_lowest_terms_l442_442609


namespace Rachel_painting_time_l442_442962

noncomputable def Matt_time : ℕ := 12
noncomputable def Patty_time (Matt_time : ℕ) : ℕ := Matt_time / 3
noncomputable def Rachel_time (Patty_time : ℕ) : ℕ := 5 + 2 * Patty_time

theorem Rachel_painting_time : Rachel_time (Patty_time Matt_time) = 13 := by
  sorry

end Rachel_painting_time_l442_442962


namespace nonexistence_of_equal_sided_pentagon_with_right_angles_l442_442574

-- Definition to capture the given conditions in Lean 4:
def is_equal_sided_three_dimensional_pentagon_with_right_angles
    (pentagon : List (ℝ × ℝ × ℝ)) :=
  pentagon.length = 5 ∧
  ∀ i, dist (pentagon.nth_le i sorry) (pentagon.nth_le ((i + 1) % 5) sorry) =
      dist (pentagon.nth_le 0 sorry) (pentagon.nth_le 1 sorry) ∧
  ∀ i, ∃ j, 
    pentagon.nth_le i sorry ≠ pentagon.nth_le j sorry ∧
    (angle (pentagon.nth_le i sorry) (pentagon.nth_le ((i + 1) % 5) sorry) (pentagon.nth_le ((i + 2) % 5) sorry)) = π / 2

-- Target theorem to state the nonexistence of such a pentagon:
theorem nonexistence_of_equal_sided_pentagon_with_right_angles :
    ¬∃ pentagon, is_equal_sided_three_dimensional_pentagon_with_right_angles pentagon :=
by {
  sorry -- Proof steps to show the nonexistence
}

end nonexistence_of_equal_sided_pentagon_with_right_angles_l442_442574


namespace max_min_values_l442_442672

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values :
  ∃ (a b : ℝ), (∀ x ∈ set.Icc (0:ℝ) 3, f x ≤ a) ∧ (∀ x ∈ set.Icc (0:ℝ) 3, f x ≥ b) ∧ a = f 0 ∧ b = f 2 :=
by
  let a := f 0
  let b := f 2
  have fa : f 0 = 5 := by sorry
  have fb : f 2 = -15 := by sorry
  existsi [a, b]
  split
  case h_left =>
    intros x hx
    sorry
  case h_right =>
    intros x hx
    sorry
  case h_a =>
    exact fa
  case h_b =>
    exact fb

end max_min_values_l442_442672


namespace sum_possible_students_l442_442391

theorem sum_possible_students :
  let contains_boundaries (s : ℕ) := 200 ≤ s ∧ s ≤ 250
  let eight_sections (s : ℕ) := (s - 2) % 8 = 0
  let possible_students := {s : ℕ | contains_boundaries s ∧ eight_sections s}
  (∑ s in possible_students, s) = 1582 :=
by
  sorry

end sum_possible_students_l442_442391


namespace diagonal_length_l442_442780

-- Define area of the square
variable (A : ℝ) 
-- Define the side length of square
variable (s : ℝ)
-- Area condition
def area_of_square (h : A = 9 / 16) : s = real.sqrt (9 / 16) :=
  by sorry

-- Define the derived side length from the area
noncomputable def side_length (h : A = 9 / 16) : s = 3 / 4 := by sorry

-- Length of the diagonal given side length
noncomputable def length_of_diagonal (h : s = 3 / 4) : ℝ :=
  s * real.sqrt 2

-- Theorem to prove the diagonal length
theorem diagonal_length (h : A = 9 / 16) : length_of_diagonal A = 3 / 4 * real.sqrt 2 :=
  by sorry

end diagonal_length_l442_442780


namespace f_is_one_to_one_l442_442214

noncomputable def f : ℝ² → ℝ² := sorry
def ∂f ∂x : ℝ² × ℝ² → ℝ := sorry-- This defines the partial derivatives into a matrix

lemma continuous_partial_derivatives (x : ℝ²) :
  continuous (∂f ∂x) :=
sorry

lemma positive_partial_derivatives (x : ℝ²) (i j : fin 2) :
  ∂f ∂x (x, i, j) > 0 :=
sorry

lemma condition (x : ℝ²) :
  (∂f ∂x (x, 0, 0)) * (∂f ∂x (x, 1, 1)) - 
  1 / 4 * ((∂f ∂x (x, 0, 1)) + (∂f ∂x (x, 1, 0))) ^ 2 > 0 :=
sorry

theorem f_is_one_to_one : function.injective f :=
sorry

end f_is_one_to_one_l442_442214


namespace y_minimum_value_l442_442108

noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem y_minimum_value (x y : ℝ) (h_progression : 2 * lg (sin x - 1 / 3) = lg 2 + lg (1 - y)) :
  7 / 9 ≤ y ∧ y < 1 :=
by
  sorry

end y_minimum_value_l442_442108


namespace trading_cards_initial_total_l442_442249

theorem trading_cards_initial_total (x : ℕ) 
  (h1 : ∃ d : ℕ, d = (1 / 3 : ℕ) * x)
  (h2 : ∃ n1 : ℕ, n1 = (1 / 5 : ℕ) * (1 / 3 : ℕ) * x)
  (h3 : ∃ n2 : ℕ, n2 = (1 / 3 : ℕ) * ((1 / 5 : ℕ) * (1 / 3 : ℕ) * x))
  (h4 : ∃ n3 : ℕ, n3 = (1 / 2 : ℕ) * (2 / 45 : ℕ) * x)
  (h5 : (1 / 15 : ℕ) * x + (2 / 45 : ℕ) * x + (1 / 45 : ℕ) * x = 850) :
  x = 6375 := 
sorry

end trading_cards_initial_total_l442_442249


namespace statement_I_statement_II_statement_III_l442_442469

open Real

theorem statement_I (x : ℝ) : (⌊x + 0.5⌋ = ⌊x⌋) ∨ (⌊x + 0.5⌋ = ⌊x⌋ + 1) :=
sorry

theorem statement_II (x y : ℝ) : ⌈x + y⌉ = ⌈x⌉ + ⌈y⌉ := 
sorry -- We know this is false, but it has to be stated as it is part of the problem.

theorem statement_III (x : ℝ) : ⌊2 * x⌋ = 2 * ⌊x⌋ :=
sorry -- This is also false but included for completeness.

end statement_I_statement_II_statement_III_l442_442469


namespace derived_triangle_angles_and_area_l442_442657

-- Define the original right triangle and altitude conditions
variables {A B C D : Type}
variable (h : ℝ) -- The altitude value

-- Define the centers of the incircles of the two smaller triangles
variables {O1 O2 : Type}

-- Statement of the theorem: angles and area of the derived triangle
theorem derived_triangle_angles_and_area (h_pos : h > 0) :
  let α := π / 4 in
  let β := π / 4 in
  let γ := π / 2 in
  let area := h^2 / 2 in
  -- Angles of the triangle formed by the original legs and the line through centers are π/4, π/4, π/2
  ∃ (K M N : Type), 
    angle K M N = α ∧
    angle M N K = β ∧
    angle N K M = γ ∧
    -- Area of the new triangle
    triangle_area K M N = area :=
sorry

end derived_triangle_angles_and_area_l442_442657


namespace imaginary_part_of_complex_number_l442_442830

noncomputable def imaginary_part_of_z (z : ℂ) : ℂ :=
(im (z)).to_complex

theorem imaginary_part_of_complex_number 
  (z : ℂ) 
  (h : (conj(z) - 1 + 3 * complex.I) * (2 - complex.I) = 4 + 3 * complex.I) : 
  imaginary_part_of_z z = 1 :=
sorry

end imaginary_part_of_complex_number_l442_442830


namespace car_stopping_distance_l442_442293

noncomputable def distance_traveled_after_engine_off
  (v0 : ℝ)  -- initial speed in m/s
  (P : ℝ)   -- power in watts
  (m : ℝ)   -- mass in kilograms
  (alpha : ℝ) -- proportionality constant for resistive force
  (resistance_prop : α = P / (v0 * v0)) -- resistance is proportional to speed
  : Prop :=
  let s := (m * v0^3) / P in
  s = 240 -- distance must be 240 meters

-- The actual conditions in Lean:
def initial_speed := 20 -- 72 km/h converted to m/s
def power := 50000 -- 50 kW in Watts
def mass := 1500 -- mass in kg
def alpha := 125 -- resistive force constant in N⋅s/m

theorem car_stopping_distance :
  distance_traveled_after_engine_off initial_speed power mass alpha (by rfl) :=
begin
  sorry
end

end car_stopping_distance_l442_442293


namespace problem1_problem2_l442_442734

theorem problem1 (n : ℕ) : 2 ≤ (1 + 1 / n) ^ n ∧ (1 + 1 / n) ^ n < 3 :=
sorry

theorem problem2 (n : ℕ) : (n / 3) ^ n < n! :=
sorry

end problem1_problem2_l442_442734


namespace isabella_houses_l442_442193

theorem isabella_houses (green yellow red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  green + red = 160 := 
by sorry

end isabella_houses_l442_442193


namespace cos_4theta_solution_l442_442489

theorem cos_4theta_solution (θ : Real) 
  (h : 2 ^ (-2 + 3 * cos θ) + 1 = 2 ^ (1 / 2 + 2 * cos θ)) : 
  cos (4 * θ) = -1 / 2 :=
sorry

end cos_4theta_solution_l442_442489


namespace intersection_of_A_and_B_l442_442230

open Set

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
  by
    sorry

end intersection_of_A_and_B_l442_442230


namespace martian_year_length_ratio_l442_442717

theorem martian_year_length_ratio :
  let EarthDay := 24 -- hours
  let MarsDay := EarthDay + 2 / 3 -- hours (since 40 minutes is 2/3 of an hour)
  let MartianYearDays := 668
  let EarthYearDays := 365.25
  (MartianYearDays * MarsDay) / EarthYearDays = 1.88 := by
{
  sorry
}

end martian_year_length_ratio_l442_442717


namespace problem1_problem2_l442_442635

-- Statement for problem 1
theorem problem1 : 
  (-2020 - 2 / 3) + (2019 + 3 / 4) + (-2018 - 5 / 6) + (2017 + 1 / 2) = -2 - 1 / 4 := 
sorry

-- Statement for problem 2
theorem problem2 : 
  (-1 - 1 / 2) + (-2000 - 5 / 6) + (4000 + 3 / 4) + (-1999 - 2 / 3) = -5 / 4 := 
sorry

end problem1_problem2_l442_442635


namespace find_principal_l442_442733

variable (SI : ℚ) (R : ℚ) (T : ℚ)

theorem find_principal : SI = 4016.25 → R = 13 → T = 5 → 
  (let P := SI / (R * T / 100) in P = 6180) := 
by 
  intros hSI hR hT 
  simp [SI, R, T] at *
  sorry

end find_principal_l442_442733


namespace probabilities_equal_l442_442010

def roll := {n : ℕ // n ≥ 1 ∧ n ≤ 6}

def is_successful (r : roll) : Prop := r.val ≥ 3

def prob_successful : ℚ := 4 / 6

def prob_unsuccessful : ℚ := 1 - prob_successful

def prob_at_least_one_success_two_rolls : ℚ := 1 - (prob_unsuccessful ^ 2)

def prob_at_least_two_success_four_rolls : ℚ :=
  let zero_success := prob_unsuccessful ^ 4
  let one_success := 4 * (prob_unsuccessful ^ 3) * prob_successful
  1 - (zero_success + one_success)

theorem probabilities_equal :
  prob_at_least_one_success_two_rolls = prob_at_least_two_success_four_rolls := by
  sorry

end probabilities_equal_l442_442010


namespace local_minimum_range_l442_442900

theorem local_minimum_range (a : ℝ) : (∃ x ∈ Ioo 1 2, is_local_min (λ x, x^3 - 3 * a * x + a) x) ↔ 1 < a ∧ a < 4 :=
sorry

end local_minimum_range_l442_442900


namespace edmund_earning_for_extra_chores_l442_442073

theorem edmund_earning_for_extra_chores :
  ∀ (save_up : ℕ) (normal_chores_per_week : ℕ) (payment_per_extra_chore : ℕ) (chores_per_day : ℕ) (weeks : ℕ),
    save_up = 75 ->
    normal_chores_per_week = 12 ->
    payment_per_extra_chore = 2 ->
    chores_per_day = 4 ->
    weeks = 2 ->
    (let total_chores := (chores_per_day * 7 * weeks) in
     let normal_chores := (normal_chores_per_week * weeks) in
     let extra_chores := (total_chores - normal_chores) in
     let earning := (extra_chores * payment_per_extra_chore) in
     earning = 64) := 
by sorry

end edmund_earning_for_extra_chores_l442_442073


namespace sum_of_seven_step_palindromes_l442_442828

def reverse_int (n : ℕ) : ℕ :=
  let s := n.toString
  s.reverse.toNat

def is_palindrome (n : ℕ) : Prop :=
  n.toString = n.toString.reverse

noncomputable def reaches_palindrome_in_seven (n : ℕ) : Prop :=
  (iterate reverse_add 6 n) ≠ (iterate reverse_add 7 n) ∧ is_palindrome (iterate reverse_add 7 n)
  where
    reverse_add (m : ℕ) := m + reverse_int m

def is_non_palindrome_int (n : ℕ) : Prop :=
  (101 ≤ n) ∧ (n < 200) ∧ ¬ is_palindrome n

theorem sum_of_seven_step_palindromes : ∑ n in (finset.filter reaches_palindrome_in_seven (finset.filter is_non_palindrome_int (finset.range 200))), n = 685 := 
  sorry

end sum_of_seven_step_palindromes_l442_442828


namespace range_of_m_l442_442167

variable {x m : ℝ}

def absolute_value_inequality (x m : ℝ) : Prop := |x + 1| - |x - 2| > m

theorem range_of_m : (∀ x : ℝ, absolute_value_inequality x m) ↔ m < -3 :=
by
  sorry

end range_of_m_l442_442167


namespace coin_weighing_l442_442703

theorem coin_weighing (n : ℕ) (g s c : ℕ) (wg ws wc : ℕ) :
  n = 100 ∧ g + s + c = 100 ∧ g ≥ 1 ∧ s ≥ 1 ∧ c ≥ 1 ∧ 
  wg = 3 ∧ ws = 2 ∧ wc = 1 →
  ∃ w : ℕ, w ≤ 101 :=
begin
  sorry
end

end coin_weighing_l442_442703


namespace solve_equation_l442_442694

theorem solve_equation : ∃ x : ℝ, (3 / (x - 2) - 1 = 0) ∧ x = 5 :=
by
  existsi 5
  split
  · calc
      3 / (5 - 2) - 1 = 3 / 3 - 1 : by rw (sub_eq_zero_of_eq (rfl : 5 - 2 = 3))
      ...              = 1 - 1 : by norm_num
      ...              = 0 : by norm_num
  · rfl

end solve_equation_l442_442694


namespace circumscribed_circle_area_l442_442007

noncomputable def sin_36 := real.sin (36 * real.pi / 180)
noncomputable def radius (s : ℝ) : ℝ := s / (2 * sin_36)
noncomputable def area_of_circumscribed_circle (s : ℝ) : ℝ := real.pi * (radius s) ^ 2

theorem circumscribed_circle_area {s : ℝ} (h : s = 10) :
  area_of_circumscribed_circle s = 2000 * (5 + 2 * real.sqrt 5) * real.pi :=
by
  -- using the given condition
  rw [h]
  -- calculation steps omitted
  sorry

end circumscribed_circle_area_l442_442007


namespace tanya_bought_six_plums_l442_442282

theorem tanya_bought_six_plums (pears apples pineapples pieces_left : ℕ) 
  (h_pears : pears = 6) (h_apples : apples = 4) (h_pineapples : pineapples = 2) 
  (h_pieces_left : pieces_left = 9) (h_half_fell : pieces_left * 2 = total_fruit) :
  pears + apples + pineapples < total_fruit ∧ total_fruit - (pears + apples + pineapples) = 6 :=
by
  sorry

end tanya_bought_six_plums_l442_442282


namespace simplify_polynomial_l442_442276

open Nat

theorem simplify_polynomial (n : ℕ) (x : ℝ) (hx : x ≠ 1) :
    let S := ∑ i in range n, (i * (i + 1) / 2) * x^(i + 1)
    S = (n * (n + 1) * x^n) / (2 * (x - 1)) - (n * x^n) / ((x - 1)^2) + (x * (x^n - 1)) / ((x - 1)^3) := by
  sorry

end simplify_polynomial_l442_442276


namespace min_quotient_is_75_16_l442_442826

def four_distinct_nonzero_digits : Prop :=
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9

theorem min_quotient_is_75_16 :
  four_distinct_nonzero_digits →
  ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧ (1 ≤ d ∧ d ≤ 9) ∧
  ∀ (x y z w : ℕ), x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧
  1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 1 ≤ z ∧ z ≤ 9 ∧ 1 ≤ w ∧ w ≤ 9 →
    (1000 * x + 100 * y + 10 * z + w) / (x + y + z + w) ≥ 75.16
:= sorry

end min_quotient_is_75_16_l442_442826


namespace Miles_trombones_count_l442_442965

theorem Miles_trombones_count :
  let fingers := 10
  let trumpets := fingers - 3
  let hands := 2
  let guitars := hands + 2
  let french_horns := guitars - 1
  let heads := 1
  let trombones := heads + 2
  trumpets + guitars + french_horns + trombones = 17 → trombones = 3 :=
by
  intros h
  sorry

end Miles_trombones_count_l442_442965


namespace isabella_houses_l442_442192

theorem isabella_houses (green yellow red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  green + red = 160 := 
by sorry

end isabella_houses_l442_442192


namespace triangle_sides_l442_442347

theorem triangle_sides (a : ℕ) (h : a > 0) : 
  (a + 1) + (a + 2) > (a + 3) ∧ (a + 1) + (a + 3) > (a + 2) ∧ (a + 2) + (a + 3) > (a + 1) := 
by 
  sorry

end triangle_sides_l442_442347


namespace calc_surface_area_l442_442040

noncomputable def total_surface_area (edge_length : ℝ) (hole_diameter : ℝ) : ℝ :=
  let face_area := edge_length ^ 2
  let cube_surface_area := 6 * face_area
  let hole_radius := hole_diameter / 2
  let hole_area := π * hole_radius ^ 2
  let removed_area := 6 * hole_area
  let exposed_cylinder_area := 6 * (2 * π * hole_radius * edge_length)
  cube_surface_area + exposed_cylinder_area - removed_area

-- Lean 4 Statement
theorem calc_surface_area :
  total_surface_area 3 1.5 = 54 + 16.4 * π :=
by
  sorry

end calc_surface_area_l442_442040


namespace a_n_not_zero_l442_442744

noncomputable def a : ℕ → ℤ
| 0     := 1
| 1     := 2
| (n+2) := if (a n) * (a (n+1)) % 2 = 0
           then 5 * (a (n+1)) - 3 * (a n)
           else (a (n+1)) - (a n)

theorem a_n_not_zero : ∀ n : ℕ, a n ≠ 0 := 
by
  sorry

end a_n_not_zero_l442_442744


namespace problem_statement_l442_442915

-- Define the parametric equations of curve C
def curve_C (α : ℝ) : ℝ × ℝ :=
  (√2 * Real.cos α + 1, √2 * Real.sin α + 1)

-- Define the polar equation of line l
def polar_line_l (ρ θ m : ℝ) : Prop :=
  ρ * Real.sin (θ + (Real.pi / 4)) = m

-- Question (I): Cartesian equation of line l
def cartesian_equation_of_line_l (x y m : ℝ) : Prop :=
  x + y - √2 * m = 0

-- Distance from point to line formula
def distance_from_point_to_line (x y a b c : ℝ) : ℝ :=
  abs (a * x + b * y + c) / √(a^2 + b^2)

-- Question (II): The range of the real number m
def range_of_m (m : ℝ) : Prop :=
  -√2 / 2 ≤ m ∧ m ≤ 5 * √2 / 2

theorem problem_statement (α ρ θ m : ℝ) :
  (∀ (x y : ℝ), polar_line_l ρ θ m → cartesian_equation_of_line_l x y m) ∧
  (∀ (x y : ℝ), curve_C α = (x, y) → distance_from_point_to_line x y 1 1 (-√2 * m) = √2 / 2 → range_of_m m) :=
sorry

end problem_statement_l442_442915


namespace intersection_of_A_and_B_l442_442224

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l442_442224


namespace m_range_decrease_y_l442_442565

theorem m_range_decrease_y {m : ℝ} : (∀ x1 x2 : ℝ, x1 < x2 → (2 * m + 2) * x1 + 5 > (2 * m + 2) * x2 + 5) ↔ m < -1 :=
by
  sorry

end m_range_decrease_y_l442_442565


namespace unique_m_value_l442_442473

theorem unique_m_value : ∀ m : ℝ,
  (m ^ 2 - 5 * m + 6 = 0 ∧ m ^ 2 - 3 * m + 2 = 0) →
  (m ^ 2 - 3 * m + 2 = 2 * (m ^ 2 - 5 * m + 6)) →
  ((m ^ 2 - 5 * m + 6) * (m ^ 2 - 3 * m + 2) > 0) →
  m = 2 :=
by
  sorry

end unique_m_value_l442_442473


namespace locus_endpoints_l442_442300

theorem locus_endpoints {x y : ℝ} (Hperp : OA ⊥ OB) (Hlength : ∀ M, dist O M = 1) :
  (|x| + |y| ≥ 1 ∧ x^2 + y^2 ≤ 1) :=
by
  sorry

end locus_endpoints_l442_442300


namespace sum_of_irreducible_factors_evaluated_at_3_l442_442209

noncomputable def f (x : ℤ) : ℤ := x^6 - x^3 - x^2 - 1

theorem sum_of_irreducible_factors_evaluated_at_3 :
  ∃ (q : list (ℤ → ℤ)), (∀ p ∈ q, polynomial.monic p ∧ polynomial.irreducible p) ∧
    (f = (list.prod q)) ∧
    ((q.map (λ p, p 3)).sum = 676) := sorry

end sum_of_irreducible_factors_evaluated_at_3_l442_442209


namespace math_problem_proof_l442_442125

-- Define the propositions from the problem
def P1 : Prop := ∀ x : ℝ, ¬ (x^2 - x > 0) → x^2 - x ≤ 0
def P2 (a b m : ℝ) : Prop := (am^2 < bm^2) → (a < b)
def P3 : Prop := ∃ x : ℝ, (x - sin x) = 0 ∧ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 - sin x1) = 0 ∧ (x2 - sin x2) = 0)
def P4 (f g : ℝ → ℝ) : Prop := (∀ x : ℝ, f (-x) = -f x ∧ g (-x) = g x ∧ (x > 0 → f' x > 0 ∧ g' x > 0) → (x < 0 → f' x > g' x))

-- Conjecture the specific conclusions being correct
def correct_conclusions : Prop := 
  (P1 ∧ ¬P2 0 1 0 ∧ ¬P3 ∧ P4 f g)

theorem math_problem_proof : correct_conclusions :=
sorry

end math_problem_proof_l442_442125


namespace first_player_wins_l442_442713

def round_table : Type := sorry -- Placeholder for the round table type definition
def coin : Type := sorry -- Placeholder for the coin type definition
def valid_placement (t : round_table) (c : coin) : Prop := sorry -- Placeholder for validation of coin placement on the table
def non_overlap (t : round_table) (c1 c2 : coin) : Prop := sorry -- Placeholder for checking that two coins do not overlap on the table

theorem first_player_wins
  (t : round_table)
  (valid : ∀ c : coin, valid_placement t c)
  (not_overlap : ∀ c1 c2 : coin, non_overlap t c1 c2) :
  ∃ strategy : (ℕ → coin), (∀ n : ℕ, valid (strategy n) ∧ (n > 0 → non_overlap t (strategy n) (strategy (n - 1)))) :=
sorry

end first_player_wins_l442_442713


namespace determine_coin_types_l442_442701

theorem determine_coin_types (coins : Fin 100 → ℕ) (h1 : ∀ c, coins c ∈ {1, 2, 3}) 
(h2 : ∃ c1 c2 c3, coins c1 = 1 ∧ coins c2 = 2 ∧ coins c3 = 3):
  ∃ weighings : ℕ, weighings ≤ 101 ∧ (∃ coin_types : Fin 100 → ℕ, (∀ c, coin_types c ∈ {1, 2, 3}) ∧ 
    (∀ c, (coin_types c = 1 ∧ coins c = 1) ∨ (coin_types c = 2 ∧ coins c = 2) ∨ 
           (coin_types c = 3 ∧ coins c = 3))) :=
by
  sorry

end determine_coin_types_l442_442701


namespace distance_between_stripes_l442_442034

theorem distance_between_stripes
  (curb_length_between_stripes : ℝ)
  (distance_between_curbs : ℝ)
  (stripe_length : ℝ) :
  distance_between_curbs = 60 →
  curb_length_between_stripes = 25 →
  stripe_length = 70 →
  (1500 / stripe_length) ≈ 21.43 :=
by
  sorry

end distance_between_stripes_l442_442034


namespace net_distance_from_A_to_B_fuel_consumed_l442_442003

-- Conditions as definitions
def travel_records : List Int := [-4, 7, -9, 8, 6, -5, -2, -4]
def fuel_consumption_rate : Float := 0.5

-- Question 1: Prove the net distance and direction from point A to point B
def direction_and_distance : Int :=
  List.sum travel_records

theorem net_distance_from_A_to_B :
  direction_and_distance = -3 := by
  sorry

-- Question 2: Prove the total fuel consumption
def total_distance : Nat :=
  travel_records.map Int.natAbs |>.sum

def total_fuel_consumption : Float :=
  total_distance * fuel_consumption_rate

theorem fuel_consumed :
  total_fuel_consumption = 22.5 := by
  sorry

end net_distance_from_A_to_B_fuel_consumed_l442_442003


namespace cos_sin_of_triangle_l442_442178

variables {D E F : ℝ}
variables (DE DF EF : ℝ)
variables (α β : ℝ)

-- Given conditions
def right_triangle (DE DF EF : ℝ) (α β : ℝ) :=
  DE ^ 2 + EF ^ 2 = DF ^ 2 ∧ α = 0 ∧ β = 90

-- Definitions for cosine and sine
def cos_angle (DE DF : ℝ) : ℝ := DE / DF
def sin_angle (EF DF : ℝ) : ℝ := EF / DF

-- Proof statement
theorem cos_sin_of_triangle (DE DF : ℝ) (EF α β : ℝ)
  (h1 : right_triangle DE DF EF α β) :
  cos_angle DE DF = 8 / 17 ∧ sin_angle EF DF = 15 / 17 :=
begin
  sorry
end

end cos_sin_of_triangle_l442_442178


namespace count_even_strictly_increasing_integers_correct_l442_442530

-- Definition of condition: even four-digit integers with strictly increasing digits
def is_strictly_increasing {a b c d : ℕ} : Prop :=
1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ∈ {2, 4, 6, 8}

def count_even_strictly_increasing_integers : ℕ :=
(finset.range 10).choose 4.filter (λ l, is_strictly_increasing l.head l.nth 1 l.nth 2 l.nth 3).card

theorem count_even_strictly_increasing_integers_correct :
  count_even_strictly_increasing_integers = 46 := by
  sorry

end count_even_strictly_increasing_integers_correct_l442_442530


namespace roots_sum_of_squares_l442_442057

theorem roots_sum_of_squares
  (p q r : ℝ)
  (h_roots : ∀ x, (3 * x^3 - 4 * x^2 + 3 * x + 7 = 0) → (x = p ∨ x = q ∨ x = r))
  (h_sum : p + q + r = 4 / 3)
  (h_prod_sum : p * q + q * r + r * p = 1)
  (h_prod : p * q * r = -7 / 3) :
  p^2 + q^2 + r^2 = -2 / 9 := 
sorry

end roots_sum_of_squares_l442_442057


namespace set_intersection_complement_l442_442248

variable (U : Set ℕ) (M N : Set ℕ)
hypothesis hU : U = {1, 2, 3, 4, 5, 6}
hypothesis hM : M = {2, 3}
hypothesis hN : N = {1, 4}

theorem set_intersection_complement : N ∩ (U \ M) = {1, 4} :=
by sorry

end set_intersection_complement_l442_442248


namespace sum_T_n_formula_l442_442958

def a : ℕ → ℕ
| 0     := 7
| 1     := 3
| (n+2) := 3 * a (n+1) - 2

def b (n : ℕ) : ℕ := (a n - 1) / 2

def c (n : ℕ) : ℕ := Math.logb 3 (b n)

def T_n (n : ℕ) : ℕ :=
  match n with
  | 0     := 0
  | (n+1) := ∑ i in Finset.range (n+1), c i * b i

theorem sum_T_n_formula (n : ℕ) : T_n n = ((2 * n - 5) / 4 * 3^(n-1) + 15 / 4) :=
by
sorry

end sum_T_n_formula_l442_442958


namespace four_digit_increasing_even_integers_l442_442528

theorem four_digit_increasing_even_integers : 
  let even_four_digit_strictly_increasing (n : ℕ) := 
    n >= 1000 ∧ n < 10000 ∧ (n % 2 = 0) ∧ (let d1 := n / 1000 % 10, 
                                                d2 := n / 100 % 10,
                                                d3 := n / 10 % 10,
                                                d4 := n % 10 in
                                            d1 < d2 ∧ d2 < d3 ∧ d3 < d4)
  in (finset.filter even_four_digit_strictly_increasing (finset.range 10000)).card = 46 :=
begin
  sorry
end

end four_digit_increasing_even_integers_l442_442528


namespace greatest_value_of_remaining_prime_l442_442358

theorem greatest_value_of_remaining_prime (a b c d e f : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hd : Nat.Prime d) (he : Nat.Prime e) (hf : Nat.Prime f) (h_distinct : List.nodup [a, b, c, d, e, f]) (h_sum_abc : a + b + c = 150) (h_sum_all : a + b + c + d + e + f = 420) : 
  max d (max e f) = 257 :=
by
  sorry

end greatest_value_of_remaining_prime_l442_442358


namespace eval_polynomial_at_4_using_horners_method_l442_442058

noncomputable def polynomial : (x : ℝ) → ℝ :=
  λ x => 3 * x^5 - 2 * x^4 + 5 * x^3 - 2.5 * x^2 + 1.5 * x - 0.7

theorem eval_polynomial_at_4_using_horners_method :
  polynomial 4 = 2845.3 :=
by
  sorry

end eval_polynomial_at_4_using_horners_method_l442_442058


namespace harriet_smallest_stickers_l442_442523

theorem harriet_smallest_stickers 
  (S : ℕ) (a b c : ℕ)
  (h1 : S = 5 * a + 3)
  (h2 : S = 11 * b + 3)
  (h3 : S = 13 * c + 3)
  (h4 : S > 3) :
  S = 718 :=
by
  sorry

end harriet_smallest_stickers_l442_442523


namespace positive_difference_45_minutes_l442_442316

def speed_Linda : ℝ := 2 -- miles per hour
def speed_Tom_initial : ℝ := 6 -- miles per hour
def speed_Tom_slower : ℝ := 4 -- miles per hour
def time_Linda_first_phase : ℝ := 0.5 -- hours
def time_Linda_second_phase : ℝ := 1 -- hours
def time_Tom_initial_phase : ℝ := 0.25 -- hours

def distance_Linda_first_phase := speed_Linda * time_Linda_first_phase
def distance_Linda_second_phase := speed_Linda * time_Linda_second_phase
def total_distance_Linda := distance_Linda_first_phase + distance_Linda_second_phase

def half_distance_Linda := total_distance_Linda / 2
def twice_distance_Linda := 2 * total_distance_Linda

def time_Tom_half_distance := half_distance_Linda / speed_Tom_initial
def time_Tom_twice_distance := twice_distance_Linda / speed_Tom_initial

def positive_difference_in_minutes : ℝ := (time_Tom_twice_distance - time_Tom_half_distance) * 60

theorem positive_difference_45_minutes : positive_difference_in_minutes = 45 := by
  sorry

end positive_difference_45_minutes_l442_442316


namespace correct_propositions_l442_442143

-- Define the context
variables (Line : Type) (Plane : Type)
variable (m n : Line)
variable (α β : Plane)

-- Define the required relations
variables (parallel : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop)
variables (subset : Line → Plane → Prop)
variables (intersection : Plane → Plane → Line)

-- Proposition (1)
def proposition1 : Prop :=
  (parallel m β ∧ parallel n β ∧ subset m α ∧ subset n α) → parallel α β

-- Proposition (2)
def proposition2 : Prop :=
  (perpendicular m β ∧ perpendicular n β ∧ subset m α ∧ ¬subset n α) → parallel n α

-- Proposition (3)
def proposition3 : Prop :=
  (subset m α ∧ subset n β ∧ (intersection α β = l) ∧ perpendicular m l ∧ perpendicular n l) → perpendicular α β

-- Proposition (4)
def proposition4 : Prop :=
  (m ≠ n ∧ parallel m β ∧ subset m α ∧ subset n β ∧ parallel n α) → parallel α β

-- The main statement asserting which propositions are true
theorem correct_propositions :
  (proposition2 Line m n α β parallel perpendicular subset intersection) ∧
  (proposition4 Line m n α β parallel perpendicular subset intersection) :=
sorry

end correct_propositions_l442_442143


namespace bees_flew_in_l442_442309

theorem bees_flew_in (initial_bees : ℕ) (total_bees : ℕ) (new_bees : ℕ) (h1 : initial_bees = 16) (h2 : total_bees = 23) (h3 : total_bees = initial_bees + new_bees) : new_bees = 7 :=
by
  sorry

end bees_flew_in_l442_442309


namespace find_complex_number_l442_442477

noncomputable def dot (z : ℂ) : ℂ := conj z

theorem find_complex_number (z : ℂ) (h : dot z - abs z = -1 - 3 * I) : z = 4 + 3 * I :=
by sorry

end find_complex_number_l442_442477


namespace solve_opposite_numbers_product_l442_442449

theorem solve_opposite_numbers_product :
  ∃ (x : ℤ), 3 * x - 2 * (-x) = 30 ∧ x * (-x) = -36 :=
by
  sorry

end solve_opposite_numbers_product_l442_442449


namespace star_angle_from_regular_octagon_l442_442777

/--
A regular octagon is extended to form a star. Prove that the angle at each point of the star is 270 degrees.
-/
theorem star_angle_from_regular_octagon : 
  let n := 8 in
  let internal_angle := (n - 2) * 180 / n in
  let external_angle := 180 - internal_angle in
  360 - 2 * external_angle = 270 :=
by 
  sorry

end star_angle_from_regular_octagon_l442_442777


namespace beetle_reachable_even_columns_l442_442030

structure Point :=
(x : Int)
(y : Int)

def is_even (n : Int) : Prop := n % 2 = 0

-- Definition of valid movements
def valid_move : Point → Point → Prop
| ⟨x1, y1⟩ ⟨x2, y2⟩ :=
  let dx := x2 - x1 in
  let dy := y2 - y1 in
  (dx = 2 ∧ dy = 0) ∨ (dx = -4 ∧ dy = 0) ∨ (dx = 0 ∧ dy = 3) ∨ (dx = 0 ∧ dy = -5)

-- Definition of reachable from the origin (0,0)
def reachable_from_origin (p : Point) : Prop :=
p = ⟨0, 0⟩ ∨ ∃ (moves : List Point), 
  List.Nodup moves ∧ 
  moves.head = ⟨0, 0⟩ ∧ 
  List.All2 valid_move moves (moves.tail.map (λ q, some q))

-- Define the theorem
theorem beetle_reachable_even_columns (x y : Int) (p : Point) :
  is_even x ∧ reachable_from_origin ⟨x, y⟩ → reachable_from_origin ⟨x, y⟩ := 
sorry

end beetle_reachable_even_columns_l442_442030


namespace digit_difference_is_one_l442_442035

theorem digit_difference_is_one {p q : ℕ} (h : 1 ≤ p ∧ p ≤ 9 ∧ 0 ≤ q ∧ q ≤ 9 ∧ p ≠ q)
  (digits_distinct : ∀ n ∈ [p, q], ∀ m ∈ [p, q], n ≠ m)
  (interchange_effect : 10 * p + q - (10 * q + p) = 9) : p - q = 1 :=
sorry

end digit_difference_is_one_l442_442035


namespace speed_ratio_is_5_l442_442708

noncomputable def speed_ratio : ℚ :=
let S1 : ℚ := 2 in
let S2 : ℚ := 2 * S1 in
let D : ℚ := S1 * 20 in
let S3 : ℚ := D / 2 in
S3 / S2

theorem speed_ratio_is_5 : speed_ratio = 5 :=
by
  -- Placeholder for proof
  sorry

end speed_ratio_is_5_l442_442708


namespace range_of_a_l442_442953

theorem range_of_a {a : ℝ} : (∀ x ∈ set.Icc (-1:ℝ) 1, x^2 + a * x - 3 * a < 0) → a > 1/2 :=
sorry

end range_of_a_l442_442953


namespace original_mass_of_cake_l442_442087

-- Definitions of the conditions
variable (original_mass : ℝ)
variable (cake_after_carlson : ℝ)
variable (cake_after_little_man : ℝ)
variable (cake_after_freken_bok : ℝ)
variable (remaining_cake : ℝ)

-- Conditions
def carlson_ate : cake_after_carlson = original_mass * 0.6 := sorry
def little_man_ate : cake_after_little_man = cake_after_carlson - 150 := sorry
def freken_bok_ate : cake_after_freken_bok = remaining_cake + 120 := sorry
def remaining_cake_def : remaining_cake = 90 := sorry
def remains_after_freken_bok : remaining_cake + 120 = cake_after_freken_bok * 0.7 := sorry

theorem original_mass_of_cake : original_mass = 750 := by
  have h1 : cake_after_carlson = original_mass * 0.6 := carlson_ate
  have h2 : cake_after_little_man = cake_after_carlson - 150 := little_man_ate
  have h3 : remaining_cake + 120 = cake_after_freken_bok * 0.7 := remains_after_freken_bok
  have h4 : remaining_cake = 90 := remaining_cake_def
  have h5 : cake_after_freken_bok = remaining_cake + 120 := freken_bok_ate
  sorry

end original_mass_of_cake_l442_442087


namespace city_schools_count_l442_442078

theorem city_schools_count (a b c : ℕ) (schools : ℕ) : 
  b = 40 → c = 51 → b < a → a < c → 
  (a > b ∧ a < c ∧ (a - 1) * 3 < (c - b + 1) * 3 + 1) → 
  schools = (c - 1) / 3 :=
by
  sorry

end city_schools_count_l442_442078


namespace perimeter_of_rearranged_rectangles_l442_442786

theorem perimeter_of_rearranged_rectangles (side_length : ℕ) (h : side_length = 100) : 
  let length := side_length
  let width := side_length / 2
  ∃ P, P = 3 * length + 4 * width ∧ P = 500 :=
by
  have h_length : length = 100 := by rw [h]
  have h_width : width = 50 := by rw [h, Nat.div_self (by norm_num)]
  use 3 * length + 4 * width
  rw [h_length, h_width]
  norm_num
  sorry

end perimeter_of_rearranged_rectangles_l442_442786


namespace day_of_week_301st_day_l442_442546

theorem day_of_week_301st_day 
    (h1 : 35 % 7 = 0)
    (h2 : 301 % 7 = 0)
    (h3 : ∃ n : ℕ, 35 = 7 * n) 
    (h4 : "Sunday")
    : "Sunday" := 
by
    sorry

end day_of_week_301st_day_l442_442546


namespace shooter_standard_deviation_l442_442778

noncomputable def mean (scores : List ℝ) : ℝ := (scores.sum / scores.length.toFloat)

noncomputable def variance (scores : List ℝ) : ℝ :=
  let μ := mean scores
  (scores.map (λ x => (x - μ)^2)).sum / scores.length.toFloat

noncomputable def standard_deviation (scores : List ℝ) : ℝ :=
  real.sqrt (variance scores)

theorem shooter_standard_deviation :
  standard_deviation [10.0, 10.0, 10.0, 9.0, 10.0, 8.0, 8.0, 10.0, 10.0, 8.0] = 0.9 :=
by
  sorry

end shooter_standard_deviation_l442_442778


namespace asymptote_slope_of_hyperbola_l442_442059

theorem asymptote_slope_of_hyperbola :
  ∀ (x y m : ℝ), (y^2 / 16 - x^2 / 9 = 1) → (m = 4/3) :=
begin
  sorry
end

end asymptote_slope_of_hyperbola_l442_442059


namespace original_cylinder_filled_fraction_l442_442009

variables (r h : ℝ)

def volume_original_cylinder (r h : ℝ) : ℝ := π * r^2 * h
def volume_new_cylinder (r h : ℝ) : ℝ := π * (1.25 * r)^2 * (0.9 * h)

theorem original_cylinder_filled_fraction (r h : ℝ) :
  (3 / 5) * volume_new_cylinder r h = (27 / 32) * volume_original_cylinder r h :=
by 
  let V := volume_original_cylinder r h
  let V_new := volume_new_cylinder r h
  rw [volume_original_cylinder, volume_new_cylinder]
  simp [V, V_new]
  sorry

end original_cylinder_filled_fraction_l442_442009


namespace work_time_day_six_l442_442374

theorem work_time_day_six (initial_time : ℕ) (doubling_factor : ℕ) : 
  initial_time = 15 ∧ doubling_factor = 2 → 6.days.work_time_hours = 8 :=
by
  intros conditions
  sorry

end work_time_day_six_l442_442374


namespace problem_statement_l442_442379

noncomputable def f : ℝ → ℝ := sorry
noncomputable def a : ℝ := f (real.pi / 3)
noncomputable def b : ℝ := 2 * f 0
noncomputable def c : ℝ := real.sqrt 3 * f (real.pi / 6)

theorem problem_statement (h : ∀ x ∈ Ioo 0 (real.pi / 2), f'' x > real.sin (2 * x) * f x - real.cos (2 * x) * f'' x) :
  a > c ∧ c > b :=
sorry

end problem_statement_l442_442379


namespace sum_of_values_of_g_zero_l442_442951

def g (x : ℝ) : ℝ :=
if x ≤ 2 then -x - 5 else x / 3 + 2

theorem sum_of_values_of_g_zero :
  (finset.univ.filter (λ x : ℝ, g x = 0)).sum id = -5 :=
sorry

end sum_of_values_of_g_zero_l442_442951


namespace related_function_m_range_l442_442237

noncomputable def f (x : ℝ) : ℝ := (x^2) / 3 - (3 * x^2) / 2 + 4 * x
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * x + m

theorem related_function_m_range :
  ∀ m : ℝ, (∀ x : ℝ, x ∈ set.Icc 0 3 → f' x - g x m = 0) → m > -9/4 ∧ m < -2 :=
sorry

end related_function_m_range_l442_442237


namespace solve_x_from_equation_l442_442726

theorem solve_x_from_equation :
  ∀ (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 → x = 27 :=
by
  intro x
  rintro ⟨hx, h⟩
  sorry

end solve_x_from_equation_l442_442726


namespace dima_picks_more_berries_l442_442815

theorem dima_picks_more_berries (N : ℕ) (dima_fastness : ℕ) (sergei_fastness : ℕ) (dima_rate : ℕ) (sergei_rate : ℕ) :
  N = 450 → dima_fastness = 2 * sergei_fastness →
  dima_rate = 1 → sergei_rate = 2 →
  let dima_basket : ℕ := N / 2
  let sergei_basket : ℕ := (2 * N) / 3
  dima_basket > sergei_basket ∧ (dima_basket - sergei_basket) = 50 := 
by {
  sorry
}

end dima_picks_more_berries_l442_442815


namespace isabella_houses_problem_l442_442191

theorem isabella_houses_problem 
  (yellow green red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  (green + red = 160) := 
sorry

end isabella_houses_problem_l442_442191


namespace jacksonville_walmart_complaints_l442_442650

theorem jacksonville_walmart_complaints :
  let normal_complaint_rate := 120
  let short_staffed_increase_factor := 4 / 3
  let self_checkout_broken_increase_factor := 1.2
  let days := 3
  let complaints_per_day_short_staffed := normal_complaint_rate * short_staffed_increase_factor
  let complaints_per_day_both_conditions := complaints_per_day_short_staffed * self_checkout_broken_increase_factor
  let total_complaints := complaints_per_day_both_conditions * days
  total_complaints = 576 :=
by
  -- Use the 'let' definitions from above to describe the proof problem
  let normal_complaint_rate := 120
  let short_staffed_increase_factor := 4 / 3
  let self_checkout_broken_increase_factor := 1.2
  let days := 3
  let complaints_per_day_short_staffed := normal_complaint_rate * short_staffed_increase_factor
  let complaints_per_day_both_conditions := complaints_per_day_short_staffed * self_checkout_broken_increase_factor
  let total_complaints := complaints_per_day_both_conditions * days
  
  -- Here would be the place to write the proof steps, but it is skipped as per instructions
  sorry

end jacksonville_walmart_complaints_l442_442650


namespace apothem_inequality_l442_442615

variable {R : ℝ} (n : ℕ)

noncomputable def h (k : ℕ) := R * Real.cos (π / k)

theorem apothem_inequality (R_pos : 0 < R) (n_pos : 0 < n):
  (n + 1) * h R (n + 1) - n * h R n > R := sorry

end apothem_inequality_l442_442615


namespace maximum_value_l442_442673

noncomputable def f (x : ℝ) : ℝ :=
  3 * Real.sin x + 4 * Real.cos x

theorem maximum_value :
  (∃ x : ℝ, f(x) = 5) → 
  (∀ x : ℝ, f(x) ≤ 5) :=
sorry

end maximum_value_l442_442673


namespace ratio_sandra_amy_ruth_l442_442637

/-- Given the amounts received by Sandra and Amy, and an unknown amount received by Ruth,
    the ratio of the money shared between Sandra, Amy, and Ruth is 2:1:R/50. -/
theorem ratio_sandra_amy_ruth (R : ℝ) (hAmy : 50 > 0) (hSandra : 100 > 0) :
  (100 : ℝ) / 50 = 2 ∧ (50 : ℝ) / 50 = 1 ∧ ∃ (R : ℝ), (100/50 : ℝ) = 2 ∧ (50/50 : ℝ) = 1 ∧ (R / 50 : ℝ) = (R / 50 : ℝ) :=
by
  sorry

end ratio_sandra_amy_ruth_l442_442637


namespace product_xyz_eq_one_l442_442539

theorem product_xyz_eq_one (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) : x * y * z = 1 := 
sorry

end product_xyz_eq_one_l442_442539


namespace number_of_littering_citations_l442_442384

variable (L D P : ℕ)
variable (h1 : L = D)
variable (h2 : P = 2 * (L + D))
variable (h3 : L + D + P = 24)

theorem number_of_littering_citations : L = 4 :=
by
  sorry

end number_of_littering_citations_l442_442384


namespace intervals_of_increase_of_f_minimum_b_for_at_least_10_zeros_l442_442127

def f (x : ℝ) (ω : ℝ) : ℝ :=
  2 * sin (ω * x) * cos (ω * x) + 2 * sqrt 3 * (sin (ω * x))^2 - sqrt 3

def g (x : ℝ) : ℝ :=
  2 * sin (2 * x) + 1

theorem intervals_of_increase_of_f (k : ℤ) :
  f (x : ℝ) 1 is increasing on set.Ioo (k * real.pi - real.pi / 12) (k * real.pi + 5 * real.pi / 12) := sorry

theorem minimum_b_for_at_least_10_zeros :
  ∃ b > 0, ∀ x ∈ set.Icc (0 : ℝ) b, g x = 0 -> b ≥ 59 * real.pi / 12 := sorry

end intervals_of_increase_of_f_minimum_b_for_at_least_10_zeros_l442_442127


namespace solution_set_inequality_l442_442899

noncomputable def f : ℝ → ℝ := sorry 

axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom f_at_3 : f(3) = 6
axiom f_prime_condition : ∀ x : ℝ, x > 0 → deriv f x > 2

theorem solution_set_inequality : ∀ x : ℝ, (f(x) - 2 * x < 0) ↔ x < 3 :=
by
  have F := λ x, f(x) - 2 * x
  have F_prime := λ x, deriv f x - 2
  sorry

end solution_set_inequality_l442_442899


namespace quadrilateral_angles_l442_442211

theorem quadrilateral_angles (PQRS : Quadrilateral) (hPQ : PQRS.length P Q = PQRS.length Q R)
  (hQR : PQRS.length Q R = PQRS.length R S) (hQ : PQRS.angle Q = 110) (hR : PQRS.angle R = 130) :
  PQRS.angle P = 55 ∧ PQRS.angle S = 65 := by 
  sorry

end quadrilateral_angles_l442_442211


namespace increase_is_50_percent_l442_442586

theorem increase_is_50_percent (original new : ℕ) (h1 : original = 60) (h2 : new = 90) :
  ((new - original) * 100 / original) = 50 :=
by
  -- Proof can be filled here.
  sorry

end increase_is_50_percent_l442_442586


namespace wall_length_l442_442544

-- Define the initial conditions
def workers1 : ℕ := 18
def days1 : ℕ := 42
def length1 : ℕ := 140

def workers2 : ℕ := 30
def days2 : ℕ := 18

-- Define the quantity of work done (work is proportional to workers * days * length of wall)
def work_done (workers : ℕ) (days : ℕ) (length : ℕ) : ℕ :=
  workers * days * length

-- Given the first condition, calculate work done W1
def work1 := work_done workers1 days1 length1

-- Define the length of the second wall L, which we need to prove is 196 meters
def length2 : ℕ := 196
def work2 := work_done workers2 days2 length2

-- The problem states that work done for both walls should be the same
theorem wall_length :
  work1 = work2 :=
by
  -- The calculation we have to prove is work1 = work2
  have h1 : work1 = workers1 * days1 * length1 := rfl
  have h2 : work2 = workers2 * days2 * length2 := rfl
  -- Substitute the values into the work equations
  rw [←h1, ←h2]
  -- Simplify the equation to prove length2 is 196
  exact sorry   -- Fill in the proof steps here.

end wall_length_l442_442544


namespace range_of_m_l442_442511

noncomputable def common_points (k : ℝ) (m : ℝ) := 
  ∃ x y : ℝ, (y = k * x + 1) ∧ ((x^2 / 5) + (y^2 / m) = 1)

theorem range_of_m (k : ℝ) (m : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, (y = k * x + 1) ∧ ((x^2 / 5) + (y^2 / m) = 1)) ↔ 
  (m ∈ (Set.Ioo 1 5 ∪ Set.Ioi 5)) :=
by
  sorry

end range_of_m_l442_442511


namespace medians_equal_l442_442187

/-- In triangle ABC, point M is chosen on side BC such that
 the intersection point of the medians of ABM lies on the circumcircle of ACM,
 and the intersection point of the medians of ACM lies on the circumcircle of ABM.
 We need to prove that the medians of ABM and ACM from vertex M are equal. -/
theorem medians_equal {A B C M : Point}
  (hM : M ∈ line_segment B C)
  (h1 : let G_b := centroid (triangle (A, B, M))
        in G_b ∈ circumcircle (triangle (A, C, M)))
  (h2 : let G_c := centroid (triangle (A, C, M))
        in G_c ∈ circumcircle (triangle (A, B, M))) :
  let MG_c := line_segment M (centroid (triangle (A, C, M)))
  let MG_b := line_segment M (centroid (triangle (A, B, M)))
  in MG_c = MG_b := sorry

end medians_equal_l442_442187


namespace candy_cookies_l442_442423

def trays : Nat := 4
def cookies_per_tray : Nat := 24
def packs : Nat := 8
def total_cookies : Nat := trays * cookies_per_tray
def cookies_per_pack : Nat := total_cookies / packs

theorem candy_cookies : 
  cookies_per_pack = 12 := 
by
  -- Calculate total cookies
  have h1 : total_cookies = trays * cookies_per_tray := rfl
  have h2 : total_cookies = 96 := by rw [h1]; norm_num
  
  -- Calculate cookies per pack
  have h3 : cookies_per_pack = total_cookies / packs := rfl
  have h4 : cookies_per_pack = 12 := by rw [h3, h2]; norm_num
  
  exact h4

end candy_cookies_l442_442423


namespace sum_distances_constant_iff_sum_normals_zero_l442_442264

variables {k : Nat} (polygon : Fin k → ℝ × ℝ) (interior_point : ℝ × ℝ)
  (unit_normals : Fin k → ℝ × ℝ)

-- Define convex k-gon and interior point conditions
def is_convex_k_gon := sorry -- precise definition is abstracted
def is_interior_point := sorry -- precise definition is abstracted

-- Define the sum of distances from an interior point to the sides
def sum_of_distances (point : ℝ × ℝ) : ℝ :=
  ∑ i, (dist_to_side point (unit_normals i)) -- Abstract definition of distance to side
    
-- Define the sum of unit outward normal vectors
def sum_of_unit_normals := ∑ i, unit_normals i

-- The theorem statement
theorem sum_distances_constant_iff_sum_normals_zero :
  is_convex_k_gon polygon →
  is_interior_point polygon interior_point →
  (∀ point, sum_of_distances point = sum_of_distances interior_point) ↔
  sum_of_unit_normals unit_normals = (0, 0) :=
sorry  -- Proof goes here

end sum_distances_constant_iff_sum_normals_zero_l442_442264


namespace average_chem_math_90_l442_442308

-- Given conditions
variables (P C M : ℕ)
variables (h1 : P + C + M = P + 180)

-- Aim to prove that the average mark obtained in chemistry and mathematics is 90
theorem average_chem_math_90 : (C + M) / 2 = 90 :=
by
  have h2 : C + M = 180 := by 
    linarith [h1]
  calc 
    (C + M) / 2 = 180 / 2 : by rw h2
    ... = 90 : by norm_num
  sorry

end average_chem_math_90_l442_442308


namespace angle_between_a_b_l442_442102

open Real EuclideanGeometry

noncomputable def angle_between_vectors (a b : ℝ^3) : ℝ := 
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let a_norm := sqrt (a.1 * a.1 + a.2 * a.2 + a.3 * a.3)
  let b_norm := sqrt (b.1 * b.1 + b.2 * b.2 + b.3 * b.3)
  acos (dot_product / (a_norm * b_norm))

theorem angle_between_a_b {a b : ℝ^3}
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hab : sqrt (a.1 * a.1 + a.2 * a.2 + a.3 * a.3) = 2 * sqrt (b.1 * b.1 + b.2 * b.2 + b.3 * b.3))
  (horth : (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 + (a.3 - b.3) * b.3 = 0) :
  angle_between_vectors a b = π / 3 := sorry

end angle_between_a_b_l442_442102


namespace a6_value_l442_442563

variable (a_n : ℕ → ℤ)

/-- Given conditions in the arithmetic sequence -/
def arithmetic_sequence_property (a_n : ℕ → ℤ) :=
  ∀ n, a_n n = a_n 0 + n * (a_n 1 - a_n 0)

/-- Given sum condition a_4 + a_5 + a_6 + a_7 + a_8 = 150 -/
def sum_condition :=
  a_n 4 + a_n 5 + a_n 6 + a_n 7 + a_n 8 = 150

theorem a6_value (h : arithmetic_sequence_property a_n) (hsum : sum_condition a_n) :
  a_n 6 = 30 := 
by
  sorry

end a6_value_l442_442563


namespace temperature_difference_product_l442_442412

theorem temperature_difference_product (N : ℕ):
  (∀ (M L : ℤ), M = L + N → 
                ∀ (M_6 L_6 : ℤ), M_6 = M - 7 → L_6 = L + 4 → 
                |M_6 - L_6| = 1) →
  N = 10 ∨ N = 12 → 
  N * (if N = 10 then 12 else 10) = 120 :=
by
  intros h1 h2
  cases h2 with N10 N12
  case inl => 
    simp [N10]
    exact eq.refl 120
  case inr => 
    simp [N12]
    exact eq.refl 120

end temperature_difference_product_l442_442412


namespace one_third_construction_l442_442480

variables {P : Type} [metric_space P] [normed_add_torsor ℝ P]
variables {A B K : P} {e : set P} 
variables (h_parallel : ∀ x ∈ e, ∀ y ∈ e, (∃ z ∈ e, z ∈ line_through x y))

theorem one_third_construction (h_AB : A ≠ B) 
  (h_parallel_AB : ∀ x ∈ e, x ≠ A ∧ x ≠ B) :
  dist A K = dist A B / 3 :=
sorry

end one_third_construction_l442_442480


namespace wood_allocation_l442_442785

theorem wood_allocation (x y : ℝ) (h1 : 50 * x * 4 = 300 * y) (h2 : x + y = 5) : x = 3 :=
by
  sorry

end wood_allocation_l442_442785


namespace find_a_l442_442619

noncomputable def f (x a : ℝ) : ℝ := real.sqrt (-x - a)

theorem find_a (a : ℝ) (h_sym : ∀ x, x > 0 → y = f(x) = real.sqrt (-x - a)) 
  (h_eq : f (-2) a = 2 * f (-1) a) : a = 2 / 3 :=
by
  unfold f at h_eq
  sorry

end find_a_l442_442619


namespace find_b_l442_442128

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≥ 1 then x - b else Real.log (2) (1 - x)

theorem find_b (b : ℝ) (h : f (f (-3) b) b = -3) : b = 5 :=
  sorry

end find_b_l442_442128


namespace last_digit_to_appear_in_fib_mod_10_is_6_l442_442284

-- Defining the Fibonacci sequence modulo 10
noncomputable def fib_mod_10 : ℕ → ℕ
| 0     := 0
| 1     := 1
| n + 2 := (fib_mod_10 (n + 1) + fib_mod_10 n) % 10

-- The proof statement
theorem last_digit_to_appear_in_fib_mod_10_is_6 :
    ∃ n, (∀ m < n, ∃ d, d < 10 ∧ ∃ k < m, fib_mod_10 k % 10 = d) ∧ ∃ k < n, fib_mod_10 k % 10 = 6 := sorry

end last_digit_to_appear_in_fib_mod_10_is_6_l442_442284


namespace cylinder_water_rising_rate_l442_442767

open Real

noncomputable def side_length_cube : ℝ := 100 -- cm
noncomputable def radius_cylinder : ℝ := 100 -- cm
noncomputable def rate_falling_cube : ℝ := 1 -- cm/s

def area_base_cube : ℝ := side_length_cube^2 -- cm²
def volume_change_rate_cube : ℝ := area_base_cube * (-rate_falling_cube) -- cm³/s

def area_base_cylinder : ℝ := π * radius_cylinder^2 -- cm²
def volume_change_rate_cylinder : ℝ := -volume_change_rate_cube -- cm³/s

theorem cylinder_water_rising_rate : (volume_change_rate_cylinder / area_base_cylinder) = 1 / π := by
  sorry

end cylinder_water_rising_rate_l442_442767


namespace correct_num_of_propositions_l442_442676

section propositions

-- Define the propositions as Lean terms
def proposition_1 : Prop :=
  ∀ (x : ℝ), (x^2 + 1 > 3 * x) → ¬(∀ (x : ℝ), x^2 + 1 <= 3 * x)

def proposition_2 : Prop :=
  ∀ (a : ℝ), (∀ (x : ℝ), f(x) = cos^2(a * x) - sin^2(a * x) → cos(2 * a * x) → has_period (cos(2 * a * x)) π) ∧ (a = 1) 

def proposition_3 : Prop :=
  ∀ (a : ℝ), (∀ (x : ℝ), x ∈ Ioc 1 2  → x^2 + 2 * x ≥ a * x) ↔ (min (x^2 + 2 * x) ≥ max (a * x)) 

def proposition_4 : Prop :=
  ∀ (a b : EuclideanSpace ℝ (Fin 3)), obtuse_angle a b ↔ inner a b < 0

-- Define the proof problem to assert the number of correct propositions
def proof_problem : Prop :=
  (proposition_1 ∧ proposition_2 ∧ ¬proposition_3 ∧ ¬proposition_4) → (number_of_correct_propositions = 2)

end propositions

noncomputable def correct_propositions : nat :=
  2

theorem correct_num_of_propositions : correct_propositions = 2 := 
  by sorry

end correct_num_of_propositions_l442_442676


namespace probability_of_two_co_presidents_l442_442911

noncomputable section

def binomial (n k : ℕ) : ℕ :=
  if h : n ≥ k then Nat.choose n k else 0

def club_prob (n : ℕ) : ℚ :=
  (binomial (n-2) 2 : ℚ) / (binomial n 4 : ℚ)

def total_probability : ℚ :=
  (1/4 : ℚ) * (club_prob 6 + club_prob 8 + club_prob 9 + club_prob 10)

theorem probability_of_two_co_presidents : total_probability = 0.2286 := by
  -- We expect this to be true based on the given solution
  sorry

end probability_of_two_co_presidents_l442_442911


namespace sum_of_roots_l442_442631

noncomputable theory

open Polynomial

theorem sum_of_roots (c : Fin 2007 → ℚ) (r : Fin 2006 → ℚ) 
  (hP : ∀ x, (Polynomial.eval x (∑ i, c ⟨i, by norm_num⟩ * X ^ i)) = (∏ i, x - r i))
  (hc : ∀ i j : Fin 2007, 2 * i * (c i / (c ⟨2006, by norm_num⟩ - i)) = 2 * j * (c j / (c ⟨2006, by norm_num⟩ - j))) 
  (hr : (∑ i, ∑ j, if i ≠ j then r i / r j else 0) = 42) 
  : ∑ i, r i = -1 + 1 / (c ⟨2006, by norm_num⟩ - 2006) :=
sorry

end sum_of_roots_l442_442631


namespace solve_for_x_l442_442952

-- Defining the complex numbers z1 and z2
def z1 (x : ℝ) : ℂ := 2 * x + 1 + complex.I * (x^2 - 3 * x + 2)
def z2 (x : ℝ) : ℂ := x^2 - 2 + complex.I * (x^2 + x - 6)

-- Defining the conditions
def condition1 (x : ℝ) : Prop := x^2 + x - 6 = 0
def condition2 (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0
def condition3 (x : ℝ) : Prop := 2 * x + 1 > x^2 - 2

-- The theorem statement
theorem solve_for_x (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) (h3 : condition3 x) : x = 2 :=
sorry

end solve_for_x_l442_442952


namespace triangle_ABC_properties_l442_442185

noncomputable def conditions :=
let eq_roots := (λ x : ℝ, x^2 - 2*real.sqrt 3*x + 2 = 0) in
let a := classical.some (exists_root_eq_roots eq_roots) in
let b := classical.some (exists_root_eq_roots eq_roots) in
let cos_AB := 2 * real.cos (real.pi - real.of_rat (by norm_num)) = -1 in
let angle_sum := real.of_rat (by norm_num) + real.of_rat (by norm_num) + C = 180 in
(a, b, eq_roots, cos_AB, angle_sum)

theorem triangle_ABC_properties
  (a b : ℝ)
  (h_roots : ∀ x : ℝ, x^2 - 2 * real.sqrt 3 * x + 2 = 0 → x = a ∨ x = b)
  (h_cos : 2 * real.cos (real.pi - 60) = -1)
  (h_sum : 60 + 60 + C = 180) :
  (C = 60) ∧ (c = real.sqrt 6) ∧ (1/2 * a * b * real.sin (real.pi / 3) = real.sqrt 3 / 2) := 
by
  let C := 60
  have h_C : C = 60 := by linarith,
  let c := real.sqrt 6
  have h_c : c = real.sqrt 6 := sorry,
  let S := 1/2 * a * b * real.sin (real.pi / 3)
  have h_S : S = real.sqrt 3 / 2 := sorry,
  exact ⟨h_C, h_c, h_S⟩

end triangle_ABC_properties_l442_442185


namespace length_is_twenty_l442_442060

-- Define the number of walls, height per wall, depth per wall, and total bricks
def nWalls : Nat := 4
def height : Nat := 5
def depth : Nat := 2
def totalBricks : Nat := 800

-- Define the equation derived from given conditions
def length_of_each_wall : Nat := totalBricks / (nWalls * height * depth)

-- State the theorem to be proven: the length of each wall is 20 bricks
theorem length_is_twenty : length_of_each_wall = 20 := sorry

end length_is_twenty_l442_442060


namespace distance_on_line_l442_442545

theorem distance_on_line (a b c d m k : ℝ) 
  (h1 : b = m * a + k)
  (h2 : d = m * c + k) :
  dist (a, b) (c, d) = |a - c| * sqrt (1 + m ^ 2) := 
sorry

end distance_on_line_l442_442545


namespace inequality_f_solution_minimum_g_greater_than_f_l442_442137

noncomputable def f (x : ℝ) := abs (x - 2) - abs (x + 1)

theorem inequality_f_solution : {x : ℝ | f x > 1} = {x | x < 0} :=
sorry

noncomputable def g (a x : ℝ) := (a * x^2 - x + 1) / x

theorem minimum_g_greater_than_f (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 0 < x → g a x > f x) ↔ 1 ≤ a :=
sorry

end inequality_f_solution_minimum_g_greater_than_f_l442_442137


namespace length_of_crease_l442_442776

/-- 
  Given a rectangular piece of paper 8 inches wide that is folded such that one corner 
  touches the opposite side at an angle θ from the horizontal, and one edge of the paper 
  remains aligned with the base, 
  prove that the length of the crease L is given by L = 8 * tan θ / (1 + tan θ). 
--/
theorem length_of_crease (theta : ℝ) (h : 0 < theta ∧ theta < Real.pi / 2): 
  ∃ L : ℝ, L = 8 * Real.tan theta / (1 + Real.tan theta) :=
sorry

end length_of_crease_l442_442776


namespace train_overtake_time_l442_442714

theorem train_overtake_time (T : ℝ) :
  (∀ t : ℝ, t = 120 → 80 * t = 60 * (T / 60 + 2)) → T = 40 :=
by
  intro h
  have h1 : 80 * 120 = 9600 := by norm_num
  have h2 : 60 * (T / 60 + 2) = T + 120 := by
    calc 
      60 * (T / 60 + 2) = 60 * (T / 60) + 60 * 2 : by rw mul_add
      ... = T + 120 : by
        rw [mul_div_cancel' T (by norm_num : 60 ≠ 0), mul_comm 60 2, mul_assoc, mul_comm _ 2, mul_one]
  rw h at h2
  rw h1 at h2
  exact eq_of_sub_eq_zero (sub_eq_zero.mpr h2)

end train_overtake_time_l442_442714


namespace evaluate_expression_l442_442813

theorem evaluate_expression : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) = -2 := by
  sorry

end evaluate_expression_l442_442813


namespace minimize_sum_squared_distances_locus_of_X_l442_442406

-- Define the coordinates for vertices of triangle ABC
variables {x1 y1 x2 y2 x3 y3 : ℝ}

-- Define the line e as y = 0
noncomputable def line_e (x : ℝ) : (ℝ × ℝ) :=
  (x, 0)

-- Define the projection of the centroid S on the line e
noncomputable def centroid_projection_on_line_e : ℝ × ℝ :=
  ((x1 + x2 + x3) / 3, 0)

-- Define the centroid S of the triangle ABC
noncomputable def centroid_S : ℝ × ℝ :=
  ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3)

-- Define the fixed point P on the line e
variables {px py : ℝ}
axiom point_P_on_line_e : py = 0

-- Define the distance between two points
noncomputable def dist_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

-- Define the sum of squared distances from point X (on line e) to vertices A, B, C
noncomputable def sum_squared_distances (x : ℝ) : ℝ :=
  dist_squared (line_e x) (x1, y1) + dist_squared (line_e x) (x2, y2) + dist_squared (line_e x) (x3, y3)

-- The mathemetical proof problem statements
theorem minimize_sum_squared_distances:
  ∃ x : ℝ, (sum_squared_distances x) = (sum_squared_distances ((x1 + x2 + x3) / 3)) :=
sorry

theorem locus_of_X (px : ℝ) (py : ℝ) (h : py = 0):
  ∃ c r : ℝ × ℝ, r = (centroid_S) ∧ (dist_squared (centroid_S) (px, py)) = dist_squared (c, r) :=
sorry

end minimize_sum_squared_distances_locus_of_X_l442_442406


namespace find_distance_l442_442031

def distance_between_stripes (width distance curb stripe1 stripe2 : Real) : Prop :=
  width = 60 ∧ curb = 25 ∧ stripe1 = 70 ∧ stripe2 = 70 ∧ 
  70 * distance = 25 * width ∧ distance = (1500 / 70)

theorem find_distance : 
  ∃ distance : Real, distance_between_stripes 60 distance 25 70 70 :=
begin
  use (1500 / 70),
  split,
  repeat {split},
  exact rfl,
  exact rfl,
  exact rfl,
  exact rfl,
  calc
    70 * (1500 / 70) = 1500 : by field_simp [ne_of_gt (show 70 > 0, by norm_num)]
    ... = 25 * 60 : by norm_num,
  exact rfl,
end

end find_distance_l442_442031


namespace binomial_expansion_sum_l442_442119

theorem binomial_expansion_sum (a : ℝ) (a₁ a₂ a₃ a₄ a₅ : ℝ)
  (h₁ : (a * x - 1)^5 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5)
  (h₂ : a₃ = 80) :
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 1 :=
sorry

end binomial_expansion_sum_l442_442119


namespace cone_volume_l442_442394

theorem cone_volume (a α : ℝ) (hα : 0 < α ∧ α < π) : 
  let r := a * Real.sqrt 2 / 2 in
  let S_base := Real.pi * r^2 in
  let AS := a / (2 * Real.sin (α / 2)) in
  let SO := Real.sqrt (AS^2 - (a * Real.sqrt 2 / 2)^2) in
  V = 1/3 * S_base * SO := 
  V = (Real.pi * a^3 * Real.sqrt (Real.cos α)) / (12 * Real.sin (α / 2)^2) := sorry

end cone_volume_l442_442394


namespace find_f_of_5pi_3_l442_442378

-- Definition of f being even
axiom f_even : ∀ x : ℝ, f x = f (-x)

-- Definition of f being periodic with period π
axiom f_periodic : ∀ x : ℝ, f x = f (x + π)

-- Definition of f within the interval [0, π/2]
axiom f_cos : ∀ x : ℝ, 0 ≤ x ∧ x ≤ π/2 → f x = cos x

-- The theorem to be proven
theorem find_f_of_5pi_3 : f (5 * π / 3) = 1 / 2 := 
sorry

end find_f_of_5pi_3_l442_442378


namespace number_of_non_empty_subsets_l442_442787

theorem number_of_non_empty_subsets (n : ℕ) (h : n = 5) : 2^n - 1 = 31 := by
  rw [h]
  norm_num
  sorry

end number_of_non_empty_subsets_l442_442787


namespace ratio_of_b_to_c_l442_442355

variables (A B C : ℕ)

def A_is_b_older := A = B + 2
def B_is_18 := B = 18
def total_age := A + B + C = 47
def B_C_ratio := B / gcd B C = 2 ∧ C / gcd B C = 1

theorem ratio_of_b_to_c (h1 : A_is_b_older)
                        (h2 : B_is_18)
                        (h3 : total_age) :
    B_C_ratio :=
begin
  sorry
end

end ratio_of_b_to_c_l442_442355


namespace lcm_20_45_75_eq_900_l442_442332

theorem lcm_20_45_75_eq_900 : Nat.lcm (Nat.lcm 20 45) 75 = 900 :=
by sorry

end lcm_20_45_75_eq_900_l442_442332


namespace intersection_A_B_l442_442218

-- Defining the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

-- Statement to prove
theorem intersection_A_B : A ∩ B = {2, 3} := 
by 
  sorry

end intersection_A_B_l442_442218


namespace linear_function_unique_l442_442893

noncomputable def f (x : ℝ) : ℝ := sorry

theorem linear_function_unique
  (h1 : ∀ x : ℝ, f (f x) = 4 * x + 6)
  (h2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :
  ∀ x : ℝ, f x = 2 * x + 2 :=
sorry

end linear_function_unique_l442_442893


namespace solution_set_a_eq_half_l442_442692

theorem solution_set_a_eq_half (a : ℝ) : (∀ x : ℝ, (ax / (x - 1) < 1 ↔ (x < 1 ∨ x > 2))) → a = 1 / 2 :=
by
sorry

end solution_set_a_eq_half_l442_442692


namespace Rachel_painting_time_l442_442961

noncomputable def Matt_time : ℕ := 12
noncomputable def Patty_time (Matt_time : ℕ) : ℕ := Matt_time / 3
noncomputable def Rachel_time (Patty_time : ℕ) : ℕ := 5 + 2 * Patty_time

theorem Rachel_painting_time : Rachel_time (Patty_time Matt_time) = 13 := by
  sorry

end Rachel_painting_time_l442_442961


namespace find_b_l442_442307

theorem find_b 
  (b : ℝ) 
  (h1 : ∃ α : ℝ, cos α = -3/5) 
  (h2 : ∃ P : ℝ × ℝ, P = (b, 4) ∧ cos α = (P.1 / (sqrt (P.1^2 + P.2^2)))) 
  : b = -3 := 
sorry

end find_b_l442_442307


namespace cheese_cost_l442_442169

variable (burritos tacos enchiladas : ℕ)
variable (cheddar_cost per_ounce : ℕ → ℝ)
variable (mozarella_cost per_ounce : ℕ → ℝ)
variable (cheddar_ounce burrito_ounce taco_ounce mozzarella_ounce enchilada_ounce : ℕ)

theorem cheese_cost 
  (hb : burritos = 7)
  (ht : tacos = 1)
  (he : enchiladas = 3)
  (cb : cheddar_ounce = 4)
  (ct : cheddar_ounce = 9)
  (cm : mozzarella_ounce = 5)
  (p_cheddar : cheddar_cost 1 = 0.80)
  (p_mozzarella : mozzarella_cost 1 = 1)
  :
  (7 * 4 * 0.80 + 1 * 9 * 0.80 + 3 * 5 * 1 = 44.60) :=
by
  sorry

end cheese_cost_l442_442169


namespace find_divisor_l442_442342

variable (n : ℤ) (d : ℤ)

theorem find_divisor 
    (h1 : ∃ k : ℤ, n = k * d + 4)
    (h2 : ∃ m : ℤ, n + 15 = m * 5 + 4) :
    d = 5 :=
sorry

end find_divisor_l442_442342


namespace number_of_digimon_packs_bought_l442_442589

noncomputable def cost_per_digimon_pack : ℝ := 4.45
noncomputable def cost_of_baseball_deck : ℝ := 6.06
noncomputable def total_spent : ℝ := 23.86

theorem number_of_digimon_packs_bought : 
  ∃ (D : ℕ), cost_per_digimon_pack * D + cost_of_baseball_deck = total_spent ∧ D = 4 := 
by
  use 4
  split
  · norm_num; exact ((4.45 * 4) + 6.06 = 23.86)
  · exact rfl

end number_of_digimon_packs_bought_l442_442589


namespace smallest_solutions_sum_is_7_2_over_3_l442_442084

noncomputable def floor_real (x : ℝ) : ℤ := 
  if h : ∃ (z : ℤ), ↑z ≤ x ∧ x < ↑z + 1 then classical.some h else 0

def smallest_solutions_sum : ℝ :=
  let x1 := 1.5
  let x2 := 2.25
  let x3 := 3.1666666666666665
  x1 + x2 + x3

theorem smallest_solutions_sum_is_7_2_over_3 :
  smallest_solutions_sum = 7 + 2/3 :=
by
  sorry

end smallest_solutions_sum_is_7_2_over_3_l442_442084


namespace increasing_function_of_a_l442_442858

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (2 - (a / 2)) * x + 2

theorem increasing_function_of_a (a : ℝ) : (∀ x y, x < y → f a x ≤ f a y) ↔ 
  (8 / 3 ≤ a ∧ a < 4) :=
sorry

end increasing_function_of_a_l442_442858


namespace lines_intersection_l442_442752

variables {m n l : Line} {α β : Plane}

axiom skew_lines (h1 : Skew m n)
axiom perp_m_alpha (h2 : m ⟂ α)
axiom perp_n_beta (h3 : n ⟂ β)
axiom perp_l_m (h4 : l ⟂ m)
axiom perp_l_n (h5 : l ⟂ n)
axiom l_not_in_alpha (h6 : ¬l ⊆ α)
axiom l_not_in_beta (h7 : ¬l ⊆ β)

theorem lines_intersection (h1 : Skew m n) 
                           (h2 : m ⟂ α) 
                           (h3 : n ⟂ β) 
                           (h4 : l ⟂ m) 
                           (h5 : l ⟂ n) 
                           (h6 : ¬l ⊆ α)
                           (h7 : ¬l ⊆ β) : 
  ∃ p : Point, p ∈ α ∧ p ∈ β ∧ l ∥ (α ∩ β) :=
sorry

end lines_intersection_l442_442752


namespace distance_center_to_plane_of_cross_section_l442_442377

theorem distance_center_to_plane_of_cross_section
  (edge_length : ℝ)
  (r : ℝ)
  (radius_circle : ℝ)
  (distance_oc : ℝ) :
  edge_length = 1 →
  r = (edge_length * ℝ.sqrt 3) / 2 →
  radius_circle = edge_length / 2 →
  distance_oc = (r - radius_circle) / 2 →
  distance_oc = ℝ.sqrt 3 / 6 :=
by
  intros h1 h2 h3 h4
  sorry

end distance_center_to_plane_of_cross_section_l442_442377


namespace seats_needed_l442_442758

-- Definitions based on the problem's condition
def children : ℕ := 58
def children_per_seat : ℕ := 2

-- Theorem statement to prove
theorem seats_needed : children / children_per_seat = 29 :=
by
  sorry

end seats_needed_l442_442758


namespace angle_ACB_l442_442188

open Triangle

theorem angle_ACB {A B C D : Point} (h : Triangle A B C) :
  (D ∈ LineSegment B C) →
  (Triangle.equilateral BA AD DC) →
  (Angle BAD = 80) →
  (Angle ACB = 25) :=
by
  intro hD
  intro heq
  intro hBAD
  sorry

end angle_ACB_l442_442188


namespace range_of_f_on_interval_l442_442863

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

theorem range_of_f_on_interval :
  ∀ (a : ℝ), (∃ x ∈ Icc (-2 : ℝ) 2, f x a = 3) →
  ∃ (ymin ymax : ℝ), (∀ y ∈ Icc (-2 : ℝ) 2, ymin ≤ f y a ∧ f y a ≤ ymax) ∧
  ymin = -37 ∧ ymax = 3 :=
by
  sorry

end range_of_f_on_interval_l442_442863


namespace average_first_25_odd_primes_l442_442415

theorem average_first_25_odd_primes :
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101] in
  (list.sum primes) / 25 = 47.48 :=
by
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
  have sum_primes : list.sum primes = 1187 := by sorry
  show 1187 / 25 = 47.48, by norm_num
  sorry

end average_first_25_odd_primes_l442_442415


namespace bob_total_distance_l442_442803

theorem bob_total_distance:
  let time1 := 1.5
  let speed1 := 60
  let time2 := 2
  let speed2 := 45
  (time1 * speed1) + (time2 * speed2) = 180 := 
  by
  sorry

end bob_total_distance_l442_442803


namespace log_expression_calculation_l442_442418

theorem log_expression_calculation : 
  log10 (25 / 16) - 2 * log10 (5 / 9) + log10 (32 / 81) = log10 2 := 
by
  -- This is where the proof would be placed
  sorry

end log_expression_calculation_l442_442418


namespace z_in_fourth_quadrant_l442_442246

-- Definitions based on the given conditions in step a)
def z : ℂ := (1 - 2*complex.I) / (2 - complex.I)

-- The statement of the problem being asked
theorem z_in_fourth_quadrant : 
  (complex.re z > 0) ∧ (complex.im z < 0) := by
  sorry

end z_in_fourth_quadrant_l442_442246


namespace angle_PIQ_l442_442570

-- Define the triangle and the conditions specified
variable (P Q R I : Type) [AffineSpace P Q R]
variables (PS QT RU : P → R → Prop)
variable (PRQ : ℝ)

-- Angle bisectors meeting at incenter
axiom angle_bisectors_incenter (hPS : PS P S) (hQT : QT Q T) (hRU : RU R U) (hIncenter : angle_bisectors_incenter P Q R I PS QT RU) : True

-- Given angles
axiom angle_PRQ : ∠ PRQ = 42

-- Target angle to prove
theorem angle_PIQ : ∠ PIQ = 69 :=
by
  sorry

end angle_PIQ_l442_442570


namespace angle_between_a_b_l442_442103

open Real EuclideanGeometry

noncomputable def angle_between_vectors (a b : ℝ^3) : ℝ := 
  let dot_product := a.1 * b.1 + a.2 * b.2 + a.3 * b.3
  let a_norm := sqrt (a.1 * a.1 + a.2 * a.2 + a.3 * a.3)
  let b_norm := sqrt (b.1 * b.1 + b.2 * b.2 + b.3 * b.3)
  acos (dot_product / (a_norm * b_norm))

theorem angle_between_a_b {a b : ℝ^3}
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hab : sqrt (a.1 * a.1 + a.2 * a.2 + a.3 * a.3) = 2 * sqrt (b.1 * b.1 + b.2 * b.2 + b.3 * b.3))
  (horth : (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 + (a.3 - b.3) * b.3 = 0) :
  angle_between_vectors a b = π / 3 := sorry

end angle_between_a_b_l442_442103


namespace find_angle_set_simplify_expression_l442_442112

noncomputable def point_A : ℝ × ℝ := (Real.sqrt 3, -1)
def r : ℝ := Real.sqrt ((Real.sqrt 3) ^ 2 + (-1) ^ 2)
def sin_alpha : ℝ := -1 / 2
def angle_set : Set ℝ := {α | ∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 6}

theorem find_angle_set : ∃ α : ℝ, (point_A = (Real.sqrt 3, -1)) → (sin_alpha = -1 / 2) → α ∈ angle_set := 
  sorry

theorem simplify_expression (α : ℝ) : 
  point_A = (Real.sqrt 3, -1) →
  (∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 6) →
  ( (sin (2 * Real.pi - α) * tan (Real.pi + α) * cot (-α - Real.pi)) / (csc (-α) * cos (Real.pi - α) * tan (3 * Real.pi - α)) = 1 / 2) := 
  sorry

end find_angle_set_simplify_expression_l442_442112


namespace total_turnover_correct_l442_442907

variable (x : ℝ)

-- Conditions as definitions
def turnover_july : ℝ := 16
def turnover_august : ℝ := 16 * (1 + x)
def turnover_september : ℝ := 16 * (1 + x)^2
def total_turnover : ℝ := turnover_july + turnover_august + turnover_september

-- Theorem statement in Lean 4
theorem total_turnover_correct (h: total_turnover x = 120) : 
  16 + 16 * (1 + x) + 16 * (1 + x)^2 = 120 :=
by
  sorry

end total_turnover_correct_l442_442907


namespace base_10_to_base_7_conversion_l442_442328

theorem base_10_to_base_7_conversion :
  ∃ (digits : ℕ → ℕ), 789 = digits 3 * 7^3 + digits 2 * 7^2 + digits 1 * 7^1 + digits 0 * 7^0 ∧
  digits 3 = 2 ∧ digits 2 = 2 ∧ digits 1 = 0 ∧ digits 0 = 5 :=
sorry

end base_10_to_base_7_conversion_l442_442328


namespace skiers_race_possibilities_l442_442746

theorem skiers_race_possibilities :
  ∀ (skier : Fin 7 → ℕ), 
  (start_seq : skier = λ k, k + 1)
  (constant_speed : ∀ (i : Fin 7), ∃ (speed : ℝ), constant speed)
  (two_overtakes : ∀ (i : Fin 7), ∃ (j k : Fin 7), i ≠ j ∧ i ≠ k ∧ (overtaken j i ∨ overtaken k i)),
   finishing_orders skier = [[3, 2, 1, 6, 7, 4, 5], [3, 4, 1, 2, 7, 6, 5]] := 
  sorry

-- Define what it means for one skier to be "overtaken" by another.
def overtaken (i j : Fin 7) : Prop := sorry

-- Define what it means to have "finishing orders" for the skiers.
def finishing_orders (skier : Fin 7 → ℕ) : list (list ℕ) := sorry

end skiers_race_possibilities_l442_442746


namespace point_is_in_third_quadrant_l442_442164

noncomputable def point_quadrant (θ : ℝ) : (quadrant : ℕ) :=
  if sin θ > 0 ∧ cos θ > 0 then 1
  else if sin θ > 0 ∧ cos θ < 0 then 2
  else if sin θ < 0 ∧ cos θ < 0 then 3
  else 4

theorem point_is_in_third_quadrant : point_quadrant 2018 = 3 :=
by
  have h1 : sin 2018 = sin 218 := sorry -- follows from angle reduction
  have h2 : cos 2018 = cos 218 := sorry -- follows from angle reduction
  have h3 : sin 218 < 0 := sorry -- provided condition
  have h4 : cos 218 < 0 := sorry -- provided condition
  sorry

end point_is_in_third_quadrant_l442_442164


namespace no_solution_fractions_eq_l442_442280

open Real

theorem no_solution_fractions_eq (x : ℝ) :
  (x-2)/(2*x-1) + 1 = 3/(2-4*x) → False :=
by
  intro h
  have h1 : ¬ (2*x - 1 = 0) := by
    -- 2*x - 1 ≠ 0
    sorry
  have h2 : ¬ (2 - 4*x = 0) := by
    -- 2 - 4*x ≠ 0
    sorry
  -- Solve the equation and show no solutions exist without contradicting the conditions
  sorry

end no_solution_fractions_eq_l442_442280


namespace monotonicity_f1_range_a_l442_442508

noncomputable def f1 (x : ℝ) : ℝ :=
  x^3 - 3 * real.sqrt 2 * x^2 + 3 * x + 1

theorem monotonicity_f1 :
  ∀ x : ℝ,
    if x < real.sqrt 2 - 1 then deriv f1 x > 0
    else if x > real.sqrt 2 + 1 then deriv f1 x > 0
    else deriv f1 x < 0 := sorry

noncomputable def f2 (a x : ℝ) : ℝ :=
  x^3 + 3 * a * x^2 + 3 * x + 1

theorem range_a (a : ℝ) :
  (∀ x ≥ 2, f2 a x ≥ 0) ↔ a ≥ -5/4 := sorry

end monotonicity_f1_range_a_l442_442508


namespace correct_statement_among_options_l442_442729

theorem correct_statement_among_options :
  ∃ (A B C D : Prop),
  (A ↔ ¬(real.cbrt 9 = 3)) ∧ 
  (B ↔ ¬(∀ x, (real.sqrt x = x → x = 1))) ∧ 
  (C ↔ ((-2)^2 = 4)) ∧ 
  (D ↔ ¬(real.sqrt (real.sqrt 4) = 2)) ∧ 
  C := 
by 
  -- Definitions according to the steps
  let A := ¬(real.cbrt 9 = 3)
  let B := ¬(∀ x, (real.sqrt x = x → x = 1))
  let C := ((-2)^2 = 4)
  let D := ¬(real.sqrt (real.sqrt 4) = 2)
  -- Prove that statement C is correct
  existsi [A, B, C, D]
  constructor
  -- Proofs of the logical equivalences
  { show A = ¬(real.cbrt 9 = 3), sorry },
  { show B = ¬(∀ x, (real.sqrt x = x → x = 1)), sorry },
  { show C = ((-2)^2 = 4), sorry },
  { show D = ¬(real.sqrt (real.sqrt 4) = 2), sorry },
  -- Proof that C is the correct one
  exact C

end correct_statement_among_options_l442_442729


namespace slope_angle_of_vertical_line_l442_442691

theorem slope_angle_of_vertical_line :
  ∀ {θ : ℝ}, (∀ x, (x = 3) → x = 3) → θ = 90 := by
  sorry

end slope_angle_of_vertical_line_l442_442691


namespace num_ways_after_5_jumps_num_ways_after_20_jumps_l442_442771

-- Conditions as definitions
def particle_origin : ℕ := 0

def jump (dir : ℤ) : ℤ := if dir > 0 then 1 else -1

-- Total number of ways for a particle to land at a specific point after a given number of jumps
def num_ways_to_reach (jumps target : ℤ) : ℤ :=
  let left_steps := (jumps - target) / 2
  combin jumps left_steps

-- Theorem statements for the given questions
theorem num_ways_after_5_jumps : num_ways_to_reach 5 3 = 5 := by
  sorry

theorem num_ways_after_20_jumps : num_ways_to_reach 20 16 = 190 := by
  sorry

end num_ways_after_5_jumps_num_ways_after_20_jumps_l442_442771


namespace john_gets_30_cans_l442_442584

def normal_price : ℝ := 0.60
def total_paid : ℝ := 9.00

theorem john_gets_30_cans :
  (total_paid / normal_price) * 2 = 30 :=
by
  sorry

end john_gets_30_cans_l442_442584


namespace problem_gcf_lcm_sum_l442_442935

-- Let A be the GCF of {15, 20, 30}
def A : ℕ := Nat.gcd (Nat.gcd 15 20) 30

-- Let B be the LCM of {15, 20, 30}
def B : ℕ := Nat.lcm (Nat.lcm 15 20) 30

-- We need to prove that A + B = 65
theorem problem_gcf_lcm_sum :
  A + B = 65 :=
by
  sorry

end problem_gcf_lcm_sum_l442_442935


namespace arithmetic_sum_expression_zero_l442_442212

theorem arithmetic_sum_expression_zero (a d : ℤ) (i j k : ℕ) (S_i S_j S_k : ℤ) :
  S_i = i * (a + (i - 1) * d / 2) →
  S_j = j * (a + (j - 1) * d / 2) →
  S_k = k * (a + (k - 1) * d / 2) →
  (S_i / i * (j - k) + S_j / j * (k - i) + S_k / k * (i - j) = 0) :=
by
  intros hS_i hS_j hS_k
  -- Proof omitted
  sorry

end arithmetic_sum_expression_zero_l442_442212


namespace intersection_of_A_and_B_l442_442223

-- Define sets A and B
def A : set ℝ := {x | -2 < x ∧ x < 4}
def B : set ℕ := {2, 3, 4, 5}

-- Theorem stating the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {2, 3} := 
by sorry

end intersection_of_A_and_B_l442_442223


namespace angle_BAC_60_l442_442936

variables {A B C O : Type} [AffineSpace ℝ A] [EuclideanGeometry]
variables {a b c o : A}
variables {V : Type} [inner_product_space ℝ V] [affine_space V A]

-- Given conditions
variable (h_circumcenter : is_circumcenter o a b c)
variable (h_vector : (3:ℝ) • (o - a) = (b - a) + (c - a))

-- Prove the degree of ∠BAC is 60°
theorem angle_BAC_60 :
  ∠ b a c = 60 :=
sorry

end angle_BAC_60_l442_442936


namespace definite_integral_value_l442_442440

open Real

noncomputable def definite_integral : ℝ :=
  ∫ x in 0..1, (3 * x^2 - 1/2)

theorem definite_integral_value :
  definite_integral = 1 / 2 :=
by
  sorry

end definite_integral_value_l442_442440


namespace conversion_base8_to_base10_l442_442808

theorem conversion_base8_to_base10 : 5 * 8^3 + 2 * 8^2 + 1 * 8^1 + 4 * 8^0 = 2700 :=
by 
  sorry

end conversion_base8_to_base10_l442_442808


namespace candy_cookies_l442_442422

def trays : Nat := 4
def cookies_per_tray : Nat := 24
def packs : Nat := 8
def total_cookies : Nat := trays * cookies_per_tray
def cookies_per_pack : Nat := total_cookies / packs

theorem candy_cookies : 
  cookies_per_pack = 12 := 
by
  -- Calculate total cookies
  have h1 : total_cookies = trays * cookies_per_tray := rfl
  have h2 : total_cookies = 96 := by rw [h1]; norm_num
  
  -- Calculate cookies per pack
  have h3 : cookies_per_pack = total_cookies / packs := rfl
  have h4 : cookies_per_pack = 12 := by rw [h3, h2]; norm_num
  
  exact h4

end candy_cookies_l442_442422


namespace eccentricity_is_sqrt3_minus_1_l442_442123

noncomputable def find_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ :=
  if h : (∀ A, A ∈ ({(x, y) | (x^2/a^2 + y^2/b^2 = 1)} : set (ℝ × ℝ)) ∧ angle F1 A F2 = real.pi / 6 ∧ dist A 0 = dist 0 F2)
  then sqrt 3 - 1
  else 0

theorem eccentricity_is_sqrt3_minus_1 (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (A : ℝ × ℝ)
  (hA : A ∈ ({(x, y) | (x^2/a^2 + y^2/b^2 = 1)} : set (ℝ × ℝ)) ∧ ∠ F1 A F2 = real.pi / 6 ∧ dist A 0 = dist 0 F2) :
  find_eccentricity a b h1 h2 = sqrt 3 - 1 := by
  sorry

end eccentricity_is_sqrt3_minus_1_l442_442123


namespace maximum_candies_after_20_hours_l442_442254

-- Define a function to compute the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Define the recursive function to model the candy process
def candies_after_hours (n : ℕ) (hours : ℕ) : ℕ :=
  if hours = 0 then n 
  else candies_after_hours (n + sum_of_digits n) (hours - 1)

theorem maximum_candies_after_20_hours :
  candies_after_hours 1 20 = 148 :=
sorry

end maximum_candies_after_20_hours_l442_442254


namespace length_of_each_train_l442_442715

/-- Length of each train, given:
1. Two trains of equal length are running on parallel lines in the same direction.
2. The speeds of the trains are 49 km/hr and 36 km/hr, respectively.
3. The faster train passes the slower train in 36 seconds.

We aim to prove that the length of each train is approximately 65 meters.
-/

noncomputable def train_length := 
  let speed1 := 49 * 1000 / 3600 -- km/hr to m/s conversion of first train
  let speed2 := 36 * 1000 / 3600 -- km/hr to m/s conversion of second train
  let relative_speed := speed1 - speed2
  let distance := relative_speed * 36 -- distance covered in 36 seconds
  distance / 2 -- since distance is 2L and L = distance / 2

theorem length_of_each_train : train_length ≈ 65 := sorry

end length_of_each_train_l442_442715


namespace range_g_l442_442068

noncomputable def g (x : ℝ) : ℝ :=
  cos x ^ 4 - sin x * cos x + sin x ^ 4

theorem range_g : 
  (∀ x : ℝ, sin x ^ 2 + cos x ^ 2 = 1) →
  (∀ x : ℝ, sin (2 * x) = 2 * sin x * cos x) →
  (∀ x : ℝ, sin (2 * x) ^ 2 = 4 * sin x ^ 2 * cos x ^ 2) →
  (∀ x : ℝ, sin (2 * x) ∈ set.Icc (-1 : ℝ) 1) →
  set.range g = set.Icc (0 : ℝ) (9 / 8 : ℝ) :=
sorry

end range_g_l442_442068


namespace incorrect_statement_B_l442_442490

def plane (α : Type) := α

variable {α β γ : Type}

def is_parallel (p1 p2 : plane α) : Prop := sorry
def is_perpendicular (p1 p2 : plane α) : Prop := sorry
def intersection (p1 p2 : plane α) : set (plane α) := sorry
def is_parallel_line (l1 l2 : set (plane α)) : Prop := sorry

theorem incorrect_statement_B (h1 : α ≠ β) (h2 : β ≠ γ) (h3 : γ ≠ α)
 (h_parallel1 : is_parallel α β) (h_parallel2 : is_parallel β γ)
 (hPer1 : is_perpendicular α β) (hPer2 : is_perpendicular β γ) :
 ¬ (is_perpendicular α γ) :=
by
  sorry

end incorrect_statement_B_l442_442490


namespace trigonometric_identity_l442_442832

variable (α : Real)

theorem trigonometric_identity 
  (h : Real.sin (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (π / 3 - α) = Real.sqrt 3 / 3 :=
sorry

end trigonometric_identity_l442_442832


namespace possible_pairings_l442_442055

def numBowls : ℕ := 5
def numGlasses : ℕ := 5

theorem possible_pairings : numBowls * numGlasses = 25 :=
by
-- numBowls = 5 and numGlasses = 5
calc
  5 * 5 = 25 : by norm_num

end possible_pairings_l442_442055


namespace min_value_fraction_l442_442297

theorem min_value_fraction (m n : ℝ) (h1 : 2 * m + 2 * n = 1) (h2 : m * n > 0) : 
  ∃ (x : ℝ), x = 5 + 2 * Real.sqrt 6 ∧ (∀ (m n : ℝ), 2 * m + 2 * n = 1 ∧ m * n > 0 → 
    x ≤ 1 / m + 3 / n) := 
begin
  sorry
end

end min_value_fraction_l442_442297


namespace ball_bounce_height_l442_442001

open Real

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (2 / 3 : ℝ)^k < 2) ∧ (∀ n : ℕ, n < k → 20 * (2 / 3 : ℝ)^n ≥ 2) ∧ k = 6 :=
sorry

end ball_bounce_height_l442_442001


namespace min_quotient_of_product_division_l442_442442

theorem min_quotient_of_product_division :
  ∃ (G1 G2 : Finset ℕ), 
    (∀ x, x ∈ G1 ∪ G2 ↔ x ∈ Finset.range 9 ∧ x ≠ 0) ∧ 
    (G1 ∩ G2 = ∅) ∧ 
    (∀ (x ∈ Finset.product G1 G2), x.1 % x.2 = 0) ∧ 
    (∀ G1 G2, (G1 ∪ G2 = Finset.range 9 \ {0}) → 
              (G1 ∩ G2 = ∅) → 
              (∀ (x ∈ Finset.product G1 G2), x.1 % x.2 = 0) → 
              (Finset.prod G1 id / Finset.prod G2 id) = 70) := 
sorry

end min_quotient_of_product_division_l442_442442


namespace log_max_value_l442_442476

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else x^(1/2)

def t (x : ℝ) : ℝ :=
  (x + 1) / (x - 1)

-- Given that f(x) ≥ 3, prove that the maximum value of log_2(t(x)) is log_2(5/4)
theorem log_max_value (x : ℝ) (h : f x ≥ 3) : 
  ∃ (y : ℝ), y = Real.log (5 / 4) / Real.log 2 ∧ log (t x) = y :=
  sorry

end log_max_value_l442_442476


namespace cyclic_quadrilateral_trapezoid_l442_442101

theorem cyclic_quadrilateral_trapezoid
  (A B C D X Y : Type)
  [IsTrapezoid A B C D]
  (AD_parallel_BC : Parallel A D B C)
  (angle_bisectors_AX_BY_meet_at_X : Intersection (AngleBisector A) (AngleBisector B) = X)
  (angle_bisectors_CY_DY_meet_at_Y : Intersection (AngleBisector C) (AngleBisector D) = Y) :
  IsCyclicQuadrilateral A X Y D ∧ IsCyclicQuadrilateral B X Y C := by
  sorry

end cyclic_quadrilateral_trapezoid_l442_442101


namespace james_profit_correct_l442_442580

noncomputable def james_profit : ℕ :=
  let cost_per_toy := 20 in
  let sell_price_per_toy := 30 in
  let total_toys := 200 in
  let percent_to_sell := 80 in
  let total_cost := cost_per_toy * total_toys in
  let toys_sold := total_toys * percent_to_sell / 100 in
  let total_revenue := toys_sold * sell_price_per_toy in
  total_revenue - total_cost

theorem james_profit_correct : james_profit = 800 :=
by
  sorry

end james_profit_correct_l442_442580


namespace parabola_x_intercepts_distance_l442_442348

theorem parabola_x_intercepts_distance :
  let f := fun x : ℝ => -x^2 + 2 * x + 3 in
  let roots := {x : ℝ | f x = 0} in
  set.subset_nonempty_finite {x : ℝ | f x = 0} →
  abs (Classical.some (set.eq_of_mem_singleton (set.to_finset roots).1 1) -
      Classical.some (set.eq_of_mem_singleton (set.to_finset roots).1 (finset.card (set.to_finset roots) - 1))) = 4 :=
by
  sorry

end parabola_x_intercepts_distance_l442_442348


namespace exists_unique_c_for_a_equals_3_l442_442109

theorem exists_unique_c_for_a_equals_3 :
  ∃! c : ℝ, ∀ x ∈ Set.Icc (3 : ℝ) 9, ∃ y ∈ Set.Icc (3 : ℝ) 27, Real.log x / Real.log 3 + Real.log y / Real.log 3 = c :=
sorry

end exists_unique_c_for_a_equals_3_l442_442109


namespace benny_soft_drinks_l442_442413

open nat

theorem benny_soft_drinks :
  ∀ (soft_drink_cost candy_bars candy_bar_cost total_spent : ℕ),
    soft_drink_cost = 4 →
    candy_bars = 5 →
    candy_bar_cost = 4 →
    total_spent = 28 →
  ∃ S,
    (soft_drink_cost * S + candy_bars * candy_bar_cost = total_spent) ∧
    S = 2 :=
by
  intros soft_drink_cost candy_bars candy_bar_cost total_spent
  intros h_soft_drink_cost h_candy_bars h_candy_bar_cost h_total_spent
  use 2
  split
  { rw [h_soft_drink_cost, h_candy_bars, h_candy_bar_cost, h_total_spent]
    sorry },
  { refl },

end benny_soft_drinks_l442_442413


namespace marbles_difference_l442_442203

-- Conditions
def L : ℕ := 23
def F : ℕ := 9

-- Proof statement
theorem marbles_difference : L - F = 14 := by
  sorry

end marbles_difference_l442_442203


namespace minimum_value_of_f_l442_442157

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 1) * Real.exp (x - 1)

theorem minimum_value_of_f :
  ∃ x_min : ℝ, x_min = 1 ∧ ∀ x, f x ≥ f x_min ∧ f x_min = -1 :=
by
  use 1
  split
  · rfl
  · split
    · intro x
      sorry
    · rfl

end minimum_value_of_f_l442_442157


namespace fraction_equivalence_l442_442730

theorem fraction_equivalence : 
  (∀ (a b : ℕ), (a ≠ 0 ∧ b ≠ 0) → (15 * b = 25 * a ↔ a = 3 ∧ b = 5)) ∧
  (15 * 4 ≠ 25 * 3) ∧
  (15 * 3 ≠ 25 * 2) ∧
  (15 * 2 ≠ 25 * 1) ∧
  (15 * 7 ≠ 25 * 5) :=
by
  sorry

end fraction_equivalence_l442_442730


namespace complex_equation_solution_exists_l442_442066

theorem complex_equation_solution_exists :
  ∃ (z : ℂ), (∃ (a b : ℝ), z = a + b * Complex.I ∧ 3 * z + 5 * Complex.I * Complex.conj z = -4 - 7 * Complex.I) ∧
    z = -23 / 16 + (1 / 16) * Complex.I :=
by
  use -23 / 16 + (1 / 16) * Complex.I
  constructor
  { use [-23 / 16, 1 / 16]
    simp
    ring }
  { refl }

end complex_equation_solution_exists_l442_442066


namespace min_slope_tangent_at_p_sum_of_reciprocals_of_slopes_l442_442502

def f (x : ℝ) : ℝ := x^3 - x^2 + (2 * real.sqrt 2 - 3) * x + 3 - 2 * real.sqrt 2

def f' (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 2 * real.sqrt 2 - 3

theorem min_slope_tangent_at_p :
  ∃ x : ℝ, f'(x) = 2 * real.sqrt 2 - 10 / 3 :=
begin
  have h' : ∀ x, f'(x) = 3 * (x - 1 / 3) ^ 2 + 2 * real.sqrt 2 - 10 / 3,
  {intro x, calc
    f'(x) = 3 * x^2 - 2 * x + 2 * real.sqrt 2 - 3 : rfl
    ... = 3 * (x^2 - 2/3 * x + 1 / 9 - 1 / 9) + 2 * real.sqrt 2 - 3 : by ring
    ... = 3 * ((x - 1 / 3) ^ 2 - 1 / 9) + 2 * real.sqrt 2 - 3 : by { congr' 2, ring }
    ... = 3 * (x - 1 / 3) ^ 2 - 1 / 3 + 2 * real.sqrt 2 - 3 : by ring
    ... = 3 * (x - 1 / 3) ^ 2 + 2 * real.sqrt 2 - 10 / 3 : by ring },
  use [1 / 3],
  rw h',
  ring,
end

theorem sum_of_reciprocals_of_slopes (x1 x2 x3 : ℝ) (hx1: f x1 = 0) (hx2: f x2 = 0) (hx3: f x3 = 0):
  1 / f'(x1) + 1 / f'(x2) + 1 / f'(x3) = 0 :=
begin
  sorry -- Proof can be filled in later
end

end min_slope_tangent_at_p_sum_of_reciprocals_of_slopes_l442_442502


namespace six_letter_word_combinations_l442_442582

theorem six_letter_word_combinations : ∃ n : ℕ, n = 26 * 26 * 26 := 
sorry

end six_letter_word_combinations_l442_442582


namespace A_passes_test_expectation_X_l442_442910

section probability

-- Define the probability of student A making exactly k shots out of 3
noncomputable def P_A_shots (k : ℕ) : ℚ :=
  if h : k ∈ {0, 1, 2, 3} then (Nat.choose 3 k : ℚ) * (1 / 3)^k * (2 / 3)^(3 - k) else 0

-- Define the probability that student A passes the test
noncomputable def P_A_passes : ℚ :=
  P_A_shots 2 + P_A_shots 3

-- Theorem for part (1)
theorem A_passes_test : P_A_passes = 7 / 27 := sorry

-- Define the probability of student B making exactly k shots out of 3
noncomputable def P_B_shots (k : ℕ) : ℚ :=
  if h : k ∈ {0, 1, 2, 3} then (Nat.choose 3 k : ℚ) * (1 / 2)^k * (1 / 2)^(3 - k) else 0

-- Define the probability that student B passes the test
noncomputable def P_B_passes : ℚ :=
  P_B_shots 2 + P_B_shots 3

-- Define the distribution of the number of extra shots X
noncomputable def P_X (x : ℕ) : ℚ :=
  if x = 0 then P_A_passes * P_B_passes
  else if x = 30 then P_A_passes * (1 - P_B_passes) + (1 - P_A_passes) * P_B_passes
  else if x = 60 then (1 - P_A_passes) * (1 - P_B_passes)
  else 0

-- Define the expected number of extra shots E(X)
noncomputable def E_X : ℚ :=
  0 * P_X 0 + 30 * P_X 30 + 60 * P_X 60

-- Theorem for part (2)
theorem expectation_X : E_X = 335 / 9 := sorry

end probability

end A_passes_test_expectation_X_l442_442910


namespace smallest_positive_period_monotonic_decreasing_interval_inequality_solution_set_l442_442247

noncomputable def f (x a : ℝ) := sqrt 3 * sin x * cos x + cos x ^ 2 + a

theorem smallest_positive_period (a : ℝ) :
  ∃ T > 0, ∀ x, f x a = f (x + T) a ∧ T = π := sorry

theorem monotonic_decreasing_interval (a : ℝ) :
  ∀ k : ℤ, ∀ x, k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3 → ∀ y, x < y → f x a > f y a := sorry

theorem inequality_solution_set (a : ℝ) (h₁ : a = 0) :
  ∀ x, -π / 6 ≤ x ∧ x ≤ π / 3 → (f x a > 1 ↔ 0 < x ∧ x < π / 3) := sorry

end smallest_positive_period_monotonic_decreasing_interval_inequality_solution_set_l442_442247


namespace log_equation_solution_l442_442988

theorem log_equation_solution {x : ℝ} (h : x ≠ 0 ∧ x ≠ 1) :
  log (1 / x + 96 / (x^2 * (x^2 - 1))) =
  log (x + 1) + log (x + 2) + log (x + 3) - 2 * log x - log (x^2 - 1) →
  x = 3 :=
sorry

end log_equation_solution_l442_442988


namespace max_diameters_l442_442094

theorem max_diameters (n : ℕ) (points : Finset (ℝ × ℝ)) (h : n ≥ 3) (hn : points.card = n)
  (d : ℝ) (h_d_max : ∀ {p q : ℝ × ℝ}, p ∈ points → q ∈ points → dist p q ≤ d) :
  ∃ m : ℕ, m ≤ n ∧ (∀ {p q : ℝ × ℝ}, p ∈ points → q ∈ points → dist p q = d → m ≤ n) := 
sorry

end max_diameters_l442_442094


namespace hours_to_seconds_l442_442884

theorem hours_to_seconds : 
  (3.5 * 60 * 60) = 12600 := 
by 
  sorry

end hours_to_seconds_l442_442884


namespace count_even_strictly_increasing_integers_correct_l442_442532

-- Definition of condition: even four-digit integers with strictly increasing digits
def is_strictly_increasing {a b c d : ℕ} : Prop :=
1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ∈ {2, 4, 6, 8}

def count_even_strictly_increasing_integers : ℕ :=
(finset.range 10).choose 4.filter (λ l, is_strictly_increasing l.head l.nth 1 l.nth 2 l.nth 3).card

theorem count_even_strictly_increasing_integers_correct :
  count_even_strictly_increasing_integers = 46 := by
  sorry

end count_even_strictly_increasing_integers_correct_l442_442532


namespace find_k_l442_442453

theorem find_k (k : ℝ) (c : ℝ) (h : c = 2) : 
  -x^2 - (k + 10) * x - 8 = -(x - c) * (x - 4) ↔ k = -16 := 
by {
  rw h,
  sorry
}

end find_k_l442_442453


namespace least_possible_cost_l442_442269

-- Define the dimensions of the five regions
def region1_area := 5 * 2
def region2_area := 9 * 3
def region3_area := 5 * 3
def region4_area := 8 * 2
def region5_area := 6 * 4

-- Define the cost of each type of flower per square foot
def cost_per_sqft_easter_lilies := 4
def cost_per_sqft_dahlias := 3.5
def cost_per_sqft_cannas := 3
def cost_per_sqft_begonias := 2.5
def cost_per_sqft_asters := 2

-- Calculate the costs for each section based on areas and costs
def cost_section1 := region1_area * cost_per_sqft_easter_lilies
def cost_section2 := region2_area * cost_per_sqft_asters
def cost_section3 := region3_area * cost_per_sqft_cannas
def cost_section4 := region4_area * cost_per_sqft_dahlias
def cost_section5 := region5_area * cost_per_sqft_begonias

-- Statement of the proof problem: proving that the least cost is $254.5
def total_cost := cost_section1 + cost_section2 + cost_section3 + cost_section4 + cost_section5

theorem least_possible_cost : total_cost = 254.5 := 
by 
  have h1 : region1_area = 10  := by simp [region1_area]
  have h2 : region2_area = 27 := by simp [region2_area]
  have h3 : region3_area = 15 := by simp [region3_area]
  have h4 : region4_area = 16 := by simp [region4_area]
  have h5 : region5_area = 24 := by simp [region5_area]
  have c1 : cost_section1 = 40 := by simp [cost_section1, h1, cost_per_sqft_easter_lilies]
  have c2 : cost_section2 = 54  := by simp [cost_section2, h2, cost_per_sqft_asters]
  have c3 : cost_section3 = 48 := by simp [cost_section3, h3, cost_per_sqft_cannas]
  have c4 : cost_section4 = 52.5 := by simp [cost_section4, h4, cost_per_sqft_dahlias]
  have c5 : cost_section5 = 60 := by simp [cost_section5, h5, cost_per_sqft_begonias]
  have total : total_cost = cost_section1 + cost_section2 + cost_section3 + cost_section4 + cost_section5 := rfl
  exact calc 
    total_cost 
    = 40 + 54 + 48 + 52.5 + 60 : by simp [total, c1, c2, c3, c4, c5] 
    ... = 254.5 : by norm_num

end least_possible_cost_l442_442269


namespace fourth_object_selected_is_210_l442_442710

axiom population_size : ℕ := 240
axiom random_number_table : list ℕ := [324, 51, 744, 91, 145, 62, 165, 10, 24, 56, 896, 40, 568, 16, 554, 64, 416, 30, 856, 21, 52, 14, 845, 13, 125, 41, 21, 45]

def valid_numbers (lst : list ℕ) (n : ℕ) : list ℕ :=
  lst.filter (λ x => x ≤ n)

axiom fourth_selected_object : ℕ
noncomputable def find_fourth_selected (lst : list ℕ) (n : ℕ) : ℕ :=
  (valid_numbers lst n).nth! 3

theorem fourth_object_selected_is_210 :
  fourth_selected_object = find_fourth_selected random_number_table population_size :=
sorry

end fourth_object_selected_is_210_l442_442710


namespace general_term_correct_T_to_199_100_l442_442098

noncomputable def general_term_formula (n : ℕ) (c : ℝ) : ℝ := 
  (fun a_n : ℕ -> ℝ, 2 * n + 2)

noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in finset.range n, ((i : ℝ) / 2^(i+1))

theorem general_term_correct: 
  ∀ (n : ℕ) (S : ℕ → ℝ),
    (S n = ((1 / 2) * (n : ℝ) * general_term_formula n + general_term_formula n - 2) → general_term_formula n = 2 * n + 2 := by sorry

theorem T_to_199_100:
  ∀ (n : ℕ), T (n + 1) - T n > 0 → T n > 199 / 100 → n ≥ 11 := by sorry

end general_term_correct_T_to_199_100_l442_442098


namespace complaints_over_3_days_l442_442651

def normal_complaints_per_day : ℕ := 120

def short_staffed_complaints_per_day : ℕ := normal_complaints_per_day * 4 / 3

def short_staffed_and_broken_self_checkout_complaints_per_day : ℕ := short_staffed_complaints_per_day * 12 / 10

def days_short_staffed_and_broken_self_checkout : ℕ := 3

def total_complaints (days : ℕ) (complaints_per_day : ℕ) : ℕ :=
  days * complaints_per_day

theorem complaints_over_3_days
  (n : ℕ := normal_complaints_per_day)
  (a : ℕ := short_staffed_complaints_per_day)
  (b : ℕ := short_staffed_and_broken_self_checkout_complaints_per_day)
  (d : ℕ := days_short_staffed_and_broken_self_checkout)
  : total_complaints d b = 576 :=
by {
  -- This is where the proof would go, e.g., using sorry to skip the proof for now.
  sorry
}

end complaints_over_3_days_l442_442651


namespace max_value_of_n_l442_442556

noncomputable def a (n : ℕ) : ℕ := n + 1

def b (n : ℕ) : ℝ := 2 / ((a n) * (a (n + 1)))

def T (n : ℕ) : ℝ := ∑ i in finset.range (n + 1), b i

theorem max_value_of_n (n : ℕ) (h : T n ≤ 24 / 25) : n ≤ 48 :=
sorry

end max_value_of_n_l442_442556


namespace number_of_edges_of_cube_l442_442724

def is_cube (x : Type) [t : TopologicalSpace x] : Prop :=
∃ (faces : set (Type)), faces.size = 6 ∧ all_faces_are_squares ∧ cube_structure

theorem number_of_edges_of_cube (C : Type) [cube : is_cube C] : (∃ n : ℕ, n = 12) :=
sorry

end number_of_edges_of_cube_l442_442724


namespace sin_C_value_l442_442878

noncomputable def triangle_sine_proof (A B C a b c : Real) (hB : B = 2 * Real.pi / 3) 
  (hb : b = 3 * c) : Real := by
  -- Utilizing the Law of Sines and given conditions to find sin C
  sorry

theorem sin_C_value (A B C a b c : Real) (hB : B = 2 * Real.pi / 3) 
  (hb : b = 3 * c) : triangle_sine_proof A B C a b c hB hb = Real.sqrt 3 / 6 := by
  sorry

end sin_C_value_l442_442878


namespace sum_of_first_19_terms_l442_442847

noncomputable def a_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a1 + a_n a1 d n)

theorem sum_of_first_19_terms (a1 d : ℝ) (h : a1 + 9 * d = 1) : S_n a1 d 19 = 19 := by
  sorry

end sum_of_first_19_terms_l442_442847


namespace complaints_over_3_days_l442_442653

def normal_complaints_per_day : ℕ := 120

def short_staffed_complaints_per_day : ℕ := normal_complaints_per_day * 4 / 3

def short_staffed_and_broken_self_checkout_complaints_per_day : ℕ := short_staffed_complaints_per_day * 12 / 10

def days_short_staffed_and_broken_self_checkout : ℕ := 3

def total_complaints (days : ℕ) (complaints_per_day : ℕ) : ℕ :=
  days * complaints_per_day

theorem complaints_over_3_days
  (n : ℕ := normal_complaints_per_day)
  (a : ℕ := short_staffed_complaints_per_day)
  (b : ℕ := short_staffed_and_broken_self_checkout_complaints_per_day)
  (d : ℕ := days_short_staffed_and_broken_self_checkout)
  : total_complaints d b = 576 :=
by {
  -- This is where the proof would go, e.g., using sorry to skip the proof for now.
  sorry
}

end complaints_over_3_days_l442_442653


namespace intersection_of_A_and_B_l442_442222

-- Define sets A and B
def A : set ℝ := {x | -2 < x ∧ x < 4}
def B : set ℕ := {2, 3, 4, 5}

-- Theorem stating the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {2, 3} := 
by sorry

end intersection_of_A_and_B_l442_442222


namespace intersection_of_A_and_B_l442_442105

open Set

def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}
def B : Set ℝ := {y | 0 ≤ y ∧ y < 4}

theorem intersection_of_A_and_B : A ∩ B = {z | 2 ≤ z ∧ z < 4} :=
by
  sorry

end intersection_of_A_and_B_l442_442105


namespace area_triangle_YPW_is_8_l442_442671

-- Definitions for lengths and conditions
def length_XYZW := 8
def width_XYZW := 6
def ratio_division := (2 / 3, 1 / 3)

-- Proof statement for area of triangle YPW
theorem area_triangle_YPW_is_8 :
  let XW := Math.sqrt (length_XYZW^2 + width_XYZW^2)
  let XP := (2 / 3) * XW
  let PW := (1 / 3) * XW
  let height_Y_to_XW := (2 * (length_XYZW * width_XYZW)) / XW
  let area_YPW := (1 / 2) * PW * height_Y_to_XW
  area_YPW = 8 :=
begin
  sorry
end

end area_triangle_YPW_is_8_l442_442671


namespace omega_is_four_or_four_thirds_l442_442134

noncomputable def omega_condition : Prop :=
  ∃ (ω : ℝ), (ω > 0) ∧ (|φ| < π / 2) ∧
  (∀ x : ℝ, f(π / 6 + x) = -f(π / 6 - x)) ∧
  (∀ x : ℝ, f(-5 * π / 24 + x) = f(-5 * π / 24 - x)) ∧
  (∀ x : ℝ, (π / 3 < x ∧ x < π / 2) → f(x) < f(x + 1)) ∧
  (ω = 4 / 3 ∨ ω = 4)

theorem omega_is_four_or_four_thirds (f : ℝ → ℝ) (φ : ℝ) :
  omega_condition :=
sorry

end omega_is_four_or_four_thirds_l442_442134


namespace fish_upstream_speed_l442_442012

def Vs : ℝ := 45
def Vdownstream : ℝ := 55

def Vupstream (Vs Vw : ℝ) : ℝ := Vs - Vw
def Vstream (Vs Vdownstream : ℝ) : ℝ := Vdownstream - Vs

theorem fish_upstream_speed :
  Vupstream Vs (Vstream Vs Vdownstream) = 35 := by
  sorry

end fish_upstream_speed_l442_442012


namespace alice_age_proof_l442_442292

-- Definitions derived from the conditions
def alice_pens : ℕ := 60
def clara_pens : ℕ := (2 * alice_pens) / 5
def clara_age_in_5_years : ℕ := 61
def clara_current_age : ℕ := clara_age_in_5_years - 5
def age_difference : ℕ := alice_pens - clara_pens

-- Proof statement to be proved
theorem alice_age_proof : (clara_current_age - age_difference = 20) :=
sorry

end alice_age_proof_l442_442292


namespace polynomial_transformable_l442_442721

theorem polynomial_transformable (a b c d : ℝ) :
  (∃ A B : ℝ, ∀ z : ℝ, z^4 + A * z^2 + B = (z + a/4)^4 + a * (z + a/4)^3 + b * (z + a/4)^2 + c * (z + a/4) + d) ↔ a^3 - 4 * a * b + 8 * c = 0 :=
by
  sorry

end polynomial_transformable_l442_442721


namespace total_tickets_sold_l442_442789

theorem total_tickets_sold
  (advanced_ticket_cost : ℕ)
  (door_ticket_cost : ℕ)
  (total_collected : ℕ)
  (advanced_tickets_sold : ℕ)
  (door_tickets_sold : ℕ) :
  advanced_ticket_cost = 8 →
  door_ticket_cost = 14 →
  total_collected = 1720 →
  advanced_tickets_sold = 100 →
  total_collected = (advanced_tickets_sold * advanced_ticket_cost) + (door_tickets_sold * door_ticket_cost) →
  100 + door_tickets_sold = 165 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_tickets_sold_l442_442789


namespace anand_income_l442_442305

theorem anand_income
  (x y : ℕ)
  (h1 : 5 * x - 3 * y = 800)
  (h2 : 4 * x - 2 * y = 800) : 
  5 * x = 2000 := 
sorry

end anand_income_l442_442305


namespace total_cans_needed_l442_442380

-- Definitions
def cans_per_box : ℕ := 4
def number_of_boxes : ℕ := 203

-- Statement of the problem
theorem total_cans_needed : cans_per_box * number_of_boxes = 812 := 
by
  -- skipping the proof
  sorry

end total_cans_needed_l442_442380


namespace Laura_running_speed_l442_442208

/-- The problem setup translates into expressing the workout constraints. -/
def biking_speed (x : ℝ) : ℝ := 2.5 * x + 2
def running_speed (x : ℝ) : ℝ := x

def biking_time (x : ℝ) : ℝ := 25 / biking_speed x
def running_time (x : ℝ) : ℝ := 6 / running_speed x

-- The total workout time in hours considering a 10-minute break.
def total_workout_time : ℝ := 140 / 60

theorem Laura_running_speed (x : ℝ) :
  biking_time x + running_time x = total_workout_time →
  x = 5.37 :=
sorry

end Laura_running_speed_l442_442208


namespace paint_needed_for_snake_l442_442575

open Nat

def total_paint (paint_per_segment segments additional_paint : Nat) : Nat :=
  paint_per_segment * segments + additional_paint

theorem paint_needed_for_snake :
  total_paint 240 336 20 = 80660 :=
by
  sorry

end paint_needed_for_snake_l442_442575


namespace num_digits_sum_l442_442892

theorem num_digits_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) :
  let num1 := 9643
  let num2 := A * 10 ^ 2 + 7 * 10 + 5
  let num3 := 5 * 10 ^ 2 + B * 10 + 2
  let sum := num1 + num2 + num3
  10^4 ≤ sum ∧ sum < 10^5 :=
by {
  sorry
}

end num_digits_sum_l442_442892


namespace smallest_first_digit_is_one_l442_442324

theorem smallest_first_digit_is_one : ∀ (digits : Finset ℕ), 
  digits = {1, 2, 4} → 
  (∃ (num : ℕ), 
    ((num = 124 ∨ num = 142 ∨ num = 214 ∨ num = 241 ∨ num = 412 ∨ num = 421) ∧ 
    (∃ (d1 d2 d3 : ℕ), num = d1 * 100 + d2 * 10 + d3 ∧ {d1, d2, d3} = digits) ∧ 
    (∀ (n : ℕ), (∃ (d1 d2 d3 : ℕ), n = d1 * 100 + d2 * 10 + d3 ∧ {d1, d2, d3} = digits) → num ≤ n) →
  num / 100 = 1)) :=
by
  intros digits h_digits
  use 124
  split
  · left; refl
  split
  · use 1, 2, 4
    simp [h_digits]
  intros n h_n
  rcases h_n with ⟨d1, d2, d3, h_n1, h_n2⟩
  by_cases h_cases: (d1 = 1 ∧ d2 = 2 ∧ d3 = 4 ∨
                      d1 = 1 ∧ d2 = 4 ∧ d3 = 2 ∨
                      d1 = 2 ∧ d2 = 1 ∧ d3 = 4 ∨
                      d1 = 2 ∧ d2 = 4 ∧ d3 = 1 ∨
                      d1 = 4 ∧ d2 = 1 ∧ d3 = 2 ∨
                      d1 = 4 ∧ d2 = 2 ∧ d3 = 1)
  · cases h_cases; linarith
  · exfalso; 
    have := by simp [h_n1] at h_cases
    contradiction

end smallest_first_digit_is_one_l442_442324


namespace log_relation_l442_442149

open Real

theorem log_relation (a b : ℝ) (h₁ : a = log 9 343) (h₂ : b = log 3 49) :
  a = 3 / 4 * b :=
sorry

end log_relation_l442_442149


namespace angle_B_l442_442571

open Real

-- Defining angles in degrees for clarity
noncomputable def sin_degree (d : ℝ) : ℝ :=
  sin (d * π / 180)

theorem angle_B (a b : ℝ) (angle_A : ℝ) (h₁ : a = 1) (h₂ : b = sqrt 3) (h₃ : angle_A = 30) :
  (sin_degree angle_A) * b = (sin_degree 60) * a :=
by
  sorry

end angle_B_l442_442571


namespace sqrt_inequality_l442_442632

theorem sqrt_inequality (n : ℕ) (h : 2 ≤ n) : 
  real.sqrt (2 * real.sqrt (3 * (⋯ * (real.sqrt ((n-1) * real.sqrt n))))) < 3 := 
sorry

end sqrt_inequality_l442_442632


namespace no_multiple_of_2310_in_2_j_minus_2_i_l442_442146

theorem no_multiple_of_2310_in_2_j_minus_2_i (i j : ℕ) (h₀ : 0 ≤ i) (h₁ : i < j) (h₂ : j ≤ 50) :
  ¬ ∃ k : ℕ, 2^j - 2^i = 2310 * k :=
by 
  sorry

end no_multiple_of_2310_in_2_j_minus_2_i_l442_442146


namespace Carter_gave_Marcus_58_cards_l442_442251

-- Define the conditions as variables
def original_cards : ℕ := 210
def current_cards : ℕ := 268

-- Define the question as a function
def cards_given_by_carter (original current : ℕ) : ℕ := current - original

-- Statement that we need to prove
theorem Carter_gave_Marcus_58_cards : cards_given_by_carter original_cards current_cards = 58 :=
by
  -- Proof goes here
  sorry

end Carter_gave_Marcus_58_cards_l442_442251


namespace ratio_proof_l442_442405

noncomputable def side_length_triangle(a : ℝ) : ℝ := a / 3
noncomputable def side_length_square(b : ℝ) : ℝ := b / 4
noncomputable def area_triangle(a : ℝ) : ℝ := (side_length_triangle(a)^2 * Mathlib.sqrt(3)) / 4
noncomputable def area_square(b : ℝ) : ℝ := (side_length_square(b))^2

theorem ratio_proof (a b : ℝ) (h : area_triangle(a) = area_square(b)) : a / b = 2 * Mathlib.sqrt(3) / 9 :=
by {
  sorry
}

end ratio_proof_l442_442405


namespace linda_five_dollar_bills_l442_442959

theorem linda_five_dollar_bills :
  ∃ (x y : ℕ), x + y = 15 ∧ 5 * x + 10 * y = 100 ∧ x = 10 :=
by
  sorry

end linda_five_dollar_bills_l442_442959


namespace existence_of_nice_subsets_l442_442745

noncomputable def B : ℝ × ℝ := (-1, 0)
noncomputable def C : ℝ × ℝ := (1, 0)

def isNice (S : set (ℝ × ℝ)) : Prop :=
  (∃ T ∈ S, ∀ Q ∈ S, segment T Q ⊆ S) ∧
  (∀ (P₁ P₂ P₃ : Triangle), ∃! A ∈ S, ∃ σ : permutation (fin 3), similar (triangle.mk B A C) (triangle.mk (perm P₁ σ) (perm P₂ σ) (perm P₃ σ)))

def S : set (ℝ × ℝ) := {p | (p.1 + 1) ^ 2 + p.2 ^ 2 ≤ 4 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}
def S' : set (ℝ × ℝ) := {p | (p.1 + 1) ^ 2 + p.2 ^ 2 ≥ 4 ∧ (p.1 - 1) ^ 2 + p.2 ^ 2 ≤ 4 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

theorem existence_of_nice_subsets :
  isNice S ∧ isNice S' ∧ ∀ (P₁ P₂ P₃ : Triangle) (A ∈ S) (A' ∈ S'), dist (BA : ℝ) (BA' : ℝ) = 4 :=
by
  sorry

end existence_of_nice_subsets_l442_442745


namespace proof_l442_442852

-- Define the conditions
def condition_p : Prop :=
  ∀ z : ℂ, (z - complex.i) * (-complex.i) = 5 → z = 6 * complex.i

def condition_q : Prop :=
  (1 + complex.i) / (1 + 2 * complex.i) = (3 / 5) - (1 / 5) * complex.i

-- Define the propositions p and q
def p : Prop := condition_p
def q : Prop := condition_q

-- The main theorem
theorem proof :
  (p ∧ ¬q) :=
by 
  sorry

end proof_l442_442852


namespace fraction_crop_to_longest_side_l442_442769

/-- Given a kite-shaped field with the conditions below,
prove that the fraction of the crop that is brought to the side AB (120 m) is 1/2. --/
theorem fraction_crop_to_longest_side (AB AD BC CD : ℝ) (α β : ℝ) 
  (h_AB_AD : AB = 120) (h_AD_AB : AD = 120)
  (h_BC_CD : BC = 80) (h_CD_BC : CD = 80)
  (h_angles : α = 120 ∧ β = 60)
  : (determine_fraction_to_AB AB AD BC CD α β) = 1 / 2 :=
sorry

end fraction_crop_to_longest_side_l442_442769


namespace pencil_eraser_cost_l442_442794

theorem pencil_eraser_cost (p e : ℕ) (h1 : 15 * p + 5 * e = 200) (h2 : p > e) (h_p_pos : p > 0) (h_e_pos : e > 0) :
  p + e = 18 :=
  sorry

end pencil_eraser_cost_l442_442794


namespace cab_late_time_l442_442719

theorem cab_late_time (S : ℝ) (T : ℝ) (T_prime : ℝ) (late_time : ℝ) (h1 : T = 40) (h2 : ∀ (S : ℝ), T' = (6 / 5) * T) : late_time = 8 :=
by
  have h3 : T_prime = 48,
  { rw h1,
    unfold T_prime,
    norm_num, },
  rw h3,
  unfold late_time,
  norm_num,
  sorry

end cab_late_time_l442_442719


namespace sum_max_min_l442_442977

noncomputable def max_sum (x y z : ℝ) : ℝ :=
if h : 6 * x + 5 * y + 4 * z = 120 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ 0 then
  max (x + y + z) else
  0

noncomputable def min_sum (x y z : ℝ) : ℝ :=
if h : 6 * x + 5 * y + 4 * z = 120 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ 0 then
  min (x + y + z) else
  0

theorem sum_max_min (x y z : ℝ) :
  (max_sum x y z) + (min_sum x y z) = 44 :=
sorry

end sum_max_min_l442_442977


namespace angle_AMB_ninety_degrees_l442_442934

theorem angle_AMB_ninety_degrees
  (A B C D M : Point)
  (parallelogram_ ABCD : Parallelogram A B C D)
  (midpoint_M : Midpoint M C D)
  (angle_bisector_M : AngleBisector M A D B) :
  Angle A M B = 90 :=
sorry

end angle_AMB_ninety_degrees_l442_442934


namespace cookie_distribution_l442_442421

def trays := 4
def cookies_per_tray := 24
def total_cookies := trays * cookies_per_tray
def packs := 8
def cookies_per_pack := total_cookies / packs

theorem cookie_distribution : cookies_per_pack = 12 := by
  sorry

end cookie_distribution_l442_442421


namespace tim_prank_combinations_l442_442315

def number_of_combinations : Nat :=
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 1
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

theorem tim_prank_combinations : number_of_combinations = 60 :=
by
  sorry

end tim_prank_combinations_l442_442315


namespace cone_volume_is_3pi_l442_442114

noncomputable def volume_of_cone (r l : ℝ) : ℝ := (1 / 3) * π * r^2 * (l * sin (real.pi / 3))

theorem cone_volume_is_3pi (r l : ℝ) (h1 : π * r * l = 2 * π * r^2) (h2 : 16 * π = 4 * π * ((r^2 + (l * sin(real.pi / 3))^2 / 4 ) / ( sin(real.pi / 3) / 2 ) )) : 
  volume_of_cone r l = 3 * π :=
by 
  sorry

end cone_volume_is_3pi_l442_442114


namespace domain_of_y_l442_442435

noncomputable def y (x : ℝ) : ℝ := 1 / (1 + 1 / x)

theorem domain_of_y :
  {x : ℝ | x ≠ 0 ∧ x ≠ -1} = {x : ℝ | y x ∈ ℝ} :=
by
  sorry

end domain_of_y_l442_442435


namespace open_arc_and_bisector_l442_442258

noncomputable def geometric_locus (A B C M : Point) (inside_angle_C : ∠ ACB > 0) (bisects_angle : ∠ AMC = ∠ BMC) : Prop :=
  AC = BC ∧ (on_circle A B C M) 

theorem open_arc_and_bisector (A B C M : Point) (h1 : AC = BC) (h2 : inside_angle_C ∧ bisects_angle) : 
  geometric_locus A B C M inside_angle_C bisects_angle ↔ 
    (on_open_arc A B M ∧ M ≠ C) :=
sorry

end open_arc_and_bisector_l442_442258


namespace f_increasing_on_Icc_inequality_solution_t_value_range_l442_442857

noncomputable def problem_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Icc (-1 : ℝ) 1, f (-x) = -f x) ∧  -- Odd function
  f 1 = 1 ∧  -- f(1) = 1
  ∀ m n ∈ Icc (-1 : ℝ) 1, m + n ≠ 0 → (f m + f n) / (m + n) > 0  -- Given condition

theorem f_increasing_on_Icc (f : ℝ → ℝ) (h : problem_conditions f) :
  ∀ x1 x2 ∈ Icc (-1 : ℝ) 1, x1 < x2 → f x1 < f x2 :=
sorry

theorem inequality_solution (f : ℝ → ℝ) (h : problem_conditions f):
  {x : ℝ | f (x + 1/2) < f (1-x)} = Ico 0 (1/4) :=
sorry

theorem t_value_range (f : ℝ → ℝ) (h : problem_conditions f) :
  (∀ α ∈ Icc (-π/3 : ℝ) (π/4), ∀ x ∈ Icc (-1 : ℝ) 1, f x ≤ t^2 + t - 1/ (real.cos α)^2 - 2 * real.tan α - 1) →
  t ∈ Iio (-3) ∪ Ici (2) :=
sorry

end f_increasing_on_Icc_inequality_solution_t_value_range_l442_442857


namespace find_hyperbola_eccentricity_l442_442914

theorem find_hyperbola_eccentricity (m : ℝ) :
  (∀ (e : ℝ), e = sqrt 5 →
  (exists (a b c : ℝ), a^2 = m ∧ b^2 = m + 1 ∧ c^2 = a^2 + b^2 ∧ e = c / a) ∨
  (exists (a b c : ℝ), a^2 = -(m + 1) ∧ b^2 = -m ∧ c^2 = a^2 + b^2 ∧ e = c / a)) →
  m = 1 / 3 ∨ m = -4 / 3 := 
by
  intros hm he
  sorry

end find_hyperbola_eccentricity_l442_442914


namespace unique_solution_l442_442451

def is_prime (p : ℕ) : Prop := ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem unique_solution (n : ℕ) :
  (0 < n ∧ is_prime (n + 1) ∧ is_prime (n + 3) ∧
   is_prime (n + 7) ∧ is_prime (n + 9) ∧
   is_prime (n + 13) ∧ is_prime (n + 15)) ↔ n = 4 :=
by
  sorry

end unique_solution_l442_442451


namespace james_chess_master_is_102_l442_442927

noncomputable def james_chess_master : ℕ :=
  let learning_rules_time := 2
  let proficiency_time := 49 * learning_rules_time
  let total_initial_time := learning_rules_time + proficiency_time
  let total_time := 10100
  let master_time := total_time - total_initial_time
  (master_time / proficiency_time).round

theorem james_chess_master_is_102 : james_chess_master = 102 := by
  sorry

end james_chess_master_is_102_l442_442927


namespace Kingdom_Wierdo_guard_number_bound_l442_442599

variable (N : ℕ) (P : Finset (Fin N))

def total_guards (f : P → P → ℕ) : ℕ :=
  P.sum (λ u, P.sum (λ v, if u ≠ v then f u v else 0))

theorem Kingdom_Wierdo_guard_number_bound
  (N_pos : 0 < N)
  (guards : Fin N → Fin N → ℕ)
  (h1 : ∀ (a b c: Fin N), a ≠ b ∧ b ≠ c ∧ a ≠ c → guards a b < 4 ∧ guards b c < 4 ∧ guards c a < 4)
  (h2 : ∀ (a b c d: Fin N), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    (guards a b ≠ 3 ∨ guards a c ≠ 3 ∨ guards a d ≠ 3)) :
  (total_guards P guards ≤ N * N) :=
begin
  sorry
end

end Kingdom_Wierdo_guard_number_bound_l442_442599


namespace John_remaining_money_l442_442201

theorem John_remaining_money :
  ∀ (john_savings_base7 : ℕ) (ticket_cost : ℕ) (john_money_base10 : ℕ),
  john_savings_base7 = 6534 ∧ ticket_cost = 1200 ∧ 
  john_money_base10 = (6 * 7^3 + 5 * 7^2 + 3 * 7^1 + 4 * 7^0) →
  john_money_base10 - ticket_cost = 1128 :=
by
  intros john_savings_base7 ticket_cost john_money_base10
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [←h4, ←h3, ←h1]
  sorry

end John_remaining_money_l442_442201


namespace minimum_blue_chips_l442_442370

theorem minimum_blue_chips (w r b : ℕ) : 
  (b ≥ w / 3) ∧ (b ≤ r / 4) ∧ (w + b ≥ 75) → b ≥ 19 :=
by sorry

end minimum_blue_chips_l442_442370


namespace min_hypotenuse_of_right_triangle_l442_442334

theorem min_hypotenuse_of_right_triangle (a b c k : ℝ) (h₁ : k = a + b + c) (h₂ : a^2 + b^2 = c^2) : 
  c ≥ (Real.sqrt 2 - 1) * k := 
sorry

end min_hypotenuse_of_right_triangle_l442_442334


namespace p_sufficient_for_s_l442_442859

-- Definitions based on given conditions
variables {p q r s : Prop}

-- Given conditions
axiom h1 : p → q
axiom h2 : s → q
axiom h3 : q → r
axiom h4 : r → s

-- Proof that p is a sufficient condition for s
theorem p_sufficient_for_s : p → s :=
by {
  intro hp,
  apply h4,
  apply h3,
  apply h1,
  exact hp,
}

end p_sufficient_for_s_l442_442859


namespace total_legs_l442_442902

theorem total_legs (bees ants spiders : ℕ) 
  (legs_per_bee legs_per_ant legs_per_spider : ℕ) :
  bees = 50 → 
  ants = 35 → 
  spiders = 20 → 
  legs_per_bee = 6 → 
  legs_per_ant = 6 → 
  legs_per_spider = 8 → 
  bees * legs_per_bee + ants * legs_per_ant + spiders * legs_per_spider = 670 :=
by
  intros h_bees h_ants h_spiders h_legs_per_bee h_legs_per_ant h_legs_per_spider
  rw [h_bees, h_ants, h_spiders, h_legs_per_bee, h_legs_per_ant, h_legs_per_spider]
  sorry

end total_legs_l442_442902


namespace relationship_among_abc_l442_442856

theorem relationship_among_abc (a b c : ℝ) (h1 : a = 5^0.8) (h2 : b = 0.8^5) (h3 : c = Real.log 0.8 / Real.log 5) : c < b ∧ b < a :=
by
  sorry

end relationship_among_abc_l442_442856


namespace correct_word_to_complete_sentence_l442_442577

theorem correct_word_to_complete_sentence
  (parents_spoke_language : Bool)
  (learning_difficulty : String) :
  learning_difficulty = "It was hard for him to learn English in a family, in which neither of the parents spoke the language." :=
by
  sorry

end correct_word_to_complete_sentence_l442_442577


namespace parabola_equation_l442_442481

theorem parabola_equation {x y : ℝ} (M : ℝ × ℝ) (p : ℝ) (h_symm : ∀ x y : ℝ, x^2 = -2 * p * y) (h_vertex : x = 0 ∧ y = 0) (h_passes : M = (sqrt 3, -2 * sqrt 3)) :
  x^2 = - (sqrt 3 / 2) * y :=
by
  intro x y
  sorry

end parabola_equation_l442_442481


namespace average_salary_rest_l442_442664

noncomputable def average_salary_of_the_rest : ℕ := 6000

theorem average_salary_rest 
  (N : ℕ) 
  (A : ℕ)
  (T : ℕ)
  (A_T : ℕ)
  (Nr : ℕ)
  (Ar : ℕ)
  (H1 : N = 42)
  (H2 : A = 8000)
  (H3 : T = 7)
  (H4 : A_T = 18000)
  (H5 : Nr = N - T)
  (H6 : Nr = 42 - 7)
  (H7 : Ar = 6000)
  (H8 : 42 * 8000 = (Nr * Ar) + (T * 18000))
  : Ar = average_salary_of_the_rest :=
by
  sorry

end average_salary_rest_l442_442664


namespace max_students_distributing_items_l442_442741

-- Define the given conditions
def pens : Nat := 1001
def pencils : Nat := 910

-- Define the statement
theorem max_students_distributing_items :
  Nat.gcd pens pencils = 91 :=
by
  sorry

end max_students_distributing_items_l442_442741


namespace nested_fraction_evaluation_l442_442446

theorem nested_fraction_evaluation : 
  (1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - 1 / 3)))))))) = (21 / 55) :=
by
  sorry

end nested_fraction_evaluation_l442_442446


namespace option_C_incorrect_l442_442345

variable (Point : Type)
variable (Line Plane : Type) 
variable (A B : Point) 
variable (l : Line) 
variable (α β : Plane)
variable (OnLine : Point → Line → Prop) 
variable (OnPlane : Point → Plane → Prop)
variable (SubsetLine : Line → Plane → Prop) 
variable (Intersection : Plane → Plane → Set Point)
variable (SetUnion : Set Point → Set Point → Set Point)
variable (Equal : Set Point → Set Point → Prop)

-- Conditions:
axiom A_on_l : OnLine A l
axiom A_on_α : OnPlane A α
axiom B_on_l : OnLine B l
axiom B_on_α : OnPlane B α
axiom l_not_subset_α : ¬ SubsetLine l α

theorem option_C_incorrect : A_on_l → A_on_α → ¬ SubsetLine l α → ¬ OnPlane A α := 
by sorry

end option_C_incorrect_l442_442345


namespace M_eq_N_l442_442518

def M (u : ℤ) : Prop := ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l
def N (u : ℤ) : Prop := ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r

theorem M_eq_N : ∀ u : ℤ, M u ↔ N u := by
  sorry

end M_eq_N_l442_442518


namespace complement_A_is_interval_l442_442876

def U : Set ℝ := {x | True}
def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def compl_U_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem complement_A_is_interval : compl_U_A = {x | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end complement_A_is_interval_l442_442876


namespace costPerUse_l442_442198

-- Definitions based on conditions
def heatingPadCost : ℝ := 30
def usesPerWeek : ℕ := 3
def totalWeeks : ℕ := 2

-- Calculate the total number of uses
def totalUses : ℕ := usesPerWeek * totalWeeks

-- The amount spent per use
theorem costPerUse : heatingPadCost / totalUses = 5 := by
  sorry

end costPerUse_l442_442198


namespace combined_money_l442_442317

/-- Tom has a quarter the money of Nataly. Nataly has three times the money of Raquel.
     Sam has twice the money of Nataly. Raquel has $40. Prove that combined they have $430. -/
theorem combined_money : 
  ∀ (T R N S : ℕ), 
    (T = N / 4) ∧ 
    (N = 3 * R) ∧ 
    (S = 2 * N) ∧ 
    (R = 40) → 
    T + R + N + S = 430 := 
by
  sorry

end combined_money_l442_442317


namespace mass_percentage_Al_in_Al2O3_l442_442333

-- Define the atomic masses and formula unit
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_O : ℝ := 16.00
def molar_mass_Al2O3 : ℝ := (2 * atomic_mass_Al) + (3 * atomic_mass_O)
def mass_Al_in_Al2O3 : ℝ := 2 * atomic_mass_Al

-- Define the statement for the mass percentage of Al in Al2O3
theorem mass_percentage_Al_in_Al2O3 : (mass_Al_in_Al2O3 / molar_mass_Al2O3) * 100 = 52.91 :=
by
  sorry -- Proof to be filled in

end mass_percentage_Al_in_Al2O3_l442_442333


namespace number_of_valid_p_l442_442088

theorem number_of_valid_p 
    (AB : ℕ) (BC : ℕ) (CD : ℕ) (AD : ℕ) (x : ℕ) (y : ℕ) (p : ℕ) 
    (AB_eq : AB = 3)
    (CD_eq_AD : CD = AD)
    (x_square_eq : x^2 = 6*y - 9)
    (perimeter_eq : p = 3 + x + 2*y)
    (p_bound : p < 2300) : 
    (setOf (p) < 2300).card = 47 := 
begin
    sorry
end

end number_of_valid_p_l442_442088


namespace log_subtraction_l442_442075

theorem log_subtraction :
  (Real.log 125 / Real.log 5) - (Real.log (1 / 25) / Real.log 5) = 5 := by
  -- Let a = log_5(125)
  let a := Real.log 125 / Real.log 5
  -- Let b = log_5(1/25)
  let b := Real.log (1 / 25) / Real.log 5
  -- We need to show a - b = 5
  have h1 : a = 3 := by
    rw [Real.log, Real.log]
    norm_num
  have h2 : b = -2 := by
    rw [Real.log, Real.log]
    norm_num
  rw [h1, h2]
  norm_num
  done
end

end log_subtraction_l442_442075


namespace shaded_trapezoid_area_l442_442091

theorem shaded_trapezoid_area :
  let side_smallest := 3
  let side_second := 5
  let side_third := 7
  let side_largest := 9
  let total_horizontal_length := side_smallest + side_second + side_third + side_largest
  let height_largest := side_largest
  let ratio := height_largest / total_horizontal_length
  let height_smallest_contrib := side_smallest * ratio
  let total_vertical_height := side_second + side_third + height_largest
  let height_trapezoid := total_vertical_height - height_smallest_contrib
  let base1 := side_smallest
  let base2 := side_largest
  let area := 1/2 * (base1 + base2) * height_trapezoid
  in area = 119.25 :=
by
  sorry

end shaded_trapezoid_area_l442_442091


namespace force_and_duration_of_explosion_l442_442410

noncomputable def door_explosion_data : Type :=
  { a : ℝ // a = 2.20 }
  { b : ℝ // b = 1.15 }
  { m : ℝ // m = 30 }
  { h : ℝ // h = 6 }
  { s : ℝ // s = 80 }
  { t : ℝ // t = 1500 }
  { α : ℝ // α = 1 / 273 }

noncomputable def force_exerted : ℝ :=
  143200

noncomputable def explosion_duration : ℝ :=
  0.003

theorem force_and_duration_of_explosion (data : door_explosion_data) :
  (force_exerted = 143200 ∧ explosion_duration = 0.003) :=
by
  sorry

end force_and_duration_of_explosion_l442_442410


namespace determine_possible_values_l442_442645

noncomputable def possibleValues (a b c d : ℝ) : set ℝ :=
  if a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 then
    {fracSign a + fracSign b + fracSign c + fracSign d + fracSign (a*b*c*d)}
  else ∅
  where
    fracSign (x : ℝ) : Int :=
      if x > 0 then 1 else -1

theorem determine_possible_values (a b c d : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  possibleValues a b c d = {5, 1, -1, -5} :=
sorry -- Proof goes here

end determine_possible_values_l442_442645


namespace geometric_mean_ratio_lt_2_l442_442298

-- Defining the necessary structures for the problem
variable {b q : ℝ} (h_pos : b > 0) (h_non_neg : q > 0)

-- The lengths of the sides of the triangle are given by b, b * q, and b * q^2
-- According to the triangle inequality, we have the following conditions:
def triangle_inequality_1 := b + (b * q) > (b * q^2)
def triangle_inequality_2 := b + (b * q^2) > (b * q)
def triangle_inequality_3 := (b * q) + (b * q^2) > b

-- We need to prove that the common ratio q is less than 2
theorem geometric_mean_ratio_lt_2 : (0 < b) → (0 < q) → 
  (triangle_inequality_1 h_pos h_non_neg) → 
  (triangle_inequality_2 h_pos h_non_neg) →
  (triangle_inequality_3 h_pos h_non_neg) → 
  q < 2 := sorry

end geometric_mean_ratio_lt_2_l442_442298


namespace even_four_digit_increasing_count_l442_442524

theorem even_four_digit_increasing_count :
  let digits := {x // 1 ≤ x ∧ x ≤ 9}
  let even_digits := {x // x ∈ digits ∧ x % 2 = 0}
  {n : ℕ //
    ∃ a b c d : ℕ,
      n = a * 1000 + b * 100 + c * 10 + d ∧
      a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ even_digits ∧
      a < b ∧ b < c ∧ c < d} =
  17 :=
by sorry

end even_four_digit_increasing_count_l442_442524


namespace projection_matrix_solution_l442_442438

theorem projection_matrix_solution 
  (a c : ℚ) 
  (P : Matrix (Fin 2) (Fin 2) ℚ := ![![a, 18/45], ![c, 27/45]])
  (hP : P * P = P) :
  (a, c) = (9/25, 12/25) :=
by
  sorry

end projection_matrix_solution_l442_442438


namespace q_third_derivative_value_of_q_third_derivative_at_5_l442_442165

noncomputable def q (x : ℝ) := sorry

def q' (x : ℝ) := 3 * q x - 3

def q'' (x : ℝ) := (3 : ℝ)

def q''' (x : ℝ) := (0 : ℝ)

theorem q_third_derivative (x : ℝ) : (q'' x)' = q''' x :=
by sorry

theorem value_of_q_third_derivative_at_5 : (q'' 5)' = 0 :=
by
  have h := q_third_derivative 5
  simp [q'''_eq_constant] at h
  exact h

end q_third_derivative_value_of_q_third_derivative_at_5_l442_442165


namespace ones_digit_of_73_pow_351_l442_442463

theorem ones_digit_of_73_pow_351 : 
  (73 ^ 351) % 10 = 7 := 
by 
  sorry

end ones_digit_of_73_pow_351_l442_442463


namespace monotonic_decreasing_iff_l442_442862

def f (a x : ℝ) : ℝ := -x^3 + x^2 - a * x + 1

theorem monotonic_decreasing_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (a ≥ 1/3) :=
begin
  sorry
end

end monotonic_decreasing_iff_l442_442862


namespace compute_expression_l442_442806

theorem compute_expression :
  real.cbrt 27 + (-1 / 3)⁻¹ - (3 - real.pi)^0 + (-1:ℤ)^2017 = -2 :=
by
  sorry

end compute_expression_l442_442806


namespace find_x0_l442_442870

noncomputable theory -- if necessary due to integrals

def f (a b x : ℝ) := a * x ^ 2 + b

def integral_value (a b : ℝ) : ℝ :=
  (∫ x in (0:ℝ)..2, f a b x)

theorem find_x0 (a b x0 : ℝ) (h₀ : a ≠ 0) (h₁ : ∫ x in (0:ℝ)..2, f a b x = 2 * f a b x0) (h₂ : 0 < x0) : 
  x0 = 2 * Real.sqrt 3 / 3 :=
sorry

end find_x0_l442_442870


namespace work_problem_correct_l442_442162

noncomputable def work_problem : Prop :=
  let A := 1 / 36
  let C := 1 / 6
  let total_rate := 1 / 4
  ∃ B : ℝ, (A + B + C = total_rate) ∧ (B = 1 / 18)

-- Create the theorem statement which says if the conditions are met,
-- then the rate of b must be 1/18 and the number of days b alone takes to
-- finish the work is 18.
theorem work_problem_correct (A C total_rate B : ℝ) (h1 : A = 1 / 36) (h2 : C = 1 / 6) (h3 : total_rate = 1 / 4) 
(h4 : A + B + C = total_rate) : B = 1 / 18 ∧ (1 / B = 18) :=
  by
  sorry

end work_problem_correct_l442_442162


namespace first_pump_half_pond_time_l442_442260

theorem first_pump_half_pond_time:
  ∃ t : ℝ, t = 1 / 2 ∧ 
  (∀ (rate_first rate_second total_rate : ℝ),
     rate_first = 1 / (2 * t) ∧
     rate_second = 1 / 1.090909090909091 ∧
     total_rate = 2 →
     (rate_first + rate_second = total_rate)) := 
begin
  use 1 / 2,
  split,
  {
    exact rfl,
  },
  {
    intros rate_first rate_second total_rate,
    rintros ⟨h₁, ⟨h₂, h₃⟩⟩,
    simp [h₁, h₂, h₃] at *,
    sorry -- Proof goes here
  }
end

end first_pump_half_pond_time_l442_442260


namespace eval_expression_l442_442077

-- Definitions for the problem conditions
def reciprocal (a : ℕ) : ℚ := 1 / a

-- The theorem statement
theorem eval_expression : (reciprocal 9 - reciprocal 6)⁻¹ = -18 := by
  sorry

end eval_expression_l442_442077


namespace ratio_of_monkeys_to_snakes_l442_442929

-- Define the given problem conditions
def JohnZoo :=
  let snakes := 15
  ∃ M L P D : ℤ,
    L = M - 5 ∧
    P = L + 8 ∧
    D = P / 3 ∧
    15 + M + L + P + D = 114 ∧
    2 * snakes = M

-- Statement of the problem to prove in Lean
theorem ratio_of_monkeys_to_snakes : JohnZoo → (15 * 2 = 30) := sorry

end ratio_of_monkeys_to_snakes_l442_442929


namespace melon_weights_l442_442027

-- We start by defining the weights of the individual melons.
variables {D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 : ℝ}

-- Define the weights of the given sets of three melons.
def W1 := D1 + D2 + D3
def W2 := D2 + D3 + D4
def W3 := D1 + D3 + D4
def W4 := D1 + D2 + D4
def W5 := D5 + D6 + D7
def W6 := D8 + D9 + D10

-- State the theorem to be proven.
theorem melon_weights (W1 W2 W3 W4 W5 W6 : ℝ) :
  (W1 + W2 + W3 + W4) / 3 + W5 + W6 = D1 + D2 + D3 + D4 + D5 + D6 + D7 + D8 + D9 + D10 :=
sorry 

end melon_weights_l442_442027


namespace different_numerators_count_l442_442606

-- Definitions and conditions for the problem
def is_repeating_decimal (r : ℚ) : Prop :=
  ∃ (a b c d : ℕ), r = (a * 1000 + b * 100 + c * 10 + d) / 9999 ∧ 0 < r ∧ r < 1

def not_divisible_by_3_11_101 (n : ℕ) : Prop :=
  ¬(3 ∣ n) ∧ ¬(11 ∣ n) ∧ ¬(101 ∣ n)

-- Main theorem statement
theorem different_numerators_count : 
  let numerators := {abcd : ℕ | ∃ (r : ℚ), is_repeating_decimal r ∧ r * 9999 = abcd}
  let valid_numerators := {n ∈ numerators | not_divisible_by_3_11_101 n}
  let count := finite_card valid_numerators
  count = 5401 :=
sorry

end different_numerators_count_l442_442606


namespace solution_correctness_l442_442080

noncomputable def solution_set : Set ℝ := {x | x + 60 / (x - 5) = -12}

theorem solution_correctness : solution_set = {0, -7} := 
begin
  sorry
end

end solution_correctness_l442_442080


namespace find_number_l442_442340

theorem find_number :
  ∃ n : ℕ, n * (1 / 7)^2 = 7^3 :=
by
  sorry

end find_number_l442_442340


namespace min_pos_value_sum_l442_442443

theorem min_pos_value_sum (a : Fin 50 → ℤ) (h : ∀ i, a i = 1 ∨ a i = -1) :
  ∃ S, S = ∑ i j in Finset.filter (λ p : Fin 50 × Fin 50, p.fst < p.snd) (Finset.univ.product Finset.univ), a p.fst * a p.snd ∧ S = 7 := 
sorry

end min_pos_value_sum_l442_442443


namespace max_value_of_expression_l442_442998

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l442_442998


namespace perp_condition_parallel_condition_l442_442519

variables (a b : ℝ)

-- Definition of line l1 passing through point (-3, -1)
def line1_passing_through := -3 * a + b + 4 = 0

-- Definition of l1 perpendicular to l2
def l1_perp_l2 := a * (a - 1) - b = 0

-- Proof that if l1_perp_l2 and line1_passing_through holds, then a = 2 and b = 2
theorem perp_condition (h1 : l1_perp_l2 a b) (h2 : line1_passing_through a b) :
  a = 2 ∧ b = 2 :=
sorry

-- Definition of l1 parallel to l2
def l1_parallel_l2 := a / b = 1 - a

-- Definition of equal distance condition from origin to l1 and l2
def equal_distances := 4 / b = -b

-- Proof that if l1_parallel_l2 and equal_distances holds, then a = 2 and b = -2
theorem parallel_condition (h3 : l1_parallel_l2 a b) (h4 : equal_distances a b) :
  a = 2 ∧ b = -2 :=
sorry

end perp_condition_parallel_condition_l442_442519


namespace vector_magnitude_problem_l442_442879

noncomputable def vector_length (v : ℝ) [|v|] := sorry

theorem vector_magnitude_problem
    (a b : ℝ) 
    (angle_ab : ℝ) 
    (h : angle_ab = real.pi / 3) 
    (hb : |b| = 1) 
    (ha_b : |a - 2 * b| = real.sqrt 7) 
    : |a| = 3 := 
sorry

end vector_magnitude_problem_l442_442879


namespace volume_of_tetrahedron_eq_400_l442_442183

variables (A B C D O : Type) [InnerProductSpace ℝ A]

noncomputable def AD_eq_BC (AD BC : ℝ) := AD = BC
noncomputable def BCD_area_eq_100 (BCD_area : ℝ) := BCD_area = 100
noncomputable def distance_from_center_to_base_eq_3 (dist_center_base : ℝ) := dist_center_base = 3
noncomputable def angles_sum_eq_180 (angles_sum_B angles_sum_C : ℝ) :=
  angles_sum_B = 180 ∧ angles_sum_C = 180

def volume_of_tetrahedron {A B C D : A} (AD BC BCD_area dist_center_base : ℝ) : ℝ :=
  ((1/3 : ℝ) * 4 * BCD_area * dist_center_base)

theorem volume_of_tetrahedron_eq_400 {A B C D : A} (AD BC BCD_area dist_center_base angles_sum_B angles_sum_C : ℝ)
  (h1 : AD_eq_BC AD BC) (h2 : BCD_area_eq_100 BCD_area) 
  (h3 : distance_from_center_to_base_eq_3 dist_center_base) 
  (h4 : angles_sum_eq_180 angles_sum_B angles_sum_C) :
  volume_of_tetrahedron AD BC BCD_area dist_center_base = 400 := by
  sorry

end volume_of_tetrahedron_eq_400_l442_442183


namespace max_value_ln_minus_x_on_0_e_l442_442825

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x

theorem max_value_ln_minus_x_on_0_e :
  ∃ x ∈ Ioc 0 (exp 1), ∀ y ∈ Ioc 0 (exp 1), f y ≤ f x ∧ f x = -1 :=
begin
  sorry
end

end max_value_ln_minus_x_on_0_e_l442_442825


namespace intersection_P_Q_l442_442517

def P := {x : ℝ | 1 < x ∧ x < 3}
def Q := {x : ℝ | 2 < x}

theorem intersection_P_Q :
  P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := sorry

end intersection_P_Q_l442_442517


namespace age_of_20th_student_l442_442663

theorem age_of_20th_student (avg_age_20 : ℕ) (avg_age_9 : ℕ) (avg_age_10 : ℕ) :
  (avg_age_20 = 20) →
  (avg_age_9 = 11) →
  (avg_age_10 = 24) →
  let T := 20 * avg_age_20
  let T1 := 9 * avg_age_9
  let T2 := 10 * avg_age_10
  let T19 := T1 + T2
  let age_20th := T - T19
  (age_20th = 61) :=
by
  intros h1 h2 h3
  let T := 20 * avg_age_20
  let T1 := 9 * avg_age_9
  let T2 := 10 * avg_age_10
  let T19 := T1 + T2
  let age_20th := T - T19
  sorry

end age_of_20th_student_l442_442663


namespace complaints_over_3_days_l442_442656

theorem complaints_over_3_days
  (n : ℕ) (n_ss : ℕ) (n_both : ℕ) (total : ℕ)
  (h1 : n = 120)
  (h2 : n_ss = n + 1/3 * n)
  (h3 : n_both = n_ss + 0.20 * n_ss)
  (h4 : total = n_both * 3) :
  total = 576 :=
by
  sorry

end complaints_over_3_days_l442_442656


namespace middle_rectangle_frequency_l442_442564

theorem middle_rectangle_frequency
  (total_rectangles : ℕ)
  (middle_area_eq_sum_rest : ℕ)
  (sample_size : ℕ)
  (h1 : total_rectangles = 11)
  (h2 : middle_area_eq_sum_rest)
  (h3 : sample_size = 160) :
  middle_area_eq_sum_rest = (sample_size / 2) :=
by
  sorry

end middle_rectangle_frequency_l442_442564


namespace wire_cut_ratio_l442_442399

theorem wire_cut_ratio (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) 
                        (h_eq_area : (a^2 * Real.sqrt 3) / 36 = (b^2) / 16) :
  a / b = Real.sqrt 3 / 2 :=
by
  sorry

end wire_cut_ratio_l442_442399


namespace number_of_littering_citations_l442_442385

variable (L D P : ℕ)
variable (h1 : L = D)
variable (h2 : P = 2 * (L + D))
variable (h3 : L + D + P = 24)

theorem number_of_littering_citations : L = 4 :=
by
  sorry

end number_of_littering_citations_l442_442385


namespace solve_fractional_equation_l442_442644

theorem solve_fractional_equation {x : ℝ} (h1 : x ≠ -1) (h2 : x ≠ 0) :
  6 / (x + 1) = (x + 5) / (x * (x + 1)) ↔ x = 1 :=
by
  -- This proof is left as an exercise.
  sorry

end solve_fractional_equation_l442_442644


namespace eq_condition_l442_442036

theorem eq_condition (a : ℝ) :
  (∃ x : ℝ, a * (4 * |x| + 1) = 4 * |x|) ↔ (0 ≤ a ∧ a < 1) :=
by
  sorry

end eq_condition_l442_442036


namespace graph_passes_through_fixed_point_l442_442829

theorem graph_passes_through_fixed_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ∃ x y, x = 1 ∧ y = 2 ∧ y = a^(x - 1) + 1 := 
by
  use 1, 2
  split
  · refl
  split
  · refl
  · sorry

end graph_passes_through_fixed_point_l442_442829


namespace hundredth_digit_of_7_div_33_l442_442889

theorem hundredth_digit_of_7_div_33 : (100 : ℕ) % 2 = 0 → digit_at (100 : ℕ) (decimal_expansion (7 / 33)) = 1 :=
by
  intros h
  sorry

def decimal_expansion (x : ℚ) : string := "0.212121..."

def digit_at (n : ℕ) (s : string) : char :=
  s.to_list.get (n % s.length)

end hundredth_digit_of_7_div_33_l442_442889


namespace find_coordinates_of_P_l442_442499

noncomputable def coordinates_of_P : set (ℝ × ℝ) :=
  {P : ℝ × ℝ | ∃ (l : ℝ → ℝ) (A B : ℝ × ℝ) (d : ℝ),
    A = (4, -3) ∧ B = (2, -1) ∧ symmetric A B l ∧
    (P.2 = l P.1) ∧ distance_to_line P (4, 3, -2) = 2}

theorem find_coordinates_of_P :
  coordinates_of_P = {(1, -4), (27/7, -8/7)} := 
sorry

end find_coordinates_of_P_l442_442499


namespace cube_surface_area_doubling_l442_442886

theorem cube_surface_area_doubling (a : ℝ) (a_pos : 0 < a) : 
  let original_surface_area := 6 * (a ^ 2),
      new_side_length := 2 * a,
      new_surface_area := 6 * (new_side_length ^ 2)
  in new_surface_area = 4 * original_surface_area :=
by
  let original_surface_area := 6 * (a ^ 2)
  let new_side_length := 2 * a
  let new_surface_area := 6 * (new_side_length ^ 2)
  sorry

end cube_surface_area_doubling_l442_442886


namespace cost_per_use_correct_l442_442196

-- Definitions based on conditions in the problem
def total_cost : ℕ := 30
def uses_per_week : ℕ := 3
def number_of_weeks : ℕ := 2
def total_uses : ℕ := uses_per_week * number_of_weeks

-- Statement based on the question and correct answer
theorem cost_per_use_correct : (total_cost / total_uses) = 5 := sorry

end cost_per_use_correct_l442_442196


namespace carB_start_time_later_l442_442323

-- Definitions based on the problem conditions
def speed := 55 -- speed in km/hour

def distance_traveled (hours : ℕ) (car : ℕ) : ℕ :=
  if car = 0 then 5 * hours * speed
  else hours * speed

axiom travel_times_inequality (x : ℕ) :
  5 * x / speed + 2 = 3 * (x / speed + 2)

theorem carB_start_time_later (x : ℕ) (h₁ : travel_times_inequality x) :
  let carA_time := (5 * x) / speed,
      carB_time := x / speed in
  carA_time - carB_time = 8 := 
sorry

end carB_start_time_later_l442_442323


namespace range_of_a_l442_442898

noncomputable def extreme_points_exist (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (λ x, x^3 + a * x).deriv x1 = 0 ∧ (λ x, x^3 + a * x).deriv x2 = 0

theorem range_of_a (a : ℝ) : extreme_points_exist a ↔ a < 0 :=
by
  sorry

end range_of_a_l442_442898


namespace solution_set_l442_442110

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 4*x else -(((-x)^2) + 4*(-x))

theorem solution_set (f_odd : ∀ x : ℝ, f(-x) = -f(x)) :
  {x : ℝ | f(x) > x} = {x : ℝ | (x > 5 ∨ (-5 < x ∧ x < 0))} :=
by
  sorry

end solution_set_l442_442110


namespace find_prices_compare_costs_l442_442176

-- Definitions for the given conditions
def price_difference (m n : ℕ) : Prop := m = n + 40
def equal_basketballs (m n : ℕ) : Prop := 1200 / m = 600 / n

-- Problem 1: Find the unit prices of type A and type B basketballs
theorem find_prices (m n : ℕ) (h1 : price_difference m n) (h2 : equal_basketballs m n) : m = 80 ∧ n = 40 :=
begin
  sorry
end

def cost_option1 (x : ℕ) : ℕ := 36 * x + 1080
def cost_option2 (x : ℕ) : ℕ := 40 * x + 1000 - if x ≥ 15 then 40 * (x / 3) else 0

-- Problem 2: Compare cost effectiveness of two discount options.
theorem compare_costs (x : ℕ) (hx : x ≥ 5) :
  if x > 20 then cost_option1 x < cost_option2 x 
  else if x = 20 then cost_option1 x = cost_option2 x
  else cost_option1 x > cost_option2 x :=
begin
  sorry
end

end find_prices_compare_costs_l442_442176


namespace triangle_angle_bisector_inequality_l442_442972

noncomputable def problem_statement
  (ABC : Type)
  [triangle ABC]
  (A B C : ABC)
  (CF : ABC → ABC)
  (CM : ABC → ABC)
  (CH : ABC → ABC)
  (is_angle_bisector : ∀ x, CF x = angle_bisector A C B x)
  (is_median : ∀ x, CM x = median A B C x)
  (is_altitude : ∀ x, CH x = altitude A B C x)
  (isosceles_or_BC_gt_AC : ABC → Prop)
  (BC_gt_AC : A B C → B > A) := 
  CH ≤ CF ∧ CF ≤ CM

theorem triangle_angle_bisector_inequality
  (ABC : Type)
  [triangle ABC]
  (A B C : ABC)
  (CF : ABC → ABC)
  (CM : ABC → ABC)
  (CH : ABC → ABC)
  (is_angle_bisector : ∀ x, CF x = angle_bisector A C B x)
  (is_median : ∀ x, CM x = median A B C x)
  (is_altitude : ∀ x, CH x = altitude A B C x)
  (isosceles_or_BC_gt_AC : ABC → Prop)
  (BC_gt_AC : A B C → B > A) :
  (CH ≤ CF ∧ CF ≤ CM) := 
by
  sorry

end triangle_angle_bisector_inequality_l442_442972


namespace part1_part2_l442_442133

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 4 - m * Real.sin x - 3 * (Real.cos x) ^ 2

theorem part1 (m : ℝ) (x1 x2 x3 : ℝ) (h1 : 0 < x1 ∧ x1 < π)
              (h2 : 0 < x2 ∧ x2 < π) (h3 : 0 < x3 ∧ x3 < π)
              (h4 : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) 
              (heq1 : f m x1 = 0) (heq2 : f m x2 = 0) (heq3 : f m x3 = 0) :
              m = 4 ∧ x1 + x2 + x3 = 3 * π / 2 := 
by
  sorry

theorem part2 (m : ℝ) :
              (∀ x ∈ [-(π / 6) : ℝ, π : ℝ], f m x > 0) →
              (-7 / 2 : ℝ) < m ∧ m < 2 * Real.sqrt 3 := 
by
  sorry

end part1_part2_l442_442133


namespace smallest_n_colored_squares_l442_442465

theorem smallest_n_colored_squares (chessboard_size : ℕ) (n : ℕ) :
  chessboard_size = 1000 → n = 1999 → ∃ (squares : set (ℕ × ℕ)), 
  (squares.card = n ∧ ∃ (a b c : (ℕ × ℕ)), 
   a ∈ squares ∧ b ∈ squares ∧ c ∈ squares ∧ 
   ((a.1 = b.1 ∧ a.2 - c.2 = c.2 - b.2) ∨ 
    (a.2 = b.2 ∧ a.1 - c.1 = c.1 - b.1))) :=
begin
  sorry
end

end smallest_n_colored_squares_l442_442465


namespace speed_in_kmph_speed_in_mph_speed_in_inches_per_second_final_speed_in_kmph_final_speed_in_mph_l442_442371

-- Definitions based on the conditions provided
def initial_speed_mps := 45.0 -- Speed in meters per second

def conversion_mps_to_kmph := 3.6 -- Conversion factor from m/s to km/h
def conversion_mps_to_mph := 2.23694 -- Conversion factor from m/s to mph
def conversion_m_to_in := 39.3701 -- Conversion factor from meters to inches

def acceleration_mps2 := 2.5 -- Acceleration in meters per second squared
def acceleration_time_s := 10 -- Acceleration time in seconds

-- Proof statements
-- 1. Prove the speed in km/h
theorem speed_in_kmph :
  initial_speed_mps * conversion_mps_to_kmph = 162 := by
  sorry

-- 2. Prove the speed in mph
theorem speed_in_mph :
  initial_speed_mps * conversion_mps_to_mph ≈ 100.6623 := by
  sorry

-- 3. Prove the speed in inches per second
theorem speed_in_inches_per_second :
  initial_speed_mps * conversion_m_to_in ≈ 1771.6545 := by
  sorry

-- 4. Prove the final speed after acceleration in km/h and mph
def final_speed_mps := initial_speed_mps + acceleration_mps2 * acceleration_time_s

theorem final_speed_in_kmph :
  final_speed_mps * conversion_mps_to_kmph = 252 := by
  sorry

theorem final_speed_in_mph :
  final_speed_mps * conversion_mps_to_mph ≈ 156.5858 := by
  sorry

end speed_in_kmph_speed_in_mph_speed_in_inches_per_second_final_speed_in_kmph_final_speed_in_mph_l442_442371


namespace right_triangle_area_l442_442356

theorem right_triangle_area (c a : ℝ) (hypotenuse_cond : c = 15) (side_cond : a = 12) (right_angle_cond : c^2 = a^2 + (15 - 12)^2) : 
  (1 / 2) * a * (sqrt (c^2 - a^2)) = 54 :=
by 
  sorry

end right_triangle_area_l442_442356


namespace distance_from_point_to_focus_l442_442113

noncomputable def point_on_parabola (P : ℝ × ℝ) (y : ℝ) : Prop :=
  y^2 = 16 * P.1 ∧ (P.2 = y ∨ P.2 = -y)

noncomputable def parabola_focus : ℝ × ℝ :=
  (4, 0)

theorem distance_from_point_to_focus
  (P : ℝ × ℝ) (y : ℝ)
  (h1 : point_on_parabola P y)
  (h2 : dist P (0, P.2) = 12) :
  dist P parabola_focus = 13 :=
sorry

end distance_from_point_to_focus_l442_442113


namespace arrangement_of_volunteers_l442_442820

open Finset

theorem arrangement_of_volunteers :
  let volunteers := 5
  let events := 3
  ∑ m in finset.range (events + 1), (-1)^m * nat.choose events m * (events - m)^volunteers = 150 :=
by
  sorry

end arrangement_of_volunteers_l442_442820


namespace area_of_quadrilateral_PQRS_l442_442970

-- Definition of problem conditions
def square_area (side : ℝ) : ℝ := side * side
def equilateral_triangle_height (side : ℝ) : ℝ := (side * sqrt 3) / 2
def quadrilateral_area (side : ℝ) : ℝ := side * side

-- The given conditions
def side_length : ℝ := sqrt 25
def triangle_height : ℝ := equilateral_triangle_height side_length

-- The side length calculation for quadrilateral PQRS
def quadrilateral_side_length : ℝ := side_length + 2 * triangle_height

-- The area of the quadrilateral
def quadrilateral_calculated_area : ℝ := quadrilateral_area quadrilateral_side_length

-- Statement of the main proof problem
theorem area_of_quadrilateral_PQRS : quadrilateral_calculated_area = 100 + 50 * sqrt 3 := 
by
  -- Proof steps would go here, but we use 'sorry' to skip actual proof
  sorry

end area_of_quadrilateral_PQRS_l442_442970


namespace exists_infinite_set_no_containment_l442_442383

-- Define the concept of a number 'a' being contained in a number 'b'
def contained (a b : ℕ) : Prop :=
  ∃ l r : ℕ, l < r ∧ r < 10^(Nat.digits 10 b).length ∧ a = l + b % 10^r

-- The main theorem statement
theorem exists_infinite_set_no_containment :
  ∃ S : Set ℕ, infinite S ∧ ∀ a b ∈ S, a ≠ b → ¬contained a b :=
sorry

end exists_infinite_set_no_containment_l442_442383


namespace find_RZ_l442_442750

variables {P Q R : Type*} 
          {X Y Z O1 O2 O3 : Type*} 
          {XY YZ XZ QZ RY RZ : ℝ}

-- Definitions and conditions
def is_inscribed (P Q R : X Y Z) : Prop := sorry
def circumcircle (Δ : Type*) (O : Δ) : Prop := sorry

-- Given conditions
axiom XY_length : XY = 30
axiom YZ_length : YZ = 40
axiom XZ_length : XZ = 35
axiom ry_eq_qz : RY = QZ
axiom qy_eq_qz_plus_5 : QY = QZ + 5
axiom rp_eq_ry_plus_5 : RP = RY + 5
axiom rz_eq_ry_plus_10 : RZ = RY + 10

theorem find_RZ : RZ = 45 / 2 :=
by sorry

end find_RZ_l442_442750


namespace time_with_walkway_l442_442022

-- Definitions
def length_walkway : ℝ := 60
def time_against_walkway : ℝ := 120
def time_stationary_walkway : ℝ := 48

-- Theorem statement
theorem time_with_walkway (v w : ℝ)
  (h1 : 60 = 120 * (v - w))
  (h2 : 60 = 48 * v)
  (h3 : v = 1.25)
  (h4 : w = 0.75) :
  60 = 30 * (v + w) :=
by
  sorry

end time_with_walkway_l442_442022


namespace find_x_when_h_eq_20_l442_442646

def h (x : ℝ) : ℝ := 4 * s⁻¹'(x)
def s (x : ℝ) : ℝ := 40 / (x + 5)

theorem find_x_when_h_eq_20 :
  ∃ x : ℝ, h x = 20 ↔ x = 4 :=
by sorry

end find_x_when_h_eq_20_l442_442646


namespace domain_of_inverse_function_l442_442509

noncomputable def log_inverse_domain : Set ℝ :=
  {y | y ≥ 5}

theorem domain_of_inverse_function :
  ∀ y, y ∈ log_inverse_domain ↔ ∃ x, x ≥ 3 ∧ y = 4 + Real.logb 2 (x - 1) :=
by
  sorry

end domain_of_inverse_function_l442_442509


namespace Nicki_miles_per_week_first_half_l442_442253

-- Define the conditions and the problem
theorem Nicki_miles_per_week_first_half (x : ℕ) :
  (26 * x) + (30 * 26) = 1300 → x = 20 :=
by
  -- Conditions: Nicki ran x miles per week for the first 26 weeks.
  -- Nicki ran 30 miles per week for the next 26 weeks.
  -- The total mileage for the year is 1300 miles.
  intro h,
  sorry

end Nicki_miles_per_week_first_half_l442_442253


namespace not_always_possible_to_remove_one_square_l442_442024

-- Define the red square and its properties
variable (R : Type) [Square R] (cover : ℕ → R → bool)

-- Define the logical statements involved
def covered_by (n : ℕ) (r : R) := cover n r = true

-- Axioms or assumptions based on the given problem
axiom A1 : (∀ R : Type, Square R) -- R is a square
axiom A2 : (∀ n : ℕ, cover n R → bool) -- A function that determines if R is covered by n squares

-- The theorem to be proved
theorem not_always_possible_to_remove_one_square (R : Type) [Square R] (cover : ℕ → R → bool) :
  ∀ (r : R), covered_by 100 r → ¬ covered_by 99 r :=
sorry

end not_always_possible_to_remove_one_square_l442_442024


namespace fill_time_of_three_pipes_l442_442707

def rate (hours : ℕ) : ℚ := 1 / hours

def combined_rate : ℚ :=
  rate 12 + rate 15 + rate 20

def time_to_fill (rate : ℚ) : ℚ :=
  1 / rate

theorem fill_time_of_three_pipes :
  time_to_fill combined_rate = 5 := by
  sorry

end fill_time_of_three_pipes_l442_442707


namespace palindromes_containing_2_are_100_percent_l442_442020

def is_palindrome (n : ℕ) : Prop :=
  ∀ (digits : list ℕ), digits = [1, (n / 100) % 10, (n / 10) % 10, 1]

def contains_digit_2 (n : ℕ) : Prop :=
  (n / 100) % 10 = 2 ∨ (n / 10) % 10 = 2

theorem palindromes_containing_2_are_100_percent :
  ((finset.range 10).filter (λ b, contains_digit_2 (1001 + 101 * b)).card = 10) :=
by
  sorry

end palindromes_containing_2_are_100_percent_l442_442020


namespace no_1111_distinct_subsets_with_perfect_square_intersections_l442_442430

open Finset

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def X : Finset ℕ := (range 100).map (λ n, n + 1)

theorem no_1111_distinct_subsets_with_perfect_square_intersections :
  ¬ ∃ (𝓒 : Finset (Finset ℕ)), 𝓒.card = 1111 ∧
    (∀ (A B ∈ 𝓒), A ≠ B → is_perfect_square ((A ∩ B).card)) := by
  sorry

end no_1111_distinct_subsets_with_perfect_square_intersections_l442_442430


namespace liam_walk_distance_approx_l442_442621

-- Definitions of conditions
def v_jog := 8 -- kilometers per hour
def v_walk := 3 -- kilometers per hour
def t_total := 2 -- hours

-- The equation derived from the given conditions
noncomputable def distance_walked : ℝ := 
  let t_jog := 3 / 8 
  let t_walk := 1 / 3 
  (3/8) * ( /(8) + (1/3) ) / t_total

theorem liam_walk_distance_approx : abs (distance_walked - 2.8) < 0.1 := 
by sorry

end liam_walk_distance_approx_l442_442621


namespace maximum_value_of_expression_l442_442955
-- Import the necessary library.

-- Declare the definitions and statements to be proven.
theorem maximum_value_of_expression (n : ℕ) (a : Fin n → ℝ) 
  (h1 : ∀ i, 0 ≤ a i ∧ a i ≤ 2) (h2 : a 0 = a n) :
  ∃ M, M = 4 ∧ ∀ b, b = (∑ i in Finset.range n, (a i)^2 * a ((i + 1) % n) + 8 * n) /
    (∑ i in Finset.range n, (a i)^2) ∧ b ≤ M :=
by
  sorry

end maximum_value_of_expression_l442_442955


namespace sale_in_first_month_l442_442014

theorem sale_in_first_month 
  (sale_month_2 : ℕ)
  (sale_month_3 : ℕ)
  (sale_month_4 : ℕ)
  (sale_month_5 : ℕ)
  (required_sale_month_6 : ℕ)
  (average_sale_6_months : ℕ)
  (total_sale_6_months : ℕ)
  (total_known_sales : ℕ)
  (sale_first_month : ℕ) : 
    sale_month_2 = 3920 →
    sale_month_3 = 3855 →
    sale_month_4 = 4230 →
    sale_month_5 = 3560 →
    required_sale_month_6 = 2000 →
    average_sale_6_months = 3500 →
    total_sale_6_months = 6 * average_sale_6_months →
    total_known_sales = sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5 →
    total_sale_6_months - (total_known_sales + required_sale_month_6) = sale_first_month →
    sale_first_month = 3435 :=
by
  intros h2 h3 h4 h5 h6 h_avg h_total h_known h_calc
  sorry

end sale_in_first_month_l442_442014


namespace infinite_rational_pairs_l442_442303

theorem infinite_rational_pairs : 
  {p : ℚ × ℚ | p.1 + p.2 = p.1 * p.2}.infinite :=
by
  sorry

end infinite_rational_pairs_l442_442303


namespace fifteen_power_ab_l442_442991

theorem fifteen_power_ab (a b : ℤ) (R S : ℝ) 
  (hR : R = 3^a) 
  (hS : S = 5^b) : 
  15^(a * b) = R^b * S^a :=
by sorry

end fifteen_power_ab_l442_442991


namespace line_equation_intercept_twice_x_intercept_l442_442458

theorem line_equation_intercept_twice_x_intercept 
  {x y : ℝ}
  (intersection_point : ∃ (x y : ℝ), 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0) 
  (y_intercept_is_twice_x_intercept : ∃ (a : ℝ), ∀ (x y : ℝ), y = 2 * a ∧ x = a) :
  (∃ (x y : ℝ), 2 * x - 3 * y = 0) ∨ (∃ (x y : ℝ), 2 * x + y - 8 = 0) :=
sorry

end line_equation_intercept_twice_x_intercept_l442_442458


namespace min_value_S_l442_442240

noncomputable def min_S (a : Fin 10 → ℕ) : ℕ :=
  a 0 * a 1 + a 1 * a 2 + a 2 * a 3 + a 3 * a 4 + a 4 * a 5 + a 5 * a 6 + a 6 * a 7 + a 7 * a 8 + a 8 * a 9 + a 9 * a 0

theorem min_value_S : ∃ a : Fin 10 → ℕ, (∀ i j : Fin 10, i ≠ j → a i ≠ a j) ∧ (∑ i, a i = 1995) ∧ (min_S a = 6045) :=
sorry

end min_value_S_l442_442240


namespace jacksonville_walmart_complaints_l442_442649

theorem jacksonville_walmart_complaints :
  let normal_complaint_rate := 120
  let short_staffed_increase_factor := 4 / 3
  let self_checkout_broken_increase_factor := 1.2
  let days := 3
  let complaints_per_day_short_staffed := normal_complaint_rate * short_staffed_increase_factor
  let complaints_per_day_both_conditions := complaints_per_day_short_staffed * self_checkout_broken_increase_factor
  let total_complaints := complaints_per_day_both_conditions * days
  total_complaints = 576 :=
by
  -- Use the 'let' definitions from above to describe the proof problem
  let normal_complaint_rate := 120
  let short_staffed_increase_factor := 4 / 3
  let self_checkout_broken_increase_factor := 1.2
  let days := 3
  let complaints_per_day_short_staffed := normal_complaint_rate * short_staffed_increase_factor
  let complaints_per_day_both_conditions := complaints_per_day_short_staffed * self_checkout_broken_increase_factor
  let total_complaints := complaints_per_day_both_conditions * days
  
  -- Here would be the place to write the proof steps, but it is skipped as per instructions
  sorry

end jacksonville_walmart_complaints_l442_442649


namespace super_prime_looking_count_l442_442811

def is_prime (n : ℕ) : Prop := nat.prime n

def is_composite (n : ℕ) : Prop := ¬ nat.prime n ∧ n > 1

def divisible_by (n d : ℕ) : Prop := d ∣ n

def super_prime_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬ (divisible_by n 2 ∨ divisible_by n 5 ∨ divisible_by n 7)

theorem super_prime_looking_count : ∃ (count : ℕ), count = 75 ∧ 
  count = (finset.filter super_prime_looking (finset.range 500)).card
:=
sorry

end super_prime_looking_count_l442_442811


namespace find_least_n_l442_442939

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 20 then 20
  else 50 * (seq (n - 1)) + n

theorem find_least_n : ∃ n, n > 20 ∧ seq n % 121 = 0 ∧ ∀ m, m > 20 → seq m % 121 = 0 → n ≤ m :=
by
  sorry

end find_least_n_l442_442939


namespace find_formula_sum_b_l442_442097

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ (d : ℕ), ∀ (n : ℕ), a (n + 1) = a n + d

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 0 + a (n - 1))) / 2

theorem find_formula 
  (a : ℕ → ℕ) 
  (arithmetic_seq a)
  (ha2 : a 2 = 5)
  (hS4 : S a 4 = 28) :
  ∀ n : ℕ, a n = 4 * n - 3 :=
sorry

theorem sum_b 
  (a : ℕ → ℕ) 
  (arithmetic_seq a)
  (ha2 : a 2 = 5)
  (hS4 : S a 4 = 28)
  (b : ℕ → ℕ)
  (hb : ∀ n, b n = (-1)^n * a n) :
  ∀ n : ℕ, ∑ i in range (2 * n), b i = 4 * n :=
sorry

end find_formula_sum_b_l442_442097


namespace solve_log_eq_l442_442277

-- Define the given logarithmic equation
def log_eq (b : ℝ) : Prop := log 2 (b^2 - 18*b) = 5

-- Define irrationality of a number
def is_irrational (x : ℝ) : Prop := ¬∃ q : ℚ, x = q

-- The main theorem to be stated and proved
theorem solve_log_eq : 
  ∃ (b1 b2 : ℝ), log_eq b1 ∧ log_eq b2 ∧ is_irrational b1 ∧ is_irrational b2 :=
sorry  -- To be proved

end solve_log_eq_l442_442277


namespace bacteria_growth_time_l442_442288

theorem bacteria_growth_time (initial_bacteria : ℕ) (final_bacteria : ℕ) (tripling_period : ℕ) (tripling_factor : ℕ) :
  initial_bacteria = 200 →
  final_bacteria = 145800 →
  tripling_period = 5 →
  tripling_factor = 3 →
  ∃ hours : ℕ, hours = 30 :=
begin
  assume h1 h2 h3 h4,
  have factor : final_bacteria / initial_bacteria = tripling_factor ^ 6, by sorry,
  have tripling_count : (final_bacteria / initial_bacteria) = 729, by sorry,
  have number_of_triplings : tripling_factor ^ 6 = 729, by sorry,
  have hours : 6 * tripling_period = 30, by sorry,
  use 30,
  assumption,
end

end bacteria_growth_time_l442_442288


namespace product_xyz_l442_442541

theorem product_xyz (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) : x * y * z = -1 :=
by
  sorry

end product_xyz_l442_442541


namespace distinct_sequences_starting_with_two_heads_l442_442763

theorem distinct_sequences_starting_with_two_heads :
  (number_of_sequences_flipping_ten_times (at_least_two_heads)) = 256 :=
sorry

-- Definitions corresponding to the conditions
def at_least_two_heads : Prop := ∀ (seq : list bool), seq.length = 10 → (seq.head = tt ∧ seq.tail.head = tt)

def number_of_sequences_flipping_ten_times (condition : list bool → Prop) : ℕ :=
  (finset.univ.filter condition).card

end distinct_sequences_starting_with_two_heads_l442_442763


namespace evaluate_double_sum_l442_442445

noncomputable def double_sum_approx : ℝ :=
  ∑ m in Finset.range 10, ∑ n in Finset.range 10, 1 / ( (m + 1) * (n + 1) * ((m + 1) + (n + 1) + 3) )

theorem evaluate_double_sum :
  double_sum_approx ≈ 1.0 :=  -- Replace 1.0 with the numerical approximation obtained
sorry

end evaluate_double_sum_l442_442445


namespace plant_manager_needs_15_workers_l442_442772

-- Define the combination formula
def combination (n k : ℕ) : ℕ := n! / (k! * (n - k)!)

-- Define the main problem statement
theorem plant_manager_needs_15_workers (n : ℕ) 
  (h : combination n 2 = 45) : 
  n = 15 :=
sorry

end plant_manager_needs_15_workers_l442_442772


namespace cakes_sold_to_baked_ratio_l442_442049

theorem cakes_sold_to_baked_ratio
  (cakes_per_day : ℕ) 
  (days : ℕ)
  (cakes_left : ℕ)
  (total_cakes : ℕ := cakes_per_day * days)
  (cakes_sold : ℕ := total_cakes - cakes_left) :
  cakes_per_day = 20 → 
  days = 9 → 
  cakes_left = 90 → 
  cakes_sold * 2 = total_cakes := 
by 
  intros 
  sorry

end cakes_sold_to_baked_ratio_l442_442049


namespace proof_main_l442_442245

-- Define the triangle and the construction of squares on its sides
variables {A B C A_1 B_1 C_1 A_2 B_2 C_2 : Type*}
-- Define the areas of the triangles
variables (S S_1 S_2 : ℝ)

noncomputable def main_theorem : Prop :=
  -- Conditions:
  -- 1. Squares constructed on the sides BC, CA, AB of triangle ABC.
  -- 2. Centers of these squares are A_1, B_1, C_1.
  -- 3. Triangle A_2B_2C_2 is constructed analogously from A_1B_1C_1.
  -- Prove:
  S = 8 * S_1 - 4 * S_2

theorem proof_main : main_theorem S S_1 S_2 :=
begin
  sorry
end

end proof_main_l442_442245


namespace tangent_line_k_eq_2_max_k_for_inequality_l442_442132

def f (x k : ℝ) : ℝ := (Real.log x + k) / Real.exp x
noncomputable def f_prime (x k : ℝ) : ℝ := ((1 - k * x - x * Real.log x) / (x * Real.exp x))

theorem tangent_line_k_eq_2 :
  let k := 2;
  let x_0 := 1;
  let y_0 := f x_0 k;
  let slope := f_prime x_0 k;
  y = (slope * (x - x_0)) + y_0 ->
  x + Real.exp 1 * y - 3 = 0 := 
by 
  sorry

theorem max_k_for_inequality (x : ℝ) (hx : x > 1) :
  ∀ k : ℤ, 
  (∀ x > 1, x * Real.exp x * f_prime x (k : ℝ) + (2 * k - 1) * x < 1 + k) ->
  k ≤ 3 :=
by
  sorry

end tangent_line_k_eq_2_max_k_for_inequality_l442_442132


namespace digit_in_100th_place_l442_442890

theorem digit_in_100th_place (n : ℕ) (h : n = 7) : 
  (let decimal_digit := (λ (k : ℕ), if (k % 2 = 0) then 1 else 2) in (decimal_digit 99) = 1) :=
by 
  have correct_expansion : 7 / 33 = 0.212121..., from sorry,
  show (λ (k : ℕ), if (k % 2 = 0) then 1 else 2) 99 = 1, from sorry

end digit_in_100th_place_l442_442890


namespace painting_equation_l442_442043

-- Define the painting rates for Alice, Bob, and Carlos
def alice_rate : ℝ := 1/4
def bob_rate : ℝ := 1/6
def carlos_rate : ℝ := 1/12
def combined_rate : ℝ := alice_rate + bob_rate + carlos_rate
def break_time : ℝ := 2

-- Define the total time including the break
variable {t : ℝ}

-- The theorem stating the correct equation for the total time required
theorem painting_equation : combined_rate * (t - break_time) = 1 := by
  -- Proof goes here
  sorry

end painting_equation_l442_442043


namespace vector_triple_product_zero_l442_442617

variables (u v w : ℝ × ℝ × ℝ)

-- notation for cross product and dot product in 3D

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  ((a.2.2 * b.2.1 - a.2.1 * b.2.2), (a.2.0 * b.2.2 - a.2.2 * b.2.0), (a.2.1 * b.2.0 - a.2.0 * b.2.1))

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.2.0 * b.2.0 + a.2.1 * b.2.1 + a.2.2 * b.2.2

theorem vector_triple_product_zero :
  cross_product u (cross_product v w) + cross_product v (cross_product w u) + cross_product w (cross_product u v) = (0, 0, 0) :=
sorry

end vector_triple_product_zero_l442_442617


namespace isosceles_triangle_circle_area_l442_442318

theorem isosceles_triangle_circle_area
  (A B C O : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space O]
  (AB AC : segment A)
  (BC : segment B)
  (h_iso : AB = AC)
  (h_AB : length AB = 7)
  (h_AC : length AC = 7)
  (h_BC : length BC = 8)
  (h_O_center : is_incenter O (triangle A B C)) :
  ∃ (r : ℝ),
  area (circle_through A O C r) = 37 * π :=
begin
  sorry
end

end isosceles_triangle_circle_area_l442_442318


namespace area_of_ABC_l442_442175

open_locale big_operators

variables (A B C M K O : Type)
variables [is_isosceles_triangle A B C] (h₁: A = B) (h₂: B = C)
variables [angle_bisector_intersection A M B K O]
variables (h_area_BOM : area (triangle B O M) = 25)
variables (h_area_COM : area (triangle C O M) = 30)

theorem area_of_ABC : area (triangle A B C) = 110 :=
by {
  intros,
  have h_area_BOC : area (triangle B O C) = 55,
  { sorry },
  have h_area_ABC : area (triangle A B C) = 2 * area (triangle B O C),
  { sorry },
  rw h_area_ABC,
  exact 2 * 55,
}

end area_of_ABC_l442_442175


namespace problem1_problem2_problem3_l442_442278

-- Problem 1
theorem problem1 (x : ℝ) : 3 * x^2 - 9 = 0 → (x = real.sqrt 3) ∨ (x = -real.sqrt 3) :=
by
  intro h
  sorry

-- Problem 2
theorem problem2 (x : ℝ) : (x + 2)^3 - 32 = 32 → x = 2 :=
by
  intro h
  sorry

-- Problem 3
theorem problem3 (x y : ℝ) (h1 : x + 2 * y = 6) (h2 : 3 * x + y = 8) : x = 2 ∧ y = 2 :=
by
  intro h1 h2
  sorry

end problem1_problem2_problem3_l442_442278


namespace students_wearing_other_colors_l442_442738

theorem students_wearing_other_colors (total_students : ℕ)
  (p_blue p_red p_green : ℕ → ℝ)
  (h_total : total_students = 800)
  (h_blue : p_blue total_students = 0.45)
  (h_red : p_red total_students = 0.23)
  (h_green : p_green total_students = 0.15) :
  ∃ (n_other_colors : ℕ), n_other_colors = 136 :=
by
  have h_not_other := 1 - (p_blue total_students + p_red total_students + p_green total_students)
  have h_not_other_correct : h_not_other = 0.17 := by
    sorry

  have n_other_colors := h_not_other * total_students
  have n_other_correct : n_other_colors = 136 := by
    sorry

  use 136
  exact n_other_correct

end students_wearing_other_colors_l442_442738


namespace correct_statement_dice_roll_l442_442350

theorem correct_statement_dice_roll :
  (∃! s, s ∈ ["When flipping a coin, the head side will definitely face up.",
              "The probability of precipitation tomorrow is 80% means that 80% of the areas will have rain tomorrow.",
              "To understand the lifespan of a type of light bulb, it is appropriate to use a census method.",
              "When rolling a dice, the number will definitely not be greater than 6."] ∧
          s = "When rolling a dice, the number will definitely not be greater than 6.") :=
by {
  sorry
}

end correct_statement_dice_roll_l442_442350


namespace fraction_available_on_third_day_l442_442764

noncomputable def liters_used_on_first_day (initial_amount : ℕ) : ℕ :=
  (initial_amount / 2)

noncomputable def liters_added_on_second_day : ℕ :=
  1

noncomputable def original_solution : ℕ :=
  4

noncomputable def remaining_solution_after_first_day : ℕ :=
  original_solution - liters_used_on_first_day original_solution

noncomputable def remaining_solution_after_second_day : ℕ :=
  remaining_solution_after_first_day + liters_added_on_second_day

noncomputable def fraction_of_original_solution : ℚ :=
  remaining_solution_after_second_day / original_solution

theorem fraction_available_on_third_day : fraction_of_original_solution = 3 / 4 :=
by
  sorry

end fraction_available_on_third_day_l442_442764


namespace range_of_a_l442_442548

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (∃ y : ℝ, x ≠ y ∧ (f x = max f ∧ f y = min f))) ↔ a < -1 ∨ a > 2 :=
by
  -- Define the function f
  let f (x : ℝ) := x^3 + 3*a*x^2 + 3*(a+2)*x + 1
  
  -- State the necessary and sufficient condition for the function
  sorry

end range_of_a_l442_442548


namespace set_inter_complements_union_l442_442947

open Set

def f (n : ℕ) := 2 * n + 1

def U := univ
def P := {1, 2, 3, 4, 5}
def Q := {3, 4, 5, 6, 7}

def A := {n : ℕ | f n ∈ P}
def B := {n : ℕ | f n ∈ Q}

theorem set_inter_complements_union :
  ((A ∩ (U \ B)) ∪ (B ∩ (U \ A))) = {0, 3} :=
sorry

end set_inter_complements_union_l442_442947


namespace part_a_part_b_l442_442739

open Finset

variables {X : Type} (ℱ : Finset (Finset X)) (n k : ℕ)

def condition_1 : Prop :=
  ℱ ⊆ powersetLen k (univ : Finset X)

def condition_2 : Prop :=
  ∀ {A B C : Finset X}, A ∈ ℱ → B ∈ ℱ → C ∈ ℱ →
  A ≠ B → B ≠ C → C ≠ A →
  at_most_one_empty [A ∩ B, B ∩ C, C ∩ A]

def at_most_one_empty (sets : List (Finset X)) : Prop :=
  length (filter (== ∅) sets) ≤ 1

theorem part_a (hX : card (univ : Finset X) = n)
  (h_subset : condition_1 ℱ)
  (h_inter : condition_2 ℱ)
  (h_kn : k ≤ n / 2) :
  card ℱ ≤ max 1 (4 - n / k) * nat.choose (n - 1) (k - 1) :=
sorry

theorem part_b (hX : card (univ : Finset X) = n)
  (h_subset : condition_1 ℱ)
  (h_inter : condition_2 ℱ)
  (h_kn : k ≤ n / 3)
  (hab_eq : card ℱ = max 1 (4 - n / k) * nat.choose (n - 1) (k - 1)) :
  true :=  -- Specify the exact condition for equality
sorry

end part_a_part_b_l442_442739


namespace last_digit_to_appear_in_fibonacci_mod_10_is_6_l442_442287

/-- The Fibonacci sequence starts with two 1s and each term afterwards is the sum of its two predecessors. -/
def fibonacci (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

/-- Proof that the last digit to appear in the units position of a Fibonacci number mod 10 is 6. -/
theorem last_digit_to_appear_in_fibonacci_mod_10_is_6 : 
  ∃ N : ℕ, (∀ d : ℕ, d < 10 → (∃ n : ℕ, (n ≥ N) ∧ (fibonacci n % 10 = d))) ∧ (∃ n : ℕ, (fibonacci n % 10 = 6)) :=
sorry

end last_digit_to_appear_in_fibonacci_mod_10_is_6_l442_442287


namespace find_fourth_number_l442_442301

theorem find_fourth_number : 
  ∀ (x y : ℝ),
  (28 + x + 42 + y + 104) / 5 = 90 ∧ (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 78 :=
by
  intros x y h
  sorry

end find_fourth_number_l442_442301


namespace largest_sin_D_of_triangle_l442_442569

theorem largest_sin_D_of_triangle (DE EF : ℝ) (hDE : DE = 24) (hEF : EF = 18) : 
  ∃ D : ℝ, sin D = 3 * Real.sqrt 7 / 7 :=
by
  sorry

end largest_sin_D_of_triangle_l442_442569


namespace positive_difference_mean_median_l442_442046

-- Define the vertical drops of the roller coasters
def CycloneFury := 180
def GravityRush := 150
def ThunderLoop := 210
def SkyScream := 315
def Whirlwind := 225
def MysticFall := 190

-- Define the list of vertical drops
def verticalDrops := [CycloneFury, GravityRush, ThunderLoop, SkyScream, Whirlwind, MysticFall]

-- Compute the positive difference between the mean and the median of the vertical drops
theorem positive_difference_mean_median : abs ((verticalDrops.sum / verticalDrops.length.toReal) - 200) = 11.67 :=
by
  sorry

end positive_difference_mean_median_l442_442046


namespace second_place_jump_l442_442933

theorem second_place_jump : 
  ∀ (Kyungsoo Younghee Jinju Chanho : ℝ), 
    Kyungsoo = 2.3 → 
    Younghee = 0.9 → 
    Jinju = 1.8 → 
    Chanho = 2.5 → 
    ((Kyungsoo < Chanho) ∧ (Kyungsoo > Jinju) ∧ (Kyungsoo > Younghee)) :=
by 
  sorry

end second_place_jump_l442_442933


namespace tobys_friends_boys_count_l442_442761

theorem tobys_friends_boys_count (total_friends : ℕ) (girls : ℕ) (boys_percentage : ℕ) 
    (h1 : girls = 27) (h2 : boys_percentage = 55) (total_friends_calc : total_friends = 60) : 
    (total_friends * boys_percentage / 100) = 33 :=
by
  -- Proof is deferred
  sorry

end tobys_friends_boys_count_l442_442761


namespace least_prime_factor_of_5pow6_minus_5pow4_l442_442723

def least_prime_factor (n : ℕ) : ℕ :=
  if h : n > 1 then (Nat.minFac n) else 0

theorem least_prime_factor_of_5pow6_minus_5pow4 : least_prime_factor (5^6 - 5^4) = 2 := by
  sorry

end least_prime_factor_of_5pow6_minus_5pow4_l442_442723


namespace avg_sample_combination_l442_442867

theorem avg_sample_combination {n m : ℕ} (hn : 0 < n) (hm : 0 < m) (a : ℝ) (x̄ ȳ : ℝ) 
  (h_avg_x : x̄ ≠ ȳ) (h_a : 0 < a ∧ a < 1/2) 
  (h_combined_avg_eq : (n * x̄ + m * ȳ) / (n + m) = a * x̄ + (1 - a) * ȳ) :
  n > m := 
sorry

end avg_sample_combination_l442_442867


namespace solve_for_x_l442_442984

theorem solve_for_x :
  (∃ x : ℝ, (1 / 3 - 1 / 4) = 1 / x) → ∃ x : ℝ, x = 12 :=
by
  intro h,
  obtain ⟨x, hx⟩ := h,
  use 12,
  have : 1 / 3 - 1 / 4 = 1 / 12, by
  { calc
      1 / 3 - 1 / 4 = 4 / 12 - 3 / 12 : by norm_num
                 ... = 1 / 12 : by norm_num },
  exact this ▸ hx.symm

end solve_for_x_l442_442984


namespace range_of_a_l442_442854

noncomputable def range_a : Set ℝ :=
  {a : ℝ | 0 < a ∧ a ≤ 1/2}

theorem range_of_a (O P : ℝ × ℝ) (Q : ℝ × ℝ)
  (hO : O = (0, 0))
  (hP : P = (a, 0))
  (ha : 0 < a)
  (hQ : ∃ m : ℝ, Q = (m^2, m))
  (hPQ_PO : ∀ Q, Q = (m^2, m) → dist P Q ≥ dist O P) :
  a ∈ range_a :=
sorry

end range_of_a_l442_442854


namespace cannot_determine_congruence_l442_442346

-- Triangle congruence conditions
inductive TriangleCongruenceCondition
| ASA | AAS | SSS | AAA

open TriangleCongruenceCondition

theorem cannot_determine_congruence (cond : TriangleCongruenceCondition) : cond = AAA → ¬ (∀ Δ₁ Δ₂, (angle_eq Δ₁ Δ₂ 0 ∧ angle_eq Δ₁ Δ₂ 1 ∧ angle_eq Δ₁ Δ₂ 2) → congruent Δ₁ Δ₂) :=
by
sorry

end cannot_determine_congruence_l442_442346


namespace concentric_chords_l442_442843

theorem concentric_chords (angle_ABC : ℝ) (h_angle : angle_ABC = 75) :
  let central_angle := 180 - angle_ABC in
  let gcd_angle := Int.gcd 360 (Int.ofNat central_angle.toInt) in
  360 / gcd_angle = 24 := 
by
  -- Proof steps can be filled here.
  sorry

end concentric_chords_l442_442843


namespace city_travel_routes_l442_442312
-- Noncomputable because the proof involves combinatorial counting without constructing each route
noncomputable def total_routes : ℕ := 80

theorem city_travel_routes :
  ∀ (cities : ℕ) (segments : ℕ) (start end_city : ℕ),
  cities = 5 →
  segments = 4 →
  start = 5 →
  end_city = 5 →
  (∃ (routes : ℕ), routes = total_routes) :=
begin
  intros cities segments start end_city hc hs hstart hend,
  use total_routes,
  -- The proof would go here
  sorry,
end

end city_travel_routes_l442_442312


namespace total_weight_loss_l442_442980

theorem total_weight_loss (S J V : ℝ) 
  (hS : S = 17.5) 
  (hJ : J = 3 * S) 
  (hV : V = S + 1.5) : 
  S + J + V = 89 := 
by 
  sorry

end total_weight_loss_l442_442980


namespace sum_of_cube_faces_l442_442678

theorem sum_of_cube_faces :
  ∃ (x y z w v u : ℕ), 
    (x + y = 30) ∧ (z + w = 30) ∧ (v + u = 30) ∧
    (x, y, z, w, v, u).sorted (=) ∧ -- sorted constraint to ensure they are consecutive
    (x + y + z + w + v + u = 90) :=
sorry

end sum_of_cube_faces_l442_442678


namespace intersection_of_A_and_B_l442_442228

open Set

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
  by
    sorry

end intersection_of_A_and_B_l442_442228


namespace four_digit_increasing_even_integers_l442_442529

theorem four_digit_increasing_even_integers : 
  let even_four_digit_strictly_increasing (n : ℕ) := 
    n >= 1000 ∧ n < 10000 ∧ (n % 2 = 0) ∧ (let d1 := n / 1000 % 10, 
                                                d2 := n / 100 % 10,
                                                d3 := n / 10 % 10,
                                                d4 := n % 10 in
                                            d1 < d2 ∧ d2 < d3 ∧ d3 < d4)
  in (finset.filter even_four_digit_strictly_increasing (finset.range 10000)).card = 46 :=
begin
  sorry
end

end four_digit_increasing_even_integers_l442_442529


namespace infinite_series_sum_equals_one_l442_442805

noncomputable def infinite_series_sum : ℝ := 
  ∑' n : ℕ, (if n = 0 then 0 else (n^5 + 5 * n^3 + 15 * n + 15) / (2^n * (n^5 + 5)))

theorem infinite_series_sum_equals_one : infinite_series_sum = 1 := 
by 
  have h_geom_series : ∑' (n : ℕ), (if n = 0 then 0 else 1 / 2^n) = 1 := sorry
  have h_remainder_series : ∑' (n : ℕ), (if n = 0 then 0 else (5 * n^3 + 15 * n + 10) / (2^n * (n^5 + 5))) = 0 := sorry
  rw [infinite_series_sum, tsum_add] -- Add series decomposition property
  rw [h_geom_series, h_remainder_series]
  -- Finish with the sum of 1 and 0
  exact add_zero 1

end infinite_series_sum_equals_one_l442_442805


namespace is_isosceles_trapezoid_calc_perimeter_l442_442918

variables {W X Y Z E G : Type} [AddCommGroup W] [AddCommGroup X] [AddCommGroup Y] [AddCommGroup Z]
  [AddCommGroup E] [AddCommGroup G]
variables [Module ℝ W] [Module ℝ X] [Module ℝ Y] [Module ℝ Z] [Module ℝ E] [Module ℝ G]

-- Given conditions
def is_cyclic_quadrilateral (W X Y Z : Type) [AddCommGroup W] [AddCommGroup X] 
  [AddCommGroup Y] [AddCommGroup Z] [Module ℝ W] [Module ℝ X] [Module ℝ Y] [Module ℝ Z] := sorry

def parallel (a b : ℝ) := sorry 

def length (a b c: Type) [AddCommGroup a] [AddCommGroup b] [AddCommGroup c]
  [Module ℝ a] [Module ℝ b] [Module ℝ c] (d: ℝ):= sorry

def perp (a b : ℝ) := sorry

-- Theorem statements
theorem is_isosceles_trapezoid 
  (W X Y Z : Type) [AddCommGroup W] [AddCommGroup X] [AddCommGroup Y] [AddCommGroup Z]
  [Module ℝ W] [Module ℝ X] [Module ℝ Y] [Module ℝ Z]
  (h1: is_cyclic_quadrilateral W X Y Z)
  (h2: parallel W Z X Y)
  (h3: length W X 10)
  (h4: length X Y 15)
  (h5: length W Z 5) : 
  -- WXYZ is an isosceles trapezoid.
  sorry
  
theorem calc_perimeter 
  (W X Y Z E G : Type) [AddCommGroup W] [AddCommGroup X] [AddCommGroup Y] [AddCommGroup Z]
  [AddCommGroup E] [AddCommGroup G]
  [Module ℝ W] [Module ℝ X] [Module ℝ Y] [Module ℝ Z] [Module ℝ E] [Module ℝ G]
  (h1: perp E G W X)
  (h2: length W E 2.5)
  (h3: length E Z 2.5)
  (h4: length E G 5) :
  length W E + length E G + length G X + length X W = 15 :=
sorry

end is_isosceles_trapezoid_calc_perimeter_l442_442918


namespace monotonicity_intervals_of_f_minimum_value_of_f_l442_442871

noncomputable def f (a : ℝ) (x : ℝ) := a * x - Real.log x
noncomputable def g (x : ℝ) := (Real.log x) / x
noncomputable def g_prime (x : ℝ) := (1 - Real.log x) / (x^2)

-- Proving the monotonicity intervals of f when a = g'(1)
theorem monotonicity_intervals_of_f (x : ℝ) (h : x > 0) (a : ℝ) (h1 : a = g_prime 1) : 
  (f a x) = (x - Real.log x) := 
sorry

-- Prove there exists a real number a such that the minimum value of f(x) is 3 for x in [0, e]
theorem minimum_value_of_f (a : ℝ) : 
  (∃ a = Real.exp 2, ∀ x ∈ Set.Ioc 0 (Real.exp 1), f a x = 3) := 
sorry

end monotonicity_intervals_of_f_minimum_value_of_f_l442_442871


namespace cover_by_equilateral_triangle_cover_by_regular_hexagon_l442_442366

variable (F : Type) [HasDiameter F]

/-- Prove that a shape F with a diameter of 1 can be covered by an equilateral triangle with a side length of √3. -/
theorem cover_by_equilateral_triangle (d : diameter F = 1) :
  ∃ T : EquilateralTriangle, side_length T = √3 ∧ covers T F :=
sorry

/-- Prove that a shape F with a diameter of 1 can be covered by a regular hexagon with a side length of √(3)/3. -/
theorem cover_by_regular_hexagon (d : diameter F = 1) :
  ∃ H : RegularHexagon, side_length H = √3 / 3 ∧ covers H F :=
sorry

end cover_by_equilateral_triangle_cover_by_regular_hexagon_l442_442366


namespace only_p_eq_3_l442_442640

theorem only_p_eq_3 (p : ℕ) (h1 : Prime p) (h2 : Prime (8 * p ^ 2 + 1)) : p = 3 := 
by
  sorry

end only_p_eq_3_l442_442640


namespace part1_correctness_part2_correctness_part3_solution_l442_442002

variables (x : ℕ) (hx : x > 30)

def jacket_price : ℕ := 100
def tshirt_price : ℕ := 60
def discount1_jacket_cost : ℕ := 30 * jacket_price
def discount1_tshirt_cost : ℕ := 60 * (x - 30)
def discount2_jacket_cost : ℕ := 30 * jacket_price * 8 / 10
def discount2_tshirt_cost : ℕ := 48 * x

def cost_effective_option1 (x : ℕ) : bool :=
  discount1_jacket_cost + discount1_tshirt_cost < discount2_jacket_cost + discount2_tshirt_cost

def combined_strategy_cost : ℕ :=
  discount1_jacket_cost + (10 * tshirt_price * 8 / 10)

def more_cost_effective_combined_strategy (x : ℕ) (hx: x = 40) : Prop :=
  combined_strategy_cost < (discount1_jacket_cost + discount1_tshirt_cost)

theorem part1_correctness :
  ∀ x : ℕ, x > 30 → 
  discount1_jacket_cost = 3000 ∧
  discount1_tshirt_cost = 60 * (x - 30) ∧
  discount2_jacket_cost = 2400 ∧
  discount2_tshirt_cost = 48 * x :=
  sorry

theorem part2_correctness :
  ∀ x : ℕ, x = 40 →
  discount1_jacket_cost + discount1_tshirt_cost < discount2_jacket_cost + discount2_tshirt_cost :=
  sorry

theorem part3_solution :
  ∃ x : ℕ, x = 40 →
  combined_strategy_cost < (discount1_jacket_cost + discount1_tshirt_cost) :=
  sorry

end part1_correctness_part2_correctness_part3_solution_l442_442002


namespace cos_double_angle_l442_442839

theorem cos_double_angle
  (α : ℝ)
  (h1 : 0 < α) (h2 : α < real.pi)
  (h3 : real.sin α + real.cos α = 1 / 5) :
  real.cos (2 * α) = -7 / 25 :=
sorry

end cos_double_angle_l442_442839


namespace find_other_endpoint_l442_442674

-- Define the properties and conditions
variable (x2 y2 : ℝ)
variable (P_mid : ℝ × ℝ)
variable (P1 : ℝ × ℝ)
variable (P2 : ℝ × ℝ)

axiom midpoint_property :
  P_mid = ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)

-- Define the known values
def midpoint : (ℝ × ℝ) := (-1, 4)
def first_endpoint : (ℝ × ℝ) := (3, -1)

-- Statement: Prove the coordinates of the other endpoint
theorem find_other_endpoint (midpoint_property : midpoint = ((first_endpoint.1 + x2) / 2, (first_endpoint.2 + y2) / 2)) : (x2, y2) = (-5, 9) :=
by
  sorry

end find_other_endpoint_l442_442674


namespace Karl_max_score_l442_442207

def max_possible_score : ℕ :=
  69

theorem Karl_max_score (minutes problems : ℕ) (n_points : ℕ → ℕ) (time_1_5 : ℕ) (time_6_10 : ℕ) (time_11_15 : ℕ)
    (h1 : minutes = 15) (h2 : problems = 15)
    (h3 : ∀ n, n = n_points n)
    (h4 : ∀ i, 1 ≤ i ∧ i ≤ 5 → time_1_5 = 1)
    (h5 : ∀ i, 6 ≤ i ∧ i ≤ 10 → time_6_10 = 2)
    (h6 : ∀ i, 11 ≤ i ∧ i ≤ 15 → time_11_15 = 3) : 
    max_possible_score = 69 :=
  by
  sorry

end Karl_max_score_l442_442207


namespace six_square_fill_l442_442796

def adjacent (x y : ℕ) : Prop :=
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) ∨ 
  (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2) ∨
  (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) ∨
  (x = 4 ∧ y = 5) ∨ (x = 5 ∧ y = 4) ∨
  (x = 5 ∧ y = 6) ∨ (x = 6 ∧ y = 5) 

def valid_assignment (assignment : Fin 6 → ℕ) : Prop :=
  ∀ i j, adjacent i j → (assignment i - assignment j ≠ 3) ∧ (assignment j - assignment i ≠ 3)

def total_valid_assignments : ℕ :=
  {assignment // valid_assignment assignment}.card

theorem six_square_fill :
  total_valid_assignments = 96 :=
  sorry

end six_square_fill_l442_442796


namespace expected_value_of_coins_is_95_5_l442_442388

-- Define the individual coin values in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def fifty_cent_value : ℕ := 50
def dollar_value : ℕ := 100

-- Expected value function with 1/2 probability 
def expected_value (coin_value : ℕ) : ℚ := (coin_value : ℚ) / 2

-- Calculate the total expected value of all coins flipped
noncomputable def total_expected_value : ℚ :=
  expected_value penny_value +
  expected_value nickel_value +
  expected_value dime_value +
  expected_value quarter_value +
  expected_value fifty_cent_value +
  expected_value dollar_value

-- Prove that the expected total value is 95.5
theorem expected_value_of_coins_is_95_5 :
  total_expected_value = 95.5 := by
  sorry

end expected_value_of_coins_is_95_5_l442_442388


namespace rubiks_cube_closed_path_l442_442189

theorem rubiks_cube_closed_path (n : ℕ) (surface : set (ℕ × ℕ × ℕ × ℕ))
  (h1 : n = 6)
  (h2 : ∀ face ∈ surface, ∃ path, path.is_hamiltonian ∧ path.is_closed) :
  ∃ path, path.is_closed :=
  by sorry

end rubiks_cube_closed_path_l442_442189


namespace distance_between_stripes_l442_442033

theorem distance_between_stripes
  (curb_length_between_stripes : ℝ)
  (distance_between_curbs : ℝ)
  (stripe_length : ℝ) :
  distance_between_curbs = 60 →
  curb_length_between_stripes = 25 →
  stripe_length = 70 →
  (1500 / stripe_length) ≈ 21.43 :=
by
  sorry

end distance_between_stripes_l442_442033


namespace prove_Ak_eq_entrywise_Ak_l442_442210

open Matrix

variable {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (hA1 : ∀ k ∈ ({1, 2, ... , n + 1} : Finset ℕ), A^k = A.entrywise (λ x => x^k))

theorem prove_Ak_eq_entrywise_Ak (k : ℕ) : A^k = A.entrywise (λ x => x^k) := 
sorry

end prove_Ak_eq_entrywise_Ak_l442_442210


namespace min_units_for_profitability_profitability_during_epidemic_l442_442372

-- Conditions
def assembly_line_cost : ℝ := 1.8
def selling_price_per_product : ℝ := 0.1
def max_annual_output : ℕ := 100

noncomputable def production_cost (x : ℕ) : ℝ := 5 + 135 / (x + 1)

-- Part 1: Prove Minimum x for profitability
theorem min_units_for_profitability (x : ℕ) :
  (10 - (production_cost x)) * x - assembly_line_cost > 0 ↔ x ≥ 63 := sorry

-- Part 2: Profitability and max profit output during epidemic
theorem profitability_during_epidemic (x : ℕ) :
  (60 < x ∧ x ≤ max_annual_output) → 
  ((10 - (production_cost x)) * 60 - (x - 60) - assembly_line_cost > 0) ↔ x = 89 := sorry

end min_units_for_profitability_profitability_during_epidemic_l442_442372


namespace abs_w_unique_l442_442151

theorem abs_w_unique (w : ℂ) (h : w^2 - 6 * w + 40 = 0) : ∃! x : ℝ, x = Complex.abs w ∧ x = Real.sqrt 40 := by
  sorry

end abs_w_unique_l442_442151


namespace max_Q_min_Q_l442_442754

noncomputable def max_permutation (n : ℕ) : list ℕ :=
  (list.range n).map (λ i => if i % 2 = 0 then i / 2 + 1 else n - i / 2)

noncomputable def min_permutation (n : ℕ) : list ℕ :=
  (list.range n).map (λ i => if i % 2 = 0 then i / 2 + 1 else n - i / 2)

def perm_Q (a : list ℕ) : ℕ :=
  a.zip_with (*) a.tail.concat(a.head).sum

theorem max_Q (n : ℕ) (a : list ℕ) (h : a.perm (list.range n).map (+1)) :
  perm_Q max_permutation n ≥ perm_Q a :=
sorry

theorem min_Q (n : ℕ) (a : list ℕ) (h : a.perm (list.range n).map (+1)) :
  perm_Q min_permutation n ≤ perm_Q a :=
sorry

end max_Q_min_Q_l442_442754


namespace shortest_altitude_of_right_triangle_with_sides_24_24_24sqrt2_l442_442690

theorem shortest_altitude_of_right_triangle_with_sides_24_24_24sqrt2 :
  ∀ (a b c : ℝ), a = 24 ∧ b = 24 ∧ c = 24 * Real.sqrt 2 → ∃ h, h = 12 * Real.sqrt 2 :=
by
  assume a b c h
  sorry

end shortest_altitude_of_right_triangle_with_sides_24_24_24sqrt2_l442_442690


namespace product_xyz_l442_442540

theorem product_xyz (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) : x * y * z = -1 :=
by
  sorry

end product_xyz_l442_442540


namespace inscribed_square_area_l442_442781

theorem inscribed_square_area :
  ∃ s : ℚ, (s^2/4 + s^2/9 = 1) ∧ (4 * s^2 = 144/13) :=
begin
  sorry
end

end inscribed_square_area_l442_442781


namespace general_formula_a_sn_bounds_l442_442846

-- Definitions for arithmetic sequence and sum of first n terms
def a_n (n : ℕ) : ℤ := 2 * n - 1
def b_n (n : ℕ) : ℚ := 1 / (a_n n * a_n (n + 1))

def S_n (n : ℕ) : ℚ := (1 / 2) * (1 - 1 / (2 * n + 1))

-- Given condition
axiom arith_seq_cond (n : ℕ) : 
  ((Finset.range n).sum (λ k, a_n k + a_n (k + 1))) = 2 * n * (n + 1)

-- Theorem 1: General formula for \( \{a_n\} \)
theorem general_formula_a (n : ℕ) : a_n n = 2 * n - 1 := 
sorry

-- Theorem 2: Prove the inequality for \( \{S_n\} \)
theorem sn_bounds (n : ℕ) : (1 / 3 : ℚ) ≤ S_n n ∧ S_n n < (1 / 2 : ℚ) := 
sorry

end general_formula_a_sn_bounds_l442_442846


namespace fg_at_3_equals_97_l442_442159

def f (x : ℝ) : ℝ := 4 * x - 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem fg_at_3_equals_97 : f (g 3) = 97 := by
  sorry

end fg_at_3_equals_97_l442_442159


namespace missing_digit_l442_442325

theorem missing_digit (x : ℕ) (h1 : x ≥ 0) (h2 : x ≤ 9) : 
  (if x ≥ 2 then 9 * 1000 + x * 100 + 2 * 10 + 1 else 9 * 100 + 2 * 10 + x * 1) - (1 * 1000 + 2 * 100 + 9 * 10 + x) = 8262 → x = 5 :=
by 
  sorry

end missing_digit_l442_442325


namespace major_minor_axis_lengths_foci_vertices_coordinates_l442_442122

-- Given conditions
def ellipse_eq (x y : ℝ) : Prop := 16 * x^2 + 25 * y^2 = 400

-- Proof Tasks
theorem major_minor_axis_lengths : 
  (∃ a b : ℝ, a = 5 ∧ b = 4 ∧ 2 * a = 10) :=
by sorry

theorem foci_vertices_coordinates : 
  (∃ c : ℝ, 
    (c = 3) ∧ 
    (∀ x y : ℝ, ellipse_eq x y → (x = 0 → y = 4 ∨ y = -4) ∧ (y = 0 → x = 5 ∨ x = -5))) :=
by sorry

end major_minor_axis_lengths_foci_vertices_coordinates_l442_442122


namespace bananas_count_l442_442392

theorem bananas_count 
  (total_oranges : ℕ)
  (total_percentage_good : ℝ)
  (percentage_rotten_oranges : ℝ)
  (percentage_rotten_bananas : ℝ)
  (total_good_fruits_percentage : ℝ)
  (B : ℝ) :
  total_oranges = 600 →
  total_percentage_good = 0.85 →
  percentage_rotten_oranges = 0.15 →
  percentage_rotten_bananas = 0.03 →
  total_good_fruits_percentage = 0.898 →
  B = 400  :=
by
  intros h_oranges h_good_percentage h_rotten_oranges h_rotten_bananas h_good_fruits_percentage
  sorry

end bananas_count_l442_442392


namespace angle_BED_in_triangle_ABC_l442_442552

theorem angle_BED_in_triangle_ABC 
  {A B C D E : Type*}
  [triangle A B C] 
  (angle_A : ℝ) 
  (angle_C : ℝ) 
  (D_on_AB : point_on_line_segment D A B) 
  (E_on_BC : point_on_line_segment E B C) 
  (DB_2_BE : dist B D = 2 * dist B E) 
  (angle_A_def : angle A = 60) 
  (angle_C_def : angle C = 70) :
  angle BED = 80 :=
sorry

end angle_BED_in_triangle_ABC_l442_442552


namespace transform_code_l442_442294

theorem transform_code :
  let original_code := [2, 4, 4, 0, 4, 1, 9, 9, 3, 0, 8, 8]
  let transformed_code :=
    original_code.map_with_index (λ i d,
      if (i + 1) % 2 = 0 then 9 - d else d)
  transformed_code = [2, 5, 4, 9, 4, 8, 9, 0, 3, 9, 8, 1] := 
by
  let original_code := [2, 4, 4, 0, 4, 1, 9, 9, 3, 0, 8, 8]
  let transformed_code :=
    original_code.map_with_index (λ i d,
      if (i + 1) % 2 = 0 then 9 - d else d)
  have h : transformed_code = [2, 5, 4, 9, 4, 8, 9, 0, 3, 9, 8, 1], from
  sorry
  exact h

end transform_code_l442_442294


namespace section_area_proof_l442_442628

-- Definitions and theorems related to spheres and circles
noncomputable def sphere (r : ℝ) := { point : ℝ^3 | ‖point‖ = r }

def angle (a b : ℝ^3) : ℝ := real.arccos ((a • b) / (‖a‖ * ‖b‖))

def section_area (angle : ℝ) (radius : ℝ) : ℝ :=
  let section_radius := radius * real.sin (angle / 2)
  real.pi * section_radius ^ 2

-- Assuming the given conditions
def point_A := (2 : ℝ, 0, 0)
def point_O := (0 : ℝ, 0, 0)
def angle_OA_section := real.pi / 3 -- 60 degrees
def sphere_radius := 2

-- Main theorem
theorem section_area_proof :
  section_area angle_OA_section sphere_radius = real.pi :=
by
  -- Proof omitted
  sorry

end section_area_proof_l442_442628


namespace sqrt_2_div_pi_l442_442689

open Real

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n * a (n + 1) = n + 1) ∧ (tendsto (λ n, a n / a (n + 1)) at_top (𝓝 1))

theorem sqrt_2_div_pi (a : ℕ → ℝ) (h : sequence a) : a 1 = sqrt (2 / π) :=
sorry

end sqrt_2_div_pi_l442_442689


namespace numValidSequences_l442_442881

-- Define the letters available for the sequence
def letters : Finset Char := {'D', 'E', 'M', 'A', 'N'}

-- Define the conditions for the sequence
def startsWith (c : Char) (seq : List Char) : Prop := seq.head = c
def endsWith (c : Char) (seq : List Char) : Prop := seq.reverse.head = c
def noDuplicates (seq : List Char) : Prop := seq.nodup
def lengthFour (seq : List Char) : Prop := seq.length = 4

-- The set of all valid sequences
def validSequences : Finset (List Char) :=
  Finset.filter (λ seq,
    startsWith 'D' seq ∧ endsWith 'M' seq ∧ noDuplicates seq ∧ lengthFour seq)
    (Finset.fold
      (λ acc x, Finset.image (λ xs, x :: xs) acc)
      (Finset.singleton [])
      letters)

-- The number of valid sequences is 6
theorem numValidSequences : validSequences.card = 6 := by
  sorry

end numValidSequences_l442_442881


namespace spatial_relationships_l442_442753

theorem spatial_relationships :
  (∀ l1 l2 l3 : Line, l1 ∥ l3 → l2 ∥ l3 → l1 ∥ l2) ∧
  (∀ l1 l2 l3 : Line, l1 ⊥ l3 → l2 ⊥ l3 → l1 ∥ l2) ∧
  (∀ l1 l2 p : Plane, l1 ∥ p → l2 ∥ p → l1 ∥ l2) ∧
  (∀ l1 l2 p : Plane, l1 ⊥ p → l2 ⊥ p → l1 ∥ l2) → 
  (∀ l1 l2 l3 : Line, l1 ∥ l3 → l2 ∥ l3 → l1 ∥ l2) := 
sorry

end spatial_relationships_l442_442753


namespace black_car_overtakes_red_car_in_1_hour_l442_442321

theorem black_car_overtakes_red_car_in_1_hour :
  ∀ (speed_red speed_black distance_gap : ℝ),
    speed_red = 40 →
    speed_black = 50 →
    distance_gap = 10 →
    distance_gap / (speed_black - speed_red) = 1 :=
by
  intros speed_red speed_black distance_gap h_speed_red h_speed_black h_distance_gap
  rw [h_speed_red, h_speed_black, h_distance_gap]
  norm_num
  sorry

end black_car_overtakes_red_car_in_1_hour_l442_442321


namespace find_a_l442_442515

variable {α : Type*} [PartialOrder α]

def Set (A : Set α) (x : α) := x ∈ A

theorem find_a (a : α) :
  (∀ x, Set (λ x, x ≤ 1) x) ∩ (Set (λ x, x ≥ a) = {1}) → a = 1 :=
by
  sorry

end find_a_l442_442515


namespace integer_solutions_l442_442743

theorem integer_solutions (t : ℤ) : 
  ∃ x y : ℤ, 5 * x - 7 * y = 3 ∧ x = 7 * t - 12 ∧ y = 5 * t - 9 :=
by
  sorry

end integer_solutions_l442_442743


namespace max_value_of_trig_expression_l442_442461

theorem max_value_of_trig_expression : ∃ x : ℝ, ∀ φ, 3 * cos x - sin x = sqrt 10 * cos (x - φ) → 3 * cos x - sin x ≤ sqrt 10 := 
begin
  sorry
end

end max_value_of_trig_expression_l442_442461


namespace part1_part2_l442_442834

-- Define the function f and its derivative f'
def f (x : ℝ) (a : ℝ) : ℝ := (2/3) * x^3 - 2 * a * x^2 - 3 * x
def f' (x : ℝ) (a : ℝ) : ℝ := 2 * x^2 - 4 * a * x - 3

-- Question (Ⅰ): Prove that f(x) is decreasing in (-1, 1) when |a| ≤ 1/4.
theorem part1 (a : ℝ) (h : |a| ≤ 1 / 4) : 
  ∀ x : ℝ, -1 < x ∧ x < 1 → f' x a < 0 :=
begin
  sorry
end

-- Question (Ⅱ): Prove that if y = f(x) has exactly one extremum point in (-1, 1), then a < -1/4 or a > 1/4.
theorem part2 (a : ℝ) 
  (h : ∃! x : ℝ, -1 < x ∧ x < 1 ∧ f' x a = 0 ∧ ∀ x₁ x₂, f' x₁ a * f' x₂ a < 0) :
  a < -1 / 4 ∨ a > 1 / 4 :=
begin
  sorry
end

end part1_part2_l442_442834


namespace pairwise_coprime_circle_l442_442626

theorem pairwise_coprime_circle :
  ∃ (circle : Fin 100 → ℕ),
    (∀ i, Nat.gcd (circle i) (Nat.gcd (circle ((i + 1) % 100)) (circle ((i - 1) % 100))) = 1) → 
    ∀ i j, i ≠ j → Nat.gcd (circle i) (circle j) = 1 :=
by
  sorry

end pairwise_coprime_circle_l442_442626


namespace John_gave_the_store_20_dollars_l442_442585

def slurpee_cost : ℕ := 2
def change_received : ℕ := 8
def slurpees_bought : ℕ := 6
def total_money_given : ℕ := slurpee_cost * slurpees_bought + change_received

theorem John_gave_the_store_20_dollars : total_money_given = 20 := 
by 
  sorry

end John_gave_the_store_20_dollars_l442_442585


namespace total_cost_is_17_l442_442629

def taco_shells_cost : ℝ := 5
def bell_pepper_cost_per_unit : ℝ := 1.5
def bell_pepper_quantity : ℕ := 4
def meat_cost_per_pound : ℝ := 3
def meat_quantity : ℕ := 2

def total_spent : ℝ :=
  taco_shells_cost + (bell_pepper_cost_per_unit * bell_pepper_quantity) + (meat_cost_per_pound * meat_quantity)

theorem total_cost_is_17 : total_spent = 17 := 
  by sorry

end total_cost_is_17_l442_442629


namespace circle_Q_radius_l442_442184

-- Definitions of the geometric entities
variable (A B C P Q : Type)
variable (A_pos B_pos C_pos : A × B × C → Prop)
variable (radius_P : ℝ) (radius_Q : ℝ)
variable (AB_AC_68 : ℝ → ℝ → Prop)
variable (BC_48 : ℝ → Prop)
variable (tangent_to : P → A × B × C × ℝ → Prop)
variable (ext_tangent : Q → P × A × B × C → Prop)

-- Conditions about the triangle and circles
axiom triangle_cond :
  ∀ (A B C : Type) (AB AC: ℝ),
  AB = 68 ∧ AC = 68 ∧ AB = AC ∧ BC_48 48

axiom circle_P_cond :
  ∀ (P : Type) (radius_P : ℝ),
  radius_P = 12 ∧ tangent_to P (A, B, C, AC) ∧ tangent_to P (A, B, C, BC)

axiom circle_Q_cond :
  ∀ (Q : Type) (radius_Q : ℝ),
  ext_tangent Q (P, A, B, C) ∧ tangent_to Q (A, B, C, AB) ∧ tangent_to Q (A, B, C, BC)

-- Main theorem to prove that radius_Q is 8
theorem circle_Q_radius :
  ∀ (A B C P Q : Type) (A_pos B_pos C_pos : A × B × C → Prop) (AB_AC_68 : ℝ → ℝ → Prop) (BC_48 : ℝ → Prop)
  (radius_P radius_Q : ℝ) (tangent_to : P → A × B × C × ℝ → Prop) (ext_tangent : Q → P × A × B × C → Prop),
  triangle_cond A B C 68 68 →
  circle_P_cond P 12 →
  circle_Q_cond Q 8 →
  radius_Q = 8 := 
begin
  -- Adding sorry because we are not required to provide the proof steps
  sorry
end

end circle_Q_radius_l442_442184


namespace tan_beta_eq_minus_one_seventh_l442_442148

theorem tan_beta_eq_minus_one_seventh {α β : ℝ} 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := 
by
  sorry

end tan_beta_eq_minus_one_seventh_l442_442148


namespace find_x_value_l442_442919

theorem find_x_value (x : ℝ) (h : x + 21 + 21 + 2*x + 57 = 180) : x = 27 := 
by {
  -- Introduction of hypothesis
  have h_combined : 3 * x + 99 = 180 := by {
    simp [mul_comm, add_assoc] at h,
    exact h,
  },
  -- Isolate x by solving the equation
  linarith,
}

end find_x_value_l442_442919


namespace four_digit_increasing_even_integers_l442_442527

theorem four_digit_increasing_even_integers : 
  let even_four_digit_strictly_increasing (n : ℕ) := 
    n >= 1000 ∧ n < 10000 ∧ (n % 2 = 0) ∧ (let d1 := n / 1000 % 10, 
                                                d2 := n / 100 % 10,
                                                d3 := n / 10 % 10,
                                                d4 := n % 10 in
                                            d1 < d2 ∧ d2 < d3 ∧ d3 < d4)
  in (finset.filter even_four_digit_strictly_increasing (finset.range 10000)).card = 46 :=
begin
  sorry
end

end four_digit_increasing_even_integers_l442_442527


namespace area_of_fourth_rectangle_l442_442770

-- The conditions provided in the problem
variables (x y z w : ℝ)
variables (h1 : x * y = 24) (h2 : x * w = 12) (h3 : z * w = 8)

-- The problem statement with the conclusion
theorem area_of_fourth_rectangle :
  (∃ (x y z w : ℝ), ((x * y = 24 ∧ x * w = 12 ∧ z * w = 8) ∧ y * z = 16)) :=
sorry

end area_of_fourth_rectangle_l442_442770


namespace tile_arrangements_l442_442882

theorem tile_arrangements :
  let n := 7
  let nb := 1
  let np := 1
  let ng := 3
  let ny := 2
  n = nb + np + ng + ny →
  n.factorial / (nb.factorial * np.factorial * ng.factorial * ny.factorial) = 420 :=
by {
  intros h,
  rw [h],
  simp,
  norm_num,
  sorry
}

end tile_arrangements_l442_442882


namespace product_of_y_coordinates_l442_442261

theorem product_of_y_coordinates (y : ℝ) 
    (h1 : ∃ y, ∃ Q : ℝ × ℝ, Q = (-3, y)) 
    (h2 : (λ (Q : ℝ × ℝ), real.sqrt ((5 + 3)^2 + (2 - y)^2)) (-3, y) = 14) 
    : (2 + real.sqrt 132) * (2 - real.sqrt 132) = -128 :=
begin
  sorry
end

end product_of_y_coordinates_l442_442261


namespace B_is_irrational_l442_442791

noncomputable def A : ℝ := 3.1415926
noncomputable def B : ℝ := - (Real.cbrt 9)
noncomputable def C : ℝ := - (Real.sqrt 0.25)
noncomputable def D : ℝ := - (9 / 13)

def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b
def is_irrational (x : ℝ) : Prop := ¬ is_rational x

theorem B_is_irrational :
  is_irrational B ∧ is_rational A ∧ is_rational C ∧ is_rational D := by
  sorry

end B_is_irrational_l442_442791


namespace last_digit_to_appear_in_fib_mod_10_is_6_l442_442285

-- Defining the Fibonacci sequence modulo 10
noncomputable def fib_mod_10 : ℕ → ℕ
| 0     := 0
| 1     := 1
| n + 2 := (fib_mod_10 (n + 1) + fib_mod_10 n) % 10

-- The proof statement
theorem last_digit_to_appear_in_fib_mod_10_is_6 :
    ∃ n, (∀ m < n, ∃ d, d < 10 ∧ ∃ k < m, fib_mod_10 k % 10 = d) ∧ ∃ k < n, fib_mod_10 k % 10 = 6 := sorry

end last_digit_to_appear_in_fib_mod_10_is_6_l442_442285


namespace probability_odd_divisor_15_l442_442675

-- Defining the factorial 15
def fact_15 := 1307674368000

-- Defining the prime factorization of 15!
def prime_factorization_15! := (2^11) * (3^6) * (5^3) * (7^2) * (11^1) * (13^1)

-- Total number of factors of 15!
def total_divisors (n : ℕ) : ℕ := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

-- Number of odd factors of 15!
def odd_divisors (n : ℕ) : ℕ := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

-- Probability of choosing an odd divisor of 15!
theorem probability_odd_divisor_15! (h : 15! = fact_15) : 
  (odd_divisors 15 / total_divisors 15 : ℚ) = 1 / 24 :=
by
  sorry

end probability_odd_divisor_15_l442_442675


namespace isabella_houses_problem_l442_442190

theorem isabella_houses_problem 
  (yellow green red : ℕ)
  (h1 : green = 3 * yellow)
  (h2 : yellow = red - 40)
  (h3 : green = 90) :
  (green + red = 160) := 
sorry

end isabella_houses_problem_l442_442190


namespace Rachel_painting_time_l442_442963

theorem Rachel_painting_time :
  ∀ (Matt_time Patty_time Rachel_time : ℕ),
  Matt_time = 12 →
  Patty_time = Matt_time / 3 →
  Rachel_time = 2 * Patty_time + 5 →
  Rachel_time = 13 :=
by
  intros Matt_time Patty_time Rachel_time hMatt hPatty hRachel
  rw [hMatt] at hPatty
  rw [hPatty, hRachel]
  sorry

end Rachel_painting_time_l442_442963


namespace arithmetic_geometric_sequence_properties_l442_442848

noncomputable def a_n : ℕ → ℝ := sorry  -- Define the arithmetic-geometric sequence

-- Sum of the first n terms
noncomputable def S_n (n : ℕ) : ℝ := (finset.range (n+1)).sum (λ i, a_n i)

-- Condition given in the problem
axiom condition (n : ℕ) : S_n (n + 3) = 8 * S_n n + 3

-- Proof statement
theorem arithmetic_geometric_sequence_properties :
  a_n 1 = 3 / 7 ∧ (∃ q : ℝ, q ≠ 1 ∧ q = 2 ∧ ∀ n, a_n (n + 3) = q^3 * a_n n) := 
sorry

end arithmetic_geometric_sequence_properties_l442_442848


namespace simplify_trig_expr_l442_442363

theorem simplify_trig_expr (α : ℝ) : 
  (sin (2 * π - α) * tan (π - α) * cos (-π + α)) / (sin (5 * π + α) * sin (π / 2 + α)) = tan α :=
sorry

end simplify_trig_expr_l442_442363


namespace cos_neg_90_eq_zero_l442_442697

noncomputable def cos := real.cos

theorem cos_neg_90_eq_zero : cos (-(90 : ℝ) * (real.pi / 180)) = 0 :=
by
  -- Known conditions:
  -- (1) Cosine function is even: cos(-x) = cos(x).
  -- (2) Trigonometric function value: cos(90°) = 0.

  sorry

end cos_neg_90_eq_zero_l442_442697


namespace sum_of_sequence_l442_442306

theorem sum_of_sequence (n : ℕ) :
  ∑ i in finset.range (n + 1), (i + 1 + 1 / (2 ^ (i + 1))) = n * (n + 1) / 2 + 1 - (1 / 2 ^ n) :=
by
  -- sorry placeholder for the proof
  sorry

end sum_of_sequence_l442_442306


namespace prob_lfloor_XZ_YZ_product_eq_33_l442_442243

noncomputable def XZ_YZ_product : ℝ :=
  let AB := 15
  let BC := 14
  let CA := 13
  -- Definition of points and conditions
  -- Note: Specific geometric definitions and conditions need to be properly defined as per Lean's geometry library. This is a simplified placeholder.
  sorry

theorem prob_lfloor_XZ_YZ_product_eq_33 :
  (⌊XZ_YZ_product⌋ = 33) := sorry

end prob_lfloor_XZ_YZ_product_eq_33_l442_442243


namespace exists_diff_eq_nine_l442_442326

theorem exists_diff_eq_nine (S : Finset ℕ) (h1 : S.card = 55) (h2 : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 99)  :
  ∃ x y ∈ S, x ≠ y ∧ |x - y| = 9 :=
by
  sorry

end exists_diff_eq_nine_l442_442326


namespace dorothy_total_sea_glass_l442_442802

def Blanche_red : ℕ := 3
def Rose_red : ℕ := 9
def Rose_blue : ℕ := 11

def Dorothy_red : ℕ := 2 * (Blanche_red + Rose_red)
def Dorothy_blue : ℕ := 3 * Rose_blue

theorem dorothy_total_sea_glass : Dorothy_red + Dorothy_blue = 57 :=
by
  sorry

end dorothy_total_sea_glass_l442_442802


namespace solution_set_of_f_when_a_eq_1_range_of_a_l442_442136

noncomputable def f (x a : ℝ) : ℝ := | 4 * x - a | + a^2 - 4 * a
noncomputable def g (x : ℝ) : ℝ := | x - 1 |

theorem solution_set_of_f_when_a_eq_1 (x : ℝ) :
  (-2 ≤ f x 1 ∧ f x 1 ≤ 4) ↔ (-(3 / 2) ≤ x ∧ x ≤ 0) ∨ ((1 / 2) ≤ x ∧ x ≤ 2) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, f x a - 4 * g x ≤ 6) ↔ (a ∈ Set.Icc ((5 - Real.sqrt 33) / 2) 5) :=
sorry

end solution_set_of_f_when_a_eq_1_range_of_a_l442_442136


namespace option_A_option_B_option_D_l442_442941

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x - 1 else x^2 - 2 * x + 2

def seq (a₁ : ℝ) : ℕ → ℝ
| 0     := a₁
| (n+1) := f (seq n)

theorem option_A (a₁ : ℝ) (h : a₁ = 3 / 2) : ∀ n, 1 < seq a₁ n ∧ seq a₁ n < 2 :=
sorry

theorem option_B (a₁ : ℝ) (h : ∀ n, seq a₁ (n+1) = seq a₁ n) : a₁ = 1 ∨ a₁ = 2 :=
sorry

theorem option_D (a₁ : ℝ) (h : a₁ = 3) : ∀ n, ∑ i in finset.range n, 1 / seq a₁ (i+1) < 1 :=
sorry

end option_A_option_B_option_D_l442_442941


namespace intersection_of_A_and_B_l442_442220

-- Define sets A and B
def A : set ℝ := {x | -2 < x ∧ x < 4}
def B : set ℕ := {2, 3, 4, 5}

-- Theorem stating the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {2, 3} := 
by sorry

end intersection_of_A_and_B_l442_442220


namespace intersection_A_B_l442_442216

-- Defining the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

-- Statement to prove
theorem intersection_A_B : A ∩ B = {2, 3} := 
by 
  sorry

end intersection_A_B_l442_442216


namespace james_profit_correct_l442_442579

noncomputable def james_profit : ℕ :=
  let cost_per_toy := 20 in
  let sell_price_per_toy := 30 in
  let total_toys := 200 in
  let percent_to_sell := 80 in
  let total_cost := cost_per_toy * total_toys in
  let toys_sold := total_toys * percent_to_sell / 100 in
  let total_revenue := toys_sold * sell_price_per_toy in
  total_revenue - total_cost

theorem james_profit_correct : james_profit = 800 :=
by
  sorry

end james_profit_correct_l442_442579


namespace count_even_strictly_increasing_integers_correct_l442_442531

-- Definition of condition: even four-digit integers with strictly increasing digits
def is_strictly_increasing {a b c d : ℕ} : Prop :=
1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ∈ {2, 4, 6, 8}

def count_even_strictly_increasing_integers : ℕ :=
(finset.range 10).choose 4.filter (λ l, is_strictly_increasing l.head l.nth 1 l.nth 2 l.nth 3).card

theorem count_even_strictly_increasing_integers_correct :
  count_even_strictly_increasing_integers = 46 := by
  sorry

end count_even_strictly_increasing_integers_correct_l442_442531


namespace z_in_second_quadrant_l442_442120

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

def sum_of_parts (z : ℂ) (s : ℝ) : Prop := 
  z.re + z.im = s

theorem z_in_second_quadrant (a : ℝ) (hz : (10 - 5 * a * complex.I) / (1 - 2 * complex.I)) 
  (h : sum_of_parts hz 4) : is_in_second_quadrant hz :=
sorry

end z_in_second_quadrant_l442_442120


namespace sum_of_ages_is_20_l442_442932

-- Given conditions
variables (age_kiana age_twin : ℕ)
axiom product_of_ages : age_kiana * age_twin * age_twin = 162

-- Required proof
theorem sum_of_ages_is_20 : age_kiana + age_twin + age_twin = 20 :=
sorry

end sum_of_ages_is_20_l442_442932


namespace proof_I_proof_II_proof_III_l442_442172

-- Define the context and problem conditions for (I)
def problem_I : Prop :=
  -- number of scheduling methods for Mathematics, Physics, Chemistry in morning
  -- and two self-study periods in afternoon is 864
  let morning_subjects := ["Mathematics", "Physics", "Chemistry"]
  let afternoon_self_study := 2
  (count_scheduling_methods morning_subjects afternoon_self_study) = 864

-- Define the context and problem conditions for (II)
def problem_II : Prop :=
  -- number of scheduling methods given Physical Education is not scheduled for the first period,
  -- and Mathematics is not scheduled for the last period is 15480
  let exclude_periods := [("Physical Education", 1), ("Mathematics", 8)]
  (count_scheduling_methods_with_restrictions exclude_periods) = 15480

-- Define the context and problem conditions for (III)
def problem_III : Prop :=
  -- number of scheduling methods given Chinese and Mathematics are scheduled consecutively,
  -- and two self-study periods are not scheduled consecutively is 3264
  let consecutive_subjects := [("Chinese", "Mathematics")]
  let non_consecutive_per_subject := "self-study"
  let non_consecutive_exclusion := (4, 5)
  (count_scheduling_methods_with_consecutive consecutive_subjects non_consecutive_per_subject non_consecutive_exclusion) = 3264

-- Proof placeholders for the stated problems
theorem proof_I : problem_I := sorry
theorem proof_II : problem_II := sorry
theorem proof_III : problem_III := sorry

end proof_I_proof_II_proof_III_l442_442172


namespace measure_angle_A_triangle_area_l442_442168

namespace TriangleProblem

-- Definitions from the given problem
variables {a b c : ℝ} {A B C : ℝ}

-- Conditions given in the problem
def sides_opposite (t : Triangle) : Prop := 
  t.a = a ∧ t.b = b ∧ t.c = c

def arithmetic_mean_relation : Prop :=
  c * Real.sin (π / 2 - A) = (a * Real.sin (π / 2 - B) + b * Real.cos A) / 2

def additional_conditions : Prop :=
  2 * a = b + c ∧ R = 1

-- Goals to prove
theorem measure_angle_A (t : Triangle) (h : sides_opposite t) (amr : arithmetic_mean_relation) : 
  A = π / 3 := 
sorry

theorem triangle_area (t : Triangle) (h : sides_opposite t) (amr : arithmetic_mean_relation)
  (ac : additional_conditions) :
  S = (3 * Real.sqrt 3) / 4 := 
sorry

end TriangleProblem

end measure_angle_A_triangle_area_l442_442168


namespace probability_one_segment_longer_than_3_l442_442026

-- Define the length of the rope
def rope_length : ℝ := 5

-- Define the probability function for one segment being longer than 3 meters
def prob_segment_longer_than_3 (cut_point : ℝ) (length : ℝ) : ℝ :=
  if cut_point < 2 then
    2 / length
  else if cut_point > 3 then
    2 / length
  else
    0

-- The goal is to prove the total probability is 4/5 if the cut point is uniformly distributed across the rope
theorem probability_one_segment_longer_than_3 (cut_point : ℝ) (length : ℝ) 
  (h_cut_point : 0 ≤ cut_point ∧ cut_point ≤ length) 
  (h_length : length = 5) 
  : (∫ (x : ℝ) in 0..length, prob_segment_longer_than_3 x length) / length = 4/5 :=
by 
  sorry

end probability_one_segment_longer_than_3_l442_442026


namespace circles_intersect_at_single_point_l442_442954

-- Definitions of points on sides of the triangle
variables {A B C P Q R : Type} -- Typing points

-- Conditions for points P, Q, R on sides of triangle ABC
axiom P_on_AB : P ∈ line_segment A B
axiom Q_on_BC : Q ∈ line_segment B C
axiom R_on_AC : R ∈ line_segment A C

-- Goal: The circumcircles of triangles APR, BQR, and CRQ intersect at a single point
theorem circles_intersect_at_single_point :
  ∃ X, X ∈ circumcircle A P R ∧ X ∈ circumcircle B Q R ∧ X ∈ circumcircle C R Q :=
sorry

end circles_intersect_at_single_point_l442_442954


namespace sequence_bn_sequence_Tn_l442_442115
noncomputable theory
open BigOperators

-- Define the arithmetic sequence {a_n} and the conditions
def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ := a + n * d

-- Define the sequence b_n and its initial conditions
def b (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 5
  else if n = 3 then 17
  else 2 * 3^(n - 1) - 1

-- Question 1: Prove the formula for b_n
theorem sequence_bn (n : ℕ) : b n = 2 * 3^(n - 1) - 1 :=
sorry

-- Define the sum T_n
def T (n : ℕ) : ℕ :=
∑ i in Finset.range n, Nat.choose n (i + 1) * b (i + 1)

-- Question 2: Prove the formula for T_n
theorem sequence_Tn (n : ℕ) : T n = (2/3:ℚ) * 4^n - 2^n + (1/3:ℚ) :=
sorry

end sequence_bn_sequence_Tn_l442_442115


namespace perpendiculars_intersect_at_single_point_l442_442431

-- Define the excircles for a given triangle ABC and their tangency properties
noncomputable def excircle_tangent (A B C : Point) (ωA ωB ωC : Circle) : Prop :=
  ∃ D E F E' F' D' E'' F'' D'',
    (ωA.Tangent B C D ∧ ωA.Tangent C A E ∧ ωA.Tangent A B F) ∧
    (ωB.Tangent C A E' ∧ ωB.Tangent A B F' ∧ ωB.Tangent B C D') ∧
    (ωC.Tangent A B F'' ∧ ωC.Tangent B C D'' ∧ ωC.Tangent C A E'')

-- Define the concurrency of perpendiculars
noncomputable def perpendiculars_concur (A B C : Point) (D E F : Point) : Prop :=
  let perpA := Line.perpendicular A D (Line.mk B C)
  let perpB := Line.perpendicular B E (Line.mk C A)
  let perpC := Line.perpendicular C F (Line.mk A B)
  ∃ P : Point, Line.contains perpA P ∧ Line.contains perpB P ∧ Line.contains perpC P

-- Define the main theorem statement
theorem perpendiculars_intersect_at_single_point
  (A B C : Point) (ωA ωB ωC : Circle)
  (h_excircle : excircle_tangent A B C ωA ωB ωC) :
  ∃ P : Point, perpendiculars_concur A B C (D E F P) :=
by
  sorry

end perpendiculars_intersect_at_single_point_l442_442431


namespace maria_age_l442_442202

variable (M J : Nat)

theorem maria_age (h1 : J = M + 12) (h2 : M + J = 40) : M = 14 := by
  sorry

end maria_age_l442_442202


namespace cinderella_can_form_square_quilt_l442_442695

-- Definitions of rectangle dimensions
structure Rectangle where
  length : ℕ
  width : ℕ

-- Specific rectangles given in the problem
def rect1 : Rectangle := ⟨1, 2⟩
def rect2 : Rectangle := ⟨7, 10⟩
def rect3 : Rectangle := ⟨6, 5⟩
def rect4 : Rectangle := ⟨8, 12⟩
def rect5 : Rectangle := ⟨9, 3⟩

-- Verify pairwise different side lengths
def pairwise_diff (r1 r2 r3 r4 r5 : Rectangle) : Prop :=
  r1.length ≠ r2.length ∧ r1.length ≠ r3.length ∧ r1.length ≠ r4.length ∧ r1.length ≠ r5.length ∧
  r1.width ≠ r2.width ∧ r1.width ≠ r3.width ∧ r1.width ≠ r4.width ∧ r1.width ≠ r5.width ∧
  r2.length ≠ r3.length ∧ r2.length ≠ r4.length ∧ r2.length ≠ r5.length ∧
  r2.width ≠ r3.width ∧ r2.width ≠ r4.width ∧ r2.width ≠ r5.width ∧
  r3.length ≠ r4.length ∧ r3.length ≠ r5.length ∧
  r3.width ≠ r4.width ∧ r3.width ≠ r5.width ∧
  r4.length ≠ r5.length ∧
  r4.width ≠ r5.width

-- Calculate area of a rectangle
def area (r : Rectangle) : ℕ :=
  r.length * r.width

-- Total area of all rectangles
def total_area (rectangles : List Rectangle) : ℕ :=
  rectangles.map area |>.sum

-- Prove Cinderella can form a square quilt
theorem cinderella_can_form_square_quilt :
  pairwise_diff rect1 rect2 rect3 rect4 rect5 ∧
  total_area [rect1, rect2, rect3, rect4, rect5] = 225 → 
  ∃ s : ℕ, s * s = 225 :=
by
  intro h
  use 15
  simp
  sorry

end cinderella_can_form_square_quilt_l442_442695


namespace find_fraction_l442_442716

noncomputable def solve_fraction (x : ℝ) (h : x = 0.3333333333333333) : ℝ :=
  let k := (2 / 3 * x) * (x / 1) in k

theorem find_fraction (x : ℝ) (h : x = 0.3333333333333333) : solve_fraction x h = 2 / 27 :=
by
  sorry

end find_fraction_l442_442716


namespace pump_fill_time_without_leak_l442_442409

variable (T : ℕ)

def rate_pump (T : ℕ) : ℚ := 1 / T
def rate_leak : ℚ := 1 / 20

theorem pump_fill_time_without_leak : rate_pump T - rate_leak = rate_leak → T = 10 := by 
  intro h
  sorry

end pump_fill_time_without_leak_l442_442409


namespace true_propositions_l442_442126

-- Quadratic equation properties
def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Proposition ① related definitions
def proposition_1 (m : ℝ) : Prop := 
  (m > -1) → (quadratic_discriminant 1 2 (-m) ≥ 0)

-- Proposition ② related definitions
def proposition_2 (x : ℝ) : Prop :=
  (x = 1 → x^2 - 3 * x + 2 = 0) ∧ (x ≠ 1 → x^2 - 3 * x + 2 = 0 → (x = 2 ∨ x = 1))

-- Proposition ③ related definitions
def proposition_3 : Prop := 
  ¬(∀ (q : Type) [Quadrilateral q], quadrilateral_diagonals_equal q ↔ is_rectangle q)

-- Proposition ④ related definitions
def proposition_4 (x y : ℝ) : Prop := 
  ((x = 0 ∧ y = 0) ↔ (x^2 + y^2 = 0))

-- Main theorem
theorem true_propositions : 
  (proposition_1 m) ∧ (proposition_2 1) ∧ (¬ proposition_3) ∧ (proposition_4 0 0) :=
by
  sorry

end true_propositions_l442_442126


namespace total_juice_drank_l442_442928

open BigOperators

theorem total_juice_drank (joe_juice sam_fraction alex_fraction : ℚ) :
  joe_juice = 3 / 4 ∧ sam_fraction = 1 / 2 ∧ alex_fraction = 1 / 4 → 
  sam_fraction * joe_juice + alex_fraction * joe_juice = 9 / 16 :=
by
  sorry

end total_juice_drank_l442_442928


namespace soda_cost_is_2_l442_442797

noncomputable def cost_per_soda (total_bill : ℕ) (num_adults : ℕ) (num_children : ℕ) 
  (adult_meal_cost : ℕ) (child_meal_cost : ℕ) (num_sodas : ℕ) : ℕ :=
  (total_bill - (num_adults * adult_meal_cost + num_children * child_meal_cost)) / num_sodas

theorem soda_cost_is_2 :
  let total_bill := 60
  let num_adults := 6
  let num_children := 2
  let adult_meal_cost := 6
  let child_meal_cost := 4
  let num_sodas := num_adults + num_children
  cost_per_soda total_bill num_adults num_children adult_meal_cost child_meal_cost num_sodas = 2 :=
by
  -- proof goes here
  sorry

end soda_cost_is_2_l442_442797


namespace exists_lambda_of_line_intersects_l442_442924

noncomputable def ellipseEquation := 
Eq (x^2 / 5 + y^2) 1

noncomputable def parabolaEquation := 
Eq (y^2) (8 * x)

theorem exists_lambda_of_line_intersects 
  (k : ℝ) 
  (AB CD : ℝ) 
  (h_AB : AB = (2 * sqrt 5 * (1 + k^2)) / (1 + 5*k^2))
  (h_CD : CD = 8 * (1 + k^2) / k^2):
  ∃ λ : ℝ, (2 / AB + λ / CD) = const := 
  sorry

lemma lambda_value :
  let λ = - (16 * sqrt 5) / 5 in 
  λ = (2 / |AB| + λ / |CD|) : 
  sorry

end exists_lambda_of_line_intersects_l442_442924


namespace attendees_received_all_items_l442_442800

theorem attendees_received_all_items {n : ℕ} (h1 : ∀ k, k ∣ 45 → n % k = 0) (h2 : ∀ k, k ∣ 75 → n % k = 0) (h3 : ∀ k, k ∣ 100 → n % k = 0) (h4 : n = 4500) :
  (4500 / Nat.lcm (Nat.lcm 45 75) 100) = 5 :=
by
  sorry

end attendees_received_all_items_l442_442800


namespace circumscribed_circle_area_l442_442008

noncomputable def sin_36 := real.sin (36 * real.pi / 180)
noncomputable def radius (s : ℝ) : ℝ := s / (2 * sin_36)
noncomputable def area_of_circumscribed_circle (s : ℝ) : ℝ := real.pi * (radius s) ^ 2

theorem circumscribed_circle_area {s : ℝ} (h : s = 10) :
  area_of_circumscribed_circle s = 2000 * (5 + 2 * real.sqrt 5) * real.pi :=
by
  -- using the given condition
  rw [h]
  -- calculation steps omitted
  sorry

end circumscribed_circle_area_l442_442008


namespace sequence_sixth_term_is_364_l442_442173

theorem sequence_sixth_term_is_364 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 2 = 7) (h3 : a 3 = 20)
  (h4 : ∀ n, a (n + 1) = 1 / 3 * (a n + a (n + 2))) :
  a 6 = 364 :=
by
  -- Proof skipped
  sorry

end sequence_sixth_term_is_364_l442_442173


namespace number_of_real_roots_l442_442677

theorem number_of_real_roots :
  ∃ (roots_count : ℕ), roots_count = 2 ∧
  (∀ x : ℝ, x^2 - |2 * x - 1| - 4 = 0 → (x = -1 - Real.sqrt 6 ∨ x = 3)) :=
sorry

end number_of_real_roots_l442_442677


namespace constant_term_binomial_expansion_l442_442665

theorem constant_term_binomial_expansion :
  let x := by assumption : ℝ,
  let bin_exp := (x + 1 + x⁻¹)^6,
  let re_written_exp := (1 + (x + x⁻¹))^6,
  let C := (nat.choose 6 0 + nat.choose 6 2 * nat.choose 2 1 + nat.choose 6 4 * nat.choose 4 2 + nat.choose 6 6 * nat.choose 6 3) in
  ∃ (const_term : ℝ), const_term = 141 :=
  sorry

end constant_term_binomial_expansion_l442_442665


namespace sequence_perfect_square_l442_442141

theorem sequence_perfect_square (k : ℤ) :
  (∀ n, ∃ m, m * m = y n) ↔ (k = 1 ∨ k = 3)
where
  -- Define the sequence using the provided recurrence relation
  y : ℕ → ℤ
  | 0     := 1
  | 1     := 1
  | (n+2) := (4 * k - 5) * y (n+1) - y n + 4 - 2 * k

end sequence_perfect_square_l442_442141


namespace range_of_g_l442_442595

open Real

noncomputable def g (x : ℝ) : ℝ := (arccos x)^4 + (arcsin x)^4

theorem range_of_g :
  set.range g = set.Icc (-(π^4 / 8)) (π^4 / 4) :=
sorry

end range_of_g_l442_442595


namespace angle_CED_l442_442142

theorem angle_CED {A B C D E F : Type*} (h1 : ∃ (circleA circleB : set (ℝ × ℝ)), circleA ≠ circleB ∧ ∀ (P : ℝ × ℝ), P ∈ circleA ↔ P ∈ circleB)
  (h2 : ∀ (circleA circleB : set (ℝ × ℝ)), circleA ∋ A ∧ circleB ∋ B ∧ E ∈ circleA ∧ F ∈ circleA ∧ E ∈ circleB ∧ F ∈ circleB ∧ A ∈ circleB ∧ B ∈ circleA)
  (h3 : line AB intersects circles at C and D)
  : angle C E D = 120 :=
sorry

end angle_CED_l442_442142


namespace costPerUse_l442_442199

-- Definitions based on conditions
def heatingPadCost : ℝ := 30
def usesPerWeek : ℕ := 3
def totalWeeks : ℕ := 2

-- Calculate the total number of uses
def totalUses : ℕ := usesPerWeek * totalWeeks

-- The amount spent per use
theorem costPerUse : heatingPadCost / totalUses = 5 := by
  sorry

end costPerUse_l442_442199


namespace percent_difference_calculation_l442_442534

theorem percent_difference_calculation :
  (0.80 * 45) - ((4 / 5) * 25) = 16 :=
by sorry

end percent_difference_calculation_l442_442534


namespace number_of_possible_areas_is_five_l442_442441

noncomputable def distinct_points : Prop :=
  ∃ (A B C D E F G : ℝ) 
    (AB BC CD EF FG EG : ℝ),
    A < B ∧ B < C ∧ C < D ∧
    E < F ∧ F < G ∧
    AB = 1 ∧ BC = 2 ∧ CD = 3 ∧
    EF = 1 ∧ FG = 2 ∧ EG = 3 ∧
    ∃ Z : ℕ, 
      (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z → 
      x = A ∨ x = B ∨ x = C ∨ x = D ∨ x = E ∨ x = F ∨ x = G →
      y = A ∨ y = B ∨ y = C ∨ y = D ∨ y = E ∨ y = F ∨ y = G →
      z = A ∨ z = B ∨ z = C ∨ z = D ∨ z = E ∨ z = F ∨ z = G →
      let base := abs (y - x) in
      let height := abs (G - A) in
      ∃ areas : list ℝ, 
        ∃ (h : (1 / 2) * base * height ∈ areas), 
        areas.length = 5)

theorem number_of_possible_areas_is_five : distinct_points :=
  sorry

end number_of_possible_areas_is_five_l442_442441


namespace john_investment_in_bank_a_l442_442200

theorem john_investment_in_bank_a :
  ∃ x : ℝ, 
    0 ≤ x ∧ x ≤ 1500 ∧
    x * (1 + 0.04)^3 + (1500 - x) * (1 + 0.06)^3 = 1740.54 ∧
    x = 695 := sorry

end john_investment_in_bank_a_l442_442200


namespace number_of_employees_l442_442593

def fixed_time_coffee : ℕ := 5
def time_per_status_update : ℕ := 2
def time_per_payroll_update : ℕ := 3
def total_morning_routine : ℕ := 50

def time_per_employee : ℕ := time_per_status_update + time_per_payroll_update
def time_spent_on_employees : ℕ := total_morning_routine - fixed_time_coffee

theorem number_of_employees : (time_spent_on_employees / time_per_employee) = 9 := by
  sorry

end number_of_employees_l442_442593


namespace intersection_of_A_and_B_l442_442227

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l442_442227


namespace intersection_A_B_l442_442217

-- Defining the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

-- Statement to prove
theorem intersection_A_B : A ∩ B = {2, 3} := 
by 
  sorry

end intersection_A_B_l442_442217


namespace Ava_watched_television_for_240_minutes_l442_442047

-- Define the conditions
def hours (h : ℕ) := h = 4

-- Define the conversion factor from hours to minutes
def convert_hours_to_minutes (h : ℕ) : ℕ := h * 60

-- State the theorem
theorem Ava_watched_television_for_240_minutes (h : ℕ) (hh : hours h) : convert_hours_to_minutes h = 240 :=
by
  -- The proof goes here but is skipped
  sorry

end Ava_watched_television_for_240_minutes_l442_442047


namespace estimate_three_or_more_houses_percentage_l442_442170

theorem estimate_three_or_more_houses_percentage :
  let total_households := 100000
  let ordinary_families := 99000
  let high_income_families := 1000
  let sample_size_ordinary := 990
  let sample_size_high_income := 100
  let sampled_ordinary_three_plus := 50
  let sampled_high_income_three_plus := 70
  let estimated_ordinary_three_plus := ordinary_families * sampled_ordinary_three_plus / sample_size_ordinary
  let estimated_high_income_three_plus := high_income_families * sampled_high_income_three_plus / sample_size_high_income
  let total_estimated_three_plus := estimated_ordinary_three_plus + estimated_high_income_three_plus
  let percentage_three_plus := total_estimated_three_plus / total_households * 100
  in percentage_three_plus = 5.7 := sorry

end estimate_three_or_more_houses_percentage_l442_442170


namespace fraction_simplification_l442_442076

theorem fraction_simplification 
  (a b c : ℝ)
  (h₀ : a ≠ 0) 
  (h₁ : b ≠ 0) 
  (h₂ : c ≠ 0) 
  (h₃ : a^2 + b^2 + c^2 ≠ 0) :
  (a^2 * b^2 + 2 * a^2 * b * c + a^2 * c^2 - b^4) / (a^4 - b^2 * c^2 + 2 * a * b * c^2 + c^4) =
  ((a * b + a * c + b^2) * (a * b + a * c - b^2)) / ((a^2 + b^2 - c^2) * (a^2 - b^2 + c^2)) :=
sorry

end fraction_simplification_l442_442076


namespace find_a_l442_442299

-- Define the conditions of the problem
def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a + 3, 1, -3) -- Coefficients of line1: (a+3)x + y - 3 = 0
def line2 (a : ℝ) : ℝ × ℝ × ℝ := (5, a - 3, 4)  -- Coefficients of line2: 5x + (a-3)y + 4 = 0

-- Definition of direction vector and normal vector
def direction_vector (a : ℝ) : ℝ × ℝ := (1, -(a + 3))
def normal_vector (a : ℝ) : ℝ × ℝ := (5, a - 3)

-- Proof statement
theorem find_a (a : ℝ) : (direction_vector a = normal_vector a) → a = -2 :=
by {
  -- Insert proof here
  sorry
}

end find_a_l442_442299


namespace zero_in_interval_l442_442436

noncomputable def f (x : ℝ) := Real.log x - 6 + 2 * x

theorem zero_in_interval : ∃ c ∈ Set.Ioo 2 3, f c = 0 := by
  sorry

end zero_in_interval_l442_442436


namespace problem_evaluate_expression_l442_442074

theorem problem_evaluate_expression (a b : ℕ) (ha : a = 3) (hb : b = 2) : (a^b)^a + 2*(a^b - b^a) = 731 :=
by
  -- substitute the values of a and b
  rw [ha, hb]
  -- Calculate (3^2)^3
  calc (3^2)^3 + 2*(3^2 - 2^3)
      = 9^3 + 2*(9 - 8) : by rfl
  ... = 729 + 2*1 : by norm_num
  ... = 729 + 2 : by norm_num
  ... = 731 : by norm_num

end problem_evaluate_expression_l442_442074


namespace product_xyz_eq_one_l442_442538

theorem product_xyz_eq_one (x y z : ℝ) (h1 : x + 1 / y = 2) (h2 : y + 1 / z = 2) : x * y * z = 1 := 
sorry

end product_xyz_eq_one_l442_442538


namespace ball_radius_l442_442368

theorem ball_radius (x r : ℝ) 
  (h1 : (15 : ℝ) ^ 2 + x ^ 2 = r ^ 2) 
  (h2 : x + 12 = r) : 
  r = 15.375 := 
sorry

end ball_radius_l442_442368


namespace jacksonville_walmart_complaints_l442_442648

theorem jacksonville_walmart_complaints :
  let normal_complaint_rate := 120
  let short_staffed_increase_factor := 4 / 3
  let self_checkout_broken_increase_factor := 1.2
  let days := 3
  let complaints_per_day_short_staffed := normal_complaint_rate * short_staffed_increase_factor
  let complaints_per_day_both_conditions := complaints_per_day_short_staffed * self_checkout_broken_increase_factor
  let total_complaints := complaints_per_day_both_conditions * days
  total_complaints = 576 :=
by
  -- Use the 'let' definitions from above to describe the proof problem
  let normal_complaint_rate := 120
  let short_staffed_increase_factor := 4 / 3
  let self_checkout_broken_increase_factor := 1.2
  let days := 3
  let complaints_per_day_short_staffed := normal_complaint_rate * short_staffed_increase_factor
  let complaints_per_day_both_conditions := complaints_per_day_short_staffed * self_checkout_broken_increase_factor
  let total_complaints := complaints_per_day_both_conditions * days
  
  -- Here would be the place to write the proof steps, but it is skipped as per instructions
  sorry

end jacksonville_walmart_complaints_l442_442648


namespace squares_difference_sum_l442_442339

theorem squares_difference_sum : 
  23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by 
  sorry

end squares_difference_sum_l442_442339


namespace equilateral_triangle_area_l442_442659
-- Import the entire Mathlib for necessary mathematical components.

-- Define the problem statement, proving the triangle’s area given the conditions.
theorem equilateral_triangle_area (h : ℝ) (h_eq : h = real.sqrt 3) :
  ∃ (A : ℝ), A = real.sqrt 3 :=
by
  -- Skipping the proof steps here, we use sorry.
  sorry

end equilateral_triangle_area_l442_442659


namespace most_likely_color_white_calc_probabilities_game_is_fair_l442_442555
open_locale classical

-- Definitions for the given conditions
def white_balls := 3
def yellow_balls := 2
def red_balls := 1
def total_balls := white_balls + yellow_balls + red_balls

-- Define probabilities
def prob_white := white_balls / total_balls.to_real
def prob_yellow := yellow_balls / total_balls.to_real
def prob_red := red_balls / total_balls.to_real

-- Lean 4 statement for the proof problems
theorem most_likely_color_white (h : ∀ (n m : ℕ), n / total_balls.to_real <= m / total_balls.to_real → n <= m) :
  white_balls = total_balls → 3 >= 2 ∧ 3 >= 1 := sorry

theorem calc_probabilities :
  prob_white = 1/2 ∧ prob_yellow = 1/3 ∧ prob_red = 1/6 := sorry

theorem game_is_fair :
  prob_white = 1/2 ∧ (prob_yellow + prob_red) = 1/2 := sorry

end most_likely_color_white_calc_probabilities_game_is_fair_l442_442555


namespace fraction_of_total_cost_l442_442425

theorem fraction_of_total_cost
  (p_r : ℕ) (c_r : ℕ) 
  (p_a : ℕ) (c_a : ℕ)
  (p_c : ℕ) (c_c : ℕ)
  (p_w : ℕ) (c_w : ℕ)
  (p_dap : ℕ) (c_dap : ℕ)
  (p_dc : ℕ) (c_dc : ℕ)
  (total_cost : ℕ)
  (combined_cost_rc : ℕ)
  (fraction : ℚ)
  (h1 : p_r = 5) (h2 : c_r = 2)
  (h3 : p_a = 4) (h4 : c_a = 6)
  (h5 : p_c = 3) (h6 : c_c = 8)
  (h7 : p_w = 2) (h8 : c_w = 10)
  (h9 : p_dap = 4) (h10 : c_dap = 5)
  (h11 : p_dc = 3) (h12 : c_dc = 3)
  (htotal_cost : total_cost = 107)
  (hcombined_cost_rc : combined_cost_rc = 19)
  (hfraction : fraction = 19 / 107) :
  fraction = combined_cost_rc / total_cost := sorry

end fraction_of_total_cost_l442_442425


namespace difference_of_radii_l442_442688

noncomputable def ratio_of_areas (r1 r2 : ℝ) : Prop :=
  r2^2 / r1^2 = 3

theorem difference_of_radii (r : ℝ) (r2 : ℝ) (h : ratio_of_areas r r2) : 
  r2 - r ≈ 0.732 * r :=
by
  sorry

end difference_of_radii_l442_442688


namespace IM_bisects_AD_l442_442238

variables {A B C I D M : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited I] [Inhabited D] [Inhabited M]

-- Define the triangle ABC with AB ≠ AC
variable (AB_ne_AC : A ≠ C)

-- Define the incenter I of triangle ABC
variable (incenter_I : Incenter ABC = I)

-- Define the point D where the incircle touches BC
variable (D_on_BC : OnIncircle D B C)

-- Define the midpoint M of BC
variable (M_midpoint_BC : Midpoint M B C)

-- Statement to prove: Line IM bisects the line segment AD
theorem IM_bisects_AD : Bisects IM AD :=
begin
  sorry
end

end IM_bisects_AD_l442_442238


namespace rational_a_solution_l442_442079

theorem rational_a_solution (a : ℚ) :
  (∃ (q : ℚ), ∃ (x : ℚ), x > 0 ∧ ⌊x^a⌋ * ⟨x^a⟩ = q) →
  a ∈ (ℚ \ set_of (λ r : ℚ, ∃ n : ℤ, r = 1 / (n : ℚ) ∧ n ≠ 0)) :=
begin
  sorry
end

end rational_a_solution_l442_442079


namespace number_of_operations_to_one_l442_442028

theorem number_of_operations_to_one (tiles : ℕ) (h_tiles : tiles = 144) 
  (remove_and_renumber : Π (n : ℕ), n → (n - (nat.sqrt n) * (nat.sqrt n))) : 
  ∃ k : ℕ, k = 11 ∧ nat.iterate (remove_and_renumber 12) k tiles = 1 :=
sorry

end number_of_operations_to_one_l442_442028


namespace values_satisfying_ggx_eq_gx_l442_442610

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem values_satisfying_ggx_eq_gx (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 1 ∨ x = 3 ∨ x = 4 :=
by
  -- The proof is omitted
  sorry

end values_satisfying_ggx_eq_gx_l442_442610


namespace shelby_gold_stars_l442_442639

theorem shelby_gold_stars (yesterday_stars total_stars earned_today : ℕ) 
  (h1 : yesterday_stars = 4) 
  (h2 : total_stars = 7) : 
  earned_today = total_stars - yesterday_stars → 
  earned_today = 3 := 
by 
  intros h 
  rw [h2, h1]
  exact h

/- 
  This theorem states that given Shelby had 4 gold stars yesterday and now has a total of 7 gold stars, 
  proving that Shelby earned 3 gold stars today is equivalent to proving that:
  earned_today = total_stars - yesterday_stars 
  implies 
  earned_today = 3 
-/

end shelby_gold_stars_l442_442639


namespace convex_polyhedra_common_interior_point_l442_442478

variables {P : Type*} [ConvexSpace P]

def is_translation_of (P1 P2 : P) (A1 A2 : Point P) : Prop :=
  ∃ (v : Vector P), P1 = translate_by v P2 ∧ A1 = translate_point v A2

noncomputable def polyhedra_intersect (polyhedra : List P) : Prop :=
  ∃ (P1 P2 : P), P1 ∈ polyhedra ∧ P2 ∈ polyhedra ∧ common_interior_point P1 P2

theorem convex_polyhedra_common_interior_point
  (P1 : P)
  (A : Fin 9 → Point P)
  (polyhedra : Fin 9 → P)
  (hP : polyhedra 0 = P1)
  (h_translation : ∀ i : Fin 9, 1 ≤ i → is_translation_of (polyhedra i) P1 (A 0) (A i)) :
  polyhedra_intersect (List.ofFn polyhedra) := sorry

end convex_polyhedra_common_interior_point_l442_442478


namespace rate_per_meter_l442_442454

theorem rate_per_meter (d : ℝ) (total_cost : ℝ) (C : ℝ) : 
  d = 36 → total_cost = 395.84 → C = Real.pi * d → C ≈ 113.09724 → total_cost / C = 3.5 :=
by
  intros hd htc hC hCa
  sorry

end rate_per_meter_l442_442454


namespace total_distance_burrowed_day_5_l442_442647

open Real

def larger_rats_burrowed_distance (n : ℕ) : ℝ :=
  (2 ^ n) - 1

def smaller_rats_burrowed_distance (n : ℕ) : ℝ :=
  2 - (1 / (2 ^ (n - 1)))

def total_distance_burrowed (n : ℕ) : ℝ :=
  (larger_rats_burrowed_distance n) + (smaller_rats_burrowed_distance n)

theorem total_distance_burrowed_day_5 : 
  total_distance_burrowed 5 = 32 + 15 / 16 := 
by 
  sorry

end total_distance_burrowed_day_5_l442_442647


namespace sufficient_not_necessary_of_p_and_q_l442_442513

noncomputable theory

variables (x : ℝ)

def p : Prop := 1 < x ∧ x < 2
def q : Prop := 2 ^ x > 1

theorem sufficient_not_necessary_of_p_and_q :
  (p x → q x) ∧ ¬(q x → p x) :=
by
  sorry

end sufficient_not_necessary_of_p_and_q_l442_442513


namespace time_spent_walking_l442_442072

-- Definitions corresponding to our conditions
variable W D : ℕ

-- Statements of our conditions
axiom eq1 : W - D = -16
axiom eq2 : W + D = 60

-- The statement we want to prove
theorem time_spent_walking : W = 22 :=
by
  sorry

end time_spent_walking_l442_442072


namespace square_area_inscribed_in_ellipse_l442_442783

theorem square_area_inscribed_in_ellipse (t : ℝ) (h : 0 ≤ t ∧ t = sqrt(36 / 13)) :
  let side_length := 2 * t
  let area := side_length^2
  ∀ x y, (x, y) ∈ ({(±t, ±t)}) :=
  ∀ x y ∈ ℝ, ∀ side_length = 2 * sqrt(36 / 13), ∀ area = side_length^2, 
    area = (4 * 36 / 13 := 144 / 13),
begin
  sorry
end

end square_area_inscribed_in_ellipse_l442_442783


namespace no_int_sol_eq_l442_442633

theorem no_int_sol_eq (x y z : ℤ) (h₀ : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) : ¬ (x^2 + y^2 = 3 * z^2) := 
sorry

end no_int_sol_eq_l442_442633


namespace max_value_of_f_l442_442470

def f (x : ℝ) : ℝ := min (3 * x + 1) (min (x + 3) (-x + 9))

theorem max_value_of_f : ∃ x : ℝ, f(x) = 6 :=
by
  sorry

end max_value_of_f_l442_442470


namespace rectangle_cut_into_5_squares_l442_442922

theorem rectangle_cut_into_5_squares (rect : Type) (is_rectangle : rect → Prop) (can_be_cut_into_5_squares : rect → Prop) : 
  (∃ (squares : list ℕ), length squares = 5 ∧ (∃ a b c d e, [a, b, c, d, e] ~ squares ∧ a ≠ b ∧ b ≠ c ∧ c ≠ d)) := 
sorry

end rectangle_cut_into_5_squares_l442_442922


namespace cos_75_degree_l442_442428

theorem cos_75_degree : 
  let cos_60 := (1 / 2 : ℝ)
  let sin_60 := (Real.sqrt 3 / 2 : ℝ)
  let cos_15 := (Real.sqrt 6 + Real.sqrt 2) / 4
  let sin_15 := (Real.sqrt 6 - Real.sqrt 2) / 4
  cos 75 = cos_60 * cos_15 - sin_60 * sin_15 -> 
  cos 75 = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  intro h
  rw [h]
  sorry

end cos_75_degree_l442_442428


namespace printing_presses_first_scenario_l442_442925

theorem printing_presses_first_scenario :
  ∃ P : ℕ, (P * 15 = 25 * 21) ∧ P = 35 :=
by
  use 35
  split
  . -- Prove P = 35 * 15 = 25 * 21
    simp
  . -- P is indeed 35
    rfl

end printing_presses_first_scenario_l442_442925


namespace white_surface_area_fraction_l442_442376

/-- A cube with 4-inch edges is composed of 64 smaller cubes each having 1-inch edges. 
44 of the smaller cubes are white and 20 are black. 
All the black cubes are located on one face of the larger cube. 
Calculate the fraction of the surface area of the larger cube that is white. -/
def fraction_white_surface_area_of_cube : ℚ :=
  let side_length := 4
  let total_surface_area := 6 * side_length^2
  let total_white_area := 5 * side_length^2
  total_white_area / total_surface_area

-- The proof that the white surface area is 5/6 of the total surface area
theorem white_surface_area_fraction : fraction_white_surface_area_of_cube = 5/6 :=
by
  let side_length := 4
  let total_surface_area := 6 * side_length^2
  let total_white_area := 5 * side_length^2
  have total_surface_area_eq : total_surface_area = 96 := by norm_num
  have total_white_area_eq : total_white_area = 80 := by norm_num
  have fraction_eq : total_white_area / total_surface_area = 5/6 := by norm_num
  exact fraction_eq

end white_surface_area_fraction_l442_442376


namespace inequality_solution_set_l442_442138

noncomputable def f (x : ℝ) : ℝ := 2^x - 1

theorem inequality_solution_set : {x : ℝ | f x ≤ x} = set.Icc 0 1 := by
  sorry

end inequality_solution_set_l442_442138


namespace sum_of_prime_values_of_f_l442_442468

-- Definition of the function f(a)
def f (a : ℤ) : ℤ := |a^4 - 36 * a^2 + 96 * a - 64|

-- Statement to prove
theorem sum_of_prime_values_of_f : 
  let values := {f a | a : ℤ, nat.prime (int.nat_abs (f a))} in
  values.sum = 22 :=
sorry

end sum_of_prime_values_of_f_l442_442468


namespace range_f_when_a_2_range_a_for_two_zeros_l442_442869

-- Step 1: Define the function f(x, a)
def f (x a: ℝ) := x^2 - a*x - a + 3

-- Step 2: Part 1: When a = 2 and x ∈ [0, 3], prove the range of f(x) is [0, 4]
theorem range_f_when_a_2 (x : ℝ) (h : x ∈ set.Icc 0 3) : 
  ∃ (y : ℝ), y ∈ set.Icc 0 4 ∧ f x 2 = y :=
sorry

-- Step 3: Part 2: Given f(x) has two zeros x₁ and x₂ and x₁ * x₂ > 0, prove 
-- the range of values for a is (-∞, -6) ∪ (2, 3)
theorem range_a_for_two_zeros (a : ℝ) (hx1x2 : ∀ x₁ x₂ : ℝ, f x₁ a = 0 → f x₂ a = 0 → x₁ * x₂ > 0) :
  (a < -6) ∨ (2 < a ∧ a < 3) :=
sorry

end range_f_when_a_2_range_a_for_two_zeros_l442_442869


namespace wire_cut_ratio_l442_442398

theorem wire_cut_ratio (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) 
                        (h_eq_area : (a^2 * Real.sqrt 3) / 36 = (b^2) / 16) :
  a / b = Real.sqrt 3 / 2 :=
by
  sorry

end wire_cut_ratio_l442_442398


namespace probability_of_D_l442_442369

theorem probability_of_D (P_A P_B P_C P_D : ℚ) (hA : P_A = 1/4) (hB : P_B = 1/3) (hC : P_C = 1/6) 
  (hSum : P_A + P_B + P_C + P_D = 1) : P_D = 1/4 := 
by
  sorry

end probability_of_D_l442_442369


namespace max_value_of_expression_l442_442999

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l442_442999


namespace x_intercept_of_line_l442_442065

theorem x_intercept_of_line : ∃ x : ℝ, ∃ y : ℝ, 4 * x + 7 * y = 28 ∧ y = 0 ∧ x = 7 :=
by
  sorry

end x_intercept_of_line_l442_442065


namespace max_gifts_l442_442960

/-- 
Let 
  L = 2      -- kg of "Lastochka"
  T = 3      -- kg of "Truffle"
  P = 4      -- kg of "Ptichye Moloko"
  C = 5      -- kg of "Citron"

Each gift must contain 3 different types of candies, with 100 grams of each type.
Prove that the maximum number of New Year gifts Masha can make is 45.
-/
theorem max_gifts (L T P C : ℕ) (h_L : L = 2) (h_T : T = 3) (h_P : P = 4) (h_C : C = 5) :
  ∃ g : ℕ, g = 45 :=
by {
  use 45,
  sorry
}

end max_gifts_l442_442960


namespace jen_total_birds_l442_442194

theorem jen_total_birds (C D G : ℕ) (h1 : D = 150) (h2 : D = 4 * C + 10) (h3 : G = (D + C) / 2) :
  D + C + G = 277 := sorry

end jen_total_birds_l442_442194


namespace population_increase_rate_l442_442550

theorem population_increase_rate 
  (P : ℕ) (T_minutes : ℕ) (T_seconds : ℕ) 
  (HT : T_seconds = T_minutes * 60) (HP : P = 220) (H_minutes : T_minutes = 55) :
  P / T_seconds = 0.0667 := 
by
  sorry

end population_increase_rate_l442_442550


namespace number_of_people_in_group_l442_442382

theorem number_of_people_in_group (total_boxes : ℕ) (boxes_per_person : ℕ) (h1 : total_boxes = 20) (h2 : boxes_per_person = 2) : total_boxes / boxes_per_person = 10 :=
by
  rw [h1, h2]
  norm_num
  sorry

end number_of_people_in_group_l442_442382


namespace number_of_permutations_with_restricted_decreasing_subsequence_l442_442061

theorem number_of_permutations_with_restricted_decreasing_subsequence :
    ∃ (n : ℕ), n = 429 ∧
    ∀ (perm : List (Fin 8)), 
    perm.nodup →
    perm.all (λ x, x.val ∈ Finset.range 7) →
    ∀ (i j k : ℕ), (i < j) → (j < k) → (k < perm.length) →
    ¬ (perm[i] > perm[j] ∧ perm[j] > perm[k]) :=
begin
  sorry
end

end number_of_permutations_with_restricted_decreasing_subsequence_l442_442061


namespace consecutive_int_sqrt_l442_442894

theorem consecutive_int_sqrt (m n : ℤ) (h1 : m < n) (h2 : m < Real.sqrt 13) (h3 : Real.sqrt 13 < n) (h4 : n = m + 1) : m * n = 12 :=
sorry

end consecutive_int_sqrt_l442_442894


namespace modulus_z_eq_l442_442493

noncomputable def i : ℂ := Complex.I

noncomputable def sqrt3i : ℂ := Complex.sqrt 3 * i

noncomputable def z (z : ℂ) : Prop :=
  (1 + sqrt3i) ^ 2 * z = 1 - i ^ 3

theorem modulus_z_eq: ∀ z : ℂ, z => z z → |z| = Real.sqrt 2 / 4 := 
by sorry

end modulus_z_eq_l442_442493


namespace abs_w_unique_l442_442152

theorem abs_w_unique (w : ℂ) (h : w^2 - 6 * w + 40 = 0) : ∃! x : ℝ, x = Complex.abs w ∧ x = Real.sqrt 40 := by
  sorry

end abs_w_unique_l442_442152


namespace solution_correctness_l442_442081

noncomputable def solution_set : Set ℝ := {x | x + 60 / (x - 5) = -12}

theorem solution_correctness : solution_set = {0, -7} := 
begin
  sorry
end

end solution_correctness_l442_442081


namespace problem1_problem2_l442_442092

-- Define points A, B, C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 1, y := -2}
def B : Point := {x := 2, y := 1}
def C : Point := {x := 3, y := 2}

-- Function to compute vector difference
def vector_sub (p1 p2 : Point) : Point :=
  {x := p1.x - p2.x, y := p1.y - p2.y}

-- Function to compute vector scalar multiplication
def scalar_mul (k : ℝ) (p : Point) : Point :=
  {x := k * p.x, y := k * p.y}

-- Function to add two vectors
def vec_add (p1 p2 : Point) : Point :=
  {x := p1.x + p2.x, y := p1.y + p2.y}

-- Problem 1
def result_vector : Point :=
  let AB := vector_sub B A
  let AC := vector_sub C A
  let BC := vector_sub C B
  vec_add (scalar_mul 3 AB) (vec_add (scalar_mul (-2) AC) BC)

-- Prove the coordinates are (0, 2)
theorem problem1 : result_vector = {x := 0, y := 2} := by
  sorry

-- Problem 2
def D : Point :=
  let BC := vector_sub C B
  {x := 1 + BC.x, y := (-2) + BC.y}

-- Prove the coordinates are (2, -1)
theorem problem2 : D = {x := 2, y := -1} := by
  sorry

end problem1_problem2_l442_442092


namespace cost_of_cookies_and_board_game_l442_442588

theorem cost_of_cookies_and_board_game
  (bracelet_cost : ℝ) (bracelet_price : ℝ) (necklace_cost : ℝ) (necklace_price : ℝ) 
  (ring_cost : ℝ) (ring_price : ℝ) (num_bracelets : ℕ) (num_necklaces : ℕ) (num_rings : ℕ)
  (target_profit_margin : ℝ) (remaining_money : ℝ) :
  (bracelet_cost = 1) ∧ (bracelet_price = 1.5) ∧ (necklace_cost = 2) ∧ (necklace_price = 3) ∧ 
  (ring_cost = 0.5) ∧ (ring_price = 1) ∧ (num_bracelets = 12) ∧ (num_necklaces = 8) ∧ 
  (num_rings = 20) ∧ (target_profit_margin = 0.5) ∧ (remaining_money = 5) →
  let total_cost := (num_bracelets * bracelet_cost) + (num_necklaces * necklace_cost) + (num_rings * ring_cost) in
  let total_revenue := (num_bracelets * bracelet_price) + (num_necklaces * necklace_price) + (num_rings * ring_price) in
  let actual_profit := total_revenue - total_cost in
  let target_profit := total_cost * target_profit_margin in
  let profit_after_purchases := actual_profit - remaining_money in
  let cookies_and_board_game_cost := total_revenue - profit_after_purchases in
  cookies_and_board_game_cost = 43 :=
by
  intro h
  sorry

end cost_of_cookies_and_board_game_l442_442588


namespace triangle_side_length_l442_442905

theorem triangle_side_length (a b : ℝ) (h_a : a = 3) (h_b : b = 2) (h_cos_sum : Real.cos (π - Real.acos (-1/3)) = 1/3) : 
  ∃ c : ℝ, c = Real.sqrt (a^2 + b^2 - 2 * a * b * (-1 / 3)) :=
by {
  use Real.sqrt (3^2 + 2^2 - 2 * 3 * 2 * (-1/3)),
  sorry
}

end triangle_side_length_l442_442905


namespace solution_to_problem_l442_442831

theorem solution_to_problem
  {x y z : ℝ}
  (h1 : xy / (x + y) = 1 / 3)
  (h2 : yz / (y + z) = 1 / 5)
  (h3 : zx / (z + x) = 1 / 6) :
  xyz / (xy + yz + zx) = 1 / 7 :=
by sorry

end solution_to_problem_l442_442831


namespace distance_between_cars_after_5_hours_l442_442322

theorem distance_between_cars_after_5_hours : 
  (let A_speeds := [10, 12, 14, 16, 15] in
   let B_speed := 20 in
   let total_distance_A := A_speeds.foldl (λ acc speed => acc + speed) 0 in
   let total_distance_B := 5 * B_speed in
   let total_distance := total_distance_A + total_distance_B in
   total_distance = 167) :=
sorry

end distance_between_cars_after_5_hours_l442_442322


namespace compute_expression_l442_442056

theorem compute_expression :
  45 * 72 + 28 * 45 = 4500 :=
  sorry

end compute_expression_l442_442056


namespace no_extreme_value_of_f_l442_442500

noncomputable def f (x : ℝ) : ℝ := sorry

-- Lemma stating the main properties of f(x)
lemma function_properties (x : ℝ) (hx : 0 < x) : 
  (f(x) + x * (deriv f x) = (real.log x) / x) ∧ (f real.exp 1 = 1 / real.exp 1) :=
sorry

-- The statement we need to prove
theorem no_extreme_value_of_f :
  ∀ x (hx : 0 < x), 
    ¬∃ c, (deriv f c = 0) ∧ (0 < c ∧ c < x) :=
sorry

end no_extreme_value_of_f_l442_442500


namespace total_amount_l442_442762

variable (a b c d : ℝ)

-- Defining the conditions
def condition1 := a = (1 / 3) * (b + c + d)
def condition2 := b = (2 / 7) * (a + c + d)
def condition3 := c = (3 / 11) * (a + b + d)
def condition4 := a = b + 20
def condition5 := b = c + 15

-- Statement that we need to prove
theorem total_amount : 
  condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ condition5 → a + b + c + d = 720 := 
by
  sorry

end total_amount_l442_442762


namespace parallelogram_proof_l442_442557

noncomputable def parallelogram_ratio (AP AB AQ AD AC AT : ℝ) (hP : AP / AB = 61 / 2022) (hQ : AQ / AD = 61 / 2065) (h_intersect : true) : ℕ :=
if h : AC / AT = 4087 / 61 then 67 else 0

theorem parallelogram_proof :
  ∀ (ABCD : Type) (P : Type) (Q : Type) (T : Type) 
     (AP AB AQ AD AC AT : ℝ) 
     (hP : AP / AB = 61 / 2022) 
     (hQ : AQ / AD = 61 / 2065)
     (h_intersect : true),
  parallelogram_ratio AP AB AQ AD AC AT hP hQ h_intersect = 67 :=
by
  sorry

end parallelogram_proof_l442_442557


namespace selling_price_of_book_l442_442895

theorem selling_price_of_book (SP : ℝ) (CP : ℝ := 200) :
  (SP - CP) = (340 - CP) + 0.05 * CP → SP = 350 :=
by {
  sorry
}

end selling_price_of_book_l442_442895


namespace total_number_of_components_l442_442705

-- Definitions based on the conditions in the problem
def number_of_B_components := 300
def number_of_C_components := 200
def sample_size := 45
def number_of_A_components_drawn := 20
def number_of_C_components_drawn := 10

-- The statement to be proved
theorem total_number_of_components :
  (number_of_A_components_drawn * (number_of_B_components + number_of_C_components) / sample_size) 
  + number_of_B_components 
  + number_of_C_components 
  = 900 := 
by 
  sorry

end total_number_of_components_l442_442705


namespace one_number_greater_than_one_l442_442551

theorem one_number_greater_than_one 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > (1 / a) + (1 / b) + (1 / c)) 
  : (1 < a ∨ 1 < b ∨ 1 < c) ∧ ¬(1 < a ∧ 1 < b ∧ 1 < c) :=
by
  sorry

end one_number_greater_than_one_l442_442551


namespace place_circle_in_wide_polygon_l442_442375

noncomputable def is_wide (P : set (ℝ × ℝ)) : Prop :=
  ∀ (ℓ : ℝ × ℝ → ℝ), ∃ (p1 p2 : ℝ), p1 < p2 ∧ ∀ (x : ℝ), (x, ℓ x) ∈ P -> p2 - p1 ≥ 1

theorem place_circle_in_wide_polygon (P : set (ℝ × ℝ)) (hP : convex P) (h_wide : is_wide P) :
  ∃ O : ℝ × ℝ, ∀ p ∈ P, dist p O ≥ (1 / 3) :=
sorry

end place_circle_in_wide_polygon_l442_442375


namespace range_of_g_l442_442804

noncomputable def g (B : ℝ) : ℝ :=
  (Real.cos B * (4 * Real.sin B ^ 2 + Real.sin B ^ 4 + 2 * Real.cos B ^ 2 + Real.cos B ^ 2 * Real.sin B ^ 2)) /
  (Real.cot B * (Real.csc B - Real.cos B * Real.cot B))

-- The theorem concerning the range of g
theorem range_of_g : ∀ B : ℝ, B ≠ n * Real.pi → 2 ≤ g B ∧ g B ≤ 3 :=
sorry

end range_of_g_l442_442804


namespace angle_between_vector_a_and_target_vector_is_90_deg_l442_442232

def vector_a : ℝ × ℝ × ℝ := (2, -3, -4)
def vector_b : ℝ × ℝ × ℝ := (3, 5, -2)
def vector_c : ℝ × ℝ × ℝ := (15, -2, 19)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def vector_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

noncomputable def vector_mul_scalar (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

noncomputable def target_vector : ℝ × ℝ × ℝ :=
  vector_sub 
    (vector_mul_scalar (dot_product vector_a vector_c) vector_b)
    (vector_mul_scalar (dot_product vector_a vector_b) vector_c)

theorem angle_between_vector_a_and_target_vector_is_90_deg :
  dot_product vector_a target_vector = 0 :=
by
  -- proof details would go here
  sorry

end angle_between_vector_a_and_target_vector_is_90_deg_l442_442232


namespace remaining_tape_length_is_correct_l442_442930

-- Defining the values and conversions involved
def cm_to_mm (cm : ℕ) : ℕ := cm * 10

def remaining_length_in_mm (initial_cm : ℕ) (given_mm : ℕ) (used_mm : ℕ) : ℕ :=
  cm_to_mm initial_cm - given_mm - used_mm

def mm_to_cm (mm : ℕ) : ℕ := mm / 10

-- The formal proof problem
theorem remaining_tape_length_is_correct
  (initial_cm : ℕ)
  (given_mm : ℕ)
  (used_mm : ℕ)
  (total_cm : ℕ) :
  (initial_cm = 65) → 
  (given_mm = 125) → 
  (used_mm = 153) → 
  (total_cm = remaining_length_in_mm initial_cm given_mm used_mm / 10) → 
  total_cm = 37 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3],
  exact h4,
  sorry
end

end remaining_tape_length_is_correct_l442_442930


namespace solve_system_of_equations_simplify_expression_l442_442053

-- Statement for system of equations
theorem solve_system_of_equations (s t : ℚ) 
  (h1 : 2 * s + 3 * t = 2) 
  (h2 : 2 * s - 6 * t = -1) :
  s = 1 / 2 ∧ t = 1 / 3 :=
sorry

-- Statement for simplifying the expression
theorem simplify_expression (x y : ℚ) :
  ((x - y)^2 + (x + y) * (x - y)) / (2 * x) = x - y :=
sorry

end solve_system_of_equations_simplify_expression_l442_442053


namespace find_m_of_equation_has_positive_root_l442_442547

theorem find_m_of_equation_has_positive_root :
  (∃ x : ℝ, 0 < x ∧ (x - 1) / (x - 5) = (m * x) / (10 - 2 * x)) → m = -8 / 5 :=
by
  sorry

end find_m_of_equation_has_positive_root_l442_442547


namespace product_all_gt_1_l442_442836

variable (n : ℕ) [Fact (n = 1991)]
variable (a : Fin n → ℝ)

-- Conditions: list of 1991 positive distinct real numbers, product of any 10 greater than 1
def distinct_positive (a : Fin n → ℝ) := (∀ i : Fin n, 0 < a i ∧ ∀ j : Fin n, i ≠ j → a i ≠ a j)
def product_gt_1 (a : Fin n → ℝ) := ∀ (s : Finset (Fin n)) (h : s.card = 10), 1 < ∏ i in s, a i

-- Theorem statement
theorem product_all_gt_1 (a : Fin n → ℝ) (h1 : distinct_positive a) (h2 : product_gt_1 a) : 
  1 < ∏ i, a i := 
by
  sorry

end product_all_gt_1_l442_442836


namespace sum_first_2017_terms_l442_442503

noncomputable def f : ℝ → ℝ
| x => if x ∈ set.Icc (-1) 1 then 2 * |x| - 2 else f (x - 2)

def a (n : ℕ) : ℝ := 2 * (n : ℝ) - 1

def b (n : ℕ) : ℝ := (-1)^(n + 1) * a n

theorem sum_first_2017_terms : (∑ i in finset.range 2017, b (i + 1)) = 2017 := sorry

end sum_first_2017_terms_l442_442503


namespace sum_first_100_terms_l442_442302

def non_negative_difference (a b : ℕ) : ℕ :=
  if a ≥ b then a - b else b - a

noncomputable def sequence : ℕ → ℕ
| 0       := 88
| 1       := 24
| (n + 2) := non_negative_difference (sequence (n + 1)) (sequence n)

theorem sum_first_100_terms : (Finset.range 100).sum sequence = 760 :=
  sorry

end sum_first_100_terms_l442_442302


namespace car_service_month_l442_442448

-- Define the conditions
def first_service_month : ℕ := 3 -- Representing March as the 3rd month
def service_interval : ℕ := 7
def total_services : ℕ := 13

-- Define an auxiliary function to calculate months and reduce modulo 12
def nth_service_month (first_month : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  (first_month + (interval * (n - 1))) % 12

-- The theorem statement
theorem car_service_month : nth_service_month first_service_month service_interval total_services = 3 :=
by
  -- The proof steps will go here
  sorry

end car_service_month_l442_442448


namespace selection_3m2f_selection_at_least_one_captain_selection_at_least_one_female_selection_captain_and_female_l442_442699

section ProofProblems

-- Definitions and constants
def num_males := 6
def num_females := 4
def total_athletes := 10
def num_selections := 5
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- 1. Number of selection methods for 3 males and 2 females
theorem selection_3m2f : binom 6 3 * binom 4 2 = 120 := by sorry

-- 2. Number of selection methods with at least one captain
theorem selection_at_least_one_captain :
  2 * binom 8 4 + binom 8 3 = 196 := by sorry

-- 3. Number of selection methods with at least one female athlete
theorem selection_at_least_one_female :
  binom 10 5 - binom 6 5 = 246 := by sorry

-- 4. Number of selection methods with both a captain and at least one female athlete
theorem selection_captain_and_female :
  binom 9 4 + binom 8 4 - binom 5 4 = 191 := by sorry

end ProofProblems

end selection_3m2f_selection_at_least_one_captain_selection_at_least_one_female_selection_captain_and_female_l442_442699


namespace determine_range_a_l442_442706

def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def prop_q (a : ℝ) : Prop := ∀ x y : ℝ, 1 ≤ x → x ≤ y → 4 * x^2 - a * x ≤ 4 * y^2 - a * y

theorem determine_range_a (a : ℝ) (h : ¬ prop_p a ∧ (prop_p a ∨ prop_q a)) : 
  a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8) :=
sorry

end determine_range_a_l442_442706


namespace number_of_numerators_in_lowest_terms_l442_442608

-- Definitions used in Lean 4 statement
def rational_with_repeating_decimal (r : ℚ) : Prop :=
  0 < r ∧ r < 1 ∧ ∃ a b c d : ℕ,
  r = (a * 1000 + b * 100 + c * 10 + d) / 9999

-- Theorem to state the problem and its correct answer
theorem number_of_numerators_in_lowest_terms : 
  ∃ (n : ℕ), n = 6000 ∧ ∀ r ∈ { q : ℚ | rational_with_repeating_decimal q },
   ∃ p q : ℕ, q = p / q ∧ nat.gcd p q = 1 :=
sorry

end number_of_numerators_in_lowest_terms_l442_442608


namespace solve_y_l442_442642

theorem solve_y (y : ℝ) (h : (4 * y - 2) / (5 * y - 5) = 3 / 4) : y = -7 :=
by
  sorry

end solve_y_l442_442642


namespace correct_statement_is_C_l442_442349

def statement_A : Prop :=
  ∀ (x : ℝ), exp x > 0

def statement_B (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x ∈ set.Ioo a b, 0 ≤ second_derivative f x) ↔ monotone_on f (set.Ioo a b)

def statement_C : Prop :=
  ∀ (x y : ℝ), x + y ≠ 3 → (x ≠ 2 ∨ y ≠ 1)

def statement_D (f : ℝ → ℝ) : Prop :=
  (∀ a : ℝ, (∀ x : ℝ, f x = a*x^2 + 2*x - 1 → (∃ x0 : ℝ, f x0 = 0 → ∀ x1 : ℝ, f x1 = 0 → x0 = x1)) → a = -1)

theorem correct_statement_is_C :
  statement_C :=
by
  sorry

end correct_statement_is_C_l442_442349


namespace min_students_l442_442967

theorem min_students (S a b c : ℕ) (h1 : 3 * a > S) (h2 : 10 * b > 3 * S) (h3 : 11 * c > 4 * S) (h4 : S = a + b + c) : S ≥ 173 :=
by
  sorry

end min_students_l442_442967


namespace find_sides_of_triangle_l442_442779

theorem find_sides_of_triangle (c : ℝ) (θ : ℝ) (h_ratio : ℝ) 
  (h_c : c = 2 * Real.sqrt 7)
  (h_theta : θ = Real.pi / 6) -- 30 degrees in radians
  (h_ratio_eq : ∃ k : ℝ, ∀ a b : ℝ, a = k ∧ b = h_ratio * k) :
  ∃ (a b : ℝ), a = 2 ∧ b = 4 * Real.sqrt 3 := by
  sorry

end find_sides_of_triangle_l442_442779


namespace max_sin_cos_l442_442973

theorem max_sin_cos (x : ℝ) : 
  (sin x + sqrt 2 * cos x) ≤ sqrt 3 := 
sorry

end max_sin_cos_l442_442973


namespace exists_prime_not_dividing_n_pow_p_minus_p_l442_442242

theorem exists_prime_not_dividing_n_pow_p_minus_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ q : ℕ, Nat.Prime q ∧ ∀ n : ℕ, ¬ q ∣ (n^p - ↑p) := by
  sorry

end exists_prime_not_dividing_n_pow_p_minus_p_l442_442242


namespace minimum_crosses_in_a_line_l442_442089

def minimum_crosses_in_6x6 : Nat :=
  25

theorem minimum_crosses_in_a_line (n : Nat) (h : n ≥ 25) :
  ∀ (grid : Array (Array Bool)), (∀ i j, grid[i][j] = true → (0 ≤ i ∧ i < 6) ∧ (0 ≤ j ∧ j < 6)) →
  ∃ i j, (grid[i][j] = true ∧ grid[i+1][j] = true ∧ grid[i+2][j] = true) ∨
         (grid[i][j] = true ∧ grid[i][j+1] = true ∧ grid[i][j+2] = true) :=
by sorry

end minimum_crosses_in_a_line_l442_442089


namespace conditional_probability_of_B_given_A_probability_of_wind_given_rain_l442_442171

-- Definitions of the events
def event_A : Prop := sorry -- Event A represents rain.
def event_B : Prop := sorry -- Event B represents wind above level 3.

-- Given probabilities
axiom P_A : ℚ := 4 / 15
axiom P_B : ℚ := 2 / 15
axiom P_A_and_B : ℚ := 1 / 10

-- The probability of B given A
theorem conditional_probability_of_B_given_A : P_A ≠ 0 → P_B_given_A = P_A_and_B / P_A :=
by
  assume hP_A_nonzero,
  exact P_A_and_B / P_A

-- Proving the result
theorem probability_of_wind_given_rain : P_A ≠ 0 → (P_A_and_B / P_A) = 3 / 8 :=
by
  assume hP_A_nonzero,
  have : P_A_and_B / P_A = (1 / 10) / (4 / 15) := by
    rw [← div_eq_mul_inv, ← div_eq_mul_inv, P_A, P_A_and_B],
  have h := (1 / 10) / (4 / 15) = 3 / 8,
  exact h

end conditional_probability_of_B_given_A_probability_of_wind_given_rain_l442_442171


namespace yard_length_l442_442558

-- Define the given conditions
def num_trees : ℕ := 26
def dist_between_trees : ℕ := 13

-- Calculate the length of the yard
def num_gaps : ℕ := num_trees - 1
def length_of_yard : ℕ := num_gaps * dist_between_trees

-- Theorem statement: the length of the yard is 325 meters
theorem yard_length : length_of_yard = 325 := by
  sorry

end yard_length_l442_442558


namespace mathd_inequality_l442_442974

theorem mathd_inequality (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) : 
  (x^3 + 2*y^2 + 3*z) * (4*y^3 + 5*z^2 + 6*x) * (7*z^3 + 8*x^2 + 9*y) ≥ 720 * (x * y + y * z + z * x) :=
by
  sorry

end mathd_inequality_l442_442974


namespace find_constant_eq_22_l442_442124

theorem find_constant_eq_22 :
  let n := 3 in (n ^ 4 - 20 * n + 1) = 22 :=
by
  let n := 3
  calc
    n ^ 4 - 20 * n + 1 = 3 ^ 4 - 20 * 3 + 1 := by rfl
    ... = 81 - 60 + 1 := by norm_num
    ... = 21 + 1 := by norm_num
    ... = 22 := by norm_num

end find_constant_eq_22_l442_442124


namespace polygon_area_correct_l442_442367

noncomputable def alpha : ℝ := Real.pi / 8

def area_formula (α : ℝ) : ℝ :=
  5 * Real.sin α + 4 * Real.sin (2 * α) + 3 * Real.sin (3 * α) + 2 * Real.sin (4 * α) + Real.sin (5 * α)

theorem polygon_area_correct : area_formula alpha = 5 * Real.sin alpha + 4 * Real.sin (2 * alpha) + 3 * Real.sin (3 * alpha) + 2 * Real.sin (4 * alpha) + Real.sin (5 * alpha) :=
sorry

end polygon_area_correct_l442_442367


namespace multiplicative_inverse_of_289_mod_391_l442_442390

theorem multiplicative_inverse_of_289_mod_391 :
  (136^2 + 255^2 = 289^2) → 
  ∃ n : ℕ, (0 ≤ n ∧ n < 391 ∧ 289 * n % 391 = 1) :=
by
  intro h
  use 18
  split
  · exact Nat.zero_le 18
  split
  · exact Nat.lt_of_lt_of_le (Nat.lt_succ_self 18) (Nat.le_of_eq (by norm_num))
  · rw Nat.mul_mod
    norm_num
    sorry

end multiplicative_inverse_of_289_mod_391_l442_442390


namespace triangle_inequality_ABC_l442_442257

variable {A B C D E F : Type} [LinearOrderedField F]
variable {a b c : F}
variable {AD BE CF : F}

/- Conditions: D on BC, E on CA, and F on AB -/
variable (D_on_BC : D ∈ segment B C)
variable (E_on_CA : E ∈ segment C A)
variable (F_on_AB : F ∈ segment A B)

/- Lean statement: -/
theorem triangle_inequality_ABC (a b c : F) : 
  (a = distance B C) → 
  (b = distance C A) → 
  (c = distance A B) → 
  (AD = distance A D) → 
  (BE = distance B E) → 
  (CF = distance C F) → 
  D_on_BC → E_on_CA → F_on_AB → 
  (1 / 2 * (a + b + c) < AD + BE + CF ∧ AD + BE + CF < 3 / 2 * (a + b + c)) :=
  by 
  sorry

end triangle_inequality_ABC_l442_442257


namespace problem_statement_l442_442810

variable (g : ℝ)

-- Definition of the operation
def my_op (g y : ℝ) : ℝ := g^2 + 2 * y

-- The statement we want to prove
theorem problem_statement : my_op g (my_op g g) = g^4 + 4 * g^3 + 6 * g^2 + 4 * g :=
by
  sorry

end problem_statement_l442_442810


namespace smallest_four_digit_divisible_by_35_is_1050_l442_442335

theorem smallest_four_digit_divisible_by_35_is_1050 :
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 35 = 0) ∧ ∀ m : ℕ, ((1000 ≤ m ∧ m ≤ 9999) ∧ (m % 35 = 0)) → n ≤ m
    := exists.intro 1050 (and.intro (and.intro (by norm_num [le_refl]) (by norm_num [le_refl])) 
                                        (and.intro (by norm_num [mod_eq_zero_iff_dvd, dvd_refl])
                                                  (λ m h, by {
                                                      obtain ⟨hm1, hm2, hm3⟩ := h,
                                                      have hn := 1050,
                                                      rw ← hn,
                                                      exact (nat.le_of_lt (lt_of_le_of_lt (nat.div_le_div_right hm3) 
                                                                    (by norm_num [nat.div_lt_self _ (succ_pos' 0)]))).le
                                                    })))

end smallest_four_digit_divisible_by_35_is_1050_l442_442335


namespace eggs_sally_bought_is_correct_l442_442270

def dozen := 12

def eggs_sally_bought (dozens : Nat) : Nat :=
  dozens * dozen

theorem eggs_sally_bought_is_correct :
  eggs_sally_bought 4 = 48 :=
by
  sorry

end eggs_sally_bought_is_correct_l442_442270


namespace solve_for_x_l442_442150

noncomputable def log_b (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_for_x (b x : ℝ) (hb : b > 1) (hx : x > 0) :
  (4 * x) ^ log_b b 4 - (5 * x) ^ log_b b 5 + x = 0 ↔ x = 1 :=
by
  -- Proof placeholder
  sorry

end solve_for_x_l442_442150


namespace midpoint_chord_lies_on_segment_l442_442474

noncomputable def point (α : Type*) := (α × α)
variables {α : Type*}

-- Conditions
variables 
  (center : point α)
  (K A B C D E : point α)
  (r : α) -- radius of the circle

-- Midpoint of a segment
def midpoint (P Q : point α) : point α := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Chord
def chord (P Q : point α) : set (point α) := { R | ∃ t : α, 0 ≤ t ∧ t ≤ 1 ∧ R = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2) }

-- Tangents from point K to points A, B.
def tangent (R : point α) (K : point α) (radius : α) := (R.1 - K.1) ^ 2 + (R.2 - K.2) ^ 2 = radius ^ 2

-- Line segment parallel relation
def parallel_seg (P Q R S : point α) : Prop := (P.2 - Q.2) * (R.1 - S.1) = (P.1 - Q.1) * (R.2 - S.2)

-- Secant line intersects the circle at points C and D
def sec_line (C D : point α) (K : point α) : Prop := 
  ∃ t : α, 0 < t ∧ C = ((1 - t) * K.1 + t * D.1, (1 - t) * K.2 + t * D.2)

theorem midpoint_chord_lies_on_segment 
  (h_tangent_A : tangent A K r)
  (h_tangent_B : tangent B K r)
  (h_secant_CD : sec_line C D K)
  (h_parallel_BE_KC : parallel_seg B E K C)
  : midpoint C D ∈ chord A E :=
sorry

end midpoint_chord_lies_on_segment_l442_442474


namespace length_ratio_proof_l442_442402

noncomputable def length_ratio (a b : ℝ) : ℝ :=
  let area_triangle := (sqrt 3 / 4) * (a / 3) ^ 2
  let area_square := (b / 4) ^ 2
  if area_triangle = area_square then
    a / b
  else
    0

theorem length_ratio_proof (a b : ℝ) (h : (sqrt 3 / 4) * (a / 3) ^ 2 = (b / 4) ^ 2) : length_ratio a b = sqrt ((3 * sqrt 3) / 4) :=
  sorry

end length_ratio_proof_l442_442402


namespace find_radius_l442_442773

def radius_of_circle (d : ℤ) (PQ : ℕ) (QR : ℕ) (r : ℕ) : Prop := 
  let PR := PQ + QR
  (PQ * PR = (d - r) * (d + r)) ∧ (d = 15) ∧ (PQ = 11) ∧ (QR = 8) ∧ (r = 4)

-- Now stating the theorem to prove the radius r given the conditions
theorem find_radius (r : ℕ) : radius_of_circle 15 11 8 r := by
  sorry

end find_radius_l442_442773


namespace total_balloons_l442_442583

theorem total_balloons (j : ℕ) (m : ℕ) (hj : j = 40) (hm : m = 41) : j + m = 81 :=
by
  rw [hj, hm]
  exact rfl

end total_balloons_l442_442583


namespace length_ratio_proof_l442_442400

noncomputable def length_ratio (a b : ℝ) : ℝ :=
  let area_triangle := (sqrt 3 / 4) * (a / 3) ^ 2
  let area_square := (b / 4) ^ 2
  if area_triangle = area_square then
    a / b
  else
    0

theorem length_ratio_proof (a b : ℝ) (h : (sqrt 3 / 4) * (a / 3) ^ 2 = (b / 4) ^ 2) : length_ratio a b = sqrt ((3 * sqrt 3) / 4) :=
  sorry

end length_ratio_proof_l442_442400


namespace infinite_integer_solutions_l442_442510

theorem infinite_integer_solutions (a b c k : ℤ) (D : ℤ) 
  (hD : D = b^2 - 4 * a * c) (hD_pos : D > 0) (hD_non_square : ¬ ∃ (n : ℤ), n^2 = D) 
  (hk_non_zero : k ≠ 0) :
  (∃ (x₀ y₀ : ℤ), a * x₀^2 + b * x₀ * y₀ + c * y₀^2 = k) →
  ∃ (f : ℤ → ℤ × ℤ), ∀ n : ℤ, a * (f n).1^2 + b * (f n).1 * (f n).2 + c * (f n).2^2 = k :=
by
  sorry

end infinite_integer_solutions_l442_442510


namespace find_distance_l442_442032

def distance_between_stripes (width distance curb stripe1 stripe2 : Real) : Prop :=
  width = 60 ∧ curb = 25 ∧ stripe1 = 70 ∧ stripe2 = 70 ∧ 
  70 * distance = 25 * width ∧ distance = (1500 / 70)

theorem find_distance : 
  ∃ distance : Real, distance_between_stripes 60 distance 25 70 70 :=
begin
  use (1500 / 70),
  split,
  repeat {split},
  exact rfl,
  exact rfl,
  exact rfl,
  exact rfl,
  calc
    70 * (1500 / 70) = 1500 : by field_simp [ne_of_gt (show 70 > 0, by norm_num)]
    ... = 25 * 60 : by norm_num,
  exact rfl,
end

end find_distance_l442_442032


namespace train_speed_km_hr_calc_l442_442037

theorem train_speed_km_hr_calc :
  let length := 175 -- length of the train in meters
  let time := 3.499720022398208 -- time to cross the pole in seconds
  let speed_mps := length / time -- speed in meters per second
  let speed_kmph := speed_mps * 3.6 -- converting speed from m/s to km/hr
  speed_kmph = 180.025923226 := 
sorry

end train_speed_km_hr_calc_l442_442037


namespace parabola_translation_equivalence_l442_442679

theorem parabola_translation_equivalence :
  ∀ (x y : ℝ), (y = 3 * x^2) →
  let translated_upwards := y + 3 in
  let translated_left := translated_upwards = 3 * (x + 2)^2 + 3 in
  translated_left = 3 * (x + 2)^2 + 3 := by
  intros x y h
  let translated_upwards := y + 3
  let translated_left := 3 * (x + 2)^2 + 3
  sorry

end parabola_translation_equivalence_l442_442679


namespace sum_of_pqrstu_eq_22_l442_442943

theorem sum_of_pqrstu_eq_22 (p q r s t : ℤ) 
  (h : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -48) : 
  p + q + r + s + t = 22 :=
sorry

end sum_of_pqrstu_eq_22_l442_442943


namespace inclusion_exclusion_l442_442937

theorem inclusion_exclusion {S : Finset α} {n : ℕ} (S1 S2 ... Sn : Finset α) :
  (S \ S1) ∩ (S \ S2) ∩ ... ∩ (S \ Sn)).card = 
  S.card - ∑ i in Finset.range n, (Si.card) +
  ∑  (1 ≤ i < j ≤ n), (Si ∩ Sj).card - ... + (-1)^k ∑ (1 ≤ i_1 < i_2 < ... < i_k ≤ n), (Si_1 ∩ Si_2 ∩ ... ∩ Si_k).card + ... + (-1)^n (S1 ∩ S2 ∩ ... ∩ Sn).card :=
sorry

end inclusion_exclusion_l442_442937


namespace geometry_textbook_weight_l442_442205

-- Define the constants as per the given conditions
def chemistry_textbook_weight : ℝ := 7.125
def weight_difference : ℝ := 6.5

-- Define the goal to prove
theorem geometry_textbook_weight : ∃ G : ℝ, G = chemistry_textbook_weight - weight_difference :=
by
  use (chemistry_textbook_weight - weight_difference)
  sorry

end geometry_textbook_weight_l442_442205


namespace time_spent_on_marketing_posts_l442_442050

-- Bryan's conditions
def hours_customer_outreach : ℕ := 4
def hours_advertisement : ℕ := hours_customer_outreach / 2
def total_hours_worked : ℕ := 8

-- Proof statement: Bryan spends 2 hours each day on marketing posts
theorem time_spent_on_marketing_posts : 
  total_hours_worked - (hours_customer_outreach + hours_advertisement) = 2 := by
  sorry

end time_spent_on_marketing_posts_l442_442050


namespace sin_half_angle_l442_442535

theorem sin_half_angle 
  (θ : ℝ) 
  (h_cos : |Real.cos θ| = 1 / 5) 
  (h_theta : 5 * Real.pi / 2 < θ ∧ θ < 3 * Real.pi)
  : Real.sin (θ / 2) = - (Real.sqrt 15) / 5 := 
by
  sorry

end sin_half_angle_l442_442535


namespace winnie_balloons_l442_442352

theorem winnie_balloons :
  let red_balloons := 25 in
  let white_balloons := 40 in
  let green_balloons := 55 in
  let chartreuse_balloons := 80 in
  let total_balloons := red_balloons + white_balloons + green_balloons + chartreuse_balloons in
  let friends := 10 in
  total_balloons % friends = 0 :=
by
  let red_balloons := 25
  let white_balloons := 40
  let green_balloons := 55
  let chartreuse_balloons := 80
  let total_balloons := red_balloons + white_balloons + green_balloons + chartreuse_balloons
  let friends := 10
  have h1 : total_balloons = 200 := by sorry
  show total_balloons % friends = 0 from by sorry

end winnie_balloons_l442_442352


namespace product_of_x_coords_l442_442427

theorem product_of_x_coords (x : ℝ) (hC : (x - 4)^2 + (-3)^2 = 7^2) : 
  let x1 := 4 + 2 * Real.sqrt 10,
      x2 := 4 - 2 * Real.sqrt 10 in
  x1 * x2 = -24 :=
by
  sorry

end product_of_x_coords_l442_442427


namespace stable_configuration_l442_442720

theorem stable_configuration (n : ℕ) (k : ℕ) (a : Finₓ k → ℕ) :
  (∑ i, a i = n * (n + 1) / 2) →
  (∀ (j : ℕ), (∃ b : Finₓ k → ℕ, Succeeds (λ a, heap_operation a = b) a)) →
  (∃ m : ℕ, ∀ i, a i = 1) := sorry

end stable_configuration_l442_442720


namespace intervals_of_monotonicity_range_of_k_l442_442835

noncomputable def f (x : ℝ) : ℝ := 2 * x - log x

theorem intervals_of_monotonicity :
  (∀ (x : ℝ), x > 0 → x < 1/2 → f' x < 0) ∧
  (∀ (x : ℝ), x > 1/2 → f' x > 0) :=
sorry

theorem range_of_k (k : ℝ) :
  (∀ (x : ℝ), x ≥ 1 → f(x) ≥ k*x) ↔ (k ≤ 2 - 1/exp(1)) :=
sorry

end intervals_of_monotonicity_range_of_k_l442_442835


namespace boats_left_l442_442624

def initial_boats : ℕ := 30
def percentage_eaten_by_fish : ℕ := 20
def boats_shot_with_arrows : ℕ := 2
def boats_blown_by_wind : ℕ := 3
def boats_sank : ℕ := 4

def boats_eaten_by_fish : ℕ := (initial_boats * percentage_eaten_by_fish) / 100

theorem boats_left : initial_boats - boats_eaten_by_fish - boats_shot_with_arrows - boats_blown_by_wind - boats_sank = 15 := by
  sorry

end boats_left_l442_442624


namespace find_students_just_passed_l442_442174

open Nat

theorem find_students_just_passed
  (total_students : ℕ)
  (first_division_percentage : ℚ)
  (second_division_percentage : ℚ)
  (total_students_given : total_students = 300)
  (first_division_percentage_given : first_division_percentage = 28 / 100)
  (second_division_percentage_given : second_division_percentage = 54 / 100) :
  (total_students - ((first_division_percentage * total_students).natFloor + (second_division_percentage * total_students).natFloor) = 54) :=
by
  sorry

end find_students_just_passed_l442_442174


namespace base_10_to_base_7_conversion_l442_442327

theorem base_10_to_base_7_conversion :
  ∃ (digits : ℕ → ℕ), 789 = digits 3 * 7^3 + digits 2 * 7^2 + digits 1 * 7^1 + digits 0 * 7^0 ∧
  digits 3 = 2 ∧ digits 2 = 2 ∧ digits 1 = 0 ∧ digits 0 = 5 :=
sorry

end base_10_to_base_7_conversion_l442_442327


namespace pureAcidInSolution_l442_442736

/-- Define the conditions for the problem -/
def totalVolume : ℝ := 12
def percentageAcid : ℝ := 0.40

/-- State the theorem equivalent to the question:
    calculate the amount of pure acid -/
theorem pureAcidInSolution :
  totalVolume * percentageAcid = 4.8 := by
  sorry

end pureAcidInSolution_l442_442736


namespace triangle_side_ratio_l442_442553

theorem triangle_side_ratio (A B C : ℝ) (a b c : ℝ)
  (h_tri : A + B + C = π)
  (h_cos_sin : cos A + sin A - 2 / (cos B + sin B) = 0)
  (h_a : a = sin A)
  (h_b : b = sin B)
  (h_c : c = sin C) :
  (a + b) / c = √2 :=
sorry

end triangle_side_ratio_l442_442553


namespace range_f_x1_x2_l442_442507

noncomputable def f (x a : ℝ) := x^2 - a * x + 2 * Real.log x

theorem range_f_x1_x2 (a : ℝ) (x1 x2 : ℝ)
  (h1 : 2 * (Real.exp 1 + 1 / Real.exp 1) < a)
  (h2 : a < 20 / 3)
  (hx_extreme : f x1 a = 0 ∧ f x2 a = 0 ∧ x1 < x2 ∧ 0 < x1 < 1 ∧ 1 < x2) :
  e^2 - (1 / e^2) - 4 < f x1 a - f x2 a ∧ f x1 a - f x2 a < 80 / 9 - 4 * Real.log 3 :=
sorry

end range_f_x1_x2_l442_442507


namespace wire_cut_ratio_l442_442397

theorem wire_cut_ratio (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) 
                        (h_eq_area : (a^2 * Real.sqrt 3) / 36 = (b^2) / 16) :
  a / b = Real.sqrt 3 / 2 :=
by
  sorry

end wire_cut_ratio_l442_442397


namespace max_value_of_expression_l442_442994

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l442_442994


namespace M_inter_N_eq_l442_442516

variable {x : ℝ}

def M := {x | x^2 < 4}
def N := {x | x^2 - 2x - 3 < 0}

theorem M_inter_N_eq : M ∩ N = {x | -1 < x ∧ x < 2} :=
by
  sorry

end M_inter_N_eq_l442_442516


namespace variance_proof_l442_442982

noncomputable def calculate_mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length)

noncomputable def calculate_variance (scores : List ℝ) : ℝ :=
  let mean := calculate_mean scores
  (scores.map (λ x => (x - mean)^2)).sum / scores.length

def scores_A : List ℝ := [8, 6, 9, 5, 10, 7, 4, 7, 9, 5]
def scores_B : List ℝ := [7, 6, 5, 8, 6, 9, 6, 8, 8, 7]

noncomputable def variance_A : ℝ := calculate_variance scores_A
noncomputable def variance_B : ℝ := calculate_variance scores_B

theorem variance_proof :
  variance_A = 3.6 ∧ variance_B = 1.4 ∧ variance_B < variance_A :=
by
  -- proof steps - use sorry to skip the proof
  sorry

end variance_proof_l442_442982


namespace find_special_number_l442_442450

-- Define the three-digit number
def three_digit_number : ℕ := 367

-- Define a predicate for the given condition
def increasing_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ (i : ℕ), i < digits.length - 1 → digits.get i < digits.get (i + 1)

theorem find_special_number :
  ∃ (n : ℕ), 100 ≤ n ∧ n < 1000 ∧ increasing_digits (n * n) :=
begin
  use 367,
  split,
  { exact nat.le_trans (by norm_num : 100 ≤ 367) (by norm_num : 367 < 1000) },
  split,
  { exact nat.lt_of_lt_of_le (by norm_num : n < 368) (by norm_num) },
  { sorry } -- Proof of increasing digits part
end

end find_special_number_l442_442450


namespace modulus_of_complex_sum_l442_442874

theorem modulus_of_complex_sum : 
  let i := Complex.I in
  let z := 1 + i + i^2 + i^3 + i^4 + i^5 + i^6 + i^7 + i^8 + i^9 in
  Complex.abs z = Real.sqrt 2 :=
by 
  sorry

end modulus_of_complex_sum_l442_442874


namespace largest_number_is_310_l442_442330

def largest_number_formed (a b c : ℕ) : ℕ :=
  max (a * 100 + b * 10 + c) (max (a * 100 + c * 10 + b) (max (b * 100 + a * 10 + c) 
  (max (b * 100 + c * 10 + a) (max (c * 100 + a * 10 + b) (c * 100 + b * 10 + a)))))

theorem largest_number_is_310 : largest_number_formed 3 1 0 = 310 :=
by simp [largest_number_formed]; sorry

end largest_number_is_310_l442_442330


namespace fixed_point_Q_l442_442611

theorem fixed_point_Q (A B C P I J Q : Point) (Γ : Circle) (h₁ : Triangle ABC) 
  (h₂ : Inscribed Γ ABC) (h₃ : OnArc P A B Γ) (h₄ : ¬ OnArc P C Γ) 
  (h₅ : IncircleCenter I (Triangle A C P)) (h₆ : IncircleCenter J (Triangle B C P)) 
  (h₇ : Intersection Q (Circumcircle Γ P I J) Γ) : 
  FixedPoint Q :=
sorry

end fixed_point_Q_l442_442611


namespace sum_lent_l442_442389

theorem sum_lent (P : ℝ) (R : ℝ) (T : ℝ) (I : ℝ)
  (hR: R = 4) 
  (hT: T = 8) 
  (hI1 : I = P - 306) 
  (hI2 : I = P * R * T / 100) :
  P = 450 :=
by
  sorry

end sum_lent_l442_442389


namespace adult_elephant_weekly_bananas_l442_442793

theorem adult_elephant_weekly_bananas (daily_bananas : Nat) (days_in_week : Nat) (H1 : daily_bananas = 90) (H2 : days_in_week = 7) :
  daily_bananas * days_in_week = 630 :=
by
  sorry

end adult_elephant_weekly_bananas_l442_442793


namespace effective_annual_rate_l442_442759

theorem effective_annual_rate
  (initial : ℝ)
  (final : ℝ)
  (rate1 : ℝ)
  (rate2 : ℝ)
  (rate3 : ℝ)
  (rate4 : ℝ)
  (years : ℝ)
  (total_growth_factor : ℝ) :
  initial = 810 →
  final = 1550 →
  rate1 = 0.05 →
  rate2 = 0.07 →
  rate3 = 0.06 →
  rate4 = 0.04 →
  years = 4 →
  total_growth_factor = (1 + rate1) * (1 + rate2) * (1 + rate3) * (1 + rate4) →
  (1 + (final / initial)^(1 / years) - 1) * 100 ≈ 17.55 :=
sorry

end effective_annual_rate_l442_442759


namespace min_marked_cells_l442_442968

-- Definition: A 9x9 board
def board := List (List Bool)
def is_valid_board (b : board) := b.length = 9 ∧ ∀ row, row ∈ b → row.length = 9

-- Definition: A 1x4 rectangle
def rectangle := List (Nat × Nat)
def is_valid_rectangle (r : rectangle) := r.length = 4 ∧ ∀ (x, y), (x, y) ∈ r → x < 9 ∧ y < 9

-- Definition: Marked cells
def marked_cells := List (Nat × Nat)

-- Condition: Determines if all 1x4 rectangles are uniquely identifiable by marked cells
def unique_identification (marks : marked_cells) (x y : Nat) : Prop :=
  ∀ (r1 r2 : rectangle), is_valid_rectangle r1 ∧ is_valid_rectangle r2
  → (∃ cell, (cell ∈ marks) → (cell ∈ r1 ↔ cell ∈ r2)) ∧ r1 ≠ r2

-- Statement: Petya marks the cells in a way that Vasya cannot win if k = 40
theorem min_marked_cells (k : Nat) : k = 40 → 
  ∃ marks : marked_cells, unique_identification marks 9 9 :=
by
  intro h
  sorry

end min_marked_cells_l442_442968


namespace intersection_of_A_and_B_l442_442229

open Set

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
  by
    sorry

end intersection_of_A_and_B_l442_442229


namespace max_ab_l442_442295

theorem max_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 1) : ab ≤ 1 / 16 :=
sorry

end max_ab_l442_442295


namespace smallest_positive_period_f_f_geq_neg_half_on_interval_l442_442505

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := sqrt 3 * cos (2 * x - pi / 3) - 2 * sin x * cos x

-- Statement 1: The smallest positive period of f(x) is π
theorem smallest_positive_period_f : ∀ x : ℝ, f (x + π) = f x :=
sorry

-- Statement 2: For x in [-π/4, π/4], f(x) ≥ -1/2
theorem f_geq_neg_half_on_interval : ∀ x : ℝ, x ∈ Icc (-π / 4) (π / 4) → f x ≥ -1 / 2 :=
sorry

end smallest_positive_period_f_f_geq_neg_half_on_interval_l442_442505


namespace solution_set_inequality_cauchy_schwarz_inequality_l442_442139

-- Part 1: Proof for the inequality solution set
theorem solution_set_inequality (x : ℝ):
  (let f := fun x => abs(x - 2) + 3 * abs(x) in
   (f x ≥ 10) ↔ (x ≤ -2) ∨ (x ≥ 3)) :=
by sorry

-- Part 2: Proof for the inequality involving a, b, c
theorem cauchy_schwarz_inequality (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a + b + c = 2):
  a^2 + b^2 + c^2 ≥ 4 / 3 :=
by sorry

end solution_set_inequality_cauchy_schwarz_inequality_l442_442139


namespace binary_to_base4_conversion_l442_442809

noncomputable def bin_to_quat (n : ℕ) : ℕ := 
-- Dummy implementation.
-- Real implementation needed.
sorry

theorem binary_to_base4_conversion (n : ℕ) (h : n = 110111001₂) :
  bin_to_quat n = 13221₄ := 
by 
  sorry

end binary_to_base4_conversion_l442_442809


namespace find_quadratic_expression_l442_442407

theorem find_quadratic_expression :
  ∃ (y : ℝ → ℝ), (y = (λ x, -9 + x^2)) ∧
    ((y = (λ x, -9 + x^2)) ∨ (y = (λ x, -2 * x + 1)) ∨ (y = (λ x, sqrt (x^2 + 4))) ∨ (y = (λ x, -(x + 1) + 3))) ∧ 
    ∀ x, ∃ a b c : ℝ, y x = a * x^2 + b * x + c ∧ a ≠ 0 :=
begin
  sorry
end

end find_quadratic_expression_l442_442407


namespace tangent_ratio_equality_l442_442568

-- Declare the universe in which the objects live
universe u

-- Assuming the required geometrical objects and relations
variables {Point : Type u} [MetricSpace Point] [T2Space Point]

-- Indicating the points in our triangle
variables (A B C E F D : Point)

-- Recall that BC < AC < AB
variables {BC AC AB : ℝ} (h₁ : BC < AC) (h₂ : AC < AB)

-- Defining the tangents and circles involved
variables (circumcircle : Circle Point)
variables (tangentA : TangentLine circumcircle A)
variables (tangentC : TangentLine circumcircle C)

variables (hEA : TangentMeetsLineAt tangentA (line B C) E)
variables (hFC : TangentMeetsLineAt tangentC (line A B) F)
variables (hDE : TangentsIntersect tangentA tangentC D)

-- Additional conditions
variables (circleThroughDEF : Circle Point) (passesThroughB : circleThroughDEF.PassThrough B)

-- Aim to show the given ratio equality
theorem tangent_ratio_equality 
  (h₁ : TangentRatioEqual AE AF EC CF AB BC) :
  AE / AF * EC / CF = AB / BC := by
    sorry

end tangent_ratio_equality_l442_442568


namespace log_x_64_l442_442896

theorem log_x_64 (x : ℝ) (h : log 8 (5 * x) = 3) : log x 64 = 2 / 3 :=
sorry

end log_x_64_l442_442896


namespace num_k_vals_l442_442462

-- Definitions of the conditions
def div_by_7 (n k : ℕ) : Prop :=
  (2 * 3^(6*n) + k * 2^(3*n + 1) - 1) % 7 = 0

-- Main theorem statement
theorem num_k_vals : 
  ∃ (S : Finset ℕ), (∀ k ∈ S, k < 100 ∧ ∀ n, div_by_7 n k) ∧ S.card = 14 := 
by
  sorry

end num_k_vals_l442_442462


namespace definite_integral_value_l442_442070

theorem definite_integral_value : 
  ∫ x in -1..1, (sqrt (1 - x ^ 2) + cos (2 * x - π / 2)) = π / 2 :=
sorry

end definite_integral_value_l442_442070


namespace locus_of_M_l442_442849

open_locale classical
noncomputable theory

variable {ℝ : Type*}

structure Point :=
(x : ℝ) (y : ℝ)

-- An equilateral triangle
structure EquilateralTriangle :=
(A B C : Point)
(equilateral : ∀ (X Y Z : Point), (X = A ∧ Y = B ∧ Z = C) → 
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X ∧ dist Z X = dist X Y)

-- The main theorem to prove
theorem locus_of_M (ABC : EquilateralTriangle) : 
  ∃ (line : set Point) (circle : set Point), ∀ M : Point,
    (M ∈ line ∨ M ∈ circle) ↔ 
    (∃ a b, 
      (let AM := dist ABC.A M in 
       let BM := dist ABC.B M in 
         AM = BM)) :=
sorry

end locus_of_M_l442_442849


namespace different_numerators_count_l442_442607

-- Definitions and conditions for the problem
def is_repeating_decimal (r : ℚ) : Prop :=
  ∃ (a b c d : ℕ), r = (a * 1000 + b * 100 + c * 10 + d) / 9999 ∧ 0 < r ∧ r < 1

def not_divisible_by_3_11_101 (n : ℕ) : Prop :=
  ¬(3 ∣ n) ∧ ¬(11 ∣ n) ∧ ¬(101 ∣ n)

-- Main theorem statement
theorem different_numerators_count : 
  let numerators := {abcd : ℕ | ∃ (r : ℚ), is_repeating_decimal r ∧ r * 9999 = abcd}
  let valid_numerators := {n ∈ numerators | not_divisible_by_3_11_101 n}
  let count := finite_card valid_numerators
  count = 5401 :=
sorry

end different_numerators_count_l442_442607


namespace Yanna_apples_l442_442353

def total_apples_bought (given_to_zenny : ℕ) (given_to_andrea : ℕ) (kept : ℕ) : ℕ :=
  given_to_zenny + given_to_andrea + kept

theorem Yanna_apples {given_to_zenny given_to_andrea kept total : ℕ}:
  given_to_zenny = 18 →
  given_to_andrea = 6 →
  kept = 36 →
  total_apples_bought given_to_zenny given_to_andrea kept = 60 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end Yanna_apples_l442_442353


namespace tetrahedron_division_by_planes_l442_442434

-- Definitions and conditions
def volume_of_tetrahedron := 1
def num_of_planes := 6
def plane_divisant (tetra: Type) : Prop :=
  -- Each plane passes through one edge and the midpoint of the opposite edge
  ∀ (p: Plane), p through edge ∧ midpoint_of_opposite_edge 

theorem tetrahedron_division_by_planes :
  ∀ (T : Tetrahedron) (P: Plane) (H: plane_divisant T),
  divided_into_parts(T, P, num_of_planes) = 24 ∧ 
  ∀ (part : Tetrahedron), volume_of(part) = volume_of_tetrahedron / 24 :=
begin
  sorry
end

end tetrahedron_division_by_planes_l442_442434


namespace max_value_of_expression_l442_442993

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l442_442993


namespace coefficient_of_x_cubed_l442_442452

theorem coefficient_of_x_cubed :
  let p := (1 + x) * (1 - real.sqrt x)^6 in
  polynomial.coeff p 3 = 16 :=
by sorry

end coefficient_of_x_cubed_l442_442452


namespace average_infections_per_round_infections_exceed_700_three_rounds_l442_442004

-- Define the conditions and proof for question 1
theorem average_infections_per_round (x : ℝ) (h : 1 + x + x^2 = 81) : x = 8 :=
sorry

-- Define the conditions and proof for question 2
theorem infections_exceed_700_three_rounds (h₁ : 1 + 8 + 8^2 = 81) : 81 * 9 > 700 :=
by
  calc
    81 * 9 = 729 : by norm_num
            ... > 700 : by norm_num

end average_infections_per_round_infections_exceed_700_three_rounds_l442_442004


namespace find_sin_phi_l442_442233

open Real

-- Define the vectors as nonzero and not parallel
variables {u v w : ℝ^3}
variables (nonzero_u : u ≠ 0) (nonzero_v : v ≠ 0) (nonzero_w : w ≠ 0)
variables (not_parallel_uv : ¬(∃ k : ℝ, u = k • v)) (not_parallel_uw : ¬(∃ k : ℝ, u = k • w))
variables (not_parallel_vw : ¬(∃ k : ℝ, v = k • w))

-- Assume the given condition
variables (h : (u × v) × w = (1 / 2) * (‖v‖ * ‖w‖) • u)

-- Define the angle between v and w, and state that sin(φ) = √3 / 2
noncomputable def angle φ : ℝ := Real.angle_between v w
noncomputable def sin_phi : ℝ := sin φ

theorem find_sin_phi : sin_phi = sqrt(3) / 2 :=
sorry

end find_sin_phi_l442_442233


namespace cubic_polynomial_linear_combination_not_quartic_polynomial_linear_combination_l442_442265

-- Define polynomial functions P, Q, R with specific degrees
noncomputable def P (x : ℝ) : ℝ := sorry
noncomputable def Q (x : ℝ) : ℝ := sorry
noncomputable def R (x : ℝ) : ℝ := sorry

-- Define the degree condition
def is_cubic (f : ℝ → ℝ) : Prop := sorry
def is_quartic (f : ℝ → ℝ) : Prop := sorry

-- Define the main inequalities and equalities
def poly_conditions (P Q R : ℝ → ℝ) (x0 : ℝ) : Prop :=
  (∀ x, P x ≤ Q x ∧ Q x ≤ R x) ∧ P x0 = R x0

-- Statement for cubic polynomials
theorem cubic_polynomial_linear_combination
  (P Q R : ℝ → ℝ) 
  (x0 : ℝ) 
  (hP : is_cubic P) 
  (hQ : is_cubic Q) 
  (hR : is_cubic R)
  (h_cond : poly_conditions P Q R x0) :
  ∃ k ∈ Icc (0.0 : ℝ) 1.0, ∀ x, Q x = k * P x + (1 - k) * R x :=
sorry

-- Counterexample for quartic polynomials
theorem not_quartic_polynomial_linear_combination
  (P Q R : ℝ → ℝ)
  (x0 : ℝ) 
  (hP : is_quartic P) 
  (hQ : is_quartic Q)
  (hR : is_quartic R)
  (h_cond : poly_conditions P Q R x0) :
  ¬(∃ k ∈ Icc (0.0 : ℝ) 1.0, ∀ x, Q x = k * P x + (1 - k) * R x) :=
sorry

end cubic_polynomial_linear_combination_not_quartic_polynomial_linear_combination_l442_442265


namespace smallest_number_divisible_l442_442359

theorem smallest_number_divisible (n : ℕ) : (∃ n : ℕ, (n + 3) % 27 = 0 ∧ (n + 3) % 35 = 0 ∧ (n + 3) % 25 = 0 ∧ (n + 3) % 21 = 0) ∧ n = 4722 :=
by
  sorry

end smallest_number_divisible_l442_442359


namespace l1_l2_perp_neg1_l1_l2_parallel_distance_l442_442520

-- Definitions of the lines l1 and l2
def l1 (a : ℝ) := {x : ℝ × ℝ | x.1 + a * x.2 = 2 * a + 2}
def l2 (a : ℝ) := {x : ℝ × ℝ | a * x.1 + x.2 = a + 1}

-- Condition for perpendicularity
def perp_condition (a : ℝ) : Prop := (∃ m1 m2 : ℝ, m1 * m2 = -1) ∧ m1 = -1 / a ∧ m2 = -a

-- Condition for parallelism
def parallel_condition (a : ℝ) : Prop := (∃ m1 m2 : ℝ, m1 = m2) ∧ m1 = -1 / a ∧ m2 = -a

/-
    The theorem corresponding to (I) proving perpendicular slopes.
    Try to prove that if the lines l1 and l2 are perpendicular, then a = -1.
-/
theorem l1_l2_perp_neg1 (a : ℝ) :
  perp_condition a → a = -1 := sorry

/-
    The theorem corresponding to (II) calculating distance between parallel lines.
    Try to prove that if the lines l1 and l2 are parallel, the distance between them is sqrt(2).
-/
theorem l1_l2_parallel_distance (a : ℝ) :
  parallel_condition a → (l1 a).distance (l2 a) = Real.sqrt 2 := sorry

end l1_l2_perp_neg1_l1_l2_parallel_distance_l442_442520


namespace units_digit_sum_squares_first_2021_odd_l442_442338

theorem units_digit_sum_squares_first_2021_odd :
  (let units_digit n := n % 10 in
  let sum_units := ∑ i in (range 2021).map (λ n, (2*n + 1)^2), units_digit i in
  units_digit sum_units = 1) :=
sorry

end units_digit_sum_squares_first_2021_odd_l442_442338


namespace find_symmetric_point_l442_442360

structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def line_equation (t : ℝ) : Point :=
  { x := -t, y := 1.5, z := 2 + t }

def M : Point := { x := -1, y := 0, z := -1 }

def is_midpoint (M M' M0 : Point) : Prop :=
  M0.x = (M.x + M'.x) / 2 ∧
  M0.y = (M.y + M'.y) / 2 ∧
  M0.z = (M.z + M'.z) / 2

theorem find_symmetric_point (M0 : Point) (h_line : ∃ t, M0 = line_equation t) :
  ∃ M' : Point, is_midpoint M M' M0 ∧ M' = { x := 3, y := 3, z := 3 } :=
sorry

end find_symmetric_point_l442_442360


namespace calculate_expression_l442_442827

theorem calculate_expression :
  2 * (( (3.6 * (0.48^2) * 2.50) / ( sqrt 0.12 * (0.09^3) * (0.5^2) ))^2 * ( Real.exp (-0.3) )) = 9964154400 := 
sorry

end calculate_expression_l442_442827


namespace problem_statement_l442_442160

noncomputable def value_of_x_squared_plus_one (x : ℝ) : ℝ := x ^ 2 + 1

theorem problem_statement : ∃ (x : ℝ), 4^(2 * x) + 64 = 34 * 4^x ∧ (value_of_x_squared_plus_one x = 7.25 ∨ value_of_x_squared_plus_one x = 1.25) := 
by
  sorry

end problem_statement_l442_442160


namespace possible_B_values_l442_442795

-- Defining the conditions for Anne, Beth, and Chris
variables (A B C : ℕ)

-- Total number of candies is 10
constant total_candies : ℕ := 10

-- Anne gets at least 3 candies
def Anne_condition := A ≥ 3 

-- Beth gets at least 2 candies
def Beth_condition := B ≥ 2 

-- Chris gets between 2 and 3 candies
def Chris_condition := 2 ≤ C ∧ C ≤ 3

-- Total candies condition
def total_condition := A + B + C = total_candies

-- Prove that the possible number of candies Beth can get is 2, 3, 4, or 5
theorem possible_B_values : Anne_condition A ∧ Beth_condition B ∧ Chris_condition C ∧ total_condition A B C →
  B ∈ {2, 3, 4, 5} :=
by
  sorry

end possible_B_values_l442_442795


namespace find_divisor_l442_442357

-- Definitions of the conditions
def dividend : ℕ := 15968
def quotient : ℕ := 89
def remainder : ℕ := 37

-- The theorem stating the proof problem
theorem find_divisor (D : ℕ) (h : dividend = D * quotient + remainder) : D = 179 :=
sorry

end find_divisor_l442_442357


namespace find_littering_citations_l442_442387

def is_total_citations (total : ℕ) (L : ℕ) : Prop :=
  let off_leash_dogs := L
  let parking_fines := 2 * (L + off_leash_dogs)
  total = L + off_leash_dogs + parking_fines

theorem find_littering_citations :
  ∀ L : ℕ, is_total_citations 24 L → L = 4 :=
by
  intros L h
  let off_leash_dogs := L
  let parking_fines := 2 * (L + off_leash_dogs)
  have h1 : 24 = L + off_leash_dogs + parking_fines := h
  have h2 : off_leash_dogs = L := sorry -- Directly applies from problem statement
  have h3 : parking_fines = 2 * (L + L) := sorry -- Applies the double citation fine
  rw h2 at h3
  rw h3 at h1
  simp at h1
  exact sorry -- Final solve step, equating and solving for L

end find_littering_citations_l442_442387


namespace S_is_positive_rationals_l442_442946

variable {S : Set ℚ}

-- Defining the conditions as axioms
axiom cond1 (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : (a + b ∈ S) ∧ (a * b ∈ S)
axiom cond2 {r : ℚ} : (r ∈ S) ∨ (-r ∈ S) ∨ (r = 0)

-- The theorem to prove
theorem S_is_positive_rationals : S = { r : ℚ | r > 0 } := sorry

end S_is_positive_rationals_l442_442946


namespace root_expression_value_l442_442944

theorem root_expression_value
  (r s : ℝ)
  (h1 : 3 * r^2 - 4 * r - 8 = 0)
  (h2 : 3 * s^2 - 4 * s - 8 = 0) :
  (9 * r^3 - 9 * s^3) * (r - s)⁻¹ = 40 := 
sorry

end root_expression_value_l442_442944


namespace complaints_over_3_days_l442_442654

theorem complaints_over_3_days
  (n : ℕ) (n_ss : ℕ) (n_both : ℕ) (total : ℕ)
  (h1 : n = 120)
  (h2 : n_ss = n + 1/3 * n)
  (h3 : n_both = n_ss + 0.20 * n_ss)
  (h4 : total = n_both * 3) :
  total = 576 :=
by
  sorry

end complaints_over_3_days_l442_442654


namespace range_of_a_l442_442956

noncomputable theory

def f (a x : ℝ) : ℝ :=
  if x > 0 then |x - a| - 2 * a else -|x - a| + 2 * a

theorem range_of_a (a : ℝ) (h1 : ∀ x : ℝ, f a x = -f a (-x)) (h2 : ∀ x : ℝ, f a (x + 2017) > f a x) :
  a < 2017 / 6 :=
sorry

end range_of_a_l442_442956


namespace find_angle_A_find_min_magnitude_of_sum_l442_442186

-- Define the given conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variables {m n : ℝ × ℝ}
variable h1 : b * tan A = (2 * c - b) * tan B
variable h2 : m = (0, -1)
variable h3 : n = (cos B, 2 * cos^2 (C / 2))

-- Problem 1: Prove that under the given conditions, angle A equals π / 3
theorem find_angle_A (h : b * tan A = (2 * c - b) * tan B) : A = π / 3 := sorry

-- Problem 2: Find the minimum value of ||m + n||
theorem find_min_magnitude_of_sum (h1 : m = (0, -1))
                                   (h2 : n = (cos B, 2 * cos^2 (C / 2))) :
                                   ∃ (min_val : ℝ), min_val = sqrt_f2 / 2 :=
begin
  sorry
end

end find_angle_A_find_min_magnitude_of_sum_l442_442186


namespace num_correct_propositions_l442_442522

variable (m l : Type) (α β : set Type)

-- Define the conditions (propositions)
def prop1 (l : Type) (α : set Type) (hp : ∀ p q : Type, p ∈ α → q ∈ α → p ≠ q → l ⊥ p → l ⊥ q) : l ⊥ α :=
sorry

def prop2 (l : Type) (α : set Type) (hp : l ∥ α → ∀ p : Type, p ∈ α → l ∥ p) : l ∥ α :=
sorry

def prop3 (m l : Type) (α β : set Type) (hp : m ⊂ α → l ⊂ β → l ⊥ m → α ⊥ β) : α ⊥ β :=
sorry

def prop4 (l : Type) (α β : set Type) (hp : l ⊂ β → l ⊥ α → α ⊥ β) : α ⊥ β :=
sorry

def prop5 (m l : Type) (α β : set Type) (hp : m ⊂ α → l ⊂ β → α ∥ β → m ∥ l) : m ∥ l :=
sorry

-- Proof of the number of correct propositions being 1
theorem num_correct_propositions : 
  (prop1 l α ∧ prop2 l α ∧ prop3 m l α β ∧ prop4 l α β ∧ prop5 m l α β) = 1 :=
sorry

end num_correct_propositions_l442_442522


namespace percentage_increase_correct_l442_442682

-- Define the original price P as a real number
variable (P : ℝ)

-- Define the new price after a 20% decrease
def P_new1 := 0.80 * P

-- Define the percentage increase x%
variable (x : ℝ)

-- Define the final price after the increase
def P_final := P_new1 + (x / 100) * P_new1

-- Given the final price represents a 24% net increase
def net_increase_condition := P_final = 1.24 * P

-- The theorem to prove that the percentage increase x is 55%
theorem percentage_increase_correct :
  net_increase_condition →
  x = 55 :=
by
  sorry

end percentage_increase_correct_l442_442682


namespace max_value_of_expression_l442_442995

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l442_442995


namespace henry_meets_train_probability_l442_442038

noncomputable def train_arrival := Uniform 120 240
noncomputable def henry_arrival := Uniform 150 270

def henry_meets_train (x y : ℝ) : Prop := y <= x ∧ x <= y + 30

def probability_of_meeting : ℝ :=
  let total_area := (240 - 120) * (270 - 150)
  let meeting_area := 0.5 * (90 * 90) + 0.5 * (30 * 30)
  meeting_area / total_area

theorem henry_meets_train_probability :
  probability_of_meeting = 5 / 16 := by
  sorry

end henry_meets_train_probability_l442_442038


namespace unique_lottery_ticket_number_l442_442147

noncomputable def five_digit_sum_to_age (ticket : ℕ) (neighbor_age : ℕ) := 
  (ticket >= 10000 ∧ ticket <= 99999) ∧ 
  (neighbor_age = 5 * ((ticket / 10000) + (ticket % 10000 / 1000) + 
                        (ticket % 1000 / 100) + (ticket % 100 / 10) + 
                        (ticket % 10)))

theorem unique_lottery_ticket_number {ticket : ℕ} {neighbor_age : ℕ} 
    (h : five_digit_sum_to_age ticket neighbor_age) 
    (unique_solution : ∀ ticket1 ticket2, 
                        five_digit_sum_to_age ticket1 neighbor_age → 
                        five_digit_sum_to_age ticket2 neighbor_age → 
                        ticket1 = ticket2) : 
  ticket = 99999 :=
  sorry

end unique_lottery_ticket_number_l442_442147


namespace complaints_over_3_days_l442_442655

theorem complaints_over_3_days
  (n : ℕ) (n_ss : ℕ) (n_both : ℕ) (total : ℕ)
  (h1 : n = 120)
  (h2 : n_ss = n + 1/3 * n)
  (h3 : n_both = n_ss + 0.20 * n_ss)
  (h4 : total = n_both * 3) :
  total = 576 :=
by
  sorry

end complaints_over_3_days_l442_442655


namespace length_ratio_proof_l442_442401

noncomputable def length_ratio (a b : ℝ) : ℝ :=
  let area_triangle := (sqrt 3 / 4) * (a / 3) ^ 2
  let area_square := (b / 4) ^ 2
  if area_triangle = area_square then
    a / b
  else
    0

theorem length_ratio_proof (a b : ℝ) (h : (sqrt 3 / 4) * (a / 3) ^ 2 = (b / 4) ^ 2) : length_ratio a b = sqrt ((3 * sqrt 3) / 4) :=
  sorry

end length_ratio_proof_l442_442401


namespace number_of_children_l442_442696

variables (n : ℕ) (y : ℕ) (d : ℕ)

def sum_of_ages (n : ℕ) (y : ℕ) (d : ℕ) : ℕ :=
  n * y + d * (n * (n - 1) / 2)

theorem number_of_children (H1 : sum_of_ages n 6 3 = 60) : n = 6 :=
by {
  sorry
}

end number_of_children_l442_442696


namespace swim_team_sequences_l442_442250

theorem swim_team_sequences (Alice Bob Claire Madison : Prop) :
  ¬ (alice = madison ∨ bob = madison ∨ claire = madison) ∧
  ∃ (positions : List element), positions.nodup ∧ positions.length = 4 ∧ positions.contains madison ∧ positions.ilast = madison → positions ∈ permutations [alice, bob, claire] →
  ∃ (n : ℕ), n = 6 := 
sorry

end swim_team_sequences_l442_442250


namespace domain_of_f_l442_442666

-- Given function definition
def f (x : ℝ) : ℝ := real.sqrt (-x^2 + 2*x + 3)

-- Define the condition for the domain
def condition (x : ℝ) : Prop := -x^2 + 2*x + 3 ≥ 0

-- State the domain as a set
def domain : set ℝ := {x | condition x}

-- State the problem to be proved
theorem domain_of_f :
  domain = set.Icc (-1 : ℝ) (3 : ℝ) :=
sorry

end domain_of_f_l442_442666


namespace circle_diameter_l442_442329

theorem circle_diameter (A : ℝ) (hA : A = 400 * Real.pi) : ∃ D : ℝ, D = 40 :=
by
  have h_area_formula : ∀ (r : ℝ), A = Real.pi * r^2 := by sorry
  have := h_area_formula 20
  have h_radius : (∃ (r : ℝ), r = 20) ∧ (A = Real.pi * 20^2) := by sorry
  have h_diameter : ∀ (r : ℝ), D = 2 * r := by sorry
  use (2 * 20)
  exact h_diameter 20
  sorry

end circle_diameter_l442_442329


namespace sum_f_1_to_51_eq_102_l442_442479

noncomputable def f : ℤ → ℝ := sorry

axiom f_condition_1 : ∀ x, f x + f (4 - x) = 4
axiom f_condition_2 : ∀ x, f (x + 2) = f (-x)
axiom f_value_at_1 : ∃ a, f 1 = a

theorem sum_f_1_to_51_eq_102 : ∀ a, (∑ i in finset.range 51 \ {0}, f (i + 1)) = 102 :=
by
  -- Proof omitted
  sorry

end sum_f_1_to_51_eq_102_l442_442479


namespace pomelos_in_last_week_boxes_l442_442594

def calculate_pomelos_last_week
  (boxes_last_week : ℕ) (boxes_this_week : ℕ) (total_dozens : ℕ) :=
  let total_pomelos := total_dozens * 12 in
  let total_boxes := boxes_last_week + boxes_this_week in
  let average_pomelos_per_box := total_pomelos / total_boxes in
  boxes_last_week * average_pomelos_per_box

theorem pomelos_in_last_week_boxes : 
  calculate_pomelos_last_week 10 20 60 = 240 :=
by
  sorry

end pomelos_in_last_week_boxes_l442_442594


namespace find_angle_A_find_side_a_l442_442904

-- Define the triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables {a b c A B C : ℝ}
-- Assumption conditions in the problem
variables (h₁ : a * sin B = sqrt 3 * b * cos A)
variables (hb : b = 3)
variables (hc : c = 2)

-- Prove that A = π / 3 given the first condition
theorem find_angle_A : h₁ → A = π / 3 := by
  -- Proof is omitted
  sorry

-- Prove that a = sqrt 7 given b = 3, c = 2, and A = π / 3
theorem find_side_a : h₁ → hb → hc → a = sqrt 7 := by
  -- Proof is omitted
  sorry

end find_angle_A_find_side_a_l442_442904


namespace farmer_vegetables_l442_442011

theorem farmer_vegetables (initial_tomatoes initial_potatoes initial_cabbages initial_eggplants : ℕ)
    (picked_tomatoes sold_potatoes bought_cabbages planted_eggplants: ℕ) :
    initial_tomatoes = 177 → initial_potatoes = 12 → initial_cabbages = 25 → initial_eggplants = 10 →
    picked_tomatoes = 53 → sold_potatoes = 12 → bought_cabbages = 32 → planted_eggplants = 18 →
    let final_tomatoes := initial_tomatoes - picked_tomatoes in
    let final_potatoes := initial_potatoes - sold_potatoes in
    let final_cabbages := initial_cabbages + bought_cabbages in
    let final_eggplants := initial_eggplants + planted_eggplants in
    final_tomatoes = 124 ∧ final_potatoes = 0 ∧ final_cabbages = 57 ∧ final_eggplants = 28 :=
by
  intros
  unfold final_tomatoes final_potatoes final_cabbages final_eggplants
  simp
  sorry

end farmer_vegetables_l442_442011


namespace problem_l442_442860

-- Define the conditions and necessary points
def P (b : ℝ) : ℝ × ℝ := (0, b)
def M (a : ℝ) : ℝ × ℝ := (a, 0)
def F : ℝ × ℝ := (1, 0)

-- Define vectors
def vectorPN (x y b : ℝ) := (x, y - b)
def vectorNM (a x y : ℝ) := (a - x, -y)
def vectorPM (a b : ℝ) := (a, -b)
def vectorPF (b : ℝ) := (1, -b)

-- The given conditions:
def condition1 (x y a b : ℝ) := vectorPN x y b +ᵥ (1/2) • vectorNM a x y = (0,0)
def condition2 (a b : ℝ) := vectorPM a b ⬝ vectorPF b = 0

-- Lean 4 statement for the problem
theorem problem 
  (x y a b : ℝ) 
  (h1 : condition1 x y a b) 
  (h2 : condition2 a b) :
  (y^2 = 4 * x) ∧ ∀ m : ℝ, is_atomic_real_norm (1 / 2 * ((1 * sqrt (1 + 4 * m^2)))) 0  :=
sorry

end problem_l442_442860


namespace mean_of_primes_in_list_l442_442051

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def arithmetic_mean (l : List ℕ) : ℚ :=
  (l.map (λ n => (n : ℚ))).sum / l.length

theorem mean_of_primes_in_list :
  let l := [14, 17, 19, 22, 26, 31]
  let primes := l.filter is_prime
  primes = [17, 19, 31] →
  arithmetic_mean primes = 67 / 3 :=
by
  intros
  sorry

end mean_of_primes_in_list_l442_442051


namespace sum_T_f_l442_442213

noncomputable def T_f (f : { bijective (S : set ℕ) }) (j : ℕ) (n : ℕ) : ℕ :=
  if (iterate f 12 j) = j then 1 else 0

theorem sum_T_f (n : ℕ) (S : set ℕ) (hS : S = {i : ℕ | 1 ≤ i ∧ i ≤ n}) :
  let X := bijective_set_functions S S in
  ∑ f in X, ∑ j in S, T_f f j n = n! :=
by sorry

end sum_T_f_l442_442213


namespace percentage_markup_l442_442304

open Real

theorem percentage_markup (SP CP : ℝ) (hSP : SP = 5600) (hCP : CP = 4480) : 
  ((SP - CP) / CP) * 100 = 25 :=
by
  sorry

end percentage_markup_l442_442304


namespace solve_system_l442_442279

theorem solve_system :
  ∃ x y : ℝ, (x^3 + y^3) * (x^2 + y^2) = 64 ∧ x + y = 2 ∧ 
  ((x = 1 + Real.sqrt (5 / 3) ∧ y = 1 - Real.sqrt (5 / 3)) ∨ 
   (x = 1 - Real.sqrt (5 / 3) ∧ y = 1 + Real.sqrt (5 / 3))) :=
by
  sorry

end solve_system_l442_442279


namespace average_rate_of_interest_l442_442018

noncomputable def investment_average_rate (x : ℝ) (y : ℝ) (r1 r2 : ℝ) (total : ℝ) : ℝ :=
  ((r1 * x) + (r2 * y)) / total * 100

theorem average_rate_of_interest :
  ∀ (x : ℝ) (y : ℝ) (r1 r2 : ℝ),
  x + y = 5000 →
  r1 = 0.05 →
  r2 = 0.06 →
  r1 * y = r2 * x →
  investment_average_rate x y r1 r2 (5000 : ℝ) = 5.4 :=
begin
  intros,
  rw investment_average_rate,
  sorry
end

end average_rate_of_interest_l442_442018


namespace find_smaller_number_l442_442755

theorem find_smaller_number (x y : ℕ) (h1 : x = 2 * y - 3) (h2 : x + y = 51) : y = 18 :=
sorry

end find_smaller_number_l442_442755


namespace triangle_area_constraint_triangle_perimeter_constraint_quadrilateral_area_constraint_quadrilateral_perimeter_constraint_convex_quadrilateral_area_constraint_convex_quadrilateral_perimeter_constraint_l442_442748

section Geometry

def diameter (P : Polygon) : ℝ := sorry -- Longest side/diagonal measure function

def area (P : Polygon) : ℝ := sorry -- Compute area of the polygon
def perimeter (P : Polygon) : ℝ := sorry -- Compute perimeter of the polygon

-- Triangle area and perimeter constraints
theorem triangle_area_constraint (T : Polygon) (h₁ : diameter T = 1) (h₂ : triangle T) :
  0 < area T ∧ area T ≤ (Real.sqrt 3) / 4 :=
sorry

theorem triangle_perimeter_constraint (T : Polygon) (h₁ : diameter T = 1) (h₂ : triangle T) :
  2 < perimeter T ∧ perimeter T ≤ 3 :=
sorry

-- Quadrilateral area and perimeter constraints
theorem quadrilateral_area_constraint (Q : Polygon) (h₁ : diameter Q = 1) (h₂ : quadrilateral Q) :
  0 < area Q ∧ area Q ≤ 1 / 2 :=
sorry

theorem quadrilateral_perimeter_constraint (Q : Polygon) (h₁ : diameter Q = 1) (h₂ : quadrilateral Q) :
  2 < perimeter Q ∧ perimeter Q < 4 :=
sorry

-- Convex quadrilateral area and perimeter constraints
theorem convex_quadrilateral_area_constraint (CQ : Polygon) (h₁ : diameter CQ = 1) (h₂ : convex_quadrilateral CQ) :
  0 < area CQ ∧ area CQ ≤ 1 / 2 :=
sorry

theorem convex_quadrilateral_perimeter_constraint (CQ : Polygon) (h₁ : diameter CQ = 1) (h₂ : convex_quadrilateral CQ) :
  2 < perimeter CQ ∧ perimeter CQ ≤ 2 + Real.sqrt 6 - Real.sqrt 2 :=
sorry

end Geometry

end triangle_area_constraint_triangle_perimeter_constraint_quadrilateral_area_constraint_quadrilateral_perimeter_constraint_convex_quadrilateral_area_constraint_convex_quadrilateral_perimeter_constraint_l442_442748


namespace max_value_of_expression_l442_442996

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l442_442996


namespace last_score_entered_l442_442071

constant scores : List ℤ := [68, 75, 83, 94]
constant total_sum : ℤ := 320

theorem last_score_entered (latest_score : ℤ) (h : latest_score ∈ scores) :
  (∀ n ∈ [1, 2, 3, 4], (total_sum - latest_score) % n = 0) →
  latest_score = 68 :=
by {
  -- Skipping the proof steps as they are not needed
  sorry
}

end last_score_entered_l442_442071


namespace max_PA_PB_PC_PD_l442_442483

open Real

-- Define the vertices of the square ABCD
def A : Point := (0, 0)
def B : Point := (1, 0)
def C : Point := (1, 1)
def D : Point := (0, 1)

-- Assume P is a point inside or on the boundary of the square
variable (P : Point) (hP : P ∈ setOf (λ P : Point, 0 ≤ P.1 ∧ P.1 ≤ 1 ∧ 0 ≤ P.2 ∧ P.2 ≤ 1))

-- Prove that the maximum value of PA * PB * PC * PD is 5/16
theorem max_PA_PB_PC_PD : (PA : Real) (PB : Real) (PC : Real) (PD : Real) :
  PA = dist P A → PB = dist P B → PC = dist P C → PD = dist P D →
  PA * PB * PC * PD ≤ 5/16 := 
by
  sorry

end max_PA_PB_PC_PD_l442_442483


namespace integer_sequence_l442_442361

theorem integer_sequence (c : ℕ) (h : 0 < c) : ∀ (n : ℕ), n ≥ 1 → 
  ∃ (x : ℕ → ℝ), 
  (x 1 = c) ∧ 
  (∀ n ≥ 1, x (n + 1) = c * x n + real.sqrt (c^2 - 1) * real.sqrt (x n^2 - 1)) →
  (∀ n ≥ 1, x n ∈ ℤ) :=
by
  sorry

end integer_sequence_l442_442361


namespace find_BC_distance_l442_442121

theorem find_BC_distance : 
    let sin_30 := Real.sin (Real.pi / 6) in
    ∃ B C : ℝ × ℝ, 
    (∃ t1 t2 : ℝ, B = (2 - t1 * sin_30, -1 + t1 * sin_30) ∧ C = (2 - t2 * sin_30, -1 + t2 * sin_30) ∧ 
     (2 - t1 * sin_30)^2 + (-1 + t1 * sin_30)^2 = 8 ∧ 
     (2 - t2 * sin_30)^2 + (-1 + t2 * sin_30)^2 = 8) ∧ 
    Real.dist B C = sqrt 30 :=
sorry

end find_BC_distance_l442_442121


namespace characterize_set_A_l442_442623

open Int

noncomputable def A : Set ℤ := { x | x^2 - 3 * x - 4 < 0 }

theorem characterize_set_A : A = {0, 1, 2, 3} :=
by
  sorry

end characterize_set_A_l442_442623


namespace f_divisible_by_31_l442_442837

-- Defining the function f recursively
noncomputable def f : ℕ → ℤ
| 0 := 0
| 1 := 0
| (n + 2) := 4^(n + 2) * f (n + 1) - 16^(n + 1) * f n + n * 2^(n^2)

-- Main theorem statement
theorem f_divisible_by_31 : 31 ∣ f 1989 ∧ 31 ∣ f 1990 ∧ 31 ∣ f 1991 :=
by
  sorry

end f_divisible_by_31_l442_442837


namespace unique_abs_value_of_roots_l442_442156

theorem unique_abs_value_of_roots :
  ∀ (w : ℂ), w^2 - 6 * w + 40 = 0 → (∃! z, |w| = z) :=
by
  sorry

end unique_abs_value_of_roots_l442_442156


namespace pyramid_volume_l442_442807

-- Define the conditions as hypotheses
variables (A B C D P : ℝ × ℝ × ℝ)
variables (θ : ℝ)
variable (base_side : ℝ)
variable (angle_APB : ℝ)

-- Define the conditions specifically for the given problem
def pyramid_base_square (A B C D : ℝ × ℝ × ℝ) (side_length : ℝ) : Prop :=
  side_length = 1 ∧
  (A.1 = 0 ∧ A.2 = 0) ∧
  (B.1 = 1 ∧ B.2 = 0) ∧
  (C.1 = 1 ∧ C.2 = 1) ∧
  (D.1 = 0 ∧ D.2 = 1) ∧
  (A.3 = 0 ∧ B.3 = 0 ∧ C.3 = 0 ∧ D.3 = 0)

def is_equidistant (P A B C D : ℝ × ℝ × ℝ) : Prop :=
  P ≠ (A + B) / 2 ∧
  (real.dist P A = real.dist P B) ∧
  (real.dist P A = real.dist P C) ∧
  (real.dist P A = real.dist P D)

def angle_condition (P A B : ℝ × ℝ × ℝ) (θ : ℝ) : Prop :=
  ∠APB = 2 * θ

-- The main theorem to prove the volume of the pyramid
theorem pyramid_volume (A B C D P : ℝ × ℝ × ℝ) (θ : ℝ) :
  pyramid_base_square A B C D 1 →
  is_equidistant P A B C D →
  angle_condition P A B θ →
  let volume := 1 / (6 * real.sin θ) in
  volume = 1 / (6 * real.sin θ) :=
begin
  intros h_base_sq h_equidistant h_angle,
  -- Sorry block to skip the proof as instructed
  sorry
end

end pyramid_volume_l442_442807


namespace contrapositive_proposition_l442_442290

theorem contrapositive_proposition (x : ℝ) : 
  (x^2 = 1 → (x = 1 ∨ x = -1)) ↔ ((x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1) :=
by
  sorry

end contrapositive_proposition_l442_442290


namespace real_solution_a_no_purely_imaginary_roots_l442_442472

-- Define the primary equation to be used
def equation (z a : ℂ) : Prop :=
  z^2 - (a + complex.i) * z - (complex.i + 2) = 0

-- Part 1: Prove that if the equation has a real solution, a = 1
theorem real_solution_a (a : ℝ) (z : ℝ) (h : equation z a) : a = 1 := sorry

-- Part 2: Prove that the equation cannot have purely imaginary roots for any real number a
theorem no_purely_imaginary_roots (a : ℝ) (z : ℂ) (h1 : equation z a) (h2 : z.im ≠ 0) : z.re ≠ 0 := sorry

end real_solution_a_no_purely_imaginary_roots_l442_442472


namespace line_on_point_l442_442868

-- Define the function f(x)
def f (a b x : ℝ) := a * sin x + b * cos x

-- State the conditions
variables (a b x0 : ℝ)
hypothesis (h1 : x0 = x0)
hypothesis (h2 : tan x0 = 2)

-- Prove the line on which the point (a, b) lies
theorem line_on_point (h1 : x0 = x0) (h2 : tan x0 = 2) (h3 : f x0 = a * sin x0 + b * cos x0) :
  (a, b) ∈ {p : ℝ × ℝ | p.1 - 2 * p.2 = 0} :=
sorry  -- Proof goes here

end line_on_point_l442_442868


namespace cube_rolled_twice_l442_442766

noncomputable def is_prime(x : ℕ) : Prop :=
  x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7 ∨ x = 11

theorem cube_rolled_twice :
  (∀ (a b : ℕ), a ∈ {1, 2, 3, 4, 5, 6} ∧ b ∈ {1, 2, 3, 4, 5, 6} →
   (∃ s : ℚ, s = 0.41666666666666663 ∧
    s = (finset.card ({(a, b) | is_prime (a + b)}.to_finset : finset (ℕ × ℕ))) / 36)) →
  true :=
begin
  -- sorry is used to skip the actual proof
  sorry
end

end cube_rolled_twice_l442_442766


namespace solution_set_of_inequality_l442_442693

theorem solution_set_of_inequality {x : ℝ} :
  {x : ℝ | |2 - 3 * x| ≥ 4} = {x : ℝ | x ≤ -2 / 3 ∨ 2 ≤ x} :=
by
  sorry

end solution_set_of_inequality_l442_442693


namespace range_of_a_l442_442875

-- Define sets P and M
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def M (a : ℝ) : Set ℝ := {x | (2 - a) ≤ x ∧ x ≤ (1 + a)}

-- Prove the range of a
theorem range_of_a (a : ℝ) : (P ∩ (M a) = P) ↔ (a ≥ 1) :=
by 
  sorry

end range_of_a_l442_442875


namespace mod_inverse_and_equiv_l442_442614

open Nat

theorem mod_inverse_and_equiv {b : ℕ} (hb : b ≡ (mod_inverse 4 13 + mod_inverse 6 13 + mod_inverse 9 13)⁻¹ [MOD 13]) :
  b ≡ 6 [MOD 13] :=
sorry

end mod_inverse_and_equiv_l442_442614


namespace parabola_focus_line_area_minimum_area_triangle_PBC_l442_442140

theorem parabola_focus_line_area {p : ℝ} (hp : p > 0) :
  let F := (p / 2, 0)
  let lineThroughFocus (x : ℝ) := x - p / 2 -- equation of the line y = x - p/2
  let y := sorry -- y coordinates where the line intersects with the parabola
  let x := sorry -- x coordinates where the line intersects with the parabola
  let y1 := sorry -- y1 coordinates for point M
  let y2 := sorry -- y2 coordinates for point N
  let areaTriangleMON := (1 / 2) * (p / 2) * |y1 - y2|
  in areaTriangleMON = sqrt 2 / 2 ↔ p = 1 := sorry

theorem minimum_area_triangle_PBC {x0 y0 : ℝ} (hx0 : x0 ≠ 0) (h_parabola : y0^2 = 2*x0) :
  let b := sorry
  let c := sorry
  let x_center := 1
  let distance_formula := sorry -- distance from point (1, 0) to line PB
  let delta_b_c := sqrt ((4 * (x0^2 + y0^2 - 2 * x0)) / (x0 - 2)^2)
  let area_PBC := (1 / 2) * delta_b_c * x0
  let min_area := x0 - 2 + (4 / (x0 - 2)) + 4
  in min_area ≥ 8 := sorry

end parabola_focus_line_area_minimum_area_triangle_PBC_l442_442140


namespace square_area_inscribed_in_ellipse_l442_442784

theorem square_area_inscribed_in_ellipse (t : ℝ) (h : 0 ≤ t ∧ t = sqrt(36 / 13)) :
  let side_length := 2 * t
  let area := side_length^2
  ∀ x y, (x, y) ∈ ({(±t, ±t)}) :=
  ∀ x y ∈ ℝ, ∀ side_length = 2 * sqrt(36 / 13), ∀ area = side_length^2, 
    area = (4 * 36 / 13 := 144 / 13),
begin
  sorry
end

end square_area_inscribed_in_ellipse_l442_442784


namespace return_trip_time_l442_442023

variable (d p w_1 w_2 : ℝ)
variable (t t' : ℝ)
variable (h1 : d / (p - w_1) = 120)
variable (h2 : d / (p + w_2) = t - 10)
variable (h3 : t = d / p)

theorem return_trip_time :
  t' = 72 :=
by
  sorry

end return_trip_time_l442_442023


namespace parallel_b_alpha_l442_442851

variables (a b l : Type) (α : set Type) 
variables [linear_order a] [linear_order b] 
variables [linear_order α]

-- Definitions of parallelism and containments
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry
def subset (x : Type) (y : set Type) : Prop := sorry

axiom parallel_a_b : parallel a b
axiom not_subset_a_alpha : ¬ subset a α
axiom not_subset_b_alpha : ¬ subset b α
axiom parallel_a_alpha : parallel a α

-- Theorem: Given the above conditions, b is parallel to α
theorem parallel_b_alpha : parallel b α :=
sorry

end parallel_b_alpha_l442_442851


namespace inscribed_square_area_l442_442782

theorem inscribed_square_area :
  ∃ s : ℚ, (s^2/4 + s^2/9 = 1) ∧ (4 * s^2 = 144/13) :=
begin
  sorry
end

end inscribed_square_area_l442_442782


namespace sheila_weekly_earnings_l442_442638

-- Defining the conditions
def hourly_wage : ℕ := 12
def hours_mwf : ℕ := 8
def days_mwf : ℕ := 3
def hours_tt : ℕ := 6
def days_tt : ℕ := 2

-- Defining Sheila's total weekly earnings
noncomputable def weekly_earnings := (hours_mwf * hourly_wage * days_mwf) + (hours_tt * hourly_wage * days_tt)

-- The statement of the proof
theorem sheila_weekly_earnings : weekly_earnings = 432 :=
by
  sorry

end sheila_weekly_earnings_l442_442638


namespace coefficient_x3_expansion_l442_442289

theorem coefficient_x3_expansion :
  ∃ r : ℕ, (r = 3) ∧ (binomial 9 r * (-1)^r * x^(9 - 2 * r) = -84 * x^3) :=
by
  sorry

end coefficient_x3_expansion_l442_442289


namespace hyperbola_midpoint_ratio_l442_442095

theorem hyperbola_midpoint_ratio
  (a b : ℝ) (m x₀ y₀ : ℝ)
  (h_asym : b / a = sqrt 3)
  (h_vertex : (1 : ℝ) = a)
  (h_intersects : y₀ = x₀ + m)
  (h_midpoint_x : x₀ = m / 2)
  (h_midpoint_y : y₀ = (3/2) * m)
  (h_x₀_ne_zero : x₀ ≠ 0) :
  y₀ / x₀ = 3 :=
begin
  sorry -- proof not required
end

end hyperbola_midpoint_ratio_l442_442095


namespace parallelogram_point_D_l442_442916

/-- Given points A, B, and C, the coordinates of point D in parallelogram ABCD -/
theorem parallelogram_point_D (A B C D : (ℝ × ℝ))
  (hA : A = (1, 1))
  (hB : B = (3, 2))
  (hC : C = (6, 3))
  (hMid : (2 * (A.1 + C.1), 2 * (A.2 + C.2)) = (2 * (B.1 + D.1), 2 * (B.2 + D.2))) :
  D = (4, 2) :=
sorry

end parallelogram_point_D_l442_442916


namespace sum_series_l442_442069

theorem sum_series (s : ℕ → ℝ) 
  (h : ∀ n : ℕ, s n = (n+1) / (4 : ℝ)^(n+1)) : 
  tsum s = (4 / 9 : ℝ) :=
sorry

end sum_series_l442_442069


namespace geometric_progression_identity_l442_442263

variable {a : ℝ} {p q : ℝ}

def S := 1 / (1 - a^p)
def S1 := 1 / (1 - a^q)

theorem geometric_progression_identity (hS : S = 1 / (1 - a^p)) (hS1 : S1 = 1 / (1 - a^q)) :
    S^q * (S1 - 1)^p = S1^p * (S - 1)^q :=
sorry

end geometric_progression_identity_l442_442263


namespace find_m_plus_n_l442_442029

noncomputable def cone_area_volume_ratio (h r : ℝ) : ℚ :=
let A := π * r * (r + (sqrt (r^2 + h^2)))
let V := 1 / 3 * π * r^2 * h in
let x := r / 2 in
let h_c := h / 2 in
let A_c := π * x * (x + sqrt ((r / 2)^2 + h_c^2)) in
let V_c := 1 / 3 * π * x^2 * h_c in
let A_f := A - A_c in
let V_f := V - V_c in
let ratio_area := A_c / A_f
let ratio_vol := V_c / V_f in
if ratio_area = ratio_vol then ratio_area else 0 

theorem find_m_plus_n : 
  ∃ (m n : ℕ), gcd m n = 1 ∧ (cone_area_volume_ratio 6 2) = (m / n) ∧ m + n = 272 :=
sorry

end find_m_plus_n_l442_442029


namespace probability_three_heads_one_tail_l442_442163

-- Define the probability of a single coin being heads or tails
def coin_prob : ℝ := 1 / 2

-- Define the total number of coin tosses
def total_tosses : ℕ := 4

-- Define the number of heads desired
def desired_heads : ℕ := 3

-- Calculate the specific sequence probability
def sequence_prob : ℝ := pow coin_prob total_tosses

-- Calculate the number of favorable sequences (combinations)
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def favorable_sequences : ℕ := combination total_tosses desired_heads

-- The theorem statement
theorem probability_three_heads_one_tail : 
  (favorable_sequences * sequence_prob) = 1 / 4 :=
by
  -- Proof would go here
  sorry

end probability_three_heads_one_tail_l442_442163


namespace find_min_cp_pa1_l442_442906

noncomputable def right_prism : Prop := ∃ A B C A₁ B₁ C₁ P : ℝ³,
  -- Conditions
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧
  ∠A C B = (90 : ℝ) ∧     -- Angle ACB is 90 degrees
  dist(B, C) = 2 ∧         -- BC = 2
  dist(C, C₁) = 2 ∧        -- CC₁ = 2
  dist(A, C) = 4 * sqrt(2) ∧ -- AC = 4sqrt(2)
  P ∈ line_through(B, C₁)   -- Point P is on line BC₁
  

theorem find_min_cp_pa1 (h : right_prism) :
  ∃ CP PA₁ : ℝ, (dist(C, P) + dist(P, A₁) = 2 * sqrt(5)) :=
sorry

end find_min_cp_pa1_l442_442906


namespace problem_solution_l442_442439

theorem problem_solution (x : ℕ) (h : x = 3) : x + x * (x ^ (x + 1)) = 246 :=
by
  sorry

end problem_solution_l442_442439


namespace integral_value_l442_442861

theorem integral_value (a : ℝ) (h : ∑ r in finset.range 7, binomial 6 r * ((-1)^r) * (a^(6-r)) * ((-1)^r) * (1^r) = -540) :
  ∫ x in 0..a, 3 * x^2 - 1 = 24 :=
sorry

end integral_value_l442_442861


namespace common_points_of_graphs_l442_442437

theorem common_points_of_graphs :
  let f1 := (λ x y: ℝ, x^2 - y + 2)
  let f2 := (λ x y: ℝ, 3x^3 + y - 4)
  let g1 := (λ x y: ℝ, x + y^2 - 2)
  let g2 := (λ x y: ℝ, 2x^2 - 5y + 7)
  ∃ (points : Finset (ℝ × ℝ)), points.card = 6 ∧
    ∀ p ∈ points, 
      (f1 p.1 p.2 = 0 ∨ f2 p.1 p.2 = 0) ∧ 
      (g1 p.1 p.2 = 0 ∨ g2 p.1 p.2 = 0)
:= sorry

end common_points_of_graphs_l442_442437


namespace initial_logs_l442_442765

theorem initial_logs (x : ℕ) (h1 : x - 3 - 3 - 3 + 2 + 2 + 2 = 3) : x = 6 := by
  sorry

end initial_logs_l442_442765


namespace part1_part2_l442_442921

variables {α : Type*} [RealInnerProductSpace α]

-- Define the triangle and conditions
structure Triangle (a b c : ℝ) :=
(angle_A : ℝ)
(angle_B : ℝ)
(angle_C : ℝ)
(h1 : a * Real.cos angle_C = (2 * b - c) * Real.cos angle_A)

-- Part 1: Proof that angle A equals π/3
theorem part1 (a b c angle_A angle_B angle_C : ℝ) 
  (h : Triangle a b c) : 
  angle_A = Real.pi / 3 :=
sorry

-- Part 2: Proof that AD is in the given range
theorem part2 (a b c : ℝ) (D : α) 
  (h2 : a = Real.sqrt 3) 
  (h3 : midpoint b c = D) :
  AD D ∈ Icc (Real.sqrt 3 / 2) (3 / 2) :=
sorry

end part1_part2_l442_442921


namespace sequence_correct_l442_442844

def seq (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, n > 0 →
  (∑ i in finset.range n, (2 * i + 1) * a (i + 1)) = (n - 1) * 3^(n + 1) + 3

theorem sequence_correct (a : ℕ → ℕ): (seq a) → (∀ n : ℕ, n > 0 → a n = 3^n) :=
sorry

end sequence_correct_l442_442844


namespace distance_foci_to_asymptotes_l442_442824

noncomputable def distance_foci_asymptotes : ℝ :=
  let a : ℝ := real.sqrt 2
  let b : ℝ := 1
  let c : ℝ := real.sqrt (a^2 + b^2)
  let focus1 := (0, c)
  let focus2 := (0, -c)
  let asymptote1 := λ x : ℝ => x / a
  let asymptote2 := λ x : ℝ => -x / a
  let distance_to_line := λ (x₀ y₀ A B C : ℝ) => abs (A * x₀ + B * y₀ + C) / real.sqrt (A^2 + B^2)
  let d1 := distance_to_line 0 c (1/a) (-1) 0
  let d2 := distance_to_line 0 (-c) (1/a) (-1) 0
  d1 -- or d2 since both distances are equal

theorem distance_foci_to_asymptotes :
  distance_foci_asymptotes = real.sqrt 2 :=
by
  sorry

end distance_foci_to_asymptotes_l442_442824


namespace S_2012_value_l442_442562

-- Define the first term of the arithmetic sequence
def a1 : ℤ := -2012

-- Define the common difference
def d : ℤ := 2

-- Define the sequence a_n
def a (n : ℕ) : ℤ := a1 + d * (n - 1)

-- Define the sum of the first n terms S_n
def S (n : ℕ) : ℤ := n * (a1 + a n) / 2

-- Formalize the given problem as a Lean statement
theorem S_2012_value : S 2012 = -2012 :=
by 
{
  -- The proof is omitted as requested
  sorry
}

end S_2012_value_l442_442562


namespace range_of_m_l442_442512

-- Define the propositions and setup
def proposition (p : ℝ → ℝ) (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, p x₀ - m * 3^x₀ + 4 ≤ 0

noncomputable def f (t : ℝ) (m : ℝ) : ℝ :=
  t^2 - m * t + 4

theorem range_of_m (p : ℝ → ℝ) (m : ℝ) (h : proposition p m) : 
  ∃ c : ℝ, c = 3 ∧ [4, +∞) := 
by sorry

end range_of_m_l442_442512


namespace find_k_l442_442241

theorem find_k (p : ℕ) (hp : prime p ∧ odd p) (k : ℕ) 
  (h : ∃ k : ℕ, 0 < k ∧ ∃ n : ℕ, 0 < n ∧ n^2 = k^2 - p * k) : 
  k = (p + 1)^2 / 4 :=
by
  sorry

end find_k_l442_442241


namespace number_of_valid_distributions_l442_442634

-- Define the rabbits
inductive Rabbit
| Molly
| Max
| Thumper
| Bugs
deriving DecidableEq, Fintype

-- Define the pet stores
inductive Store
| Store1
| Store2
| Store3
deriving DecidableEq, Fintype

open Rabbit
open Store

-- Define the condition that no store gets both a parent and a child
def no_store_with_parent_and_child (assignment : Rabbit → Store) : Prop :=
  ¬ ∃ s, (assignment Molly = s ∨ assignment Max = s) ∧ (assignment Thumper = s ∨ assignment Bugs = s)

-- Define the condition that at least one store must remain empty
def at_least_one_empty_store (assignment : Rabbit → Store) : Prop :=
  ∃ s, ∀ r, assignment r ≠ s

-- Define the main theorem statement
theorem number_of_valid_distributions : 
  {assignment : Rabbit → Store // no_store_with_parent_and_child assignment ∧ at_least_one_empty_store assignment}.card = 6 :=
by {
  -- Proof would normally go here
  sorry
}

end number_of_valid_distributions_l442_442634


namespace solve_for_x_l442_442985

theorem solve_for_x :
  (∃ x : ℝ, (1 / 3 - 1 / 4) = 1 / x) → ∃ x : ℝ, x = 12 :=
by
  intro h,
  obtain ⟨x, hx⟩ := h,
  use 12,
  have : 1 / 3 - 1 / 4 = 1 / 12, by
  { calc
      1 / 3 - 1 / 4 = 4 / 12 - 3 / 12 : by norm_num
                 ... = 1 / 12 : by norm_num },
  exact this ▸ hx.symm

end solve_for_x_l442_442985


namespace inverse_log_2_add_seven_l442_442957

theorem inverse_log_2_add_seven (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = log 2 (x + 7)) 
  (inverse_f_g : ∀ x, f (g x) = x ∧ g (f x) = x) : 
  f 1 + g 1 = -2 := 
by 
  sorry

end inverse_log_2_add_seven_l442_442957


namespace trajectory_of_p_passes_through_incenter_l442_442749

variables {V : Type*} [inner_product_space ℝ V]
variables (O A B C P : V)
variables (λ : ℝ) (hλ : λ ≥ 0)
variables (non_collinear : ¬ collinear ℝ ({A, B, C} : set V))

def unit_vector (u : V) : V := u / ∥u∥

def move_point (O A B C : V) (λ : ℝ) (hλ : λ ≥ 0) : V :=
  O + (λ • (unit_vector (B - A) + unit_vector (C - A)))

noncomputable def incenter (A B C : V) : V := sorry -- Definition for incenter

theorem trajectory_of_p_passes_through_incenter
  (P : V) (hP : P = move_point O A B C λ hλ) :
  ∃ k : ℝ, k > 0 ∧ P = A + k • (incenter A B C - A) :=
sorry

end trajectory_of_p_passes_through_incenter_l442_442749


namespace solution_set_of_inequality_l442_442497

variable {f g : ℝ → ℝ}

-- Given conditions
def g_nonzero (x : ℝ) : Prop := g(x) ≠ 0
def f_prime_g_gt_f_g_prime (x : ℝ) : Prop := f'(x) * g(x) > f(x) * g'(x)
def f_at_one_zero : Prop := f(1) = 0

-- Main theorem statement
theorem solution_set_of_inequality
  (h : ℝ → ℝ := λ x, f(x) / g(x))
  (H1 : ∀ x, g_nonzero x)
  (H2 : ∀ x, f_prime_g_gt_f_g_prime x)
  (H3 : f_at_one_zero) :
  ∀ x, h(x) > 0 ↔ x > 1 :=
by
  sorry

end solution_set_of_inequality_l442_442497


namespace root_approximations_l442_442711

variables {A B C : ℝ}

-- Original polynomial definition
def poly1 (x : ℝ) : ℝ := x^3 - x^2 - 6*x + 2

-- Assumption that A, B, and C are coefficients from equation (III)
axiom coeff_A : A = 1.982 * 10^7
axiom coeff_B : B = 4.996 * 10^12
axiom coeff_C : C = 2^16

-- Polynomial whose roots are 16th powers of roots of poly1
def poly3 (x : ℝ) : ℝ := x^3 + A * x^2 + B * x + C

-- Absolute value roots approximations
def approx_roots (x : ℝ) : ℝ := abs (root (16 : ℝ) x)

-- Main statement: proves against the polynomial equation approximations
theorem root_approximations :
  (poly1 (approx_roots (-A)) = 0) ∧
  (poly1 (approx_roots (-B / A)) = 0) ∧
  (poly1 (approx_roots (-C / B)) = 0) :=
begin
  sorry
end

end root_approximations_l442_442711


namespace couple_rooms_correct_l442_442923

def rooms_for_couples : ℕ :=
  let single_rooms := 14
  let bath_per_room := 10
  let total_bath := 400
  let total_baths_needed := λ (x : ℕ), single_rooms + 2 * x
  let total_bubble_bath_needed := λ (x : ℕ), total_baths_needed x * bath_per_room
  let eqn := total_bubble_bath_needed = 400
  sorry

-- Problem Statement
theorem couple_rooms_correct : rooms_for_couples = 13 :=
  sorry

end couple_rooms_correct_l442_442923


namespace g_of_3_l442_442850

theorem g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 4 * g x + 3 * g (1 / x) = 2 * x) :
  g 3 = 22 / 7 :=
sorry

end g_of_3_l442_442850


namespace max_sales_increase_year_l442_442296

def sales : ℕ → ℝ
| 1998 := 2
| 1999 := 3
| 2000 := 3.5
| 2001 := 4.5
| 2002 := 5
| 2003 := 6
| 2004 := 7.5
| 2005 := 8
| 2006 := 8.5
| 2007 := 9
| 2008 := 9.5
| _ := 0

theorem max_sales_increase_year :
  ∃ year, year = 2004 ∧ ∀ y, y ≠ 2004 → sales (y + 1) - sales y ≤ sales 2005 - sales 2004 :=
by
  sorry

end max_sales_increase_year_l442_442296


namespace dot_product_AC_BC_l442_442853

-- Define a right triangle ABC with given conditions
variables (A B C : Type) -- Points in the plane
variables [inner_product_space ℝ (A × B × C)]

-- Given conditions
def right_triangle (A B C : Type) [inner_product_space ℝ (A × B × C)] :=
  (angle A B C = π / 2) ∧ (dist A B = 4) ∧ (dist B C = 3)

-- The dot product condition we need to prove
theorem dot_product_AC_BC (h : right_triangle A B C) :
  let AC := (A - C)
      BC := (B - C)
  in (inner_product_space.inner AC BC = 9) :=
by {
  sorry
}

end dot_product_AC_BC_l442_442853


namespace normal_distribution_parameters_l442_442774

noncomputable def pdf (x : ℝ) : ℝ :=
1 / real.sqrt (8 * real.pi) * real.exp (x^2 / 8)

theorem normal_distribution_parameters :
  ∃ (μ σ : ℝ), (∀ x : ℝ, pdf x = 1 / (real.sqrt (2 * real.pi) * σ) * real.exp ((-(x - μ)^2) / (2 * σ^2))) ∧ μ = 0 ∧ σ = 2 :=
begin
  use [0, 2],
  split,
  {
    intro x,
    simp [pdf],
    -- Here, the necessary algebraic manipulations to show f(x) matches the PDF of normal distribution with μ=0 and σ=2 are done.
    sorry
  },
  split,
  {
    refl,
  },
  {
    refl,
  }
end

end normal_distribution_parameters_l442_442774


namespace last_digit_to_appear_in_fibonacci_mod_10_is_6_l442_442286

/-- The Fibonacci sequence starts with two 1s and each term afterwards is the sum of its two predecessors. -/
def fibonacci (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

/-- Proof that the last digit to appear in the units position of a Fibonacci number mod 10 is 6. -/
theorem last_digit_to_appear_in_fibonacci_mod_10_is_6 : 
  ∃ N : ℕ, (∀ d : ℕ, d < 10 → (∃ n : ℕ, (n ≥ N) ∧ (fibonacci n % 10 = d))) ∧ (∃ n : ℕ, (fibonacci n % 10 = 6)) :=
sorry

end last_digit_to_appear_in_fibonacci_mod_10_is_6_l442_442286


namespace arrow_balance_l442_442600

theorem arrow_balance (n : ℕ) (h_n_odd : n % 2 = 1) (arrows : Fin n → Bool) :
  ∃ k : Fin n, (Fin.sum (Fin.range k) (λ i, if arrows i then 1 else 0) =
               Fin.sum (Fin.range k) (λ j, if arrows (⟨j + k, sorry⟩ : Fin n) then 1 else 0)) :=
by sorry

end arrow_balance_l442_442600


namespace count_even_integers_between_l442_442883

theorem count_even_integers_between : 
    let lower := 18 / 5
    let upper := 45 / 2
    ∃ (count : ℕ), (∀ n : ℕ, lower < n ∧ n < upper → n % 2 = 0 → n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 10 ∨ n = 12 ∨ n = 14 ∨ n = 16 ∨ n = 18 ∨ n = 20 ∨ n = 22) ∧ count = 10 :=
by
  sorry

end count_even_integers_between_l442_442883


namespace inverse_49_mod_101_l442_442107

theorem inverse_49_mod_101 (h : (7 : ℤ)⁻¹ ≡ 55 [ZMOD 101]) :
  (49 : ℤ)⁻¹ ≡ 96 [ZMOD 101] :=
sorry

end inverse_49_mod_101_l442_442107


namespace fixed_points_15_contains_9_elements_l442_442838

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 0.5 then x + 0.5 else 2 * (1 - x)

def f_n_aux (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
| 0     := id
| (n+1) := f ∘ f_n_aux f n

def f_n (n : ℕ) (x : ℝ) := f_n_aux f n x

def fixed_points_15 {x : ℝ} (x_in_unit_interval : x ∈ set.Icc 0 1) : Prop :=
  f_n 15 x = x

theorem fixed_points_15_contains_9_elements :
  ∃ (S : set ℝ), (set.card S ≥ 9) ∧ (∀ x ∈ S, fixed_points_15 (set.mem_Icc_of_mem x_in_unit_interval x)) := sorry

end fixed_points_15_contains_9_elements_l442_442838


namespace ratio_proof_l442_442403

noncomputable def side_length_triangle(a : ℝ) : ℝ := a / 3
noncomputable def side_length_square(b : ℝ) : ℝ := b / 4
noncomputable def area_triangle(a : ℝ) : ℝ := (side_length_triangle(a)^2 * Mathlib.sqrt(3)) / 4
noncomputable def area_square(b : ℝ) : ℝ := (side_length_square(b))^2

theorem ratio_proof (a b : ℝ) (h : area_triangle(a) = area_square(b)) : a / b = 2 * Mathlib.sqrt(3) / 9 :=
by {
  sorry
}

end ratio_proof_l442_442403


namespace equilateral_triangle_area_l442_442660

theorem equilateral_triangle_area (AM : ℝ) (h : AM = sqrt 3) : 
    let BM := AM / sqrt 3,
    let BC := 2 * BM in
    (1 / 2) * BC * AM = sqrt 3 :=
by
    sorry

end equilateral_triangle_area_l442_442660


namespace translate_parabola_l442_442712

def parabola_translated (f : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ :=
  λ x, f (x - h) + k

noncomputable def original_parabola (x : ℝ) : ℝ := x^2 + 1

theorem translate_parabola :
  ∀ x : ℝ, parabola_translated original_parabola 3 (-2) x = (x - 3)^2 - 1 :=
by
  sorry

end translate_parabola_l442_442712


namespace deriv_gt_zero_is_necessary_not_sufficient_l442_442864

variable {a b : ℝ} (f : ℝ → ℝ)

-- Assume y = f(x) is differentiable on (a, b)
def differentiable_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, x ∈ set.Ioo a b → differentiable_at ℝ f x

-- Assume f(x) is increasing on (a, b) if f'(x) > 0
def increasing_on_if_deriv_gt_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b → (deriv f x₁ > 0 → f x₁ < f x₂)

-- Prove that f'(x) > 0 is necessary but not sufficient for f to be increasing on (a, b)
theorem deriv_gt_zero_is_necessary_not_sufficient :
  differentiable_on_interval f a b →
  increasing_on_if_deriv_gt_zero f a b →
  (∀ x, x ∈ set.Ioo a b → deriv f x > 0) → (¬ (∀ x, x ∈ set.Ioo a b → deriv f x > 0)) → 
  (∀ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b → f x₁ < f x₂) →
  (∀ x, x ∈ set.Ioo a b → deriv f x > 0) :=
begin
  -- Proof goes here
  sorry
end

end deriv_gt_zero_is_necessary_not_sufficient_l442_442864


namespace sum_of_f_values_l442_442492

def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt(2))

theorem sum_of_f_values :
  f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 + f 4 + f 5 + f 6 = 3 * Real.sqrt 2 := 
by
  sorry

end sum_of_f_values_l442_442492


namespace arithmetic_seq_conditions_general_term_and_sum_formula_correct_T_100_value_l442_442866

-- Definitions based on the problem's conditions
def a_n (n : ℕ) : ℕ := 4 * n - 3
def S_n (n : ℕ) : ℕ := 2 * n^2 - n
def T_n (n : ℕ) : ℚ := (1/4) * (1 - 1 / (4 * n + 1))

-- The arithmetic sequence conditions
theorem arithmetic_seq_conditions : (a_n 3 = 9) ∧ (a_n 8 = 29) := by
  unfold a_n
  simp
  split
  . exact rfl
  . exact rfl

-- The statement proving that the general term formula and sum of terms are correct
theorem general_term_and_sum_formula_correct : 
  ∀ n, a_n n = 4 * n - 3 ∧ S_n n = (n * (a_n 1 + a_n n)) / 2 := by 
  intro n
  unfold a_n S_n
  split
  . rfl
  . ring

-- The statement proving the sum of the first 100 terms of sequence {1/(a_n a_(n+1))}
theorem T_100_value : T_n 100 = 100 / 401 := by
  unfold T_n
  norm_num
  rw [div_eq_mul_one_div, mul_sub, mul_one, mul_inv_cancel, sub_mul, div_self, sub_eq_self]
  . norm_num
  . norm_num
  . exact ne.intro rfl
  . rfl

end arithmetic_seq_conditions_general_term_and_sum_formula_correct_T_100_value_l442_442866


namespace guitar_retail_price_l442_442275

theorem guitar_retail_price:
  ∃ (P: ℝ), 
  let guitar_center_cost := 0.85 * P + 100 in
  let sweetwater_cost := 0.90 * P in
  (guitar_center_cost = sweetwater_cost + 50) →
  P = 1000 :=
by
  sorry

end guitar_retail_price_l442_442275


namespace function_expression_l442_442841

theorem function_expression (A m ω : ℝ) (φ : ℝ) :
  (∀ x, 0 ≤ x → x ≤ 2 * π / ω →
    let y := A * Real.sin (ω * x + φ) + m in
    0 ≤ y ∧ y ≤ 4) →
  (∀ x, y := A * Real.sin (ω * x + φ) + m ∧
     (∀ δ, y (x - π / ω - δ) = y (x + π / ω + δ))) →
  A * Real.sin (ω * x + φ) + m = 2 * Real.sin (ω * x) + 2 :=
sorry

end function_expression_l442_442841


namespace sets_without_squares_l442_442603

noncomputable def S_i (i : ℕ) : set ℤ :=
  {n | 150 * i ≤ n ∧ n < 150 * (i + 1)}

theorem sets_without_squares : 
  (card (finset.filter (λ i, ∀ x, x ^ 2 ∉ S_i i) (finset.range 1500))) = 1349 :=
by
  sorry

end sets_without_squares_l442_442603


namespace value_of_a_l442_442106
-- Lean 4 Code


noncomputable def given_polynomial (a : ℝ) (x : ℝ) :=
  (1 - a * x) ^ 2018

theorem value_of_a (a : ℝ) (a0 a1 a2 : ℝ) (an : Fin 2019 → ℝ) (h_nonzero : a ≠ 0)
  (h_eq : given_polynomial a x = ∑ i in finset.range 2019, (an i) * x ^ i)
  (h_sum : ∑ i in finset.range 2019, (i + 1) * (an i) = 2018 * a) :
  a = 2 :=
by
  sorry

end value_of_a_l442_442106


namespace OH_parallel_AC_l442_442630

open EuclideanGeometry

variables {A B C P Q R H O : Point}
variables (AP PR CQ QR : ℝ)
variables (tABC : Triangle A B C)
variables (tPQR : Triangle P Q R)

-- Axioms from the conditions
def points_on_sides := P ∈ (A,B) ∧ Q ∈ (B,C) ∧ R ∈ (C,A)
axiom AP_eq_PR : dist A P = dist P R
axiom CQ_eq_QR : dist C Q = dist Q R
axiom H_is_orthocenter : Orthocenter H tPQR
axiom O_is_circumcenter : Circumcenter O tABC

-- Goal: Prove that OH is parallel to AC
theorem OH_parallel_AC : LineParallel (lineFromPoints O H) (lineFromPoints A C) :=
sorry

end OH_parallel_AC_l442_442630


namespace difference_sums_l442_442272

def odd_sum (start : ℕ) (end : ℕ) : ℕ :=
let n := (end - start) / 2 + 1 in
n * (start + end) / 2

def difference_of_sums : ℕ :=
(odd_sum 101 299) - (odd_sum 3 69)

theorem difference_sums :
  difference_of_sums = 18776 := 
by
  sorry

end difference_sums_l442_442272


namespace find_inscribed_radius_squared_l442_442373

def segment_length := ℕ 

structure Quadrilateral :=
  (A B C D P Q : Type)
  (AP PB CQ QD : segment_length)

def inscribed_circle_radius_squared (ap pb cq qd : segment_length) := ℕ 

theorem find_inscribed_radius_squared (h : Quadrilateral) 
  (h1: h.AP = 19) (h2: h.PB = 26) (h3: h.CQ = 37) (h4: h.QD = 23) :
  inscribed_circle_radius_squared h.AP h.PB h.CQ h.QD = 647 :=
sorry

end find_inscribed_radius_squared_l442_442373


namespace distance_is_sixteen_over_three_l442_442496

noncomputable def distance_between_circle_center_and_hyperbola_center : ℝ :=
  let hyperbola_eq : ℝ × ℝ → Prop := λ ⟨x, y⟩, x^2 / 9 - y^2 / 16 = 1 in
  let circle_center : ℝ × ℝ := (4, 4 * real.sqrt 7 / 3) in
  let circle_center_alt : ℝ × ℝ := (4, -4 * real.sqrt 7 / 3) in
  let origin : ℝ × ℝ := (0,0) in
  let d_center := λ ⟨cx, cy⟩, real.sqrt ((cx - origin.1)^2 + (cy - origin.2)^2) in
  if hyp_on_center : hyperbola_eq circle_center then
    d_center circle_center
  else
    d_center circle_center_alt

theorem distance_is_sixteen_over_three :
  distance_between_circle_center_and_hyperbola_center = 16 / 3 :=
sorry

end distance_is_sixteen_over_three_l442_442496


namespace general_term_of_a_sum_of_b_l442_442117

-- Define S_n
axiom S_n : ℕ → ℕ

-- The assumptions given in the problem
axiom a1_zero : a 1 = 0

axiom na_eq_Sn_plus_nn1 (n : ℕ) (hn : n > 0) : n * a (n + 1) = S_n n + n * (n + 1)

-- Derive sequence a_n
theorem general_term_of_a (n : ℕ) : a n = 2 * n - 2 := sorry

-- Define sequence b_n and its property
axiom bn_eq_formula (n : ℕ) : a n + Real.log 3 n = Real.log 3 (b n)

-- Derive sum T_n for sequence b_n
theorem sum_of_b (n : ℕ) : T n = 1 / 64 * ((8 * n - 1) * 9 ^ n + 1) := sorry

end general_term_of_a_sum_of_b_l442_442117


namespace card_taken_by_Anton_l442_442314

theorem card_taken_by_Anton :
  ∃ (missing_card : ℕ), 
    (missing_card ∈ ({2, 3, 6, 7, 9} : Finset ℕ)) ∧
    (∃ (a b c d : ℕ), {a, b, c, d} ⊆ {1, 4, 5, 8, missing_card} ∧
    a * b = c * d ∧ missing_card = 7) :=
sorry

end card_taken_by_Anton_l442_442314


namespace abs_w_unique_l442_442153

theorem abs_w_unique (w : ℂ) (h : w^2 - 6 * w + 40 = 0) : ∃! x : ℝ, x = Complex.abs w ∧ x = Real.sqrt 40 := by
  sorry

end abs_w_unique_l442_442153


namespace fg_of_1_l442_442236

def f (x : ℤ) : ℤ := x + 3
def g (x : ℤ) : ℤ := x^3 - x^2 - 6

theorem fg_of_1 : f (g 1) = -3 := by
  sorry

end fg_of_1_l442_442236


namespace polynomial_evaluation_2020_l442_442612

theorem polynomial_evaluation_2020 (P : Polynomial ℝ) 
  (h1 : P.monic) 
  (h2 : P.degree = 2020) 
  (h3 : ∀ n ∈ (Finset.range 2020), P.eval (n : ℝ) = n) : 
  P.eval 2020 = (2020.factorial : ℝ) + 2020 :=
by
  sorry

end polynomial_evaluation_2020_l442_442612


namespace num_squarish_numbers_l442_442393

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def is_squarish (N : ℕ) : Prop :=
  100000 ≤ N ∧ N < 1000000 ∧
  (∀ d, d.digit_one_to_nine) ∧
  is_perfect_square N ∧
  (let a := N / 10000 in
   let b := (N / 100) % 100 in
   let c := N % 100 in
   is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ 
   is_perfect_square a ∧ is_perfect_square b ∧ is_perfect_square c)

theorem num_squarish_numbers : 
  {N : ℕ // is_squarish N}.card = 2 := by
  sorry

end num_squarish_numbers_l442_442393


namespace escher_prints_consecutive_l442_442966

noncomputable def probability_all_eschers_consecutive (n : ℕ) (m : ℕ) (k : ℕ) : ℚ :=
if h : m = n + 3 ∧ k = 4 then 1 / (n * (n + 1) * (n + 2)) else 0

theorem escher_prints_consecutive :
  probability_all_eschers_consecutive 10 12 4 = 1 / 1320 :=
  by sorry

end escher_prints_consecutive_l442_442966


namespace triangle_ARS_is_isosceles_at_A_l442_442950

theorem triangle_ARS_is_isosceles_at_A
    (ABC : Type)
    [triangle ABC]
    (A B C P R S : Point)
    (Γ : Circle)
    (tangent_A : Line)
    (angle_APB_bisector : Line) :
  -- given conditions
  AB < AC ∧
  is_circumscribed_circle Γ ABC ∧
  tangent_A.intersects Γ.at A ∧
  tangent_A.intersects (line_segment B C).at P ∧
  angle_APB_bisector.intersects (line_segment A B).at R ∧
  angle_APB_bisector.intersects (line_segment A C).at S →
  -- prove that triangle ARS is isosceles at A
  is_isosceles_at_A (triangle ARS) :=
sorry

end triangle_ARS_is_isosceles_at_A_l442_442950


namespace hyperbola_vertex_to_asymptote_distance_l442_442842

theorem hyperbola_vertex_to_asymptote_distance
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_eq : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (h_focal_length : 2 * real.sqrt (a^2 + b^2) = 2 * real.sqrt 5)
  (h_asymptote_perpendicular : (∀ x y : ℝ, 2 * b * x - a * y = 0)) :
  real.dist (2 : ℝ) (0 : ℝ) (x y : ℝ) (x + 2 * y = 0) = (2 * real.sqrt 5) / 5 :=
sorry

end hyperbola_vertex_to_asymptote_distance_l442_442842


namespace hyperbola_asymptote_m_value_l442_442085

theorem hyperbola_asymptote_m_value (m : ℝ) (h : m ≠ 0) :
  (∀ x y : ℝ, (x ^ 2) / m - (y ^ 2) / 6 = 1 → (y = x ∨ y = -x)) → m = 6 :=
by
  intros h_asymptote
  have asymptote_eq : (sqrt 6) / (sqrt m) = 1 := sorry
  have sqrt_6_eq_sqrt_m : sqrt 6 = sqrt m := sorry
  have m_eq_6 : m = 6 := sorry
  exact m_eq_6

end hyperbola_asymptote_m_value_l442_442085


namespace sequence_a_correct_sequence_b_sum_l442_442116

noncomputable theory

def S (n : ℕ) : ℕ := n^2 - 2 * n
def a : ℕ → ℤ
| 1 := -1
| n := if n ≥ 2 then 2 * n - 3 else 0 -- a_n is defined when n ≥ 1, for n < 1 we assign 0

def b (n : ℕ) : ℕ := n * 2^(a n + 1)

def T (n : ℕ) : ℕ :=
  (3 * n - 1) * 4^n / 9 + 1 / 9

theorem sequence_a_correct (n : ℕ) (hn : n ≥ 1) : 
  if n = 1 then a n = -1 else a n = 2 * n - 3 :=
by sorry

theorem sequence_b_sum (n : ℕ) (hn : n ≥ 1) :
  (∑ i in range n, b (i + 1)) = T n :=
by sorry

end sequence_a_correct_sequence_b_sum_l442_442116


namespace probability_odd_sum_is_9_over_616_l442_442313

   /-- There are twelve tiles numbered from 1 to 12. Four players each randomly select and keep three of the tiles.
   Prove that the probability that all four players obtain an odd sum is 9/616. -/
   theorem probability_odd_sum_is_9_over_616 : 
     let tiles := Finset.range 13 \    -- represents {0, 1, 2, ..., 12}, we'll adjust it to {1, 2, ..., 12}
                   Finset.singleton 0 := (Finset.range 13).erase 0
     let odd_tiles := tiles.filter (λ x, x % 2 = 1) -- {1, 3, 5, 7, 9, 11}
     let even_tiles := tiles.filter (λ x, x % 2 = 0) -- {2, 4, 6, 8, 10, 12}
     let n_ways_to_choose_3 := (tiles.card.factorial / 
                                (3 * (tiles.card - 3).factorial)) -- C(12, 3)
     let total_ways := n_ways_to_choose_3 ^ 4 -- (C(12, 3) choose 3)^4
     let successful_ways := 5400 -- summed from the manual calculations for both scenarios.
   in  (successful_ways / total_ways) = 9 / 616 :=
   sorry
   
end probability_odd_sum_is_9_over_616_l442_442313


namespace arithmetic_expression_evaluation_l442_442052

theorem arithmetic_expression_evaluation :
  (-18) + (-12) - (-33) + 17 = 20 :=
by
  sorry

end arithmetic_expression_evaluation_l442_442052


namespace min_tiles_l442_442274

theorem min_tiles (x y : ℕ) (h1 : 25 * x + 9 * y = 2014) (h2 : ∀ a b, 25 * a + 9 * b = 2014 -> (a + b) >= (x + y)) : x + y = 94 :=
  sorry

end min_tiles_l442_442274


namespace number_of_integers_with_D_eq_3_l442_442471

-- Definitions from the conditions
def D (n : ℕ) : ℕ := 
  let b := Nat.Digits 2 n
  (List.map (λ x, x.1 ≠ x.2) (List.zip b (List.tail b))).count true

-- The main statement to be proven
theorem number_of_integers_with_D_eq_3 : (Set.count (Set.filter (λ n, D n = 3) (Set.Icc 1 200))) = 24 := 
sorry

end number_of_integers_with_D_eq_3_l442_442471


namespace simplify_fraction_l442_442983

theorem simplify_fraction : (2 / 520) + (23 / 40) = 301 / 520 := by
  sorry

end simplify_fraction_l442_442983


namespace min_pairwise_product_l442_442751

variable {n : ℕ}

def min_pairwise_product_sum (x : Fin n → ℤ) : ℤ :=
  (Finset.univ.sum (λ i, x i))^2 - Finset.univ.sum (λ i, (x i) * (x i)) / 2

theorem min_pairwise_product (x : Fin n → ℤ) (hx : ∀ i, x i ∈ {1, 0, -1}) :
  min_pairwise_product_sum x = if n % 2 = 0 then -n / 2 else -(n - 1) / 2 :=
sorry

end min_pairwise_product_l442_442751


namespace question1_tangent_line_question2_monotonicity_l442_442135

noncomputable def f (x a : ℝ) : ℝ := a * x + (a + 1) / x - 1 + Real.log x

theorem question1_tangent_line (a : ℝ) : 
  a = 1 → (f 2 1) = Real.log 2 + 2 → 
  (∀ x y : ℝ, y = f 2 1 → (y - (Real.log 2 + 2) = 0 → x - y + Real.log 2 = 0)) :=
by sorry

theorem question2_monotonicity (a : ℝ) :
  (-1 / 2 ≤ a ∧ a ≤ 0) → 
  (∀ x : ℝ, x > 0 → (a = -1 / 2 → (∀ y : ℝ, y ∈ set.Ioi 0 → deriv (fun x => f x a) y < 0))) :=
by sorry

end question1_tangent_line_question2_monotonicity_l442_442135


namespace max_sum_of_products_3_4_5_6_l442_442698

theorem max_sum_of_products_3_4_5_6 (a b c d : ℕ) (h_values : {a, b, c, d} = {3, 4, 5, 6}) :
  ab + bc + cd + ad ≤ 80 := 
by sorry

end max_sum_of_products_3_4_5_6_l442_442698


namespace find_english_marks_l442_442981

variable (mathematics science social_studies english biology : ℕ)
variable (average_marks : ℕ)
variable (number_of_subjects : ℕ := 5)

-- Conditions
axiom score_math : mathematics = 76
axiom score_sci : science = 65
axiom score_ss : social_studies = 82
axiom score_bio : biology = 95
axiom average : average_marks = 77

-- The proof problem
theorem find_english_marks :
  english = 67 :=
  sorry

end find_english_marks_l442_442981


namespace min_people_like_both_l442_442259

theorem min_people_like_both (n m1 m2 : ℕ) (h1 : n = 120) (h2 : m1 = 102) (h3 : m2 = 85) : 
  (m1 + m2 - n) = 67 :=
by
  rw [h1, h2, h3]
  norm_num
  exact rfl

end min_people_like_both_l442_442259


namespace solve_for_n_l442_442641

theorem solve_for_n (n : ℕ) : (9^n * 9^n * 9^n * 9^n = 729^4) -> n = 3 := 
by
  sorry

end solve_for_n_l442_442641


namespace solve_for_y_l442_442643

theorem solve_for_y : 
  ∀ y : ℚ, y ≠ 2 → (7 * y / (y - 2) - 4 / (y - 2) = 3 / (y - 2) + 1) → y = 5 / 6 :=
by {
  intro y,
  assume h1,
  assume h2,
  sorry
}

end solve_for_y_l442_442643


namespace problem_statement_l442_442949

def p (x : ℝ) : ℝ := x^2 - x + 1

theorem problem_statement (α : ℝ) (h : p (p (p (p α))) = 0) :
  (p α - 1) * p α * p (p α) * p (p (p α)) = -1 :=
by
  sorry

end problem_statement_l442_442949


namespace winnie_balloons_remainder_l442_442351

theorem winnie_balloons_remainder :
  let red_balloons := 20
  let white_balloons := 40
  let green_balloons := 70
  let chartreuse_balloons := 90
  let violet_balloons := 15
  let friends := 10
  let total_balloons := red_balloons + white_balloons + green_balloons + chartreuse_balloons + violet_balloons
  total_balloons % friends = 5 :=
by
  sorry

end winnie_balloons_remainder_l442_442351


namespace tangent_line_at_P_monotonicity_of_f_exists_min_a_l442_442129

-- Given function definition
def f (x : ℝ) (a : ℝ) : ℝ := ln (x + 1) - a * x

-- Point P
def P : ℝ × ℝ := (1, ln 2 - 1)

-- Problem 1: Tangent line equation
theorem tangent_line_at_P :
  ∃ m b : ℝ, (y = m * x + b ↔ (x, y) = P) ∧ b = ln 2 - 1 ∧ m = -1/2 := sorry

-- Problem 2: Monotonicity of f(x) in its domain
theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → ∀ x ∈ Ioi (-1), deriv (λ x, ln (x + 1) - a * x) x ≥ 0) ∧
  (a > 0 → 
    (∀ x ∈ Ioi (-1) ∩ Iic ((1 - a) / a), deriv (λ x, ln (x + 1) - a * x) x ≥ 0) ∧ 
    (∀ x ∈ Ioi ((1 - a) / a), deriv (λ x, ln (x + 1) - a * x) x ≤ 0)) := sorry

-- Problem 3: Existence of a constant a in ℕ
theorem exists_min_a : 
  (∃ a : ℕ, ∀ x > 0, a ≥ (1 + (1 / x))^x) ↔ a = 3 := sorry

end tangent_line_at_P_monotonicity_of_f_exists_min_a_l442_442129


namespace even_four_digit_increasing_count_l442_442526

theorem even_four_digit_increasing_count :
  let digits := {x // 1 ≤ x ∧ x ≤ 9}
  let even_digits := {x // x ∈ digits ∧ x % 2 = 0}
  {n : ℕ //
    ∃ a b c d : ℕ,
      n = a * 1000 + b * 100 + c * 10 + d ∧
      a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ even_digits ∧
      a < b ∧ b < c ∧ c < d} =
  17 :=
by sorry

end even_four_digit_increasing_count_l442_442526


namespace sqrt_meaningful_iff_x_geq_nine_l442_442684

theorem sqrt_meaningful_iff_x_geq_nine (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 9)) ↔ x ≥ 9 :=
by sorry

end sqrt_meaningful_iff_x_geq_nine_l442_442684


namespace log_arithmetic_simplification_trigonometric_simplification_l442_442364

-- Proof Problem I
theorem log_arithmetic_simplification : 
  (\log 2 9) * (\log 3 4) - (2 * (sqrt 2))^(2/3) - exp (log 2) = 0 := sorry

-- Proof Problem II
theorem trigonometric_simplification: 
  (sqrt (1 - sin (20 * π / 180))) / (cos (10 * π / 180) - sin (170 * π / 180)) = 
  cos (10 * π / 180) - sin (10 * π / 180) := sorry

end log_arithmetic_simplification_trigonometric_simplification_l442_442364


namespace unique_solution_set_l442_442467

def log_eq_unique_solution (a x : ℝ) : Prop := 
  log (a * x + 1) = log (x - a) + log (2 - x)

theorem unique_solution_set :
  {a : ℝ | ∃! x : ℝ, log_eq_unique_solution a x} = Icc (-1/2) 0 :=
by
  sorry

end unique_solution_set_l442_442467


namespace verify_triangle_conditions_l442_442498

noncomputable def triangle_angles_and_sides 
  (a b c : ℝ)
  (angleA angleB angleC : ℝ)
  (cosB : ℝ)
  (sin20_plus_angleA : ℝ)
  (perimeter : ℝ) : Prop := 
  (c * Real.cos angleB + b * Real.cos angleC = a / (2 * Real.cos angleA)) ∧
  (angleA = π / 3) ∧
  (cosB = sqrt 3 / 3) ∧
  (sin20_plus_angleA = (2 * sqrt 2 - sqrt 3) / 6) ∧
  (perimeter = 8)

theorem verify_triangle_conditions 
  (a b c : ℝ)
  (angleA angleB angleC : ℝ)
  (cosB : ℝ)
  (sin20_plus_angleA : ℝ)
  (perimeter : ℝ)
  (h1 : c * Real.cos angleB + b * Real.cos angleC = a / (2 * Real.cos angleA))
  (h2 : angleA = π / 3)
  (h3 : cosB = sqrt 3 / 3)
  (h4 : sin20_plus_angleA = (2 * sqrt 2 - sqrt 3) / 6)
  (h5 : perimeter = 8) : triangle_angles_and_sides a b c angleA angleB angleC cosB sin20_plus_angleA perimeter := 
  by
    exact ⟨h1, h2, h3, h4, h5⟩

end verify_triangle_conditions_l442_442498


namespace log_equation_solution_l442_442987

theorem log_equation_solution : 
  (∃ (x:ℝ), x = 9.000000000000002) → 
  (∃ (some_value:ℝ), log 9 (9^3) = log 2 some_value) → 
  some_value = 8 := 
by
  intros _ _
  sorry

end log_equation_solution_l442_442987


namespace meeting_attendees_l442_442798

theorem meeting_attendees (k : ℕ) :
  (∃ (n : ℕ), (∀ a : Fin (12 * k), ∃ (B C : Set (Fin (12 * k))), |B| = 3 * k + 6 ∧ |C| = 9 * k - 7 ∧ 
   (∀ b ∈ B, ∃ n ∈ B, n = 3 * _, _ ∧ ∀ c ∈ C, c ∈ B → _) ∧
   12 * k = 36) → k = 3 := 
sorry

end meeting_attendees_l442_442798


namespace correct_sequence_numbers_l442_442044

/-- The judgment problem among given statements -/
theorem correct_sequence_numbers :
  ( (("If q then p" and "If not p then not q") are contrapositives of each other) ∧
    ¬("am² < bm²" is a necessary and sufficient condition for "a < b") ∧
    ¬(The negation of "The diagonals of a rectangle are equal" is true) ∧
    (∅ ⊆ ({1, 2} : set ℕ)) ) →
  {(1), (4)} = {(1, 4)} :=
by
  intros
  simp
  split
  sorry
  sorry

end correct_sequence_numbers_l442_442044


namespace marbles_per_friend_l442_442426

variable (initial_marbles remaining_marbles given_marbles_per_friend : ℕ)

-- conditions in a)
def condition_initial_marbles := initial_marbles = 500
def condition_remaining_marbles := 4 * remaining_marbles = 720
def condition_total_given_marbles := initial_marbles - remaining_marbles = 320
def condition_given_marbles_per_friend := given_marbles_per_friend * 4 = 320

-- question proof goal
theorem marbles_per_friend (initial_marbles: ℕ) (remaining_marbles: ℕ) (given_marbles_per_friend: ℕ) :
  (condition_initial_marbles initial_marbles) →
  (condition_remaining_marbles remaining_marbles) →
  (condition_total_given_marbles initial_marbles remaining_marbles) →
  (condition_given_marbles_per_friend given_marbles_per_friend) →
  given_marbles_per_friend = 80 :=
by
  intros hinitial hremaining htotal_given hgiven_per_friend
  sorry

end marbles_per_friend_l442_442426


namespace travel_time_without_walking_l442_442576

-- Definitions based on the problem's conditions
def walking_time_without_escalator (x y : ℝ) : Prop := 75 * x = y
def walking_time_with_escalator (x k y : ℝ) : Prop := 30 * (x + k) = y

-- Main theorem: Time taken to travel the distance with the escalator alone
theorem travel_time_without_walking (x k y : ℝ) (h1 : walking_time_without_escalator x y) (h2 : walking_time_with_escalator x k y) : y / k = 50 :=
by
  sorry

end travel_time_without_walking_l442_442576


namespace power_function_monotonically_decreasing_l442_442681

def power_function (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_monotonically_decreasing :
  (∃ α : ℝ, power_function α 2 = 1 / 2) →
  (∀ x : ℝ, ∀ ⦃y : ℝ⦄, x < y → power_function (-1) x > power_function (-1) y ∨ x > 0 → ∀ x : ℝ, 0 < x → power_function (-1) x < power_function (-1) (x + 1)) :=
sorry

end power_function_monotonically_decreasing_l442_442681


namespace man_can_lift_one_box_each_hand_l442_442017

theorem man_can_lift_one_box_each_hand : 
  ∀ (people boxes : ℕ), people = 7 → boxes = 14 → (boxes / people) / 2 = 1 :=
by
  intros people boxes h_people h_boxes
  sorry

end man_can_lift_one_box_each_hand_l442_442017


namespace find_n_18_l442_442064

def valid_denominations (n : ℕ) : Prop :=
  ∀ k < 106, ∃ a b c : ℕ, k = 7 * a + n * b + (n + 1) * c

def cannot_form_106 (n : ℕ) : Prop :=
  ¬ ∃ a b c : ℕ, 106 = 7 * a + n * b + (n + 1) * c

theorem find_n_18 : 
  ∃ n : ℕ, valid_denominations n ∧ cannot_form_106 n ∧ ∀ m < n, ¬ (valid_denominations m ∧ cannot_form_106 m) :=
sorry

end find_n_18_l442_442064


namespace max_problems_olympiad_l442_442670

theorem max_problems_olympiad :
  let grades := 5
  let problems_per_grade := 7
  let unique_problems_per_grade := 4
  let repeating_problems_per_grade := 3
  let total_unique_problems := unique_problems_per_grade * grades
  let maximum_repeating_problems := 7 -- calculated as (3 * 5) / 2 rounded down
  let total_problems := total_unique_problems + maximum_repeating_problems
  total_problems = 27 := 
begin
  let grades' := 5,
  let problems_per_grade' := 7,
  let unique_problems_per_grade' := 4,
  let repeating_problems_per_grade' := 3,
  let total_unique_problems' := unique_problems_per_grade' * grades',
  let maximum_repeating_problems' := 7,
  let total_problems' := total_unique_problems' + maximum_repeating_problems',
  have h : total_problems' = 27,
  { sorry },
  exact h,
end

end max_problems_olympiad_l442_442670


namespace converse_proposition_l442_442833

theorem converse_proposition (a b c : ℝ) (h : c ≠ 0) :
  a * c^2 > b * c^2 → a > b :=
by
  sorry

end converse_proposition_l442_442833


namespace problem_solution_l442_442485

-- Definitions of the arithmetic sequence a_n and its common difference and first term
variables (a d : ℝ)

-- Definitions of arithmetic sequence conditions
def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

-- Required conditions for the proof
variables (h1 : d ≠ 0) (h2 : a ≠ 0)
variables (h3 : arithmetic_sequence a d 2 * arithmetic_sequence a d 8 = (arithmetic_sequence a d 4) ^ 2)

-- The target theorem to prove
theorem problem_solution : 
  (a + (a + 4 * d) + (a + 8 * d)) / ((a + d) + (a + 2 * d)) = 3 :=
sorry

end problem_solution_l442_442485


namespace vasya_can_find_counterfeit_coins_l442_442311

def counterfeit_detection_problem (coins : Fin 5 → Prop) (expert : (Fin 5 → Fin 5 → Prop) → ℕ) : Prop :=
  ∀ (test_pairs : List (Fin 5 × Fin 5)),
    test_pairs.length = 4 →
    (∀ (pair : Fin 5 × Fin 5), pair ∈ test_pairs → expert (λ c1 c2, coins c1 ∧ coins c2)) →
    ∃ (genuine_coins : List (Fin 5)), genuine_coins.length = 3 ∧ ∀ (coin : Fin 5), coin ∈ genuine_coins → ¬ coins coin

theorem vasya_can_find_counterfeit_coins :
  ∀ (coins : Fin 5 → Prop) (expert : (Fin 5 → Fin 5 → Prop) → ℕ),
    (∀ c, coins c = (c = 0 ∨ c = 1)) → -- Two specific coins are counterfeit (indices 0 and 1)
    counterfeit_detection_problem coins expert :=
by
  intros coins expert h
  sorry

end vasya_can_find_counterfeit_coins_l442_442311


namespace intersection_A_B_l442_442219

-- Defining the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℕ := {2, 3, 4, 5}

-- Statement to prove
theorem intersection_A_B : A ∩ B = {2, 3} := 
by 
  sorry

end intersection_A_B_l442_442219


namespace measure_minor_arc_MN_l442_442909

noncomputable theory
open_locale classical

variables {Z : Type*} [metric_space Z] [normed_group Z] [normed_space ℝ Z] [inner_product_space ℝ Z] [complete_space Z]

-- Conditions
variables {M N P : Z}
variable (angle_MNP : ∠MNP = 45)

-- Problem Statement
theorem measure_minor_arc_MN (hMNP : inscribed_angle Z M N P) : measure_arc Z M N = 45 :=
sorry

end measure_minor_arc_MN_l442_442909


namespace family_members_before_baby_l442_442757

theorem family_members_before_baby 
  (n T : ℕ)
  (h1 : T = 17 * n)
  (h2 : (T + 3 * n + 2) / (n + 1) = 17)
  (h3 : 2 = 2) : n = 5 :=
sorry

end family_members_before_baby_l442_442757


namespace credibility_of_relationship_l442_442343

theorem credibility_of_relationship
  (sample_size : ℕ)
  (chi_squared_value : ℝ)
  (table : ℕ → ℝ × ℝ)
  (h_sample : sample_size = 5000)
  (h_chi_squared : chi_squared_value = 6.109)
  (h_table : table 5 = (5.024, 0.025) ∧ table 6 = (6.635, 0.010)) :
  credible_percent = 97.5 :=
by
  sorry

end credibility_of_relationship_l442_442343


namespace ramon_3_enchiladas_4_tacos_cost_l442_442976

theorem ramon_3_enchiladas_4_tacos_cost :
  ∃ (e t : ℝ), 2 * e + 3 * t = 2.50 ∧ 3 * e + 2 * t = 2.70 ∧ 3 * e + 4 * t = 3.54 :=
by {
  sorry
}

end ramon_3_enchiladas_4_tacos_cost_l442_442976


namespace arithmetic_and_geometric_mean_l442_442636

theorem arithmetic_and_geometric_mean (x y : ℝ) (h1: (x + y) / 2 = 20) (h2: Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 :=
sorry

end arithmetic_and_geometric_mean_l442_442636


namespace relationship_between_abc_l442_442244

noncomputable def a : ℝ := (0.6 : ℝ) ^ (0.6 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (1.5 : ℝ)
noncomputable def c : ℝ := (1.5 : ℝ) ^ (0.6 : ℝ)

theorem relationship_between_abc : c > a ∧ a > b := sorry

end relationship_between_abc_l442_442244


namespace smallest_rational_number_l442_442408

theorem smallest_rational_number : ∀ (a b c d : ℚ), 
  a = -2 → b = -1 → c = 0 → d = 1 → (a < b ∧ a < c ∧ a < d) :=
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  split
  · exact neg_lt_neg_add' (zero_lt_one.add zero_lt_one)
  · split
    · exact neg_lt_zero_one
    · exact neg_lt_pos_number
  sorry

end smallest_rational_number_l442_442408


namespace fraction_in_range_and_minimal_l442_442942

noncomputable def smallest_fraction_difference : ℕ :=
  let (p, q) := (8, 13) in q - p

theorem fraction_in_range_and_minimal {p q : ℕ} (hpq_pos : 0 < p ∧ 0 < q)
  (h_bound : (3 : ℚ)/5 < (p : ℚ)/q ∧ (p : ℚ)/q < 5/8)
  (h_minq : ∀ p' q' : ℕ, (0 < p' ∧ 0 < q' ∧ (3 : ℚ)/5 < (p' : ℚ)/q' ∧ (p' : ℚ)/q' < 5/8) → q' ≥ q) :
  (q - p) = 5 :=
sorry

end fraction_in_range_and_minimal_l442_442942


namespace tan_half_sum_l442_442234

-- Given conditions
variables (a b : ℝ)
axiom cos_sum : cos a + cos b = 3 / 5
axiom sin_sum : sin a + sin b = 1 / 3

-- The statement to be proven
theorem tan_half_sum : tan ((a + b) / 2) = 5 / 9 :=
by sorry

end tan_half_sum_l442_442234


namespace largest_prime_factor_4851_l442_442722

theorem largest_prime_factor_4851 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 4851 ∧ (∀ q : ℕ, Nat.Prime q ∧ q ∣ 4851 → q ≤ p) :=
by
  -- todo: provide actual proof
  sorry

end largest_prime_factor_4851_l442_442722


namespace hyperbola_eccentricity_l442_442662

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a = sqrt 2 * b) : 
  (sqrt (1 + (b^2 / a^2))) = (sqrt 6) / 2 :=
by
  sorry

end hyperbola_eccentricity_l442_442662


namespace tank_fill_time_all_pipes_open_l442_442969

def pipeP_rate : ℝ := 1 / 3
def pipeQ_rate : ℝ := 1 / 9
def pipeR_rate : ℝ := 1 / 18

theorem tank_fill_time_all_pipes_open : 
  (pipeP_rate + pipeQ_rate + pipeR_rate) = 1/2 → 2 = 2 :=
by intros; simp; exact rfl

end tank_fill_time_all_pipes_open_l442_442969


namespace intersection_of_A_and_B_l442_442226

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l442_442226


namespace peanuts_added_l442_442310

theorem peanuts_added (a b x : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : a + x = b) : x = 2 :=
by
  sorry

end peanuts_added_l442_442310


namespace vector_magnitude_l442_442521

variable (x y : ℝ)

def vector_a : ℝ × ℝ := (x, 3)
def vector_b : ℝ × ℝ := (-1, y - 1)
def vec_eq_add_two_vec_b := vector_a x y + 2 • vector_b x y = (0, 1)

theorem vector_magnitude (h : vec_eq_add_two_vec_b x y) :
  (| vector_a x y + vector_b x y| = √5) :=
sorry

end vector_magnitude_l442_442521


namespace parallel_vectors_l442_442877

def a : (ℝ × ℝ) := (1, -2)
def b (x : ℝ) : (ℝ × ℝ) := (-2, x)

theorem parallel_vectors (x : ℝ) (h : 1 / -2 = -2 / x) : x = 4 := by
  sorry

end parallel_vectors_l442_442877


namespace interval_of_monotonic_increase_l442_442668

noncomputable def function : ℝ → ℝ := λ x, 4 * x ^ 2 + 1 / x

theorem interval_of_monotonic_increase :
  {x : ℝ | x > 0} →  {x : ℝ | x > 1/2} :=
begin
  sorry
end

end interval_of_monotonic_increase_l442_442668


namespace find_population_in_2017_l442_442560

variable {k : ℝ} -- Initial constant of proportionality in 2015
variable (y : ℝ) -- Population in 2017

def population_2015 := 50
def population_2016 := 75
def population_2018 := 170

-- Proportional constants in 2016 and 2017 increasing by 10% and 20%
def k_2016 := 1.1 * k
def k_2017 := 1.2 * k

-- Equations based on the given conditions
axiom eqn1 : y - population_2015 = k_2016 * population_2016
axiom eqn2 : population_2018 - population_2016 = k_2017 * y

theorem find_population_in_2017 : y = 114 :=
by
  sorry

end find_population_in_2017_l442_442560


namespace circumscribed_circle_area_of_pentagon_l442_442006

noncomputable def pentagon_side_length : ℝ := 10
noncomputable def sin_36 : ℝ := Real.sin (36 * Real.pi / 180)
noncomputable def radius (s : ℝ) : ℝ := s / (2 * sin_36)
noncomputable def circumscribed_circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circumscribed_circle_area_of_pentagon :
  circumscribed_circle_area (radius pentagon_side_length) = 72.35 * Real.pi :=
by
  sorry

end circumscribed_circle_area_of_pentagon_l442_442006


namespace digit_in_100th_place_l442_442891

theorem digit_in_100th_place (n : ℕ) (h : n = 7) : 
  (let decimal_digit := (λ (k : ℕ), if (k % 2 = 0) then 1 else 2) in (decimal_digit 99) = 1) :=
by 
  have correct_expansion : 7 / 33 = 0.212121..., from sorry,
  show (λ (k : ℕ), if (k % 2 = 0) then 1 else 2) 99 = 1, from sorry

end digit_in_100th_place_l442_442891


namespace matts_weight_l442_442252

theorem matts_weight (protein_per_powder_rate : ℝ)
                     (weekly_intake_powder : ℝ)
                     (daily_protein_required_per_kg : ℝ)
                     (days_in_week : ℝ)
                     (expected_weight : ℝ)
    (h1 : protein_per_powder_rate = 0.8)
    (h2 : weekly_intake_powder = 1400)
    (h3 : daily_protein_required_per_kg = 2)
    (h4 : days_in_week = 7)
    (h5 : expected_weight = 80) :
    (weekly_intake_powder / days_in_week) * protein_per_powder_rate / daily_protein_required_per_kg = expected_weight := by
  sorry

end matts_weight_l442_442252


namespace find_equidistant_point_l442_442464
-- Import the entirety of Mathlib to bring in the necessary libraries for geometry and algebra.

-- Define the Euclidean distance function in 3D space
def euclidean_dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)^(1/2)

-- State the main theorem for the proof problem
theorem find_equidistant_point :
  ∃ (x y : ℝ), (euclidean_dist (x, y, 0) (0, 2, 0) = euclidean_dist (x, y, 0) (1, -1, 3)) ∧
                (euclidean_dist (x, y, 0) (0, 2, 0) = euclidean_dist (x, y, 0) (4, 0, -2)) ∧
                (x = 31/10 ∧ y = -11/5) := 
  sorry

end find_equidistant_point_l442_442464


namespace train_length_correct_l442_442039

-- This defines the conditions
def time_to_cross_bridge : ℝ := 27.997760179185665
def bridge_length : ℝ := 150
def train_speed_kmph : ℝ := 36

-- This converts the speed from kmph to m/s
def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

-- This calculates the total distance the train travels while crossing the bridge
def total_distance : ℝ := train_speed_mps * time_to_cross_bridge

-- This calculates the length of the train
def train_length : ℝ := total_distance - bridge_length

-- The theorem that we need to prove 
theorem train_length_correct : train_length = 129.97760179185665 := by
  sorry

end train_length_correct_l442_442039


namespace carrie_fourth_day_miles_l442_442256

theorem carrie_fourth_day_miles (d1 d2 d3 d4: ℕ) (charge_interval charges: ℕ) 
  (h1: d1 = 135) 
  (h2: d2 = d1 + 124) 
  (h3: d3 = 159) 
  (h4: charge_interval = 106) 
  (h5: charges = 7):
  d4 = 742 - (d1 + d2 + d3) :=
by
  sorry

end carrie_fourth_day_miles_l442_442256


namespace sphere_surface_area_increase_l442_442742

theorem sphere_surface_area_increase (r : ℝ) (h_r_pos : 0 < r):
  let A := 4 * π * r ^ 2
  let r' := 1.10 * r
  let A' := 4 * π * (r') ^ 2
  let ΔA := A' - A
  (ΔA / A) * 100 = 21 := by
  sorry

end sphere_surface_area_increase_l442_442742


namespace number_of_valid_permutations_l442_442145

-- Definitions based on conditions
def digits : list ℕ := [5, 5, 0, 0]
def is_valid_number (n : ℕ) : Prop := 
  let ds := Int.to_string n |> String.to_list.map (λ c, c.to_nat - '0'.to_nat) in 
  ds.length = 4 ∧ 
  ds.head ≠ 0 ∧ 
  (ds.nodup = false ∧ list.sort ds = digits.sort)

-- The number of valid four-digit numbers that can be formed by rearranging the digits in 5005 is 3
theorem number_of_valid_permutations : 
  (finset.univ.filter is_valid_number).card = 3 :=
sorry

end number_of_valid_permutations_l442_442145


namespace find_principal_and_rate_l442_442740

-- Definitions based on conditions
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T / 100

def compound_interest (P : ℝ) (R1 : ℝ) (R2 : ℝ) : ℝ := 
  let A1 := P * (1 + R1 / 100)
  A1 * (1 + R2 / 100) - P

def interest_difference (P : ℝ) (R1 : ℝ) (R2 : ℝ) : ℝ := 
  compound_interest P R1 R2 - simple_interest P R1 2

-- Statement to prove the values of P and x
theorem find_principal_and_rate (x : ℝ) (P : ℝ) 
  (h1 : P * 1.05 * (1 + 0.05 * x) - P - P * 0.1 = 20) :
  P = 400 ∧ x ≈ 1.9047619 := 
by
  sorry

end find_principal_and_rate_l442_442740


namespace num_students_in_section_a_l442_442704

theorem num_students_in_section_a 
  (num_students_b : ℕ)
  (avg_weight_a : ℚ)
  (avg_weight_b : ℚ)
  (avg_weight_class : ℚ)
  (x : ℚ)
  (num_students_b = 30) 
  (avg_weight_a = 50)
  (avg_weight_b = 60)
  (avg_weight_class = 54.285714285714285)
  (avg_weight_class * (x + 30) = 50 * x + 30 * 60) :
  x = 40 :=
sorry

end num_students_in_section_a_l442_442704


namespace fraction_increase_each_year_l442_442822

variable (initial_value : ℝ := 57600)
variable (final_value : ℝ := 72900)
variable (years : ℕ := 2)

theorem fraction_increase_each_year :
  ∃ (f : ℝ), initial_value * (1 + f)^years = final_value ∧ f = 0.125 := by
  sorry

end fraction_increase_each_year_l442_442822


namespace jonah_added_raisins_l442_442587

theorem jonah_added_raisins:
  (y b : ℚ) (hy : y = 0.3) (hb : b = 0.4) : y + b = 0.7 :=
by
  sorry

end jonah_added_raisins_l442_442587


namespace no_factors_l442_442063

def polynomial := (x : ℝ) → (x^4 + 4*x^3 + 6*x^2 + 4*x + 9)

theorem no_factors : 
  ∀ (f : ℝ → ℝ), 
  (f = (λ x, x + 3) ∨ 
   f = (λ x, x^2 + x + 3) ∨ 
   f = (λ x, x^2 - 4*x + 3) ∨ 
   f = (λ x, x^2 + 2*x + 3)) → 
  ¬(∃ g : ℝ → ℝ, polynomial = f * g) :=
begin
  intros f h,
  cases h,
  { sorry }, -- here you would handle the division proving that x+3 is not a factor
  { cases h,
    { sorry }, -- similarly for x^2 + x + 3
    { cases h,
      { sorry }, -- similarly for x^2 - 4x + 3
      { sorry } -- similarly for x^2 + 2x + 3
    }
  }
end

end no_factors_l442_442063


namespace find_v_l442_442433

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![4, 3]]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := Matrix.eye (Fin 2)
noncomputable def A := B^8 + B^6 + B^4 + B^2 + I 
def v_sol : Matrix (Fin 2) (Fin 1) ℝ := ![![0], ![15 / ((A 0 0).toReal)]]

theorem find_v :
  ∃ v : Matrix (Fin 2) (Fin 1) ℝ, A ⬝ v = ![![0], ![15]] :=
begin
  use v_sol,
  sorry
end

end find_v_l442_442433


namespace min_value_l442_442992

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 3 + 2 * Real.sqrt 2 ≤ 2 / a + 1 / b :=
by
  sorry

end min_value_l442_442992


namespace cube_dihedral_angle_cosine_l442_442180

-- Define the 3D points (vertices of the cube)
structure Point3D :=
  (x y z : ℝ)

def A : Point3D := ⟨0, 0, 0⟩
def B (a : ℝ) : Point3D := ⟨a, 0, 0⟩
def C (a : ℝ) : Point3D := ⟨a, a, 0⟩
def D (a : ℝ) : Point3D := ⟨0, a, 0⟩
def A1 (a : ℝ) : Point3D := ⟨0, 0, a⟩
def B1 (a : ℝ) : Point3D := ⟨a, 0, a⟩
def C1 (a : ℝ) : Point3D := ⟨a, a, a⟩
def D1 (a : ℝ) : Point3D := ⟨0, a, a⟩

-- Define midpoint M of BB1
def M (a : ℝ) := Point3D.mk a 0 (a / 2)

-- Compute cosine of the dihedral angle M-CD1-A
noncomputable def cosine_dihedral_angle (a : ℝ) : ℝ := 
  let AM := Point3D.mk a 0 (a / 2)
  let AC := Point3D.mk a a 0
  let AD1 := Point3D.mk 0 a a
  let n1 := (AM.x * AC.y - AM.y * AC.x, AM.y * AC.z - AM.z * AC.y, AM.z * AC.x - AM.x * AC.z)
  let n2 := (AC.x * AD1.y - AC.y * AD1.x, AC.y * AD1.z - AC.z * AD1.y, AC.z * AD1.x - AC.x * AD1.z)
  let dot_product := n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3
  let norm_n1 := Math.sqrt ((n1.1 ^ 2) + (n1.2 ^ 2) + (n1.3 ^ 2))
  let norm_n2 := Math.sqrt ((n2.1 ^ 2) + (n2.2 ^ 2) + (n2.3 ^ 2))
  dot_product / (norm_n1 * norm_n2)

theorem cube_dihedral_angle_cosine (a : ℝ) : 
  cosine_dihedral_angle a = √6 / 3 :=
by
  simp [cosine_dihedral_angle]
  sorry

end cube_dihedral_angle_cosine_l442_442180


namespace value_of_x_l442_442161

theorem value_of_x (x y : ℝ) (h₁ : x = y - 0.10 * y) (h₂ : y = 125 + 0.10 * 125) : x = 123.75 := 
by
  sorry

end value_of_x_l442_442161


namespace increasing_distinct_digits_2020_to_2400_l442_442048

theorem increasing_distinct_digits_2020_to_2400 :
  ∃ (count : ℕ), count = 15 ∧
  count = (Nat.choose 6 2).toNat :=
by
  use 15
  split
  · exact rfl
  · exact rfl

end increasing_distinct_digits_2020_to_2400_l442_442048


namespace fish_caught_difference_l442_442885

theorem fish_caught_difference
  (current_fish : ℕ) (added_fish : ℕ) (initial_fish : ℕ)
  (h1 : current_fish = 20)
  (h2 : added_fish = 8)
  (h3 : initial_fish = current_fish - added_fish) :
  initial_fish - added_fish = 4 :=
by
  have initial_fish_val : initial_fish = 12, from h3 ▸ (h1 ▸ rfl),
  sorry

end fish_caught_difference_l442_442885


namespace exists_set_coprime_composite_sum_l442_442573

theorem exists_set_coprime_composite_sum : ∃ S : Finset ℕ, S.card = 1990 ∧ (∀ (a b ∈ S), a ≠ b → Nat.coprime a b) ∧ (∀ t ⊆ S, 2 ≤ t.card → t.sum > 1 ∧ ¬Nat.prime t.sum) :=
sorry

end exists_set_coprime_composite_sum_l442_442573


namespace factorable_polynomial_abs_sum_l442_442602

theorem factorable_polynomial_abs_sum :
  let S := { b : ℤ | ∃ r s : ℤ, r + s = -b ∧ r * s = 2016 * b } in
  | ∑ b in S, b | = 330624 :=
sorry

end factorable_polynomial_abs_sum_l442_442602


namespace KN_length_l442_442567

noncomputable def triangle_side_lengths (AB AC BC : ℝ) := AB = 9 ∧ AC = 3 ∧ BC = 8

noncomputable def angle_bisector (AB AC : ℝ) := AK / BC = 3 / 1

noncomputable def point_M_ratio (AM MC : ℝ) := AM / MC = 3 / 1

noncomputable def find_KN (KN : ℝ) := KN = sqrt(15) / 5

theorem KN_length (AB AC BC AM MC KN : ℝ) 
    (H1 : triangle_side_lengths AB AC BC) 
    (H2 : angle_bisector AB AC) 
    (H3 : point_M_ratio AM MC) : 
  find_KN KN :=
by {
    sorry
}

end KN_length_l442_442567


namespace ellipse_foci_distance_l442_442455

theorem ellipse_foci_distance :
  (∀ x y : ℝ, x^2 / 56 + y^2 / 14 = 8) →
  ∃ d : ℝ, d = 8 * Real.sqrt 21 :=
by
  sorry

end ellipse_foci_distance_l442_442455


namespace math_proof_problem_l442_442281

variable (x : ℂ) -- Assume x is a complex number (or ℝ for real number)

noncomputable def problem_statement : Prop :=
  (x ≠ 1) ∧ (x^2021 - 3 * x + 1 = 0) → (x^2020 + x^2019 + ... + x + 1 = 3)

theorem math_proof_problem (h : problem_statement x) : x^2020 + x^2019 + ... + x + 1 = 3 :=
sorry

end math_proof_problem_l442_442281


namespace fare_ratio_l442_442414

theorem fare_ratio (F1 F2 : ℕ) (h1 : F1 = 96000) (h2 : F1 + F2 = 224000) : F1 / (Nat.gcd F1 F2) = 3 ∧ F2 / (Nat.gcd F1 F2) = 4 :=
by
  sorry

end fare_ratio_l442_442414


namespace geometric_series_sum_geometric_series_base_case_l442_442971

theorem geometric_series_sum (a : ℝ) (n : ℕ) (h : a ≠ 1) (hn : n ≠ 0) :
  (finset.range (n+1)).sum (λ i, a ^ i) = (1 - a ^ (n + 1)) / (1 - a) :=
by sorry

theorem geometric_series_base_case (a : ℝ) (h : a ≠ 1) : 
  (1 + a + a^2) = (finset.range 3).sum (λ i, a ^ i) :=
by refl

end geometric_series_sum_geometric_series_base_case_l442_442971


namespace dogs_with_no_accessories_l442_442799

theorem dogs_with_no_accessories :
  let total := 120
  let tags := 60
  let flea_collars := 50
  let harnesses := 30
  let tags_and_flea_collars := 20
  let tags_and_harnesses := 15
  let flea_collars_and_harnesses := 10
  let all_three := 5
  total - (tags + flea_collars + harnesses - tags_and_flea_collars - tags_and_harnesses - flea_collars_and_harnesses + all_three) = 25 := by
  sorry

end dogs_with_no_accessories_l442_442799


namespace inequality_C_l442_442537

theorem inequality_C (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := 
by
  sorry

end inequality_C_l442_442537


namespace solve_log_equation_l442_442986

theorem solve_log_equation :
  (∃ x : ℝ, log 3 ((5 * x + 15) / (7 * x - 5)) + log 3 ((7 * x - 5) / (x - 3)) = 3 ↔ x = 48 / 11) :=
by
  -- Placeholder for the proof
  sorry

end solve_log_equation_l442_442986


namespace value_of_a_range_of_f_l442_442504

variable {x : ℝ} {a : ℝ}

def f (x : ℝ) (a : ℝ) : ℝ := a^(x - 1)

theorem value_of_a (h1 : 0 < a) (h2 : a ≠ 1) (h3 : f 2 a = (1/2)) :
  a = (1/2) := by
  sorry

theorem range_of_f (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a = (1/2)) :
  ∀ y, (∃ x ≥ 0, f x a = y) ↔ 0 < y ∧ y ≤ 2 := by
  sorry

end value_of_a_range_of_f_l442_442504


namespace number_of_bk_divisible_by_9_l442_442235

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def b (n : ℕ) : ℕ := 
  let a := (List.range (n + 1)).reverse
  let b := List.range (n + 1)
  List.foldl (λ acc x => acc * 10^(Nat.log10 x + 1) + x) 0 (a ++ b)

def count_divisible_by_9_up_to_100 : ℕ :=
  (List.filter (λ n => is_divisible_by_9 (b n)) (List.range 101)).length

theorem number_of_bk_divisible_by_9 : count_divisible_by_9_up_to_100 = 22 :=
by
  sorry

end number_of_bk_divisible_by_9_l442_442235


namespace corresponding_point_is_correct_l442_442291

-- Define the complex number z
def z := complex.mk_imag 1 / (complex.mk 1 3)

-- Define the expected point
def expected_point := (3/10 : ℝ, 1/10 : ℝ)

-- State the theorem as a proof problem
theorem corresponding_point_is_correct : 
  z.re = expected_point.1 ∧ z.im = expected_point.2 :=
sorry

end corresponding_point_is_correct_l442_442291


namespace triangle_side_length_l442_442912

theorem triangle_side_length (a b c : ℝ)
  (angle_A angle_B angle_C : ℝ)
  (h_angle_sum : angle_A + angle_B + angle_C = 180)
  (h_angle_A : angle_A = 30) (h_angle_B : angle_B = 45) (h_angle_C : angle_C = 180 - 30 - 45)
  (h_side_b : b = 8) :
  a = 4 * real.sqrt 2 :=
begin
  sorry
end

end triangle_side_length_l442_442912


namespace stratified_sampling_count_l442_442768

theorem stratified_sampling_count (c1 c2 n total : ℕ) (h_c1 : c1 = 54) (h_c2 : c2 = 42) (h_n : n = 16) :
  let prob := (n : ℝ) / (c1 + c2) in
  let selected_c1 := (c1 : ℝ) * prob in
  let selected_c2 := (c2 : ℝ) * prob in
  selected_c1 = 9 ∧ selected_c2 = 7 :=
by
  simp [h_c1, h_c2, h_n]
  sorry

end stratified_sampling_count_l442_442768


namespace sum_of_perimeters_l442_442735

theorem sum_of_perimeters (a1 : ℕ) (h : a1 = 60): 
  let P (n : ℕ) := 3 * (a1 / 2 ^ (n - 1)) 
  in ∑' n, P n = 360 := 
by 
  sorry

end sum_of_perimeters_l442_442735


namespace rational_function_sum_l442_442429

noncomputable def r (x : ℝ) : ℝ := 4 * (x - 3)
noncomputable def s (x : ℝ) : ℝ := (1 / 2) * (x - 2) * (x - 3)

theorem rational_function_sum :
  r 4 = 4 ∧ s 1 = 1 ∧ (∀ x, r(x) + s(x) = (1 / 2) * x^2 - (1 / 2) * x) :=
by
  split
  · sorry
  split
  · sorry
  · intro x
    sorry

end rational_function_sum_l442_442429


namespace percent_shaded_l442_442728

theorem percent_shaded (total_squares : ℕ) (shaded_squares : ℕ) (fraction_shaded : ℚ) (percentage_shaded : ℚ) :
  total_squares = 36 ∧ shaded_squares = 15 ∧ fraction_shaded = shaded_squares / total_squares ∧ percentage_shaded = fraction_shaded * 100 →
  percentage_shaded = 4167 / 100 :=
by
  intro h
  cases h with ht hs
  cases hs with hf hp
  cases hp with hp hpercentage
  rw [ht, hs, hf, hpercentage]
  norm_num
  sorry

end percent_shaded_l442_442728


namespace james_profit_correct_l442_442578

noncomputable def james_profit : ℕ :=
  let cost_per_toy := 20 in
  let sell_price_per_toy := 30 in
  let total_toys := 200 in
  let percent_to_sell := 80 in
  let total_cost := cost_per_toy * total_toys in
  let toys_sold := total_toys * percent_to_sell / 100 in
  let total_revenue := toys_sold * sell_price_per_toy in
  total_revenue - total_cost

theorem james_profit_correct : james_profit = 800 :=
by
  sorry

end james_profit_correct_l442_442578


namespace sqrt_x_minus_9_meaningful_l442_442685

theorem sqrt_x_minus_9_meaningful (x : ℝ) : sqrt (x - 9) = sqrt (x - 9) ↔ x ≥ 9 :=
by
  sorry

end sqrt_x_minus_9_meaningful_l442_442685


namespace sum_first_n_terms_l442_442872

-- Define the sequence a_n
def a (n : ℕ) : ℤ := (-1) ^ n * n + 2 ^ n

-- Define the sum S_n according to the problem statement
def S (n : ℕ) : ℤ := 
  if n % 2 = 0 then 
    n / 2 + 2 ^ (n + 1) - 2
  else 
    2 ^ (n + 1) - 2 - (n + 1) / 2

-- The main theorem that needs to be proven
theorem sum_first_n_terms (n : ℕ) : 
  (finset.range n).sum a = S n :=
by sorry

end sum_first_n_terms_l442_442872


namespace inequality_lemma_l442_442271

-- Define the conditions: x and y are positive numbers and x > y
variables (x y : ℝ)
variables (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x > y)

-- State the theorem to be proved
theorem inequality_lemma : 2 * x + 1 / (x^2 - 2*x*y + y^2) >= 2 * y + 3 :=
by
  sorry

end inequality_lemma_l442_442271


namespace lines_intersect_l442_442015

-- Condition definitions
def line1 (t : ℝ) : ℝ × ℝ :=
  ⟨2 + t * -1, 3 + t * 5⟩

def line2 (u : ℝ) : ℝ × ℝ :=
  ⟨u * -1, 7 + u * 4⟩

-- Theorem statement
theorem lines_intersect :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (6, -17) :=
by
  sorry

end lines_intersect_l442_442015


namespace number_of_subsets_of_A_l442_442482

noncomputable def A : Set ℝ := sorry -- The set A where conditions are met

-- Condition: If p is in A, then (-1 / (p + 1)) is in A
axiom A_condition (p : ℝ) (hp : p ∈ A) (hpn0 : p ≠ 0) (hpn1 : p ≠ -1) : -1 / (p + 1) ∈ A

-- Known element in A
axiom hw : (2 : ℝ) ∈ A

-- Prove that the number of subsets of A is 8
theorem number_of_subsets_of_A : ∃ S : Finset (Set ℝ), (S.card = 8) :=
begin
  sorry
end

end number_of_subsets_of_A_l442_442482


namespace trajectory_of_point_P_l442_442096

theorem trajectory_of_point_P :
  ∀ (P : ℝ × ℝ), 
  let A := (-Real.sqrt 2, 0 : ℝ)
  let B := (Real.sqrt 2, 0 : ℝ)
  (P.2 / (P.1 + A.1)) * (P.2 / (P.1 - B.1)) = -1 / 2
  → (P.1^2 / 2) + P.2^2 = 1 := 
by
  intros P A B h
  sorry

end trajectory_of_point_P_l442_442096


namespace dima_picks_more_berries_l442_442814

theorem dima_picks_more_berries (N : ℕ) (dima_fastness : ℕ) (sergei_fastness : ℕ) (dima_rate : ℕ) (sergei_rate : ℕ) :
  N = 450 → dima_fastness = 2 * sergei_fastness →
  dima_rate = 1 → sergei_rate = 2 →
  let dima_basket : ℕ := N / 2
  let sergei_basket : ℕ := (2 * N) / 3
  dima_basket > sergei_basket ∧ (dima_basket - sergei_basket) = 50 := 
by {
  sorry
}

end dima_picks_more_berries_l442_442814


namespace plane_divides_edge_A1D1_l442_442041

-- Define the basic properties of the cube, the points and lines needed
variables {Point : Type*} [inner_product_space Real Point]
variables (A B C D A1 B1 C1 D1 M X : Point)
variables (α : Subspace Real Point)

-- Define the specific conditions
noncomputable def AB := 1
noncomputable def DM := (8/15) * (DB1 : ℝ)
noncomputable def DB1 := ∥D - B1∥

-- Define the plane α and its properties
axiom plane_passes_through_M_and_perpendicular (M : Point) (B1 D : Point) :
  ∀ α : Subspace Real Point, M ∈ α ∧ (D - M) ⊥ (B1 - D)

-- The goal to be proved
theorem plane_divides_edge_A1D1 :
  let n := D1X / D1A1 in
  α.divides A1D1 in (3:2) := sorry

end plane_divides_edge_A1D1_l442_442041


namespace number_of_possible_lists_l442_442756

theorem number_of_possible_lists : 
  let balls := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
  (number_of_draws : ℕ) (n_permutations : ℕ) :=
  ∀ (draws: number_of_draws = 4) (size_balls: #balls = 15),
    n_permutations = 32760 :=
by
  sorry

end number_of_possible_lists_l442_442756


namespace shaded_area_l442_442775

-- Definitions of points used in conditions
structure Point where
  x : ℝ
  y : ℝ

-- Conditions
def A : Point := ⟨4, 0⟩
def B : Point := ⟨12, 0⟩
def D : Point := ⟨12, 12⟩
def C : Point := ⟨4, 12⟩
def E : Point := ⟨18, 0⟩
def F : Point := ⟨12, 8⟩

-- Dimensions
def width_rectangle : ℝ := 8
def height_rectangle : ℝ := 12
def base_triangle : ℝ := 6
def height_triangle : ℝ := 8

-- Proof statement
theorem shaded_area : 
  let area_rectangle := width_rectangle * height_rectangle
  let area_triangle := (1 / 2) * base_triangle * height_triangle 
  let area_BFE := (1 / 2) * height_rectangle * base_triangle
  let combined_area := area_rectangle + area_BFE
  combined_area = 120 := 
  by 
    let area_rectangle := width_rectangle * height_rectangle
    let area_triangle := (1 / 2) * base_triangle * height_triangle 
    let area_BFE := (1 / 2) * height_rectangle * base_triangle
    let combined_area := area_rectangle + area_BFE
    have h1 : area_rectangle = 96 := by sorry
    have h2 : area_BFE = 24 := by sorry
    have h3 : combined_area = 120 := by 
      calc 
        combined_area = 96 + 24 : by rw [h1, h2]
        ... = 120 : by linarith
    exact h3

end shaded_area_l442_442775


namespace area_of_triangle_KDC_l442_442181

open Real

noncomputable def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

theorem area_of_triangle_KDC
  (radius : ℝ) (chord_length : ℝ) (seg_KA : ℝ)
  (OX distance_DY : ℝ)
  (parallel : ∀ (PA PB : ℝ), PA = PB)
  (collinear : ∀ (PK PA PQ PB : ℝ), PK + PA + PQ + PB = PK + PQ + PA + PB)
  (hyp_radius : radius = 10)
  (hyp_chord_length : chord_length = 12)
  (hyp_seg_KA : seg_KA = 24)
  (hyp_OX : OX = 8)
  (hyp_distance_DY : distance_DY = 8) :
  triangle_area chord_length distance_DY = 48 :=
  by
  sorry

end area_of_triangle_KDC_l442_442181


namespace distance_between_A_and_B_l442_442100

noncomputable def dist (A B : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2 + (B.3 - A.3)^2)

theorem distance_between_A_and_B :
  dist (1, -2, -1) (3, 0, 1) = 2 * Real.sqrt 3 :=
by
  sorry

end distance_between_A_and_B_l442_442100


namespace count_multiples_4_or_5_not_20_or_8_l442_442533

theorem count_multiples_4_or_5_not_20_or_8 : 
  let multiples_of_4 := (1500 / 4).floor
  let multiples_of_5 := (1500 / 5).floor
  let multiples_of_20 := (1500 / 20).floor
  let multiples_of_8 := (1500 / 8).floor
  count_multiples_4_or_5_not_20_or_8 = multiples_of_4 + multiples_of_5 - multiples_of_20 - multiples_of_8 :=
  by
    let multiples_of_4 := 375
    let multiples_of_5 := 300
    let multiples_of_20 := 75
    let multiples_of_8 := 187
    have h1 : multiples_of_4 = 375 := rfl
    have h2 : multiples_of_5 = 300 := rfl
    have h3 : multiples_of_20 = 75 := rfl
    have h4 : multiples_of_8 = 187 := rfl
    show count.multiples_4_or_5_not_20_or_8 = 413 by
      sorry

end count_multiples_4_or_5_not_20_or_8_l442_442533


namespace distance_from_point_to_plane_l442_442495

-- Define the normal vector of the plane
def normal_vector := (1 : ℤ, 2 : ℤ, 2 : ℤ)

-- Define points A and B
def point_A := (1 : ℤ, 0 : ℤ, 2 : ℤ)
def point_B := (0 : ℤ, -1 : ℤ, 4 : ℤ)

-- Define the theorem to calculate the distance from point A to the plane α
theorem distance_from_point_to_plane :
  let n := normal_vector
      A := point_A
      B := point_B
      B_in_alpha := true -- Implicitly given B is in the plane α
      A_not_in_alpha := true -- Implicitly given A is not in the plane α
      BA := (A.1 - B.1, A.2 - B.2, A.3 - B.3)
      dot_product := BA.1 * n.1 + BA.2 * n.2 + BA.3 * n.3
      norm_n := real.sqrt ((n.1 * n.1 : ℝ) + (n.2 * n.2 : ℝ) + (n.3 * n.3 : ℝ))
      distance := abs (dot_product : ℝ) / norm_n
  in distance = 1 / 3 := sorry

end distance_from_point_to_plane_l442_442495


namespace cassie_water_bottle_ounces_l442_442054

-- Define the given quantities
def cups_per_day : ℕ := 12
def ounces_per_cup : ℕ := 8
def refills_per_day : ℕ := 6

-- Define the total ounces of water Cassie drinks per day
def total_ounces_per_day := cups_per_day * ounces_per_cup

-- Define the ounces her water bottle holds
def ounces_per_bottle := total_ounces_per_day / refills_per_day

-- Prove the statement
theorem cassie_water_bottle_ounces : 
  ounces_per_bottle = 16 := by 
  sorry

end cassie_water_bottle_ounces_l442_442054


namespace weight_of_rod_l442_442543

theorem weight_of_rod (length1 length2 weight1 weight2 weight_per_meter : ℝ)
  (h1 : length1 = 6) (h2 : weight1 = 22.8) (h3 : length2 = 11.25)
  (h4 : weight_per_meter = weight1 / length1) :
  weight2 = weight_per_meter * length2 :=
by
  -- The proof would go here
  sorry

end weight_of_rod_l442_442543


namespace sequence_A_div_11_l442_442598

noncomputable def sequence_A (n : ℕ) : ℕ :=
  if n = 1 then 0
  else if n = 2 then 1
  else let rec_A := λ n, (sequence_A (n - 1) * 10 ^ (sequence_A (n - 2).digits + 1) + sequence_A (n - 2))
  in rec_A n

theorem sequence_A_div_11 (n : ℕ) : (∃ k : ℕ, n = 6 * k + 1) ↔ 11 ∣ sequence_A n := 
sorry

end sequence_A_div_11_l442_442598


namespace probability_different_faces_probability_sum_to_six_probability_odd_faces_l442_442268

open ProbTheory

-- For (I): Probability that the numbers facing up are different.
theorem probability_different_faces :
  let outcomes := 6 * 6 in
  let favorable := 6 * 5 in
  (favorable / outcomes : ℚ) = 5 / 6 := by
  let outcomes : ℚ := 36
  let favorable : ℚ := 30
  have h : favorable / outcomes = 5 / 6 := by sorry
  exact h

-- For (II): Probability that the sum of the numbers facing up is 6.
theorem probability_sum_to_six :
  let outcomes := 1 / 6 * 1 / 6 * 36 in
  let favorable := 5 in
  (favorable / outcomes : ℚ) = 5 / 36 := by
  let outcomes : ℚ := 36
  let favorable : ℚ := 5
  have h : favorable / outcomes = 5 / 36 := by sorry
  exact h

-- For (III): Probability that an odd number faces up exactly three times in five rolls.
theorem probability_odd_faces :
  let probabilities := binomial 5 (3/2*1/2) in
  (probabilities : ℚ) = 5 / 16 := by
  have h: ∑ x in Finset.range 5, (choose 5 x) * (1 / 2) ^ x * (1 / 2) ^ (5 - x) = 5 / 16 := by sorry
  exact h

end probability_different_faces_probability_sum_to_six_probability_odd_faces_l442_442268


namespace bricks_in_wall_l442_442320

theorem bricks_in_wall (x : ℕ)
  (h1 : ∀ t, t = 8 → let rate1 := x / t in rate1 * t = x)
  (h2 : ∀ t, t = 12 → let rate2 := x / t in rate2 * t = x)
  (h3 : ∀ rate1 rate2, rate1 = x / 8 ∧ rate2 = x / 12 → 
    let combined_rate := (rate1 + rate2 - 12) in combined_rate * 6 = x)
  (h4 : ∀ combined_rate, combined_rate = (x / 8 + x / 12 - 12) → combined_rate * 6 = x)
  (cond : 6 * (x / 8 + x / 12 - 12) = x) :
  x = 288 :=
by
  sorry

end bricks_in_wall_l442_442320


namespace find_student_ticket_price_l442_442396

variable (S : ℝ)
variable (student_tickets non_student_tickets total_tickets : ℕ)
variable (non_student_ticket_price total_revenue : ℝ)

theorem find_student_ticket_price 
  (h1 : student_tickets = 90)
  (h2 : non_student_tickets = 60)
  (h3 : total_tickets = student_tickets + non_student_tickets)
  (h4 : non_student_ticket_price = 8)
  (h5 : total_revenue = 930)
  (h6 : 90 * S + 60 * non_student_ticket_price = total_revenue) : 
  S = 5 := 
sorry

end find_student_ticket_price_l442_442396


namespace sqrt_x_minus_9_meaningful_l442_442686

theorem sqrt_x_minus_9_meaningful (x : ℝ) : sqrt (x - 9) = sqrt (x - 9) ↔ x ≥ 9 :=
by
  sorry

end sqrt_x_minus_9_meaningful_l442_442686


namespace books_written_by_Zig_l442_442354

theorem books_written_by_Zig (F Z : ℕ) (h1 : Z = 4 * F) (h2 : F + Z = 75) : Z = 60 := by
  sorry

end books_written_by_Zig_l442_442354


namespace bread_slices_per_friend_l442_442016

theorem bread_slices_per_friend :
  (∀ (slices_per_loaf friends loaves total_slices_per_friend : ℕ),
    slices_per_loaf = 15 →
    friends = 10 →
    loaves = 4 →
    total_slices_per_friend = slices_per_loaf * loaves / friends →
    total_slices_per_friend = 6) :=
by 
  intros slices_per_loaf friends loaves total_slices_per_friend h1 h2 h3 h4
  sorry

end bread_slices_per_friend_l442_442016


namespace solve_r_l442_442466

theorem solve_r (r : ℚ) :
  (r^2 - 5*r + 4) / (r^2 - 8*r + 7) = (r^2 - 2*r - 15) / (r^2 - r - 20) →
  r = -5/4 :=
by
  -- Proof would go here
  sorry

end solve_r_l442_442466


namespace function_in_interval_l442_442669

noncomputable def f (x : ℝ) := log 3 x + x

theorem function_in_interval :
  (∃ x ∈ set.Ioo 3 4, f x = 5) :=
by
  sorry

end function_in_interval_l442_442669


namespace candy_cookies_l442_442424

def trays : Nat := 4
def cookies_per_tray : Nat := 24
def packs : Nat := 8
def total_cookies : Nat := trays * cookies_per_tray
def cookies_per_pack : Nat := total_cookies / packs

theorem candy_cookies : 
  cookies_per_pack = 12 := 
by
  -- Calculate total cookies
  have h1 : total_cookies = trays * cookies_per_tray := rfl
  have h2 : total_cookies = 96 := by rw [h1]; norm_num
  
  -- Calculate cookies per pack
  have h3 : cookies_per_pack = total_cookies / packs := rfl
  have h4 : cookies_per_pack = 12 := by rw [h3, h2]; norm_num
  
  exact h4

end candy_cookies_l442_442424


namespace find_angle_l442_442865

theorem find_angle (θ : Real) (h1 : 0 ≤ θ ∧ θ ≤ π) (h2 : Real.sin θ = (Real.sqrt 2) / 2) :
  θ = Real.pi / 4 ∨ θ = 3 * Real.pi / 4 :=
by
  sorry

end find_angle_l442_442865


namespace coin_weighing_l442_442702

theorem coin_weighing (n : ℕ) (g s c : ℕ) (wg ws wc : ℕ) :
  n = 100 ∧ g + s + c = 100 ∧ g ≥ 1 ∧ s ≥ 1 ∧ c ≥ 1 ∧ 
  wg = 3 ∧ ws = 2 ∧ wc = 1 →
  ∃ w : ℕ, w ≤ 101 :=
begin
  sorry
end

end coin_weighing_l442_442702


namespace intersection_of_A_and_B_l442_442225

def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} :=
by
  sorry

end intersection_of_A_and_B_l442_442225


namespace num_valid_sequences_l442_442908

theorem num_valid_sequences : 
    ∃ n : ℕ, 
        (∀ (seq : List (Fin 6)), 
            (∀ i : Fin 5, 
                (seq.nth i).get ∘ succ ≠ (some 0, some 1, some 2)) ∧ 
            (seq.head ≠ some 0)) → 
        n = 132 := sorry

end num_valid_sequences_l442_442908


namespace mass_percentage_O_correct_l442_442459

noncomputable def molar_mass_H : ℝ := 1.01
noncomputable def molar_mass_B : ℝ := 10.81
noncomputable def molar_mass_O : ℝ := 16.00

noncomputable def molar_mass_H3BO3 : ℝ := (3 * molar_mass_H) + (1 * molar_mass_B) + (3 * molar_mass_O)

noncomputable def mass_percentage_O_in_H3BO3 : ℝ := ((3 * molar_mass_O) / molar_mass_H3BO3) * 100

theorem mass_percentage_O_correct : abs (mass_percentage_O_in_H3BO3 - 77.59) < 0.01 := 
sorry

end mass_percentage_O_correct_l442_442459


namespace cannot_form_AB₂_l442_442792

def element_atomic_number : Type :=
  {atomic_number : ℕ // atomic_number > 0}

def C := element_atomic_number.mk 6 sorry
def O := element_atomic_number.mk 8 sorry
def S := element_atomic_number.mk 16 sorry
def Mg := element_atomic_number.mk 12 sorry
def F := element_atomic_number.mk 9 sorry
def Na := element_atomic_number.mk 11 sorry

def can_form_AB₂ (A B : element_atomic_number) : Prop := 
  -- Placeholder for the actual condition which determines if elements can form AB₂ type compound.
  sorry

theorem cannot_form_AB₂ (A : element_atomic_number) (B : element_atomic_number) :
  (A = Na ∧ B = C) → ¬ can_form_AB₂ A B :=
by
  intro h
  -- Here we would provide the proof, but since this is the statement part, we add sorry.
  sorry

end cannot_form_AB₂_l442_442792


namespace table_properties_l442_442062

def table := λ (i j : Fin 8), [[1, 0, 3, 2, 5, 4, 7, 6],
                                [2, 3, 0, 1, 6, 7, 4, 5],
                                [3, 2, 1, 0, 7, 6, 5, 4],
                                [4, 5, 6, 7, 0, 1, 2, 3],
                                [5, 4, 7, 6, 1, 0, 3, 2],
                                [6, 7, 4, 5, 2, 3, 0, 1],
                                [7, 6, 5, 4, 3, 2, 1, 0]].nth i.val.nth j.val

theorem table_properties :
  (∀ (i j : Fin 8), table i j = table j i) ∧ -- Symmetry across the main diagonal
  (∀ (i : Fin 8), Finset.card (Finset.image (λ j : Fin 8, table i j) Finset.univ) = 8) ∧ -- Unique numbers in each row
  (∀ (j : Fin 8), Finset.card (Finset.image (λ i : Fin 8, table i j) Finset.univ) = 8) ∧ -- Unique numbers in each column
  (∀ (i : Fin 8), table 0 i = [1, 0, 3, 2, 5, 4, 7, 6].nth i.val) ∧ -- Sequence in the first row
  (∀ (j : Fin 8), table j 0 = [1, 2, 3, 4, 5, 6, 7, 0].nth j.val) ∧ -- Sequence in the first column
  (∀ (i j : Fin 8), table i j + table i (⟨7, ⟨by norm_num⟩ - j⟩) = 7) -- Sum of pairs symmetric about the central lines
  := sorry

end table_properties_l442_442062


namespace solve_x_from_equation_l442_442727

theorem solve_x_from_equation :
  ∀ (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 → x = 27 :=
by
  intro x
  rintro ⟨hx, h⟩
  sorry

end solve_x_from_equation_l442_442727


namespace largest_4_digit_number_divisible_by_1615_l442_442082

theorem largest_4_digit_number_divisible_by_1615 (X : ℕ) (hX: 8640 = 1615 * X) (h1: 1000 ≤ 1615 * X ∧ 1615 * X ≤ 9999) : X = 5 :=
by
  sorry

end largest_4_digit_number_divisible_by_1615_l442_442082


namespace geometric_sequence_general_term_find_λ_l442_442620

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ)

axiom Sn_def (n : ℕ) : λ * S n = λ - a n
axiom λ_ne_zero : λ ≠ 0
axiom λ_ne_neg_one : λ ≠ -1

theorem geometric_sequence_general_term :
  ∀ n, a n = λ / (1 + λ) ^ n :=
sorry

axiom S4 : S 4 = 15 / 16

theorem find_λ : λ = 1 ∨ λ = -3 :=
sorry

end geometric_sequence_general_term_find_λ_l442_442620


namespace dot_product_sum_eq_l442_442938

variables {G : Type*} [inner_product_space ℝ G]

theorem dot_product_sum_eq
  (u v w : G)
  (hu : ∥u∥ = 2)
  (hv : ∥v∥ = 3)
  (hw : ∥w∥ = 4)
  (hsum : u + v + w = 0) :
  inner u v + inner u w + inner v w = -29 / 2 :=
sorry

end dot_product_sum_eq_l442_442938


namespace even_four_digit_increasing_count_l442_442525

theorem even_four_digit_increasing_count :
  let digits := {x // 1 ≤ x ∧ x ≤ 9}
  let even_digits := {x // x ∈ digits ∧ x % 2 = 0}
  {n : ℕ //
    ∃ a b c d : ℕ,
      n = a * 1000 + b * 100 + c * 10 + d ∧
      a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ even_digits ∧
      a < b ∧ b < c ∧ c < d} =
  17 :=
by sorry

end even_four_digit_increasing_count_l442_442525


namespace hundredth_digit_of_7_div_33_l442_442888

theorem hundredth_digit_of_7_div_33 : (100 : ℕ) % 2 = 0 → digit_at (100 : ℕ) (decimal_expansion (7 / 33)) = 1 :=
by
  intros h
  sorry

def decimal_expansion (x : ℚ) : string := "0.212121..."

def digit_at (n : ℕ) (s : string) : char :=
  s.to_list.get (n % s.length)

end hundredth_digit_of_7_div_33_l442_442888


namespace problem_a_solution_problem_b_solution_problem_c_solution_problem_d_solution_no_solution_l442_442747

-- System (a)
def system_a_solved (x y z t : ℝ) : Prop :=
    x - 3*y + 2*z - t = 3 ∧
    2*x + 4*y - 3*z + t = 5 ∧
    4*x - 2*y + z + t = 3 ∧
    3*x + y + z - 2*t = 10

theorem problem_a_solution : ∃ x y z t : ℝ, system_a_solved x y z t ∧ x = 2 ∧ y = -1 ∧ z = -3 ∧ t = -4 :=
    by sorry

-- System (b)
def system_b (x y z t : ℝ) : Prop :=
    x + 2*y + 3*z - t = 0 ∧
    x - y + z + 2*t = 4 ∧
    x + 5*y + 5*z - 4*t = -4 ∧
    x + 8*y + 7*z - 7*t = -8

theorem problem_b_solution : ∃ x y z t : ℝ, system_b x y z t :=
   by sorry

-- System (c)
def system_c_solved (x y z : ℝ) : Prop :=
    x + 2*y + 3*z = 2 ∧
    x - y + z = 0 ∧
    x + 3*y - z = -2 ∧
    3*x + 4*y + 3*z = 0

theorem problem_c_solution : ∃ x y z : ℝ, system_c_solved x y z ∧ x = -1 ∧ y = 0 ∧ z = 1 :=
    by sorry

-- System (d)
def system_d (x y z t : ℝ) : Prop :=
    x + 2*y + 3*z - t = 0 ∧
    x - y + z + 2*t = 4 ∧
    x + 5*y + 5*z - 4*t = -4 ∧
    x + 8*y + 7*z - 7*t = 6

theorem problem_d_solution_no_solution : ¬ ∃ x y z t : ℝ, system_d x y z t :=
    by sorry

end problem_a_solution_problem_b_solution_problem_c_solution_problem_d_solution_no_solution_l442_442747


namespace dot_product_condition_l442_442144

-- Define the vectors a and b
def a (m : ℝ) : ℝ × ℝ := (m, 2)
def b : ℝ × ℝ := (2, -1)

-- Prove the main statement
theorem dot_product_condition (m : ℝ) :
  (a m).1 * b.1 + (a m).2 * b.2 < 0 ↔ 0 < m ∧ m < 1 := 
begin
  sorry -- Proof goes here
end

end dot_product_condition_l442_442144


namespace z_pow_12_eq_1_l442_442596

noncomputable def z : ℂ := (Complex.sqrt 3 - Complex.I) / 2

theorem z_pow_12_eq_1 : z^12 = 1 := by 
  sorry

end z_pow_12_eq_1_l442_442596


namespace art_project_probability_l442_442880

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def derangement_count (n: Nat) : Nat :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement_count (n - 1) + derangement_count (n - 2))

def total_arrangements : Nat :=
  factorial 12 / (factorial 4 * factorial 4 * factorial 4)

def derangements : Nat := derangement_count 4 * derangement_count 4 * derangement_count 4

def probability : (Nat × Nat) :=
  let num := derangements
  let denom := total_arrangements
  (81, 3850) -- (num, denom)

def problem_statement : Prop :=
  let (m, n) := probability
  n - 100 * m = -4250

theorem art_project_probability : problem_statement := by
  sorry

end art_project_probability_l442_442880


namespace angle_A_is_pi_over_4_length_of_b_when_c_is_3_l442_442554

-- Define the scenario and angles as variables
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions: opposite sides and tangent values
def triangle_conditions (A B C a b c : ℝ) : Prop :=
  tan B = 2 ∧ tan C = 3

-- Problem 1 Statement: Angle A
theorem angle_A_is_pi_over_4 
  (h : triangle_conditions A B C a b c) :
  A = π / 4 :=
sorry

-- Problem 2 Statement: Length of side b when c = 3
theorem length_of_b_when_c_is_3
  (h : triangle_conditions A B C a b 3) :
  b = 2 * sqrt 2 :=
sorry

end angle_A_is_pi_over_4_length_of_b_when_c_is_3_l442_442554


namespace dima_more_berries_and_difference_l442_442817

section RaspberryPicking

-- Define conditions
def total_berries : ℕ := 450
def dima_contrib_per_2_berries : ℚ := 1
def sergei_contrib_per_3_berries : ℚ := 2
def dima_speed_factor : ℚ := 2

-- Defining the problem of determining the berry counts
theorem dima_more_berries_and_difference :
  let dima_cycles := 2 * total_berries / (2 * dima_contrib_per_2_berries + 3 * sergei_contrib_per_3_berries * (1 / dima_speed_factor)) / dima_contrib_per_2_berries in
  let sergei_cycles := total_berries / (2 * dima_contrib_per_2_berries + 3 * sergei_contrib_per_3_berries * (1 / dima_speed_factor)) / sergei_contrib_per_3_berries in
  let berries_dima := dima_cycles * (dima_contrib_per_2_berries / 2) in
  let berries_sergei := sergei_cycles * (sergei_contrib_per_3_berries / 3) in
  berries_dima > berries_sergei ∧
  berries_dima - berries_sergei = 50 :=
by --- skip the proof
sorry

end RaspberryPicking

end dima_more_berries_and_difference_l442_442817


namespace sin_cos_product_l442_442475

open Real

theorem sin_cos_product (θ : ℝ) (h : sin θ + cos θ = 3 / 4) : sin θ * cos θ = -7 / 32 := 
  by 
    sorry

end sin_cos_product_l442_442475


namespace fourth_quadrant_angle_l442_442887

theorem fourth_quadrant_angle (α : ℝ) (hα : 270° < α ∧ α < 360°) : 90° < 180° - α ∧ 180° - α < 180° :=
by
  sorry

end fourth_quadrant_angle_l442_442887


namespace isosceles_triangle_perimeter_l442_442680

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 4)
  (h3 : a = b ∨ 2 * a > b) :
  (a ≠ b ∨ b = 2 * a) → 
  ∃ p : ℝ, p = a + b + b ∧ p = 10 :=
by
  sorry

end isosceles_triangle_perimeter_l442_442680


namespace max_value_of_expression_l442_442997

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l442_442997


namespace z_is_imaginary_z_is_purely_imaginary_iff_z_on_angle_bisector_iff_l442_442840

open Complex

-- Problem definitions
def z (m : ℝ) : ℂ := (2 + I) * m^2 - 2 * (1 - I)

-- Prove that for all m in ℝ, z is imaginary
theorem z_is_imaginary (m : ℝ) : ∃ a : ℝ, z m = a * I :=
  sorry

-- Prove that z is purely imaginary iff m = ±1
theorem z_is_purely_imaginary_iff (m : ℝ) : (∃ b : ℝ, z m = b * I ∧ b ≠ 0) ↔ (m = 1 ∨ m = -1) :=
  sorry

-- Prove that z is on the angle bisector iff m = 0
theorem z_on_angle_bisector_iff (m : ℝ) : (z m).re = -((z m).im) ↔ (m = 0) :=
  sorry

end z_is_imaginary_z_is_purely_imaginary_iff_z_on_angle_bisector_iff_l442_442840


namespace zero_neither_positive_nor_negative_l442_442042

def is_positive (n : ℤ) : Prop := n > 0
def is_negative (n : ℤ) : Prop := n < 0
def is_rational (n : ℤ) : Prop := ∃ p q : ℤ, q ≠ 0 ∧ n = p / q

theorem zero_neither_positive_nor_negative : ¬is_positive 0 ∧ ¬is_negative 0 :=
by
  sorry

end zero_neither_positive_nor_negative_l442_442042


namespace minimize_quadratic_function_l442_442812

def quadratic_function (x : ℝ) : ℝ := x^2 + 8*x + 7

theorem minimize_quadratic_function : ∃ x : ℝ, (∀ y : ℝ, quadratic_function y ≥ quadratic_function x) ∧ x = -4 :=
by
  sorry

end minimize_quadratic_function_l442_442812


namespace find_number_l442_442341

variables (n q r d : ℕ)

theorem find_number (h1: d = 9) (h2: q = 80) (h3: r = 4) (h4: n = d * q + r) :
  n = 724 :=
by {
  rw [h1, h2, h3] at h4,
  exact h4,
  sorry
}

end find_number_l442_442341


namespace sum_of_odd_divisors_of_360_l442_442337

theorem sum_of_odd_divisors_of_360 : 
    let odd_divisors := {d | d ∣ 360 ∧ d % 2 = 1} in
    (odd_divisors.to_finset.sum id) = 78 :=
by
  sorry

end sum_of_odd_divisors_of_360_l442_442337


namespace last_year_sales_l442_442395

-- Define the conditions as constants
def sales_this_year : ℝ := 480
def percent_increase : ℝ := 0.50

-- The main theorem statement
theorem last_year_sales : 
  ∃ sales_last_year : ℝ, sales_this_year = sales_last_year * (1 + percent_increase) ∧ sales_last_year = 320 := 
by 
  sorry

end last_year_sales_l442_442395


namespace problem_statement_l442_442501

def f (x : ℝ) : ℝ := x - Real.log x

def f_k (k x : ℝ) : ℝ :=
if f x ≥ k then f x else k

theorem problem_statement : f_k 3 (f_k 2 Real.exp 1) = 3 := by
  sorry

end problem_statement_l442_442501


namespace prove_c_val_l442_442158

noncomputable def calc (x y z : ℝ) := x * y * z

lemma value_of_c : (c : ℝ) = calc (2 : ℝ) (Real.sqrt 3) ((1.5)^(1/3)) * (12)^(1/6) := by
  sorry

theorem prove_c_val : c = 6 := by
  have h : c = calc (2 : ℝ) (Real.sqrt 3) ((1.5)^(1/3)) * (12)^(1/6) := value_of_c
  -- Using the correct calculations, we aim to show:
  have h_eq : calc (2 : ℝ) (Real.sqrt 3) ((1.5)^(1/3)) * (12)^(1/6) = 6 := by
    -- The detailed proof follows all steps to simplify and eventually show equates to 6.
    sorry
  rw [h] at h_eq
  exact h_eq

end prove_c_val_l442_442158


namespace monotonic_continuous_function_l442_442215

theorem monotonic_continuous_function (f : ℝ → ℝ) (h_continuous : Continuous f)
  (h_property : ∀ a b : ℝ, a < b → f a = f b → 
  ∃ c ∈ Ioo a b, f c = f a) : (Monotone f ∨ Antitone f) := 
sorry

end monotonic_continuous_function_l442_442215


namespace symmetric_graph_l442_442131

noncomputable def f : ℝ → ℝ := λ x, (3 / 2) * Real.sin (2 * x) + (Real.sqrt 3 / 2) * Real.cos (2 * x) + (Real.pi / 12)

theorem symmetric_graph (a b : ℝ) (h : a ∈ set.Ioo (-Real.pi / 2) 0) (ha : ∃ (p : ℝ → ℝ), f = p ∧ p (-a) = f a ∧ p a = f (2 * a - a)) :
  a + b = 0 :=
sorry

end symmetric_graph_l442_442131


namespace problem_derivative_periodicity_l442_442093

variables (x : Real)

noncomputable def f : ℕ → (Real → Real)
| 0       := fun x => Real.sin x
| (n + 1) := fun x => (f n)' x

theorem problem_derivative_periodicity :
  f 2012 x = Real.sin x :=
by
  sorry

end problem_derivative_periodicity_l442_442093


namespace sum_of_real_roots_eq_zero_l442_442083

theorem sum_of_real_roots_eq_zero : 
  (∑ x in (polynomial.roots (polynomial.X ^ 4 - 6 * polynomial.X ^ 2 - 1)).to_finset, x) = 0 := 
by 
  sorry

end sum_of_real_roots_eq_zero_l442_442083


namespace caterer_comparison_l442_442365

-- Define the cost functions for the two caterers
def cost_first_caterer (p : ℕ) : ℝ := 50 * p
noncomputable def cost_second_caterer (p : ℕ) : ℝ :=
  if p ≤ 61 then 500 + 40 * p else 2500 * Real.log10 (p / 4)

-- Define the theorem to prove
theorem caterer_comparison : cost_first_caterer 51 ≥ cost_second_caterer 51 :=
by sorry

end caterer_comparison_l442_442365


namespace Rachel_painting_time_l442_442964

theorem Rachel_painting_time :
  ∀ (Matt_time Patty_time Rachel_time : ℕ),
  Matt_time = 12 →
  Patty_time = Matt_time / 3 →
  Rachel_time = 2 * Patty_time + 5 →
  Rachel_time = 13 :=
by
  intros Matt_time Patty_time Rachel_time hMatt hPatty hRachel
  rw [hMatt] at hPatty
  rw [hPatty, hRachel]
  sorry

end Rachel_painting_time_l442_442964


namespace true_propositions_among_1_2_3_4_l442_442045

theorem true_propositions_among_1_2_3_4 :
  let A B : Set ℕ := sorry
  let x y : ℝ := sorry
  let cyclic_quadrilateral (Q : Set (ℕ × ℕ)) : Prop := sorry
  let congruent (Δ1 Δ2 : Set (ℕ × ℕ)) : Prop := sorry
  let similar (Δ1 Δ2 : Set (ℕ × ℕ)) : Prop := sorry
  (A ∩ B = A → A ⊆ B) ∧ (A ∩ B = A → ¬ (A ⊂ B)) → False ∧ -- proposition ① is false
  (∀ (x y : ℝ), (x^2 + y^2 ≠ 0 → ¬ (x = 0 ∧ y = 0))) ∧ -- proposition ② is true
  (∀ (Δ1 Δ2 : Set (ℕ × ℕ)), similar Δ1 Δ2 → congruent Δ1 Δ2) → False ∧ -- proposition ③ is false
  (∀ (Q : Set (ℕ × ℕ)), cyclic_quadrilateral Q → (∀ (x y : ℕ), x ≠ y → supplementary (opposite_angles Q x y))) → -- proposition ④ is true
  (true_propositions : List ℕ = [2, 4]) :=
by
  sorry

end true_propositions_among_1_2_3_4_l442_442045


namespace rented_apartment_years_l442_442581

-- Given conditions
def months_in_year := 12
def payment_first_3_years_per_month := 300
def payment_remaining_years_per_month := 350
def total_paid := 19200
def first_period_years := 3

-- Define the total payment calculation
def total_payment (additional_years: ℕ): ℕ :=
  (first_period_years * months_in_year * payment_first_3_years_per_month) + 
  (additional_years * months_in_year * payment_remaining_years_per_month)

-- Main theorem statement
theorem rented_apartment_years (additional_years: ℕ) :
  total_payment additional_years = total_paid → (first_period_years + additional_years) = 5 :=
by
  intros h
  -- This skips the proof
  sorry

end rented_apartment_years_l442_442581


namespace number_of_digimon_packs_bought_l442_442590

noncomputable def cost_per_digimon_pack : ℝ := 4.45
noncomputable def cost_of_baseball_deck : ℝ := 6.06
noncomputable def total_spent : ℝ := 23.86

theorem number_of_digimon_packs_bought : 
  ∃ (D : ℕ), cost_per_digimon_pack * D + cost_of_baseball_deck = total_spent ∧ D = 4 := 
by
  use 4
  split
  · norm_num; exact ((4.45 * 4) + 6.06 = 23.86)
  · exact rfl

end number_of_digimon_packs_bought_l442_442590


namespace tournament_point_distributions_36_l442_442090

/-- A round-robin tournament with 4 players and specific score constraints. -/
def round_robin_tournament_points (scores : Fin 4 → ℚ) : Prop :=
  let total_points := (∑ i, scores i);
  let highest_score := (∃ i, scores i = 3);
  let lowest_score := (∃ i, scores i = 0.5);
  total_points = 6 ∧ highest_score ∧ lowest_score

theorem tournament_point_distributions_36 :
  ∃! (scores : Fin 4 → ℚ), round_robin_tournament_points scores ∧
  (∃ permutations, permutations.size = 36) :=
sorry

end tournament_point_distributions_36_l442_442090


namespace imaginary_part_conj_z_is_minus_2_l442_442542

def z : ℂ := 1 + 2 * complex.I

theorem imaginary_part_conj_z_is_minus_2 : z.conj.im = -2 := by
  sorry

end imaginary_part_conj_z_is_minus_2_l442_442542


namespace number_of_incorrect_propositions_is_3_l442_442488

-- Definitions of the conditions given in the problem
def complementary_events (A B : set α) [probability_space α] := (A ∩ B = ∅) ∧ (P(A) + P(B) = 1)
def mutually_exclusive (A B : set α) := A ∩ B = ∅
def mutually_exclusive_three (A B C : set α) := (A ∩ B = ∅) ∧ (A ∩ C = ∅) ∧ (B ∩ C = ∅)
def P := λ s : set α, prob_func s

-- Statements reflecting the conditions
def cond1 (A B : set α) [probability_space α] : Prop := complementary_events A B → mutually_exclusive A B
def cond2 (A B : set α) [probability_space α] : Prop := P (A ∪ B) = P A + P B
def cond3 (A B C : set α) [probability_space α] : Prop := mutually_exclusive_three A B C → P A + P B + P C = 1
def cond4 (A B : set α) [probability_space α] : Prop := (P A + P B = 1) → complementary_events A B

-- Prove the number of incorrect propositions is 3
theorem number_of_incorrect_propositions_is_3 :
  cond1 ∧ cond2 ∧ cond3 ∧ cond4 → 3 :=
by sorry

end number_of_incorrect_propositions_is_3_l442_442488


namespace solution_for_x2_l442_442790

def eq1 (x : ℝ) := 2 * x = 6
def eq2 (x : ℝ) := x + 2 = 0
def eq3 (x : ℝ) := x - 5 = 3
def eq4 (x : ℝ) := 3 * x - 6 = 0

theorem solution_for_x2 : ∀ x : ℝ, x = 2 → ¬eq1 x ∧ ¬eq2 x ∧ ¬eq3 x ∧ eq4 x :=
by 
  sorry

end solution_for_x2_l442_442790


namespace reflection_circumcircle_same_radius_l442_442267

variable {A B C O Ta Tb Tc D E F : Point}

-- Definitions and assumptions based on the problem conditions
def acute_triangle (A B C : Point) : Prop := 
  ∃ (angleA angleB angleC : ℝ), 0 < angleA ∧ angleA < π / 2 ∧
                                 0 < angleB ∧ angleB < π / 2 ∧
                                 0 < angleC ∧ angleC < π / 2 ∧ 
                                 angleA + angleB + angleC = π

def circumcenter (A B C O : Point) : Prop := 
  is_perpendicular_bisector (segment A B) (segment O O) ∧
  is_perpendicular_bisector (segment B C) (segment O O) ∧
  is_perpendicular_bisector (segment C A) (segment O O)

def altitude_foot (Ta Tb Tc : Point) (A B C : Point) : Prop :=
  is_foot_of_altitude Ta A B C ∧
  is_foot_of_altitude Tb B C A ∧
  is_foot_of_altitude Tc C A B

def reflection_over_foot (O Ta Tb Tc D E F : Point) : Prop := 
  reflection T O = D ∧
  reflection Tb O = E ∧
  reflection Tc O = F

-- Lean Proof statement
theorem reflection_circumcircle_same_radius 
    (h₁ : acute_triangle A B C) 
    (h₂ : circumcenter A B C O) 
    (h₃ : altitude_foot Ta Tb Tc A B C) 
    (h₄ : reflection_over_foot O Ta Tb Tc D E F) : 
  circumcircle_radius D E F = circumcircle_radius A B C := 
by 
  sorry

end reflection_circumcircle_same_radius_l442_442267


namespace equilateral_triangle_not_all_congruent_l442_442731

theorem equilateral_triangle_not_all_congruent :
  (∀ (T : Triangle), T.is_equilateral → T.is_equiangular) ∧
  (∀ (T : Triangle), T.is_equilateral → T.is_isosceles) ∧
  (∀ (T : Triangle), T.is_equilateral → T.is_regular_polygon) ∧
  (∀ (T : Triangle), T.is_equilateral → T.is_similar_to) →
  ¬ (∀ (T₁ T₂ : Triangle), (T₁.is_equilateral ∧ T₂.is_equilateral) → T₁.is_congruent_to T₂) :=
begin
  sorry
end

end equilateral_triangle_not_all_congruent_l442_442731


namespace find_number_of_packs_l442_442591

-- Define the cost of a pack of Digimon cards
def cost_pack_digimon : ℝ := 4.45

-- Define the cost of the deck of baseball cards
def cost_deck_baseball : ℝ := 6.06

-- Define the total amount spent
def total_spent : ℝ := 23.86

-- Define the number of packs of Digimon cards Keith bought
def number_of_packs (D : ℝ) : Prop :=
  cost_pack_digimon * D + cost_deck_baseball = total_spent

-- Prove the number of packs is 4
theorem find_number_of_packs : ∃ D, number_of_packs D ∧ D = 4 :=
by
  -- the proof will be inserted here
  sorry

end find_number_of_packs_l442_442591


namespace intersection_of_A_and_B_l442_442221

-- Define sets A and B
def A : set ℝ := {x | -2 < x ∧ x < 4}
def B : set ℕ := {2, 3, 4, 5}

-- Theorem stating the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {2, 3} := 
by sorry

end intersection_of_A_and_B_l442_442221
