import Mathlib
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Quotients
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.List
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Div
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Functions
import Mathlib.Geometry.Circle
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Line
import Mathlib.Geometry.Triangle
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Real

namespace smallest_number_with_prime_factors_of_24_l100_100933

theorem smallest_number_with_prime_factors_of_24 : 
  ∃ b : ℕ, (∀ p : ℕ, p.prime → p ∣ 24 → p ∣ b) ∧ b = 6 :=
by
  sorry

end smallest_number_with_prime_factors_of_24_l100_100933


namespace tan_45_eq_1_l100_100005

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100005


namespace tan_of_45_deg_l100_100090

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100090


namespace tan_45_deg_eq_one_l100_100151

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100151


namespace smallest_diameter_of_tablecloth_l100_100531

theorem smallest_diameter_of_tablecloth (a : ℝ) (h : a = 1) : ∃ d : ℝ, d = Real.sqrt 2 ∧ (∀ (x : ℝ), x < d → ¬(∀ (y : ℝ), (y^2 + y^2 = x^2) → y ≤ a)) :=
by 
  sorry

end smallest_diameter_of_tablecloth_l100_100531


namespace compare_AC_CD_CB_l100_100672

variable {x y : ℝ}

-- Define the equation of the ellipse
def on_ellipse (A : ℝ × ℝ) : Prop :=
  let (m, n) := A in (m^2 / 8 + n^2 / 2 = 1 ∧ 0 < m ∧ m < 2 * real.sqrt 2 ∧ 0 < n)

-- Define the coordinates of the vertex C
def C : ℝ × ℝ := (2 * real.sqrt 2, 0)

-- Define the reflection point B
def B (A : ℝ × ℝ) : ℝ × ℝ :=
  let (m, n) := A in (-m, -n)

-- Define the condition for point D lying on the perpendicular from A to x-axis, intersecting BC
def D (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, ((A.1 - 2 * real.sqrt 2) * A.2) / (A.1 + 2 * real.sqrt 2))

-- Define the distance squared between two points
def dist2 (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Statement of the main theorem
theorem compare_AC_CD_CB (A : ℝ × ℝ) :
  on_ellipse A →
  dist2 A C < dist2 (D A) C * dist2 (D A) (B A) :=
sorry

end compare_AC_CD_CB_l100_100672


namespace problem_statement_l100_100715

theorem problem_statement (x : ℝ) (h : √(10 + x) + √(30 - x) = 8) : 
  (10 + x) * (30 - x) = 144 := 
  sorry

end problem_statement_l100_100715


namespace tan_45_eq_1_l100_100012

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100012


namespace ratio_wheelbarrow_to_earnings_l100_100913

theorem ratio_wheelbarrow_to_earnings :
  let duck_price := 10
  let chicken_price := 8
  let chickens_sold := 5
  let ducks_sold := 2
  let resale_earn := 60
  let total_earnings := chickens_sold * chicken_price + ducks_sold * duck_price
  let wheelbarrow_cost := resale_earn / 2
  (wheelbarrow_cost / total_earnings = 1 / 2) :=
by
  sorry

end ratio_wheelbarrow_to_earnings_l100_100913


namespace tan_45_deg_eq_one_l100_100166

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100166


namespace parabola_hyperbola_focus_shared_hyperbola_asymptotes_l100_100292

-- Definitions and conditions
def parabola_focus (a : ℝ) := (-a, 0)
def hyperbola_foci (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c = real.sqrt (a^2 + b^2)) := (c, 0)

-- Proving that a = √3
theorem parabola_hyperbola_focus_shared {a b c : ℝ} 
    (h_parabola : parabola_focus 2)
    (h_hyperbola : hyperbola_foci (real.sqrt 3) 1 2 ∧ c = 2 ∧ c^2 = 4) :
    a = real.sqrt 3 :=
sorry

-- Proving the equations of the asymptotes
theorem hyperbola_asymptotes {a b : ℝ} 
    (a_sqrt3 : a = real.sqrt 3) 
    (b1 : b = 1) :
    ∀ x y, y = (1 / real.sqrt 3) * x ∨ y = -(1 / real.sqrt 3) * x :=
sorry

end parabola_hyperbola_focus_shared_hyperbola_asymptotes_l100_100292


namespace largest_set_of_good_cubes_l100_100650

def cube_set_size (n : ℕ) : ℕ :=
  (n - 1) ^ 3 + (n - 2) ^ 3

theorem largest_set_of_good_cubes (n : ℕ) (h : 2 ≤ n) :
  ∃ S : finset (fin 3 → fin n), 
    (∀ (s₁ s₂ : fin 3 → fin n), s₁ ∈ S ∧ s₂ ∈ S → ¬ properly_contains s₁ s₂ ∧ ¬ properly_contains s₂ s₁) ∧ 
    (S.card = cube_set_size n) := 
sorry

-- Additional definitions that might be necessary

-- Properly contains definition
def properly_contains (s₁ s₂ : fin 3 → fin n) : Prop :=
  ∀ i, s₁ i < s₂ i - 1 ∧ s₂ i > s₁ i + 1

end largest_set_of_good_cubes_l100_100650


namespace length_of_MN_l100_100333

variable (X Y Z A M N : Type)
variable [Hilbert X Y Z A M N]

-- Conditions
variable (hYZ : distance Y Z = 28)
variable (hXA : distance X A = 24)
variable (hYA : distance Y A = 10)
variable (hPerpendicular : is_perpendicular (line_through X A) (line_through Y Z))
variable (hMIncenter : is_incenter M (triangle X Y A))
variable (hNIncenter : is_incenter N (triangle X Z A))

-- Proof Statement
theorem length_of_MN : distance M N = 2 * sqrt 26 :=
sorry

end length_of_MN_l100_100333


namespace sequence_square_sum_l100_100208

variable (a : ℕ → ℕ)

-- Defining the condition for the sum of the sequence
def sequence_sum (n : ℕ) : Prop :=
  ∑ i in Finset.range n, a (i + 1) = 2^n - 1

-- Main theorem statement
theorem sequence_square_sum (n : ℕ) (h : 0 < n) (h_sum : sequence_sum a n) :
  ∑ i in Finset.range n, (a (i + 1))^2 = (1/3) * (4^n - 1) :=
by
  sorry

end sequence_square_sum_l100_100208


namespace smallest_integer_mod_conditions_l100_100494

theorem smallest_integer_mod_conditions : 
  ∃ x : ℕ, 
  (x % 4 = 3) ∧ (x % 3 = 2) ∧ (∀ y : ℕ, (y % 4 = 3) ∧ (y % 3 = 2) → x ≤ y) ∧ x = 11 :=
by
  sorry

end smallest_integer_mod_conditions_l100_100494


namespace tan_45_deg_l100_100137

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100137


namespace functional_equation_solution_l100_100300

theorem functional_equation_solution (f : ℝ → ℝ)
    (h1 : ∀ x y : ℝ, f(x + y) = f(x) * f(y))
    (h2 : f(2) = 4) :
    f(8) = 256 := by
  sorry

end functional_equation_solution_l100_100300


namespace total_cost_mulch_l100_100512

-- Define the conditions
def tons_to_pounds (tons : ℕ) : ℕ := tons * 2000

def price_per_pound : ℝ := 2.5

-- Define the statement to prove
theorem total_cost_mulch (mulch_in_tons : ℕ) (h₁ : mulch_in_tons = 3) : 
  tons_to_pounds mulch_in_tons * price_per_pound = 15000 :=
by
  -- The proof would normally go here.
  sorry

end total_cost_mulch_l100_100512


namespace solution_of_binary_linear_equation_l100_100403

theorem solution_of_binary_linear_equation :
  ∃ (x y : ℤ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 :=
by
  use 2
  use 2
  split
  . calc
    2 + 2 * 2 = 2 + 4 := by rfl
    _ = 6         := by rfl
  . rfl
  . rfl

end solution_of_binary_linear_equation_l100_100403


namespace find_y_l100_100276

-- Define the points and slope conditions
def point_R : ℝ × ℝ := (-3, 4)
def x2 : ℝ := 5

-- Define the y coordinate and its corresponding condition
def y_condition (y : ℝ) : Prop := (y - 4) / (5 - (-3)) = 1 / 2

-- The main theorem stating the conditions and conclusion
theorem find_y (y : ℝ) (h : y_condition y) : y = 8 :=
by
  sorry

end find_y_l100_100276


namespace tan_of_45_deg_l100_100071

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100071


namespace range_of_a_l100_100654

variable (a x y : ℝ)

def proposition_p : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def proposition_q : Prop :=
  (1 - a) * (a - 3) < 0

theorem range_of_a (h1 : proposition_p a) (h2 : proposition_q a) : 
  (0 ≤ a ∧ a < 1) ∨ (3 < a ∧ a < 4) :=
by
  sorry

end range_of_a_l100_100654


namespace jan_more_miles_than_ian_l100_100885

noncomputable def distance_diff (d t s : ℝ) : ℝ :=
  let han_distance := (s + 10) * (t + 2)
  let jan_distance := (s + 15) * (t + 3)
  jan_distance - (d + 100)

theorem jan_more_miles_than_ian {d t s : ℝ} (H : d = s * t) (H_han : d + 100 = (s + 10) * (t + 2)) : distance_diff d t s = 165 :=
by {
  sorry
}

end jan_more_miles_than_ian_l100_100885


namespace solve_ZAMENA_l100_100973

noncomputable def ZAMENA : ℕ := 541234

theorem solve_ZAMENA :
  ∃ (A M E H : ℕ), 
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H ∧
    ZAMENA = 541234 :=
by {
  have h1 : ∃ (A M E H : ℕ), 
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H, sorry,
  exact ⟨2, 1, 2, 3, h1⟩,
  repeat { sorry }
}

end solve_ZAMENA_l100_100973


namespace tan_five_pi_over_four_l100_100586

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
sorry

end tan_five_pi_over_four_l100_100586


namespace median_eq_altitude_eq_l100_100668

noncomputable def median_to_AC_line_eq (A B C : ℝ × ℝ) : Prop :=
  let D := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  D = (1, 4) ∧ ∃ m b, ∀ x y, y = m * x + b ↔ 2 * x + y - 6 = 0

noncomputable def altitude_to_AB_line_eq (A B C : ℝ × ℝ) : Prop :=
  ∃ m1 m2 b, m1 = (B.2 - A.2) / (B.1 - A.1) ∧ m2 = -1 / m1 ∧ 
  ∀ x y, y - C.2 = m2 * (x - C.1) ↔ 4 * x + 7 * y + 3 = 0

-- Assume the vertices of the triangle ABC
def A : ℝ × ℝ := (8, 5)
def B : ℝ × ℝ := (4, -2)
def C : ℝ × ℝ := (-6, 3)

-- Proof statements
theorem median_eq : median_to_AC_line_eq A B C := 
  sorry

theorem altitude_eq : altitude_to_AB_line_eq A B C := 
  sorry

end median_eq_altitude_eq_l100_100668


namespace common_sum_l100_100444

theorem common_sum (a l : ℤ) (n r c : ℕ) (S x : ℤ) 
  (h_a : a = -18) 
  (h_l : l = 30) 
  (h_n : n = 49) 
  (h_S : S = (n * (a + l)) / 2) 
  (h_r : r = 7) 
  (h_c : c = 7) 
  (h_sum_eq : r * x = S) :
  x = 42 := 
sorry

end common_sum_l100_100444


namespace count_symmetric_points_l100_100811

theorem count_symmetric_points (M G T : Point)
  (h1 : M ≠ G) (h2 : G ≠ T) (h3 : M ≠ T)
  (h_no_symmetry : ¬(∃ l, IsAxisOfSymmetry ⟨M, G, T⟩ l)) :
  ∃ n : ℕ, (n = 5 ∨ n = 6) ∧ ∀ U : Point, (HasSymmetricFigure ⟨M, G, T, U⟩ → U ∈ Set.PointsWithSymmetry ⟨M, G, T⟩) :=
begin
  sorry
end

end count_symmetric_points_l100_100811


namespace largest_reciprocal_l100_100879

-- Definitions for the given numbers
def a := 1/4
def b := 3/7
def c := 2
def d := 10
def e := 2023

-- Statement to prove the problem
theorem largest_reciprocal :
  (1/a) > (1/b) ∧ (1/a) > (1/c) ∧ (1/a) > (1/d) ∧ (1/a) > (1/e) :=
by
  sorry

end largest_reciprocal_l100_100879


namespace tan_45_eq_1_l100_100202

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100202


namespace tan_45_deg_l100_100044

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100044


namespace tangent_line_at_1_3_l100_100728

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then real.log (-x) + 3 * x else -real.log x + 3 * x

def isOdd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

theorem tangent_line_at_1_3 :
  isOdd f →
  (∀ x < 0, f x = real.log (-x) + 3 * x) →
  (∀ x > 0, f x = -real.log x + 3 * x) →
  ∃ m b : ℝ, (m = 2 ∧ b = 1 ∧ ∀ x: ℝ, f' x = -1 / x + 3 ∧ 
  y - 3 = m * (x - 1)) :=
by
  intro h₀ h₁ h₂
  use 2, 1
  split
  sorry -- Skip the actual proof.

end tangent_line_at_1_3_l100_100728


namespace find_a_plus_b_l100_100322

theorem find_a_plus_b (a b : ℝ) (h : |a - 2| + |b + 3| = 0) : a + b = -1 :=
by {
  sorry
}

end find_a_plus_b_l100_100322


namespace retail_price_l100_100928

theorem retail_price (W M : ℝ) (hW : W = 20) (hM : M = 80) : W + (M / 100) * W = 36 := by
  sorry

end retail_price_l100_100928


namespace problem1_problem2_l100_100336

-- Definitions for curves and conditions
def C1 (θ : ℝ) : ℝ := 4 * Real.sin θ

def C2 (m α : ℝ) (t : ℝ) : (ℝ × ℝ) := (m + t * Real.cos α, t * Real.sin α)

-- First proof statement: Proving the relationship between OB, OC, and OA
theorem problem1 (φ : ℝ) : 
  let OA := 4 * Real.sin φ
  let OB := 4 * Real.sin (φ + π / 4)
  let OC := 4 * Real.sin (φ - π / 4)
  |OB| + |OC| = Real.sqrt 2 * |OA| := sorry

-- Second proof statement: Finding m and α when φ = 5π / 12 and points B and C are on C2
theorem problem2 (t : ℝ) (m α : ℝ) 
  (h1 : C2 m α t = (-Real.sqrt 3, 3))
  (h2 : C2 m α t = (Real.sqrt 3, 1)) :
  m = 2 * Real.sqrt 3 ∧ α = 5 * π / 6 := sorry

end problem1_problem2_l100_100336


namespace number_of_digits_of_9975_l100_100870

theorem number_of_digits_of_9975 : ∃ d : ℕ, d = 4 ∧ ∃ n : ℕ, n = 9975 ∧ (n % 35 = 0) ∧ (n.toNatDigits.length = d) :=
by
  sorry

end number_of_digits_of_9975_l100_100870


namespace forum_members_l100_100923

theorem forum_members (M : ℕ)
  (h1 : ∀ q a, a = 3 * q)
  (h2 : ∀ h d, q = 3 * h * d)
  (h3 : 24 * (M * 3 * (24 + 3 * 72)) = 57600) : M = 200 :=
by
  sorry

end forum_members_l100_100923


namespace hawks_score_l100_100334

theorem hawks_score (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 18) : y = 16 := by
  sorry

end hawks_score_l100_100334


namespace tan_45_deg_l100_100124

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100124


namespace inequality_proof_l100_100660

theorem inequality_proof {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hsum : x + y + z = 1) :
    (2 * x^2 / (y + z)) + (2 * y^2 / (z + x)) + (2 * z^2 / (x + y)) ≥ 1 := sorry

end inequality_proof_l100_100660


namespace parabola_focal_param_l100_100834

variable {p : ℝ} (h₀ : p > 0)
variable (F : ℝ × ℝ) (h₁ : F = (p / 2, 0))
variable (N : ℝ) -- N is a point on the x-axis
variable {A B : ℝ × ℝ} -- Points A and B
variable (line_eq : ∃ k : ℝ, ∀ x : ℝ, F.2 + k * (x - F.1) = (λ y, y^2 = 2 * p * x)) -- Equation of the line passing through F
variable (h₂ : (N, 0) = (λ x, x-axis intersects directrix))

variable dot_product_cond : (\overrightarrow{NB} \cdot \overrightarrow{AB} = 0)
variable dist_cond : (|\overrightarrow{AF}| - |\overrightarrow{BF}| = 4)

theorem parabola_focal_param {p : ℝ} (h₀ : p > 0)
  (F : ℝ × ℝ) (h₁ : F = (p / 2, 0))
  (N : ℝ × ℝ)
  {A B : ℝ × ℝ}
  (line_eq : ∃ k : ℝ, ∀ x : ℝ, F.2 + k * (x - F.1) = (λ y, y^2 = 2 * p * x))
  (dot_product_cond : (\overrightarrow{NB} \cdot \overrightarrow{AB} = 0))
  (dist_cond : (|\overrightarrow{AF}| - |\overrightarrow{BF}| = 4)) : p = 2 := 
sorry

end parabola_focal_param_l100_100834


namespace symmetric_point_in_fourth_quadrant_l100_100320

theorem symmetric_point_in_fourth_quadrant (a : ℝ) (h : a < 0) :
  let P := (-a^2 - 1, -a + 3),
      P1 := (a^2 + 1, a - 3) in
  P1.1 > 0 ∧ P1.2 < 0 :=
by sorry

end symmetric_point_in_fourth_quadrant_l100_100320


namespace second_trial_addition_amount_l100_100763

variable (optimal_min optimal_max: ℝ) (phi: ℝ)

def method_618 (optimal_min optimal_max phi: ℝ) :=
  let x1 := optimal_min + (optimal_max - optimal_min) * phi
  let x2 := optimal_max + optimal_min - x1
  x2

theorem second_trial_addition_amount:
  optimal_min = 10 ∧ optimal_max = 110 ∧ phi = 0.618 →
  method_618 10 110 0.618 = 48.2 :=
by
  intro h
  simp [method_618, h]
  sorry

end second_trial_addition_amount_l100_100763


namespace area_AEFGCB_theorem_l100_100818

noncomputable def area_AEFGCB (ABCD_square: ∀ (A B C D: ℝ × ℝ), (∃ s : ℝ, 
  A = (0, s) ∧ 
  B = (0, 0) ∧ 
  C = (s, 0) ∧ 
  D = (s, s)) 
  ∧ segment_perpendicular (A: ℝ × ℝ) (E: ℝ × ℝ) (D: ℝ × ℝ)
  ∧ rectangle_EFGH (E: ℝ × ℝ) (F: ℝ × ℝ) (G: ℝ × ℝ) (H: ℝ × ℝ) (EF: ℝ) (FG: ℝ) (G_on_DC: G.1 = D.1)
  ) : ℝ := 76

theorem area_AEFGCB_theorem : 
  ∀ (A B C D E F G H : ℝ × ℝ), 
  (∃ s : ℝ, 
  A = (0, s) ∧ 
  B = (0, 0) ∧ 
  C = (s, 0) ∧ 
  D = (s, s)) 
  ∧ segment_perpendicular (A, E, D)
  ∧ rectangle_EFGH (E, F, 4, 10, G, H)
  → area_AEFGCB (A, B, C, D, E, F, G, H) = 76 :=
λ A B C D E F G H h, sorry

end area_AEFGCB_theorem_l100_100818


namespace max_distance_from_point_to_line_l100_100843

def line (λ : ℝ) : ℝ × ℝ → Prop :=
  fun p => (1 + 3 * λ) * p.1 + (1 + λ) * p.2 - 2 - 4 * λ = 0

def point := (-2 : ℝ, -1 : ℝ)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem max_distance_from_point_to_line : 
  ∀ (λ : ℝ), 
  ∃ A : ℝ × ℝ, 
  (∀ P : ℝ × ℝ, line λ A) ∧ 
  (distance point A = real.sqrt 13) := 
sorry

end max_distance_from_point_to_line_l100_100843


namespace find_max_value_l100_100623

noncomputable def f (x : ℝ) : ℝ := (x * (1 - x)) / ((x + 1) * (x + 2) * (2 * x + 1))

theorem find_max_value :
  ∃ x ∈ set.Ioc 0 1, ∀ y ∈ set.Ioc 0 1, f(y) ≤ f(x) ∧ f(x) = (8 * real.sqrt 2 - 5 * real.sqrt 5) / 3 :=
sorry

end find_max_value_l100_100623


namespace trapezoid_integer_base_pairs_l100_100436

theorem trapezoid_integer_base_pairs :
  let h := 60
  let A := 1800
  let condition := ∀ (b1 b2: ℕ), (h * (b1 + b2)) / 2 = A → b1 % 10 = 0 ∧ b2 % 10 = 0
  condition → ∃ b1 b2 : ℕ, (h * (b1 + b2)) / 2 = A ∧ b1 % 10 = 0 ∧ b2 % 10 = 0 ∧
                             {b1, b2}.card = 4 :=
begin
  intros h A condition,
  use [0, 60, 10, 50, 20, 40, 30, 30],
  split,
  { apply condition,
    norm_num },
  { repeat { split; norm_num } }
end

end trapezoid_integer_base_pairs_l100_100436


namespace sufficient_but_not_necessary_condition_not_necessary_condition_l100_100258

theorem sufficient_but_not_necessary_condition 
  (a b : ℝ) : (b < a ∧ a < 0) → (1 / b > 1 / a) :=
begin
  sorry
end

theorem not_necessary_condition 
  (a b : ℝ) : (1 / b > 1 / a) → ¬ (b < a ∧ a < 0) :=
begin
  sorry
end

end sufficient_but_not_necessary_condition_not_necessary_condition_l100_100258


namespace tan_of_45_deg_l100_100086

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100086


namespace tan_five_pi_over_four_l100_100584

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
sorry

end tan_five_pi_over_four_l100_100584


namespace six_people_round_table_l100_100740

def seatingArrangements (n : ℕ) : ℕ :=
  if n > 0 then (n - 1)! else 0

theorem six_people_round_table : seatingArrangements 6 = 120 := by
  sorry

end six_people_round_table_l100_100740


namespace find_greatest_integer_lt_M_over_50_l100_100277

theorem find_greatest_integer_lt_M_over_50
  (M : ℕ)
  (h : (1 / (Nat.factorial 3 * Nat.factorial 16) + 
        1 / (Nat.factorial 4 * Nat.factorial 15) + 
        1 / (Nat.factorial 5 * Nat.factorial 14) + 
        1 / (Nat.factorial 6 * Nat.factorial 13) = 
        M / (Nat.factorial 1 * Nat.factorial 18))) :
  (⌊ M / 50 ⌋) = 275 :=
sorry

end find_greatest_integer_lt_M_over_50_l100_100277


namespace tan_45_deg_l100_100048

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100048


namespace tan_45_deg_eq_one_l100_100033

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100033


namespace intersection_l100_100481

noncomputable def intersection_point (L₁ L₂ : ℝ → ℝ → Prop) (p : ℝ × ℝ) :=
  ∃ x y, L₁ x y ∧ L₂ x y ∧ (x, y) = p

def line1 := λ x y : ℝ, y = 3 * x + 4

def line2 := λ x y : ℝ, y = -1/3 * x + (2 + (1 / 3) * 3)

theorem intersection : intersection_point line1 line2 (-3/10, 3.1) := 
  sorry

end intersection_l100_100481


namespace tan_45_eq_1_l100_100186

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100186


namespace complex_quadrant_l100_100323

theorem complex_quadrant (θ : ℝ) (hθ : θ ∈ Set.Ioo (3/4 * Real.pi) (5/4 * Real.pi)) :
  let z := Complex.mk (Real.cos θ + Real.sin θ) (Real.sin θ - Real.cos θ)
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end complex_quadrant_l100_100323


namespace sequence_decreasing_bound_sum_l100_100801

noncomputable def a : ℕ → ℝ 
| 0 => 0       -- this value is unused, as we start at n=1
| (n + 1) => 
  if h : n = 0 then
    if a 1 > 0 then a 1 else 1  -- minimal valid starter value
  else
    a n + (n / a n)

theorem sequence_decreasing (a : ℕ → ℝ) 
  (h_pos : ∀ n, a (n+1) = a n + (n / a n)) 
  (h_init : a 1 > 0) :
  ∀ n ≥ 2, a n - n > a (n+1) - (n+1) :=
by
  sorry

theorem bound_sum (a : ℕ → ℝ) 
  (h_pos : ∀ n, a (n+1) = a n + (n / a n)) 
  (h_init : a 1 > 0) :
  ∃ c : ℝ, ∀ n ≥ 2, (∑ k in finset.range n, (a (k+1) - (k+1)) / (k+2)) ≤ c :=
by
  sorry

end sequence_decreasing_bound_sum_l100_100801


namespace tan_45_eq_1_l100_100003

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100003


namespace sum_of_ages_is_220_l100_100503

-- Definitions based on the conditions
def father_age (S : ℕ) := (7 * S) / 4
def sum_ages (F S : ℕ) := F + S

-- The proof statement
theorem sum_of_ages_is_220 (F S : ℕ) (h1 : 4 * F = 7 * S)
  (h2 : 3 * (F + 10) = 5 * (S + 10)) : sum_ages F S = 220 :=
by
  sorry

end sum_of_ages_is_220_l100_100503


namespace tan_45_deg_eq_one_l100_100110

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100110


namespace saved_fraction_l100_100931

-- Given Definitions
variables {P : ℝ} -- The monthly take-home pay (P should be a real number)
variables {f : ℝ} -- The fraction of her take-home pay saved each month

-- Conditions
def monthly_save := f * P
def monthly_not_save := (1 - f) * P
def yearly_save := 12 * monthly_save
def condition : Prop := yearly_save = 6 * monthly_not_save

-- Theorem: The worker saved one-third of her take-home pay each month
theorem saved_fraction (h : condition) : f = 1 / 3 :=
sorry

end saved_fraction_l100_100931


namespace quadrilateral_inscribable_l100_100781

-- Define the circle, chord, and relevant points
variables (Γ : Type) [circle Γ]
variables (BC : chord Γ)
variables (A : midpoint_of_arc BC)
variables (D E : point Γ)
variables (F G : set_of_intersections A -> BC)

-- Define the inscribability of the quadrilateral DFGE
theorem quadrilateral_inscribable :
  is_chord A D ∧ is_chord A E ∧ intersects F G BC A D E →
  inscribable_in_circle (quadrilateral D F G E) :=
sorry

end quadrilateral_inscribable_l100_100781


namespace derivative_at_two_l100_100799

theorem derivative_at_two {f : ℝ → ℝ} (f_deriv : ∀x, deriv f x = 2 * x - 4) : deriv f 2 = 0 := 
by sorry

end derivative_at_two_l100_100799


namespace symmetric_point_x_axis_l100_100353

theorem symmetric_point_x_axis (x y z : ℝ) : 
    (x, -y, -z) = (-2, -1, -9) :=
by 
  sorry

end symmetric_point_x_axis_l100_100353


namespace championship_outcomes_l100_100858

theorem championship_outcomes (students events : ℕ) (h_students : students = 3) (h_events : events = 2) : 
  students ^ events = 9 :=
by
  rw [h_students, h_events]
  have h : 3 ^ 2 = 9 := by norm_num
  exact h

end championship_outcomes_l100_100858


namespace tan_45_eq_1_l100_100013

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100013


namespace sqrt_sum_eq_8_l100_100712

theorem sqrt_sum_eq_8 (x : ℝ) (h : √(10 + x) + √(30 - x) = 8) : (10 + x) * (30 - x) = 144 :=
sorry

end sqrt_sum_eq_8_l100_100712


namespace tan_45_deg_eq_one_l100_100098

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100098


namespace element_correspondence_l100_100349

-- Define the sets A and B
def A := {p : ℝ × ℝ | true}
def B := {p : ℝ × ℝ | true}

-- Define the mapping f
def f : ℝ × ℝ → ℝ × ℝ := λ xy, (xy.1 - xy.2, xy.1 + xy.2)

-- The theorem to prove that f maps (-1, 2) to (-3, 1)
theorem element_correspondence :
  f (-1, 2) = (-3, 1) :=
sorry

end element_correspondence_l100_100349


namespace last_four_digits_of_5_pow_2017_l100_100808

theorem last_four_digits_of_5_pow_2017 : (5 ^ 2017) % 10000 = 3125 :=
by sorry

end last_four_digits_of_5_pow_2017_l100_100808


namespace tan_45_deg_eq_one_l100_100109

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100109


namespace general_term_arithmetic_sequence_l100_100746

variable (a : ℕ → ℝ) (d : ℝ)

/-- An arithmetic sequence -/
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a (n + 1) = a n + d

theorem general_term_arithmetic_sequence
  (h3 : a 3 + a 4 = 4) 
  (h5 : a 5 + a 7 = 6) 
  (arith_seq : arithmetic_seq a d) :
  a = (fun n => (2 * n + 3) / 5) :=
by
  sorry

end general_term_arithmetic_sequence_l100_100746


namespace alicia_tax_and_earnings_l100_100938

noncomputable def alicia_hourly_wage : ℝ := 2500 -- 25 dollars in cents
noncomputable def tax_rate : ℝ := 0.025 -- 2.5%

theorem alicia_tax_and_earnings :
  let tax_deduction := alicia_hourly_wage * tax_rate in
  let after_tax_earnings := alicia_hourly_wage - tax_deduction in
  tax_deduction = 62.5 ∧ after_tax_earnings = 2437.5 :=
by
  sorry

end alicia_tax_and_earnings_l100_100938


namespace matchsticks_total_l100_100580

theorem matchsticks_total
  (boxes : ℕ) (matchboxes_per_box : ℕ) (sticks_per_matchbox : ℕ)
  (h1 : boxes = 4)
  (h2 : matchboxes_per_box = 20)
  (h3 : sticks_per_matchbox = 300) :
  boxes * matchboxes_per_box * sticks_per_matchbox = 24000 :=
by
  rw [h1, h2, h3]
  norm_num
  rfl

end matchsticks_total_l100_100580


namespace max_sum_first_n_terms_l100_100838

def a_n (n : ℕ) : ℤ := -7 * n + 30

def sum_first_n_terms (n : ℕ) : ℤ :=
  (finset.range n).sum (λ k, a_n (k + 1))

theorem max_sum_first_n_terms : 
  ∃ n : ℕ, sum_first_n_terms n = sum_first_n_terms 4 := sorry

end max_sum_first_n_terms_l100_100838


namespace tan_sum_pi_over_4_l100_100696

open Real

theorem tan_sum_pi_over_4 {α : ℝ} (h₁ : cos (2 * α) + sin α * (2 * sin α - 1) = 2 / 5) (h₂ : π / 4 < α) (h₃ : α < π) : 
    tan (α + π / 4) = 1 / 7 := sorry

end tan_sum_pi_over_4_l100_100696


namespace product_of_x_coords_of_intersections_l100_100537

theorem product_of_x_coords_of_intersections (k : ℝ) (b1 b2 b3 b4 b5 : ℝ) (hk : k ≠ 0) :
  let lines := [b1, b2, b3, b4, b5],
      intersections := lines.map (λ b, (λ x, k*x + b) = (λ x, k/x)) in
  let products := intersections.map (λ eq, let a := k, b := b, c := -k in -c / a) in
  products.prod = (-1) :=
by
  sorry

end product_of_x_coords_of_intersections_l100_100537


namespace unique_square_with_hypotenuse_vertices_l100_100780

variable {A B C : Type}
variable [EuclideanGeometry.AffineSpace ℝ A]
variable [EuclideanGeometry.AffineSpace ℝ B]
variable [EuclideanGeometry.AffineSpace ℝ C]

-- Definitions related to the conditions
def isRightTriangle (A B C : Type) [EuclideanGeometry.AffineSpace ℝ A] [EuclideanGeometry.AffineSpace ℝ B] [EuclideanGeometry.AffineSpace ℝ C] : Prop :=
  ∃ (o : A), EuclideanGeometry.orthogonal (A -ᵥ o) (C -ᵥ o)

def isHypotenuse (AB : Type) : Prop :=
  ∀ (A B : Type) [EuclideanGeometry.AffineSpace ℝ A] [EuclideanGeometry.AffineSpace ℝ B], (A ≠ B) → ∃ (C : Type), [EuclideanGeometry.AffineSpace ℝ C] ∧ (isRightTriangle A B C)

-- Main theorem statement
theorem unique_square_with_hypotenuse_vertices (A B C : Type) [EuclideanGeometry.AffineSpace ℝ A] [EuclideanGeometry.AffineSpace ℝ B] [EuclideanGeometry.AffineSpace ℝ C]
(h₁ : isRightTriangle A B C) (h₂ : isHypotenuse (A, B)) : ∃! (S : Set (Set (EuclideanGeometry.Point ℝ))), ((∃ a b : EuclideanGeometry.Point ℝ, a ∈ S ∧ b ∈ S ∧ EuclideanGeometry.segment a b = Segment (A, B)) ∧ (∀ s ∈ S, EuclideanGeometry.square s)) :=
by sorry

end unique_square_with_hypotenuse_vertices_l100_100780


namespace prob_both_questions_correct_l100_100893

theorem prob_both_questions_correct (P_A P_B P_compl_A_and_compl_B P_A_and_B : ℕ) 
  (h1 : P_A = 63) 
  (h2 : P_B = 49) 
  (h3 : P_compl_A_and_compl_B = 20) 
  (h4 : P_A_and_B = P_A + P_B - (100 - P_compl_A_and_compl_B)) : 
  P_A_and_B = 32 :=
by 
  rw [h1, h2, h3, h4]
  norm_num

end prob_both_questions_correct_l100_100893


namespace cube_volume_increase_l100_100726

theorem cube_volume_increase (s : ℝ) (h : s > 0) :
  let new_volume := (1.4 * s) ^ 3
  let original_volume := s ^ 3
  let increase_percentage := ((new_volume - original_volume) / original_volume) * 100
  increase_percentage = 174.4 := by
  sorry

end cube_volume_increase_l100_100726


namespace tan_45_deg_l100_100129

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100129


namespace periodic_function_l100_100376

noncomputable def f : ℝ → ℝ := sorry

variables (a : ℚ) (b c d : ℝ)

theorem periodic_function (h : ∀ x : ℝ, f (x + a + b) - f (x + b) = c * (x + 2 * a + int.floor x - 2 * (x + a) - int.floor b) + d) :
  ∃ (T : ℝ), ∀ x : ℝ, f (x + T) = f x :=
sorry

end periodic_function_l100_100376


namespace find_m_range_l100_100299

noncomputable def f (x m : ℝ) : ℝ := x * abs (x - m) + 2 * x - 3

theorem find_m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ m ≤ f x₂ m)
    ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end find_m_range_l100_100299


namespace set_intersection_complement_l100_100309

universe u

variable (U A B : Set Nat)

def Complement (U : Set Nat) (B : Set Nat) : Set Nat := {x | x ∈ U ∧ x ∉ B}

theorem set_intersection_complement : 
    U = {1, 2, 3, 4} → A = {1, 2} → B = {1, 4} → A ∩ (Complement U B) = {2} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  simp [Complement]
  sorry

end set_intersection_complement_l100_100309


namespace prob_t_prob_vowel_l100_100754

def word := "mathematics"
def total_letters : ℕ := 11
def t_count : ℕ := 2
def vowel_count : ℕ := 4

-- Definition of being a letter "t"
def is_t (c : Char) : Prop := c = 't'

-- Definition of being a vowel
def is_vowel (c : Char) : Prop := c = 'a' ∨ c = 'e' ∨ c = 'i'

theorem prob_t : (t_count : ℚ) / total_letters = 2 / 11 :=
by
  sorry

theorem prob_vowel : (vowel_count : ℚ) / total_letters = 4 / 11 :=
by
  sorry

end prob_t_prob_vowel_l100_100754


namespace intersection_point_correct_l100_100478

noncomputable def line1 (x : ℝ) : ℝ := 3 * x + 4
noncomputable def line2 (x : ℝ) : ℝ := - (1/3) * x + 5

def intersection_point : ℝ × ℝ :=
  let x := 3 / 10 in
  (x, line1 x)

theorem intersection_point_correct :
  intersection_point = (3 / 10, 49 / 10) := by
  sorry

end intersection_point_correct_l100_100478


namespace calculate_f_g_f_l100_100379

def f (x : ℤ) : ℤ := 5 * x + 5
def g (x : ℤ) : ℤ := 6 * x + 5

theorem calculate_f_g_f : f (g (f 3)) = 630 := by
  sorry

end calculate_f_g_f_l100_100379


namespace construct_plane_through_a_parallel_to_b_l100_100662

-- Definitions
variable (a b : Line) (skew_ab : SkewLines a b)

-- Theorem statement
theorem construct_plane_through_a_parallel_to_b (a b : Line) (skew_ab : SkewLines a b) :
  ∃ P : Plane, LiesIn a P ∧ Parallel b P :=
sorry

end construct_plane_through_a_parallel_to_b_l100_100662


namespace sqrt_div2_eq_10_implies_x_eq_400_l100_100874

def condition (x : ℝ) : Prop := (sqrt x) / 2 = 10

theorem sqrt_div2_eq_10_implies_x_eq_400 {x : ℝ} (h : condition x) : x = 400 :=
sorry

end sqrt_div2_eq_10_implies_x_eq_400_l100_100874


namespace tan_of_45_deg_l100_100088

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100088


namespace product_of_common_divisors_eq_182250000_l100_100627

-- Definitions from conditions
def divisors (n : ℤ) : Set ℤ := {d | d ∣ n}

def common_divisors (a b : ℤ) : Set ℤ := divisors a ∩ divisors b

-- The main theorem to prove
theorem product_of_common_divisors_eq_182250000 :
  ∏ x in common_divisors 210 30, x = 182250000 := 
sorry

end product_of_common_divisors_eq_182250000_l100_100627


namespace greatest_whole_number_satisfying_inequality_l100_100610

-- Define the problem condition
def inequality (x : ℤ) : Prop := 5 * x - 4 < 3 - 2 * x

-- Prove that under this condition, the greatest whole number satisfying it is 0
theorem greatest_whole_number_satisfying_inequality : ∃ (n : ℤ), inequality n ∧ ∀ (m : ℤ), inequality m → m ≤ n :=
begin
  use 0,
  split,
  { -- Proof that 0 satisfies the inequality
    unfold inequality,
    linarith, },
  { -- Proof that 0 is the greatest whole number satisfying the inequality
    intro m,
    unfold inequality,
    intro hm,
    linarith, }
end

#check greatest_whole_number_satisfying_inequality

end greatest_whole_number_satisfying_inequality_l100_100610


namespace rosy_fish_is_twelve_l100_100391

/-- Let lilly_fish be the number of fish Lilly has. -/
def lilly_fish : ℕ := 10

/-- Let total_fish be the total number of fish Lilly and Rosy have together. -/
def total_fish : ℕ := 22

/-- Prove that the number of fish Rosy has is equal to 12. -/
theorem rosy_fish_is_twelve : (total_fish - lilly_fish) = 12 :=
by sorry

end rosy_fish_is_twelve_l100_100391


namespace arrangement_count_l100_100903

theorem arrangement_count (boys girls : Finset ℕ) (A : ℕ)
  (hb : boys.card = 3) (hg : girls.card = 2) (A ∈ boys) :
  ∀ n : ℕ, n = 48 →  -- There are 48 valid arrangements
    (∀ pos : Fin 5 → ℕ,
     -- Boy A must not stand at either end
     (pos 0 ≠ A ∧ pos 4 ≠ A) →

     -- No two girls stand next to each other
     (∀ i : Fin 4, pos i ∈ girls → pos (i + 1) ∉ girls) →

     -- Count the total valid arrangements
     ∃ count : ℕ, count = 48) :=
by
  sorry

end arrangement_count_l100_100903


namespace possible_values_for_b2_l100_100926

theorem possible_values_for_b2 :
  ∀ (b : Nat → Nat),
    b 1 = 1019 →
    b 2 < 1019 →
    b 2 % 2 = 0 →
    (∀ n, b (n + 2) = Nat.abs (b (n + 1) - b n)) →
    b 3006 = 2 →
    ∃ (possible_values : Finset ℕ), possible_values.card = 509 ∧ ∀ x, x ∈ possible_values ↔ x < 1019 ∧ x % 2 = 0 :=
by
  sorry

end possible_values_for_b2_l100_100926


namespace calories_in_250g_of_lemonade_l100_100803

structure Lemonade :=
(lemon_juice_grams : ℕ)
(sugar_grams : ℕ)
(water_grams : ℕ)
(lemon_juice_calories_per_100g : ℕ)
(sugar_calories_per_100g : ℕ)
(water_calories_per_100g : ℕ)

def calorie_count (l : Lemonade) : ℕ :=
(l.lemon_juice_grams * l.lemon_juice_calories_per_100g / 100) +
(l.sugar_grams * l.sugar_calories_per_100g / 100) +
(l.water_grams * l.water_calories_per_100g / 100)

def total_weight (l : Lemonade) : ℕ :=
l.lemon_juice_grams + l.sugar_grams + l.water_grams

def caloric_density (l : Lemonade) : ℚ :=
calorie_count l / total_weight l

theorem calories_in_250g_of_lemonade :
  ∀ (l : Lemonade), 
  l = { lemon_juice_grams := 200, sugar_grams := 300, water_grams := 500,
        lemon_juice_calories_per_100g := 40,
        sugar_calories_per_100g := 390,
        water_calories_per_100g := 0 } →
  (caloric_density l * 250 = 312.5) :=
sorry

end calories_in_250g_of_lemonade_l100_100803


namespace A_subset_B_implies_a_l100_100306

theorem A_subset_B_implies_a (a : ℝ) : 
  (∀ x : ℝ, (x ∈ {x | a * x = x^2}) → x ∈ {0, 1, 2}) ↔ (a = 0 ∨ a = 1 ∨ a = 2) :=
by
  sorry

end A_subset_B_implies_a_l100_100306


namespace tan_of_45_deg_l100_100087

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100087


namespace equivalent_single_number_l100_100329

-- Conditions as definitions
def x : ℝ := (4^3) / 8
def y : ℝ := Real.sqrt (7 / 5)

-- The theorem stating the equivalent number
theorem equivalent_single_number : x * y = (8 * Real.sqrt 35) / 5 := 
by
  sorry

end equivalent_single_number_l100_100329


namespace socorro_training_days_l100_100426

def total_hours := 5
def minutes_per_hour := 60
def total_training_minutes := total_hours * minutes_per_hour

def minutes_multiplication_per_day := 10
def minutes_division_per_day := 20
def daily_training_minutes := minutes_multiplication_per_day + minutes_division_per_day

theorem socorro_training_days:
  total_training_minutes / daily_training_minutes = 10 :=
by
  -- proof omitted
  sorry

end socorro_training_days_l100_100426


namespace training_days_l100_100425

def total_minutes : ℕ := 5 * 60
def minutes_per_day : ℕ := 10 + 20

theorem training_days :
  total_minutes / minutes_per_day = 10 :=
by
  sorry

end training_days_l100_100425


namespace tan_45_deg_eq_one_l100_100171

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100171


namespace middle_term_coefficient_l100_100828

theorem middle_term_coefficient (n k : ℕ) (x : ℝ) (h : n = 8) (m : 2 * k = n) :
  binomial n k * (2 ^ k) = 1120 :=
by
  -- Here, n = 8, k = 4 are the specific values for this problem
  let n : ℕ := 8
  have h : 2 * k = 8 := sorry  -- Given condition
  let k : ℕ := 4
  sorry -- Proving the binomial coefficient and calculation

end middle_term_coefficient_l100_100828


namespace tan_45_eq_1_l100_100010

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100010


namespace pints_of_cider_l100_100699

def pintCider (g : ℕ) (p : ℕ) : ℕ :=
  g / 20 + p / 40

def totalApples (f : ℕ) (h : ℕ) (a : ℕ) : ℕ :=
  f * h * a

theorem pints_of_cider (g p : ℕ) (farmhands : ℕ) (hours : ℕ) (apples_per_hour : ℕ)
  (H1 : g = 1)
  (H2 : p = 2)
  (H3 : farmhands = 6)
  (H4 : hours = 5)
  (H5 : apples_per_hour = 240) :
  pintCider (apples_per_hour * farmhands * hours / 3)
            (apples_per_hour * farmhands * hours * 2 / 3) = 120 :=
by
  sorry

end pints_of_cider_l100_100699


namespace tan_45_eq_1_l100_100009

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100009


namespace nice_string_properties_l100_100993

open List

-- Define what it means for a string to be nice.
def is_nice (s : List Char) : Prop :=
  (∀ c ∈ ['A'.. 'Z'], c ∈ s) ∧
  (∀ π : List Char, 
    Multiset.card ((subsequences s).filter (λ t, t = π)) = 
      Multiset.card ((subsequences s).filter (λ t, t = ['A'..'Z'])))

-- The main theorem to prove
theorem nice_string_properties :
  (∃ s : List Char, is_nice s) ∧ 
  (∀ s : List Char, is_nice s → length s ≥ 2022) :=
by
  sorry

end nice_string_properties_l100_100993


namespace solution_of_binary_linear_equation_l100_100404

theorem solution_of_binary_linear_equation :
  ∃ (x y : ℤ), x + 2 * y = 6 ∧ x = 2 ∧ y = 2 :=
by
  use 2
  use 2
  split
  . calc
    2 + 2 * 2 = 2 + 4 := by rfl
    _ = 6         := by rfl
  . rfl
  . rfl

end solution_of_binary_linear_equation_l100_100404


namespace find_z_l100_100673

theorem find_z 
  (m : ℕ)
  (h1 : (1^(m+1) / 5^(m+1)) * (1^18 / z^18) = 1 / (2 * 10^35))
  (hm : m = 34) :
  z = 4 := 
sorry

end find_z_l100_100673


namespace tan_45_deg_eq_one_l100_100156

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100156


namespace distinct_positive_seq_inequality_l100_100270

noncomputable def sum_reciprocal_geq {a : ℕ → ℕ} (n : ℕ) : Prop :=
  ∑ k in finset.range n, (1 : ℚ) / a k ≥ ∑ k in finset.range n, (1 : ℚ) / (k + 1)

theorem distinct_positive_seq_inequality (a : ℕ → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_pos : ∀ k, 0 < a k)
  (h_inj : function.injective a) :
  ∀ n : ℕ, sum_reciprocal_geq a n :=
by
  intros
  sorry

end distinct_positive_seq_inequality_l100_100270


namespace required_sampling_methods_l100_100335

-- Defining the given conditions
def total_households : Nat := 2000
def farmer_households : Nat := 1800
def worker_households : Nat := 100
def intellectual_households : Nat := total_households - farmer_households - worker_households
def sample_size : Nat := 40

-- Statement representing the proof problem
theorem required_sampling_methods :
  stratified_sampling_needed ∧ systematic_sampling_needed ∧ simple_random_sampling_needed :=
sorry

end required_sampling_methods_l100_100335


namespace quad_ineq_solution_sets_same_quad_ineq_solution_sets_not_same_statement_Q_neither_necessary_nor_sufficient_l100_100797

variable {a1 a2 b1 b2 c1 c2 : ℝ}

theorem quad_ineq_solution_sets_same (h : a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) :
  ∀ x : ℝ, (a1 * x^2 + b1 * x + c1 > 0) ↔ (a2 * x^2 + b2 * x + c2 > 0) :=
sorry

theorem quad_ineq_solution_sets_not_same (h : ¬(a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2)) :
  ∃ x : ℝ, (a1 * x^2 + b1 * x + c1 > 0) ↔ ¬(a2 * x^2 + b2 * x + c2 > 0) :=
sorry

theorem statement_Q_neither_necessary_nor_sufficient :
  ((∃ (a1 a2 b1 b2 c1 c2 : ℝ), quad_ineq_solution_sets_same (⟨eq.refl _, eq.refl _⟩)) →
  (∃ x : ℝ, (a1 * x^2 + b1 * x + c1 > 0) = (a2 * x^2 + b2 * x + c2 > 0))) ∧
  ((¬(∃ (a1 a2 b1 b2 c1 c2 : ℝ), quad_ineq_solution_sets_same (⟨eq.refl _, eq.refl _⟩))) →
  (∃ x : ℝ, (a1 * x^2 + b1 * x + c1 > 0) ≠ (a2 * x^2 + b2 * x + c2 > 0))) :=
sorry

end quad_ineq_solution_sets_same_quad_ineq_solution_sets_not_same_statement_Q_neither_necessary_nor_sufficient_l100_100797


namespace jane_final_answer_l100_100532

theorem jane_final_answer (x : ℕ) (hx : x = 8) : 
  let y := 3 * (x + 3)
  let jane_result := 3 * (y - 2) + 5
  jane_result = 98 :=
by
  rw [hx]
  let y := 3 * (x + 3)
  let jane_result := 3 * (y - 2) + 5
  sorry

end jane_final_answer_l100_100532


namespace leo_current_weight_l100_100894

variable (L K : ℝ)

noncomputable def leo_current_weight_predicate :=
  (L + 10 = 1.5 * K) ∧ (L + K = 180)

theorem leo_current_weight : leo_current_weight_predicate L K → L = 104 := by
  sorry

end leo_current_weight_l100_100894


namespace sum_of_odd_terms_l100_100529

theorem sum_of_odd_terms
  (n : ℕ)
  (a d : ℤ)
  (hn : n = 4020)
  (hd : d = 2)
  (sum : ℕ → ℤ)
  (hsum : (sum n) = 10614)
  (seq : ℕ → ℤ) 
  (hseq0 : seq 0 = a)
  (hseqn : ∀ k, k < n - 1 → seq (k + 1) = seq k + d) :
  let S := ∑ i in finset.range (n/2), seq (2 * i) in
  S = 3297 := 
by
  sorry

end sum_of_odd_terms_l100_100529


namespace probability_of_sum_23_is_7_over_200_l100_100912

/-
A fair, twenty-faced die has 19 of its faces numbered from 1 through 18 and 20, and has one blank face.
Another fair, twenty-faced die has 19 of its faces numbered from 1 through 7 and 9 through 20, and has one blank face.
When the two dice are rolled, what is the probability that the sum of the two numbers facing up will be 23?
-/

def die_1_faces := {x | (1 ≤ x ∧ x ≤ 18) ∨ (x = 20)}
def die_2_faces := {x | (1 ≤ x ∧ x ≠ 8 ∧ x ≤ 20)}

def num_ways_to_sum_23 : ℕ :=
  -- List pairs that sum to 23, taking into account the missing faces
  let pairs := [(3, 20), (5, 18), (6, 17), (7, 16), (9, 14), (10, 13), (11, 12), (12, 11), 
                (13, 10), (14, 9), (16, 7), (17, 6), (18, 5), (20, 3)] 
  in pairs.length

def total_possible_outcomes : ℕ := 20 * 20

def probability_sum_23 : ℚ := num_ways_to_sum_23 / total_possible_outcomes

theorem probability_of_sum_23_is_7_over_200 :
  probability_sum_23 = 7 / 200 := by sorry

end probability_of_sum_23_is_7_over_200_l100_100912


namespace tan_45_eq_1_l100_100200

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100200


namespace tan_of_45_deg_l100_100085

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100085


namespace companion_set_problem_l100_100722

def is_companion_set (A : Set ℚ) : Prop :=
  ∀ (x : ℚ), x ∈ A → (x ≠ 0 ∧ (1 / x) ∈ A)

def non_empty_subsets_with_companion_relations (M : Set ℚ) : ℕ :=
  Set.card { A : Set ℚ | A ⊆ M ∧ is_companion_set A }

theorem companion_set_problem : 
  let M := { -1, 0, 1/2, 2, 3 : ℚ } in
  non_empty_subsets_with_companion_relations M = 3 := by
  sorry

end companion_set_problem_l100_100722


namespace jan_more_miles_than_ian_l100_100886

noncomputable def distance_diff (d t s : ℝ) : ℝ :=
  let han_distance := (s + 10) * (t + 2)
  let jan_distance := (s + 15) * (t + 3)
  jan_distance - (d + 100)

theorem jan_more_miles_than_ian {d t s : ℝ} (H : d = s * t) (H_han : d + 100 = (s + 10) * (t + 2)) : distance_diff d t s = 165 :=
by {
  sorry
}

end jan_more_miles_than_ian_l100_100886


namespace area_quadrilateral_one_third_triangle_l100_100759

-- Given conditions
variable {A B C E F K J : Type} [IsTriangle A B C]
variable [TrisectionPoint E F A B]
variable [TrisectionPoint K J A C]

-- Prove statement
theorem area_quadrilateral_one_third_triangle (S : ℝ) :
  Area (quadrilateral E F J K) = (1 / 3) * Area (triangle A B C) :=
sorry

end area_quadrilateral_one_third_triangle_l100_100759


namespace conclusion1_conclusion2_conclusion3_conclusion4_l100_100284

theorem conclusion1 (f : ℝ → ℝ) (n : ℕ) (h : f = (λ x, x^n)) : 
  (deriv^[5] f) 1 = 120 := sorry

theorem conclusion2 (f : ℝ → ℝ) (h : f = cos) : 
  (deriv^[4] f) = f := sorry

theorem conclusion3 (f : ℝ → ℝ) (h : f = exp) (n : ℕ) (hn : n > 0): 
  (deriv^[n] f) = f := sorry

theorem conclusion4 (f g : ℝ → ℝ) (n : ℕ) (hf : ∀ (k : ℕ), k ≤ n → differentiable ℝ (iterated_deriv k f)) 
    (hg : ∀ (k : ℕ), k ≤ n → differentiable ℝ (iterated_deriv k g)) (h : h = λ x, f x * g x) : 
  (iterated_deriv n h) = λ x, (iterated_deriv n f x * iterated_deriv n g x) := sorry

example : conclusion1 ∧ conclusion2 ∧ conclusion3 ∧ ¬ conclusion4 := 
  ⟨conclusion1, conclusion2, conclusion3, by { intro h, sorry }⟩

end conclusion1_conclusion2_conclusion3_conclusion4_l100_100284


namespace trigonometric_expression_l100_100263

theorem trigonometric_expression (α : ℝ) (h : cos (π / 4 + α) = sqrt 2 / 3) : 
  sin (2 * α) / (1 - sin α + cos α) = 1 / 3 :=
sorry

end trigonometric_expression_l100_100263


namespace statement_A_statement_B_statement_C_l100_100356

variable {α : Type}

-- Conditions for statement A
def angle_greater (A B : ℝ) : Prop := A > B
def sin_greater (A B : ℝ) : Prop := Real.sin A > Real.sin B

-- Conditions for statement B
def acute_triangle (A B C : ℝ) : Prop := A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2
def sin_greater_than_cos (A B : ℝ) : Prop := Real.sin A > Real.cos B

-- Conditions for statement C
def obtuse_triangle (C : ℝ) : Prop := C > Real.pi / 2

-- Statement A in Lean
theorem statement_A (A B : ℝ) : angle_greater A B → sin_greater A B :=
sorry

-- Statement B in Lean
theorem statement_B {A B C : ℝ} : acute_triangle A B C → sin_greater_than_cos A B :=
sorry

-- Statement C in Lean
theorem statement_C {a b c : ℝ} (h : a^2 + b^2 < c^2) : obtuse_triangle C :=
sorry

-- Statement D in Lean (proof not needed as it's incorrect)
-- Theorem is omitted since statement D is incorrect

end statement_A_statement_B_statement_C_l100_100356


namespace tan_of_45_deg_l100_100089

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100089


namespace weight_comparison_l100_100745

def weights : List ℕ := [4, 4, 5, 7, 9, 120]

def median_weight : ℚ :=
  let sorted_weights := weights |>.sort
  (sorted_weights[2] + sorted_weights[3]) / 2

def average_weight : ℚ :=
  (weights.sum : ℚ) / weights.length

-- The proof statement
theorem weight_comparison : average_weight - median_weight = 19 := by
  sorry

end weight_comparison_l100_100745


namespace compute_fg_neg1_l100_100321

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem compute_fg_neg1 : f (g (-1)) = 3 := by
  sorry

end compute_fg_neg1_l100_100321


namespace find_x_l100_100707

theorem find_x (x : ℤ) (h : 3^(x-5) = 9^3) : x = 11 := by
  sorry

end find_x_l100_100707


namespace find_f_f_of_one_fourth_l100_100268

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else 3 ^ x

theorem find_f_f_of_one_fourth : f (f (1 / 4)) = 1 / 9 := by
  sorry

end find_f_f_of_one_fourth_l100_100268


namespace find_u_v_l100_100384

theorem find_u_v (u v : ℤ) (huv_pos : 0 < v ∧ v < u) (area_eq : u^2 + 3 * u * v = 615) : 
  u + v = 45 :=
sorry

end find_u_v_l100_100384


namespace find_line_equation_through_focus_l100_100670

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1
noncomputable def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
noncomputable def line_through_point (k : ℝ) (P : ℝ × ℝ) : ℝ → ℝ := λ x, k * (x - P.1) + P.2

theorem find_line_equation_through_focus
  (k : ℝ)
  (F₁ : ℝ × ℝ := (-1, 0))
  (A B : ℝ × ℝ)
  (hC_A : ellipse A.1 A.2)
  (hC_B : ellipse B.1 B.2)
  (h_AB_dist : distance A B = 8 * real.sqrt 2 / 7) 
  (h_line : ∃ k, ∀ x, A.2 = line_through_point k F₁ A.1 ∧ B.2 = line_through_point k F₁ B.1) :
  (∃ k : ℝ, ∀ x : ℝ, line_through_point k F₁ x = real.sqrt 3 * (x + 1) ∨ line_through_point k F₁ x = -real.sqrt 3 * (x + 1)) :=
sorry

end find_line_equation_through_focus_l100_100670


namespace find_x_l100_100708

theorem find_x (x : ℤ) (h : 3^(x-5) = 9^3) : x = 11 := by
  sorry

end find_x_l100_100708


namespace solve_number_systems_l100_100821

theorem solve_number_systems (a b c : ℕ) :
  (a^2 + 2 * b^2 + 2 = 3 * c + 18) ∧ (2 * a^2 + 4 * b^2 + 7 = 7 * c + 15) →
  a = 4 ∧ b = 6 ∧ c = 24 :=
by
  intro h,
  sorry

end solve_number_systems_l100_100821


namespace greatest_whole_number_inequality_l100_100606

theorem greatest_whole_number_inequality :
  ∃ x : ℕ, (5 * x - 4 < 3 - 2 * x) ∧ ∀ y : ℕ, (5 * y - 4 < 3 - 2 * y → y ≤ x) :=
sorry

end greatest_whole_number_inequality_l100_100606


namespace prove_ZAMENA_l100_100957

theorem prove_ZAMENA :
  ∃ (A M E H Z : ℕ),
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ Z ∧ Z ≤ 5 ∧
    (3 > A) ∧ (A > M) ∧ (M < E) ∧ (E < H) ∧ (H < A) ∧
    (A ≠ M) ∧ (A ≠ E) ∧ (A ≠ H) ∧ (A ≠ Z) ∧
    (M ≠ E) ∧ (M ≠ H) ∧ (M ≠ Z) ∧
    (E ≠ H) ∧ (E ≠ Z) ∧
    (H ≠ Z) ∧
    Z = 5 ∧
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234 :=
begin
  sorry
end

end prove_ZAMENA_l100_100957


namespace prove_ZAMENA_l100_100956

theorem prove_ZAMENA :
  ∃ (A M E H Z : ℕ),
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ Z ∧ Z ≤ 5 ∧
    (3 > A) ∧ (A > M) ∧ (M < E) ∧ (E < H) ∧ (H < A) ∧
    (A ≠ M) ∧ (A ≠ E) ∧ (A ≠ H) ∧ (A ≠ Z) ∧
    (M ≠ E) ∧ (M ≠ H) ∧ (M ≠ Z) ∧
    (E ≠ H) ∧ (E ≠ Z) ∧
    (H ≠ Z) ∧
    Z = 5 ∧
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234 :=
begin
  sorry
end

end prove_ZAMENA_l100_100956


namespace min_val_zero_num_monotonic_intervals_l100_100301

-- 1. Prove minimum value is 0 when a=0
theorem min_val_zero (f : ℝ → ℝ) (e : ℝ) (h : e = Real.exp 1) : 
  (∀ x : ℝ, x ≥ 0 → f x = Real.exp x - e * x → f 1 = 0) :=
sorry -- Proof required

-- 2. Prove number of monotonic intervals when 1 < a < e
theorem num_monotonic_intervals (f : ℝ → ℝ) (a e : ℝ) (h1 : e = Real.exp 1) 
  (h2 : 1 < a) (h3 : a < e) :
  (∀ f' : ℝ → ℝ, (∀ x, f' x = (Real.exp x) - (a * x) + a - e) → 
  (if 1 < a ∧ a ≤ e - 1 then 
    ∃ x0, x0 ∈ (0, ∞) ∧ (∀ x, x < x0 → f' x ≤ 0) ∧ (∀ x, x > x0 → f' x > 0)
    else 
    ∃ x0 x1, x0 < x1 ∧ x0 ∈ (0, (Real.log a)) ∧ (∀ x, x < x0 → f' x > 0) ∧ 
      (∀ x, x > x1 → f' x > 0) ∧ (∀ x, x0 < x ∧ x < x1 → f' x < 0)))
:=
sorry -- Proof required

end min_val_zero_num_monotonic_intervals_l100_100301


namespace complex_simplification_l100_100416

theorem complex_simplification : (4 - 3 * Complex.i)^2 = 7 - 24 * Complex.i := 
  sorry

end complex_simplification_l100_100416


namespace zamena_inequalities_l100_100965

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end zamena_inequalities_l100_100965


namespace tan_five_pi_over_four_l100_100588

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l100_100588


namespace training_days_l100_100424

def total_minutes : ℕ := 5 * 60
def minutes_per_day : ℕ := 10 + 20

theorem training_days :
  total_minutes / minutes_per_day = 10 :=
by
  sorry

end training_days_l100_100424


namespace nabla_eq_37_l100_100565

def nabla (a b : ℤ) : ℤ := a * b + a - b

theorem nabla_eq_37 : nabla (-5) (-7) = 37 := by
  sorry

end nabla_eq_37_l100_100565


namespace neg_pi_lt_neg_314_neg_abs_neg_01_lt_neg_neg_01_l100_100581

theorem neg_pi_lt_neg_314 : -Real.pi < -3.14 := sorry

theorem neg_abs_neg_01_lt_neg_neg_01 : -Real.abs (-0.1) < -(-0.1) := sorry

end neg_pi_lt_neg_314_neg_abs_neg_01_lt_neg_neg_01_l100_100581


namespace max_pieces_with_single_cut_min_cuts_to_intersect_all_pieces_l100_100251

theorem max_pieces_with_single_cut (n : ℕ) (h : n = 4) :
  (∃ m : ℕ, m = 23) :=
sorry

theorem min_cuts_to_intersect_all_pieces (n : ℕ) (h : n = 4) :
  (∃ k : ℕ, k = 3) :=
sorry

noncomputable def pieces_of_cake : ℕ := 23

noncomputable def cuts_required : ℕ := 3

end max_pieces_with_single_cut_min_cuts_to_intersect_all_pieces_l100_100251


namespace minimum_number_of_gloves_l100_100443

theorem minimum_number_of_gloves (participants : ℕ) (gloves_per_participant : ℕ) (total_participants : participants = 63) (each_participant_needs_2_gloves : gloves_per_participant = 2) : 
  participants * gloves_per_participant = 126 :=
by
  rcases participants, gloves_per_participant, total_participants, each_participant_needs_2_gloves
  -- sorry to skip the proof
  sorry

end minimum_number_of_gloves_l100_100443


namespace max_parties_l100_100575

theorem max_parties (p : ℕ) (h : p > 0) :
  ∀ (P : finset (finset (fin p))), pairwise (λ S T, S ∩ T ≠ ∅) → (∀ S ∈ P, ∀ T ∈ P, S ≠ T) → P.card ≤ 2 ^ (p - 1) :=
by sorry

end max_parties_l100_100575


namespace possible_mn_values_l100_100352

theorem possible_mn_values :
  (∀ (a : ℕ → ℕ) (S : ℕ → ℝ),
    a 1 = 1 →
    (∀ n, n ≥ 2 → 3^(n-1) * a n = 3^(n-2) * a (n-1) - 2 * 3^(n-2) + 2) →
    (∀ n, S n = ∑ i in range n, (a (i+1) + 1) / (i+1)) →
    (∀ n, ∀ m, m > 0 →
      (3^m + 1) * (S n - m) / (3^m * (S (n+1) - m)) < 1) →
    ∃ mn, mn ∈ {1, 2, 4})
:= sorry

end possible_mn_values_l100_100352


namespace exists_natural_numbers_satisfying_sum_l100_100415

open Nat

theorem exists_natural_numbers_satisfying_sum (n : ℕ) :
  ∃ (x : Fin n → ℕ), (∀ i j, i < j → x i < x j) ∧ (∑ i in Finset.range n, 1 / (x i : ℝ) - (1 / (∏ i in Finset.range n, x i : ℝ)) ∈ ℕ ∪ {0}) :=
sorry

end exists_natural_numbers_satisfying_sum_l100_100415


namespace part_1_part_2_l100_100264

-- Part 1: Given conditions
def f (x m : ℝ) := Real.exp x - m * x

theorem part_1 (m : ℝ) (h : ∀ x > 0, (x - 2) * f x m + m * x^2 + 2 > 0) : m ≥ 1 / 2 := sorry

-- Part 2: Given conditions and desired proof
theorem part_2 (x1 x2 m : ℝ) (hx1 : Real.exp x1 = m * x1) (hx2 : Real.exp x2 = m * x2) : x1 + x2 > 2 := sorry

end part_1_part_2_l100_100264


namespace cubes_sum_to_91_l100_100447

theorem cubes_sum_to_91
  (a b : ℤ)
  (h : a^3 + b^3 = 91) : a * b = 12 :=
sorry

end cubes_sum_to_91_l100_100447


namespace tan_45_deg_eq_one_l100_100122

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100122


namespace expr_value_l100_100658

variable (a : ℝ)
variable (h : a^2 - 3 * a - 1011 = 0)

theorem expr_value : 2 * a^2 - 6 * a + 1 = 2023 :=
by
  -- insert proof here
  sorry

end expr_value_l100_100658


namespace Q1_Q2_Q3_l100_100692

-- Given sequences {a_n} and {b_n}
variables {a b : ℕ → ℤ}
-- Sums of first n terms
def S (n : ℕ) : ℤ := (finset.range n).sum a
def T (n : ℕ) : ℤ := (finset.range n).sum b
-- Given conditions
axiom a1 : a 1 = 1
axiom S2 : S 2 = 4
axiom recurrence : ∀ n : ℕ, n > 0 → 3 * S (n + 1) = 2 * S n + S (n + 2) + a n
-- Questions
theorem Q1 : ∀ n : ℕ, n > 0 → a n = 2 * n - 1 := sorry
theorem Q2 (h : ∀ n : ℕ, n > 0 → S n > T n) : ∀ n : ℕ, n > 0 → a n > b n := sorry
theorem Q3 : ∀ k : ℕ, k > 0 → (b 1 = a 1) → (b 2 = a 2) → ∀ n : ℕ, 
  n > 0 → (T n = (3^(n - 1) - 1) / 2) → (S n = n ^ 2) → 
  (a n + 2 * T n) / (b n + 2 * S n) = a k → n = 1 ∨ n = 2 := sorry

end Q1_Q2_Q3_l100_100692


namespace product_of_solutions_l100_100316

theorem product_of_solutions (x : ℚ) (h : abs (12 / x + 3) = 2) :
  x = -12 ∨ x = -12 / 5 → x₁ * x₂ = 144 / 5 := by
  sorry

end product_of_solutions_l100_100316


namespace trigonometric_identity_example_l100_100555

theorem trigonometric_identity_example :
  (cos (2 * Real.pi / 180) / sin (47 * Real.pi / 180) + cos (88 * Real.pi / 180) / sin (133 * Real.pi / 180)) = Real.sqrt 2 := by
sorry

end trigonometric_identity_example_l100_100555


namespace find_m_l100_100572

theorem find_m 
  (m : ℤ) 
  (h1 : ∀ x y : ℤ, -3 * x + y = m → 2 * x + y = 28 → x = -6) : 
  m = 58 :=
by 
  sorry

end find_m_l100_100572


namespace find_angle_A_l100_100760

noncomputable def exists_angle_A (A B C : ℝ) (a b : ℝ) : Prop :=
  C = (A + B) / 2 ∧ 
  A + B + C = 180 ∧ 
  (a + b) / 2 = Real.sqrt 3 + 1 ∧ 
  C = 2 * Real.sqrt 2

theorem find_angle_A : ∃ A B C a b, 
  exists_angle_A A B C a b ∧ (A = 75 ∨ A = 45) :=
by
  -- This is where the detailed proof would go
  sorry

end find_angle_A_l100_100760


namespace sum_of_roots_g_4y_eq_13_l100_100794

def g (x : ℝ) : ℝ := x^3 - 2 * x + 5

theorem sum_of_roots_g_4y_eq_13 :
  let y_roots := {y : ℝ | g(4 * y) = 13}
  ∑ y in y_roots, y = 0 :=
by
  sorry

end sum_of_roots_g_4y_eq_13_l100_100794


namespace area_quadrilateral_ABCD_is_448_l100_100747

-- Definitions of points and conditions
structure Point (α : Type) := (x : α) (y : α)
variables {α : Type} [LinearOrderedField α]

def right_angle (a b c : Point α) : Prop := 
  (b.y - a.y) * (c.x - b.x) = (b.x - a.x) * (c.y - b.y)
def is_45_45_90_triangle (a b c : Point α) : Prop := 
  right_angle a b c ∧ (b.y - a.y) = (b.x - a.x)

noncomputable def distance (a b : Point α) : α :=
  ( (b.x - a.x)^2 + (b.y - a.y)^2 )^(1/2)

noncomputable def area_triangle (a b c : Point α) : α :=
  abs ((a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) / 2)

-- Coordinates of points
constants A B C D E : Point ℝ
constant AE : distance A E = 32
constant triangles_A_BE_B_E_C_E_D_E_C_right_angled : right_angle A E B ∧ right_angle B E C ∧ right_angle C E D
constant angles_45_degrees : is_45_45_90_triangle A E B ∧ is_45_45_90_triangle B E C ∧ is_45_45_90_triangle C E D

theorem area_quadrilateral_ABCD_is_448 :
  area_triangle A B E + area_triangle B C E + area_triangle C D E = 448 := 
sorry

end area_quadrilateral_ABCD_is_448_l100_100747


namespace probability_error_not_exceeding_15_l100_100877

open ProbabilityTheory

noncomputable def probability_measurement_error_not_exceeding_15mm : ℝ :=
2 * cdf stdNormal 1.5 - 1

theorem probability_error_not_exceeding_15 :
  P (λ x : ℝ, -15 < x ∧ x < 15) = 0.8664 :=
by
  let X : ℝ → MeasureTheory.ProbMeasure ℝ := λ x, stdNormalPDF (μ := 0) (σ := 10)
  let δ : ℝ := 15
  have h : 2 * cdf stdNormal (δ / 10) - 1 = 0.8664 := sorry
  exact h

end probability_error_not_exceeding_15_l100_100877


namespace intersection_empty_l100_100691

open Set

variable {U : Type} (A B : Set ℝ)

def universal_set := U = ℝ

def set_A := A = {m : ℝ | 3 ≤ m ∧ m < 7}

def set_B := B = {m : ℝ | 2 < m ∧ m ≤ 10}

def complement_B := Bᶜ = {m : ℝ | m ≤ 2 ∨ m > 10}

theorem intersection_empty : universal_set U ∧ set_A A ∧ set_B B → 
  A ∩ Bᶜ = ∅ := by
  sorry

end intersection_empty_l100_100691


namespace verify_system_of_equations_l100_100209

/-- Define a structure to hold the conditions of the problem -/
structure TreePurchasing :=
  (cost_A : ℕ)
  (cost_B : ℕ)
  (diff_A_B : ℕ)
  (total_cost : ℕ)
  (x : ℕ)
  (y : ℕ)

/-- Given conditions for purchasing trees -/
def example_problem : TreePurchasing :=
  { cost_A := 100,
    cost_B := 80,
    diff_A_B := 8,
    total_cost := 8000,
    x := 0,
    y := 0 }

/-- The theorem to prove that the equations match given conditions -/
theorem verify_system_of_equations (data : TreePurchasing) (h_diff : data.x - data.y = data.diff_A_B) (h_cost : data.cost_A * data.x + data.cost_B * data.y = data.total_cost) : 
  (data.x - data.y = 8) ∧ (100 * data.x + 80 * data.y = 8000) :=
  by
    sorry

end verify_system_of_equations_l100_100209


namespace inverse_variation_l100_100410

theorem inverse_variation (a : ℕ) (b : ℝ) (h : a * b = 400) (h₀ : a = 3200) : b = 0.125 :=
by sorry

end inverse_variation_l100_100410


namespace tan_45_deg_eq_one_l100_100101

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100101


namespace problem_equivalent_l100_100718

theorem problem_equivalent (x : Real) : (sqrt (10 + x) + sqrt (30 - x) = 8) → (10 + x) * (30 - x) = 144 := by
  sorry

end problem_equivalent_l100_100718


namespace minnie_longer_time_l100_100393

-- Define Minnie and Penny's speeds
def minnie_flat_speed : ℝ := 20
def minnie_downhill_speed : ℝ := 30
def minnie_uphill_speed : ℝ := 5

def penny_flat_speed : ℝ := 30
def penny_downhill_speed : ℝ := 40
def penny_uphill_speed : ℝ := 10

-- Define travel segments
def uphill_distance : ℝ := 10
def downhill_distance : ℝ := 15
def flat_distance : ℝ := 20

-- Define the total time calculations
def minnie_time : ℝ := (uphill_distance / minnie_uphill_speed) + (downhill_distance / minnie_downhill_speed) + (flat_distance / minnie_flat_speed)
def penny_time : ℝ := (uphill_distance / penny_uphill_speed) + (downhill_distance / penny_downhill_speed) + (flat_distance / penny_flat_speed)

-- Define the time difference and convert to minutes
def time_difference_minutes : ℝ := (minnie_time - penny_time) * 60

-- Claim that the difference is approximately 65 minutes
theorem minnie_longer_time : abs (time_difference_minutes - 87.5) < 5 :=
by sorry

end minnie_longer_time_l100_100393


namespace students_not_taking_test_l100_100395

theorem students_not_taking_test (total_students students_q1 students_q2 students_both not_taken : ℕ)
  (h_total : total_students = 30)
  (h_q1 : students_q1 = 25)
  (h_q2 : students_q2 = 22)
  (h_both : students_both = 22)
  (h_not_taken : not_taken = total_students - students_q2) :
  not_taken = 8 := by
  sorry

end students_not_taking_test_l100_100395


namespace zamena_inequalities_l100_100962

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end zamena_inequalities_l100_100962


namespace period_six_l100_100325

variable {R : Type} [LinearOrderedField R]

def symmetric1 (f : R → R) : Prop := ∀ x : R, f (2 + x) = f (2 - x)
def symmetric2 (f : R → R) : Prop := ∀ x : R, f (5 + x) = f (5 - x)

theorem period_six (f : R → R) (h1 : symmetric1 f) (h2 : symmetric2 f) : ∀ x : R, f (x + 6) = f x :=
sorry

end period_six_l100_100325


namespace nancy_total_cost_l100_100516

-- Define the initial costs per set
def crystalBeadCost := 12
def metalBeadCost := 15
def glassBeadCost := 8

-- Define the quantities purchased
def crystalBeadQuantity := 3
def metalBeadQuantity := 4
def glassBeadQuantity := 2

-- Define the discount and tax rates
def crystalBeadDiscount := 0.10
def metalBeadTax := 0.05
def glassBeadDiscount := 0.07

-- Calculate the total cost for each type before discounts and taxes
def totalCrystalBeadCost := crystalBeadQuantity * crystalBeadCost
def totalMetalBeadCost := metalBeadQuantity * metalBeadCost
def totalGlassBeadCost := glassBeadQuantity * glassBeadCost

-- Apply the discounts and taxes
def discountedCrystalBeadCost := totalCrystalBeadCost * (1 - crystalBeadDiscount)
def taxedMetalBeadCost := totalMetalBeadCost * (1 + metalBeadTax)
def discountedGlassBeadCost := totalGlassBeadCost * (1 - glassBeadDiscount)

-- Calculate the final total cost
def finalTotalCost := discountedCrystalBeadCost + taxedMetalBeadCost + discountedGlassBeadCost

theorem nancy_total_cost :
  finalTotalCost = 110.28 :=
by
  sorry

end nancy_total_cost_l100_100516


namespace greatest_whole_number_inequality_l100_100607

theorem greatest_whole_number_inequality :
  ∃ x : ℕ, (5 * x - 4 < 3 - 2 * x) ∧ ∀ y : ℕ, (5 * y - 4 < 3 - 2 * y → y ≤ x) :=
sorry

end greatest_whole_number_inequality_l100_100607


namespace necessary_but_not_sufficient_for_odd_function_l100_100829

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f (x)

theorem necessary_but_not_sufficient_for_odd_function (f : ℝ → ℝ) :
  (f 0 = 0) ↔ is_odd_function f :=
sorry

end necessary_but_not_sufficient_for_odd_function_l100_100829


namespace focus_of_parabola_l100_100830

theorem focus_of_parabola (x y : ℝ) (h : x^2 = 16 * y) : (0, 4) = (0, 4) :=
by {
  sorry
}

end focus_of_parabola_l100_100830


namespace jennifer_sandwich_fraction_l100_100764

theorem jennifer_sandwich_fraction :
  ∃ (x : ℚ), 180 * x + (1 / 6) * 180 + (1 / 2) * 180 = 180 - 24 ∧ x = 1 / 5 :=
by
  let total := 180
  let sandwich_fraction : ℚ := x
  let museum_ticket_fraction : ℚ := 1 / 6
  let book_fraction : ℚ := 1 / 2
  let money_left := 24

  have h1 : total - money_left = 156 := by norm_num
  existsi sandwich_fraction
  have h2 : 
    180 * sandwich_fraction + (1 / 6) * 180 + (1 / 2) * 180 = 156 := by ring
  split
  · exact h2
  · sorry

end jennifer_sandwich_fraction_l100_100764


namespace zamena_inequalities_l100_100961

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end zamena_inequalities_l100_100961


namespace zamena_inequalities_l100_100964

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end zamena_inequalities_l100_100964


namespace tan_45_deg_l100_100134

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100134


namespace probability_prime_and_greater_than_4_l100_100905

namespace Probability

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

def successful_outcomes_6_sided := { x : ℕ | x ∈ {2, 3, 5} }.card
def successful_outcomes_8_sided := { x : ℕ | x ∈ {5, 6, 7, 8} }.card

theorem probability_prime_and_greater_than_4 :
  (successful_outcomes_6_sided * successful_outcomes_8_sided : ℚ) / (6 * 8) = 1 / 4 := by
sorry

end Probability

end probability_prime_and_greater_than_4_l100_100905


namespace train_speed_calculation_l100_100533

def train_length : ℝ := 150
def crossing_time : ℝ := 14.998800095992321
def speed_converted_kmph : ℝ := 36.002399359758

theorem train_speed_calculation :
  (train_length / crossing_time) * 3.6 = speed_converted_kmph :=
begin
  sorry
end

end train_speed_calculation_l100_100533


namespace johns_money_left_l100_100367

def dog_walking_days_in_april := 26
def earnings_per_day := 10
def money_spent_on_books := 50
def money_given_to_sister := 50

theorem johns_money_left : (dog_walking_days_in_april * earnings_per_day) - (money_spent_on_books + money_given_to_sister) = 160 := 
by
  sorry

end johns_money_left_l100_100367


namespace tan_45_deg_l100_100132

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100132


namespace tan_45_deg_eq_one_l100_100029

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100029


namespace trains_cross_time_l100_100898

-- Define the conditions
def length_of_train : ℝ := 120
def time_to_cross_post_train1 : ℝ := 10
def time_to_cross_post_train2 : ℝ := 14

-- Define the speeds based on conditions
def speed_train1 : ℝ := length_of_train / time_to_cross_post_train1
def speed_train2 : ℝ := length_of_train / time_to_cross_post_train2

-- Define the relative speed when trains travel in opposite directions
def relative_speed : ℝ := speed_train1 + speed_train2

-- Define the total distance to be covered when crossing each other
def total_distance : ℝ := length_of_train + length_of_train

-- Define the time taken to cross each other based on the relative speed
def time_to_cross_each_other : ℝ := total_distance / relative_speed

-- State the proof problem
theorem trains_cross_time : time_to_cross_each_other = 11.67 := 
sorry

end trains_cross_time_l100_100898


namespace zamena_solution_l100_100971

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end zamena_solution_l100_100971


namespace M_is_range_of_sq_function_l100_100686

noncomputable theory

def M : set ℝ := {y | ∃ x : ℝ, y = x^2}

theorem M_is_range_of_sq_function : M = {y | ∃ x : ℝ, y = x^2} :=
by
  sorry

end M_is_range_of_sq_function_l100_100686


namespace fraction_of_garden_occupied_by_triangle_beds_l100_100924

theorem fraction_of_garden_occupied_by_triangle_beds :
  ∀ (rect_height rect_width trapezoid_short_base trapezoid_long_base : ℝ) 
    (num_triangles : ℕ) 
    (triangle_leg_length : ℝ)
    (total_area_triangles : ℝ)
    (total_garden_area : ℝ)
    (fraction : ℝ),
  rect_height = 10 → rect_width = 30 →
  trapezoid_short_base = 20 → trapezoid_long_base = 30 → num_triangles = 3 →
  triangle_leg_length = 10 / 3 →
  total_area_triangles = 3 * (1 / 2 * (triangle_leg_length ^ 2)) →
  total_garden_area = rect_height * rect_width →
  fraction = total_area_triangles / total_garden_area →
  fraction = 1 / 18 := by
  intros rect_height rect_width trapezoid_short_base trapezoid_long_base
         num_triangles triangle_leg_length total_area_triangles
         total_garden_area fraction
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end fraction_of_garden_occupied_by_triangle_beds_l100_100924


namespace tan_45_deg_eq_one_l100_100162

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100162


namespace point_in_second_quadrant_l100_100744

theorem point_in_second_quadrant
  (x y : ℝ)
  (hx : x = -3)
  (hy : y = 2 * Real.sqrt 2)
  (h_neg_x : x < 0)
  (h_pos_y : y > 0) :
  x < 0 ∧ y > 0 :=
by {
  split,
  exact h_neg_x,
  exact h_pos_y,
  sorry -- this 'sorry' placeholder indicates that steps are skipped, proving left for the user.
}

end point_in_second_quadrant_l100_100744


namespace line_through_P_with_opposite_sign_intercepts_l100_100832

theorem line_through_P_with_opposite_sign_intercepts 
  (P : ℝ × ℝ) (hP : P = (3, -2)) 
  (h : ∀ (A B : ℝ), A ≠ 0 → B ≠ 0 → A * B < 0) : 
  (∀ (x y : ℝ), (x = 5 ∧ y = -5) → (5 * x - 5 * y - 25 = 0)) ∨ (∀ (x y : ℝ), (3 * y = -2) → (y = - (2 / 3) * x)) :=
sorry

end line_through_P_with_opposite_sign_intercepts_l100_100832


namespace other_point_on_circle_on_x_axis_l100_100752

-- Definition of a circle with center at the origin and radius 16
def circleC (x y : ℝ) : Prop :=
  x^2 + y^2 = 16^2

-- The other point on circleC that lies on the x-axis
theorem other_point_on_circle_on_x_axis : circleC (-16, 0) → (∃ x, circleC x 0 ∧ x = 16) :=
by
  intro h
  use 16
  sorry

end other_point_on_circle_on_x_axis_l100_100752


namespace specific_value_of_n_l100_100639

theorem specific_value_of_n (n : ℕ) 
  (A_n : ℕ → ℕ)
  (C_n : ℕ → ℕ → ℕ)
  (h1 : A_n n ^ 2 = C_n n (n-3)) :
  n = 8 :=
sorry

end specific_value_of_n_l100_100639


namespace hundred_chicken_problem_l100_100346

theorem hundred_chicken_problem :
  ∃ (x y : ℕ), x + y + 81 = 100 ∧ 5 * x + 3 * y + 81 / 3 = 100 := 
by
  sorry

end hundred_chicken_problem_l100_100346


namespace infinitely_many_m_f_m_lt_f_m_plus_one_infinitely_many_m_f_m_gt_f_m_plus_one_l100_100252

noncomputable def f (n : ℕ) : ℚ := 
  (1 : ℚ) / n * ∑ k in Finset.range n.succ, ⌊(n : ℚ) / k.succ⌋

theorem infinitely_many_m_f_m_lt_f_m_plus_one :
  ∃ᵐ m in (Filter.atTop : Filter ℕ), f m < f (m + 1) := sorry

theorem infinitely_many_m_f_m_gt_f_m_plus_one :
  ∃ᵐ m in (Filter.atTop : Filter ℕ), f m > f (m + 1) := sorry

end infinitely_many_m_f_m_lt_f_m_plus_one_infinitely_many_m_f_m_gt_f_m_plus_one_l100_100252


namespace interval_of_decrease_l100_100645

theorem interval_of_decrease (b c : ℝ) :
  -- Conditions: Function passes through (1, 0) and (3, 0)
  (1:ℝ)^2 + b*1 + c = 0 ∧ (3:ℝ)^2 + b*3 + c = 0 →
  -- To Prove: Interval of decrease is (-∞, 2)
  ∀ x : ℝ, deriv (λ x : ℝ, x^2 + b*x + c) x < 0 ↔ x < 2 := 
by 
  intros h x
  sorry

end interval_of_decrease_l100_100645


namespace propositions_correctness_l100_100674

noncomputable def proposition1 (l : Line) (l₁ l₂ : Line) (π : Plane) : Prop :=
  (l ⊥ l₁) ∧ (l ⊥ l₂) ∧ (l₁ ∈ π) ∧ (l₂ ∈ π) → (l ⊥ π)

noncomputable def proposition2 (π₁ π₂ : Plane) (l₁ l₂ : Line) : Prop :=
  (π₁ ∩ π₂ = ∅) ∧ (l₁ ∈ π₁) ∧ (l₂ ∈ π₂) ∧ (l₁ ∥ l₂) → (π₁ ∥ π₂)

noncomputable def proposition3 (l₁ l₂ : Line) (π : Plane) : Prop :=
  (l₁ ⊥ π) ∧ (l₂ ⊥ π) → (l₁ ∥ l₂)

noncomputable def proposition4 (π₁ π₂ : Plane) (l : Line) (a : Line) : Prop :=
  π₁ ⊥ π₂ ∧ (l ∈ π₁) ∧ ¬ (l ⊥ a) ∧ (a = π₁ ∩ π₂) → ¬ (l ⊥ π₂)

theorem propositions_correctness : 
  (proposition3 l₁ l₂ π = true) ∧ (proposition4 π₁ π₂ l a = true) 
  ∧ (proposition1 l l₁ l₂ π = false) ∧ (proposition2 π₁ π₂ l₁ l₂ = false) :=
sorry

end propositions_correctness_l100_100674


namespace max_value_f_l100_100622

-- Define the function f(x) for x in the interval (0, 1]
def f (x : ℝ) : ℝ := x * (1 - x) / ((x + 1) * (x + 2) * (2 * x + 1))

-- Define the interval (0, 1]
def within_interval (x : ℝ) : Prop := 0 < x ∧ x ≤ 1

-- Statement of the problem
theorem max_value_f :
  ∃ x ∈ set.Icc 0 1, (f x ≠ f x) := sorry

end max_value_f_l100_100622


namespace tan_45_eq_1_l100_100181

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100181


namespace diagonal_length_in_quadrilateral_l100_100742

theorem diagonal_length_in_quadrilateral
  (A B C D : Type*)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (AB BC CD DA AC : ℝ)
  (angle_ADC : ℝ) :
  AB = 10 →
  BC = 10 →
  CD = 17 →
  DA = 17 →
  angle_ADC = 60 → 
  AC = 17 :=
by sorry

end diagonal_length_in_quadrilateral_l100_100742


namespace negation_of_all_squares_positive_l100_100446

theorem negation_of_all_squares_positive :
  ¬ (∀ x : ℝ, x * x > 0) ↔ ∃ x : ℝ, x * x ≤ 0 :=
by sorry

end negation_of_all_squares_positive_l100_100446


namespace set_intersection_example_l100_100798

theorem set_intersection_example :
  let M := {0, 1, 3}
  let N := {0, 1, 7}
  M ∩ N = {0, 1} :=
by
  let M := {0, 1, 3}
  let N := {0, 1, 7}
  sorry

end set_intersection_example_l100_100798


namespace tan_45_deg_l100_100060

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100060


namespace tan_45_deg_l100_100043

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100043


namespace convert_base5_412_to_base7_l100_100996

def base5_to_dec (n : Nat) : Nat :=
  let d2 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n0 := n1 / 10
  let d0 := n0 % 10
  d0 * 25 + d1 * 5 + d2

def dec_to_base7 (n : Nat) : Nat :=
  let r2 := n % 7
  let n1 := n / 7
  let r1 := n1 % 7
  let n0 := n1 / 7
  let r0 := n0 % 7
  r0 * 100 + r1 * 10 + r2

theorem convert_base5_412_to_base7 : 
  dec_to_base7 (base5_to_dec 412) = 212 :=
by
  sorry

end convert_base5_412_to_base7_l100_100996


namespace tan_of_45_deg_l100_100077

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100077


namespace tan_45_deg_eq_one_l100_100023

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100023


namespace arithmetic_sequence_first_term_l100_100850

theorem arithmetic_sequence_first_term
  (a : ℕ) -- First term of the arithmetic sequence
  (d : ℕ := 3) -- Common difference, given as 3
  (n : ℕ := 20) -- Number of terms, given as 20
  (S : ℕ := 650) -- Sum of the sequence, given as 650
  (h : S = (n / 2) * (2 * a + (n - 1) * d)) : a = 4 := 
by
  sorry

end arithmetic_sequence_first_term_l100_100850


namespace gcd_polynomials_l100_100659

theorem gcd_polynomials (a : ℤ) (h_odd_multiple : ∃ k : ℤ, a = 4123 * k ∧ odd a) :
    Int.gcd (4 * a^2 + 35 * a + 81) (3 * a + 8) = 1 := by
  sorry

end gcd_polynomials_l100_100659


namespace experts_win_probability_l100_100222

noncomputable def probability_of_experts_winning (p : ℝ) (q : ℝ) (needed_expert_wins : ℕ) (needed_audience_wins : ℕ) : ℝ :=
  p ^ 4 + 4 * (p ^ 3 * q)

-- Probability values
def p : ℝ := 0.6
def q : ℝ := 1 - p

-- Number of wins needed
def needed_expert_wins : ℕ := 3
def needed_audience_wins : ℕ := 2

theorem experts_win_probability :
  probability_of_experts_winning p q needed_expert_wins needed_audience_wins = 0.4752 :=
by
  -- Proof would go here
  sorry

end experts_win_probability_l100_100222


namespace smallest_positive_integer_conditioned_l100_100492

theorem smallest_positive_integer_conditioned :
  ∃ a : ℕ, a > 0 ∧ (a % 4 = 3) ∧ (a % 3 = 2) ∧ ∀ b : ℕ, b > 0 ∧ (b % 4 = 3) ∧ (b % 3 = 2) → a ≤ b :=
begin
  sorry
end

end smallest_positive_integer_conditioned_l100_100492


namespace tan_45_deg_eq_one_l100_100041

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100041


namespace common_ratio_of_geometric_sequence_l100_100290

theorem common_ratio_of_geometric_sequence (a_1 q : ℝ) (hq : q ≠ 1) 
  (S : ℕ → ℝ) (hS: ∀ n, S n = a_1 * (1 - q^n) / (1 - q))
  (arithmetic_seq : 2 * S 7 = S 8 + S 9) :
  q = -2 :=
by sorry

end common_ratio_of_geometric_sequence_l100_100290


namespace parabola_intersection_length_rectangle_side_length_parabola_intersection_extended_l100_100449

-- Part (a)
theorem parabola_intersection_length :
  let p := λ x : ℝ, 25 - x^2,
      A : ℝ × ℝ := (-5, 0),
      B : ℝ × ℝ := (5, 0) in
  dist A B = 10 :=
sorry

-- Part (b)
theorem rectangle_side_length (BD AB : ℝ) (BD_eq : BD = 26) (AB_eq : AB = 10) :
  let AD := Real.sqrt (BD^2 - AB^2),
      BC := AD in
  BC = 24 :=
sorry

-- Part (c)
theorem parabola_intersection_extended :
  let p := λ x : ℝ, 25 - x^2,
      E : ℝ × ℝ := (-7, -24),
      F : ℝ × ℝ := (7, -24) in
  dist E F = 14 :=
sorry

end parabola_intersection_length_rectangle_side_length_parabola_intersection_extended_l100_100449


namespace seniority_ranking_l100_100215

-- Define the colleagues and the statements.
def Colleague := Type
constant Ella : Colleague
constant Mark : Colleague
constant Nora : Colleague

-- Define the statements
constant Statement_I : Prop -- Mark has the highest seniority
constant Statement_II : Prop -- Ella does not have the highest seniority
constant Statement_III : Prop -- Nora is not the least senior

-- Exactly one of the statements is true
constant exactly_one_true : (Statement_I ∨ Statement_II ∨ Statement_III) ∧ 
  ¬(Statement_I ∧ Statement_II) ∧ ¬(Statement_II ∧ Statement_III) ∧ 
  ¬(Statement_III ∧ Statement_I)

--Proving the only valid ranking
theorem seniority_ranking : 
  exactly_one_true →
  (Ella > Nora > Mark)
 := 
by
  sorry

end seniority_ranking_l100_100215


namespace tan_of_45_deg_l100_100083

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100083


namespace find_x_l100_100636

theorem find_x (x : ℕ) (h₁ : 3 * (Nat.factorial 8) / (Nat.factorial (8 - x)) = 4 * (Nat.factorial 9) / (Nat.factorial (9 - (x - 1)))) : x = 6 :=
sorry

end find_x_l100_100636


namespace pipe_B_fill_time_l100_100471

theorem pipe_B_fill_time (T : ℕ) (h1 : 50 > 0) (h2 : 30 > 0)
  (h3 : (1/50 + 1/T = 1/30)) : T = 75 := 
sorry

end pipe_B_fill_time_l100_100471


namespace cos_alpha_value_l100_100261

theorem cos_alpha_value (α : ℝ) (hα : 0 < α ∧ α < real.pi)
  (h_equation : 1 - 2 * real.sin (2 * α) = real.cos (2 * α)) :
  real.cos α = real.sqrt 5 / 5 :=
sorry

end cos_alpha_value_l100_100261


namespace base6_arithmetic_l100_100981

def base6_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  let n3 := n2 / 10
  let d3 := n3 % 10
  let n4 := n3 / 10
  let d4 := n4 % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0

def base10_to_base6 (n : ℕ) : ℕ :=
  let b4 := n / 6^4
  let r4 := n % 6^4
  let b3 := r4 / 6^3
  let r3 := r4 % 6^3
  let b2 := r3 / 6^2
  let r2 := r3 % 6^2
  let b1 := r2 / 6^1
  let b0 := r2 % 6^1
  b4 * 10000 + b3 * 1000 + b2 * 100 + b1 * 10 + b0

theorem base6_arithmetic : 
  base10_to_base6 ((base6_to_base10 45321 - base6_to_base10 23454) + base6_to_base10 14553) = 45550 :=
by
  sorry

end base6_arithmetic_l100_100981


namespace experts_eventual_win_probability_l100_100235

noncomputable def experts_win_probability (p : ℝ) (q : ℝ) (experts_needed : ℕ) (audience_needed : ℕ) (max_rounds : ℕ) : ℝ :=
  let e_all_wins := p ^ max_rounds
  let e_three_wins_one_loss := (Nat.choose max_rounds (max_rounds - 1)) * (p ^ (max_rounds - 1)) * q
  e_all_wins + e_three_wins_one_loss

theorem experts_eventual_win_probability :
  ∀ (p : ℝ) (q : ℝ),
  p = 0.6 →
  q = 1 - p →
  experts_win_probability p q 3 2 4 = 0.4752 := by
  intros p q hp hq
  have h : experts_win_probability p q 3 2 4 = 0.6 ^ 4 + 4 * (0.6 ^ 3) * 0.4 := sorry
  rw [hp, hq] at h
  norm_num at h
  exact h

end experts_eventual_win_probability_l100_100235


namespace smallest_k_for_9x9_l100_100490

def three_cell_corner := ({ (1, 1), (1, 2), (2, 1) } : set (ℕ × ℕ))

def covers_two_marked {m n : ℕ} (corner : set (ℕ × ℕ)) (marks : set (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ),
    (i < m - 1) → (j < n - 1) →
    (corner.map (λ x => (x.1 + i, x.2 + j)) ∩ marks).size ≥ 2

def minimal_marks (m n k : ℕ) : Prop :=
  ∀ (marks : set (ℕ × ℕ)),
    marks.size = k →
    covers_two_marked three_cell_corner marks

theorem smallest_k_for_9x9 : minimal_marks 9 9 56 ∧ ∀ k, minimal_marks 9 9 k → 56 ≤ k :=
sorry

end smallest_k_for_9x9_l100_100490


namespace find_total_count_l100_100860

theorem find_total_count (N S : ℕ) (f : Fin N → ℝ)
  (h_avg : (∑ i, f i) / N = 44)
  (h_avg_first11 : (∑ i in finset.range 11, f i) / 11 = 48)
  (h_avg_last11 : (∑ i in finset.range (N - 11), f (i + N - 11 - 1)) / 11 = 41)
  (h_11th : f 10 = 55) :
  N = 21 :=
by
  sorry

end find_total_count_l100_100860


namespace smallest_possible_number_bob_l100_100937

theorem smallest_possible_number_bob (bob_number alice_number : ℕ) (h_alice : alice_number = 30)
  (h_bob : ∀ p : ℕ, prime p → p ∣ alice_number → p ∣ bob_number) : bob_number ≥ 30 :=
by
  -- Skipping the proof steps
  sorry

end smallest_possible_number_bob_l100_100937


namespace solve_for_y_l100_100429

theorem solve_for_y (y : ℝ) : (y - 3)^3 = (1 / 27 : ℝ)^(-1) → y = 6 := by
  sorry

end solve_for_y_l100_100429


namespace find_max_value_l100_100624

noncomputable def f (x : ℝ) : ℝ := (x * (1 - x)) / ((x + 1) * (x + 2) * (2 * x + 1))

theorem find_max_value :
  ∃ x ∈ set.Ioc 0 1, ∀ y ∈ set.Ioc 0 1, f(y) ≤ f(x) ∧ f(x) = (8 * real.sqrt 2 - 5 * real.sqrt 5) / 3 :=
sorry

end find_max_value_l100_100624


namespace strictly_increasing_interval_l100_100678

-- Define the function and given conditions
def f (ω x : ℝ) : ℝ := sqrt 3 * sin (ω * x) + cos (ω * x)

-- Main theorem statement
theorem strictly_increasing_interval (ω : ℝ) (k : ℤ) :
  (ω > 0) →
  (∃ T > 0, (∀ x, f ω (x + T) = f ω x) ∧ T = π / ω) →
  ∀ x, (f ω x > f ω (x - (π / ω))) → 
  ((k : ℝ) * π - π / 3 ≤ x ∧ x ≤ (k : ℝ) * π + π / 6) :=
begin
  sorry
end

end strictly_increasing_interval_l100_100678


namespace john_money_left_l100_100369

noncomputable def total_earned (days_worked : ℕ) (earnings_per_day : ℕ) : ℕ := days_worked * earnings_per_day

noncomputable def total_spent (spent_books: ℕ) (spent_kaylee: ℕ) : ℕ := spent_books + spent_kaylee

def amount_left (total_earned : ℕ) (total_spent : ℕ) : ℕ := total_earned - total_spent

theorem john_money_left : 
  let days_worked := 26 in
  let earnings_per_day := 10 in
  let spent_books := 50 in
  let spent_kaylee := 50 in
  amount_left (total_earned days_worked earnings_per_day) (total_spent spent_books spent_kaylee) = 160 :=
by
  sorry

end john_money_left_l100_100369


namespace problem_statement_l100_100407

noncomputable def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n

theorem problem_statement (m n p q : ℕ) (h₁ : m ≠ p) (h₂ : is_integer ((mn + pq : ℚ) / (m - p))) :
  is_integer ((mq + np : ℚ) / (m - p)) :=
sorry

end problem_statement_l100_100407


namespace part1_part2_l100_100697

-- Given conditions
variables {a b : ℝ} 
axiom h1 : a + b = 3
axiom h2 : ab = 1
axiom h3 : a < b

-- Part 1: Prove a^2 + b^2 = 7
theorem part1 : a^2 + b^2 = 7 :=
sorry

-- Part 2: Prove a - b = -√5 when a < b
theorem part2 : a - b = -real.sqrt 5 :=
sorry

end part1_part2_l100_100697


namespace tan_45_deg_eq_one_l100_100040

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100040


namespace tan_of_45_deg_l100_100091

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100091


namespace tan_45_deg_eq_one_l100_100034

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100034


namespace distinct_banners_l100_100515

inductive Color
| red
| white
| blue
| green
| yellow

def adjacent_different (a b : Color) : Prop := a ≠ b

theorem distinct_banners : 
  ∃ n : ℕ, n = 320 ∧ ∀ strips : Fin 4 → Color, 
    adjacent_different (strips 0) (strips 1) ∧ 
    adjacent_different (strips 1) (strips 2) ∧ 
    adjacent_different (strips 2) (strips 3) :=
sorry

end distinct_banners_l100_100515


namespace tan_45_deg_l100_100133

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100133


namespace tan_45_deg_eq_one_l100_100108

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100108


namespace find_multiplier_l100_100527

theorem find_multiplier (A : ℕ) (hA : A = 30) (N : ℕ) :
  ((A + 5) * N - (A - 5) * N = A) → N = 3 :=
by
  intro h
  rw hA at h
  have h_eq : (35 * N - 25 * N = 30) := h
  -- We can see the left side of the equation simplifies to 10 * N
  -- hence (10 * N = 30)
  sorry

end find_multiplier_l100_100527


namespace tangent_line_to_exp_curve_l100_100665

theorem tangent_line_to_exp_curve (k : ℝ) 
  (h_tangent : ∃ x0, exp x0 = x0 + k ∧ deriv (λ x : ℝ, exp x) x0 = deriv (λ x : ℝ, x + k) x0) :
  k = 1 :=
begin
  sorry
end

end tangent_line_to_exp_curve_l100_100665


namespace correct_calculation_l100_100878

theorem correct_calculation :
  let A := (sqrt 3 + sqrt 3 ≠ sqrt 6)
  let B := (sqrt 27 ≠ 3)
  let C := (sqrt (4/25) ≠ ±2/5)
  let D := (cbrt 8 = 2)
  D :=
by
  -- Definitions based on problem conditions
  have A_def : sqrt 3 + sqrt 3 ≠ sqrt 6 := by sorry
  have B_def : sqrt 27 ≠ 3 := by sorry
  have C_def : sqrt (4/25) ≠ ±2/5 := by sorry
  have D_def : cbrt 8 = 2 := by sorry
  -- Assert D is the correct calculation
  exact D_def

end correct_calculation_l100_100878


namespace solution_set_f_x_le_x_cubed_l100_100788

theorem solution_set_f_x_le_x_cubed :
  ∀ (f : ℝ → ℝ),
    (∀ x : ℝ, x ≠ 0 → f (x + 1) = -f (-x + 1)) →
    (f 1 = 1) →
    (∀ (x1 x2 : ℝ), x1 > 0 → x2 > 0 → x1 ≠ x2 → (x2^3 * f x1 - x1^3 * f x2) / (x1 - x2) > 0) →
    {x : ℝ | f x ≤ x^3} = {x : ℝ | x ∣ (x ≤ 0 ∧ x ≥ -1) ∨ (x > 0 ∧ x ≤ 1)} :=
begin
  intros f Hsym Hf1 Hcond,
  sorry
end

end solution_set_f_x_le_x_cubed_l100_100788


namespace tan_45_deg_eq_one_l100_100116

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100116


namespace normal_distribution_prob_l100_100800

noncomputable def normal_probabilities (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  (X ~ N(μ, σ^2)) ∧
  (P(X < 1) = 1/2) ∧
  (P(X > 2) = 1/5) ∧
  (P(X < 0) = 1/5)

theorem normal_distribution_prob (X : ℝ → ℝ) (μ σ : ℝ) :
  normal_probabilities X μ σ →
  P(0 < X < 1) = 0.3 := 
sorry

end normal_distribution_prob_l100_100800


namespace total_team_points_l100_100578

/-- Emily's team won a dodgeball game. Emily scored 23 points,
each of the other players scored 2 points, and there were 8 players
on her team. This proof verifies the total points scored by the team. -/
theorem total_team_points :
  ∃ team_points, team_points = 23 + (7 * 2) ∧ team_points = 37 :=
by
  let emily_points := 23
  let other_players_points := 7 * 2
  let total_points := emily_points + other_players_points
  use total_points
  split
  · rfl
  · rfl

end total_team_points_l100_100578


namespace average_of_data_set_is_five_l100_100648

def data_set : List ℕ := [2, 5, 5, 6, 7]

def sum_of_data_set : ℕ := data_set.sum
def count_of_data_set : ℕ := data_set.length

theorem average_of_data_set_is_five :
  (sum_of_data_set / count_of_data_set) = 5 :=
by
  sorry

end average_of_data_set_is_five_l100_100648


namespace tan_five_pi_over_four_l100_100591

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l100_100591


namespace solution_set_of_inequality_l100_100632

theorem solution_set_of_inequality (x : ℝ) (h: x ≠ 0) : 
  (x > 1/x) ↔ x ∈ (Set.Ioo (-∞) (-1) ∪ Set.Ioo 1 ∞) :=
by
  sorry

end solution_set_of_inequality_l100_100632


namespace divide_payment_correctly_l100_100867

-- Define the number of logs contributed by each person
def logs_troikin : ℕ := 3
def logs_pyaterkin : ℕ := 5
def logs_bestoplivny : ℕ := 0

-- Define the total number of logs
def total_logs : ℕ := logs_troikin + logs_pyaterkin + logs_bestoplivny

-- Define the total number of logs used equally
def logs_per_person : ℚ := total_logs / 3

-- Define the total payment made by Bestoplivny 
def total_payment : ℕ := 80

-- Define the cost per log
def cost_per_log : ℚ := total_payment / logs_per_person

-- Define the contribution of each person to Bestoplivny
def bestoplivny_from_troikin : ℚ := logs_troikin - logs_per_person
def bestoplivny_from_pyaterkin : ℚ := logs_pyaterkin - (logs_per_person - bestoplivny_from_troikin)

-- Define the kopecks received by Troikina and Pyaterkin
def kopecks_troikin : ℚ := bestoplivny_from_troikin * cost_per_log
def kopecks_pyaterkin : ℚ := bestoplivny_from_pyaterkin * cost_per_log

-- Main theorem to prove the correct division of kopecks
theorem divide_payment_correctly : kopecks_troikin = 10 ∧ kopecks_pyaterkin = 70 :=
by
  -- ... Proof goes here
  sorry

end divide_payment_correctly_l100_100867


namespace intersection_point_of_lines_l100_100693

theorem intersection_point_of_lines : 
  ∃ x y : ℝ, (3 * x + 4 * y - 2 = 0) ∧ (2 * x + y + 2 = 0) ∧ (x = -2) ∧ (y = 2) := 
by 
  sorry

end intersection_point_of_lines_l100_100693


namespace kenneth_earnings_l100_100771

-- Define the initial conditions
def spent_percentage : ℝ := 0.1
def remaining_amount : ℝ := 405

-- Define the total earnings we need to prove
def total_earnings : ℝ := remaining_amount / (1 - spent_percentage)

-- The statement to be proved
theorem kenneth_earnings : total_earnings = 450 :=
by
  sorry

end kenneth_earnings_l100_100771


namespace natasha_avg_speed_climbing_l100_100896

theorem natasha_avg_speed_climbing :
  ∀ (D : ℝ), 
  (4 : ℝ) * (1.5 : ℝ) = D →
  D = 2 * (1.5 : ℝ) + 2 * 1.5 →
  (6 : ℝ) = 2 * D / 2 →
  D / 4 = 1.5 :=
by {
  intros,
  sorry
}

end natasha_avg_speed_climbing_l100_100896


namespace cider_pints_produced_l100_100702

/-- Define the conditions as constants and parameters -/
def golden_apples_per_pint := 20
def pink_apples_per_pint := 40
def farmhands := 6
def apples_per_hour_per_farmhand := 240
def hours_worked := 5
def golden_to_pink_ratio (golden: ℕ) (pink: ℕ) := golden = 2 * pink

/-- Main theorem statement -/
theorem cider_pints_produced : 
  ∀ (golden picked_apples : ℕ),
  let total_apples := picked_apples * farmhands * hours_worked in
  let total_pints := total_apples / (golden_apples_per_pint + pink_apples_per_pint) in
  total_apples = 7200 → 
  total_pints = 120 :=
by
  intros golden picked_apples total_apples total_pints h_tot_apples
  sorry

end cider_pints_produced_l100_100702


namespace tan_five_pi_over_four_l100_100592

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l100_100592


namespace jan_drove_more_than_ian_l100_100887

variables (d t s : ℝ)

-- Ian's distance relation
def ian_distance : Prop := d = s * t

-- Han's driving relation derived by simplifying d + 100 = (s + 10) * (t + 2)
def han_condition : Prop := 5 * t + s = 40

-- Jan's distance relation
def jan_condition : Prop := m = (s + 15) * (t + 3)

-- The final proposition we need to prove
def jan_drove_165_more : Prop := (s + 15) * (t + 3) - d = 165

-- Final theorem statement
theorem jan_drove_more_than_ian (h1 : ian_distance d t s) (h2 : han_condition d t s) (h3 : jan_condition d t s) :
  jan_drove_165_more d t s :=
sorry

end jan_drove_more_than_ian_l100_100887


namespace tan_45_eq_1_l100_100195

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100195


namespace tan_45_deg_eq_one_l100_100017

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100017


namespace restore_axes_with_parabola_l100_100400

-- Define the given parabola y = x^2
def parabola (x : ℝ) : ℝ := x^2

-- Problem: Prove that you can restore the coordinate axes using the given parabola and tools.
theorem restore_axes_with_parabola : 
  ∃ O X Y : ℝ × ℝ, 
  (∀ x, parabola x = (x, x^2).snd) ∧ 
  (X.fst = 0 ∧ Y.snd = 0) ∧
  (O = (0,0)) :=
sorry

end restore_axes_with_parabola_l100_100400


namespace tan_of_45_deg_l100_100081

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100081


namespace kenneth_earnings_l100_100774

-- Definitions according to conditions
def spent_percent : ℝ := 0.10
def left_amount : ℝ := 405.0
def total_earnings : ℝ := left_amount / (1 - spent_percent)

-- Theorem statement
theorem kenneth_earnings (spent_percent : ℝ) (left_amount : ℝ) : total_earnings = 450 :=
by
  sorry

end kenneth_earnings_l100_100774


namespace odd_periodic_function_value_l100_100287

variable {f : ℝ → ℝ}

theorem odd_periodic_function_value :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, f (x + 4) = f x) →
  (∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2) →
  f 7 = -2 :=
by
  intros odd_func periodic_func interval_def
  -- Proof goes here
  skip

end odd_periodic_function_value_l100_100287


namespace tan_45_deg_eq_one_l100_100121

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100121


namespace fish_remaining_l100_100396

variable (initial_fish : ℝ) (given_fish : ℝ)

theorem fish_remaining (initial_fish_given : initial_fish = 47.0) (given_fish_given : given_fish = 22.0) :
  initial_fish - given_fish = 25.0 :=
by
  rw [initial_fish_given, given_fish_given]
  norm_num
  sorry

end fish_remaining_l100_100396


namespace tan_45_eq_1_l100_100198

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100198


namespace tan_45_deg_eq_one_l100_100025

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100025


namespace zamena_solution_l100_100969

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end zamena_solution_l100_100969


namespace coefficient_x2_in_expansion_l100_100348

theorem coefficient_x2_in_expansion : 
  let x := (λ x : ℝ, x - 2 / x) in
  (x 1) ^ 4 * (binom 4 1) * (-2) = -8 :=
by sorry

end coefficient_x2_in_expansion_l100_100348


namespace find_m_l100_100641

variables (a b : ℝ × ℝ) (m : ℝ)

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 + v2.1, v1.2 + v2.2)

def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
(c * v.1, c * v.2)

def parallel (v1 v2 : ℝ × ℝ) : Prop :=
v1.1 * v2.2 = v1.2 * v2.1

theorem find_m :
  let a := (1, 2) in
  let b := (-3, 0) in
  parallel (vector_add (scalar_mul 2 a) b) (vector_add a (scalar_mul (-m) b)) →
  m = -1/2 :=
by
  sorry

end find_m_l100_100641


namespace thomas_needs_more_money_l100_100460

-- Define the conditions in Lean
def weeklyAllowance : ℕ := 50
def hourlyWage : ℕ := 9
def hoursPerWeek : ℕ := 30
def weeklyExpenses : ℕ := 35
def weeksInYear : ℕ := 52
def carCost : ℕ := 15000

-- Define the total earnings for the first year
def firstYearEarnings : ℕ :=
  weeklyAllowance * weeksInYear

-- Define the weekly earnings from the second year job
def secondYearWeeklyEarnings : ℕ :=
  hourlyWage * hoursPerWeek

-- Define the total earnings for the second year
def secondYearEarnings : ℕ :=
  secondYearWeeklyEarnings * weeksInYear

-- Define the total earnings over two years
def totalEarnings : ℕ :=
  firstYearEarnings + secondYearEarnings

-- Define the total expenses over two years
def totalExpenses : ℕ :=
  weeklyExpenses * (2 * weeksInYear)

-- Define the net savings after two years
def netSavings : ℕ :=
  totalEarnings - totalExpenses

-- Define the amount more needed for the car
def amountMoreNeeded : ℕ :=
  carCost - netSavings

-- The theorem to prove
theorem thomas_needs_more_money : amountMoreNeeded = 2000 := by
  sorry

end thomas_needs_more_money_l100_100460


namespace linda_needs_additional_batches_l100_100392

theorem linda_needs_additional_batches:
  let classmates := 24
  let cookies_per_classmate := 10
  let dozen := 12
  let cookies_per_batch := 4 * dozen
  let cookies_needed := classmates * cookies_per_classmate
  let chocolate_chip_batches := 2
  let oatmeal_raisin_batches := 1
  let cookies_made := (chocolate_chip_batches + oatmeal_raisin_batches) * cookies_per_batch
  let remaining_cookies := cookies_needed - cookies_made
  let additional_batches := remaining_cookies / cookies_per_batch
  additional_batches = 2 :=
by
  sorry

end linda_needs_additional_batches_l100_100392


namespace tan_45_deg_eq_one_l100_100022

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100022


namespace max_value_of_function_l100_100617

theorem max_value_of_function : 
  (∀ x : ℝ, f x = 6 * sin x + 8 * cos x) →
  ∃ m : ℝ, (∀ x : ℝ, f x ≤ m) ∧ (∃ x₀ : ℝ, f x₀ = m) :=
begin
  sorry
end

end max_value_of_function_l100_100617


namespace average_first_100_terms_l100_100563

def a (n : ℕ) : ℤ := (-1)^(n + 1) * (n : ℤ)^2

theorem average_first_100_terms : 
  (∑ k in finset.range 100, a (k + 1) : ℤ) / 100 = -50.5 :=
by sorry

end average_first_100_terms_l100_100563


namespace cyclic_quadrilateral_area_l100_100445

-- Definitions of the side lengths and cyclic property of quadrilateral ABCD
def AB : ℝ := 2
def BC : ℝ := 7
def CD : ℝ := 6
def DA : ℝ := 9

def is_cyclic_quadrilateral (AB BC CD DA : ℝ) : Prop := -- Should be true if the quadrilateral is cyclic
  sorry -- Replace with actual definition if necessary

-- Assertion about the area of the cyclic quadrilateral
theorem cyclic_quadrilateral_area (h_cyclic: is_cyclic_quadrilateral AB BC CD DA) : 
  let area := 30 in 
  area = 30 := 
by
  sorry -- Placeholder for actual proof

end cyclic_quadrilateral_area_l100_100445


namespace probability_sqrt2_inequality_l100_100750

theorem probability_sqrt2_inequality (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) :
  (1 / π * ∫ θ in 0..π, if sqrt 2 ≤ sqrt 2 * cos θ + sqrt 2 * sin θ ∧ sqrt 2 * cos θ + sqrt 2 * sin θ ≤ 2 then 1 else 0) = 1 / 2 :=
by
  sorry

end probability_sqrt2_inequality_l100_100750


namespace probability_of_combined_event_l100_100864

/-- Define a standard deck of 52 cards -/
def deck_size := 52

/-- Define the number of hearts in a standard deck -/
def hearts := 13

/-- Define the number of diamonds -/
def diamonds := 13

/-- Define the number of aces -/
def aces := 4

/-- Define the number of cards that are neither diamonds nor aces -/
def non_diamond_non_ace := deck_size - (diamonds + aces)

/-- Calculate the probability that a single card drawn is a heart -/
def prob_heart := hearts * (1 / deck_size)

/-- Calculate the probability that a single card drawn is neither a diamond nor an ace -/
def prob_not_diamond_nor_ace := non_diamond_non_ace * (1 / deck_size)

noncomputable def probability :=
  let prob_heart := hearts / deck_size in
  let prob_not_diamond_nor_ace := non_diamond_non_ace / deck_size in
  let prob_at_least_one_diamond_or_ace := 1 - ((prob_not_diamond_nor_ace) ^ 2) in
  let prob_combined := prob_at_least_one_diamond_or_ace * prob_heart in
  prob_combined

theorem probability_of_combined_event :
  probability = 88 / 676 :=
by
  -- proof steps here
  sorry

end probability_of_combined_event_l100_100864


namespace thomas_needs_more_money_l100_100459

-- Define the conditions in Lean
def weeklyAllowance : ℕ := 50
def hourlyWage : ℕ := 9
def hoursPerWeek : ℕ := 30
def weeklyExpenses : ℕ := 35
def weeksInYear : ℕ := 52
def carCost : ℕ := 15000

-- Define the total earnings for the first year
def firstYearEarnings : ℕ :=
  weeklyAllowance * weeksInYear

-- Define the weekly earnings from the second year job
def secondYearWeeklyEarnings : ℕ :=
  hourlyWage * hoursPerWeek

-- Define the total earnings for the second year
def secondYearEarnings : ℕ :=
  secondYearWeeklyEarnings * weeksInYear

-- Define the total earnings over two years
def totalEarnings : ℕ :=
  firstYearEarnings + secondYearEarnings

-- Define the total expenses over two years
def totalExpenses : ℕ :=
  weeklyExpenses * (2 * weeksInYear)

-- Define the net savings after two years
def netSavings : ℕ :=
  totalEarnings - totalExpenses

-- Define the amount more needed for the car
def amountMoreNeeded : ℕ :=
  carCost - netSavings

-- The theorem to prove
theorem thomas_needs_more_money : amountMoreNeeded = 2000 := by
  sorry

end thomas_needs_more_money_l100_100459


namespace tan_45_deg_eq_one_l100_100037

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100037


namespace A_div_B_l100_100999

noncomputable def A : ℝ := ∑' n in {1, 7, -11, -17, 19, 23, ...}.filter (fun n => ¬ ∃ k, n = 5 * (2 * k + 1)),
  1 / (n^2)

noncomputable def B : ℝ := ∑' n in {5, -15, 25, -35, 45, -55, ...}.filter (fun n => ∃ k, n = 5 * (2 * k + 1)), 
  1 / (n^2)

theorem A_div_B : A / B = 26 := sorry

end A_div_B_l100_100999


namespace player_reach_wingspan_l100_100452

theorem player_reach_wingspan :
  ∀ (rim_height player_height jump_height reach_above_rim reach_with_jump reach_wingspan : ℕ),
  rim_height = 120 →
  player_height = 72 →
  jump_height = 32 →
  reach_above_rim = 6 →
  reach_with_jump = player_height + jump_height →
  reach_wingspan = (rim_height + reach_above_rim) - reach_with_jump →
  reach_wingspan = 22 :=
by
  intros rim_height player_height jump_height reach_above_rim reach_with_jump reach_wingspan
  intros h_rim_height h_player_height h_jump_height h_reach_above_rim h_reach_with_jump h_reach_wingspan
  rw [h_rim_height, h_player_height, h_jump_height, h_reach_above_rim] at *
  simp at *
  sorry

end player_reach_wingspan_l100_100452


namespace smallest_k_l100_100371

open Function

noncomputable def f (a b : ℕ) (M : ℤ) (n : ℤ) : ℤ :=
  if n ≤ M then n + a else n - b

def f_iter (a b : ℕ) (M : ℤ) : ℕ → ℤ → ℤ
| 0, n     => n
| (k+1), n => f a b M (f_iter k n)

theorem smallest_k (a b : ℕ) (h : 1 ≤ a ∧ a ≤ b) :
  ∃ (k : ℕ), (k = (a + b) / Nat.gcd a b) ∧ f_iter a b (Int.floor ((a + b) / 2 : ℤ)) k 0 = 0 :=
  sorry

end smallest_k_l100_100371


namespace kenneth_earnings_l100_100772

-- Define the initial conditions
def spent_percentage : ℝ := 0.1
def remaining_amount : ℝ := 405

-- Define the total earnings we need to prove
def total_earnings : ℝ := remaining_amount / (1 - spent_percentage)

-- The statement to be proved
theorem kenneth_earnings : total_earnings = 450 :=
by
  sorry

end kenneth_earnings_l100_100772


namespace measure_angle_AOC_l100_100271

-- Definitions representing vectors and their magnitudes
variables {V : Type*} [InnerProductSpace ℝ V]
variables (A B C O : V)

-- The problem setup: circumcenter \(O\), vectors from \(O\) to \(A\), \(B\), \(C\) and the given condition
noncomputable def problem_statement : Prop :=
  ∥A - O∥ = 1 ∧ ∥B - O∥ = 1 ∧ ∥C - O∥ = 1 ∧
  (A - O) + (√3) • (B - O) + 2 • (C - O) = 0

-- The proof goal: measure of ∠AOC
noncomputable def angle_AOC : ℝ :=
  real.angle (A - O) (C - O)

-- The Lean statement to prove the measure of ∠AOC
theorem measure_angle_AOC (h : problem_statement A B C O) : angle_AOC A B C O = 2 * π / 3 :=
by sorry

end measure_angle_AOC_l100_100271


namespace michael_and_truck_never_meet_l100_100805

-- Definitions from the conditions
variable {michael_speed : ℕ} (6)
variable {truck_speed : ℕ} (12)
variable {stop_duration : ℕ} (40)
variable {initial_position_michael : ℕ} (0)
variable {initial_position_truck : ℕ} (300)
variable {total_walk_time : ℕ} (900)
variable {meeting_count : ℕ}

-- Statement of the problem 
theorem michael_and_truck_never_meet :
  michael_speed = 6 →
  truck_speed = 12 →
  stop_duration = 40 →
  initial_position_michael = 0 →
  initial_position_truck = 300 →
  total_walk_time = 900 →
  meeting_count = 0 :=
by
  sorry

end michael_and_truck_never_meet_l100_100805


namespace solve_ZAMENA_l100_100976

noncomputable def ZAMENA : ℕ := 541234

theorem solve_ZAMENA :
  ∃ (A M E H : ℕ), 
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H ∧
    ZAMENA = 541234 :=
by {
  have h1 : ∃ (A M E H : ℕ), 
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H, sorry,
  exact ⟨2, 1, 2, 3, h1⟩,
  repeat { sorry }
}

end solve_ZAMENA_l100_100976


namespace probability_twelfth_roll_last_l100_100724

theorem probability_twelfth_roll_last :
  let die := {1, 2, 3, 4, 5, 6, 7, 8} in
  let outcome_space := finset.univ : finset (fin 8) in
  let first_roll_prob := 1 in
  let subsequent_roll_prob := (7/8)^(11-1) in
  let final_roll_prob := 1/8 in
  let total_prob := first_roll_prob * subsequent_roll_prob * final_roll_prob in
  total_prob = 282475249 / 8589934592 :=
by
  sorry

end probability_twelfth_roll_last_l100_100724


namespace eccentricity_calculation_l100_100671

noncomputable def eccentricity_of_ellipse (a b c : ℝ) (ha : a > 2) (hb : b = 2) (hc : c = 2) : ℝ :=
  let e := c / (Real.sqrt (b^2 + c^2))
  e

theorem eccentricity_calculation :
  eccentricity_of_ellipse 2 (Real.sqrt 2) 2 (by linarith) rfl rfl = (Real.sqrt 2) / 2 :=
by
  sorry

end eccentricity_calculation_l100_100671


namespace geometric_sequence_expression_value_l100_100736

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_expression_value
  (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, 0 < a n)
  (h3 : a 3 = sqrt 2 - 1)
  (h5 : a 5 = sqrt 2 + 1) :
  a 3 ^ 2 + 2 * a 2 * a 6 + a 3 * a 7 = 8 :=
sorry

end geometric_sequence_expression_value_l100_100736


namespace max_distance_from_point_to_line_l100_100844

def line (λ : ℝ) : ℝ × ℝ → Prop :=
  fun p => (1 + 3 * λ) * p.1 + (1 + λ) * p.2 - 2 - 4 * λ = 0

def point := (-2 : ℝ, -1 : ℝ)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem max_distance_from_point_to_line : 
  ∀ (λ : ℝ), 
  ∃ A : ℝ × ℝ, 
  (∀ P : ℝ × ℝ, line λ A) ∧ 
  (distance point A = real.sqrt 13) := 
sorry

end max_distance_from_point_to_line_l100_100844


namespace compare_a_b_c_l100_100642

def a : ℝ := Real.log 0.7 / Real.log 2
def b : ℝ := 0.7 ^ 2
def c : ℝ := 2 ^ 0.3

theorem compare_a_b_c : a < b ∧ b < c := by
  sorry

end compare_a_b_c_l100_100642


namespace tan_45_deg_l100_100065

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100065


namespace max_value_of_f_l100_100620

-- Define the function f(x) = 6 * sin x + 8 * cos x
def f (x : ℝ) : ℝ := 6 * Real.sin x + 8 * Real.cos x

-- The theorem to prove that the maximum value of f(x) is 10
theorem max_value_of_f : ∃ x : ℝ, f x = 10 :=
sorry

end max_value_of_f_l100_100620


namespace experts_win_probability_l100_100220

noncomputable def probability_of_experts_winning (p : ℝ) (q : ℝ) (needed_expert_wins : ℕ) (needed_audience_wins : ℕ) : ℝ :=
  p ^ 4 + 4 * (p ^ 3 * q)

-- Probability values
def p : ℝ := 0.6
def q : ℝ := 1 - p

-- Number of wins needed
def needed_expert_wins : ℕ := 3
def needed_audience_wins : ℕ := 2

theorem experts_win_probability :
  probability_of_experts_winning p q needed_expert_wins needed_audience_wins = 0.4752 :=
by
  -- Proof would go here
  sorry

end experts_win_probability_l100_100220


namespace tan_45_eq_1_l100_100201

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100201


namespace complex_simplification_l100_100417

theorem complex_simplification : (4 - 3 * Complex.i)^2 = 7 - 24 * Complex.i := 
  sorry

end complex_simplification_l100_100417


namespace experts_eventual_win_probability_l100_100236

noncomputable def experts_win_probability (p : ℝ) (q : ℝ) (experts_needed : ℕ) (audience_needed : ℕ) (max_rounds : ℕ) : ℝ :=
  let e_all_wins := p ^ max_rounds
  let e_three_wins_one_loss := (Nat.choose max_rounds (max_rounds - 1)) * (p ^ (max_rounds - 1)) * q
  e_all_wins + e_three_wins_one_loss

theorem experts_eventual_win_probability :
  ∀ (p : ℝ) (q : ℝ),
  p = 0.6 →
  q = 1 - p →
  experts_win_probability p q 3 2 4 = 0.4752 := by
  intros p q hp hq
  have h : experts_win_probability p q 3 2 4 = 0.6 ^ 4 + 4 * (0.6 ^ 3) * 0.4 := sorry
  rw [hp, hq] at h
  norm_num at h
  exact h

end experts_eventual_win_probability_l100_100236


namespace tan_45_deg_eq_one_l100_100118

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100118


namespace tan_of_45_deg_l100_100092

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100092


namespace distance_between_points_distance_meeting_point_to_midpoint_l100_100812

def person_travel (speed min : ℕ) : ℕ := speed * min

def total_distance (speed_a speed_b min: ℕ) : ℕ := 
  person_travel speed_a min + person_travel speed_b min

def midpoint_distance (total_d: ℕ) : ℕ := total_d / 2

def meeting_point_to_midpoint (distance_midpoint distance_甲: ℕ) : ℕ := 
  abs (distance_midpoint - distance_甲)

theorem distance_between_points
  (speed_甲 speed_乙 time: ℕ) : total_distance speed_甲 speed_乙 time = 1200 := by
  sorry

theorem distance_meeting_point_to_midpoint
  (speed_甲 speed_乙 time: ℕ) : 
    meeting_point_to_midpoint (midpoint_distance (total_distance speed_甲 speed_乙 time))
                               (person_travel speed_甲 time) = 50 := by
  sorry

end distance_between_points_distance_meeting_point_to_midpoint_l100_100812


namespace prove_ZAMENA_l100_100958

theorem prove_ZAMENA :
  ∃ (A M E H Z : ℕ),
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ Z ∧ Z ≤ 5 ∧
    (3 > A) ∧ (A > M) ∧ (M < E) ∧ (E < H) ∧ (H < A) ∧
    (A ≠ M) ∧ (A ≠ E) ∧ (A ≠ H) ∧ (A ≠ Z) ∧
    (M ≠ E) ∧ (M ≠ H) ∧ (M ≠ Z) ∧
    (E ≠ H) ∧ (E ≠ Z) ∧
    (H ≠ Z) ∧
    Z = 5 ∧
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234 :=
begin
  sorry
end

end prove_ZAMENA_l100_100958


namespace length_A_l100_100388

open Real

theorem length_A'B'_correct {A B C A' B' : ℝ × ℝ} :
  A = (0, 10) →
  B = (0, 15) →
  C = (3, 9) →
  (A'.1 = A'.2) →
  (B'.1 = B'.2) →
  (C.2 - A.2) / (C.1 - A.1) = ((B.2 - C.2) / (B.1 - C.1)) →
  (dist A' B') = 2.5 * sqrt 2 :=
by
  intros
  sorry

end length_A_l100_100388


namespace parallelogram_sum_l100_100526

-- Define the points in the coordinate plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a specific parallelogram given its vertices
def ParallelogramSum (A B C D : Point) : ℝ :=
  let side_length := Math.sqrt ((C.x - A.x)^2 + (C.y - A.y)^2)
  let base := B.x - A.x
  let height := C.y - A.y
  let perimeter := 2 * base + 2 * side_length
  let area := base * height
  perimeter + area

-- Statement of the theorem to be proved
theorem parallelogram_sum (A B C D : Point) 
  (hA : A = ⟨2, 1⟩)
  (hB : B = ⟨7, 1⟩)
  (hC : C = ⟨5, 6⟩)
  (hD : D = ⟨10, 6⟩)
  : ParallelogramSum A B C D = 35 + 2 * Real.sqrt 34 := by
  -- Skip the proof
  sorry

end parallelogram_sum_l100_100526


namespace tan_of_45_deg_l100_100079

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100079


namespace multiple_of_n_eventually_written_l100_100362

theorem multiple_of_n_eventually_written (a b n : ℕ) (h_a_pos: 0 < a) (h_b_pos: 0 < b)  (h_ab_neq: a ≠ b) (h_n_pos: 0 < n) :
  ∃ m : ℕ, m % n = 0 :=
sorry

end multiple_of_n_eventually_written_l100_100362


namespace tan_45_deg_l100_100053

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100053


namespace sandy_spent_on_shirt_l100_100414

-- Define the conditions
def cost_of_shorts : ℝ := 13.99
def cost_of_jacket : ℝ := 7.43
def total_spent_on_clothes : ℝ := 33.56

-- Define the amount spent on the shirt
noncomputable def cost_of_shirt : ℝ :=
  total_spent_on_clothes - (cost_of_shorts + cost_of_jacket)

-- Prove that Sandy spent $12.14 on the shirt
theorem sandy_spent_on_shirt : cost_of_shirt = 12.14 :=
by
  sorry

end sandy_spent_on_shirt_l100_100414


namespace fruit_seller_price_l100_100914

theorem fruit_seller_price (CP SP : ℝ) (h1 : SP = 0.90 * CP) (h2 : 1.10 * CP = 13.444444444444445) : 
  SP = 11 :=
sorry

end fruit_seller_price_l100_100914


namespace curious_numbers_count_eq_30_l100_100802

-- Define what it means to be a curious number
def is_curious (n : ℕ) : Prop :=
  let sum_digits := (n / 100) + (n % 100 / 10) + (n % 10)
  ∃ d : ℕ, d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ (n - sum_digits) = d * 111

-- Define the range of three-digit numbers
def three_digit_numbers : list ℕ := list.filter (λ n, n ≥ 100 ∧ n < 1000) (list.range 1000)

-- Define the count of curious numbers in the range of three-digit numbers
def count_three_digit_curious_numbers : ℕ :=
  list.length (list.filter is_curious three_digit_numbers)

theorem curious_numbers_count_eq_30 : count_three_digit_curious_numbers = 30 := by
  sorry

end curious_numbers_count_eq_30_l100_100802


namespace tan_of_45_deg_l100_100074

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100074


namespace tan_45_deg_eq_one_l100_100154

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100154


namespace greatest_whole_number_solution_l100_100604

theorem greatest_whole_number_solution : ∃ k : ℕ, 5 * k - 4 < 3 - 2 * k ∧ ∀ m : ℕ, (m > k → ¬ (5 * m - 4 < 3 - 2 * m)) :=
by
  exists 0
  split
  {
    rw [nat.cast_zero, mul_zero, sub_zero, zero_sub, neg_lt, neg_zero]
    exact zero_lt_three
  }
  {
    intros m hm
    rw [nat.not_lt]
    exact nat.le_of_lt_succ hm
  }

end greatest_whole_number_solution_l100_100604


namespace initials_vowel_probability_l100_100733

theorem initials_vowel_probability :
  let vowels := {'A', 'E', 'I', 'O', 'U', 'Y', 'W'}
  let total_initial_pairs := 26
  let vowel_initial_pairs := 7
  let probability := (vowel_initial_pairs : ℚ) / total_initial_pairs
  probability = 7 / 26 :=
by
  sorry

end initials_vowel_probability_l100_100733


namespace decreasing_interval_of_f_l100_100845

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ (x ^ 2) - 9

theorem decreasing_interval_of_f :
  ∃ a b : ℝ, (0 < a) → (f is_strict_anti_on a b) → (a, b) = (0, +∞) :=
  sorry

end decreasing_interval_of_f_l100_100845


namespace complex_square_simplification_l100_100418

theorem complex_square_simplification : (4 - 3 * complex.I) ^ 2 = 7 - 24 * complex.I :=
by
  sorry

end complex_square_simplification_l100_100418


namespace probability_of_7_tails_l100_100542

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

def P (n k : ℕ) (p q : ℚ) : ℚ :=
  binom n k * p^k * q^(n-k)

theorem probability_of_7_tails :
  P 10 7 (2/3) (1/3) = 5120 / 19683 := sorry

end probability_of_7_tails_l100_100542


namespace problem_statement_l100_100716

theorem problem_statement (x : ℝ) (h : √(10 + x) + √(30 - x) = 8) : 
  (10 + x) * (30 - x) = 144 := 
  sorry

end problem_statement_l100_100716


namespace larger_exceeds_smaller_by_16_l100_100458

-- Define the smaller number S and the larger number L in terms of the ratio 7:11
def S : ℕ := 28
def L : ℕ := (11 * S) / 7

-- State the theorem that the larger number exceeds the smaller number by 16
theorem larger_exceeds_smaller_by_16 : L - S = 16 :=
by
  -- Proof steps will go here
  sorry

end larger_exceeds_smaller_by_16_l100_100458


namespace probability_of_experts_winning_l100_100229

-- Definitions required from the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p
def current_expert_score : ℕ := 3
def current_audience_score : ℕ := 4

-- The main theorem to state
theorem probability_of_experts_winning : 
  p^4 + 4 * p^3 * q = 0.4752 := 
by sorry

end probability_of_experts_winning_l100_100229


namespace tan_45_deg_eq_one_l100_100123

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100123


namespace find_greater_number_l100_100455

theorem find_greater_number (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 6) (h3 : x * y = 216) (h4 : x > y) : x = 18 := 
sorry

end find_greater_number_l100_100455


namespace second_sunday_l100_100741

theorem second_sunday {month : Type} [semigroup month] 
  (even_wednesdays : set ℕ)
  (three_wednesdays_on_even_dates : even_wednesdays = {2, 16, 30}) : 
  ∃ (second_sunday_day : ℕ), second_sunday_day = 13 :=
by
  -- Conditions: ⋃ ∅ = {2, 16, 30}. Hence, checking 2, 9, 16, 23, and 30 fitting for Wednesdays, 
  -- which comprise three even dates among these Wednesdays. Starting from here, count the Sundays 
  -- accordingly to find 6 and 13.
  sorry

end second_sunday_l100_100741


namespace tan_45_deg_l100_100068

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100068


namespace digit_410_of_7_over_29_l100_100476

noncomputable def decimal_expansion_of_7_over_29 : ℚ := 7 / 29

noncomputable def repeating_cycle_length_7_over_29 : ℕ := 28

noncomputable def repeating_cycle_7_over_29 : list ℕ :=
  [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7]

theorem digit_410_of_7_over_29 :
  (repeating_cycle_7_over_29[(410 % repeating_cycle_length_7_over_29) - 1] = 8) :=
by {
  -- Translate the position into the repeating cycle
  have h : 410 % 28 = 22, 
  {exact nat.mod_eq_of_lt 22 dec_trivial,},
  -- Check that the 22nd digit in the repeating cycle is 8
  exact sorry,
}

end digit_410_of_7_over_29_l100_100476


namespace segment_length_eq_3sqrt13_l100_100703

theorem segment_length_eq_3sqrt13 : 
  let p1 := (1 : ℝ, 2 : ℝ)
  let p2 := (10 : ℝ, 8 : ℝ)
  dist p1 p2 = 3 * Real.sqrt 13 :=
by
  let p1 := (1 : ℝ, 2 : ℝ)
  let p2 := (10 : ℝ, 8 : ℝ)
  sorry

end segment_length_eq_3sqrt13_l100_100703


namespace tan_45_deg_eq_one_l100_100026

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100026


namespace intersection_A_B_l100_100785

def A := {1, 2, 4}
def f (x : ℝ) := Real.log x / Real.log 2 -- defining the log base 2

noncomputable def B := Set.image f A

theorem intersection_A_B : (A ∩ B) = {1, 2} := by
  -- introduce the sets and the function
  sorry

end intersection_A_B_l100_100785


namespace tan_45_deg_eq_one_l100_100103

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100103


namespace solve_ZAMENA_l100_100977

noncomputable def ZAMENA : ℕ := 541234

theorem solve_ZAMENA :
  ∃ (A M E H : ℕ), 
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H ∧
    ZAMENA = 541234 :=
by {
  have h1 : ∃ (A M E H : ℕ), 
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H, sorry,
  exact ⟨2, 1, 2, 3, h1⟩,
  repeat { sorry }
}

end solve_ZAMENA_l100_100977


namespace tan_45_deg_eq_one_l100_100174

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100174


namespace tan_45_deg_eq_one_l100_100169

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100169


namespace chord_of_ellipse_through_point_l100_100296

theorem chord_of_ellipse_through_point (M : ℝ × ℝ) (M_inside : (M = (2, 1)) ∧ (M.1^2 / 16 + M.2^2 / 4 < 1)) :
  ∃ k : ℝ, line_eqn : ℝ × ℝ → Prop, 
  (∀ (P : ℝ × ℝ), line_eqn P ↔ P.2 - 1 = k * (P.1 - 2)) ∧ 
  (line_eqn (2, 1)) ∧ 
  (∀ (P: ℝ × ℝ), (P.1^2 / 16 + P.2^2 / 4 = 1 → line_eqn P)) 
  ∧ 
  (∀ (x1 x2: ℝ) (y1 y2: ℝ), 
    x1^2 / 16 + y1^2 / 4 = 1 ∧ x2^2 / 16 + y2^2 / 4 = 1 
    ∧ k = -1/2 → 
      ∃ line_eqn, (∀ (P : ℝ × ℝ), line_eqn P ↔ P.2 - 1 = (-1/2) * (P.1 - 2)) 
      ∧ (line_eqn (2, 1)) 
      ∧ (∀ (P: ℝ × ℝ), (P.1^2 / 16 + P.2^2 / 4 = 1 → line_eqn P)) 
      ∧ line_eqn = (λ P, P.1 + 2 * P.2 - 4 = 0)
  ) := 
sorry

end chord_of_ellipse_through_point_l100_100296


namespace bees_directions_at_15_feet_l100_100466

structure Position where
  x : ℝ
  y : ℝ
  z : ℝ

def moveA (pos : Position) (cycle : ℕ) : Position :=
  let cycles = cycle / 3
  let rem = cycle % 3
  let pos' := { pos with x := pos.x + cycles, y := pos.y + cycles, z := pos.z + (cycles * 2)}
  match rem with
  | 0 => pos'
  | 1 => { pos' with x := pos'.x + 1 }
  | 2 => { pos' with y := pos'.y + 1 }
  | _ => pos' -- This case never happens

def moveB (pos : Position) (cycle : ℕ) : Position :=
  let cycles = cycle / 2
  let rem = cycle % 2
  let pos' := { pos with x := pos.x - (cycles * 1), y := pos.y - (cycles * 1.5)}
  match rem with
  | 0 => pos'
  | 1 => { pos' with x := pos'.x - 1, y := pos'.y - 1.5 }
  | _ => pos' -- This case never happens

def dist (p1 p2 : Position) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem bees_directions_at_15_feet :
  ∃ cycle : ℕ, (dist (moveA ⟨0, 0, 0⟩ cycle) (moveB ⟨0, 0, 0⟩ cycle) = 15) ∧
  (match cycle % 3 with
   | 0 => (∃ dA dB, dA = "east" ∧ dB = "west")
   | _ => False) :=
by
  sorry

end bees_directions_at_15_feet_l100_100466


namespace greatest_area_difference_l100_100205

theorem greatest_area_difference 
    (a b c d : ℕ) 
    (H1 : 2 * (a + b) = 100)
    (H2 : 2 * (c + d) = 100)
    (H3 : ∀i j : ℕ, 2 * (i + j) = 100 → i * j ≤ a * b)
    : 373 ≤ a * b - (c * d) := 
sorry

end greatest_area_difference_l100_100205


namespace triangular_scarf_proportions_l100_100370

-- Let the total area of the original square be 1 unit.
constant original_area : ℝ := 1

-- The given conditions
constant black_fraction : ℝ := 1 / 6
constant gray_fraction : ℝ := 1 / 3

-- Derived areas from the given fractions
constant black_area : ℝ := black_fraction * original_area
constant gray_area : ℝ := gray_fraction * original_area
constant white_area : ℝ := original_area - black_area - gray_area

theorem triangular_scarf_proportions :
  ∀ (triangle_area := original_area / 2),
  triangle_area * black_fraction / original_area * 2 = 1 / 12 ∧
  triangle_area * gray_fraction / original_area * 2 = 1 / 6 ∧
  triangle_area * white_area / original_area * 1 = 1 / 4 :=
by
  sorry

end triangular_scarf_proportions_l100_100370


namespace total_weight_30_boxes_proof_profit_proof_l100_100522

section AppleStore

-- Define the conditions
def purchase_price_per_box := 60 -- yuan
def total_boxes := 400
def selling_price_per_kg := 10 -- yuan

-- Weighing records of 30 boxes
def weight_diff := [-0.2, -0.1, 0, 0.1, 0.2, 0.5]
def num_boxes := [5, 8, 2, 6, 8, 1]

-- Calculate the total weight of 30 boxes
def total_weight_of_30_boxes := 
  30 * 10 + (5 * -0.2 + 8 * -0.1 + 2 * 0 + 6 * 0.1 + 8 * 0.2 + 1 * 0.5)

-- Calculate the average weight per box
def avg_weight_per_box := total_weight_of_30_boxes / 30
  
-- Calculate the total weight using average weight per box
def total_weight_of_batch := avg_weight_per_box * total_boxes

-- Calculate the revenue from selling the batch
def revenue_from_batch := total_weight_of_batch * selling_price_per_kg

-- Calculate the cost of purchasing the batch
def cost_of_purchase := purchase_price_per_box * total_boxes

-- Calculate the profit
def profit := revenue_from_batch - cost_of_purchase

-- Prove that the total weight of the 30 boxes is 300.9 kg
theorem total_weight_30_boxes_proof : total_weight_of_30_boxes = 300.9 :=
by sorry

-- Prove that the profit from selling the 400 boxes is 16120 yuan
theorem profit_proof : profit = 16120 :=
by sorry

end AppleStore

end total_weight_30_boxes_proof_profit_proof_l100_100522


namespace socorro_training_days_l100_100428

def total_hours := 5
def minutes_per_hour := 60
def total_training_minutes := total_hours * minutes_per_hour

def minutes_multiplication_per_day := 10
def minutes_division_per_day := 20
def daily_training_minutes := minutes_multiplication_per_day + minutes_division_per_day

theorem socorro_training_days:
  total_training_minutes / daily_training_minutes = 10 :=
by
  -- proof omitted
  sorry

end socorro_training_days_l100_100428


namespace F_is_decreasing_l100_100286

variable {f : ℝ → ℝ}
variable (hf : ∀ x y, x < y → f(x) ≤ f(y))

def F (x : ℝ) : ℝ := f (1 - x) - f (1 + x)

theorem F_is_decreasing : ∀ x y, x < y → F x ≥ F y :=
by 
  sorry

end F_is_decreasing_l100_100286


namespace point_on_circle_l100_100644

theorem point_on_circle : 
  ∀ (P : ℝ × ℝ), (P = (-3, 4)) → (dist (0, 0) P = 5) → (dist (0, 0) P = 5)
:= by
  intros P H1 H2
  rw H1 at H2
  rw dist_eq_norm at H2
  simp at H2
  exact H2

end point_on_circle_l100_100644


namespace ratio_of_blackberries_is_one_third_l100_100513

-- Definitions of conditions
def total_berries : ℕ := 42
def raspberries : ℕ := total_berries / 2
def blueberries : ℕ := 7
def blackberries : ℕ := total_berries - raspberries - blueberries

-- Theorem statement
theorem ratio_of_blackberries_is_one_third : blackberries.to_rat / total_berries.to_rat = (1 : ℚ) / 3 := by
  sorry

end ratio_of_blackberries_is_one_third_l100_100513


namespace perimeter_of_square_l100_100435

theorem perimeter_of_square (s : ℝ) (h : s^2 = 588) : (4 * s) = 56 * Real.sqrt 3 :=
by
  sorry

end perimeter_of_square_l100_100435


namespace problem_1_problem_2_problem_3_problem_4_l100_100556

theorem problem_1 : -4 + (-16) - (-3) = -17 := by
  sorry

theorem problem_2 : (- (1 / 2) + 1 / 6 - 3 / 8) * (-24) = 1 := by
  sorry

theorem problem_3 : 2 * (-5) + 2^2 - 3 / (1 / 2) = -12 := by
  sorry

theorem problem_4 : -1^4 - | -10 | / (1 / 2) * 2 + (-4)^2 = -25 := by
  sorry

end problem_1_problem_2_problem_3_problem_4_l100_100556


namespace maximum_value_f_l100_100613

def f (x : ℝ) : ℝ := 6 * Real.sin x + 8 * Real.cos x

theorem maximum_value_f : ∃ y : ℝ, ∀ x : ℝ, f(x) ≤ y ∧ y = 10 :=
sorry

end maximum_value_f_l100_100613


namespace tan_of_45_deg_l100_100093

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100093


namespace order_of_numbers_l100_100448

theorem order_of_numbers (x y z : ℝ) (h1 : 7^0.2 > 1) (h2 : 0 < 0.2^7) (h3 : 0.2^7 < 1) (h4 : log 0.2 < 0) :
  (7^0.2 > 0.2^7 ∧ 0.2^7 > log 0.2) :=
by
  sorry

end order_of_numbers_l100_100448


namespace remainder_fraction_l100_100988

def remainder (x y : ℝ) : ℝ :=
  x - y * Real.floor (x / y)

theorem remainder_fraction :
  remainder (5/13) (7/9) = 5/13 :=
by
  sorry

end remainder_fraction_l100_100988


namespace binomial_coefficient_10_5_l100_100985

open Nat

theorem binomial_coefficient_10_5 : binomial 10 5 = 252 := by
  sorry

end binomial_coefficient_10_5_l100_100985


namespace girls_joined_school_l100_100738

theorem girls_joined_school
  (initial_girls : ℕ)
  (initial_boys : ℕ)
  (total_pupils_after : ℕ)
  (computed_new_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  total_pupils_after = 1346 →
  computed_new_girls = total_pupils_after - (initial_girls + initial_boys) →
  computed_new_girls = 418 :=
by
  intros h_initial_girls h_initial_boys h_total_pupils_after h_computed_new_girls
  sorry

end girls_joined_school_l100_100738


namespace zamena_inequalities_l100_100963

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end zamena_inequalities_l100_100963


namespace problem_statement_l100_100717

theorem problem_statement (x : ℝ) (h : √(10 + x) + √(30 - x) = 8) : 
  (10 + x) * (30 - x) = 144 := 
  sorry

end problem_statement_l100_100717


namespace tangent_lines_pass_through_fixed_point_area_of_quadrilateral_ADBE_l100_100295

noncomputable def quadratic_curve (x : ℝ) : ℝ :=
  x^2 / 2

def line_y_neg_half (x : ℝ) : ℝ :=
  -1 / 2

def tangent_at_point (x1 : ℝ) (x : ℝ) : ℝ :=
  x1 * x - x1^2 / 2

def fixed_point : ℝ × ℝ :=
  (0, 1 / 2)

theorem tangent_lines_pass_through_fixed_point :
  ∀ x1 x2 : ℝ, tangent_at_point x1 0 = 1 / 2 ∧ tangent_at_point x2 0 = 1 / 2 → 
  (quadratic_curve x1 = quadratic_curve x1 ∧ quadratic_curve x2 = quadratic_curve x2 ∧ 
  (∃ y : ℝ, tangent_at_point x1 y = -1 / 2 ∧ tangent_at_point x2 y = -1 / 2 )) →
  ∃ x : ℝ, x = fixed_point.1 := sorry

noncomputable def circle_center : ℝ × ℝ :=
  (0, 5 / 2)

def area_quadrilateral (t : ℝ) : ℝ :=
  if t = 0 then 3
  else if t = 1 ∨ t = -1 then 4 * real.sqrt 2
  else 0

theorem area_of_quadrilateral_ADBE :
  ∀ t : ℝ, ∃ x1 x2,
  quadratic_curve x1 = 1 / 2 * (x1 + x2) ∧
  quadratic_curve x2 = 1 / 2 * (x1 + x2) ∧
  area_quadrilateral t ∈ ({3, 4 * real.sqrt 2} : set ℝ) := sorry

end tangent_lines_pass_through_fixed_point_area_of_quadrilateral_ADBE_l100_100295


namespace maximum_value_f_l100_100614

def f (x : ℝ) : ℝ := 6 * Real.sin x + 8 * Real.cos x

theorem maximum_value_f : ∃ y : ℝ, ∀ x : ℝ, f(x) ≤ y ∧ y = 10 :=
sorry

end maximum_value_f_l100_100614


namespace tan_45_deg_eq_one_l100_100164

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100164


namespace log10_cubic_solution_l100_100289

noncomputable def log10 (x: ℝ) : ℝ := Real.log x / Real.log 10

open Real

theorem log10_cubic_solution 
  (x : ℝ) 
  (hx1 : x < 1) 
  (hx2 : (log10 x)^3 - log10 (x^4) = 640) : 
  (log10 x)^4 - log10 (x^4) = 645 := 
by 
  sorry

end log10_cubic_solution_l100_100289


namespace tan_45_deg_l100_100140

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100140


namespace units_digit_sum_squares_1001_odds_l100_100497

theorem units_digit_sum_squares_1001_odds :
  (Σ (n : ℕ) in Finset.range 1001, (2 * n + 1)^2) % 10 = 1 :=
sorry

end units_digit_sum_squares_1001_odds_l100_100497


namespace simplify_cubic_root_l100_100820

theorem simplify_cubic_root (a b c : ℕ) (h1 : a = 20) (h2 : b = 70) (h3 : c = 110) :
  (∛(a^3 + b^3 + c^3) = 120) :=
  by
    subst h1
    subst h2
    subst h3
    -- here we skip the proof with sorry
    sorry

end simplify_cubic_root_l100_100820


namespace tan_45_deg_l100_100136

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100136


namespace expected_value_inequalities_expected_value_X1_expected_value_Sm_Sn_inv_expected_value_Sn_Sm_inv_l100_100791

noncomputable def X (n : ℕ) : ℕ → ℝ := sorry -- you need to define this function based on the conditions given

variable {m n : ℕ}

def E_X1 : ℝ := sorry -- expectation of X_1
def E_X1_inv : ℝ := sorry -- expectation of X_1^{-1}

def S (k : ℕ) : ℝ := ∑ i in finset.range k, (X n i)

axiom conditions : 
  (∀ i, X n i > 0) ∧ 
  (∀ i j, i ≠ j → X n i ∘ X n j = 0) ∧ 
  E_X1 < ∞ ∧ 
  E_X1_inv < ∞

theorem expected_value_inequalities :
  (1 : ℝ) / n * E_X1_inv ≥ E_X1 := sorry

theorem expected_value_X1 :
  E_X1 * (1 / n) = 1 / n := sorry

theorem expected_value_Sm_Sn_inv (h : m ≤ n) :
  (E (S m * (1 / S n))) = m / n := sorry

theorem expected_value_Sn_Sm_inv (h : m ≤ n) :
  (E (S n * (1 / S m))) = 1 + (n - m) * E_X1 * (1 / S m) := sorry

end expected_value_inequalities_expected_value_X1_expected_value_Sm_Sn_inv_expected_value_Sn_Sm_inv_l100_100791


namespace hyperbola_has_given_equation_l100_100345

noncomputable def hyperbola_eq (x y : ℝ) : Prop :=
  x ^ 2 - (y ^ 2) / 4 = 1

theorem hyperbola_has_given_equation :
  (∀ (x : ℝ), y = 2 * x ∨ y = -2 * x) ∧
  (∃ (x : ℝ) (y : ℝ), x = sqrt 2 ∧ y = 2 ∧ hyperbola_eq x y) →
  (∀ x y : ℝ, hyperbola_eq x y → x ^ 2 - (y ^ 2) / 4 = 1) :=
by
  sorry

end hyperbola_has_given_equation_l100_100345


namespace daniel_total_worth_l100_100564

theorem daniel_total_worth
    (sales_tax_paid : ℝ)
    (sales_tax_rate : ℝ)
    (cost_tax_free_items : ℝ)
    (tax_rate_pos : 0 < sales_tax_rate) :
    sales_tax_paid = 0.30 →
    sales_tax_rate = 0.05 →
    cost_tax_free_items = 18.7 →
    ∃ (x : ℝ), 0.05 * x = 0.30 ∧ (x + cost_tax_free_items = 24.7) := by
    sorry

end daniel_total_worth_l100_100564


namespace tan_45_deg_eq_one_l100_100039

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100039


namespace robinson_family_children_count_l100_100824

theorem robinson_family_children_count 
  (m : ℕ) -- mother's age
  (f : ℕ) (f_age : f = 50) -- father's age is 50
  (x : ℕ) -- number of children
  (y : ℕ) -- average age of children
  (h1 : (m + 50 + x * y) / (2 + x) = 22)
  (h2 : (m + x * y) / (1 + x) = 18) :
  x = 6 := 
sorry

end robinson_family_children_count_l100_100824


namespace max_triangle_area_AB_l100_100272

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def eccentricity (a c : ℝ) : ℝ :=
  c / a

theorem max_triangle_area_AB (a b c : ℝ) :
  (a > 0) ∧ (b > 0) ∧ (a > b) ∧ (eccentricity a c = sqrt 2 / 2) ∧ ((2 * b^2 / a = sqrt 2)) ∧ (a^2 - b^2 = c^2) →
  (ellipse_equation a b x y) →
  ellipse_equation a b x y = ellipse_equation (sqrt 2) 1 x y ∧
  (∀ (k : ℝ) (P : ℝ × ℝ) (A B : ℝ × ℝ) (O : ℝ × ℝ), P.1 = 0 ∧ P.2 = 2 ∧ 
  (∃ l : ℝ, y = k * x + 2 ∧ (k^2 > 3/2)) ∧
  (A.1, A.2) ≠ (B.1, B.2) →
  |A.2 - B.2| = 3 / 2) :=
by
  sorry

end max_triangle_area_AB_l100_100272


namespace tan_45_deg_l100_100147

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100147


namespace missing_shirts_l100_100810

-- Definition of conditions
def pairs_of_trousers : ℕ := 10
def price_per_trousers : ℕ := 9
def total_cost : ℕ := 140
def price_per_shirt : ℕ := 5
def claimed_number_of_shirts : ℕ := 2

-- The theorem to be proved
theorem missing_shirts : 
  let cost_of_trousers := pairs_of_trousers * price_per_trousers,
      cost_of_shirts := total_cost - cost_of_trousers,
      number_of_shirts := cost_of_shirts / price_per_shirt,
      missing_shirts := number_of_shirts - claimed_number_of_shirts
  in
  missing_shirts = 8 :=
by
  sorry

end missing_shirts_l100_100810


namespace divisibility_of_polynomial_l100_100375

noncomputable def P (z : ℂ) : ℂ := sorry -- Assuming P(z) is given so we can define it rigorously later

theorem divisibility_of_polynomial (P : ℂ → ℂ) (h_degree : degree P = 1992)
  (h_distinct_zeros : ∃ S : set ℂ, S.card = 1992 ∧ ∀ z ∈ S, P z = 0) :
  ∃ (a : fin 1992 → ℂ), ∀ z : ℂ, P z = 0 → 
  let Q := (finset.range 1991).foldl (λ acc i, (acc (z - a ⟨i, nat.lt_of_lt_succ i.2⟩))^2 - a ⟨i + 1, nat.lt_of_succ_lt_succ i.2⟩) (λ x, (x - a ⟨0, sorry⟩)^2 - a ⟨1, sorry⟩)
  in Q z = 0 :=
begin
  sorry
end

end divisibility_of_polynomial_l100_100375


namespace punger_needs_pages_l100_100409

theorem punger_needs_pages (p c h : ℕ) (h_p : p = 60) (h_c : c = 7) (h_h : h = 10) : 
  (p * c) / h = 42 := 
by
  sorry

end punger_needs_pages_l100_100409


namespace speed_last_segment_l100_100767

-- Definitions corresponding to conditions
def drove_total_distance : ℝ := 150
def total_time_minutes : ℝ := 120
def time_first_segment_minutes : ℝ := 40
def speed_first_segment_mph : ℝ := 70
def speed_second_segment_mph : ℝ := 75

-- The statement of the problem
theorem speed_last_segment :
  let total_distance : ℝ := drove_total_distance
  let total_time : ℝ := total_time_minutes / 60
  let time_first_segment : ℝ := time_first_segment_minutes / 60
  let time_second_segment : ℝ := time_first_segment
  let time_last_segment : ℝ := time_first_segment
  let distance_first_segment : ℝ := speed_first_segment_mph * time_first_segment
  let distance_second_segment : ℝ := speed_second_segment_mph * time_second_segment
  let distance_two_segments : ℝ := distance_first_segment + distance_second_segment
  let distance_last_segment : ℝ := total_distance - distance_two_segments
  let speed_last_segment := distance_last_segment / time_last_segment
  speed_last_segment = 80 := 
  sorry

end speed_last_segment_l100_100767


namespace ratio_of_guesses_l100_100827

-- Definitions based on the problem statement.
def G1 : ℕ := 100
def G4 : ℕ := 525
def G2 := (λ G2 : ℕ, G4 = (G1 + G2 + (G2 - 200)) / 3 + 25)

-- The final statement we need to prove in Lean 4.
theorem ratio_of_guesses (guess_2 : ℕ) (h : G2 guess_2) : guess_2 / G1 = 8 := by
    sorry

end ratio_of_guesses_l100_100827


namespace max_value_of_quadratic_l100_100486

theorem max_value_of_quadratic : ∀ (x : ℝ), -9 * x^2 + 27 * x + 15 ≤ 141 / 4 :=
begin
  sorry
end

end max_value_of_quadratic_l100_100486


namespace sum_of_integers_l100_100328

open Classical

theorem sum_of_integers (a : ℤ) :
  (∀ y : ℚ, y - 1 ≥ (2 * y - 1) / 3 → y ≥ 2) →
  (∀ y : ℚ, - (y - a) / 2 > 0 → y < a) →
  (∀ x : ℚ, (a : ℚ) / (x + 1) + 1 = (x + a) / (x - 1) → x = -2 * a - 1 → x < 0) →
  (sum (filter (λ a, -1 / 2 < a ∧ a ≤ 2 ∧ a ≠ 0) (-2..3).to_list) = 3) :=
by
  sorry

end sum_of_integers_l100_100328


namespace tan_45_eq_1_l100_100187

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100187


namespace tan_45_deg_eq_one_l100_100173

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100173


namespace retailer_profit_percentage_l100_100891

-- Definitions based on conditions
def cost_price (market_price_per_pen : ℝ) : ℝ :=
  market_price_per_pen * 36 / 120

def selling_price_per_pen (market_price_per_pen : ℝ) : ℝ :=
  market_price_per_pen * 0.99

def selling_price (market_price_per_pen : ℝ) (quantity : ℕ) : ℝ :=
  (selling_price_per_pen market_price_per_pen) * quantity

def profit (market_price_per_pen : ℝ) : ℝ :=
  (selling_price market_price_per_pen 120) - (cost_price market_price_per_pen)

def profit_percentage (market_price_per_pen : ℝ) : ℝ :=
  (profit market_price_per_pen) / (cost_price market_price_per_pen) * 100

-- Statement to prove
theorem retailer_profit_percentage :
  profit_percentage 1 = 230 := 
sorry

end retailer_profit_percentage_l100_100891


namespace product_of_common_divisors_of_210_and_30_l100_100630

theorem product_of_common_divisors_of_210_and_30 : 
  (∏ d in ({d : ℤ | d ∣ 210 ∧ d ∣ 30}), d) = 324000000 := by
  sorry

end product_of_common_divisors_of_210_and_30_l100_100630


namespace equidistant_from_midpoint_l100_100467

noncomputable def midpoint (P Q : Point) : Point := 
  ⟨(P.1 + Q.1) / 2, (P.2 + Q.2) / 2⟩

variables
  {ω1 ω2 : Type} -- assume ω1 and ω2 are types representing the circles
  {O1 O2 A B C D : Point} -- assume Point is a type for geometric points
  (on_circle_ω1 : ω1 → Point → Prop) -- predicate indicating a point is on circle ω1
  (on_circle_ω2 : ω2 → Point → Prop) -- predicate indicating a point is on circle ω2

 -- conditions
  (H1 : on_circle_ω1 ω1 A)
  (H2 : on_circle_ω2 ω2 A)
  (H3 : on_circle_ω1 ω1 B)
  (H4 : on_circle_ω2 ω2 B)
  (H5 : on_circle_ω1 ω1 C)
  (H6 : on_circle_ω2 ω2 D)
  (H7 : equidistant_from_line AB C D)

-- Prove that C and D are equidistant from the midpoint of the segment O1 O2
theorem equidistant_from_midpoint : 
  let M := midpoint O1 O2 in 
  dist C M = dist D M :=
sorry

end equidistant_from_midpoint_l100_100467


namespace quadrilateral_area_is_10_5_l100_100568

variables {A : Type*} [field A]

structure Point (A : Type*) :=
(x : A)
(y : A)

def shoelace_area (P Q R S : Point A) : A :=
1 / 2 * abs ((P.x * Q.y + Q.x * R.y + R.x * S.y + S.x * P.y) - (Q.x * P.y + R.x * Q.y + S.x * R.y + P.x * S.y))

noncomputable def quadrilateral_area :=
shoelace_area ⟨2, 1⟩ ⟨1, 6⟩ ⟨5, 5⟩ ⟨7, 7⟩

theorem quadrilateral_area_is_10_5 : quadrilateral_area = 10.5 :=
by
sorry

end quadrilateral_area_is_10_5_l100_100568


namespace Jack_can_make_all_coins_T_l100_100372

-- Defining the conditions
def valid_n (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 2021

def initial_coins (coins : List Bool) (n : ℕ) : Prop :=
  coins.length = 2021 ∧ List.nth coins (n - 1) = some false ∧ (∀ m, m ≠ n - 1 → List.nth coins m = some true)

def operation (coins : List Bool) (pos : ℕ) : List Bool :=
  match coins with
  | [] => []
  | (h::t) =>
    match List.splitAt pos t with
    | (l, []) => (l ++ [!h])
    | (l, b::r) =>
      if h = false then
        (l ++ (h :: !b :: [!List.head r] ++ List.tail r))
      else
        (l ++ (h :: b :: r))

def can_turn_all_T (coins : List Bool) : Prop :=
  ∀ i, List.nth coins i = some false

def solve (n : ℕ) : Prop :=
  valid_n n ∧ ∃ coins : List Bool, initial_coins coins n ∧
    (∃ ops : list ℕ, (List.foldl operation coins ops  |> can_turn_all_T))

-- The proof statement
theorem Jack_can_make_all_coins_T (n : ℕ) : solve n ↔ n = 1011 :=
by
  sorry

end Jack_can_make_all_coins_T_l100_100372


namespace tan_45_deg_eq_one_l100_100120

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100120


namespace rosa_calls_pages_l100_100311

theorem rosa_calls_pages (pages_last_week : ℝ) (pages_this_week : ℝ) (h_last_week : pages_last_week = 10.2) (h_this_week : pages_this_week = 8.6) : pages_last_week + pages_this_week = 18.8 :=
by sorry

end rosa_calls_pages_l100_100311


namespace sqrt_sum_eq_8_l100_100714

theorem sqrt_sum_eq_8 (x : ℝ) (h : √(10 + x) + √(30 - x) = 8) : (10 + x) * (30 - x) = 144 :=
sorry

end sqrt_sum_eq_8_l100_100714


namespace symmetric_words_l100_100997

def symmetric_words_count : ℕ := 12

theorem symmetric_words :
  let positions_nat_total := 6
  let positions_n := 2
  let symmetric_property (s : String) : Prop := s = s.reverse
  (∃ (arrangements : Finset (Finset ℕ)). 
    arrangements.card = 3 
    ∧ (∀ w ∈ arrangements, symmetric_property w) 
    ∧ arrangements.card * 4 = symmetric_words_count) := sorry

end symmetric_words_l100_100997


namespace experts_win_probability_l100_100223

noncomputable def probability_of_experts_winning (p : ℝ) (q : ℝ) (needed_expert_wins : ℕ) (needed_audience_wins : ℕ) : ℝ :=
  p ^ 4 + 4 * (p ^ 3 * q)

-- Probability values
def p : ℝ := 0.6
def q : ℝ := 1 - p

-- Number of wins needed
def needed_expert_wins : ℕ := 3
def needed_audience_wins : ℕ := 2

theorem experts_win_probability :
  probability_of_experts_winning p q needed_expert_wins needed_audience_wins = 0.4752 :=
by
  -- Proof would go here
  sorry

end experts_win_probability_l100_100223


namespace cider_pints_produced_l100_100701

/-- Define the conditions as constants and parameters -/
def golden_apples_per_pint := 20
def pink_apples_per_pint := 40
def farmhands := 6
def apples_per_hour_per_farmhand := 240
def hours_worked := 5
def golden_to_pink_ratio (golden: ℕ) (pink: ℕ) := golden = 2 * pink

/-- Main theorem statement -/
theorem cider_pints_produced : 
  ∀ (golden picked_apples : ℕ),
  let total_apples := picked_apples * farmhands * hours_worked in
  let total_pints := total_apples / (golden_apples_per_pint + pink_apples_per_pint) in
  total_apples = 7200 → 
  total_pints = 120 :=
by
  intros golden picked_apples total_apples total_pints h_tot_apples
  sorry

end cider_pints_produced_l100_100701


namespace calculate_fraction_l100_100441

noncomputable def r (c : ℝ) (x : ℝ) := c * x * (x - 4)
noncomputable def s (x : ℝ) := (x - 4) * (x - 3)

theorem calculate_fraction (h4 : ∀ c, ∃ x, r c x = 0) 
  (hz : ∃ c, r c 0 = 0)
  (hasymp : ∀ c, ∀ x, ∃ y, r c x / s x = y → y = -2)
  (vasymp : ∀ c, ∃ x, s x = 0 → x = 3) :
  ∃ c, r c 1 / s 1 = 1 :=
by
  sorry

end calculate_fraction_l100_100441


namespace area_pentagon_eq_l100_100899

variables {A B C D E X Y : Type}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
variables [metric_space X] [metric_space Y]
variable [cyclic_pentagon A B C D E]
variable (h1 : distance A C = distance B D) (h2 : distance B D = distance C E)
variable (h3 : intersect A C B D = X)
variable (h4 : intersect B D C E = Y)
variable (h5 : distance A X = 6)
variable (h6 : distance X Y = 4)
variable (h7 : distance Y E = 7)

theorem area_pentagon_eq : 
  let a := 27 in 
  let b := 15 in 
  let c := 2 in 
  let area := (a * sqrt b) / c in 
  pentagon_area A B C D E = area ∧ 100 * a + 10 * b + c = 2852 :=
sorry

end area_pentagon_eq_l100_100899


namespace rectangle_opposite_sides_equal_square_all_sides_equal_l100_100737

-- Definition of Properties of a Rectangle
def is_rectangle (R : Type) [rect : Rect R] :=
  ∀ (a b : R), rect.oppositeSides a b → a = b

-- Definition of Properties of a Square
def is_square (S : Type) [sq : Square S] :=
  ∀ (a b c d : S), sq.allSidesEqual a b c d → (a = b ∧ b = c ∧ c = d)

-- Proving the properties
theorem rectangle_opposite_sides_equal (R : Type) [rect : Rect R] :
  is_rectangle R :=
sorry

theorem square_all_sides_equal (S : Type) [sq : Square S] :
  is_square S :=
sorry

end rectangle_opposite_sides_equal_square_all_sides_equal_l100_100737


namespace sum_first_12_terms_l100_100454

variable (S : ℕ → ℝ)

def sum_of_first_n_terms (n : ℕ) : ℝ :=
  S n

theorem sum_first_12_terms (h₁ : sum_of_first_n_terms 4 = 30) (h₂ : sum_of_first_n_terms 8 = 100) :
  sum_of_first_n_terms 12 = 210 := 
sorry

end sum_first_12_terms_l100_100454


namespace tan_45_deg_l100_100127

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100127


namespace maximum_value_of_function_y_l100_100304

noncomputable def function_y (x : ℝ) : ℝ :=
  x * (3 - 2 * x)

theorem maximum_value_of_function_y : ∃ (x : ℝ), 0 < x ∧ x ≤ 1 ∧ function_y x = 9 / 8 :=
by
  sorry

end maximum_value_of_function_y_l100_100304


namespace smallest_positive_integer_conditioned_l100_100491

theorem smallest_positive_integer_conditioned :
  ∃ a : ℕ, a > 0 ∧ (a % 4 = 3) ∧ (a % 3 = 2) ∧ ∀ b : ℕ, b > 0 ∧ (b % 4 = 3) ∧ (b % 3 = 2) → a ≤ b :=
begin
  sorry
end

end smallest_positive_integer_conditioned_l100_100491


namespace tan_45_eq_1_l100_100192

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100192


namespace tan_45_deg_eq_one_l100_100177

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100177


namespace greatest_whole_number_inequality_l100_100605

theorem greatest_whole_number_inequality :
  ∃ x : ℕ, (5 * x - 4 < 3 - 2 * x) ∧ ∀ y : ℕ, (5 * y - 4 < 3 - 2 * y → y ≤ x) :=
sorry

end greatest_whole_number_inequality_l100_100605


namespace tan_45_eq_1_l100_100196

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100196


namespace ZAMENA_correct_l100_100945

-- Define the digits and assigning them to the letters
noncomputable def A : ℕ := 2
noncomputable def M : ℕ := 1
noncomputable def E : ℕ := 2
noncomputable def H : ℕ := 3

-- Define the number ZAMENA
noncomputable def ZAMENA : ℕ :=
  5 * 10^5 + A * 10^4 + M * 10^3 + E * 10^2 + H * 10 + A

-- Define the inequalities and constraints
lemma satisfies_inequalities  : 3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Define uniqueness of each digit set
lemma unique_digits: A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ M ≠ E ∧ M ≠ H ∧ E ≠ H := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Prove that the number ZAMENA matches the specified number
theorem ZAMENA_correct : ZAMENA = 541234 := by
  unfold ZAMENA A M E H
  norm_num

end ZAMENA_correct_l100_100945


namespace john_money_left_l100_100368

noncomputable def total_earned (days_worked : ℕ) (earnings_per_day : ℕ) : ℕ := days_worked * earnings_per_day

noncomputable def total_spent (spent_books: ℕ) (spent_kaylee: ℕ) : ℕ := spent_books + spent_kaylee

def amount_left (total_earned : ℕ) (total_spent : ℕ) : ℕ := total_earned - total_spent

theorem john_money_left : 
  let days_worked := 26 in
  let earnings_per_day := 10 in
  let spent_books := 50 in
  let spent_kaylee := 50 in
  amount_left (total_earned days_worked earnings_per_day) (total_spent spent_books spent_kaylee) = 160 :=
by
  sorry

end john_money_left_l100_100368


namespace stratified_sampling_total_sample_size_l100_100918

-- Definitions based on conditions
def pure_milk_brands : ℕ := 30
def yogurt_brands : ℕ := 10
def infant_formula_brands : ℕ := 35
def adult_milk_powder_brands : ℕ := 25
def sampled_infant_formula_brands : ℕ := 7

-- The goal is to prove that the total sample size n is 20.
theorem stratified_sampling_total_sample_size : 
  let total_brands := pure_milk_brands + yogurt_brands + infant_formula_brands + adult_milk_powder_brands
  let sampling_fraction := sampled_infant_formula_brands / infant_formula_brands
  let pure_milk_samples := pure_milk_brands * sampling_fraction
  let yogurt_samples := yogurt_brands * sampling_fraction
  let adult_milk_samples := adult_milk_powder_brands * sampling_fraction
  let n := pure_milk_samples + yogurt_samples + sampled_infant_formula_brands + adult_milk_samples
  n = 20 :=
by
  sorry

end stratified_sampling_total_sample_size_l100_100918


namespace evaluate_polynomial_at_2_l100_100218

theorem evaluate_polynomial_at_2 : (2^4 + 2^3 + 2^2 + 2 + 2) = 32 := 
by
  sorry

end evaluate_polynomial_at_2_l100_100218


namespace distance_travelled_by_car_l100_100705

theorem distance_travelled_by_car : 
  let train_speed := 100 -- in miles per hour
  let car_speed := (2/3 : ℝ) * train_speed -- in miles per hour
  let time := (30 / 60 : ℝ) -- 30 minutes converted to hours
  car_speed * time = (100 / 3 : ℝ) :=
by {
  -- Provide the assumptions and apply the required calculation
  dsimp [car_speed, time],
  rw mul_comm,
  congr,
  norm_num,
}

end distance_travelled_by_car_l100_100705


namespace length_of_platform_l100_100502

theorem length_of_platform (length_of_train : ℕ) (speed_kmph : ℕ) (time_s : ℕ) (L : ℕ) :
  length_of_train = 160 → speed_kmph = 72 → time_s = 25 → (L = 340) :=
by
  sorry

end length_of_platform_l100_100502


namespace seq_strictly_increasing_l100_100646

noncomputable def a_seq (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2^(n-1)

noncomputable def b_seq (k : ℝ) (n : ℕ) : ℝ :=
  k * a_seq n - 2

noncomputable def S_seq (k : ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, b_seq k (i + 1)

theorem seq_strictly_increasing (k : ℝ) : (∀ n : ℕ, n > 0 → S_seq k (n+1) > S_seq k n) ↔ (k > 1 / 2) :=
sorry

end seq_strictly_increasing_l100_100646


namespace range_of_y0_l100_100656

noncomputable def hyperbola_h : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 2 - p.2^2 = 1}

theorem range_of_y0 (x_0 y_0 : ℝ) (F1 F2 : ℝ × ℝ) (M ∈ hyperbola_h) 
(h_dot_product : (-sqrt 3 - x_0) * -x_0 + (-y_0) * -y_0 < 0) :
  -sqrt 3 / 3 < y_0 ∧ y_0 < sqrt 3 / 3 := sorry

end range_of_y0_l100_100656


namespace amc_problem_l100_100653

theorem amc_problem (a b : ℕ) (h : ∀ n : ℕ, 0 < n → a^n + n ∣ b^n + n) : a = b :=
sorry

end amc_problem_l100_100653


namespace problem_equivalent_l100_100719

theorem problem_equivalent (x : Real) : (sqrt (10 + x) + sqrt (30 - x) = 8) → (10 + x) * (30 - x) = 144 := by
  sorry

end problem_equivalent_l100_100719


namespace exists_member_with_few_enemies_friends_l100_100735

theorem exists_member_with_few_enemies_friends (n q : ℕ) (h1 : q > 0) :
  (∀ (G : Fintype (Sym2 (Fin n))),
    (∃ F : set (Sym2 (Fin n)), F.card = q) ∧
    (∀ A B C : Fin n, A ≠ B ∧ B ≠ C ∧ A ≠ C → 
      (Sym2.mk A B ∈ F ∨ Sym2.mk B C ∈ F ∨ Sym2.mk C A ∈ F))) →
  ∃ v : Fin n, 
  ∃ E : set (Sym2 (Fin n)), 
  E.card ≤ q * (1 - (4 * q : ℕ) / n^2) :=
sorry

end exists_member_with_few_enemies_friends_l100_100735


namespace tan_45_deg_eq_one_l100_100163

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100163


namespace trigonometric_identity_l100_100992

theorem trigonometric_identity :
  (2 * real.cos (10 * real.pi / 180) - real.sin (20 * real.pi / 180)) / real.cos (20 * real.pi / 180) = real.sqrt 3 :=
by
  have h1 : 2 * real.cos (10 * real.pi / 180) = 2 * real.sin (80 * real.pi / 180), from sorry,
  have h2 : real.sin (80 * real.pi / 180) = real.sin (60 * real.pi / 180 + 20 * real.pi /180), from sorry,
  sorry

end trigonometric_identity_l100_100992


namespace tan_45_deg_eq_one_l100_100119

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100119


namespace eliot_account_percentage_increase_l100_100412

theorem eliot_account_percentage_increase :
  ∃ P : ℝ,
  let A := 225
  let E := 210
  let A_new := A + 0.1 * A -- Al's new account amount
  let E_new := E + (P / 100) * E -- Eliot's new account amount
  A > E ∧
  A - E = (1 / 12) * (A + E) ∧
  A_new = E_new + 21 ∧
  E = 210 →
  (P ≈ 7.86) := sorry

end eliot_account_percentage_increase_l100_100412


namespace find_YW_in_triangle_l100_100757

theorem find_YW_in_triangle
  (X Y Z W : Type)
  (d_XZ d_YZ d_XW d_CW : ℝ)
  (h_XZ : d_XZ = 10)
  (h_YZ : d_YZ = 10)
  (h_XW : d_XW = 12)
  (h_CW : d_CW = 5) : 
  YW = 29 / 12 :=
sorry

end find_YW_in_triangle_l100_100757


namespace combin_10_5_l100_100982

theorem combin_10_5 : nat.choose 10 5 = 252 := by
  sorry

end combin_10_5_l100_100982


namespace triangle_isosceles_l100_100359

noncomputable def is_isosceles (A B C : Point) : Prop :=
∃ (x : Point), (dist A x = dist B x) ∧ (x ∈ line_segment A B)

theorem triangle_isosceles
  (A B C L H M : Point)
  (angle_bisector : is_angle_bisector A L B C)
  (altitude : is_altitude B H A C)
  (midpoint_M : midpoint M A B)
  (perpendicular_bisector : is_perpendicular_bisector M L H) :
  is_isosceles A B C :=
sorry

end triangle_isosceles_l100_100359


namespace factorize_expression_l100_100239

theorem factorize_expression (x y : ℝ) : x^2 * y + 2 * x * y + y = y * (x + 1)^2 :=
by
  sorry

end factorize_expression_l100_100239


namespace solve_ZAMENA_l100_100975

noncomputable def ZAMENA : ℕ := 541234

theorem solve_ZAMENA :
  ∃ (A M E H : ℕ), 
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H ∧
    ZAMENA = 541234 :=
by {
  have h1 : ∃ (A M E H : ℕ), 
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H, sorry,
  exact ⟨2, 1, 2, 3, h1⟩,
  repeat { sorry }
}

end solve_ZAMENA_l100_100975


namespace tan_45_eq_1_l100_100204

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100204


namespace plane_equation_correct_l100_100666

-- Define the point on the plane
def point_on_plane : ℝ × ℝ × ℝ := (10, -2, 5)

-- Define the normal vector to the plane (vector from origin to point_on_plane)
def normal_vector : ℝ × ℝ × ℝ := (10, -2, 5)

-- Define the function representing the plane equation
def plane_eq (x y z : ℝ) : ℝ :=
  10 * x - 2 * y + 5 * z - 129

-- The theorem that represents the problem statement
theorem plane_equation_correct :
  ∀ x y z : ℝ, (point_on_plane.fst * (x - point_on_plane.1) +
                  point_on_plane.snd * (y - point_on_plane.2) +
                  point_on_plane.snd * (z - point_on_plane.3) = 0) →
  plane_eq x y z = 0 := 
sorry

end plane_equation_correct_l100_100666


namespace tan_45_deg_eq_one_l100_100175

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100175


namespace cos_angle_between_unit_vectors_l100_100281

variables {α : Type*} [NormedField α] [AddCommGroup α] [Module α α] [Inner α]
open InnerProductSpace

theorem cos_angle_between_unit_vectors (a b : α) (h1 : ‖a‖ = 1) (h2 : ‖b‖ = 1) (h3 : inner a (a + 3 • b) = -1) :
  inner a b = -2 / 3 :=
by
  sorry

end cos_angle_between_unit_vectors_l100_100281


namespace pints_of_cider_l100_100700

def pintCider (g : ℕ) (p : ℕ) : ℕ :=
  g / 20 + p / 40

def totalApples (f : ℕ) (h : ℕ) (a : ℕ) : ℕ :=
  f * h * a

theorem pints_of_cider (g p : ℕ) (farmhands : ℕ) (hours : ℕ) (apples_per_hour : ℕ)
  (H1 : g = 1)
  (H2 : p = 2)
  (H3 : farmhands = 6)
  (H4 : hours = 5)
  (H5 : apples_per_hour = 240) :
  pintCider (apples_per_hour * farmhands * hours / 3)
            (apples_per_hour * farmhands * hours * 2 / 3) = 120 :=
by
  sorry

end pints_of_cider_l100_100700


namespace jog_time_l100_100315

theorem jog_time (jog_constant : ∀ {d1 d2 t1 t2 : ℝ}, d2 ≠ 0 → t1 = d1 * t2 / d2) (d_gym d_park t_park : ℝ) :
  d_gym = 1.5 → d_park = 3 → t_park = 30 → ∃ t_gym : ℝ, t_gym = 15 := by
  intros h1 h2 h3
  use (t_park * d_gym / d_park)
  rw [h1, h2, h3]
  norm_num
  sorry

end jog_time_l100_100315


namespace monica_first_class_20_l100_100394

noncomputable def monica_first_class_students (X : ℕ) : Prop :=
  let second_and_third := 25 + 25 in
  let fourth := X / 2 in
  let fifth_and_sixth := 28 + 28 in
  (X + second_and_third + fourth + fifth_and_sixth = 136)

theorem monica_first_class_20 :
  ∃ X : ℕ, monica_first_class_students X ∧ X = 20 :=
by
  sorry

end monica_first_class_20_l100_100394


namespace max_consecutive_divisible_l100_100386

-- Define the sequence x_i based on the given conditions
def sequence (m : ℕ) (hm : 1 < m) : ℕ → ℕ
| i := if i < m then 2^i else ∑ j in finset.range m, sequence m hm (i - j - 1)

-- Define the proof problem statement
theorem max_consecutive_divisible (m : ℕ) (hm : 1 < m) :
  ∀ x, (∀ i, 0 ≤ i → x i = sequence m hm i) →
  ∀ k, (∃ n, ∀ i, n ≤ i ∧ i < n + k → x i % m = 0) → k ≤ m - 1 :=
sorry

end max_consecutive_divisible_l100_100386


namespace experts_eventual_win_probability_l100_100237

noncomputable def experts_win_probability (p : ℝ) (q : ℝ) (experts_needed : ℕ) (audience_needed : ℕ) (max_rounds : ℕ) : ℝ :=
  let e_all_wins := p ^ max_rounds
  let e_three_wins_one_loss := (Nat.choose max_rounds (max_rounds - 1)) * (p ^ (max_rounds - 1)) * q
  e_all_wins + e_three_wins_one_loss

theorem experts_eventual_win_probability :
  ∀ (p : ℝ) (q : ℝ),
  p = 0.6 →
  q = 1 - p →
  experts_win_probability p q 3 2 4 = 0.4752 := by
  intros p q hp hq
  have h : experts_win_probability p q 3 2 4 = 0.6 ^ 4 + 4 * (0.6 ^ 3) * 0.4 := sorry
  rw [hp, hq] at h
  norm_num at h
  exact h

end experts_eventual_win_probability_l100_100237


namespace tan_45_deg_eq_one_l100_100097

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100097


namespace tan_45_eq_1_l100_100180

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100180


namespace find_n_from_A_C_l100_100638

noncomputable def A_n (n : ℕ) : ℕ := n! / (n - 2)!
noncomputable def C_n (n : ℕ) (k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem find_n_from_A_C (n : ℕ) (h : (A_n n)^2 = C_n n (n - 3)) : n = 8 := by
  sorry

end find_n_from_A_C_l100_100638


namespace triangle_equilateral_or_right_l100_100869

-- Define the Triangle and Point structures
structure Triangle :=
(A B C : ℝ × ℝ)

structure Point :=
(P : ℝ × ℝ)

-- Define conditions for equal areas and equal perimeters for sub-triangles formed by point P inside triangle ABC.
def equal_areas (ABC : Triangle) (P : Point) : Prop :=
let PA := abs ((ABC.A.1 - P.P.1) * (ABC.C.2 - ABC.B.2) - (ABC.A.2 - P.P.2) * (ABC.C.1 - ABC.B.1)) in
let PB := abs ((ABC.B.1 - P.P.1) * (ABC.A.2 - ABC.C.2) - (ABC.B.2 - P.P.2) * (ABC.A.1 - ABC.C.1)) in
let PC := abs ((ABC.C.1 - P.P.1) * (ABC.B.2 - ABC.A.2) - (ABC.C.2 - P.P.2) * (ABC.B.1 - ABC.A.1)) in
PA = PB ∧ PB = PC

def equal_perimeters (ABC : Triangle) (P : Point) : Prop :=
let PA := dist ABC.A P.P + dist P.P ABC.B + dist ABC.B ABC.A in
let PB := dist ABC.B P.P + dist P.P ABC.C + dist ABC.C ABC.B in
let PC := dist ABC.C P.P + dist P.P ABC.A + dist ABC.A ABC.C in
PA = PB ∧ PB = PC

-- Define the main theorem to prove that if there exists such a point P, triangle ABC is either equilateral or right-angled.
theorem triangle_equilateral_or_right (ABC : Triangle) :
  (∃ P : Point, equal_areas ABC P ∧ equal_perimeters ABC P) →
  (is_equilateral ABC ∨ is_right_triangle ABC) :=
sorry

end triangle_equilateral_or_right_l100_100869


namespace determine_a_l100_100405

noncomputable def is_equidistant (P A B : ℝ × ℝ) (line_x : ℝ) :=
  dist P A = dist P B ∧ dist P A = abs (fst P - line_x) ∧ dist P B = abs (fst P - line_x)

theorem determine_a (a : ℝ) :
  (∀ P : ℝ × ℝ,
    is_equidistant P (0.5, 0) (a, 2) (-0.5) → exists_unique (λ P, is_equidistant P (0.5, 0) (a, 2) (-0.5))
  ) ↔ a = - 0.5 ∨ a = 0.5 :=
sorry

end determine_a_l100_100405


namespace correct_option_is_A_l100_100880

theorem correct_option_is_A :
  (random_event "Xiao Li wins the first prize in a sports lottery") ∧
  ¬(certain_event (coin_tosses 10 "heads up 5 times")) ∧
  ¬(certain_event "it rains heavily during the Qingming Festival") ∧
  ¬(impossible_event (λ a : ℚ, |a| ≥ 0)) →
  correct_option = "A" :=
begin
  sorry
end

end correct_option_is_A_l100_100880


namespace mows_in_summer_l100_100770

theorem mows_in_summer (S : ℕ) (h1 : 8 - S = 3) : S = 5 :=
sorry

end mows_in_summer_l100_100770


namespace find_third_number_in_proportion_l100_100723

theorem find_third_number_in_proportion (x : ℝ) (third_number : ℝ) (h1 : x = 0.9) (h2 : 0.75 / 6 = x / third_number) : third_number = 5 := by
  sorry

end find_third_number_in_proportion_l100_100723


namespace tan_of_45_deg_l100_100075

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100075


namespace socorro_training_days_l100_100420

variable (total_training_time_per_day : ℕ) (total_training_time : ℕ)

theorem socorro_training_days (h1 : total_training_time = 300) 
                              (h2 : total_training_time_per_day = 30) :
                              total_training_time / total_training_time_per_day = 10 := 
begin
  rw [h1, h2],
  norm_num,
end

end socorro_training_days_l100_100420


namespace socorro_training_days_l100_100421

variable (total_training_time_per_day : ℕ) (total_training_time : ℕ)

theorem socorro_training_days (h1 : total_training_time = 300) 
                              (h2 : total_training_time_per_day = 30) :
                              total_training_time / total_training_time_per_day = 10 := 
begin
  rw [h1, h2],
  norm_num,
end

end socorro_training_days_l100_100421


namespace find_s_alpha_plus_beta_l100_100250

variable {α β : ℝ} (f : ℝ → ℝ) (s : ℝ → ℝ)

axiom h1 : 0 < α ∧ α ≤ π / 2
axiom h2 : 0 < β ∧ β ≤ π / 2
axiom h3 : f (α + π / 2) = 10 / 13
axiom h4 : f (3 * β ^ 2) = 6 / 5

theorem find_s_alpha_plus_beta : s (α + β) = 16 / 65 :=
sorry

end find_s_alpha_plus_beta_l100_100250


namespace tan_45_deg_eq_one_l100_100021

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100021


namespace tan_45_deg_l100_100146

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100146


namespace apples_for_48_oranges_l100_100364

theorem apples_for_48_oranges (o a : ℕ) (h : 8 * o = 6 * a) (ho : o = 48) : a = 36 :=
by
  sorry

end apples_for_48_oranges_l100_100364


namespace range_of_y_l100_100213

noncomputable def y (b : Fin 15 → ℕ) : ℝ :=
  ∑ i in Finset.range 15, b ⟨i, sorry⟩ / 5^(i+1)

theorem range_of_y :
  (∀ i, b i = 0 ∨ b i = 3) → 0 ≤ y b ∧ y b < 0.9 :=
sorry

end range_of_y_l100_100213


namespace calc1_calc2_calc3_calc4_l100_100990

theorem calc1 : (-16) - 25 + (-43) - (-39) = -45 := by
  sorry

theorem calc2 : (-3 / 4)^2 * (-8 + 1 / 3) = -69 / 16 := by
  sorry

theorem calc3 : 16 / (- (1 / 2)) * (3 / 8) - | -45 | / 9 = -17 := by
  sorry

theorem calc4 : -1 ^ 2024 - (2 - 0.75) * (2 / 7) * (4 - (-5)^2) = 13 / 2 := by
  sorry

end calc1_calc2_calc3_calc4_l100_100990


namespace tan_45_deg_eq_one_l100_100152

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100152


namespace fraction_of_milk_is_one_fourth_l100_100365

theorem fraction_of_milk_is_one_fourth :
  ∀ (cup1_tea_initial cup2_milk_initial : ℚ)
    (cup1_transfer_fraction cup2_transfer_fraction1 cup1_transfer_fraction_back : ℚ),
    cup1_tea_initial = 6 →
    cup2_milk_initial = 6 →
    cup1_transfer_fraction = 1/3 →
    cup2_transfer_fraction1 = 1/4 →
    cup1_transfer_fraction_back = 1/6 →
    let cup1_tea_after_first_transfer := cup1_tea_initial * (1 - cup1_transfer_fraction),
        cup2_milk_after_first_transfer := cup2_milk_initial,
        cup2_tea_after_first_transfer := cup1_tea_initial * cup1_transfer_fraction,
        cup2_total_after_first_transfer := cup2_milk_after_first_transfer + cup2_tea_after_first_transfer,
        transfer_to_cup1 := cup2_total_after_first_transfer * cup2_transfer_fraction1,
        cup1_tea_after_second_transfer := cup1_tea_after_first_transfer + transfer_to_cup1 * (cup2_tea_after_first_transfer / cup2_total_after_first_transfer),
        cup1_milk_after_second_transfer := 1.5,
        cup1_total_after_second_transfer := cup1_tea_after_second_transfer + cup1_milk_after_second_transfer,
        transfer_back_to_cup2 := cup1_total_after_second_transfer * cup1_transfer_fraction_back,
        cup1_tea_after_final_transfer := cup1_tea_after_second_transfer - transfer_back_to_cup2 * (cup1_tea_after_second_transfer / cup1_total_after_second_transfer),
        cup1_milk_after_final_transfer := cup1_milk_after_second_transfer - transfer_back_to_cup2 * (cup1_milk_after_second_transfer / cup1_total_after_second_transfer),
        cup1_total_after_final_transfer := cup1_tea_after_final_transfer + cup1_milk_after_final_transfer
    in
    (cup1_milk_after_final_transfer / cup1_total_after_final_transfer) = 1/4 := by
  intros _ _ _ _ _ h1 h2 h3 h4 h5
  let cup1_tea_after_first_transfer := 6 * (1 - 1/3)
  let cup2_milk_after_first_transfer := 6
  let cup2_tea_after_first_transfer := 6 * 1/3
  let cup2_total_after_first_transfer := 6 + 6 * 1/3
  let transfer_to_cup1 := (6 + 2) * (1/4)
  let cup1_tea_after_second_transfer := 6 * (2/3) + 2 * (2 / 8)
  let cup1_milk_after_second_transfer :=  2 * (6 / 8)
  let cup1_total_after_second_transfer := (4 + 0.5 + 6 / 4)
  let transfer_back_to_cup2 := 6 * (1 / 6)
  let cup1_tea_after_final_transfer := 4.5 - 1 * (4.5 / 6)
  let cup1_milk_after_final_transfer := 1.5 - 1 * (1.5 / 6)
  let cup1_total_after_final_transfer := (3.75 + 1.25)
  show (1.25 / 5) = 1 / 4
  sorry

end fraction_of_milk_is_one_fourth_l100_100365


namespace tan_45_eq_1_l100_100178

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100178


namespace tan_45_eq_1_l100_100179

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100179


namespace ellipse_standard_equation_l100_100651

noncomputable def semi_minor_axis := 3
noncomputable def focal_distance := 4

theorem ellipse_standard_equation :
    ∀ (a b c : ℝ),
    b = semi_minor_axis →
    2 * c = focal_distance →
    a = sqrt (b^2 + c^2) →
    (∀ x y, (y^2 / a^2 + x^2 / b^2 = 1) → 
    (y^2 / 13 + x^2 / 9 = 1)) sorry

end ellipse_standard_equation_l100_100651


namespace count_4_digit_integers_l100_100704

theorem count_4_digit_integers :
  let first_two_digit_options := {2, 4, 7}
  let last_two_digit_options := {1, 3, 9} in
  ∃ n: ℕ, n = 54 ∧
  ∃ (x1 x2 x3 x4 : ℕ), 
    x1 ∈ first_two_digit_options ∧
    x2 ∈ first_two_digit_options ∧
    x3 ∈ last_two_digit_options ∧
    x4 ∈ last_two_digit_options ∧
    x3 ≠ x4 ∧
    n = (((x1 * 10 + x2) * 10 + x3) * 10 + x4) :=
by sorry

end count_4_digit_integers_l100_100704


namespace square_brush_ratio_l100_100921

theorem square_brush_ratio (s w : ℝ) (h : s > 0 ∧ w > 0) 
  (area_painted_one_third : (w^2 + 2*real.sqrt 2 * ((s - w * real.sqrt 2) / 2) ^ 2) = (s^2) / 3) : 
  w = (real.sqrt 2 - 1) * s :=
sorry

end square_brush_ratio_l100_100921


namespace tan_45_deg_l100_100128

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100128


namespace tan_45_eq_1_l100_100008

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100008


namespace angle_of_vectors_l100_100275

variables {V : Type} [inner_product_space ℝ V] (a b : V)

theorem angle_of_vectors
  (h1 : ‖a + 2 • b‖ = real.sqrt 7 * ‖a‖)
  (h2 : ‖a + 2 • b‖ = real.sqrt 7 * ‖b‖)
  (ha : a ≠ 0)
  (hb : b ≠ 0) :
  inner_product_space.angle a b = real.pi / 3 :=
sorry

end angle_of_vectors_l100_100275


namespace upstream_distance_l100_100517

-- Define the conditions
def V_s : ℝ := 1 -- Speed of the stream in km/h
def distance_downstream : ℝ := 100 -- Distance traveled downstream in km
def time_downstream : ℝ := 10 -- Time taken to travel downstream in hours
def time_upstream : ℝ := 25 -- Time taken to travel upstream in hours

-- Define the effective speeds
def V_downstream : ℝ := distance_downstream / time_downstream -- Effective downstream speed
def V_b : ℝ := V_downstream - V_s -- Speed of the boat in still water
def V_upstream : ℝ := V_b - V_s -- Effective upstream speed

-- Define the distance travelled upstream
def D_upstream : ℝ := V_upstream * time_upstream -- Distance traveled upstream

-- The statement to prove
theorem upstream_distance : D_upstream = 200 :=
by
  sorry

end upstream_distance_l100_100517


namespace isosceles_triangle_locus_ellipse_area_l100_100273

theorem isosceles_triangle_locus_ellipse_area
  (b h s k : ℝ) (k_gt_one : k > 1) (squared_dist_sum_eq : ∀ (P : ℝ × ℝ),
    let (x, y) := P in 
    (x + b)^2 + y^2 + (x - b)^2 + y^2 + x^2 + (y - h)^2 = k * s^2)
  (A_eq : -b^2 + h^2 = -s^2) :
  (∃ (c_x c_y : ℝ), (0 = c_x) ∧ (h/3 = c_y) ∧
                    (squared_dist_sum_eq = ellipse c_x c_y k s A_eq)) ∧
  area ABC = b * h :=
sorry

end isosceles_triangle_locus_ellipse_area_l100_100273


namespace length_of_crease_eq_l100_100217

theorem length_of_crease_eq (x y pq : ℝ) (hx : x = 2) (hy : y = 3) (eq_triangle : ∀ A B C : ℝ, ∃ a b c : ℝ, a = b ∧ b = c ∧ a = 5) 
(hcos1 : (5 - x)^2 = x^2 + 5^2 - 2 * x * 5 * real.cos (real.pi / 3))
(hcos2 : (5 - y)^2 = y^2 + 3^2 - 2 * y * 3 * real.cos (real.pi / 3))
(hcos3 : pq^2 = ((15 / 7) ^ 2 + (35 / 7) ^ 2 - 2 * (15 / 7) * (35 / 7) * real.cos (real.pi / 3))) :
pq = 20 / 7 := 
sorry

end length_of_crease_eq_l100_100217


namespace range_of_a_l100_100667

variable {a : ℝ}
variable h : ∀ x : ℝ, a * x^2 - 2 * a * x - 2 ≤ 0

theorem range_of_a : -2 ≤ a ∧ a ≤ 0 := 
by
  sorry

end range_of_a_l100_100667


namespace solve_system_of_equations_l100_100822

theorem solve_system_of_equations 
  (a b c s : ℝ) (x y z : ℝ)
  (h1 : y^2 - z * x = a * (x + y + z)^2)
  (h2 : x^2 - y * z = b * (x + y + z)^2)
  (h3 : z^2 - x * y = c * (x + y + z)^2)
  (h4 : a^2 + b^2 + c^2 - (a * b + b * c + c * a) = a + b + c) :
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ x + y + z = 0) ∨
  ((x + y + z ≠ 0) ∧
   (x = (2 * c - a - b + 1) * s) ∧
   (y = (2 * a - b - c + 1) * s) ∧
   (z = (2 * b - c - a + 1) * s)) :=
by
  sorry

end solve_system_of_equations_l100_100822


namespace probability_event_M_l100_100464

open Classical

noncomputable def sample_count_A := 4
noncomputable def sample_count_B := 2
noncomputable def sample_count_C := 1
noncomputable def total_samples := 7

-- Define sets representing samples from each workshop
def samples_A := {A₁, A₂, A₃, A₄}
def samples_B := {B₁, B₂}
def samples_C := {C₁}
def all_samples := samples_A ∪ samples_B ∪ samples_C

-- Function to count the number of pairs with different workshops
def event_M_pairs : set (set ℕ) :=
  {pair | (pair ⊆ all_samples) ∧ (1 < |pair|) = 2 ∧
  ∃ a b, a ∈ pair ∧ b ∈ pair ∧ a ≠ b ∧
  ((a ∈ samples_A) ∧ (b ∈ samples_B) ∨
   (a ∈ samples_A) ∧ (b ∈ samples_C) ∨ 
   (a ∈ samples_B) ∧ (b ∈ samples_C))}

-- Total possible pairs from 7 samples
def total_pairs := (finset.univ : finset (fin 7)).powerset.filter (λ s, s.card = 2)

-- Number of successful pairs for event M
def successful_pairs_M := total_pairs.filter (λ s, s ∈ event_M_pairs)

theorem probability_event_M : 
  (successful_pairs_M.card : ℚ) / (total_pairs.card : ℚ) = 2 / 3 :=
sorry

end probability_event_M_l100_100464


namespace range_of_f1_plus_f5_l100_100793

-- Define the polynomial function and the conditions on its coefficients
def f (x : ℝ) (a b c d : ℝ) : ℝ := a * x ^ 3 + b * x ^ 2 + c * x + d

-- Define the conditions given in the problem
variables (a b c d : ℝ)
variable (t : ℝ)
hypothesis (zero1 : 0 < t)
hypothesis (t_lt_1 : t < 1)
hypothesis (f2 : f 2 a b c d = t / 2)
hypothesis (f3 : f 3 a b c d = t / 3)
hypothesis (f4 : f 4 a b c d = t / 4)

-- The main statement to prove
theorem range_of_f1_plus_f5 : (0 < f 1 a b c d + f 5 a b c d) ∧ (f 1 a b c d + f 5 a b c d < 1) := 
by 
  sorry

end range_of_f1_plus_f5_l100_100793


namespace M_inter_N_l100_100308

def M : Set ℝ := { x | (x - 3) * (x + 1) ≥ 0 }
def N : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }

theorem M_inter_N : M ∩ N = Icc (-2 : ℝ) (-1 : ℝ) :=
by
  sorry

end M_inter_N_l100_100308


namespace ellipse_equation_intersection_point_l100_100669

-- Given conditions
def ellipse_eqn (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

def eccentricity_eqn (a c : ℝ) (e : ℝ) : Prop :=
  e = c / a

def circle_eqn (x : ℝ) (y : ℝ) : Prop :=
  (x - real.sqrt 3)^2 + y^2 = 7

-- Proof that the ellipse equation holds given the conditions
theorem ellipse_equation (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b) (e : ℝ) (ecc_eqn : eccentricity_eqn a (real.sqrt (a^2 - b^2)) e) :
  a = 2 ∧ b = 1 ∧ ellipse_eqn a b a_pos b_pos a_gt_b :=
by sorry

-- Proof that the coordinates of point A are determined correctly given the conditions
def line_eqn (k m : ℝ) : ℝ → ℝ := λ x, k * x + m

theorem intersection_point (k m : ℝ) (A : ℝ × ℝ) (line : ∀ x y, y = line_eqn k m x)
  (circle : circle_eqn A.1 A.2) :
  ∃ A : ℝ × ℝ, circle_eqn A.1 A.2 ∧ line_eqn k m A.1 = A.2 :=
by sorry

end ellipse_equation_intersection_point_l100_100669


namespace number_of_elements_in_T_l100_100795

noncomputable def g(x : ℝ) : ℝ := (x + 8) / (x - 1)

def gn : ℕ → (ℝ → ℝ)
| 0     := g
| (n+1) := g ∘ gn n

def T : Set ℝ := {x : ℝ | ∃ n : ℕ, gn n x = x}

theorem number_of_elements_in_T : fintype.card T = 2 := by
  sorry

end number_of_elements_in_T_l100_100795


namespace tan_45_deg_l100_100143

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100143


namespace probability_at_least_one_spade_and_no_hearts_l100_100916

theorem probability_at_least_one_spade_and_no_hearts :
  let P_not_spade := (39 / 52 : ℝ)
  let P_spade := (1 - P_not_spade)
  let P_no_hearts_5 := P_not_spade ^ 5
  let P_at_least_one_spade := 1 - P_not_spade ^ 5
  let P_no_hearts_and_spade := P_at_least_one_spade * P_no_hearts_5
  P_no_hearts_and_spade = 189723 / 1048576 :=
sorry

end probability_at_least_one_spade_and_no_hearts_l100_100916


namespace sum_sequence_formula_l100_100683

-- Definitions corresponding to the conditions:
def a (n : ℕ) : ℚ := 1 / ((2 * n - 1) * (2 * n + 1))

def S (n : ℕ) : ℚ := ∑ i in Finset.range (n + 1), a (i + 1)

-- The main theorem statement:
theorem sum_sequence_formula (n : ℕ) : S n = n / (2 * n + 1) := sorry

end sum_sequence_formula_l100_100683


namespace tan_45_deg_l100_100125

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100125


namespace tan_45_deg_eq_one_l100_100106

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100106


namespace int_solutions_l100_100241

theorem int_solutions (a b : ℤ) (h : a^2 + b = b^2022) : (a, b) = (0, 0) ∨ (a, b) = (0, 1) :=
by {
  sorry
}

end int_solutions_l100_100241


namespace hyperbola_equation_l100_100753

def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a - y^2 / b = 1

def passes_through_point (a b : ℝ) (x y : ℝ) (px py : ℝ) : Prop :=
  (px, py) = (2 * Real.sqrt 2, -Real.sqrt 2) ∧ is_hyperbola a b px py

def has_asymptotes (b a : ℝ) : Prop :=
  b / a = Real.sqrt 2

theorem hyperbola_equation :
  ∃ (a b : ℝ), passes_through_point a b 2 (Real.sqrt 2) (-Real.sqrt 2)
  ∧ has_asymptotes b a ∧ (a = 7 ∧ b = 14) ∧ is_hyperbola 7 14 2 (Real.sqrt 2) := 
sorry

end hyperbola_equation_l100_100753


namespace tan_45_eq_1_l100_100002

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100002


namespace tan_of_45_deg_l100_100072

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100072


namespace domain_of_function_l100_100246

noncomputable def domain_function := { x : ℝ | ∃ (k : ℤ), 
  ((2 * k * Real.pi + Real.pi / 2 < x ∧ x ≤ 2 * k * Real.pi + Real.pi) ∨ (x = 2 * k * Real.pi)) }

theorem domain_of_function :
  ∀ x, (sqrt (Real.sin x) + sqrt (-Real.tan x)).dom ↔ (∃ (k : ℤ), 
  ((2 * k * Real.pi + Real.pi / 2 < x ∧ x ≤ 2 * k * Real.pi + Real.pi) ∨ (x = 2 * k * Real.pi))) :=
sorry

end domain_of_function_l100_100246


namespace tan_of_45_deg_l100_100094

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100094


namespace water_remain_l100_100907

theorem water_remain (poured_out : ℝ) (h : poured_out = 0.45) : 1 - poured_out = 0.55 :=
by
  rw [h]
  have : 1 - 0.45 = 0.55 := by norm_num
  exact this
  sorry

end water_remain_l100_100907


namespace range_of_inclination_angle_of_tangent_line_l100_100634

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - x^2

theorem range_of_inclination_angle_of_tangent_line :
  (∀ x : ℝ, (0 ≤ real.arctan (f' x) ∧ real.arctan (f' x) < π/2)
      ∨ (3 * π / 4 ≤ real.arctan (f' x) ∧ real.arctan (f' x) < π)) :=
sorry

end range_of_inclination_angle_of_tangent_line_l100_100634


namespace train_length_l100_100897

theorem train_length (speed_faster speed_slower : ℝ) (time_sec : ℝ) (length_each_train : ℝ) :
  speed_faster = 47 ∧ speed_slower = 36 ∧ time_sec = 36 ∧ 
  (length_each_train = 55 ↔ 2 * length_each_train = ((speed_faster - speed_slower) * (1000/3600) * time_sec)) :=
by {
  -- We declare the speeds in km/hr and convert the relative speed to m/s for calculation.
  sorry
}

end train_length_l100_100897


namespace prove_ZAMENA_l100_100955

theorem prove_ZAMENA :
  ∃ (A M E H Z : ℕ),
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ Z ∧ Z ≤ 5 ∧
    (3 > A) ∧ (A > M) ∧ (M < E) ∧ (E < H) ∧ (H < A) ∧
    (A ≠ M) ∧ (A ≠ E) ∧ (A ≠ H) ∧ (A ≠ Z) ∧
    (M ≠ E) ∧ (M ≠ H) ∧ (M ≠ Z) ∧
    (E ≠ H) ∧ (E ≠ Z) ∧
    (H ≠ Z) ∧
    Z = 5 ∧
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234 :=
begin
  sorry
end

end prove_ZAMENA_l100_100955


namespace first_discount_percentage_l100_100847

noncomputable def saree_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  initial_price * (1 - discount1 / 100) * (1 - discount2 / 100)

theorem first_discount_percentage (x : ℝ) : saree_price 400 x 20 = 240 → x = 25 :=
by sorry

end first_discount_percentage_l100_100847


namespace solve_xy_equation_l100_100432

theorem solve_xy_equation (x y : ℝ) (h : x ^ 2 + 2 * real.sqrt 3 * x + y - 4 * real.sqrt y + 7 = 0) : 
  x = -real.sqrt 3 ∧ y = 4 :=
by
  sorry

end solve_xy_equation_l100_100432


namespace seventh_term_of_geometric_sequence_l100_100523

theorem seventh_term_of_geometric_sequence (r : ℝ) 
  (h1 : 3 * r^5 = 729) : 3 * r^6 = 2187 :=
sorry

end seventh_term_of_geometric_sequence_l100_100523


namespace quadratic_inequality_condition_l100_100437

theorem quadratic_inequality_condition (a b c : ℝ) (h : a < 0) (disc : b^2 - 4 * a * c ≤ 0) : 
  ∀ x : ℝ, a * x^2 + b * x + c ≤ 0 :=
sorry

end quadratic_inequality_condition_l100_100437


namespace tan_45_deg_l100_100051

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100051


namespace exists_point_with_distance_inequality_l100_100260

open EuclideanGeometry

variable {n : ℕ} (h_n : n > 0) (points : Fin (2 * n) → Point) (lines : Fin (3 * n) → Line)

theorem exists_point_with_distance_inequality :
  ∃ P : Point, (∑ i in Finset.univ, dist P (lines i)) < (∑ j in Finset.univ, dist P (points j)) :=
sorry

end exists_point_with_distance_inequality_l100_100260


namespace combin_10_5_l100_100983

theorem combin_10_5 : nat.choose 10 5 = 252 := by
  sorry

end combin_10_5_l100_100983


namespace vector_problem_1_vector_problem_2_vector_problem_3_l100_100782

section vector_problems

open Real

-- Define vectors a, b, c, and d and other conditions
def a : ℝ × ℝ := (-1, 1)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c (y : ℝ) : ℝ × ℝ := (5, y)
def d : ℝ × ℝ := (8, 6)

-- Conditions
axiom b_parallel_d : ∀ x, b x = (4, 3) ⟷ 6 * x = 24
axiom perpendicular_cond : ∀ y, (4 * a + d) · c y = 0 ⟷ 20 + 10 * y = 0

-- Compute projection of c in direction of a
def projection_c_a (y : ℝ) : ℝ := 
  let cos_theta := ((a.1 * 5 + a.2 * y) / ((sqrt (a.1 ^ 2 + a.2 ^ 2)) * (sqrt (5 ^ 2 + y ^ 2))))
  (sqrt ((5 ^ 2 + y ^ 2))) * cos_theta

-- Find lambda1 and lambda2
noncomputable def find_lambdas (λ1 λ2 : ℝ) : Prop :=
  ∃ λ1 λ2, (λ1 = -23 / 7) ∧ (λ2 = 3 / 7) ∧ (c (-2) = λ1 • a + λ2 • (4, 3))

theorem vector_problem_1 : b 4 = (4, 3) ∧ c (-2) = (5, -2) := by
  sorry

theorem vector_problem_2 : projection_c_a (-2) = - (7 / 2) * (sqrt 2) := by
  sorry

theorem vector_problem_3 : find_lambdas (-23 / 7) (3 / 7) := by
  sorry

end vector_problems

end vector_problem_1_vector_problem_2_vector_problem_3_l100_100782


namespace smallest_d_for_inverse_g_l100_100786

def g (x : ℝ) := (x - 3)^2 - 8

theorem smallest_d_for_inverse_g : ∃ d : ℝ, (∀ x y : ℝ, x ≠ y → x ≥ d → y ≥ d → g x ≠ g y) ∧ ∀ d' : ℝ, d' < 3 → ∃ x y : ℝ, x ≠ y ∧ x ≥ d' ∧ y ≥ d' ∧ g x = g y :=
by
  sorry

end smallest_d_for_inverse_g_l100_100786


namespace find_f_neg_2_l100_100664

theorem find_f_neg_2 (f : ℝ → ℝ) (b x : ℝ) (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, x ≥ 0 → f x = x^2 - 3*x + b) (h3 : f 0 = 0) : f (-2) = 2 := by
sorry

end find_f_neg_2_l100_100664


namespace tan_45_deg_eq_one_l100_100042

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100042


namespace tan_45_deg_eq_one_l100_100168

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100168


namespace tan_45_deg_eq_one_l100_100099

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100099


namespace edward_cards_l100_100577

noncomputable def num_cards_each_binder : ℝ := (7496.5 + 27.7) / 23
noncomputable def num_cards_fewer_binder : ℝ := num_cards_each_binder - 27.7

theorem edward_cards : 
  (⌊num_cards_each_binder + 0.5⌋ = 327) ∧ (⌊num_cards_fewer_binder + 0.5⌋ = 299) :=
by
  sorry

end edward_cards_l100_100577


namespace largest_average_even_terms_l100_100573

def multiples (n : ℕ) (limit : ℕ) : List ℕ := List.range (limit / n) |>.map (λ i => n * (i + 1))

def even_positions (l : List ℕ) : List ℕ := l.toList.filter_with_index (λ i _ => even i)

def average (l : List ℕ) : ℚ := ((l.foldr (· + ·) 0) : ℚ) / l.length

theorem largest_average_even_terms : 
  let m3 := even_positions (multiples 3 50)
  let m4 := even_positions (multiples 4 50)
  let m5 := even_positions (multiples 5 50)
  let m7 := even_positions (multiples 7 50)
  let m9 := even_positions (multiples 9 50)
  average m4 = 28 ∧ average m7 = 28 ∧ average m3 < 28 ∧ average m5 < 28 ∧ average m9 < 28 := 
by {
  sorry
}

end largest_average_even_terms_l100_100573


namespace smallest_prime_divisor_of_3_pow_19_add_11_pow_23_l100_100873

theorem smallest_prime_divisor_of_3_pow_19_add_11_pow_23 :
  ∀ (n : ℕ), Prime n → n ∣ 3^19 + 11^23 → n = 2 :=
by
  sorry

end smallest_prime_divisor_of_3_pow_19_add_11_pow_23_l100_100873


namespace tan_of_45_deg_l100_100078

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100078


namespace maximum_value_S_l100_100758

theorem maximum_value_S (a b c S : ℝ) (h1 : 3 * a^2 = 2 * b^2 + c^2) (h2 : S = 1/2 * b * c * sin (arccos (b^2 + c^2 - a^2) / (2*b*c))) :
  ∃ b c, (b = sqrt 2 * c) ∧ (S / (b^2 + 2 * c^2) = sqrt 14 / 24) :=
by { sorry }

end maximum_value_S_l100_100758


namespace find_sixtieth_pair_l100_100675

def diagonal_number (n : ℕ) : ℕ := n * (n + 1) / 2

def find_pair_on_diagonal (diagonal : ℕ) (position : ℕ) : (ℕ × ℕ) :=
  (position, diagonal + 1 - position)

theorem find_sixtieth_pair :
  let diagonal := 10
  let position := 60 - diagonal_number diagonal
  let sixtieth_pair := find_pair_on_diagonal (diagonal + 1) position
  sixtieth_pair = (5, 6) :=
by {
  let diagonal := 10,
  have h1 : diagonal_number diagonal = 55,
  have h2 : 60 - diagonal_number diagonal = 5,
  let position := 5,
  let sixtieth_pair := find_pair_on_diagonal (diagonal + 1) position,
  show sixtieth_pair = (5, 6),
  sorry
}

end find_sixtieth_pair_l100_100675


namespace tan_45_eq_1_l100_100197

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100197


namespace equidistant_P_AP_BP_CP_DP_l100_100776

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def distance (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2

def A : Point := ⟨10, 0, 0⟩
def B : Point := ⟨0, -6, 0⟩
def C : Point := ⟨0, 0, 8⟩
def D : Point := ⟨0, 0, 0⟩
def P : Point := ⟨5, -3, 4⟩

theorem equidistant_P_AP_BP_CP_DP :
  distance P A = distance P B ∧ distance P B = distance P C ∧ distance P C = distance P D := 
sorry

end equidistant_P_AP_BP_CP_DP_l100_100776


namespace smallest_positive_period_monotonically_increasing_interval_max_min_values_l100_100677

def f (x : ℝ) : ℝ := cos x ^ 2 + sqrt 3 * sin x * cos x

theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x :=
by
  use π
  sorry

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x, x ∈ set.Icc (k * π - π / 3) (k * π + π / 3) → ∀ y ⬝ y ∈ set.Icc (x, x) →
    f y ≤ f (y + ε) :=
sorry

theorem max_min_values : 
  ∃ max min, 
    ∀ x, x ∈ set.Icc (-π / 6) (π / 3) → 
      f x ≤ max ∧ f x ≥ min :=
by
  use (3/2)
  use 0
  sorry

end smallest_positive_period_monotonically_increasing_interval_max_min_values_l100_100677


namespace tan_45_deg_eq_one_l100_100104

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100104


namespace profit_percentage_for_A_is_20_l100_100528

/-- Definitions of key values which are conditions in the problem -/
def CP_A : ℝ := 148
def SP_B : ℝ := 222
def profit_percentage_B : ℝ := 0.25

/-- Computation of the cost price for B (CP_B) based on the conditions -/
def CP_B : ℝ := SP_B / (1 + profit_percentage_B)

/-- Computation of profit percentage for A when selling the cricket bat to B -/
def profit_percentage_A : ℝ := ((CP_B - CP_A) / CP_A) * 100

/-- Statement to be proven: the profit percentage for A when selling the cricket bat to B is 20% -/
theorem profit_percentage_for_A_is_20 : profit_percentage_A = 20 :=
by
  sorry

end profit_percentage_for_A_is_20_l100_100528


namespace rhombus_area_l100_100470

theorem rhombus_area:
  let s := 4 in
  let h := (s * Real.sqrt 3 / 2) in
  let d1 := 4 * Real.sqrt 3 - 4 in
  let d2 := s in
  (1/2 : ℝ) * d1 * d2 = 8 * Real.sqrt 3 - 8 :=
by
  sorry

end rhombus_area_l100_100470


namespace overall_avg_mark_l100_100337

def classA_students : ℕ := 24
def classB_students : ℕ := 50
def classC_students : ℕ := 36
def classD_students : ℕ := 15

def classA_avg_mark : ℝ := 40
def classB_avg_mark : ℝ := 60
def classC_avg_mark : ℝ := 55
def classD_avg_mark : ℝ := 70

theorem overall_avg_mark :
  let total_students := classA_students + classB_students + classC_students + classD_students in
  let total_marks := 
    classA_students * classA_avg_mark + 
    classB_students * classB_avg_mark + 
    classC_students * classC_avg_mark + 
    classD_students * classD_avg_mark in
  (total_marks / total_students) = 55.92 :=
by
  sorry

end overall_avg_mark_l100_100337


namespace range_of_y_eq_x_squared_l100_100684

noncomputable def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem range_of_y_eq_x_squared :
  M = { y : ℝ | ∃ x : ℝ, y = x^2 } := by
  sorry

end range_of_y_eq_x_squared_l100_100684


namespace probability_of_experts_winning_l100_100233

-- Definitions required from the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p
def current_expert_score : ℕ := 3
def current_audience_score : ℕ := 4

-- The main theorem to state
theorem probability_of_experts_winning : 
  p^4 + 4 * p^3 * q = 0.4752 := 
by sorry

end probability_of_experts_winning_l100_100233


namespace find_phi_l100_100303

noncomputable def f (x ϕ : ℝ) : ℝ := Real.sin (2 * x + ϕ)

-- Define the conditions as hypotheses
axiom condition1 (ϕ : ℝ) : f (π / 3) ϕ = 1 / 2
axiom condition2 (ϕ : ℝ) : f (-π / 3) ϕ = 1 / 2

-- Define the value of ϕ that needs to be proved
def solution (k : ℤ) : ℝ := 2 * k * Real.pi - Real.pi / 2

-- The statement to be proved
theorem find_phi (k : ℤ) : 
  f (π / 3) (solution k) = 1 / 2 ∧ f (-π / 3) (solution k) = 1 / 2 :=
sorry

end find_phi_l100_100303


namespace limit_expression_l100_100285

open Classical

variable (f : ℝ → ℝ) (x₀ : ℝ)

-- Given condition: f'(x₀) = -3
def derivative_condition : Prop :=
  deriv f x₀ = -3

-- Proof statement
theorem limit_expression (h : ℝ) (hf : deriv f x₀ = -3) :
  tendsto (λ h, (f (x₀ + h) - f (x₀ - h)) / h) (𝓝 0) (𝓝 (-6)) :=
by {
  sorry
}

end limit_expression_l100_100285


namespace triangle_incirle_bisect_median_area_l100_100355

theorem triangle_incirle_bisect_median_area (A B C : Point)
  (hAB : distance A B = 24)
  (bisect_median : ∃ M : Point, is_median A M B C ∧ incircle_bisects_median A M)
  (harea : ∃ k p : ℤ, area A B C = k * real.sqrt p ∧ ¬ is_divisible_by_square_of_prime p) 
: k + p = 42 := 
sorry

end triangle_incirle_bisect_median_area_l100_100355


namespace area_triangle_ellipse_l100_100778

theorem area_triangle_ellipse 
  (M : ℝ × ℝ)
  (a b : ℝ)
  (F1 F2 : ℝ × ℝ)
  (h1 : a = 5)
  (h2 : b = 4)
  (ellipse_eq : M.1 ^ 2 / 25 + M.2 ^ 2 / 16 = 1)
  (angle_condition : angle F1 M F2 = π / 6)
  (foci_position : ∥F1∥ = 3 ∧ ∥F2∥ = 3) :
  area_triangle F1 M F2 = 16 * (2 - sqrt 3) := 
sorry

end area_triangle_ellipse_l100_100778


namespace line_MN_intersects_AD_BC_l100_100777

-- We define the points and lines as required in the conditions
structure Circle (α : Type*) := 
  (center : α)
  (radius : ℝ)

universe u

variables {α : Type u} [AddCommGroup α] [VectorSpace ℝ α]

-- Defining points on the circle and the perpendicular points
variables (A B C D M N : α)

-- Given conditions
variables (circle : Circle α) 
          (AB CD : AffineSubspace ℝ α)
          (on_circle_A : A ∈ circle)
          (on_circle_B : B ∈ circle)
          (on_circle_C : C ∈ circle)
          (on_circle_D : D ∈ circle)
          (perpendicular_A : ∃ (l : AffineSubspace ℝ α), l.IsPerpendicular (line_through A B))
          (perpendicular_C : ∃ (l : AffineSubspace ℝ α), l.IsPerpendicular (line_through C D))
          (perp_inter_M : M = intersection_point (perpendicular_through A) (perpendicular_through C))
          (perpendicular_B : ∃ (l : AffineSubspace ℝ α), l.IsPerpendicular (line_through A B))
          (perpendicular_D : ∃ (l : AffineSubspace ℝ α), l.IsPerpendicular (line_through C D))
          (perp_inter_N : N = intersection_point (perpendicular_through B) (perpendicular_through D))

-- Statement to be proven
theorem line_MN_intersects_AD_BC 
  (MN : Line α) 
  (BC_intersect_AD : ∃ (P : α), P ∈ (line_through B C) ∧ P ∈ (line_through A D)) 
  (H_MN_DEF : MN = line_through M N) : 
  ∃ (P : α), P ∈ MN ∧ P ∈ BC_intersect_AD :=
begin
  -- Proof omitted
  sorry
end

end line_MN_intersects_AD_BC_l100_100777


namespace petya_vasya_meet_at_lantern_64_l100_100539

-- Define the total number of lanterns and intervals
def total_lanterns : ℕ := 100
def total_intervals : ℕ := total_lanterns - 1

-- Define the positions of Petya and Vasya at a given time
def petya_initial : ℕ := 1
def vasya_initial : ℕ := 100
def petya_position : ℕ := 22
def vasya_position : ℕ := 88

-- Define the number of intervals covered by Petya and Vasya
def petya_intervals_covered : ℕ := petya_position - petya_initial
def vasya_intervals_covered : ℕ := vasya_initial - vasya_position

-- Define the combined intervals covered
def combined_intervals_covered : ℕ := petya_intervals_covered + vasya_intervals_covered

-- Define the interval after which Petya and Vasya will meet
def meeting_intervals : ℕ := total_intervals - combined_intervals_covered

-- Define the final meeting point according to Petya's travel
def meeting_lantern : ℕ := petya_initial + (meeting_intervals / 2)

theorem petya_vasya_meet_at_lantern_64 : meeting_lantern = 64 := by {
  -- Proof goes here
  sorry
}

end petya_vasya_meet_at_lantern_64_l100_100539


namespace tan_45_eq_1_l100_100194

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100194


namespace max_value_f_l100_100621

-- Define the function f(x) for x in the interval (0, 1]
def f (x : ℝ) : ℝ := x * (1 - x) / ((x + 1) * (x + 2) * (2 * x + 1))

-- Define the interval (0, 1]
def within_interval (x : ℝ) : Prop := 0 < x ∧ x ≤ 1

-- Statement of the problem
theorem max_value_f :
  ∃ x ∈ set.Icc 0 1, (f x ≠ f x) := sorry

end max_value_f_l100_100621


namespace tan_45_deg_eq_one_l100_100019

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100019


namespace venue_cost_correct_l100_100768

noncomputable def cost_per_guest : ℤ := 500
noncomputable def johns_guests : ℤ := 50
noncomputable def wifes_guests : ℤ := johns_guests + (60 * johns_guests) / 100
noncomputable def total_wedding_cost : ℤ := 50000
noncomputable def guests_cost : ℤ := wifes_guests * cost_per_guest
noncomputable def venue_cost : ℤ := total_wedding_cost - guests_cost

theorem venue_cost_correct : venue_cost = 10000 := 
  by
  -- Proof can be filled in here.
  sorry

end venue_cost_correct_l100_100768


namespace range_of_k_l100_100269

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then
  -x^2 + x + 1
else if 1 ≤ x ∧ x < 2 then
  (1/2)^|x - (3/2)|
else
  4 * f(x - 2)

def a_n (n : ℕ) : ℝ := 
  let interval := [2*n  , 2*n+2]
  let vals := [for i in interval do f i]
  finset.max vals

def S_n (n : ℕ) : ℝ := 
  list.sum (list.of_fn (λ i, a_n i) n)

theorem range_of_k (k : ℝ) (h : ∀ n : ℕ, S_n n < k) : k ≥ (5 / 3) :=
sorry

end range_of_k_l100_100269


namespace zero_of_f_in_interval_l100_100856

def f (x : ℝ) : ℝ := Real.log x + 2 * x - 1

theorem zero_of_f_in_interval :
  ∃ (c : ℝ), (c > (1/2) ∧ c < 1) ∧ f c = 0 :=
by
  have h1 : f (1 / 2) < 0 := sorry
  have h2 : f 1 > 0 := sorry
  use_exists
  apply IntermediateValueTheorem
  -- Remaining part of the proof is omitted
  sorry

end zero_of_f_in_interval_l100_100856


namespace ME_squared_eq_MJ_mul_MH_l100_100790

variable {A B C H J M E : Point}
variable [TriangleABC A B C] [FootOfAltitude H A B C] [FootOfAngleBisector J A B C] 
variable [MidpointBC M B C] [PointOfTangency E B C]

theorem ME_squared_eq_MJ_mul_MH :
  dist M E ^ 2 = dist M J * dist M H := 
sorry

end ME_squared_eq_MJ_mul_MH_l100_100790


namespace number_of_valid_3_digit_integers_l100_100547

theorem number_of_valid_3_digit_integers :
  {N : ℕ // N ≥ 100 ∧ N < 1000 ∧ ((N % 4 + N % 7 - 50) % 100 = (2 * N % 100))}.card = 10 :=
sorry

end number_of_valid_3_digit_integers_l100_100547


namespace probability_distinct_real_solutions_l100_100783

theorem probability_distinct_real_solutions : ∀ (c : ℝ), c ∈ set.Icc (-19 : ℝ) 19 →
  ∃ (u v : ℕ), nat.coprime u v ∧ (↑u / ↑v : ℝ) = 17 / 19 ∧ u + v = 36 :=
by {
  sorry
}

end probability_distinct_real_solutions_l100_100783


namespace missing_shirts_l100_100809

-- Definition of conditions
def pairs_of_trousers : ℕ := 10
def price_per_trousers : ℕ := 9
def total_cost : ℕ := 140
def price_per_shirt : ℕ := 5
def claimed_number_of_shirts : ℕ := 2

-- The theorem to be proved
theorem missing_shirts : 
  let cost_of_trousers := pairs_of_trousers * price_per_trousers,
      cost_of_shirts := total_cost - cost_of_trousers,
      number_of_shirts := cost_of_shirts / price_per_shirt,
      missing_shirts := number_of_shirts - claimed_number_of_shirts
  in
  missing_shirts = 8 :=
by
  sorry

end missing_shirts_l100_100809


namespace condition_sufficiency_but_not_necessity_l100_100453

variable (p q : Prop)

theorem condition_sufficiency_but_not_necessity:
  (¬ (p ∨ q) → ¬ p) ∧ (¬ p → ¬ (p ∨ q) → False) := 
by
  sorry

end condition_sufficiency_but_not_necessity_l100_100453


namespace prove_ZAMENA_l100_100959

theorem prove_ZAMENA :
  ∃ (A M E H Z : ℕ),
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ Z ∧ Z ≤ 5 ∧
    (3 > A) ∧ (A > M) ∧ (M < E) ∧ (E < H) ∧ (H < A) ∧
    (A ≠ M) ∧ (A ≠ E) ∧ (A ≠ H) ∧ (A ≠ Z) ∧
    (M ≠ E) ∧ (M ≠ H) ∧ (M ≠ Z) ∧
    (E ≠ H) ∧ (E ≠ Z) ∧
    (H ≠ Z) ∧
    Z = 5 ∧
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234 :=
begin
  sorry
end

end prove_ZAMENA_l100_100959


namespace tan_45_deg_l100_100067

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100067


namespace range_of_m_l100_100694

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 1) : 
  ∀ m : ℝ, (x + 2 * y > m) ↔ (m < 8) :=
by 
  sorry

end range_of_m_l100_100694


namespace zamena_correct_l100_100950

variables {M E H A : ℕ}

def valid_digits := {1, 2, 3, 4, 5}

-- Define the conditions as hypotheses
def conditions : Prop :=
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
  M ∈ valid_digits ∧ E ∈ valid_digits ∧ H ∈ valid_digits ∧ A ∈ valid_digits ∧
  ∀ x y, x ∈ {M, E, H, A} ∧ y ∈ {M, E, H, A} → x ≠ y

-- Define the proof problem that these conditions imply the correct answer
theorem zamena_correct : conditions → M = 1 ∧ E = 2 ∧ H = 3 ∧ A = 4 ∧ 10000 * 5 + 1000 * A + 100 * M + 10 * E + H = 541234 :=
by sorry

end zamena_correct_l100_100950


namespace three_digit_numbers_divisible_by_11_l100_100314

theorem three_digit_numbers_divisible_by_11 :
  let count := (999 / 11).toInt - (100 / 11).toInt + 1
  count = 81 :=
by
  let count := (999 / 11).toInt - (100 / 11).toInt + 1
  show count = 81
  sorry

end three_digit_numbers_divisible_by_11_l100_100314


namespace find_exercise_books_l100_100739

theorem find_exercise_books
  (pencil_ratio pen_ratio exercise_book_ratio eraser_ratio : ℕ)
  (total_pencils total_ratio_units : ℕ)
  (h1 : pencil_ratio = 10)
  (h2 : pen_ratio = 2)
  (h3 : exercise_book_ratio = 3)
  (h4 : eraser_ratio = 4)
  (h5 : total_pencils = 150)
  (h6 : total_ratio_units = pencil_ratio + pen_ratio + exercise_book_ratio + eraser_ratio) :
  (total_pencils / pencil_ratio) * exercise_book_ratio = 45 :=
by
  sorry

end find_exercise_books_l100_100739


namespace tan_45_deg_l100_100050

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100050


namespace find_radius_of_k_l100_100520

def radius_k (R : ℝ) := 
  let k₁_radius := 8
  let k₂_radius := 15
  -- Condition: Shaded area inside k but outside k₁ equals the total shaded areas inside k₂
  (π * k₁_radius^2 + π * (k₂_radius^2 - k₁_radius^2)) = 
  (π * k₂_radius^2)
  
theorem find_radius_of_k : 
  (∃ R : ℝ, radius_k R ∧ R = 17 ∧ R > 0) := sorry

end find_radius_of_k_l100_100520


namespace experts_win_eventually_l100_100227

noncomputable def p : ℝ := 0.6
noncomputable def q : ℝ := 1 - p

def experts_need : ℕ := 3
def audience_need : ℕ := 2
def max_rounds : ℕ := 4

def winning_probability (experts_need : ℕ) (audience_need : ℕ) (p q : ℝ) : ℝ :=
  if experts_need = 3 ∧ audience_need = 2 ∧ p = 0.6 ∧ q = 0.4 then
    p^4 + 4 * p^3 * q
  else
    0

theorem experts_win_eventually :
  winning_probability experts_need audience_need p q = 0.4752 := sorry

end experts_win_eventually_l100_100227


namespace min_value_of_trig_function_l100_100248

-- Define the required trigonometric identity
def pythagorean_identity (x : ℝ) : Prop := sin^2 x + cos^2 x = 1

-- Define the function under consideration
def target_function (x : ℝ) : ℝ := (sin x + cos x)^4

-- The proof statement of the minimum value
theorem min_value_of_trig_function : ∀ x : ℝ, pythagorean_identity x → (∃ y : ℝ, target_function x = y ∧ y = 0) :=
by
  intros
  sorry

end min_value_of_trig_function_l100_100248


namespace sin_600_eq_neg_sqrt_3_div_2_l100_100901

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l100_100901


namespace closest_multiple_of_12_to_1987_l100_100498

def divisible_by_3 (n : ℕ) : Prop := n % 3 = 0
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0
def divisible_by_12 (n : ℕ) : Prop := divisible_by_3 n ∧ divisible_by_4 n

def distance (a b : ℕ) : ℕ := if a > b then a - b else b - a

theorem closest_multiple_of_12_to_1987 : 
  ∃ (n : ℕ), divisible_by_12 n ∧ ∀ (m : ℕ), divisible_by_12 m → distance 1987 m ≥ distance 1987 n :=
begin
  use 1984,
  split,
  { unfold divisible_by_12, split; norm_num, },
  { intros m h,
    unfold divisible_by_12 at h,
    unfold distance,
    cases h with h3 h4,
    sorry,  -- Proof steps to show 1984 is indeed the closest multiple to 1987 goes here
  }
end

end closest_multiple_of_12_to_1987_l100_100498


namespace frustum_lateral_surface_area_l100_100915

theorem frustum_lateral_surface_area (r1 r2 h : ℝ) (r1_eq : r1 = 10) (r2_eq : r2 = 4) (h_eq : h = 6) :
  let s := Real.sqrt (h^2 + (r1 - r2)^2)
  let A := Real.pi * (r1 + r2) * s
  A = 84 * Real.pi * Real.sqrt 2 :=
by
  sorry

end frustum_lateral_surface_area_l100_100915


namespace tan_45_deg_l100_100069

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100069


namespace real_a_range_l100_100836

noncomputable def f (x : ℝ) : ℝ :=
if (0 ≤ x) ∧ (x < 1) then l - x else - f (x - 1)

def g (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ :=
f x - a * x

def has_4_zeros (g : ℝ → ℝ) : Prop := -- A predicate that defines g has 4 zero points
sorry

theorem real_a_range (a : ℝ) (l : ℝ) :
  (has_4_zeros (g f a)) ↔ a ∈ (set.Ioo (1/5) (1/4)) ∪ (set.Ioo (-1/4) (-1/5)) := sorry

end real_a_range_l100_100836


namespace fraction_supercooled_water_freezing_l100_100863

variable (c : ℝ) (λ : ℝ) (T_initial : ℝ) (T_final : ℝ)

theorem fraction_supercooled_water_freezing (c_val : c = 4200) (λ_val : λ = 330000)
  (T_initial_val : T_initial = -10) (T_final_val : T_final = 0) :
  (c * 10 / λ) ≈ 0.13 :=
by
  sorry

end fraction_supercooled_water_freezing_l100_100863


namespace emily_can_see_leo_for_16_2_minutes_l100_100545

theorem emily_can_see_leo_for_16_2_minutes :
  ∀ (emily_speed leo_speed : ℝ) (distance_ahead distance_behind : ℝ), 
  emily_speed = 15 → leo_speed = 10 → 
  distance_ahead = 0.75 → distance_behind = 0.6 → 
  let relative_speed := emily_speed - leo_speed in
  let time_to_overtake := distance_ahead / relative_speed in
  let time_until_behind := distance_behind / relative_speed in
  (time_to_overtake + time_until_behind) * 60 = 16.2 :=
by
  intros emily_speed leo_speed distance_ahead distance_behind
  intros h1 h2 h3 h4
  let relative_speed := emily_speed - leo_speed
  let time_to_overtake := distance_ahead / relative_speed
  let time_until_behind := distance_behind / relative_speed
  sorry

end emily_can_see_leo_for_16_2_minutes_l100_100545


namespace parallelogram_area_l100_100848

theorem parallelogram_area :
  ∀ (z1 z2 z3 z4 : ℂ) 
    (h1 : z1^2 = 4 + 4 * sqrt 15 * complex.I) 
    (h2 : z2^2 = 4 + 4 * sqrt 15 * complex.I) 
    (h3 : z3^2 = 2 + 2 * sqrt 3 * complex.I) 
    (h4 : z4^2 = 2 + 2 * sqrt 3 * complex.I), 
  let area := 2 * sqrt 21 - 2 * sqrt 3 in
  exists (p q r s : ℕ),
    area = p * sqrt q - r * sqrt s ∧
    ¬ ∃ (k : ℕ), k > 1 ∧ q = k^2 ∧ ¬ ∃ (k : ℕ), k > 1 ∧ s = k^2 ∧
    p + q + r + s = 20 :=
begin
  intros,
  sorry
end

end parallelogram_area_l100_100848


namespace number_of_ordered_pairs_l100_100706

theorem number_of_ordered_pairs : 
  (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a^2 + b^2 = 50) ↔ 3 :=
sorry

end number_of_ordered_pairs_l100_100706


namespace tan_45_deg_l100_100062

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100062


namespace experts_win_probability_l100_100219

noncomputable def probability_of_experts_winning (p : ℝ) (q : ℝ) (needed_expert_wins : ℕ) (needed_audience_wins : ℕ) : ℝ :=
  p ^ 4 + 4 * (p ^ 3 * q)

-- Probability values
def p : ℝ := 0.6
def q : ℝ := 1 - p

-- Number of wins needed
def needed_expert_wins : ℕ := 3
def needed_audience_wins : ℕ := 2

theorem experts_win_probability :
  probability_of_experts_winning p q needed_expert_wins needed_audience_wins = 0.4752 :=
by
  -- Proof would go here
  sorry

end experts_win_probability_l100_100219


namespace exchange_rate_change_l100_100550

theorem exchange_rate_change :
  let initial := 32.6587
  let final := 56.2584
  let change := final - initial
  let rounded_change := Real.round change
  rounded_change = 24 :=
by
  sorry

end exchange_rate_change_l100_100550


namespace triangle_area_l100_100868

noncomputable def radius1 : ℝ := 1 / 18
noncomputable def radius2 : ℝ := 2 / 9
noncomputable def AL : ℝ := 1 / 9
noncomputable def CM : ℝ := 1 / 6

theorem triangle_area 
  (r : ℝ) (R : ℝ) (AL : ℝ) (CM : ℝ) 
  (touching1 : r = radius1) 
  (touching2 : R = radius2) 
  (lenAL : AL = 1 / 9) 
  (lenCM : CM = 1 / 6) 
: 
  let AC := AL + 2 * real.sqrt(r * R) + CM in
  let sinB := (4/5 * -7/25 + 3/5 * 24/25) / 1 * real.sqrt((real.of_rat (4/5)^2 - 1) * (real.of_rat (9/25)^2 - 1) / ((real.of_rat (1 / 9)^2 + real.of_rat (1 / 18)^2) * (real.of_rat (1 / 6)^2 + real.of_rat (2 / 9)^2))) in
  let AB := AC * 24/25 / sinB in
  1 / 2 * AB * AC * 4/5 = 15 / 22 :=
begin
  intros,
  unfold radius1 radius2,
  subst_vars,
  let AC := AL + 2 * real.sqrt(r * R) + CM,
  have h_kn : 2 * real.sqrt(r * R) = 2 / 9,
  { rw [sqrt_div, sqrt_mul, mul_div_assoc, div_self, mul_one, div_div],
    linarith, linarith },
  let sinB := (4/5 * -7/25 + 3/5 * 24/25) / (1 * sqrt((real.of_rat (4/5)^2 - 1) * (real.of_rat (9/25)^2 - 1) / ((real.of_rat (1 / 9)^2 + real.of_rat (1 / 18)^2) * (real.of_rat (1 / 6)^2 + real.of_rat (2 / 9)^2))),
  let AB := AC * (24/25) / sinB,
  exact sorry
end

end triangle_area_l100_100868


namespace product_of_all_possible_y_coords_l100_100814

theorem product_of_all_possible_y_coords :
  ∃ y1 y2 : ℝ, (-4) ∈ ℝ ∧
  (∀ y, dist (7, 3 : ℝ × ℝ) (-4, y) = 13 → 
   (y = y1 ∨ y = y2) ∧ y1 * y2 = -39) := 
sorry

end product_of_all_possible_y_coords_l100_100814


namespace carpenter_material_cost_l100_100908

theorem carpenter_material_cost (total_estimate hourly_rate num_hours : ℝ) 
    (h1 : total_estimate = 980)
    (h2 : hourly_rate = 28)
    (h3 : num_hours = 15) : 
    total_estimate - hourly_rate * num_hours = 560 := 
by
  sorry

end carpenter_material_cost_l100_100908


namespace sum_of_integer_solutions_l100_100875

theorem sum_of_integer_solutions : 
  (∑ n in Finset.filter (λ n, |n| < |n - 3|) (Finset.Icc (-7) 12), n) = -27 := 
by
  sorry

end sum_of_integer_solutions_l100_100875


namespace tan_45_deg_eq_one_l100_100018

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100018


namespace probability_of_odd_total_l100_100398

open ProbabilityTheory

-- Given conditions in Lean definitions
def prob_odd_die : ℚ := 11 / 21
def prob_even_die : ℚ := 10 / 21

-- The goal to prove
theorem probability_of_odd_total :
  (prob_odd_die * prob_odd_die) + (prob_even_die * prob_even_die) = 221 / 441 := 
by
  sorry

end probability_of_odd_total_l100_100398


namespace road_construction_problem_l100_100440

-- Define the constants for the roadwork problem
constant road_length : ℕ := 1000
constant days_A_initial : ℕ := 5
constant days_together : ℕ := 8
constant meters_left : ℕ := 30
constant days_A_work2 : ℕ := 2
constant meters_less_by_A : ℕ := 20
constant days_B_work2 : ℕ := 3
constant total_days : ℕ := 12
constant days_together_initial : ℕ := 4

-- Team outputs
constant x : ℕ -- Output of Team A (meters/day)
constant y : ℕ -- Output of Team B (meters/day)
constant m : ℕ -- Additional output required by Team B (meters/day)

-- Main theorem to prove the conditions and answer
theorem road_construction_problem:
  (13 * x + 8 * y = 970) → 
  (2 * x + meters_less_by_A = 3 * y) → 
  (4 * (x + y) + 8 * (y + m) ≥ road_length) → 
  x = 50 ∧ 
  y = 40 ∧ 
  m ≥ 40 := 
by
  sorry

end road_construction_problem_l100_100440


namespace tan_45_eq_1_l100_100000

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100000


namespace circle_radius_l100_100941

variable (E : Type)
variable [Ellipse E]
  (major_axis : ℝ)
  (minor_axis : ℝ)
  (focus : E)
  (r : ℝ)

-- Conditions
def ellipse_property : Prop :=
  major_axis = 12 ∧ minor_axis = 6

def circle_tangent_property : Prop :=
  ∃ center : E, center = focus ∧ ∀ point ∈ cons.ellipse, circle.radius point = r

-- Theorem statement
theorem circle_radius (ellipse_property : ellipse_property) (circle_tangent_property : circle_tangent_property) : 
  r = 3 := 
sorry

end circle_radius_l100_100941


namespace intersection_eq_one_l100_100688

def M : Set ℕ := {0, 1}
def N : Set ℕ := {y | ∃ x ∈ M, y = x^2 + 1}

theorem intersection_eq_one : M ∩ N = {1} := 
by
  sorry

end intersection_eq_one_l100_100688


namespace tan_45_eq_1_l100_100015

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100015


namespace tan_45_deg_l100_100046

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100046


namespace possible_values_of_n_l100_100354

theorem possible_values_of_n : 
  ∀ n : ℕ, 1 < n ∧ n < 9 → 7 = { k : ℕ | 1 < k ∧ k < 9 }.to_finset.card :=
  by sorry

end possible_values_of_n_l100_100354


namespace solution1_solution2_l100_100680

def f (x : ℝ) := |x - 2|

theorem solution1 (x : ℝ) : f(x) + f(x + 1) ≥ 5 ↔ x ≥ 4 ∨ x ≤ -1 := 
by sorry

theorem solution2 (a b : ℝ) (h1 : |a| > 1) (h2 : f(a * b) > |a| * f(b / a)) : |b| > 2 := 
by sorry

end solution1_solution2_l100_100680


namespace number_of_six_digit_numbers_formable_by_1_2_3_4_l100_100530

theorem number_of_six_digit_numbers_formable_by_1_2_3_4
  (digits : Finset ℕ := {1, 2, 3, 4})
  (pairs_count : ℕ := 2)
  (non_adjacent_pair : ℕ := 1)
  (adjacent_pair : ℕ := 1)
  (six_digit_numbers : ℕ := 432) :
  ∃ (n : ℕ), n = 432 :=
by
  -- Proof will go here
  sorry

end number_of_six_digit_numbers_formable_by_1_2_3_4_l100_100530


namespace derivative_at_minus_2_l100_100318

theorem derivative_at_minus_2 (f : ℝ → ℝ) 
  (h : filter.tendsto (λ Δx, (f (-2 + Δx) - f (-2 - Δx)) / Δx) (nhds 0) (nhds (-2))) :
  deriv f (-2) = -1 := 
sorry

end derivative_at_minus_2_l100_100318


namespace tan_45_deg_l100_100047

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100047


namespace max_distance_from_P_to_l_l100_100842

open Real

noncomputable def point_P : ℝ × ℝ := (-2, -1)

def line_l (λ : ℝ) := { p : ℝ × ℝ | (1 + 3 * λ) * p.1 + (1 + λ) * p.2 - 2 - 4 * λ = 0 }

theorem max_distance_from_P_to_l : ∃ A : ℝ × ℝ, (λ : ℝ) → A ∈ line_l λ ∧ dist point_P A = sqrt 13 := by
  sorry

end max_distance_from_P_to_l_l100_100842


namespace log9_4500_round_to_nearest_l100_100552

theorem log9_4500_round_to_nearest :
  9^3 = 729 → 9^4 = 6561 → 729 < 4500 → 4500 < 6561 → Int.round (Real.log 4500 / Real.log 9) = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end log9_4500_round_to_nearest_l100_100552


namespace number_of_distinct_products_l100_100779

def is_divisor (a b : ℕ) : Prop := b % a = 0

def S : Set ℕ := {n | is_divisor n 40000 ∧ n > 0}

def is_product_of_two_distinct_elems (m : ℕ) : Prop :=
  ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ m = a * b

theorem number_of_distinct_products : (Set.card {m | is_product_of_two_distinct_elems m}) = 73 :=
by 
  sorry

end number_of_distinct_products_l100_100779


namespace angle_AQD_eq_angle_BQC_l100_100468

-- Define points P and Q
variables (P Q : Point)

-- Define the three circles and their intersections
variables (circle1 circle2 : Circle)
variables (circle3 : Circle) (h_center : circle3.center = P)
variables (A B : Point) (h_A : A ∈ circle1 ∧ A ∈ circle3) (h_B : B ∈ circle1 ∧ B ∈ circle3)
variables (C D : Point) (h_C : C ∈ circle2 ∧ C ∈ circle3) (h_D : D ∈ circle2 ∧ D ∈ circle3)
variables (h_intersect1 : P ∈ circle1 ∧ Q ∈ circle1)
variables (h_intersect2 : P ∈ circle2 ∧ Q ∈ circle2)

-- The theorem statement
theorem angle_AQD_eq_angle_BQC :
  ∠(A, Q, D) = ∠(B, Q, C) :=
sorry

end angle_AQD_eq_angle_BQC_l100_100468


namespace rotation_matrix_90_deg_clockwise_l100_100482

theorem rotation_matrix_90_deg_clockwise :
  ∃ (N : matrix (fin 2) (fin 2) ℝ),
    (∀ v : vector ℝ 2, (v = ![(1 : ℝ), 1] ∨ v = ![-1, 1] ∨ v = ![-1, -1] ∨ v = ![1, -1]) →
    ((N.mul_vec v = ![v.x - v.y, v.x + v.y])) ∧
    N = ![![(0 : ℝ), 1], ![-1, 0]]) :=
begin
  use ![![(0 : ℝ), 1], ![-1, 0]],
  intros v hv,
  split,
  {
    cases hv; simp [matrix.mul_vec, matrix.dot_product, fin.val_zero_eq, fin.val_one_eq],
    case or.inl {
      -- Case v = [1, 1]
      simp
    },
    case or.inr or.inl {
      -- Case v = [-1, 1]
      simp
    },
    case or.inr or.inr or.inl {
      -- Case v = [-1, -1]
      simp
    },
    case or.inr or.inr or.inr {
      -- Case v = [1, -1]
      simp
    }
  },
  refl
end

end rotation_matrix_90_deg_clockwise_l100_100482


namespace sum_of_first_10_terms_value_of_x_when_30_zeros_in_100_terms_l100_100698

open Nat

noncomputable def sequence (x : ℕ) : ℕ → ℕ
| 0       => 0
| 1       => 1
| 2       => x
| (n + 1) => abs (sequence x n - sequence x (n - 1))

theorem sum_of_first_10_terms (x : ℕ) (h : x = 2) : 
  (Finset.range 10).sum (sequence x) = 9 :=
by
  sorry

theorem value_of_x_when_30_zeros_in_100_terms (x : ℕ) 
  (h : (Finset.range 101).filter (λ n => sequence x n = 0).card = 30) : 
  x = 6 ∨ x = 7 :=
by
  sorry

end sum_of_first_10_terms_value_of_x_when_30_zeros_in_100_terms_l100_100698


namespace vampire_count_after_two_nights_l100_100463

noncomputable def vampire_growth : Nat :=
  let first_night_new_vampires := 3 * 7
  let total_vampires_after_first_night := first_night_new_vampires + 3
  let second_night_new_vampires := total_vampires_after_first_night * (7 + 1)
  second_night_new_vampires + total_vampires_after_first_night

theorem vampire_count_after_two_nights : vampire_growth = 216 :=
by
  -- Skipping the detailed proof steps for now
  sorry

end vampire_count_after_two_nights_l100_100463


namespace max_value_of_function_l100_100615

theorem max_value_of_function : 
  (∀ x : ℝ, f x = 6 * sin x + 8 * cos x) →
  ∃ m : ℝ, (∀ x : ℝ, f x ≤ m) ∧ (∃ x₀ : ℝ, f x₀ = m) :=
begin
  sorry
end

end max_value_of_function_l100_100615


namespace solve_for_z_l100_100430

noncomputable def solve_z (z : ℂ) : Prop :=
  3 + 2 * complex.I * z = 4 - 5 * complex.I * z

theorem solve_for_z : ∃ z : ℂ, solve_z z ∧ z = -complex.I / 7 :=
by
  sorry

end solve_for_z_l100_100430


namespace tan_of_45_deg_l100_100095

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100095


namespace excircle_tangent_segment_length_l100_100649

theorem excircle_tangent_segment_length (A B C M : ℝ) 
  (h1 : A + B + C = 1) 
  (h2 : M = (1 / 2)) : 
  M = 1 / 2 := 
  by
    -- This is where the proof would go
    sorry

end excircle_tangent_segment_length_l100_100649


namespace selina_sells_pants_l100_100819

theorem selina_sells_pants (P : ℕ) 
  (pants_price shorts_price shirts_price final_money shirt_purchase_price : ℕ) 
  (shorts_sold shirts_sold new_shirts : ℕ) 
  (total_earned : ℕ) :
  pants_price = 5 →
  shorts_price = 3 →
  shirts_price = 4 →
  final_money = 30 →
  shirt_purchase_price = 10 →
  shorts_sold = 5 →
  shirts_sold = 5 →
  new_shirts = 2 →
  total_earned = final_money + new_shirts * shirt_purchase_price →
  total_earned = pants_price * P + shorts_price * shorts_sold + shirts_price * shirts_sold →
  P = 3 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  rw [h1, h2, h3, h4, h5, h6, h7, h8, h9]
  sorry

end selina_sells_pants_l100_100819


namespace smallest_integer_mod_conditions_l100_100493

theorem smallest_integer_mod_conditions : 
  ∃ x : ℕ, 
  (x % 4 = 3) ∧ (x % 3 = 2) ∧ (∀ y : ℕ, (y % 4 = 3) ∧ (y % 3 = 2) → x ≤ y) ∧ x = 11 :=
by
  sorry

end smallest_integer_mod_conditions_l100_100493


namespace tan_45_deg_l100_100150

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100150


namespace f_2010_minus_sin_l100_100382

noncomputable def f : ℕ → (ℝ → ℝ)
| 0       := λ x, Real.sin x
| (n+1)   := λ x, (f n) x.derivative

theorem f_2010_minus_sin (x : ℝ) : f 2010 x = -Real.sin x :=
by
  sorry

end f_2010_minus_sin_l100_100382


namespace experts_eventual_win_probability_l100_100238

noncomputable def experts_win_probability (p : ℝ) (q : ℝ) (experts_needed : ℕ) (audience_needed : ℕ) (max_rounds : ℕ) : ℝ :=
  let e_all_wins := p ^ max_rounds
  let e_three_wins_one_loss := (Nat.choose max_rounds (max_rounds - 1)) * (p ^ (max_rounds - 1)) * q
  e_all_wins + e_three_wins_one_loss

theorem experts_eventual_win_probability :
  ∀ (p : ℝ) (q : ℝ),
  p = 0.6 →
  q = 1 - p →
  experts_win_probability p q 3 2 4 = 0.4752 := by
  intros p q hp hq
  have h : experts_win_probability p q 3 2 4 = 0.6 ^ 4 + 4 * (0.6 ^ 3) * 0.4 := sorry
  rw [hp, hq] at h
  norm_num at h
  exact h

end experts_eventual_win_probability_l100_100238


namespace not_q_true_l100_100725

theorem not_q_true (p q : Prop) (hp : p = true) (hq : q = false) : ¬q = true :=
by
  sorry

end not_q_true_l100_100725


namespace tan_45_deg_eq_one_l100_100167

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100167


namespace maximum_profit_l100_100910

noncomputable def R (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then
  10.8 - (1/30) * x^2
else
  108 / x - 1000 / (3 * x^2)

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then
  x * R x - (10 + 2.7 * x)
else
  x * R x - (10 + 2.7 * x)

theorem maximum_profit : 
  ∃ x : ℝ, (0 < x ∧ x ≤ 10 → W x = 8.1 * x - (x^3 / 30) - 10) ∧ 
           (x > 10 → W x = 98 - 1000 / (3 * x) - 2.7 * x) ∧ 
           (∃ xmax : ℝ, xmax = 9 ∧ W 9 = 38.6) := 
sorry

end maximum_profit_l100_100910


namespace correct_transformation_D_l100_100883

theorem correct_transformation_D : ∀ x, 2 * (x + 1) = x + 7 → x = 5 :=
by
  intro x
  sorry

end correct_transformation_D_l100_100883


namespace find_other_number_l100_100451

theorem find_other_number (B : ℕ)
  (HCF : Nat.gcd 24 B = 12)
  (LCM : Nat.lcm 24 B = 312) :
  B = 156 :=
by
  sorry

end find_other_number_l100_100451


namespace ruby_reading_homework_l100_100807

theorem ruby_reading_homework :
  ∃ R : ℕ, 
    let nina_math := 4 * 6 in
    let nina_reading := 8 * R in
    nina_math + nina_reading = 48 ∧ R = 3 :=
by {
  -- The formulation of the proof steps will go here.
  sorry,
}

end ruby_reading_homework_l100_100807


namespace exchange_rate_change_l100_100549

theorem exchange_rate_change (initial final : ℝ) (h_initial : initial = 32.6587) (h_final : final = 56.2584) : 
  Float.round (final - initial) = 24 := by
  sorry

end exchange_rate_change_l100_100549


namespace number_of_non_summables_l100_100307

def the_set : Set ℕ := {1, 2, 3, 5, 8, 13, 21, 34, 55}
def range_set : Set ℕ := { n | 3 ≤ n ∧ n ≤ 89 }

def sum_of_two (s : Set ℕ) (n : ℕ) : Prop :=
  ∃ x y ∈ s, x + y = n

theorem number_of_non_summables : 
  let non_summable_set := range_set \ { n | sum_of_two the_set n } in
  non_summable_set.card = 53 :=
by
  sorry

end number_of_non_summables_l100_100307


namespace correct_statements_l100_100881

-- Definitions based on conditions
def generatrices_of_cone (l1 l2 : Line) (cone : Cone) : Prop := 
  cone.contains_generatrix l1 ∧ cone.contains_generatrix l2

def lines_extend_to_meet (l1 l2 : Line) : Prop := 
  ∃ p : Point, l1.contains p ∧ l2.contains p

def two_lines_parallel_iff_no_common_points (l1 l2 : Line) : Prop := 
  l1.parallel_to l2 ↔ ¬ ∃ p : Point, l1.contains p ∧ l2.contains p

def plane_determined_by_point_and_line (p : Point) (l : Line) (pl : Plane) : Prop :=
  (pl.contains_point p ∧ pl.contains_line l) → pl.unique_determined

def line_parallel_to_plane_has_inf_parallel_in_plane (l : Line) (pl : Plane) : Prop :=
  l.parallel_to pl → ∃ infinitely_many_lines : Line, pl.contains_line line ∧ line.parallel_to l

-- Statements to prove:
theorem correct_statements : 
  (∀ l1 l2 cone, generatrices_of_cone l1 l2 cone → lines_extend_to_meet l1 l2) ∧ 
  (∀ l1 l2, ¬ two_lines_parallel_iff_no_common_points l1 l2) ∧
  (∀ p l pl, ¬ plane_determined_by_point_and_line p l pl) ∧ 
  (∀ l pl, line_parallel_to_plane_has_inf_parallel_in_plane l pl) :=
by 
  sorry

end correct_statements_l100_100881


namespace tan_five_pi_over_four_l100_100594

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l100_100594


namespace triangle_dimensions_l100_100929

theorem triangle_dimensions (a b c : ℕ) (h1 : a > b) (h2 : b > c)
  (h3 : a = 2 * c) (h4 : b - 2 = c) (h5 : 2 * a / 3 = b) :
  a = 12 ∧ b = 8 ∧ c = 6 :=
by
  sorry

end triangle_dimensions_l100_100929


namespace tan_45_eq_1_l100_100193

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100193


namespace tan_45_deg_eq_one_l100_100176

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100176


namespace tan_45_deg_eq_one_l100_100036

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100036


namespace barb_paid_less_than_half_l100_100546

variable (savings original_price : ℕ)

theorem barb_paid_less_than_half (h1 : savings = 80) (h2 : original_price = 180) :
  abs ((original_price / 2) - (original_price - savings)) = 10 :=
by {
  sorry
}

end barb_paid_less_than_half_l100_100546


namespace good_numbers_transformation_l100_100920

def is_digit (n : ℕ) : Prop := n > 0 ∧ n < 10
def is_good_number (n : ℕ) : Prop := ∀ d ∈ n.digits 10, is_digit d
def is_special_number (k : ℕ) (n : ℕ) : Prop :=
  n.digits 10.length ≥ k ∧ (∀ i j, i < j → (n.digits 10) !! i < (n.digits 10) !! j)

theorem good_numbers_transformation (k : ℕ) :
  (k = 8) →
  (∀ n m : ℕ, is_good_number n → is_good_number m → 
  (∃ p, is_special_number k p ∧ (n.append_special p = m ∨ m.append_special p = n 
  ∨ ∃ q, n.insert_special q = m ∨ m.insert_special q = n 
  ∨ ∃ r, n.erase_special r = m ∨ m.erase_special r = n))) :=
by
  intros _
  sorry

end good_numbers_transformation_l100_100920


namespace tan_45_deg_eq_one_l100_100035

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100035


namespace tan_45_deg_l100_100055

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100055


namespace thomas_savings_l100_100462

def first_year_earnings (weekly_allowance_year1 : ℕ) : ℕ :=
  weekly_allowance_year1 * 52

def second_year_earnings (hourly_rate_year2 hours_per_week_year2 : ℕ) : ℕ :=
  hourly_rate_year2 * hours_per_week_year2 * 52

def total_earnings (first_year second_year : ℕ) : ℕ :=
  first_year + second_year

def total_expenses (personal_expenses : ℕ) : ℕ :=
  personal_expenses * (52 * 2)

def savings (earnings expenses : ℕ) : ℕ :=
  earnings - expenses

def amount_needed (car_cost savings : ℕ) : ℕ :=
  car_cost - savings

theorem thomas_savings : 
  ∀ (weekly_allowance_year1 hourly_rate_year2 hours_per_week_year2 car_cost personal_expenses : ℕ),
  years = 2 →
  weekly_allowance_year1 = 50 →
  hourly_rate_year2 = 9 →
  hours_per_week_year2 = 30 →
  car_cost = 15000 →
  personal_expenses = 35 →
  amount_needed car_cost (
    savings (
      total_earnings 
        (first_year_earnings weekly_allowance_year1)
        (second_year_earnings hourly_rate_year2 hours_per_week_year2)
      )
    (total_expenses personal_expenses)
  ) = 2000
:=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end thomas_savings_l100_100462


namespace sin_cos_sum_l100_100657

-- Define the angle θ in the third quadrant
def in_third_quadrant (θ : Real) : Prop :=
  θ > π ∧ θ < 3 * π / 2

-- Define the condition of the problem
def tan_condition (θ : Real) : Prop :=
  tan (θ - π / 4) = 1 / 3

-- State the main proof goal
theorem sin_cos_sum (θ : Real) (hθ : in_third_quadrant θ) (htan : tan_condition θ) :
  sin θ + cos θ = - (3 / 5) * sqrt 5 :=
sorry

end sin_cos_sum_l100_100657


namespace part1_part2_l100_100787

theorem part1 (u v w : ℤ) (h_uv : gcd u v = 1) (h_vw : gcd v w = 1) (h_wu : gcd w u = 1) 
: gcd (u * v + v * w + w * u) (u * v * w) = 1 :=
sorry

theorem part2 (u v w : ℤ) (b := u * v + v * w + w * u) (c := u * v * w) (h : gcd b c = 1) 
: gcd u v = 1 ∧ gcd v w = 1 ∧ gcd w u = 1 :=
sorry

end part1_part2_l100_100787


namespace num_solutions_of_complex_numbers_l100_100626

theorem num_solutions_of_complex_numbers (z : ℂ) (h1 : |z| = 1) 
(h2 : |z / conj z - conj z / z| = 1) : 
  ∃ sol_set, sol_set.card = 4 ∧ (∀ z ∈ sol_set, |z| = 1 ∧ |z / conj z - conj z / z| = 1) := 
  sorry

end num_solutions_of_complex_numbers_l100_100626


namespace jan_drove_more_than_ian_l100_100888

variables (d t s : ℝ)

-- Ian's distance relation
def ian_distance : Prop := d = s * t

-- Han's driving relation derived by simplifying d + 100 = (s + 10) * (t + 2)
def han_condition : Prop := 5 * t + s = 40

-- Jan's distance relation
def jan_condition : Prop := m = (s + 15) * (t + 3)

-- The final proposition we need to prove
def jan_drove_165_more : Prop := (s + 15) * (t + 3) - d = 165

-- Final theorem statement
theorem jan_drove_more_than_ian (h1 : ian_distance d t s) (h2 : han_condition d t s) (h3 : jan_condition d t s) :
  jan_drove_165_more d t s :=
sorry

end jan_drove_more_than_ian_l100_100888


namespace min_value_of_OC_l100_100695

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

def OA : ℝ × ℝ := (3, 1)
def OB : ℝ × ℝ := (-1, 3)

def OC (m n : ℝ) : ℝ × ℝ := (3 * m + n, m - 3 * n)

theorem min_value_of_OC (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  vector_norm (OC m n) = real.sqrt 5 :=
sorry

end min_value_of_OC_l100_100695


namespace minimum_value_expression_l100_100259

theorem minimum_value_expression (x : ℝ) (h : x > 4) : 
  ∃ (m : ℝ), m = 6 ∧ ∀ y : ℝ, y = (x + 5) / (Real.sqrt (x - 4)) → y ≥ m :=
by
  -- proof goes here
  sorry

end minimum_value_expression_l100_100259


namespace intersection_point_correct_l100_100479

noncomputable def line1 (x : ℝ) : ℝ := 3 * x + 4
noncomputable def line2 (x : ℝ) : ℝ := - (1/3) * x + 5

def intersection_point : ℝ × ℝ :=
  let x := 3 / 10 in
  (x, line1 x)

theorem intersection_point_correct :
  intersection_point = (3 / 10, 49 / 10) := by
  sorry

end intersection_point_correct_l100_100479


namespace triangle_angle_and_incircle_area_l100_100357

theorem triangle_angle_and_incircle_area (a b c : ℝ) (A B C : ℝ) (BD AD : ℝ):
  a + 2 * b = c * Real.cos B + sqrt 3 * c * Real.sin B →
  BD = 35 / 4 →
  AD = 21 / 4 →
  ∠ ACB = 120 → 
  is_area_of_incircle := 3 * Real.pi :=
sorry

end triangle_angle_and_incircle_area_l100_100357


namespace slope_PQ_is_zero_l100_100919

noncomputable theory

-- Definitions based on the conditions in the problem
def curve_C (x y : ℝ) : Prop := y^2 = 4 * x

def point_P (k : ℝ) : ℝ × ℝ := (0, 1 / k)

def point_A (k : ℝ) (x y : ℝ) : Prop := 
  curve_C x y ∧ y = k * x + 1 / k

def midpoint_Q (k : ℝ) : ℝ × ℝ := (1 / (2 * k^2), 1 / k)

-- Proof goal statement
theorem slope_PQ_is_zero (k : ℝ) (A : ℝ × ℝ) 
  (hA : point_A k A.1 A.2) :
  let P := point_P k in
  let Q := midpoint_Q k in
  (Q.2 - P.2) / (Q.1 - P.1) = 0 :=
by 
  -- The proof content will go here.
  sorry

end slope_PQ_is_zero_l100_100919


namespace find_n_from_A_C_l100_100637

noncomputable def A_n (n : ℕ) : ℕ := n! / (n - 2)!
noncomputable def C_n (n : ℕ) (k : ℕ) : ℕ := n! / (k! * (n - k)!)

theorem find_n_from_A_C (n : ℕ) (h : (A_n n)^2 = C_n n (n - 3)) : n = 8 := by
  sorry

end find_n_from_A_C_l100_100637


namespace tan_five_pi_over_four_l100_100589

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l100_100589


namespace tan_45_deg_l100_100049

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100049


namespace solve_ZAMENA_l100_100978

noncomputable def ZAMENA : ℕ := 541234

theorem solve_ZAMENA :
  ∃ (A M E H : ℕ), 
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H ∧
    ZAMENA = 541234 :=
by {
  have h1 : ∃ (A M E H : ℕ), 
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H, sorry,
  exact ⟨2, 1, 2, 3, h1⟩,
  repeat { sorry }
}

end solve_ZAMENA_l100_100978


namespace tan_five_pi_over_four_l100_100585

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
sorry

end tan_five_pi_over_four_l100_100585


namespace tan_45_deg_l100_100145

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100145


namespace village_population_increase_rate_l100_100473

theorem village_population_increase_rate (initial_population_X : ℕ) (initial_population_Y : ℕ)
  (rate_decrease_X : ℕ) (time : ℕ) (population_equality_time : ℕ) :
  initial_population_X = 78000 →
  initial_population_Y = 42000 →
  rate_decrease_X = 1200 →
  time = 18 →
  population_equality_time = 18 →
  let final_population_X := initial_population_X - rate_decrease_X * time in
  let final_population_Y := initial_population_Y + rate_increase_Y * time in
  final_population_X = final_population_Y →
  rate_increase_Y = 800 :=
by {
  intros h1 h2 h3 h4 h5;
  rw [h1, h2, h3, h4, h5];
  let final_population_X := 78000 - 1200 * 18;
  let final_population_Y := 42000 + rate_increase_Y * 18;
  change final_population_X = final_population_Y;
  let rate_increase_Y := 800;
  sorry
}

end village_population_increase_rate_l100_100473


namespace cousins_initial_money_l100_100559

theorem cousins_initial_money (x : ℕ) :
  let Carmela_initial := 7
  let num_cousins := 4
  let gift_each := 1
  Carmela_initial - num_cousins * gift_each = x + gift_each →
  x = 2 :=
by
  intro h
  sorry

end cousins_initial_money_l100_100559


namespace triangle_is_isosceles_l100_100330

def triangle_shape (A B C : ℝ) (h : A + B + C = Real.pi) : Prop :=
  ∃ (a b c : ℝ), ∠ a b c = A ∧ ∠ b a c = B ∧ ∠ a c b = C

theorem triangle_is_isosceles (A B C : ℝ) (h₁ : A + B + C = Real.pi) (h₂ : Real.sin C = 2 * Real.cos A * Real.sin B) :
  A = B ∨ B = C ∨ C = A :=
by
  sorry

end triangle_is_isosceles_l100_100330


namespace prove_inequality_l100_100389

noncomputable def conditions_prove : Prop :=
  ∀ (a b c : ℝ), 
    0 < a ∧ 0 < b ∧ 0 < c → 
    a^b > b^a ∧ b^c > c^b → 
    a^c > c^a

theorem prove_inequality (a b c : ℝ) (hab: 0 < a) (hbc: 0 < b) (hcc: 0 < c)
  (h1: a^b > b^a) (h2: b^c > c^b) : a^c > c^a :=
begin
  sorry
end

end prove_inequality_l100_100389


namespace tan_45_deg_l100_100056

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100056


namespace tan_45_deg_l100_100059

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100059


namespace tan_45_eq_1_l100_100185

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100185


namespace find_AC_and_sin_ACF_l100_100942

-- Define a structure for the conditions given
structure CircleTangentsTriangle where
  AB : ℝ
  CE_ratio : ℝ
  DE_ratio : ℝ
  DF_ratio : ℝ

-- The problem condition instance
def problem_instance : CircleTangentsTriangle :=
  { AB := 24,
    CE_ratio := 4,
    DE_ratio := 3,
    DF_ratio := 3 }

-- Define an example proof statement for the problem
theorem find_AC_and_sin_ACF (c : CircleTangentsTriangle)
  (h1 : c.AB = 24)
  (h2 : c.CE_ratio / c.DE_ratio = 4 / 3)
  (h3 : c.DE_ratio / c.DF_ratio = 3 / 3) :
  (AC = 8 * Real.sqrt 5) ∧ (Real.sin (angle_ACF c) = Real.sqrt 6 / 6) :=
by
  sorry -- Proof goes here

end find_AC_and_sin_ACF_l100_100942


namespace tan_45_deg_l100_100057

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100057


namespace ratio_female_to_male_l100_100433

namespace DeltaSportsClub

variables (f m : ℕ) -- number of female and male members
-- Sum of ages of female and male members respectively
def sum_ages_females := 35 * f
def sum_ages_males := 30 * m
-- Total sum of ages
def total_sum_ages := sum_ages_females f + sum_ages_males m
-- Total number of members
def total_members := f + m

-- Given condition on the average age of all members
def average_age_condition := (total_sum_ages f m) / (total_members f m) = 32

-- The target theorem to prove the ratio of female to male members
theorem ratio_female_to_male (h : average_age_condition f m) : f/m = 2/3 :=
by sorry

end DeltaSportsClub

end ratio_female_to_male_l100_100433


namespace functional_equation_l100_100784

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f (x * f y + x^2) = x * y + x + f x) :
    let m := 1,
        t := f 3,
        v := m * t in
    v = 3 :=
  by
  sorry

end functional_equation_l100_100784


namespace ZAMENA_correct_l100_100947

-- Define the digits and assigning them to the letters
noncomputable def A : ℕ := 2
noncomputable def M : ℕ := 1
noncomputable def E : ℕ := 2
noncomputable def H : ℕ := 3

-- Define the number ZAMENA
noncomputable def ZAMENA : ℕ :=
  5 * 10^5 + A * 10^4 + M * 10^3 + E * 10^2 + H * 10 + A

-- Define the inequalities and constraints
lemma satisfies_inequalities  : 3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Define uniqueness of each digit set
lemma unique_digits: A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ M ≠ E ∧ M ≠ H ∧ E ≠ H := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Prove that the number ZAMENA matches the specified number
theorem ZAMENA_correct : ZAMENA = 541234 := by
  unfold ZAMENA A M E H
  norm_num

end ZAMENA_correct_l100_100947


namespace area_triangle_AGE_l100_100401

noncomputable def point := (ℝ × ℝ)
noncomputable def triangle_area (a b c : point) : ℝ :=
  0.5 * abs ((fst a - fst c) * (snd b - snd a) - (fst a - fst b) * (snd c - snd a))

theorem area_triangle_AGE : 
  let A := (0, 0) 
  let B := (5, 0) 
  let C := (5, 5) 
  let D := (0, 5) 
  let E := (5, 2) 
  ∃ G : point, 
  (circumcircle (A, B, E)) ∩ (line (B, D)) = {G} ∧
  triangle_area A G E = 44.5 :=
by 
  sorry

end area_triangle_AGE_l100_100401


namespace tan_45_deg_eq_one_l100_100117

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100117


namespace tan_45_deg_l100_100148

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100148


namespace zamena_correct_l100_100953

variables {M E H A : ℕ}

def valid_digits := {1, 2, 3, 4, 5}

-- Define the conditions as hypotheses
def conditions : Prop :=
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
  M ∈ valid_digits ∧ E ∈ valid_digits ∧ H ∈ valid_digits ∧ A ∈ valid_digits ∧
  ∀ x y, x ∈ {M, E, H, A} ∧ y ∈ {M, E, H, A} → x ≠ y

-- Define the proof problem that these conditions imply the correct answer
theorem zamena_correct : conditions → M = 1 ∧ E = 2 ∧ H = 3 ∧ A = 4 ∧ 10000 * 5 + 1000 * A + 100 * M + 10 * E + H = 541234 :=
by sorry

end zamena_correct_l100_100953


namespace vector_magnitude_difference_l100_100661

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ∥a∥ = 3) (hb : ∥b∥ = 2) (angle_ab : real.angle a b = real.pi / 3)

theorem vector_magnitude_difference :
  ∥a - b∥ = real.sqrt 7 :=
sorry

end vector_magnitude_difference_l100_100661


namespace remainder_when_divided_by_24_l100_100499

theorem remainder_when_divided_by_24 (m k : ℤ) (h : m = 288 * k + 47) : m % 24 = 23 :=
by
  sorry

end remainder_when_divided_by_24_l100_100499


namespace zero_vector_no_direction_incorrect_l100_100882

theorem zero_vector_no_direction_incorrect 
  (length_zero_vec : ∀ (v : ℝ^n), v = 0 → |v| = 0)
  (parallel_zero_vec : ∀ (v1 v2 : ℝ^n), v1 = 0 → (v1 ∥ v2))
  (arbitrary_direction_zero_vec : ∀ (v : ℝ^n), v = 0 → direction v = arbitrary) :
  ¬ (∀ (v : ℝ^n), v = 0 → has_no_direction v) :=
sorry

end zero_vector_no_direction_incorrect_l100_100882


namespace tan_45_eq_1_l100_100007

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100007


namespace tan_45_eq_1_l100_100004

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100004


namespace experts_win_eventually_l100_100225

noncomputable def p : ℝ := 0.6
noncomputable def q : ℝ := 1 - p

def experts_need : ℕ := 3
def audience_need : ℕ := 2
def max_rounds : ℕ := 4

def winning_probability (experts_need : ℕ) (audience_need : ℕ) (p q : ℝ) : ℝ :=
  if experts_need = 3 ∧ audience_need = 2 ∧ p = 0.6 ∧ q = 0.4 then
    p^4 + 4 * p^3 * q
  else
    0

theorem experts_win_eventually :
  winning_probability experts_need audience_need p q = 0.4752 := sorry

end experts_win_eventually_l100_100225


namespace beef_weight_after_processing_l100_100927

noncomputable def weight_after_processing (initial_weight: ℝ) (lost_percentage: ℝ): ℝ := 
  initial_weight * (1 - lost_percentage / 100)

theorem beef_weight_after_processing : weight_after_processing 1500 50 = 750 := by
sory

end beef_weight_after_processing_l100_100927


namespace leak_empty_time_l100_100540

theorem leak_empty_time
  (pump_fill_time : ℝ)
  (leak_fill_time : ℝ)
  (pump_fill_rate : pump_fill_time = 5)
  (leak_fill_rate : leak_fill_time = 10)
  : (1 / 5 - 1 / leak_fill_time)⁻¹ = 10 :=
by
  -- you can fill in the proof here
  sorry

end leak_empty_time_l100_100540


namespace experts_win_eventually_l100_100226

noncomputable def p : ℝ := 0.6
noncomputable def q : ℝ := 1 - p

def experts_need : ℕ := 3
def audience_need : ℕ := 2
def max_rounds : ℕ := 4

def winning_probability (experts_need : ℕ) (audience_need : ℕ) (p q : ℝ) : ℝ :=
  if experts_need = 3 ∧ audience_need = 2 ∧ p = 0.6 ∧ q = 0.4 then
    p^4 + 4 * p^3 * q
  else
    0

theorem experts_win_eventually :
  winning_probability experts_need audience_need p q = 0.4752 := sorry

end experts_win_eventually_l100_100226


namespace Anish_has_2000_tiles_l100_100541

def number_of_tiles (n : ℕ) : ℕ :=
  n^2 + 64

def tiles_short (n : ℕ) : ℕ :=
  (n + 1)^2 - 25

theorem Anish_has_2000_tiles :
  ∃ n : ℕ, number_of_tiles n = 2000 := 
by
  have h : ∀ n, number_of_tiles n = tiles_short n :=
    λ n, by sorry
  use 44
  rw h
  sorry

end Anish_has_2000_tiles_l100_100541


namespace greatest_whole_number_satisfying_inequality_l100_100609

-- Define the problem condition
def inequality (x : ℤ) : Prop := 5 * x - 4 < 3 - 2 * x

-- Prove that under this condition, the greatest whole number satisfying it is 0
theorem greatest_whole_number_satisfying_inequality : ∃ (n : ℤ), inequality n ∧ ∀ (m : ℤ), inequality m → m ≤ n :=
begin
  use 0,
  split,
  { -- Proof that 0 satisfies the inequality
    unfold inequality,
    linarith, },
  { -- Proof that 0 is the greatest whole number satisfying the inequality
    intro m,
    unfold inequality,
    intro hm,
    linarith, }
end

#check greatest_whole_number_satisfying_inequality

end greatest_whole_number_satisfying_inequality_l100_100609


namespace fruit_problem_l100_100514

theorem fruit_problem :
  let apples_initial := 7
  let oranges_initial := 8
  let mangoes_initial := 15
  let grapes_initial := 12
  let strawberries_initial := 5
  let apples_taken := 3
  let oranges_taken := 4
  let mangoes_taken := 4
  let grapes_taken := 7
  let strawberries_taken := 3
  let apples_remaining := apples_initial - apples_taken
  let oranges_remaining := oranges_initial - oranges_taken
  let mangoes_remaining := mangoes_initial - mangoes_taken
  let grapes_remaining := grapes_initial - grapes_taken
  let strawberries_remaining := strawberries_initial - strawberries_taken
  let total_remaining := apples_remaining + oranges_remaining + mangoes_remaining + grapes_remaining + strawberries_remaining
  let total_taken := apples_taken + oranges_taken + mangoes_taken + grapes_taken + strawberries_taken
  total_remaining = 26 ∧ total_taken = 21 := by
    sorry

end fruit_problem_l100_100514


namespace distinct_powers_l100_100505

-- Definitions for the conditions
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def is_cube (n : ℕ) : Prop := ∃ m : ℕ, m * m * m = n
def is_fifth_power (n : ℕ) : Prop := ∃ m : ℕ, m ^ 5 = n
def is_seventh_power (n : ℕ) : Prop := ∃ m : ℕ, m ^ 7 = n

-- Given statement
theorem distinct_powers (N a1 a2 b1 b2 c1 c2 d1 d2 : ℕ) :
  N = a1 - a2 ∧ N = b1 - b2 ∧ N = c1 - c2 ∧ N = d1 - d2 ∧
  is_square a1 ∧ is_square a2 ∧
  is_cube b1 ∧ is_cube b2 ∧
  is_fifth_power c1 ∧ is_fifth_power c2 ∧
  is_seventh_power d1 ∧ is_seventh_power d2 →
  a1 ≠ b1 ∧ a1 ≠ c1 ∧ a1 ≠ d1 ∧ b1 ≠ c1 ∧ b1 ≠ d1 ∧ c1 ≠ d1 :=
begin
  sorry
end

end distinct_powers_l100_100505


namespace length_of_AB_l100_100351

-- Define the given parametric equations for the curve C1
def curve_C1_parametric (φ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos φ, Real.sin φ)

-- Define the polar equation of curve C1
def curve_C1_polar (θ : ℝ) : ℝ :=
  Real.sqrt(4 / (Real.cos θ^2 + 4 * Real.sin θ^2))

-- Define the ordinary equation for curve C2
def curve_C2_ordinary (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * Real.sqrt 7 * y = 0

-- Define the polar equation for curve C2
def curve_C2_polar (θ : ℝ) : ℝ :=
  2 * Real.sqrt 7 * Real.sin θ

-- Define the function to calculate the rho value at given theta
def rho_value_C1 (θ : ℝ) : ℝ :=
  Real.sqrt (4 / (Real.cos θ^2 + 4 * Real.sin θ^2))

-- Define the function to calculate rho for C2 at given theta
def rho_value_C2 (θ : ℝ) : ℝ :=
  2 * Real.sqrt 7 * Real.sin θ

variable θ : ℝ
variable (rho_A: ℝ := rho_value_C1 (θ := Real.pi / 6)) 
variable (rho_B: ℝ := rho_value_C2 (θ := Real.pi / 6)) 

-- Define and prove the main statement in Lean 4
theorem length_of_AB : rho_B - rho_A = (3 * Real.sqrt 7) / 7 := by
  sorry

end length_of_AB_l100_100351


namespace tan_45_deg_eq_one_l100_100160

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100160


namespace square_heptagon_intersection_l100_100849

open EuclideanGeometry

def square_circumscribed_circle_condition
  (k : circle)
  (A B C D O : point)
  (e : line)
  (P Q R : point) : Prop :=
  square A B C D ∧ 
  circumscribed_circle k A B C D ∧ 
  center O k ∧ 
  tangent e k C ∧
  on extension_line A C P ∧
  intersects_line P D e Q ∧
  perpendicular_line P B R

-- Main Theorem
theorem square_heptagon_intersection
  (k : circle)
  (A B C D O : point)
  (e : line)
  (P Q R : point)
  (h : square_circumscribed_circle_condition k A B C D O e P Q R)
  (h1 : dist Q R = dist O A) :
  intersects_perpendicular_bisector_at_heptagon_vertices k A O P :=
begin
  sorry
end

end square_heptagon_intersection_l100_100849


namespace zamena_correct_l100_100949

variables {M E H A : ℕ}

def valid_digits := {1, 2, 3, 4, 5}

-- Define the conditions as hypotheses
def conditions : Prop :=
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
  M ∈ valid_digits ∧ E ∈ valid_digits ∧ H ∈ valid_digits ∧ A ∈ valid_digits ∧
  ∀ x y, x ∈ {M, E, H, A} ∧ y ∈ {M, E, H, A} → x ≠ y

-- Define the proof problem that these conditions imply the correct answer
theorem zamena_correct : conditions → M = 1 ∧ E = 2 ∧ H = 3 ∧ A = 4 ∧ 10000 * 5 + 1000 * A + 100 * M + 10 * E + H = 541234 :=
by sorry

end zamena_correct_l100_100949


namespace gumball_problem_l100_100766
-- Step d: Lean 4 statement conversion

/-- 
  Suppose Joanna initially had 40 gumballs, Jacques had 60 gumballs, 
  and Julia had 80 gumballs.
  Joanna purchased 5 times the number of gumballs she initially had,
  Jacques purchased 3 times the number of gumballs he initially had,
  and Julia purchased 2 times the number of gumballs she initially had.
  Prove that after adding their purchases:
  1. Each person will have 240 gumballs.
  2. If they combine all their gumballs and share them equally, 
     each person will still get 240 gumballs.
-/
theorem gumball_problem :
  let joanna_initial := 40 
  let jacques_initial := 60 
  let julia_initial := 80 
  let joanna_final := joanna_initial + 5 * joanna_initial 
  let jacques_final := jacques_initial + 3 * jacques_initial 
  let julia_final := julia_initial + 2 * julia_initial 
  let total_gumballs := joanna_final + jacques_final + julia_final 
  (joanna_final = 240) ∧ (jacques_final = 240) ∧ (julia_final = 240) ∧ 
  (total_gumballs / 3 = 240) :=
by
  let joanna_initial := 40 
  let jacques_initial := 60 
  let julia_initial := 80 
  let joanna_final := joanna_initial + 5 * joanna_initial 
  let jacques_final := jacques_initial + 3 * jacques_initial 
  let julia_final := julia_initial + 2 * julia_initial 
  let total_gumballs := joanna_final + jacques_final + julia_final 
  
  have h_joanna : joanna_final = 240 := sorry
  have h_jacques : jacques_final = 240 := sorry
  have h_julia : julia_final = 240 := sorry
  have h_total : total_gumballs / 3 = 240 := sorry
  
  exact ⟨h_joanna, h_jacques, h_julia, h_total⟩

end gumball_problem_l100_100766


namespace seventh_term_in_geometric_sequence_l100_100761

theorem seventh_term_in_geometric_sequence :
  ∃ r, (4 * r^8 = 2097152) ∧ (4 * r^6 = 1048576) :=
by
  sorry

end seventh_term_in_geometric_sequence_l100_100761


namespace tan_45_eq_1_l100_100006

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100006


namespace histogram_area_equals_total_count_l100_100851

def Histogram : Type := List (ℝ × ℕ)

def sumOfAreas (histogram : Histogram) : ℕ :=
  histogram.foldr (λ (rect : ℝ × ℕ) (acc : ℕ), acc + (rect.1 * rect.2).natAbs) 0

theorem histogram_area_equals_total_count (histogram : Histogram) :
  sumOfAreas(histogram) = histogram.foldr (λ (rect : ℝ × ℕ) (acc : ℕ), acc + rect.2) 0 := 
sorry

end histogram_area_equals_total_count_l100_100851


namespace letters_typing_order_count_l100_100342
open BigOperators

/-- In an office, letters are handled in Last In, First Out (LIFO) manner using a stack. Letters 1 to 7 must have been placed in the stack prior to 8 and 9, and 9 could only have been added and typed after 8. We want to determine the number of possible orders in which the untyped letters (including possibly the letter 10) can be processed after lunch, considering letters 8 and 9 are already typed. -/
theorem letters_typing_order_count :
  ∑ k in finset.range 8, nat.choose 7 k * (k + 1) = 576 :=
sorry

end letters_typing_order_count_l100_100342


namespace tan_of_45_deg_l100_100070

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100070


namespace product_of_digits_of_next_palindromic_square_l100_100855

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in s = s.reverse

def is_perfect_square (n : ℕ) : Prop :=
  let root := nat.sqrt n in root * root = n

theorem product_of_digits_of_next_palindromic_square (y : ℕ) (h1 : y > 2021) (h2 : is_palindrome y) (h3 : is_perfect_square y) :
  (y.digits.sum) = 4 :=
sorry

end product_of_digits_of_next_palindromic_square_l100_100855


namespace tan_5pi_over_4_l100_100600

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l100_100600


namespace david_minimum_next_score_l100_100998

variable (scores : List ℝ)
variable (next_score : ℝ)

theorem david_minimum_next_score
  (h_scores : scores = [88, 92, 75, 83, 90])
  (h_len : scores.length = 5)
  (h_next_score : next_score ≥ 110) :
  (scores.sum + next_score) / (scores.length + 1) ≥ ((scores.sum / scores.length) + 4) := by
suffices avg_85_6 : (scores.sum / scores.length) = 85.6
  from (scores.sum + next_score) / (scores.length + 1) ≥ avg_85_6 + 4
  -- Add proof details here if necessary, otherwise:
sorry

end david_minimum_next_score_l100_100998


namespace thomas_savings_l100_100461

def first_year_earnings (weekly_allowance_year1 : ℕ) : ℕ :=
  weekly_allowance_year1 * 52

def second_year_earnings (hourly_rate_year2 hours_per_week_year2 : ℕ) : ℕ :=
  hourly_rate_year2 * hours_per_week_year2 * 52

def total_earnings (first_year second_year : ℕ) : ℕ :=
  first_year + second_year

def total_expenses (personal_expenses : ℕ) : ℕ :=
  personal_expenses * (52 * 2)

def savings (earnings expenses : ℕ) : ℕ :=
  earnings - expenses

def amount_needed (car_cost savings : ℕ) : ℕ :=
  car_cost - savings

theorem thomas_savings : 
  ∀ (weekly_allowance_year1 hourly_rate_year2 hours_per_week_year2 car_cost personal_expenses : ℕ),
  years = 2 →
  weekly_allowance_year1 = 50 →
  hourly_rate_year2 = 9 →
  hours_per_week_year2 = 30 →
  car_cost = 15000 →
  personal_expenses = 35 →
  amount_needed car_cost (
    savings (
      total_earnings 
        (first_year_earnings weekly_allowance_year1)
        (second_year_earnings hourly_rate_year2 hours_per_week_year2)
      )
    (total_expenses personal_expenses)
  ) = 2000
:=
by
  intros _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end thomas_savings_l100_100461


namespace tan_45_eq_1_l100_100188

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100188


namespace experts_eventual_win_probability_l100_100234

noncomputable def experts_win_probability (p : ℝ) (q : ℝ) (experts_needed : ℕ) (audience_needed : ℕ) (max_rounds : ℕ) : ℝ :=
  let e_all_wins := p ^ max_rounds
  let e_three_wins_one_loss := (Nat.choose max_rounds (max_rounds - 1)) * (p ^ (max_rounds - 1)) * q
  e_all_wins + e_three_wins_one_loss

theorem experts_eventual_win_probability :
  ∀ (p : ℝ) (q : ℝ),
  p = 0.6 →
  q = 1 - p →
  experts_win_probability p q 3 2 4 = 0.4752 := by
  intros p q hp hq
  have h : experts_win_probability p q 3 2 4 = 0.6 ^ 4 + 4 * (0.6 ^ 3) * 0.4 := sorry
  rw [hp, hq] at h
  norm_num at h
  exact h

end experts_eventual_win_probability_l100_100234


namespace tan_five_pi_over_four_l100_100593

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l100_100593


namespace find_a5_and_sum_l100_100282

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) > a n

-- Given conditions
def given_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
is_geometric_sequence a q ∧ is_increasing_sequence a ∧ a 2 = 3 ∧ a 4 - a 3 = 18

-- Theorem to prove
theorem find_a5_and_sum {a : ℕ → ℝ} {q : ℝ} (h : given_conditions a q) :
  a 5 = 81 ∧ (a 1 + a 2 + a 3 + a 4 + a 5) = 121 :=
by
  -- Placeholder for the actual proof
  sorry

end find_a5_and_sum_l100_100282


namespace tan_of_45_deg_l100_100084

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100084


namespace greatest_whole_number_solution_l100_100603

theorem greatest_whole_number_solution : ∃ k : ℕ, 5 * k - 4 < 3 - 2 * k ∧ ∀ m : ℕ, (m > k → ¬ (5 * m - 4 < 3 - 2 * m)) :=
by
  exists 0
  split
  {
    rw [nat.cast_zero, mul_zero, sub_zero, zero_sub, neg_lt, neg_zero]
    exact zero_lt_three
  }
  {
    intros m hm
    rw [nat.not_lt]
    exact nat.le_of_lt_succ hm
  }

end greatest_whole_number_solution_l100_100603


namespace triangle_area_l100_100865

theorem triangle_area (base height : ℕ) (h_base : base = 35) (h_height : height = 12) :
  (1 / 2 : ℚ) * base * height = 210 := by
  sorry

end triangle_area_l100_100865


namespace range_of_a_l100_100679

section

variables (f : ℝ → ℝ) (A : set ℝ) (a : ℝ)
def is_positive_real_set (s : set ℝ) := ∀ x ∈ s, x > 0

theorem range_of_a (h1 : ∀ x, f x = x^2)
                   (h2 : A = {x | f (x + 1) = a * x ∧ x ∈ ℝ})
                   (h3 : ¬ is_positive_real_set (A ∪ set.Ioi 0)) :
                   a ∈ set.Ioi 0 :=
sorry

end

end range_of_a_l100_100679


namespace poplusq_roster_method_l100_100689

def P := {4, 5}
def Q := {1, 2, 3}
def PoplusQ := {x | ∃ (p ∈ P) (q ∈ Q), x = p - q}

theorem poplusq_roster_method : PoplusQ = {1, 2, 3, 4} :=
by
  sorry

end poplusq_roster_method_l100_100689


namespace possible_values_of_k_l100_100732

theorem possible_values_of_k (k : ℕ) (N : ℕ) (h₁ : (k * (k + 1)) / 2 = N^2) (h₂ : N < 100) :
  k = 1 ∨ k = 8 ∨ k = 49 :=
sorry

end possible_values_of_k_l100_100732


namespace oranges_now_is_50_l100_100538

def initial_fruits : ℕ := 150
def remaining_fruits : ℕ := initial_fruits / 2
def num_limes (L : ℕ) (O : ℕ) : Prop := O = 2 * L
def total_remaining_fruits (L : ℕ) (O : ℕ) : Prop := O + L = remaining_fruits

theorem oranges_now_is_50 : ∃ O L : ℕ, num_limes L O ∧ total_remaining_fruits L O ∧ O = 50 := by
  sorry

end oranges_now_is_50_l100_100538


namespace product_of_common_divisors_eq_182250000_l100_100628

-- Definitions from conditions
def divisors (n : ℤ) : Set ℤ := {d | d ∣ n}

def common_divisors (a b : ℤ) : Set ℤ := divisors a ∩ divisors b

-- The main theorem to prove
theorem product_of_common_divisors_eq_182250000 :
  ∏ x in common_divisors 210 30, x = 182250000 := 
sorry

end product_of_common_divisors_eq_182250000_l100_100628


namespace part1_part2_l100_100274

-- Definitions for the given lines
def l1 : ℝ × ℝ → Prop := λ p, p.1 - 2 * p.2 + 3 = 0
def l2 : ℝ × ℝ → Prop := λ p, 2 * p.1 + 3 * p.2 - 8 = 0
def l3 : ℝ × ℝ → Prop := λ p, p.1 + 3 * p.2 + 1 = 0

-- Definition for point M as the intersection of l1 and l2
def M : ℝ × ℝ := (1, 2)

-- Definitions for the lines to be proved (based on correct answers)
def line_parallel_to_l3 : ℝ × ℝ → Prop := λ p, p.1 + 3 * p.2 - 7 = 0
def line_with_distance : ℝ × ℝ → Prop := λ p, (p.1 = 1) ∨ (3 * p.1 + 4 * p.2 - 11 = 0)

-- The theorem to be proved (in two parts)
theorem part1 : ∀ p, (p = M → line_parallel_to_l3 p) :=
by sorry

theorem part2 : ∀ p, (p = M → line_with_distance p) :=
by sorry

end part1_part2_l100_100274


namespace solve_equation_l100_100431

theorem solve_equation :
  ∀ x : ℝ, (x * (2 * x + 4) = 10 + 5 * x) ↔ (x = -2 ∨ x = 2.5) :=
by
  sorry

end solve_equation_l100_100431


namespace solve_3x_plus_5_squared_l100_100711

theorem solve_3x_plus_5_squared (x : ℝ) (h : 5 * x - 6 = 15 * x + 21) : 
  3 * (x + 5) ^ 2 = 2523 / 100 :=
by
  sorry

end solve_3x_plus_5_squared_l100_100711


namespace tan_45_deg_l100_100135

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100135


namespace XYZ_collinear_l100_100895

-- Definitions based on the conditions in part a)
variables {A B C H D E X Y Z : Point}

def is_orthocenter (H : Point) (A B C : Triangle) : Prop := sorry
def perp (l1 l2 : Line) : Prop := sorry
def passes_through (H : Point) (l : Line) : Prop := sorry
def intersection (l1 l2 : Line) : Point := sorry
def line_parallel (p1 p2 p3 p4 : Point) : Prop := sorry
def collinear (X Y Z : Point) : Prop := sorry

-- Assuming necessary geometry definitions and properties are provided as sorries

-- Main statement
theorem XYZ_collinear 
  (orthocenter H: is_orthocenter H ⟨A, B, C⟩)
  (l1 l2 : Line)
  (H_l1 : passes_through H l1)
  (H_l2 : passes_through H l2)
  (l1_perp_l2 : perp l1 l2)
  (D := intersection l1 ⟨B, C⟩)
  (Z := intersection l1 ⟨A, B⟩)
  (E := intersection l2 ⟨B, C⟩)
  (X := intersection l2 ⟨A, C⟩)
  (d1_parallel_AC : line_parallel D X A C)
  (d2_parallel_AB : line_parallel E Y A B)
  (Y := intersection ⟨D, X⟩ ⟨E, Y⟩) :
  collinear X Y Z :=
sorry

end XYZ_collinear_l100_100895


namespace tan_45_deg_eq_one_l100_100112

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100112


namespace no_solution_to_system_l100_100243

theorem no_solution_to_system : ∀ (x y : ℝ), ¬ (y^2 - (⌊x⌋ : ℝ)^2 = 2001 ∧ x^2 + (⌊y⌋ : ℝ)^2 = 2001) :=
by sorry

end no_solution_to_system_l100_100243


namespace grains_of_rice_difference_l100_100536

theorem grains_of_rice_difference : 
  let grains_on_square (k : ℕ) := 2 ^ k in
  let sum_of_first_n_squares (n : ℕ) := (List.range n).map (λ k, grains_on_square (k + 1)).sum in
  grains_on_square 15 - sum_of_first_n_squares 12 = 24578 :=
by 
  sorry

end grains_of_rice_difference_l100_100536


namespace positive_integer_fraction_count_l100_100256

  theorem positive_integer_fraction_count :
    {n : ℕ // 0 < n ∧ n < 42 ∧ ∃ k : ℕ, n = k * (42 - n)}.card = 7 :=
  sorry
  
end positive_integer_fraction_count_l100_100256


namespace cells_at_end_of_8th_day_l100_100909

theorem cells_at_end_of_8th_day :
  let initial_cells := 5
  let factor := 3
  let toxin_factor := 1 / 2
  let cells_after_toxin := (initial_cells * factor * factor * factor * toxin_factor : ℤ)
  let final_cells := cells_after_toxin * factor 
  final_cells = 201 :=
by
  sorry

end cells_at_end_of_8th_day_l100_100909


namespace pyramid_volume_correct_l100_100456

variables {V : ℝ} {α : ℝ}

def pyramid_volume (V: ℝ) (α: ℝ) : ℝ :=
  (2 * V * sin(α) * cos(α / 2)^2) / π

theorem pyramid_volume_correct {V : ℝ} {α : ℝ} (V_pos : 0 < V) (alpha_pos : 0 < α ∧ α < π) :
  let V_p := pyramid_volume V α in
  V_p = (2 * V * sin(α) * cos(α / 2)^2) / π :=
by
  sorry

end pyramid_volume_correct_l100_100456


namespace find_b_eq_neg_three_l100_100294

theorem find_b_eq_neg_three (b : ℝ) (h : (2 - b) / 5 = -(2 * b + 1) / 5) : b = -3 :=
by
  sorry

end find_b_eq_neg_three_l100_100294


namespace Monica_saved_per_week_l100_100806

theorem Monica_saved_per_week(amount_per_cycle : ℕ) (weeks_per_cycle : ℕ) (num_cycles : ℕ) (total_saved : ℕ) :
  num_cycles = 5 →
  weeks_per_cycle = 60 →
  (amount_per_cycle * num_cycles) = total_saved →
  total_saved = 4500 →
  total_saved / (weeks_per_cycle * num_cycles) = 75 := 
by
  intros
  sorry

end Monica_saved_per_week_l100_100806


namespace min_roots_sin_eq_zero_l100_100507

open Real

theorem min_roots_sin_eq_zero (k0 k1 k2 : ℕ) (h : k0 < k1 ∧ k1 < k2) (A1 A2 : ℝ) :
  ∃ x ∈ Ico 0 (2 * π), 
  (sin (k0 * x) + A1 * sin (k1 * x) + A2 * sin (k2 * x) = 0) :=
sorry

end min_roots_sin_eq_zero_l100_100507


namespace johns_money_left_l100_100366

def dog_walking_days_in_april := 26
def earnings_per_day := 10
def money_spent_on_books := 50
def money_given_to_sister := 50

theorem johns_money_left : (dog_walking_days_in_april * earnings_per_day) - (money_spent_on_books + money_given_to_sister) = 160 := 
by
  sorry

end johns_money_left_l100_100366


namespace punger_needs_pages_l100_100408

theorem punger_needs_pages (p c h : ℕ) (h_p : p = 60) (h_c : c = 7) (h_h : h = 10) : 
  (p * c) / h = 42 := 
by
  sorry

end punger_needs_pages_l100_100408


namespace tan_five_pi_over_four_l100_100590

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l100_100590


namespace num_solutions_of_complex_numbers_l100_100625

theorem num_solutions_of_complex_numbers (z : ℂ) (h1 : |z| = 1) 
(h2 : |z / conj z - conj z / z| = 1) : 
  ∃ sol_set, sol_set.card = 4 ∧ (∀ z ∈ sol_set, |z| = 1 ∧ |z / conj z - conj z / z| = 1) := 
  sorry

end num_solutions_of_complex_numbers_l100_100625


namespace product_of_integers_between_sqrt_115_l100_100854

theorem product_of_integers_between_sqrt_115 :
  ∃ a b : ℕ, 100 < 115 ∧ 115 < 121 ∧ a = 10 ∧ b = 11 ∧ a * b = 110 := by
  sorry

end product_of_integers_between_sqrt_115_l100_100854


namespace distance_focus_hyperbola_asymptote_l100_100278

noncomputable def distance_focus_to_asymptote : ℝ :=
let C := λ x y : ℝ, x^2 / 3 - y^2 / 3 = 1 in
let F := (sqrt 6, 0) in
let asymptote := λ x y : ℝ, y = x in
(real.sqrt 6) / real.sqrt 2

theorem distance_focus_hyperbola_asymptote :
  distance_focus_to_asymptote = real.sqrt 3 :=
sorry

end distance_focus_hyperbola_asymptote_l100_100278


namespace min_sum_main_diagonal_l100_100341

theorem min_sum_main_diagonal (n : ℕ) (a : ℕ → ℕ → ℕ)
  (H1 : ∀ i j k, i < j → a i k < a j k)
  (H2 : ∀ i j k, i < j → a k i < a k j)
  (H3 : ∀ i j, 1 ≤ a i j ∧ a i j ≤ n^2 ∧ ∀ xi yi xj yj, (xi ≠ xj ∨ yi ≠ yj) → a xi yi ≠ a xj yj) :
  (∑ i in Finset.range n, (a i i)) = (∑ i in Finset.range n, (i + 1)^2) :=
by
  sorry

end min_sum_main_diagonal_l100_100341


namespace smallest_even_sum_l100_100562

theorem smallest_even_sum :
  ∃ (a b c : Int), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ b ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ c ∈ ({8, -4, 3, 27, 10} : Set Int) ∧ (a + b + c) % 2 = 0 ∧ (a + b + c) = 14 := sorry

end smallest_even_sum_l100_100562


namespace f_monotonically_increasing_on_1_to_infinity_l100_100298

noncomputable def f (x : ℝ) : ℝ := x + 1/x

theorem f_monotonically_increasing_on_1_to_infinity :
  ∀ x y : ℝ, 1 < x → x < y → f x < f y := 
sorry

end f_monotonically_increasing_on_1_to_infinity_l100_100298


namespace rebecca_income_l100_100411

variable (R : ℝ) -- Rebecca's current yearly income (denoted as R)
variable (increase : ℝ := 7000) -- The increase in Rebecca's income
variable (jimmy_income : ℝ := 18000) -- Jimmy's yearly income
variable (combined_income : ℝ := (R + increase) + jimmy_income) -- Combined income after increase
variable (new_income_ratio : ℝ := 0.55) -- Proportion of total income that is Rebecca's new income

theorem rebecca_income : (R + increase) = new_income_ratio * combined_income → R = 15000 :=
by
  sorry

end rebecca_income_l100_100411


namespace students_more_than_pets_l100_100214

-- Definition of given conditions
def num_students_per_classroom := 20
def num_rabbits_per_classroom := 2
def num_goldfish_per_classroom := 3
def num_classrooms := 5

-- Theorem stating the proof problem
theorem students_more_than_pets :
  let total_students := num_students_per_classroom * num_classrooms
  let total_pets := (num_rabbits_per_classroom + num_goldfish_per_classroom) * num_classrooms
  total_students - total_pets = 75 := by
  sorry

end students_more_than_pets_l100_100214


namespace new_year_markup_percentage_l100_100922

-- Given conditions as definitions in Lean
variables (C : ℝ) (M : ℝ)
def initial_price := 1.20 * C
def price_new_year := initial_price * (1 + M / 100)
def final_price := price_new_year * (1 - 0.07)
def profit := 1.395 * C

-- The goal is to prove that M = 25 given the conditions.
theorem new_year_markup_percentage:
  final_price = profit →
  M = 25 :=
by
  sorry

end new_year_markup_percentage_l100_100922


namespace find_incenter_eq_common_chords_l100_100242

variable {A B C P : Type} 
variable [metric_space P] [metric_space A] [metric_space B] [metric_space C]

def is_incenter (P A B C : P) : Prop :=
  ∀ p : P, ∃ d : ℝ, dist p P = d ∧ dist p A = d ∧ dist p B = d ∧ dist p C = d

theorem find_incenter_eq_common_chords (P A B C : P) (h_in_triangle : P ∈ triangle ABC) 
  (h_eq_chords : ∀ circles PA PB PC, PA = PB ∧ PB = PC ∧ PC = PA) : is_incenter P A B C := 
by
  sorry

end find_incenter_eq_common_chords_l100_100242


namespace integral_transformation_example_l100_100987

theorem integral_transformation_example :
  ∫ x in 2..7, (x - 3)^2 = 65 / 3 := by
  sorry

end integral_transformation_example_l100_100987


namespace zamena_solution_l100_100968

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end zamena_solution_l100_100968


namespace positive_difference_of_solutions_eq_12_l100_100872

noncomputable def positive_difference_of_solutions : ℝ :=
  let equation : ℝ → Prop := λ r, (r^2 - 3*r - 17) / (r + 4) = 2*r + 7
  let solutions := {r : ℝ | (r + 3) * (r + 15) = 0}
  let sol1 := -3
  let sol2 := -15
  abs(sol1 - sol2)

theorem positive_difference_of_solutions_eq_12 :
  positive_difference_of_solutions = 12 :=
sorry

end positive_difference_of_solutions_eq_12_l100_100872


namespace find_a_l100_100350

noncomputable def C1Parametric (a : ℝ) (t : ℝ) : ℝ × ℝ :=
  (a + (Real.sqrt 2 / 2) * t, 1 + (Real.sqrt 2 / 2) * t)

def CartesianC1 (a : ℝ) : Prop :=
  ∀ x y : ℝ, y = x - a + 1 

def CartesianC2 : Prop :=
  ∀ x y : ℝ, y ^ 2 = 4 * x

def intersection_condition (a t1 t2 : ℝ) : Prop :=
  let x1 := a + (Real.sqrt 2 / 2) * t1 in
  let y1 := 1 + (Real.sqrt 2 / 2) * t1 in
  let x2 := a + (Real.sqrt 2 / 2) * t2 in
  let y2 := 1 + (Real.sqrt 2 / 2) * t2 in
  x1 = x2 ∧ y1 = y2 ∧ y1 ^ 2 = 4 * x1 ∧ y2 ^ 2 = 4 * x2 ∧ t1 = 2 * t2

theorem find_a (a : ℝ) (t1 t2 : ℝ) (h_intersect : intersection_condition a t1 t2) : a = 1 / 36 :=
  sorry

end find_a_l100_100350


namespace dorothy_price_per_doughnut_l100_100574

noncomputable def price_per_doughnut (cost profit number_of_doughnuts : ℝ) : ℝ :=
  (cost + profit) / number_of_doughnuts

theorem dorothy_price_per_doughnut :
  ∀ (cost profit number_of_doughnuts : ℝ),
  cost = 53 → profit = 22 → number_of_doughnuts = 25 → 
  price_per_doughnut cost profit number_of_doughnuts = 3 := 
by
  intros cost profit number_of_doughnuts h_cost h_profit h_doughnuts
  rw [h_cost, h_profit, h_doughnuts]
  unfold price_per_doughnut
  norm_num
  sorry

end dorothy_price_per_doughnut_l100_100574


namespace cone_lateral_area_l100_100521

/-- A cone with a base radius of 6 cm and a height of 8 cm has a lateral area of 60π cm². -/
theorem cone_lateral_area :
  let r := 6    -- base radius in cm
  let h := 8    -- height in cm
  ∃ A : ℝ, A = 60 * Real.pi ∧
    A = let l := Real.sqrt (r^2 + h^2) in   -- slant height using Pythagorean theorem
        (1 / 2) * (2 * Real.pi * r) * l     -- lateral area formula
:= sorry

end cone_lateral_area_l100_100521


namespace tan_45_eq_1_l100_100182

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100182


namespace part1_solution_set_part2_min_value_l100_100302

def f (x : ℝ) : ℝ :=
  abs (2 * x - 1)

def g (x : ℝ) : ℝ :=
  f x + f (x - 1)

theorem part1_solution_set :
  {x : ℝ | f x + abs (x + 1) < 2} = {x : ℝ | 0 < x ∧ x < 2 / 3} :=
by
  sorry

theorem part2_min_value (m n : ℝ) (hm : 0 < m) (hn : 0 < n) 
  (h : m + n = 2) :
  ∀ x : ℝ, g x = 2 → ∀ (m n : ℝ) (hm : 0 < m) (hn : 0 < n)
  (h : m + n = 2), ∃ (m : ℝ) (n : ℝ),
  (g x = a) -> (a = 2) -> (m + n = 2) -> 
  (∀ x, g x = 2) in range) (minimum_value : ℝ) 
  (h_min : minimum_value = (4 / m) + 1 / n) : minimum_value = 9 / 2 := 
by
  sorry

end part1_solution_set_part2_min_value_l100_100302


namespace tan_45_deg_eq_one_l100_100032

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100032


namespace area_triangle_ABC_eq_8k_l100_100755

variables {A B C M N P O : Type} [Field M] [Field N]
variables [AddCommGroup N] [VectorSpace M N]
variables (M N) (A B C P O : M)
variables (k : N)
variables [Triangle ABC] [Midpoints AB AC M N] [Parallel NP BC] [Extended AM P] [Centroid ABC O]

def area_triangle_MPN : N := k

theorem area_triangle_ABC_eq_8k :
  area (triangle ABC) = 8 * area_triangle_MPN k :=
sorry

end area_triangle_ABC_eq_8k_l100_100755


namespace max_f_value_up_to_2010_max_f_value_attainable_l100_100792

def floor (x : ℝ) := int.floor x

def f : ℕ → ℕ
| 0 := 0
| n := f (floor (n / 2)) + n - 2 * floor (n / 2)

theorem max_f_value_up_to_2010 : ∀ m, m ≥ 0 → m ≤ 2010 → f m ≤ 10 := sorry

theorem max_f_value_attainable : ∃ m, m ≥ 0 ∧ m ≤ 2010 ∧ f m = 10 := sorry

end max_f_value_up_to_2010_max_f_value_attainable_l100_100792


namespace solve_ZAMENA_l100_100974

noncomputable def ZAMENA : ℕ := 541234

theorem solve_ZAMENA :
  ∃ (A M E H : ℕ), 
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H ∧
    ZAMENA = 541234 :=
by {
  have h1 : ∃ (A M E H : ℕ), 
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H, sorry,
  exact ⟨2, 1, 2, 3, h1⟩,
  repeat { sorry }
}

end solve_ZAMENA_l100_100974


namespace distinct_integer_values_of_f_l100_100633

def f (x : ℝ) : ℝ := 
  Real.floor x + Real.floor (2 * x) + Real.floor ((5 / 3) * x) + Real.floor (3 * x) + Real.floor (4 * x)

theorem distinct_integer_values_of_f :
  {n : ℤ | ∃ x : ℝ, 0 ≤ x ∧ x ≤ 100 ∧ f x = n}.to_finset.card = 734 :=
sorry

end distinct_integer_values_of_f_l100_100633


namespace tan_45_deg_eq_one_l100_100113

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100113


namespace arithmetic_sequence_sum_l100_100293

variable (a : ℕ → ℝ)
variable (d : ℝ)

axiom a2_a5 : a 2 + a 5 = 4
axiom a6_a9 : a 6 + a 9 = 20

theorem arithmetic_sequence_sum : a 4 + a 7 = 12 := by
  sorry

end arithmetic_sequence_sum_l100_100293


namespace complex_square_simplification_l100_100419

theorem complex_square_simplification : (4 - 3 * complex.I) ^ 2 = 7 - 24 * complex.I :=
by
  sorry

end complex_square_simplification_l100_100419


namespace tan_45_deg_eq_one_l100_100170

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100170


namespace zamena_solution_l100_100970

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end zamena_solution_l100_100970


namespace tan_of_45_deg_l100_100096

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100096


namespace find_length_of_AB_l100_100682

noncomputable def segment_length_between_intersections
  (x y : ℝ) :
  let θ := Arc.cos ((x - 1) / √3),
      ρ_C := √ ((x - 1)^2 + (y / √3)^2),
      θ_l := Arc.cos (3 * √3 / ρ_C),
      ρ_l := 3 * √3 / cos (θ_l - (π / 6)) in
  ρ_l - ρ_C = 4

axiom curve_C_eqn (θ : ℝ) : 
  (let x := 1 + √3 * cos θ,
       y := √3 * sin θ in
  (x - 1)^2 + y^2 = 3)

axiom line_l_eqn (θ : ℝ) :
  (let ρ := 3 * √3 / cos (θ - (π / 6)) in
  ρ * cos (θ - (π / 6)) = 3 * √3)

axiom intersection_OT_with_C (θ : ℝ) :
  θ = π / 3 -> (let ρ := √3 in ρ = 2)

axiom intersection_OT_with_l (θ : ℝ) :
  θ = π / 3 -> (let ρ := 6 in ρ = 6)

theorem find_length_of_AB :
  segment_length_between_intersections 1 √3 = 4 :=
sorry

end find_length_of_AB_l100_100682


namespace ZAMENA_correct_l100_100948

-- Define the digits and assigning them to the letters
noncomputable def A : ℕ := 2
noncomputable def M : ℕ := 1
noncomputable def E : ℕ := 2
noncomputable def H : ℕ := 3

-- Define the number ZAMENA
noncomputable def ZAMENA : ℕ :=
  5 * 10^5 + A * 10^4 + M * 10^3 + E * 10^2 + H * 10 + A

-- Define the inequalities and constraints
lemma satisfies_inequalities  : 3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Define uniqueness of each digit set
lemma unique_digits: A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ M ≠ E ∧ M ≠ H ∧ E ≠ H := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Prove that the number ZAMENA matches the specified number
theorem ZAMENA_correct : ZAMENA = 541234 := by
  unfold ZAMENA A M E H
  norm_num

end ZAMENA_correct_l100_100948


namespace ratio_school_to_soccer_to_home_l100_100558

-- Definitions based on the conditions
def distance_to_grocery_store : ℝ := 8
def distance_to_pick_up_kids : ℝ := 6
def distance_to_soccer_practice : ℝ := 12
def car_mileage : ℝ := 25 -- miles per gallon
def gas_cost_per_gallon : ℝ := 2.50
def total_gas_expenditure : ℝ := 5

-- The proof statement
theorem ratio_school_to_soccer_to_home :
  let total_distance := (total_gas_expenditure / gas_cost_per_gallon) * car_mileage in
  let distance_for_errands := distance_to_grocery_store + distance_to_pick_up_kids + distance_to_soccer_practice in
  let distance_home : ℝ := total_distance - distance_for_errands in
  let ratio := distance_to_soccer_practice / distance_home in
  ratio = 1 / 2 :=
by
  sorry

end ratio_school_to_soccer_to_home_l100_100558


namespace tan_45_eq_1_l100_100191

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100191


namespace zamena_solution_l100_100967

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end zamena_solution_l100_100967


namespace new_average_age_l100_100825

theorem new_average_age:
  ∀ (initial_avg_age new_persons_avg_age : ℝ) (initial_count new_persons_count : ℕ),
    initial_avg_age = 16 →
    new_persons_avg_age = 15 →
    initial_count = 20 →
    new_persons_count = 20 →
    (initial_avg_age * initial_count + new_persons_avg_age * new_persons_count) / 
    (initial_count + new_persons_count) = 15.5 :=
by
  intros initial_avg_age new_persons_avg_age initial_count new_persons_count
  intros h1 h2 h3 h4
  
  sorry

end new_average_age_l100_100825


namespace tan_45_eq_1_l100_100001

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100001


namespace max_profit_l100_100397

def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 30 then 5 * x + 180
  else if 30 < x ∧ x ≤ 110 then 602 + (20000 / (x * (x + 10))) - (10000 / x)
  else 0

def g (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 30 then -5 * x^2 + 420 * x - 3
  else if 30 < x ∧ x ≤ 110 then -2 * x - 20000 / (x + 10) + 597
  else 0

theorem max_profit :
  ∃ x : ℝ, 0 < x ∧ x ≤ 110 ∧ g x = 9320 ∧ 90 ≤ x ∧ x ≤ 90 :=
begin
  sorry
end

end max_profit_l100_100397


namespace lines_parallel_l100_100508

variables {Γ1 Γ2 : Type*} [circle Γ1] [circle Γ2]
variables {A B P Q P' Q' : point}
variables (d : line) (d' : line)

-- Conditions
axiom circles_intersect : A ∈ Γ1 ∧ A ∈ Γ2 ∧ B ∈ Γ1 ∧ B ∈ Γ2
axiom line_d_contains : ∀ (x : point), x ∈ d → x = A ∨ x = P ∨ x = Q
axiom line_d'_contains : ∀ (x : point), x ∈ d' → x = B ∨ x = P' ∨ x = Q'

-- The proof we want to construct
theorem lines_parallel : is_parallel (line_through P P') (line_through Q Q') :=
sorry

end lines_parallel_l100_100508


namespace tan_45_deg_eq_one_l100_100114

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100114


namespace tan_45_deg_l100_100052

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100052


namespace find_c_plus_1_over_b_l100_100823

theorem find_c_plus_1_over_b (a b c : ℝ) (h1: a * b * c = 1) 
    (h2: a + 1 / c = 7) (h3: b + 1 / a = 12) : c + 1 / b = 21 / 83 := 
by 
    sorry

end find_c_plus_1_over_b_l100_100823


namespace zamena_correct_l100_100952

variables {M E H A : ℕ}

def valid_digits := {1, 2, 3, 4, 5}

-- Define the conditions as hypotheses
def conditions : Prop :=
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
  M ∈ valid_digits ∧ E ∈ valid_digits ∧ H ∈ valid_digits ∧ A ∈ valid_digits ∧
  ∀ x y, x ∈ {M, E, H, A} ∧ y ∈ {M, E, H, A} → x ≠ y

-- Define the proof problem that these conditions imply the correct answer
theorem zamena_correct : conditions → M = 1 ∧ E = 2 ∧ H = 3 ∧ A = 4 ∧ 10000 * 5 + 1000 * A + 100 * M + 10 * E + H = 541234 :=
by sorry

end zamena_correct_l100_100952


namespace tiling_polygons_l100_100576

theorem tiling_polygons : ∃ n, n = 3 ∧
  (∃ angles : List ℕ, 
     angles = [60, 90, 120] ∧
     ∀ θ ∈ angles, 360 % θ = 0) :=
by
  -- Define the internal angles of the given regular polygons:
  let angles := [60, 90, 108, 120, 135]
  -- Filter the angles that can tile the plane:
  let tiling_angles := angles.filter (λ θ, 360 % θ = 0)
  -- Assert that there are exactly 3 angles that can tile the plane:
  have h : tiling_angles = [60, 90, 120], by sorry
  -- Provide the proof for the existence statement:
  exact ⟨3, by simp [h], exists.intro tiling_angles (by simp [h])⟩

end tiling_polygons_l100_100576


namespace tan_45_deg_l100_100149

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100149


namespace tau_mul_of_coprime_l100_100816

open Nat  

theorem tau_mul_of_coprime (a b : ℕ) (h_coprime : gcd a b = 1) : tau (a * b) = tau a * tau b := 
  sorry

end tau_mul_of_coprime_l100_100816


namespace right_focus_ellipse_l100_100831

def ellipse_parametric_equations (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

theorem right_focus_ellipse :
  (∃ θ : ℝ, ellipse_parametric_equations θ = (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)) →
  (1, 0) ∈ {(c, 0) | ∃ a b : ℝ, a = 2 ∧ b = Real.sqrt 3 ∧ c = Real.sqrt (a^2 - b^2)} :=
by
  sorry

end right_focus_ellipse_l100_100831


namespace first_millipede_segments_l100_100906

theorem first_millipede_segments :
  ∃ x : ℕ, (x + 4 * x + 500 = 800) ∧ (x = 60) :=
by
  use 60
  split
  sorry -- First part of the proof: prove the condition x + 4 * x + 500 = 800
  refl  -- Second part of the proof: x = 60 is trivially true as x has been instantiated as 60

end first_millipede_segments_l100_100906


namespace area_difference_of_tablets_l100_100504

theorem area_difference_of_tablets 
  (d1 d2 : ℝ) (s1 s2 : ℝ)
  (h1 : d1 = 6) (h2 : d2 = 5) 
  (hs1 : d1^2 = 2 * s1^2) (hs2 : d2^2 = 2 * s2^2) 
  (A1 : ℝ) (A2 : ℝ) (hA1 : A1 = s1^2) (hA2 : A2 = s2^2)
  : A1 - A2 = 5.5 := 
sorry

end area_difference_of_tablets_l100_100504


namespace tan_45_deg_eq_one_l100_100161

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100161


namespace reverse_difference_divisible_by_9_l100_100817

theorem reverse_difference_divisible_by_9 {n : ℕ} (d : ℕ → ℕ) (k : ℕ)
  (hk : ∀ i, i < k → d i < 10) :
  let m := ∑ i in Finset.range k, d (k - 1 - i) * 10^i in
  (n = ∑ i in Finset.range k, d i * 10^i) → 
  (n - m) % 9 = 0 := 
by 
  sorry

end reverse_difference_divisible_by_9_l100_100817


namespace tan_of_45_deg_l100_100073

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100073


namespace max_value_of_quadratic_l100_100485

theorem max_value_of_quadratic : ∀ (x : ℝ), -9 * x^2 + 27 * x + 15 ≤ 141 / 4 :=
begin
  sorry
end

end max_value_of_quadratic_l100_100485


namespace integral_sqrt_1_minus_x_sq_plus_2x_l100_100986

theorem integral_sqrt_1_minus_x_sq_plus_2x :
  ∫ x in (0 : Real)..1, (Real.sqrt (1 - x^2) + 2 * x) = (Real.pi + 4) / 4 := by
  sorry

end integral_sqrt_1_minus_x_sq_plus_2x_l100_100986


namespace find_x_l100_100709

theorem find_x (x : ℤ) (h : 3^(x - 5) = 9^3) : x = 11 :=
by
  sorry

end find_x_l100_100709


namespace prove_coefficient_condition_implies_n_equals_8_l100_100727

noncomputable def coefficient_of_x3_equals_4_times_x2 (n : ℕ) : Prop :=
  (2^3 * nat.choose n 3) = 4 * (2^2 * nat.choose n 2)

theorem prove_coefficient_condition_implies_n_equals_8 :
  (∃ n : ℕ, coefficient_of_x3_equals_4_times_x2 n) → n = 8 := by
  sorry

end prove_coefficient_condition_implies_n_equals_8_l100_100727


namespace solution_l100_100611

noncomputable def find_m (m : ℕ) : Prop :=
  ∃ m, -180 ≤ m ∧ m ≤ 180 ∧ cos (m:ℝ * (real.pi / 180)) = sin (318 * (real.pi / 180))

theorem solution : find_m 132 ∨ find_m (-132) :=
  sorry

end solution_l100_100611


namespace find_even_function_domain_l100_100265

-- Define the conditions and the proof statement
theorem find_even_function_domain 
  (a : ℝ) (H : a ∈ ({-1, 2, 1/2, 3} : set ℝ)) :
  (∀ x : ℝ, ∃ y : ℝ, y = x^a) ∧ (∀ x : ℝ, ∀ y : ℝ, y = x^a → y = (-x)^a) ↔ (a = 2) :=
by { sorry }

end find_even_function_domain_l100_100265


namespace max_value_of_f_l100_100619

-- Define the function f(x) = 6 * sin x + 8 * cos x
def f (x : ℝ) : ℝ := 6 * Real.sin x + 8 * Real.cos x

-- The theorem to prove that the maximum value of f(x) is 10
theorem max_value_of_f : ∃ x : ℝ, f x = 10 :=
sorry

end max_value_of_f_l100_100619


namespace minimum_a_l100_100305

open Real

theorem minimum_a (a : ℝ) : (∀ (x y : ℝ), x > 0 → y > 0 → (1 / x + a / y) ≥ (16 / (x + y))) → a ≥ 9 := by
sorry

end minimum_a_l100_100305


namespace weather_conditions_l100_100980

variable (T : ℝ) (sunny : Prop) (W : ℝ) (crowded : Prop)

axiom crowded_condition : T ≥ 85 ∧ sunny ∧ W < 10 → crowded
axiom not_crowded : ¬ crowded

theorem weather_conditions :
  ¬ crowded → (T < 85 ∨ ¬ sunny ∨ W ≥ 10) :=
by
  intro h₁
  show T < 85 ∨ ¬ sunny ∨ W ≥ 10
  have h₂ := λ h : (¬ (T ≥ 85 ∧ sunny ∧ W < 10)) , (T < 85 ∨ ¬ sunny ∨ W ≥ 10),
  apply h₂
  intro h₃
  cases h₃
  contradiction
  sorry

end weather_conditions_l100_100980


namespace ZAMENA_correct_l100_100944

-- Define the digits and assigning them to the letters
noncomputable def A : ℕ := 2
noncomputable def M : ℕ := 1
noncomputable def E : ℕ := 2
noncomputable def H : ℕ := 3

-- Define the number ZAMENA
noncomputable def ZAMENA : ℕ :=
  5 * 10^5 + A * 10^4 + M * 10^3 + E * 10^2 + H * 10 + A

-- Define the inequalities and constraints
lemma satisfies_inequalities  : 3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Define uniqueness of each digit set
lemma unique_digits: A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ M ≠ E ∧ M ≠ H ∧ E ≠ H := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Prove that the number ZAMENA matches the specified number
theorem ZAMENA_correct : ZAMENA = 541234 := by
  unfold ZAMENA A M E H
  norm_num

end ZAMENA_correct_l100_100944


namespace tan_45_deg_l100_100139

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100139


namespace sum_of_roots_l100_100635

theorem sum_of_roots (k : ℝ) (m : ℝ) (h : k ≠ 0) :
  (∀ x : ℝ, k * x^3 + 2 * k * x^2 + 6 * k * x + 2 = 0 → sum_roots = m) → m = -2 :=
by
  sorry

end sum_of_roots_l100_100635


namespace tan_45_deg_eq_one_l100_100031

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100031


namespace total_games_l100_100765

def joan_games_this_year : ℕ := 4
def joan_games_last_year : ℕ := 9

theorem total_games (this_year_games last_year_games : ℕ) 
    (h1 : this_year_games = joan_games_this_year) 
    (h2 : last_year_games = joan_games_last_year) : 
    this_year_games + last_year_games = 13 := 
by
  rw [h1, h2]
  exact rfl

end total_games_l100_100765


namespace tan_45_eq_1_l100_100183

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100183


namespace probability_of_experts_winning_l100_100232

-- Definitions required from the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p
def current_expert_score : ℕ := 3
def current_audience_score : ℕ := 4

-- The main theorem to state
theorem probability_of_experts_winning : 
  p^4 + 4 * p^3 * q = 0.4752 := 
by sorry

end probability_of_experts_winning_l100_100232


namespace tan_45_eq_1_l100_100199

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100199


namespace tan_45_deg_eq_one_l100_100111

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100111


namespace tan_5pi_over_4_l100_100598

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l100_100598


namespace prod_terms_3_pow_503_l100_100338

noncomputable theory

-- Define the sequence and its properties
def geometric_sequence (a : ℕ → ℝ) (c : ℝ) : Prop := ∀ n, a n * a (n + 1) = c

-- Define the specific sequence {a_n} with common product 6 and a_6 = 2
def sequence_an (a : ℕ → ℝ) : Prop := geometric_sequence a 6 ∧ a 6 = 2

-- Define the product of terms a_1, a_5, a_9, ..., a_2005, a_2009
def terms_product (a : ℕ → ℝ) : ℝ :=
  (finset.range (503)).product (λ k, a (4 * k + 1))

-- Theorem stating the product of the specific terms is 3^503
theorem prod_terms_3_pow_503 (a : ℕ → ℝ) (h : sequence_an a) : 
  terms_product a = 3 ^ 503 :=
sorry

end prod_terms_3_pow_503_l100_100338


namespace tan_five_pi_over_four_l100_100596

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l100_100596


namespace Peter_initially_had_33_marbles_l100_100813

-- Definitions based on conditions
def lostMarbles : Nat := 15
def currentMarbles : Nat := 18

-- Definition for the initial marbles calculation
def initialMarbles (lostMarbles : Nat) (currentMarbles : Nat) : Nat :=
  lostMarbles + currentMarbles

-- Theorem statement
theorem Peter_initially_had_33_marbles : initialMarbles lostMarbles currentMarbles = 33 := by
  sorry

end Peter_initially_had_33_marbles_l100_100813


namespace perimeter_of_isosceles_triangle_l100_100748

theorem perimeter_of_isosceles_triangle 
    (P Q R S : Point) 
    (A B C : Point) 
    (triangle_is_isosceles : A ≠ B ∧ AB = AC) 
    (circle_P : Circle P 2) 
    (circle_Q : Circle Q 2) 
    (circle_R : Circle R 2) 
    (circle_S : Circle S 2)
    (PQ_tangent_AB : tangent PQ AB) 
    (PR_tangent_BC : tangent PR BC) 
    (PS_tangent_AC : tangent PS AC) 
    (PQ_parallel_AB : parallel PQ AB)
    (PR_parallel_BC : parallel PR BC)
    (PS_parallel_CA : parallel PS CA)
    (PQ_length : distance P Q = 4)
    (PR_length : distance P R = 4)
    (PS_length : distance P S = 4)
    : perimeter ABC = 36 :=
sorry

end perimeter_of_isosceles_triangle_l100_100748


namespace tan_5pi_over_4_l100_100597

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l100_100597


namespace number_playing_all_three_sports_l100_100339

-- Define the conditions as constants or variables
variables (M N B T Ba B∩T B∩Ba T∩Ba B∩T∩Ba : ℕ)

-- The given conditions
def total_members : Prop := M = 60
def no_sport_members : Prop := N = 10
def badminton_players : Prop := B = 25
def tennis_players : Prop := T = 30
def basketball_players : Prop := Ba = 15
def badminton_and_tennis : Prop := B∩T = 15
def badminton_and_basketball : Prop := B∩Ba = 10
def tennis_and_basketball : Prop := T∩Ba = 5

-- The resulting equation from the inclusion-exclusion principle
def inclusion_exclusion : Prop := 
  B + T + Ba - B∩T - B∩Ba - T∩Ba + B∩T∩Ba = M - N

-- The theorem we want to prove
theorem number_playing_all_three_sports : 
  total_members ∧ no_sport_members ∧ badminton_players ∧ tennis_players ∧ basketball_players ∧ badminton_and_tennis ∧ badminton_and_basketball ∧ tennis_and_basketball → 
  inclusion_exclusion → 
  B∩T∩Ba = 10 :=
sorry

end number_playing_all_three_sports_l100_100339


namespace sqrt_sum_eq_8_l100_100713

theorem sqrt_sum_eq_8 (x : ℝ) (h : √(10 + x) + √(30 - x) = 8) : (10 + x) * (30 - x) = 144 :=
sorry

end sqrt_sum_eq_8_l100_100713


namespace problem_equivalent_l100_100720

theorem problem_equivalent (x : Real) : (sqrt (10 + x) + sqrt (30 - x) = 8) → (10 + x) * (30 - x) = 144 := by
  sorry

end problem_equivalent_l100_100720


namespace max_value_of_function_l100_100616

theorem max_value_of_function : 
  (∀ x : ℝ, f x = 6 * sin x + 8 * cos x) →
  ∃ m : ℝ, (∀ x : ℝ, f x ≤ m) ∧ (∃ x₀ : ℝ, f x₀ = m) :=
begin
  sorry
end

end max_value_of_function_l100_100616


namespace minimum_connected_components_l100_100207

/-- We start with two points A, B on a 6*7 lattice grid. We say two points 
  X, Y are connected if one can reflect several times with respect to points A, B 
  and reach from X to Y. Prove that the minimum number of connected components 
  over all choices of A, B is 8. -/
theorem minimum_connected_components (A B : ℕ × ℕ) 
  (hA : A.1 < 6 ∧ A.2 < 7) (hB : B.1 < 6 ∧ B.2 < 7) :
  ∃ k, k = 8 :=
sorry

end minimum_connected_components_l100_100207


namespace smallest_number_with_prime_factors_of_24_l100_100934

theorem smallest_number_with_prime_factors_of_24 : 
  ∃ b : ℕ, (∀ p : ℕ, p.prime → p ∣ 24 → p ∣ b) ∧ b = 6 :=
by
  sorry

end smallest_number_with_prime_factors_of_24_l100_100934


namespace trains_opposite_directions_passing_time_l100_100472

/-- Two trains travel in opposite directions. The speed of the slower train is 36 kmph,
    and the speed of the faster train is 45 kmph. The length of the faster train is 270.0216 meters.
    We want to prove that the time it takes for the man in the slower train to pass the faster train
    is 12 seconds. -/
theorem trains_opposite_directions_passing_time
  (speed_slower_train : ℝ := 36) -- in kmph
  (speed_faster_train : ℝ := 45) -- in kmph
  (length_faster_train : ℝ := 270.0216) -- in meters
  (relative_speed_kmph : ℝ := speed_slower_train + speed_faster_train) -- in kmph
  (conversion_factor : ℝ := 1000 / 3600) -- conversion factor from kmph to m/s
  (relative_speed_mps : ℝ := relative_speed_kmph * conversion_factor) -- in m/s
  (time_to_pass : ℝ := length_faster_train / relative_speed_mps) -- in seconds
  : time_to_pass ≈ 12 := 
begin
  sorry
end

end trains_opposite_directions_passing_time_l100_100472


namespace calc1_calc2_calc3_calc4_l100_100989

theorem calc1 : (-16) - 25 + (-43) - (-39) = -45 := by
  sorry

theorem calc2 : (-3 / 4)^2 * (-8 + 1 / 3) = -69 / 16 := by
  sorry

theorem calc3 : 16 / (- (1 / 2)) * (3 / 8) - | -45 | / 9 = -17 := by
  sorry

theorem calc4 : -1 ^ 2024 - (2 - 0.75) * (2 / 7) * (4 - (-5)^2) = 13 / 2 := by
  sorry

end calc1_calc2_calc3_calc4_l100_100989


namespace ratio_square_areas_l100_100487

theorem ratio_square_areas (r : ℝ) (h1 : r > 0) :
  let s1 := 2 * r / Real.sqrt 5
  let area1 := (s1) ^ 2
  let h := r * Real.sqrt 3
  let s2 := r
  let area2 := (s2) ^ 2
  area1 / area2 = 4 / 5 := by
  sorry

end ratio_square_areas_l100_100487


namespace determine_constants_l100_100567

theorem determine_constants (a b : ℝ) :
  (∀ θ : ℝ, sin θ ^ 3 = a * sin (3 * θ) + b * sin θ) →
  (a = -1/4 ∧ b = 3/4) :=
by
  sorry

end determine_constants_l100_100567


namespace smallest_m_rightmost_nonzero_digit_odd_l100_100377

/--
Let b (n : ℕ) := (factorial (n + 5)) / (factorial n) for each positive integer n.
Prove that the smallest positive integer m such that the rightmost non-zero digit of b m is odd is 4.
-/
theorem smallest_m_rightmost_nonzero_digit_odd : 
  (∃ m: ℕ, 0 < m ∧ ∀ n ≤ m, (rightmost_non_zero_digit (factorial (n + 5) / factorial n)).odd) :=
sorry

end smallest_m_rightmost_nonzero_digit_odd_l100_100377


namespace supplement_of_quadruple_of_complement_of_30deg_l100_100477

-- Definitions for initial conditions
def initial_angle : ℝ := 30.0
def complement (theta : ℝ) : ℝ := 90.0 - theta
def quadruple (theta : ℝ) : ℝ := 4.0 * theta
def supplement (theta : ℝ) : ℝ := if theta <= 180.0 then 180.0 - theta else 360.0 - theta

-- Theorem statement
theorem supplement_of_quadruple_of_complement_of_30deg 
  (initial_angle := 30.0) : 
  supplement (quadruple (complement initial_angle)) = 120.0 :=
by
  sorry

end supplement_of_quadruple_of_complement_of_30deg_l100_100477


namespace tan_45_deg_eq_one_l100_100024

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100024


namespace experts_win_eventually_l100_100224

noncomputable def p : ℝ := 0.6
noncomputable def q : ℝ := 1 - p

def experts_need : ℕ := 3
def audience_need : ℕ := 2
def max_rounds : ℕ := 4

def winning_probability (experts_need : ℕ) (audience_need : ℕ) (p q : ℝ) : ℝ :=
  if experts_need = 3 ∧ audience_need = 2 ∧ p = 0.6 ∧ q = 0.4 then
    p^4 + 4 * p^3 * q
  else
    0

theorem experts_win_eventually :
  winning_probability experts_need audience_need p q = 0.4752 := sorry

end experts_win_eventually_l100_100224


namespace polynomial_has_real_root_l100_100244

theorem polynomial_has_real_root (b : ℝ) : ∃ x : ℝ, (x^4 + b * x^3 + 2 * x^2 + b * x - 2 = 0) := sorry

end polynomial_has_real_root_l100_100244


namespace marble_prob_calc_l100_100734

noncomputable def marbles := 2003 + 1999
noncomputable def p_s_same_color := (real.of_rat 2005003 + real.of_rat 1995001) / (real.of_rat (marbles * (marbles - 1) / 2))
noncomputable def p_d_diff_color := real.of_rat (2003 * 1999) / (real.of_rat (marbles * (marbles - 1) / 2))

theorem marble_prob_calc :
  abs (p_s_same_color - p_d_diff_color) = real.of_rat 5993 / real.of_rat 8004001 := 
sorry

end marble_prob_calc_l100_100734


namespace domain_f_l100_100439

def f (x : ℝ) : ℝ := real.log (1 - x)

theorem domain_f : {y : ℝ | ∃ x : ℝ, f x = y} = {x | x < 1} := by
  sorry

end domain_f_l100_100439


namespace max_value_quadratic_l100_100483

theorem max_value_quadratic : ∃ x : ℝ, -9 * x^2 + 27 * x + 15 = 35.25 :=
sorry

end max_value_quadratic_l100_100483


namespace missing_digit_correct_l100_100833

theorem missing_digit_correct (d : ℕ) (d < 10) : (3 + 4 + 7 + d + 9) % 9 = 0 → d = 4 :=
by sorry

end missing_digit_correct_l100_100833


namespace derivative_at_minus_2_l100_100317

theorem derivative_at_minus_2 (f : ℝ → ℝ) 
  (h : filter.tendsto (λ Δx, (f (-2 + Δx) - f (-2 - Δx)) / Δx) (nhds 0) (nhds (-2))) :
  deriv f (-2) = -1 := 
sorry

end derivative_at_minus_2_l100_100317


namespace time_away_l100_100525

def hour_angle (n : ℝ) : ℝ := 210 + n / 2
def minute_angle (n : ℝ) : ℝ := 6 * n
def angle_diff (n : ℝ) : ℝ := |(hour_angle n - minute_angle n)|

theorem time_away (n : ℝ) : angle_diff n = 120 → n = 43.64 :=
by
  sorry

end time_away_l100_100525


namespace simplified_expression_l100_100378

def f (x : ℝ) : ℝ := 3 * x + 4
def g (x : ℝ) : ℝ := 2 * x - 1

theorem simplified_expression :
  (f (g (f 3))) / (g (f (g 3))) = 79 / 37 :=
by  sorry

end simplified_expression_l100_100378


namespace trajectory_is_parabola_l100_100853

noncomputable def point := (ℝ × ℝ)

def distance_to_line (P : point) (a b c : ℝ) : ℝ :=
  |a * P.1 + b * P.2 + c| / sqrt (a^2 + b^2)

def distance_to_point (P1 P2 : point) : ℝ :=
  sqrt ((P1.1 - P2.1)^2 + (P1.2 - P2.2)^2)

def satisfies_condition (P : point) : Prop :=
  |distance_to_line P 1 0 5 - distance_to_point P (2, 0)| = 3

theorem trajectory_is_parabola (P : point) (h : satisfies_condition P) : P.2^2 = 8 * P.1 :=
by sorry  -- Proof will be inserted here

end trajectory_is_parabola_l100_100853


namespace modulus_sqrt2_l100_100775

theorem modulus_sqrt2 (x y : ℝ) (h : (1 + complex.I) * (x : ℂ) = 1 + y * complex.I) :
  complex.abs (x + y * complex.I) = real.sqrt 2 :=
sorry

end modulus_sqrt2_l100_100775


namespace number_of_adult_dogs_l100_100762

theorem number_of_adult_dogs (x : ℕ) (h : 2 * 50 + x * 100 + 2 * 150 = 700) : x = 3 :=
by
  -- Definitions from conditions
  have cost_cats := 2 * 50
  have cost_puppies := 2 * 150
  have total_cost := 700
  
  -- Using the provided hypothesis to assert our proof
  sorry

end number_of_adult_dogs_l100_100762


namespace finite_values_iff_integer_l100_100380

-- Definition of the sequence
noncomputable def f (x : ℝ) : ℝ := Real.sin (π * x)
def xSequence (x0 : ℝ) : ℕ → ℝ
| 0     := x0
| (n+1) := f (xSequence n)

-- The proof statement
theorem finite_values_iff_integer (x0 : ℝ) : 
  (∀ n m : ℕ, xSequence x0 n = xSequence x0 m → n = m) ↔ (∃ k : ℤ, x0 = k) :=
sorry

end finite_values_iff_integer_l100_100380


namespace clothing_store_gross_profit_l100_100890

noncomputable def purchase_price : ℝ := 81
noncomputable def original_selling_price (S : ℝ) := S = purchase_price + 0.25 * S
noncomputable def decrease (S : ℝ) := 0.2 * S
noncomputable def new_selling_price (S : ℝ) := S - decrease(S)
noncomputable def gross_profit (original_price : ℝ) := new_selling_price(original_price) - purchase_price

theorem clothing_store_gross_profit (S : ℝ) : 
  original_selling_price(S) → 
  gross_profit(S) = 5.40 :=
by
  sorry

end clothing_store_gross_profit_l100_100890


namespace num_satisfying_integers_l100_100312

theorem num_satisfying_integers : 
  {n : ℕ | (n + 6) * (n - 1) * (n - 15) < 0}.card = 13 :=
sorry

end num_satisfying_integers_l100_100312


namespace max_t_squared_l100_100534

theorem max_t_squared (r : ℝ) (D E F : Point) (hD : D ≠ E) (hE : E ≠ F) (hF : F ≠ D)
  (hDE : segment D E.radius = 2 * r)
  (hInscribed : ∃ (O : Point), is_center_of_circle O [(D,E,F)] ∧ dist O D = r ∧ dist O E = r ∧ dist O F = r) :
  ∃ t : ℝ, (t = dist D F + dist E F) ∧ (t^2 ≤ 8 * r^2) :=
by
  sorry

end max_t_squared_l100_100534


namespace problem_l100_100310

variable (x m : ℝ)

theorem problem (h1 : ∀ x, x^2 - 4x + 3 < 0 ↔ 1 < x ∧ x < 3)
                (h2 : ∀ x, x^2 - 6x + 8 < 0 ↔ 2 < x ∧ x < 4)
                (h3 : 2 < x ∧ x < 3 → 2x^2 - 9x + m < 0) :
                m < 9 := sorry

end problem_l100_100310


namespace tan_45_deg_eq_one_l100_100100

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100100


namespace tan_45_eq_1_l100_100203

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100203


namespace product_of_common_divisors_of_210_and_30_l100_100629

theorem product_of_common_divisors_of_210_and_30 : 
  (∏ d in ({d : ℤ | d ∣ 210 ∧ d ∣ 30}), d) = 324000000 := by
  sorry

end product_of_common_divisors_of_210_and_30_l100_100629


namespace tan_45_eq_1_l100_100011

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100011


namespace correct_statements_count_l100_100835

theorem correct_statements_count : 
    (accurate_to_hundred_thousandth (2.236 * 10^8)) ∧ 
    (∀ a b : ℝ, (opposite_numbers a b → a ≠ 0 → b ≠ 0 → (a / b) = -1)) ∧ 
    (∀ a : ℝ, (|a| = -a → a < 0)) ∧ 
    (∀ a x b y : ℝ, (-7*a^x*b^3 + a^4*b^y = -6*a^4*b^3 → x + y = 7)) ∧ 
    (∀ a b c : ℝ, (a + b + c = 0 → (a ≠ 0 → (solve_eq (ax + b = -c) = 1) → (correct_count = 3)))
:=
sorry

end correct_statements_count_l100_100835


namespace binomial_coefficient_10_5_l100_100984

open Nat

theorem binomial_coefficient_10_5 : binomial 10 5 = 252 := by
  sorry

end binomial_coefficient_10_5_l100_100984


namespace tan_45_deg_eq_one_l100_100157

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100157


namespace quadratic_function_decrease_when_x_leq_3_l100_100730

theorem quadratic_function_decrease_when_x_leq_3 (m : ℝ) :
  (∀ x : ℝ, x ≤ 3 → ((x-m)^2 - 1) is decreasing with respect to x) → m ≥ 3 :=
sorry

end quadratic_function_decrease_when_x_leq_3_l100_100730


namespace shop_makes_off_each_jersey_l100_100434

-- Definition of conditions
variable (J : ℝ)

-- Condition: t-shirt price is $192
def t_shirt_price := 192

-- Condition: A t-shirt costs $158 more than a jersey
def condition := J + 158 = t_shirt_price

-- Theorem: How much does the shop make off each jersey?
theorem shop_makes_off_each_jersey : J = 34 :=
by
  rw [condition]
  sorry

end shop_makes_off_each_jersey_l100_100434


namespace tan_45_eq_1_l100_100014

theorem tan_45_eq_1 :
  let Q : ℝ × ℝ := (1 / Real.sqrt 2, 1 / Real.sqrt 2)
  in ∃ (Q : ℝ × ℝ), Q = (1 / Real.sqrt 2, 1 / Real.sqrt 2) → 
     Real.tan (Real.pi / 4) = 1 := 
by {
  -- The proof is left as an exercise.
  sorry
}

end tan_45_eq_1_l100_100014


namespace find_a_odd_function_l100_100297

noncomputable def f (a x : ℝ) := Real.log (Real.sqrt (x^2 + 1) - a * x)

theorem find_a_odd_function :
  ∀ a : ℝ, (∀ x : ℝ, f a (-x) + f a x = 0) ↔ (a = 1 ∨ a = -1) := by
  sorry

end find_a_odd_function_l100_100297


namespace inverse_function_value_l100_100381

-- Define the function
def f (x: ℝ) := sqrt x - 1

-- Define the inverse function condition
def f_inv (y: ℝ) := y = 3 -> ∃ x: ℝ, f x = y ∧ x = 16

-- The main theorem stating the proof problem
theorem inverse_function_value :
  f_inv 3 :=
by
  sorry

end inverse_function_value_l100_100381


namespace factorial_program_constructs_l100_100991

theorem factorial_program_constructs (n : ℕ) : 
  ∀ (input_construct loop_construct wend_construct : String), 
     input_construct = "INPUT" ∧ loop_construct = "WHILE" ∧ wend_construct = "WEND" → 
     (input_construct, loop_construct, wend_construct) = ("INPUT", "WHILE", "WEND") :=
by
  intro input_construct loop_construct wend_construct
  intro h
  rw [←h.1, ←h.2.1, ←h.2.2]
  rfl

end factorial_program_constructs_l100_100991


namespace max_value_of_f_l100_100618

-- Define the function f(x) = 6 * sin x + 8 * cos x
def f (x : ℝ) : ℝ := 6 * Real.sin x + 8 * Real.cos x

-- The theorem to prove that the maximum value of f(x) is 10
theorem max_value_of_f : ∃ x : ℝ, f x = 10 :=
sorry

end max_value_of_f_l100_100618


namespace tan_45_deg_eq_one_l100_100016

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100016


namespace tan_45_deg_l100_100058

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100058


namespace tan_45_deg_l100_100045

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100045


namespace range_of_y_eq_x_squared_l100_100685

noncomputable def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem range_of_y_eq_x_squared :
  M = { y : ℝ | ∃ x : ℝ, y = x^2 } := by
  sorry

end range_of_y_eq_x_squared_l100_100685


namespace value_of_a_plus_b_l100_100731

noncomputable def quadratic_inequality (a b : ℝ) : Prop :=
  ∀ x : ℝ, (x > -1 ∧ x < 1/3) → ax^2 + bx + 1 > 0

theorem value_of_a_plus_b (a b : ℝ) (h : quadratic_inequality a b) : a + b = -5 :=
sorry

end value_of_a_plus_b_l100_100731


namespace training_days_l100_100423

def total_minutes : ℕ := 5 * 60
def minutes_per_day : ℕ := 10 + 20

theorem training_days :
  total_minutes / minutes_per_day = 10 :=
by
  sorry

end training_days_l100_100423


namespace inverse_of_P_is_R_l100_100852

variable (P Q R S : Type) [has_mul P] [has_mul Q] [has_mul R] [has_mul S]

-- Conditions
axiom h1 : P * S = P 
axiom h2 : S * P = P 
axiom h3 : Q * S = Q 
axiom h4 : S * Q = Q 
axiom h5 : R * S = R 
axiom h6 : S * R = R 
axiom h7 : S * S = S 
axiom h8 : P * R = S 
axiom h9 : R * P = S 

-- Theorem stating that the inverse of P is R
theorem inverse_of_P_is_R : ∃ a, P * a = S ∧ a * P = S := 
by {
  use R,
  exact ⟨h8, h9⟩,
}

end inverse_of_P_is_R_l100_100852


namespace Daniel_candy_distribution_l100_100210

theorem Daniel_candy_distribution :
  ∃ (k : ℕ), (k ≤ 25) ∧ (25 - k) % 4 = 0 :=
begin
  use 1,
  split,
  { norm_num },
  { norm_num }
end

end Daniel_candy_distribution_l100_100210


namespace plywood_perimeter_difference_l100_100904

theorem plywood_perimeter_difference:
  ∃ (rectangles : ℕ → set (ℝ × ℝ)),
    (∀ i, rectangles i ⊆ (set.univ : set (ℝ × ℝ))) ∧
    (∀ i j, i ≠ j → rectangles i ∩ rectangles j = ∅) ∧
    (⋃ i, rectangles i) = set.univ ∧
    (∀ i, (rectangles i).nonempty) ∧
    (∀ i, ∃ (length width : ℝ), (rectangles i) = {(x, y) | x = length ∧ y = width}) ∧
    (exactly $8$ congruent rectangles) ∧
    ∃ (max_perimeter min_perimeter: ℝ), max_perimeter - min_perimeter = 21 :=
sorry

end plywood_perimeter_difference_l100_100904


namespace ratio_of_areas_l100_100846

theorem ratio_of_areas (w r : ℝ) (h : 3 * w = π * r) :
  let l := 2 * w
      area_rectangle := l * w
      area_circle := π * r^2
      ratio := area_rectangle / area_circle
  in ratio = 2 * π / 9 :=
by
  let l := 2 * w
  let area_rectangle := l * w
  let area_circle := π * r^2
  let ratio := area_rectangle / area_circle
  have h1 : l = 2 * w := rfl
  have h2 : area_rectangle = 2 * w^2 := by rw [h1]; ring
  have h3 : area_circle = π * r^2 := rfl
  have h4 : w = π * r / 3 := by ... sorry
  have h5 : area_rectangle = 2 * (π * r / 3)^2 := by rw [h4]; ring
  have h6 : ratio = 2 * π^2 * r^2 / (9 * π * r^2) := by rw [h5, h3]; ring
  have h7 : ratio = 2 * π / 9 := by rw [h6]; field_simp
  exact h7

end ratio_of_areas_l100_100846


namespace proof_problem_l100_100288

-- Given properties of the function f
variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f x = -f (-x))
variable (h_even_shift : ∀ x, f(x + 2) = f(-x - 2))
variable (h_f3 : f 3 = 3)

-- Our goal is to prove that f 7 + f 4 = -3
theorem proof_problem : f 7 + f 4 = -3 :=
sorry

end proof_problem_l100_100288


namespace tan_45_deg_eq_one_l100_100020

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100020


namespace cyclic_quadrilateral_l100_100789

-- Definitions for the geometric objects
variables (A B C D M N : Type)
variables (triangle_ABC : Triangle A B C)
variables (D : Point)
variables (M N : Point)

-- Conditions based on the problem statement
def is_angle_bisector (A B C : Point) : Prop := sorry
def is_perpendicular_bisector (A B : Point) : Prop := sorry
def intersection (l1 l2 : Line) : Point := sorry
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Specifying the problem conditions
axiom cond1 : is_angle_bisector A B (intersection (angle_bisector A C) (line_segment B C))
axiom cond2 : is_perpendicular_bisector (line_segment A D) (line_segment B C)
axiom cond3 : M = intersection (perpendicular_bisector (line_segment A D)) (angle_bisector B)
axiom cond4 : N = intersection (perpendicular_bisector (line_segment A D)) (angle_bisector C)

theorem cyclic_quadrilateral :
  is_cyclic_quadrilateral A D M N :=
sorry

end cyclic_quadrilateral_l100_100789


namespace tan_5pi_over_4_l100_100599

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l100_100599


namespace M_is_range_of_sq_function_l100_100687

noncomputable theory

def M : set ℝ := {y | ∃ x : ℝ, y = x^2}

theorem M_is_range_of_sq_function : M = {y | ∃ x : ℝ, y = x^2} :=
by
  sorry

end M_is_range_of_sq_function_l100_100687


namespace train_crossing_time_l100_100361

-- Define the conditions as constants
def train_length : ℝ := 400 -- the length of the train in meters
def train_speed_kmh : ℝ := 144 -- the speed of the train in km/hr
def kmh_to_ms (s : ℝ) : ℝ := s * 1000 / 3600 -- conversion factor from km/hr to m/s

-- Define the equivalent Lean 4 statement for the problem
theorem train_crossing_time :
  let train_speed_ms := kmh_to_ms train_speed_kmh in
  train_length / train_speed_ms = 10 := by
  sorry

end train_crossing_time_l100_100361


namespace trivia_team_students_l100_100861

theorem trivia_team_students (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) (total_students : ℕ) :
  not_picked = 17 →
  groups = 8 →
  students_per_group = 6 →
  total_students = not_picked + groups * students_per_group →
  total_students = 65 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end trivia_team_students_l100_100861


namespace tan_45_deg_l100_100066

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100066


namespace probability_complement_l100_100326

noncomputable def X : ℝ → ℝ := sorry  -- Normally distributed variable placeholder

theorem probability_complement (a : ℝ) (h₁ : P(X < a) = 0.2) : P(X < 4 - a) = 0.8 :=
by
  sorry

end probability_complement_l100_100326


namespace find_x_l100_100710

theorem find_x (x : ℤ) (h : 3^(x - 5) = 9^3) : x = 11 :=
by
  sorry

end find_x_l100_100710


namespace kids_staying_home_correct_l100_100332

def N : ℕ := 898051
def p : ℝ := 0.745
def kids_staying_home (N : ℕ) (p : ℝ) : ℤ := Int.ofNat N - Int.ofNat (round (N * p))

theorem kids_staying_home_correct :
  kids_staying_home N p = 228703 := sorry

end kids_staying_home_correct_l100_100332


namespace socorro_training_days_l100_100427

def total_hours := 5
def minutes_per_hour := 60
def total_training_minutes := total_hours * minutes_per_hour

def minutes_multiplication_per_day := 10
def minutes_division_per_day := 20
def daily_training_minutes := minutes_multiplication_per_day + minutes_division_per_day

theorem socorro_training_days:
  total_training_minutes / daily_training_minutes = 10 :=
by
  -- proof omitted
  sorry

end socorro_training_days_l100_100427


namespace tan_five_pi_over_four_l100_100595

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
  by
  sorry

end tan_five_pi_over_four_l100_100595


namespace smallest_n_inequality_l100_100279

theorem smallest_n_inequality : 
  ∃ (n : ℕ), (n > 0) ∧ ( ∀ m : ℕ, (m > 0) ∧ ( m < n ) → ¬( ( 1 : ℚ ) / m - ( 1 / ( m + 1 : ℚ ) ) < ( 1 / 15 ) ) ) ∧ ( ( 1 : ℚ ) / n - ( 1 / ( n + 1 : ℚ ) ) < ( 1 / 15 ) ) :=
sorry

end smallest_n_inequality_l100_100279


namespace tan_45_deg_l100_100064

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100064


namespace die_face_down_shaded_triangle_l100_100911

-- Define the tetrahedral die with faces labeled 1, 2, 3, 4.
structure TetrahedralDie where
  faces : Fin 4 → Fin 4
  invariant : ∀ i, faces i = i

-- Define the initial condition and the rotation mechanism
def initial_condition : TetrahedralDie := ⟨id, λ i, rfl⟩

-- The main theorem to be proved
theorem die_face_down_shaded_triangle (d : TetrahedralDie) (initial_face_down : d.faces 3 = 3) :
  (∃ n : ℕ, (d.faces ((3 + n) % 4) = 0)) := 
sorry

end die_face_down_shaded_triangle_l100_100911


namespace nice_circles_intersecting_segment_l100_100399

noncomputable def is_nice_circle (r : ℝ) : Prop :=
  ∃ (m n : ℤ), r ^ 2 = (m^2 + n^2 : ℤ)

theorem nice_circles_intersecting_segment :
  let A : (ℝ × ℝ) := (20, 15)
  let B : (ℝ × ℝ) := (20, 16)
  let segment := {t : ℝ | 15 < t ∧ t < 16}
  ∃ (r_list : List ℝ), 
  (∀ r ∈ r_list, r = Real.sqrt (400 + t ^ 2) ∧ t ∈ segment ∧ is_nice_circle r) ∧ 
  List.length r_list = 10 :=
by
  let A := (20, 15 : ℝ)
  let B := (20, 16 : ℝ)
  let segment := {t : ℝ | 15 < t ∧ t < 16}
  let r_list := [sqrt 626, sqrt 628, sqrt 629, sqrt 634, sqrt 637, sqrt 640, sqrt 641, sqrt 648, sqrt 650, sqrt 653]
  use r_list
  split
  { intros r hr
    split
    { use sqrt (400 + t ^ 2),
      repeat { sorry } },
    { sorry }
  }
  { sorry }

end nice_circles_intersecting_segment_l100_100399


namespace sum_of_distinct_prime_factors_1728_eq_5_l100_100571

theorem sum_of_distinct_prime_factors_1728_eq_5 : 
  let n := 1728,
  let prime_factors := {2, 3},
  (prime_factors.sum : ℕ) = 5 := 
by
  let n := 1728
  have prime_factors := {2, 3}
  sorry

end sum_of_distinct_prime_factors_1728_eq_5_l100_100571


namespace bob_water_percentage_is_36_l100_100560

variable (corn_water_per_acre : ℕ := 20)
variable (cotton_water_per_acre : ℕ := 80)
variable (beans_water_per_acre : ℕ := 2 * corn_water_per_acre)

variable (bob_corn_acres : ℕ := 3)
variable (bob_cotton_acres : ℕ := 9)
variable (bob_beans_acres : ℕ := 12)

variable (brenda_corn_acres : ℕ := 6)
variable (brenda_cotton_acres : ℕ := 7)
variable (brenda_beans_acres : ℕ := 14)

variable (bernie_corn_acres : ℕ := 2)
variable (bernie_cotton_acres : ℕ := 12)

def water_needed (corn_acres cotton_acres beans_acres : ℕ) : ℕ :=
   (corn_acres * corn_water_per_acre) +
   (cotton_acres * cotton_water_per_acre) +
   (beans_acres * beans_water_per_acre)

def farmer_bob_water : ℕ :=
  water_needed bob_corn_acres bob_cotton_acres bob_beans_acres

def farmer_brenda_water : ℕ :=
  water_needed brenda_corn_acres brenda_cotton_acres brenda_beans_acres

def farmer_bernie_water : ℕ :=
  water_needed bernie_corn_acres bernie_cotton_acres 0

def total_water : ℕ :=
  farmer_bob_water + farmer_brenda_water + farmer_bernie_water

def bob_percentage_water : ℕ :=
  (farmer_bob_water * 100) / total_water

/-- Prove that the percentage of the total water used that will go to Farmer Bob's farm is 36% -/
theorem bob_water_percentage_is_36 :
  bob_percentage_water = 36 := by
  sorry

end bob_water_percentage_is_36_l100_100560


namespace simplify_fraction_l100_100253

theorem simplify_fraction (n : ℕ) (hn : 0 < n) :
  (∑ k in Finset.range (n + 1), (1 : ℝ) / Nat.choose (n + 1) k) /
  (∑ k in Finset.range (n + 1), (k : ℝ) / Nat.choose (n + 1) k) = 2 / n := 
sorry

end simplify_fraction_l100_100253


namespace entrance_exit_ways_equal_49_l100_100862

-- Define the number of gates on each side
def south_gates : ℕ := 4
def north_gates : ℕ := 3

-- Define the total number of gates
def total_gates : ℕ := south_gates + north_gates

-- State the theorem and provide the expected proof structure
theorem entrance_exit_ways_equal_49 : (total_gates * total_gates) = 49 := 
by {
  sorry
}

end entrance_exit_ways_equal_49_l100_100862


namespace exists_sphere_passing_through_four_vertices_l100_100995

open EuclideanGeometry

structure PointInSpace :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def distance (p1 p2 : PointInSpace) : ℝ :=
  ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2 + (p1.z - p2.z) ^ 2).sqrt

theorem exists_sphere_passing_through_four_vertices
  (A B C D : PointInSpace)
  (h_non_coplanar : ¬∃ (α β γ δ : ℝ), α * A.x + β * B.x + γ * C.x + δ * D.x = 0
                                        ∧ α * A.y + β * B.y + γ * C.y + δ * D.y = 0
                                        ∧ α * A.z + β * B.z + γ * C.z + δ * D.z = 0) :
  ∃ O : PointInSpace, distance O A = distance O B ∧ distance O A = distance O C ∧ distance O A = distance O D :=
by
  sorry

end exists_sphere_passing_through_four_vertices_l100_100995


namespace initial_necklaces_15_l100_100917

variable (N E : ℕ)
variable (initial_necklaces : ℕ) (initial_earrings : ℕ) (store_necklaces : ℕ) (store_earrings : ℕ) (mother_earrings : ℕ) (total_jewelry : ℕ)

axiom necklaces_eq_initial : N = initial_necklaces
axiom earrings_eq_15 : E = initial_earrings
axiom initial_earrings_15 : initial_earrings = 15
axiom store_necklaces_eq_initial : store_necklaces = initial_necklaces
axiom store_earrings_eq_23_initial : store_earrings = 2 * initial_earrings / 3
axiom mother_earrings_eq_115_store : mother_earrings = 1 * store_earrings / 5
axiom total_jewelry_is_57 : total_jewelry = 57
axiom jewelry_pieces_eq : 2 * initial_necklaces + initial_earrings + store_earrings + mother_earrings = total_jewelry

theorem initial_necklaces_15 : initial_necklaces = 15 := by
  sorry

end initial_necklaces_15_l100_100917


namespace mean_of_roots_independent_of_line_l100_100206

noncomputable def mean_of_roots : ℝ :=
  let f := λ x : ℝ, 2*x^4 + 7*x^3 + 3*x - 5
  let line (m b : ℝ) := λ x : ℝ, m * x + b
  -- We consider the case where f(x) = line(m, b)(x) has four distinct roots x_i
  in -7 / 8

theorem mean_of_roots_independent_of_line 
  (x : ℝ)
  (f := λ x : ℝ, 2*x^4 + 7*x^3 + 3*x - 5)
  (line : ℝ → ℝ → ℝ → ℝ := λ m b x, m * x + b)
  (roots : list ℝ)
  (h_roots : roots.length = 4)
  (h_distinct : roots.nodup)
  (h_intersect : ∀ (xi : ℝ), xi ∈ roots → f xi = line m b xi)
  : (roots.sum / 4) = mean_of_roots := 
sorry

end mean_of_roots_independent_of_line_l100_100206


namespace second_number_mod_12_l100_100488

theorem second_number_mod_12 (x : ℕ) (h : (1274 * x * 1277 * 1285) % 12 = 6) : x % 12 = 1 := 
by 
  sorry

end second_number_mod_12_l100_100488


namespace fly_distance_from_bottom_l100_100930

theorem fly_distance_from_bottom (area_window : ℝ)
  (triangles_count : ℝ)
  (equal_area_of_triangles : (area_window / triangles_count)) :
  area_window = 81 → triangles_count = 6 → 
  let side_length := real.sqrt area_window in
  let area_of_each_triangle := area_window / triangles_count in
  let bottom_two_triangles_area := 2 * area_of_each_triangle in
  let h := (2 * bottom_two_triangles_area) / side_length in
  h = 6 := 
begin
  intros area_eq triangles_eq,
  simp [*, real.sqrt_eq_rfl], -- simplify using the provided conditions and basic computations
  simp only with field_simps, -- simplifying field operations
  sorry -- Fill in the rest of the proof
end

end fly_distance_from_bottom_l100_100930


namespace domain_of_function_l100_100211

def f (x : ℝ) : ℝ := Real.log (1 / (1 - x))

theorem domain_of_function :
  ∀ x, f x = Real.log (1 / (1 - x)) → x < 1 := sorry

end domain_of_function_l100_100211


namespace greatest_whole_number_solution_l100_100602

theorem greatest_whole_number_solution : ∃ k : ℕ, 5 * k - 4 < 3 - 2 * k ∧ ∀ m : ℕ, (m > k → ¬ (5 * m - 4 < 3 - 2 * m)) :=
by
  exists 0
  split
  {
    rw [nat.cast_zero, mul_zero, sub_zero, zero_sub, neg_lt, neg_zero]
    exact zero_lt_three
  }
  {
    intros m hm
    rw [nat.not_lt]
    exact nat.le_of_lt_succ hm
  }

end greatest_whole_number_solution_l100_100602


namespace max_distance_from_P_to_l_l100_100841

open Real

noncomputable def point_P : ℝ × ℝ := (-2, -1)

def line_l (λ : ℝ) := { p : ℝ × ℝ | (1 + 3 * λ) * p.1 + (1 + λ) * p.2 - 2 - 4 * λ = 0 }

theorem max_distance_from_P_to_l : ∃ A : ℝ × ℝ, (λ : ℝ) → A ∈ line_l λ ∧ dist point_P A = sqrt 13 := by
  sorry

end max_distance_from_P_to_l_l100_100841


namespace range_of_t_in_region_l100_100729

theorem range_of_t_in_region : (t : ℝ) → ((1 - t + 1 > 0) → t < 2) :=
by
  intro t
  intro h
  sorry

end range_of_t_in_region_l100_100729


namespace ring_area_l100_100469

theorem ring_area (r1 r2 : ℝ) (h1 : r1 = 12) (h2 : r2 = 5) : 
  (π * r1^2) - (π * r2^2) = 119 * π := 
by simp [h1, h2]; sorry

end ring_area_l100_100469


namespace unique_positive_real_solution_l100_100313

-- Define the function
def f (x : ℝ) : ℝ := x^11 + 9 * x^10 + 19 * x^9 + 2023 * x^8 - 1421 * x^7 + 5

-- Prove the statement
theorem unique_positive_real_solution : ∃! x : ℝ, x > 0 ∧ f x = 0 :=
by
  sorry

end unique_positive_real_solution_l100_100313


namespace cos_angle_AMB_l100_100510

-- Define the conditions of the problem
variables (s l h : ℝ) -- s is the side length of the base, l is the slant height, h is the height of the pyramid.
variables (A B C D E M : Type) -- Points in the pyramid.
variable [regular_square_pyramid A B C D E] -- The given pyramid is regular square.
variable [vertex_pyramid A] -- A is the vertex.
variable [square_base_pyramid B C D E] -- BCDE is the square base.
variable [midpoint M B D] -- M is the midpoint of diagonal BD.

-- Prove the value of cosine of angle ∠AMB
theorem cos_angle_AMB : 
  cos_angle A M B = (l^2 + h^2) / (2 * l * (sqrt (h^2 + s^2 / 2))) :=
sorry

end cos_angle_AMB_l100_100510


namespace parallel_lines_iff_lambda_neg_one_l100_100509

theorem parallel_lines_iff_lambda_neg_one (λ : ℝ) :
  (∀ x y : ℝ, x + λ * y + 9 = 0 ∧ (λ - 2) * x + 3 * y + 3 * λ = 0 → (λ = -1)) ∧
  (λ = -1 → ∀ x y : ℝ, x + λ * y + 9 = 0 ∧ (λ - 2) * x + 3 * y + 3 * λ = 0) :=
by
  sorry

end parallel_lines_iff_lambda_neg_one_l100_100509


namespace tan_45_deg_eq_one_l100_100172

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100172


namespace tan_five_pi_over_four_l100_100583

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
sorry

end tan_five_pi_over_four_l100_100583


namespace area_of_trajectory_l100_100340

theorem area_of_trajectory (V A B C O M N T: ℝ3) (h1: V.distance_to(A) = 10) (h2: V.distance_to(B) = 10) (h3: V.distance_to(C) = 10)
    (h4: midpoint(A, B, C, O)) (h5: M ∈ line_segment(V, O)) (h6: N ∈ triangle(A, B, C))
    (h7: midpoint(M, N, T)): 
    trajectory_area(T, 1) = 2 * π :=
sorry

end area_of_trajectory_l100_100340


namespace magnitude_product_l100_100652

theorem magnitude_product (z1 z2 : ℂ) (hz1 : |z1| = 3) (hz2 : z2 = 2 + I) : |z1 * z2| = 3 * Real.sqrt 5 :=
by
  sorry

end magnitude_product_l100_100652


namespace exchange_rate_change_l100_100548

theorem exchange_rate_change (initial final : ℝ) (h_initial : initial = 32.6587) (h_final : final = 56.2584) : 
  Float.round (final - initial) = 24 := by
  sorry

end exchange_rate_change_l100_100548


namespace train_cross_pole_time_l100_100360

noncomputable def train_time_to_cross_pole (length : ℕ) (speed_km_per_hr : ℕ) : ℕ :=
  let speed_m_per_s := speed_km_per_hr * 1000 / 3600
  length / speed_m_per_s

theorem train_cross_pole_time :
  train_time_to_cross_pole 100 72 = 5 :=
by
  unfold train_time_to_cross_pole
  sorry

end train_cross_pole_time_l100_100360


namespace intersection_l100_100480

noncomputable def intersection_point (L₁ L₂ : ℝ → ℝ → Prop) (p : ℝ × ℝ) :=
  ∃ x y, L₁ x y ∧ L₂ x y ∧ (x, y) = p

def line1 := λ x y : ℝ, y = 3 * x + 4

def line2 := λ x y : ℝ, y = -1/3 * x + (2 + (1 / 3) * 3)

theorem intersection : intersection_point line1 line2 (-3/10, 3.1) := 
  sorry

end intersection_l100_100480


namespace integral_evaluation_l100_100506

-- Defining the integrand
def integrand (x : ℝ) : ℝ := (4 * x^4 + 2 * x^2 - x - 3) / (x * (x - 1) * (x + 1))

-- Asserting the integral evaluation for the given integrand
theorem integral_evaluation : 
  ∫ (x : ℝ) in Real.integrand, integrand x dx = 2 * x^2 + 3 * Real.log (Real.abs x) + Real.log (Real.abs (x - 1)) + 2 * Real.log (Real.abs (x + 1)) + C :=
sorry

end integral_evaluation_l100_100506


namespace point_in_fourth_quadrant_l100_100385

noncomputable def z : ℂ := complex.I * (2 - complex.I) * complex.I
noncomputable def z_conj : ℂ := conj z

theorem point_in_fourth_quadrant : 
    let point := (z_conj.re, z_conj.im) in 
    z_conj.re > 0 ∧ z_conj.im < 0 := 
by
  sorry -- We use sorry to skip the proof

end point_in_fourth_quadrant_l100_100385


namespace probability_ace_both_times_probability_ace_given_ace_first_draw_l100_100457

theorem probability_ace_both_times :
  let total_cards := 52
  let aces := 4
  let first_draw := aces.to_fractions / total_cards.to_fractions
  let remaining_aces := aces - 1
  let remaining_cards := total_cards - 1
  let second_draw := remaining_aces.to_fractions / remaining_cards.to_fractions
  first_draw * second_draw = (1 / 221 : ℚ) := 
sorry

theorem probability_ace_given_ace_first_draw :
  let total_cards := 52
  let aces := 4
  let first_draw := aces.to_fractions / total_cards.to_fractions
  let remaining_aces := aces - 1
  let remaining_cards := total_cards - 1
  let second_draw := remaining_aces.to_fractions / remaining_cards.to_fractions
  (second_draw) = (1 / 17 : ℚ) := 
sorry

end probability_ace_both_times_probability_ace_given_ace_first_draw_l100_100457


namespace tan_five_pi_over_four_l100_100587

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end tan_five_pi_over_four_l100_100587


namespace train_cross_bridge_time_l100_100892

theorem train_cross_bridge_time (train_length : ℕ) (bridge_length : ℕ) (train_speed_kmph : ℕ) 
  (h1 : train_length = 70) (h2 : bridge_length = 80) (h3 : train_speed_kmph = 36) :
  let train_speed_mps := train_speed_kmph * 1000 / 3600 in
  let total_distance := train_length + bridge_length in
  total_distance / train_speed_mps = 15 := 
by
  sorry

end train_cross_bridge_time_l100_100892


namespace vectors_not_coplanar_l100_100544

open Matrix

def a : Fin 3 → ℝ := ![7, 3, 4]
def b : Fin 3 → ℝ := ![-1, -2, -1]
def c : Fin 3 → ℝ := ![4, 2, 4]

def scalarTripleProduct (a b c : Fin 3 → ℝ) : ℝ :=
  let m := ![
    a, b, c
  ]
  det m

theorem vectors_not_coplanar : scalarTripleProduct a b c ≠ 0 :=
by
  unfold scalarTripleProduct
  simp
  have : det ![
      ![7, -1, 4],
      ![3, -2, 2],
      ![4, -1, 4]
    ] = -18 := by sorry
  rw [this]
  exact dec_trivial

end vectors_not_coplanar_l100_100544


namespace probability_of_experts_winning_l100_100231

-- Definitions required from the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p
def current_expert_score : ℕ := 3
def current_audience_score : ℕ := 4

-- The main theorem to state
theorem probability_of_experts_winning : 
  p^4 + 4 * p^3 * q = 0.4752 := 
by sorry

end probability_of_experts_winning_l100_100231


namespace problem_statement_l100_100647

noncomputable def a_seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else 3 * 4^(n-2)

noncomputable def S_seq (n : ℕ) : ℕ :=
  4^(n-1)

noncomputable def b_seq (n : ℕ) : ℝ :=
  if n = 1 then 3/8 else
  3 * (4 : ℝ)^(n-2) / ((4^(n-2) + 1) * (4^(n-1) + 1))

noncomputable def T_seq (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n+1), b_seq (i+1)

theorem problem_statement (n : ℕ) (hn : n > 0) :
  (∀ n ≥ 2, a_seq (n+1) * S_seq (n-1) - a_seq n * S_seq n = 0) → 
  (S_seq 1 = 1) → 
  (S_seq 2 = 4) → 
  (∀ n ≥ 2, S_seq (n+1) = S_seq n * 4) → 
  (∀ n, S_seq n = 4^(n-1)) ∧ 
  (a_seq 1 = 1) ∧ 
  (a_seq 2 = 3) ∧ 
  ((∀ n ≥ 2, a_seq n = 3 * 4^(n-2))) ∧
  (∀ n, 3/8 ≤ T_seq n ∧ T_seq n < 7/8) := by
  intros h1 h2 h3 h4
  -- Proof would go here
  sorry

end problem_statement_l100_100647


namespace tan_45_eq_1_l100_100184

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100184


namespace tan_45_deg_eq_one_l100_100159

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100159


namespace additional_birds_flew_up_l100_100511

-- Defining the conditions from the problem
def original_birds : ℕ := 179
def total_birds : ℕ := 217

-- Defining the question to be proved as a theorem
theorem additional_birds_flew_up : 
  total_birds - original_birds = 38 :=
by
  sorry

end additional_birds_flew_up_l100_100511


namespace thirteen_fact_mod_seventeen_l100_100255

theorem thirteen_fact_mod_seventeen : ∀ {n : ℕ}, nat.prime 17 → (13.fact % 17) = 9 :=
by
  intro n h_prime
  have h1 : 16.fact % 17 = 16!,
    -- Proof of Wilson's theorem would go here
    sorry
  have h2 : 16.fact = (16 * 15 * 14 * 13.fact),
    -- Factorization of 16! would go here
    sorry
  have h_mod : (16 * 15 * 14 * 13.fact) % 17 = -1,
    -- Simplification using modular arithmetic
    sorry
  have h_prod : (16 * 15 * 14) % 17 = 11,
    -- Calculation of product modulo 17
    sorry
  have inv11 : 11⁻¹ % 17 = 8,
    -- Finding modular inverse
    sorry
  have h_final : 13.fact % 17 = 9,
    -- Final result using modular inverse and multiplicative property
    sorry
  exact h_final

end thirteen_fact_mod_seventeen_l100_100255


namespace exchange_rate_change_l100_100551

theorem exchange_rate_change :
  let initial := 32.6587
  let final := 56.2584
  let change := final - initial
  let rounded_change := Real.round change
  rounded_change = 24 :=
by
  sorry

end exchange_rate_change_l100_100551


namespace hexagonal_pattern_invariance_l100_100994

noncomputable theory

open Function

-- Define the rigid motion transformations

def rotation_120 (p : Point) : (Point → Point) := sorry
def translation (d : ℝ) : (Point → Point) := sorry
def reflection_across_ell (p : Point) : (Point → Point) := sorry
def reflection_perpendicular_to_ell (p : Point) : (Point → Point) := sorry

-- Define the conditions
def is_rotation_valid (p : Point) : Prop := rotation_120 p = id
def is_translation_valid (d : ℝ) : Prop := translation d = id
def is_reflection_across_ell_valid (p : Point) : Prop := reflection_across_ell p = id
def is_reflection_perpendicular_to_ell_valid (p : Point) : Prop := reflection_perpendicular_to_ell p = id

-- Main theorem
theorem hexagonal_pattern_invariance :
  ∃ p : Point, ∃ d : ℝ,
  is_rotation_valid p ∧
  is_translation_valid d ∧
  is_reflection_across_ell_valid p ∧
  is_reflection_perpendicular_to_ell_valid p := sorry

end hexagonal_pattern_invariance_l100_100994


namespace tan_45_deg_l100_100144

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100144


namespace maximum_value_f_l100_100612

def f (x : ℝ) : ℝ := 6 * Real.sin x + 8 * Real.cos x

theorem maximum_value_f : ∃ y : ℝ, ∀ x : ℝ, f(x) ≤ y ∧ y = 10 :=
sorry

end maximum_value_f_l100_100612


namespace tan_45_deg_l100_100142

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100142


namespace product_of_de_l100_100859

theorem product_of_de (d e : ℤ) (h1: ∀ (r : ℝ), r^2 - r - 1 = 0 → r^6 - (d : ℝ) * r - (e : ℝ) = 0) : 
  d * e = 40 :=
by
  sorry

end product_of_de_l100_100859


namespace ZAMENA_correct_l100_100943

-- Define the digits and assigning them to the letters
noncomputable def A : ℕ := 2
noncomputable def M : ℕ := 1
noncomputable def E : ℕ := 2
noncomputable def H : ℕ := 3

-- Define the number ZAMENA
noncomputable def ZAMENA : ℕ :=
  5 * 10^5 + A * 10^4 + M * 10^3 + E * 10^2 + H * 10 + A

-- Define the inequalities and constraints
lemma satisfies_inequalities  : 3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Define uniqueness of each digit set
lemma unique_digits: A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ M ≠ E ∧ M ≠ H ∧ E ≠ H := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Prove that the number ZAMENA matches the specified number
theorem ZAMENA_correct : ZAMENA = 541234 := by
  unfold ZAMENA A M E H
  norm_num

end ZAMENA_correct_l100_100943


namespace students_in_front_l100_100902

theorem students_in_front (total_students : ℕ) (students_behind : ℕ) (students_total : total_students = 25) (behind_Yuna : students_behind = 9) :
  (total_students - (students_behind + 1)) = 15 :=
by
  sorry

end students_in_front_l100_100902


namespace problem_area_l100_100374

noncomputable def point := (ℝ × ℝ)
def distance (p1 p2 : point) : ℝ := real.sqrt (((p2.1 - p1.1) ^ 2) + ((p2.2 - p1.2) ^ 2))

-- Define points A, B, C, D, X, and Y
def A : point := (-5, 0)
def X : point := (0, 0)
def Y : point := (3, 0)
def C : point := (7, 0)

-- Given conditions
def AX : ℝ := distance A X = 5
def XY : ℝ := distance X Y = 3
def YC : ℝ := distance Y C = 4
def angle_AXD : Prop := ∀ (Dx Dy : ℝ), Dx ≠ 0 → Dy ≠ 0 → real.atan2 Dy Dx - real.atan2 0 (-5) = real.pi / 2
def angle_BYC : Prop := ∀ (Bx By : ℝ), Bx ≠ 0 → By ≠ 0 → real.atan2 By Bx - real.atan2 0 3 = real.pi / 2

-- Define the rectangle
def B : point := (3, ?d)
def D : point := (-5, ?d)

-- Proof problem: the area of rectangle ABCD (base * height)
def rectangle_area (A B C D : point) : ℝ := distance A C * ?d

theorem problem_area : rectangle_area A (3, ?d) C (-5, ?d) = 60 :=
by
  -- Using the given distances and angles
  sorry

end problem_area_l100_100374


namespace find_x_l100_100262

variable (x : ℝ)

def a : ℝ × ℝ × ℝ := (1, 2, 3)
def b (x : ℝ) : ℝ × ℝ × ℝ := (-1, 1, x)
def orthogonal (u v : ℝ × ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

theorem find_x (h : orthogonal a (b x)) : x = -1/3 := by
  sorry

end find_x_l100_100262


namespace g_property_g_16_solution_l100_100383

noncomputable def g : ℕ → ℕ := sorry

theorem g_property (a b : ℕ) : 3 * g (a^2 + b^2) = (g a)^3 + (g b)^3 := sorry

theorem g_16_solution : 
  let possible_values := [0, 1, 81] in
  let n := possible_values.length in
  let s := possible_values.sum in
  n * s = 246 := 
by {
  let possible_values := [0, 1, 81],
  let n := possible_values.length,
  let s := possible_values.sum,
  have h_n : n = 3 := rfl,
  have h_s : s = 82 := rfl,
  calc
    n * s = 3 * 82 : by rw [h_n, h_s]
      ... = 246 : by norm_num
}

end g_property_g_16_solution_l100_100383


namespace find_g5_l100_100837

variable (g : ℝ → ℝ)

-- Formal definition of the condition for the function g in the problem statement.
def functional_eq_condition :=
  ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2

-- The main statement to prove g(5) = 1 under the given condition.
theorem find_g5 (h : functional_eq_condition g) :
  g 5 = 1 :=
sorry

end find_g5_l100_100837


namespace tan_45_deg_eq_one_l100_100155

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100155


namespace tan_45_deg_l100_100131

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100131


namespace distance_between_points_le_sqrt2_div_2_l100_100543

theorem distance_between_points_le_sqrt2_div_2 :
  ∀ (points : Fin 5 → ℝ × ℝ), (∀ i, points i.1 ≥ 0 ∧ points i.1 ≤ 1 ∧ points i.2 ≥ 0 ∧ points i.2 ≤ 1) →
    ∃ (i j : Fin 5), i ≠ j ∧ dist (points i) (points j) ≤ sqrt 2 / 2 :=
by sorry

end distance_between_points_le_sqrt2_div_2_l100_100543


namespace area_of_rectangle_l100_100438

theorem area_of_rectangle (A B C D E F : Type) 
    [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] 
    (A_perpendicular : ∀ (l : A → B) (l' : A → C) (DB : D → E), l ⊥ l' ∧ l' ⊥ DB)
    (segment_lengths : Metric.dist D E = 2 ∧ Metric.dist E F = 2 ∧ Metric.dist F D = 4) 
    (rectangle_ABCD : Rectangle A B C D) 
    : Area (rectangle A B C D) = 16 * sqrt 2 :=
sorry

end area_of_rectangle_l100_100438


namespace cyclic_quadrilateral_AD_squared_l100_100373

theorem cyclic_quadrilateral_AD_squared 
  (A B C D : Type) [IsCyclicQuadrilateral A B C D]
  (R: ℝ) (circumradius : R = 100 * Real.sqrt 3)
  (AC : ℝ) (AC_length : AC = 300)
  (DBC : ℝ) (angle_DBC : DBC = 15) : 
  (AD : ℝ) (AD_squared : AD^2 = 60000) :=
  sorry

end cyclic_quadrilateral_AD_squared_l100_100373


namespace max_abs_diff_l100_100450

-- Definitions of the functions f and g
def f (x : ℝ) := x^2
def g (x : ℝ) := x^3

-- The absolute difference on the interval
def absDifference (a b : ℝ) (f g : ℝ → ℝ) := 
  ⨆ x ∈ set.Icc a b, abs (f x - g x)

-- Declaration of the goal
theorem max_abs_diff : absDifference 0 1 f g = (4 / 27) := 
by
  sorry

end max_abs_diff_l100_100450


namespace turtles_remaining_l100_100524

noncomputable def num_turtles_remained_on_log : ℕ :=
  let original_turtles := 25
  let turtles_climbed := 5 * original_turtles - 4
  let total_turtles := original_turtles + turtles_climbed
  let frightened_turtles := total_turtles / 3
  total_turtles - frightened_turtles

theorem turtles_remaining (h : num_turtles_remained_on_log = 98) : 
  let original_turtles := 25
  let turtles_climbed := 5 * original_turtles - 4
  let total_turtles := original_turtles + turtles_climbed
  let frightened_turtles := total_turtles / 3
  total_turtles - frightened_turtles = 98 :=
by
  subst h
  sorry

end turtles_remaining_l100_100524


namespace find_M_and_solutions_l100_100249

open Real

def max_value_of_abs_expr : ℝ :=
  (5 / 2) * sqrt 2

def solution_set : Set ℝ :=
  { x | -sqrt 2 ≤ x ∧ x ≤ sqrt 2 }

theorem find_M_and_solutions (M : ℝ) :
  M = max_value_of_abs_expr →
  { x | |x - sqrt 2| + |x + 2 * sqrt 2| ≤ M } = solution_set :=
begin
  sorry
end

end find_M_and_solutions_l100_100249


namespace tan_of_45_deg_l100_100080

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100080


namespace smallest_bobs_number_l100_100936

theorem smallest_bobs_number (bob_number : ℕ) :
  let alice_number := 24 in
  prime_factors alice_number ⊆ prime_factors bob_number →
  bob_number > 0 →
  ∃ n : ℕ, n = 6 ∧ prime_factors alice_number ⊆ prime_factors n :=
by
  sorry

end smallest_bobs_number_l100_100936


namespace price_on_friday_is_correct_l100_100804

-- Define initial price on Tuesday
def price_on_tuesday : ℝ := 50

-- Define the percentage increase on Wednesday (20%)
def percentage_increase : ℝ := 0.20

-- Define the percentage discount on Friday (15%)
def percentage_discount : ℝ := 0.15

-- Define the price on Wednesday after the increase
def price_on_wednesday : ℝ := price_on_tuesday * (1 + percentage_increase)

-- Define the price on Friday after the discount
def price_on_friday : ℝ := price_on_wednesday * (1 - percentage_discount)

-- Theorem statement to prove that the price on Friday is 51 dollars
theorem price_on_friday_is_correct : price_on_friday = 51 :=
by
  sorry

end price_on_friday_is_correct_l100_100804


namespace avg_price_l100_100413

-- Definitions based on conditions from the problem
def total_books := 65 + 55
def total_amount := 1080 + 840
def average_price_per_book := total_amount / total_books

-- Statement that we need to prove
theorem avg_price (h1 : total_books = 120) (h2 : total_amount = 1920): average_price_per_book = 16 := 
by
  rw [←h1, ←h2]
  exact rfl

end avg_price_l100_100413


namespace symmetry_at_1_range_of_a_l100_100796

open Set Real

-- Define the function f with its property
noncomputable def f : ℝ → ℝ := 
λ x, if (-1 ≤ x ∧ x ≤ 1) then x^3 else 0

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x

-- 1. Prove that x = 1 is a line of symmetry for the graph of f
theorem symmetry_at_1 : ∀ x : ℝ, f (1 + x) = f (1 - x) := sorry

-- 2. Define the expression of f(x) for x ∈ [1, 5]
noncomputable def f_piecewise (x : ℝ) : ℝ :=
if 1 ≤ x ∧ x ≤ 3 then -(x - 2) ^ 3 else if 3 ≤ x ∧ x ≤ 5 then (x - 4) ^ 3 else 0

-- 3. Prove that the range of a such that {x | |f x| > a} ≠ ∅ is a < 1
theorem range_of_a (a : ℝ) (h : ∃ x, |f x| > a) : a < 1 := sorry

end symmetry_at_1_range_of_a_l100_100796


namespace zamena_correct_l100_100954

variables {M E H A : ℕ}

def valid_digits := {1, 2, 3, 4, 5}

-- Define the conditions as hypotheses
def conditions : Prop :=
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
  M ∈ valid_digits ∧ E ∈ valid_digits ∧ H ∈ valid_digits ∧ A ∈ valid_digits ∧
  ∀ x y, x ∈ {M, E, H, A} ∧ y ∈ {M, E, H, A} → x ≠ y

-- Define the proof problem that these conditions imply the correct answer
theorem zamena_correct : conditions → M = 1 ∧ E = 2 ∧ H = 3 ∧ A = 4 ∧ 10000 * 5 + 1000 * A + 100 * M + 10 * E + H = 541234 :=
by sorry

end zamena_correct_l100_100954


namespace tan_45_deg_eq_one_l100_100165

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100165


namespace family_work_solution_l100_100857

noncomputable def family_work_problem : Prop :=
  ∃ (M W : ℕ),
    M + W = 15 ∧
    (M * (9/120) + W * (6/180) = 1) ∧
    W = 3

theorem family_work_solution : family_work_problem :=
by
  sorry

end family_work_solution_l100_100857


namespace experts_win_probability_l100_100221

noncomputable def probability_of_experts_winning (p : ℝ) (q : ℝ) (needed_expert_wins : ℕ) (needed_audience_wins : ℕ) : ℝ :=
  p ^ 4 + 4 * (p ^ 3 * q)

-- Probability values
def p : ℝ := 0.6
def q : ℝ := 1 - p

-- Number of wins needed
def needed_expert_wins : ℕ := 3
def needed_audience_wins : ℕ := 2

theorem experts_win_probability :
  probability_of_experts_winning p q needed_expert_wins needed_audience_wins = 0.4752 :=
by
  -- Proof would go here
  sorry

end experts_win_probability_l100_100221


namespace tan_of_45_deg_l100_100082

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100082


namespace tan_of_45_deg_l100_100076

theorem tan_of_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_of_45_deg_l100_100076


namespace triangle_area_range_of_expression_l100_100756

noncomputable section

variable (A B C : ℝ)
variable (a b c : ℝ)

-- Problem 1
def angles_form_arithmetic_sequence : Prop := (B = (A + C) / 2)
def sides_and_sum : Prop := (b = 7 ∧ a + c = 13)

theorem triangle_area (h : angles_form_arithmetic_sequence A B C) (h1 : sides_and_sum a b c) :
  let S := 1 / 2 * a * c * Real.sin 60 in
  S = 10 * Real.sqrt 3 := by
sorry

-- Problem 2
theorem range_of_expression (h : angles_form_arithmetic_sequence A B C) :
  ∃ x y, (x, y) = (1, 2) ∧
  ∀ (A : ℝ), 0 < A ∧ A < 2 * π / 3 →
  let expr := Real.sqrt 3 * Real.sin A + Real.sin (B - π / 6) in
  expr ∈ Ioc x y := by
sorry

end triangle_area_range_of_expression_l100_100756


namespace tan_45_deg_eq_one_l100_100107

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100107


namespace sum_of_roots_l100_100496

theorem sum_of_roots : 
  let equation := λ x : ℝ, (x - 5)^2 - 9
  in (∃ R1 R2 : ℝ, equation R1 = 0 ∧ equation R2 = 0 ∧ R1 + R2 = 10) :=
by
  sorry

end sum_of_roots_l100_100496


namespace area_Behd_l100_100358

-- Given conditions in the problem
variables 
  (A B C D E F G H : Type) 
  [RealVectorSpace A] 
  [RealVectorSpace B] 
  [RealVectorSpace C] 
  [RealVectorSpace D] 
  [RealVectorSpace E] 
  [RealVectorSpace F] 
  [RealVectorSpace G] 
  [RealVectorSpace H]
  (BE EF FC : ℝ)
  (BD AD : ℝ)
  (area_AHG area_GFC : ℝ)

-- Specific lengths and ratios given as conditions
axiom BE_eq_1 : BE = 1
axiom EF_eq_6 : EF = 6
axiom FC_eq_2 : FC = 2
axiom BD_eq_2AD : BD = 2 * AD
axiom area_AHG_eq_4d86 : area_AHG = 4.86
axiom area_GFC_eq_2 : area_GFC = 2

-- The question to prove that area of quadrilateral BEHD is 2.86
theorem area_Behd (BE EF FC BD AD : Real) (area_AHG area_GFC : Real) 
  (BE_eq_1 : BE = 1)
  (EF_eq_6 : EF = 6)
  (FC_eq_2 : FC = 2)
  (BD_eq_2AD : BD = 2 * AD)
  (area_AHG_eq_4d86 : area_AHG = 4.86)
  (area_GFC_eq_2 : area_GFC = 2) :
  ∃ area_BEHD, area_BEHD = 2.86 := 
by
  sorry

end area_Behd_l100_100358


namespace tan_45_deg_eq_one_l100_100038

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100038


namespace part_I_intersection_distance_part_II_max_OP_OQ_l100_100751

-- Part I: Prove distance from the intersection point of C₃ and C₄ to the pole
theorem part_I_intersection_distance : 
  ∃ θ : ℝ, (0 < θ ∧ θ < π / 2) ∧ (1 + cos θ) = ((1 + sqrt 5) / 2) * (1 / cos θ) :=
by
  sorry

-- Part II: Prove the maximum value of |OP| + |OQ|
theorem part_II_max_OP_OQ : 
  ∃ α : ℝ, (0 < α ∧ α < π / 2) ∧ 1 + 2 * sin α + cos α = 1 + sqrt 5 :=
by
  sorry

end part_I_intersection_distance_part_II_max_OP_OQ_l100_100751


namespace tan_45_deg_eq_one_l100_100158

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100158


namespace tan_45_deg_l100_100130

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100130


namespace tan_45_deg_eq_one_l100_100027

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100027


namespace length_of_bridge_l100_100501

theorem length_of_bridge (speed_kmh : ℝ) (time_min : ℝ) (length_m : ℝ) :
  speed_kmh = 5 → time_min = 15 → length_m = 1250 :=
by
  sorry

end length_of_bridge_l100_100501


namespace compute_c_l100_100655

noncomputable def polynomial := (x : ℝ) → x ^ 4 + c * x ^ 3 + d * x ^ 2 + e * x - 48 = 0

theorem compute_c (c d e : ℚ) 
  (h1 : -2 + 3 * Real.sqrt 5 ∈ {x : ℝ | polynomial x}) 
  (h2 : -2 - 3 * Real.sqrt 5 ∈ {x : ℝ | polynomial x}) : 
  c = 5 := 
sorry

end compute_c_l100_100655


namespace tan_alpha_eq_one_third_cos2alpha_over_expr_l100_100319

theorem tan_alpha_eq_one_third_cos2alpha_over_expr (α : ℝ) (h : Real.tan α = 1/3) :
  (Real.cos (2 * α)) / (2 * Real.sin α * Real.cos α + (Real.cos α)^2) = 8 / 15 :=
by
  -- This is the point where the proof steps will go, but we leave it as a placeholder.
  sorry

end tan_alpha_eq_one_third_cos2alpha_over_expr_l100_100319


namespace tan_45_deg_eq_one_l100_100153

theorem tan_45_deg_eq_one :
  let θ := 45
  (∀ θ, tan θ = sin θ / cos θ) →
  (sin 45 = sqrt 2 / 2) →
  (cos 45 = sqrt 2 / 2) →
  tan 45 = 1 :=
by
  intros θ h_tan h_sin h_cos
  sorry

end tan_45_deg_eq_one_l100_100153


namespace kenneth_earnings_l100_100773

-- Definitions according to conditions
def spent_percent : ℝ := 0.10
def left_amount : ℝ := 405.0
def total_earnings : ℝ := left_amount / (1 - spent_percent)

-- Theorem statement
theorem kenneth_earnings (spent_percent : ℝ) (left_amount : ℝ) : total_earnings = 450 :=
by
  sorry

end kenneth_earnings_l100_100773


namespace no_very_convex_function_l100_100557

noncomputable def is_very_convex (f : ℝ → ℝ) := 
  ∀ x y : ℝ, (f(x) + f(y)) / 2 ≥ f((x + y) / 2) + |x - y|

theorem no_very_convex_function : ¬ ∃ f : ℝ → ℝ, is_very_convex f :=
by
  sorry

end no_very_convex_function_l100_100557


namespace tan_45_deg_l100_100054

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100054


namespace remainder_of_polynomial_l100_100876

theorem remainder_of_polynomial (p : ℤ[X])
  (h1 : ∀ x, p.eval (-2) = 8)
  (h2 : ∀ x, p.eval (-4) = -10) : 
  ∃ a b : ℤ, (∀ x, (p.eval x = ((x + 2) * (x + 4) * (q.eval x) + a * x + b))) ∧ a = 9 ∧ b = 26 := 
sorry 

end remainder_of_polynomial_l100_100876


namespace yogurt_combinations_l100_100932

theorem yogurt_combinations (flavors toppings : ℕ) (hflavors : flavors = 5) (htoppings : toppings = 8) :
  (flavors * Nat.choose toppings 3 = 280) :=
by
  rw [hflavors, htoppings]
  sorry

end yogurt_combinations_l100_100932


namespace experts_win_eventually_l100_100228

noncomputable def p : ℝ := 0.6
noncomputable def q : ℝ := 1 - p

def experts_need : ℕ := 3
def audience_need : ℕ := 2
def max_rounds : ℕ := 4

def winning_probability (experts_need : ℕ) (audience_need : ℕ) (p q : ℝ) : ℝ :=
  if experts_need = 3 ∧ audience_need = 2 ∧ p = 0.6 ∧ q = 0.4 then
    p^4 + 4 * p^3 * q
  else
    0

theorem experts_win_eventually :
  winning_probability experts_need audience_need p q = 0.4752 := sorry

end experts_win_eventually_l100_100228


namespace only_number_smaller_than_zero_l100_100939

theorem only_number_smaller_than_zero : ∀ (x : ℝ), (x = 5 ∨ x = 2 ∨ x = 0 ∨ x = -Real.sqrt 2) → x < 0 → x = -Real.sqrt 2 :=
by
  intro x hx h
  sorry

end only_number_smaller_than_zero_l100_100939


namespace trig_min_90_degrees_l100_100247

theorem trig_min_90_degrees (A : ℝ) (hA : cos (A / 2) + sin (A / 2) <= cos (π / 4) + sin (π / 4) := by sorry) : 
  (cos (A / 2) + sin (A / 2)) = cos (π / 4) + sin (π / 4) → 
  A = π / 2 :=
sorry

end trig_min_90_degrees_l100_100247


namespace symmetric_paths_on_5x5_chessboard_l100_100474

-- Define the board and vertices
def board := ℤ × ℤ
def vertices := {v : board | 0 ≤ v.1 ∧ v.1 ≤ 5 ∧ 0 ≤ v.2 ∧ v.2 ≤ 5}

def is_vertex (v : board) : Prop :=
  0 ≤ v.1 ∧ v.1 ≤ 5 ∧ 0 ≤ v.2 ∧ v.2 ≤ 5

-- Define path and conditions
def is_path (p : list board) : Prop :=
  p ≠ [] ∧ p.head = p.last ∧ ∀ v ∈ p, is_vertex v ∧ ∀ ⦃a b⦄, a ∈ p ∧ b ∈ p ∧ a ≠ b → a ≠ b

def is_symmetric (p : list board) :=
  ∀ (v : board), v ∈ p ↔ (5 - v.1, v.2) ∈ p ∧ (v.1, 5 - v.2) ∈ p

noncomputable def valid_paths := [
  -- Define the valid paths AB, BC, CA
  -- Assuming these paths were pre-computed and verified somehow.
  [(0,0), (1,0), (2,0), (3,0), (4,0), (5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (4,5), (3,5), (2,5), (1,5), (0,5), (0,4), (0,3), (0,2), (0,1)],
  [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (1,5), (2,5), (3,5), (4,5), (5,5), (5,4), (5,3), (5,2), (5,1), (5,0), (4,0), (3,0), (2,0), (1,0)]
  /* List the coordinates corresponding to the paths AB, BC, CA */
]

theorem symmetric_paths_on_5x5_chessboard : ∀ p, is_path p ∧ is_symmetric p → p ∈ valid_paths :=
sorry

end symmetric_paths_on_5x5_chessboard_l100_100474


namespace gcd_of_324_and_135_l100_100840

theorem gcd_of_324_and_135 : Nat.gcd 324 135 = 27 :=
by
  sorry

end gcd_of_324_and_135_l100_100840


namespace weight_of_b_is_37_l100_100826

variables {a b c : ℝ}

-- Conditions
def average_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 45
def average_ab (a b : ℝ) : Prop := (a + b) / 2 = 40
def average_bc (b c : ℝ) : Prop := (b + c) / 2 = 46

-- Statement to prove
theorem weight_of_b_is_37 (h1 : average_abc a b c) (h2 : average_ab a b) (h3 : average_bc b c) : b = 37 :=
by {
  sorry
}

end weight_of_b_is_37_l100_100826


namespace tan_45_deg_l100_100061

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100061


namespace coeff_fourth_term_expansion_l100_100569

theorem coeff_fourth_term_expansion :
  (3 : ℚ) ^ 2 * (-1 : ℚ) / 8 * (Nat.choose 8 3) = -63 :=
by
  sorry

end coeff_fourth_term_expansion_l100_100569


namespace tan_45_deg_eq_one_l100_100115

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100115


namespace magnitude_of_z_l100_100721

-- Definition of the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Definition of the complex number z
def z : ℂ := i - 1

-- Theorem statement: Prove that the magnitude of z is sqrt 2
theorem magnitude_of_z : Complex.abs z = Real.sqrt 2 := 
by 
  sorry

end magnitude_of_z_l100_100721


namespace diagonals_intersect_at_midpoint_l100_100815

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem diagonals_intersect_at_midpoint :
  let rect_vert1 := (2 : ℝ, -6 : ℝ)
  let rect_vert2 := (12 : ℝ, 8 : ℝ)
  midpoint rect_vert1 rect_vert2 = (7, 1) :=
by
  let rect_vert1 := (2 : ℝ, -6 : ℝ)
  let rect_vert2 := (12 : ℝ, 8 : ℝ)
  let mp := midpoint rect_vert1 rect_vert2
  sorry

end diagonals_intersect_at_midpoint_l100_100815


namespace sum_of_inscribed_angles_in_pentagon_l100_100925

theorem sum_of_inscribed_angles_in_pentagon (P : Type) [EuclideanGeometry P] {c : circle P} 
  (pentagon : regular_polygon P 5) (h_in : inscribed_in_circle pentagon c) : 
  sum_of_inscribed_angles pentagon = 180 := 
sorry

end sum_of_inscribed_angles_in_pentagon_l100_100925


namespace smallest_bobs_number_l100_100935

theorem smallest_bobs_number (bob_number : ℕ) :
  let alice_number := 24 in
  prime_factors alice_number ⊆ prime_factors bob_number →
  bob_number > 0 →
  ∃ n : ℕ, n = 6 ∧ prime_factors alice_number ⊆ prime_factors n :=
by
  sorry

end smallest_bobs_number_l100_100935


namespace find_number_l100_100240

theorem find_number (N: ℕ): (N % 131 = 112) ∧ (N % 132 = 98) → 1000 ≤ N ∧ N ≤ 9999 ∧ N = 1946 :=
sorry

end find_number_l100_100240


namespace complex_number_is_real_l100_100500

theorem complex_number_is_real (m : ℝ) (z : ℂ := complex.mk (1 / (m + 5)) (m^2 + 2 * m - 15)) :
  (z.im = 0) → (m ≠ -5) → (m = 3) :=
begin
  intros h_im h_ne,
  sorry
end

end complex_number_is_real_l100_100500


namespace greatest_whole_number_satisfying_inequality_l100_100608

-- Define the problem condition
def inequality (x : ℤ) : Prop := 5 * x - 4 < 3 - 2 * x

-- Prove that under this condition, the greatest whole number satisfying it is 0
theorem greatest_whole_number_satisfying_inequality : ∃ (n : ℤ), inequality n ∧ ∀ (m : ℤ), inequality m → m ≤ n :=
begin
  use 0,
  split,
  { -- Proof that 0 satisfies the inequality
    unfold inequality,
    linarith, },
  { -- Proof that 0 is the greatest whole number satisfying the inequality
    intro m,
    unfold inequality,
    intro hm,
    linarith, }
end

#check greatest_whole_number_satisfying_inequality

end greatest_whole_number_satisfying_inequality_l100_100608


namespace zamena_correct_l100_100951

variables {M E H A : ℕ}

def valid_digits := {1, 2, 3, 4, 5}

-- Define the conditions as hypotheses
def conditions : Prop :=
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
  M ∈ valid_digits ∧ E ∈ valid_digits ∧ H ∈ valid_digits ∧ A ∈ valid_digits ∧
  ∀ x y, x ∈ {M, E, H, A} ∧ y ∈ {M, E, H, A} → x ≠ y

-- Define the proof problem that these conditions imply the correct answer
theorem zamena_correct : conditions → M = 1 ∧ E = 2 ∧ H = 3 ∧ A = 4 ∧ 10000 * 5 + 1000 * A + 100 * M + 10 * E + H = 541234 :=
by sorry

end zamena_correct_l100_100951


namespace min_max_value_when_a_is_negative_one_monotonicity_range_of_a_l100_100267

-- Lean statement part for (1)
theorem min_max_value_when_a_is_negative_one :
  let f (x : ℝ) (a : ℝ) := x^2 + 2*a*x + 2 in
  (f 1 (-1) = 1) ∧ (f (-3) (-1) = 17) :=
by
  -- min_value (f 1 (-1) = 1)
  -- max_value (f (-3) (-1) = 17)
  sorry

-- Lean statement part for (2)
theorem monotonicity_range_of_a :
  let f (x : ℝ) (a : ℝ) := x^2 + 2*a*x + 2 in
  ∃ a : ℝ, (a ≤ -3 ∨ a ≥ 3) ∧ is_monotonic_on (f(x) a) (Set.Icc (-3) (3)) :=
by
  -- proof that the function is monotonic on the given range of a
  sorry

end min_max_value_when_a_is_negative_one_monotonicity_range_of_a_l100_100267


namespace log_condition_necessary_but_not_sufficient_l100_100283

theorem log_condition_necessary_but_not_sufficient (e : ℝ) (a b : ℝ) 
  (h_e : e = Real.exp 1) (h_a_pos : a > 0) (h_a_neq_one : a ≠ 1) 
  (h_b_pos : b > 0) (h_b_neq_one : b ≠ 1) : 
  ¬(∀ a b, log a 2 > log b e → (0 < a ∧ a < b ∧ b < 1)) ∧ 
  (0 < a ∧ a < b ∧ b < 1 → log a 2 > log b e) :=
by
  sorry

end log_condition_necessary_but_not_sufficient_l100_100283


namespace alpha_numeric_puzzle_l100_100749

theorem alpha_numeric_puzzle : 
  ∀ (a b c d e f g h i : ℕ),
  (∀ x y : ℕ, x ≠ 0 → y ≠ 0 → x ≠ y) →
  100 * a + 10 * b + c + 100 * d + 10 * e + f + 100 * g + 10 * h + i = 1665 → 
  c + f + i = 15 →
  b + e + h = 15 :=
by
  intros a b c d e f g h i distinct nonzero_sum unit_digits_sum
  sorry

end alpha_numeric_puzzle_l100_100749


namespace domain_of_function_correct_l100_100570

noncomputable def domain_of_function (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (2 - x > 0) ∧ (Real.logb 10 (2 - x) ≠ 0)

theorem domain_of_function_correct :
  {x : ℝ | domain_of_function x} = {x : ℝ | x ∈ Set.Icc (-1 : ℝ) 1 \ {1}} ∪ {x : ℝ | x ∈ Set.Ioc 1 2} :=
by
  sorry

end domain_of_function_correct_l100_100570


namespace sum_of_roots_of_x_squared_eq_16x_minus_10_l100_100495

theorem sum_of_roots_of_x_squared_eq_16x_minus_10 :
  (∑ x in ({x : ℝ | x^2 = 16 * x - 10}), x) = 16 := 
sorry

end sum_of_roots_of_x_squared_eq_16x_minus_10_l100_100495


namespace tan_45_deg_l100_100141

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100141


namespace two_distinct_real_roots_of_quadratic_l100_100257

def operation (a b : ℝ) : ℝ := a * b^2 - b

theorem two_distinct_real_roots_of_quadratic (k : ℝ) :
  ∃ x : ℝ, (1 ※ x = k) ∧ unique (root for 1 ※ x = k) ↔ (k > -1/4) :=
begin
  sorry
end

end two_distinct_real_roots_of_quadratic_l100_100257


namespace soda_relationship_l100_100390

variable (J : ℝ) (L : ℝ) (A : ℝ)

def condition1 : Prop := L = 1.50 * J
def condition2 : Prop := A = 1.25 * J
def goal : Prop := (L - A) / A = 0.20

theorem soda_relationship (h1 : condition1 J L) (h2 : condition2 J A) : goal J L A :=
by
  sorry

end soda_relationship_l100_100390


namespace ZAMENA_correct_l100_100946

-- Define the digits and assigning them to the letters
noncomputable def A : ℕ := 2
noncomputable def M : ℕ := 1
noncomputable def E : ℕ := 2
noncomputable def H : ℕ := 3

-- Define the number ZAMENA
noncomputable def ZAMENA : ℕ :=
  5 * 10^5 + A * 10^4 + M * 10^3 + E * 10^2 + H * 10 + A

-- Define the inequalities and constraints
lemma satisfies_inequalities  : 3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Define uniqueness of each digit set
lemma unique_digits: A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ M ≠ E ∧ M ≠ H ∧ E ≠ H := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Prove that the number ZAMENA matches the specified number
theorem ZAMENA_correct : ZAMENA = 541234 := by
  unfold ZAMENA A M E H
  norm_num

end ZAMENA_correct_l100_100946


namespace angle_between_given_planes_is_zero_l100_100245

def angle_between_planes (a₁ b₁ c₁ d₁ a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  let n₁ := (a₁, b₁, c₁)
  let n₂ := (a₂, b₂, c₂)
  let dot_product := a₁ * a₂ + b₁ * b₂ + c₁ * c₂
  let magnitude_n₁ := Real.sqrt (a₁ ^ 2 + b₁ ^ 2 + c₁ ^ 2)
  let magnitude_n₂ := Real.sqrt (a₂ ^ 2 + b₂ ^ 2 + c₂ ^ 2)
  Real.arccos (dot_product / (magnitude_n₁ * magnitude_n₂))

theorem angle_between_given_planes_is_zero :
  angle_between_planes 6 2 (-4) 17 9 3 (-6) (-4) = 0 :=
by sorry

end angle_between_given_planes_is_zero_l100_100245


namespace find_YQ_l100_100900

noncomputable def EF := 9
noncomputable def FG := 15
noncomputable def VQ := 30
noncomputable def volume_ratio := 9
noncomputable def height_ratio := real.cbrt volume_ratio -- cube root of volume ratio
noncomputable def V'Q := VQ / height_ratio
noncomputable def V'Y := VQ - V'Q
noncomputable def E'F := EF / height_ratio
noncomputable def F'G := FG / height_ratio
noncomputable def distance_EF_sq := (E'F / 2) ^ 2 + (F'G / 2) ^ 2
noncomputable def hypothenuse := real.sqrt distance_EF_sq
noncomputable def centroid_height := hypothenuse / 2
noncomputable def YQ := centroid_height + V'Y

theorem find_YQ : 
  YQ = (real.sqrt ((E'F / 2) ^ 2 + (F'G / 2) ^ 2) / 2) + 20 := 
sorry

end find_YQ_l100_100900


namespace find_YB_l100_100331

-- Definitions of the geometric setup and given conditions.
variable (triangleXYZ : Type)
variable (X Y Z N D B : triangleXYZ)
variable (XY_len : ℝ)
variable (angleXYZ angleYXZ : ℝ)
variable (midpointN : Prop)
variable (lineYD_perp_YD : Prop)
variable (extension_B : Prop)

-- The core conditions translated to Lean definitions.
axiom angleXYZ_eq_60 : angleXYZ = 60
axiom angleYXZ_eq_30 : angleYXZ = 30
axiom XY_eq_1 : XY_len = 1
axiom N_midpoint_property : midpointN
axiom YD_perpendicular_to_ZN : lineYD_perp_YD
axiom extension_property : extension_B

-- The proof statement without proof.
theorem find_YB : YB = 1 ∧ YB = (2 - (0:ℝ)^0.5) / 2 ∧ (let p := 2, q := 0, r := 2 in p + q + r = 4) :=
by
  sorry

end find_YB_l100_100331


namespace boat_trip_time_with_current_l100_100518

-- Definitions based on conditions
def distance : ℝ := 96
def time_against : ℝ := 8
def boat_speed_still : ℝ := 15.6
def current_speed (x : ℝ) : ℝ := x
def trip_speed_against (x : ℝ) : ℝ := boat_speed_still - current_speed x
def trip_speed_with (x : ℝ) : ℝ := boat_speed_still + current_speed x

-- Main statement
theorem boat_trip_time_with_current (x : ℝ) (h1 : distance = trip_speed_against x * time_against) : 
  let T := distance / trip_speed_with x in
  T = 5 :=
by
  sorry

end boat_trip_time_with_current_l100_100518


namespace standard_form_ellipse_eqn_l100_100940

theorem standard_form_ellipse_eqn (center_vertex : (2, 0)) (focus : (1, 0)) :
  ∃ a b : ℝ, a = 2 ∧ b^2 = 3 ∧ (∀ x y, x^2 / 4 + y^2 / 3 = 1) :=
by
  use [2, sqrt 3]
  split
  exact rfl
  split
  repeat { sorry }

end standard_form_ellipse_eqn_l100_100940


namespace product_modulo_10_l100_100489

-- Define the numbers involved
def a := 2457
def b := 7623
def c := 91309

-- Define the modulo operation we're interested in
def modulo_10 (n : Nat) : Nat := n % 10

-- State the theorem we want to prove
theorem product_modulo_10 :
  modulo_10 (a * b * c) = 9 :=
sorry

end product_modulo_10_l100_100489


namespace tan_45_deg_eq_one_l100_100105

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100105


namespace max_value_quadratic_l100_100484

theorem max_value_quadratic : ∃ x : ℝ, -9 * x^2 + 27 * x + 15 = 35.25 :=
sorry

end max_value_quadratic_l100_100484


namespace interval_of_increase_slope_inequality_l100_100676

-- 1st problem: Interval of monotonic increase for f when m = 1
theorem interval_of_increase (m : Real) (hm : m = 1) : 
  ∀ x : Real, 0 < x ∧ x < 2 → monotone (fun x => 4 * log x - (1 / 2) * m * x^2)

-- 2nd problem: Prove the inequality for g
theorem slope_inequality (m x₁ x₂ x₀ : Real) (hm : m > 0) (hx₁x₂ : x₁ ≠ x₂) 
  (g : Real → Real := fun x => 4 * log x - (1 / 2) * m * x^2 + (4 - m) * x) 
  (k : Real) (hk : k = deriv g x₀) : x₁ + x₂ > 2 * x₀ := sorry

end interval_of_increase_slope_inequality_l100_100676


namespace tan_45_deg_eq_one_l100_100030

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100030


namespace gcd_840_1764_l100_100442

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l100_100442


namespace complex_quadrant_l100_100347

theorem complex_quadrant : 
  ∀ (z : ℂ), z = (1 + 3 * complex.I) / (3 + complex.I) → 
  z.re > 0 ∧ z.im > 0 :=
by
  intros z hz
  -- Proof steps would go here
  sorry

end complex_quadrant_l100_100347


namespace tan_45_eq_1_l100_100190

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100190


namespace distinct_points_distance_l100_100406

theorem distinct_points_distance (a b : ℝ) (h1 : a ≠ b)
  (h2 : (∃ y : ℝ, (y, a) = (Real.sqrt Real.exp 1, a)) ∧ 
        ∃ y : ℝ, (y, b) = (Real.sqrt Real.exp 1, b))
  (h3 : ∀ y : ℝ, y ^ 2 + (Real.sqrt Real.exp 1) ^ 6 = 3 * (Real.sqrt Real.exp 1) ^ 3 * y + 1) :
  |a - b| = Real.sqrt (5 * Real.exp 3 + 4) := 
sorry

end distinct_points_distance_l100_100406


namespace tan_45_deg_l100_100126

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100126


namespace tan_five_pi_over_four_l100_100582

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
sorry

end tan_five_pi_over_four_l100_100582


namespace tan_45_deg_l100_100138

theorem tan_45_deg : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_l100_100138


namespace nguyen_needs_pairs_l100_100402

-- Definitions based on the problem conditions
def feet_per_pair : ℝ := 8.5
def yards_to_feet (yards : ℝ) : ℝ := yards * 3
def fabric_initial_yards : ℝ := 3.5
def fabric_needed_feet : ℝ := 49
def total_fabric (initial_yards : ℝ) (needed_feet : ℝ) : ℝ := yards_to_feet(initial_yards) + needed_feet
def pairs_of_pants (total_fabric : ℝ) (feet_per_pair : ℝ) : ℝ := total_fabric / feet_per_pair

-- The theorem to be proved
theorem nguyen_needs_pairs :
  pairs_of_pants (total_fabric fabric_initial_yards fabric_needed_feet) feet_per_pair = 7 :=
begin
  sorry
end

end nguyen_needs_pairs_l100_100402


namespace max_value_M_l100_100889

def is_permutation (xs : List ℕ) : Prop :=
  xs.perm (List.range' 1 2004)

def M (xs : List ℕ) : ℕ :=
  (List.nthLe xs 0 sorry - List.nthLe xs 1 sorry).natAbs +
  (List.nthLe xs 2 sorry - List.nthLe xs 3 sorry).natAbs +
  -- This pattern continues for all necessary pairs
  (List.nthLe xs 2002 sorry - List.nthLe xs 2003 sorry).natAbs

theorem max_value_M : ∀ xs : List ℕ, is_permutation xs → M xs ≤ 2004 ∧ (∃ ys : List ℕ, is_permutation ys ∧ M ys = 2004) :=
by {
  -- no proof content, using sorry
  sorry
}

end max_value_M_l100_100889


namespace inequality_proof_l100_100387

theorem inequality_proof {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) :
  (1 / Real.sqrt (1 + x^2)) + (1 / Real.sqrt (1 + y^2)) ≤ (2 / Real.sqrt (1 + x * y)) :=
by
  sorry

end inequality_proof_l100_100387


namespace first_player_wins_with_optimal_play_l100_100216

-- Define the conditions (structure of the equations)
def equations := 
  [λ x : ℕ, x = x,
   λ x y : ℕ, x = y + x,
   λ x y z : ℕ, x = y + z + x,
   λ x y z w : ℕ, x = y + z + w + x,
   λ x y z w v : ℕ, x = y + z + w + v + x,
   λ x y z w v u : ℕ, x = y + z + w + v + u + x]

-- Theorem statement: Prove that the first player wins with the correct strategy
theorem first_player_wins_with_optimal_play (e : equations) : 
  ∃ strategy : list ℕ → list ℕ, (∀ eq : ℕ, eq ∈ e → true) ∧ first_player_wins :=
sorry

end first_player_wins_with_optimal_play_l100_100216


namespace tan_45_eq_1_l100_100189

theorem tan_45_eq_1 (hcos : real.cos (real.pi / 4) = real.sqrt 2 / 2) (hsin : real.sin (real.pi / 4) = real.sqrt 2 / 2) : real.tan (real.pi / 4) = 1 :=
by
  have h : real.tan (real.pi / 4) = real.sin (real.pi / 4) / real.cos (real.pi / 4) := real.tan_eq_sin_div_cos _
  rw [hsin, hcos] at h
  simp at h
  exact h.symm

end tan_45_eq_1_l100_100189


namespace tan_45_deg_l100_100063

theorem tan_45_deg : (Real.tan (Float.pi / 4)) = 1 :=
by
  sorry

end tan_45_deg_l100_100063


namespace polygon_interior_angles_540_l100_100327

theorem polygon_interior_angles_540 (h : ∑ (i : Fin 1), 180 * (5 - 2) = 540) : 5 = 5 :=
by
  sorry

end polygon_interior_angles_540_l100_100327


namespace overall_effect_l100_100871
noncomputable def effect (x : ℚ) : ℚ :=
  ((x * (5 / 6)) * (1 / 10)) + (2 / 3)

theorem overall_effect (x : ℚ) : effect x = (x * (5 / 6) * (1 / 10)) + (2 / 3) :=
  by
  sorry

-- Prove for initial number 1
example : effect 1 = 3 / 4 :=
  by
  sorry

end overall_effect_l100_100871


namespace four_digits_right_decimal_l100_100475

noncomputable def first_four_digits (x : ℝ) : ℕ :=
  floor (x * 10000) % 10000

theorem four_digits_right_decimal :
  first_four_digits ((10 ^ 2000 + 1 : ℝ) ^ (11 / 7)) = 7140 :=
by
  -- lets use sorry to skip the actual proof
  sorry

end four_digits_right_decimal_l100_100475


namespace codomain_of_sqrt_one_minus_x_squared_l100_100212

theorem codomain_of_sqrt_one_minus_x_squared : 
  ∀ y ∈ set.range (λ x : ℝ, sqrt (1 - x^2)), y >= 0 ∧ y <= 1 :=
sorry

end codomain_of_sqrt_one_minus_x_squared_l100_100212


namespace shortest_chord_through_M_is_x_plus_y_minus_1_eq_0_l100_100291

noncomputable def circle_C : Set (ℝ × ℝ) := { p | (p.1^2 + p.2^2 - 4*p.1 - 2*p.2) = 0 }

def point_M_in_circle : Prop :=
  (1, 0) ∈ circle_C

theorem shortest_chord_through_M_is_x_plus_y_minus_1_eq_0 :
  point_M_in_circle →
  ∃ (a b c : ℝ), a * 1 + b * 0 + c = 0 ∧
  ∀ (x y : ℝ), (a * x + b * y + c = 0) → (x + y - 1 = 0) :=
by
  sorry

end shortest_chord_through_M_is_x_plus_y_minus_1_eq_0_l100_100291


namespace triangle_area_DEF_l100_100866

section
variable (D : ℝ × ℝ) (E : ℝ × ℝ) (F : ℝ × ℝ)
variable (hD : D = (2, 1))
variable (hE : E = (1, 4))
variable (hF : F.1 + F.2 = 6)

theorem triangle_area_DEF : 
  let area := (1/2) * abs (D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2))
  in area = 6 := 
sorry

end

end triangle_area_DEF_l100_100866


namespace john_spends_on_burritos_l100_100769

theorem john_spends_on_burritos (burritos_num calories_per_burrito burgers_num calories_per_burger burger_cost calories_more_per_dollar : ℕ) 
    (h_burritos : burritos_num = 10)
    (h_calories_per_burrito : calories_per_burrito = 120)
    (h_burgers : burgers_num = 5)
    (h_calories_per_burger : calories_per_burger = 400)
    (h_burger_cost : burger_cost = 8)
    (h_calories_more_per_dollar : calories_more_per_dollar = 50) : 
    (burritos_num * calories_per_burrito) / ((burgers_num * calories_per_burger) / burger_cost - calories_more_per_dollar) = 6 := 
by 
  rw [h_burritos, h_calories_per_burrito, h_burgers, h_calories_per_burger, h_burger_cost, h_calories_more_per_dollar]
  norm_num

end john_spends_on_burritos_l100_100769


namespace polynomial_remainder_l100_100631

open Polynomials

theorem polynomial_remainder :
  let P := λ x : ℝ, 5 * x ^ 4 - 13 * x ^ 3 + 3 * x ^ 2 - x + 15 in
  P 3 = 93 := by
  sorry

end polynomial_remainder_l100_100631


namespace option_b_option_c_option_d_l100_100266

section perfect_sets

def perfect_set (A : Set ℝ) := A.Nonempty ∧ ∃ (s : ℝ) (p : ℝ), s = p ∧ 
  ∃ a, a ∈ A ∧ ∀ b, b ∈ A → b = a ∨ b + a = p ∧ (∀ c, c ∈ A → c = a ∨ c * a = p)

theorem option_b (a1 a2 : ℝ) (h : a1 ≠ a2) (h1 : a1 > 0) (h2 : a2 > 0) (hs : perfect_set {a1, a2}) :
  a1 > 2 ∨ a2 > 2 := sorry

theorem option_c : ∃ S : Set (Set ℝ), S.Infinite ∧ ∀ s ∈ S, perfect_set s ∧ s.card = 2 := sorry

theorem option_d (n : ℕ) (A : Set ℕ) (h : n ≥ 2 ∧ ∀ x ∈ A, x > 0 ∧ ∃ s p, s = p ∧ 
  ∃ a, a ∈ A ∧ ∀ b, b ∈ A → b = a ∨ b + a = p ∧ (∀ c, c ∈ A → c = a ∨ c * a = p)) :
  ∃! n, n = 3 := sorry

end perfect_sets

end option_b_option_c_option_d_l100_100266


namespace rounding_and_scientific_notation_l100_100579

def number : ℝ := 0.000359
def significant_figures : ℕ := 2
def scientific_notation (x : ℝ) (n : ℤ) : Prop := x = 3.6 ∧ n = -4

theorem rounding_and_scientific_notation :
  ∃ (x : ℝ) (n : ℤ), x × 10 ^ n = number ∧ scientific_notation x n :=
by
  use 3.6
  use -4
  constructor
  · unfold number
    norm_num
  · sorry

end rounding_and_scientific_notation_l100_100579


namespace water_tank_capacity_l100_100535

theorem water_tank_capacity (full_capacity : ℕ) 
  (h1 : full_capacity * 0.4 = (full_capacity * 0.9 - 36)) : 
  full_capacity = 72 := 
by
  sorry

end water_tank_capacity_l100_100535


namespace parabola_equation_l100_100839

-- Condition 1: The parabola passes through the point (2,7)
def passes_through (a b : ℝ) (x y : ℝ) : Prop := (x, y) = (a, b)

-- Condition 2: The y-coordinate of the focus is 6
def focus_y_coordinate (focus_y : ℝ) : Prop := focus_y = 6

-- Condition 3: The axis of symmetry is parallel to the x-axis
def axis_of_symmetry_parallel_x (parallel : bool) : Prop := parallel = true

-- Condition 4: The vertex lies on the y-axis
def vertex_on_y_axis (vertex_x : ℝ) : Prop := vertex_x = 0

-- To ensure the coefficients are integers and other constraints
def gcd_condition (a b c d e f : ℤ) : Prop :=
  c > 0 ∧ Int.gcd [|a|, |b|, |c|, |d|, |e|, |f|] = 1

-- Main proof statement
theorem parabola_equation
  (hx : passes_through 2 7 x y)
  (hy : focus_y_coordinate focus_y)
  (hz : axis_of_symmetry_parallel_x true)
  (hw : vertex_on_y_axis vx)
  (gcd_cond : gcd_condition 0 0 2 (-1) (-24) 72)
  : 2 * y ^ 2 - x - 24 * y + 72 = 0 := 
sorry

end parabola_equation_l100_100839


namespace socorro_training_days_l100_100422

variable (total_training_time_per_day : ℕ) (total_training_time : ℕ)

theorem socorro_training_days (h1 : total_training_time = 300) 
                              (h2 : total_training_time_per_day = 30) :
                              total_training_time / total_training_time_per_day = 10 := 
begin
  rw [h1, h2],
  norm_num,
end

end socorro_training_days_l100_100422


namespace distinct_integers_count_l100_100553

theorem distinct_integers_count : 
  (Finset.card (Finset.image (fun n => Int.floor ((n : ℕ)^2 / 500)) (Finset.range 1000))) = 2001 := 
sorry

end distinct_integers_count_l100_100553


namespace find_angle_between_vectors_l100_100643

variables {ℝ} (a b : euclidean_space ℝ (fin 2))
hypothesis h1 : euclidean_space.norm a = 1
hypothesis h2 : euclidean_space.norm b = sqrt 2
hypothesis h3 : inner_product_space.inner a (a - b) = 0

theorem find_angle_between_vectors : 
  let θ := real.angle_of_vectors a b in θ = real.pi / 4 :=
begin
  sorry
end

end find_angle_between_vectors_l100_100643


namespace bug_path_probability_l100_100519

open ProbabilityTheory

-- Definitions of the conditions
def start_vertex := V -- V represents the starting vertex.
def move_along_edges (v : V) : V := sorry -- Function describing movement along edges
def equal_probability (e1 e2 e3 : E) : Prop := e1.probability = e2.probability ∧ e2.probability = e3.probability
def independent_choices (choices : List E) : Prop := ∀ i, indep choice i
def eight_moves (moves : List Move) : Prop := moves.length = 8
def returns_to_start (moves : List Move) (start : V) : Prop := (List.last moves).end_vertex = start
def exactly_twice (moves : List Move) (vertices : List V) : Prop := ∀ v, count v vertices = 2

-- This probabilistic event
def valid_path (moves : List Move) (start : V) : Prop :=
  eight_moves moves ∧
  returns_to_start moves start ∧
  ∃ (v1 v2 : V), exactly_twice moves [v1, v2] ∧
  (∀ v ≠ v1 ∧ v ≠ v2 → count v moves ≤ 1)

-- The probability that after eight moves the bug will return 
-- to its starting vertex while visiting exactly two other vertices exactly twice 
-- and the remaining vertices not more than once
theorem bug_path_probability :
  ∃ p : ℚ, p = (1 / 729) ∧ valid_path moves start_vertex := 
begin
  sorry
end

end bug_path_probability_l100_100519


namespace specific_value_of_n_l100_100640

theorem specific_value_of_n (n : ℕ) 
  (A_n : ℕ → ℕ)
  (C_n : ℕ → ℕ → ℕ)
  (h1 : A_n n ^ 2 = C_n n (n-3)) :
  n = 8 :=
sorry

end specific_value_of_n_l100_100640


namespace tan_5pi_over_4_l100_100601

theorem tan_5pi_over_4 : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end tan_5pi_over_4_l100_100601


namespace minimum_value_of_a_plus_b_l100_100681

-- Conditions 1: Definitions for a and b and given the equation involving them.
def line_through_point (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : Prop :=
  (3 / a + 2 / b = 1)

-- Theorem: Minimum value of a + b given the conditions.
theorem minimum_value_of_a_plus_b : 
  ∀ a b : ℝ, a > 0 → b > 0 → line_through_point a b (by assumption) (by assumption) -> a + b = 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_of_a_plus_b_l100_100681


namespace correct_sentence_given_uncountable_nouns_l100_100465

/-- Given the translation of the Chinese abstract noun "消息" as "news" or "information" and their 
     nature as uncountable nouns in English, prove that the sentence 
     "They sent us word of the latest happenings" is grammatically correct without using any article. -/
theorem correct_sentence_given_uncountable_nouns (translation : String)
  (h_translation : translation = "news" ∨ translation = "information")
  (uncountable : ∀ t, t = "news" ∨ t = "information" → ¬ (∃ a, a = "a" ∨ a = "the")) :
  "They sent us word of the latest happenings" :=
  sorry

end correct_sentence_given_uncountable_nouns_l100_100465


namespace hyperbola_asymptotes_and_point_l100_100663

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 8 - y^2 / 2 = 1

theorem hyperbola_asymptotes_and_point 
  (x y : ℝ)
  (asymptote1 : ∀ x, y = (1/2) * x)
  (asymptote2 : ∀ x, y = (-1/2) * x)
  (point : (x, y) = (4, Real.sqrt 2))
: hyperbola_equation x y :=
sorry

end hyperbola_asymptotes_and_point_l100_100663


namespace quadratic_root_value_l100_100324

theorem quadratic_root_value (a b : ℤ) (h : 2 * a - b = -3) : 6 * a - 3 * b + 6 = -3 :=
by 
  sorry

end quadratic_root_value_l100_100324


namespace zamena_solution_l100_100972

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end zamena_solution_l100_100972


namespace prove_ZAMENA_l100_100960

theorem prove_ZAMENA :
  ∃ (A M E H Z : ℕ),
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ Z ∧ Z ≤ 5 ∧
    (3 > A) ∧ (A > M) ∧ (M < E) ∧ (E < H) ∧ (H < A) ∧
    (A ≠ M) ∧ (A ≠ E) ∧ (A ≠ H) ∧ (A ≠ Z) ∧
    (M ≠ E) ∧ (M ≠ H) ∧ (M ≠ Z) ∧
    (E ≠ H) ∧ (E ≠ Z) ∧
    (H ≠ Z) ∧
    Z = 5 ∧
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234 :=
begin
  sorry
end

end prove_ZAMENA_l100_100960


namespace infinite_multiples_of_7_exists_l100_100566

def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ ∀ n > 1, a n = a (n - 1) + a (n / 2)

theorem infinite_multiples_of_7_exists (a : ℕ → ℕ) (h_seq : sequence a) : 
  ∃ᶠ n in at_top, 7 ∣ a n :=
sorry

end infinite_multiples_of_7_exists_l100_100566


namespace tan_45_deg_eq_one_l100_100102

/-
  Given:
  cos (45 deg) = sqrt 2 / 2,
  sin (45 deg) = sqrt 2 / 2,
  Prove: 
  tan (45 deg) = 1
-/

theorem tan_45_deg_eq_one : 
  (real.cos (real.pi / 4) = real.sqrt 2 / 2) → 
  (real.sin (real.pi / 4) = real.sqrt 2 / 2) → 
  real.tan (real.pi / 4) = 1 :=
by
  intros hcos hsin
  sorry

end tan_45_deg_eq_one_l100_100102


namespace tan_45_deg_eq_one_l100_100028

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l100_100028


namespace function_below_x_axis_l100_100254

theorem function_below_x_axis (k : ℝ) :
  (∀ x : ℝ, (k^2 - k - 2) * x^2 - (k - 2) * x - 1 < 0) ↔ (-2 / 5 < k ∧ k ≤ 2) :=
by
  sorry

end function_below_x_axis_l100_100254


namespace probability_of_experts_winning_l100_100230

-- Definitions required from the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p
def current_expert_score : ℕ := 3
def current_audience_score : ℕ := 4

-- The main theorem to state
theorem probability_of_experts_winning : 
  p^4 + 4 * p^3 * q = 0.4752 := 
by sorry

end probability_of_experts_winning_l100_100230


namespace bisection_method_exists_x_l100_100884

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem bisection_method_exists_x :
  ∃ x ∈ Icc 1 1.5, |f x| ≤ 0.001 :=
sorry

end bisection_method_exists_x_l100_100884


namespace division_science_notation_l100_100554

theorem division_science_notation :
  (16.036 * 10^3) / (3.672 * 10^(-2)) ≈ 436621.978022 := by
sorry

end division_science_notation_l100_100554


namespace find_m_l100_100280

theorem find_m (m : ℝ) : 
  (m + 4 ≠ 0) →
  (∀ x : ℝ, ((2 / 3) * (m + 4) * x^|m - 3| + 6 > 0) → ((m ≠ 0) → |m| - 3 = 1)) → 
  m = 4 := 
by 
  intros h1 h2 h3
  sorry

end find_m_l100_100280


namespace statue_original_cost_l100_100363

noncomputable def original_cost (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  selling_price / (1 + profit_rate)

theorem statue_original_cost :
  original_cost 660 0.20 = 550 := 
by
  sorry

end statue_original_cost_l100_100363


namespace intersection_complement_A_l100_100690
-- Import necessary library

-- Definitions for conditions
def U := ℝ
def A := { x : ℝ | 2 < x ∧ x ≤ 4 }
def B := { 3, 4 }

-- Complement of B in U
def complement_U_B := { x : ℝ | x ≠ 3 ∧ x ≠ 4 }

-- The statement to prove
theorem intersection_complement_A (U : Type) [topological_space U] :
  A ∩ complement_U_B = (set.Ioo (2 : ℝ) (3) ∪ set.Ioo (3) (4)) :=
by sorry

end intersection_complement_A_l100_100690


namespace fifth_term_of_sequence_eq_121_l100_100561

theorem fifth_term_of_sequence_eq_121 :
  let seq (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), 3 ^ i
  seq 4 = 121 :=
by
  sorry

end fifth_term_of_sequence_eq_121_l100_100561


namespace prove_p_and_q_l100_100979

variable (x_0 : ℝ) (m x y : ℝ)

def p : Prop := ∃ x_0 > 0, real.log x_0 = -1

-- Represents the equation being an ellipse with foci on the x-axis if m > 1
def q : Prop := ∀ m > 1, ∀ x y, x^2 + m * y^2 = 1 → m ≠ 0 ∧ (0 < 1 / m < 1)

theorem prove_p_and_q : p ∧ q := by
  sorry

end prove_p_and_q_l100_100979


namespace polar_equations_and_intersection_product_find_OP_OQ_product_l100_100743

open Real

-- Definitions of the given parametric equations and line equation
def parametric_eq_C1 (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 + 2 * cos α, 2 + 2 * sin α)

def eq_line_C2 (x : ℝ) : ℝ :=
  (sqrt 3 / 3) * x

-- The proof problem statement
theorem polar_equations_and_intersection_product :
  (∀ (x y : ℝ), (x = sqrt 3 + 2 * cos (atan2 y x) ∧ y = 2 + 2 * sin (atan2 y x)) ↔
    ((x - sqrt 3)^2 + (y - 2)^2 = 4 ∧ (y = sqrt 3 / 3 * x))):=
  sorry

theorem find_OP_OQ_product :
  (∀ (P Q : ℝ × ℝ), (P = (sqrt 3 / 3, 3)) ∧ (Q = (sqrt 3 / 3, 3)) → 
   (|P.right| * |Q.right| = 3)) :=
  sorry

end polar_equations_and_intersection_product_find_OP_OQ_product_l100_100743


namespace distance_f_to_midpoint_de_l100_100344

/-
Given a right triangle DEF with sides DE = 15, DF = 20, and hypotenuse EF = 25,
prove that the distance from F to the midpoint of DE is 12.5 units.
-/

theorem distance_f_to_midpoint_de :
  ∀ (D E F : Point) (DE DF EF : ℝ),
  is_right_triangle D E F →
  (segment_length D E = 15) →
  (segment_length D F = 20) →
  (segment_length E F = 25) →
  distance F (midpoint D E) = 12.5 :=
by
  intros D E F DE DF EF h_right_triangle h_de h_df h_ef
  sorry

end distance_f_to_midpoint_de_l100_100344


namespace zamena_inequalities_l100_100966

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end zamena_inequalities_l100_100966


namespace find_sin_A_l100_100343

-- Definitions and constants
variables {A B C : ℝ}
variables {sin cos tan : ℝ → ℝ}
variable (h ∷ ∀ {θ : ℝ}, tan θ = sin θ / cos θ)

-- Given conditions
def isRightTriangle (A B C : ℝ) : Prop :=
  A^2 + B^2 = C^2

def triangleABC : Prop :=
  isRightTriangle (BC : ℝ) (AB : ℝ) (AC : ℝ)

def equation (A : ℝ) (sin cos tan : ℝ → ℝ) : Prop :=
  3 * sin A = 4 * cos A + tan A

-- Final theorem
theorem find_sin_A (A : ℝ) (sin cos tan : ℝ → ℝ) [triangleABC] [equation A sin cos tan] : Prop :=
  sin A = (2 * Real.sqrt 2) / 3 :=
sorry

end find_sin_A_l100_100343
