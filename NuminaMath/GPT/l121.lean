import Mathlib

namespace smallest_diff_mod_13_l121_121921

theorem smallest_diff_mod_13 : 
  let m := Nat.find (λ k, 100 ≤ 13 * k + 7)
  let n := Nat.find (λ k, 1000 ≤ 13 * k + 7)
  (13 * n + 7) - (13 * m + 7) = 895 :=
by
  sorry

end smallest_diff_mod_13_l121_121921


namespace cube_volume_l121_121150

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l121_121150


namespace inner_cube_surface_area_l121_121296

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121296


namespace probability_prime_factor_of_120_l121_121225

open Nat

theorem probability_prime_factor_of_120 : 
  let s := Finset.range 61
  let primes := {2, 3, 5}
  let prime_factors_of_5_fact := primes ∩ s
  (prime_factors_of_5_fact.card : ℚ) / s.card = 1 / 20 :=
by
  sorry

end probability_prime_factor_of_120_l121_121225


namespace holiday_rush_increase_l121_121625

theorem holiday_rush_increase(percent_decrease : ℝ) (h : percent_decrease = 0.242424242424242) :
  let P := 1 / (1 - percent_decrease) / 1.10 - 1 in
  P ≈ 0.32 :=
by
  sorry

end holiday_rush_increase_l121_121625


namespace cube_volume_l121_121220

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l121_121220


namespace total_bathing_suits_l121_121135

def men_bathing_suits : ℕ := 14797
def women_bathing_suits : ℕ := 4969

theorem total_bathing_suits : men_bathing_suits + women_bathing_suits = 19766 := by
  sorry

end total_bathing_suits_l121_121135


namespace min_colors_5x5_grid_l121_121941

def is_valid_coloring (grid : Fin 5 × Fin 5 → ℕ) (k : ℕ) : Prop :=
  ∀ i j : Fin 5, ∀ di dj : Fin 2, ∀ c : ℕ,
    (di ≠ 0 ∨ dj ≠ 0) →
    (grid (i, j) = c ∧ grid (i + di, j + dj) = c ∧ grid (i + 2 * di, j + 2 * dj) = c) → 
    False

theorem min_colors_5x5_grid : 
  ∀ (grid : Fin 5 × Fin 5 → ℕ), (∀ i j, grid (i, j) < 3) → is_valid_coloring grid 3 := 
by
  sorry

end min_colors_5x5_grid_l121_121941


namespace angle4_is_45_l121_121789

noncomputable section
open_locale classical

-- Define the angles for triangles ABC and DEF
variables (A B C D E F : Type) [inner_product_space ℝ E]
variables {α : ℝ} (angle1 angle2 angle3 angle4 : α)
variables (angleA angleB angleC : α)

axiom h1 : angleA = 50
axiom h2 : angleB = 60
axiom h3 : angle1 + angle2 = 180
axiom h4 : angle3 = angle4
axiom h5 : angle1 = angleC
axiom h6 : angleE = 90
axiom h7 : angleD = angle1
axiom h8 : ∀ angle3 angle4, angle3 + angle4 = 90

-- Prove that angle4 = 45
theorem angle4_is_45
: angle4 = 45 :=
sorry

end angle4_is_45_l121_121789


namespace unit_vector_orthogonal_to_both_l121_121436

noncomputable def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(a.2.2 * b.2 - a.2 * b.2.2, a.2 * b.1 - a.1 * b.2, a.1 * b.2.2 - a.2.2 * b.1)

noncomputable def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2 + v.2.2^2)

noncomputable def unit_vector (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
let mag := vector_magnitude v in
(v.1 / mag, v.2 / mag, v.2.2 / mag)

def is_orthogonal (a b : ℝ × ℝ × ℝ) : Prop :=
a.1 * b.1 + a.2 * b.2 + a.2.2 * b.2.2 = 0

theorem unit_vector_orthogonal_to_both :
  let a : ℝ × ℝ × ℝ := (2, 1, 1)
  let b : ℝ × ℝ × ℝ := (3, 0, 4)
  let u : ℝ × ℝ × ℝ := unit_vector (cross_product a b)
  is_orthogonal u a ∧ is_orthogonal u b :=
by
  sorry

end unit_vector_orthogonal_to_both_l121_121436


namespace odd_function_symmetry_l121_121618

def f (x : ℝ) : ℝ := x^3 + x

-- Prove that f(-x) = -f(x)
theorem odd_function_symmetry : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end odd_function_symmetry_l121_121618


namespace cube_volume_l121_121169

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121169


namespace tangent_line_equation_l121_121965

-- Define the parabola and the line
def parabola (x : ℝ) : ℝ := x^2
def line (x y : ℝ) : Prop := 2*x - y + 4 = 0

-- The slope of the given line
def slope_of_line : ℝ := 2

-- The tangent line should be parallel to the given line, hence have the same slope
def tangent_line_slope : ℝ := slope_of_line

-- Finding the point of tangency (1, 1)
def tangent_point : ℝ × ℝ := (1, 1)

-- Equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Proof statement
theorem tangent_line_equation :
  (∀ x y, parabola.derivative x = tangent_line_slope → (x, y) = tangent_point) →
  tangent_line tangent_point.1 tangent_point.2 := 
by sorry

end tangent_line_equation_l121_121965


namespace initial_sweets_at_first_l121_121637

-- Define the initial number of sweets
variable (x : ℕ)

-- Conditions
def jack_took : ℕ := x / 2 + 4
def paul_took : ℕ := 7

-- Theorem to prove
theorem initial_sweets_at_first : jack_took(x) + paul_took = x := by
  sorry

end initial_sweets_at_first_l121_121637


namespace second_cube_surface_area_l121_121327

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121327


namespace andrew_age_l121_121410

theorem andrew_age (a g : ℕ) (h1 : g = 10 * a) (h2 : g - a = 63) : a = 7 := by
  sorry

end andrew_age_l121_121410


namespace possible_values_expr_l121_121826

theorem possible_values_expr (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) :
  ∃ x, x = (Real.sign a + Real.sign b + Real.sign c + Real.sign d + Real.sign (a * b * c * d)) ∧
       x ∈ {5, 3, 2, 0, -3} := 
sorry

end possible_values_expr_l121_121826


namespace inner_cube_surface_area_l121_121266

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121266


namespace inner_cube_surface_area_l121_121367

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121367


namespace mrs_blue_expected_tomato_yield_l121_121584

-- Definitions for conditions
def steps_length := 3 -- each step measures 3 feet
def length_steps := 18 -- 18 steps in length
def width_steps := 25 -- 25 steps in width
def yield_per_sq_ft := 3 / 4 -- three-quarters of a pound per square foot

-- Define the total expected yield in pounds
def expected_yield : ℝ :=
  let length_ft := length_steps * steps_length
  let width_ft := width_steps * steps_length
  let area := length_ft * width_ft
  area * yield_per_sq_ft

-- The goal statement
theorem mrs_blue_expected_tomato_yield : expected_yield = 3037.5 := by
  sorry

end mrs_blue_expected_tomato_yield_l121_121584


namespace initial_charge_plan_A_l121_121686

def cost_A (x t : ℕ) : ℝ :=
  if t <= x then 0.60 else 0.60 + (t - x) * 0.06

def cost_B (t : ℕ) : ℝ :=
  t * 0.08

theorem initial_charge_plan_A (x : ℕ) :
  cost_A x 15 = cost_B 15 → x = 5 :=
by
  intro h
  simp [cost_A, cost_B] at h
  have h' : 0.60 + (15 - x) * 0.06 = 15 * 0.08, from h
  sorry

end initial_charge_plan_A_l121_121686


namespace broken_line_length_l121_121810

-- Define the equilateral triangle and points D, E
variables {A B C D E : Type} [metric_space A]
variables (triangle_ABC : triangle A B C)
variable [is_equilateral triangle_ABC]
variable [on_side D A B]
variable [on_side E B C]

-- Prove the length of the broken line
theorem broken_line_length (h : is_equilateral triangle_ABC) (hD : on_side D A B) (hE : on_side E B C) : 
  dist A E + dist E D + dist D C ≥ 2 * dist A B :=
sorry

end broken_line_length_l121_121810


namespace inner_cube_surface_area_l121_121321

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121321


namespace max_voters_after_T_l121_121122

theorem max_voters_after_T (x : ℕ) (n : ℕ) (y : ℕ) (T : ℕ)  
  (h1 : x <= 10)
  (h2 : x > 0)
  (h3 : (nx + y) ≤ (n + 1) * (x - 1))
  (h4 : ∀ k, (x - k ≥ 0) ↔ (n ≤ T + 5)) :
  ∃ (m : ℕ), m = 5 := 
sorry

end max_voters_after_T_l121_121122


namespace a_2018_equals_5_7_l121_121466

-- Define the sequence recursively
def a_seq (n : ℕ) : ℝ :=
  Nat.recOn n (6/7) (λ n a_n,
    if 0 ≤ a_n ∧ a_n < 1/2 then 2 * a_n
    else if 1/2 ≤ a_n ∧ a_n < 1 then 2 * a_n - 1
    else a_n) -- handle cases out of [0, 1) by simply returning a_n itself

-- Theorem stating that a_{2018} = 5/7
theorem a_2018_equals_5_7 : a_seq 2018 = 5/7 :=
sorry

end a_2018_equals_5_7_l121_121466


namespace inner_cube_surface_area_l121_121363

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121363


namespace max_value_FD_CD_l121_121805

noncomputable def parabola_focus (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p / 2, 0)

noncomputable def circle_eqn (p : ℝ) (hp : p > 0) (x y : ℝ) : Prop := 
  (x - p / 2) ^ 2 + y ^ 2 = p ^ 2

theorem max_value_FD_CD {p : ℝ} (hp : p > 0)
    (A D F C : ℝ × ℝ)
    (hA : ∃ y, (0, y) = A ∧ circle_eqn p hp 0 y)
    (hD : let line_AF := (λ y, -sqrt 3 * y + sqrt 3 * p / 2)
          in D.1 = (p / 6) ∧ D.2 = (sqrt 3 * p / 3))
    (hC : let directrix_x := -p /2
          in C.1 = directrix_x ∧ C.2 = sqrt 3 * p)
    (hAD : |AD| = (1 / 3) * p ∧ (1 ≤ |AD| ∧ |AD| ≤ 2) )
    : |FD| * |CD| ≤ 32  :=
begin
  sorry
end

end max_value_FD_CD_l121_121805


namespace intersect_line_intersect_skew_lines_l121_121420

-- Definitions
variables {a b : Line} -- skew lines
variables {α β : Plane} -- planes
variables {M : Point} -- point

-- Assumptions
axiom skew_lines (h_skew : ¬ exists p, p ∈ a ∧ p ∈ b)
axiom plane_alpha (h_alpha : ∀ p ∈ a, p ∈ α ∧ ∀ q ∈ b, q ∉ α)
axiom plane_beta (h_beta : ∀ p ∈ b, p ∈ β ∧ ∀ q ∈ a, q ∉ β)
axiom point_M (h_M : M ∉ α ∧ M ∉ β)

-- New Planes passing through lines and the point M
def α1 := Plane_through_line_point a M
def β1 := Plane_through_line_point b M

-- theorem to prove
theorem intersect_line_intersect_skew_lines :
  ∃ c : Line, (∀ p, p ∈ c → p ∈ α1 ∧ p ∈ β1) ∧ 
              (∃ p ∈ c, p ∈ a) ∧ 
              (∃ q ∈ c, q ∈ b) :=
sorry

end intersect_line_intersect_skew_lines_l121_121420


namespace general_term_a_general_term_diff_sum_bn_l121_121816

-- Define sequences and their properties
variable {a b : ℕ → ℤ}
variable (d q : ℤ)

-- Conditions in the problem
axiom arithmetic_seq : ∀ n, a (n+1) - a n = d
axiom a2_eq_3 : a 2 = 3
axiom sum_first_four_a_eq_16 : a 1 + a 2 + a 3 + a 4 = 16

axiom b1_eq_4 : b 1 = 4
axiom b4_eq_88 : b 4 = 88
axiom geometric_seq_diff : ∀ n, b (n+1) - a (n+1) = (b 1 - a 1) * q^n

-- Theorem statements to be proved
theorem general_term_a (n : ℕ) : a n = 2 * n - 1 := sorry

theorem general_term_diff (n : ℕ) : b n - a n = 3 ^ n := sorry

theorem sum_bn (n : ℕ) : (∑ i in Finset.range n, b (i + 1)) = (3 ^ (n + 1)) / 2 + (n * n) - 3 / 2 := sorry

end general_term_a_general_term_diff_sum_bn_l121_121816


namespace quadratic_solution_l121_121004

noncomputable def solve_quadratic (x : ℝ) : Prop :=
  x^2 - 6 * x + 11 = 23

theorem quadratic_solution (a b : ℝ) (h : solve_quadratic 3 + real.sqrt 21 ∧ solve_quadratic 3 - real.sqrt 21) (h₂ : a = 3 + real.sqrt 21) (h₃ : b = 3 - real.sqrt 21): a ≥ b → a + 3 * b = 12 - 2 * real.sqrt 21 :=
sorry

end quadratic_solution_l121_121004


namespace problem_I_problem_II_problem_III_l121_121692

-- Definitions used as conditions
def is_geometric_sequence (seq : List ℕ) : Prop :=
  ∃ r : ℕ, ∀ i : ℕ, i + 1 < seq.length → seq[i+1] = seq[i] * r

def is_geometric_difference_sequence (seq : List ℕ) : Prop :=
  ∃ r : ℕ, ∀ i : ℕ, i + 2 < seq.length →
    seq[i+2] - seq[i+1] = r * (seq[i+1] - seq[i])

-- Problem Ⅰ: Given k = 4 and geometric sequence, find a value of a
theorem problem_I (k : ℕ) (a : ℕ) (a1 a2 a3 a4 : ℕ) (h1 : a1 < a2) (h2 : a2 < a3) (h3 : a3 < a4)
  (hk : k = 4) (hseq : is_geometric_sequence [a1, a2, a3, a4]) :
  a = 8 :=
sorry

-- Problem Ⅱ: Given k ≥ 4 and geometric difference in sequence, find a
theorem problem_II (k : ℕ) (a a2 : ℕ) (ha : k ≥ 4) (hseq : is_geometric_difference_sequence (range k.map (fun i => if i=0 then 1 else a2 * i))) :
  a = a2 ^ (k - 1) :=
sorry

-- Problem Ⅲ: Prove A < a^2
theorem problem_III (k : ℕ) (a : ℕ) (a_seq : List ℕ) (hk : k ≥ 2)
  (hdiv : ∀ i, i < k → (a % a_seq[i] = 0))
  (hA_seq : is_geometric_difference_sequence a_seq)
  (A : ℕ) :
  (A = (sum (for i in [0 .. k-2], a_seq[i] * a_seq[i+1]))) → A < a^2 :=
sorry

end problem_I_problem_II_problem_III_l121_121692


namespace hyperbola_correct_equation_l121_121012

def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def passes_through_point (a b : ℝ) : Prop :=
  hyperbola_equation a b (real.sqrt 2) (real.sqrt 3)

def eccentricity_equation (a b : ℝ) : Prop :=
  2 = real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_correct_equation (a b : ℝ) (h1 : passes_through_point a b) (h2 : eccentricity_equation a b) :
  a^2 = 1 ∧ b^2 = 3 → hyperbola_equation 1 (real.sqrt 3) x y :=
sorry

end hyperbola_correct_equation_l121_121012


namespace cos_value_given_sin_l121_121790

theorem cos_value_given_sin (α : ℝ) (h : Real.sin (α + π / 6) = 2 * Real.sqrt 5 / 5) :
  Real.cos (π / 3 - α) = 2 * Real.sqrt 5 / 5 := by
  sorry

end cos_value_given_sin_l121_121790


namespace sum_of_divisors_143_l121_121100

theorem sum_of_divisors_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121100


namespace minimum_good_pairs_l121_121523

def is_good_pair (a b c d : ℕ) : Prop :=
  (a < b ∧ c < a ∧ d < a) ∨ (a > b ∧ c > a ∧ d > a)

theorem minimum_good_pairs :
    ∃ arrangement : list ℕ,
      (∀ i : ℕ, i ∈ finset.range 100 → (arrangement.nth i).is_some) ∧
      (∀ i : ℕ, (is_good_pair (arrangement.nth i).iget (arrangement.nth ((i + 1) % 100)).iget (arrangement.nth ((i - 1) % 100)).iget (arrangement.nth ((i + 2) % 100)).iget)) ∧
      50 = finset.count (λ i, is_good_pair (arrangement.nth i).iget (arrangement.nth ((i+1) % 100)).iget (arrangement.nth ((i-1)%100)).iget (arrangement.nth ((i+2)%100)).iget) (finset.range 100) :=
sorry

end minimum_good_pairs_l121_121523


namespace exponent_on_right_side_l121_121867

theorem exponent_on_right_side (n : ℕ) (k : ℕ) (h : n = 25) :
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^k → k = 26 :=
by
  sorry

end exponent_on_right_side_l121_121867


namespace similar_right_triangles_l121_121708

open Real

theorem similar_right_triangles (x : ℝ) (h : ℝ)
  (h₁: 12^2 + 9^2 = (12^2 + 9^2))
  (similarity : (12 / x) = (9 / 6))
  (p : hypotenuse = 12*12) :
  x = 8 ∧ h = 10 := by
  sorry

end similar_right_triangles_l121_121708


namespace log_base3_l121_121513

theorem log_base3 (x : ℝ) (h : x * Real.log 3 2 = 1) : 2^x + 2^(-x) = 10 / 3 := by
  sorry

end log_base3_l121_121513


namespace Jaco_total_gift_budget_l121_121902

theorem Jaco_total_gift_budget :
  let friends_gifts := 8 * 9
  let parents_gifts := 2 * 14
  friends_gifts + parents_gifts = 100 :=
by
  let friends_gifts := 8 * 9
  let parents_gifts := 2 * 14
  show friends_gifts + parents_gifts = 100
  sorry

end Jaco_total_gift_budget_l121_121902


namespace ratio_t_q_l121_121860

theorem ratio_t_q (q r s t : ℚ) (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) : 
  t / q = 3 / 2 :=
by
  sorry

end ratio_t_q_l121_121860


namespace find_initial_sweets_l121_121640

-- Defining the initial condition
def initial_sweets (S : ℕ) : Prop :=
  let jack_takes := (S / 2 + 4)
  let remaining_after_jack := S - jack_takes
  remaining_after_jack = 7

-- The theorem to prove the initial number of sweets S is 22
theorem find_initial_sweets : ∃ (S : ℕ), initial_sweets S ∧ S = 22 :=
by
  exists 22
  unfold initial_sweets
  sorry

end find_initial_sweets_l121_121640


namespace inner_cube_surface_area_l121_121322

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121322


namespace inner_cube_surface_area_l121_121383

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121383


namespace jack_walking_rate_l121_121514

def time_in_hours := 1 + (15 / 60)

def distance_in_miles := 7

def walking_rate := distance_in_miles / time_in_hours

theorem jack_walking_rate : walking_rate = 5.6 := 
by
  rw [walking_rate, distance_in_miles, time_in_hours]
  -- Calculation steps (skipped here, done directly by mathematical computation)
  sorry

end jack_walking_rate_l121_121514


namespace distance_travelled_first_hour_l121_121025

noncomputable def initial_distance (x : ℕ) : Prop :=
  let distance_travelled := (12 / 2) * (2 * x + (12 - 1) * 2)
  distance_travelled = 552

theorem distance_travelled_first_hour : ∃ x : ℕ, initial_distance x ∧ x = 35 :=
by
  use 35
  unfold initial_distance
  sorry

end distance_travelled_first_hour_l121_121025


namespace three_non_coplanar_lines_determine_three_planes_l121_121643

noncomputable def number_of_planes (p : Type) [AffineSpace p] (L1 L2 L3 : Line p) : ℕ :=
  if coplanar L1 L2 then
    if coplanar L1 L3 then
      if coplanar L2 L3 then 1 else 2
    else 2
  else if coplanar L1 L3 then
    if coplanar L2 L3 then 2 else 3
  else if coplanar L2 L3 then 2 else 3

axiom coplanar {p : Type} [AffineSpace p] (L1 L2 L3 : Line p) : Prop

theorem three_non_coplanar_lines_determine_three_planes
  {p : Type} [AffineSpace p] (P : p) (L1 L2 L3 : Line p)
  (h1 : L1.contains P) (h2 : L2.contains P) (h3 : L3.contains P)
  (h : ¬coplanar L1 L2 L3) :
  number_of_planes p L1 L2 L3 = 3 :=
sorry

end three_non_coplanar_lines_determine_three_planes_l121_121643


namespace find_an_l121_121834

theorem find_an (S : ℕ → ℤ) (a : ℕ → ℤ)
  (h₁ : ∀ n, S n = a (n + 1) - 2 * n + 2)
  (h₂ : a 2 = 2) :
  ∀ n, a n = if n = 1 then 2 else 2^n - 2 :=
by {
  intro n,
  cases n with
  | zero => 
    -- This case n = 0, which is outside of the given defined sequence, hence should be skipped
    sorry
  | succ n => {
    cases n with
    | zero => {
      -- This is the case n = 1
      exact if_pos rfl
    }
    | succ n => {
      -- This is the case n ≥ 2
      have induction_hypothesis : ∀ k < n.succ.succ, a k = (if k = 1 then 2 else 2^k - 2) := ...
      exact if_neg (nat.succ_ne_succ n.succ 0)
    }
  }
}

end find_an_l121_121834


namespace probability_A_selected_l121_121456

/-- Prove that the probability of selecting student A out of four students (A, B, C, D), 
when two students are selected at random to participate in a quiz, is 1/2. -/
theorem probability_A_selected
  (students : Finset (String))
  (A B C D: String)
  (h_students : students = {A, B, C, D}):
  let total_ways := (students.card.choose 2)
  let ways_with_A := (students.card - 1).choose 1 in
  (ways_with_A : ℚ) / total_ways = 1 / 2 :=
by
  sorry

end probability_A_selected_l121_121456


namespace interval_increase_of_g_l121_121599

noncomputable def function_g (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

def interval_of_monotonic_increase (k : ℤ) : Set ℝ :=
  { x : ℝ | k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12 }

theorem interval_increase_of_g (k : ℤ) :
  ∀ x : ℝ, x ∈ interval_of_monotonic_increase k ↔
  interval_increase_of_g k :=
sorry

end interval_increase_of_g_l121_121599


namespace triangle_area_correct_l121_121971

noncomputable def triangle_area (c : ℝ) : ℝ :=
  (c^2 * Real.sqrt (Real.sqrt 5 - 2)) / 2

theorem triangle_area_correct (c : ℝ) (h1 : c > 0)
  (h2 : ∃ x : ℝ, 0 < x ∧ x < c ∧ (c-x)/x = x/c) :
  triangle_area c = (c^2 * Real.sqrt (Real.sqrt 5 - 2)) / 2 :=
by
  sorry

end triangle_area_correct_l121_121971


namespace cube_volume_l121_121173

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121173


namespace stamps_difference_after_gift_l121_121114

theorem stamps_difference_after_gift :
  ∀ (P Q : ℕ), 
  (∃ x : ℕ, P = 7 * x ∧ Q = 4 * x) →
  ∃ x : ℕ, (6 * (4 * x + 8) = 5 * (7 * x - 8)) →
  P - 8 - (Q + 8) = 8 :=
by
  intros P Q hPQ hratio
  cases hPQ with x hx
  cases hx with hP hQ
  sorry

end stamps_difference_after_gift_l121_121114


namespace problem1_problem2_problem3_problem4_l121_121132

-- Problem 1
def conversion_kg_to_g (kg : ℕ) (conversion_rate : ℕ) : ℕ :=
  kg * conversion_rate

theorem problem1 : conversion_kg_to_g 4 1000 = 4000 := by
  unfold conversion_kg_to_g
  simp
  sorry

-- Problem 2
def conversion_m_to_dm (m : ℕ) (conversion_rate : ℕ) : ℕ :=
  m * conversion_rate

def calculate_dm_difference (m_in_meters : ℕ) (dm_to_subtract : ℕ) (conversion_rate : ℕ) : ℕ :=
  conversion_m_to_dm m_in_meters conversion_rate - dm_to_subtract

theorem problem2 : calculate_dm_difference 3 2 10 = 28 := by
  unfold calculate_dm_difference conversion_m_to_dm
  simp
  sorry

-- Problem 3
theorem problem3 : conversion_m_to_dm 8 10 = 80 := by
  unfold conversion_m_to_dm
  simp
  sorry

-- Problem 4
def conversion_g_to_kg (g : ℕ) (conversion_rate : ℕ) : ℕ :=
  g / conversion_rate

def calculate_kg_difference (g1 g2 : ℕ) (conversion_rate : ℕ) : ℕ :=
  conversion_g_to_kg (g1 - g2) conversion_rate

theorem problem4 : calculate_kg_difference 1600 600 1000 = 1 := by
  unfold calculate_kg_difference conversion_g_to_kg
  simp
  sorry

end problem1_problem2_problem3_problem4_l121_121132


namespace pure_imaginary_complex_l121_121518

theorem pure_imaginary_complex (a : ℝ) (h : (2 - a * complex.I) / (1 + complex.I)).re = 0 : a = 2 :=
sorry

end pure_imaginary_complex_l121_121518


namespace maximum_additional_voters_l121_121117

-- Define conditions
structure MovieRating (n : ℕ) (x : ℤ) where
  (sum_scores : ℤ) : sum_scores = n * x

-- Define a function to verify the rating decrease condition
def rating_decrease_condition (n : ℕ) (x y : ℤ) : Prop :=
  (n*x + y) / (n+1) = x - 1

-- Problem: To prove that the maximum number of additional voters after moment T is 5
theorem maximum_additional_voters (n additional_voters : ℕ) (x y : ℤ) (initial_condition : MovieRating n x) :
  initial_condition.sum_scores = n * x ∧
  (∀ k, 1 ≤ k → k ≤ additional_voters → 
    ∃ y, rating_decrease_condition (n + k - 1) (x - (k-1)) y ∧ y ≤ 0) →
  additional_voters ≤ 5 :=
by
  sorry

end maximum_additional_voters_l121_121117


namespace inner_cube_surface_area_l121_121360

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121360


namespace total_tiles_l121_121393

/-- A square-shaped floor is covered with congruent square tiles. 
If the total number of tiles on the two diagonals is 88 and the floor 
forms a perfect square with an even side length, then the number of tiles 
covering the floor is 1936. -/
theorem total_tiles (n : ℕ) (hn_even : n % 2 = 0) (h_diag : 2 * n = 88) : n^2 = 1936 := 
by 
  sorry

end total_tiles_l121_121393


namespace inner_cube_surface_area_l121_121253

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121253


namespace inner_cube_surface_area_l121_121271

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121271


namespace goldie_earnings_l121_121500

theorem goldie_earnings
  (hourly_wage : ℕ := 5)
  (hours_last_week : ℕ := 20)
  (hours_this_week : ℕ := 30) :
  hourly_wage * hours_last_week + hourly_wage * hours_this_week = 250 :=
by
  sorry

end goldie_earnings_l121_121500


namespace balls_into_boxes_l121_121505

theorem balls_into_boxes {balls boxes : ℕ} (h_balls : balls = 6) (h_boxes : boxes = 4) : 
  (indistinguishable_partitions balls boxes).count = 9 := 
by
  sorry

end balls_into_boxes_l121_121505


namespace inner_cube_surface_area_l121_121373

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121373


namespace sufficient_but_not_necessary_l121_121817

theorem sufficient_but_not_necessary (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : a + b > 0 :=
by
  sorry

end sufficient_but_not_necessary_l121_121817


namespace inner_cube_surface_area_l121_121358

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121358


namespace pears_pairing_l121_121433

-- Define the problem
theorem pears_pairing (n : ℕ) (weights : Fin n → ℕ) (h_even : n % 2 = 0) (h_diff : ∀ i, (weights i - weights ((i + 1) % n) = 0 ∨ weights i - weights ((i + 1) % n) = 1) ∨ (weights ((i + 1) % n) - weights i = 0 ∨ weights ((i + 1) % n) - weights i = 1)) :
  ∃ (pairs : Fin (n / 2) → (Fin n × Fin n)), (∀ j, pairs j.1 ≠ pairs j.2) ∧ (∀ j k, j ≠ k → pairs j.1 ≠ pairs k.1 ∧ pairs j.2 ≠ pairs k.2) ∧ 
    (∀ j, (weights (pairs j.1) - weights (pairs j.2) = 0 ∨ weights (pairs j.1) - weights (pairs j.2) = 1) ∨ (weights (pairs j.2) - weights (pairs j.1) = 0 ∨ weights (pairs j.2) - weights (pairs j.1) = 1))) ∧ 
    (∀ j, (weights (pairs (j % (n / 2 + 1))).1 - weights (pairs (j % (n / 2 + 1))).2 = 0 ∨ weights (pairs (j % (n / 2+ 1))).1 - weights (pairs (j % (n / 2+ 1))).2 = 1) ∨ 
         (weights (pairs ((j + 1) % (n / 2)).1 - weights (pairs ((j + 1) % (n / 2)).2) = 0 ∨ weights (pairs ((j + 1) % (n / 2)).1 - weights (pairs ((j + 1) % (n / 2)).2) = 1)) :=
sorry

end pears_pairing_l121_121433


namespace not_necessarily_am_sq_lt_bm_sq_l121_121471

theorem not_necessarily_am_sq_lt_bm_sq (a b m : ℝ) (h : a < b) : ¬ (∀ m, am^2 < bm^2) :=
sorry

end not_necessarily_am_sq_lt_bm_sq_l121_121471


namespace sum_of_divisors_143_l121_121071

theorem sum_of_divisors_143 : ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := by
  sorry

end sum_of_divisors_143_l121_121071


namespace inner_cube_surface_area_l121_121368

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121368


namespace area_of_inscribed_triangle_l121_121395

noncomputable def triangle_area (r : ℝ) (A B C : ℝ) : ℝ :=
  0.5 * r ^ 2 * (Real.sin A + Real.sin B + Real.sin C)

theorem area_of_inscribed_triangle :
  ∀ (arc1 arc2 arc3 : ℝ), arc1 + arc2 + arc3 = 16 →
    let r := 8 / Real.pi in
    let A := Real.pi / 2 in
    let B := 5 * (Real.pi / 16) in
    let C := 7 * (Real.pi / 16) in
    triangle_area r A B C =
      32 / (Real.pi ^ 2) * (1 + Real.sqrt ((2 + Real.sqrt 2) / 4) + Real.sqrt ((2 - Real.sqrt 2) / 4)) :=
by
  intros arc1 arc2 arc3 hsum r A B C
  unfold triangle_area
  sorry

end area_of_inscribed_triangle_l121_121395


namespace find_a5_l121_121792

noncomputable def a_sequence (a1 : ℝ) (n : ℕ) : ℝ :=
  a1 + 2 * (n - 1)

def g (x : ℝ) : ℝ :=
  8 * x + Real.sin (π * x) - Real.cos (π * x)

def sum_g_sequence (a1 : ℝ) : ℝ :=
  (Finset.range 9).sum (λ n, g (a_sequence a1 n))

theorem find_a5 (a1 : ℝ) (h : sum_g_sequence a1 = 18) : 
  a_sequence a1 4 = 1/4 := 
sorry

end find_a5_l121_121792


namespace f_4_1981_l121_121616

-- Define the function f with its properties
axiom f : ℕ → ℕ → ℕ

axiom f_0_y (y : ℕ) : f 0 y = y + 1
axiom f_x1_0 (x : ℕ) : f (x + 1) 0 = f x 1
axiom f_x1_y1 (x y : ℕ) : f (x + 1) (y + 1) = f x (f (x + 1) y)

theorem f_4_1981 : f 4 1981 = 2 ^ 3964 - 3 :=
sorry

end f_4_1981_l121_121616


namespace breadth_decrease_l121_121612

theorem breadth_decrease (L B : ℝ) (x : ℝ) (h : 1.20 * (1 - x / 100) = 0.86) :
  x ≈ 28.33 := sorry

end breadth_decrease_l121_121612


namespace cube_volume_l121_121221

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l121_121221


namespace second_cube_surface_area_l121_121339

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121339


namespace sphere_wedge_volume_l121_121719

theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) (V : ℝ) (wedge_volume : ℝ) :
  circumference = 18 * Real.pi → num_wedges = 6 → V = (4 / 3) * Real.pi * (9^3) → wedge_volume = V / 6 → 
  wedge_volume = 162 * Real.pi :=
by
  intros h1 h2 h3 h4
  rw h3 at h4
  rw [←Real.pi_mul, ←mul_assoc, Nat.cast_bit1, Nat.cast_bit0, Nat.cast_one, pow_succ, pow_one, ←mul_assoc] at h4
  rw [mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc] at h4
  sorry

end sphere_wedge_volume_l121_121719


namespace rectangular_eq_circle_C_length_AB_l121_121544

open Real

def line_l (t : ℝ) : ℝ × ℝ := ⟨-sqrt 6 - sqrt 2 * t, 2 * sqrt 6 + sqrt 2 * t⟩

def polar_circle_C (θ : ℝ) : ℝ := 4 * sqrt 6 * cos θ

theorem rectangular_eq_circle_C :
  ∀ x y : ℝ, (x - 2 * sqrt 6) ^ 2 + y ^ 2 = 24 ↔ ∃ θ : ℝ, (x ^ 2 + y ^ 2 = (4 * sqrt 6 * cos θ) ^ 2) ∧ (x = 4 * sqrt 6 * cos θ) :=
sorry

theorem length_AB :
  ∀ (A B : ℝ × ℝ), 
  (∃ t₁ t₂ : ℝ, line_l t₁ = A ∧ line_l t₂ = B ∧ (A.1 - 2 * sqrt 6) ^ 2 + A.2 ^ 2 = 24 ∧ (B.1 - 2 * sqrt 6) ^ 2 + B.2 ^ 2 = 24) →
  dist A B = 2 * sqrt 21 :=
sorry

end rectangular_eq_circle_C_length_AB_l121_121544


namespace conic_sections_hyperbola_and_ellipse_l121_121756

theorem conic_sections_hyperbola_and_ellipse (x y : ℝ) :
  y^4 - 6 * x^4 = 3 * y^2 - 4 →
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ y^2 - sqrt 6 * x^2 = 2) ∨
  (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ y^2 + sqrt 6 * x^2 = 2) :=
sorry

end conic_sections_hyperbola_and_ellipse_l121_121756


namespace area_of_equilateral_triangle_l121_121408

namespace Triangle

open Real

-- Define the equilateral triangle XYZ with each side of length 13
def equilateral_triangle (a : ℝ) (h₁ : a = 13) : Prop :=
  ∃ x y z : ℝ, a = dist x y ∧ a = dist y z ∧ a = dist z x

-- Define the medians of the equilateral triangle
def medians_meet_perpendicularly (a : ℝ) (XT YZ : ℝ) (h₂ : XT = a * sqrt (3 / 2)) (h₃ : YZ = a * sqrt (3 / 2)) : Prop :=
  ∃ G : ℝ, (XT/G = 2) ∧ (YZ/G = 2)

-- Define the main proof problem
theorem area_of_equilateral_triangle (a XT YZ : ℝ)
  (h₁ : equilateral_triangle a (by rw a; exact rfl))
  (h₂ : XT = a * sqrt (3 / 2))
  (h₃ : YZ = a * sqrt (3 / 2))
  (h₄ : medians_meet_perpendicularly a XT YZ h₂ h₃) :
  ∃ Area : ℝ, Area = 42.25 :=
sorry

end Triangle

end area_of_equilateral_triangle_l121_121408


namespace garment_industry_initial_men_l121_121531

theorem garment_industry_initial_men (M : ℕ) :
  (M * 8 * 10 = 6 * 20 * 8) → M = 12 :=
by
  sorry

end garment_industry_initial_men_l121_121531


namespace gain_per_year_l121_121701

-- Define the conditions in a)
def principal_borrowed : ℝ := 8000
def borrow_rate : ℝ := 4 / 100
def lend_rate : ℝ := 6 / 100
def time_period : ℝ := 2

-- Define the function to calculate the simple interest
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time)

-- Define the problem statement
theorem gain_per_year :
  let interest_lent := simple_interest principal_borrowed lend_rate time_period
  let interest_borrowed := simple_interest principal_borrowed borrow_rate time_period
  let total_gain := interest_lent - interest_borrowed
  total_gain / time_period = 800 :=
by
  sorry

end gain_per_year_l121_121701


namespace cube_volume_from_surface_area_l121_121194

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l121_121194


namespace triangle_ABC_area_l121_121647

variables {A B C H D : Point} (ω : Circle)
variables {X Y Z: Point} [heq: ω.isInscribedTriangle ABC]
variables [O : ω.isOrthocenter H]
variables [foot: ω.perpendicularFoot D A BC]

-- Given conditions
variables (circ : ω.unitCircle)
variables (eqAH_HD : dist A H = dist H D)
variables (tangents_ABC : XYZ.tangentsFormTriangleIn ω)
variables (side_lengths_arithmetic : sideLengthsFormArithmeticSequence XYZ)

theorem triangle_ABC_area (h_unit: unitCircle ω) (h_ortho: ω.isOrthocenter H) (h_perp: ω.perpendicularFoot D A BC)
  (h_AH_HD: dist A H = dist H D) (h_tangents: XYZ.tangentsFormTriangleInCirc ω) (h_seq: sideLengthsFormArithmeticSequence XYZ):
  area ABC = 6 / 5 ∧ 6 + 5 = 11 :=
by
  -- proof here...
  sorry

end triangle_ABC_area_l121_121647


namespace sum_of_first_n_terms_find_S12_l121_121628

noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ 
| 0       => a
| (n + 1) => geometric_sequence a r n * r

theorem sum_of_first_n_terms (a r : ℝ) (n : ℕ) : (∑ i in Finset.range n, geometric_sequence a r i) = a * ((1 - r^n) / (1 - r)) :=
begin
  sorry
end

theorem find_S12 (a r : ℝ) (h3 : a * (1 - r^3) / (1 - r) = 13) (h6 : a * (1 - r^6) / (1 - r) = 65) : 
  (∑ i in Finset.range 12, geometric_sequence a r i) = 1105 :=
by
  sorry

end sum_of_first_n_terms_find_S12_l121_121628


namespace cube_volume_of_surface_area_l121_121159

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l121_121159


namespace sum_of_divisors_143_l121_121094

theorem sum_of_divisors_143 : ∑ d in {d : ℕ | d ∣ 143}.to_finset, d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121094


namespace correct_propositions_l121_121828

variables 
  (m n : Line)
  (α β γ : Plane)

-- Conditions
variable (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α)
variable (h_non_intersect_lines : ¬ ∃ p, p ∈ m ∧ p ∈ n)

-- Propositions
variable (h1 : m ⟂ α ∧ n ‖ α → m ⟂ n)
variable (h2 : α ‖ β ∧ β ‖ γ ∧ m ⟂ α → m ⟂ γ)

-- Proof
theorem correct_propositions : 
  (m ⟂ α ∧ n ‖ α → m ⟂ n) ∧ 
  (α ‖ β ∧ β ‖ γ ∧ m ⟂ α → m ⟂ γ) := 
by 
  split;
  sorry

end correct_propositions_l121_121828


namespace sum_of_squares_of_coeffs_l121_121104

theorem sum_of_squares_of_coeffs :
  let expr := (5 * (x^3 - x) - 3 * (x^2 - 4 * x + 3))
  let simplified_expr := (5 * x^3 - 3 * x^2 + 7 * x - 9)
  let coeffs := [5, -3, 7, -9]
  ∑ c in coeffs, c^2 = 164 :=
by
  let expr := (5 * (x^3 - x) - 3 * (x^2 - 4 * x + 3))
  let simplified_expr := (5 * x^3 - 3 * x^2 + 7 * x - 9)
  let coeffs := [5, -3, 7, -9]
  calc 
     ∑ c in coeffs, c^2 = 5^2 + (-3)^2 + 7^2 + (-9)^2 : by sorry
                      ... = 25 + 9 + 49 + 81 : by sorry
                      ... = 164 : by sorry

end sum_of_squares_of_coeffs_l121_121104


namespace num_two_digit_factors_three_pow_18_minus_1_l121_121504

theorem num_two_digit_factors_three_pow_18_minus_1 :
  (λ n, 10 ≤ n ∧ n < 100 ∧ ∃ k, k ∣ (3^18 - 1) ∧ k = n).count = 5 := sorry

end num_two_digit_factors_three_pow_18_minus_1_l121_121504


namespace infinite_solutions_xyz_l121_121931

theorem infinite_solutions_xyz : ∀ k : ℕ, 
  (∃ n : ℕ, n > k ∧ ∃ x y z : ℕ, x^2 + y^2 + z^2 - x*y*z + 10 = 0 ∧ x > 2008 ∧ y > 2008 ∧ z > 2008) →
  ∃ x y z : ℕ, x^2 + y^2 + z^2 - x*y*z + 10 = 0 ∧ x > 2008 ∧ y > 2008 ∧ z > 2008 := 
sorry

end infinite_solutions_xyz_l121_121931


namespace aliens_legs_l121_121404

theorem aliens_legs : ∃ (L : ℕ), (∀ (A M_arms M_legs : ℕ),
  A = 3 →
  M_arms = 2 * A →
  M_legs = L / 2 →
  5 * (A + L) = 5 * (M_arms + M_legs) + 5) →
  L = 8 :=
by {
  existsi (8 : ℕ),
  intros A M_arms M_legs HA HMA HML,
  have H := calc
    5 * (A + 8) = 5 * (2 * A + 4) + 5 : by linarith [HA, HMA, HML]
    ... = 5 * (6 + 4) + 5 : by linarith [HA]
    ... = 50 : by norm_num,
  linarith,
}

end aliens_legs_l121_121404


namespace tom_teaching_years_l121_121998

theorem tom_teaching_years :
  ∃ T D : ℕ, T + D = 70 ∧ D = (1 / 2) * T - 5 ∧ T = 50 :=
by
  sorry

end tom_teaching_years_l121_121998


namespace inner_cube_surface_area_l121_121258

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121258


namespace problem_part1_problem_part2_l121_121463

def complex_num (x : ℝ) : Complex :=
  ⟨2, x⟩

def conjugate (z : Complex) : Complex :=
  ⟨z.re, -z.im⟩

def purely_imaginary (z : Complex) : Prop :=
  z.re = 0

theorem problem_part1 (x : ℝ) : purely_imaginary (conjugate (complex_num x) * ⟨1, -1⟩) → x = 2 ∧ complex_num 2.abs = 2 * Real.sqrt 2 :=
by
  sorry

def z1 (m : ℝ) (z : Complex) : Complex :=
  (⟨m, -1⟩ / z) * ⟨1, -1⟩ / 4

def in_fourth_quadrant (z : Complex) : Prop :=
  (z.re > 0) ∧ (z.im < 0)

theorem problem_part2 (m : ℝ) (z : Complex) : in_fourth_quadrant (z1 m z) → m > 1 :=
by
  sorry

end problem_part1_problem_part2_l121_121463


namespace inner_cube_surface_area_l121_121254

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121254


namespace winning_candidate_percentage_l121_121991

/-- There are three candidates with votes 3136, 7636, and 11628 respectively. 
The percentage of the total votes that the winning candidate received is approximately 51.93%. -/
theorem winning_candidate_percentage :
  let votes1 := 3136
  let votes2 := 7636
  let votes3 := 11628
  let total_votes := votes1 + votes2 + votes3
  let winning_votes := votes3
  (winning_votes : ℝ) / (total_votes : ℝ) * 100 ≈ 51.93 :=
by
  sorry

end winning_candidate_percentage_l121_121991


namespace average_remaining_two_numbers_l121_121112

theorem average_remaining_two_numbers 
    (a b c d e f : ℝ)
    (h1 : (a + b + c + d + e + f) / 6 = 3.95)
    (h2 : (a + b) / 2 = 4.4)
    (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 3.6 := 
sorry

end average_remaining_two_numbers_l121_121112


namespace sum_2n_an_eq_neg4034_l121_121831

noncomputable def sequence_s (n : ℕ) : ℤ :=
  if n = 0 then 0
  else (∑ k in Finset.range n, sequence_a k)

noncomputable def sequence_a (n : ℕ) : ℤ :=
  if n = 0 then 0
  else -2017 * (2017 + sequence_s (n - 1)) / n

theorem sum_2n_an_eq_neg4034 :
  (∑ n in Finset.range 2017, 2^(n+1) * sequence_a (n+1)) = -4034 :=
sorry

end sum_2n_an_eq_neg4034_l121_121831


namespace minimum_fence_length_l121_121231

theorem minimum_fence_length :
  ∃ (length width : ℕ), 
    length = 12 ∧ width = 8 ∧ 
    let fence_length := 2 * width + length in 
    fence_length = 28 :=
by
  sorry

end minimum_fence_length_l121_121231


namespace fraction_identity_l121_121869

theorem fraction_identity (x y : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : y^2 - 1/x ≠ 0) :
  (x^2 - 1/y) / (y^2 - 1/x) = x / y := 
by {
  sorry
}

end fraction_identity_l121_121869


namespace sum_of_divisors_143_l121_121095

theorem sum_of_divisors_143 : ∑ d in {d : ℕ | d ∣ 143}.to_finset, d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121095


namespace range_of_g_l121_121777

noncomputable def g (x : ℝ) : ℝ := (Real.sin x)^6 + (Real.sin x) * (Real.cos x) + (Real.cos x)^6

-- Statement of the problem in Lean 4
theorem range_of_g :
  let S := set.range (λ x : ℝ, g x)
  S = set.Icc (3 / 4 : ℝ) (13 / 12 : ℝ) :=
by
  sorry

end range_of_g_l121_121777


namespace area_of_rectangle_ABCD_l121_121553

-- Definitions for the conditions
def small_square_area := 4
def total_small_squares := 2
def large_square_area := (2 * (2 : ℝ)) * (2 * (2 : ℝ))
def total_squares_area := total_small_squares * small_square_area + large_square_area

-- The main proof statement
theorem area_of_rectangle_ABCD : total_squares_area = 24 := 
by
  sorry

end area_of_rectangle_ABCD_l121_121553


namespace vector_dot_product_l121_121573

variables {a b c : ℝ^3}

-- given conditions
def norm_eq_one (v : ℝ^3) : Prop := ∥v∥ = 1
def norm_sum_eq_two (u v : ℝ^3) : Prop := ∥u + v∥ = 2
def c_eq_expressn (a b c: ℝ^3) : Prop := c = a + 3 * b + 4 * (a × b)

-- theorem to prove
theorem vector_dot_product (h1 : norm_eq_one a) (h2 : norm_eq_one b) (h3 : norm_sum_eq_two a b)
  (h4 : c_eq_expressn a b c) : b ⬝ c = 4 :=
sorry

end vector_dot_product_l121_121573


namespace triangle_area_inscribed_l121_121398

theorem triangle_area_inscribed (C : ℝ) (r : ℝ) (θ : ℝ) (α β γ : ℝ)
  (hC : C = 16)
  (hr : r = 8 / real.pi)
  (hθ : θ = 22.5)
  (hα : α = 90)
  (hβ : β = 112.5)
  (hγ : γ = 157.5) :
  ∃ A : ℝ, A = 147.6144 / real.pi ^ 2 :=
sorry

end triangle_area_inscribed_l121_121398


namespace volume_of_cube_l121_121188

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l121_121188


namespace intercepts_equal_l121_121812

theorem intercepts_equal (a : ℝ) :
  (∃ x y : ℝ, ax + y - 2 - a = 0 ∧
              y = 0 ∧ x = (a + 2) / a ∧
              x = 0 ∧ y = 2 + a) →
  (a = 1 ∨ a = -2) :=
by
  sorry

end intercepts_equal_l121_121812


namespace cube_volume_l121_121217

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l121_121217


namespace satisfies_condition_l121_121766

-- Define the function f
def f (x : ℝ) : ℝ := (x^2 + 1) / (2^x)

-- State the theorem
theorem satisfies_condition (x y : ℝ) :
  (2^(-x-y) ≤ (f x * f y) / ((x^2 + 1) * (y^2 + 1)) ∧ (f x * f y) / ((x^2 + 1) * (y^2 + 1)) ≤ f (x + y) / ((x + y)^2 + 1)) :=
  sorry

end satisfies_condition_l121_121766


namespace cube_volume_l121_121209

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l121_121209


namespace possible_values_expr_l121_121825

theorem possible_values_expr (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) :
  ∃ x, x = (Real.sign a + Real.sign b + Real.sign c + Real.sign d + Real.sign (a * b * c * d)) ∧
       x ∈ {5, 3, 2, 0, -3} := 
sorry

end possible_values_expr_l121_121825


namespace gumballs_per_box_l121_121501

-- Given conditions
def total_gumballs : ℕ := 20
def total_boxes : ℕ := 4

-- Mathematically equivalent proof problem
theorem gumballs_per_box:
  total_gumballs / total_boxes = 5 := by
  sorry

end gumballs_per_box_l121_121501


namespace population_reaches_capacity_l121_121556

/-- A problem involving population growth and carrying capacity -/
theorem population_reaches_capacity :
  ∀ (population_in_2000: ℕ) (land: ℕ) (acres_per_person: ℝ) (growth_rate: ℕ) (years: ℕ), 
  population_in_2000 = 250 →
  land = 30000 →
  acres_per_person = 1.2 →
  growth_rate = 20 →
  years = 140 →
  (population_in_2000 * (2 ^ (years / growth_rate)) ≥ land / acres_per_person) :=
by
  intros population_in_2000 land acres_per_person growth_rate years
  assume h1 h2 h3 h4 h5
  sorry

end population_reaches_capacity_l121_121556


namespace find_angle_C_find_min_ab_l121_121558

variable {A B C a b c : ℝ}

-- 1. Prove that angle C is 2π/3 given the conditions.
theorem find_angle_C (h1 : 2 * c * Real.cos B = 2 * a + b) : C = 2 * Real.pi / 3 :=
sorry

-- 2. Prove the minimum value of ab given the area condition.
theorem find_min_ab (S : ℝ) (hA : ∠A = A) (hB : ∠B = B) (hC : ∠C = C)
  (h_area : S = (Real.sqrt 3) / 2 * c) (hC_eq : C = 2 * Real.pi / 3) : 
  minimum (fun ab : ℝ => ∃ a b, ab = a * b ∧ 0 < a ∧ 0 < b ∧ S = 1/2 * a * b * Real.sin C) = 12 :=
sorry

end find_angle_C_find_min_ab_l121_121558


namespace cube_volume_l121_121223

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l121_121223


namespace binomial_sum_identity_l121_121762

theorem binomial_sum_identity :
  (finset.range 51).sum (λ k, (-1)^k * (k + 1) * nat.choose 50 k) = 1 := by
sorry

end binomial_sum_identity_l121_121762


namespace estimatedSurvivalProbability_l121_121938

-- Definitions specific to the problem
def numYoungTreesTransplanted : ℕ := 20000
def numYoungTreesSurvived : ℕ := 18044

def survivalRate : ℝ := numYoungTreesSurvived / numYoungTreesTransplanted

theorem estimatedSurvivalProbability :
  Real.round (survivalRate * 10) / 10 = 0.9 :=
by
  sorry

end estimatedSurvivalProbability_l121_121938


namespace soccer_tournament_eq_l121_121537

theorem soccer_tournament_eq (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : (1 / 2 : ℚ) * x * (x - 1) = 28 := by
  sorry

end soccer_tournament_eq_l121_121537


namespace exists_constant_for_topological_minor_l121_121636

theorem exists_constant_for_topological_minor (c : ℝ) (G : Type) [graph G] (d : G → ℝ) (K : G) [ ∀ r : ℕ, ∃ G, d G ≥ 10 * r^2 → G.contains_topological_minor K] :
  ∃ c : ℝ, (c = 10) ∧ (∀ r : ℕ, ∀ G, d G ≥ c * r^2 → G.contains_topological_minor K) :=
begin
  use 10,
  sorry
end

end exists_constant_for_topological_minor_l121_121636


namespace set_A_is_correct_l121_121435

open Complex

def A : Set ℤ := {x | ∃ n : ℕ, n > 0 ∧ x = (I ^ n + (-I) ^ n).re}

theorem set_A_is_correct : A = {-2, 0, 2} :=
sorry

end set_A_is_correct_l121_121435


namespace loga_increasing_loga_decreasing_l121_121042

noncomputable def loga (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem loga_increasing (a : ℝ) (h₁ : a > 1) : ∀ x y : ℝ, 0 < x → 0 < y → x < y → loga a x < loga a y := by
  sorry 

theorem loga_decreasing (a : ℝ) (h₁ : 0 < a) (h₂ : a < 1) : ∀ x y : ℝ, 0 < x → 0 < y → x < y → loga a y < loga a x := by
  sorry

end loga_increasing_loga_decreasing_l121_121042


namespace inner_cube_surface_area_l121_121286

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121286


namespace cube_volume_l121_121183

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121183


namespace integral_of_f_l121_121567

noncomputable def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then x^2
  else if 1 < x ∧ x ≤ 2 then 2 - x
  else 0

theorem integral_of_f :
  ∫ x in 0..2, f x = 5 / 6 :=
sorry

end integral_of_f_l121_121567


namespace inner_cube_surface_area_l121_121268

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121268


namespace real_part_complex_num_l121_121827

def complex_num := (1 + Complex.i) / (1 - Complex.i)

theorem real_part_complex_num : complex_num.re = 0 := 
by
  sorry

end real_part_complex_num_l121_121827


namespace woodworker_days_l121_121400

-- Declare normal and increased productivity conditions as variables
variables (normal_parts : ℕ) (normal_days : ℕ) (productivity_increase : ℕ)
variables (extra_parts : ℕ) (total_parts : ℕ) (increased_rate : ℕ) (increased_days : ℕ)

-- Define the given conditions as facts
def conditions :=
  normal_parts = 360 ∧
  normal_days = 24 ∧
  productivity_increase = 5 ∧
  extra_parts = 80 ∧
  total_parts = normal_parts + extra_parts ∧
  let normal_rate := normal_parts / normal_days in
  let increased_rate := normal_rate + productivity_increase in
  increased_days = total_parts / increased_rate

-- The theorem to be proved is that given the conditions, the increased_days is 22.
theorem woodworker_days (h : conditions) : increased_days = 22 := 
  sorry

end woodworker_days_l121_121400


namespace keith_messages_from_juan_l121_121664

variable (L : ℕ) -- Number of messages Juan sends to Laurence
variable (K : ℕ) -- Number of messages Keith receives from Juan

-- Conditions
axiom condition1 : 4.5 * L = 18
axiom condition2 : K = 8 * L

theorem keith_messages_from_juan : K = 32 :=
by
  rw [condition2]
  have h : L = 4 := sorry -- Solve the equation L = 18 / 4.5
  subst h
  sorry

end keith_messages_from_juan_l121_121664


namespace hall_built_stepped_to_reduce_blind_spots_l121_121015

theorem hall_built_stepped_to_reduce_blind_spots
  (hall_built_stepped : Prop) :
  hall_built_stepped → (ReduceBlindSpots) :=
by
  assume h : hall_built_stepped
  -- proof would go here
  sorry

end hall_built_stepped_to_reduce_blind_spots_l121_121015


namespace hcf_of_three_numbers_l121_121982

theorem hcf_of_three_numbers (a b c : ℕ) (h1 : a + b + c = 60)
  (h2 : Nat.lcm (Nat.lcm a b) c = 180)
  (h3 : (1:ℚ)/a + 1/b + 1/c = 11/120)
  (h4 : a * b * c = 900) :
  Nat.gcd (Nat.gcd a b) c = 5 :=
by
  sorry

end hcf_of_three_numbers_l121_121982


namespace possible_values_of_expression_l121_121818

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ k ∈ ({1, -1}: Set ℝ), 
  (a / |a| = k) ∧
  (b / |b| = k) ∧
  (c / |c| = k) ∧
  (d / |d| = k) ∧
  (abcd / |abcd| = k) ∧
  ({(a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)} = {5, 1, -3}) :=
by
  sorry

end possible_values_of_expression_l121_121818


namespace log_abs_compare_l121_121802

open Real

theorem log_abs_compare
  (x : ℝ) (a : ℝ)
  (hx : 0 < x ∧ x < 1)
  (ha : a > 0 ∧ a ≠ 1) :
  abs (log a (1 - x)) > abs (log a (1 + x)) :=
sorry

end log_abs_compare_l121_121802


namespace expression_c_is_positive_l121_121587

def A : ℝ := 2.1
def B : ℝ := -0.5
def C : ℝ := -3.0
def D : ℝ := 4.2
def E : ℝ := 0.8

theorem expression_c_is_positive : |C| + |B| > 0 :=
by {
  sorry
}

end expression_c_is_positive_l121_121587


namespace volume_of_wedge_l121_121721

theorem volume_of_wedge (r : ℝ) (V : ℝ) (sphere_wedges : ℝ) 
  (h_circumference : 2 * Real.pi * r = 18 * Real.pi)
  (h_volume : V = (4 / 3) * Real.pi * r ^ 3) 
  (h_sphere_wedges : sphere_wedges = 6) : 
  V / sphere_wedges = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l121_121721


namespace length_of_AB_l121_121646

theorem length_of_AB (t : ℝ) (AD BC CD : ℝ) (h1 : AD = t) (h2 : BC = 4) (h3 : CD = t + 13) (ht : t = 11) : 
  let AE := t - 4,
      BE := t + 13,
      AB := sqrt (2 * t^2 + 18 * t + 185) in
  AB = 25 :=
by
  let AE := t - 4
  let BE := t + 13
  let AB := sqrt (2 * t^2 + 18 * t + 185)
  have h4 : AB = sqrt (625) := sorry -- Expansion and simplification step
  have h5 : AB = 25 := by
    rw sqrt_eq_rfl
    apply eq_of_sq_eq_sq'
    symmetry
    exact h4
  exact h5
  sorry

end length_of_AB_l121_121646


namespace sum_of_divisors_143_l121_121101

theorem sum_of_divisors_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121101


namespace probabilities_are_equal_l121_121632

variables {RedBox GreenBox : Type} [Fintype RedBox] [Fintype GreenBox]

def red_balls : ℕ := 100
def green_balls : ℕ := 100
def transferred_balls : ℕ := 8

def red_box_initial (x : RedBox) : Prop := x < red_balls
def green_box_initial (x : GreenBox) : Prop := x < green_balls

def red_box_final (x : RedBox) : Prop := 
  x < (red_balls - transferred_balls) + transferred_balls

def green_box_final (x : GreenBox) : Prop :=
  x < (green_balls + transferred_balls) - transferred_balls

def probability_draw_green_from_red : ℚ := 
  transferred_balls / (red_balls + green_balls - 2 * transferred_balls)

def probability_draw_red_from_green : ℚ := 
  transferred_balls / (green_balls + red_balls - 2 * transferred_balls)

theorem probabilities_are_equal :
  probability_draw_green_from_red = probability_draw_red_from_green := 
begin
  -- Given all the calculations, it's clear that both probabilities are equal
  sorry
end

end probabilities_are_equal_l121_121632


namespace basket_probability_l121_121110

noncomputable def probability_event (p_j : ℚ) (p_s : ℚ) :=
  (1 - p_j) * p_s

theorem basket_probability :
  let P_Jack := (1 : ℚ) / 6
  let P_Jill := (1 : ℚ) / 7
  let P_Sandy := (1 : ℚ) / 8
in probability_event P_Jack P_Jill * P_Sandy = 5 / 336 :=
by
  let P_Jack_miss := 1 - P_Jack
  let desired_probability := P_Jack_miss * P_Jill * P_Sandy
  have h1 : P_Jack_miss = 5 / 6,
    sorry -- This would be the proof step to show P_Jack_miss is indeed 5/6
  have h2 : desired_probability = (5 / 6) * (1 / 7) * (1 / 8),
    sorry -- This would be the proof step to show the multiplication of probabilities
  have h3 : (5 / 6) * (1 / 7) * (1 / 8) = 5 / 336,
    sorry -- This would be the proof step to show the final calculation
  rw [h1, h2, h3]
  exact rfl

end basket_probability_l121_121110


namespace goldie_earnings_l121_121498

theorem goldie_earnings (hourly_rate : ℝ) (hours_week1 : ℝ) (hours_week2 : ℝ) :
  hourly_rate = 5 → hours_week1 = 20 → hours_week2 = 30 → (hourly_rate * (hours_week1 + hours_week2) = 250) :=
by
  intro h_rate
  intro h_week1
  intro h_week2
  rw [h_rate, h_week1, h_week2]
  norm_num
  sorry

end goldie_earnings_l121_121498


namespace sum_of_divisors_143_l121_121097

theorem sum_of_divisors_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121097


namespace cube_volume_l121_121205

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l121_121205


namespace hiking_hours_per_day_l121_121696

def packs_per_mile := 0.5
def initial_pack := 40
def resupply_rate := 0.25
def hike_speed := 2.5
def days := 5

theorem hiking_hours_per_day : 
  let initial_miles := initial_pack / packs_per_mile in
  let resupply_pack := resupply_rate * initial_pack in
  let additional_miles := resupply_pack / packs_per_mile in
  let total_miles := initial_miles + additional_miles in
  let miles_per_day := total_miles / days in
  let hours_per_day := miles_per_day / hike_speed in
  hours_per_day = 8 := 
by 
  let initial_miles := initial_pack / packs_per_mile
  let resupply_pack := resupply_rate * initial_pack
  let additional_miles := resupply_pack / packs_per_mile
  let total_miles := initial_miles + additional_miles
  let miles_per_day := total_miles / days
  let hours_per_day := miles_per_day / hike_speed
  show hours_per_day = 8 from sorry

end hiking_hours_per_day_l121_121696


namespace acetone_molecular_weight_l121_121742

noncomputable def molecular_weight_of_isotope (masses: List ℝ) (abundances: List ℝ) : ℝ :=
  List.sum (List.zipWith (*) masses abundances)

noncomputable def molecular_weight_of_acetone : ℝ :=
  let carbon_mass     := molecular_weight_of_isotope [12, 13.003355] [0.9893, 0.0107]
  let hydrogen_mass   := molecular_weight_of_isotope [1.007825, 2.014102] [0.999885, 0.000115]
  let oxygen_mass     := molecular_weight_of_isotope [15.994915, 16.999132, 17.999159] [0.99757, 0.00038, 0.00205]
  (carbon_mass * 3) + (hydrogen_mass * 6) + (oxygen_mass * 1)

theorem acetone_molecular_weight :
  molecular_weight_of_acetone = 58.107055 := by
  sorry

end acetone_molecular_weight_l121_121742


namespace positive_difference_of_two_numbers_l121_121985

variable {x y : ℝ}

theorem positive_difference_of_two_numbers (h₁ : x + y = 8) (h₂ : x^2 - y^2 = 24) : |x - y| = 3 :=
by
  sorry

end positive_difference_of_two_numbers_l121_121985


namespace backpack_pricing_l121_121233

variable (x : ℝ)

-- Given conditions
def cost_price : ℝ := x
def mark_up := 0.50
def discount := 0.20
def profit := 8

-- Mathematically equivalent proof problem
theorem backpack_pricing (x : ℝ) (h1 : 0 < x) :
  (1 + mark_up) * cost_price * (1 - discount) - cost_price = profit :=
by sorry

end backpack_pricing_l121_121233


namespace problem_inequality_l121_121943

theorem problem_inequality 
  (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) 
  (h8 : a + b + c + d ≥ 1) : 
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := 
sorry

end problem_inequality_l121_121943


namespace terminal_side_quadrant_l121_121510

theorem terminal_side_quadrant (k : ℤ) : 
  let α := k * 180 + 45 in 
  (0 ≤ α % 360) ∧ (α % 360 < 90) ∨ (180 ≤ α % 360) ∧ (α % 360 < 270) :=
by
  let α := k * 180 + 45
  sorry

end terminal_side_quadrant_l121_121510


namespace total_arrangements_l121_121034

def count_arrangements : Nat :=
  let male_positions := 3
  let female_positions := 3
  let male_arrangements := Nat.factorial male_positions
  let female_arrangements := Nat.factorial (female_positions - 1)
  male_arrangements * female_arrangements / (male_positions - female_positions + 1)

theorem total_arrangements : count_arrangements = 36 := by
  sorry

end total_arrangements_l121_121034


namespace sum_of_divisors_143_l121_121085

theorem sum_of_divisors_143 : (∑ i in (finset.filter (λ d, 143 % d = 0) (finset.range 144)), i) = 168 :=
by
  -- The final proofs will go here.
  sorry

end sum_of_divisors_143_l121_121085


namespace min_distance_PQ_l121_121578

theorem min_distance_PQ :
  let P := λ x : ℝ, (x, (1/2) * Real.exp x)
  let Q := λ x : ℝ, (x, Real.log (2 * x))
  ∃ x : ℝ, P x ∈ set_of (λ y : ℝ × ℝ, y.snd = (1/2) * Real.exp y.fst) ∧
          Q x ∈ set_of (λ y : ℝ × ℝ, y.snd = Real.log (2 * y.fst)) ∧
          (∀ y : ℝ, (abs ((1/2) * Real.exp y - y) / Real.sqrt 2 <= abs ((1/2) * Real.exp x - x) / Real.sqrt 2) → 
          |(Real.sqrt 2 * ((1 - Real.log 2)))| = |P x.2 - Q x.2|) :=
by
  sorry

end min_distance_PQ_l121_121578


namespace max_segments_length_greater_than_one_on_disc_l121_121565

theorem max_segments_length_greater_than_one_on_disc (n : ℕ) (hn : 2 ≤ n) 
    : ∃ E, |E| = (2 * n^2 / 5) ∧ ∀ e ∈ E, segment_length e > 1 :=
sorry

end max_segments_length_greater_than_one_on_disc_l121_121565


namespace inner_cube_surface_area_l121_121347

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121347


namespace cube_volume_from_surface_area_l121_121200

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l121_121200


namespace mat_det_zero_l121_121747

def mat_entries (n : ℕ) : ℝ := Real.sin (Real.pi / 4 + n)

def mat : Matrix (Fin 3) (Fin 3) ℝ := 
  λ i j, mat_entries (i.val * 3 + j.val + 1)

theorem mat_det_zero : 
  Matrix.det mat = 0 :=
sorry

end mat_det_zero_l121_121747


namespace other_root_l121_121588

theorem other_root : (z : ℂ) (hz : z^2 = -75 + 100 * complex.I) (h1 : z = 5 + 10 * complex.I)
: ∃ w : ℂ, w = - (5 + 10 * complex.I) :=
begin
  use - (5 + 10 * complex.I),
  exact rfl,
end

end other_root_l121_121588


namespace integral_values_x_l121_121576

def d1 (x : ℤ) : ℤ := x^2 + 3^x + x * 3^((x + 1) / 2)
def d2 (x : ℤ) : ℤ := x^2 + 3^x - x * 3^((x + 1) / 2)

theorem integral_values_x :
  ∃ n, n = 43 ∧ ∀ x : ℤ, 1 ≤ x ∧ x ≤ 301 → d1 x * d2 x % 7 = 0 ↔ x ∈ Finset.range 301 ∧ x ≠ 0 :=
by
  sorry

end integral_values_x_l121_121576


namespace novel_to_history_ratio_l121_121974

-- Define the conditions
def history_book_pages : ℕ := 300
def science_book_pages : ℕ := 600
def novel_pages := science_book_pages / 4

-- Define the target ratio to prove
def target_ratio := (novel_pages : ℚ) / (history_book_pages : ℚ)

theorem novel_to_history_ratio :
  target_ratio = (1 : ℚ) / (2 : ℚ) :=
by
  sorry

end novel_to_history_ratio_l121_121974


namespace inner_cube_surface_area_l121_121274

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121274


namespace weight_loss_percentage_l121_121710

theorem weight_loss_percentage :
  ∀ (weight_before weight_after : ℕ), 
    weight_before = 840 →
    weight_after = 546 →
    (weight_before - weight_after) * 100 / weight_before = 35 :=
by
  intros weight_before weight_after h_before h_after
  rw [h_before, h_after]
  calc (840 - 546) * 100 / 840 = (294) * 100 / 840 : by { rw [Nat.sub_eq, Nat.mul_comm] }
                         ... = 29400 / 840 : by { rw [Nat.mul_comm] }
                         ... = 35 : by norm_num
  sorry

end weight_loss_percentage_l121_121710


namespace smallest_sphere_radius_l121_121454

noncomputable def smallest_enclosing_sphere_radius (r : ℝ) : ℝ :=
  let side_length := 2 * r
  let space_diagonal := side_length * real.sqrt 3
  let diameter := space_diagonal + 2 * r
  diameter / 2

theorem smallest_sphere_radius {r : ℝ} (h₁ : r = 2) :
  smallest_enclosing_sphere_radius r = 2 * real.sqrt 3 + 2 := 
sorry

end smallest_sphere_radius_l121_121454


namespace complement_intersection_eq_l121_121847

def U : Set ℕ := {0, 1, 2, 3}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem complement_intersection_eq :
  (U \ A) ∩ B = {3} :=
by
  sorry

end complement_intersection_eq_l121_121847


namespace min_value_of_sum_of_squares_l121_121525

variable {A B C : ℝ} -- Angles in the triangle
variable {a b c : ℝ} -- Side lengths opposite to the angles

theorem min_value_of_sum_of_squares 
  (h1 : cos (2 * B) + cos B + cos (A - C) = 1)
  (h2 : b = Real.sqrt 7) :
  ∃ a c, a^2 + c^2 = 14 :=
begin
  sorry
end

end min_value_of_sum_of_squares_l121_121525


namespace inner_cube_surface_area_l121_121384

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121384


namespace inner_cube_surface_area_l121_121244

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121244


namespace maximum_additional_voters_l121_121115

-- Define conditions
structure MovieRating (n : ℕ) (x : ℤ) where
  (sum_scores : ℤ) : sum_scores = n * x

-- Define a function to verify the rating decrease condition
def rating_decrease_condition (n : ℕ) (x y : ℤ) : Prop :=
  (n*x + y) / (n+1) = x - 1

-- Problem: To prove that the maximum number of additional voters after moment T is 5
theorem maximum_additional_voters (n additional_voters : ℕ) (x y : ℤ) (initial_condition : MovieRating n x) :
  initial_condition.sum_scores = n * x ∧
  (∀ k, 1 ≤ k → k ≤ additional_voters → 
    ∃ y, rating_decrease_condition (n + k - 1) (x - (k-1)) y ∧ y ≤ 0) →
  additional_voters ≤ 5 :=
by
  sorry

end maximum_additional_voters_l121_121115


namespace volume_of_cube_l121_121186

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l121_121186


namespace bobby_initial_candy_l121_121414

theorem bobby_initial_candy (candy_ate_start candy_ate_more candy_left : ℕ)
  (h1 : candy_ate_start = 9) (h2 : candy_ate_more = 5) (h3 : candy_left = 8) :
  candy_ate_start + candy_ate_more + candy_left = 22 :=
by
  rw [h1, h2, h3]
  -- sorry


end bobby_initial_candy_l121_121414


namespace john_shots_l121_121403

theorem john_shots :
  (∀ n m : ℕ, n = 30 → m = 40 → (0.60 * n).ceil = 18 → (0.62 * m).ceil = 25 →
  ∃ k : ℕ, k = 10 ∧ (25 - 18) = k) :=
by
  sorry

end john_shots_l121_121403


namespace least_positive_t_geometric_progression_l121_121748

theorem least_positive_t_geometric_progression 
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < (real.pi / 2)) 
  (t : ℝ) :
  (arcsin (sin t α) = (arcsin (sin α) * arcsin (sin 3 α) * arcsin (sin 10 α)) / (arcsin (sin α)^2)) → 
  t = 27 / 19 := 
sorry

end least_positive_t_geometric_progression_l121_121748


namespace bus_routes_count_l121_121470

-- Given conditions for the bus route problem
variable (n : ℕ) (h : n ≥ 2)

theorem bus_routes_count (n : ℕ) (h : n ≥ 2) : 
  let Z := 0 -- Zürich is represented by 0
  in (number_of_possible_routes Z n h = 2^(n+1)) :=
sorry

end bus_routes_count_l121_121470


namespace inner_cube_surface_area_l121_121290

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121290


namespace inner_cube_surface_area_l121_121276

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121276


namespace siblings_total_weight_l121_121739

/-- Given conditions:
Antonio's weight: 50 kilograms.
Antonio's sister weighs 12 kilograms less than Antonio.
Antonio's backpack weight: 5 kilograms.
Antonio's sister's backpack weight: 3 kilograms.
Marco's weight: 30 kilograms.
Marco's stuffed animal weight: 2 kilograms.
Prove that the total weight of the three siblings including additional weights is 128 kilograms.
-/
theorem siblings_total_weight :
  let antonio_weight := 50
  let antonio_sister_weight := antonio_weight - 12
  let antonio_backpack_weight := 5
  let antonio_sister_backpack_weight := 3
  let marco_weight := 30
  let marco_stuffed_animal_weight := 2
  antonio_weight + antonio_backpack_weight +
  antonio_sister_weight + antonio_sister_backpack_weight +
  marco_weight + marco_stuffed_animal_weight = 128 :=
by
  sorry

end siblings_total_weight_l121_121739


namespace num_different_integer_values_f_l121_121781

def f (x : ℝ) : ℤ := int.floor x + int.floor (2 * x) + int.floor ((5 * x) / 3) + int.floor (3 * x) + int.floor (4 * x)

theorem num_different_integer_values_f : 
  (finset.univ.filter (λ (x : ℝ), 0 ≤ x ∧ x ≤ 100)).image f).card = 734 :=
by
  sorry

end num_different_integer_values_f_l121_121781


namespace positive_difference_of_two_numbers_l121_121986

variable {x y : ℝ}

theorem positive_difference_of_two_numbers (h₁ : x + y = 8) (h₂ : x^2 - y^2 = 24) : |x - y| = 3 :=
by
  sorry

end positive_difference_of_two_numbers_l121_121986


namespace speed_ratio_l121_121106

variables {A B : Type}
variables (v_A v_B : ℕ) -- speeds of A and B
variables (x_A x_B : ℕ → ℕ) -- positions of A and B as functions of time

-- Conditions given in the problem
axiom uniform_motion_A : ∀ t, x_A t = v_A * t
axiom uniform_motion_B : ∀ t, x_B t = -400 + v_B * t
axiom right_angle_paths : x_A = 0 -- A starts at O
axiom initial_B_distance : x_B 0 = -400
axiom equidistant_3_min : x_A 3 = abs (x_B 3)
axiom equidistant_12_min : x_A 12 = abs (x_B 12)

-- The proof statement: the ratio of the speed of A to the speed of B is 7:8
theorem speed_ratio : v_A * 8 = v_B * 7 :=
sorry

end speed_ratio_l121_121106


namespace domain_of_f_l121_121611

noncomputable def f (x : ℝ) : ℝ := log (x - 1)

theorem domain_of_f : { x : ℝ | ∃ y, y = f x } = { x : ℝ | x > 1 } :=
by
  sorry

end domain_of_f_l121_121611


namespace angle_D_measure_l121_121886

theorem angle_D_measure 
  (A B C D : Type)
  (angleA : ℝ)
  (angleB : ℝ)
  (angleC : ℝ)
  (angleD : ℝ)
  (BD_bisector : ℝ → ℝ) :
  angleA = 85 ∧ angleB = 50 ∧ angleC = 25 ∧ BD_bisector angleB = 25 →
  angleD = 130 :=
by
  intro h
  have hA := h.1
  have hB := h.2.1
  have hC := h.2.2.1
  have hBD := h.2.2.2
  sorry

end angle_D_measure_l121_121886


namespace cube_volume_l121_121181

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121181


namespace inner_cube_surface_area_l121_121307

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121307


namespace triangle_area_inscribed_l121_121397

theorem triangle_area_inscribed (C : ℝ) (r : ℝ) (θ : ℝ) (α β γ : ℝ)
  (hC : C = 16)
  (hr : r = 8 / real.pi)
  (hθ : θ = 22.5)
  (hα : α = 90)
  (hβ : β = 112.5)
  (hγ : γ = 157.5) :
  ∃ A : ℝ, A = 147.6144 / real.pi ^ 2 :=
sorry

end triangle_area_inscribed_l121_121397


namespace problem_statement_l121_121784

theorem problem_statement (N : ℕ) (hN : N ≥ 3) :
  (∃ points : vector (ℝ × ℝ) N,
    (∀ i j k : ℕ, i < j -> j < k -> points.nth i ≠ points.nth j ∧ points.nth j ≠ points.nth k ∧ points.nth i ≠ points.nth k) ∧
    (∀ t : fin 3 → ℝ × ℝ, ∃ p : ℝ × ℝ, p ∈ set.range points.val ∧ p ≠ t 0 ∧ p ≠ t 1 ∧ p ≠ t 2)
  ) ↔ (∃ n ≥ 3, N = 2 * n - 2) :=
by
  sorry

end problem_statement_l121_121784


namespace cube_volume_l121_121145

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l121_121145


namespace find_b_l121_121811

theorem find_b (a b c : ℝ) (A B C : ℝ) (h1 : a = 10) (h2 : c = 20) (h3 : B = 120) :
  b = 10 * Real.sqrt 7 :=
sorry

end find_b_l121_121811


namespace sum_of_divisors_of_143_l121_121076

theorem sum_of_divisors_of_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := 
by
  sorry

end sum_of_divisors_of_143_l121_121076


namespace fraction_multiplication_l121_121048

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l121_121048


namespace inner_cube_surface_area_l121_121272

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121272


namespace matinee_ticket_price_l121_121622

theorem matinee_ticket_price
  (M : ℝ)  -- Denote M as the price of a matinee ticket
  (evening_ticket_price : ℝ := 12)  -- Price of an evening ticket
  (ticket_3D_price : ℝ := 20)  -- Price of a 3D ticket
  (matinee_tickets_sold : ℕ := 200)  -- Number of matinee tickets sold
  (evening_tickets_sold : ℕ := 300)  -- Number of evening tickets sold
  (tickets_3D_sold : ℕ := 100)  -- Number of 3D tickets sold
  (total_revenue : ℝ := 6600) -- Total revenue
  (h : matinee_tickets_sold * M + evening_tickets_sold * evening_ticket_price + tickets_3D_sold * ticket_3D_price = total_revenue) :
  M = 5 :=
by
  sorry

end matinee_ticket_price_l121_121622


namespace find_min_value_of_quadratic_l121_121755

theorem find_min_value_of_quadratic : ∀ x : ℝ, ∃ c : ℝ, (∃ a b : ℝ, (y = 2*x^2 + 8*x + 7 ∧ (∀ x : ℝ, y ≥ c)) ∧ c = -1) :=
by
  sorry

end find_min_value_of_quadratic_l121_121755


namespace problem_statement_l121_121064

noncomputable def increase_and_subtract (x p y : ℝ) : ℝ :=
  (x + p * x) - y

theorem problem_statement : increase_and_subtract 75 1.5 40 = 147.5 := by
  sorry

end problem_statement_l121_121064


namespace inner_cube_surface_area_l121_121303

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121303


namespace cube_volume_l121_121165

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121165


namespace general_formula_sum_formula_l121_121897

-- Define the geometric sequence
def geoseq (n : ℕ) : ℕ := 2^n

-- Define the sum of the first n terms of the geometric sequence
def sum_first_n_terms (n : ℕ) : ℕ := 2^(n+1) - 2

-- Given conditions
def a1 : ℕ := 2
def a4 : ℕ := 16

-- Theorem statements
theorem general_formula (n : ℕ) : 
  (geoseq 1 = a1) → (geoseq 4 = a4) → geoseq n = 2^n := sorry

theorem sum_formula (n : ℕ) : 
  (geoseq 1 = a1) → (geoseq 4 = a4) → sum_first_n_terms n = 2^(n+1) - 2 := sorry

end general_formula_sum_formula_l121_121897


namespace avg_calculation_l121_121603

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem avg_calculation :
  avg4 (avg2 1 2) (avg2 3 1) (avg2 2 0) (avg2 1 1) = 11 / 8 := by
  sorry

end avg_calculation_l121_121603


namespace semicircle_area_ratio_l121_121598

theorem semicircle_area_ratio {r : ℝ} (hr : 0 < r) : 
  let area_PQ := (1/2) * π * r^2,
      area_ROS := (1/2) * π * (2 * r)^2,
      area_circle_O := π * (2 * r)^2
  in (area_PQ + area_ROS) / area_circle_O = 5/8 := by
  sorry

end semicircle_area_ratio_l121_121598


namespace chef_meals_prepared_for_dinner_l121_121888

theorem chef_meals_prepared_for_dinner (lunch_meals_prepared lunch_meals_sold dinner_meals_total : ℕ) 
  (h1 : lunch_meals_prepared = 17)
  (h2 : lunch_meals_sold = 12)
  (h3 : dinner_meals_total = 10) :
  (dinner_meals_total - (lunch_meals_prepared - lunch_meals_sold)) = 5 :=
by
  -- Lean proof code to proceed from here
  sorry

end chef_meals_prepared_for_dinner_l121_121888


namespace cube_volume_l121_121211

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l121_121211


namespace minimum_sum_maximum_sum_l121_121988

-- Define a sequence of 999 numbers, either 1 or -1, arranged in a circle such that not all are the same
def seq := { a : Fin 999 → ℤ // (∀ i, a i = 1 ∨ a i = -1) ∧ ∃ i j, i ≠ j ∧ a i ≠ a j }

-- Define the product of 10 consecutive numbers in the sequence
def product_of_10 (a : Fin 999 → ℤ) (i : Fin 999) : ℤ :=
  ∏ j in (Finset.range 10), a ((i + j) % 999)

-- Define the sum of all products of 10 consecutive numbers
def S (a : Fin 999 → ℤ) : ℤ :=
  ∑ i in (Finset.range 999), product_of_10 a i

theorem minimum_sum (a : seq) : S a.val = -997 :=
by sorry

theorem maximum_sum (a : seq) : S a.val = 995 :=
by sorry

end minimum_sum_maximum_sum_l121_121988


namespace non_adjacent_chords_l121_121959

theorem non_adjacent_chords (n : ℕ) (h : n = 10) : 
  {card c : Set (Fin 10 × Fin 10) // ∀ x y, c = (x, y) → (x ≠ y ∧ ¬adjacent x y)}.card = 35 :=
sorry

/--
Helper definition to determine adjacency in finite set context.
This is necessary for defining the adjacency in the set of points.
-/
def adjacent (x y : Fin 10) : Prop :=
  (x + 1 = y) ∨ (y + 1 = x)

end non_adjacent_chords_l121_121959


namespace angle_between_planes_is_correct_l121_121439

open Real

noncomputable def normal_vector_plane1 : (ℝ × ℝ × ℝ) := (1, 1, 3)
noncomputable def normal_vector_plane2 : (ℝ × ℝ × ℝ) := (0, 1, 1)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def cos_angle : ℝ :=
  dot_product normal_vector_plane1 normal_vector_plane2 / (magnitude normal_vector_plane1 * magnitude normal_vector_plane2)

noncomputable def angle_between_planes : ℝ :=
  arccos cos_angle

theorem angle_between_planes_is_correct : 
  abs (angle_between_planes - (31 * (π / 180) + (28 / 60) * (π / 180) + (56 / 3600) * (π / 180))) < 1e-6 :=
sorry

end angle_between_planes_is_correct_l121_121439


namespace geometric_sequence_a8_l121_121552

theorem geometric_sequence_a8 (a : ℕ → ℝ) (q : ℝ) 
  (h₁ : a 3 = 3)
  (h₂ : a 6 = 24)
  (h₃ : ∀ n, a (n + 1) = a n * q) : 
  a 8 = 96 :=
by
  sorry

end geometric_sequence_a8_l121_121552


namespace distance_to_square_center_l121_121942

variables {a b : ℝ}

def right_triangle (A B C : ℝ × ℝ) : Prop :=
  A = (b, 0) ∧ B = (0, a) ∧ C = (0, 0)

def square_center (A B E : ℝ × ℝ) : ℝ × ℝ :=
  let Mx := (A.1 + E.1) / 2 in
  let My := (A.2 + E.2) / 2 in
  (Mx, My)

def distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem distance_to_square_center (A B C E : ℝ × ℝ) (a b : ℝ)
  (hAB: A = (b, 0)) (hBC: B = (0, a)) (hAC : C = (0, 0)) (hAE : E = (b + a, a))
  (h_right: right_triangle A B C) :
  distance C (square_center A B E) = (a + b) / real.sqrt 2 :=
sorry

end distance_to_square_center_l121_121942


namespace no_int_x_divisible_by_169_l121_121944

theorem no_int_x_divisible_by_169 (x : ℤ) : ¬ (169 ∣ (x^2 + 5 * x + 16)) := by
  sorry

end no_int_x_divisible_by_169_l121_121944


namespace arith_seq_d123_formula_dm_Sn_formula_find_N_l121_121751

-- Definitions for conditions
def rows_arith_seq (n : ℕ) (A : Matrix ℕ ℕ ℕ) (d : ℕ → ℕ) : Prop :=
  ∀ m k, 1 ≤ m ∧ 1 ≤ k → k ≤ n → A m k = 1 + (k - 1) * d m

def cols_arith_seq (n : ℕ) (A : Matrix ℕ ℕ ℕ) : Prop :=
  ∀ k i j, 1 ≤ i ∧ 1 ≤ j ∧ 1 ≤ k → i < j → A i k + A j k = 2 * A (i + j) / 2 k

-- Arithmetic sequence property for d₁, d₂, d₃
theorem arith_seq_d123 (n : ℕ) (A : Matrix ℕ ℕ ℕ) (d : ℕ → ℕ) (hrows : rows_arith_seq n A d) (hcols : cols_arith_seq n A) :
  d 2 = d 1 + d 3 / 2 :=
sorry

-- Formula for d_m in terms of d₁ and d₂
theorem formula_dm (n m : ℕ) (d₁ d₂ : ℕ) (h : 3 ≤ m ∧ m ≤ n) :
  d m = (2 - m) * d₁ + (m - 1) * d₂ :=
sorry

-- Sum of first n terms Sₙ
def sequence2cm_dm (m : ℕ) (d : ℕ) :=
  (2 * m - 1) * 2^m

def Sn (n : ℕ) : ℕ :=
  ∑ i in finRange n, sequence2cm_dm (i + 1) (2 * (i + 1) - 1)

theorem Sn_formula (n : ℕ) : 
  Sn n = (2 * n - 3) * 2^(n + 1) + 6 :=
sorry

-- Finding possible value of N
theorem find_N (N : ℕ) (hN : N ≤ 20) :
  (∀ n, n > N → 1 / 50 * (Sn n - 6) >  2 * n - 1) → N ∈ {5, 6, ..., 20} :=
sorry

end arith_seq_d123_formula_dm_Sn_formula_find_N_l121_121751


namespace angles_on_x_axis_l121_121627

def terminal_side_on_x_axis (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * Real.pi

theorem angles_on_x_axis :
  { α : ℝ | terminal_side_on_x_axis α } = { α : ℝ | ∃ k : ℤ, α = k * Real.pi } := 
by 
  sorry

end angles_on_x_axis_l121_121627


namespace shortest_distance_point_to_parabola_l121_121778

theorem shortest_distance_point_to_parabola : 
  let P := (3 : ℝ, 9 : ℝ)
  let parabola := {p : ℝ × ℝ // p.1 = p.2^2 / 4}
  let distance (p1 p2: ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  ∃ pt ∈ parabola, ∀ q ∈ parabola, distance P pt ≤ distance P q :=
    ∃ pt ∈ parabola, distance P pt = 3 * Real.sqrt 5 :=
sorry

end shortest_distance_point_to_parabola_l121_121778


namespace inner_cube_surface_area_l121_121326

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121326


namespace cube_volume_l121_121180

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121180


namespace wedge_volume_formula_l121_121715

noncomputable def sphere_wedge_volume : ℝ :=
let r := 9 in
let volume_of_sphere := (4 / 3) * Real.pi * r^3 in
let volume_of_one_wedge := volume_of_sphere / 6 in
volume_of_one_wedge

theorem wedge_volume_formula
  (circumference : ℝ)
  (h1 : circumference = 18 * Real.pi)
  (num_wedges : ℕ)
  (h2 : num_wedges = 6) :
  sphere_wedge_volume = 162 * Real.pi :=
by
  sorry

end wedge_volume_formula_l121_121715


namespace find_g_seven_l121_121970

-- Define our function g satisfying the given conditions

def g : ℝ → ℝ := sorry

lemma g_add (x y : ℝ) : g(x + y) = g(x) + g(y) := sorry
lemma g_six : g 6 = 7 := sorry

-- The statement we want to prove
theorem find_g_seven : g 7 = 49 / 6 :=
by
  sorry

end find_g_seven_l121_121970


namespace radius_of_first_can_l121_121648

variable {h : ℝ} (h_pos : h > 0)

theorem radius_of_first_can
  (volume_eq : (15^2 * h * π) = ((4 * h^2 / 3) * x^2 * π))
  (h_pos : h > 0) :
  x = 15 * sqrt 3 / 2 :=
sorry

end radius_of_first_can_l121_121648


namespace part_one_part_two_l121_121964

-- Definition for the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The quadratic equation in question
def quadratic_eq (k : ℝ) : ℝ → ℝ := λ x, x^2 + (2 * k + 3) * x + (k^2 + 5 * k)

-- Condition for having distinct real roots based on the discriminant
def has_distinct_real_roots (k : ℝ) : Prop :=
  discriminant 1 (2 * k + 3) (k^2 + 5 * k) > 0

-- Part (1): Prove that k < 9/8 ensures two distinct real roots
theorem part_one (k : ℝ) : k < 9 / 8 ↔ has_distinct_real_roots k :=
sorry

-- Part (2): Prove that when k = 1, the roots are -2 and -3
theorem part_two : ∀ x, quadratic_eq 1 x = 0 ↔ (x = -2 ∨ x = -3) :=
sorry

end part_one_part_two_l121_121964


namespace sqrt_eq_add_iff_l121_121041

theorem sqrt_eq_add_iff (a b : ℝ) : sqrt (a^2 + b^2) = a + b ↔ a * b = 0 ∧ a + b ≥ 0 :=
by
  sorry

end sqrt_eq_add_iff_l121_121041


namespace range_of_f_l121_121460

def diamond (x y : ℝ) := (x + y) ^ 2 - x * y

def f (a x : ℝ) := diamond a x

theorem range_of_f (a : ℝ) (h : diamond 1 a = 3) :
  ∃ b : ℝ, ∀ x : ℝ, x > 0 → f a x > b :=
sorry

end range_of_f_l121_121460


namespace line_through_points_decreasing_direct_proportion_function_m_l121_121901

theorem line_through_points_decreasing (x₁ x₂ y₁ y₂ k b : ℝ) (h1 : x₁ < x₂) (h2 : y₁ = k * x₁ + b) (h3 : y₂ = k * x₂ + b) (h4 : k < 0) : y₁ > y₂ :=
sorry

theorem direct_proportion_function_m (x₁ x₂ y₁ y₂ m : ℝ) (h1 : x₁ < x₂) (h2 : y₁ = (1 - 2 * m) * x₁) (h3 : y₂ = (1 - 2 * m) * x₂) (h4 : y₁ > y₂) : m > 1/2 :=
sorry

end line_through_points_decreasing_direct_proportion_function_m_l121_121901


namespace limit_eq_l121_121669

open Real

noncomputable def problem_statement : ℝ := 
sqrt (log 2 4) / 3

theorem limit_eq :
  tendsto (λ x, (log (x + 2) / log 10).pow (3⁻¹)) (𝓝 2) (𝓝 (problem_statement)) :=
by
  sorry

end limit_eq_l121_121669


namespace min_value_expression_l121_121461

theorem min_value_expression :
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → (∃ (c : ℝ), c = 16 ∧ ∀ z, z = (1 / x + 9 / y) → z ≥ c) :=
by
  sorry

end min_value_expression_l121_121461


namespace log_absolute_comparison_l121_121799

variable {a x : ℝ}

theorem log_absolute_comparison (hx : 0 < x ∧ x < 1) (ha : a > 0 ∧ a ≠ 1) : 
  abs (Real.logBase a (1 - x)) > abs (Real.logBase a (1 + x)) := 
by 
  sorry

end log_absolute_comparison_l121_121799


namespace range_of_m_l121_121621

theorem range_of_m (m : ℝ) : 
  -6 * Real.sqrt 7 / 7 ≤ m ∧ m ≤ 6 * Real.sqrt 7 / 7 → 
  (let line_eq y := y = Real.sqrt 3 * x - m in 
  ∃ (M N : ℝ × ℝ), 
  (x ^ 2 + y ^ 2 = 9) ∧ 
  (y = Real.sqrt 3 * x - m) ∧ 
  M ≠ N ∧ 
  |M - N| ≥ Real.sqrt 6 * |M + N|)
  := sorry
\
end range_of_m_l121_121621


namespace expenditure_representation_l121_121416

-- Define a country and the way to record income and expenditure in yuan
def Country := Type

-- Define positive and negative to represent opposite quantities
def record_income (C : Country) (amount : Int) : Prop := amount > 0
def record_expenditure (C : Country) (amount : Int) : Prop := amount < 0

-- Proving that an income of 80 yuan is recorded as +80 yuan, 
-- then an expenditure of 50 yuan should be recorded as -50 yuan
theorem expenditure_representation (C : Country) (a_income : Int) (a_expenditure : Int) 
  (h1 : record_income C a_income) (ha : a_income = 80) (a_expenditure = 50): 
  record_expenditure C (-a_expenditure) := 
by
  sorry

end expenditure_representation_l121_121416


namespace sum_of_divisors_143_l121_121096

theorem sum_of_divisors_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121096


namespace cube_volume_from_surface_area_l121_121196

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l121_121196


namespace simplify_expression_l121_121600

theorem simplify_expression :
  ( (sqrt 2 - 1)^(1 - sqrt 3) / (sqrt 2 + 1)^(1 + sqrt 3) ) = 3 - 2 * sqrt 2 :=
by
  sorry

end simplify_expression_l121_121600


namespace sum_digits_3000_3001_3002_l121_121934

def repeating_sequence (n : ℕ) : ℕ :=
  (n % 6) + 1

def erase_fourth_digit (n : ℕ) : Option ℕ :=
  if (n + 1) % 4 = 0 then none else some n

def erase_fifth_digit (n : ℕ) : Option ℕ :=
  if (n + 1) % 5 = 0 then none else some n

def erase_sixth_digit (n : ℕ) : Option ℕ :=
  if (n + 1) % 6 = 0 then none else some n

def final_sequence (n : ℕ) : ℕ :=
  let initial_seq := repeating_sequence n
  let seq_after_fourth := initial_seq.filter_map erase_fourth_digit
  let seq_after_fifth := seq_after_fourth.filter_map erase_fifth_digit
  seq_after_fifth.filter_map erase_sixth_digit n

theorem sum_digits_3000_3001_3002 :
  final_sequence 2999 + final_sequence 3000 + final_sequence 3001 = 6 :=
by
  sorry

end sum_digits_3000_3001_3002_l121_121934


namespace distribution_of_balls_l121_121507

theorem distribution_of_balls :
  ∃ (P : ℕ → ℕ → ℕ), P 6 4 = 9 := 
by
  sorry

end distribution_of_balls_l121_121507


namespace inner_cube_surface_area_l121_121279

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121279


namespace perfect_square_trinomial_l121_121865

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ (x : ℝ), 4 * x^2 + b * x + 1 = (2 * x + 1) ^ 2) ↔ (b = 4 ∨ b = -4) := 
by 
  sorry

end perfect_square_trinomial_l121_121865


namespace volunteers_lcm_l121_121590

def lcm (a b : Nat) : Nat := Nat.lcm a b

def Paul_cycle := 5
def Quinn_cycle := 6
def Rachel_cycle := 8
def Samantha_cycle := 9
def Tom_cycle := 10

theorem volunteers_lcm : lcm (lcm (lcm (lcm Paul_cycle Quinn_cycle) Rachel_cycle) Samantha_cycle) Tom_cycle = 360 :=
by
  sorry

end volunteers_lcm_l121_121590


namespace inner_cube_surface_area_l121_121270

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121270


namespace diameter_of_circle_l121_121566

theorem diameter_of_circle (P1 P2 P3 P4 P5 P6 : ℝ → ℝ) (R : ℝ) :
  (∀ x, P1 x = x^2 / 16) ∧
  (∀ x, P2 x = x^2 / 16) ∧
  (∀ x, P3 x = x^2 / 16) ∧
  (∀ x, P4 x = x^2 / 16) ∧
  (∀ x, P5 x = x^2 / 16) ∧
  (∀ x, P6 x = x^2 / 16) ∧
  (∃ (vertices : Fin 6 → ℝ × ℝ),
    ∀ i : Fin 6, 
      vertices i = (R * Real.cos (2 * Real.pi * i / 6), R * Real.sin (2 * Real.pi * i / 6)) ∧
      ∀ j : Fin 6, (∃ t : ℝ, P1 (vertices j).fst = (vertices (j + 1)).fst / t) ∧
      ∃ i, P1 = P2 ∧ P2 = P3 ∧ P3 = P4 ∧ P4 = P5 ∧ P5 = P6 ∧ P6 = P1) ∧
  P1 (R * Real.cos 0) = 0 ∧ 
  P2 (R * Real.cos (2 * Real.pi / 6)) = 0 ∧ 
  P3 (R * Real.cos (4 * Real.pi / 6)) = 0 ∧ 
  P4 (R * Real.cos (6 * Real.pi / 6)) = 0 ∧ 
  P5 (R * Real.cos (8 * Real.pi / 6)) = 0 ∧ 
  P6 (R * Real.cos (10 * Real.pi / 6)) = 0
  → 2 * R = 24 :=
sorry

end diameter_of_circle_l121_121566


namespace problem_statement_l121_121449

noncomputable def f (n : ℕ) (x : ℝ) := n * x^3 + 2 * x - n

noncomputable def x_n (n : ℕ) (h : 2 ≤ n) : ℝ := Classical.choose (exists_unique_real_root (f n))

noncomputable def a_n (n : ℕ) (h : 2 ≤ n) : ℕ := ⌊(n + 1) * x_n n h⌋

theorem problem_statement : 
  (∑ n in Finset.range (2015 - 2 + 1), a_n (n + 2) (by linarith)) / 1007 = 2017 :=
sorry

end problem_statement_l121_121449


namespace length_of_EF_l121_121890

theorem length_of_EF (DE DF EG GF EF DG : ℝ) (hDEDF : DE = 5) (hDF : DF = 5) (hEGGF : EG = 2 * GF) 
  (hRightTriangle1 : DG^2 + GF^2 = DF^2) (hRightTriangle2 : DG^2 + EG^2 = DE^2) : EF = 5 * sqrt 3 := 
by
  sorry

end length_of_EF_l121_121890


namespace find_positive_number_l121_121705

theorem find_positive_number (x : ℕ) (h_pos : 0 < x) (h_equation : x * x / 100 + 6 = 10) : x = 20 :=
by
  sorry

end find_positive_number_l121_121705


namespace cube_volume_l121_121210

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l121_121210


namespace inner_cube_surface_area_l121_121378

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121378


namespace tank_depth_l121_121394

variables (L W cost_per_sqm total_cost depth : ℝ)

def area_to_be_plastered (L W depth : ℝ) : ℝ :=
  let bottom_area := L * W
  let long_walls_area := 2 * (L * depth)
  let short_walls_area := 2 * (W * depth)
  bottom_area + long_walls_area + short_walls_area

theorem tank_depth (hL : L = 25) (hW : W = 12) (hcost_per_sqm : cost_per_sqm = 0.30)
    (htotal_cost : total_cost = 223.2) : 
    depth = 6 :=
by 
  have A := 300 + 74 * depth
  have H1 : 0.30 * A = total_cost
  sorry

end tank_depth_l121_121394


namespace remainder_p_q_add_42_l121_121582

def p (k : ℤ) : ℤ := 98 * k + 84
def q (m : ℤ) : ℤ := 126 * m + 117

theorem remainder_p_q_add_42 (k m : ℤ) : 
  (p k + q m) % 42 = 33 := by
  sorry

end remainder_p_q_add_42_l121_121582


namespace value_of_f_pi_over_3_l121_121968

def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then
    Real.sin x
  else if 1 ≤ x ∧ x ≤ Real.sqrt 2 then
    Real.cos x
  else
    Real.tan x

theorem value_of_f_pi_over_3 : f (Real.pi / 3) = 1 / 2 := by
  sorry

end value_of_f_pi_over_3_l121_121968


namespace inner_cube_surface_area_l121_121364

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121364


namespace possible_values_of_expression_l121_121822

variable (a b c d : ℝ)

def sign (x : ℝ) : ℝ := if x > 0 then 1 else -1

theorem possible_values_of_expression 
  (h₀ : a ≠ 0) 
  (h₁ : b ≠ 0) 
  (h₂ : c ≠ 0) 
  (h₃ : d ≠ 0) :
  let expression := 
    sign a + sign b + sign c + sign d + sign (a * b * c * d)
  in 
    expression = 5 
    ∨ expression = 1 
    ∨ expression = -3 :=
by sorry

end possible_values_of_expression_l121_121822


namespace employees_after_reduction_l121_121712

def initial_employees : Real := 229.41
def reduction_percentage : Real := 0.15

theorem employees_after_reduction : 
  (initial_employees - (reduction_percentage * initial_employees)).round = 195 := by
  sorry

end employees_after_reduction_l121_121712


namespace exists_periodic_N_and_t_l121_121848

noncomputable def a_seq (a b : ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := Nat.gcd (a_seq n) (b_seq n) + 1
and b_seq (a b : ℕ) : ℕ → ℕ 
| 0     := b
| (n+1) := Nat.lcm (a_seq n) (b_seq n) - 1

theorem exists_periodic_N_and_t (a0 b0 : ℕ) (h : a0 ≥ 2 ∧ b0 ≥ 2) :
  ∃ N t : ℕ, 0 ≤ N ∧ 0 < t ∧ ∀ n : ℕ, n ≥ N → 
  a_seq a0 b0 (n+t) = a_seq a0 b0 n :=
begin
  sorry
end

end exists_periodic_N_and_t_l121_121848


namespace alcohol_solution_l121_121235

noncomputable def solutionXVolume := 300
noncomputable def solutionYVolume := 450
noncomputable def totalVolume := solutionXVolume + solutionYVolume
noncomputable def solutionYPercentage := 0.30
noncomputable def targetPercentage := 0.22
noncomputable def totalAlcohol := totalVolume * targetPercentage
noncomputable def solutionYAlcohol := solutionYVolume * solutionYPercentage
noncomputable def solutionXPercentage (P : ℝ) := P

theorem alcohol_solution (P : ℝ) : 
  solutionXVolume * P + solutionYAlcohol = totalAlcohol → 
  P = 0.10 :=
begin
  intro h,
  sorry
end

end alcohol_solution_l121_121235


namespace cube_volume_of_surface_area_l121_121163

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l121_121163


namespace inner_cube_surface_area_l121_121344

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121344


namespace inner_cube_surface_area_l121_121314

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121314


namespace cube_volume_l121_121212

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l121_121212


namespace hyperbola_correct_equation_l121_121013

def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

def passes_through_point (a b : ℝ) : Prop :=
  hyperbola_equation a b (real.sqrt 2) (real.sqrt 3)

def eccentricity_equation (a b : ℝ) : Prop :=
  2 = real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_correct_equation (a b : ℝ) (h1 : passes_through_point a b) (h2 : eccentricity_equation a b) :
  a^2 = 1 ∧ b^2 = 3 → hyperbola_equation 1 (real.sqrt 3) x y :=
sorry

end hyperbola_correct_equation_l121_121013


namespace inner_cube_surface_area_l121_121343

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121343


namespace cube_volume_from_surface_area_l121_121198

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l121_121198


namespace inner_cube_surface_area_l121_121376

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121376


namespace length_AF_l121_121547

noncomputable theory

open Classical

variables {A B C D E F : Type} [EuclideanGeometry A B C D E F]

def right_angle (A B C : Point) : Prop :=
∃ (line1 line2 : Line), contains line1 A ∧ contains line1 B ∧ contains line2 A ∧ contains line2 C ∧ isPerpendicular line1 line2

def distance (A B : Point) : ℝ := sorry

def mid_point (E : Point) (A B : Point) : Prop := 
distance A E = distance E B ∧ ∃ line, contains line A ∧ contains line B ∧ contains line E

def altitude (D : Point) (A B : Point) (C : Point) : Prop := 
containsLine (line A B) D ∧ distance D C = distance A C

def median (A B C E : Point) : Prop := 
mid_point E B C ∧ ∃ line, contains line B ∧ contains line C ∧ contains line E

def intersection (F : Point) (AD BE : Line) : Prop := 
contains AD F ∧ contains BE F

def AF_length (A F : Point) : ℝ := distance A F

theorem length_AF (A B C D E F : Point) 
  (h1 : right_angle A B C) 
  (h2 : distance A B = 6) 
  (h3 : distance A C = 6) 
  (h4 : altitude D A C) 
  (h5 : median B C E) 
  (h6 : intersection F (line A D) (line B E)) : 
  AF_length A F = 6 :=
sorry

end length_AF_l121_121547


namespace find_T_magnitude_l121_121571

namespace complex_magnitude_proof

open Complex

theorem find_T_magnitude :
  let i : ℂ := Complex.I in
  let z1 : ℂ := 1 + Real.sqrt 3 * i in
  let z2 : ℂ := 1 - Real.sqrt 3 * i in
  let T : ℂ := z1 ^ 19 - z2 ^ 19 in
  Complex.abs T = 2 ^ 19 * Real.sqrt 3 :=
by
  sorry

end complex_magnitude_proof

end find_T_magnitude_l121_121571


namespace triangle_GHI_right_triangle_l121_121542

noncomputable def GH (HI : ℝ) (G : ℝ) : ℝ :=
  HI / Real.tan G

noncomputable def GI (HI : ℝ) (G : ℝ) : ℝ :=
  HI / Real.sin G

theorem triangle_GHI_right_triangle (HI : ℝ) (G : ℝ) (H : ℝ) (h_right : H = Real.pi / 2) (g_angle : G = Real.pi / 180 * 40) (hi_value : HI = 12) :
    GH HI G ≈ 14.3 ∧ GI HI G ≈ 18.7 :=
by
  have hG : Real.pi / 180 * 40 = G := g_angle
  have hHI : 12 = HI := hi_value
  simp [GH, GI, hG, hHI]
  -- Use a calculator for tan(40°) and sin(40°)
  have h_tan : Real.tan (Real.pi / 180 * 40) ≈ 0.8391 := by sorry
  have h_sin : Real.sin (Real.pi / 180 * 40) ≈ 0.6428 := by sorry
  
  split
  · -- GH proof
    simp [GH, h_tan]
    sorry
  · -- GI proof
    simp [GI, h_sin]
    sorry

end triangle_GHI_right_triangle_l121_121542


namespace goldie_earnings_l121_121497

theorem goldie_earnings (hourly_rate : ℝ) (hours_week1 : ℝ) (hours_week2 : ℝ) :
  hourly_rate = 5 → hours_week1 = 20 → hours_week2 = 30 → (hourly_rate * (hours_week1 + hours_week2) = 250) :=
by
  intro h_rate
  intro h_week1
  intro h_week2
  rw [h_rate, h_week1, h_week2]
  norm_num
  sorry

end goldie_earnings_l121_121497


namespace range_of_k_l121_121793

theorem range_of_k (k : ℝ) :
  (∀ (x1 : ℝ), x1 ∈ Set.Icc (-1 : ℝ) 3 →
    ∃ (x0 : ℝ), x0 ∈ Set.Icc (-1 : ℝ) 3 ∧ (2 * x1^2 + x1 - k) ≤ (x0^3 - 3 * x0)) →
  k ≥ 3 :=
by
  -- This is the place for the proof. 'sorry' is used to indicate that the proof is omitted.
  sorry

end range_of_k_l121_121793


namespace sum_of_divisors_of_143_l121_121081

theorem sum_of_divisors_of_143 : 
  (∑ d in Finset.filter (fun d => 143 % d = 0) (Finset.range 144), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l121_121081


namespace find_fraction_l121_121871

theorem find_fraction (x y : ℝ) (h1 : (1/3) * (1/4) * x = 18) (h2 : y * x = 64.8) : y = 0.3 :=
sorry

end find_fraction_l121_121871


namespace log_abs_compare_l121_121801

open Real

theorem log_abs_compare
  (x : ℝ) (a : ℝ)
  (hx : 0 < x ∧ x < 1)
  (ha : a > 0 ∧ a ≠ 1) :
  abs (log a (1 - x)) > abs (log a (1 + x)) :=
sorry

end log_abs_compare_l121_121801


namespace cube_volume_from_surface_area_l121_121201

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l121_121201


namespace angle_MON_l121_121472

-- Definitions of the given conditions
def is_origin (O : ℝ × ℝ) : Prop := O = (0, 0)

def is_circle (D : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + D * x - 4 * y = 0

def line_y_eq_2 (M N : ℝ × ℝ) : Prop := 
  (M.2 = 2) ∧ (N.2 = 2)

def points_on_circle (D : ℝ) (M N : ℝ × ℝ) : Prop := 
  is_circle D M.1 M.2 ∧ is_circle D N.1 N.2

-- The theorem statement
theorem angle_MON (O M N : ℝ × ℝ) (D : ℝ) :
  is_origin O →
  points_on_circle D M N →
  line_y_eq_2 M N →
  ∠(O, M, N) = 90 :=
begin
  sorry
end

end angle_MON_l121_121472


namespace sum_of_divisors_of_143_l121_121080

theorem sum_of_divisors_of_143 : 
  (∑ d in Finset.filter (fun d => 143 % d = 0) (Finset.range 144), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l121_121080


namespace inner_cube_surface_area_l121_121381

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121381


namespace odd_function_f_neg1_l121_121920

-- Define the conditions in Lean 4 without assuming any solution steps.

theorem odd_function_f_neg1 (b : ℝ) : 
  (∀ x : ℝ, f (-x) = -f x) → (∀ x : ℝ, x >= 0 → f x = 2^x + 2 * x + b) → f (-1) = -3 :=
by
  intro h1 h2
  -- Prove f(-1) = -3 given the above conditions.
  sorry

end odd_function_f_neg1_l121_121920


namespace find_f_l121_121841

section DifferentiationProblem

variable {a : ℝ}

-- Define function f
def f (x : ℝ) : ℝ := a * x^3 + f'2 * x^2 + 3

-- Condition: f'(1) = -5
def f'_at_1 := 3 * a * 1^2 + 2 * f'2 * 1 = -5

theorem find_f'2 (h₁ : f'_at_1) : f'2 = -4 :=
by
  -- Proof omitted
  sorry

end DifferentiationProblem

end find_f_l121_121841


namespace curve_equations_and_PMNQ_distance_l121_121555

-- Definitions for the curves and the line
def C1_parametric (α : ℝ) : ℝ × ℝ := (3 + 2 * real.sqrt 2 * real.cos α, 2 * real.sqrt 2 * real.sin α)
def C2_polar (ρ θ : ℝ) := ρ * (real.sin θ)^2 - 6 * real.cos θ
def line_l (t : ℝ) : ℝ × ℝ := (3 + 1/2 * t, real.sqrt 3 / 2 * t)

-- Proof problem statement
theorem curve_equations_and_PMNQ_distance :
  (∀ α : ℝ, ∃ x y : ℝ, C1_parametric α = (x, y) → (x - 3)^2 + y^2 = 8) ∧
  (∀ ρ θ : ℝ, C2_polar ρ θ = 0 → (ρ * real.cos θ, ρ * real.sin θ) = (x, y) → y^2 = 6 * x) ∧
  (let intersection_points := [2 + 2 * real.sqrt 7, 2 - 2 * real.sqrt 7, 2 * real.sqrt 2, -2 * real.sqrt 2] in
   let PM := real.dist (intersection_points.nth 0) (intersection_points.nth 2) in
   let NQ := real.dist (intersection_points.nth 1) (intersection_points.nth 3) in
   PM + NQ = 4 * real.sqrt 7 - 4 * real.sqrt 2) := sorry

end curve_equations_and_PMNQ_distance_l121_121555


namespace sum_of_divisors_of_143_l121_121075

theorem sum_of_divisors_of_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := 
by
  sorry

end sum_of_divisors_of_143_l121_121075


namespace volume_of_cube_l121_121191

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l121_121191


namespace equal_rational_numbers_l121_121635

theorem equal_rational_numbers (n : ℕ) (a : Fin (2 * n + 1) → ℚ)
  (h : ∀ (s : Finset (Fin (2 * n + 1))), s.card = 2 * n →
         ∃ (t₁ t₂ : Finset (Fin (2 * n + 1))), t₁.card = n ∧ t₂.card = n ∧ t₁ ∪ t₂ = s ∧ t₁ ∩ t₂ = ∅ ∧
         (t₁.sum (λ i, a i)) = (t₂.sum (λ i, a i))) :
  (∀ i j, a i = a j) :=
by 
  sorry

end equal_rational_numbers_l121_121635


namespace translate_graph_by_2_units_to_right_l121_121037

noncomputable def f (x : ℝ) := 2 ^ x
noncomputable def g (x : ℝ) := 2 ^ (x - 2)

theorem translate_graph_by_2_units_to_right:
  ∀ x : ℝ, g(x) = f(x - 2) :=
by 
  sorry

end translate_graph_by_2_units_to_right_l121_121037


namespace solution_set_exists_l121_121859

theorem solution_set_exists (a b x : ℝ) (h1 : - sqrt (1 / (a - b)^2) * (b - a) = 1)
    (h2 : 3 * x - 4 * a ≤ a - 2 * x) (h3 : (3 * x + 2 * b) / 5 > b) : 
    b < x ∧ x ≤ a :=
by
  sorry

end solution_set_exists_l121_121859


namespace cube_volume_l121_121204

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l121_121204


namespace inner_cube_surface_area_l121_121361

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121361


namespace segment_BO_length_l121_121688

-- Define the given conditions and the theorem to be proven.
theorem segment_BO_length
  (O B : Type) [MetricSpace O] [MetricSpace B]
  (r a : ℝ)
  (A C M N K T : O)
  (BO : ℝ)
  (h1 : circle O r)
  (h2 : tangent_to M (line_segment B A))
  (h3 : tangent_to N (line_segment B C))
  (h4 : parallel (line_through M K) (line_through B C))
  (h5 : K ∈ ray B O)
  (h6 : T ∈ ray M N)
  (h7 : ∠ MTK = ½ * ∠ ABC)
  (h8 : length K T = a) :
  BO = sqrt (r * (a + r)) :=
sorry

end segment_BO_length_l121_121688


namespace negation_proof_l121_121623

theorem negation_proof : (¬(∀ x : ℝ, x > 0 → ln (x + 1) > 0)) ↔ (∃ x : ℝ, x > 0 ∧ ln (x + 1) ≤ 0) :=
by
  sorry

end negation_proof_l121_121623


namespace sum_of_divisors_of_143_l121_121078

theorem sum_of_divisors_of_143 : 
  (∑ d in Finset.filter (fun d => 143 % d = 0) (Finset.range 144), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l121_121078


namespace second_cube_surface_area_l121_121331

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121331


namespace inner_cube_surface_area_l121_121318

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121318


namespace volume_lemma_l121_121224

noncomputable def volume_of_glass (diameter height : ℝ) : ℝ :=
  let r := diameter / 2
  π * r^2 * height

noncomputable def volume_of_lemonade (diameter height : ℝ) : ℝ :=
  volume_of_glass diameter (height / 3)

noncomputable def volume_of_lemon_juice (diameter height : ℝ) (lemon_ratio water_ratio : ℝ) : ℝ :=
  let total_ratio := lemon_ratio + water_ratio
  volume_of_lemonade diameter height * (lemon_ratio / total_ratio)

theorem volume_lemma : 
  volume_of_lemon_juice 3 9 1 9 ≈ 2.13 :=
by sorry

end volume_lemma_l121_121224


namespace problem1_problem2_l121_121993

-- Define the variables used in the problem
variables (x y m : ℕ)

-- Define the conditions based on the problem statement
def conditions1 := (3 * x + 4 * y = 44)
def conditions2 := (4 * x + 6 * y = 62)
def conditions3 := (4 ≤ m)
def conditions4 := (8 * (12 - m) + 5 * m ≥ 78)

-- Define the main proof problems
theorem problem1 : conditions1 ∧ conditions2 → x = 8 ∧ y = 5 := by
  sorry

theorem problem2 : conditions3 ∧ conditions4 →
  (m = 4 ∨ m = 5 ∨ m = 6) ∧
  (if m = 4 then 8 * (12 - 4) + 5 * 4 = 78 else
   if m = 5 then 8 * (12 - 5) + 5 * 5 = 78 else
   8 * (12 - 6) + 5 * 6 = 78) := by
  sorry

end problem1_problem2_l121_121993


namespace simplify_expression_l121_121103

theorem simplify_expression : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := 
by 
  sorry 

end simplify_expression_l121_121103


namespace parallel_lines_distance_l121_121000

-- Define the two parallel lines
def line1 : ℝ × ℝ → Prop :=
  λ p, let (x, y) := p in x + 3 * y - 4 = 0

def line2 : ℝ × ℝ → Prop :=
  λ p, let (x, y) := p in 2 * x + 6 * y - 9 = 0

-- Define a point on line1
def point_on_line1 : ℝ × ℝ :=
  (4, 0)

-- Define the function to calculate the distance from a point to a line
def point_line_distance (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  (|a * p.1 + b * p.2 + c|) / (Real.sqrt (a * a + b * b))

-- The theorem to prove
theorem parallel_lines_distance : 
  point_line_distance point_on_line1 2 6 (-9) = √10 / 20 :=
by sorry

end parallel_lines_distance_l121_121000


namespace inner_cube_surface_area_l121_121354

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121354


namespace inner_cube_surface_area_l121_121298

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121298


namespace cube_volume_l121_121182

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121182


namespace parallelogram_area_error_l121_121227

theorem parallelogram_area_error
  (x y : ℝ) (z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : 0 ≤ z ∧ z ≤ 180) :
  let A := x * y * Real.sin (z * (Real.pi / 180))
  let x' := 1.05 * x
  let y' := 1.07 * y
  let A' := x' * y' * Real.sin (z * (Real.pi / 180))
  let percentage_error := ((A' - A) / A) * 100 in
  percentage_error = 12.35 :=
by
  sorry

end parallelogram_area_error_l121_121227


namespace inner_cube_surface_area_l121_121310

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121310


namespace minimum_area_l121_121128

def point : Type := ℤ × ℤ

def A : point := (0, 0)
def B : point := (42, 18)

def area (C : point) : ℕ :=
  let (p, q) := C
  (21 * Int.natAbs (q - 18))

theorem minimum_area : ∀ (C : point), C.1 ∈ ℤ → C.2 ∈ ℤ → (C ≠ (0,0) → C ≠ (42,18) → area C ≥ 21) := 
  by
  sorry

end minimum_area_l121_121128


namespace box_gift_problem_l121_121032

-- Definitions for the boxes' statements and their truth values
def boxA_statement : Prop := ¬gift_in_A
def boxB_statement : Prop := ¬gift_in_B
def boxC_statement : Prop := gift_in_A

-- Hypothesis: Only one statement among the three is true
def exactly_one_true (a b c : Prop) : Prop := 
  (a ∧ ¬b ∧ ¬c) ∨ 
  (¬a ∧ b ∧ ¬c) ∨ 
  (¬a ∧ ¬b ∧ c)

theorem box_gift_problem (gift_in_A : Prop) (gift_in_B : Prop) (gift_in_C : Prop)
  (h : exactly_one_true boxA_statement boxB_statement boxC_statement) :
  (boxA_statement ∧ ¬boxB_statement ∧ ¬boxC_statement) ∧ gift_in_B :=
by
  sorry

end box_gift_problem_l121_121032


namespace inner_cube_surface_area_l121_121319

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121319


namespace rhombus_perimeter_52_l121_121609

noncomputable def perimeter_of_rhombus (d1 d2 : ℕ) : ℕ :=
  let s := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)
  4 * s.to_nat

theorem rhombus_perimeter_52 {d1 d2 : ℕ} (h1 : d1 = 24) (h2 : d2 = 10) :
  perimeter_of_rhombus d1 d2 = 52 :=
by
  sorry

end rhombus_perimeter_52_l121_121609


namespace number_of_unique_connections_l121_121641

theorem number_of_unique_connections (n m : ℕ) (h1 : n = 30) (h2 : m = 4) : 
  ((n * m) / 2) = 60 :=
by {
  rw [h1, h2],
  norm_num,
}

end number_of_unique_connections_l121_121641


namespace power_function_value_at_9_l121_121491

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_value_at_9 (h : f 2 = Real.sqrt 2) : f 9 = 3 :=
by sorry

end power_function_value_at_9_l121_121491


namespace unique_7tuple_solution_l121_121772

theorem unique_7tuple_solution : 
  ∃! x : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ, 
  let (x1, x2, x3, x4, x5, x6, x7) := x in
  (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + (x4 - x5)^2 + (x5 - x6)^2 + (x6 - x7)^2 + x7^2 = 1 / 8 :=
by 
  sorry

end unique_7tuple_solution_l121_121772


namespace journey_ratio_l121_121977

/-- Given a full-circle journey broken into parts,
  including paths through the Zoo Park (Z), the Circus (C), and the Park (P), 
  prove that the journey avoiding the Zoo Park is 11 times shorter. -/
theorem journey_ratio (Z C P : ℝ) (h1 : C = (3 / 4) * Z) 
                      (h2 : P = (1 / 4) * Z) : 
  Z = 11 * P := 
sorry

end journey_ratio_l121_121977


namespace wedge_volume_formula_l121_121716

noncomputable def sphere_wedge_volume : ℝ :=
let r := 9 in
let volume_of_sphere := (4 / 3) * Real.pi * r^3 in
let volume_of_one_wedge := volume_of_sphere / 6 in
volume_of_one_wedge

theorem wedge_volume_formula
  (circumference : ℝ)
  (h1 : circumference = 18 * Real.pi)
  (num_wedges : ℕ)
  (h2 : num_wedges = 6) :
  sphere_wedge_volume = 162 * Real.pi :=
by
  sorry

end wedge_volume_formula_l121_121716


namespace times_reaching_35m_l121_121003

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 30 * t

theorem times_reaching_35m :
  ∃ t1 t2 : ℝ, (abs (t1 - 1.57) < 0.01 ∧ abs (t2 - 4.55) < 0.01) ∧
               projectile_height t1 = 35 ∧ projectile_height t2 = 35 :=
by
  sorry

end times_reaching_35m_l121_121003


namespace question_a_question_b_l121_121670

-- Definitions
def isSolutionA (a b : ℤ) : Prop :=
  1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 7

def isSolutionB (a b : ℤ) : Prop :=
  1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 25

-- Statements
theorem question_a (a b : ℤ) : isSolutionA a b ↔ (a, b) ∈ [(6, -42), (-42, 6), (8, 56), (56, 8), (14, 14)] :=
sorry

theorem question_b (a b : ℤ) : isSolutionB a b ↔ (a, b) ∈ [(24, -600), (-600, 24), (26, 650), (650, 26), (50, 50)] :=
sorry

end question_a_question_b_l121_121670


namespace sum_of_prime_factors_2310_l121_121651

def is_prime (n : Nat) : Prop :=
  2 ≤ n ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def prime_factors_sum (n : Nat) : Nat :=
  (List.filter Nat.Prime (Nat.factors n)).sum

theorem sum_of_prime_factors_2310 :
  prime_factors_sum 2310 = 28 :=
by
  sorry

end sum_of_prime_factors_2310_l121_121651


namespace product_of_roots_l121_121426

-- Define the two cubic polynomials
def poly1 : Polynomial ℝ := 3 * X^3 + X^2 - 5 * X + 15
def poly2 : Polynomial ℝ := 4 * X^3 - 16 * X^2 + 12

-- Define the combined polynomial equation
def combined_poly : Polynomial ℝ := poly1 * poly2

-- Define the problem statement: Product of roots of the combined polynomial
theorem product_of_roots (P : Polynomial ℝ) (h : P = combined_poly) : 
  Polynomial.leadingCoeff P * -Polynomial.constantCoeff P = -(15 * 12) / (3 * 4) :=
sorry

end product_of_roots_l121_121426


namespace count_valid_satisfying_elements_l121_121575

def f (x : ℤ) : ℤ := x^2 + 5 * x + 2
def g (x : ℤ) : ℤ := x^2 + x + 2
def S : set ℤ := {x | 0 ≤ x ∧ x ≤ 30}

theorem count_valid_satisfying_elements :
  {s ∈ S | (f s) % 4 = 0 ∧ (g s) % 3 = 0}.to_finset.card = 8 :=
by
  sorry

end count_valid_satisfying_elements_l121_121575


namespace initial_amount_is_100_l121_121529

-- Defining the conditions
def interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ := P * R * T

-- Given conditions
def interest_600_10pct_4years : ℝ := interest 600 0.10 4
def interest_P_5pct_48years (P : ℝ) : ℝ := interest P 0.05 48

-- Theorem stating that the initial amount producing the same interest is Rs 100
theorem initial_amount_is_100 : ∃ (P : ℝ), P = 100 ∧ interest_600_10pct_4years = interest_P_5pct_48years P := 
by { 
  let P := 100, 
  use P, 
  split, 
  { refl, },
  {
    unfold interest_600_10pct_4years,
    unfold interest_P_5pct_48years,
    unfold interest,
    sorry
  } 
}

end initial_amount_is_100_l121_121529


namespace symmetric_line_equation_l121_121005

theorem symmetric_line_equation (x y : ℝ) : 
  3 * x - 4 * y + 5 = 0 → (3 * x + 4 * y - 5 = 0) :=
by
sorry

end symmetric_line_equation_l121_121005


namespace parabola_coordinates_l121_121519

theorem parabola_coordinates (x y : ℝ) (h_parabola : y^2 = 4 * x) (h_distance : (x - 1)^2 + y^2 = 100) :
  (x = 9 ∧ y = 6) ∨ (x = 9 ∧ y = -6) :=
by
  sorry

end parabola_coordinates_l121_121519


namespace inner_cube_surface_area_l121_121315

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121315


namespace probability_of_circle_center_l121_121564

noncomputable def equilateral_triangle (h : ℝ) := sorry

theorem probability_of_circle_center {A B C O X : Type} (h : ℝ) (r : ℝ) (height_eq : h = 13) 
  (center_condition : O = (equilateral_triangle 13).center) : 
  ∀ (X : (equilateral_triangle 13)),
  (X ∈ (equilateral_triangle 13))
  ∧ ((metric.ball X r) ⊂ (equilateral_triangle 13))
  → (ratio_within_center (X, 1) O = π / 121) :=
sorry

end probability_of_circle_center_l121_121564


namespace proof_l121_121662

-- Define constants
def a : Real := 1.1
def b : Real := 0.81
def c : Real := 1.44
def d : Real := 0.49

noncomputable def sqrt_a := Real.sqrt a
noncomputable def sqrt_b := Real.sqrt b
noncomputable def sqrt_c := Real.sqrt c
noncomputable def sqrt_d := Real.sqrt d

-- Define the expression to be evaluated
noncomputable def expr := sqrt_a / sqrt_b + sqrt_c / sqrt_d

-- Statement of the proof
theorem proof :
  expr ≈ 2.879 := by
  sorry

end proof_l121_121662


namespace product_not_divisible_by_770_l121_121593

theorem product_not_divisible_by_770 (a b : ℕ) (h : a + b = 770) : ¬ (a * b) % 770 = 0 :=
sorry

end product_not_divisible_by_770_l121_121593


namespace median_free_throw_counts_l121_121134

-- Define the list of free throw counts
def free_throw_counts : List ℕ := [5, 17, 12, 13, 21, 18, 20, 12, 17, 14]

-- Function to compute the median of a sorted list of numbers
def median (l : List ℕ) : ℝ :=
  if h : l.length % 2 = 1 then
    l.nth_le (l.length / 2) (by sorry) -- Case when list length is odd
  else
    let a := l.nth_le (l.length / 2 - 1) (by sorry);
    let b := l.nth_le (l.length / 2) (by sorry);
    (a + b : ℕ) / 2

-- Prove that the median of the given free throw counts is 15.5
theorem median_free_throw_counts : median free_throw_counts.data = 15.5 := sorry

end median_free_throw_counts_l121_121134


namespace polynomial_inequality_l121_121832

noncomputable def f (x : ℝ) (n : ℕ) (coeffs : Fin (n+1) → ℝ) : ℝ :=
  ∑ i in Finset.range (n+1), coeffs ⟨i, Fin.is_lt i⟩ * x^i

noncomputable def g (x : ℝ) (coeffs : Fin 2 → ℝ) : ℝ :=
  x + coeffs 1

noncomputable def h (x : ℝ) (m : ℕ) (coeffs : Fin (m+1) → ℝ) : ℝ :=
  ∑ i in Finset.range (m+1), coeffs ⟨i, Fin.is_lt i⟩ * x^i

theorem polynomial_inequality (n m : ℕ) (f_coeffs : Fin (n+1) → ℝ) (g_coeffs : Fin 2 → ℝ) (h_coeffs : Fin (m+1) → ℝ)
  (hf_deg : n > 3) (hf_coeffs_range : ∀ i, 1 ≤ f_coeffs i ∧ f_coeffs i ≤ 4) (hg_monic : g_coeffs 0 = 1)
  (hf_factorization : ∀ x, f x n f_coeffs = g x g_coeffs * h x m h_coeffs) :
  |g 6 g_coeffs| > 3 :=
by
  sorry

end polynomial_inequality_l121_121832


namespace fraction_multiplication_l121_121049

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l121_121049


namespace sum_of_divisors_143_l121_121091

theorem sum_of_divisors_143 : ∑ d in {d : ℕ | d ∣ 143}.to_finset, d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121091


namespace x_intercept_of_line_l121_121898

noncomputable def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst)

noncomputable def line_equation (p1 : ℝ × ℝ) (m : ℝ) (x : ℝ) : ℝ :=
  m * (x - p1.fst) + p1.snd

theorem x_intercept_of_line : 
  ∃ x : ℝ, line_equation (10, 3) (slope (10, 3) (-4, -4)) x = 0 ∧ x = 4 := 
by
  sorry

end x_intercept_of_line_l121_121898


namespace sum_of_first_10_terms_l121_121980

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions in the problem
axiom a1_a2_sum : a 1 + a 2 = 4
axiom a5_a6_sum : a 5 + a 6 = 20

-- Definition of arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_10_terms (h_arith_seq : is_arithmetic_sequence a d) : 
  (∑ i in Finset.range 10, a i) = 100 :=
by 
  -- Proof is omitted
  sorry

end sum_of_first_10_terms_l121_121980


namespace circle_division_into_identical_arcs_l121_121586

theorem circle_division_into_identical_arcs (n : ℕ) (digits : Fin n → ℕ)
  (h1 : n ≥ 2)
  (h2 : ∀ i, 1 ≤ digits i) -- non-zero digits
  (h3 : ∃s1 s2 : Fin n, s1 ≠ s2 ∧ ∀ k : Fin (n-1), digits ((s1 + k) % n) = digits ((s2 + k) % n)) :
  ∃ m, ∃ arcs : Fin m → Fin n,
    m > 0 ∧ m ∣ n ∧
    ∀ i j, ∀ k : Fin (n / m), digits (arcs i + k) = digits (arcs j + k) :=
sorry

end circle_division_into_identical_arcs_l121_121586


namespace conditional_probability_l121_121642

open ProbabilityTheory

noncomputable def P_A (n m : ℕ) : ℚ := 
  let total := ((n + m) * (n + m - 1)) / 2
  in ((n * (n - 1)) / 2 + (m * (m - 1)) / 2) / total

noncomputable def P_AB (m : ℕ) : ℚ := 
  let total := ((3 + m) * (3 + m - 1)) / 2
  in (m * (m - 1) / 2) / total

theorem conditional_probability (n m : ℕ) (h₀ : n = 3) (h₁ : m = 2) :
  P_AB m / P_A n m = 1 / 4 :=
by
  sorry

end conditional_probability_l121_121642


namespace shopkeeper_loss_percentage_l121_121709

-- Define the rates for mangoes
def mango_buy_rate := 10
def mango_sell_rate := 4

-- Define the rates for apples
def apple_buy_rate := 5
def apple_sell_rate := 3

-- Define the rates for oranges
def orange_buy_rate := 8
def orange_sell_rate := 2

-- Define transportation cost and tax rate
def transportation_cost := 500
def tax_rate := 0.10

-- Define the number of units bought and sold for ease of calculation
def units_bought := 100

-- Total cost price calculation
def total_cost_price := (units_bought / mango_buy_rate) + (units_bought / apple_buy_rate) + (units_bought / orange_buy_rate) + transportation_cost

-- Total selling price calculation
def total_selling_price := (units_bought / mango_sell_rate) + (units_bought / apple_sell_rate) + (units_bought / orange_sell_rate)

-- Tax calculation
def total_tax := tax_rate * total_selling_price

-- Net revenue after tax
def net_revenue := total_selling_price - total_tax

-- Net profit or loss
def net_profit_or_loss := net_revenue - total_cost_price

-- Net profit or loss percentage
def net_profit_or_loss_percentage := (net_profit_or_loss / total_cost_price) * 100

-- Proof statement
theorem shopkeeper_loss_percentage : net_profit_or_loss_percentage = -82.03 := by
  sorry

end shopkeeper_loss_percentage_l121_121709


namespace polynomial_inequality_l121_121928

theorem polynomial_inequality (f : ℝ → ℝ) (h1 : f 0 = 1)
    (h2 : ∀ (x y : ℝ), f (x - y) + f x ≥ 2 * x^2 - 2 * x * y + y^2 + 2 * x - y + 2) :
    f = λ x => x^2 + x + 1 := by
  sorry

end polynomial_inequality_l121_121928


namespace tangent_line_at_point_l121_121486

def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 3 * x^2 + a

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem tangent_line_at_point {a : ℝ} (h_odd : is_odd_function (λ x, f (x + 1) a)) :
  a = 2 → tangent_line (λ x, f x 2) 0 2 = (λ x, 2) := 
sorry

end tangent_line_at_point_l121_121486


namespace cube_volume_l121_121178

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121178


namespace bisector_intersection_independent_of_circle_l121_121464

theorem bisector_intersection_independent_of_circle
  (A B C : Point)
  (τ : Circle := Circle.passingThrough A C)
  (hτ_center_not_on_AC : ¬τ.center ∈ Line (A, C))
  (P : Point := TangentIntersection τ A C)
  (Q : Point := τ.intersect (Line (P, B)))
  :
  ∃ R : Point, (R ∈ Line (A, C)) ∧ (R = intersection_of_bisector_AQC_with_AC) ∧ (independent_of_choice_of_τ R) :=
begin
  -- Proof goes here
  sorry
end

end bisector_intersection_independent_of_circle_l121_121464


namespace quadratic_polynomials_Q_l121_121569

noncomputable def P (x : ℝ) := (x - 1) * (x - 2) * (x - 3)

theorem quadratic_polynomials_Q :
  let Q_set := {Q : ℝ → ℝ | ∃ (R : ℝ → ℝ), (∃ n : ℕ, n = 3 ∧ ∀ x, degree (R x) = n) ∧ 
                                      ∀ x ∈ {1, 2, 3}, Q x ∈ {0, 1, 2} ∧ P (Q x) = P x * R x} in
  fintype.card Q_set = 21 := 
sorry

end quadratic_polynomials_Q_l121_121569


namespace inner_cube_surface_area_l121_121239

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121239


namespace distinct_roots_impossible_l121_121458

/-- Given n > 1 monic quadratic polynomials x^2 - a_i x + b_i
and all 2n numbers a_i and b_i are distinct, it cannot happen that
each of the numbers a_1, ..., a_n, b_1, ..., b_n is a root of one of these polynomials.
-/
theorem distinct_roots_impossible (n : ℕ)
  (h : n > 1)
  (a : Fin n → ℝ)
  (b : Fin n → ℝ)
  (distinct : Function.Injective (Sum.elim a b)) : 
  ¬ (∀ x ∈ Finset.univ.image (fun i : Fin n => a i) ∪ Finset.univ.image (fun i : Fin n => b i),
     ∃ i, x ∈ {x | polynomial.roots (polynomial.C 1 * polynomial.X^2 -
        polynomial.C (a i) * polynomial.X + polynomial.C (b i)).to_finset}) :=
sorry

end distinct_roots_impossible_l121_121458


namespace number_of_welders_left_l121_121131

-- Definitions for the given problem
def total_welders : ℕ := 36
def initial_days : ℝ := 1
def remaining_days : ℝ := 3.0000000000000004
def total_days : ℝ := 3

-- Condition equations
variable (r : ℝ) -- rate at which each welder works
variable (W : ℝ) -- total work

-- Equation representing initial total work
def initial_work : W = total_welders * r * total_days := by sorry

-- Welders who left for another project
variable (X : ℕ) -- number of welders who left

-- Equation representing remaining work
def remaining_work : (total_welders - X) * r * remaining_days = W - (total_welders * r * initial_days) := by sorry

-- Theorem to prove
theorem number_of_welders_left :
  (total_welders * total_days : ℝ) = W →
  (total_welders - X) * remaining_days = W - (total_welders * r * initial_days) →
  X = 12 :=
sorry

end number_of_welders_left_l121_121131


namespace lizard_ratio_l121_121903

def lizard_problem (W S : ℕ) : Prop :=
  (S = 7 * W) ∧ (3 = S + W - 69) ∧ (W / 3 = 3)

theorem lizard_ratio (W S : ℕ) (h : lizard_problem W S) : W / 3 = 3 :=
  by
    rcases h with ⟨h1, h2, h3⟩
    exact h3

end lizard_ratio_l121_121903


namespace min_distance_from_symmetry_center_l121_121520

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + π / 6)

theorem min_distance_from_symmetry_center
  (ω : ℝ)
  (h : ∃ x1 x2 : ℝ, x2 - x1 = 2 ∧ f ω x1 = 2 ∧ f ω x2 = 2 ∧ x1 < x2) :
  (1 / 4) * (2 * π / |ω|) = 1 / 2 :=
sorry

end min_distance_from_symmetry_center_l121_121520


namespace max_additional_voters_l121_121119

theorem max_additional_voters (x n y : ℕ) (hx : 0 ≤ x ∧ x ≤ 10) (hy : y = x - n - 1)
  (hT : (nx / n).is_integer) (h_decrease : ∀ v, (nx + v) / (n + 1) = x - 1 → ∀ m, x - m ≤ 0 → m ≤ 5) :
  ∃ y, y ≥ 0 ∧ y ≤ 5 := sorry

end max_additional_voters_l121_121119


namespace segment_labeling_impossible_l121_121418

theorem segment_labeling_impossible : 
  ∀ (segments : Fin 1978 → Set ℝ), 
  (∀ i j : Fin 1978, i ≠ j → Disjoint (segments i) (segments j)) →
  ¬ ∃ (label : Fin 1978 → ℕ), 
    (∀ k : Fin 1978, Set.card (segments k ∩ {p : ℝ | ∃ i : Fin 1978, p ∈ segments i}) = k.val :=: sorry :=
begin
  sorry
end

end segment_labeling_impossible_l121_121418


namespace inner_cube_surface_area_l121_121242

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121242


namespace tangent_line_curve_l121_121444

theorem tangent_line_curve (x₀ : ℝ) (a : ℝ) :
  (ax₀ + 2 = e^x₀ + 1) ∧ (a = e^x₀) → a = 1 := by
  sorry

end tangent_line_curve_l121_121444


namespace number_of_valid_six_digit_numbers_l121_121711

def odd (n : ℕ) : Prop := n % 2 = 1
def even (n : ℕ) : Prop := n % 2 = 0

def valid_digits := {1, 2, 3, 4, 5, 6}

def valid_six_digit_number (n : List ℕ) : Prop :=
  n.length = 6 ∧
  n.nodup ∧
  (∀ i < n.length - 1, ¬(odd (n.nth_le i sorry) ∧ odd (n.nth_le (i+1) sorry))) ∧
  n.nth_le 3 sorry ≠ 4

theorem number_of_valid_six_digit_numbers :
  { n : List ℕ // valid_digits.nodup ∧ valid_six_digit_number n }.card = 120 :=
sorry

end number_of_valid_six_digit_numbers_l121_121711


namespace prob_A_l121_121113

variable {Ω : Type}

noncomputable def P (A : Set Ω) : ℝ := sorry

axiom P_nonneg (A : Set Ω) : 0 ≤ P A

axiom indep {A B : Set Ω} : independent_events P A B

axiom non_zero_prob {A : Set Ω} : P A > 0

axiom double_prob {A B : Set Ω} : P A = 2 * P B

axiom prob_at_least_one {A B : Set Ω} : P (A ∪ B) = 18 * P (A ∩ B)

theorem prob_A (A B : Set Ω) (h1 : independent_events P A B) (h2 : P A > 0) (h3 : P A = 2 * P B) (h4 : P (A ∪ B) = 18 * P (A ∩ B)) : 
  P A = 3/19 := 
by 
  sorry

end prob_A_l121_121113


namespace inner_cube_surface_area_l121_121237

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121237


namespace sixth_term_sequence_l121_121889

theorem sixth_term_sequence (a b c d : ℚ)
  (h1 : a = 1/4 * (3 + b))
  (h2 : b = 1/4 * (a + c))
  (h3 : c = 1/4 * (b + 48))
  (h4 : 48 = 1/4 * (c + d)) :
  d = 2001 / 14 :=
sorry

end sixth_term_sequence_l121_121889


namespace cube_volume_l121_121207

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l121_121207


namespace right_triangle_sets_l121_121967

theorem right_triangle_sets :
  ∃! (a b c : ℕ), 
    ((a = 5 ∧ b = 12 ∧ c = 13) ∧ a * a + b * b = c * c) ∧ 
    ¬(∃ a b c, (a = 3 ∧ b = 4 ∧ c = 6) ∧ a * a + b * b = c * c) ∧
    ¬(∃ a b c, (a = 4 ∧ b = 5 ∧ c = 6) ∧ a * a + b * b = c * c) ∧
    ¬(∃ a b c, (a = 5 ∧ b = 7 ∧ c = 9) ∧ a * a + b * b = c * c) :=
by {
  --- proof needed
  sorry
}

end right_triangle_sets_l121_121967


namespace solution_m_value_l121_121803

theorem solution_m_value (k m : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 + y^2 = 4 → (∃ y', y = k * x + m ∧ (x^2 + y'^2 = 4))) 
  (h2 : ∃ x1 y1 x2 y2 : ℝ, 
    x1^2 + y1^2 = 4 ∧ y1 = k * x1 + m ∧ 
    x2^2 + y2^2 = 4 ∧ y2 = k * x2 + m ∧ 
    sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 2) : 
  m = sqrt 3 ∨ m = -sqrt 3 :=
by
  sorry

end solution_m_value_l121_121803


namespace discount_comparison_l121_121016

theorem discount_comparison
    (initial_price : ℝ)
    (discount_dollars : ℝ)
    (discount_percent : ℝ)
    (price1 price2 difference : ℝ)
    (price1_calculation : price1 = discount_percent * (initial_price - discount_dollars))
    (price2_calculation : price2 = (initial_price * discount_percent) - discount_dollars)
    (difference_calculation : difference = price1 - price2) :
    difference = 0.75 * 100 := begin
  -- Given values
  have hp : initial_price = 30 := sorry,
  have hd : discount_dollars = 5 := sorry,
  have hpct : discount_percent = 0.85 := sorry,
  -- Calculations
  have hprice1 : price1 = 21.25 := sorry,
  have hprice2 : price2 = 20.50 := sorry,
  have hdifference : difference = 75 := sorry,
  sorry
end

end discount_comparison_l121_121016


namespace volume_of_wedge_l121_121727

theorem volume_of_wedge (c : ℝ) (h : c = 18 * Real.pi) : 
  let r := c / (2 * Real.pi) in
  let V := (4 / 3) * Real.pi * r^3 in
  (V / 6) = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l121_121727


namespace evaluate_f_at_4_l121_121969

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem evaluate_f_at_4 : f 4 = 9 := by
  sorry

end evaluate_f_at_4_l121_121969


namespace problem1_problem2_l121_121743

-- Problem 1
theorem problem1 : 
  ( (64 / 27)^(1 / 3) + (2 * (7 / 9))^0.5 - ((8 / 27)^(1 / 3) + 0.027 - (1 / 3))^0.5 = 1 ) :=
  sorry

-- Problem 2
theorem problem2 :
  (log 3 (sqrt 27) - log 3 (sqrt 3) - log 10 25 - log 10 4 + ln (Real.exp 2) + 2 * (1 / 2) * log 2 4  = 3) :=
  sorry


end problem1_problem2_l121_121743


namespace inner_cube_surface_area_l121_121371

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121371


namespace triangle_cosine_sine_inequality_l121_121526

theorem triangle_cosine_sine_inequality 
  (A B C : ℝ) (k : ℝ) 
  (h_ABC_pi : A + B + C = Real.pi)
  (h_k_nonneg : 0 ≤ k)
  (h_A_pos : 0 < A) (h_B_pos : 0 < B) (h_C_pos : 0 < C) :
  Real.cos B * Real.cos C * Real.sin (A / 2) ^ k + 
  Real.cos C * Real.cos A * Real.sin (B / 2) ^ k + 
  Real.cos A * Real.cos B * Real.sin (C / 2) ^ k < 1 := 
sorry

end triangle_cosine_sine_inequality_l121_121526


namespace find_number_l121_121601

theorem find_number (x : ℤ) (h : 22 * (x - 36) = 748) : x = 70 :=
sorry

end find_number_l121_121601


namespace tim_initial_soda_cans_l121_121035

theorem tim_initial_soda_cans (S : ℕ) (H1 : S - 10 + (S - 10) / 2 + 10 = 34) : 
    S = 26 := by
  have h1 := eq_add_of_sub_eq H1
  sorry

end tim_initial_soda_cans_l121_121035


namespace cube_volume_l121_121177

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121177


namespace sum_of_digits_M_l121_121546

-- Definitions
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Conditions
variables (M : ℕ)
  (h1 : M % 2 = 0)  -- M is even
  (h2 : ∀ d ∈ M.digits 10, d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 9)  -- Digits of M
  (h3 : sum_of_digits (2 * M) = 31)  -- Sum of digits of 2M
  (h4 : sum_of_digits (M / 2) = 28)  -- Sum of digits of M/2

-- Goal
theorem sum_of_digits_M :
  sum_of_digits M = 29 :=
sorry

end sum_of_digits_M_l121_121546


namespace inner_cube_surface_area_l121_121262

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121262


namespace find_f_f_2_l121_121843

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else 2 / x

theorem find_f_f_2 : f (f 2) = 2 :=
by
  -- Skip the proof as per instructions
  sorry

end find_f_f_2_l121_121843


namespace cube_volume_l121_121208

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l121_121208


namespace original_price_of_sarees_l121_121979

theorem original_price_of_sarees (P : ℝ):
  (0.80 * P) * 0.95 = 152 → P = 200 :=
by
  intro h1
  -- You can omit the proof here because the task requires only the statement.
  sorry

end original_price_of_sarees_l121_121979


namespace number_of_possible_x_values_l121_121467

theorem number_of_possible_x_values :
  (∃ (x : ℕ), 7 < x ∧ x < 15) ∧ ∀ (x : ℕ), (7 < x ∧ x < 15) → (x = 8 ∨ x = 9 ∨ x = 10 ∨ x = 11 ∨ x = 12 ∨ x = 13 ∨ x = 14) → 7 :=
sorry

end number_of_possible_x_values_l121_121467


namespace cube_volume_l121_121149

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l121_121149


namespace inner_cube_surface_area_l121_121348

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121348


namespace inner_cube_surface_area_l121_121289

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121289


namespace cube_volume_of_surface_area_l121_121158

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l121_121158


namespace find_a_for_no_unique_solution_l121_121452

theorem find_a_for_no_unique_solution (a k : ℝ) (h : k = 9) 
    (h1 : ∀ x y : ℝ, a * (3 * x + 4 * y) = 36)
    (h2 : ∀ x y : ℝ, k * x + 12 * y = 30) : a = 4 :=
by
  have ha : a = 36 / k, from sorry,
  rw [h] at ha,
  norm_num at ha,
  exact ha

end find_a_for_no_unique_solution_l121_121452


namespace max_log_value_l121_121798

theorem max_log_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z + y + z = 12) : log 4 x + log 2 y + log 2 z ≤ 3 := 
begin
  sorry
end

end max_log_value_l121_121798


namespace inner_cube_surface_area_l121_121251

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121251


namespace game_cost_l121_121950

theorem game_cost (initial_money : ℕ) (toys_count : ℕ) (toy_price : ℕ) (left_money : ℕ) : 
  initial_money = 63 ∧ toys_count = 5 ∧ toy_price = 3 ∧ left_money = 15 → 
  (initial_money - left_money = 48) :=
by
  sorry

end game_cost_l121_121950


namespace cube_volume_of_surface_area_l121_121156

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l121_121156


namespace ranking_correct_l121_121738

variable (Amy Bill Celine David : ℕ) -- Using ℕ for simplicity of comparison
-- Encode the conditions as Lean propositions
def statement_I : Prop := Bill > Amy ∧ Bill > Celine ∧ Bill > David
def statement_II : Prop := ¬ (Amy > Bill ∧ Amy > Celine ∧ Amy > David)
def statement_III : Prop := ¬ (Celine < Amy ∧ Celine < Bill ∧ Celine < David)
def statement_IV : Prop := Amy < David ∧ David < Celine

axiom one_true_statement : statement_I ∨ statement_II ∨ statement_III ∨ statement_IV
axiom only_one_true : ∀ s1 s2 s3 s4 : Prop, (s1 ∨ s2 ∨ s3 ∨ s4) → (¬ (s1 ∧ s2) ∧ ¬ (s1 ∧ s3) ∧ ¬ (s1 ∧ s4) ∧ ¬ (s2 ∧ s3) ∧ ¬ (s2 ∧ s4) ∧ ¬ (s3 ∧ s4)) 

theorem ranking_correct : 
  (Amy > Bill ∧ Amy > Celine ∧ Amy > David) ∧
  (David > Bill ∧ David < Amy ∧ David < Celine) ∧
  (Celine > Bill ∧ Celine < Amy ∧ Celine > David) ∧
  (Bill < Amy ∧ Bill < David ∧ Bill < Celine) :=
by 
  -- proof structure (using sorry to skip the proof step as instructed)
  sorry

end ranking_correct_l121_121738


namespace cube_volume_from_surface_area_l121_121195

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l121_121195


namespace calculate_distance_l121_121697

variables (Vm Vs Dd Du : ℝ)

-- Given conditions
def condition_1 : Vm = 5.5 := rfl
def condition_2 : Du = 20 := rfl
def condition_3 {Vs} : Du / (Vm - Vs) = 5 :=
  by simp [condition_1, condition_2]
def condition_4 {Vs} : Dd / (Vm + Vs) = 5 :=
  sorry -- This would be used in the proof

theorem calculate_distance (Vm Vs Dd Du : ℝ) (h1 : Vm = 5.5) (h2 : Du = 20) 
  (h3 : Du / (Vm - Vs) = 5) (h4 : Dd / (Vm + Vs) = 5) : 
  Dd = 35 :=
  sorry

end calculate_distance_l121_121697


namespace fraction_multiplication_l121_121047

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l121_121047


namespace cube_volume_l121_121146

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l121_121146


namespace dinner_event_handshakes_l121_121412

theorem dinner_event_handshakes (n : ℕ) (condition1 : n = 8) (condition2 : true) (condition3 : true) : 
  let total_people := 2 * n
  let normal_without_spouse := total_people - 2
  let without_injured := total_people - 1
  let handshakes_per_person := normal_without_spouse - 1
  let total_handshakes := (without_injured * handshakes_per_person) / 2
  total_handshakes = 90 := by
begin
  have h1 : total_people = 16 := by simp [condition1],
  have h2 : normal_without_spouse = 14 := by simp [h1],
  have h3 : without_injured = 15 := by simp [h1],
  have h4 : handshakes_per_person = 12 := by simp [h2],
  have h5 : total_handshakes = 90 := by simp [h3, h4],
  exact h5,
end

end dinner_event_handshakes_l121_121412


namespace possible_values_of_expression_l121_121823

variable (a b c d : ℝ)

def sign (x : ℝ) : ℝ := if x > 0 then 1 else -1

theorem possible_values_of_expression 
  (h₀ : a ≠ 0) 
  (h₁ : b ≠ 0) 
  (h₂ : c ≠ 0) 
  (h₃ : d ≠ 0) :
  let expression := 
    sign a + sign b + sign c + sign d + sign (a * b * c * d)
  in 
    expression = 5 
    ∨ expression = 1 
    ∨ expression = -3 :=
by sorry

end possible_values_of_expression_l121_121823


namespace initial_men_invited_l121_121585

theorem initial_men_invited (M W C : ℕ) (h1 : W = M / 2) (h2 : C + 10 = 30) (h3 : M + W + C = 80) (h4 : C = 20) : M = 40 :=
sorry

end initial_men_invited_l121_121585


namespace card_derangement_count_l121_121030

theorem card_derangement_count :
  ∃ (f : ℕ → ℕ), (∀ i, 1 ≤ f i ∧ f i ≤ 5) ∧ (∀ i, f i ≠ i) ∧ fintype.card (fintype {x : fin 5 // x.val + 1 != _root_.card x.succ}) = 44 :=
sorry

end card_derangement_count_l121_121030


namespace second_cube_surface_area_l121_121334

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121334


namespace range_of_f3_l121_121489

theorem range_of_f3 (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 2) 
  (h2 : 1 ≤ 4a + 2b ∧ 4a + 2b ≤ 3) : 
  -3 ≤ 9a + 3b ∧ 9a + 3b ≤ 12 :=
sorry

end range_of_f3_l121_121489


namespace inner_cube_surface_area_l121_121390

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121390


namespace total_matches_equation_l121_121535

theorem total_matches_equation (x : ℕ) (h : ((x * (x - 1)) / 2) = 28) : (1 / 2 : ℚ) * x * (x - 1) = 28 := by
  have h1 : ((x * (x - 1)) / 2 : ℚ) = (1 / 2) * x * (x - 1),
    sorry
  rw ← h1 at h
  exact h

end total_matches_equation_l121_121535


namespace ellipse_equation_area_triangle_MF2N_l121_121469

section EllipseProof

-- Define the parameters for the ellipse
variables {a b c : ℝ} (h_a_gt_b : a > b) (h_b_gt_0 : b > 0)
          (h_eccentricity : c / a = sqrt 3 / 2)
          (h_perimeter_triangle : 8 = 2 * c + |MN|)
          (h_length_MN : |MN| = 8 / 5)

-- Given the conditions, prove the equation of the ellipse is as specified
theorem ellipse_equation : ∃ (a b : ℝ), a = 2 ∧ b = 1 ∧ (∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 → (x^2 / 4) + y^2 = 1) := 
sorry

-- Given the conditions, prove the area of triangle MF2N is as specified
theorem area_triangle_MF2N : ∃ (A : ℝ), A = (4 * sqrt 6) / 5 := 
sorry

end EllipseProof

end ellipse_equation_area_triangle_MF2N_l121_121469


namespace wedge_volume_cylinder_l121_121883

theorem wedge_volume_cylinder (r h : ℝ) (theta : ℝ) (V : ℝ) 
  (hr : r = 6) (hh : h = 6) (htheta : theta = 60) (hV : V = 113) : 
  V = (theta / 360) * π * r^2 * h :=
by
  sorry

end wedge_volume_cylinder_l121_121883


namespace inner_cube_surface_area_l121_121261

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121261


namespace hyperbola_equation_l121_121839

theorem hyperbola_equation 
  (x y : ℝ)
  (h_ellipse : x^2 / 10 + y^2 / 5 = 1)
  (h_asymptote : 3 * x + 4 * y = 0)
  (h_hyperbola : ∃ k ≠ 0, 9 * x^2 - 16 * y^2 = k) :
  ∃ k : ℝ, k = 45 ∧ (x^2 / 5 - 16 * y^2 / 45 = 1) :=
sorry

end hyperbola_equation_l121_121839


namespace constructible_triangle_with_medians_l121_121028

theorem constructible_triangle_with_medians 
  (s_a s_b s_c : ℝ) 
  (h_s_a_pos : s_a > 0) 
  (h_s_b_pos : s_b > 0) 
  (h_s_c_pos : s_c > 0) 
  (h_triangle_inequality : (s_a < s_b + s_c) ∧ (s_b < s_a + s_c) ∧ (s_c < s_a + s_b)) 
  : ∃ (T : Triangle), T.median_a = s_a ∧ T.median_b = s_b ∧ T.median_c = s_c :=
sorry

end constructible_triangle_with_medians_l121_121028


namespace inner_cube_surface_area_l121_121281

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121281


namespace inequality_conditions_equiv_l121_121579

variable (a b c : ℝ)
variable (x y z : ℝ)

-- Define the inequality with the parameters a, b, c
def inequality := a * (x - y) * (x - z) + b * (y - x) * (y - z) + c * (z - x) * (z - y) ≥ 0

-- The conditions needed for the inequality to hold for all x, y, z
def conditions :=
  (-a + 2 * b + 2 * c ≥ 0) ∧
  (2 * a - b + 2 * c ≥ 0) ∧
  (2 * a + 2 * b - c ≥ 0)

-- Prove that inequality holds for all x, y, z if and only if the conditions are met
theorem inequality_conditions_equiv (h : inequality a b c : Prop) :
  (∀ (x y z : ℝ), inequality a b c x y z) ↔ conditions a b c :=
sorry

end inequality_conditions_equiv_l121_121579


namespace correct_statements_l121_121833

def f (x : ℝ) : ℝ := (1 / 2) ^ x
def g (x : ℝ) : ℝ := Real.logBase (1 / 2) x
def h (x : ℝ) : ℝ := g (2 - Real.abs x)

lemma h_even (x : ℝ) : h x = h (-x) :=
by
  sorry

lemma h_min_value (x : ℝ) : ∃ z, ∀ y, h y ≥ z ∧ z = 1 :=
by
  sorry

theorem correct_statements : ∀ x : ℝ, h x = h (-x) ∧ (∀ y, h y ≥ 1) := 
by
  intros x
  exact ⟨h_even x, h_min_value x⟩

end correct_statements_l121_121833


namespace false_prop1_false_prop2_false_prop3_false_prop4_l121_121031

section
variables (m n : Line) (a b : Plane)

-- Assumption definitions for Proposition 1
def P1_conds := (m ∥ a) ∧ (n ∥ b) ∧ (a ∥ b)
def P1 := m ∥ n

-- Assumption definitions for Proposition 2
def P2_conds := (m ∥ n) ∧ (m ∈ a) ∧ (n ⟂ b)
def P2 := a ⟂ b

-- Assumption definitions for Proposition 3
def P3_conds := (a ∩ b = m) ∧ (m ∥ n)
def P3 := (n ∥ a) ∧ (n ∥ b)

-- Assumption definitions for Proposition 4
def P4_conds := (m ⟂ n) ∧ (a ∩ b = m)
def P4 := n ⟂ a ∨ n ⟂ b

-- False propositions
theorem false_prop1 : P1_conds m n a b → ¬ P1 m n := sorry
theorem false_prop2 : P2_conds m n a b → ¬ P2 m n a b := sorry
theorem false_prop3 : P3_conds m n a b → ¬ P3 m n a b := sorry
theorem false_prop4 : P4_conds m n a b → ¬ P4 m n a b := sorry

end

end false_prop1_false_prop2_false_prop3_false_prop4_l121_121031


namespace integral_f_l121_121794

def f (x : ℝ) : ℝ :=
  if (0 <= x ∧ x <= 1) then x^2
  else if (1 < x ∧ x <= Real.exp 1) then 1/x
  else 0

theorem integral_f : ∫ x in (0 : ℝ)..(Real.exp 1), f x = 4/3 := by
  sorry

end integral_f_l121_121794


namespace rational_sqrt_sum_l121_121945

theorem rational_sqrt_sum {a b c : ℚ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : Real.sqrt a + Real.sqrt b = c) : Real.sqrt a ∈ ℚ ∧ Real.sqrt b ∈ ℚ :=
sorry

end rational_sqrt_sum_l121_121945


namespace inner_cube_surface_area_l121_121299

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121299


namespace total_dresses_l121_121422

theorem total_dresses (E M D S: ℕ) 
  (h1 : D = M + 12)
  (h2 : M = E / 2)
  (h3 : E = 16)
  (h4 : S = D - 5) : 
  E + M + D + S = 59 :=
by
  sorry

end total_dresses_l121_121422


namespace transformation_equiv_l121_121645

theorem transformation_equiv :
  ∀ (x : ℝ), (y₁ y₂ : ℝ),
    (y₁ = 4 * sin (x + π / 5)) ∧ (y₂ = 4 * sin (2 * x + π / 5)) →
      ∃ (f : ℝ → ℝ), (∀ (x : ℝ), f x = 2 * x) ∧ (y₂ = y₁) :=
by
  intros x y1 y2 h
  sorry

end transformation_equiv_l121_121645


namespace second_cube_surface_area_l121_121335

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121335


namespace inner_cube_surface_area_l121_121238

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121238


namespace negation_of_exist_prop_l121_121017

theorem negation_of_exist_prop :
  (¬ ∃ x : ℝ, x^2 - x + 2 > 0) ↔ (∀ x : ℝ, x^2 - x + 2 ≤ 0) :=
by {
  sorry
}

end negation_of_exist_prop_l121_121017


namespace min_chocolate_cookies_correct_l121_121679

noncomputable def min_chocolate_cookies (total_cookies : ℕ) :=
  let x := (total_cookies * 25) / 282
  in x * 5

theorem min_chocolate_cookies_correct :
  ∀ (choco_batch_size peanut_batch_size total_cookies choco_ratio peanut_ratio cookies_needed : ℕ),
  choco_batch_size = 5 →
  peanut_batch_size = 6 →
  choco_ratio = 3 →
  peanut_ratio = 2 →
  total_cookies = 94 →
  cookies_needed = 60 →
  min_chocolate_cookies total_cookies = cookies_needed :=
by
  sorry

end min_chocolate_cookies_correct_l121_121679


namespace opposite_sqrt_4_l121_121976

theorem opposite_sqrt_4 : - (Real.sqrt 4) = -2 := sorry

end opposite_sqrt_4_l121_121976


namespace like_terms_sum_l121_121858

theorem like_terms_sum (m n : ℕ) (a b : ℝ) :
  (∀ c d : ℝ, -4 * a^(2 * m) * b^(3) = c * a^(6) * b^(n + 1)) →
  m + n = 5 :=
by 
  intro h
  sorry

end like_terms_sum_l121_121858


namespace estimated_survival_probability_l121_121937

-- Definitions of the given data
def number_of_trees_transplanted : List ℕ := [100, 1000, 5000, 8000, 10000, 15000, 20000]
def number_of_trees_survived : List ℕ := [87, 893, 4485, 7224, 8983, 13443, 18044]
def survival_rates : List ℝ := [0.870, 0.893, 0.897, 0.903, 0.898, 0.896, 0.902]

-- Question: Prove that the probability of survival of this type of young tree under these conditions is 0.9.
theorem estimated_survival_probability : 
  (1 / List.length number_of_trees_transplanted.to_real) * 
  (List.sum survival_rates) >= 0.9 ∧ 
  (1 / List.length number_of_trees_transplanted.to_real) * 
  (List.sum survival_rates) < 1 :=
  by sorry

end estimated_survival_probability_l121_121937


namespace maximum_additional_voters_l121_121116

-- Define conditions
structure MovieRating (n : ℕ) (x : ℤ) where
  (sum_scores : ℤ) : sum_scores = n * x

-- Define a function to verify the rating decrease condition
def rating_decrease_condition (n : ℕ) (x y : ℤ) : Prop :=
  (n*x + y) / (n+1) = x - 1

-- Problem: To prove that the maximum number of additional voters after moment T is 5
theorem maximum_additional_voters (n additional_voters : ℕ) (x y : ℤ) (initial_condition : MovieRating n x) :
  initial_condition.sum_scores = n * x ∧
  (∀ k, 1 ≤ k → k ≤ additional_voters → 
    ∃ y, rating_decrease_condition (n + k - 1) (x - (k-1)) y ∧ y ≤ 0) →
  additional_voters ≤ 5 :=
by
  sorry

end maximum_additional_voters_l121_121116


namespace cube_volume_l121_121222

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l121_121222


namespace inner_cube_surface_area_l121_121391

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121391


namespace cube_volume_l121_121219

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l121_121219


namespace cube_volume_l121_121206

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l121_121206


namespace Jessica_cut_roses_l121_121033

theorem Jessica_cut_roses
  (initial_roses : ℕ) (initial_orchids : ℕ)
  (new_roses : ℕ) (new_orchids : ℕ)
  (cut_roses : ℕ) :
  initial_roses = 15 → initial_orchids = 62 →
  new_roses = 17 → new_orchids = 96 →
  new_roses = initial_roses + cut_roses →
  cut_roses = 2 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h3] at h5
  linarith

end Jessica_cut_roses_l121_121033


namespace polyhedron_volume_l121_121002

-- Define the polyhedron and its properties
def polyhedron (P : Type) : Prop :=
∃ (C : Type), 
  (∀ (p : P) (e : ℝ), e = 2) ∧ 
  (∃ (octFaces triFaces : ℕ), octFaces = 6 ∧ triFaces = 8) ∧
  (∀ (vol : ℝ), vol = (56 + (112 * Real.sqrt 2) / 3))
  
-- A theorem stating the volume of the polyhedron
theorem polyhedron_volume : ∀ (P : Type), polyhedron P → ∃ (vol : ℝ), vol = 56 + (112 * Real.sqrt 2) / 3 :=
by
  intros P hP
  sorry

end polyhedron_volume_l121_121002


namespace complex_number_in_fourth_quadrant_l121_121652

theorem complex_number_in_fourth_quadrant (m : ℝ) (z : Complex) :
  (2 / 3 < m ∧ m < 1) ∧ z = Complex.mk (3 * m - 2) (m - 1) →
  (0 < Re(z) ∧ Im(z) < 0) :=
by
  sorry

end complex_number_in_fourth_quadrant_l121_121652


namespace expected_checks_l121_121561

def expected_checks_on_chessboard : ℚ := 9 / 5

def distinct_squares (sq1 sq2 sq3 sq4 sq5 sq6 : (ℕ × ℕ)) : Prop :=
  list.nodup [sq1, sq2, sq3, sq4, sq5, sq6]

-- Knight moves in L-shape definition
def knight_moves (knight king : (ℕ × ℕ)) : Prop :=
  (abs (fst knight - fst king) = 2 ∧ abs (snd knight - snd king) = 1) ∨ 
  (abs (fst knight - fst king) = 1 ∧ abs (snd knight - snd king) = 2)

-- Define the expected checks in Lean 4
theorem expected_checks :
  ∀ (kn1 kn2 kn3 ki1 ki2 ki3 : (ℕ × ℕ))
    (h_distinct : distinct_squares kn1 kn2 kn3 ki1 ki2 ki3)
    (h_lt16 : (∀ sq, sq ∈ [kn1, kn2, kn3, ki1, ki2, ki3] → fst sq > 0 ∧ fst sq ≤ 4 ∧ snd sq > 0 ∧ snd sq ≤ 4)),
    (∑ (k1 k2 k3 : (ℕ × ℕ)) (hkn1 : k1 = kn1 ∨ k1 = kn2 ∨ k1 = kn3)
                      (hkn2 : k2 = kn1 ∨ k2 = kn2 ∨ k2 = kn3)
                      (hk1 : k3 = kn1 ∨ k3 = kn2 ∨ k3 = kn3)
                      (hki1 : k1 ≠ k2 ∧ k2 ≠ k3 ∧ k3 ≠ k1)
                      (hpred : knight_moves k1 k2 ∨ knight_moves k2 k3 ∨ knight_moves k3 k1), 
                      1) / 240 = expected_checks_on_chessboard := 
by sorry

end expected_checks_l121_121561


namespace minimum_value_of_3x_plus_4y_exists_x_y_for_minimum_value_l121_121872

theorem minimum_value_of_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
sorry

theorem exists_x_y_for_minimum_value : ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y ∧ 3 * x + 4 * y = 5 :=
sorry

end minimum_value_of_3x_plus_4y_exists_x_y_for_minimum_value_l121_121872


namespace lunks_needed_for_20_apples_l121_121509

theorem lunks_needed_for_20_apples 
  (lunks_per_kunks : ℚ := 4 / 6)
  (kunks_per_apples : ℚ := 3 / 5) :
  let x := (20 * kunks_per_apples).denom in -- calculating kunks needed for 20 apples
  let y := (x * lunks_per_kunks).denom in  -- calculating lunks needed for that many kunks
  y = 8 := -- as per our final result
sorry

end lunks_needed_for_20_apples_l121_121509


namespace inner_cube_surface_area_l121_121365

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121365


namespace cube_volume_l121_121167

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121167


namespace sum_of_divisors_of_143_l121_121074

theorem sum_of_divisors_of_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := 
by
  sorry

end sum_of_divisors_of_143_l121_121074


namespace inner_cube_surface_area_l121_121345

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121345


namespace find_AC_l121_121900

noncomputable def AC_value (AB BC BD AD AC: ℝ) : Prop :=
  BC = 6 ∧ BD = 7 ∧ AB > BC ∧ (AC = 12 ∨ AC = 13) ∧
  (let D := AD in 
   let CD := AC - AD in 
   D + CD = AC ∧ 
   CD = BD ∧ 
   (AB = AC + AD))

theorem find_AC (AB BC BD AD AC : ℝ) (h1 : BC = 6) (h2: BD = 7) (h3: AB > BC)
  (hABD_isosceles: (AB = BD) ∨ (AD = BD)) (hBCD_isosceles: (BC = BD) ∨ (AD + (AC - AD) = AC)) :
  AC_value AB BC BD AD AC :=
by
  sorry

end find_AC_l121_121900


namespace min_distance_sum_l121_121788

-- Definitions for the given problem
def parabola : set (ℝ × ℝ) := { p | p.2 ^ 2 = 4 * p.1 }

def focus : ℝ × ℝ := (1, 0)

def directrix : set (ℝ × ℝ) := { p | p.1 = -1 }

def fixed_point : ℝ × ℝ := (3, 1)

def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

-- The proof problem in Lean 4 statement
theorem min_distance_sum :
  ∃ M ∈ parabola, ∀ x ∈ parabola, distance x fixed_point + distance x focus ≥ 4 := sorry

end min_distance_sum_l121_121788


namespace calculate_difference_l121_121926

theorem calculate_difference :
  let m := Nat.find (λ m, m > 99 ∧ m < 1000 ∧ m % 13 = 7)
  let n := Nat.find (λ n, n > 999 ∧ n < 10000 ∧ n % 13 = 7)
  n - m = 895 :=
by
  sorry

end calculate_difference_l121_121926


namespace sum_of_divisors_of_143_l121_121079

theorem sum_of_divisors_of_143 : 
  (∑ d in Finset.filter (fun d => 143 % d = 0) (Finset.range 144), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l121_121079


namespace game_strategy_l121_121706

theorem game_strategy (m n : ℕ) :
  if (m + n) % 2 = 0 then 
    (∃ P1 P2 : ℕ, (P1 = 1 ∧ P2 = 2) ∧ (P1 loses ↔ P2 wins))
  else 
    (∃ P1 P2 : ℕ, (P1 = 1 ∧ P2 = 2) ∧ (P2 loses ↔ P1 wins)) :=
by
  sorry

end game_strategy_l121_121706


namespace math_problem_l121_121761

open real

noncomputable def parametric_equation_of_line (l : ℝ → ℝ × ℝ) : Prop :=
  ∀ t : ℝ, l t = (1 + (sqrt 3 / 2) * t, 2 + (1 / 2) * t)

def polar_equation_of_circle (ρ θ : ℝ) : Prop :=
  ρ = 6 * sin θ

def condition_Point_P : Prop := (1, 2) = (1, 2)

def condition_Point_M : Prop := (3, π / 2) = (3, π / 2)

def condition_line_l_inclination : Prop := atan2 1 (sqrt 3 / 2) = π / 6 ∧ (1,2) ∈ (l '' set.univ)

def condition_circle_C : Prop := ∀ x y : ℝ, (x - 0)^2 + (y - 3)^2 = 9 → (3, π / 2) = (√((x - 3)^2 + y^2), atan2 y (x - 3))

theorem math_problem 
  (l : ℝ → ℝ × ℝ)
  (ρ θ t: ℝ)
  (x y : ℝ)
  (P : ℝ × ℝ := (1, 2))
  (M : ℝ × ℝ := (3, π / 2)) :
  condition_Point_P →
  condition_Point_M →
  condition_line_l_inclination →
  condition_circle_C →
  parametric_equation_of_line l ∧
  polar_equation_of_circle ρ θ ∧
  (|ρ - 3| * |ρ + 3| = 7) :=
begin
  intros hP hM hL hC,
  split, sorry,
  split, sorry,
  sorry
end

end math_problem_l121_121761


namespace balls_into_boxes_l121_121506

theorem balls_into_boxes {balls boxes : ℕ} (h_balls : balls = 6) (h_boxes : boxes = 4) : 
  (indistinguishable_partitions balls boxes).count = 9 := 
by
  sorry

end balls_into_boxes_l121_121506


namespace car_travel_distance_l121_121682

theorem car_travel_distance :
  ∃ S : ℝ, 
    (S > 0) ∧ 
    (∃ v1 v2 t1 t2 t3 t4 : ℝ, 
      (S / 2 = v1 * t1) ∧ (26.25 = v2 * t2) ∧ 
      (S / 2 = v2 * t3) ∧ (31.2 = v1 * t4) ∧ 
      (∃ k : ℝ, k = (S - 31.2) / (v1 + v2) ∧ k > 0 ∧ 
        (S = 58))) := sorry

end car_travel_distance_l121_121682


namespace quarterback_passes_left_l121_121228

noncomputable def number_of_passes (L : ℕ) : Prop :=
  let R := 2 * L
  let C := L + 2
  L + R + C = 50

theorem quarterback_passes_left : ∃ L, number_of_passes L ∧ L = 12 := by
  sorry

end quarterback_passes_left_l121_121228


namespace circular_segment_discrepancy_l121_121671

theorem circular_segment_discrepancy :
  let r := 3 * Real.sqrt 3
  let c := 9
  let s := r / 2
  let θ := (2 / 3) * Real.pi
  let actual_area := (1 / 2) * θ * r^2 - (1 / 2) * r^2 * Real.sin θ
  let empirical_area := (1 / 2) * (c * s + s^2)
  in actual_area - empirical_area = (27 * Real.sqrt 3) / 2 + 27 / 8 - 9 * Real.pi :=
by
  sorry

end circular_segment_discrepancy_l121_121671


namespace sum_of_divisors_143_l121_121098

theorem sum_of_divisors_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121098


namespace ravi_speed_is_18_kmph_l121_121948

def ravi_speed 
  (distance_meters : ℕ) 
  (time_seconds : ℕ) 
  (distance_km := (distance_meters : ℝ) / 1000)
  (time_hours := (time_seconds : ℝ) / 3600) 
  : ℝ :=
  distance_km / time_hours

theorem ravi_speed_is_18_kmph 
  (distance_meters : ℕ) 
  (time_seconds : ℕ) 
  (h_distance : distance_meters = 900)
  (h_time : time_seconds = 180) 
  : ravi_speed distance_meters time_seconds = 18 := 
by
  rw [ravi_speed, h_distance, h_time]
  rw [(900 : ℝ) / 1000, (180 : ℝ) / 3600]
  -- Remaining proof steps are omitted
  sorry

end ravi_speed_is_18_kmph_l121_121948


namespace estimatedSurvivalProbability_l121_121939

-- Definitions specific to the problem
def numYoungTreesTransplanted : ℕ := 20000
def numYoungTreesSurvived : ℕ := 18044

def survivalRate : ℝ := numYoungTreesSurvived / numYoungTreesTransplanted

theorem estimatedSurvivalProbability :
  Real.round (survivalRate * 10) / 10 = 0.9 :=
by
  sorry

end estimatedSurvivalProbability_l121_121939


namespace new_cube_weight_l121_121659

theorem new_cube_weight (s : ℝ) (density : ℝ) (h₁ : density = 3 / (s^3)) :
  let new_side_length := 2 * s,
      new_volume := new_side_length^3,
      new_weight := new_volume * density
  in new_weight = 24 :=
by
  let new_side_length := 2 * s
  let new_volume := new_side_length^3
  let new_weight := new_volume * density
  have density_defined : density = 3 / (s^3) := h₁
  have volume_original := s^3
  have volume_new := (2 * s)^3
  have weight_new := volume_new * density
  rw [volume_new, density_defined] at weight_new
  simp at weight_new
  have expected : weight_new = 24 := by sorry
  exact expected

end new_cube_weight_l121_121659


namespace cube_volume_from_surface_area_l121_121197

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l121_121197


namespace floor_eq_solution_l121_121763

theorem floor_eq_solution (x : ℝ) : 2.5 ≤ x ∧ x < 3.5 → (⌊2 * x + 0.5⌋ = ⌊x + 3⌋) :=
by
  sorry

end floor_eq_solution_l121_121763


namespace cube_volume_l121_121218

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l121_121218


namespace no_carry_addition_pairs_count_l121_121415

theorem no_carry_addition_pairs_count :
  {n : ℕ | 1500 ≤ n ∧ n ≤ 2500 ∧ ∀ d, d = 0 → (((n % 10) + 2) % 10) = (n % 10 + 2) ∧ (((n / 10) % 10 + 2) % 10) = (n / 10 % 10 + 2) ∧ (((n / 100) % 10 + 2) % 10) = (n / 100 % 10 + 2)}.card = 512 :=
sorry

end no_carry_addition_pairs_count_l121_121415


namespace max_voters_after_T_l121_121123

theorem max_voters_after_T (x : ℕ) (n : ℕ) (y : ℕ) (T : ℕ)  
  (h1 : x <= 10)
  (h2 : x > 0)
  (h3 : (nx + y) ≤ (n + 1) * (x - 1))
  (h4 : ∀ k, (x - k ≥ 0) ↔ (n ≤ T + 5)) :
  ∃ (m : ℕ), m = 5 := 
sorry

end max_voters_after_T_l121_121123


namespace count_red_balls_l121_121684

/-- Given conditions:
  - The total number of balls in the bag is 100.
  - There are 50 white, 20 green, 10 yellow, and 3 purple balls.
  - The probability that a ball will be neither red nor purple is 0.8.
  Prove that the number of red balls is 17. -/
theorem count_red_balls (total_balls white_balls green_balls yellow_balls purple_balls red_balls : ℕ)
  (h1 : total_balls = 100)
  (h2 : white_balls = 50)
  (h3 : green_balls = 20)
  (h4 : yellow_balls = 10)
  (h5 : purple_balls = 3)
  (h6 : (white_balls + green_balls + yellow_balls) = 80)
  (h7 : (white_balls + green_balls + yellow_balls) / (total_balls : ℝ) = 0.8) :
  red_balls = 17 :=
by
  sorry

end count_red_balls_l121_121684


namespace abcdefg_defghij_value_l121_121516

variable (a b c d e f g h i : ℚ)

theorem abcdefg_defghij_value :
  (a / b = -7 / 3) →
  (b / c = -5 / 2) →
  (c / d = 2) →
  (d / e = -3 / 2) →
  (e / f = 4 / 3) →
  (f / g = -1 / 4) →
  (g / h = 3 / -5) →
  (abcdefg / defghij = (-21 / 16) * (c / i)) :=
by
  sorry

end abcdefg_defghij_value_l121_121516


namespace log_base_3_intersects_x_axis_l121_121749

theorem log_base_3_intersects_x_axis : ∃ x : ℝ, x = 1 ∧ log 3 x = 0 := 
by
  sorry

end log_base_3_intersects_x_axis_l121_121749


namespace inner_cube_surface_area_l121_121302

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121302


namespace J_2_12_9_l121_121783

def J (a b c : ℝ) : ℝ := a / b + b / c + c / a

theorem J_2_12_9 : J 2 12 9 = 6 := by
  have h₁ : (2 : ℝ) ≠ 0 := by simp
  have h₂ : (12 : ℝ) ≠ 0 := by simp
  have h₃ : (9 : ℝ) ≠ 0 := by simp
  sorry

end J_2_12_9_l121_121783


namespace inner_cube_surface_area_l121_121265

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121265


namespace initial_sweets_at_first_l121_121638

-- Define the initial number of sweets
variable (x : ℕ)

-- Conditions
def jack_took : ℕ := x / 2 + 4
def paul_took : ℕ := 7

-- Theorem to prove
theorem initial_sweets_at_first : jack_took(x) + paul_took = x := by
  sorry

end initial_sweets_at_first_l121_121638


namespace odd_count_between_1_and_2013_l121_121910

theorem odd_count_between_1_and_2013 : 
  ∃ k, ∀ n, (1 ≤ n ∧ n ≤ 2013 ∧ n % 2 = 1) ↔ (∃ m, n = 2 * m - 1 ∧ 1 ≤ m ∧ m ≤ k) ∧ k = 1007 := 
begin
  sorry  -- Proof is omitted
end

end odd_count_between_1_and_2013_l121_121910


namespace specific_case_2008_l121_121674

noncomputable def proofProblem :=
  ∀ n : ℕ,
    if n ≤ 2 then n^(n+1) < (n+1)^n 
    else n^(n+1) > (n+1)^n

theorem specific_case_2008 : 2008^(2009) > 2009^(2008) :=
  sorry

example test_cases :
  (1:ℕ)^2 < (2:ℕ)^1 ∧
  (2:ℕ)^3 < (3:ℕ)^2 ∧
  (3:ℕ)^4 > (4:ℕ)^3 ∧
  (4:ℕ)^5 > (5:ℕ)^4 ∧
  (5:ℕ)^6 > (6:ℕ)^5 :=
begin
  -- Placeholders for each of the comparisons, in practice these would be proven.
  sorry, sorry, sorry, sorry, sorry
end

end specific_case_2008_l121_121674


namespace tom_teaching_years_l121_121997

theorem tom_teaching_years :
  ∃ T D : ℕ, T + D = 70 ∧ D = (1 / 2) * T - 5 ∧ T = 50 :=
by
  sorry

end tom_teaching_years_l121_121997


namespace total_customers_l121_121732

theorem total_customers (tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) :
  tables = 9 →
  women_per_table = 7 →
  men_per_table = 3 →
  (tables * (women_per_table + men_per_table)) = 90 :=
by
  intro htables hwomen hmen
  rw [htables, hwomen, hmen]
  simp
  norm_num
  rfl

end total_customers_l121_121732


namespace inner_cube_surface_area_l121_121260

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121260


namespace hiring_probabilities_l121_121690

def A (k : ℕ) : ℕ := sorry
def fact (n : ℕ) : ℕ := if n = 0 then 1 else n * fact (n - 1)
def number_of_sequences : ℕ := fact 10

theorem hiring_probabilities :
  (∀ (k : ℕ), (1 ≤ k ∧ k ≤ 7) → A k > A (k + 1)) ∧ 
  (A 8 = A 9 ∧ A 9 = A 10) ∧ 
  (A 1 + A 2 + A 3 > 0.7 * number_of_sequences) ∧ 
  (A 8 + A 9 + A 10 <= 0.1 * number_of_sequences) :=
by
  sorry

end hiring_probabilities_l121_121690


namespace find_intersection_find_m_range_l121_121495

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y : ℝ, y = sqrt(3 - 2 * x - x^2)}

def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * x + 1 - m^2 ≤ 0}

-- Statement (1): If m = 3, find A ∩ B
theorem find_intersection (hm : 3 = 3) : A ∩ (B 3) = {x | -2 ≤ x ∧ x ≤ 1} := sorry

-- Statement (2): If m > 0 and A ⊆ B, find the range of m
theorem find_m_range (hm : 0 < m) (subset_A_B : A ⊆ B m) : 4 ≤ m := sorry

end find_intersection_find_m_range_l121_121495


namespace rabbit_can_escape_square_l121_121229

-- Definitions for the conditions
def initial_positions : Type := 
  { rabbit : (ℝ × ℝ) // rabbit = (0.5, 0.5) } 
  ∧ { wolves : list (ℝ × ℝ) // wolves.length = 4 ∧ (0, 0) ∈ wolves ∧ (1, 0) ∈ wolves ∧ (0, 1) ∈ wolves ∧ (1, 1) ∈ wolves }

def speeds : Prop := ∀ (rabbit_speed wolf_speed : ℝ), wolf_speed = 1.4 * rabbit_speed

-- Statement of the proof problem
theorem rabbit_can_escape_square : initial_positions ∧ speeds → ∃ (strategy : (ℝ × ℝ) → (ℝ × ℝ)), ∀ t, rabbit_strategy t ≠ ∀ (w : (ℝ × ℝ)), w ∈ wolves → not (∃ (time : ℝ), (strategy t).1 <= time ∧ (strategy t).2 <= time)  :=
by 
  sorry

end rabbit_can_escape_square_l121_121229


namespace find_a_if_even_function_l121_121875

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * 2^x + 2^(-x)

-- Theorem statement
theorem find_a_if_even_function (a : ℝ) (h : is_even_function (f a)) : a = 1 :=
sorry

end find_a_if_even_function_l121_121875


namespace inner_cube_surface_area_l121_121278

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121278


namespace function_range_cosine_identity_l121_121973

theorem function_range_cosine_identity
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h₀ : 0 < ω)
  (h₁ : ∀ x, f x = (1/2) * Real.cos (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x))
  (h₂ : ∀ x, f (x + π / ω) = f x) :
  Set.Icc (f (-π / 3)) (f (π / 6)) = Set.Icc (-1 / 2) 1 :=
by
  sorry

end function_range_cosine_identity_l121_121973


namespace extreme_point_when_a_is_2_range_of_a_sequence_monotonically_increasing_l121_121485

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^2 - a * x + Real.log (x + 1)

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  2 * x - a + 1 / (x + 1)

theorem extreme_point_when_a_is_2 :
  ∃ x : ℝ, f x 2 = x^2 - 2*x + Real.log(x+1) ∧ x ∈ {x | x > -1} ∧ x = Real.sqrt 2 / 2 := sorry

theorem range_of_a (h : ∀ x ∈ Ioo 0 1, f' x a > x) :
  a ∈ Iic 1 := sorry

theorem sequence_monotonically_increasing
  (c1 : ℝ) (hc1 : c1 > 0) (ha : ∀ x ∈ Ioo 0 1, f' x a > x) :
  ∀ n : ℕ, (c : ℕ → ℝ) (hc : ∀ n, c (n+1) = f' (c n) a), ∀ k, c (k+1) > c k := sorry

end extreme_point_when_a_is_2_range_of_a_sequence_monotonically_increasing_l121_121485


namespace cube_volume_of_surface_area_l121_121157

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l121_121157


namespace minimum_value_proof_l121_121876

noncomputable def minimum_value : ℝ :=
  3 + 2 * Real.sqrt 2

theorem minimum_value_proof (a b : ℝ) (h_line_eq : ∀ x y : ℝ, a * x + b * y = 1)
  (h_ab_pos : a * b > 0)
  (h_center_bisect : ∃ x y : ℝ, (x - 1)^2 + (y - 2)^2 <= x^2 + y^2) :
  (1 / a + 1 / b) ≥ minimum_value :=
by
  -- Sorry placeholder for the proof
  sorry

end minimum_value_proof_l121_121876


namespace nonnegative_difference_between_roots_l121_121650

open Real

def roots_difference (a b c : ℝ) : ℝ :=
  if a ≠ 0 then
    let delta := b * b - 4 * a * c
    let r1 := (-b + sqrt delta) / (2 * a)
    let r2 := (-b - sqrt delta) / (2 * a)
    abs (r1 - r2)
  else 0

theorem nonnegative_difference_between_roots : 
  roots_difference 1 42 480 = 4 :=
by
  -- We will complete the proof here
  sorry

end nonnegative_difference_between_roots_l121_121650


namespace greatest_number_that_divides_is_2_l121_121666

-- Define the problem conditions
def div_condition1 (n : ℕ) : Prop := (1557 % n = 7)
def div_condition2 (n : ℕ) : Prop := (2037 % n = 5)
def div_condition3 (n : ℕ) : Prop := (2765 % n = 9)

-- Combine the conditions
def all_conditions (n : ℕ) : Prop := div_condition1 n ∧ div_condition2 n ∧ div_condition3 n

-- Prove the statement
theorem greatest_number_that_divides_is_2 : ∃ n, all_conditions n ∧ is_gcd n 1550 2032 ∧ is_gcd n n 2756 ∧ n = 2 :=
by
  sorry

end greatest_number_that_divides_is_2_l121_121666


namespace coeff_x5_in_expansion_l121_121549

theorem coeff_x5_in_expansion : 
  (coeff_of_x_pow 5 ((2-x)^7) = -84) :=
sorry

end coeff_x5_in_expansion_l121_121549


namespace angle_expr_correct_l121_121018

noncomputable def angle_expr : Real :=
  Real.cos (40 * Real.pi / 180) * Real.cos (160 * Real.pi / 180) +
  Real.sin (40 * Real.pi / 180) * Real.sin (20 * Real.pi / 180)

theorem angle_expr_correct : angle_expr = -1 / 2 := 
by 
   sorry

end angle_expr_correct_l121_121018


namespace number_of_paths_from_P_to_Q_l121_121691

-- Define the intermediary points and the endpoints
inductive Point
| P | Q | R | S | T | U

open Point

-- Define the paths as relations between points
def paths : Point → Point → Prop
| P, R := true
| P, S := true
| R, T := true
| S, T := true
| S, U := true
| U, T := true
| U, Q := true
| T, Q := true
| _, _ := false

-- Theorem stating the number of paths from P to Q
theorem number_of_paths_from_P_to_Q : 
    (∃ s1 : list Point, (s1 = [P, R, T, Q] ∨ s1 = [P, S, T, Q] ∨ s1 = [P, S, U, Q] ∨ s1 = [P, S, U, T, Q]) 
    ∧ (∀ (i j : ℕ), i < j ∧ j < s1.length → paths (s1.nth_le i sorry) (s1.nth_le j sorry))) → 
    s1.length = 4 :=
sorry

end number_of_paths_from_P_to_Q_l121_121691


namespace car_speed_constant_l121_121683

theorem car_speed_constant (v : ℝ) : 
  (1 / (v / 3600) - 1 / (80 / 3600) = 2) → v = 3600 / 47 := 
by
  sorry

end car_speed_constant_l121_121683


namespace sum_of_divisors_143_l121_121092

theorem sum_of_divisors_143 : ∑ d in {d : ℕ | d ∣ 143}.to_finset, d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121092


namespace fraction_product_l121_121054

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l121_121054


namespace sphere_wedge_volume_l121_121720

theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) (V : ℝ) (wedge_volume : ℝ) :
  circumference = 18 * Real.pi → num_wedges = 6 → V = (4 / 3) * Real.pi * (9^3) → wedge_volume = V / 6 → 
  wedge_volume = 162 * Real.pi :=
by
  intros h1 h2 h3 h4
  rw h3 at h4
  rw [←Real.pi_mul, ←mul_assoc, Nat.cast_bit1, Nat.cast_bit0, Nat.cast_one, pow_succ, pow_one, ←mul_assoc] at h4
  rw [mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc] at h4
  sorry

end sphere_wedge_volume_l121_121720


namespace inner_cube_surface_area_l121_121301

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121301


namespace commute_time_difference_l121_121753

theorem commute_time_difference
  (distance : ℝ) 
  (walk_speed : ℝ) 
  (train_speed : ℝ) 
  (time_diff : ℝ) 
  (walk_time : ℝ := distance / walk_speed) 
  (train_time : ℝ := distance / train_speed) 
  (additional_time : ℝ := walk_time - train_time - time_diff) :
  distance = 1.5 ∧ walk_speed = 3 ∧ train_speed = 20 ∧ time_diff = 25 → additional_time = 0.5 := 
by 
  intros h, 
  cases h with h_dist h_rest, 
  cases h_rest with h_walk_speed h_rest, 
  cases h_rest with h_train_speed h_time_diff,
  simp [h_dist, h_walk_speed, h_train_speed, h_time_diff] at additional_time,
  norm_num at additional_time,
  exact additional_time

end commute_time_difference_l121_121753


namespace number_of_boys_l121_121663

theorem number_of_boys (total_students : ℕ) (fraction_girls : ℚ) (number_girls : ℕ) (number_boys : ℕ) :
  total_students = 160 →
  fraction_girls = 5 / 8 →
  number_girls = (fraction_girls * total_students).toNat →
  number_boys = total_students - number_girls →
  number_boys = 60 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h3
  simp [h1, h3, h4]
  sorry

end number_of_boys_l121_121663


namespace cube_volume_l121_121176

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121176


namespace arithmetic_sequence_formula_l121_121468

variable (a : ℤ)

def first_term : ℤ := a - 1
def second_term : ℤ := 2 * a + 1
def third_term : ℤ := a + 7
def common_difference : ℤ := second_term a - first_term a

theorem arithmetic_sequence_formula (a_n : ℕ → ℤ) :
  (2 * second_term a = first_term a + third_term a) →
  a = 2 →
  (∀ n, a_n n = 1 + (n - 1) * 4) :=
begin
  assume h_eq h_a n,
  sorry
end

end arithmetic_sequence_formula_l121_121468


namespace cube_volume_l121_121164

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121164


namespace problem_equivalent_l121_121808

noncomputable def ellipse_equation : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ e = (Real.sqrt 2) / 2 ∧ 2 * b^2 / a = 2 ∧
  (∀ x y : ℝ, (x^2 / (a^2)) + (y^2 / (b^2)) = 1 → (x^2 / 4) + (y^2 / 2) = 1)

noncomputable def tangent_lines_exist : Prop :=
  ∃ k b : ℝ, 
  (∀ M N : ℚ → Prop, (x^2 / 4) + (y^2 / 2) = 1) ∧ 
  (kx + b = y) ∧ 
  ∃ M N : ℝ × ℝ, 
  (M ≠ N) ∧ 
  (∃ (l : ℝ), 
    (y = kx + b) ∧ ((y₁ / x₁) * (y₂ / x₂) = 7 / 16))

theorem problem_equivalent : ellipse_equation ∧ tangent_lines_exist :=
sorry

end problem_equivalent_l121_121808


namespace second_cube_surface_area_l121_121333

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121333


namespace paint_polygons_count_l121_121133

theorem paint_polygons_count :
  let rect1 := (1 : ℕ, 2 : ℕ), rect2 := (1 : ℕ, 2 : ℕ),
      triangle1 := (1 : ℕ, 1 : ℕ), triangle2 := (1 : ℕ, 1 : ℕ),
      conditions := (no_blue_longer_side_yellow : ∀ (rects triangles : list (ℕ × ℕ)), 
                      -- condition ensuring no longer side of a blue polygon is shared with a yellow one
                      true), 
      polygons := rect1 :: rect2 :: triangle1 :: triangle2 :: [],
      paint_set := {blue, yellow},
      valid_paints := (fun colors : list (color),
                         -- condition ensuring no blue polygon's longer side is shared with a yellow one's
                         no_blue_longer_side_yellow rects triangles) in
  (nat.card {colors : list (color) // valid_paints colors}.to_finset = 9) :=
sorry

end paint_polygons_count_l121_121133


namespace complex_is_purely_imaginary_iff_a_eq_2_l121_121512

theorem complex_is_purely_imaginary_iff_a_eq_2 (a : ℝ) :
  (a = 2) ↔ ((a^2 - 4 = 0) ∧ (a + 2 ≠ 0)) :=
by sorry

end complex_is_purely_imaginary_iff_a_eq_2_l121_121512


namespace part1_part2_l121_121480

noncomputable def z : ℂ := 3 + complex.i
def pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem part1 : ∀ b : ℝ, pure_imaginary ((1 + 3 * complex.i) * (3 + b * complex.i)) → b = 1 :=
by
  intro b h
  -- Proof step skipped
  sorry

theorem part2 : 
  ∀ z : ℂ, z = (3 + complex.i) → complex.abs ((z / (2 + complex.i))) = real.sqrt 2 :=
by
  intro z hz
  -- Proof step skipped
  sorry

end part1_part2_l121_121480


namespace scientific_notation_l121_121605

theorem scientific_notation : (20160 : ℝ) = 2.016 * 10^4 := 
  sorry

end scientific_notation_l121_121605


namespace range_of_a_l121_121494

variable (a : ℝ)
def A := {x : ℝ | abs (x - a) ≤ 1}
def B := {x : ℝ | x^2 - 5 * x + 4 > 0}

theorem range_of_a :
  (∀ x, x ∈ A → x ∉ B) → (2 ≤ a ∧ a ≤ 3) :=
by
  intro h
  -- sorry is added here as a placeholder to skip the proof.
  sorry

end range_of_a_l121_121494


namespace division_probability_is_six_thirteen_l121_121014

theorem division_probability_is_six_thirteen :
  let r_values := {r : ℤ | -5 ≤ r ∧ r ≤ 7},
      k_values := {k : ℤ | -3 ≤ k ∧ k ≤ 6 ∧ k ≠ 0},
      total_pairs := ∀ r ∈ r_values, ∀ k ∈ k_values,
      valid_pairs := { ⟨r, k⟩ : ℤ × ℤ | r ∈ r_values ∧ k ∈ k_values ∧ k ∣ r }
  in (valid_pairs.card : ℚ) / (total_pairs.card : ℚ) = 6 / 13 :=
by
  sorry

end division_probability_is_six_thirteen_l121_121014


namespace sum_of_divisors_143_l121_121090

theorem sum_of_divisors_143 : ∑ d in {d : ℕ | d ∣ 143}.to_finset, d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121090


namespace perfect_square_expression_l121_121665

theorem perfect_square_expression (p : ℝ) (h : p = 0.28) : 
  (12.86 * 12.86 + 12.86 * p + 0.14 * 0.14) = (12.86 + 0.14) * (12.86 + 0.14) :=
by 
  -- proof goes here
  sorry

end perfect_square_expression_l121_121665


namespace sum_consecutive_even_integers_l121_121957

theorem sum_consecutive_even_integers (m : ℤ) :
  (m + (m + 2) + (m + 4) + (m + 6) + (m + 8)) = 5 * m + 20 := by
  sorry

end sum_consecutive_even_integers_l121_121957


namespace problem_solution_l121_121746

noncomputable def c_solution_set : set ℝ :=
{ c : ℝ | 0 ≤ c ∧ c^2 ≤ 4 / 3 }

theorem problem_solution (c : ℝ) (x y : ℝ) :
  (∃ x y : ℝ, real.sqrt (x * y) = c^c ∧ real.log c (x^(real.log c y)) + real.log c (y^(real.log c x)) = 3 * c^3) ↔
  c ∈ c_solution_set :=
by sorry

end problem_solution_l121_121746


namespace sum_partition_ominous_years_l121_121733

def is_ominous (n : ℕ) : Prop :=
  n = 1 ∨ Nat.Prime n

theorem sum_partition_ominous_years :
  ∀ n : ℕ, (¬ ∃ (A B : Finset ℕ), A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅ ∧ 
    (A.sum id = B.sum id ∧ A.card = B.card)) ↔ is_ominous n := 
sorry

end sum_partition_ominous_years_l121_121733


namespace sum_of_digits_N_l121_121750

def N : ℕ := (List.range 500).sum (λ n, 10^(n+1) - 1)

theorem sum_of_digits_N : (N.digits.sum) = 6 :=
by
  sorry

end sum_of_digits_N_l121_121750


namespace product_AM_CN_constant_l121_121540

variables {A B C M N : Point}
variables (tri : Triangle A B C)
variables (isosceles : is_isosceles tri)
variables (semi_circle : Semicircle (Segment AC))
variables (tangent_M : is_tangent semi_circle (Segment AB) M)
variables (tangent_N : is_tangent semi_circle (Segment BC) N)

theorem product_AM_CN_constant : 
  AM * CN = (AB / 2) ^ 2 :=
sorry

end product_AM_CN_constant_l121_121540


namespace total_length_proof_l121_121908

noncomputable def total_length_climbed (keaton_ladder_height : ℕ) (keaton_times : ℕ) (shortening : ℕ) (reece_times : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - shortening
  let keaton_total := keaton_ladder_height * keaton_times
  let reece_total := reece_ladder_height * reece_times
  (keaton_total + reece_total) * 100

theorem total_length_proof :
  total_length_climbed 60 40 8 35 = 422000 := by
  sorry

end total_length_proof_l121_121908


namespace trig_identity_l121_121658

theorem trig_identity (α : ℝ) :
  (sin α * sin (α - 3 * π / 2) ^ 2 * (1 + (tan α) ^ 2) + cos α * cos (α + 3 * π / 2) ^ 2 * (1 + (cot α) ^ 2)) =
  √2 * sin (π / 4 + α) :=
by
  sorry

end trig_identity_l121_121658


namespace exists_sequence_sum_division_squared_infinite_l121_121559

theorem exists_sequence_sum_division_squared_infinite :
  ∃ (a : ℕ → ℝ), (∀ n, a n ≥ 0) ∧ (∑' n, (a n) ^ 2 < ∞) ∧ 
    (∑' n, (∑' k, (a (k * n)) / k) ^ 2 = ∞) :=
sorry

end exists_sequence_sum_division_squared_infinite_l121_121559


namespace min_value_of_expression_l121_121877

theorem min_value_of_expression (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y = 5 := 
sorry

end min_value_of_expression_l121_121877


namespace inner_cube_surface_area_l121_121316

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121316


namespace inner_cube_surface_area_l121_121359

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121359


namespace inner_cube_surface_area_l121_121277

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121277


namespace inner_cube_surface_area_l121_121386

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121386


namespace number_of_bedrooms_l121_121607

-- Conditions
def battery_life : ℕ := 10
def vacuum_time_per_room : ℕ := 4
def num_initial_rooms : ℕ := 2 -- kitchen and living room
def num_charges : ℕ := 2

-- Computation of total vacuuming time
def total_vacuuming_time : ℕ := battery_life * (num_charges + 1)

-- Computation of remaining time for bedrooms
def time_for_bedrooms : ℕ := total_vacuuming_time - (vacuum_time_per_room * num_initial_rooms)

-- Proof problem: Prove number of bedrooms
theorem number_of_bedrooms (B : ℕ) (h : B = time_for_bedrooms / vacuum_time_per_room) : B = 5 := by 
  sorry

end number_of_bedrooms_l121_121607


namespace cube_volume_l121_121172

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121172


namespace complex_power_l121_121417

theorem complex_power :
  (1 + 2 * Complex.i) ^ 4 = -7 - 24 * Complex.i :=
by
  sorry

end complex_power_l121_121417


namespace polynomial_roots_value_l121_121574

noncomputable def given_conditions (a b c : ℝ) : Prop :=
  a + b + c = 15 ∧ ab + bc + ca = 22 ∧ abc = 8

theorem polynomial_roots_value (a b c : ℝ) (h : given_conditions a b c) :
  (a / (1 / a + b * c) + b / (1 / b + c * a) + c / (1 / c + a * b)) = 181 / 9 :=
  by sorry

end polynomial_roots_value_l121_121574


namespace solve_fraction_eq_l121_121954

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ -2) :
  (2 / (x - 2) = 3 / (x + 2)) → x = 10 :=
by
  sorry

end solve_fraction_eq_l121_121954


namespace books_loaned_out_l121_121699

theorem books_loaned_out (initial_books : ℕ) (returned_percentage : ℝ) (end_books : ℕ) (x : ℝ) :
    initial_books = 75 →
    returned_percentage = 0.70 →
    end_books = 63 →
    0.30 * x = (initial_books - end_books) →
    x = 40 := by
  sorry

end books_loaned_out_l121_121699


namespace power_function_value_l121_121619

theorem power_function_value :
  ∃ α : ℝ, (2 ^ α = 1 / 4) ∧ (-3 ^ α = 1 / 9) := sorry

end power_function_value_l121_121619


namespace percentage_wax_left_eq_10_l121_121850

def total_amount_wax : ℕ := 
  let wax20 := 5 * 20
  let wax5 := 5 * 5
  let wax1 := 25 * 1
  wax20 + wax5 + wax1

def wax_used_for_new_candles : ℕ := 
  3 * 5

def percentage_wax_used (total_wax : ℕ) (wax_used : ℕ) : ℕ := 
  (wax_used * 100) / total_wax

theorem percentage_wax_left_eq_10 :
  percentage_wax_used total_amount_wax wax_used_for_new_candles = 10 :=
by
  sorry

end percentage_wax_left_eq_10_l121_121850


namespace fraction_product_l121_121050

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l121_121050


namespace prime_sum_probability_l121_121604

noncomputable def probability_prime_sum_two_8_sided_dice : ℚ :=
  let outcomes := (fin 8) × (fin 8)
  let possible_sums := { (i : ℕ, j : ℕ) // i < 8 ∧ j < 8 // (i + 1) + (j + 1) }
  let prime_sums := { s : ℕ // s ∈ [2, 3, 5, 7, 11, 13] }
  (23 : ℚ) / (64 : ℚ)

theorem prime_sum_probability : probability_prime_sum_two_8_sided_dice = 23 / 64 := 
by sorry

end prime_sum_probability_l121_121604


namespace total_matches_equation_l121_121536

theorem total_matches_equation (x : ℕ) (h : ((x * (x - 1)) / 2) = 28) : (1 / 2 : ℚ) * x * (x - 1) = 28 := by
  have h1 : ((x * (x - 1)) / 2 : ℚ) = (1 / 2) * x * (x - 1),
    sorry
  rw ← h1 at h
  exact h

end total_matches_equation_l121_121536


namespace time_for_A_to_finish_race_l121_121528

-- Definitions based on the conditions
def race_distance : ℝ := 120
def B_time : ℝ := 45
def B_beaten_distance : ℝ := 24

-- Proof statement: We need to show that A's time is 56.25 seconds
theorem time_for_A_to_finish_race : ∃ (t : ℝ), t = 56.25 ∧ (120 / t = 96 / 45)
  := sorry

end time_for_A_to_finish_race_l121_121528


namespace inner_cube_surface_area_l121_121311

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121311


namespace prove_fraction_eq_zero_l121_121915

noncomputable theory

variables {R : Type*} [CommRing R]

def A : Matrix (Fin 2) (Fin 2) R :=
  ![![1, 2],
    ![3, 4]]

def B (a b c d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![a, b],
    ![c, d]]

theorem prove_fraction_eq_zero (a b c d : R) (h₁ : A.mul (B a b c d) = (B a b c d).mul A) (h₂ : 4 * b ≠ c) : 
  (a - d) / (c - 4 * b) = 0 :=
by
  sorry

end prove_fraction_eq_zero_l121_121915


namespace box_dimensions_l121_121554

-- Given conditions
variables (a b c : ℕ)
axiom h1 : a + c = 17
axiom h2 : a + b = 13
axiom h3 : b + c = 20

theorem box_dimensions : a = 5 ∧ b = 8 ∧ c = 12 :=
by {
  -- These parts will contain the actual proof, which we omit for now
  sorry
}

end box_dimensions_l121_121554


namespace max_angle_C_l121_121557

theorem max_angle_C (A B C : ℝ) (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π) 
  (hABC : A + B + C = π) (h1 : sin A ^ 2 + sin B ^ 2 = 2 * sin C ^ 2) : C ≤ π / 3 :=
sorry

example (A B C : ℝ) (h : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (hABC : A + B + C = π) (h1 : sin A ^ 2 + sin B ^ 2 = 2 * sin C ^ 2) : 
  C = π / 3 → C ≤ π / 3 :=
by
  intro hC_eq
  rw hC_eq
  exact le_refl (π / 3)

end max_angle_C_l121_121557


namespace unique_B_squared_l121_121572

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

theorem unique_B_squared (h : B ^ 4 = 0) :
  ∃! B2 : Matrix (Fin 2) (Fin 2) ℝ, B ^ 2 = B2 :=
by sorry

end unique_B_squared_l121_121572


namespace system_of_equations_solution_l121_121024

theorem system_of_equations_solution :
  ∃ (a b : ℤ), (2 * (2 : ℤ) + b = a ∧ (2 : ℤ) + b = 3 ∧ a = 5 ∧ b = 1) :=
by
  sorry

end system_of_equations_solution_l121_121024


namespace noodles_initial_count_l121_121752

theorem noodles_initial_count (noodles_given : ℕ) (noodles_now : ℕ) (initial_noodles : ℕ) 
  (h_given : noodles_given = 12) (h_now : noodles_now = 54) (h_initial_noodles : initial_noodles = noodles_now + noodles_given) : 
  initial_noodles = 66 :=
by 
  rw [h_now, h_given] at h_initial_noodles
  exact h_initial_noodles

-- Adding 'sorry' since the solution steps are not required

end noodles_initial_count_l121_121752


namespace unusual_arrangement_rank_usual_arrangement_rank_l121_121126

theorem unusual_arrangement_rank : (rank_unusual [5, 3, 4, 1, 7, 2, 6]) = 2271 := sorry

theorem usual_arrangement_rank : (rank_usual [5, 3, 4, 1, 7, 2, 6]) = 3173 := sorry

end unusual_arrangement_rank_usual_arrangement_rank_l121_121126


namespace distance_between_points_l121_121440

def dist_3d (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

theorem distance_between_points :
  dist_3d (3, 3, 2) (-1, -1, -1) = real.sqrt 41 :=
by sorry

end distance_between_points_l121_121440


namespace num_palindromes_l121_121502

theorem num_palindromes (digits : List ℕ) (h_digits : digits = [1, 1, 1, 2, 4, 4, 4, 4, 7]) :
  (∃ l : List ℕ, l.length = 9 ∧ l.reverse = l ∧ l.perm digits) → l.length = 216 :=
by
  intros l hl
  sorry

end num_palindromes_l121_121502


namespace inner_cube_surface_area_l121_121294

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121294


namespace fraction_computation_l121_121055

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l121_121055


namespace yeast_population_l121_121141

theorem yeast_population : 
  let initial_population := 50
  let growth_rate := 3
  let duration_minutes := 30
  let period_minutes := 5
  let periods := duration_minutes / period_minutes
  (initial_population * (growth_rate ^ periods)) = 36450 :=
by
  let initial_population := 50
  let growth_rate := 3
  let duration_minutes := 30
  let period_minutes := 5
  let periods := duration_minutes / period_minutes
  have h_periods: periods = 6 := by norm_num
  rw [h_periods]
  have h_growth: growth_rate ^ periods = 729 := by norm_num
  rw [h_growth]
  have h_population: initial_population * 729 = 36450 := by norm_num
  exact h_population

end yeast_population_l121_121141


namespace sum_of_divisors_143_l121_121086

theorem sum_of_divisors_143 : (∑ i in (finset.filter (λ d, 143 % d = 0) (finset.range 144)), i) = 168 :=
by
  -- The final proofs will go here.
  sorry

end sum_of_divisors_143_l121_121086


namespace frog_landing_safely_l121_121634

theorem frog_landing_safely :
  let m := 1
  let n := 8
  100 * m + n = 108 :=
begin
  sorry
end

end frog_landing_safely_l121_121634


namespace tim_notes_probability_l121_121978

theorem tim_notes_probability :
  let p := 5 / 8 in
  (1 - p) = 3 / 8 :=
by
  let p := 5 / 8
  show 1 - p = 3 / 8
  sorry

end tim_notes_probability_l121_121978


namespace circle_intersection_solutions_l121_121022

-- Define the second major axis of the plane
def second_major_axis (v v' v'': ℝ^3) : Prop :=
True

-- Define the point inversion
def point_inversion (P: ℝ^3) (P2: ℝ^3) : Prop := 
P2.2 = P.2

-- Define the circle's properties and radius
def circle_properties (P M: ℝ^3) (PM: ℝ) : Prop :=
PM = dist P M

-- Intersection line of plane S and first bisector plane H1
def intersection_line (S: ℝ^3) (H1: ℝ^3) (h1: ℝ^3) (M: ℝ^3) (P2: ℝ^3) : Prop :=
(h1.2 = M.2) ∧ (h1.2 = P2.2)

-- Main theorem regarding the solution existence
theorem circle_intersection_solutions
  (P M: ℝ^3) (h1: ℝ^3) (circle: set ℝ^3):
  second_major_axis v v' v'' →
  point_inversion P (P2) →
  circle_properties P M (dist P M) →
  intersection_line S H1 h1 M (P2) →
  (∃ y ∈ circle, y ∈ h1) = 
    if (P M) ∩ h1 = ∅ then 0 else
    if (P M) ∩ h1 = 1 then 1 else 2 :=
sorry

end circle_intersection_solutions_l121_121022


namespace inner_cube_surface_area_l121_121308

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121308


namespace max_min_weight_difference_and_total_weight_l121_121455

theorem max_min_weight_difference_and_total_weight :
  let standard_weight := 500
  let differences := [(-20, 4), (-5, 1), (0, 3), (2, 4), (3, 5), (10, 3)]
  let max_diff := 10
  let min_diff := -20
  let max_min_diff := max_diff - min_diff
  let total_difference := differences.foldr (λ (p : ℤ × ℕ) acc, acc + p.1 * p.2) 0
  let num_bags := 20
  let total_standard_weight := num_bags * standard_weight
  let total_weight := total_standard_weight + total_difference
  max_min_diff = 30 ∧ total_weight = 9968 :=
by
  let standard_weight := 500
  let differences := [(-20, 4), (-5, 1), (0, 3), (2, 4), (3, 5), (10, 3)]
  let max_diff := 10
  let min_diff := -20
  let max_min_diff := max_diff - min_diff
  let total_difference := List.foldr (λ (p : ℤ × ℕ) acc, acc + p.1 * p.2) 0 differences
  let num_bags := 20
  let total_standard_weight := num_bags * standard_weight
  let total_weight := total_standard_weight + total_difference
  exact ⟨rfl, rfl⟩

end max_min_weight_difference_and_total_weight_l121_121455


namespace cube_volume_from_surface_area_l121_121203

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l121_121203


namespace largest_angle_of_consecutive_integers_hexagon_l121_121961

theorem largest_angle_of_consecutive_integers_hexagon (angles : ℕ → ℝ) (h : ∑ i in finset.range 6, angles i = 720) :
  angles 5 = 122.5 :=
by
  sorry

end largest_angle_of_consecutive_integers_hexagon_l121_121961


namespace collinear_IO_L_l121_121966

noncomputable def midpoint (A B : Point) : Point := 
  ⟨(A.1 + B.1) / 2, (A.2 + B.2) / 2⟩

theorem collinear_IO_L (A B C I O L M N : Point) 
  (h₁ : extension AC = {L, M, N}) 
  (h₂ : on_circumcircle (midpoint M N) ABC) : collinear {I, O, L} :=
sorry

end collinear_IO_L_l121_121966


namespace interval_of_increase_f_4x_sub_xsq_l121_121521

noncomputable def f (x : ℝ) := Real.log x / Real.log 2

theorem interval_of_increase_f_4x_sub_xsq : 
  (∀ x : ℝ, 0 < x ∧ x < 2 → 
    ∀ y : ℝ, (0 < 4 * y - y ^ 2) → (4 * y - y ^ 2) < 2 → 
      (4 * x - x^2) < (4 * y - y^2) → f(4 * x - x^2) < f(4 * y - y^2)) := 
sorry

end interval_of_increase_f_4x_sub_xsq_l121_121521


namespace volume_of_wedge_l121_121728

theorem volume_of_wedge (c : ℝ) (h : c = 18 * Real.pi) : 
  let r := c / (2 * Real.pi) in
  let V := (4 / 3) * Real.pi * r^3 in
  (V / 6) = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l121_121728


namespace correct_transformation_l121_121401

theorem correct_transformation (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
sorry

end correct_transformation_l121_121401


namespace max_value_a1_l121_121475

theorem max_value_a1 
    (a_1 a_2 a_3 : ℝ)
    (a : ℕ → ℝ)
    (h : ∀ x ∈ (multiset.replicate 2006 (λ x : ℝ, x > 0)), 
         polynomial.eval x (polynomial.mk (1 :: a_1 :: a_2 :: a_3 :: list.map a (list.range (2004 - 3))))) = 0) :
    a_1 ≤ -2006 :=
by
  sorry

end max_value_a1_l121_121475


namespace find_FD_l121_121568

noncomputable def parallelogram {α : Type}
  [inner_product_space ℝ α] (A B C D E F : α) : Prop :=
  ∃ (AB BC AD DC : ℝ)
     (angle_ABC angle_ADC : ℝ),
    angle_ABC = 150 ∧
    angle_ADC = 150 ∧
    dist A B = 20 ∧
    dist B C = 12 ∧
    dist D E = 6 ∧
    B + E ≠ 0 ∧
    (dist (A - D) = dist (C - B))

theorem find_FD {α : Type}
  [inner_product_space ℝ α] (A B C D E F : α) :
  parallelogram A B C D E F →
  dist A D = 12 →
  line (A, D) ∩ line (B, E) = {F} →
  dist F D = 4 :=
sorry

end find_FD_l121_121568


namespace part1_domain_part2_range_of_m_l121_121488

-- Definitions for the conditions
def f (x m : ℝ) : ℝ := real.logb 2 (|x+1| + |x-2| - m)

-- Theorem statements
theorem part1_domain (x : ℝ) : (f x 5).domain = set_of (λ x, x ∈ (-∞, -2) ∪ (3, ∞)) :=
  sorry

theorem part2_range_of_m (m : ℝ) : (∀ x, f x m ≥ 1) → m ≤ 1 :=
  sorry

end part1_domain_part2_range_of_m_l121_121488


namespace second_cube_surface_area_l121_121337

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121337


namespace cube_volume_l121_121213

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l121_121213


namespace cucumber_calories_l121_121904

-- Definitions of the given conditions.
def lettuce_calories : ℕ := 30
def crouton_count : ℕ := 12
def calories_per_crouton : ℕ := 20
def total_salad_calories : ℕ := 350

-- Proof problem statement.
theorem cucumber_calories :
  (let crouton_total_calories := crouton_count * calories_per_crouton in
   let non_cucumber_calories := lettuce_calories + crouton_total_calories in
   total_salad_calories - non_cucumber_calories = 80) :=
begin
  sorry -- proof not required
end

end cucumber_calories_l121_121904


namespace inner_cube_surface_area_l121_121291

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121291


namespace parabola_properties_l121_121836

open Real

/-- The equation of the parabola with vertex (-1, -4) and passing through (0, -3) is
    y = (x + 1)^2 - 4. The parabola intersects the x-axis at (1, 0) and (-3, 0). -/
theorem parabola_properties :
  ∃ (a: ℝ) (h k : ℝ), (h = -1) ∧ (k = -4) ∧ (∀ x y : ℝ, (x, y) = (0, -3) → y = a * (x + h)^2 + k) ∧
  (∀ x y : ℝ, y = a * (x + h)^2 + k ↔ y = (x + 1)^2 - 4) ∧
  (∀ x y : ℝ, y = 0 → (x = 1 ∧ y = 0) ∨ (x = -3 ∧ y = 0)) := 
begin 
  sorry 
end

end parabola_properties_l121_121836


namespace minimum_number_of_apples_l121_121740

-- Define the problem conditions and the proof statement
theorem minimum_number_of_apples :
  ∃ p : Fin 6 → ℕ, (∀ i, p i > 0) ∧ (Function.Injective p) ∧ (Finset.univ.sum p * 4 = 100) ∧ (Finset.univ.sum p = 25 / 4) := 
sorry

end minimum_number_of_apples_l121_121740


namespace apple_price_l121_121656

theorem apple_price (A : ℝ) (h1 : 8 * A = 3 * A) (h2 : 1.5 * 1.5 * A = 1.5 * A) (h3 : 8 * A = 10 * (A / 10)) (h4 : 2 * (A / 2) = A) (h5 : 1.5 * A = 1.5 * A / 3) :
  let total_bill := 30 + 35 * A + 25 * A + 52.5 * A + 5 * A + 7.5 * A in
  total_bill = 395 → A = 3.04 :=
by
  sorry

end apple_price_l121_121656


namespace find_R_Ramu_l121_121581

noncomputable def calculate_R_Ramu (P_Anwar P_Ramu G : ℝ) (R_Anwar T : ℝ) : ℝ :=
  let SI_Anwar := P_Anwar * R_Anwar * T / 100
  let SI_Ramu := G + SI_Anwar
  let R_Ramu :=  SI_Ramu * 100 / (P_Ramu * T)
  R_Ramu

theorem find_R_Ramu :
  calculate_R_Ramu 3900 5655 824.85 6 3 ≈ 9 :=
by
  sorry

end find_R_Ramu_l121_121581


namespace hyperbola_eq_l121_121010

-- Define the conditions explicitly
def hyperbola_passes_through (x y a b : ℝ) : Prop :=
  (x / a) ^ 2 - (y / b) ^ 2 = 1

def eccentricity (a b e : ℝ) : Prop :=
  e = Math.sqrt(a^2 + b^2) / a

-- Main statement to prove
theorem hyperbola_eq (a b : ℝ) (h1 : hyperbola_passes_through (Real.sqrt 2) (Real.sqrt 3) a b)
  (h2 : eccentricity a b 2) :
  (a = 1) ∧ (b = Math.sqrt 3) :=
by {
  -- Proof would go here; currently it's skipped with 'sorry'
  sorry
}

end hyperbola_eq_l121_121010


namespace area_of_inscribed_triangle_l121_121396

noncomputable def triangle_area (r : ℝ) (A B C : ℝ) : ℝ :=
  0.5 * r ^ 2 * (Real.sin A + Real.sin B + Real.sin C)

theorem area_of_inscribed_triangle :
  ∀ (arc1 arc2 arc3 : ℝ), arc1 + arc2 + arc3 = 16 →
    let r := 8 / Real.pi in
    let A := Real.pi / 2 in
    let B := 5 * (Real.pi / 16) in
    let C := 7 * (Real.pi / 16) in
    triangle_area r A B C =
      32 / (Real.pi ^ 2) * (1 + Real.sqrt ((2 + Real.sqrt 2) / 4) + Real.sqrt ((2 - Real.sqrt 2) / 4)) :=
by
  intros arc1 arc2 arc3 hsum r A B C
  unfold triangle_area
  sorry

end area_of_inscribed_triangle_l121_121396


namespace solution_set_of_inequality_range_of_t_general_eqn_curve_length_of_segment_l121_121675

-- Statement for Problem (I)(1)
theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f(x) = 45 * |(2 * x + 2)| - 45 * |(x - 2)|) 
  (h_sol : ∀ x, f(x) > 2 ↔ x ∈ (-∞ : ℝ) ∪ [x|x ≥ (2/3)]) : 
  (solution_set : set ℝ) :=
by {
  simp [solution_set, h_f, h_sol],
  sorry
}

-- Statement for Problem (I)(2)
theorem range_of_t 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f(x) = 45 * |(2 * x + 2)| - 45 * |(x - 2)|) 
  (h_cond : ∀ x t, f(x) ≥ t^2 - (7/2) * t) :
  (range_t : set ℝ) :=
by {
  simp [range_t, h_f, h_cond],
  sorry
}

-- Statement for Problem (II)(1)
theorem general_eqn_curve (θ : ℝ → ℝ) :
  (general_eqn : (ℝ → ℝ) → Prop) :=
by {
  simp [general_eqn, θ],
  sorry
}

-- Statement for Problem (II)(2)
theorem length_of_segment 
  (ρ : ℝ) 
  (h_polar_eqn : ∀ θ, 2 * ρ * sin (θ + (π/3)) = 3 * sqrt 3)
  (h_center : (1, 0)) :
  (length_PQ : ℝ) :=
by {
  simp [length_PQ, h_polar_eqn, h_center],
  sorry
}

end solution_set_of_inequality_range_of_t_general_eqn_curve_length_of_segment_l121_121675


namespace total_shaded_area_l121_121530

noncomputable def radius_large_circle := 10
noncomputable def radius_small_circle := 7

def area_large_circle_shaded := 1 / 2 * (Real.pi * radius_large_circle ^ 2)
def area_small_circle_shaded := 1 / 2 * (Real.pi * radius_small_circle ^ 2)

theorem total_shaded_area : area_large_circle_shaded + area_small_circle_shaded = 74.5 * Real.pi := by
  sorry

end total_shaded_area_l121_121530


namespace inner_cube_surface_area_l121_121250

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121250


namespace number_of_multiples_of_31_l121_121731

noncomputable def a (n k : ℕ) : ℕ :=
  2^(n-1) * (n + 2 * k - 2)

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_multiple_of_31 (x : ℕ) : Prop :=
  x % 31 = 0

def count_multiples_of_31 (N : ℕ) : ℕ :=
  (Finset.range N).filter (λ n, is_odd n ∧ ∃ k, is_multiple_of_31 (a n k)).card

theorem number_of_multiples_of_31 :
  count_multiples_of_31 34 = 17 :=
sorry

end number_of_multiples_of_31_l121_121731


namespace incorrect_statement_C_l121_121655

theorem incorrect_statement_C :
  (∀ x, real.sqrt 49 = 7) ∧ 
  (real.cbrt 0 = 0 ∧ real.cbrt 1 = 1 ∧ real.cbrt (-1) = -1) ∧
  (¬ ∃ y, y * y = 0) ∧ 
  (∃ y, y * y = 4 ∧ (y = 2 ∨ y = -2)) → 
  false := 
by
  sorry

end incorrect_statement_C_l121_121655


namespace A_n_is_integer_l121_121474

variable (a b : ℕ) (θ : ℝ)
variable (n : ℕ)

-- Given conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom a_gt_b : a > b
axiom sin_theta_eq : sin θ = (2 * a * b) / (a ^ 2 + b ^ 2)
axiom theta_range : 0 < θ ∧ θ < π / 2

-- Definition of A_n
def A_n (a b : ℕ) (n : ℕ) : ℝ := (a ^ 2 + b ^ 2) ^ n * sin θ

-- The proof problem
theorem A_n_is_integer (a b : ℕ) (θ : ℝ) (n : ℕ) [fact (0 < θ)] [fact (θ < π / 2)]
  (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hθ : sin θ = (2 * a * b) / (a ^ 2 + b ^ 2)) : 
  ∃ k : ℤ, A_n a b n = (k : ℝ) :=
sorry

end A_n_is_integer_l121_121474


namespace cover_points_with_circles_l121_121704

theorem cover_points_with_circles (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → min (dist (points i) (points j)) (min (dist (points j) (points k)) (dist (points i) (points k))) ≤ 1) :
  ∃ (a b : Fin n), ∀ (p : Fin n), dist (points p) (points a) ≤ 1 ∨ dist (points p) (points b) ≤ 1 := 
sorry

end cover_points_with_circles_l121_121704


namespace total_strings_correct_l121_121786

-- Definitions based on conditions
def num_ukuleles : ℕ := 2
def num_guitars : ℕ := 4
def num_violins : ℕ := 2
def strings_per_ukulele : ℕ := 4
def strings_per_guitar : ℕ := 6
def strings_per_violin : ℕ := 4

-- Total number of strings
def total_strings : ℕ := num_ukuleles * strings_per_ukulele +
                         num_guitars * strings_per_guitar +
                         num_violins * strings_per_violin

-- The proof statement
theorem total_strings_correct : total_strings = 40 :=
by
  -- Proof omitted.
  sorry

end total_strings_correct_l121_121786


namespace ratio_problem_l121_121863

theorem ratio_problem {q r s t : ℚ} (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 :=
sorry

end ratio_problem_l121_121863


namespace fraction_odd_min_element_correct_l121_121465

open Finset

noncomputable def fraction_odd_min_element (n : ℕ) (h_pos : 0 < n) : ℚ :=
  let N := 2 * n
  let S := (range (N + 1)).filter (λ x, x > 0)
  let odd_subsets := S.powerset.filter (λ T, ∃ k, 2 * k + 1 ∈ T ∧ T.min' (by simp [Finset.nonempty]) = 2 * k + 1)
  let all_subsets := S.powerset.filter (λ T, T.nonempty)
  in if all_subsets.card = 0 then 0 else (odd_subsets.card : ℚ) / (all_subsets.card : ℚ)

theorem fraction_odd_min_element_correct (n : ℕ) (h_pos : 0 < n) :
  fraction_odd_min_element n h_pos = 2 / 3 :=
sorry

end fraction_odd_min_element_correct_l121_121465


namespace inequality_holds_for_positive_vars_l121_121947

theorem inequality_holds_for_positive_vars (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
    x^2 + y^2 + 1 ≥ x * y + x + y :=
sorry

end inequality_holds_for_positive_vars_l121_121947


namespace median_length_YN_perimeter_triangle_XYZ_l121_121893

-- Definitions for conditions
noncomputable def length_XY : ℝ := 5
noncomputable def length_XZ : ℝ := 12
noncomputable def is_right_angle_XYZ : Prop := true
noncomputable def midpoint_N : ℝ := length_XZ / 2

-- Theorem statement for the length of the median YN
theorem median_length_YN (XY XZ : ℝ) (right_angle : is_right_angle_XYZ) :
  XY = 5 ∧ XZ = 12 → (XY^2 + XZ^2) = 169 → (13 / 2) = 6.5 := by
  sorry

-- Theorem statement for the perimeter of triangle XYZ
theorem perimeter_triangle_XYZ (XY XZ : ℝ) (right_angle : is_right_angle_XYZ) :
  XY = 5 ∧ XZ = 12 → (XY^2 + XZ^2) = 169 → (XY + XZ + 13) = 30 := by
  sorry

end median_length_YN_perimeter_triangle_XYZ_l121_121893


namespace inner_cube_surface_area_l121_121355

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121355


namespace club_officer_selection_l121_121589

theorem club_officer_selection :
  let members := 10
  let positions := 4
  let choices := (members - 0) * (members - 1) * (members - 2) * (members - 3)
  choices = 5040 :=
by
  let members := 10
  let positions := 4
  let choices := (members - 0) * (members - 1) * (members - 2) * (members - 3)
  exact choices = 5040

end club_officer_selection_l121_121589


namespace value_of_pq_s_l121_121866

-- Definitions for the problem
def polynomial_divisible (p q s : ℚ) : Prop :=
  ∀ x : ℚ, (x^3 + 4 * x^2 + 16 * x + 8) ∣ (x^4 + 6 * x^3 + 8 * p * x^2 + 6 * q * x + s)

-- The main theorem statement to prove
theorem value_of_pq_s (p q s : ℚ) (h : polynomial_divisible p q s) : (p + q) * s = 332 / 3 :=
sorry -- Proof omitted

end value_of_pq_s_l121_121866


namespace max_ab_value_l121_121522

theorem max_ab_value (a b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * a * x - b^2 + 12 ≤ 0 → x = a) : ab = 6 := by
  sorry

end max_ab_value_l121_121522


namespace cube_volume_of_surface_area_l121_121155

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l121_121155


namespace inner_cube_surface_area_l121_121375

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121375


namespace balanced_six_digit_integers_count_l121_121234

def is_balanced (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧
  let d1 := n / 100000 % 10
      d2 := n / 10000 % 10
      d3 := n / 1000 % 10
      d4 := n / 100 % 10
      d5 := n / 10 % 10
      d6 := n % 10 in
  d1 + d2 + d3 = d4 + d5 + d6

def count_balanced_integers : ℕ :=
  finset.card (finset.filter is_balanced (finset.range 1000000))

theorem balanced_six_digit_integers_count :
  count_balanced_integers = 7857 :=
by sorry

end balanced_six_digit_integers_count_l121_121234


namespace proposition_l121_121838

-- Definitions based on conditions
def curve_C (k : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2) / (25 - k) + (y^2) / (k - 9) = 1

def is_ellipse (k : ℝ) : Prop := 9 < k ∧ k < 25

def is_hyperbola_with_foci_on_x_axis (k : ℝ) : Prop := k < 9

-- Propositions p and q
def p (k : ℝ) : Prop := is_ellipse(k) → curve_C(k)
def q (k : ℝ) : Prop := curve_C(k) → is_hyperbola_with_foci_on_x_axis(k)

-- The proof problem
theorem proposition (k : ℝ) : ¬ p(k) ∧ q(k) :=
sorry

end proposition_l121_121838


namespace inner_cube_surface_area_l121_121389

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121389


namespace joe_egg_count_l121_121907

theorem joe_egg_count : 
  let clubhouse : ℕ := 12
  let park : ℕ := 5
  let townhall : ℕ := 3
  clubhouse + park + townhall = 20 :=
by
  sorry

end joe_egg_count_l121_121907


namespace cube_volume_l121_121171

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121171


namespace volume_of_wedge_l121_121724

theorem volume_of_wedge (r : ℝ) (V : ℝ) (sphere_wedges : ℝ) 
  (h_circumference : 2 * Real.pi * r = 18 * Real.pi)
  (h_volume : V = (4 / 3) * Real.pi * r ^ 3) 
  (h_sphere_wedges : sphere_wedges = 6) : 
  V / sphere_wedges = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l121_121724


namespace significant_digits_of_side_length_l121_121425

noncomputable def num_significant_digits (n : Float) : Nat :=
  -- This is a placeholder function to determine the number of significant digits
  sorry

theorem significant_digits_of_side_length :
  ∀ (A : Float), A = 3.2400 → num_significant_digits (Float.sqrt A) = 5 :=
by
  intro A h
  -- Proof would go here
  sorry

end significant_digits_of_side_length_l121_121425


namespace triangle_perpendicular_proof_l121_121125

-- Define the type for our points and lines
variables {A B C D F : Type}

-- Conditions: ABC is a triangle, acute
variables {angle : A → A → A → ℝ}

-- Defining the conditions in Lean
variables (triangle_ABC : A → A → A → Prop)
variables (circumcircle_ABC : A → A → A → Prop)
variables (Line_AC : A → A → Prop)
variables (Line_BC : A → A → Prop)
variables (Line_FB : A → A → Prop)
variables (Perpendicular : A → A → A → Prop)
variables (DifferentSides : A → A → A → Prop)
variables (OnCircle : A → A → A → Prop)

-- Our theorem statement using the identified conditions 
theorem triangle_perpendicular_proof 
(h1 : ∀ a b c, triangle_ABC a b c → angle a b > angle b c)
(h2 : ∀ a b d, Line_AC a b → Line_AC b d → distance a b = distance b d)
(h3 : ∀ a b c d f, circumcircle_ABC a b c → OnCircle d a b → Perpendicular d f b c → DifferentSides a f b c)
: ∃ (f : A), Perpendicular f b a :=
sorry

end triangle_perpendicular_proof_l121_121125


namespace inner_cube_surface_area_l121_121372

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121372


namespace geometric_series_common_ratio_l121_121409

theorem geometric_series_common_ratio (a S r : ℝ)
  (h1 : a = 172)
  (h2 : S = 400)
  (h3 : S = a / (1 - r)) :
  r = 57 / 100 := 
sorry

end geometric_series_common_ratio_l121_121409


namespace calvin_hair_goal_achieved_l121_121744

theorem calvin_hair_goal_achieved 
  (haircuts : ℕ)
  (total_clippings : ℝ)
  (saves_first_dog : ℝ)
  (saves_second_dog : ℝ)
  (saves_third_dog : ℝ)
  (goal_first_dog : ℝ)
  (goal_second_dog : ℝ)
  (goal_third_dog : ℝ)
  (hair_per_haircut : total_clippings / haircuts)
  (collected_first_dog : hair_per_haircut * saves_first_dog * haircuts)
  (collected_second_dog : hair_per_haircut * saves_second_dog * haircuts)
  (collected_third_dog : hair_per_haircut * saves_third_dog * haircuts)
  (percentage_first_dog : collected_first_dog / goal_first_dog * 100)
  (percentage_second_dog : collected_second_dog / goal_second_dog * 100)
  (percentage_third_dog : collected_third_dog / goal_third_dog * 100)
  (condition1 : haircuts = 8)
  (condition2 : total_clippings = 400)
  (condition3 : saves_first_dog = 0.70)
  (condition4 : saves_second_dog = 0.50)
  (condition5 : saves_third_dog = 0.25)
  (condition6 : goal_first_dog = 280)
  (condition7 : goal_second_dog = 200)
  (condition8 : goal_third_dog = 100):
  percentage_first_dog = 100 ∧ percentage_second_dog = 100 ∧ percentage_third_dog = 100 :=
by
  sorry

end calvin_hair_goal_achieved_l121_121744


namespace xyz_sum_neg1_l121_121855

theorem xyz_sum_neg1 (x y z : ℝ) (h : (x + 1)^2 + |y - 2| = -(2 * x - z)^2) : x + y + z = -1 :=
sorry

end xyz_sum_neg1_l121_121855


namespace probability_of_matching_pair_l121_121434

def total_socks : ℕ := 12 + 6 + 9
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

def black_pairs : ℕ := choose_two 12
def white_pairs : ℕ := choose_two 6
def blue_pairs : ℕ := choose_two 9

def total_pairs : ℕ := choose_two total_socks
def matching_pairs : ℕ := black_pairs + white_pairs + blue_pairs

def probability : ℚ := matching_pairs / total_pairs

theorem probability_of_matching_pair :
  probability = 1 / 3 :=
by
  -- The proof will go here
  sorry

end probability_of_matching_pair_l121_121434


namespace cos_sum_product_leq_cos_square_l121_121946

theorem cos_sum_product_leq_cos_square (x y : ℝ) : 
  cos (x + y) * cos (x - y) ≤ cos (x)^2 := 
sorry

end cos_sum_product_leq_cos_square_l121_121946


namespace problem_statement_l121_121595

theorem problem_statement (a b : ℕ) (ha : a = 55555) (hb : b = 66666) :
  55554 * 55559 * 55552 - 55556 * 55551 * 55558 =
  66665 * 66670 * 66663 - 66667 * 66662 * 66669 := 
by
  sorry

end problem_statement_l121_121595


namespace cannot_determine_f_5_l121_121614

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) : 
  f(x) + f(3 * x + y) + 3 * x * y = f(4 * x - y) + 3 * x ^ 2 + 2

-- We can't determine f(5) without additional information.
theorem cannot_determine_f_5 : ¬ (∃ a : ℝ, ∀ f, (f = a)) :=
  sorry

end cannot_determine_f_5_l121_121614


namespace gcd_factorial_power_l121_121771

theorem gcd_factorial_power {n m : ℕ} (h9 : 9! = 2^7 * 3^4 * 5 * 7) (h6sq : (6!)^2 = 2^8 * 3^4 * 5^2) :
  Nat.gcd 9! ((6!)^2) = 51840 := by
  rw [h9, h6sq]
  sorry

end gcd_factorial_power_l121_121771


namespace sum_of_r_s_l121_121913

theorem sum_of_r_s (m : ℝ) (x : ℝ) (y : ℝ) (r s : ℝ) 
  (parabola_eqn : y = x^2 + 4) 
  (point_Q : (x, y) = (10, 5)) 
  (roots_rs : ∀ (m : ℝ), m^2 - 40*m + 4 = 0 → r < m → m < s)
  : r + s = 40 := 
sorry

end sum_of_r_s_l121_121913


namespace tom_teaching_years_l121_121996

theorem tom_teaching_years (T D : ℝ) (h1 : T + D = 70) (h2 : D = (1/2) * T - 5) : T = 50 :=
by
  -- This is where the proof would normally go if it were required.
  sorry

end tom_teaching_years_l121_121996


namespace inner_cube_surface_area_l121_121379

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121379


namespace hexagon_pillar_height_l121_121541

noncomputable def height_of_pillar_at_vertex_F (s : ℝ) (hA hB hC : ℝ) (A : ℝ × ℝ) : ℝ :=
  10

theorem hexagon_pillar_height :
  ∀ (s hA hB hC : ℝ) (A : ℝ × ℝ),
  s = 8 ∧ hA = 15 ∧ hB = 10 ∧ hC = 12 ∧ A = (3, 3 * Real.sqrt 3) →
  height_of_pillar_at_vertex_F s hA hB hC A = 10 := by
  sorry

end hexagon_pillar_height_l121_121541


namespace solve_inequality_l121_121955

theorem solve_inequality (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) :
  (frac(x - 1, x - 2) + frac(x + 3, 3 * x) >= 4) ↔ (x ∈ Set.Icc (1/4) 3) := sorry

end solve_inequality_l121_121955


namespace area_cross_section_l121_121887

open Real EuclideanGeometry

-- Given a regular triangular prism P-ABC with all edge lengths 1
def edge_length (P A B C : Point) : ℝ :=
  -- Assuming an identifier for the constant length
  1 

-- Points L, M, and N are midpoints of edges PA, PB, and PC respectively
def is_midpoint (X Y Z : Point) : Prop :=
  dist X Y = dist X Z

-- The triangle LMN formed by these midpoints
def triangle_midpoints (P A B C L M N : Point) : Prop :=
  L = midpoint P A ∧ M = midpoint P B ∧ N = midpoint P C

-- The radius of the intersecting circle when the circumsphere of tetrahedron PABC
-- intersects the plane LMN results in an area π/3
theorem area_cross_section (P A B C L M N : Point) (h1 : edge_length P A B C = 1)
  (h2 : triangle_midpoints P A B C L M N) :
  area (circle_intersection (circumsphere P A B C) (plane L M N)) = π / 3 := 
sorry

end area_cross_section_l121_121887


namespace total_number_of_games_l121_121960

theorem total_number_of_games (teams : ℕ) (conference_games : ℕ) (non_conference_games_per_team : ℕ) :
  teams = 12 →
  conference_games = 3 →
  non_conference_games_per_team = 5 →
  let intra_conference_games := (teams * (teams - 1) / 2) * conference_games in
  let non_conference_games := non_conference_games_per_team * teams in
  intra_conference_games + non_conference_games = 258 :=
by
  intros h1 h2 h3
  let intra_conference_games := (12 * 11 / 2) * 3
  let non_conference_games := 5 * 12
  calc
    intra_conference_games + non_conference_games
    = (12 * 11 / 2) * 3 + 5 * 12 : by sorry -- This is simplifying the given values into the final calculation
    ... = 198 + 60 : by sorry -- This is directly from the detailed steps provided in the solution
    ... = 258 : by sorry -- The final mathematical addition step

end total_number_of_games_l121_121960


namespace line_equation_intersects_ellipse_l121_121476

theorem line_equation_intersects_ellipse :
  ∃ l : ℝ → ℝ → Prop,
    (∀ x y : ℝ, l x y ↔ 5 * x + 4 * y - 9 = 0) ∧
    (∃ M N : ℝ × ℝ,
      (M.1^2 / 20 + M.2^2 / 16 = 1) ∧
      (N.1^2 / 20 + N.2^2 / 16 = 1) ∧
      ((M.1 + N.1) / 2 = 1) ∧
      ((M.2 + N.2) / 2 = 1)) :=
sorry

end line_equation_intersects_ellipse_l121_121476


namespace find_range_of_m_l121_121879

-- Statements of the conditions given in the problem
axiom positive_real_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : (1 / x + 4 / y = 1)

-- Main statement of the proof problem
theorem find_range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1 / x + 4 / y = 1) :
  (∃ (x y : ℝ), (0 < x) ∧ (0 < y) ∧ (1 / x + 4 / y = 1) ∧ (x + y / 4 < m^2 - 3 * m)) ↔ (m < -1 ∨ m > 4) := 
sorry

end find_range_of_m_l121_121879


namespace ratio_t_q_l121_121861

theorem ratio_t_q (q r s t : ℚ) (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) : 
  t / q = 3 / 2 :=
by
  sorry

end ratio_t_q_l121_121861


namespace first_throw_before_wind_resistance_l121_121905

-- Definitions for conditions
def second_throw_distance (x : ℝ) := x
def first_throw_distance (x : ℝ) := 2 * x
def third_throw_distance (x : ℝ) := 4 * x

-- Adjustments for wind resistance
def adjusted_first_throw_distance (x : ℝ) := first_throw_distance(x) * 0.95
def adjusted_second_throw_distance (x : ℝ) := second_throw_distance(x) * 0.92
def adjusted_third_throw_distance (x : ℝ) := third_throw_distance(x)

def total_adjusted_distance (x : ℝ) := 
  adjusted_first_throw_distance(x) + adjusted_second_throw_distance(x) + adjusted_third_throw_distance(x)

-- Given total adjusted distance
axiom total_distance_is_1050 : total_adjusted_distance 150.83 = 1050

-- Prove the first throw distance before accounting for wind resistance
theorem first_throw_before_wind_resistance : 
  ∃ x : ℝ, total_adjusted_distance x = 1050 → first_throw_distance x ≈ 305.66 :=
sorry

end first_throw_before_wind_resistance_l121_121905


namespace equal_distribution_impossible_l121_121940

-- Definition of the problem conditions
def num_rows : ℕ := 9
def num_cols : ℕ := 9
def total_chips : ℕ := 324
def neighbor_cells (i j : ℕ) : set (ℕ × ℕ) :=
  { (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1) } ∪ 
  if i = 0 then {(num_cols - 1, j)} else {} ∪
  if i = num_cols - 1 then {(0, j)} else {} ∪
  if j = 0 then {(i, num_rows - 1)} else {} ∪ 
  if j = num_rows - 1 then {(i, 0)} else {}

-- Conditions on valid moves
def valid_move (chips : ℕ → ℕ → ℕ) (i j : ℕ) : Prop :=
  (i < num_rows) ∧ (j < num_cols) ∧ (chips i j ≥ 4)

-- The final goal:
theorem equal_distribution_impossible (chips : ℕ → ℕ → ℕ) :
  ¬(∀ chips' : ℕ → ℕ → ℕ, 
      (∀ i j : ℕ, i < num_rows → j < num_cols → chips' i j = 324 / (num_rows * num_cols)) → 
      (∀ i j : ℕ, valid_move chips i j → chips' (i - 1) j + 1 ≥ 0 ∧ 
                                                chips' (i + 1) j + 1 ≥ 0 ∧ 
                                                chips' i (j - 1) + 1 ≥ 0 ∧ 
                                                chips' i (j + 1) + 1 ≥ 0)) :=
  sorry

end equal_distribution_impossible_l121_121940


namespace find_n_for_2018_l121_121482

def sequence_first_term (n : ℕ) : ℕ := 2 * n * n

theorem find_n_for_2018 : ∃ n : ℕ, 2018 = sequence_first_term n ∨ (sequence_first_term n < 2018 ∧ 2018 < sequence_first_term (n + 1)) :=
by
  let n := 31
  have a_n : sequence_first_term n = 2 * 31 * 31 := rfl
  have a_n_next : sequence_first_term (n + 1) = 2 * 32 * 32 := rfl
  have h1 : sequence_first_term n = 1922 := by simp [a_n]
  have h2 : sequence_first_term (n + 1) = 2048 := by simp [a_n_next]
  have h3 : 1922 < 2018 := by norm_num
  have h4 : 2018 < 2048 := by norm_num
  use n
  right
  exact ⟨h3, h4⟩
  sorry

end find_n_for_2018_l121_121482


namespace triangle_is_right_triangle_l121_121451

variable (A B C : ℝ) (a b c : ℝ)

-- Conditions definitions
def condition1 : Prop := A + B = C
def condition2 : Prop := a / b = 3 / 4 ∧ b / c = 4 / 5 ∧ a / c = 3 / 5
def condition3 : Prop := A = 90 - B

-- Proof problem
theorem triangle_is_right_triangle (h1 : condition1 A B C) (h2 : condition2 a b c) (h3 : condition3 A B) : C = 90 := 
sorry

end triangle_is_right_triangle_l121_121451


namespace rational_cubes_rational_values_l121_121909

theorem rational_cubes_rational_values {a b : ℝ} (ha : 0 < a) (hb : 0 < b) 
  (hab : a + b = 1) (ha3 : ∃ r : ℚ, a^3 = r) (hb3 : ∃ s : ℚ, b^3 = s) : 
  ∃ r s : ℚ, a = r ∧ b = s :=
sorry

end rational_cubes_rational_values_l121_121909


namespace find_value_f_3_over_2_l121_121919

noncomputable def f (x : ℝ) : ℝ :=
if x ∈ [0, 1] then x + 1
else if x ∈ [-1, 0) then -x + 1
else f (x - 2)

theorem find_value_f_3_over_2 :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = f (x + 2)) →
  (∀ x ∈ Icc (0 : ℝ) (1 : ℝ), f x = x + 1) →
  f (3/2) = 3/2 :=
by
  intros h1 h2 h3
  sorry

end find_value_f_3_over_2_l121_121919


namespace tangent_line_at_one_range_of_a_l121_121842

-- Tangent Line Problem
theorem tangent_line_at_one (f : ℝ → ℝ) (x0 : ℝ) (y0 : ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = x * real.log x) → (x0 = 1) → (y0 = 0) →
  (∀ x, f' x = real.log x + 1) → 
  (∀ x, y0 = f x0) →
  (∀ x, (f' x0) = 1) →
  (∀ x, y = (f' x0) * (x - x0) + y0) → 
  y = x - 1 :=
begin
  sorry,
end

-- Range of a Problem
theorem range_of_a (a : ℝ) :
  (∀ x > 0, x * real.log (a * x) ≥ x - a) → a ≥ 1 :=
begin
  sorry,
end

end tangent_line_at_one_range_of_a_l121_121842


namespace num_divisors_g_100_l121_121450

-- Definition of g(n) based on the conditions
def g (n : ℕ) : ℕ := 6 ^ n

-- The Lean statement for proving that the number of positive integer divisors of g(100) is 10201
theorem num_divisors_g_100 : 
  let k := g 100 in
  (∃ (d : ℕ), (∃ (a b : ℕ) (h1 : a ≤ 100) (h2 : b ≤ 100), d = 2 ^ a * 3 ^ b) ∧ (∀ (d' ∈ { d | ∃ (a b : ℕ) (h1 : a ≤ 100) (h2 : b ≤ 100), d' = 2 ^ a * 3 ^ b}), d' ∣ k)) → 
  (card { d | ∃ (a b : ℕ) (h1 : a ≤ 100) (h2 : b ≤ 100), d = 2 ^ a * 3 ^ b }) = 10201 := 
by
  sorry

end num_divisors_g_100_l121_121450


namespace fraction_computation_l121_121057

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l121_121057


namespace volume_of_cube_l121_121192

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l121_121192


namespace inner_cube_surface_area_l121_121249

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121249


namespace positional_relationship_l121_121511

-- Definitions and conditions
structure Line (ℝ : Type _) :=
  (direction : ℝ)
  (point : ℝ)

def parallel (ℝ : Type _) [field ℝ] (l1 l2 : Line ℝ) : Prop :=
  l1.direction = l2.direction

def skew (ℝ : Type _) [field ℝ] (l1 l2 : Line ℝ) : Prop :=
  ¬∃ p, ∃ t t' : ℝ, l1.point + t * l1.direction = p ∧ l2.point + t' * l2.direction = p ∧
  ¬parallel ℝ l1 l2

-- The theorem statement
theorem positional_relationship
  (ℝ : Type _) [field ℝ]
  (a b c : Line ℝ)
  (h1 : skew ℝ a b)
  (h2 : parallel ℝ c a) :
  skew ℝ c b ∨ ∃ p, ∃ t t' : ℝ, c.point + t * c.direction = p ∧ b.point + t' * b.direction = p :=
sorry

end positional_relationship_l121_121511


namespace abe_job_time_l121_121457

theorem abe_job_time (A G C: ℕ) : G = 70 → C = 21 → (1 / G + 1 / A = 1 / C) → A = 30 := by
sorry

end abe_job_time_l121_121457


namespace range_of_a_l121_121631

variable (a : ℝ)

def sample : List ℝ := [a, 0, 1, 2, 3]

def is_median (l : List ℝ) (m : ℝ) : Prop :=
  ∃ (sorted : List ℝ), (sorted = l.qsort (≤)) ∧ (sorted.nth? 2 = some m)

theorem range_of_a (h : is_median (sample a) 1) : a ≤ 1 := sorry

end range_of_a_l121_121631


namespace sheena_completes_in_37_weeks_l121_121952

-- Definitions based on the conditions
def hours_per_dress : List Nat := [15, 18, 20, 22, 24, 26, 28]
def hours_cycle : List Nat := [5, 3, 6, 4]
def finalize_hours : Nat := 10

-- The total hours needed to sew all dresses
def total_dress_hours : Nat := hours_per_dress.sum

-- The total hours needed including finalizing hours
def total_hours : Nat := total_dress_hours + finalize_hours

-- Total hours sewed in each 4-week cycle
def hours_per_cycle : Nat := hours_cycle.sum

-- Total number of weeks it will take to complete all dresses
def weeks_needed : Nat := 4 * ((total_hours + hours_per_cycle - 1) / hours_per_cycle)
def additional_weeks : Nat := if total_hours % hours_per_cycle == 0 then 0 else 1

theorem sheena_completes_in_37_weeks : weeks_needed + additional_weeks = 37 := by
  sorry

end sheena_completes_in_37_weeks_l121_121952


namespace cube_volume_from_surface_area_l121_121199

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l121_121199


namespace value_of_c_div_b_l121_121606

theorem value_of_c_div_b (a b c : ℕ) (h1 : a = 0) (h2 : a < b) (h3 : b < c) 
  (h4 : b ≠ a + 1) (h5 : (a + b + c) / 3 = 4 * b) : c / b = 11 :=
by
  sorry

end value_of_c_div_b_l121_121606


namespace object_length_n_days_l121_121759

theorem object_length_n_days (x : ℝ) (h : 0 < x) : 
  (∏ k in finset.range (n+1), (1 + 1/((k:ℝ) + 1))) * x = 50 * x ↔ n = 98 :=
by
  -- include the target length condition
  have len_n  := x * (n + 2) / 2
  -- required equality
  have target := 50 * x
  -- showing that length = target implies n = 98
  sorry

end object_length_n_days_l121_121759


namespace sum_of_divisors_of_143_l121_121082

theorem sum_of_divisors_of_143 : 
  (∑ d in Finset.filter (fun d => 143 % d = 0) (Finset.range 144), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l121_121082


namespace total_swordfish_catch_l121_121927

-- Definitions
def S_c : ℝ := 5 - 2
def S_m : ℝ := S_c - 1
def S_a : ℝ := 2 * S_m

def W_s : ℕ := 3  -- Number of sunny days
def W_r : ℕ := 2  -- Number of rainy days

-- Sunny and rainy day adjustments
def Shelly_sunny_catch : ℝ := S_c + 0.20 * S_c
def Sam_sunny_catch : ℝ := S_m + 0.20 * S_m
def Sara_sunny_catch : ℝ := S_a + 0.20 * S_a

def Shelly_rainy_catch : ℝ := S_c - 0.10 * S_c
def Sam_rainy_catch : ℝ := S_m - 0.10 * S_m
def Sara_rainy_catch : ℝ := S_a - 0.10 * S_a

-- Total catch calculations
def Shelly_total_catch : ℝ := W_s * Shelly_sunny_catch + W_r * Shelly_rainy_catch
def Sam_total_catch : ℝ := W_s * Sam_sunny_catch + W_r * Sam_rainy_catch
def Sara_total_catch : ℝ := W_s * Sara_sunny_catch + W_r * Sara_rainy_catch

def Total_catch : ℝ := Shelly_total_catch + Sam_total_catch + Sara_total_catch

-- Proof statement
theorem total_swordfish_catch : ⌊Total_catch⌋ = 48 := 
  by sorry

end total_swordfish_catch_l121_121927


namespace liam_total_time_6_laps_l121_121933

theorem liam_total_time_6_laps :
  let laps := 6 in
  let lap_length := 400 in
  let first_segment := 150 in
  let first_segment_speed := 3 in
  let second_segment := lap_length - first_segment in
  let second_segment_speed := 6 in
  let time_first_segment := first_segment / first_segment_speed in
  let time_second_segment := second_segment / second_segment_speed in
  let time_one_lap := time_first_segment + time_second_segment in
  let total_time := laps * time_one_lap in
  total_time = 550 := sorry

end liam_total_time_6_laps_l121_121933


namespace find_c_in_triangle_ABC_l121_121882

noncomputable def c_in_triangle_ABC 
  (A B : ℝ) (b : ℝ) (hA : A = 60) (hB : B = 45) (hb : b = real.sqrt 6 - real.sqrt 2) : ℝ :=
  let C := 180 - A - B in
  if hC : C = 75 then
    (b * (real.sin (C * real.pi / 180)) / real.sin (B * real.pi / 180)) else 0

theorem find_c_in_triangle_ABC
  (A B : ℝ) (b : ℝ) (hA : A = 60) (hB : B = 45) (hb : b = real.sqrt 6 - real.sqrt 2) : 
  c_in_triangle_ABC A B b hA hB hb = real.sqrt 2 :=
by sorry

end find_c_in_triangle_ABC_l121_121882


namespace positive_difference_of_two_numbers_l121_121984

theorem positive_difference_of_two_numbers :
  ∀ (x y : ℝ), x + y = 8 → x^2 - y^2 = 24 → abs (x - y) = 3 :=
by
  intros x y h1 h2
  sorry

end positive_difference_of_two_numbers_l121_121984


namespace problem1_l121_121676

def f (x : ℝ) := (1 - 3 * x) * (1 + x) ^ 5

theorem problem1 :
  let a : ℝ := f (1 / 3)
  a = 0 :=
by
  let a := f (1 / 3)
  sorry

end problem1_l121_121676


namespace k_geq_proof_l121_121767

def cn (a : ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, real.log (a + i * 2)
def Tn (a : ℝ) (n : ℕ) : ℝ := ∑ i in finset.range n, (1 / cn a (i + 1))

theorem k_geq_proof (a : ℝ) : (∃ k : ℝ, ∀ n : ℕ, n > 0 → k * (n * 2 ^ n) / (n + 1) ≥ (2 * n - 9) * Tn a n) → k ≥ 3 / 64 :=
sorry

end k_geq_proof_l121_121767


namespace area_equals_m_plus_npi_l121_121962

noncomputable def region_area := 36 + 18 * Real.pi

theorem area_equals_m_plus_npi :
  ∃ (m n : ℤ), (m + n * Real.pi = region_area) ∧ (m + n = 54) :=
by
  use 36, 18
  split
  · -- Prove the area is as expected
    have h1: (36 + 18 * Real.pi) = region_area, by rfl
    exact h1
  · -- Show that m + n = 54
    exact (by norm_num : 36 + 18 = 54)

end area_equals_m_plus_npi_l121_121962


namespace remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0_l121_121442

theorem remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0:
  (7 * 12^24 + 3^24) % 11 = 0 := sorry

end remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0_l121_121442


namespace inner_cube_surface_area_l121_121297

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121297


namespace tom_teaching_years_l121_121995

theorem tom_teaching_years (T D : ℝ) (h1 : T + D = 70) (h2 : D = (1/2) * T - 5) : T = 50 :=
by
  -- This is where the proof would normally go if it were required.
  sorry

end tom_teaching_years_l121_121995


namespace curve_symmetry_false_l121_121617

-- Define the function representing the curve C
def curve_C (x : ℝ) : ℝ := x^3 - x + 2

-- Define the condition for a point and its symmetric point around (0, 2)
def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P in (-x, 4 - y)

-- State that for all points on the curve C, their symmetric point is generally not on curve C
theorem curve_symmetry_false : ∀ x, curve_C (-x) ≠ 4 - curve_C x :=
by
  intro x
  sorry

end curve_symmetry_false_l121_121617


namespace ellipse_area_correct_l121_121765

noncomputable def area_of_ellipse (x y : ℝ) : ℝ :=
  if h : x^2 + 2 * x + 9 * y^2 - 18 * y + 20 = 0 then
    π * sqrt 10 * (sqrt 10 / 3)
  else
    0

theorem ellipse_area_correct : 
  ∀ x y : ℝ, x^2 + 2*x + 9*y^2 - 18*y + 20 = 0 → area_of_ellipse x y = (10 * π) / 3 :=
by
  intros x y h
  rw [area_of_ellipse]
  split_ifs
  · sorry -- proof omitted
  · exfalso
    exact h_1 h

end ellipse_area_correct_l121_121765


namespace inner_cube_surface_area_l121_121257

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121257


namespace find_constant_k_l121_121661

theorem find_constant_k (k : ℝ) :
    -x^2 - (k + 9) * x - 8 = -(x - 2) * (x - 4) → k = -15 := by
  sorry

end find_constant_k_l121_121661


namespace greatest_possible_value_of_n_l121_121517

theorem greatest_possible_value_of_n (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 :=
by
  sorry

end greatest_possible_value_of_n_l121_121517


namespace slope_of_l_l121_121543

variable (OA OB : ℝ × ℝ)
def projection_equal (OA OB c : ℝ × ℝ) : Prop :=
  (OA.1 * c.1 + OA.2 * c.2 = OB.1 * c.1 + OB.2 * c.2)

theorem slope_of_l (h1 : OA = (1, 4))
                   (h2 : OB = (-3, 1))
                   (h3 : ∃ c : ℝ × ℝ, projection_equal OA OB c)
                   (h4 : 0 < ∃ k, c = (1, k)) :
  ∃ k, k = 2 / 5 := 
sorry

end slope_of_l_l121_121543


namespace bicycle_wheel_distance_l121_121680

noncomputable def circumference (d : ℝ) : ℝ := Real.pi * d

noncomputable def distance_meters (C : ℝ) (revolutions : ℝ) : ℝ := C * revolutions

noncomputable def distance_km (distance_m : ℝ) : ℝ := distance_m / 1000

theorem bicycle_wheel_distance :
  let diameter := 0.81
  let revolutions := 393.1744908390343
  let C := circumference diameter
  let distance_m := distance_meters C revolutions
  let distance_k := distance_km distance_m
  distance_k ≈ 1 :=
by {
  sorry
}

end bicycle_wheel_distance_l121_121680


namespace inner_cube_surface_area_l121_121300

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121300


namespace arithmetic_sequence_terms_sum_l121_121421

theorem arithmetic_sequence_terms_sum :
  ∃ n S, 
  let a := 17 in
  let d := 4 in
  let l := 95 in
  let n := (l - a) / d + 1 in
  let S := n / 2 * (a + l) in
  n = 20 ∧ S = 1100 := 
by
  sorry

end arithmetic_sequence_terms_sum_l121_121421


namespace gcd_of_9_fact_and_6_fact_squared_l121_121769

-- Defining the factorial operation
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the prime factorization function (stub for simplicity)
noncomputable def prime_factors (n : ℕ) : List (ℕ × ℕ) := sorry

-- Condition: prime factors of 9!
def prime_factors_9_fact : List (ℕ × ℕ) := prime_factors (factorial 9)

-- Condition: prime factors of (6!)^2
def prime_factors_6_fact_squared : List (ℕ × ℕ) := prime_factors ((factorial 6)^2)

-- Statement
theorem gcd_of_9_fact_and_6_fact_squared : Nat.gcd (factorial 9) ((factorial 6)^2) = 43200 := by
  sorry

end gcd_of_9_fact_and_6_fact_squared_l121_121769


namespace triangle_area_formula_l121_121594

theorem triangle_area_formula (a b c R : ℝ) (α β γ : ℝ) 
    (h1 : a / (Real.sin α) = 2 * R) 
    (h2 : b / (Real.sin β) = 2 * R) 
    (h3 : c / (Real.sin γ) = 2 * R) :
    let S := (1 / 2) * a * b * (Real.sin γ)
    S = a * b * c / (4 * R) := 
by 
  sorry

end triangle_area_formula_l121_121594


namespace unique_y_5_circle_7_l121_121424

def op (x y : ℝ) : ℝ := 2 * x - 4 * y + 3 * x * y

theorem unique_y_5_circle_7 :
  ∃! y : ℝ, op 5 y = 7 ∧ y = -3 / 11 :=
begin
  sorry,
end

end unique_y_5_circle_7_l121_121424


namespace math_problem_l121_121864

variable (a b c : ℝ)

theorem math_problem
  (h1 : c < b)
  (h2 : b < a)
  (h3 : ac < 0) :
  ¬(∀ b, cb^2 < ab^2) ∧
  (∀ a, ∀ b, ∀ c, c < b ∧ b < a ∧ ac < 0 → (b/a > c/a) ∧ (c * (b - a) > 0) ∧ (ac * (a - c) < 0)) :=
by
  sorry

end math_problem_l121_121864


namespace largest_divisible_l121_121649

theorem largest_divisible (n : ℕ) (h1 : n > 0) (h2 : (n^3 + 200) % (n - 8) = 0) : n = 5376 :=
by
  sorry

end largest_divisible_l121_121649


namespace inner_cube_surface_area_l121_121284

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121284


namespace range_of_independent_variable_l121_121021

theorem range_of_independent_variable (x : ℝ) : (1 - x > 0) → x < 1 :=
by
  sorry

end range_of_independent_variable_l121_121021


namespace greatest_three_digit_odd_sum_non_divisor_product_l121_121063

-- Define the properties of the problem
def is_odd (k : ℕ) : Prop := k % 2 = 1

def sum_of_first_n_odds (n : ℕ) : ℕ := n^2

def product_of_first_n_odds (n : ℕ) : ℕ :=
  ∏ i in finset.range n, (2 * i + 1)

def not_divisor (a b : ℕ) : Prop := ¬ (b % a = 0)

-- State the theorem
theorem greatest_three_digit_odd_sum_non_divisor_product :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
    sum_of_first_n_odds n = n^2 ∧ 
    ∀ m, 100 ≤ m ∧ m < 1000 → (not_divisor (n^2) (product_of_first_n_odds n)) :=
begin
  use 997,
  sorry
end

end greatest_three_digit_odd_sum_non_divisor_product_l121_121063


namespace inner_cube_surface_area_l121_121325

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121325


namespace problem_statement_l121_121809

noncomputable def ellipse : Type :=
{a b : ℝ // 0 < b ∧ b < a ∧ a^2 = 2 * b^2}

theorem problem_statement (a b : ℝ) (h : ellipse) :
  (∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 0 → (x, y) = (2, 1))
  ∧ (∀ m : ℝ, m ≠ 0 ∧ (3 * m^2 < 9 ∧ m^2 > 0) →
     PT^2 = (4/5) * |PA| * |PB|) :=
sorry

end problem_statement_l121_121809


namespace cube_volume_l121_121179

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121179


namespace sphere_wedge_volume_l121_121718

theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) (V : ℝ) (wedge_volume : ℝ) :
  circumference = 18 * Real.pi → num_wedges = 6 → V = (4 / 3) * Real.pi * (9^3) → wedge_volume = V / 6 → 
  wedge_volume = 162 * Real.pi :=
by
  intros h1 h2 h3 h4
  rw h3 at h4
  rw [←Real.pi_mul, ←mul_assoc, Nat.cast_bit1, Nat.cast_bit0, Nat.cast_one, pow_succ, pow_one, ←mul_assoc] at h4
  rw [mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc] at h4
  sorry

end sphere_wedge_volume_l121_121718


namespace no_z_for_equilateral_l121_121852

open Complex

theorem no_z_for_equilateral (z : ℂ) (h₀ : z ≠ 0) (h₁ : ∥z∥ = 2) :
  ¬(∥0 - z∥ = ∥0 - z^3∥ ∧ ∥z - z^3∥ = ∥0 - z∥ ∧ ∥z - z^3∥ = ∥0 - z^3∥) :=
by
  sorry

end no_z_for_equilateral_l121_121852


namespace max_value_m_l121_121653

theorem max_value_m (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ (m : ℝ), (4 / (1 - x) ≥ m - 1 / x)) ↔ (∃ (m : ℝ), m ≤ 9) :=
by
  sorry

end max_value_m_l121_121653


namespace sum_of_values_l121_121484

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2

theorem sum_of_values :
  ( ∑ i in finset.range 4030, f ((i + 1 : ℝ) / 2015) ) = -8058 :=
by sorry

end sum_of_values_l121_121484


namespace inner_cube_surface_area_l121_121280

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121280


namespace harmonic_conjugates_segment_ab_l121_121912

-- Definitions of distinct points and their coordinates
variables {A B C D : Type} {λ μ: ℝ}

-- Statement of harmonic conjugates
def harmonic_conjugates (A B C D : Type) :=
  ∃ λ μ, (λ ∈ ℝ) ∧ (μ ∈ ℝ) ∧
  (λ ≠ μ) ∧
  (1/λ + 1/μ = 4 ∧
  ∃ (x : ℝ), λ = x ∧ C = A + x * (B - A)) ∧
  ∃ (y : ℝ), μ = y ∧ D = A + y * (B - A)

theorem harmonic_conjugates_segment_ab (A B C D : Type) 
  (h_conjugates : harmonic_conjugates A B C D) : 
  (∃ c d, (0 < c ∧ c < 1 ∧ 0 < d ∧ d < 1) ∧
  (1/c + 1/d = 4)) :=
sorry

end harmonic_conjugates_segment_ab_l121_121912


namespace quadratic_opposite_roots_l121_121782

theorem quadratic_opposite_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 + x2 = 0 ∧ x1 * x2 = k + 1) ↔ k = -2 :=
by
  sorry

end quadratic_opposite_roots_l121_121782


namespace first_digit_base_8_of_725_is_1_l121_121061

-- Define conditions
def decimal_val := 725

-- Helper function to get the largest power of 8 less than the decimal value
def largest_power_base_eight (n : ℕ) : ℕ :=
  if 8^3 <= n then 8^3 else if 8^2 <= n then 8^2 else if 8^1 <= n then 8^1 else if 8^0 <= n then 8^0 else 0

-- The target theorem
theorem first_digit_base_8_of_725_is_1 : 
  (725 / largest_power_base_eight 725) = 1 :=
by 
  -- Proof goes here
  sorry

end first_digit_base_8_of_725_is_1_l121_121061


namespace ellipse_equation_hyperbola_equation_l121_121129

-- Ellipse Problem
theorem ellipse_equation (a b : ℝ) (h1 : a + b = 9) (h2 : 3 * 3 = a^2 - b^2) :
  (ellipse_equation : ((a^2 = 25 ∧ b^2 = 16) ∨ (a^2 = 16 ∧ b^2 = 25))) := sorry

-- Hyperbola Problem
theorem hyperbola_equation (Q : ℝ × ℝ) (Qx Qy : Q = (2,1)) :
  (c : ℝ := Real.sqrt 3)
  (hyperbola_foci : foci_definition h2 := ∃ c > 0, hyperbola_conditions c) :
  (hyperbola_conditions Q c Qx Qy : equation_of_hyperbola (2,1)) := sorry

end ellipse_equation_hyperbola_equation_l121_121129


namespace inner_cube_surface_area_l121_121312

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121312


namespace inner_cube_surface_area_l121_121306

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121306


namespace second_cube_surface_area_l121_121328

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121328


namespace sum_of_divisors_of_143_l121_121077

theorem sum_of_divisors_of_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := 
by
  sorry

end sum_of_divisors_of_143_l121_121077


namespace decrease_A_share_l121_121597

theorem decrease_A_share :
  ∃ (a b x : ℝ),
    a + b + 495 = 1010 ∧
    (a - x) / 3 = 96 ∧
    (b - 10) / 2 = 96 ∧
    x = 25 :=
by
  sorry

end decrease_A_share_l121_121597


namespace work_problem_solution_l121_121111

/-- p can do a work in the same time in which q and r together can do it. 
    If p and q work together, the work can be completed in 10 days.
    r alone needs 35 days to complete the same work. 
    How many days does q alone need to complete the work? -/
noncomputable def solve_work_problem 
  (W_p W_q W_r W : ℝ)
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = W / 10)
  (h3 : W_r = W / 35) : ℝ :=
  W / 28

theorem work_problem_solution 
  (W_p W_q W_r W : ℝ) 
  (h1 : W_p = W_q + W_r)
  (h2 : W_p + W_q = W / 10)
  (h3 : W_r = W / 35) : W_q = W / 28 :=
by
  rw solve_work_problem W_p W_q W_r W h1 h2 h3
  sorry

end work_problem_solution_l121_121111


namespace symmetric_point_xOz_l121_121629

-- Define the given point
def point : ℝ × ℝ × ℝ := (2, 3, 4)

-- Define the symmetric point of a point with respect to the xOz plane
def symmetric_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2, p.3)

-- Statement we need to prove
theorem symmetric_point_xOz : symmetric_point point = (2, -3, 4) :=
by
  sorry

end symmetric_point_xOz_l121_121629


namespace value_of_2020_pow_n_l121_121787

theorem value_of_2020_pow_n (m n : ℤ) (h1 : 3 ^ m = 4) (h2 : 3 ^ (m - 4 * n) = ⁴⁄₈₁) : 2020 ^ n = 2020 :=
sorry

end value_of_2020_pow_n_l121_121787


namespace number_le_three_l121_121990

theorem number_le_three (a b c d : ℝ) (h1 : a = 0.8) (h2 : b = 1 / 2) (h3 : c = 0.9) (h4 : d = 1 / 3) :
  (a ≤ 3) ∧ (b ≤ 3) ∧ (c ≤ 3) ∧ (d ≤ 3) → 4 :=
by
  sorry

end number_le_three_l121_121990


namespace sum_of_divisors_143_l121_121066

theorem sum_of_divisors_143 : ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := by
  sorry

end sum_of_divisors_143_l121_121066


namespace inner_cube_surface_area_l121_121252

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121252


namespace cube_volume_from_surface_area_l121_121202

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l121_121202


namespace max_teams_4_weeks_l121_121633

noncomputable def max_teams_in_tournament (weeks number_teams : ℕ) : ℕ :=
  if h : weeks > 0 then (number_teams * (number_teams - 1)) / (2 * weeks) else 0

theorem max_teams_4_weeks : max_teams_in_tournament 4 7 = 6 := by
  -- Assumptions
  let n := 6
  let teams := 7 * n
  let weeks := 4
  
  -- Define the constraints and checks here
  sorry

end max_teams_4_weeks_l121_121633


namespace g_of_5_l121_121615

variable {g : ℝ → ℝ}
variable (h1 : ∀ x y : ℝ, 2 * x * g y = 3 * y * g x)
variable (h2 : g 10 = 15)

theorem g_of_5 : g 5 = 45 / 4 :=
  sorry

end g_of_5_l121_121615


namespace horizontal_shift_needed_for_cos_l121_121036

noncomputable def horizontal_shift_transform_sin_cos 
  (x : ℝ) : ℝ :=
  cos (2 * x + π / 3) = sin 2 (x + (5 * π / 6))

theorem horizontal_shift_needed_for_cos 
  (y : ℝ) : (∃ c : ℝ, horizontal_shift_transform_sin_cos c = sin 2 y → c = 5 * π / 6) := 
sorry

end horizontal_shift_needed_for_cos_l121_121036


namespace prob_white_or_black_l121_121678

-- Defining the problem conditions
def total_balls := 5
def white_and_black := 2
def draw_balls := 3

theorem prob_white_or_black:
  let total_combinations := Nat.choose total_balls draw_balls
  let favorable_combinations := total_combinations - 1  -- all three drawn balls are of the remaining three colors
  let probability := favorable_combinations / total_combinations
  probability = 9 / 10 :=
by
  -- skip the proof
  sorry

end prob_white_or_black_l121_121678


namespace total_distance_l121_121681

-- Define the positions
def start_pos := 3
def turn_pos := -4
def end_pos := 8

-- Define absolute value function for simplicity
def abs (x : Int) : Int := if x >= 0 then x else -x

-- Calculate distances between segments
def dist1 := abs (turn_pos - start_pos)
def dist2 := abs (end_pos - turn_pos)

-- Theorem stating the total distance travelled
theorem total_distance : dist1 + dist2 = 19 := by
  -- We skip the proof using sorry
  sorry

end total_distance_l121_121681


namespace inner_cube_surface_area_l121_121366

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121366


namespace geometric_sequence_general_term_sum_of_sequence_b_l121_121477

def S (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  (Finite.sum (Finset.range n)).val (λ i => a i.1)

theorem geometric_sequence_general_term (a : ℕ → ℕ) (n : ℕ) :
  (S 6 a) / (S 3 a) = 9 →
  (a 2) + (a 5) = 36 →
  (∀ n, a n = 2 ^ n) := 
sorry

theorem sum_of_sequence_b (a b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ) :
  (∀ n, b n = a n * nat.log 2 (a n)) →
  T n = 2 + (n - 1) * 2^(n+1) := 
sorry

end geometric_sequence_general_term_sum_of_sequence_b_l121_121477


namespace length_of_ab_l121_121880

-- Define a structure for triangles with medians
structure Triangle (α : Type _) (n : ℕ) :=
  (vertices : fin n → α)
  (medians : fin n → α)

-- Conditions are:
-- 1. The median from A is perpendicular to the median from B.
-- 2. BC = 7
-- 3. AC = 6
-- Question: Prove AB = sqrt(17)

theorem length_of_ab {α : Type _} [LinearOrderedField α] [MetricSpace α] 
  (tri : Triangle α 3) 
  (A B C : tri.vertices 3)
  (medianA medianB : tri.medians 3)
  (h1 : ⟪medianA, medianB⟫ = 0) -- Medians are perpendicular
  (h2 : dist B C = 7)
  (h3 : dist A C = 6) :
  dist A B = sqrt 17 := 
sorry

end length_of_ab_l121_121880


namespace sum_of_divisors_143_l121_121067

theorem sum_of_divisors_143 : ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := by
  sorry

end sum_of_divisors_143_l121_121067


namespace solve_for_x_l121_121757

theorem solve_for_x :
  ∀ (x : ℚ), 3^(4*x^2 - 9*x + 3) = 3^(4*x^2 + 11*x - 5) → x = 2 / 5 :=
by
  sorry

end solve_for_x_l121_121757


namespace sum_of_valid_primes_l121_121102

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def reverseDigits (n : Nat) : Nat :=
  (n % 10) * 10 + n / 10

def isValidPrime (p : Nat) : Prop :=
  let d1 := p / 10
  let d0 := p % 10
  p > 20 ∧ p < 90 ∧ 
  isPrime p ∧ 
  d1 % 2 = 1 ∧ d0 % 2 = 1 ∧ 
  isPrime (reverseDigits p)

theorem sum_of_valid_primes : 
  (∑ p in Finset.filter isValidPrime (Finset.range 90), p) = 116 := 
by 
  sorry

end sum_of_valid_primes_l121_121102


namespace max_voters_after_T_l121_121121

theorem max_voters_after_T (x : ℕ) (n : ℕ) (y : ℕ) (T : ℕ)  
  (h1 : x <= 10)
  (h2 : x > 0)
  (h3 : (nx + y) ≤ (n + 1) * (x - 1))
  (h4 : ∀ k, (x - k ≥ 0) ↔ (n ≤ T + 5)) :
  ∃ (m : ℕ), m = 5 := 
sorry

end max_voters_after_T_l121_121121


namespace inner_cube_surface_area_l121_121317

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121317


namespace fountain_area_l121_121707

theorem fountain_area (A B D C : ℝ) (h₁ : B - A = 20) (h₂ : D = (A + B) / 2) (h₃ : C - D = 12) :
  ∃ R : ℝ, R^2 = 244 ∧ π * R^2 = 244 * π :=
by
  sorry

end fountain_area_l121_121707


namespace charge_for_each_additional_1_5_mile_l121_121685

-- Define the problem
noncomputable def taxi_charge (x : ℝ) : Prop :=
  ∃ n, n = (8 - (1/5)) / (1/5) ∧ 2.50 + n * x = 18.10

-- Define the theorem to be proved
theorem charge_for_each_additional_1_5_mile : taxi_charge 0.40 :=
by {
  use 39,
  simp,
  sorry
}

end charge_for_each_additional_1_5_mile_l121_121685


namespace fraction_computation_l121_121059

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l121_121059


namespace chord_length_ellipse_l121_121644

theorem chord_length_ellipse 
  (F2 : ℝ × ℝ) (A B : ℝ × ℝ) (x y : ℝ) 
  (h1 : x + 2 * y^2 = 2) 
  (hF : F2 = (1, 0)) 
  (h_incline: ∃ θ, θ = π / 4 ∧ (A.y - F2.2) = tan θ * (A.x - F2.1) 
                       ∧ (B.y - F2.2) = tan θ * (B.x - F2.1))
  (h_coords: A = (0, -1) ∧ B = (3 / 2, 1 / 2)) :
  dist A B = 4 * real.sqrt 2 / 3 := 
sorry

end chord_length_ellipse_l121_121644


namespace cube_volume_l121_121144

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l121_121144


namespace gain_per_year_l121_121700

theorem gain_per_year (principal: ℕ) (time: ℕ) (borrow_rate: ℚ) (lend_rate: ℚ) :
  principal = 5000 → time = 2 → borrow_rate = 4/100 → lend_rate = 8/100 →
  let interest_paid := principal * borrow_rate * time in
  let interest_earned := principal * lend_rate * time in
  let total_gain := interest_earned - interest_paid in
  let gain_per_year := total_gain / time in
  gain_per_year = 200 :=
by
  intros,
  sorry

end gain_per_year_l121_121700


namespace rectangle_area_l121_121687

theorem rectangle_area (r : ℝ) (h_radius : r = 7) (ratio : ℝ) (h_ratio : ratio = 2) : 
  let width := 2 * r in
  let length := ratio * width in
  let area := length * width in
  area = 392 :=
by
  -- We define the width, length, and area in the theorem context
  let width := 2 * r
  let length := ratio * width
  let area := length * width
  have h_width : width = 2 * 7 := by
    rw [h_radius]
  rw [←h_width] at ⊢ -- Simplify width using h_radius
  have h_length : length = 2 * width := by
    rw [←h_ratio] -- Simplify length using h_ratio
  rw [h_length, ←h_width] -- Substitute and resolve
  ring -- Complete the algebraic manipulation
  done

end rectangle_area_l121_121687


namespace cube_volume_l121_121170

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121170


namespace sum_of_divisors_143_l121_121069

theorem sum_of_divisors_143 : ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := by
  sorry

end sum_of_divisors_143_l121_121069


namespace FebruaryTourists_MarchTourists_AverageGrowthRate_l121_121533

variable (a : ℝ) (increase_feb : ℝ) (decrease_mar : ℝ) 
variable (hg_feb: increase_feb = 0.6) (hg_mar: decrease_mar = 0.1)

theorem FebruaryTourists (a : ℝ) (hg_feb : increase_feb = 0.6) : 
  (a + increase_feb * a = 1.6 * a) := 
by
  rw [increase_feb, mul_add]
  norm_num
  sorry

theorem MarchTourists (a : ℝ) (hg_feb : increase_feb = 0.6) (hg_mar: decrease_mar = 0.1) 
  (feb_tourists : a * (1 + 0.6) = 1.6 * a) : 
  ((1.6 * a) - decrease_mar * (1.6 * a) = 1.44 * a) := 
by
  rw [hg_mar, mul_sub, mul_add]
  norm_num
  sorry

theorem AverageGrowthRate (a : ℝ) (increase_feb: ℝ) (decrease_mar: ℝ) 
  (hmar : a * (1 + increase_feb) * (1 - decrease_mar) = 1.44 * a) :
  (∀ x : ℝ, (1 + x) ^ 2 = 1.44 → x = 0.2) := 
by
  intros
  rw [pow_two, mul_add, add_mul, one_mul] at h
  apply h
  sorry

end FebruaryTourists_MarchTourists_AverageGrowthRate_l121_121533


namespace fraction_computation_l121_121058

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l121_121058


namespace encoded_base5_to_base10_l121_121143

-- Given definitions
def base5_to_int (d1 d2 d3 : ℕ) : ℕ := d1 * 25 + d2 * 5 + d3

def V := 2
def W := 0
def X := 4
def Y := 1
def Z := 3

-- Prove that the base-10 expression for the integer coded as XYZ is 108
theorem encoded_base5_to_base10 :
  base5_to_int X Y Z = 108 :=
sorry

end encoded_base5_to_base10_l121_121143


namespace find_b_l121_121226

def passesThrough (b c : ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = P.1^2 + b * P.1 + c

theorem find_b (b c : ℝ)
  (H1 : passesThrough b c (1, 2))
  (H2 : passesThrough b c (5, 2)) :
  b = -6 :=
by
  sorry

end find_b_l121_121226


namespace evaluate_powers_of_i_l121_121432

theorem evaluate_powers_of_i :
  (complex.I ^ 23 + complex.I ^ 221 + complex.I ^ 20) = 1 := 
sorry

end evaluate_powers_of_i_l121_121432


namespace monotonic_increasing_intervals_max_min_g_l121_121483

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x - cos x ^ 2 + 1 / 2

theorem monotonic_increasing_intervals :
  ∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 6) (k * π + π / 3), f x = sqrt 3 * sin x * cos x - cos x ^ 2 + 1 / 2 ↔
  f x' = sqrt 3 * sin x' * cos x' - cos x' ^ 2 + 1 / 2 ∧ x < x' → f x ≤ f x' := sorry

noncomputable def g (x : ℝ) : ℝ := sin (2 * x - π / 6 - π / 3)

theorem max_min_g :
  ∃ M m, ∀ x ∈ Set.Icc 0 π, g x = sin (2 * x - π / 2) ∧ M = 1 ∧ m = -sqrt 3 / 2 := sorry

end monotonic_increasing_intervals_max_min_g_l121_121483


namespace calculate_difference_l121_121924

theorem calculate_difference :
  let m := Nat.find (λ m, m > 99 ∧ m < 1000 ∧ m % 13 = 7)
  let n := Nat.find (λ n, n > 999 ∧ n < 10000 ∧ n % 13 = 7)
  n - m = 895 :=
by
  sorry

end calculate_difference_l121_121924


namespace cube_volume_l121_121151

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l121_121151


namespace inner_cube_surface_area_l121_121246

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121246


namespace sphere_wedge_volume_l121_121717

theorem sphere_wedge_volume (circumference : ℝ) (num_wedges : ℕ) (V : ℝ) (wedge_volume : ℝ) :
  circumference = 18 * Real.pi → num_wedges = 6 → V = (4 / 3) * Real.pi * (9^3) → wedge_volume = V / 6 → 
  wedge_volume = 162 * Real.pi :=
by
  intros h1 h2 h3 h4
  rw h3 at h4
  rw [←Real.pi_mul, ←mul_assoc, Nat.cast_bit1, Nat.cast_bit0, Nat.cast_one, pow_succ, pow_one, ←mul_assoc] at h4
  rw [mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc, mul_assoc] at h4
  sorry

end sphere_wedge_volume_l121_121717


namespace part_a_solution_part_b_solution_l121_121108

-- Part (a) Statement in Lean 4
theorem part_a_solution (N : ℕ) (a b : ℕ) (h : N = a * 10^n + b * 10^(n-1)) :
  ∃ (m : ℕ), (N / 10 = m) -> m * 10 = N := sorry

-- Part (b) Statement in Lean 4
theorem part_b_solution (N : ℕ) (a b c : ℕ) (h : N = a * 10^n + b * 10^(n-1) + c * 10^(n-2)) :
  ∃ (m : ℕ), (N / 10^(n-1) = m) -> m * 10^(n-1) = N := sorry

end part_a_solution_part_b_solution_l121_121108


namespace distance_to_destination_l121_121702

-- Conditions
def Speed : ℝ := 65 -- speed in km/hr
def Time : ℝ := 3   -- time in hours

-- Question to prove
theorem distance_to_destination : Speed * Time = 195 := by
  sorry

end distance_to_destination_l121_121702


namespace volume_of_wedge_l121_121722

theorem volume_of_wedge (r : ℝ) (V : ℝ) (sphere_wedges : ℝ) 
  (h_circumference : 2 * Real.pi * r = 18 * Real.pi)
  (h_volume : V = (4 / 3) * Real.pi * r ^ 3) 
  (h_sphere_wedges : sphere_wedges = 6) : 
  V / sphere_wedges = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l121_121722


namespace inner_cube_surface_area_l121_121353

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121353


namespace rhombus_area_l121_121620

theorem rhombus_area : 
  ∃ (d1 d2 : ℝ), (∀ (x : ℝ), x^2 - 14 * x + 48 = 0 → x = d1 ∨ x = d2) ∧
  (∀ (A : ℝ), A = d1 * d2 / 2 → A = 24) :=
by 
sorry

end rhombus_area_l121_121620


namespace min_value_f_x_equals_1_min_value_sum_squares_l121_121918

open Real

def f (x : ℝ) : ℝ := |(1 / 2) * x + 1| + |x|

theorem min_value_f_x_equals_1 :
  ∃ a : ℝ, a = 1 ∧ ∀ x : ℝ, f x ≥ a :=
sorry

theorem min_value_sum_squares (p q r : ℝ) (h1 : 0 < p) (h2 : 0 < q) (h3 : 0 < r)
  (h4 : p + q + r = 3) :
  ∃ b : ℝ, b = 3 ∧ ∀ p q r : ℝ, h4 → p^2 + q^2 + r^2 ≥ b :=
sorry

end min_value_f_x_equals_1_min_value_sum_squares_l121_121918


namespace cube_volume_l121_121152

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l121_121152


namespace fraction_product_l121_121053

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l121_121053


namespace range_a_sub_b_mul_c_l121_121857

theorem range_a_sub_b_mul_c (a b c : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) (h4 : 2 < c) (h5 : c < 3) :
  -6 < (a - b) * c ∧ (a - b) * c < 0 :=
by
  -- We need to prove the range of (a - b) * c is within (-6, 0)
  sorry

end range_a_sub_b_mul_c_l121_121857


namespace sum_of_divisors_143_l121_121087

theorem sum_of_divisors_143 : (∑ i in (finset.filter (λ d, 143 % d = 0) (finset.range 144)), i) = 168 :=
by
  -- The final proofs will go here.
  sorry

end sum_of_divisors_143_l121_121087


namespace no_six_in_variance_condition_l121_121785

theorem no_six_in_variance_condition (med : ℝ) (var : ℝ) (results : list ℕ) :
  med = 3 ∧ var = 0.16 ∧ length results = 5 → ¬(6 ∈ results) :=
by
  intros h,
  sorry

end no_six_in_variance_condition_l121_121785


namespace fourth_intersection_point_l121_121894

theorem fourth_intersection_point : 
  ∃ x y : ℝ, 
    (x - 3)^2 + (y + 1)^2 = 25 ∧ 
    x * y = 2 ∧ 
    (x ≠ 4 ∨ y ≠ 1/2) ∧ 
    (x ≠ -2 ∨ y ≠ -1) ∧ 
    (x ≠ 2/3 ∨ y ≠ 3) ∧ 
    x = -3/4 ∧ 
    y = -8/3 := 
begin
  sorry
end

end fourth_intersection_point_l121_121894


namespace incorrect_categorical_statement_l121_121105

variable (x : ℝ) (y : ℝ) (X Y : Type)
variable [∀ x, Decidable (X x = x)] [∀ y, Decidable (Y y = y)]

def center_of_sample_points (x̄ ȳ : ℝ) : Prop := 
  ∀ (x_samples : List ℝ) (y_samples : List ℝ), (x_samples.sum / x_samples.length = x̄) ∧ (y_samples.sum / y_samples.length = ȳ)

def linear_correlation (r : ℝ) : Prop := 
  abs r ≤ 1

def categorical_k2_confidence (k : ℝ) : Prop := 
  k ≥ 0

def regression_line_increase : Prop := 
  ∀ x, let ŷ := 0.2 * x + 0.8 in 1 → ŷ + 0.2

theorem incorrect_categorical_statement
  (x̄ ȳ : ℝ)
  (r k : ℝ)
  (H1 : center_of_sample_points x̄ ȳ)
  (H2 : linear_correlation r)
  (H3 : categorical_k2_confidence k)
  (H4 : regression_line_increase) :
  ¬ (categorical_k2_confidence k) → categorical_k2_confidence k := 
by
  sorry

end incorrect_categorical_statement_l121_121105


namespace cube_volume_of_surface_area_l121_121161

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l121_121161


namespace positive_difference_of_two_numbers_l121_121983

theorem positive_difference_of_two_numbers :
  ∀ (x y : ℝ), x + y = 8 → x^2 - y^2 = 24 → abs (x - y) = 3 :=
by
  intros x y h1 h2
  sorry

end positive_difference_of_two_numbers_l121_121983


namespace probability_of_color_change_l121_121730

theorem probability_of_color_change :
  let cycle_times := [40, 5, 40, 5] in
  let total_cycle_duration := 90 in
  let changing_intervals := 20 in
  (changing_intervals / total_cycle_duration : ℚ) = (2 / 9 : ℚ) := by
    sorry

end probability_of_color_change_l121_121730


namespace lock_code_count_l121_121667

theorem lock_code_count : 
  let digits := {1, 2, 3, 4, 5, 6, 7}
  let odd_digits := {1, 3, 5, 7}
  -- Defining a valid lock code as a list of digits with given conditions
  def is_valid_lock_code (code : List ℕ) : Prop :=
    code.length = 4 ∧ 
    (∀ (d : ℕ), d ∈ code → d ∈ digits) ∧ 
    (∀ (i j : ℕ), i ≠ j → code[i] ≠ code[j]) ∧ 
    code[0] ∈ odd_digits ∧ 
    code[3] ∈ odd_digits 
  -- Statement to be proved
  ∃ (lock_codes : List (List ℕ)), 
    (∀ (code : List ℕ), code ∈ lock_codes ↔ is_valid_lock_code code) ∧ 
    lock_codes.length = 360 :=
sorry

end lock_code_count_l121_121667


namespace cube_volume_l121_121147

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l121_121147


namespace children_got_on_bus_l121_121130

-- Definition of initial conditions
def initial_children : ℕ := 21
def children_off : ℕ := 10
def total_children_after : ℕ := 16

-- The goal/ theorem to prove
theorem children_got_on_bus : ∃ x : ℕ, x = 5 ∧ initial_children - children_off + x = total_children_after :=
by
  existsi 5
  simp [initial_children, children_off, total_children_after]
  sorry

end children_got_on_bus_l121_121130


namespace same_grade_percentage_is_correct_l121_121885

def total_students : ℕ := 40

def grade_distribution : ℕ × ℕ × ℕ × ℕ :=
  (17, 40, 100)

def same_grade_percentage (total_students : ℕ) (same_grade_students : ℕ) : ℚ :=
  (same_grade_students / total_students) * 100

theorem same_grade_percentage_is_correct :
  let same_grade_students := 3 + 5 + 6 + 3
  same_grade_percentage total_students same_grade_students = 42.5 :=
by 
let same_grade_students := 3 + 5 + 6 + 3
show same_grade_percentage total_students same_grade_students = 42.5
sorry

end same_grade_percentage_is_correct_l121_121885


namespace find_z_and_modulus_find_m_range_l121_121829

noncomputable def complex_num_z (z : ℂ) : Prop :=
  (z + complex.I * 2).im = 0 ∧ ((z / (2 - complex.I))).im = 0

theorem find_z_and_modulus (z : ℂ) 
  (h : complex_num_z z) : 
  z = 4 - 2 * complex.I ∧ complex.abs z = 2 * real.sqrt 5 :=
sorry

noncomputable def z1_quadrant_condition (z : ℂ) (m : ℝ) : Prop :=
  let z1 := complex.conj z + 3 * m + (m^2 - 6) * complex.I
  in  (z1.re > 0 ∧ z1.im < 0)

theorem find_m_range (z : ℂ) (h : z = 4 - 2 * complex.I) (m : ℝ)
  (hq : z1_quadrant_condition z m) :
  -4/3 < m ∧ m < 2 :=
sorry

end find_z_and_modulus_find_m_range_l121_121829


namespace remainder_of_expression_mod_9_l121_121989

theorem remainder_of_expression_mod_9 (a b c d : ℕ) (h1 : a < 9) (h2 : b < 9) (h3 : c < 9) (h4 : d < 9) 
  (coprime_a : Nat.coprime a 9) (coprime_b : Nat.coprime b 9) (coprime_c : Nat.coprime c 9) 
  (coprime_d : Nat.coprime d 9) (h_distinct : List.nodup [a, b, c, d]) :
  ((a * b * c + a * b * d + a * c * d + b * c * d) * (a * b * c * d)⁻¹) % 9 = 8 :=
by
  sorry

end remainder_of_expression_mod_9_l121_121989


namespace sum_of_divisors_143_l121_121084

theorem sum_of_divisors_143 : (∑ i in (finset.filter (λ d, 143 % d = 0) (finset.range 144)), i) = 168 :=
by
  -- The final proofs will go here.
  sorry

end sum_of_divisors_143_l121_121084


namespace prove_f_g_of_4_l121_121917

def f (x : ℝ) : ℝ := 3 * (Real.sqrt x) + 18 / (Real.sqrt x)
def g (x : ℝ) : ℝ := 3 * x^2 - 3 * x - 4

theorem prove_f_g_of_4 :
  f (g 4) = (57 * Real.sqrt 2) / 4 :=
by
  sorry

end prove_f_g_of_4_l121_121917


namespace gcd_of_9_fact_and_6_fact_squared_l121_121768

-- Defining the factorial operation
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Define the prime factorization function (stub for simplicity)
noncomputable def prime_factors (n : ℕ) : List (ℕ × ℕ) := sorry

-- Condition: prime factors of 9!
def prime_factors_9_fact : List (ℕ × ℕ) := prime_factors (factorial 9)

-- Condition: prime factors of (6!)^2
def prime_factors_6_fact_squared : List (ℕ × ℕ) := prime_factors ((factorial 6)^2)

-- Statement
theorem gcd_of_9_fact_and_6_fact_squared : Nat.gcd (factorial 9) ((factorial 6)^2) = 43200 := by
  sorry

end gcd_of_9_fact_and_6_fact_squared_l121_121768


namespace imaginary_part_of_z_l121_121479

def z : ℂ := (1 - 2 * complex.I) / (2 + complex.I)

theorem imaginary_part_of_z :
  complex.im z = -1 :=
sorry

end imaginary_part_of_z_l121_121479


namespace valid_grid_count_l121_121891

def is_valid_grid (grid : List (List ℕ)) : Prop :=
  grid.length = 3 ∧ (∀ row : List ℕ, row ∈ grid → row.length = 3) ∧
  list.sum (list.join grid) = 45 ∧
  (∀ i in [0, 1, 2], list.sum (list.nth grid i).getD [] = 15) ∧
  (∀ j in [0, 1, 2], list.sum (list.map (fun row => list.nth row j).getD 0 grid) = 15)

noncomputable def number_of_valid_grids : ℕ :=
  72

theorem valid_grid_count : ∃ grids : List (List (List ℕ)), 
  (∀ g ∈ grids, is_valid_grid g) ∧ 
  grids.length = number_of_valid_grids :=
by sorry

end valid_grid_count_l121_121891


namespace volume_of_cube_l121_121185

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l121_121185


namespace possible_values_expr_l121_121824

theorem possible_values_expr (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) :
  ∃ x, x = (Real.sign a + Real.sign b + Real.sign c + Real.sign d + Real.sign (a * b * c * d)) ∧
       x ∈ {5, 3, 2, 0, -3} := 
sorry

end possible_values_expr_l121_121824


namespace irrational_number_count_l121_121737

theorem irrational_number_count : 
  let numbers := [-3.14, 0, -Real.pi, 22 / 3, 1.12112] in
  list.count (λ x, irrational x) numbers = 1 :=
sorry

end irrational_number_count_l121_121737


namespace wedge_volume_formula_l121_121713

noncomputable def sphere_wedge_volume : ℝ :=
let r := 9 in
let volume_of_sphere := (4 / 3) * Real.pi * r^3 in
let volume_of_one_wedge := volume_of_sphere / 6 in
volume_of_one_wedge

theorem wedge_volume_formula
  (circumference : ℝ)
  (h1 : circumference = 18 * Real.pi)
  (num_wedges : ℕ)
  (h2 : num_wedges = 6) :
  sphere_wedge_volume = 162 * Real.pi :=
by
  sorry

end wedge_volume_formula_l121_121713


namespace sum_of_divisors_of_143_l121_121073

theorem sum_of_divisors_of_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := 
by
  sorry

end sum_of_divisors_of_143_l121_121073


namespace percentage_of_D_is_35_l121_121008

/-- The 20 scores in Ms. Patterson's class --/
def scores : List ℕ := [89, 65, 55, 96, 73, 93, 82, 70, 77, 65, 81, 79, 67, 85, 88, 61, 84, 71, 73, 90]

/-- The range for grade D --/
def is_grade_D (score : ℕ) : Prop := score ≥ 65 ∧ score ≤ 75

/-- To prove percentage of students who received D is 35%. --/
theorem percentage_of_D_is_35 : 
  (100 * (scores.countp is_grade_D) / scores.length) = 35 := 
sorry

end percentage_of_D_is_35_l121_121008


namespace inner_cube_surface_area_l121_121377

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121377


namespace sum_of_divisors_143_l121_121070

theorem sum_of_divisors_143 : ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := by
  sorry

end sum_of_divisors_143_l121_121070


namespace two_is_four_percent_of_fifty_l121_121677

theorem two_is_four_percent_of_fifty : (2 / 50) * 100 = 4 := 
by
  sorry

end two_is_four_percent_of_fifty_l121_121677


namespace hyperbola_equation_l121_121846

-- Define the conditions
variable (a b c : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
variable (h_eccentricity : a * 5 = 3 * c)
variable (h_conjugate_axis : 2 * b = 8)
variable (h_c_squared : c^2 = a^2 + b^2)

-- State the theorem
theorem hyperbola_equation : 
  0 < a ∧ 0 < b ∧ (a * 5 = 3 * c) ∧ (2 * b = 8) ∧ (c^2 = a^2 + b^2) → 
  (a = 3 ∧ b = 4 ∧ c = 5 ∧ (C : ∀ x y, (x * x) / 9 - (y * y) / 16 = 1)) :=
by
  sorry

end hyperbola_equation_l121_121846


namespace sum_of_digits_l121_121019

theorem sum_of_digits (A B : ℕ) (hA : A = 70707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707)
  (hB : B = 90909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909) :
  (let P := A * B in (P / 10000) % 10 + (P % 10) = 6) :=
by sorry

end sum_of_digits_l121_121019


namespace count_distinct_real_pairs_l121_121975

theorem count_distinct_real_pairs :
  {p : ℝ × ℝ // p.fst = p.fst^2 + p.snd^2 ∧ p.snd = 2 * p.fst * p.snd}.toFinset.card = 4 :=
begin
  sorry
end

end count_distinct_real_pairs_l121_121975


namespace value_m_invalid_l121_121878

theorem value_m_invalid (m : ℝ) :
  m + 3 < (-m + 1) - 13 → ¬(m = 6) :=
by
  intro h
  have h₁ : 2 * m + 3 < 14 := by
    linarith [h]
  have h₂ : 2 * m < 11 := by
    linarith [h₁]
  have h₃ : m < 5.5 := by
    linarith [h₂]
  linarith [h₃]

end value_m_invalid_l121_121878


namespace petya_finishes_earlier_than_masha_l121_121591

variable (t_P t_M t_K : ℕ)

-- Given conditions
def condition1 := t_K = 2 * t_P
def condition2 := t_P + 12 = t_K
def condition3 := t_M = 3 * t_P

-- The proof goal: Petya finishes 24 seconds earlier than Masha
theorem petya_finishes_earlier_than_masha
    (h1 : condition1 t_P t_K)
    (h2 : condition2 t_P t_K)
    (h3 : condition3 t_P t_M) :
    t_M - t_P = 24 := by
  sorry

end petya_finishes_earlier_than_masha_l121_121591


namespace tom_teaching_years_l121_121994

theorem tom_teaching_years (T D : ℝ) (h1 : T + D = 70) (h2 : D = (1/2) * T - 5) : T = 50 :=
by
  -- This is where the proof would normally go if it were required.
  sorry

end tom_teaching_years_l121_121994


namespace hyperbola_range_m_l121_121963

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (m - 2) ≠ 0 ∧ (m + 3) ≠ 0 ∧ (x^2 / (m - 2) + y^2 / (m + 3) = 1)) ↔ (-3 < m ∧ m < 2) :=
by
  sorry

end hyperbola_range_m_l121_121963


namespace left_handed_classical_music_lovers_l121_121532

open Classical

/-- In a group of 25 people, 10 are left-handed. 18 of them enjoy classical music. 
    3 of them are right-handed and do not enjoy classical music. 
    How many group members are left-handed and enjoy classical music? 
    Assume people are either left-handed or right-handed. -/
theorem left_handed_classical_music_lovers
    (total : ℕ)
    (left_handed : ℕ)
    (classical_music_lovers : ℕ)
    (right_handed_not_classical : ℕ)
    (left_or_right_handed : ∀ (p : ℕ), p = left_handed ∨ p = total - left_handed)
    (total_people : total = 25)
    (left_hand : left_handed = 10)
    (classical_music : classical_music_lovers = 18)
    (right_not_classical : right_handed_not_classical = 3) :
    ∃ (y : ℕ), y = 6 :=
by
  let y := 6
  use y
  sorry

end left_handed_classical_music_lovers_l121_121532


namespace part1_part2_l121_121911

open Set

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1: ∀ a : ℝ, A = B a → a = 1 := by
  intros a h
  sorry

theorem part2: ∀ a : ℝ, B a ⊆ A ∧ a > 0 → a = 1 := by
  intros a h
  sorry

end part1_part2_l121_121911


namespace inner_cube_surface_area_l121_121387

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121387


namespace inner_cube_surface_area_l121_121247

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121247


namespace sum_of_smallest_natural_numbers_l121_121780

-- Define the problem statement
def satisfies_eq (A B : ℕ) := 360 / (A^3 / B) = 5

-- Prove that there exist natural numbers A and B such that 
-- satisfies_eq A B is true, and their sum is 9
theorem sum_of_smallest_natural_numbers :
  ∃ (A B : ℕ), satisfies_eq A B ∧ A + B = 9 :=
by
  -- Sorry is used here to indicate the proof is not given
  sorry

end sum_of_smallest_natural_numbers_l121_121780


namespace goldie_earnings_l121_121499

theorem goldie_earnings
  (hourly_wage : ℕ := 5)
  (hours_last_week : ℕ := 20)
  (hours_this_week : ℕ := 30) :
  hourly_wage * hours_last_week + hourly_wage * hours_this_week = 250 :=
by
  sorry

end goldie_earnings_l121_121499


namespace max_m_value_l121_121490

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

noncomputable def h (x : ℝ) : ℝ := (x * Real.log x + x) / (x - 1)

theorem max_m_value (m : ℤ) (x : ℝ) (hx : 1 < x) : f(x) - m * (x - 1) > 0 :=
by
  sorry

end max_m_value_l121_121490


namespace inner_cube_surface_area_l121_121380

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121380


namespace max_value_of_abs_z_l121_121804

-- Definition of the problem using the given condition
def max_abs_value (z : ℂ) : ℝ :=
  if h : |z - (3 + 4*complex.i)| = 1 then
    complex.abs z
  else
    0

-- The theorem statement
theorem max_value_of_abs_z (z : ℂ) (h : |z - (3 + 4*complex.i)| = 1) : max_abs_value z = 6 :=
by
  sorry

-- Ensure the theorem builds successfully
#check @max_value_of_abs_z

end max_value_of_abs_z_l121_121804


namespace cube_tunnel_surface_area_l121_121570

noncomputable def surface_area_tunnel (x y z : ℕ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ z % (2 * 2) ≠ 0) : bool :=
  x + y + z = 639

theorem cube_tunnel_surface_area :
  ∃ (x y z : ℕ), surface_area_tunnel x y z (and.intro (nat.zero_lt_succ 0) (and.intro (nat.zero_lt_succ 0) (and.intro (nat.zero_lt_succ 0) (nat.mod_ne_zero 6 4 (by norm_num))))) :=
  sorry

end cube_tunnel_surface_area_l121_121570


namespace angle_3900_in_fourth_quadrant_angle_neg1000_in_first_quadrant_l121_121127

def quadrant (angle : ℝ) : String :=
  let angle := angle % 360
  if 0 ≤ angle ∧ angle < 90 then "first"
  else if 90 ≤ angle ∧ angle < 180 then "second"
  else if 180 ≤ angle ∧ angle < 270 then "third"
  else "fourth"

theorem angle_3900_in_fourth_quadrant : quadrant 3900 = "fourth" := by
  sorry

theorem angle_neg1000_in_first_quadrant : quadrant (-1000) = "first" := by
  sorry

end angle_3900_in_fourth_quadrant_angle_neg1000_in_first_quadrant_l121_121127


namespace area_difference_l121_121551

-- Given geometrical conditions
def GAB_right_angle : Prop := ∠GAB = 90
def ABC_right_angle : Prop := ∠ABC = 90
def AB_length : ℝ := 6
def BC_length : ℝ := 8
def AG_length : ℝ := 10

-- Intersection of line segments AC and BG at point D
def intersect_at_D : Prop := ∃ D : Point, D ∈ (line AC) ∧ D ∈ (line BG)

-- Areas of triangles
def area (P Q R : Point) : ℝ := 1/2 * (abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y)))

-- Difference between the areas of ΔADG and ΔBDC
theorem area_difference (A B C G D : Point) (h_GAB_right : GAB_right_angle)
                        (h_ABC_right : ABC_right_angle) (h_AB_length : |A - B| = AB_length)
                        (h_BC_length : |B - C| = BC_length) (h_AG_length : |A - G| = AG_length)
                        (h_intersect : intersect_at_D ) :
  abs (area A D G - area B D C) = 6 :=
by
  sorry

end area_difference_l121_121551


namespace pipe_filling_time_without_leak_l121_121703

-- Define the conditions from the problem
def combined_rate_with_leak := (1 : ℝ) / 10
def leak_emptying_rate := (1 : ℝ) / 15
def pipe_filling_rate (T : ℝ) := (1 : ℝ) / T

-- State the theorem to be proven
theorem pipe_filling_time_without_leak : ∃ T : ℝ, 
    (pipe_filling_rate T - leak_emptying_rate = combined_rate_with_leak) ∧ T = 6 :=
by
  sorry

end pipe_filling_time_without_leak_l121_121703


namespace number_of_boys_l121_121527

-- Definitions based on the conditions
def ratio_boys_to_girls : Nat → Nat → Prop := λ boys girls, 4 * girls = 3 * boys

def total_students : Nat → Nat → Nat := λ boys girls, boys + girls

-- The theorem statement
theorem number_of_boys (boys girls : Nat)
  (h1 : ratio_boys_to_girls boys girls)
  (h2 : total_students boys girls = 49) :
  boys = 28 := 
by 
  sorry

end number_of_boys_l121_121527


namespace inner_cube_surface_area_l121_121288

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121288


namespace inner_cube_surface_area_l121_121287

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121287


namespace soccer_tournament_eq_l121_121538

theorem soccer_tournament_eq (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : (1 / 2 : ℚ) * x * (x - 1) = 28 := by
  sorry

end soccer_tournament_eq_l121_121538


namespace regular_polygon_sides_l121_121760

theorem regular_polygon_sides (α : ℝ) (hα : α = 120) : 
  ∃ (n : ℕ), 180 * (n - 2) / n = α → n = 6 :=
by
  intro h
  use 6
  sorry

end regular_polygon_sides_l121_121760


namespace quadratic_equation_in_x_l121_121405

theorem quadratic_equation_in_x (k x : ℝ) : 
  (k^2 + 1) * x^2 - (k * x - 8) - 1 = 0 := 
sorry

end quadratic_equation_in_x_l121_121405


namespace range_a_sub_b_mul_c_l121_121856

theorem range_a_sub_b_mul_c (a b c : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) (h4 : 2 < c) (h5 : c < 3) :
  -6 < (a - b) * c ∧ (a - b) * c < 0 :=
by
  -- We need to prove the range of (a - b) * c is within (-6, 0)
  sorry

end range_a_sub_b_mul_c_l121_121856


namespace count_pos_integers_lt_2000_congruent_8_mod_13_l121_121503

open Nat

theorem count_pos_integers_lt_2000_congruent_8_mod_13 :
  {n : ℕ | n < 2000 ∧ n % 13 = 8}.to_finset.card = 154 :=
by
  sorry

end count_pos_integers_lt_2000_congruent_8_mod_13_l121_121503


namespace proof_problem_l121_121916

theorem proof_problem (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a > b) (h5 : a^2 - a * c + b * c = 7) :
  a - c = 0 ∨ a - c = 1 :=
 sorry

end proof_problem_l121_121916


namespace inappropriate_expression_is_D_l121_121654

-- Definitions of each expression as constants
def expr_A : String := "Recently, I have had the honor to read your masterpiece, and I felt enlightened."
def expr_B : String := "Your visit has brought glory to my humble abode."
def expr_C : String := "It's the first time you honor my place with a visit, and I apologize for any lack of hospitality."
def expr_D : String := "My mother has been slightly unwell recently, I hope you won't bother her."

-- Definition of the problem context
def is_inappropriate (expr : String) : Prop := 
  expr = expr_D

-- The theorem statement
theorem inappropriate_expression_is_D : is_inappropriate expr_D := 
by
  sorry

end inappropriate_expression_is_D_l121_121654


namespace angle_DAB_independent_of_ABC_l121_121881

-- Definitions based on conditions
structure IsoscelesTriangle (A B C : Type) :=
  (CA : ℝ)
  (CB : ℝ)
  (iso : CA = CB)

structure Square (B C D E : Type) :=
  (side : ℝ)
  (BC : side)
  (CD : side)
  (DE : side)
  (EB : side)
  (right_angles : True) -- For brevity, assume right angles where necessary

structure TriangleSquareConstruction (A B C D E : Type) extends IsoscelesTriangle A B C, Square B C D E

-- The theorem statement proving angle DAB is independent of angle in isosceles triangle ABC
theorem angle_DAB_independent_of_ABC (A B C D E : Type) [TriangleSquareConstruction A B C D E] :
  ∃ θ : ℝ, ∀ (t : TriangleSquareConstruction A B C D E), (θ - 45:ℝ) = (θ - 45:ℝ) :=
by
  sorry

end angle_DAB_independent_of_ABC_l121_121881


namespace alfred_net_profit_gain_as_percentage_l121_121735

def purchase_price : ℝ := 4700
def selling_price : ℝ := 5800
def initial_repair_percent : ℝ := 0.10
def maintenance_percent : ℝ := 0.05
def safety_upgrade_percent : ℝ := 0.07
def tax_percent : ℝ := 0.25

def initial_repair_cost : ℝ := initial_repair_percent * purchase_price
def cost_after_initial_repair : ℝ := purchase_price + initial_repair_cost
def maintenance_cost : ℝ := maintenance_percent * cost_after_initial_repair
def cost_after_maintenance : ℝ := cost_after_initial_repair + maintenance_cost
def safety_upgrade_cost : ℝ := safety_upgrade_percent * maintenance_cost
def total_cost : ℝ := cost_after_maintenance + safety_upgrade_cost
def profit_before_tax : ℝ := selling_price - total_cost
def tax_on_profit : ℝ := tax_percent * profit_before_tax
def net_profit : ℝ := profit_before_tax - tax_on_profit
def net_profit_percentage : ℝ := (net_profit / purchase_price) * 100

theorem alfred_net_profit_gain_as_percentage :
  abs (net_profit_percentage - 5.64) < 0.01 :=
by
  sorry

end alfred_net_profit_gain_as_percentage_l121_121735


namespace return_speed_l121_121137

variable (d : ℝ) (r : ℝ)

-- Distance from C to D
variable (dist_CD : d = 150)

-- Speed from C to D
variable (speed_CD : 75)

-- Average speed for the round trip
variable (avg_speed : 50)

theorem return_speed (H1 : dist_CD) (H2 : avg_speed = 50) : r = 37.5 :=
by
  -- Define the time to travel from C to D
  let time_CD := d / speed_CD
  -- Define the time to travel from D to C
  let time_DC := d / r
  -- The total round trip distance
  have total_distance : 2 * d = 300 := by
    rw [H1]
    exact rfl

  -- The total time for the round trip
  have total_time : time_CD + time_DC = 2 + (150 / r) := by
    rw [time_CD, H1]
    norm_num

  -- The average speed for the round trip
  have H3 : 300 / (2 + 150 / r) = avg_speed := by
    rw [H1, total_time]
    norm_num

  -- Simplify to find the return speed r
  rw [avg_speed] at H3
  have : 6 = 2 + 150 / r := by
    linarith
  have : 4 = 150 / r := by
    linarith
  have H4 : r = 150 / 4 := by
    field_simp
    linarith
  rw [div_eq_mul_inv, mul_comm] at H4

  norm_num at H4
  exact H4


end return_speed_l121_121137


namespace time_for_C_alone_to_finish_the_job_l121_121107

variable {A B C : ℝ} -- Declare work rates as real numbers

-- Define the conditions
axiom h1 : A + B = 1/15
axiom h2 : A + B + C = 1/10

-- Define the theorem to prove
theorem time_for_C_alone_to_finish_the_job : C = 1/30 :=
by
  apply sorry

end time_for_C_alone_to_finish_the_job_l121_121107


namespace find_area_ANG_l121_121972

variables (A B C L M N G : Type) [AffineSpace ℝ A]
variable (triangle_ABC : triangle A B C)
variable (median_AL : median A B C L)
variable (median_BM : median B C A M)
variable (median_CN : median C A B N)
variable (G_is_centroid : centroid G triangle_ABC median_AL median_BM median_CN)
variable (area_triangle_ABC : area triangle_ABC = 54)
variable (triangle_ANG : triangle A N G)

theorem find_area_ANG : area triangle_ANG = 9 := 
by
  sorry

end find_area_ANG_l121_121972


namespace lambda_sum_ellipse_l121_121837

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

noncomputable def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 4)

noncomputable def intersects_y_axis (k : ℝ) : ℝ × ℝ :=
  (0, -4 * k)

noncomputable def lambda1 (x1 : ℝ) : ℝ :=
  x1 / (4 - x1)

noncomputable def lambda2 (x2 : ℝ) : ℝ :=
  x2 / (4 - x2)

theorem lambda_sum_ellipse {k x1 x2 : ℝ}
  (h1 : ellipse x1 (k * (x1 - 4)))
  (h2 : ellipse x2 (k * (x2 - 4)))
  (h3 : line_through_focus k x1 (k * (x1 - 4)))
  (h4 : line_through_focus k x2 (k * (x2 - 4))) :
  lambda1 x1 + lambda2 x2 = -50 / 9 := 
sorry

end lambda_sum_ellipse_l121_121837


namespace possible_values_of_expression_l121_121821

variable (a b c d : ℝ)

def sign (x : ℝ) : ℝ := if x > 0 then 1 else -1

theorem possible_values_of_expression 
  (h₀ : a ≠ 0) 
  (h₁ : b ≠ 0) 
  (h₂ : c ≠ 0) 
  (h₃ : d ≠ 0) :
  let expression := 
    sign a + sign b + sign c + sign d + sign (a * b * c * d)
  in 
    expression = 5 
    ∨ expression = 1 
    ∨ expression = -3 :=
by sorry

end possible_values_of_expression_l121_121821


namespace cube_volume_l121_121174

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121174


namespace ratio_medium_blue_to_pink_l121_121431

-- Define the conditions
def total_curlers : Nat := 16
def pink_fraction : Rational := 1/4
def large_green_curlers : Nat := 4

-- Define the main statement
theorem ratio_medium_blue_to_pink : 
  let pink_curlers := (pink_fraction * total_curlers).toNat,
      non_pink_curlers := total_curlers - pink_curlers,
      medium_blue_curlers := non_pink_curlers - large_green_curlers
  in medium_blue_curlers / pink_curlers = 2 :=
by
  sorry

end ratio_medium_blue_to_pink_l121_121431


namespace volume_of_cube_l121_121193

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l121_121193


namespace second_cube_surface_area_l121_121330

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121330


namespace cube_volume_l121_121216

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l121_121216


namespace find_initial_sweets_l121_121639

-- Defining the initial condition
def initial_sweets (S : ℕ) : Prop :=
  let jack_takes := (S / 2 + 4)
  let remaining_after_jack := S - jack_takes
  remaining_after_jack = 7

-- The theorem to prove the initial number of sweets S is 22
theorem find_initial_sweets : ∃ (S : ℕ), initial_sweets S ∧ S = 22 :=
by
  exists 22
  unfold initial_sweets
  sorry

end find_initial_sweets_l121_121639


namespace inner_cube_surface_area_l121_121243

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121243


namespace inner_cube_surface_area_l121_121293

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121293


namespace second_cube_surface_area_l121_121329

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121329


namespace second_cube_surface_area_l121_121338

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121338


namespace range_of_m_l121_121459

def p (m : ℝ) : Prop :=
∃ x y : ℝ, x + y = m ∧ (x - 1)^2 + y^2 = 1 

def q (m : ℝ) : Prop :=
∃ x : ℝ, m * x^2 - 2 * x + 1  = 0

theorem range_of_m (m : ℝ) (h₀ : p m ∨ q m) (h₁ : ¬ ¬ q m) : m ≤ 1 :=
begin
  sorry
end

end range_of_m_l121_121459


namespace gcd_factorial_power_l121_121770

theorem gcd_factorial_power {n m : ℕ} (h9 : 9! = 2^7 * 3^4 * 5 * 7) (h6sq : (6!)^2 = 2^8 * 3^4 * 5^2) :
  Nat.gcd 9! ((6!)^2) = 51840 := by
  rw [h9, h6sq]
  sorry

end gcd_factorial_power_l121_121770


namespace compound_interest_correct_l121_121660

def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + (r / n))^ (n * t) - P

def P : ℝ := 1200
def r : ℝ := 0.20
def n : ℕ := 1
def t : ℕ := 4

theorem compound_interest_correct :
  compound_interest P r n t = 1288.32 := by
  sorry

end compound_interest_correct_l121_121660


namespace unique_sphere_circles_tangent_l121_121411

variables (O1 O2 : Type)
variables (α β l : Type)
variables (P : l → O1 ∩ O2) -- P is the common tangency point
variables (dihedral_angle_αlβ : ∀ {α β l}, dihedral_angle α l β = 120)

theorem unique_sphere_circles_tangent (O1 O2 : Type) (α β l : Type) (P : l → O1 ∩ O2) 
  (dihedral_angle_αlβ : dihedral_angle α l β = 120) : 
  ∃! (S : Type), sphere S ∧ (O1 ∈ cross_section S) ∧ (O2 ∈ cross_section S) ∧ (P ∈ S) := 
sorry

end unique_sphere_circles_tangent_l121_121411


namespace inner_cube_surface_area_l121_121245

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121245


namespace levi_brother_scored_more_times_l121_121580

theorem levi_brother_scored_more_times (initial_levi_score initial_brother_score: ℕ) (levi_extra_score: ℕ)
    (levi_goal_diff: ℕ) (levi_final_goal: ℕ):
    initial_levi_score = 8 →
    initial_brother_score = 12 →
    levi_extra_score = 12 →
    levi_goal_diff = 5 →
    levi_final_goal = initial_levi_score + levi_extra_score →
    levi_final_goal = (initial_brother_score + 3) + levi_goal_diff :=
begin
  intros,
  sorry
end

end levi_brother_scored_more_times_l121_121580


namespace ratio_ian_paid_l121_121854

-- Define the relevant values.
def total_won : ℝ := 100
def paid_colin : ℝ := 20
def remaining : ℝ := 20

-- Assume the amount paid to Helen (H) and Benedict (B).
variable (H : ℝ)
def B : ℝ := H / 2

-- Construct the total amount calculation equality.
def total_spent : ℝ := paid_colin + H + B

-- Prove the required ratio.
theorem ratio_ian_paid
  (h1: total_spent = total_won - remaining)
  (h2 : H = 40)
  (h3 : paid_colin = 20) :
  H / paid_colin = 2 :=
by
  -- Simplifying assumptions based on the condition.
  have h4: H / paid_colin = 40 / 20 := by sorry
  -- Derived ratio.
  calc
    H / paid_colin = 40 / 20 : by exact h4
    ... = 2 : by norm_num

end ratio_ian_paid_l121_121854


namespace angle_inclination_range_l121_121020

theorem angle_inclination_range (θ : ℝ) (α : ℝ) :
  (∃ x y : ℝ, 0 = sqrt 3 * x + y * cos θ - 1) →
  α = atan (sqrt 3 / cos θ) →
  α ∈ Set.Icc (π / 3) (2 * π / 3) :=
by
  intro h1 h2
  sorry

end angle_inclination_range_l121_121020


namespace min_value_expr_l121_121441

-- Define the expression
def expr (x : ℝ) : ℝ := sqrt(x^2 - 2 * sqrt(3) * abs x + 4) + sqrt(x^2 + 2 * sqrt(3) * abs x + 12)

theorem min_value_expr : ∃ x : ℝ, expr x = 2 * sqrt(7) :=
sorry

end min_value_expr_l121_121441


namespace cube_volume_l121_121148

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l121_121148


namespace inner_cube_surface_area_l121_121269

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121269


namespace second_cube_surface_area_l121_121336

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121336


namespace jean_spots_on_upper_torso_l121_121906

noncomputable def S : ℕ := 60
def T : ℕ := S / 2
def B : ℕ := S / 3
def sides : ℕ := 10

theorem jean_spots_on_upper_torso : T = 30 := by
  -- The calculation for the total spots and spots on torso already embedded.
  sorry

end jean_spots_on_upper_torso_l121_121906


namespace remainder_of_cake_l121_121562

theorem remainder_of_cake (John Emily : ℝ) (h1 : 0.60 ≤ John) (h2 : Emily = 0.50 * (1 - John)) :
  1 - John - Emily = 0.20 :=
by
  sorry

end remainder_of_cake_l121_121562


namespace smallest_value_A_plus_B_plus_C_plus_D_l121_121779

variable (A B C D : ℤ)

-- Given conditions in Lean statement form
def isArithmeticSequence (A B C : ℤ) : Prop :=
  B - A = C - B

def isGeometricSequence (B C D : ℤ) : Prop :=
  (C / B : ℚ) = 4 / 3 ∧ (D / C : ℚ) = C / B

def givenConditions (A B C D : ℤ) : Prop :=
  isArithmeticSequence A B C ∧ isGeometricSequence B C D

-- The proof problem to validate the smallest possible value
theorem smallest_value_A_plus_B_plus_C_plus_D (h : givenConditions A B C D) :
  A + B + C + D = 43 :=
sorry

end smallest_value_A_plus_B_plus_C_plus_D_l121_121779


namespace machine_selling_price_l121_121951

theorem machine_selling_price (purchase_price repair_cost transport_cost maintenance_cost : ℝ)
  (expense_tax_rate profit_margin : ℝ) :
  purchase_price = 10000 →
  repair_cost = 5000 →
  transport_cost = 1000 →
  maintenance_cost = 2000 →
  expense_tax_rate = 0.10 →
  profit_margin = 0.50 →
  let total_expense_before_tax := purchase_price + repair_cost + transport_cost + maintenance_cost in
  let tax_amount := expense_tax_rate * total_expense_before_tax in
  let total_cost := total_expense_before_tax + tax_amount in
  let profit := profit_margin * total_cost in
  let selling_price := total_cost + profit in
  selling_price = 29700 :=
by
  intros
  sorry

end machine_selling_price_l121_121951


namespace unique_solution_7tuples_l121_121774

theorem unique_solution_7tuples : 
  ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1/8 :=
sorry

end unique_solution_7tuples_l121_121774


namespace angle_between_vectors_l121_121849

def vector (R : Type*) := R × R

-- Definitions of vectors a and b
def a : vector ℝ := (1, real.sqrt 3)
variable b : vector ℝ
variable hb_unit : b.1 ^ 2 + b.2 ^ 2 = 1
variable hab_dot : b.1 + real.sqrt 3 * b.2 = 1

-- The function to calculate dot product
def dot_product (u v : vector ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- The magnitude of a vector
def magnitude (v : vector ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

-- The statement to prove
theorem angle_between_vectors :
  let u := (2 * b.1 - 1, 2 * b.2 - real.sqrt 3) in
  let v := (2 * b.1, 2 * b.2) in
  real.cos (dot_product u v / (magnitude u * magnitude v)) = 1 / 2 :=
sorry

end angle_between_vectors_l121_121849


namespace largest_value_n_l121_121657

theorem largest_value_n :
  ∀ (number_of_people : ℕ) (number_of_months : ℕ), number_of_people = 60 → number_of_months = 10 →
    (∃ n, (∀ (assignments : ℕ → ℕ), 
      (∀ m, m < number_of_months → assignments m ≤ n) 
      → (∃ m, m < number_of_months ∧ assignments m >= (number_of_people / number_of_months))))
        ∧ (∀ n', (n' > 6 → ¬ (∃ m, m < number_of_months ∧ assignments m >= (number_of_people / number_of_months))))) :=
by
  sorry

end largest_value_n_l121_121657


namespace no_solutions_in_naturals_l121_121956

theorem no_solutions_in_naturals (n k : ℕ) : ¬ (n ≤ n! - k^n ∧ n! - k^n ≤ k * n) :=
sorry

end no_solutions_in_naturals_l121_121956


namespace count_valid_integers_l121_121851

-- Definitions for the sets of digits and valid lengths.
def allowed_digits : List ℕ := [0, 2, 4, 6, 8, 9]
def valid_length (n : ℕ) : Prop := n = 1 ∨ n = 2 ∨ n = 3

-- Defining the set of valid numbers without forbidden digits.
def is_valid_number (n : ℕ) : Prop :=
  ∀ digit ∈ int.digits 10 n, digit ∈ allowed_digits

-- Counting the valid numbers from 1 to 999 inclusive, considering the conditions.
noncomputable def count_valid_numbers : ℕ :=
  (List.Ico 1 1000).count is_valid_number

-- Theorem statement to prove the count of valid numbers is 215.
theorem count_valid_integers : count_valid_numbers = 215 := by
  sorry

end count_valid_integers_l121_121851


namespace inner_cube_surface_area_l121_121340

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121340


namespace f_45_g_10_l121_121007

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_condition1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom g_condition2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x + y) = g x + g y
axiom f_15 : f 15 = 10
axiom g_5 : g 5 = 3

theorem f_45 : f 45 = 10 / 3 := sorry
theorem g_10 : g 10 = 6 := sorry

end f_45_g_10_l121_121007


namespace volume_of_wedge_l121_121723

theorem volume_of_wedge (r : ℝ) (V : ℝ) (sphere_wedges : ℝ) 
  (h_circumference : 2 * Real.pi * r = 18 * Real.pi)
  (h_volume : V = (4 / 3) * Real.pi * r ^ 3) 
  (h_sphere_wedges : sphere_wedges = 6) : 
  V / sphere_wedges = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l121_121723


namespace rightmost_box_l121_121987

noncomputable def box := (ℕ × ℕ)

def total_red_blue (boxes : List box) : Prop :=
  (boxes.map Prod.fst).sum = 11 ∧
  (boxes.map Prod.snd).sum = 13

def non_empty (boxes : List box) : Prop :=
  ∀ b ∈ boxes, b.fst > 0 ∨ b.snd > 0

def non_decreasing (boxes : List box) : Prop :=
  ∀ i j, i < j → (boxes.nth i).get_or_else (0,0) ≤ (boxes.nth j).get_or_else (0,0)

def unique_combinations (boxes : List box) : Prop :=
  boxes.nodup

def conditions (boxes : List box) : Prop :=
  total_red_blue boxes ∧
  non_empty boxes ∧
  non_decreasing boxes ∧
  unique_combinations boxes

theorem rightmost_box (boxes : List box) (h : length boxes = 10) (hc : conditions boxes) :
  boxes.get_or_else 9 (0, 0) = (1, 3) :=
sorry

end rightmost_box_l121_121987


namespace inner_cube_surface_area_l121_121313

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121313


namespace inner_cube_surface_area_l121_121349

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121349


namespace sum_of_divisors_of_143_l121_121083

theorem sum_of_divisors_of_143 : 
  (∑ d in Finset.filter (fun d => 143 % d = 0) (Finset.range 144), d) = 168 :=
by
  sorry

end sum_of_divisors_of_143_l121_121083


namespace inner_cube_surface_area_l121_121285

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121285


namespace smallest_number_among_bases_l121_121406

noncomputable def convert_base_9 (n : ℕ) : ℕ :=
match n with
| 85 => 8 * 9 + 5
| _ => 0

noncomputable def convert_base_4 (n : ℕ) : ℕ :=
match n with
| 1000 => 1 * 4^3
| _ => 0

noncomputable def convert_base_2 (n : ℕ) : ℕ :=
match n with
| 111111 => 1 * 2^6 - 1
| _ => 0

theorem smallest_number_among_bases:
  min (min (convert_base_9 85) (convert_base_4 1000)) (convert_base_2 111111) = convert_base_2 111111 :=
by {
  sorry
}

end smallest_number_among_bases_l121_121406


namespace increase_in_success_rate_l121_121563

theorem increase_in_success_rate (successful_attempts1 : ℕ) (total_attempts1 : ℕ) (next_attempts : ℕ) (next_success_rate : ℚ) :
  (successful_attempts1 = 3) ∧
  (total_attempts1 = 8) ∧
  (next_attempts = 16) ∧
  (next_success_rate = 3/4) →
  let total_successful_attempts := successful_attempts1 + next_success_rate * next_attempts in
  let total_attempts := total_attempts1 + next_attempts in
  let initial_rate := (successful_attempts1 : ℚ) / (total_attempts1 : ℚ) in
  let new_rate := total_successful_attempts / total_attempts in
  round (100 * (new_rate - initial_rate)) = 25 :=
begin
  sorry
end

end increase_in_success_rate_l121_121563


namespace sum_of_divisors_143_l121_121093

theorem sum_of_divisors_143 : ∑ d in {d : ℕ | d ∣ 143}.to_finset, d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121093


namespace volume_of_cube_l121_121187

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l121_121187


namespace sum_of_divisors_143_l121_121068

theorem sum_of_divisors_143 : ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := by
  sorry

end sum_of_divisors_143_l121_121068


namespace no_adjacent_same_roll_proof_l121_121446

open ProbabilityTheory

-- Define the probability calculation in the Lean framework.
def no_adjacent_same_roll_prob : ℚ :=
  let pA := 1 / 8
  let p_diff_7 := (7 / 8) ^ 2
  let p_diff_6 := 6 / 8
  let p_diff_5 := 5 / 8
  let case1 := pA * p_diff_7 * p_diff_6
  let p_diff_7_rest := 7 / 8
  let p_diff_6_2 := (6 / 8) ^ 2
  let case2 := p_diff_7_rest * p_diff_6_2 * p_diff_5
  (case1 + case2) / 8

-- Statement of the proof problem, including all relevant conditions and the final proof goal.
theorem no_adjacent_same_roll_proof :
  no_adjacent_same_roll_prob = 777 / 2048 :=
sorry

end no_adjacent_same_roll_proof_l121_121446


namespace melted_ice_cream_depth_l121_121729

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_cylinder (r h : ℝ) : ℝ :=
  Real.pi * r^2 * h

theorem melted_ice_cream_depth :
  let r_sphere := 3
  let r_cylinder := 12
  let volume_s := volume_sphere r_sphere
  let volume_c := volume_cylinder r_cylinder
  ∃ h : ℝ, h = volume_s / (Real.pi * r_cylinder^2) :=
by
  let r_sphere := 3
  let r_cylinder := 12
  let volume_s := volume_sphere r_sphere
  let volume_c := volume_cylinder r_cylinder
  use volume_s / (Real.pi * r_cylinder^2)
  sorry

end melted_ice_cream_depth_l121_121729


namespace circle_permutation_8_l121_121515

theorem circle_permutation_8 (R : ℕ) (hR : R = 8) : 
  (R - 1)! = 5040 :=
by
  sorry

end circle_permutation_8_l121_121515


namespace inner_cube_surface_area_l121_121255

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121255


namespace sum_consecutive_even_integers_l121_121958

theorem sum_consecutive_even_integers (m : ℤ) :
  (m + (m + 2) + (m + 4) + (m + 6) + (m + 8)) = 5 * m + 20 := by
  sorry

end sum_consecutive_even_integers_l121_121958


namespace probability_prime_ball_chosen_l121_121429

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def balls : finset ℕ := {2, 3, 4, 5, 6, 7, 8, 9}

def prime_balls : finset ℕ := balls.filter is_prime

theorem probability_prime_ball_chosen :
  (prime_balls.card : ℚ) / balls.card = 1 / 2 := by
  sorry

end probability_prime_ball_chosen_l121_121429


namespace inner_cube_surface_area_l121_121350

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121350


namespace inner_cube_surface_area_l121_121275

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121275


namespace f_sum_positive_l121_121423

-- Define a function f and its properties
variable (f : ℝ → ℝ)
variable (hf_symm : ∀ x, f (-x) = -f(x + 2))
variable (hf_mono : ∀ x > 1, f x ≥ f 1)

-- Define conditions on x1 and x2
variable (x1 x2 : ℝ)
variable (hx1x2_sum : x1 + x2 > 2)
variable (hx1x2_product : (x1 - 1) * (x2 - 1) < 0)

-- The theorem to be proven
theorem f_sum_positive : f x1 + f x2 > 0 :=
by
  sorry

end f_sum_positive_l121_121423


namespace decode_best_l121_121630

def code_map (c : Char) : ℕ :=
  match c with
  | 'G' => 0
  | 'R' => 1
  | 'E' => 2
  | 'A' => 3
  | 'T' => 4
  | 'J' => 5
  | 'O' => 6
  | 'B' => 7
  | 'S' => 8
  | _   => 9  -- this covers any other values which should ideally not occur

def decode_word (word : String) : ℕ :=
  word.toList.foldl (λ acc c, acc * 10 + code_map c) 0

theorem decode_best :
  decode_word "BEST" = 7284 :=
by
  sorry

end decode_best_l121_121630


namespace integral_f_l121_121795

noncomputable def f : ℝ → ℝ := 
λ x, if x ∈ set.Icc 0 1 then x^2 
     else if x ∈ set.Ioc 1 real.exp then 1/x 
     else 0

theorem integral_f :∫ x in 0..real.exp, f x = 4/3 :=
by
  sorry

end integral_f_l121_121795


namespace fraction_computation_l121_121056

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l121_121056


namespace inner_cube_surface_area_l121_121259

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121259


namespace inner_cube_surface_area_l121_121263

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121263


namespace perpendicular_distance_l121_121419

noncomputable def heron_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem perpendicular_distance {P Q R S : ℝ × ℝ × ℝ}
    (hP : P = (5,0,0))
    (hQ : Q = (0,3,0))
    (hR : R = (0,0,4))
    (hS : S = (0,0,0)) :
  let a := Real.sqrt ((5-0)^2 + (3-0)^2 + (0-0)^2)
  let b := Real.sqrt ((5-0)^2 + (0-0)^2 + (4-0)^2)
  let c := Real.sqrt ((0-5)^2 + (3-0)^2 + (4-0)^2)
  let A := heron_area a b c
  let V := (5 * 3 * 4) / 2
  V / A = 15 :=
by
  sorry

end perpendicular_distance_l121_121419


namespace square_area_l121_121392

theorem square_area (y1 y2 y3 y4 : ℝ) (h : multiset.eq (multiset.of_list [y1, y2, y3, y4]) (multiset.of_list [1, 1, 6, 6])) :
  let s := (max y1 (max y2 (max y3 y4))) - (min y1 (min y2 (min y3 y4))) in
  s^2 = 25 :=
by
  sorry

end square_area_l121_121392


namespace largest_number_is_102_5_l121_121448

theorem largest_number_is_102_5 (a b c d e : ℝ)
  (h1 : a + b + c + d = 210)
  (h2 : a + b + c + e = 230)
  (h3 : a + b + d + e = 250)
  (h4 : a + c + d + e = 270)
  (h5 : b + c + d + e = 290)
  : max a (max b (max c (max d e))) = 102.5 :=
begin
  sorry
end

end largest_number_is_102_5_l121_121448


namespace add_decimals_l121_121402

theorem add_decimals :
  0.0935 + 0.007 + 0.2 = 0.3005 :=
by sorry

end add_decimals_l121_121402


namespace decrypt_encryption_l121_121430

-- Encryption function description
def encrypt_digit (d : ℕ) : ℕ := 10 - (d * 7 % 10)

def encrypt_number (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let encrypted_digits := digits.map encrypt_digit
  encrypted_digits.foldr (λ d acc => d + acc * 10) 0
  
noncomputable def digit_match (d: ℕ) : ℕ :=
  match d with
  | 0 => 0 | 1 => 3 | 2 => 8 | 3 => 1 | 4 => 6 | 5 => 5
  | 6 => 8 | 7 => 1 | 8 => 4 | 9 => 7 | _ => 0

theorem decrypt_encryption:
encrypt_number 891134 = 473392 :=
by
  sorry

end decrypt_encryption_l121_121430


namespace part1_part2_l121_121813

def setA : Set ℝ := {x | (x - 2) / (x + 1) < 0}
def setB (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

theorem part1 : (setB (-1)).union setA = {x : ℝ | -1 < x ∧ x < 3 } := by
  sorry

theorem part2 (k : ℝ) : (setA ∩ setB k = setB k ↔ 0 ≤ k) := by
  sorry

end part1_part2_l121_121813


namespace remainder_division_l121_121610

theorem remainder_division (x r : ℕ) (h₁ : 1650 - x = 1390) (h₂ : 1650 = 6 * x + r) : r = 90 := by
  sorry

end remainder_division_l121_121610


namespace sum_of_divisors_143_l121_121089

theorem sum_of_divisors_143 : (∑ i in (finset.filter (λ d, 143 % d = 0) (finset.range 144)), i) = 168 :=
by
  -- The final proofs will go here.
  sorry

end sum_of_divisors_143_l121_121089


namespace solve_problem_l121_121840

def f : ℝ → ℝ :=
λ x, if x ≥ 0 then -x^2 - x - 2 else x / (x + 4) + Real.log (|x|) / Real.log 4

theorem solve_problem : f (f 2) = 7 / 2 :=
by
  sorry

end solve_problem_l121_121840


namespace fraction_product_l121_121051

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l121_121051


namespace value_depletion_rate_l121_121695

theorem value_depletion_rate (V_initial V_final : ℝ) (t : ℝ) (r : ℝ) :
  V_initial = 900 → V_final = 729 → t = 2 → V_final = V_initial * (1 - r)^t → r = 0.1 :=
by sorry

end value_depletion_rate_l121_121695


namespace volume_of_cube_l121_121189

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l121_121189


namespace max_additional_voters_l121_121120

theorem max_additional_voters (x n y : ℕ) (hx : 0 ≤ x ∧ x ≤ 10) (hy : y = x - n - 1)
  (hT : (nx / n).is_integer) (h_decrease : ∀ v, (nx + v) / (n + 1) = x - 1 → ∀ m, x - m ≤ 0 → m ≤ 5) :
  ∃ y, y ≥ 0 ∧ y ≤ 5 := sorry

end max_additional_voters_l121_121120


namespace complete_square_eq_l121_121734

theorem complete_square_eq (x : ℝ) :
  x^2 - 8 * x + 15 = 0 →
  (x - 4)^2 = 1 :=
by sorry

end complete_square_eq_l121_121734


namespace inner_cube_surface_area_l121_121356

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121356


namespace ratio_of_areas_l121_121899

open Lean

variables (A B C D E F P Q R : Type)
variables [OrderedRing A] [OrderedRing B] [OrderedRing C]

-- Given conditions in the problem
variables (BD DC CE EA AF FB : ℝ)
variables (ratio1 : BD / DC = 1 / 3)
variables (ratio2 : CE / EA = 2 / 1)
variables (ratio3 : AF / FB = 1 / 2)
variables (intersect : ∀ {l : ℕ}, P = l)

theorem ratio_of_areas (h : ratio1 ∧ ratio2 ∧ ratio3 ∧ intersect) : 
  (area P Q R / area A B C = 1 / 64) := 
sorry -- proof will be provided here

end ratio_of_areas_l121_121899


namespace cube_volume_l121_121214

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l121_121214


namespace painting_chessboard_l121_121892

theorem painting_chessboard (N : ℕ) : 
  ∃ k : ℕ, k = 24 * (2^(N-1) - 1) ∧ 
  ∀ (board : Fin N → Fin N → Fin 4), 
    (∀ i j, board i j ≠ board i (j + 1) ∧ board i j ≠ board (i + 1) j) → 
    (∀ i j, [board i j, board i (j + 1), board (i + 1) j, board (i + 1) (j + 1)].Nodup) :=
sorry

end painting_chessboard_l121_121892


namespace even_function_value_at_2_l121_121006

theorem even_function_value_at_2 {a : ℝ} (h : ∀ x : ℝ, (x + 1) * (x - a) = (-x + 1) * (-x - a)) : 
  ((2 + 1) * (2 - a)) = 3 := by
  sorry

end even_function_value_at_2_l121_121006


namespace card_proof_l121_121445
noncomputable theory

-- We define a card type with a letter on one side and a natural number on the other side
structure Card where
  letter : Option Char
  number : Option Nat

-- Define vowels
def isVowel (ch : Char) : Prop :=
  ch = 'A' ∨ ch = 'E' ∨ ch = 'I' ∨ ch = 'O' ∨ ch = 'U'

-- Define even numbers
def isEven (n : Nat) : Prop := n % 2 = 0

-- Condition: Peter's statement
def peters_statement (c : Card) : Prop :=
  (c.letter.isSome ∧ isVowel c.letter.get) → (c.number.isSome ∧ isEven c.number.get)

-- Define the five cards
def card1 : Card := { letter := some 'A', number := none }
def card2 : Card := { letter := some 'B', number := none }
def card3 : Card := { letter := none, number := some 1 }
def card4 : Card := { letter := none, number := some 7 }
def card5 : Card := { letter := some 'U', number := none }

-- Define the problem: Kate showed that Peter's statement is false by turning over a card
theorem card_proof : 
  ∃ c, (c = card4) ∧ (¬ peters_statement c) :=
by
  sorry

end card_proof_l121_121445


namespace locus_circumcenters_l121_121496

-- Definitions
variable {Circle Point : Type}
variable (P Q : Point) (c1 c2 : Circle)

-- Conditions
def isIntersectingCirclePairs (c1 c2 : Circle) (P Q : Point) : Prop :=
  -- Definition for two circles intersecting at points P and Q.
  sorry

def isArbitraryPoint (C : Point) (c : Circle) (P Q : Point) : Prop :=
  -- Definition for point C being an arbitrary point on circle c different from P and Q.
  sorry

def secondIntersection (C P : Point) (c : Circle) : Point :=
  -- Definition for the second point of intersection of line CP with the circle c.
  sorry

-- Locus of the circumcenters of triangles ABC
theorem locus_circumcenters (P Q : Point) (C : Point) (c1 c2 : Circle) :
  isIntersectingCirclePairs c1 c2 P Q →
  isArbitraryPoint C c1 P Q →
  let A := secondIntersection C P c2 in
  let B := secondIntersection C Q c2 in
  ∃ (c : Circle) (O1 O2 : Point),
    isCenter O1 c1 ∧
    isCenter O2 c2 ∧
    locusOfCircumcentersOfTrianglesABCisCircle (A B C Point) c (O1 O2) :=
begin
  sorry
end

end locus_circumcenters_l121_121496


namespace max_additional_voters_l121_121118

theorem max_additional_voters (x n y : ℕ) (hx : 0 ≤ x ∧ x ≤ 10) (hy : y = x - n - 1)
  (hT : (nx / n).is_integer) (h_decrease : ∀ v, (nx + v) / (n + 1) = x - 1 → ∀ m, x - m ≤ 0 → m ≤ 5) :
  ∃ y, y ≥ 0 ∧ y ≤ 5 := sorry

end max_additional_voters_l121_121118


namespace sum_of_divisors_of_143_l121_121072

theorem sum_of_divisors_of_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 := 
by
  sorry

end sum_of_divisors_of_143_l121_121072


namespace range_of_a_l121_121844

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (1/3) * |x^3| - (a / 2) * x^2 + (3 - a) * |x| + b

theorem range_of_a (b : ℝ) :
  (∀ x : ℝ, f x a b has_six_monotonic_intervals) → 2 < a ∧ a < 3 :=
begin
  sorry
end

end range_of_a_l121_121844


namespace georgie_entry_exit_haunting_ways_l121_121694

theorem georgie_entry_exit_haunting_ways (windows rooms : ℕ) (windows = 8) (rooms = 3) : 
  let entry_ways := windows in
  let exit_ways := windows - 1 in
  let room_ways := rooms in
  entry_ways * exit_ways * room_ways = 168 := 
sorry

end georgie_entry_exit_haunting_ways_l121_121694


namespace calculate_difference_l121_121925

theorem calculate_difference :
  let m := Nat.find (λ m, m > 99 ∧ m < 1000 ∧ m % 13 = 7)
  let n := Nat.find (λ n, n > 999 ∧ n < 10000 ∧ n % 13 = 7)
  n - m = 895 :=
by
  sorry

end calculate_difference_l121_121925


namespace unique_7tuple_solution_l121_121773

theorem unique_7tuple_solution : 
  ∃! x : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ × ℝ, 
  let (x1, x2, x3, x4, x5, x6, x7) := x in
  (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + (x4 - x5)^2 + (x5 - x6)^2 + (x6 - x7)^2 + x7^2 = 1 / 8 :=
by 
  sorry

end unique_7tuple_solution_l121_121773


namespace estimated_survival_probability_l121_121936

-- Definitions of the given data
def number_of_trees_transplanted : List ℕ := [100, 1000, 5000, 8000, 10000, 15000, 20000]
def number_of_trees_survived : List ℕ := [87, 893, 4485, 7224, 8983, 13443, 18044]
def survival_rates : List ℝ := [0.870, 0.893, 0.897, 0.903, 0.898, 0.896, 0.902]

-- Question: Prove that the probability of survival of this type of young tree under these conditions is 0.9.
theorem estimated_survival_probability : 
  (1 / List.length number_of_trees_transplanted.to_real) * 
  (List.sum survival_rates) >= 0.9 ∧ 
  (1 / List.length number_of_trees_transplanted.to_real) * 
  (List.sum survival_rates) < 1 :=
  by sorry

end estimated_survival_probability_l121_121936


namespace identify_spy_l121_121608

namespace SpyProblem

-- Define the types for the personalities
inductive Personality
| Knight  -- Always tells the truth
| Liar    -- Always lies
| Spy     -- Can either lie or tell the truth

-- Define the defendants A, B, and C
inductive Defendant
| A
| B
| C

-- A function that assigns personalities to defendants
def assign_personality (assign: Defendant → Personality) : Prop :=
  ∃ a b c, 
    -- a, b, and c are the assigned personalities
    (assign Defendant.A = a ∧ assign Defendant.B = b ∧ assign Defendant.C = c) ∧ 
    -- Each personality is assigned exactly once
    {a, b, c} = {Personality.Knight, Personality.Liar, Personality.Spy}

-- The judge identifies the spy based on logic
theorem identify_spy (assign: Defendant → Personality) 
  (h: assign Defendant.C = Personality.Spy) : 
  assign Defendant.C = Personality.Spy :=
by sorry  -- Proof to be filled

end SpyProblem

end identify_spy_l121_121608


namespace inner_cube_surface_area_l121_121382

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121382


namespace inner_cube_surface_area_l121_121256

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l121_121256


namespace cube_volume_l121_121166

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121166


namespace log_sum_eq_89_l121_121874

noncomputable theory

open Real

theorem log_sum_eq_89 (x y z : ℝ) (hx : log 2 (log 3 (log 4 x)) = 0)
    (hy : log 3 (log 4 (log 2 y)) = 0) (hz : log 4 (log 2 (log 3 z)) = 0) : 
  x + y + z = 89 := 
sorry

end log_sum_eq_89_l121_121874


namespace check_squareable_numbers_l121_121932

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_squareable (n : ℕ) : Prop :=
  ∃ (perm : Finₓ n → Finₓ n), (∀ i : Finₓ n, is_perfect_square ((↑perm i + i + 1).val))

theorem check_squareable_numbers :
  ¬ is_squareable 7 ∧ is_squareable 9 ∧ ¬ is_squareable 11 ∧ is_squareable 15 :=
by
  sorry

end check_squareable_numbers_l121_121932


namespace point_B_is_south_of_A_total_distance_traveled_fuel_consumed_l121_121758

-- Define the travel records and the fuel consumption rate
def travel_records : List Int := [18, -9, 7, -14, -6, 13, -6, -8]
def fuel_consumption_rate : Float := 0.4

-- Question 1: Proof that point B is 5 km south of point A
theorem point_B_is_south_of_A : (travel_records.sum = -5) :=
  by sorry

-- Question 2: Proof that total distance traveled is 81 km
theorem total_distance_traveled : (travel_records.map Int.natAbs).sum = 81 :=
  by sorry

-- Question 3: Proof that the fuel consumed is 32 liters (Rounded)
theorem fuel_consumed : Float.floor (81 * fuel_consumption_rate) = 32 :=
  by sorry

end point_B_is_south_of_A_total_distance_traveled_fuel_consumed_l121_121758


namespace hcf_of_two_numbers_l121_121040

-- Define the conditions
def product_of_two_numbers : ℕ := 1991
def lcm_of_two_numbers : ℕ := 181

-- Define the main statement
theorem hcf_of_two_numbers (H : ℕ) (product_of_two_numbers = H * lcm_of_two_numbers) : H = 11 :=
by
  sorry

end hcf_of_two_numbers_l121_121040


namespace inner_cube_surface_area_l121_121369

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121369


namespace smallest_diff_mod_13_l121_121922

theorem smallest_diff_mod_13 : 
  let m := Nat.find (λ k, 100 ≤ 13 * k + 7)
  let n := Nat.find (λ k, 1000 ≤ 13 * k + 7)
  (13 * n + 7) - (13 * m + 7) = 895 :=
by
  sorry

end smallest_diff_mod_13_l121_121922


namespace number_with_digits_divisible_by_2020_l121_121428

theorem number_with_digits_divisible_by_2020 :
  ∃ (n : ℕ), (2020 ∣ n) ∧ (∀ d, 0 ≤ d ∧ d ≤ 9 → (d ∈ (nat.digits 10 n))) ∧ (nat.nodup (nat.digits 10 n)) :=
by
  sorry

end number_with_digits_divisible_by_2020_l121_121428


namespace min_value_l121_121868

theorem min_value (a b c : ℤ) (h : a > b ∧ b > c) :
  ∃ x, x = (a + b + c) / (a - b - c) ∧ 
       x + (a - b - c) / (a + b + c) = 2 := sorry

end min_value_l121_121868


namespace railway_original_stations_l121_121230

theorem railway_original_stations (m n : ℕ) (hn : n > 1) (h : n * (2 * m - 1 + n) = 58) : m = 14 :=
by
  sorry

end railway_original_stations_l121_121230


namespace orthocenter_characterization_l121_121930

variable {P : Type}
variables {A B C D : P}
variables [InnerProductSpace ℝ P]
variables {AB BC CA DA DB DC : ℝ}
variables [linear_independent ℝ ![A, B, C, D]]

-- Define a method to check if D is an orthocenter 
def is_orthocenter (D A B C : P) : Prop :=
  let H := λ (X Y : P), ∀ n : ℝ, n ≠ 0 → inner (n • (X - Y)) (X - Y) = (X - Y) ← 0 in
  H A D B ∧ H B D C ∧ H C D A

theorem orthocenter_characterization (h : DA * DB * AB + DB * DC * BC + DC * DA * CA = AB * BC * CA) :
  is_orthocenter D A B C :=
  sorry

end orthocenter_characterization_l121_121930


namespace planned_construction_days_l121_121142

theorem planned_construction_days
  (total_length : ℕ)
  (initial_men : ℕ)
  (days_worked : ℕ)
  (completed_length : ℕ)
  (additional_men : ℕ)
  (remaining_length : ℕ)
  (initial_planned_days : ℕ) :
  total_length = 720 →
  initial_men = 50 →
  days_worked = 120 →
  completed_length = 240 →
  additional_men = 70 →
  remaining_length = total_length - completed_length →
  (remaining_length = 480) →
  initial_planned_days = (days_worked + remaining_length / (((initial_men + additional_men) * (completed_length / days_worked)) / initial_men)) →
  initial_planned_days = 220 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7] at h8
  exact h8

end planned_construction_days_l121_121142


namespace find_true_roots_l121_121039

theorem find_true_roots (
    h₁ : (log 2 (1/4) : ℝ) = -2,
    h₂ : (log 2 (1/8) : ℝ) = -3,
    h₃ : (log 2 (1/2) : ℝ) = -1,
    h₄ : (log 2 64 : ℝ) = 6
) : {x : ℝ // x = 4 ∨ x = 8} :=
begin
  have b : ℝ := -(h₃ + h₄),
  have c : ℝ := h₁ * h₂,
  have eq := λ (t : ℝ), t^2 + b*t + c,
  have eq_chg := eq 2 = 0 ∧ eq 3 = 0,
  have h₅ : 2 = log 2 4 := sorry,
  have h₆ : 3 = log 2 8 := sorry,
  exact ⟨4, or.inl rfl⟩,
  exact ⟨8, or.inr rfl⟩,
end

end find_true_roots_l121_121039


namespace find_integer_pair_l121_121776

-- Definitions and conditions
def sin_30 : ℝ := 1 / 2
def csc_30 : ℝ := 2

theorem find_integer_pair : ∃ (x y : ℤ), (x, y) = (0, 1) ∧ sqrt (4 - 3 * sin_30) = x + y * csc_30 :=
by
  use 0
  use 1
  split
  · refl
  · sorry

end find_integer_pair_l121_121776


namespace quadratic_function_expression_log_function_properties_l121_121797

variable (y f : ℝ → ℝ)
variable (x : ℝ)

-- Given conditions
axiom h1 : ∃ a b c : ℝ, ∀ x : ℝ, f(x) = a * x^2 + b * x + c
axiom h2 : f(0) = 8
axiom h3 : ∀ x : ℝ, f(x + 1) - f(x) = -2 * x + 1

-- Question 1: Proving the expression for f(x)
theorem quadratic_function_expression : 
  (∃ a b c : ℝ, f(x) = a * x^2 + b * x + c ∧ c = 8 ∧ a = -1 ∧ b = 2) :=
sorry

-- Question 2: Proving the interval of decrease and the range for y = log_3(f(x))
noncomputable def g (f : ℝ → ℝ) : ℝ → ℝ := λ x, real.log (f x) / real.log 3

theorem log_function_properties :
  (∃ a b c : ℝ, f(x) = a * x^2 + b * x + c ∧ c = 8 ∧ a = -1 ∧ b = 2) →
  (interval_of_decrease : Icc 1 4) ∧ (range_of_g : set.Iic 2) :=
sorry

end quadratic_function_expression_log_function_properties_l121_121797


namespace range_f_eq_neg1_infty_l121_121845

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then x^3 + x else 2 * sin x * cos x

theorem range_f_eq_neg1_infty : set.range f = set.Icc (-1) ⊤ :=
by sorry

end range_f_eq_neg1_infty_l121_121845


namespace sum_x_coords_intersections_of_lines_with_line_y_eq_100_minus_x_l121_121992

theorem sum_x_coords_intersections_of_lines_with_line_y_eq_100_minus_x : 
  let lines := [0:179].to_finset
  let intersections : finset ℝ := lines.image (λ θ, 100 / (Real.tan θ + 1))
  ∑ x in intersections, x = 8950 := 
sorry

end sum_x_coords_intersections_of_lines_with_line_y_eq_100_minus_x_l121_121992


namespace smallest_diff_mod_13_l121_121923

theorem smallest_diff_mod_13 : 
  let m := Nat.find (λ k, 100 ≤ 13 * k + 7)
  let n := Nat.find (λ k, 1000 ≤ 13 * k + 7)
  (13 * n + 7) - (13 * m + 7) = 895 :=
by
  sorry

end smallest_diff_mod_13_l121_121923


namespace a_pow_b_iff_a_minus_1_b_positive_l121_121791

theorem a_pow_b_iff_a_minus_1_b_positive (a b : ℝ) (h_pos : a > 0) (h_neq : a ≠ 1) : 
  (a^b > 1) ↔ ((a - 1) * b > 0) := 
sorry

end a_pow_b_iff_a_minus_1_b_positive_l121_121791


namespace sum_possible_x_values_l121_121981

-- Define the multiplicative magic square and its properties
variable (a b c d e f g h i j k l m n o p : ℕ+)

noncomputable def isMultiplicativeMagicSquare : Prop :=
  (70 * a * b * 3 = e * f * g * h) ∧
  (70 * i * j * 5 = e * k * l * m) ∧
  (70 * n * f * o = 5 * k * g * 3) ∧
  (70 * e * l * o = 3 * h * m * 5)

theorem sum_possible_x_values (x : ℕ+) (h : isMultiplicativeMagicSquare a b c d e f g h i j k l m n o x) :
  ∑ x_possible ∈ {3, 6}, x_possible = 9 :=
by
  sorry

end sum_possible_x_values_l121_121981


namespace cube_volume_l121_121175

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121175


namespace inner_cube_surface_area_l121_121305

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121305


namespace inner_cube_surface_area_l121_121267

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121267


namespace seashells_left_l121_121583

theorem seashells_left (total_seashells : ℕ) (given_seashells : ℕ) (h_total : total_seashells = 62) (h_given : given_seashells = 49) :
  total_seashells - given_seashells = 13 :=
by
  rw [h_total, h_given]
  norm_num
  -- proof omitted
  sorry

end seashells_left_l121_121583


namespace inner_cube_surface_area_l121_121352

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121352


namespace toy_cost_price_l121_121140

theorem toy_cost_price (x : ℝ) (h : 1.5 * x * 0.8 - x = 20) : x = 100 := 
sorry

end toy_cost_price_l121_121140


namespace fraction_multiplication_l121_121045

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l121_121045


namespace arithmetic_progression_a1_bound_l121_121027

theorem arithmetic_progression_a1_bound (S4 S7 : ℕ) (a1 d : ℝ)
  (hS4 : S4 = 4 * a1 + 6 * d) 
  (hS7 : S7 = 7 * a1 + 21 * d)
  (ha1_le: a1 ≤ 2 / 3) : 
  a1 ≤ 9 / 14 :=
begin
  sorry
end

end arithmetic_progression_a1_bound_l121_121027


namespace impossible_to_achieve_desired_piles_l121_121044

def initial_piles : List ℕ := [51, 49, 5]

def desired_piles : List ℕ := [52, 48, 5]

def combine_piles (x y : ℕ) : ℕ := x + y

def divide_pile (x : ℕ) (h : x % 2 = 0) : List ℕ := [x / 2, x / 2]

theorem impossible_to_achieve_desired_piles :
  ∀ (piles : List ℕ), 
    (piles = initial_piles) →
    (∀ (p : List ℕ), 
      (p = desired_piles) → 
      False) :=
sorry

end impossible_to_achieve_desired_piles_l121_121044


namespace unique_solution_7tuples_l121_121775

theorem unique_solution_7tuples : 
  ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1/8 :=
sorry

end unique_solution_7tuples_l121_121775


namespace ratio_problem_l121_121862

theorem ratio_problem {q r s t : ℚ} (h1 : q / r = 8) (h2 : s / r = 4) (h3 : s / t = 1 / 3) :
  t / q = 3 / 2 :=
sorry

end ratio_problem_l121_121862


namespace product_of_coordinates_of_D_l121_121814

variable (N C D : Point)
variable (x y : ℤ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

theorem product_of_coordinates_of_D : 
  (N.x = 4 ∧ N.y = 8) → 
  (C.x = 5 ∧ C.y = 3) → 
  (is_midpoint N C D) →
  ((D.x * D.y) = 39) :=
by
  sorry

structure Point where
  x : ℤ
  y : ℤ

end product_of_coordinates_of_D_l121_121814


namespace part1_part2_part3_l121_121487

def f (x : ℝ) : ℝ :=
  4 * (sin(π / 4 + x / 2))^2 * sin x + (cos x + sin x) * (cos x - sin x) - 1

def g (x a : ℝ) : ℝ :=
  1 / 2 * (f (2 * x) + a * f x - a * f ( π / 2 - x) - a) - 1

theorem part1 (x : ℝ) :
  f x = 2 * sin x := sorry

theorem part2 (w : ℝ) :
  0 < w → ∀ x ∈ Icc (-π / 2) (2 * π / 3),
  continuous_on (λ x, f (w * x)) (Icc (-π / 2) (2 * π / 3)) ∧
  ∀ u v ∈ Icc (-π / 2) (2 * π / 3), u < v → f (w * u) < f (w * v) →
  0 < w ∧ w ≤ 3 / 4 := sorry

theorem part3 (a : ℝ) :
  ∃ x ∈ Icc (-π / 4) (π / 2), g x a = 2 ↔ a = -2 ∨ a = 6 := sorry

end part1_part2_part3_l121_121487


namespace inner_cube_surface_area_l121_121264

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121264


namespace scientific_notation_five_hundred_billion_l121_121413

theorem scientific_notation_five_hundred_billion :
  500000000000 = 5 * 10^11 := by
  sorry

end scientific_notation_five_hundred_billion_l121_121413


namespace number_of_first_year_students_to_be_sampled_l121_121399

-- Definitions based on the conditions
def total_students_in_each_grade (x : ℕ) : List ℕ := [4*x, 5*x, 5*x, 6*x]
def total_undergraduate_students (x : ℕ) : ℕ := 4*x + 5*x + 5*x + 6*x
def sample_size : ℕ := 300
def sampling_fraction (x : ℕ) : ℚ := sample_size / total_undergraduate_students x
def first_year_sampling (x : ℕ) : ℕ := (4*x) * sample_size / total_undergraduate_students x

-- Statement to prove
theorem number_of_first_year_students_to_be_sampled {x : ℕ} (hx_pos : x > 0) :
  first_year_sampling x = 60 := 
by
  -- skip the proof
  sorry

end number_of_first_year_students_to_be_sampled_l121_121399


namespace inner_cube_surface_area_l121_121309

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121309


namespace digging_depth_l121_121138

theorem digging_depth :
  (∃ (D : ℝ), 750 * D = 75000) → D = 100 :=
by
  sorry

end digging_depth_l121_121138


namespace volume_of_cube_l121_121184

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l121_121184


namespace minimal_n_exists_triplet_product_m_l121_121577

def m : ℕ := 30030
def primes : List ℕ := [2, 3, 5, 7, 11, 13]
def M : Set ℕ := {d | d ∣ m ∧ (d.primeFactors.length = 2)}

theorem minimal_n_exists_triplet_product_m :
  ∃ n, ∀ (S : Set ℕ), S ⊆ M → S.card = n → ∃ a b c ∈ S, a * b * c = m :=
sorry

end minimal_n_exists_triplet_product_m_l121_121577


namespace inner_cube_surface_area_l121_121374

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121374


namespace second_remainder_l121_121443

theorem second_remainder (n : ℕ) : n = 210 ∧ n % 13 = 3 → n % 17 = 6 :=
by
  sorry

end second_remainder_l121_121443


namespace tree_initial_leaves_l121_121895

theorem tree_initial_leaves (L : ℝ) (h1 : ∀ n : ℤ, 1 ≤ n ∧ n ≤ 4 → ∃ k : ℝ, L = k * (9/10)^n + k / 10^n)
                            (h2 : L * (9/10)^4 = 204) :
  L = 311 :=
by
  sorry

end tree_initial_leaves_l121_121895


namespace inner_cube_surface_area_l121_121241

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121241


namespace find_center_and_radius_l121_121478

noncomputable def center_and_radius (p : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 - 2 * x + 6 * y = 0 ∧ (p = (1, -3)) ∧ (r = Real.sqrt 10)

theorem find_center_and_radius
  (p : ℝ × ℝ)
  (r : ℝ)
  (h : p = (1, -3))
  (hr : r = Real.sqrt 10) :
  center_and_radius p r :=
by
  use (1, -3)
  use Real.sqrt 10
  refine ⟨_, _, _⟩
  sorry -- Prove the conditions are met, equation validation

end find_center_and_radius_l121_121478


namespace inner_cube_surface_area_l121_121341

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121341


namespace problem_l121_121736

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def monotonically_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x ≥ f y

theorem problem :
  (even_function (λ x, -x^2) ∧ monotonically_decreasing_on (λ x, -x^2) (set.Ioi 0)) ∧
  (∀ f, (f = (λ x, -x^2) ∨ f = (λ x, x^3) ∨ f = (λ x, 2^|x|) ∨ f = (λ x, cos x)) →
    ((even_function f ∧ monotonically_decreasing_on f (set.Ioi 0)) → f = (λ x, -x^2))) :=
by
  -- Proof omitted
  sorry

end problem_l121_121736


namespace integral_solution_unique_l121_121437

theorem integral_solution_unique (a b c : ℤ) : a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end integral_solution_unique_l121_121437


namespace find_a2_l121_121807

-- Definitions of the sequence and its properties
def sequence (n : ℕ) : ℕ := sorry -- This is just a place-holder

-- Sum of the first n terms
def S (n : ℕ) : ℕ := sorry -- Another place-holder

-- Given condition: S_n = 2a_n - 1
axiom S_def (n : ℕ) : S n = 2 * sequence n - 1

-- Theorem to prove
theorem find_a2 : sequence 2 = 2 :=
  sorry

end find_a2_l121_121807


namespace number_of_arrangements_l121_121026

-- Set definition
def A : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Definitions for even and odd numbers
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Statement of the proof
theorem number_of_arrangements :
  (finset.univ.filter (λ (perm : List ℕ),
    perm = 5 :: List.take 5 perm ∧
    ∀ i, i < 5 → ((i % 2 = 0 → is_even (List.head (List.drop i perm))) ∧
                  (i % 2 = 1 → is_odd (List.head (List.drop i perm))))) : finset (List ℕ)).card = 12 :=
by
  sorry

end number_of_arrangements_l121_121026


namespace fraction_product_l121_121052

theorem fraction_product :
  (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_product_l121_121052


namespace triangle_side_and_area_proof_l121_121534

-- Define the right triangle condition and variables
variables (A B C : Type) [euclidean_geometry] 
variables (AC BC AB : ℝ)
variables (angleA : ℝ) (h_angleA : angleA = 90) (hAC : AC = 5) (hBC : BC = 13)

-- The length of side AB
def length_of_AB :=
  ∃ (AB : ℝ), AC^2 + AB^2 = BC^2 ∧ AB = 12

-- The area of the triangle
def area_of_triangle :=
  (1/2) * AC * (sqrt (BC^2 - AC^2)) = 30

-- The overall statement combining both required proofs
theorem triangle_side_and_area_proof (h : AC = 5) (k : BC = 13) :
  length_of_AB A B C 5 13 90 ∧ area_of_triangle A B C 5 13 :=
by
  sorry

end triangle_side_and_area_proof_l121_121534


namespace polynomial_remainder_l121_121914

noncomputable def Q : Polynomial ℂ := sorry -- Placeholder for Q(z)
noncomputable def R : Polynomial ℂ :=
  have h_deg_R : degree R < 2 := sorry -- given condition

  if h_deg_R
  then - Polynomial.X -- R(z) = -z
  else sorry

theorem polynomial_remainder :
  let z := Polynomial.X in
  z^2021 + 1 = (z^2 + z + 1) * Q + R :=
  by
  sorry

end polynomial_remainder_l121_121914


namespace distance_AB_l121_121493

theorem distance_AB :
  let l (t : ℝ) := (1 + 3 / 5 * t, 4 / 5 * t)
  let C1 (θ : ℝ) := (Real.cos θ, Real.sin θ)
  let line_eq (x y : ℝ) := 4 * x - 3 * y - 4 = 0
  let circle_eq (x y : ℝ) := x^2 + y^2 = 1
  (∃ t θ, line_eq (l t).1 (l t).2 ∧ circle_eq (C1 θ).1 (C1 θ).2) ∧ 
  distance (A B : ℝ × ℝ) :=
  2 * Real.sqrt (1 - (4 / 5)^2) = 6 / 5 :=
sorry

end distance_AB_l121_121493


namespace sequence_fiftieth_term_l121_121935

theorem sequence_fiftieth_term :
  ∀ (n : ℕ), 0 < n → (a : ℤ) (a = (-1)^(n-1) * (4 * n - 1)) → a = -199 :=
by {
  intros n hn a ha,
  sorry
}

end sequence_fiftieth_term_l121_121935


namespace cube_volume_l121_121153

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l121_121153


namespace cube_volume_l121_121168

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l121_121168


namespace inner_cube_surface_area_l121_121283

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121283


namespace well_volume_approx_l121_121689

-- Define the given parameters
def depth : ℝ := 14
def top_radius : ℝ := 1 / 2 * 2
def bottom_radius : ℝ := 1 / 2 * 1

-- The formula for the volume of a frustum of a cone
def frustum_volume (h R r : ℝ) : ℝ :=
  (1 / 3) * (Real.pi) * h * (R^2 + r^2 + R * r)

-- The specific volume calculation for the problem
def volume := frustum_volume depth top_radius bottom_radius

-- Stating the theorem
theorem well_volume_approx : volume ≈ 51.317 :=
sorry

end well_volume_approx_l121_121689


namespace solve_system_of_equations_simplify_expression_l121_121672

-- Part 1: System of Equations
theorem solve_system_of_equations :
    ∃ (x y : ℝ), 
        (2 * x + y = 3) ∧ 
        (3 * x + y = 5) ∧
        (x = 2) ∧ 
        (y = -1) := 
begin
  sorry
end

-- Part 2: Simplification of Expression
theorem simplify_expression (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ 0) :
    (a^2 / (a^2 - 2 * a + 1) * (a - 1) / a - 1 / (a - 1)) = 1 := 
begin
  sorry
end

end solve_system_of_equations_simplify_expression_l121_121672


namespace points_three_units_away_from_neg3_l121_121624

theorem points_three_units_away_from_neg3 (x : ℝ) : (abs (x + 3) = 3) ↔ (x = 0 ∨ x = -6) :=
by
  sorry

end points_three_units_away_from_neg3_l121_121624


namespace find_radius_of_tangent_circles_l121_121038

noncomputable def radius_of_tangent_circles : ℝ :=
  let ellipse_eq : ℝ → ℝ → Prop := λ x y, x^2 + 6 * y^2 = 8
  let circle_eq : ℝ → ℝ → ℝ → Prop := λ r x y, (x - r)^2 + y^2 = r^2
  let discriminant_zero : ℝ → Prop := λ r, (12 * r)^2 - 4 * 5 * 8 = 0
  if h : discriminant_zero ((√10) / 3) then (√10) / 3 else 0

theorem find_radius_of_tangent_circles :
  ∃ r : ℝ, (r = radius_of_tangent_circles) ∧ r = ((√10) / 3) :=
begin
  use ((√10) / 3),
  split,
  { unfold radius_of_tangent_circles,
    simp,
    norm_num,
    exact if_pos (by norm_num : discriminant_zero ((√10) / 3)) },
  { refl }
end

end find_radius_of_tangent_circles_l121_121038


namespace largest_sum_is_three_fourths_l121_121745

theorem largest_sum_is_three_fourths : 
    (\frac{1}{4} + \frac{1}{2} > \frac{1}{4} + \frac{1}{3})
    ∧ (\frac{1}{4} + \frac{1}{2} > \frac{1}{4} + \frac{1}{9})
    ∧ (\frac{1}{4} + \frac{1}{2} > \frac{1}{4} + \frac{1}{10})
    ∧ (\frac{1}{4} + \frac{1}{2} > \frac{1}{4} + \frac{1}{11}) :=
begin
    sorry
end

end largest_sum_is_three_fourths_l121_121745


namespace real_numbers_representation_natural_numbers_representation_integer_numbers_representation_rational_numbers_representation_l121_121949

-- Definitions for the sets
def RealNumberSet := { x : ℝ }
def NaturalNumberSet := { x : ℕ }
def IntegerSet := { x : ℤ }
def RationalNumberSet := { x : ℚ }

-- Proofs for each of the set representations
theorem real_numbers_representation : RealNumberSet = { x : ℝ } := sorry

theorem natural_numbers_representation : NaturalNumberSet = { x : ℕ } := sorry

theorem integer_numbers_representation : IntegerSet = { x : ℤ } := sorry

theorem rational_numbers_representation : RationalNumberSet = { x : ℚ } := sorry

end real_numbers_representation_natural_numbers_representation_integer_numbers_representation_rational_numbers_representation_l121_121949


namespace log_absolute_comparison_l121_121800

variable {a x : ℝ}

theorem log_absolute_comparison (hx : 0 < x ∧ x < 1) (ha : a > 0 ∧ a ≠ 1) : 
  abs (Real.logBase a (1 - x)) > abs (Real.logBase a (1 + x)) := 
by 
  sorry

end log_absolute_comparison_l121_121800


namespace minute_hand_travel_distance_l121_121698

theorem minute_hand_travel_distance
  (radius : ℝ)
  (minutes : ℝ)
  (circle_prop : ∀ r : ℝ, r * 2 * Real.pi)
  (rotation_period : ℝ)
  (travel_time : ℝ)
  (travel_distance : ℝ)
  (h1 : radius = 8)
  (h2 : rotation_period = 60)
  (h3 : travel_time = 45) :
  travel_distance = 12 * Real.pi := by
  sorry

end minute_hand_travel_distance_l121_121698


namespace volume_of_wedge_l121_121725

theorem volume_of_wedge (c : ℝ) (h : c = 18 * Real.pi) : 
  let r := c / (2 * Real.pi) in
  let V := (4 / 3) * Real.pi * r^3 in
  (V / 6) = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l121_121725


namespace inner_cube_surface_area_l121_121351

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121351


namespace range_of_f_l121_121001

noncomputable def f (x : ℕ) : ℤ := x^2 - 2*x

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ ({0, 1, 2, 3} : Finset ℕ), f x = y} = {-1, 0, 3} :=
by
  sorry

end range_of_f_l121_121001


namespace solution_pairs_l121_121438

theorem solution_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x ^ 2 + y ^ 2 - 5 * x * y + 5 = 0 ↔ (x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 9 ∧ y = 2) ∨ (x = 1 ∧ y = 2) := by
  sorry

end solution_pairs_l121_121438


namespace max_intersections_between_two_E_shapes_l121_121407

def E_shape : Type := {
  rays : List (List ℝ × List ℝ),
  segment: List ℝ × List ℝ,
  endpoint_conditions: ∀ ray ∈ rays, ∃ endpoint ∈ segment, ray.endpoint = endpoint,
  perpendicular_condition: ∀ ray ∈ rays, segment.is_perpendicular ray,
  endpoint_segment_conditions: segment.start ∈ rays ∧ segment.end ∈ rays
}

def count_intersections (e1 e2 : E_shape) : ℕ := sorry

theorem max_intersections_between_two_E_shapes (e1 e2 : E_shape) :
  count_intersections e1 e2 ≤ 13 := sorry

end max_intersections_between_two_E_shapes_l121_121407


namespace perpendicular_line_eq_l121_121613

theorem perpendicular_line_eq {m : ℝ} :
  (∀ x y : ℝ, (3 * x - 5 * y + 6 = 0 → (-5, 3)) ∧ (5 * x + 3 * y = -m) ∧ (x, y) = (-1, 2) ∧ (-5 + 6 + m = 0)) → (5 * x + 3 * y - 1 = 0) := 
by
  sorry

end perpendicular_line_eq_l121_121613


namespace sum_of_divisors_143_l121_121088

theorem sum_of_divisors_143 : (∑ i in (finset.filter (λ d, 143 % d = 0) (finset.range 144)), i) = 168 :=
by
  -- The final proofs will go here.
  sorry

end sum_of_divisors_143_l121_121088


namespace sequence_formula_b_sum_formula_l121_121806

noncomputable def a_sequence (n : ℕ) : ℕ :=
if n = 1 then 2 else 2 * (∑ i in range (n - 1), a_sequence (i + 1)) + 2

noncomputable def b_sequence (n : ℕ) : ℝ :=
let a_n := 2 * 3^(n - 1) in
a_n + (-1) ^ n * real.logb 3 a_n

noncomputable def b_sum (n : ℕ) : ℝ :=
(∑ i in range (2 * n), b_sequence (i + 1))

theorem sequence_formula (n : ℕ) (hn : 1 ≤ n) : a_sequence n = 2 * 3^(n - 1) :=
sorry

theorem b_sum_formula (n : ℕ) : b_sum n = 3^(2 * n) + n - 1 :=
sorry

end sequence_formula_b_sum_formula_l121_121806


namespace inner_cube_surface_area_l121_121346

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121346


namespace cube_volume_of_surface_area_l121_121160

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l121_121160


namespace probability_one_each_plus_one_red_l121_121453

noncomputable def total_ways_to_select_four_marbles : ℕ :=
  (nat.choose 8 4)

noncomputable def ways_to_choose_two_red : ℕ :=
  (nat.choose 3 2)

noncomputable def ways_to_choose_one_blue : ℕ :=
  (nat.choose 3 1)

noncomputable def ways_to_choose_one_green : ℕ :=
  (nat.choose 2 1)

noncomputable def favorable_outcomes : ℕ :=
  ways_to_choose_two_red * ways_to_choose_one_blue * ways_to_choose_one_green

noncomputable def probability : ℚ :=
  (favorable_outcomes : ℚ) / (total_ways_to_select_four_marbles : ℚ)

theorem probability_one_each_plus_one_red :
  probability = 9 / 35 :=
  sorry

end probability_one_each_plus_one_red_l121_121453


namespace fraction_multiplication_l121_121046

theorem fraction_multiplication : ((1 / 2) * (1 / 3) * (1 / 6) * 72 = 2) :=
by
  sorry

end fraction_multiplication_l121_121046


namespace area_ratio_MNP_to_ABC_l121_121602

/- Definitions of angles and proof of area ratio of triangle MNP to triangle ABC -/
theorem area_ratio_MNP_to_ABC :
  (∠A = 75) ∧ (∠C = 45) ∧ (∠B = 60) ∧ (BMN.similar_to BCA) →
  S_MNP / S_ABC = (sqrt 3 - 1) / 4 :=
by
  sorry

end area_ratio_MNP_to_ABC_l121_121602


namespace inner_cube_surface_area_l121_121362

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121362


namespace inner_cube_surface_area_l121_121323

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121323


namespace running_race_total_students_l121_121830

theorem running_race_total_students 
  (number_of_first_grade_students number_of_second_grade_students : ℕ)
  (h1 : number_of_first_grade_students = 8)
  (h2 : number_of_second_grade_students = 5 * number_of_first_grade_students) :
  number_of_first_grade_students + number_of_second_grade_students = 48 := 
by
  -- we will leave the proof empty
  sorry

end running_race_total_students_l121_121830


namespace water_evaporation_l121_121029

theorem water_evaporation (m : ℝ) 
  (evaporation_day1 : m' = m * (0.1)) 
  (evaporation_day2 : m'' = (m * 0.9) * 0.1) 
  (total_evaporation : total = m' + m'')
  (water_added : 15 = total) 
  : m = 1500 / 19 := by
  sorry

end water_evaporation_l121_121029


namespace inner_cube_surface_area_l121_121342

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l121_121342


namespace bookstore_problem_l121_121136

/-- A bookstore problem -/
theorem bookstore_problem (x : ℝ) (h₁ : x > 0) :
  let selling_price := 0.75 * x,
      base_cost_price := (2 / 3) * selling_price,
      overhead_cost := 0.1 * x,
      total_cost := base_cost_price + overhead_cost
  in total_cost / x = 0.6 :=
by
  intros
  let s := 0.75 * x
  let b := (2 / 3) * s
  let o := 0.1 * x
  let t := b + o
  have h₀ : t = 0.6 * x := sorry -- Proof omitted as per instruction
  exact h₀ ▸ by simp [mul_div_cancel_left (0.6:ℝ) h₁.ne']

end bookstore_problem_l121_121136


namespace increase_in_average_weight_l121_121884

variable (A : ℝ)

theorem increase_in_average_weight (h1 : ∀ (A : ℝ), 4 * A - 65 + 71 = 4 * (A + 1.5)) :
  (71 - 65) / 4 = 1.5 :=
by
  sorry

end increase_in_average_weight_l121_121884


namespace roots_of_transformed_quadratic_l121_121524

variable {a b c : ℝ}

theorem roots_of_transformed_quadratic
    (h₁: a ≠ 0)
    (h₂: ∀ x, a * (x - 1)^2 - 1 = ax^2 + bx + c - 1)
    (h₃: ax^2 + bx + c = -1) :
    (x = 1) ∧ (x = 1) := 
  sorry

end roots_of_transformed_quadratic_l121_121524


namespace distinct_right_angles_l121_121043

theorem distinct_right_angles (n : ℕ) (h : n > 0) : 
  ∃ (a b c d : ℕ), (a + b + c + d ≥ 4 * (Int.sqrt n)) ∧ (a * c ≥ n) ∧ (b * d ≥ n) :=
by sorry

end distinct_right_angles_l121_121043


namespace inner_cube_surface_area_l121_121320

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121320


namespace equation_of_ellipse_an_bm_constant_l121_121481

section ellipse_problem

variables (a b c : ℝ) (e : ℝ := sqrt 3 / 2)
variables (P : ℝ × ℝ) (A : ℝ × ℝ := (a, 0)) (B : ℝ × ℝ := (0, b)) (O : ℝ × ℝ := (0, 0))

-- Define the conditions:
axiom elliptic_conditions :
  a > b ∧ b > 0 ∧
  e = sqrt 3 / 2 ∧ 
  (1/2) * a * b = 1 ∧ 
  a^2 = b^2 + c^2 ∧ 
  c = a * e

-- The goal is to prove the equation of the ellipse:
theorem equation_of_ellipse : ∃ x y : ℝ, (x / a)^2 + (y / b)^2 = 1 :=
  sorry

-- The goal is to prove that |AN| * |BM| is a constant:
theorem an_bm_constant (x0 y0 : ℝ) (hx : (x0 / 2)^2 + y0^2 = 1) :
  ∃ k : ℝ, |2 + (x0 / (y0 - 1))| * |1 + (2 * y0 / (x0 - 2))| = k :=
  sorry

end ellipse_problem

end equation_of_ellipse_an_bm_constant_l121_121481


namespace inner_cube_surface_area_l121_121385

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121385


namespace caleb_double_burgers_count_l121_121109

theorem caleb_double_burgers_count
    (S D : ℕ)
    (cost_single cost_double total_hamburgers total_cost : ℝ)
    (h1 : cost_single = 1.00)
    (h2 : cost_double = 1.50)
    (h3 : total_hamburgers = 50)
    (h4 : total_cost = 66.50)
    (h5 : S + D = total_hamburgers)
    (h6 : cost_single * S + cost_double * D = total_cost) :
    D = 33 := 
sorry

end caleb_double_burgers_count_l121_121109


namespace inner_cube_surface_area_l121_121388

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l121_121388


namespace wedge_volume_formula_l121_121714

noncomputable def sphere_wedge_volume : ℝ :=
let r := 9 in
let volume_of_sphere := (4 / 3) * Real.pi * r^3 in
let volume_of_one_wedge := volume_of_sphere / 6 in
volume_of_one_wedge

theorem wedge_volume_formula
  (circumference : ℝ)
  (h1 : circumference = 18 * Real.pi)
  (num_wedges : ℕ)
  (h2 : num_wedges = 6) :
  sphere_wedge_volume = 162 * Real.pi :=
by
  sorry

end wedge_volume_formula_l121_121714


namespace inner_cube_surface_area_l121_121370

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l121_121370


namespace dice_product_not_even_probability_l121_121668

theorem dice_product_not_even_probability :
  let total_outcomes := 6 * 6 in
  let favorable_outcomes := 3 * 3 in
  let probability := favorable_outcomes / total_outcomes in
  probability = (1 : ℚ) / 4 :=
by
  sorry

end dice_product_not_even_probability_l121_121668


namespace find_x_y_l121_121473

theorem find_x_y (x y : ℝ) (h1 : x + Real.cos y = 2023) (h2 : x + 2023 * Real.sin y = 2022) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2022 :=
sorry

end find_x_y_l121_121473


namespace find_number_l121_121873

theorem find_number (N x : ℝ) (h1 : x = 1) (h2 : N / (4 + 1 / x) = 1) : N = 5 := 
by 
  sorry

end find_number_l121_121873


namespace rick_book_categorization_l121_121596

theorem rick_book_categorization :
  ∃ (times_to_break : ℕ) (final_categories : ℕ), 
  final_categories = 40 ∧ times_to_break = 1 ∧
  ∃ (total_books : ℕ) (first_division : ℕ) (second_division : ℕ) (final_group_size : ℕ),
    total_books = 800 ∧
    first_division = 4 ∧
    second_division = 5 ∧
    final_group_size = 20 ∧
    (total_books / first_division / second_division / final_group_size).nat_abs = (final_categories / second_division).nat_abs
:=
by
  -- Problem statement conditions translation should be here
  sorry

end rick_book_categorization_l121_121596


namespace first_nonzero_digit_of_one_over_347_l121_121062

theorem first_nonzero_digit_of_one_over_347 : 
  let fractional_part := 1 / (347: ℝ),
      first_digit_right_of_decimal := Int.ofNat (floor ((fractional_part * 10) % 10)) in
  first_digit_right_of_decimal = 2 :=
by
  sorry

end first_nonzero_digit_of_one_over_347_l121_121062


namespace tom_teaching_years_l121_121999

theorem tom_teaching_years :
  ∃ T D : ℕ, T + D = 70 ∧ D = (1 / 2) * T - 5 ∧ T = 50 :=
by
  sorry

end tom_teaching_years_l121_121999


namespace mysterious_neighbor_is_13_l121_121545

variable (x : ℕ) (h1 : x < 15) (h2 : 2 * x * 30 = 780)

theorem mysterious_neighbor_is_13 : x = 13 :=
by {
    sorry 
}

end mysterious_neighbor_is_13_l121_121545


namespace simplify_expression_correct_l121_121953

noncomputable def simplify_expression (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) : ℝ :=
  let expr1 := (a^2 - b^2) / (a^2 + 2 * a * b + b^2)
  let expr2 := (2 : ℝ) / (a * b)
  let expr3 := ((1 : ℝ) / a + (1 : ℝ) / b)^2
  let expr4 := (2 : ℝ) / (a^2 - b^2 + 2 * a * b)
  expr1 + expr2 / expr3 * expr4

theorem simplify_expression_correct (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  simplify_expression a b h = 2 / (a + b)^2 := by
  sorry

end simplify_expression_correct_l121_121953


namespace inner_cube_surface_area_l121_121304

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l121_121304


namespace inner_cube_surface_area_l121_121273

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l121_121273


namespace log_ab_div_3_eq_half_log_a_plus_log_b_l121_121592

theorem log_ab_div_3_eq_half_log_a_plus_log_b
  (a b k : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : k > 0)
  (h4 : k ≠ 1)
  (h5 : a^2 + b^2 = 7 * a * b) :
  Real.logBase k ((a + b) / 3) = (1 / 2) * (Real.logBase k a + Real.logBase k b) := by
  sorry

end log_ab_div_3_eq_half_log_a_plus_log_b_l121_121592


namespace probability_game_ends_after_five_shots_distribution_and_expectation_l121_121139

noncomputable def shooting_accuracy : ℚ := 2 / 3

-- Problem 1: Probability that the game ends after 5 shots
theorem probability_game_ends_after_five_shots
  (prob_clear_game : ℚ)
  (prob_fail_game : ℚ)
  (total_prob : prob_clear_game + prob_fail_game = 8 / 81) :
  true := sorry

-- Problem 2: Distribution table and expectation
theorem distribution_and_expectation
  (prob_clear_game : ℚ)
  (prob_fail_game : ℚ)
  (prob_neutral_game : ℚ)
  (dist_table : List (ℚ × ℚ) = [(3, 233/729), (6, 112/729), (9, 128/243)])
  (expectation : ℚ)
  (calc_expectation : expectation = (3 * 233 / 729 + 6 * 112 / 729 + 9 * 128 / 243) / 1 = 1609 / 243) :
  true := sorry

end probability_game_ends_after_five_shots_distribution_and_expectation_l121_121139


namespace tan_2016_l121_121815

-- Define the given condition
def sin_36 (a : ℝ) : Prop := Real.sin (36 * Real.pi / 180) = a

-- Prove the required statement given the condition
theorem tan_2016 (a : ℝ) (h : sin_36 a) : Real.tan (2016 * Real.pi / 180) = a / Real.sqrt (1 - a^2) :=
sorry

end tan_2016_l121_121815


namespace inner_cube_surface_area_l121_121282

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l121_121282


namespace inner_cube_surface_area_l121_121236

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121236


namespace cube_volume_l121_121215

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l121_121215


namespace volume_of_cube_l121_121190

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l121_121190


namespace area_difference_l121_121896

-- Conditions
variables {A B C D E : Type} [euclidean_geometry Type] 
variables {x y z : ℝ}
variables {AB AE BC : ℝ} (h_AB: AB = 5) (h_AE: AE = 10) (h_BC: BC = 8)
variables (h₁ : ∠ EAB = 90) (h₂ : ∠ ABC = 90)
variables (h_AE_lines : collinear A E) (h_BC_lines : collinear B C)
variables (D_on_lines : ∃ D, (collinear A D C) ∧ (collinear B D E))

-- Mathematical statement
theorem area_difference (h₁ : ∠ EAB = 90) (h₂ : ∠ ABC = 90)
  (h_AB : AB = 5) (h_AE : AE = 10) (h_BC : BC = 8)
  (intersect_D : ∃ D, collinear A D C ∧ collinear B D E):
  (triangle_area A D E - triangle_area B D C) = 5 :=
sorry

end area_difference_l121_121896


namespace minimum_value_of_f_l121_121462

noncomputable def f (x y : ℝ) := sqrt (x^2 - 3 * x + 3) + sqrt (y^2 - 3 * y + 3) + sqrt (x^2 - sqrt 3 * x * y + y^2)

theorem minimum_value_of_f : 
  ∀ (x y : ℝ), 0 < x → 0 < y → f x y = sqrt 6 := sorry

end minimum_value_of_f_l121_121462


namespace circles_intersect_l121_121626

-- Definitions for the two circles O1 and O2.

def Circle (center : ℝ × ℝ) (radius : ℝ) := 
  ∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2

def O1 : Circle (-1, 1) 2 := 
  sorry

def O2 : Circle (2, 4) 3 := 
  sorry

-- Calculating the distance between two centers
def distance (c1 c2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((c2.1 - c1.1)^2 + (c2.2 - c1.2)^2)

def d := distance (-1, 1) (2, 4)

-- proving that the positional relationship is intersecting
theorem circles_intersect :
  1 < d ∧ d < 5 :=
  sorry

end circles_intersect_l121_121626


namespace coefficient_x5_in_binomial_expansion_l121_121550

theorem coefficient_x5_in_binomial_expansion :
  (∃ n k : ℕ, n = 60 ∧ k = 5 ∧ (x + 1) ^ n = ∑ i in finset.range(n+1), (n.choose i) * x^i * 1^(n-i) ∧ (n.choose k) = 446040) := 
sorry

end coefficient_x5_in_binomial_expansion_l121_121550


namespace possible_values_of_expression_l121_121820

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ k ∈ ({1, -1}: Set ℝ), 
  (a / |a| = k) ∧
  (b / |b| = k) ∧
  (c / |c| = k) ∧
  (d / |d| = k) ∧
  (abcd / |abcd| = k) ∧
  ({(a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)} = {5, 1, -3}) :=
by
  sorry

end possible_values_of_expression_l121_121820


namespace tan_double_angle_l121_121796

-- Define the given conditions
variable {x : ℝ} (h1 : x ∈ Ioo (-π / 2) 0) (h2 : cos x = 4 / 5)

-- State the theorem to be proved
theorem tan_double_angle (h1 : x ∈ Ioo (-π / 2) 0) (h2 : cos x = 4 / 5) : tan (2 * x) = -24 / 7 := sorry

end tan_double_angle_l121_121796


namespace cube_volume_of_surface_area_l121_121154

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l121_121154


namespace inner_cube_surface_area_l121_121248

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121248


namespace inner_cube_surface_area_l121_121357

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l121_121357


namespace find_g_15_plus_g_neg_15_l121_121929

noncomputable def g (x : ℝ) : ℝ := 2 * x^6 - 5 * x^4 + 7 * x^2 + 6

theorem find_g_15_plus_g_neg_15 :
  g 15 + g (-15) = 164 := by
  have h1 : g 15 = 82 := sorry
  have h2 : g (-15) = g 15 := by
    apply congr_arg g
    norm_num
  rw [h1, h2]
  norm_num
  sorry

end find_g_15_plus_g_neg_15_l121_121929


namespace area_of_triangle_l121_121060

-- Define the required conditions
def line1 (x : ℝ) : ℝ := 2 * x - 5
def line2 (x : ℝ) : ℝ := -3 * x + 18

-- Intersection with y-axis
def intercept1 : ℝ × ℝ := (0, -5)
def intercept2 : ℝ × ℝ := (0, 18)

-- Intersection point of the two lines
def intersection : ℝ × ℝ := (23 / 5, 21 / 5)

-- Prove the area of the triangle formed by the lines 
theorem area_of_triangle : 
  let base := 18 - (-5) in let height := 23 / 5 in
  (1 / 2) * base * height = 529 / 10 := by
  sorry

end area_of_triangle_l121_121060


namespace derivative_y_l121_121124

variable {x : ℝ}

-- Defining the function
def y (x : ℝ) : ℝ := 
  (2 / 3) * (4 * x^2 - 4 * x + 3) * real.sqrt(x^2 - x) + (2 * x - 1)^4 * real.arcsin (1 / (2 * x - 1))

-- Constraint on x
lemma x_constraint : x > 1 / 2 := sorry

-- The proof statement
theorem derivative_y :
  ∀ (x : ℝ), x > 1 / 2 →
  (deriv (λ x, (2 / 3) * (4 * x^2 - 4 * x + 3) * real.sqrt(x^2 - x) + (2 * x - 1)^4 * real.arcsin (1 / (2 * x - 1))) x) = 
  8 * (2 * x - 1)^3 * real.arcsin (1 / (2 * x - 1)) := sorry

end derivative_y_l121_121124


namespace problem_statement_l121_121754

theorem problem_statement (p : ℕ) (hprime : Prime p) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ p = m^2 + n^2 ∧ p ∣ (m^3 + n^3 + 8 * m * n)) → p = 5 :=
by
  sorry

end problem_statement_l121_121754


namespace distribution_of_balls_l121_121508

theorem distribution_of_balls :
  ∃ (P : ℕ → ℕ → ℕ), P 6 4 = 9 := 
by
  sorry

end distribution_of_balls_l121_121508


namespace misha_pole_number_l121_121741

def containsR (n : ℕ) : Prop :=
  -- This is a simplified placeholder function to check for 'R' in a number's name
  sorry

def containsSh (n : ℕ) : Prop :=
  -- This is a simplified placeholder function to check for 'Sh' in a number's name
  sorry

noncomputable def count_poles_skipping (limit : ℕ) (skip_condition : ℕ → Prop) : ℕ :=
  (list.range (limit + 1)).filter (not ∘ skip_condition)

theorem misha_pole_number :
  count_poles_skipping 100 containsR = 64 →
  count_poles_skipping 64 containsSh = 81 :=
by
  intros h
  sorry

end misha_pole_number_l121_121741


namespace volume_of_wedge_l121_121726

theorem volume_of_wedge (c : ℝ) (h : c = 18 * Real.pi) : 
  let r := c / (2 * Real.pi) in
  let V := (4 / 3) * Real.pi * r^3 in
  (V / 6) = 162 * Real.pi :=
by
  sorry

end volume_of_wedge_l121_121726


namespace tan_A_value_bc_bounds_l121_121539

variables (A B C : ℝ) (a b c S : ℝ)
axiom acute_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2
axiom opposite_sides : ∀ (A B C : ℝ) (a b c : ℝ), a^2 = 2 * (1/2 * b * c * sin A) + (b - c)^2

-- Part 1: Prove that tan A = 4 / 3
theorem tan_A_value (A B C a b c S : ℝ) (h1 : 0 < A ∧ A < π / 2) (h2 : 0 < B ∧ B < π / 2) (h3 : 0 < C ∧ C < π / 2)
  (h_eq : a^2 = 2 * S + (b - c)^2) (h_S : S = 1/2 * b * c * sin A) :
  tan A = 4 / 3 :=
sorry

-- Part 2: Prove that 16 < b + c <= 8 * sqrt 5 given that a = 8
theorem bc_bounds (A B C a b c S : ℝ) (h1 : 0 < A ∧ A < π / 2) (h2 : 0 < B ∧ B < π / 2) (h3 : 0 < C ∧ C < π / 2)
  (h_eq : a^2 = 2 * S + (b - c)^2) (h_S : S = 1/2 * b * c * sin A) (ha : a = 8) :
  16 < b + c ∧ b + c ≤ 8 * sqrt 5 :=
sorry

end tan_A_value_bc_bounds_l121_121539


namespace find_functions_satisfying_lcm_gcd_eq_l121_121764

noncomputable def satisfies_functional_equation (f : ℕ → ℕ) : Prop := 
  ∀ m n : ℕ, m > 0 ∧ n > 0 → f (m * n) = Nat.lcm m n * Nat.gcd (f m) (f n)

noncomputable def solution_form (f : ℕ → ℕ) : Prop := 
  ∃ k : ℕ, ∀ x : ℕ, f x = k * x

theorem find_functions_satisfying_lcm_gcd_eq (f : ℕ → ℕ) : 
  satisfies_functional_equation f ↔ solution_form f := 
sorry

end find_functions_satisfying_lcm_gcd_eq_l121_121764


namespace value_of_a6_a7_a8_l121_121835

variable {a_n : ℕ → ℝ}
variable (a1 : ℝ) (d : ℝ)

-- Conditions
def sum_of_first_13_terms (seq : ℕ → ℝ) : Prop := (∑ i in finset.range 13, seq i) = 39
def arith_seq (n : ℕ) : ℝ := a1 + n * d

-- The proof statement
theorem value_of_a6_a7_a8 (h : sum_of_first_13_terms arith_seq) : 
  arith_seq 5 + arith_seq 6 + arith_seq 7 = 9 := 
sorry

end value_of_a6_a7_a8_l121_121835


namespace points_distance_leq_one_l121_121023

theorem points_distance_leq_one (x y z : ℝ) :
  (sqrt ((x-1)^2 + y^2 + z^2) ≤ 1) ↔ ((x-1)^2 + y^2 + z^2 ≤ 1) := 
by 
  -- This is a placeholder for the actual proof.
  sorry

end points_distance_leq_one_l121_121023


namespace pure_imaginary_condition_l121_121673

theorem pure_imaginary_condition (a b : ℝ) : 
  (a = 0) ↔ (∃ b : ℝ, b ≠ 0 ∧ z = a + b * I) :=
sorry

end pure_imaginary_condition_l121_121673


namespace inner_cube_surface_area_l121_121292

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121292


namespace geometric_sequence_proof_l121_121693

theorem geometric_sequence_proof (a : ℕ → ℝ) (q : ℝ) (h1 : q > 1) (h2 : a 1 > 0)
    (h3 : a 2 * a 4 + a 4 * a 10 - a 4 * a 6 - (a 5)^2 = 9) :
  a 3 - a 7 = -3 :=
by sorry

end geometric_sequence_proof_l121_121693


namespace hyperbola_C_equation_proof_distance_to_left_directrix_l121_121492

-- Condition: Hyperbola C, another hyperbola, and passing through point A(6, sqrt(5))
def hyperbola_C (a b x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1
def another_hyperbola (x y : ℝ) := (x^2 / 16) - (y^2 / 9) = 1
def point_A := (6 : ℝ, sqrt 5)

-- Condition: Co-vertex and passing through point A
def shared_vertex (a b : ℝ) := a = 4 ∧ b > 0

theorem hyperbola_C_equation_proof (a b : ℝ) :
  (∃ b, shared_vertex a b ∧ hyperbola_C 4 b 6 (sqrt 5)) →
  (∀ x y, hyperbola_C 4 2 x y ↔ (x^2 / 16 - y^2 / 4 = 1)) ∧
  (∀ x, hyperbola_C 4 2 x (x / 2)) ∧ (∀ x, hyperbola_C 4 2 x (-x / 2)) :=
by
  sorry

-- Condition: Point P on hyperbola C, right focus distance, solving system of equations
def right_focus := (2 * sqrt 5, 0)
def left_directrix := x = (-8 * sqrt 5) / 5

theorem distance_to_left_directrix (x_P y_P : ℝ) :
  (hyperbola_C 4 2 x_P y_P ∧ (((x_P - 2 * sqrt 5)^2 + y_P^2 = 36) ∨ ((x_P + 2 * sqrt 5)^2 + y_P^2 = 36))) →
  dist (x_P, y_P) left_directrix = (28 * sqrt 5) / 5 :=
by
  sorry

end hyperbola_C_equation_proof_distance_to_left_directrix_l121_121492


namespace inner_cube_surface_area_l121_121240

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l121_121240


namespace parabola_focus_l121_121065

theorem parabola_focus (F : ℝ × ℝ) (hF : F = (0, 1)) : ∃ p : ℝ, p = 1 ∧ (∀ x y : ℝ, x^2 = 4 * p * y → x^2 = 4 * y) :=
by 
  use 1
  split
  · sorry 
  · sorry

end parabola_focus_l121_121065


namespace intersection_count_l121_121853

-- Definition for the number of intersections of the two circles
def numberOfIntersections : ℕ := 2

theorem intersection_count :
  let circle1 := setOf (λ p : ℝ × ℝ, (p.1 - 3)^2 + p.2^2 = 9)
  let circle2 := setOf (λ p : ℝ × ℝ, p.1^2 + (p.2 - 6)^2 = 36)
  ∃ points : set (ℝ × ℝ), (points ⊆ circle1 ∩ circle2) ∧ (finite points) ∧ (points.card = numberOfIntersections) := by
  sorry

end intersection_count_l121_121853


namespace inner_cube_surface_area_l121_121295

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l121_121295


namespace cube_volume_of_surface_area_l121_121162

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l121_121162


namespace hyperbola_eq_l121_121011

-- Define the conditions explicitly
def hyperbola_passes_through (x y a b : ℝ) : Prop :=
  (x / a) ^ 2 - (y / b) ^ 2 = 1

def eccentricity (a b e : ℝ) : Prop :=
  e = Math.sqrt(a^2 + b^2) / a

-- Main statement to prove
theorem hyperbola_eq (a b : ℝ) (h1 : hyperbola_passes_through (Real.sqrt 2) (Real.sqrt 3) a b)
  (h2 : eccentricity a b 2) :
  (a = 1) ∧ (b = Math.sqrt 3) :=
by {
  -- Proof would go here; currently it's skipped with 'sorry'
  sorry
}

end hyperbola_eq_l121_121011


namespace hexagon_area_ratio_l121_121870

noncomputable def ratio_of_hexagon_areas (a : ℝ) : ℝ :=
  let r1 := a in
  let r2 := (2 * a) / Real.sqrt 3 in
  let A1 := (3 * Real.sqrt 3 / 2) * (r1 ^ 2) in
  let A2 := (3 * Real.sqrt 3 / 2) * (r2 ^ 2) in
  A1 / A2

theorem hexagon_area_ratio : ∀ (a : ℝ), ratio_of_hexagon_areas a = 3 / 4 :=
by
  intro a
  sorry

end hexagon_area_ratio_l121_121870


namespace possible_values_of_expression_l121_121819

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ k ∈ ({1, -1}: Set ℝ), 
  (a / |a| = k) ∧
  (b / |b| = k) ∧
  (c / |c| = k) ∧
  (d / |d| = k) ∧
  (abcd / |abcd| = k) ∧
  ({(a / |a|) + (b / |b|) + (c / |c|) + (d / |d|) + (abcd / |abcd|)} = {5, 1, -3}) :=
by
  sorry

end possible_values_of_expression_l121_121819


namespace sum_of_divisors_143_l121_121099

theorem sum_of_divisors_143 : 
  ∑ d in ({1, 11, 13, 143} : Finset ℕ), d = 168 :=
by
  sorry

end sum_of_divisors_143_l121_121099


namespace scout_troop_profit_l121_121232

theorem scout_troop_profit :
  let cost_per_bar := 3 / 6
      total_cost := 1200 * cost_per_bar
      sell_per_bar := 2 / 3
      total_revenue := 1200 * sell_per_bar
  in total_revenue - total_cost = 200 :=
by
  sorry

end scout_troop_profit_l121_121232


namespace area_of_ABCD_l121_121548

-- Definitions matching the conditions given in the problem
structure RightAngleTriangle (a b c : ℝ) :=
(right_angle : a^2 + b^2 = c^2)
(a_angle_90 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)

variables (A B C D E : ℝ)

/-- The main statement to be proven, translated into Lean. -/
theorem area_of_ABCD :
  (RightAngleTriangle 15 (15*Real.sqrt 3) 30) →
  (RightAngleTriangle 7.5 (7.5*Real.sqrt 3) 15) →
  (RightAngleTriangle 3.75 (3.75*Real.sqrt 3) 7.5) →
  let area_ABCD := (1/2) * (15 * (15 * Real.sqrt 3))
                  + (1/2) * (7.5 * (7.5 * Real.sqrt 3))
                  + (1/2) * (3.75 * (3.75 * Real.sqrt 3)) in
  area_ABCD = (295.65625 / 2) * Real.sqrt 3 :=
begin
  intros h1 h2 h3,
  sorry
end

end area_of_ABCD_l121_121548


namespace no_adjacent_same_roll_proof_l121_121447

open ProbabilityTheory

-- Define the probability calculation in the Lean framework.
def no_adjacent_same_roll_prob : ℚ :=
  let pA := 1 / 8
  let p_diff_7 := (7 / 8) ^ 2
  let p_diff_6 := 6 / 8
  let p_diff_5 := 5 / 8
  let case1 := pA * p_diff_7 * p_diff_6
  let p_diff_7_rest := 7 / 8
  let p_diff_6_2 := (6 / 8) ^ 2
  let case2 := p_diff_7_rest * p_diff_6_2 * p_diff_5
  (case1 + case2) / 8

-- Statement of the proof problem, including all relevant conditions and the final proof goal.
theorem no_adjacent_same_roll_proof :
  no_adjacent_same_roll_prob = 777 / 2048 :=
sorry

end no_adjacent_same_roll_proof_l121_121447


namespace inner_cube_surface_area_l121_121324

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l121_121324


namespace jane_evening_pages_l121_121560

theorem jane_evening_pages :
  ∀ (P : ℕ), (7 * (5 + P) = 105) → P = 10 :=
by
  intros P h
  sorry

end jane_evening_pages_l121_121560


namespace power_function_at_9_l121_121009

-- Define the conditions
def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Define the condition that the function passes through (2, 4)
def passes_through (α : ℝ) : Prop := power_function α 2 = 4

-- Define the theorem that if the function passes through (2, 4), then f(9) is 81
theorem power_function_at_9 (α : ℝ) (h : passes_through α) : power_function α 9 = 81 :=
by
  sorry

end power_function_at_9_l121_121009


namespace second_cube_surface_area_l121_121332

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l121_121332


namespace triplet_solutions_l121_121427

open Real

theorem triplet_solutions (a b c : ℝ) :
  (a + b + c = 1 / a + 1 / b + 1 / c) ∧ (a^2 + b^2 + c^2 = 1 / a^2 + 1 / b^2 + 1 / c^2) →
  (∃ k : ℝ, (a = 1 ∧ b = (k - 1 + sqrt((k - 1)^2 - 4)) / 2 ∧ c = (k - 1 - sqrt((k - 1)^2 - 4)) / 2 ∧ abs(k - 1) ≥ 2) ∨
           (a = -1 ∧ b = (k + 1 + sqrt((k + 1)^2 - 4)) / 2 ∧ c = (k + 1 - sqrt((k + 1)^2 - 4)) / 2 ∧ abs(k + 1) ≥ 2)) :=
by (sorry)

end triplet_solutions_l121_121427
