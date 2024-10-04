import MathLib
import Mathlib
import Mathlib.Algebra.Arithmetic.Sum
import Mathlib.Algebra.Factorial
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupWithZero.Power
import Mathlib.Algebra.Rat
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Trigonometry
import Mathlib.Analysis.Complex.Polynomial
import Mathlib.Analysis.Conics
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Choose
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Probability.ProbabilityMassFunction
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.GroupTheory.Subgroup.Basic
import Mathlib.NumberTheory.Primes
import Mathlib.Probability.Basic
import Mathlib.Probability.Notation
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import data.nat.basic
import mathlib
import tactic

namespace triangle_reflection_PQ_l135_135893

theorem triangle_reflection_PQ (P Q R Q' R' M S T : Point)
  (hPM : median P M Q R)
  (hReflect : reflection P M Q = Q' ∧ reflection P M R = R')
  (hPS : dist P S = 8)
  (hSR : dist S R = 16)
  (hQT : dist Q T = 12) :
  dist P Q = 8 * Real.sqrt 17 := 
sorry

end triangle_reflection_PQ_l135_135893


namespace perimeter_of_triangle_l135_135509

namespace TrianglePerimeter

variable (a b c k : ℝ)
variable (Δ : ℝ)

-- Definition of the quadratic equation and its discriminant
def discriminant (a b c : ℝ) : ℝ := b * b - 4 * a * c
def quadratic_eq (k : ℝ) := (x : ℝ) -> x^2 - (k + 2) * x + 2 * k = 0

-- Conditions
def is_isosceles_traiangle (a b c : ℝ) : Prop := a = b ∨ a = c ∨ b = c

-- Perimeter calculation
def triangle_perimeter (a b c : ℝ) : ℝ := a + b + c

-- Main theorem
theorem perimeter_of_triangle 
  (h_discriminant: Δ = (k - 2)^2) 
  (h_quadratic: ∀ x, quadratic_eq k x) 
  (h_roots: k = 2) 
  (h_isosceles : is_isosceles_traiangle a 2 2) : 
  triangle_perimeter 1 2 2 = 5 :=
sorry

end TrianglePerimeter

end perimeter_of_triangle_l135_135509


namespace cos_alpha_beta_value_l135_135739

noncomputable def cos_alpha_beta (α β : ℝ) : ℝ :=
  Real.cos (α + β)

theorem cos_alpha_beta_value (α β : ℝ)
  (h1 : Real.cos α - Real.cos β = -3/5)
  (h2 : Real.sin α + Real.sin β = 7/4) :
  cos_alpha_beta α β = -569/800 :=
by
  sorry

end cos_alpha_beta_value_l135_135739


namespace local_minimum_f_b_neg4_b_value_range_l135_135833

noncomputable def f (x b : ℝ) : ℝ := 3 * x - 1 / x + b * Real.log x

-- Part 1: Local minimum of f when b = -4
theorem local_minimum_f_b_neg4 : local_min (λ x => f x (-4)) 1 :=
sorry

-- Part 2: Range of b values for the inequality condition
theorem b_value_range (e : ℝ) (h_e : Real.exp 1 = e) : 
  { b : ℝ | ∃ x : ℝ, x ∈ set.Icc 1 e ∧ (4 * x - 1 / x - f x b < -(1 + b) / x) } = 
  set.Ioo (-∞) (-2) ∪ set.Ioo ((e ^ 2 + 1) / (e - 1)) ∞ :=
sorry

end local_minimum_f_b_neg4_b_value_range_l135_135833


namespace number_of_distinct_pairs_l135_135774

theorem number_of_distinct_pairs : 
  ∃ (s : Finset (ℝ × ℝ)), 
  (∀ p ∈ s, (p.1 = p.1 ^ 3 + p.2 ^ 2) ∧ (p.2 = 3 * p.1 * p.2)) ∧ 
  s.card = 5 :=
by
  sorry

end number_of_distinct_pairs_l135_135774


namespace surface_area_of_brick_l135_135087

-- Define the dimensions of the brick
def length := 10
def width := 4
def height := 3

-- Define the areas of the faces
def area_largest_faces := 2 * (length * width)
def area_medium_faces := 2 * (length * height)
def area_smallest_faces := 2 * (width * height)

-- Statement of the surface area theorem
theorem surface_area_of_brick : area_largest_faces + area_medium_faces + area_smallest_faces = 164 := by
  sorry

end surface_area_of_brick_l135_135087


namespace inequality_A_inequality_B_inequality_D_l135_135792

variable (a b : ℝ)

theorem inequality_A (a_gt_0 : a > 0) (b_gt_0 : b > 0) (sum_eq_4 : a + b = 4) : a^2 + b^2 ≥ 8 :=
  sorry

theorem inequality_B (a_gt_0 : a > 0) (b_gt_0 : b > 0) (sum_eq_4 : a + b = 4) : 2^a + 2^b ≥ 8 :=
  sorry

theorem inequality_D (a_gt_0 : a > 0) (b_gt_0 : b > 0) (sum_eq_4 : a + b = 4) : 1 / a + 4 / b ≥ 9 / 4 :=
  sorry

end inequality_A_inequality_B_inequality_D_l135_135792


namespace oranges_taken_l135_135845

theorem oranges_taken (initial_oranges remaining_oranges taken_oranges : ℕ) 
  (h1 : initial_oranges = 60) 
  (h2 : remaining_oranges = 25) 
  (h3 : taken_oranges = initial_oranges - remaining_oranges) : 
  taken_oranges = 35 :=
by
  -- Proof is omitted, as instructed.
  sorry

end oranges_taken_l135_135845


namespace trig_identity_proof_l135_135810

theorem trig_identity_proof (α β : ℝ) 
  (h : (cos α ^ 6 / cos β ^ 3) + (sin α ^ 6 / sin β ^ 3) = 1) : 
  (sin β ^ 6 / sin α ^ 3) + (cos β ^ 6 / cos α ^ 3) = 1 :=
by
  sorry

end trig_identity_proof_l135_135810


namespace binary_digit_one_l135_135110
-- We import the necessary libraries

-- Define the problem and prove the statement as follows
def fractional_part_in_binary (x : ℝ) : ℕ → ℕ := sorry

def sqrt_fractional_binary (k : ℕ) (i : ℕ) : ℕ :=
  fractional_part_in_binary (Real.sqrt ((k : ℝ) * (k + 1))) i

theorem binary_digit_one {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  ∃ i, n + 1 ≤ i ∧ i ≤ 2 * n + 1 ∧ sqrt_fractional_binary k i = 1 :=
sorry

end binary_digit_one_l135_135110


namespace triangle_is_isosceles_l135_135220

/-- Given a triangle ABC with an interior point M such that the angles are specified as follows,
    prove that the triangle is isosceles. -/
theorem triangle_is_isosceles (A B C M : Type*) (angle_MAB angle_MBA angle_MAC angle_MCA : ℝ) :
  angle_MAB = 10 ∧ angle_MBA = 20 ∧ angle_MAC = 40 ∧ angle_MCA = 30 →
  is_isosceles (triangle A B C) :=
by
  sorry

end triangle_is_isosceles_l135_135220


namespace midpoint_product_coordinates_l135_135454

theorem midpoint_product_coordinates :
  let x1 := 8
      y1 := -4
      x2 := -2
      y2 := 10
      midpoint_x := (x1 + x2) / 2
      midpoint_y := (y1 + y2) / 2
  in midpoint_x * midpoint_y = 9 :=
by {
  let x1 := 8
  let y1 := -4
  let x2 := -2
  let y2 := 10
  let midpoint_x := (x1 + x2) / 2
  let midpoint_y := (y1 + y2) / 2
  have h1 : midpoint_x = 3 := by sorry
  have h2 : midpoint_y = 3 := by sorry
  have h3 : 3 * 3 = 9 := by sorry
  exact h3
}

end midpoint_product_coordinates_l135_135454


namespace avg_mark_excluded_students_l135_135633

constant total_students : ℕ := 35
constant avg_mark_entire_class : ℝ := 80
constant excluded_students : ℕ := 5
constant avg_mark_remaining_students : ℝ := 90

theorem avg_mark_excluded_students :
  let total_marks := avg_mark_entire_class * total_students,
      remaining_students := total_students - excluded_students,
      total_marks_remaining := avg_mark_remaining_students * remaining_students,
      total_marks_excluded := total_marks - total_marks_remaining,
      avg_mark_excluded := total_marks_excluded / excluded_students
  in avg_mark_excluded = 20 := sorry

end avg_mark_excluded_students_l135_135633


namespace simple_interest_principal_l135_135647

variable (SI : ℝ) (R : ℝ) (T : ℝ)
variable (P : ℝ)

#check simple_interest_formula

-- Conditions
def conditions : Prop :=
  SI = 400 ∧ R = 25 ∧ T = 2

-- Proof problem
theorem simple_interest_principal (h : conditions) : P = 800 :=
sorry

end simple_interest_principal_l135_135647


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135343

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) ∧ 
            ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ (m.digits.sum = 27) → m ≤ n :=
begin
  use 999,
  split,
  { -- 999 is a three-digit number 
    norm_num,
  },
  split,
  { -- 999 is less than or equal to 999
    norm_num,
  },
  split,
  { -- 999 is a multiple of 9
    norm_num,
  },
  split,
  { -- The sum of the digits of 999 is 27
    norm_num,
  },
  { -- For any three-digit number m, if it is a multiple of 9 and the sum of its digits is 27, then m ≤ 999
    intros m hm1,
    cases hm1 with hm2 hm3,
    cases hm3 with hm4 hm5,
    exact le_of_lt (by linarith),
    sorry
  },
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135343


namespace vector_arithmetic_l135_135478

open_locale classical

noncomputable def a : ℝ × ℝ := (2, 1)
noncomputable def b : ℝ × ℝ := (2, -2)

theorem vector_arithmetic : (2 : ℝ) • a - b = (2, 4) :=
by {
  simp [a, b, smul_smul, sub_eq_add_neg],
  sorry
}

end vector_arithmetic_l135_135478


namespace outer_radius_correct_l135_135381

noncomputable def outer_radius_of_track(inner_circumference : ℝ, track_width : ℝ) : ℝ := 
  inner_circumference / (2 * Real.pi) + track_width

theorem outer_radius_correct :
  outer_radius_of_track 880 25 ≈ 165.01 :=
by 
  -- Specify needed Real.approx
  have approx_ri: inner_circumference / (2 * Real.pi) ≈ 140.01 := sorry,
  -- Combine the width
  show 880 / (2 * Real.pi) + 25 ≈ 165.01,
  -- Conclude with the predefined approx_ri result
  exact sorry

end outer_radius_correct_l135_135381


namespace complex_conjugate_product_l135_135543

-- Define the complex number z
def z : ℂ := 1 + complex.i

-- Define the conjugate of z
def z_conj : ℂ := complex.conj z

-- The theorem that needs to be proven
theorem complex_conjugate_product :
  z * z_conj = 2 := 
sorry

end complex_conjugate_product_l135_135543


namespace bisection_method_requires_all_structures_l135_135369

-- Define the function f(x) = x^2 - 5
def f (x : ℝ) : ℝ := x ^ 2 - 5

-- Define the bisection method requirements as conditions
theorem bisection_method_requires_all_structures
  (a b : ℝ) 
  (h1 : f a * f b < 0) 
  (tolerance : ℝ) 
  (h2 : tolerance > 0) :
  (∃ (next_interval : ℝ × ℝ) 
     (h3 : next_interval.1 < next_interval.2), 
     (∃ r : ℝ, 
      f r = 0 ∨ (next_interval.1 < r ∧ r < next_interval.2 ∧ abs(f r) < tolerance)
     )
  ) :=
sorry

end bisection_method_requires_all_structures_l135_135369


namespace square_of_ratio_bounds_l135_135470

theorem square_of_ratio_bounds (n : ℕ) (h : n > 1) :
  let P_n := ∏ i in finset.range n, (2 * i + 1 : ℚ) / (2 * (i + 1) : ℚ) in
  (1 / (4 * n) : ℚ) < P_n^2 ∧ P_n^2 < (3 / (8 * n) : ℚ) := 
by
  sorry

end square_of_ratio_bounds_l135_135470


namespace rhombus_area_l135_135242

-- Define the given conditions as parameters
variables (EF GH : ℝ) -- Sides of the rhombus
variables (d1 d2 : ℝ) -- Diagonals of the rhombus

-- Statement of the theorem
theorem rhombus_area
  (rhombus_EFGH : ∀ (EF GH : ℝ), EF = GH)
  (perimeter_EFGH : 4 * EF = 40)
  (diagonal_EG_length : d1 = 16)
  (d1_half : d1 / 2 = 8)
  (side_length : EF = 10)
  (pythagorean_theorem : EF^2 = (d1 / 2)^2 + (d2 / 2)^2)
  (calculate_FI : d2 / 2 = 6)
  (diagonal_FG_length : d2 = 12) :
  (1 / 2) * d1 * d2 = 96 :=
sorry

end rhombus_area_l135_135242


namespace stratified_sampling_sophomores_l135_135025

theorem stratified_sampling_sophomores
  (freshmen : ℕ) (sophomores : ℕ) (juniors : ℕ) (total_selected : ℕ)
  (H_freshmen : freshmen = 550) (H_sophomores : sophomores = 700) (H_juniors : juniors = 750) (H_total_selected : total_selected = 100) :
  sophomores * total_selected / (freshmen + sophomores + juniors) = 35 :=
by
  sorry

end stratified_sampling_sophomores_l135_135025


namespace jim_fills_pool_l135_135906

theorem jim_fills_pool (J : ℝ) : 
  (1 / J) + (1 / 45) + (1 / 90) = (1 / 15) → J = 30 :=
by 
  intro h
  have : 1 / J + 1 / 45 + 1 / 90 = 1 / 45 + 1 / 90 + 1 / 15,
  { sorry }
  have : 1 / J = 1 / 30,
  { sorry }
  exact eq_of_div_eq_div (by normNum) this

end jim_fills_pool_l135_135906


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135363

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ 100 ≤ n ∧ n < 1000 ∧ (9 ∣ n) ∧ (∑ digit in n.digits, digit = 27) :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135363


namespace compute_expression_l135_135749

noncomputable def quadratic_roots (a b c : ℝ) :
  {x : ℝ × ℝ // a * x.fst^2 + b * x.fst + c = 0 ∧ a * x.snd^2 + b * x.snd + c = 0} :=
  let Δ := b^2 - 4 * a * c
  let root1 := (-b + Real.sqrt Δ) / (2 * a)
  let root2 := (-b - Real.sqrt Δ) / (2 * a)
  ⟨(root1, root2), by sorry⟩

theorem compute_expression :
  let roots := quadratic_roots 5 (-3) (-4)
  let x1 := roots.val.fst
  let x2 := roots.val.snd
  2 * x1^2 + 3 * x2^2 = (178 : ℝ) / 25 := by
  sorry

end compute_expression_l135_135749


namespace non_congruent_triangles_with_perimeter_10_l135_135152

theorem non_congruent_triangles_with_perimeter_10 :
  ∃ (T : Finset (Finset (ℕ × ℕ × ℕ))),
    (∀ (t ∈ T), let (a, b, c) := t in a ≤ b ∧ b ≤ c ∧
                  a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧
    T.card = 4 :=
by
  sorry

end non_congruent_triangles_with_perimeter_10_l135_135152


namespace sphere_inscribed_in_trihedral_angle_l135_135409

theorem sphere_inscribed_in_trihedral_angle
  (O S K L M : Point)
  (r : ℝ)
  (area1 area2 : ℝ)
  (condition1 : Sphere O r)
  (condition2 : InscribedInTrihedralAngle S O K L M)
  (condition3 : PlanePerpendicularSections area1 area2) :
  ∠KSO = Real.arcsin (1 / 7) ∧ CrossSectionArea K L M = 576 / 49 :=
by
  sorry

end sphere_inscribed_in_trihedral_angle_l135_135409


namespace find_m_l135_135165

theorem find_m (m : ℝ) (h1 : (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 12))) (h2 : m ≠ 0) : m = 12 :=
by
  sorry

end find_m_l135_135165


namespace f_2012_eq_3_l135_135481

noncomputable def f (a b α β : ℝ) (x : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem f_2012_eq_3 
  (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) 
  (h : f a b α β 2011 = 5) : 
  f a b α β 2012 = 3 :=
by
  sorry

end f_2012_eq_3_l135_135481


namespace triangle_sides_angles_l135_135971

theorem triangle_sides_angles (a b c A B C : ℝ) (h1: A = 2 * B) 
  (h2: sin C * sin (A - B) = sin B * sin (C - A)) 
  (h3: A + B + C = π) :
  (C = 5 * π / 8) ∧ (2 * a^2 = b^2 + c^2) :=
by
  -- Proof omitted
  sorry

end triangle_sides_angles_l135_135971


namespace perfect_square_expression_l135_135276

theorem perfect_square_expression : 
    ∀ x : ℝ, (11.98 * 11.98 + 11.98 * x + 0.02 * 0.02 = (11.98 + 0.02)^2) → (x = 0.4792) :=
by
  intros x h
  -- sorry placeholder for the proof
  sorry

end perfect_square_expression_l135_135276


namespace parabola_and_line_intersection_line_passing_through_directrix_l135_135487

noncomputable theory

open Real

-- Given conditions
def parabola (p : ℝ) (hp : 0 < p) : Prop := ∀ (x y : ℝ), y^2 = 2 * p * x
def line_intersects_parabola_at_points (x1 x2 y1 y2 : ℝ) (hx : x1 < x2) : Prop := 
  ∃ (p : ℝ) (line : ℝ → ℝ), (line = (λ x, sqrt(2) * (x - p/2))) ∧ (x1 < x2) ∧ (y1 = sqrt(2)*(x1 - p/2) ∧ y2 = sqrt(2)*(x2 - p/2)) ∧ (y1^2 = 2 * p * x1) ∧ (y2^2 = 2 * p * x2)
def distance_ab (x1 x2 y1 y2 : ℝ) : Prop := ∥(x2 - x1, y2 - y1)∥ = 6

-- Proof problem
theorem parabola_and_line_intersection (p : ℝ) (hp : 0 < p) (x1 x2 y1 y2 : ℝ) (hx : x1 < x2)
    (hintersect : line_intersects_parabola_at_points x1 x2 y1 y2 hx) 
    (hab : distance_ab x1 x2 y1 y2) : p = 2 ∧ (∀ (x y : ℝ), y^2 = 4 * x) := sorry

theorem line_passing_through_directrix (p : ℝ) (hp : 0 < p) (x y : ℝ)
    (hparabola : parabola p hp)
    (hf_prime : x = -1)
    (hinner_product : ∀ (x1 x2 : ℝ) (y1 y2 : ℝ), 
      (let M := (x1, y1), N := (x2, y2), F_prime := (-1, 0) in 
       (x1 + 1) * (x2 + 1) + y1 * y2 = 12)) 
    (hintersect : line_intersects_parabola_at_points x y) : 
  ∃ k : ℝ, k = sqrt(2)/2 ∨ k = -sqrt(2)/2 ∧ 
    ∀ x : ℝ, y = k * (x + 1) := sorry

end parabola_and_line_intersection_line_passing_through_directrix_l135_135487


namespace max_k_value_11_divisor_l135_135391

noncomputable def Legendre (n p : ℕ) : ℕ :=
  if p = 1 then 0 else
  (n / p) + Legendre (n / p) p

theorem max_k_value_11_divisor :
  let prod := (∏ i in (finset.range (1986 - 1000 + 1)).image (λ i, i + 1000), i),
      power11 := 11
  in
  (prod / power11 ^ 99) ∈ (ℕ) := sorry

end max_k_value_11_divisor_l135_135391


namespace correct_quadratic_equation_l135_135899

theorem correct_quadratic_equation :
  ∃ (a b c : ℝ), a = 1 ∧ b = -8 ∧ c = 24 ∧ (∀ x : ℝ, (x - 5) * (x - 3) = a * x^2 + b * x + c) ∧ (∀ x : ℝ, (x + 6) * (x + 4) = a * x^2 + b * x + c) ∧ (x^2 - 8x + 24 = 0) := 
  sorry

end correct_quadratic_equation_l135_135899


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135319

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((nat.digits 10 n).sum = 27) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ ((nat.digits 10 m).sum = 27) → m ≤ n) :=
begin
  use 999,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intro m,
    intro hm,
    cases hm,
    cases hm_left,
    cases hm_left_left,
    cases hm_left_right,
    cases hm_right,
    sorry
  },
sorry,
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135319


namespace Thabo_owns_25_hardcover_nonfiction_books_l135_135629

variable (H P F : ℕ)

-- Conditions
def condition1 := P = H + 20
def condition2 := F = 2 * P
def condition3 := H + P + F = 160

-- Goal
theorem Thabo_owns_25_hardcover_nonfiction_books (H P F : ℕ) (h1 : condition1 H P) (h2 : condition2 P F) (h3 : condition3 H P F) : H = 25 :=
by
  sorry

end Thabo_owns_25_hardcover_nonfiction_books_l135_135629


namespace number_of_solutions_l135_135853

theorem number_of_solutions :
  {x : ℝ | |x - 1| = |x - 2| + |x - 3|}.finite.to_finset.card = 2 :=
sorry

end number_of_solutions_l135_135853


namespace number_subsets_property_p_l135_135997

def has_property_p (a b : ℕ) : Prop := 17 ∣ (a + b)

noncomputable def num_subsets_with_property_p : ℕ :=
  -- sorry, put computation result here using the steps above but skipping actual computation for brevity
  3928

theorem number_subsets_property_p :
  num_subsets_with_property_p = 3928 := sorry

end number_subsets_property_p_l135_135997


namespace maximize_revenue_l135_135699

theorem maximize_revenue (p : ℝ) (h : p ≤ 30) : 
  (∀ q : ℝ, q ≤ 30 → (150 * 18.75 - 4 * (18.75:ℝ)^2) ≥ (150 * q - 4 * q^2)) ↔ p = 18.75 := 
sorry

end maximize_revenue_l135_135699


namespace arrangements_AB_adjacent_l135_135019

theorem arrangements_AB_adjacent (A B C D E : Type) :
  (∃ l : List (List Type), l.permutations.length = 24 ∧
  ∀ perm ∈ l.permutations, ∃ ab_rest_perm : List Type,
    ab_rest_perm = [A, B] ∧ perm = [ab_rest_perm, C, D, E] ∨ 
    perm = [C, ab_rest_perm, D, E] ∨ perm = [C, D, ab_rest_perm, E] ∨
    perm = [C, D, E, ab_rest_perm]) := 
sorry

end arrangements_AB_adjacent_l135_135019


namespace min_x9_minus_x1_is_9_l135_135255

theorem min_x9_minus_x1_is_9 :
  ∃ (x : Fin 9 → ℕ), (∀ i j, i < j → x i < x j) ∧ (∑ i, x i = 220) ∧ ((x 8 - x 0) = 9) :=
by
  sorry

end min_x9_minus_x1_is_9_l135_135255


namespace fg_equals_gf_l135_135860

theorem fg_equals_gf (m n p q : ℝ) (h : m + q = n + p) : ∀ x : ℝ, (m * (p * x + q) + n = p * (m * x + n) + q) :=
by sorry

end fg_equals_gf_l135_135860


namespace systematic_sampling_eighth_group_l135_135413

theorem systematic_sampling_eighth_group
  (total_employees : ℕ)
  (target_sample : ℕ)
  (third_group_value : ℕ)
  (group_count : ℕ)
  (common_difference : ℕ)
  (eighth_group_value : ℕ) :
  total_employees = 840 →
  target_sample = 42 →
  third_group_value = 44 →
  group_count = total_employees / target_sample →
  common_difference = group_count →
  eighth_group_value = third_group_value + (8 - 3) * common_difference →
  eighth_group_value = 144 :=
sorry

end systematic_sampling_eighth_group_l135_135413


namespace part1_C_value_part2_triangle_equality_l135_135983

noncomputable theory

variables (a b c : ℝ) (A B C : ℝ)
variables (h1 : A + B + C = Real.pi) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) (h3 : A = 2 * B)

-- Part 1: Proving that C = 5π/8 given the conditions
theorem part1_C_value :
  C = 5 * Real.pi / 8 :=
begin
  sorry
end

-- Part 2: Proving that 2a^2 = b^2 + c^2 given the conditions
theorem part2_triangle_equality :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
begin
  sorry
end

end part1_C_value_part2_triangle_equality_l135_135983


namespace average_percentage_reduction_l135_135396

theorem average_percentage_reduction (x : ℝ) (hx : 0 < x ∧ x < 1)
  (initial_price final_price : ℝ)
  (h_initial : initial_price = 25)
  (h_final : final_price = 16)
  (h_reduction : final_price = initial_price * (1-x)^2) :
  x = 0.2 :=
by {
  --". Convert fraction \( = x / y \)", proof is omitted
  sorry
}

end average_percentage_reduction_l135_135396


namespace tan_alpha_eq_neg_2_sqrt_2_l135_135098

theorem tan_alpha_eq_neg_2_sqrt_2
  (α : ℝ)
  (h1 : sin (α + π / 2) = 1 / 3)
  (h2 : α ∈ set.Ioo (-π / 2) 0) :
  tan α = -2 * real.sqrt 2 :=
sorry

end tan_alpha_eq_neg_2_sqrt_2_l135_135098


namespace seq_2016_plus_seq_3_l135_135520

open BigOperators

-- Define the sequence a
def seq (n : ℕ) : ℝ :=
  if n = 1 then 1
  else if n = 2 then -1/2 + Real.sqrt 5 / 2
  else if n % 6 = 1 then 1
  else if n % 6 = 2 then -1/2 + Real.sqrt 5 / 2
  else sorry -- complete the rest based on recursion properties

-- Define the recursion property
def rec_property (n : ℕ) (a : ℝ) : Prop :=
  (seq (n + 2) = 1 / (a + 1))

-- Prove the final result
theorem seq_2016_plus_seq_3 :
  seq 2016 + seq 3 = Real.sqrt 5 / 2 :=
by
  sorry

end seq_2016_plus_seq_3_l135_135520


namespace valid_inequalities_count_l135_135729

theorem valid_inequalities_count :
  (¬ (sqrt 5 + sqrt 9 > 2 * sqrt 7)) ∧
  (∀ (a b c : ℝ), a^2 + 2 * b^2 + 3 * c^2 ≥ (1 / 6) * (a + 2 * b + 3 * c)^2) ∧
  (∀ x : ℝ, exp x ≥ x + 1) →
  2 = 2 :=
by
  sorry

end valid_inequalities_count_l135_135729


namespace inequality_proof_l135_135915

variable (b : ℝ)

theorem inequality_proof (k : ℕ) (hk : k ∈ ({2, 3, 4} : Set ℕ)) (hb : 0 ≤ b) :
  let n := 2^k - 1
  in 1 + ∑ i in Finset.range (n + 1), b^(i * k) ≥ (1 + b^n)^k := 
  sorry

end inequality_proof_l135_135915


namespace find_b_minus_d_l135_135501

noncomputable def b (a : ℕ) := a^(3/2)
noncomputable def d (c : ℕ) := c^(5/4)

theorem find_b_minus_d (a b c d : ℕ) (h1 : log a b = 3/2) (h2 : log c d = 5/4) (h3 : a - c = 9) :
  b - d = 93 :=
by
  sorry

end find_b_minus_d_l135_135501


namespace minimum_shots_to_hit_1x4_ship_in_5x5_grid_l135_135052

def grid := matrix (fin 5) (fin 5) bool

def ship_1x4_placements (g: grid) : set (set (fin 5 × fin 5)) :=
  { s | (∃ r c, c + 3 < 5 ∧ s = {(r, c), (r, c + 1), (r, c + 2), (r, c + 3)}) ∨ 
       (∃ r c, r + 3 < 5 ∧ s = {(r, c), (r + 1, c), (r + 2, c), (r + 3, c)}) }

def shots := finset (fin 5 × fin 5)

theorem minimum_shots_to_hit_1x4_ship_in_5x5_grid (shots_fired : shots) (h: shots_fired.card = 6) :
  ∀ s ∈ ship_1x4_placements, (shots_fired.to_set ∩ s).nonempty := sorry

end minimum_shots_to_hit_1x4_ship_in_5x5_grid_l135_135052


namespace correct_statements_l135_135525

-- Define the circles C1 and C2
def C1_eq (x y : ℝ) : Prop := x^2 + y^2 = 9
def C2_eq (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 16

-- Define the equation of the intersecting chord
def chord_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 9 = 0

-- Define the length of the chord
def chord_length : ℝ := 24 / 5

-- Define the maximum distance between points on the circles
def max_distance : ℝ := 12

-- Theorem statements
theorem correct_statements :
  (∃ x y : ℝ, C1_eq x y ∧ chord_eq x y) ∧
  (∃ x y : ℝ, C2_eq x y ∧ chord_eq x y) ∧
  chord_length = 24 / 5 ∧
  max_distance = 12 :=
by
  sorry

end correct_statements_l135_135525


namespace distance_between_parallel_lines_l135_135443

open EuclideanGeometry

noncomputable def point_a : ℝ × ℝ := (3, -2)
noncomputable def point_b : ℝ × ℝ := (4, -1)
noncomputable def dir_d : ℝ × ℝ := (2, 1)

theorem distance_between_parallel_lines :
  let v := (point_b.1 - point_a.1, point_b.2 - point_a.2) in
  let proj_v_d := ((v.1 * dir_d.1 + v.2 * dir_d.2) / (dir_d.1 * dir_d.1 + dir_d.2 * dir_d.2)) • dir_d in
  let orth_v := (v.1 - proj_v_d.1, v.2 - proj_v_d.2) in
  let dist := (orth_v.1^2 + orth_v.2^2).sqrt in
  dist = sqrt 10 / 5 :=
by
  sorry

end distance_between_parallel_lines_l135_135443


namespace monotonicity_intervals_inequality_condition_l135_135140

noncomputable def f (x : ℝ) := Real.exp x * (x^2 + 2 * x + 1)

theorem monotonicity_intervals :
  (∀ x ∈ Set.Iio (-3 : ℝ), 0 < (Real.exp x * ((x + 3) * (x + 1)))) ∧
  (∀ x ∈ Set.Ioo (-3 : ℝ) (-1 : ℝ), 0 > (Real.exp x * ((x + 3) * (x + 1)))) ∧
  (∀ x ∈ Set.Ioi (-1 : ℝ), 0 < (Real.exp x * ((x + 3) * (x + 1)))) := sorry

theorem inequality_condition (a : ℝ) : 
  (∀ x > 0, Real.exp x * (x^2 + 2 * x + 1) > a * x^2 + a * x + 1) ↔ a ≤ 3 := sorry

end monotonicity_intervals_inequality_condition_l135_135140


namespace day_256_2003_l135_135862

def day_of_week := ℕ

def sunday : day_of_week := 0
def monday : day_of_week := 1
def tuesday : day_of_week := 2
def wednesday : day_of_week := 3
def thursday : day_of_week := 4
def friday : day_of_week := 5
def saturday : day_of_week := 6

-- Condition: the 40th day of 2003 is Sunday
def day_40_2003 := sunday

-- Function to determine the day of the week of the nth day of the year given the mth day
def day_of_nth(n : ℕ) (mth_day : day_of_week) := (mth_day + (n - 40)) % 7

-- Prove: the 256th day of 2003 is Saturday given that the 40th day of 2003 is Sunday
theorem day_256_2003 : day_of_nth 256 day_40_2003 = saturday := 
by
    -- We set the 40th day as Sunday and compute the 256th day
    unfold day_40_2003 day_of_nth sunday saturday
    unfold day_40_2003
    sorry

end day_256_2003_l135_135862


namespace part_a_part_b_part_c_l135_135895

noncomputable theory

-- Part (a)
theorem part_a (A B C D S T P Q : Point) (r : ℝ)
  (h1 : Circle B 1)
  (h2 : Circle D r)
  (h3 : Tangent (Line A D) B)
  (h4 : Tangent (Line A S) B P)
  (h5 : Tangent (Line A S) D Q)
  (h6 : Tangent (Line A T) B P)
  (h7 : Tangent (Line A T) D Q)
  (h8 : Perpendicular (Segment S T) (Line A D) C)
  (h9 : Tangent (Segment S T) B)
  (h10 : Tangent (Segment S T) D)
  (h11 : AS = ST)
  (h12 : ST = AT) :
  r = 3 :=
sorry

-- Part (b)
theorem part_b (A B C D S T P Q : Point) (r : ℝ)
  (h1 : Circle B 1)
  (h2 : Circle D r)
  (h3 : Tangent (Line A D) B)
  (h4 : Tangent (Line A S) B P)
  (h5 : Tangent (Line A S) D Q)
  (h6 : Tangent (Line A T) B P)
  (h7 : Tangent (Line A T) D Q)
  (h8 : Perpendicular (Segment S T) (Line A D) C)
  (h9 : Tangent (Segment S T) B)
  (h10 : Tangent (Segment S T) D)
  (h11 : DQ = QP) :
  r = 4 :=
sorry

-- Part (c)
theorem part_c (A B C D S T P Q O V W : Point) (r : ℝ)
  (h1 : Circle B 1)
  (h2 : Circle D r)
  (h3 : Tangent (Line A D) B)
  (h4 : Tangent (Line A S) B P)
  (h5 : Tangent (Line A S) D Q)
  (h6 : Tangent (Line A T) B P)
  (h7 : Tangent (Line A T) D Q)
  (h8 : Perpendicular (Segment S T) (Line A D) C)
  (h9 : Tangent (Segment S T) B)
  (h10 : Tangent (Segment S T) D)
  (h11 : Circle O (segment A S))
  (h12 : O = center (Circle passing_through A S T))
  (h13 : Intersect (Circle D r) (Circle O (segment A S)) V W)
  (h14 : Perpendicular OV DV) :
  r = 2 + sqrt 5 :=
sorry

end part_a_part_b_part_c_l135_135895


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135315

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 + d2 + d3

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, is_three_digit n ∧ is_multiple_of_9 n ∧ digit_sum n = 27 ∧
  ∀ m : ℕ, is_three_digit m ∧ is_multiple_of_9 m ∧ digit_sum m = 27 → m ≤ n := 
by 
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135315


namespace number_of_solutions_l135_135076

noncomputable def g (n : ℤ) : ℤ :=
  (⌈119 * n / 120⌉ - ⌊120 * n / 121⌋ : ℤ)

theorem number_of_solutions : (finset.range 12120).filter (λ n, g n = 1).card = 12120 :=
sorry

end number_of_solutions_l135_135076


namespace complex_product_conjugate_l135_135033

theorem complex_product_conjugate : (1 + Complex.I) * (1 - Complex.I) = 2 := 
by 
  -- Lean proof goes here
  sorry

end complex_product_conjugate_l135_135033


namespace sum_terms_of_arithmetic_seq_l135_135054

noncomputable def f (x : ℝ) : ℝ :=
if x ≠ 1 then 1 / |x - 1| else 1

def h (x b : ℝ) : ℝ :=
(f x) ^ 2 + b * f x + (b ^ 2) / 2 - 5 / 8

theorem sum_terms_of_arithmetic_seq (b : ℝ)
  (x1 x2 x3 x4 x5 : ℝ)
  (h_roots : ∀ {b}, 
    ∃ (x1 x2 x3 x4 x5 : ℝ), 
        x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5 ∧ 
        h x1 b = 0 ∧ h x2 b = 0 ∧ h x3 b = 0 ∧ h x4 b = 0 ∧ h x5 b = 0 ∧
        (x2 - x1) = (x3 - x2) ∧ 
        (x3 - x2) = (x4 - x3) ∧ 
        (x4 - x3) = (x5 - x4)) : 
  x1 + x2 + x3 + x4 + x5 + (x1 - (x3 - x2)) + (x1 - 2 * (x3 - x2)) + (x5 + (x3 - x2)) + (x5 + 2 * (x3 - x2)) = 35 :=
sorry

end sum_terms_of_arithmetic_seq_l135_135054


namespace shared_boundary_segment_l135_135785

noncomputable section

open Classical

-- Definitions based on conditions
def grid_square_size : ℕ := 55

def corner_pieces_count : ℕ := 400
def individual_cells_count : ℕ := 500

/-
Given a 55 x 55 grid square from which 400 three-cell corner pieces and 
500 individual cells are cut out, prove that some two cut-out figures 
share a boundary segment.
-/
theorem shared_boundary_segment :
  ∃ (fig1 fig2 : set (ℕ × ℕ)), 
  fig1 ≠ fig2 ∧ (fig1 ∩ fig2).nonempty :=
sorry

end shared_boundary_segment_l135_135785


namespace more_regular_than_diet_l135_135402

-- Define the conditions
def num_regular_soda : Nat := 67
def num_diet_soda : Nat := 9

-- State the theorem
theorem more_regular_than_diet :
  num_regular_soda - num_diet_soda = 58 :=
by
  sorry

end more_regular_than_diet_l135_135402


namespace metrizable_of_compact_Hausdorff_union_metrizable_subsets_l135_135625

variable {K : Type*} [TopologicalSpace K] [CompactSpace K] [T2Space K]
variable (A : ℕ → set K)
variable [∀ n, MetrizableSpace (A n)]

-- Assume K = ⋃₀ (set.range A) and ∀ n m, n < m → A n ⊆ A m

theorem metrizable_of_compact_Hausdorff_union_metrizable_subsets (hK : ∀ k, k ∈ K ↔ ∃ n, k ∈ A n)
  (h_increasing: ∀ n m, n < m → A n ⊆ A m) : MetrizableSpace K :=
sorry

end metrizable_of_compact_Hausdorff_union_metrizable_subsets_l135_135625


namespace sue_payment_is_900_l135_135623
noncomputable theory

def total_cost := 2100
def days_in_week := 7
def sister_days := 4
def sue_days := days_in_week - sister_days

def sue_fraction := (sue_days : ℚ) / days_in_week

def sue_payment := total_cost * sue_fraction

theorem sue_payment_is_900 : sue_payment = 900 := 
by
  sorry

end sue_payment_is_900_l135_135623


namespace central_angle_common_chord_l135_135634

theorem central_angle_common_chord:
  (∀ x y : ℝ, (x - 2)^2 + y^2 = 4 → x^2 + (y - 2)^2 = 4 → true) →
  ∃ θ : ℝ, θ = (2 : ℝ) * real.pi / 4 :=
by
  sorry

end central_angle_common_chord_l135_135634


namespace inscribed_square_in_rectangle_with_triangles_l135_135576

theorem inscribed_square_in_rectangle_with_triangles :
  ∀ (a b : ℝ), a = 16 → b = 20 →
  ∃ y : ℝ, y = sqrt (2) * (2 * sqrt (89) - (2 * sqrt (267) / 3)) := by
  intros
  rw [H, H_1]
  refine ⟨_, rfl⟩
  rw [sqrt_mul', sqrt_two_mul, mul_div, two_mul_div, sqrt_mul, sqrt_mul, div_eq_div_iff, mul_assoc, mul_left_comm, mul_comm, ← mul_assoc, ← mul_assoc, div_sq]
  calc
    sqrt (2 * (4 * 89)) - 2 * sqrt (267) / 3 =
    sqrt (8 * 89) - 2 * sqrt (267) / 3 : by rw mul_assoc
    ... = sqrt (267 * 3) / 3 : by rw ← sqrt_div; field_simp; ring
    ... = sqrt (267) : by rw sqrt_div (sqrt 267) (sqrt 3)

end inscribed_square_in_rectangle_with_triangles_l135_135576


namespace sin_arithmetic_sequence_l135_135766

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 360) :
  (sin a + sin (3 * a) = 2 * sin (2 * a)) ↔ 
  (a = 30 ∨ a = 150 ∨ a = 210 ∨ a = 330) :=
by
  sorry

end sin_arithmetic_sequence_l135_135766


namespace oil_vinegar_new_ratio_l135_135881

theorem oil_vinegar_new_ratio (initial_oil initial_vinegar new_vinegar : ℕ) 
    (h1 : initial_oil / initial_vinegar = 3 / 1)
    (h2 : new_vinegar = (2 * initial_vinegar)) :
    initial_oil / new_vinegar = 3 / 2 :=
by
  sorry

end oil_vinegar_new_ratio_l135_135881


namespace product_of_real_roots_l135_135546

noncomputable def equation : Polynomial ℝ :=
  Polynomial.polynomial_of_function (λ x : ℝ, x^2 + 9 * x + 13 - 2 * sqrt(x^2) - 9 * x - 21)

theorem product_of_real_roots :
  let P := (5 : ℝ) in
  ∀ x : ℝ, (equation.eval x = 0) → ∃ a b : ℝ, (a ≠ b) ∧ (equation.eval a = 0) ∧ (equation.eval b = 0) ∧ (a * b = P) :=
by
  sorry

end product_of_real_roots_l135_135546


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135314

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 + d2 + d3

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, is_three_digit n ∧ is_multiple_of_9 n ∧ digit_sum n = 27 ∧
  ∀ m : ℕ, is_three_digit m ∧ is_multiple_of_9 m ∧ digit_sum m = 27 → m ≤ n := 
by 
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135314


namespace not_black_cows_count_l135_135246

theorem not_black_cows_count (total_cows : ℕ) (black_cows : ℕ) (h1 : total_cows = 18) (h2 : black_cows = 5 + total_cows / 2) :
  total_cows - black_cows = 4 :=
by 
  -- Insert the actual proof here
  sorry

end not_black_cows_count_l135_135246


namespace arithmetic_sequence_X_value_l135_135877

theorem arithmetic_sequence_X_value :
  ∃ (a1 a2 a3 d1 d2 d3 : ℤ), 
    a1 = 15 ∧ 
    a2 = 10 ∧ 
    a3 = 5 ∧
    d2 = 5 ∧
    d3 = 5 ∧ 
    ∀ n m : ℤ,
    m = 7 ∧ arithmetic_seq a2 d2 n 
    ∧ seq_val (S a3 d3 1) - seq_val (S a1 d1 m) = 20 := sorry

end arithmetic_sequence_X_value_l135_135877


namespace tangent_line_equation_l135_135773

noncomputable def f : ℝ → ℝ := λ x, Real.log (x + 2) - 3 * x

theorem tangent_line_equation :
  let x0 := -1
  let y0 := 3
  let y' := λ x, 1 / (x + 2) - 3
  let k := y' x0
  let tangent_line := λ x y, y - y0 = k * (x - x0)
  let final_eq := 2 * x + y - 1 = 0
  tangent_line x0 y0 = final_eq :=
by
  sorry

end tangent_line_equation_l135_135773


namespace find_b1_l135_135447

theorem find_b1 (b : ℕ → ℕ)
  (h : ∀ n ≥ 2, (∑ i in Finset.range (n + 1), b i) = (n + 1)^2 * b n)
  (h50 : b 50 = 2) :
  b 1 = 2550 :=
sorry

end find_b1_l135_135447


namespace length_of_train_l135_135723

-- Definitions for the given conditions:
def speed : ℝ := 60   -- in kmph
def time : ℝ := 20    -- in seconds
def platform_length : ℝ := 213.36  -- in meters

-- Conversion factor from km/h to m/s
noncomputable def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Total distance covered by train while crossing the platform
noncomputable def total_distance (speed_in_kmph : ℝ) (time_in_seconds : ℝ) : ℝ := 
  (kmph_to_mps speed_in_kmph) * time_in_seconds

-- Length of the train
noncomputable def train_length (total_distance_covered : ℝ) (platform_len : ℝ) : ℝ :=
  total_distance_covered - platform_len

-- Expected length of the train
def expected_train_length : ℝ := 120.04

-- Theorem to prove the length of the train given the conditions
theorem length_of_train : 
  train_length (total_distance speed time) platform_length = expected_train_length :=
by 
  sorry

end length_of_train_l135_135723


namespace intersection_union_G_equality_l135_135995

def E : Set ℕ := {x | x < 6 ∧ 0 < x}
def F : Set ℕ := {x | (x - 1) * (x - 2) = 0}
def G (a : ℕ) : Set ℕ := {a, a^2 + 1}

theorem intersection_union (E F : Set ℕ) :
  E = {1, 2, 3, 4, 5} → F = {1, 2} → E ∩ F = {1, 2} ∧ E ∪ F = {1, 2, 3, 4, 5} :=
by young_a sorry

theorem G_equality (a : ℕ) :
  F ⊆ G a → G a ⊆ F → a = 1 :=
by G.a sorry

end intersection_union_G_equality_l135_135995


namespace knight_moves_2009_reachable_squares_l135_135707

theorem knight_moves_2009_reachable_squares :
  let start_pos := (0, 0) in
  let moves := 2009 in
  ∃ squares_reachable : Finset (ℕ × ℕ),
    knight_moves start_pos moves = squares_reachable ∧ squares_reachable.card = 32 :=
sorry

end knight_moves_2009_reachable_squares_l135_135707


namespace smallest_of_three_consecutive_odd_numbers_l135_135648

theorem smallest_of_three_consecutive_odd_numbers (x : ℤ) 
(h_sum : x + (x+2) + (x+4) = 69) : x = 21 :=
by
  sorry

end smallest_of_three_consecutive_odd_numbers_l135_135648


namespace part1_proof_part2_proof_l135_135946

-- Definitions for triangle sides and angles
variables {A B C a b c : ℝ}

-- Condition 1
def condition1 : Prop := sin C * sin (A - B) = sin B * sin (C - A)

-- Condition 2
def condition2 : Prop := A = 2 * B

-- Proof Problem 1
theorem part1_proof : condition1 → condition2 → C = 5 / 8 * π :=
by sorry

-- Proof Problem 2
theorem part2_proof : condition1 → condition2 → 2 * a^2 = b^2 + c^2 :=
by sorry

end part1_proof_part2_proof_l135_135946


namespace necessary_but_not_sufficient_l135_135116

variable (x : ℝ)

-- Definitions of conditions p and q
def p : Prop := 1 ≤ x ∧ x ≤ 4
def q : Prop := abs (x - 2) > 1

theorem necessary_but_not_sufficient :
  p → (¬q ↔ (1 ≤ x ∧ x ≤ 3)) →
  (¬q → p) ∧ ¬(p → ¬q) :=
by
  sorry

end necessary_but_not_sufficient_l135_135116


namespace cube_chalk_marks_different_orientations_l135_135232

theorem cube_chalk_marks_different_orientations (cube : Type) (table : Type) :
  ∃ (orientation1 orientation2 : cube → table), 
  (orientation1 ≠ orientation2) ∧ 
  (∃ marked_points : fin 100 → table, marked_points orientation1 ≠ marked_points orientation2) :=
sorry

end cube_chalk_marks_different_orientations_l135_135232


namespace projection_correct_l135_135407

open Matrix

def proj (a b : Fin 2 → ℚ) : Fin 2 → ℚ :=
  let dot1 := a 0 * b 0 + a 1 * b 1
  let dot2 := b 0 * b 0 + b 1 * b 1
  let scalar := dot1 / dot2
  fun i => scalar * b i

theorem projection_correct :
  proj (λ i, if i = 0 then -4 else 1) (λ i, if i = 0 then 1 else -1) =
    (λ i, if i = 0 then -5/2 else 5/2) :=
  sorry

end projection_correct_l135_135407


namespace no_algorithm_for_connectedness_in_fewer_than_2016_queries_l135_135180

theorem no_algorithm_for_connectedness_in_fewer_than_2016_queries 
    (V : Type) [Fintype V] [DecidableEq V] (E : Type) [DecidableEq E]
    [Graph V E] 
    (hV : Fintype.card V = 64) : ¬ ∃ (f : (V → V → bool) → bool), 
    (∀ g : V → V → bool, f g = connected g) → 
    ∃ g : V → V → bool, queries_for_connectedness f g < 2016
  := sorry

end no_algorithm_for_connectedness_in_fewer_than_2016_queries_l135_135180


namespace trig_identity_at_fixed_point_l135_135831

noncomputable def func (x a : ℝ) := 4 + Real.log (x - 2) / Real.log a

theorem trig_identity_at_fixed_point :
  ∀ (a : ℝ), 0 < a → a ≠ 1 → func 3 a = 4 →
  let α := Real.arctan (4 / 3) in
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 10 :=
by
  intros a ha1 ha2 hfx ha
  let α := Real.arctan (4 / 3)
  sorry

end trig_identity_at_fixed_point_l135_135831


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135338

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n % 9 = 0 ∧ (n.digits.sum = 27) ∧
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ m % 9 = 0 ∧ (m.digits.sum = 27) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135338


namespace triangle_sides_angles_l135_135968

theorem triangle_sides_angles (a b c A B C : ℝ) (h1: A = 2 * B) 
  (h2: sin C * sin (A - B) = sin B * sin (C - A)) 
  (h3: A + B + C = π) :
  (C = 5 * π / 8) ∧ (2 * a^2 = b^2 + c^2) :=
by
  -- Proof omitted
  sorry

end triangle_sides_angles_l135_135968


namespace solve_quadratic_inequality_l135_135618

noncomputable def quadratic_inequality_solution (a : ℝ) (a_ne_zero : a ≠ 0) : Set ℝ :=
if a < 0 then { x | (1/a) < x ∧ x < 1 }
else if 0 < a ∧ a < 1 then { x | (x < 1 ∨ 1 < x ∧ x < 1/a) } ∪ { x | (1/a) < x }
else if a = 1 then { x | x < 1 } ∪ { x | 1 < x }
else { x | x < 1/a } ∪ { x | 1 < x }

theorem solve_quadratic_inequality (a : ℝ) (a_ne_zero : a ≠ 0) :
  ∀ x, ax^2 - (a+1)x + 1 > 0 ↔ x ∈ quadratic_inequality_solution a a_ne_zero :=
by sorry

end solve_quadratic_inequality_l135_135618


namespace problem_part1_problem_part2_l135_135938

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l135_135938


namespace jay_savings_first_week_l135_135904

theorem jay_savings_first_week :
  ∀ (x : ℕ), (x + (x + 10) + (x + 20) + (x + 30) = 60) → x = 0 :=
by
  intro x h
  sorry

end jay_savings_first_week_l135_135904


namespace initial_distance_between_trains_l135_135666

/-- Definitions based on given problem conditions -/
def length_train1 : ℝ := 90
def length_train2 : ℝ := 95
def speed_train1_kmph : ℝ := 64
def speed_train2_kmph : ℝ := 92
def meeting_time_seconds : ℝ := 5.768769267689355

/-- Speed conversion from km/h to m/s -/
def kmph_to_mps (speed: ℝ) : ℝ := speed * 1000 / 3600

def speed_train1_mps : ℝ := kmph_to_mps speed_train1_kmph
def speed_train2_mps : ℝ := kmph_to_mps speed_train2_kmph

/-- Distance calculations for the distances each train covers -/
def distance_train1 : ℝ := speed_train1_mps * meeting_time_seconds
def distance_train2 : ℝ := speed_train2_mps * meeting_time_seconds

/-- Initial distance is the sum of the distances covered -/
def initial_distance : ℝ := distance_train1 + distance_train2

/-- The property we need to prove: The initial distance is 250 meters -/
theorem initial_distance_between_trains :
  initial_distance = 250 := 
by
  sorry

end initial_distance_between_trains_l135_135666


namespace isosceles_triangle_triangle_area_l135_135551

noncomputable def area_of_Δ (a b c : ℝ) (cosA : ℝ) : ℝ :=
  1/2 * b * c * (Real.sqrt (1 - cosA^2))

theorem isosceles_triangle {a b c : ℝ} (h : b * Real.cos c = a * Real.cos B^2 + b * Real.cos A * Real.cos B) :
  B = c :=
sorry

theorem triangle_area {a b c : ℝ} (cosA : ℝ) (cosA_eq : cosA = 7/8) (perimeter : a + b + c = 5) 
  (b_eq_c : b = c) :
  area_of_Δ a b c cosA = Real.sqrt 15 / 4 :=
sorry

end isosceles_triangle_triangle_area_l135_135551


namespace unsuitable_temperature_for_refrigerator_l135_135290

theorem unsuitable_temperature_for_refrigerator:
  let avg_temp := -18
  let variation := 2
  let min_temp := avg_temp - variation
  let max_temp := avg_temp + variation
  let temp_A := -17
  let temp_B := -18
  let temp_C := -19
  let temp_D := -22
  temp_D < min_temp ∨ temp_D > max_temp := by
  sorry

end unsuitable_temperature_for_refrigerator_l135_135290


namespace no_solution_for_equation_l135_135039

/-- The given equation expressed using letters as unique digits:
    ∑ (letters as digits) from БАРАНКА + БАРАБАН + КАРАБАС = ПАРАЗИТ
    We aim to prove that there are no valid digit assignments satisfying the equation. -/
theorem no_solution_for_equation :
  ∀ (b a r n k s p i t: ℕ),
  b ≠ a ∧ b ≠ r ∧ b ≠ n ∧ b ≠ k ∧ b ≠ s ∧ b ≠ p ∧ b ≠ i ∧ b ≠ t ∧
  a ≠ r ∧ a ≠ n ∧ a ≠ k ∧ a ≠ s ∧ a ≠ p ∧ a ≠ i ∧ a ≠ t ∧
  r ≠ n ∧ r ≠ k ∧ r ≠ s ∧ r ≠ p ∧ r ≠ i ∧ r ≠ t ∧
  n ≠ k ∧ n ≠ s ∧ n ≠ p ∧ n ≠ i ∧ n ≠ t ∧
  k ≠ s ∧ k ≠ p ∧ k ≠ i ∧ k ≠ t ∧
  s ≠ p ∧ s ≠ i ∧ s ≠ t ∧
  p ≠ i ∧ p ≠ t ∧
  i ≠ t →
  100000 * b + 10000 * a + 1000 * r + 100 * a + 10 * n + k +
  100000 * b + 10000 * a + 1000 * r + 100 * a + 10 * b + a + n +
  100000 * k + 10000 * a + 1000 * r + 100 * a + 10 * b + a + s ≠ 
  100000 * p + 10000 * a + 1000 * r + 100 * a + 10 * z + i + t :=
sorry

end no_solution_for_equation_l135_135039


namespace number_of_roses_l135_135295

theorem number_of_roses 
  (R L T : ℕ)
  (h1 : R + L + T = 100)
  (h2 : R = L + 22)
  (h3 : R = T - 20) : R = 34 := 
sorry

end number_of_roses_l135_135295


namespace dogs_grouping_l135_135259

theorem dogs_grouping (dogs : Finset α) (fluffy nipper : α) :
  dogs.card = 12 ∧ fluffy ∈ dogs ∧ nipper ∈ dogs →
  ∃ g1 g2 g3 : Finset α,
    (g1.card = 4 ∧ g2.card = 5 ∧ g3.card = 3) ∧
    (fluffy ∈ g1) ∧ (nipper ∈ g2) ∧
    (g1 ∪ g2 ∪ g3 = dogs) ∧ (g1 ∩ g2 = ∅) ∧ (g1 ∩ g3 = ∅) ∧ (g2 ∩ g3 = ∅) ∧
    (∃ n : ℕ, n = 4200) :=
by
  sorry

end dogs_grouping_l135_135259


namespace find_extrema_l135_135813

theorem find_extrema (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  let y := (1 / 4) ^ (x - 1) - 4 * (1 / 2) ^ x + 2 in
  (∀ y, y = ((1 / 4) ^ (x - 1) - 4 * (1 / 2) ^ x + 2) → 1 ≤ y ∧ y ≤ 2) :=
by
  sorry

end find_extrema_l135_135813


namespace B_cycling_speed_l135_135015

/--
A walks at 10 kmph. 10 hours after A starts, B cycles after him at a certain speed.
B catches up with A at a distance of 200 km from the start. Prove that B's cycling speed is 20 kmph.
-/
theorem B_cycling_speed (speed_A : ℝ) (time_A_to_start_B : ℝ) 
  (distance_at_catch : ℝ) (B_speed : ℝ)
  (h1 : speed_A = 10) 
  (h2 : time_A_to_start_B = 10)
  (h3 : distance_at_catch = 200)
  (h4 : distance_at_catch = speed_A * time_A_to_start_B + speed_A * (distance_at_catch / speed_B)) :
    B_speed = 20 := by
  sorry

end B_cycling_speed_l135_135015


namespace middle_angle_of_triangle_l135_135692

theorem middle_angle_of_triangle (α β γ : ℝ) 
  (h1 : 0 < β) (h2 : β < 90) 
  (h3 : α ≤ β) (h4 : β ≤ γ) 
  (h5 : α + β + γ = 180) :
  True :=
by
  -- Proof would go here
  sorry

end middle_angle_of_triangle_l135_135692


namespace circle_chords_properties_l135_135885

theorem circle_chords_properties
  (t k : ℝ) 
  (h1 : ∀ A B C D R O : Point, IsUnitCircle O
             → Chord A B
             → Chord C D
             → t = |A - C| = |C - B| = |D - R |
             → k = |C - D|
             → ∀ O C D : Point, (parallel_to_radius O R A B)
  (h2 : ∀ A C B D R : Point, parallel_to_radius O R AB)
  (h3 : Chord C D = k)
  (h4 : Chord AC = t)
  (h5 : Chord CB = t)
  (h6 : Chord DR = t)
  (h7 : ∀ O C D : Point, IsUnitCircle O )
  (h8 : ∀ O A : Point, IsUnitCircle O R )
  ) :
  (k - t = sqrt 2) ∧ (k * t = 2) ∧ (k^2 - t^2 = 2) := by
     sorry

end circle_chords_properties_l135_135885


namespace CP_length_l135_135627

-- Define the variables and conditions
variables (A B C : Type) [AddCommGroup C] [Module ℝ C]
variables (B : Submodule ℝ C) (C B : Submodule ℝ C) (A B C : Point ℝ) (r : ℝ)

-- Conditions
def right_triangle (A B C : Point ℝ) : Prop := 
  ∃ (AB : ℝ) (BC : ℝ) (AC : ℝ), AB = (A - B).norm ∧ BC = (B - C).norm ∧ 
  AC = (A - C).norm ∧ (A - B) ⊥ (B - C)

-- Given lengths
def given_lengths (A B C : Point ℝ) : Prop :=
  (A - C).norm = real.sqrt 85 ∧ (A - B).norm = 7

-- Theorem
theorem CP_length (A B C : Point ℝ) (P : Point ℝ) (r : ℝ) 
  (h1 : right_triangle A B C)
  (h2 : given_lengths A B C)
  (h3 : center_circle : center_circle A B P r) 
  : (C - P).norm = 6 :=
  sorry

end CP_length_l135_135627


namespace additional_money_required_l135_135043

   theorem additional_money_required (patricia_money lisa_money charlotte_money total_card_cost : ℝ) 
       (h1 : patricia_money = 6)
       (h2 : lisa_money = 5 * patricia_money)
       (h3 : lisa_money = 2 * charlotte_money)
       (h4 : total_card_cost = 100) :
     (total_card_cost - (patricia_money + lisa_money + charlotte_money) = 49) := 
   by
     sorry
   
end additional_money_required_l135_135043


namespace max_min_diff_l135_135177

def sequence (n : ℕ) : ℝ := 5 * (2 / 5)^(2 * n - 2) - 4 * (2 / 5)^(n - 1)

theorem max_min_diff (n : ℕ) (hn : n > 0) :
  (∃ p q, p = 1 ∧ q = 2 ∧ (sequence q - sequence p = 1)) :=
begin
  use [1, 2],
  split,
  { refl, },
  split,
  { refl, },
  sorry
end

end max_min_diff_l135_135177


namespace find_reduced_price_l135_135012

noncomputable def original_price (P : ℝ) (R := 0.8 * P) : Prop :=
let kgs_more := (800 / R) - (800 / P) in
kgs_more = 5

theorem find_reduced_price (P R : ℝ) (h1 : original_price P R) : R = 32 := by
  -- We only write the statement, without proof.
  sorry

end find_reduced_price_l135_135012


namespace eval_poly_roots_l135_135590

noncomputable def polynomial_roots : {a b c : ℝ} :=
{ a, b, c :
  polynomial.coeff (polynomial.X ^ 3 - 15 * polynomial.X ^ 2 + 25 * polynomial.X - 12) 3 = 0 ∧
  polynomial.coeff (polynomial.X ^ 3 - 15 * polynomial.X ^ 2 + 25 * polynomial.X - 12) 2 = -15 ∧
  polynomial.coeff (polynomial.X ^ 3 - 15 * polynomial.X ^ 2 + 25 * polynomial.X - 12) 1 = 25 ∧
  polynomial.coeff (polynomial.X ^ 3 - 15 * polynomial.X ^ 2 + 25 * polynomial.X - 12) 0 = -12 }

open polynomial

theorem eval_poly_roots :
  let a, b, c := polynomial_roots
  (1 + a) * (1 + b) * (1 + c) = 53 :=
by
  let a := ∃ x, by rw [a_polynomial_roots x]
  let b := ∃ x, by rw [b_polynomial_roots x]
  let c := ∃ x, by rw [c_polynomial_roots x]
  have h₁ : a + b + c = 15 := Vieta_sum a b c;
  have h₂ : a * b + b * c + c * a = 25 := Vieta_products a b c;
  have h₃ : a * b * c = 12 := Vieta_product a b c;
  calc
    (1 + a) * (1 + b) * (1 + c) = 1 + (a + b + c) + (ab + bc + ca) + abc : by ring
    ... = 1 + 15 + 25 + 12 : by rw [h₁, h₂, h₃]
    ... = 53 : by norm_num

end eval_poly_roots_l135_135590


namespace probability_plane_passes_through_center_l135_135713

noncomputable theory

open_locale classical

-- Define a regular tetrahedron
structure RegularTetrahedron :=
  (vertices : fin 4 → ℝ × ℝ × ℝ)
  (equilateral_faces : ∀ (a b c : fin 4), 
    a ≠ b → b ≠ c → a ≠ c → 
    let v1 := vertices a,
        v2 := vertices b,
        v3 := vertices c in
    dist v1 v2 = dist v2 v3 ∧ dist v2 v3 = dist v3 v1) -- Equilateral faces

-- Prove that choosing any 3 vertices out of 4 makes a plane that passes through the center of the tetrahedron
theorem probability_plane_passes_through_center 
  (T : RegularTetrahedron) (v1 v2 v3 : fin 4) 
  (h1 : v1 ≠ v2) (h2 : v2 ≠ v3) (h3 : v1 ≠ v3) : 
  let plane := {p : ℝ × ℝ × ℝ // ∃ α β : ℝ, p = α • T.vertices v1 + β • T.vertices v2 + (1 - α - β) • T.vertices v3} in 
  ∃ center, center ∈ plane :=
sorry

end probability_plane_passes_through_center_l135_135713


namespace find_n_l135_135218

variable (P : ℕ → ℝ) (n : ℕ)

def polynomialDegree (P : ℕ → ℝ) (deg : ℕ) : Prop :=
  ∀ k, k > deg → P k = 0

def zeroValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n + 1)).map (λ k => 2 * k) → P i = 0

def twoValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n)).map (λ k => 2 * k + 1) → P i = 2

def specialValue (P : ℕ → ℝ) (n : ℕ) : Prop :=
  P (2 * n + 1) = -30

theorem find_n :
  (∃ n, polynomialDegree P (2 * n) ∧ zeroValues P n ∧ twoValues P n ∧ specialValue P n) →
  n = 2 :=
by
  sorry

end find_n_l135_135218


namespace geometric_sequence_term_50_l135_135897

theorem geometric_sequence_term_50 (a r : ℤ) (h₁ : a = 5) (h₂ : a * r = -15) : 
  let r : ℤ := -3 in
  a * r ^ 49 = -5 * 3 ^ 49 :=
by
  sorry

end geometric_sequence_term_50_l135_135897


namespace magic_square_sum_l135_135277

theorem magic_square_sum (square : matrix (fin 3) (fin 3) (fin 10)) (h : ∀ i j, square i j < 10) 
  (distinct : ∀ i1 j1 i2 j2, (i1 ≠ i2 ∨ j1 ≠ j2) → square i1 j1 ≠ square i2 j2) :
  ∀ i, (∑ j, square i j) = 15 :=
by
  -- Conditions for the integers from 1 to 9
  let nums := finset.range 9  -- {0, 1, ..., 8} corresponds to {1, 2, ..., 9}
  have one_to_nine : ∀ i j, 1 ≤ square i j ∧ square i j ≤ 9 := 
    λ i j, ⟨nat.succ_le_of_lt (fin.is_lt (square i j)), nat.lt_succ_iff.mp (h i j)⟩

  -- The condition for all distinct elements
  have injective_square : function.injective (λ (p : fin 3 × fin 3), square p.1 p.2) :=
    by
      rintro ⟨i1, j1⟩ ⟨i2, j2⟩ H
      by_cases h_cases : i1 = i2 ∧ j1 = j2
      . exact congr_arg _ h_cases
      . suffices (square i1 j1 = square i2 j2) by
          exact (distinct i1 j1 i2 j2 h_cases.elim_lt.left) (this)
        exact H

  -- The condition for sums being equal
  have rows_sum : ∀ i, ∑ j, (square i j) = (∑ i in (nums.image square), id) / 3 :=
      by sorry

  -- Calculating final result using given conditions
  have total_sum : (∑ x in (nums.image (id : fin 10 → ℤ)), x) = 45 := by sorry

  have val_per_row :  (45 : ℤ) / (3 : ℤ) = 15 := by norm_num

  exact sorry

end magic_square_sum_l135_135277


namespace delta_value_l135_135465

noncomputable def delta : ℝ :=
  Real.arccos (
    (Finset.range 3600).sum (fun k => Real.sin ((2539 + k) * Real.pi / 180)) ^ Real.cos (2520 * Real.pi / 180) +
    (Finset.range 3599).sum (fun k => Real.cos ((2521 + k) * Real.pi / 180)) +
    Real.cos (6120 * Real.pi / 180)
  )

theorem delta_value : delta = 71 :=
by
  sorry

end delta_value_l135_135465


namespace determine_range_of_a_l135_135865

variables {a : ℝ}
def f (x : ℝ) := sin x ^ 3 + a * cos x ^ 2

theorem determine_range_of_a (h : ∃ x ∈ Ioo 0 π, is_min_on f (Ioo 0 π) x) : 0 < a :=
sorry

end determine_range_of_a_l135_135865


namespace P_at_neg_one_l135_135994

noncomputable def P (x : ℝ)
  (a b c d e f: ℝ) : ℝ :=
  (4*x^4 - 52*x^3 + a*x^2 + b*x + c) * (10*x^4 - 160*x^3 + d*x^2 + e*x + f)

theorem P_at_neg_one (a b c d e f : ℝ)
  (h_roots : ∀ μ ∈ {1, 2, 3, 4, 5, -1} : set ℝ, is_root (P x a b c d e f) μ) :
  P (-1) a b c d e f = 2494800 :=
begin
  sorry
end

end P_at_neg_one_l135_135994


namespace shaded_area_ratio_l135_135045

theorem shaded_area_ratio (area_large_square : ℝ) (area_small_square : ℝ)
  (grid_size: ℕ) (shaded_squares: ℕ):
  area_large_square = 81 ∧
  area_small_square = 1 ∧
  grid_size = 9 ∧
  shaded_squares = 45 →
  (shaded_squares * area_small_square / area_large_square) = 5 / 9 :=
by 
  intros h
  cases h with h1 h
  cases h with h2 h
  cases h with h3 h4
  rw [h1, h2]
  exact congr_arg2 (/) rfl (by linarith : area_large_square = 81)
  simplify
  norm_num
  sorry

end shaded_area_ratio_l135_135045


namespace positive_integers_satisfying_inequality_l135_135453

-- Define the assertion that there are exactly 5 positive integers x satisfying the given inequality
theorem positive_integers_satisfying_inequality :
  (∃! x : ℕ, 4 < x ∧ x < 10 ∧ (10 * x)^4 > x^8 ∧ x^8 > 2^16) :=
sorry

end positive_integers_satisfying_inequality_l135_135453


namespace largest_of_a_b_c_l135_135122

noncomputable def a : ℝ := 1 / 2
noncomputable def b : ℝ := Real.log 3 / Real.log 4
noncomputable def c : ℝ := Real.sin (Real.pi / 8)

theorem largest_of_a_b_c : b = max (max a b) c :=
by
  have ha : a = 1 / 2 := rfl
  have hb : b = Real.log 3 / Real.log 4 := rfl
  have hc : c = Real.sin (Real.pi / 8) := rfl
  sorry

end largest_of_a_b_c_l135_135122


namespace part1_part2_l135_135926

theorem part1 (A B : ℝ) (h1 : A = 2 * B) : C = 5 * Real.pi / 8 :=
sorry

theorem part2 (a b c A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) :
   2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l135_135926


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135340

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n % 9 = 0 ∧ (n.digits.sum = 27) ∧
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ m % 9 = 0 ∧ (m.digits.sum = 27) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135340


namespace person_A_money_left_l135_135234

-- We define the conditions and question in terms of Lean types.
def initial_money_ratio : ℚ := 7 / 6
def money_spent_A : ℚ := 50
def money_spent_B : ℚ := 60
def final_money_ratio : ℚ := 3 / 2
def x : ℚ := 30

-- The theorem to prove the amount of money left by person A
theorem person_A_money_left 
  (init_ratio : initial_money_ratio = 7 / 6)
  (spend_A : money_spent_A = 50)
  (spend_B : money_spent_B = 60)
  (final_ratio : final_money_ratio = 3 / 2)
  (hx : x = 30) : 3 * x = 90 := by 
  sorry

end person_A_money_left_l135_135234


namespace smallest_prime_not_expressible_l135_135086

theorem smallest_prime_not_expressible (a b : ℕ) :
  ¬ (∃ a b : ℕ, 41 = |3^a - 2^b|) ∧ 
  (∀ p : ℕ, Prime p → p < 41 → ∃ a b : ℕ, p = |3^a - 2^b|) :=
  by 
    sorry

end smallest_prime_not_expressible_l135_135086


namespace sum_of_angles_l135_135702

theorem sum_of_angles (α β : ℝ) (hα : α = 20) (hβ : β = 30) : 
    let θ1 := 2 * α in
    let θ2 := 2 * β in
    let φ1 := θ1 / 2 in
    let φ2 := θ2 / 2 in
    φ1 + φ2 = 50 :=
by
    intros
    rw [hα, hβ]
    dsimp
    norm_num
    sorry 

end sum_of_angles_l135_135702


namespace condition_inequality_l135_135123

theorem condition_inequality (x y : ℝ) :
  (¬ (x ≤ y → |x| ≤ |y|)) ∧ (¬ (|x| ≤ |y| → x ≤ y)) :=
by
  sorry

end condition_inequality_l135_135123


namespace find_constant_l135_135469

theorem find_constant (m : ℕ) (h1 : ∀ m, m % 2 = 1 -> [m] = 3 * m)
                      (h2 : ∀ m, m % 2 = 0 -> [m] = constant * m)
                      (h3 : [5] * [6] = 15) : constant = 1 / 6 :=
by
  sorry

end find_constant_l135_135469


namespace intersection_A_B_l135_135492

noncomputable def A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 ≤ 0}
noncomputable def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l135_135492


namespace annual_savings_l135_135227

-- defining the conditions
def current_speed := 10 -- in Mbps
def current_bill := 20 -- in dollars
def bill_30Mbps := 2 * current_bill -- in dollars
def bill_20Mbps := current_bill + 10 -- in dollars
def months_in_year := 12

-- calculating the annual costs
def annual_cost_30Mbps := bill_30Mbps * months_in_year
def annual_cost_20Mbps := bill_20Mbps * months_in_year

-- statement of the problem
theorem annual_savings : (annual_cost_30Mbps - annual_cost_20Mbps) = 120 := by
  sorry -- prove the statement

end annual_savings_l135_135227


namespace additional_money_required_l135_135042

   theorem additional_money_required (patricia_money lisa_money charlotte_money total_card_cost : ℝ) 
       (h1 : patricia_money = 6)
       (h2 : lisa_money = 5 * patricia_money)
       (h3 : lisa_money = 2 * charlotte_money)
       (h4 : total_card_cost = 100) :
     (total_card_cost - (patricia_money + lisa_money + charlotte_money) = 49) := 
   by
     sorry
   
end additional_money_required_l135_135042


namespace harmonic_mean_2_3_6_l135_135433

-- Define the harmonic mean function for three numbers
def harmonic_mean (a b c : ℝ) : ℝ :=
  3 / (1 / a + 1 / b + 1 / c)

-- Prove that the harmonic mean of 2, 3, and 6 is equal to 3
theorem harmonic_mean_2_3_6 : harmonic_mean 2 3 6 = 3 :=
by
  -- This statement is enough to state the theorem
  sorry

end harmonic_mean_2_3_6_l135_135433


namespace required_extra_money_l135_135040

theorem required_extra_money 
(Patricia_money Lisa_money Charlotte_money : ℕ) 
(hP : Patricia_money = 6) 
(hL : Lisa_money = 5 * Patricia_money) 
(hC : Lisa_money = 2 * Charlotte_money) 
(cost : ℕ) 
(hCost : cost = 100) : 
  cost - (Patricia_money + Lisa_money + Charlotte_money) = 49 := 
by 
  sorry

end required_extra_money_l135_135040


namespace cos_36_is_correct_l135_135441

noncomputable def cos_36_eq : Prop :=
  let b := Real.cos (Real.pi * 36 / 180)
  let a := Real.cos (Real.pi * 72 / 180)
  (a = 2 * b^2 - 1) ∧ (b = (1 + Real.sqrt 5) / 4)

theorem cos_36_is_correct : cos_36_eq :=
by sorry

end cos_36_is_correct_l135_135441


namespace discriminant_of_quadratic_l135_135772

def a := 5
def b := 5 + 1/5
def c := 1/5
def discriminant (a b c : ℚ) := b^2 - 4 * a * c

theorem discriminant_of_quadratic :
  discriminant a b c = 576 / 25 :=
by
  sorry

end discriminant_of_quadratic_l135_135772


namespace product_of_distinct_roots_l135_135857

theorem product_of_distinct_roots (x1 x2 : ℝ) (hx1 : x1 ^ 2 - 2 * x1 = 1) (hx2 : x2 ^ 2 - 2 * x2 = 1) (h_distinct : x1 ≠ x2) : 
  x1 * x2 = -1 := 
  sorry

end product_of_distinct_roots_l135_135857


namespace polygon_reciprocal_sum_inequality_l135_135884

theorem polygon_reciprocal_sum_inequality (n : ℕ) (A : Fin n → ℝ) (h_n_ge_3 : 3 ≤ n) :
  (∑ i, (1 / A i)) ≥ n^2 / ((n - 2) * Real.pi) :=
sorry

end polygon_reciprocal_sum_inequality_l135_135884


namespace volume_of_solid_l135_135745

def x_y_relation (x y : ℝ) : Prop := x = (y - 2)^(1/3)
def x1 (x : ℝ) : Prop := x = 1
def y1 (y : ℝ) : Prop := y = 1

theorem volume_of_solid :
  ∀ (x y : ℝ),
    (x_y_relation x y ∧ x1 x ∧ y1 y) →
    ∃ V : ℝ, V = (44 / 7) * Real.pi :=
by
  -- Proof will go here
  sorry

end volume_of_solid_l135_135745


namespace groupDivisionWays_l135_135265

-- Definitions based on conditions
def numDogs : ℕ := 12
def group1Size : ℕ := 4
def group2Size : ℕ := 5
def group3Size : ℕ := 3
def fluffy : ℕ := 1 -- Fluffy's assigned position
def nipper : ℕ := 2 -- Nipper's assigned position

-- Function to compute binomial coefficients
def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom (n+1) k

-- Theorem to prove the number of ways to form the groups
theorem groupDivisionWays :
  (binom 10 3 * binom 7 4) = 4200 :=
by
  sorry

end groupDivisionWays_l135_135265


namespace seating_arrangement_count_l135_135890

-- Define the conditions.
def chairs : ℕ := 7
def people : ℕ := 5
def end_chairs : ℕ := 3

-- Define the main theorem to prove the number of arrangements.
theorem seating_arrangement_count :
  (end_chairs * 2) * (6 * 5 * 4 * 3) = 2160 := by
  sorry

end seating_arrangement_count_l135_135890


namespace correct_propositions_l135_135830

-- Conditions
def prop1 := ∀ x : ℝ, cos(5 * π / 2 - 2 * x) = cos (5 * π / 2 - 2 * (-x))
def prop2 := ∀ x y : ℝ, (-π/4) ≤ x → x < y → y ≤ π/4 → sin (x + π/4) < sin (y + π/4)
def prop3 := ∀ x : ℝ, sin (2 * (π / 8 - x) + 5 * π / 4) = sin (2 * (π / 8 + x) + 5 * π / 4)
def prop4 := ∀ x : ℝ, cos (2 * (x + π / 3) - π / 3) = cos (2 * x)

-- Theorem
theorem correct_propositions : (if prop1 then 1 else 0) + (if prop2 then 1 else 0) + (if prop3 then 1 else 0) + (if prop4 then 1 else 0) = 2 := by
  sorry

end correct_propositions_l135_135830


namespace exists_solution_real_l135_135085

theorem exists_solution_real (m : ℝ) :
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5) ↔ m ≠ 3 / 2 :=
by
  sorry

end exists_solution_real_l135_135085


namespace water_polo_team_combinations_l135_135233

theorem water_polo_team_combinations : 
  let n : ℕ := 18 in
  let ways := n * (n - 1) * (Nat.factorial 16 / (Nat.factorial 10)) in
  ways = 2459528 :=
by
  sorry

end water_polo_team_combinations_l135_135233


namespace intersection_M_N_l135_135839

def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℤ := {x | x^2 < 2}

theorem intersection_M_N : (M ∩ (N : Set ℝ)) = ({0} : Set ℝ) :=
by
  sorry

end intersection_M_N_l135_135839


namespace wire_length_square_field_l135_135685

theorem wire_length_square_field (A : ℝ) (h : A = 69696) : 
  let s := Real.sqrt A in 
  let perimeter := 4 * s in 
  let wire_length := 15 * perimeter in 
  wire_length = 15840 :=
by
  sorry

end wire_length_square_field_l135_135685


namespace rocky_first_round_knockouts_l135_135614

-- Define the conditions
def total_fights : ℕ := 190
def knockout_percentage : ℝ := 0.50
def first_round_knockout_percentage : ℝ := 0.20

-- Calculate the number of total knockouts
def total_knockouts : ℕ := (knockout_percentage * total_fights).toNat

-- Calculate the number of first-round knockouts
def first_round_knockouts : ℕ := (first_round_knockout_percentage * total_knockouts).toNat

-- Main theorem to prove
theorem rocky_first_round_knockouts :
  first_round_knockouts = 19 :=
by sorry

end rocky_first_round_knockouts_l135_135614


namespace regression_line_equation_l135_135864

variables (x y : Type) [linear_ordered_field x] [linear_ordered_field y]

/-- Define the points in the experiment -/
def A := (1 : ℝ, 3 : ℝ)
def B := (2 : ℝ, 3.8 : ℝ)
def C := (3 : ℝ, 5.2 : ℝ)
def D := (4 : ℝ, 6 : ℝ)

/-- Calculate the mean of x -/
noncomputable def mean_x : ℝ := (1 + 2 + 3 + 4) / 4

/-- Calculate the mean of y -/
noncomputable def mean_y : ℝ := (3 + 3.8 + 5.2 + 6) / 4

/-- Calculate the slope of the regression line -/
noncomputable def slope_m : ℝ := ((1 - mean_x) * (3 - mean_y) + (2 - mean_x) * (3.8 - mean_y) + (3 - mean_x) * (5.2 - mean_y) + (4 - mean_x) * (6 - mean_y)) /
                          ((1 - mean_x)^2 + (2 - mean_x)^2 + (3 - mean_x)^2 + (4 - mean_x)^2)

/-- Calculate the y-intercept of the regression line -/
noncomputable def y_intercept_b : ℝ := mean_y - slope_m * mean_x

/-- Define the regression line equation -/
noncomputable def regression_line : x → ℝ := λ x, slope_m * x + y_intercept_b

/-- Prove the regression line matches the calculated line -/
theorem regression_line_equation : regression_line = λ (x : ℝ), 1.05 * x - 0.9 :=
by sorry

end regression_line_equation_l135_135864


namespace ben_remaining_money_l135_135028

/-- Ben's remaining money after business operations /-- 
theorem ben_remaining_money : 
  let initial_money := 2000
  let cheque := 600
  let debtor_payment := 800
  let maintenance_cost := 1200
  initial_money - cheque + debtor_payment - maintenance_cost = 1000 := 
by
  -- Initial money
  let initial_money := 2000
  -- Cheque amount
  let cheque := 600
  -- Debtor payment amount
  let debtor_payment := 800
  -- Maintenance cost
  let maintenance_cost := 1200
  -- Calculation
  have h₁ : initial_money - cheque = 2000 - 600 := by rfl
  let money_after_cheque := 2000 - 600
  have h₂ : money_after_cheque + debtor_payment = 1400 + 800 := by rfl
  let money_after_debtor := 1400 + 800
  have h₃ : money_after_debtor - maintenance_cost = 2200 - 1200 := by rfl
  let remaining_money := 2200 - 1200
  -- Assertion
  show remaining_money = 1000 from sorry

end ben_remaining_money_l135_135028


namespace farmer_animals_l135_135399

theorem farmer_animals : 
  ∃ g s : ℕ, 
    35 * g + 40 * s = 2000 ∧ 
    g = 2 * s ∧ 
    (0 < g ∧ 0 < s) ∧ 
    g = 36 ∧ s = 18 := 
by 
  sorry

end farmer_animals_l135_135399


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135354

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135354


namespace all_points_collinear_l135_135109

theorem all_points_collinear (points : Finset (ℝ × ℝ))
  (hcondition : ∀ {p1 p2 : ℝ × ℝ}, p1 ∈ points → p2 ∈ points → p1 ≠ p2 → 
    ∃ p3 ∈ points, p3 ≠ p1 ∧ p3 ≠ p2 ∧ collinear ℝ {p1, p2, p3}) :
  ∃ line : ℝ → ℝ → Prop, ∀ p ∈ points, line p :=
begin
  sorry
end

end all_points_collinear_l135_135109


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135321

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((nat.digits 10 n).sum = 27) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ ((nat.digits 10 m).sum = 27) → m ≤ n) :=
begin
  use 999,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intro m,
    intro hm,
    cases hm,
    cases hm_left,
    cases hm_left_left,
    cases hm_left_right,
    cases hm_right,
    sorry
  },
sorry,
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135321


namespace leo_words_per_line_l135_135585

def words_per_line
  (total_words : ℕ)
  (words_left : ℕ)
  (lines_per_page : ℕ)
  (pages_written : ℕ) : ℕ :=
  (total_words - words_left) / (lines_per_page * pages_written)

theorem leo_words_per_line {w : ℕ} :
  words_per_line 400 100 20 1.5 = 10 :=
by
  sorry

end leo_words_per_line_l135_135585


namespace lambda_condition_l135_135819

variables (λ : ℝ)

def i := (1 : ℝ, 0 : ℝ)
def j := (0 : ℝ, 1 : ℝ)

def a := (i.1, i.2 - 2)
def b := (i.1, i.2 + λ)

def acute_angle (a b : ℝ × ℝ) : Prop :=
  (a.1 * b.1 + a.2 * b.2 > 0) 

theorem lambda_condition (h: acute_angle a b):
  λ < 1/2 ∧ λ ≠ -2 := 
sorry

end lambda_condition_l135_135819


namespace part1_part2_l135_135924

theorem part1 (A B : ℝ) (h1 : A = 2 * B) : C = 5 * Real.pi / 8 :=
sorry

theorem part2 (a b c A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) :
   2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l135_135924


namespace correct_statements_l135_135681

-- Definition of non-zero vectors and collinearity
def non_zero_vector (v : Vec) : Prop := v ≠ 0
def collinear (v w : Vec) : Prop := ∃ k : ℝ, k ≠ 0 ∧ v = k • w

-- Definitions to evaluate the correctness of the statements
def statement_a (AB CD : Vec) : Prop := collinear AB CD → (∃ (A B C D : Point), same_line A B C D)
def statement_b (a b c: Vec) : Prop := (parallel a b ∧ parallel b c) → (parallel a c ∨ antiparallel a c)
def statement_c (AB CD: Vec) : Prop := non_zero_vector AB ∧ non_zero_vector CD ∧ collinear AB CD → (angle AB CD = 0 ∨ angle AB CD = 180)
def statement_d (v w: Vec) : Prop := (parallel v w ↔ collinear v w)

-- Main theorem stating that C and D are correct
theorem correct_statements :
  ∀ (AB CD a b c v w: Vec), 
  (statement_c AB CD) ∧ (statement_d v w) :=
by sorry

end correct_statements_l135_135681


namespace honors_students_count_l135_135656

variable {total_students : ℕ}
variable {total_girls total_boys : ℕ}
variable {honors_girls honors_boys : ℕ}

axiom class_size_constraint : total_students < 30
axiom prob_girls_honors : (honors_girls : ℝ) / total_girls = 3 / 13
axiom prob_boys_honors : (honors_boys : ℝ) / total_boys = 4 / 11
axiom total_students_eq : total_students = total_girls + total_boys
axiom honors_girls_value : honors_girls = 3
axiom honors_boys_value : honors_boys = 4

theorem honors_students_count : 
  honors_girls + honors_boys = 7 :=
by
  sorry

end honors_students_count_l135_135656


namespace sufficient_not_necessary_condition_l135_135818

variable {a b : ℝ} {f : ℝ → ℝ} (x : ℝ)

theorem sufficient_not_necessary_condition (h_diff : ∀ x ∈ set.Ioo a b, differentiable_at ℝ f x)
  (h_f_prime_neg : ∀ x ∈ set.Ioo a b, deriv f x < 0) :
  (∀ x1 x2 ∈ set.Ioo a b, x1 < x2 → f x1 > f x2) ∧
  (∃ g : ℝ → ℝ, (∀ x1 x2 ∈ set.Ioo a b, x1 < x2 → g x1 > g x2) ∧ 
    (∀ x ∈ set.Ioo a b, differentiable_at ℝ g x ∧ deriv g x ≤ 0)) :=
sorry

end sufficient_not_necessary_condition_l135_135818


namespace find_star_1993_1935_l135_135759

axiom star (x y : ℕ) : ℕ
axiom star_idempotent (x : ℕ) : star x x = 0
axiom star_assoc (x y z : ℕ) : star x (star y z) = star x y + z

theorem find_star_1993_1935 : star 1993 1935 = 58 :=
by
  sorry

end find_star_1993_1935_l135_135759


namespace OK_eq_3R_l135_135211

variables {A B C I L K O : Type*}
variables [non_isosceles_triangle A B C]
variables [incenter I A B C]
variables [circumcircle_radius A B C R]
variables [external_angle_bisector AL B C]
variables [point_on_perpendicular_bisector K B C]
variables [perpendicular IL IK]

theorem OK_eq_3R (h1 : triangle A B C)
                 (h2 : incenter I A B C)
                 (h3 : circumcircle_radius A B C R)
                 (h4 : external_angle_bisector AL ∠BAC B C)
                 (h5 : point_on_perpendicular_bisector K B C)
                 (h6 : perpendicular IL IK) :
    distance O K = 3 * R :=
sorry

end OK_eq_3R_l135_135211


namespace sqrt_fraction_simplification_l135_135037

theorem sqrt_fraction_simplification : (√18 - √2) / √2 = 2 := by
  sorry

end sqrt_fraction_simplification_l135_135037


namespace circumcenter_of_GHI_lies_on_l_l135_135800

theorem circumcenter_of_GHI_lies_on_l {A B C D E F G H I : Type} [Point A] [Point B] [Point C] [Point D] [Point E] [Point F] [Point G] [Point H] [Point I]
  (l : Line) (triangle : Triangle A B C) (on_D_l : OnLine D l) (on_E_l : OnLine E l) (on_F_l : OnLine F l)
  (D_on_BC : OnLine D (side BC triangle)) (E_on_CA : OnLine E (side CA triangle)) (F_on_AB : OnLine F (side AB triangle))
  (G_circumcenter_AEF : Circumcenter G (triangle A E F)) (H_circumcenter_BDF : Circumcenter H (triangle B D F)) (I_circumcenter_CDE : Circumcenter I (triangle C D E)) :
  OnLine (circumcenter (triangle G H I)) l :=
  sorry

end circumcenter_of_GHI_lies_on_l_l135_135800


namespace geometric_sequence_condition_l135_135190

theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) (h1 : 0 < a 1) 
  (h2 : ∀ n, a (n + 1) = a n * q) :
  (a 1 < a 3) ↔ (a 1 < a 3) ∧ (a 3 < a 6) :=
sorry

end geometric_sequence_condition_l135_135190


namespace fried_busy_frog_l135_135046

open ProbabilityTheory

def initial_position : (ℤ × ℤ) := (0, 0)

def possible_moves : List (ℤ × ℤ) := [(0, 0), (1, 0), (0, 1)]

def p (n : ℕ) (pos : ℤ × ℤ) : ℚ :=
  if pos = initial_position then 1 else 0

noncomputable def transition (n : ℕ) (pos : ℤ × ℤ) : ℚ :=
  if pos = (0, 0) then 1/3 * p n (0, 0)
  else if pos = (0, 1) then 1/3 * p n (0, 0) + 1/3 * p n (0, 1)
  else if pos = (1, 0) then 1/3 * p n (0, 0) + 1/3 * p n (1, 0)
  else 0

noncomputable def p_1 (pos : ℤ × ℤ) : ℚ := transition 0 pos

noncomputable def p_2 (pos : ℤ × ℤ) : ℚ := transition 1 pos

noncomputable def p_3 (pos : ℤ × ℤ) : ℚ := transition 2 pos

theorem fried_busy_frog :
  p_3 (0, 0) = 1/27 :=
by
  sorry

end fried_busy_frog_l135_135046


namespace Katka_polygon_perimeter_l135_135691

theorem Katka_polygon_perimeter :
  let rect_perimeter n := 2 * (n + (n + 1))
  let total_perimeter := 2 * (List.range 20).sum * List.range 21).sum
  total_perimeter = 880 := by
  sorry

end Katka_polygon_perimeter_l135_135691


namespace part1_part2_l135_135927

theorem part1 (A B : ℝ) (h1 : A = 2 * B) : C = 5 * Real.pi / 8 :=
sorry

theorem part2 (a b c A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) :
   2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l135_135927


namespace problem_part1_problem_part2_l135_135937

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l135_135937


namespace arithmetic_sequence_term_l135_135198

theorem arithmetic_sequence_term :
  ∀ a : ℕ → ℕ, (a 1 = 1) → (∀ n : ℕ, a (n + 1) - a n = 2) → (a 6 = 11) :=
by
  intros a h1 hrec
  sorry

end arithmetic_sequence_term_l135_135198


namespace modulus_of_z_l135_135987

theorem modulus_of_z (r k : ℝ) (z : ℂ) (hr : |r| < 2) (hk : |k| < 3)
  (hz : z + k * z⁻¹ = r) : |z| = real.sqrt ((r^2 - 2 * k) / 2) :=
by
  sorry

end modulus_of_z_l135_135987


namespace number_of_correct_propositions_l135_135657

open Set

def prop1 := {0} = (∅ : Set ℕ)
def prop2 := ∀ a : ℕ, a ∉ 0 :: ∅ → -a ∉ (Nat : Set ℤ)
def prop3 := (λ A : Set ℝ, ∀ x : ℝ, x^2 - 2 * x + 1 = 0 → x ∈ A) = {1}
def prop4 := {x : ℕ | 6 % x = 0}.Finite

theorem number_of_correct_propositions :
  (¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) ↔ 1 = (1 : ℕ) := by
  sorry

end number_of_correct_propositions_l135_135657


namespace shaded_fraction_is_four_fifteenths_l135_135719

noncomputable def shaded_fraction : ℚ :=
  let a := (1/4 : ℚ)
  let r := (1/16 : ℚ)
  a / (1 - r)

theorem shaded_fraction_is_four_fifteenths :
  shaded_fraction = (4 / 15 : ℚ) := sorry

end shaded_fraction_is_four_fifteenths_l135_135719


namespace minimum_attempts_to_match_codes_to_safes_l135_135294

theorem minimum_attempts_to_match_codes_to_safes 
  (n : ℕ) (h₁ : n = 7) 
  (safes : Fin n)
  (codes : Fin n)
  : (finset.range(n).sum (λ i, n - 1 - i) = 21) :=
by
  sorry

end minimum_attempts_to_match_codes_to_safes_l135_135294


namespace circle_line_distance_l135_135491

theorem circle_line_distance (m : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ abs (x + y - m / sqrt 2) = 1) ↔ -sqrt 2 < m ∧ m < sqrt 2 :=
by
  sorry

end circle_line_distance_l135_135491


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135322

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((nat.digits 10 n).sum = 27) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ ((nat.digits 10 m).sum = 27) → m ≤ n) :=
begin
  use 999,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intro m,
    intro hm,
    cases hm,
    cases hm_left,
    cases hm_left_left,
    cases hm_left_right,
    cases hm_right,
    sorry
  },
sorry,
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135322


namespace probability_unit_digit_nonzero_l135_135094

theorem probability_unit_digit_nonzero :
  let digits := {1, 2, 3, 4}
  let total_outcomes := Nat.choose 4 2
  let favorable_outcomes := 9
  favorable_outcomes / total_outcomes = 3 / 4 :=
by
  sorry

end probability_unit_digit_nonzero_l135_135094


namespace part1_part2_l135_135925

theorem part1 (A B : ℝ) (h1 : A = 2 * B) : C = 5 * Real.pi / 8 :=
sorry

theorem part2 (a b c A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) :
   2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l135_135925


namespace problem_l135_135859

def x : ℕ := 660
def percentage_25_of_x : ℝ := 0.25 * x
def percentage_12_of_1500 : ℝ := 0.12 * 1500
def difference_of_percentages : ℝ := percentage_12_of_1500 - percentage_25_of_x

theorem problem : difference_of_percentages = 15 := by
  -- begin proof (content replaced by sorry)
  sorry

end problem_l135_135859


namespace no_four_pairwise_perpendicular_lines_in_space_l135_135900

theorem no_four_pairwise_perpendicular_lines_in_space :
  ¬ ∃ (l1 l2 l3 l4: ℝ → ℝ × ℝ × ℝ),
    (∀ t, (l1 t).1 * (l2 t).1 + (l1 t).2 * (l2 t).2 + (l1 t).3 * (l2 t).3 = 0) ∧
    (∀ t, (l1 t).1 * (l3 t).1 + (l1 t).2 * (l3 t).2 + (l1 t).3 * (l3 t).3 = 0) ∧
    (∀ t, (l1 t).1 * (l4 t).1 + (l1 t).2 * (l4 t).2 + (l1 t).3 * (l4 t).3 = 0) ∧
    (∀ t, (l2 t).1 * (l3 t).1 + (l2 t).2 * (l3 t).2 + (l2 t).3 * (l3 t).3 = 0) ∧
    (∀ t, (l2 t).1 * (l4 t).1 + (l2 t).2 * (l4 t).2 + (l2 t).3 * (l4 t).3 = 0) ∧
    (∀ t, (l3 t).1 * (l4 t).1 + (l3 t).2 * (l4 t).2 + (l3 t).3 * (l4 t).3 = 0) := sorry

end no_four_pairwise_perpendicular_lines_in_space_l135_135900


namespace sum_of_coefficients_zero_l135_135446

theorem sum_of_coefficients_zero (A B C D E F : ℝ) :
  (∀ x : ℝ,
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
by
  intro h
  -- Proof omitted
  sorry

end sum_of_coefficients_zero_l135_135446


namespace james_collected_on_first_day_l135_135903

-- Conditions
variables (x : ℕ) -- the number of tins collected on the first day
variable (h1 : 500 = x + 3 * x + (3 * x - 50) + 4 * 50) -- total number of tins collected

-- Theorem to be proved
theorem james_collected_on_first_day : x = 50 :=
by
  sorry

end james_collected_on_first_day_l135_135903


namespace square_pond_side_length_l135_135712

-- Define the conditions
def length_garden : ℕ := 15
def width_garden : ℕ := 10
def original_area : ℕ := length_garden * width_garden
def remaining_area : ℕ := original_area / 2

/-- Given the original area of a rectangular garden and the requirement that the remaining area
after building a square pond is half of the original area, prove the side length of the square pond. -/
theorem square_pond_side_length (x : ℝ) (hx : x^2 = 75) : x = 5 * real.sqrt 3 :=
by
  sorry

end square_pond_side_length_l135_135712


namespace line_through_circumcenter_l135_135886

variables {A B C H D E F O : Type} [triangle : triangle A B C]
variables [is_acute_triangle : acute_triangle A B C]
variables [altitude : altitude_from B H A B C]
variables [midpoint_D : midpoint D A B]
variables [midpoint_E : midpoint E A C]
variables [reflection_F : reflection H F (line D E)]
variables [circumcenter : circumcenter A B C O]
variables [line_BF : line_through B F O]

theorem line_through_circumcenter : passes_through (line B F) O :=
sorry

end line_through_circumcenter_l135_135886


namespace max_triangle_area_from_intersecting_line_tangent_and_curve_l135_135106

open Real EuclideanGeometry

section
variables {x y r m : ℝ}
def circle (r : ℝ) := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = r ^ 2}
def line_tangent (a b c : ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
def curve_trajectory (x y : ℝ) := x ^ 2 / 4 + y ^ 2 = 1
def perp_distance (l : ℝ) : ℝ := l / 2
def triangle_area (m : ℝ) : ℝ := 2 * sqrt (m ^ 2 * (13 - m ^ 2)) / 13

theorem max_triangle_area_from_intersecting_line_tangent_and_curve
    (r : ℝ) (h₁ : r = 2)
    (h₂ : ∀ p ∈ circle r, ∃ B : ℝ × ℝ, B.1 = p.1 ∧ B.2 = 0 ∧ AB ⊥ ⊥-axis ∧ ∃ N, 2 * NB = AB)
    (h₃ : max_area = 1) :
  curve_trajectory x y ∧ max (triangle_area m) = 1 := 
begin
  -- No proof needed, as this is a statement only.
  sorry
end

end max_triangle_area_from_intersecting_line_tangent_and_curve_l135_135106


namespace sum_of_coefficients_of_polynomial_equals_neg_two_l135_135476

theorem sum_of_coefficients_of_polynomial_equals_neg_two {x : ℝ} :
  let a := fun (n : ℕ) => (1-2*x)^n.coeff n in
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7) = -2 := by
  let a := fun (n : ℕ) => if n = 0 then 1 else (1-2) ^ n
  sorry

end sum_of_coefficients_of_polynomial_equals_neg_two_l135_135476


namespace solve_crime_12_days_l135_135057

noncomputable def solve_crime_within_12_days (total_people criminal witness: ℕ) (invite : ℕ → set ℕ) : Prop :=
total_people = 80 ∧
(criminal < total_people) ∧
(witness < total_people) ∧
(∀ day: ℕ, day < 12 → 
  (invite day).subset (set.range total_people) → 
  (witness ∈ invite day ∧ criminal ∉ invite day) →
  ∃ criminal: ℕ, criminal ∈ set.range total_people)

theorem solve_crime_12_days :
  ∃ (invite : ℕ → set ℕ), solve_crime_within_12_days 80 criminal witness invite :=
sorry

end solve_crime_12_days_l135_135057


namespace wendi_chickens_l135_135301

theorem wendi_chickens : 
  let initial_chickens := 4
  let doubled_chickens := initial_chickens * 2
  let after_dog := doubled_chickens - 1
  let found_chickens := 10 - 4
  let total_chickens := after_dog + found_chickens
  in total_chickens = 13 :=
by
  let initial_chickens := 4
  let doubled_chickens := initial_chickens * 2
  let after_dog := doubled_chickens - 1
  let found_chickens := 10 - 4
  let total_chickens := after_dog + found_chickens
  sorry

end wendi_chickens_l135_135301


namespace reassemble_into_square_conditions_l135_135004

noncomputable def graph_paper_figure : Type := sorry
noncomputable def is_cuttable_into_parts (figure : graph_paper_figure) (parts : ℕ) : Prop := sorry
noncomputable def all_parts_are_triangles (figure : graph_paper_figure) (parts : ℕ) : Prop := sorry
noncomputable def can_reassemble_to_square (figure : graph_paper_figure) : Prop := sorry

theorem reassemble_into_square_conditions :
  ∀ (figure : graph_paper_figure), 
  (is_cuttable_into_parts figure 4 ∧ can_reassemble_to_square figure) ∧ 
  (is_cuttable_into_parts figure 5 ∧ all_parts_are_triangles figure 5 ∧ can_reassemble_to_square figure) :=
sorry

end reassemble_into_square_conditions_l135_135004


namespace Carol_cleaning_time_l135_135416

theorem Carol_cleaning_time 
(Alice_time : ℕ) 
(Bob_time : ℕ) 
(Carol_time : ℕ) 
(h1 : Alice_time = 40) 
(h2 : Bob_time = 3 * Alice_time / 4) 
(h3 : Carol_time = 2 * Bob_time) :
  Carol_time = 60 := 
sorry

end Carol_cleaning_time_l135_135416


namespace proof_problem_l135_135752

def op1 (a b : ℝ) : ℝ := (a * b) / (a + b)
def op2 (a b : ℝ) : ℝ := (a - b) / (a / b)

theorem proof_problem : op2 (op1 6 4) 1.2 = 0.6 := by
  sorry

end proof_problem_l135_135752


namespace max_value_of_f_l135_135279

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^4 + 2*x^2 + 3

-- State the theorem: the maximum value of f(x) is 4
theorem max_value_of_f : ∃ x : ℝ, f x = 4 := sorry

end max_value_of_f_l135_135279


namespace triangle_BKL_is_isosceles_l135_135213

noncomputable theory -- Declare non-computable due to geometric constructs potentially needing classical logic

-- Define the triangle and points
variables (A B C H K L : Point)
variables (d : Line)

-- Conditions
def is_tangent_at_B (d : Line) (circumcircle : Circle) (B : Point) : Prop := 
  tangent_to_circle d circumcircle B

def is_orthocenter (H : Point) (A B C : Point) : Prop := 
  orthocenter_of_triangle H A B C

def orthogonal_projection (H : Point) (d : Line) (K : Point) : Prop := 
  orthogonal_projection_on_line H d K

def is_midpoint (L : Point) (A C : Point) : Prop := 
  is_midpoint_of_segment L A C

-- Main theorem
theorem triangle_BKL_is_isosceles 
  (circumcircle : Circle)
  (h_tangent : is_tangent_at_B d circumcircle B)
  (h_orthocenter : is_orthocenter H A B C)
  (h_projection : orthogonal_projection H d K)
  (h_midpoint : is_midpoint L A C)
  : is_isosceles_triangle B K L :=
sorry -- Proof is not required and is skipped

end triangle_BKL_is_isosceles_l135_135213


namespace tangent_lines_y_eq_x_sq_through_P_l135_135753

theorem tangent_lines_y_eq_x_sq_through_P (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  ∃ m b, (y = m * x + b ∨ y = m * x + b) ∧ ((m = 2 ∧ b = -1) ∨ (m = 10 ∧ b = -25)) :=
begin
  sorry,
end

end tangent_lines_y_eq_x_sq_through_P_l135_135753


namespace find_pairs_satisfying_conditions_l135_135071

theorem find_pairs_satisfying_conditions :
  ∀ (m n : ℕ), (0 < m ∧ 0 < n) →
               (∃ k : ℤ, m^2 - 4 * n = k^2) →
               (∃ l : ℤ, n^2 - 4 * m = l^2) →
               (m = 4 ∧ n = 4) ∨ (m = 5 ∧ n = 6) ∨ (m = 6 ∧ n = 5) :=
by
  intros m n hmn h1 h2
  sorry

end find_pairs_satisfying_conditions_l135_135071


namespace tiling_ratio_l135_135115

theorem tiling_ratio (n a b : ℕ) (ha : a ≠ 0) (H : b = a * 2^(n/2)) :
  b / a = 2^(n/2) :=
  by
  sorry

end tiling_ratio_l135_135115


namespace sports_but_not_literary_l135_135001

theorem sports_but_not_literary:
  ∀ (N S L E X: ℕ), 
  N = 60 → 
  S = 28 →
  L = 26 →
  E = 12 →
  X = S - (N - E - L) →
  X = 22 :=
by
  intros N S L E X hN hS hL hE hX
  rw [hN, hS, hL, hE] at hX
  have h_total := N - E
  rw [h_total] at hX
  have h_overlap := h_total - L
  rw [h_overlap] at hX
  prf_sorry


end sports_but_not_literary_l135_135001


namespace find_C_prove_relation_l135_135954

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A), and A = 2B,
prove that C = 5/8 * π. -/
theorem find_C
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  C = ⅝ * Real.pi :=
sorry

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A),
prove that 2 * a ^ 2 = b ^ 2 + c ^ 2. -/
theorem prove_relation
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
sorry

end find_C_prove_relation_l135_135954


namespace compute_g_l135_135055

def g (x : ℝ) : ℝ :=
if x ≥ 0 then -2 * x ^ 2 + 1 else 2 * x + 9

theorem compute_g (h : g (g (g (g (g (g 2))))) = -185) : h := 
by 
  sorry

end compute_g_l135_135055


namespace least_number_remainder_l135_135075

theorem least_number_remainder (n : ℕ) (hn : n = 115) : n % 38 = 1 ∧ n % 3 = 1 := by
  sorry

end least_number_remainder_l135_135075


namespace adam_tickets_left_l135_135735

-- Define the initial number of tickets, cost per ticket, and total spending on the ferris wheel
def initial_tickets : ℕ := 13
def cost_per_ticket : ℕ := 9
def total_spent : ℕ := 81

-- Define the number of tickets Adam has after riding the ferris wheel
def tickets_left (initial_tickets cost_per_ticket total_spent : ℕ) : ℕ :=
  initial_tickets - (total_spent / cost_per_ticket)

-- Proposition to prove that Adam has 4 tickets left
theorem adam_tickets_left : tickets_left initial_tickets cost_per_ticket total_spent = 4 :=
by
  sorry

end adam_tickets_left_l135_135735


namespace sum_first_13_terms_l135_135565

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n m : ℕ, a n = a m + (n - m) * d

theorem sum_first_13_terms (h : is_arithmetic_sequence a) (h2 : a 3 + a 11 = 6) :
  let S_13 := (13 / 2) * (a 1 + a 13) in
  S_13 = 39 :=
by
  sorry

end sum_first_13_terms_l135_135565


namespace mr_white_flower_count_l135_135605

theorem mr_white_flower_count :
  ∀ (steps_length : ℕ) (steps_width : ℕ) (step_in_feet : ℕ) (area_per_flower : ℕ),
  steps_length = 18 →
  steps_width = 24 →
  step_in_feet = 3 →
  area_per_flower = 2 →
  let feet_length := steps_length * step_in_feet in
  let feet_width := steps_width * step_in_feet in
  let area := feet_length * feet_width in
  let flowers := area / area_per_flower in
  flowers = 1944 :=
begin
  intros steps_length steps_width step_in_feet area_per_flower h_steps_length h_steps_width h_step_in_feet h_area_per_flower,
  let feet_length := steps_length * step_in_feet,
  let feet_width := steps_width * step_in_feet,
  let area := feet_length * feet_width,
  let flowers := area / area_per_flower,
  rw [h_steps_length, h_steps_width, h_step_in_feet, h_area_per_flower],
  simp,
  sorry
end

end mr_white_flower_count_l135_135605


namespace minimum_points_to_win_l135_135559

theorem minimum_points_to_win (race_points : Fin 4 → ℕ) (h_no_ties : ∀ i j, i ≠ j → race_points i ≠ race_points j) :
  (∀ s : Fin 4 → ℕ, s = race_points → (sum (Fin4 → race_points) ≥ 13 → ∃ i, s i = (4 + 4 + 4 + 1))) :=
sorry

end minimum_points_to_win_l135_135559


namespace min_value_expression_geq_twosqrt3_l135_135827

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  (1/(x-1)) + (3/(y-1))

theorem min_value_expression_geq_twosqrt3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : (1/x) + (1/y) = 1) : 
  min_value_expression x y >= 2 * Real.sqrt 3 :=
by
  sorry

end min_value_expression_geq_twosqrt3_l135_135827


namespace trigonometric_inequality_in_triangle_l135_135573

theorem trigonometric_inequality_in_triangle
  (A B C : ℝ)
  (h1 : A + B + C = π)
  (h2 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h3 : A < π ∧ B < π ∧ C < π) :
  (cos ((B - C) / 2) / cos (A / 2)) + 
  (cos ((C - A) / 2) / cos (B / 2)) + 
  (cos ((A - B) / 2) / cos (C / 2)) 
  ≤ 2 * (cot A + cot B + cot C) :=
begin
  sorry
end

end trigonometric_inequality_in_triangle_l135_135573


namespace area_A0B0C0_eq_2_area_AC1BA1CB1_area_A0B0C0_geq_4_area_ABC_l135_135386

-- Definitions of geometric entities
variables {ABC A1 B1 C1 A0 B0 C0 : Type}
variables [circumcircle ABC] [angle_bisectors A B C] [External_angle_bisectors B C]
variables (A0 : meets_externally ABC A1) (B0 : meets_externally ABC B1) (C0 : meets_externally ABC C1)

-- Main theorems to prove
theorem area_A0B0C0_eq_2_area_AC1BA1CB1 :
  area A0B0C0 = 2 * area AC1BA1CB1 :=
sorry

theorem area_A0B0C0_geq_4_area_ABC :
  area A0B0C0 ≥ 4 * area ABC :=
sorry

end area_A0B0C0_eq_2_area_AC1BA1CB1_area_A0B0C0_geq_4_area_ABC_l135_135386


namespace minimum_value_g_on_interval_l135_135796

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x ^ 3 + (1 - a) / 2 * x ^ 2 - a * x - a

noncomputable def g (a t : ℝ) : ℝ := f a (-1) - f a t

theorem minimum_value_g_on_interval (a : ℝ) (h : a = 1) :
  ∃ t ∈ set.Icc (-3 : ℝ) (-1), g a t = 4 / 3 :=
begin
  sorry
end

end minimum_value_g_on_interval_l135_135796


namespace part1_part2_l135_135921

theorem part1 (A B : ℝ) (h1 : A = 2 * B) : C = 5 * Real.pi / 8 :=
sorry

theorem part2 (a b c A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) :
   2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l135_135921


namespace functional_equation_solution_l135_135592

noncomputable def f : ℝ → ℝ :=
  λ x, -3 * x^3 + 3 * x^2 - 6 * x + 2

theorem functional_equation_solution :
  (∀ x y : ℝ, f(x * y) = f((x^3 + y^3) / 2) + 3 * (x - y)^2) ∧ (f 0 = 2) ↔
  (∀ x : ℝ, f(x) = -3 * x^3 + 3 * x^2 - 6 * x + 2) :=
begin
  sorry
end

end functional_equation_solution_l135_135592


namespace min_num_stamps_is_17_l135_135375

-- Definitions based on problem conditions
def initial_num_stamps : ℕ := 2 + 5 + 3 + 1
def initial_cost : ℝ := 2 * 0.10 + 5 * 0.20 + 3 * 0.50 + 1 * 2
def remaining_cost : ℝ := 10 - initial_cost
def additional_stamps : ℕ := 2 + 2 + 1 + 1
def total_stamps : ℕ := initial_num_stamps + additional_stamps

-- Proof that the minimum number of stamps bought is 17
theorem min_num_stamps_is_17 : total_stamps = 17 := by
  sorry

end min_num_stamps_is_17_l135_135375


namespace question_1_question_2_l135_135869

variable (A B C a b c : ℝ)

-- Condition: In \( \triangle ABC \), sides opposite to \(A\), \(B\), and \(C\) are \(a\), \(b\), and \(c\) respectively.
-- Condition: \(\cos B = \frac{4}{5}\).
axiom cos_B : cos B = 4 / 5

-- Question 1: If \( c = 2a \), find the value of \( \frac{\sin B}{\sin C} \).
theorem question_1 (h1 : c = 2 * a) : sin B / sin C = 3 * sqrt 5 / 10 := by
  sorry

-- Question 2: If \( C - B = \frac{\pi}{4} \), find the value of \( \sin A \).
theorem question_2 (h2 : C - B = Real.pi / 4) : sin A = 31 * Real.sqrt 2 / 50 := by
  sorry

end question_1_question_2_l135_135869


namespace cos_alpha_beta_value_l135_135740

noncomputable def cos_alpha_beta (α β : ℝ) : ℝ :=
  Real.cos (α + β)

theorem cos_alpha_beta_value (α β : ℝ)
  (h1 : Real.cos α - Real.cos β = -3/5)
  (h2 : Real.sin α + Real.sin β = 7/4) :
  cos_alpha_beta α β = -569/800 :=
by
  sorry

end cos_alpha_beta_value_l135_135740


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135358

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ 100 ≤ n ∧ n < 1000 ∧ (9 ∣ n) ∧ (∑ digit in n.digits, digit = 27) :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135358


namespace chord_length_intercepted_by_circle_on_line_l135_135708

variables {t : ℝ}

def param_line_x (t : ℝ) : ℝ := -2 + t
def param_line_y (t : ℝ) : ℝ := 1 - t

def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 25

theorem chord_length_intercepted_by_circle_on_line :
  ∃ length : ℝ, 
    length = √82 ∧ 
    ∃ t₁ t₂ : ℝ, 
      (param_line_x t₁, param_line_y t₁) ∈ {p : ℝ × ℝ | circle_eq p.1 p.2} ∧ 
      (param_line_x t₂, param_line_y t₂) ∈ {p : ℝ × ℝ | circle_eq p.1 p.2} :=
begin
  sorry
end

end chord_length_intercepted_by_circle_on_line_l135_135708


namespace probability_xi_12_l135_135555

noncomputable def P_xi_eq_12 : ℝ :=
  let p_red := 3 / 8
  let p_white := 5 / 8
  let n := 12
  let k := 10
  let x := 11
  let y := 9
  nat.choose x y * (p_red ^ y) * (p_white ^ (x - y)) * p_red

theorem probability_xi_12 :
  let p_red := 3 / 8
  let p_white := 5 / 8
  let n := 12
  let k := 10
  let x := 11
  let y := 9
  let P_xi := nat.choose x y * (p_red ^ y) * (p_white ^ (x - y)) * p_red
  P_xi = C_{11}^{9} \cdot \left(\dfrac{3}{8}\right)^{9} \cdot \left(\dfrac{5}{8}\right)^{2} \cdot \dfrac{3}{8} := by
  sorry

end probability_xi_12_l135_135555


namespace arrangement_count_is_48_l135_135828

theorem arrangement_count_is_48 :
  let persons := ["A", "B", "甲", "乙", "丙"]
  let A := persons[0]
  let B := persons[1]
  let adults := persons.drop 2
  let arrangements_count (line : List String) : Nat := 
    if (line.any (· = A) && line.take 1 ≠ A && line.reverse.take 1 ≠ A &&
      (line.filter (fun x => x ∈ adults)).count_within_two = 2)
    then 1 else 0
  in List.permutations persons |>.sum arrangements_count = 48 :=
sorry

end arrangement_count_is_48_l135_135828


namespace one_non_congruent_triangle_with_perimeter_10_l135_135158

def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 10

def are_non_congruent (a b c : ℕ) (x y z : ℕ) : Prop :=
  ¬ (a = x ∧ b = y ∧ c = z ∨ a = x ∧ b = z ∧ c = y ∨ a = y ∧ b = x ∧ c = z ∨ 
     a = y ∧ b = z ∧ c = x ∨ a = z ∧ b = x ∧ c = y ∨ a = z ∧ b = y ∧ c = x)

theorem one_non_congruent_triangle_with_perimeter_10 :
  ∃ a b c : ℕ, is_valid_triangle a b c ∧ perimeter a b c ∧
  ∀ x y z : ℕ, is_valid_triangle x y z ∧ perimeter x y z → are_non_congruent a b c x y z → false :=
sorry

end one_non_congruent_triangle_with_perimeter_10_l135_135158


namespace colored_ints_square_diff_l135_135022

-- Define a coloring function c as a total function from ℤ to a finite set {0, 1, 2}
def c : ℤ → Fin 3 := sorry

-- Lean 4 statement for the problem
theorem colored_ints_square_diff : 
  ∃ a b : ℤ, a ≠ b ∧ c a = c b ∧ ∃ k : ℤ, a - b = k ^ 2 :=
sorry

end colored_ints_square_diff_l135_135022


namespace value_range_f_l135_135652

def f (x : ℝ) : ℝ :=
  Matrix.det ![![2, real.cos x], ![real.sin x, -1]]

theorem value_range_f :
  ∀ x : ℝ, 
  -5/2 ≤ f x ∧ f x ≤ -3/2 :=
by
  sorry

end value_range_f_l135_135652


namespace family_ate_doughnuts_l135_135395

variable (box_initial : ℕ) (box_left : ℕ) (dozen : ℕ)

-- Define the initial and remaining conditions
def dozen_value : ℕ := 12
def box_initial_value : ℕ := 2 * dozen_value
def doughnuts_left_value : ℕ := 16

theorem family_ate_doughnuts (h1 : box_initial = box_initial_value) (h2 : box_left = doughnuts_left_value) :
  box_initial - box_left = 8 := by
  -- h1 says the box initially contains 2 dozen, which is 24.
  -- h2 says that there are 16 doughnuts left.
  sorry

end family_ate_doughnuts_l135_135395


namespace ends_with_six_l135_135117

theorem ends_with_six (M N : ℕ) (h1 : M > 10) (h2 : N > 10) (h3 : digit_count M = digit_count N) (h4 : M = 3 * N)
  (h5 : ∃(k : ℕ), (M = N + 2 * 10 ^ k ∧ ∀ i, i ≠ k -> ∃ (d_i r_i : ℕ), r_i % 2 = 1 ∧ M.digits.get i = (N.digits.get i + r_i)))
  : N % 10 = 6 :=
by
  sorry

end ends_with_six_l135_135117


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135339

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n % 9 = 0 ∧ (n.digits.sum = 27) ∧
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ m % 9 = 0 ∧ (m.digits.sum = 27) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135339


namespace woman_weaves_ten_day_units_l135_135189

theorem woman_weaves_ten_day_units 
  (a₁ d : ℕ)
  (h₁ : 4 * a₁ + 6 * d = 24)
  (h₂ : a₁ + 6 * d = a₁ * (a₁ + d)) :
  a₁ + 9 * d = 21 := 
by
  sorry

end woman_weaves_ten_day_units_l135_135189


namespace eight_points_in_circle_distance_lt_one_l135_135206

noncomputable theory

open set
open metric

def circle : set (euclidean_space (fin 2)) :=
  {p | ∥p∥ ≤ 1}

theorem eight_points_in_circle_distance_lt_one
  (ps : fin 8 → euclidean_space (fin 2))
  (hps : ∀ i, ps i ∈ circle):
  ∃ (i j : fin 8), i ≠ j ∧ dist (ps i) (ps j) < 1 :=
sorry

end eight_points_in_circle_distance_lt_one_l135_135206


namespace q_0_plus_q_5_l135_135216

-- Define the properties of the polynomial q(x)
variable (q : ℝ → ℝ)
variable (monic_q : ∀ x, ∃ a b c d e f, a = 1 ∧ q x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f)
variable (deg_q : ∀ x, degree q = 5)
variable (q_1 : q 1 = 26)
variable (q_2 : q 2 = 52)
variable (q_3 : q 3 = 78)

-- State the theorem to find q(0) + q(5)
theorem q_0_plus_q_5 : q 0 + q 5 = 58 :=
sorry

end q_0_plus_q_5_l135_135216


namespace sanoop_initial_tshirts_l135_135249

theorem sanoop_initial_tshirts (n : ℕ) (T : ℕ) 
(avg_initial : T = n * 526) 
(avg_remaining : T - 673 = (n - 1) * 505) 
(avg_returned : 673 = 673) : 
n = 8 := 
by 
  sorry

end sanoop_initial_tshirts_l135_135249


namespace ordered_triples_count_l135_135534

open Real

theorem ordered_triples_count :
  ∃ (S : Finset (ℝ × ℝ × ℝ)),
    (∀ (a b c : ℝ), (a, b, c) ∈ S ↔ (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a + b ∧ ca = b)) ∧
    S.card = 2 := 
sorry

end ordered_triples_count_l135_135534


namespace sum_of_smallest_solutions_l135_135092

def greatest_integer_function (x : ℝ) : ℤ := floor x

def equation (x : ℝ) (n : ℤ) : Prop := x - (n : ℝ) = 2 / (n : ℝ)^2

theorem sum_of_smallest_solutions : 
  let x1 := 2.5
  let x2 := 3 + (2/9)
  let x3 := 4 + (1/8)
  x1 + x2 + x3 = 9.847 :=
by
  -- Proof omitted
  sorry

end sum_of_smallest_solutions_l135_135092


namespace exists_same_color_rectangle_l135_135748

open Finset

-- Define the grid size
def gridSize : ℕ := 12

-- Define the type of colors
inductive Color
| red
| white
| blue

-- Define a point in the grid
structure Point :=
(x : ℕ)
(y : ℕ)
(hx : x ≥ 1 ∧ x ≤ gridSize)
(hy : y ≥ 1 ∧ y ≤ gridSize)

-- Assume a coloring function
def color (p : Point) : Color := sorry

-- The theorem statement
theorem exists_same_color_rectangle :
  ∃ (p1 p2 p3 p4 : Point),
    p1.x = p2.x ∧ p3.x = p4.x ∧
    p1.y = p3.y ∧ p2.y = p4.y ∧
    color p1 = color p2 ∧
    color p1 = color p3 ∧
    color p1 = color p4 :=
sorry

end exists_same_color_rectangle_l135_135748


namespace specialFourDigitNumbersCount_l135_135069

noncomputable def numberOfSpecialFourDigitNumbers : ℕ :=
  let isSpecial (N : ℕ) : Prop :=
    let digits := [N / 1000, (N / 100) % 10, (N / 10) % 10, N % 10]
    (digits.map (fun a => a * a)).sum = 2 * digits.sum
  in (List.range' 1000 9000).filter isSpecial).length

theorem specialFourDigitNumbersCount :
  numberOfSpecialFourDigitNumbers = 12 :=
  sorry

end specialFourDigitNumbersCount_l135_135069


namespace sum_of_sequence_l135_135568

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0     := 0 -- Consider a_0 as 0 to make n positive and adjust indexing accordingly
| 1     := 1
| (n+2) := 2 * a (n + 1) - n + 2

-- Define the sum of the first n terms of the sequence {a_n}
def S (n : ℕ) : ℕ :=
  ∑ i in Finset.range n, a (i + 1)

theorem sum_of_sequence (n : ℕ) : S n = 2^n - 1 + (n * (n - 1)) / 2 :=
by
  sorry

end sum_of_sequence_l135_135568


namespace opera_house_earnings_l135_135731

-- Definitions corresponding to the conditions
def num_rows : Nat := 150
def seats_per_row : Nat := 10
def ticket_cost : Nat := 10
def pct_not_taken : Nat := 20

-- Calculations based on conditions
def total_seats := num_rows * seats_per_row
def seats_not_taken := total_seats * pct_not_taken / 100
def seats_taken := total_seats - seats_not_taken
def earnings := seats_taken * ticket_cost

-- The theorem to prove
theorem opera_house_earnings : earnings = 12000 := sorry

end opera_house_earnings_l135_135731


namespace incorrect_propositions_l135_135498

variables {m n : Line} {α β γ : Plane}

-- Conditions
axiom non_coincident_lines : m ≠ n
axiom non_coincident_planes : α ≠ β ∧ β ≠ γ ∧ γ ≠ α

-- Propositions
def prop1 (h1 : m ⟂ α) (h2 : m ⟂ β) : α ∥ β := sorry  -- Proposition (1): correct (proven by theorem of parallel planes)
def prop2 (h1 : α ⟂ γ) (h2 : β ⟂ γ) : ¬(α ∥ β) := sorry  -- Proposition (2): incorrect
def prop3 (h1 : m ⊆ α) (h2 : n ⊆ β) : ¬(α ∥ β) := sorry  -- Proposition (3): incorrect
def prop4 (h1 : ¬(m ∥ β)) (h2 : ¬(β ∥ γ)) : ¬(m ∥ γ) ∨ m ⊆ γ := sorry  -- Proposition (4): incorrect

theorem incorrect_propositions :
  ¬(prop2 α β γ) ∧ ¬(prop3 m n α β) ∧ ¬(prop4 m β γ) :=
by
  sorry

end incorrect_propositions_l135_135498


namespace sum_of_rational_roots_l135_135777

-- Define the polynomial
def h (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

-- State the theorem
theorem sum_of_rational_roots : 
  (∃ x1 x2 x3 : ℝ, h(x1) = 0 ∧ h(x2) = 0 ∧ h(x3) = 0 ∧ (x1 + x2 + x3 = 6)) :=
sorry

end sum_of_rational_roots_l135_135777


namespace problem_1_problem_2_l135_135517

variables {Plane Line : Type}
variables (α β : Plane) (m : Line)

-- Definitions for conditions
def parallel (x y : Plane) : Prop := sorry
def perpendicular (x y : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Conditions
axiom cond_i    : ∀ {m α : Line}, parallel m α
axiom cond_ii   : ∀ {m α : Line}, perpendicular m α
axiom cond_iii  : ∀ {m : Line} {α : Plane}, line_in_plane m α
axiom cond_iv   : ∀ {α β : Plane}, perpendicular α β
axiom cond_v    : ∀ {α β : Plane}, parallel α β

-- Proof problems
theorem problem_1 : (cond_v α β) → (cond_iii m α) → (parallel m β) :=
by sorry

theorem problem_2 : (cond_v α β) → (cond_ii m α) → (perpendicular m β) :=
by sorry

end problem_1_problem_2_l135_135517


namespace sin_transformation_l135_135789

theorem sin_transformation (α : ℝ) (h : Real.sin (3 * Real.pi / 2 + α) = 3 / 5) :
  Real.sin (Real.pi / 2 + 2 * α) = -7 / 25 :=
by
  sorry

end sin_transformation_l135_135789


namespace part1_proof_part2_proof_l135_135945

-- Definitions for triangle sides and angles
variables {A B C a b c : ℝ}

-- Condition 1
def condition1 : Prop := sin C * sin (A - B) = sin B * sin (C - A)

-- Condition 2
def condition2 : Prop := A = 2 * B

-- Proof Problem 1
theorem part1_proof : condition1 → condition2 → C = 5 / 8 * π :=
by sorry

-- Proof Problem 2
theorem part2_proof : condition1 → condition2 → 2 * a^2 = b^2 + c^2 :=
by sorry

end part1_proof_part2_proof_l135_135945


namespace polygon_intersections_l135_135243

theorem polygon_intersections (n4 n5 n7 n9 : ℕ) (h_n4 : n4 = 4) (h_n5 : n5 = 5) (h_n7 : n7 = 7) (h_n9 : n9 = 9) 
    (h_inscribed :
      ∀ (a b : ℕ), a ∈ {n4, n5, n7, n9} → b ∈ {n4, n5, n7, n9} → 
      (∀ (i j : ℕ), ¬ (i = j ∧ i ∈ finset.range a ∧ j ∈ finset.range b)) ∧
      (∀ (x y z : ℕ), ¬(x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x ∈ finset.range a ∧ y ∈ finset.range b ∧ z ∈ finset.range (a + b))) ) :
    ∃ n : ℕ, n = 58 :=
by
  use 58
  sorry

end polygon_intersections_l135_135243


namespace geo_seq_sum_S4_l135_135121

noncomputable def geom_seq_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geo_seq_sum_S4 {a : ℝ} {q : ℝ} (h1 : a * q^2 - a = 15) (h2 : a * q - a = 5) :
  geom_seq_sum a q 4 = 75 :=
by
  sorry

end geo_seq_sum_S4_l135_135121


namespace no_term_of_form_3_pow_alpha_5_pow_beta_l135_135147
open BigOperators

def v : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := 8 * v (n + 1) - v n

theorem no_term_of_form_3_pow_alpha_5_pow_beta (n : ℕ) (α β : ℕ) (hα : 0 < α) (hβ : 0 < β) :
  v n ≠ 3 ^ α * 5 ^ β :=
sorry

end no_term_of_form_3_pow_alpha_5_pow_beta_l135_135147


namespace find_alpha_l135_135826

noncomputable def curve_eq (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

noncomputable def line_eq_passes_point (alpha t : ℝ) : ℝ × ℝ :=
  (t * Real.cos alpha, 1 + t * Real.sin alpha)

theorem find_alpha (alpha : ℝ) (x y : ℝ → ℝ) (PA PB: ℝ → ℝ → ℝ) : 
  (curve_eq x y) ∧
  (∀ t, line_eq_passes_point alpha t = (x t, y t)) ∧ 
  (∀ A B, PA x y = sqrt 5 ∧ PB x y = sqrt 5) → 
  (alpha = π / 3 ∨ alpha = 2 * π / 3) :=
sorry

end find_alpha_l135_135826


namespace counting_multiples_l135_135535

theorem counting_multiples (h8 : floor (200 / 8) = 25)
                           (h11 : floor (200 / 11) = 18)
                           (h88 : floor (200 / 88) = 2) : 
                           Nat :=
  (have h_not8_not11: 25 - 2 = 23, by sorry)
  (have h_not11_not8: 18 - 2 = 16, by sorry)
  show Nat, from 23 + 16

example: counting_multiples 25 18 2 = 39 :=
by sorry

end counting_multiples_l135_135535


namespace true_propositions_l135_135143

variable (x y : ℝ)

def p : Prop := x > y → -x < -y
def q : Prop := x < y → x^2 < y^2
def option1 : Prop := p ∧ q
def option2 : Prop := p ∨ q
def option3 : Prop := p ∧ ¬q
def option4 : Prop := ¬p ∨ q

theorem true_propositions : (option2 x y) ∧ (option3 x y) :=
by 
  sorry

end true_propositions_l135_135143


namespace largest_power_of_five_dividing_Q_l135_135214

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def num_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625 + n / 3125 -- Enough terms for n ≤ 100

theorem largest_power_of_five_dividing_Q :
  let Q := (factorial 100) / (2^50 * factorial 50) in
  ∃ m, 5^m ∣ Q ∧ ∀ k, 5^k ∣ Q → k ≤ 12 :=
by
  let Q := (factorial 100) / (2^50 * factorial 50)
  have h1 : num_factors_of_five 100 = 24 := by
    dsimp [num_factors_of_five, factorial]
    calc 100 / 5 + 100 / 25 + 100 / 125 + 100 / 625 + 100 / 3125 = 24 : by norm_num
  have h2 : num_factors_of_five 50 = 12 := by
    dsimp [num_factors_of_five, factorial]
    calc 50 / 5 + 50 / 25 + 50 / 125 + 50 / 625 + 50 / 3125 = 12 : by norm_num
  use 12
  split
  · -- Proof that 5^12 divides Q
    sorry -- Detailed mathematical proof goes here
  · -- Proof that if 5^k divides Q, then k ≤ 12
    sorry -- Detailed mathematical proof goes here

end largest_power_of_five_dividing_Q_l135_135214


namespace circumscribed_and_inscribed_circle_ratio_compute_m_plus_n_l135_135111

theorem circumscribed_and_inscribed_circle_ratio (s : ℝ) (h₀ : s > 0) :
  let area_circumscribed := π * s^2
  let area_inscribed := π * (s * (√3 / 2))^2
  let ratio := area_circumscribed / area_inscribed
  ratio = (4 / 3) :=
by
  sorry

/-- Given the ratio of the areas is 4/3, calculate m + n where the ratio m/n is in simplest form -/
theorem compute_m_plus_n (m n : ℕ) (h₀ : Nat.gcd m n = 1) (h₁ : (4:ℚ) / 3 = m / n) : 
  m + n = 7 :=
by
  sorry

end circumscribed_and_inscribed_circle_ratio_compute_m_plus_n_l135_135111


namespace pentagonal_pyramid_base_area_l135_135649

theorem pentagonal_pyramid_base_area (total_surface_area lateral_surface_area base_area : ℝ) 
  (h₁: total_surface_area = 30) 
  (h₂: lateral_surface_area = 25) 
  (h₃: base_area = total_surface_area - lateral_surface_area) : 
  base_area = 5 := 
by 
  rw [h₃, h₁, h₂]
  norm_num
  sorry

end pentagonal_pyramid_base_area_l135_135649


namespace inequality_relation_l135_135129

noncomputable def f : ℝ → ℝ := sorry

def a : ℝ := Real.log (1 / Real.pi)
def b : ℝ := (Real.log Real.pi)^2
def c : ℝ := Real.log (Real.sqrt Real.pi)

-- Defining even function and decreasing function property
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) := ∀ x y ∈ I, x < y → f x > f y

axiom even_function : is_even f
axiom decreasing_function : is_decreasing_on f {x : ℝ | 0 < x}

theorem inequality_relation : f c > f a ∧ f a > f b := sorry

end inequality_relation_l135_135129


namespace floor_log_sum_l135_135783

noncomputable def floor : ℝ → ℤ := λ x, ⌊x⌋

theorem floor_log_sum : (floor (Real.log 1 / Real.log 3) + floor (Real.log 2 / Real.log 3) + 
                         floor (Real.log 3 / Real.log 3) + floor (Real.log 4 / Real.log 3) + 
                         floor (Real.log 5 / Real.log 3) + floor (Real.log 6 / Real.log 3) + 
                         floor (Real.log 7 / Real.log 3) + floor (Real.log 8 / Real.log 3) + 
                         floor (Real.log 9 / Real.log 3) + floor (Real.log 10 / Real.log 3) + 
                         floor (Real.log 11 / Real.log 3)) = 12 := 
sorry

end floor_log_sum_l135_135783


namespace four_digit_palindrome_sum_of_digits_l135_135047

def is_digit (n : ℕ) : Prop := n < 10

def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), is_digit a ∧ is_digit b ∧ a ≠ 0 ∧
  n = 1000 * a + 100 * b + 10 * b + a

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem four_digit_palindrome_sum_of_digits :
  sum_of_digits
    (∑ n in (Finset.filter is_palindrome (Finset.range 10000)), n) = 18 := sorry

end four_digit_palindrome_sum_of_digits_l135_135047


namespace reading_time_difference_l135_135374

theorem reading_time_difference (xanthia_speed molly_speed book_length : ℕ)
  (hx : xanthia_speed = 120) (hm : molly_speed = 60) (hb : book_length = 300) :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 150 :=
by
  -- We acknowledge the proof here would use the given values
  sorry

end reading_time_difference_l135_135374


namespace probability_triangle_side_decagon_l135_135786

theorem probability_triangle_side_decagon (decagon_vertices : Finset ℕ) (h : decagon_vertices.card = 10) :
  ∃ p : ℚ, p = 7 / 12 ∧
    (∃ triangles : Finset (Finset ℕ), triangles.card = Finset.choose decagon_vertices.card 3 ∧
      (∃ favorable_triangles : Finset (Finset ℕ), 
        favorable_triangles.card = 70 ∧ 
        favorable_triangles ⊆ triangles)) :=
begin
  sorry
end

end probability_triangle_side_decagon_l135_135786


namespace no_real_solutions_l135_135536

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (x - 3 * x + 8)^2 + 4 = -2 * |x| :=
by
  sorry

end no_real_solutions_l135_135536


namespace solve_for_y_l135_135617

theorem solve_for_y (y : ℝ) : 
  (16 ^ (2 * y - 4) = (1 / 4) ^ (5 - y)) → y = 1 :=
by
  sorry

end solve_for_y_l135_135617


namespace problem_statement_l135_135806

variable {a : ℕ → ℤ}
variable {n : ℕ}

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 2 * n - 1

def geometric_property (a : ℕ → ℤ) : Prop :=
  (a 1 + a 2) * (a 1 + a 2) = 2 * a 1 * (a 1 + a 4)

def b (a n : ℕ → ℤ) :=
  a n + 2 ^ (n - 1)

def sum_first_n (f : ℕ → ℤ ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, f i

theorem problem_statement (a b : ℕ → ℤ) (n : ℕ)
  (H1 : arithmetic_sequence a)
  (H2 : b n = a n + 2 ^ (n - 1)) :
  geometric_property a ∧ sum_first_n (b a) n = n^2 + 2^n - 1 := by
  sorry

end problem_statement_l135_135806


namespace solution_l135_135068

noncomputable def problem : Prop :=
  cos (45 * real.pi / 180) * cos (15 * real.pi / 180) + sin (45 * real.pi / 180) * sin (15 * real.pi / 180) = sqrt 3 / 2

theorem solution : problem := by
  sorry

end solution_l135_135068


namespace number_of_possible_ticket_values_l135_135424

noncomputable def possible_ticket_values (x : ℕ) : Prop :=
  x > 0 ∧ (60 % x = 0) ∧ (84 % x = 0) ∧ (126 % x = 0)

theorem number_of_possible_ticket_values : 
  (finset.card (finset.filter possible_ticket_values (finset.range 127))) = 4 :=
begin
  sorry
end

end number_of_possible_ticket_values_l135_135424


namespace sequence_general_term_R_n_less_T_n_l135_135803

-- Define the sequence and sum of the first n terms
def a_n (n : ℕ) := 2 * n - 1
def S_n (n : ℕ) := n^2
def T_n (n : ℕ) := (∑ k in finset.range n, 2 / (sqrt (a_n k) + sqrt (a_n (k + 1))))
def R_n (n : ℕ) := ∏ k in finset.range n, a_n k / (a_n k + 1)

-- Prove the general term formula for the sequence
theorem sequence_general_term (n : ℕ) : a_n n = 2 * n - 1 :=
sorry

-- Prove that the sum inequalities hold
theorem R_n_less_T_n (n : ℕ) : R_n n < T_n n :=
sorry

end sequence_general_term_R_n_less_T_n_l135_135803


namespace sum_of_cubes_condition_l135_135221

theorem sum_of_cubes_condition (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_condition : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) :
  a^3 + b^3 + c^3 = -36 := 
by
  sorry

end sum_of_cubes_condition_l135_135221


namespace area_of_triangle_APD_is_correct_l135_135241

noncomputable def area_triangle_APD 
  (ABCD : ℝ)
  (area_ABCD : ℝ)
  (P Q : ℝ)
  (h1 : ABCD = 50)
  (h2 : P = 1 / 3 * ABCD)
  (h3 : Q = 2 / 3 * ABCD)
  : Prop :=
  (area_triangle_APD = 1 / 3 * area_ABCD)

theorem area_of_triangle_APD_is_correct 
  (ABCD : ℝ) 
  (P Q : ℝ)
  (h1 : area_ABCD = 50)
  (h2 : P = 1 / 3 * ABCD)
  (h3 : Q = 2 / 3 * ABCD) 
  : area_triangle_APD ABCD area_ABCD P Q h1 h2 h3 :=
sorry

end area_of_triangle_APD_is_correct_l135_135241


namespace find_BC_l135_135891

variable (A B C D M N : Point)
variable (BC : ℝ)

-- Defining variables and their properties
def isParallelogram (A B C D : Point) : Prop := 
  -- definition of a parallelogram

def bisectAngleAt (P Q R S T : Point) : Prop :=
  -- definition of angle bisector property

-- Conditions
axiom h1 : isParallelogram A B C D
axiom h2 : AB = 3
axiom h3 : bisectAngleAt A B M D N
axiom h4 : BM / MN = 1 / 5

-- Statement to be proved
theorem find_BC : BC = 21 := 
by { sorry }

end find_BC_l135_135891


namespace find_m_l135_135505

variables (x m : ℝ)

def equation (x m : ℝ) : Prop := 3 * x - 2 * m = 4

theorem find_m (h1 : equation 6 m) : m = 7 :=
by
  sorry

end find_m_l135_135505


namespace line_properties_l135_135836

theorem line_properties :
  let line_eq := ∀ x y : ℝ, 2 * x + y + 3 = 0 in
  (∀ x y, (x = -1) → (y = 1) → ¬line_eq x y) ∧
  (∀ m b, line_eq = (λ x y, y = m * x + b) → m = -2 ∧ b = -3) :=
by
  let line_eq := ∀ x y : ℝ, 2 * x + y + 3 = 0
  sorry

end line_properties_l135_135836


namespace probability_distance_less_than_7500_l135_135134

theorem probability_distance_less_than_7500 : 
  let distances := [
    ("Bangkok", "Cape Town", 6500),
    ("Bangkok", "Honolulu", 6800),
    ("Bangkok", "London", 6100),
    ("Cape Town", "Honolulu", 11800),
    ("Cape Town", "London", 6200),
    ("Honolulu", "London", 7400)
  ]
  in
  let threshold := 7500 in
  let count := distances.filter (λ d, (d.2 < threshold)).length 
  in
  count.to_rat / distances.length.to_rat = 5/6 := 
by
  intros
  sorry

end probability_distance_less_than_7500_l135_135134


namespace number_of_men_at_tables_l135_135726

theorem number_of_men_at_tables
  (num_tables : ℝ)
  (num_women : ℝ)
  (avg_customers_per_table : ℝ)
  (total_customers : ℝ) 
  (h1 : num_tables = 9.0)
  (h2 : num_women = 7.0)
  (h3 : avg_customers_per_table = 1.111111111)
  (h4 : total_customers = num_tables * avg_customers_per_table) :
    total_customers - num_women = 3.0 :=
by
  -- We assume the truth of these conditions
  have h5 : 9.0 * 1.111111111 = 10.0 := sorry
  -- Therefore, we can derive the total number of customers
  have h6 : total_customers = 10.0 := sorry
  -- Finally, we conclude the number of men is 3.0
  show total_customers - num_women = 3.0, from sorry

end number_of_men_at_tables_l135_135726


namespace product_neg_six_l135_135640

theorem product_neg_six (m b : ℝ)
  (h1 : m = 2)
  (h2 : b = -3) : m * b < -3 := by
-- Proof skipped
sorry

end product_neg_six_l135_135640


namespace general_term_formula_find_T_n_l135_135489

-- Problem 1: General term formula of arithmetic sequence
theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 = 2) 
  (h2 : ∀ n, a n > 0) 
  (h3 : 2 * (S 6 + a 6) = S 4 + a 4 + S 5 + a 5) : 
  ∀ n, a n = (1/2)^(n-2) :=
by
  sorry

-- Problem 2: Find \(T_n\) with given \(b_n\) and sequence \(a_n\)
theorem find_T_n (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (h1 : ∀ n, a n = (1/2)^(n-2)) 
  (h2 : ∀ n, b n = Real.log (a (2*n - 1)) / Real.log (1/2)) 
  (h3 : ∀ n, T n = ∑ i in Finset.range n, 2 / (b i * b (i+1))) : 
  ∀ n, T n = -2 * n / (2 * n - 1) :=
by
  sorry

end general_term_formula_find_T_n_l135_135489


namespace minimum_groups_l135_135095

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def fraction (a b : ℕ) : Prop :=
  a < b ∧ is_odd a ∧ is_odd b

def fractions (xs : List (ℕ × ℕ)) : Prop :=
  xs.length = 12 ∧ (∀ (a b : ℕ), (a, b) ∈ xs → fraction a b)

def unique_fractions (xs : List (ℕ × ℕ)) : Prop :=
  ∀ (p : ℕ × ℕ) (q : ℕ × ℕ), p.1 * q.2 ≠ p.2 * q.1

theorem minimum_groups (xs : List (ℕ × ℕ) ) (Hxs: fractions xs) (Hunique: unique_fractions xs) :
  ∃ (n : ℕ), n = 7 ∧ ∀ (g : List (List (ℕ × ℕ))), (∀ (l : List (ℕ × ℕ)), l ∈ g → ∃ (y : (ℕ × ℕ)), ∀ (x ∈ l), x = y) → g.length = n :=
sorry

end minimum_groups_l135_135095


namespace parabolic_arch_height_l135_135284

noncomputable def arch_height (a : ℝ) : ℝ :=
  a * (0 : ℝ)^2

theorem parabolic_arch_height :
  ∃ (a : ℝ), (∫ x in (-4 : ℝ)..4, a * x^2) = (160 : ℝ) ∧ arch_height a = 30 :=
by
  sorry

end parabolic_arch_height_l135_135284


namespace coeff_x2_in_product_l135_135671

noncomputable def P (x : ℝ) := x^3 - 4 * x^2 + 6 * x - 2
noncomputable def Q (x : ℝ) := 3 * x^2 - 2 * x + 5

theorem coeff_x2_in_product :
  (∃ (p q : ℝ → ℝ), p x = x^3 - 4 * x^2 + 6 * x - 2 ∧ q x = 3 * x^2 - 2 * x + 5
   ∧ (coeff (p * q) 2) = -38) :=
begin
  use [P, Q],
  split,
  { refl },
  split,
  { refl },
  { sorry }
end

end coeff_x2_in_product_l135_135671


namespace journey_speed_l135_135709

theorem journey_speed (v : ℚ) 
  (equal_distance : ∀ {d}, (d = 0.22) → ((0.66 / 3) = d))
  (total_distance : ∀ {d}, (d = 660 / 1000) → (660 / 1000 = 0.66))
  (total_time : ∀ {t} , (t = 11 / 60) → (11 / 60 = t)): 
  (0.22 / 2 + 0.22 / v + 0.22 / 6 = 11 / 60) → v = 1.2 := 
by 
  sorry

end journey_speed_l135_135709


namespace train_speed_in_km_h_l135_135698

-- Definitions from the conditions
def train_length : ℝ := 160
def crossing_time : ℝ := 8

-- Conversion factor
def meters_per_second_to_kilometers_per_hour : ℝ := 3.6

-- Speed calculation
def speed_m_s : ℝ := train_length / crossing_time

-- Conversion to km/h
def speed_km_h : ℝ := speed_m_s * meters_per_second_to_kilometers_per_hour

-- The theorem to prove
theorem train_speed_in_km_h : speed_km_h = 72 := 
by {
  sorry
}

end train_speed_in_km_h_l135_135698


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135348

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) ∧ 
            ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ (m.digits.sum = 27) → m ≤ n :=
begin
  use 999,
  split,
  { -- 999 is a three-digit number 
    norm_num,
  },
  split,
  { -- 999 is less than or equal to 999
    norm_num,
  },
  split,
  { -- 999 is a multiple of 9
    norm_num,
  },
  split,
  { -- The sum of the digits of 999 is 27
    norm_num,
  },
  { -- For any three-digit number m, if it is a multiple of 9 and the sum of its digits is 27, then m ≤ 999
    intros m hm1,
    cases hm1 with hm2 hm3,
    cases hm3 with hm4 hm5,
    exact le_of_lt (by linarith),
    sorry
  },
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135348


namespace symmetric_point_exists_l135_135270

-- Let's define the problem in Lean 4
theorem symmetric_point_exists :
  ∃ (x y : ℝ), 
    let P := (-2 : ℝ, 1 : ℝ) in 
    let line := (x : ℝ, y : ℝ) → x + y - 3 = 0 in
    let mid_point := (x + P.1) / 2 = -3 / 2 ∧ (y + P.2) / 2 = 2 in
    line x y ∧ ((y - P.2) / (x - P.1) = -1) ∧ ((x, y) = (3, 4)) :=
by
  sorry

end symmetric_point_exists_l135_135270


namespace plants_producing_flowers_l135_135529

noncomputable def germinate_percent_daisy : ℝ := 0.60
noncomputable def germinate_percent_sunflower : ℝ := 0.80
noncomputable def produce_flowers_percent : ℝ := 0.80
noncomputable def daisy_seeds_planted : ℕ := 25
noncomputable def sunflower_seeds_planted : ℕ := 25

theorem plants_producing_flowers : 
  let daisy_plants_germinated := germinate_percent_daisy * daisy_seeds_planted,
      sunflower_plants_germinated := germinate_percent_sunflower * sunflower_seeds_planted,
      total_plants_germinated := daisy_plants_germinated + sunflower_plants_germinated,
      plants_that_produce_flowers := produce_flowers_percent * total_plants_germinated
  in plants_that_produce_flowers = 28 :=
by
  sorry

end plants_producing_flowers_l135_135529


namespace range_of_m_l135_135451

def is_double_mean_value_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂, a < x₁ ∧ x₁ < x₂ ∧ x₂ < b ∧
           Deriv.deriv f x₁ = (f b - f a) / (b - a) ∧
           Deriv.deriv f x₂ = (f b - f a) / (b - a)

theorem range_of_m : 
  ∀ (m : ℝ), 
  is_double_mean_value_function (λ x : ℝ, (1/3) * x^3 - (m/2) * x^2) 0 2 →
  (4/3) < m ∧ m < (8/3) := 
begin
  intros,
  sorry
end

end range_of_m_l135_135451


namespace eight_points_in_circle_l135_135203

theorem eight_points_in_circle :
  ∀ (P : Fin 8 → ℝ × ℝ), 
  (∀ i, (P i).1^2 + (P i).2^2 ≤ 1) → 
  ∃ (i j : Fin 8), i ≠ j ∧ ((P i).1 - (P j).1)^2 + ((P i).2 - (P j).2)^2 < 1 :=
by
  sorry

end eight_points_in_circle_l135_135203


namespace tan_double_angle_value_l135_135815

theorem tan_double_angle_value (α : ℝ) (h1 : sin (2 * α) = -sin α) (h2 : α ∈ set.Ioo (π / 2) π) :
  tan (2 * α) = sqrt 3 :=
sorry

end tan_double_angle_value_l135_135815


namespace problem_1_max_value_problem_2_good_sets_count_l135_135549

noncomputable def goodSetMaxValue : ℤ :=
  2012

noncomputable def goodSetCount : ℤ :=
  1006

theorem problem_1_max_value {M : Set ℤ} (hM : ∀ x, x ∈ M ↔ |x| ≤ 2014) :
  ∀ a b c : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (1 / a + 1 / b = 2 / c) →
  (a + c = 2 * b) →
  a ∈ M ∧ b ∈ M ∧ c ∈ M →
  ∃ P : Set ℤ, P = {a, b, c} ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
  goodSetMaxValue = 2012 :=
sorry

theorem problem_2_good_sets_count {M : Set ℤ} (hM : ∀ x, x ∈ M ↔ |x| ≤ 2014) :
  ∀ a b c : ℤ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (1 / a + 1 / b = 2 / c) →
  (a + c = 2 * b) →
  a ∈ M ∧ b ∈ M ∧ c ∈ M →
  ∃ P : Set ℤ, P = {a, b, c} ∧ a ∈ P ∧ b ∈ P ∧ c ∈ P ∧
  goodSetCount = 1006 :=
sorry

end problem_1_max_value_problem_2_good_sets_count_l135_135549


namespace part1_part2_l135_135962

variable {A B C a b c : ℝ}

theorem part1 (h₁ : A = 2 * B) (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : C = 5 / 8 * π :=
  sorry

theorem part2 (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
  sorry

end part1_part2_l135_135962


namespace years_passed_l135_135394

def initial_ages : List ℕ := [19, 34, 37, 42, 48]

def new_ages (x : ℕ) : List ℕ :=
  initial_ages.map (λ age => age + x)

-- Hypothesis: The new ages fit the following stem-and-leaf plot structure
def valid_stem_and_leaf (ages : List ℕ) : Bool :=
  ages = [25, 31, 34, 37, 43, 48]

theorem years_passed : ∃ x : ℕ, valid_stem_and_leaf (new_ages x) := by
  sorry

end years_passed_l135_135394


namespace value_of_f_12_l135_135005

theorem value_of_f_12 (f : ℕ → ℤ) 
  (h1 : f 2 = 5)
  (h2 : f 3 = 7)
  (h3 : ∀ m n : ℕ, 0 < m → 0 < n → f m + f n = f (m * n)) :
  f 12 = 17 :=
by
  sorry

end value_of_f_12_l135_135005


namespace find_a_b_l135_135513

theorem find_a_b
  (f : ℝ → ℝ) (a b : ℝ) (h_a_ne_zero : a ≠ 0) (h_f : ∀ x, f x = x^3 + 3 * x^2 + 1)
  (h_eq : ∀ x, f x - f a = (x - b) * (x - a)^2) :
  a = -2 ∧ b = 1 :=
by
  sorry

end find_a_b_l135_135513


namespace interval_of_monotonic_increase_l135_135138

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 - 2 * x - 3)

theorem interval_of_monotonic_increase :
  (∀ x, -∞ < x ∧ x ≤ -1 ∨ 3 ≤ x ∧ x < +∞ → ∃ y, f y < f x ∧ y < x) ∧
  (∀ x, 3 ≤ x ∧ x < +∞ → ∀ y, y > x → f y > f x) :=
sorry

end interval_of_monotonic_increase_l135_135138


namespace number_of_white_balls_l135_135888

theorem number_of_white_balls (total_balls : ℕ) (yellow_frequency : ℝ) (yellow_balls : ℕ) :
  total_balls = 20 ∧ yellow_frequency = 0.6 ∧ yellow_balls = 12 → (total_balls - yellow_balls = 8) :=
by
  intros h
  obtain ⟨ht, hy, hx⟩ := h
  rw [ht, hx]
  simp
  exact rfl

end number_of_white_balls_l135_135888


namespace accurate_reading_l135_135727

-- Definitions based on the given conditions
def smallest_division := 0.01
def marker_low := 10.41
def marker_high := 10.55
def marking_increment := 0.01
def midpoint (a : ℝ) (b : ℝ) : ℝ := (a + b) / 2

-- Theorem stating the main proof problem
theorem accurate_reading :
  smallest_division = 0.01 →
  marker_low = 10.41 →
  marker_high = 10.55 →
  marking_increment = 0.01 →
  (10.41 < 10.45) →
  (10.45 < 10.55) →
  midpoint 10.41 10.55 ≠ 10.45 →
  (10.45 ∈ set.Icc 10.41 10.55) →
  ∃ x : ℝ, x = 10.45 :=
by
  intros h_sd h_ml h_mh h_mi h_range_low h_range_high h_mid_near h_within_bounds
  have near_midpoint : 10.48 = midpoint 10.41 10.55 :=
    by calc
      (10.41 + 10.55) / 2 = 10.48 : by norm_num
  use 10.45
  sorry

end accurate_reading_l135_135727


namespace sin_neg_600_eq_neg_sqrt3_div_2_l135_135677

def is_periodic (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f (x + period) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the sine function on degrees
def sin_deg (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180)

theorem sin_neg_600_eq_neg_sqrt3_div_2 :
  is_periodic sin_deg 360 ∧
  is_odd sin_deg ∧
  sin_deg 60 = Real.sqrt 3 / 2 →
  sin_deg (-600) = - Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_600_eq_neg_sqrt3_div_2_l135_135677


namespace find_x_l135_135459

theorem find_x (x : ℝ) (h : x + 2.75 + 0.158 = 2.911) : x = 0.003 :=
sorry

end find_x_l135_135459


namespace bacon_calories_l135_135580

-- Define the given conditions
def total_breakfast_calories : ℕ := 1120
def calories_per_pancake : ℕ := 120
def number_of_pancakes : ℕ := 6
def calories_per_bowl_of_cereal : ℕ := 200
def number_of_bacon_strips : ℕ := 2

-- Define the proof statement
theorem bacon_calories :
  let total_pancakes_calories := number_of_pancakes * calories_per_pancake,
      total_cereal_calories := calories_per_bowl_of_cereal,
      calories_from_pancakes_and_cereal := total_pancakes_calories + total_cereal_calories,
      total_bacon_calories := total_breakfast_calories - calories_from_pancakes_and_cereal,
      calories_per_bacon_strip := total_bacon_calories / number_of_bacon_strips
  in calories_per_bacon_strip = 100 :=
by
  -- calculation proof steps would go here
  sorry

end bacon_calories_l135_135580


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135330

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ (100 ≤ n ∧ n < 1000) ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) :=
by {
  sorry
}

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135330


namespace cylinder_not_isosceles_trapezoid_cross_section_l135_135417

-- Definitions of the solids and their properties
def isosceles_trapezoid (shape : Type) : Prop := sorry

def cross_section (solid : Type) : Type := sorry

-- Definitions of the given solids
def Frustum : Type := sorry
def Cylinder : Type := sorry
def Cube : Type := sorry
def TriangularPrism : Type := sorry

-- Property that each solid does or doesn't have a cross-section that is an isosceles trapezoid
axiom frustum_has_isosceles_trapezoid : isosceles_trapezoid (cross_section Frustum)
axiom cylinder_has_rectangle : ∀ (s : Type), s = cross_section Cylinder → ¬ isosceles_trapezoid s
axiom cube_cross_section_variability : ∃ (s : Type), s = cross_section Cube ∧ isosceles_trapezoid s
axiom triangular_prism_has_isosceles_trapezoid : isosceles_trapezoid (cross_section TriangularPrism)

-- The main theorem
theorem cylinder_not_isosceles_trapezoid_cross_section : 
  ∀ (s : Type), s = cross_section Cylinder → ¬ isosceles_trapezoid s :=
begin
  intros s hs,
  exact cylinder_has_rectangle s hs,
end

end cylinder_not_isosceles_trapezoid_cross_section_l135_135417


namespace fraction_adjustment_l135_135545

theorem fraction_adjustment :
  ∀ (a b : ℝ), a / b = 0.75 →
  (a * 1.12) / (b * 0.98) ≈ 0.8571 :=
by
  sorry

end fraction_adjustment_l135_135545


namespace problem_b100_eq_14853_l135_135056

def seq_b : ℕ → ℕ
| 1       := 3
| (n + 1) := seq_b n + 3 * n

theorem problem_b100_eq_14853 : seq_b 100 = 14853 := by
  sorry

end problem_b100_eq_14853_l135_135056


namespace area_of_shaded_region_l135_135058

open Real

noncomputable def line1 (x : ℝ) : ℝ := -3/10 * x + 5
noncomputable def line2 (x : ℝ) : ℝ := -1.5 * x + 9

theorem area_of_shaded_region : 
  ∫ x in (2:ℝ)..6, (line2 x - line1 x) = 8 :=
by
  sorry

end area_of_shaded_region_l135_135058


namespace digit_divisible_by_9_l135_135304

theorem digit_divisible_by_9 (A : ℕ) (h : 72A4 % 9 = 0) : A = 5 := by
  have sum_digits : 7 + 2 + A + 4 = 13 + A := by
    ring
  have div_by_9 : (13 + A) % 9 = 0 := by
    rw [sum_digits, h]
    sorry
  have eq_mod : A % 9 = 5 := by
    sorry
  exact eq_mod

end digit_divisible_by_9_l135_135304


namespace probability_of_sum_3_or_6_l135_135871

noncomputable def probability_event_sums_3_or_6 : ℝ :=
  let balls := {1, 2, 3, 4, 5}
  let total_outcomes := choose 5 2
  let favorable_outcomes := 3  -- (1,2), (1,5), (2,4)
  favorable_outcomes / total_outcomes

theorem probability_of_sum_3_or_6 :
  probability_event_sums_3_or_6 = 3 / 10 :=
by
  sorry

end probability_of_sum_3_or_6_l135_135871


namespace geometric_sequence_third_term_l135_135705

theorem geometric_sequence_third_term (r : ℕ) (a : ℕ) (h1 : a = 6) (h2 : a * r^3 = 384) : a * r^2 = 96 :=
by
  sorry

end geometric_sequence_third_term_l135_135705


namespace part1_part2_l135_135922

theorem part1 (A B : ℝ) (h1 : A = 2 * B) : C = 5 * Real.pi / 8 :=
sorry

theorem part2 (a b c A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) :
   2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l135_135922


namespace apples_in_basket_l135_135854

-- Definitions based on conditions
def total_apples : ℕ := 138
def apples_per_box : ℕ := 18

-- Problem: prove the number of apples in the basket
theorem apples_in_basket : (total_apples % apples_per_box) = 12 :=
by 
  -- Skip the proof part by adding sorry
  sorry

end apples_in_basket_l135_135854


namespace range_of_m_l135_135539

theorem range_of_m (m : ℝ) (h : m = real.sqrt 5 - 1) : 1 < m ∧ m < 2 := 
by
  sorry

end range_of_m_l135_135539


namespace modulus_of_complex_l135_135067

-- Formalization of the complex number modulus problem
theorem modulus_of_complex :
  abs (complex.mk (7/8) 3) = 25 / 8 :=
by
  sorry

end modulus_of_complex_l135_135067


namespace gcd_9011_2147_l135_135673

theorem gcd_9011_2147 : Int.gcd 9011 2147 = 1 := sorry

end gcd_9011_2147_l135_135673


namespace other_solution_l135_135499

theorem other_solution (x : ℚ) (h : 30*x^2 + 13 = 47*x - 2) (hx : x = 3/5) : x = 5/6 ∨ x = 3/5 := by
  sorry

end other_solution_l135_135499


namespace students_left_is_6_l135_135185

-- Start of the year students
def initial_students : ℕ := 11

-- New students arrived during the year
def new_students : ℕ := 42

-- Students at the end of the year
def final_students : ℕ := 47

-- Definition to calculate the number of students who left
def students_left (initial new final : ℕ) : ℕ := (initial + new) - final

-- Statement to prove
theorem students_left_is_6 : students_left initial_students new_students final_students = 6 :=
by
  -- We skip the proof using sorry
  sorry

end students_left_is_6_l135_135185


namespace problem_solution_l135_135693

noncomputable theory

open EuclideanGeometry

-- Definitions of points and circles based on given conditions
variables {w1 w2 : Circle} {X A B C D E F G H : Point}

-- Hypotheses capturing the problem's conditions
def problem_hypotheses : Prop :=
  w1 ≠ w2 ∧ -- different diameters
  tangent w1 w2 X ∧ -- externally tangent at X
  on w1 A ∧ on w1 B ∧ -- A and B on w1
  on w2 C ∧ on w2 D ∧ -- C and D on w2
  common_tangent AC w1 w2 ∧ -- AC common tangent
  common_tangent BD w1 w2 ∧ -- BD common tangent
  inter_line w1 AB CX E ∧ inter_circle w1 CX F (second_time E) ∧ -- CX intersects AB at E and w1 at F second time
  circumscribed_circle E F B G (second_time F) ∧ -- Circle (EFB) intersects AF at G second time
  intersect_lines AX CD H -- AX ∩ CD = H

-- Theorem statement proving points collinearity
theorem problem_solution (h : problem_hypotheses) : collinear E G H :=
by
  sorry -- This part would contain the actual proof

end problem_solution_l135_135693


namespace number_of_possible_n_values_l135_135387

-- Definition of the problem conditions
def valid_sequence (n : ℕ) (a : Fin n → ℤ) :=
  ∀ i : Fin n, a i = abs (a (i + 1) - a (i + 2))

def sum_is_278 (n : ℕ) (a : Fin n → ℤ) :=
  (Finset.univ.sum a) = 278

-- The main theorem statement
theorem number_of_possible_n_values : 
  ∃ (n_1 n_2 : ℕ), n_1 ≠ n_2 ∧ 
  ∀ n : ℕ, (∃ a : Fin n → ℤ, valid_sequence n a ∧ sum_is_278 n a) → n = n_1 ∨ n = n_2 :=
begin
  sorry
end

end number_of_possible_n_values_l135_135387


namespace problem_part1_problem_part2_l135_135939

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l135_135939


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135331

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ (100 ≤ n ∧ n < 1000) ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) :=
by {
  sorry
}

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135331


namespace circumcircle_of_triangle_pab_eqn_l135_135820

theorem circumcircle_of_triangle_pab_eqn :
  ∀ (m : ℝ),
  (∀ P : ℝ × ℝ, P = (3, 4) → P.1 ^ 2 / m - P.2 ^ 2 / 2 = 1) →
  let A := (-sqrt m, 0)
  let B := (sqrt m, 0)
  ∃ center : ℝ × ℝ, center = (0, 3) ∧
  (center.fst - 0) ^ 2 + (center.snd - 3) ^ 2 = 10 :=
by
  intros m hP
  have hA : (-sqrt m, 0) = (-sqrt m, 0) := rfl
  have hB : (sqrt m, 0) = (sqrt m, 0) := rfl
  have hCenter : (0, 3) = (0, 3) := rfl
  exists (0, 3)
  split
  exact rfl
  sorry

end circumcircle_of_triangle_pab_eqn_l135_135820


namespace benny_books_l135_135032

variable (B : ℕ) -- the number of books Benny had initially

theorem benny_books (h : B - 10 + 33 = 47) : B = 24 :=
sorry

end benny_books_l135_135032


namespace range_of_k_l135_135176

theorem range_of_k (k : ℝ) : (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^2 + 2*x1 - k = 0) ∧ (x2^2 + 2*x2 - k = 0)) ↔ k > -1 :=
by
  sorry

end range_of_k_l135_135176


namespace first_year_with_digit_sum_5_after_2021_l135_135552

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

/-- The first year after 2021 in which the sum of the digits is 5 is 2030. -/
theorem first_year_with_digit_sum_5_after_2021 : 
  ∃ n : ℕ, n > 2021 ∧ sum_of_digits n = 5 ∧ ∀ m : ℕ, 2021 < m ∧ m < n → sum_of_digits m ≠ 5 :=
by {
  use 2030,
  sorry -- Proof omitted
}

end first_year_with_digit_sum_5_after_2021_l135_135552


namespace books_bought_l135_135908

noncomputable def totalCost : ℤ :=
  let numFilms := 9
  let costFilm := 5
  let numCDs := 6
  let costCD := 3
  let costBook := 4
  let totalSpent := 79
  totalSpent - (numFilms * costFilm + numCDs * costCD)

theorem books_bought : ∃ B : ℤ, B * 4 = totalCost := by
  sorry

end books_bought_l135_135908


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135324

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((nat.digits 10 n).sum = 27) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ ((nat.digits 10 m).sum = 27) → m ≤ n) :=
begin
  use 999,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intro m,
    intro hm,
    cases hm,
    cases hm_left,
    cases hm_left_left,
    cases hm_left_right,
    cases hm_right,
    sorry
  },
sorry,
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135324


namespace danny_reaches_steve_house_in_31_minutes_l135_135750

theorem danny_reaches_steve_house_in_31_minutes:
  ∃ (t : ℝ), 2 * t - t = 15.5 * 2 ∧ t = 31 := sorry

end danny_reaches_steve_house_in_31_minutes_l135_135750


namespace problem1_problem2_problem3_l135_135582

-- Problem Conditions
def inductive_reasoning (s: Sort _) (g: Sort _) : Prop := 
  ∀ (x: s → g), true 

def probabilistic_conclusion : Prop :=
  ∀ (x : Prop), true

def analogical_reasoning (a: Sort _) : Prop := 
  ∀ (x: a), true 

-- The Statements to be Proved
theorem problem1 : ¬ inductive_reasoning Prop Prop = true := 
sorry

theorem problem2 : probabilistic_conclusion = true :=
sorry 

theorem problem3 : ¬ analogical_reasoning Prop = true :=
sorry 

end problem1_problem2_problem3_l135_135582


namespace cos_fourth_power_sum_l135_135746

open Real

theorem cos_fourth_power_sum : 
  (∑ k in Finset.range 91, (cos (k * π / 180)) ^ 4) = 46 := 
by
  sorry

end cos_fourth_power_sum_l135_135746


namespace apples_purchased_by_danny_l135_135236

theorem apples_purchased_by_danny (pinky_apples : ℕ) (total_apples : ℕ) (danny_apples : ℕ) :
  pinky_apples = 36 → total_apples = 109 → danny_apples = total_apples - pinky_apples → danny_apples = 73 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end apples_purchased_by_danny_l135_135236


namespace ratio_perimeter_l135_135256

-- Define the dimensions of the original rectangle
def orig_length : ℝ := 12
def orig_width : ℝ := 8

-- Define the folding procedure and resulting dimensions
def fold_horizontal (len : ℝ) : ℝ := len / 2
def fold_vertical (len : ℝ) : ℝ := len / 2

-- Define the folded dimensions
def folded_length1 : ℝ := fold_horizontal orig_length
def folded_width1 : ℝ := orig_width

def folded_length2 : ℝ := fold_horizontal orig_width
def folded_width2 : ℝ := fold_vertical orig_length

-- Define the perimeter calculation
def perimeter (length : ℝ) (width : ℝ) : ℝ := 2 * (length + width)

-- Perimeters of the resulting rectangles
def smaller_perimeter : ℝ := perimeter folded_length2 folded_width2
def larger_perimeter : ℝ := perimeter folded_length2 folded_width2

-- Lean statement to prove that the ratio is 1
theorem ratio_perimeter (h1 : folded_length2 = 4) (h2 : folded_width2 = 6) :
  smaller_perimeter / larger_perimeter = 1 :=
by
  simp [smaller_perimeter, larger_perimeter, perimeter, folded_length2, folded_width2, h1, h2]
  sorry

end ratio_perimeter_l135_135256


namespace sqrt3_op_sqrt3_eq_12_l135_135378

noncomputable def custom_op (x y : ℝ) : ℝ :=
  (x + y)^2 - (x - y)^2

theorem sqrt3_op_sqrt3_eq_12 : custom_op (real.sqrt 3) (real.sqrt 3) = 12 :=
by
  sorry

end sqrt3_op_sqrt3_eq_12_l135_135378


namespace combined_work_rate_l135_135377

def work_done_in_one_day (A B : ℕ) (work_to_days : ℕ -> ℕ) : ℚ :=
  (work_to_days A + work_to_days B)

theorem combined_work_rate (A : ℕ) (B : ℕ) (work_to_days : ℕ -> ℕ) :
  work_to_days A = 1/18 ∧ work_to_days B = 1/9 → work_done_in_one_day A B (work_to_days) = 1/6 :=
by
  sorry

end combined_work_rate_l135_135377


namespace part1_C_value_part2_triangle_equality_l135_135977

noncomputable theory

variables (a b c : ℝ) (A B C : ℝ)
variables (h1 : A + B + C = Real.pi) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) (h3 : A = 2 * B)

-- Part 1: Proving that C = 5π/8 given the conditions
theorem part1_C_value :
  C = 5 * Real.pi / 8 :=
begin
  sorry
end

-- Part 2: Proving that 2a^2 = b^2 + c^2 given the conditions
theorem part2_triangle_equality :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
begin
  sorry
end

end part1_C_value_part2_triangle_equality_l135_135977


namespace problem_part1_problem_part2_l135_135940

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l135_135940


namespace rook_reaches_upper_right_in_expected_70_minutes_l135_135048

section RookMoves

noncomputable def E : ℝ := 70

-- Definition of expected number of minutes considering the row and column moves.
-- This is a direct translation from the problem's correct answer.
def rook_expected_minutes_to_upper_right (E_0 E_1 : ℝ) : Prop :=
  E_0 = (70 : ℝ) ∧ E_1 = (70 : ℝ)

theorem rook_reaches_upper_right_in_expected_70_minutes : E = 70 := sorry

end RookMoves

end rook_reaches_upper_right_in_expected_70_minutes_l135_135048


namespace part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135930

-- Definitions for the conditions in the problem
variable {A B C a b c : ℝ}

-- Given conditions and problem setup
axiom triangle_ABC_sides : ∀ {a b c : ℝ}, sides a b c
axiom triangle_ABC_angles : ∀ {A B C : ℝ}, angles A B C
axiom sin_relation : ∀ {A B C : ℝ},
  sin C * sin (A - B) = sin B * sin (C - A)

-- Prove Part (1): If A = 2B, then C = 5π/8
theorem part1_A_eq_2B_implies_C :
  A = 2 * B → C = 5 * π / 8 :=
by
  intro h
  sorry

-- Prove Part (2): 2a² = b² + c²
theorem part2_2a_squared_eq_b_squared_plus_c_squared :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
by
  sorry

end part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135930


namespace digit_making_527B_divisible_by_9_l135_135669

theorem digit_making_527B_divisible_by_9 (B : ℕ) : 14 + B ≡ 0 [MOD 9] → B = 4 :=
by
  intro h
  -- sorry is used in place of the actual proof.
  sorry

end digit_making_527B_divisible_by_9_l135_135669


namespace count_non_congruent_triangles_with_perimeter_10_l135_135157

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def unique_triangles_with_perimeter_10 : finset (ℕ × ℕ × ℕ) :=
  ((finset.range 11).product (finset.range 11)).product (finset.range 11)
  |>.filter (λ t, 
    let (a, b, c) := t.fst.fst, t.fst.snd, t.snd in
      a + b + c = 10 ∧ a ≤ b ∧ b ≤ c ∧ is_triangle a b c)

theorem count_non_congruent_triangles_with_perimeter_10 : 
  unique_triangles_with_perimeter_10.card = 3 := 
sorry

end count_non_congruent_triangles_with_perimeter_10_l135_135157


namespace millie_bracelets_lost_l135_135604

theorem millie_bracelets_lost:
  ∀ (initial_bracelets remaining_bracelets bracelets_lost : ℕ),
  initial_bracelets = 9 →
  remaining_bracelets = 7 →
  bracelets_lost = 2 →
  initial_bracelets - remaining_bracelets = bracelets_lost :=
begin
  intros initial_bracelets remaining_bracelets bracelets_lost h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
end

end millie_bracelets_lost_l135_135604


namespace decompose_2000_prime_factors_number_of_factors_2000_l135_135901

-- Define the necessary conditions and properties
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ n : ℕ, n ∣ p → n = 1 ∨ n = p

def prime_factors (n : ℕ) : list ℕ := sorry -- This function returns the list of prime factors of n

def product_of_list (l : list ℕ) : ℕ := sorry -- This function returns the product of the elements in the list

-- Given: 2000 can be decomposed into its prime factors
theorem decompose_2000_prime_factors :
  ∃ a b : ℕ, a = 4 ∧ b = 3 ∧ product_of_list ([2^a, 5^b]) = 2000 :=
sorry

-- Prove: The number of different positive factors of 2000 is 20
theorem number_of_factors_2000 : ∀ n : ℕ,
  (n = 2000 → (∀ p : ℕ, p ∈ (list.fin_range n) → is_prime p) →
  ∃ k : ℕ, k = 20 ∧ list.length (list.filter (λ x, x ∣ n) (list.fin_range (n+1))) = k) :=
sorry

end decompose_2000_prime_factors_number_of_factors_2000_l135_135901


namespace limit_example_l135_135239
noncomputable theory

open_locale classical

theorem limit_example :
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1 / 3| ∧ |x - 1 / 3| < δ → |(15 * x^2 - 2 * x - 1) / (x - 1 / 3) - 8| < ε :=
begin
  sorry

end limit_example_l135_135239


namespace overall_average_is_63_point_4_l135_135873

theorem overall_average_is_63_point_4 : 
  ∃ (n total_marks : ℕ) (avg_marks : ℚ), 
  n = 50 ∧ 
  (∃ (marks_group1 marks_group2 marks_group3 marks_remaining : ℕ), 
    marks_group1 = 6 * 95 ∧
    marks_group2 = 4 * 0 ∧
    marks_group3 = 10 * 80 ∧
    marks_remaining = (n - 20) * 60 ∧
    total_marks = marks_group1 + marks_group2 + marks_group3 + marks_remaining) ∧ 
  avg_marks = total_marks / n ∧ 
  avg_marks = 63.4 := 
by 
  sorry

end overall_average_is_63_point_4_l135_135873


namespace find_point_on_line_l135_135763

theorem find_point_on_line :
  ∃ x : ℝ, let y := 3 in
  let p1 := (2, -5) in
  let p2 := (6, 7) in
  let p3 := (x, y) in
  let slope := (7 - (-5)) / (6 - 2) in
  let slope_p3_p1 := (y - (-5)) / (x - 2) in
  p3 ∈ set_of (λ p3, slope_p3_p1 = slope ∧ x = 14 / 3) :=
sorry

end find_point_on_line_l135_135763


namespace solution_set_for_f_l135_135856

def f (x : ℝ) : ℝ := 3 - 2 * x

theorem solution_set_for_f (x : ℝ) :
  |f(x+1) + 2| ≤ 3 ↔ 0 ≤ x ∧ x ≤ 3 :=
by
  sorry

end solution_set_for_f_l135_135856


namespace glucose_solution_volume_l135_135401

theorem glucose_solution_volume :
  ∀ (V : ℕ),
    (∀ (gram_sol : ℕ) (cc_sol : ℕ), gram_sol = 10 → cc_sol = 100 → gram_sol / cc_sol = 0.1) ∧
    (4.5 = 10 * V / 100) → V = 45 :=
by
  sorry

end glucose_solution_volume_l135_135401


namespace triangle_sides_angles_l135_135972

theorem triangle_sides_angles (a b c A B C : ℝ) (h1: A = 2 * B) 
  (h2: sin C * sin (A - B) = sin B * sin (C - A)) 
  (h3: A + B + C = π) :
  (C = 5 * π / 8) ∧ (2 * a^2 = b^2 + c^2) :=
by
  -- Proof omitted
  sorry

end triangle_sides_angles_l135_135972


namespace smallest_n_l135_135471

theorem smallest_n (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3)
  (h4 : x * y * z ∣ (x + y + z)^n) : n = 13 :=
sorry

end smallest_n_l135_135471


namespace sequence_8123_appears_l135_135199

theorem sequence_8123_appears :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 5, a n = (a (n-1) + a (n-2) + a (n-3) + a (n-4)) % 10) ∧
  (a 1 = 1 ∧ a 2 = 2 ∧ a 3 = 3 ∧ a 4 = 4) ∧
  (∃ n, a n = 8 ∧ a (n+1) = 1 ∧ a (n+2) = 2 ∧ a (n+3) = 3) :=
sorry

end sequence_8123_appears_l135_135199


namespace non_confusing_five_digit_numbers_correct_l135_135607

def is_palindrome (n : ℕ) : Prop :=
  let str := n.to_string
  str = str.reverse

def num_non_palindromic_five_digit_numbers : ℕ :=
  let total := 90000
  let palindromes := 9 * 10 * 10
  total - palindromes

theorem non_confusing_five_digit_numbers_correct : 
  num_non_palindromic_five_digit_numbers = 89100 :=
by
  sorry

end non_confusing_five_digit_numbers_correct_l135_135607


namespace no_solution_l135_135072

theorem no_solution (x y n : ℕ) (hx : 0 < x) (hy : 0 < y) (hn : 0 < n) : 
  ¬ (x^2 + y^2 + 41 = 2^n) :=
by sorry

end no_solution_l135_135072


namespace minimum_force_to_submerge_l135_135674

-- Definitions from conditions
def volume_cube (V : ℝ) := V = 10 * 10^(-6)
def density_cube (ρ_cube : ℝ) := ρ_cube = 500
def density_water (ρ_water : ℝ) := ρ_water = 1000
def gravity (g : ℝ) := g = 10

-- Theorem to prove the required force
theorem minimum_force_to_submerge (V ρ_cube ρ_water g : ℝ)
  (hV : volume_cube V)
  (hρ_cube : density_cube ρ_cube)
  (hρ_water : density_water ρ_water)
  (hg : gravity g) :
  let mass := ρ_cube * V,
      F_weight := mass * g,
      F_buoyant := ρ_water * V * g,
      F_push := F_buoyant - F_weight
  in F_push = 0.05 := sorry

end minimum_force_to_submerge_l135_135674


namespace fraction_second_year_not_third_year_l135_135882

theorem fraction_second_year_not_third_year (N T S : ℕ) (hN : N = 100) (hT : T = N / 2) (hS : S = N * 3 / 10) :
  (S / (N - T) : ℚ) = 3 / 5 :=
by
  rw [hN, hT, hS]
  norm_num
  sorry

end fraction_second_year_not_third_year_l135_135882


namespace jim_net_paycheck_l135_135581

-- Let’s state the problem conditions:
def biweekly_gross_pay : ℝ := 1120
def retirement_percentage : ℝ := 0.25
def tax_deduction : ℝ := 100

-- Define the amount deduction for the retirement account
def retirement_deduction (gross : ℝ) (percentage : ℝ) : ℝ := gross * percentage

-- Define the remaining paycheck after all deductions
def net_paycheck (gross : ℝ) (retirement : ℝ) (tax : ℝ) : ℝ :=
  gross - retirement - tax

-- The theorem to prove:
theorem jim_net_paycheck :
  net_paycheck biweekly_gross_pay (retirement_deduction biweekly_gross_pay retirement_percentage) tax_deduction = 740 :=
by
  sorry

end jim_net_paycheck_l135_135581


namespace find_a4_l135_135506

noncomputable def S : ℕ → ℤ
| 0 => 0
| 1 => -1
| n+1 => 3 * S n + 2^(n+1) - 3

def a : ℕ → ℤ
| 0 => 0
| 1 => -1
| n+1 => 3 * a n + 2^n

theorem find_a4 (h1 : ∀ n ≥ 2, S n = 3 * S (n - 1) + 2^n - 3) (h2 : a 1 = -1) : a 4 = 11 :=
by
  sorry

end find_a4_l135_135506


namespace kindergarten_solution_l135_135183

def kindergarten_cards (x y z t : ℕ) : Prop :=
  (x + y = 20) ∧ (z + t = 30) ∧ (y + z = 40) → (x + t = 10)

theorem kindergarten_solution : ∃ (x y z t : ℕ), kindergarten_cards x y z t :=
by {
  sorry
}

end kindergarten_solution_l135_135183


namespace find_third_root_of_polynomial_l135_135297

noncomputable def polynomial : Polynomial ℚ := Polynomial.mk
  ![-(6 - 2/14 : ℚ), 2 (1 / 14 + 40 / 21 : ℚ), - (3 * 1 / 14 - 40 / 21 : ℚ), 1 / 14]

theorem find_third_root_of_polynomial :
  (root_of_polynomial polynomial 1) ∧ 
  (root_of_polynomial polynomial -3) ∧ 
  (∀ x, x ≠ 1 ∧ x ≠ -3 → root_of_polynomial polynomial x ↔ x = 322 / 21) :=
by sorry

end find_third_root_of_polynomial_l135_135297


namespace minimum_distance_l135_135518

noncomputable def pointM_polar := (4 * Real.sqrt 2, Real.pi / 4)
def curveC_polar_eq (theta : ℝ) : ℝ := Real.sqrt (12 / (1 + 2 * (Real.sin theta)^2))
def line1_param_eq (t : ℝ) : ℝ × ℝ := (6 + t, t)

def pointM_rect := (4, 4)
def midpoint_P (alpha : ℝ) (M : ℝ × ℝ) (N : ℝ × ℝ) : ℝ × ℝ := 
  ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

def distance_line_1 (P : ℝ × ℝ) : ℝ := 
  Real.abs (P.1 - P.2 - 6) / Real.sqrt 2

theorem minimum_distance (alpha : ℝ) (h_alpha : 0 ≤ alpha ∧ alpha < 2 * Real.pi) :
  ∃ t, distance_line_1 ((Real.sqrt 3 * Real.cos alpha + 2), (Real.sin alpha + 2)) = 2 * Real.sqrt 2 :=
sorry

end minimum_distance_l135_135518


namespace find_a1_l135_135488

theorem find_a1 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_rec : ∀ n ≥ 2, a n + 2 * S n * S (n - 1) = 0)
  (h_S5 : S 5 = 1 / 11) : 
  a 1 = 1 / 3 := 
sorry

end find_a1_l135_135488


namespace sin_alpha_value_l135_135814

open Real


theorem sin_alpha_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : π / 2 < β ∧ β < π)
  (h_sin_alpha_beta : sin (α + β) = 3 / 5) (h_cos_beta : cos β = -5 / 13) :
  sin α = 33 / 65 := 
by
  sorry

end sin_alpha_value_l135_135814


namespace sufficient_but_not_necessary_not_necessary_l135_135191

theorem sufficient_but_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (a * (b + 1) > a^2) :=
sorry

theorem not_necessary (a b : ℝ) : (a * (b + 1) > a^2 → b > a ∧ a > 0) → false :=
sorry

end sufficient_but_not_necessary_not_necessary_l135_135191


namespace square_area_l135_135569

theorem square_area 
  (ABCD : ℝ → ℝ → Prop)
  (E DC : ℝ)
  (midpoint_E : E = DC / 2)
  (intersection_F : ∀ B E AC, ∃ F, line_segment B E intersects diagonal A C)
  (area_AFED : ℝ)
  (h1 : ABCD)
  (h2 : E)
  (h3 : intersection_F)
  (h4 : area_AFED = 45) : 
  ∃ area_ABCD, area_ABCD = 108 :=
begin
  sorry
end

end square_area_l135_135569


namespace part1_proof_part2_proof_l135_135947

-- Definitions for triangle sides and angles
variables {A B C a b c : ℝ}

-- Condition 1
def condition1 : Prop := sin C * sin (A - B) = sin B * sin (C - A)

-- Condition 2
def condition2 : Prop := A = 2 * B

-- Proof Problem 1
theorem part1_proof : condition1 → condition2 → C = 5 / 8 * π :=
by sorry

-- Proof Problem 2
theorem part2_proof : condition1 → condition2 → 2 * a^2 = b^2 + c^2 :=
by sorry

end part1_proof_part2_proof_l135_135947


namespace parabola_focus_value_of_a_l135_135390

theorem parabola_focus_value_of_a :
  (∀ a : ℝ, (∃ y : ℝ, y = a * (0^2) ∧ (0, y) = (0, 3 / 8)) → a = 2 / 3) := by
sorry

end parabola_focus_value_of_a_l135_135390


namespace proof_theorem_l135_135641

noncomputable def proof_problem : Prop :=
  let a := 4 ^ 0.2
  let b := 3 ^ 0.5
  let c := 3 ^ 0.4
  let d := Real.log 0.4 0.5
  4 ^ 0.2 = 2 ^ 0.4 ∧
  1 < 2 ^ 0.4 ∧
  2 ^ 0.4 < 3 ^ 0.4 ∧
  3 ^ 0.5 > 3 ^ 0.4 ∧
  Real.log 0.4 0.5 ∈ Set.Ioo 0 1 ∧
  d < a ∧ a < c ∧ c < b

theorem proof_theorem : proof_problem := by
  sorry

end proof_theorem_l135_135641


namespace part1_proof_part2_proof_l135_135948

-- Definitions for triangle sides and angles
variables {A B C a b c : ℝ}

-- Condition 1
def condition1 : Prop := sin C * sin (A - B) = sin B * sin (C - A)

-- Condition 2
def condition2 : Prop := A = 2 * B

-- Proof Problem 1
theorem part1_proof : condition1 → condition2 → C = 5 / 8 * π :=
by sorry

-- Proof Problem 2
theorem part2_proof : condition1 → condition2 → 2 * a^2 = b^2 + c^2 :=
by sorry

end part1_proof_part2_proof_l135_135948


namespace jiwoo_initial_money_l135_135905

def initial_money (M : ℝ) : Prop :=
  (M / 2 - 2000) / 2 - (M / 4 - 4000) = 0

theorem jiwoo_initial_money : ∃ M: ℝ, initial_money M := 
begin
  use 16000,
  sorry
end

end jiwoo_initial_money_l135_135905


namespace three_scientists_same_topic_l135_135292

theorem three_scientists_same_topic :
  ∃ (A B C : Fin 17), ∃ (t : Fin 3), 
    (∀ (x y : Fin 17), x ≠ y → ∃ (t : Fin 3), (x, y) ∈ t) ∧ 
    (A ≠ B) ∧ (B ≠ C) ∧ (A ≠ C) ∧ 
    ∀ (x y : Fin 17), (x = A ∨ x = B ∨ x = C) ∧ (y = A ∨ y = B ∨ y = C) → 
    (x, y) ∈ t :=
by
  sorry

end three_scientists_same_topic_l135_135292


namespace find_n_l135_135219

variable (P : ℕ → ℝ) (n : ℕ)

def polynomialDegree (P : ℕ → ℝ) (deg : ℕ) : Prop :=
  ∀ k, k > deg → P k = 0

def zeroValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n + 1)).map (λ k => 2 * k) → P i = 0

def twoValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n)).map (λ k => 2 * k + 1) → P i = 2

def specialValue (P : ℕ → ℝ) (n : ℕ) : Prop :=
  P (2 * n + 1) = -30

theorem find_n :
  (∃ n, polynomialDegree P (2 * n) ∧ zeroValues P n ∧ twoValues P n ∧ specialValue P n) →
  n = 2 :=
by
  sorry

end find_n_l135_135219


namespace B_works_in_4_5_days_l135_135700

theorem B_works_in_4_5_days :
  ∀ (daysA worksForA remainingWorkForB daysB : ℝ),
    daysA = 15 →
    worksForA = 5 →
    remainingWorkForB = 3 →
    (1 - worksForA / daysA) = 2 / 3 →
    (remainingWorkForB / (2 / 3)) = 4.5 →
      B_works_in_4_5_days := sorry

end B_works_in_4_5_days_l135_135700


namespace remainder_div_x_plus_1_l135_135101

noncomputable def f (x : ℝ) : ℝ := x^8 + 3

theorem remainder_div_x_plus_1 : 
  (f (-1) = 4) := 
by
  sorry

end remainder_div_x_plus_1_l135_135101


namespace even_function_m_value_l135_135168

def f (x m : ℝ) : ℝ := (x - 2) * (x - m)

theorem even_function_m_value (m : ℝ) :
  (∀ x : ℝ, f x m = f (-x) m) → m = -2 := by
  sorry

end even_function_m_value_l135_135168


namespace statementA_statementC_statementD_l135_135825

variable {a x1 x2 x3 : ℝ}

-- Define the first curve y = x / e^x
def curve1 (x : ℝ) : ℝ := x / Real.exp x

-- Define the second curve y = ln x / x
def curve2 (x : ℝ) : ℝ := Real.log x / x

-- Define the line y = a
def line (a x : ℝ) : ℝ := a

-- Conditions that define the points A, B, and C
def pointA_condition (x1 a : ℝ) : Prop := curve1 x1 = a
def pointB_condition (x2 a : ℝ) : Prop := curve1 x2 = a ∧ curve2 x2 = a
def pointC_condition (x3 a : ℝ) : Prop := curve2 x3 = a

-- Statements that need to be proven
theorem statementA (hB : pointB_condition x2 a) : x2 = a * Real.exp x2 := sorry
theorem statementC (hB : pointB_condition x2 a) (hC : pointC_condition x3 a) : x3 = Real.exp x2 := sorry
theorem statementD (hA : pointA_condition x1 a) (hB : pointB_condition x2 a) (hC : pointC_condition x3 a) : x1 + x3 > 2 * x2 := sorry

end statementA_statementC_statementD_l135_135825


namespace number_of_valid_m_l135_135594

-- Define our conditions as mathematical statements.
def is_4_digit_number (m : ℕ) : Prop := 1000 ≤ m ∧ m ≤ 9999

def quotient_remainder_condition (m a b : ℕ) : Prop :=
  m = 50 * a + b ∧ 0 ≤ b ∧ b < 50

def divisible_by_seven (n : ℕ) : Prop := n % 7 = 0

def count_valid_m : ℕ :=
  (Finset.range 10000).filter (λ m, is_4_digit_number m ∧
                                     ∃ a b, quotient_remainder_condition m a b ∧
                                            divisible_by_seven (a + b)).card

theorem number_of_valid_m : count_valid_m = 3836 :=
  sorry

end number_of_valid_m_l135_135594


namespace exp_calculation_l135_135430

theorem exp_calculation : 0.125^8 * (-8)^7 = -0.125 :=
by
  -- conditions used directly in proof
  have h1 : 0.125 = 1 / 8 := sorry
  have h2 : (-1)^7 = -1 := sorry
  -- the problem statement
  sorry

end exp_calculation_l135_135430


namespace negation_of_all_exp_monotonic_l135_135840

theorem negation_of_all_exp_monotonic :
  ¬ (∀ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f x < f y) → (∃ g : ℝ → ℝ, ∃ x y : ℝ, x < y ∧ g x ≥ g y)) :=
sorry

end negation_of_all_exp_monotonic_l135_135840


namespace minimal_poster_wall_area_l135_135916

theorem minimal_poster_wall_area (m n : ℕ) (h_mn : m ≥ n) (h_m_pos : m > 0) (h_n_pos : n > 0) :
  ∃ A : ℕ, A = m * (n * (n + 1)) / 2 :=
by
  use m * (n * (n + 1)) / 2
  sorry

end minimal_poster_wall_area_l135_135916


namespace veranda_width_l135_135637

def room_length : ℕ := 17
def room_width : ℕ := 12
def veranda_area : ℤ := 132

theorem veranda_width :
  ∃ (w : ℝ), (17 + 2 * w) * (12 + 2 * w) - 17 * 12 = 132 ∧ w = 2 :=
by
  use 2
  sorry

end veranda_width_l135_135637


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135320

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((nat.digits 10 n).sum = 27) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ ((nat.digits 10 m).sum = 27) → m ≤ n) :=
begin
  use 999,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intro m,
    intro hm,
    cases hm,
    cases hm_left,
    cases hm_left_left,
    cases hm_left_right,
    cases hm_right,
    sorry
  },
sorry,
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135320


namespace max_length_vector_bc_cos_beta_value_l135_135150

open Real

noncomputable def vector_a (α : ℝ) : ℝ × ℝ := (cos α, sin α)
noncomputable def vector_b (β : ℝ) : ℝ × ℝ := (cos β, sin β)
def vector_c : ℝ × ℝ := (-1, 0)

noncomputable def vector_length (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)

theorem max_length_vector_bc (β : ℝ) : 0 ≤ β ∧ β ≤ 2 * π → vector_length (⟨cos β - 1, sin β⟩) ≤ 2 :=
by sorry

theorem cos_beta_value (β : ℝ) (α : ℝ) (h : α = π / 3) (h_perp : vector_a α ⬝ ⟨cos β - 1, sin β⟩ = 0) :
  β = 2 * π * int.floor (β / (2 * π)) ∨ β = 2 * π * int.floor (β / (2 * π)) + 2 * π / 3 →
  (cos β = -1 / 2 ∨ cos β = 1) := 
by sorry

end max_length_vector_bc_cos_beta_value_l135_135150


namespace simplify_expression_l135_135615

variable (x : ℝ)
hypothesis h : x ≠ -1

theorem simplify_expression (h : x ≠ -1) : (x + 1)⁻¹ - 2 = - (2 * x + 1) / (x + 1) :=
by
  sorry

end simplify_expression_l135_135615


namespace max_area_of_triangle_l135_135114

theorem max_area_of_triangle (a b c : ℝ) 
  (h1 : ∀ (a b c : ℝ), S = a^2 - (b - c)^2)
  (h2 : b + c = 8) : 
  S ≤ 64 / 17 :=
sorry

end max_area_of_triangle_l135_135114


namespace number_of_solutions_l135_135077

noncomputable def g (n : ℤ) : ℤ :=
  (⌈119 * n / 120⌉ - ⌊120 * n / 121⌋ : ℤ)

theorem number_of_solutions : (finset.range 12120).filter (λ n, g n = 1).card = 12120 :=
sorry

end number_of_solutions_l135_135077


namespace problem_l135_135112

variables (a b : ℝ)
def sequence (n : ℕ) : ℝ := 
  if n = 1 then a else
  if n = 2 then b else
  sequence (n-1) - sequence (n-2)

def S (n : ℕ) := ∑ i in finset.range (n + 1), if i = 0 then 0 else sequence a b i

theorem problem (a b : ℝ) :
  sequence a b 100 = -a ∧ S a b 100 = 2 * b - a :=
sorry

end problem_l135_135112


namespace perpendicular_bisector_eq_distance_equal_solution_l135_135188

-- Define point A
def A : ℝ × ℝ := (-3, -4)
-- Define point B
def B : ℝ × ℝ := (6, 3)
-- Define the line equation with a parameter m
def line (m : ℝ) : ℝ × ℝ → Prop := λ p, p.1 + m * p.2 + 1 = 0

-- Prove the perpendicular bisector equation for points A and B
theorem perpendicular_bisector_eq :
  ∀ x y : ℝ, 9 * x + 7 * y - 10 = 0 ↔
  ∃ (m : ℝ), ∀ (p : ℝ × ℝ), ((line m p) → 
  ∥p.1 + 3∥ + ∥p.2 + 4∥ = ∥p.1 - 6∥ + ∥p.2 - 3∥ ∧ m = -9/7) :=
sorry

-- Prove that the only valid m is 5, given distances from A and B to line are equal
theorem distance_equal_solution :
  ∀ m : ℝ,
  (Real.abs (4 * m + 2)) / (Real.sqrt (1 + m ^ 2)) = 
  (Real.abs (3 * m + 7)) / (Real.sqrt (1 + m ^ 2)) →
  m = 5 :=
sorry

end perpendicular_bisector_eq_distance_equal_solution_l135_135188


namespace exists_constants_pqr_l135_135912

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ := 
  !![2, 3, 0;
     3, 2, 3;
     0, 3, 2]

def I : Matrix (Fin 3) (Fin 3) ℝ := 
  1

def Z : Matrix (Fin 3) (Fin 3) ℝ := 
  0

theorem exists_constants_pqr :
  ∃ (p q r : ℝ), 
  p = 38 / 3 ∧
  q = -50 / 3 ∧
  r = -10 ∧
  B^3 + p • B^2 + q • B + r • I = Z := 
sorry

end exists_constants_pqr_l135_135912


namespace find_mod_z_l135_135217

theorem find_mod_z (z w : ℂ) (h1 : abs (3 * z - w) = 30)
    (h2 : abs (z + 3 * w) = 6)
    (h3 : abs (z + w) = 3) :
    ∃ a : ℝ, |z| = real.sqrt a := sorry

end find_mod_z_l135_135217


namespace days_elapsed_l135_135426

theorem days_elapsed
  (initial_amount : ℕ)
  (daily_spending : ℕ)
  (total_savings : ℕ)
  (doubling_factor : ℕ)
  (additional_amount : ℕ)
  :
  initial_amount = 50 →
  daily_spending = 15 →
  doubling_factor = 2 →
  additional_amount = 10 →
  2 * (initial_amount - daily_spending) * total_savings + additional_amount = 500 →
  total_savings = 7 :=
by
  intros h_initial h_spending h_doubling h_additional h_total
  sorry

end days_elapsed_l135_135426


namespace find_C_prove_relation_l135_135952

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A), and A = 2B,
prove that C = 5/8 * π. -/
theorem find_C
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  C = ⅝ * Real.pi :=
sorry

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A),
prove that 2 * a ^ 2 = b ^ 2 + c ^ 2. -/
theorem prove_relation
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
sorry

end find_C_prove_relation_l135_135952


namespace distance_between_points_A_B_l135_135196

def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem distance_between_points_A_B :
  let A := polar_to_cartesian 2 (Real.pi / 6)
  let B := polar_to_cartesian 2 (-Real.pi / 6)
  distance A B = 2 := by
  sorry

end distance_between_points_A_B_l135_135196


namespace find_quadratic_equation_roots_l135_135300

theorem find_quadratic_equation_roots (x y : ℝ) (h1 : x + y = 10) (h2 : |x - y| = 12) : 
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ z, a * z^2 + b * z + c = 0 ↔ z = x ∨ z = y) ∧ (a = 1 ∧ b = -10 ∧ c = -11) :=
by
  use [1, -10, -11]
  split
  { rw [Ne.def, zero_eq_one], trivial }
  split
  { intro z,
    split
    { intro h,
      sorry -- Proof that if z is a root, then z = x or z = y
    },
    { intro hz,
      sorry -- Proof that if z = x or z = y, then z is a root
    }
  }
  { exact ⟨rfl, rfl, rfl⟩ }

end find_quadratic_equation_roots_l135_135300


namespace hcf_36_84_l135_135684

def highestCommonFactor (a b : ℕ) : ℕ := Nat.gcd a b

theorem hcf_36_84 : highestCommonFactor 36 84 = 12 := by
  sorry

end hcf_36_84_l135_135684


namespace merchant_profit_percentage_is_correct_l135_135008

-- Defining the basic constants.
def cost_price : ℝ := 100
def marked_price : ℝ := cost_price + 0.75 * cost_price
def discount : ℝ := 0.10 * marked_price
def selling_price : ℝ := marked_price - discount
def profit : ℝ := selling_price - cost_price
def profit_percentage : ℝ := (profit / cost_price) * 100

-- Stating the problem: Proving that the profit percentage is 57.5%
theorem merchant_profit_percentage_is_correct : profit_percentage = 57.5 := by
  -- Placeholder for initializing the Lean theorem.
  sorry

end merchant_profit_percentage_is_correct_l135_135008


namespace count_non_congruent_triangles_with_perimeter_10_l135_135155

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def unique_triangles_with_perimeter_10 : finset (ℕ × ℕ × ℕ) :=
  ((finset.range 11).product (finset.range 11)).product (finset.range 11)
  |>.filter (λ t, 
    let (a, b, c) := t.fst.fst, t.fst.snd, t.snd in
      a + b + c = 10 ∧ a ≤ b ∧ b ≤ c ∧ is_triangle a b c)

theorem count_non_congruent_triangles_with_perimeter_10 : 
  unique_triangles_with_perimeter_10.card = 3 := 
sorry

end count_non_congruent_triangles_with_perimeter_10_l135_135155


namespace average_students_per_school_l135_135910

-- Conditions
def schools : ℝ := 25.0
def totalStudents : ℝ := 247.0
def averageStudents : ℝ := totalStudents / schools

-- Statement
theorem average_students_per_school :
  averageStudents = 9.88 :=
by
  sorry

end average_students_per_school_l135_135910


namespace part1_part2_l135_135961

variable {A B C a b c : ℝ}

theorem part1 (h₁ : A = 2 * B) (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : C = 5 / 8 * π :=
  sorry

theorem part2 (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
  sorry

end part1_part2_l135_135961


namespace count_friend_or_enemy_triples_l135_135878

-- Define a group of 30 people
noncomputable def people : Finset (Fin 30) := Finset.univ

-- Define the property of having exactly 6 enemies
def has_six_enemies (person : Fin 30) (enemies : Finset (Fin 30)) : Prop := 
  enemies.card = 6 ∧ ∀ e ∈ enemies, e ≠ person

-- Statement: There are 1990 ways to choose 3 people such that all are friends or all are enemies
theorem count_friend_or_enemy_triples (enemy_sets : (Fin 30) → Finset (Fin 30)) 
  (h : ∀ person, has_six_enemies person (enemy_sets person)) : 
  ∃ count : ℕ, count = 1990 ∧ count = 
    (people.choose 3).card - Finset.card (Finset.filter 
      (λ grp, ∃ a b c, {a, b, c} = grp ∧ 
        ((enemy_sets a).card = (enemy_sets b).card 
        ∧ (enemy_sets b).card = (enemy_sets c).card)) 
      (people.choose 3)) :=
sorry

end count_friend_or_enemy_triples_l135_135878


namespace sum_of_underlined_numbers_non_negative_l135_135697

-- Definitions used in the problem
def is_positive (n : Int) : Prop := n > 0
def underlined (nums : List Int) : List Int := sorry -- Define underlining based on conditions

def sum_of_underlined_numbers (nums : List Int) : Int :=
  (underlined nums).sum

-- The proof problem statement
theorem sum_of_underlined_numbers_non_negative
  (nums : List Int)
  (h_len : nums.length = 100) :
  0 < sum_of_underlined_numbers nums := sorry

end sum_of_underlined_numbers_non_negative_l135_135697


namespace max_cities_l135_135014

def city (X : Type) := X

variable (A B C D E : Prop)

-- Conditions as given in the problem
axiom condition1 : A → B
axiom condition2 : D ∨ E
axiom condition3 : B ↔ ¬C
axiom condition4 : C ↔ D
axiom condition5 : E → (A ∧ D)

-- Proof problem: Given the conditions, prove that the maximum set of cities that can be visited is {C, D}
theorem max_cities (h1 : A → B) (h2 : D ∨ E) (h3 : B ↔ ¬C) (h4 : C ↔ D) (h5 : E → (A ∧ D)) : (C ∧ D) ∧ ¬A ∧ ¬B ∧ ¬E :=
by
  -- The core proof would use the constraints to show C and D, and exclude A, B, E
  sorry

end max_cities_l135_135014


namespace find_number_log_base_2_l135_135651

theorem find_number_log_base_2 (x : ℝ) (h : log x / log 2 = 2) : x = 4 :=
by sorry

end find_number_log_base_2_l135_135651


namespace ellipse_equation_l135_135508

noncomputable def ellipse_focus (a b : ℝ) (ha : a > b) (hb : b > 0) (c : ℝ) (hc : c ^ 2 = a ^ 2 - b ^ 2) :=
  ∃ (F : ℝ × ℝ), F = (3, 0) ∧ ∃ (A B : ℝ × ℝ),
    (1 / a ^ 2 * A.1 ^ 2 + 1 / b ^ 2 * A.2 ^ 2 = 1) ∧
    (1 / a ^ 2 * B.1 ^ 2 + 1 / b ^ 2 * B.2 ^ 2 = 1) ∧
    (A.1 + B.1) / 2 = 1 ∧
    (A.2 + B.2) / 2 = -1

theorem ellipse_equation :
  ∀ (a b : ℝ) (ha : a > b) (hb : b > 0) (c : ℝ) (hc : c ^ 2 = a ^ 2 - b ^ 2),
  (ellipse_focus a b ha hb c hc) → a ^ 2 = 18 ∧ b ^ 2 = 9 :=
begin
  intros,
  sorry
end

end ellipse_equation_l135_135508


namespace odd_and_monotonically_decreasing_l135_135824

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := cos (2 * x + φ)

theorem odd_and_monotonically_decreasing (φ : ℝ) (hφ1 : -π/2 < φ) (hφ2 : φ < π/2) :
  (∃ φ, (13 * π / 6) + φ = k * π ∧ -π/2 < φ ∧ φ < π/2 → f (x + π / 3) φ = -sin (2 * x)) :=
by sorry

end odd_and_monotonically_decreasing_l135_135824


namespace tangent_line_equation_l135_135482

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 5

def point_A : ℝ × ℝ := (1, -2)

theorem tangent_line_equation :
  ∀ x y : ℝ, (y = 4 * x - 6) ↔ (fderiv ℝ f (point_A.1) x = 4) ∧ (y = f (point_A.1) + 4 * (x - point_A.1)) := by
  sorry

end tangent_line_equation_l135_135482


namespace sufficient_but_not_necessary_not_necessary_l135_135192

theorem sufficient_but_not_necessary (a b : ℝ) (h : b > a ∧ a > 0) : (a * (b + 1) > a^2) :=
sorry

theorem not_necessary (a b : ℝ) : (a * (b + 1) > a^2 → b > a ∧ a > 0) → false :=
sorry

end sufficient_but_not_necessary_not_necessary_l135_135192


namespace no_integer_roots_l135_135610

-- Define a predicate for checking if a number is odd
def is_odd (a : ℤ) : Prop := a % 2 = 1

-- Define the polynomial with integer coefficients
def P (a : list ℤ) (x : ℤ) : ℤ := 
  (a.zipWithIndex.map (λ (ai, i), ai * x ^ i)).sum

-- The main theorem stating the polynomial does not have integer roots
theorem no_integer_roots (a : list ℤ) (h0 : is_odd (P a 0)) (h1 : is_odd (P a 1)) :
  ∀ r : ℤ, P a r ≠ 0 := 
sorry

end no_integer_roots_l135_135610


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135312

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 + d2 + d3

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, is_three_digit n ∧ is_multiple_of_9 n ∧ digit_sum n = 27 ∧
  ∀ m : ℕ, is_three_digit m ∧ is_multiple_of_9 m ∧ digit_sum m = 27 → m ≤ n := 
by 
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135312


namespace power_logarithm_simplification_l135_135676

theorem power_logarithm_simplification : (729 ^ (Real.log 3250 / Real.log 3)) ^ (1 / 6) = 3250 := by
  have h1 : 729 = 3^6 := by sorry
  have h2 : (3^6 : ℝ) = Real.exp (6 * Real.log 3) := by sorry
  have h3 : (729 : ℝ) = Real.exp (6 * Real.log 3) := by sorry
  have h4 : (Real.log 3250 / Real.log 3) = (Real.log (3250^1)) := by sorry
  sorry

end power_logarithm_simplification_l135_135676


namespace pirate_coins_l135_135397

-- Define the conditions
def coins (y : ℕ) (j : ℕ) : ℕ :=
  (14! * y) / (15^14)

-- Define the statement we want to prove
theorem pirate_coins :
  ∃ y : ℕ, y = 512512 ∧ ∀ j : ℕ, j ∈ Finset.range 15 → 
  let y_j := (15 - j) in y_j * y / 15^(j + 1) ∈ ℕ :=
sorry

end pirate_coins_l135_135397


namespace num_perfect_square_21n_le_500_l135_135082

theorem num_perfect_square_21n_le_500 : 
  {n : ℕ | n ≤ 500 ∧ ∃ k : ℕ, 21 * n = k ^ 2}.to_finset.card = 4 := 
by sorry

end num_perfect_square_21n_le_500_l135_135082


namespace adam_books_l135_135415

theorem adam_books (before_books total_shelves books_per_shelf after_books leftover_books bought_books : ℕ)
  (h_before: before_books = 56)
  (h_shelves: total_shelves = 4)
  (h_books_per_shelf: books_per_shelf = 20)
  (h_leftover: leftover_books = 2)
  (h_after: after_books = (total_shelves * books_per_shelf) + leftover_books)
  (h_difference: bought_books = after_books - before_books) :
  bought_books = 26 :=
by
  sorry

end adam_books_l135_135415


namespace ratio_B_to_A_l135_135744

-- Definitions for conditions
def w_B : ℕ := 275 -- weight of element B in grams
def w_X : ℕ := 330 -- total weight of compound X in grams

-- Statement to prove
theorem ratio_B_to_A : (w_B:ℚ) / (w_X - w_B) = 5 :=
by 
  sorry

end ratio_B_to_A_l135_135744


namespace average_score_l135_135380

def s1 : ℕ := 65
def s2 : ℕ := 67
def s3 : ℕ := 76
def s4 : ℕ := 80
def s5 : ℕ := 95

theorem average_score : (s1 + s2 + s3 + s4 + s5) / 5 = 76.6 := by
  sorry

end average_score_l135_135380


namespace largest_x_value_l135_135059

theorem largest_x_value
  (x : ℝ)
  (h : (17 * x^2 - 46 * x + 21) / (5 * x - 3) + 7 * x = 8 * x - 2)
  : x = 5 / 3 :=
sorry

end largest_x_value_l135_135059


namespace painter_time_remaining_l135_135009

theorem painter_time_remaining (total_rooms : ℕ) (time_per_room : ℕ) (rooms_painted : ℕ) (remaining_hours : ℕ)
  (h1 : total_rooms = 12) (h2 : time_per_room = 7) (h3 : rooms_painted = 5) 
  (h4 : remaining_hours = (total_rooms - rooms_painted) * time_per_room) : 
  remaining_hours = 49 :=
by
  sorry

end painter_time_remaining_l135_135009


namespace root_sum_squares_bound_l135_135280

noncomputable def polynomial_bounds (f : Polynomial ℂ) (a : Fin n → ℂ) (z : Fin n → ℂ) :=
  (∑ k, ∥a k∥^2 ≤ 1) → (∑ k, ∥z k∥^2 ≤ n)

theorem root_sum_squares_bound (f : Polynomial ℂ)
  (h_roots : {z : Fin n → ℂ // ∀ i, f.eval (z i) = 0})
  (h_coeffs : {a : Fin n → ℂ // ∃ (p : Polynomial ℂ),
    p.coeffs = ( List.ofFn a.toFun ) }) :
  ∑ k in Finset.univ, ∥h_roots.val k∥^2 ≤ n :=
begin
  sorry
end

end root_sum_squares_bound_l135_135280


namespace correct_percentage_is_500_over_7_l135_135879

-- Given conditions
variable (x : ℕ)
def total_questions : ℕ := 7 * x
def missed_questions : ℕ := 2 * x

-- Definition of the fraction and percentage calculation
def correct_fraction : ℚ := (total_questions x - missed_questions x : ℕ) / total_questions x
def correct_percentage : ℚ := correct_fraction x * 100

-- The theorem to prove
theorem correct_percentage_is_500_over_7 : correct_percentage x = 500 / 7 :=
by
  -- Proof goes here
  sorry

end correct_percentage_is_500_over_7_l135_135879


namespace day_50_of_year_N_minus_1_l135_135574

-- Definitions for the problem conditions
def day_of_week (n : ℕ) : ℕ := n % 7

-- Given that the 250th day of year N is a Friday
axiom day_250_of_year_N_is_friday : day_of_week 250 = 5

-- Given that the 150th day of year N+1 is a Friday
axiom day_150_of_year_N_plus_1_is_friday : day_of_week 150 = 5

-- Calculate the day of the week for the 50th day of year N-1
theorem day_50_of_year_N_minus_1 :
  day_of_week 50 = 4 :=
  sorry

end day_50_of_year_N_minus_1_l135_135574


namespace handshake_remainder_l135_135182

noncomputable def handshakes (n : ℕ) (k : ℕ) : ℕ := sorry

theorem handshake_remainder :
  handshakes 12 3 % 1000 = 850 :=
sorry

end handshake_remainder_l135_135182


namespace largest_number_less_than_0_7_l135_135309

theorem largest_number_less_than_0_7 : 
  (∃ x ∈ {0.8, (1 : ℝ) / 2, 0.9, (1 : ℝ) / 3}, x < 0.7 ∧ (∀ y ∈ {0.8, (1 : ℝ) / 2, 0.9, (1 : ℝ) / 3}, y < 0.7 → y ≤ x)) → (1 : ℝ) / 2 = 0.5 :=
by
  sorry

end largest_number_less_than_0_7_l135_135309


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135327

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ (100 ≤ n ∧ n < 1000) ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) :=
by {
  sorry
}

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135327


namespace find_angle_C_l135_135868

variable {A B C a b c : ℝ}

theorem find_angle_C (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π)
  (h7 : A + B + C = π) (h8 : a > 0) (h9 : b > 0) (h10 : c > 0) 
  (h11 : (a + b - c) * (a + b + c) = a * b) : C = 2 * π / 3 :=
by
  sorry

end find_angle_C_l135_135868


namespace trains_cross_time_l135_135383

theorem trains_cross_time
  (L : ℝ) (T1 T2 : ℝ) (S1 S2 : ℝ) (CombinedLength RSpeed T : ℝ) :
  L = 120 ∧ T1 = 15 ∧ T2 = 20 ∧ 
  S1 = L / T1 ∧ S2 = L / T2 ∧
  RSpeed = S1 + S2 ∧ 
  CombinedLength = 2 * L ∧
  T = CombinedLength / RSpeed → 
  T ≈ 17.14 :=
by
  intros
  sorry

end trains_cross_time_l135_135383


namespace common_minimum_period_l135_135088

noncomputable def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f(x + T) = f(x)

theorem common_minimum_period (f : ℝ → ℝ) (h : ∀ x, f(x+4) + f(x-4) = f(x)) :
  ∃ T, is_periodic f T ∧ (∀ T', T' < T → ¬is_periodic f T') :=
sorry

end common_minimum_period_l135_135088


namespace sum_reciprocal_roots_l135_135985

noncomputable def problem_statement : Prop :=
  let a_poly := λ x : ℂ, x^2020 + x^2019 + ∑ i in (range 2020).succ, x^i - 2023 in
  let roots := {a : ℂ | a_poly a = 0} in
  ∑ a in roots, 1 / (1 - a) = 1354730

theorem sum_reciprocal_roots :
  problem_statement :=
by
  sorry

end sum_reciprocal_roots_l135_135985


namespace max_min_values_of_a_circle_D_equation_l135_135107

-- Theorems related to Part (Ⅰ)
theorem max_min_values_of_a (x y : ℝ) (h : (x - 3)^2 + (y - 4)^2 = 4) :
    (4 - 2 * sqrt 2 + 1) ≤ y - x ∧ y - x ≤ (4 + 2 * sqrt 2 - 2) :=
    sorry

-- Definitions and theorems related to Part (Ⅱ)
theorem circle_D_equation (a b : ℝ) (ha : a + b = 2) (r : ℝ) (hr : r = 3) :
    (a - 3)^2 + (b + 1)^2 = 9 ∨ (a + 2)^2 + (b - 4)^2 = 9 :=
    sorry


end max_min_values_of_a_circle_D_equation_l135_135107


namespace non_congruent_triangles_with_perimeter_10_l135_135153

theorem non_congruent_triangles_with_perimeter_10 :
  ∃ (T : Finset (Finset (ℕ × ℕ × ℕ))),
    (∀ (t ∈ T), let (a, b, c) := t in a ≤ b ∧ b ≤ c ∧
                  a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧
    T.card = 4 :=
by
  sorry

end non_congruent_triangles_with_perimeter_10_l135_135153


namespace sue_payment_is_900_l135_135624
noncomputable theory

def total_cost := 2100
def days_in_week := 7
def sister_days := 4
def sue_days := days_in_week - sister_days

def sue_fraction := (sue_days : ℚ) / days_in_week

def sue_payment := total_cost * sue_fraction

theorem sue_payment_is_900 : sue_payment = 900 := 
by
  sorry

end sue_payment_is_900_l135_135624


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135325

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((nat.digits 10 n).sum = 27) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ ((nat.digits 10 m).sum = 27) → m ≤ n) :=
begin
  use 999,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intro m,
    intro hm,
    cases hm,
    cases hm_left,
    cases hm_left_left,
    cases hm_left_right,
    cases hm_right,
    sorry
  },
sorry,
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135325


namespace tangent_lengths_sum_l135_135663

theorem tangent_lengths_sum 
    (R r : ℝ) -- Radii of larger and smaller circles respectively
    (A B C D : Point) -- Points representing vertices of the triangle and tangency point
    (l_A l_B l_C : ℝ) -- Tangent lengths from A, B, C to the smaller circle
    (circle_larger circle_smaller : Circle) -- The larger and smaller circles
    (circum_radius_eq : ∀ (P Q : Point), distance P Q = R) -- Equilateral triangle inscribed in the larger circle
    (tangency : Point → Circle → ℝ) -- Tangency function giving tangent lengths
    (hA : tangency A circle_smaller = l_A)
    (hB : tangency B circle_smaller = l_B)
    (hC : tangency C circle_smaller = l_C)
    : l_C = l_A + l_B := 
  sorry

end tangent_lengths_sum_l135_135663


namespace morgan_list_count_l135_135228

theorem morgan_list_count :
  ((900 / 30).to_int, (27000 / 30).to_int) = (30, 900) →
  (30.to_nat - 900.to_nat + 1) = 871 :=
by
  intro h
  sorry

end morgan_list_count_l135_135228


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135362

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ 100 ≤ n ∧ n < 1000 ∧ (9 ∣ n) ∧ (∑ digit in n.digits, digit = 27) :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135362


namespace hortense_flower_production_l135_135531

-- Define the initial conditions
def daisy_seeds : ℕ := 25
def sunflower_seeds : ℕ := 25
def daisy_germination_rate : ℚ := 0.60
def sunflower_germination_rate : ℚ := 0.80
def flower_production_rate : ℚ := 0.80

-- Prove the number of plants that produce flowers
theorem hortense_flower_production :
  (daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) * flower_production_rate = 28 :=
by sorry

end hortense_flower_production_l135_135531


namespace arithmetic_geometric_inequality_l135_135844

variables {a b A1 A2 G1 G2 x y d q : ℝ}
variables (h₀ : 0 < a) (h₁ : 0 < b)
variables (h₂ : a = x - 3 * d) (h₃ : A1 = x - d) (h₄ : A2 = x + d) (h₅ : b = x + 3 * d)
variables (h₆ : a = y / q^3) (h₇ : G1 = y / q) (h₈ : G2 = y * q) (h₉ : b = y * q^3)
variables (h₁₀ : x - 3 * d = y / q^3) (h₁₁ : x + 3 * d = y * q^3)

theorem arithmetic_geometric_inequality : A1 * A2 ≥ G1 * G2 :=
by {
  sorry
}

end arithmetic_geometric_inequality_l135_135844


namespace locus_of_midpoints_l135_135108

-- Definitions of the given conditions
variable (γ : Type) [MetricSpace γ] [NormedAddCommGroup γ] [NormedSpace ℝ γ]
variable (O A : γ)
variable (r : ℝ) (hO : ∃ B : γ, dist O B = r)

-- The statement we want to prove
theorem locus_of_midpoints (hA : dist O A > r) : 
  set_of (λ M : γ, ∃ B C : γ, 
    dist O B = r ∧ 
    dist O C = r ∧ 
    ∃ a b : ℝ, vector_between A B = a • (M - O) ∧ vector_between A C = b • (M - O) ∧ 
    dist B M = dist C M ∧ 
    M = (B + C) / 2) = {M : γ | dist O M = (dist O A) / 2} := 
by
  sorry

end locus_of_midpoints_l135_135108


namespace quadrilateral_DQPR_is_parallelogram_l135_135805

open EuclideanGeometry

/- The problem setup -/
variables {A B C O P Q R D : Point}
variables {abc : Triangle}
variable {circO : Circle}
variable [Circumcenter circO abc O]
variable [Extended A O P]
variable [AngleBisector P A angle B P C]
variable [Perpendicular PQ A B Q]
variable [Perpendicular PR A C R]
variable [Perpendicular D A B C]

/- Assumptions specific to the problem -/
axiom h1 : is_acute_triangle abc
axiom h2 : scalene abc
axiom h3 : A_O_extends_to_P
axiom h4 : PA_bisects_angle_BPC
axiom h5 : PQ_perpendicular_AB Q
axiom h6 : PR_perpendicular_AC R
axiom h7 : AD_perpendicular_BC

/- Statement to prove -/
theorem quadrilateral_DQPR_is_parallelogram 
  (h1 : is_acute_triangle abc)
  (h2 : scalene abc)
  (h3 : A_O_extends_to_P)
  (h4 : PA_bisects_angle_BPC)
  (h5 : PQ_perpendicular_AB Q)
  (h6 : PR_perpendicular_AC R)
  (h7 : AD_perpendicular_BC) :
  is_parallelogram D Q P R :=
by
  sorry

end quadrilateral_DQPR_is_parallelogram_l135_135805


namespace number_of_integers_fulfilling_condition_l135_135078

theorem number_of_integers_fulfilling_condition :
  ∃ (n_total : ℕ), n_total = 14520 ∧ ∀ n : ℤ,
    1 + (floor ((120 * n) / 121)) = ceil ((119 * n) / 120) → 
    n_total = 14520 :=
begin
  sorry
end

end number_of_integers_fulfilling_condition_l135_135078


namespace find_a_value_l135_135136

theorem find_a_value : 
  (∀ x, (3 * (x - 2) - 4 * (x - 5 / 4) = 0) ↔ ( ∃ a, ((2 * x - a) / 3 - (x - a) / 2 = x - 1) ∧ a = -11 )) := sorry

end find_a_value_l135_135136


namespace apartments_reduction_l135_135473

-- Definitions based on given conditions
def initial_entrances: ℕ := 5
def initial_floors_per_entrance: ℕ := 2
def apartments_per_floor: ℕ := 1

def first_modification_entrances (initial_entrances: ℕ) : ℕ := initial_entrances - 2
def first_modification_floors_per_entrance (initial_floors_per_entrance: ℕ) : ℕ := initial_floors_per_entrance + 3

def second_modification_entrances (first_mod_entrances: ℕ) : ℕ := first_mod_entrances - 2
def second_modification_floors_per_entrance (first_mod_floors: ℕ) : ℕ := first_mod_floors + 3

-- Initial number of apartments
def initial_apartments (initial_entrances initial_floors_per_entrance apartments_per_floor: ℕ): ℕ :=
  initial_entrances * initial_floors_per_entrance * apartments_per_floor

-- Number of apartments after first modification
def first_mod_apartments (entrances floors_per_entrance apartments_per_floor: ℕ) : ℕ := 
  entrances * floors_per_entrance * apartments_per_floor

-- Number of apartments after second modification
def second_mod_apartments (entrances floors_per_entrance apartments_per_floor: ℕ) : ℕ := 
  entrances * floors_per_entrance * apartments_per_floor

-- The theorem to prove
theorem apartments_reduction : 
  let initial_apts := initial_apartments initial_entrances initial_floors_per_entrance apartments_per_floor 
  let first_mod_entrances := first_modification_entrances initial_entrances
  let first_mod_floors := first_modification_floors_per_entrance initial_floors_per_entrance
  let second_mod_entrances := second_modification_entrances first_mod_entrances
  let second_mod_floors := second_modification_floors_per_entrance first_mod_floors
  let second_mod_apts := second_mod_apartments second_mod_entrances second_mod_floors apartments_per_floor
  in second_mod_apts < initial_apts := 
by {
  sorry
}

end apartments_reduction_l135_135473


namespace final_price_on_monday_l135_135999

open Real

def initial_price : Real := 50
def wednesday_increase_rate : Real := 0.15
def thursday_decrease_rate : Real := 0.05
def monday_discount_rate : Real := 0.20

theorem final_price_on_monday :
  let wednesday_price := initial_price * (1 + wednesday_increase_rate),
      thursday_price := wednesday_price * (1 - thursday_decrease_rate),
      discount := thursday_price * monday_discount_rate,
      monday_price := thursday_price - discount
  in
    monday_price = 43.7 :=
by
  sorry

end final_price_on_monday_l135_135999


namespace triangle_sides_angles_l135_135974

theorem triangle_sides_angles (a b c A B C : ℝ) (h1: A = 2 * B) 
  (h2: sin C * sin (A - B) = sin B * sin (C - A)) 
  (h3: A + B + C = π) :
  (C = 5 * π / 8) ∧ (2 * a^2 = b^2 + c^2) :=
by
  -- Proof omitted
  sorry

end triangle_sides_angles_l135_135974


namespace ordered_pair_solution_l135_135466

theorem ordered_pair_solution :
  ∃ x y : ℚ, 7 * x - 50 * y = 3 ∧ 3 * y - x = 5 ∧ x = -259 / 29 ∧ y = -38 / 29 :=
by sorry

end ordered_pair_solution_l135_135466


namespace part1_proof_part2_proof_l135_135951

-- Definitions for triangle sides and angles
variables {A B C a b c : ℝ}

-- Condition 1
def condition1 : Prop := sin C * sin (A - B) = sin B * sin (C - A)

-- Condition 2
def condition2 : Prop := A = 2 * B

-- Proof Problem 1
theorem part1_proof : condition1 → condition2 → C = 5 / 8 * π :=
by sorry

-- Proof Problem 2
theorem part2_proof : condition1 → condition2 → 2 * a^2 = b^2 + c^2 :=
by sorry

end part1_proof_part2_proof_l135_135951


namespace calculate_minimal_total_cost_l135_135870

structure GardenSection where
  area : ℕ
  flower_cost : ℚ

def garden := [
  GardenSection.mk 10 2.75, -- Orchids
  GardenSection.mk 14 2.25, -- Violets
  GardenSection.mk 14 1.50, -- Hyacinths
  GardenSection.mk 15 1.25, -- Tulips
  GardenSection.mk 25 0.75  -- Sunflowers
]

def total_cost (sections : List GardenSection) : ℚ :=
  sections.foldr (λ s acc => s.area * s.flower_cost + acc) 0

theorem calculate_minimal_total_cost :
  total_cost garden = 117.5 := by
  sorry

end calculate_minimal_total_cost_l135_135870


namespace gate_paid_more_l135_135418

def pre_booked_economy_cost : Nat := 10 * 140
def pre_booked_business_cost : Nat := 10 * 170
def total_pre_booked_cost : Nat := pre_booked_economy_cost + pre_booked_business_cost

def gate_economy_cost : Nat := 8 * 190
def gate_business_cost : Nat := 12 * 210
def gate_first_class_cost : Nat := 10 * 300
def total_gate_cost : Nat := gate_economy_cost + gate_business_cost + gate_first_class_cost

theorem gate_paid_more {gate_paid_more_cost : Nat} :
  total_gate_cost - total_pre_booked_cost = 3940 :=
by
  sorry

end gate_paid_more_l135_135418


namespace excircle_bisects_angle_bisector_l135_135570

/-- In the triangle \(ABC\) with the segment \(CL\) as the angle bisector. The \(C\)-excircle with center at the point \(I_C\) touches the side of the \(AB\) at the point \(D\) and the extensions of sides \(CA\) and \(CB\) at points \(P\) and \(Q\), respectively. It turned out that the length of the segment \(CD\) is equal to the radius of this excircle. Prove that the line \(PQ\) bisects the segment \(I_CL\). -/
theorem excircle_bisects_angle_bisector
  (ABC : Triangle)
  (CL : Segment)
  (I_C : Point)
  (D P Q : Point)
  (h1 : is_angle_bisector CL)
  (h2 : touches_excircle I_C D P Q)
  (h3 : length CD = radius_excircle I_C):
  bisects PQ (segment I_C L) :=
sorry

end excircle_bisects_angle_bisector_l135_135570


namespace part1_part2_l135_135964

variable {A B C a b c : ℝ}

theorem part1 (h₁ : A = 2 * B) (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : C = 5 / 8 * π :=
  sorry

theorem part2 (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
  sorry

end part1_part2_l135_135964


namespace exists_consec_2004_not_divisible_but_product_divisible_l135_135089

theorem exists_consec_2004_not_divisible_but_product_divisible 
  (n : ℕ) 
  (h : ∃ (seq_2010 : Fin 2010 → ℕ), 
    (∀ i, ¬ n ∣ seq_2010 i) ∧
    n ∣ (∏ i, seq_2010 i)) :
  ∃ (seq_2004 : Fin 2004 → ℕ), 
    (∀ j, ¬ n ∣ seq_2004 j) ∧
    n ∣ (∏ j, seq_2004 j) :=
sorry

end exists_consec_2004_not_divisible_but_product_divisible_l135_135089


namespace problem1_problem2_problem3_l135_135495

-- Definitions of arithmetic and geometric sequences
def arithmetic (a_n : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a_n n = a_n 0 + n * d
def geometric (b_n : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, b_n n = b_n 0 * q ^ n
def E (m p r : ℕ) := m < p ∧ p < r
def common_difference_greater_than_one (m p r : ℕ) := (p - m = r - p) ∧ (p - m > 1)

-- Problem (1)
theorem problem1 (a_n b_n : ℕ → ℝ) (d q : ℝ) (h₁: arithmetic a_n d) (h₂: geometric b_n q) (hne: q ≠ 1 ∧ q ≠ -1)
  (h: a_n 0 + b_n 1 = a_n 1 + b_n 2 ∧ a_n 1 + b_n 2 = a_n 2 + b_n 0) :
  q = -1/2 :=
sorry

-- Problem (2)
theorem problem2 (a_n b_n : ℕ → ℝ) (d q : ℝ) (m p r : ℕ) (h₁: arithmetic a_n d) (h₂: geometric b_n q) (hne: q ≠ 1 ∧ q ≠ -1)
  (hE: E m p r) (hDiff: common_difference_greater_than_one m p r)
  (h: a_n m + b_n p = a_n p + b_n r ∧ a_n p + b_n r = a_n r + b_n m) :
  q = - (1/2)^(1/3) :=
sorry

-- Problem (3)
theorem problem3 (a_n b_n : ℕ → ℝ) (m p r : ℕ) (hE: E m p r)
  (hG: ∀ n : ℕ, b_n n = (-1/2)^((n:ℕ)-1)) (h: a_n m + b_n m = 0 ∧ a_n p + b_n p = 0 ∧ a_n r + b_n r = 0) :
  ∃ (E : ℕ × ℕ × ℕ) (a : ℕ → ℝ), (E = ⟨1, 3, 4⟩ ∧ ∀ n : ℕ, a n = 3/8 * n - 11/8) :=
sorry

end problem1_problem2_problem3_l135_135495


namespace same_color_probability_l135_135164

def sides := 12
def violet_sides := 3
def orange_sides := 4
def lime_sides := 5

def prob_violet := violet_sides / sides
def prob_orange := orange_sides / sides
def prob_lime := lime_sides / sides

theorem same_color_probability :
  (prob_violet * prob_violet) + (prob_orange * prob_orange) + (prob_lime * prob_lime) = 25 / 72 :=
by
  sorry

end same_color_probability_l135_135164


namespace final_prices_correct_l135_135410

def discount (price : ℝ) (percent : ℝ) : ℝ :=
  price - (price * (percent / 100))

def apply_discounts_electronics (initial_price : ℝ) : ℝ :=
  let day1 := discount initial_price 10
  let day2 := discount day1 14
  let day3 := discount day2 12
  let day4 := discount day3 8
  day4

def apply_discounts_clothes (initial_price : ℝ) : ℝ :=
  let day1 := discount initial_price 10
  let day2 := discount day1 12
  let day3 := day2 - 20
  let day4 := discount day3 5
  day4

theorem final_prices_correct :
  apply_discounts_electronics 480 = 300.78 ∧ apply_discounts_clothes 260 = 176.62 :=
  by
    sorry

end final_prices_correct_l135_135410


namespace number_of_people_with_blue_eyes_l135_135654

theorem number_of_people_with_blue_eyes :
  let total_people : ℕ := 100 in
  let brown_eyes : ℕ := total_people / 2 in
  let black_eyes : ℕ := total_people / 4 in
  let green_eyes : ℕ := 6 in
  let total_non_blue_eyes : ℕ := brown_eyes + black_eyes + green_eyes in
  let blue_eyes : ℕ := total_people - total_non_blue_eyes in
  blue_eyes = 19 :=
by {
  sorry
}

end number_of_people_with_blue_eyes_l135_135654


namespace sin_arithmetic_sequence_l135_135769

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 2 * Real.pi) : 
  (Real.sin a + Real.sin (3 * a) = 2 * Real.sin (2 * a)) ↔ (a = Real.pi) :=
sorry

end sin_arithmetic_sequence_l135_135769


namespace range_of_h_sum_l135_135084

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 5*x^2)

theorem range_of_h_sum {a b : ℝ} 
  (h_range : ∀ y, (y ∈ set.Ioo a b ↔ ∃ x, h(x) = y)) : a + b = 1 :=
by
  have h_def : h = λ x, 3 / (3 + 5*x^2) := rfl
  sorry

end range_of_h_sum_l135_135084


namespace water_flow_volume_l135_135715

-- Definitions for conditions
def river_depth : ℝ := 2 -- depth in meters
def river_width : ℝ := 45 -- width in meters
def flow_rate_kmph : ℝ := 7 -- flow rate in kilometers per hour

-- Conversion constants
def km_to_m (km : ℝ) : ℝ := 1000 * km
def hour_to_min (hour : ℝ) : ℝ := 60 * hour

-- Flow rate in meters per minute
def flow_rate_m_per_min : ℝ := (km_to_m flow_rate_kmph) / hour_to_min 1

-- Cross-sectional area of the river
def cross_sectional_area : ℝ := river_depth * river_width

-- Volume of water flowing into the sea per minute
def volume_per_minute : ℝ := cross_sectional_area * flow_rate_m_per_min

theorem water_flow_volume :
  volume_per_minute = 10500.3 := by
  -- Placeholder for the proof; we know the answer from the conditions and solution
  sorry

end water_flow_volume_l135_135715


namespace exists_ten_digit_number_divisible_by_1980_l135_135207

theorem exists_ten_digit_number_divisible_by_1980 : 
  ∃ n : ℕ, (digits n) = [0,1,2,3,4,5,6,7,8,9] ∧ n % 1980 = 0 := by
sorry

end exists_ten_digit_number_divisible_by_1980_l135_135207


namespace length_of_integer_eq_24_l135_135780

theorem length_of_integer_eq_24 (k : ℕ) (h1 : k > 1) (h2 : ∃ (p1 p2 p3 p4 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ k = p1 * p2 * p3 * p4) : k = 24 := by
  sorry

end length_of_integer_eq_24_l135_135780


namespace find_value_at_2007_l135_135126

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f(x) = f(-x)
axiom symmetry_around_2 (x : ℝ) : f(2 + x) = f(2 - x)
axiom value_at_negative_3 : f (-3) = -2

theorem find_value_at_2007 : f (2007) = -2 := by
  sorry

end find_value_at_2007_l135_135126


namespace apples_needed_for_two_weeks_l135_135063

theorem apples_needed_for_two_weeks :
  ∀ (apples_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ),
  apples_per_day = 1 → days_per_week = 7 → weeks = 2 →
  apples_per_day * days_per_week * weeks = 14 :=
by
  intros apples_per_day days_per_week weeks h1 h2 h3
  sorry

end apples_needed_for_two_weeks_l135_135063


namespace number_of_n_l135_135081

theorem number_of_n (n : ℕ) (hn : n ≤ 500) (hk : ∃ k : ℕ, 21 * n = k^2) : 
  ∃ m : ℕ, m = 4 := by
  sorry

end number_of_n_l135_135081


namespace solve_grape_rate_l135_135734

noncomputable def grape_rate (G : ℝ) : Prop :=
  11 * G + 7 * 50 = 1428

theorem solve_grape_rate : ∃ G : ℝ, grape_rate G ∧ G = 98 :=
by
  exists 98
  sorry

end solve_grape_rate_l135_135734


namespace min_distance_from_vertex_l135_135419

theorem min_distance_from_vertex (A B C O : ℝ × ℝ) (R : ℝ) (h_triangle : IsAcuteTriangle A B C)
  (h_circumcenter : IsCircumcenter O A B C) (h_area : triangle_area A B C = 1) :
  ∃ P : ℝ × ℝ, P = O ∧ ∀ v ∈ {A, B, C}, dist P v ≥ 2 / Real.sqrt 3 :=
by
  sorry

end min_distance_from_vertex_l135_135419


namespace simplify_expression_l135_135616

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((a * cbrt b / (b * sqrt (a ^ 3))) ^ (3 / 2) + (sqrt a / (a * root 8 (b ^ 3))) ^ 2) / (a ^ (1 / 4) + b ^ (1 / 4)) = (1 / (a * b)) :=
by
  sorry

end simplify_expression_l135_135616


namespace length_four_implies_value_twenty_four_l135_135781

-- Definition of prime factors of an integer
def prime_factors (n : ℕ) : List ℕ := sorry

-- Definition of the length of an integer
def length_of_integer (n : ℕ) : ℕ :=
  List.length (prime_factors n)

-- Statement of the problem
theorem length_four_implies_value_twenty_four (k : ℕ) (h1 : k > 1) (h2 : length_of_integer k = 4) : k = 24 :=
by
  sorry

end length_four_implies_value_twenty_four_l135_135781


namespace find_retail_price_l135_135682

-- Define the wholesale price
def wholesale_price : ℝ := 90

-- Define the profit as 20% of the wholesale price
def profit (w : ℝ) : ℝ := 0.2 * w

-- Define the selling price as the wholesale price plus the profit
def selling_price (w p : ℝ) : ℝ := w + p

-- Define the selling price as 90% of the retail price t
def discount_selling_price (t : ℝ) : ℝ := 0.9 * t

-- Prove that the retail price t is 120 given the conditions
theorem find_retail_price :
  ∃ t : ℝ, wholesale_price + (profit wholesale_price) = discount_selling_price t → t = 120 :=
by
  sorry

end find_retail_price_l135_135682


namespace find_digit_B_l135_135303

theorem find_digit_B (B : ℕ) (h1 : B < 10) : 3 ∣ (5 + 2 + B + 6) → B = 2 :=
by
  sorry

end find_digit_B_l135_135303


namespace sum_of_sequence_l135_135436

noncomputable def sequence (n : ℕ) : ℕ → ℕ
| 0     := 12
| (n+1) := 4 * sequence n - n + 1

noncomputable def sum_sn (S : ℕ → ℕ) (n : ℕ) : ℕ :=
  (2 * 4 ^ n - 3 * n^2 - 3 * n - 26) / 6

theorem sum_of_sequence (S : ℕ → ℕ) (n : ℕ) (h1 : S 1 = sequence 1) :
  sum_sn S n = 
    (2 * 4 ^ n - 3 * n^2 - 3 * n - 26) / 6 := by
  sorry

end sum_of_sequence_l135_135436


namespace sqrt_fraction_simplification_l135_135036

theorem sqrt_fraction_simplification : (√18 - √2) / √2 = 2 := by
  sorry

end sqrt_fraction_simplification_l135_135036


namespace reading_time_difference_l135_135373

theorem reading_time_difference (xanthia_speed molly_speed book_length : ℕ)
  (hx : xanthia_speed = 120) (hm : molly_speed = 60) (hb : book_length = 300) :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = 150 :=
by
  -- We acknowledge the proof here would use the given values
  sorry

end reading_time_difference_l135_135373


namespace problem_statement_l135_135514

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x / Real.exp x
noncomputable def F (x : ℝ) : ℝ := f x - g x
noncomputable def m (x x₀ : ℝ) : ℝ := if x ≤ x₀ then f x else g x

-- Statement of the theorem
theorem problem_statement (x₀ x₁ x₂ n : ℝ) (hx₀ : x₀ ∈ Set.Ioo 1 2)
  (hF_root : F x₀ = 0)
  (hm_roots : m x₁ x₀ = n ∧ m x₂ x₀ = n ∧ 1 < x₁ ∧ x₁ < x₀ ∧ x₀ < x₂) :
  x₁ + x₂ > 2 * x₀ :=
sorry

end problem_statement_l135_135514


namespace number_of_right_angle_triangle_points_l135_135564

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (4, 3)

-- Definition of point C (x coordinate varies, y coordinate is 0 as it is on the x-axis)
def C (x : ℝ) : ℝ × ℝ := (x, 0)

-- Function to check if the triangle ABC is right-angled at C
def is_right_angle_triangle (A B C : ℝ × ℝ) : Prop :=
  let (ax, ay) := A
  let (bx, by) := B
  let (cx, cy) := C
  (ax - cx) * (bx - cx) + (ay - cy) * (by - cy) = 0

-- The main theorem statement
theorem number_of_right_angle_triangle_points : 
  (finset.card (finset.filter (λ x, is_right_angle_triangle A B (C x)) (finset.Icc (-2 : ℝ) 4))) = 3 :=
sorry

end number_of_right_angle_triangle_points_l135_135564


namespace max_factors_of_x10_minus_1_l135_135643

noncomputable def max_non_const_factors : ℕ :=
  let x := polynomial.C (1 : ℤ) in -- polynomials with real coefficients
  let p := x^10 - 1 in
  3

-- Prove that the largest possible value of m is 3
theorem max_factors_of_x10_minus_1 : max_non_const_factors = 3 := 
  sorry

end max_factors_of_x10_minus_1_l135_135643


namespace clock_angle_at_3_30_l135_135630

theorem clock_angle_at_3_30 :
  let angle_hour_hand := 90 + (30 / 60) * 30,
      angle_minute_hand := 30 * 6 in
  |angle_minute_hand - angle_hour_hand| = 75 :=
by
  let angle_hour_hand := 90 + (30 / 60) * 30
  let angle_minute_hand := 30 * 6
  have h1 : angle_hour_hand = 105 := by sorry
  have h2 : angle_minute_hand = 180 := by sorry
  have h3 : |180 - 105| = 75 := by sorry
  exact h3

end clock_angle_at_3_30_l135_135630


namespace part1_proof_part2_proof_l135_135944

-- Definitions for triangle sides and angles
variables {A B C a b c : ℝ}

-- Condition 1
def condition1 : Prop := sin C * sin (A - B) = sin B * sin (C - A)

-- Condition 2
def condition2 : Prop := A = 2 * B

-- Proof Problem 1
theorem part1_proof : condition1 → condition2 → C = 5 / 8 * π :=
by sorry

-- Proof Problem 2
theorem part2_proof : condition1 → condition2 → 2 * a^2 = b^2 + c^2 :=
by sorry

end part1_proof_part2_proof_l135_135944


namespace birthday_candles_l135_135742

def number_of_red_candles : ℕ := 18
def number_of_green_candles : ℕ := 37
def number_of_yellow_candles := number_of_red_candles / 2
def total_age : ℕ := 85
def total_candles_so_far := number_of_red_candles + number_of_yellow_candles + number_of_green_candles
def number_of_blue_candles := total_age - total_candles_so_far

theorem birthday_candles :
  number_of_yellow_candles = 9 ∧
  number_of_blue_candles = 21 ∧
  (number_of_red_candles + number_of_yellow_candles + number_of_green_candles + number_of_blue_candles) = total_age :=
by
  sorry

end birthday_candles_l135_135742


namespace good_time_equals_bad_time_l135_135271

-- Define the condition of "good" time in terms of angles
noncomputable def is_good_time (hour_angle minute_angle second_angle : ℝ) : Prop :=
  hour_angle < minute_angle ∧ minute_angle < second_angle

-- Define the angles made by the hour, minute, and second hands at time t (in hours)
noncomputable def hour_angle (t : ℝ) : ℝ := 30 * t + t / 120
noncomputable def minute_angle (t : ℝ) : ℝ := 6 * t
noncomputable def second_angle (t : ℝ) : ℝ := 360 * t

-- Define a 24-hour time range and the partition of "good" and "bad" times
def good_time_period (t : ℝ) : Prop :=
  is_good_time (hour_angle t) (minute_angle t) (second_angle t)

def bad_time_period (t : ℝ) : Prop :=
  ¬ good_time_period t

-- The theorem to be proved
theorem good_time_equals_bad_time :
  ∀ t : ℝ, 0 ≤ t ∧ t < 24 → (∫ t in 0..24, good_time_period t) = (∫ t in 0..24, bad_time_period t) :=
by
  sorry

end good_time_equals_bad_time_l135_135271


namespace number_of_valid_subsets_l135_135448

noncomputable def count_valid_subsets (n : ℕ) (h : n ≥ 2) : ℕ :=
  2^(n-1)

theorem number_of_valid_subsets (n : ℕ) (h : n ≥ 2) (A : set ℕ)
  (H_A : A = { k | 1 ≤ k ∧ k ≤ 2^n }) :
  (∃ B ⊆ A, (∀ x y ∈ B, x ≠ y → ¬(x + y == 2^m ∧ m ∈ ℕ)) ∧ (∀ x ∈ A, ∃ b ∈ B, x + b = 2^m ∧ m ∈ ℕ ∧ b ≠ x)) →
  count_valid_subsets n h = 2^(n-1) :=
sorry

end number_of_valid_subsets_l135_135448


namespace trigonometric_identities_l135_135479

theorem trigonometric_identities
  (theta : Real)
  (h1 : sin (theta - π / 3) = 1 / 3) :
  sin (theta + 2 * π / 3) = -1 / 3 ∧ cos (theta - 5 * π / 6) = 1 / 3 :=
by
  sorry

end trigonometric_identities_l135_135479


namespace solution_exists_l135_135717

noncomputable def sequence (b : ℕ → ℕ) : Prop :=
(∀ n, b (2 * n + 1) = b n) ∧ (∀ n, b (2 * n + 2) = b n + b (n + 1)) ∧ b 0 = 1

theorem solution_exists (b : ℕ → ℕ) (h : sequence b) : b 2015 = 6 := 
sorry

end solution_exists_l135_135717


namespace digits_right_of_decimal_l135_135533

theorem digits_right_of_decimal {x : ℚ} 
  (h : x = (5^8)/(12^5 * 625)) : 
  (number_of_digits_right_of_decimal x) = 9 := 
sorry

end digits_right_of_decimal_l135_135533


namespace problem_part1_problem_part2_l135_135941

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l135_135941


namespace find_C_prove_relation_l135_135958

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A), and A = 2B,
prove that C = 5/8 * π. -/
theorem find_C
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  C = ⅝ * Real.pi :=
sorry

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A),
prove that 2 * a ^ 2 = b ^ 2 + c ^ 2. -/
theorem prove_relation
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
sorry

end find_C_prove_relation_l135_135958


namespace find_smallest_n_l135_135003

theorem find_smallest_n : ∃ n : ℕ, (n - 4)^3 > (n^3 / 2) ∧ ∀ m : ℕ, m < n → (m - 4)^3 ≤ (m^3 / 2) :=
by
  sorry

end find_smallest_n_l135_135003


namespace quadratic_b_value_l135_135678

theorem quadratic_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + b * x - 12 < 0 ↔ x < 3 ∨ x > 7) → b = 10 :=
by 
  sorry

end quadratic_b_value_l135_135678


namespace solve_system_l135_135620

theorem solve_system {x y z : ℝ} (h1 : x^2 + y^2 = z^2) (h2 : x * z = y^2) (h3 : x * y = 10) :
  (x = sqrt 10 ∧ y = sqrt 10 ∧ z = sqrt 10) ∨ (x = -sqrt 10 ∧ y = -sqrt 10 ∧ z = -sqrt 10) :=
by {
  sorry
}

end solve_system_l135_135620


namespace opera_house_earnings_l135_135730

-- Definitions corresponding to the conditions
def num_rows : Nat := 150
def seats_per_row : Nat := 10
def ticket_cost : Nat := 10
def pct_not_taken : Nat := 20

-- Calculations based on conditions
def total_seats := num_rows * seats_per_row
def seats_not_taken := total_seats * pct_not_taken / 100
def seats_taken := total_seats - seats_not_taken
def earnings := seats_taken * ticket_cost

-- The theorem to prove
theorem opera_house_earnings : earnings = 12000 := sorry

end opera_house_earnings_l135_135730


namespace range_of_f_l135_135455

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+1) + 3

theorem range_of_f : Set.range f = Set.Ici 2 := 
by 
  sorry

end range_of_f_l135_135455


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135349

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) ∧ 
            ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ (m.digits.sum = 27) → m ≤ n :=
begin
  use 999,
  split,
  { -- 999 is a three-digit number 
    norm_num,
  },
  split,
  { -- 999 is less than or equal to 999
    norm_num,
  },
  split,
  { -- 999 is a multiple of 9
    norm_num,
  },
  split,
  { -- The sum of the digits of 999 is 27
    norm_num,
  },
  { -- For any three-digit number m, if it is a multiple of 9 and the sum of its digits is 27, then m ≤ 999
    intros m hm1,
    cases hm1 with hm2 hm3,
    cases hm3 with hm4 hm5,
    exact le_of_lt (by linarith),
    sorry
  },
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135349


namespace number_of_n_l135_135080

theorem number_of_n (n : ℕ) (hn : n ≤ 500) (hk : ∃ k : ℕ, 21 * n = k^2) : 
  ∃ m : ℕ, m = 4 := by
  sorry

end number_of_n_l135_135080


namespace smallest_possible_perimeter_after_folds_l135_135440

/-- 
Charlie folds a \(\frac{17}{2}\)-inch by 11-inch piece of paper in half twice, each time along a straight line parallel to one of the paper's edges.
We aim to prove that the smallest possible perimeter of the piece after two such folds is \(\frac{39}{2}\) inches.
-/
theorem smallest_possible_perimeter_after_folds: 
  let a := 17 / 2
  let b := 11
  (∀ p1 p2, 
     (p1 = 2 * a + 2 * (b / 4) ∨ p1 = 2 * (a / 4) + 2 * b) ∧ 
     (p2 = 2 * a + 2 * (b / 2) ∨ p2 = 2 * (a / 2) + 2 * b) →
     (min p1 p2) = 39 / 2
  ) sorry

end smallest_possible_perimeter_after_folds_l135_135440


namespace correlation_coefficients_l135_135694

-- Definition of the variables and constants
def relative_risks_starting_age : List (ℕ × ℝ) := [(16, 15.10), (18, 12.81), (20, 9.72), (22, 3.21)]
def relative_risks_cigarettes_per_day : List (ℕ × ℝ) := [(10, 7.5), (20, 9.5), (30, 16.6)]

def r1 : ℝ := -- The correlation coefficient between starting age and relative risk
  sorry

def r2 : ℝ := -- The correlation coefficient between number of cigarettes per day and relative risk
  sorry

theorem correlation_coefficients :
  r1 < 0 ∧ 0 < r2 :=
by {
  -- Proof is skipped with sorry
  sorry
}

end correlation_coefficients_l135_135694


namespace find_C_prove_relation_l135_135953

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A), and A = 2B,
prove that C = 5/8 * π. -/
theorem find_C
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  C = ⅝ * Real.pi :=
sorry

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A),
prove that 2 * a ^ 2 = b ^ 2 + c ^ 2. -/
theorem prove_relation
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
sorry

end find_C_prove_relation_l135_135953


namespace percentage_died_by_bombardment_l135_135560

-- Definitions
def initial_population : ℕ := 4079
def final_population : ℕ := 3294

-- Main statement to prove
theorem percentage_died_by_bombardment (x : ℚ) :
  initial_population - (x / 100) * initial_population - 0.15 * (initial_population - (x / 100) * initial_population) = final_population →
  x ≈ 4.99 :=
by
  sorry

end percentage_died_by_bombardment_l135_135560


namespace range_of_m_sum_of_zeros_l135_135544

noncomputable def y (x m : ℝ) : ℝ := Math.sin x - m / 2

theorem range_of_m (x₁ x₂ : ℝ) :
  (∃ x₁ x₂ ∈ Set.Icc (Real.pi / 3) (4 * Real.pi / 3), y x₁ m = 0 ∧ y x₂ m = 0 ∧ x₁ ≠ x₂) →
  sqrt 3 ≤ m ∧ m < 2 :=
sorry

theorem sum_of_zeros (x₁ x₂ : ℝ) (m : ℝ) :
  (∃ x₁ x₂ ∈ Set.Icc (Real.pi / 3) (4 * Real.pi / 3), y x₁ m = 0 ∧ y x₂ m = 0 ∧ x₁ ≠ x₂) →
  x₁ + x₂ = Real.pi :=
sorry

end range_of_m_sum_of_zeros_l135_135544


namespace eval_expression_l135_135760

theorem eval_expression :
  (2011 * (2012 * 10001) * (2013 * 100010001)) - (2013 * (2011 * 10001) * (2012 * 100010001)) =
  -2 * 2012 * 2013 * 10001 * 100010001 :=
by
  sorry

end eval_expression_l135_135760


namespace sum_of_triangles_l135_135257

def triangle (a b c : ℕ) : ℕ :=
  (a * b) + c

theorem sum_of_triangles : 
  triangle 4 2 3 + triangle 5 3 2 = 28 :=
by
  sorry

end sum_of_triangles_l135_135257


namespace find_coefficients_sum_l135_135812

theorem find_coefficients_sum :
  let f := (2 * x - 1) ^ 5 + (x + 2) ^ 4
  let a_0 := 15
  let a_1 := 42
  let a_2 := -16
  let a_5 := 32
  (|a_0| + |a_1| + |a_2| + |a_5| = 105) := 
by {
  sorry
}

end find_coefficients_sum_l135_135812


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135360

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ 100 ≤ n ∧ n < 1000 ∧ (9 ∣ n) ∧ (∑ digit in n.digits, digit = 27) :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135360


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135355

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135355


namespace y_is_defined_iff_x_not_equal_to_10_l135_135281

def range_of_independent_variable (x : ℝ) : Prop :=
  x ≠ 10

theorem y_is_defined_iff_x_not_equal_to_10 (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 10)) ↔ range_of_independent_variable x :=
by sorry

end y_is_defined_iff_x_not_equal_to_10_l135_135281


namespace exists_regular_1990_gon_l135_135208

-- Definition of the problem: existence of a 1990-gon with specific properties
theorem exists_regular_1990_gon
  (A : ℕ → ℝ × ℝ)
  (equal_angles : ∀ i, angle (A i) (A (i + 1 mod 1990)) (A (i + 2 mod 1990)) = 2 * π / 1990)
  (side_lengths_perm : ∃ (perm : {x // x ∈ finset.range 1990 | 1 ≤ x + 1} → ℕ),
    (∀ (i : {x // x ∈ finset.range 1990 | 1 ≤ x + 1}),
     (dist (A i) (A (i + 1 mod 1990))) = (perm i + 1) ^ 2) ∧
    finset.univ.image (λ i, perm i) = finset.image (pow 2) (finset.range 1990))) :
  ∃ (A : Π i : fin (1990), ℝ × ℝ), true :=
sorry

end exists_regular_1990_gon_l135_135208


namespace regression_a_value_l135_135516

noncomputable def a_value (n : Nat) (_ : n = 11) 
  (sum_x : ℝ) (_ : sum_x = 66) 
  (sum_y : ℝ) (_ : sum_y = 132)
  (regression_eq : ∀ x y, y = 0.3 * x + a) : ℝ :=
  12 - 0.3 * (sum_x / n)

theorem regression_a_value : 
  a_value 11 rfl 66 rfl 132 rfl (λ x y, y = 0.3 * x + a) = 10.2 :=
sorry

end regression_a_value_l135_135516


namespace ratio_siblings_l135_135577

theorem ratio_siblings (M J C : ℕ) 
  (hM : M = 60)
  (hJ : J = 4 * M - 60)
  (hJ_C : J = C + 135) :
  (C : ℚ) / M = 3 / 4 :=
by
  sorry

end ratio_siblings_l135_135577


namespace xyz_value_l135_135811

theorem xyz_value (x y z : ℝ)
    (h1 : (x + y + z) * (x * y + x * z + y * z) = 30)
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 14) :
    x * y * z = 16 / 3 := by
    sorry

end xyz_value_l135_135811


namespace not_suitable_for_storing_l135_135289

-- Define the acceptable temperature range conditions for storing dumplings
def acceptable_range (t : ℤ) : Prop :=
  -20 ≤ t ∧ t ≤ -16

-- Define the specific temperatures under consideration
def temp_A : ℤ := -17
def temp_B : ℤ := -18
def temp_C : ℤ := -19
def temp_D : ℤ := -22

-- Define a theorem stating that temp_D is not in the acceptable range
theorem not_suitable_for_storing (t : ℤ) (h : t = temp_D) : ¬ acceptable_range t :=
by {
  sorry
}

end not_suitable_for_storing_l135_135289


namespace num_perfect_square_21n_le_500_l135_135083

theorem num_perfect_square_21n_le_500 : 
  {n : ℕ | n ≤ 500 ∧ ∃ k : ℕ, 21 * n = k ^ 2}.to_finset.card = 4 := 
by sorry

end num_perfect_square_21n_le_500_l135_135083


namespace range_of_m_l135_135537

theorem range_of_m (x m : ℝ) (h1 : -1 ≤ x ∧ x ≤ 1) (h2 : |x - m| ≤ 2) : -1 ≤ m ∧ m ≤ 1 :=
sorry

end range_of_m_l135_135537


namespace proof_problem_l135_135863

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

theorem proof_problem 
  (h_deriv : ∀ x, f'(x) = derivative f x)
  (h_ineq : ∀ x, f(x) < f'(x) - 2) :
  f(2022) - real.exp(1) * f(2021) > 2 * (real.exp(1) - 1) := 
sorry

end proof_problem_l135_135863


namespace problem_statement_l135_135212

def S : Finset ℕ  -- assuming set of size 11
def s : Fin 12 → S  -- a random 12-tuple (s_1, s_2, ..., s_12)
def π : Equiv.Perm S  -- a permutation of S

theorem problem_statement :
  let a := 10 ^ 12 + 4 in
  (∑ i in Finset.range 12, ite (s(i + 1) = π(s(i))) (0 : ℕ) 1) • _ = _ :=
sorry

end problem_statement_l135_135212


namespace fred_total_earnings_l135_135784

def fred_earnings (earnings_per_hour hours_worked : ℝ) : ℝ := earnings_per_hour * hours_worked

theorem fred_total_earnings :
  fred_earnings 12.5 8 = 100 := by
sorry

end fred_total_earnings_l135_135784


namespace digit_1997th_decimal_place_1_over_22_l135_135672

theorem digit_1997th_decimal_place_1_over_22 : 
  (let repeating_sequence := "045" in repeating_sequence[1997 % repeating_sequence.length] = '0') :=
by
  sorry

end digit_1997th_decimal_place_1_over_22_l135_135672


namespace find_line_equation_l135_135502

/-
Given:
  - A circle with equation (x - 3)^2 + (y - 2)^2 = 4
  - A line passing through the point (2, 0)
  - The chord length intercepted by this circle is 2√3

Prove:
  The equation of the line is either x = 2 or 3x - 4y - 6 = 0.
-/

noncomputable def circle_center : ℝ × ℝ := (3, 2)
noncomputable def circle_radius : ℝ := 2
noncomputable def point_P : ℝ × ℝ := (2, 0)
noncomputable def chord_length : ℝ := 2 * Real.sqrt 3

theorem find_line_equation :
  (∃ k : ℝ, ∀ x y : ℝ, (y = k * (x - 2) ∨ x = 2) ∧ 
                    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 4) ∧
                    ((chord_length / 2)^2 + (distance circle_center (midpoint point_P ⟨x, y⟩))^2 = circle_radius^2)) →
  (∀ x y : ℝ, (x = 2 ∨ 3 * x - 4 * y - 6 = 0)) := sorry

end find_line_equation_l135_135502


namespace problem_l135_135794

theorem problem (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f(x) = x^5 + a*x^3 + b*x - 8)
  (h2 : f(-2) = 10) : 
  f(2) = -26 := by
sorry

end problem_l135_135794


namespace total_distance_in_yards_l135_135420

variables (a r : ℝ)

theorem total_distance_in_yards :
  (let first_half_distance := a / 4 in
   let first_half_time := 2 * r in
   let first_half_speed := first_half_distance / first_half_time in
   let second_half_speed := 2 * first_half_speed in
   let second_half_time := 120 in
   let second_half_distance := second_half_speed * second_half_time in
   let total_distance_feet := first_half_distance + second_half_distance in
   let total_distance_yards := total_distance_feet / 3 in
   total_distance_yards = (121 * a) / 12) :=
by sorry

end total_distance_in_yards_l135_135420


namespace simplify_expression_l135_135253

theorem simplify_expression (i : ℂ) (h : i^2 = -1) : 
  3 * (4 - 2 * i) + 2 * i * (3 + i) + 5 * (-1 + i) = 5 + 5 * i :=
by
  sorry

end simplify_expression_l135_135253


namespace real_part_of_z_l135_135635

theorem real_part_of_z (z : ℂ) (h : z * (1 + complex.I) = -1) : z.re = -1/2 :=
sorry

end real_part_of_z_l135_135635


namespace sufficient_but_not_necessary_condition_l135_135193

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : b > a) (h2 : a > 0) : 
  (a * (b + 1) > a^2) ∧ ¬(∀ (a b : ℝ), a * (b + 1) > a^2 → b > a ∧ a > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l135_135193


namespace part1_C_value_part2_triangle_equality_l135_135976

noncomputable theory

variables (a b c : ℝ) (A B C : ℝ)
variables (h1 : A + B + C = Real.pi) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) (h3 : A = 2 * B)

-- Part 1: Proving that C = 5π/8 given the conditions
theorem part1_C_value :
  C = 5 * Real.pi / 8 :=
begin
  sorry
end

-- Part 2: Proving that 2a^2 = b^2 + c^2 given the conditions
theorem part2_triangle_equality :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
begin
  sorry
end

end part1_C_value_part2_triangle_equality_l135_135976


namespace part1_C_value_part2_triangle_equality_l135_135980

noncomputable theory

variables (a b c : ℝ) (A B C : ℝ)
variables (h1 : A + B + C = Real.pi) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) (h3 : A = 2 * B)

-- Part 1: Proving that C = 5π/8 given the conditions
theorem part1_C_value :
  C = 5 * Real.pi / 8 :=
begin
  sorry
end

-- Part 2: Proving that 2a^2 = b^2 + c^2 given the conditions
theorem part2_triangle_equality :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
begin
  sorry
end

end part1_C_value_part2_triangle_equality_l135_135980


namespace min_correct_answers_l135_135880

theorem min_correct_answers (x : ℕ) (hx : 10 * x - 5 * (30 - x) > 90) : x ≥ 17 :=
by {
  -- calculations and solution steps go here.
  sorry
}

end min_correct_answers_l135_135880


namespace cos_identity_l135_135120

theorem cos_identity (α : ℝ) (h : Real.cos (π / 4 - α) = -1 / 3) :
  Real.cos (3 * π / 4 + α) = 1 / 3 :=
sorry

end cos_identity_l135_135120


namespace age_difference_l135_135422

theorem age_difference (A B n : ℕ) (h1 : A = B + n) (h2 : A - 1 = 3 * (B - 1)) (h3 : A = B^2) : n = 2 :=
by
  sorry

end age_difference_l135_135422


namespace inequality_proof_l135_135240

variable (ha la r R : ℝ)
variable (α β γ : ℝ)

-- Conditions
def condition1 : Prop := ha / la = Real.cos ((β - γ) / 2)
def condition2 : Prop := 8 * Real.sin (α / 2) * Real.sin (β / 2) * Real.sin (γ / 2) = 2 * r / R

-- The theorem to be proved
theorem inequality_proof (h1 : condition1 ha la β γ) (h2 : condition2 α β γ r R) :
  Real.cos ((β - γ) / 2) ≥ Real.sqrt (2 * r / R) :=
sorry

end inequality_proof_l135_135240


namespace bd_correct_l135_135371

open Real EuclideanSpace 

-- Define vector operations and conditions
variables (a b : EuclideanSpace) (m : ℝ)

-- Define vector a and b for condition C.
def vectorA : EuclideanSpace := ![1, 2]
def vectorB : EuclideanSpace := ![m, 1]

-- Define correctness properties for B and D.
def statement_B_is_correct : Prop :=
  ∀ (u v : EuclideanSpace), u ≠ 0 ∧ v ≠ 0 ∧ ∃ k, u = -k • v → (u - k • v = 0)

def statement_D_is_correct : Prop :=
  ∀ (a b : EuclideanSpace), 
    a ≠ 0 ∧ b ≠ 0 ∧ (∥a + b∥ = ∥a - b∥) → (∠(a, b) = π / 2)

-- The main theorem stating B and D are correct.
theorem bd_correct : statement_B_is_correct ∧ statement_D_is_correct :=
by sorry

end bd_correct_l135_135371


namespace largest_expression_depends_on_values_l135_135679

theorem largest_expression_depends_on_values (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a ≠ b) :
  ¬(∀ a b : ℝ, (h₀ : a > 0) → (h₁ : b > 0) → (h₂ : a ≠ b) → 
    (let expr_I := (a + 1/a) * (b + 1/b) in
     let expr_II := (Real.sqrt(a * b) + 1/Real.sqrt(a * b))^2 in
     let expr_III := ((a + b)/2 + 2/(a + b))^2 in
     ∃ max_expr : ℝ, (max_expr = expr_I ∨ max_expr = expr_II ∨ max_expr = expr_III) 
     ∧ (∀ (e: ℝ), e = expr_I ∨ e = expr_II ∨ e = expr_III → e ≤ max_expr))) :=
sorry

end largest_expression_depends_on_values_l135_135679


namespace remainder_zero_division_l135_135061

theorem remainder_zero_division :
  ∀ x : ℂ, (x^2 - x + 1 = 0) →
    ((x^5 + x^4 - x^3 - x^2 + 1) * (x^3 - 1)) % (x^2 - x + 1) = 0 :=
by sorry

end remainder_zero_division_l135_135061


namespace find_tu_l135_135843

variables {V : Type*} [inner_product_space ℝ V]
variables (a b p : V)

-- Condition: Norm equality
def norm_eq_condition : Prop :=
  ∥p - b∥ = 3 * ∥p - a∥

-- Goal: Prove the existence of the pair (t, u) == (9/8, -1/8)
theorem find_tu (h : norm_eq_condition) : 
  ∃ t u : ℝ, t = 9 / 8 ∧ u = -1 / 8 := 
sorry

end find_tu_l135_135843


namespace jerry_needs_more_figures_l135_135579

theorem jerry_needs_more_figures (initial_figures added_figures initial_books final_pattern_figures : ℕ)
  (hf : initial_figures = 5)
  (ha : added_figures = 7)
  (hb : initial_books = 9)
  (hf_pattern : final_pattern_figures = 2)
  (hc : final_pattern_figures * initial_books > initial_figures + added_figures) :
  ∃ (additional_figures : ℕ), additional_figures = final_pattern_figures * initial_books - (initial_figures + added_figures) :=
by
  existsi (final_pattern_figures * initial_books - (initial_figures + added_figures))
  have hfig : initial_figures + added_figures = 12
  simp [hf, ha, hfig, final_pattern_figures, hb]
  sorry

end jerry_needs_more_figures_l135_135579


namespace rectangular_to_polar_l135_135449

theorem rectangular_to_polar (x y : ℝ) (hx : x = 1) (hy : y = -1) :
  ∃ r θ : ℝ, r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 
  r = Real.sqrt (x^2 + y^2) ∧ θ = if y ≥ 0 then Real.atan (y / x) else 2 * Real.pi + Real.atan (y / x) :=
by 
  use [Real.sqrt 2, 7 * Real.pi / 4]
  have : Real.sqrt (1^2 + (-1)^2) = Real.sqrt 2 := by norm_num [Real.sqrt]
  have : if -1 ≥ 0 then Real.atan (-1 / 1) else 2 * Real.pi + Real.atan (-1 / 1) = 7 * Real.pi / 4 := by norm_num [Real.atan]
  split
  · norm_num
  split
  · linarith [Real.pi_pos]
  split
  · linarith
  split
  · exact this
  exact this

end rectangular_to_polar_l135_135449


namespace problem_solution_l135_135540

/-- Define the problem conditions --/
variables (p1 p2 p_neither p_both : ℝ)

/-- Assign specific values to the conditions --/
def problem_conditions : Prop :=
  p1 = 75 ∧ p2 = 30 ∧ p_neither = 20 ∧ (100 - p_neither) = (p1 + p2 - p_both)

/-- Problem statement based on given conditions --/
theorem problem_solution (h : problem_conditions p1 p2 p_neither p_both) : p_both = 25 :=
by
  /- Assume the conditions -/
  rcases h with ⟨hp1, hp2, hp_neither, hp_at_least_one⟩
  /- Proof omitted -/
  sorry

end problem_solution_l135_135540


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135351

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135351


namespace polynomial_same_root_count_l135_135591

open Complex

variables {f g : ℂ → ℂ} {γ : ℂ → ℂ}

def closed_non_self_intersecting (γ : ℂ → ℂ) : Prop := sorry -- This would need a proper definition

theorem polynomial_same_root_count 
  (hf : ∀ z, Polynomial f) 
  (hg : ∀ z, Polynomial g) 
  (hγ : closed_non_self_intersecting γ) 
  (hcond : ∀ z ∈ {z : ℂ | γ z}, ∥f z - g z∥ < ∥f z∥ + ∥g z∥) : 
  (∑ z in (Finset.univ.filter (λ z, ∃ x, f x = 0 ∧ x = z)), 1) = 
  (∑ z in (Finset.univ.filter (λ z, ∃ x, g x = 0 ∧ x = z)), 1) := 
sorry

end polynomial_same_root_count_l135_135591


namespace problem1_problem2_problem3_l135_135148

-- Mathematical definitions translated from the problem
def seq_a (a : ℝ) : ℕ → ℝ
| 0       := a
| (n + 1) := 0 -- Placeholder, this will depend on further context

def seq_b (a : ℝ) : ℕ → ℝ
| 0       := 1
| (n + 1) := 0 -- Placeholder, to be defined from the conditions

def seq_c (a : ℝ) : ℕ → ℝ
| 0       := 3
| (n + 1) := 0 -- Placeholder, to be defined from the conditions

-- Problem 1: General formula for the sequence {c_n - b_n}
theorem problem1 (a : ℝ) :
  (∀ n, seq_b a (n + 1) = (seq_a a n + seq_c a n) / 2) →
  (∀ n, seq_c a (n + 1) = (seq_a a n + seq_b a n) / 2) →
  (∀ n, seq_c a n - seq_b a n = 2 * (- 1/2)^(n - 1)) :=
sorry

-- Problem 2: Value of a if sequences {a_n} and {b_n + c_n} are constant
theorem problem2 :
  (∀ n, seq_a a n = a) →
  (∀ n, seq_b a n + seq_c a n = 4) →
  a = 2 :=
sorry

-- Problem 3: Range of values for a such that M_n < 5/2
theorem problem3 (S_n T_n M_n : ℕ → ℝ) (a : ℝ) :
  (∀ n, seq_a a n = a^n) →
  (∀ n, S_n n = (∑ i in finset.range (n + 1), seq_b a i)) →
  (∀ n, T_n n = (∑ i in finset.range n, seq_c a i)) →
  (∀ n, M_n n = 2 * S_n (n + 1) - T_n n) →
  (∀ n, M_n n < 5/2) →
  -1 < a ∧ a < 0 ∨ 0 < a ∧ a ≤ 1/3 :=
sorry

end problem1_problem2_problem3_l135_135148


namespace minimum_y_value_inequality_proof_l135_135480
-- Import necessary Lean library

-- Define a > 0, b > 0, and a + b = 1
variables {a b : ℝ}
variables (h_a_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 1)

-- Statement for Part (I): Prove the minimum value of y is 25/4
theorem minimum_y_value :
  (a + 1/a) * (b + 1/b) = 25/4 :=
sorry

-- Statement for Part (II): Prove the inequality
theorem inequality_proof :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 :=
sorry

end minimum_y_value_inequality_proof_l135_135480


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135357

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135357


namespace expression_equals_5_l135_135434

def expression_value : ℤ := 8 + 15 / 3 - 2^3

theorem expression_equals_5 : expression_value = 5 :=
by
  sorry

end expression_equals_5_l135_135434


namespace find_initial_terms_l135_135113

theorem find_initial_terms (a : ℕ → ℕ) (h : ∀ n, a (n + 3) = a (n + 2) * (a (n + 1) + 2 * a n))
  (a6 : a 6 = 2288) : a 1 = 5 ∧ a 2 = 1 ∧ a 3 = 2 :=
by
  sorry

end find_initial_terms_l135_135113


namespace point_symmetric_second_quadrant_l135_135866

theorem point_symmetric_second_quadrant (m : ℝ) 
  (symmetry : ∃ x y : ℝ, P = (-m, m-3) ∧ (-x, -y) = (x, y)) 
  (second_quadrant : ∃ x y : ℝ, P = (-m, m-3) ∧ x < 0 ∧ y > 0) : 
  m < 0 := 
sorry

end point_symmetric_second_quadrant_l135_135866


namespace Jenny_recycling_l135_135578

theorem Jenny_recycling:
  let bottle_weight := 6
  let can_weight := 2
  let glass_jar_weight := 8
  let max_weight := 100
  let num_cans := 20
  let bottle_value := 10
  let can_value := 3
  let glass_jar_value := 12
  let total_money := (num_cans * can_value) + (7 * glass_jar_value) + (0 * bottle_value)
  total_money = 144 ∧ num_cans = 20 ∧ glass_jars = 7 ∧ bottles = 0 := by sorry

end Jenny_recycling_l135_135578


namespace odd_function_phi_value_l135_135169

open Real

theorem odd_function_phi_value (f : ℝ → ℝ) (φ : ℝ) (h1 : ∀ x, f(x) = 2 * sin (x + φ) - cos x) (h2 : ∀ x, f (-x) = -f x) : φ = π / 6 :=
by
  sorry

end odd_function_phi_value_l135_135169


namespace area_of_ABCD_l135_135741

def area_approximation (h0 h1 h2 h3 h4 h5 : ℝ) (d : ℝ) : ℝ :=
  d * (h0 + h1 + h2 + h3 + h4 + h5)

theorem area_of_ABCD :
  let d := 2 in
  let h0 := 5 in
  let h1 := 5.25 in
  let h2 := 5.5 in
  let h3 := 5.6 in
  let h4 := 5.5 in
  let h5 := 5.5 in
  area_approximation h0 h1 h2 h3 h4 h5 d * 2 = 53.7 :=
by
  sorry

end area_of_ABCD_l135_135741


namespace sum_of_products_inequality_l135_135251

theorem sum_of_products_inequality {n : ℕ} (hpos : 0 < n) 
  (a b : Fin n → ℝ) (ha : ∀ i j, i ≤ j → a i ≥ a j) 
  (hb : ∀ i j, i ≤ j → b i ≥ b j) : 
  (∑ i, (a i * b i)) ≥ (1 / n) * (∑ i, a i) * (∑ i, b i) := 
by 
  sorry

end sum_of_products_inequality_l135_135251


namespace non_congruent_triangles_with_perimeter_10_l135_135154

theorem non_congruent_triangles_with_perimeter_10 :
  ∃ (T : Finset (Finset (ℕ × ℕ × ℕ))),
    (∀ (t ∈ T), let (a, b, c) := t in a ≤ b ∧ b ≤ c ∧
                  a + b + c = 10 ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∧
    T.card = 4 :=
by
  sorry

end non_congruent_triangles_with_perimeter_10_l135_135154


namespace find_h_l135_135867

theorem find_h (x : ℝ) : 
  ∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - (-3 / 2))^2 + k :=
sorry

end find_h_l135_135867


namespace necessary_but_not_sufficient_l135_135456

theorem necessary_but_not_sufficient (x : ℝ) :
  ((x > 0) ↔ ((x-2)*(x-4) < 0)) ↔ (x ∈ (-∞, 2) ∪ (4, ∞)) :=
by
  sorry

end necessary_but_not_sufficient_l135_135456


namespace each_boy_brought_nine_cups_l135_135293

/--
There are 30 students in Ms. Leech's class. Twice as many girls as boys are in the class.
There are 10 boys in the class and the total number of cups brought by the students 
in the class is 90. Prove that each boy brought 9 cups.
-/
theorem each_boy_brought_nine_cups (students girls boys cups : ℕ) 
  (h1 : students = 30) 
  (h2 : girls = 2 * boys) 
  (h3 : boys = 10) 
  (h4 : cups = 90) 
  : cups / boys = 9 := 
sorry

end each_boy_brought_nine_cups_l135_135293


namespace problem_part1_problem_part2_l135_135942

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l135_135942


namespace max_perimeter_l135_135202

-- Assume variables and given conditions
variables {A B C : Real} {a b c : Real}

-- Given conditions
def conditions : Prop :=
  (a = 2) ∧ (a * Real.sin B = Real.sqrt 3 * b * Real.cos A)

-- The theorem to prove
theorem max_perimeter : conditions → a + b + c ≤ 6 :=
sorry

end max_perimeter_l135_135202


namespace weight_loss_percentage_at_final_weigh_in_l135_135689

-- Let W be the initial weight.
variable (W : ℝ)

-- Define the conditions of the problem
def initial_weight := W
def weight_lost := 0.13 * W
def weight_after_loss := W - weight_lost
def additional_weight_due_to_clothes := 0.02 * weight_after_loss
def weight_at_final_weigh_in := weight_after_loss + additional_weight_due_to_clothes
def measured_weight_loss := initial_weight - weight_at_final_weigh_in
def measured_percentage_weight_loss := (measured_weight_loss / initial_weight) * 100

-- Prove that the measured percentage weight loss is 11.26%
theorem weight_loss_percentage_at_final_weigh_in : 
  measured_percentage_weight_loss initial_weight weight_lost weight_after_loss additional_weight_due_to_clothes weight_at_final_weigh_in measured_weight_loss = 11.26 := 
sorry

end weight_loss_percentage_at_final_weigh_in_l135_135689


namespace vampire_needs_7_gallons_per_week_l135_135414

-- Define conditions given in the problem
def pints_per_person : ℕ := 2
def people_per_day : ℕ := 4
def days_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8

-- Prove the vampire needs 7 gallons of blood per week to survive
theorem vampire_needs_7_gallons_per_week :
  (pints_per_person * people_per_day * days_per_week) / pints_per_gallon = 7 := 
by 
  sorry

end vampire_needs_7_gallons_per_week_l135_135414


namespace original_price_of_petrol_l135_135011

theorem original_price_of_petrol (P : ℝ) (h : 0.9 * P * 190 / (0.9 * P) = 190 / P + 5) : P = 4.22 :=
by
  -- The proof goes here
  sorry

end original_price_of_petrol_l135_135011


namespace inradius_sum_l135_135797

theorem inradius_sum {A B C D : Point} (h : CircumscribedQuadrilateral A B C D) :
  let r := inradius A B C D,
      r1 := inradius_triangle A B C,
      r2 := inradius_triangle A C D
  in r < r1 + r2 :=
by sorry

end inradius_sum_l135_135797


namespace remainder_correct_l135_135776

noncomputable def p (x : ℝ) : ℝ := 3 * x ^ 8 - 2 * x ^ 5 + 5 * x ^ 3 - 9
noncomputable def d (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1
noncomputable def r (x : ℝ) : ℝ := 29 * x - 32

theorem remainder_correct (x : ℝ) :
  ∃ q : ℝ → ℝ, p x = d x * q x + r x :=
sorry

end remainder_correct_l135_135776


namespace adult_meals_sold_l135_135026

theorem adult_meals_sold (k a : ℕ) (h1 : 10 * a = 7 * k) (h2 : k = 70) : a = 49 :=
by
  sorry

end adult_meals_sold_l135_135026


namespace pipe_B_leak_time_l135_135000

noncomputable def pipe_fill_time : ℝ := 10
noncomputable def combined_fill_time : ℝ := 29.999999999999993

theorem pipe_B_leak_time :
  ∃ T : ℝ, (1 / pipe_fill_time) - (1 / T) = 1 / combined_fill_time ∧ T = 15 :=
by
  existsi (15 : ℝ)
  split
  sorry

end pipe_B_leak_time_l135_135000


namespace rocky_first_round_knockouts_l135_135611

-- Definitions
def total_fights : ℕ := 190
def knockout_percentage : ℝ := 0.50
def first_round_knockout_percentage : ℝ := 0.20

-- Calculation of knockouts
def total_knockouts : ℕ := (total_fights * knockout_percentage).to_nat
def first_round_knockouts : ℕ := (total_knockouts * first_round_knockout_percentage).to_nat

-- Theorem to prove
theorem rocky_first_round_knockouts : first_round_knockouts = 19 :=
by
  -- skipping the actual proof with sorry
  sorry

end rocky_first_round_knockouts_l135_135611


namespace ben_remaining_money_l135_135029

/-- Ben's remaining money after business operations /-- 
theorem ben_remaining_money : 
  let initial_money := 2000
  let cheque := 600
  let debtor_payment := 800
  let maintenance_cost := 1200
  initial_money - cheque + debtor_payment - maintenance_cost = 1000 := 
by
  -- Initial money
  let initial_money := 2000
  -- Cheque amount
  let cheque := 600
  -- Debtor payment amount
  let debtor_payment := 800
  -- Maintenance cost
  let maintenance_cost := 1200
  -- Calculation
  have h₁ : initial_money - cheque = 2000 - 600 := by rfl
  let money_after_cheque := 2000 - 600
  have h₂ : money_after_cheque + debtor_payment = 1400 + 800 := by rfl
  let money_after_debtor := 1400 + 800
  have h₃ : money_after_debtor - maintenance_cost = 2200 - 1200 := by rfl
  let remaining_money := 2200 - 1200
  -- Assertion
  show remaining_money = 1000 from sorry

end ben_remaining_money_l135_135029


namespace sum_first_32_terms_bn_l135_135504

noncomputable def a_n (n : ℕ) : ℝ := 3 * n + 1

noncomputable def b_n (n : ℕ) : ℝ :=
  1 / ((a_n n) * Real.sqrt (a_n (n + 1)) + (a_n (n + 1)) * Real.sqrt (a_n n))

noncomputable def sum_bn (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) b_n

theorem sum_first_32_terms_bn : sum_bn 32 = 2 / 15 := 
sorry

end sum_first_32_terms_bn_l135_135504


namespace ratio_MCQ_to_ACQ_l135_135556

axiom trisect_angle (A B C P Q: Point) (h1: Line P) (h2: Line Q) (h3: Line A B C) 
  (hpq_trisect: trisects_angle h1 h2 h3) : trisect_angle A B C P Q

axiom bisect_angle (P Q C M: Point) (h4: Line CM) (h5: Line P Q) (h6: Angle P Q C) 
  (h7: bisects_angle h4 h6) : bisect_angle P Q C M

theorem ratio_MCQ_to_ACQ (A B C P Q M: Point) (h1: Line P) (h2: Line Q) (h3: Line A B C) 
  (h4: Line CM) (hpq_trisect: trisects_angle h1 h2 h3) (hcm_bisect: bisects_angle h4 (Angle P Q C)) :
  (measure (Angle M C Q)) / (measure (Angle A C Q)) = 1 / 4 :=
by
  sorry

end ratio_MCQ_to_ACQ_l135_135556


namespace count_even_odd_numbers_l135_135849

theorem count_even_odd_numbers (a b : ℕ) (h1 : 300 < a) (h2 : b < 520) :
  (∃ evens odds, 
    (∀ x, 300 < x ∧ x < 520 ∧ even x → evens x) ∧
    (∀ x, 300 < x ∧ x < 520 ∧ odd x → odds x) ∧ 
    evens.size + odds.size = 219) :=
sorry

end count_even_odd_numbers_l135_135849


namespace impossible_painting_chessboard_polygon_l135_135049

def infinite_grid (G : set (ℤ × ℤ)) :=
  ∀ x y : ℤ, (x, y) ∈ G

def chessboard_polygon (F : set (ℤ × ℤ)) : Prop :=
  ∃ (vertices : list (ℤ × ℤ)), 
    (∀ i, i < list.length vertices → (i+1 < list.length vertices → 
    (vertices.nth i).1 = (vertices.nth (i+1)).1 ∨ (vertices.nth i).2 = (vertices.nth (i+1)).2)) ∧ 
    (simple_polygon vertices)

-- Statement of the problem
theorem impossible_painting_chessboard_polygon :
  ∃ F, chessboard_polygon F ∧ 
  (∀ paint : set (ℤ × ℤ), 
    (∀ congruent_F : set (ℤ × ℤ), 
        congruent_F = { p : ℤ × ℤ | ∃ (tx ty : ℤ) (zx zy : ℤ), (zx, zy) ∈ F ∧ p = (zx + tx, zy + ty)} → 
        ∃ cell : ℤ × ℤ, cell ∈ paint ∧ cell ∈ congruent_F) ∧ 
    (∀ congruent_F : set (ℤ × ℤ), 
        congruent_F = { p : ℤ × ℤ | ∃ (tx ty : ℤ) (zx zy : ℤ), (zx, zy) ∈ F ∧ p = (zx + tx, zy + ty)} → 
        ncard {cell : ℤ × ℤ | cell ∈ paint ∧ cell ∈ congruent_F} ≤ 2020) → false) :=
sorry

end impossible_painting_chessboard_polygon_l135_135049


namespace general_term_formula_sum_of_first_10_terms_l135_135268

noncomputable def a_n (n : ℕ) : ℤ := 3 * n + 5

theorem general_term_formula (n : ℕ) :
  a_n 3 = 14 ∧ a_n 5 = 20 :=
by {
  unfold a_n,
  split;
  norm_num,
}

theorem sum_of_first_10_terms :
  (List.range 10).sum (λ n, a_n (n + 1)) = 215 :=
by {
  unfold a_n,
  norm_num,
}

end general_term_formula_sum_of_first_10_terms_l135_135268


namespace ben_remaining_money_l135_135030

variable (initial_capital : ℝ := 2000) 
variable (payment_to_supplier : ℝ := 600)
variable (payment_from_debtor : ℝ := 800)
variable (maintenance_cost : ℝ := 1200)
variable (remaining_capital : ℝ := 1000)

theorem ben_remaining_money
  (h1 : initial_capital = 2000)
  (h2 : payment_to_supplier = 600)
  (h3 : payment_from_debtor = 800)
  (h4 : maintenance_cost = 1200) :
  remaining_capital = (initial_capital - payment_to_supplier + payment_from_debtor - maintenance_cost) :=
sorry

end ben_remaining_money_l135_135030


namespace unsuitable_temperature_for_refrigerator_l135_135291

theorem unsuitable_temperature_for_refrigerator:
  let avg_temp := -18
  let variation := 2
  let min_temp := avg_temp - variation
  let max_temp := avg_temp + variation
  let temp_A := -17
  let temp_B := -18
  let temp_C := -19
  let temp_D := -22
  temp_D < min_temp ∨ temp_D > max_temp := by
  sorry

end unsuitable_temperature_for_refrigerator_l135_135291


namespace sqrt_7_estimate_l135_135064

theorem sqrt_7_estimate : (2 : Real) < Real.sqrt 7 ∧ Real.sqrt 7 < 3 → (Real.sqrt 7 - 1) / 2 < 1 := 
by
  intro h
  sorry

end sqrt_7_estimate_l135_135064


namespace find_garden_perimeter_l135_135631

noncomputable def garden_perimeter (a : ℝ) (P : ℝ) : Prop :=
  a = 2 * P + 14.25 ∧ a = 90.25

theorem find_garden_perimeter :
  ∃ P : ℝ, garden_perimeter 90.25 P ∧ P = 38 :=
by
  sorry

end find_garden_perimeter_l135_135631


namespace solve_for_x_l135_135062

theorem solve_for_x : ∃ x : ℚ, 5 * (x - 10) = 6 * (3 - 3 * x) + 10 ∧ x = 3.391 := 
by 
  sorry

end solve_for_x_l135_135062


namespace sixty_percent_of_fifty_greater_than_forty_two_percent_of_thirty_l135_135163

theorem sixty_percent_of_fifty_greater_than_forty_two_percent_of_thirty :
  let a := (60 / 100) * 50
  let b := (42 / 100) * 30
  a - b = 17.4 :=
by
  let a := (60 / 100) * 50
  let b := (42 / 100) * 30
  show a - b = 17.4
  sorry

end sixty_percent_of_fifty_greater_than_forty_two_percent_of_thirty_l135_135163


namespace product_of_roots_l135_135307

theorem product_of_roots :
  (Real.root 256 4) * (Real.root 8 3) * (Real.sqrt 16) = 32 :=
sorry

end product_of_roots_l135_135307


namespace triangle_sides_angles_l135_135969

theorem triangle_sides_angles (a b c A B C : ℝ) (h1: A = 2 * B) 
  (h2: sin C * sin (A - B) = sin B * sin (C - A)) 
  (h3: A + B + C = π) :
  (C = 5 * π / 8) ∧ (2 * a^2 = b^2 + c^2) :=
by
  -- Proof omitted
  sorry

end triangle_sides_angles_l135_135969


namespace square_of_1005_l135_135444

theorem square_of_1005 : (1005 : ℕ)^2 = 1010025 := 
  sorry

end square_of_1005_l135_135444


namespace gcd_six_digit_repeat_l135_135421

theorem gcd_six_digit_repeat (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) : 
  ∀ m : ℕ, m = 1001 * n → (gcd m 1001 = 1001) :=
by
  sorry

end gcd_six_digit_repeat_l135_135421


namespace land_profit_area_l135_135404

theorem land_profit_area 
  (total_land : ℕ)
  (sons : ℕ)
  (annual_profit_per_son : ℕ)
  (quarterly_profit : ℕ)
  (hectare_to_m2 : ℕ)
  (area_per_son : ℕ)
  (annual_profit_per_land_area : ℕ)
  (land_profit_area_hectare_fraction : ℕ)
  (profit_area_m2 : ℕ) :
  total_land = 3 ∧
  sons = 8 ∧
  (area_per_son = total_land / sons) ∧
  annual_profit_per_son = 10000 ∧
  quarterly_profit = 500 ∧
  hectare_to_m2 = 10000 ∧
  (annual_profit_per_land_area = quarterly_profit * 4) ∧
  (land_profit_area_hectare_fraction = annual_profit_per_son / annual_profit_per_land_area) ∧
  (profit_area_m2 = (area_per_son / land_profit_area_hectare_fraction) * hectare_to_m2) →
  profit_area_m2 = 750 :=
begin
  sorry
end

end land_profit_area_l135_135404


namespace F_equiv_A_l135_135103

-- Define the function F
def F : ℝ → ℝ := sorry

-- Given condition
axiom F_property (x : ℝ) : F ((1 - x) / (1 + x)) = x

-- The theorem that needs to be proved
theorem F_equiv_A (x : ℝ) : F (-2 - x) = -2 - F x := sorry

end F_equiv_A_l135_135103


namespace triangle_sides_angles_l135_135970

theorem triangle_sides_angles (a b c A B C : ℝ) (h1: A = 2 * B) 
  (h2: sin C * sin (A - B) = sin B * sin (C - A)) 
  (h3: A + B + C = π) :
  (C = 5 * π / 8) ∧ (2 * a^2 = b^2 + c^2) :=
by
  -- Proof omitted
  sorry

end triangle_sides_angles_l135_135970


namespace arithmetic_sequence_third_term_l135_135225

theorem arithmetic_sequence_third_term (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ) :
  (S 5 = 10) ∧ (S n = n * (a 1 + a n) / 2) ∧ (a 5 = a 1 + 4 * d) ∧ 
  (∀ n, a n = a 1 + (n-1) * d) → (a 3 = 2) :=
by
  intro h
  sorry

end arithmetic_sequence_third_term_l135_135225


namespace part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135935

-- Definitions for the conditions in the problem
variable {A B C a b c : ℝ}

-- Given conditions and problem setup
axiom triangle_ABC_sides : ∀ {a b c : ℝ}, sides a b c
axiom triangle_ABC_angles : ∀ {A B C : ℝ}, angles A B C
axiom sin_relation : ∀ {A B C : ℝ},
  sin C * sin (A - B) = sin B * sin (C - A)

-- Prove Part (1): If A = 2B, then C = 5π/8
theorem part1_A_eq_2B_implies_C :
  A = 2 * B → C = 5 * π / 8 :=
by
  intro h
  sorry

-- Prove Part (2): 2a² = b² + c²
theorem part2_2a_squared_eq_b_squared_plus_c_squared :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
by
  sorry

end part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135935


namespace symmetrical_complex_product_l135_135598

noncomputable def z1 : ℂ := 3 + complex.I
noncomputable def z2 : ℂ := -3 + complex.I

theorem symmetrical_complex_product :
  z1 * z2 = -10 :=
by
  have h1 : z1 = 3 + complex.I := rfl
  have h2 : z2 = -3 + complex.I := rfl
  sorry

end symmetrical_complex_product_l135_135598


namespace max_ab_min_fraction_l135_135791

-- Question 1: Maximum value of ab
theorem max_ab (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 3 * a + 7 * b = 10) : ab ≤ 25/21 := sorry

-- Question 2: Minimum value of (3/a + 7/b)
theorem min_fraction (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 3 * a + 7 * b = 10) : 3/a + 7/b ≥ 10 := sorry

end max_ab_min_fraction_l135_135791


namespace calculate_expression_l135_135437

theorem calculate_expression :
  4 + ((-2)^2) * 2 + (-36) / 4 = 3 := by
  sorry

end calculate_expression_l135_135437


namespace exists_M_for_double_seq_l135_135587

-- Definitions based on given conditions
def strictly_increasing_seq (a : ℕ → ℕ) := ∀ n, a n < a (n + 1)

def satisfies_divisibility_property (a : ℕ → ℕ) (N : ℕ) :=
  ∀ n, n > N → a (n + 1) ∣ (∑ i in Finset.range (n + 1), a i)

-- The main theorem to be proved
theorem exists_M_for_double_seq (a : ℕ → ℕ) (N : ℕ)
  (h_inc : strictly_increasing_seq a)
  (h_div : satisfies_divisibility_property a N) :
  ∃ M, ∀ m, m > M → a (m + 1) = 2 * a m :=
sorry

end exists_M_for_double_seq_l135_135587


namespace find_irrational_in_list_sqrt2_irrational_cube_root_8_rational_two_sevenths_rational_pi_approx_rational_l135_135273

theorem find_irrational_in_list :
  ∃ x ∈ {Real.sqrt 2, Real.sqrt 8, 2/7, 3.14}, Real.is_irrational x :=
by {
  sorry
}

theorem sqrt2_irrational : Real.is_irrational (Real.sqrt 2) :=
by {
  sorry
}

theorem cube_root_8_rational : ∀ (x : ℝ), x^3 = 8 → Rational x :=
by {
  sorry
}

theorem two_sevenths_rational : Rational (2/7) :=
by {
  sorry
}

theorem pi_approx_rational : Rational 3.14 :=
by {
  sorry
}

end find_irrational_in_list_sqrt2_irrational_cube_root_8_rational_two_sevenths_rational_pi_approx_rational_l135_135273


namespace infection_probability_l135_135553

theorem infection_probability
  (malaria_percent : ℝ)
  (zika_percent : ℝ)
  (vaccine_reduction : ℝ)
  (prob_random_infection : ℝ)
  (P : ℝ) :
  malaria_percent = 0.40 →
  zika_percent = 0.20 →
  vaccine_reduction = 0.50 →
  prob_random_infection = 0.15 →
  0.15 = (0.40 * 0.50 * P) + (0.20 * P) →
  P = 0.375 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end infection_probability_l135_135553


namespace number_of_blue_candles_l135_135743

def total_candles : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def blue_candles : ℕ := total_candles - (yellow_candles + red_candles)

theorem number_of_blue_candles : blue_candles = 38 :=
by
  unfold blue_candles
  unfold total_candles yellow_candles red_candles
  sorry

end number_of_blue_candles_l135_135743


namespace measure_11kg_ways_l135_135841

def generating_function_1_gram : Polynomial Int := polynomial.of_finsupp {0 := 1, 1 := 1, 2 := 1, 3 := 1}
def generating_function_2_gram : Polynomial Int := polynomial.of_finsupp {0 := 1, 2 := 1, 4 := 1, 6 := 1, 8 := 1}
def generating_function_4_gram : Polynomial Int := polynomial.of_finsupp {0 := 1, 4 := 1, 8 := 1}

theorem measure_11kg_ways :
  let G := generating_function_1_gram * generating_function_2_gram * generating_function_4_gram
  polynomial.coeff G 11 = 4 :=
by
  sorry

end measure_11kg_ways_l135_135841


namespace not_car_probability_l135_135821

-- Defining the probabilities of taking different modes of transportation.
def P_train : ℝ := 0.5
def P_car : ℝ := 0.2
def P_plane : ℝ := 0.3

-- Defining the event that these probabilities are for mutually exclusive events
axiom mutually_exclusive_events : P_train + P_car + P_plane = 1

-- Statement of the theorem to prove
theorem not_car_probability : P_train + P_plane = 0.8 := 
by 
  -- Use the definitions and axiom provided
  sorry

end not_car_probability_l135_135821


namespace no_such_alpha_beta_exists_l135_135757

-- Lean 4 statement that captures the essence of the problem and its conditions
theorem no_such_alpha_beta_exists (α β : ℝ) (hα : 0 < α) (hβ : 0 < β) :
  ¬(∀ n : ℕ, ∃ m : ℕ, n = if m % 2 = 0 then 2020 * (m / 2 + 1)
                         else if m % 4 == 1 then ⌊(m/2 + 1) * α⌋
                         else ⌊(m/2 + 1) * β⌋) := 
sorry

end no_such_alpha_beta_exists_l135_135757


namespace find_vector_magnitude_l135_135124

noncomputable def vector_a (x : ℝ) : ℝ × ℝ × ℝ := (x, 1, 0)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ × ℝ := (1, y, 0)
noncomputable def vector_c : ℝ × ℝ × ℝ := (2, -4, 0)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def is_perpendicular (u v : ℝ × ℝ × ℝ) : Prop :=
  dot_product u v = 0

def are_parallel (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2, k * v.3)

noncomputable def vector_add (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2, u.3 + v.3)

noncomputable def vector_magnitude (u : ℝ × ℝ × ℝ) :=
  real.sqrt (u.1^2 + u.2^2 + u.3^2)

theorem find_vector_magnitude (x y : ℝ) 
  (h1 : vector_a x = vector_a x)
  (h2 : vector_b y = vector_b y)
  (h3 : vector_c = vector_c)
  (h4 : is_perpendicular (vector_a x) vector_c) 
  (h5 : are_parallel (vector_b y) vector_c) :
  vector_magnitude (vector_add (vector_a x) (vector_b y)) = real.sqrt 10 :=
sorry

end find_vector_magnitude_l135_135124


namespace Liam_homework_assignments_l135_135411

theorem Liam_homework_assignments : 
  let assignments_needed (points : ℕ) : ℕ := match points with
    | 0     => 0
    | n+1 =>
        if n+1 <= 4 then 1
        else (4 + (((n+1) - 1)/4 - 1))

  30 <= 4 + 8 + 12 + 16 + 20 + 24 + 28 + 16 → ((λ points => List.sum (List.map assignments_needed (List.range points))) 30) = 128 :=
by
  sorry

end Liam_homework_assignments_l135_135411


namespace part1_part2_l135_135960

variable {A B C a b c : ℝ}

theorem part1 (h₁ : A = 2 * B) (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : C = 5 / 8 * π :=
  sorry

theorem part2 (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
  sorry

end part1_part2_l135_135960


namespace largest_monochromatic_subgraph_l135_135889

theorem largest_monochromatic_subgraph :
  ∀ (A B : finset ℕ) (e : A × B → ℕ) (R : ℕ) (X Y : ℕ), 
  A.card = 2011 → B.card = 2011 → (∀ a ∈ A, ∀ b ∈ B, e (a, b) ≤ 19) →
  (∃ (k : ℕ), k < 213 ∧ k > 0 ∧ 
  ∀ f : A × B → finset (A ∪ B), 
  (∀ i ∈ A ∪ B, ∃ ai bi : A × B, ai.1 ∈ A ∧ ai.2 ∈ B ∧ bi.1 ∈ A ∧ bi.2 ∈ B ∧ e ai = e bi ∧ (f (ai.1, bi.2)).card ≤ k) 
) :=
begin
  sorry
end

end largest_monochromatic_subgraph_l135_135889


namespace water_formed_at_equilibrium_l135_135452

theorem water_formed_at_equilibrium :
  ∀ (NH4NO3 NaOH X NaNO3 H2O Kc : ℝ),
  NH4NO3 = 2 → NaOH = 2 → X = 1 → NaNO3 = 0 → H2O = 0 →
  Kc = 5 →
  (∀ x : ℝ, (Kc = (x^2 * (2 * x)^4) / ((2 - x)^4) → x ≈ 2 )) →
  2 * 2 = 4 :=
by
  intros NH4NO3 NaOH X NaNO3 H2O Kc hNH4NO3 hNaOH hX hNaNO3 hH2O hKc hsol
  sorry

end water_formed_at_equilibrium_l135_135452


namespace aquatic_reserve_l135_135561

theorem aquatic_reserve (fishes_in_each_body: ℕ) (total_fishes: ℕ)
  (h1: fishes_in_each_body = 175) (h2: total_fishes = 1050) :
  total_fishes / fishes_in_each_body = 6 :=
by
  rw [h1, h2]
  norm_num
  sorry

end aquatic_reserve_l135_135561


namespace count_divisible_by_2016_l135_135875

theorem count_divisible_by_2016 (n : ℕ) (hn : n > 1)
  (numbering_count : Π (X : ℕ), ℕ)
  (unique_flight_route : ∀ (X Y : ℕ), X ≠ Y → ∃! p : List ℕ, p.head = X ∧ p.last = Y ∧ ∀ (i j : ℕ), i < j → p[i] < p[j])
  (hdiv : ∀ (X : ℕ), X ≠ 0 → 2016 ∣ numbering_count X) :
  2016 ∣ numbering_count 0 :=
sorry

end count_divisible_by_2016_l135_135875


namespace complex_conjugate_in_fourth_quadrant_l135_135500

theorem complex_conjugate_in_fourth_quadrant (z : ℂ) (h : z = (Complex.I / (Complex.I + 2))) :
  Complex.conj(z).im < 0 ∧ Complex.conj(z).re > 0 :=
by
  sorry

end complex_conjugate_in_fourth_quadrant_l135_135500


namespace triangle_sides_l135_135755

axiom p : ℝ
axiom q : ℝ

theorem triangle_sides (h_right_triangle : (p + 2 * q)^2 = p^2 + (p + q)^2)
                        (h_side_limit : p + 2 * q ≤ 12) :
    p = (1 + Real.sqrt 7) / 2 ∧ q = 1 :=
begin
    sorry
end

end triangle_sides_l135_135755


namespace josh_marbles_l135_135907

theorem josh_marbles (initial_marbles lost_marbles : ℕ) (h_initial : initial_marbles = 9) (h_lost : lost_marbles = 5) :
  initial_marbles - lost_marbles = 4 :=
by
  sorry

end josh_marbles_l135_135907


namespace max_value_f_l135_135119

noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 2) / (2*x - 2)

theorem max_value_f (x : ℝ) (h : -4 < x ∧ x < 1) : ∃ y, f y = -1 ∧ (∀ z, f z ≤ f y) :=
by 
  sorry

end max_value_f_l135_135119


namespace correctCalculation_l135_135680

variable (x : Type) [MonoidWithZero x]

-- Define all the conditions
def condA : Prop := (x^4 + x^2 = x^6)
def condB : Prop := (x^3 - x^2 = x)
def condC : Prop := ((x^3)^2 = x^6)
def condD : Prop := (x^6 / x^3 = x^2)

-- The main statement that needs to be proven. 
theorem correctCalculation : condC :=
by {
  -- Proof goes here; currently, we'll skip it with 'sorry'.
  sorry
}

end correctCalculation_l135_135680


namespace sum_of_roots_l135_135858

theorem sum_of_roots (x1 x2 : ℝ) (h : x1^2 + 5*x1 - 1 = 0 ∧ x2^2 + 5*x2 - 1 = 0) : x1 + x2 = -5 :=
sorry

end sum_of_roots_l135_135858


namespace longest_side_of_triangle_l135_135133

theorem longest_side_of_triangle (a b c : ℕ) (h1 : a = 3) (h2 : b = 5) 
    (cond : a^2 + b^2 - 6 * a - 10 * b + 34 = 0) 
    (triangle_ineq1 : a + b > c)
    (triangle_ineq2 : a + c > b)
    (triangle_ineq3 : b + c > a)
    (hScalene: a ≠ b ∧ b ≠ c ∧ a ≠ c) : c = 6 ∨ c = 7 := 
by {
  sorry
}

end longest_side_of_triangle_l135_135133


namespace geom_seq_a7_a10_sum_l135_135876

theorem geom_seq_a7_a10_sum (a_n : ℕ → ℝ) (q a1 : ℝ)
  (h_seq : ∀ n, a_n (n + 1) = a1 * (q ^ n))
  (h1 : a1 + a1 * q = 2)
  (h2 : a1 * (q ^ 2) + a1 * (q ^ 3) = 4) :
  a_n 7 + a_n 8 + a_n 9 + a_n 10 = 48 := 
sorry

end geom_seq_a7_a10_sum_l135_135876


namespace white_square_not_always_covered_l135_135668

theorem white_square_not_always_covered :
  ∀ (squares : Set (ℕ × ℕ)) (white_square : (ℕ × ℕ)),
  (∃ black_squares : Set (Set (ℕ × ℕ)),
    (black_squares.card = 19 ∧ white_square ∈ squares) ∧
    (∀ black_square ∈ black_squares, 
      ∀ (x, y) ∈ black_square, (x, y) ∈ squares)) →
  (∀ removal_square ∈ black_squares,
    ∃ (x, y) ∈ white_square, ¬ (∃ remaining_squares ∈ (black_squares \ {removal_square}), (x, y) ∈ ⋃ sq ∈ remaining_squares)) :=
begin
  sorry
end

end white_square_not_always_covered_l135_135668


namespace part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135934

-- Definitions for the conditions in the problem
variable {A B C a b c : ℝ}

-- Given conditions and problem setup
axiom triangle_ABC_sides : ∀ {a b c : ℝ}, sides a b c
axiom triangle_ABC_angles : ∀ {A B C : ℝ}, angles A B C
axiom sin_relation : ∀ {A B C : ℝ},
  sin C * sin (A - B) = sin B * sin (C - A)

-- Prove Part (1): If A = 2B, then C = 5π/8
theorem part1_A_eq_2B_implies_C :
  A = 2 * B → C = 5 * π / 8 :=
by
  intro h
  sorry

-- Prove Part (2): 2a² = b² + c²
theorem part2_2a_squared_eq_b_squared_plus_c_squared :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
by
  sorry

end part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135934


namespace max_distance_from_point_to_line_l135_135527

-- Definitions of circles and the condition of being externally tangent
def circle1_eq (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 4 * x + m = 0
def circle2_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 2 * Real.sqrt 2)^2 = 4
def are_externally_tangent (m : ℝ) : Prop :=
  let center1 := (2, 0)
  let center2 := (3, -2 * Real.sqrt 2)
  let radius1 := Real.sqrt (4 - m)
  let radius2 := 2
  Real.dist center1 center2 = radius1 + radius2

-- The distance function from a point to a line
noncomputable def dist_point_to_line (x y : ℝ) : ℝ := abs (3 * x - 4 * y + 4) / Real.sqrt 25

-- The maximum distance from any point on Circle1 to the line
theorem max_distance_from_point_to_line (m : ℝ) :
  are_externally_tangent m →
  ∃ P : ℝ × ℝ, circle1_eq P.1 P.2 m ∧ 
    max (dist_point_to_line P.1 P.2) 3 = 3 :=
by
  sorry

end max_distance_from_point_to_line_l135_135527


namespace max_value_of_f_on_interval_l135_135861

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

noncomputable def g (x : ℝ) : ℝ := x + 1 / (x^2)

theorem max_value_of_f_on_interval (p q : ℝ) 
  (h_min_val : ∀ x : ℝ, g (∛2) = f (∛2) p q)
  (h_pq : p * ∛2 + q = ∛2) :
  ∃ x ∈ set.Icc (1 : ℝ) 2, f x p q = 4 - (5 / 2) * ∛2 + (∛4) := 
begin
  sorry
end

end max_value_of_f_on_interval_l135_135861


namespace imaginary_part_of_z_l135_135486

def z := (1 - (complex.i : ℂ)) / complex.i

theorem imaginary_part_of_z : complex.im z = -1 :=
by
  -- Placeholder for the proof
  sorry

end imaginary_part_of_z_l135_135486


namespace calc_nabla_example_l135_135167

-- Define the custom operation ∇
def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

-- State the proof problem
theorem calc_nabla_example : op_nabla (op_nabla 2 3) (op_nabla 4 5) = 49 / 56 := by
  sorry

end calc_nabla_example_l135_135167


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135353

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135353


namespace no_adjacent_abc_seating_l135_135186

theorem no_adjacent_abc_seating : 
  let total_arrangements := Nat.factorial 8
  let abc_unit_arrangements := Nat.factorial 3
  let reduced_arrangements := Nat.factorial 6
  total_arrangements - reduced_arrangements * abc_unit_arrangements = 36000 :=
by 
  sorry

end no_adjacent_abc_seating_l135_135186


namespace count_valid_even_numbers_l135_135162

def is_even (n : ℕ) : Prop := n % 2 = 0

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  d < 10 ∧ ∃ (k : ℕ), (d * 10^k < n) ∧ (n < (d + 1) * 10^k)

def valid_even_numbers_up_to (limit : ℕ) : ℕ :=
  (finset.range (limit + 1)).filter (λ n, is_even n ∧ ¬ contains_digit n 5).card

theorem count_valid_even_numbers : valid_even_numbers_up_to 500 = 404 :=
by sorry

end count_valid_even_numbers_l135_135162


namespace train_speed_l135_135412

def length_train := 250
def length_bridge := 180
def time_crossing := 20
def total_distance := length_train + length_bridge
def speed := total_distance / time_crossing

theorem train_speed :
  speed = 21.5 := 
sorry

end train_speed_l135_135412


namespace find_lambda_l135_135097

def vec_a : ℝ × ℝ × ℝ := (-3, 2, 5)
def vec_b (λ : ℝ) : ℝ × ℝ × ℝ := (1, λ, -1)
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_lambda (λ : ℝ) (h : dot_product vec_a (vec_b λ) = 0) : λ = 4 :=
by sorry

end find_lambda_l135_135097


namespace volume_of_pyramid_l135_135659

noncomputable def slant_height := Real.sqrt 8
noncomputable def alpha := Real.pi / 6
noncomputable def beta := Real.pi / 4

theorem volume_of_pyramid 
  (A O₁ O₂ O₃ : ℝ) 
  (h₁ : ∀ {A O₁}, slant_height * Real.cos alpha = 2 * Real.sqrt 3)
  (h₂ : ∀ {A O₃}, slant_height * Real.cos beta = 2) :
  volume_of_pyramid O₁ O₂ O₃ A = Real.sqrt (Real.sqrt 3 + 1) :=
sorry

end volume_of_pyramid_l135_135659


namespace monotonicity_intervals_and_min_value_F_min_value_max_k_l135_135511

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonicity_intervals_and_min_value :
  (∀ x > 0, x = e⁻¹ ↔ f(x) = -1/e) ∧
  (∀ x, x > e⁻¹ → ∀ y, y > x → f(y) > f(x)) ∧
  (∀ x, 0 < x ∧ x < e⁻¹ → ∀ y, 0 < y ∧ y < x → f(y) < f(x)) :=
sorry

noncomputable def F (x : ℝ) (a : ℝ) : ℝ := (f(x) - a) / x

theorem F_min_value (a : ℝ) : (∀ x ∈ Set.Icc 1 Real.exp1, F(x, a) ≥ 3/2) ↔ a = -Real.sqrt Real.exp1 :=
sorry

theorem max_k : 
  (∀ x > 1, ∀ k : ℤ, f(x) + x - k * (x - 1) > 0) →
  (∃ k_max : ℤ, ∀ k : ℤ, k ≤ k_max) ∧ k_max = 3 :=
sorry

end monotonicity_intervals_and_min_value_F_min_value_max_k_l135_135511


namespace sum_of_two_even_numbers_is_even_l135_135372

  theorem sum_of_two_even_numbers_is_even (a b : ℤ) (ha : ∃ k : ℤ, a = 2 * k) (hb : ∃ m : ℤ, b = 2 * m) : ∃ n : ℤ, a + b = 2 * n := by
    sorry
  
end sum_of_two_even_numbers_is_even_l135_135372


namespace cylindrical_plane_l135_135778

open Set

-- Define a cylindrical coordinate point (r, θ, z)
structure CylindricalCoord where
  r : ℝ
  theta : ℝ
  z : ℝ

-- Condition 1: In cylindrical coordinates, z is the height
def height_in_cylindrical := λ coords : CylindricalCoord => coords.z 

-- Condition 2: z is constant c
variable (c : ℝ)

-- The theorem to be proven
theorem cylindrical_plane (c : ℝ) :
  {p : CylindricalCoord | p.z = c} = {q : CylindricalCoord | q.z = c} :=
by
  sorry

end cylindrical_plane_l135_135778


namespace cartesian_equation_of_line_parametric_equation_of_circle_minimum_distance_l135_135822
noncomputable theory
open_locale real

/-- Conditions: 
* Equation of circle M: (x-4)^2 + y^2 = 1
* Polar equation of line l: ρ sin (θ + π / 6) = 1/2
--/

def equation_of_circle (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1

def polar_equation_of_line (ρ θ : ℝ) : Prop := ρ * sin (θ + π / 6) = 1 / 2

-- Cartesian equation of line l
theorem cartesian_equation_of_line (x y : ℝ) :
  (∃ ρ θ : ℝ, x = ρ * cos θ ∧ y = ρ * sin θ ∧ polar_equation_of_line ρ θ) →
  x + sqrt 3 * y = 1 := 
sorry

-- Parametric equation of circle M
theorem parametric_equation_of_circle (φ x y : ℝ) :
  equation_of_circle x y ↔ 
  (x = 4 + cos φ ∧ y = sin φ) := 
sorry

-- Minimum distance from a point on circle M to line l
theorem minimum_distance (φ : ℝ) :
  let x := 4 + cos φ,
      y := sin φ in
  (equation_of_circle x y) →
  (∃ d : ℝ, d = abs (x + sqrt 3 * y - 1) / 2 ∧ ∀ M, M ∈ set_of (equation_of_circle) → d ≤ 1 / 2) := 
sorry

end cartesian_equation_of_line_parametric_equation_of_circle_minimum_distance_l135_135822


namespace sandy_correct_sums_l135_135688

-- Definitions based on the conditions
variables (c i : ℕ)

-- Conditions as Lean statements
axiom h1 : 3 * c - 2 * i = 65
axiom h2 : c + i = 30

-- Proof goal
theorem sandy_correct_sums : c = 25 := 
by
  sorry

end sandy_correct_sums_l135_135688


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135341

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n % 9 = 0 ∧ (n.digits.sum = 27) ∧
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ m % 9 = 0 ∧ (m.digits.sum = 27) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135341


namespace find_valid_n_l135_135460

def validArrangement (n : ℕ) : Prop :=
  ∃ (nums : List ℕ), 
    nums.length = 2 * n ∧
    (∀ k, 1 ≤ k ∧ k ≤ n →
      ∃ i j, i < j ∧ nums.nth i = some k ∧ nums.nth j = some k ∧ (j - i - 1) = k)

theorem find_valid_n (n : ℕ) : 
  (validArrangement n) ↔ (∃ l, n = 4 * l ∨ n = 4 * l - 1) :=
by
  sorry

end find_valid_n_l135_135460


namespace _l135_135595

noncomputable def triangle := {A B C : Type}
noncomputable def length {X Y : Type} := ℝ
noncomputable def P := sorry
noncomputable theorem length_SP:
  ∃ (A B C T S P : Type), 
      (length AB = 8) ∧ 
      (length BC = 9) ∧ 
      (length CA = 10) ∧ 
      (tangent_to_circumcircle A T) ∧ 
      (intersects_line_at BC T) ∧ 
      (circle_centered_at T A S) ∧ 
      (intersects_AC_second_time A S) ∧ 
      (angle_bisector SBA S A P) → 
      length SP = 225 / 13 :=
sorry

end _l135_135595


namespace binomial_parameters_unique_l135_135096

   -- Definitions of the conditions
   def is_binomial (X : ℕ → ℝ) (n : ℕ) (p : ℝ) : Prop := 
     ∃ (B : ℕ → ℕ → ℝ), 
       (∀ k, B n k = (n.choose k : ℝ) * p^k * (1-p)^(n-k)) ∧ 
       (X = λ k, B n k)
   
   def expected_value (X : ℕ → ℝ) (μ : ℝ) : Prop := 
     ∃ (EX : ℝ), 
       (EX = ∑ k, k * X k) ∧ 
       (EX = μ)

   def variance (X : ℕ → ℝ) (σ2 : ℝ) : Prop := 
     ∃ (DX : ℝ), 
       (DX = ∑ k, k^2 * X k - (∑ k, k * X k)^2) ∧ 
       (DX = σ2)

   -- The proof problem statement
   theorem binomial_parameters_unique 
       (X : ℕ → ℝ) 
       (n p : ℝ)
       (h1 : is_binomial X n p) 
       (h2 : expected_value X 8) 
       (h3 : variance X 1.6)
       : n = 10 ∧ p = 0.8 :=
   sorry
   
end binomial_parameters_unique_l135_135096


namespace plane_perpendicular_parallel_l135_135389

noncomputable def three_planes (α β γ : Type) : Prop :=
∃ (points : α × β × γ), true

theorem plane_perpendicular_parallel {α β γ : Type}
  (H : three_planes α β γ)
  (h1 : α ⊥ β)
  (h2 : β ∥ γ) :
  α ⊥ γ :=
sorry

end plane_perpendicular_parallel_l135_135389


namespace suitable_altitude_for_planting_l135_135701

theorem suitable_altitude_for_planting (x : ℕ) :
  (16 ≤ 22 - (x / 100) * 0.5) ∧ (22 - (x / 100) * 0.5 ≤ 20) ↔ (400 ≤ x ∧ x ≤ 1200) := 
sorry

end suitable_altitude_for_planting_l135_135701


namespace rocky_first_round_knockouts_l135_135612

-- Definitions
def total_fights : ℕ := 190
def knockout_percentage : ℝ := 0.50
def first_round_knockout_percentage : ℝ := 0.20

-- Calculation of knockouts
def total_knockouts : ℕ := (total_fights * knockout_percentage).to_nat
def first_round_knockouts : ℕ := (total_knockouts * first_round_knockout_percentage).to_nat

-- Theorem to prove
theorem rocky_first_round_knockouts : first_round_knockouts = 19 :=
by
  -- skipping the actual proof with sorry
  sorry

end rocky_first_round_knockouts_l135_135612


namespace maximal_cross_section_area_prism_l135_135711

theorem maximal_cross_section_area_prism :
  let A := (6, 6, 0)
  let B := (-6, 6, 0)
  let C := (-6, -6, 0)
  let D := (6, -6, 0)
  (plane_eq := fun x y z => 5 * x - 8 * y + 3 * z)
  (hA := 6)
  (hB := 36)
  (hC := 4)
  (hD := -16)
  let E := (6, 6, hA)
  let F := (-6, 6, hB)
  let G := (-6, -6, hC)
  let H := (6, -6, hD)
  let EF := (-12, 0, 30)
  let EG := (-12, -12, -2)
  let EF_cross_EG := (360, 324, 144)
  let area := ( real.sqrt (360^2 + 324^2 + 144^2) ) / 2
  area = 252 := 
sorry

end maximal_cross_section_area_prism_l135_135711


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135350

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135350


namespace locus_of_midpoints_of_chords_l135_135917

/-- Definitions (conditions) -/
variables {A B O : Point} (h_circle : Circle O) (h_A_on_circle : A ∈ h_circle.points) (h_B_on_circle : B ∈ h_circle.points) 

/-- Statement of the locus of midpoints of chords with endpoints on different arcs. -/
theorem locus_of_midpoints_of_chords :
  let arc1 := h_circle.arc A B,
      arc2 := h_circle.arc B A,
      semi1 := h_circle.semicircle_on_diameter A O,
      semi2 := h_circle.semicircle_on_diameter B O in
  ∀ M : Point, (M ∈ (semi1.interiors ∪ semi2.interiors) \ (semi1.interiors ∩ semi2.interiors)) ↔
    (∃ (C D : Point), C ∈ arc1 ∧ D ∈ arc2 ∧ M = midpoint C D) :=
begin
  sorry
end

end locus_of_midpoints_of_chords_l135_135917


namespace complete_job_days_l135_135392

-- Variables and Conditions
variables (days_5_8 : ℕ) (days_1 : ℕ)

-- Assume that completing 5/8 of the job takes 10 days
def five_eighths_job_days := 10

-- Find days to complete one job at the same pace. 
-- This is the final statement we need to prove
theorem complete_job_days
  (h : 5 * days_1 = 8 * days_5_8) :
  days_1 = 16 := by
  -- Proof is omitted.
  sorry

end complete_job_days_l135_135392


namespace find_A_l135_135515

noncomputable theory

def M : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![2, -3],
  ![1, -1]
]

def M_inv : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![-1, 3],
  ![-1, 2]
]

def A' : Fin 2 → ℚ := ![13, 5]

theorem find_A (x y : ℚ) (A : Fin 2 → ℚ) : 
  (M * A = A') → 
  (M⁻¹ = M_inv) → 
  (A = ![2, -3]) :=
begin
  sorry
end

end find_A_l135_135515


namespace watermelon_seeds_l135_135667

theorem watermelon_seeds (n_slices : ℕ) (total_seeds : ℕ) (B W : ℕ) 
  (h1: n_slices = 40) 
  (h2: B = W) 
  (h3 : n_slices * B + n_slices * W = total_seeds)
  (h4 : total_seeds = 1600) : B = 20 :=
by {
  sorry
}

end watermelon_seeds_l135_135667


namespace proof_of_problem_l135_135497

noncomputable def f : ℝ → ℝ := sorry  -- define f as a function in ℝ to ℝ

theorem proof_of_problem 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_f1 : f 1 = 1)
  (h_periodic : ∀ x : ℝ, f (x + 6) = f x + f 3) :
  f 2015 + f 2016 = -1 := 
sorry

end proof_of_problem_l135_135497


namespace intersection_of_M_and_N_l135_135523

def set_M : Set ℝ := {x : ℝ | x^2 - x ≥ 0}
def set_N : Set ℝ := {x : ℝ | x < 2}

theorem intersection_of_M_and_N :
  set_M ∩ set_N = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} :=
by
  sorry

end intersection_of_M_and_N_l135_135523


namespace cubes_of_roots_l135_135747

theorem cubes_of_roots (a b c : ℝ) (h1 : a + b + c = 2) (h2 : ab + ac + bc = 2) (h3 : abc = 3) : 
  a^3 + b^3 + c^3 = 9 :=
by
  sorry

end cubes_of_roots_l135_135747


namespace square_area_l135_135718

theorem square_area (a : ℝ) (h : ℝ) (w : ℝ) (areas_equal : ∀ i j, i ≠ j → (a^2 / 5) = (a * h / 5)) (r_width : w = 5) :
  a = 20 → (a * a) = 400 :=
by
  sorry

end square_area_l135_135718


namespace part1_C_value_part2_triangle_equality_l135_135982

noncomputable theory

variables (a b c : ℝ) (A B C : ℝ)
variables (h1 : A + B + C = Real.pi) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) (h3 : A = 2 * B)

-- Part 1: Proving that C = 5π/8 given the conditions
theorem part1_C_value :
  C = 5 * Real.pi / 8 :=
begin
  sorry
end

-- Part 2: Proving that 2a^2 = b^2 + c^2 given the conditions
theorem part2_triangle_equality :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
begin
  sorry
end

end part1_C_value_part2_triangle_equality_l135_135982


namespace opera_house_earnings_l135_135733

theorem opera_house_earnings :
  let rows := 150
  let seats_per_row := 10
  let ticket_cost := 10
  let total_seats := rows * seats_per_row
  let seats_not_taken := total_seats * 20 / 100
  let seats_taken := total_seats - seats_not_taken
  let total_earnings := ticket_cost * seats_taken
  total_earnings = 12000 := by
sorry

end opera_house_earnings_l135_135733


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135313

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 + d2 + d3

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, is_three_digit n ∧ is_multiple_of_9 n ∧ digit_sum n = 27 ∧
  ∀ m : ℕ, is_three_digit m ∧ is_multiple_of_9 m ∧ digit_sum m = 27 → m ≤ n := 
by 
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135313


namespace sequence_properties_l135_135988

noncomputable def arithmetic_mean (a b : ℝ) := (a + b) / 2
noncomputable def quadratic_mean (a b : ℝ) := real.sqrt ((a^2 + b^2) / 2)
noncomputable def harmonic_mean (a b : ℝ) := 2 / ((1 / a) + (1 / b))

def sequences (x y : ℝ) (hxy : x ≠ y) (h : x > 0 ∧ y > 0) :=
  let A := λ n, if n = 1 then arithmetic_mean x y else 0 -- placeholder
  let Q := λ n, if n = 1 then quadratic_mean x y else 0 -- placeholder
  let H := λ n, if n = 1 then harmonic_mean x y else 0 -- placeholder
  ∀ n, n ≥ 1 → 
    (A (n + 1) = arithmetic_mean (A n) (H n)) ∧ 
    (Q (n + 1) = quadratic_mean (A n) (H n)) ∧ 
    (H (n + 1) = harmonic_mean (A n) (H n))

theorem sequence_properties (x y : ℝ) (hxy : x ≠ y) (h : x > 0 ∧ y > 0):
  let A := λ n, if n = 1 then arithmetic_mean x y else 0 -- placeholder
  let Q := λ n, if n = 1 then quadratic_mean x y else 0 -- placeholder
  let H := λ n, if n = 1 then harmonic_mean x y else 0 -- placeholder
  (∀ n, n ≥ 1 → A n > 0) ∧
  (∀ n, n ≥ 1 → Q n > 0) ∧
  (∀ n, n ≥ 1 → H n > 0) ∧
  (∀ k : ℕ, A k > A (k+1)) ∧
  (∀ k : ℕ, Q k > Q (k+1)) ∧
  (∀ k : ℕ, H k < H (k+1)) := sorry

end sequence_properties_l135_135988


namespace find_a_and_b_maximize_profit_l135_135020

variable (a b x : ℝ)

-- The given conditions
def condition1 : Prop := 2 * a + b = 120
def condition2 : Prop := 4 * a + 3 * b = 270
def constraint : Prop := 75 ≤ 300 - x

-- The questions translated into a proof problem
theorem find_a_and_b :
  condition1 a b ∧ condition2 a b → a = 45 ∧ b = 30 :=
by
  intros h
  sorry

theorem maximize_profit (a : ℝ) (b : ℝ) (x : ℝ) :
  condition1 a b → condition2 a b → constraint x →
  x = 75 → (300 - x) = 225 → 
  (10 * x + 20 * (300 - x) = 5250) :=
by
  intros h1 h2 hc hx hx1
  sorry

end find_a_and_b_maximize_profit_l135_135020


namespace product_of_roots_l135_135308

theorem product_of_roots :
  (Real.root 256 4) * (Real.root 8 3) * (Real.sqrt 16) = 32 :=
sorry

end product_of_roots_l135_135308


namespace solve_inequality_l135_135816

noncomputable def f : ℝ → ℝ := sorry

axiom condition1 : ∀ x y : ℝ, (0 < x ∧ 0 < y) → f(x + y) = f(x) * f(y)
axiom condition2 : ∀ x : ℝ, 1 < x → f(x) > 2
axiom condition3 : f(2) = 4

theorem solve_inequality (x : ℝ) (hx : 0 < x) :
  (f(x^2) > 2 * f(x + 1)) ↔ (x > 2) := 
sorry

end solve_inequality_l135_135816


namespace isosceles_base_angle_l135_135887

/-- In an isosceles triangle, given that one angle is 80 degrees, 
    the measure of the base angle could be either 50 degrees or 80 degrees. -/
theorem isosceles_base_angle (T : Triangle) (isosceles_T : T.is_isosceles) (angle_80 : ∃ A B C : T.vertices, ∃ (α : ℝ), α = 80 ∧ T.angle A B C = α) :
  ∃ β : ℝ, (β = 50 ∨ β = 80) ∧ (∃ A B C : T.vertices, T.angle A B = β ∧ T.angle A C = β) :=
begin
  sorry
end

end isosceles_base_angle_l135_135887


namespace volume_of_regular_triangular_pyramid_l135_135468

theorem volume_of_regular_triangular_pyramid (r h : ℝ) :
  ∀ V : ℝ, (∀ (pyramid : Type), (pyramid is_regular_triangual_pyramid)
  ∧ (pyramid.height = h) ∧ (pyramid.inscribed_sphere_radius = r)) → 
  V = (2 + h * real.sqrt(3))^2 * h^2 * real.sqrt(3) / (12 * (h^2 - 2 * r * h)) := sorry

end volume_of_regular_triangular_pyramid_l135_135468


namespace number_of_integers_fulfilling_condition_l135_135079

theorem number_of_integers_fulfilling_condition :
  ∃ (n_total : ℕ), n_total = 14520 ∧ ∀ n : ℤ,
    1 + (floor ((120 * n) / 121)) = ceil ((119 * n) / 120) → 
    n_total = 14520 :=
begin
  sorry
end

end number_of_integers_fulfilling_condition_l135_135079


namespace minimum_lambda_l135_135132

noncomputable def a_n (n : ℕ) : ℕ := n * (n + 1)
noncomputable def b_n (n : ℕ) : ℝ := 1 / (2 * (n + 1) * n)
noncomputable def S_n (n : ℕ) : ℝ := ∑ i in Finset.range n, a_n i
noncomputable def T_n (n : ℕ) : ℝ := ∑ i in Finset.range n, b_n i

theorem minimum_lambda (λ : ℝ) :
  (∀ n : ℕ, n > 0 → λ > T_n n) ↔ λ = 1 / 2 := by
    sorry

end minimum_lambda_l135_135132


namespace sin_double_angle_l135_135494

variable (α : Real)
variable (h : sin α + cos α = 1 / 2)

theorem sin_double_angle (h : sin α + cos α = 1 / 2) : sin (2 * α) = -3 / 4 := 
by
  sorry

end sin_double_angle_l135_135494


namespace janet_can_buy_max_9_notebooks_l135_135209

noncomputable def max_notebooks (p c n : ℕ) : Prop :=
  p ≥ 1 ∧ c ≥ 1 ∧ n ≥ 1 ∧ (3 * p + 4 * c + 10 * n = 100) → n = 9

theorem janet_can_buy_max_9_notebooks : ∃ p c n, max_notebooks p c n :=
begin
  use 1,
  use 1,
  use 9,
  unfold max_notebooks,
  split,
  { norm_num, },
  split,
  { norm_num, },
  split,
  { norm_num, },
  linarith,
end

end janet_can_buy_max_9_notebooks_l135_135209


namespace equation_has_two_solutions_l135_135851

theorem equation_has_two_solutions : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ∀ x : ℝ, ¬ ( |x - 1| = |x - 2| + |x - 3| ) ↔ (x ≠ x₁ ∧ x ≠ x₂) :=
sorry

end equation_has_two_solutions_l135_135851


namespace no_positive_integer_n_such_that_14n_plus_19_is_prime_l135_135252

theorem no_positive_integer_n_such_that_14n_plus_19_is_prime :
  ∀ n : Nat, 0 < n → ¬ Nat.Prime (14^n + 19) :=
by
  intro n hn
  sorry

end no_positive_integer_n_such_that_14n_plus_19_is_prime_l135_135252


namespace smallest_positive_period_and_interval_max_and_min_on_interval_l135_135834

noncomputable def f (x : ℝ) : ℝ :=
  cos (2 * x - π / 6) * sin (2 * x) - 1 / 4

theorem smallest_positive_period_and_interval :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π / 2) ∧
  (∀ k : ℤ, ∃ a b, [a, b] = [π / 6 + k * π / 2, 5 * π / 12 + k * π / 2]) :=
sorry

theorem max_and_min_on_interval :
  ∃ a b, ∀ x ∈ [-π / 4, 0], f x ≤ a ∧ f x ≥ b ∧ a = 1 / 4 ∧ b = -1 / 2 :=
sorry

end smallest_positive_period_and_interval_max_and_min_on_interval_l135_135834


namespace match_result_third_vs_seventh_l135_135393

-- Define the conditions of the tournament
variables {Player : Type*} [Fintype Player]
def score : Player → ℝ := sorry
def match_result (p1 p2 : Player) : ℝ := if p1 = p2 then 0 else (if true then 1 else 0) -- win representation (placeholder)

-- Define the players and their orderings
variable players : Finset Player
variables (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : Player)
variable h : is_strictly_decreasing (score '' {a₁, a₂, a₃, a₄, a₅, a₆, a₇, a₈})

-- Define special condition about the scores known
variable h_cond : score a₂ = score a₅ + score a₆ + score a₇ + score a₈

-- Lean statement
theorem match_result_third_vs_seventh : match_result a₃ a₇ = 1 :=
sorry

end match_result_third_vs_seventh_l135_135393


namespace angle_CKL_eq_angle_ABC_l135_135231

-- Definitions of the problem setup
variables {A B C L K : Type} [MetricSpace A]

def is_angle_bisector (A B C L : Type) [MetricSpace A] :=
  (dist A L = dist A C) ∧ -- AL = AC
  (interior_angle A B C / 2 = interior_angle A L C)

def chosen_point (K L B C : Type) [MetricSpace A] :=
  (dist C K = dist B L)

-- Statement of the theorem
theorem angle_CKL_eq_angle_ABC 
    (ABC : triangle A B C)
    (H1 : is_angle_bisector A B C L)
    (H2 : chosen_point K L B C) :
  angle K C L = angle A B C :=
sorry

end angle_CKL_eq_angle_ABC_l135_135231


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135310

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 + d2 + d3

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, is_three_digit n ∧ is_multiple_of_9 n ∧ digit_sum n = 27 ∧
  ∀ m : ℕ, is_three_digit m ∧ is_multiple_of_9 m ∧ digit_sum m = 27 → m ≤ n := 
by 
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135310


namespace main_proposition_l135_135137

-- Definitions of the propositions
def p1 (a : ℝ) : Prop := ∀ x y : ℝ, a^x + x ≤ a^y + y → x ≤ y
def p2 : Prop := ∃ a b : ℝ, a^2 - a * b + b^2 < 0
def p3 (α β : ℝ) : Prop := (∃ k : ℤ, α = 2 * k * Real.pi + β)

-- Assumptions based on the solution
axiom H1 : ∀ a : ℝ, 0 < a → a ≠ 1 → ¬ p1 a
axiom H2 : ¬ p2
axiom H3 : ∀ α β : ℝ, p3 α β

-- Prove the main proposition
theorem main_proposition (α β : ℝ) : ¬ p2 ∧ p3 α β :=
by
  split
  exact H2
  exact H3 α β

end main_proposition_l135_135137


namespace enclosed_region_area_l135_135991

def g (x : ℝ) : ℝ := 2 - real.sqrt (4 - x^2)

theorem enclosed_region_area : 
  (let area := (2 * (real.pi - 2)) in real.floor (area * 100) / 100 = 2.28) :=
by sorry

end enclosed_region_area_l135_135991


namespace area_of_triangle_is_correct_l135_135724

def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_of_triangle_is_correct :
  triangle_area (-3) 2 8 (-3) 3 5 = 31.5 :=
by
  sorry

end area_of_triangle_is_correct_l135_135724


namespace percentage_return_l135_135187

theorem percentage_return
  (income : ℝ)
  (investment_cost : ℝ)
  (h_income : income = 650)
  (h_investment_cost : investment_cost = 6240) :
  ((income / investment_cost) * 100) ≈ 10.42 :=
by
  simp [h_income, h_investment_cost]
  unfold Real.div
  apply Real.approx
  -- Calculation omitted here
  sorry

end percentage_return_l135_135187


namespace teacher_dispatch_plans_l135_135013

theorem teacher_dispatch_plans :
  let teachers := {A, B, C, D, E}
  let areas := {area1, area2, area3}
  let condition1 := ∃ area ∈ areas, A ∈ area ∧ C ∈ area  -- A and C must be in the same area
  let condition2 := ¬∃ area ∈ areas, A ∈ area ∧ B ∈ area  -- A and B must not be in the same area
  let condition3 := ∀ area ∈ areas, ∃ teacher ∈ teachers, teacher ∈ area  -- each area has at least one teacher
  let dispatch_plans := {plan : set (set (A ∪ B ∪ C ∪ D ∪ E)) // condition1 ∧ condition2 ∧ condition3 → finset.card plan = 3}
  in finset.card dispatch_plans = 30 :=
sorry

end teacher_dispatch_plans_l135_135013


namespace bamboo_middle_node_capacity_l135_135267

def capacities_form_arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

theorem bamboo_middle_node_capacity :
  ∃ (a : ℕ → ℚ) (d : ℚ), 
    capacities_form_arithmetic_sequence a d ∧ 
    (a 1 + a 2 + a 3 = 4) ∧
    (a 6 + a 7 + a 8 + a 9 = 3) ∧
    (a 5 = 67 / 66) :=
  sorry

end bamboo_middle_node_capacity_l135_135267


namespace find_n_l135_135461

def isValidDigit (d : ℕ) : Prop := d < 10

def sum_of_squares_of_digits (m : ℕ) : ℕ :=
  (nat.digits 10 m).filter isValidDigit |>.map (λ d, d * d) |>.sum

def is_multiple_of_2022 (m : ℕ) : Prop := m % 2022 = 0

def isValidN (n : ℕ) : Prop :=
  ∃ m : ℕ, is_multiple_of_2022 m ∧ sum_of_squares_of_digits m = n

theorem find_n : ∀ n : ℕ, n > 0 →
  (isValidN n ↔ (n = 3 ∨ n = 5 ∨ n = 6 ∨ n ≥ 8)) :=
by
  sorry

end find_n_l135_135461


namespace real_part_condition_imaginary_part_condition_real_z_for_m_minus_2_imaginary_z_for_m_3_l135_135135

-- Define the complex number z
def complex_z (m : ℝ) : ℂ :=
  (m^2 - m - 6) / (m + 3) + complex.mk 0 (m^2 + 5*m + 6)

-- Real number condition
theorem real_part_condition (m : ℝ) (h1 : m ≠ -3) (h2 : m^2 + 5 * m + 6 = 0) : complex_z m = (m^2 - m - 6) / (m + 3) :=
by sorry

-- Pure imaginary number condition
theorem imaginary_part_condition (m : ℝ) (h1 : m ≠ -3) (h2 : m^2 + 5 * m + 6 ≠ 0) (h3 : m^2 - m - 6 = 0) : complex_z m = complex.mk 0 (m^2 + 5 * m + 6) :=
by sorry

-- Prove that for m = -2, complex_z m is real number
theorem real_z_for_m_minus_2 : complex_z (-2) = (2 / 1) :=
by {
  have h1: -2 ≠ -3, by linarith,
  have h2: (-2)^2 + 5*(-2) + 6 = 0, by norm_num,
  exact real_part_condition (-2) h1 h2
}

-- Prove that for m = 3, complex_z m is pure imaginary number
theorem imaginary_z_for_m_3 : complex_z 3 = complex.mk 0 24 :=
by {
  have h1: 3 ≠ -3, by linarith,
  have h2: 3^2 + 5 * 3 + 6 ≠ 0, by norm_num,
  have h3: 3^2 - 3 - 6 = 0, by norm_num,
  exact imaginary_part_condition 3 h1 h2 h3
}

end real_part_condition_imaginary_part_condition_real_z_for_m_minus_2_imaginary_z_for_m_3_l135_135135


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135335

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n % 9 = 0 ∧ (n.digits.sum = 27) ∧
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ m % 9 = 0 ∧ (m.digits.sum = 27) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135335


namespace samuel_remaining_distance_l135_135248

noncomputable def remaining_distance
  (total_distance : ℕ)
  (segment1_speed : ℕ) (segment1_time : ℕ)
  (segment2_speed : ℕ) (segment2_time : ℕ)
  (segment3_speed : ℕ) (segment3_time : ℕ)
  (segment4_speed : ℕ) (segment4_time : ℕ) : ℕ :=
  total_distance -
  (segment1_speed * segment1_time +
   segment2_speed * segment2_time +
   segment3_speed * segment3_time +
   segment4_speed * segment4_time)

theorem samuel_remaining_distance :
  remaining_distance 1200 60 2 70 3 50 4 80 5 = 270 :=
by
  sorry

end samuel_remaining_distance_l135_135248


namespace perfect_cube_prime_form_unique_l135_135609

theorem perfect_cube_prime_form_unique (p : ℕ) (n : ℕ) (h1 : prime p) (h2 : n = 2 * p + 1) (h3 : ∃ x : ℕ, n = x^3) : n = 27 :=
sorry

end perfect_cube_prime_form_unique_l135_135609


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135342

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) ∧ 
            ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ (m.digits.sum = 27) → m ≤ n :=
begin
  use 999,
  split,
  { -- 999 is a three-digit number 
    norm_num,
  },
  split,
  { -- 999 is less than or equal to 999
    norm_num,
  },
  split,
  { -- 999 is a multiple of 9
    norm_num,
  },
  split,
  { -- The sum of the digits of 999 is 27
    norm_num,
  },
  { -- For any three-digit number m, if it is a multiple of 9 and the sum of its digits is 27, then m ≤ 999
    intros m hm1,
    cases hm1 with hm2 hm3,
    cases hm3 with hm4 hm5,
    exact le_of_lt (by linarith),
    sorry
  },
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135342


namespace TriangleABC_PC_eq_16_l135_135662

-- Define the triangle and point P with conditions
structure TriangleABC (A B C P : Type) :=
  (right_angle_at_B : ∠B = 90)
  (PA_eq_14 : PA = 14)
  (PB_eq_8 : PB = 8)
  (angle_APB_90 : ∠APB = 90)
  (angle_BPC_135 : ∠BPC = 135)
  (angle_CPA_135 : ∠CPA = 135)

-- State the theorem to prove PC = 16
theorem TriangleABC_PC_eq_16 
  {A B C P : Type}
  (tabc : TriangleABC A B C P) :
  PC = 16 :=
sorry

end TriangleABC_PC_eq_16_l135_135662


namespace area_excluding_holes_l135_135006

theorem area_excluding_holes (x : ℝ) :
  let A_large : ℝ := (x + 8) * (x + 6)
  let A_hole : ℝ := (2 * x - 4) * (x - 3)
  A_large - 2 * A_hole = -3 * x^2 + 34 * x + 24 := by
  sorry

end area_excluding_holes_l135_135006


namespace find_break_with_binary_search_l135_135758

def wire_length : ℝ := 1  -- 1 meter long wire
def break_invisible : Prop := ∀ (p : ℝ), ¬visible_break p -- Break is not visible from the outside
def no_conductivity : Prop := ∀ (p : ℝ), ¬conductive p -- Break prevents conductivity

theorem find_break_with_binary_search (wire_length: ℝ) (break_invisible: Prop) (no_conductivity: Prop) :
  ∃ checks: ℕ, checks = 5 ∧ ∀ (break_location: ℝ), binary_search break_location → break_location ≤ 0.04 :=
sorry

end find_break_with_binary_search_l135_135758


namespace solve_for_a_l135_135104

theorem solve_for_a (a : ℝ) : (a - complex.I)^2 = 2 * complex.I → a = -1 :=
by
  intro h
  have h := congr_arg complex.re (complex.ext_iff.mp h).left
  sorry

end solve_for_a_l135_135104


namespace school_play_seating_l135_135093

theorem school_play_seating :
  (∃ (rows : ℕ) (standard_per_row : ℕ) (extra_per_row : ℕ) (unoccupied : ℕ),
    rows = 40 ∧ standard_per_row = 20 ∧ extra_per_row = 5 ∧ unoccupied = 10 ∧
    let total_rows := rows,
        standard_rows := total_rows / 2,
        extra_rows := total_rows / 2,
        total_standard_chairs := standard_rows * standard_per_row,
        total_extra_chairs := extra_rows * (standard_per_row + extra_per_row),
        total_chairs := total_standard_chairs + total_extra_chairs,
        seats_taken := total_chairs - unoccupied
    in seats_taken = 890 ) :=
sorry

end school_play_seating_l135_135093


namespace books_in_either_but_not_both_l135_135021

theorem books_in_either_but_not_both (shared_books alice_books bob_unique_books : ℕ) 
    (h1 : shared_books = 12) 
    (h2 : alice_books = 26)
    (h3 : bob_unique_books = 8) : 
    (alice_books - shared_books) + bob_unique_books = 22 :=
by
  sorry

end books_in_either_but_not_both_l135_135021


namespace general_term_formula_smallest_m_l135_135626

-- Define the arithmetic sequence and its sum condition
def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def S_n (a : ℕ → ℝ) (n : ℕ) := n * a ((n - 1) / 2)

axiom S7_eq_7 : S_n a 7 = 7
axiom S15_eq_75 : S_n a 15 = 75

-- Derive the general term formula of the sequence
theorem general_term_formula : ∃ d, ∃ a_4 : ℝ, d = 1 ∧ a_4 = 1 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
begin
  sorry,
end

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 2 * (n - 3) + 5

-- Define the sum T_n
def T_n (n : ℕ) : ℝ :=
  (∑ k in finset.range n, 1 / (b k * b (k + 1)))

-- Prove the smallest positive integer m such that T_n < m / 4
theorem smallest_m (n : ℕ) : ∃ m : ℕ, m = 2 ∧ ∀ n, T_n n < m / 4 :=
begin
  sorry,
end

end general_term_formula_smallest_m_l135_135626


namespace polynomial_division_result_q_neg1_r_1_sum_l135_135596

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 + 5 * x^3 - 4 * x^2 + 2 * x + 1
noncomputable def d (x : ℝ) : ℝ := x^2 + 2 * x - 3
noncomputable def q (x : ℝ) : ℝ := 3 * x^2 + x
noncomputable def r (x : ℝ) : ℝ := 7 * x + 4

theorem polynomial_division_result : f (-1) = q (-1) * d (-1) + r (-1)
  ∧ f 1 = q 1 * d 1 + r 1 :=
by sorry

theorem q_neg1_r_1_sum : (q (-1) + r 1) = 13 :=
by sorry

end polynomial_division_result_q_neg1_r_1_sum_l135_135596


namespace tan_angle_sum_l135_135538

variable (a b : ℝ)

-- Definitions converted from conditions
def tan_sum_condition := tan a + tan b = 12
def cot_sum_condition := cot a + cot b = 5

-- Problem statement
theorem tan_angle_sum : tan_sum_condition a b → cot_sum_condition a b → tan (a + b) = -60 / 7 :=
by
  intros h1 h2
  sorry

end tan_angle_sum_l135_135538


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135317

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 + d2 + d3

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, is_three_digit n ∧ is_multiple_of_9 n ∧ digit_sum n = 27 ∧
  ∀ m : ℕ, is_three_digit m ∧ is_multiple_of_9 m ∧ digit_sum m = 27 → m ≤ n := 
by 
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135317


namespace part1_proof_part2_proof_l135_135950

-- Definitions for triangle sides and angles
variables {A B C a b c : ℝ}

-- Condition 1
def condition1 : Prop := sin C * sin (A - B) = sin B * sin (C - A)

-- Condition 2
def condition2 : Prop := A = 2 * B

-- Proof Problem 1
theorem part1_proof : condition1 → condition2 → C = 5 / 8 * π :=
by sorry

-- Proof Problem 2
theorem part2_proof : condition1 → condition2 → 2 * a^2 = b^2 + c^2 :=
by sorry

end part1_proof_part2_proof_l135_135950


namespace six_digit_divisibility_l135_135754

theorem six_digit_divisibility (n : ℕ) (h_valid_n : n ∈ finset.range 10)
  (h1 : (354 * 1000 + n * 100 + 28) % 4 = 0)
  (h2 : (3 + 5 + 4 + n + 2 + 8) % 9 = 0) : n = 5 :=
by
  sorry

end six_digit_divisibility_l135_135754


namespace rocky_first_round_knockouts_l135_135613

-- Define the conditions
def total_fights : ℕ := 190
def knockout_percentage : ℝ := 0.50
def first_round_knockout_percentage : ℝ := 0.20

-- Calculate the number of total knockouts
def total_knockouts : ℕ := (knockout_percentage * total_fights).toNat

-- Calculate the number of first-round knockouts
def first_round_knockouts : ℕ := (first_round_knockout_percentage * total_knockouts).toNat

-- Main theorem to prove
theorem rocky_first_round_knockouts :
  first_round_knockouts = 19 :=
by sorry

end rocky_first_round_knockouts_l135_135613


namespace ratio_of_full_boxes_l135_135737

theorem ratio_of_full_boxes 
  (F H : ℕ)
  (boxes_count_eq : F + H = 20)
  (parsnips_count_eq : 20 * F + 10 * H = 350) :
  F / (F + H) = 3 / 4 := 
by
  -- proof will be placed here
  sorry

end ratio_of_full_boxes_l135_135737


namespace sin_105_eq_l135_135431

theorem sin_105_eq : sin (105 * (Real.pi / 180)) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
  sorry

end sin_105_eq_l135_135431


namespace part1_C_value_part2_triangle_equality_l135_135981

noncomputable theory

variables (a b c : ℝ) (A B C : ℝ)
variables (h1 : A + B + C = Real.pi) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) (h3 : A = 2 * B)

-- Part 1: Proving that C = 5π/8 given the conditions
theorem part1_C_value :
  C = 5 * Real.pi / 8 :=
begin
  sorry
end

-- Part 2: Proving that 2a^2 = b^2 + c^2 given the conditions
theorem part2_triangle_equality :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
begin
  sorry
end

end part1_C_value_part2_triangle_equality_l135_135981


namespace three_circles_concurrent_or_parallel_l135_135989

noncomputable def concurrent_or_parallel 
  (C1 C2 C3 : Type) [circle C1] [circle C2] [circle C3] 
  (A1 B1 A2 B2 A3 B3 : Type) : Prop :=
  intersects C2 C3 A1 ∧ intersects C2 C3 B1 ∧
  intersects C1 C3 A2 ∧ intersects C1 C3 B2 ∧
  intersects C1 C2 A3 ∧ intersects C1 C2 B3 →
  are_concurrent_or_parallel A1 B1 A2 B2 A3 B3

theorem three_circles_concurrent_or_parallel 
  (C1 C2 C3 : Type) [hC1 : circle C1] [hC2 : circle C2] [hC3 : circle C3]
  (A1 B1 A2 B2 A3 B3 : Type) 
  (h1 : intersects C2 C3 A1)
  (h2 : intersects C2 C3 B1)
  (h3 : intersects C1 C3 A2)
  (h4 : intersects C1 C3 B2)
  (h5 : intersects C1 C2 A3)
  (h6 : intersects C1 C2 B3) : 
  are_concurrent_or_parallel A1 B1 A2 B2 A3 B3 := 
  by sorry

end three_circles_concurrent_or_parallel_l135_135989


namespace tangency_condition_l135_135131

theorem tangency_condition :
  (∀ a b k : ℝ, 
    (∀ f : ℝ → ℝ, f = (λ x, x^3 + a * x + b) → 
    f 1 = 3 ∧ f' 1 = k) →
    (∀ g : ℝ → ℝ, g = (λ x, k * x + 1) → 
    g 1 = 3) →
    (f' (1 : ℝ)) = k) →
  ∃ a b : ℝ, k = 2 ∧ a = -1 ∧ b = 1 ∧ 2 * a + b = -1 := 
by
  sorry

end tangency_condition_l135_135131


namespace area_of_triangle_APB_l135_135720

-- Definitions and conditions
def square_side_length : ℝ := 8
def point_A : ℝ × ℝ := (0, 0)
def point_B : ℝ × ℝ := (8, 0)
def point_C : ℝ × ℝ := (4, 8)
def point_P : ℝ × ℝ := (4, 4)

-- All segments are equal in length
def PA_len_eq_PB_len_eq_PC_len : Prop :=
  dist point_A point_P = dist point_B point_P ∧
  dist point_A point_P = dist point_C point_P

-- PC is perpendicular to FD
def PC_perpendicular_FD : Prop :=
  let point_D : ℝ × ℝ := (8, 8) in
  let point_F : ℝ × ℝ := (0, 8) in
  let slope_PC := (point_P.2 - point_C.2) / (point_P.1 - point_C.1) in
  let slope_FD := (point_F.2 - point_D.2) / (point_F.1 - point_D.1) in
  slope_PC * slope_FD = -1

-- Area of triangle APB
def area_of_APB : ℝ := 1 / 2 * 8 * 3

theorem area_of_triangle_APB :
  PA_len_eq_PB_len_eq_PC_len →
  PC_perpendicular_FD →
  area_of_APB = 12 :=
by
  intros
  sorry

end area_of_triangle_APB_l135_135720


namespace new_average_is_100_l135_135801

theorem new_average_is_100 (nums : List ℝ) (h_len : nums.length = 7) (h_avg : (nums.sum / 7) = 20) : 
  let multipliers := [2, 3, 4, 5, 6, 7, 8] in
  let new_nums := List.zipWith (· * ·) nums multipliers in
  (new_nums.sum / 7) = 100 :=
by
  sorry

end new_average_is_100_l135_135801


namespace find_m_l135_135829

noncomputable def ellipse := {p : ℝ × ℝ | (p.1 ^ 2 / 25) + (p.2 ^ 2 / 16) = 1}
noncomputable def hyperbola (m : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2 / m) - (p.2 ^ 2 / 5) = 1}

theorem find_m (m : ℝ) (h1 : ∃ f : ℝ × ℝ, f ∈ ellipse ∧ f ∈ hyperbola m) : m = 4 := by
  sorry

end find_m_l135_135829


namespace infinite_series_computation_l135_135993

noncomputable def infinite_series_sum (a b : ℝ) : ℝ :=
  ∑' n : ℕ, if n = 0 then (0 : ℝ) else
    (1 : ℝ) / ((2 * (n - 1 : ℕ) * a - (n - 2 : ℕ) * b) * (2 * n * a - (n - 1 : ℕ) * b))

theorem infinite_series_computation (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_ineq : a > b) :
  infinite_series_sum a b = 1 / ((2 * a - b) * (2 * b)) :=
by
  sorry

end infinite_series_computation_l135_135993


namespace seeds_total_l135_135400

theorem seeds_total (wednesday_seeds thursday_seeds : ℕ) (h_wed : wednesday_seeds = 20) (h_thu : thursday_seeds = 2) : (wednesday_seeds + thursday_seeds) = 22 := by
  sorry

end seeds_total_l135_135400


namespace sophomores_in_program_l135_135883

theorem sophomores_in_program (total_students : ℕ) (not_sophomores_nor_juniors : ℕ) 
    (percentage_sophomores_debate : ℚ) (percentage_juniors_debate : ℚ) 
    (eq_debate_team : ℚ) (total_students := 40) 
    (not_sophomores_nor_juniors := 5) 
    (percentage_sophomores_debate := 0.20) 
    (percentage_juniors_debate := 0.25) 
    (eq_debate_team := (percentage_sophomores_debate * S = percentage_juniors_debate * J)) :
    ∀ (S J : ℚ), S + J = total_students - not_sophomores_nor_juniors → 
    (S = 5 * J / 4) → S = 175 / 9 := 
by 
  sorry

end sophomores_in_program_l135_135883


namespace problem1_problem2_l135_135438

theorem problem1 : (1 : ℝ) * ((-2 : ℝ)^2) - (27 : ℝ)^(1 / 3) + (16 : ℝ)^(1 / 2) + (-1 : ℝ)^2023 = 4 := 
by
    sorry

theorem problem2 : abs (real.sqrt (2) - real.sqrt (3)) + 2 * real.sqrt 2 = real.sqrt 3 + real.sqrt 2 := 
by
    sorry

end problem1_problem2_l135_135438


namespace wendi_chickens_l135_135302

theorem wendi_chickens : 
  let initial_chickens := 4
  let doubled_chickens := initial_chickens * 2
  let after_dog := doubled_chickens - 1
  let found_chickens := 10 - 4
  let total_chickens := after_dog + found_chickens
  in total_chickens = 13 :=
by
  let initial_chickens := 4
  let doubled_chickens := initial_chickens * 2
  let after_dog := doubled_chickens - 1
  let found_chickens := 10 - 4
  let total_chickens := after_dog + found_chickens
  sorry

end wendi_chickens_l135_135302


namespace value_of_four_inch_cube_l135_135405

theorem value_of_four_inch_cube (value_one_inch_cube : ℝ) (volume_ratio : ℝ) : 
  value_one_inch_cube = 300 → volume_ratio = 64 → 
  let value_four_inch_cube := value_one_inch_cube * volume_ratio in
  value_four_inch_cube = 19200 := 
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end value_of_four_inch_cube_l135_135405


namespace probability_xi_l135_135224

open ProbabilityTheory

noncomputable def xi_dist (a : ℝ) : ℕ → ℝ := 
λ k, if k ∈ {1, 2, 3, 4, 5} then a * k else 0

theorem probability_xi :
  ∃ (a : ℝ), (a * (1 + 2 + 3 + 4 + 5) = 1) ∧
  (pmf.of_fn (xi_dist a)).prob (λ x, (1/10 < x) ∧ (x < 1/2)) = 1/5 :=
by
  sorry

end probability_xi_l135_135224


namespace part1_part2_l135_135696

-- Part 1: Proving the minimum number of questions needed
def need_at_least_11_questions (row1 : Fin 10 → ℤ) (row2 : Fin 10 → ℤ) : Prop :=
  ∀ (b : (Fin 10 → ℤ) → ℕ → ℤ), 
    (∀ i j k l : Fin 10, 
      row1 i + row2 k = row2 j + row1 l ∧
      ∃ i, row1 i = b row1 i ∨ row2 i = b row2 i) →
      ∃ q : ℕ, q = 11

-- Part 2: Proving the minimum number of remaining numbers in a grid
def min_remaining_numbers_in_grid (m n : ℕ) (grid : Fin m → Fin n → ℤ) : Prop :=
  ∀ (erase : Fin m → Fin n → Bool),
    (∀ i j k l : Fin m,
      ∑ a b, (erase i a = false ∨ erase j a = false ∨ erase k b = false ∨ erase l b = false) ∧
      grid i j + grid k l = grid i l + grid k j) →
        ∃ remaining : ℕ, remaining = m + n - 1

-- Statements
theorem part1: ∀ (row1 row2 : Fin 10 → ℤ), need_at_least_11_questions row1 row2 := sorry

theorem part2: ∀ (m n : ℕ) (grid : Fin m → Fin n → ℤ), 
  min_remaining_numbers_in_grid m n grid := sorry

end part1_part2_l135_135696


namespace time_C_proof_l135_135403

-- Definitions of the problem
def principal_B : ℝ := 5000
def time_B : ℕ := 2
def principal_C : ℝ := 3000
def total_interest : ℝ := 3300
def interest_rate : ℝ := 0.15

-- Simple interest function
def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T

-- Interest from B
def interest_B := simple_interest principal_B interest_rate time_B

-- Conditions
def condition_interest_sum : Prop :=
  let interest_C := total_interest - interest_B
  ∃ (time_C : ℝ), simple_interest principal_C interest_rate time_C = interest_C

-- Proof statement
theorem time_C_proof : condition_interest_sum :=
begin
  -- Proving the statement
  sorry
end

end time_C_proof_l135_135403


namespace part1_part2_l135_135920

theorem part1 (A B : ℝ) (h1 : A = 2 * B) : C = 5 * Real.pi / 8 :=
sorry

theorem part2 (a b c A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) :
   2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l135_135920


namespace sufficient_but_not_necessary_condition_l135_135194

variable (a b : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : b > a) (h2 : a > 0) : 
  (a * (b + 1) > a^2) ∧ ¬(∀ (a b : ℝ), a * (b + 1) > a^2 → b > a ∧ a > 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l135_135194


namespace f_is_constant_l135_135795

noncomputable def f (x θ : ℝ) : ℝ :=
  (Real.cos (x - θ))^2 + (Real.cos x)^2 - 2 * Real.cos θ * Real.cos (x - θ) * Real.cos x

theorem f_is_constant (θ : ℝ) : ∀ x, f x θ = (Real.sin θ)^2 :=
by
  intro x
  sorry

end f_is_constant_l135_135795


namespace magnitude_conjugate_l135_135485

open Complex

theorem magnitude_conjugate (z : ℂ) (hz : z = 1 - 2 * I) : |conj z| = Real.sqrt 5 := 
by
  sorry

end magnitude_conjugate_l135_135485


namespace union_of_sets_l135_135593

def is_imaginary_unit (i : ℂ) : Prop :=
  i^2 = -1

theorem union_of_sets (i : ℂ) (h : is_imaginary_unit i) :
  let A := {1, i}
  let B := { -1/i, (1-i)^2/2 }
  A ∪ B = {1, i, -i} :=
sorry

end union_of_sets_l135_135593


namespace leila_toys_l135_135911

theorem leila_toys:
  ∀ (x : ℕ),
  (∀ l m : ℕ, l = 2 * x ∧ m = 3 * 19 ∧ m = l + 7 → x = 25) :=
by
  sorry

end leila_toys_l135_135911


namespace sin_arithmetic_sequence_l135_135764

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 360) :
  (sin a + sin (3 * a) = 2 * sin (2 * a)) ↔ 
  (a = 30 ∨ a = 150 ∨ a = 210 ∨ a = 330) :=
by
  sorry

end sin_arithmetic_sequence_l135_135764


namespace f_zero_f_odd_l135_135128

-- Definition of the function f over ℝ with given condition.
variable {f : ℝ → ℝ}
axiom functional_eq : ∀ (x y : ℝ), f(x + y) = f(x) + f(y)

-- Statement for the value of f(0)
theorem f_zero : f 0 = 0 := 
sorry

-- Statement for the definition of a specific function that satisfies the conditions.
def specific_function (x : ℝ) : ℝ :=
  let k : ℝ := 1 -- or any constant k
  f x = k * x

-- Statement for f(x) being an odd function.
theorem f_odd : ∀ (x : ℝ), f (−x) = −f x := 
sorry

end f_zero_f_odd_l135_135128


namespace gcd_of_10011_and_15015_l135_135074

theorem gcd_of_10011_and_15015 : 
  ∀ (p q : ℕ), p = 10011 → q = 15015 → gcd 10011 15015 = 1001 :=
by
  intros p q hp hq
  rw [hp, hq]
  -- Factorization step
  have hfac1 : p = 11 * 1001 := by sorry
  have hfac2 : q = 15 * 1001 := by sorry
  -- GCD computation step
  have h_gcd_11_15 : gcd 11 15 = 1 := by sorry
  exact sorry

end gcd_of_10011_and_15015_l135_135074


namespace max_value_of_function_l135_135493

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1.5) : 
  ∃ m, ∀ y, y = 4 * x * (3 - 2 * x) → m = 9 / 2 :=
sorry

end max_value_of_function_l135_135493


namespace ben_savings_l135_135429

theorem ben_savings:
  ∃ x : ℕ, (50 - 15) * x * 2 + 10 = 500 ∧ x = 7 :=
by
  -- Definitions based on conditions
  let daily_savings := 50 - 15
  have h1 : daily_savings = 35 := by norm_num
  let total_savings := daily_savings * x
  let doubled_savings := 2 * total_savings
  let final_savings := doubled_savings + 10

  -- Existence of x such that (50 - 15) * x * 2 + 10 = 500 and x = 7 
  use 7
  split
  { -- Show that the equation holds
    show final_savings = 500,
    calc
      final_savings = (daily_savings * 7 * 2) + 10 : by sorry
                   ... = 500 : by norm_num
  }
  { -- Show that x = 7
    refl
  }
  sorry

end ben_savings_l135_135429


namespace russell_oranges_taken_l135_135848

-- Conditions
def initial_oranges : ℕ := 60
def oranges_left : ℕ := 25

-- Statement to prove
theorem russell_oranges_taken : ℕ :=
  initial_oranges - oranges_left = 35

end russell_oranges_taken_l135_135848


namespace sum_of_primes_equals_l135_135445

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit_prime (p : ℕ) : Prop :=
  p ≥ 10 ∧ p < 100 ∧ is_prime p

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def sum_of_special_primes := ∑ p in Finset.filter (λ p, ∃ q, is_prime q ∧ is_perfect_square (100 * q + p)) 
                                                (Finset.filter is_two_digit_prime (Finset.range 100)), id

theorem sum_of_primes_equals : sum_of_special_primes = 179 :=
sorry

end sum_of_primes_equals_l135_135445


namespace monotonic_interval_area_triangle_ABC_l135_135832

-- Define the function f(x)
def f (x : ℝ) := 2 * real.sqrt 3 * real.sin x * real.cos x - real.cos (2 * x)

-- State the monotonic interval for f(x)
theorem monotonic_interval :
  ∀ k : ℤ, ∀ x : ℝ, (k * real.pi - real.pi / 6) ≤ x ∧ x ≤ (k * real.pi + real.pi / 3) → 
  monotone_on f (set.Icc (k * real.pi - real.pi / 6) (k * real.pi + real.pi / 3)) :=
sorry

-- State the area theorem for triangle ABC
theorem area_triangle_ABC :
  ∀ (A B C : ℝ) (a b c : ℝ), f A = 2 ∧ C = real.pi / 4 ∧ c = 2 →
  ∃ S : ℝ, S = (1 / 2) * a * c * real.sin B ∧ S = (3 + real.sqrt 3) / 2 :=
sorry

end monotonic_interval_area_triangle_ABC_l135_135832


namespace Asya_Petya_l135_135756

theorem Asya_Petya (a b : ℕ) (ha : 100 ≤ a ∧ a < 1000) (hb : 100 ≤ b ∧ b < 1000) 
  (h : 1000 * a + b = 7 * a * b) : a = 143 ∧ b = 143 :=
by
  sorry

end Asya_Petya_l135_135756


namespace find_s_for_g_l135_135986

def g (x : ℝ) (s : ℝ) : ℝ := 3*x^4 - 2*x^3 + 2*x^2 + x + s

theorem find_s_for_g (s : ℝ) : g (-1) s = 0 ↔ s = -6 :=
by
  sorry

end find_s_for_g_l135_135986


namespace trajectory_of_M_min_area_of_triangle_l135_135563

-- Definitions for Question 1
def Point := ℝ × ℝ
def A : Point := (1/2, 0)
def is_on_line (B : Point) : Prop := B.fst = -1/2
def intersects_y_axis (P Q : Point) : Point := (0, P.snd + Q.snd) / 2
def perpendicular (u v : Point) : Prop := (u.fst * v.fst + u.snd * v.snd) = 0
def equal_length (M B A : Point) : Prop := (B.fst - M.fst)^2 + (B.snd - M.snd)^2 = (A.fst - M.fst)^2 + (A.snd - M.snd)^2

-- Statement of Question 1 to be proved
theorem trajectory_of_M (B : Point) (C : Point) (M : Point) (OC : Point) 
  (hB : is_on_line B) (hC : C = intersects_y_axis A B) 
  (hM1 : perpendicular (M.fst - B.fst, M.snd - B.snd) OC)
  (hM2 : perpendicular (M.fst - C.fst, M.snd - C.snd) (B.fst - C.fst, B.snd - C.snd))
  (hM3 : equal_length M B A) : M.snd^2 = 2 * M.fst :=
sorry

-- Definitions for Question 2
def is_on_parabola (P : Point) : Prop := P.snd^2 = 2 * P.fst
def circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def triangle_area (b c : ℝ) : ℝ := ((b - c) * b) / 2

-- Statement of Question 2 to be proved
theorem min_area_of_triangle 
  (P : Point) (b c : ℝ) 
  (hP : is_on_parabola P) 
  (hCircle : circle P.fst P.snd)
  (hb_gt_c : b > c) : 
  let S := triangle_area b c 
  in S ≥ 8 :=
sorry

end trajectory_of_M_min_area_of_triangle_l135_135563


namespace no_magic_square_using_first_nine_primes_l135_135558

-- Definitions of the magic square and the first nine prime numbers
def is_magic_square (m : matrix (fin 3) (fin 3) ℕ) : Prop :=
  ∀ i j, ∑ k in finset.univ, (m i k) = ∑ k in finset.univ, (m k j) ∧
  ∑ k in finset.univ, (m k k) = ∑ k in finset.univ, (m k (2 - k))

def is_prime (n : ℕ) : Prop := nat.prime n

noncomputable def first_nine_primes : list ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Main theorem
theorem no_magic_square_using_first_nine_primes : 
  ¬ ∃ (m : matrix (fin 3) (fin 3) ℕ),
    is_magic_square m ∧ ∀ i j, m i j ∈ first_nine_primes :=
by -- Proof goes here
  sorry

end no_magic_square_using_first_nine_primes_l135_135558


namespace tan_equality_condition_l135_135105

open Real

theorem tan_equality_condition (α β : ℝ) :
  (α = β) ↔ (tan α = tan β) :=
sorry

end tan_equality_condition_l135_135105


namespace sin_of_3halfpiplus2theta_l135_135099

theorem sin_of_3halfpiplus2theta (θ : ℝ) (h : Real.tan θ = 1 / 3) : Real.sin (3 * π / 2 + 2 * θ) = -4 / 5 := 
by 
  sorry

end sin_of_3halfpiplus2theta_l135_135099


namespace find_C_prove_relation_l135_135956

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A), and A = 2B,
prove that C = 5/8 * π. -/
theorem find_C
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  C = ⅝ * Real.pi :=
sorry

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A),
prove that 2 * a ^ 2 = b ^ 2 + c ^ 2. -/
theorem prove_relation
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
sorry

end find_C_prove_relation_l135_135956


namespace number_of_zeros_of_g_l135_135512

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then 3 - 2 * x else x^2

noncomputable def g (x : ℝ) : ℝ := f x - 2

theorem number_of_zeros_of_g : set.count {x : ℝ | g x = 0}.to_finset == 2 :=
by
  sorry

end number_of_zeros_of_g_l135_135512


namespace range_of_a_exists_l135_135807

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 1
def g (x a : ℝ) : ℝ := 2 * x + a

theorem range_of_a_exists (a : ℝ) :
  (∃ x1 x2 ∈ Icc (1 / 2 : ℝ) 1, f x1 = g x2 a) ↔ -3 ≤ a ∧ a ≤ -3 / 2 :=
by
  sorry

end range_of_a_exists_l135_135807


namespace part1_part2_l135_135599

section
variable (x a : ℝ)

def p (a x : ℝ) : Prop :=
  x^2 - 4*a*x + 3*a^2 < 0 ∧ a > 0

def q (x : ℝ) : Prop :=
  (x - 3) / (x - 2) ≤ 0

theorem part1 (h1 : p 1 x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

theorem part2 (h2 : ∀ x, ¬p a x → ¬q x) : 1 < a ∧ a ≤ 2 := by
  sorry

end

end part1_part2_l135_135599


namespace cylinder_radius_exists_l135_135714

theorem cylinder_radius_exists (r h : ℕ) (pr : r ≥ 1) :
  (π * ↑r ^ 2 * ↑h = 2 * π * ↑r * (↑h + ↑r)) ↔
  (r = 3 ∨ r = 4 ∨ r = 6) :=
by
  sorry

end cylinder_radius_exists_l135_135714


namespace pentagon_equality_l135_135235

variables {A B C D E : Type} [EuclideanGeometry A]

def area (X Y Z : A) : ℝ := sorry -- Area of triangle XYZ

theorem pentagon_equality (ABCDE : ConvexPentagon E A B C D) :
  area E A C * area E B D = area E A B * area E C D + area E B C * area E D A :=
sorry

end pentagon_equality_l135_135235


namespace ben_remaining_money_l135_135031

variable (initial_capital : ℝ := 2000) 
variable (payment_to_supplier : ℝ := 600)
variable (payment_from_debtor : ℝ := 800)
variable (maintenance_cost : ℝ := 1200)
variable (remaining_capital : ℝ := 1000)

theorem ben_remaining_money
  (h1 : initial_capital = 2000)
  (h2 : payment_to_supplier = 600)
  (h3 : payment_from_debtor = 800)
  (h4 : maintenance_cost = 1200) :
  remaining_capital = (initial_capital - payment_to_supplier + payment_from_debtor - maintenance_cost) :=
sorry

end ben_remaining_money_l135_135031


namespace product_of_values_for_a_l135_135278

theorem product_of_values_for_a : 
  ∃ a b : ℝ, (3*(sqrt 10) = sqrt ((3*a - 5)^2 + (2*a - 5)^2) ∧ 3*(sqrt 10) = sqrt ((3*b - 5)^2 + (2*b - 5)^2)) ∧ 
             a * b = -40/13 :=
sorry

end product_of_values_for_a_l135_135278


namespace dot_product_xy_l135_135125

open Real

noncomputable def dot_product
  (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem dot_product_xy :
  -- Given conditions
  let a := (1 : ℝ, 0)
  let b := (1 / 2, sqrt 3 / 2)
  let x := (2 * a.1 - b.1, 2 * a.2 - b.2)
  let y := (3 * b.1 - a.1, 3 * b.2 - a.2)
  -- We want to show x ∙ y = -3/2
  dot_product x y = -3 / 2 := 
sorry

end dot_product_xy_l135_135125


namespace fraction_to_decimal_1_fraction_to_decimal_2_decimal_to_fraction_1_decimal_to_fraction_2_l135_135053

theorem fraction_to_decimal_1 : (60 / 4 : ℚ) = 15 := by
  calc
  (60 / 4 : ℚ) = 15 : by norm_num

theorem fraction_to_decimal_2 : (19 / 6 : ℚ) = 3.167 := by
  calc
  (19 / 6 : ℚ) = 3167 / 1000 : by norm_num
  ... = (3167 / 1000 : ℚ) : by norm_num
  ... = 3.167 : by sorry  -- rounding to 3 decimal places needs explanation

theorem decimal_to_fraction_1 : (0.25 : ℚ) = 1 / 4 := by
  calc
  (0.25 : ℚ) = 25 / 100 : by norm_num
  ... = 1 / 4 : by norm_num

theorem decimal_to_fraction_2 : (0.08 : ℚ) = 2 / 25 := by
  calc
  (0.08 : ℚ) = 8 / 100 : by norm_num
  ... = 2 / 25 : by norm_num

end fraction_to_decimal_1_fraction_to_decimal_2_decimal_to_fraction_1_decimal_to_fraction_2_l135_135053


namespace length_of_integer_eq_24_l135_135779

theorem length_of_integer_eq_24 (k : ℕ) (h1 : k > 1) (h2 : ∃ (p1 p2 p3 p4 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ k = p1 * p2 * p3 * p4) : k = 24 := by
  sorry

end length_of_integer_eq_24_l135_135779


namespace x3_plus_y3_values_l135_135050

noncomputable def x_y_satisfy_eqns (x y : ℝ) : Prop :=
  y^2 - 3 = (x - 3)^3 ∧ x^2 - 3 = (y - 3)^2 ∧ x ≠ y

theorem x3_plus_y3_values (x y : ℝ) (h : x_y_satisfy_eqns x y) :
  x^3 + y^3 = 27 + 3 * Real.sqrt 3 ∨ x^3 + y^3 = 27 - 3 * Real.sqrt 3 :=
  sorry

end x3_plus_y3_values_l135_135050


namespace runner_time_difference_l135_135716

theorem runner_time_difference (v : ℝ) (h1 : 0 < v) (h2 : 0 < 20 / v) (h3 : 8 = 40 / v) :
  8 - (20 / v) = 4 := by
  sorry

end runner_time_difference_l135_135716


namespace jack_morning_emails_l135_135902

-- Define the conditions as constants
def totalEmails : ℕ := 10
def emailsAfternoon : ℕ := 3
def emailsEvening : ℕ := 1

-- Problem statement to prove emails in the morning
def emailsMorning : ℕ := totalEmails - (emailsAfternoon + emailsEvening)

-- The theorem to prove
theorem jack_morning_emails : emailsMorning = 6 := by
  sorry

end jack_morning_emails_l135_135902


namespace paint_cost_l135_135706

theorem paint_cost (cost_A_per_kg : ℝ) (coverage_A_per_kg : ℝ) (area_A : ℝ)
                   (cost_B_per_kg : ℝ) (coverage_B_per_kg : ℝ) (area_B : ℝ)
                   (cost_C_per_kg : ℝ) (coverage_C_per_kg : ℝ) (area_C : ℝ) :
  cost_A_per_kg = 60 ∧ coverage_A_per_kg = 20 ∧ area_A = 50 ∧
  cost_B_per_kg = 45 ∧ coverage_B_per_kg = 25 ∧ area_B = 80 ∧
  cost_C_per_kg = 80 ∧ coverage_C_per_kg = 30 ∧ area_C = 30 →
  let kg_A := area_A / coverage_A_per_kg in
  let kg_B := area_B / coverage_B_per_kg in
  let kg_C := area_C / coverage_C_per_kg in
  let cost_A := kg_A * cost_A_per_kg in
  let cost_B := kg_B * cost_B_per_kg in
  let cost_C := kg_C * cost_C_per_kg in
  cost_A + cost_B + cost_C = 374 :=
by
  intros h
  sorry

end paint_cost_l135_135706


namespace inequality_solution_set_l135_135286

theorem inequality_solution_set :
  ( ∀ x : ℝ, ( (1 / 2) ^ (x - x^2) < log 3 81 ) ↔ (-1 < x ∧ x < 2) ) :=
begin
  sorry
end

end inequality_solution_set_l135_135286


namespace savings_with_discount_l135_135584

theorem savings_with_discount :
  let original_price := 3.00
  let discount_rate := 0.30
  let discounted_price := original_price * (1 - discount_rate)
  let number_of_notebooks := 7
  let total_cost_without_discount := number_of_notebooks * original_price
  let total_cost_with_discount := number_of_notebooks * discounted_price
  total_cost_without_discount - total_cost_with_discount = 6.30 :=
by
  sorry

end savings_with_discount_l135_135584


namespace find_C_prove_relation_l135_135957

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A), and A = 2B,
prove that C = 5/8 * π. -/
theorem find_C
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  C = ⅝ * Real.pi :=
sorry

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A),
prove that 2 * a ^ 2 = b ^ 2 + c ^ 2. -/
theorem prove_relation
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
sorry

end find_C_prove_relation_l135_135957


namespace x_plus_y_value_l135_135686

def sum_evens_40_to_60 : ℕ :=
  (40 + 42 + 44 + 46 + 48 + 50 + 52 + 54 + 56 + 58 + 60)

def num_evens_40_to_60 : ℕ := 11

theorem x_plus_y_value : sum_evens_40_to_60 + num_evens_40_to_60 = 561 := by
  sorry

end x_plus_y_value_l135_135686


namespace one_belt_one_road_problem_1_one_belt_one_road_problem_2_l135_135802

theorem one_belt_one_road_problem_1 (m n : ℝ) :
  (∃ (P : ℝ × ℝ), P = (0, 1) ∧
   (∀ (x : ℝ), x^2 - 2 * x + n = x) ∧
   (∀ (y : ℝ), y = m * 1 + 1)) →
  n = 1 ∧ m = -1 := sorry

theorem one_belt_one_road_problem_2 :
  (∃ (a h k : ℝ),
   (∀ (x : ℝ), 2 * x - 4 = 6 / x) ∧
   (a != 0) ∧
   (h,y) = (3, 2) ∨ (h, y) = (-1, -6) ∧
   (y = 2 * (x + h) ^ 2 + k ∨ y = (-⅔) * (x - 3) ^ 2 + 2)) :=
  exists_intro 2 (exists_intro 1 (exists_intro (-6)
    (and.intro
      (exists_intro 1
        (forall.intro 
          (λ x, 2 * x - 4 = 6 / x)
        )
      )
      (exists_intro (-⅔)
        (forall.intro
          (λ x, 2 * x - 4 = 6 / x)
        )
      )
    )
  )) := sorry

end one_belt_one_road_problem_1_one_belt_one_road_problem_2_l135_135802


namespace binary_arithmetic_l135_135762

-- Define the binary numbers 11010_2, 11100_2, and 100_2
def x : ℕ := 0b11010 -- base 2 number 11010 in base 10 representation
def y : ℕ := 0b11100 -- base 2 number 11100 in base 10 representation
def d : ℕ := 0b100   -- base 2 number 100 in base 10 representation

-- Define the correct answer
def correct_answer : ℕ := 0b10101101 -- base 2 number 10101101 in base 10 representation

-- The proof problem statement
theorem binary_arithmetic : (x * y) / d = correct_answer := by
  sorry

end binary_arithmetic_l135_135762


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135329

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ (100 ≤ n ∧ n < 1000) ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) :=
by {
  sorry
}

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135329


namespace primes_ge_3_are_4k_pm1_infinitely_many_primes_4k_minus1_l135_135384

-- Part 1: Prove that every prime number >= 3 is of the form 4k-1 or 4k+1
theorem primes_ge_3_are_4k_pm1 (p : ℕ) (hp_prime: Nat.Prime p) (hp_ge_3: p ≥ 3) : 
  ∃ k : ℕ, p = 4 * k + 1 ∨ p = 4 * k - 1 :=
by
  sorry

-- Part 2: Prove that there are infinitely many primes of the form 4k-1
theorem infinitely_many_primes_4k_minus1 : 
  ∀ (n : ℕ), ∃ (p : ℕ), Nat.Prime p ∧ p = 4 * k - 1 ∧ p > n :=
by
  sorry

end primes_ge_3_are_4k_pm1_infinitely_many_primes_4k_minus1_l135_135384


namespace problem_statement_l135_135222

def g (x : ℝ) : ℝ :=
  x^2 - 5 * x

theorem problem_statement (x : ℝ) :
  (g (g x) = g x) ↔ (x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1) :=
by
  sorry

end problem_statement_l135_135222


namespace total_cats_handled_last_year_l135_135583

theorem total_cats_handled_last_year (num_adult_cats : ℕ) (two_thirds_female : ℕ) (seventy_five_percent_litters : ℕ) 
                                     (kittens_per_litter : ℕ) (adopted_returned : ℕ) :
  num_adult_cats = 120 →
  two_thirds_female = (2 * num_adult_cats) / 3 →
  seventy_five_percent_litters = (3 * two_thirds_female) / 4 →
  kittens_per_litter = 3 →
  adopted_returned = 15 →
  num_adult_cats + seventy_five_percent_litters * kittens_per_litter + adopted_returned = 315 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end total_cats_handled_last_year_l135_135583


namespace log_product_2015_l135_135600

noncomputable def x_n (n : Nat) (hn : n > 0) : ℝ := n / (n + 1)

theorem log_product_2015 : 
  (∑ (i : Fin 2014), Real.logb 2015 (x_n (i + 1) i.is_pos)) = -1 := 
by 
  -- proof omitted
  sorry

end log_product_2015_l135_135600


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135334

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n % 9 = 0 ∧ (n.digits.sum = 27) ∧
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ m % 9 = 0 ∧ (m.digits.sum = 27) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135334


namespace complex_real_imag_eq_l135_135547

theorem complex_real_imag_eq (b : ℝ) (h : (2 + b) / 5 = (2 * b - 1) / 5) : b = 3 :=
  sorry

end complex_real_imag_eq_l135_135547


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135345

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) ∧ 
            ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ (m.digits.sum = 27) → m ≤ n :=
begin
  use 999,
  split,
  { -- 999 is a three-digit number 
    norm_num,
  },
  split,
  { -- 999 is less than or equal to 999
    norm_num,
  },
  split,
  { -- 999 is a multiple of 9
    norm_num,
  },
  split,
  { -- The sum of the digits of 999 is 27
    norm_num,
  },
  { -- For any three-digit number m, if it is a multiple of 9 and the sum of its digits is 27, then m ≤ 999
    intros m hm1,
    cases hm1 with hm2 hm3,
    cases hm3 with hm4 hm5,
    exact le_of_lt (by linarith),
    sorry
  },
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135345


namespace unique_spicy_pair_l135_135439

def is_spicy (n : ℕ) : Prop :=
  let A := (n / 100) % 10
  let B := (n / 10) % 10
  let C := n % 10
  n = A^3 + B^3 + C^3

theorem unique_spicy_pair : ∃! n : ℕ, is_spicy n ∧ is_spicy (n + 1) ∧ 100 ≤ n ∧ n < 1000 ∧ n = 370 := 
sorry

end unique_spicy_pair_l135_135439


namespace find_a_if_f_even_l135_135174

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * 2^x + 2^(-x)

-- The proof problem
theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, f a (-x) = f a x) : a = 1 :=
sorry

end find_a_if_f_even_l135_135174


namespace oranges_taken_l135_135846

theorem oranges_taken (initial_oranges remaining_oranges taken_oranges : ℕ) 
  (h1 : initial_oranges = 60) 
  (h2 : remaining_oranges = 25) 
  (h3 : taken_oranges = initial_oranges - remaining_oranges) : 
  taken_oranges = 35 :=
by
  -- Proof is omitted, as instructed.
  sorry

end oranges_taken_l135_135846


namespace radical_multiplication_l135_135306

noncomputable def root4 (x : ℝ) : ℝ := x ^ (1/4)
noncomputable def root3 (x : ℝ) : ℝ := x ^ (1/3)
noncomputable def root2 (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_multiplication : root4 256 * root3 8 * root2 16 = 32 := by
  sorry

end radical_multiplication_l135_135306


namespace size_A_geq_size_B_l135_135475

variables {S : set (ℤ × ℤ)}

def is_valid_A (A : set (ℤ × ℤ)) : Prop :=
  A ⊆ S ∧ ∀ (p1 p2 : ℤ × ℤ), p1 ∈ A → p2 ∈ A → p1 ≠ p2 → p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2

def is_valid_B (B : set ℤ) : Prop :=
  ∀ (p : ℤ × ℤ), p ∈ S → p.1 ∈ B ∨ p.2 ∈ B

def max_A : set (ℤ × ℤ) :=
  classical.some (set.exists_maximal is_valid_A)

def min_B : set ℤ :=
  classical.some (set.exists_minimal (λ B, is_valid_B B ∧ ∀ B', is_valid_B B' → B ⊆ B'))

lemma valid_max_A : is_valid_A max_A := 
  classical.some_spec (set.exists_maximal is_valid_A)

lemma valid_min_B : is_valid_B min_B ∧ ∀ B', is_valid_B B' → min_B ⊆ B' :=
  classical.some_spec (set.exists_minimal (λ B, is_valid_B B ∧ ∀ B', is_valid_B B' → B ⊆ B'))

theorem size_A_geq_size_B : fintype.card max_A ≥ fintype.card min_B :=
  sorry

end size_A_geq_size_B_l135_135475


namespace plants_producing_flowers_l135_135530

noncomputable def germinate_percent_daisy : ℝ := 0.60
noncomputable def germinate_percent_sunflower : ℝ := 0.80
noncomputable def produce_flowers_percent : ℝ := 0.80
noncomputable def daisy_seeds_planted : ℕ := 25
noncomputable def sunflower_seeds_planted : ℕ := 25

theorem plants_producing_flowers : 
  let daisy_plants_germinated := germinate_percent_daisy * daisy_seeds_planted,
      sunflower_plants_germinated := germinate_percent_sunflower * sunflower_seeds_planted,
      total_plants_germinated := daisy_plants_germinated + sunflower_plants_germinated,
      plants_that_produce_flowers := produce_flowers_percent * total_plants_germinated
  in plants_that_produce_flowers = 28 :=
by
  sorry

end plants_producing_flowers_l135_135530


namespace lower_limit_for_x_l135_135173

variable {n : ℝ} {x : ℝ} {y : ℝ}

theorem lower_limit_for_x (h1 : x > n) (h2 : x < 8) (h3 : y > 8) (h4 : y < 13) (h5 : y - x = 7) : x = 2 :=
sorry

end lower_limit_for_x_l135_135173


namespace problem_1_problem_2_l135_135038

-- Problem 1
theorem problem_1 :
  -((1 / 2) / 3) * (3 - (-3)^2) = 1 :=
by
  sorry

-- Problem 2
theorem problem_2 {x : ℝ} (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (2 * x) / (x^2 - 4) - 1 / (x - 2) = 1 / (x + 2) :=
by
  sorry

end problem_1_problem_2_l135_135038


namespace simplify_factorial_fraction_l135_135442

theorem simplify_factorial_fraction (N : ℕ) : 
  ((N + 1)! * N) / ((N + 2)!) = N / (N + 2) :=
by
  sorry

end simplify_factorial_fraction_l135_135442


namespace pentagon_angles_adjacent_l135_135728

theorem pentagon_angles_adjacent (P : Type) [convex_pentagon P]
  (sides_eq : ∀ (s1 s2 : side P), s1 =s= s2)
  (angles_different : ∀ (a1 a2 : angle P), a1 ≠ a2) :
  ∃ (side : side P), ∃ (a1 a2 : angle P), a1 = max_angle P ∧ a2 = min_angle P ∧ adjacent a1 a2 side :=
sorry

end pentagon_angles_adjacent_l135_135728


namespace dogs_grouping_l135_135260

theorem dogs_grouping (dogs : Finset α) (fluffy nipper : α) :
  dogs.card = 12 ∧ fluffy ∈ dogs ∧ nipper ∈ dogs →
  ∃ g1 g2 g3 : Finset α,
    (g1.card = 4 ∧ g2.card = 5 ∧ g3.card = 3) ∧
    (fluffy ∈ g1) ∧ (nipper ∈ g2) ∧
    (g1 ∪ g2 ∪ g3 = dogs) ∧ (g1 ∩ g2 = ∅) ∧ (g1 ∩ g3 = ∅) ∧ (g2 ∩ g3 = ∅) ∧
    (∃ n : ℕ, n = 4200) :=
by
  sorry

end dogs_grouping_l135_135260


namespace ratio_invariance_l135_135298

/-!
# Invariance of a Certain Ratio

## Problem Statement

Through point \( S \), lines \( a, b, c, \) and \( d \) are drawn. Line \( l \) intersects them at points \( A, B, C, \) and \( D \). Prove that the quantity \( \frac{AC \cdot BD}{BC \cdot AD} \) does not depend on the choice of line \( l \).
-/

-- Definitions
variables {S A B C D : Type} 
variables {a b c d : line}
variables [PointOnLine S a] [PointOnLine S b] [PointOnLine S c] [PointOnLine S d]
variables {l : line}
variables [Intersect l a A] [Intersect l b B] [Intersect l c C] [Intersect l d D]

-- Theorem statement
theorem ratio_invariance (S A B C D : Point) (a b c d l : Line) 
  (haS : online S a) (hbS : online S b) (hcS : online S c) (hdS : online S d)
  (haA : online A a) (hbB : online B b) (hcC : online C c) (hdD : online D d)
  (hlA : online A l) (hlB : online B l) (hlC : online C l) (hlD : online D l):
  (AC BD BC AD : ℝ) : 
  AC * BD ≠ 0 -> BC * AD ≠ 0 -> 
  (AC * BD) / (BC * AD) = (sin(α) * sin (γ)) / (sin (α + β) * sin (β + γ)) :=
begin
  sorry
end

end ratio_invariance_l135_135298


namespace avg_speed_is_40_l135_135683

noncomputable def average_speed (x : ℝ) : ℝ :=
  let time1 := x / 40
  let time2 := 2 * x / 20
  let total_time := time1 + time2
  let total_distance := 5 * x
  total_distance / total_time

theorem avg_speed_is_40 (x : ℝ) (hx : x > 0) :
  average_speed x = 40 := by
  sorry

end avg_speed_is_40_l135_135683


namespace combined_shape_area_l135_135406

noncomputable def area_of_combined_shape (base_bc : ℝ) (height_a_to_bc : ℝ) (ce_length : ℝ) : ℝ :=
  let area_parallelogram := base_bc * height_a_to_bc
  let ac_length := Real.sqrt (base_bc ^ 2 + height_a_to_bc ^ 2)
  let area_triangle := 0.5 * ac_length * ce_length
  area_parallelogram + area_triangle

theorem combined_shape_area (base_bc : ℝ) (height_a_to_bc : ℝ) (ce_length : ℝ) :
  base_bc = 6 → height_a_to_bc = 3 → ce_length = 2 → area_of_combined_shape base_bc height_a_to_bc ce_length = 18 + 3 * Real.sqrt 5 :=
by
  intros h_base h_height h_ce
  rw [h_base, h_height, h_ce]
  unfold area_of_combined_shape
  norm_num
  repeat { rw Real.sqrt_mul_self }
  exact sqrt_pos.2 (by norm_num : 45 > 0)
  sorry

end combined_shape_area_l135_135406


namespace factory_exceeded_production_plan_l135_135554

noncomputable def January_production (x : ℝ) := 1.05 * (x / 2)
noncomputable def February_production (x : ℝ) := 1.04 * January_production x

theorem factory_exceeded_production_plan (x : ℝ) :
  1.071 * x = (January_production x + February_production x) :=
by
  have jan_prod : January_production x = 1.05 * (x / 2) := rfl
  have feb_prod : February_production x = 1.04 * (1.05 * (x / 2)) := rfl
  have total_prod : January_production x + February_production x = 1.05 * (x / 2) + 1.04 * 1.05 * (x / 2) := by rw [jan_prod, feb_prod]
  have total_prod_simplified : January_production x + February_production x = (1.05 + 1.092) * (x / 2) := by
    rw [total_prod]
    norm_num
  have total_prod_final : January_production x + February_production x = 1.071 * x := by
    rw [total_prod_simplified]
    norm_num
  exact total_prod_final
  sorry

end factory_exceeded_production_plan_l135_135554


namespace find_h_k_l135_135586

variable (G : Type) [Group G] [Fintype G] (K : Set G)

theorem find_h_k (hK : K.card > Fintype.card G / 2) (g : G) :
  ∃ h k ∈ K, g = h * k :=
sorry

end find_h_k_l135_135586


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135316

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 + d2 + d3

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, is_three_digit n ∧ is_multiple_of_9 n ∧ digit_sum n = 27 ∧
  ∀ m : ℕ, is_three_digit m ∧ is_multiple_of_9 m ∧ digit_sum m = 27 → m ≤ n := 
by 
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135316


namespace opera_house_earnings_l135_135732

theorem opera_house_earnings :
  let rows := 150
  let seats_per_row := 10
  let ticket_cost := 10
  let total_seats := rows * seats_per_row
  let seats_not_taken := total_seats * 20 / 100
  let seats_taken := total_seats - seats_not_taken
  let total_earnings := ticket_cost * seats_taken
  total_earnings = 12000 := by
sorry

end opera_house_earnings_l135_135732


namespace percentage_less_than_y_is_70_percent_less_than_z_l135_135550

variable {x y z : ℝ}

theorem percentage_less_than (h1 : x = 1.20 * y) (h2 : x = 0.36 * z) : y = 0.3 * z :=
by
  sorry

theorem y_is_70_percent_less_than_z (h : y = 0.3 * z) : (1 - y / z) * 100 = 70 :=
by
  sorry

end percentage_less_than_y_is_70_percent_less_than_z_l135_135550


namespace distance_from_center_to_line_l135_135274

noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

def center_of_circle_c1 : ℝ × ℝ := (1, 0)

def line1 : ℝ × ℝ × ℝ := (√3 / 3, -1, 0)

theorem distance_from_center_to_line :
  distance_point_to_line (center_of_circle_c1.1) (center_of_circle_c1.2) (line1.1) (line1.2) (line1.3) = 1 / 2 :=
by
  sorry

end distance_from_center_to_line_l135_135274


namespace bowling_average_before_last_match_l135_135007

noncomputable def original_bowling_average (A : ℝ) : Prop :=
  let new_wickets := 175 + 8 in
  let new_total_runs := 175 * A + 26 in
  let new_average := new_total_runs / new_wickets in
  new_average = A - 0.4

theorem bowling_average_before_last_match : original_bowling_average 12.4 :=
by
  unfold original_bowling_average
  sorry

end bowling_average_before_last_match_l135_135007


namespace octahedron_has_eulerian_circuit_cube_has_no_eulerian_circuit_l135_135704

-- Part (a) - Octahedron
/- 
A connected graph representing an octahedron. 
Each vertex has a degree of 4, making the graph Eulerian.
-/
theorem octahedron_has_eulerian_circuit : 
  ∃ circuit : List (ℕ × ℕ), 
    (∀ (u v : ℕ), List.elem (u, v) circuit ↔ List.elem (v, u) circuit) ∧
    (∃ start, ∀ v ∈ circuit, v = start) :=
sorry

-- Part (b) - Cube
/- 
A connected graph representing a cube.
Each vertex has a degree of 3, making it impossible for the graph to be Eulerian.
-/
theorem cube_has_no_eulerian_circuit : 
  ¬ ∃ (circuit : List (ℕ × ℕ)), 
    (∀ (u v : ℕ), List.elem (u, v) circuit ↔ List.elem (v, u) circuit) ∧
    (∃ start, ∀ v ∈ circuit, v = start) :=
sorry

end octahedron_has_eulerian_circuit_cube_has_no_eulerian_circuit_l135_135704


namespace tan_double_sum_angles_l135_135572

variable {A B : Real}
variables (h1 : sin A = 3 / 5) (h2 : tan B = 2) (hTriangle : A + B + (π - (A + B)) = π)

theorem tan_double_sum_angles : tan (2 * (A + B)) = 44 / 117 :=
by
  sorry

end tan_double_sum_angles_l135_135572


namespace part1_part2_l135_135522

-- Definitions of the sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 7 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Part 1: Intersection of A and B when m = 5
theorem part1 (m := 5) : (A ∩ B m) = {x | 6 ≤ x ∧ x ≤ 7} :=
  sorry

-- Part 2: Range of m where x ∈ A is necessary but not sufficient for x ∈ B
theorem part2 : (∀ x, x ∈ A → x ∈ B) → (2 ≤ m ∧ m ≤ 4) :=
  sorry

end part1_part2_l135_135522


namespace point_not_in_second_quadrant_l135_135892

theorem point_not_in_second_quadrant (n : ℝ) : ¬ ((n + 1 < 0) ∧ (2 * n - 1 > 0)) :=
by
  intro h
  cases h with h1 h2
  -- Proof would go here
  sorry

end point_not_in_second_quadrant_l135_135892


namespace mountain_distance_l135_135665

noncomputable def distance_to_top := 1550

variables {A_up_speed A_down_speed B_up_speed B_down_speed : ℕ}
variables {distance_meet : ℕ}
variables {halfway_distance : ℕ}

-- Conditions
axiom A_B_start_same_time_same_path : True
axiom descending_speeds_triple_ascending : A_down_speed = 3 * A_up_speed ∧ B_down_speed = 3 * B_up_speed
axiom meet_150_meters_from_top : distance_meet = 150
axiom A_returns_B_halfway : halfway_distance = distance_to_top / 2

-- Theorem to prove
theorem mountain_distance : distance_to_top = 1550 :=
begin
  sorry
end

end mountain_distance_l135_135665


namespace radical_multiplication_l135_135305

noncomputable def root4 (x : ℝ) : ℝ := x ^ (1/4)
noncomputable def root3 (x : ℝ) : ℝ := x ^ (1/3)
noncomputable def root2 (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_multiplication : root4 256 * root3 8 * root2 16 = 32 := by
  sorry

end radical_multiplication_l135_135305


namespace positive_integers_equality_l135_135091

theorem positive_integers_equality :
  ∃ (p : Finset ℕ), p.card = 500 ∧ 
  (∀ n ∈ p, ∀ t : ℝ, (complex.cos t + complex.sin t * complex.i) ^ n = complex.cos (n * t) + complex.sin (n * t) * complex.i) := 
sorry

end positive_integers_equality_l135_135091


namespace no_solution_absval_equation_l135_135770

theorem no_solution_absval_equation (x : ℝ) : ¬ (|2*x - 5| = 3*x + 1) :=
by
  sorry

end no_solution_absval_equation_l135_135770


namespace find_polynomial_expansion_sum_l135_135477

noncomputable def polynomial_expansion_sum (a : ℤ) (a_k : ℕ → ℤ) : ℤ :=
  a + a_k 1 + a_k 2 + a_k 3 + a_k 4 + a_k 5 + a_k 6 + a_k 7 + a_k 8 + a_k 9 + a_k 10 + a_k 11

theorem find_polynomial_expansion_sum :
  (∀ x : ℝ, (x^2 + 1) * (2 * x + 1)^9 = 
    ∑ k in (Finset.range 12), (a_k k : ℝ) * (x + 2)^k) →
  polynomial_expansion_sum a a_k = -2 :=
by
  sorry -- proof is not required

end find_polynomial_expansion_sum_l135_135477


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135318

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((nat.digits 10 n).sum = 27) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ ((nat.digits 10 m).sum = 27) → m ≤ n) :=
begin
  use 999,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intro m,
    intro hm,
    cases hm,
    cases hm_left,
    cases hm_left_left,
    cases hm_left_right,
    cases hm_right,
    sorry
  },
sorry,
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135318


namespace problem_part1_problem_part2_l135_135943

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l135_135943


namespace find_C_prove_relation_l135_135959

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A), and A = 2B,
prove that C = 5/8 * π. -/
theorem find_C
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  C = ⅝ * Real.pi :=
sorry

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A),
prove that 2 * a ^ 2 = b ^ 2 + c ^ 2. -/
theorem prove_relation
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
sorry

end find_C_prove_relation_l135_135959


namespace max_value_of_y_l135_135171

noncomputable def max_value_y (x : ℝ) : ℝ := -x^2 + 2 * x + 3

theorem max_value_of_y : ∃ x : ℝ, max_value_y x = 4 :=
by
  exists 1
  simp
  sorry

end max_value_of_y_l135_135171


namespace hyperbola_eccentricity_l135_135996

theorem hyperbola_eccentricity (a b e c x0 y0 : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : e^2 = 1 + b^2/a^2) (h4 : x0 = c + sqrt 2 / 2 * c) (h5 : y0 = sqrt 2 / 2 * c) 
  (h6 : x0/a^2 = y0/b^2) : e = sqrt (sqrt 2) := 
sorry

end hyperbola_eccentricity_l135_135996


namespace total_handshakes_l135_135425

-- Definitions and conditions
def num_dwarves := 25
def num_elves := 18

def handshakes_among_dwarves : ℕ := num_dwarves * (num_dwarves - 1) / 2
def handshakes_between_dwarves_and_elves : ℕ := num_elves * num_dwarves

-- Total number of handshakes
theorem total_handshakes : handshakes_among_dwarves + handshakes_between_dwarves_and_elves = 750 := by 
  sorry

end total_handshakes_l135_135425


namespace complex_number_in_second_quadrant_l135_135102

open Complex

-- Given definition in conditions
def i : ℂ := Complex.I

-- Goal statement
theorem complex_number_in_second_quadrant : 
  let z := (1 + 2 * I) / (1 + 2 * I^3)
  in z.re < 0 ∧ z.im > 0 := 
by
  have h1 : I^3 = -I := by simp [pow_succ, pow_two, I_mul_I]
  have h2 : (1 + 2 * I^3) = (1 - 2 * I) := by rw [h1]; simp
  let z := (1 + 2 * I) / (1 - 2 * I)
  have h3 : (z = (-3 / 5) + (4 / 5) * I) := by
    calc
      (1 + 2 * I) / (1 - 2 * I)
          = (1 + 2 * I) * (1 - 2 * I).conj / ((1 - 2 * I) * (1 - 2 * I).conj) : by rw [mul_conj]
      ... = ((1 + 2 * I) * (1 + 2 * I)) / ((1 - 2 * I) * (1 + 2 * I)) : by simp
      ... = ((1 + 4 * I - 4) / 5) : by calc
        ... = (1 - 4 + 4 * I) / 5 : by ring
        ... = (-3 / 5 + (4 / 5) * I) : by field_simp
  rw h3
  exact ⟨by norm_num, by norm_num⟩
  sorry

end complex_number_in_second_quadrant_l135_135102


namespace sum_of_special_two_digit_integers_l135_135675

theorem sum_of_special_two_digit_integers : 
  let eligible (n : ℕ) := 
    10 ≤ n ∧ n ≤ 99 ∧ 
    ∃ a b, n = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 
          (a * b > 0 ∧ a * b ∣ n)
  in
  (∑ n in finset.filter eligible (finset.range 100), n) = 1575 := 
by sorry

end sum_of_special_two_digit_integers_l135_135675


namespace equal_angles_l135_135398

variables {A B C I C1 A1 B1 X Y Z : Type*}

-- Conditions
axiom incircle_touches_sides :
  (tangent_circle : A → B → C → I → A1 → B1 → C1 → (Circle A B C)) -- The circle touches sides AB, BC, CA respectively at C1, A1, B1.

axiom lines_intersect :
  (AI : Line A I) → 
  (CI : Line C I) → 
  (B1I : Line I B1) → 
  (A1C1 : Line A1 C1) →
  X = AI.intersection A1C1 ∧
  Y = CI.intersection A1C1 ∧
  Z = B1I.intersection A1C1

-- Statement to be proved
theorem equal_angles : 
  ∀ (incircle_touches_sides : A → B → C → I → A1 → B1 → C1 → (Circle A B C))
    (lines_intersect : X = AI ∧ Y = CI ∧ Z = B1I) :
  ∠ Y B1 Z = ∠ X B1 Z :=
sorry

end equal_angles_l135_135398


namespace part1_part2_l135_135139

noncomputable def f (a x : ℝ) : ℝ := (a / 2) * x * x - (a - 2) * x - 2 * x * Real.log x
noncomputable def f' (a x : ℝ) : ℝ := a * x - a - 2 * Real.log x

theorem part1 (a : ℝ) : (∀ x > 0, f' a x ≥ 0) ↔ a = 2 :=
sorry

theorem part2 (a x1 x2 : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : f' a x1 = 0) (h4 : f' a x2 = 0) (h5 : x1 < x2) : 
  x2 - x1 > 4 / a - 2 :=
sorry

end part1_part2_l135_135139


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135332

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ (100 ≤ n ∧ n < 1000) ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) :=
by {
  sorry
}

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135332


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135365

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ 100 ≤ n ∧ n < 1000 ∧ (9 ∣ n) ∧ (∑ digit in n.digits, digit = 27) :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135365


namespace rhombus_area_correct_l135_135285

noncomputable def rhombus_area (side d1 : ℝ) (angle_deg : ℝ) : ℝ :=
  let angle_rad := angle_deg * (Real.pi / 180) in -- Convert the angle to radians
  let d2 := 2 * side * Real.cos angle_rad in -- Calculate the other diagonal
  (d1 * d2) / 2 -- Calculate the area using the diagonals

theorem rhombus_area_correct :
  rhombus_area 20 16 35 ≈ 262.144 :=
by
  unfold rhombus_area
  simp
  sorry -- Skipping the detailed trigonometric calculation

end rhombus_area_correct_l135_135285


namespace sue_payment_correct_l135_135622

noncomputable def sue_payment := 2100 * (3 / 7)

theorem sue_payment_correct : sue_payment = 900 := by
  sorry

end sue_payment_correct_l135_135622


namespace non_black_cows_l135_135245

-- Define the main problem conditions
def total_cows : ℕ := 18
def black_cows : ℕ := (total_cows / 2) + 5

-- Statement to prove the number of non-black cows
theorem non_black_cows :
  total_cows - black_cows = 4 :=
by
  sorry

end non_black_cows_l135_135245


namespace CaO_H2O_to_CaOH2_l135_135775

theorem CaO_H2O_to_CaOH2 (n_CaO n_H2O: ℕ) (h1: n_CaO = 1) (h2: n_H2O = 1) : 
  ∃ n_CaOH2 : ℕ, n_CaOH2 = 1 :=
by 
  -- The conditions are directly given
  use 1
  -- Correct solution would follow by chemical equation balance
  sorry

end CaO_H2O_to_CaOH2_l135_135775


namespace count_groupings_l135_135263

theorem count_groupings (dogs : Finset ℕ) (Fluffy Nipper : ℕ) (h : Fluffy ≠ Nipper) (h_count : dogs.card = 12) :
  ∃ (g1 g2 g3 : Finset ℕ), 
    g1.card = 4 ∧ g2.card = 5 ∧ g3.card = 3 ∧
    Fluffy ∈ g1 ∧ Nipper ∈ g2 ∧
    (∀ x, x ∈ g1 ∨ x ∈ g2 ∨ x ∈ g3) ∧
     ∑ y in dogs, 1 = 12 ∧
     (∏ (a ∈ Finset.choose 10 3), ∏ (b ∈ Finset.choose 7 4), 1) = 4200
:= sorry

end count_groupings_l135_135263


namespace parabola_equation_l135_135299

-- Definitions:
variables {P Q : Point}

structure Point :=
  (x : ℝ)
  (y : ℝ)

def parabola (p : ℝ) (P Q : Point) : Prop :=
  P.y ^ 2 = 2 * p * P.x ∧ Q.y ^ 2 = 2 * p * Q.x

def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- The Lean 4 statement:
theorem parabola_equation
  (p : ℝ)
  (P Q : Point)
  (hPQparabola : parabola p P Q)
  (hMidpoint3 : (midpoint P Q).x = 3)
  (hPQdistance : distance P Q = 10) :
  p = 4 /\
  (∀ x y, y ^ 2 = 2 * p * x -> y ^ 2 = 8 * x) :=
by
  sorry

end parabola_equation_l135_135299


namespace length_of_YZ_l135_135894

theorem length_of_YZ (h1 : ∠ COB = 90°) (h2 : perpendicular OZ CB) (h3 : OC = 15) 
                     (h4 : OB = 15) : YZ = (30 - 15 * real.sqrt 2) / 2 :=
by
  sorry

end length_of_YZ_l135_135894


namespace percent_value_in_quarters_l135_135376

def nickel_value : ℕ := 5
def quarter_value : ℕ := 25
def num_nickels : ℕ := 80
def num_quarters : ℕ := 40

def value_in_nickels : ℕ := num_nickels * nickel_value
def value_in_quarters : ℕ := num_quarters * quarter_value
def total_value : ℕ := value_in_nickels + value_in_quarters

theorem percent_value_in_quarters :
  (value_in_quarters : ℚ) / total_value = 5 / 7 :=
by
  sorry

end percent_value_in_quarters_l135_135376


namespace bridge_length_example_l135_135721

noncomputable def length_of_bridge (train_length : ℝ) (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time
  total_distance - train_length

theorem bridge_length_example :
  length_of_bridge 100 34.997200223982084 36 = 249.97200223982084 :=
by
  unfold length_of_bridge
  have speed_conversion : 36 * 1000 / 3600 = 10 := by norm_num
  rw speed_conversion
  norm_num
  done

end bridge_length_example_l135_135721


namespace part1_part2_l135_135965

variable {A B C a b c : ℝ}

theorem part1 (h₁ : A = 2 * B) (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : C = 5 / 8 * π :=
  sorry

theorem part2 (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
  sorry

end part1_part2_l135_135965


namespace part1_C_value_part2_triangle_equality_l135_135979

noncomputable theory

variables (a b c : ℝ) (A B C : ℝ)
variables (h1 : A + B + C = Real.pi) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) (h3 : A = 2 * B)

-- Part 1: Proving that C = 5π/8 given the conditions
theorem part1_C_value :
  C = 5 * Real.pi / 8 :=
begin
  sorry
end

-- Part 2: Proving that 2a^2 = b^2 + c^2 given the conditions
theorem part2_triangle_equality :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
begin
  sorry
end

end part1_C_value_part2_triangle_equality_l135_135979


namespace area_of_region_sec_csc_l135_135463

theorem area_of_region_sec_csc (θ : ℝ) :
    let r1 := sec θ - 1
    let r2 := csc θ - 1
    (0 ≤ θ ∧ θ ≤ π/2) →
    let x_bound := 1 - cos θ
    let y_bound := 1 - sin θ
    (0 ≤ x_bound ∧ x_bound ≤ 2) →
    (0 ≤ y_bound ∧ y_bound ≤ 2) →
    (x_bound * y_bound) = 4 :=
by
  sorry

end area_of_region_sec_csc_l135_135463


namespace dogs_grouping_l135_135258

theorem dogs_grouping (dogs : Finset α) (fluffy nipper : α) :
  dogs.card = 12 ∧ fluffy ∈ dogs ∧ nipper ∈ dogs →
  ∃ g1 g2 g3 : Finset α,
    (g1.card = 4 ∧ g2.card = 5 ∧ g3.card = 3) ∧
    (fluffy ∈ g1) ∧ (nipper ∈ g2) ∧
    (g1 ∪ g2 ∪ g3 = dogs) ∧ (g1 ∩ g2 = ∅) ∧ (g1 ∩ g3 = ∅) ∧ (g2 ∩ g3 = ∅) ∧
    (∃ n : ℕ, n = 4200) :=
by
  sorry

end dogs_grouping_l135_135258


namespace geometric_arithmetic_sequence_difference_l135_135181

theorem geometric_arithmetic_sequence_difference
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (q : ℝ)
  (hq : q > 0)
  (ha1 : a 1 = 2)
  (ha2 : a 2 = a 1 * q)
  (ha4 : a 4 = a 1 * q ^ 3)
  (ha5 : a 5 = a 1 * q ^ 4)
  (harith : 2 * (a 4 + 2 * a 5) = 2 * a 2 + (a 4 + 2 * a 5))
  (hS : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  S 10 - S 4 = 2016 :=
by
  sorry

end geometric_arithmetic_sequence_difference_l135_135181


namespace shortest_distance_to_circle_origin_l135_135368

theorem shortest_distance_to_circle_origin : 
  let origin := (0, 0)
  ∃ (center : ℝ × ℝ) (r : ℝ), -- center and radius
    circle_eq : (∀ x y : ℝ, x^2 - 30 * x + y^2 - 8 * y + 325 = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = r^2) ∧
    euclidean_dist origin center - r = Real.sqrt 241 - 2 * Real.sqrt 21
  :=
begin
  -- Definitions and conditions:
  let origin := (0, 0),
  let circle_eq := (λ x y : ℝ, x^2 - 30 * x + y^2 - 8 * y + 325 = 0),
  -- Proof goes here. 
  sorry
end

end shortest_distance_to_circle_origin_l135_135368


namespace equation_has_two_solutions_l135_135850

theorem equation_has_two_solutions : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ∀ x : ℝ, ¬ ( |x - 1| = |x - 2| + |x - 3| ) ↔ (x ≠ x₁ ∧ x ≠ x₂) :=
sorry

end equation_has_two_solutions_l135_135850


namespace max_value_of_expression_l135_135992

theorem max_value_of_expression :
  ∀ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 → 
  a^2 * b^2 * c^2 + (2 - a)^2 * (2 - b)^2 * (2 - c)^2 ≤ 64 :=
by sorry

end max_value_of_expression_l135_135992


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135333

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ (100 ≤ n ∧ n < 1000) ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) :=
by {
  sorry
}

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135333


namespace exponent_2_prime_factorization_30_exponent_5_prime_factorization_30_l135_135197

open Nat

theorem exponent_2_prime_factorization_30! :
  nat.factorial_prime_pow 30 2 = 26 := by
  sorry

theorem exponent_5_prime_factorization_30! :
  nat.factorial_prime_pow 30 5 = 7 := by
  sorry

end exponent_2_prime_factorization_30_exponent_5_prime_factorization_30_l135_135197


namespace problem_statement_l135_135238

namespace MathProof

def p : Prop := (2 + 4 = 7)
def q : Prop := ∀ x : ℝ, x = 1 → x^2 ≠ 1

theorem problem_statement : ¬ (p ∧ q) ∧ (p ∨ q) :=
by
  -- To be filled in
  sorry

end MathProof

end problem_statement_l135_135238


namespace sue_payment_correct_l135_135621

noncomputable def sue_payment := 2100 * (3 / 7)

theorem sue_payment_correct : sue_payment = 900 := by
  sorry

end sue_payment_correct_l135_135621


namespace maximal_number_of_positive_integers_l135_135588

open BigOperators

variable (x : Fin 2024 → ℕ)
variable (h_increasing : ∀ i j : Fin 2024, i < j → x i < x j)
variable (h_positive : ∀ i : Fin 2024, x i > 0)

noncomputable def p (i : Fin 2024) : ℝ :=
  ∏ k in Finset.range (i + 1), (x k - 1 / (x k : ℝ))

theorem maximal_number_of_positive_integers :
  (Finset.range 2024).filter (λ i, (p x i).denom = 1).card = 1012 := sorry

end maximal_number_of_positive_integers_l135_135588


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135323

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((nat.digits 10 n).sum = 27) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ ((nat.digits 10 m).sum = 27) → m ≤ n) :=
begin
  use 999,
  split,
  { linarith },
  split,
  { linarith },
  split,
  { norm_num },
  split,
  { norm_num },
  {
    intro m,
    intro hm,
    cases hm,
    cases hm_left,
    cases hm_left_left,
    cases hm_left_right,
    cases hm_right,
    sorry
  },
sorry,
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135323


namespace x_intercept_rotation_30_degrees_eq_l135_135998

noncomputable def x_intercept_new_line (x0 y0 : ℝ) (θ : ℝ) (a b c : ℝ) : ℝ :=
  let m := a / b
  let m' := (m + θ.tan) / (1 - m * θ.tan)
  let x_intercept := x0 - (y0 * (b - m * c)) / (m' * (b - m * c) - a)
  x_intercept

theorem x_intercept_rotation_30_degrees_eq :
  x_intercept_new_line 7 4 (Real.pi / 6) 4 (-7) 28 = 7 - (4 * (7 * Real.sqrt 3 - 4) / (4 * Real.sqrt 3 + 7)) :=
by 
  -- detailed math proof goes here 
  sorry

end x_intercept_rotation_30_degrees_eq_l135_135998


namespace count_groupings_l135_135262

theorem count_groupings (dogs : Finset ℕ) (Fluffy Nipper : ℕ) (h : Fluffy ≠ Nipper) (h_count : dogs.card = 12) :
  ∃ (g1 g2 g3 : Finset ℕ), 
    g1.card = 4 ∧ g2.card = 5 ∧ g3.card = 3 ∧
    Fluffy ∈ g1 ∧ Nipper ∈ g2 ∧
    (∀ x, x ∈ g1 ∨ x ∈ g2 ∨ x ∈ g3) ∧
     ∑ y in dogs, 1 = 12 ∧
     (∏ (a ∈ Finset.choose 10 3), ∏ (b ∈ Finset.choose 7 4), 1) = 4200
:= sorry

end count_groupings_l135_135262


namespace triangle_sides_angles_l135_135973

theorem triangle_sides_angles (a b c A B C : ℝ) (h1: A = 2 * B) 
  (h2: sin C * sin (A - B) = sin B * sin (C - A)) 
  (h3: A + B + C = π) :
  (C = 5 * π / 8) ∧ (2 * a^2 = b^2 + c^2) :=
by
  -- Proof omitted
  sorry

end triangle_sides_angles_l135_135973


namespace maximum_simultaneous_glows_l135_135664

def first_light_interval : ℕ := 21
def second_light_interval : ℕ := 31
def start_time_in_seconds : ℕ := 7078   -- 1:57:58 AM in seconds since midnight
def end_time_in_seconds : ℕ := 12047    -- 3:20:47 AM in seconds since midnight
def time_duration : ℕ := end_time_in_seconds - start_time_in_seconds
def lcm_interval (a b: ℕ) : ℕ := Nat.lcm a b
def lcm_of_intervals : ℕ := lcm_interval first_light_interval second_light_interval

theorem maximum_simultaneous_glows : (time_duration / lcm_of_intervals) = 7 := by
  have lcm_val := lcm_of_intervals
  calc time_duration / lcm_val = 7
  sorry

end maximum_simultaneous_glows_l135_135664


namespace seq_is_arithmetic_sum_bn_l135_135141

-- Part (I)
theorem seq_is_arithmetic
  (m : ℝ) (a : ℕ → ℝ)
  (hm_pos : 0 < m) (hm_ne_one : m ≠ 1)
  (h_geo_seq : ∀ n : ℕ, f (a (n + 1)) = m * f (a n))
  (h_f_def : ∀ x : ℝ, f x = m^x) :
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

-- Part (II)
theorem sum_bn
  (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ)
  (h_an : ∀ n, a n = n + 1)
  (h_bn : ∀ n, b n = a n * f (a n))
  (h_S : ∀ n, S n = ∑ i in range n, b i)
  (h_f_def : ∀ x : ℝ, f x = 2^x) :
  S n = 2^(n + 2) * n :=
sorry

end seq_is_arithmetic_sum_bn_l135_135141


namespace number_of_valid_permutations_l135_135636

theorem number_of_valid_permutations :
  let digits := {1, 2, 3, 4, 5, 6} in
  let valid_permutation (p : List Nat) : Prop := 
    p.length = 6 ∧ p.nodup ∧ 
    (∀ i j, i < j → p[i] = 1 → p[j] = 2) ∧
    (∀ i j, i < j → p[i] = 3 → p[j] = 4) in
  ∃ l : List (List Nat), l.length = 180 ∧ 
    (∀ p, p ∈ l → valid_permutation p) :=
by 
  sorry

end number_of_valid_permutations_l135_135636


namespace square_difference_identity_l135_135065

theorem square_difference_identity (a b : ℕ) : (a - b)^2 = a^2 - 2 * a * b + b^2 :=
  by sorry

lemma evaluate_expression : (101 - 2)^2 = 9801 :=
  by
    have h := square_difference_identity 101 2
    exact h

end square_difference_identity_l135_135065


namespace number_of_bijections_f_satisfying_f_12_eq_x_l135_135690

theorem number_of_bijections_f_satisfying_f_12_eq_x :
  let A := {1, 2, 3, 4, 5, 6}
  let is_bijection (f : ℕ → ℕ) : Prop := ∀ a b ∈ A, f a = f b → a = b ∧ ∀ y ∈ A, ∃ x ∈ A, f x = y
  let f_comp (f : ℕ → ℕ) (n : ℕ) (x : ℕ) : ℕ := 
    if n = 0 then x 
    else f (f_comp f (n - 1) x)
  in 
  (A.finite : Finite A) ∧
  ∀ f : ℕ → ℕ, (is_bijection f) →(∀ x ∈ A, f_comp f 12 x = x) ↔ ∃ permutations f, f = 576 := -- statement of the problem 
sorry -- proof goes here

end number_of_bijections_f_satisfying_f_12_eq_x_l135_135690


namespace prob_B_l135_135382

variables {Ω : Type} [MeasurableSpace Ω] {P : MeasureTheory.ProbabilityMeasure Ω}
variables {A B : Set Ω}

-- Conditions:
def prob_A : P[A] = 0.4 := sorry
def prob_A_and_B : P[A ∩ B] = 0.25 := sorry
def prob_A_or_B : P[A ∪ B] = 0.6 := sorry

-- The proof goal:
theorem prob_B :
  P[B] = 0.45 :=
by
  -- The proof would go here.
  sorry

end prob_B_l135_135382


namespace integral_one_over_x_l135_135457

theorem integral_one_over_x:
  ∫ x in (1 : ℝ)..(Real.exp 1), 1 / x = 1 := 
by 
  sorry

end integral_one_over_x_l135_135457


namespace map_distance_l135_135230

theorem map_distance {
  (map_distance_scale: ℝ)
  (real_distance_scale: ℝ)
  (actual_real_distance: ℝ)
} :
  (map_distance_scale / 0.6 = real_distance_scale / 6.6) →
  (map_distance_scale / real_distance_scale = 80.5 / 11) →
  (actual_real_distance = 885.5) →
  (map_distance = actual_real_distance / (real_distance_scale / 0.6)) →
  map_distance = 80.5 :=
by
  intros h_scale_ratio h_distance_on_map h_real_distance h_map_distance
  rw [h_scale_ratio, h_distance_on_map, h_real_distance, h_map_distance]
  sorry

end map_distance_l135_135230


namespace area_of_triangle_l135_135507

theorem area_of_triangle (A B C : ℝ) (a c : ℝ) (d B_value: ℝ) (h1 : A + B + C = 180) 
                         (h2 : A = B - d) (h3 : C = B + d) (h4 : a = 4) (h5 : c = 3)
                         (h6 : B = 60) :
  (1 / 2) * a * c * Real.sin (B * Real.pi / 180) = 3 * Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l135_135507


namespace relationship_l135_135855

-- Define the conditions
variables (a b c d : ℝ) (x y z q : ℝ)
hypothesis h1 : a^x = c^q
hypothesis h2 : c^y = a^z
hypothesis h3 : a^x = b
hypothesis h4 : c^y = d

-- Prove the main statement
theorem relationship (a b c d x y z q : ℝ) 
  (h1 : a^x = c^q) (h2 : c^y = a^z) (h3 : a^x = b) (h4 : c^y = d) : 
  x * y = q * z := 
sorry

end relationship_l135_135855


namespace count_three_digit_numbers_div_by_5_l135_135787

theorem count_three_digit_numbers_div_by_5 : 
  let digits := {0, 1, 2, 3, 4, 5}
  ∃ count : ℕ, count = 36 ∧ (∀ n ∈ digits,
    (n ≠ 0 → n ≠ 5 → count_f (λ x, x ∈ digits ∧ x ∉ {n}) = 20) ∧ 
    (n = 0 ∨ n = 5) → 
    count_f (λ x, x ∈ digits ∧ x ≠ n ∧ (n = 0 ∨ x ≠ 0 ∧ x ≠ 5)) = 16)∧
  (count = 20 + 16) :=
  sorry

end count_three_digit_numbers_div_by_5_l135_135787


namespace f_is_odd_l135_135638

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2^x - 2^(-x)

-- Define the proposition that f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x :=
by
  assume x
  sorry

end f_is_odd_l135_135638


namespace area_of_trapezoid_l135_135201

theorem area_of_trapezoid 
  (AB CD AD BC AC : ℝ)
  (h1 : AD = CD)
  (h2 : AB = 2 * CD)
  (h3 : BC = 24)
  (h4 : AC = 10) :
  let area := 180 in
  1 / 2 * (AB + CD) * BD = area :=
by
  sorry

end area_of_trapezoid_l135_135201


namespace g_periodic_6_f_2008_l135_135990

variable (f : ℝ → ℝ)

-- Given conditions
axiom f_x3_le_f_x_plus_3 : ∀ x : ℝ, f(x + 3) ≤ f(x) + 3
axiom f_x2_ge_f_x_plus_2 : ∀ x : ℝ, f(x + 2) ≥ f(x) + 2

-- Define g(x)
def g (x : ℝ) := f(x) - x

-- Question 1: Prove that g(x) is periodic with period 6
theorem g_periodic_6 : ∀ x : ℝ, g(f)(x + 6) = g(f)(x) := by
  sorry

-- Question 2: Given f(994) = 992, find f(2008)
axiom f_994 : f(994) = 992

theorem f_2008 : f(2008) = 2006 := by
  sorry

end g_periodic_6_f_2008_l135_135990


namespace find_line_equation_l135_135464

noncomputable def line_equation := 12 * x + 8 * y - 15 = 0

def eq_condition_one := 3 * x + 2 * y - 6 = 0
def eq_condition_two := 6 * x + 4 * y - 3 = 0

theorem find_line_equation (x y : ℝ) :
  ∃ c : ℝ, ∀ (l : ℝ → ℝ → Prop), l = (λ x y, 12 * x + 8 * y - 15 = 0) :=
sorry

end find_line_equation_l135_135464


namespace calculate_expression_l135_135367

theorem calculate_expression : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end calculate_expression_l135_135367


namespace eight_points_in_circle_l135_135204

theorem eight_points_in_circle :
  ∀ (P : Fin 8 → ℝ × ℝ), 
  (∀ i, (P i).1^2 + (P i).2^2 ≤ 1) → 
  ∃ (i j : Fin 8), i ≠ j ∧ ((P i).1 - (P j).1)^2 + ((P i).2 - (P j).2)^2 < 1 :=
by
  sorry

end eight_points_in_circle_l135_135204


namespace gilda_final_percentage_l135_135788

def Gilda_marble_percentage (x : ℝ) : ℝ :=
  let after_cousin := 1.30 * x
  let after_pedro := after_cousin - 0.25 * after_cousin
  let after_ebony := after_pedro - 0.15 * after_pedro
  let after_jimmy := after_ebony - 0.30 * after_ebony
  (after_jimmy / after_cousin) * 100

theorem gilda_final_percentage (x : ℝ) (hx : x > 0) : Gilda_marble_percentage x = 44.625 :=
  by
  sorry

end gilda_final_percentage_l135_135788


namespace part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135931

-- Definitions for the conditions in the problem
variable {A B C a b c : ℝ}

-- Given conditions and problem setup
axiom triangle_ABC_sides : ∀ {a b c : ℝ}, sides a b c
axiom triangle_ABC_angles : ∀ {A B C : ℝ}, angles A B C
axiom sin_relation : ∀ {A B C : ℝ},
  sin C * sin (A - B) = sin B * sin (C - A)

-- Prove Part (1): If A = 2B, then C = 5π/8
theorem part1_A_eq_2B_implies_C :
  A = 2 * B → C = 5 * π / 8 :=
by
  intro h
  sorry

-- Prove Part (2): 2a² = b² + c²
theorem part2_2a_squared_eq_b_squared_plus_c_squared :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
by
  sorry

end part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135931


namespace number_of_n_for_fraction_square_l135_135472

theorem number_of_n_for_fraction_square :
  (∃ n : ℤ, ∃ k : ℤ, n / (30 - 2 * n) = k^2) → (n = 0) :=
by sorry

example : finset.card {n : ℤ | ∃ k : ℤ, n / (30 - 2 * n) = k^2} = 1 :=
by sorry

end number_of_n_for_fraction_square_l135_135472


namespace typesetter_times_l135_135548

theorem typesetter_times (α β γ : ℝ) (h1 : 1 / β - 1 / α = 10)
                                        (h2 : 1 / β - 1 / γ = 6)
                                        (h3 : 9 * (α + β) = 10 * (β + γ)) :
    α = 1 / 20 ∧ β = 1 / 30 ∧ γ = 1 / 24 :=
by {
  sorry
}

end typesetter_times_l135_135548


namespace price_correct_l135_135687

noncomputable def price_per_glass_second_day (O G : ℝ) (price_day_one revenue_same : ℝ) : ℝ :=
let price_day_two := 0.40 in
price_day_two

theorem price_correct {O G : ℝ} (price_day_one : ℝ) (revenue_same : 2 * G * price_day_one = 3 * G * price_per_glass_second_day O G price_day_one revenue_same) :
  price_per_glass_second_day O G price_day_one revenue_same = 0.40 :=
sorry

end price_correct_l135_135687


namespace value_of_x_l135_135541

theorem value_of_x (x : ℝ) : 3 - 5 + 7 = 6 - x → x = 1 :=
by
  intro h
  sorry

end value_of_x_l135_135541


namespace count_groupings_l135_135261

theorem count_groupings (dogs : Finset ℕ) (Fluffy Nipper : ℕ) (h : Fluffy ≠ Nipper) (h_count : dogs.card = 12) :
  ∃ (g1 g2 g3 : Finset ℕ), 
    g1.card = 4 ∧ g2.card = 5 ∧ g3.card = 3 ∧
    Fluffy ∈ g1 ∧ Nipper ∈ g2 ∧
    (∀ x, x ∈ g1 ∨ x ∈ g2 ∨ x ∈ g3) ∧
     ∑ y in dogs, 1 = 12 ∧
     (∏ (a ∈ Finset.choose 10 3), ∏ (b ∈ Finset.choose 7 4), 1) = 4200
:= sorry

end count_groupings_l135_135261


namespace binom_not_divisible_l135_135070

theorem binom_not_divisible (k : ℤ) : k ≠ 1 → ∃ᶠ n in Filter.atTop, (n + k) ∣ (Nat.choose (2 * n) n) := 
sorry

end binom_not_divisible_l135_135070


namespace find_C_prove_relation_l135_135955

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A), and A = 2B,
prove that C = 5/8 * π. -/
theorem find_C
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  C = ⅝ * Real.pi :=
sorry

/-- Let ΔABC have sides a, b, c opposite to angles A, B, C respectively.
Given sin C * sin (A - B) = sin B * sin (C - A),
prove that 2 * a ^ 2 = b ^ 2 + c ^ 2. -/
theorem prove_relation
  (a b c A B C : ℝ)
  (h₁ : sin C * sin (A - B) = sin B * sin (C - A))
  (h₂ : A = 2 * B) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
sorry

end find_C_prove_relation_l135_135955


namespace find_x_solutions_l135_135462

theorem find_x_solutions (x : ℝ) :
  (√(√(43 - 2 * x)) + √(√(37 + 2 * x)) = 4) ↔ (x = -19 ∨ x = 21) :=
by
  sorry

end find_x_solutions_l135_135462


namespace sum_of_all_possible_S_l135_135521

def A : set ℤ := {-5, -4, 0, 6, 7, 9, 11, 12}

def S (X : set ℤ) : ℤ := X.sum

theorem sum_of_all_possible_S (S : ℤ) : (∑ X in (finset.powerset (finset.from_set A)), S (X.val)) = 4608 := sorry

end sum_of_all_possible_S_l135_135521


namespace number_of_squares_in_3x3_grid_l135_135567

-- Definitions based on conditions
def grid_3x3 : Set (ℕ × ℕ) := {(x, y) | x ∈ {0, 1, 2} ∧ y ∈ {0, 1, 2}}

-- Statement of the problem
theorem number_of_squares_in_3x3_grid : 
  (∃ n : ℕ, n = 5 ∧ 
    (∀ (a b c d : (ℕ × ℕ)), (a ∈ grid_3x3) ∧ (b ∈ grid_3x3) ∧ (c ∈ grid_3x3) ∧ (d ∈ grid_3x3) → 
      (is_square a b c d) → 
      ((a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) → 
      ∃ (set_of_squares : Finset (Finset (ℕ × ℕ))), set_of_squares.card = n))) := sorry

end number_of_squares_in_3x3_grid_l135_135567


namespace PQRS_is_parallelogram_l135_135179

-- Definition representing the convex quadrilateral with midpoints
structure ConvexQuadrilateral (A B C D : Point) :=
(midpoint_AB : Point)
(midpoint_BC : Point)
(midpoint_CD : Point)
(midpoint_DA : Point)
(segment_intersects : ∀ P Q R S, (P ≠ Q ∧ Q ≠ R ∧ R ≠ S ∧ S ≠ P) →
  (midpoint_AB = P ∧ midpoint_BC = Q ∧ midpoint_CD = R ∧ midpoint_DA = S) →
  segments_intersect_and_divide_each_other_into_three_equal_parts P Q R S)

-- The proof problem stating PQRS is a parallelogram
theorem PQRS_is_parallelogram {A B C D P Q R S : Point} 
  (H : ConvexQuadrilateral A B C D) :
  H.segment_intersects P Q R S → Parallelogram P Q R S :=
sorry

end PQRS_is_parallelogram_l135_135179


namespace no_isosceles_triangle_from_geometric_progression_l135_135655

theorem no_isosceles_triangle_from_geometric_progression :
  (∀ (l : list ℝ),
    l ⊆ list.map (λ n : ℕ, 0.9 ^ n) (list.range 100) →
    ∀ a b c ∈ l, a = b →
      ¬ (a + b > c ∧ a + c > b ∧ b + c > a)) :=
by
  sorry

end no_isosceles_triangle_from_geometric_progression_l135_135655


namespace height_difference_l135_135018

variable (H_A H_B : ℝ)

-- Conditions
axiom B_is_66_67_percent_more_than_A : H_B = H_A * 1.6667

-- Proof statement
theorem height_difference (H_A H_B : ℝ) (h : H_B = H_A * 1.6667) : 
  (H_B - H_A) / H_B * 100 = 40 := by
sorry

end height_difference_l135_135018


namespace inverse_function_point_l135_135175

theorem inverse_function_point (a : ℝ) :
  (∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y) ∧ g 4 = 1) →
  a = 3 :=
by
  let f := λ x : ℝ, Real.log (x + 1) / Real.log 2 + a
  assume ⟨g, hg1, hg2, hg3⟩
  have h : f 1 = 4
  sorry

end inverse_function_point_l135_135175


namespace max_non_managers_l135_135178

/-- In a department, the ratio of managers to non-managers must always be greater than 7:32.
If the maximum number of non-managers is 36, then prove that the highest number cannot exceed 36,
under the given ratio constraint. -/
theorem max_non_managers (M N : ℕ) (h1 : M > 0)
  (h2 : N ≤ 36) (h3 : (M:ℚ) / (N:ℚ) > 7 / 32) : N = 36 :=
begin
  -- Given that we have the initial ratio condition and the constraints on N,
  -- we aim to prove that N must be equal to the maximum given value.
  suffices : N = 36, from this,
  
  -- Assume the contrary, and proceed by contradiction to establish that N cannot be less than 36.
  by_contradiction h,
  have h_leq := lt_of_le_of_ne (le_of_not_gt h) (ne.symm h),
  -- The ratio condition given:
  -- (M / N) > 7 / 32
  
  -- Inequality setup, translated to real numbers:
  let bound := 252 / 32,
  have M_gt : (M:ℚ) > bound := by 
    calc 
      (M:ℚ) = (M:ℚ)    : by simp 
      ...    > (252:ℚ)/32 : by sorry, -- This follows from the ratio constraint (h3)

  -- Contradiction establishment
  calc 
    252 / 32 ≈ 7.875 : by simp [bound]
    ... < N    : by sorry, -- As per the condition of maximum N
  -- However, nothing contradicts this calculation that allows N to be smaller 

  -- Decompose the above to ensure contradiction 
  suffices h_suff : M ≤ 8, from this,  λ (nq:ne.symm h) this (232 / 32 h2).ne h_suff  
   -- Substitute N=36 favorable bound Value with standards in place ensure maximum falls within 36 range
--Hence 
⟩

end max_non_managers_l135_135178


namespace total_students_in_class_l135_135023

variable (K M Both Total : ℕ)

theorem total_students_in_class
  (hK : K = 38)
  (hM : M = 39)
  (hBoth : Both = 32)
  (hTotal : Total = K + M - Both) :
  Total = 45 := 
by
  rw [hK, hM, hBoth] at hTotal
  exact hTotal

end total_students_in_class_l135_135023


namespace Q1_Q2_Q3_l135_135510

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem Q1 (a : ℝ) : f a = -1 / 3 → a = 2 := by
  sorry

theorem Q2 (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ -1) : f (1 / x) = -f x := by
  sorry

theorem Q3 : (∑ (i : ℤ) in Finset.range ((2018 - 2) * 2 + 1), f (if i < 2017 then 1 / (i + 2 : ℝ) else (i - 2016 : ℝ))) = 0 := by
  sorry

end Q1_Q2_Q3_l135_135510


namespace range_of_a_exp_product_ineq_l135_135835

noncomputable def f (x a : ℝ) := x * Real.log x - (a / 2) * x ^ 2 - x + a

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 ∈ Set.Ioi (0 : ℝ), x1 ≠ x2 ∧ f'.eval x1 = 0 ∧ f'.eval x2 = 0) ↔ 0 < a ∧ a < 1 / Real.exp 1 := sorry 

theorem exp_product_ineq (n : ℕ) (hn : 0 < n) :
  (∏ i in Finset.range n, (Real.exp 1 + 1 / 2 ^ (i + 1))) < Real.exp (n + 1 / Real.exp 1) := sorry

end range_of_a_exp_product_ineq_l135_135835


namespace trig_identity_proof_l135_135149

theorem trig_identity_proof (x y a b c : ℝ) (h1: sin x + sin y = 2a) (h2: cos x + cos y = 2b) (h3: tan x + tan y = 2c) :
  a * (b + a * c) = c * (a^2 + b^2)^2 :=
by
  sorry

end trig_identity_proof_l135_135149


namespace part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135928

-- Definitions for the conditions in the problem
variable {A B C a b c : ℝ}

-- Given conditions and problem setup
axiom triangle_ABC_sides : ∀ {a b c : ℝ}, sides a b c
axiom triangle_ABC_angles : ∀ {A B C : ℝ}, angles A B C
axiom sin_relation : ∀ {A B C : ℝ},
  sin C * sin (A - B) = sin B * sin (C - A)

-- Prove Part (1): If A = 2B, then C = 5π/8
theorem part1_A_eq_2B_implies_C :
  A = 2 * B → C = 5 * π / 8 :=
by
  intro h
  sorry

-- Prove Part (2): 2a² = b² + c²
theorem part2_2a_squared_eq_b_squared_plus_c_squared :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
by
  sorry

end part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135928


namespace triangle_min_value_l135_135898

open Real

theorem triangle_min_value
  (A B C : ℝ)
  (h_triangle: A + B + C = π)
  (h_sin: sin (2 * A + B) = 2 * sin B) :
  tan A + tan C + 2 / tan B ≥ 2 :=
sorry

end triangle_min_value_l135_135898


namespace sin_arithmetic_sequence_l135_135768

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 2 * Real.pi) : 
  (Real.sin a + Real.sin (3 * a) = 2 * Real.sin (2 * a)) ↔ (a = Real.pi) :=
sorry

end sin_arithmetic_sequence_l135_135768


namespace relationship_among_a_b_c_l135_135100

theorem relationship_among_a_b_c 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = (1 / 2) ^ (3 / 2))
  (hb : b = Real.log pi)
  (hc : c = Real.logb 0.5 (3 / 2)) :
  c < a ∧ a < b :=
by 
  sorry

end relationship_among_a_b_c_l135_135100


namespace sum_1_to_100_eq_5050_l135_135695

-- Define the function that calculates the sum from 1 to n using a loop structure
def sum_to_n (n : ℕ) : ℕ :=
  let sum := 0
  let rec loop (i : ℕ) (accum : ℕ) : ℕ :=
    if i > n then accum
    else loop (i + 1) (accum + i)
  loop 1 sum

-- Statement of the theorem
theorem sum_1_to_100_eq_5050 : sum_to_n 100 = 5050 :=
by sorry

end sum_1_to_100_eq_5050_l135_135695


namespace required_extra_money_l135_135041

theorem required_extra_money 
(Patricia_money Lisa_money Charlotte_money : ℕ) 
(hP : Patricia_money = 6) 
(hL : Lisa_money = 5 * Patricia_money) 
(hC : Lisa_money = 2 * Charlotte_money) 
(cost : ℕ) 
(hCost : cost = 100) : 
  cost - (Patricia_money + Lisa_money + Charlotte_money) = 49 := 
by 
  sorry

end required_extra_money_l135_135041


namespace find_a9_l135_135484

theorem find_a9 
  (α : Type) [Field α] (x : α) (a : Fin 11 → α) :
  (1 - x) ^ 10 = ∑ i in Finset.range 11, a i * (1 + x) ^ i →
  a 9 = -20 := 
by
  sorry

end find_a9_l135_135484


namespace mag_conjugate_eq_sqrt_5_l135_135798

variable (z : ℂ)
variable (h : z = 2 - 1 * complex.I)

theorem mag_conjugate_eq_sqrt_5 (h : z = 2 - 1 * complex.I) : complex.abs (conj z) = sqrt 5 := by
  rw [h, complex.conj]
  simp
  rw [complex.abs]
  norm_num
  sorry

end mag_conjugate_eq_sqrt_5_l135_135798


namespace part_1_proof_part_2_proof_part_3_proof_l135_135496

-- Given conditions
variables {a b c : ℝ} {A B C : ℝ}

-- Condition 1: sides opposite the angles in triangle ABC
axiom sides_of_triangle (A B C : ℝ) (a b c : ℝ) : a = b ∧ b = c ∧ c = a

-- Condition 2: Given equation
axiom given_equation (A + C = π - B) (Sin B / (Sin A + Sin C) = (a - c) / (b - sqrt 2 * c)) : A = π / 4

-- Define the correct answer
def part_1_solution : ℝ := π / 4

theorem part_1_proof : A = part_1_solution :=
by
  exact given_equation

-- Given a = sqrt 2, circumcenter O and finding minimum value of vector expression
variable {O : Point}

axiom circumcenter_radius (a = sqrt 2) (A = π / 4) (O = circumcenter A B C) : r = 1

def part_2_solution : ℝ := 3 - sqrt 5

theorem part_2_proof : 
  |3 * OA + 2 * OB + OC| = part_2_solution :=
by
  exact circumcenter_radius

-- Given P is a moving point on circumcircle, finding maximum value of dot product
variable {P : Point}

axiom moving_point_max (P : Point on circumcircle) : (P - B) • (P - C) = sqrt 2 + 1

def part_3_solution : ℝ := sqrt 2 + 1

theorem part_3_proof : (P - B) • (P - C) = part_3_solution :=
by
  exact moving_point_max

end part_1_proof_part_2_proof_part_3_proof_l135_135496


namespace bonnets_friday_less_than_thursday_l135_135606

/-- Problem Definitions -/
def bonnets_monday := 10
def bonnets_tuesday_wednesday := 2 * bonnets_monday
def bonnets_thursday := bonnets_monday + 5
def bonnets_sent_per_orphanage := 11
def orphanages := 5

/-- Theorem Statement -/
theorem bonnets_friday_less_than_thursday :
  let total_bonnets := bonnets_monday + bonnets_tuesday_wednesday + bonnets_thursday
  let total_bonnets_sent := bonnets_sent_per_orphanage * orphanages
  let bonnets_friday := total_bonnets_sent - total_bonnets
  bonnets_thursday - bonnets_friday = 5 :=
begin
  sorry,
end

end bonnets_friday_less_than_thursday_l135_135606


namespace quadratic_two_distinct_real_roots_l135_135519

theorem quadratic_two_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 2*(m-2)*x1 + m^2 = 0 ∧ x2^2 - 2*(m-2)*x2 + m^2 = 0)
  → m < 1 :=
begin
  sorry
end

end quadratic_two_distinct_real_roots_l135_135519


namespace not_black_cows_count_l135_135247

theorem not_black_cows_count (total_cows : ℕ) (black_cows : ℕ) (h1 : total_cows = 18) (h2 : black_cows = 5 + total_cows / 2) :
  total_cows - black_cows = 4 :=
by 
  -- Insert the actual proof here
  sorry

end not_black_cows_count_l135_135247


namespace binom_n_n_minus_3_l135_135034

theorem binom_n_n_minus_3 (n : ℕ) (h : n ≥ 3) : (nat.choose n (n-3)) = (n * (n-1) * (n-2)) / 6 := 
by sorry

end binom_n_n_minus_3_l135_135034


namespace negation_of_some_is_all_not_isosceles_l135_135144

-- Define the proposition p
def p : Prop := ∃ (T : Type) [_inst : Triangle T], is_isosceles T

-- Negation of the proposition p
def np : Prop := ¬p

-- Equivalent statement we want to prove
theorem negation_of_some_is_all_not_isosceles :
  np ↔ ∀ (T : Type) [_inst : Triangle T], ¬is_isosceles T :=
by
  -- proof goes here
  sorry

end negation_of_some_is_all_not_isosceles_l135_135144


namespace pyramid_base_side_length_l135_135632

theorem pyramid_base_side_length (A : ℝ) (h : ℝ) (s : ℝ)
  (hA : A = 200)
  (hh : h = 40)
  (hface : A = (1 / 2) * s * h) : 
  s = 10 :=
by
  sorry

end pyramid_base_side_length_l135_135632


namespace circles_are_separate_l135_135526

def circle_center (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

theorem circles_are_separate :
  circle_center 0 0 1 x y → 
  circle_center 3 (-4) 3 x' y' →
  dist (0, 0) (3, -4) > 1 + 3 :=
by
  intro h₁ h₂
  sorry

end circles_are_separate_l135_135526


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135356

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135356


namespace positional_relationship_l135_135799

variable {Point Line Plane : Type} -- Assuming we have these types in our geometry library

-- Definitions of perpendicular, subset, and parallel relationships
def perp (m : Line) (α : Plane) : Prop := sorry
def perp (m n : Line) : Prop := sorry
def subset (n : Line) (α : Plane) : Prop := sorry
def parallel (n : Line) (α : Plane) : Prop := sorry

-- Given variables and conditions
variable (m n : Line) (α : Plane)
variable (h1 : perp m α)
variable (h2 : perp m n)

-- A Lean theorem directly capturing the problem
theorem positional_relationship :
  parallel n α ∨ subset n α :=
sorry

end positional_relationship_l135_135799


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135361

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ 100 ≤ n ∧ n < 1000 ∧ (9 ∣ n) ∧ (∑ digit in n.digits, digit = 27) :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135361


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135326

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ (100 ≤ n ∧ n < 1000) ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) :=
by {
  sorry
}

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135326


namespace bunbun_solution_l135_135653

noncomputable def maximal_sum_arcs : ℝ := 2042222

def bunbun_problem : Prop :=
  ∃ (points : ℕ → ℝ) (γ : circle) (n : ℕ),
    n = 2022 ∧
    circumference γ = 2022 ∧
    (∀ i, points i ∈ γ) ∧
    (∀ i, points i ≠ points (i + 1)) ∧
    sum_of_arcs points γ n = maximal_sum_arcs

theorem bunbun_solution : bunbun_problem := 
by {
  sorry
}

end bunbun_solution_l135_135653


namespace max_distance_from_circle_to_line_l135_135562
noncomputable def max_distance_circle_to_line : ℝ :=
  let radius := (sqrt 6) / 2
  let distance := 1 / 2
  radius + distance

theorem max_distance_from_circle_to_line :
  ∀ θ : ℝ, 
    let x := (sqrt 6 / 2) * cos θ
    let y := (sqrt 6 / 2) * sin θ
    | sqrt (7 * x - y)^2 + sqrt 2 = sqrt 7 + sqrt 1 + sqrt 5 :=
    max_distance_circle_to_line = (sqrt 6) / 2 + (1 / 2) :=
begin
  sorry
end

end max_distance_from_circle_to_line_l135_135562


namespace largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135352

theorem largest_three_digit_multiple_of_9_with_digits_sum_27 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ ((n / 100) + ((n % 100) / 10) + (n % 10) = 27) ∧ n = 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digits_sum_27_l135_135352


namespace magnitude_of_complex_number_l135_135066

def complex_magnitude (z : ℂ) : ℝ := abs z

theorem magnitude_of_complex_number :
  let z : ℂ := 1 - (1 / 2) * Complex.I in
  complex_magnitude z = (Real.sqrt 5) / 2 :=
by
  sorry

end magnitude_of_complex_number_l135_135066


namespace part1_part2_l135_135963

variable {A B C a b c : ℝ}

theorem part1 (h₁ : A = 2 * B) (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : C = 5 / 8 * π :=
  sorry

theorem part2 (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
  sorry

end part1_part2_l135_135963


namespace undecided_voters_percentage_l135_135603

theorem undecided_voters_percentage
  (biff_percent : ℝ)
  (total_people : ℤ)
  (marty_votes : ℤ)
  (undecided_percent : ℝ) :
  biff_percent = 0.45 →
  total_people = 200 →
  marty_votes = 94 →
  undecided_percent = ((total_people - (marty_votes + (biff_percent * total_people))) / total_people) * 100 →
  undecided_percent = 8 :=
by 
  intros h1 h2 h3 h4
  sorry

end undecided_voters_percentage_l135_135603


namespace cardinality_relation_l135_135016

def A_n (n : ℕ) : Finset (List Char) :=
  {w ∈ (Finset.univ : Finset (List Char)).filter (λ l, l.length = n ∧ ∀ i, (i < n - 1) → ¬((l[i] = 'a' ∧ l[i + 1] = 'a') ∨ (l[i] = 'b' ∧ l[i + 1] = 'b')))}

def B_n (n : ℕ) : Finset (List Char) :=
  {w ∈ (Finset.univ : Finset (List Char)).filter (λ l, l.length = n ∧ ∀ i, (i < n - 2) → ¬(l[i] ≠ l[i+1] ∧ l[i] ≠ l[i+2] ∧ l[i+1] ≠ l[i+2]))}

theorem cardinality_relation (n : ℕ) : (B_n (n + 1)).card = 3 * (A_n n).card := sorry

end cardinality_relation_l135_135016


namespace initial_students_proof_l135_135027

def initial_students (e : ℝ) (transferred : ℝ) (left : ℝ) : ℝ :=
  e + transferred + left

theorem initial_students_proof : initial_students 28 10 4 = 42 :=
  by
    -- This is where the proof would go, but we use 'sorry' to skip it.
    sorry

end initial_students_proof_l135_135027


namespace range_of_a_l135_135118

theorem range_of_a(p q: Prop)
  (hp: p ↔ (a = 0 ∨ (0 < a ∧ a < 4)))
  (hq: q ↔ (-1 < a ∧ a < 3))
  (hpor: p ∨ q)
  (hpand: ¬(p ∧ q)):
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := by sorry

end range_of_a_l135_135118


namespace valid_even_integers_le_2024_l135_135044

noncomputable def count_valid_even_integers (upper_bound : ℕ) : ℕ :=
  (list.filter (λ n, even n ∧ n ≤ upper_bound ∧
    ∃ (f : fin n → fin n → Prop), 
      (∀ m < n, ∀ k < n, f m k → (m + 1 + k + 1) % 3 = 0)
  ) (list.range (upper_bound + 1))).length

theorem valid_even_integers_le_2024 : count_valid_even_integers 2024 = 675 := by
  sorry

end valid_even_integers_le_2024_l135_135044


namespace circles_intersect_l135_135644

noncomputable def circle1_center : ℝ × ℝ := (0, 0)
noncomputable def circle1_radius : ℝ := 2

noncomputable def circle2_center : ℝ × ℝ := (2, 0)
noncomputable def circle2_radius : ℝ := 3

theorem circles_intersect :
    let d := dist circle1_center circle2_center in
    let R := circle1_radius in
    let r := circle2_radius in
    abs (R - r) < d ∧ d < R + r := 
by
    sorry

end circles_intersect_l135_135644


namespace triangle_sides_angles_l135_135975

theorem triangle_sides_angles (a b c A B C : ℝ) (h1: A = 2 * B) 
  (h2: sin C * sin (A - B) = sin B * sin (C - A)) 
  (h3: A + B + C = π) :
  (C = 5 * π / 8) ∧ (2 * a^2 = b^2 + c^2) :=
by
  -- Proof omitted
  sorry

end triangle_sides_angles_l135_135975


namespace molecular_weight_C4H1O_l135_135771

def AtomicWeights (element : String) : Float :=
  if element = "C" then 12.01
  else if element = "H" then 1.008
  else if element = "O" then 16.00
  else 0

def MolecularWeight (formula : List (String × Nat)) : Float :=
  formula.foldr (λ (pair : String × Nat) acc, acc + pair.snd * AtomicWeights pair.fst) 0

theorem molecular_weight_C4H1O :
  MolecularWeight [("C", 4), ("H", 1), ("O", 1)] = 65.048 :=
by
  simp [MolecularWeight, AtomicWeights]
  simp [(*), (+)]
  sorry

end molecular_weight_C4H1O_l135_135771


namespace area_of_circle_between_chords_l135_135872

-- Given definitions
def circle_radius : ℝ := 8
def chord_distance : ℝ := 4

-- Declaring and stating the problem in Lean
theorem area_of_circle_between_chords 
  (R : ℝ)
  (d : ℝ)
  (calculate_area : Π (R : ℝ) (d : ℝ), ℝ :=λ (R d : ℝ), 26.04)
  (hR : R = circle_radius)
  (hd : d = chord_distance) :
  calculate_area R d = 26.04 := sorry

end area_of_circle_between_chords_l135_135872


namespace part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135932

-- Definitions for the conditions in the problem
variable {A B C a b c : ℝ}

-- Given conditions and problem setup
axiom triangle_ABC_sides : ∀ {a b c : ℝ}, sides a b c
axiom triangle_ABC_angles : ∀ {A B C : ℝ}, angles A B C
axiom sin_relation : ∀ {A B C : ℝ},
  sin C * sin (A - B) = sin B * sin (C - A)

-- Prove Part (1): If A = 2B, then C = 5π/8
theorem part1_A_eq_2B_implies_C :
  A = 2 * B → C = 5 * π / 8 :=
by
  intro h
  sorry

-- Prove Part (2): 2a² = b² + c²
theorem part2_2a_squared_eq_b_squared_plus_c_squared :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
by
  sorry

end part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135932


namespace sequence_2016_eq_6_l135_135146

noncomputable def sequence (n : ℕ) : ℚ :=
if n = 1 then 2
else if n = 2 then 1/3
else sequence (n - 1) / sequence (n - 2)

theorem sequence_2016_eq_6 :
  sequence 2016 = 6 :=
sorry

end sequence_2016_eq_6_l135_135146


namespace cos_neg_thirty_deg_cos_thirty_deg_cos_minus_thirty_eq_sqrt_three_div_two_l135_135467

theorem cos_neg_thirty_deg :
  (Real.cos (-30 * Real.pi / 180)) = (Real.cos (30 * Real.pi / 180)) :=
begin
  -- Using the even property of cosine
  exact Real.cos_neg (30 * Real.pi / 180),
end

theorem cos_thirty_deg :
  (Real.cos (30 * Real.pi / 180)) = sqrt 3 / 2 :=
begin
  -- This value comes from special triangles or unit circle
  apply Real.cos_pi_div_six,
end

theorem cos_minus_thirty_eq_sqrt_three_div_two :
  (Real.cos (-30 * Real.pi / 180)) = sqrt 3 / 2 :=
by
  apply eq.trans cos_neg_thirty_deg cos_thirty_deg

end cos_neg_thirty_deg_cos_thirty_deg_cos_minus_thirty_eq_sqrt_three_div_two_l135_135467


namespace expected_value_girls_left_of_boys_l135_135736


noncomputable def expected_girls_to_left_of_all_boys (n m : ℕ) [fact (0 < n)] [fact (0 < m)] : ℝ :=
  let total := n + m
  (m : ℝ) / (total + 1)

theorem expected_value_girls_left_of_boys : 
  let n := 10
  let m := 7
  expected_girls_to_left_of_all_boys n m = 7 / 11 :=
by
  -- This is the key part of the theorem as described in the task
  sorry

end expected_value_girls_left_of_boys_l135_135736


namespace compute_value_of_expression_l135_135215

theorem compute_value_of_expression (p q : ℝ) (h₁ : 3 * p ^ 2 - 5 * p - 12 = 0) (h₂ : 3 * q ^ 2 - 5 * q - 12 = 0) :
  (3 * p ^ 2 - 3 * q ^ 2) / (p - q) = 5 :=
by
  sorry

end compute_value_of_expression_l135_135215


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135311

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def digit_sum (n : ℕ) : ℕ := 
  let d1 := n / 100 in
  let d2 := (n % 100) / 10 in
  let d3 := n % 10 in
  d1 + d2 + d3

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, is_three_digit n ∧ is_multiple_of_9 n ∧ digit_sum n = 27 ∧
  ∀ m : ℕ, is_three_digit m ∧ is_multiple_of_9 m ∧ digit_sum m = 27 → m ≤ n := 
by 
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135311


namespace initial_noodles_l135_135450

variable (d w e r : ℕ)

-- Conditions
def gave_to_william (w : ℕ) := w = 15
def gave_to_emily (e : ℕ) := e = 20
def remaining_noodles (r : ℕ) := r = 40

-- The statement to be proven
theorem initial_noodles (h1 : gave_to_william w) (h2 : gave_to_emily e) (h3 : remaining_noodles r) : d = w + e + r := by
  -- Proof will be filled in later.
  sorry

end initial_noodles_l135_135450


namespace sum_m_n_eq_one_l135_135483

variable (m n : ℝ)
def A : Set ℝ := {2, Real.logBase 7 m}
def B : Set ℝ := {m, n}
def intersection_condition := A ∩ B = {0}

theorem sum_m_n_eq_one (m n : ℝ)
  (h1 : A m = {2, Real.logBase 7 m})
  (h2 : B m n = {m, n})
  (h3 : intersection_condition m n) :
  m + n = 1 := 
sorry

end sum_m_n_eq_one_l135_135483


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135328

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ (100 ≤ n ∧ n < 1000) ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) :=
by {
  sorry
}

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135328


namespace count_non_congruent_triangles_with_perimeter_10_l135_135156

def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def unique_triangles_with_perimeter_10 : finset (ℕ × ℕ × ℕ) :=
  ((finset.range 11).product (finset.range 11)).product (finset.range 11)
  |>.filter (λ t, 
    let (a, b, c) := t.fst.fst, t.fst.snd, t.snd in
      a + b + c = 10 ∧ a ≤ b ∧ b ≤ c ∧ is_triangle a b c)

theorem count_non_congruent_triangles_with_perimeter_10 : 
  unique_triangles_with_perimeter_10.card = 3 := 
sorry

end count_non_congruent_triangles_with_perimeter_10_l135_135156


namespace bananas_to_pears_l135_135423

theorem bananas_to_pears:
  (∀ b a o p : ℕ, 
    6 * b = 4 * a → 
    5 * a = 3 * o → 
    4 * o = 7 * p → 
    36 * b = 28 * p) :=
by
  intros b a o p h1 h2 h3
  -- We need to prove 36 * b = 28 * p under the given conditions
  sorry

end bananas_to_pears_l135_135423


namespace find_number_l135_135370

noncomputable def number_with_point_one_percent (x : ℝ) : Prop :=
  0.1 * x / 100 = 12.356

theorem find_number :
  ∃ x : ℝ, number_with_point_one_percent x ∧ x = 12356 :=
by
  sorry

end find_number_l135_135370


namespace vectors_parallel_iff_l135_135528

-- Define the vectors a and b as given in the conditions
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, m + 1)

-- Define what it means for two vectors to be parallel
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

-- The statement that we need to prove
theorem vectors_parallel_iff (m : ℝ) : parallel a (b m) ↔ m = 1 := by
  sorry

end vectors_parallel_iff_l135_135528


namespace train_speed_correct_l135_135722

noncomputable def train_speed (train_length : ℝ) (bridge_length : ℝ) (time_seconds : ℝ) : ℝ :=
  (train_length + bridge_length) / time_seconds

theorem train_speed_correct :
  train_speed (400 : ℝ) (300 : ℝ) (45 : ℝ) = 700 / 45 :=
by
  sorry

end train_speed_correct_l135_135722


namespace matrix_identity_l135_135210

open Matrix

-- Define matrix B
def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2, 3], ![2, 1, 2], ![3, 2, 1]]

-- Define identity matrix I
def I : Matrix (Fin 3) (Fin 3) ℝ := 1

-- Define the constants
def p' : ℝ := -5
def q' : ℝ := -8
def r' : ℝ := -6

-- Assertion to prove
theorem matrix_identity :
  B^3 + p' • B^2 + q' • B + r' • I = 0 := 
by
  sorry

end matrix_identity_l135_135210


namespace groupDivisionWays_l135_135264

-- Definitions based on conditions
def numDogs : ℕ := 12
def group1Size : ℕ := 4
def group2Size : ℕ := 5
def group3Size : ℕ := 3
def fluffy : ℕ := 1 -- Fluffy's assigned position
def nipper : ℕ := 2 -- Nipper's assigned position

-- Function to compute binomial coefficients
def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom (n+1) k

-- Theorem to prove the number of ways to form the groups
theorem groupDivisionWays :
  (binom 10 3 * binom 7 4) = 4200 :=
by
  sorry

end groupDivisionWays_l135_135264


namespace smallest_n_l135_135090

theorem smallest_n (n : ℕ) (h1 : n ≥ 1)
  (h2 : ∃ k : ℕ, 2002 * n = k ^ 3)
  (h3 : ∃ m : ℕ, n = 2002 * m ^ 2) :
  n = 2002^5 := sorry

end smallest_n_l135_135090


namespace sin_arithmetic_sequence_l135_135765

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 360) :
  (sin a + sin (3 * a) = 2 * sin (2 * a)) ↔ 
  (a = 30 ∨ a = 150 ∨ a = 210 ∨ a = 330) :=
by
  sorry

end sin_arithmetic_sequence_l135_135765


namespace midpoint_complex_l135_135566

-- Define the complex points A and B
def A : ℂ := 6 + 5 * complex.I
def B : ℂ := -2 + 3 * complex.I

-- Define midpoint calculation in complex numbers
def midpoint (z1 z2 : ℂ) : ℂ := (z1 + z2) / 2

-- The target complex number at point C
def C : ℂ := midpoint A B

theorem midpoint_complex :
  C = 2 + 4 * complex.I :=
by
  -- Define points A and B
  let A : ℂ := 6 + 5 * complex.I
  let B : ℂ := -2 + 3 * complex.I

  -- Define the midpoint C
  let C : ℂ := midpoint A B

  -- Goal is to prove that C is 2 + 4i
  show C = 2 + 4 * complex.I
  sorry

end midpoint_complex_l135_135566


namespace sum_exists_l135_135250

theorem sum_exists 
  (n : ℕ) 
  (hn : n ≥ 5) 
  (k : ℕ) 
  (hk : k > (n + 1) / 2) 
  (a : ℕ → ℕ) 
  (ha1 : ∀ i, 1 ≤ a i) 
  (ha2 : ∀ i, a i < n) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j):
  ∃ i j l, i ≠ j ∧ a i + a j = a l := 
by 
  sorry

end sum_exists_l135_135250


namespace ben_savings_l135_135428

theorem ben_savings:
  ∃ x : ℕ, (50 - 15) * x * 2 + 10 = 500 ∧ x = 7 :=
by
  -- Definitions based on conditions
  let daily_savings := 50 - 15
  have h1 : daily_savings = 35 := by norm_num
  let total_savings := daily_savings * x
  let doubled_savings := 2 * total_savings
  let final_savings := doubled_savings + 10

  -- Existence of x such that (50 - 15) * x * 2 + 10 = 500 and x = 7 
  use 7
  split
  { -- Show that the equation holds
    show final_savings = 500,
    calc
      final_savings = (daily_savings * 7 * 2) + 10 : by sorry
                   ... = 500 : by norm_num
  }
  { -- Show that x = 7
    refl
  }
  sorry

end ben_savings_l135_135428


namespace funds_after_12_months_repayment_possible_l135_135237

-- Define the initial conditions
def initial_investment : ℝ := 100000
def monthly_profit_rate : ℝ := 0.2
def monthly_expense_rate : ℝ := 0.1
def fixed_monthly_costs : ℝ := 3000
def annual_interest_rate : ℝ := 0.05
def months_in_year : ℕ := 12

-- Define the monthly update function
def monthly_update (funds : ℝ) : ℝ :=
    let profit := funds * (1 + monthly_profit_rate)
    let after_expense := profit * (1 - monthly_expense_rate)
    after_expense - fixed_monthly_costs

-- Compute the funds after 12 months
noncomputable def final_funds : ℝ :=
    (List.iterate monthly_update months_in_year initial_investment).last

-- Calculate whether the maker can repay the loan
def can_repay_loan (final_funds : ℝ) : Prop :=
    final_funds >= initial_investment * (1 + annual_interest_rate)

theorem funds_after_12_months :
    final_funds ≈ 194890 := sorry

theorem repayment_possible :
    can_repay_loan final_funds := sorry

end funds_after_12_months_repayment_possible_l135_135237


namespace find_k_for_parallel_lines_l135_135808

theorem find_k_for_parallel_lines 
  (k : ℝ)
  (l1 : ∀ x y : ℝ, x + 2 * y - 7 = 0)
  (l2 : ∀ x y : ℝ, 2 * x + k * x + 3 = 0)
  (parallel : l1 = l2) : k = 4 := 
by
  sorry

end find_k_for_parallel_lines_l135_135808


namespace problem_statement_l135_135035

theorem problem_statement : 
  2 * Real.sin (Float.pi / 3) - |Real.sqrt 3 - 2| - Real.sqrt 12 + (-1 / 2) ^ (-2 : ℤ) = 2 := 
sorry

end problem_statement_l135_135035


namespace part1_proof_part2_proof_l135_135949

-- Definitions for triangle sides and angles
variables {A B C a b c : ℝ}

-- Condition 1
def condition1 : Prop := sin C * sin (A - B) = sin B * sin (C - A)

-- Condition 2
def condition2 : Prop := A = 2 * B

-- Proof Problem 1
theorem part1_proof : condition1 → condition2 → C = 5 / 8 * π :=
by sorry

-- Proof Problem 2
theorem part2_proof : condition1 → condition2 → 2 * a^2 = b^2 + c^2 :=
by sorry

end part1_proof_part2_proof_l135_135949


namespace union_complement_eq_l135_135602

noncomputable def I := {x : ℤ | |x| < 3}
def A := {1, 2}
def B := {-2, -1, 2}
def complement_I (s : Set ℤ) : Set ℤ := {x ∈ I | x ∉ s}

theorem union_complement_eq :
  A ∪ (complement_I B) = {0, 1, 2} := by
  sorry

end union_complement_eq_l135_135602


namespace part1_part2_l135_135195

open Real

noncomputable def curve_parametric (α : ℝ) : ℝ × ℝ :=
  (2 + sqrt 10 * cos α, sqrt 10 * sin α)

noncomputable def curve_polar (ρ θ : ℝ) : Prop :=
  ρ^2 - 4 * ρ * cos θ - 6 = 0

noncomputable def line_polar (ρ θ : ℝ) : Prop :=
  ρ * cos θ + 2 * ρ * sin θ - 12 = 0

theorem part1 (α : ℝ) : ∃ ρ θ : ℝ, curve_polar ρ θ :=
  sorry

theorem part2 : ∃ ρ1 ρ2 : ℝ, curve_polar ρ1 (π / 4) ∧ line_polar ρ2 (π / 4) ∧ abs (ρ1 - ρ2) = sqrt 2 :=
  sorry

end part1_part2_l135_135195


namespace evaluate_P_l135_135913

noncomputable def P (x : ℝ) : ℝ := x^3 - 6*x^2 - 5*x + 4

theorem evaluate_P (y : ℝ) (z : ℝ) (hz : ∀ n : ℝ, z * P y = P (y - n) + P (y + n)) : P 2 = -22 := by
  sorry

end evaluate_P_l135_135913


namespace problem_1_problem_2_l135_135601

def f (x : ℝ) : ℝ := 2^x
def g (x : ℝ) : ℝ := 2^(-x)

theorem problem_1 (x : ℝ) (h : f(x) = 4 * g(x) + 3) : x = 2 :=
sorry

theorem problem_2 (a : ℝ) (h : ∃ x ∈ Set.Icc 0 4, f(a + x) - g(-2*x) ≥ 3) : 
  a ≥ 1 + 1 / 2 * Real.log 3 / Real.log 2 :=
sorry

end problem_1_problem_2_l135_135601


namespace part1_part2_l135_135923

theorem part1 (A B : ℝ) (h1 : A = 2 * B) : C = 5 * Real.pi / 8 :=
sorry

theorem part2 (a b c A B C : ℝ) 
  (h1 : sin C * sin (A - B) = sin B * sin (C - A)) 
  (h2 : A = 2 * B) :
   2 * a^2 = b^2 + c^2 :=
sorry

end part1_part2_l135_135923


namespace coefficient_of_linear_term_l135_135269

def quadratic_equation : Polynomial ℝ := 5 * X ^ 2 - 2 * X + 2

theorem coefficient_of_linear_term :
  polynomial.coeff quadratic_equation 1 = -2 := by
  sorry

end coefficient_of_linear_term_l135_135269


namespace int_inequality_l135_135914

variables {a b : ℝ} (f g : ℝ → ℝ)
  (h₀ : ∀ x ∈ Icc a b, 0 ≤ f x ∧ 0 ≤ g x)
  (h₁ : ContinuousOn f (Icc a b))
  (h₂ : ContinuousOn g (Icc a b))
  (h₃ : ∀ x ∈ Icc a b, f x ≤ f b ∧ g x ≤ g b)
  (h₄ : ∀ x ∈ Icc a b, ∫ t in a..x, sqrt (f t) ≤ ∫ t in a..x, sqrt (g t))
  (h₅ : ∫ t in a..b, sqrt (f t) = ∫ t in a..b, sqrt (g t))

theorem int_inequality : ∫ t in a..b, sqrt (1 + f t) ≥ ∫ t in a..b, sqrt (1 + g t) :=
sorry

end int_inequality_l135_135914


namespace quadratic_inequality_solution_l135_135287

theorem quadratic_inequality_solution : 
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x < -1 ∨ x > 3 / 2} :=
by
  sorry

end quadratic_inequality_solution_l135_135287


namespace students_in_second_class_eq_50_l135_135658

theorem students_in_second_class_eq_50 (x : ℕ) :
  (30 * 40 + x * 90) = (30 + x) * 71.25 → x = 50 :=
by
  intro h
  sorry

end students_in_second_class_eq_50_l135_135658


namespace sum_of_geometric_sequence_l135_135130

variables {α : Type*} [Field α]

def geometric_sequence (a : α) (q : α) (n : ℕ) : α :=
  a * q ^ n

theorem sum_of_geometric_sequence 
  (a : α) (q : α) (h_q : q = 2) 
  (h_sum4 : a * (1 + q + q^2 + q^3) = 1) : 
  let S4 := a * (1 + q + q^2 + q^3),
      S8 := S4 + S4 * q^4 
  in S8 = 17 := 
by
  have h_sum4_S4 : S4 = 1 := by rw [S4, h_sum4]
  have h_S4_q : S4 * q^4 = 16 := by rw [h_sum4_S4, h_q, pow_succ, pow_succ, pow_one, pow_one, pow_one]
  sorry

end sum_of_geometric_sequence_l135_135130


namespace range_of_t_l135_135837

-- Definitions from conditions
def parabola (x y : ℝ) := y^2 = 4 * x
def focus := (1, 0)
def line (x y : ℝ) := y = x - 1
def D (t : ℝ) := (-2, t)
def circle_E (x y : ℝ) := (x - 3)^2 + (y - 2)^2 = 16

-- Proof statement
theorem range_of_t : ∀ t : ℝ, point_on_parabola (-2) t → (2 - Real.sqrt 7) ≤ t ∧ t ≤ (2 + Real.sqrt 7) := sorry

end range_of_t_l135_135837


namespace altitudes_condition_eq_area_area_eq_altitudes_condition_altitudes_iff_area_l135_135503

open Real

variables {A B C D E F : Point} {BC CA AB EF FD DE : ℝ}
variables (R S : ℝ) (hR1 : circumradius A B C = R) (hR2 : acute_triangle A B C)
variables (hD : lies_on D BC) (hE : lies_on E CA) (hF : lies_on F AB)

theorem altitudes_condition_eq_area (h₁ : is_altitude AD BC) (h₂ : is_altitude BE CA) (h₃ : is_altitude CF AB) :
  area A B C = (R / 2) * (EF + FD + DE) :=
sorry

theorem area_eq_altitudes_condition (hS : area A B C = (R / 2) * (EF + FD + DE)) :
  is_altitude AD BC ∧ is_altitude BE CA ∧ is_altitude CF AB :=
sorry

theorem altitudes_iff_area : 
  (is_altitude AD BC ∧ is_altitude BE CA ∧ is_altitude CF AB) ↔ area A B C = (R / 2) * (EF + FD + DE) :=
⟨altitudes_condition_eq_area, area_eq_altitudes_condition⟩

end altitudes_condition_eq_area_area_eq_altitudes_condition_altitudes_iff_area_l135_135503


namespace appropriate_mass_units_l135_135761

def unit_of_mass_basket_of_eggs : String :=
  if 5 = 5 then "kilograms" else "unknown"

def unit_of_mass_honeybee : String :=
  if 5 = 5 then "grams" else "unknown"

def unit_of_mass_tank : String :=
  if 6 = 6 then "tons" else "unknown"

theorem appropriate_mass_units :
  unit_of_mass_basket_of_eggs = "kilograms" ∧
  unit_of_mass_honeybee = "grams" ∧
  unit_of_mass_tank = "tons" :=
by {
  -- skip the proof
  sorry
}

end appropriate_mass_units_l135_135761


namespace congruent_circumcircles_and_parallel_centers_line_l135_135597

variables {A B C H P Q E1 F1 E2 F2 O ω : Type*}
variables [geometry.on ω] [circumcircle A B C ω] [acute_triangle A B C]
variables (H_altitude_foot : altitude_foot A B C H)
variables (P_on_circle : on_circumcircle P ω) (Q_on_circle : on_circumcircle Q ω)
variables (PA_equals_PH : distance P A = distance P H)
variables (QA_equals_QH : distance Q A = distance Q H)
variables (E1_on_tangent : on_tangent_point P AC E1) (F1_on_tangent : on_tangent_point P AB F1)
variables (E2_on_tangent : on_tangent_point Q AC E2) (F2_on_tangent : on_tangent_point Q AB F2)

theorem congruent_circumcircles_and_parallel_centers_line :
  congruent_circumcircles (circumcircle A E1 F1) (circumcircle A E2 F2) ∧
  parallel (line_through_centers (circumcircle A E1 F1) (circumcircle A E2 F2)) (tangent_at ω A) :=
sorry

end congruent_circumcircles_and_parallel_centers_line_l135_135597


namespace triangle_area_eq_l135_135571

theorem triangle_area_eq :
  ∀ (A B C D E : Type)
    (area_BDE : ℝ)
    (hBE : ℝ) (hEC : ℝ) (hAD : ℝ) (hCD : ℝ),
    hEC = 2 * hBE → hCD = 2 * hAD →
    area_BDE = 14 →
    let area_ABC := 9 / 2 * area_BDE in
    area_ABC = 63
:=
begin
  intros,
  -- EC = 2 * BE, CD = 2 * AD, area(BDE) = 14 -> area(ABC) = 63
  sorry
end

end triangle_area_eq_l135_135571


namespace cos_double_angle_l135_135790

theorem cos_double_angle (α : ℝ) (h : tan α = 4 / 3) : cos (2 * α) = -7 / 25 := by
  sorry

end cos_double_angle_l135_135790


namespace weight_of_33rd_weight_l135_135184

theorem weight_of_33rd_weight :
  ∃ a : ℕ → ℕ, (∀ k, a k < a (k+1)) ∧
               (∀ k ≤ 29, a k + a (k+3) = a (k+1) + a (k+2)) ∧
               a 2 = 9 ∧
               a 8 = 33 ∧
               a 32 = 257 :=
sorry

end weight_of_33rd_weight_l135_135184


namespace symmetry_lines_le_points_l135_135283

theorem symmetry_lines_le_points {S : Type*} [fintype S] {P : Type*} [fintype P] {n m : ℕ}
  (hS : card S = n) (hP : card P = m)
  (h_points : n > 2)
  (h_sym : ∀ l ∈ P, ∃ (f : S ≃ S), f.symm l)
  : m ≤ n := 
sorry

end symmetry_lines_le_points_l135_135283


namespace problem_part1_problem_part2_l135_135936

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l135_135936


namespace part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135929

-- Definitions for the conditions in the problem
variable {A B C a b c : ℝ}

-- Given conditions and problem setup
axiom triangle_ABC_sides : ∀ {a b c : ℝ}, sides a b c
axiom triangle_ABC_angles : ∀ {A B C : ℝ}, angles A B C
axiom sin_relation : ∀ {A B C : ℝ},
  sin C * sin (A - B) = sin B * sin (C - A)

-- Prove Part (1): If A = 2B, then C = 5π/8
theorem part1_A_eq_2B_implies_C :
  A = 2 * B → C = 5 * π / 8 :=
by
  intro h
  sorry

-- Prove Part (2): 2a² = b² + c²
theorem part2_2a_squared_eq_b_squared_plus_c_squared :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
by
  sorry

end part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135929


namespace find_height_of_tank_A_l135_135628

noncomputable def height_of_tank_A (C_A C_B h_B ratio V_ratio : ℝ) : ℝ :=
  let r_A := C_A / (2 * Real.pi)
  let r_B := C_B / (2 * Real.pi)
  let V_A := Real.pi * (r_A ^ 2) * ratio
  let V_B := Real.pi * (r_B ^ 2) * h_B
  (V_ratio * V_B) / (Real.pi * (r_A ^ 2))

theorem find_height_of_tank_A :
  height_of_tank_A 8 10 8 10 0.8000000000000001 = 10 :=
by
  sorry

end find_height_of_tank_A_l135_135628


namespace distance_from_wall_to_picture_edge_l135_135710

theorem distance_from_wall_to_picture_edge
  (wall_width : ℕ)
  (picture_width : ℕ)
  (centered : Prop)
  (h1 : wall_width = 22)
  (h2 : picture_width = 4)
  (h3 : centered) :
  ∃ x : ℕ, x = 9 :=
by
  sorry

end distance_from_wall_to_picture_edge_l135_135710


namespace original_price_of_sarees_l135_135646

theorem original_price_of_sarees
  (P : ℝ)
  (h_sale_price : 0.80 * P * 0.85 = 306) :
  P = 450 :=
sorry

end original_price_of_sarees_l135_135646


namespace equation_of_line_AB_l135_135809

open Real

-- Define the curve C as y = x^2 / 2
def curve (x : ℝ) : ℝ := x^2 / 2

-- Define points A and B on the curve with the sum of their x-coordinates being 2
def A (x1 : ℝ) (hx1 : x1 ^ 2 /2 = y) : Prop := (0, x1 ) Integral  (y) /2
def B (x2 : ℝ) (hx2 : x2 ^ 2 / 2 = y) : Prop := (0, x2) Integral (y) /2
noncomputable def LineThrough (A B : ℝ):Slope:= (x-y) A =/-= x-y 1:2 asIncomplete Slope : Prop==2

-- State to prove the equation of the line AB
theorem equation_of_line_AB (x1 x2 : ℝ)
  (h_sum : x1 + x2 = 2) :
  ∃ (t : ℝ), line_through A B and the  sum and point of Source curve  align to  1 :
     ∃ h: eqation y = x + 7/2:
= sorry

end equation_of_line_AB_l135_135809


namespace one_non_congruent_triangle_with_perimeter_10_l135_135160

def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 10

def are_non_congruent (a b c : ℕ) (x y z : ℕ) : Prop :=
  ¬ (a = x ∧ b = y ∧ c = z ∨ a = x ∧ b = z ∧ c = y ∨ a = y ∧ b = x ∧ c = z ∨ 
     a = y ∧ b = z ∧ c = x ∨ a = z ∧ b = x ∧ c = y ∨ a = z ∧ b = y ∧ c = x)

theorem one_non_congruent_triangle_with_perimeter_10 :
  ∃ a b c : ℕ, is_valid_triangle a b c ∧ perimeter a b c ∧
  ∀ x y z : ℕ, is_valid_triangle x y z ∧ perimeter x y z → are_non_congruent a b c x y z → false :=
sorry

end one_non_congruent_triangle_with_perimeter_10_l135_135160


namespace fencing_required_l135_135408

-- Definitions based on problem conditions
def length (L : ℝ) : Prop := L = 30
def area (A : ℝ) : Prop := A = 600
def width (W : ℝ) : Prop := A = L * W

-- Theorem statement based on problem question and correct answer
theorem fencing_required (L A W : ℝ) (hL : length L) (hA : area A) (hW : width L W) :
  2 * W + L = 70 := 
  sorry

end fencing_required_l135_135408


namespace namjoon_rank_l135_135296

theorem namjoon_rank (total_students : ℕ) (fewer_than_namjoon : ℕ) (rank_of_namjoon : ℕ) 
  (h1 : total_students = 13) (h2 : fewer_than_namjoon = 4) : rank_of_namjoon = 9 :=
sorry

end namjoon_rank_l135_135296


namespace squared_difference_of_roots_l135_135166

theorem squared_difference_of_roots:
  ∀ (Φ φ : ℝ), (∀ x : ℝ, x^2 = 2*x + 1 ↔ (x = Φ ∨ x = φ)) ∧ Φ ≠ φ → (Φ - φ)^2 = 8 :=
by
  intros Φ φ h
  sorry

end squared_difference_of_roots_l135_135166


namespace Kimiko_age_proof_l135_135909

variables (Kimiko_age Kayla_age : ℕ)
variables (min_driving_age wait_years : ℕ)

def is_half_age (a b : ℕ) : Prop := a = b / 2
def minimum_driving_age (a b : ℕ) : Prop := a + b = 18

theorem Kimiko_age_proof
  (h1 : is_half_age Kayla_age Kimiko_age)
  (h2 : wait_years = 5)
  (h3 : minimum_driving_age Kayla_age wait_years) :
  Kimiko_age = 26 :=
sorry

end Kimiko_age_proof_l135_135909


namespace max_value_N_plus_a_b_c_d_e_l135_135984

variables {a b c d e : ℝ}
variables (h_cond : a^2 + b^2 + c^2 + d^2 + e^2 = 504)
variables (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)

def N := ac + 3bc + 4cd + 8ce
def a_N := a
def b_N := b
def c_N := c
def d_N := d
def e_N := e

theorem max_value_N_plus_a_b_c_d_e :
  N + a_N + b_N + c_N + d_N + e_N = 32 + 756 * real.sqrt 10 + 6 * real.sqrt 7 := sorry

end max_value_N_plus_a_b_c_d_e_l135_135984


namespace angle_F_is_110_l135_135226

-- Define the conditions
variable (p q : Line)
variable (E G F : Angle)
variable (parallel_pq : Parallel p q)
variable (angleE : m E = 100)
variable (angleG : m G = 70)

-- Prove the question
theorem angle_F_is_110 (parallel_pq : Parallel p q) (hE : m E = 100) (hG : m G = 70) :
  m F = 110 := 
  sorry

end angle_F_is_110_l135_135226


namespace age_of_cat_l135_135661

variables (cat_age rabbit_age dog_age : ℕ)

-- Conditions
def condition1 : Prop := rabbit_age = cat_age / 2
def condition2 : Prop := dog_age = 3 * rabbit_age
def condition3 : Prop := dog_age = 12

-- Question
def question (cat_age : ℕ) : Prop := cat_age = 8

theorem age_of_cat (h1 : condition1 cat_age rabbit_age) (h2 : condition2 rabbit_age dog_age) (h3 : condition3 dog_age) : question cat_age :=
by
  sorry

end age_of_cat_l135_135661


namespace integer_solution_pairs_l135_135161

theorem integer_solution_pairs (x y : ℤ) (h : x^4 + y^3 = 3y + 9) : 
  (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0) :=
sorry

end integer_solution_pairs_l135_135161


namespace num_integer_solutions_l135_135151

-- Definition of the equation's solution property
def is_solution (x : ℤ) : Prop :=
(x + 3) ^ (36 - x ^ 2) = 1

-- Stating the problem: proving the number of integer solutions
theorem num_integer_solutions : {x : ℤ | is_solution x}.to_finset.card = 4 :=
sorry

end num_integer_solutions_l135_135151


namespace derivative_of_composite_function_correct_l135_135272

noncomputable def derivative_of_composite_function (x : ℝ) : Prop :=
  let y := (sin (x^2))^3 in
  deriv (λ x, (sin (x^2))^3) x = 3 * x * sin (x^2) * sin (2 * x^2)

theorem derivative_of_composite_function_correct (x : ℝ) :
  derivative_of_composite_function x :=
by sorry

end derivative_of_composite_function_correct_l135_135272


namespace part1_solution_part2_solution_1_part2_solution_2_part2_solution_3_l135_135793

variable {x a : ℝ}

theorem part1_solution (h1 : a > 1 / 3) (h2 : (a * x - 1) / (x ^ 2 - 1) = 0) : x = 3 := by
  sorry

theorem part2_solution_1 (h1 : -1 < a) (h2 : a < 0) : {x | x < (1 / a) ∨ (-1 < x ∧ x < 1)} := by
  sorry

theorem part2_solution_2 (h1 : a = -1) : {x | x < 1 ∧ x ≠ -1} := by
  sorry

theorem part2_solution_3 (h1 : a < -1) : {x | x < -1 ∨ (1 / a < x ∧ x < 1)} := by
  sorry

end part1_solution_part2_solution_1_part2_solution_2_part2_solution_3_l135_135793


namespace one_non_congruent_triangle_with_perimeter_10_l135_135159

def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b

def perimeter (a b c : ℕ) : Prop :=
  a + b + c = 10

def are_non_congruent (a b c : ℕ) (x y z : ℕ) : Prop :=
  ¬ (a = x ∧ b = y ∧ c = z ∨ a = x ∧ b = z ∧ c = y ∨ a = y ∧ b = x ∧ c = z ∨ 
     a = y ∧ b = z ∧ c = x ∨ a = z ∧ b = x ∧ c = y ∨ a = z ∧ b = y ∧ c = x)

theorem one_non_congruent_triangle_with_perimeter_10 :
  ∃ a b c : ℕ, is_valid_triangle a b c ∧ perimeter a b c ∧
  ∀ x y z : ℕ, is_valid_triangle x y z ∧ perimeter x y z → are_non_congruent a b c x y z → false :=
sorry

end one_non_congruent_triangle_with_perimeter_10_l135_135159


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135344

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) ∧ 
            ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ (m.digits.sum = 27) → m ≤ n :=
begin
  use 999,
  split,
  { -- 999 is a three-digit number 
    norm_num,
  },
  split,
  { -- 999 is less than or equal to 999
    norm_num,
  },
  split,
  { -- 999 is a multiple of 9
    norm_num,
  },
  split,
  { -- The sum of the digits of 999 is 27
    norm_num,
  },
  { -- For any three-digit number m, if it is a multiple of 9 and the sum of its digits is 27, then m ≤ 999
    intros m hm1,
    cases hm1 with hm2 hm3,
    cases hm3 with hm4 hm5,
    exact le_of_lt (by linarith),
    sorry
  },
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135344


namespace maximum_equal_angles_three_intersecting_lines_l135_135660

noncomputable def max_equal_angles (angles : Finset ℝ) : ℕ :=
if H : ∀ (x ∈ angles), x ∈ {60, 120} then
  (angles.count 60).max (angles.count 120)
else
  0

theorem maximum_equal_angles_three_intersecting_lines :
  ∀ (L1 L2 L3 : ℝ → ℝ) (angles : Finset ℝ), 
  (angles.card = 12) ∧
  (∀ θ ∈ angles, θ ∈ {60, 120}) ∧ 
  (L1, L2, L3 form an equilateral triangle) → 
  max_equal_angles angles = 6 := 
by 
  sorry

end maximum_equal_angles_three_intersecting_lines_l135_135660


namespace log_equivalence_l135_135170

theorem log_equivalence (x : ℝ) (h : log 4 (x + 6) = 3) : log 9 x = log 9 58 :=
by sorry

end log_equivalence_l135_135170


namespace f_g_product_nonnegative_l135_135817

variable (f g : ℝ → ℝ)

-- Conditions on f
axiom f_increasing : ∀ x y : ℝ, x < y → f x < f y
axiom f_at_one : f 1 = 0

-- Conditions on g
axiom g_increasing : ∀ x y : ℝ, x ≤ 1 → y ≤ 1 → x < y → g x < g y
axiom g_decreasing : ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x < y → g x > g y
axiom g_at_zero : g 0 = 0
axiom g_at_four : g 4 = 0

-- Problem statement
theorem f_g_product_nonnegative : 
  { x : ℝ | f x * g x ≥ 0 } = { x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x ≤ 4) } :=
begin
  sorry -- Proof goes here
end

end f_g_product_nonnegative_l135_135817


namespace profit_ratio_7_10_l135_135282

noncomputable def profit_ratio (investment_ratio : ℕ × ℕ) (inv_period_p inv_period_q : ℕ) : ℕ × ℕ :=
let (inv_p, inv_q) := investment_ratio in
(inv_p * inv_period_p, inv_q * inv_period_q)

theorem profit_ratio_7_10 :
  profit_ratio (7, 5) 8 16 = (7, 10) :=
by
  sorry

end profit_ratio_7_10_l135_135282


namespace part1_part2_l135_135918

noncomputable def vec_m (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin (x / 4), 1)
noncomputable def vec_n (x : ℝ) : ℝ × ℝ := (cos (x / 4), cos (x / 4) ^ 2)
noncomputable def f (x : ℝ) : ℝ := (vec_m x).fst * (vec_n x).fst + (vec_m x).snd * (vec_n x).snd

theorem part1 : f π = (sqrt 3 + 1) / 2 := 
by {
  sorry
}

variables {A B C a b c : ℝ}
variable (triangle_condition : b * cos C + (1 / 2) * c = a)

theorem part2 : 
  (A + B + C = π) ∧ 
  (a = b * sin A / sin B) ∧
  (b = b * sin B / sin B) ∧
  (c = c * sin C / sin B) →
  B = π / 3 := 
by {
  sorry
}

end part1_part2_l135_135918


namespace solve_log5_eqn_l135_135254

theorem solve_log5_eqn (x : ℝ) (h : 7.74 * real.sqrt (real.log x / real.log 5) + (real.log x / real.log 5)^(1/3) = 2) : x = 5 :=
sorry

end solve_log5_eqn_l135_135254


namespace product_of_solutions_l135_135060

theorem product_of_solutions :
  let a := 2
  let b := 4
  let c := -6
  let discriminant := b^2 - 4*a*c
  ∃ (x₁ x₂ : ℝ), 2*x₁^2 + 4*x₁ - 6 = 0 ∧ 2*x₂^2 + 4*x₂ - 6 = 0 ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -3 :=
sorry

end product_of_solutions_l135_135060


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135364

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ 100 ≤ n ∧ n < 1000 ∧ (9 ∣ n) ∧ (∑ digit in n.digits, digit = 27) :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135364


namespace eliminate_t_from_system_l135_135608

theorem eliminate_t_from_system (
  t : ℝ 
): 
  let x := 1 + 2*t - 2*t^2 in
  let y := 2*(1 + t) * real.sqrt(1 - t^2) in
  y^4 + 2 * y^2 * (x^2 - 12 * x + 9) + x^4 + 8 * x^3 + 18 * x^2 - 27 = 0 :=
by
  sorry

end eliminate_t_from_system_l135_135608


namespace minimum_value_expression_l135_135524

theorem minimum_value_expression (a θ : ℝ) :
  ∃ a' ∈ ℝ, (a' - 2 * Real.cos θ) ^ 2 + (a' - 5 * Real.sqrt 2 - 2 * Real.sin θ) ^ 2 = 9 :=
sorry

end minimum_value_expression_l135_135524


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135336

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n % 9 = 0 ∧ (n.digits.sum = 27) ∧
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ m % 9 = 0 ∧ (m.digits.sum = 27) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135336


namespace second_number_is_90_l135_135379

theorem second_number_is_90 (x y z : ℕ) 
  (h1 : z = 4 * y) 
  (h2 : y = 2 * x) 
  (h3 : (x + y + z) / 3 = 165) : y = 90 := 
by
  sorry

end second_number_is_90_l135_135379


namespace remainder_of_division_l135_135366

open Polynomial -- Opening the mathematical namespace for polynomials

-- Defining the given polynomials and the result to be proven
def polynomial := 3 * X ^ 2 - 22 * X + 58
def divisor := X - 6
def remainder := 34

-- Stating the proof problem in Lean
theorem remainder_of_division :
  (polynomial % divisor) = remainder :=
sorry

end remainder_of_division_l135_135366


namespace quadratic_range_extrema_l135_135145

def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 2

theorem quadratic_range_extrema :
  let y := quadratic
  ∃ x_max x_min,
    (x_min = -2 ∧ y x_min = -2) ∧
    (x_max = -2 ∧ y x_max = 14 ∨ x_max = 5 ∧ y x_max = 7) := 
by
  sorry

end quadratic_range_extrema_l135_135145


namespace days_elapsed_l135_135427

theorem days_elapsed
  (initial_amount : ℕ)
  (daily_spending : ℕ)
  (total_savings : ℕ)
  (doubling_factor : ℕ)
  (additional_amount : ℕ)
  :
  initial_amount = 50 →
  daily_spending = 15 →
  doubling_factor = 2 →
  additional_amount = 10 →
  2 * (initial_amount - daily_spending) * total_savings + additional_amount = 500 →
  total_savings = 7 :=
by
  intros h_initial h_spending h_doubling h_additional h_total
  sorry

end days_elapsed_l135_135427


namespace frogs_finding_new_pond_l135_135557

theorem frogs_finding_new_pond :
  let initial_frogs : ℕ := 12
  let tadpole_multiplier : ℕ := 4
  let survival_rate : ℚ := 0.75
  let pond_capacity : ℕ := 20
  let total_tadpoles := tadpole_multiplier * initial_frogs
  let surviving_tadpoles := (survival_rate * total_tadpoles).to_nat
  let total_frogs := initial_frogs + surviving_tadpoles
  let frogs_needing_new_pond := total_frogs - pond_capacity
  frogs_needing_new_pond = 28 :=
by
  sorry

end frogs_finding_new_pond_l135_135557


namespace min_value_of_ratio_l135_135172

variable {z : ℂ}

def Re (z : ℂ) : ℝ := z.re

theorem min_value_of_ratio (hz : Re(z) ≠ 0) :
  ∀ z : ℂ, ∃ t : ℝ, t = (Re(z^4) / (Re(z)^4)) ∧ t ≥ -8 :=
begin
  sorry
end

end min_value_of_ratio_l135_135172


namespace problem_statement_l135_135842

-- Definitions for lines and planes
variables (Line Plane : Type) (m n l : Line) (α β γ : Plane)
variables [Parallelism Line Plane] [Perpendicularity Line Plane]

-- Given conditions
variable (cond1 : Parallel m n)
variable (cond2 : Perpendicular n γ)

-- The theorem to prove
theorem problem_statement : Perpendicular m γ :=
by sorry

end problem_statement_l135_135842


namespace solve_inequality_system_l135_135619

theorem solve_inequality_system :
  (∀ x : ℝ, (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 2 ≥ x)) →
  ∃ (integers : Set ℤ), integers = {x : ℤ | -2 < (x : ℝ) ∧ (x : ℝ) ≤ 1} ∧ integers = {-1, 0, 1} :=
by
  sorry

end solve_inequality_system_l135_135619


namespace find_integer_n_l135_135670

theorem find_integer_n : ∃ (n : ℤ), 0 ≤ n ∧ n < 9 ∧ (-1234) % 9 = n % 9 ∧ n = 8 :=
by {
  use 8,
  split,
  linarith,
  split,
  linarith,
  split,
  norm_num,
  unfold dvd,
  existsi (-137),
  norm_num,
  refl,
  refl,
}

end find_integer_n_l135_135670


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135347

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) ∧ 
            ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ (m.digits.sum = 27) → m ≤ n :=
begin
  use 999,
  split,
  { -- 999 is a three-digit number 
    norm_num,
  },
  split,
  { -- 999 is less than or equal to 999
    norm_num,
  },
  split,
  { -- 999 is a multiple of 9
    norm_num,
  },
  split,
  { -- The sum of the digits of 999 is 27
    norm_num,
  },
  { -- For any three-digit number m, if it is a multiple of 9 and the sum of its digits is 27, then m ≤ 999
    intros m hm1,
    cases hm1 with hm2 hm3,
    cases hm3 with hm4 hm5,
    exact le_of_lt (by linarith),
    sorry
  },
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135347


namespace safe_ice_time_l135_135385

-- Define given conditions
def t0 : ℝ := -20 -- air temperature in Celsius
def tB : ℝ := 0   -- water temperature in Celsius
def heat_transfer_rate : ℝ := 300 -- kJ/h per square meter
def λ : ℝ := 330 -- latent heat of fusion in kJ/kg
def c : ℝ := 2100 / 1000 -- specific heat capacity in kJ/(kg·C)
def ρ : ℝ := 900 -- density of ice in kg/m^3
def h : ℝ := 0.1 -- desired ice thickness in meters
def S : ℝ := 1 -- area in square meters

-- Compute mass per square meter of ice
def m : ℝ := ρ * S * h

-- Calculate the heat required to freeze and cool the ice
def Q_total : ℝ := (λ * m) + (c * m * (0 - t0))

-- Time required to form safe thickness of ice
def t : ℝ := Q_total / heat_transfer_rate

theorem safe_ice_time
  (h1 : t0 = -20)
  (h2 : tB = 0)
  (h3 : heat_transfer_rate = 300)
  (h4 : λ = 330)
  (h5 : c = 2.1)
  (h6 : ρ = 900)
  (h7 : h = 0.1)
  (h8 : S = 1) :
  t = 105.3 := by
  -- The proof goes here
  sorry

end safe_ice_time_l135_135385


namespace geometric_sequence_product_l135_135896

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_a5 : a 5 = 2) : a 1 * a 2 * a 3 * a 7 * a 8 * a 9 = 64 :=
by
  -- using the fact that the sequence is geometric
  have h1 : a 5 = a 1 * (r ^ 4) := sorry,
  have h2 : a 1 * r ^ 4 = 2 := h_a5,
  -- sorry {
  --   -- proceed with the proof using provided steps
  --   sorry
  -- },
  sorry

end geometric_sequence_product_l135_135896


namespace triangle_side_length_l135_135575

theorem triangle_side_length (A B : ℝ) (b : ℝ) (a : ℝ) 
  (hA : A = 60) (hB : B = 45) (hb : b = 2) 
  (h : a = b * (Real.sin A) / (Real.sin B)) :
  a = Real.sqrt 6 := by
  sorry

end triangle_side_length_l135_135575


namespace hortense_flower_production_l135_135532

-- Define the initial conditions
def daisy_seeds : ℕ := 25
def sunflower_seeds : ℕ := 25
def daisy_germination_rate : ℚ := 0.60
def sunflower_germination_rate : ℚ := 0.80
def flower_production_rate : ℚ := 0.80

-- Prove the number of plants that produce flowers
theorem hortense_flower_production :
  (daisy_germination_rate * daisy_seeds + sunflower_germination_rate * sunflower_seeds) * flower_production_rate = 28 :=
by sorry

end hortense_flower_production_l135_135532


namespace lines_are_skew_iff_l135_135073

-- We define vectors and handle the calculations in the vector space
open Matrix

def line1 (b : ℝ) (s : ℝ) : Vector3 ℝ := ⟨ [2 + 3*s, 3 + 4*s, b + 5*s] ⟩
def line2 (v : ℝ) : Vector3 ℝ := ⟨ [5 + 6*v, 2 + 3*v, 1 + 2*v] ⟩

theorem lines_are_skew_iff (b : ℝ) : 
  ¬ ∃ (s v : ℝ), line1 b s = line2 v ↔ b ≠ 4 := 
sorry

end lines_are_skew_iff_l135_135073


namespace non_black_cows_l135_135244

-- Define the main problem conditions
def total_cows : ℕ := 18
def black_cows : ℕ := (total_cows / 2) + 5

-- Statement to prove the number of non-black cows
theorem non_black_cows :
  total_cows - black_cows = 4 :=
by
  sorry

end non_black_cows_l135_135244


namespace triangle_inequality_l135_135919

variables {α : Type*} [linear_ordered_field α]

theorem triangle_inequality
  {a b c PA PB PC : α}
  (hab : a > 0)
  (hbc : b > 0)
  (hca : c > 0)
  (P_in_plane : true) : -- This condition ensures P is any point in the plane
  (PB * PC / (b * c) + PC * PA / (c * a) + PA * PB / (a * b) ≥ 1) :=
sorry

end triangle_inequality_l135_135919


namespace perimeter_of_quadrilateral_l135_135642

def ellipse_vertices (x y ℝ) : Prop :=
  (x = 2 ∨ x = -2 ∨ x = 0) ∧ (y = 4 ∨ y = -4 ∨ y = 0)

theorem perimeter_of_quadrilateral :
  (∀ x y, ellipse_vertices x y → x^2 / 4 + y^2 / 16 = 1) →
  (distance (2, 0) (0, 4) + distance (0, 4) (-2, 0) + distance (-2, 0) (0, -4) + distance (0, -4) (2, 0) = 8 * sqrt 5) :=
by
  intros h
  sorry

end perimeter_of_quadrilateral_l135_135642


namespace square_side_length_inscribed_in_hexagon_l135_135024

-- Defining the conditions
def AB : ℝ := 50
def EF : ℝ := 50 * (Real.sqrt 3 - 2)

-- The proof statement
theorem square_side_length_inscribed_in_hexagon :
  ∃ x : ℝ, 
    (∀ A B C D E F P Q R S : point, -- Assuming point is adequately defined in the context of Mathlib or otherwise.
    equilateral_hexagon A B C D E F 
    ∧ A.distance B = AB
    ∧ E.distance F = EF
    ∧ square_inscribed_in_hexagon P Q R S A B C D E F) 
    → (x = (25 * Real.sqrt 3 - 50 / Real.sqrt 3) / ((3 * Real.sqrt 3 + 6) / 2)) :=
sorry

end square_side_length_inscribed_in_hexagon_l135_135024


namespace valid_sequences_l135_135703

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

-- Define the number of valid sequences
def a (n : ℕ) : ℕ :=
if n = 0 then 0 else if n = 1 then 0 else if n = 2 then 1 else fibonacci (n - 1)

-- The theorem to prove
theorem valid_sequences (n : ℕ) (hn : n > 0) : a n = fibonacci (n - 1) :=
by {
  sorry
}

end valid_sequences_l135_135703


namespace find_b_l135_135051

def h (x : ℝ) : ℝ := 5 * x + 7

theorem find_b (b : ℝ) : h b = 0 ↔ b = -7 / 5 := by
  sorry

end find_b_l135_135051


namespace part1_part2_l135_135967

variable {A B C a b c : ℝ}

theorem part1 (h₁ : A = 2 * B) (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : C = 5 / 8 * π :=
  sorry

theorem part2 (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
  sorry

end part1_part2_l135_135967


namespace part1_C_value_part2_triangle_equality_l135_135978

noncomputable theory

variables (a b c : ℝ) (A B C : ℝ)
variables (h1 : A + B + C = Real.pi) (h2 : sin C * sin (A - B) = sin B * sin (C - A)) (h3 : A = 2 * B)

-- Part 1: Proving that C = 5π/8 given the conditions
theorem part1_C_value :
  C = 5 * Real.pi / 8 :=
begin
  sorry
end

-- Part 2: Proving that 2a^2 = b^2 + c^2 given the conditions
theorem part2_triangle_equality :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
begin
  sorry
end

end part1_C_value_part2_triangle_equality_l135_135978


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135337

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), (100 ≤ n ∧ n < 1000) ∧ n % 9 = 0 ∧ (n.digits.sum = 27) ∧
  ∀ (m : ℕ), (100 ≤ m ∧ m < 1000) ∧ m % 9 = 0 ∧ (m.digits.sum = 27) → m ≤ n :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135337


namespace number_of_solutions_l135_135852

theorem number_of_solutions :
  {x : ℝ | |x - 1| = |x - 2| + |x - 3|}.finite.to_finset.card = 2 :=
sorry

end number_of_solutions_l135_135852


namespace sin_arithmetic_sequence_l135_135767

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 2 * Real.pi) : 
  (Real.sin a + Real.sin (3 * a) = 2 * Real.sin (2 * a)) ↔ (a = Real.pi) :=
sorry

end sin_arithmetic_sequence_l135_135767


namespace not_suitable_for_storing_l135_135288

-- Define the acceptable temperature range conditions for storing dumplings
def acceptable_range (t : ℤ) : Prop :=
  -20 ≤ t ∧ t ≤ -16

-- Define the specific temperatures under consideration
def temp_A : ℤ := -17
def temp_B : ℤ := -18
def temp_C : ℤ := -19
def temp_D : ℤ := -22

-- Define a theorem stating that temp_D is not in the acceptable range
theorem not_suitable_for_storing (t : ℤ) (h : t = temp_D) : ¬ acceptable_range t :=
by {
  sorry
}

end not_suitable_for_storing_l135_135288


namespace factor_expression_l135_135458

theorem factor_expression (x : ℝ) : 
  4 * x * (x - 5) + 6 * (x - 5) = (4 * x + 6) * (x - 5) :=
by 
  sorry

end factor_expression_l135_135458


namespace quadratic_inequality_solution_l135_135142

theorem quadratic_inequality_solution 
  (x : ℝ) (b c : ℝ)
  (h : ∀ x, -x^2 + b*x + c < 0 ↔ x < -3 ∨ x > 2) :
  (6 * x^2 + x - 1 > 0) ↔ (x < -1/2 ∨ x > 1/3) := 
sorry

end quadratic_inequality_solution_l135_135142


namespace num_sheets_in_14_inch_stack_l135_135010

def thickness_per_sheet : Real := 4 / 400
def height_of_stack_cm : Real := 14 * 2.54
def number_of_sheets (height_of_stack: Real) (thickness_sheet: Real) : Real :=
  height_of_stack / thickness_sheet

theorem num_sheets_in_14_inch_stack :
  number_of_sheets height_of_stack_cm thickness_per_sheet = 3556 :=
by
  sorry

end num_sheets_in_14_inch_stack_l135_135010


namespace number_of_subsets_of_M_l135_135838

theorem number_of_subsets_of_M {P : set ℕ} (hP : P = {0, 1}) (M : set (set ℕ)) (hM : M = {x | x ⊆ P}) : 
  ∃ n : ℕ, n = 16 ∧ ∃ S : set (set (set ℕ)), S = set.powerset M ∧ S.card = n :=
sorry

end number_of_subsets_of_M_l135_135838


namespace parabola_trajectory_of_P_l135_135275

-- Define the point F
def F : ℝ × ℝ := (2, 0)

-- Define the line equation x + 2 = 0, i.e., x = -2
def line (p : ℝ × ℝ) : Prop := p.1 = -2

-- Define the property that a point P is equidistant from F and the line x = -2
def is_equidistant (P : ℝ × ℝ) : Prop :=
  real.dist P F = real.dist P (P.1, 0)

-- Define the trajectory of P which we need to prove is a parabola y^2 = 8x
def trajectory (P : ℝ × ℝ) : Prop := P.2 ^ 2 = 8 * P.1

-- The theorem statement in Lean 4
theorem parabola_trajectory_of_P (P : ℝ × ℝ) (h : is_equidistant P) : trajectory P :=
sorry

end parabola_trajectory_of_P_l135_135275


namespace unique_line_through_A_parallel_to_a_l135_135823

variables {Point Line Plane : Type}
variables {α β : Plane}
variables {a l : Line}
variables {A : Point}

-- Definitions are necessary from conditions in step a)
def parallel_to (a b : Line) : Prop := sorry -- Definition that two lines are parallel
def contains (p : Plane) (x : Point) : Prop := sorry -- Definition that a plane contains a point
def line_parallel_to_plane (a : Line) (p : Plane) : Prop := sorry -- Definition that a line is parallel to a plane

-- Given conditions in the proof problem
variable (a_parallel_α : line_parallel_to_plane a α)
variable (A_in_α : contains α A)

-- Statement to be proven: There is only one line that passes through point A and is parallel to line a, and that line is within plane α.
theorem unique_line_through_A_parallel_to_a : 
  ∃! l : Line, contains α A ∧ parallel_to l a := sorry

end unique_line_through_A_parallel_to_a_l135_135823


namespace problem_statement_l135_135229

theorem problem_statement (a b : ℝ) :
  a^2 + b^2 - a - b - a * b + 0.25 ≥ 0 ∧ (a^2 + b^2 - a - b - a * b + 0.25 = 0 ↔ ((a = 0 ∧ b = 0.5) ∨ (a = 0.5 ∧ b = 0))) :=
by 
  sorry

end problem_statement_l135_135229


namespace conference_engineers_median_l135_135002

theorem conference_engineers_median :
  ∀ (engineers : Finset ℕ), 
  engineers = (Finset.range 26) \ {15, 16, 17} →
  Finset.card engineers = 22 →
  let room_numbers := (engineers.sort Nat.le) in
  let median := (room_numbers.nth_le 10 sorry + room_numbers.nth_le 11 sorry) / 2 →
  median = 11.5 :=
by
  intros engineers h_eq h_card room_numbers median
  sorry

end conference_engineers_median_l135_135002


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135346

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 9 = 0) ∧ (n.digits.sum = 27) ∧ 
            ∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 9 = 0) ∧ (m.digits.sum = 27) → m ≤ n :=
begin
  use 999,
  split,
  { -- 999 is a three-digit number 
    norm_num,
  },
  split,
  { -- 999 is less than or equal to 999
    norm_num,
  },
  split,
  { -- 999 is a multiple of 9
    norm_num,
  },
  split,
  { -- The sum of the digits of 999 is 27
    norm_num,
  },
  { -- For any three-digit number m, if it is a multiple of 9 and the sum of its digits is 27, then m ≤ 999
    intros m hm1,
    cases hm1 with hm2 hm3,
    cases hm3 with hm4 hm5,
    exact le_of_lt (by linarith),
    sorry
  },
end

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135346


namespace minimum_shift_value_lemma_l135_135639

noncomputable def minimum_shift_value (k : ℤ) (φ : ℝ) : ℝ := (\pi / 3) - 2 * k * \pi

theorem minimum_shift_value_lemma {φ : ℝ} (h1 : φ > 0) :
  ∃ k : ℤ, φ = minimum_shift_value k ∧ φ = \pi / 3 :=
by 
  sorry

end minimum_shift_value_lemma_l135_135639


namespace part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135933

-- Definitions for the conditions in the problem
variable {A B C a b c : ℝ}

-- Given conditions and problem setup
axiom triangle_ABC_sides : ∀ {a b c : ℝ}, sides a b c
axiom triangle_ABC_angles : ∀ {A B C : ℝ}, angles A B C
axiom sin_relation : ∀ {A B C : ℝ},
  sin C * sin (A - B) = sin B * sin (C - A)

-- Prove Part (1): If A = 2B, then C = 5π/8
theorem part1_A_eq_2B_implies_C :
  A = 2 * B → C = 5 * π / 8 :=
by
  intro h
  sorry

-- Prove Part (2): 2a² = b² + c²
theorem part2_2a_squared_eq_b_squared_plus_c_squared :
  2 * a ^ 2 = b ^ 2 + c ^ 2 :=
by
  sorry

end part1_A_eq_2B_implies_C_part2_2a_squared_eq_b_squared_plus_c_squared_l135_135933


namespace donuts_count_is_correct_l135_135738

-- Define the initial number of donuts
def initial_donuts : ℕ := 50

-- Define the number of donuts Bill eats
def eaten_by_bill : ℕ := 2

-- Define the number of donuts taken by the secretary
def taken_by_secretary : ℕ := 4

-- Calculate the remaining donuts after Bill and the secretary take their portions
def remaining_after_bill_and_secretary : ℕ := initial_donuts - eaten_by_bill - taken_by_secretary

-- Define the number of donuts stolen by coworkers (half of the remaining donuts)
def stolen_by_coworkers : ℕ := remaining_after_bill_and_secretary / 2

-- Define the number of donuts left for the meeting
def donuts_left_for_meeting : ℕ := remaining_after_bill_and_secretary - stolen_by_coworkers

-- The theorem to prove
theorem donuts_count_is_correct : donuts_left_for_meeting = 22 :=
by
  sorry

end donuts_count_is_correct_l135_135738


namespace omega_range_l135_135589

theorem omega_range (ω : ℝ) (a b : ℝ) (hω_pos : ω > 0) (h_range : π ≤ a ∧ a < b ∧ b ≤ 2 * π)
  (h_sin : Real.sin (ω * a) + Real.sin (ω * b) = 2) :
  ω ∈ Set.Icc (9 / 4 : ℝ) (5 / 2) ∪ Set.Ici (13 / 4) :=
by
  sorry

end omega_range_l135_135589


namespace PB_is_sqrt10_l135_135388

noncomputable def findPB (P A B C D : ℝ × ℝ) (PA PD PC AB : ℝ) : ℝ :=
  let PB := dist P B
  have cond1 : dist P A = PA := by simp [PA]
  have cond2 : dist P D = PD := by simp [PD]
  have cond3 : dist P C = PC := by simp [PC]
  have cond4 : dist A B = AB := by simp [AB]
  PB

theorem PB_is_sqrt10
  (P A B C D : ℝ × ℝ)
  (PA : dist P A = 5)
  (PD : dist P D = 7)
  (PC : dist P C = 8)
  (AB : dist A B = 9) :
  dist P B = Real.sqrt 10 := by
  sorry

end PB_is_sqrt10_l135_135388


namespace vector_sum_is_correct_l135_135435

def v1 : ℝ × ℝ := (5, -3)
def v2 : ℝ × ℝ := (-4, 9)
def s : ℝ := 2

theorem vector_sum_is_correct : v1 + (s • v2) = (-3, 15) :=
by
  unfold v1 v2 s
  simp
  done

end vector_sum_is_correct_l135_135435


namespace smallest_m_in_interval_l135_135751

def z (n : ℕ) : ℚ 
| 0       := 3
| (n + 1) := (2 * z n^2 + 3 * z n + 6) / (z n + 8)

theorem smallest_m_in_interval :
  ∃ m : ℕ, (z m ≤ 2 + 1/2^10) ∧ (27 ≤ m) ∧ (m ≤ 80) :=
sorry

end smallest_m_in_interval_l135_135751


namespace arithmetic_series_product_l135_135223

theorem arithmetic_series_product (a b c : ℝ) (h1 : a = b - d) (h2 : c = b + d) (h3 : a * b * c = 125) (h4 : 0 < a) (h5 : 0 < b) (h6 : 0 < c) : b ≥ 5 :=
sorry

end arithmetic_series_product_l135_135223


namespace interior_diagonal_length_l135_135650

theorem interior_diagonal_length (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 26)
  (h2 : 4 * (a + b + c) = 28) : 
  (a^2 + b^2 + c^2) = 23 :=
by
  sorry

end interior_diagonal_length_l135_135650


namespace part1_part2_l135_135966

variable {A B C a b c : ℝ}

theorem part1 (h₁ : A = 2 * B) (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : C = 5 / 8 * π :=
  sorry

theorem part2 (h₂ : sin C * sin (A - B) = sin B * sin (C - A)) : 2 * a^2 = b^2 + c^2 :=
  sorry

end part1_part2_l135_135966


namespace maximize_area_minimize_length_l135_135017

-- Problem 1: Prove maximum area of the enclosure
theorem maximize_area (x y : ℝ) (h : x + 2 * y = 36) : 18 * 9 = 162 :=
by
  sorry

-- Problem 2: Prove the minimum length of steel wire mesh
theorem minimize_length (x y : ℝ) (h1 : x * y = 32) : 8 + 2 * 4 = 16 :=
by
  sorry

end maximize_area_minimize_length_l135_135017


namespace circle_area_l135_135127

-- Definitions for points, lines, and circles
def point (x y : ℝ) := (x, y)
def line (A B C : ℝ) (p : point) := A * p.1 + B * p.2 + C = 0
def circle (R : ℝ) (p : point) := p.1^2 + p.2^2 = R^2

-- Conditions in Lean definitions
def P := point 2 1
def Q := point 1 (-1)
def chord_length := 6 * real.sqrt 5 / 5

theorem circle_area (R : ℝ) (hR : R > 0) : 
  (∃ (A B C : ℝ), line A B C P ∧ A * 1 + B * (-1) + C = 0 ∧ ∃ (M N K : ℝ), line M N K P ∧ M * 2 + N * 1 + K = 0 ∧ M * (1 / 2) + N * (1 / -2) + K / 2 = 0 ∧ (2 * (N * N + 4) = K * K)) →
  ∃ (area : ℝ), area = 5 * real.pi := 
sorry

end circle_area_l135_135127


namespace tunnel_volume_correct_l135_135725

-- Definitions based on the conditions
def top_width : ℝ := 15
def bottom_width : ℝ := 5
def cross_section_area : ℝ := 400
def tunnel_length : ℝ := 300

-- Define the height calculation based on the area of a trapezoid
def trapezoid_height (a b A : ℝ) : ℝ := (2 * A) / (a + b)

-- Calculate the volume of the tunnel
def tunnel_volume (A L : ℝ) : ℝ := A * L

-- The statement to prove
theorem tunnel_volume_correct :
  tunnel_volume cross_section_area tunnel_length = 120000 :=
by
  sorry

end tunnel_volume_correct_l135_135725


namespace ellipse_properties_l135_135490

theorem ellipse_properties :
  let C : set (ℝ × ℝ) := {p | (p.1^2) / 4 + (p.2^2) / 3 = 1}
  (A : ℝ × ℝ) (f1 f2 : ℝ × ℝ),
  A = (1, 3 / 2) ∧ f1 = (-1, 0) ∧ f2 = (1, 0) →
  (C A := A ∈ C) ∧ 
  (∀ (E F : ℝ × ℝ), 
    E ∈ C → F ∈ C →
    (∃ k : ℝ, 
      E.2 - A.2 = k * (E.1 - A.1) ∧
      F.2 - A.2 = (-1 / k) * (F.1 - A.1)) →
      (let slope_EF := (F.2 - E.2) / (F.1 - E.1) in
        slope_EF = 1 / 2))
  :=
by
  sorry

end ellipse_properties_l135_135490


namespace eight_points_in_circle_distance_lt_one_l135_135205

noncomputable theory

open set
open metric

def circle : set (euclidean_space (fin 2)) :=
  {p | ∥p∥ ≤ 1}

theorem eight_points_in_circle_distance_lt_one
  (ps : fin 8 → euclidean_space (fin 2))
  (hps : ∀ i, ps i ∈ circle):
  ∃ (i j : fin 8), i ≠ j ∧ dist (ps i) (ps j) < 1 :=
sorry

end eight_points_in_circle_distance_lt_one_l135_135205


namespace russell_oranges_taken_l135_135847

-- Conditions
def initial_oranges : ℕ := 60
def oranges_left : ℕ := 25

-- Statement to prove
theorem russell_oranges_taken : ℕ :=
  initial_oranges - oranges_left = 35

end russell_oranges_taken_l135_135847


namespace probability_of_snow_two_days_l135_135645

noncomputable def probability_of_snow : ℚ := 2 / 3

theorem probability_of_snow_two_days :
  let p := probability_of_snow in
  let q := 1 - p in
  (p * p * q + p * q * p + q * p * p) = 4 / 9 :=
by
  sorry

end probability_of_snow_two_days_l135_135645


namespace total_savings_eighteen_l135_135474

theorem total_savings_eighteen :
  let fox_price := 15
  let pony_price := 18
  let discount_rate_sum := 50
  let fox_quantity := 3
  let pony_quantity := 2
  let pony_discount_rate := 50
  let total_price_without_discount := (fox_quantity * fox_price) + (pony_quantity * pony_price)
  let discounted_pony_price := (pony_price * (1 - (pony_discount_rate / 100)))
  let total_price_with_discount := (fox_quantity * fox_price) + (pony_quantity * discounted_pony_price)
  let total_savings := total_price_without_discount - total_price_with_discount
  total_savings = 18 :=
by sorry

end total_savings_eighteen_l135_135474


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135359

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 : 
  ∃ n : ℕ, n = 999 ∧ 100 ≤ n ∧ n < 1000 ∧ (9 ∣ n) ∧ (∑ digit in n.digits, digit = 27) :=
sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l135_135359


namespace gcd_euclidean_algorithm_division_steps_l135_135432

theorem gcd_euclidean_algorithm_division_steps {a b : ℕ} (h₁: a = 60) (h₂: b = 48) : 
  nat.gcd 60 48 = 12 ∧ (euclidean_domain.gcd_a 60 48).fst = 2 := by 
  sorry

end gcd_euclidean_algorithm_division_steps_l135_135432


namespace surface_area_ratio_correct_l135_135804

noncomputable def tetrahedron_edge_length (a : ℝ) := a
noncomputable def tetrahedron_surface_area (a : ℝ) := 4 * (sqrt 3 / 4) * a^2
noncomputable def inscribed_sphere_radius (a : ℝ) := (1 / 4) * (sqrt 6 / 3) * a
noncomputable def inscribed_sphere_surface_area (a : ℝ) := 4 * π * (inscribed_sphere_radius a)^2
noncomputable def surface_area_ratio (a : ℝ) := (tetrahedron_surface_area a) / (inscribed_sphere_surface_area a)

theorem surface_area_ratio_correct (a : ℝ) : surface_area_ratio a = 6 * sqrt 3 / π :=
  by
    sorry

end surface_area_ratio_correct_l135_135804


namespace length_four_implies_value_twenty_four_l135_135782

-- Definition of prime factors of an integer
def prime_factors (n : ℕ) : List ℕ := sorry

-- Definition of the length of an integer
def length_of_integer (n : ℕ) : ℕ :=
  List.length (prime_factors n)

-- Statement of the problem
theorem length_four_implies_value_twenty_four (k : ℕ) (h1 : k > 1) (h2 : length_of_integer k = 4) : k = 24 :=
by
  sorry

end length_four_implies_value_twenty_four_l135_135782


namespace find_t_l135_135542

theorem find_t (s t : ℝ) (h1 : 15 * s + 7 * t = 236) (h2 : t = 2 * s + 1) : t = 16.793 :=
by
  sorry

end find_t_l135_135542


namespace trapezoid_triangle_area_ratio_l135_135200

/-- Given a trapezoid with triangles ABC and ADC such that the ratio of their areas is 4:1 and AB + CD = 150 cm.
Prove that the length of segment AB is 120 cm. --/
theorem trapezoid_triangle_area_ratio
  (h ABC_area ADC_area : ℕ)
  (AB CD : ℕ)
  (h_ratio : ABC_area / ADC_area = 4)
  (area_ABC : ABC_area = AB * h / 2)
  (area_ADC : ADC_area = CD * h / 2)
  (h_sum : AB + CD = 150) :
  AB = 120 := 
sorry

end trapezoid_triangle_area_ratio_l135_135200


namespace groupDivisionWays_l135_135266

-- Definitions based on conditions
def numDogs : ℕ := 12
def group1Size : ℕ := 4
def group2Size : ℕ := 5
def group3Size : ℕ := 3
def fluffy : ℕ := 1 -- Fluffy's assigned position
def nipper : ℕ := 2 -- Nipper's assigned position

-- Function to compute binomial coefficients
def binom : ℕ → ℕ → ℕ
| n, 0 => 1
| 0, k => 0
| n+1, k+1 => binom n k + binom (n+1) k

-- Theorem to prove the number of ways to form the groups
theorem groupDivisionWays :
  (binom 10 3 * binom 7 4) = 4200 :=
by
  sorry

end groupDivisionWays_l135_135266


namespace students_taking_neither_l135_135874

theorem students_taking_neither (total_students : ℕ)
    (students_math : ℕ) (students_physics : ℕ) (students_chemistry : ℕ)
    (students_math_physics : ℕ) (students_physics_chemistry : ℕ) (students_math_chemistry : ℕ)
    (students_all_three : ℕ) :
    total_students = 60 →
    students_math = 40 →
    students_physics = 30 →
    students_chemistry = 25 →
    students_math_physics = 18 →
    students_physics_chemistry = 10 →
    students_math_chemistry = 12 →
    students_all_three = 5 →
    (total_students - (students_math + students_physics + students_chemistry - students_math_physics - students_physics_chemistry - students_math_chemistry + students_all_three)) = 5 :=
by
  intros
  sorry

end students_taking_neither_l135_135874
