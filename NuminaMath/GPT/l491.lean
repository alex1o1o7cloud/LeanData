import Lean
import Mathlib
import Mathlib.Algebra.AbsoluteValue
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Order.Ring
import Mathlib.Algebra.Parity
import Mathlib.Algebra.Polynomials
import Mathlib.Algebra.Prime
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecificLimits
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Nat.Primes
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Sequence.Basic
import Mathlib.Data.Set.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.Tactic
import Mathlib.Tactic.LibrarySearch
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace f_inequality_l491_491814

-- Define the function f.
def f (x : ℝ) : ℝ := x^2 - x + 13

-- The main theorem to prove the given inequality.
theorem f_inequality (x m : ℝ) (h : |x - m| < 1) : |f x - f m| < 2*(|m| + 1) :=
by
  sorry

end f_inequality_l491_491814


namespace water_added_l491_491671

theorem water_added (W x : ℕ) (h₁ : 2 * W = 5 * 10)
                    (h₂ : 2 * (W + x) = 7 * 10) :
  x = 10 :=
by
  sorry

end water_added_l491_491671


namespace part_i_l491_491453

def f1 (x : ℝ) : ℝ := |2 * x + 1/2| - |x - 3/2|

theorem part_i (x : ℝ) : f1 x ≤ 3 * x ↔ x ≥ -1/2 :=
by sorry

end part_i_l491_491453


namespace total_cans_collected_all_classes_l491_491192

-- Define the number of students in each class
def students_perez : Nat := 28
def students_johnson : Nat := 30
def students_smith : Nat := 28

-- Define the conditions for Ms. Perez's class
def perez_half_students_collected (n : Nat) : Nat := n / 2
def perez_remaining_students_collected : Nat := 12
def perez_half_collection : Nat := 20 * (perez_half_students_collected students_perez)
def perez_remaining_collection : Nat := 8 * perez_remaining_students_collected
def perez_total_collection : Nat := perez_half_collection + perez_remaining_collection

-- Define the conditions for Mr. Johnson's class
def johnson_third_students_collected (n : Nat) : Nat := n / 3
def johnson_remaining_students_collected : Nat := 18
def johnson_third_collection : Nat := 25 * (johnson_third_students_collected students_johnson)
def johnson_remaining_collection : Nat := 10 * johnson_remaining_students_collected
def johnson_total_collection : Nat := johnson_third_collection + johnson_remaining_collection

-- Define the conditions for Mrs. Smith's class
def smith_quarter_students_collected (n : Nat) : Nat := n / 4
def smith_remaining_students_collected : Nat := 16
def smith_quarter_collection : Nat := 30 * (smith_quarter_students_collected students_smith)
def smith_remaining_collection_1 : Nat := 4 * 0.5
def smith_remaining_collection_2 : Nat := 15 * smith_remaining_students_collected
def smith_total_collection : Nat := smith_quarter_collection + smith_remaining_collection_1 + smith_remaining_collection_2

-- Total collection across all classes
def total_collection : Nat := perez_total_collection + johnson_total_collection + smith_total_collection

-- Proof statement
theorem total_cans_collected_all_classes : total_collection = 1258 := by
  sorry

end total_cans_collected_all_classes_l491_491192


namespace sum_of_non_perfect_squares_l491_491239

theorem sum_of_non_perfect_squares:
  let perfect_squares := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n ≤ 300 }
  let sum_natural_300 := (300 * 301) / 2
  let sum_perfect_squares := ∑ n in finset.filter (λ n, n ≤ 300) {n | ∃ k, k ^ 2 = n}, n
  let sum_non_perfect_squares := sum_natural_300 - sum_perfect_squares
  in sum_non_perfect_squares = 43365 :=
begin
  sorry
end

end sum_of_non_perfect_squares_l491_491239


namespace simplify_quotient_l491_491416

theorem simplify_quotient (n : ℕ) (h : 0 < n) :
  let a_n := ∑ k in Finset.range (n + 1), (1 / Nat.choose n k)
  let b_n := ∑ k in Finset.range (n + 1), (k^2 / Nat.choose n k)
  (a_n / b_n) = (2 / n^2) :=
by
  sorry

end simplify_quotient_l491_491416


namespace workman_problem_l491_491269

theorem workman_problem (A B : ℝ) (h1 : A = B / 2) (h2 : (A + B) * 10 = 1) : B = 1 / 15 := by
  sorry

end workman_problem_l491_491269


namespace find_value_l491_491774

theorem find_value (a b c : ℝ) (h1 : a + b = 8) (h2 : a * b = c^2 + 16) : a + 2 * b + 3 * c = 12 := by
  sorry

end find_value_l491_491774


namespace compare_negative_fractions_l491_491358

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l491_491358


namespace derivative_of_y_l491_491603

-- Defining the function y
def y (x : ℝ) := 3 * sin (2 * x - π / 6)

-- Stating the theorem
theorem derivative_of_y :
  deriv (λ x, y x) = λ x, 6 * cos (2 * x - π / 6) :=
by
  -- Proof is omitted
  sorry

end derivative_of_y_l491_491603


namespace collinear_vectors_l491_491853

theorem collinear_vectors (λ : ℝ) : let a := (1, 2) 
                                   let b := (λ, -1) in
                                   a.1 / b.1 = a.2 / b.2 → λ = -1 / 2 :=
by
  intros a b h
  sorry

end collinear_vectors_l491_491853


namespace area_new_rectangle_l491_491782

-- Define the given rectangle's dimensions
def a : ℕ := 3
def b : ℕ := 4

-- Define the diagonal of the given rectangle
def d : ℕ := Nat.sqrt (a^2 + b^2)

-- Define the new rectangle's dimensions
def length_new : ℕ := d + a
def breadth_new : ℕ := d - b

-- The target area of the new rectangle
def area_new : ℕ := length_new * breadth_new

-- Prove that the area of the new rectangle is 8 square units
theorem area_new_rectangle (h : d = 5) : area_new = 8 := by
  -- Indicate that proof steps are not provided
  sorry

end area_new_rectangle_l491_491782


namespace distance_between_intersections_is_pi_l491_491190

theorem distance_between_intersections_is_pi (a : ℝ) : 
  ∃ x1 x2 : ℝ, (tan x1 = a) ∧ (tan x2 = a) ∧ (x2 - x1 = π) :=
sorry

end distance_between_intersections_is_pi_l491_491190


namespace part1_tangent_circles_part2_chords_l491_491016

theorem part1_tangent_circles (t : ℝ) : 
  t = 1 → 
  ∃ (a b : ℝ), 
    (x + 1)^2 + y^2 = 1 ∨ 
    (x + (2/5))^2 + (y - (9/5))^2 = (1 : ℝ) :=
by
  sorry

theorem part2_chords (t : ℝ) : 
  (∀ (k1 k2 : ℝ), 
    k1 + k2 = -3 * t / 4 ∧ 
    k1 * k2 = (t^2 - 1) / 8 ∧ 
    |k1 - k2| = 3 / 4) → 
    t = 1 ∨ t = -1 :=
by
  sorry

end part1_tangent_circles_part2_chords_l491_491016


namespace Charley_total_beads_pulled_l491_491338

-- Definitions and conditions
def initial_white_beads := 105
def initial_black_beads := 210
def initial_blue_beads := 60

def first_round_black_pulled := (2 / 7) * initial_black_beads
def first_round_white_pulled := (3 / 7) * initial_white_beads
def first_round_blue_pulled := (1 / 4) * initial_blue_beads

def first_round_total_pulled := first_round_black_pulled + first_round_white_pulled + first_round_blue_pulled

def remaining_black_beads := initial_black_beads - first_round_black_pulled
def remaining_white_beads := initial_white_beads - first_round_white_pulled
def remaining_blue_beads := initial_blue_beads - first_round_blue_pulled

def added_white_beads := 45
def added_black_beads := 80

def total_black_beads := remaining_black_beads + added_black_beads
def total_white_beads := remaining_white_beads + added_white_beads

def second_round_black_pulled := (3 / 8) * total_black_beads
def second_round_white_pulled := (1 / 3) * added_white_beads

def second_round_total_pulled := second_round_black_pulled + second_round_white_pulled

def total_beads_pulled := first_round_total_pulled + second_round_total_pulled 

-- Theorem statement
theorem Charley_total_beads_pulled : total_beads_pulled = 221 := 
by
  -- we can ignore the proof step and leave it to be filled
  sorry

end Charley_total_beads_pulled_l491_491338


namespace height_of_room_is_twelve_l491_491189

-- Defining the dimensions of the room
def length : ℝ := 25
def width : ℝ := 15

-- Defining the dimensions of the door and windows
def door_area : ℝ := 6 * 3
def window_area : ℝ := 3 * (4 * 3)

-- Total cost of whitewashing
def total_cost : ℝ := 5436

-- Cost per square foot for whitewashing
def cost_per_sqft : ℝ := 6

-- The equation to solve for height
def height_equation (h : ℝ) : Prop :=
  cost_per_sqft * (2 * (length + width) * h - (door_area + window_area)) = total_cost

theorem height_of_room_is_twelve : ∃ h : ℝ, height_equation h ∧ h = 12 := by
  -- Proof would go here
  sorry

end height_of_room_is_twelve_l491_491189


namespace second_divisor_of_152_l491_491632

theorem second_divisor_of_152 : ∀ (x : ℕ), (x - 16 = 136) ∧ (136 % 4 = 0) ∧ (136 % 8 = 0) ∧ (136 % 10 = 0) → 8 :=
by
  intro x
  sorry

end second_divisor_of_152_l491_491632


namespace problem_statement_l491_491194

def f (x : ℝ) : ℝ := cos (2 * x) + sin (x + π / 2)

theorem problem_statement :
  (∀ x : ℝ, f x = f (-x)) ∧ ∃ a b : ℝ, (∀ x : ℝ, f x ≤ a) ∧ (∀ x : ℝ, b ≤ f x) :=
by
  sorry

end problem_statement_l491_491194


namespace area_of_rhombus_is_375_l491_491655

-- define the given diagonals
def diagonal1 := 25
def diagonal2 := 30

-- define the formula for the area of a rhombus
def area_of_rhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

-- state the theorem
theorem area_of_rhombus_is_375 : area_of_rhombus diagonal1 diagonal2 = 375 := 
by 
  -- The proof is omitted as per the requirement
  sorry

end area_of_rhombus_is_375_l491_491655


namespace image_of_9_l491_491713

-- Condition: f : ℕ → ℝ such that f(x) = log_base 3 of x
def f (x : ℕ) : ℝ := Real.log x / Real.log 3

-- The goal: f(9) = 2
theorem image_of_9 : f 9 = 2 :=
by
  sorry

end image_of_9_l491_491713


namespace find_length_AB_l491_491935

theorem find_length_AB 
(distance_between_parallels : ℚ)
(radius_of_incircle : ℚ)
(is_isosceles : Prop)
(h_parallel : distance_between_parallels = 18 / 25)
(h_radius : radius_of_incircle = 8 / 3)
(h_isosceles : is_isosceles) :
  ∃ AB : ℚ, AB = 20 := 
sorry

end find_length_AB_l491_491935


namespace length_of_QY_is_31_l491_491658

theorem length_of_QY_is_31
  (P Q R : Point)
  (X Y Z : Point)
  (O₄ O₅ O₆ : Point)
  (h_inscribed : P ∈ Segment YZ ∧ Q ∈ Segment XZ ∧ R ∈ Segment XY)
  (h_circle_centers : circumcenter △PYZ = O₄ ∧ circumcenter △QXR = O₅ ∧ circumcenter △RQP = O₆)
  (XY YZ XZ : ℚ)
  (h_sides_length : XY = 29 ∧ YZ = 35 ∧ XZ = 28)
  (h_arcs : length_arc YR = length_arc QZ ∧ length_arc XR = length_arc PY ∧ length_arc XP = length_arc QY)
  (h_QY_form : ∃ p q : ℕ, p.gcd q = 1 ∧ QY = p / q) :
  ∃ (p q : ℕ), p + q = 31 :=
by
  sorry

end length_of_QY_is_31_l491_491658


namespace endpoints_of_unit_vectors_form_unit_circle_l491_491492

noncomputable def unit_circle (O : Point) : set Point :=
  { P : Point | dist O P = 1 }

theorem endpoints_of_unit_vectors_form_unit_circle (O : Point) :
  (∀ v : Vector, norm v = 1 → start_of_vector v = O) →
  endpoints_of_vectors = unit_circle O := 
sorry

end endpoints_of_unit_vectors_form_unit_circle_l491_491492


namespace parabola_properties_l491_491633

def parabola_vertex_origin : Prop :=
  ∃ p > 0, ∀ x y : ℝ, y^2 = 2 * p * x ∧ y^2 = (4 / 11) * x

def intersects_line_at_A_and_B (p : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, (A.1 + A.2 - 1 = 0) ∧ (B.1 + B.2 - 1 = 0) ∧
    |A.1 - B.1| * sqrt (1 + (4 / 11)) = (8 * sqrt 6) / 11

def no_equilateral_triangle (A B : ℝ × ℝ) : Prop :=
  ¬ ∃ C : ℝ × ℝ, C.2 = 0 ∧ 
    sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = sqrt 3 * (|A.1 - B.1| * sqrt (1 + (4 / 11))) / 2

theorem parabola_properties :
  parabola_vertex_origin ∧
  (∀ p > 0, intersects_line_at_A_and_B p) ∧
  (∀ A B, no_equilateral_triangle A B) :=
by
  sorry

end parabola_properties_l491_491633


namespace sum_y_seq_l491_491892

noncomputable def y_seq (n : ℕ) : ℕ → ℕ
| 0       := 1
| 1       := n + 1
| (k + 2) := ((n + 2) * y_seq n (k + 1) - (n + 1 - k) * y_seq n k) / (k + 2)

theorem sum_y_seq (n : ℕ) (hn : 0 < n) :
  (∑ k in Finset.range (n + 2), y_seq n k) = 2 ^ (2 * n + 1) - Nat.choose (2 * n + 2) (n + 2) :=
sorry

end sum_y_seq_l491_491892


namespace parametric_equation_perpendicular_points_l491_491921

noncomputable def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 = 9 / (cos θ^2 + 9 * sin θ^2)

theorem parametric_equation :
  ∀ x y : ℝ, (∃ ρ θ : ℝ, polar_equation ρ θ ∧ x = ρ * cos θ ∧ y = ρ * sin θ) ↔ (x^2 / 9 + y^2 = 1) :=
by
  sorry

theorem perpendicular_points (ρ1 θ1 ρ2 θ2 : ℝ) (h1 : polar_equation ρ1 θ1) (h2 : polar_equation ρ2 θ2)
  (h_perp : θ2 = θ1 + π/2 ∨ θ2 = θ1 - π/2) :
  (1 / ρ1^2 + 1 / ρ2^2) = 10 / 9 :=
by
  sorry

end parametric_equation_perpendicular_points_l491_491921


namespace f_diff_eq_l491_491842

def f (n : ℕ) : ℚ := 1 / 4 * (n * (n + 1) * (n + 3))

theorem f_diff_eq (r : ℕ) : 
  f (r + 1) - f r = 1 / 4 * (3 * r^2 + 11 * r + 8) :=
by {
  sorry
}

end f_diff_eq_l491_491842


namespace least_possible_n_l491_491767

theorem least_possible_n :
  ∃ n m : ℕ, 0 < n ∧ 0 < m ∧ (711 / 1000 : ℝ) ≤ m / n ∧ m / n < 712 / 1000 ∧
  ∀ n' m' : ℕ, 0 < n' ∧ 0 < m' ∧ (711 / 1000 : ℝ) ≤ m' / n' ∧ m' / n' < 712 / 1000 → n' ≥ n :=
begin
  use [45, 32],
  split,
  { norm_num },
  split,
  { norm_num },
  split,
  { norm_num,
    linarith },
  split,
  { norm_num,
    linarith },
  intro n' m',
  intros hn' hm' hl hu,
  have hfrac : 711 / 1000 < 32 / 45 := by norm_num,
  have hfrac' : m' / n' < 712 / 1000 := hu,
  linarith,
  sorry
end

end least_possible_n_l491_491767


namespace determine_sum_l491_491377

noncomputable def S (n : ℕ) : Set ℕ := {i | (i > 0) ∧ (i ≤ n)}

noncomputable def S_k (n k : ℕ) (hk : k ∈ S n) : Set (Set ℕ) := 
  {X | X ⊆ S n ∧ k ∉ X ∧ X ≠ ∅}

noncomputable def S_k_star (n k : ℕ) (hk : k ∈ S n) : ℝ :=
  ∑ X in S_k n k hk, (∏ i in X, (i : ℝ))⁻¹

theorem determine_sum (n k : ℕ) (hk : k ∈ S n) : 
  S_k_star n k hk = (n * k - 1) / (k + 1) := 
sorry

end determine_sum_l491_491377


namespace no_arith_geom_prog_in_cyclic_or_circumscribed_quad_l491_491923

theorem no_arith_geom_prog_in_cyclic_or_circumscribed_quad (a b c d : ℝ) :
  (is_cyclic_quad A B C D ∨ is_circumscribed_quad A B C D) →
  ¬((∃ r, a = b + r ∧ b = c + r ∧ c = d + r) ∨ (∃ q, a = b * q ∧ b = c * q ∧ c = d * q)) ∨ (a = b ∧ b = c ∧ c = d) :=
by
  sorry

end no_arith_geom_prog_in_cyclic_or_circumscribed_quad_l491_491923


namespace find_point_C_l491_491284

noncomputable def point_on_z_axis (z : ℝ) : ℝ × ℝ × ℝ := (0, 0, z)
def point_A : ℝ × ℝ × ℝ := (1, 0, 2)
def point_B : ℝ × ℝ × ℝ := (1, 1, 1)

theorem find_point_C :
  ∃ C : ℝ × ℝ × ℝ, (C = point_on_z_axis 1) ∧ (dist C point_A = dist C point_B) :=
by
  sorry

end find_point_C_l491_491284


namespace probability_factor_of_30_correct_l491_491251

noncomputable def probability_factor_of_30 : ℚ :=
  let n := 30
  let divisors_of_30 : Finset ℕ := {d ∈ (Finset.range (n + 1)) | n % d = 0}
  (divisors_of_30.card : ℚ) / n

theorem probability_factor_of_30_correct :
  probability_factor_of_30 = 4 / 15 :=
by
  sorry

end probability_factor_of_30_correct_l491_491251


namespace like_terms_sum_l491_491796

theorem like_terms_sum (m n : ℕ) (h1 : 2 * m = 4) (h2 : 3 - n = 1) : m + n = 4 :=
by
  sorry

end like_terms_sum_l491_491796


namespace proof_problem_l491_491121

open Set

variable {α : Type*} [LinearOrderedField α]

def U : Set α := univ
def M : Set α := {x | x < (1 : α)}
def N : Set α := {x | (-1 : α) < x ∧ x < (2 : α)}

theorem proof_problem : {x | (2 : α) ≤ x} = compl (M ∪ N) := 
by
  intro x
  simp
  sorry

end proof_problem_l491_491121


namespace compute_fraction_sum_l491_491535

-- Define the equation whose roots are a, b, c
def cubic_eq (x : ℝ) : Prop := x^3 - 6*x^2 + 11*x = 12

-- State the main theorem
theorem compute_fraction_sum 
  (a b c : ℝ) 
  (ha : cubic_eq a) 
  (hb : cubic_eq b) 
  (hc : cubic_eq c) :
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) → 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) → 
  ∃ (r : ℝ), r = -23/12 ∧ (ab/c + bc/a + ca/b) = r := 
  sorry

end compute_fraction_sum_l491_491535


namespace symmetry_xoy_l491_491094

-- Define the symmetry in the given problem
def symmetric_point (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, -p.3)

-- Given point
def point : ℝ × ℝ × ℝ := (1, 2, 3)

-- Statement of the proof problem
theorem symmetry_xoy : symmetric_point point = (1, 2, -3) :=
by
  sorry

end symmetry_xoy_l491_491094


namespace find_largest_p_with_positive_integer_roots_l491_491403

theorem find_largest_p_with_positive_integer_roots :
  ∃ p : ℝ, (∀ k : ℝ, (polynomial.eval k (5 * polynomial.X ^ 3 - 5 * (p + 1) * polynomial.X ^ 2 + (71 * p - 1) * polynomial.X + 1) = 66 * p) → (∃ r1 r2 r3 : ℝ, r1 > 0 ∧ r2 > 0 ∧ r3 > 0 ∧ 
(p = Real.max (r1 + r2 + r3))) ∧ p = 76 ) :=
sorry

end find_largest_p_with_positive_integer_roots_l491_491403


namespace probability_born_in_2008_l491_491143

/-- Define the relevant dates and days calculation -/
def num_days_between : ℕ :=
  29 + 31 + 30 + 31 + 31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

def num_days_2008 : ℕ :=
  31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

theorem probability_born_in_2008 :
  (num_days_2008 : ℚ) / num_days_between = 244 / 365 :=
by
  -- The proof is omitted for brevity
  sorry

end probability_born_in_2008_l491_491143


namespace dihedral_angle_regular_triangular_pyramid_l491_491614

theorem dihedral_angle_regular_triangular_pyramid :
  ∀ (h := 1) (l := real.sqrt 5),
    let D := point.mk 0 0 (h : ℝ),
    let A : point := point.mk 0 0 0,
    let B : point := point.mk (real.sqrt 3) 1 0,
    let C : point := point.mk (- (real.sqrt 3)) 1 0,
    let M : point := centroid_point [A, B, C],
    let AK := midpoint_point A B,
    let K := midpoint_point B C in
  dihedral_angle_base A B C D = ((45.0 / 2 * 3.141592653589793 / 180) : ℝ) :=
sorry

end dihedral_angle_regular_triangular_pyramid_l491_491614


namespace prob_at_least_two_correct_l491_491875

-- Probability of guessing a question correctly
def prob_correct := 1 / 6

-- Probability of guessing a question incorrectly
def prob_incorrect := 5 / 6

-- Binomial probability mass function for k successes out of n trials
def binom_pmf (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * (p ^ k) * ((1 - p) ^ (n - k))

-- Calculate probability P(X = 0)
def prob_X0 := binom_pmf 6 0 prob_correct

-- Calculate probability P(X = 1)
def prob_X1 := binom_pmf 6 1 prob_correct

-- Theorem for the desired probability
theorem prob_at_least_two_correct : 
  1 - (prob_X0 + prob_X1) = 34369 / 58420 := by
  sorry

end prob_at_least_two_correct_l491_491875


namespace total_number_of_flowers_is_correct_l491_491218

-- Define the conditions
def number_of_pots : ℕ := 544
def flowers_per_pot : ℕ := 32
def total_flowers : ℕ := number_of_pots * flowers_per_pot

-- State the theorem to be proved
theorem total_number_of_flowers_is_correct :
  total_flowers = 17408 :=
by
  sorry

end total_number_of_flowers_is_correct_l491_491218


namespace sum_non_solution_values_l491_491116

theorem sum_non_solution_values (A B C : ℝ) (h : ∀ x : ℝ, (x+B) * (A*x+36) / ((x+C) * (x+9)) = 4) :
  ∃ M : ℝ, M = - (B + 9) := 
sorry

end sum_non_solution_values_l491_491116


namespace penny_frogs_count_l491_491310

theorem penny_frogs_count :
  let tree_frogs := 55
  let poison_frogs := 10
  let wood_frogs := 13
  tree_frogs + poison_frogs + wood_frogs = 78 :=
by
  let tree_frogs := 55
  let poison_frogs := 10
  let wood_frogs := 13
  show tree_frogs + poison_frogs + wood_frogs = 78
  sorry

end penny_frogs_count_l491_491310


namespace gcd_problem_l491_491706

def a : ℕ := 101^5 + 1
def b : ℕ := 101^5 + 101^3 + 1

theorem gcd_problem : Nat.gcd a b = 1 := by
  sorry

end gcd_problem_l491_491706


namespace range_of_a_for_semi_fixed_points_l491_491119

def semi_fixed_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop := f x₀ = -x₀

theorem range_of_a_for_semi_fixed_points {a : ℝ} :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧
    semi_fixed_point (λ x, a * x^3 - 3 * x^2 - x + 1) x₁ ∧
    semi_fixed_point (λ x, a * x^3 - 3 * x^2 - x + 1) x₂ ∧
    semi_fixed_point (λ x, a * x^3 - 3 * x^2 - x + 1) x₃) ↔
  (a > -2 ∧ a < 0 ∨ a > 0 ∧ a < 2) :=
by
  sorry

end range_of_a_for_semi_fixed_points_l491_491119


namespace AR_perpendicular_BC_l491_491900

-- Given conditions:
variables {A B C P Q X Y R : Type}
variables {ω Ω_A : Type}
variables [Circumcircle ω ABC] [Excircle Ω_A BC]

-- Intersection points of circles
variables (h₁ : IntersectionPoints ω Ω_A X Y)
-- Projections of A onto tangent lines
variables (h₂ : ProjectionOf A (TangentLine Ω_A X) P)
variables (h₃ : ProjectionOf A (TangentLine Ω_A Y) Q)
variables (h₄ : TangentIntersection (Circumcircle APX) P (Circumcircle AQY) Q R)

-- Goal to prove
theorem AR_perpendicular_BC : Perpendicular AR BC :=
sorry

end AR_perpendicular_BC_l491_491900


namespace max_value_of_x_plus_2y_on_ellipse_l491_491208

theorem max_value_of_x_plus_2y_on_ellipse :
  (∃ (x y : ℝ), (x^2 / 6 + y^2 / 4 = 1) ∧ (∀ (x y : ℝ), (x^2 / 6 + y^2 / 4 = 1) → (x + 2 * y) ≤ sqrt 22)) :=
sorry

end max_value_of_x_plus_2y_on_ellipse_l491_491208


namespace transformed_function_graph_l491_491943

noncomputable def g (x : ℝ) : ℝ :=
  if h : x ∈ [-4, -1] then 3 * x + 2
  else if h : x ∈ [-1, 3] then -real.sqrt (9 - (x - 1) ^ 2) + 2
  else if h : x ∈ [3, 5] then -3 * (x - 3) + 1
  else 0

theorem transformed_function_graph :
  ∀ x : ℝ, y = -2 * g x - 1 ↔
  (x ∈ [-4, -1] ∧ y = -6 * x - 5) ∨
  (x ∈ [-1, 3] ∧ y = 2 * real.sqrt(9 - (x - 1) ^ 2) - 5) ∨
  (x ∈ [3, 5] ∧ y = 6 * x - 21) :=
by sorry

end transformed_function_graph_l491_491943


namespace probability_same_color_is_7_9_l491_491059

def sides_first_die : ℕ := 12
def sides_second_die : ℕ := 15

def red_sides_first_die : ℕ := 3
def blue_sides_first_die : ℕ := 4
def green_sides_first_die : ℕ := 5

def red_sides_second_die : ℕ := 5
def blue_sides_second_die : ℕ := 3
def green_sides_second_die : ℕ := 7

noncomputable def probability_same_color : ℚ :=
(red_sides_first_die.to_rat / sides_first_die) * (red_sides_second_die.to_rat / sides_second_die) +
(blue_sides_first_die.to_rat / sides_first_die) * (blue_sides_second_die.to_rat / sides_second_die) +
(green_sides_first_die.to_rat / sides_first_die) * (green_sides_second_die.to_rat / sides_second_die)

theorem probability_same_color_is_7_9 : probability_same_color = 7 / 9 :=
by
  sorry

end probability_same_color_is_7_9_l491_491059


namespace doris_weeks_to_meet_expenses_l491_491724

def doris_weekly_hours : Nat := 5 * 3 + 5 -- 5 weekdays (3 hours each) + 5 hours on Saturday
def doris_hourly_rate : Nat := 20 -- Doris earns $20 per hour
def doris_weekly_earnings : Nat := doris_weekly_hours * doris_hourly_rate -- The total earnings per week
def doris_monthly_expenses : Nat := 1200 -- Doris's monthly expense

theorem doris_weeks_to_meet_expenses : ∃ w : Nat, doris_weekly_earnings * w ≥ doris_monthly_expenses ∧ w = 3 :=
by
  sorry

end doris_weeks_to_meet_expenses_l491_491724


namespace neil_fraction_l491_491881

-- Given conditions
variable (total_marbles : ℕ := 400)
variable (marbles_per_pack : ℕ := 10)
variable (given_fraction_manny : ℚ := 1/4)
variable (packs_kept_leo : ℕ := 25)

-- Calculated values
def total_packs := total_marbles / marbles_per_pack
def packs_given_away := total_packs - packs_kept_leo
def packs_given_manny := (given_fraction_manny * packs_given_away).toNat  -- considering whole packs
def packs_given_neil := packs_given_away - packs_given_manny

-- Theorem to prove
theorem neil_fraction (h1 : total_packs = 40) (h2 : packs_given_away = 15) 
                       (h3 : packs_given_manny = 3) (h4 : packs_given_neil = 12) :
  (packs_given_neil : ℚ) / packs_given_away = 4 / 5 :=
by
  sorry

end neil_fraction_l491_491881


namespace compare_rat_neg_l491_491348

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l491_491348


namespace find_a_value_l491_491474

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end find_a_value_l491_491474


namespace unique_multiple_l491_491635

def is_composed_by_digits (n : ℕ) (digits : List ℕ) : Prop :=
  let ds := List.ofFn (Nat.digits 10 n)
  List.length ds = List.length digits ∧ List.perm ds digits

theorem unique_multiple (n : ℕ) : 
  ∃ m : ℕ, m ≠ n ∧ m ∈ {x | is_composed_by_digits x [2, 4, 5, 7]} ∧ n / m = 3 := sorry

end unique_multiple_l491_491635


namespace difference_in_perimeter_is_50_cm_l491_491056

-- Define the lengths of the four ribbons
def ribbon_lengths (x : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (x, x + 25, x + 50, x + 75)

-- Define the perimeter of the first shape
def perimeter_first_shape (x : ℕ) : ℕ :=
  2 * x + 230

-- Define the perimeter of the second shape
def perimeter_second_shape (x : ℕ) : ℕ :=
  2 * x + 280

-- Define the main theorem that the difference in perimeter is 50 cm
theorem difference_in_perimeter_is_50_cm (x : ℕ) :
  perimeter_second_shape x - perimeter_first_shape x = 50 := by
  sorry

end difference_in_perimeter_is_50_cm_l491_491056


namespace geometric_sequence_sum_l491_491533

noncomputable def sum_of_first_n_terms (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) (h_a1 : a_1 = 1) (h_a5 : a_5 = 16) :
  sum_of_first_n_terms 1 q 7 = 127 :=
by
  sorry

end geometric_sequence_sum_l491_491533


namespace chameleons_to_blue_l491_491165

-- Define a function that simulates the biting between chameleons and their resulting color changes
def color_transition (color_biter : ℕ) (color_bitten : ℕ) : ℕ :=
  if color_bitten = 1 then color_biter + 1
  else if color_bitten = 2 then color_biter + 2
  else if color_bitten = 3 then color_biter + 3
  else if color_bitten = 4 then color_biter + 4
  else 5  -- Once it reaches color 5 (blue), it remains blue.

-- Define the main theorem statement that given 5 red chameleons, all can be turned to blue.
theorem chameleons_to_blue : ∀ (red_chameleons : ℕ), red_chameleons = 5 → 
  ∃ (sequence_of_bites : ℕ → (ℕ × ℕ)), (∀ (c : ℕ), c < 5 → color_transition c (sequence_of_bites c).fst = 5) :=
by sorry

end chameleons_to_blue_l491_491165


namespace specific_four_digits_property_l491_491755

theorem specific_four_digits_property :
  ∃ a b c d : ℕ, 100 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 999 ∧
  (∀ x y z w: ℕ, {x, y, z, w} = {a, b, c, d} → 
    (x + y + z) % w = 0 ∧ (x + y + w) % z = 0 ∧ (x + z + w) % y = 0 ∧ (y + z + w) % x = 0) :=
  by {
    use 100,
    use 200,
    use 300,
    use 600,
    split, exact dec_trivial,
    split, linarith,
    split, linarith,
    split, linarith,
    split, linarith,
    intros x y z w hxyz,
    rw [set.singleton, hxyz],
    split,
      show (200 + 300 + 600) % 100 = 0,
      exact by norm_num,
    split,
      show (100 + 300 + 600) % 200 = 0,
      exact by norm_num,
    split,
      show (100 + 200 + 600) % 300 = 0,
      exact by norm_num,
    show (100 + 200 + 300) % 600 = 0,
    exact by norm_num,
  sorry

end specific_four_digits_property_l491_491755


namespace log_base_16_of_4_l491_491740

theorem log_base_16_of_4 : 
  (16 = 2^4) →
  (4 = 2^2) →
  (∀ (b a c : ℝ), b > 0 → b ≠ 1 → c > 0 → c ≠ 1 → log b a = log c a / log c b) →
  log 16 4 = 1 / 2 :=
by
  intros h1 h2 h3
  sorry

end log_base_16_of_4_l491_491740


namespace find_natisfies_conditions_l491_491240

theorem find_natisfies_conditions :
  ∃ (n : ℤ), 0 ≤ n ∧ n < 7 ∧ (-2222 ≡ n [ZMOD 7]) :=
begin
  use 3,
  split,
  { exact dec_trivial, }, -- 0 ≤ 3
  split,
  { exact dec_trivial, }, -- 3 < 7
  { norm_num, }, -- -2222 ≡ 3 [ZMOD 7]
end

end find_natisfies_conditions_l491_491240


namespace checkers_cannot_move_to_left_half_l491_491908

def checkerboard (n : Nat) := Fin n × Fin n

/-- Initial configuration of checkers on a 10x10 checkerboard. -/
structure initial_configuration (cb : checkerboard 10) :=
(lower_left : ∀ xy, cb xy → xy ∈ { (x, y) | x < 5 ∧ y < 5})
(upper_right : ∀ xy, cb xy → xy ∈ { (x, y) | x ≥ 5 ∧ y ≥ 5})

/-- Checkers can jump over an adjacent checker to the next free square. -/
inductive move : checkerboard 10 → checkerboard 10 → Prop
| horizontal (x y : Fin 10) : move (x, y) (x+2, y) 
| vertical (x y : Fin 10) : move (x, y) (x, y+2)
| diagonal (x y : Fin 10) : move (x, y) (x+2, y+2)

/-- Check if final configuration has all checkers on the left half of the board. -/
def final_configuration (cb : checkerboard 10) (xy : cb) :=
xy ∈ { (x, y) | x < 5 }

/-- Theorem stating it is impossible to move all checkers to the left half of the board. -/
theorem checkers_cannot_move_to_left_half (cb : checkerboard 10)
  (init : initial_configuration cb) :
  ¬(∃ cb', (∀ xy, move cb xy cb' xy) ∧ (∀ xy, cb' xy → final_configuration cb' xy)) :=
sorry

end checkers_cannot_move_to_left_half_l491_491908


namespace polyline_count_l491_491025

def polyline (P : ℕ → ℤ × ℤ) :=
  ∀ i, ∃ (x y : int), P i = (x, y) ∧ dist (P i) (P (i+1)) = 1

theorem polyline_count (n : ℕ) (P : ℕ → ℤ × ℤ) (hP0 : P 0 = (0,0)) (hPn : ∃ x, P n = (x, 0)) :
  ∃ F : ℕ → ℕ, F n = nat.choose (2*n) n :=
by 
  sorry

end polyline_count_l491_491025


namespace BA_bisects_CBD_l491_491895

theorem BA_bisects_CBD
  {A B C H D X Y : Type*} [EuclideanGeometry A B C H D X Y]
  (hABC_acute : acute_triangle A B C)
  (circumcircle_ABC : circumcircle A B C)
  (tangent_at_A : tangent_line circumcircle_ABC A)
  (X_projection : projection B tangent_at_A)
  (Y_projection : projection B (line_through A C))
  (H_orthocenter : orthocenter B X_projection Y_projection H)
  (CH_line : line_through C H)
  (D_intersection : intersection CH_line tangent_at_A D) :
  bisects_angle (line_through B A) (angle C B D) :=
sorry

end BA_bisects_CBD_l491_491895


namespace find_parameter_a_l491_491401

noncomputable theory

def has_eight_solutions (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    ((a * |y^3| + a * |x^3| - 8) * (x^6 + y^6 - 3 * a^2) = 0) ∧
    (sqrt (x^6 * y^6) = a)

theorem find_parameter_a :
  {a : ℝ | has_eight_solutions a} = {a | 0 < a ∧ a < 2 / 3} ∪ {2} ∪ {a | 2 * 2^(1/3) < a} :=
sorry

end find_parameter_a_l491_491401


namespace event_probability_l491_491673

noncomputable def probability_event (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 4 then -- x chosen from [0, 4]
    if (-1) ≤ Real.log (x + 1/2) / Real.log (1/2) ∧ Real.log (x + 1/2) / Real.log (1/2) ≤ 1 then
      3/8 -- probability of the event
    else
      0
  else
    0

theorem event_probability : 
  ∫ (x : ℝ) in 0..4, function.Indicator (fun x => (-1) ≤ Real.log (x + 1/2) / Real.log (1/2) ∧ Real.log (x + 1/2) / Real.log (1/2) ≤ 1) (fun _ => 1) x / (4 - 0) = 3 / 8 :=
by 
  sorry

end event_probability_l491_491673


namespace simplify_evaluate_expr_l491_491925

theorem simplify_evaluate_expr (x : ℕ) (h : x = 2023) : (x + 1) ^ 2 - x * (x + 1) = 2024 := 
by 
  sorry

end simplify_evaluate_expr_l491_491925


namespace sum_of_segments_eq_two_l491_491211

theorem sum_of_segments_eq_two
  (ABCD : ℝ)
  (A B C D K M N L : point)
  (s1 s2 s3 s4 s0 : ℝ)
  (side_len_eq_one: ABCD = 1)
  (areas : s0 = s1 + s2 + s3 + s4) :
  AL + BK + CM + DN = 2 :=
begin
  sorry
end

end sum_of_segments_eq_two_l491_491211


namespace coffee_fraction_after_transfers_l491_491320

theorem coffee_fraction_after_transfers (
  -- Initial conditions
  coffee_in_cup1_start: ℚ := 5,
  cream_in_cup2_start: ℚ := 7,
  transfer1: ℚ := 2,
  transfer2: ℚ := 3,
  transfer3: ℚ := 1
) : 
  -- Resulting condition to prove
  let coffee_in_cup2_after_first_transfer := coffee_in_cup1_start - transfer1,
      total_liquid_in_cup2_after_first_transfer := cream_in_cup2_start + transfer1,
      mixture_back_transfer := transfer2,
      coffee_transferred_back := mixture_back_transfer * (transfer1 / total_liquid_in_cup2_after_first_transfer),
      cream_transferred_back := mixture_back_transfer * (cream_in_cup2_start / total_liquid_in_cup2_after_first_transfer),
      coffee_in_cup1_after_second_transfer := coffee_in_cup2_after_first_transfer + coffee_transferred_back,
      cream_in_cup1_after_second_transfer := cream_transferred_back,
      total_liquid_in_cup1_after_second_transfer := coffee_in_cup1_after_second_transfer + cream_in_cup1_after_second_transfer,
      final_transfer_back := transfer3,
      final_coffee_in_cup1 := coffee_in_cup1_after_second_transfer - (final_transfer_back * (coffee_in_cup1_after_second_transfer / total_liquid_in_cup1_after_second_transfer)),
      final_cream_in_cup1 := cream_in_cup1_after_second_transfer - (final_transfer_back * (cream_in_cup1_after_second_transfer / total_liquid_in_cup1_after_second_transfer)),
      final_total_liquid_in_cup1 := final_coffee_in_cup1 + final_cream_in_cup1 in
  (final_coffee_in_cup1 / final_total_liquid_in_cup1) = 37 / 84 :=
sorry

end coffee_fraction_after_transfers_l491_491320


namespace eating_possible_values_l491_491261

def A : Set ℝ := {-1, 1 / 2, 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x ^ 2 = 1 ∧ a ≥ 0}

-- Full eating relation: B ⊆ A
-- Partial eating relation: (B ∩ A).Nonempty ∧ ¬(B ⊆ A ∨ A ⊆ B)

def is_full_eating (a : ℝ) : Prop := B a ⊆ A
def is_partial_eating (a : ℝ) : Prop :=
  (B a ∩ A).Nonempty ∧ ¬(B a ⊆ A ∨ A ⊆ B a)

theorem eating_possible_values :
  {a : ℝ | is_full_eating a ∨ is_partial_eating a} = {0, 1, 4} :=
by
  sorry

end eating_possible_values_l491_491261


namespace at_least_two_positive_l491_491918

theorem at_least_two_positive (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a + b + c > 0) (h5 : a * b + b * c + c * a > 0) :
  (∃ x y : ℝ, (x ≠ y ∧ ((x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c)) ∧ x > 0 ∧ y > 0)) :=
by
  sorry

end at_least_two_positive_l491_491918


namespace translate_point_on_sin_graph_l491_491230

-- Given conditions and the problem statement as per Lean's syntax.
theorem translate_point_on_sin_graph (k : ℤ) (s : ℝ) (t : ℝ) (P : ℝ × ℝ) (P' : ℝ × ℝ) (hP : P = (t, 1)) 
  (hP_on_sin2x : ∃ t, P = (t, 1) ∧ sin(2 * t) = 1) (hP' : P' = (t + s, 1)) 
  (hP'_on_sin2x_pi3 : ∃ t s, P' = (t + s, 1) ∧ sin(2 * (t + s) - π / 3) = 1) :
  (t = k * π + π / 4) ∧ (∃ s_min : ℝ, s_min = π / 6) :=
by
  sorry

end translate_point_on_sin_graph_l491_491230


namespace service_center_location_l491_491941

theorem service_center_location : 
  ∀ (milepost4 milepost9 : ℕ), 
  milepost4 = 30 → milepost9 = 150 → 
  (∃ milepost_service_center : ℕ, milepost_service_center = milepost4 + ((milepost9 - milepost4) / 2)) → 
  milepost_service_center = 90 :=
by
  intros milepost4 milepost9 h4 h9 hsc
  sorry

end service_center_location_l491_491941


namespace triangle_base_length_l491_491513

theorem triangle_base_length (x : ℝ) :
  (∃ s : ℝ, 4 * s = 64 ∧ s * s = 256) ∧ (32 * x / 2 = 256) → x = 16 := by
  sorry

end triangle_base_length_l491_491513


namespace problem_real_numbers_l491_491183

noncomputable theory

open_locale big_operators

theorem problem_real_numbers 
  {n : ℕ} (h_n : 1 < n)
  (a : fin n → ℝ) 
  (h_pos : ∀ i, 0 < a i) 
  (h_sum_eq : ∑ i, a i = ∑ i, (a i)⁻¹) : 
  ∑ i, (n - 1 + a i)⁻¹ ≤ 1 :=
sorry

end problem_real_numbers_l491_491183


namespace probability_even_product_of_two_eight_sided_dice_rolls_l491_491771

theorem probability_even_product_of_two_eight_sided_dice_rolls :
  let outcomes := (1..8).product (1..8)
  let even_outcomes := { (d1, d2) ∈ outcomes | (d1 * d2) % 2 = 0 }
  even_outcomes.card.toFloat / outcomes.card.toFloat = 3 / 4 := by
  sorry

end probability_even_product_of_two_eight_sided_dice_rolls_l491_491771


namespace min_value_of_expression_l491_491804

-- positive real numbers a and b
variables (a b : ℝ)
variables (ha : 0 < a) (hb : 0 < b)
-- given condition: 1/a + 9/b = 6
variable (h : 1 / a + 9 / b = 6)

theorem min_value_of_expression : (a + 1) * (b + 9) ≥ 16 := by
  sorry

end min_value_of_expression_l491_491804


namespace probability_Ephraim_same_heads_Keiko_l491_491878

theorem probability_Ephraim_same_heads_Keiko :
  let outcomes_Keiko := {0, 1}
  let outcomes_Ephraim := {0, 1, 2}
  let favorable_outcomes := (1, 1) :: (0, 0) :: (1, 2) :: []
  let total_outcomes := [(h1, h2) | h1 ∈ outcomes_Keiko, h2 ∈ outcomes_Ephraim ]
  (favorable_outcomes.length / total_outcomes.length) = 3/8 :=
by
  sorry

end probability_Ephraim_same_heads_Keiko_l491_491878


namespace provisions_remaining_days_l491_491298

-- Definitions based on the conditions
def initial_men : ℕ := 1000
def initial_provisions_days : ℕ := 60
def days_elapsed : ℕ := 15
def reinforcement_men : ℕ := 1250

-- Mathematical computation for Lean
def total_provisions : ℕ := initial_men * initial_provisions_days
def provisions_left : ℕ := initial_men * (initial_provisions_days - days_elapsed)
def total_men_after_reinforcement : ℕ := initial_men + reinforcement_men

-- Statement to prove
theorem provisions_remaining_days : provisions_left / total_men_after_reinforcement = 20 :=
by
  -- The proof steps will be filled here, but for now, we use sorry to skip them.
  sorry

end provisions_remaining_days_l491_491298


namespace election_results_l491_491858

-- Define the votes received by each candidate as a list
def votes : List ℕ := [3562, 4841, 7353, 8209, 2769, 6038, 6568, 9315, 3027, 7946]

-- Define constants for the percentages required to win in each round
def first_round_percentage := 32.5 / 100
def second_round_percentage := 40 / 100

-- Define a function to calculate the total number of votes
def total_votes : ℕ := votes.sum

-- Define a function to calculate the minimum votes needed to win in the first round
def minimum_first_round_votes : Float := first_round_percentage * total_votes

-- Define a function that checks if any candidate won in the first round (i.e. received more than the minimum votes)
def first_round_winner : Bool := List.any votes (λ v => v > minimum_first_round_votes)

-- Define the top three candidates based on their votes
def top_three_candidates : List ℕ := List.sort (· > ·) votes |>.take 3

-- Define a function to calculate the total votes received by the top three candidates
def total_votes_top_three : ℕ := top_three_candidates.sum

-- Define a function to calculate the minimum votes needed to win in the second round
def minimum_second_round_votes : Float := second_round_percentage * total_votes_top_three

-- The proof problem in Lean 4
theorem election_results :
  ¬first_round_winner ∧
  top_three_candidates = [9315, 8209, 7946] ∧
  minimum_second_round_votes = 10188 :=
by
  sorry

end election_results_l491_491858


namespace sixteen_occupied_board_l491_491683

def spy_vision (n m : ℕ) (board : square := sorry) : Prop :=
  ∀ (i j : ℕ), spy_occupied board (i, j) → 
  ¬ (spy_occupied board (i + 1, j) ∨ spy_occupied board (i + 2, j) ∨
     spy_occupied board (i, j + 1) ∨ spy_occupied board (i, j - 1))


theorem sixteen_occupied_board : 
  ∃ (board : fin 6 × fin 6 → bool), -- a 6x6 board representation
    (∀ (i j : fin 6), board (i, j) = tt → 
      ¬ (board (i + 1, j) = tt ∨ board (i + 2, j) = tt ∨
         board (i, j + 1) = tt ∨ board (i, j - 1) = tt)) ∧
    (∑ (i : fin 6) (j : fin 6), board (i, j) = 18) :=
sorry

end sixteen_occupied_board_l491_491683


namespace log_base_equality_l491_491746

theorem log_base_equality : log 4 / log 16 = 1 / 2 := 
by sorry

end log_base_equality_l491_491746


namespace chromium_percentage_new_alloy_l491_491496

theorem chromium_percentage_new_alloy :
  let wA := 15
  let pA := 0.12
  let wB := 30
  let pB := 0.08
  let wC := 20
  let pC := 0.20
  let wD := 35
  let pD := 0.05
  let total_weight := wA + wB + wC + wD
  let total_chromium := (wA * pA) + (wB * pB) + (wC * pC) + (wD * pD)
  total_weight = 100 ∧ total_chromium = 9.95 → total_chromium / total_weight * 100 = 9.95 :=
by
  sorry

end chromium_percentage_new_alloy_l491_491496


namespace solve_for_x_l491_491214

theorem solve_for_x (x : ℝ) : 3 * x^2 + 15 = 3 * (2 * x + 20) → (x = 5 ∨ x = -3) :=
sorry

end solve_for_x_l491_491214


namespace range_of_independent_variable_of_sqrt_l491_491629

theorem range_of_independent_variable_of_sqrt (x : ℝ) : (2 * x - 3 ≥ 0) ↔ (x ≥ 3 / 2) := sorry

end range_of_independent_variable_of_sqrt_l491_491629


namespace probability_maxim_born_in_2008_l491_491149

def days_in_month : ℕ → ℕ → ℕ 
| 2007 9 := 29
| 2007 10 := 31
| 2007 11 := 30
| 2007 12 := 31
| 2008 1 := 31
| 2008 2 := 29   -- leap year
| 2008 3 := 31
| 2008 4 := 30
| 2008 5 := 31
| 2008 6 := 30
| 2008 7 := 31
| 2008 8 := 31
| _ _ := 0

def total_days : ℕ :=
list.sum [days_in_month 2007 9, days_in_month 2007 10, days_in_month 2007 11, days_in_month 2007 12,
          days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

def days_in_2008 : ℕ :=
list.sum [days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

theorem probability_maxim_born_in_2008 : 
  (days_in_2008 : ℚ) / (total_days : ℚ) = 244 / 365 :=
by
  -- Placeholder for actual proof. This theorem is set properly and ready for proof development.
  sorry

end probability_maxim_born_in_2008_l491_491149


namespace curve_C1_eq_curve_C2_eq_product_of_distances_l491_491446

theorem curve_C1_eq : ∀ θ : ℝ, (x = √2 * Real.cos θ) ∧ (y = Real.sin θ) → (x^2 / 2 + y^2 = 1) := sorry

theorem curve_C2_eq : (∀ (θ : ℝ) (ρ : ℝ), ρ * Real.cos θ + ρ * Real.sin θ = 1 → (x + y = 1)) := sorry

theorem |AB|_val : |AB| = (4 * Real.sqrt 2) / 3 := sorry

theorem product_of_distances (M : ℝ × ℝ) (A B : ℝ × ℝ) :
  M = (-1, 2) ∧ (x = √2 * Real.cos θ) ∧ (y = Real.sin θ) ∧ ((x + y = 1) ∧ (dist M A) * (dist M B) = 14 / 3) := sorry

end curve_C1_eq_curve_C2_eq_product_of_distances_l491_491446


namespace repeating_decimal_to_fraction_l491_491752

theorem repeating_decimal_to_fraction :
  ∀ x : ℝ, x = 7 + 326 / 999 + 326 / 999^2 + 326 / 999^3 + ... → x = 22 / 3 :=
by
  intro x hx
  sorry

end repeating_decimal_to_fraction_l491_491752


namespace triangle_find_C_angle_triangle_find_perimeter_l491_491425

variable (A B C a b c : ℝ)

theorem triangle_find_C_angle
  (h1 : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c) :
  C = π / 3 :=
sorry

theorem triangle_find_perimeter
  (h1 : 2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c)
  (h2 : c = Real.sqrt 7)
  (h3 : a * b = 6) :
  a + b + c = 5 + Real.sqrt 7 :=
sorry

end triangle_find_C_angle_triangle_find_perimeter_l491_491425


namespace max_OP_OQ_product_l491_491089

noncomputable def C1_param_eq : ℝ → ℝ × ℝ :=
λ t, (4 * t^2, 4 * t)

noncomputable def C2_param_eq : ℝ → ℝ × ℝ :=
λ φ, (Real.cos φ, 1 + Real.sin φ)

noncomputable def C1_cart_eq (x y : ℝ) : Prop :=
y = 2 * Real.sqrt x

noncomputable def C2_polar_eq (ρ θ : ℝ) : Prop :=
ρ = 2 * Real.sin θ

noncomputable def OP_length (α : ℝ) : ℝ :=
4 * Real.cos α / Real.sin α^2

noncomputable def OQ_length (α : ℝ) : ℝ :=
2 * Real.sin α

theorem max_OP_OQ_product :
  ∃ α : ℝ, α ∈ set.Icc (Real.pi / 6) (Real.pi / 4) ∧ (OP_length α * OQ_length α) = 8 * Real.sqrt 3 := sorry

end max_OP_OQ_product_l491_491089


namespace average_marks_physics_chemistry_l491_491304

theorem average_marks_physics_chemistry
  (P C M : ℕ)
  (h1 : (P + C + M) / 3 = 60)
  (h2 : (P + M) / 2 = 90)
  (h3 : P = 140) :
  (P + C) / 2 = 70 :=
by
  sorry

end average_marks_physics_chemistry_l491_491304


namespace mb_less_than_neg_one_point_five_l491_491612

theorem mb_less_than_neg_one_point_five (m b : ℚ) (h1 : m = 3/4) (h2 : b = -2) : m * b < -1.5 :=
by {
  -- sorry skips the proof
  sorry
}

end mb_less_than_neg_one_point_five_l491_491612


namespace points_on_circle_l491_491793

theorem points_on_circle
  (a : ℝ)
  (P : ℝ × ℝ)
  : 
  let A := (0, a * (Real.sqrt 3) / 2)
  let B := (a / 2, 0)
  let C := (-a / 2, 0)
  PA^2 = PB^2 + PC^2 →
  (P.1)^2 + ((P.2) + (a * (Real.sqrt 3) / 2))^2 = a^2 := 
sorry

end points_on_circle_l491_491793


namespace find_k_l491_491051

variable {x y k : ℝ}

theorem find_k (h1 : 3 * x + 4 * y = k + 2) 
             (h2 : 2 * x + y = 4) 
             (h3 : x + y = 2) :
  k = 4 := 
by
  sorry

end find_k_l491_491051


namespace complex_mag_calc_l491_491731

noncomputable def complex_mag : ℂ := -3 - (5 / 4) * complex.I

theorem complex_mag_calc : complex.abs complex_mag = 13 / 4 := 
by sorry

end complex_mag_calc_l491_491731


namespace compare_fractions_l491_491343

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l491_491343


namespace vertex_distance_l491_491129

theorem vertex_distance : ∀ (C D : ℝ × ℝ), 
  (C = (-3, -4)) ∧ (D = (2, 4)) → 
  (dist C D = Real.sqrt 89) :=
by
  intros C D h
  cases h with hC hD
  rw [hC, hD]
  -- Distance formula
  have dist_sq : ((-3 - 2)^2 + (-4 - 4)^2) = 89 := sorry
  rw [dist, Real.sqrt] at dist_sq
  exact dist_sq

end vertex_distance_l491_491129


namespace sam_long_sleeve_shirts_l491_491178

theorem sam_long_sleeve_shirts (short_sleeve_washed : ℕ) (total_washed : ℕ) (not_washed : ℕ) (short_sleeve : ℕ) : 
  short_sleeve = 40 → total_washed = 29 → not_washed = 34 → short_sleeve_washed = 40 → 
  (total_washed + not_washed) - short_sleeve_washed = 23 :=
by
  intros h₁ h₂ h₃ h₄
  have h₅ : total_washed + not_washed = 63 := by
    rw [h₂, h₃]
  rw [h₅, h₄]
  exact rfl

end sam_long_sleeve_shirts_l491_491178


namespace part_one_part_two_l491_491461

section
variable (a : ℝ)

def A := {x : ℝ | 0 < x ∧ x ≤ 4 }
def B := {y : ℝ | y < a}

-- Question 1: When a = 2, we prove A ∩ B = {z : ℝ | 0 < z ∧ z < 2}
theorem part_one (h : a = 2) : (A ∩ B) = {z : ℝ | 0 < z ∧ z < 2} :=
by
  sorry

-- Question 2: When A ∪ B = B, we prove the range of a is (4, ∞)
theorem part_two (h : A ∪ B = B) : 4 < a :=
by
  sorry

end

end part_one_part_two_l491_491461


namespace vasya_wins_with_optimal_play_l491_491912

-- We are given a game on a 100 x 100 grid
def grid_size : ℕ := 100

-- Petya starts first and colors one cell black
def petya_first_move : ℕ := 1

-- On each move, a player can color a vertical or horizontal white rectangular area 1 x n
-- where n matches or exceeds by one the number of cells colored by the previous player.
def valid_move(n m : ℕ) : Prop := m = n ∨ m = n + 1

-- The player who cannot make a move loses
def player_cannot_move_loses : Prop := sorry

-- Strategy: Vasya maintains a symmetric move relative to the center of the grid.
def symmetric_strategy : Prop := sorry

-- Proving that with optimal play from both players, Vasya wins.
theorem vasya_wins_with_optimal_play 
  (h_grid : (n : ℕ) → n ≤ grid_size^2)
  (h_first_move : petya_first_move = 1)
  (h_valid_move : ∀ (n m : ℕ), valid_move(n, m)) :
  Vasya_wins :=
sorry

end vasya_wins_with_optimal_play_l491_491912


namespace perpendicular_vectors_have_value_m_l491_491831

theorem perpendicular_vectors_have_value_m 
  (m : ℝ) 
  (a : ℝ × ℝ × ℝ := (1, 2, -1))
  (b : ℝ × ℝ × ℝ := (m, m + 2, 1))
  (h : (a.1 * b.1 + a.2 * b.2 + a.3 * b.3) = 0) : m = -1 := 
by {
  sorry
}

end perpendicular_vectors_have_value_m_l491_491831


namespace sum_of_squares_of_roots_eq_zero_l491_491708

theorem sum_of_squares_of_roots_eq_zero :
  let f : Polynomial ℝ := Polynomial.C 50 + Polynomial.monomial 3 (-2) + Polynomial.monomial 7 5 + Polynomial.monomial 10 1
  ∀ (r : ℝ), r ∈ Multiset.toFinset f.roots → r ^ 2 = 0 :=
by
  sorry

end sum_of_squares_of_roots_eq_zero_l491_491708


namespace find_x_eq_3_plus_sqrt7_l491_491531

variable (x y : ℝ)
variable (h1 : x > y)
variable (h2 : x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40)
variable (h3 : x * y + x + y = 8)

theorem find_x_eq_3_plus_sqrt7 (h1 : x > y) (h2 : x^2 * y^2 + x^2 + y^2 + 2 * x * y = 40) (h3 : x * y + x + y = 8) : 
  x = 3 + Real.sqrt 7 :=
sorry

end find_x_eq_3_plus_sqrt7_l491_491531


namespace arithmetic_expression_correct_l491_491973

theorem arithmetic_expression_correct :
  let a := 10
  let b := 10
  let c := 4
  let d := 2
  (d + c / a) * a = 24 :=
by
  let a := 10
  let b := 10
  let c := 4
  let d := 2
  calc
    (d + c / a) * a
        = (2 + 4 / 10) * 10 : by rw [←add_mul, mul_comm 10 (4 / 10), div_mul_cancel'] -- real arithmetic correctness
    ... = 24 : by norm_num [div_eq_mul_inv, ←mul_assoc, mul_inv_cancel_left]

end arithmetic_expression_correct_l491_491973


namespace total_pieces_of_gum_l491_491920

theorem total_pieces_of_gum (packages pieces_per_package : ℕ) 
  (h_packages : packages = 9)
  (h_pieces_per_package : pieces_per_package = 15) : 
  packages * pieces_per_package = 135 := by
  subst h_packages
  subst h_pieces_per_package
  exact Nat.mul_comm 9 15 ▸ rfl

end total_pieces_of_gum_l491_491920


namespace probability_permutations_l491_491120

theorem probability_permutations (T : set (list ℕ)) (hT : ∀ l ∈ T, l.length = 6 ∧ l.head ≠ 1 ∧ l.head ≠ 6 ∧ (∃ m n, m ≠ n ∧ m ≠ l.head ∧ n ≠ l.head ∧ l = [l.head, m, n, (l.head + 4) % 6 + 1, (l.head + 5) % 6 + 1, (l.head + 3) % 6 + 1])) :
  let total_permutations := 6!
    in let valid_permutations := (6! - 2 * 5!)
    in let odd_second_term := 2 * 3 * 4! + 2 * 3 * 4!
    in let probability := odd_second_term / valid_permutations
    in let c := 3
    in let d := 5
    in c + d = 8 :=
by
  sorry

end probability_permutations_l491_491120


namespace eliza_height_l491_491391

theorem eliza_height
  (n : ℕ) (H_total : ℕ) 
  (sib1_height : ℕ) (sib2_height : ℕ) (sib3_height : ℕ)
  (eliza_height : ℕ) (last_sib_height : ℕ) :
  n = 5 →
  H_total = 330 →
  sib1_height = 66 →
  sib2_height = 66 →
  sib3_height = 60 →
  eliza_height = last_sib_height - 2 →
  H_total = sib1_height + sib2_height + sib3_height + eliza_height + last_sib_height →
  eliza_height = 68 :=
by
  intros n_eq H_total_eq sib1_eq sib2_eq sib3_eq eliza_eq H_sum_eq
  sorry

end eliza_height_l491_491391


namespace starfish_arms_l491_491702

variable (x : ℕ)

theorem starfish_arms :
  (7 * x + 14 = 49) → (x = 5) := by
  sorry

end starfish_arms_l491_491702


namespace teacher_arrangement_correct_l491_491639

def number_of_teacher_arrangements (total_teachers : ℕ) (class1_max_teachers : ℕ) (class2_max_teachers : ℕ) : ℕ :=
  if total_teachers = 6 ∧ class1_max_teachers ≤ 4 ∧ class2_max_teachers ≤ 4 then
    (nat.choose total_teachers 4 * 2) + (nat.choose total_teachers 3)
  else
    0

theorem teacher_arrangement_correct :
  number_of_teacher_arrangements 6 4 4 = 50 :=
by 
  simp [number_of_teacher_arrangements, nat.choose]
  sorry

end teacher_arrangement_correct_l491_491639


namespace hyperbola_fixed_point_l491_491044

theorem hyperbola_fixed_point (x y x_0 y_0 : ℝ) (h_hyperbola : x^2 / 4 - y^2 / 3 = 1)
  (h_A1 : x = -2 ∧ y = 0) (h_A2 : x = 2 ∧ y = 0) (h_P : x_0^2 / 4 - y_0^2 / 3 = 1 ∧ y_0 ≠ 0) :
  let M1 := (1, (3 * y_0 / (x_0 + 2))) in
  let M2 := (1, (-y_0 / (x_0 - 2))) in
  let circle_eq := (x - 1)^2 + y^2 - (1 / 2) * ((3 * y_0 / (x_0 + 2)) - (-y_0 / (x_0 - 2)))^2 = 0 in
  let fixed_point := (1, 0) in
  circle_eq fixed_point sorry

end hyperbola_fixed_point_l491_491044


namespace expected_number_of_games_probability_B_wins_the_match_l491_491913

-- Conditions
constant Va : ℝ := 0.3
constant Vd : ℝ := 0.5
constant Vb : ℝ := 1 - Va - Vd

constant V2 : ℝ := Va^2 + Vb^2

constant draw_in_3_games : ℝ := Vd^3
constant win_A_win_B_draw : ℝ := 3 * Va * Vb * Vd
constant V4 : ℝ := draw_in_3_games + win_A_win_B_draw

constant V3 : ℝ := 1 - (V2 + V4)

-- Part (a): Expected number of games
noncomputable def M : ℝ := 2 * V2 + 3 * V3 + 4 * V4

theorem expected_number_of_games : M = 3.175 := by
  sorry

-- Part (b): Probability that B wins the match
constant B_wins_2_games : ℝ := Vb^2
constant B_wins_1_and_2_draws : ℝ := 3 * Vb * Vd^2
constant B_wins_and_A_wins_and_draw : ℝ := 2 * Vb^2 * (1 - Vb)
constant B_wins_after_3_draws : ℝ := V4 * Vb

noncomputable def P_B_wins_the_match : ℝ := 
  B_wins_2_games + B_wins_1_and_2_draws + B_wins_and_A_wins_and_draw + B_wins_after_3_draws

theorem probability_B_wins_the_match : P_B_wins_the_match = 0.315 := by
  sorry

end expected_number_of_games_probability_B_wins_the_match_l491_491913


namespace relationship_between_a_and_b_l491_491003

theorem relationship_between_a_and_b 
  (a b : ℝ) 
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : Real.exp a + 2 * a = Real.exp b + 3 * b) : 
  a > b :=
sorry

end relationship_between_a_and_b_l491_491003


namespace count_uphill_integers_ending_in_6_divisible_by_9_l491_491675

-- Definition of uphill integer
def is_uphill_integer (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ (i : ℕ), i < digits.length - 1 → digits.get i < digits.get (i + 1)

-- Definition of ending in 6
def ends_in_6 (n : ℕ) : Prop :=
  n % 10 = 6

-- Definition of divisible by 9
def divisible_by_9 (n : ℕ) : Prop :=
  n.digits 10.sum % 9 = 0

-- The set of digits considered
def set_digits : List ℕ := [1, 2, 3, 4]

-- Valid uphill integers ending in 6
def valid_uphill_ending_in_6 : List ℕ :=
  let candidates := set_digits.subsets.map (λ s, (s.sum + 6))
  candidates.filter (λ s, s.digits 10.sum % 9 = 0)

-- Counting valid uphill integers ending in 6 that are divisible by 9
theorem count_uphill_integers_ending_in_6_divisible_by_9 : 
  valid_uphill_ending_in_6.length = 2 :=
by
  sorry

end count_uphill_integers_ending_in_6_divisible_by_9_l491_491675


namespace sequence_formula_l491_491014

theorem sequence_formula (S : ℕ+ → ℕ) (a : ℕ+ → ℝ)
  (hS : ∀ n, S n = n - 5 * a n + 23) :
  ∀ n, a n = 3 * (5 / 6)^(n - 1) + 1 :=
by
  -- Sorry to skip the proof.
  sorry

end sequence_formula_l491_491014


namespace representation_sum_of_triplets_l491_491398

theorem representation_sum_of_triplets :
  ∃ (a b c d : ℕ) (T1 T2 T3 T4 : set ℕ),
    T1 = {1, 2, 3} ∧
    T2 = {0, 3, 6} ∧
    T3 = {0, 9, 18} ∧
    T4 = {0, 27, 54} ∧
    (∀ n ∈ finset.range (81 + 1).val, ∃ x y z w : ℕ, x ∈ T1 ∧ y ∈ T2 ∧ z ∈ T3 ∧ w ∈ T4 ∧ n = x + y + z + w) :=
by
  sorry

end representation_sum_of_triplets_l491_491398


namespace inequality_solution_set_range_of_a_l491_491816

def f (x : ℝ) : ℝ := abs (x + 4) - abs (x - 1)

theorem inequality_solution_set : 
  ∀ x : ℝ, (f x > 3 ↔ x > 0) :=
by
  unfold f
  sorry

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, f x + 1 ≤ 4 ^ a - 5 * 2 ^ a) ↔ a ∈ set.Iic 0 ∪ set.Ici 2 :=
by
  unfold f
  sorry

end inequality_solution_set_range_of_a_l491_491816


namespace seed_production_l491_491577

theorem seed_production :
  ∀ (initial_seeds : ℕ) (germination_rate : ℝ) (seed_count_per_plant : ℕ),
    initial_seeds = 1 →
    germination_rate = 0.5 →
    seed_count_per_plant = 50 →
    let new_plants := (initial_seeds * seed_count_per_plant) * germination_rate in
    new_plants * seed_count_per_plant = 1250 :=
by
  intros initial_seeds germination_rate seed_count_per_plant h1 h2 h3
  let new_plants := (initial_seeds * seed_count_per_plant) * germination_rate
  have : new_plants = 25, by {
    rw [h1, h3, h2],
    norm_num,
  }
  rw this
  norm_num

end seed_production_l491_491577


namespace line_and_curve_separate_min_distance_transformed_curve_and_line_l491_491456

open Real

/-- Define the line l -/
def line (t : ℝ) : ℝ × ℝ :=
  (2 + sqrt 2 / 2 * t, sqrt 2 / 2 * t)

/-- Define the original curve C -/
def curve (θ : ℝ) : ℝ × ℝ :=
  (cos θ, sin θ)

/-- Define the transformed curve C2 -/
def transformed_curve (θ : ℝ) : ℝ × ℝ :=
  (1 / 2 * cos θ, sqrt 3 / 2 * sin θ)

/-- Prove positional relationship between l and C: they are separate -/
theorem line_and_curve_separate : ∀ t θ, (line t).fst - (line t).snd - 2 = 0 → (curve θ).fst ^ 2 + (curve θ).snd ^ 2 = 1 → sqrt 2 > 1 := 
by
  sorry

/-- Prove the minimum distance from a point on C2 to line l is sqrt 2 / 2 -/
theorem min_distance_transformed_curve_and_line : ∀ θ, ∃ d, d = abs (sin (π / 6 - θ) - 2) / sqrt 2 ∧ d = sqrt 2 / 2 :=
by
  sorry

end line_and_curve_separate_min_distance_transformed_curve_and_line_l491_491456


namespace intersection_complement_eq_l491_491827

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {x | x^2 - 5 * x + 4 < 0}

theorem intersection_complement_eq :
  A ∩ {x | x ≤ 1 ∨ x ≥ 4} = {0, 1} := by
  sorry

end intersection_complement_eq_l491_491827


namespace find_analytical_expression_l491_491429

open Real

-- Definitions for the function and the conditions
axiom f : ℝ → ℝ
axiom f_add : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_frac : f (2 / 3) = 4

-- The target theorem to be proven
theorem find_analytical_expression : ∀ x, f x = 8 ^ x :=
by
  sorry

end find_analytical_expression_l491_491429


namespace expression_evaluates_to_2023_l491_491257

theorem expression_evaluates_to_2023 : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 :=
by 
  sorry

end expression_evaluates_to_2023_l491_491257


namespace infinite_solutions_factorial_eq_l491_491174

theorem infinite_solutions_factorial_eq (m n k : ℕ) (h1: 1 < m) (h2: 1 < n) (h3: 1 < k) :
  ∃ (n : ℕ), (1 < n) ∧ ((n! - 1)! * n! = n!) :=
by
  sorry

end infinite_solutions_factorial_eq_l491_491174


namespace general_formula_a_seq_l491_491013

noncomputable def a_seq (n : ℕ+) : ℚ :=
  if h : n = 1 
  then 3 / 2 
  else
    let rec a : ℕ+ → ℚ 
        | 1 := 3 / 2
        | n+1 := (1 / 2) * (a n) + 1 
    in a n

theorem general_formula_a_seq (n : ℕ+) : a_seq n = 2 - (1 / 2^n) := by
  sorry

end general_formula_a_seq_l491_491013


namespace triangle_area_sum_l491_491641

variables {A B C : ℝ}
variables (r R : ℝ) (a b c : ℕ)

theorem triangle_area_sum (h₁ : r = 4) (h₂ : R = 13) 
  (h₃ : 4 * real.cos B = real.cos A + real.cos C) :
  (a, b, c : ℕ) (h₄ : triangle_area (4, 13) (4 * cos B = cos A + cos C) = 832 * sqrt 66 / 65 )
  (h₅ : ∃ a b c : ℕ, 
    a * sqrt b / c = 832 * sqrt 66 / 65 ∧ nat.gcd a c = 1 ∧ ¬∃ p : ℕ, p^2 ∣ b ) :
  a + b + c = 963 := 
sorry

end triangle_area_sum_l491_491641


namespace sum_of_squares_not_divisible_by_5_or_13_l491_491133

-- Definition of the set T
def T (n : ℤ) : ℤ :=
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2

-- The theorem to prove
theorem sum_of_squares_not_divisible_by_5_or_13 (n : ℤ) :
  ¬ (T n % 5 = 0) ∧ ¬ (T n % 13 = 0) :=
by
  sorry

end sum_of_squares_not_divisible_by_5_or_13_l491_491133


namespace perp_condition_l491_491283

noncomputable def perp_cond (α : Type) [plane α] (a b l : line α) : Prop :=
  (a ≠ b) ∧ (a ⊆ α) ∧ (b ⊆ α) ∧ (l ∉ α) ∧ (l ⊥ a) ∧ (l ⊥ b) → (∃ (l ⊥ α), (a ∩ b = ∅ → l ⊥ α))

theorem perp_condition (α : Type) [plane α] (a b l : line α) :
  (a ≠ b) → (a ∈ α) → (b ∈ α) → (l ∉ α) → (l ⊥ a) → (l ⊥ b) → (a ∩ b = ∅) → (l ⊥ α) :=
by
  sorry

end perp_condition_l491_491283


namespace find_b_l491_491801

theorem find_b (a b c d : ℝ)
  (h1 : ∀ r : ℂ, polynomial.has_root (polynomial.map (algebra_map ℝ ℂ) (X^4 + C a * X^3 + C b * X^2 + C c * X + C d)) r → ¬ is_real r)
  (h2 : ∃ x1 x2 x3 x4 : ℂ, 
        x1 * x2 = 13 + complex.I ∧ 
        x3 + x4 = 3 + 4 * complex.I ∧ 
        polynomial.has_root (polynomial.map (algebra_map ℝ ℂ) (X^4 + C a * X^3 + C b * X^2 + C c * X + C d)) x1 ∧ 
        polynomial.has_root (polynomial.map (algebra_map ℝ ℂ) (X^4 + C a * X^3 + C b * X^2 + C c * X + C d)) x2 ∧ 
        polynomial.has_root (polynomial.map (algebra_map ℝ ℂ) (X^4 + C a * X^3 + C b * X^2 + C c * X + C d)) x3 ∧ 
        polynomial.has_root (polynomial.map (algebra_map ℝ ℂ) (X^4 + C a * X^3 + C b * X^2 + C c * X + C d)) x4
  ) :
  b = 51 := 
sorry

end find_b_l491_491801


namespace primes_div_diff_l491_491179

theorem primes_div_diff (n : ℕ) : ∃ (p q : ℕ), prime p ∧ prime q ∧ p ≠ q ∧ n ∣ (p - q) :=
by
  sorry

end primes_div_diff_l491_491179


namespace domain_of_f_l491_491245

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : { x : ℝ | f x = (1 / ((x - 3) + (x - 9))) } = {x : ℝ | x ∈ (-∞, 6) ∪ (6, ∞)} :=
by
  sorry

end domain_of_f_l491_491245


namespace semi_circle_radius_calculation_l491_491623

noncomputable def radius_of_semi_circle (P : ℝ) : ℝ :=
  P / (Real.pi + 2)

theorem semi_circle_radius_calculation :
  radius_of_semi_circle 35.11707782401829 ≈ 6.82842712474619 :=
by
  sorry

end semi_circle_radius_calculation_l491_491623


namespace sasha_made_an_error_l491_491974

theorem sasha_made_an_error :
  ∀ (f : ℕ → ℤ), 
  (∀ n, 1 ≤ n → n ≤ 9 → f n = n ∨ f n = -n) →
  (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 = 21) →
  (f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 = 20) →
  false :=
by
  intros f h_cons h_volodya_sum h_sasha_sum
  sorry

end sasha_made_an_error_l491_491974


namespace find_weight_of_second_square_l491_491308

-- Define the initial conditions
def uniform_density_thickness (density : ℝ) (thickness : ℝ) : Prop :=
  ∀ (l₁ l₂ : ℝ), l₁ = l₂ → density = thickness

-- Define the first square properties
def first_square (side_length₁ weight₁ : ℝ) : Prop :=
  side_length₁ = 4 ∧ weight₁ = 16

-- Define the second square properties
def second_square (side_length₂ : ℝ) : Prop :=
  side_length₂ = 6

-- Define the proportional relationship between the area and weight
def proportional_weight (side_length₁ weight₁ side_length₂ weight₂ : ℝ) : Prop :=
  (side_length₁^2 / weight₁) = (side_length₂^2 / weight₂)

-- Lean statement to prove the weight of the second square
theorem find_weight_of_second_square (density thickness side_length₁ weight₁ side_length₂ weight₂ : ℝ)
  (h_density_thickness : uniform_density_thickness density thickness)
  (h_first_square : first_square side_length₁ weight₁)
  (h_second_square : second_square side_length₂)
  (h_proportional_weight : proportional_weight side_length₁ weight₁ side_length₂ weight₂) : 
  weight₂ = 36 :=
by 
  sorry

end find_weight_of_second_square_l491_491308


namespace angle_between_hands_10pm_l491_491756

theorem angle_between_hands_10pm (h_mark_angle : ℝ) (h_mark_angle_eq : h_mark_angle = 30) :
  let angle := 2 * h_mark_angle in angle = 60 :=
by
  rw [h_mark_angle_eq]
  sorry

end angle_between_hands_10pm_l491_491756


namespace find_regionB_area_l491_491710

def regionB_area_proof : Prop :=
  let B := {z : ℂ | 
    0 ≤ (z.re) / 50 ∧ (z.re) / 50 ≤ 1 ∧ 
    0 ≤ (z.im) / 50 ∧ (z.im) / 50 ≤ 1 ∧
    0 ≤ (50 * (z.re) / (z.re ^ 2 + z.im ^ 2)) ∧ (50 * (z.re) / (z.re ^ 2 + z.im ^ 2)) ≤ 1 ∧
    0 ≤ (50 * (z.im) / (z.re ^ 2 + z.im ^ 2)) ∧ (50 * (z.im) / (z.re ^ 2 + z.im ^ 2)) ≤ 1} in
  ∃ (z : set ℂ), z = B ∧ measure_theory.measure_space.volume z = 2500 - 312.5 * Real.pi

theorem find_regionB_area : regionB_area_proof :=
sorry

end find_regionB_area_l491_491710


namespace AD_bisects__l491_491180

-- Definitions from the conditions in the problem
variables (A B C D P Q : Point)
variables (ABC_tri : Triangle A B C)
variables (D_on_BC : OnSegment D B C)
variables (c1 c2 : Circle)
variables (hc1 : c1.pass_through A D ∧ c1.center ∈ Line(A, C))
variables (hc2 : c2.pass_through A D ∧ c2.center ∈ Line(A, B))
variables (P_on_AB : OnCircle P c1 ∧ P ≠ A)
variables (Q_on_AC : OnCircle Q c2 ∧ Q ≠ A)

-- Prove that AD bisects angle PDQ
theorem AD_bisects_∠PDQ :
  AngleBisector (Segment (A, D)) (Angle ∠ P D Q) :=
begin
  sorry
end

end AD_bisects__l491_491180


namespace proof_problem_l491_491517

def circle_center : ℝ × ℝ := (3, 0)

def line_equation (p : ℝ × ℝ) : Prop := p.1 + p.2 = 5

def circle_tangent_to_line (center : ℝ × ℝ) (line_eq : ℝ × ℝ → Prop) : Prop :=
  let d := abs (center.1 + center.2 - 5) / real.sqrt (1^2 + 1^2)
  d = real.sqrt 2

def polar_coordinate_equation (ρ θ : ℝ) : Prop :=
  ρ^2 - 6 * ρ * real.cos θ + 7 = 0

def given_condition (OA OB : ℝ) : Prop :=
  1/OA + 1/OB = 1/7

def Cartesian_equation_of_line (slope x : ℝ) : ℝ := slope * x

theorem proof_problem :
  (circle_tangent_to_line circle_center line_equation) →
  (∀ ρ θ, polar_coordinate_equation ρ θ) →
  (∀ α OA OB, (given_condition OA OB) ∧ (0 < α ∧ α < real.pi / 2 ∧ ρ > 0) →
    Cartesian_equation_of_line (real.sqrt 35) = λ x, real.sqrt 35 * x) :=
by
  intros h1 h2 h3
  sorry

end proof_problem_l491_491517


namespace probability_maxim_born_in_2008_l491_491146

def days_in_month : ℕ → ℕ → ℕ 
| 2007 9 := 29
| 2007 10 := 31
| 2007 11 := 30
| 2007 12 := 31
| 2008 1 := 31
| 2008 2 := 29   -- leap year
| 2008 3 := 31
| 2008 4 := 30
| 2008 5 := 31
| 2008 6 := 30
| 2008 7 := 31
| 2008 8 := 31
| _ _ := 0

def total_days : ℕ :=
list.sum [days_in_month 2007 9, days_in_month 2007 10, days_in_month 2007 11, days_in_month 2007 12,
          days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

def days_in_2008 : ℕ :=
list.sum [days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

theorem probability_maxim_born_in_2008 : 
  (days_in_2008 : ℚ) / (total_days : ℚ) = 244 / 365 :=
by
  -- Placeholder for actual proof. This theorem is set properly and ready for proof development.
  sorry

end probability_maxim_born_in_2008_l491_491146


namespace inequality_holds_l491_491580

noncomputable def region_validity (x y : ℝ) :=
  (x > 0 ∧ y < 0) ∨
  (x > 0 ∧ y > 0 ∧ x < y ∧ xy > 1) ∨
  (x < 0 ∧ y < 0 ∧ x > y ∧ xy < 1)

theorem inequality_holds (x y : ℝ) (h : region_validity x y) :
  3^(1 / (x + 1 / x)) > 3^(1 / (y + 1 / y)) :=
sorry

end inequality_holds_l491_491580


namespace axis_of_symmetry_l491_491489

-- Define the even function property for f(x-2)
def even_function_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x - 2) = f(-x + 2)

-- State the theorem
theorem axis_of_symmetry (f : ℝ → ℝ) (h : even_function_property f) :
  ∃ c : ℝ, ∀ x : ℝ, f(x) = f(2 * c - x) ∧ c = -2 :=
by
  sorry

end axis_of_symmetry_l491_491489


namespace compare_rat_neg_l491_491351

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l491_491351


namespace average_speed_of_trip_l491_491567

-- Define the individual sections
def distance1 := 600
def speed1 := 30

def distance2 := 300
def speed2 := 15

def distance3 := 500
def speed3 := 25

def distance4 := 400
def speed4 := 40

def totalDistance := 1800

-- Define the times taken for each section
def time1 := distance1 / speed1
def time2 := distance2 / speed2
def time3 := distance3 / speed3
def time4 := distance4 / speed4

-- Total time calculation
def totalTime := time1 + time2 + time3 + time4

-- Average speed calculation
def averageSpeed := totalDistance / totalTime

theorem average_speed_of_trip : averageSpeed = 25.71 := 
  by 
    -- Skipping the proof steps
    sorry

end average_speed_of_trip_l491_491567


namespace seed_production_l491_491576

theorem seed_production :
  ∀ (initial_seeds : ℕ) (germination_rate : ℝ) (seed_count_per_plant : ℕ),
    initial_seeds = 1 →
    germination_rate = 0.5 →
    seed_count_per_plant = 50 →
    let new_plants := (initial_seeds * seed_count_per_plant) * germination_rate in
    new_plants * seed_count_per_plant = 1250 :=
by
  intros initial_seeds germination_rate seed_count_per_plant h1 h2 h3
  let new_plants := (initial_seeds * seed_count_per_plant) * germination_rate
  have : new_plants = 25, by {
    rw [h1, h3, h2],
    norm_num,
  }
  rw this
  norm_num

end seed_production_l491_491576


namespace coeff_of_x3_l491_491847

/-- Define the polynomial P(x) as the sum of (1 + x)^k from k=1 to k=2017 -/
noncomputable def P (x: ℚ) : Polynomial ℚ := 
  Polynomial.sum (λ k : ℕ, if 1 ≤ k ∧ k ≤ 2017 then Polynomial.C (1 : ℚ) + Polynomial.X else 0)

/-- Define a_3 as the coefficient of x^3 in P(x) -/
noncomputable def a_3 : ℚ := (P 3).coeff 3

/-- The main statement to be proved: The coefficient a_3 of x^3 in P(x) is equal to binomial of 2018 choose 4 -/
theorem coeff_of_x3 {a_3 : ℚ} : a_3 = (choose 2018 4) := sorry

end coeff_of_x3_l491_491847


namespace choir_final_score_l491_491086

theorem choir_final_score (content_score sing_score spirit_score : ℕ)
  (content_weight sing_weight spirit_weight : ℝ)
  (h_content : content_weight = 0.30) 
  (h_sing : sing_weight = 0.50) 
  (h_spirit : spirit_weight = 0.20) 
  (h_content_score : content_score = 90)
  (h_sing_score : sing_score = 94)
  (h_spirit_score : spirit_score = 95) :
  content_weight * content_score + sing_weight * sing_score + spirit_weight * spirit_score = 93 := by
  sorry

end choir_final_score_l491_491086


namespace range_of_a_l491_491028

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : log a 2 < 1) : (0 < a ∧ a < 1) ∨ (a > 2) :=
sorry

end range_of_a_l491_491028


namespace proper_subset_count_l491_491955

theorem proper_subset_count (S : set ℕ) (h : S = {1, 2, 3}) : 
  (∃ n, n = 7 ∧ true) := 
by sorry

end proper_subset_count_l491_491955


namespace three_integers_sum_divisible_by_three_l491_491433

theorem three_integers_sum_divisible_by_three (a b c d e : ℤ) : 
  ∃ x y z, x ∈ {a, b, c, d, e} ∧ y ∈ {a, b, c, d, e} ∧ z ∈ {a, b, c, d, e} ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x + y + z) % 3 = 0 := 
by 
  sorry

end three_integers_sum_divisible_by_three_l491_491433


namespace domain_of_f_l491_491246

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : { x : ℝ | f x = (1 / ((x - 3) + (x - 9))) } = {x : ℝ | x ∈ (-∞, 6) ∪ (6, ∞)} :=
by
  sorry

end domain_of_f_l491_491246


namespace transport_goods_with_11_trucks_l491_491776

-- Definitions of the conditions translated into Lean

-- Define the total weight of cargo
def total_cargo_weight : ℝ := 13.5

-- Define the max weight of a single box
def max_box_weight : ℝ := 0.35

-- Define the number of trucks
def num_trucks : ℕ := 11

-- Define the load capacity of a single truck
def truck_capacity : ℝ := 1.5

-- The theorem we want to prove
theorem transport_goods_with_11_trucks :
  ∃ (packing : ℕ → ℝ) (total_boxes : ℕ),
    (∀ n, n < num_trucks → packing n ≤ truck_capacity) ∧ -- Each truck's load is within capacity
    (total_boxes * max_box_weight ≥ total_cargo_weight) ∧  -- The total weight is enough to carry all the cargo
    (total_boxes ≤ (num_trucks * (truck_capacity / max_box_weight)).toNat) := -- Number of boxes packing into trucks
begin
  sorry
end

end transport_goods_with_11_trucks_l491_491776


namespace ellipse_find_m_l491_491792

theorem ellipse_find_m (a b m e : ℝ) 
  (h1 : a^2 = 4) 
  (h2 : b^2 = m)
  (h3 : e = 1/2) :
  m = 3 := 
by
  sorry

end ellipse_find_m_l491_491792


namespace pencil_pen_costs_l491_491601

noncomputable def cost_of_items (p q : ℝ) : ℝ := 4 * p + 4 * q

theorem pencil_pen_costs (p q : ℝ) (h1 : 6 * p + 3 * q = 5.40) (h2 : 3 * p + 5 * q = 4.80) : cost_of_items p q = 4.80 :=
by
  sorry

end pencil_pen_costs_l491_491601


namespace log_base_equality_l491_491744

theorem log_base_equality : log 4 / log 16 = 1 / 2 := 
by sorry

end log_base_equality_l491_491744


namespace amount_of_water_in_first_tank_l491_491608

theorem amount_of_water_in_first_tank 
  (C : ℝ)
  (H1 : 0 < C)
  (H2 : 0.45 * C = 450)
  (water_in_first_tank : ℝ)
  (water_in_second_tank : ℝ := 450)
  (additional_water_needed : ℝ := 1250)
  (total_capacity : ℝ := 2 * C)
  (total_water_needed : ℝ := 2000) : 
  water_in_first_tank = 300 :=
by 
  sorry

end amount_of_water_in_first_tank_l491_491608


namespace necessary_condition_not_sufficient_condition_l491_491914

def P (x : ℝ) := x > 0
def Q (x : ℝ) := x > -2

theorem necessary_condition : ∀ x: ℝ, P x → Q x := 
by sorry

theorem not_sufficient_condition : ∃ x: ℝ, Q x ∧ ¬ P x := 
by sorry

end necessary_condition_not_sufficient_condition_l491_491914


namespace six_divisibility_l491_491584

theorem six_divisibility (N : ℕ) (a : ℕ → ℕ) (k : ℕ) (hN : N = ∑ i in finset.range (k + 1), a i * 10^i) :
  (N % 6 = 0) ↔ ((a 0 + 4 * ∑ i in finset.range k, a (i + 1)) % 6 = 0) :=
by sorry

end six_divisibility_l491_491584


namespace grazing_duration_C_l491_491660

theorem grazing_duration_C (
  (nA nB nC nD : ℕ) (mA mB mD : ℕ) (rA RTCPCMMC : ℕ)
  (hA : nA = 24) (hB : nB = 10) (hC : nC = 35) (hD : nD = 21) 
  (mA : mA = 3) (mB : mB = 5) (mD : mD = 3)
  (rA : rA = 1440) (RT : RT = 6500) 
  (CPCM : CPCM = 20)
) : MC = 4 := 
sorry

end grazing_duration_C_l491_491660


namespace no_earnings_left_over_l491_491109

/-!
  Problem:
  Prove that John cannot have any earnings left over given the conditions:
  1. John spent 40 percent of his earnings last month on rent.
  2. He spent 30 percent less than what he spent on rent to purchase a new dishwasher.
  3. He spent 15 percent more on groceries than he spent on rent.
-/

open_locale classical

noncomputable theory

def johns_earnings_last_month : ℝ := 100 -- Assuming John's earnings last month were $100 for easy percent calculations.

def rent_expenses (earnings : ℝ) : ℝ := earnings * 0.4
def dishwasher_expenses (earnings : ℝ) : ℝ := (rent_expenses earnings) * 0.7
def groceries_expenses (earnings : ℝ) : ℝ := (rent_expenses earnings) * 1.15

theorem no_earnings_left_over :
  ∀ (earnings : ℝ),
  let total_expenses := (rent_expenses earnings) + (dishwasher_expenses earnings) + (groceries_expenses earnings) in
  total_expenses ≥ earnings → (earnings - total_expenses) ≤ 0 :=
by {
  intros earnings total_expenses,
  dsimp [johns_earnings_last_month, rent_expenses, dishwasher_expenses, groceries_expenses] at *,
  sorry,
}

end no_earnings_left_over_l491_491109


namespace newPerimeter_l491_491591

/- Define the conditions given in the problem -/
structure TileShape where
  tiles : Nat
  perimeter : Nat

-- Initial shape formed by 10 tiles with given perimeter
def initialShape : TileShape := 
  ⟨10, 18⟩

-- Adding 3 tiles to the shape
def addTiles (shape : TileShape) (additionalTiles : Nat) : TileShape :=
  -- Here we would define the logic of how the tiles are added and how the perimeter is computed
  -- But for the statement we assume it satisfies given conditions and new perimeter can be 17
  sorry

-- Proof problem statement
theorem newPerimeter 
  (initialShape.tiles = 10) 
  (initialShape.perimeter = 18)
  : ∃ newShape : TileShape, newShape.perimeter = 17 :=
sorry

end newPerimeter_l491_491591


namespace mira_jogging_distance_l491_491562

def jogging_speed : ℝ := 5 -- speed in miles per hour
def jogging_hours_per_day : ℝ := 2 -- hours per day
def days_count : ℕ := 5 -- number of days

theorem mira_jogging_distance :
  (jogging_speed * jogging_hours_per_day * days_count : ℝ) = 50 :=
by
  sorry

end mira_jogging_distance_l491_491562


namespace Joe_spent_800_on_hotel_l491_491108

noncomputable def Joe'sExpenses : Prop :=
  let S := 6000 -- Joe's total savings
  let F := 1200 -- Expense on the flight
  let FD := 3000 -- Expense on food
  let R := 1000 -- Remaining amount after all expenses
  let H := S - R - (F + FD) -- Calculating hotel expense
  H = 800 -- We need to prove the hotel expense equals $800

theorem Joe_spent_800_on_hotel : Joe'sExpenses :=
by {
  -- Proof goes here; currently skipped
  sorry
}

end Joe_spent_800_on_hotel_l491_491108


namespace odd_function_value_at_negative_three_l491_491063

variable {f : ℝ → ℝ}

def is_odd_on_interval (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Icc a b, f(-x) = -f(x)

theorem odd_function_value_at_negative_three :
  is_odd_on_interval (-3) 3 f → f 3 = -2 → f (-3) + f 0 = 2 :=
  by
  intros h_odd h_f3
  sorry

end odd_function_value_at_negative_three_l491_491063


namespace problem1_problem2_l491_491813

-- Definition of f(x)
def f (x a : ℝ) : ℝ := Real.exp x - a * x + a / 2

-- First problem: proving a = e given f'(1) = 0
theorem problem1 (a : ℝ) (h1 : 0 < a) (h2 : ∀ x, x ≥ 0 → Real.exp x - a = 0 → x = 1) : a = Real.exp 1 := sorry

-- Second problem: proving a = 2e given f(x) > 0 for x < 1
theorem problem2 (a : ℝ) (h : ∀ x, x < 1 → f x a > 0) : a = 2 * Real.exp 1 := sorry

end problem1_problem2_l491_491813


namespace trigonometric_comparison_l491_491125

theorem trigonometric_comparison :
  let a := Real.sin (33 * Real.pi / 180)
  let b := Real.cos (58 * Real.pi / 180)
  let c := Real.tan (34 * Real.pi / 180)
  c > a ∧ a > b := by
    let a := Real.sin (33 * Real.pi / 180)
    let b := Real.cos (58 * Real.pi / 180)
    let c := Real.tan (34 * Real.pi / 180)
    have hb : b = Real.sin (32 * Real.pi / 180) := by
      rw [Real.cos_eq_sin_pi_div_two_sub]
      norm_num
    
    have hab : a > b := by
      rw [hb]
      exact Real.sin_monotone (lt_add_of_pos_right _ (by norm_num))
    
    have hac : c > a := by
      have hc : c = Real.sin (34 * Real.pi / 180) / Real.cos (34 * Real.pi / 180) := by
        rfl
      have hsc : Real.sin (34 * Real.pi / 180) > Real.sin (33 * Real.pi / 180) := 
        Real.sin_monotone (lt_add_of_pos_right _ (by norm_num))
      have hcc : 0 < Real.cos (34 * Real.pi / 180) := by
        exact Real.cos_pos_of_mem_Ioo (by norm_num; linarith [pi_pos])
      rw [hc]
      exact (div_lt_one hcc).mpr hsc
    
    exact ⟨hac, hab⟩ := sorry

end trigonometric_comparison_l491_491125


namespace dessert_ordering_ways_l491_491905

-- Defining the types and assumptions based on the problem description
def desserts := {pie, cake, rodgrod, creme_brulee}

-- Proving that the number of ways to order the meal
theorem dessert_ordering_ways : 
  ∃ n : ℕ, n = 4 * 3 :=
by {
  use 12,
  exact (nat.succ_mul 4 3).symm,
  sorry
}

end dessert_ordering_ways_l491_491905


namespace diagonal_of_rectangular_prism_l491_491677

noncomputable def diagonal_length (l w h : ℝ) : ℝ :=
  Real.sqrt (l^2 + w^2 + h^2)

theorem diagonal_of_rectangular_prism :
  diagonal_length 15 25 20 = 25 * Real.sqrt 2 :=
by
  sorry

end diagonal_of_rectangular_prism_l491_491677


namespace part1_part2_l491_491041

def f (a : ℝ) (x : ℝ) : ℝ :=
  2 * |x - 2| + a * x

def h (a : ℝ) (x : ℝ) : ℝ :=
  f a (Real.sin x) - 2

theorem part1 (a : ℝ) : (-2 ≤ a ∧ a ≤ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end part1_part2_l491_491041


namespace complex_multiplication_l491_491599

def i : ℂ := complex.I

theorem complex_multiplication : i * (1 - i) = 1 + i := by
  sorry

end complex_multiplication_l491_491599


namespace monotonically_increasing_interval_l491_491405

noncomputable def f (x : ℝ) : ℝ := 4 * x^2 + 1 / x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 1 / 2 → (∀ y : ℝ, y < x → f y < f x) :=
by
  intro x h
  intro y hy
  sorry

end monotonically_increasing_interval_l491_491405


namespace ivan_ball_arrangement_l491_491102

theorem ivan_ball_arrangement (n : ℕ) : 
  let guests := 3 * n
  let hats := {hat | hat = "A" ∨ hat = "B" ∨ hat = "C"}
  let hat_count (h : String) := n
  ⋆ each_guest_gets_hat (h_diff) :
    h_diff ∈ hats → hat_count h_diff = n 
             (∀ guest, ∃ (hat : String), hat ∈ hats ∧ hat_count (hat) = n)
  in
  let arrangement := (3*n)!
  ⋆ each_circle :
    (∀ circle, circle_length circle mod 3 = 0) ∧ 
             (∀ circle, ∀ i, circle[i] = cycle[ABC])
  in
  arrange_in_circles guests = arrangement 

end ivan_ball_arrangement_l491_491102


namespace no_real_solution_implies_a_range_l491_491851

noncomputable def quadratic (a x : ℝ) : ℝ := x^2 - 4 * x + a^2

theorem no_real_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, quadratic a x ≤ 0 → false) ↔ a < -2 ∨ a > 2 := 
sorry

end no_real_solution_implies_a_range_l491_491851


namespace num_integer_solutions_abs_eq_3_l491_491616

theorem num_integer_solutions_abs_eq_3 :
  (∀ (x y : ℤ), (|x| + |y| = 3) → 
  ∃ (s : Finset (ℤ × ℤ)), s.card = 12 ∧ (∀ (a b : ℤ), (a, b) ∈ s ↔ (|a| + |b| = 3))) :=
by
  sorry

end num_integer_solutions_abs_eq_3_l491_491616


namespace g_neg_7_eq_1_l491_491126

def f (x : ℝ) : ℝ := 5 * x - 12
def g (y : ℝ) : ℝ := match (Classical.choose (exists_solution (f:=f) (y))) with
  | some x => 3 * x^2 + 4 * x - 6
  | none => 0

noncomputable def exists_solution (f : ℝ → ℝ) (y : ℝ) : Prop := ∃ x, f(x) = y

theorem g_neg_7_eq_1 : g (-7) = 1 := by
  sorry

end g_neg_7_eq_1_l491_491126


namespace eliza_height_l491_491392

theorem eliza_height
  (n : ℕ) (H_total : ℕ) 
  (sib1_height : ℕ) (sib2_height : ℕ) (sib3_height : ℕ)
  (eliza_height : ℕ) (last_sib_height : ℕ) :
  n = 5 →
  H_total = 330 →
  sib1_height = 66 →
  sib2_height = 66 →
  sib3_height = 60 →
  eliza_height = last_sib_height - 2 →
  H_total = sib1_height + sib2_height + sib3_height + eliza_height + last_sib_height →
  eliza_height = 68 :=
by
  intros n_eq H_total_eq sib1_eq sib2_eq sib3_eq eliza_eq H_sum_eq
  sorry

end eliza_height_l491_491392


namespace problem_arithmetic_l491_491983

theorem problem_arithmetic : (3^1 - 2 + 4^2 + 1) ^ (-2 : ℤ) * 4 = 1 / 81 := by
  sorry

end problem_arithmetic_l491_491983


namespace num_arrangements_five_stars_l491_491221

-- Define the types for blocks
inductive Block
| one_star : Block
| two_star : Block
| three_star : Block

open Block

-- Define the conditions
def blocks_one_star := 5
def blocks_two_star := 2
def blocks_three_star := 1

-- The theorem to state the proof problem
theorem num_arrangements_five_stars : 
  ∃ (n : ℕ), n = 13 ∧ (∃ (l : List Block), l.length = 5 ∧ l.sum (λ b, match b with
    | one_star => 1
    | two_star => 2
    | three_star => 3
  end) = 5 ∧ (∀ b ∈ l, b = one_star → l.count one_star ≤ blocks_one_star)
    ∧ (∀ b ∈ l, b = two_star → l.count two_star ≤ blocks_two_star)
    ∧ (∀ b ∈ l, b = three_star → l.count three_star ≤ blocks_three_star)) :=
sorry

end num_arrangements_five_stars_l491_491221


namespace difference_carnations_daisies_l491_491324

/-
  Ariana bought 120 flowers.
  1/3 of them are roses.
  20% of them are tulips.
  The rest are carnations and daisies in a ratio 4:3.
  Prove the difference between the number of carnations and daisies.
-/

theorem difference_carnations_daisies : 
  let total_flowers := 120
  let roses := total_flowers / 3
  let tulips := (20 * total_flowers) / 100
  let remaining_flowers := total_flowers - (roses + tulips)
  let ratio_sum := 4 + 3
  let x := remaining_flowers / ratio_sum
  let carnations := 4 * x
  let daisies := 3 * x
  carnations - daisies = 8 :=
by 
  let total_flowers := 120
  let roses := total_flowers / 3
  let tulips := (20 * total_flowers) / 100
  let remaining_flowers := total_flowers - (roses + tulips)
  let ratio_sum := 4 + 3
  let x := remaining_flowers / ratio_sum
  let carnations := 4 * x
  let daisies := 3 * x
  show carnations - daisies = 8 from sorry

end difference_carnations_daisies_l491_491324


namespace equality_condition_l491_491659

def proof_inequality (a b c : ℝ) (h1 : 4 * a * c - b^2 ≥ 0) (h2 : a > 0) : 
  a + c - Real.sqrt((a - c)^2 + b^2) ≤ (4 * a * c - b^2) / (2 * a) :=
sorry

theorem equality_condition (a b c : ℝ) (h1 : 4 * a * c - b^2 ≥ 0) (h2 : a > 0) :
  (a + c - Real.sqrt((a - c)^2 + b^2) = (4 * a * c - b^2) / (2 * a)) ↔ (b = 0 ∧ a ≥ c) ∨ (4 * a * c = b^2) :=
sorry

end equality_condition_l491_491659


namespace lines_perpendicular_l491_491072

-- Definition of lines and their relationships
def Line : Type := ℝ × ℝ × ℝ → Prop

variables (a b c : Line)

-- Condition 1: a is perpendicular to b
axiom perp (a b : Line) : Prop
-- Condition 2: b is parallel to c
axiom parallel (b c : Line) : Prop

-- Theorem to prove: 
theorem lines_perpendicular (h1 : perp a b) (h2 : parallel b c) : perp a c :=
sorry

end lines_perpendicular_l491_491072


namespace rachel_age_when_emily_half_age_l491_491729

theorem rachel_age_when_emily_half_age 
  (E_0 : ℕ) (R_0 : ℕ) (h1 : E_0 = 20) (h2 : R_0 = 24) 
  (age_diff : R_0 - E_0 = 4) : 
  ∃ R : ℕ, ∃ E : ℕ, E = R / 2 ∧ R = E + 4 ∧ R = 8 :=
by
  sorry

end rachel_age_when_emily_half_age_l491_491729


namespace ellipse_condition_l491_491437

theorem ellipse_condition (k : ℝ) :
  (4 < k ∧ k < 9) ↔ (9 - k > 0 ∧ k - 4 > 0 ∧ 9 - k ≠ k - 4) :=
by sorry

end ellipse_condition_l491_491437


namespace pentagonal_pyramid_faces_l491_491645

-- Definition of a pentagonal pyramid
structure PentagonalPyramid where
  base_sides : Nat := 5
  triangular_faces : Nat := 5

-- The goal is to prove that the total number of faces is 6
theorem pentagonal_pyramid_faces (P : PentagonalPyramid) : P.base_sides + 1 = 6 :=
  sorry

end pentagonal_pyramid_faces_l491_491645


namespace maximum_value_sequence_l491_491195

noncomputable def a (n : ℕ) : ℚ :=
  n / (n^2 + 90 : ℚ)

theorem maximum_value_sequence {n : ℕ} (min_n : 1 ≤ n) :
  ∃ n, a n = (1 / 19 : ℚ) :=
begin
  -- Proof will go here.
  sorry
end

end maximum_value_sequence_l491_491195


namespace eight_digit_integers_count_l491_491835

-- Definition of number of choices for the first digit and the remaining digits
def countFirstDigitChoices : ℕ := 9
def countOtherDigitChoices : ℕ := 10
def numberOfEightDigitIntegers : ℕ := 90_000_000

-- Theorem stating the proof problem
theorem eight_digit_integers_count :
  countFirstDigitChoices * countOtherDigitChoices ^ 7 = numberOfEightDigitIntegers := by
  sorry

end eight_digit_integers_count_l491_491835


namespace solve_fraction_eq_l491_491181

theorem solve_fraction_eq :
  ∀ x : ℝ, (x - 3 ≠ 0) → ((x + 6) / (x - 3) = 4) → x = 6 :=
by
  intros x h_nonzero h_eq
  sorry

end solve_fraction_eq_l491_491181


namespace domain_transformation_l491_491822

theorem domain_transformation (f : ℝ → ℝ) :
  (∀ x, -sqrt 3 ≤ x → x ≤ sqrt 3 → f (x^2 - 1) ∈ Set.Icc (-sqrt 3) sqrt 3) →
  (∀ x, f x ∈ Set.Icc (-1 : ℝ) 2) :=
by
  sorry

end domain_transformation_l491_491822


namespace aimee_poll_l491_491312

theorem aimee_poll (P : ℕ) (h1 : 35 ≤ 100) (h2 : 39 % (P/2) = 39) : P = 120 := 
by sorry

end aimee_poll_l491_491312


namespace fred_balloons_l491_491769

theorem fred_balloons (T S D F : ℕ) (hT : T = 72) (hS : S = 46) (hD : D = 16) (hTotal : T = F + S + D) : F = 10 := 
by
  sorry

end fred_balloons_l491_491769


namespace aimee_poll_l491_491311

theorem aimee_poll (P : ℕ) (h1 : 35 ≤ 100) (h2 : 39 % (P/2) = 39) : P = 120 := 
by sorry

end aimee_poll_l491_491311


namespace integral_arctan_sqrt3_l491_491751

-- Define the integral from 0 to sqrt(3) of dx / (1 + x^2)
theorem integral_arctan_sqrt3 : ∫ x in 0..(Real.sqrt 3), 1 / (1 + x^2) = Real.pi / 3 := 
by
  sorry

end integral_arctan_sqrt3_l491_491751


namespace equilateral_triangle_area_ratio_l491_491926

theorem equilateral_triangle_area_ratio :
  let side_small := 1
  let perim_small := 3 * side_small
  let total_fencing := 6 * perim_small
  let side_large := total_fencing / 3
  let area_small := (Real.sqrt 3) / 4 * side_small ^ 2
  let area_large := (Real.sqrt 3) / 4 * side_large ^ 2
  let total_area_small := 6 * area_small
  total_area_small / area_large = 1 / 6 :=
by
  sorry

end equilateral_triangle_area_ratio_l491_491926


namespace point_exists_if_square_or_rhombus_l491_491285

-- Definitions to state the problem
structure Point (α : Type*) := (x : α) (y : α)
structure Rectangle (α : Type*) := (A B C D : Point α)

-- Definition of equidistant property
def isEquidistant (α : Type*) [LinearOrderedField α] (P : Point α) (R : Rectangle α) : Prop :=
  let d1 := abs (P.y - R.A.y)
  let d2 := abs (P.y - R.C.y)
  let d3 := abs (P.x - R.A.x)
  let d4 := abs (P.x - R.B.x)
  d1 = d2 ∧ d2 = d3 ∧ d3 = d4

-- Theorem stating the problem
theorem point_exists_if_square_or_rhombus {α : Type*} [LinearOrderedField α]
  (R : Rectangle α) : 
  (∃ P : Point α, isEquidistant α P R) ↔ 
  (∃ (a b : α), (a ≠ b ∧ R.A.x = a ∧ R.B.x = b ∧ R.C.y = a ∧ R.D.y = b) ∨ 
                (a = b ∧ R.A.x = a ∧ R.B.x = b ∧ R.C.y = a ∧ R.D.y = b)) :=
sorry

end point_exists_if_square_or_rhombus_l491_491285


namespace c_alone_finishes_in_6_days_l491_491277

theorem c_alone_finishes_in_6_days (a b c : ℝ) (W : ℝ) :
  (1 / 36) * W + (1 / 18) * W + (1 / c) * W = (1 / 4) * W → c = 6 :=
by
  intros h
  simp at h
  sorry

end c_alone_finishes_in_6_days_l491_491277


namespace cost_per_yellow_ink_l491_491579

def initial_amount : ℕ := 50
def cost_per_black_ink : ℕ := 11
def num_black_inks : ℕ := 2
def cost_per_red_ink : ℕ := 15
def num_red_inks : ℕ := 3
def additional_amount_needed : ℕ := 43
def num_yellow_inks : ℕ := 2

theorem cost_per_yellow_ink :
  let total_cost_needed := initial_amount + additional_amount_needed
  let total_black_ink_cost := cost_per_black_ink * num_black_inks
  let total_red_ink_cost := cost_per_red_ink * num_red_inks
  let total_non_yellow_cost := total_black_ink_cost + total_red_ink_cost
  let total_yellow_ink_cost := total_cost_needed - total_non_yellow_cost
  let cost_per_yellow_ink := total_yellow_ink_cost / num_yellow_inks
  cost_per_yellow_ink = 13 :=
by
  sorry

end cost_per_yellow_ink_l491_491579


namespace monotonically_decreasing_interval_l491_491451

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 12)
def f' (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 12)
def y (x : ℝ) : ℝ := 2 * f x + f' x

theorem monotonically_decreasing_interval :
  ∀ x, Real.pi / 12 ≤ x ∧ x ≤ 7 * Real.pi / 12 → ∀ x₁ x₂, x₁ ∈ Icc (Real.pi / 12) (7 * Real.pi / 12) →
  x₂ ∈ Icc (Real.pi / 12) (7 * Real.pi / 12) → x₁ < x₂ → y x₁ ≥ y x₂ :=
sorry

end monotonically_decreasing_interval_l491_491451


namespace area_of_transformed_region_l491_491132

noncomputable def matrix : Matrix (Fin 2) (Fin 2) ℝ := ![![3, -1], ![5, 2]]

def region_area (T : Set (ℝ × ℝ)) (area : ℝ) : Prop :=
  ∃ (s : ℝ), s = area ∧ MeasurableSet T ∧ MeasureTheory.measureSpace.is_measurable (MeasureTheory.Measure (ℝ × ℝ)) T

theorem area_of_transformed_region (T : Set (ℝ × ℝ)) (hT : region_area T 15) :
  let A := matrix in
  let det_A := Matrix.det A in
  let T' := {p' | ∃ p, p ∈ T ∧ p' = A.mulVec p} in
  region_area T' 165 :=
sorry

end area_of_transformed_region_l491_491132


namespace find_angles_g_monotonic_interval_l491_491078

-- Definitions for the given problem conditions
variables (A B C a b c : ℝ)
variable (k : ℤ)

-- Conditions
axiom condition1 : a^2 - (b - c)^2 = bc
axiom condition2 : cos A * cos B = (sin A + cos C) / 2
axiom C_value : C = π / 2

-- Question 1: Angles A and B
theorem find_angles (h1 : a^2 - (b - c)^2 = bc) 
                    (h2 : cos A * cos B = (sin A + cos C) / 2) :
  A = π / 3 ∧ B = π / 6 :=
sorry

-- Function transformation and monotonic interval
def f (x : ℝ) : ℝ := sin (2 * x + C)
def g (x : ℝ) : ℝ := cos (2 * x - π / 6) + 2

-- Interval of monotonic decrease for g(x)
theorem g_monotonic_interval :
  ∀ k : ℤ, ∃ (a b : ℝ), a = k * π + π / 12 ∧ b = k * π + 7 * π / 12 ∧ ∀ x, a ≤ x ∧ x ≤ b → decreasing_on g (interval a b) :=
sorry

end find_angles_g_monotonic_interval_l491_491078


namespace tylenol_intake_proof_l491_491902

noncomputable def calculate_tylenol_intake_grams
  (tablet_mg : ℕ) (tablets_per_dose : ℕ) (hours_per_dose : ℕ) (total_hours : ℕ) : ℕ :=
  let doses := total_hours / hours_per_dose
  let total_mg := doses * tablets_per_dose * tablet_mg
  total_mg / 1000

theorem tylenol_intake_proof : calculate_tylenol_intake_grams 500 2 4 12 = 3 :=
  by sorry

end tylenol_intake_proof_l491_491902


namespace find_m_l491_491826

-- Let's define the sets A and B.
def A : Set ℝ := {-1, 1, 3}
def B (m : ℝ) : Set ℝ := {3, m^2}

-- We'll state the problem as a theorem
theorem find_m (m : ℝ) (h : B m ⊆ A) : m = 1 ∨ m = -1 :=
by sorry

end find_m_l491_491826


namespace find_length_AC_l491_491787

open Triangle

variables {A B C P T Q M : Point}

-- Given conditions
variable (h1 : Triangle A B C)
variable (h2 : Line_parallel_to_side_AC_P_and_T : Parallel (line_through A C) (line_through P T))
variable (h3 : Median_A_M : IsMedian A M (A B C))
variable (h4 : Point_Q_on_AM : Q ∈ line_through A M ∧ Q ∈ line_through P T)
variable (h5 : PQ_len : dist P Q = 3)
variable (h6 : QT_len : dist Q T = 5)

-- Proof goal
theorem find_length_AC : dist A C = 11 :=
sorry

end find_length_AC_l491_491787


namespace total_journey_distance_l491_491300

variable (D : ℝ) (T : ℝ) (v₁ : ℝ) (v₂ : ℝ)

theorem total_journey_distance :
  T = 10 → 
  v₁ = 21 → 
  v₂ = 24 → 
  (T = (D / (2 * v₁)) + (D / (2 * v₂))) → 
  D = 224 :=
by
  intros hT hv₁ hv₂ hDistance
  -- Proof goes here
  sorry

end total_journey_distance_l491_491300


namespace union_inter_distrib_inter_union_distrib_l491_491274

section
variables {α : Type*} (A B C : Set α)

-- Problem (a)
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) :=
sorry

-- Problem (b)
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) :=
sorry
end

end union_inter_distrib_inter_union_distrib_l491_491274


namespace major_axis_length_l491_491524

-- Define the problem setup
structure Cylinder :=
  (base_radius : ℝ)
  (height : ℝ)

structure Sphere :=
  (radius : ℝ)

-- Define the conditions
def cylinder : Cylinder :=
  { base_radius := 6, height := 0 }  -- height isn't significant for this problem

def sphere1 : Sphere :=
  { radius := 6 }

def sphere2 : Sphere :=
  { radius := 6 }

def distance_between_centers : ℝ :=
  13

-- Statement of the problem in Lean 4
theorem major_axis_length : 
  cylinder.base_radius = 6 →
  sphere1.radius = 6 →
  sphere2.radius = 6 →
  distance_between_centers = 13 →
  ∃ major_axis_length : ℝ, major_axis_length = 13 :=
by
  intros h1 h2 h3 h4
  existsi 13
  sorry

end major_axis_length_l491_491524


namespace oscar_leap_longer_l491_491728

noncomputable def elmer_strides (poles : ℕ) (strides_per_gap : ℕ) (distance_miles : ℝ) : ℝ :=
  let total_distance := distance_miles * 5280  -- convert miles to feet
  let total_strides := (poles - 1) * strides_per_gap
  total_distance / total_strides

noncomputable def oscar_leaps (poles : ℕ) (leaps_per_gap : ℕ) (distance_miles : ℝ) : ℝ :=
  let total_distance := distance_miles * 5280  -- convert miles to feet
  let total_leaps := (poles - 1) * leaps_per_gap
  total_distance / total_leaps

theorem oscar_leap_longer (poles : ℕ) (strides_per_gap leaps_per_gap : ℕ) (distance_miles : ℝ) :
  poles = 51 -> strides_per_gap = 50 -> leaps_per_gap = 15 -> distance_miles = 1.25 ->
  let elmer_stride := elmer_strides poles strides_per_gap distance_miles
  let oscar_leap := oscar_leaps poles leaps_per_gap distance_miles
  (oscar_leap - elmer_stride) * 12 = 74 :=
by
  intros h_poles h_strides h_leaps h_distance
  have elmer_stride := elmer_strides poles strides_per_gap distance_miles
  have oscar_leap := oscar_leaps poles leaps_per_gap distance_miles
  sorry

end oscar_leap_longer_l491_491728


namespace num_true_propositions_is_one_l491_491450

noncomputable def f (x a : ℝ) : ℝ := Real.log (x + a) - Real.sin x

def proposition1_false (x : ℝ) : Prop := 
  ¬ (∀ x ∈ Ioo 0 Real.exp 1, f x 0 < 0)

def proposition2_true (x : ℝ) (a : ℝ) : Prop := 
  ∀ x ∈ Ioi 0, a ≥ Real.exp 1 → f x a > 0

def proposition3_false (x : ℝ) : Prop := 
  ¬ (∃ x ∈ Ioi 2, f x 1 = 0)

theorem num_true_propositions_is_one : 
  ¬ proposition1_false ∧ proposition2_true ∧ ¬ proposition3_false :=
sorry

end num_true_propositions_is_one_l491_491450


namespace exists_admissible_triangulation_bound_l491_491287

def polygon (n : ℕ) := { P : Type* // ∃ (sides : set (list P)), sides.card = n ∧ P.connected s }

def triangulation (P : polygon) := 
  { T : set (triangle P.1) // ∀ t₁ t₂ ∈ T, t₁ ∩ t₂ ∈ {∅, {vertex}, {side}} ∧ P.1 ∪ t₁ = P.1 } 

def admissible_triangulation (P : polygon n) :=
  ∀ v ∈ P.1, v.is_internal → (∃ t ∈ P.2, v ∈ t) >= 6

noncomputable def M_n (n : ℕ) := 
  ∃ (M : ℕ), ∀ (P : polygon n) (T : admissible_triangulation P), T.card ≤ M

theorem exists_admissible_triangulation_bound (n : ℕ) :
  ∃ (Mn : ℕ), ∀ (P : polygon n) (T : admissible_triangulation P), T.card ≤ Mn :=
sorry

end exists_admissible_triangulation_bound_l491_491287


namespace sum_of_distinct_prime_divisors_l491_491982

theorem sum_of_distinct_prime_divisors (n : ℕ) (h : n = 3600) :
  (∑ p in {2, 3, 5}, p) = 10 :=
sorry

end sum_of_distinct_prime_divisors_l491_491982


namespace compare_negative_fractions_l491_491355

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l491_491355


namespace sin_decreasing_interval_l491_491204

theorem sin_decreasing_interval (k : ℤ) :
  ∀ x, (k * π - π / 12) ≤ x ∧ x ≤ (k * π + 5 * π / 12) →
       deriv (λ x : ℝ, real.sin (π / 3 - 2 * x)) x < 0 := 
by
  sorry

end sin_decreasing_interval_l491_491204


namespace total_baseball_cards_is_100_l491_491560

-- Define the initial number of baseball cards Mike has
def initial_baseball_cards : ℕ := 87

-- Define the number of baseball cards Sam gave to Mike
def given_baseball_cards : ℕ := 13

-- Define the total number of baseball cards Mike has now
def total_baseball_cards : ℕ := initial_baseball_cards + given_baseball_cards

-- State the theorem that the total number of baseball cards is 100
theorem total_baseball_cards_is_100 : total_baseball_cards = 100 := by
  sorry

end total_baseball_cards_is_100_l491_491560


namespace trajectory_equation_minimum_value_MN_l491_491009

-- Condition for the trajectory of the moving point P
def condition_trajectory (P F : ℝ × ℝ) (line_x : ℝ) (ratio : ℝ) : Prop :=
  ∀ (x y : ℝ), P = (x, y) → sqrt ((x - F.1)^2 + y^2) / abs (x - line_x) = ratio

-- Definition of point symmetry with respect to origin
def symmetric_point (P : ℝ × ℝ) : ℝ × ℝ :=
  (-P.1, -P.2)

-- Dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Problem 1: Trajectory Equation
theorem trajectory_equation :
  ∀ (P F : ℝ × ℝ) (line_x ratio : ℝ), F = (sqrt 2, 0) → line_x = 2 * sqrt 2 →
  ratio = sqrt 2 / 2 → condition_trajectory P F line_x ratio →
  ∀ (x y : ℝ), P = (x, y) → (x^2 / 4 + y^2 / 2 = 1) := 
by
  intros
  sorry

-- Problem 2: Minimum Value of |MN|
theorem minimum_value_MN :
  ∀ (F E M N : ℝ × ℝ), F = (sqrt 2, 0) → E = symmetric_point F →
  M = (2 * sqrt 2, ?M_y) → N = (2 * sqrt 2, ?N_y) →
  dot_product (E.1 - M.1, E.2 - M.2) (F.1 - N.1, F.2 - N.2) = 0 →
  ∀ (y1 y2 : ℝ), M = (2 * sqrt 2, y1) → N = (2 * sqrt 2, y2) → 
  y1 > y2 → abs (y1 - y2) = 2 * sqrt 6 := 
by
  intros
  sorry

end trajectory_equation_minimum_value_MN_l491_491009


namespace probability_of_five_is_max_l491_491288

def bag : finset ℕ := {1, 2, 3, 4, 5, 6}

def cards_selected : finset (finset ℕ) :=
  finset.powersetLen 4 bag

def five_is_max (s : finset ℕ) : Prop :=
  ∃ k ∈ s, k = 5 ∧ ∀ m ∈ s, m ≤ 5

theorem probability_of_five_is_max :
  let total_ways := (cards_selected.card : ℚ),
      favorable_ways := (finset.filter five_is_max cards_selected).card
  in favorable_ways / total_ways = 4 / 15 :=
by
  sorry

end probability_of_five_is_max_l491_491288


namespace find_radius_l491_491293

variable {r : ℝ}

-- Conditions from the problem
def area_eq : Prop := π * r ^ 2 = x
def circum_eq : Prop := 2 * π * r = y
def sum_eq : Prop := x + y = 100 * π

-- Theorem to prove
theorem find_radius (h1 : area_eq) (h2 : circum_eq) (h3 : sum_eq) : r = -1 + Real.sqrt 101 := 
  sorry

end find_radius_l491_491293


namespace remaining_volume_sphere_after_drill_l491_491295

noncomputable def remaining_volume (R : ℝ) : ℝ :=
  let V_sphere := (4 / 3) * π * R^3
  let V_cylinder := 6 * π * (R^2 - 9)
  let V_caps := 18 * π * (R - 1)
  V_sphere - V_cylinder - V_caps

theorem remaining_volume_sphere_after_drill (R : ℝ) : remaining_volume R = 36 * π :=
by sorry

end remaining_volume_sphere_after_drill_l491_491295


namespace product_or_double_is_perfect_square_l491_491530

variable {a b c : ℤ}

-- Conditions
def sides_of_triangle (a b c : ℤ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def no_common_divisor (a b c : ℤ) : Prop := gcd (gcd a b) c = 1

def all_fractions_are_integers (a b c : ℤ) : Prop :=
  (a + b - c) ≠ 0 ∧ (b + c - a) ≠ 0 ∧ (c + a - b) ≠ 0 ∧
  ((a^2 + b^2 - c^2) % (a + b - c) = 0) ∧ 
  ((b^2 + c^2 - a^2) % (b + c - a) = 0) ∧ 
  ((c^2 + a^2 - b^2) % (c + a - b) = 0)

-- Mathematical proof problem statement in Lean 4
theorem product_or_double_is_perfect_square (a b c : ℤ) 
  (h1 : sides_of_triangle a b c)
  (h2 : no_common_divisor a b c)
  (h3 : all_fractions_are_integers a b c) :
  ∃ k : ℤ, k^2 = (a + b - c) * (b + c - a) * (c + a - b) ∨ 
           k^2 = 2 * (a + b - c) * (b + c - a) * (c + a - b) := sorry

end product_or_double_is_perfect_square_l491_491530


namespace color_four_triangles_l491_491373

-- Define the problem as a Lean 4 statement
theorem color_four_triangles :
  let colors := {red, white, blue} 
  ∃ (coloring : Fin 12 → colors), 
  (∀ (i j : Fin 12), connected i j → coloring i ≠ coloring j) →
  ∃ (count : Nat), count = 162 :=
by
  let colors := {red, white, blue}
  sorry

end color_four_triangles_l491_491373


namespace dot_product_value_l491_491805

variables (a b : EuclideanSpace ℝ (fin 3))
variables (theta : ℝ)
variables (norm_a : ℝ) (norm_b : ℝ)

-- Given conditions
axiom angle_between : θ = real.pi / 3
axiom norm_a_def : ∥a∥ = 2 * real.sqrt 2
axiom norm_b_def : ∥b∥ = real.sqrt 3

-- Prove the dot product is √6
theorem dot_product_value : ∥a∥ * ∥b∥ * real.cos θ = real.sqrt 6 := 
by 
  -- Start the proof here
  sorry

end dot_product_value_l491_491805


namespace distance_between_skew_lines_l491_491897

variables {A B C D E F : Point}
variable {l m : Line}
variable {distance : ℝ}

-- Conditions
def line_contains_points := ∀ {A B C : Point}, l.contains A ∧ l.contains B ∧ l.contains C
def AB_eq_BC := dist A B = dist B C
def perpendiculars_to_m := ∀ {A D B E C F : Point}, 
  (AD ⊥ m) ∧ (BE ⊥ m) ∧ (CF ⊥ m)
def perpendicular_lengths := 
  dist A D = √15 ∧ dist B E = (7 / 2) ∧ dist C F = √10

-- Prove the distance between lines l and m
theorem distance_between_skew_lines 
  (h1 : line_contains_points)
  (h2 : AB_eq_BC)
  (h3 : perpendiculars_to_m)
  (h4 : perpendicular_lengths) :
  distance l m = √6 := 
sorry

end distance_between_skew_lines_l491_491897


namespace original_numbers_product_l491_491222

theorem original_numbers_product (a b c d x : ℕ) 
  (h1 : a + b + c + d = 243)
  (h2 : a + 8 = x)
  (h3 : b - 8 = x)
  (h4 : c * 8 = x)
  (h5 : d / 8 = x) : 
  (min (min a (min b (min c d))) * max a (max b (max c d))) = 576 :=
by 
  sorry

end original_numbers_product_l491_491222


namespace probability_of_safe_flight_l491_491681

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * real.pi * r^3

theorem probability_of_safe_flight : 
  let R_outer := 3
  let R_safe := 2
  let V_outer := volume_of_sphere R_outer
  let V_safe := volume_of_sphere R_safe
  (V_safe / V_outer) = 8 / 27 :=
by
  sorry

end probability_of_safe_flight_l491_491681


namespace trajectory_equation_no_such_line_exists_l491_491794

open Real EuclideanGeometry

-- Coordinates of points A, Q
def A : ℝ × ℝ := (8, 0)
def Q : ℝ × ℝ := (-1, 0)

-- Question 1: Prove the trajectory equation
theorem trajectory_equation (B P : ℝ × ℝ) (C : Type) :
  -- Conditions
  B.1 = 0 ∧ P.2 = 0 ∧ P.1 = 0 ∧ C.2 = 0 →
  (A.1 - B.1) * (P.1 - B.1) + B.2 * (P.2 - B.2) = 0 ∧ 
  B.1 - C.1 = P.1 - C.1 ∧ B.2 - C.2 = P.2 - C.2 → 
  -- Conclusion
  P.2^2 = -4 * P.1 :=
sorry

-- Define line l passing through A
def line_l (k : ℝ) : ℝ × ℝ → Prop := λ M, M.2 = k * M.1 - 8 * k

-- Question 2: Prove such a line does not exist
theorem no_such_line_exists (k : ℝ) (M N : ℝ × ℝ) :
  -- Conditions
  line_l k M ∧ line_l k N ∧ 
  ∃ (k : ℝ), k² < 1 / 8 ∧
  (Q.1 - M.1) * (Q.1 - N.1) + (Q.2 - M.2) * (Q.2 - N.2) = 97 →
  -- Conclusion
  false :=
sorry

end trajectory_equation_no_such_line_exists_l491_491794


namespace painting_cost_l491_491200

theorem painting_cost (
  B : ℝ) 
  (h_length : 3 * B = 12.24744871391589) 
  (rate : ℝ := 2) : 
  2 * (12.24744871391589 * (12.24744871391589 / 3)) = 100 :=
by
  -- Definitions used in Lean
  let L := 12.24744871391589
  let length_eq : L = 3 * B := h_length
  -- Calculate area
  let area := L * B
  -- Calculate total cost
  let total_cost := area * rate
  -- Expected result
  exact eq.trans (mul_assoc 2 area rate) sorry

end painting_cost_l491_491200


namespace phi_is_17_l491_491841

noncomputable def phi (θ : ℝ) : ℝ :=
  if θ = 30 then 17 else 0

theorem phi_is_17 (θ : ℝ) (h : θ = 30) :
  ∃ φ, (φ = 17) ∧ (sqrt 3 * real.cos (θ * real.pi / 180) = real.sin (φ * real.pi / 180) + real.cos (φ * real.pi / 180)) :=
begin
  use phi 30,
  split,
  { rw phi,
    simp [h] },
  { have h1 : sqrt 3 * real.cos (θ * real.pi / 180) = sqrt 3 * (sqrt 3 / 2) := by simp [h],
    have h2 : sqrt 3 * (sqrt 3 / 2) = 3 / 2 := by ring,
    rw ←h1,
    rw h2,
    norm_num,
    sorry }
end

end phi_is_17_l491_491841


namespace triangle_height_l491_491987

theorem triangle_height (area base : ℝ) (h_area : area = 9.31) (h_base : base = 4.9) : (2 * area) / base = 3.8 :=
by
  sorry

end triangle_height_l491_491987


namespace poll_total_l491_491317

-- Define the conditions
variables (men women : ℕ)
variables (pct_favor : ℝ := 35) (women_opposed : ℕ := 39)
noncomputable def total_people (men women : ℕ) : ℕ := men + women

-- We need to prove the total number of people polled, given the conditions
theorem poll_total (h1 : men = women)
  (h2 : (pct_favor / 100) * women + (39 : ℝ) / (65 / 100) = (women: ℝ)) :
  total_people men women = 120 :=
sorry

end poll_total_l491_491317


namespace inv_81_mod_101_l491_491441

theorem inv_81_mod_101 
    (h1 : 9⁻¹ ≡ 90 [ZMOD 101]) : 81⁻¹ ≡ 20 [ZMOD 101] := 
by 
  sorry

end inv_81_mod_101_l491_491441


namespace log_base_16_of_4_eq_half_l491_491736

noncomputable def logBase (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_base_16_of_4_eq_half :
  logBase 16 4 = 1 / 2 := by
sorry

end log_base_16_of_4_eq_half_l491_491736


namespace oil_bill_january_l491_491991

-- Define the constants and variables
variables (F J : ℝ)

-- Define the conditions
def condition1 : Prop := F / J = 5 / 4
def condition2 : Prop := (F + 45) / J = 3 / 2

-- Define the main theorem stating the proof problem
theorem oil_bill_january 
  (h1 : condition1 F J) 
  (h2 : condition2 F J) : 
  J = 180 :=
sorry

end oil_bill_january_l491_491991


namespace sum_of_squares_l491_491228

theorem sum_of_squares (x : ℕ) (h : 2 * x = 14) : (3 * x)^2 + (2 * x)^2 + (5 * x)^2 = 1862 := 
by 
  sorry

end sum_of_squares_l491_491228


namespace inequality_for_positive_reals_l491_491764

theorem inequality_for_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / real.sqrt (a^2 + 8 * b * c)) + (b / real.sqrt (b^2 + 8 * c * a)) + (c / real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end inequality_for_positive_reals_l491_491764


namespace carol_rectangle_length_l491_491337

theorem carol_rectangle_length :
  ∀ (L_Carol : ℝ),
  ∀ (width_Carol : ℝ = 5) (length_Jordan : ℝ = 12) (width_Jordan : ℝ = 10),
  (width_Carol * L_Carol) = (length_Jordan * width_Jordan) →
  L_Carol = 24 :=
by
  intros L_Carol width_Carol length_Jordan width_Jordan h
  sorry

end carol_rectangle_length_l491_491337


namespace profits_ratio_l491_491630

-- Definitions
def investment_ratio (p q : ℕ) := 7 * p = 5 * q
def investment_period_p := 10
def investment_period_q := 20

-- Prove the ratio of profits
theorem profits_ratio (p q : ℕ) (h1 : investment_ratio p q) :
  (7 * p * investment_period_p / (5 * q * investment_period_q)) = 7 / 10 :=
sorry

end profits_ratio_l491_491630


namespace sum_equals_factorial_l491_491412

-- Define the binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the polynomial p(x) = x^n
noncomputable def poly (x : ℝ) (n : ℕ) : ℝ := x^n

-- Define the difference operator Delta
noncomputable def Delta (p : ℝ → ℝ) (x : ℝ) : ℝ := p(x) - p(x - 1)

-- Define the sum to be proven
noncomputable def sum_expression (n : ℕ) (x : ℝ) : ℝ :=
  ∑ i in Finset.range (n + 1), (-1)^i * binomial_coeff n i * (x - i)^n

-- The conjecture to be proven
theorem sum_equals_factorial (n : ℕ) (x : ℝ) : sum_expression n x = n! := by
  sorry

end sum_equals_factorial_l491_491412


namespace R2_perfect_fit_l491_491786

variables {n : ℕ} (x y : Fin n → ℝ) (b a : ℝ)

-- Condition: Observations \( (x_i, y_i) \) such that \( y_i = bx_i + a \)
def observations (i : Fin n) : Prop :=
  y i = b * x i + a

-- Condition: \( e_i = 0 \) for all \( i \)
def no_error (i : Fin n) : Prop := (b * x i + a + 0 = y i)

theorem R2_perfect_fit (h_obs: ∀ i, observations x y b a i)
                       (h_no_error: ∀ i, no_error x y b a i) : R_squared = 1 := by
  sorry

end R2_perfect_fit_l491_491786


namespace trigonometric_expression_l491_491030

theorem trigonometric_expression (a : ℝ) (h : a ≠ 0) :
  let α := atan 3 in
  (cos α - sin α) / (sin α + cos α) = -1 / 2 :=
by
  -- skip the proof for now
  sorry

end trigonometric_expression_l491_491030


namespace vector_dot_and_magnitude_l491_491424

open_locale real_inner_product_space

variables (a b : ℝ × ℝ × ℝ)

noncomputable def dot_product (x y : ℝ × ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2 + x.3 * y.3

noncomputable def magnitude (x : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (x.1 ^ 2 + x.2 ^ 2 + x.3 ^ 2)

theorem vector_dot_and_magnitude :
  let a : ℝ × ℝ × ℝ := (1, 2, real.sqrt 3),
      b : ℝ × ℝ × ℝ := (-1, real.sqrt 3, 0) in
  dot_product a b + magnitude b = 2 * real.sqrt 3 + 1 :=
by
  simp only [dot_product, magnitude, real.sqrt, (*), (+), (^), (-)]
  sorry

end vector_dot_and_magnitude_l491_491424


namespace sin_squared_sum_l491_491705

noncomputable def sum_sin_squared_angles : ℝ :=
  (Real.sin (4 * Real.pi / 180))^2 + 
  (Real.sin (8 * Real.pi / 180))^2 + 
  (Real.sin (12 * Real.pi / 180))^2 + 
  ... + 
  (Real.sin (176 * Real.pi / 180))^2

theorem sin_squared_sum : sum_sin_squared_angles = 45 / 2 :=
sorry

end sin_squared_sum_l491_491705


namespace max_tiles_l491_491176

-- Definitions of constants used
def tile_length : ℝ := 20 -- cm
def tile_width : ℝ := 30 -- cm
def floor_length : ℝ := 400 -- cm
def floor_width : ℝ := 600 -- cm

-- Definition of the areas
def area_tile : ℝ := tile_length * tile_width
def area_floor : ℝ := floor_length * floor_width

-- Lean theorem statement proving the number of tiles
theorem max_tiles : area_floor / area_tile = 400 := by
  -- Given areas calculated and used in this theorem
  -- Showing the final mathematical equivalence required
  have h1: area_floor = 240000 := by rw [floor_length, floor_width]
  have h2: area_tile = 600 := by rw [tile_length, tile_width]
  have h3: 240000 / 600 = 400 := by norm_num
  rw [←h1, ←h2]
  exact h3

end max_tiles_l491_491176


namespace part1_part2_part3_l491_491043

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + log ((1 - x) / (1 + x))

-- Part (1)
theorem part1 (h : deriv (f a) 0 = 0) : a = 2 := 
by sorry

-- Part (2)
theorem part2 (a : ℝ) (h : a = 4) : ∃ n, (n = 3 ∧ ∀ x, f a x = 0 → x ∈ Ioo (-1) 1)  :=
by sorry

-- Part (3)
theorem part3 (a : ℝ) : (0 ≤ a ∧ a ≤ 2) → ∀ x y, x < y → f a x ≤ f a y :=
by sorry

end part1_part2_part3_l491_491043


namespace harkamal_payment_l491_491276

theorem harkamal_payment :
  let grapes_kg := 9
  let grape_rate_per_kg := 70
  let mangoes_kg := 9
  let mango_rate_per_kg := 55
  let cost_of_grapes := grapes_kg * grape_rate_per_kg
  let cost_of_mangoes := mangoes_kg * mango_rate_per_kg
  let total_payment := cost_of_grapes + cost_of_mangoes
  total_payment = 1125 :=
by
  let grapes_kg := 9
  let grape_rate_per_kg := 70
  let mangoes_kg := 9
  let mango_rate_per_kg := 55
  let cost_of_grapes := grapes_kg * grape_rate_per_kg
  let cost_of_mangoes := mangoes_kg * mango_rate_per_kg
  let total_payment := cost_of_grapes + cost_of_mangoes
  sorry

end harkamal_payment_l491_491276


namespace coefficient_of_x5_in_binomial_expansion_l491_491088

noncomputable def binomial_coefficient (n k : ℕ) : ℤ :=
  if h : k ≤ n then nat.choose n k else 0

theorem coefficient_of_x5_in_binomial_expansion :
  (∑ r in finset.range (7 + 1), binomial_coefficient 7 r * (-2 : ℤ)^r * (x : ℤ)^(14 - 3 * r)) = -280 :=
by
  sorry

end coefficient_of_x5_in_binomial_expansion_l491_491088


namespace periodic_odd_function_value_at_7_l491_491127

noncomputable def f : ℝ → ℝ := sorry -- Need to define f appropriately, skipped for brevity

theorem periodic_odd_function_value_at_7
    (f_odd : ∀ x : ℝ, f (-x) = -f x)
    (f_periodic : ∀ x : ℝ, f (x + 4) = f x)
    (f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) :
    f 7 = -1 := sorry

end periodic_odd_function_value_at_7_l491_491127


namespace Marge_pulled_3_weeds_l491_491140

theorem Marge_pulled_3_weeds (
  seeds_planted : ℕ,
  seeds_never_grew : ℕ,
  fraction_eaten : ℚ,
  fraction_strangled : ℚ,
  num_plants_end_up_with : ℕ,
  num_weeds_kept : ℕ
) : 
  seeds_planted = 23 →
  seeds_never_grew = 5 →
  fraction_eaten = 1/3 →
  fraction_strangled = 1/3 →
  num_plants_end_up_with = 9 →
  num_weeds_kept = 1 →
  (let
    seeds_that_grew := seeds_planted - seeds_never_grew;
    seeds_eaten := fraction_eaten * seeds_that_grew;
    uneaten_plants := seeds_that_grew - seeds_eaten;
    plants_strangled := fraction_strangled * uneaten_plants;
    not_strangled_plants := uneaten_plants - plants_strangled;
    plants_with_kept_weed := num_plants_end_up_with 
  in 
    plants_strangled - num_weeds_kept = 3 ) :=
by
  intros h_seeds_planted h_seeds_never_grew h_fraction_eaten h_fraction_strangled h_num_plants_end_up_with h_num_weeds_kept
  -- Here the proof would be constructed to show that Marge pulled 3 weeds.
  sorry

end Marge_pulled_3_weeds_l491_491140


namespace orthocenter_triangle_l491_491166

variables {A B C A1 B1 C1 H : Type}
variables {dist : Type}

// Assuming we have necessary definitions for points and their intersection properties
variables (acute_triangle : Type) (on_sides : A → B → C → A1 → B1 → C1 → Prop)
          (intersects_at : A → A1 → B → B1 → C → C1 → H → Prop )
          (orthocenter : triangle → H → Prop)

axiom acute_condition (t : acute_triangle) : on_sides A B C A1 B1 C1 → intersects_at A A1 B B1 C C1 H → Prop
axiom on_sides_cond (x : acute_triangle) : Prop 
axiom intersection_cond (y : acute_triangle) : Prop
axiom ortho_cond (z : acute_triangle) : Prop

-- The main statement we need to prove
theorem orthocenter_triangle (t : acute_triangle) : 
  (acute_condition t ∧ on_sides_cond t ∧ intersection_cond t) 
  → (AH • A1H = BH • B1H ∧ BH • B1H = CH • C1H) ↔ ortho_cond t := 
sorry -- Proof is not provided here

end orthocenter_triangle_l491_491166


namespace sign_of_f_l491_491800

variable {R : Type*} [LinearOrderedField R]

noncomputable def f : R → R := sorry

def odd (f : R → R) := ∀ x, f (-x) = - (f x)
def monotone_decreasing_nonneg (f : R → R) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem sign_of_f (h_odd : odd f) 
                  (h_mono_dec : monotone_decreasing_nonneg f)
                  (h_cond : ∀ ⦃x y⦄, x + y < 0 → x < -y)
                  (x1 x2 : R) 
                  (h_sum : x1 + x2 < 0) : 
                  f x1 + f x2 > 0 :=
sorry

end sign_of_f_l491_491800


namespace dandelion_seed_production_l491_491572

theorem dandelion_seed_production :
  (one_seed : ℕ) (produced_seeds : ℕ)
  (germinated_fraction : ℚ)
  (new_seedlings_count : ℕ)
  (seed_count_after_two_months : ℕ) :
  one_seed = 1 →
  produced_seeds = 50 →
  germinated_fraction = 1/2 →
  new_seedlings_count = produced_seeds * germinated_fraction.numerator / germinated_fraction.denominator →
  seed_count_after_two_months = new_seedlings_count * produced_seeds →
  seed_count_after_two_months = 1250 :=
by
  intros
  sorry

end dandelion_seed_production_l491_491572


namespace percentage_decrease_returns_original_value_l491_491303

theorem percentage_decrease_returns_original_value (x : ℝ) (hx : x > 0) :
  ∃ d : ℝ, d = (1 - (1 / 1.30)) ∧ d * 100 ≈ 23.08 :=
by
  sorry  -- proof to be provided later

end percentage_decrease_returns_original_value_l491_491303


namespace negative_fraction_comparison_l491_491369

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l491_491369


namespace carmens_brush_length_is_381_mm_l491_491701

noncomputable theory

def brush_length_in_millimeters (L_Carmen L′Carmen : ℝ) : Prop :=
  L′Carmen = 1.25 * L_Carmen ∧ L_Carmen = (L_Carmen * 25.4) ∧ L_Carmen = 381

theorem carmens_brush_length_is_381_mm
  (L_C : ℝ) (L′C : ℝ)
  (percent_conversion : ∀ (x : ℝ), x * 0.01 = x / 100)
  (conversion_factor : ℝ)
  (c_length_in_mm : ℝ)
  (h_LC : L_C = 12)
  (h_conversion_factor : conversion_factor = 25.4)
  (h_percent_conversion : percent_conversion 125 = 1.25)
  (h_carmen_brush_length : brush_length_in_millimeters (√L′C) (L_C * 1.25)) :
  h_carmen_brush_length := 
begin
  sorry           -- Proof comes here
end

end carmens_brush_length_is_381_mm_l491_491701


namespace faster_train_cross_time_l491_491235

/-- Statement of the problem in Lean 4 -/
theorem faster_train_cross_time :
  let speed_faster_train_kmph := 72
  let speed_slower_train_kmph := 36
  let length_faster_train_m := 180
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18 : ℝ)
  let time_taken := length_faster_train_m / relative_speed_mps
  time_taken = 18 :=
by
  sorry

end faster_train_cross_time_l491_491235


namespace compare_fractions_l491_491342

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l491_491342


namespace mass_percentage_Ca_in_mixture_l491_491715

theorem mass_percentage_Ca_in_mixture :
  let mass_CaCO3 := 20.0
  let mass_MgCl2 := 10.0
  let molar_mass_Ca := 40.08
  let molar_mass_C := 12.01
  let molar_mass_O := 16.00
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_Ca_in_CaCO3 := mass_CaCO3 * (molar_mass_Ca / molar_mass_CaCO3)
  let total_mass := mass_CaCO3 + mass_MgCl2
  let percentage_Ca := (mass_Ca_in_CaCO3 / total_mass) * 100
  percentage_Ca = 26.69 :=
by
  let mass_CaCO3 := 20.0
  let mass_MgCl2 := 10.0
  let molar_mass_Ca := 40.08
  let molar_mass_C := 12.01
  let molar_mass_O := 16.00
  let molar_mass_CaCO3 := molar_mass_Ca + molar_mass_C + 3 * molar_mass_O
  let mass_Ca_in_CaCO3 := mass_CaCO3 * (molar_mass_Ca / molar_mass_CaCO3)
  let total_mass := mass_CaCO3 + mass_MgCl2
  let percentage_Ca := (mass_Ca_in_CaCO3 / total_mass) * 100
  have : percentage_Ca = 26.69 := by sorry
  exact this

end mass_percentage_Ca_in_mixture_l491_491715


namespace range_of_independent_variable_x_in_sqrt_function_l491_491626

theorem range_of_independent_variable_x_in_sqrt_function :
  (∀ x : ℝ, ∃ y : ℝ, y = sqrt (2 * x - 3)) → x ≥ 3 / 2 :=
sorry

end range_of_independent_variable_x_in_sqrt_function_l491_491626


namespace triangle_ABC_area_l491_491620

noncomputable theory
open_locale classical

structure Point where
  x : ℝ
  y : ℝ

def reflect_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_y_equals_neg_x (p : Point) : Point :=
  { x := -p.y, y := -p.x }

def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)))

theorem triangle_ABC_area :
  let A := Point.mk 2 5 in
  let B := reflect_y_axis A in
  let C := reflect_y_equals_neg_x B in
  area_of_triangle A B C = 14 :=
by
  sorry

end triangle_ABC_area_l491_491620


namespace abs_gt_one_iff_square_inequality_l491_491600

theorem abs_gt_one_iff_square_inequality (x : ℝ) : |x| > 1 ↔ x^2 - 1 > 0 := 
sorry

end abs_gt_one_iff_square_inequality_l491_491600


namespace combined_sum_of_interior_numbers_of_eighth_and_ninth_rows_l491_491212

theorem combined_sum_of_interior_numbers_of_eighth_and_ninth_rows :
  (2 ^ (8 - 1) - 2) + (2 ^ (9 - 1) - 2) = 380 :=
by
  -- The steps of the proof would go here, but for the purpose of this task:
  sorry

end combined_sum_of_interior_numbers_of_eighth_and_ninth_rows_l491_491212


namespace exists_equal_squares_mod_100_l491_491173

theorem exists_equal_squares_mod_100 (s : Finset ℤ) (h : s.card = 51) :
  ∃ x y ∈ s, x ≠ y ∧ (x^2 % 100 = y^2 % 100) :=
by
  sorry

end exists_equal_squares_mod_100_l491_491173


namespace length_of_second_train_l491_491686

theorem length_of_second_train :
  ∀ (length_first_train speed_first_train speed_second_train time_crossing relative_speed length_second_train : ℝ),
    length_first_train = 108 →
    speed_first_train = 50 * 1000 / 3600 →
    speed_second_train = 82 * 1000 / 3600 →
    relative_speed = speed_first_train + speed_second_train →
    time_crossing = 6 →
    length_second_train = (relative_speed * time_crossing) - length_first_train →
    length_second_train = 112.02 :=
by
  intros length_first_train speed_first_train speed_second_train time_crossing relative_speed length_second_train
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end length_of_second_train_l491_491686


namespace value_of_a_l491_491484

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end value_of_a_l491_491484


namespace Joan_paid_158_l491_491654

theorem Joan_paid_158 (J K : ℝ) (h1 : J + K = 400) (h2 : 2 * J = K + 74) : J = 158 :=
by
  sorry

end Joan_paid_158_l491_491654


namespace relationship_a_b_log_inequality_l491_491779

-- Definition of the functions f and g
def f (x : ℝ) : ℝ := Real.log x
def g (x a b : ℝ) : ℝ := f x + a * x^2 + b * x

-- Condition that the tangent at (1, g(1)) is parallel to the x-axis
def tangent_parallel (a b : ℝ) : Prop :=
  (deriv (fun x => g x a b)) 1 = 0

-- The relationship between a and b
theorem relationship_a_b (a b : ℝ) (h : tangent_parallel a b) : b = -2 * a - 1 :=
  sorry

-- The inequality for n ∈ ℕ*
theorem log_inequality (n : ℕ) (hn : n > 0) : Real.log (1 + n) > ∑ i in Finset.range(n + 1).erase 0, (i - 1 : ℝ) / (i : ℝ) ^ 2 :=
  sorry

end relationship_a_b_log_inequality_l491_491779


namespace part_a_part_b_part_c_l491_491279

/-- (a) Given that p = 33 and q = 216, show that the equation f(x) = 0 has 
three distinct integer solutions and the equation g(x) = 0 has two distinct integer solutions.
-/
theorem part_a (p q : ℕ) (h_p : p = 33) (h_q : q = 216) :
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = 216 ∧ x1 + x2 + x3 = 33 ∧ x1 = 0))
  ∧ (∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = 216 ∧ y1 + y1 = 22)) := sorry

/-- (b) Suppose that the equation f(x) = 0 has three distinct integer solutions 
and the equation g(x) = 0 has two distinct integer solutions. Prove the necessary conditions 
for p and q.
-/
theorem part_b (p q : ℕ) 
  (h_f : ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = q ∧ x1 + x2 + x3 = p))
  (h_g : ∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = q ∧ y1 + y1 = 2 * p)) :
  (∃ k : ℕ, p = 3 * k) ∧ (∃ l : ℕ, q = 9 * l) ∧ (∃ m n : ℕ, p^2 - 3 * q = m^2 ∧ p^2 - 4 * q = n^2) := sorry

/-- (c) Prove that there are infinitely many pairs of positive integers (p, q) for which:
1. The equation f(x) = 0 has three distinct integer solutions.
2. The equation g(x) = 0 has two distinct integer solutions.
3. The greatest common divisor of p and q is 3.
-/
theorem part_c :
  ∃ (p q : ℕ) (infinitely_many : ℕ → Prop),
  (∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1 * x2 * x3 = q ∧ x1 + x2 + x3 = p))
  ∧ (∃ y1 y2 : ℤ, y1 ≠ y2 ∧ (3 * y1 * y2 = q ∧ y1 + y1 = 2 * p))
  ∧ ∃ k : ℕ, gcd p q = 3 ∧ infinitely_many k := sorry

end part_a_part_b_part_c_l491_491279


namespace monotonic_f_iff_condition_inequality_l491_491040

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 2 * (a + 1) * Real.log x - a * x
def g (x : ℝ) : ℝ := (1 / 2) * x^2 - x

-- Proof Problem 1: Prove that f(x) is monotonic in (0, +∞) if and only if a ∈ [-1, 0]
theorem monotonic_f_iff (a : ℝ) : 
  (∀ x y : ℝ, 0 < x → 0 < y → x ≠ y → (f a x - f a y) * (x - y) ≤ 0) ↔ (a ∈ set.Icc (-1 : ℝ) 0) :=
sorry

-- Proof Problem 2: If -1 < a < 7, for any x1, x2 ∈ (1, +∞), x1 ≠ x2, it holds that 
-- (f(x1) - f(x2)) / (g(x1) - g(x2)) > -1.
theorem condition_inequality (a x1 x2 : ℝ) (h1 : -1 < a) (h2 : a < 7)
  (h3 : 1 < x1) (h4 : 1 < x2) (h5 : x1 ≠ x2) : 
  (f a x1 - f a x2) / (g x1 - g x2) > -1 :=
sorry

end monotonic_f_iff_condition_inequality_l491_491040


namespace cross_product_self_zero_l491_491468

variable (v w : V) [Vector 𝐑 3 V]

axiom cross_product_property : ∀ v w : V, 
  ((v -ᵥ 0) - (w -ᵥ 0)) = (v × w) - ᵥ 0

theorem cross_product_self_zero :
  ⟦v × w⟧ → ⟦2 • v + w × 2 • v + w = ⟦0⟧⟧ := by
sorry

end cross_product_self_zero_l491_491468


namespace solve_for_x_l491_491762

variable {x : ℝ}

def is_positive (x : ℝ) : Prop := x > 0

def area_of_triangle_is_150 (x : ℝ) : Prop :=
  let base := 2 * x
  let height := 3 * x
  (1/2) * base * height = 150

theorem solve_for_x (hx : is_positive x) (ha : area_of_triangle_is_150 x) : x = 5 * Real.sqrt 2 := by
  sorry

end solve_for_x_l491_491762


namespace max_omega_value_l491_491413

open Real

noncomputable def g (x : ℝ) (ω : ℕ) : ℝ := sin (ω * x + π / 6)

theorem max_omega_value : 
  (∃ ω : ℕ, monotone (λ x, g x ω) ∧ (interval (π / 6) (π / 4)) ⊆ set_of (λ x, ∃ k : ℤ, (π / 6) * ω + (π / 6) ≥ 2 * k * π - (π / 2) ∧ (π / 4) * ω + (π / 6) ≤ 2 * k * π + (π / 2)) ∧ ω = 9) := 
  sorry

end max_omega_value_l491_491413


namespace sqrt_eqn_solution_l491_491070

theorem sqrt_eqn_solution (x : ℝ) : sqrt (5 + sqrt x) = 4 → x = 121 :=
by
  intro h
  sorry

end sqrt_eqn_solution_l491_491070


namespace not_in_M_4n2_l491_491131

def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

theorem not_in_M_4n2 (n : ℤ) : ¬ (4 * n + 2 ∈ M) :=
by
sorry

end not_in_M_4n2_l491_491131


namespace common_tangent_l491_491074

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * cos x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b * x + 1

theorem common_tangent (a b m : ℝ) (h1 : f a 0 = g b 0)
    (h2 : (fun x => -a * sin x) 0 = (fun x => 2 * x + b) 0) : a + b = 1 := 
by {
  -- skipping the proof
  sorry
}

end common_tangent_l491_491074


namespace quadratic_inequality_solution_l491_491419

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 4 * x > 45 ↔ x < -9 ∨ x > 5 := 
  sorry

end quadratic_inequality_solution_l491_491419


namespace visited_iceland_l491_491498

theorem visited_iceland :
  ∀ (total people visited_both visited_norway visited_neither : ℕ),
  total = 60 →
  visited_norway = 23 →
  visited_both = 31 →
  visited_neither = 33 →
  (let visited_iceland := total - visited_neither - visited_norway + visited_both in
  visited_iceland = 35) :=
begin
  intros total visited_both visited_norway visited_neither ht hn hb hnne,
  simp only [],
  have h1 : total - visited_neither - visited_norway + visited_both = 60 - 33 - 23 + 31,
  { rw [ht, hn, hb, hnne] },
  simp only [] at h1,
  exact h1,
sorry
end

end visited_iceland_l491_491498


namespace degenerate_ellipse_value_c_l491_491376

theorem degenerate_ellipse_value_c (c : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 14 * y + c = 0) ∧
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 14 * y + c = 0 → (x+1)^2 + (y-7)^2 = 0) ↔ c = 52 :=
by
  sorry

end degenerate_ellipse_value_c_l491_491376


namespace arithmetic_progression_x_value_l491_491193

theorem arithmetic_progression_x_value {x : ℝ} 
  (term1 : ℝ) (term2 : ℝ) (term3 : ℝ)
  (h1 : term1 = 2 * x - 3)
  (h2 : term2 = 3 * x + 1)
  (h3 : term3 = 5 * x - 1)
  (h4 : term2 - term1 = term3 - term2) : 
  x = 6 :=
by
  have h5 : term2 - term1 = x + 4 := by { rw [h1, h2], ring }
  have h6 : term3 - term2 = 2 * x - 2 := by { rw [h2, h3], ring }
  have h7 : x + 4 = 2 * x - 2 := by { rw [h4, h5, h6] }
  linarith

end arithmetic_progression_x_value_l491_491193


namespace complex_polynomial_counterexample_l491_491896

variable (f g h : ℂ[X])

theorem complex_polynomial_counterexample (f g h : ℂ[X]) :
  f ^ 2 = X * g ^ 2 + X * h ^ 2 :=
begin
  use [0, 1, complex.I],
  simp,
end

end complex_polynomial_counterexample_l491_491896


namespace probability_born_in_2008_l491_491142

/-- Define the relevant dates and days calculation -/
def num_days_between : ℕ :=
  29 + 31 + 30 + 31 + 31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

def num_days_2008 : ℕ :=
  31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

theorem probability_born_in_2008 :
  (num_days_2008 : ℚ) / num_days_between = 244 / 365 :=
by
  -- The proof is omitted for brevity
  sorry

end probability_born_in_2008_l491_491142


namespace josh_wallet_after_transactions_l491_491876

noncomputable def wallet_initial : ℕ := 300
noncomputable def business_investment : ℕ := 2000
noncomputable def debt : ℕ := 500
noncomputable def stockA_ratio : ℚ := 0.4
noncomputable def stockB_ratio : ℚ := 0.3
noncomputable def stockC_ratio : ℚ := 0.3

noncomputable def stockA_price_change : ℚ := 1.2
noncomputable def stockB_price_change : ℚ := 1.3
noncomputable def stockC_price_change : ℚ := 0.9

theorem josh_wallet_after_transactions :
  let investment_A := stockA_ratio * business_investment in
  let investment_B := stockB_ratio * business_investment in
  let investment_C := stockC_ratio * business_investment in
  let new_value_A := investment_A * stockA_price_change in
  let new_value_B := investment_B * stockB_price_change in
  let new_value_C := investment_C * stockC_price_change in
  let total_after_sales := new_value_A + new_value_B + new_value_C in
  let remaining_after_debt := total_after_sales - debt in
  let final_wallet := wallet_initial + remaining_after_debt in
  final_wallet = 2080 :=
by {
  sorry
}

end josh_wallet_after_transactions_l491_491876


namespace find_matrix_N_l491_491402

theorem find_matrix_N:
  ∃ (N : Matrix (Fin 2) (Fin 2) ℚ), 
    N ⬝ (λ i, if i = 0 then 2 else 1) = (λ i, if i = 0 then 5 else 4) ∧ 
    N ⬝ (λ i, if i = 0 then 1 else -4) = (λ i, if i = 0 then 0 else -9) ∧ 
    N = (λ i j, 
      if i = 0 
      then (if j = 0 then 20/9 else 5/9) 
      else (if j = 0 then 7/9 else 22/9)) :=
by
  let M := (λ i j, 
    if i = 0 
    then (if j = 0 then 20/9 else 5/9) 
    else (if j = 0 then 7/9 else 22/9))
  use M
  sorry

end find_matrix_N_l491_491402


namespace arithmetic_sequence_sum_l491_491647

theorem arithmetic_sequence_sum (x y : ℤ) (h1 : ∃ d, d = 9 - 3) 
  (h2 : x = 9 + (classical.some h1)) 
  (h3 : y = x + (classical.some h1)) 
  (h_seq : x = 15 ∧ y = 21) : 
  x + y = 36 := 
  sorry

end arithmetic_sequence_sum_l491_491647


namespace poll_total_l491_491319

-- Define the conditions
variables (men women : ℕ)
variables (pct_favor : ℝ := 35) (women_opposed : ℕ := 39)
noncomputable def total_people (men women : ℕ) : ℕ := men + women

-- We need to prove the total number of people polled, given the conditions
theorem poll_total (h1 : men = women)
  (h2 : (pct_favor / 100) * women + (39 : ℝ) / (65 / 100) = (women: ℝ)) :
  total_people men women = 120 :=
sorry

end poll_total_l491_491319


namespace allocation_proof_l491_491389

def student : Type := {A, B, C, D, E}
def village : Type := {V1, V2, V3}

noncomputable def valid_allocations_count : ℕ := 
  let entities := {AB, C, D, E}
  let ways_to_group := nat.choose 4 2
  let ways_to_assign := nat.factorial 3
  ways_to_group * ways_to_assign

theorem allocation_proof : valid_allocations_count = 36 := 
  sorry

end allocation_proof_l491_491389


namespace sum_of_happy_numbers_l491_491785

theorem sum_of_happy_numbers :
  let a : ℕ → ℝ := λ n, if n = 1 then 1 else Real.log (n + 1) / Real.log n,
      is_happy (k : ℕ) := ∃ n : ℕ, 1 ≤ n ∧ k = 2^n - 1,
      sum_happy_in_range (l u : ℕ) : ℕ :=
        Finset.sum (Finset.filter is_happy (Finset.Icc l u)) id,
  sum_happy_in_range 1 2022 = 2036 :=
by sorry

end sum_of_happy_numbers_l491_491785


namespace inequality_holds_l491_491754

theorem inequality_holds (x : ℝ) (hx : 0 < x ∧ x < 4) :
  ∀ y : ℝ, y > 0 → (4 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y) / (x + y) > 3 * x^2 * y) :=
by
  intros y hy_gt_zero
  sorry

end inequality_holds_l491_491754


namespace polynomial_division_quotient_l491_491408

theorem polynomial_division_quotient :
  (∃ q r : ℤ[X], (r.degree < (Polynomial.degree (Polynomial.C 3 * Polynomial.X + Polynomial.C 1))) ∧
    (Polynomial.C 4 * Polynomial.X ^ 5 - Polynomial.C 5 * Polynomial.X ^ 4 + Polynomial.C 7 * Polynomial.X ^ 3 - Polynomial.C 15 * Polynomial.X ^ 2 + Polynomial.C 9 * Polynomial.X - Polynomial.C 3 
     = ((Polynomial.C (4/3) * Polynomial.X ^ 4 - Polynomial.C (17/9) * Polynomial.X ^ 3 + Polynomial.C (56/27) * Polynomial.X ^ 2 - Polynomial.C (167/81) * Polynomial.X 
     + Polynomial.C (500/243)) * (Polynomial.C 3 * Polynomial.X + Polynomial.C 1)) + r)) := sorry

end polynomial_division_quotient_l491_491408


namespace jerry_added_action_figures_l491_491874

theorem jerry_added_action_figures :
  ∃ (x : ℕ), 2 + x + 4 = 10 :=
begin
  unfold,
  sorry
end

end jerry_added_action_figures_l491_491874


namespace circumscribed_circle_tangent_AB_l491_491783

-- Define points and segments in the Cartesian plane
variables {A B C D X Y : Point ℝ}
variables (AB CD AD BC : line ℝ)

-- Define conditions
axiom condition1 : right_angle ∠ A
axiom condition2 : parallel BC AD
axiom condition3 : length BC = 1
axiom condition4 : length AD = 4
axiom condition5 : ∃ (A B : Point ℝ), X ∈ line(A, B)
axiom condition6 : ∃ (C D : Point ℝ), Y ∈ line(C, D)
axiom condition7 : distance X Y = 2
axiom condition8 : perpendicular XY CD

-- Define the problem
theorem circumscribed_circle_tangent_AB :
  circumscribed_circle_triangle X C D ⊥ line(A, B) :=
sorry

end circumscribed_circle_tangent_AB_l491_491783


namespace Amanda_tickets_third_day_l491_491690

theorem Amanda_tickets_third_day :
  (let total_tickets := 80
   let first_day_tickets := 5 * 4
   let second_day_tickets := 32

   total_tickets - (first_day_tickets + second_day_tickets) = 28) :=
by
  sorry

end Amanda_tickets_third_day_l491_491690


namespace problem_statement_l491_491891

-- Define k as the integral of (sin x - cos x) from 0 to pi
noncomputable def k := ∫ x in 0..π, (sin x - cos x)

-- Define a function representing the polynomial expansion
def poly (k : ℝ) (x : ℝ) : ℝ := (1 - k * x) ^ 8

-- State the theorem to be proven
theorem problem_statement : let k := ∫ x in 0..π, (sin x - cos x)
                           in k = 2 → 
                              (∃ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ), 
                                poly k x = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 
                                ∧ (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8) = 0)
                          := sorry

end problem_statement_l491_491891


namespace compare_neg_fractions_l491_491361

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l491_491361


namespace hyperbola_transformation_and_properties_l491_491640

theorem hyperbola_transformation_and_properties :
  let O1 := (1, 2)
  let origin_translation (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - 1, p.2 - 2)
  let hyperbola_eq (x y : ℝ) : Prop := y = (2 * x - 3) / (x - 1)
  let new_hyperbola_eq (X Y : ℝ) : Prop := Y = -1 / X
  let vertex_in_new_coordinates (X Y : ℝ) : Prop := 
    (new_hyperbola_eq X Y) ∧ 
    ((X = -1 ∧ Y = 1) ∨ (X = 1 ∧ Y = -1))
  let vertex_in_old_coordinates (x y : ℝ) : Prop :=
    (hyperbola_eq x y) ∧ 
    ((x = 0 ∧ y = 3) ∨ (x = 2 ∧ y = 1))
  let real_axis_new (X Y : ℝ) : Prop := Y = -X
  let real_axis_old (x y : ℝ) : Prop := x + y = 3
  let imaginary_axis_new (X Y : ℝ) : Prop := Y = X
  let imaginary_axis_old (x y : ℝ) : Prop := x = y - 1
in
  (∀ X Y, new_hyperbola_eq X Y ↔ hyperbola_eq (X + O1.1) (Y + O1.2)) ∧
  (vertex_in_new_coordinates -1 1 ∧ vertex_in_new_coordinates 1 -1) ∧
  (vertex_in_old_coordinates 0 3 ∧ vertex_in_old_coordinates 2 1) ∧
  (real_axis_new X Y ↔ real_axis_old (X + O1.1) (Y + O1.2)) ∧
  (imaginary_axis_new X Y ↔ imaginary_axis_old (X + O1.1) (Y + O1.2)) :=
sorry

end hyperbola_transformation_and_properties_l491_491640


namespace find_y_solution_l491_491843

theorem find_y_solution :
  ∃ y : ℝ, y = sqrt (3 + y) ∧ y = (1 + sqrt 13) / 2 :=
by
  sorry

end find_y_solution_l491_491843


namespace decimal_fraction_denominator_l491_491188

theorem decimal_fraction_denominator : ∃ (d : ℕ), (0.666...) = (n : ℕ) / d ∧ nat.coprime n d :=
by
  sorry

end decimal_fraction_denominator_l491_491188


namespace point_on_xoz_plane_l491_491093

def Point := ℝ × ℝ × ℝ

def lies_on_plane_xoz (p : Point) : Prop :=
  p.2 = 0

theorem point_on_xoz_plane :
  lies_on_plane_xoz (-2, 0, 3) :=
by
  sorry

end point_on_xoz_plane_l491_491093


namespace problem1_problem2_l491_491471

-- Problem 1 Conditions
def Z (m : ℝ) : ℂ := (5 * m^2) / (1 - (2:ℂ)*Complex.i) - (1 + 5 * Complex.i) * m - 3 * (2 + Complex.i)

-- Problem 1 Assertion
theorem problem1 (m : ℝ) : Complex.im (Z m) ≠ 0 → m = -2 :=
sorry

-- Problem 2 Conditions
def is_ineq (m : ℝ) : Prop := m^2 - (m^2 - 3 * m) * Complex.i < (m^2 - 4 * m + 3) * Complex.i + 10

-- Problem 2 Assertion
theorem problem2 (m : ℝ) (h : is_ineq m) : m = 3 :=
sorry

end problem1_problem2_l491_491471


namespace maxim_birth_probability_l491_491156

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end maxim_birth_probability_l491_491156


namespace determine_length_x_l491_491372

theorem determine_length_x 
  (x : ℝ) 
  (h_total_area : 9 * x^2 + 16 * x^2 + 6 * x^2 = 1000) : 
  x = 10 * real.sqrt(31) / 31 :=
by
  sorry

end determine_length_x_l491_491372


namespace slope_angle_of_tangent_at_one_l491_491037

def f (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x + 2 / x

theorem slope_angle_of_tangent_at_one (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = - f a x) : 
  ∠ 𝚤 (λ y, -y) = 3 * Real.pi / 4 :=
by
  -- Definition of an odd function implies a = 0
  have ha : a = 0 := sorry
  -- Define the function with a = 0
  let f' := λ x : ℝ, x + 2 / x
  -- Calculate the derivative of f'
  let df' := λ x : ℝ, deriv f' x
  -- Slope of the tangent line at x = 1
  have slope_at_1 : df' 1 = -1 := sorry
  -- Determine the slope angle
  exact sorry

end slope_angle_of_tangent_at_one_l491_491037


namespace find_a_value_l491_491475

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end find_a_value_l491_491475


namespace units_digit_a_2017_l491_491825

noncomputable def a_n (n : ℕ) : ℝ :=
  (Real.sqrt 2 + 1) ^ n - (Real.sqrt 2 - 1) ^ n

theorem units_digit_a_2017 : (Nat.floor (a_n 2017)) % 10 = 2 :=
  sorry

end units_digit_a_2017_l491_491825


namespace even_function_increasing_l491_491038

noncomputable def example_function (x m : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

theorem even_function_increasing (m : ℝ) (h : ∀ x : ℝ, example_function x m = example_function (-x) m) :
  (∀ x y : ℝ, x < 0 ∧ y < 0 ∧ x < y → example_function x m < example_function y m) :=
by 
  sorry

end even_function_increasing_l491_491038


namespace product_of_extrema_eq_one_l491_491893

theorem product_of_extrema_eq_one (x y : ℝ) (h : 5 * x^2 + 10 * x * y + 4 * y^2 = 1) :
  let p := (5 - 3 * Real.sqrt 5) / 4,
      q := (5 + 3 * Real.sqrt 5) / 4 in
  p * q = 1 :=
by
  sorry

end product_of_extrema_eq_one_l491_491893


namespace karen_cases_pickup_l491_491068

theorem karen_cases_pickup (total_boxes cases_per_box: ℕ) (h1 : total_boxes = 36) (h2 : cases_per_box = 12):
  total_boxes / cases_per_box = 3 :=
by
  -- We insert a placeholder to skip the proof here
  sorry

end karen_cases_pickup_l491_491068


namespace express_y_in_terms_of_x_l491_491065

variable (p : ℝ)

def x : ℝ := 1 + 3^p
def y : ℝ := 1 + 3^(-p)

theorem express_y_in_terms_of_x (p : ℝ) : y p = (x p) / ((x p) - 1) :=
by
  sorry

end express_y_in_terms_of_x_l491_491065


namespace cyclic_quadrilateral_dissect_l491_491005

theorem cyclic_quadrilateral_dissect (n : ℕ) (h : n > 4) : 
  ∀ (Q : Quadrilateral), isCyclic Q → 
  ∃ (quadrilaterals : List Quadrilateral), 
     (∀ q ∈ quadrilaterals, isCyclic q) ∧ quadrilaterals.length = n := 
sorry

end cyclic_quadrilateral_dissect_l491_491005


namespace greatest_integer_l491_491568

-- Definitions derived from the problem's conditions
structure Trapezoid where
  base1 base2 : ℝ
  base_diff : base2 - base1 = 100

def mid_segment_length (t : Trapezoid) : ℝ :=
  (t.base1 + t.base2) / 2

def ratio_areas (t : Trapezoid) : ℝ := 2 / 3

def equal_area_segment (t : Trapezoid) : ℝ :=
  let AB := t.base1
  let DC := t.base2
  let h := 1 -- arbitrary height since only ratios matter
  let y := (ratio_areas t) * h -- simplified height ratio
  AB + y

def x_segment (t : Trapezoid) : ℝ :=
  equal_area_segment t

-- Main statement to be proven
theorem greatest_integer (t : Trapezoid) : 
  let x := x_segment t
  let target := x^2 / 100
  int.floor target = 181 := by
  sorry

end greatest_integer_l491_491568


namespace existsValidGrid_l491_491509

-- Define the type of our grid as a 3x3 matrix of integers
def Grid := Matrix (Fin 3) (Fin 3) ℕ

-- Define the grid with numbers from the solution
def exampleGrid : Grid :=
  ![
    ![1, 7, 2],
    ![6, 5, 9],
    ![3, 8, 4]
  ]

-- A condition that checks if given grid satisfies the problem conditions
def isValidGrid (grid : Grid) : Prop :=
  (∀ i : Fin 3, (∑ j, grid i j) % 5 = 0) ∧ -- each row sum divisible by 5
  (∀ j : Fin 3, (∑ i, grid i j) % 5 = 0) ∧ -- each column sum divisible by 5
  ((grid 0 0 + grid 1 1 + grid 2 2) % 5 = 0) ∧ -- main diagonal sum divisible by 5
  ((grid 0 2 + grid 1 1 + grid 2 0) % 5 = 0) -- secondary diagonal sum divisible by 5

theorem existsValidGrid : ∃ grid : Grid, isValidGrid grid :=
  ⟨exampleGrid, by
    unfold isValidGrid
    -- Prove the grid satisfies the conditions
    dsimp only [exampleGrid]
    split,
    { intro i,
      fin_cases i <;>
      simp },
    split,
    { intro j,
      fin_cases j <;>
      simp },
    split;
    simp⟩

end existsValidGrid_l491_491509


namespace convex_polygon_cyclic_iff_assignable_pairs_l491_491541

variable {n : ℕ}
variable (A : Fin n → Point)
variable (b c : Fin n → ℝ)

noncomputable def cyclic (A : Fin n → Point) : Prop := 
  ∃ O : Point, ∀ i, is_on_circle O (A i)

theorem convex_polygon_cyclic_iff_assignable_pairs 
  (n : ℕ) (A : Fin n → Point) (h_convex : convex_polygon A) :
  (cyclic A) ↔ (∃ (b c : Fin n → ℝ),
    ∀ i j : Fin n, 1 ≤ i.val → i ≤ j → 
    dist (A i) (A j) = b j * c i - b i * c j) :=
sorry

end convex_polygon_cyclic_iff_assignable_pairs_l491_491541


namespace range_of_k_for_distinct_roots_l491_491076
-- Import necessary libraries

-- Define the quadratic equation and conditions
noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the property of having distinct real roots
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c > 0

-- Define the specific problem instance and range condition
theorem range_of_k_for_distinct_roots (k : ℝ) :
  has_two_distinct_real_roots 1 2 k ↔ k < 1 :=
by
  sorry

end range_of_k_for_distinct_roots_l491_491076


namespace sqrt_of_0_09_l491_491957

theorem sqrt_of_0_09 : Real.sqrt 0.09 = 0.3 :=
by
  -- Mathematical problem restates that the square root of 0.09 equals 0.3
  sorry

end sqrt_of_0_09_l491_491957


namespace three_quantities_change_l491_491581

-- Declare the points and the triangle condition
variables {P A B M N : Type}
variables [midpoint_condition : Midpoint P A M]
variables [midpoint_condition' : Midpoint P B N]
variables [constant_PA : LengthConstant P A]

-- We need to assert that as P moves, the length of PA remains constant, and then conclude about the quantities.
theorem three_quantities_change
  (h_midpoints : midpoint_condition)
  (h_midpoints' : midpoint_condition')
  (h_constant_PA : constant_PA) :
  ∃ (x : ℕ), x = 3 :=
by
  sorry

end three_quantities_change_l491_491581


namespace damage_in_dollars_l491_491299

noncomputable def euros_to_dollars (euros : ℝ) : ℝ := euros * (1 / 0.9)

theorem damage_in_dollars :
  euros_to_dollars 45000000 = 49995000 :=
by
  -- This is where the proof would go
  sorry

end damage_in_dollars_l491_491299


namespace sequence_a_n_l491_491431

theorem sequence_a_n (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, S n = 3 + 2^n) →
  (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) ↔ 
  (∀ n : ℕ, a n = if n = 1 then 5 else 2^(n-1)) :=
by
  sorry

end sequence_a_n_l491_491431


namespace value_of_A_l491_491552

theorem value_of_A (h p a c k e : ℤ) 
  (H : h = 8)
  (PACK : p + a + c + k = 50)
  (PECK : p + e + c + k = 54)
  (CAKE : c + a + k + e = 40) : 
  a = 25 :=
by 
  sorry

end value_of_A_l491_491552


namespace compare_rat_neg_l491_491347

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l491_491347


namespace compare_rat_neg_l491_491349

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l491_491349


namespace power_of_2_multiplication_l491_491282

theorem power_of_2_multiplication : (16^3) * (4^4) * (32^2) = 2^30 := by
  sorry

end power_of_2_multiplication_l491_491282


namespace trig_identity_l491_491002

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + Real.sin α) = 3 / 4 :=
by
  sorry

end trig_identity_l491_491002


namespace median_first_fifteen_positive_integers_l491_491977

theorem median_first_fifteen_positive_integers : 
  (List.median [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] = 8) :=
by
  sorry

end median_first_fifteen_positive_integers_l491_491977


namespace area_enclosed_abs_val_relation_l491_491976

theorem area_enclosed_abs_val_relation : 
  (∑ in {x : ℝ // |x| + |3 * y| ≤ 12}.to_finset, 1) = 96 :=
sorry

end area_enclosed_abs_val_relation_l491_491976


namespace profit_percentage_approx_30_l491_491697

def wheat1_weight : ℝ := 30
def wheat1_price_per_kg : ℝ := 11.50
def wheat2_weight : ℝ := 20
def wheat2_price_per_kg : ℝ := 14.25
def selling_price_per_kg : ℝ := 16.38

def total_cost : ℝ := (wheat1_weight * wheat1_price_per_kg) + (wheat2_weight * wheat2_price_per_kg)
def total_weight : ℝ := wheat1_weight + wheat2_weight
def cost_price_per_kg : ℝ := total_cost / total_weight
def total_selling_price : ℝ := total_weight * selling_price_per_kg
def profit : ℝ := total_selling_price - total_cost
def percentage_profit : ℝ := (profit / total_cost) * 100

theorem profit_percentage_approx_30 :
  percentage_profit ≈ 30 :=
sorry

end profit_percentage_approx_30_l491_491697


namespace rectangle_area_formula_l491_491207

-- Define the given conditions: perimeter is 20, one side length is x
def rectangle_perimeter (P x : ℝ) (w : ℝ) : Prop := P = 2 * (x + w)
def rectangle_area (x w : ℝ) : ℝ := x * w

-- The theorem to prove
theorem rectangle_area_formula (x : ℝ) (h_perimeter : rectangle_perimeter 20 x (10 - x)) : 
  rectangle_area x (10 - x) = x * (10 - x) := 
by 
  sorry

end rectangle_area_formula_l491_491207


namespace value_of_a_l491_491483

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end value_of_a_l491_491483


namespace days_for_x_to_finish_work_alone_l491_491657

variable (d x y : ℕ)

theorem days_for_x_to_finish_work_alone :
  (∀ y_rate work_remaining x_rate : ℚ,
    y_rate = 1/15 →
    work_remaining = 1 - 5 * y_rate →
    work_remaining = 2/3 →
    x_rate = work_remaining / 12 →
    (x_rate = 1/18) →
    (d * x_rate = 1) →
    d = 18) :=
begin
  intros,
  sorry
end

end days_for_x_to_finish_work_alone_l491_491657


namespace solve_trig_eq_l491_491928

theorem solve_trig_eq (n : ℤ) (x : ℝ) : 
  (2 * n * π < x ∧ x < (2 * n + 1) * π) ∧
  (sin x) ^ (arctan (sin x + cos x)) = (csc x) ^ (arctan (sin 2 * x) + π / 4) ↔
   x = 2 * n * π + π / 2 ∨ x = 2 * n * π + 3 * π / 4 :=
by sorry

end solve_trig_eq_l491_491928


namespace secret_society_friends_l491_491225

/-- A secret society with 2011 members where each member can occasionally give one dollar to each of their friends,
    such that members can redistribute their money arbitrarily, forms a connected, cycle-free graph 
    (i.e., a tree) with exactly 2010 edges. -/
theorem secret_society_friends {V : Type} [Fintype V] [DecidableEq V] 
  (members : Finset V) (edges : Finset (V × V))
  (h_members : members.card = 2011)
  (h_edges : ∀ v : V, v ∈ members → ∃ (u : V), (v, u) ∈ edges ∨ (u, v) ∈ edges)
  (h_connected : ∀ (balance : V → ℤ), 
    ∃ (transfer : V → ℤ), 
    ∀ v : V, v ∈ members → balance v + ∑ (u : V) in members.filter (λ u, (v, u) ∈ edges ∨ (u, v) ∈ edges), transfer u = 0) :
  edges.card = 2010 :=
sorry

end secret_society_friends_l491_491225


namespace axis_of_symmetry_of_g_l491_491611

noncomputable def f (x : ℝ) : ℝ := sin (2 * x) - sqrt 3 * cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * (x - π / 12) - π / 3)

theorem axis_of_symmetry_of_g : ∃ k ∈ ℤ, x = k * (π / 2) ∧ k = 1 :=
by
  -- Statement only, no proof provided.
  sorry

end axis_of_symmetry_of_g_l491_491611


namespace height_of_pyramid_l491_491668

-- Define the volumes
def volume_cube (s : ℕ) : ℕ := s^3
def volume_pyramid (b : ℕ) (h : ℕ) : ℕ := (b^2 * h) / 3

-- Given constants
def s := 6
def b := 12

-- Given volume equality
def volumes_equal (s : ℕ) (b : ℕ) (h : ℕ) : Prop :=
  volume_cube s = volume_pyramid b h

-- The statement to prove
theorem height_of_pyramid (h : ℕ) (h_eq : volumes_equal s b h) :
  h = 9 := sorry

end height_of_pyramid_l491_491668


namespace smallest_three_digit_multiple_of_13_l491_491980

theorem smallest_three_digit_multiple_of_13 : ∃ (n : ℕ), n ≥ 100 ∧ n < 1000 ∧ 13 ∣ n ∧ (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ 13 ∣ m → n ≤ m) ∧ n = 104 :=
by
  sorry

end smallest_three_digit_multiple_of_13_l491_491980


namespace find_a_b_largest_x_l491_491457

def polynomial (a b x : ℤ) : ℤ := 2 * (a * x - 3) - 3 * (b * x + 5)

-- Given conditions
variables (a b : ℤ)
#check polynomial

-- Part 1: Prove the values of a and b
theorem find_a_b (h1 : polynomial a b 2 = -31) (h2 : a + b = 0) : a = -1 ∧ b = 1 :=
by sorry

-- Part 2: Given a and b found in Part 1, find the largest integer x such that P > 0
noncomputable def P (x : ℤ) : ℤ := -5 * x - 21

theorem largest_x {a b : ℤ} (ha : a = -1) (hb : b = 1) : ∃ x : ℤ, P x > 0 ∧ ∀ y : ℤ, (P y > 0 → y ≤ x) :=
by sorry

end find_a_b_largest_x_l491_491457


namespace rain_probability_l491_491622

-- Define the probability of rain on any given day, number of trials, and specific number of successful outcomes.
def prob_rain_each_day : ℚ := 1/5
def num_days : ℕ := 10
def num_rainy_days : ℕ := 3

-- Define the binomial probability mass function
def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

-- Statement to prove
theorem rain_probability : binomial_prob num_days num_rainy_days prob_rain_each_day = 1966080 / 9765625 :=
by
  sorry

end rain_probability_l491_491622


namespace maxim_birth_probability_l491_491160

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end maxim_birth_probability_l491_491160


namespace restore_acute_triangle_l491_491919

theorem restore_acute_triangle (A B C D H O : Point) (h_acute : is_acute_triangle A B C)
  (h_orthocenter : orthocenter A B C = H)
  (h_foot : foot B A C = D)
  (h_circumcenter_BHC : circumcenter B H C = O) :
  ∃ B' C', is_triangle A B' C' ∧ is_acute_triangle A B' C' ∧ foot B' A C' = D ∧ circumcenter B' H C' = O := 
sorry

end restore_acute_triangle_l491_491919


namespace valid_numbers_count_l491_491464

noncomputable def count_valid_numbers : ℕ :=
  let a := 7
  let possible_d := {0, 2, 4, 6, 8}
  let pair_bc := [(4,5), (4,6), (4,7), (5,6), (5,7), (6,7)]
  in 1 * possible_d.card * pair_bc.length

theorem valid_numbers_count : count_valid_numbers = 30 := by
  sorry

end valid_numbers_count_l491_491464


namespace largest_incircle_radius_l491_491374

theorem largest_incircle_radius (AB BC CD DA: ℝ) (hAB: AB = 14) (hBC: BC = 9) (hCD: CD = 7) (hDA: DA = 12) : 
  let s := (AB + BC + CD + DA) / 2 in
  let A := Real.sqrt ((s - AB) * (s - BC) * (s - CD) * (s - DA)) in
  let r := A / s in
  r = 2 * Real.sqrt 6 :=
by
  have s := (14 + 9 + 7 + 12) / 2
  have A := Real.sqrt ((s - 14) * (s - 9) * (s - 7) * (s - 12))
  have r := A / s
  have : s = 21 := by linarith
  have : A = Real.sqrt (10584) := by simp [*]
  have : r = 2 * Real.sqrt 6 := by simp [*]
  exact this

end largest_incircle_radius_l491_491374


namespace car_speed_l491_491848

/-- 
If a tire rotates at 400 revolutions per minute, and the circumference of the tire is 6 meters, 
the speed of the car is 144 km/h.
-/
theorem car_speed (rev_per_min : ℕ) (circumference : ℝ) (speed : ℝ) :
  rev_per_min = 400 → circumference = 6 → speed = 144 :=
by
  intro h_rev h_circ
  sorry

end car_speed_l491_491848


namespace part1_monotonically_increasing_intervals_part2_cos_2x₀_l491_491452

-- Part (1)
def f (x : ℝ) : ℝ := sin x * cos x - sqrt 3 * sin x ^ 2 + sqrt 3 / 2

theorem part1_monotonically_increasing_intervals (k : ℤ) :
  - 5 * π / 12 + k * π ≤ x ∧ x ≤ π / 12 + k * π → monotone f :=
sorry

-- Part (2)
def x₀ (h : f x₀ = 3 / 5) (h₀ : x₀ ∈ Icc (π / 6) (π / 3)) : ℝ := sorry

theorem part2_cos_2x₀ (x₀ : ℝ) (hx₀ : f x₀ = 3 / 5) (hx₀_range : x₀ ∈ Icc (π / 6) (π / 3)) :
  cos (2 * x₀) = (3 * sqrt 3 - 4) / 10 :=
sorry

end part1_monotonically_increasing_intervals_part2_cos_2x₀_l491_491452


namespace remainder_2_pow_2015_mod_20_l491_491978

/-- 
  Given that powers of 2 modulo 20 follow a repeating cycle every 4 terms:
  2, 4, 8, 16, 12
  
  Prove that the remainder when \(2^{2015}\) is divided by 20 is 8.
-/
theorem remainder_2_pow_2015_mod_20 : (2 ^ 2015) % 20 = 8 :=
by
  -- The proof is to be filled in.
  sorry

end remainder_2_pow_2015_mod_20_l491_491978


namespace Maxim_born_in_2008_probability_l491_491152

def birth_interval_start := ⟨2007, 9, 2⟩ -- September 2, 2007
def birth_interval_end := ⟨2008, 8, 31⟩ -- August 31, 2008
def days_in_interval := 365
def days_in_2008 := 244

theorem Maxim_born_in_2008_probability :
  (days_in_2008 : ℝ) / (days_in_interval : ℝ) = 244 / 365 :=
sorry

end Maxim_born_in_2008_probability_l491_491152


namespace Carnots_Theorem_l491_491585

theorem Carnots_Theorem (A B C A1 B1 C1: Point) 
    (H1: is_triangle (A, B, C))
    (H2: drop_perpendicular A1 B C)
    (H3: drop_perpendicular B1 C A)
    (H4: drop_perpendicular C1 A B) :
  (∃ M, 
    M ∈ (intersect_perpendiculars (A1, B, C)) ∧ 
    M ∈ (intersect_perpendiculars (B1, C, A)) ∧ 
    M ∈ (intersect_perpendiculars (C1, A, B))) ↔ 
  (dist_squared A1 B + dist_squared C1 A + dist_squared B1 C = 
    dist_squared B1 A + dist_squared A1 C + dist_squared C1 B) := 
sorry

end Carnots_Theorem_l491_491585


namespace order_of_a_b_c_l491_491799

noncomputable def a : ℝ := 4^(2/3)
noncomputable def b : ℝ := 3^(2/3)
noncomputable def c : ℝ := 25^(1/3)

theorem order_of_a_b_c : b < a ∧ a < c := by
  have ha : a = 4^(2/3) := rfl
  have hb : b = 3^(2/3) := rfl
  have hc : c = 25^(1/3) := rfl
  -- We simplify the cubed expressions
  have ha_cubed : a^3 = 16 := by
    calc
      a^3 = (4^(2/3))^3 : by rw ha
      ... = 4^2 : by norm_num
      ... = 16 : by norm_num
  have hb_cubed : b^3 = 9 := by
    calc
      b^3 = (3^(2/3))^3 : by rw hb
      ... = 3^2 : by norm_num
      ... = 9 : by norm_num
  have hc_cubed : c^3 = 25 := by
    calc
      c^3 = (25^(1/3))^3 : by rw hc
      ... = 25 : by norm_num
  -- Compare their cubed values
  have hc_gt_ha : c^3 > a^3 := by
    calc
      25 > 16 : by norm_num
  have ha_gt_hb : a^3 > b^3 := by
    calc
      16 > 9 : by norm_num
  -- Since a, b, c are positive, we conclude by taking cube roots (preserves inequality)
  exact ⟨by exact real.rpow_lt_rpow_of_exponent_lt zero_lt_three (real.rpow_pos_of_pos (by norm_num) (2/3)) ha_gt_hb, by exact real.rpow_lt_rpow_of_exponent_lt zero_lt_three (real.rpow_pos_of_pos (by norm_num) (2/3)) hc_gt_ha⟩

end order_of_a_b_c_l491_491799


namespace range_of_m_inequality_a_b_l491_491821

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 2|

theorem range_of_m (m : ℝ) : (∀ x, f x ≥ |m - 1|) → -2 ≤ m ∧ m ≤ 4 :=
sorry

theorem inequality_a_b (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a^2 + b^2 = 2) : 
  a + b ≥ 2 * a * b :=
sorry

end range_of_m_inequality_a_b_l491_491821


namespace kindergarten_children_l491_491084

theorem kindergarten_children (x y z n : ℕ) 
  (h1 : 2 * x + 3 * y + 4 * z = n)
  (h2 : x + y + z = 26)
  : n = 24 := 
sorry

end kindergarten_children_l491_491084


namespace value_of_f_one_range_of_a_l491_491448

def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2*x else -x^2

theorem value_of_f_one : f 1 = -1 := 
  by {
    have h : 1 ≥ 0 := le_of_lt (by linarith),
    rw if_neg (not_lt_of_ge h),
    norm_num,
  }

theorem range_of_a (a : ℝ) : f a ≤ 3 → a ∈ Set.Ici (-3) :=
  by {
    intro h,
    cases lt_or_ge a 0 with h_pos h_nonpos,
    { -- Case when a < 0
      have : a^2 + 2*a ≤ 3 := by linarith,
      sorry -- Add proof steps for quadratic inequality here
    },
    { -- Case when a ≥ 0
      have : -a^2 ≤ 3 := by linarith,
      sorry -- Add proof steps for solving inequality here
    }
  }

end value_of_f_one_range_of_a_l491_491448


namespace find_k_l491_491400

theorem find_k : ∃ k : ℕ, ∀ n : ℕ, n > 0 → (2^n + 11) % (2^k - 1) = 0 ↔ k = 4 :=
by
  sorry

end find_k_l491_491400


namespace edge_length_approx_17_1_l491_491664

-- Define the base dimensions of the rectangular vessel
def length_base : ℝ := 20
def width_base : ℝ := 15

-- Define the rise in water level
def rise_water_level : ℝ := 16.376666666666665

-- Calculate the area of the base
def area_base : ℝ := length_base * width_base

-- Calculate the volume of the cube (which is equal to the volume of water displaced)
def volume_cube : ℝ := area_base * rise_water_level

-- Calculate the edge length of the cube
def edge_length_cube : ℝ := volume_cube^(1/3)

-- Statement: The edge length of the cube is approximately 17.1 cm
theorem edge_length_approx_17_1 : abs (edge_length_cube - 17.1) < 0.1 :=
by sorry

end edge_length_approx_17_1_l491_491664


namespace eq_op_op_op_92_l491_491712

noncomputable def opN (N : ℝ) : ℝ := 0.75 * N + 2

theorem eq_op_op_op_92 : opN (opN (opN 92)) = 43.4375 :=
by
  sorry

end eq_op_op_op_92_l491_491712


namespace horizontal_length_tv_screen_l491_491139

theorem horizontal_length_tv_screen : 
  ∀ (a b : ℝ), (a / b = 4 / 3) → (a ^ 2 + b ^ 2 = 27 ^ 2) → a = 21.5 := 
by 
  sorry

end horizontal_length_tv_screen_l491_491139


namespace volume_ratio_of_cube_cut_l491_491602

/-
  The cube ABCDEFGH has its side length assumed to be 1.
  The points K, L, M divide the vertical edges AA', BB', CC'
  respectively, in the ratios 1:2, 1:3, 1:4. 
  We need to prove that the plane KLM cuts the cube into
  two parts such that the volume ratio of the two parts is 4:11.
-/
theorem volume_ratio_of_cube_cut (s : ℝ) (K L M : ℝ) :
  ∃ (Vbelow Vabove : ℝ), 
    s = 1 → 
    K = 1/3 → 
    L = 1/4 → 
    M = 1/5 → 
    Vbelow / Vabove = 4 / 11 :=
sorry

end volume_ratio_of_cube_cut_l491_491602


namespace event_probability_eq_one_eighth_l491_491674

noncomputable def event_occurrence_probability : ℝ :=
  let interval_length := (4 : ℝ) - (-4 : ℝ) in
  let event_length := (4 : ℝ) - (3 : ℝ) + (-(4 : ℝ)) - (-(5 : ℝ)) in
  event_length / interval_length

theorem event_probability_eq_one_eighth :
  event_occurrence_probability = (1 / 8 : ℝ) := by
  sorry

end event_probability_eq_one_eighth_l491_491674


namespace Josephine_sold_10_liters_l491_491163

def milk_sold (n1 n2 n3 : ℕ) (v1 v2 v3 : ℝ) : ℝ :=
  (v1 * n1) + (v2 * n2) + (v3 * n3)

theorem Josephine_sold_10_liters :
  milk_sold 3 2 5 2 0.75 0.5 = 10 :=
by
  sorry

end Josephine_sold_10_liters_l491_491163


namespace log_two_bounds_l491_491487

theorem log_two_bounds (h1 : 10^3 = 1000) 
                       (h2 : 10^4 = 10000) 
                       (h3 : 2^10 = 1024) 
                       (h6 : 2^13 = 8192) :
  3 / 10 < log 10 2 ∧ log 10 2 < 4 / 13 :=
by
  sorry

end log_two_bounds_l491_491487


namespace cost_of_one_basketball_deck_l491_491558

theorem cost_of_one_basketball_deck (total_money_spent : ℕ) 
  (mary_sunglasses_cost : ℕ) (mary_jeans_cost : ℕ) 
  (rose_shoes_cost : ℕ) (rose_decks_count : ℕ) 
  (mary_total_cost : total_money_spent = 2 * mary_sunglasses_cost + mary_jeans_cost)
  (rose_total_cost : total_money_spent = rose_shoes_cost + 2 * (total_money_spent - rose_shoes_cost) / rose_decks_count) :
  (total_money_spent - rose_shoes_cost) / rose_decks_count = 25 := 
by 
  sorry

end cost_of_one_basketball_deck_l491_491558


namespace complement_intersection_l491_491550

-- Defining the universal set U and subsets A and B
def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {2, 3, 4}
def B : Finset ℕ := {3, 4, 5}

-- Proving the complement of the intersection of A and B in U
theorem complement_intersection : (U \ (A ∩ B)) = {1, 2, 5} :=
by sorry

end complement_intersection_l491_491550


namespace orthocenters_collinear_l491_491020

-- Assume we are working in a Euclidean plane
variables {P : Type*} [EuclideanGeometry P]

-- Consider the points and triangle as given
variables (A B C K M L N : P)
variables (hKMB : K ∈ OpenSegment A B M) (hLNC : L ∈ OpenSegment A C N)
variables (ratio_eq : (dist B K) / (dist K M) = (dist C L) / (dist L N))

-- Define a function to get the orthocenter
def orthocenter (A B C : P) : P := sorry

-- Define the orthocenters of the specified triangles
def H1 := orthocenter A B C
def H2 := orthocenter A K L
def H3 := orthocenter A M N

-- The theorem stating the collinearity of the orthocenters
theorem orthocenters_collinear (hKMB : K ∈ OpenSegment A B M) 
(hLNC : L ∈ OpenSegment A C N) 
(ratio_eq : (dist B K) / (dist K M) = (dist C L) / (dist L N)) : 
Collinear P [H1 H2 H3] := sorry

end orthocenters_collinear_l491_491020


namespace problem_statement_l491_491445

noncomputable def f : ℝ → ℝ := 
λ x, if h : 0 ≤ x ∧ x < 2 then log 2 (x+1) else 
     if 0 ≤ x then 
       have hx : 0 ≤ x-2, from sub_nonneg_of_le (le_of_lt (lt_of_not_ge (by simp [h]))),
       log 2 ((x - 2)+1)
     else 
       log 2 ((-x - 2)+1) 

theorem problem_statement : 
  (∀ x, f (-x) = f x) →
  (∀ x, 0 ≤ x → f (x + 2) = f x) →
  (∀ x, 0 ≤ x ∧ x < 2 → f x = real.log (x + 1) / real.log 2) →
  f (-2009) + f (2010) = 1 :=
begin
  intros h_even h_periodic h_interval,
  sorry
end

end problem_statement_l491_491445


namespace simplest_square_root_l491_491648

-- Define the options
def option_A := sqrt (1 / 2)
def option_B := sqrt 0.8
def option_C := sqrt 9
def option_D := sqrt 5

-- State the theorem that option_D is the simplest square root
theorem simplest_square_root : option_A ≠ option_D ∧ option_B ≠ option_D ∧ option_C ≠ option_D ∧ (∀ x, x ≠ option_D → x ≠ sqrt 5) := 
by sorry

end simplest_square_root_l491_491648


namespace smallest_odd_number_with_four_prime_factors_l491_491979

theorem smallest_odd_number_with_four_prime_factors (n : ℕ) (hodd : n % 2 = 1) (hp : ∃ p1 p2 p3 p4, nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧ p1 * p2 * p3 * p4 = n ∧ (p1 > 11 ∨ p2 > 11 ∨ p3 > 11 ∨ p4 > 11)) : n = 1365 :=
by
  sorry

end smallest_odd_number_with_four_prime_factors_l491_491979


namespace negative_fraction_comparison_l491_491366

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l491_491366


namespace carousel_seating_l491_491168

theorem carousel_seating:
  exists (yellow blue red : ℕ), 
  (yellow + blue + red = 100) ∧ 
  (∃ (b7_r3_gap : ℕ) (y7_r23_gap : ℕ),
    b7_r3_gap = 99 / 2 ∧ y7_r23_gap = 99 / 2 ∧ -- since opposite seats in a circle with 100 seats
    b7_r3_gap + y7_r23_gap + 2 = 100) ∧ -- gap calculations
  (∃ b7_y7_red23_gap : ℕ, b7_y7_red23_gap = 19 ∧  23 - 3 - 1 = 19 ∧ -- between red 3 and red 23 
    (b7_y7_red23_gap = 6 + blue_seats ∧ blue_seats = 13)  ∧
    (yellow = 34 ∧ blue = 20 ∧ red = 46))
:=
begin 
  sorry 
end

end carousel_seating_l491_491168


namespace cyclic_quadrilaterals_similar_triangles_l491_491278

variable {α : Type} [EuclideanGeometry α] -- Assuming Euclidean Geometry

-- Definitions of points and orthocenter
variables {A B C A' B' C' H : α}

-- Conditions 
variable (triangle_ABC : Triangle A B C)
variable (A'_feet : AltitudeFoot A B C A')
variable (B'_feet : AltitudeFoot B A C B')
variable (C'_feet : AltitudeFoot C A B C')
variable (H_is_orthocenter : Orthocenter A B C H)

-- The main statements to prove
theorem cyclic_quadrilaterals :
  CyclicQuadrilateral A B A' B' ∧
  CyclicQuadrilateral B C B' C' ∧
  CyclicQuadrilateral C A C' A' ∧
  CyclicQuadrilateral A B' H C' ∧
  CyclicQuadrilateral B A' H C' ∧
  CyclicQuadrilateral C B' H A' :=
sorry

theorem similar_triangles :
  SimilarTriangle A B C A B' C' ∧
  SimilarTriangle A B C A' B C' ∧
  SimilarTriangle A B C A' B' C :=
sorry

end cyclic_quadrilaterals_similar_triangles_l491_491278


namespace locus_is_two_rays_l491_491944

noncomputable def locus_of_point (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, F1 = (-3, 0) ∧ F2 = (3, 0) ∧ abs (real.sqrt ((M.1 - F1.1)^2 + (M.2 - F1.2)^2) - real.sqrt ((M.1 - F2.1)^2 + (M.2 - F2.2)^2)) = 6

theorem locus_is_two_rays (M : ℝ × ℝ) :
  locus_of_point M (-3, 0) (3, 0) ↔ ( ∀ x y : ℝ, ∃ M ∈ (set.univ : set (ℝ × ℝ)), 
  (abs (real.sqrt ((M.1 - (-3,0).1)^2 + (M.2 - (-3,0).2)^2) - real.sqrt ((M.1 - (3,0).1)^2 + (M.2 - (3,0).2)^2)) = 6)) := 
by
  sorry

end locus_is_two_rays_l491_491944


namespace sum_fractional_parts_of_zeta_even_l491_491763

noncomputable def Riemann_zeta (x : ℝ) : ℝ :=
  ∑' (n : ℕ), (1 : ℝ) / (n^x)

def fractional_part (x : ℝ) : ℝ :=
  x - ⌊x⌋

theorem sum_fractional_parts_of_zeta_even :
  ∑' (k : ℕ) in finset.range (2), fractional_part (Riemann_zeta (2 * k)) = 1 / 6 :=
begin
  sorry -- Proof will be provided here
end

end sum_fractional_parts_of_zeta_even_l491_491763


namespace problem_statement_l491_491006

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : ∃ x1 x2 x3 : ℝ, (x1 < x2 ∧ x2 < x3) ∧ (x3 = b) ∧
    (|x1|^(1/2) + |x1 + a|^(1/2) = b) ∧
    (|x2|^(1/2) + |x2 + a|^(1/2) = b) ∧
    (|x3|^(1/2) + |x3 + a|^(1/2) = b ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)) :
  a + b = 144 := sorry

end problem_statement_l491_491006


namespace largest_sum_ABC_l491_491859

theorem largest_sum_ABC (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_prod : A * B * C = 2401) :
  A + B + C ≤ 351 :=
sorry

end largest_sum_ABC_l491_491859


namespace josephine_total_milk_l491_491162

-- Define the number of containers and the amount of milk they hold
def cnt_1 : ℕ := 3
def qty_1 : ℚ := 2

def cnt_2 : ℕ := 2
def qty_2 : ℚ := 0.75

def cnt_3 : ℕ := 5
def qty_3 : ℚ := 0.5

-- Define the total amount of milk sold
def total_milk_sold : ℚ := cnt_1 * qty_1 + cnt_2 * qty_2 + cnt_3 * qty_3

theorem josephine_total_milk : total_milk_sold = 10 := by
  -- This is the proof placeholder
  sorry

end josephine_total_milk_l491_491162


namespace total_reams_l491_491055

theorem total_reams (h_r : ℕ) (s_r : ℕ) : h_r = 2 → s_r = 3 → h_r + s_r = 5 :=
by
  intro h_eq s_eq
  sorry

end total_reams_l491_491055


namespace springtown_hardware_orders_in_october_l491_491880

def claw_hammer_pattern (n : ℕ) : ℕ :=
  match n with
  | 1 => 3
  | 2 => 4
  | 3 => 6
  | 4 => 9
  | _ => claw_hammer_pattern 4 + (n - 3)

def ball_peen_hammer_pattern (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | 2 => 3
  | 3 => 7
  | 4 => 11
  | _ => ball_peen_hammer_pattern 4 + 4

def total_hammers_in_october : ℕ := 
  let claw_hammers_october := claw_hammer_pattern 5
  let ball_peen_hammers_october := ball_peen_hammer_pattern 5
  let total_hammers := claw_hammers_october + ball_peen_hammers_october
  let seasonal_increase := total_hammers * 5 / 100
  total_hammers + Int.ceil seasonal_increase

theorem springtown_hardware_orders_in_october : total_hammers_in_october = 30 := 
  by sorry

end springtown_hardware_orders_in_october_l491_491880


namespace find_a_value_l491_491480

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end find_a_value_l491_491480


namespace problem_part1_problem_part2_l491_491546

open Real

-- Part (1)
theorem problem_part1 : ∀ x > 0, log x ≤ x - 1 := 
by 
  sorry -- proof goes here


-- Part (2)
theorem problem_part2 : (∀ x > 0, log x ≤ a * x + (a - 1) / x - 1) → 1 ≤ a := 
by 
  sorry -- proof goes here

end problem_part1_problem_part2_l491_491546


namespace solve_for_3x2_plus_6_l491_491846

theorem solve_for_3x2_plus_6 (x : ℚ) (h : 5 * x + 3 = 2 * x - 4) : 3 * (x^2 + 6) = 103 / 3 :=
by
  sorry

end solve_for_3x2_plus_6_l491_491846


namespace john_patients_per_year_l491_491110

theorem john_patients_per_year :
  let patients_per_day_1 := 20 in
  let patients_per_day_2 := patients_per_day_1 * 6 / 5 in -- 20% more means 120% of the original or 1.2 times
  let days_per_week := 5 in
  let weeks_per_year := 50 in
  let total_patients_per_week := (patients_per_day_1 * days_per_week) + (patients_per_day_2 * days_per_week) in
  total_patients_per_week * weeks_per_year = 11000 :=
by
  sorry

end john_patients_per_year_l491_491110


namespace area_of_path_correct_construction_cost_correct_l491_491271

def length_of_field : ℕ := 75
def width_of_field : ℕ := 55
def width_of_path : ℕ := 2.5.to_nat -- Converting 2.5 meters to an integer rounding
def cost_per_sq_m : ℕ := 7

-- Calculate the length and width of the field including the path
def total_length : ℕ := length_of_field + 2 * width_of_path
def total_width : ℕ := width_of_field + 2 * width_of_path

-- Calculate the area of the field including the path
def area_with_path : ℕ := total_length * total_width

-- Calculate the area of the original field
def area_of_field : ℕ := length_of_field * width_of_field

-- Calculate the area of the path
def area_of_path : ℕ := area_with_path - area_of_field

-- Calculate the cost of constructing the path
def construction_cost : ℕ := area_of_path * cost_per_sq_m

-- Prove the area of the path is 675 sq m and the cost is Rs. 4725
theorem area_of_path_correct :
  area_of_path = 675 := by
  sorry

theorem construction_cost_correct :
  construction_cost = 4725 := by
  sorry

end area_of_path_correct_construction_cost_correct_l491_491271


namespace words_per_minute_after_break_l491_491698

variable (w : ℕ)

theorem words_per_minute_after_break (h : 10 * 5 - (w * 5) = 10) : w = 8 := by
  sorry

end words_per_minute_after_break_l491_491698


namespace law_I_false_law_II_false_law_III_true_l491_491886

def averaged_with (a b : ℝ) : ℝ := (a + b) / 2

theorem law_I_false : ∀ x y z : ℝ,
  averaged_with (x + 1) (y + z) ≠ averaged_with (x + 1) y + averaged_with (x + 1) z := 
by sorry

theorem law_II_false : ∀ x y z : ℝ,
  x + averaged_with y z ≠ averaged_with (x + 2 * y) (x + 2 * z) := 
by sorry

theorem law_III_true : ∀ x y z : ℝ,
  averaged_with x (averaged_with y z) = averaged_with (averaged_with x y) (averaged_with x z) :=
by sorry

end law_I_false_law_II_false_law_III_true_l491_491886


namespace solution_set_inequality_l491_491444

open Real

-- Given conditions
variable {f : ℝ → ℝ}
variable hDom : ∀ x, 0 < x → f x > 0
variable hDeriv : ∀ x, 0 < x → f' x < f x

theorem solution_set_inequality :
  (∀ x > 0, f x > f' x) →
  { x | e^(x+2) * f(x^2 - x) > e^(x^2) * f(2) } = { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x ∧ x < 2 } :=
by
  sorry

end solution_set_inequality_l491_491444


namespace six_digit_alternating_count_l491_491947

theorem six_digit_alternating_count : 
  ∃ n : ℕ, n = 72 ∧ 
  (∀ (L : List ℕ), 
    (L.length = 6) ∧ 
    (∀ l, l ∈ L → l ∈ [1, 2, 3, 4, 5, 6]) ∧ 
    (L.nodup) ∧ 
    (∀ i, i < 5 → (L.get? i).bind (λ d1, (L.get? (i + 1)).map (λ d2, (d1 % 2 ≠ d2 % 2))) = some true) →
    L.length * (L.get 0).some + 
    L.length * (L.get 1).some + 
    L.length * (L.get 2).some + 
    L.length * (L.get 3).some + 
    L.length * (L.get 4).some + 
    L.length * (L.get 5).some = 72) :=
by {
    sorry
}

end six_digit_alternating_count_l491_491947


namespace max_marks_l491_491651

theorem max_marks (M : ℝ) (h1 : 0.40 * M = 200) : M = 500 := by
  sorry

end max_marks_l491_491651


namespace angle_of_inclination_of_regular_pyramid_l491_491592

theorem angle_of_inclination_of_regular_pyramid :
  ∀ (pyramid : Type) (base : Type) (lateral_face : Type)
    [regular_pyramid pyramid base lateral_face]
    (angle_between_base_and_lateral_face : ℝ)
    (angle_between_side_edge_and_base_edge : ℝ),
  angle_between_base_and_lateral_face = angle_between_side_edge_and_base_edge
  → angle_between_base_and_lateral_face = 35 + 15 / 60 :=
begin
  sorry
end

end angle_of_inclination_of_regular_pyramid_l491_491592


namespace no_a_satisfy_quadratic_equation_l491_491766

theorem no_a_satisfy_quadratic_equation :
  ∀ (a : ℕ), (a > 0) ∧ (a ≤ 100) ∧
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁ * x₂ = 2 * a^2 ∧ x₁ + x₂ = -(3*a + 1)) → false := by
  sorry

end no_a_satisfy_quadratic_equation_l491_491766


namespace proof_problem_l491_491280

variables {A B C D P M N : Type}
variable [real_vector_space A]
variable [real_vector_space B]
variable [real_vector_space C]
variable [real_vector_space D]
variable [real_vector_space P]
variable [real_vector_space M]
variable [real_vector_space N]

def rectangle (A B C D : Type) : Prop := 
  ∃ (D P : Type), perpendicular D P ∧ perpendicular P M ∧ perpendicular P N

theorem proof_problem (h : rectangle A B C D)
  (h1 : ∀ P M : Type, perpendicular P M) 
  (h2 : ∀ P N : Type, perpendicular P N) 
  (h3 : ∀ A C : Type, perpendicular A C) : 
  PM^{2/3} + PN^{2/3} = AC^{2/3} := 
by 
  sorry

end proof_problem_l491_491280


namespace find_n_l491_491123

open Real

-- Definitions based on conditions
def follows_binomial_distribution := (ξ n P : ℝ) → Type :=
  Eξ : n * P = 15,
  Dξ : n * P * (1 - P) = 11.25

-- The mathematical proof problem
theorem find_n {n P : ℝ} (h₁ : n * P = 15) (h₂ : n * P * (1 - P) = 11.25) : n = 60 :=
by
  sorry

end find_n_l491_491123


namespace ms_walker_speed_l491_491565

variables {v : ℝ}

def ms_walker_speed_home (d_to_work : ℝ) (speed_to_work : ℝ) (round_trip_time : ℝ) (d_home : ℝ) :=
  let time_to_work := d_to_work / speed_to_work in
  let time_home := d_home / v in
  time_to_work + time_home = round_trip_time

theorem ms_walker_speed:
  ms_walker_speed_home 24 60 1 24 ↔ v = 40 :=
by sorry

end ms_walker_speed_l491_491565


namespace log_base_16_of_4_l491_491748

theorem log_base_16_of_4 : log 16 4 = 1 / 2 := by
  sorry

end log_base_16_of_4_l491_491748


namespace smallest_odd_number_with_next_primes_l491_491253

theorem smallest_odd_number_with_next_primes :
  let primes := [13, 17, 19, 23]
  ∧ primes.all Prime :=
  ∃ n : ℕ, odd n ∧ (∀ p ∈ primes, p ∣ n) ∧ n = 13 * 17 * 19 * 23 :=
by {
  sorry
}

end smallest_odd_number_with_next_primes_l491_491253


namespace x3_y3_sum_sq_sq_leq_4a10_equality_holds_when_x_eq_y_l491_491060

-- Conditions: x, y are positive real numbers and x + y = 2a
variables {x y a : ℝ}
variable (hxy : x + y = 2 * a)
variable (hx_pos : 0 < x)
variable (hy_pos : 0 < y)

-- Math proof problem: Prove the inequality
theorem x3_y3_sum_sq_sq_leq_4a10 : 
  x^3 * y^3 * (x^2 + y^2)^2 ≤ 4 * a^10 :=
by sorry

-- Equality condition: Equality holds when x = y
theorem equality_holds_when_x_eq_y (h : x = y) :
  x^3 * y^3 * (x^2 + y^2)^2 = 4 * a^10 :=
by sorry

end x3_y3_sum_sq_sq_leq_4a10_equality_holds_when_x_eq_y_l491_491060


namespace doris_weeks_to_cover_expenses_l491_491722

-- Define the constants and conditions from the problem
def hourly_rate : ℝ := 20
def monthly_expenses : ℝ := 1200
def weekday_hours_per_day : ℝ := 3
def weekdays_per_week : ℝ := 5
def saturday_hours : ℝ := 5

-- Calculate total hours worked per week
def weekly_hours := (weekday_hours_per_day * weekdays_per_week) + saturday_hours

-- Calculate weekly earnings
def weekly_earnings := hourly_rate * weekly_hours

-- Finally, the number of weeks required to meet the monthly expenses
def required_weeks := monthly_expenses / weekly_earnings

-- The theorem to prove
theorem doris_weeks_to_cover_expenses : required_weeks = 3 := by
  -- We skip the proof but indicate it needs to be provided
  sorry

end doris_weeks_to_cover_expenses_l491_491722


namespace volume_of_pyramid_is_one_l491_491488

noncomputable def pyramid_volume (base_edge : ℝ) (side_edge : ℝ) (is_midpoint : bool) : ℝ :=
if base_edge = 2 ∧ side_edge = sqrt 3 ∧ is_midpoint then 1 else 0

theorem volume_of_pyramid_is_one :
  pyramid_volume 2 (sqrt 3) true = 1 :=
by sorry

end volume_of_pyramid_is_one_l491_491488


namespace compare_negative_fractions_l491_491354

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l491_491354


namespace solve_problem_l491_491423

noncomputable def problem_statement : Prop :=
  ∀ (α β : ℝ),
  (cos α = -3/5 ∧ (π/2 < α ∧ α < π)) →
  (sin β = -12/13 ∧ (π < β ∧ β < 3 * π / 2)) →
  cos (β - α) = -33/65

theorem solve_problem : problem_statement := 
by
  intros α β hα hβ
  obtain ⟨h_cos_α, h_interval_α⟩ := hα
  obtain ⟨h_sin_β, h_interval_β⟩ := hβ
  sorry

end solve_problem_l491_491423


namespace problem_statement_l491_491138

-- Define the sets U, M, and N
def U := {x : ℕ | 0 < x ∧ x ≤ 6}
def M := {1, 4, 5}
def N := {2, 3, 4}

-- Define the complement of N in U
def CU_N := {x ∈ U | x ∉ N}

-- The theorem statement
theorem problem_statement : M ∩ CU_N = {1, 5} := by
  sorry

end problem_statement_l491_491138


namespace find_a_value_l491_491478

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end find_a_value_l491_491478


namespace find_gamma_delta_l491_491232

def Area (γ δ λ : ℚ) : ℚ := γ * λ - δ * λ^2

theorem find_gamma_delta (γ δ : ℚ) :
  let λ := (36 : ℚ) / 2 in
  let area_triangle := 270 / 2 in -- Area(ΔDEF) is 270, half of which is 135
  Area γ δ λ = area_triangle ->
  (δ = 45 / 7668) ∧ (γ = 9720 / 1278) :=
sorry

end find_gamma_delta_l491_491232


namespace boat_distance_upstream_l491_491502

-- Define the conditions
def river_speed : ℝ := 2
def boat_speed : ℝ := 6
def total_time : ℝ := 12

-- Define the question
theorem boat_distance_upstream : 
  let D := 32 -- assumed correct answer
  let effective_upstream_speed := boat_speed - river_speed
  let effective_downstream_speed := boat_speed + river_speed
  let T_up := D / effective_upstream_speed
  let T_down := D / effective_downstream_speed
  in
  T_up + T_down = total_time :=
by
  -- The proof is skipped
  sorry

end boat_distance_upstream_l491_491502


namespace simple_interest_calculation_l491_491306

theorem simple_interest_calculation :
  let P : ℝ := 8925
  let R : ℝ := 9
  let T : ℝ := 5
  SI = (P * R * T) / 100 := 4016.25 := sorry

end simple_interest_calculation_l491_491306


namespace largest_common_value_largest_common_value_achieved_l491_491185

theorem largest_common_value (b : ℕ) : 
  b < 1000 ∧ b % 4 = 3 ∧ b % 7 = 5 → b ≤ 989 :=
begin
  sorry
end 

theorem largest_common_value_achieved : 
  ∃ (b : ℕ), b < 1000 ∧ b % 4 = 3 ∧ b % 7 = 5 ∧ b = 989 :=
begin
  use 989,
  split,
  { -- b < 1000
    exact nat.lt_of_le_of_lt le_rfl (by norm_num),
  },
  split,
  { -- b % 4 = 3
    norm_num,
  },
  split,
  { -- b % 7 = 5
    norm_num,
  },
  -- b = 989
  refl,
end

end largest_common_value_largest_common_value_achieved_l491_491185


namespace tg_identity_l491_491994

open Real

def tg (θ : ℝ) : ℝ := tan θ

theorem tg_identity (x : ℝ) (k : ℤ) :
  tg x * tg (20 * (π / 180)) + tg (20 * (π / 180)) * tg (40 * (π / 180)) + tg (40 * (π / 180)) * tg x = 1 →
  ∃ k : ℤ, x = 30 * (π / 180) + k * 180 * (π / 180) :=
by {
  sorry
}

end tg_identity_l491_491994


namespace log_base_16_of_4_eq_half_l491_491737

noncomputable def logBase (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_base_16_of_4_eq_half :
  logBase 16 4 = 1 / 2 := by
sorry

end log_base_16_of_4_eq_half_l491_491737


namespace train_track_length_l491_491236

theorem train_track_length (speed1 speed2 distance2 avg_time : ℕ) (h_speed1 : speed1 = 50) (h_speed2 : speed2 = 80) (h_distance2 : distance2 = 240) (h_avg_time : avg_time = 4) :
  let time2 := distance2 / speed2 in
  let time1 := avg_time * 2 - time2 in
  let distance1 := speed1 * time1 in
  distance1 = 250 := 
by 
  sorry

end train_track_length_l491_491236


namespace four_coloring_contradiction_l491_491267

noncomputable def dist (p q : ℝ × ℝ) :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def triangle_vertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((0, 0), (1, 0), (0, 1)) -- A = (0,0), B = (1,0), C = (0,1)

def X (A B C : ℝ × ℝ) : set (ℝ × ℝ) :=
  {p | (∃ t ∈ set.Icc 0 1, p = (t, 0)) ∨ (∃ t ∈ set.Icc 0 1, p = (0, t)) ∨ (∃ t ∈ set.Icc 0 1, p = (t, t))}

theorem four_coloring_contradiction :
  ∀ (c : (ℝ × ℝ) → fin 4), ∃ (p q : ℝ × ℝ), p ∈ X (0, 0) (1, 0) (0, 1) ∧ q ∈ X (0, 0) (1, 0) (0, 1) ∧ c p = c q ∧ dist p q ≥ 2 - real.sqrt 2 := sorry

end four_coloring_contradiction_l491_491267


namespace domain_of_function_l491_491606

noncomputable def domain_of_f : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem domain_of_function : ∀ x : ℝ, (0 ≤ x ∧ x ≤ 3) ∧ (1 ≤ x) ↔ (1 ≤ x ∧ x ≤ 3) :=
by
  intro x
  split
  . intro h
    cases h with h1 h2
    split
    . exact h2
    . exact h1.2
  . intro h
    split
    . split
      . exact le_trans zero_le_one h.1
      . exact h.2
    . exact h.1
  sorry

end domain_of_function_l491_491606


namespace doris_weeks_to_cover_expenses_l491_491723

-- Define the constants and conditions from the problem
def hourly_rate : ℝ := 20
def monthly_expenses : ℝ := 1200
def weekday_hours_per_day : ℝ := 3
def weekdays_per_week : ℝ := 5
def saturday_hours : ℝ := 5

-- Calculate total hours worked per week
def weekly_hours := (weekday_hours_per_day * weekdays_per_week) + saturday_hours

-- Calculate weekly earnings
def weekly_earnings := hourly_rate * weekly_hours

-- Finally, the number of weeks required to meet the monthly expenses
def required_weeks := monthly_expenses / weekly_earnings

-- The theorem to prove
theorem doris_weeks_to_cover_expenses : required_weeks = 3 := by
  -- We skip the proof but indicate it needs to be provided
  sorry

end doris_weeks_to_cover_expenses_l491_491723


namespace region_in_plane_l491_491426

def f (x : ℝ) : ℝ := x^2 - 6 * x + 5

theorem region_in_plane (x y : ℝ) :
  (f x + f y ≤ 0) ∧ (f x - f y ≥ 0) ↔
  ((x - 3)^2 + (y - 3)^2 ≤ 8) ∧ ((x ≥ y ∧ x + y ≥ 6) ∨ (x ≤ y ∧ x + y ≤ 6)) :=
by
  sorry

end region_in_plane_l491_491426


namespace find_p_plus_q_l491_491281

noncomputable theory

def points_on_same_line (A B C: Point) : Prop := False -- Placeholder definition

def triangle_circumcircle_center (A B C: Triangle) : Point := Point.zero -- Placeholder definition

-- Points and lengths
variables {A B C X Y Z O1 O2 O3 : Point}

-- Circles and circumcircles
variables (circumcircle_ABC : Circle)
variables [circumcircle_XBC : Circle]
variables [circumcircle_YAC : Circle]
variables [circumcircle_ZAB : Circle]

-- Triangle lengths
variables (AB BC AC : ℝ)
variables (XC YC XB YAY ZB XA AZ AY : ℝ)

-- Given/known data
axiom length_AB : AB = 26
axiom length_BC : BC = 30
axiom length_AC : AC = 28

-- Arc condition means equal segment lengths
axiom arc_BZ_YC : ZB = YC
axiom arc_AZ_XC : AZ = XC
axiom arc_AY_XB : AY = XB

-- Problem statement: find p + q where XC = p / q, and p, q are relatively prime
theorem find_p_plus_q (p q : ℕ) (hrelatively_prime : nat.coprime p q) : 
  let XC := (4 : ℝ) in
  let p := 4 in
  let q := 1 in
  XC = p / q :=
by sorry

end find_p_plus_q_l491_491281


namespace nearest_integer_sum_values_g_equals_2024_l491_491669

noncomputable def g (x : ℝ) : ℝ := sorry -- g is a function from ℝ to ℝ

theorem nearest_integer_sum_values_g_equals_2024 :
  (∀ x : ℝ, x ≠ 0 → 3 * g x + 2 * g (1 / x) = 7 * x + 6) →
  (forall {x}, g x = 2024 → x ≠ 0) →
  (∀ S : set ℝ, (∀ x : ℝ, g x = 2024 → x ∈ S) →
    abs ((∑ x in S, x) - 482) < 1) :=
  sorry

end nearest_integer_sum_values_g_equals_2024_l491_491669


namespace stan_needs_more_minutes_l491_491590

/-- Stan has 10 songs each of 3 minutes and 15 songs each of 2 minutes. His run takes 100 minutes.
    Prove that he needs 40 more minutes of songs in his playlist. -/
theorem stan_needs_more_minutes 
    (num_3min_songs : ℕ) 
    (num_2min_songs : ℕ) 
    (time_per_3min_song : ℕ) 
    (time_per_2min_song : ℕ) 
    (total_run_time : ℕ) 
    (given_minutes_3min_songs : num_3min_songs = 10)
    (given_minutes_2min_songs : num_2min_songs = 15)
    (given_time_per_3min_song : time_per_3min_song = 3)
    (given_time_per_2min_song : time_per_2min_song = 2)
    (given_total_run_time : total_run_time = 100)
    : num_3min_songs * time_per_3min_song + num_2min_songs * time_per_2min_song = 60 →
      total_run_time - (num_3min_songs * time_per_3min_song + num_2min_songs * time_per_2min_song) = 40 := 
by
    sorry

end stan_needs_more_minutes_l491_491590


namespace slices_eaten_l491_491186

theorem slices_eaten (total_slices : Nat) (slices_left : Nat) (expected_slices_eaten : Nat) :
  total_slices = 32 →
  slices_left = 7 →
  expected_slices_eaten = 25 →
  total_slices - slices_left = expected_slices_eaten :=
by
  intros
  sorry

end slices_eaten_l491_491186


namespace find_principal_l491_491652

-- Definitions used directly from the problem conditions in a)
def SimpleInterest : ℝ := 4020.75
def Rate : ℝ := 9 / 100 -- converting percentage to a fraction
def Time : ℝ := 5
def Principal (SI R T : ℝ) : ℝ := SI / (R * T)

-- We need to prove Principal(SI, R, T) = 8935
theorem find_principal :
  Principal SimpleInterest Rate Time = 8935 := 
sorry

end find_principal_l491_491652


namespace symmetric_point_origin_l491_491187

theorem symmetric_point_origin (P : ℝ × ℝ) (h : P = (3, -4)) :
  let symmetric_P := (-P.1, -P.2)
  symmetric_P = (-3, 4) :=
by
  simp [*, h]
  sorry

end symmetric_point_origin_l491_491187


namespace value_of_a_l491_491485

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end value_of_a_l491_491485


namespace point_not_in_region_l491_491986

theorem point_not_in_region (x y : ℝ) : (3 * x + 2 * y) > 3 ↔ ¬(x = 0 ∧ y = 0) :=
by {
  sorry,
}

end point_not_in_region_l491_491986


namespace inverse_function_domain_l491_491075

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem inverse_function_domain :
  ∃ (g : ℝ → ℝ), (∀ x, 0 ≤ x → f (g x) = x) ∧ (∀ y, 0 ≤ y → g (f y) = y) ∧ (∀ x, 0 ≤ x ↔ 0 ≤ g x) :=
by
  sorry

end inverse_function_domain_l491_491075


namespace train_length_is_800_meters_l491_491273

/-- Define the speeds in km/h -/
def train_speed_kmph : ℝ := 100
def motorbike_speed_kmph : ℝ := 64

/-- Convert the speeds to m/s -/
def kmph_to_mps (kmph : ℝ) : ℝ := (kmph * 1000) / 3600

/-- Define the speeds in m/s -/
def train_speed_mps := kmph_to_mps train_speed_kmph
def motorbike_speed_mps := kmph_to_mps motorbike_speed_kmph

/-- The time it takes for the train to overtake the motorbike in seconds -/
def overtaking_time_seconds : ℝ := 80

/-- The relative speed in m/s -/
def relative_speed : ℝ := train_speed_mps - motorbike_speed_mps

/-- Prove the length of the train -/
theorem train_length_is_800_meters : relative_speed * overtaking_time_seconds = 800 := by
  sorry

end train_length_is_800_meters_l491_491273


namespace negative_fraction_comparison_l491_491365

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l491_491365


namespace range_ratio_l491_491862

-- Given data
def C1_cartesian (x y : ℝ) : Prop := x + y = 2
def C2_parametric (φ : ℝ) (x y : ℝ) : Prop :=
  φ ∈ Ico 0 (2 * Real.pi) ∧ (
    x = 3 + 3 * Real.cos φ ∧
    y = 3 * Real.sin φ)

def ray (α : ℝ) (ρ θ : ℝ) : Prop := θ = α ∧ ρ ≥ 0

-- Definitions for polar forms
def C1_polar (ρ θ : ℝ) : Prop := ρ * Real.sin(θ + Real.pi / 4) = Real.sqrt 2
def C2_polar (ρ θ : ℝ) : Prop := ρ = 6 * Real.cos θ

-- Definitions for ratios
def dist_OA (α : ℝ) : ℝ := Real.sqrt 2 / (Real.sin(α + Real.pi / 4))
def dist_OB (α : ℝ) : ℝ := 6 * Real.abs (Real.cos α)

-- Problem statement to prove
theorem range_ratio (α : ℝ) (h : α ∈ Icc 0 (Real.pi / 4)) :
  3 ≤ dist_OB α / dist_OA α ∧ dist_OB α / dist_OA α ≤ 3 / 2 * (Real.sqrt 2 + 1) := sorry

end range_ratio_l491_491862


namespace Power_of_Point_l491_491663

theorem Power_of_Point 
(A B C : Point)
(H1 H2 : Point) 
(P Q S : Point) 
(h1 : Circle (A B C)) 
(h2 : Line H1 H2)
(h3 : H1 ∈ Line A B ∧ H2 ∈ Line A B) 
(h4 : Segment P Q ∈ h1) 
(h5 : S ∈ (Line H1 H2) ∧ S ∈ (Line C (extension_point A B C))) :
  SP * SQ = SH1 * SH2 := 
by
  sorry

end Power_of_Point_l491_491663


namespace bisection_next_m_l491_491644

def f : ℝ → ℝ := sorry -- Exact form of the function f is not given

open Real

theorem bisection_next_m (h1 : f 1 = -2) (h3 : f 3 = 0.625) (h2 : f 2 = -0.984) : 
  ∃ m, m = 2.5 :=
begin
  use 2.5,
  -- Proof omitted
  sorry
end

end bisection_next_m_l491_491644


namespace max_average_numbers_l491_491885

-- Let 2S be the total weight of a set of 100 weights.
variable (S : ℕ)

-- A natural number k is an "average number" 
-- if there exists a subset of k weights that sums to S.
def is_average_number (k : ℕ) (weights : Fin 100 → ℕ) : Prop :=
  ∃ (subset : Finset (Fin 100)), subset.card = k ∧ subset.sum (λ i, weights i) = S

-- Define the set of weights consisting of 100 elements. 
variable (weights : Fin 100 → ℕ)
variable (h_total_weight : (Finset.univ : Finset (Fin 100)).sum weights = 2 * S)

-- Each natural number k such that 1 ≤ k ≤ 100 can potentially be checked if it's an average number.
-- We prove the maximum number of average numbers.
theorem max_average_numbers : 
  ∃ (count : ℕ), count = 97 ∧ ∀ k, is_average_number k weights → k ≤ 97 := 
  sorry

end max_average_numbers_l491_491885


namespace sampledSequence_correct_l491_491219

-- Define the total number of students
def totalStudents : ℕ := 60

-- Define the number of students sampled
def sampledStudents : ℕ := 5

-- Define the sampling interval
def samplingInterval : ℕ := totalStudents / sampledStudents

-- The sequence that is to be proved as the systematic sampled sequence
def sampledSequence := [6, 18, 30, 42, 54]

-- The theorem to be proved
theorem sampledSequence_correct : 
  ∃ start, ∀ i, i < sampledStudents → sampledSequence[i] = start + i * samplingInterval :=
by
  sorry

end sampledSequence_correct_l491_491219


namespace share_per_person_is_135k_l491_491103

noncomputable def calculate_share : ℝ :=
  (0.90 * (500000 * 1.20)) / 4

theorem share_per_person_is_135k : calculate_share = 135000 :=
by
  sorry

end share_per_person_is_135k_l491_491103


namespace necessary_condition_l491_491916

theorem necessary_condition (x : ℝ) (h : x > 0) : x > -2 :=
by {
  exact lt_trans (by norm_num) h,
}

end necessary_condition_l491_491916


namespace poll_total_l491_491318

-- Define the conditions
variables (men women : ℕ)
variables (pct_favor : ℝ := 35) (women_opposed : ℕ := 39)
noncomputable def total_people (men women : ℕ) : ℕ := men + women

-- We need to prove the total number of people polled, given the conditions
theorem poll_total (h1 : men = women)
  (h2 : (pct_favor / 100) * women + (39 : ℝ) / (65 / 100) = (women: ℝ)) :
  total_people men women = 120 :=
sorry

end poll_total_l491_491318


namespace polynomial_factorization_l491_491447

noncomputable def polynomial_expr (a b c : ℝ) :=
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2)

noncomputable def factored_form (a b c : ℝ) :=
  (a - b) * (b - c) * (c - a) * (b^2 + c^2 + a^2)

theorem polynomial_factorization (a b c : ℝ) :
  polynomial_expr a b c = factored_form a b c :=
by {
  sorry
}

end polynomial_factorization_l491_491447


namespace distinct_positive_rationals_sum_squares_l491_491583

theorem distinct_positive_rationals_sum_squares (n : ℕ) (h : 0 < n) :
  ∃ S : finset ℚ, S.card = n ∧ (∀ x ∈ S, 0 < x) ∧ (∀ x y ∈ S, x ≠ y) ∧ (∑ x in S, x^2) = n := sorry

end distinct_positive_rationals_sum_squares_l491_491583


namespace number_of_possible_values_m_l491_491852

-- Definitions of the lines
def l1 : LinearMap ℝ (ℝ × ℝ) ℝ := fun p ↦ 4 * p.1 + p.2 - 4
def l2 (m : ℝ) : LinearMap ℝ (ℝ × ℝ) ℝ := fun p ↦ m * p.1 + p.2
def l3 (m : ℝ) : LinearMap ℝ (ℝ × ℝ) ℝ := fun p ↦ 2 * p.1 - 3 * m * p.2 - 4

-- Condition for lines not forming a triangle
def not_form_triangle (m : ℝ) : Prop :=
  (l1.slope = l2 m.slope) ∨ (l1.slope = l3 m.slope) ∨ (l2 m.slope = l3 m.slope) ∨
  (∃ (x y : ℝ), l1 (x, y) = 0 ∧ l2 m (x, y) = 0 ∧ l3 m (x, y) = 0)

-- The Lean 4 statement
theorem number_of_possible_values_m : 
  ∃ (S : Set ℝ), (∀ m ∈ S, not_form_triangle m) ∧ S.card = 4 :=
by
  sorry   -- Proof to be provided

end number_of_possible_values_m_l491_491852


namespace find_a_b_sum_l491_491942

-- Conditions
def f (x a b : ℝ) : ℝ := x^3 - a * x^2 - b * x + a^2

def f_prime (x a b : ℝ) : ℝ := 3 * x^2 - 2 * a * x - b

theorem find_a_b_sum (a b : ℝ) (h1 : f_prime 1 a b = 0) (h2 : f 1 a b = 10) : a + b = 7 := 
sorry

end find_a_b_sum_l491_491942


namespace compare_fractions_l491_491704

theorem compare_fractions :
  (2 : ℝ) / 3 < real.sqrt 2 / 2 :=
sorry

end compare_fractions_l491_491704


namespace noah_ava_zoo_trip_l491_491906

theorem noah_ava_zoo_trip (zoo_ticket_cost : ℝ) (bus_fare_cost : ℝ) (lunch_snacks_remaining : ℝ) (noah_ava_count : ℝ) :
  zoo_ticket_cost = 5 → 
  bus_fare_cost = 1.5 → 
  lunch_snacks_remaining = 24 →
  noah_ava_count = 2 →
  noah_ava_count * zoo_ticket_cost + noah_ava_count * bus_fare_cost * 2 + lunch_snacks_remaining = 40 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end noah_ava_zoo_trip_l491_491906


namespace quadratic_minimum_val_l491_491676

theorem quadratic_minimum_val (p q x : ℝ) (hp : p > 0) (hq : q > 0) : 
  (∀ x, x^2 - 2 * p * x + 4 * q ≥ p^2 - 4 * q) := 
by
  sorry

end quadratic_minimum_val_l491_491676


namespace problem1_domain_valid_problem2_domain_valid_l491_491757

-- Definition of the domains as sets.

def domain1 (x : ℝ) : Prop := ∃ k : ℤ, x = 2 * k * Real.pi

def domain2 (x : ℝ) : Prop := (-3 ≤ x ∧ x < -Real.pi / 2) ∨ (0 < x ∧ x < Real.pi / 2)

-- Theorem statements for the domains.

theorem problem1_domain_valid (x : ℝ) : (∀ y : ℝ, y = Real.log (Real.cos x) → y ≥ 0) ↔ domain1 x := sorry

theorem problem2_domain_valid (x : ℝ) : 
  (∀ y : ℝ, y = Real.log (Real.sin (2 * x)) + Real.sqrt (9 - x ^ 2) → y ∈ Set.Icc (-3) 3) ↔ domain2 x := sorry

end problem1_domain_valid_problem2_domain_valid_l491_491757


namespace polygon_sides_l491_491500

theorem polygon_sides (ext_angle : ℝ) (sum_ext_angles : ℝ) (h1 : ext_angle = 72) (h2 : sum_ext_angles = 360) :
  ∃ n : ℕ, sum_ext_angles / ext_angle = n ∧ n = 5 := 
by
  use (sum_ext_angles / ext_angle).to_nat
  split
  · sorry  -- This will be the proof that the number of sides calculated is 5.
  · sorry  -- This will be the proof that the number of sides equals 5.

end polygon_sides_l491_491500


namespace lolita_milk_per_week_l491_491554

def weekday_milk : ℕ := 3
def saturday_milk : ℕ := 2 * weekday_milk
def sunday_milk : ℕ := 3 * weekday_milk
def total_milk_week : ℕ := 5 * weekday_milk + saturday_milk + sunday_milk

theorem lolita_milk_per_week : total_milk_week = 30 := 
by 
  sorry

end lolita_milk_per_week_l491_491554


namespace max_value_a2a6_l491_491863

noncomputable def max_prod_arithmetic_seq (a : ℕ → ℝ) (h : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_4 : a 3 = 2) : ℝ :=
  max (a 1 * a 5)

theorem max_value_a2a6 (a : ℕ → ℝ) (h_arithmetic : ∀ n, a n = a 0 + n * (a 1 - a 0))
  (h_a4 : a 3 = 2) : max_prod_arithmetic_seq a h_arithmetic h_a4 = 4 := 
sorry

end max_value_a2a6_l491_491863


namespace sum_of_roots_l491_491952

theorem sum_of_roots (r p q : ℝ) 
  (h1 : (3 : ℝ) * r ^ 3 - (9 : ℝ) * r ^ 2 - (48 : ℝ) * r - (12 : ℝ) = 0)
  (h2 : (3 : ℝ) * p ^ 3 - (9 : ℝ) * p ^ 2 - (48 : ℝ) * p - (12 : ℝ) = 0)
  (h3 : (3 : ℝ) * q ^ 3 - (9 : ℝ) * q ^ 2 - (48 : ℝ) * q - (12 : ℝ) = 0)
  (roots_distinct : r ≠ p ∧ r ≠ q ∧ p ≠ q) :
  r + p + q = 3 := 
sorry

end sum_of_roots_l491_491952


namespace parabola_property_l491_491547

noncomputable def parabola : { p : ℝ // p > 0 } → Set (ℝ × ℝ)
| ⟨p, _⟩ := { ⟨x, y⟩ | y^2 = 2 * p * x }

noncomputable def is_focus_of_parabola : (ℝ × ℝ) → (ℝ × ℝ) → Prop
| ⟨x₀, y₀⟩ ⟨1, 0⟩ := x₀ = 1 ∧ y₀ = 0

noncomputable def intersection_points (parabola : Set (ℝ × ℝ)) (line : ℝ → (ℝ × ℝ) → Prop) : Set (ℝ × ℝ) := 
{ p ∈ parabola | ∃ k : ℝ, line k p }

theorem parabola_property :
  ∀ {p : ℝ} (h : p > 0),
  let C := parabola ⟨p, h⟩ in
  C = { ⟨x, y⟩ | y^2 = 4 * x } → 
  ∀ (P1 P2 Q1 Q2 : ℝ × ℝ) (l1 l2 : ℝ → (ℝ × ℝ) → Prop),
    is_focus_of_parabola (1, 0) (1, 0) →
    l1 ≠ l2 →
    (∀ k, l1 k (1, 0)) ∧ (∀ k, l2 k (1, 0)) ∧ 
    P1 ∈ intersection_points C l1 ∧ P2 ∈ intersection_points C l1 ∧ 
    Q1 ∈ intersection_points C l2 ∧ Q2 ∈ intersection_points C l2 →
    (1 / real.dist P1 P2 + 1 / real.dist Q1 Q2 = 1 / 4) := 
sorry

end parabola_property_l491_491547


namespace average_after_15th_inning_l491_491289

theorem average_after_15th_inning (A : ℝ) 
    (h_avg_increase : (14 * A + 75) = 15 * (A + 3)) : 
    A + 3 = 33 :=
by {
  sorry
}

end average_after_15th_inning_l491_491289


namespace wire_length_after_two_bends_is_three_l491_491854

-- Let's define the initial length and the property of bending the wire.
def initial_length : ℕ := 12

def half_length (length : ℕ) : ℕ :=
  length / 2

-- Define the final length after two bends.
def final_length_after_two_bends : ℕ :=
  half_length (half_length initial_length)

-- The theorem stating that the final length is 3 cm after two bends.
theorem wire_length_after_two_bends_is_three :
  final_length_after_two_bends = 3 :=
by
  -- The proof can be added later.
  sorry

end wire_length_after_two_bends_is_three_l491_491854


namespace triangle_inequality_abc_l491_491788

theorem triangle_inequality_abc {a b c : ℝ} 
  (h_area : (1 / 4) = (1 / 4)) -- Area of the triangle
  (h_r : 1 = 1)               -- Radius of the circumcircle
  (h_eq : a * b * c = 1)      -- Derived condition from given info
  :
  let s := sqrt a + sqrt b + sqrt c in
  let t := (1 / a) + (1 / b) + (1 / c) in
  t ≥ s :=
begin
  intros,
  sorry
end

end triangle_inequality_abc_l491_491788


namespace sum_of_products_not_equal_two_l491_491510

theorem sum_of_products_not_equal_two
    (table : Fin 10 → Fin 10 → ℤ)
    (h1 : ∀ i j, table i j = 1 ∨ table i j = -1) :
  (∑ i : Fin 10, ∏ j, table i j) + (∑ j : Fin 10, ∏ i, table i j) ≠ 2 := 
sorry

end sum_of_products_not_equal_two_l491_491510


namespace desiredCircleEquation_l491_491032

-- Definition of the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 + x - 6*y + 3 = 0

-- Definition of the given line
def givenLine (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- The required proof problem statement
theorem desiredCircleEquation :
  (∀ P Q : ℝ × ℝ, givenCircle P.1 P.2 ∧ givenLine P.1 P.2 → givenCircle Q.1 Q.2 ∧ givenLine Q.1 Q.2 →
  (P ≠ Q) → 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0)) :=
by
  -- Proof omitted
  sorry

end desiredCircleEquation_l491_491032


namespace borya_number_l491_491689

theorem borya_number (n : ℕ) (h1 : 10 ≤ n) (h2 : n < 100) 
  (h3 : (n * 2 + 5) * 5 = 715) : n = 69 :=
sorry

end borya_number_l491_491689


namespace garden_length_l491_491083

theorem garden_length (columns : ℕ) (distance_between_trees : ℕ) (boundary_distance : ℕ) (h_columns : columns = 12) (h_distance_between_trees : distance_between_trees = 2) (h_boundary_distance : boundary_distance = 5) : 
  ((columns - 1) * distance_between_trees + 2 * boundary_distance) = 32 :=
by 
  sorry

end garden_length_l491_491083


namespace compare_rat_neg_l491_491352

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l491_491352


namespace problem_a_b_c_ge_neg2_l491_491901

theorem problem_a_b_c_ge_neg2 {a b c : ℝ} (ha : a < 0) (hb : b < 0) (hc : c < 0) :
  (a + 1 / b > -2) ∨ (b + 1 / c > -2) ∨ (c + 1 / a > -2) → False :=
by
  sorry

end problem_a_b_c_ge_neg2_l491_491901


namespace minimum_income_l491_491171

noncomputable section

def monic_quadratic (a b c : ℤ) : Prop :=
  a = 1

def root_of_quadratic (f : ℤ → ℤ) (x : ℤ) : Prop :=
  f x = 0

def distinct_roots (fs : list (ℤ → ℤ)) : Prop :=
  ∀ (x : ℤ), ∃ (f : ℤ → ℤ) (g : ℤ → ℤ), f ≠ g ∧ root_of_quadratic f x ∧ root_of_quadratic g x

theorem minimum_income (fs : list (ℤ → ℤ)) (h_len : fs.length = 1004)
  (h_monic : ∀ f ∈ fs, ∃ a b c : ℤ, f = λ x, a*x^2 + b*x + c ∧ monic_quadratic a b c)
  (h_roots : ∃ (roots : list ℤ), roots.length = 2008 ∧ ∀ x ∈ roots, ∃ f ∈ fs, root_of_quadratic f x) :
  ∀ i j, i ≠ j → (fs.nth i) ≠ (fs.nth j) → ∀ x, (fs.nth i x) ≠ (fs.nth j x) :=
by
  sorry

end minimum_income_l491_491171


namespace computer_price_increase_l491_491850

theorem computer_price_increase
  (P : ℝ)
  (h1 : 1.30 * P = 351) :
  (P + 1.30 * P) / P = 2.3 := by
  sorry

end computer_price_increase_l491_491850


namespace aimee_poll_l491_491313

theorem aimee_poll (P : ℕ) (h1 : 35 ≤ 100) (h2 : 39 % (P/2) = 39) : P = 120 := 
by sorry

end aimee_poll_l491_491313


namespace maximum_b_value_l491_491052

noncomputable def f (a x : ℝ) := (1 / 2) * x ^ 2 + a * x
noncomputable def g (a b x : ℝ) := 2 * a ^ 2 * Real.log x + b

theorem maximum_b_value (a b : ℝ) (h_a : 0 < a) :
  (∃ x : ℝ, f a x = g a b x ∧ (deriv (f a) x = deriv (g a b) x))
  → b ≤ Real.exp (1 / 2) := 
sorry

end maximum_b_value_l491_491052


namespace find_k_l491_491778

noncomputable def sequence (n : ℕ) : ℕ := sorry -- placeholder for the sequence definition 

axiom a_1 : sequence 1 = 24
axiom a_2 : sequence 2 = 51
axiom a_k_zero (k : ℕ) : sequence k = 0

axiom recursion_relation (n k : ℕ) (h_1 : 2 ≤ n ∧ n ≤ k - 1) :
  sequence (n + 1) = sequence (n - 1) - n / sequence n

theorem find_k (k : ℕ) (h_k : ∀ n, 2 ≤ n ∧ n ≤ k - 1 → sequence (n + 1) = sequence (n - 1) - n / sequence n)
  (h_1 : sequence 1 = 24) (h_2 : sequence 2 = 51) (h_k0 : sequence k = 0) : k = 50 :=
sorry

end find_k_l491_491778


namespace number_of_valid_questioning_methods_l491_491494

noncomputable def combination (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def permutation (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

def valid_questioning_methods : ℕ :=
  let domestic_teams := 6
  let foreign_teams := 3
  let case1 := combination domestic_teams 2 * combination foreign_teams 1 * permutation 2 2
  let case2 := combination domestic_teams 1 * combination foreign_teams 2 * permutation 3 3
  case1 + case2

theorem number_of_valid_questioning_methods : valid_questioning_methods = 198 := sorry

end number_of_valid_questioning_methods_l491_491494


namespace necessary_condition_not_sufficient_condition_l491_491915

def P (x : ℝ) := x > 0
def Q (x : ℝ) := x > -2

theorem necessary_condition : ∀ x: ℝ, P x → Q x := 
by sorry

theorem not_sufficient_condition : ∃ x: ℝ, Q x ∧ ¬ P x := 
by sorry

end necessary_condition_not_sufficient_condition_l491_491915


namespace value_of_y_l491_491839

theorem value_of_y (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 24) : y = 96 :=
by
  sorry

end value_of_y_l491_491839


namespace max_value_y_l491_491472

theorem max_value_y : ∀ x : ℝ, ∃ y : ℝ, y = -4 * x^2 + 10 ∧ (∀ x : ℝ, y ≤ -4 * x^2 + 10) :=
by
  intro x
  use 10
  split
  { rw [mul_zero, add_zero] }
  intro x
  calc -4 * x^2 + 10 ≤ 10 : sorry

end max_value_y_l491_491472


namespace calculate_BPC_minus_AQC_l491_491523

-- The context setup: a triangle ABC with points P, Q, H, and R as described
variables {A B C P Q H R : Type}
variables [inhabited A] [inhabited B] [inhabited C]
variables [inhabited P] [inhabited Q] [inhabited H]

-- Defining the conditions
def triangle_ABC (A B C : Type) : Prop := true -- Placeholder for the triangle definition
def altitude_AP (A P H : Type) : Prop := true  -- Placeholder for altitude AP meeting at H
def altitude_BQ (B Q H : Type) : Prop := true  -- Placeholder for altitude BQ meeting at H

-- Given conditions
def HP_eq_3 (H P : Type) : Prop := sorry -- Placeholder for HP = 3
def HQ_eq_7 (H Q : Type) : Prop := sorry -- Placeholder for HQ = 7
def BR_eq_2AR (B R A : Type) : Prop := sorry -- Placeholder for BR = 2AR

-- The goal statement to be proven
theorem calculate_BPC_minus_AQC :
  triangle_ABC A B C →
  altitude_AP A P H →
  altitude_BQ B Q H →
  HP_eq_3 H P →
  HQ_eq_7 H Q →
  BR_eq_2AR B R A →
  (BP * PC - AQ * QC = -40) :=
by
  sorry

end calculate_BPC_minus_AQC_l491_491523


namespace sqrt_0_09_eq_0_3_l491_491959

theorem sqrt_0_09_eq_0_3 : Real.sqrt 0.09 = 0.3 := 
by 
  sorry

end sqrt_0_09_eq_0_3_l491_491959


namespace eight_digit_positive_integers_l491_491833

theorem eight_digit_positive_integers : 
  let choices_first_digit := 9 in
  let choices_other_digits := 10 in
  choices_first_digit * choices_other_digits ^ 7 = 90000000 :=
by 
  simp only [choices_first_digit, choices_other_digits],
  norm_num

end eight_digit_positive_integers_l491_491833


namespace no_intersection_abs_value_graphs_l491_491057

theorem no_intersection_abs_value_graphs : 
  ∀ (x : ℝ), ¬ (|3 * x + 6| = -|4 * x - 1|) :=
by
  intro x
  sorry

end no_intersection_abs_value_graphs_l491_491057


namespace delta_max_success_rate_l491_491504

noncomputable def delta_max_success_ratio : ℚ :=
  ∃ (x y z w u v : ℕ), 
    (0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧ 0 < u ∧ 0 < v) ∧
    (x/y < 3/5) ∧
    (z/w < 3/5) ∧ 
    (u < v) ∧ 
    (y + w + v = 600) ∧
    (x + z + u) / 600 = 539 / 600

theorem delta_max_success_rate : delta_max_success_ratio :=
sorry

end delta_max_success_rate_l491_491504


namespace maxim_birth_probability_l491_491158

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end maxim_birth_probability_l491_491158


namespace probability_at_least_four_girls_l491_491588

noncomputable def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

theorem probability_at_least_four_girls (n : ℕ) (k : ℕ) (p : ℝ) (q : ℝ) :
  n = 7 ∧ k ≥ 4 ∧ p = 0.5 ∧ q = 0.5 →
  (∑ i in finset.Icc 4 7, (binomial 7 i * p^i * q^(7-i))) = 1/2 :=
by
  sorry

end probability_at_least_four_girls_l491_491588


namespace ratio_of_pete_to_susan_l491_491911

noncomputable def Pete_backward_speed := 12 -- in miles per hour
noncomputable def Pete_handstand_speed := 2 -- in miles per hour
noncomputable def Tracy_cartwheel_speed := 4 * Pete_handstand_speed -- in miles per hour
noncomputable def Susan_forward_speed := Tracy_cartwheel_speed / 2 -- in miles per hour

theorem ratio_of_pete_to_susan :
  Pete_backward_speed / Susan_forward_speed = 3 := 
sorry

end ratio_of_pete_to_susan_l491_491911


namespace log_base_16_of_4_l491_491742

theorem log_base_16_of_4 : 
  (16 = 2^4) →
  (4 = 2^2) →
  (∀ (b a c : ℝ), b > 0 → b ≠ 1 → c > 0 → c ≠ 1 → log b a = log c a / log c b) →
  log 16 4 = 1 / 2 :=
by
  intros h1 h2 h3
  sorry

end log_base_16_of_4_l491_491742


namespace find_m_and_tan_alpha_l491_491789

theorem find_m_and_tan_alpha (α : ℝ) (m : ℝ) (h1 : sin α = 1 / 3) (h2 : cos α = (4 * m^2 + 1)^(-1/2)) :
  (m = sqrt 2 ∨ m = -sqrt 2) ∧ (tan α = sqrt 2 / 4 ∨ tan α = -sqrt 2 / 4) :=
by
  sorry

end find_m_and_tan_alpha_l491_491789


namespace polar_to_rectangular_l491_491711

theorem polar_to_rectangular (r θ : ℝ) (h1 : r = 3) (h2 : θ = Real.pi / 2) :
  ∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ (x, y) = (0, 3) :=
by
  use [r * Real.cos θ, r * Real.sin θ]
  rw [h1, h2]
  simp
  split
  { exact Real.cos_pi_div_two }
  split
  { exact Real.sin_pi_div_two }
  { sorry }

end polar_to_rectangular_l491_491711


namespace coeff_x5_in_expansion_l491_491383

noncomputable def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem coeff_x5_in_expansion
  (x : ℂ) : 
  (coeff (X : ℂ[X]) 5 ((X^2 - X - 2)^3)) = -3 :=
by 
  sorry

end coeff_x5_in_expansion_l491_491383


namespace Amanda_tickets_third_day_l491_491691

theorem Amanda_tickets_third_day :
  (let total_tickets := 80
   let first_day_tickets := 5 * 4
   let second_day_tickets := 32

   total_tickets - (first_day_tickets + second_day_tickets) = 28) :=
by
  sorry

end Amanda_tickets_third_day_l491_491691


namespace kim_probability_red_and_blue_shoe_l491_491111

noncomputable def probability_red_and_blue_shoe : ℚ :=
  if Kim_has_10_pairs_of_shoes : ∃ s : Finset (Fin 20), s.card = 20 ∧ (∀ p ∈ powerset_len 2 s, pair_color_differs p) -- Conditions Kim_has_10_pairs_of_shoes
  then
    let red_probability := 2/20 in  -- Probability of picking a red shoe
    let blue_probability := 2/19 in -- Probability of picking a blue shoe given a red shoe has been picked
    red_probability * blue_probability  -- Combined probability of both events
  else
    0

theorem kim_probability_red_and_blue_shoe (Kim_has_10_pairs_of_shoes : ∃ s : Finset (Fin 20), s.card = 20 ∧ (∀ p ∈ powerset_len 2 s, pair_color_differs p)) :
  probability_red_and_blue_shoe = 1 / 95 :=
by
  swap  -- This theorem proves the equality
  sorry

end kim_probability_red_and_blue_shoe_l491_491111


namespace find_a_value_l491_491479

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end find_a_value_l491_491479


namespace line_through_A_with_equal_intercepts_circle_with_diameter_AB_l491_491018

theorem line_through_A_with_equal_intercepts (A B : ℝ × ℝ) (hA : A = (4, 6)) (hB : B = (-2, 4)) :
  ∃ l : ℝ → ℝ, (∀ x, l x = 10 - x) :=
sorry

theorem circle_with_diameter_AB (A B : ℝ × ℝ) (hA : A = (4, 6)) (hB : B = (-2, 4)) :
  ∃ center : ℝ × ℝ, ∃ r : ℝ,
    center = (1, 5) ∧ r = sqrt 10 ∧ (∀ x y, ((x - 1)^2 + (y - 5)^2 = 10)) :=
sorry

end line_through_A_with_equal_intercepts_circle_with_diameter_AB_l491_491018


namespace general_formula_a_proof_sum_b_telescope_l491_491026

noncomputable def general_formula_a (n : ℕ) : ℕ := 2 * n - 1

def sum_S (n : ℕ) (a : ℕ → ℕ) : ℕ := ∑ i in range n, a i

theorem general_formula_a_proof (n : ℕ) (S_n : ℕ) (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≠ 0 → n * a n - sum_S n a = n^2 - n)
  : a n = general_formula_a n :=
sorry

noncomputable def b (n : ℕ) (a : ℕ → ℕ) : ℚ := 
  2 ^ (a n) / ((2 ^ (a n) - 1) * (2 ^ (a (n + 1)) - 1))

theorem sum_b_telescope (n : ℕ) (a : ℕ → ℕ) (T_n : ℚ) 
  (h : ∀ n : ℕ, a n = general_formula_a n)
  : ∑ i in range n, b i a = T_n :=
  T_n = 1 / 3 * (1 - 1 / (2^(2*n+1) - 1))
sorry

end general_formula_a_proof_sum_b_telescope_l491_491026


namespace moles_of_acetic_acid_l491_491836

-- Definitions
def acetic_acid : Type := Unit
def sodium_hydroxide : Type := Unit
def sodium_acetate : Type := Unit
def water : Type := Unit

-- Given conditions (balanced reaction)
def reaction (acetic_acid sodium_hydroxide sodium_acetate water : Type) :=
  (acetic_acid, sodium_hydroxide) → (sodium_acetate, water)

-- Question: Prove that 2 moles of acetic acid react with 2 moles of sodium hydroxide
theorem moles_of_acetic_acid (acetic_acid sodium_hydroxide sodium_acetate water : Type)
  (r : reaction acetic_acid sodium_hydroxide sodium_acetate water) : 
  (2 * acetic_acid → (2 * sodium_acetate, 2 * water)) :=
sorry

end moles_of_acetic_acid_l491_491836


namespace no_three_collinear_l491_491544

theorem no_three_collinear (p : ℕ) (hp : p.prime) (hp_odd : p % 2 = 1) :
  ∃ (points : Fin p → Fin p × Fin p), 
    (∀ i : Fin p, points i = (i, ⟨i.val^2 % p, sorry⟩)) ∧ 
    (∀ i j k : Fin p, i ≠ j → j ≠ k → i ≠ k → 
      ¬ collinear ℝ 
        {points i, points j, points k}) :=
sorry

end no_three_collinear_l491_491544


namespace area_trapezoid_ABCD_l491_491505

-- Definitions of necessary conditions
variables {P : Type} [linear_ordered_field P]
variables {A B C D E : P}
variables {area : P → P → P → P}

-- Trapezoid ABCD with AB parallel to CD
variables (h1 : AB == CD)
-- The intersection of diagonals AC and BD at E
variables (h2 : E = int AC BD)
-- The area of triangles ABE and ADE
variables (h3 : area A B E = 40)
variables (h4 : area A D E = 25)

-- The goal to prove the area of trapezoid ABCD is 115
theorem area_trapezoid_ABCD : 
  area A B C + area C D A + area B C E + area C D E = 115 :=
sorry

end area_trapezoid_ABCD_l491_491505


namespace fraction_of_juniors_is_correct_l491_491503

-- Definitions for the conditions
def students_total : ℕ := 120
def freshman (F : ℕ) : Prop := ∃ S : ℕ, F = 2 * S
def junior (J : ℕ) : Prop := ∃ N : ℕ, J = 4 * N
def equation (F S J N : ℕ) : Prop := (1/2 : ℚ) * F + (1/3 : ℚ) * S = (2/3 : ℚ) * J - (1/4 : ℚ) * N
def student_sum (F S J N : ℕ) : Prop := F + S + J + N = students_total
def at_least_one_each (F S J N : ℕ) : Prop := F > 0 ∧ S > 0 ∧ J > 0 ∧ N > 0

-- Given conditions
variables (F S J N : ℕ)
axioms (h1 : freshman F)
       (h2 : junior J)
       (h3 : equation F S J N)
       (h4 : student_sum F S J N)
       (h5 : at_least_one_each F S J N)

-- Theorem proof stub
theorem fraction_of_juniors_is_correct : (J : ℚ) / students_total = 32 / 167 :=
by 
  sorry

end fraction_of_juniors_is_correct_l491_491503


namespace triangle_equilateral_l491_491118

noncomputable def point := (ℝ × ℝ)

noncomputable def D : point := (0, 0)
noncomputable def E : point := (2, 0)
noncomputable def F : point := (1, Real.sqrt 3)

noncomputable def dist (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def D' (l : ℝ) : point :=
  let ED := dist E D
  (D.1 + l * ED * (Real.sqrt 3), D.2 + l * ED)

noncomputable def E' (l : ℝ) : point :=
  let DF := dist D F
  (E.1 + l * DF * (Real.sqrt 3), E.2 + l * DF)

noncomputable def F' (l : ℝ) : point :=
  let DE := dist D E
  (F.1 - 2 * l * DE, F.2 + (Real.sqrt 3 - l * DE))

theorem triangle_equilateral (l : ℝ) (h : l = 1 / Real.sqrt 3) :
  let DD' := dist D (D' l)
  let EE' := dist E (E' l)
  let FF' := dist F (F' l)
  dist (D' l) (E' l) = dist (E' l) (F' l) ∧ dist (E' l) (F' l) = dist (F' l) (D' l) ∧ dist (F' l) (D' l) = dist (D' l) (E' l) := sorry

end triangle_equilateral_l491_491118


namespace shortest_third_side_l491_491687

theorem shortest_third_side (a b : ℝ) (h_a : a = 5) (h_b : b = 12) :
  (∃ c : ℝ, c = real.sqrt (b^2 - a^2) ∧ c ∈ set.Icc 0 b) :=
begin
  use real.sqrt (12^2 - 5^2),
  split; norm_num,
  { simp [real.sqrt_nonneg], },
end

end shortest_third_side_l491_491687


namespace probability_sqrt_floor_correct_l491_491539

noncomputable def probability_sqrt_floor (x : ℝ) : Prop :=
  (169 ≤ x ∧ x < 196) → 
  (150 ≤ x ∧ x < 250) ∧ 
  (⌊real.sqrt x⌋ = 13) ∧ 
  (⌊real.sqrt (200 * x)⌋ = 190) → 
  ∃ (p : ℝ), p = 19 / 300

theorem probability_sqrt_floor_correct : 
  probability_sqrt_floor := 
begin
  sorry
end

end probability_sqrt_floor_correct_l491_491539


namespace sequence_formula_and_inequality_l491_491548

theorem sequence_formula_and_inequality :
  (∀ (S : ℕ → ℕ) (a : ℕ → ℕ), 
    a 1 = 2 ∧ 
    a 2 = 8 ∧
    (∀ n ≥ 2, S (n + 1) + 4 * S (n - 1) = 5 * S n) →
    (∀ n, a n = 2^(2*n-1)) ∧
    (let T : ℕ → ℤ := λ n, ∑ i in finset.range n, log 2 (a (i + 1))
     in ∀ n, n ≤ 504 → (∏ i in finset.range (n - 1), (1 - (1 / T (i + 2))) ≥ 1011 / 2018)) :=
begin
  sorry
end

end sequence_formula_and_inequality_l491_491548


namespace log_b5_b9_eq_4_l491_491865

noncomputable theory

variables {a_n : ℕ → ℝ} {b_n : ℕ → ℝ}

-- given conditions
def arithmetic_sequence := ∃ d : ℝ, (∀ n, a_n (n + 1) = a_n n + d) ∧ d ≠ 0
def condition := 2 * a_n 5 - a_n 7 ^ 2 + 2 * a_n 9 = 0
def geometric_sequence := ∃ r : ℝ, (∀ n, b_n (n + 1) = b_n n * r) ∧ b_n 7 = a_n 7

-- the theorem to prove
theorem log_b5_b9_eq_4 
  (h_arith : arithmetic_sequence)
  (h_cond : condition)
  (h_geom : geometric_sequence) :
  ∃ (log_val : ℝ), log_val = 4 :=
begin
  sorry
end

end log_b5_b9_eq_4_l491_491865


namespace seating_order_initial_l491_491237

theorem seating_order_initial (
  Sharik_right : ∃ order : List String, order.last = "Sharik",
  Matroskin_flea_on_left : 
    ∀ order : List String, 
    (order = ["Matroskin", "Dyadya Fyodor", "Sharik"] 
    → order.head = "Matroskin")) :
    ∃ order : List String, order = ["Matroskin", "Dyadya Fyodor", "Pechkin", "Sharik"] := 
sorry

end seating_order_initial_l491_491237


namespace lolita_milk_per_week_l491_491553

def weekday_milk : ℕ := 3
def saturday_milk : ℕ := 2 * weekday_milk
def sunday_milk : ℕ := 3 * weekday_milk
def total_milk_week : ℕ := 5 * weekday_milk + saturday_milk + sunday_milk

theorem lolita_milk_per_week : total_milk_week = 30 := 
by 
  sorry

end lolita_milk_per_week_l491_491553


namespace min_distance_from_curve_to_focus_l491_491203

noncomputable def minDistanceToFocus (x y θ : ℝ) : ℝ :=
  let a := 3
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  a - c

theorem min_distance_from_curve_to_focus :
  ∀ θ : ℝ, minDistanceToFocus (2 * Real.cos θ) (3 * Real.sin θ) θ = 3 - Real.sqrt 5 :=
by
  sorry

end min_distance_from_curve_to_focus_l491_491203


namespace domain_of_f_l491_491244

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : 
  {x : ℝ | ¬ ((x - 3) + (x - 9) = 0)} = 
  {x : ℝ | x ≠ 6} := 
by
  sorry

end domain_of_f_l491_491244


namespace Jan_more_miles_than_Ian_l491_491268

-- Definitions
variables {d s t m : ℝ}
variable (n : ℝ)

-- Conditions
def Ian_distance := d = s * t
def Han_distance := d + 120 = (s + 10) * (t + 2)
def Jan_distance := m = (s + 15) * (t + 3)

-- Proof goal
theorem Jan_more_miles_than_Ian :
  Ian_distance →
  Han_distance →
  Jan_distance →
  (2 * s + 20 * t = 100) →
  n = m - d →
  n = 145 :=
by
  intros Ian_dist Han_dist Jan_dist eq4 def_n
  sorry

end Jan_more_miles_than_Ian_l491_491268


namespace range_of_independent_variable_of_sqrt_l491_491628

theorem range_of_independent_variable_of_sqrt (x : ℝ) : (2 * x - 3 ≥ 0) ↔ (x ≥ 3 / 2) := sorry

end range_of_independent_variable_of_sqrt_l491_491628


namespace average_of_x_l491_491069

theorem average_of_x (x : ℝ) (h : sqrt (3 * x^2 + 4) = sqrt 28) :
  ((2 * sqrt 2 + (-2 * sqrt 2)) / 2) = 0 :=
by
  sorry

end average_of_x_l491_491069


namespace fill_cistern_time_l491_491234

theorem fill_cistern_time (A B C : ℕ) (hA : A = 10) (hB : B = 12) (hC : C = 50) :
    1 / (1 / A + 1 / B - 1 / C) = 300 / 49 :=
by
  sorry

end fill_cistern_time_l491_491234


namespace evaluate_expression_l491_491396

theorem evaluate_expression (x : ℕ) (hx : x = 3) :
  x + x * (x^x) + (x^(x^x)) = 7,625,597,485,071 :=
by
  rw [hx]
  norm_num
  sorry

end evaluate_expression_l491_491396


namespace books_read_last_month_l491_491649

namespace BookReading

variable (W : ℕ) -- Number of books William read last month.

-- Conditions
axiom cond1 : ∃ B : ℕ, B = 3 * W -- Brad read thrice as many books as William did last month.
axiom cond2 : W = 2 * 8 -- This month, William read twice as much as Brad, who read 8 books.
axiom cond3 : ∃ (B_prev : ℕ) (B_curr : ℕ), B_prev = 3 * W ∧ B_curr = 8 ∧ W + 16 = B_prev + B_curr + 4 -- Total books equation

theorem books_read_last_month : W = 2 := by
  sorry

end BookReading

end books_read_last_month_l491_491649


namespace initial_avg_mark_l491_491595

variable (A : ℝ) -- The initial average mark

-- Conditions
def num_students : ℕ := 33
def avg_excluded_students : ℝ := 40
def num_excluded_students : ℕ := 3
def avg_remaining_students : ℝ := 95

-- Equation derived from the problem conditions
def initial_avg :=
  A * num_students - avg_excluded_students * num_excluded_students = avg_remaining_students * (num_students - num_excluded_students)

theorem initial_avg_mark :
  initial_avg A →
  A = 90 :=
by
  intro h
  sorry

end initial_avg_mark_l491_491595


namespace find_percentage_l491_491845

theorem find_percentage : 
  ∀ (P : ℕ), 
  (50 - 47 = (P / 100) * 15) →
  P = 20 := 
by
  intro P h
  sorry

end find_percentage_l491_491845


namespace normal_distribution_probability_l491_491081

noncomputable def X : MeasureTheory.ProbabilityMeasure ℝ := sorry

theorem normal_distribution_probability (σ : ℝ) (hσ : σ > 0) 
  (h : MeasureTheory.Measure.proba (λ X : ℝ, X > 80) ∧ X < 120 = 0.8) :
  MeasureTheory.Measure.proba (λ X : ℝ, X > 0) ∧ X < 80 = 0.1 :=
sorry

end normal_distribution_probability_l491_491081


namespace maximum_value_of_f_values_of_x_when_f_max_l491_491404

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x + π / 6) - 2 * cos x

theorem maximum_value_of_f :
  ∀ x : ℝ, f x ≤ 2 :=
sorry

theorem values_of_x_when_f_max :
  ∃ k : ℤ, ∀ x : ℝ, f x = 2 ↔ x = 2 * k * π + 2 * π / 3 :=
sorry

end maximum_value_of_f_values_of_x_when_f_max_l491_491404


namespace measure_Z_is_19_6_l491_491099

def measure_angle_X : ℝ := 72
def measure_Y (measure_Z : ℝ) : ℝ := 4 * measure_Z + 10
def angle_sum_condition (measure_Z : ℝ) : Prop :=
  measure_angle_X + (measure_Y measure_Z) + measure_Z = 180

theorem measure_Z_is_19_6 :
  ∃ measure_Z : ℝ, measure_Z = 19.6 ∧ angle_sum_condition measure_Z :=
by
  sorry

end measure_Z_is_19_6_l491_491099


namespace find_2theta_plus_phi_l491_491439

variable (θ φ : ℝ)
variable (hθ : 0 < θ ∧ θ < π / 2)
variable (hφ : 0 < φ ∧ φ < π / 2)
variable (tan_hθ : Real.tan θ = 2 / 5)
variable (cos_hφ : Real.cos φ = 1 / 2)

theorem find_2theta_plus_phi : 2 * θ + φ = π / 4 := by
  sorry

end find_2theta_plus_phi_l491_491439


namespace total_pears_picked_l491_491561

theorem total_pears_picked :
  let mike_pears := 8
  let jason_pears := 7
  let fred_apples := 6
  -- The total number of pears picked is the sum of Mike's and Jason's pears.
  mike_pears + jason_pears = 15 :=
by {
  sorry
}

end total_pears_picked_l491_491561


namespace min_value_exp_sum_eq_4sqrt2_l491_491182

theorem min_value_exp_sum_eq_4sqrt2 {a b : ℝ} (h : a + b = 3) : 2^a + 2^b ≥ 4 * Real.sqrt 2 :=
by
  sorry

end min_value_exp_sum_eq_4sqrt2_l491_491182


namespace q_true_given_not_p_and_p_or_q_l491_491491

theorem q_true_given_not_p_and_p_or_q (p q : Prop) (hnp : ¬p) (hpq : p ∨ q) : q :=
by
  sorry

end q_true_given_not_p_and_p_or_q_l491_491491


namespace limit_log_sqrt_l491_491707

-- Define the function we are interested in
def f (x : ℝ) : ℝ := (Real.log (x^2 + 1)) / (1 - Real.sqrt (x^2 + 1))

-- Statement of the theorem
theorem limit_log_sqrt : 
  filter.tendsto f (nhds 0) (nhds (-2)) := 
sorry

end limit_log_sqrt_l491_491707


namespace proof_x_squared_minus_y_squared_l491_491473

theorem proof_x_squared_minus_y_squared (x y : ℚ) (h1 : x + y = 9 / 14) (h2 : x - y = 3 / 14) :
  x^2 - y^2 = 27 / 196 := by
  sorry

end proof_x_squared_minus_y_squared_l491_491473


namespace average_growth_rate_inequality_l491_491085

theorem average_growth_rate_inequality (p q x : ℝ) (h₁ : (1+x)^2 = (1+p)*(1+q)) (h₂ : p ≠ q) :
  x < (p + q) / 2 :=
sorry

end average_growth_rate_inequality_l491_491085


namespace probability_of_winning_set_l491_491965

def winning_probability : ℚ :=
  let total_cards := 9
  let total_draws := 3
  let same_color_sets := 3
  let same_letter_sets := 3
  let total_ways_to_draw := Nat.choose total_cards total_draws
  let total_favorable_outcomes := same_color_sets + same_letter_sets
  let probability := total_favorable_outcomes / total_ways_to_draw
  probability

theorem probability_of_winning_set :
  winning_probability = 1 / 14 :=
by
  sorry

end probability_of_winning_set_l491_491965


namespace nursing_home_milk_l491_491388

theorem nursing_home_milk :
  ∃ x y : ℕ, (2 * x + 16 = y) ∧ (4 * x - 12 = y) ∧ (x = 14) ∧ (y = 44) :=
by
  sorry

end nursing_home_milk_l491_491388


namespace min_dot_product_value_l491_491054

open Real EuclideanSpace

noncomputable def min_value (a b c : EuclideanSpace ℝ (Fin 3)) (h1 : ‖a‖ = 2) (h2 : ‖b‖ = sqrt 2) (h3 : inner a b = -2) : ℝ :=
  (inner (a - c) (b - c))

theorem min_dot_product_value {a b c : EuclideanSpace ℝ (Fin 3)} (h1 : ‖a‖ = 2) (h2 : ‖b‖ = sqrt 2) (h3 : inner a b = -2) :
  ∃ c : EuclideanSpace ℝ (Fin 3), min_value a b c h1 h2 h3 = -5/2 :=
sorry

end min_dot_product_value_l491_491054


namespace sum_of_primes_l491_491411

theorem sum_of_primes (p q : ℕ) (h1 : p.prime) (h2 : q.prime) (h3 : ∃ k : ℕ, p^2 + p * q + q^2 = k^2) :
  p = 3 ∨ p = 5 → p + q = 8 := by
  sorry

end sum_of_primes_l491_491411


namespace smallest_positive_solution_of_equation_l491_491646

theorem smallest_positive_solution_of_equation :
  ∃ x : ℝ, x > 0 ∧ (x^4 - 40 * x^2 + 400 = 0) ∧ (∀ y : ℝ, y > 0 ∧ (y^4 - 40 * y^2 + 400 = 0) → x ≤ y) ∧ x = 2 * real.sqrt 5 := 
by
  sorry

end smallest_positive_solution_of_equation_l491_491646


namespace find_value_of_a_l491_491807

theorem find_value_of_a (x a : ℝ) (h : 2 * x - a + 5 = 0) (h_x : x = -2) : a = 1 :=
by
  sorry

end find_value_of_a_l491_491807


namespace theorem_QRP_l491_491521

noncomputable def angle_QRP (PQ PR QR: ℝ) (angle_RPS: ℝ) (angle_QRP: ℝ) : Prop :=
  PQ = PR ∧ ∃ S, QS = PS ∧ angle_RPS = 75 ∧ angle_QRP = 35

theorem theorem_QRP (PQ PR QR: ℝ) (QS PS: ℝ) (angle_RPS: ℝ) :
  PQ = PR ∧ QS = PS ∧ angle_RPS = 75 → ∃ angle_QRP, angle_QRP = 35 :=
by
  intro h
  use 35
  sorry

end theorem_QRP_l491_491521


namespace socks_worn_l491_491963

theorem socks_worn (h1 : 3 = 3)
                   (h2 : ∀ s₁ s₂ : ℕ, s₁ ≠ s₂ → (∃ p, p.savored (s₁, s₂)))
                   (h3 : ∃ l, l = 6):
  ∃ n, n = 3 := 
sorry

end socks_worn_l491_491963


namespace collinear_iff_lambda1_lambda2_eq_one_l491_491797

variables {V : Type*} [InnerProductSpace ℝ V] -- Assume V is an ℝ-vector space with inner product
variables (a b : V) (λ1 λ2 : ℝ)
variables (A B C : V)
hypothesis non_parallel : ¬ (∃ k : ℝ, a = k • b)
hypothesis AB_eq : B - A = λ1 • a + b
hypothesis AC_eq : C - A = a + λ2 • b

theorem collinear_iff_lambda1_lambda2_eq_one :
  ∀ A B C : V, (Collinear ℝ ({A, B, C} : Set V)) ↔ (λ1 * λ2 = 1) :=
sorry

end collinear_iff_lambda1_lambda2_eq_one_l491_491797


namespace application_form_choices_l491_491305

theorem application_form_choices (majors : Finset ℕ) (A : ℕ) (h_majors : majors.card = 7) (h_A : A ∈ majors) :
  let available_majors := majors.erase A in
  let ways_first_two := Finset.choose available_majors 2 in
  let ways_next_three := Finset.choose majors.erase (Finset.erase available_majors) 3 in
  let total_ways := ways_first_two.card * ways_next_three.card in
  total_ways = 150 :=
by
  sorry

end application_form_choices_l491_491305


namespace negative_fraction_comparison_l491_491368

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l491_491368


namespace negative_fraction_comparison_l491_491370

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l491_491370


namespace external_tangents_collinear_l491_491828

-- Define three pairwise disjoint circles with centers O1, O2, and O3
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

variable (C1 C2 C3 : Circle)

axiom disjoint (C1 C2 C3 : Circle) : (∀ (O1 O2 : ℝ × ℝ), O1 ≠ O2 ∧ O1 ∉ set.circle O2 C2.radius ∧ O2 ∉ set.circle O1 C1.radius)

-- Definitions of intersection points of external tangents of these circles
axiom T12 : ℝ × ℝ
axiom T23 : ℝ × ℝ
axiom T31 : ℝ × ℝ

-- The proof goal to show that T12, T23, and T31 are collinear
theorem external_tangents_collinear (C1 C2 C3 : Circle) (h : disjoint C1 C2 C3) :
  ∃ (l : ℝ × ℝ → Prop), l T12 ∧ l T23 ∧ l T31 :=
sorry

end external_tangents_collinear_l491_491828


namespace exists_a_cond1_exists_a_cond2_exists_a_cond3_l491_491339

-- Defining the sets A and B
def set_A (a : ℤ) := {1, 3, a^2 + 3a - 4}
def set_B (a : ℤ) := {0, 6, a^2 + 4a - 2, a + 3}

-- Condition 1: A ∩ B = {3}
theorem exists_a_cond1 : ∃ a : ℤ, {3} = set_A a ∩ set_B a := by
  sorry

-- Condition 2: A ∩ B = {6}
theorem exists_a_cond2 : ∃ a : ℤ, {6} = set_A a ∩ set_B a := by
  sorry

-- Condition 3: A ∩ B = {3, 6}
theorem exists_a_cond3 : ∃ a : ℤ, {3, 6} = set_A a ∩ set_B a := by
  sorry

end exists_a_cond1_exists_a_cond2_exists_a_cond3_l491_491339


namespace find_integer_solutions_l491_491753

theorem find_integer_solutions (k : ℕ) (hk : k > 1) : 
  ∃ x y : ℤ, y^k = x^2 + x ↔ (k = 2 ∧ (x = 0 ∨ x = -1)) ∨ (k > 2 ∧ y^k ≠ x^2 + x) :=
by
  sorry

end find_integer_solutions_l491_491753


namespace train_crossing_time_l491_491100

variable (length_of_train : Real) (speed_in_kmph : Real)
variable (speed_in_mps : Real) (time_to_cross : Real)

# The length of the train is 90 meters.
axiom length_of_train_axiom : length_of_train = 90

# The speed of the train is 72 km/hr.
axiom speed_in_kmph_axiom : speed_in_kmph = 72

# Convert the speed from km/hr to m/s using the conversion factor.
def convert_speed (speed_kmph : Real) : Real :=
  speed_kmph * (1000 / 3600)

axiom speed_in_mps_def : speed_in_mps = convert_speed speed_in_kmph

# The time it takes for the train to cross the electric pole.
def crossing_time (length : Real) (speed : Real) : Real :=
  length / speed

axiom time_to_cross_def : time_to_cross = crossing_time length_of_train speed_in_mps

# Prove that the time to cross the electric pole is 4.5 seconds.
theorem train_crossing_time : time_to_cross = 4.5 :=
by
  -- conditions brought in as assumptions
  have h1 : length_of_train = 90 := length_of_train_axiom
  have h2 : speed_in_kmph = 72 := speed_in_kmph_axiom
  have h3 : speed_in_mps = 20 := by
    rw [←speed_in_kmph_axiom, speed_in_mps_def]
    norm_num [convert_speed]

  -- combining the conditions
  show time_to_cross = 4.5
  rw [time_to_cross_def, h3, h1]
  norm_num [crossing_time]
sory

end train_crossing_time_l491_491100


namespace campaign_meaning_l491_491988

-- Define a function that gives the meaning of "campaign" as a noun
def meaning_of_campaign_noun : String :=
  "campaign, activity"

-- The theorem asserts that the meaning of "campaign" as a noun is "campaign, activity"
theorem campaign_meaning : meaning_of_campaign_noun = "campaign, activity" :=
by
  -- We add sorry here because we are not required to provide the proof
  sorry

end campaign_meaning_l491_491988


namespace farm_entrance_fee_for_students_is_five_l491_491527

theorem farm_entrance_fee_for_students_is_five
  (students : ℕ) (adults : ℕ) (adult_fee : ℕ) (total_cost : ℕ) (student_fee : ℕ)
  (h_students : students = 35)
  (h_adults : adults = 4)
  (h_adult_fee : adult_fee = 6)
  (h_total_cost : total_cost = 199)
  (h_equation : students * student_fee + adults * adult_fee = total_cost) :
  student_fee = 5 :=
by
  sorry

end farm_entrance_fee_for_students_is_five_l491_491527


namespace planes_intersect_and_parallel_to_l_l491_491029

-- Definitions and conditions
variable (m n l : Line) (α β : Plane)
variable (h_skew : skew m n) -- m and n are skew lines
variable (h_m_perp_alpha : m ⟂ α) -- m ⟂ α
variable (h_n_perp_beta : n ⟂ β) -- n ⟂ β
variable (h_l_perp_m : l ⟂ m) -- l ⟂ m
variable (h_l_perp_n : l ⟂ n) -- l ⟂ n
variable (h_l_not_in_alpha : ¬ l ⊆ α) -- l ⊆ α
variable (h_l_not_in_beta : ¬ l ⊆ β) -- l ⊆ β

-- Conclude their relationship
theorem planes_intersect_and_parallel_to_l :
  (∃ (p : Line), intersect α β p) ∧ 
  (∀ p, intersect α β p → p ∥ l) := 
sorry

end planes_intersect_and_parallel_to_l_l491_491029


namespace smallest_k_base_representation_l491_491716

theorem smallest_k_base_representation :
  ∃ k : ℕ, (k > 0) ∧ (∀ n k, 0 = (42 * (1 - k^(n+1))/(1 - k))) ∧ (0 = (4 * (53 * (1 - k^(n+1))/(1 - k)))) →
  (k = 11) := sorry

end smallest_k_base_representation_l491_491716


namespace hundredth_term_non_perfect_square_sequence_l491_491015

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def non_perfect_square_seq : ℕ → ℕ
| n := if is_perfect_square n then non_perfect_square_seq (n + 1) else n

theorem hundredth_term_non_perfect_square_sequence :
  non_perfect_square_seq 100 = 110 :=
sorry

end hundredth_term_non_perfect_square_sequence_l491_491015


namespace integer_part_sum_l491_491012

noncomputable def a_seq : ℕ → ℚ
| 0       := 3 / 4
| (n + 1) := 1 + (a_seq n) * ((a_seq n) - 1)

theorem integer_part_sum (s : ℕ → ℚ) (h₀ : s 0 = 3 / 4)
    (h₁ : ∀ n : ℕ, s (n + 1) = 1 + (s n) * ((s n) - 1)) :
  let sum := ∑ i in finRange 2017, 1 / s i
  in ⌊sum⌋ = 2 := sorry

end integer_part_sum_l491_491012


namespace f_monotonicity_on_neg_f_max_min_on_interval_l491_491818

def f (x : ℝ) : ℝ := 1 / (1 + x^2)

-- Monotonicity proof
theorem f_monotonicity_on_neg :
  ∀ (x y : ℝ), x < y ∧ y < 0 → f x < f y := by
  sorry

-- Maximum and minimum values proof
theorem f_max_min_on_interval :
  ∃ (x_min x_max : ℝ), x_min = -3 ∧ x_max = -1 ∧ 
  (∀ x ∈ (set.interval (-3 : ℝ) -1), f x_min ≤ f x ∧ f x ≤ f x_max) ∧
  f x_min = 1 / 10 ∧ f x_max = 1 / 2 := by
  sorry

end f_monotonicity_on_neg_f_max_min_on_interval_l491_491818


namespace sum_first_10_terms_log_seq_l491_491515

noncomputable def geometric_seq_sum_log : ℕ → ℝ
| 3 := 5
| 8 := 2
| _ := sorry -- Define the rest of the sequence explicitly if needed

theorem sum_first_10_terms_log_seq :
  let a: ℕ → ℝ := geometric_seq_sum_log in
  a 3 = 5 ∧ a 8 = 2 →
  (∑ i in finset.range 10, real.log (a (i + 1))) = 5 :=
by sorry

end sum_first_10_terms_log_seq_l491_491515


namespace compare_neg_fractions_l491_491360

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l491_491360


namespace polygons_sides_l491_491953

def sum_of_angles (x y : ℕ) : ℕ :=
(x - 2) * 180 + (y - 2) * 180

def num_diagonals (x y : ℕ) : ℕ :=
x * (x - 3) / 2 + y * (y - 3) / 2

theorem polygons_sides (x y : ℕ) (hx : x * (x - 3) / 2 + y * (y - 3) / 2 - (x + y) = 99) 
(hs : sum_of_angles x y = 21 * (x + y + num_diagonals x y) - 39) :
x = 17 ∧ y = 3 ∨ x = 3 ∧ y = 17 :=
by
  sorry

end polygons_sides_l491_491953


namespace correct_substitution_l491_491985

variables (x y : ℝ)

theorem correct_substitution :
  (2 * x - y = 5) ∧ (y = 1 + x) → (2 * x - (1 + x) = 5) :=
by
  intros h,
  cases h with h1 h2,
  rw h2,
  sorry

end correct_substitution_l491_491985


namespace queen_jack_likes_5_card_hands_l491_491586

theorem queen_jack_likes_5_card_hands : 
  let qj_cards := 8  -- 4 queens + 4 jacks
  ∃ num_hands : ℕ, num_hands = Nat.choose qj_cards 5 ∧ num_hands = 56 := 
by
  use Nat.choose 8 5
  dsimp [Nat.choose]
  have factorial_8 := Nat.factorial 8
  have factorial_5 := Nat.factorial 5
  have factorial_3 := Nat.factorial 3
  calc
    Nat.choose 8 5 = Nat.factorial 8 / (Nat.factorial 5 * Nat.factorial 3) := sorry
    ... = 56 := sorry

end queen_jack_likes_5_card_hands_l491_491586


namespace eccentricity_of_hyperbola_l491_491191

def hyperbola_eccentricity (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2) / a

theorem eccentricity_of_hyperbola : hyperbola_eccentricity 1 1 = real.sqrt 2 := by
  sorry

end eccentricity_of_hyperbola_l491_491191


namespace matrix_sequence_product_l491_491335

theorem matrix_sequence_product :
  (List.foldl (λ acc m, acc ⬝ m) (1 : Matrix (Fin 2) (Fin 2) ℤ)
    (List.map (λ k, Matrix.of ![![1, 2 * k], ![0, 1]]) (List.range 50).map (λ n, n + 1))) =
  (Matrix.of ![![1, 2550], ![0, 1]]) :=
by
  sorry

end matrix_sequence_product_l491_491335


namespace area_of_triangle_PF1F2_is_24_l491_491438

noncomputable def area_triangle : ℝ :=
  let F1 := (10 / 2, 0) -- Assume generic coordinates based on the given conditions
  let F2 := (-10 / 2, 0)
  let P := (7, 0) -- Assume P is a specific point satisfying the condition
in
  1 / 2 * 8 * 6 -- |PF_1| = 8, |PF_2| = 6

theorem area_of_triangle_PF1F2_is_24 :
  area_triangle = 24 := 
  sorry

end area_of_triangle_PF1F2_is_24_l491_491438


namespace garden_area_increase_l491_491999

theorem garden_area_increase
  (length_of_rect : ℝ) (width_of_rect : ℝ)
  (fencing : ℝ)
  (perimeter : length_of_rect + width_of_rect + length_of_rect + width_of_rect = fencing)
  (side_length_of_square : side_length_of_square = fencing / 4) :
  let area_rect := length_of_rect * width_of_rect in
  let area_square := side_length_of_square * side_length_of_square in
  area_square - area_rect = 506.25 :=
by
  let length_of_rect := 60
  let width_of_rect := 15
  let fencing := 150
  have perimeter_proof : 60 + 15 + 60 + 15 = 150 := by sorry
  have side_length_square_proof : 150 / 4 = 37.5 := by sorry
  let area_rect := length_of_rect * width_of_rect
  let area_square := (150/4) * (150/4)
  show area_square - area_rect = 506.25 from sorry

end garden_area_increase_l491_491999


namespace hyperbola_eccentricity_l491_491442

theorem hyperbola_eccentricity
  (C : ℝ → ℝ → Prop)
  (h_asymptotes : ∀ x y, C x y → y = 2 * x ∨ y = -2 * x)
  (h_through_point : C 1 3) :
  ∃ e : ℝ, e = sqrt 5 / 2 := by
  sorry

end hyperbola_eccentricity_l491_491442


namespace maximum_number_of_teams_l491_491883

theorem maximum_number_of_teams (c : ℕ) (h_c_even : c % 2 = 0) (h_c_ge_4 : c ≥ 4)
    (h_conditions :
      ∀ (teams : list (ℕ × ℕ × ℕ)),
      (∀ x, x ∈ teams → x.2.1 ≠ x.2.2) ∧     -- each team has a home uniform with two different colors
      (∀ x, x ∈ teams → x.2.3 ≠ x.2.1 ∧ x.2.3 ≠ x.2.2) ∧   -- away uniform's color different from home uniform's colors
      (∀ x ∈ teams, ∀ y ∈ teams, x.2.1 = y.2.1 → x.2.2 = y.2.2 → x.2.3 ≠ y.2.3) ∧   -- same home uniform with different away uniform
      list.length teams ≤ c ∧    -- at most c distinct colors
      (∀ x ∈ teams, ∀ y ∈ teams, x ≠ y →
        (¬ (x.2.1 = y.2.1 ∨ x.2.1 = y.2.2 ∨ x.2.2 = y.2.1 ∨ x.2.2 = y.2.2) ∨   -- no team clashes with both uniforms of another team
         ¬ (y.2.1 = x.2.3 ∨ y.2.2 = x.2.3)))
    ) :
  list.length teams ≤ c * nat.floor (c ^ 2 / 4) :=
by sorry

end maximum_number_of_teams_l491_491883


namespace cube_of_99999_is_correct_l491_491888

theorem cube_of_99999_is_correct : (99999 : ℕ)^3 = 999970000299999 :=
by
  sorry

end cube_of_99999_is_correct_l491_491888


namespace find_a_value_l491_491477

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end find_a_value_l491_491477


namespace problem_1_part1_problem_1_part2_problem_1_part3_l491_491490

-- Part (1): Define and prove non-self-adjoint property
def isTwoOrderSelfAdjoint (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁ ∈ D, ∃ x₂ ∈ D, f x₁ * f x₂ = 2 ∧ ∀ x₂' ∈ D, f x₁ * f x₂' = 2 → x₂ = x₂'

theorem problem_1_part1 : ¬isTwoOrderSelfAdjoint (λ x, log 2 (x^2 + 1)) (Icc 0 (sqrt 7)) :=
by
  sorry

-- Part (2): Define and solve for b
def isOneOrderSelfAdjoint (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁ ∈ D, ∃ x₂ ∈ D, f x₁ * f x₂ = 1 ∧ ∀ x₂' ∈ D, f x₁ * f x₂' = 1 → x₂ = x₂'

theorem problem_1_part2 (b : ℝ) : isOneOrderSelfAdjoint (λ x, 4^(x - 1)) (Icc (1/2) b) ↔ b = 3/2 :=
by 
  sorry

-- Part (3): Define and determine range of a
def isTwoOrderAdjoint (f g : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁ ∈ D, ∃ x₂ ∈ D, f x₁ * g x₂ = 2 ∧ ∀ x₂' ∈ D, f x₁ * g x₂' = 2 → x₂ = x₂'

theorem problem_1_part3 (a : ℝ) : isTwoOrderAdjoint (λ x, 4/(x + 2)) (λ x, x^2 - 2*a*x + a^2 - 1) (Icc 0 2) ↔
  a ∈ Set.Union (Icc (-sqrt 2) (2 - sqrt 3)) (Icc (sqrt 3) (2 + sqrt 2)) :=
by 
  sorry

end problem_1_part1_problem_1_part2_problem_1_part3_l491_491490


namespace conjugate_in_third_quadrant_l491_491809

def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

noncomputable def z : ℂ := -1 + complex.i

theorem conjugate_in_third_quadrant : is_in_third_quadrant (conj z) :=
by
  sorry

end conjugate_in_third_quadrant_l491_491809


namespace equivalent_expression_l491_491384

theorem equivalent_expression :
  (2 + 5) * (2^2 + 5^2) * (2^4 + 5^4) * (2^8 + 5^8) * (2^16 + 5^16) * (2^32 + 5^32) * (2^64 + 5^64) =
  5^128 - 2^128 := by
  sorry

end equivalent_expression_l491_491384


namespace sum_sin_10k_eq_tan_85_l491_491802

theorem sum_sin_10k_eq_tan_85 :
  (∃ (p q : ℕ), (p.gcd q = 1 ∧ p / q < 90 ∧ (∑ k in finset.range 18, real.sin (10 * (k + 1) * real.pi / 180)) = real.tan (p * real.pi / q / 180)) ∧ p + q = 86) :=
by
  have angle_eq : 10 * real.pi / 180 = real.of_rat(1/18);
  rw [← angle_eq];
  existsi 85;
  existsi 1;
  have gcd_pq : nat.gcd 85 1 = 1 := nat.gcd_rec 85 1;
  have p_lt_90 : 85 / 1 < 90 := begin norm_num end;
  have sum_eq_tan : (∑ k in finset.range 18, real.sin (10 * (k + 1) * real.pi / 180)) = real.tan (85 * real.pi / 180) := sorry;
  simp [gcd_pq, p_lt_90, sum_eq_tan];
  exact 86;

end sum_sin_10k_eq_tan_85_l491_491802


namespace compare_fractions_l491_491341

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l491_491341


namespace area_of_extended_quadrilateral_l491_491709

theorem area_of_extended_quadrilateral 
  (EF GH HE: Point)
  (F' G' H' E': Point)
  (EF_len FG_len GH_len HE_len: ℝ)
  (EFGH_area: ℝ)
  (hEF : EF.len = 5)
  (hFG : FG.len = 8)
  (hGH : GH.len = 9)
  (hHE : HE.len = 10)
  (hEFG: convex_quadrilateral EF FG GH HE)
  (hEFGH_area : EFGH.area = 15) :
  (extended_quadrilateral E'F'G'H').area = 45 := 
sorry

end area_of_extended_quadrilateral_l491_491709


namespace point_on_line_coordinates_l491_491382

theorem point_on_line_coordinates (x : ℝ) : 
  (∃ x, (λ p: ℝ × ℝ, p = (2, 10) ∨ p = (-2, 4)) x ∧ 
          (λ q: ℝ × ℝ, q = (x, -3)) x ∧ 
          (λ m, m = (10 - 4) / (2 - (-2)) ∧ 
           m = (-3 -10) / (x - 2))) -> 
  x = -20 / 3 :=
by sorry

end point_on_line_coordinates_l491_491382


namespace range_of_a_l491_491459

noncomputable def quadratic (a x : ℝ) : ℝ := a * x^2 + a * x + 1
noncomputable def quadratic_eq (x₀ a : ℝ) : Prop := x₀^2 - x₀ + a = 0

theorem range_of_a (a : ℝ) (p : ∀ x : ℝ, quadratic a x > 0) (q : ∃ x₀ : ℝ, quadratic_eq x₀ a) : 0 ≤ a ∧ a ≤ 1/4 :=
  sorry

end range_of_a_l491_491459


namespace unbounded_function_exists_l491_491528

theorem unbounded_function_exists :
  ∃ (f : ℚ → ℝ), (∀ (h : ℚ) (x₀ : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ (x : ℚ), abs (x - x₀) < δ → abs (f (x + h) - f x) < ε) ∧
  ∀ I : Set ℝ, IsInterval I → ∃ x y ∈ I, ∀ M : ℝ, ∃ z ∈ I, f z > M :=
sorry

end unbounded_function_exists_l491_491528


namespace cost_price_per_meter_correct_l491_491684

-- Define the types and constants
def TypeA : Type := Unit
def TypeB : Type := Unit
def TypeC : Type := Unit

-- Define the conditions
constant total_length : ℕ := 45
constant total_price : ℕ := 4500

constant profit_per_meter_A : ℕ := 14
constant profit_per_meter_B : ℕ := 18
constant profit_per_meter_C : ℕ := 22

constant length_sold_A : ℕ := 15
constant length_sold_B : ℕ := 10
constant length_sold_C : ℕ := 20

-- Calculate selling price per meter
def selling_price_per_meter : ℕ := total_price / total_length

-- Calculate cost price per meter for each type
def cost_price_per_meter_A : ℕ := selling_price_per_meter - profit_per_meter_A
def cost_price_per_meter_B : ℕ := selling_price_per_meter - profit_per_meter_B
def cost_price_per_meter_C : ℕ := selling_price_per_meter - profit_per_meter_C

-- Define the theorem to be proved
theorem cost_price_per_meter_correct :
  cost_price_per_meter_A = 86 ∧
  cost_price_per_meter_B = 82 ∧
  cost_price_per_meter_C = 78 :=
by
  sorry

end cost_price_per_meter_correct_l491_491684


namespace determine_pairs_l491_491381

-- Definitions:
variable (c d : ℝ) (a : ℕ → ℝ)

-- Conditions:
def condition1 := ∀ n ≥ 1, a n > 0
def condition2 := ∀ n ≥ 1, a n ≥ c * a (n + 1) + d * (∑ j in Finset.range (n-1), a j)

-- Lean theorem statement:
theorem determine_pairs (h1 : condition1 a) (h2 : condition2 c d a) :
  (c ≤ 0) ∨ (d ≤ 0) ∨ (0 < c ∧ c < 1 ∧ d ≤ (c - 1)^2 / (4 * c)) :=
begin
  sorry  -- Proof goes here
end

end determine_pairs_l491_491381


namespace log_base_equality_l491_491745

theorem log_base_equality : log 4 / log 16 = 1 / 2 := 
by sorry

end log_base_equality_l491_491745


namespace statement_A_is_false_statement_B_is_false_statement_C_is_false_statement_D_is_true_l491_491053

universe u

-- Definitions for planes and lines
variables (Point : Type u) 
variable [geometry Point]

-- Define distinct planes and lines
variables {α β : Plane Point} (h1 : α ≠ β)
variables {a b c : Line Point} (h2 : a ≠ b) (h3 : b ≠ c) (h4 : a ≠ c)

-- Given conditions for each statement
variables (hA1 : perpendicular a b) (hA2 : perpendicular b c)
variables (hB1 : parallel a α) (hB2 : parallel b α)
variables (hC1 : perpendicular a b) (hC2 : perpendicular a α)
variables (hD1 : perpendicular a α) (hD2 : perpendicular a β)

theorem statement_A_is_false : ¬ (parallel a c) :=
sorry

theorem statement_B_is_false : ¬ (parallel a b) :=
sorry

theorem statement_C_is_false : ¬ (parallel b α) :=
sorry

theorem statement_D_is_true : parallel α β :=
sorry

end statement_A_is_false_statement_B_is_false_statement_C_is_false_statement_D_is_true_l491_491053


namespace ab_value_in_triangle_l491_491096

theorem ab_value_in_triangle (a b c : ℝ) (C : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : C = 60) :
  a * b = 4 / 3 :=
by sorry

end ab_value_in_triangle_l491_491096


namespace function_properties_l491_491266

theorem function_properties (f : ℝ → ℝ) : 
  (∀ x1 x2, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x, 0 < x → (deriv f x) > 0) ∧
  (∀ x, deriv f (-x) = -(deriv f x)) → 
  f = (λ x, x ^ 2)
by
  sorry

end function_properties_l491_491266


namespace garden_roller_area_l491_491297

theorem garden_roller_area (length : ℝ) (area_5rev : ℝ) (d1 d2 : ℝ) (π : ℝ) :
  length = 4 ∧ area_5rev = 88 ∧ π = 22 / 7 ∧ d2 = 1.4 →
  let circumference := π * d2
  let area_rev := circumference * length
  let new_area_5rev := 5 * area_rev
  new_area_5rev = 88 :=
by
  sorry

end garden_roller_area_l491_491297


namespace find_a_l491_491196

theorem find_a (a : ℤ) (h1 : ∃ b c : ℤ, ∀ x : ℤ, y : ℤ, y = a * x^2 + b * x + c ∧ ((x, y) = (2, 5) ∨ (x, y) = (3, 6))) : a = 1 :=
sorry

end find_a_l491_491196


namespace prime_cubed_plus_seven_composite_l491_491061

theorem prime_cubed_plus_seven_composite (P : ℕ) (hP_prime : Nat.Prime P) (hP3_plus_5_prime : Nat.Prime (P ^ 3 + 5)) : ¬ Nat.Prime (P ^ 3 + 7) :=
by
  sorry

end prime_cubed_plus_seven_composite_l491_491061


namespace send_more_money_l491_491871

/-- Mathematical constants representing each letter mapping to a digit -/
variable (S E N D M O R Y : ℕ)

/-- Conditions: all variables are distinct digits, and the addition is correct -/
axiom distinct_digits : S ≠ E ∧ S ≠ N ∧ S ≠ D ∧ S ≠ M ∧ S ≠ O ∧ S ≠ R ∧ S ≠ Y ∧ E ≠ N ∧ E ≠ D ∧ E ≠ M ∧ E ≠ O ∧ E ≠ R ∧ E ≠ Y ∧ N ≠ D ∧ N ≠ M ∧ N ≠ O ∧ N ≠ R ∧ N ≠ Y ∧ D ≠ M ∧ D ≠ O ∧ D ≠ R ∧ D ≠ Y ∧ M ≠ O ∧ M ≠ R ∧ M ≠ Y ∧ O ≠ R ∧ O ≠ Y ∧ R ≠ Y

axiom valid_digits : S < 10 ∧ E < 10 ∧ N < 10 ∧ D < 10 ∧ M < 10 ∧ O < 10 ∧ R < 10 ∧ Y < 10

noncomputable def send := 1000 * S + 100 * E + 10 * N + D
noncomputable def more := 1000 * M + 100 * O + 10 * R + E
noncomputable def money := 10000 * M + 1000 * O + 100 * N + 10 * E + Y

theorem send_more_money (S E N D M O R Y : ℕ) 
  (h_distinct : distinct_digits S E N D M O R Y) 
  (h_valid : valid_digits S E N D M O R Y) :
  send S E N D = 9000 + 600 + 50 + D ∧
  more M O R E = 10000 + 9000 + 700 + E ∧
  money M O N E Y = 19000 + 9000 + 600 + 50 + D + E := 19000 + 9000 + 600 + 50 := 
      sorry

end send_more_money_l491_491871


namespace find_y_in_terms_of_x_l491_491066

theorem find_y_in_terms_of_x (p : ℝ) (x y : ℝ) (h1 : x = 1 + 3^p) (h2 : y = 1 + 3^(-p)) : y = x / (x - 1) :=
by
  sorry

end find_y_in_terms_of_x_l491_491066


namespace problem1_problem2_l491_491700

-- Problem 1: Prove that (x + y + z)² - (x + y - z)² = 4z(x + y) for x, y, z ∈ ℝ
theorem problem1 (x y z : ℝ) : (x + y + z)^2 - (x + y - z)^2 = 4 * z * (x + y) := 
sorry

-- Problem 2: Prove that (a + 2b)² - 2(a + 2b)(a - 2b) + (a - 2b)² = 16b² for a, b ∈ ℝ
theorem problem2 (a b : ℝ) : (a + 2 * b)^2 - 2 * (a + 2 * b) * (a - 2 * b) + (a - 2 * b)^2 = 16 * b^2 := 
sorry

end problem1_problem2_l491_491700


namespace participants_count_l491_491617

theorem participants_count (x y : ℕ) 
    (h1 : y = x + 41)
    (h2 : y = 3 * x - 35) : 
    x = 38 ∧ y = 79 :=
by
  sorry

end participants_count_l491_491617


namespace mappings_satisfy_condition_l491_491549

variable (M N : Type) (f : M → N)

noncomputable def mappings_count (M : Finset ℂ) (N : Finset ℂ) :=
  N.filter_map (λ x, N.filter_map (λ y, N.filter_map (λ z, if x + y + z = 0 then some (x, y, z) else none)))

theorem mappings_satisfy_condition : 
  let M := ({x, y, z} : Finset ℂ)
  let N := ({-1, 0, 1} : Finset ℂ)
  in (mappings_count M N).card = 7 := 
by
  sorry

end mappings_satisfy_condition_l491_491549


namespace natural_numbers_satisfying_conditions_l491_491385

variable (a b : ℕ)

theorem natural_numbers_satisfying_conditions :
  (90 < a + b ∧ a + b < 100) ∧ (0.9 < (a : ℝ) / b ∧ (a : ℝ) / b < 0.91) ↔ (a = 46 ∧ b = 51) ∨ (a = 47 ∧ b = 52) := by
  sorry

end natural_numbers_satisfying_conditions_l491_491385


namespace sin_minus_cos_eq_l491_491062

theorem sin_minus_cos_eq (α : ℝ) (h1 : sin α * cos α = 1 / 8) (h2 : (π / 4) < α ∧ α < (π / 2)) :
  sin α - cos α = sqrt 3 / 2 :=
sorry

end sin_minus_cos_eq_l491_491062


namespace distinct_letters_count_l491_491909

noncomputable def distinct_collections_of_letters : ℕ :=
  let vowels := ['O', 'O', 'I', 'Y']
  let consonants := ['B', 'G', 'L']
  let indistinguishable_vowels := (vowels.erase_dup).length != vowels.length
  let indistinguishable_consonants := (consonants.erase_dup).length != consonants.length
  if indistinguishable_vowels && indistinguishable_consonants then 2 else sorry

theorem distinct_letters_count :
  distinct_collections_of_letters = 2 :=
by
  sorry

end distinct_letters_count_l491_491909


namespace necessary_condition_l491_491917

theorem necessary_condition (x : ℝ) (h : x > 0) : x > -2 :=
by {
  exact lt_trans (by norm_num) h,
}

end necessary_condition_l491_491917


namespace slope_parallel_condition_l491_491940

theorem slope_parallel_condition (k : ℝ) :
  let A := (-4 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, -4 : ℝ)
  let X := (5 : ℝ, 8 : ℝ)
  let Y := (19 : ℝ, k)
  (Y.2 - X.2) / (Y.1 - X.1) = (B.2 - A.2) / (B.1 - A.1) → k = -6 :=
by
  let A := (-4 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, -4 : ℝ)
  let X := (5 : ℝ, 8 : ℝ)
  let m_AB := (B.2 - A.2) / (B.1 - A.1)
  let m_XY := (19 - 5 : ℝ)
  intro h
  calc k
     = 8 + 14 * m_AB : by sorry
  ... = 8 + 14 * -1 : by sorry
  ... = 8 - 14 : by sorry
  ... = -6 : by rfl

end slope_parallel_condition_l491_491940


namespace solve_a_l491_491379

def F (a b c : ℚ) : ℚ := a * b^3 + c

theorem solve_a :
  ∃ (a : ℚ), F(a, 3, 10) = F(a, 5, 20) ∧ a = -5 / 49 :=
by
  use -5 / 49
  dsimp [F]
  simp
  ring
  sorry

end solve_a_l491_491379


namespace number_of_ways_to_express_n_as_sum_l491_491898

noncomputable def P (n k : ℕ) : ℕ := sorry
noncomputable def Q (n k : ℕ) : ℕ := sorry

theorem number_of_ways_to_express_n_as_sum (n : ℕ) (k : ℕ) (h : k ≥ 2) : P n k = Q n k := sorry

end number_of_ways_to_express_n_as_sum_l491_491898


namespace swimmer_distance_downstream_l491_491307

noncomputable def swimmer_problem (c : ℝ) (t : ℝ) (d_upstream : ℝ) (d_downstream : ℝ) : Prop :=
  let v := (d_upstream / t + c) / 2 in
  d_downstream = t * (v + c)

theorem swimmer_distance_downstream :
  swimmer_problem 4.5 5 10 55 :=
by
  unfold swimmer_problem
  sorry

end swimmer_distance_downstream_l491_491307


namespace functional_equation_l491_491135

def is_divisible (x y : ℕ) : Prop := ∃ k : ℕ, x = k * y

theorem functional_equation 
  (f : ℕ → ℕ) 
  (h : ∀ a b : ℕ, 1 ≤ a → 1 ≤ b → is_divisible (a^2 + f(a) * f(b)) (f(a) + b)) :
  ∀ n : ℕ, 1 ≤ n → f(n) = n :=
by
  sorry

end functional_equation_l491_491135


namespace rectangle_area_l491_491199

-- Define the problem conditions
def length : ℝ := 7  -- length in meters
def width_dm : ℝ := 30  -- width in decimeters
def width_m : ℝ := width_dm / 10  -- convert decimeters to meters

-- Define the area calculation
def area (l w : ℝ) : ℝ := l * w

-- State the theorem
theorem rectangle_area : area length width_m = 21 := by
  sorry

end rectangle_area_l491_491199


namespace area_of_quadrilateral_ABCD_l491_491866

-- Definitions for the given conditions
variables (A B C D E : Type) [Point : MeasurableSpace Point]
variables (dist : Point → Point → ℝ) -- Distance function
variables (angle : Point → Point → Point → ℝ) -- Angle function

-- Given conditions
variables (AD AE AB : ℝ)
variables (angle_DAB angle_DCB angle_AEC : ℝ)

-- Quadrilateral ABCD and point E setup
variables (quad : Quadrilateral ABCD)
variables (e_point : IsPoint E) (a_is_point : IsPoint A)
variables (d_line_ab : IsLineSegment AD AB)
variables (angles : ∀ {A B C}, angle A B C = 90)
variables (length_AE : AE = 5)

-- Mathematically equivalent proof problem
theorem area_of_quadrilateral_ABCD :
  dist A E = 5 ∧ ∀ {A D C B}, 
  AD = AB ∧ angle DAB = 90 ∧ angle DCB = 90 ∧ angle AEC = 90 →
  (area quad = 25) :=
begin
  sorry
end

end area_of_quadrilateral_ABCD_l491_491866


namespace DirichletProperties_l491_491939

def DirichletFunction (x : ℝ) : ℝ :=
  if x ∈ ℚ then 1 else 0

def Proposition1 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

def Proposition2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(f(x)) = 0

def Proposition3 (f : ℝ → ℝ) (T : ℚ) : Prop :=
  T ≠ 0 → ∀ x : ℝ, f(x + T) = f(x)

def Proposition4 (f : ℝ → ℝ) : Prop :=
  ¬ ∃ x1 x2 x3 : ℝ, f(x1) = 1 ∧ f(x2) = 1 ∧ f(x3) = 1 ∧
  (x2 - x1 = x3 - x2 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1)

theorem DirichletProperties : (Proposition1 DirichletFunction) ∧ ¬ (Proposition2 DirichletFunction) ∧ (∃ T : ℚ, Proposition3 DirichletFunction T) ∧ ¬ (Proposition4 DirichletFunction) :=
by
  sorry

end DirichletProperties_l491_491939


namespace smallest_m_l491_491115

theorem smallest_m (n : ℕ) (h : n > 2) :
  ∃ m : ℕ, (∀ (f : set (ℕ × ℕ) → ℕ), (∀ (e : ℕ × ℕ), e ∈ (set.univ : set (ℕ × ℕ)) → f e ∈ (set.range (λ x, x + 1 ≤ 3))) → 
  ∀ (i j : ℕ), i ≠ j → (∑ k in (finset.filter (λ x, x.1 = i ∨ x.2 = i) finset.univ).to_finset, f k) ≠ 
                 (∑ k in (finset.filter (λ x, x.1 = j ∨ x.2 = j) finset.univ).to_finset, f k))) :=
by
  use 3
  intro f hf i j hij
  apply distinct_priorities  -- Assuming distinct_priorities is a helper lemma/proof that verifies uniqueness under the conditions.
  sorry

end smallest_m_l491_491115


namespace glass_heavier_than_plastic_l491_491215

-- Define the conditions
def condition1 (G : ℕ) : Prop := 3 * G = 600
def condition2 (G P : ℕ) : Prop := 4 * G + 5 * P = 1050

-- Define the theorem to prove
theorem glass_heavier_than_plastic (G P : ℕ) (h1 : condition1 G) (h2 : condition2 G P) : G - P = 150 :=
by
  sorry

end glass_heavier_than_plastic_l491_491215


namespace dandelion_seed_production_l491_491570

theorem dandelion_seed_production :
  ∀ (initial_seeds : ℕ), initial_seeds = 50 →
  ∀ (germination_rate : ℚ), germination_rate = 1 / 2 →
  ∀ (new_seed_rate : ℕ), new_seed_rate = 50 →
  (initial_seeds * germination_rate * new_seed_rate) = 1250 :=
by
  intros initial_seeds h1 germination_rate h2 new_seed_rate h3
  sorry

end dandelion_seed_production_l491_491570


namespace ratio_lena_kevin_after_5_more_l491_491112

variables (L K N : ℕ)

def lena_initial_candy : ℕ := 16
def lena_gets_more : ℕ := 5
def kevin_candy_less_than_nicole : ℕ := 4
def lena_more_than_nicole : ℕ := 5

theorem ratio_lena_kevin_after_5_more
  (lena_initial : L = lena_initial_candy)
  (lena_to_multiple_of_kevin : L + lena_gets_more = K * 3) 
  (kevin_less_than_nicole : K = N - kevin_candy_less_than_nicole)
  (lena_more_than_nicole_condition : L = N + lena_more_than_nicole) :
  (L + lena_gets_more) / K = 3 :=
sorry

end ratio_lena_kevin_after_5_more_l491_491112


namespace shirt_tie_combinations_l491_491931

noncomputable def shirts : ℕ := 8
noncomputable def ties : ℕ := 7
noncomputable def forbidden_combinations : ℕ := 2

theorem shirt_tie_combinations :
  shirts * ties - forbidden_combinations = 54 := by
  sorry

end shirt_tie_combinations_l491_491931


namespace josephine_total_milk_l491_491161

-- Define the number of containers and the amount of milk they hold
def cnt_1 : ℕ := 3
def qty_1 : ℚ := 2

def cnt_2 : ℕ := 2
def qty_2 : ℚ := 0.75

def cnt_3 : ℕ := 5
def qty_3 : ℚ := 0.5

-- Define the total amount of milk sold
def total_milk_sold : ℚ := cnt_1 * qty_1 + cnt_2 * qty_2 + cnt_3 * qty_3

theorem josephine_total_milk : total_milk_sold = 10 := by
  -- This is the proof placeholder
  sorry

end josephine_total_milk_l491_491161


namespace amy_total_distance_equals_168_l491_491696

def amy_biked_monday := 12

def amy_biked_tuesday (monday: ℕ) := 2 * monday - 3

def amy_biked_other_day (previous_day: ℕ) := previous_day + 2

def total_distance_bike_week := 
  let monday := amy_biked_monday
  let tuesday := amy_biked_tuesday monday
  let wednesday := amy_biked_other_day tuesday
  let thursday := amy_biked_other_day wednesday
  let friday := amy_biked_other_day thursday
  let saturday := amy_biked_other_day friday
  let sunday := amy_biked_other_day saturday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem amy_total_distance_equals_168 : 
  total_distance_bike_week = 168 := by
  sorry

end amy_total_distance_equals_168_l491_491696


namespace eight_digit_integers_count_l491_491834

-- Definition of number of choices for the first digit and the remaining digits
def countFirstDigitChoices : ℕ := 9
def countOtherDigitChoices : ℕ := 10
def numberOfEightDigitIntegers : ℕ := 90_000_000

-- Theorem stating the proof problem
theorem eight_digit_integers_count :
  countFirstDigitChoices * countOtherDigitChoices ^ 7 = numberOfEightDigitIntegers := by
  sorry

end eight_digit_integers_count_l491_491834


namespace incorrect_statement_l491_491122

variables (α β : Set Set)
variables (l m : Set)
variables (perp : Set → Set → Prop)
variables (paral : Set → Set → Prop)
variables (non_overlapping_planes : (Set → Set → Prop))
variables (non_overlapping_lines : (Set → Set → Prop))

variable h1 : non_overlapping_planes α β
variable h2 : non_overlapping_lines l m

def statement_A := ∀ l α β, perp l α → perp l β → paral α β
def statement_B := ∀ l m α, perp l α → perp m α → paral l m
def statement_C := ∀ l α β, perp l α → paral α β → perp l β
def statement_D := ∀ l α β, perp l α → perp β α → paral l β

theorem incorrect_statement :
  ¬ statement_D :=
  sorry

end incorrect_statement_l491_491122


namespace expression_evaluates_to_2023_l491_491256

theorem expression_evaluates_to_2023 : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 :=
by 
  sorry

end expression_evaluates_to_2023_l491_491256


namespace inequality_solution_l491_491930

theorem inequality_solution 
  (x y : ℝ) 
  (h : sqrt 3 * tan x - real.sqrt (real.sqrt (sin y)) - sqrt (3 / cos x ^ 2 + real.sqrt (sin y) - 6) ≥ sqrt 3) :
  ∃ (n k : ℤ), x = π / 4 + π * n ∧ y = π * k := 
sorry

end inequality_solution_l491_491930


namespace problem_l491_491520

noncomputable def a_seq (n : ℕ) : ℚ := sorry

def is_geometric_sequence (seq : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n : ℕ, seq (n + 1) = q * seq n

theorem problem (h_positive : ∀ n : ℕ, 0 < a_seq n)
                (h_ratio : ∀ n : ℕ, 2 * a_seq n = 3 * a_seq (n + 1))
                (h_product : a_seq 1 * a_seq 4 = 8 / 27) :
  is_geometric_sequence a_seq (2 / 3) ∧ 
  (∃ n : ℕ, a_seq n = 16 / 81 ∧ n = 6) :=
by
  sorry

end problem_l491_491520


namespace dandelion_seed_production_l491_491573

theorem dandelion_seed_production :
  (one_seed : ℕ) (produced_seeds : ℕ)
  (germinated_fraction : ℚ)
  (new_seedlings_count : ℕ)
  (seed_count_after_two_months : ℕ) :
  one_seed = 1 →
  produced_seeds = 50 →
  germinated_fraction = 1/2 →
  new_seedlings_count = produced_seeds * germinated_fraction.numerator / germinated_fraction.denominator →
  seed_count_after_two_months = new_seedlings_count * produced_seeds →
  seed_count_after_two_months = 1250 :=
by
  intros
  sorry

end dandelion_seed_production_l491_491573


namespace Maxim_born_in_2008_probability_l491_491151

def birth_interval_start := ⟨2007, 9, 2⟩ -- September 2, 2007
def birth_interval_end := ⟨2008, 8, 31⟩ -- August 31, 2008
def days_in_interval := 365
def days_in_2008 := 244

theorem Maxim_born_in_2008_probability :
  (days_in_2008 : ℝ) / (days_in_interval : ℝ) = 244 / 365 :=
sorry

end Maxim_born_in_2008_probability_l491_491151


namespace hyperbola_equation_l491_491824

theorem hyperbola_equation (a b : ℝ) (x y : ℝ) (P F : Point)
  (ha : 0 < a)
  (hb : 0 < b)
  (hf : F = (2,0))
  (h_intersect : True) -- assume they intersect, precise point not needed explicitly here
  (h_PF : |PF| = 5)
  (h_focus_coincide : c = 2)
  : (x^2 - y^2 / 3 = 1) :=
begin
  sorry
end

end hyperbola_equation_l491_491824


namespace smallest_multiple_of_18_all_digits_9_or_0_l491_491198

theorem smallest_multiple_of_18_all_digits_9_or_0 :
  ∃ (m : ℕ), (m > 0) ∧ (m % 18 = 0) ∧ (∀ d ∈ (m.digits 10), d = 9 ∨ d = 0) ∧ (m / 18 = 5) :=
sorry

end smallest_multiple_of_18_all_digits_9_or_0_l491_491198


namespace probability_factor_of_30_correct_l491_491252

noncomputable def probability_factor_of_30 : ℚ :=
  let n := 30
  let divisors_of_30 : Finset ℕ := {d ∈ (Finset.range (n + 1)) | n % d = 0}
  (divisors_of_30.card : ℚ) / n

theorem probability_factor_of_30_correct :
  probability_factor_of_30 = 4 / 15 :=
by
  sorry

end probability_factor_of_30_correct_l491_491252


namespace Maxim_born_in_2008_probability_l491_491153

def birth_interval_start := ⟨2007, 9, 2⟩ -- September 2, 2007
def birth_interval_end := ⟨2008, 8, 31⟩ -- August 31, 2008
def days_in_interval := 365
def days_in_2008 := 244

theorem Maxim_born_in_2008_probability :
  (days_in_2008 : ℝ) / (days_in_interval : ℝ) = 244 / 365 :=
sorry

end Maxim_born_in_2008_probability_l491_491153


namespace common_ratio_of_geometric_series_l491_491322

theorem common_ratio_of_geometric_series (a S r : ℝ) (h1 : a = 500) (h2 : S = 2500) (h3 : a / (1 - r) = S) : r = 4 / 5 :=
by
  rw [h1, h2] at h3
  sorry

end common_ratio_of_geometric_series_l491_491322


namespace pool_ground_area_l491_491387

theorem pool_ground_area (length width depth : ℕ) (h_length : length = 5) (h_width : width = 4) (h_depth : depth = 2) :
  length * width = 20 := 
by
  -- Given: length = 5, width = 4, depth = 2
  -- Need to show: length * width = 20
  rw [h_length, h_width]
  norm_num

end pool_ground_area_l491_491387


namespace sara_spent_on_movies_l491_491907

def cost_of_movie_tickets : ℝ := 2 * 10.62
def cost_of_rented_movie : ℝ := 1.59
def cost_of_purchased_movie : ℝ := 13.95

theorem sara_spent_on_movies :
  cost_of_movie_tickets + cost_of_rented_movie + cost_of_purchased_movie = 36.78 := by
  sorry

end sara_spent_on_movies_l491_491907


namespace doris_weeks_to_meet_expenses_l491_491725

def doris_weekly_hours : Nat := 5 * 3 + 5 -- 5 weekdays (3 hours each) + 5 hours on Saturday
def doris_hourly_rate : Nat := 20 -- Doris earns $20 per hour
def doris_weekly_earnings : Nat := doris_weekly_hours * doris_hourly_rate -- The total earnings per week
def doris_monthly_expenses : Nat := 1200 -- Doris's monthly expense

theorem doris_weeks_to_meet_expenses : ∃ w : Nat, doris_weekly_earnings * w ≥ doris_monthly_expenses ∧ w = 3 :=
by
  sorry

end doris_weeks_to_meet_expenses_l491_491725


namespace dandelion_seed_production_l491_491574

theorem dandelion_seed_production :
  (one_seed : ℕ) (produced_seeds : ℕ)
  (germinated_fraction : ℚ)
  (new_seedlings_count : ℕ)
  (seed_count_after_two_months : ℕ) :
  one_seed = 1 →
  produced_seeds = 50 →
  germinated_fraction = 1/2 →
  new_seedlings_count = produced_seeds * germinated_fraction.numerator / germinated_fraction.denominator →
  seed_count_after_two_months = new_seedlings_count * produced_seeds →
  seed_count_after_two_months = 1250 :=
by
  intros
  sorry

end dandelion_seed_production_l491_491574


namespace sqrt_0_09_eq_0_3_l491_491960

theorem sqrt_0_09_eq_0_3 : Real.sqrt 0.09 = 0.3 := 
by 
  sorry

end sqrt_0_09_eq_0_3_l491_491960


namespace line_passes_through_fixed_point_min_area_S_l491_491045

section
variables (k : ℝ)

-- Definition of the line l
def line (k : ℝ) : ℝ → ℝ → Prop := λ x y, k * x + y + 1 + 2 * k = 0

-- Definition for point A and B
def point_A (k : ℝ) : ℝ × ℝ := (- (1 + 2 * k) / k, 0)
def point_B (k : ℝ) : ℝ × ℝ := (0, - (1 + 2 * k))

-- Prove the fixed point
theorem line_passes_through_fixed_point (k : ℝ) : line k (-2) (-1) :=
by simp [line, add_comm]

-- Area of triangle AOB
def area_triangle_AOB (k : ℝ) : ℝ :=
1 / 2 * abs ((point_A k).1) * abs ((point_B k).2)

-- Minimum area condition
def valid_k (k : ℝ) : Prop :=
(λ x, - (1 + 2 * k) / k < 0) ∧ (λ y, 1 + 2 * k > 0)

theorem min_area_S (k : ℝ) (hk : valid_k k) :
∃ k₀, k₀ = 1/2 ∧ area_triangle_AOB k = 4 ∧ line k₀ (1 * x) (2 * y + 4 = 0)
 :=
sorry

end

end line_passes_through_fixed_point_min_area_S_l491_491045


namespace students_in_all_classes_l491_491962

theorem students_in_all_classes (total_students : ℕ) (students_photography : ℕ) (students_music : ℕ) (students_theatre : ℕ) (students_dance : ℕ) (students_at_least_two : ℕ) (students_in_all : ℕ) :
  total_students = 30 →
  students_photography = 15 →
  students_music = 18 →
  students_theatre = 12 →
  students_dance = 10 →
  students_at_least_two = 18 →
  students_in_all = 4 :=
by
  intros
  sorry

end students_in_all_classes_l491_491962


namespace complex_div_eq_two_l491_491699

theorem complex_div_eq_two : 
  ( (1 + complex.sqrt 3 * complex.I) ^ 2 / (complex.sqrt 3 * complex.I - 1) = 2) :=
by
  sorry

end complex_div_eq_two_l491_491699


namespace AM_GM_inequality_problem_l491_491427

theorem AM_GM_inequality_problem (n : ℕ) (x : Fin n → ℝ) (h : ∀ i, 0 < x i) : 
  (∑ i in Finset.range n, x (i+1 % n.succ).succ ^ (i + 2) / x (i+1 % n).succ ^ i) + 
  x 0 ^ (n + 1) / x n ^ n ≥ 
  2 * (∑ i in Finset.range (n - 1), x (i + 1).succ) - (n - 2) * x 0 :=
sorry

end AM_GM_inequality_problem_l491_491427


namespace eating_possible_values_l491_491262

def A : Set ℝ := {-1, 1 / 2, 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x ^ 2 = 1 ∧ a ≥ 0}

-- Full eating relation: B ⊆ A
-- Partial eating relation: (B ∩ A).Nonempty ∧ ¬(B ⊆ A ∨ A ⊆ B)

def is_full_eating (a : ℝ) : Prop := B a ⊆ A
def is_partial_eating (a : ℝ) : Prop :=
  (B a ∩ A).Nonempty ∧ ¬(B a ⊆ A ∨ A ⊆ B a)

theorem eating_possible_values :
  {a : ℝ | is_full_eating a ∨ is_partial_eating a} = {0, 1, 4} :=
by
  sorry

end eating_possible_values_l491_491262


namespace negative_fraction_comparison_l491_491367

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l491_491367


namespace student_escapes_l491_491090

variable (R v : ℝ)
variable (r : ℝ)
variable (π : ℝ)

-- The condition for r
axiom h_r : (1 - π / 4) * R < r ∧ r < R / 4

-- The student's time to reach the edge
def t_student : ℝ := R / v

-- The teacher's time to reach the student's exit point
def t_teacher : ℝ := (π * R) / (4 * v)

theorem student_escapes (hπ : 0 < π) (hv : 0 < v) (hR : 0 < R) : t_student R v < t_teacher R v :=
by
  sorry

end student_escapes_l491_491090


namespace count_non_similar_regular_2310_pointed_stars_l491_491837

theorem count_non_similar_regular_2310_pointed_stars :
  let n := 2310 in
  let coprime_count := (Finset.range n).filter (λ m, Nat.gcd m n = 1).card - 2;
  (coprime_count / 2) = 224 :=
by
  let n := 2310
  let p := [2, 3, 5, 7, 11]
  -- Calculate phi(n)
  have factorization : n = p.foldl (*) 1 := by sorry
  have euler_totient : (Nat.totient n - 2) / 2 = 224 := by sorry
  exact euler_totient

end count_non_similar_regular_2310_pointed_stars_l491_491837


namespace jars_blue_marble_difference_l491_491233

theorem jars_blue_marble_difference
    (x y : ℝ) 
    (eq_marble_counts : 10 * x = 7 * y)
    (eq_green_marbles : 3 * x + 2 * y = 145) 
    : abs ((7*x) - (5*y)) ≈ 4 :=
  sorry

end jars_blue_marble_difference_l491_491233


namespace proof_problem_l491_491791

-- Defining the arithmetic sequence {a_n} and conditions
def a_n (n : ℕ) : ℕ := n

-- Defining the geometric sequence {b_n}
def b_n (n : ℕ) : ℕ := 2^(n-1)

-- Defining the sequence {c_n}
def c_n (n : ℕ) : ℚ := 1 / (a_n n * log 2 (b_n (n + 2)))

-- Sum of the first n terms of the sequence {c_n}
def T_n (n : ℕ) : ℚ := ∑ i in Finset.range n, c_n (i + 1)

-- Proving the sequences match the given forms and sum formula for T_n
theorem proof_problem : (∀ n : ℕ, a_n n = n) ∧ 
                        (∀ n : ℕ, b_n n = 2^(n-1)) ∧ 
                        (∀ n : ℕ, T_n n = n / (n + 1)) := 
by
  sorry

end proof_problem_l491_491791


namespace function_properties_l491_491536

noncomputable def f (x : ℝ) : ℝ := Real.sin (x * Real.cos x)

theorem function_properties :
  (f x = -f (-x)) ∧
  (∀ x, 0 < x ∧ x < Real.pi / 2 → 0 < f x) ∧
  ¬(∃ T, ∀ x, f (x + T) = f x) ∧
  (∀ n : ℤ, f (n * Real.pi) = 0) := 
by
  sorry

end function_properties_l491_491536


namespace smallest_n_value_l491_491540

theorem smallest_n_value 
  (n : ℕ) 
  (x : ℕ → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ x i) 
  (h_sum : (Finset.range n).sum x = 1) 
  (h_sum_squares : (Finset.range n).sum (λ i, (x i)^2) ≤ 1/49) : 
  49 ≤ n := 
sorry

end smallest_n_value_l491_491540


namespace sin_angle_QPS_l491_491604

theorem sin_angle_QPS (P Q R S : Point) (h_triangle : right_triangle P Q R) 
  (h_midpoint : midpoint S Q R) (h_tan : tan (angle Q P R) = 3 / 2) :
  sin (angle Q P S) = 3 / 5 := 
sorry

end sin_angle_QPS_l491_491604


namespace slope_of_line_intersecting_hyperbola_l491_491803

-- Define points A and B on the hyperbola and the midpoint C
structure Point where
  x : ℝ
  y : ℝ

def on_hyperbola (p : Point) : Prop :=
  p.x^2 - p.y^2 = 1

def midpoint (a b : Point) (c : Point) : Prop :=
  c.x = (a.x + b.x) / 2 ∧ c.y = (a.y + b.y) / 2

-- The main theorem to prove
theorem slope_of_line_intersecting_hyperbola {A B : Point} (C : Point) 
  (h_A_on_hyperbola : on_hyperbola A) 
  (h_B_on_hyperbola : on_hyperbola B) 
  (h_mid_AB_C : midpoint A B C) 
  (h_C : C = ⟨2, 1⟩) : 
  let slope (p1 p2 : Point) := (p2.y - p1.y) / (p2.x - p1.x) in
  slope A B = 2 := 
by sorry

end slope_of_line_intersecting_hyperbola_l491_491803


namespace solve_for_lambda_l491_491463

-- Definitions of the vectors and the parallelism condition
def a : ℝ × ℝ := (2, -1)
def b (λ : ℝ) : ℝ × ℝ := (λ, -3)

def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = k • v

-- Stating the theorem based on the given conditions and the question
theorem solve_for_lambda (λ : ℝ) (h : parallel a (b λ)) : λ = 6 := 
sorry

end solve_for_lambda_l491_491463


namespace find_f_zero_and_f_f_zero_l491_491817

-- Definition of the piecewise function f
def f : ℝ → ℝ :=
  λ x =>
    if h : x < 1 then 2 ^ x
    else log x / log 3  -- Use natural logarithm to define log base 3

theorem find_f_zero_and_f_f_zero : 
  f 0 = 1 ∧ f (f 0) = 0 :=
by
  sorry

end find_f_zero_and_f_f_zero_l491_491817


namespace seed_production_l491_491575

theorem seed_production :
  ∀ (initial_seeds : ℕ) (germination_rate : ℝ) (seed_count_per_plant : ℕ),
    initial_seeds = 1 →
    germination_rate = 0.5 →
    seed_count_per_plant = 50 →
    let new_plants := (initial_seeds * seed_count_per_plant) * germination_rate in
    new_plants * seed_count_per_plant = 1250 :=
by
  intros initial_seeds germination_rate seed_count_per_plant h1 h2 h3
  let new_plants := (initial_seeds * seed_count_per_plant) * germination_rate
  have : new_plants = 25, by {
    rw [h1, h3, h2],
    norm_num,
  }
  rw this
  norm_num

end seed_production_l491_491575


namespace expected_balls_in_original_position_l491_491390

/-- 
Prove that the expected number of balls that occupy their original positions 
after Chris randomly swaps any two balls and Silva swaps two adjacent balls is 2.
-/
theorem expected_balls_in_original_position : 
  (expected_value (λ b : Ball, ball_in_original_position_after_swaps b 8 8)) = 2 :=
sorry

end expected_balls_in_original_position_l491_491390


namespace min_max_x_l491_491114

theorem min_max_x (n : ℕ) (hn : 0 < n) (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = n * x + n * y) : 
  n + 1 ≤ x ∧ x ≤ n * (n + 1) :=
by {
  sorry  -- Proof goes here
}

end min_max_x_l491_491114


namespace compare_fractions_l491_491344

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l491_491344


namespace pyramid_height_value_l491_491665

-- Let s be the edge length of the cube
def cube_edge_length : ℝ := 6

-- Let b be the base edge length of the square-based pyramid
def pyramid_base_edge_length : ℝ := 12

-- Let V_cube be the volume of the cube
def volume_cube (s : ℝ) : ℝ := s ^ 3

-- Let V_pyramid be the volume of the pyramid
def volume_pyramid (b h : ℝ) : ℝ := (1 / 3) * (b ^ 2) * h

-- The given volumes are equal
def volumes_equal : Prop :=
  volume_cube cube_edge_length = volume_pyramid pyramid_base_edge_length h

-- Prove that the height h of the pyramid is 4.5
theorem pyramid_height_value (h : ℝ) (cube_edge_length pyramid_base_edge_length : ℝ) (volumes_equal : Prop) :
  h = 4.5 :=
sorry

end pyramid_height_value_l491_491665


namespace closest_integer_sum_l491_491460

noncomputable def sequence (n : ℕ) : ℝ := sorry -- Define the general term of the sequence a_n.

-- Initial conditions
axiom a_initial : sequence 1 = 2 ∧ sequence 2 = 2

-- Recurrence relation condition
axiom recurrence_relation : ∀ (n : ℕ), n ≥ 2 →
  (2 * sequence (n - 1) * sequence n) / (sequence (n - 1) * sequence (n + 1) - sequence n^2) = (n^3 - n)

-- Statement to be proved
theorem closest_integer_sum : 
  let S := ∑ k in finset.range 2010 \ finset.range 1, (sequence (k + 2)) / (sequence (k + 1))
  in abs (S - 3015) < 1 :=
by
  sorry

end closest_integer_sum_l491_491460


namespace connected_to_all_eventually_l491_491216

-- Formalizing the problem conditions
variables (n : ℕ) (n_geq_3 : n ≥ 3)

-- Formalizing the state of the ferry routes
noncomputable def initial_connected (is_connected : set (ℕ × ℕ)) : Prop :=
  ¬ (∃ (A B : set ℕ), (A ∪ B = set.univ) ∧ A ≠ ∅ ∧ B ≠ ∅ ∧ (∀ a ∈ A, ∀ b ∈ B, (a, b) ∉ is_connected ∧ (b, a) ∉ is_connected))

-- Defining the annual change in ferry routes
noncomputable def year_change (is_connected : set (ℕ × ℕ)) (X Y : ℕ) : set (ℕ × ℕ) :=
let T := { t | (X, t) ∈ is_connected ∨ (Y, t) ∈ is_connected } in
(is_connected \ {(X, Y), (Y, X)}) ∪ { (t, Y) | t ∈ T ∧ (X, t) ∈ is_connected } ∪ { (t, X) | t ∈ T ∧ (Y, t) ∈ is_connected }

-- The main theorem to prove
theorem connected_to_all_eventually (is_connected : set (ℕ × ℕ)) 
  (initial_conn : initial_connected n is_connected) :
  ∃ k, ∀ j, k ≤ j → ∃ v, ∀ i, i ≠ v → (v, i) ∈ (iterate year_change is_connected j) :=
sorry

end connected_to_all_eventually_l491_491216


namespace functional_equation_solution_l491_491399

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x + f(y)) - f(x) = (x + f(y))^4 - x^4) → 
  (∀ x : ℝ, (f(x) = 0 ∨ ∃ k : ℝ, f(x) = x^4 + k)) :=
begin
  sorry
end

end functional_equation_solution_l491_491399


namespace table_tennis_pairing_methods_l491_491637

theorem table_tennis_pairing_methods (m f : ℕ) (hm : m = 5) (hf : f = 4) : 
  (nat.choose m 2) * (nat.choose f 2) * 2 = 120 :=
by
  -- Theorem statement represents the problem's conditions and the expected result
  rw [hm, hf]
  -- The remaining proof steps would go here
  sorry

end table_tennis_pairing_methods_l491_491637


namespace weekly_milk_consumption_l491_491556

def milk_weekday : Nat := 3
def milk_saturday := 2 * milk_weekday
def milk_sunday := 3 * milk_weekday

theorem weekly_milk_consumption : (5 * milk_weekday) + milk_saturday + milk_sunday = 30 := by
  sorry

end weekly_milk_consumption_l491_491556


namespace integer_equation_solution_exists_l491_491224

theorem integer_equation_solution_exists :
  ∃ (a b c d : ℤ), abs a > 1000000 ∧ abs b > 1000000 ∧ abs c > 1000000 ∧ abs d > 1000000 ∧ 
  (1/a : ℚ) + (1/b) + (1/c) + (1/d) = (1:ℚ) / (a * b * c * d) :=
by
  let n := 1000001
  let a := -n
  let b := n + 1
  let c := n * (n + 1) + 1
  let d := n * (n + 1) * (n * (n + 1) + 1)
  use [a, b, c, d]
  -- prove the rest of the conditions
  sorry

end integer_equation_solution_exists_l491_491224


namespace evaluate_modulus_l491_491732

def complex := ℂ
def c : complex := -3 - (5 / 4) * complex.I

theorem evaluate_modulus : complex.abs c = 13 / 4 :=
by sorry

end evaluate_modulus_l491_491732


namespace ants_movement_impossible_l491_491226

theorem ants_movement_impossible (initial_positions final_positions : Fin 3 → ℝ × ℝ) :
  initial_positions 0 = (0,0) ∧ initial_positions 1 = (0,1) ∧ initial_positions 2 = (1,0) →
  final_positions 0 = (-1,0) ∧ final_positions 1 = (0,1) ∧ final_positions 2 = (1,0) →
  (∀ t : ℕ, ∃ m : Fin 3, 
    ∀ i : Fin 3, (i ≠ m → ∃ k l : ℝ, 
      (initial_positions i).2 - l * (initial_positions i).1 = 0 ∧ 
      ∀ (p : ℕ → ℝ × ℝ), p 0 = initial_positions i ∧ p t = final_positions i → 
      (p 0).1 + k * (p 0).2 = 0)) →
  false :=
by 
  sorry

end ants_movement_impossible_l491_491226


namespace Amanda_gift_money_l491_491321

-- Definitions of the costs and remaining money based on the problem statement
def cost_tape : ℕ := 9
def cost_headphone : ℕ := 25
def remaining_money : ℕ := 7

-- The total gift money Amanda received
def gift_money : ℕ := 2 * cost_tape + cost_headphone + remaining_money

-- Proof statement that Amanda received $50 as a gift
theorem Amanda_gift_money : gift_money = 50 := 
by
  unfold gift_money cost_tape cost_headphone remaining_money
  calc
    2 * 9 + 25 + 7
    _ = 18 + 25 + 7 : by simp
    _ = 43 + 7     : by simp
    _ = 50         : by simp
  done

end Amanda_gift_money_l491_491321


namespace determinant_is_77_l491_491371

def matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, 4, -1], ![0, 3, 2], ![5, -1, 3]]

theorem determinant_is_77 : matrix.det = 77 := by
  sorry

end determinant_is_77_l491_491371


namespace sin_identity_proof_l491_491773

theorem sin_identity_proof (α : ℝ) (h : Real.cos (α + π / 6) = sqrt 3 / 3) :
  Real.sin (2 * α - π / 6) = 1 / 3 :=
by
  sorry

end sin_identity_proof_l491_491773


namespace weeks_to_cover_expense_l491_491719

-- Definitions and the statement of the problem
def hourly_rate : ℕ := 20
def monthly_expense : ℕ := 1200
def weekday_hours : ℕ := 3
def saturday_hours : ℕ := 5

theorem weeks_to_cover_expense : 
  ∀ (w : ℕ), (5 * weekday_hours + saturday_hours) * hourly_rate * w ≥ monthly_expense → w >= 3 := 
sorry

end weeks_to_cover_expense_l491_491719


namespace weeks_to_cover_expense_l491_491718

-- Definitions and the statement of the problem
def hourly_rate : ℕ := 20
def monthly_expense : ℕ := 1200
def weekday_hours : ℕ := 3
def saturday_hours : ℕ := 5

theorem weeks_to_cover_expense : 
  ∀ (w : ℕ), (5 * weekday_hours + saturday_hours) * hourly_rate * w ≥ monthly_expense → w >= 3 := 
sorry

end weeks_to_cover_expense_l491_491718


namespace abs_sum_k_element_subset_le_one_l491_491021

theorem abs_sum_k_element_subset_le_one
    (n k : ℕ) (h1 : k ≤ n - 2)
    (a : Fin n → ℝ)
    (h2 : ∀ (s : Finset (Fin n)), s.card = k → |s.sum (λ i, a i)| ≤ 1)
    (h3 : |a 0| ≥ 1) :
    ∀ i : Fin n, 1 ≤ i → |a 0| + |a i| ≤ 2 := sorry

end abs_sum_k_element_subset_le_one_l491_491021


namespace ai_sum_is_neg_121_l491_491422

theorem ai_sum_is_neg_121 (a a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (2 * (1 : ℤ) - 1) ^ 5 = a * 1^5 + a_1 * 1^4 + a_2 * 1^3 + a_3 * 1^2 + a_4 * 1 + a_5 →
  (2 * (-1 : ℤ) - 1) ^ 5 = -a * (-1)^5 + a_1 * (-1)^4 + a_2 * (-1)^3 + a_3 * (-1)^2 + a_4 * (-1) + a_5 →
  a_1 + a_3 + a_5 = -121 :=
begin
  intros h1 h2,
  sorry,
end

end ai_sum_is_neg_121_l491_491422


namespace gcf_shared_prime_l491_491525

theorem gcf_shared_prime {m n : ℕ} (P1 P2 P3 P4 Q1 Q2 Q3 : ℕ) 
  (hmf : ∀ p, Prime p → (p = P1 ∨ p = P2 ∨ p = P3 ∨ p = P4) → Prime p)
  (hnf : ∀ q, Prime q → (q = Q1 ∨ q = Q2 ∨ q = Q3) → Prime q)
  (h_shared : P1 = Q1)
  (hmn : ∃ p, Prime p ∧ (p = P1 ∨ p = P2 ∨ p = P3 ∨ p = P4) ∧
          ∃ q, Prime q ∧ (q = Q1 ∨ q = Q2 ∨ q = Q3) ∧
          Frobenius_eq (P1 * P2 * P3 * P4 * Q2 * Q3) (m * n))
  (hmn_pf : ∃ i j, (i ≠ j) ∧ (i ≠ P1 ∧ i ≠ P2 ∧ i ≠ P3 ∧ i ≠ P4 ∧ 
                              j ≠ P1 ∧ j ≠ P2 ∧ j ≠ P3 ∧ j ≠ P4)) :
    gcd m n = P1 :=
by sorry

end gcf_shared_prime_l491_491525


namespace toms_friend_decks_l491_491229

theorem toms_friend_decks
  (cost_per_deck : ℕ)
  (tom_decks : ℕ)
  (total_spent : ℕ)
  (h1 : cost_per_deck = 8)
  (h2 : tom_decks = 3)
  (h3 : total_spent = 64) :
  (total_spent - tom_decks * cost_per_deck) / cost_per_deck = 5 := by
  sorry

end toms_friend_decks_l491_491229


namespace angle_constraints_l491_491545

open Real

noncomputable def angle_in_degrees := Real

def triangle (A B C : Type) : Prop := True
def is_circumcenter_on_segment (A B C B1 C1 : Type) : Prop := True
def angle_geq (A B C : Type) (x y : angle_in_degrees) : Prop := x ≥ y

theorem angle_constraints (A B C : Type) 
  (B1 C1 : Type)
  (h_triangle : triangle A B C)
  (h_circumcenter : is_circumcenter_on_segment A B C B1 C1)
  (h_angle_condition : ∀ α β γ, α = ∠BAC ∧ β = ∠ABC ∧ γ ≥ β)
  : 
  ∃ (α β γ : angle_in_degrees), 
  45 < α ∧ α <= 51.8296 ∧
  45 < β ∧ β <= 64.0906 ∧
  45 < γ ∧ γ < 90 :=
sorry

end angle_constraints_l491_491545


namespace solve_fractional_equation_l491_491951

theorem solve_fractional_equation : ∀ x : ℝ, (2 * x / (x - 1) = 3) ↔ x = 3 := 
by
  sorry

end solve_fractional_equation_l491_491951


namespace tan_theta_is_2_minus_sqrt_3_l491_491772

noncomputable def θ : ℝ := sorry
def h1 (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2
def h2 (θ : ℝ) : Prop :=
  (sin θ + cos θ) ^ 2 + sqrt 3 * cos (2 * θ) = 3

theorem tan_theta_is_2_minus_sqrt_3 (θ : ℝ) (h1 : h1 θ) (h2 : h2 θ) :
  tan θ = 2 - sqrt 3 :=
sorry

end tan_theta_is_2_minus_sqrt_3_l491_491772


namespace quadratic_function_graph_opens_downwards_l491_491264

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

-- The problem statement to prove
theorem quadratic_function_graph_opens_downwards :
  (∀ x : ℝ, (quadratic_function (x + 1) - quadratic_function x) < (quadratic_function x - quadratic_function (x - 1))) :=
by
  -- Proof omitted
  sorry

end quadratic_function_graph_opens_downwards_l491_491264


namespace p_sufficient_not_necessary_for_q_l491_491458

-- Define the propositions p and q based on the given conditions
def p (α : ℝ) : Prop := α = Real.pi / 4
def q (α : ℝ) : Prop := Real.sin α = Real.cos α

-- Theorem that states p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q (α : ℝ) : p α → (q α) ∧ ¬(q α → p α) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l491_491458


namespace base_log_eq_l491_491760

theorem base_log_eq (b : ℝ) : (∀ x : ℝ, (7^(x + 8) = 9^x) → (x = log b (7^8))) → b = 9 / 7 :=
by
  intro h
  apply h
  sorry

end base_log_eq_l491_491760


namespace ophelia_current_age_proof_l491_491861

variable (Lennon_age : ℕ)
variable (Mike_age : ℕ)
variable (Ophelia_age_in_10_years : ℕ)
variable (Ophelia_current_age : ℕ)
variable (years_ahead : ℕ)

-- Conditions
def conditions :=
  Lennon_age = 8 ∧ 
  Mike_age = 13 ∧ 
  years_ahead = 10 ∧ 
  Ophelia_age_in_10_years = 3 * (Lennon_age + years_ahead) ∧
  Ophelia_current_age = Ophelia_age_in_10_years - years_ahead

-- Theorem: Prove Ophelia's current age
theorem ophelia_current_age_proof (h : conditions) : Ophelia_current_age = 44 :=
sorry

end ophelia_current_age_proof_l491_491861


namespace Maxim_born_in_2008_probability_l491_491154

def birth_interval_start := ⟨2007, 9, 2⟩ -- September 2, 2007
def birth_interval_end := ⟨2008, 8, 31⟩ -- August 31, 2008
def days_in_interval := 365
def days_in_2008 := 244

theorem Maxim_born_in_2008_probability :
  (days_in_2008 : ℝ) / (days_in_interval : ℝ) = 244 / 365 :=
sorry

end Maxim_born_in_2008_probability_l491_491154


namespace domain_of_f_l491_491242

noncomputable def f (x : ℝ) : ℝ := (x^3 - 8) / (x + 8)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, y = f x} = set.univ \ {-8} :=
by
  sorry

end domain_of_f_l491_491242


namespace num_ways_to_seat_11_around_table_l491_491860

def factorial (n : ℕ) : ℕ :=
match n with
  | 0     => 1
  | (n + 1) => (n + 1) * factorial n
  end

theorem num_ways_to_seat_11_around_table : 
  let distinguished_position := 1
  let num_people := 11
  (num_people - distinguished_position)! = 3628800 :=
by
  sorry

end num_ways_to_seat_11_around_table_l491_491860


namespace acute_angle_at_7_50_l491_491247

-- Definition of the problem condition: given time is 7:50 on a standard clock
def angle_between_clock_hands (h m : ℕ) : ℝ :=
  let minute_angle := (m % 60) * 6  -- Minute hand moves 6 degrees each minute
  let hour_angle := (h % 12) * 30 + (m % 60) * 0.5  -- Hour hand moves 0.5 degrees each minute
  let angle_diff := abs (minute_angle - hour_angle)
  if angle_diff > 180 then 360 - angle_diff else angle_diff

-- Theorem stating the desired property
theorem acute_angle_at_7_50 : angle_between_clock_hands 7 50 = 65 := by
  sorry

end acute_angle_at_7_50_l491_491247


namespace smallest_integer_groups_l491_491417

theorem smallest_integer_groups (n : ℕ) (d : ℕ) (a : fin d → ℝ) (h_sum : ∑ i, a i = n) (h_bounds : ∀ i, 0 ≤ a i ∧ a i ≤ 1) : 
  ∃ k : ℕ, k = 2 * n - 1 ∧ (∀ (partition : Π (i : fin k), set (fin d)), (∀ j, ∑ i in (partition j), a i ≤ 1) → ∃ t : fin k, ∀ j, j ≠ t → (partition j : set (fin d)) = ∅) :=
begin
  sorry
end

end smallest_integer_groups_l491_491417


namespace range_of_sqrt_1_minus_exp_div_3_l491_491625

def range_of_function (x : ℝ) : ℝ :=
  real.sqrt (1 - (1 / 3)^x)

theorem range_of_sqrt_1_minus_exp_div_3 :
  set.range range_of_function = set.Ico 0 1 :=
sorry

end range_of_sqrt_1_minus_exp_div_3_l491_491625


namespace rhombus_diagonal_ratio_l491_491117

-- Definitions for the conditions
variables (A B C D E F G I K : Type) [euclidean_geometry.point A B] [euclidean_geometry.point B C] [euclidean_geometry.point C D] [euclidean_geometry.point D A]
variables [euclidean_geometry.rhombus ABCD]
variables [euclidean_geometry.intersection E (diagonal AC) (diagonal BD)]
variables [euclidean_geometry.midpoint F (segment BE)]
variables [euclidean_geometry.midpoint G (segment AD)]
variables [euclidean_geometry.intersection I (line FG) (line AC)]
variables [euclidean_geometry.reflection K A I]

-- The theorem to prove
theorem rhombus_diagonal_ratio : EK / EA = 1 / 2 :=
by
  sorry

end rhombus_diagonal_ratio_l491_491117


namespace max_sqrt_3x_plus_sqrt_2y_min_3_x_plus_2_y_l491_491004

variable {x y : ℝ}

theorem max_sqrt_3x_plus_sqrt_2y (h1 : x > 0) (h2 : y > 0) (h3 : 3 * x + 2 * y = 10) :
  sqrt (3 * x) + sqrt (2 * y) ≤ 2 * sqrt 5 := sorry

theorem min_3_x_plus_2_y (h1 : x > 0) (h2 : y > 0) (h3 : 3 * x + 2 * y = 10) :
  3 / x + 2 / y ≥ 5 / 2 := sorry

end max_sqrt_3x_plus_sqrt_2y_min_3_x_plus_2_y_l491_491004


namespace aimee_poll_l491_491314

theorem aimee_poll (W P : ℕ) (h1 : 0.35 * W = 21) (h2 : 2 * W = P) : P = 120 :=
by
  -- proof in Lean is omitted, placeholder
  sorry

end aimee_poll_l491_491314


namespace numbers_not_divisible_by_2_to_16_l491_491331

theorem numbers_not_divisible_by_2_to_16 : 
  (∃ (n : ℕ), n = 30030) → 
  (∀ k ∈ (set.range 2 ∪ set.range 3 ∪ set.range 4 ∪ set.range 5 ∪ set.range 6 ∪ set.range 7 ∪ set.range 8 ∪ set.range 9 ∪ set.range 10 ∪ set.range 11 ∪ set.range 12 ∪ set.range 13 ∪ set.range 14 ∪ set.range 15 ∪ set.range 16), k|30030 → False) → 
  (finset.range 30030).filter (λ x, ¬∃ k ∈ finset.range 2 ∪ finset.range 3 ∪ finset.range 4 ∪ finset.range 5 ∪ finset.range 6 ∪ finset.range 7 ∪ finset.range 8 ∪ finset.range 9 ∪ finset.range 10 ∪ finset.range 11 ∪ finset.range 12 ∪ finset.range 13 ∪ finset.range 14 ∪ finset.range 15 ∪ finset.range 16, (k | x)).card = 5760 := by {
  sorry
}

end numbers_not_divisible_by_2_to_16_l491_491331


namespace find_area_of_triangle_formed_by_centers_l491_491167

-- Define the problem conditions
def isosceles_right_triangle (a b c : ℝ) : Prop := 
  a = b ∧ c = a * Real.sqrt 2

def centers_of_squares (a : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let d := (a / 2, a / 2)
  let e := (a + a / 2, a / 2)
  let f := (a * (1 + Real.sqrt 2) / 2, a * (1 + Real.sqrt 2) / 2)
  (d, e, f)

def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * Real.abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The main theorem to prove
theorem find_area_of_triangle_formed_by_centers (a b c : ℝ) (h : isosceles_right_triangle a b c) : 
  triangle_area (centers_of_squares a).1 (centers_of_squares a).2.1 (centers_of_squares a).2.2 = c^2 / 2 := 
sorry

end find_area_of_triangle_formed_by_centers_l491_491167


namespace exists_k_with_odd_solutions_l491_491770

-- Definitions based on conditions:
def f (x : ℤ) : ℤ := sorry
def g (y : ℤ) : ℤ := sorry

-- Interval for x and y
def interval := set.Icc (-100 : ℤ) 100

-- Function to count the number of solutions to f(x) - g(y) = k
def n_k (k : ℤ) : ℕ := 
  finset.card ((finset.filter (λ xy, f xy.1 - g xy.2 = k)
                 ((finset.Icc (-100) 100).product (finset.Icc (-100) 100))))

-- The theorem stating the goal
theorem exists_k_with_odd_solutions : ∃ k : ℤ, odd (n_k k) := 
sorry

end exists_k_with_odd_solutions_l491_491770


namespace man_investment_l491_491670

def face_value := 10 -- face value of each share in Rs.
def quoted_price := 9.50 -- quoted price of each share in Rs.
def dividend_rate := 14 / 100 -- dividend rate as a fraction
def annual_income := 728 -- annual income from dividends in Rs.

def dividend_per_share := dividend_rate * face_value -- calculating dividend per share
def number_of_shares := annual_income / dividend_per_share -- calculating number of shares
def total_investment := number_of_shares * quoted_price -- calculating total investment

theorem man_investment : total_investment = 4940 := by
  sorry

end man_investment_l491_491670


namespace triangle_side_length_l491_491434

theorem triangle_side_length (A B C : Type)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (AB AC : ℝ) (H_AB : AB = 3) (H_AC : AC = 4)
  (dot_prod : ℝ) (H_dot : dot_prod = 6)
  (cos_angle : ℝ) (cos_def : dot_prod = AB * AC * cos_angle) :
  let BC := sqrt (AB^2 + AC^2 - 2 * AB * AC * cos_angle) 
  in BC = sqrt 13 := sorry

end triangle_side_length_l491_491434


namespace tangent_line_eq_l491_491938

theorem tangent_line_eq (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 2 * x + 1) :
  ∀ x y : ℝ, (x = 1 ∧ y = f 1) → (tangent_line f 1 x y) = (x - y - 1 = 0) := by
  -- This is where the proof would go
  sorry

def tangent_line (f : ℝ → ℝ) (a x y : ℝ) : Prop :=
  y - f a = (deriv f a) * (x - a)

end tangent_line_eq_l491_491938


namespace equal_segments_l491_491134

variable {A B C D E F H P Q R : Point}

-- Definitions
def acute_triangle (A B C : Point) : Prop := ∀ (angle : Angle), angle.lies_within_triangle A B C → angle.is_acute
def midpoint (D : Point) (A B : Point) : Prop := D.x = ½ * (A.x + B.x) ∧ D.y = ½ * (A.y + B.y)
def is_orthocenter (H : Point) (A B C : Point) : Prop := collinear (H, altitudes A B C)
def circle_contains_triangle (H A B C : Point) : Prop := ∃ (r : ℝ), ∀ (P : Point), dist P H = r → lies_within_circle A B C H r

-- Theorem statement
theorem equal_segments
  (h_triangle : acute_triangle A B C)
  (h_midpoints : midpoint D B C ∧ midpoint E A C ∧ midpoint F A B)
  (h_orthocenter : is_orthocenter H A B C)
  (h_circle : circle_contains_triangle H A B C)
  (h_intersects : ∃ (P Q R : Point), line_segment_intersect_circle EF P ∧ line_segment_intersect_circle FD Q ∧ line_segment_intersect_circle DE R) :
  dist A P = dist B Q ∧ dist B Q = dist C R :=
sorry

end equal_segments_l491_491134


namespace min_major_axis_of_ellipse_l491_491806

theorem min_major_axis_of_ellipse (b c a : ℝ) (h1 : b * c = 1) (h2 : a^2 = b^2 + c^2) : 2 * a ≥ 2 * Real.sqrt 2 :=
begin
  sorry
end

end min_major_axis_of_ellipse_l491_491806


namespace circle_and_line_properties_l491_491518

noncomputable def polar_coordinate_equation (ρ θ : ℝ) : Prop := 
  ρ^2 - 6 * ρ * Real.cos θ + 7 = 0

noncomputable def line_equation (x y : ℝ) : Prop := 
  y = Real.sqrt 35 * x

theorem circle_and_line_properties :
  (∀ (x y : ℝ), y = x + 5 → x = 3 ∧ y = 0 → ρ θ, polar_coordinate_equation ρ θ) ∧
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi / 2 ∧ ρ > 0 ∧ 
     (∀ {OA OB : ℝ}, 1 / Real.abs OA + 1 / Real.abs OB = 1 / 7 → α = y → α = x) → 
     ∃ x y, line_equation x y) :=
by
  sorry

end circle_and_line_properties_l491_491518


namespace each_person_gets_after_taxes_l491_491106

-- Definitions based strictly on problem conditions
def house_price : ℝ := 500000
def market_multiplier : ℝ := 1.2
def brothers_count : ℕ := 3
def tax_rate : ℝ := 0.1

-- Derived conditions
def selling_price : ℝ := house_price * market_multiplier
def total_people : ℕ := 1 + brothers_count
def share_before_taxes : ℝ := selling_price / total_people
def tax_amount_per_person : ℝ := share_before_taxes * tax_rate
def final_amount_per_person : ℝ := share_before_taxes - tax_amount_per_person

-- Problem: Prove the final amount each person receives
theorem each_person_gets_after_taxes : final_amount_per_person = 135000 := by
  sorry

end each_person_gets_after_taxes_l491_491106


namespace range_satisfying_fx2_le1_l491_491380

open set real

-- Define an even function f on ℝ that is monotonically increasing on [0, +∞)
variable {f : ℝ → ℝ}
hypothesis hf_even : ∀ x : ℝ, f x = f (-x)
hypothesis hf_mono : ∀ (x y : ℝ), 0 ≤ x → x ≤ y → f x ≤ f y
hypothesis hf_neg2 : f (-2) = 1

-- The range of x that satisfies f(x - 2) ≤ 1 is [0, 4]
theorem range_satisfying_fx2_le1 : { x : ℝ | f (x - 2) ≤ 1 } = Icc 0 4 :=
by
  sorry

end range_satisfying_fx2_le1_l491_491380


namespace impossible_to_get_nine_zeros_l491_491000

-- Define the initial circle configuration of ones and zeros
def initial_circle_config (circle : List ℕ) : Prop :=
  circle.length = 9 ∧ (circle.count 0 = 5) ∧ (circle.count 1 = 4)

-- Define the operation on adjacent elements in circle
def operation (a b : ℕ) : ℕ := if a = b then 0 else 1

-- Define the iterative process applying the operation to the circle
def next_circle (circle : List ℕ) : List ℕ :=
  (List.zipWith operation circle (circle.tail ++ [circle.head])).dropLast.dropLast

-- Prove that it's impossible to end up with a circle of nine zeros
theorem impossible_to_get_nine_zeros (circle : List ℕ) :
  initial_circle_config circle → ¬(∃ k, (iterate next_circle k circle) = List.replicate 9 0) :=
by
  sorry

end impossible_to_get_nine_zeros_l491_491000


namespace intersection_eq_set_l491_491023

-- Define set A based on the inequality
def A : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define set B based on the inequality
def B : Set ℝ := {x | 0 ≤ Real.log (x + 1) / Real.log 2 ∧ Real.log (x + 1) / Real.log 2 < 2}

-- Translate the question to a lean theorem
theorem intersection_eq_set : (A ∩ B) = {x | 0 ≤ x ∧ x < 1} := 
sorry

end intersection_eq_set_l491_491023


namespace abs_gt_implies_nec_not_suff_l491_491775

theorem abs_gt_implies_nec_not_suff {a b : ℝ} : 
  (|a| > b) → (∀ (a b : ℝ), a > b → |a| > b) ∧ ¬(∀ (a b : ℝ), |a| > b → a > b) :=
by
  sorry

end abs_gt_implies_nec_not_suff_l491_491775


namespace polynomial_has_complex_root_l491_491048

-- Defining the polynomial f(x)
def f (x : Complex) : Complex :=
  a_{2007} * x^2007 + a_{2006} * x^2006 + a_{2005} * x^2005 + 
  a_{2004} * x^2004 + a_{2003} * x^2003 + a_{2002} * x^2002 + 
  a_{2001} * x^2001 + a_{2000} * x^2000 + a_{1999} * x^1999 + 
  -- continue adding all terms until 
  a_{3} * x^3 + 2 * x^2 + x + 1

theorem polynomial_has_complex_root (a_{2007} a_{2006} a_{2005} a_{2004} a_{2003} a_{2002} a_{2001} a_{2000} a_{1999} : Complex) :
  ∃ x : Complex, f x = 0 :=
sorry

end polynomial_has_complex_root_l491_491048


namespace circle_through_point_and_tangent_to_lines_l491_491781

theorem circle_through_point_and_tangent_to_lines :
  ∃ h k,
     ((h, k) = (4 / 5, 3 / 5) ∨ (h, k) = (4, -1)) ∧ 
     ((x - h)^2 + (y - k)^2 = 5) :=
by
  let P := (3, 1)
  let l1 := fun x y => x + 2 * y + 3 
  let l2 := fun x y => x + 2 * y - 7 
  sorry

end circle_through_point_and_tangent_to_lines_l491_491781


namespace polynomial_count_l491_491714

def polynomial_existence (n : ℕ) : Prop :=
  ∃ (a : Fin n → ℤ), n + (Finset.univ.sum (λ i, abs (a i))) = 6

theorem polynomial_count : ∑ n in Finset.range 5, 
  (if h : polynomial_existence n then 1 else 0) = 25 := sorry

end polynomial_count_l491_491714


namespace efficiency_increase_l491_491291

-- Define the given conditions
def T_orig : ℕ := 20
def T_act : ℕ := 18
def E_orig : ℝ := 1 / T_orig
def E_act : ℝ := 1 / T_act
def x : ℝ -- Percentage increase in work efficiency

-- Formalize the problem statement in Lean 4
theorem efficiency_increase :
  E_act = E_orig * (1 + x / 100) →
  1 / 18 = 1 / 20 * (1 + x / 100) :=
by
  sorry

end efficiency_increase_l491_491291


namespace geometric_sequence_common_ratio_l491_491856

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : a 3 = 1/2)
  (h3 : a 1 * (1 + q) = 3) :
  q = 1/2 :=
sorry

end geometric_sequence_common_ratio_l491_491856


namespace find_p_plus_q_l491_491231

noncomputable def side_lengths_DE := 15
noncomputable def side_lengths_EF := 36
noncomputable def side_lengths_FD := 39

variable (ω : ℝ) (γ δ : ℝ)

def Area_parallelogram (ω : ℝ) : ℝ := γ * ω - δ * ω^2

def Herons_formula : ℝ :=
  let s := 45 in
  Real.sqrt (45 * (45 - side_lengths_DE) * (45 - side_lengths_EF) * (45 - side_lengths_FD))

theorem find_p_plus_q (p q : ℕ) (hpq_coprime : Nat.coprime p q) (h : δ = (p : ℝ) / (q : ℝ)) :
    (p + q = 17) := by
  have h1 : 36 * γ - 1296 * δ = 0 := sorry {-From condition 4-}
  have h2 : 135 = γ * 18 * 36 - δ * 18^2 := sorry {-From condition 5-}
  
  have area_DEF : ℝ := Herons_formula
  have : area_DEF = 270 := by
    unfold Herons_formula
    sorry
  have half_area_DEF := area_DEF / 2
  have : 135 = 324 * δ := by
    unfold Area_parallelogram at h2
    sorry
  have : δ = 5 / 12 := by
    sorry
  have : q = 12 := by
    sorry
  have : p = 5 := by
    sorry
  have : p + q = 17 := by
    sorry
  exact this


end find_p_plus_q_l491_491231


namespace intersection_eq_l491_491022

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

-- Prove the intersection of A and B is {0, 1}
theorem intersection_eq : A ∩ B = {0, 1} := by
  sorry

end intersection_eq_l491_491022


namespace triangle_two_solutions_l491_491870

theorem triangle_two_solutions {a b A : ℝ} (ha : a = 2 * Real.sqrt 3) (hb : b = 4) 
  (hA : A = π / 6 ∨ Real.cos A = 3 / 5) : 
  (∃ B C : ℝ, B ≠ C ∧ ∃ (triangle1 triangle2 : Triangle), 
    triangle1.sides = (a, b, 2 * a * b * Real.sin A) ∧ 
    triangle2.sides = (a, b, 2 * a * b * Real.sin A) ∧ triangle1.angles = (A, B, 180 - (A + B)) ∧ 
    triangle2.angles = (A, C, 180 - (A + C))) :=
by {
  sorry
}

end triangle_two_solutions_l491_491870


namespace construct_isosceles_triangle_l491_491378

variables (A B C : Type) [metric_space A]

noncomputable def height_to_base (A B C : A) : ℝ := sorry -- Define the function for height to base
noncomputable def height_to_side (A B C : A) : ℝ := sorry -- Define the function for height to side

noncomputable def is_isosceles (A B C : Type) [metric_space A] : Prop := dist A B = dist A C

-- Define the problem and conditions
theorem construct_isosceles_triangle (m_a m_b : ℝ) (H : 2 * m_a > m_b) :
  ∃ (A B C : A), is_isosceles A B C ∧ height_to_base A B C = m_a ∧ height_to_side B A C = m_b :=
sorry -- Proof or construction to follow

end construct_isosceles_triangle_l491_491378


namespace average_of_N_l491_491241

theorem average_of_N (N : ℤ) (h1 : (1:ℚ)/3 < N/90) (h2 : N/90 < (2:ℚ)/5) : 31 ≤ N ∧ N ≤ 35 → (N = 31 ∨ N = 32 ∨ N = 33 ∨ N = 34 ∨ N = 35) → (31 + 32 + 33 + 34 + 35) / 5 = 33 := by
  sorry

end average_of_N_l491_491241


namespace max_b_for_no_lattice_points_l491_491375

theorem max_b_for_no_lattice_points :
  ∃ b, (∀ m ∈ Ioo (1/3 : ℝ) b, ∃ (x : ℕ), (0 < x ∧ x ≤ 150) → ¬ (∃ y : ℕ, y = m * x + 3)) ∧ b = (50 / 151 : ℚ) :=
sorry

end max_b_for_no_lattice_points_l491_491375


namespace measure_angle_PQR_l491_491511

-- Definitions for the problem conditions
variables (R S P Q : Type) [line : ∀ {P Q R : Type}, Prop]
variables (angle : Type → Type)
variable (measure : angle → ℝ)
variables (RSP : line R S P) (QSP : angle) (h_QSP : measure QSP = 80)

-- Straight line condition: angles on a straight line sum to 180 degrees
axiom straight_line_angle_sum :
  ∀ {A B C : Type} (l : line A B C) (θ1 θ2 : angle),
  measure θ1 + measure θ2 = 180

-- Isosceles triangle property: base angles are equal
axiom isosceles_triangle :
  ∀ {A B C : Type} (AB AC : Type) (θAB θAC θA : angle),
  AB = AC → measure θAB = measure θAC

-- Triangle angle sum property: sum of angles in a triangle is 180 degrees
axiom triangle_angle_sum :
  ∀ {A B C : Type} (θA θB θC : angle),
  measure θA + measure θB + measure θC = 180

-- Exterior angle theorem: exterior angle is the sum of the two opposite interior angles
axiom exterior_angle_theorem :
  ∀ {A B C D : Type} (θAB θBC θAD : angle),
  measure θAD = measure θAB + measure θBC

-- Problem statement
theorem measure_angle_PQR :
  ∀ (R S P Q : Type) (RSP : line R S P) (QSP : angle) (h_QSP : measure QSP = 80)
  (PS SQ : Type) (h_PS_SQ : PS = SQ) (PQ RS : Type)
  (θPQ θRS θQR : angle)
  (h_straight_line : RSP = line R S P)
  (h_isosceles1 : isosceles_triangle PQ RS θPQ θRS θQR)
  (h_isosceles2 : isosceles_triangle PS SQ θPQ θQR θRS),
  measure θQR = 90 :=
sorry

end measure_angle_PQR_l491_491511


namespace area_of_triangle_XYZ_l491_491087

-- Define the problem statement
theorem area_of_triangle_XYZ 
  {X Y Z : Type} 
  [incidence_geometry Y] 
  (hXY : distance X Y = 8 * sqrt 2) 
  (hXZ : ∠ X Y Z = ∠ Z Y X)
  (right_triangle_XYZ : is_right_triangle X Y Z) :
  area (triangle X Y Z) = 64 := 
by
  sorry

end area_of_triangle_XYZ_l491_491087


namespace arithmetic_sequence_properties_l491_491508

/-- In the arithmetic sequence {a_n}, S_n is the sum of its first n terms. 
Given S_6 < S_7 and S_7 > S_8, prove:
1. The common difference d < 0;
2. S_9 < S_6;
3. S_7 is the maximum value among all S_n.
/-- 
theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : S 6 < S 7)
  (h2 : S 7 > S 8)
  (S_def : ∀ n, S n = ∑ i in finset.range n, a (i + 1)):
  let d := a 8 - a 7 in
  d < 0 ∧ S 9 < S 6 ∧ ∀ m, S 7 ≥ S m :=
by
  sorry

end arithmetic_sequence_properties_l491_491508


namespace log_base_16_of_4_eq_half_l491_491735

noncomputable def logBase (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_base_16_of_4_eq_half :
  logBase 16 4 = 1 / 2 := by
sorry

end log_base_16_of_4_eq_half_l491_491735


namespace Maxim_born_in_2008_probability_l491_491155

def birth_interval_start := ⟨2007, 9, 2⟩ -- September 2, 2007
def birth_interval_end := ⟨2008, 8, 31⟩ -- August 31, 2008
def days_in_interval := 365
def days_in_2008 := 244

theorem Maxim_born_in_2008_probability :
  (days_in_2008 : ℝ) / (days_in_interval : ℝ) = 244 / 365 :=
sorry

end Maxim_born_in_2008_probability_l491_491155


namespace sqrt_10_cos_theta_l491_491443

def vector (α : Type*) := (α × α)

variables {α : Type*} [linear_ordered_field α]

-- Define the vectors a and b
def a : vector α := (3, 3)
def b : vector α := (1, 2)

-- Define the angle θ between vectors a and b
noncomputable def θ := real.angle (a.1, a.2) (b.1, b.2)

-- The value to be proved: sqrt 10 * cos θ = 3
theorem sqrt_10_cos_theta :
  real.sqrt 10 * real.cos θ = 3 := 
sorry

end sqrt_10_cos_theta_l491_491443


namespace max_area_equilateral_triangle_in_rectangle_l491_491948

def rectangle_sides (length : ℝ) (width : ℝ) (A B C D: ℝ × ℝ): Prop :=
  A = (0, 0) ∧ B = (length, 0) ∧ C = (length, width) ∧ D = (0, width)

def equilateral_triangle (A B C : ℝ × ℝ): Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

def fits_within_rectangle (A B C : ℝ × ℝ) (length width : ℝ) : Prop :=
  ∀ (v : ℝ × ℝ), v = A ∨ v = B ∨ v = C → (0 ≤ v.1 ∧ v.1 ≤ length ∧ 0 ≤ v.2 ∧ v.2 ≤ width)

theorem max_area_equilateral_triangle_in_rectangle :
  ∀ (length width : ℝ) (A B C D P Q R : ℝ × ℝ),
  rectangle_sides length width A B C D →
  length = 11 → width = 10 →
  equilateral_triangle P Q R →
  fits_within_rectangle P Q R length width →
  ∃ (area : ℝ), area = 221 * real.sqrt 3 - 330 :=
by
  sorry

end max_area_equilateral_triangle_in_rectangle_l491_491948


namespace sector_arc_length_l491_491011

noncomputable def arc_length (R : ℝ) (θ : ℝ) : ℝ :=
  θ / 180 * Real.pi * R

theorem sector_arc_length
  (central_angle : ℝ) (area : ℝ) (arc_length_answer : ℝ)
  (h1 : central_angle = 120)
  (h2 : area = 300 * Real.pi) :
  arc_length_answer = 20 * Real.pi :=
by
  sorry

end sector_arc_length_l491_491011


namespace intersect_line_segment_l491_491795

variables (A B : Point) (l : Line) (k : ℝ)

def point_A : Point := ⟨1, 3⟩
def point_B : Point := ⟨-2, -1⟩
def line_l (k : ℝ) : Line := 
  fun x => k * (x - 2) + 1

theorem intersect_line_segment (k : ℝ) :
  ∃ P Q : Point, P = point_A ∧ Q = point_B ∧ 
  P.y = k * (P.x - 2) + 1 ∧
  Q.y = k * (Q.x - 2) + 1 →
  -2 ≤ k ∧ k ≤ 0.5 :=
sorry

end intersect_line_segment_l491_491795


namespace percentage_of_tree_A_l491_491177

theorem percentage_of_tree_A
  (a b : ℕ)
  (h1 : a + b = 10)
  (h2 : 6 * a + 5 * b = 55) :
  a / 10 = 0.5 :=
by
  sorry

end percentage_of_tree_A_l491_491177


namespace basketball_team_selection_l491_491578

theorem basketball_team_selection:
  let n := 16
  let quadruplets := {Ben, Bob, Bill, Bert: String}
  let choose := Nat.choose
  (∃ (players: Finset String), players.card = n ∧ quadruplets ⊆ players) →
  ((quadruplets.card == 4) →
    (choose 4 3 * choose 12 3 = 880)) :=
by
  intros n quadruplets choose players h_cards h_quadruplets
  rw [h_quadruplets]
  -- We can add the detailed computation here, but skipping to keep proof correct.
  exact sorry

end basketball_team_selection_l491_491578


namespace probability_at_least_one_even_l491_491650

theorem probability_at_least_one_even :
  let usable_digits := {0, 3, 5, 7, 8, 9}
  let even_digits := {0, 8}
  let code_length := 4
  let num_usable_digits := 6
  let num_usable_odd_digits := 4
  (1 - ((num_usable_odd_digits ^ code_length) / (num_usable_digits ^ code_length))) = (65 / 81) := by
  sorry

end probability_at_least_one_even_l491_491650


namespace find_m_plus_n_l491_491594

def area_of_triangle (a b c h_a h_b h_c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  (Math.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem find_m_plus_n (a b c : ℝ) (h_a : a * 42 = 39 * b) (h_b : b * 36.4 = 36.4 * c) (h_c : c * 39 = 42 * a) :
  let area := area_of_triangle a b c 42 39 36.4 in
  ∃ m n : ℤ, Nat.gcd m n = 1 ∧ (2 * area = (m : ℝ)) ∧ (2 * 42 * 36.4 * (1 / a) = (n : ℝ)) ∧ (m + n) = 3553 :=
sorry

end find_m_plus_n_l491_491594


namespace brad_unsold_glasses_l491_491330

theorem brad_unsold_glasses
  (yield_per_gallon : ℕ := 16)
  (cost_per_gallon : ℝ := 3.50)
  (gallons_made : ℕ := 2)
  (price_per_glass : ℕ := 1)
  (glasses_drank : ℕ := 5)
  (net_profit : ℝ := 14) :
  let total_glasses_produced := yield_per_gallon * gallons_made in
  let total_glasses_for_sale := total_glasses_produced - glasses_drank in
  let total_cost := cost_per_gallon * gallons_made in
  let total_revenue := total_cost + net_profit in
  let glasses_sold := total_revenue / price_per_glass in
  let unsold_glasses := total_glasses_for_sale - glasses_sold in
  unsold_glasses = 6 :=
by
  sorry

end brad_unsold_glasses_l491_491330


namespace family_ages_l491_491497

-- Define the conditions
variables (D M S F : ℕ)

-- Condition 1: In the year 2000, the mother was 4 times the daughter's age.
axiom mother_age : M = 4 * D

-- Condition 2: In the year 2000, the father was 6 times the son's age.
axiom father_age : F = 6 * S

-- Condition 3: The son is 1.5 times the age of the daughter.
axiom son_age_ratio : S = 3 * D / 2

-- Condition 4: In the year 2010, the father became twice the mother's age.
axiom father_mother_2010 : F + 10 = 2 * (M + 10)

-- Condition 5: The age gap between the mother and father has always been the same.
axiom age_gap_constant : F - M = (F + 10) - (M + 10)

-- Define the theorem
theorem family_ages :
  D = 10 ∧ S = 15 ∧ M = 40 ∧ F = 90 ∧ (F - M = 50) := sorry

end family_ages_l491_491497


namespace find_a_and_b_l491_491840

theorem find_a_and_b (a b : ℚ) (h : ∀ (n : ℕ), 1 / ((2 * n - 1) * (2 * n + 1)) = a / (2 * n - 1) + b / (2 * n + 1)) : 
  a = 1/2 ∧ b = -1/2 := 
by 
  sorry

end find_a_and_b_l491_491840


namespace paul_spent_81_90_l491_491170

-- Define the original price of each racket
def originalPrice : ℝ := 60

-- Define the discount rates
def firstDiscount : ℝ := 0.20
def secondDiscount : ℝ := 0.50

-- Define the sales tax rate
def salesTax : ℝ := 0.05

-- Define the prices after discount
def firstRacketPrice : ℝ := originalPrice * (1 - firstDiscount)
def secondRacketPrice : ℝ := originalPrice * (1 - secondDiscount)

-- Define the total price before tax
def totalPriceBeforeTax : ℝ := firstRacketPrice + secondRacketPrice

-- Define the total sales tax
def totalSalesTax : ℝ := totalPriceBeforeTax * salesTax

-- Define the total amount spent
def totalAmountSpent : ℝ := totalPriceBeforeTax + totalSalesTax

-- The statement to prove
theorem paul_spent_81_90 : totalAmountSpent = 81.90 := 
by
  sorry

end paul_spent_81_90_l491_491170


namespace polygon_diagonal_theorem_l491_491420

-- Defining what it means to have a convex n-gon with one side of length 1 and integer diagonals.
def convex_ngon_has_integer_diagonals (n : ℕ) :=
  ∃ (vertices : Fin n → ℝ × ℝ),
    convex vertices ∧
    (side_length vertices 0 1 = 1) ∧
    (∀ (i j : Fin n), i ≠ j → ∃ k, length (vertices i, vertices k) = length (vertices k, vertices j) ∧ is_integer (length (vertices i, vertices j)))

-- The main theorem
theorem polygon_diagonal_theorem :
  ∀ n, convex_ngon_has_integer_diagonals n ↔ n = 4 ∨ n = 5 :=
by
  sorry

end polygon_diagonal_theorem_l491_491420


namespace cos_double_angle_l491_491551
noncomputable section

variables {α : Type*} [linearOrderedField α]

def vec_a (α : ℝ) : ℝ × ℝ :=
  (Math.cos α, 1/2)

theorem cos_double_angle (α : ℝ) (h : (Math.cos α)^2 + (1/2)^2 = 1/2^2) :
  Math.cos (2 * α) = -1/2 :=
by
  sorry

end cos_double_angle_l491_491551


namespace probability_maxim_born_in_2008_l491_491148

def days_in_month : ℕ → ℕ → ℕ 
| 2007 9 := 29
| 2007 10 := 31
| 2007 11 := 30
| 2007 12 := 31
| 2008 1 := 31
| 2008 2 := 29   -- leap year
| 2008 3 := 31
| 2008 4 := 30
| 2008 5 := 31
| 2008 6 := 30
| 2008 7 := 31
| 2008 8 := 31
| _ _ := 0

def total_days : ℕ :=
list.sum [days_in_month 2007 9, days_in_month 2007 10, days_in_month 2007 11, days_in_month 2007 12,
          days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

def days_in_2008 : ℕ :=
list.sum [days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

theorem probability_maxim_born_in_2008 : 
  (days_in_2008 : ℚ) / (total_days : ℚ) = 244 / 365 :=
by
  -- Placeholder for actual proof. This theorem is set properly and ready for proof development.
  sorry

end probability_maxim_born_in_2008_l491_491148


namespace find_circle_dimensions_l491_491597

noncomputable def circle_problem 
  (H: ℝ) (BC: ℝ) (DH: ℝ) (HC: ℝ) 
  (r: ℝ) (R: ℝ) 
  (ω_radius: ℝ) (Ω_radius: ℝ) 
  (radius_ω: ω_radius = r) 
  (radius_Ω: Ω_radius = R) 
  (BC_len: BC = 5) 
  (DH_len: DH = 2) 
  (HC_len: HC = 2)
  : Prop :=
  let HP := H in 
  let expected_HP_length := (5 * Real.sqrt 14) / 2 in
  let expected_r := 3 * Real.sqrt 7 / Real.sqrt 10 in
  let expected_R := 3 * Real.sqrt 35 / Real.sqrt 2 in
  HP = expected_HP_length ∧ ω_radius = expected_r ∧ Ω_radius = expected_R

-- Statement that H, ω_radius, and Ω_radius match the calculated values
theorem find_circle_dimensions 
  (HP: ℝ)
  (ω_radius: ℝ)
  (Ω_radius: ℝ)
  : circle_problem HP 5 2 2 ω_radius Ω_radius := 
by 
  sorry

end find_circle_dimensions_l491_491597


namespace both_players_can_force_tie_l491_491323

def player : Type := ℕ -- Represent players as natural numbers (0: Ana, 1: Benito)
def card : Type := ℕ   -- Cards are identified by natural numbers

-- Defining initial condition: Ana has card 0
def initial_card_distribution (p : player) (c : card) : Prop :=
  p = 0 ∧ c = 0

-- Defining the function that represents the possible state of the game after each round
def game_state (rounds : ℕ) (ana_score benito_score : ℕ) : Prop :=
  rounds = 2020 ∧ ana_score = benito_score

/-- Both players can force a tie in the game starting from the initial state. -/
theorem both_players_can_force_tie :
  ∃ strategy_ana strategy_benito : ℕ → ℕ → Prop,
    (∀ rounds ana_score benito_score,
      initial_card_distribution 0 0 →
      game_state rounds ana_score benito_score) →
    ana_score = benito_score :=
by
  sorry

end both_players_can_force_tie_l491_491323


namespace doris_weeks_to_meet_expenses_l491_491726

def doris_weekly_hours : Nat := 5 * 3 + 5 -- 5 weekdays (3 hours each) + 5 hours on Saturday
def doris_hourly_rate : Nat := 20 -- Doris earns $20 per hour
def doris_weekly_earnings : Nat := doris_weekly_hours * doris_hourly_rate -- The total earnings per week
def doris_monthly_expenses : Nat := 1200 -- Doris's monthly expense

theorem doris_weeks_to_meet_expenses : ∃ w : Nat, doris_weekly_earnings * w ≥ doris_monthly_expenses ∧ w = 3 :=
by
  sorry

end doris_weeks_to_meet_expenses_l491_491726


namespace trapezoid_area_correct_l491_491248

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Trapezoid :=
  (A B C D : Point)

def area_of_trapezoid (AB CD h : ℝ) : ℝ :=
  (1 / 2) * (AB + CD) * h

-- Define the vertices of the trapezoid
def A : Point := ⟨0, 0⟩
def B : Point := ⟨0, -2⟩
def C : Point := ⟨4, 0⟩
def D : Point := ⟨4, 6⟩

-- Create the Trapezoid ABCD
def trapezoid_ABCD : Trapezoid := ⟨A, B, C, D⟩

-- Define the distances (lengths of bases and height)
def length_AB : ℝ := abs (B.y - A.y)
def length_CD : ℝ := abs (D.y - C.y)
def height_AC : ℝ := abs (C.x - A.x)

-- Calculating the area
def calculated_area : ℝ := area_of_trapezoid length_AB length_CD height_AC

theorem trapezoid_area_correct :
  calculated_area = 16 := by
  sorry

end trapezoid_area_correct_l491_491248


namespace Josephine_sold_10_liters_l491_491164

def milk_sold (n1 n2 n3 : ℕ) (v1 v2 v3 : ℝ) : ℝ :=
  (v1 * n1) + (v2 * n2) + (v3 * n3)

theorem Josephine_sold_10_liters :
  milk_sold 3 2 5 2 0.75 0.5 = 10 :=
by
  sorry

end Josephine_sold_10_liters_l491_491164


namespace inequality_proof_l491_491618

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : y > 2 / (x - y)) : x^2 > y^2 + 4 :=
by
  sorry

end inequality_proof_l491_491618


namespace linear_equation_example_l491_491265

theorem linear_equation_example
  (A : ℝ → ℝ → Prop := λ x y, x^2 + x = 3)
  (B : ℝ → ℝ → Prop := λ x y, 5 * x + 2 * x = 5 * y + 3)
  (C : ℝ → ℝ → Prop := λ x y, (1 / 2) * x - 9 = 3)
  (D : ℝ → ℝ → Prop := λ x y, 2 / (x + 1) = 2) :
  ∀ (x : ℝ), C x 0 → 
  ∀ (y : ℝ), B x y → ¬A x y ∧ ¬D x y :=
by
  -- Proof steps would be inserted here
  sorry

end linear_equation_example_l491_491265


namespace perimeters_equal_l491_491529

variable {A B C T M N X Y : Type} [EquilateralTriangle A B C]
  (T_on_AC_and_arcs_AB_BC : T ∈ AC ∩ (arc AB ∩ arc BC))
  (MT_parallel_BC : MT ∥ BC)
  (NT_parallel_AB : NT ∥ AB)
  (AN_intersects_MT_at_X : ∃ X, intersect AN MT = X)
  (CM_intersects_NT_at_Y : ∃ Y, intersect CM NT = Y)

theorem perimeters_equal (polys : (polygon [A, X, Y, C] = polygon [X, M, B, N, Y])) :
  perimeter (polygon [A, X, Y, C]) = perimeter (polygon [X, M, B, N, Y]) := sorry

end perimeters_equal_l491_491529


namespace main_theorem_l491_491868

noncomputable def rectangular_equation_C : String :=
  "x^2 = 4y"

noncomputable def ordinary_equation_l : String :=
  "y = x + 1"

noncomputable def distance_MN : ℝ :=
  8

def proof_problem : Prop :=
  let C := "polar equation: ρ * sin(θ)^2 + 4 * sin(θ) - ρ = 0"
  let l := "parametric equations: x = 2 + √2/2 * t, y = 3 + √2/2 * t"
  rectangular_equation_C ∧ ordinary_equation_l ∧ (distance_MN = 8)

theorem main_theorem : proof_problem :=
  by sorry

end main_theorem_l491_491868


namespace find_certain_number_l491_491238

theorem find_certain_number 
  (x : ℝ) 
  (h : ( (x + 2 - 6) * 3 ) / 4 = 3) 
  : x = 8 :=
by
  sorry

end find_certain_number_l491_491238


namespace minimize_sum_of_products_l491_491899

-- Define points in a plane
variables {Point : Type} [inner_product_space ℝ Point]

-- Define triangle vertices and centroid 
variables {A B C G : Point}

-- Define squared distances 
noncomputable def squared_distance (P Q : Point) : ℝ := ⟪P - Q, P - Q⟫ 

-- Given a triangle ABC with centroid G
def centroid (A B C : Point) : Point := (A + B + C) / 3

theorem minimize_sum_of_products 
  (A B C : Point) (G := centroid A B C) :
  ∀ P : Point, 
    (squared_distance P A * squared_distance G A +
     squared_distance P B * squared_distance G B +
     squared_distance P C * squared_distance G C) ≥
    (squared_distance A G ^ 2 + squared_distance B G ^ 2 + squared_distance C G ^ 2) / 9 :=
by sorry

end minimize_sum_of_products_l491_491899


namespace min_length_MN_l491_491945

theorem min_length_MN (a b : ℝ) (H h : ℝ) (MN : ℝ) (midsegment_eq_4 : (a + b) / 2 = 4)
    (area_div_eq_half : (a + MN) / 2 * h = (MN + b) / 2 * H) : MN = 4 :=
by
  sorry

end min_length_MN_l491_491945


namespace value_of_a_l491_491482

noncomputable def number : ℕ := 21 * 25 * 45 * 49

theorem value_of_a (a : ℕ) (h : a^3 = number) : a = 105 :=
sorry

end value_of_a_l491_491482


namespace Q_value_2023_l491_491889

noncomputable def Q (n : ℕ) : ℚ :=
  (list.range (n - 2)).map (λ k, 1 - (1 : ℚ) / (↑k + 3)).prod

theorem Q_value_2023 : Q 2023 = 2 / 2023 :=
by
  sorry

end Q_value_2023_l491_491889


namespace train_speed_proof_l491_491685

-- Define the conditions
def length_of_train := 100 -- in metres
def length_of_bridge := 275 -- in metres
def time_cross_bridge := 30 -- in seconds

-- Define the expected speed calculation (in km/hr)
def expected_speed_km_hr := 45

-- Prove the speed of the train is 45 km/hr given the conditions
theorem train_speed_proof : (length_of_train + length_of_bridge) / time_cross_bridge * 3.6 = expected_speed_km_hr :=
by
  sorry

end train_speed_proof_l491_491685


namespace round_robin_tournament_l491_491857

theorem round_robin_tournament :
  ∀ (a b : Fin 18 → ℕ),
    (∀ i, a i + b i = 17) →
    (∑ i, a i = ∑ i, b i) →
    (∑ i, (a i) ^ 2 = ∑ i, (b i) ^ 2) :=
by
  intro a b h1 h2
  sorry

end round_robin_tournament_l491_491857


namespace modulus_z2_sqrt_2_l491_491808

variable (a : ℂ) (i : ℂ)

noncomputable def z1 := (1 : ℂ) + i
noncomputable def z2 := a - i

theorem modulus_z2_sqrt_2 (h : z1 = (1 + i) ∧ z2 = (a - i) ∧ (z1 * complex.conj z2).im = 0) :
  complex.abs z2 = real.sqrt 2 :=
sorry

end modulus_z2_sqrt_2_l491_491808


namespace medial_triangle_similar_to_original_l491_491101

-- Define the given triangle ABC with vertices A, B, C
variables {A B C D E F : Type} [Inhabited D] [Inhabited E] [Inhabited F]

-- Define that D, E, F are midpoints of sides BC, CA, AB respectively
def midpoints (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  let D := midpoint B C
  let E := midpoint C A
  let F := midpoint A B in
  true 

-- Define the medial triangle DEF
noncomputable def medial_triangle (A B C D E F : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F] 
  (h1 : midpoints A B C) : Type :=
DEF

-- Prove that the medial triangle DEF is similar to the original triangle ABC
theorem medial_triangle_similar_to_original (A B C D E F : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace F]
  (h1 : midpoints A B C) : 
  similar_triangles A B C D E F (medial_triangle A B C D E F h1) := 
sorry

end medial_triangle_similar_to_original_l491_491101


namespace log_a10_eq_5_l491_491954

theorem log_a10_eq_5 (a : ℕ → ℝ) (r : ℝ) (h1: ∀ n, a (n+1) = r * a n) (h2 : 0 < r) (h3: a 3 * a 11 = 16) (h4: ∀ n, 0 < a n) :
  log 2 (a 10) = 5 :=
by
  sorry

end log_a10_eq_5_l491_491954


namespace probability_born_in_2008_l491_491145

/-- Define the relevant dates and days calculation -/
def num_days_between : ℕ :=
  29 + 31 + 30 + 31 + 31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

def num_days_2008 : ℕ :=
  31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

theorem probability_born_in_2008 :
  (num_days_2008 : ℚ) / num_days_between = 244 / 365 :=
by
  -- The proof is omitted for brevity
  sorry

end probability_born_in_2008_l491_491145


namespace term_in_sequence_l491_491432

noncomputable theory
def seq (n : ℕ) : ℕ :=
if n = 1 then 
  1 
else 
  ((2:ℕ) ^ (n-1)) * seq (n-1)

theorem term_in_sequence (n : ℕ) : 
  ∃ n, seq n = 64 :=
by
  sorry

end term_in_sequence_l491_491432


namespace mary_income_l491_491989

variable (J T M : ℝ)

axiom condition1 : M = 1.5 * T
axiom condition2 : T = 0.6 * J

theorem mary_income : M = 0.9 * J := 
by
  have h1 : T = 0.6 * J := condition2
  have h2 : M = 1.5 * T := condition1
  calc
    M = 1.5 * T    : h2
    ... = 1.5 * (0.6 * J) : by rw h1
    ... = 0.9 * J : by norm_num [mul_assoc] 

end mary_income_l491_491989


namespace probability_heads_die_l491_491563

theorem probability_heads_die :
  ∀ (coin_flip : ℕ → bool) (roll_die : ℕ → ℕ),
    (∀ n, (coin_flip n = tt ∨ coin_flip n = ff)) → -- Coin results are either heads (tt) or tails (ff)
    (∀ n, (coin_flip n = tt) ↔ (coin_flip 0 = tt)) → -- Coin flips are independent and fair
    (∀ n, (1 ≤ roll_die n ∧ roll_die n ≤ 6) → -- Die has 6 sides
      (roll_die n = 5 ∨ roll_die n = 6) ↔ (roll_die n > 4)) → -- Number greater than 4 is either 5 or 6
    ∃ (p : ℚ), p = 1 / 6 :=
begin
  sorry
end

end probability_heads_die_l491_491563


namespace apples_in_basket_l491_491080

theorem apples_in_basket (A : ℕ) (rotten_percent : ℝ) (good_apples : ℕ) 
    (h1 : rotten_percent = 0.12) 
    (h2 : good_apples = 66) 
    (h3 : 0.88 * A = ↑good_apples) : 
    A = 75 :=
by
  sorry

end apples_in_basket_l491_491080


namespace who_is_correct_highest_water_level_l491_491213

-- Define the water levels over the week
def water_level_changes : List Float :=
  [0.0, -0.5, 1.5, 0.5, -1.0, -0.3, 0.5]

-- Define the initial water level and warning level
def initial_water_level : Float := 30.0
def warning_level : Float := 32.0

-- Compute the water level for each day
def water_levels : List Float :=
  List.scanl (+) initial_water_level water_level_changes

-- Statements to be proven
theorem who_is_correct : "Xiao Li" = "Xiao Li" := by
  sorry

theorem highest_water_level : (List.maximum water_levels, List.maximum water_levels < warning_level) = (some 31.5, true) := by
  sorry

end who_is_correct_highest_water_level_l491_491213


namespace smallest_polynomial_degree_l491_491932

theorem smallest_polynomial_degree 
    (r1 r2 r3 r4 r5 r6 : ℚ√ℚ)
    (h1 : r1 = 3 + 2 * real.sqrt 3 ∨ r1 = 3 - 2 * real.sqrt 3)
    (h2 : r2 = -3 - 2 * real.sqrt 3 ∨ r2 = -3 + 2 * real.sqrt 3)
    (h3 : r3 = 2 + real.sqrt 5 ∨ r3 = 2 - real.sqrt 5)
    (h4 : r4 = 3 + 2 * real.sqrt 3)
    (h5 : r5 = -3 - 2 * real.sqrt 3)
    (h6 : r6 = 2 + real.sqrt 5):
    ∃ (p : ℚ[X]), p.degree = 6 ∧ 
    ∀ x, p.is_root x → (x = 3 + 2 * real.sqrt 3 ∨ x = 3 - 2 * real.sqrt 3 
    ∨ x = -3 - 2 * real.sqrt 3 ∨ x = -3 + 2 * real.sqrt 3 ∨ x = 2 + real.sqrt 5 ∨ x = 2 - real.sqrt 5) := 
by 
    sorry

end smallest_polynomial_degree_l491_491932


namespace compare_negative_fractions_l491_491357

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l491_491357


namespace solve_for_a_l491_491010

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x >= 0 then 4^x else 2^(a - x)

theorem solve_for_a (a : ℝ) (h : a ≠ 1) (h_f_eq : f a (1 - a) = f a (a - 1)) : a = 1 / 2 :=
by
  sorry

end solve_for_a_l491_491010


namespace sqrt_of_0_09_l491_491958

theorem sqrt_of_0_09 : Real.sqrt 0.09 = 0.3 :=
by
  -- Mathematical problem restates that the square root of 0.09 equals 0.3
  sorry

end sqrt_of_0_09_l491_491958


namespace remainder_when_divided_by_x_minus_2_l491_491410

def p (x : ℕ) : ℕ := x^5 - 2 * x^3 + 4 * x + 5

theorem remainder_when_divided_by_x_minus_2 : p 2 = 29 := 
by {
  sorry
}

end remainder_when_divided_by_x_minus_2_l491_491410


namespace range_of_y_l491_491624

noncomputable def y (x : ℝ) : ℝ := sin x - abs (sin x)

theorem range_of_y : Set.range y = Set.Icc (-2 : ℝ) 0 :=
by
  sorry

end range_of_y_l491_491624


namespace quadratic_always_positive_l491_491949

theorem quadratic_always_positive (x : ℝ) : x^2 + x + 1 > 0 :=
sorry

end quadratic_always_positive_l491_491949


namespace smaller_number_between_5_and_8_l491_491703

theorem smaller_number_between_5_and_8 :
  min 5 8 = 5 :=
by
  sorry

end smaller_number_between_5_and_8_l491_491703


namespace avg_visitors_is_correct_l491_491270

-- Define the number of days in the month
def days_in_month : ℕ := 30

-- Define the average number of visitors on Sundays
def avg_visitors_sunday : ℕ := 510

-- Define the average number of visitors on other days
def avg_visitors_other_days : ℕ := 240

-- Define the number of Sundays in the month
def sundays_in_month : ℕ := 4

-- Define the number of other days in the month
def other_days_in_month : ℕ := days_in_month - sundays_in_month

-- Define the total visitors on Sundays
def total_visitors_sundays : ℕ := sundays_in_month * avg_visitors_sunday

-- Define the total visitors on other days
def total_visitors_other_days : ℕ := other_days_in_month * avg_visitors_other_days

-- Define the total number of visitors in the month
def total_visitors : ℕ := total_visitors_sundays + total_visitors_other_days

-- Define the average number of visitors per day
def avg_visitors_per_day : ℕ := total_visitors / days_in_month

-- The theorem to prove
theorem avg_visitors_is_correct : avg_visitors_per_day = 276 := by
  sorry

end avg_visitors_is_correct_l491_491270


namespace eq_solutions_count_l491_491407

theorem eq_solutions_count : 
  (set.count {x : ℝ | 8^(x^2 - 4*x + 3) = 1}) = 2 :=
begin
  sorry
end

end eq_solutions_count_l491_491407


namespace total_people_in_line_l491_491984

theorem total_people_in_line (n : ℕ) (h : n = 5): n + 2 = 7 :=
by
  -- This is where the proof would normally go, but we omit it with "sorry"
  sorry

end total_people_in_line_l491_491984


namespace personC_start_time_l491_491325

-- Define constants and parameters
def A : ℝ := 0
def B : ℝ := 1
def C : ℝ := 1/3
def D : ℝ := 2/3

-- Define the times at which each person starts
def startA : ℝ := 8 + 0 / 60
def startB : ℝ := 8 + 12 / 60
def meetAandB : ℝ := 8 + 24 / 60
def meetAandC : ℝ := 8 + 30 / 60

-- Define the walking speeds
def speedA := (C - A) / (meetAandB - startA)
def speedB := (C - B) / (meetAandB - startB)
def speedC := (D - B) / (meetAandB - startB) * 3

-- Definition of Person C's start time to be proven
def startC := meetAandB - (D - B) / speedB * 3

-- The theorem to prove
theorem personC_start_time : startC = 8 + 16 / 60 := sorry

end personC_start_time_l491_491325


namespace fraction_of_married_women_correct_l491_491327

noncomputable def fraction_married_women (total_men : ℕ) (_ : total_men = 7)
  (prob_single : ℚ) (h : prob_single = 3 / 7) : ℚ :=
  let single_men := total_men * prob_single
  let married_men := total_men - single_men
  let married_women := married_men
  let total_people := total_men + married_women
  in married_women / total_people

theorem fraction_of_married_women_correct : 
  fraction_married_women 7 (by rfl) (3/7) (by rfl) = 4 / 11 :=
  sorry

end fraction_of_married_women_correct_l491_491327


namespace matrixSequenceProductCorrect_l491_491333

open Matrix

-- Define the sequence of matrices
def matrixSeq (k : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  if h : k > 0 ∧ k ≤ 50 then 
    ![![1, (2 * k : ℤ)],![0, 1]]
  else 
    1 -- Identity matrix for purposes of definition outside bounds

-- Define the product over the matrix sequence
def matrixProduct : Matrix (Fin 2) (Fin 2) ℤ :=
  List.prod $ List.map matrixSeq (List.range 1 51)

-- The expected resultant matrix
def expectedMatrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2550],![0, 1]]

-- The theorem to be proven
theorem matrixSequenceProductCorrect : matrixProduct = expectedMatrix := 
  sorry

end matrixSequenceProductCorrect_l491_491333


namespace conjugate_in_fourth_quadrant_l491_491008

open Complex

theorem conjugate_in_fourth_quadrant (z : ℂ) (h : (1 - 2 * complex.I) * z = complex.abs (2 * real.sqrt 2 + complex.I) - complex.I) :
  0 < z.conj.re ∧ z.conj.im < 0 := 
sorry

end conjugate_in_fourth_quadrant_l491_491008


namespace determine_base_l491_491499

theorem determine_base (r : ℕ) (a b x : ℕ) (h₁ : r ≤ 100) 
  (h₂ : x = a * r + a) (h₃ : a < r) (h₄ : a > 0) 
  (h₅ : x^2 = b * r^3 + b) : r = 2 ∨ r = 23 :=
by
  sorry

end determine_base_l491_491499


namespace properSubsets_of_set_012_l491_491946

def properSubsetsCount {α : Type*} (S : set α) : ℕ :=
  if h : S = ∅ then 0
  else
    let n := S.to_finset.card in
    2 ^ n - 1

theorem properSubsets_of_set_012 :
  properSubsetsCount {0, 1, 2} = 7 :=
by sorry

end properSubsets_of_set_012_l491_491946


namespace log_base_16_of_4_l491_491741

theorem log_base_16_of_4 : 
  (16 = 2^4) →
  (4 = 2^2) →
  (∀ (b a c : ℝ), b > 0 → b ≠ 1 → c > 0 → c ≠ 1 → log b a = log c a / log c b) →
  log 16 4 = 1 / 2 :=
by
  intros h1 h2 h3
  sorry

end log_base_16_of_4_l491_491741


namespace max_f_value_l491_491936

noncomputable def f (x : ℝ) : ℝ := 2^x + 2 - 3 * 4^x

def domain_M (x : ℝ) : Prop := 3 - 4 * x + x^2 > 0

theorem max_f_value : ∃ x, domain_M x ∧ ∀ y, domain_M y → f y ≤ f x := by
  use 3.5  -- Example value within the domain
  intro y
  sorry  -- Placeholder for the actual proof

end max_f_value_l491_491936


namespace find_number_l491_491844

theorem find_number (x n : ℝ) (h1 : 0.12 / x * n = 12) (h2 : x = 0.1) : n = 10 := by
  sorry

end find_number_l491_491844


namespace line_passes_through_fixed_point_l491_491829

-- Defining the vectors
def a (k : ℝ) : ℝ × ℝ := (k + 2, 1)
def b (b : ℝ) : ℝ × ℝ := (-b, 1)

-- Condition for collinearity
def collinear (k b : ℝ) : Prop := a k.1 * b b.2 = a k.2 * b b.1

-- The line equation
def line_eq (k b x : ℝ) : ℝ := k * x + b

-- Statement to be proved
theorem line_passes_through_fixed_point (k b : ℝ) (hk : k + 2 = -b) :
  line_eq k b 1 = -2 :=
by
  sorry

end line_passes_through_fixed_point_l491_491829


namespace eating_relationship_l491_491259

open Set

-- Definitions of the sets A and B
def A : Set ℝ := {-1, 1/2, 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≥ 0 ∧ a * x^2 = 1}

-- Definitions of relationships
def full_eating (A B : Set ℝ) : Prop := A ⊆ B ∨ B ⊆ A
def partial_eating (A B : Set ℝ) : Prop := ∃ x, x ∈ A ∧ x ∈ B ∧ ¬ full_eating A B

-- Main theorem
theorem eating_relationship (a : ℝ) :
  (full_eating A (B a) ∨ partial_eating A (B a)) ↔ (a = 0 ∨ a = 1 ∨ a = 4) := by
  sorry

end eating_relationship_l491_491259


namespace buttons_ratio_l491_491557

theorem buttons_ratio
  (initial_buttons : ℕ)
  (shane_multiplier : ℕ)
  (final_buttons : ℕ)
  (total_buttons_after_shane : ℕ) :
  initial_buttons = 14 →
  shane_multiplier = 3 →
  final_buttons = 28 →
  total_buttons_after_shane = initial_buttons + shane_multiplier * initial_buttons →
  (total_buttons_after_shane - final_buttons) / total_buttons_after_shane = 1 / 2 :=
by
  intros
  sorry

end buttons_ratio_l491_491557


namespace sequence_sum_l491_491784

theorem sequence_sum (x : ℕ → ℝ) (d : ℝ) 
  (h_harmonic : ∀ n : ℕ, n > 0 → (1 / x (n + 1) - 1 / x n = d)) 
  (h_sum : (finset.range 20).sum (λ n, x (n + 1)) = 200) :
  x 5 + x 16 = 20 :=
sorry

end sequence_sum_l491_491784


namespace Amanda_ticket_sales_goal_l491_491693

theorem Amanda_ticket_sales_goal :
  let total_tickets : ℕ := 80
  let first_day_sales : ℕ := 5 * 4
  let second_day_sales : ℕ := 32
  total_tickets - (first_day_sales + second_day_sales) = 28 :=
by
  sorry

end Amanda_ticket_sales_goal_l491_491693


namespace count_integers_satisfying_inequalities_l491_491765

theorem count_integers_satisfying_inequalities :
  {n : Int | 3 ≤ n ∧ n ≤ 7 ∧ (Real.sqrt (3 * n - 1) ≤ Real.sqrt (5 * n - 7)) ∧ (Real.sqrt (5 * n - 7) < Real.sqrt (3 * n + 8))}.card = 5 := 
by 
  sorry

end count_integers_satisfying_inequalities_l491_491765


namespace cheryl_needed_first_material_l491_491197

noncomputable def cheryl_material (x : ℚ) : ℚ :=
  x + 1 / 3 - 3 / 8

theorem cheryl_needed_first_material
  (h_total_used : 0.33333333333333326 = 1 / 3) :
  cheryl_material x = 1 / 3 → x = 3 / 8 :=
by
  intros
  rw [h_total_used] at *
  sorry

end cheryl_needed_first_material_l491_491197


namespace inequality_solution_l491_491882

noncomputable def f (a b x : ℝ) : ℝ := 1 / Real.sqrt x + 1 / Real.sqrt (a + b - x)

theorem inequality_solution 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (x : ℝ) 
  (hx : x ∈ Set.Ioo (min a b) (max a b)) : 
  f a b x < f a b a ∧ f a b x < f a b b := 
sorry

end inequality_solution_l491_491882


namespace question_part_one_l491_491035

noncomputable def q1 (a : ℚ) (x : ℚ) : Prop := abs a * x = abs (a + 1) - x

theorem question_part_one :
  (∀ a : ℚ, q1 a 0 → a = -1) ∧ (∀ a : ℚ, q1 a 1 → (a ≥ 0)) :=
by
  split
  {
    intro a,
    intro h,
    -- Proof here
    sorry
  }
  {
    intro a,
    intro h,
    -- Proof here
    sorry
  }

end question_part_one_l491_491035


namespace shooters_hit_target_at_least_once_in_three_attempts_l491_491414

theorem shooters_hit_target_at_least_once_in_three_attempts 
  (p_shooter_hit_target : ℚ) 
  (h_p_shooter : p_shooter_hit_target = 2 / 3) 
  (num_shooters : ℕ) 
  (h_num_shooters : num_shooters = 5) 
  (num_attempts : ℕ) 
  (h_num_attempts : num_attempts = 3) 
  : (1 - (1 - (p_shooter_hit_target ^ num_shooters)) ^ num_attempts) = 1 - (1 - ((2 / 3) ^ 5)) ^ 3 :=
by
  -- First, we note p_shooter_hit_target = 2 / 3 by assumption.
  rw [h_p_shooter, h_num_shooters, h_num_attempts]
  sorry

end shooters_hit_target_at_least_once_in_three_attempts_l491_491414


namespace relationship_between_M_and_N_l491_491050

def M := { x : ℝ | 9^x - 4 * 3^(x+1) + 27 = 0 }
def N := { x : ℝ | log 2 (x + 1) + log 2 x = log 2 6 }

theorem relationship_between_M_and_N : N ⊂ M := by
  sorry

end relationship_between_M_and_N_l491_491050


namespace speed_of_second_train_is_30_l491_491970

-- Define the lengths of the trains and the speed of the first train
def length_train1 : ℝ := 140 / 1000 -- in kilometers
def length_train2 : ℝ := 280 / 1000 -- in kilometers
def speed_train1 : ℝ := 42 -- in kmph
def time_to_clear : ℝ := 20.99832013438925 / 3600 -- in hours

-- Define the total distance to be covered
def total_distance : ℝ := length_train1 + length_train2

-- Define the relative speed considering trains running towards each other
def relative_speed : ℝ := total_distance / time_to_clear

-- Define the question a term
def speed_of_second_train : ℝ := relative_speed - speed_train1

theorem speed_of_second_train_is_30 : speed_of_second_train = 30 :=
by
  calc
  speed_of_second_train = total_distance / time_to_clear - speed_train1 : by rfl
  ... = (0.14 + 0.28) / (20.99832013438925 / 3600) - 42 : by rfl
  ... = 0.42 / (20.99832013438925 / 3600) - 42 : by rfl
  ... ≈ 72 - 42 : calculational, with the values given
  ... = 30 : by rfl
  done

#eval speed_of_second_train -- should output 30

end speed_of_second_train_is_30_l491_491970


namespace problem_statement_l491_491019

-- Definition of point and distance ratio conditions
structure Point (α : Type) := (x : α) (y : α)

def F : Point ℝ := {x := 1, y := 0}

def distance_to_line (M : Point ℝ) (a : ℝ) : ℝ :=
  abs (M.x - a)

def distance (P Q : Point ℝ) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- Main theorem statement
theorem problem_statement (M : Point ℝ) (d : ℝ)
  (h1 : distance M F / (distance_to_line M 4) = 1 / 2)
  (h2 : ∃ A B C D : Point ℝ, quadrilateral_with_sides F A B C F D (∃ A B C D : Point ℝ) minimum_area_quadrilateral_conds :

  (∀ M : Point ℝ, distance M F = 1/2 * distance_to_line M 4) ∧ 
  quadrilateral_with_sides ABCD (minimum_area_quadrilateral_conds)
  → 
  exists_curve E :
  ∀ (curve E), equation_eq_curve
    (position_on_curve E (distances_of_lines_to_point AB, CD))

  : tq_equivalent_problem 

(M equation_min_area_quadrilateral_conds algebra_eqns_h2) :

  filter cond_quot_eq_math_problem
of : curve E associativity_eqs_isomorphic 

   
:= begin 
 -- The Lean language 
 sorry
end

end problem_statement_l491_491019


namespace sum_of_roots_l491_491470

theorem sum_of_roots (a b : ℝ) (h1 : a^2 - 4*a - 2023 = 0) (h2 : b^2 - 4*b - 2023 = 0) : a + b = 4 :=
sorry

end sum_of_roots_l491_491470


namespace math_problem_l491_491046

noncomputable
def line_system (θ : ℝ) : set (ℝ × ℝ) := { p : ℝ × ℝ | p.1 * Real.cos θ + p.2 * Real.sin θ = 1 }

theorem math_problem (θ : ℝ) :
  (∀ p : ℝ × ℝ, p ∉ (line_system θ) → p.1^2 + p.2^2 < 1) ∧
  (¬∀ θ₁ θ₂ : ℝ, θ₁ ≠ θ₂ → ∀ p : ℝ × ℝ, p ∈ (line_system θ₁) ∧ p ∈ (line_system θ₂) → False) ∧
  (¬∃ q : ℝ × ℝ, ∀ θ : ℝ, q ∈ (line_system θ)) ∧
  (∀ n : ℕ, n ≥ 3 → ∃ (polygon : list (ℝ × ℝ)), (set.of_list polygon).card = n ∧
    ∀ i, polygon.nth i ∈ (line_system θ)) :=
by
  sorry

end math_problem_l491_491046


namespace race_distance_correct_l491_491169

noncomputable def solve_race_distance : ℝ :=
  let Vq := 1          -- assume Vq as some positive real number (could be normalized to 1 for simplicity)
  let Vp := 1.20 * Vq  -- P runs 20% faster
  let head_start := 300 -- Q's head start in meters
  let Dp := 1800       -- distance P runs

  -- time taken by P
  let time_p := Dp / Vp
  -- time taken by Q, given it has a 300 meter head start
  let Dq := Dp - head_start
  let time_q := Dq / Vq

  Dp

theorem race_distance_correct :
  let Vq := 1          -- assume Vq as some positive real number (could be normalized to 1 for simplicity)
  let Vp := 1.20 * Vq  -- P runs 20% faster
  let head_start := 300 -- Q's head start in meters
  let Dp := 1800       -- distance P runs
  -- time taken by P
  let time_p := Dp / Vp
  -- time taken by Q, given it has a 300 meter head start
  let Dq := Dp - head_start
  let time_q := Dq / Vq

  time_p = time_q := by
  sorry

end race_distance_correct_l491_491169


namespace faster_speed_l491_491486

theorem faster_speed (S : ℝ) (actual_speed : ℝ := 10) (extra_distance : ℝ := 20) (actual_distance : ℝ := 20) :
  actual_distance / actual_speed = (actual_distance + extra_distance) / S → S = 20 :=
by
  sorry

end faster_speed_l491_491486


namespace product_series_evaluation_l491_491397

theorem product_series_evaluation :
  (∏ k in Finset.range (91), (1 - (1 / (k + 10 : ℝ)))) = 9 / 100 :=
by
  sorry

end product_series_evaluation_l491_491397


namespace g_ln_half_l491_491296

noncomputable def f : ℝ → ℝ :=
sorry

noncomputable def g (x : ℝ) : ℝ :=
  f(x) * Real.cos(x) + 1

theorem g_ln_half :
  (∀ x, f(x) + f(-x) = 0) →
  g(Real.ln(2)) = -2 →
  g(Real.ln(1/2)) = 4 :=
by
  intros h1 h2
  sorry

end g_ln_half_l491_491296


namespace horner_value_v2_l491_491643

noncomputable def horner (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldr (λ a_i acc, acc * x + a_i) 0

def coefficients : List ℝ := [5, 2, 3.5, -2.6, 1.7, -0.8]

theorem horner_value_v2 :
  let x : ℝ := 5
  let a := coefficients.reverse
  let v0 := a.getLast 0
  let v1 := v0 * x + a.ilast 1
  let v2 := v1 * x + a.ilast 2
  v2 = 138.5 :=
by
  sorry

end horner_value_v2_l491_491643


namespace log_base_8_of_256_l491_491734

theorem log_base_8_of_256 : Real.log_base 8 256 = 8 / 3
:= sorry

end log_base_8_of_256_l491_491734


namespace angle_between_diagonals_l491_491593

theorem angle_between_diagonals :
  ∃ α : ℝ, ∃ d₁ d₂ : ℝ, ∃ A : ℝ,
  d₁ = 6 ∧ d₂ = 2 ∧ A = 3 ∧
  A = 1/2 * d₁ * d₂ * real.sin α ∧
  α = real.pi / 6 :=
begin
  sorry
end

end angle_between_diagonals_l491_491593


namespace smallest_fraction_for_x_eq_8_l491_491415

theorem smallest_fraction_for_x_eq_8 :
  let x := 8
  in min (min (min (8 / x) (8 / (x + 2))) (8 / (x - 2))) ((x + 5) / 8) ((x - 2) / 8) = (x - 2) / 8 :=
by
  let x := 8
  sorry

end smallest_fraction_for_x_eq_8_l491_491415


namespace polygon_with_three_times_exterior_angle_sum_is_octagon_l491_491301

theorem polygon_with_three_times_exterior_angle_sum_is_octagon
  (n : ℕ)
  (h : (n - 2) * 180 = 3 * 360) : n = 8 := by
  sorry

end polygon_with_three_times_exterior_angle_sum_is_octagon_l491_491301


namespace remainder_of_x150_l491_491759

theorem remainder_of_x150 (x : ℝ) : 
  let quotient := (x + 2)^4
  let remainder := 4433600 * x^3 + 26496900 * x^2 + 53015100 * x + 35427301
in x^150 = quotient * (x^146 + other_terms) + remainder :=
by sorry

end remainder_of_x150_l491_491759


namespace cos_R_proof_l491_491501

def right_triangle_cosine := 
  ∀ (P Q R : Type) 
  (hPQ : (PQ : ℝ) = 8) 
  (hQR : (QR : ℝ) = 10) 
  (hRight : angle P = 90),

  let PR := real.sqrt (QR ^ 2 - PQ ^ 2) in
  cos (angle R) = PR / QR

theorem cos_R_proof : right_triangle_cosine :=
begin
  intros,
  -- we skip the proof here
  sorry
end

end cos_R_proof_l491_491501


namespace train_crosses_platform_in_39_seconds_l491_491998

theorem train_crosses_platform_in_39_seconds :
  ∀ (length_train length_platform : ℝ) (time_cross_signal : ℝ),
  length_train = 300 →
  length_platform = 25 →
  time_cross_signal = 36 →
  ((length_train + length_platform) / (length_train / time_cross_signal)) = 39 := by
  intros length_train length_platform time_cross_signal
  intros h_length_train h_length_platform h_time_cross_signal
  rw [h_length_train, h_length_platform, h_time_cross_signal]
  sorry

end train_crosses_platform_in_39_seconds_l491_491998


namespace meat_loss_calculation_l491_491587

theorem meat_loss_calculation
  (loss : ℝ) (initial_meat : ℝ) (num_hamburgers_original : ℕ) (num_hamburgers_new : ℕ) :
  loss = 0.1 →
  initial_meat = 5 →
  num_hamburgers_original = 10 →
  num_hamburgers_new = 30 →
  let meat_per_hamburger := initial_meat / num_hamburgers_original in
  let effective_meat_per_hamburger := meat_per_hamburger / (1 - loss) in
  let total_meat_needed := effective_meat_per_hamburger * num_hamburgers_new in
  total_meat_needed = 17 :=
by
  intros h_loss h_initial_meat h_num_hamburgers_original h_num_hamburgers_new
  let meat_per_hamburger := initial_meat / num_hamburgers_original
  let effective_meat_per_hamburger := meat_per_hamburger / (1 - loss)
  let total_meat_needed := effective_meat_per_hamburger * num_hamburgers_new
  sorry

end meat_loss_calculation_l491_491587


namespace area_triangle_ABC_l491_491428

open Complex

noncomputable def Z := Complex
def A (Z : Complex) := Z
def B (Z : Complex) := Z^2
def C (Z : Complex) := Z - Z^2

def absZ (Z : Complex) := Complex.abs Z = Real.sqrt 2
def imagZ2 (Z : Complex) := (Z^2).im = 2

-- Function to calculate the area of triangle given three complex points
def area_ABC (Z : Complex) : Real := 
  let p1 := (A Z).re + (A Z).im * Complex.I
  let p2 := (B Z).re + (B Z).im * Complex.I
  let p3 := (C Z).re + (C Z).im * Complex.I
  Real.abs ((p1.re * (p2.im - p3.im) + p2.re * (p3.im - p1.im) + p3.re * (p1.im - p2.im)) / 2)

-- Prove the area of triangle ABC is either 4 or 1 given the conditions
theorem area_triangle_ABC (Z : Complex) (h1 : absZ Z) (h2 : imagZ2 Z) :
      area_ABC Z = 4 ∨ area_ABC Z = 1 := 
sorry

end area_triangle_ABC_l491_491428


namespace complex_quadrant_l491_491091

theorem complex_quadrant :
  ∀ (z : ℂ), z = complex.mk (Real.sin 2) (Real.cos 2) → Real.sin 2 > 0 → Real.cos 2 < 0 → (0 < z.re ∧ z.re ∧ z.im < 0) :=
by
  intros z hz hsin hcos
  -- Proof steps go here
  sorry

end complex_quadrant_l491_491091


namespace amy_total_score_l491_491993

theorem amy_total_score :
  let points_per_treasure := 4
  let treasures_first_level := 6
  let treasures_second_level := 2
  let score_first_level := treasures_first_level * points_per_treasure
  let score_second_level := treasures_second_level * points_per_treasure
  let total_score := score_first_level + score_second_level
  total_score = 32 := by
sorry

end amy_total_score_l491_491993


namespace distance_from_origin_l491_491613

theorem distance_from_origin (i : ℂ) (h_i : i = complex.I) :
  complex.abs (2 * i / (1 - i)) = real.sqrt 2 := by
  sorry

end distance_from_origin_l491_491613


namespace abs_a1_b1_eq_6_l491_491615

theorem abs_a1_b1_eq_6 (a_1 a_2 ... a_m b_1 b_2 ... b_n : ℕ)
  (h1 : 2021 = (nat.factorial a_1 * nat.factorial a_2 * ... * nat.factorial a_m) / (nat.factorial b_1 * nat.factorial b_2 * ... * nat.factorial b_n))
  (h2 : a_1 ≥ a_2 ∧ a_2 ≥ ... ∧ a_m ≥ 1)
  (h3 : b_1 ≥ b_2 ∧ b_2 ≥ ... ∧ b_n ≥ 1)
  (h4 : ∀ a, ∀ b, (a = a_1 ∧ b = b_1) → (a + b = a_1 + b_1))
  : |a_1 - b_1| = 6 := 
sorry

end abs_a1_b1_eq_6_l491_491615


namespace Kat_training_hours_l491_491877

theorem Kat_training_hours
  (h_strength_times : ℕ)
  (h_strength_hours : ℝ)
  (h_boxing_times : ℕ)
  (h_boxing_hours : ℝ)
  (h_times : h_strength_times = 3)
  (h_strength : h_strength_hours = 1)
  (b_times : h_boxing_times = 4)
  (b_hours : h_boxing_hours = 1.5) :
  h_strength_times * h_strength_hours + h_boxing_times * h_boxing_hours = 9 :=
by
  sorry

end Kat_training_hours_l491_491877


namespace interval_of_alpha_l491_491469

theorem interval_of_alpha (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : sin α + cos α = tan α) :
  π / 4 < α ∧ α < π / 3 :=
by
  sorry

end interval_of_alpha_l491_491469


namespace log_base_16_of_4_l491_491750

theorem log_base_16_of_4 : log 16 4 = 1 / 2 := by
  sorry

end log_base_16_of_4_l491_491750


namespace compare_fractions_l491_491346

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l491_491346


namespace probability_born_in_2008_l491_491141

/-- Define the relevant dates and days calculation -/
def num_days_between : ℕ :=
  29 + 31 + 30 + 31 + 31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

def num_days_2008 : ℕ :=
  31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

theorem probability_born_in_2008 :
  (num_days_2008 : ℚ) / num_days_between = 244 / 365 :=
by
  -- The proof is omitted for brevity
  sorry

end probability_born_in_2008_l491_491141


namespace hyperbola_focus_distance_l491_491607

theorem hyperbola_focus_distance
  (b : ℝ)
  (hb : b > 0)
  (h : ∀ (y x : ℝ), y^2 / 9 - x^2 / b^2 = 1)
  (eccentricity : ∀ a c : ℝ, c = 2 * a ∧ c = sqrt (9 + b^2)) :
  ∃ d : ℝ, d = 3 * sqrt 3 := 
by
  sorry

end hyperbola_focus_distance_l491_491607


namespace range_f_l491_491210

def f (x : ℝ) : ℝ := 4 * (Real.sin x) ^ 2 - 4 * (Real.sin x) * (Real.sin (2 * x)) + (Real.sin (2 * x)) ^ 2

theorem range_f : set.Icc 0 ((27 : ℝ) / 4) = set.range f :=
sorry

end range_f_l491_491210


namespace smallest_x_l491_491258

theorem smallest_x :
  ∃ (x : ℕ), x % 4 = 3 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ ∀ y : ℕ, (y % 4 = 3 ∧ y % 5 = 4 ∧ y % 6 = 5) → y ≥ x := 
sorry

end smallest_x_l491_491258


namespace compute_x_l491_491542

variables (a b c x : ℝ)

-- Defining the conditions
def non_zero_product (a b c : ℝ) : Prop :=
  (a + b) ≠ 0 ∧ (b + c) ≠ 0 ∧ (c + a) ≠ 0

def given_conditions (a b c : ℝ) (x : ℝ) : Prop :=
  (a^2 / (a + b)) = (a^2 / (a + c)) + 20 ∧
  (b^2 / (b + c)) = (b^2 / (b + a)) + 14 ∧
  (c^2 / (c + a)) = (c^2 / (c + b)) + x

theorem compute_x (a b c : ℝ) (x : ℝ) (h1 : non_zero_product a b c) (h2 : given_conditions a b c x) : x = -34 :=
sorry

end compute_x_l491_491542


namespace money_left_after_purchases_l491_491175

variable (originalAmount : ℝ) (spentLunch : ℝ) (spentClothes : ℝ) (spentDVD : ℝ) (spentSupplies : ℝ)

-- Define the conditions:
def conditions : Prop :=
  originalAmount = 200 ∧
  spentLunch = originalAmount * (1/4) ∧
  spentClothes = originalAmount * (1/5) ∧
  spentDVD = originalAmount * (1/10) ∧
  spentSupplies = originalAmount * (1/8)

-- Define the theorem to prove:
theorem money_left_after_purchases
  (h : conditions originalAmount spentLunch spentClothes spentDVD spentSupplies) :
  (originalAmount - (spentLunch + spentClothes + spentDVD + spentSupplies)) = 65 :=
begin
  sorry
end

end money_left_after_purchases_l491_491175


namespace journeymen_percentage_l491_491990

-- Definitions based on the conditions
def total_employees : ℕ := 20210
def fraction_journeymen : ℝ := 2 / 7
def journeymen : ℝ := (fraction_journeymen * total_employees)
def remaining_journeymen : ℝ := journeymen / 2
def total_remaining_employees : ℝ := total_employees - remaining_journeymen
def percentage_remaining_journeymen : ℝ := (remaining_journeymen / total_remaining_employees) * 100

-- The proof statement
theorem journeymen_percentage : percentage_remaining_journeymen ≈ 16.67 := by 
  sorry

end journeymen_percentage_l491_491990


namespace total_games_in_season_l491_491682

theorem total_games_in_season
  (n_teams : ℕ) (n_divisions : ℕ) (teams_per_division : ℕ)
  (intra_division_games_twice : ∀ (d : ℕ) (h : d < n_divisions) (t : ℕ) (ht : t < teams_per_division), ℕ)
  (inter_division_games_once : ∀ (d₁ d₂ : ℕ) (hd : d₁ ≠ d₂) (t₁ t₂ : ℕ) (ht₁ : t₁ < teams_per_division) (ht₂ : t₂ < teams_per_division), ℕ) :
  n_teams / n_divisions = teams_per_division →
  (intra_division_games_twice = (λ _ _ _ _, 7 * 2)) →
  (inter_division_games_once = (λ _ _ _ _ _, 1)) →
  n_teams = 16 →
  n_divisions = 2 →
  ∑ d in finset.range n_divisions, ∑ t in finset.range teams_per_division, (14 + 8) / 2 = 176 :=
by
  intros h1 h2 h3 h4 h5 
  sorry

end total_games_in_season_l491_491682


namespace pyramid_height_value_l491_491666

-- Let s be the edge length of the cube
def cube_edge_length : ℝ := 6

-- Let b be the base edge length of the square-based pyramid
def pyramid_base_edge_length : ℝ := 12

-- Let V_cube be the volume of the cube
def volume_cube (s : ℝ) : ℝ := s ^ 3

-- Let V_pyramid be the volume of the pyramid
def volume_pyramid (b h : ℝ) : ℝ := (1 / 3) * (b ^ 2) * h

-- The given volumes are equal
def volumes_equal : Prop :=
  volume_cube cube_edge_length = volume_pyramid pyramid_base_edge_length h

-- Prove that the height h of the pyramid is 4.5
theorem pyramid_height_value (h : ℝ) (cube_edge_length pyramid_base_edge_length : ℝ) (volumes_equal : Prop) :
  h = 4.5 :=
sorry

end pyramid_height_value_l491_491666


namespace right_triangle_exists_and_r_inscribed_circle_l491_491653

theorem right_triangle_exists_and_r_inscribed_circle (d : ℝ) (hd : d > 0) :
  ∃ (a b c : ℝ), 
    a < b ∧ 
    a^2 + b^2 = c^2 ∧
    b = a + d ∧ 
    c = b + d ∧ 
    (a + b - c) / 2 = d :=
by
  sorry

end right_triangle_exists_and_r_inscribed_circle_l491_491653


namespace probability_divisor_l491_491250

theorem probability_divisor (n : ℕ) (hn : n = 30) :
  (Nat.card {x // x ∣ n ∧ 1 ≤ x ∧ x ≤ n}.to_finset) / (Nat.card {x // 1 ≤ x ∧ x ≤ n}.to_finset : ℚ) = 4 / 15 :=
by
  sorry

end probability_divisor_l491_491250


namespace Amanda_ticket_sales_goal_l491_491695

theorem Amanda_ticket_sales_goal :
  let total_tickets : ℕ := 80
  let first_day_sales : ℕ := 5 * 4
  let second_day_sales : ℕ := 32
  total_tickets - (first_day_sales + second_day_sales) = 28 :=
by
  sorry

end Amanda_ticket_sales_goal_l491_491695


namespace values_of_a_and_k_max_profit_at_2_l491_491934

variable {a k x y : ℝ}

-- Definitions based on the conditions
def y_def : ℝ → ℝ → ℝ
| a x => if (1 < x ∧ x ≤ 3) then a * (x - 4)^2 + 6 / (x - 1) 
         else if (3 < x ∧ x ≤ 5) then k * x + 7 
         else 0

def profit (x : ℝ) : ℝ := 
  if (1 < x ∧ x ≤ 3) then (a * (x - 4)^2 + 6 / (x - 1)) * (x - 1) - x + 1 
  else if (3 < x ∧ x ≤ 5) then (k * x + 7) * (x - 1) - x + 1 
  else 0

-- The two goals to be proved in Lean
theorem values_of_a_and_k : (∀ (x : ℝ), y_def 1 x = (if (1 < x ∧ x ≤ 3) then (x - 4)^2 + 6 / (x - 1) 
                                                    else if (3 < x ∧ x ≤ 5) then -x + 7
                                                    else 0)) ∧
                            (y_def a 3 = 4) ∧ 
                            (y_def k 5 = 2) ∧
                            (a = 1) ∧ (k = -1) := 
  sorry

theorem max_profit_at_2 : (profit 2 = 10) ∧ 
                          (∀ x, profit x ≤ profit 2) :=
  sorry

end values_of_a_and_k_max_profit_at_2_l491_491934


namespace contrapositive_example_l491_491812

variable (a b : ℝ)

theorem contrapositive_example
  (h₁ : a > 0)
  (h₃ : a + b < 0) :
  b < 0 := 
sorry

end contrapositive_example_l491_491812


namespace monotonicity_a_one_range_of_a_sum_inequality_l491_491815

namespace MathProofs

-- Define the function f
def f (x a : ℝ) : ℝ := x * Real.exp (a * x) - Real.exp x

-- Part 1: Monotonicity when a = 1
theorem monotonicity_a_one (x : ℝ) :
    (x > 0 → (Real.exp x * (x - 1)).deriv > 0) ∧
    (x < 0 → (Real.exp x * (x - 1)).deriv < 0) := sorry

-- Part 2: Range of values for a when x > 0, f(x) < -1
theorem range_of_a (x : ℝ) (hx : x > 0) :
    (∀ a : ℝ, f x a < -1 → a ≤ 1/2) := sorry

-- Part 3: Proving the inequality
theorem sum_inequality (n : ℕ) (hn : 0 < n) :
    1/n.succ.succ + (1 / Real.sqrt ((n.succ.succ : ℝ) ^ 2 + ↑n.succ.succ)) + ∑ i in Finset.range n.succ \ {0}, 1 / Real.sqrt (i ^ 2 + i) > Real.log (n + 1) := sorry

end MathProofs

end monotonicity_a_one_range_of_a_sum_inequality_l491_491815


namespace domain_of_f_l491_491243

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : 
  {x : ℝ | ¬ ((x - 3) + (x - 9) = 0)} = 
  {x : ℝ | x ≠ 6} := 
by
  sorry

end domain_of_f_l491_491243


namespace range_f1_l491_491409

noncomputable def f1 (x : ℝ) : ℝ := 2^(x + 2) - 3 * 4^x

theorem range_f1 : set.range f1 = set.Ioc (-(4:ℝ)) (4/3) :=
begin
  sorry
end

end range_f1_l491_491409


namespace pens_in_each_pack_l491_491879

-- Given the conditions
def Kendra_packs : ℕ := 4
def Tony_packs : ℕ := 2
def pens_kept_each : ℕ := 2
def friends : ℕ := 14

-- Theorem statement
theorem pens_in_each_pack : ∃ (P : ℕ), Kendra_packs * P + Tony_packs * P - pens_kept_each * 2 - friends = 0 ∧ P = 3 := by
  sorry

end pens_in_each_pack_l491_491879


namespace decimal_to_fraction_l491_491275

theorem decimal_to_fraction (h : 0.36 = 36 / 100): (36 / 100 = 9 / 25) := by
    sorry

end decimal_to_fraction_l491_491275


namespace max_area_of_triangle_ABC_l491_491095

-- Define a triangle with given conditions.
-- AB = 2 and AC = 2BC.
variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
noncomputable def dist {α : Type} [metric_space α] : α → α → ℝ := sorry

def triangle_area_max (X Y Z : Type)
  [metric_space X] [metric_space Y] [metric_space Z]
  (AB AC BC : ℝ) (h1 : AB = 2) (h2 : AC = 2 * BC) : ℝ :=
  sorry

theorem max_area_of_triangle_ABC (A B C : Type)
  [metric_space A] [metric_space B] [metric_space C]
  (AB AC BC : ℝ) (h1 : dist A B = 2) (h2 : dist A C = 2 * dist B C) :
  triangle_area_max A B C AB h1 h2 = 4 / 3 :=
sorry

end max_area_of_triangle_ABC_l491_491095


namespace max_value_expression_l491_491136

theorem max_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (eq_condition : x^2 - 3 * x * y + 4 * y^2 - z = 0) : 
  ∃ (M : ℝ), M = 1 ∧ (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x^2 - 3 * x * y + 4 * y^2 - z = 0 → (2/x + 1/y - 2/z) ≤ M) := 
by
  sorry

end max_value_expression_l491_491136


namespace probability_maxim_born_in_2008_l491_491150

def days_in_month : ℕ → ℕ → ℕ 
| 2007 9 := 29
| 2007 10 := 31
| 2007 11 := 30
| 2007 12 := 31
| 2008 1 := 31
| 2008 2 := 29   -- leap year
| 2008 3 := 31
| 2008 4 := 30
| 2008 5 := 31
| 2008 6 := 30
| 2008 7 := 31
| 2008 8 := 31
| _ _ := 0

def total_days : ℕ :=
list.sum [days_in_month 2007 9, days_in_month 2007 10, days_in_month 2007 11, days_in_month 2007 12,
          days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

def days_in_2008 : ℕ :=
list.sum [days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

theorem probability_maxim_born_in_2008 : 
  (days_in_2008 : ℚ) / (total_days : ℚ) = 244 / 365 :=
by
  -- Placeholder for actual proof. This theorem is set properly and ready for proof development.
  sorry

end probability_maxim_born_in_2008_l491_491150


namespace min_period_f_pi_range_m_l491_491036

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.sin (π / 4 - x))^2 - sqrt 3 * Real.cos (2 * x)

theorem min_period_f_pi : ∀ x : ℝ, f (x + π) = f x :=
by
  -- Proof details omitted
  sorry

theorem range_m :
  (∀ x, 0 ≤ x ∧ x ≤ π / 6 → f(x) < m + 2) → m > -1 - sqrt 3 :=
by
  -- Proof details omitted
  sorry

end min_period_f_pi_range_m_l491_491036


namespace smallest_three_digit_number_with_product_of_digits_eight_and_even_digit_exists_l491_491981

theorem smallest_three_digit_number_with_product_of_digits_eight_and_even_digit_exists :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ 
            (∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧ 
                            a * b * c = 8 ∧ 
                            (even a ∨ even b ∨ even c) ∧ 
                            (a,b,c) ∈ {(1, 2, 4), (1, 4, 2), (2, 1, 4), (2, 4, 1), (4, 1, 2), (4, 2, 1), (1, 1, 8)}) ∧
            n = 124 := sorry

end smallest_three_digit_number_with_product_of_digits_eight_and_even_digit_exists_l491_491981


namespace log_sum_expression_l491_491001

theorem log_sum_expression (n : ℝ) (h : 3^n = 2) : log 3 6 + log 3 8 = 4 * n + 1 :=
sorry

end log_sum_expression_l491_491001


namespace dots_not_visible_l491_491421

theorem dots_not_visible (total_dots visible_sum : ℕ) (h1 : total_dots = 84) (h2 : visible_sum = 26) : total_dots - visible_sum = 58 :=
by
  rw [h1, h2]
  norm_num
  sorry

end dots_not_visible_l491_491421


namespace count_4x4_increasing_arrays_l491_491937

def is_valid_array (arr : Array (Array Nat)) : Prop :=
  arr.size = 4 ∧
  (∀ i, i < 4 → (arr[i]).size = 4) ∧
  (∀ i j, i < 4 → j < 4 → 1 ≤ arr[i][j] ∧ arr[i][j] ≤ 16) ∧
  (∀ i, i < 4 → (∀ j k, j < k → arr[i][j] < arr[i][k])) ∧
  (∀ j, j < 4 → (∀ i k, i < k → arr[i][j] < arr[k][j]))

def count_valid_4x4_arrays : Nat :=
  36

theorem count_4x4_increasing_arrays :
  ∃! arr : Array (Array Nat), is_valid_array arr :=
by 
  exact count_valid_4x4_arrays 
 
end count_4x4_increasing_arrays_l491_491937


namespace largest_n_l491_491790

variable {a : ℕ → ℝ} -- The arithmetic sequence
variable {d : ℝ} -- Common difference
variable {a1 : ℝ} -- First term of the sequence

-- Conditions
variable h_pos : a1 > 0 -- The first term is positive
variable h_seq : ∀ n : ℕ, a (n + 1) = a n + d -- Definition of arithmetic sequence
variable h2005 : a 2005 + a 2006 > 0 
variable h2005_mul : a 2005 * a 2006 < 0 

-- Required to prove
theorem largest_n (n : ℕ) (h1 : n ≤ 4010) : n ≠ 4011 → (∑ i in Finset.range n, a i) > 0 :=
by
  intro h1 h2
  sorry

end largest_n_l491_491790


namespace eating_relationship_l491_491260

open Set

-- Definitions of the sets A and B
def A : Set ℝ := {-1, 1/2, 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≥ 0 ∧ a * x^2 = 1}

-- Definitions of relationships
def full_eating (A B : Set ℝ) : Prop := A ⊆ B ∨ B ⊆ A
def partial_eating (A B : Set ℝ) : Prop := ∃ x, x ∈ A ∧ x ∈ B ∧ ¬ full_eating A B

-- Main theorem
theorem eating_relationship (a : ℝ) :
  (full_eating A (B a) ∨ partial_eating A (B a)) ↔ (a = 0 ∨ a = 1 ∨ a = 4) := by
  sorry

end eating_relationship_l491_491260


namespace find_f_2m_l491_491543

noncomputable def f : ℤ → ℤ := sorry  -- strictly increasing function
noncomputable def g : ℤ → ℤ := sorry  -- strictly increasing function

axiom f_strictly_increasing : ∀ {a b : ℤ}, a < b → f a < f b
axiom g_strictly_increasing : ∀ {a b : ℤ}, a < b → g a < g b

axiom fg_disjoint : ∀ {a : ℤ}, a > 0 → ¬(f a = g a)
axiom fg_union : ∀ {a : ℤ}, a > 0 → ∃ b : ℤ, f b = a ∨ g b = a

axiom g_def : ∀ {m : ℤ}, m > 0 → g m = f (f m) + 1

theorem find_f_2m {m : ℤ} (h : m > 0) : f (2 * m) = m + (int.floor (real.sqrt (5) * m)) := by
  sorry

end find_f_2m_l491_491543


namespace doris_weeks_to_cover_expenses_l491_491721

-- Define the constants and conditions from the problem
def hourly_rate : ℝ := 20
def monthly_expenses : ℝ := 1200
def weekday_hours_per_day : ℝ := 3
def weekdays_per_week : ℝ := 5
def saturday_hours : ℝ := 5

-- Calculate total hours worked per week
def weekly_hours := (weekday_hours_per_day * weekdays_per_week) + saturday_hours

-- Calculate weekly earnings
def weekly_earnings := hourly_rate * weekly_hours

-- Finally, the number of weeks required to meet the monthly expenses
def required_weeks := monthly_expenses / weekly_earnings

-- The theorem to prove
theorem doris_weeks_to_cover_expenses : required_weeks = 3 := by
  -- We skip the proof but indicate it needs to be provided
  sorry

end doris_weeks_to_cover_expenses_l491_491721


namespace triangle_base_length_l491_491512

theorem triangle_base_length (x : ℝ) :
  (∃ s : ℝ, 4 * s = 64 ∧ s * s = 256) ∧ (32 * x / 2 = 256) → x = 16 := by
  sorry

end triangle_base_length_l491_491512


namespace colonization_ways_l491_491058

/-
Define the constraints and conditions based on the given data in the problem.
-/
def earth_like_planets : ℕ := 5
def mars_like_planets : ℕ := 6
def earth_like_units : ℕ := 2
def mars_like_units : ℕ := 1
def total_units : ℕ := 14

/-
Define the main problem statement indicating the correct number of ways to allocate colonies.
-/
theorem colonization_ways : 
  ∃! ways : ℕ, 
    (ways = 
      (finset.card (set.finite.to_finset (set.to_set ({n // n ≤ earth_like_planets}))) * 
       finset.card (set.finite.to_finset (set.to_set ({m // m ≤ mars_like_planets}))))) ∧ 
    ((2 * finset.card (set.finite.to_finset (set.to_set ({n // n ≤ earth_like_planets}))) + 
     finset.card (set.finite.to_finset (set.to_set ({m // m ≤ mars_like_planets})))) = total_units) := 
  sorry

end colonization_ways_l491_491058


namespace dandelion_seed_production_l491_491571

theorem dandelion_seed_production :
  ∀ (initial_seeds : ℕ), initial_seeds = 50 →
  ∀ (germination_rate : ℚ), germination_rate = 1 / 2 →
  ∀ (new_seed_rate : ℕ), new_seed_rate = 50 →
  (initial_seeds * germination_rate * new_seed_rate) = 1250 :=
by
  intros initial_seeds h1 germination_rate h2 new_seed_rate h3
  sorry

end dandelion_seed_production_l491_491571


namespace z_in_first_quadrant_l491_491810

-- Define the complex number z
def z : ℂ := (1 / (1 + (I : ℂ))) + I

-- Prove that z lies in the first quadrant by showing both the real and imaginary parts are positive
theorem z_in_first_quadrant : 0 < z.re ∧ 0 < z.im :=
sorry

end z_in_first_quadrant_l491_491810


namespace digit_sum_of_expression_l491_491254

theorem digit_sum_of_expression :
  let expr := 2^2010 * 5^2005 * 7 in
  (sum_of_digits expr) = 8 :=
by
  sorry

end digit_sum_of_expression_l491_491254


namespace range_of_m_l491_491820

noncomputable def f (x : ℝ) : ℝ :=
x^3 + real.log (real.sqrt (x^2 + 1) + x)

theorem range_of_m (m : ℝ) 
  (h : ∀ x : ℝ, f (2^x - 4^x) + f (m * 2^x - 3) < 0) : 
  m < 2 * real.sqrt 3 - 1 :=
sorry

end range_of_m_l491_491820


namespace compare_rat_neg_l491_491350

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l491_491350


namespace mixture_release_volume_l491_491309

theorem mixture_release_volume (V C x : ℝ) (hV : V = 8) (hC : C = 16) (hA : 9) :
    let o_init := C / 100 * V in
    let o_after_first := o_init - (C / 100 * x) in
    let o_add_nitrogen := o_after_first * (V / (V - x)) in
    let o_after_second := o_add_nitrogen - (o_add_nitrogen * (x / V)) in
    o_after_second = hA / 100 * V → 
    x = 2 := 
by 
  sorry

end mixture_release_volume_l491_491309


namespace exists_non_regular_triangle_with_similar_medians_as_sides_l491_491526

theorem exists_non_regular_triangle_with_similar_medians_as_sides 
  (a b c : ℝ) 
  (s_a s_b s_c : ℝ)
  (h1 : 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2)
  (h2 : 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2)
  (h3 : 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2)
  (similarity_cond : (2*c^2 + 2*b^2 - a^2) / c^2 = (2*c^2 + 2*a^2 - b^2) / b^2 ∧ (2*c^2 + 2*a^2 - b^2) / b^2 = (2*a^2 + 2*b^2 - c^2) / a^2) :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (∃ (s_a s_b s_c : ℝ), 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2 ∧ 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2 ∧ 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2) ∧
  ((2*c^2 + 2*b^2 - a^2) / c^2 = (2*c^2 + 2*a^2 - b^2) / b^2 ∧ (2*c^2 + 2*a^2 - b^2) / b^2 = (2*a^2 + 2*b^2 - c^2) / a^2) :=
sorry

end exists_non_regular_triangle_with_similar_medians_as_sides_l491_491526


namespace ratio_DB_BP_l491_491642

theorem ratio_DB_BP (k1 k2 x0: ℝ) (h : k1 > k2 > 0):
  let P := (x0, k1 / x0)
  let C := (x0, 0)
  let A := (x0, k2 / x0)
  let D := (0, k1 / x0)
  let B := (x0 / 3, k1 / x0)
  let PA := dist P A
  let PC := dist P C
  let DB := dist D B
  let BP := dist B P
  PA / PC = 2 / 3 → DB / BP = 1 / 2 :=
sorry

end ratio_DB_BP_l491_491642


namespace probability_divisor_l491_491249

theorem probability_divisor (n : ℕ) (hn : n = 30) :
  (Nat.card {x // x ∣ n ∧ 1 ≤ x ∧ x ≤ n}.to_finset) / (Nat.card {x // 1 ≤ x ∧ x ≤ n}.to_finset : ℚ) = 4 / 15 :=
by
  sorry

end probability_divisor_l491_491249


namespace probability_white_black_l491_491220

variable (a b : ℕ)

theorem probability_white_black (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  (2 * a * b) / (a + b) / (a + b - 1) = (2 * (a * b) : ℝ) / ((a + b) * (a + b - 1): ℝ) :=
by sorry

end probability_white_black_l491_491220


namespace eliza_height_is_68_l491_491394

-- Define the known heights of the siblings
def height_sibling_1 : ℕ := 66
def height_sibling_2 : ℕ := 66
def height_sibling_3 : ℕ := 60

-- The total height of all 5 siblings combined
def total_height : ℕ := 330

-- Eliza is 2 inches shorter than the last sibling
def height_difference : ℕ := 2

-- Define the heights of the siblings
def height_remaining_siblings := total_height - (height_sibling_1 + height_sibling_2 + height_sibling_3)

-- The height of the last sibling
def height_last_sibling := (height_remaining_siblings + height_difference) / 2

-- Eliza's height
def height_eliza := height_last_sibling - height_difference

-- We need to prove that Eliza's height is 68 inches
theorem eliza_height_is_68 : height_eliza = 68 := by
  sorry

end eliza_height_is_68_l491_491394


namespace weekly_milk_consumption_l491_491555

def milk_weekday : Nat := 3
def milk_saturday := 2 * milk_weekday
def milk_sunday := 3 * milk_weekday

theorem weekly_milk_consumption : (5 * milk_weekday) + milk_saturday + milk_sunday = 30 := by
  sorry

end weekly_milk_consumption_l491_491555


namespace instantaneous_velocity_at_1_l491_491867

noncomputable def h (t : ℝ) : ℝ := -4.9 * t^2 + 6.5 * t + 10

theorem instantaneous_velocity_at_1 :
  (deriv h 1) = -3.3 :=
by
  sorry

end instantaneous_velocity_at_1_l491_491867


namespace min_value_expression_l491_491537

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) * (1 / x + 4 / y) ≥ 9 :=
sorry

end min_value_expression_l491_491537


namespace part1_part2_l491_491454

def f (x m : ℝ) : ℝ := |x - m| - |x + 3 * m|

theorem part1 (x : ℝ) : 
  (∀ (x : ℝ), f x 1 ≥ 1 → x ≤ -3 / 2) :=
sorry

theorem part2 (x t : ℝ) (h : ∀ (x t : ℝ), f x m < |2 + t| + |t - 1|) : 
  0 < m ∧ m < 3 / 4 :=
sorry

end part1_part2_l491_491454


namespace find_a_value_l491_491481

theorem find_a_value (a : ℤ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
by
  sorry

end find_a_value_l491_491481


namespace compare_neg_fractions_l491_491363

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l491_491363


namespace train_crosses_bridge_in_60_seconds_l491_491201

def length_of_bridge : ℝ := 200
def length_of_train : ℝ := 100
def speed_of_train : ℝ := 5

def total_distance : ℝ := length_of_train + length_of_bridge
def time_to_cross_bridge : ℝ := total_distance / speed_of_train

theorem train_crosses_bridge_in_60_seconds :
  time_to_cross_bridge = 60 := by
  sorry

end train_crosses_bridge_in_60_seconds_l491_491201


namespace triangle_median_lengths_l491_491903

variable (D E F P Q : Type)
variable [MetricSpace D]
variable [MetricSpace E]
variable [MetricSpace F]
variable [MetricSpace P]
variable [MetricSpace Q]

/-- DE is the length of the side DE of triangle DEF given that the medians DP and EQ are 
perpendicular and their lengths are 24 and 32, respectively. -/
theorem triangle_median_lengths (h1 : MetricSpace.toReal (dist D P) = 24) 
    (h2 : MetricSpace.toReal (dist E Q) = 32)
    (h3 : Midpoint P E F)
    (h4 : Midpoint Q D F)
    (h5 : MedianPerpendicular D P E Q) 
  :  MetricSpace.toReal (dist D E) = 80 / 3 := 
sorry

-- Supporting definitions
class Midpoint (P : Type) (a b : P) : Prop :=
(eq : dist P a b/2 = dist P a P = dist P P b)

class MedianPerpendicular (D: Type) (P: Type) (E: Type) (Q: Type) : Prop :=
(eq:  MetricSpace.toReal (dist D P) * MetricSpace.toReal (dist E Q) = 0)

-- Note: The formalism and definitions here aim to set up the conditions 
-- and the result but require leveraging MetricSpace features, types, and 
-- properties which would need to be precisely defined in a full implementation.

end triangle_median_lengths_l491_491903


namespace value_of_expression_is_correct_l491_491961

-- Defining the sub-expressions as Lean terms
def three_squared : ℕ := 3^2
def intermediate_result : ℕ := three_squared - 3
def final_result : ℕ := intermediate_result^2

-- The statement we need to prove
theorem value_of_expression_is_correct : final_result = 36 := by
  sorry

end value_of_expression_is_correct_l491_491961


namespace min_weighings_to_determine_counterfeit_l491_491217

theorem min_weighings_to_determine_counterfeit (c : ℕ) (k : ℕ) (f : ℕ → ℤ) :
  (c = 25) → (k = 12) →
  (∀ i, f i ∈ {-1, 0, 1}) →
  (∃ a, (a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1) → ∀ i, f i ≠ a) →
  (∀ n : ℕ, k = n → c - n = 13) →
  (∀ I J : finset ℕ, (I.card = k) → (J.card = k) → (disjoint I J) → 
  (abs (\sum i in I, f i) = abs (\sum i in J, f i)) ∨ (abs (\sum i in I, f i) ≠ abs (\sum i in J, f i))) ↔
  (∃ I J : finset ℕ, 
  (I.card = k) ∧ (J.card = k) ∧ (disjoint I J) ∧ 
  (abs (\sum i in I, f i) ≠ abs (\sum i in J, f i)) →
  ∀ n, (c - n = 1) → (abs (\sum i in I, f i) = 1 ∨ abs (\sum i in J, f i) = 1)) :=
begin
  sorry
end

end min_weighings_to_determine_counterfeit_l491_491217


namespace log_10_850_consecutive_integers_l491_491956

theorem log_10_850_consecutive_integers : 
  (2:ℝ) < Real.log 850 / Real.log 10 ∧ Real.log 850 / Real.log 10 < (3:ℝ) →
  ∃ (a b : ℕ), (a = 2) ∧ (b = 3) ∧ (2 < Real.log 850 / Real.log 10) ∧ (Real.log 850 / Real.log 10 < 3) ∧ (a + b = 5) :=
by
  sorry

end log_10_850_consecutive_integers_l491_491956


namespace sin_B_sin_C_perimeter_ABC_l491_491097

variable (A B C : ℝ) (a b c : ℝ)
variable (sin sin : ℝ → ℝ) (cos : ℝ → ℝ) (π : ℝ)
variable (h_area : a^2 = 3 * sin A * (1/2) * a * c * sin B)
variable (h_cosBC : 6 * cos B * cos C = 1)
variable (h_a_val : a = 3)

-- Statement for (1)
theorem sin_B_sin_C (h_area : a^2 = 3 * sin A * (1/2) * a * c * sin B) :
  sin B * sin C = 2 / 3 := sorry

-- Statement for (2)
theorem perimeter_ABC (h_cosBC : 6 * cos B * cos C = 1) (h_a_val : a = 3) :
  a + b + c = 3 + real.sqrt 33 := sorry

end sin_B_sin_C_perimeter_ABC_l491_491097


namespace increasing_function_range_l491_491449

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≤ 1 then x^2 + a * x - 2 else -a ^ x

theorem increasing_function_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : 0 < a ∧ a ≤ 1/2 :=
by sorry

end increasing_function_range_l491_491449


namespace solution_inequality_l491_491440

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_increasing_on_nonnegative (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

noncomputable def solution_set (f : ℝ → ℝ) : set ℝ := {x | x > 2 ∨ (0 < x ∧ x < 1/2)}

theorem solution_inequality
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_increasing : is_increasing_on_nonnegative f)
  (h_f1_zero : f 1 = 0) :
  {x | f (real.log x / real.log 2) > 0} = solution_set f := 
sorry

end solution_inequality_l491_491440


namespace tom_runs_five_days_a_week_l491_491968

-- Define the conditions
def speed : ℝ := 8 -- speed in miles per hour
def time_per_day : ℝ := 1.5 -- time per day in hours
def total_distance_per_week : ℝ := 60 -- total distance per week in miles

-- Define the calculation for total distance per day
def distance_per_day (speed time : ℝ) : ℝ := speed * time

-- Define the calculation for the number of days he runs in a week
def number_of_days (total_distance distance_per_day : ℝ) : ℝ := total_distance / distance_per_day

-- Prove that Tom runs 5 days a week, given the conditions.
theorem tom_runs_five_days_a_week : number_of_days total_distance_per_week (distance_per_day speed time_per_day) = 5 := by
  sorry

end tom_runs_five_days_a_week_l491_491968


namespace aimee_poll_l491_491316

theorem aimee_poll (W P : ℕ) (h1 : 0.35 * W = 21) (h2 : 2 * W = P) : P = 120 :=
by
  -- proof in Lean is omitted, placeholder
  sorry

end aimee_poll_l491_491316


namespace orthocenter_of_triangle_ABC_l491_491047

noncomputable def orthocenter (A B C : ℝ × ℝ) : ℝ × ℝ :=
  sorry -- function to find the orthocenter

theorem orthocenter_of_triangle_ABC :
  let A := (5, -1 : ℝ)
  let B := (4, -8 : ℝ)
  let C := (-4, -4 : ℝ)
  orthocenter A B C = (3, -5 : ℝ) :=
by
  simp [A, B, C, orthocenter]
  sorry

end orthocenter_of_triangle_ABC_l491_491047


namespace students_receive_stickers_l491_491904

theorem students_receive_stickers:
  ∀ (gold silver bronze total_students : ℕ),
    gold = 50 →
    silver = 2 * gold →
    bronze = silver - 20 →
    total_students * 46 = gold + silver + bronze →
    total_students = 5 :=
by
  intros gold silver bronze total_students
  assume h1 h2 h3 h4
  sorry

end students_receive_stickers_l491_491904


namespace num_tangent_circles_l491_491130

theorem num_tangent_circles (C1 C2 : Circle) (r : ℝ) (C1_prop : C1.radius = 7) 
  (C2_prop : C2.radius = 7) (tangent_prop : C1.tangent C2) : 
  (count_tangent_circles C1 C2 26) = 6 := 
by
  sorry

end num_tangent_circles_l491_491130


namespace compare_fractions_l491_491345

def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

theorem compare_fractions : a > b :=
by {
  sorry
}

end compare_fractions_l491_491345


namespace part_i_part_ii_l491_491884

-- Definitions
def f (x : ℝ) : ℝ := 1 / (1 + x)

def g_n (n : ℕ) (x : ℝ) : ℝ :=
  x + (List.range n).map (λ i, (nat.iterate f (i+1)) x).sum

def F : ℕ → ℝ
| 0     :=  0
| 1     :=  1
| (n+2) := F (n+1) + F n

-- Part (i)
theorem part_i (x y : ℝ) (hxy : x > y) (hx : x > 0) (hy : y > 0) (n : ℕ) : 
  g_n n x > g_n n y := sorry

-- Part (ii)
theorem part_ii (n : ℕ) : 
  g_n n 1 = Real.sum (List.range n).map (λ i, F (i + 1) / F (i + 2)) := sorry

end part_i_part_ii_l491_491884


namespace max_value_abs_expression_l491_491435

noncomputable def maximum_value_expression : ℝ :=
  12

theorem max_value_abs_expression (x y : ℝ) (h : x^2 + 4y^2 ≤ 4) :
  |x + 2 * y - 4| + |3 - x - y| ≤ maximum_value_expression :=
sorry

end max_value_abs_expression_l491_491435


namespace vote_percentages_correct_l491_491873

noncomputable def verify_vote_percentages (V_W V_P V_A V_T : ℕ) : Prop :=
  V_W = V_P + 20196 ∧
  V_A = 15684 ∧
  V_W + V_P + V_A = V_T ∧
  V_T = 196554 ∧
  (V_W.to_rat / V_T.to_rat) * 100 = 51.14 ∧
  (V_P.to_rat / V_T.to_rat) * 100 = 40.87 ∧
  (V_A.to_rat / V_T.to_rat) * 100 = 7.98

theorem vote_percentages_correct :
  ∃ V_W V_P V_A V_T : ℕ, verify_vote_percentages V_W V_P V_A V_T :=
sorry

end vote_percentages_correct_l491_491873


namespace cat_catch_time_l491_491290

-- Conditions
def t_rat : ℝ := 6
def s_cat : ℝ := 90
def s_rat : ℝ := 36
def d_rat : ℝ := s_rat * t_rat
def s_relative : ℝ := s_cat - s_rat

-- Proof statement
theorem cat_catch_time : ∃ t : ℝ, d_rat = s_relative * t ∧ t = 4 :=
by
  use 4
  split
  sorry

end cat_catch_time_l491_491290


namespace pipe_network_renovation_l491_491967

theorem pipe_network_renovation 
  (total_length : Real)
  (efficiency_increase : Real)
  (days_ahead_of_schedule : Nat)
  (days_completed : Nat)
  (total_period : Nat)
  (original_daily_renovation : Real)
  (additional_renovation : Real)
  (h1 : total_length = 3600)
  (h2 : efficiency_increase = 20 / 100)
  (h3 : days_ahead_of_schedule = 10)
  (h4 : days_completed = 20)
  (h5 : total_period = 40)
  (h6 : (3600 / original_daily_renovation) - (3600 / (1.2 * original_daily_renovation)) = 10)
  (h7 : 20 * (72 + additional_renovation) >= 3600 - 1440) :
  (1.2 * original_daily_renovation = 72) ∧ (additional_renovation >= 36) :=
by
  sorry

end pipe_network_renovation_l491_491967


namespace eight_digit_positive_integers_l491_491832

theorem eight_digit_positive_integers : 
  let choices_first_digit := 9 in
  let choices_other_digits := 10 in
  choices_first_digit * choices_other_digits ^ 7 = 90000000 :=
by 
  simp only [choices_first_digit, choices_other_digits],
  norm_num

end eight_digit_positive_integers_l491_491832


namespace T_frame_sum_l491_491326

-- State the given conditions
theorem T_frame_sum (x : ℤ) (h : 5 * x + 21 = 101) : 
  let smallest := x - 1 in
  let largest := x + 14 in
  smallest = 15 ∧ largest = 30 :=
by
  sorry

end T_frame_sum_l491_491326


namespace compare_negative_fractions_l491_491356

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l491_491356


namespace angle_MHS_is_15_l491_491493

-- Define the various angles and points
variables (P Q R H M S : Type) [Fintype P] [Fintype Q] [Fintype R]
variables (angle_P_QR angle_Q angle_R : ℝ)
variables (altitude_PH median_QM bisector_QS : Type)

-- Assuming the given conditions
axiom angle_P_def : angle_P_QR = 120
axiom angle_Q_def : angle_Q = 30
axiom angle_R_def : angle_R = 30
axiom altitude_PH_def : altitude_PH = 90
axiom median_QM_def : ∀ (P R), median_QM = (P + R) / 2
axiom bisector_QS_def : bisector_QS = angle_Q / 2

-- Define the proof problem
theorem angle_MHS_is_15 :
  (angle_P_QR = 120) ∧ (angle_Q = 30) ∧ (angle_R = 30)
  ∧ (altitude_PH = 90)
  ∧ (∀ (P R), median_QM = (P + R) / 2)
  ∧ (bisector_QS = angle_Q / 2)
  → ∃ (M H S : Type), ∠ MHS = 15 :=
sorry

end angle_MHS_is_15_l491_491493


namespace geometric_sequence_product_l491_491092

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_product (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
  (h : a 3 = -1) : a 1 * a 2 * a 3 * a 4 * a 5 = -1 :=
by
  sorry

end geometric_sequence_product_l491_491092


namespace percentage_is_36_point_4_l491_491286

def part : ℝ := 318.65
def whole : ℝ := 875.3

theorem percentage_is_36_point_4 : (part / whole) * 100 = 36.4 := 
by sorry

end percentage_is_36_point_4_l491_491286


namespace evaluate_modulus_l491_491733

def complex := ℂ
def c : complex := -3 - (5 / 4) * complex.I

theorem evaluate_modulus : complex.abs c = 13 / 4 :=
by sorry

end evaluate_modulus_l491_491733


namespace digit_C_equals_one_l491_491992

-- Define the scope of digits
def is_digit (n : ℕ) : Prop := n < 10

-- Define the equality for sums of digits
def sum_of_digits (A B C : ℕ) : Prop := A + B + C = 10

-- Main theorem to prove C = 1
theorem digit_C_equals_one (A B C : ℕ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hSum : sum_of_digits A B C) : C = 1 :=
sorry

end digit_C_equals_one_l491_491992


namespace circle_and_line_properties_l491_491519

noncomputable def polar_coordinate_equation (ρ θ : ℝ) : Prop := 
  ρ^2 - 6 * ρ * Real.cos θ + 7 = 0

noncomputable def line_equation (x y : ℝ) : Prop := 
  y = Real.sqrt 35 * x

theorem circle_and_line_properties :
  (∀ (x y : ℝ), y = x + 5 → x = 3 ∧ y = 0 → ρ θ, polar_coordinate_equation ρ θ) ∧
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi / 2 ∧ ρ > 0 ∧ 
     (∀ {OA OB : ℝ}, 1 / Real.abs OA + 1 / Real.abs OB = 1 / 7 → α = y → α = x) → 
     ∃ x y, line_equation x y) :=
by
  sorry

end circle_and_line_properties_l491_491519


namespace num_values_of_n_with_prime_sum_of_divisors_l491_491890

def f (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, n % d = 0).sum

theorem num_values_of_n_with_prime_sum_of_divisors :
  (Finset.filter (λ n, Nat.Prime (f n)) (Finset.range 31)).card = 5 := by
  sorry

end num_values_of_n_with_prime_sum_of_divisors_l491_491890


namespace cab_is_late_l491_491975

def usual_speed (S : ℝ) : ℝ := S
def reduced_speed (S : ℝ) : ℝ := (5 / 6) * S
def usual_time : ℝ := 75
def increased_time (T : ℝ) : ℝ := (6 / 5) * T

theorem cab_is_late (S : ℝ) (T : ℝ) (T' : ℝ) :
  reduced_speed S * T' = usual_speed S * T → 
  T = usual_time →
  T' = increased_time T →
  T' - T = 15 :=
by
  sorry

end cab_is_late_l491_491975


namespace range_of_k_l491_491430

-- Define the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + k

-- Define the condition that the function does not pass through the third quadrant
def does_not_pass_third_quadrant (k : ℝ) : Prop :=
  ∀ x : ℝ, (x < 0 ∧ linear_function k x < 0) → false

-- Theorem statement proving the range of k
theorem range_of_k (k : ℝ) : does_not_pass_third_quadrant k ↔ (0 ≤ k ∧ k < 2) :=
by
  sorry

end range_of_k_l491_491430


namespace maximize_profit_l491_491507

noncomputable def P (x : ℝ) : ℝ := 12 + 10 * x
  
def Q (x : ℝ) : ℝ := 
  if x ≤ 16 then 
    -0.5 * x^2 + 22 * x 
  else 
    224

noncomputable def f (x : ℝ) : ℝ := Q(x) - P(x)

theorem maximize_profit :
  f x = (if x ≤ 16 then -0.5 * x^2 + 12 * x - 12 else 212 - 10 * x) ∧
  ∀ x, f x ≤ 60 :=
sorry

end maximize_profit_l491_491507


namespace length_QR_l491_491098

variable (A B C Q R : Point)
variable (AB BC CA QR : ℝ)

-- Given conditions
axiom given_AB : AB = 13
axiom given_BC : BC = 12
axiom given_CA : CA = 5
axiom given_circle_P : ∃ (P : Circle) 
  (radius_P : ℝ) 
  (M : Point)
  (on_circle_C : P.on_circle C) 
  (tangent_to_CA : P.tangent CA M)
  (midpoint_M : M = (A + C) / 2) 
  (smallest_radius : ∀ (r : ℝ) (Q R : Point), 
      P.on_circle Q ∧ P.on_circle R → r ≥ radius_P) 
  (Q_on_AB : P.on_circle Q ∧ Q ≠ C ∧ Q ∈ segment A B) 
  (R_on_BC : P.on_circle R ∧ R ≠ C ∧ R ∈ segment B C), true

-- Prove that the length of segment QR is 5
theorem length_QR : QR = 5 :=
by
  sorry

end length_QR_l491_491098


namespace trapezoid_ratio_l491_491887

-- Define the isosceles trapezoid properties and the point inside it
noncomputable def isosceles_trapezoid (r s : ℝ) (hr : r > s) (triangle_areas : List ℝ) : Prop :=
  triangle_areas = [2, 3, 4, 5]

-- Define the problem statement
theorem trapezoid_ratio (r s : ℝ) (hr : r > s) (areas : List ℝ) (hareas : isosceles_trapezoid r s hr areas) :
  r / s = 2 + Real.sqrt 2 := sorry

end trapezoid_ratio_l491_491887


namespace intersection_A_B_is_1_4_close_l491_491137

noncomputable def A : Set ℝ := {x | 1 < x ∧ x < 5}
noncomputable def B : Set ℝ := {x | x^2 - 3 * x - 4 ≤ 0}
noncomputable def intersection_set : Set ℝ := A ∩ B

theorem intersection_A_B_is_1_4_close : intersection_set = {x | 1 < x ∧ x ≤ 4} :=
by
  sorry

end intersection_A_B_is_1_4_close_l491_491137


namespace matrix_sequence_product_l491_491334

theorem matrix_sequence_product :
  (List.foldl (λ acc m, acc ⬝ m) (1 : Matrix (Fin 2) (Fin 2) ℤ)
    (List.map (λ k, Matrix.of ![![1, 2 * k], ![0, 1]]) (List.range 50).map (λ n, n + 1))) =
  (Matrix.of ![![1, 2550], ![0, 1]]) :=
by
  sorry

end matrix_sequence_product_l491_491334


namespace number_of_students_who_went_to_church_l491_491964

-- Define the number of chairs and the number of students.
variables (C S : ℕ)

-- Define the first condition: 9 students per chair with one student left.
def condition1 := S = 9 * C + 1

-- Define the second condition: 10 students per chair with one chair vacant.
def condition2 := S = 10 * C - 10

-- The theorem to be proved.
theorem number_of_students_who_went_to_church (h1 : condition1 C S) (h2 : condition2 C S) : S = 100 :=
by
  -- Proof goes here
  sorry

end number_of_students_who_went_to_church_l491_491964


namespace rectangular_prism_volume_l491_491386

theorem rectangular_prism_volume (a b c V : ℝ) (h1 : a * b = 20) (h2 : b * c = 12) (h3 : a * c = 15) (hb : b = 5) : V = 75 :=
  sorry

end rectangular_prism_volume_l491_491386


namespace lcm_hcf_product_l491_491849

theorem lcm_hcf_product (A B : ℕ) (h_prod : A * B = 18000) (h_hcf : Nat.gcd A B = 30) : Nat.lcm A B = 600 :=
sorry

end lcm_hcf_product_l491_491849


namespace monotonicity_of_g_l491_491042

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.log x
noncomputable def g (a : ℝ) (x : ℝ) := f(a, x) - (x - 1) / (x + 1)

theorem monotonicity_of_g (a : ℝ) (x : ℝ) (h0 : a > 0) :
  (if a ≥ (1 / 2) then ∀ x > 0, g(a, x) is strictly increasing
   else ∃ x1 x2 > 0, g(a, x) is strictly increasing on (0, x1), 
                        strictly decreasing on (x1, x2), 
                        strictly increasing on (x2, +∞)) := sorry

end monotonicity_of_g_l491_491042


namespace share_per_person_is_135k_l491_491104

noncomputable def calculate_share : ℝ :=
  (0.90 * (500000 * 1.20)) / 4

theorem share_per_person_is_135k : calculate_share = 135000 :=
by
  sorry

end share_per_person_is_135k_l491_491104


namespace least_possible_value_of_c_l491_491966

theorem least_possible_value_of_c (a b c : ℕ) 
  (h1 : a + b + c = 60) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : b = a + 13) : c = 45 :=
sorry

end least_possible_value_of_c_l491_491966


namespace prop3_l491_491462

-- Definitions for distinct lines a and b, and distinct planes alpha and beta
variables {Line Plane : Type}
variable  (a b : Line)
variable  (alpha beta : Plane)

-- Definitions for perpendicular and parallel (to be fleshed out as precise Lean predicates)
variables (perp parallel : Line → Plane → Prop)
variables (parallel_lines : Line → Line → Prop)
variables  (in_plane : Line → Plane → Prop)

-- The proposition to be proved
theorem prop3 (h1 : perp a alpha) (h2 : parallel b alpha) : perp a b :=
sorry

end prop3_l491_491462


namespace find_f_of_6_minus_a_l491_491039

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(x-1) - 2 else -real.log (x + 1) / real.log 2

theorem find_f_of_6_minus_a (a : ℝ) (h₁ : f a = -3) : f (6 - a) = -7/4 :=
  sorry

end find_f_of_6_minus_a_l491_491039


namespace train_length_l491_491272

theorem train_length 
  (speed_kmh : ℝ) 
  (time_sec : ℝ) 
  (conversion_factor : 1000 / 3600 ≈ 5 / 18) -- Approximated conversion factor
  (speed : speed_kmh = 56) 
  (time : time_sec = 9) : 
  ∃ length_of_train : ℝ, length_of_train ≈ 140.04 :=
by
  sorry

end train_length_l491_491272


namespace modulus_of_z_l491_491777

open Complex

-- Define the given complex numbers and the condition
def a : ℂ := 8 + 6 * Complex.I
def b : ℂ := 5 + 12 * Complex.I
def z : ℂ := b / a

-- The theorem to prove |z| = 13 / 10
theorem modulus_of_z :
  |z| = 13 / 10 := by
  sorry

end modulus_of_z_l491_491777


namespace angle_AP_AB_l491_491830

-- Define the vectors
def AP : ℝ × ℝ := (1, Real.sqrt 3)
def PB : ℝ × ℝ := (- Real.sqrt 3, 1)

-- Define AB as the sum of AP and PB
def AB : ℝ × ℝ := (AP.1 + PB.1, AP.2 + PB.2)

-- Calculate the angle between AP and AB
def angle_between_vectors (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.acos ((v1.1 * v2.1 + v1.2 * v2.2) / (Real.sqrt (v1.1^2 + v1.2^2) * Real.sqrt (v2.1^2 + v2.2^2)))

-- Prove that the angle between AP and AB is π / 4
theorem angle_AP_AB : angle_between_vectors AP AB = Real.pi / 4 := by
  sorry

end angle_AP_AB_l491_491830


namespace OP_coordinate_l491_491761

-- Define the parameters as noncomputable real numbers
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def e : ℝ := sorry

-- Define the point P and its conditions
def is_between (x y z : ℝ) := x ≤ y ∧ y ≤ z

def P_condition (x : ℝ) :=
  ∃ a b c e : ℝ, 
  let LHS := (a - x) * (x - c),
  let RHS := 2 * (b - x) * (x - e),
  is_between b x c ∧ LHS = RHS

-- Prove that OP equals the specific value
theorem OP_coordinate : 
  ∃ (x : ℝ), P_condition x ∧ 
  x = (-(a - b + e - 2 * c) + sqrt ((a - b + e - 2 * c)^2 - 4 * (a * c - 2 * b * e))) / 2 :=
sorry

end OP_coordinate_l491_491761


namespace find_prime_factors_l491_491329

-- Define n and the prime numbers p and q
def n : ℕ := 400000001
def p : ℕ := 20201
def q : ℕ := 19801

-- Main theorem statement
theorem find_prime_factors (hn : n = p * q) 
  (hp : Prime p) 
  (hq : Prime q) : 
  n = 400000001 ∧ p = 20201 ∧ q = 19801 := 
by {
  sorry
}

end find_prime_factors_l491_491329


namespace minimum_value_of_m_plus_n_l491_491073

-- Define the conditions and goals as a Lean 4 statement with a proof goal.
theorem minimum_value_of_m_plus_n (m n : ℝ) (h : m * n > 0) (hA : m + n = 3 * m * n) : m + n = 4 / 3 :=
sorry

end minimum_value_of_m_plus_n_l491_491073


namespace maxim_birth_probability_l491_491157

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end maxim_birth_probability_l491_491157


namespace log_base_16_of_4_l491_491749

theorem log_base_16_of_4 : log 16 4 = 1 / 2 := by
  sorry

end log_base_16_of_4_l491_491749


namespace total_fencing_required_l491_491302

noncomputable def field_width : ℝ := 40
noncomputable def field_area : ℝ := 1200
noncomputable def first_obstacle_side : ℝ := 8
noncomputable def second_obstacle_side : ℝ := 4

theorem total_fencing_required :
  let field_length := field_area / field_width in
  let field_fencing := field_length + 2 * field_width in
  let first_obstacle_fencing := 4 * first_obstacle_side in
  let second_obstacle_fencing := 4 * second_obstacle_side in
  field_fencing + first_obstacle_fencing + second_obstacle_fencing = 148 :=
by
  sorry

end total_fencing_required_l491_491302


namespace complex_magnitude_l491_491033

theorem complex_magnitude (z : ℂ) (hz : |z| = 1) : |(z + 1) + complex.I * (7 - z)| ≠ 5 * real.sqrt 3 :=
sorry

end complex_magnitude_l491_491033


namespace arithmetic_sequence_formula_and_sum_l491_491864

theorem arithmetic_sequence_formula_and_sum 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ)
  (h0 : a 1 = 1) 
  (h1 : a 3 = -3)
  (hS : ∃ k, S k = -35):
  (∀ n, a n = 3 - 2 * n) ∧ (∃ k, S k = -35 ∧ k = 7) :=
by
  -- Given an arithmetic sequence where a_1 = 1 and a_3 = -3,
  -- prove that the general formula is a_n = 3 - 2n
  -- and the sum of the first k terms S_k = -35 implies k = 7
  sorry

end arithmetic_sequence_formula_and_sum_l491_491864


namespace log_base_16_of_4_l491_491747

theorem log_base_16_of_4 : log 16 4 = 1 / 2 := by
  sorry

end log_base_16_of_4_l491_491747


namespace dominoes_can_be_horizontal_l491_491997

def domino (c1 c2 : (ℕ × ℕ)) := (c1.1 = c2.1 ∧ (c1.2 = c2.2 + 1 ∨ c1.2 = c2.2 - 1)) ∨
                                  (c1.2 = c2.2 ∧ (c1.1 = c2.1 + 1 ∨ c1.1 = c2.1 - 1))

def non_overlapping (dominoes : set ((ℕ × ℕ) × (ℕ × ℕ))) : Prop :=
  ∀ (d1 d2 : (ℕ × ℕ) × (ℕ × ℕ)), d1 ∈ dominoes → d2 ∈ dominoes → d1 ≠ d2 →
    d1.fst ≠ d2.fst ∧ d1.fst ≠ d2.snd ∧ d1.snd ≠ d2.fst ∧ d1.snd ≠ d2.snd

theorem dominoes_can_be_horizontal :
  ∃ (dominoes : set ((ℕ × ℕ) × (ℕ × ℕ))),
    card dominoes = 32 ∧ non_overlapping dominoes → 
    ∃ f : (ℕ × ℕ) → (ℕ × ℕ), 
      (∀ c1 c2, domino c1 c2 → domino (f c1) (f c2)) ∧
      (∀ d ∈ dominoes, ∃ y, f d = (y, (y.1, y.2 + 1))) :=
sorry

end dominoes_can_be_horizontal_l491_491997


namespace benches_required_l491_491294

theorem benches_required (students_base5 : ℕ := 312) (base_student_seating : ℕ := 5) (seats_per_bench : ℕ := 3) : ℕ :=
  let chairs := 3 * base_student_seating^2 + 1 * base_student_seating^1 + 2 * base_student_seating^0
  let benches := (chairs / seats_per_bench) + if (chairs % seats_per_bench > 0) then 1 else 0
  benches

example : benches_required = 28 :=
by sorry

end benches_required_l491_491294


namespace cycle_reappear_l491_491869

/-- Given two sequences with cycle lengths 6 and 4, prove the sequences will align on line number 12 -/
theorem cycle_reappear (l1 l2 : ℕ) (h1 : l1 = 6) (h2 : l2 = 4) :
  Nat.lcm l1 l2 = 12 := by
  sorry

end cycle_reappear_l491_491869


namespace negation_of_universal_prop_l491_491049

theorem negation_of_universal_prop :
  (¬ (∀ x : ℝ, x ≥ 2)) ↔ (∃ x : ℝ, x < 2) :=
by 
  intro h
  use some x : ℝ
  sorry

end negation_of_universal_prop_l491_491049


namespace three_sum_at_least_fifty_l491_491436

theorem three_sum_at_least_fifty (a : Fin 7 → ℕ) (h_distinct : ∀ i j : Fin 7, i ≠ j → a i ≠ a j) (h_sum : (∑ i, a i) = 100) :
  ∃ (i j k : Fin 7), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i + a j + a k ≥ 50 :=
by
  sorry

end three_sum_at_least_fifty_l491_491436


namespace find_value_of_expression_l491_491067

-- Conditions translated to Lean 4 definitions
variable (a b : ℝ)
axiom h1 : (a^2 * b^3) / 5 = 1000
axiom h2 : a * b = 2

-- The theorem stating what we need to prove
theorem find_value_of_expression :
  (a^3 * b^2) / 3 = 2 / 705 :=
by
  sorry

end find_value_of_expression_l491_491067


namespace solve_floor_eq_l491_491927

noncomputable theory

open Real

def floor (x : ℝ) : ℤ := Int.floor x

theorem solve_floor_eq (n : ℤ) (x : ℝ) :
  (x ∈ Set.Ioo (2 * π * n) (π / 8 + 2 * π * n) ∨ 
   x ∈ Set.Ioc (π / 8 + 2 * π * n) (π / 4 + 2 * π * n) ∨ 
   x = 3 * π / 4 + 2 * π * n) →
  (2 * floor (cos (2 * x)) - floor (sin x) = 3 * floor (sin (4 * x))) :=
by
  sorry

end solve_floor_eq_l491_491927


namespace part1_part2_l491_491202

section
variable {a m : Real} (ha : a > 0) (hm : m ≠ 0)
variable {x y : Real}

-- Define the slope conditions and the curve equation for part 1
def curve_C (x y : Real) : Prop :=
  m * x^2 - y^2 = m * a^2

-- Define the circle equation for C1 (m = -1)
def curve_C1 (x y : Real) : Prop :=
  x^2 + y^2 = a^2

-- Define the foci positions for C2
noncomputable def F_1 (m a : Real) : Real × Real :=
  (-a * Real.sqrt(1 + m), 0)

noncomputable def F_2 (m a : Real) : Real × Real :=
  (a * Real.sqrt(1 + m), 0)

-- Statement for Question 1: Prove the equation of curve C
theorem part1 (hm : m ≠ 0): (m * x^2 - y^2 = m * a^2) :=
sorry

-- Statement for Question 2: Prove the existence of point N on C1 satisfying area condition
theorem part2 (m : Real) (hm : m ∈ Set.Icc (-1:ℝ) 0  ∪ Set.Ioi 0) : 
  ∃ (x₀ y₀ : Real), (x₀^2 + y₀^2 = a^2) ∧ ((1/2) * 2 * a * Real.sqrt (1 + m) * |y₀| = |m| * a^2) :=
sorry

end

end part1_part2_l491_491202


namespace fraction_of_short_students_l491_491495

theorem fraction_of_short_students 
  (total_students tall_students average_students : ℕ) 
  (htotal : total_students = 400) 
  (htall : tall_students = 90) 
  (haverage : average_students = 150) : 
  (total_students - (tall_students + average_students)) / total_students = 2 / 5 :=
by
  sorry

end fraction_of_short_students_l491_491495


namespace radical_axis_through_intersection_tangent_point_l491_491017

-- Definitions and conditions

structure Circle (S : Type) :=
(center : S)
(radius : ℝ)

variables {S : Type} [metric_space S]

def touches_externally (C1 C2 : Circle S) :=
dist C1.center C2.center = C1.radius + C2.radius

def tangent_point (C1 C2 : Circle S) := 
{P : S // dist P C1.center = C1.radius ∧ dist P C2.center = C2.radius ∧ touches_externally C1 C2}

noncomputable def radical_axis (C1 C2 : Circle S) : S → Prop :=
λ P, dist P C1.center ^ 2 - C1.radius ^ 2 = dist P C2.center ^ 2 - C2.radius ^ 2

-- Problem statement

theorem radical_axis_through_intersection_tangent_point 
  (S1 S2 S3 S4 : Circle S)
  (h12 : touches_externally S1 S2)
  (h23 : touches_externally S2 S3)
  (h34 : touches_externally S3 S4)
  (h41 : touches_externally S4 S1)
  (A1 : tangent_point S1 S2)
  (A2 : tangent_point S2 S3)
  (A3 : tangent_point S3 S4)
  (A4 : tangent_point S4 S1)
  (X : S)
  (hX : ∃ (X1 : S), X ∈ [A1, A4] ∧ X1 ∈ [A2, A3] ∧ X = X1) :
  radical_axis S1 S3 X :=
sorry

end radical_axis_through_intersection_tangent_point_l491_491017


namespace volunteer_task_assignment_l491_491619

theorem volunteer_task_assignment : 
  let num_volunteers := 5
  let num_tasks := 3
  ∃ num_assignments,
    num_assignments = 150 ∧
    (∀ v t,
      0 < num_assignments ∧
      num_assignments ≤ 3^num_volunteers - 3 * 2^num_volunteers + 3) :=
by
  sorry

end volunteer_task_assignment_l491_491619


namespace correct_calculation_l491_491263

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := 
by sorry

end correct_calculation_l491_491263


namespace hyperbola_eccentricity_l491_491031

theorem hyperbola_eccentricity (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) 
    (h : dist (complex.abs (-real.sqrt 2, 0)) (complex.abs ((b/a), 0)) = real.sqrt 5 / 5) :
    eccentricity (\frac{\sqrt{a^2 + b^2}}{a}) := sorry

end hyperbola_eccentricity_l491_491031


namespace value_of_x7_plus_64x2_l491_491894

-- Let x be a real number such that x^3 + 4x = 8.
def x_condition (x : ℝ) : Prop := x^3 + 4 * x = 8

-- We need to determine the value of x^7 + 64x^2.
theorem value_of_x7_plus_64x2 (x : ℝ) (h : x_condition x) : x^7 + 64 * x^2 = 128 :=
by
  sorry

end value_of_x7_plus_64x2_l491_491894


namespace sequence_value_2_pow_100_l491_491631

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (2 * n) = a n + n

theorem sequence_value_2_pow_100 (a : ℕ → ℕ) (h : sequence a) : a (2 ^ 100) = 2 ^ 100 :=
by
  cases h with h1 h2
  sorry

end sequence_value_2_pow_100_l491_491631


namespace upper_limit_even_digit_range_l491_491636

theorem upper_limit_even_digit_range :
  ∃ n, (∀ k, 500 < k ∧ k ≤ n → (k % 10 = 0 ∨ k % 10 = 2 ∨ k % 10 = 4 ∨ k % 10 = 6 ∨ k % 10 = 8)) ∧
  (∃ s : finset ℤ, s.card = 251 ∧ ∀ k ∈ s, 500 < k ∧ k ≤ n ∧ (k % 10 = 0 ∨ k % 10 = 2 ∨ k % 10 = 4 ∨ k % 10 = 6 ∨ k % 10 = 8)) → 
  n = 1002 :=
by
  sorry

end upper_limit_even_digit_range_l491_491636


namespace max_det_l491_491532

open Real

noncomputable def v : ℝ × ℝ × ℝ := (3, 2, -2)
noncomputable def w : ℝ × ℝ × ℝ := (2, -1, 4)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((a.2.2 * b.2.1 - a.2.1 * b.2.2), (a.1 * b.2.2 - a.2.2 * b.1), (a.2.1 * b.1 - a.1 * b.2.1))

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

theorem max_det :
  magnitude (cross_product v w) = sqrt 149 :=
by sorry

end max_det_l491_491532


namespace y_eq_fraction_x_l491_491128

variable (p : ℝ)

-- Conditions
def x : ℝ := 1 + 2^p
def y : ℝ := 1 + 2^(-p)

-- Proof problem statement
theorem y_eq_fraction_x (p : ℝ) : y p = x p / (x p - 1) :=
  sorry

end y_eq_fraction_x_l491_491128


namespace each_person_gets_after_taxes_l491_491105

-- Definitions based strictly on problem conditions
def house_price : ℝ := 500000
def market_multiplier : ℝ := 1.2
def brothers_count : ℕ := 3
def tax_rate : ℝ := 0.1

-- Derived conditions
def selling_price : ℝ := house_price * market_multiplier
def total_people : ℕ := 1 + brothers_count
def share_before_taxes : ℝ := selling_price / total_people
def tax_amount_per_person : ℝ := share_before_taxes * tax_rate
def final_amount_per_person : ℝ := share_before_taxes - tax_amount_per_person

-- Problem: Prove the final amount each person receives
theorem each_person_gets_after_taxes : final_amount_per_person = 135000 := by
  sorry

end each_person_gets_after_taxes_l491_491105


namespace percentage_solution_l491_491071

def percentage_of (p : ℝ) (x : ℝ) : ℝ := (p / 100) * x

theorem percentage_solution :
  ∃ P : ℝ, percentage_of P 600 = percentage_of 50 960 ∧ P = 80 :=
by
  sorry

end percentage_solution_l491_491071


namespace distribute_objects_l491_491406

theorem distribute_objects (n r : ℕ) (h : n ≤ r) :
  ∃ ways : ℕ, ways = Nat.choose (r - 1) (n - 1) ∧ ways = ways :=
by
  sorry

end distribute_objects_l491_491406


namespace new_angle_intersection_l491_491678

theorem new_angle_intersection (n m : ℕ) (h₁ : n > 4) (h₂ : m > 3) (h₃ : m < n) : 
  let internal_angle_larger := (n - 2) * 180 / n,
      internal_angle_smaller := (m - 2) * 180 / m,
      half_external_angle_larger := 180 / n,
      new_angle := half_external_angle_larger * 2 + internal_angle_smaller
  in new_angle = 360 / n + (m - 2) * 180 / m := sorry

end new_angle_intersection_l491_491678


namespace cindy_correct_answer_l491_491340

theorem cindy_correct_answer (x : ℝ) (h : (x - 5) / 7 = 15) :
  (x - 7) / 5 = 20.6 :=
by
  sorry

end cindy_correct_answer_l491_491340


namespace coefficient_x3_in_expansion_l491_491598

theorem coefficient_x3_in_expansion :
  ∑ r in finset.range 6, (nat.choose 5 r) * (2:ℤ)^(5-r) * (-1:ℤ)^r * x^(5 - 2 * r) = -80 * x^3 :=
by
  sorry

end coefficient_x3_in_expansion_l491_491598


namespace Amanda_tickets_third_day_l491_491692

theorem Amanda_tickets_third_day :
  (let total_tickets := 80
   let first_day_tickets := 5 * 4
   let second_day_tickets := 32

   total_tickets - (first_day_tickets + second_day_tickets) = 28) :=
by
  sorry

end Amanda_tickets_third_day_l491_491692


namespace complex_mag_calc_l491_491730

noncomputable def complex_mag : ℂ := -3 - (5 / 4) * complex.I

theorem complex_mag_calc : complex.abs complex_mag = 13 / 4 := 
by sorry

end complex_mag_calc_l491_491730


namespace log_base_16_of_4_l491_491739

theorem log_base_16_of_4 : 
  (16 = 2^4) →
  (4 = 2^2) →
  (∀ (b a c : ℝ), b > 0 → b ≠ 1 → c > 0 → c ≠ 1 → log b a = log c a / log c b) →
  log 16 4 = 1 / 2 :=
by
  intros h1 h2 h3
  sorry

end log_base_16_of_4_l491_491739


namespace sparrows_among_non_pigeons_l491_491079

theorem sparrows_among_non_pigeons (perc_sparrows perc_pigeons perc_parrots perc_crows : ℝ)
  (h_sparrows : perc_sparrows = 0.40)
  (h_pigeons : perc_pigeons = 0.20)
  (h_parrots : perc_parrots = 0.15)
  (h_crows : perc_crows = 0.25) :
  (perc_sparrows / (1 - perc_pigeons) * 100) = 50 :=
by
  sorry

end sparrows_among_non_pigeons_l491_491079


namespace sum_of_cubes_of_roots_l491_491107

open Polynomial

variable {R : Type*} [CommRing R] [IsDomain R]

theorem sum_of_cubes_of_roots (n : ℕ) (a : Fin n → R) (p : R[X])
  (h_monic : p.monic) (h_deg : p.natDegree = n)
  (h_coeff : p.coeff (n - 1) = -2) (h_a_eq_neg_aₙ₋₂ : p.coeff (n - 1) = -a (n - 2))
  (h_zero : p.coeff (n - 3) = 0) :
  (p.roots.sum ^ 3 - 3 * p.roots.sum * p.roots.prod (coe_fn Polynomial.Mul) + 3 * 0) = -4 :=
by
  sorry

end sum_of_cubes_of_roots_l491_491107


namespace find_value_of_2a_minus_d_l491_491064

def g (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (2 * c * x - d)

theorem find_value_of_2a_minus_d (a b c d : ℝ) (h1 : abcd ≠ 0)
    (h2 : ∀ x, x ≠ d / (2 * c) → g a b c d (g a b c d x) = x) :
  2 * a - d = 0 := by 
  sorry

end find_value_of_2a_minus_d_l491_491064


namespace ratio_natasha_carla_l491_491566

noncomputable def natasha_money : ℝ := 60
noncomputable def cosima_money := arbitrary ℝ
noncomputable def carla_money : ℝ := 2 * cosima_money

axiom profit_condition : 36 = (2 / 5) * (natasha_money + carla_money + cosima_money)

theorem ratio_natasha_carla (h : profit_condition) : natasha_money / carla_money = 3 :=
by
  sorry

end ratio_natasha_carla_l491_491566


namespace sequence_A_decreases_sequence_G_decreases_l491_491538

noncomputable def A : ℕ → ℝ := λ n : ℕ,
  if n = 0 then (x + y) / 2 else
  (A (n - 1) + G (n - 1)) / 2

noncomputable def G : ℕ → ℝ := λ n : ℕ,
  if n = 0 then real.sqrt (x * y) else
  real.sqrt (A (n - 1) * G (n - 1))

theorem sequence_A_decreases (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  ∀ n : ℕ, A x y (n + 1) < A x y n :=
by
  sorry

theorem sequence_G_decreases (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  ∀ n : ℕ, G x y (n + 1) < G x y n :=
by
  sorry

end sequence_A_decreases_sequence_G_decreases_l491_491538


namespace secretary_longest_hours_l491_491184

theorem secretary_longest_hours (x : ℕ) : 
  let s1 := x,
      s2 := 2 * x,
      s3 := 5 * x,
      total := s1 + s2 + s3
  in total = 120 → s3 = 75 :=
by
  intros h
  sorry

end secretary_longest_hours_l491_491184


namespace solution_is_singleton_l491_491950

def solution_set : Set (ℝ × ℝ) := { (x, y) | 2 * x + y = 3 ∧ x - 2 * y = -1 }

theorem solution_is_singleton : solution_set = { (1, 1) } :=
by
  sorry

end solution_is_singleton_l491_491950


namespace tetrahedron_volume_l491_491522

open Real

noncomputable def volume_of_tetrahedron
  (AC : ℝ) (angle_BC : ℝ) (angle_CB : ℝ) (dist_AB_folded : ℝ) : ℝ :=
let AB := AC / cos angle_CB,
    BC := AC * tan angle_CB,
    M := AC * sqrt 3,
    base_area := (1 / 2) * AC * BC,
    height := 1 in
(1 / 3) * base_area * height

theorem tetrahedron_volume 
  (AC : ℝ := 2) 
  (angle_C : ℝ := π / 2) 
  (angle_B : ℝ := π / 6) 
  (dist_AB_folded : ℝ := 2 * sqrt 2) : 
  volume_of_tetrahedron AC angle_B angle_C dist_AB_folded = (2 * sqrt 2) / 3 :=
sorry

end tetrahedron_volume_l491_491522


namespace motorist_travel_time_l491_491672

noncomputable def total_time (dist1 dist2 speed1 speed2 : ℝ) : ℝ :=
  (dist1 / speed1) + (dist2 / speed2)

theorem motorist_travel_time (speed1 speed2 : ℝ) (total_dist : ℝ) (half_dist : ℝ) :
  speed1 = 60 → speed2 = 48 → total_dist = 324 → half_dist = total_dist / 2 →
  total_time half_dist half_dist speed1 speed2 = 6.075 :=
by
  intros h1 h2 h3 h4
  simp [total_time, h1, h2, h3, h4]
  sorry

end motorist_travel_time_l491_491672


namespace eliza_height_is_68_l491_491393

-- Define the known heights of the siblings
def height_sibling_1 : ℕ := 66
def height_sibling_2 : ℕ := 66
def height_sibling_3 : ℕ := 60

-- The total height of all 5 siblings combined
def total_height : ℕ := 330

-- Eliza is 2 inches shorter than the last sibling
def height_difference : ℕ := 2

-- Define the heights of the siblings
def height_remaining_siblings := total_height - (height_sibling_1 + height_sibling_2 + height_sibling_3)

-- The height of the last sibling
def height_last_sibling := (height_remaining_siblings + height_difference) / 2

-- Eliza's height
def height_eliza := height_last_sibling - height_difference

-- We need to prove that Eliza's height is 68 inches
theorem eliza_height_is_68 : height_eliza = 68 := by
  sorry

end eliza_height_is_68_l491_491393


namespace positive_perfect_squares_less_than_4_million_div_36_l491_491465

noncomputable def count_perfect_squares : ℕ :=
  let upper_limit := 2000
  let multiples_of_12 := (upper_limit - 1) / 12 # Number of multiples of 12 below 2000 (not inclusive of 2000)
  multiples_of_12 + 1 # Include the 2000 multiple itself

theorem positive_perfect_squares_less_than_4_million_div_36 : count_perfect_squares = 166 := by sorry

end positive_perfect_squares_less_than_4_million_div_36_l491_491465


namespace find_the_numbers_l491_491727

theorem find_the_numbers (a1 a2 a3 a4 a5 a6 : ℕ) :
  (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧ a1 ≠ a6 ∧
   a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧ a2 ≠ a6 ∧
   a3 ≠ a4 ∧ a3 ≠ a5 ∧ a3 ≠ a6 ∧ 
   a4 ≠ a5 ∧ a4 ≠ a6 ∧ 
   a5 ≠ a6) ∧
  (a1 ≤ 7 ∧ a2 ≤ 7 ∧ a3 ≤ 7 ∧ a4 ≤ 7 ∧ a5 ≤ 7 ∧ a6 ≤ 7) ∧
  (a1 + a2 + a3 + a4 + a5 + a6 = 25) →
  {a1, a2, a3, a4, a5, a6} = {1, 2, 4, 5, 6, 7} :=
by
  sorry

end find_the_numbers_l491_491727


namespace mrs_oaklyn_profit_is_correct_l491_491564

def cost_of_buying_rugs (n : ℕ) (cost_per_rug : ℕ) : ℕ :=
  n * cost_per_rug

def transportation_fee (n : ℕ) (fee_per_rug : ℕ) : ℕ :=
  n * fee_per_rug

def selling_price_before_tax (n : ℕ) (price_per_rug : ℕ) : ℕ :=
  n * price_per_rug

def total_tax (price_before_tax : ℕ) (tax_rate : ℕ) : ℕ :=
  price_before_tax * tax_rate / 100

def total_selling_price_after_tax (price_before_tax : ℕ) (tax_amount : ℕ) : ℕ :=
  price_before_tax + tax_amount

def profit (selling_price_after_tax : ℕ) (cost_of_buying : ℕ) (transport_fee : ℕ) : ℕ :=
  selling_price_after_tax - (cost_of_buying + transport_fee)

def rugs := 20
def cost_per_rug := 40
def transport_fee_per_rug := 5
def price_per_rug := 60
def tax_rate := 10

theorem mrs_oaklyn_profit_is_correct : 
  profit 
    (total_selling_price_after_tax 
      (selling_price_before_tax rugs price_per_rug) 
      (total_tax (selling_price_before_tax rugs price_per_rug) tax_rate)
    )
    (cost_of_buying_rugs rugs cost_per_rug) 
    (transportation_fee rugs transport_fee_per_rug) 
  = 420 :=
by sorry

end mrs_oaklyn_profit_is_correct_l491_491564


namespace bus_routes_in_city_l491_491082

noncomputable def bus_routes_problem (n : ℕ) : ℕ :=
  if n = 3 then 1 else if n = 7 then 7 else 0

theorem bus_routes_in_city {n : ℕ} (h1 : ∀ route, route ∈ {3}) (h2 : ∀ stops, ∃ route, route has_bus_route stops)
    (h3 : ∀ route1 route2, route1 ≠ route2 → ∃ common_stop, common_stop ∈ route1 ∧ common_stop ∈ route2) :
  bus_routes_problem n = 1 ∨ bus_routes_problem n = 7 :=
sorry

end bus_routes_in_city_l491_491082


namespace dandelion_seed_production_l491_491569

theorem dandelion_seed_production :
  ∀ (initial_seeds : ℕ), initial_seeds = 50 →
  ∀ (germination_rate : ℚ), germination_rate = 1 / 2 →
  ∀ (new_seed_rate : ℕ), new_seed_rate = 50 →
  (initial_seeds * germination_rate * new_seed_rate) = 1250 :=
by
  intros initial_seeds h1 germination_rate h2 new_seed_rate h3
  sorry

end dandelion_seed_production_l491_491569


namespace distance_from_focus_to_asymptote_is_sqrt3_l491_491605

-- Define the given hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 3 = 1

-- Coordinates of the right focus of the hyperbola
def focus : ℝ × ℝ :=
  (sqrt 5, 0)

-- Equation of the asymptote of the hyperbola
def asymptote_eq (x y : ℝ) : Prop :=
  2 * y - sqrt 6 * x = 0

-- Definition of the distance from a point to a line
def distance_point_line (f : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * f.1 + b * f.2 + c) / sqrt (a^2 + b^2)

-- The distance from the focus to the asymptote is √3
theorem distance_from_focus_to_asymptote_is_sqrt3 : 
  distance_point_line focus (sqrt 6) (-2) 0 = sqrt 3 :=
by
  sorry

end distance_from_focus_to_asymptote_is_sqrt3_l491_491605


namespace ineq_factorial_equality_at_one_iff_l491_491172

theorem ineq_factorial (n : ℕ) (hn : 1 ≤ n) : (1/3 : ℚ) * n^2 + (1/2 : ℚ) * n + (1 / 6 : ℚ) ≥ (n.factorial : ℚ) ^ (2 / n : ℚ) :=
sorry

theorem equality_at_one_iff (n : ℕ) (hn : 1 ≤ n) : ((1/3 : ℚ) * n^2 + (1/2 : ℚ) * n + (1 / 6 : ℚ) = (n.factorial : ℚ) ^ (2 / n : ℚ)) ↔ n = 1 :=
sorry

end ineq_factorial_equality_at_one_iff_l491_491172


namespace total_profit_when_investing_1_25_in_A_maximize_total_profit_l491_491506

-- Part 1: Total profit when investing 1.25 million yuan in City A
theorem total_profit_when_investing_1_25_in_A:
  let a := 1.25
  let b := 2 - a
  let P := 2 * Real.sqrt(5 * a) - 8
  let Q := (1 / 4) * b + 3
  P + Q = 63.75 :=
by
  sorry

-- Part 2: Invest to maximize total profit
noncomputable def total_profit (a : ℝ) : ℝ :=
  2 * Real.sqrt(5 * a) - 8 + (1 / 4) * (2 - a) + 3

theorem maximize_total_profit:
  let a := 0.8
  let b := 2 - a
  total_profit a = 65 :=
by
  sorry

end total_profit_when_investing_1_25_in_A_maximize_total_profit_l491_491506


namespace sum_of_solutions_eq_145_l491_491336

theorem sum_of_solutions_eq_145 : 
  let f (x : ℝ) := |2 * x - |100 - 2 * x||
  let solutions := { x | f x = x }
  finset.sum (solutions.to_finset) = 145 :=
by
  let f (x : ℝ) := |2 * x - |100 - 2 * x||
  have solutions : set ℝ := { x | f x = x }
  sorry

end sum_of_solutions_eq_145_l491_491336


namespace maxim_birth_probability_l491_491159

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end maxim_birth_probability_l491_491159


namespace inequality_system_solution_l491_491455

theorem inequality_system_solution (a b x : ℝ) 
  (h1 : x - a > 2)
  (h2 : x + 1 < b)
  (h3 : -1 < x)
  (h4 : x < 1) :
  (a + b) ^ 2023 = -1 :=
by 
  sorry

end inequality_system_solution_l491_491455


namespace evaluate_expression_l491_491395

theorem evaluate_expression : sqrt (9 / 4) + sqrt (16 / 9) = 17 / 6 :=
by
  sorry

end evaluate_expression_l491_491395


namespace lap_time_improvement_correct_l491_491559

def initial_lap_time (total_time : ℕ) (laps : ℕ) : ℝ :=
  total_time / laps

def current_lap_time (total_time : ℕ) (laps : ℕ) : ℝ :=
  total_time / laps

noncomputable def lap_time_improvement (initial_time current_time : ℝ) : ℝ :=
  initial_time - current_time

theorem lap_time_improvement_correct :
  let t1 := initial_lap_time 45 15 in
  let t2 := current_lap_time 42 18 in
  lap_time_improvement t1 t2 = 0.67 :=
by
  sorry

end lap_time_improvement_correct_l491_491559


namespace log_base_16_of_4_eq_half_l491_491738

noncomputable def logBase (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem log_base_16_of_4_eq_half :
  logBase 16 4 = 1 / 2 := by
sorry

end log_base_16_of_4_eq_half_l491_491738


namespace maxima_s_l491_491113

variables {α : Type*} [linear_ordered_comm_ring α]

def is_permutation (a b : list α) : Prop :=
∀ x, list.count x a = list.count x b

def score (x b : list α) : α :=
(list.zip_with (λ xi bi, (xi - bi)^2) x b).sum

def satisfies_condition (b : list α) (k : ℕ) : Prop :=
(b.take k).sum ≥ 0

noncomputable def max_score (a : list α) : α :=
(list.permutations a).map (λ σ, list.inf (list.sort (score sorry σ))).sup

theorem maxima_s (a : list α) (h : a.sum = 0) :
  ∀ σ, (∀ k ∈ list.range (σ.length + 1), satisfies_condition σ k) ↔ max_score a = score sorry σ :=
sorry

end maxima_s_l491_491113


namespace pairs_of_polygons_with_angle_difference_l491_491838

theorem pairs_of_polygons_with_angle_difference :
  ∃ (pairs : ℕ), pairs = 52 ∧ ∀ (n k : ℕ), n > k ∧ (360 / k - 360 / n = 1) :=
sorry

end pairs_of_polygons_with_angle_difference_l491_491838


namespace average_minutes_run_per_day_l491_491328

theorem average_minutes_run_per_day (f : ℕ) (h_nonzero : f ≠ 0)
  (third_avg fourth_avg fifth_avg : ℕ)
  (third_avg_eq : third_avg = 14)
  (fourth_avg_eq : fourth_avg = 18)
  (fifth_avg_eq : fifth_avg = 8)
  (third_count fourth_count fifth_count : ℕ)
  (third_count_eq : third_count = 3 * fourth_count)
  (fourth_count_eq : fourth_count = f / 2)
  (fifth_count_eq : fifth_count = f) :
  (third_avg * third_count + fourth_avg * fourth_count + fifth_avg * fifth_count) / (third_count + fourth_count + fifth_count) = 38 / 3 :=
by
  sorry

end average_minutes_run_per_day_l491_491328


namespace number_of_ways_to_select_one_person_number_of_ways_to_select_one_person_each_type_l491_491292

-- Definitions based on the conditions
def peopleO : ℕ := 28
def peopleA : ℕ := 7
def peopleB : ℕ := 9
def peopleAB : ℕ := 3

-- Proof for Question 1
theorem number_of_ways_to_select_one_person : peopleO + peopleA + peopleB + peopleAB = 47 := by
  sorry

-- Proof for Question 2
theorem number_of_ways_to_select_one_person_each_type : peopleO * peopleA * peopleB * peopleAB = 5292 := by
  sorry

end number_of_ways_to_select_one_person_number_of_ways_to_select_one_person_each_type_l491_491292


namespace range_of_a_l491_491823

open Real

noncomputable def f (x : ℝ) : ℝ := abs (log x)

noncomputable def g (x : ℝ) : ℝ := 
  if 0 < x ∧ x ≤ 1 then 0 
  else abs (x^2 - 4) - 2

noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |h x| = a → has_four_real_roots : Prop) ↔ (1 ≤ a ∧ a < 2 - log 2) := sorry

end range_of_a_l491_491823


namespace cost_of_paving_l491_491656

noncomputable def length := 5.5
noncomputable def width := 4
noncomputable def rate := 800

theorem cost_of_paving : 
  let area := length * width in
  let cost := area * rate in
  cost = 17600 :=
by
  sorry

end cost_of_paving_l491_491656


namespace lake_fish_count_l491_491418

theorem lake_fish_count 
  (white_ducks black_ducks multicolor_ducks : ℕ)
  (fish_per_white fish_per_black fish_per_multicolor : ℕ)
  (h_white : white_ducks = 3) 
  (h_black : black_ducks = 7) 
  (h_multicolor : multicolor_ducks = 6) 
  (h_fish_white : fish_per_white = 5) 
  (h_fish_black : fish_per_black = 10) 
  (h_fish_multicolor : fish_per_multicolor = 12) :
  (white_ducks * fish_per_white + black_ducks * fish_per_black + multicolor_ducks * fish_per_multicolor) = 157 := 
by 
  rw [h_white, h_black, h_multicolor, h_fish_white, h_fish_black, h_fish_multicolor]
  exact dec_trivial -- 3 * 5 + 7 * 10 + 6 * 12 = 157

end lake_fish_count_l491_491418


namespace probability_born_in_2008_l491_491144

/-- Define the relevant dates and days calculation -/
def num_days_between : ℕ :=
  29 + 31 + 30 + 31 + 31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

def num_days_2008 : ℕ :=
  31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

theorem probability_born_in_2008 :
  (num_days_2008 : ℚ) / num_days_between = 244 / 365 :=
by
  -- The proof is omitted for brevity
  sorry

end probability_born_in_2008_l491_491144


namespace abs_h_minus_2k_l491_491717

-- Defining conditions from the problem:
def polynomial (x : ℝ) (h k : ℝ) := 3 * x^3 - h * x + k

-- Given conditions using the Remainder Theorem:
def condition1 (h k : ℝ) := polynomial 3 h k = 0
def condition2 (h k : ℝ) := polynomial (-1) h k = 0

-- Main statement we need to prove
theorem abs_h_minus_2k (h k : ℝ) (h_cond1 : condition1 h k) (h_cond2 : condition2 h k) :
  |h - 2 * k| = 57 :=
by sorry

end abs_h_minus_2k_l491_491717


namespace avg_one_fourth_class_l491_491610

variable (N : ℕ) -- Total number of students

-- Define the average grade for the entire class
def avg_entire_class : ℝ := 84

-- Define the average grade of three fourths of the class
def avg_three_fourths_class : ℝ := 80

-- Statement to prove
theorem avg_one_fourth_class (A : ℝ) (h1 : 1/4 * A + 3/4 * avg_three_fourths_class = avg_entire_class) : 
  A = 96 := 
sorry

end avg_one_fourth_class_l491_491610


namespace pyramid_volume_and_surface_area_l491_491933

-- Definitions based on the conditions
noncomputable def base_is_pentagon : Prop := base ABCDE is convex equilateral pentagon
noncomputable def length_MA_eq_one (v : vertex) : Prop := dist M v = 1 ∀ v ∈ {A, B, C, D, E}
noncomputable def MA_perp_MCD : Prop := ∀ (M A C D : Point), MA ⟂ Plane MCD

-- The theorem statement
theorem pyramid_volume_and_surface_area :
  base_is_pentagon → 
  length_MA_eq_one →
  MA_perp_MCD →
  volume M ABCDE ≈ 0.293 ∧ surface_area M ABCDE ≈ 3.28 :=
by
  sorry

end pyramid_volume_and_surface_area_l491_491933


namespace negation_of_existential_l491_491205

theorem negation_of_existential :
  ¬ (∃ x : ℝ, x^2 - 2 * x - 3 < 0) ↔ ∀ x : ℝ, x^2 - 2 * x - 3 ≥ 0 :=
by sorry

end negation_of_existential_l491_491205


namespace part1_q1_part1_q2_part2_l491_491680

section part1

-- Conditions for part 1
def P (event : String) : ℝ := sorry
def P_cond (event1 event2 : String) : ℝ := sorry

axiom initial_prob : P "A1" = 0.5 ∧ P "B1" = 0.5
axiom cond_prob1 : P_cond "A2" "A1" = 0.6
axiom cond_prob2 : P_cond "A2" "B1" = 0.8

theorem part1_q1 : P "A2" = 0.7 :=
by sorry

theorem part1_q2 : P_cond "A1" "A2" = 3/7 :=
by sorry

end part1


section part2

-- Survey data conditions
def n : ℕ := 100
def a : ℕ := 28
def b : ℕ := 57
def c : ℕ := 12
def d : ℕ := 3

-- Chi-square critical values for alpha levels
def x_alpha (alpha : ℝ) : ℝ :=
  match alpha with
  | 0.1   => 2.706
  | 0.05  => 3.841
  | 0.01  => 6.635
  | 0.005 => 7.879
  | _     => 0 -- Assuming default zero for unmatched alphas

-- Given alpha for our test
def alpha : ℝ := 0.005

-- Chi-square computation from given data
def chi_squared : ℝ :=
  n * ((a * d - b * c)^2 : ℕ) / (↑(a + b) * ↑(c + d) * ↑(a + c) * ↑(b + d))

theorem part2 :
  chi_squared > x_alpha alpha :=
by sorry

end part2

end part1_q1_part1_q2_part2_l491_491680


namespace problem_statement_l491_491466

theorem problem_statement (A B : ℝ) (hA : A = 10 * π / 180) (hB : B = 35 * π / 180) :
  (1 + Real.tan A) * (1 + Real.sin B) = 
  1 + Real.tan A + (Real.sqrt 2 / 2) * (Real.cos A - Real.sin A) + Real.tan A * (Real.sqrt 2 / 2) * (Real.cos A - Real.sin A) :=
by
  sorry

end problem_statement_l491_491466


namespace weeks_to_cover_expense_l491_491720

-- Definitions and the statement of the problem
def hourly_rate : ℕ := 20
def monthly_expense : ℕ := 1200
def weekday_hours : ℕ := 3
def saturday_hours : ℕ := 5

theorem weeks_to_cover_expense : 
  ∀ (w : ℕ), (5 * weekday_hours + saturday_hours) * hourly_rate * w ≥ monthly_expense → w >= 3 := 
sorry

end weeks_to_cover_expense_l491_491720


namespace g_linear_l491_491609

theorem g_linear (g : ℝ → ℝ) (h : ∀ d : ℝ, g(d + 2) - g(d) = 8) : g(1) - g(7) = -24 :=
sorry

end g_linear_l491_491609


namespace range_of_function_l491_491209

theorem range_of_function : ∀ x : ℝ, (x ≤ 1 / 2) → 1 - 2 * x ≥ 0 → (3 + x - real.sqrt (1 - 2 * x) ≤ 7 / 2) :=
by
  intro x h1 h2
  sorry

end range_of_function_l491_491209


namespace Amanda_ticket_sales_goal_l491_491694

theorem Amanda_ticket_sales_goal :
  let total_tickets : ℕ := 80
  let first_day_sales : ℕ := 5 * 4
  let second_day_sales : ℕ := 32
  total_tickets - (first_day_sales + second_day_sales) = 28 :=
by
  sorry

end Amanda_ticket_sales_goal_l491_491694


namespace simplify_and_evaluate_expr_l491_491589

theorem simplify_and_evaluate_expr :
  let x := (Real.sqrt 5 + 1) / 2 
  let y := (Real.sqrt 5 - 1) / 2 
  (x - 2*y)^2 + x*(5*y - x) - 4*y^2 = 1 :=
by
  let x := (Real.sqrt 5 + 1) / 2 
  let y := (Real.sqrt 5 - 1) / 2 
  sorry

end simplify_and_evaluate_expr_l491_491589


namespace even_odd_partitions_equal_l491_491780

def is_even_partition (n : ℕ) (xs : List ℕ) : Prop :=
  (∀ x ∈ xs, x % 2 = 0) ∧ xs.sum = n

def is_odd_partition (n : ℕ) (xs : List ℕ) : Prop :=
  (∀ x ∈ xs, x % 2 = 1) ∧ xs.sum = n

def count_even_partitions (n : ℕ) : ℕ :=
  (List.range (n + 1)).countp (λ k, is_even_partition n (List.replicate k k))

def count_odd_partitions (n : ℕ) : ℕ :=
  (List.range (n + 1)).countp (λ k, is_odd_partition n (List.replicate k k))

theorem even_odd_partitions_equal (n : ℕ) :
  count_even_partitions n = count_odd_partitions n ↔ n = 2 ∨ n = 4 :=
by
  sorry

end even_odd_partitions_equal_l491_491780


namespace mowing_lawn_time_l491_491872

theorem mowing_lawn_time (pay_mow : ℝ) (rate_hour : ℝ) (time_plant : ℝ) (charge_flowers : ℝ) :
  pay_mow = 15 → rate_hour = 20 → time_plant = 2 → charge_flowers = 45 → 
  (charge_flowers + pay_mow) / rate_hour - time_plant = 1 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- This is an outline, so the actual proof steps are omitted
  sorry

end mowing_lawn_time_l491_491872


namespace min_value_vector_sum_perpendicular_vector_sum_l491_491811

-- Definitions based on conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1
def perpendicular_lines_through (P Q : ℝ × ℝ) (C : ℝ × ℝ) : Prop := 
  ((P.fst - C.fst) * (Q.fst - C.fst) + (P.snd - C.snd) * (Q.snd - C.snd) = 0)

-- Definitions based on questions
constants (P Q F1 F2 : ℝ × ℝ)
constants (PF1 PF2 PQ O : ℝ × ℝ)

-- Question 1: Solution statement
theorem min_value_vector_sum (C : ℝ × ℝ) (O : ℝ × ℝ) (P F1 F2 : ℝ × ℝ) (h : ellipse P.fst P.snd) : 
  (P.fst = 1) ∨ (P.snd = 1) → 
  ‖PF1 + PF2‖ = 2 :=
sorry

-- Question 2: Solution statement
theorem perpendicular_vector_sum (C O : ℝ × ℝ) (P Q F1 F2 : ℝ × ℝ) (h1 : ellipse P.fst P.snd) (h2 : ellipse Q.fst Q.snd) (P_perp_Q : perpendicular_lines_through P Q C) : 
  (∀ k : ℝ, k = sqrt ((2 * sqrt 10 - 5) / 10)) :=
sorry

end min_value_vector_sum_perpendicular_vector_sum_l491_491811


namespace cylinder_radius_in_cone_l491_491679

-- Define the conditions
def cone_diameter := 18
def cone_height := 20
def cylinder_height_eq_diameter {r : ℝ} := 2 * r

-- Define the theorem to prove
theorem cylinder_radius_in_cone : ∃ r : ℝ, r = 90 / 19 ∧ (20 - 2 * r) / r = 20 / 9 :=
by
  sorry

end cylinder_radius_in_cone_l491_491679


namespace log_base_equality_l491_491743

theorem log_base_equality : log 4 / log 16 = 1 / 2 := 
by sorry

end log_base_equality_l491_491743


namespace compare_neg_fractions_l491_491359

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l491_491359


namespace simplify_expression_l491_491924

theorem simplify_expression : (2 * 3 * b * 4 * (b ^ 2) * 5 * (b ^ 3) * 6 * (b ^ 4)) = 720 * (b ^ 10) :=
by
  sorry

end simplify_expression_l491_491924


namespace compare_neg_fractions_l491_491362

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l491_491362


namespace compare_neg_fractions_l491_491364

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l491_491364


namespace measure_of_angle_A_and_area_of_triangle_l491_491855

theorem measure_of_angle_A_and_area_of_triangle (a b c : ℕ) (A : ℝ)
  (h1 : b = 4 * real.sqrt 2)
  (h2 : c = real.sqrt 2 * a)
  (m : ℝ × ℝ := (real.cos A, real.sin A))
  (n : ℝ × ℝ := (real.sqrt 2 - real.sin A, real.cos A))
  (dot_product_eq : prod.fst m * prod.fst n + prod.snd m * prod.snd n = 1): 
  (A = real.pi / 4) ∧ (1 / 2 * b * c * real.sin A = 16) :=
by 
  sorry

end measure_of_angle_A_and_area_of_triangle_l491_491855


namespace matrixSequenceProductCorrect_l491_491332

open Matrix

-- Define the sequence of matrices
def matrixSeq (k : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  if h : k > 0 ∧ k ≤ 50 then 
    ![![1, (2 * k : ℤ)],![0, 1]]
  else 
    1 -- Identity matrix for purposes of definition outside bounds

-- Define the product over the matrix sequence
def matrixProduct : Matrix (Fin 2) (Fin 2) ℤ :=
  List.prod $ List.map matrixSeq (List.range 1 51)

-- The expected resultant matrix
def expectedMatrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![1, 2550],![0, 1]]

-- The theorem to be proven
theorem matrixSequenceProductCorrect : matrixProduct = expectedMatrix := 
  sorry

end matrixSequenceProductCorrect_l491_491332


namespace average_speed_of_second_girl_l491_491969

-- Define the conditions
def opposite_speeds (v : ℝ) : Prop := 
  ∀ (t : ℝ) (d : ℝ), t = 5 ∧ d = 75 → d = t * (5 + v)

-- Define the proof problem
theorem average_speed_of_second_girl : ∃ v : ℝ, v = 10 ∧ opposite_speeds v :=
by {
  use 10,
  split,
  { refl },
  { intros t d h,
    cases h with ht hd,
    rw [ht, hd],
    norm_num,
  }
}

end average_speed_of_second_girl_l491_491969


namespace min_area_OBX_l491_491995

structure Point : Type :=
  (x : ℤ)
  (y : ℤ)

def O : Point := ⟨0, 0⟩
def B : Point := ⟨11, 8⟩

def area_triangle (A B C : Point) : ℚ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def in_rectangle (X : Point) : Prop :=
  0 ≤ X.x ∧ X.x ≤ 11 ∧ 0 ≤ X.y ∧ X.y ≤ 8

theorem min_area_OBX : ∃ (X : Point), in_rectangle X ∧ area_triangle O B X = 1 / 2 :=
sorry

end min_area_OBX_l491_491995


namespace relationship_among_a_b_c_l491_491124

noncomputable def a : ℝ := 0.2^2
noncomputable def b : ℝ := 2^0.2
noncomputable def c : ℝ := Real.log10 (a + b - 1)

theorem relationship_among_a_b_c : b > a ∧ a > c := by
  sorry

end relationship_among_a_b_c_l491_491124


namespace count_valid_prime_pairs_l491_491027

theorem count_valid_prime_pairs (x y : ℕ) (h₁ : Prime x) (h₂ : Prime y) (h₃ : x ≠ y) (h₄ : (621 * x * y) % (x + y) = 0) : 
  ∃ p, p = 6 := by
  sorry

end count_valid_prime_pairs_l491_491027


namespace lead_points_l491_491634

-- Define final scores
def final_score_team : ℕ := 68
def final_score_green : ℕ := 39

-- Prove the lead
theorem lead_points : final_score_team - final_score_green = 29 :=
by
  sorry

end lead_points_l491_491634


namespace units_digit_of_product_l491_491255

/-
Problem: What is the units digit of the product of the first three even positive composite numbers?
Conditions: 
- The first three even positive composite numbers are 4, 6, and 8.
Proof: Prove that the units digit of their product is 2.
-/

def even_positive_composite_numbers := [4, 6, 8]
def product := List.foldl (· * ·) 1 even_positive_composite_numbers
def units_digit (n : Nat) := n % 10

theorem units_digit_of_product : units_digit product = 2 := by
  sorry

end units_digit_of_product_l491_491255


namespace number_of_women_per_table_l491_491688

theorem number_of_women_per_table
  (tables : ℕ) (men_per_table : ℕ) 
  (total_customers : ℕ) (total_tables : tables = 9) 
  (men_at_each_table : men_per_table = 3) 
  (customers : total_customers = 90) 
  (total_men : 3 * 9 = 27) 
  (total_women : 90 - 27 = 63) :
  (63 / 9 = 7) :=
by
  sorry

end number_of_women_per_table_l491_491688


namespace lower_limit_of_prime_range_l491_491922

theorem lower_limit_of_prime_range :
  ∃ x : ℕ, (prime x ∧ x + 16 ≤ 85 ∧ x + 16 = 83) ∧ x = 67 :=
begin
  sorry
end

end lower_limit_of_prime_range_l491_491922


namespace inequality_solution_l491_491929

open Set

theorem inequality_solution (x : ℝ) :
  ((x^2 * (x + 1) / (-x^2 - 5*x + 6)) ≤ 0 ↔ (-6 < x ∧ x ≤ -1) ∨ (x = 0) ∨ (x > 1)) := 
begin
  sorry
end

end inequality_solution_l491_491929


namespace domain_f_when_m_1_value_of_m_for_odd_function_l491_491819

noncomputable def f (x m : ℝ) : ℝ := Real.logBase 2 ((x - m) / (x + 1))

open Set

-- Part 1: Domain of the function when m = 1
theorem domain_f_when_m_1 : 
  ∀ x : ℝ, (1 < x ∨ x < -1) ↔ 0 < ((x - 1) / (x + 1)) :=
sorry

-- Part 2: Value of m for f(x) to be an odd function
theorem value_of_m_for_odd_function :
  (∀ x : ℝ, f (-x) (1:ℝ) + f x (1:ℝ) = 0) ↔ (1 = 1) :=
sorry

end domain_f_when_m_1_value_of_m_for_odd_function_l491_491819


namespace sequence_geometric_l491_491582

theorem sequence_geometric (c q : ℝ) (h : c * q ≠ 0) : 
  ∃ r : ℝ, ∀ n : ℕ, a_{n+1} = r * a_n :=
by
  -- Define the given sequence
  let a : ℕ → ℝ := λ n, c * q^n 
  -- We need to show the existence of a constant ratio r
  use q
  intros n
  -- Prove the ratio between consecutive terms is constant
  have : a (n + 1) = c * q^(n + 1), by sorry
  show a (n + 1) = q * a n,
  calc
    a (n + 1) = c * q^(n + 1) : sorry
           ... = q * (c * q^n) : sorry
           ... = q * a n : by refl
  done
sorry

end sequence_geometric_l491_491582


namespace smallest_sum_of_two_3digit_numbers_l491_491971

-- Define a function that represents the condition:
def valid_3digit_number (a b c : ℕ) : Prop :=
  a ∈ {1, 2, 3} ∧ b ∈ {1, 2, 3, 7, 8, 9} ∧ c ∈ {1, 2, 3, 7, 8, 9} ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c

noncomputable def smallest_sum : ℕ := 
  @Classical.some (Σₜ : ℕ × ℕ, valid_3digit_number t.1 t.2) 
                  (by apply Classical.inhabitedNonempty;
                      exact ⟨(100 * 1 + 10 * 3 + 8, 100 * 2 + 10 * 7 + 9), by simp [valid_3digit_number]⟩)

theorem smallest_sum_of_two_3digit_numbers : smallest_sum = 417 :=
by sorry

end smallest_sum_of_two_3digit_numbers_l491_491971


namespace average_of_remaining_numbers_l491_491596

theorem average_of_remaining_numbers (sum : ℕ) (average : ℕ) (remaining_sum : ℕ) (remaining_average : ℚ) :
  (average = 90) →
  (sum = 1080) →
  (remaining_sum = sum - 72 - 84) →
  (remaining_average = remaining_sum / 10) →
  remaining_average = 92.4 :=
by
  sorry

end average_of_remaining_numbers_l491_491596


namespace max_angle_focus_ratio_l491_491034

def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in (x^2 / 16) + (y^2 / 4) = 1

def is_on_line (P : ℝ × ℝ) : Prop :=
  let (x, y) := P in x - sqrt 3 * y + 8 + 2 * sqrt 3 = 0

def is_focus (F : ℝ × ℝ) (a : ℝ) : Prop :=
  F = (-2 * sqrt 3, 0) ∨ F = (2 * sqrt 3, 0)

theorem max_angle_focus_ratio :
  ∀ (P F1 F2 : ℝ × ℝ),
    is_on_ellipse P →
    is_focus F1 4 →
    is_focus F2 4 →
    is_on_line P →
    (∠ F1 P F2 maximizes) → 
    (| P - F1 | / | P - F2 | = sqrt 3 - 1) :=
by sorry

end max_angle_focus_ratio_l491_491034


namespace ball_bounce_height_l491_491661

theorem ball_bounce_height (initial_height : ℝ) (r : ℝ) (k : ℕ) : 
  initial_height = 1000 → r = 1/2 → (r ^ k * initial_height < 1) → k = 10 := by
sorry

end ball_bounce_height_l491_491661


namespace find_a_value_l491_491476

theorem find_a_value (a : ℕ) (h : a^3 = 21 * 25 * 45 * 49) : a = 105 :=
sorry

end find_a_value_l491_491476


namespace count_distinct_ordered_pairs_l491_491758

theorem count_distinct_ordered_pairs :
  let lattice_points := [(3, 4), (-3, -4), (4, 3), (-4, -3), (5, 0), (-5, 0), (0, 5), (0, -5), 
                         (-3, 4), (3, -4), (-4, 3), (4, -3)] in
  let count_pairs := 12 * (12 - 1) / 2 - 6 + 12 in
  count_pairs = 72 :=
by
  let lattice_points := [(3, 4), (-3, -4), (4, 3), (-4, -3), (5, 0), (-5, 0), (0, 5), (0, -5), 
                         (-3, 4), (3, -4), (-4, 3), (4, -3)]
  let count_pairs := 12 * (12 - 1) / 2 - 6 + 12
  have count_pairs_eq : count_pairs = 72 := by sorry
  exact count_pairs_eq

end count_distinct_ordered_pairs_l491_491758


namespace quadratic_condition_l491_491768

theorem quadratic_condition (m : ℝ) : m^2 - 3 * m - 2 = 2 → m = 4 :=
by
  intro h
  have h_eq : m^2 - 3 * m - 4 = 0 := by linarith
  have h_roots : (m - 4) * (m + 1) = 0 := by poly
  cases eq_of_mul_eq_zero h_roots with m_4 m_neg1
  { exact m_4 }
  { exfalso, linarith }

end quadratic_condition_l491_491768


namespace lydia_ate_24_ounces_l491_491227

theorem lydia_ate_24_ounces (total_fruit_pounds : ℕ) (mario_oranges_ounces : ℕ) (nicolai_peaches_pounds : ℕ) (total_fruit_ounces mario_oranges_ounces_in_ounces nicolai_peaches_ounces_in_ounces : ℕ) :
  total_fruit_pounds = 8 →
  mario_oranges_ounces = 8 →
  nicolai_peaches_pounds = 6 →
  total_fruit_ounces = total_fruit_pounds * 16 →
  mario_oranges_ounces_in_ounces = mario_oranges_ounces →
  nicolai_peaches_ounces_in_ounces = nicolai_peaches_pounds * 16 →
  (total_fruit_ounces - mario_oranges_ounces_in_ounces - nicolai_peaches_ounces_in_ounces) = 24 :=
by
  sorry

end lydia_ate_24_ounces_l491_491227


namespace range_of_independent_variable_x_in_sqrt_function_l491_491627

theorem range_of_independent_variable_x_in_sqrt_function :
  (∀ x : ℝ, ∃ y : ℝ, y = sqrt (2 * x - 3)) → x ≥ 3 / 2 :=
sorry

end range_of_independent_variable_x_in_sqrt_function_l491_491627


namespace Tia_drove_192_more_miles_l491_491910

noncomputable def calculate_additional_miles (s_C t_C : ℝ) : ℝ :=
  let d_C := s_C * t_C
  let d_M := (s_C + 8) * (t_C + 3)
  let d_T := (s_C + 12) * (t_C + 4)
  d_T - d_C

theorem Tia_drove_192_more_miles (s_C t_C : ℝ) (h1 : d_M = d_C + 120) (h2 : d_M = (s_C + 8) * (t_C + 3)) : calculate_additional_miles s_C t_C = 192 :=
by {
  sorry
}

end Tia_drove_192_more_miles_l491_491910


namespace max_value_of_sequence_is_12_l491_491467

-- Definitions and conditions
def sequence (a : ℕ → ℤ) : Prop :=
  ∃ (S : ℕ → ℤ), (∀ n, S n = -n^2 + 6*n + 7) ∧ (∀ n, a (n + 1) = S (n + 1) - S n)

-- The theorem to prove the maximum value of the sequence
theorem max_value_of_sequence_is_12 : 
  ∀ (a : ℕ → ℤ), sequence a → ∃ n, a n = 12 :=
begin
  intros a h,
  sorry -- The proof is omitted
end

end max_value_of_sequence_is_12_l491_491467


namespace octahedron_common_sum_is_39_l491_491206

-- Define the vertices of the regular octahedron with numbers from 1 to 12
def vertices : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the property that the sum of four numbers at the vertices of each triangle face is the same
def common_sum (faces : List (List ℕ)) (k : ℕ) : Prop :=
  ∀ face ∈ faces, face.sum = k

-- Define the faces of the regular octahedron
def faces : List (List ℕ) := [
  [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 5, 9, 6],
  [2, 6, 10, 7], [3, 7, 11, 8], [4, 8, 12, 5], [1, 9, 2, 10]
]

-- Prove that the common sum is 39
theorem octahedron_common_sum_is_39 : common_sum faces 39 :=
  sorry

end octahedron_common_sum_is_39_l491_491206


namespace arithmetic_sequence_sum_10_l491_491798

noncomputable def sum_of_first_n_terms (a d : ℤ) (n : ℕ) : ℤ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem arithmetic_sequence_sum_10 (a_1 a_3 a_7 a_9 : ℤ)
    (h1 : ∃ a_1, a_3 = a_1 - 4)
    (h2 : a_7 = a_1 - 12)
    (h3 : a_9 = a_1 - 16)
    (h4 : a_7 * a_7 = a_3 * a_9)
    : sum_of_first_n_terms a_1 (-2) 10 = 110 :=
by 
  sorry

end arithmetic_sequence_sum_10_l491_491798


namespace proof_problem_l491_491516

def circle_center : ℝ × ℝ := (3, 0)

def line_equation (p : ℝ × ℝ) : Prop := p.1 + p.2 = 5

def circle_tangent_to_line (center : ℝ × ℝ) (line_eq : ℝ × ℝ → Prop) : Prop :=
  let d := abs (center.1 + center.2 - 5) / real.sqrt (1^2 + 1^2)
  d = real.sqrt 2

def polar_coordinate_equation (ρ θ : ℝ) : Prop :=
  ρ^2 - 6 * ρ * real.cos θ + 7 = 0

def given_condition (OA OB : ℝ) : Prop :=
  1/OA + 1/OB = 1/7

def Cartesian_equation_of_line (slope x : ℝ) : ℝ := slope * x

theorem proof_problem :
  (circle_tangent_to_line circle_center line_equation) →
  (∀ ρ θ, polar_coordinate_equation ρ θ) →
  (∀ α OA OB, (given_condition OA OB) ∧ (0 < α ∧ α < real.pi / 2 ∧ ρ > 0) →
    Cartesian_equation_of_line (real.sqrt 35) = λ x, real.sqrt 35 * x) :=
by
  intros h1 h2 h3
  sorry

end proof_problem_l491_491516


namespace probability_five_out_of_six_correct_l491_491223

noncomputable def probability_exactly_five_correct : ℕ → ℝ
  | 6 := 0
  | _ := sorry

theorem probability_five_out_of_six_correct :
  probability_exactly_five_correct 6 = 0 :=
begin
  simp [probability_exactly_five_correct],
end

end probability_five_out_of_six_correct_l491_491223


namespace arithmetic_expression_correct_l491_491972

theorem arithmetic_expression_correct :
  let a := 10
  let b := 10
  let c := 4
  let d := 2
  (d + c / a) * a = 24 :=
by
  let a := 10
  let b := 10
  let c := 4
  let d := 2
  calc
    (d + c / a) * a
        = (2 + 4 / 10) * 10 : by rw [←add_mul, mul_comm 10 (4 / 10), div_mul_cancel'] -- real arithmetic correctness
    ... = 24 : by norm_num [div_eq_mul_inv, ←mul_assoc, mul_inv_cancel_left]

end arithmetic_expression_correct_l491_491972


namespace projection_of_b_onto_a_l491_491024

def a : ℝ × ℝ × ℝ := (2, -1, 2)
def b : ℝ × ℝ × ℝ := (1, -2, 1)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def norm_squared (v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product v v

def scalar_mult (c : ℝ) (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (c * v.1, c * v.2, c * v.3)

def proj (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  scalar_mult (dot_product u v / norm_squared u) u

theorem projection_of_b_onto_a :
  proj a b = (4/3, -2/3, 4/3) :=
by
  sorry

end projection_of_b_onto_a_l491_491024


namespace probability_maxim_born_in_2008_l491_491147

def days_in_month : ℕ → ℕ → ℕ 
| 2007 9 := 29
| 2007 10 := 31
| 2007 11 := 30
| 2007 12 := 31
| 2008 1 := 31
| 2008 2 := 29   -- leap year
| 2008 3 := 31
| 2008 4 := 30
| 2008 5 := 31
| 2008 6 := 30
| 2008 7 := 31
| 2008 8 := 31
| _ _ := 0

def total_days : ℕ :=
list.sum [days_in_month 2007 9, days_in_month 2007 10, days_in_month 2007 11, days_in_month 2007 12,
          days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

def days_in_2008 : ℕ :=
list.sum [days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

theorem probability_maxim_born_in_2008 : 
  (days_in_2008 : ℚ) / (total_days : ℚ) = 244 / 365 :=
by
  -- Placeholder for actual proof. This theorem is set properly and ready for proof development.
  sorry

end probability_maxim_born_in_2008_l491_491147


namespace at_least_one_gt_one_l491_491534

variable (a b : ℝ)

theorem at_least_one_gt_one (h : a + b > 2) : a > 1 ∨ b > 1 :=
by
  sorry

end at_least_one_gt_one_l491_491534


namespace new_price_is_correct_l491_491621

-- Define the original price of the book and the percentage increase
def original_price : ℝ := 300
def percentage_increase : ℝ := 60
def percentage_multiplier : ℝ := percentage_increase / 100

-- Define the computation of the new price based on the given conditions
def new_price : ℝ := original_price * (1 + percentage_multiplier)

-- The problem statement: Prove that the new price is $480
theorem new_price_is_correct : new_price = 480 :=
by
  -- Proof goes here
  sorry

end new_price_is_correct_l491_491621


namespace ellipse_ratio_l491_491514

-- Define the points A, B, F1, F2 on the ellipse
variables {a b c : ℝ}

-- Definitions based on the problem's setup
def A := (a, 0)
def B := (0, b)
def F1 := (-c, 0)
def F2 := (c, 0)

-- Conditions given in the problem
axiom major_minor_dot_product_eq_zero :
  let AF1 := (a + c, 0)
  let AF2 := (a - c, 0)
  let BF1 := (c, b)
  let BF2 := (-c, b)
  (AF1.1 * AF2.1 + AF1.2 * AF2.2) + (BF1.1 * BF2.1 + BF1.2 * BF2.2) = 0

-- The theorem to prove
theorem ellipse_ratio : a^2 + b^2 - 2 * c^2 = 0 → (|sqrt (a^2 + b^2)| / |2 * c|) = (sqrt 2 / 2) :=
sorry

end ellipse_ratio_l491_491514


namespace slab_length_is_150_cm_l491_491996

def total_area : ℝ := 67.5 -- Total floor area in sq. meters
def num_slabs : ℕ := 30    -- Number of slabs

def area_per_slab (t_area : ℝ) (n_slabs : ℕ) : ℝ :=
  t_area / n_slabs

def area_in_sq_cm (area : ℝ) : ℝ :=
  area * 10000           -- Conversion factor from sq. meters to sq. centimeters
  
def length_of_slab (area_cm : ℝ) : ℝ :=
  real.sqrt area_cm      -- Length is the square root of the area

theorem slab_length_is_150_cm :
  length_of_slab (area_in_sq_cm (area_per_slab total_area num_slabs)) = 150 := by
  sorry

end slab_length_is_150_cm_l491_491996


namespace height_of_pyramid_l491_491667

-- Define the volumes
def volume_cube (s : ℕ) : ℕ := s^3
def volume_pyramid (b : ℕ) (h : ℕ) : ℕ := (b^2 * h) / 3

-- Given constants
def s := 6
def b := 12

-- Given volume equality
def volumes_equal (s : ℕ) (b : ℕ) (h : ℕ) : Prop :=
  volume_cube s = volume_pyramid b h

-- The statement to prove
theorem height_of_pyramid (h : ℕ) (h_eq : volumes_equal s b h) :
  h = 9 := sorry

end height_of_pyramid_l491_491667


namespace compare_negative_fractions_l491_491353

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end compare_negative_fractions_l491_491353


namespace garden_area_l491_491077

theorem garden_area (P b l: ℕ) (hP: P = 900) (hb: b = 190) (hl: l = P / 2 - b):
  l * b = 49400 := 
by
  sorry

end garden_area_l491_491077


namespace aimee_poll_l491_491315

theorem aimee_poll (W P : ℕ) (h1 : 0.35 * W = 21) (h2 : 2 * W = P) : P = 120 :=
by
  -- proof in Lean is omitted, placeholder
  sorry

end aimee_poll_l491_491315


namespace minimum_discount_l491_491662

theorem minimum_discount (C M : ℝ) (profit_margin : ℝ) (x : ℝ) 
  (hC : C = 800) (hM : M = 1200) (hprofit_margin : profit_margin = 0.2) :
  (M * x - C ≥ C * profit_margin) → (x ≥ 0.8) :=
by
  -- Here, we need to solve the inequality given the conditions
  sorry

end minimum_discount_l491_491662


namespace NO_leq_2MO_l491_491638

-- Define the triangle and its centroid
variables (A B C O M N : Point)
variable [is_triangle A B C]
variable [is_centroid O A B C]
variable (line_through_O : Line)
variable (intersects_AB_at_M : line_through_O ∉ {A, B})
variable (intersects_AC_at_N : line_through_O ∉ {A, C})

theorem NO_leq_2MO
: (NO ≤ 2 * MO) (O : Point) (A B C M N : Point)
    [is_triangle A B C]
    [is_centroid O A B C]
    (line_through_O : Line)
    (intersects_AB_at_M : line_through_O ∉ {A, B})
    (intersects_AC_at_N : line_through_O ∉ {A, C}) :
    M ∈ line_through_O ∧ N ∈ line_through_O → NO ≤ 2 * MO :=
begin
  sorry -- proof goes here
end

end NO_leq_2MO_l491_491638


namespace max_min_value_d_l491_491007

-- Definitions of the given conditions
def circle_eqn (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Definition of the distance squared from a point to a fixed point
def dist_sq (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Definition of the function d
def d (P : ℝ × ℝ) : ℝ := dist_sq P A + dist_sq P B

-- The main theorem that we need to prove
theorem max_min_value_d :
  (∀ P : ℝ × ℝ, circle_eqn P.1 P.2 → d P ≤ 74) ∧
  (∃ P : ℝ × ℝ, circle_eqn P.1 P.2 ∧ d P = 74) ∧
  (∀ P : ℝ × ℝ, circle_eqn P.1 P.2 → 34 ≤ d P) ∧
  (∃ P : ℝ × ℝ, circle_eqn P.1 P.2 ∧ d P = 34) :=
sorry

end max_min_value_d_l491_491007
